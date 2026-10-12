"""Joint continuous and active/null field on an exchangeable source canvas."""

from dataclasses import dataclass

import torch
from torch import nn

from .model import AdaLN, SetBlock, TimeEmbedding
from .null_slots import ACTIVE, MASK


class NullSlotSetField(nn.Module):
    def __init__(
        self, *, d_state, d_model, n_layers, n_heads, d_cond, k_max
    ):
        super().__init__()
        if d_model % n_heads or d_model % 2:
            raise ValueError("model width must be even and divisible by number of heads")
        self.d_state = d_state
        self.k_max = k_max
        self.state_input = nn.Linear(d_state, d_model)
        self.status_input = nn.Embedding(3, d_model)
        self.time = TimeEmbedding(d_model)
        self.blocks = nn.ModuleList(
            SetBlock(d_model, n_heads, d_cond) for _ in range(n_layers)
        )
        self.norm = AdaLN(d_model)
        self.velocity_head = nn.Linear(d_model, d_state)
        self.status_head = nn.Linear(d_model, 2)

    def _validate(self, t, state, status):
        if state.ndim != 3 or state.shape[1] != self.k_max:
            raise ValueError(f"state must contain exactly {self.k_max} slots")
        if state.shape[2] != self.d_state:
            raise ValueError(f"state dimension must be {self.d_state}")
        if status.shape != state.shape[:2]:
            raise ValueError("status shape must match the source canvas")
        if status.dtype != torch.long or torch.any((status < 0) | (status > 2)):
            raise ValueError("status values must be integer ACTIVE/NULL/MASK IDs")
        if t.shape != (len(state),):
            raise ValueError("flow time must contain one scalar per record")
        if not torch.isfinite(state).all() or not torch.isfinite(t).all():
            raise ValueError("state and flow time must be finite")

    def forward(self, t, state, status, cond):
        self._validate(t, state, status)
        h = self.state_input(state) + self.status_input(status)
        time = self.time(t)
        all_slots = torch.ones(
            status.shape, dtype=torch.bool, device=status.device
        )
        for block in self.blocks:
            h = block(h, time, cond, all_slots)
        h = self.norm(h, time)
        return self.velocity_head(h), self.status_head(h)


class NullSlotConditionalFlow(nn.Module):
    """Encode MEG and denoise a fixed source canvas without a global K head."""

    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder
        self.k_max = config["k_max"]
        self.d_in = config.get("d_in", 4)
        self.field = NullSlotSetField(
            d_state=self.d_in,
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_cond=config["d_cond"],
            k_max=self.k_max,
        )

    def encode(self, y):
        return self.encoder(y)

    def velocity_and_status(self, t, state, status, cond):
        return self.field(t, state, status, cond)

    def forward(self, t, state, status, y):
        return self.velocity_and_status(t, state, status, self.encode(y))


@dataclass(frozen=True)
class NullSlotSample:
    state: torch.Tensor
    active_mask: torch.Tensor
    active_probability: torch.Tensor
    forced_nonempty: torch.Tensor


def required_committed(step, steps, k_max):
    """Cumulative number of statuses resolved after a zero-based step."""
    return ((step + 1) * k_max) // steps


@torch.no_grad()
def sample_null_slots(model, y, *, steps=100, generator=None):
    """Integrate all slots while progressively committing active/null statuses."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    if generator is None:
        raise ValueError("a torch.Generator is required")
    cond = model.encode(y)
    state = torch.randn(
        (len(y), model.k_max, model.d_in),
        dtype=y.dtype,
        device=y.device,
        generator=generator,
    )
    status = torch.full(
        (len(y), model.k_max), MASK, dtype=torch.long, device=y.device
    )
    active_probability = torch.zeros_like(status, dtype=state.dtype)
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full(
            (len(y),), (step + 0.5) * dt, dtype=y.dtype, device=y.device
        )
        velocity, logits = model.velocity_and_status(t, state, status, cond)
        state.add_(velocity, alpha=dt)
        probabilities = logits.float().softmax(-1)
        active_probability = probabilities[..., ACTIVE].to(state.dtype)
        confidence, predicted = probabilities.max(-1)
        target_count = required_committed(step, steps, model.k_max)
        for batch_index in range(len(y)):
            committed = int((status[batch_index] != MASK).sum())
            needed = target_count - committed
            if needed <= 0:
                continue
            candidates = torch.where(status[batch_index] == MASK)[0]
            order = confidence[batch_index, candidates].argsort(descending=True)
            selected = candidates[order[:needed]]
            status[batch_index, selected] = predicted[batch_index, selected]
    if torch.any(status == MASK):
        raise RuntimeError("progressive status schedule did not resolve every slot")
    active_mask = status == ACTIVE
    forced_nonempty = ~active_mask.any(1)
    for batch_index in torch.where(forced_nonempty)[0]:
        selected = active_probability[batch_index].argmax()
        active_mask[batch_index, selected] = True
    return NullSlotSample(
        state=state,
        active_mask=active_mask,
        active_probability=active_probability,
        forced_nonempty=forced_nonempty,
    )
