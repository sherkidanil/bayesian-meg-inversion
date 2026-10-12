"""Unknown-cardinality fields for the variable-source experiment."""

import torch
from torch import nn
from .model import SetVelocityField


class UnknownKConditionalFlow(nn.Module):
    """Infer K from the recording, then run the field with that many tokens."""

    def __init__(self, encoder, config, architecture):
        super().__init__()
        self.encoder = encoder
        self.k_max = config["k_max"]
        self.d_in = config.get("d_in", 3)
        self.architecture = architecture
        d_cond = config["d_cond"]
        self.count_head = nn.Linear(d_cond, self.k_max)
        if architecture == "transformer":
            self.field = SetVelocityField(
                d_in=self.d_in,
                d_cond=d_cond,
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
            )
        elif architecture == "mlp":
            layers = [
                nn.Linear(self.k_max * (self.d_in + 1) + d_cond + 1, config["width"]),
                nn.SiLU(),
            ]
            for _ in range(config["depth"] - 1):
                layers.extend([nn.Linear(config["width"], config["width"]), nn.SiLU()])
            layers.append(nn.Linear(config["width"], self.k_max * self.d_in))
            self.field = nn.Sequential(*layers)
        else:
            raise ValueError("unknown variable-K architecture")

    def encode(self, y):
        cond = self.encoder(y)
        return cond, self.count_head(cond.mean(1))

    def velocity(self, t, theta, cond):
        if (
            theta.ndim != 3
            or theta.shape[2] != self.d_in
            or not 1 <= theta.shape[1] <= self.k_max
        ):
            raise ValueError("theta must contain 1..K_max source-state tokens")
        if not torch.isfinite(theta).all():
            raise ValueError("theta must contain finite source states")
        if self.architecture == "transformer":
            all_slots = torch.ones(
                theta.shape[:2], dtype=torch.bool, device=theta.device
            )
            return self.field(t, theta, cond, all_slots)
        k = theta.shape[1]
        padded = torch.zeros(
            (len(theta), self.k_max, self.d_in), dtype=theta.dtype, device=theta.device
        )
        padded[:, :k] = theta
        slot_mask = torch.zeros(
            (len(theta), self.k_max), dtype=theta.dtype, device=theta.device
        )
        slot_mask[:, :k] = 1
        inputs = torch.cat(
            [padded.flatten(1), slot_mask, cond.mean(1), t[:, None]], dim=-1
        )
        return self.field(inputs).reshape(-1, self.k_max, self.d_in)[:, :k]

    def forward(self, t, theta, y):
        cond, logits = self.encode(y)
        return self.velocity(t, theta, cond), logits


@torch.no_grad()
def sample_unknown_k(model, y, *, steps=100, generator=None):
    """Predict cardinality, then integrate exactly that many source tokens."""
    if steps < 1:
        raise ValueError("sampling steps must be positive")
    cond, logits = model.encode(y)
    counts = logits.argmax(1) + 1
    mask = torch.arange(model.k_max, device=y.device)[None] < counts[:, None]
    output = torch.full(
        (len(y), model.k_max, model.d_in),
        float("nan"),
        dtype=y.dtype,
        device=y.device,
    )
    dt = 1.0 / steps
    for k in counts.unique(sorted=True):
        selected = counts == k
        theta = torch.randn(
            (int(selected.sum()), int(k), model.d_in),
            dtype=y.dtype,
            device=y.device,
            generator=generator,
        )
        group_cond = cond[selected]
        for step in range(steps):
            t = torch.full(
                (len(theta),), (step + 0.5) * dt, dtype=y.dtype, device=y.device
            )
            theta = theta + dt * model.velocity(t, theta, group_cond)
        output[selected, :k] = theta
    return output, mask, logits
