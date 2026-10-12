"""Exchangeable fixed-canvas utilities for variable-cardinality source sets."""

from scipy.optimize import linear_sum_assignment
import torch


ACTIVE = 0
NULL = 1
MASK = 2


def _validate_canvas(prior, sources, source_mask):
    if (
        prior.ndim != 3
        or sources.shape != prior.shape
        or source_mask.shape != prior.shape[:2]
    ):
        raise ValueError("prior, sources and source_mask have incompatible shape")
    if prior.device.type != "cpu" or sources.device.type != "cpu":
        raise ValueError("rectangular assignment requires CPU tensors")
    if source_mask.dtype != torch.bool:
        raise ValueError("source_mask must be boolean")
    if not torch.isfinite(prior).all() or not torch.isfinite(sources[source_mask]).all():
        raise ValueError("active source states and prior slots must be finite")
    if torch.any(source_mask.sum(1) < 1):
        raise ValueError("every record must contain at least one source")


def match_sources_to_slots(prior, sources, source_mask):
    """Match each unordered target set to a subset of Gaussian canvas slots."""
    _validate_canvas(prior, sources, source_mask)
    target = prior.clone()
    active = torch.zeros(prior.shape[:2], dtype=torch.bool)
    for batch_index in range(len(prior)):
        truth = sources[batch_index, source_mask[batch_index]]
        cost = torch.cdist(truth.float(), prior[batch_index].float()).square().numpy()
        truth_index, slot_index = linear_sum_assignment(cost)
        truth_index = torch.as_tensor(truth_index, dtype=torch.long)
        slot_index = torch.as_tensor(slot_index, dtype=torch.long)
        target[batch_index, slot_index] = truth[truth_index]
        active[batch_index, slot_index] = True
    return target, active


def null_slot_flow_batch(states, source_mask, generator):
    """Create a rectified-flow probe and partially revealed slot statuses."""
    if generator is None:
        raise ValueError("a torch.Generator is required")
    if states.device.type != "cpu" or source_mask.device.type != "cpu":
        raise ValueError("null-slot probes must be created on CPU")
    batch, k_max, _ = states.shape
    prior = torch.randn(states.shape, dtype=states.dtype, generator=generator)
    endpoint, active = match_sources_to_slots(prior, states, source_mask)
    t = torch.rand(batch, dtype=states.dtype, generator=generator)
    velocity = endpoint - prior
    state_t = (1 - t[:, None, None]) * prior + t[:, None, None] * endpoint
    status_target = torch.where(
        active,
        torch.full(active.shape, ACTIVE, dtype=torch.long),
        torch.full(active.shape, NULL, dtype=torch.long),
    )
    reveal_count = torch.floor(t * k_max).long()
    reveal_order = torch.rand((batch, k_max), generator=generator).argsort(1)
    reveal_rank = reveal_order.argsort(1)
    revealed = reveal_rank < reveal_count[:, None]
    status_t = torch.full((batch, k_max), MASK, dtype=torch.long)
    status_t[revealed] = status_target[revealed]
    return t, state_t, velocity, status_t, status_target, active
