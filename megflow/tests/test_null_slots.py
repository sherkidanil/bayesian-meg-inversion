"""Contracts for the exchangeable active/null source canvas."""

import pytest
import torch


def test_rectangular_assignment_is_canvas_equivariant():
    from megflow.null_slots import match_sources_to_slots

    prior = torch.tensor([[[0.0, 0.0], [4.0, 0.0], [9.0, 0.0]]])
    sources = torch.tensor([[[3.9, 0.0], [9.1, 0.0], [0.0, 0.0]]])
    source_mask = torch.tensor([[True, True, False]])

    target, active = match_sources_to_slots(prior, sources, source_mask)

    assert active.tolist() == [[False, True, True]]
    torch.testing.assert_close(target[0, 1], sources[0, 0])
    torch.testing.assert_close(target[0, 2], sources[0, 1])
    torch.testing.assert_close(target[0, 0], prior[0, 0])

    permutation = torch.tensor([2, 0, 1])
    p_target, p_active = match_sources_to_slots(
        prior[:, permutation], sources, source_mask
    )
    torch.testing.assert_close(p_target, target[:, permutation])
    torch.testing.assert_close(p_active, active[:, permutation])


@pytest.mark.parametrize("k", range(1, 6))
def test_rectangular_assignment_preserves_every_source_vector(k):
    from megflow.null_slots import match_sources_to_slots

    prior = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    sources = prior.flip(1).clone()
    source_mask = torch.arange(5)[None] < k
    target, active = match_sources_to_slots(prior, sources, source_mask)

    assert int(active.sum()) == k
    actual = {tuple(row.tolist()) for row in target[0, active[0]]}
    expected = {tuple(row.tolist()) for row in sources[0, :k]}
    assert actual == expected
    torch.testing.assert_close(target[0, ~active[0]], prior[0, ~active[0]])


def test_null_slot_flow_batch_builds_zero_velocity_nulls_and_masked_statuses():
    from megflow.null_slots import ACTIVE, MASK, NULL, null_slot_flow_batch

    states = torch.tensor(
        [
            [[-2.0, 0.0], [3.0, 1.0], [0.0, 0.0]],
            [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    source_mask = torch.tensor([[True, True, False], [True, False, False]])
    batch = null_slot_flow_batch(
        states, source_mask, torch.Generator().manual_seed(17)
    )
    t, state_t, velocity, status_t, status_target, active = batch

    assert t.shape == (2,)
    assert state_t.shape == velocity.shape == (2, 3, 2)
    assert status_t.shape == status_target.shape == active.shape == (2, 3)
    assert active.sum(1).tolist() == [2, 1]
    assert torch.count_nonzero(velocity[~active]) == 0
    endpoint = state_t + (1 - t[:, None, None]) * velocity
    for row in range(2):
        actual = endpoint[row, active[row]]
        expected = states[row, source_mask[row]]
        actual = actual[actual[:, 0].argsort()]
        expected = expected[expected[:, 0].argsort()]
        torch.testing.assert_close(actual, expected)
    assert torch.equal(status_target[active], torch.full_like(status_target[active], ACTIVE))
    assert torch.equal(status_target[~active], torch.full_like(status_target[~active], NULL))
    revealed = status_t != MASK
    torch.testing.assert_close(revealed.sum(1), torch.floor(t * 3).long())
    torch.testing.assert_close(status_t[revealed], status_target[revealed])
    assert torch.equal(status_t[~revealed], torch.full_like(status_t[~revealed], MASK))


def test_null_slot_flow_batch_uses_only_the_supplied_generator():
    from megflow.null_slots import null_slot_flow_batch

    states = torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(4))
    source_mask = torch.tensor(
        [[True, False, False, False], [True, True, False, False], [True] * 4]
    )
    torch.manual_seed(1)
    first = null_slot_flow_batch(
        states, source_mask, torch.Generator().manual_seed(22)
    )
    torch.manual_seed(999)
    second = null_slot_flow_batch(
        states, source_mask, torch.Generator().manual_seed(22)
    )
    for actual, expected in zip(first, second):
        torch.testing.assert_close(actual, expected)


def test_null_slot_helpers_reject_invalid_inputs():
    from megflow.null_slots import match_sources_to_slots, null_slot_flow_batch

    prior = torch.zeros(1, 3, 2)
    sources = torch.zeros(1, 3, 2)
    with pytest.raises(ValueError, match="shape"):
        match_sources_to_slots(prior, sources[:, :2], torch.ones(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="at least one"):
        null_slot_flow_batch(
            sources, torch.zeros(1, 3, dtype=torch.bool), torch.Generator()
        )


def tiny_null_slot_model(k_max=5):
    from megflow.encoder import RawEncoder
    from megflow.null_slot_model import NullSlotConditionalFlow

    return NullSlotConditionalFlow(
        RawEncoder(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        dict(
            k_max=k_max,
            d_in=4,
            d_cond=12,
            d_model=32,
            n_layers=2,
            n_heads=4,
        ),
    )


def test_null_slot_field_is_permutation_equivariant_without_count_head():
    model = tiny_null_slot_model()
    state = torch.randn(3, 5, 4)
    status = torch.randint(0, 3, (3, 5))
    y, t = torch.randn(3, 4, 40), torch.rand(3)
    permutation = torch.tensor([3, 0, 4, 1, 2])

    velocity, logits = model(t, state, status, y)
    p_velocity, p_logits = model(
        t, state[:, permutation], status[:, permutation], y
    )

    torch.testing.assert_close(
        p_velocity, velocity[:, permutation], atol=1e-5, rtol=0
    )
    torch.testing.assert_close(p_logits, logits[:, permutation], atol=1e-5, rtol=0)
    assert velocity.shape == state.shape
    assert logits.shape == (3, 5, 2)
    assert not hasattr(model, "count_head")


def test_null_slot_field_has_finite_gradients_through_both_heads():
    model = tiny_null_slot_model(k_max=3)
    state = torch.randn(2, 3, 4)
    status = torch.tensor([[2, 2, 0], [2, 1, 2]])
    velocity, logits = model(
        torch.rand(2), state, status, torch.randn(2, 4, 40)
    )
    (velocity.square().mean() + logits.square().mean()).backward()

    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.field.velocity_head.weight.grad is not None
    assert model.field.status_head.weight.grad is not None


@pytest.mark.parametrize(
    ("state_shape", "status", "message"),
    [
        ((2, 2, 4), torch.zeros(2, 2, dtype=torch.long), "exactly 3"),
        ((2, 3, 3), torch.zeros(2, 3, dtype=torch.long), "state dimension"),
        ((2, 3, 4), torch.full((2, 3), 3, dtype=torch.long), "status"),
    ],
)
def test_null_slot_field_rejects_invalid_canvas(state_shape, status, message):
    model = tiny_null_slot_model(k_max=3)
    with pytest.raises(ValueError, match=message):
        model(
            torch.rand(2),
            torch.randn(*state_shape),
            status,
            torch.randn(2, 4, 40),
        )


class ScriptedNullSlotModel:
    k_max = 3
    d_in = 2

    def __init__(self, logits):
        self.logits = torch.as_tensor(logits, dtype=torch.float32)
        self.status_history = []

    def encode(self, y):
        return torch.zeros(len(y), 1, 2)

    def velocity_and_status(self, t, state, status, cond):
        self.status_history.append(status.clone())
        logits = self.logits.to(state).expand(len(state), -1, -1).clone()
        return torch.zeros_like(state), logits


@pytest.mark.parametrize("steps", [2, 3, 7])
def test_progressive_sampler_commits_every_slot_on_schedule(steps):
    from megflow.null_slot_model import required_committed, sample_null_slots
    from megflow.null_slots import MASK

    model = ScriptedNullSlotModel(
        [[[0.0, 1.0], [4.0, 0.0], [0.0, 2.0]]]
    )
    sample = sample_null_slots(
        model,
        torch.zeros(2, 1, 1),
        steps=steps,
        generator=torch.Generator().manual_seed(9),
    )

    assert torch.all(model.status_history[0] == MASK)
    committed_before = [int((row[0] != MASK).sum()) for row in model.status_history]
    assert committed_before == [0] + [required_committed(i, steps, 3) for i in range(steps - 1)]
    assert sample.active_mask.shape == (2, 3)
    assert sample.active_mask.tolist() == [[False, True, False]] * 2
    assert sample.state.shape == (2, 3, 2)
    assert sample.active_probability.shape == (2, 3)
    assert not sample.forced_nonempty.any()


def test_progressive_sampler_commits_highest_confidence_first_and_never_changes_it():
    from megflow.null_slot_model import sample_null_slots
    from megflow.null_slots import ACTIVE, MASK

    model = ScriptedNullSlotModel(
        [[[0.0, 1.0], [5.0, 0.0], [0.0, 2.0]]]
    )
    sample_null_slots(
        model,
        torch.zeros(1, 1, 1),
        steps=3,
        generator=torch.Generator().manual_seed(2),
    )

    assert model.status_history[1].tolist() == [[MASK, ACTIVE, MASK]]
    assert model.status_history[2][0, 1] == ACTIVE


def test_progressive_sampler_is_seeded_and_documents_nonempty_correction():
    from megflow.null_slot_model import sample_null_slots

    model = ScriptedNullSlotModel(
        [[[-1.0, 4.0], [-2.0, 4.0], [-3.0, 4.0]]]
    )
    first = sample_null_slots(
        model,
        torch.zeros(2, 1, 1),
        steps=4,
        generator=torch.Generator().manual_seed(8),
    )
    second = sample_null_slots(
        ScriptedNullSlotModel(model.logits),
        torch.zeros(2, 1, 1),
        steps=4,
        generator=torch.Generator().manual_seed(8),
    )

    torch.testing.assert_close(first.state, second.state, rtol=0, atol=0)
    torch.testing.assert_close(
        first.active_probability, second.active_probability, rtol=0, atol=0
    )
    assert first.forced_nonempty.tolist() == [True, True]
    assert first.active_mask.tolist() == [[True, False, False]] * 2


def test_progressive_sampler_rejects_nonpositive_steps():
    from megflow.null_slot_model import sample_null_slots

    with pytest.raises(ValueError, match="positive"):
        sample_null_slots(
            ScriptedNullSlotModel(torch.zeros(1, 3, 2)),
            torch.zeros(1, 1, 1),
            steps=0,
            generator=torch.Generator(),
        )


def test_null_slot_loss_separates_active_null_and_masked_status_terms():
    from torch.nn import functional as F
    from megflow.null_slots import ACTIVE, MASK, NULL
    from megflow.train import null_slot_flow_errors

    prediction = torch.tensor([[[2.0], [3.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    logits = torch.tensor(
        [[[2.0, -1.0], [-3.0, 4.0]]], requires_grad=True
    )
    status_target = torch.tensor([[ACTIVE, NULL]])
    status_input = torch.tensor([[MASK, NULL]])
    active_mask = torch.tensor([[True, False]])

    active, null, status, correct, count = null_slot_flow_errors(
        prediction,
        target,
        logits,
        status_target,
        status_input,
        active_mask,
    )
    expected_status = F.cross_entropy(logits[:, :1].reshape(-1, 2), torch.tensor([ACTIVE]))
    torch.testing.assert_close(active, torch.tensor([4.0]))
    torch.testing.assert_close(null, torch.tensor([9.0]))
    torch.testing.assert_close(status, expected_status[None])
    assert correct.tolist() == [1]
    assert count.tolist() == [1]

    total = (active + 0.1 * null + 0.1 * status).mean()
    total.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[4.0], [0.6]]]))
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


def test_validation_cardinality_logits_receive_no_revealed_statuses():
    from megflow.null_slots import MASK
    from megflow.train import null_slot_inference_logits

    class RecordingModel:
        def __init__(self):
            self.status = None

        def __call__(self, t, state, status, y):
            self.status = status.clone()
            return torch.zeros_like(state), torch.zeros(*status.shape, 2)

    model = RecordingModel()
    logits = null_slot_inference_logits(
        model,
        torch.rand(2),
        torch.randn(2, 3, 4),
        torch.randn(2, 4, 10),
    )
    assert logits.shape == (2, 3, 2)
    assert torch.all(model.status == MASK)


def test_build_model_dispatches_null_slots_only_to_transformer():
    from omegaconf import OmegaConf
    from pathlib import Path
    from megflow.null_slot_model import NullSlotConditionalFlow
    from megflow.train import build_model

    config = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parents[1] / "config/v15_null_slots.yaml"),
        resolve=True,
    )
    model = build_model(config, "transformer")
    assert isinstance(model, NullSlotConditionalFlow)
    assert model.k_max == 10
    assert not hasattr(model, "count_head")
    with pytest.raises(ValueError, match="Transformer"):
        build_model(config, "mlp")


def test_null_slot_training_end_to_end(tmp_path, geometry):
    import json
    import numpy as np
    from megflow import v15
    from megflow.dataset import SPLITS
    from megflow.train import build_model, run_training
    from megflow.v15_dataset import audit_splits, write_split

    simulation = v15.default_simulation_config()
    simulation.update(k_max=3, min_separation_mm=2.0)
    config = dict(
        milestone="V15",
        dataset=dict(
            pool=10,
            shard_size=2,
            counts={split: 3 for split in SPLITS},
            simulation=simulation,
        ),
        encoder=dict(
            n_sensors=4,
            widths=[8, 16, 16],
            kernel_size=3,
            n_queries=4,
            cond_dim=12,
        ),
        model=dict(
            cardinality_mode="null_slots",
            target="xyz_radius",
            radius_transform="log",
            mlp_width=32,
            mlp_depth=2,
            d_model=32,
            n_layers=2,
            n_heads=4,
        ),
        training=dict(
            seed=404,
            coupling="rectangular_hungarian",
            null_flow_weight=0.1,
            existence_loss_weight=0.1,
            sampler_steps=10,
            normalize="shared_source_train_only",
            steps=3,
            batch_size=3,
            lr=0.001,
            warmup=1,
            weight_decay=0.01,
            clip_norm=1.0,
            ema_decay=0.9,
            bf16=False,
            eval_every=3,
            log_every=3,
            max_seconds=60,
        ),
    )
    leadfield = np.random.default_rng(4).normal(size=(4, 6)) * 1e-7
    for split_index, split in enumerate(SPLITS):
        records = [
            v15.simulate(
                leadfield,
                geometry,
                100 * split_index + record_index,
                simulation,
                apply_filter=False,
            )
            for record_index in range(3)
        ]
        for record in records:
            record["filtered"] = True
        write_split(
            iter(records),
            tmp_path / "data" / split,
            split,
            config["dataset"],
            "anatomy",
            "code",
        )
    audit_splits(tmp_path / "data")

    output, result = run_training(
        config, "transformer", tmp_path / "data", tmp_path / "runs", device="cpu"
    )

    assert result["status"] == "complete" and result["step"] == 3
    assert 0 <= result["status_accuracy"] <= 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["cardinality_mode"] == "null_slots"
    assert manifest["count_head"] is False
    assert manifest["predicts_k"] is True
    saved = torch.load(output / "last.pt", weights_only=False)
    reloaded = build_model(config, "transformer")
    reloaded.load_state_dict(saved["ema"])
    assert all(torch.isfinite(parameter).all() for parameter in reloaded.parameters())
