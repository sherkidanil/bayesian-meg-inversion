import json

import numpy as np
import pytest


def test_select_bank_cardinality_uses_mean_posterior_and_observed_fallback():
    from megflow.v15_candidate_bank import select_bank_cardinality

    probabilities = np.array(
        [[0.0, 0.1, 0.7, 0.2], [0.0, 0.1, 0.6, 0.3]]
    )
    assert select_bank_cardinality(probabilities, [2, 2]) == (2, False)
    assert select_bank_cardinality(probabilities, [1, 3]) == (3, True)


def test_select_bank_cardinality_rejects_invalid_probability_mass():
    from megflow.v15_candidate_bank import select_bank_cardinality

    with pytest.raises(ValueError, match="normalized"):
        select_bank_cardinality([[0.0, 0.2, 0.2]], [1])
    with pytest.raises(ValueError, match="class zero"):
        select_bank_cardinality([[0.1, 0.9, 0.0]], [1])


def test_geodesic_alignment_restores_swapped_sources_and_radii(geometry):
    from megflow.v15_candidate_bank import align_candidate_bank

    centers = np.array([[0, 5], [5, 0]])
    radii = np.array([[10.0, 20.0], [21.0, 11.0]])
    aligned_centers, aligned_radii, permutation_rate = align_candidate_bank(
        centers, radii, scores=[0.9, 0.8], geometry=geometry
    )

    np.testing.assert_array_equal(aligned_centers, [[0, 5], [0, 5]])
    np.testing.assert_allclose(aligned_radii, [[10.0, 20.0], [11.0, 21.0]])
    assert permutation_rate == 0.5


def test_weighted_geodesic_medoid_and_radius_aggregation(geometry):
    from megflow.v15_candidate_bank import (
        aggregate_aligned_bank,
        weighted_geodesic_medoid,
    )

    assert weighted_geodesic_medoid(
        [0, 1, 2], [0.1, 0.2, 0.7], geometry, cutoff_mm=30.0
    ) == 2
    centers, radii = aggregate_aligned_bank(
        np.array([[0, 5], [1, 4]]),
        np.array([[10.0, 20.0], [11.0, 21.0]]),
        np.array([0.8, 0.2]),
        geometry,
        cutoff_mm=30.0,
    )
    np.testing.assert_array_equal(centers, [0, 5])
    np.testing.assert_allclose(radii, [10.2, 20.2])


def test_gibbs_weights_are_stable_and_normalized():
    from megflow.v15_candidate_bank import gibbs_weights

    weights = gibbs_weights([0.90, 0.901, 0.899], tau=1.2e-3)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0)


def test_candidate_scores_use_predicted_radii_and_fixed_cardinality(geometry):
    from megflow.v15_candidate_bank import score_candidate_bank

    leadfield = np.array(
        [
            [1.0, 0.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    y = leadfield[:, [0]] @ np.array([[1.0, -1.0, 2.0, -2.0]])
    scores = score_candidate_bank(
        centers=np.array([[0], [1]]),
        radii=np.array([[1.0], [1.0]]),
        y=y,
        leadfield=leadfield,
        geometry=geometry,
    )
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]


def test_decode_bank_prefix_selects_k_scores_and_aggregates(geometry):
    from megflow.v15_candidate_bank import decode_bank_prefix
    from megflow.v15_evaluate import evaluate_record

    leadfield = np.eye(6)[:4]
    y = leadfield[:, [0]] @ np.array([[1.0, -1.0, 2.0, -2.0]])
    xyz = np.full((3, 2, 3), np.nan)
    radii = np.full((3, 2), np.nan)
    mask = np.array([[True, False], [True, True], [True, False]])
    xyz[0, 0], xyz[1], xyz[2, 0] = (
        geometry.vertices[0],
        geometry.vertices[[1, 4]],
        geometry.vertices[0],
    )
    radii[0, 0], radii[1], radii[2, 0] = 1.0, [1.0, 1.0], 1.2
    probabilities = np.array(
        [[0.0, 0.9, 0.1], [0.0, 0.8, 0.2], [0.0, 0.9, 0.1]]
    )

    decoded = decode_bank_prefix(
        xyz_mm=xyz,
        radii_mm=radii,
        mask=mask,
        cardinality_probability=probabilities,
        y=y,
        leadfield=leadfield,
        geometry=geometry,
        tau=1.2e-3,
        cutoff_mm=30.0,
    )

    assert decoded["selected_k"] == 1
    assert decoded["eligible_candidate_count"] == 2
    assert decoded["cardinality_fallback"] is False
    assert decoded["effective_sample_size"] >= 1.0
    assert decoded["score_max"] >= decoded["score_min"]
    np.testing.assert_array_equal(decoded["centers"], [0])
    aggregate = evaluate_record(
        true_centers=[0],
        true_radii=[1.0],
        predicted_xyz=geometry.vertices[decoded["centers"]],
        predicted_radii=decoded["radii"],
        geometry=geometry,
    )
    assert aggregate["predicted_k"] == 1
    assert aggregate["geodesic_penalized_sum_mm"] == 0.0


@pytest.mark.parametrize("mode", ["count_head", "null_slots"])
def test_sample_candidate_bank_is_seeded_for_one_shared_prefix_bank(mode):
    import torch

    from megflow.v15_candidate_bank import sample_candidate_bank

    class CountModel:
        k_max = 3
        d_in = 4

        def encode(self, y):
            logits = torch.tensor([[5.0, 0.0, 0.0]], device=y.device).repeat(
                len(y), 1
            )
            return torch.zeros(len(y), 1, 2, device=y.device), logits

        def velocity(self, t, state, cond):
            return torch.zeros_like(state)

    class NullModel:
        k_max = 3
        d_in = 4

        def encode(self, y):
            return torch.zeros(len(y), 1, 2, device=y.device)

        def velocity_and_status(self, t, state, status, cond):
            logits = torch.tensor(
                [[[0.0, 4.0], [4.0, 0.0], [4.0, 0.0]]], device=y.device
            ).repeat(len(state), 1, 1)
            return torch.zeros_like(state), logits

    model = CountModel() if mode == "count_head" else NullModel()
    checkpoint = dict(state_mean=torch.zeros(4), state_std=torch.ones(4))
    y = torch.zeros(1, 4, 10)
    five = sample_candidate_bank(
        model,
        checkpoint,
        y,
        mode=mode,
        steps=2,
        bank_size=5,
        record_seed=19,
        radius_bounds=(8.0, 25.0),
    )
    repeated = sample_candidate_bank(
        model,
        checkpoint,
        y,
        mode=mode,
        steps=2,
        bank_size=5,
        record_seed=19,
        radius_bounds=(8.0, 25.0),
    )
    assert five.xyz_mm.shape == (5, 3, 3)
    assert five.radii_mm.shape == (5, 3)
    assert five.mask.shape == (5, 3)
    np.testing.assert_array_equal(five.xyz_mm, repeated.xyz_mm)
    np.testing.assert_array_equal(five.mask, repeated.mask)
    assert five.xyz_mm[:3].shape == (3, 3, 3)
    assert five.cardinality_probability[:3].shape == (3, 4)


def test_record_tensor_copies_read_only_memory():
    import warnings

    from megflow.v15_candidate_bank import record_tensor

    values = np.zeros((1, 4, 10), dtype="f4")
    values.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tensor = record_tensor(values, "cpu")
    assert tuple(tensor.shape) == values.shape
    tensor[0, 0, 0] = 1
    assert values[0, 0, 0] == 0


def test_prefixes_must_be_increasing_unique_and_within_bank():
    from megflow.v15_candidate_bank import validate_prefixes

    assert validate_prefixes([20, 50, 100], 100) == (20, 50, 100)
    for invalid in ([20, 20], [50, 20], [20, 101], [0, 20]):
        with pytest.raises(ValueError, match="prefix"):
            validate_prefixes(invalid, 100)


def test_raw_recording_reader_crosses_shard_boundaries(tmp_path):
    from megflow.v15_candidate_bank import RawRecordingReader

    first = np.arange(2 * 3 * 4, dtype="f4").reshape(2, 3, 4)
    second = np.arange(2 * 3 * 4, 4 * 3 * 4, dtype="f4").reshape(2, 3, 4)
    np.savez(tmp_path / "a.npz", Y=first)
    np.savez(tmp_path / "b.npz", Y=second)
    manifest = {
        "n_records": 4,
        "shards": [
            {"file": "a.npz", "start": 0, "stop": 2},
            {"file": "b.npz", "start": 2, "stop": 4},
        ],
    }
    with RawRecordingReader(tmp_path, manifest) as reader:
        np.testing.assert_array_equal(reader[1], first[1])
        np.testing.assert_array_equal(reader[2], second[0])
        with pytest.raises(IndexError):
            reader[4]


def test_candidate_bank_artifacts_round_trip_and_refuse_overwrite(tmp_path):
    from megflow.v15_candidate_bank import write_candidate_bank_artifacts

    output = tmp_path / "bank"
    metadata = {
        "split": "dev",
        "bank_size": 2,
        "prefixes": [1, 2],
        "k_max": 2,
    }
    summaries = {"1": {"geodesic_penalized_mae_mm": 4.0}}
    records = {"1": [{"record_index": 0, "predicted_k": 1}]}
    arrays = {
        "xyz_mm": np.zeros((1, 2, 2, 3), dtype="f4"),
        "radii_mm": np.ones((1, 2, 2), dtype="f4"),
        "mask": np.ones((1, 2, 2), dtype=bool),
        "cardinality_probability": np.full((1, 2, 3), 1 / 3, dtype="f4"),
        "forced_nonempty": np.zeros((1, 2), dtype=bool),
        "record_seed": np.array([7], dtype="uint64"),
    }
    write_candidate_bank_artifacts(
        output,
        summaries=summaries,
        records=records,
        bank_arrays=arrays,
        metadata=metadata,
    )
    saved = json.loads((output / "summary.json").read_text())
    assert saved["metadata"] == metadata
    assert saved["metrics_by_prefix"] == summaries
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["frozen"] is True
    with np.load(output / "bank.npz", allow_pickle=False) as bank:
        assert bank["xyz_mm"].shape == (1, 2, 2, 3)
    with pytest.raises(FileExistsError):
        write_candidate_bank_artifacts(
            output,
            summaries=summaries,
            records=records,
            bank_arrays=arrays,
            metadata=metadata,
        )


def test_heldout_bank_requires_frozen_dev_protocol(tmp_path):
    from megflow.v15_candidate_bank import load_candidate_bank_protocol, main

    assert callable(main)

    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"split": "dev", "frozen": True, "bank_size": 100}))
    assert load_candidate_bank_protocol(protocol)["bank_size"] == 100
    protocol.write_text(json.dumps({"split": "heldout", "frozen": True, "bank_size": 100}))
    with pytest.raises(ValueError, match="dev"):
        load_candidate_bank_protocol(protocol)
