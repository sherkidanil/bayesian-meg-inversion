import numpy as np
import pytest
import torch

from megflow.dataset import sha256
from megflow.m1_article_decoder import (
    FROZEN_M1_DECODER,
    align_m1_proposals,
    decode_m1_bank,
    candidate_matrices,
    gibbs_weights,
    load_frozen_m1_protocol,
    load_m1_evaluation_arrays,
    m1_localization_metrics,
    refine_centroids,
    sample_fixed_k_bank,
    score_proposals,
    validate_m1_protocol_matches,
    weighted_slot_medoids,
    write_m1_artifacts,
)


class TinyFixedKModel:
    def encoder(self, y):
        return y.mean((1, 2), keepdim=True)

    def field(self, t, state, cond, mask):
        return torch.ones_like(state) * (1 + t[:, None, None])


def test_frozen_protocol_matches_article_operating_point():
    assert FROZEN_M1_DECODER == {
        "bank_size": 800,
        "euler_steps": 96,
        "initial_noise_scale": 1.5,
        "signal_rank": 6,
        "gibbs_tau": 0.0011995147478874274,
        "refinement_passes": 4,
        "radii_mm": [10.0, 8.0, 10.0],
    }


def test_gibbs_weights_are_stable_and_normalized():
    weights = gibbs_weights([0.90, 0.901, 0.899])
    assert weights.argmax() == 1
    assert weights.sum() == 1.0
    assert np.isfinite(weights).all()


def test_component_medoid_prefers_component_mass_then_distance(geometry):
    draws = np.array([[0, 0], [3, 1], [3, 2]])
    weights = np.array([0.45, 0.35, 0.20])
    assert weighted_slot_medoids(draws, weights, geometry).tolist() == [3, 1]


def test_candidate_matrices_gather_slot_specific_dictionaries():
    dictionaries = np.stack(
        [
            np.arange(15).reshape(3, 5),
            100 + np.arange(15).reshape(3, 5),
            200 + np.arange(15).reshape(3, 5),
        ]
    )
    proposals = np.array([[0, 1, 2], [4, 3, 2]])
    actual = candidate_matrices(dictionaries, proposals)
    assert actual.shape == (2, 3, 3)
    np.testing.assert_array_equal(actual[0, :, 0], dictionaries[0, :, 0])
    np.testing.assert_array_equal(actual[0, :, 1], dictionaries[1, :, 1])
    np.testing.assert_array_equal(actual[1, :, 2], dictionaries[2, :, 2])


def test_score_proposals_uses_rank_six_joint_subspace():
    rng = np.random.default_rng(4)
    dictionaries = rng.normal(size=(3, 8, 5))
    proposals = np.array([[0, 1, 2], [2, 3, 4]])
    truth_matrix = candidate_matrices(dictionaries, proposals[:1])[0]
    y = truth_matrix @ rng.normal(size=(3, 20)) + 0.01 * rng.normal(size=(8, 20))
    scores, signal = score_proposals(proposals, dictionaries, y, signal_rank=6)
    assert scores.shape == (2,)
    assert signal.shape == (8, 6)
    assert scores[0] > scores[1]
    assert np.all((0 <= scores) & (scores <= 1))


def test_refinement_uses_slot_specific_dictionary(geometry):
    dictionaries = np.zeros((2, 3, 6))
    dictionaries[0, :, 0] = [1, 0, 0]
    dictionaries[0, :, 1] = [1, 0, 0]
    dictionaries[1, :, 4] = [0, 1, 0]
    dictionaries[1, :, 5] = [0, 0, 1]
    signal = np.eye(3)[:, [0, 2]]

    result = refine_centroids(
        [0, 4], dictionaries, signal, geometry, max_passes=4
    )

    assert result.centroids.tolist() == [0, 5]
    assert result.n_moves == 1
    assert result.n_passes == 2


def test_sample_bank_is_seeded_nested_and_denormalized():
    model = TinyFixedKModel()
    y = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    mean = np.array([10.0, 20.0, 30.0])
    std = np.array([2.0, 3.0, 4.0])

    large = sample_fixed_k_bank(
        model,
        y,
        mean,
        std,
        bank_size=5,
        steps=2,
        initial_noise_scale=1.5,
        seed=17,
    )
    small = sample_fixed_k_bank(
        model,
        y,
        mean,
        std,
        bank_size=3,
        steps=2,
        initial_noise_scale=1.5,
        seed=17,
    )

    assert large.shape == (5, 3, 3)
    np.testing.assert_array_equal(large[:3], small)
    generator = torch.Generator().manual_seed(17)
    initial = 1.5 * torch.stack(
        [torch.randn((3, 3), generator=generator) for _ in range(5)]
    )
    expected_normalized = initial + 0.5 * 1.0 + 0.5 * 1.5
    expected = expected_normalized.numpy() * std.astype("f4") + mean.astype("f4")
    np.testing.assert_allclose(large, expected, rtol=0, atol=1e-6)


def test_decode_m1_bank_scores_aggregates_and_refines_without_truth(geometry):
    dictionaries = np.zeros((3, 4, 6))
    for slot, vertex in enumerate([0, 1, 3]):
        dictionaries[slot, slot, vertex] = 1
    dictionaries[1, 1, 2] = 1
    proposals = np.array(
        [
            geometry.vertices[[0, 1, 3]],
            geometry.vertices[[0, 2, 3]],
            geometry.vertices[[0, 2, 3]],
        ]
    )
    y = candidate_matrices(dictionaries, np.array([[0, 2, 3]]))[0] @ np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, -1, -1]], dtype=float
    )
    y += 1e-3 * np.random.default_rng(3).normal(size=y.shape)

    decoded = decode_m1_bank(
        proposals,
        y,
        dictionaries,
        geometry,
        signal_rank=3,
        refinement_passes=2,
        tau=0.1,
    )

    assert decoded["projected_vertices"].shape == (3, 3)
    assert decoded["scores"].shape == (3,)
    assert decoded["weights"].sum() == pytest.approx(1.0)
    assert decoded["refined_centers"].tolist() == [0, 2, 3]
    assert "true_centers" not in decoded


def test_align_m1_proposals_assigns_eight_mm_slot_then_matches_ten_mm_slots(geometry):
    dictionaries = np.zeros((3, 4, 6))
    dictionaries[0, :, 0] = dictionaries[2, :, 0] = [1, 0, 0, 0]
    dictionaries[0, :, 3] = dictionaries[2, :, 3] = [0, 0, 1, 0]
    dictionaries[1, :, 2] = [0, 1, 0, 0]
    raw = np.array([[0, 2, 3], [3, 0, 2], [2, 3, 0]])
    target = candidate_matrices(dictionaries, np.array([[0, 2, 3]]))[0]
    y = target @ np.eye(3)

    aligned, scores, diagnostics, _ = align_m1_proposals(
        raw, dictionaries, y, geometry, signal_rank=3
    )

    np.testing.assert_array_equal(aligned, np.tile([0, 2, 3], (3, 1)))
    assert scores.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert diagnostics["permutation_fraction"] == pytest.approx(2 / 3)


def test_m1_metrics_preserve_disconnected_error(geometry):
    metrics = m1_localization_metrics(
        true_centers=[0, 3, 2],
        predicted_centers=[0, 1, 2],
        geometry=geometry,
        disconnected_penalty_mm=30.0,
    )
    assert metrics["geodesic_mae_mm"] is None
    assert metrics["geodesic_disconnected_count"] == 1
    assert metrics["geodesic_connected_mae_mm"] == 0.0
    assert metrics["geodesic_mae_at_30_mm"] == 10.0
    assert metrics["euclidean_mae_mm"] == pytest.approx(np.sqrt(104) / 3)


def test_load_m1_evaluation_arrays_checks_shard_and_concatenates(tmp_path):
    directory = tmp_path / "dev"
    directory.mkdir()
    for index in range(2):
        np.savez(
            directory / f"part-{index}.npz",
            Y=np.full((1, 2, 4), index, dtype="f4"),
            Y_pool=np.full((1, 2, 2), index, dtype="f4"),
            centers=np.array([[index, index + 1, index + 2]]),
            radii=np.array([[10.0, 8.0, 10.0]], dtype="f4"),
            record_id=np.array([f"record-{index}"]),
        )
    shards = [
        {"file": f"part-{i}.npz", "sha256": sha256(directory / f"part-{i}.npz"), "n_records": 1}
        for i in range(2)
    ]
    manifest = {"split": "dev", "n_records": 2, "shards": shards}
    (directory / "manifest.json").write_text(__import__("json").dumps(manifest))

    loaded = load_m1_evaluation_arrays(directory)

    assert loaded["Y"].shape == (2, 2, 4)
    assert loaded["Y_pool"].shape == (2, 2, 2)
    assert loaded["record_id"].tolist() == ["record-0", "record-1"]
    (directory / "part-1.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_m1_evaluation_arrays(directory)


def test_artifacts_are_immutable_and_heldout_requires_matching_dev_protocol(tmp_path):
    output = tmp_path / "result"
    metadata = {
        "split": "dev",
        "architecture": "transformer",
        **FROZEN_M1_DECODER,
        "checkpoint_sha256": "checkpoint",
        "anatomy_content_sha256": "anatomy",
        "dev_manifest_sha256": "dev",
        "heldout_manifest_sha256": "heldout",
    }
    records = [{"record_index": 0, "geodesic_mae_mm": None}]
    bank = {
        "xyz_mm": np.zeros((1, 2, 3, 3), dtype="f4"),
        "projected_vertices": np.zeros((1, 2, 3), dtype="i8"),
        "scores": np.ones((1, 2)),
        "weights": np.full((1, 2), 0.5),
        "medoid_centers": np.zeros((1, 3), dtype="i8"),
        "refined_centers": np.zeros((1, 3), dtype="i8"),
    }

    write_m1_artifacts(
        output, summary={"geodesic_mae_mm": None}, records=records,
        bank_arrays=bank, metadata=metadata
    )

    protocol = load_frozen_m1_protocol(output / "protocol.json")
    validate_m1_protocol_matches(protocol, {**metadata, "split": "heldout"})
    changed = {**metadata, "split": "heldout", "bank_size": 799}
    with pytest.raises(ValueError, match="bank_size"):
        validate_m1_protocol_matches(protocol, changed)
    with pytest.raises(FileExistsError):
        write_m1_artifacts(
            output, summary={}, records=[], bank_arrays=bank, metadata=metadata
        )
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"split":"heldout","frozen":true}')
    with pytest.raises(ValueError, match="dev"):
        load_frozen_m1_protocol(invalid)
