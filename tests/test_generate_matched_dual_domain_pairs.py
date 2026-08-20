from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent if HERE.name == "tests" else HERE
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from generate_matched_dual_domain_pairs import (
    anna_good_snr_mask,
    deterministic_split_indices,
    generate_noise_latents,
    map_surface_normals_to_vertices,
    project_noise_latents,
    sensor_pair_from_latents,
    shift_centroids_anna,
    split_payload,
    validate_dual_domain_payload,
    validate_written_bundle,
    write_locked_splits,
)


def test_shift_centroids_replays_ten_neighbor_min_distance_correction():
    vertices = np.array(
        [
            [0.000, 0.0, 0.0],
            [0.032, 0.0, 0.0],
            [0.080, 0.0, 0.0],
            [0.003, 0.0, 0.0],
            [0.031, 0.0, 0.0],
            [0.083, 0.0, 0.0],
            [0.040, 0.0, 0.0],
        ]
    )

    result = shift_centroids_anna(
        np.array([0, 1, 2]),
        vertices,
        delta_m=np.array([0.003, 0.0, 0.0]),
        min_dist_mm=30.0,
        k_neighbors=4,
    )

    assert result["correction_triggered"] is True
    assert result["valid"] is True
    assert np.array_equal(result["simple_centroids"], [3, 1, 5])
    assert np.array_equal(result["shifted_centroids"], [3, 6, 5])
    assert np.min(result["pair_distances_m"]) >= 0.030


def test_sensor_pair_reuses_domain_noise_for_original_and_shifted():
    leadfield = np.array([[1.0, 2.0], [3.0, 5.0]])
    temporal = np.array([[1.0, 0.0], [0.0, 1.0]])
    spatial_original = np.eye(2)
    spatial_shifted = np.array([[0.5, 0.5], [0.0, 1.0]])
    sensor_noise = np.array([[0.2, -0.2], [0.1, -0.1]])

    result = sensor_pair_from_latents(
        leadfield,
        spatial_original,
        spatial_shifted,
        temporal,
        sensor_noise,
        noise_level=0.3,
    )

    clean_original = result["clean_original"]
    clean_shifted = result["clean_shifted"]
    assert np.allclose(result["Y_original"] - clean_original, 0.3 * sensor_noise)
    assert np.allclose(result["Y_shifted"] - clean_shifted, 0.3 * sensor_noise)


def test_anna_good_snr_mask_requires_all_four_views_to_pass():
    params = {
        "t_start_sources": np.array([0, 2, 4]),
        "t_stop_sources": np.array([2, 4, 6]),
    }
    good = np.zeros((2, 1, 8), dtype=float)
    good[:, :, :6] = 4.0
    good[:, :, 6:] = 1.0
    bad = good.copy()
    bad[1, :, 2:4] = 1.0

    mask, per_view = anna_good_snr_mask(
        {
            "rebuilt_original": good,
            "rebuilt_shifted": good,
            "anna_original": good,
            "anna_shifted": bad,
        },
        params,
        threshold=3.5,
    )

    assert mask.tolist() == [True, False]
    assert per_view["anna_shifted"].tolist() == [True, False]


def test_deterministic_split_is_disjoint_and_exactly_half():
    dev, heldout = deterministic_split_indices(700, dev_size=350, seed=20260811)

    assert len(dev) == len(heldout) == 350
    assert len(set(dev) & set(heldout)) == 0
    assert sorted(np.concatenate([dev, heldout]).tolist()) == list(range(700))
    assert np.array_equal(dev, deterministic_split_indices(700, 350, 20260811)[0])


def test_split_payload_slices_sample_arrays_and_preserves_leadfields():
    payload = {
        "Ys_rebuilt": np.arange(12).reshape(4, 3),
        "centroids_idx": np.arange(8).reshape(4, 2),
        "G_rebuilt": np.eye(3),
        "G_anna": np.eye(3) * 2,
        "params_spatial": np.array({"N_src": 2}, dtype=object),
    }

    result = split_payload(payload, np.array([3, 1]), n_samples=4)

    assert np.array_equal(result["Ys_rebuilt"], payload["Ys_rebuilt"][[3, 1]])
    assert np.array_equal(result["centroids_idx"], payload["centroids_idx"][[3, 1]])
    assert np.array_equal(result["G_rebuilt"], payload["G_rebuilt"])
    assert np.array_equal(result["G_anna"], payload["G_anna"])


def test_noise_latents_are_reused_but_projected_separately_by_domain():
    params = {
        "Fs": 200.0,
        "ntpoints": 20,
        "window": 0.05,
        "N_noise_sources": 2,
        "pad": 20,
    }
    leadfield_a = np.array([[1.0, 0.2, 0.0, 0.4], [0.1, 0.3, 1.0, 0.0]])
    leadfield_b = np.array([[0.3, 1.0, 0.2, 0.0], [1.0, 0.0, 0.1, 0.5]])

    latents = generate_noise_latents(4, params, np.random.default_rng(91))
    repeated = generate_noise_latents(4, params, np.random.default_rng(91))
    noise_a = project_noise_latents(leadfield_a, latents, params)
    noise_b = project_noise_latents(leadfield_b, latents, params)

    assert np.array_equal(latents["source_indices"], repeated["source_indices"])
    assert np.allclose(latents["source_noise"], repeated["source_noise"])
    assert noise_a.shape == noise_b.shape == (2, 20)
    assert np.isclose(noise_a.std(ddof=1), 1.0)
    assert np.isclose(noise_b.std(ddof=1), 1.0)
    assert not np.allclose(noise_a, noise_b)


def _minimal_dual_domain_payload(n_samples=4):
    sensor = np.arange(n_samples * 2 * 3, dtype=np.float32).reshape(n_samples, 2, 3)
    return {
        "Ys_rebuilt": sensor,
        "Ys_rebuilt_shifted": sensor + 1,
        "Ys_anna": sensor + 2,
        "Ys_anna_shifted": sensor + 3,
        "thetas": np.arange(n_samples * 9, dtype=np.float32).reshape(n_samples, 9),
        "thetas_shifted": np.arange(n_samples * 9, dtype=np.float32).reshape(n_samples, 9) + 1,
        "centroids_idx": np.tile(np.array([0, 1, 2]), (n_samples, 1)),
        "centroids_idx_shifted": np.tile(np.array([1, 2, 3]), (n_samples, 1)),
        "candidate_seeds": np.arange(n_samples),
        "G_rebuilt": np.ones((2, 4), dtype=np.float32),
        "G_anna": np.full((2, 4), 2, dtype=np.float32),
        "vertices": np.zeros((4, 3), dtype=np.float32),
        "edge_index": np.array([[1, 0]], dtype=np.int32),
        "edge_weight_mm": np.array([1.0], dtype=np.float32),
        "normals": np.zeros((4, 3), dtype=np.float32),
        "params_spatial": np.array({"N_src": 3}, dtype=object),
        "params_temporal": np.array({"ntpoints": 3}, dtype=object),
        "params_noise": np.array({"noise_level": 0.3}, dtype=object),
        "noise_level": np.array(0.3, dtype=np.float32),
        "meta": np.array({"generator": "test"}, dtype=object),
    }


def test_payload_requires_all_four_sensor_views_and_persisted_leadfields():
    payload = _minimal_dual_domain_payload()

    assert validate_dual_domain_payload(payload, expected_n=4) == (4, 2, 3)

    del payload["Ys_anna_shifted"]
    with np.testing.assert_raises_regex(KeyError, "Ys_anna_shifted"):
        validate_dual_domain_payload(payload, expected_n=4)


def test_write_locked_splits_persists_hashes_and_exact_disjoint_indices(tmp_path):
    payload = _minimal_dual_domain_payload()

    manifest = write_locked_splits(
        payload,
        tmp_path,
        expected_n=4,
        dev_size=2,
        split_seed=20260811,
    )

    assert set(manifest["files"]) == {"full", "dev", "heldout"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"].values())
    assert manifest["split"] == {"seed": 20260811, "dev_size": 2, "heldout_size": 2}
    with np.load(tmp_path / "dev.npz", allow_pickle=True) as dev:
        dev_indices = dev["full_sample_indices"]
        assert dev["Ys_rebuilt"].shape[0] == 2
        assert np.array_equal(dev["G_rebuilt"], payload["G_rebuilt"])
    with np.load(tmp_path / "heldout.npz", allow_pickle=True) as heldout:
        heldout_indices = heldout["full_sample_indices"]
        assert heldout["Ys_anna_shifted"].shape[0] == 2
    assert not set(dev_indices.tolist()) & set(heldout_indices.tolist())
    validation = validate_written_bundle(tmp_path)
    assert validation["valid"] is True
    assert validation["split_union_size"] == 4


def test_surface_normals_are_rotated_from_mri_to_forward_head_coordinates():
    source_vertices = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    source_normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mri_to_head = np.array(
        [
            [0.0, -1.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    target_vertices = np.array([[0.08, -0.2, 0.3], [0.1, -0.19, 0.3]])

    normals, maximum_distance = map_surface_normals_to_vertices(
        target_vertices,
        source_vertices,
        source_normals,
        mri_to_head,
    )

    assert maximum_distance < 1e-12
    assert np.allclose(normals, [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
