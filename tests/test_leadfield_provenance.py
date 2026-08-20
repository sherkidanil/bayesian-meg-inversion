from pathlib import Path
import inspect
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meg_cfm.leadfield_provenance import (
    assert_matching_geometry,
    capture_environment,
    column_direction_diagnostics,
    infer_fixed_leadfield_identity,
    load_anna_localization_g,
    optimal_global_scale,
    principal_angles_deg,
    projector_patch_angles,
    parse_args,
    sample_distinct_triplets,
    sha256_file,
    summarize_angles,
    truth_residual_for_provenance,
)


def test_optimal_global_scale_removes_pure_scalar_difference():
    rebuilt = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    anna = 2.75 * rebuilt

    result = optimal_global_scale(target=anna, candidate=rebuilt)

    assert result["alpha"] == pytest.approx(2.75)
    assert result["relative_residual"] == pytest.approx(0.0, abs=1e-14)


def test_column_direction_diagnostics_reports_sign_aligned_equivalence():
    rebuilt = np.array([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    anna = rebuilt * np.array([[4.0, -2.0]])

    result = column_direction_diagnostics(anna, rebuilt)

    assert result["raw_relative_frobenius"] > 0.0
    assert result["sign_aligned_relative_frobenius"] == pytest.approx(0.0, abs=1e-14)
    assert result["median_absolute_cosine"] == pytest.approx(1.0)


def test_principal_angles_ignore_invertible_right_mixing():
    rng = np.random.default_rng(4)
    basis = rng.normal(size=(12, 3))
    mixed = basis @ np.array([[2.0, 1.0, 0.0], [0.0, -3.0, 0.5], [0.0, 0.0, 4.0]])

    angles = principal_angles_deg(basis, mixed)

    assert angles.shape == (3,)
    assert np.allclose(angles, 0.0, atol=1e-6)


def test_angle_summary_reports_distribution():
    rows = np.array([[0.0, 10.0, 20.0], [5.0, 15.0, 25.0]])

    summary = summarize_angles(rows)

    assert summary["all_angles_deg"]["median"] == pytest.approx(12.5)
    assert summary["max_angle_per_case_deg"]["median"] == pytest.approx(22.5)
    assert summary["n_cases"] == 2


def test_triplet_sampling_is_deterministic_and_distinct():
    first = sample_distinct_triplets(n_vertices=20, count=7, seed=19)
    second = sample_distinct_triplets(n_vertices=20, count=7, seed=19)

    assert np.array_equal(first, second)
    assert first.shape == (7, 3)
    assert all(len(set(row.tolist())) == 3 for row in first)


def test_environment_manifest_hashes_inputs_and_records_versions(tmp_path):
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"leadfield provenance")

    manifest = capture_environment(
        input_files={"evidence": evidence},
        arrays={"G": np.eye(3)},
        command="leadfield_provenance.py --seed 19",
        seed=19,
        git_commit="deadbeef",
    )

    assert manifest["seed"] == 19
    assert manifest["git_commit"] == "deadbeef"
    assert manifest["input_files"]["evidence"]["sha256"] == sha256_file(evidence)
    assert manifest["arrays"]["G"]["shape"] == [3, 3]
    for package in ("numpy", "scipy", "mne", "nibabel"):
        assert package in manifest["packages"]


def test_anna_localization_loader_requires_one_constant_leadfield(tmp_path):
    path = tmp_path / "anna_dg0.npz"
    leadfield = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    leadfield[1] = leadfield[0]
    np.savez(path, G_for_simulation=leadfield)

    loaded, diagnostic = load_anna_localization_g(path)

    assert loaded.shape == (3, 4)
    assert np.array_equal(loaded, leadfield[0])
    assert diagnostic == {"n_matrices": 2, "max_abs_variation": 0.0}


def test_anna_localization_loader_rejects_perturbed_per_sample_g(tmp_path):
    path = tmp_path / "anna_dgnz.npz"
    leadfield = np.zeros((2, 3, 4), dtype=np.float64)
    leadfield[1, 0, 0] = 1e-6
    np.savez(path, G_for_simulation=leadfield)

    with pytest.raises(ValueError, match="not constant"):
        load_anna_localization_g(path)


def test_provenance_cli_exposes_no_perturbed_g_argument():
    args = parse_args(
        [
            "--anna_dg0_npz",
            "anna_dg0.npz",
            "--fixed_npz",
            "fixed.npz",
            "--out_dir",
            "report",
        ]
    )

    assert args.anna_dg0_npz == "anna_dg0.npz"
    assert "anna_dgnz" not in inspect.signature(parse_args).parameters


def test_matching_geometry_rejects_changed_vertices():
    reference = {
        "vertices": np.zeros((3, 3)),
        "edge_index": np.array([[0, 1], [1, 2]]),
        "edge_weight_mm": np.array([1.0, 1.0]),
    }
    changed = {**reference, "vertices": np.ones((3, 3))}

    with pytest.raises(ValueError, match="vertices"):
        assert_matching_geometry(reference, changed, label="changed")


def test_patch_angles_are_zero_for_global_leadfield_scaling():
    from scipy.sparse import csr_matrix

    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    leadfield = np.array(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]]
    )
    params = {
        "R_mm": np.array([0.1, 0.1]),
        "sigma_mm_scale": 0.5,
        "A0": np.array([1.0, 1.0]),
    }
    centroids = np.array([[0, 1], [2, 3]])

    rows = projector_patch_angles(leadfield, -3.0 * leadfield, centroids, graph, params)

    assert rows.shape == (2, 2)
    assert np.allclose(rows, 0.0, atol=1e-6)


def test_truth_residual_is_separated_from_honest_estimator_api():
    import residual_rescoring
    from scipy.sparse import csr_matrix

    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    localization_g = np.eye(3)
    params = {
        "R_mm": np.array([0.1]),
        "sigma_mm_scale": 0.5,
        "A0": np.array([1.0]),
    }
    y = localization_g[:, [1]] @ np.array([[1.0, 2.0]])

    residual = truth_residual_for_provenance(
        y, np.array([1]), localization_g, graph, params
    )

    assert residual == pytest.approx(0.0, abs=1e-12)
    assert not hasattr(residual_rescoring, "truth_residual_for_provenance")
    assert "true_centroids" in inspect.signature(truth_residual_for_provenance).parameters


def test_fixed_identity_inference_requires_anna_calibration_and_paired_consensus():
    calibrated = {
        "fixed": {
            "fraction_rebuilt_lower": 298 / 300,
            "fraction_anna_lower": 2 / 300,
        },
        "anna_dg0": {
            "fraction_rebuilt_lower": 2 / 315,
            "fraction_anna_lower": 313 / 315,
        },
    }

    result = infer_fixed_leadfield_identity(calibrated)

    assert result["classification"] == "rebuilt_like"
    assert result["calibration_passed"] is True
    assert result["decisive_fraction"] == pytest.approx(0.95)

    ambiguous = {
        **calibrated,
        "fixed": {
            "fraction_rebuilt_lower": 0.55,
            "fraction_anna_lower": 0.45,
        },
    }
    assert infer_fixed_leadfield_identity(ambiguous)["classification"] == "unidentified"

    failed_calibration = {
        **calibrated,
        "anna_dg0": {
            "fraction_rebuilt_lower": 0.60,
            "fraction_anna_lower": 0.40,
        },
    }
    result = infer_fixed_leadfield_identity(failed_calibration)
    assert result["classification"] == "calibration_failed"
    assert result["calibration_passed"] is False
