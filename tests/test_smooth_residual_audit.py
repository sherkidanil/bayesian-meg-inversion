from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent if HERE.name == "tests" else HERE
sys.path.insert(0, str(MODULE_ROOT))

from smooth_residual_audit import (
    HONEST_MODEL_NAME,
    assert_dev_tuning_input,
    candidate_estimates,
    compute_argmin_jump_diagnostics,
    derive_temperature_grid,
    evaluate_estimate_batch,
    load_locked_heldout_selection,
    select_preregistered_candidate,
)


def test_temperature_grid_is_deterministic_and_dimensionally_scaled():
    residuals = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.8]])

    quadratic = derive_temperature_grid(residuals, "quadratic")
    linear = derive_temperature_grid(residuals, "linear")

    assert quadratic["score_gap_reference"] == pytest.approx(0.075)
    assert quadratic["tau_reference"] == pytest.approx(np.sqrt(0.075))
    assert linear["score_gap_reference"] == pytest.approx(0.15)
    assert linear["tau_reference"] == pytest.approx(0.15)
    assert np.allclose(
        quadratic["temperatures"],
        np.sqrt(0.075) * np.power(2.0, np.arange(-8, 9)),
    )
    assert np.array_equal(
        quadratic["temperatures"], derive_temperature_grid(residuals, "quadratic")["temperatures"]
    )


def test_selection_minimizes_deltax_subject_to_ten_percent_accuracy_constraint():
    rows = [
        {"estimator": "argmin", "mae_euc_mm": 2.0, "deltax_euc_mm": 6.0},
        {
            "estimator": "gibbs",
            "score_form": "quadratic",
            "tau": 0.1,
            "mae_euc_mm": 2.19,
            "deltax_euc_mm": 4.0,
        },
        {
            "estimator": "gibbs",
            "score_form": "linear",
            "tau": 0.2,
            "mae_euc_mm": 2.21,
            "deltax_euc_mm": 3.0,
        },
    ]

    selected = select_preregistered_candidate(rows)

    assert selected["score_form"] == "quadratic"
    assert selected["tau"] == pytest.approx(0.1)
    assert selected["accuracy_limit_mm"] == pytest.approx(2.2)


def test_selection_ties_break_by_mae_then_quadratic_then_lower_tau():
    rows = [
        {"estimator": "argmin", "mae_euc_mm": 3.0, "deltax_euc_mm": 7.0},
        {
            "estimator": "gibbs",
            "score_form": "linear",
            "tau": 0.05,
            "mae_euc_mm": 3.1,
            "deltax_euc_mm": 5.0,
        },
        {
            "estimator": "gibbs",
            "score_form": "quadratic",
            "tau": 0.2,
            "mae_euc_mm": 3.1,
            "deltax_euc_mm": 5.0,
        },
        {
            "estimator": "gibbs",
            "score_form": "quadratic",
            "tau": 0.1,
            "mae_euc_mm": 3.1,
            "deltax_euc_mm": 5.0,
        },
    ]

    selected = select_preregistered_candidate(rows)

    assert selected["score_form"] == "quadratic"
    assert selected["tau"] == pytest.approx(0.1)


def test_tune_input_requires_dev_marker_and_honest_model(tmp_path):
    dev = tmp_path / "dev.npz"
    heldout = tmp_path / "heldout.npz"
    np.savez(dev, split_kind=np.array("dev"))
    np.savez(heldout, split_kind=np.array("heldout"))

    assert HONEST_MODEL_NAME == "trial81_honest_4965"
    assert_dev_tuning_input(dev, HONEST_MODEL_NAME)
    with pytest.raises(ValueError, match="held-out"):
        assert_dev_tuning_input(heldout, HONEST_MODEL_NAME)
    with pytest.raises(ValueError, match=HONEST_MODEL_NAME):
        assert_dev_tuning_input(dev, "trial81_push_3771")


def test_candidate_estimates_and_primary_metrics_use_paired_inputs():
    draws = np.array(
        [
            [
                [[[0.0, 0.0, 0.0]], [[0.002, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0]], [[0.001, 0.0, 0.0]]],
            ]
        ]
    )
    residuals = np.array([[[0.1, 0.2], [0.2, 0.1]]])
    vertices = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]])
    geodesic = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

    estimates, weight_metrics = candidate_estimates(draws, residuals, estimator="argmin")
    metrics = evaluate_estimate_batch(
        estimates,
        np.array([[0]]),
        np.array([[1]]),
        vertices,
        geodesic,
    )

    assert weight_metrics["mean_effective_sample_size"] == pytest.approx(1.0)
    assert metrics["mae_euc_mm"] == pytest.approx(0.0)
    assert metrics["deltax_euc_mm"] == pytest.approx(1.0)
    assert metrics["mae_geo_mm"] == pytest.approx(0.0)
    assert metrics["deltax_geo_mm"] == pytest.approx(1.0)
    assert metrics["mae_geo_nonfinite"] == 0
    assert metrics["deltax_geo_nonfinite"] == 0


def test_argmin_jump_diagnostics_separate_index_switch_from_trajectory_motion():
    draws = np.array(
        [
            [
                [[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]],
            ],
            [
                [[[0.0, 0.0, 0.0]], [[0.002, 0.0, 0.0]]],
                [[[0.001, 0.0, 0.0]], [[0.003, 0.0, 0.0]]],
            ],
        ]
    )
    residuals = np.array(
        [
            [[0.1, 0.2], [0.1, 0.2]],
            [[0.1, 0.2], [0.2, 0.1]],
        ]
    )
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.003, 0.0, 0.0]]
    )
    geodesic = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)

    result = compute_argmin_jump_diagnostics(draws, residuals, vertices, geodesic)

    assert result["switch_fraction"] == pytest.approx(0.5)
    assert result["index_switched"].tolist() == [False, True]
    assert result["selected_theta_mm"][1] == pytest.approx(3.0)
    assert result["same_index_original_winner_theta_mm"][1] == pytest.approx(1.0)
    assert result["same_index_shifted_winner_theta_mm"][1] == pytest.approx(1.0)
    assert result["within_original_theta_mm"][1] == pytest.approx(2.0)
    assert result["within_shifted_theta_mm"][1] == pytest.approx(2.0)


def test_heldout_lock_requires_frozen_selection_and_bundle_hashes(tmp_path):
    dev = tmp_path / "dev.npz"
    heldout = tmp_path / "heldout.npz"
    np.savez(dev, split_kind=np.array("dev"))
    np.savez(heldout, split_kind=np.array("heldout"))

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "files": {
            "dev": {"path": "dev.npz", "sha256": digest(dev), "samples": 1},
            "heldout": {
                "path": "heldout.npz",
                "sha256": digest(heldout),
                "samples": 1,
            },
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    selection = {
        "mode": "dev_tune",
        "model_key": HONEST_MODEL_NAME,
        "dataset_sha256": digest(dev),
        "selected": {
            "candidate_id": "quadratic_2^-4",
            "estimator": "gibbs",
            "score_form": "quadratic",
            "tau": 0.02,
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))

    loaded = load_locked_heldout_selection(heldout, selection_path)

    assert loaded["selected"]["tau"] == pytest.approx(0.02)
    selection["dataset_sha256"] = "0" * 64
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="dev hash"):
        load_locked_heldout_selection(heldout, selection_path)
    with pytest.raises(ValueError, match="held-out"):
        load_locked_heldout_selection(dev, selection_path)
