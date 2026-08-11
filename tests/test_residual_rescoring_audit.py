from pathlib import Path
import inspect
import json
import sys

import numpy as np
import pytest

MODULE_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "tests"
    else Path(__file__).resolve().parent
)
sys.path.insert(0, str(MODULE_ROOT))

from residual_rescoring_audit import (
    aggregate_spearman_diagnostics,
    build_commit_provenance,
    draw_seed_for_sample,
    frozen_smooth_coordinates,
    load_frozen_smooth_selection,
    load_anna_audit_inputs,
    mean_assignment_error,
    parse_args,
    pooled_within_sample_rank_spearman,
    rank_nonfinite_worst,
    spearman_with_nonfinite_worst,
    validate_localization_g,
)


def _write_scenario(path: Path) -> None:
    params_spatial = {
        "R_mm": np.array([10.0]),
        "sigma_mm_scale": 0.5,
        "A0": np.array([1.0]),
    }
    np.savez(
        path,
        Ys=np.ones((2, 3, 4)),
        Ys_shifted=2.0 * np.ones((2, 3, 4)),
        centroids_idx=np.array([[0], [1]]),
        centroids_idx_shifted=np.array([[1], [2]]),
        vertices=np.zeros((4, 3)),
        edge_index=np.array([[0, 1], [1, 2], [2, 3]]),
        edge_weight_mm=np.ones(3),
        params_spatial=params_spatial,
        G_for_simulation=np.full((2, 3, 4), np.nan),
    )


def test_anna_loader_uses_constant_dg0_g_and_ignores_scenario_g(tmp_path):
    anna_dg0 = tmp_path / "anna_dg0.npz"
    localization = np.arange(12, dtype=np.float64).reshape(3, 4)
    np.savez(anna_dg0, G_for_simulation=np.stack([localization, localization]))
    scenario = tmp_path / "anna_dgnz.npz"
    _write_scenario(scenario)

    loaded = load_anna_audit_inputs(anna_dg0, scenario)

    assert np.array_equal(loaded["localization_G"], localization)
    assert loaded["localization_G"].ndim == 2
    assert "G_for_simulation" not in loaded["scenario"]


def test_localization_validator_rejects_per_sample_leadfields():
    with pytest.raises(ValueError, match="2-D"):
        validate_localization_g(np.zeros((5, 3, 4)))


def test_anna_loader_api_has_no_perturbed_g_argument():
    parameters = inspect.signature(load_anna_audit_inputs).parameters

    assert tuple(parameters) == ("anna_dg0_npz", "scenario_npz")
    assert not any("dgnz_g" in name or "simulation_g" in name for name in parameters)


def test_mean_assignment_error_is_permutation_invariant():
    distance = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ]
    )

    assert mean_assignment_error(np.array([0, 2]), np.array([2, 0]), distance) == 0.0


def test_nonfinite_errors_are_tied_at_the_worst_rank():
    ranks = rank_nonfinite_worst(np.array([1.0, np.inf, 3.0, np.nan]))

    assert np.allclose(ranks, [1.0, 3.5, 2.0, 3.5])


def test_spearman_keeps_nonfinite_geodesics_as_worst():
    residuals = np.array([0.1, 0.2, 0.3])
    errors = np.array([1.0, np.inf, 3.0])

    assert spearman_with_nonfinite_worst(residuals, errors) == pytest.approx(0.5)


def test_pooled_spearman_ranks_within_each_sample_before_pooling():
    residuals = [np.array([1.0, 2.0, 3.0]), np.array([30.0, 10.0, 20.0])]
    errors = [np.array([2.0, 4.0, 6.0]), np.array([3.0, 1.0, 2.0])]

    assert pooled_within_sample_rank_spearman(residuals, errors) == pytest.approx(1.0)


def test_sample_draw_seed_is_deterministic_and_input_kind_independent():
    assert draw_seed_for_sample(20260810, 17) == draw_seed_for_sample(20260810, 17)
    assert draw_seed_for_sample(20260810, 17) != draw_seed_for_sample(20260810, 18)


def test_correlation_summary_reports_per_sample_raw_and_within_rank():
    residuals = [np.array([1.0, 2.0, 3.0]), np.array([30.0, 10.0, 20.0])]
    errors = [np.array([2.0, 4.0, 6.0]), np.array([3.0, 1.0, 2.0])]

    summary = aggregate_spearman_diagnostics(residuals, errors)

    assert summary["n_samples"] == 2
    assert summary["per_sample_mean"] == pytest.approx(1.0)
    assert summary["pooled_within_sample_rank"] == pytest.approx(1.0)
    assert np.isfinite(summary["pooled_raw"])


def test_cli_requires_one_dg0_g_source_and_has_no_dgnz_g_option():
    args = parse_args(
        [
            "--anna_dg0_npz",
            "anna_dg0.npz",
            "--scenario_npz",
            "anna_dgnz.npz",
            "--out_dir",
            "report",
        ]
    )

    assert args.anna_dg0_npz == "anna_dg0.npz"
    assert args.scenario_npz == "anna_dgnz.npz"
    assert "anna_dgnz_g" not in vars(args)


def test_cli_accepts_explicit_canonical_model_key_for_alias():
    args = parse_args(
        [
            "--anna_dg0_npz", "anna_dg0.npz",
            "--scenario_npz", "anna_dg0.npz",
            "--out_dir", "report",
            "--model_key", "trial81_honest_4965",
        ]
    )

    assert args.model_key == "trial81_honest_4965"


def test_commit_provenance_preserves_start_head_when_repository_changes():
    provenance = build_commit_provenance("start-head", "write-head")

    assert provenance == {
        "run_start_git_commit": "start-head",
        "artifact_write_git_commit": "write-head",
        "repository_changed_during_run": True,
    }


def test_frozen_smooth_selection_requires_honest_quadratic_dev_choice(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "mode": "dev_tune",
                "model_key": "trial81_honest_4965",
                "selected": {
                    "estimator": "gibbs",
                    "score_form": "quadratic",
                    "tau": 0.25,
                },
            }
        )
    )

    selected = load_frozen_smooth_selection(path, "trial81_honest_4965")

    assert selected["tau"] == pytest.approx(0.25)
    assert selected["score_form"] == "quadratic"
    with pytest.raises(ValueError, match="model key"):
        load_frozen_smooth_selection(path, "trial81_push_3771")


def test_frozen_smooth_coordinates_use_only_draws_residuals_and_frozen_choice():
    draws = np.array([[[0.0]], [[10.0]]])
    residuals = np.array([0.0, 1.0])
    selected = {"estimator": "gibbs", "score_form": "quadratic", "tau": 0.1}

    estimate, weights = frozen_smooth_coordinates(draws, residuals, selected)

    assert estimate.shape == (1, 1)
    assert estimate[0, 0] < 1e-10
    assert weights.sum() == pytest.approx(1.0)
