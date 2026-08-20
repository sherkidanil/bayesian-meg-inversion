from __future__ import annotations

import numpy as np
import pytest

from aggregate_source_misspecification import _canonical_cfm_artifact
from source_misspecification import (
    PrecomputedGaussianCache,
    build_global_common_mask,
    classify_misspecification_outcome,
    degradation_summary,
    extract_frozen_s400_selection,
    hierarchical_relative_degradation_difference,
    scaled_spatial_params,
)


def test_canonical_cfm_artifact_maps_nominal_s400_schema():
    nominal = {
        "full_sample_indices": np.asarray([4, 7]),
        "s400_decoded_mae_geo_mm": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "s400_decoded_valid_branch": np.asarray(
            [[True, True], [True, False]], dtype=bool
        ),
        "s400_decoded_valid_pair": np.asarray([True, False], dtype=bool),
    }

    canonical = _canonical_cfm_artifact(nominal)

    np.testing.assert_array_equal(canonical["full_sample_indices"], [4, 7])
    np.testing.assert_array_equal(canonical["mae_geo_mm"], nominal["s400_decoded_mae_geo_mm"])
    np.testing.assert_array_equal(canonical["valid_branch"], nominal["s400_decoded_valid_branch"])
    np.testing.assert_array_equal(canonical["valid_pair"], nominal["s400_decoded_valid_pair"])


def test_scaled_spatial_params_changes_only_radii_and_accepts_only_frozen_scales():
    original = {
        "R_mm": np.asarray([10.0, 8.0, 10.0]),
        "A0": np.ones(3),
        "sigma_mm_scale": 1.0 / 3.0,
    }

    scaled = scaled_spatial_params(original, 0.7)

    np.testing.assert_array_equal(scaled["R_mm"], [7.0, 5.6, 7.0])
    np.testing.assert_array_equal(scaled["A0"], original["A0"])
    assert scaled["sigma_mm_scale"] == original["sigma_mm_scale"]
    np.testing.assert_array_equal(original["R_mm"], [10.0, 8.0, 10.0])
    with pytest.raises(ValueError, match="0.7 and 1.3"):
        scaled_spatial_params(original, 0.8)


def test_degradation_summary_reports_absolute_and_percent_change_from_same_mask():
    nominal = np.asarray([1.0, 2.0, 3.0])
    misspecified = np.asarray([2.0, 3.0, 4.0])

    result = degradation_summary(nominal, misspecified)

    assert result["nominal_mean_mm"] == 2.0
    assert result["misspecified_mean_mm"] == 3.0
    assert result["absolute_degradation_mm"] == 1.0
    assert result["relative_degradation_percent"] == 50.0


def test_hierarchical_difference_resamples_training_seeds_and_cases():
    patch_nominal = np.asarray([1.0, 1.0, 1.0, 1.0])
    patch_wrong = np.asarray([3.0, 3.0, 3.0, 3.0])
    cfm_nominal = np.asarray([[2.0] * 4, [4.0] * 4])
    cfm_wrong = np.asarray([[3.0] * 4, [5.0] * 4])

    result = hierarchical_relative_degradation_difference(
        patch_nominal,
        patch_wrong,
        cfm_nominal,
        cfm_wrong,
        n_resamples=500,
        seed=17,
    )

    assert result["n_seeds"] == 2
    assert result["n_cases"] == 4
    assert result["patch_relative_degradation_percent"] == 200.0
    assert result["cfm_relative_degradation_percent"] == pytest.approx(100.0 / 3.0)
    assert result["difference_patch_minus_cfm_percentage_points"] > 0.0
    assert result["hierarchical_bootstrap_ci95_percentage_points"][0] > 0.0


def test_outcome_rule_requires_same_significant_direction_at_both_scales():
    stronger_patch = {
        "relative_difference": {
            "hierarchical_bootstrap_ci95_percentage_points": [1.0, 5.0],
            "wilcoxon_p_patch_greater": 0.01,
            "wilcoxon_p_patch_less": 0.99,
        }
    }
    stronger_cfm = {
        "relative_difference": {
            "hierarchical_bootstrap_ci95_percentage_points": [-5.0, -1.0],
            "wilcoxon_p_patch_greater": 0.99,
            "wilcoxon_p_patch_less": 0.01,
        }
    }
    unresolved = {
        "relative_difference": {
            "hierarchical_bootstrap_ci95_percentage_points": [-1.0, 2.0],
            "wilcoxon_p_patch_greater": 0.2,
            "wilcoxon_p_patch_less": 0.8,
        }
    }

    assert classify_misspecification_outcome(
        {"0.7": stronger_patch, "1.3": stronger_patch}
    ) == "learned_prior_more_robust"
    assert classify_misspecification_outcome(
        {"0.7": stronger_cfm, "1.3": stronger_cfm}
    ) == "patch_rap_more_robust"
    assert classify_misspecification_outcome(
        {"0.7": stronger_patch, "1.3": unresolved}
    ) == "comparable_or_direction_dependent"


def test_frozen_s400_selection_requires_nominal_quadratic_medoid():
    summary = {
        "heldout_by_draw_count": {
            "400": {
                "selection": {
                    "n_draws": 400,
                    "estimator": "gibbs",
                    "aggregator": "medoid",
                    "score_form": "quadratic",
                    "tau": 0.0125,
                }
            }
        }
    }

    selection = extract_frozen_s400_selection(summary)

    assert selection["tau"] == 0.0125
    broken = {"heldout_by_draw_count": {"400": {"selection": dict(selection)}}}
    broken["heldout_by_draw_count"]["400"]["selection"]["estimator"] = "argmin"
    with pytest.raises(ValueError, match="quadratic medoid"):
        extract_frozen_s400_selection(broken)


def test_global_common_mask_intersects_all_methods_scales_and_seeds():
    headline = np.asarray([True, True, True, False])
    patch_valid = {
        "nominal": np.asarray([True, True, True, True]),
        "0.7": np.asarray([True, False, True, True]),
        "1.3": np.asarray([True, True, True, True]),
    }
    cfm_valid = {
        "nominal": np.asarray([[True, True, True, True], [True, True, True, True]]),
        "0.7": np.asarray([[True, True, True, True], [True, True, False, True]]),
        "1.3": np.asarray([[True, True, True, True], [True, True, True, True]]),
    }

    mask = build_global_common_mask(headline, patch_valid, cfm_valid)

    np.testing.assert_array_equal(mask, [True, False, False, False])


def test_precomputed_gaussian_cache_returns_slot_specific_topographies():
    dictionaries = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    cache = PrecomputedGaussianCache(dictionaries)

    matrix = cache.matrix(np.asarray([3, 1]))

    np.testing.assert_array_equal(matrix[:, 0], dictionaries[0, :, 3])
    np.testing.assert_array_equal(matrix[:, 1], dictionaries[1, :, 1])
    assert matrix.shape == (3, 2)
    with pytest.raises(IndexError, match="outside"):
        cache.matrix(np.asarray([4, 1]))
