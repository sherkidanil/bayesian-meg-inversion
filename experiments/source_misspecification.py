"""Helpers for the frozen ±30% source-radius misspecification experiment."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import wilcoxon


FROZEN_RADIUS_SCALES = (0.7, 1.3)


class PrecomputedGaussianCache:
    """Expose precomputed slot/vertex topographies through the decoder cache API."""

    def __init__(self, dictionaries: np.ndarray):
        value = np.asarray(dictionaries, dtype=np.float64)
        if value.ndim != 3 or not np.all(np.isfinite(value)):
            raise ValueError("dictionaries must have shape (slots, sensors, vertices)")
        if value.shape[0] == 0 or value.shape[2] == 0:
            raise ValueError("dictionaries must be nonempty")
        self._dictionaries = value

    @property
    def cache_size(self) -> int:
        return int(self._dictionaries.shape[0] * self._dictionaries.shape[2])

    def matrix(self, centroids: np.ndarray) -> np.ndarray:
        vertices = np.asarray(centroids, dtype=np.int64)
        slots, _sensors, vertex_count = self._dictionaries.shape
        if vertices.shape != (slots,):
            raise ValueError(f"expected centroid shape {(slots,)}, got {vertices.shape}")
        if np.any(vertices < 0) or np.any(vertices >= vertex_count):
            raise IndexError("centroid is outside the precomputed dictionary")
        return np.column_stack(
            [
                self._dictionaries[slot, :, int(vertex)]
                for slot, vertex in enumerate(vertices)
            ]
        )


def extract_frozen_s400_selection(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the nominal dev-frozen S=400 quadratic medoid selection."""
    try:
        selection = dict(summary["heldout_by_draw_count"]["400"]["selection"])
    except (KeyError, TypeError) as error:
        raise ValueError("summary does not contain a frozen S=400 selection") from error
    expected = {
        "n_draws": 400,
        "estimator": "gibbs",
        "aggregator": "medoid",
        "score_form": "quadratic",
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("S=400 selection must be the nominal quadratic medoid")
    tau = float(selection.get("tau", np.nan))
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("S=400 selection must contain a positive finite tau")
    selection["tau"] = tau
    return selection


def build_global_common_mask(
    headline_mask: np.ndarray,
    patch_valid_by_scale: Mapping[str, np.ndarray],
    cfm_valid_by_scale: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Intersect nominal and both misspecified variants across methods/seeds."""
    expected_keys = {"nominal", "0.7", "1.3"}
    if set(patch_valid_by_scale) != expected_keys or set(cfm_valid_by_scale) != expected_keys:
        raise ValueError("validity mappings require nominal, 0.7, and 1.3")
    mask = np.asarray(headline_mask)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError("headline_mask must be one-dimensional boolean")
    common = mask.copy()
    for scale in ("nominal", "0.7", "1.3"):
        patch = np.asarray(patch_valid_by_scale[scale])
        cfm = np.asarray(cfm_valid_by_scale[scale])
        if patch.dtype != np.bool_ or patch.shape != mask.shape:
            raise ValueError("patch validity must align with headline_mask")
        if cfm.dtype != np.bool_ or cfm.ndim != 2 or cfm.shape[1] != mask.size:
            raise ValueError("CFM validity must have shape (seeds, cases)")
        common &= patch & np.all(cfm, axis=0)
    return common


def scaled_spatial_params(
    params_spatial: Mapping[str, Any], scale: float
) -> dict[str, Any]:
    """Copy spatial parameters while scaling only radii by a frozen factor."""
    factor = float(scale)
    if factor not in FROZEN_RADIUS_SCALES:
        raise ValueError("radius scale must be exactly 0.7 and 1.3")
    result = dict(params_spatial)
    radii = np.asarray(params_spatial["R_mm"], dtype=np.float64)
    if radii.ndim != 1 or radii.size == 0 or np.any(radii <= 0.0):
        raise ValueError("R_mm must be a nonempty positive vector")
    result["R_mm"] = radii.copy() * factor
    if "A0" in result:
        result["A0"] = np.asarray(result["A0"], dtype=np.float64).copy()
    return result


def degradation_summary(
    nominal_errors_mm: np.ndarray, misspecified_errors_mm: np.ndarray
) -> dict[str, float | int]:
    """Summarize paired degradation on one already-frozen finite mask."""
    nominal = np.asarray(nominal_errors_mm, dtype=np.float64).reshape(-1)
    wrong = np.asarray(misspecified_errors_mm, dtype=np.float64).reshape(-1)
    if nominal.shape != wrong.shape or nominal.size == 0:
        raise ValueError("paired errors must have one nonempty shape")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(wrong)):
        raise ValueError("degradation inputs must already be finite")
    nominal_mean = float(np.mean(nominal))
    wrong_mean = float(np.mean(wrong))
    if nominal_mean <= 0.0:
        raise ValueError("nominal mean must be positive")
    absolute = wrong_mean - nominal_mean
    return {
        "n_cases": int(nominal.size),
        "nominal_mean_mm": nominal_mean,
        "misspecified_mean_mm": wrong_mean,
        "absolute_degradation_mm": absolute,
        "relative_degradation_percent": 100.0 * absolute / nominal_mean,
    }


def _wilcoxon_pvalue(first: np.ndarray, second: np.ndarray, alternative: str) -> float:
    differences = np.asarray(first) - np.asarray(second)
    if np.all(differences == 0.0):
        return 1.0
    return float(
        wilcoxon(first, second, alternative=alternative, zero_method="wilcox").pvalue
    )


def hierarchical_relative_degradation_difference(
    patch_nominal_mm: np.ndarray,
    patch_misspecified_mm: np.ndarray,
    cfm_nominal_by_seed_mm: np.ndarray,
    cfm_misspecified_by_seed_mm: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare relative degradation while resampling CFM seeds and paired cases."""
    patch_nominal = np.asarray(patch_nominal_mm, dtype=np.float64).reshape(-1)
    patch_wrong = np.asarray(patch_misspecified_mm, dtype=np.float64).reshape(-1)
    cfm_nominal = np.asarray(cfm_nominal_by_seed_mm, dtype=np.float64)
    cfm_wrong = np.asarray(cfm_misspecified_by_seed_mm, dtype=np.float64)
    if patch_nominal.shape != patch_wrong.shape or patch_nominal.size == 0:
        raise ValueError("patch errors must have one nonempty paired shape")
    expected = (cfm_nominal.shape[0], patch_nominal.size)
    if (
        cfm_nominal.ndim != 2
        or cfm_nominal.shape != expected
        or cfm_wrong.shape != expected
        or cfm_nominal.shape[0] < 2
    ):
        raise ValueError("CFM errors must align as (at least two seeds, cases)")
    if not all(
        np.all(np.isfinite(value))
        for value in (patch_nominal, patch_wrong, cfm_nominal, cfm_wrong)
    ):
        raise ValueError("relative-degradation inputs must already be finite")
    repeats = int(n_resamples)
    if repeats != n_resamples or repeats <= 0:
        raise ValueError("n_resamples must be a positive integer")

    patch_base = float(np.mean(patch_nominal))
    cfm_base = float(np.mean(cfm_nominal))
    if patch_base <= 0.0 or cfm_base <= 0.0:
        raise ValueError("nominal means must be positive")
    patch_relative = 100.0 * float(np.mean(patch_wrong - patch_nominal)) / patch_base
    cfm_relative = 100.0 * float(np.mean(cfm_wrong - cfm_nominal)) / cfm_base
    difference = patch_relative - cfm_relative

    cfm_seed_bases = np.mean(cfm_nominal, axis=1)
    patch_case_relative = 100.0 * (patch_wrong - patch_nominal) / patch_base
    cfm_case_relative = np.mean(
        100.0 * (cfm_wrong - cfm_nominal) / cfm_seed_bases[:, None], axis=0
    )

    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(repeats, dtype=np.float64)
    n_seeds, n_cases = cfm_nominal.shape
    for repeat in range(repeats):
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        case_indices = rng.integers(0, n_cases, size=n_cases)
        patch_base_draw = float(np.mean(patch_nominal[case_indices]))
        patch_delta_draw = float(
            np.mean((patch_wrong - patch_nominal)[case_indices])
        )
        cfm_base_draw = float(
            np.mean(cfm_nominal[seed_indices][:, case_indices])
        )
        cfm_delta_draw = float(
            np.mean((cfm_wrong - cfm_nominal)[seed_indices][:, case_indices])
        )
        bootstrap[repeat] = (
            100.0 * patch_delta_draw / patch_base_draw
            - 100.0 * cfm_delta_draw / cfm_base_draw
        )
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n_seeds": int(n_seeds),
        "n_cases": int(n_cases),
        "patch_relative_degradation_percent": patch_relative,
        "cfm_relative_degradation_percent": cfm_relative,
        "difference_patch_minus_cfm_percentage_points": difference,
        "hierarchical_bootstrap_ci95_percentage_points": [
            float(lower),
            float(upper),
        ],
        "wilcoxon_p_patch_greater": _wilcoxon_pvalue(
            patch_case_relative, cfm_case_relative, "greater"
        ),
        "wilcoxon_p_patch_less": _wilcoxon_pvalue(
            patch_case_relative, cfm_case_relative, "less"
        ),
        "wilcoxon_units": (
            "per-case degradation divided by each method/seed nominal mean"
        ),
        "n_resamples": repeats,
        "bootstrap_seed": int(seed),
    }


def classify_misspecification_outcome(
    results_by_scale: Mapping[str, Mapping[str, Any]],
) -> str:
    """Apply the interpretation rule frozen before real misspecification runs."""
    if set(results_by_scale) != {"0.7", "1.3"}:
        raise ValueError("outcome classification requires scales 0.7 and 1.3")
    comparisons = [results_by_scale[key]["relative_difference"] for key in ("0.7", "1.3")]
    patch_worse = all(
        float(item["hierarchical_bootstrap_ci95_percentage_points"][0]) > 0.0
        and float(item["wilcoxon_p_patch_greater"]) < 0.05
        for item in comparisons
    )
    patch_better = all(
        float(item["hierarchical_bootstrap_ci95_percentage_points"][1]) < 0.0
        and float(item["wilcoxon_p_patch_less"]) < 0.05
        for item in comparisons
    )
    if patch_worse:
        return "learned_prior_more_robust"
    if patch_better:
        return "patch_rap_more_robust"
    return "comparable_or_direction_dependent"
