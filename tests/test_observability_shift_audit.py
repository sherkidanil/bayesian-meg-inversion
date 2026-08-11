from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent if HERE.name == "tests" else HERE
sys.path.insert(0, str(MODULE_ROOT))

from observability_shift_audit import (
    compute_observability_shift,
    correlate_observability_with_mae_asymmetry,
)


def test_observability_uses_matching_leadfield_column_norm_log_ratio():
    leadfield = np.array(
        [
            [1.0, np.e, np.e**2, np.e**3],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    result = compute_observability_shift(
        leadfield,
        np.array([[0, 1]]),
        np.array([[1, 3]]),
    )

    assert np.allclose(result["source_log_ratio"], [[1.0, 2.0]])
    assert result["sample_mean_log_ratio"][0] == pytest.approx(1.5)
    assert np.allclose(result["original_column_norm"], [[1.0, np.e]])
    assert np.allclose(result["shifted_column_norm"], [[np.e, np.e**3]])


def test_correlations_use_original_minus_shifted_mae_and_report_subsets():
    log_ratio = np.arange(6, dtype=float)
    normal_angles = np.column_stack([log_ratio, log_ratio / 2.0, log_ratio / 3.0])
    mae_shifted = np.ones(6)
    mae_original = mae_shifted + log_ratio
    corrected = np.array([False, False, False, True, True, True])

    result = correlate_observability_with_mae_asymmetry(
        sample_mean_log_ratio=log_ratio,
        normal_angles_deg=normal_angles,
        mae_original_mm=mae_original,
        mae_shifted_mm=mae_shifted,
        correction_triggered=corrected,
    )

    assert result["all"]["n"] == 6
    assert result["all"]["spearman_log_ratio_vs_original_minus_shifted_mae"] == pytest.approx(1.0)
    assert result["correction"]["n"] == 3
    assert result["no_correction"]["n"] == 3
    assert result["all"]["mean_original_minus_shifted_mae_mm"] == pytest.approx(2.5)
    assert result["all"]["spearman_mean_normal_angle_vs_asymmetry"] == pytest.approx(1.0)
    assert result["all"]["spearman_max_normal_angle_vs_asymmetry"] == pytest.approx(1.0)


def test_nonfinite_geodesic_pairs_are_excluded_without_invalid_subtraction():
    with np.errstate(invalid="raise"):
        result = correlate_observability_with_mae_asymmetry(
            sample_mean_log_ratio=np.array([0.0, 1.0, 2.0]),
            normal_angles_deg=np.zeros((3, 3)),
            mae_original_mm=np.array([np.inf, 2.0, 4.0]),
            mae_shifted_mm=np.array([np.inf, 1.0, 2.0]),
            correction_triggered=np.array([False, False, True]),
        )

    assert result["all"]["n_finite_asymmetry"] == 2
    assert result["all"]["mean_original_minus_shifted_mae_mm"] == pytest.approx(1.5)
