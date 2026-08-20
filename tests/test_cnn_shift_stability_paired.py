from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnn_shift_stability_paired import (
    aggregate_theta_draws,
    align_prediction_to_truth,
    parse_args,
    paired_shift_distances,
    safe_ratio,
    summarize_distribution,
)


def test_estimator_flag_and_dispatch_support_truth_free_medoid():
    args = parse_args(
        ["--dataset_npz", "paired.npz", "--estimator", "posterior_medoid"]
    )
    assert args.estimator == "posterior_medoid"

    draws = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    naive = aggregate_theta_draws(draws, "posterior_mean_naive")
    medoid = aggregate_theta_draws(draws, "posterior_medoid")

    assert np.allclose(naive.reshape(3, 3)[:, 0], [1.0, 1.0, 1.0])
    assert np.array_equal(medoid, draws[0].reshape(-1))


def test_align_prediction_to_truth_uses_true_source_order_as_anchor():
    vertices_m = np.array(
        [
            [0.000, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.003, 0.0, 0.0],
        ]
    )

    def euclid(a, b):
        return float(np.linalg.norm(vertices_m[a] - vertices_m[b]) * 1000.0)

    c_true_a = np.array([0, 1, 2])
    c_pred_a = np.array([2, 0, 1])
    c_true_b = np.array([1, 2, 3])
    c_pred_b = np.array([3, 1, 2])

    aligned_a, perm_a, mean_a = align_prediction_to_truth(c_true_a, c_pred_a, euclid)
    aligned_b, perm_b, mean_b = align_prediction_to_truth(c_true_b, c_pred_b, euclid)

    assert tuple(perm_a) == (1, 2, 0)
    assert tuple(perm_b) == (1, 2, 0)
    assert np.array_equal(aligned_a, [0, 1, 2])
    assert np.array_equal(aligned_b, [1, 2, 3])
    assert mean_a == 0.0
    assert mean_b == 0.0


def test_paired_shift_distances_reports_euclidean_and_mesh_geodesic_mm():
    vertices_m = np.array(
        [
            [0.000, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.002, 0.0, 0.0],
            [0.003, 0.0, 0.0],
        ]
    )
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

    euc_mm, geo_mm = paired_shift_distances(
        np.array([0, 1, 2]),
        np.array([1, 2, 3]),
        vertices_m,
        graph,
    )

    assert np.allclose(euc_mm, [1.0, 1.0, 1.0])
    assert np.allclose(geo_mm, [1.0, 1.0, 1.0])


def test_safe_ratio_marks_zero_input_shift_as_undefined():
    assert safe_ratio(6.0, 3.0) == 2.0
    assert np.isnan(safe_ratio(0.0, 0.0))
    assert np.isnan(safe_ratio(2.0, 0.0))


def test_summarize_distribution_ignores_nonfinite_values_and_reports_tail():
    summary = summarize_distribution([1.0, np.nan, 3.0, np.inf])

    assert summary == {
        "n": 2,
        "n_nonfinite": 2,
        "mean": 2.0,
        "median": 2.0,
        "std": 1.0,
        "p25": 1.5,
        "p75": 2.5,
        "p90": 2.8,
        "max": 3.0,
    }
