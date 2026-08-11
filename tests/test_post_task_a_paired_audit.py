from pathlib import Path
import csv
import sys

import numpy as np
from scipy.sparse import csr_matrix

TEST_ROOT = Path(__file__).resolve().parent
if TEST_ROOT.name == "tests":
    TEST_ROOT = TEST_ROOT.parent
sys.path.insert(0, str(TEST_ROOT))

from post_task_a_paired_audit import (
    compute_paired_deltax,
    read_estimator_metrics,
    infer_neighbor_correction,
    pairwise_source_distances_mm,
    parse_args,
    summarize_masked,
)


def test_pairwise_source_distances_reports_all_three_pairs_in_mm():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.0, 0.004, 0.0]]
    )

    distances = pairwise_source_distances_mm(np.array([[0, 1, 2]]), vertices)

    assert distances.shape == (1, 3)
    assert np.allclose(distances[0], [3.0, 4.0, 5.0])


def test_neighbor_correction_is_inferred_from_the_generators_simple_snap_rule():
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
    original = np.array([[0, 1, 2]])
    actual_shifted = np.array([[3, 6, 5]])

    result = infer_neighbor_correction(
        original,
        actual_shifted,
        vertices,
        delta_m=np.array([0.003, 0.0, 0.0]),
        min_dist_m=0.030,
    )

    assert np.array_equal(result["simple_centroids"], [[3, 1, 5]])
    assert result["triggered_by_rule"].tolist() == [True]
    assert result["changed_from_simple"].tolist() == [True]
    assert np.allclose(result["simple_min_distance_mm"], [29.0])


def test_paired_deltax_aligns_each_prediction_to_its_own_truth_before_pairing():
    vertices = np.array(
        [[0.000, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0],
         [0.003, 0.0, 0.0], [0.004, 0.0, 0.0], [0.005, 0.0, 0.0]]
    )
    graph = csr_matrix(
        np.diag(np.ones(5), 1) + np.diag(np.ones(5), -1)
    )
    result = compute_paired_deltax(
        truth_original=np.array([[0, 2, 4]]),
        truth_shifted=np.array([[1, 3, 5]]),
        prediction_original=np.array([[4, 0, 2]]),
        prediction_shifted=np.array([[3, 5, 1]]),
        vertices_m=vertices,
        graph=graph,
    )

    assert np.array_equal(result["aligned_original"], [[0, 2, 4]])
    assert np.array_equal(result["aligned_shifted"], [[1, 3, 5]])
    assert np.allclose(result["euc_mm"], [[1.0, 1.0, 1.0]])
    assert np.allclose(result["geo_mm"], [[1.0, 1.0, 1.0]])


def test_summarize_masked_keeps_subset_size_and_nonfinite_count():
    summary = summarize_masked(
        np.array([1.0, np.nan, 5.0]), np.array([True, True, False])
    )

    assert summary["n_selected"] == 2
    assert summary["n"] == 1
    assert summary["n_nonfinite"] == 1
    assert summary["mean"] == 1.0


def test_previous_primary_deltax_is_explicitly_euclidean():
    args = parse_args(
        [
            "--scenario_npz", "scenario.npz",
            "--selected_predictions_npz", "selected.npz",
            "--task_a_per_sample_csv", "metrics.csv",
            "--out_dir", "out",
            "--previous_deltax_euc_mm", "6.205",
        ]
    )

    assert args.previous_deltax_euc_mm == 6.205


def test_metric_reader_selects_the_requested_estimator(tmp_path):
    path = tmp_path / "metrics.csv"
    rows = []
    for estimator, value in (("residual_selected", 1.0), ("frozen_smooth", 2.0)):
        for sample in range(2):
            for kind in ("original", "shifted"):
                rows.append(
                    {
                        "estimator": estimator,
                        "sample_index": sample,
                        "input_kind": kind,
                        "mae_euc_mm": value + sample,
                        "mae_geo_mm": value + sample + 0.5,
                    }
                )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metrics = read_estimator_metrics(path, 2, "frozen_smooth")

    assert metrics["original_mae_euc_mm"].tolist() == [2.0, 3.0]
    assert metrics["shifted_mae_geo_mm"].tolist() == [2.5, 3.5]
