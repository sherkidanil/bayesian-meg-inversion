#!/usr/bin/env python3
"""Paired post-Task-A audit using already saved truth-free predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from cnn_shift_stability_paired import (
    align_prediction_to_truth,
    make_euclidean_dist_fn,
    make_geodesic_dist_fn,
    paired_shift_distances,
    summarize_distribution,
)


PAIR_INDICES = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)


def no_cross_component_pair_mask(
    *,
    truth_original: np.ndarray,
    truth_shifted: np.ndarray,
    aligned_original: np.ndarray,
    aligned_shifted: np.ndarray,
    graph: csr_matrix,
) -> np.ndarray:
    """Keep pairs whose aligned predictions stay in each truth's mesh component."""
    arrays = [
        np.asarray(value, dtype=np.int64)
        for value in (
            truth_original,
            truth_shifted,
            aligned_original,
            aligned_shifted,
        )
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("truth and aligned prediction arrays must have equal shape")
    if arrays[0].ndim != 2:
        raise ValueError("truth and aligned prediction arrays must be two-dimensional")
    _n_components, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    truth_a, truth_b, prediction_a, prediction_b = arrays
    original_clean = labels[truth_a] == labels[prediction_a]
    shifted_clean = labels[truth_b] == labels[prediction_b]
    return np.all(original_clean & shifted_clean, axis=1)


def pairwise_source_distances_mm(
    centroids: np.ndarray, vertices_m: np.ndarray
) -> np.ndarray:
    centroids = np.asarray(centroids, dtype=np.int64)
    vertices = np.asarray(vertices_m, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected centroid shape (N, 3), got {centroids.shape}")
    coords = vertices[centroids]
    return np.stack(
        [
            np.linalg.norm(coords[:, left] - coords[:, right], axis=1) * 1000.0
            for left, right in PAIR_INDICES
        ],
        axis=1,
    )


def infer_neighbor_correction(
    original_centroids: np.ndarray,
    shifted_centroids: np.ndarray,
    vertices_m: np.ndarray,
    *,
    delta_m: np.ndarray,
    min_dist_m: float,
) -> dict[str, np.ndarray]:
    """Replay the generator's simple snap and identify the min-distance branch."""
    original = np.asarray(original_centroids, dtype=np.int64)
    shifted = np.asarray(shifted_centroids, dtype=np.int64)
    vertices = np.asarray(vertices_m, dtype=np.float64)
    delta = np.asarray(delta_m, dtype=np.float64)
    if original.shape != shifted.shape or original.ndim != 2 or original.shape[1] != 3:
        raise ValueError("Original and shifted centroids must both have shape (N, 3)")
    if delta.shape != (3,):
        raise ValueError(f"Expected one 3-D shift vector, got {delta.shape}")

    targets = vertices[original] + delta[None, None, :]
    _distance, simple_flat = cKDTree(vertices).query(targets.reshape(-1, 3), k=1)
    simple = np.asarray(simple_flat, dtype=np.int64).reshape(original.shape)
    simple_pair_mm = pairwise_source_distances_mm(simple, vertices)
    simple_min_mm = np.min(simple_pair_mm, axis=1)
    return {
        "simple_centroids": simple,
        "simple_pairwise_distance_mm": simple_pair_mm,
        "simple_min_distance_mm": simple_min_mm,
        "triggered_by_rule": simple_min_mm < float(min_dist_m) * 1000.0,
        "changed_from_simple": np.any(simple != shifted, axis=1),
    }


def _graph_from_edges(
    edge_index: np.ndarray, edge_weight_mm: np.ndarray, n_vertices: int
) -> csr_matrix:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError(f"edge_index must be 2-D, got {edges.shape}")
    if edges.shape[0] == 2:
        source, target = edges
    elif edges.shape[1] == 2:
        source, target = edges.T
    else:
        raise ValueError(f"Cannot interpret edge_index shape {edges.shape}")
    weights = np.asarray(edge_weight_mm, dtype=np.float64).reshape(-1)
    if weights.size != source.size:
        raise ValueError("edge_index and edge_weight_mm lengths differ")
    graph = csr_matrix((weights, (source, target)), shape=(n_vertices, n_vertices))
    return graph.maximum(graph.T).tocsr()


def compute_paired_deltax(
    *,
    truth_original: np.ndarray,
    truth_shifted: np.ndarray,
    prediction_original: np.ndarray,
    prediction_shifted: np.ndarray,
    vertices_m: np.ndarray,
    graph: csr_matrix,
) -> dict[str, np.ndarray]:
    truth_a = np.asarray(truth_original, dtype=np.int64)
    truth_b = np.asarray(truth_shifted, dtype=np.int64)
    pred_a = np.asarray(prediction_original, dtype=np.int64)
    pred_b = np.asarray(prediction_shifted, dtype=np.int64)
    if not (truth_a.shape == truth_b.shape == pred_a.shape == pred_b.shape):
        raise ValueError("All truth and prediction arrays must have the same shape")

    n_samples = truth_a.shape[0]
    aligned_a = np.empty_like(pred_a)
    aligned_b = np.empty_like(pred_b)
    euc_mm = np.empty_like(pred_a, dtype=np.float64)
    geo_mm = np.empty_like(pred_a, dtype=np.float64)
    assignment_a = np.empty(n_samples, dtype=np.float64)
    assignment_b = np.empty(n_samples, dtype=np.float64)
    geo_assignment_cache: dict[int, np.ndarray] = {}
    geo_pair_cache: dict[int, np.ndarray] = {}
    geo_dist = make_geodesic_dist_fn(graph, geo_assignment_cache)
    euc_dist = make_euclidean_dist_fn(vertices_m)
    for sample in range(n_samples):
        aligned_a[sample], _perm_a, assignment_a[sample] = align_prediction_to_truth(
            truth_a[sample], pred_a[sample], geo_dist, fallback_dist_fn=euc_dist
        )
        aligned_b[sample], _perm_b, assignment_b[sample] = align_prediction_to_truth(
            truth_b[sample], pred_b[sample], geo_dist, fallback_dist_fn=euc_dist
        )
        euc_mm[sample], geo_mm[sample] = paired_shift_distances(
            aligned_a[sample], aligned_b[sample], vertices_m, graph, geo_pair_cache
        )
    return {
        "aligned_original": aligned_a,
        "aligned_shifted": aligned_b,
        "euc_mm": euc_mm,
        "geo_mm": geo_mm,
        "assignment_original_geo_mm": assignment_a,
        "assignment_shifted_geo_mm": assignment_b,
    }


def summarize_masked(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size != mask.size:
        raise ValueError("values and mask lengths differ")
    summary = summarize_distribution(values[mask])
    return {"n_selected": int(np.sum(mask)), **summary}


def read_estimator_metrics(
    path: Path, n_samples: int, estimator: str
) -> dict[str, np.ndarray]:
    output = {
        f"{kind}_{metric}": np.full(n_samples, np.nan, dtype=np.float64)
        for kind in ("original", "shifted")
        for metric in ("mae_euc_mm", "mae_geo_mm")
    }
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["estimator"] != estimator:
                continue
            sample = int(row["sample_index"])
            kind = row["input_kind"]
            for metric in ("mae_euc_mm", "mae_geo_mm"):
                output[f"{kind}_{metric}"][sample] = float(row[metric])
    for name, values in output.items():
        if np.any(np.isnan(values)):
            raise ValueError(f"Missing {estimator} rows for {name}")
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(summary: dict[str, Any]) -> str:
    return (
        f"{summary['mean']:.4f} (median {summary['median']:.4f}; "
        f"n={summary['n']}, nonfinite={summary['n_nonfinite']})"
    )


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    subsets = summary["estimator_mae_by_subset"]
    estimator = summary["estimator"]
    lines = [
        f"# Post-Task-A paired audit: {summary['scenario']}",
        "",
        f"## Paired ΔX′ for `{estimator}`",
        "",
        f"- Euclidean, per-sample mean: {_fmt(summary['deltax_euc_mm_by_sample'])} mm",
        f"- Geodesic, per-sample mean: {_fmt(summary['deltax_geo_mm_by_sample'])} mm",
        f"- Euclidean on all pairs: {_fmt(summary['deltax_subsets']['all']['euc_mm'])} mm",
        f"- Euclidean without this method's cross-component misses: {_fmt(summary['deltax_subsets']['no_cross_component']['euc_mm'])} mm",
        f"- Cross-component pairs excluded only from the symmetric subset: {summary['deltax_subsets']['n_cross_component_excluded']}/{summary['n_samples']}.",
        f"- Previous posterior-mean Euclidean ΔX′: {summary['previous_deltax_euc_mm']} mm",
        "- Predictions were aligned independently to the corresponding original and shifted truths; truth is used only to define the controlled paired stability metric.",
        "",
        "## Shift-construction asymmetry",
        "",
        f"- Neighbor-correction branch inferred: {summary['neighbor_correction']['n_triggered']}/{summary['n_samples']} ({100.0 * summary['neighbor_correction']['fraction_triggered']:.2f}%).",
        f"- Actual shifted centroids differ from the simple nearest snap: {summary['neighbor_correction']['n_changed_from_simple']}/{summary['n_samples']}.",
        f"- Original minimum pair spacing: {_fmt(summary['source_spacing']['original_min_mm'])} mm",
        f"- Shifted minimum pair spacing: {_fmt(summary['source_spacing']['shifted_min_mm'])} mm",
        f"- Shifted minus original minimum spacing: {_fmt(summary['source_spacing']['shifted_minus_original_min_mm'])} mm",
        "",
        f"## `{estimator}` localization MAE",
        "",
        "| subset | input | MAE euc, mm | MAE geo, mm |",
        "|---|---|---:|---:|",
    ]
    for subset in ("all", "no_correction", "correction"):
        for kind in ("original", "shifted"):
            lines.append(
                f"| {subset} | {kind} | {_fmt(subsets[subset][kind]['mae_euc_mm'])} | {_fmt(subsets[subset][kind]['mae_geo_mm'])} |"
            )
    lines.extend(
        [
            "",
            "The correction split diagnoses whether ΔX′ mixes sensitivity to the input shift with a systematic change in source separability. It does not alter the estimator.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario_npz", required=True)
    parser.add_argument("--selected_predictions_npz", required=True)
    parser.add_argument("--task_a_per_sample_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scenario", default="unknown")
    parser.add_argument("--estimator", default="residual_selected")
    parser.add_argument("--previous_deltax_euc_mm", type=float, default=float("nan"))
    parser.add_argument("--shift_mm", nargs=3, type=float, default=(3.0, 3.0, 1.0))
    parser.add_argument("--min_dist_mm", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    scenario_path = Path(args.scenario_npz).resolve()
    selected_path = Path(args.selected_predictions_npz).resolve()
    metric_path = Path(args.task_a_per_sample_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(scenario_path, allow_pickle=True)
    selected = np.load(selected_path, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    truth_a = np.asarray(data["centroids_idx"], dtype=np.int64)
    truth_b = np.asarray(data["centroids_idx_shifted"], dtype=np.int64)
    sample_indices = np.asarray(selected["sample_indices"], dtype=np.int64)
    n_samples = len(sample_indices)
    if not np.array_equal(sample_indices, np.arange(n_samples)):
        raise ValueError("This audit requires contiguous sample indices starting at zero")
    truth_a = truth_a[sample_indices]
    truth_b = truth_b[sample_indices]
    kinds = [str(value) for value in selected["input_kinds"]]
    if kinds != ["original", "shifted"]:
        raise ValueError(f"Unexpected selected prediction order: {kinds}")
    predictions = np.asarray(selected["centroids_selected"], dtype=np.int64)
    if predictions.shape != (n_samples, 2, 3):
        raise ValueError(f"Unexpected prediction shape {predictions.shape}")

    graph = _graph_from_edges(data["edge_index"], data["edge_weight_mm"], len(vertices))
    deltax = compute_paired_deltax(
        truth_original=truth_a,
        truth_shifted=truth_b,
        prediction_original=predictions[:, 0],
        prediction_shifted=predictions[:, 1],
        vertices_m=vertices,
        graph=graph,
    )
    no_cross_component = no_cross_component_pair_mask(
        truth_original=truth_a,
        truth_shifted=truth_b,
        aligned_original=deltax["aligned_original"],
        aligned_shifted=deltax["aligned_shifted"],
        graph=graph,
    )
    deltax_euc_by_sample = np.mean(deltax["euc_mm"], axis=1)
    deltax_geo_by_sample = np.mean(deltax["geo_mm"], axis=1)
    correction = infer_neighbor_correction(
        truth_a,
        truth_b,
        vertices,
        delta_m=np.asarray(args.shift_mm, dtype=np.float64) / 1000.0,
        min_dist_m=float(args.min_dist_mm) / 1000.0,
    )
    spacing_a = pairwise_source_distances_mm(truth_a, vertices)
    spacing_b = pairwise_source_distances_mm(truth_b, vertices)
    metrics = read_estimator_metrics(metric_path, n_samples, args.estimator)
    corrected = correction["triggered_by_rule"]
    masks = {
        "all": np.ones(n_samples, dtype=bool),
        "no_correction": ~corrected,
        "correction": corrected,
    }
    subset_summary: dict[str, Any] = {}
    for subset, mask in masks.items():
        subset_summary[subset] = {}
        for kind in ("original", "shifted"):
            subset_summary[subset][kind] = {
                metric: summarize_masked(metrics[f"{kind}_{metric}"], mask)
                for metric in ("mae_euc_mm", "mae_geo_mm")
            }

    summary: dict[str, Any] = {
        "scenario": args.scenario,
        "estimator": args.estimator,
        "scenario_npz": str(scenario_path),
        "selected_predictions_npz": str(selected_path),
        "task_a_per_sample_csv": str(metric_path),
        "n_samples": n_samples,
        "previous_deltax_euc_mm": float(args.previous_deltax_euc_mm),
        "paired_alignment": "truth-anchored independently within original and shifted inputs",
        "deltax_euc_mm_by_sample": summarize_distribution(deltax_euc_by_sample),
        "deltax_euc_mm_by_source": summarize_distribution(deltax["euc_mm"].reshape(-1)),
        "deltax_geo_mm_by_sample": summarize_distribution(deltax_geo_by_sample),
        "deltax_geo_mm_by_source": summarize_distribution(deltax["geo_mm"].reshape(-1)),
        "deltax_subsets": {
            "all": {
                "euc_mm": summarize_masked(
                    deltax_euc_by_sample, np.ones(n_samples, dtype=bool)
                ),
                "geo_mm": summarize_masked(
                    deltax_geo_by_sample, np.ones(n_samples, dtype=bool)
                ),
            },
            "no_cross_component": {
                "euc_mm": summarize_masked(deltax_euc_by_sample, no_cross_component),
                "geo_mm": summarize_masked(deltax_geo_by_sample, no_cross_component),
            },
            "n_cross_component_excluded": int(np.sum(~no_cross_component)),
        },
        "neighbor_correction": {
            "shift_mm": [float(x) for x in args.shift_mm],
            "min_dist_mm": float(args.min_dist_mm),
            "n_triggered": int(np.sum(corrected)),
            "fraction_triggered": float(np.mean(corrected)),
            "n_changed_from_simple": int(np.sum(correction["changed_from_simple"])),
            "rule_changed_disagreement": int(np.sum(corrected != correction["changed_from_simple"])),
        },
        "source_spacing": {
            "original_all_pairs_mm": summarize_distribution(spacing_a.reshape(-1)),
            "shifted_all_pairs_mm": summarize_distribution(spacing_b.reshape(-1)),
            "original_min_mm": summarize_distribution(np.min(spacing_a, axis=1)),
            "shifted_min_mm": summarize_distribution(np.min(spacing_b, axis=1)),
            "shifted_minus_original_min_mm": summarize_distribution(
                np.min(spacing_b, axis=1) - np.min(spacing_a, axis=1)
            ),
        },
        "estimator_mae_by_subset": subset_summary,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    for sample in range(n_samples):
        row: dict[str, Any] = {
            "sample_index": sample,
            "neighbor_correction_triggered": bool(corrected[sample]),
            "changed_from_simple_snap": bool(correction["changed_from_simple"][sample]),
            "no_cross_component_miss": bool(no_cross_component[sample]),
            "simple_snap_min_spacing_mm": float(correction["simple_min_distance_mm"][sample]),
            "original_min_spacing_mm": float(np.min(spacing_a[sample])),
            "shifted_min_spacing_mm": float(np.min(spacing_b[sample])),
            "deltax_euc_mean_mm": float(np.mean(deltax["euc_mm"][sample])),
            "deltax_geo_mean_mm": float(np.mean(deltax["geo_mm"][sample])),
        }
        for source in range(3):
            row[f"deltax_euc_source{source}_mm"] = float(deltax["euc_mm"][sample, source])
            row[f"deltax_geo_source{source}_mm"] = float(deltax["geo_mm"][sample, source])
        for name, values in metrics.items():
            row[name] = float(values[sample])
        rows.append(row)
    _write_csv(out_dir / "per_sample_paired.csv", rows)
    np.savez_compressed(
        out_dir / "paired_arrays.npz",
        sample_indices=sample_indices,
        aligned_original=deltax["aligned_original"],
        aligned_shifted=deltax["aligned_shifted"],
        deltax_euc_mm=deltax["euc_mm"],
        deltax_geo_mm=deltax["geo_mm"],
        no_cross_component_mask=no_cross_component,
        neighbor_correction_triggered=corrected,
        simple_centroids=correction["simple_centroids"],
        source_spacing_original_mm=spacing_a,
        source_spacing_shifted_mm=spacing_b,
    )
    _write_report(out_dir / "report.md", summary)
    print(json.dumps({
        "out_dir": str(out_dir),
        "deltax_euc_mm": summary["deltax_euc_mm_by_sample"]["mean"],
        "deltax_geo_mm": summary["deltax_geo_mm_by_sample"]["mean"],
        "correction_fraction": summary["neighbor_correction"]["fraction_triggered"],
    }, indent=2))


if __name__ == "__main__":
    main()
