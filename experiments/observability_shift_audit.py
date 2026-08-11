#!/usr/bin/env python3
"""Audit whether a fixed centroid shift changes lead-field observability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


def compute_observability_shift(
    leadfield: np.ndarray,
    original_centroids: np.ndarray,
    shifted_centroids: np.ndarray,
) -> dict[str, np.ndarray]:
    g = np.asarray(leadfield, dtype=np.float64)
    original = np.asarray(original_centroids, dtype=np.int64)
    shifted = np.asarray(shifted_centroids, dtype=np.int64)
    if g.ndim != 2 or not np.all(np.isfinite(g)):
        raise ValueError("leadfield must be a finite two-dimensional matrix")
    if original.shape != shifted.shape or original.ndim != 2:
        raise ValueError("centroid arrays must have equal shape (samples, sources)")
    if np.any(original < 0) or np.any(shifted < 0) or np.any(original >= g.shape[1]) or np.any(shifted >= g.shape[1]):
        raise ValueError("centroid index is outside the lead field")
    column_norm = np.linalg.norm(g, axis=0)
    original_norm = column_norm[original]
    shifted_norm = column_norm[shifted]
    if np.any(original_norm <= 0.0) or np.any(shifted_norm <= 0.0):
        raise ValueError("observability ratio is undefined for a zero lead-field column")
    source_log_ratio = np.log(shifted_norm / original_norm)
    return {
        "original_column_norm": original_norm,
        "shifted_column_norm": shifted_norm,
        "source_log_ratio": source_log_ratio,
        "sample_mean_log_ratio": np.mean(source_log_ratio, axis=1),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64).reshape(-1)
    second = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    if np.sum(valid) < 2 or np.std(first[valid]) == 0.0 or np.std(second[valid]) == 0.0:
        return float("nan")
    return float(spearmanr(first[valid], second[valid]).statistic)


def correlate_observability_with_mae_asymmetry(
    *,
    sample_mean_log_ratio: np.ndarray,
    normal_angles_deg: np.ndarray,
    mae_original_mm: np.ndarray,
    mae_shifted_mm: np.ndarray,
    correction_triggered: np.ndarray,
) -> dict[str, dict[str, Any]]:
    log_ratio = np.asarray(sample_mean_log_ratio, dtype=np.float64).reshape(-1)
    angles = np.asarray(normal_angles_deg, dtype=np.float64)
    original = np.asarray(mae_original_mm, dtype=np.float64).reshape(-1)
    shifted = np.asarray(mae_shifted_mm, dtype=np.float64).reshape(-1)
    corrected = np.asarray(correction_triggered, dtype=bool).reshape(-1)
    if angles.ndim != 2 or not (
        len(log_ratio) == len(angles) == len(original) == len(shifted) == len(corrected)
    ):
        raise ValueError("observability, angles, MAE, and correction arrays do not align")
    paired_finite = np.isfinite(original) & np.isfinite(shifted)
    asymmetry = np.full_like(original, np.nan)
    asymmetry[paired_finite] = original[paired_finite] - shifted[paired_finite]
    mean_angle = np.mean(angles, axis=1)
    max_angle = np.max(angles, axis=1)
    masks = {
        "all": np.ones(len(log_ratio), dtype=bool),
        "correction": corrected,
        "no_correction": ~corrected,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        valid_asymmetry = asymmetry[mask]
        finite_asymmetry = valid_asymmetry[np.isfinite(valid_asymmetry)]
        result[name] = {
            "n": int(np.sum(mask)),
            "n_finite_asymmetry": int(len(finite_asymmetry)),
            "mean_original_minus_shifted_mae_mm": (
                float(np.mean(finite_asymmetry)) if finite_asymmetry.size else float("nan")
            ),
            "spearman_log_ratio_vs_original_minus_shifted_mae": _spearman(
                log_ratio[mask], valid_asymmetry
            ),
            "spearman_mean_normal_angle_vs_asymmetry": _spearman(
                mean_angle[mask], valid_asymmetry
            ),
            "spearman_max_normal_angle_vs_asymmetry": _spearman(
                max_angle[mask], valid_asymmetry
            ),
        }
    return result


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    return {
        "n": int(raw.size),
        "n_finite": int(finite.size),
        "n_nonfinite": int(raw.size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "p10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
        "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _correction_mask_from_simple_snap(
    original: np.ndarray, vertices: np.ndarray, *, min_dist_mm: float = 30.0
) -> np.ndarray:
    targets = vertices[np.asarray(original, dtype=np.int64)] + np.array(
        [0.003, 0.003, 0.001]
    )
    simple = cKDTree(vertices).query(targets.reshape(-1, 3), k=1)[1].reshape(
        np.asarray(original).shape
    )
    coords = vertices[simple]
    distances = np.stack(
        [
            np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1),
            np.linalg.norm(coords[:, 0] - coords[:, 2], axis=1),
            np.linalg.norm(coords[:, 1] - coords[:, 2], axis=1),
        ],
        axis=1,
    )
    return np.min(distances, axis=1) < float(min_dist_mm) / 1000.0


def _heldout_metric_groups(path: Path, n_samples: int) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    groups: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["candidate_id"], row["domain"])
            if key not in groups:
                groups[key] = {
                    metric: np.full(n_samples, np.nan, dtype=np.float64)
                    for metric in (
                        "mae_euc_original_mm",
                        "mae_euc_shifted_mm",
                        "mae_geo_original_mm",
                        "mae_geo_shifted_mm",
                    )
                }
            index = int(row["heldout_sample_index"])
            for metric in groups[key]:
                groups[key][metric][index] = float(row[metric])
    for key, metrics in groups.items():
        for metric, values in metrics.items():
            if np.any(np.isnan(values)):
                raise ValueError(f"missing held-out rows for {key} {metric}")
    return groups


def _anna_residual_selected_metrics(path: Path, n_samples: int) -> dict[str, np.ndarray]:
    output = {
        metric: np.full((n_samples, 2), np.nan, dtype=np.float64)
        for metric in ("mae_euc_mm", "mae_geo_mm")
    }
    kinds = {"original": 0, "shifted": 1}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("estimator") != "residual_selected":
                continue
            sample = int(row["sample_index"])
            kind = kinds[row["input_kind"]]
            for metric in output:
                output[metric][sample, kind] = float(row[metric])
    for metric, values in output.items():
        if np.any(np.isnan(values)):
            raise ValueError(f"missing Anna residual-selected rows for {metric}")
    return output


def _observability_summary(result: dict[str, np.ndarray]) -> dict[str, Any]:
    source = result["source_log_ratio"]
    return {
        "source_log_ratio": {
            f"source_{index}": _distribution(source[:, index])
            for index in range(source.shape[1])
        },
        "all_sources_log_ratio": _distribution(source),
        "sample_mean_log_ratio": _distribution(result["sample_mean_log_ratio"]),
        "fraction_samples_positive_mean_log_ratio": float(
            np.mean(result["sample_mean_log_ratio"] > 0.0)
        ),
        "original_column_norm": _distribution(result["original_column_norm"]),
        "shifted_column_norm": _distribution(result["shifted_column_norm"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-heldout", required=True)
    parser.add_argument("--heldout-per-sample", required=True)
    parser.add_argument("--anna-dg0", required=True)
    parser.add_argument("--anna-task-a-per-sample", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    module_root = Path(__file__).resolve().parent
    repository = module_root.parent
    start_git = _git_state(repository)
    if start_git["dirty"]:
        raise RuntimeError("observability audit requires a clean repository")
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    matched_path = Path(args.matched_heldout).resolve()
    heldout_metrics_path = Path(args.heldout_per_sample).resolve()
    anna_path = Path(args.anna_dg0).resolve()
    anna_metrics_path = Path(args.anna_task_a_per_sample).resolve()
    with np.load(matched_path, allow_pickle=True) as payload:
        matched = {
            key: np.array(payload[key])
            for key in (
                "centroids_idx",
                "centroids_idx_shifted",
                "normal_angles_deg",
                "correction_triggered",
                "G_rebuilt",
                "G_anna",
            )
        }
    n_matched = len(matched["centroids_idx"])
    heldout_groups = _heldout_metric_groups(heldout_metrics_path, n_matched)
    summary: dict[str, Any] = {
        "matched_heldout": {"n_samples": n_matched, "domains": {}},
        "existing_anna_dg0": {},
    }
    for domain, g_key in (("rebuilt", "G_rebuilt"), ("anna", "G_anna")):
        observability = compute_observability_shift(
            matched[g_key], matched["centroids_idx"], matched["centroids_idx_shifted"]
        )
        domain_summary: dict[str, Any] = {
            "observability": _observability_summary(observability),
            "estimators": {},
        }
        for (candidate, row_domain), metrics in heldout_groups.items():
            if row_domain != domain:
                continue
            domain_summary["estimators"][candidate] = {
                "euclidean": correlate_observability_with_mae_asymmetry(
                    sample_mean_log_ratio=observability["sample_mean_log_ratio"],
                    normal_angles_deg=matched["normal_angles_deg"],
                    mae_original_mm=metrics["mae_euc_original_mm"],
                    mae_shifted_mm=metrics["mae_euc_shifted_mm"],
                    correction_triggered=matched["correction_triggered"],
                ),
                "geodesic": correlate_observability_with_mae_asymmetry(
                    sample_mean_log_ratio=observability["sample_mean_log_ratio"],
                    normal_angles_deg=matched["normal_angles_deg"],
                    mae_original_mm=metrics["mae_geo_original_mm"],
                    mae_shifted_mm=metrics["mae_geo_shifted_mm"],
                    correction_triggered=matched["correction_triggered"],
                ),
            }
        summary["matched_heldout"]["domains"][domain] = domain_summary
    with np.load(anna_path, allow_pickle=True) as payload:
        anna_original = np.asarray(payload["centroids_idx"], dtype=np.int64)
        anna_shifted = np.asarray(payload["centroids_idx_shifted"], dtype=np.int64)
        anna_vertices = np.asarray(payload["vertices"], dtype=np.float64)
        anna_angles = np.asarray(payload["normal_angles_deg"], dtype=np.float64)
        matrices = np.asarray(payload["G_for_simulation"])
        anna_g = np.asarray(matrices[0], dtype=np.float64)
        if not np.array_equal(matrices, np.broadcast_to(anna_g, matrices.shape)):
            raise ValueError("Anna dG0 lead field is not constant")
    anna_observability = compute_observability_shift(anna_g, anna_original, anna_shifted)
    anna_correction = _correction_mask_from_simple_snap(anna_original, anna_vertices)
    anna_metrics = _anna_residual_selected_metrics(anna_metrics_path, len(anna_original))
    summary["existing_anna_dg0"] = {
        "n_samples": int(len(anna_original)),
        "estimator": "residual_selected (historical push_3771 Task A)",
        "observability": _observability_summary(anna_observability),
        "euclidean": correlate_observability_with_mae_asymmetry(
            sample_mean_log_ratio=anna_observability["sample_mean_log_ratio"],
            normal_angles_deg=anna_angles,
            mae_original_mm=anna_metrics["mae_euc_mm"][:, 0],
            mae_shifted_mm=anna_metrics["mae_euc_mm"][:, 1],
            correction_triggered=anna_correction,
        ),
        "geodesic": correlate_observability_with_mae_asymmetry(
            sample_mean_log_ratio=anna_observability["sample_mean_log_ratio"],
            normal_angles_deg=anna_angles,
            mae_original_mm=anna_metrics["mae_geo_mm"][:, 0],
            mae_shifted_mm=anna_metrics["mae_geo_mm"][:, 1],
            correction_triggered=anna_correction,
        ),
    }
    end_git = _git_state(repository)
    if start_git != end_git:
        raise RuntimeError("repository changed during observability audit")
    summary.update(
        {
            "inputs": {
                "matched_heldout": {"path": str(matched_path), "sha256": _sha256_file(matched_path)},
                "heldout_per_sample": {"path": str(heldout_metrics_path), "sha256": _sha256_file(heldout_metrics_path)},
                "anna_dg0": {"path": str(anna_path), "sha256": _sha256_file(anna_path)},
                "anna_task_a_per_sample": {"path": str(anna_metrics_path), "sha256": _sha256_file(anna_metrics_path)},
            },
            "run_start_git": start_git,
            "artifact_write_git": end_git,
            "repository_changed_during_run": False,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# Lead-field observability under the fixed centroid shift",
        "",
        "Positive `log(||G_shifted|| / ||G_original||)` means the shifted centroid has greater lead-field column norm.",
        "MAE asymmetry is `original - shifted`; positive values mean the shifted input is easier.",
        "",
        "## Matched held-out",
        "",
        "| domain | mean log norm ratio | positive samples | estimator | metric | MAE asymmetry | rho(log ratio) | rho(mean angle) | corrected asymmetry |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for domain, domain_summary in summary["matched_heldout"]["domains"].items():
        obs = domain_summary["observability"]
        for estimator, metrics in domain_summary["estimators"].items():
            for metric, subsets in metrics.items():
                all_values = subsets["all"]
                corrected = subsets["correction"]
                report.append(
                    f"| {domain} | {obs['sample_mean_log_ratio']['mean']:+.5f} | "
                    f"{100.0 * obs['fraction_samples_positive_mean_log_ratio']:.1f}% | "
                    f"`{estimator}` | {metric} | "
                    f"{all_values['mean_original_minus_shifted_mae_mm']:+.4f} | "
                    f"{all_values['spearman_log_ratio_vs_original_minus_shifted_mae']:+.3f} | "
                    f"{all_values['spearman_mean_normal_angle_vs_asymmetry']:+.3f} | "
                    f"{corrected['mean_original_minus_shifted_mae_mm']:+.4f} |"
                )
    report.extend(["", "## Existing Anna dG0", ""])
    anna_obs = summary["existing_anna_dg0"]["observability"]
    report.append(
        f"- Mean log norm ratio: {anna_obs['sample_mean_log_ratio']['mean']:+.5f}; "
        f"positive samples: {100.0 * anna_obs['fraction_samples_positive_mean_log_ratio']:.1f}%."
    )
    for metric in ("euclidean", "geodesic"):
        all_values = summary["existing_anna_dg0"][metric]["all"]
        corrected = summary["existing_anna_dg0"][metric]["correction"]
        report.append(
            f"- {metric}: original-minus-shifted MAE {all_values['mean_original_minus_shifted_mae_mm']:+.4f} mm; "
            f"rho(log ratio) {all_values['spearman_log_ratio_vs_original_minus_shifted_mae']:+.3f}; "
            f"corrected subset {corrected['mean_original_minus_shifted_mae_mm']:+.4f} mm."
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
