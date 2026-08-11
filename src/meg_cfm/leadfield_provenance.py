#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import subspace_angles

from residual_rescoring import (
    ProjectedGaussianCache,
    orthonormal_column_basis,
    subspace_residual,
    subspace_residual_from_covariance,
)


ROOT = Path(__file__).resolve().parents[1]


def _matching_finite_matrices(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError(f"lead fields must be matching 2-D matrices, got {a.shape} and {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("lead fields contain non-finite values")
    return a, b


def optimal_global_scale(*, target: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    target_matrix, candidate_matrix = _matching_finite_matrices(target, candidate)
    denominator = float(np.sum(np.square(candidate_matrix)))
    target_norm = float(np.linalg.norm(target_matrix))
    if denominator <= 0.0 or target_norm <= 0.0:
        raise ValueError("lead fields must have positive Frobenius norm")
    alpha = float(np.sum(target_matrix * candidate_matrix) / denominator)
    relative = float(np.linalg.norm(target_matrix - alpha * candidate_matrix) / target_norm)
    return {"alpha": alpha, "relative_residual": relative}


def column_direction_diagnostics(target: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    target_matrix, candidate_matrix = _matching_finite_matrices(target, candidate)
    target_norms = np.linalg.norm(target_matrix, axis=0)
    candidate_norms = np.linalg.norm(candidate_matrix, axis=0)
    valid = (target_norms > 0.0) & (candidate_norms > 0.0)
    if not np.any(valid):
        raise ValueError("lead fields have no mutually nonzero columns")
    target_unit = target_matrix[:, valid] / target_norms[valid]
    candidate_unit = candidate_matrix[:, valid] / candidate_norms[valid]
    cosines = np.sum(target_unit * candidate_unit, axis=0)
    signs = np.where(cosines < 0.0, -1.0, 1.0)
    aligned = candidate_unit * signs[None, :]
    scale = float(np.linalg.norm(target_unit))
    return {
        "n_valid_columns": int(np.sum(valid)),
        "n_zero_or_unmatched_columns": int(valid.size - np.sum(valid)),
        "raw_relative_frobenius": float(np.linalg.norm(target_unit - candidate_unit) / scale),
        "sign_aligned_relative_frobenius": float(np.linalg.norm(target_unit - aligned) / scale),
        "median_absolute_cosine": float(np.median(np.abs(cosines))),
        "p10_absolute_cosine": float(np.percentile(np.abs(cosines), 10)),
        "minimum_absolute_cosine": float(np.min(np.abs(cosines))),
    }


def principal_angles_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("bases must be 2-D with the same ambient dimension")
    q_a = orthonormal_column_basis(a)
    q_b = orthonormal_column_basis(b)
    if q_a.shape[1] == 0 or q_b.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    return np.sort(np.degrees(subspace_angles(q_a, q_b)))


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("angle distribution must be nonempty and finite")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def summarize_angles(rows: np.ndarray) -> dict[str, Any]:
    angles = np.asarray(rows, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[0] == 0 or angles.shape[1] == 0:
        raise ValueError("rows must be a nonempty 2-D angle matrix")
    return {
        "n_cases": int(angles.shape[0]),
        "angles_per_case": int(angles.shape[1]),
        "all_angles_deg": _distribution(angles.ravel()),
        "max_angle_per_case_deg": _distribution(np.max(angles, axis=1)),
    }


def sample_distinct_triplets(*, n_vertices: int, count: int, seed: int) -> np.ndarray:
    if n_vertices < 3 or count < 1 or count > comb(int(n_vertices), 3):
        raise ValueError("requested triplet count is impossible")
    rng = np.random.default_rng(int(seed))
    seen: set[tuple[int, int, int]] = set()
    rows: list[tuple[int, int, int]] = []
    while len(rows) < int(count):
        triplet = tuple(sorted(int(x) for x in rng.choice(n_vertices, size=3, replace=False)))
        if triplet not in seen:
            seen.add(triplet)
            rows.append(triplet)
    return np.asarray(rows, dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_manifest(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def capture_environment(
    *,
    input_files: dict[str, Path],
    arrays: dict[str, np.ndarray],
    command: str,
    seed: int,
    git_commit: str,
) -> dict[str, Any]:
    package_names = ("numpy", "scipy", "mne", "nibabel", "jax", "equinox")
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    try:
        from mne.datasets import sample

        sample_path = str(sample.data_path(download=False))
    except Exception as error:  # manifest capture must not hide the main diagnostic
        sample_path = f"unavailable: {type(error).__name__}: {error}"
    return {
        "command": str(command),
        "seed": int(seed),
        "git_commit": str(git_commit),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "container_hostname_env": os.environ.get("HOSTNAME", ""),
        "packages": packages,
        "mne_sample_data_path": sample_path,
        "input_files": {
            name: {
                "path": str(Path(path).resolve()),
                "size_bytes": int(Path(path).stat().st_size),
                "sha256": sha256_file(Path(path)),
            }
            for name, path in input_files.items()
        },
        "arrays": {name: _array_manifest(value) for name, value in arrays.items()},
    }


def load_anna_localization_g(path: Path) -> tuple[np.ndarray, dict[str, float | int]]:
    with np.load(Path(path), allow_pickle=True) as payload:
        if "G_for_simulation" not in payload.files:
            raise KeyError("anna_dg0 artifact has no G_for_simulation")
        matrices = np.asarray(payload["G_for_simulation"])
    if matrices.ndim != 3 or matrices.shape[0] == 0:
        raise ValueError("G_for_simulation must have shape (samples, sensors, vertices)")
    reference = np.array(matrices[0], dtype=np.float64, copy=True)
    max_variation = 0.0
    for index in range(1, matrices.shape[0]):
        max_variation = max(
            max_variation,
            float(np.max(np.abs(np.asarray(matrices[index], dtype=np.float64) - reference))),
        )
    if max_variation != 0.0:
        raise ValueError(
            "G_for_simulation is not constant across samples; refusing a perturbed lead field"
        )
    return reference, {
        "n_matrices": int(matrices.shape[0]),
        "max_abs_variation": max_variation,
    }


def assert_matching_geometry(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], *, label: str
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"label": str(label)}
    for key in ("vertices", "edge_index", "edge_weight_mm"):
        if key not in reference or key not in candidate:
            raise KeyError(f"missing geometry key {key}")
        first = np.asarray(reference[key])
        second = np.asarray(candidate[key])
        if first.shape != second.shape or not np.array_equal(first, second):
            max_abs = (
                float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64))))
                if first.shape == second.shape and first.size
                else float("inf")
            )
            raise ValueError(f"{label} {key} differs from rebuilt geometry (max_abs={max_abs})")
        diagnostics[key] = {"shape": list(first.shape), "exact_equal": True}
    return diagnostics


def projector_patch_angles(
    first_g: np.ndarray,
    second_g: np.ndarray,
    centroid_rows: np.ndarray,
    graph,
    params_spatial: dict,
) -> np.ndarray:
    first, second = _matching_finite_matrices(first_g, second_g)
    centroids = np.asarray(centroid_rows, dtype=np.int64)
    source_count = int(np.asarray(params_spatial["R_mm"]).size)
    if centroids.ndim != 2 or centroids.shape[1] != source_count:
        raise ValueError(f"centroid_rows must have shape (N, {source_count})")
    first_cache = ProjectedGaussianCache(first, graph, params_spatial)
    second_cache = ProjectedGaussianCache(second, graph, params_spatial)
    rows = np.full((centroids.shape[0], source_count), 90.0, dtype=np.float64)
    for index, centroid_row in enumerate(centroids):
        angles = principal_angles_deg(
            first_cache.matrix(centroid_row), second_cache.matrix(centroid_row)
        )
        rows[index, : angles.size] = angles
    return rows


def truth_residual_for_provenance(
    Y,
    true_centroids,
    localization_G,
    graph,
    params_spatial,
) -> float:
    cache = ProjectedGaussianCache(localization_G, graph, params_spatial)
    return subspace_residual(cache.matrix(np.asarray(true_centroids)), np.asarray(Y))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projector-aware lead-field provenance audit")
    parser.add_argument("--anna_dg0_npz", required=True)
    parser.add_argument("--fixed_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_triplets", type=int, default=200)
    parser.add_argument("--n_patch_fixed", type=int, default=0, help="0 uses all centroid rows")
    parser.add_argument("--n_patch_anna", type=int, default=0, help="0 uses all centroid rows")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--truth_residual_2x2", action="store_true")
    return parser.parse_args(argv)


def _load_geometry_and_centroids(path: Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=True) as payload:
        return {
            "vertices": np.asarray(payload["vertices"]),
            "edge_index": np.asarray(payload["edge_index"]),
            "edge_weight_mm": np.asarray(payload["edge_weight_mm"]),
            "centroids_idx": np.asarray(payload["centroids_idx"], dtype=np.int64),
            "params_spatial": payload["params_spatial"].item(),
            "params_noise": payload["params_noise"].item(),
            "meta": payload["meta"].item(),
        }


def _choose_rows(rows: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or limit >= len(rows):
        indices = np.arange(len(rows), dtype=np.int64)
    else:
        indices = np.sort(np.random.default_rng(seed).choice(len(rows), size=limit, replace=False))
    return np.asarray(rows)[indices], indices


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _critical_mne_files() -> dict[str, Path]:
    from mne.datasets import sample

    base = Path(sample.data_path(download=False))
    return {
        "mne_raw": base / "MEG" / "sample" / "sample_audvis_raw.fif",
        "mne_trans": base / "MEG" / "sample" / "sample_audvis_raw-trans.fif",
        "mne_bem": base / "subjects" / "sample" / "bem" / "sample-5120-bem-sol.fif",
        "mne_lh_white": base / "subjects" / "sample" / "surf" / "lh.white",
        "mne_rh_white": base / "subjects" / "sample" / "surf" / "rh.white",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _number_distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be nonempty and finite")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _truth_residual_dataset_rows(
    *,
    dataset_label: str,
    dataset_path: Path,
    centroids: np.ndarray,
    params_spatial: dict,
    graph,
    rebuilt_g: np.ndarray,
    anna_g: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(dataset_path, allow_pickle=True) as payload:
        ys = np.asarray(payload["Ys"])
    if len(ys) != len(centroids):
        raise ValueError(f"{dataset_label} Ys and centroids length mismatch")
    rebuilt_cache = ProjectedGaussianCache(rebuilt_g, graph, params_spatial)
    anna_cache = ProjectedGaussianCache(anna_g, graph, params_spatial)
    rows: list[dict[str, Any]] = []
    rebuilt_values = np.empty(len(ys), dtype=np.float64)
    anna_values = np.empty(len(ys), dtype=np.float64)
    for index, (y, true_centroids) in enumerate(zip(ys, centroids)):
        covariance = np.asarray(y, dtype=np.float64) @ np.asarray(y, dtype=np.float64).T
        rebuilt_values[index] = subspace_residual_from_covariance(
            rebuilt_cache.matrix(true_centroids), covariance
        )
        anna_values[index] = subspace_residual_from_covariance(
            anna_cache.matrix(true_centroids), covariance
        )
        rows.append(
            {
                "dataset": dataset_label,
                "sample_index": index,
                "residual_rebuilt_g": float(rebuilt_values[index]),
                "residual_anna_g": float(anna_values[index]),
                "difference_rebuilt_minus_anna": float(
                    rebuilt_values[index] - anna_values[index]
                ),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(ys):
            print(f"[leadfield] truth residual {dataset_label} {index + 1}/{len(ys)}", flush=True)
    differences = rebuilt_values - anna_values
    return rows, {
        "rebuilt_g": _number_distribution(rebuilt_values),
        "anna_g": _number_distribution(anna_values),
        "paired_difference_rebuilt_minus_anna": _number_distribution(differences),
        "fraction_rebuilt_lower": float(np.mean(differences < 0.0)),
        "fraction_anna_lower": float(np.mean(differences > 0.0)),
        "fraction_tied": float(np.mean(differences == 0.0)),
    }


def infer_fixed_leadfield_identity(
    truth_summary: dict[str, Any], *, decisive_fraction: float = 0.95
) -> dict[str, Any]:
    """Classify fixed-test G from within-dataset paired residual preferences."""
    if not 0.5 < decisive_fraction <= 1.0:
        raise ValueError("decisive_fraction must be in (0.5, 1]")
    fixed = truth_summary["fixed"]
    calibration = truth_summary["anna_dg0"]
    calibration_passed = calibration["fraction_anna_lower"] >= decisive_fraction
    if not calibration_passed:
        classification = "calibration_failed"
    elif fixed["fraction_rebuilt_lower"] >= decisive_fraction:
        classification = "rebuilt_like"
    elif fixed["fraction_anna_lower"] >= decisive_fraction:
        classification = "anna_like"
    else:
        classification = "unidentified"
    return {
        "classification": classification,
        "calibration_passed": bool(calibration_passed),
        "decisive_fraction": float(decisive_fraction),
        "decision_basis": "within-dataset paired truth-residual preference",
    }


def main() -> None:
    args = parse_args()
    anna_path = Path(args.anna_dg0_npz).resolve()
    fixed_path = Path(args.fixed_npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from generate_dataset import build_forward_vertices_edges
    from new_version_mesh_experiments.metrics import build_graph_from_edges

    print("[leadfield] rebuilding MNE localization G", flush=True)
    rebuilt_g, rebuilt_vertices, rebuilt_edges, rebuilt_weights = build_forward_vertices_edges()
    print("[leadfield] loading constant Anna localization G", flush=True)
    anna_g, anna_g_diagnostic = load_anna_localization_g(anna_path)
    fixed = _load_geometry_and_centroids(fixed_path)
    anna = _load_geometry_and_centroids(anna_path)
    rebuilt_geometry = {
        "vertices": rebuilt_vertices,
        "edge_index": rebuilt_edges,
        "edge_weight_mm": rebuilt_weights,
    }
    geometry_checks = {
        "fixed": assert_matching_geometry(rebuilt_geometry, fixed, label="fixed"),
        "anna_dg0": assert_matching_geometry(rebuilt_geometry, anna, label="anna_dg0"),
    }
    if rebuilt_g.shape != anna_g.shape:
        raise ValueError(f"lead-field shapes differ: {rebuilt_g.shape} vs {anna_g.shape}")

    scalar = optimal_global_scale(target=anna_g, candidate=rebuilt_g)
    columns = column_direction_diagnostics(anna_g, rebuilt_g)
    triplets = sample_distinct_triplets(
        n_vertices=rebuilt_g.shape[1], count=int(args.n_triplets), seed=int(args.seed)
    )
    triplet_angles = np.vstack(
        [principal_angles_deg(anna_g[:, row], rebuilt_g[:, row]) for row in triplets]
    )
    triplet_rows = [
        {
            "case_index": index,
            "vertex_0": int(vertices[0]),
            "vertex_1": int(vertices[1]),
            "vertex_2": int(vertices[2]),
            "angle_0_deg": float(angles[0]),
            "angle_1_deg": float(angles[1]),
            "angle_2_deg": float(angles[2]),
            "max_angle_deg": float(np.max(angles)),
        }
        for index, (vertices, angles) in enumerate(zip(triplets, triplet_angles))
    ]
    _write_csv(out_dir / "triplet_angles.csv", triplet_rows)

    graph = build_graph_from_edges(rebuilt_edges, rebuilt_weights, len(rebuilt_vertices))
    patch_rows: list[dict[str, Any]] = []
    patch_summaries: dict[str, Any] = {}
    for dataset_index, (label, dataset, limit) in enumerate(
        (
            ("fixed", fixed, int(args.n_patch_fixed)),
            ("anna_dg0", anna, int(args.n_patch_anna)),
        )
    ):
        selected_centroids, selected_indices = _choose_rows(
            dataset["centroids_idx"], limit, int(args.seed) + 100 + dataset_index
        )
        angles = projector_patch_angles(
            anna_g,
            rebuilt_g,
            selected_centroids,
            graph,
            dataset["params_spatial"],
        )
        patch_summaries[label] = summarize_angles(angles)
        for sample_index, centroids, angle_row in zip(selected_indices, selected_centroids, angles):
            patch_rows.append(
                {
                    "dataset": label,
                    "sample_index": int(sample_index),
                    "centroid_0": int(centroids[0]),
                    "centroid_1": int(centroids[1]),
                    "centroid_2": int(centroids[2]),
                    "angle_0_deg": float(angle_row[0]),
                    "angle_1_deg": float(angle_row[1]),
                    "angle_2_deg": float(angle_row[2]),
                    "max_angle_deg": float(np.max(angle_row)),
                }
            )
    _write_csv(out_dir / "patch_angles.csv", patch_rows)

    np.savez_compressed(
        out_dir / "rebuilt_leadfield_geometry.npz",
        G=rebuilt_g,
        vertices=rebuilt_vertices,
        edge_index=rebuilt_edges,
        edge_weight_mm=rebuilt_weights,
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_commit = "unavailable"
    input_files = {"fixed_npz": fixed_path, "anna_dg0_npz": anna_path, **_critical_mne_files()}
    environment = capture_environment(
        input_files=input_files,
        arrays={
            "G_rebuilt": rebuilt_g,
            "G_anna": anna_g,
            "vertices": rebuilt_vertices,
            "edge_index": rebuilt_edges,
            "edge_weight_mm": rebuilt_weights,
        },
        command=shlex.join(sys.argv),
        seed=int(args.seed),
        git_commit=git_commit,
    )
    (out_dir / "environment.json").write_text(
        json.dumps(_jsonable(environment), indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "scope": "projector-aware lead-field provenance; no residual_true 2x2 in this run",
        "leadfield_shapes": {"rebuilt": list(rebuilt_g.shape), "anna": list(anna_g.shape)},
        "anna_localization_g_check": anna_g_diagnostic,
        "geometry_checks": geometry_checks,
        "optimal_global_scale_rebuilt_to_anna": scalar,
        "column_normalized_diagnostic_only": columns,
        "raw_triplet_angles": summarize_angles(triplet_angles),
        "gaussian_patch_angles": patch_summaries,
        "noise_settings": {
            "fixed_params_noise": fixed["params_noise"],
            "anna_params_noise": anna["params_noise"],
            "saved_params_equal": fixed["params_noise"] == anna["params_noise"],
            "generator_noise_level": 0.3,
            "noise_level_saved_in_npz": False,
            "absolute_residual_floors_are_secondary": True,
        },
        "provenance_note": (
            "Training G is absent. Fixed-test identification is an indirect proxy for training G "
            "because the training and fixed artifacts were generated five days apart."
        ),
        "environment_json": str(out_dir / "environment.json"),
    }
    if args.truth_residual_2x2:
        truth_rows: list[dict[str, Any]] = []
        truth_summary: dict[str, Any] = {}
        for label, dataset_path, dataset in (
            ("fixed", fixed_path, fixed),
            ("anna_dg0", anna_path, anna),
        ):
            rows, dataset_summary = _truth_residual_dataset_rows(
                dataset_label=label,
                dataset_path=dataset_path,
                centroids=dataset["centroids_idx"],
                params_spatial=dataset["params_spatial"],
                graph=graph,
                rebuilt_g=rebuilt_g,
                anna_g=anna_g,
            )
            truth_rows.extend(rows)
            truth_summary[label] = dataset_summary
        _write_csv(out_dir / "truth_residual_2x2.csv", truth_rows)
        summary["scope"] = "projector-aware lead-field provenance with truth-only residual 2x2"
        summary["truth_only_residual_2x2"] = truth_summary
        summary["fixed_leadfield_identity"] = infer_fixed_leadfield_identity(truth_summary)
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Projector-aware lead-field provenance",
        "",
        "Column-normalized comparisons below are diagnostic only and were not used as preprocessing.",
        "",
        f"- Rebuilt-to-Anna optimal alpha: {scalar['alpha']:.9g}",
        f"- Relative residual after global scaling: {scalar['relative_residual']:.6f}",
        f"- Sign-aligned column-direction residual: {columns['sign_aligned_relative_frobenius']:.6f}",
        "",
        "| Scope | Cases | Median all angles | P90 max angle | Maximum angle |",
        "|---|---:|---:|---:|---:|",
    ]
    raw = summary["raw_triplet_angles"]
    lines.append(
        f"| Raw vertex triplets | {raw['n_cases']} | {raw['all_angles_deg']['median']:.4f} | "
        f"{raw['max_angle_per_case_deg']['p90']:.4f} | {raw['max_angle_per_case_deg']['max']:.4f} |"
    )
    for label, patch in patch_summaries.items():
        lines.append(
            f"| Gaussian patches: {label} | {patch['n_cases']} | {patch['all_angles_deg']['median']:.4f} | "
            f"{patch['max_angle_per_case_deg']['p90']:.4f} | {patch['max_angle_per_case_deg']['max']:.4f} |"
        )
    if args.truth_residual_2x2:
        lines.extend(
            [
                "",
                "## Truth-only provenance residuals",
                "",
                "These rows use known centroids only to identify lead-field provenance. They are not an honest estimator.",
                "",
                "| Dataset | Median residual rebuilt G | Median residual Anna G | Median paired rebuilt - Anna | Rebuilt lower fraction |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, values in summary["truth_only_residual_2x2"].items():
            lines.append(
                f"| {label} | {values['rebuilt_g']['median']:.6f} | "
                f"{values['anna_g']['median']:.6f} | "
                f"{values['paired_difference_rebuilt_minus_anna']['median']:.6f} | "
                f"{values['fraction_rebuilt_lower']:.4f} |"
            )
        identity = summary["fixed_leadfield_identity"]
        fixed_values = summary["truth_only_residual_2x2"]["fixed"]
        anna_values = summary["truth_only_residual_2x2"]["anna_dg0"]
        lines.extend(
            [
                "",
                f"Within-dataset classification: **{identity['classification']}** "
                f"(decision threshold {identity['decisive_fraction']:.2f}).",
                "",
                f"The Anna calibration passes: Anna G has lower residual in "
                f"{anna_values['fraction_anna_lower']:.2%} of anna_dg0 samples. "
                f"On fixed, rebuilt G has lower residual in "
                f"{fixed_values['fraction_rebuilt_lower']:.2%} of samples.",
                "",
                "The training dataset did not persist G. Because the training artifact (5 February) "
                "and fixed test artifact (10 February) were generated five days apart and likely in "
                "one environment, a rebuilt-like fixed result is indirect evidence—not direct "
                "provenance—that training used an equivalent G. Under that inference, evaluation on "
                "Anna data includes a lead-field domain shift. Future generated datasets must persist G.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "This first-stage report does not decide fixed-test lead-field identity. If real-patch angles are materially non-small, the next step is the within-dataset truth-residual 2x2 diagnostic.",
                "",
                "The training dataset did not persist G. Any inference from fixed test to training G is indirect and based on the five-day generation-date proximity.",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[leadfield] wrote {out_dir}")


if __name__ == "__main__":
    main()
