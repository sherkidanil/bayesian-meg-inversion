#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np
from numpy.lib import format as npy_format


def configure_project_import_path(
    module_file: str | Path = __file__, search_path: list[str] | None = None
) -> Path:
    project_root = Path(module_file).resolve().parent.parent
    target = sys.path if search_path is None else search_path
    if str(project_root) not in target:
        target.insert(0, str(project_root))
    return project_root


def current_git_commit(project_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def read_npy_prefix_from_npz(
    npz_path: str | Path, key: str, n_rows: int
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Stream a C-order prefix from a compressed NPZ without inflating the full array."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    member = f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive, archive.open(member) as stream:
        version = npy_format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(stream)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY version {version}")
        if fortran_order:
            raise ValueError("streamed prefix requires C-order NPY data")
        if not shape:
            raise ValueError("streamed array must have a row dimension")
        count_rows = min(int(n_rows), int(shape[0]))
        values_per_row = int(np.prod(shape[1:], dtype=np.int64))
        byte_count = count_rows * values_per_row * np.dtype(dtype).itemsize
        chunks: list[bytes] = []
        remaining = byte_count
        while remaining:
            chunk = stream.read(min(remaining, 16 * 1024 * 1024))
            if not chunk:
                raise EOFError(f"unexpected end of {member}")
            chunks.append(chunk)
            remaining -= len(chunk)
    array = np.frombuffer(b"".join(chunks), dtype=dtype).copy()
    return array.reshape((count_rows, *shape[1:])), tuple(int(x) for x in shape)


def classify_paired_preference(
    differences_rebuilt_minus_anna: np.ndarray,
    *,
    anna_calibration_fraction: float,
    decisive_fraction: float = 0.95,
) -> dict[str, float | str | bool]:
    differences = np.asarray(differences_rebuilt_minus_anna, dtype=np.float64)
    if differences.ndim != 1 or differences.size == 0 or not np.all(np.isfinite(differences)):
        raise ValueError("paired differences must be a nonempty finite vector")
    rebuilt_fraction = float(np.mean(differences < 0.0))
    anna_fraction = float(np.mean(differences > 0.0))
    calibration_passed = float(anna_calibration_fraction) >= decisive_fraction
    if not calibration_passed:
        classification = "calibration_failed"
    elif rebuilt_fraction >= decisive_fraction:
        classification = "rebuilt_like"
    elif anna_fraction >= decisive_fraction:
        classification = "anna_like"
    else:
        classification = "unidentified"
    return {
        "classification": classification,
        "calibration_passed": calibration_passed,
        "decisive_fraction": float(decisive_fraction),
        "fraction_rebuilt_lower": rebuilt_fraction,
        "fraction_anna_lower": anna_fraction,
        "fraction_tied": float(np.mean(differences == 0.0)),
        "median_difference_rebuilt_minus_anna": float(np.median(differences)),
    }


def historical_dependency(filename: str) -> str:
    mapping = {
        "meg_flow_dataset_10000_new_leadfield_error.npz": (
            "4.589 mm sent to Anna; oracle-contaminated localization result"
        ),
        "test_dataset_good_idxs_mesh.npz": (
            "good_idxs three-model report (240 samples) and model ranking"
        ),
        "meg_flow_dataset_10000_new.npz": "training domain of the evaluated CFM model",
    }
    return mapping.get(Path(filename).name, "unmapped historical result")


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truth-only historical lead-field provenance")
    parser.add_argument("--anna_dg0_npz", required=True)
    parser.add_argument("--calibration_summary", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=300)
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = configure_project_import_path()
    run_start_git_commit = current_git_commit(project_root)
    if args.max_samples <= 0:
        raise ValueError("max_samples must be positive")

    from generate_dataset import build_forward_vertices_edges
    from leadfield_provenance import load_anna_localization_g
    from metrics import build_graph_from_edges
    from residual_rescoring import ProjectedGaussianCache, subspace_residual_from_covariance

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.calibration_summary, encoding="utf-8") as stream:
        calibration = json.load(stream)
    anna_fraction = float(
        calibration["truth_only_residual_2x2"]["anna_dg0"]["fraction_anna_lower"]
    )

    print("[historical] rebuilding server G", flush=True)
    rebuilt_g, rebuilt_vertices, rebuilt_edges, rebuilt_weights = build_forward_vertices_edges()
    anna_g, anna_check = load_anna_localization_g(Path(args.anna_dg0_npz))
    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        dataset_path = Path(dataset_name).resolve()
        started = time.perf_counter()
        with np.load(dataset_path, allow_pickle=True) as payload:
            centroids = np.asarray(payload["centroids_idx"], dtype=np.int64)
            vertices = np.asarray(payload["vertices"])
            edges = np.asarray(payload["edge_index"])
            weights = np.asarray(payload["edge_weight_mm"])
            params_spatial = payload["params_spatial"].item()
        if not np.array_equal(vertices, rebuilt_vertices):
            raise ValueError(f"{dataset_path.name} vertices differ from rebuilt geometry")
        if not np.array_equal(edges, rebuilt_edges) or not np.array_equal(weights, rebuilt_weights):
            raise ValueError(f"{dataset_path.name} graph differs from rebuilt geometry")
        n_samples = min(len(centroids), int(args.max_samples))
        ys, ys_full_shape = read_npy_prefix_from_npz(dataset_path, "Ys", n_samples)
        graph = build_graph_from_edges(edges, weights, len(vertices))
        rebuilt_cache = ProjectedGaussianCache(rebuilt_g, graph, params_spatial)
        anna_cache = ProjectedGaussianCache(anna_g, graph, params_spatial)
        rebuilt_values = np.empty(n_samples, dtype=np.float64)
        anna_values = np.empty(n_samples, dtype=np.float64)
        for index in range(n_samples):
            covariance = np.asarray(ys[index], dtype=np.float64) @ np.asarray(
                ys[index], dtype=np.float64
            ).T
            rebuilt_values[index] = subspace_residual_from_covariance(
                rebuilt_cache.matrix(centroids[index]), covariance
            )
            anna_values[index] = subspace_residual_from_covariance(
                anna_cache.matrix(centroids[index]), covariance
            )
            sample_rows.append(
                {
                    "dataset": dataset_path.name,
                    "sample_index": index,
                    "residual_rebuilt_g": float(rebuilt_values[index]),
                    "residual_anna_g": float(anna_values[index]),
                    "difference_rebuilt_minus_anna": float(
                        rebuilt_values[index] - anna_values[index]
                    ),
                }
            )
            if (index + 1) % 50 == 0 or index + 1 == n_samples:
                print(
                    f"[historical] {dataset_path.name} {index + 1}/{n_samples}",
                    flush=True,
                )
        differences = rebuilt_values - anna_values
        decision = classify_paired_preference(
            differences, anna_calibration_fraction=anna_fraction
        )
        summaries.append(
            {
                "dataset": str(dataset_path),
                "file_mtime": dataset_path.stat().st_mtime,
                "full_ys_shape": list(ys_full_shape),
                "sample_rule": f"prefix_{n_samples}_of_generator-shuffled_rows",
                "dependency": historical_dependency(dataset_path.name),
                "rebuilt_residual": _distribution(rebuilt_values),
                "anna_residual": _distribution(anna_values),
                "paired_difference": _distribution(differences),
                **decision,
                "wall_seconds": float(time.perf_counter() - started),
            }
        )
        del ys

    artifact_write_git_commit = current_git_commit(project_root)
    result = {
        "scope": "truth-only historical dataset provenance; not an estimator",
        "run_start_git_commit": run_start_git_commit,
        "artifact_write_git_commit": artifact_write_git_commit,
        "repository_changed_during_run": (
            run_start_git_commit != artifact_write_git_commit
        ),
        "anna_calibration_fraction": anna_fraction,
        "calibration_summary": str(Path(args.calibration_summary).resolve()),
        "leadfield_environment_reference": calibration.get("environment_json"),
        "anna_g_check": anna_check,
        "sample_selection": "first max_samples rows; generator output order is randomized",
        "datasets": summaries,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "per_sample_residuals.csv", sample_rows)
    report = [
        "# Historical lead-field provenance",
        "",
        "This is a truth-only provenance audit and is not an honest localization estimator.",
        "",
        "| Dataset | Rows | Domain | Rebuilt lower | Median rebuilt−Anna | Historical dependency |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in summaries:
        report.append(
            f"| `{Path(row['dataset']).name}` | {row['rebuilt_residual']['n']} | "
            f"`{row['classification']}` | {row['fraction_rebuilt_lower']:.4f} | "
            f"{row['median_difference_rebuilt_minus_anna']:.6f} | {row['dependency']} |"
        )
    report.extend(
        [
            "",
            "The training dataset does not persist G. Its classification is direct evidence about the generating sensor data, but the exact historical matrix remains unrecoverable as a file.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[historical] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
