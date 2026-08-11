#!/usr/bin/env python3
"""Evaluate residual-selected posterior draws on a single-input dataset."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.sparse.csgraph import dijkstra

from residual_rescoring_audit import draw_seed_for_sample


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
GDG_ROOT = PROJECT_ROOT / "new_version_mesh_experiments_g_dg"
DATASET_KEYS = (
    "Ys",
    "centroids_idx",
    "vertices",
    "edge_index",
    "edge_weight_mm",
    "params_spatial",
)


def load_dataset_and_leadfield(
    dataset_path: str | Path, leadfield_path: str | Path
) -> dict[str, Any]:
    with np.load(Path(dataset_path), allow_pickle=True) as payload:
        missing = set(DATASET_KEYS) - set(payload.files)
        if missing:
            raise KeyError(f"dataset is missing required keys: {sorted(missing)}")
        dataset = {
            key: payload[key].item() if key == "params_spatial" else np.asarray(payload[key])
            for key in DATASET_KEYS
        }
    with np.load(Path(leadfield_path), allow_pickle=True) as payload:
        if "G" not in payload.files or "vertices" not in payload.files:
            raise KeyError("leadfield artifact must contain G and vertices")
        leadfield = np.asarray(payload["G"], dtype=np.float64)
        leadfield_vertices = np.asarray(payload["vertices"])
    if not np.array_equal(dataset["vertices"], leadfield_vertices):
        raise ValueError("dataset and leadfield vertices differ")
    if leadfield.shape != (dataset["Ys"].shape[1], len(dataset["vertices"])):
        raise ValueError(
            f"G shape {leadfield.shape} does not match sensors/vertices "
            f"{(dataset['Ys'].shape[1], len(dataset['vertices']))}"
        )
    if not np.all(np.isfinite(leadfield)):
        raise ValueError("G contains non-finite values")
    return {"dataset": dataset, "localization_G": leadfield}


def paired_comparisons(
    runs: list[dict[str, Any]], *, metric: str
) -> list[dict[str, Any]]:
    model_keys = sorted({str(row["model_key"]) for row in runs})
    output: list[dict[str, Any]] = []
    for model_a, model_b in combinations(model_keys, 2):
        rows_a = {
            (int(row["repeat"]), int(row["data_seed"])): row
            for row in runs
            if row["model_key"] == model_a
        }
        rows_b = {
            (int(row["repeat"]), int(row["data_seed"])): row
            for row in runs
            if row["model_key"] == model_b
        }
        if rows_a.keys() != rows_b.keys():
            raise ValueError(f"Unpaired runs for {model_a} and {model_b}")
        keys = sorted(rows_a)
        differences = [
            float(rows_a[key][metric]) - float(rows_b[key][metric]) for key in keys
        ]
        output.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "differences_a_minus_b": differences,
                "mean_difference_a_minus_b": float(np.mean(differences)),
                "a_wins": int(sum(value < 0.0 for value in differences)),
                "b_wins": int(sum(value > 0.0 for value in differences)),
                "ties": int(sum(value == 0.0 for value in differences)),
                "data_seeds": [key[1] for key in keys],
            }
        )
    return output


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _distribution(values: list[float]) -> dict[str, float | int]:
    raw = np.asarray(values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    return {
        "n": int(raw.size),
        "n_finite": int(finite.size),
        "n_nonfinite": int(raw.size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_spec(model_key: str) -> dict[str, Any]:
    from eval_cnn_checkpoints_25march import checkpoint_specs

    matches = [spec for spec in checkpoint_specs() if spec["model_key"] == model_key]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous model_key: {model_key}")
    return matches[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_npz", required=True)
    parser.add_argument("--leadfield_npz", required=True)
    parser.add_argument("--model_key", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_draws", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_start_commit = _git_commit()
    if args.n_draws <= 0:
        raise ValueError("n_draws must be positive")
    if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for path in (MODULE_ROOT, GDG_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import jax
    import jax.numpy as jnp

    from eval_cnn_checkpoints_25march import (
        compute_metrics_one_precomputed,
        load_checkpoint_runtime,
    )
    from eval_posterior_mean_100 import make_multi_sampler
    from metrics import build_graph_from_edges, project_coords_to_vertices
    from optuna_sweep_cnn import pool_time
    from residual_rescoring import residual_rescore_draws

    dataset_path = Path(args.dataset_npz).resolve()
    leadfield_path = Path(args.leadfield_npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_dataset_and_leadfield(dataset_path, leadfield_path)
    data = loaded["dataset"]
    localization_g = loaded["localization_G"]
    n_total = len(data["Ys"])
    n_samples = min(n_total, args.limit) if args.limit > 0 else n_total

    spec = _resolve_spec(args.model_key)
    runtime = load_checkpoint_runtime(spec)
    pool = int(runtime["pool"])
    theta_mean = np.asarray(runtime["theta_mean"], dtype=np.float32)
    theta_std = np.asarray(runtime["theta_std"], dtype=np.float32)
    sampler = make_multi_sampler(
        runtime["model"], theta_dim=9, steps=int(runtime["ode_steps"])
    )
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    graph = build_graph_from_edges(
        data["edge_index"], data["edge_weight_mm"], len(vertices)
    )
    print("[dataset-rescore] precomputing geodesic matrix", flush=True)
    full_geo_mm = np.asarray(dijkstra(graph, directed=False), dtype=np.float64)
    params_spatial = data["params_spatial"]
    radii = params_spatial.get("R_mm", [10.0, 8.0, 10.0])
    amplitudes = params_spatial.get("A0", None)

    rows: list[dict[str, Any]] = []
    selected_centroids = np.empty((n_samples, 3), dtype=np.int64)
    selected_indices = np.empty(n_samples, dtype=np.int64)
    started = time.perf_counter()
    for sample_index in range(n_samples):
        draw_seed = draw_seed_for_sample(args.seed, sample_index)
        draw_keys = jax.random.split(jax.random.PRNGKey(draw_seed), args.n_draws)
        y = np.asarray(data["Ys"][sample_index], dtype=np.float32)
        pooled = pool_time(y[None, ...], pool=pool).astype(np.float32)[0]
        normalized = np.asarray(sampler(jnp.asarray(pooled), draw_keys), dtype=np.float32)
        theta_draws = (
            normalized * theta_std[None, :] + theta_mean[None, :]
        ).reshape(args.n_draws, 3, 3)
        residuals, selected_index = residual_rescore_draws(
            theta_draws,
            y,
            localization_g,
            vertices,
            graph,
            params_spatial,
        )
        prediction = project_coords_to_vertices(theta_draws[selected_index], vertices)
        metrics = compute_metrics_one_precomputed(
            C_true=np.asarray(data["centroids_idx"][sample_index], dtype=np.int64),
            C_pred=prediction,
            vertices_m=vertices,
            full_geo_mm=full_geo_mm,
            R_mm=radii,
            A0=amplitudes,
            thresholds_mm=(5, 10, 15, 20, 30),
        )
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "residual_selected_index": int(selected_index),
            "residual_selected_value": float(residuals[selected_index]),
        }
        row.update({key: float(value) for key, value in metrics.items()})
        rows.append(row)
        selected_centroids[sample_index] = prediction
        selected_indices[sample_index] = selected_index
        if (sample_index + 1) % 10 == 0 or sample_index + 1 == n_samples:
            print(
                f"[dataset-rescore] {sample_index + 1}/{n_samples}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    metric_names = sorted(
        key for key in rows[0] if key not in {
            "sample_index", "residual_selected_index", "residual_selected_value"
        }
    )
    metrics = {name: _distribution([float(row[name]) for row in rows]) for name in metric_names}
    artifact_commit = _git_commit()
    summary = {
        "dataset_npz": str(dataset_path),
        "leadfield_npz": str(leadfield_path),
        "leadfield_key": "G",
        "model_key": args.model_key,
        "repeat": int(args.repeat),
        "data_seed": int(args.data_seed),
        "eval_seed": int(args.seed),
        "n_total": int(n_total),
        "n_samples": int(n_samples),
        "n_draws": int(args.n_draws),
        "estimator": "residual_selected",
        "truth_used_for_selection": False,
        "run_start_git_commit": run_start_commit,
        "artifact_write_git_commit": artifact_commit,
        "repository_changed_during_run": run_start_commit != artifact_commit,
        "jax_devices": str(jax.devices()),
        "wall_seconds": float(time.perf_counter() - started),
        "metrics": metrics,
        "selected_index": _distribution(selected_indices.astype(float).tolist()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "per_sample_metrics.csv", rows)
    np.savez_compressed(
        out_dir / "selected_predictions.npz",
        sample_indices=np.arange(n_samples, dtype=np.int64),
        centroids_selected=selected_centroids,
        draw_indices_selected=selected_indices,
    )
    (out_dir / "report.md").write_text(
        "\n".join(
            [
                "# Residual-selected dataset evaluation",
                "",
                f"- Dataset: `{dataset_path}`",
                f"- Explicit localization G: `{leadfield_path}` key `G`",
                f"- Model: `{args.model_key}`; samples: {n_samples}; draws: {args.n_draws}",
                "- Selection uses sensor data, G, mesh, and dataset-level spatial parameters; truth is used only after selection for metrics.",
                "",
                f"- MAE euc: {metrics['mae_euc_mm']['mean']:.4f} mm",
                f"- MAE geo: {metrics['mae_geo_mm']['mean']:.4f} mm ({metrics['mae_geo_mm']['n_nonfinite']} non-finite)",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "mae_euc_mm": metrics["mae_euc_mm"]["mean"],
        "mae_geo_mm": metrics["mae_geo_mm"]["mean"],
        "n_nonfinite_geo": metrics["mae_geo_mm"]["n_nonfinite"],
    }, indent=2))


if __name__ == "__main__":
    main()
