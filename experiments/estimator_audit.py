#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.csgraph import dijkstra

from metrics import optimal_assignment


ROOT = Path(__file__).resolve().parents[1]
MESH_ROOT = ROOT / "new_version_mesh_experiments"
GDG_ROOT = ROOT / "new_version_mesh_experiments_g_dg"
DEFAULT_MODEL_DIR = MESH_ROOT / "best-model-folder"
DEFAULT_DATASET = MESH_ROOT / "best_examples" / "test_dataset_mesh.npz"
HONEST_ESTIMATORS = (
    "single_draw",
    "posterior_mean_naive",
    "posterior_mean_aligned",
    "posterior_medoid",
)
ALL_ESTIMATORS = HONEST_ESTIMATORS + ("oracle_best_of_n",)
IDENTITY_PERM = np.arange(3, dtype=np.int64)


def _validate_draws(theta_draws: np.ndarray) -> np.ndarray:
    draws = np.asarray(theta_draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1:] != (3, 3):
        raise ValueError(f"Expected theta_draws [S,3,3], got {draws.shape}")
    if draws.shape[0] < 1:
        raise ValueError("theta_draws is empty")
    return draws


def _continuous_assignment(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, ...]:
    source_ids = np.arange(3, dtype=np.int64)

    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(reference[int(a)] - candidate[int(b)]))

    _mean_distance, permutation = optimal_assignment(source_ids, source_ids, distance, K=3)
    if permutation is None:
        raise RuntimeError("Continuous source assignment failed")
    return tuple(int(x) for x in permutation)


def align_draws_to_reference(
    theta_draws: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align every draw to draw 0 in continuous Euclidean coordinate space."""
    draws = _validate_draws(theta_draws)
    aligned = np.empty_like(draws)
    permutations = np.empty((draws.shape[0], 3), dtype=np.int64)
    aligned[0] = draws[0]
    permutations[0] = IDENTITY_PERM
    nonidentity = 0
    for index in range(1, draws.shape[0]):
        permutation = np.asarray(
            _continuous_assignment(draws[0], draws[index]), dtype=np.int64
        )
        permutations[index] = permutation
        aligned[index] = draws[index, permutation]
        nonidentity += int(not np.array_equal(permutation, IDENTITY_PERM))
    denominator = draws.shape[0] - 1
    fraction = float(nonidentity / denominator) if denominator > 0 else 0.0
    return aligned, permutations, fraction


def posterior_mean_naive(theta_draws: np.ndarray) -> np.ndarray:
    return np.mean(_validate_draws(theta_draws), axis=0)


def posterior_mean_aligned(theta_draws: np.ndarray) -> tuple[np.ndarray, float]:
    aligned, _permutations, nonidentity_fraction = align_draws_to_reference(theta_draws)
    return np.mean(aligned, axis=0), nonidentity_fraction


def _pairwise_permutation_invariant_distances(theta_draws: np.ndarray) -> np.ndarray:
    draws = _validate_draws(theta_draws)
    best = np.full((draws.shape[0], draws.shape[0]), np.inf, dtype=np.float64)
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    for permutation in permutations:
        candidate = draws[:, permutation, :]
        delta = draws[:, None, :, :] - candidate[None, :, :, :]
        distance = np.mean(np.linalg.norm(delta, axis=-1), axis=-1)
        best = np.minimum(best, distance)
    return best


def posterior_medoid(theta_draws: np.ndarray) -> np.ndarray:
    """Return a posterior draw selected without access to ground truth."""
    draws = _validate_draws(theta_draws)
    pairwise = _pairwise_permutation_invariant_distances(draws)
    medoid_index = int(np.argmin(np.sum(pairwise, axis=1)))
    return np.array(draws[medoid_index], copy=True)


def estimate_honest(
    theta_draws: np.ndarray, estimator: str
) -> tuple[np.ndarray, dict[str, float]]:
    draws = _validate_draws(theta_draws)
    if estimator == "single_draw":
        return np.array(draws[0], copy=True), {}
    if estimator == "posterior_mean_naive":
        return posterior_mean_naive(draws), {}
    if estimator == "posterior_mean_aligned":
        coords, fraction = posterior_mean_aligned(draws)
        return coords, {"slot_nonidentity_fraction": float(fraction)}
    if estimator == "posterior_medoid":
        return posterior_medoid(draws), {}
    raise ValueError(f"Unknown honest estimator: {estimator}")


def _finite_aggregate(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        return {"n_valid": 0}
    output: dict[str, float | int] = {"n_valid": len(rows)}
    for key in sorted(rows[0]):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = float(np.mean(finite)) if finite.size else float("nan")
        output[f"{key}_n_nonfinite"] = int(values.size - finite.size)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Estimator audit",
        "",
        f"- Model: `{summary['model_key']}`",
        f"- Dataset: `{summary['dataset_npz']}`",
        f"- Samples: {summary['n_samples']}",
        f"- Seed: {summary['seed']}",
        "",
        "All estimators at a given N use exactly the same posterior draws. `oracle_best_of_n` is ground-truth dependent and is never an eligible deployed estimator.",
        "",
        "| N | Estimator | MAE geo, mm | MAE euc, mm | Hit geo @10 | Hit geo @15 | N valid |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for n_draws in summary["n_draws"]:
        group = summary["results"][str(n_draws)]
        for estimator in summary["estimators"]:
            metrics = group[estimator]
            lines.append(
                f"| {n_draws} | `{estimator}` | {metrics.get('mae_geo_mm', float('nan')):.4f} | "
                f"{metrics.get('mae_euc_mm', float('nan')):.4f} | "
                f"{metrics.get('hit_geo@10mm', float('nan')):.4f} | "
                f"{metrics.get('hit_geo@15mm', float('nan')):.4f} | {metrics.get('n_valid', 0)} |"
            )
        slot = group["slot_permutation_diagnostic"]
        lines.extend(
            [
                "",
                f"- N={n_draws}: mean non-identity alignment fraction excluding draw 0 = "
                f"{slot['mean_nonidentity_fraction']:.6f}; fraction of samples with any non-identity draw = "
                f"{slot['samples_with_any_nonidentity_fraction']:.6f}.",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare honest and oracle posterior estimators")
    parser.add_argument("--dataset_npz", default=str(DEFAULT_DATASET))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model_key", default="", help="checkpoint_specs model key; overrides --model_dir")
    parser.add_argument("--estimators", nargs="+", choices=ALL_ESTIMATORS, default=list(ALL_ESTIMATORS))
    parser.add_argument("--n_draws", nargs="+", type=int, default=[25, 100])
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def _load_spec(args: argparse.Namespace):
    if args.model_key:
        from eval_cnn_checkpoints_25march import checkpoint_specs

        matches = [spec for spec in checkpoint_specs() if spec["model_key"] == args.model_key]
        if len(matches) != 1:
            raise ValueError(f"Unknown model_key={args.model_key!r}")
        return matches[0]
    from eval_posterior_mean_100 import build_spec_from_model_dir

    return build_spec_from_model_dir(Path(args.model_dir).resolve())


def main() -> None:
    args = _parse_args()
    if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if str(MESH_ROOT) not in sys.path:
        sys.path.insert(0, str(MESH_ROOT))
    if str(GDG_ROOT) not in sys.path:
        sys.path.insert(0, str(GDG_ROOT))

    import jax
    import jax.numpy as jnp

    from eval_cnn_checkpoints_25march import (
        compute_metrics_one_precomputed,
        load_checkpoint_runtime,
    )
    from eval_posterior_mean_100 import make_multi_sampler
    from metrics import build_graph_from_edges, project_coords_to_vertices
    from optuna_sweep_cnn import pool_time

    spec = _load_spec(args)
    runtime = load_checkpoint_runtime(spec)
    dataset_path = Path(args.dataset_npz).resolve()
    data = np.load(dataset_path, allow_pickle=True)
    ys = np.asarray(data["Ys"])
    centroids = np.asarray(data["centroids_idx"], dtype=np.int64)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    edge_index = np.asarray(data["edge_index"])
    edge_weight_mm = np.asarray(data["edge_weight_mm"])
    params_spatial = data["params_spatial"].item() if "params_spatial" in data.files else {}
    r_mm = params_spatial.get("R_mm", [10.0, 8.0, 10.0])
    a0 = params_spatial.get("A0", None)
    n_samples = min(len(ys), args.limit) if args.limit > 0 else len(ys)
    pool = int(runtime["pool"])
    ode_steps = int(runtime["ode_steps"])
    theta_mean = np.asarray(runtime["theta_mean"], dtype=np.float32)
    theta_std = np.asarray(runtime["theta_std"], dtype=np.float32)

    graph = build_graph_from_edges(edge_index, edge_weight_mm, len(vertices))
    print("[estimator_audit] precomputing all-pairs mesh distances", flush=True)
    full_geo_mm = np.asarray(dijkstra(graph, directed=False), dtype=np.float64)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    per_sample_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for n_draws in args.n_draws:
        sampler = make_multi_sampler(runtime["model"], theta_dim=9, steps=ode_steps)
        key = jax.random.PRNGKey(int(args.seed))
        metric_rows: dict[str, list[dict[str, float]]] = {
            estimator: [] for estimator in args.estimators
        }
        permutation_fractions: list[float] = []
        for sample_index in range(n_samples):
            y = ys[sample_index : sample_index + 1].astype(np.float32)
            pooled = pool_time(y, pool=pool).astype(np.float32)[0]
            key, sample_key = jax.random.split(key, 2)
            draw_keys = jax.random.split(sample_key, int(n_draws))
            x_norm = np.asarray(sampler(jnp.asarray(pooled), draw_keys), dtype=np.float32)
            theta_draws = (
                x_norm * theta_std[None, :] + theta_mean[None, :]
            ).reshape(int(n_draws), 3, 3)

            _aligned, _permutations, nonidentity_fraction = align_draws_to_reference(theta_draws)
            permutation_fractions.append(nonidentity_fraction)
            c_true = centroids[sample_index]

            for estimator in args.estimators:
                if estimator == "oracle_best_of_n":
                    best_metrics = None
                    best_score = float("inf")
                    for coords in theta_draws:
                        c_pred = project_coords_to_vertices(coords, vertices)
                        metrics = compute_metrics_one_precomputed(
                            C_true=c_true,
                            C_pred=c_pred,
                            vertices_m=vertices,
                            full_geo_mm=full_geo_mm,
                            R_mm=r_mm,
                            A0=a0,
                            thresholds_mm=(5, 10, 15, 20, 30),
                        )
                        score = float(metrics.get("mae_geo_mm", np.inf))
                        if np.isfinite(score) and score < best_score:
                            best_score = score
                            best_metrics = metrics
                    metrics = best_metrics
                else:
                    coords, _diagnostics = estimate_honest(theta_draws, estimator)
                    c_pred = project_coords_to_vertices(coords, vertices)
                    metrics = compute_metrics_one_precomputed(
                        C_true=c_true,
                        C_pred=c_pred,
                        vertices_m=vertices,
                        full_geo_mm=full_geo_mm,
                        R_mm=r_mm,
                        A0=a0,
                        thresholds_mm=(5, 10, 15, 20, 30),
                    )
                row: dict[str, Any] = {
                    "sample_index": int(sample_index),
                    "n_draws": int(n_draws),
                    "estimator": estimator,
                    "slot_nonidentity_fraction": float(nonidentity_fraction),
                }
                if metrics is not None:
                    clean_metrics = {name: float(value) for name, value in metrics.items()}
                    metric_rows[estimator].append(clean_metrics)
                    row.update(clean_metrics)
                per_sample_rows.append(row)
            if (sample_index + 1) % 10 == 0 or sample_index + 1 == n_samples:
                print(
                    f"[estimator_audit] N={n_draws} {sample_index + 1}/{n_samples}",
                    flush=True,
                )

        results[str(n_draws)] = {
            estimator: _finite_aggregate(metric_rows[estimator])
            for estimator in args.estimators
        }
        fractions = np.asarray(permutation_fractions, dtype=np.float64)
        results[str(n_draws)]["slot_permutation_diagnostic"] = {
            "mean_nonidentity_fraction": float(np.mean(fractions)),
            "median_nonidentity_fraction": float(np.median(fractions)),
            "samples_with_any_nonidentity_fraction": float(np.mean(fractions > 0.0)),
            "n_samples": int(fractions.size),
        }

    summary = {
        "model_key": str(runtime["model_key"]),
        "model_path": str(runtime["model_path"]),
        "dataset_npz": str(dataset_path),
        "n_samples": int(n_samples),
        "n_draws": [int(value) for value in args.n_draws],
        "estimators": list(args.estimators),
        "seed": int(args.seed),
        "pool": pool,
        "ode_steps": ode_steps,
        "jax_devices": str(jax.devices()),
        "results": results,
        "wall_seconds": float(time.perf_counter() - started),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(out_dir / "per_sample_metrics.csv", per_sample_rows)
    _write_report(out_dir / "report.md", summary)
    print(f"[estimator_audit] wrote {out_dir}")


if __name__ == "__main__":
    main()
