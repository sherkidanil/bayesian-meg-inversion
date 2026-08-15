#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


ROOT = Path(__file__).resolve().parents[1]
MESH_ROOT = ROOT / "new_version_mesh_experiments"
GDG_ROOT = ROOT / "new_version_mesh_experiments_g_dg"
DEFAULT_MODEL_DIR = MESH_ROOT / "best-model-folder"
REFERENCE_MESH = MESH_ROOT / "best_examples" / "test_dataset_mesh.npz"
REQUIRED_KEYS = {
    "Ys",
    "Ys_shifted",
    "centroids_idx",
    "centroids_idx_shifted",
    "real_inputs_delta_m",
    "vertices",
    "edge_index",
    "edge_weight_mm",
}


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def align_prediction_to_truth(
    c_true: np.ndarray,
    c_pred: np.ndarray,
    dist_fn: Callable[[int, int], float],
    fallback_dist_fn: Callable[[int, int], float] | None = None,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    """Return predictions ordered by the corresponding true-source index."""
    from metrics import optimal_assignment

    c_true = np.asarray(c_true, dtype=np.int64)
    c_pred = np.asarray(c_pred, dtype=np.int64)
    mean_dist, perm = optimal_assignment(c_true, c_pred, dist_fn, K=len(c_true))
    if perm is None or not np.isfinite(mean_dist):
        if fallback_dist_fn is None:
            raise RuntimeError("No finite truth-to-prediction assignment")
        mean_dist, perm = optimal_assignment(c_true, c_pred, fallback_dist_fn, K=len(c_true))
    if perm is None or not np.isfinite(mean_dist):
        raise RuntimeError("No finite truth-to-prediction assignment, including fallback")
    perm_tuple = tuple(int(x) for x in perm)
    return c_pred[np.asarray(perm_tuple, dtype=np.int64)], perm_tuple, float(mean_dist)


def paired_shift_distances(
    c_pred_aligned: np.ndarray,
    c_pred_shifted_aligned: np.ndarray,
    vertices_m: np.ndarray,
    graph: csr_matrix,
    geo_cache: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute source-wise Euclidean and mesh-geodesic displacement in mm."""
    a = np.asarray(c_pred_aligned, dtype=np.int64)
    b = np.asarray(c_pred_shifted_aligned, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"Prediction shapes differ: {a.shape} vs {b.shape}")
    vertices = np.asarray(vertices_m, dtype=np.float64)
    euc_mm = np.linalg.norm(vertices[a] - vertices[b], axis=1) * 1000.0

    cache = geo_cache if geo_cache is not None else {}
    geo_mm = np.empty(a.shape[0], dtype=np.float64)
    for i, (source, target) in enumerate(zip(a, b)):
        source_i = int(source)
        if source_i not in cache:
            cache[source_i] = np.asarray(
                dijkstra(graph, directed=False, indices=source_i), dtype=np.float64
            )
        geo_mm[i] = float(cache[source_i][int(target)])
    return euc_mm.astype(np.float64), geo_mm


def safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    numerator = float(numerator)
    return numerator / denominator if np.isfinite(numerator) else float("nan")


def summarize_distribution(values) -> dict[str, float | int]:
    raw = np.asarray(list(values), dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return {
            "n": 0,
            "n_nonfinite": int(raw.size),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(finite.size),
        "n_nonfinite": int(raw.size - finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def make_geodesic_dist_fn(
    graph: csr_matrix, cache: dict[int, np.ndarray]
) -> Callable[[int, int], float]:
    def distance(a: int, b: int) -> float:
        a = int(a)
        if a not in cache:
            cache[a] = np.asarray(
                dijkstra(graph, directed=False, indices=a), dtype=np.float64
            )
        return float(cache[a][int(b)])

    return distance


def make_euclidean_dist_fn(vertices_m: np.ndarray) -> Callable[[int, int], float]:
    vertices = np.asarray(vertices_m, dtype=np.float64)

    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(vertices[int(a)] - vertices[int(b)]) * 1000.0)

    return distance


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No metric rows to aggregate")
    output: dict[str, float] = {}
    for key in sorted(rows[0]):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        output[key] = float(np.mean(values)) if values.size else float("nan")
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_scenario(dataset_path: Path) -> tuple[str, float | None]:
    name = dataset_path.name.lower()
    if "dgnz" in name or "dg_nonzero" in name:
        return "dG != 0", 6.13
    if "dg0" in name:
        return "dG = 0", 5.85
    return "unknown", None


def validate_and_load_dataset(dataset_path: Path, pool: int) -> dict[str, Any]:
    data = np.load(dataset_path, allow_pickle=True)
    missing = REQUIRED_KEYS - set(data.files)
    if missing:
        raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")

    reference = np.load(REFERENCE_MESH, allow_pickle=True)
    for key in ("vertices", "edge_index", "edge_weight_mm"):
        if not np.array_equal(data[key], reference[key]):
            raise RuntimeError(f"Mesh compatibility check failed for {key}")

    ys = np.asarray(data["Ys"])
    ys_shifted = np.asarray(data["Ys_shifted"])
    centroids = np.asarray(data["centroids_idx"], dtype=np.int64)
    centroids_shifted = np.asarray(data["centroids_idx_shifted"], dtype=np.int64)
    delta_m = np.asarray(data["real_inputs_delta_m"], dtype=np.float64)
    n = ys.shape[0]
    if ys.shape != ys_shifted.shape:
        raise ValueError(f"Ys shapes differ: {ys.shape} vs {ys_shifted.shape}")
    if ys.shape[1:] != reference["Ys"].shape[1:]:
        raise ValueError(
            f"Model input shape mismatch: {ys.shape[1:]} vs {reference['Ys'].shape[1:]}"
        )
    if ys.shape[-1] % int(pool) != 0:
        raise ValueError(f"Time dimension {ys.shape[-1]} is not divisible by pool={pool}")
    if centroids.shape != (n, 3) or centroids_shifted.shape != (n, 3):
        raise ValueError("Expected exactly three original and shifted centroids per sample")
    if delta_m.shape != (n, 3):
        raise ValueError(f"Expected real_inputs_delta_m shape {(n, 3)}, got {delta_m.shape}")
    mean_delta_mm = float(np.mean(delta_m) * 1000.0)
    if not 3.0 <= mean_delta_mm <= 6.0:
        raise ValueError(
            f"Input delta unit checksum failed: mean={mean_delta_mm:.6f} mm, expected near 4.54 mm"
        )

    params_spatial = data["params_spatial"].item() if "params_spatial" in data.files else {}
    return {
        "Ys": ys,
        "Ys_shifted": ys_shifted,
        "centroids_idx": centroids,
        "centroids_idx_shifted": centroids_shifted,
        "real_inputs_delta_m": delta_m,
        "vertices": np.asarray(data["vertices"], dtype=np.float32),
        "edge_index": np.asarray(data["edge_index"]),
        "edge_weight_mm": np.asarray(data["edge_weight_mm"]),
        "params_spatial": params_spatial,
    }


def aggregate_theta_draws(theta_draws: np.ndarray, estimator: str) -> np.ndarray:
    """Apply a truth-free estimator and return one flattened 9-D prediction."""
    from estimator_audit import estimate_honest

    estimate, _diagnostics = estimate_honest(
        np.asarray(theta_draws).reshape(-1, 3, 3), estimator
    )
    return np.asarray(estimate, dtype=np.float32).reshape(-1)


def predict_posterior_mean(
    y_single: np.ndarray,
    draw_keys,
    sampler,
    pool_time,
    pool: int,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    vertices_m: np.ndarray,
    jnp,
    estimator: str = "posterior_mean_naive",
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_single, dtype=np.float32)[None, ...]
    pooled = pool_time(y, pool=pool).astype(np.float32)[0]
    x_norm = np.asarray(sampler(jnp.asarray(pooled), draw_keys), dtype=np.float32)
    theta_draws = x_norm * theta_std[None, :] + theta_mean[None, :]
    theta_mean_pred = aggregate_theta_draws(theta_draws, estimator)
    from metrics import project_coords_to_vertices

    c_pred = project_coords_to_vertices(theta_mean_pred.reshape(3, 3), vertices_m)
    return theta_mean_pred.astype(np.float32), c_pred.astype(np.int64)


def measure_inference_timing(
    *,
    sampler,
    y_single: np.ndarray,
    pool_time,
    pool: int,
    theta_mean: np.ndarray,
    theta_std: np.ndarray,
    vertices_m: np.ndarray,
    n_posterior: int,
    seed: int,
    repeats: int,
    estimator: str,
    jax,
    jnp,
) -> dict[str, Any]:
    y_raw = np.asarray(y_single, dtype=np.float32)
    pooled = pool_time(y_raw[None, ...], pool=pool).astype(np.float32)[0]
    compile_keys = jax.random.split(jax.random.PRNGKey(seed), n_posterior)
    start = time.perf_counter()
    compiled = sampler(jnp.asarray(pooled), compile_keys)
    compiled.block_until_ready()
    compile_seconds = time.perf_counter() - start

    repeat_keys = [
        jax.random.split(jax.random.PRNGKey(seed + 1 + i), n_posterior)
        for i in range(int(repeats))
    ]
    sampler_seconds: list[float] = []
    for keys in repeat_keys:
        start = time.perf_counter()
        value = sampler(jnp.asarray(pooled), keys)
        value.block_until_ready()
        sampler_seconds.append(time.perf_counter() - start)

    end_to_end_seconds: list[float] = []
    from metrics import project_coords_to_vertices

    for keys in repeat_keys:
        start = time.perf_counter()
        pooled_i = pool_time(y_raw[None, ...], pool=pool).astype(np.float32)[0]
        x_norm = np.asarray(sampler(jnp.asarray(pooled_i), keys), dtype=np.float32)
        theta = aggregate_theta_draws(
            x_norm * theta_std[None, :] + theta_mean[None, :], estimator
        )
        project_coords_to_vertices(theta.reshape(3, 3), vertices_m)
        end_to_end_seconds.append(time.perf_counter() - start)

    sampler_ms = np.asarray(sampler_seconds) * 1000.0
    end_to_end_ms = np.asarray(end_to_end_seconds) * 1000.0
    return {
        "compile_and_first_call_seconds": float(compile_seconds),
        "steady_sampler_ms_mean": float(np.mean(sampler_ms)),
        "steady_sampler_ms_median": float(np.median(sampler_ms)),
        "steady_sampler_ms_std": float(np.std(sampler_ms)),
        "steady_end_to_end_ms_mean": float(np.mean(end_to_end_ms)),
        "steady_end_to_end_ms_median": float(np.median(end_to_end_ms)),
        "steady_end_to_end_ms_std": float(np.std(end_to_end_ms)),
        "repeats": int(repeats),
        "n_posterior": int(n_posterior),
    }


def write_report_md(path: Path, summary: dict[str, Any]) -> None:
    euc = summary["output_delta_euc_mm_by_sample"]
    geo = summary["output_delta_geo_mm_by_sample"]
    inp = summary["input_delta_euc_mm_by_sample"]
    ratio = summary["ratio_euc_by_sample"]
    rap = summary.get("rap_music_delta_euc_mm")
    lines = [
        "# Paired CFM centroid-shift stability",
        "",
        "## Main result",
        "",
        "| Scenario | Input ΔX Euclidean mean, mm | RAP MUSIC ΔX′, mm | CFM ΔX′ Euclidean mean, mm | CFM ΔX′ geodesic mean, mm |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| `{summary['scenario']}` | {inp['mean']:.4f} | {rap if rap is not None else 'n/a'} | {euc['mean']:.4f} | {geo['mean']:.4f} |",
        "",
        "The Euclidean CFM number is the directly comparable quantity because RAP MUSIC and `real_inputs_delta_m` use 3-D Euclidean distance. Geodesic displacement is supplementary.",
        "",
        "## ΔX′ distributions over per-sample means",
        "",
        "| Metric | Mean | Median | Std | P25 | P75 | Max | N finite |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Euclidean, mm | {euc['mean']:.4f} | {euc['median']:.4f} | {euc['std']:.4f} | {euc['p25']:.4f} | {euc['p75']:.4f} | {euc['max']:.4f} | {euc['n']} |",
        f"| Geodesic, mm | {geo['mean']:.4f} | {geo['median']:.4f} | {geo['std']:.4f} | {geo['p25']:.4f} | {geo['p75']:.4f} | {geo['max']:.4f} | {geo['n']} |",
        f"| ΔX′/ΔX (Euclidean) | {ratio['mean']:.4f} | {ratio['median']:.4f} | {ratio['std']:.4f} | {ratio['p25']:.4f} | {ratio['p75']:.4f} | {ratio['max']:.4f} | {ratio['n']} |",
        "",
        "## Localization sanity checks",
        "",
        "| Input | MAE geo, mm | MAE euc, mm | Hit geo @10 | Hit geo @15 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Original Ys | {summary['baseline_original']['mae_geo_mm']:.4f} | {summary['baseline_original']['mae_euc_mm']:.4f} | {summary['baseline_original']['hit_geo@10mm']:.4f} | {summary['baseline_original']['hit_geo@15mm']:.4f} |",
        f"| Shifted Ys | {summary['baseline_shifted']['mae_geo_mm']:.4f} | {summary['baseline_shifted']['mae_euc_mm']:.4f} | {summary['baseline_shifted']['hit_geo@10mm']:.4f} | {summary['baseline_shifted']['hit_geo@15mm']:.4f} |",
        "",
        "## Permutation resolution",
        "",
        "For each pair, prediction A is assigned to `centroids_idx` and prediction B is independently assigned to `centroids_idx_shifted` by minimum mesh-geodesic distance (Euclidean fallback only if the mesh distance is non-finite). The dataset's true source index `j` then anchors the two assignments, and ΔX′ is measured between the two predictions aligned to that same `j`. Model output slots are never compared directly.",
        "",
        "The two posterior means use identical base-noise draw keys (common random numbers) for each pair. This isolates the response to changing `Y` from Monte Carlo variation due solely to different posterior draws.",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{summary['dataset_npz']}`",
        f"- Model: `{summary['model_dir']}`",
        f"- Samples: {summary['n_evaluated']}",
        f"- Posterior draws: {summary['n_posterior']}",
        f"- Estimator: `{summary['estimator']}`",
        f"- ODE steps: {summary['ode_steps']}",
        f"- Pool: {summary['pool']}",
        f"- Seed: {summary['seed']}",
        f"- JAX devices: `{summary['jax_devices']}`",
        "",
        "## Limitations",
        "",
        f"- Ratios are undefined for {ratio['n_nonfinite']} sample(s) with zero mean input ΔX; those rows remain in displacement summaries but are excluded from ratio aggregation.",
        f"- The posterior estimate uses {summary['n_posterior']} finite Monte Carlo draws and is not an exact expectation.",
        "- Assignment is truth-anchored and therefore measures controlled experimental stability; it is not an unsupervised correspondence method for real data without ground truth.",
    ]
    timing = summary.get("timing")
    if timing:
        lines.extend(
            [
                "",
                "## Timing for one localization",
                "",
                f"- Compile + first sampler call: {timing['compile_and_first_call_seconds']:.4f} s",
                f"- Steady ODE sampler: {timing['steady_sampler_ms_mean']:.3f} ± {timing['steady_sampler_ms_std']:.3f} ms",
                f"- Steady end-to-end (pooling, sampler, posterior mean, mesh projection): {timing['steady_end_to_end_ms_mean']:.3f} ± {timing['steady_end_to_end_ms_std']:.3f} ms",
                f"- Repeats: {timing['repeats']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from estimator_audit import HONEST_ESTIMATORS

    parser = argparse.ArgumentParser(description="Paired CFM stability under true centroid shifts")
    parser.add_argument("--dataset_npz", required=True)
    parser.add_argument("--gpu", default="1", help="CUDA device index; ignored when JAX_PLATFORMS=cpu")
    parser.add_argument("--n_posterior", type=int, default=0, help="0 uses the model's result.json value")
    parser.add_argument(
        "--estimator",
        choices=HONEST_ESTIMATORS,
        default="posterior_mean_naive",
        help="Truth-free posterior decoder",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--timing_repeats", type=int, default=10)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if str(MESH_ROOT) not in sys.path:
        sys.path.insert(0, str(MESH_ROOT))
    if str(GDG_ROOT) not in sys.path:
        sys.path.insert(0, str(GDG_ROOT))

    import jax
    import jax.numpy as jnp

    from eval_cnn_checkpoints_25march import load_checkpoint_runtime
    from eval_posterior_mean_100 import build_spec_from_model_dir, make_multi_sampler
    from metrics import build_graph_from_edges, compute_metrics_one
    from optuna_sweep_cnn import pool_time

    dataset_path = Path(args.dataset_npz).resolve()
    model_dir = DEFAULT_MODEL_DIR.resolve()
    spec = build_spec_from_model_dir(model_dir)
    runtime = load_checkpoint_runtime(spec)
    pool = int(runtime["pool"])
    ode_steps = int(runtime["ode_steps"])
    n_posterior = int(args.n_posterior) if args.n_posterior > 0 else int(spec["n_post_samples"])
    data = validate_and_load_dataset(dataset_path, pool=pool)
    n_total = int(data["Ys"].shape[0])
    n_eval = min(n_total, int(args.limit)) if args.limit > 0 else n_total
    indices = np.arange(n_eval, dtype=np.int64)

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (MESH_ROOT / "reports" / f"shift_paired_{dataset_path.stem}_{timestamp()}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[paired] dataset={dataset_path}")
    print(f"[paired] model_dir={model_dir}")
    print(f"[paired] JAX devices={jax.devices()}")
    print(f"[paired] n_eval={n_eval}, n_posterior={n_posterior}, ode_steps={ode_steps}, pool={pool}")

    vertices = data["vertices"]
    graph = build_graph_from_edges(
        data["edge_index"], data["edge_weight_mm"], vertices.shape[0]
    )
    theta_mean = np.asarray(runtime["theta_mean"], dtype=np.float32)
    theta_std = np.asarray(runtime["theta_std"], dtype=np.float32)
    sampler = make_multi_sampler(runtime["model"], theta_dim=theta_mean.shape[0], steps=ode_steps)

    timing = None
    if args.timing:
        timing = measure_inference_timing(
            sampler=sampler,
            y_single=data["Ys"][0],
            pool_time=pool_time,
            pool=pool,
            theta_mean=theta_mean,
            theta_std=theta_std,
            vertices_m=vertices,
            n_posterior=n_posterior,
            seed=int(args.seed) + 900000,
            repeats=int(args.timing_repeats),
            estimator=args.estimator,
            jax=jax,
            jnp=jnp,
        )
        print(f"[paired] timing={timing}")

    params_spatial = data["params_spatial"]
    r_mm = params_spatial.get("R_mm", [10.0, 8.0, 10.0])
    a0 = params_spatial.get("A0", None)
    geo_assignment_cache: dict[int, np.ndarray] = {}
    geo_pair_cache: dict[int, np.ndarray] = {}
    geo_dist = make_geodesic_dist_fn(graph, geo_assignment_cache)
    euc_dist = make_euclidean_dist_fn(vertices)

    key = jax.random.PRNGKey(int(args.seed))
    rows: list[dict[str, Any]] = []
    baseline_a_rows: list[dict[str, float]] = []
    baseline_b_rows: list[dict[str, float]] = []
    theta_pred_a = np.empty((n_eval, 9), dtype=np.float32)
    theta_pred_b = np.empty((n_eval, 9), dtype=np.float32)
    c_pred_a_all = np.empty((n_eval, 3), dtype=np.int64)
    c_pred_b_all = np.empty((n_eval, 3), dtype=np.int64)
    c_aligned_a_all = np.empty((n_eval, 3), dtype=np.int64)
    c_aligned_b_all = np.empty((n_eval, 3), dtype=np.int64)

    started = time.perf_counter()
    for out_i, sample_index in enumerate(indices):
        key, sample_key = jax.random.split(key)
        draw_keys = jax.random.split(sample_key, n_posterior)
        theta_a, c_pred_a = predict_posterior_mean(
            data["Ys"][sample_index], draw_keys, sampler, pool_time, pool,
            theta_mean, theta_std, vertices, jnp, estimator=args.estimator,
        )
        theta_b, c_pred_b = predict_posterior_mean(
            data["Ys_shifted"][sample_index], draw_keys, sampler, pool_time, pool,
            theta_mean, theta_std, vertices, jnp, estimator=args.estimator,
        )
        c_true_a = data["centroids_idx"][sample_index]
        c_true_b = data["centroids_idx_shifted"][sample_index]
        aligned_a, perm_a, assignment_a = align_prediction_to_truth(
            c_true_a, c_pred_a, geo_dist, fallback_dist_fn=euc_dist
        )
        aligned_b, perm_b, assignment_b = align_prediction_to_truth(
            c_true_b, c_pred_b, geo_dist, fallback_dist_fn=euc_dist
        )
        output_euc_mm, output_geo_mm = paired_shift_distances(
            aligned_a, aligned_b, vertices, graph, geo_cache=geo_pair_cache
        )
        input_euc_mm = data["real_inputs_delta_m"][sample_index] * 1000.0

        metric_a = compute_metrics_one(
            C_true=c_true_a,
            C_pred=c_pred_a,
            vertices_m=vertices,
            knn_graph=graph,
            R_mm=r_mm,
            A0=a0,
            thresholds_mm=(5, 10, 15, 20, 30),
        )
        metric_b = compute_metrics_one(
            C_true=c_true_b,
            C_pred=c_pred_b,
            vertices_m=vertices,
            knn_graph=graph,
            R_mm=r_mm,
            A0=a0,
            thresholds_mm=(5, 10, 15, 20, 30),
        )
        baseline_a_rows.append({k: float(v) for k, v in metric_a.items()})
        baseline_b_rows.append({k: float(v) for k, v in metric_b.items()})

        input_mean = float(np.mean(input_euc_mm))
        output_euc_mean = float(np.mean(output_euc_mm))
        output_geo_mean = float(np.mean(output_geo_mm))
        row: dict[str, Any] = {
            "sample_index": int(sample_index),
            "input_delta_euc_mean_mm": input_mean,
            "output_delta_euc_mean_mm": output_euc_mean,
            "output_delta_geo_mean_mm": output_geo_mean,
            "ratio_euc_mean": safe_ratio(output_euc_mean, input_mean),
            "ratio_geo_over_euc_input_mean": safe_ratio(output_geo_mean, input_mean),
            "assignment_a_mean_geo_mm": float(assignment_a),
            "assignment_b_mean_geo_mm": float(assignment_b),
            "perm_a": ";".join(map(str, perm_a)),
            "perm_b": ";".join(map(str, perm_b)),
        }
        for source in range(3):
            row[f"true_a_{source}"] = int(c_true_a[source])
            row[f"true_b_{source}"] = int(c_true_b[source])
            row[f"pred_a_aligned_{source}"] = int(aligned_a[source])
            row[f"pred_b_aligned_{source}"] = int(aligned_b[source])
            row[f"input_delta_euc_src{source}_mm"] = float(input_euc_mm[source])
            row[f"output_delta_euc_src{source}_mm"] = float(output_euc_mm[source])
            row[f"output_delta_geo_src{source}_mm"] = float(output_geo_mm[source])
            row[f"ratio_euc_src{source}"] = safe_ratio(output_euc_mm[source], input_euc_mm[source])
        for metric_name, value in metric_a.items():
            row[f"original_{metric_name}"] = float(value)
        for metric_name, value in metric_b.items():
            row[f"shifted_{metric_name}"] = float(value)
        rows.append(row)

        theta_pred_a[out_i] = theta_a
        theta_pred_b[out_i] = theta_b
        c_pred_a_all[out_i] = c_pred_a
        c_pred_b_all[out_i] = c_pred_b
        c_aligned_a_all[out_i] = aligned_a
        c_aligned_b_all[out_i] = aligned_b
        if (out_i + 1) % 10 == 0 or out_i + 1 == n_eval:
            elapsed = time.perf_counter() - started
            print(f"[paired] {out_i + 1}/{n_eval} samples, elapsed={elapsed:.1f}s", flush=True)

    scenario, rap_music = infer_scenario(dataset_path)
    summary = {
        "scenario": scenario,
        "dataset_npz": str(dataset_path),
        "model_dir": str(model_dir),
        "n_total": n_total,
        "n_evaluated": n_eval,
        "n_posterior": n_posterior,
        "estimator": args.estimator,
        "ode_steps": ode_steps,
        "pool": pool,
        "seed": int(args.seed),
        "gpu_argument": str(args.gpu),
        "jax_platform": jax.default_backend(),
        "jax_devices": str(jax.devices()),
        "common_random_numbers": True,
        "assignment_metric": "mesh_geodesic_with_euclidean_fallback",
        "rap_music_delta_euc_mm": rap_music,
        "input_delta_euc_mm_by_sample": summarize_distribution(
            row["input_delta_euc_mean_mm"] for row in rows
        ),
        "input_delta_euc_mm_by_source": summarize_distribution(
            row[f"input_delta_euc_src{source}_mm"] for row in rows for source in range(3)
        ),
        "output_delta_euc_mm_by_sample": summarize_distribution(
            row["output_delta_euc_mean_mm"] for row in rows
        ),
        "output_delta_euc_mm_by_source": summarize_distribution(
            row[f"output_delta_euc_src{source}_mm"] for row in rows for source in range(3)
        ),
        "output_delta_geo_mm_by_sample": summarize_distribution(
            row["output_delta_geo_mean_mm"] for row in rows
        ),
        "output_delta_geo_mm_by_source": summarize_distribution(
            row[f"output_delta_geo_src{source}_mm"] for row in rows for source in range(3)
        ),
        "ratio_euc_by_sample": summarize_distribution(row["ratio_euc_mean"] for row in rows),
        "ratio_geo_by_sample": summarize_distribution(
            row["ratio_geo_over_euc_input_mean"] for row in rows
        ),
        "baseline_original": aggregate_metric_rows(baseline_a_rows),
        "baseline_shifted": aggregate_metric_rows(baseline_b_rows),
        "timing": timing,
        "wall_seconds": float(time.perf_counter() - started),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(out_dir / "per_sample_metrics.csv", rows)
    np.savez_compressed(
        out_dir / "paired_predictions.npz",
        sample_indices=indices,
        theta_pred=theta_pred_a,
        theta_pred_shifted=theta_pred_b,
        centroids_pred=c_pred_a_all,
        centroids_pred_shifted=c_pred_b_all,
        centroids_pred_aligned=c_aligned_a_all,
        centroids_pred_shifted_aligned=c_aligned_b_all,
    )
    write_report_md(out_dir / "report.md", summary)
    print(f"[paired] wrote {out_dir}")
    print(
        "[paired] result "
        f"input={summary['input_delta_euc_mm_by_source']['mean']:.4f} mm, "
        f"output_euc={summary['output_delta_euc_mm_by_sample']['mean']:.4f} mm, "
        f"output_geo={summary['output_delta_geo_mm_by_sample']['mean']:.4f} mm, "
        f"baseline_geo={summary['baseline_original']['mae_geo_mm']:.4f} mm"
    )


if __name__ == "__main__":
    main()
