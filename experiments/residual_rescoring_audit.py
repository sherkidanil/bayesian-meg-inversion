#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
GDG_ROOT = PROJECT_ROOT / "new_version_mesh_experiments_g_dg"
DEFAULT_MODEL_DIR = MODULE_ROOT / "best-model-folder"


SCENARIO_KEYS = (
    "Ys",
    "Ys_shifted",
    "centroids_idx",
    "centroids_idx_shifted",
    "vertices",
    "edge_index",
    "edge_weight_mm",
    "params_spatial",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_smooth_selection(
    path: str | Path, expected_model_key: str
) -> dict[str, Any]:
    """Load the pre-dev frozen quadratic Gibbs choice without outcome access."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("mode") != "dev_tune":
        raise ValueError("frozen selection must come from dev_tune mode")
    if payload.get("model_key") != expected_model_key:
        raise ValueError("frozen selection model key does not match runtime")
    selected = dict(payload.get("selected", {}))
    if selected.get("estimator") != "gibbs":
        raise ValueError("frozen selection must use the Gibbs estimator")
    if selected.get("score_form") != "quadratic":
        raise ValueError("frozen selection must use the pre-registered quadratic score")
    tau = float(selected.get("tau", float("nan")))
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("frozen selection tau must be finite and positive")
    selected["tau"] = tau
    return selected


def frozen_smooth_coordinates(
    theta_draws: np.ndarray,
    residuals: np.ndarray,
    selection: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen truth-free decoder to one posterior draw set."""
    from residual_rescoring import residual_weighted_mean

    return residual_weighted_mean(
        theta_draws,
        residuals,
        tau=float(selection["tau"]),
        score_form=str(selection["score_form"]),
    )


def validate_localization_g(value: np.ndarray) -> np.ndarray:
    leadfield = np.asarray(value, dtype=np.float64)
    if leadfield.ndim != 2:
        raise ValueError(f"localization_G must be a 2-D matrix, got {leadfield.shape}")
    if not np.all(np.isfinite(leadfield)):
        raise ValueError("localization_G contains non-finite values")
    return leadfield


def _load_constant_anna_g(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as payload:
        matrices = np.asarray(payload["G_for_simulation"])
    if matrices.ndim != 3 or matrices.shape[0] == 0:
        raise ValueError("anna_dg0 G_for_simulation must have shape (N, M, V)")
    reference = np.asarray(matrices[0], dtype=np.float64)
    if not np.allclose(matrices, reference[None, ...], rtol=0.0, atol=0.0):
        raise ValueError("anna_dg0 G_for_simulation is not constant across samples")
    return validate_localization_g(reference)


def load_anna_audit_inputs(
    anna_dg0_npz: str | Path, scenario_npz: str | Path
) -> dict[str, Any]:
    """Load the one allowed localization G and scenario data without its simulation G."""
    localization_g = _load_constant_anna_g(Path(anna_dg0_npz))
    with np.load(Path(scenario_npz), allow_pickle=True) as payload:
        missing = set(SCENARIO_KEYS) - set(payload.files)
        if missing:
            raise KeyError(f"scenario is missing required keys: {sorted(missing)}")
        scenario: dict[str, Any] = {}
        for key in SCENARIO_KEYS:
            value = payload[key]
            scenario[key] = value.item() if key == "params_spatial" else np.asarray(value)
    if scenario["Ys"].shape[1] != localization_g.shape[0]:
        raise ValueError("scenario sensor dimension does not match anna_dg0 localization G")
    if scenario["vertices"].shape[0] != localization_g.shape[1]:
        raise ValueError("scenario mesh does not match anna_dg0 localization G")
    return {"localization_G": localization_g, "scenario": scenario}


def mean_assignment_error(
    true_vertices: np.ndarray,
    predicted_vertices: np.ndarray,
    pairwise_distance: np.ndarray,
) -> float:
    truth = np.asarray(true_vertices, dtype=np.int64)
    prediction = np.asarray(predicted_vertices, dtype=np.int64)
    distance = np.asarray(pairwise_distance, dtype=np.float64)
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError("true and predicted vertices must be equal-length vectors")
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("pairwise_distance must be square")
    best = float("inf")
    for order in permutations(range(len(prediction))):
        values = distance[truth, prediction[np.asarray(order, dtype=np.int64)]]
        candidate = float(np.mean(values))
        if candidate < best:
            best = candidate
    return best


def rank_nonfinite_worst(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("values must be nonempty")
    finite = np.isfinite(array)
    surrogate = np.empty_like(array)
    if np.any(finite):
        surrogate[finite] = array[finite]
        maximum = float(np.max(array[finite]))
        surrogate[~finite] = np.nextafter(maximum, np.inf)
    else:
        surrogate.fill(0.0)
    return np.asarray(rankdata(surrogate, method="average"), dtype=np.float64)


def _rank_correlation(left_ranks: np.ndarray, right_ranks: np.ndarray) -> float:
    left = np.asarray(left_ranks, dtype=np.float64).reshape(-1)
    right = np.asarray(right_ranks, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size < 2:
        raise ValueError("rank arrays must have equal shape and at least two values")
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman_with_nonfinite_worst(left: np.ndarray, right: np.ndarray) -> float:
    return _rank_correlation(rank_nonfinite_worst(left), rank_nonfinite_worst(right))


def pooled_within_sample_rank_spearman(
    residuals_by_sample: Iterable[np.ndarray], errors_by_sample: Iterable[np.ndarray]
) -> float:
    residual_groups = list(residuals_by_sample)
    error_groups = list(errors_by_sample)
    if len(residual_groups) != len(error_groups) or not residual_groups:
        raise ValueError("residual and error groups must be nonempty and equal length")
    residual_ranks: list[np.ndarray] = []
    error_ranks: list[np.ndarray] = []
    for residuals, errors in zip(residual_groups, error_groups):
        residual_array = np.asarray(residuals).reshape(-1)
        error_array = np.asarray(errors).reshape(-1)
        if residual_array.shape != error_array.shape:
            raise ValueError("each sample must have equal residual and error counts")
        residual_ranks.append(rank_nonfinite_worst(residual_array))
        error_ranks.append(rank_nonfinite_worst(error_array))
    return _rank_correlation(np.concatenate(residual_ranks), np.concatenate(error_ranks))


def draw_seed_for_sample(base_seed: int, sample_index: int) -> int:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    state = np.random.SeedSequence([int(base_seed), int(sample_index)]).generate_state(
        1, dtype=np.uint32
    )
    return int(state[0])


def aggregate_spearman_diagnostics(
    residuals_by_sample: Iterable[np.ndarray], errors_by_sample: Iterable[np.ndarray]
) -> dict[str, float | int]:
    residual_groups = [np.asarray(value, dtype=np.float64) for value in residuals_by_sample]
    error_groups = [np.asarray(value, dtype=np.float64) for value in errors_by_sample]
    if len(residual_groups) != len(error_groups) or not residual_groups:
        raise ValueError("residual and error groups must be nonempty and equal length")
    per_sample = np.asarray(
        [
            spearman_with_nonfinite_worst(residuals, errors)
            for residuals, errors in zip(residual_groups, error_groups)
        ],
        dtype=np.float64,
    )
    finite = per_sample[np.isfinite(per_sample)]
    return {
        "n_samples": int(len(per_sample)),
        "n_finite_per_sample": int(finite.size),
        "per_sample_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "per_sample_median": float(np.median(finite)) if finite.size else float("nan"),
        "pooled_raw": spearman_with_nonfinite_worst(
            np.concatenate(residual_groups), np.concatenate(error_groups)
        ),
        "pooled_within_sample_rank": pooled_within_sample_rank_spearman(
            residual_groups, error_groups
        ),
    }


def _mean_assignment_cost(cost: np.ndarray) -> float:
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("assignment cost must be square")
    best = float("inf")
    for order in permutations(range(matrix.shape[1])):
        candidate = float(np.mean(matrix[np.arange(matrix.shape[0]), order]))
        if candidate < best:
            best = candidate
    return best


def draw_errors(
    true_vertices: np.ndarray,
    predicted_vertices: np.ndarray,
    vertices_m: np.ndarray,
    full_geo_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(true_vertices, dtype=np.int64)
    predictions = np.asarray(predicted_vertices, dtype=np.int64)
    vertices = np.asarray(vertices_m, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1:] != truth.shape:
        raise ValueError("predicted_vertices must have shape (N_draws, N_sources)")
    euc = np.empty(len(predictions), dtype=np.float64)
    geo = np.empty(len(predictions), dtype=np.float64)
    for index, prediction in enumerate(predictions):
        euc_cost = (
            np.linalg.norm(
                vertices[truth][:, None, :] - vertices[prediction][None, :, :], axis=2
            )
            * 1000.0
        )
        euc[index] = _mean_assignment_cost(euc_cost)
        geo[index] = mean_assignment_error(truth, prediction, full_geo_mm)
    return euc, geo


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    raw = np.asarray(list(values), dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    return {
        "n": int(raw.size),
        "n_finite": int(finite.size),
        "n_nonfinite": int(raw.size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "std": float(np.std(finite)) if finite.size else float("nan"),
        "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
    }


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    result: dict[str, Any] = {}
    for key in sorted({name for row in rows for name in row}):
        result[key] = _distribution(float(row[key]) for row in rows if key in row)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def build_commit_provenance(
    run_start_git_commit: str, artifact_write_git_commit: str
) -> dict[str, str | bool]:
    return {
        "run_start_git_commit": str(run_start_git_commit),
        "artifact_write_git_commit": str(artifact_write_git_commit),
        "repository_changed_during_run": (
            str(run_start_git_commit) != str(artifact_write_git_commit)
        ),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Truth-free sensor-residual posterior rescoring",
        "",
        f"- Scenario: `{summary['scenario_npz']}`",
        f"- Localization G: `{summary['localization_g_source']}` element 0, verified constant",
        "- Scenario `G_for_simulation` was not loaded or used by the estimator.",
        f"- Samples: {summary['n_samples']}; draws per input: {summary['n_draws']}",
        "- `residual_selected`, `single_draw`, `posterior_mean_naive`, and `posterior_medoid` are truth-free.",
        "- Oracle rows and draw-error/Spearman columns are diagnostic-only.",
        "",
        "## Honest and diagnostic localization metrics",
        "",
        "| Input | Estimator | Honest | MAE euc, mm | MAE geo, mm | Geo non-finite |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for input_kind, estimators in summary["metrics"].items():
        for estimator, values in estimators.items():
            euc = values.get("mae_euc_mm", {})
            geo = values.get("mae_geo_mm", {})
            lines.append(
                f"| {input_kind} | `{estimator}` | "
                f"{'yes' if not estimator.startswith('oracle_') else 'no'} | "
                f"{euc.get('mean', float('nan')):.4f} | "
                f"{geo.get('mean', float('nan')):.4f} | {geo.get('n_nonfinite', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Residual/error rank association",
            "",
            "| Input | Error | Mean per-sample Spearman | Pooled raw | Pooled within-sample rank |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for input_kind, correlations in summary["correlations"].items():
        for error_name, values in correlations.items():
            lines.append(
                f"| {input_kind} | {error_name} | {values['per_sample_mean']:.4f} | "
                f"{values['pooled_raw']:.4f} | {values['pooled_within_sample_rank']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Residual selection uses only Y, the persisted Anna localization G, mesh geometry, and spatial parameters. Truth is used only after selection for this audit. Absolute CFM MAE on Anna-domain data remains a cross-domain number and must not be compared directly with matched-G RAP MUSIC until the in-domain control is available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truth-free sensor-residual posterior draw audit on Anna data"
    )
    parser.add_argument("--anna_dg0_npz", required=True)
    parser.add_argument("--scenario_npz", required=True)
    parser.add_argument("--model_dir", default="")
    parser.add_argument(
        "--model_key",
        default="",
        help="Explicit canonical identity when result.json uses an internal candidate_id",
    )
    parser.add_argument(
        "--selection",
        default="",
        help="Frozen dev selection.json; adds frozen_smooth without retuning",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_draws", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--gpu", default="1")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_start_git_commit = _git_commit()
    if args.n_draws <= 0:
        raise ValueError("n_draws must be positive")
    if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for path in (MODULE_ROOT, GDG_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import jax
    import jax.numpy as jnp

    from estimator_audit import estimate_honest
    from eval_cnn_checkpoints_25march import (
        compute_metrics_one_precomputed,
        load_checkpoint_runtime,
    )
    from eval_posterior_mean_100 import build_spec_from_model_dir, make_multi_sampler
    from metrics import build_graph_from_edges, project_coords_to_vertices
    from optuna_sweep_cnn import pool_time
    from residual_rescoring import residual_rescore_draws

    anna_path = Path(args.anna_dg0_npz).resolve()
    scenario_path = Path(args.scenario_npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_anna_audit_inputs(anna_path, scenario_path)
    localization_g = loaded["localization_G"]
    data = loaded["scenario"]
    n_total = int(len(data["Ys"]))
    n_samples = min(n_total, int(args.limit)) if args.limit > 0 else n_total
    if n_samples <= 0:
        raise ValueError("no samples selected")

    model_dir = Path(args.model_dir).resolve() if args.model_dir else DEFAULT_MODEL_DIR.resolve()
    spec = build_spec_from_model_dir(model_dir)
    if args.model_key:
        spec["model_key"] = str(args.model_key)
    runtime = load_checkpoint_runtime(spec)
    frozen_selection = (
        load_frozen_smooth_selection(args.selection, str(runtime["model_key"]))
        if args.selection
        else None
    )
    pool = int(runtime["pool"])
    ode_steps = int(runtime["ode_steps"])
    theta_mean = np.asarray(runtime["theta_mean"], dtype=np.float32)
    theta_std = np.asarray(runtime["theta_std"], dtype=np.float32)
    sampler = make_multi_sampler(runtime["model"], theta_dim=9, steps=ode_steps)

    vertices = np.asarray(data["vertices"], dtype=np.float32)
    graph = build_graph_from_edges(
        data["edge_index"], data["edge_weight_mm"], len(vertices)
    )
    print("[rescore] precomputing all-pairs geodesic distances", flush=True)
    full_geo_mm = np.asarray(dijkstra(graph, directed=False), dtype=np.float64)
    params_spatial = data["params_spatial"]
    radii = params_spatial.get("R_mm", [10.0, 8.0, 10.0])
    amplitudes = params_spatial.get("A0", None)

    per_draw_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    metric_groups: dict[str, dict[str, list[dict[str, float]]]] = {
        kind: {} for kind in ("original", "shifted")
    }
    residual_groups: dict[str, list[np.ndarray]] = {kind: [] for kind in metric_groups}
    euc_groups: dict[str, list[np.ndarray]] = {kind: [] for kind in metric_groups}
    geo_groups: dict[str, list[np.ndarray]] = {kind: [] for kind in metric_groups}
    selected_theta = np.empty((n_samples, 2, 3, 3), dtype=np.float32)
    selected_centroids = np.empty((n_samples, 2, 3), dtype=np.int64)
    selected_indices = np.empty((n_samples, 2), dtype=np.int64)
    smooth_theta = (
        np.empty((n_samples, 2, 3, 3), dtype=np.float32)
        if frozen_selection is not None
        else None
    )
    smooth_centroids = (
        np.empty((n_samples, 2, 3), dtype=np.int64)
        if frozen_selection is not None
        else None
    )

    started = time.perf_counter()
    for sample_index in range(n_samples):
        seed = draw_seed_for_sample(args.seed, sample_index)
        draw_keys = jax.random.split(jax.random.PRNGKey(seed), int(args.n_draws))
        for kind_index, (input_kind, y_key, truth_key) in enumerate(
            (
                ("original", "Ys", "centroids_idx"),
                ("shifted", "Ys_shifted", "centroids_idx_shifted"),
            )
        ):
            y = np.asarray(data[y_key][sample_index], dtype=np.float32)
            pooled = pool_time(y[None, ...], pool=pool).astype(np.float32)[0]
            normalized = np.asarray(
                sampler(jnp.asarray(pooled), draw_keys), dtype=np.float32
            )
            theta_draws = (
                normalized * theta_std[None, :] + theta_mean[None, :]
            ).reshape(int(args.n_draws), 3, 3)
            residuals, residual_index = residual_rescore_draws(
                theta_draws,
                y,
                localization_g,
                vertices,
                graph,
                params_spatial,
            )
            predicted = np.asarray(
                [project_coords_to_vertices(draw, vertices) for draw in theta_draws],
                dtype=np.int64,
            )
            truth = np.asarray(data[truth_key][sample_index], dtype=np.int64)
            euc_errors, geo_errors = draw_errors(truth, predicted, vertices, full_geo_mm)
            residual_groups[input_kind].append(residuals)
            euc_groups[input_kind].append(euc_errors)
            geo_groups[input_kind].append(geo_errors)
            corr_euc = spearman_with_nonfinite_worst(residuals, euc_errors)
            corr_geo = spearman_with_nonfinite_worst(residuals, geo_errors)

            oracle_euc_index = int(np.argmin(euc_errors))
            finite_geo = np.where(np.isfinite(geo_errors), geo_errors, np.inf)
            oracle_geo_index = int(np.argmin(finite_geo))
            for draw_index in range(int(args.n_draws)):
                per_draw_rows.append(
                    {
                        "sample_index": sample_index,
                        "input_kind": input_kind,
                        "draw_index": draw_index,
                        "residual": float(residuals[draw_index]),
                        "error_euc_mm": float(euc_errors[draw_index]),
                        "error_geo_mm": float(geo_errors[draw_index]),
                        "error_geo_finite": bool(np.isfinite(geo_errors[draw_index])),
                        "is_residual_selected": draw_index == residual_index,
                        "is_oracle_euc": draw_index == oracle_euc_index,
                        "is_oracle_geo": draw_index == oracle_geo_index,
                    }
                )

            honest_coords = {
                "residual_selected": theta_draws[residual_index],
                "single_draw": theta_draws[0],
                "posterior_mean_naive": estimate_honest(
                    theta_draws, "posterior_mean_naive"
                )[0],
                "posterior_medoid": estimate_honest(theta_draws, "posterior_medoid")[0],
            }
            smooth_weights = None
            if frozen_selection is not None:
                smooth_coords, smooth_weights = frozen_smooth_coordinates(
                    theta_draws, residuals, frozen_selection
                )
                honest_coords["frozen_smooth"] = smooth_coords
            diagnostic_coords = {
                "oracle_best_euc": theta_draws[oracle_euc_index],
                "oracle_best_geo": theta_draws[oracle_geo_index],
            }
            for estimator, coords in {**honest_coords, **diagnostic_coords}.items():
                prediction = project_coords_to_vertices(coords, vertices)
                metrics = compute_metrics_one_precomputed(
                    C_true=truth,
                    C_pred=prediction,
                    vertices_m=vertices,
                    full_geo_mm=full_geo_mm,
                    R_mm=radii,
                    A0=amplitudes,
                    thresholds_mm=(5, 10, 15, 20, 30),
                )
                clean = {name: float(value) for name, value in metrics.items()}
                metric_groups[input_kind].setdefault(estimator, []).append(clean)
                row: dict[str, Any] = {
                    "sample_index": sample_index,
                    "input_kind": input_kind,
                    "estimator": estimator,
                    "honest": estimator in honest_coords,
                    "spearman_residual_euc": float(corr_euc),
                    "spearman_residual_geo": float(corr_geo),
                    "residual_selected_index": residual_index,
                    "oracle_best_euc_index": oracle_euc_index,
                    "oracle_best_geo_index": oracle_geo_index,
                }
                if estimator == "frozen_smooth" and smooth_weights is not None:
                    row["effective_sample_size"] = float(
                        1.0 / np.sum(np.square(smooth_weights))
                    )
                row.update(clean)
                per_sample_rows.append(row)

            selected_theta[sample_index, kind_index] = theta_draws[residual_index]
            selected_centroids[sample_index, kind_index] = predicted[residual_index]
            selected_indices[sample_index, kind_index] = residual_index
            if frozen_selection is not None:
                assert smooth_theta is not None and smooth_centroids is not None
                smooth_theta[sample_index, kind_index] = smooth_coords
                smooth_centroids[sample_index, kind_index] = project_coords_to_vertices(
                    smooth_coords, vertices
                )
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == n_samples:
            print(
                f"[rescore] {sample_index + 1}/{n_samples}, elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    correlations = {
        kind: {
            "euclidean": aggregate_spearman_diagnostics(
                residual_groups[kind], euc_groups[kind]
            ),
            "geodesic": aggregate_spearman_diagnostics(
                residual_groups[kind], geo_groups[kind]
            ),
        }
        for kind in metric_groups
    }
    commit_provenance = build_commit_provenance(
        run_start_git_commit, _git_commit()
    )
    summary = {
        "scenario_npz": str(scenario_path),
        "localization_g_source": str(anna_path),
        "localization_g_policy": "anna_dg0 element zero verified constant; scenario G ignored",
        "model_dir": str(model_dir),
        "model_key": str(runtime["model_key"]),
        "n_total": n_total,
        "n_samples": n_samples,
        "n_draws": int(args.n_draws),
        "seed": int(args.seed),
        "common_random_numbers_original_shifted": True,
        "common_random_numbers_meaning": (
            "same initial-noise keys only; conditional ODE trajectories differ with Y"
        ),
        "pool": pool,
        "ode_steps": ode_steps,
        "jax_devices": str(jax.devices()),
        "git_commit": run_start_git_commit,
        **commit_provenance,
        "wall_seconds": float(time.perf_counter() - started),
        "correlations": correlations,
        "metrics": {
            kind: {
                estimator: _aggregate_metrics(rows)
                for estimator, rows in estimators.items()
            }
            for kind, estimators in metric_groups.items()
        },
        "selected_index": {
            "original": _distribution(selected_indices[:, 0]),
            "shifted": _distribution(selected_indices[:, 1]),
        },
        "frozen_smooth": (
            {
                **frozen_selection,
                "selection_path": str(Path(args.selection).resolve()),
                "selection_sha256": _sha256_file(args.selection),
                "tuned_on_current_scenario": False,
            }
            if frozen_selection is not None
            else None
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(out_dir / "per_sample_metrics.csv", per_sample_rows)
    _write_csv(out_dir / "per_draw_metrics.csv", per_draw_rows)
    np.savez_compressed(
        out_dir / "selected_predictions.npz",
        sample_indices=np.arange(n_samples, dtype=np.int64),
        input_kinds=np.asarray(["original", "shifted"]),
        theta_selected=selected_theta,
        centroids_selected=selected_centroids,
        draw_indices_selected=selected_indices,
    )
    if frozen_selection is not None:
        assert smooth_theta is not None and smooth_centroids is not None
        np.savez_compressed(
            out_dir / "smooth_predictions.npz",
            sample_indices=np.arange(n_samples, dtype=np.int64),
            input_kinds=np.asarray(["original", "shifted"]),
            theta_selected=smooth_theta,
            centroids_selected=smooth_centroids,
            estimator=np.asarray("frozen_smooth"),
            score_form=np.asarray(frozen_selection["score_form"]),
            tau=np.asarray(float(frozen_selection["tau"])),
            selection_sha256=np.asarray(_sha256_file(args.selection)),
        )
    environment = {
        "git_commit": run_start_git_commit,
        **commit_provenance,
        "python": sys.version,
        "jax_devices": str(jax.devices()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scenario_npz": str(scenario_path),
        "anna_dg0_npz": str(anna_path),
        "localization_g_shape": list(localization_g.shape),
        "localization_g_dtype": str(localization_g.dtype),
        "scenario_g_loaded": False,
    }
    (out_dir / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(out_dir / "report.md", summary)
    print(f"[rescore] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
