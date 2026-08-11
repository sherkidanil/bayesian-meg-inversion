#!/usr/bin/env python3
"""Pre-registered development and locked evaluation for smooth residual decoding."""

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
from scipy.spatial import cKDTree

from residual_rescoring import residual_gibbs_weights, residual_weighted_mean


HONEST_MODEL_NAME = "trial81_honest_4965"
TEMPERATURE_EXPONENTS = np.arange(-8, 9, dtype=np.int64)
ACCURACY_MULTIPLIER = 1.10
SENSOR_VIEW_KEYS = (
    "Ys_rebuilt",
    "Ys_rebuilt_shifted",
    "Ys_anna",
    "Ys_anna_shifted",
)
MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
GDG_ROOT = PROJECT_ROOT / "new_version_mesh_experiments_g_dg"


def derive_temperature_grid(
    residuals: np.ndarray, score_form: str
) -> dict[str, Any]:
    """Derive a deterministic, outcome-free grid from within-input score gaps."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 2:
        raise ValueError("residuals must have at least two draws per input")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("residuals must be finite and non-negative")
    if score_form == "quadratic":
        scores = np.square(values)
    elif score_form == "linear":
        scores = values
    else:
        raise ValueError("score_form must be 'quadratic' or 'linear'")
    gaps = np.median(scores, axis=-1) - np.min(scores, axis=-1)
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        raise ValueError("residual scores have no positive median-to-minimum gaps")
    score_gap_reference = float(np.median(positive))
    tau_reference = (
        float(np.sqrt(score_gap_reference))
        if score_form == "quadratic"
        else score_gap_reference
    )
    temperatures = tau_reference * np.power(2.0, TEMPERATURE_EXPONENTS)
    return {
        "score_form": score_form,
        "score_gap_reference": score_gap_reference,
        "tau_reference": tau_reference,
        "exponents": TEMPERATURE_EXPONENTS.copy(),
        "temperatures": np.asarray(temperatures, dtype=np.float64),
    }


def select_preregistered_candidate(
    rows: Iterable[dict[str, Any]], *, accuracy_multiplier: float = ACCURACY_MULTIPLIER
) -> dict[str, Any]:
    """Apply the frozen accuracy constraint, objective, and deterministic tie-breaks."""
    candidates = [dict(row) for row in rows]
    baselines = [row for row in candidates if row.get("estimator") == "argmin"]
    if len(baselines) != 1:
        raise ValueError("candidate rows must contain exactly one argmin baseline")
    baseline_accuracy = float(baselines[0]["mae_euc_mm"])
    if not np.isfinite(baseline_accuracy) or baseline_accuracy < 0.0:
        raise ValueError("argmin MAE must be finite and non-negative")
    limit = float(accuracy_multiplier) * baseline_accuracy
    eligible = [
        row
        for row in candidates
        if np.isfinite(float(row["mae_euc_mm"]))
        and np.isfinite(float(row["deltax_euc_mm"]))
        and float(row["mae_euc_mm"]) <= limit + 1e-12
    ]
    if not eligible:
        raise ValueError("no candidate satisfies the pre-registered accuracy constraint")

    def ordering(row: dict[str, Any]) -> tuple[float, float, int, float]:
        form_priority = {"quadratic": 0, "linear": 1}.get(
            str(row.get("score_form", "")), 2
        )
        tau = float(row.get("tau", 0.0))
        return (
            float(row["deltax_euc_mm"]),
            float(row["mae_euc_mm"]),
            form_priority,
            tau,
        )

    selected = dict(min(eligible, key=ordering))
    selected.update(
        {
            "accuracy_multiplier": float(accuracy_multiplier),
            "argmin_mae_euc_mm": baseline_accuracy,
            "accuracy_limit_mm": limit,
            "selection_rule": (
                "minimum dev Euclidean Delta X prime subject to MAE_euc <= "
                "1.10 * argmin MAE; ties: lower MAE, quadratic, lower tau"
            ),
        }
    )
    return selected


def assert_dev_tuning_input(dataset_path: str | Path, model_key: str) -> None:
    if str(model_key) != HONEST_MODEL_NAME:
        raise ValueError(f"tuning requires explicit model key {HONEST_MODEL_NAME}")
    assert_dev_split(dataset_path)


def assert_dev_split(dataset_path: str | Path) -> None:
    dataset = Path(dataset_path)
    with np.load(dataset, allow_pickle=True) as payload:
        if "split_kind" not in payload.files:
            raise ValueError("tuning dataset has no split marker")
        split_kind = str(payload["split_kind"].item())
    if split_kind != "dev":
        raise ValueError("held-out data are forbidden in tune mode; expected dev split")


def load_locked_heldout_selection(
    dataset_path: str | Path, selection_path: str | Path
) -> dict[str, Any]:
    """Verify the held-out file against its bundle and the frozen dev selection."""
    dataset = Path(dataset_path).resolve()
    selection_file = Path(selection_path).resolve()
    with np.load(dataset, allow_pickle=True) as payload:
        if "split_kind" not in payload.files or str(payload["split_kind"].item()) != "heldout":
            raise ValueError("locked evaluation requires the held-out split")
    manifest_path = dataset.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_entry = manifest["files"]["heldout"]
    if dataset.name != heldout_entry["path"]:
        raise ValueError("held-out path does not match the bundle manifest")
    if _sha256_file(dataset) != heldout_entry["sha256"]:
        raise ValueError("held-out file hash differs from the bundle manifest")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("mode") != "dev_tune" or "selected" not in selection:
        raise ValueError("selection file is not a frozen dev-tune result")
    if selection.get("model_key") != HONEST_MODEL_NAME:
        raise ValueError(f"selection does not use {HONEST_MODEL_NAME}")
    if selection.get("dataset_sha256") != manifest["files"]["dev"]["sha256"]:
        raise ValueError("frozen selection dev hash differs from the bundle manifest")
    return selection


def _best_permutation(cost: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("assignment cost must be square")
    best_order: tuple[int, ...] | None = None
    best_cost = float("inf")
    for order in permutations(range(matrix.shape[1])):
        candidate = float(np.mean(matrix[np.arange(matrix.shape[0]), order]))
        if candidate < best_cost:
            best_cost = candidate
            best_order = order
    if best_order is None:
        return np.arange(matrix.shape[0], dtype=np.int64), float("inf")
    return np.asarray(best_order, dtype=np.int64), best_cost


def evaluate_estimate_batch(
    estimates: np.ndarray,
    truth_original: np.ndarray,
    truth_shifted: np.ndarray,
    vertices_m: np.ndarray,
    full_geodesic_mm: np.ndarray,
) -> dict[str, Any]:
    """Compute the pre-registered dev MAE and truth-anchored Euclidean Delta X prime."""
    coords = np.asarray(estimates, dtype=np.float64)
    truth = np.stack(
        [
            np.asarray(truth_original, dtype=np.int64),
            np.asarray(truth_shifted, dtype=np.int64),
        ],
        axis=1,
    )
    vertices = np.asarray(vertices_m, dtype=np.float64)
    geodesic = np.asarray(full_geodesic_mm, dtype=np.float64)
    if coords.ndim != 4 or coords.shape[1:] != (2, truth.shape[2], 3):
        raise ValueError("estimates must have shape (samples, 2, sources, 3)")
    if coords.shape[:3] != truth.shape:
        raise ValueError("estimates and truth arrays do not align")
    predicted = cKDTree(vertices).query(coords.reshape(-1, 3), k=1)[1].reshape(
        truth.shape
    )
    mae = np.empty((len(coords), 2), dtype=np.float64)
    mae_geo = np.empty((len(coords), 2), dtype=np.float64)
    aligned = np.empty_like(predicted)
    for sample_index in range(len(coords)):
        for kind_index in range(2):
            true_vertices = truth[sample_index, kind_index]
            predicted_vertices = predicted[sample_index, kind_index]
            euclidean_cost = (
                np.linalg.norm(
                    vertices[true_vertices][:, None, :]
                    - vertices[predicted_vertices][None, :, :],
                    axis=2,
                )
                * 1000.0
            )
            _euclidean_order, mae[sample_index, kind_index] = _best_permutation(
                euclidean_cost
            )
            geodesic_cost = geodesic[true_vertices[:, None], predicted_vertices[None, :]]
            alignment_order, alignment_cost = _best_permutation(geodesic_cost)
            mae_geo[sample_index, kind_index] = alignment_cost
            if not np.isfinite(alignment_cost):
                alignment_order, _ = _best_permutation(euclidean_cost)
            aligned[sample_index, kind_index] = predicted_vertices[alignment_order]
    deltax_per_sample = np.mean(
        np.linalg.norm(
            vertices[aligned[:, 0]] - vertices[aligned[:, 1]], axis=2
        )
        * 1000.0,
        axis=1,
    )
    deltax_geo_per_sample = np.mean(
        geodesic[aligned[:, 0], aligned[:, 1]], axis=1
    )

    def finite_mean(values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return float(np.mean(finite)) if finite.size else float("nan")

    per_sample_mae_geo = np.mean(mae_geo, axis=1)
    return {
        "mae_euc_mm": float(np.mean(mae)),
        "mae_euc_original_mm": float(np.mean(mae[:, 0])),
        "mae_euc_shifted_mm": float(np.mean(mae[:, 1])),
        "mae_geo_mm": finite_mean(mae_geo),
        "mae_geo_original_mm": finite_mean(mae_geo[:, 0]),
        "mae_geo_shifted_mm": finite_mean(mae_geo[:, 1]),
        "mae_geo_nonfinite": int(np.sum(~np.isfinite(mae_geo))),
        "deltax_euc_mm": float(np.mean(deltax_per_sample)),
        "deltax_geo_mm": finite_mean(deltax_geo_per_sample),
        "deltax_geo_nonfinite": int(np.sum(~np.isfinite(deltax_geo_per_sample))),
        "per_sample_mae_euc_mm": np.mean(mae, axis=1),
        "per_sample_mae_euc_original_mm": mae[:, 0],
        "per_sample_mae_euc_shifted_mm": mae[:, 1],
        "per_sample_mae_geo_mm": per_sample_mae_geo,
        "per_sample_mae_geo_original_mm": mae_geo[:, 0],
        "per_sample_mae_geo_shifted_mm": mae_geo[:, 1],
        "per_sample_deltax_euc_mm": deltax_per_sample,
        "per_sample_deltax_geo_mm": deltax_geo_per_sample,
        "predicted_centroids": predicted,
        "aligned_centroids": aligned,
    }


def candidate_estimates(
    theta_draws: np.ndarray,
    residuals: np.ndarray,
    *,
    estimator: str,
    score_form: str | None = None,
    tau: float | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    draws = np.asarray(theta_draws, dtype=np.float64)
    scores = np.asarray(residuals, dtype=np.float64)
    if draws.ndim != 5 or scores.shape != draws.shape[:3]:
        raise ValueError("draws/residuals must have shapes (N, 2, D, K, 3)/(N, 2, D)")
    estimates = np.empty(draws.shape[:2] + draws.shape[3:], dtype=np.float64)
    ess: list[float] = []
    entropy: list[float] = []
    for sample_index in range(draws.shape[0]):
        for kind_index in range(draws.shape[1]):
            if estimator == "argmin":
                weights = np.zeros(draws.shape[2], dtype=np.float64)
                weights[int(np.argmin(scores[sample_index, kind_index]))] = 1.0
                estimate = np.tensordot(
                    weights, draws[sample_index, kind_index], axes=(0, 0)
                )
            elif estimator == "unweighted_mean":
                weights = np.full(draws.shape[2], 1.0 / draws.shape[2])
                estimate = np.mean(draws[sample_index, kind_index], axis=0)
            elif estimator == "gibbs":
                if score_form is None or tau is None:
                    raise ValueError("Gibbs candidates require score_form and tau")
                estimate, weights = residual_weighted_mean(
                    draws[sample_index, kind_index],
                    scores[sample_index, kind_index],
                    tau=float(tau),
                    score_form=score_form,
                )
            else:
                raise ValueError(f"unknown estimator {estimator}")
            estimates[sample_index, kind_index] = estimate
            ess.append(float(1.0 / np.sum(np.square(weights))))
            positive = weights[weights > 0.0]
            entropy.append(float(-np.sum(positive * np.log(positive))))
    return estimates, {
        "mean_effective_sample_size": float(np.mean(ess)),
        "mean_weight_entropy": float(np.mean(entropy)),
    }


def compute_argmin_jump_diagnostics(
    theta_draws: np.ndarray,
    residuals: np.ndarray,
    vertices_m: np.ndarray,
    full_geodesic_mm: np.ndarray,
) -> dict[str, Any]:
    """Separate argmin index switches from same-key motion and within-input jumps."""
    draws = np.asarray(theta_draws, dtype=np.float64)
    scores = np.asarray(residuals, dtype=np.float64)
    vertices = np.asarray(vertices_m, dtype=np.float64)
    geodesic = np.asarray(full_geodesic_mm, dtype=np.float64)
    if draws.ndim != 5 or scores.shape != draws.shape[:3] or draws.shape[1] != 2:
        raise ValueError("draws/residuals must have paired shapes (N, 2, D, K, 3)/(N, 2, D)")
    rows = np.arange(draws.shape[0], dtype=np.int64)
    original_index = np.argmin(scores[:, 0], axis=1)
    shifted_index = np.argmin(scores[:, 1], axis=1)
    switched = original_index != shifted_index
    original_winner_a = draws[rows, 0, original_index]
    original_winner_b = draws[rows, 1, original_index]
    shifted_winner_a = draws[rows, 0, shifted_index]
    shifted_winner_b = draws[rows, 1, shifted_index]

    pair_definitions = {
        "selected": (original_winner_a, shifted_winner_b),
        "same_index_original_winner": (original_winner_a, original_winner_b),
        "same_index_shifted_winner": (shifted_winner_a, shifted_winner_b),
        "within_original": (original_winner_a, shifted_winner_a),
        "within_shifted": (original_winner_b, shifted_winner_b),
    }
    result: dict[str, Any] = {
        "original_argmin_index": original_index,
        "shifted_argmin_index": shifted_index,
        "index_switched": switched,
        "switch_fraction": float(np.mean(switched)),
    }
    tree = cKDTree(vertices)
    for label, (left, right) in pair_definitions.items():
        result[f"{label}_theta_mm"] = np.mean(
            np.linalg.norm(left - right, axis=2) * 1000.0, axis=1
        )
        left_centroids = tree.query(left.reshape(-1, 3), k=1)[1].reshape(left.shape[:2])
        right_centroids = tree.query(right.reshape(-1, 3), k=1)[1].reshape(right.shape[:2])
        result[f"{label}_centroid_euc_mm"] = np.mean(
            np.linalg.norm(
                vertices[left_centroids] - vertices[right_centroids], axis=2
            )
            * 1000.0,
            axis=1,
        )
        result[f"{label}_centroid_geo_mm"] = np.mean(
            geodesic[left_centroids, right_centroids], axis=1
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_digest(path: Path) -> dict[str, Any]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    entries = {str(candidate.relative_to(path)): _sha256_file(candidate) for candidate in files}
    digest = hashlib.sha256()
    for name, value in entries.items():
        digest.update(name.encode())
        digest.update(value.encode())
    return {"sha256": digest.hexdigest(), "files": entries}


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _draw_seed(base_seed: int, sample_index: int) -> int:
    return int(
        np.random.SeedSequence([int(base_seed), int(sample_index)]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    return {
        "n": int(raw.size),
        "n_finite": int(finite.size),
        "n_nonfinite": int(raw.size - finite.size),
        "mean": float(np.mean(finite)) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
    }


def run_tune(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.out_dir).resolve()
    assert_dev_tuning_input(dataset_path, args.model_key)
    start_git = _git_state()
    if start_git["dirty"]:
        raise RuntimeError("tuning requires a clean repository")
    output_dir.mkdir(parents=True, exist_ok=False)
    if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for path in (MODULE_ROOT, GDG_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import jax
    import jax.numpy as jnp

    from eval_cnn_checkpoints_25march import checkpoint_specs, load_checkpoint_runtime
    from eval_posterior_mean_100 import make_multi_sampler
    from metrics import build_graph_from_edges
    from optuna_sweep_cnn import pool_time
    from residual_rescoring import residual_rescore_draws

    with np.load(dataset_path, allow_pickle=True) as payload:
        required = (
            "Ys_rebuilt",
            "Ys_rebuilt_shifted",
            "centroids_idx",
            "centroids_idx_shifted",
            "G_rebuilt",
            "vertices",
            "edge_index",
            "edge_weight_mm",
            "params_spatial",
            "full_sample_indices",
        )
        data = {
            key: payload[key].item() if key == "params_spatial" else np.array(payload[key])
            for key in required
        }
    n_total = len(data["Ys_rebuilt"])
    n_samples = min(n_total, int(args.limit)) if args.limit > 0 else n_total
    if n_samples <= 0:
        raise ValueError("no dev samples selected")
    if n_samples != n_total:
        for key in (
            "Ys_rebuilt",
            "Ys_rebuilt_shifted",
            "centroids_idx",
            "centroids_idx_shifted",
            "full_sample_indices",
        ):
            data[key] = data[key][:n_samples]
    if n_samples != 350 and not args.allow_smoke_size:
        raise ValueError(f"pre-registered tune run requires 350 dev samples, found {n_samples}")
    matching_specs = [
        spec for spec in checkpoint_specs() if spec["model_key"] == args.model_key
    ]
    if len(matching_specs) != 1:
        raise ValueError(f"model key {args.model_key!r} is missing or ambiguous")
    spec = matching_specs[0]
    runtime = load_checkpoint_runtime(spec)
    if str(runtime["model_key"]) != HONEST_MODEL_NAME:
        raise ValueError(f"runtime is not the explicit {HONEST_MODEL_NAME} checkpoint")
    pool = int(runtime["pool"])
    ode_steps = int(runtime["ode_steps"])
    theta_mean = np.asarray(runtime["theta_mean"], dtype=np.float32)
    theta_std = np.asarray(runtime["theta_std"], dtype=np.float32)
    sampler = make_multi_sampler(runtime["model"], theta_dim=9, steps=ode_steps)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    graph = build_graph_from_edges(
        data["edge_index"], data["edge_weight_mm"], len(vertices)
    )
    full_geo_mm = np.asarray(dijkstra(graph, directed=False), dtype=np.float64)
    theta_draws = np.empty((n_samples, 2, args.n_draws, 3, 3), dtype=np.float32)
    residuals = np.empty((n_samples, 2, args.n_draws), dtype=np.float64)
    started = time.perf_counter()
    for sample_index in range(n_samples):
        seed = _draw_seed(args.seed, sample_index)
        draw_keys = jax.random.split(jax.random.PRNGKey(seed), args.n_draws)
        for kind_index, key in enumerate(("Ys_rebuilt", "Ys_rebuilt_shifted")):
            y = np.asarray(data[key][sample_index], dtype=np.float32)
            pooled = pool_time(y[None, ...], pool=pool).astype(np.float32)[0]
            normalized = np.asarray(
                sampler(jnp.asarray(pooled), draw_keys), dtype=np.float32
            )
            draws = (normalized * theta_std[None, :] + theta_mean[None, :]).reshape(
                args.n_draws, 3, 3
            )
            theta_draws[sample_index, kind_index] = draws
            residuals[sample_index, kind_index] = residual_rescore_draws(
                draws,
                y,
                data["G_rebuilt"],
                vertices,
                graph,
                data["params_spatial"],
            )[0]
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == n_samples:
            print(
                f"[smooth-dev] {sample_index + 1}/{n_samples}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    np.savez_compressed(
        output_dir / "dev_draws_residuals.npz",
        theta_draws=theta_draws,
        residuals=residuals,
        full_sample_indices=data["full_sample_indices"],
        seed=np.array(args.seed, dtype=np.int64),
        n_draws=np.array(args.n_draws, dtype=np.int64),
    )
    grids = {
        form: derive_temperature_grid(residuals.reshape(-1, args.n_draws), form)
        for form in ("quadratic", "linear")
    }
    candidates: list[dict[str, Any]] = [
        {"candidate_id": "argmin", "estimator": "argmin"},
        {"candidate_id": "unweighted_mean", "estimator": "unweighted_mean"},
    ]
    for form, grid in grids.items():
        for exponent, temperature in zip(
            grid["exponents"], grid["temperatures"], strict=True
        ):
            candidates.append(
                {
                    "candidate_id": f"{form}_2^{int(exponent)}",
                    "estimator": "gibbs",
                    "score_form": form,
                    "temperature_exponent": int(exponent),
                    "tau": float(temperature),
                }
            )
    curve_rows: list[dict[str, Any]] = []
    candidate_metrics: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        estimates, weight_metrics = candidate_estimates(
            theta_draws,
            residuals,
            estimator=str(candidate["estimator"]),
            score_form=candidate.get("score_form"),
            tau=candidate.get("tau"),
        )
        metrics = evaluate_estimate_batch(
            estimates,
            data["centroids_idx"],
            data["centroids_idx_shifted"],
            vertices,
            full_geo_mm,
        )
        row = {
            **candidate,
            **weight_metrics,
            **{
                key: value
                for key, value in metrics.items()
                if not isinstance(value, np.ndarray)
            },
        }
        curve_rows.append(row)
        candidate_metrics[str(candidate["candidate_id"])] = {
            "estimates": estimates,
            "metrics": metrics,
        }
    selected = select_preregistered_candidate(curve_rows)
    dataset_hash = _sha256_file(dataset_path)
    model_files = {
        key: Path(spec[key]).resolve()
        for key in ("model_path", "stats_path", "meta_path")
    }
    model_hash = {
        "files": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in model_files.items()
        }
    }
    combined_model_digest = hashlib.sha256()
    for key, entry in sorted(model_hash["files"].items()):
        combined_model_digest.update(key.encode())
        combined_model_digest.update(entry["sha256"].encode())
    model_hash["sha256"] = combined_model_digest.hexdigest()
    selection = {
        "schema_version": 1,
        "mode": "dev_tune",
        "model_name": HONEST_MODEL_NAME,
        "model_key": str(runtime["model_key"]),
        "model_digest": model_hash,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "dataset_split_kind": "dev",
        "n_samples": n_samples,
        "n_draws": int(args.n_draws),
        "seed": int(args.seed),
        "temperature_grids": grids,
        "selected": selected,
        "run_start_git": start_git,
        "artifact_write_git": _git_state(),
        "wall_seconds": float(time.perf_counter() - started),
    }
    selection["repository_changed_during_run"] = (
        selection["run_start_git"] != selection["artifact_write_git"]
    )
    if selection["repository_changed_during_run"]:
        raise RuntimeError("repository changed during dev tuning")
    _write_csv(output_dir / "curve.csv", curve_rows)
    _atomic_json(output_dir / "selection.json", selection)
    selected_id = str(selected["candidate_id"])
    selected_result = candidate_metrics[selected_id]
    selected_metrics = selected_result["metrics"]
    np.savez_compressed(
        output_dir / "selected_dev_predictions.npz",
        estimates=selected_result["estimates"].astype(np.float32),
        predicted_centroids=selected_metrics["predicted_centroids"],
        aligned_centroids=selected_metrics["aligned_centroids"],
        per_sample_mae_euc_mm=selected_metrics["per_sample_mae_euc_mm"],
        per_sample_deltax_euc_mm=selected_metrics["per_sample_deltax_euc_mm"],
        argmin_indices=np.argmin(residuals, axis=2),
        full_sample_indices=data["full_sample_indices"],
    )
    summary = {
        "selection": selection,
        "curve_rows": curve_rows,
        "jax_devices": str(jax.devices()),
        "pool": pool,
        "ode_steps": ode_steps,
    }
    _atomic_json(output_dir / "summary.json", summary)
    report = [
        "# Smooth residual decoder: pre-registered dev selection",
        "",
        f"- Model: `{HONEST_MODEL_NAME}`",
        f"- Dataset: `{dataset_path}` (`dev`, n={n_samples})",
        f"- Draws per input: {args.n_draws}",
        "- Quadratic score is the theoretical default; linear score is an empirical comparator.",
        "- These are Gibbs/pseudo-posterior weights, not importance weights.",
        "- Weighted averaging is licensed by the prior 0.014% permutation-audit failure rate.",
        "",
        "## Frozen selection",
        "",
        f"- Candidate: `{selected_id}`",
        f"- Dev MAE euc: {selected['mae_euc_mm']:.4f} mm",
        f"- Dev Delta X prime euc: {selected['deltax_euc_mm']:.4f} mm",
        f"- Accuracy limit: {selected['accuracy_limit_mm']:.4f} mm",
        "- Held-out data were not loaded by this command.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_json_ready(selected), indent=2, sort_keys=True), flush=True)


def run_jumps(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    draws_path = Path(args.draws_npz).resolve()
    output_dir = Path(args.out_dir).resolve()
    assert_dev_split(dataset_path)
    start_git = _git_state()
    if start_git["dirty"]:
        raise RuntimeError("jump diagnostics require a clean repository")
    output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(dataset_path, allow_pickle=True) as payload:
        vertices = np.asarray(payload["vertices"], dtype=np.float64)
        edge_index = np.asarray(payload["edge_index"])
        edge_weight_mm = np.asarray(payload["edge_weight_mm"])
        dataset_indices = np.asarray(payload["full_sample_indices"], dtype=np.int64)
    with np.load(draws_path, allow_pickle=False) as draws_payload:
        theta_draws = np.asarray(draws_payload["theta_draws"], dtype=np.float64)
        residuals = np.asarray(draws_payload["residuals"], dtype=np.float64)
        draw_indices = np.asarray(draws_payload["full_sample_indices"], dtype=np.int64)
    if not np.array_equal(dataset_indices, draw_indices):
        raise ValueError("dev dataset and saved draws use different full-sample indices")
    for path in (MODULE_ROOT, GDG_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from metrics import build_graph_from_edges

    graph = build_graph_from_edges(edge_index, edge_weight_mm, len(vertices))
    full_geo_mm = np.asarray(dijkstra(graph, directed=False), dtype=np.float64)
    diagnostics = compute_argmin_jump_diagnostics(
        theta_draws, residuals, vertices, full_geo_mm
    )
    switched = np.asarray(diagnostics["index_switched"], dtype=bool)
    masks = {
        "all": np.ones(len(switched), dtype=bool),
        "switched": switched,
        "unchanged": ~switched,
    }
    measures = sorted(key for key in diagnostics if key.endswith("_mm"))
    by_subset = {
        subset: {
            measure: _distribution(np.asarray(diagnostics[measure])[mask])
            for measure in measures
        }
        for subset, mask in masks.items()
    }
    excess = {
        measure: (
            by_subset["switched"][measure]["mean"]
            - by_subset["unchanged"][measure]["mean"]
        )
        for measure in measures
    }
    end_git = _git_state()
    if start_git != end_git:
        raise RuntimeError("repository changed during jump diagnostics")
    summary = {
        "schema_version": 1,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "draws_path": str(draws_path),
        "draws_sha256": _sha256_file(draws_path),
        "n_samples": int(len(switched)),
        "switch_count": int(np.sum(switched)),
        "switch_fraction": float(np.mean(switched)),
        "by_subset": by_subset,
        "switched_minus_unchanged_mean_mm": excess,
        "common_key_interpretation": (
            "draw index identifies common initial noise only; paired ODE trajectories remain Y-conditional"
        ),
        "run_start_git": start_git,
        "artifact_write_git": end_git,
        "repository_changed_during_run": False,
    }
    _atomic_json(output_dir / "summary.json", summary)
    np.savez_compressed(
        output_dir / "jump_arrays.npz",
        full_sample_indices=draw_indices,
        **{key: value for key, value in diagnostics.items() if isinstance(value, np.ndarray)},
    )
    rows: list[dict[str, Any]] = []
    for sample_index in range(len(switched)):
        row: dict[str, Any] = {
            "dev_sample_index": sample_index,
            "full_sample_index": int(draw_indices[sample_index]),
            "original_argmin_index": int(diagnostics["original_argmin_index"][sample_index]),
            "shifted_argmin_index": int(diagnostics["shifted_argmin_index"][sample_index]),
            "index_switched": bool(switched[sample_index]),
        }
        row.update(
            {measure: float(diagnostics[measure][sample_index]) for measure in measures}
        )
        rows.append(row)
    _write_csv(output_dir / "per_sample.csv", rows)
    report = [
        "# Argmin winner-switch and jump decomposition (dev)",
        "",
        f"- Samples: {len(switched)}",
        f"- Winner index changed: {np.sum(switched)}/{len(switched)} ({100.0 * np.mean(switched):.2f}%).",
        "- Equal indices label equal initial-noise keys, not equal conditional theta trajectories.",
        "- `same_index_*` measures conditional trajectory motion at a fixed key.",
        "- `within_*` measures the discrete winner-to-counterfactual jump at a fixed input.",
        "",
        "## Mean displacement, mm",
        "",
        "| measure | all | switched | unchanged | switched - unchanged |",
        "|---|---:|---:|---:|---:|",
    ]
    for measure in measures:
        report.append(
            f"| `{measure}` | {by_subset['all'][measure]['mean']:.4f} | "
            f"{by_subset['switched'][measure]['mean']:.4f} | "
            f"{by_subset['unchanged'][measure]['mean']:.4f} | {excess[measure]:+.4f} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"switch_fraction": summary["switch_fraction"]}, indent=2))


def _verify_model_digest(selection: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    expected = selection.get("model_digest", {}).get("files", {})
    actual: dict[str, Any] = {}
    for key in ("model_path", "stats_path", "meta_path"):
        path = Path(spec[key]).resolve()
        digest = _sha256_file(path)
        actual[key] = {"path": str(path), "sha256": digest}
        if key not in expected or expected[key].get("sha256") != digest:
            raise ValueError(f"frozen model digest differs at {key}")
    return actual


def _paired_finite_difference(
    anna_values: np.ndarray, rebuilt_values: np.ndarray
) -> dict[str, Any]:
    anna = np.asarray(anna_values, dtype=np.float64).reshape(-1)
    rebuilt = np.asarray(rebuilt_values, dtype=np.float64).reshape(-1)
    if anna.shape != rebuilt.shape:
        raise ValueError("paired domain arrays have different shapes")
    valid = np.isfinite(anna) & np.isfinite(rebuilt)
    difference = anna[valid] - rebuilt[valid]
    return {
        "paired_finite": int(np.sum(valid)),
        "paired_nonfinite": int(len(valid) - np.sum(valid)),
        "anna_minus_rebuilt": _distribution(difference),
    }


def run_heldout(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    selection_path = Path(args.selection).resolve()
    output_dir = Path(args.out_dir).resolve()
    selection = load_locked_heldout_selection(dataset_path, selection_path)
    start_git = _git_state()
    if start_git["dirty"]:
        raise RuntimeError("held-out evaluation requires a clean repository")
    output_dir.mkdir(parents=True, exist_ok=False)
    if os.environ.get("JAX_PLATFORMS", "").lower() != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for path in (MODULE_ROOT, GDG_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import jax
    import jax.numpy as jnp

    from eval_cnn_checkpoints_25march import checkpoint_specs, load_checkpoint_runtime
    from eval_posterior_mean_100 import make_multi_sampler
    from metrics import build_graph_from_edges
    from optuna_sweep_cnn import pool_time
    from residual_rescoring import residual_rescore_draws

    matching_specs = [
        spec
        for spec in checkpoint_specs()
        if spec["model_key"] == selection["model_key"]
    ]
    if len(matching_specs) != 1:
        raise ValueError("frozen model key is missing or ambiguous")
    spec = matching_specs[0]
    verified_model_files = _verify_model_digest(selection, spec)
    runtime = load_checkpoint_runtime(spec)
    if str(runtime["model_key"]) != HONEST_MODEL_NAME:
        raise ValueError("resolved runtime differs from frozen honest model")
    with np.load(dataset_path, allow_pickle=True) as payload:
        required = (
            *SENSOR_VIEW_KEYS,
            "centroids_idx",
            "centroids_idx_shifted",
            "G_rebuilt",
            "G_anna",
            "vertices",
            "edge_index",
            "edge_weight_mm",
            "params_spatial",
            "full_sample_indices",
        )
        data = {
            key: payload[key].item() if key == "params_spatial" else np.array(payload[key])
            for key in required
        }
    n_total = len(data["Ys_rebuilt"])
    n_samples = min(n_total, int(args.limit)) if args.limit > 0 else n_total
    if n_samples != 350 and not args.allow_smoke_size:
        raise ValueError(f"locked held-out run requires 350 samples, found {n_samples}")
    if n_samples != n_total:
        for key in (
            *SENSOR_VIEW_KEYS,
            "centroids_idx",
            "centroids_idx_shifted",
            "full_sample_indices",
        ):
            data[key] = data[key][:n_samples]
    pool = int(runtime["pool"])
    ode_steps = int(runtime["ode_steps"])
    theta_mean = np.asarray(runtime["theta_mean"], dtype=np.float32)
    theta_std = np.asarray(runtime["theta_std"], dtype=np.float32)
    sampler = make_multi_sampler(runtime["model"], theta_dim=9, steps=ode_steps)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    graph = build_graph_from_edges(
        data["edge_index"], data["edge_weight_mm"], len(vertices)
    )
    full_geo_mm = np.asarray(dijkstra(graph, directed=False), dtype=np.float64)
    domains = (
        ("rebuilt", "Ys_rebuilt", "Ys_rebuilt_shifted", "G_rebuilt"),
        ("anna", "Ys_anna", "Ys_anna_shifted", "G_anna"),
    )
    theta_draws = np.empty(
        (n_samples, len(domains), 2, args.n_draws, 3, 3), dtype=np.float32
    )
    residuals = np.empty(
        (n_samples, len(domains), 2, args.n_draws), dtype=np.float64
    )
    started = time.perf_counter()
    for sample_index in range(n_samples):
        seed = _draw_seed(args.seed, sample_index)
        draw_keys = jax.random.split(jax.random.PRNGKey(seed), args.n_draws)
        for domain_index, (_domain, original_key, shifted_key, g_key) in enumerate(domains):
            for kind_index, y_key in enumerate((original_key, shifted_key)):
                y = np.asarray(data[y_key][sample_index], dtype=np.float32)
                pooled = pool_time(y[None, ...], pool=pool).astype(np.float32)[0]
                normalized = np.asarray(
                    sampler(jnp.asarray(pooled), draw_keys), dtype=np.float32
                )
                draws = (
                    normalized * theta_std[None, :] + theta_mean[None, :]
                ).reshape(args.n_draws, 3, 3)
                theta_draws[sample_index, domain_index, kind_index] = draws
                residuals[sample_index, domain_index, kind_index] = residual_rescore_draws(
                    draws,
                    y,
                    data[g_key],
                    vertices,
                    graph,
                    data["params_spatial"],
                )[0]
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == n_samples:
            print(
                f"[smooth-heldout] {sample_index + 1}/{n_samples}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    np.savez_compressed(
        output_dir / "heldout_draws_residuals.npz",
        theta_draws=theta_draws,
        residuals=residuals,
        full_sample_indices=data["full_sample_indices"],
        seed=np.array(args.seed, dtype=np.int64),
        n_draws=np.array(args.n_draws, dtype=np.int64),
    )
    frozen = dict(selection["selected"])
    candidates = [
        {"candidate_id": "argmin", "estimator": "argmin"},
        {
            "candidate_id": "frozen_smooth",
            "estimator": str(frozen["estimator"]),
            "score_form": frozen.get("score_form"),
            "tau": frozen.get("tau"),
        },
        {"candidate_id": "unweighted_mean", "estimator": "unweighted_mean"},
    ]
    aggregated: dict[str, dict[str, Any]] = {}
    detailed: dict[str, dict[str, Any]] = {}
    prediction_payload: dict[str, np.ndarray] = {
        "full_sample_indices": np.asarray(data["full_sample_indices"]),
    }
    per_sample_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        aggregated[candidate_id] = {}
        detailed[candidate_id] = {}
        for domain_index, (domain, *_keys) in enumerate(domains):
            estimates, weight_metrics = candidate_estimates(
                theta_draws[:, domain_index],
                residuals[:, domain_index],
                estimator=str(candidate["estimator"]),
                score_form=candidate.get("score_form"),
                tau=candidate.get("tau"),
            )
            metrics = evaluate_estimate_batch(
                estimates,
                data["centroids_idx"],
                data["centroids_idx_shifted"],
                vertices,
                full_geo_mm,
            )
            aggregated[candidate_id][domain] = {
                **candidate,
                **weight_metrics,
                **{
                    key: value
                    for key, value in metrics.items()
                    if not isinstance(value, np.ndarray)
                },
            }
            detailed[candidate_id][domain] = metrics
            prediction_payload[f"{candidate_id}_{domain}_estimates"] = estimates.astype(
                np.float32
            )
            prediction_payload[
                f"{candidate_id}_{domain}_predicted_centroids"
            ] = metrics["predicted_centroids"]
            for sample_index in range(n_samples):
                per_sample_rows.append(
                    {
                        "heldout_sample_index": sample_index,
                        "full_sample_index": int(data["full_sample_indices"][sample_index]),
                        "candidate_id": candidate_id,
                        "domain": domain,
                        "mae_euc_original_mm": float(
                            metrics["per_sample_mae_euc_original_mm"][sample_index]
                        ),
                        "mae_euc_shifted_mm": float(
                            metrics["per_sample_mae_euc_shifted_mm"][sample_index]
                        ),
                        "mae_geo_original_mm": float(
                            metrics["per_sample_mae_geo_original_mm"][sample_index]
                        ),
                        "mae_geo_shifted_mm": float(
                            metrics["per_sample_mae_geo_shifted_mm"][sample_index]
                        ),
                        "deltax_euc_mm": float(
                            metrics["per_sample_deltax_euc_mm"][sample_index]
                        ),
                        "deltax_geo_mm": float(
                            metrics["per_sample_deltax_geo_mm"][sample_index]
                        ),
                    }
                )
    domain_cost: dict[str, Any] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rebuilt_metrics = detailed[candidate_id]["rebuilt"]
        anna_metrics = detailed[candidate_id]["anna"]
        domain_cost[candidate_id] = {
            measure: _paired_finite_difference(
                anna_metrics[measure], rebuilt_metrics[measure]
            )
            for measure in (
                "per_sample_mae_euc_mm",
                "per_sample_mae_euc_original_mm",
                "per_sample_mae_euc_shifted_mm",
                "per_sample_mae_geo_mm",
                "per_sample_mae_geo_original_mm",
                "per_sample_mae_geo_shifted_mm",
                "per_sample_deltax_euc_mm",
                "per_sample_deltax_geo_mm",
            )
        }
    end_git = _git_state()
    if start_git != end_git:
        raise RuntimeError("repository changed during held-out evaluation")
    summary = {
        "schema_version": 1,
        "mode": "locked_heldout",
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "selection_path": str(selection_path),
        "selection_sha256": _sha256_file(selection_path),
        "frozen_selection": frozen,
        "model_key": HONEST_MODEL_NAME,
        "verified_model_files": verified_model_files,
        "n_samples": n_samples,
        "n_draws": int(args.n_draws),
        "seed": int(args.seed),
        "aggregated": aggregated,
        "domain_shift_cost_anna_minus_rebuilt": domain_cost,
        "run_start_git": start_git,
        "artifact_write_git": end_git,
        "repository_changed_during_run": False,
        "wall_seconds": float(time.perf_counter() - started),
        "jax_devices": str(jax.devices()),
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "selection_used.json", selection)
    _write_csv(output_dir / "per_sample.csv", per_sample_rows)
    np.savez_compressed(output_dir / "predictions.npz", **prediction_payload)
    report = [
        "# Locked held-out smooth residual evaluation",
        "",
        f"- Model: `{HONEST_MODEL_NAME}`",
        f"- Frozen candidate: `{frozen['candidate_id']}`",
        f"- Samples: {n_samples}; draws per input: {args.n_draws}",
        "- No temperature grid or model selection was performed in this command.",
        "- Each sensor domain used its matching persisted localization lead field.",
        "",
        "## Held-out metrics",
        "",
        "| estimator | domain | MAE euc | MAE geo | geo non-finite | Delta X euc | Delta X geo | Delta X geo non-finite |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        for domain, *_keys in domains:
            row = aggregated[candidate_id][domain]
            report.append(
                f"| `{candidate_id}` | {domain} | {row['mae_euc_mm']:.4f} | "
                f"{row['mae_geo_mm']:.4f} | {row['mae_geo_nonfinite']} | "
                f"{row['deltax_euc_mm']:.4f} | {row['deltax_geo_mm']:.4f} | "
                f"{row['deltax_geo_nonfinite']} |"
            )
    report.extend(
        [
            "",
            "## Paired natural domain-shift cost (Anna minus rebuilt), mm",
            "",
            "| estimator | MAE euc | Delta X euc |",
            "|---|---:|---:|",
        ]
    )
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        cost = domain_cost[candidate_id]
        report.append(
            f"| `{candidate_id}` | "
            f"{cost['per_sample_mae_euc_mm']['anna_minus_rebuilt']['mean']:+.4f} | "
            f"{cost['per_sample_deltax_euc_mm']['anna_minus_rebuilt']['mean']:+.4f} |"
        )
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(aggregated), indent=2, sort_keys=True), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("tune", "jumps", "heldout"), default="tune"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-key", default="")
    parser.add_argument("--draws-npz", default="")
    parser.add_argument("--selection", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-smoke-size", action="store_true")
    args = parser.parse_args(argv)
    if args.n_draws <= 0:
        parser.error("--n-draws must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "tune":
        if not args.model_key:
            raise ValueError("tune mode requires --model-key")
        run_tune(args)
    elif args.mode == "jumps":
        if not args.draws_npz:
            raise ValueError("jumps mode requires --draws-npz")
        run_jumps(args)
    elif args.mode == "heldout":
        if not args.selection:
            raise ValueError("heldout mode requires --selection")
        run_heldout(args)


if __name__ == "__main__":
    main()
