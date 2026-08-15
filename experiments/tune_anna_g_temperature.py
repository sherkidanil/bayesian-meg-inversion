#!/usr/bin/env python3
"""Freeze quadratic residual temperature for the honest Anna-G checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.sparse.csgraph import dijkstra

MODULE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = (
    MODULE_ROOT.parent
    if MODULE_ROOT.name == "new_version_mesh_experiments"
    else MODULE_ROOT
)
for _path in (MODULE_ROOT, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from smooth_residual_audit import (
    candidate_estimates,
    derive_temperature_grid,
    evaluate_estimate_batch,
    select_preregistered_candidate,
)
from train_anna_g_honest import EXPECTED_HONEST_PARAMS


MODEL_KEY = "honest_4965_annaG"
REQUIRED_DEV_KEYS = (
    "Ys",
    "Ys_shifted",
    "centroids_idx",
    "centroids_idx_shifted",
    "G_for_simulation",
    "vertices",
    "edge_index",
    "edge_weight_mm",
    "params_spatial",
    "params_temporal",
    "full_sample_indices",
    "split_kind",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def locked_model_spec(model_dir: Path) -> dict[str, Any]:
    directory = model_dir.resolve()
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("candidate_id") != MODEL_KEY:
        raise ValueError(f"checkpoint candidate_id must be {MODEL_KEY}")
    for key, expected in EXPECTED_HONEST_PARAMS.items():
        if result.get(key) != expected:
            raise ValueError(
                f"checkpoint parameter {key} differs: "
                f"expected {expected!r}, found {result.get(key)!r}"
            )
    for name in ("model.eqx", "stats.npz", "result.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return {
        "model_key": MODEL_KEY,
        "display_name": "honest_4965 retrained on Anna G",
        "protocol_note": "single locked honest configuration; no hyperparameter sweep",
        "model_path": str((directory / "model.eqx").resolve()),
        "stats_path": str((directory / "stats.npz").resolve()),
        "meta_path": str(result_path.resolve()),
        "n_post_samples": int(EXPECTED_HONEST_PARAMS["n_post_samples"]),
        **{
            key: EXPECTED_HONEST_PARAMS[key]
            for key in (
                "cond_dim",
                "cnn_width1",
                "cnn_width2",
                "cnn_kernel",
                "cnn_stride",
                "enc_act",
                "vf_width",
                "vf_depth",
                "vf_act",
            )
        },
    }


def load_single_domain_dev(dataset_path: Path) -> dict[str, Any]:
    with np.load(dataset_path, allow_pickle=True) as payload:
        missing = [key for key in REQUIRED_DEV_KEYS if key not in payload.files]
        if missing:
            raise KeyError(", ".join(missing))
        if str(payload["split_kind"].item()) != "dev":
            raise ValueError("temperature tuning requires the frozen dev split")
        data = {
            key: (
                dict(payload[key].item())
                if key in {"params_spatial", "params_temporal"}
                else np.array(payload[key], copy=True)
            )
            for key in REQUIRED_DEV_KEYS
            if key != "split_kind"
        }
    if data["Ys"].shape != data["Ys_shifted"].shape:
        raise ValueError("paired sensor arrays differ in shape")
    if data["Ys"].ndim != 3:
        raise ValueError("paired sensor arrays must have shape (samples, sensors, time)")
    if data["G_for_simulation"].ndim != 2:
        raise ValueError("G_for_simulation must be stored once as a 2-D matrix")
    if data["G_for_simulation"].shape[0] != data["Ys"].shape[1]:
        raise ValueError("lead field and sensor arrays do not align")
    return data


def _draw_seed(base_seed: int, sample_index: int) -> int:
    return int(
        np.random.SeedSequence([int(base_seed), int(sample_index)]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    def json_ready(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(key): json_ready(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [json_ready(child) for child in item]
        if isinstance(item, np.ndarray):
            return json_ready(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item

    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-smoke-size", action="store_true")
    parser.add_argument("--residual-bandpass-hz", nargs=2, type=float, default=None)
    parser.add_argument("--residual-filter-order", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_draws <= 0:
        raise ValueError("n-draws must be positive")
    repository = REPOSITORY_ROOT
    start_git = _git_state(repository)
    if start_git["dirty"]:
        raise RuntimeError("temperature tuning requires a clean repository")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    spec = locked_model_spec(args.model_dir)
    data = load_single_domain_dev(args.dataset)
    n_total = len(data["Ys"])
    n_samples = min(n_total, args.limit) if args.limit > 0 else n_total
    if n_samples != 350 and not args.allow_smoke_size:
        raise ValueError(f"locked tune requires 350 dev samples, found {n_samples}")
    for key in (
        "Ys",
        "Ys_shifted",
        "centroids_idx",
        "centroids_idx_shifted",
        "full_sample_indices",
    ):
        data[key] = data[key][:n_samples]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    mesh_root = repository / "new_version_mesh_experiments"
    gdg_root = repository / "new_version_mesh_experiments_g_dg"
    for path in (mesh_root, gdg_root):
        sys.path.insert(0, str(path))
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from eval_cnn_checkpoints_25march import load_checkpoint_runtime  # noqa: PLC0415
    from eval_posterior_mean_100 import make_multi_sampler  # noqa: PLC0415
    from metrics import build_graph_from_edges  # noqa: PLC0415
    from optuna_sweep_cnn import pool_time  # noqa: PLC0415
    from residual_rescoring import (  # noqa: PLC0415
        butterworth_bandpass_sensor_data,
        residual_rescore_draws,
    )

    runtime = load_checkpoint_runtime(spec)
    if runtime["model_key"] != MODEL_KEY:
        raise ValueError("loaded runtime is not the locked Anna-G checkpoint")
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
    if args.residual_bandpass_hz is None:
        residual_preprocessing = {"kind": "none"}
    else:
        low_hz, high_hz = (float(value) for value in args.residual_bandpass_hz)
        residual_preprocessing = {
            "kind": "butterworth_bandpass",
            "low_hz": low_hz,
            "high_hz": high_hz,
            "order": int(args.residual_filter_order),
            "sfreq_hz": float(data["params_temporal"]["Fs"]),
            "phase": "zero",
        }
    started = time.perf_counter()
    for sample_index in range(n_samples):
        seed = _draw_seed(args.seed, sample_index)
        draw_keys = jax.random.split(jax.random.PRNGKey(seed), args.n_draws)
        for kind_index, sensor_key in enumerate(("Ys", "Ys_shifted")):
            y = np.asarray(data[sensor_key][sample_index], dtype=np.float32)
            pooled = pool_time(y[None, ...], pool=pool).astype(np.float32)[0]
            normalized = np.asarray(
                sampler(jnp.asarray(pooled), draw_keys), dtype=np.float32
            )
            draws = (normalized * theta_std[None, :] + theta_mean[None, :]).reshape(
                args.n_draws, 3, 3
            )
            theta_draws[sample_index, kind_index] = draws
            residual_y = y
            if residual_preprocessing["kind"] == "butterworth_bandpass":
                residual_y = butterworth_bandpass_sensor_data(
                    y,
                    sfreq_hz=residual_preprocessing["sfreq_hz"],
                    low_hz=residual_preprocessing["low_hz"],
                    high_hz=residual_preprocessing["high_hz"],
                    order=residual_preprocessing["order"],
                )
            residuals[sample_index, kind_index] = residual_rescore_draws(
                draws,
                residual_y,
                data["G_for_simulation"],
                vertices,
                graph,
                data["params_spatial"],
            )[0]
        if (sample_index + 1) % 5 == 0 or sample_index + 1 == n_samples:
            print(
                f"[anna-g-tau] {sample_index + 1}/{n_samples}; "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    args.out_dir.mkdir(parents=True)
    np.savez_compressed(
        args.out_dir / "dev_draws_residuals.npz",
        theta_draws=theta_draws,
        residuals=residuals,
        full_sample_indices=data["full_sample_indices"],
        seed=np.array(args.seed, dtype=np.int64),
        n_draws=np.array(args.n_draws, dtype=np.int64),
    )
    grid = derive_temperature_grid(residuals.reshape(-1, args.n_draws), "quadratic")
    candidates: list[dict[str, Any]] = [
        {"candidate_id": "argmin", "estimator": "argmin"}
    ]
    for exponent, temperature in zip(
        grid["exponents"], grid["temperatures"], strict=True
    ):
        candidates.append(
            {
                "candidate_id": f"quadratic_2^{int(exponent)}",
                "estimator": "gibbs",
                "score_form": "quadratic",
                "temperature_exponent": int(exponent),
                "tau": float(temperature),
            }
        )
    rows = []
    estimates_by_id = {}
    metrics_by_id = {}
    for candidate in candidates:
        estimates, weight_metrics = candidate_estimates(
            theta_draws,
            residuals,
            estimator=candidate["estimator"],
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
            **{key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)},
        }
        rows.append(row)
        estimates_by_id[candidate["candidate_id"]] = estimates
        metrics_by_id[candidate["candidate_id"]] = metrics
    selected = select_preregistered_candidate(rows)
    selected_id = selected["candidate_id"]
    model_files = {
        key: Path(spec[key]) for key in ("model_path", "stats_path", "meta_path")
    }
    model_digest = {
        key: {"path": str(path), "sha256": _sha256_file(path)}
        for key, path in model_files.items()
    }
    combined = hashlib.sha256()
    for key, entry in sorted(model_digest.items()):
        combined.update(key.encode())
        combined.update(entry["sha256"].encode())
    end_git = _git_state(repository)
    selection = {
        "schema_version": 1,
        "mode": "dev_tune",
        "model_key": MODEL_KEY,
        "model_digest": {"files": model_digest, "sha256": combined.hexdigest()},
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": _sha256_file(args.dataset),
        "dataset_split_kind": "dev",
        "n_samples": n_samples,
        "n_draws": args.n_draws,
        "seed": args.seed,
        "score_form_lock": "quadratic",
        "residual_preprocessing": residual_preprocessing,
        "temperature_grid": grid,
        "selected": selected,
        "run_start_git": start_git,
        "artifact_write_git": end_git,
        "repository_changed_during_run": start_git != end_git,
        "wall_seconds": time.perf_counter() - started,
    }
    if selection["repository_changed_during_run"] or end_git["dirty"]:
        raise RuntimeError("repository changed during temperature tuning")
    _atomic_json(args.out_dir / "selection.json", selection)
    _atomic_json(args.out_dir / "curve.json", {"rows": rows})
    chosen_metrics = metrics_by_id[selected_id]
    np.savez_compressed(
        args.out_dir / "selected_dev_predictions.npz",
        estimates=estimates_by_id[selected_id].astype(np.float32),
        predicted_centroids=chosen_metrics["predicted_centroids"],
        aligned_centroids=chosen_metrics["aligned_centroids"],
        per_sample_mae_euc_mm=chosen_metrics["per_sample_mae_euc_mm"],
        per_sample_deltax_euc_mm=chosen_metrics["per_sample_deltax_euc_mm"],
        full_sample_indices=data["full_sample_indices"],
    )
    (args.out_dir / "report.md").write_text(
        "\n".join(
            [
                "# Anna-G quadratic temperature freeze",
                "",
                f"- Model: `{MODEL_KEY}`",
                f"- Dataset: `{args.dataset.resolve()}` (`dev`, n={n_samples})",
                f"- Draws per input: {args.n_draws}",
                "- Score form was locked to quadratic before outcomes.",
                "- Selection: minimum Euclidean Delta X prime subject to MAE <= 1.1 x argmin MAE.",
                f"- Selected: `{selected_id}`; tau={selected.get('tau')}",
                f"- Dev MAE euc: {selected['mae_euc_mm']:.4f} mm",
                f"- Dev Delta X prime euc: {selected['deltax_euc_mm']:.4f} mm",
                "- No held-out or Anna outcome dataset was loaded.",
                "",
            ]
        )
    )
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
