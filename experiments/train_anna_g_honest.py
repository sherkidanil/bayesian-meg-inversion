#!/usr/bin/env python3
"""Train the single locked honest_4965 configuration on Anna-G data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

import numpy as np


EXPECTED_HONEST_PARAMS: dict[str, Any] = {
    "pool": 12,
    "lr": 0.0020157680257224214,
    "batch_size": 256,
    "steps": 16000,
    "cond_dim": 192,
    "cnn_width1": 64,
    "cnn_width2": 128,
    "cnn_kernel": 15,
    "cnn_stride": 1,
    "enc_act": "relu",
    "vf_width": 512,
    "vf_depth": 3,
    "vf_act": "silu",
    "permute_sources": False,
    "min_perm_loss": False,
    "use_weight_decay": True,
    "weight_decay": 4.928287692020457e-08,
    "clip_norm": 0.0,
    "ode_steps": 96,
    "n_post_samples": 9,
    "seed": 303,
}

GEOMETRY_KEYS = ("vertices", "edge_index", "edge_weight_mm")
PARAMETER_KEYS = ("params_spatial", "params_temporal", "params_noise")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def load_locked_hyperparameters(reference_result: Path) -> dict[str, Any]:
    reference = json.loads(reference_result.read_text())
    for key, expected in EXPECTED_HONEST_PARAMS.items():
        if key not in reference or reference[key] != expected:
            raise ValueError(
                f"honest_4965 parameter {key} differs: "
                f"expected {expected!r}, found {reference.get(key)!r}"
            )
    return {**EXPECTED_HONEST_PARAMS, "candidate_id": "honest_4965_annaG"}


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(np.asarray(left), np.asarray(right))
    return bool(left == right)


def validate_training_dataset_contract(
    train_path: Path, anna_path: Path, reference_path: Path
) -> dict[str, Any]:
    with np.load(train_path, allow_pickle=True) as train, np.load(
        anna_path, allow_pickle=True
    ) as anna, np.load(reference_path, allow_pickle=True) as reference:
        for key in GEOMETRY_KEYS:
            if not np.array_equal(train[key], anna[key]) or not np.array_equal(
                train[key], reference[key]
            ):
                raise ValueError(f"training geometry differs at {key}")
        train_g = np.asarray(train["G_for_simulation"])
        anna_g = np.asarray(anna["G_for_simulation"][0])
        if train_g.ndim != 2 or not np.array_equal(train_g, anna_g):
            raise ValueError("training dataset does not persist Anna lead field")
        for key in PARAMETER_KEYS:
            train_value = dict(train[key].item())
            reference_value = dict(reference[key].item())
            if not _nested_equal(train_value, reference_value):
                raise ValueError(f"historical generator parameters differ at {key}")
        stored_noise = np.asarray(train["noise_level"])
        expected_noise = np.array(0.3, dtype=np.float32)
        if stored_noise.dtype != np.float32 or not np.array_equal(
            stored_noise, expected_noise
        ):
            raise ValueError("historical noise_level must remain 0.3")
        meta = dict(train["meta"].item())
        if meta.get("only_changed_factor") != "leadfield":
            raise ValueError("training metadata does not lock leadfield as the only factor")
    return {
        "only_changed_factor": "leadfield",
        "leadfield_array_sha256": _array_sha256(train_g),
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "jax", "equinox", "optax"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": packages,
    }


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--fixed-test", type=Path, required=True)
    parser.add_argument("--anna-dg0", type=Path, required=True)
    parser.add_argument("--reference-train", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository = Path(__file__).resolve().parent
    start_git = _git_state(repository)
    if start_git["dirty"]:
        raise RuntimeError("training requires a clean repository at run start")
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    contract = validate_training_dataset_contract(
        args.train, args.anna_dg0, args.reference_train
    )
    cfg = load_locked_hyperparameters(args.reference_result)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    mesh_root = repository / "new_version_mesh_experiments"
    sys.path.insert(0, str(mesh_root))
    from trial81_fulltest_search import (  # noqa: PLC0415
        load_datasets,
        train_and_evaluate_candidate,
    )

    full, fixed_test = load_datasets(args.train, args.fixed_test)
    args.out_dir.mkdir(parents=True)
    result = train_and_evaluate_candidate(
        cfg=cfg,
        full=full,
        fixed_test=fixed_test,
        out_dir=args.out_dir,
        phase="final",
    )
    end_git = _git_state(repository)
    if start_git != end_git or end_git["dirty"]:
        raise RuntimeError("repository changed during training")
    provenance = {
        "schema": "honest-4965-annaG-training-v1",
        "model_name": "honest_4965_annaG",
        "contract": contract,
        "hyperparameters": cfg,
        "environment": _environment(),
        "git": end_git,
        "inputs": {
            "train": {"path": str(args.train), "sha256": _sha256_file(args.train)},
            "fixed_test": {
                "path": str(args.fixed_test),
                "sha256": _sha256_file(args.fixed_test),
            },
            "anna_dg0": {"path": str(args.anna_dg0), "sha256": _sha256_file(args.anna_dg0)},
            "reference_result": {
                "path": str(args.reference_result),
                "sha256": _sha256_file(args.reference_result),
            },
        },
        "training_result": result,
        "best_model_alias_changed": False,
    }
    (args.out_dir / "training_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
