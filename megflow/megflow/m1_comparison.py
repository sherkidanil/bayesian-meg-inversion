"""Paired statistical comparison for frozen M1 decoder outputs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .dataset import sha256


PROTOCOL_KEYS = (
    "split",
    "bank_size",
    "euler_steps",
    "initial_noise_scale",
    "signal_rank",
    "gibbs_tau",
    "refinement_passes",
    "radii_mm",
    "evaluation_seed",
    "config_sha256",
    "anatomy_content_sha256",
    "dev_manifest_sha256",
    "heldout_manifest_sha256",
    "sensor_projector_sha256",
    "decoder_code_sha256",
)

PAIRED_METRICS = (
    "refined_geodesic_mae_at_30_mm",
    "refined_geodesic_diameter_penalized_mae_mm",
    "refined_euclidean_mae_mm",
    "refined_geodesic_disconnected_count",
    "mean_effective_sample_size",
)


def _load_result(path):
    directory = Path(path)
    summary = json.loads((directory / "summary.json").read_text())
    records = json.loads((directory / "records.json").read_text())
    if summary["metadata"].get("record_count") not in (None, len(records)):
        raise ValueError("summary and record count differ")
    return summary, records


def _paired_stat(differences, *, bootstrap_samples, rng):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired differences must be a nonempty finite vector")
    draws = values[
        rng.integers(0, len(values), size=(int(bootstrap_samples), len(values)))
    ].mean(axis=1)
    pvalue = (
        1.0
        if np.all(values == 0)
        else float(wilcoxon(values, alternative="two-sided").pvalue)
    )
    return {
        "transformer_minus_mlp_mean": float(values.mean()),
        "bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "wilcoxon_two_sided_p": pvalue,
        "n_pairs": int(len(values)),
    }


def _model_summary(summary, records):
    medoid = np.asarray(
        [row["medoid_geodesic_mae_at_30_mm"] for row in records], dtype=float
    )
    refined = np.asarray(
        [row["refined_geodesic_mae_at_30_mm"] for row in records], dtype=float
    )
    return {
        **summary["metrics"],
        "refinement_gain_geo_at_30_mm": float(np.mean(medoid - refined)),
    }


def compare_m1_outputs(
    mlp_path,
    transformer_path,
    *,
    bootstrap_samples=10_000,
    bootstrap_seed=20260924,
):
    """Compare aligned MLP and Transformer records under one frozen protocol."""
    mlp_summary, mlp_records = _load_result(mlp_path)
    transformer_summary, transformer_records = _load_result(transformer_path)
    mlp_metadata = mlp_summary["metadata"]
    transformer_metadata = transformer_summary["metadata"]
    if mlp_metadata.get("architecture") != "mlp":
        raise ValueError("MLP result has the wrong architecture")
    if transformer_metadata.get("architecture") != "transformer":
        raise ValueError("Transformer result has the wrong architecture")
    for key in PROTOCOL_KEYS:
        if mlp_metadata.get(key) != transformer_metadata.get(key):
            raise ValueError(f"paired protocols differ: {key}")
    if len(mlp_records) != len(transformer_records) or not mlp_records:
        raise ValueError("paired outputs must contain the same nonzero record count")
    for mlp_row, transformer_row in zip(mlp_records, transformer_records):
        identity = ("record_index", "record_id", "sample_seed")
        if any(mlp_row[key] != transformer_row[key] for key in identity):
            raise ValueError("paired record identities differ")

    rng = np.random.default_rng(int(bootstrap_seed))
    paired = {}
    for metric in PAIRED_METRICS:
        mlp_values = np.asarray([row[metric] for row in mlp_records], dtype=float)
        transformer_values = np.asarray(
            [row[metric] for row in transformer_records], dtype=float
        )
        paired[metric] = _paired_stat(
            transformer_values - mlp_values,
            bootstrap_samples=bootstrap_samples,
            rng=rng,
        )
    mlp_model = _model_summary(mlp_summary, mlp_records)
    transformer_model = _model_summary(transformer_summary, transformer_records)
    mlp_primary = float(mlp_model["refined"]["geodesic_mae_at_30_mm"])
    transformer_primary = float(
        transformer_model["refined"]["geodesic_mae_at_30_mm"]
    )
    winner = "tie"
    if transformer_primary < mlp_primary:
        winner = "transformer"
    elif mlp_primary < transformer_primary:
        winner = "mlp"
    return {
        "schema": "m1-frozen-decoder-paired-comparison-v1",
        "split": mlp_metadata["split"],
        "record_count": len(mlp_records),
        "protocol": {key: mlp_metadata[key] for key in PROTOCOL_KEYS},
        "checkpoint_sha256": {
            "mlp": mlp_metadata["checkpoint_sha256"],
            "transformer": transformer_metadata["checkpoint_sha256"],
        },
        "models": {"mlp": mlp_model, "transformer": transformer_model},
        "paired": paired,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "controlled_winner_geo_at_30": winner,
        "historical_reference": {
            "model": "V1 temporal-attention CFM, seed 303",
            "geodesic_mae_mm": 1.8713077781222642,
            "record_mask_n": 286,
            "branches_n": 572,
            "scope": "locked historical heldout common mask over original and shifted branches",
            "comparison_limit": (
                "Reference only: it uses another checkpoint, dataset representation, "
                "paired-domain mask, and training protocol."
            ),
        },
    }


def artifact_hashes(paths):
    result = {}
    for requested in paths:
        path = Path(requested)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file_path in files:
            if file_path.is_file():
                result[str(file_path)] = {
                    "bytes": file_path.stat().st_size,
                    "sha256": sha256(file_path),
                }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-hashes", type=Path, required=True)
    parser.add_argument("--extra-artifact", type=Path, action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    args = parser.parse_args(argv)
    if args.output.exists() or args.artifact_hashes.exists():
        raise FileExistsError("comparison outputs are immutable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison = compare_m1_outputs(
        args.mlp,
        args.transformer,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    hashes = artifact_hashes(
        [args.mlp, args.transformer, args.output, *args.extra_artifact]
    )
    args.artifact_hashes.write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
