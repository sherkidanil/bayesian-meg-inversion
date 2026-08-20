#!/usr/bin/env python3
"""Aggregate the frozen ±30% source-radius misspecification experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from source_misspecification import (
    build_global_common_mask,
    classify_misspecification_outcome,
    degradation_summary,
    hierarchical_relative_degradation_difference,
)


SCALES = ("0.7", "1.3")


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _canonical_cfm_artifact(
    artifact: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Normalize historical nominal S=400 and new misspecification schemas."""
    if "valid_pair" in artifact:
        return artifact
    required = {
        "mae_geo_mm": "s400_decoded_mae_geo_mm",
        "valid_branch": "s400_decoded_valid_branch",
        "valid_pair": "s400_decoded_valid_pair",
    }
    missing = [source for source in required.values() if source not in artifact]
    if missing:
        raise KeyError(f"CFM artifact lacks canonical and S=400 fields: {missing}")
    canonical = dict(artifact)
    canonical.update({target: artifact[source] for target, source in required.items()})
    return canonical


def _pair_geo(artifact: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.asarray(artifact["mae_geo_mm"], dtype=np.float64), axis=1)


def _bootstrap_degradation(
    nominal: np.ndarray,
    wrong: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    base = np.asarray(nominal, dtype=np.float64).reshape(-1)
    shifted = np.asarray(wrong, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(base), size=(int(n_resamples), len(base)))
    nominal_means = np.mean(base[indices], axis=1)
    absolute = np.mean((shifted - base)[indices], axis=1)
    relative = 100.0 * absolute / nominal_means
    return {
        "absolute_bootstrap_ci95_mm": np.quantile(absolute, [0.025, 0.975]).tolist(),
        "relative_bootstrap_ci95_percent": np.quantile(
            relative, [0.025, 0.975]
        ).tolist(),
    }


def _hierarchical_cfm_degradation(
    nominal: np.ndarray,
    wrong: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    base = np.asarray(nominal, dtype=np.float64)
    shifted = np.asarray(wrong, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    repeats = int(n_resamples)
    n_seeds, n_cases = base.shape
    absolute = np.empty(repeats, dtype=np.float64)
    relative = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        case_indices = rng.integers(0, n_cases, size=n_cases)
        base_draw = base[seed_indices][:, case_indices]
        wrong_draw = shifted[seed_indices][:, case_indices]
        nominal_mean = float(np.mean(base_draw))
        absolute[repeat] = float(np.mean(wrong_draw - base_draw))
        relative[repeat] = 100.0 * absolute[repeat] / nominal_mean
    return {
        "absolute_hierarchical_bootstrap_ci95_mm": np.quantile(
            absolute, [0.025, 0.975]
        ).tolist(),
        "relative_hierarchical_bootstrap_ci95_percent": np.quantile(
            relative, [0.025, 0.975]
        ).tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with np.load(Path(args.locked_masks).resolve(), allow_pickle=False) as payload:
        indices = np.asarray(payload["full_sample_indices"], dtype=np.int64)
        headline = np.asarray(payload["headline_mask"], dtype=bool)

    labels = list(args.seed_label)
    if not (
        len(labels)
        == len(args.cfm_nominal)
        == len(args.cfm_scale_070)
        == len(args.cfm_scale_130)
    ):
        raise ValueError("CFM seed labels and artifact lists must have equal lengths")
    patch = {
        "nominal": _load(Path(args.patch_nominal).resolve()),
        "0.7": _load(Path(args.patch_scale_070).resolve()),
        "1.3": _load(Path(args.patch_scale_130).resolve()),
    }
    cfm: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "nominal": {},
        "0.7": {},
        "1.3": {},
    }
    for label, nominal, low, high in zip(
        labels,
        args.cfm_nominal,
        args.cfm_scale_070,
        args.cfm_scale_130,
        strict=True,
    ):
        cfm["nominal"][label] = _canonical_cfm_artifact(
            _load(Path(nominal).resolve())
        )
        cfm["0.7"][label] = _canonical_cfm_artifact(_load(Path(low).resolve()))
        cfm["1.3"][label] = _canonical_cfm_artifact(_load(Path(high).resolve()))
    for method in list(patch.values()) + [
        artifact for scale in cfm.values() for artifact in scale.values()
    ]:
        if not np.array_equal(method["full_sample_indices"], indices):
            raise ValueError("one or more misspecification artifacts are not aligned")

    patch_valid = {
        scale: np.asarray(artifact["valid_pair"], dtype=bool)
        for scale, artifact in patch.items()
    }
    cfm_valid = {
        scale: np.stack(
            [np.asarray(cfm[scale][label]["valid_pair"], dtype=bool) for label in labels]
        )
        for scale in ("nominal", *SCALES)
    }
    common = build_global_common_mask(headline, patch_valid, cfm_valid)
    if not np.any(common):
        raise ValueError("global all-method/all-scale mask is empty")

    patch_geo = {scale: _pair_geo(artifact) for scale, artifact in patch.items()}
    cfm_geo = {
        scale: np.stack([_pair_geo(cfm[scale][label]) for label in labels])
        for scale in ("nominal", *SCALES)
    }
    repeats = int(args.bootstrap_resamples)
    by_scale: dict[str, Any] = {}
    for scale_index, scale in enumerate(SCALES):
        patch_summary = degradation_summary(
            patch_geo["nominal"][common], patch_geo[scale][common]
        )
        patch_summary.update(
            _bootstrap_degradation(
                patch_geo["nominal"][common],
                patch_geo[scale][common],
                n_resamples=repeats,
                seed=int(args.bootstrap_seed) + 10 * scale_index,
            )
        )
        cfm_per_seed = {
            label: degradation_summary(
                cfm_geo["nominal"][seed_index, common],
                cfm_geo[scale][seed_index, common],
            )
            for seed_index, label in enumerate(labels)
        }
        cfm_summary = degradation_summary(
            cfm_geo["nominal"][:, common].reshape(-1),
            cfm_geo[scale][:, common].reshape(-1),
        )
        cfm_summary["per_seed"] = cfm_per_seed
        cfm_summary.update(
            _hierarchical_cfm_degradation(
                cfm_geo["nominal"][:, common],
                cfm_geo[scale][:, common],
                n_resamples=repeats,
                seed=int(args.bootstrap_seed) + 10 * scale_index + 1,
            )
        )
        relative_difference = hierarchical_relative_degradation_difference(
            patch_geo["nominal"][common],
            patch_geo[scale][common],
            cfm_geo["nominal"][:, common],
            cfm_geo[scale][:, common],
            n_resamples=repeats,
            seed=int(args.bootstrap_seed) + 10 * scale_index + 2,
        )
        by_scale[scale] = {
            "patch_rap": patch_summary,
            "cfm": cfm_summary,
            "relative_difference": relative_difference,
        }

    outcome = classify_misspecification_outcome(by_scale)
    nonfinite = {
        "patch_rap": {
            scale: int(np.sum(~np.asarray(patch[scale]["valid_branch"], dtype=bool)))
            for scale in ("nominal", *SCALES)
        },
        "cfm_by_seed": {
            label: {
                scale: int(
                    np.sum(~np.asarray(cfm[scale][label]["valid_branch"], dtype=bool))
                )
                for scale in ("nominal", *SCALES)
            }
            for label in labels
        },
    }
    result: dict[str, Any] = {
        "schema": "source-radius-misspecification-aggregate-v1",
        "protocol_correction": {
            "six_point_three_five_zero_mm": (
                "point-source R->0 ablation, not the ±30% result"
            ),
            "historical_0p7_1p3_numbers": (
                "single-seed S=100 hard-argmin Anna-G/Anna-domain; not comparable "
                "to the three-seed S=400 server-domain baseline"
            ),
            "current_cfm": (
                "three V1 training seeds; S=400; nominal dev-frozen quadratic-medoid "
                "tau reused; only decoder Phi radii perturbed; no retraining"
            ),
        },
        "interpretation_rule_frozen_before_real_runs": {
            "learned_prior_more_robust": (
                "patch relative degradation exceeds CFM at both scales, with lower "
                "hierarchical CI > 0 and one-sided Wilcoxon p < 0.05"
            ),
            "patch_rap_more_robust": (
                "CFM relative degradation exceeds patch at both scales, with upper "
                "hierarchical CI < 0 and reverse one-sided Wilcoxon p < 0.05"
            ),
            "fallback": "comparable_or_direction_dependent",
        },
        "headline_pairs": int(np.sum(headline)),
        "global_common_pairs": int(np.sum(common)),
        "seed_labels": labels,
        "by_scale": by_scale,
        "nonfinite_branches_out_of_700": nonfinite,
        "outcome": outcome,
        "variant_2_warranted": outcome == "comparable_or_direction_dependent",
    }
    np.savez_compressed(
        output / "masks_and_errors.npz",
        full_sample_indices=indices,
        headline_mask=headline,
        global_common_mask=common,
        patch_nominal_geo_mm=patch_geo["nominal"],
        patch_scale_070_geo_mm=patch_geo["0.7"],
        patch_scale_130_geo_mm=patch_geo["1.3"],
        cfm_nominal_geo_by_seed_mm=cfm_geo["nominal"],
        cfm_scale_070_geo_by_seed_mm=cfm_geo["0.7"],
        cfm_scale_130_geo_by_seed_mm=cfm_geo["1.3"],
    )
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-masks", required=True)
    parser.add_argument("--patch-nominal", required=True)
    parser.add_argument("--patch-scale-070", required=True)
    parser.add_argument("--patch-scale-130", required=True)
    parser.add_argument("--seed-label", action="append", required=True)
    parser.add_argument("--cfm-nominal", action="append", required=True)
    parser.add_argument("--cfm-scale-070", action="append", required=True)
    parser.add_argument("--cfm-scale-130", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
