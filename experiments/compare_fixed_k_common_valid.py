#!/usr/bin/env python3
"""Compare fixed-K=3 methods on identical records without 30 mm penalties.

A record belongs to a common-valid set only when every matched geodesic
distance is finite and strictly below 30 mm for every method defining that
set. The reported MAE is the ordinary, uncapped mean of the saved raw matched
distances. The 30 mm threshold is used only to select records; no distance is
clipped or replaced in the reported metric.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

CUTOFF_MM = 30.0
BOOTSTRAP_SEED = 20261005
BOOTSTRAP_SAMPLES = 200_000
REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "reports" / "transformer-common-valid"
RAW = ROOT / "raw"
DERIVED = ROOT

METHODS = {
    "mlp_gibbs": {
        "label": "MLP + Gibbs-weighted medoid",
        "path": RAW / "mlp_gibbs_records.json",
    },
    "transformer_gibbs": {
        "label": "Transformer + Gibbs-weighted medoid",
        "path": RAW / "transformer_gibbs_records.json",
    },
    "transformer_maxrho": {
        "label": "Transformer + max-rho",
        "path": RAW / "transformer_maxrho_records.json",
    },
    "finetuned_transformer_maxrho": {
        "label": "Fine-tuned Transformer + max-rho",
        "path": RAW / "finetuned_transformer_maxrho_records.json",
    },
}

SETS = {
    "architecture": ["mlp_gibbs", "transformer_gibbs"],
    "all_four": list(METHODS),
}


def load_method(path: Path) -> dict[int, dict]:
    rows = json.loads(path.read_text())
    output = {}
    for row in rows:
        index = int(row["record_index"])
        raw = row["refined_assigned_geodesic_mm"]
        distances = [
            float(value) if value is not None and math.isfinite(value) else None for value in raw
        ]
        if len(distances) != 3:
            raise AssertionError(f"record {index}: expected K=3, got {len(distances)}")
        valid = all(value is not None and value < CUTOFF_MM for value in distances)
        output[index] = {
            "record_id": row["record_id"],
            "distances_mm": distances,
            "valid": valid,
            "mae_geo_mm": float(np.mean(distances)) if valid else None,
        }
    if len(output) != len(rows):
        raise AssertionError(f"duplicate record_index in {path}")
    return output


def check_alignment(data: dict[str, dict[int, dict]]) -> list[int]:
    method_names = list(data)
    reference = data[method_names[0]]
    indices = sorted(reference)
    for method in method_names[1:]:
        if sorted(data[method]) != indices:
            raise AssertionError(f"record indices differ for {method}")
        for index in indices:
            if data[method][index]["record_id"] != reference[index]["record_id"]:
                raise AssertionError(f"record_id differs for {method}, index {index}")
    if len(indices) != 350:
        raise AssertionError(f"expected 350 records, got {len(indices)}")
    return indices


def descriptive(values: np.ndarray) -> dict:
    return {
        "mean_mae_geo_mm": float(np.mean(values)),
        "median_record_mae_geo_mm": float(np.median(values)),
        "p90_record_mae_geo_mm": float(np.quantile(values, 0.9)),
        "max_record_mae_geo_mm": float(np.max(values)),
    }


def paired_statistics(mlp: np.ndarray, transformer: np.ndarray) -> dict:
    difference = transformer - mlp
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(difference)
    batch_size = 10_000
    bootstrap_means = []
    remaining = BOOTSTRAP_SAMPLES
    while remaining:
        size = min(batch_size, remaining)
        sampled = rng.integers(0, n, size=(size, n))
        bootstrap_means.append(np.mean(difference[sampled], axis=1))
        remaining -= size
    bootstrap_means = np.concatenate(bootstrap_means)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    p_value = float(
        wilcoxon(
            difference,
            alternative="two-sided",
            zero_method="wilcox",
            method="approx",
        ).pvalue
    )
    return {
        "mean_paired_difference_transformer_minus_mlp_mm": float(np.mean(difference)),
        "relative_change_vs_mlp_percent": float(100.0 * np.mean(difference) / np.mean(mlp)),
        "bootstrap_95_ci_difference_mm": [float(ci_low), float(ci_high)],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "wilcoxon_two_sided_approximate_p": p_value,
        "transformer_wins": int(np.sum(difference < 0.0)),
        "ties": int(np.sum(difference == 0.0)),
        "transformer_losses": int(np.sum(difference > 0.0)),
    }


def write_manifest(
    name: str,
    common_indices: list[int],
    methods: list[str],
    data: dict[str, dict[int, dict]],
) -> None:
    path = DERIVED / f"fixed_k_common_valid_{name}_records.csv"
    fields = ["record_index", "record_id"]
    for method in methods:
        fields.extend([f"{method}_assigned_mm", f"{method}_mae_geo_mm"])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index in common_indices:
            row = {
                "record_index": index,
                "record_id": data[methods[0]][index]["record_id"],
            }
            for method in methods:
                row[f"{method}_assigned_mm"] = json.dumps(
                    data[method][index]["distances_mm"], separators=(",", ":")
                )
                row[f"{method}_mae_geo_mm"] = data[method][index]["mae_geo_mm"]
            writer.writerow(row)


def main() -> None:
    data = {name: load_method(config["path"]) for name, config in METHODS.items()}
    indices = check_alignment(data)
    DERIVED.mkdir(parents=True, exist_ok=True)

    summary = {
        "definition": {
            "k": 3,
            "total_records": len(indices),
            "valid_record": (
                "all three refined_assigned_geodesic_mm values are finite, "
                "non-null, and strictly below 30.0 mm for every method in the set"
            ),
            "metric": (
                "mean over records of the per-record mean of three raw matched "
                "geodesic distances; no clipping and no 30 mm substitution"
            ),
        },
        "individual_method_validity": {},
        "sets": {},
    }
    flat_rows = []

    for method in METHODS:
        valid_count = sum(data[method][index]["valid"] for index in indices)
        summary["individual_method_validity"][method] = {
            "label": METHODS[method]["label"],
            "valid_records": valid_count,
            "valid_percent": 100.0 * valid_count / len(indices),
            "invalid_records": len(indices) - valid_count,
            "invalid_percent": 100.0 * (len(indices) - valid_count) / len(indices),
        }

    for set_name, methods in SETS.items():
        common_indices = [
            index for index in indices if all(data[method][index]["valid"] for method in methods)
        ]
        excluded = len(indices) - len(common_indices)
        set_result = {
            "methods_defining_set": methods,
            "used_records": len(common_indices),
            "coverage_percent": 100.0 * len(common_indices) / len(indices),
            "excluded_records": excluded,
            "excluded_percent": 100.0 * excluded / len(indices),
            "methods": {},
        }
        for method in methods:
            values = np.asarray([data[method][index]["mae_geo_mm"] for index in common_indices])
            metrics = descriptive(values)
            set_result["methods"][method] = {
                "label": METHODS[method]["label"],
                **metrics,
            }
            flat_rows.append(
                {
                    "set": set_name,
                    "method": METHODS[method]["label"],
                    "used_records": len(common_indices),
                    "coverage_percent": set_result["coverage_percent"],
                    "excluded_records": excluded,
                    "excluded_percent": set_result["excluded_percent"],
                    **metrics,
                }
            )

        if set_name == "architecture":
            mlp = np.asarray([data["mlp_gibbs"][index]["mae_geo_mm"] for index in common_indices])
            transformer = np.asarray(
                [data["transformer_gibbs"][index]["mae_geo_mm"] for index in common_indices]
            )
            set_result["paired_comparison"] = paired_statistics(mlp, transformer)

        summary["sets"][set_name] = set_result
        write_manifest(set_name, common_indices, methods, data)

    (DERIVED / "fixed_k_common_valid_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (DERIVED / "fixed_k_common_valid_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)


if __name__ == "__main__":
    main()
