#!/usr/bin/env python3
"""Combine paired-shift run artifacts into one summary and per-sample table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing the paired-shift run folders",
    )
    parser.add_argument(
        "--stamp",
        default="20260810_141840",
        help="Run suffix shared by the input folders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    runs = {
        "dG = 0": root / f"shift_paired_dg0_{args.stamp}",
        "dG != 0": root / f"shift_paired_dgnz_{args.stamp}",
    }
    cpu = root / f"shift_paired_cpu_timing_{args.stamp}"
    summaries = {
        name: json.loads((path / "summary.json").read_text()) for name, path in runs.items()
    }
    cpu_summary = json.loads((cpu / "summary.json").read_text())
    combined = {
        "created_utc": "2026-08-10",
        "estimator": "posterior_mean",
        "common_random_numbers": True,
        "n_posterior": 25,
        "ode_steps": 96,
        "pool": 12,
        "rap_music_delta_euc_mm": {"dG = 0": 5.85, "dG != 0": 6.13},
        "scenarios": summaries,
        "timing": {
            "gpu_dG0": summaries["dG = 0"]["timing"],
            "gpu_dG_nonzero": summaries["dG != 0"]["timing"],
            "cpu": cpu_summary["timing"],
        },
        "reference_posterior_mean_rechecks": {
            "n25_fixed_test_mae_geo_mm": 9.479570,
            "n25_fixed_test_mae_euc_mm": 4.885824,
            "n100_fixed_test_mae_geo_mm": 9.093386,
            "n100_fixed_test_mae_euc_mm": 4.645638,
        },
        "artifacts": {
            "dG0_run": str(runs["dG = 0"]),
            "dG_nonzero_run": str(runs["dG != 0"]),
            "cpu_timing_run": str(cpu),
            "fixed_test_recheck_n25": str(root / "shift_paired_reference_recheck_20260810_141514"),
            "fixed_test_recheck_n100": str(
                root / "shift_paired_reference_recheck_n100_20260810_141514"
            ),
        },
    }
    (root / f"shift_paired_summary_{args.stamp}.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for scenario, path in runs.items():
        with (path / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows.extend({"scenario": scenario, **row} for row in csv.DictReader(stream))
    fields = ["scenario"] + sorted({key for row in rows for key in row if key != "scenario"})
    with (root / f"shift_paired_per_sample_metrics_{args.stamp}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
