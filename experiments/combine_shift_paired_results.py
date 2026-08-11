#!/usr/bin/env python3
import csv
import json
from pathlib import Path


ROOT = Path("/mnt/local/tester/meg/reports")
STAMP = "20260810_141840"
RUNS = {
    "dG = 0": ROOT / f"shift_paired_dg0_{STAMP}",
    "dG != 0": ROOT / f"shift_paired_dgnz_{STAMP}",
}
CPU = ROOT / f"shift_paired_cpu_timing_{STAMP}"


def main():
    summaries = {name: json.loads((path / "summary.json").read_text()) for name, path in RUNS.items()}
    cpu = json.loads((CPU / "summary.json").read_text())
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
            "cpu": cpu["timing"],
        },
        "reference_posterior_mean_rechecks": {
            "n25_fixed_test_mae_geo_mm": 9.479570,
            "n25_fixed_test_mae_euc_mm": 4.885824,
            "n100_fixed_test_mae_geo_mm": 9.093386,
            "n100_fixed_test_mae_euc_mm": 4.645638,
        },
        "artifacts": {
            "dG0_run": str(RUNS["dG = 0"]),
            "dG_nonzero_run": str(RUNS["dG != 0"]),
            "cpu_timing_run": str(CPU),
            "fixed_test_recheck_n25": str(ROOT / "shift_paired_reference_recheck_20260810_141514"),
            "fixed_test_recheck_n100": str(ROOT / "shift_paired_reference_recheck_n100_20260810_141514"),
        },
    }
    (ROOT / f"shift_paired_summary_{STAMP}.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    for scenario, path in RUNS.items():
        with (path / "per_sample_metrics.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({"scenario": scenario, **row})
    fields = ["scenario"] + sorted({key for row in rows for key in row if key != "scenario"})
    with (ROOT / f"shift_paired_per_sample_metrics_{STAMP}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
