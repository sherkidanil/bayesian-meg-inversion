#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


RAP = {
    "dG = 0": {"delta_x_prime_mm": 5.85, "mae_geo_mm": 4.05},
    "dG != 0": {"delta_x_prime_mm": 6.13, "mae_geo_mm": 4.78},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dg0", type=Path, required=True)
    parser.add_argument("--dgnz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scenarios = {
        "dG = 0": json.loads((args.dg0 / "summary.json").read_text(encoding="utf-8")),
        "dG != 0": json.loads((args.dgnz / "summary.json").read_text(encoding="utf-8")),
    }
    rows = []
    for scenario, summary in scenarios.items():
        rows.append(
            {
                "scenario": scenario,
                "input_delta_x_mm": summary["input_delta_euc_mm_by_source"]["mean"],
                "cfm_delta_x_prime_mm": summary["output_delta_euc_mm_by_sample"]["mean"],
                "cfm_mae_geo_mm": summary["baseline_original"]["mae_geo_mm"],
                "rap_music_delta_x_prime_mm": RAP[scenario]["delta_x_prime_mm"],
                "rap_music_mae_geo_mm": RAP[scenario]["mae_geo_mm"],
            }
        )
    payload = {
        "estimator": scenarios["dG = 0"]["estimator"],
        "n_posterior": scenarios["dG = 0"]["n_posterior"],
        "common_random_numbers": True,
        "rows": rows,
        "source_runs": {"dG = 0": str(args.dg0), "dG != 0": str(args.dgnz)},
        "rap_music_accuracy_source": "collaborator-feedback.md: MAE Geo was 4.05 mm, became 4.78 mm",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Honest paired CFM stability summary",
        "",
        f"Estimator: `{payload['estimator']}`, N={payload['n_posterior']}; same draw keys for each paired Y/Y-shifted prediction.",
        "",
        "| Scenario | Input delta X, mm | CFM delta X prime, mm | CFM MAE geo, mm | RAP MUSIC delta X prime, mm | RAP MUSIC MAE geo, mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['input_delta_x_mm']:.4f} | "
            f"{row['cfm_delta_x_prime_mm']:.4f} | {row['cfm_mae_geo_mm']:.4f} | "
            f"{row['rap_music_delta_x_prime_mm']:.2f} | {row['rap_music_mae_geo_mm']:.2f} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
