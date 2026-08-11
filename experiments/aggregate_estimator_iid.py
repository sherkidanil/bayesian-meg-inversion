#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = ("trial81_original", "trial81_honest_4965", "trial81_push_3771")
DATA_SEEDS = (202603260, 202603261, 202603262)
EVAL_SEEDS = (202610260, 202610261, 202610262)
ESTIMATOR = "posterior_mean_naive"
N_DRAWS = 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    runs: list[dict[str, object]] = []
    for model in MODELS:
        for repeat in range(1, 4):
            source = root / model / f"repeat{repeat}" / "summary.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            metrics = payload["results"][str(N_DRAWS)][ESTIMATOR]
            runs.append(
                {
                    "model_key": model,
                    "repeat": repeat,
                    "data_seed": DATA_SEEDS[repeat - 1],
                    "eval_seed": EVAL_SEEDS[repeat - 1],
                    "mae_geo_mm": metrics["mae_geo_mm"],
                    "mae_euc_mm": metrics["mae_euc_mm"],
                    "hit_geo@10mm": metrics["hit_geo@10mm"],
                    "hit_geo@15mm": metrics["hit_geo@15mm"],
                    "n_nonfinite_geo": metrics["mae_geo_mm_n_nonfinite"],
                    "source": str(source),
                }
            )

    ranked: list[dict[str, object]] = []
    for model in MODELS:
        group = [row for row in runs if row["model_key"] == model]
        item: dict[str, object] = {"model_key": model, "runs": len(group)}
        for metric in ("mae_geo_mm", "mae_euc_mm", "hit_geo@10mm", "hit_geo@15mm"):
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values))
        item["n_nonfinite_geo_total"] = int(sum(int(row["n_nonfinite_geo"]) for row in group))
        ranked.append(item)
    ranked.sort(key=lambda row: float(row["mae_geo_mm_mean"]))
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    result = {
        "selection_frozen_on_dev": {"estimator": ESTIMATOR, "n_draws": N_DRAWS},
        "iid_protocol": {
            "data_seeds": list(DATA_SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "note": "Estimator and draw count were selected on fixed dev before these IID runs.",
        },
        "runs": runs,
        "ranking": ranked,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Honest estimator confirmation on Fresh IID-100 x 3",
        "",
        f"Frozen estimator: `{ESTIMATOR}`, N={N_DRAWS}. It was selected on fixed dev before IID evaluation.",
        "",
        "| Rank | Model | MAE geo, mm | MAE euc, mm | Hit geo @10 | Hit geo @15 | Nonfinite geo |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['rank']} | {row['model_key']} | "
            f"{row['mae_geo_mm_mean']:.4f} +/- {row['mae_geo_mm_std']:.4f} | "
            f"{row['mae_euc_mm_mean']:.4f} +/- {row['mae_euc_mm_std']:.4f} | "
            f"{row['hit_geo@10mm_mean']:.4f} +/- {row['hit_geo@10mm_std']:.4f} | "
            f"{row['hit_geo@15mm_mean']:.4f} +/- {row['hit_geo@15mm_std']:.4f} | "
            f"{row['n_nonfinite_geo_total']} |"
        )
    lines.extend(["", "## Per-repeat results", ""])
    for row in runs:
        lines.append(
            f"- {row['model_key']} repeat {row['repeat']} (data seed {row['data_seed']}, "
            f"eval seed {row['eval_seed']}): MAE geo {row['mae_geo_mm']:.4f} mm."
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(ranked, indent=2))


if __name__ == "__main__":
    main()
