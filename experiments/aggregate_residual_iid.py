#!/usr/bin/env python3
"""Aggregate paired residual-selected IID runs across models and data seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from residual_rescoring_dataset import paired_comparisons


MODELS = ("trial81_original", "trial81_honest_4965", "trial81_push_3771")


def aggregate_runs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("No run summaries")
    runs: list[dict[str, Any]] = []
    for summary in summaries:
        if summary["estimator"] != "residual_selected" or summary["n_draws"] != 100:
            raise ValueError("Expected residual_selected with 100 draws")
        if summary["repository_changed_during_run"]:
            raise ValueError("Repository changed during a run")
        runs.append(
            {
                "model_key": summary["model_key"],
                "repeat": int(summary["repeat"]),
                "data_seed": int(summary["data_seed"]),
                "eval_seed": int(summary["eval_seed"]),
                "n_samples": int(summary["n_samples"]),
                "mae_geo_mm": float(summary["metrics"]["mae_geo_mm"]["mean"]),
                "mae_euc_mm": float(summary["metrics"]["mae_euc_mm"]["mean"]),
                "n_nonfinite_geo": int(
                    summary["metrics"]["mae_geo_mm"]["n_nonfinite"]
                ),
                "git_commit": summary["run_start_git_commit"],
            }
        )

    ranking: list[dict[str, Any]] = []
    for model in sorted({str(row["model_key"]) for row in runs}):
        group = sorted(
            [row for row in runs if row["model_key"] == model],
            key=lambda row: int(row["repeat"]),
        )
        geo = np.asarray([row["mae_geo_mm"] for row in group], dtype=np.float64)
        euc = np.asarray([row["mae_euc_mm"] for row in group], dtype=np.float64)
        ranking.append(
            {
                "model_key": model,
                "runs": len(group),
                "mae_geo_mm_mean": float(np.mean(geo)),
                "mae_geo_mm_std": float(np.std(geo)),
                "mae_euc_mm_mean": float(np.mean(euc)),
                "mae_euc_mm_std": float(np.std(euc)),
                "n_nonfinite_geo_total": int(
                    sum(row["n_nonfinite_geo"] for row in group)
                ),
            }
        )
    ranking.sort(key=lambda row: row["mae_geo_mm_mean"])
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return {
        "protocol": {
            "estimator": "residual_selected",
            "n_draws": 100,
            "comparison": "paired by repeat and data_seed",
        },
        "runs": sorted(runs, key=lambda row: (row["model_key"], row["repeat"])),
        "ranking": ranking,
        "paired_geo": paired_comparisons(runs, metric="mae_geo_mm"),
        "paired_euc": paired_comparisons(runs, metric="mae_euc_mm"),
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Residual-selected Fresh IID-100 × 3 ranking",
        "",
        "All comparisons are paired by identical `data_seed`; the ranking metric is finite-only mean mesh-geodesic MAE, matching the prior project reports.",
        "",
        "| Rank | Model | MAE geo, mm | MAE euc, mm | Non-finite geo |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in result["ranking"]:
        lines.append(
            f"| {row['rank']} | `{row['model_key']}` | "
            f"{row['mae_geo_mm_mean']:.4f} ± {row['mae_geo_mm_std']:.4f} | "
            f"{row['mae_euc_mm_mean']:.4f} ± {row['mae_euc_mm_std']:.4f} | "
            f"{row['n_nonfinite_geo_total']} |"
        )
    lines.extend(["", "## Per-repeat paired results", ""])
    for row in result["runs"]:
        lines.append(
            f"- `{row['model_key']}`, repeat {row['repeat']}, data_seed {row['data_seed']}: "
            f"geo {row['mae_geo_mm']:.4f} mm; euc {row['mae_euc_mm']:.4f} mm; "
            f"non-finite geo {row['n_nonfinite_geo']}."
        )
    lines.extend(["", "## Pairwise differences (A − B), geodesic mm", ""])
    for comparison in result["paired_geo"]:
        differences = ", ".join(f"{value:+.4f}" for value in comparison["differences_a_minus_b"])
        lines.append(
            f"- `{comparison['model_a']}` − `{comparison['model_b']}`: [{differences}], "
            f"mean {comparison['mean_difference_a_minus_b']:+.4f}; wins "
            f"{comparison['a_wins']}:{comparison['b_wins']} (ties {comparison['ties']})."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = [root / model / f"repeat{repeat}" / "summary.json" for model in MODELS for repeat in range(1, 4)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run summaries: {missing}")
    result = aggregate_runs(
        [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    )
    (root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_report(root / "report.md", result)
    print(json.dumps(result["ranking"], indent=2))


if __name__ == "__main__":
    main()
