from pathlib import Path
import sys

TEST_ROOT = Path(__file__).resolve().parent
if TEST_ROOT.name == "tests":
    TEST_ROOT = TEST_ROOT.parent
sys.path.insert(0, str(TEST_ROOT))

from aggregate_residual_iid import aggregate_runs


def test_aggregate_runs_ranks_geo_and_preserves_paired_differences():
    summaries = []
    for model, values in {"a": [4.0, 6.0], "b": [5.0, 5.5]}.items():
        for repeat, value in enumerate(values, start=1):
            summaries.append(
                {
                    "model_key": model,
                    "repeat": repeat,
                    "data_seed": 100 + repeat,
                    "eval_seed": 200 + repeat,
                    "n_samples": 10,
                    "n_draws": 100,
                    "estimator": "residual_selected",
                    "repository_changed_during_run": False,
                    "run_start_git_commit": "abc",
                    "artifact_write_git_commit": "abc",
                    "metrics": {
                        "mae_geo_mm": {"mean": value, "n_nonfinite": 0},
                        "mae_euc_mm": {"mean": value / 2, "n_nonfinite": 0},
                    },
                }
            )

    result = aggregate_runs(summaries)

    assert [row["model_key"] for row in result["ranking"]] == ["a", "b"]
    comparison = result["paired_geo"][0]
    assert comparison["differences_a_minus_b"] == [-1.0, 0.5]
    assert comparison["a_wins"] == 1
    assert comparison["b_wins"] == 1
