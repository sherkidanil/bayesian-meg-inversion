import json

import pytest

from megflow.m1_comparison import compare_m1_outputs


def _write_result(path, architecture, values):
    path.mkdir()
    metadata = {
        "split": "dev",
        "architecture": architecture,
        "bank_size": 800,
        "euler_steps": 96,
        "initial_noise_scale": 1.5,
        "signal_rank": 6,
        "gibbs_tau": 0.0011995147478874274,
        "refinement_passes": 4,
        "radii_mm": [10.0, 8.0, 10.0],
        "evaluation_seed": 20260923,
        "config_sha256": "config",
        "anatomy_content_sha256": "anatomy",
        "dev_manifest_sha256": "dev",
        "heldout_manifest_sha256": "heldout",
        "sensor_projector_sha256": "projector",
        "decoder_code_sha256": "decoder",
        "checkpoint_sha256": f"checkpoint-{architecture}",
    }
    summary = {
        "metadata": metadata,
        "metrics": {
            "medoid": {"geodesic_mae_at_30_mm": sum(values) / len(values) + 1},
            "refined": {"geodesic_mae_at_30_mm": sum(values) / len(values)},
            "mean_effective_sample_size": 12.0,
        },
    }
    records = []
    for index, value in enumerate(values):
        records.append(
            {
                "record_index": index,
                "record_id": f"record-{index}",
                "sample_seed": 100 + index,
                "mean_effective_sample_size": 10.0 + index,
                "medoid_geodesic_mae_at_30_mm": value + 1,
                "refined_geodesic_mae_at_30_mm": value,
                "refined_geodesic_diameter_penalized_mae_mm": value + 2,
                "refined_euclidean_mae_mm": value / 2,
                "refined_geodesic_disconnected_count": index % 2,
            }
        )
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "records.json").write_text(json.dumps(records))


def test_compare_m1_outputs_is_paired_and_reports_refinement(tmp_path):
    mlp = tmp_path / "mlp"
    transformer = tmp_path / "transformer"
    _write_result(mlp, "mlp", [4.0, 6.0, 8.0])
    _write_result(transformer, "transformer", [3.0, 4.0, 5.0])

    result = compare_m1_outputs(
        mlp, transformer, bootstrap_samples=200, bootstrap_seed=17
    )

    assert result["split"] == "dev"
    assert result["record_count"] == 3
    primary = result["paired"]["refined_geodesic_mae_at_30_mm"]
    assert primary["transformer_minus_mlp_mean"] == pytest.approx(-2.0)
    assert primary["bootstrap_ci95"][1] < 0
    assert result["models"]["mlp"]["refinement_gain_geo_at_30_mm"] == 1.0
    assert result["models"]["transformer"]["refinement_gain_geo_at_30_mm"] == 1.0
    assert result["controlled_winner_geo_at_30"] == "transformer"


def test_compare_m1_outputs_rejects_unpaired_protocols(tmp_path):
    mlp = tmp_path / "mlp"
    transformer = tmp_path / "transformer"
    _write_result(mlp, "mlp", [4.0, 6.0])
    _write_result(transformer, "transformer", [3.0, 5.0])
    summary_path = transformer / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["metadata"]["bank_size"] = 400
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="bank_size"):
        compare_m1_outputs(mlp, transformer)
