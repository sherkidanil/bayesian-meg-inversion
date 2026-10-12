"""Pure and I/O contracts for frozen V15 evaluation."""

import json

import numpy as np
import pytest


def test_evaluation_loader_accepts_only_dev_and_heldout(tmp_path, geometry):
    from megflow.dataset import sha256
    from megflow import v15
    from megflow.v15_dataset import load_evaluation_arrays, write_split

    simulation = v15.default_simulation_config()
    simulation.update(k_max=2, min_separation_mm=2.0)
    config = dict(
        pool=10,
        shard_size=1,
        counts=dict(train=1, val=1, dev=1, heldout=1),
        simulation=simulation,
    )
    leadfield = np.random.default_rng(4).normal(size=(4, 6)) * 1e-7
    for index, split in enumerate(("dev", "heldout")):
        record = v15.simulate(
            leadfield, geometry, index, simulation, apply_filter=False
        )
        write_split(
            iter([record]),
            tmp_path / split,
            split,
            config,
            "anatomy",
            "code",
            require_filtered=False,
        )
        (tmp_path / "audit.json").write_text(
            json.dumps(
                dict(manifest_sha256={split: sha256(tmp_path / split / "manifest.json")})
            )
        )
        loaded = load_evaluation_arrays(tmp_path / split)
        assert set(loaded) == {
            "Y",
            "theta",
            "radii",
            "centers",
            "mask",
            "K",
            "manifest",
        }
        assert loaded["Y"].shape[0] == 1
        assert loaded["centers"].shape == (1, 2)
    manifest = json.loads((tmp_path / "dev/manifest.json").read_text())
    manifest["split"] = "train"
    (tmp_path / "dev/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dev or heldout"):
        load_evaluation_arrays(tmp_path / "dev")


def test_evaluation_loader_rejects_a_manifest_not_frozen_by_root_audit(
    tmp_path, geometry
):
    from megflow import v15
    from megflow.v15_dataset import load_evaluation_arrays, write_split

    simulation = v15.default_simulation_config()
    simulation.update(k_max=1, min_separation_mm=2.0)
    config = dict(
        pool=10,
        shard_size=1,
        counts=dict(train=1, val=1, dev=1, heldout=1),
        simulation=simulation,
    )
    leadfield = np.random.default_rng(4).normal(size=(4, 6)) * 1e-7
    record = v15.simulate(leadfield, geometry, 7, simulation, apply_filter=False)
    write_split(
        iter([record]),
        tmp_path / "dev",
        "dev",
        config,
        "anatomy",
        "code",
        require_filtered=False,
    )
    (tmp_path / "audit.json").write_text(
        json.dumps(dict(manifest_sha256=dict(dev="wrong")))
    )
    with pytest.raises(ValueError, match="audit"):
        load_evaluation_arrays(tmp_path / "dev")


def test_nearest_vertex_projection_and_center_only_matching(geometry):
    from megflow.v15_evaluate import matched_source_errors, nearest_vertex_indices

    points = geometry.vertices[[0, 2]] + np.array([[0.1, 0, 0], [-0.1, 0, 0]])
    projected = nearest_vertex_indices(points, geometry.vertices)
    np.testing.assert_array_equal(projected, [0, 2])

    # Radii are intentionally adversarial. Matching must still follow centers.
    matched = matched_source_errors(
        true_centers=np.array([0, 2]),
        predicted_centers=np.array([2, 0]),
        true_radii=np.array([8.0, 25.0]),
        predicted_radii=np.array([8.0, 25.0]),
        geometry=geometry,
    )
    np.testing.assert_array_equal(matched["true_index"], [0, 1])
    np.testing.assert_array_equal(matched["predicted_index"], [1, 0])
    np.testing.assert_allclose(matched["euclidean_mm"], 0)
    np.testing.assert_allclose(matched["geodesic_mm"], 0)
    np.testing.assert_allclose(matched["radius_error_mm"], [17, -17])


def test_matching_preserves_disconnected_geodesic_errors(geometry):
    from megflow.v15_evaluate import matched_source_errors

    matched = matched_source_errors(
        true_centers=np.array([0]),
        predicted_centers=np.array([3]),
        true_radii=np.array([10.0]),
        predicted_radii=np.array([12.0]),
        geometry=geometry,
    )
    assert np.isinf(matched["geodesic_mm"][0])
    assert matched["disconnected_count"] == 1
    assert matched["radius_error_mm"].tolist() == [2.0]


def test_geodesic_candidate_matching_reports_connected_and_penalized_errors(
    geometry,
):
    from megflow.v15_evaluate import geodesic_candidate_errors

    matched = geodesic_candidate_errors(
        true_centers=np.array([0, 2]),
        predicted_centers=np.array([1, 2]),
        geometry=geometry,
        disconnected_penalty_mm=30.0,
    )

    np.testing.assert_allclose(matched["geodesic_mm"], [2.0, 0.0])
    assert matched["connected_count"] == 2
    assert matched["disconnected_count"] == 0
    assert matched["match_count"] == 2
    assert matched["connected_sum_mm"] == pytest.approx(2.0)
    assert matched["penalized_sum_mm"] == pytest.approx(2.0)


def test_geodesic_candidate_matching_minimizes_disconnections_first(geometry):
    from megflow.v15_evaluate import geodesic_candidate_errors

    matched = geodesic_candidate_errors(
        true_centers=np.array([0, 3]),
        predicted_centers=np.array([2, 1]),
        geometry=geometry,
        disconnected_penalty_mm=30.0,
    )

    assert matched["connected_count"] == 1
    assert matched["disconnected_count"] == 1
    assert matched["match_count"] == 2
    assert matched["connected_sum_mm"] == pytest.approx(2.0)
    assert matched["penalized_sum_mm"] == pytest.approx(32.0)


def test_geodesic_candidate_penalty_clips_long_connected_paths():
    from megflow.v15_evaluate import geodesic_candidate_errors

    class LongPathGeometry:
        def geodesic(self, true_centers, predicted_centers):
            return np.full(np.broadcast_shapes(true_centers.shape, predicted_centers.shape), 45.0)

    matched = geodesic_candidate_errors(
        true_centers=np.array([0]),
        predicted_centers=np.array([1]),
        geometry=LongPathGeometry(),
        disconnected_penalty_mm=30.0,
    )

    assert matched["connected_sum_mm"] == pytest.approx(45.0)
    assert matched["penalized_sum_mm"] == pytest.approx(30.0)


def test_recompute_geodesic_summary_uses_saved_projected_candidates(geometry):
    from megflow.v15_evaluate import recompute_geodesic_summary

    data = dict(
        centers=np.array([[0], [3]]),
        mask=np.ones((2, 1), dtype=bool),
        K=np.ones(2, dtype=np.int64),
        manifest=dict(split="dev"),
    )
    records = [
        dict(record_index=0, draw=0, true_k=1, projected_centers=[1]),
        dict(record_index=1, draw=0, true_k=1, projected_centers=[2]),
    ]

    summary = recompute_geodesic_summary(
        records, data, geometry, disconnected_penalty_mm=30.0
    )

    assert summary["record_count"] == 2
    assert summary["geodesic_connected_mae_mm"] == pytest.approx(2.0)
    assert summary["geodesic_penalized_mae_mm"] == pytest.approx(16.0)
    assert summary["geodesic_disconnected_rate"] == pytest.approx(0.5)
    assert summary["per_k"]["1"]["record_count"] == 2


def test_recompute_geodesic_summary_rejects_duplicate_or_missing_records(geometry):
    from megflow.v15_evaluate import recompute_geodesic_summary

    data = dict(
        centers=np.array([[0], [1]]),
        mask=np.ones((2, 1), dtype=bool),
        K=np.ones(2, dtype=np.int64),
        manifest=dict(split="dev"),
    )
    duplicate = [
        dict(record_index=0, draw=0, true_k=1, projected_centers=[0]),
        dict(record_index=0, draw=0, true_k=1, projected_centers=[1]),
    ]
    with pytest.raises(ValueError, match="exactly one primary draw"):
        recompute_geodesic_summary(duplicate, data, geometry)


def test_count_metrics_report_confusion_mae_and_per_k_accuracy():
    from megflow.v15_evaluate import count_metrics

    metrics = count_metrics(
        true_count=np.array([1, 1, 2, 2, 3]),
        predicted_count=np.array([1, 2, 2, 1, 3]),
        k_max=3,
    )
    assert metrics["exact_accuracy"] == pytest.approx(0.6)
    assert metrics["mae"] == pytest.approx(0.4)
    assert metrics["per_k_accuracy"] == {"1": 0.5, "2": 0.5, "3": 1.0}
    assert metrics["confusion"] == [[1, 1, 0], [1, 1, 0], [0, 0, 1]]


def test_poisson_binomial_probabilities_are_normalized_and_conditioned_nonempty():
    from megflow.v15_evaluate import poisson_binomial

    probabilities = poisson_binomial(np.array([0.2, 0.7, 0.5]))
    assert probabilities.shape == (4,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] > 0

    conditioned = poisson_binomial(
        np.array([0.2, 0.7, 0.5]), condition_nonempty=True
    )
    assert conditioned[0] == 0
    assert conditioned.sum() == pytest.approx(1.0)


def test_record_metrics_include_ospa_radius_bias_and_duplicates(geometry):
    from megflow.v15_evaluate import evaluate_record

    result = evaluate_record(
        true_centers=np.array([0, 2]),
        true_radii=np.array([10.0, 20.0]),
        predicted_xyz=np.array(
            [geometry.vertices[0], geometry.vertices[0], geometry.vertices[2]]
        ),
        predicted_radii=np.array([11.0, 12.0, 18.0]),
        geometry=geometry,
        ospa_cutoff_mm=30.0,
    )
    assert result["true_k"] == 2 and result["predicted_k"] == 3
    assert result["duplicate_count"] == 1
    assert result["ospa_mm"] == pytest.approx(10.0)
    assert result["radius_mae_mm"] == pytest.approx(1.5)
    assert result["radius_bias_mm"] == pytest.approx(-0.5)


def test_sample_sources_decodes_both_cardinality_modes_without_truth_k():
    import torch
    from megflow.v15_evaluate import sample_sources

    class CountModel:
        k_max = 3
        d_in = 4

        def encode(self, y):
            logits = torch.tensor([[0.0, 5.0, 0.0]], device=y.device).repeat(len(y), 1)
            return torch.zeros(len(y), 1, 2, device=y.device), logits

        def velocity(self, t, state, cond):
            return torch.zeros_like(state)

    class NullModel:
        k_max = 3
        d_in = 4

        def encode(self, y):
            return torch.zeros(len(y), 1, 2, device=y.device)

        def velocity_and_status(self, t, state, status, cond):
            logits = torch.tensor(
                [[[4.0, 0.0], [0.0, 4.0], [4.0, 0.0]]], device=y.device
            ).repeat(len(y), 1, 1)
            return torch.zeros_like(state), logits

    checkpoint = dict(state_mean=torch.zeros(4), state_std=torch.ones(4))
    y = torch.zeros(2, 4, 10)
    for mode, model, expected_k in [
        ("count_head", CountModel(), 2),
        ("null_slots", NullModel(), 2),
    ]:
        decoded = sample_sources(
            model,
            checkpoint,
            y,
            mode=mode,
            steps=3,
            generator=torch.Generator().manual_seed(7),
            radius_bounds=(8.0, 25.0),
        )
        assert decoded.mask.sum(1).tolist() == [expected_k, expected_k]
        assert np.isfinite(decoded.xyz_mm[decoded.mask]).all()
        assert ((decoded.radii_mm[decoded.mask] >= 8) & (decoded.radii_mm[decoded.mask] <= 25)).all()
        assert np.isnan(decoded.xyz_mm[~decoded.mask]).all()
        if mode == "null_slots":
            assert decoded.active_probability.shape == (2, 3)
            assert decoded.calibration_kind == "approximate_poisson_binomial"
        else:
            assert decoded.active_probability is None
            assert decoded.calibration_kind == "native_count_head"
        assert decoded.cardinality_probability.shape == (2, 4)
        np.testing.assert_allclose(decoded.cardinality_probability.sum(1), 1.0)


def test_aggregate_metrics_preserve_infinite_geodesics_and_correct_k_subset():
    from megflow.v15_evaluate import aggregate_record_metrics

    records = [
        dict(
            true_k=1,
            predicted_k=1,
            ospa_mm=2.0,
            center_mae_mm=1.0,
            geodesic_mae_mm=3.0,
            radius_mae_mm=2.0,
            radius_bias_mm=-2.0,
            disconnected_count=0,
            duplicate_count=0,
            forced_nonempty=False,
            geodesic_connected_sum_mm=10.0,
            geodesic_connected_count=1,
            geodesic_penalized_sum_mm=10.0,
            geodesic_match_count=1,
            geodesic_disconnected_count=0,
        ),
        dict(
            true_k=2,
            predicted_k=1,
            ospa_mm=20.0,
            center_mae_mm=4.0,
            geodesic_mae_mm=float("inf"),
            radius_mae_mm=5.0,
            radius_bias_mm=5.0,
            disconnected_count=1,
            duplicate_count=1,
            forced_nonempty=True,
            geodesic_connected_sum_mm=20.0,
            geodesic_connected_count=2,
            geodesic_penalized_sum_mm=50.0,
            geodesic_match_count=3,
            geodesic_disconnected_count=1,
        ),
    ]
    summary = aggregate_record_metrics(records, k_max=3)
    assert summary["count"]["exact_accuracy"] == 0.5
    assert summary["ospa_mm"] == 11.0
    assert np.isinf(summary["geodesic_mae_mm"])
    assert summary["correct_k_center_mae_mm"] == 1.0
    assert summary["forced_nonempty_rate"] == 0.5
    assert summary["per_k"]["1"]["center_mae_mm"] == 1.0
    assert summary["geodesic_connected_mae_mm"] == pytest.approx(10.0)
    assert summary["geodesic_penalized_mae_mm"] == pytest.approx(15.0)
    assert summary["geodesic_disconnected_rate"] == pytest.approx(0.25)


def test_evaluate_model_is_record_seeded_and_returns_aggregate(geometry):
    import torch
    from megflow.v15_evaluate import evaluate_model

    class CountOneModel:
        k_max = 3
        d_in = 4

        def encode(self, y):
            logits = torch.tensor([[5.0, 0.0, 0.0]], device=y.device).repeat(len(y), 1)
            return torch.zeros(len(y), 1, 2, device=y.device), logits

        def velocity(self, t, state, cond):
            return torch.zeros_like(state)

    data = dict(
        Y=np.zeros((2, 4, 10), dtype="f4"),
        centers=np.array([[0, -1, -1], [2, -1, -1]]),
        radii=np.array([[10.0, np.nan, np.nan], [12.0, np.nan, np.nan]]),
        mask=np.array([[True, False, False], [True, False, False]]),
        K=np.array([1, 1]),
        manifest=dict(split="dev"),
    )
    checkpoint = dict(state_mean=torch.zeros(4), state_std=torch.ones(4))
    first = evaluate_model(
        CountOneModel(),
        checkpoint,
        data,
        geometry,
        mode="count_head",
        steps=2,
        device="cpu",
        evaluation_seed=11,
    )
    second = evaluate_model(
        CountOneModel(),
        checkpoint,
        data,
        geometry,
        mode="count_head",
        steps=2,
        device="cpu",
        evaluation_seed=11,
    )
    assert first["summary"] == second["summary"]
    assert first["records"] == second["records"]
    assert first["runtime_seconds"] >= 0
    assert first["summary"]["count"]["exact_accuracy"] == 1.0
    assert len(first["records"]) == 2
    assert first["records"][0]["sample_seed"] != first["records"][1]["sample_seed"]


def test_multidraw_evaluation_reports_posterior_cardinality_stability(geometry):
    import torch
    from megflow.v15_evaluate import evaluate_model

    class CountOneModel:
        k_max = 2
        d_in = 4

        def encode(self, y):
            return torch.zeros(len(y), 1, 2), torch.tensor([[5.0, 0.0]])

        def velocity(self, t, state, cond):
            return torch.zeros_like(state)

    data = dict(
        Y=np.zeros((1, 4, 10), dtype="f4"),
        centers=np.array([[0, -1]]),
        radii=np.array([[10.0, np.nan]]),
        mask=np.array([[True, False]]),
        K=np.array([1]),
        manifest=dict(split="dev"),
    )
    result = evaluate_model(
        CountOneModel(),
        dict(state_mean=torch.zeros(4), state_std=torch.ones(4)),
        data,
        geometry,
        mode="count_head",
        steps=1,
        device="cpu",
        evaluation_seed=3,
        draw_count=3,
    )
    assert len(result["records"]) == 3
    assert [record["draw"] for record in result["records"]] == [0, 1, 2]
    assert result["posterior_stability"]["draw_count"] == 3
    assert result["posterior_stability"]["records_with_variable_k_rate"] == 0.0


def test_cardinality_calibration_uses_saved_distributions():
    from megflow.v15_evaluate import cardinality_calibration

    records = [
        dict(true_k=1, cardinality_probability=[0.0, 0.8, 0.2]),
        dict(true_k=2, cardinality_probability=[0.0, 0.25, 0.75]),
    ]
    metrics = cardinality_calibration(records, k_max=2)
    assert metrics["nll"] == pytest.approx(-np.mean(np.log([0.8, 0.75])))
    assert metrics["brier"] == pytest.approx(np.mean([0.08, 0.125]))


def test_heldout_requires_a_matching_frozen_dev_protocol(tmp_path):
    from megflow.v15_evaluate import load_frozen_protocol

    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(dict(split="dev", frozen=True, sampler_steps=100)))
    assert load_frozen_protocol(path)["sampler_steps"] == 100

    path.write_text(json.dumps(dict(split="heldout", frozen=True, sampler_steps=100)))
    with pytest.raises(ValueError, match="dev"):
        load_frozen_protocol(path)


def test_evaluation_artifacts_are_finite_json_csv_and_npz(tmp_path):
    from megflow.v15_evaluate import write_evaluation_artifacts

    evaluation = dict(
        summary=dict(
            geodesic_mae_mm=float("inf"),
            disconnected_assignments=1,
            count=dict(exact_accuracy=1.0),
        ),
        records=[
            dict(
                record_index=0,
                true_k=1,
                predicted_k=1,
                geodesic_mae_mm=float("inf"),
                disconnected_count=1,
                predicted_slot_indices=[1],
                predicted_xyz_mm=[[1.0, 2.0, 3.0]],
                predicted_radii_mm=[12.0],
                active_probability=[0.8, 0.1],
                cardinality_probability=[0.0, 0.7, 0.3],
            )
        ],
    )
    metadata = dict(
        split="dev",
        mode="null_slots",
        sampler_steps=100,
        evaluation_seed=17,
        checkpoint_sha256="checkpoint",
        config_sha256="config",
        k_max=2,
    )
    write_evaluation_artifacts(tmp_path / "evaluation", evaluation, metadata)

    summary = json.loads((tmp_path / "evaluation/summary.json").read_text())
    assert summary["metrics"]["geodesic_mae_mm"] is None
    assert summary["metrics"]["disconnected_assignments"] == 1
    protocol = json.loads((tmp_path / "evaluation/protocol.json").read_text())
    assert protocol["frozen"] is True and protocol["split"] == "dev"
    assert (tmp_path / "evaluation/records.csv").is_file()
    with np.load(tmp_path / "evaluation/predictions.npz", allow_pickle=False) as arrays:
        assert arrays["xyz_mm"].shape == (1, 2, 3)
        assert arrays["mask"].tolist() == [[False, True]]
        np.testing.assert_allclose(arrays["active_probability"], [[0.8, 0.1]])
        assert arrays["cardinality_probability"].shape == (1, 3)


def test_heldout_protocol_must_match_model_and_sampling_settings():
    from megflow.v15_evaluate import validate_protocol_matches

    protocol = dict(
        mode="null_slots",
        sampler_steps=100,
        checkpoint_sha256="a",
        config_sha256="b",
        evaluation_seed=17,
        ospa_cutoff_mm=30.0,
        dataset_manifest_sha256="dataset",
        dataset_audit_sha256="audit",
        heldout_manifest_sha256="heldout",
        anatomy_sha256="anatomy",
        evaluator_code_sha256="code",
        projection="nearest_surface_vertex_euclidean",
        matching="hungarian_euclidean_centers_only",
        radius_bounds_mm=[8.0, 25.0],
        draw_count=1,
    )
    validate_protocol_matches(protocol, dict(protocol))
    changed = dict(protocol, sampler_steps=50)
    with pytest.raises(ValueError, match="sampler_steps"):
        validate_protocol_matches(protocol, changed)
    changed = dict(protocol, dataset_audit_sha256="changed")
    with pytest.raises(ValueError, match="dataset_audit_sha256"):
        validate_protocol_matches(protocol, changed)


def test_evaluation_inputs_require_configured_anatomy_and_dataset_config(tmp_path):
    from megflow.dataset import sha256
    from megflow.v15_evaluate import validate_evaluation_inputs

    anatomy = tmp_path / "anatomy.npz"
    anatomy.write_bytes(b"anatomy")
    config = dict(dataset=dict(anatomy_sha256=sha256(anatomy), marker="same"))
    manifest = dict(anatomy_sha256=sha256(anatomy), config=config["dataset"])
    validate_evaluation_inputs(config, manifest, anatomy)

    changed = dict(dataset=dict(anatomy_sha256="wrong", marker="same"))
    with pytest.raises(ValueError, match="anatomy"):
        validate_evaluation_inputs(changed, manifest, anatomy)


def test_evaluator_cli_has_source_hash_for_protocol_provenance():
    from megflow import v15_evaluate

    assert callable(v15_evaluate.source_hash)
