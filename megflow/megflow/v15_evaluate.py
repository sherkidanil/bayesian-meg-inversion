"""Shared source-set decoding and metrics for V15 cardinality experiments."""

import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch

from .dataset import source_hash
from .metrics import ospa


def nearest_vertex_indices(points_mm, vertices_mm):
    points = np.asarray(points_mm, dtype=float)
    vertices = np.asarray(vertices_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("source coordinates must have shape (K, 3)")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("surface vertices must have shape (N, 3)")
    return cKDTree(vertices).query(points)[1].astype(np.int64)


def geodesic_candidate_errors(
    *, true_centers, predicted_centers, geometry, disconnected_penalty_mm=30.0
):
    """Match projected candidates on the cortical graph.

    The assignment minimizes the number of disconnected pairs before finite
    geodesic distance. Reporting keeps connected and cutoff-penalized sums
    separate so cross-component failures cannot disappear from MAE_GEO.
    """
    true_centers = np.asarray(true_centers, dtype=np.int64)
    predicted_centers = np.asarray(predicted_centers, dtype=np.int64)
    if disconnected_penalty_mm <= 0:
        raise ValueError("disconnected geodesic penalty must be positive")
    match_count = min(len(true_centers), len(predicted_centers))
    if not match_count:
        return dict(
            true_index=np.empty(0, dtype=np.int64),
            predicted_index=np.empty(0, dtype=np.int64),
            geodesic_mm=np.empty(0),
            connected_count=0,
            disconnected_count=0,
            match_count=0,
            connected_sum_mm=0.0,
            penalized_sum_mm=0.0,
        )
    distances = np.asarray(
        geometry.geodesic(true_centers[:, None], predicted_centers[None, :]),
        dtype=float,
    )
    finite = distances[np.isfinite(distances)]
    dominant_cost = (
        (float(finite.max()) + 1.0) * (match_count + 1)
        if finite.size
        else 1.0
    )
    true_index, predicted_index = linear_sum_assignment(
        np.where(np.isfinite(distances), distances, dominant_cost)
    )
    assigned = distances[true_index, predicted_index]
    connected = np.isfinite(assigned)
    return dict(
        true_index=true_index,
        predicted_index=predicted_index,
        geodesic_mm=assigned,
        connected_count=int(connected.sum()),
        disconnected_count=int((~connected).sum()),
        match_count=int(len(assigned)),
        connected_sum_mm=float(assigned[connected].sum()),
        penalized_sum_mm=float(
            np.where(
                connected,
                np.minimum(assigned, disconnected_penalty_mm),
                disconnected_penalty_mm,
            ).sum()
        ),
    )


def aggregate_geodesic_matches(rows):
    connected_count = sum(row["geodesic_connected_count"] for row in rows)
    match_count = sum(row["geodesic_match_count"] for row in rows)
    disconnected_count = sum(row["geodesic_disconnected_count"] for row in rows)
    return dict(
        geodesic_connected_mae_mm=(
            sum(row["geodesic_connected_sum_mm"] for row in rows)
            / connected_count
            if connected_count
            else None
        ),
        geodesic_penalized_mae_mm=(
            sum(row["geodesic_penalized_sum_mm"] for row in rows) / match_count
            if match_count
            else None
        ),
        geodesic_disconnected_rate=(
            disconnected_count / match_count if match_count else 0.0
        ),
        geodesic_connected_count=int(connected_count),
        geodesic_disconnected_count=int(disconnected_count),
        geodesic_match_count=int(match_count),
    )


def recompute_geodesic_summary(
    records, data, geometry, *, disconnected_penalty_mm=30.0, draw=0
):
    """Recompute geodesic metrics from immutable projected candidate records."""
    primary = [record for record in records if record.get("draw", 0) == draw]
    indices = [int(record["record_index"]) for record in primary]
    expected = list(range(len(data["centers"])))
    if sorted(indices) != expected or len(set(indices)) != len(indices):
        raise ValueError("records must contain exactly one primary draw per dataset row")
    rows = []
    for record in primary:
        index = int(record["record_index"])
        true_mask = np.asarray(data["mask"][index], dtype=bool)
        true_centers = np.asarray(data["centers"][index][true_mask], dtype=np.int64)
        if int(record["true_k"]) != len(true_centers):
            raise ValueError("saved prediction truth count differs from dataset")
        matched = geodesic_candidate_errors(
            true_centers=true_centers,
            predicted_centers=np.asarray(record["projected_centers"], dtype=np.int64),
            geometry=geometry,
            disconnected_penalty_mm=disconnected_penalty_mm,
        )
        rows.append(
            dict(
                record_index=index,
                true_k=len(true_centers),
                geodesic_connected_sum_mm=matched["connected_sum_mm"],
                geodesic_connected_count=matched["connected_count"],
                geodesic_penalized_sum_mm=matched["penalized_sum_mm"],
                geodesic_match_count=matched["match_count"],
                geodesic_disconnected_count=matched["disconnected_count"],
            )
        )
    per_k = {}
    for k in sorted({row["true_k"] for row in rows}):
        selected = [row for row in rows if row["true_k"] == k]
        per_k[str(k)] = dict(record_count=len(selected), **aggregate_geodesic_matches(selected))
    return dict(
        split=data["manifest"]["split"],
        draw=draw,
        record_count=len(rows),
        disconnected_penalty_mm=float(disconnected_penalty_mm),
        **aggregate_geodesic_matches(rows),
        per_k=per_k,
    )


def matched_source_errors(
    *, true_centers, predicted_centers, true_radii, predicted_radii, geometry
):
    """Match centers by Euclidean distance for center-error diagnostics."""
    true_centers = np.asarray(true_centers, dtype=np.int64)
    predicted_centers = np.asarray(predicted_centers, dtype=np.int64)
    true_radii = np.asarray(true_radii, dtype=float)
    predicted_radii = np.asarray(predicted_radii, dtype=float)
    if len(true_centers) != len(true_radii) or len(predicted_centers) != len(
        predicted_radii
    ):
        raise ValueError("every source center must have one radius")
    if not len(true_centers) or not len(predicted_centers):
        return dict(
            true_index=np.empty(0, dtype=np.int64),
            predicted_index=np.empty(0, dtype=np.int64),
            euclidean_mm=np.empty(0),
            geodesic_mm=np.empty(0),
            radius_error_mm=np.empty(0),
            disconnected_count=0,
        )
    true_xyz = geometry.vertices[true_centers]
    predicted_xyz = geometry.vertices[predicted_centers]
    euclidean_cost = np.linalg.norm(
        true_xyz[:, None, :] - predicted_xyz[None, :, :], axis=-1
    )
    true_index, predicted_index = linear_sum_assignment(euclidean_cost)
    euclidean = euclidean_cost[true_index, predicted_index]
    geodesic = np.asarray(
        geometry.geodesic(
            true_centers[true_index], predicted_centers[predicted_index]
        ),
        dtype=float,
    )
    radius_error = (
        predicted_radii[predicted_index] - true_radii[true_index]
    )
    return dict(
        true_index=true_index,
        predicted_index=predicted_index,
        euclidean_mm=euclidean,
        geodesic_mm=geodesic,
        radius_error_mm=radius_error,
        disconnected_count=int((~np.isfinite(geodesic)).sum()),
    )


def count_metrics(true_count, predicted_count, k_max):
    truth = np.asarray(true_count, dtype=np.int64)
    predicted = np.asarray(predicted_count, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("true and predicted counts must be equal one-dimensional arrays")
    if (
        (truth < 1).any()
        or (truth > k_max).any()
        or (predicted < 1).any()
        or (predicted > k_max).any()
    ):
        raise ValueError("source counts must lie in 1..K_max")
    confusion = np.zeros((k_max, k_max), dtype=np.int64)
    np.add.at(confusion, (truth - 1, predicted - 1), 1)
    per_k = {
        str(k): float((predicted[truth == k] == k).mean())
        for k in range(1, k_max + 1)
        if np.any(truth == k)
    }
    return dict(
        exact_accuracy=float((truth == predicted).mean()),
        mae=float(np.abs(truth - predicted).mean()),
        per_k_accuracy=per_k,
        confusion=confusion.tolist(),
    )


def poisson_binomial(active_probability, *, condition_nonempty=False):
    probabilities = np.asarray(active_probability, dtype=float)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("active probabilities must be a finite vector")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("active probabilities must lie in [0,1]")
    distribution = np.zeros(len(probabilities) + 1, dtype=float)
    distribution[0] = 1.0
    for index, probability in enumerate(probabilities):
        previous = distribution.copy()
        distribution[0] = previous[0] * (1 - probability)
        distribution[1 : index + 2] = (
            previous[1 : index + 2] * (1 - probability)
            + previous[: index + 1] * probability
        )
    if condition_nonempty:
        distribution[0] = 0.0
        normalizer = distribution.sum()
        if normalizer == 0:
            distribution[1:] = 1 / len(probabilities)
        else:
            distribution /= normalizer
    return distribution


def evaluate_record(
    *,
    true_centers,
    true_radii,
    predicted_xyz,
    predicted_radii,
    geometry,
    ospa_cutoff_mm=30.0,
):
    true_centers = np.asarray(true_centers, dtype=np.int64)
    true_radii = np.asarray(true_radii, dtype=float)
    predicted_xyz = np.asarray(predicted_xyz, dtype=float)
    predicted_radii = np.asarray(predicted_radii, dtype=float)
    predicted_centers = nearest_vertex_indices(predicted_xyz, geometry.vertices)
    matched = matched_source_errors(
        true_centers=true_centers,
        predicted_centers=predicted_centers,
        true_radii=true_radii,
        predicted_radii=predicted_radii,
        geometry=geometry,
    )
    candidate_geodesic = geodesic_candidate_errors(
        true_centers=true_centers,
        predicted_centers=predicted_centers,
        geometry=geometry,
        disconnected_penalty_mm=ospa_cutoff_mm,
    )
    radius_error = (
        predicted_radii[candidate_geodesic["predicted_index"]]
        - true_radii[candidate_geodesic["true_index"]]
    )
    euclidean = matched["euclidean_mm"]
    euclidean_geodesic = matched["geodesic_mm"]
    return dict(
        true_k=int(len(true_centers)),
        predicted_k=int(len(predicted_centers)),
        projected_centers=predicted_centers.tolist(),
        duplicate_count=int(len(predicted_centers) - len(np.unique(predicted_centers))),
        ospa_mm=ospa(
            true_centers,
            predicted_centers,
            geometry,
            c_cutoff=ospa_cutoff_mm,
        ),
        center_mae_mm=float(np.mean(euclidean)) if len(euclidean) else None,
        geodesic_mae_mm=(
            float(np.mean(euclidean_geodesic)) if len(euclidean_geodesic) else None
        ),
        radius_mae_mm=float(np.mean(np.abs(radius_error))) if len(radius_error) else None,
        radius_bias_mm=float(np.mean(radius_error)) if len(radius_error) else None,
        radius_match_true_index=candidate_geodesic["true_index"].tolist(),
        radius_match_predicted_index=candidate_geodesic["predicted_index"].tolist(),
        radius_error_mm=radius_error.tolist(),
        disconnected_count=matched["disconnected_count"],
        geodesic_connected_sum_mm=candidate_geodesic["connected_sum_mm"],
        geodesic_connected_count=candidate_geodesic["connected_count"],
        geodesic_penalized_sum_mm=candidate_geodesic["penalized_sum_mm"],
        geodesic_match_count=candidate_geodesic["match_count"],
        geodesic_disconnected_count=candidate_geodesic["disconnected_count"],
    )


@dataclass(frozen=True)
class DecodedSourceBatch:
    xyz_mm: np.ndarray
    radii_mm: np.ndarray
    mask: np.ndarray
    active_probability: np.ndarray | None
    cardinality_probability: np.ndarray
    calibration_kind: str
    forced_nonempty: np.ndarray


def sample_sources(
    model,
    checkpoint,
    y,
    *,
    mode,
    steps,
    generator,
    radius_bounds=(8.0, 25.0),
):
    """Sample and decode sources without accepting truth cardinality."""
    if mode == "count_head":
        from .v15_model import sample_unknown_k

        state, mask, count_logits = sample_unknown_k(
            model, y, steps=steps, generator=generator
        )
        active_probability = None
        count_probability = count_logits.float().softmax(-1).detach().cpu().numpy()
        cardinality_probability = np.zeros(
            (len(y), model.k_max + 1), dtype="f4"
        )
        cardinality_probability[:, 1:] = count_probability
        calibration_kind = "native_count_head"
        forced_nonempty = torch.zeros(len(y), dtype=torch.bool, device=y.device)
    elif mode == "null_slots":
        from .null_slot_model import sample_null_slots

        sample = sample_null_slots(
            model, y, steps=steps, generator=generator
        )
        state = sample.state
        mask = sample.active_mask
        active_probability = sample.active_probability.detach().cpu().numpy()
        cardinality_probability = np.stack(
            [
                poisson_binomial(probability, condition_nonempty=True)
                for probability in active_probability
            ]
        ).astype("f4")
        calibration_kind = "approximate_poisson_binomial"
        forced_nonempty = sample.forced_nonempty
    else:
        raise ValueError("mode must be count_head or null_slots")
    from .v15 import denormalize_source_state

    mean = checkpoint["state_mean"].detach().cpu().numpy()
    std = checkpoint["state_std"].detach().cpu().numpy()
    state_numpy = state.detach().float().cpu().numpy()
    mask_numpy = mask.detach().cpu().numpy()
    xyz, radii = denormalize_source_state(
        state_numpy,
        mask_numpy,
        mean,
        std,
        radius_bounds=radius_bounds,
    )
    return DecodedSourceBatch(
        xyz_mm=xyz,
        radii_mm=radii,
        mask=mask_numpy,
        active_probability=active_probability,
        cardinality_probability=cardinality_probability,
        calibration_kind=calibration_kind,
        forced_nonempty=forced_nonempty.detach().cpu().numpy(),
    )


def cardinality_calibration(records, *, k_max):
    if not records:
        raise ValueError("at least one record is required for calibration")
    probabilities = np.asarray(
        [record["cardinality_probability"] for record in records], dtype=float
    )
    truth = np.asarray([record["true_k"] for record in records], dtype=np.int64)
    if probabilities.shape != (len(records), k_max + 1):
        raise ValueError("cardinality distributions have incompatible shape")
    if not np.allclose(probabilities.sum(1), 1.0, atol=1e-6):
        raise ValueError("cardinality distributions are not normalized")
    one_hot = np.eye(k_max + 1)[truth]
    return dict(
        nll=float(-np.log(probabilities[np.arange(len(truth)), truth].clip(1e-12)).mean()),
        brier=float(np.square(probabilities - one_hot).sum(1).mean()),
        mean_confidence=float(probabilities.max(1).mean()),
    )


def aggregate_record_metrics(records, *, k_max):
    if not records:
        raise ValueError("at least one evaluation record is required")
    truth = np.asarray([record["true_k"] for record in records])
    predicted = np.asarray([record["predicted_k"] for record in records])

    def mean(key, selected=None):
        rows = records if selected is None else [
            record for record, keep in zip(records, selected) if keep
        ]
        values = [record[key] for record in rows if record.get(key) is not None]
        return float(np.mean(values)) if values else None

    correct = truth == predicted
    total_predicted = int(predicted.sum())

    has_candidate_geodesics = all(
        "geodesic_match_count" in record for record in records
    )
    per_k = {}
    for k in sorted(set(truth.tolist())):
        selected = truth == k
        per_k[str(k)] = dict(
            n=int(selected.sum()),
            count_accuracy=float((predicted[selected] == k).mean()),
            ospa_mm=mean("ospa_mm", selected),
            center_mae_mm=mean("center_mae_mm", selected),
            geodesic_mae_mm=mean("geodesic_mae_mm", selected),
            radius_mae_mm=mean("radius_mae_mm", selected),
            radius_bias_mm=mean("radius_bias_mm", selected),
        )
        if has_candidate_geodesics:
            per_k[str(k)].update(
                aggregate_geodesic_matches(
                    [
                        record
                        for record, keep in zip(records, selected)
                        if keep
                    ]
                )
            )
    summary = dict(
        count=count_metrics(truth, predicted, k_max),
        ospa_mm=mean("ospa_mm"),
        center_mae_mm=mean("center_mae_mm"),
        geodesic_mae_mm=mean("geodesic_mae_mm"),
        radius_mae_mm=mean("radius_mae_mm"),
        radius_bias_mm=mean("radius_bias_mm"),
        correct_k_center_mae_mm=mean("center_mae_mm", correct),
        correct_k_geodesic_mae_mm=mean("geodesic_mae_mm", correct),
        correct_k_radius_mae_mm=mean("radius_mae_mm", correct),
        disconnected_assignments=int(
            sum(record["disconnected_count"] for record in records)
        ),
        duplicate_source_rate=(
            sum(record["duplicate_count"] for record in records) / total_predicted
            if total_predicted
            else 0.0
        ),
        forced_nonempty_rate=float(
            np.mean([record.get("forced_nonempty", False) for record in records])
        ),
        per_k=per_k,
    )
    if has_candidate_geodesics:
        summary.update(aggregate_geodesic_matches(records))
    if all(record.get("cardinality_probability") is not None for record in records):
        summary["cardinality_calibration"] = cardinality_calibration(
            records, k_max=k_max
        )
        summary["calibration_kind"] = records[0].get("calibration_kind")
    return summary


def _sample_seed(evaluation_seed, split, record_index, draw):
    payload = f"{evaluation_seed}:{split}:{record_index}:{draw}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def evaluate_model(
    model,
    checkpoint,
    data,
    geometry,
    *,
    mode,
    steps,
    device,
    evaluation_seed,
    draw=0,
    draw_count=1,
    radius_bounds=(8.0, 25.0),
    ospa_cutoff_mm=30.0,
):
    """Evaluate every record with a seed independent of batching/order."""
    if draw_count < 1:
        raise ValueError("draw_count must be positive")
    started = time.monotonic()
    device = torch.device(device)
    if hasattr(model, "eval"):
        model.eval()
    records = []
    split = data["manifest"]["split"]
    for index in range(len(data["Y"])):
        for draw_index in range(draw, draw + draw_count):
            seed = _sample_seed(evaluation_seed, split, index, draw_index)
            generator = torch.Generator(device=device.type).manual_seed(seed)
            y = torch.from_numpy(np.asarray(data["Y"][index : index + 1])).to(device)
            decoded = sample_sources(
                model,
                checkpoint,
                y,
                mode=mode,
                steps=steps,
                generator=generator,
                radius_bounds=radius_bounds,
            )
            active = decoded.mask[0]
            truth = data["mask"][index]
            record = evaluate_record(
                true_centers=data["centers"][index, truth],
                true_radii=data["radii"][index, truth],
                predicted_xyz=decoded.xyz_mm[0, active],
                predicted_radii=decoded.radii_mm[0, active],
                geometry=geometry,
                ospa_cutoff_mm=ospa_cutoff_mm,
            )
            record.update(
                record_index=index,
                draw=draw_index,
                sample_seed=seed,
                predicted_slot_indices=np.where(active)[0].tolist(),
                predicted_xyz_mm=decoded.xyz_mm[0, active].tolist(),
                predicted_radii_mm=decoded.radii_mm[0, active].tolist(),
                forced_nonempty=bool(decoded.forced_nonempty[0]),
                active_probability=(
                    decoded.active_probability[0].tolist()
                    if decoded.active_probability is not None
                    else None
                ),
                cardinality_probability=decoded.cardinality_probability[0].tolist(),
                calibration_kind=decoded.calibration_kind,
            )
            records.append(record)
    primary = [record for record in records if record["draw"] == draw]
    variable_count = []
    for index in range(len(data["Y"])):
        counts = {
            record["predicted_k"]
            for record in records
            if record["record_index"] == index
        }
        variable_count.append(len(counts) > 1)
    return dict(
        summary=aggregate_record_metrics(primary, k_max=model.k_max),
        records=records,
        posterior_stability=dict(
            draw_count=draw_count,
            records_with_variable_k_rate=float(np.mean(variable_count)),
            mean_unique_predicted_k=float(
                np.mean(
                    [
                        len(
                            {
                                record["predicted_k"]
                                for record in records
                                if record["record_index"] == index
                            }
                        )
                        for index in range(len(data["Y"]))
                    ]
                )
            ),
        ),
        runtime_seconds=time.monotonic() - started,
        runtime_per_sample_seconds=(time.monotonic() - started) / len(records),
    )


def load_frozen_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("split") != "dev" or protocol.get("frozen") is not True:
        raise ValueError("heldout evaluation requires a frozen dev protocol")
    if not isinstance(protocol.get("sampler_steps"), int) or protocol["sampler_steps"] < 1:
        raise ValueError("dev protocol has invalid sampler steps")
    return protocol


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_evaluation_artifacts(output, evaluation, metadata):
    """Write immutable reviewable evaluation artifacts without pickle arrays."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        metadata=_json_finite(metadata),
        metrics=_json_finite(evaluation["summary"]),
        posterior_stability=_json_finite(evaluation.get("posterior_stability")),
        runtime_seconds=evaluation.get("runtime_seconds"),
        runtime_per_sample_seconds=evaluation.get("runtime_per_sample_seconds"),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    records = evaluation["records"]
    (output / "records.json").write_text(
        json.dumps(_json_finite(records), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    scalar_fields = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if not isinstance(value, (list, dict, tuple))
        }
    )
    with (output / "records.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: _json_finite(record.get(key))
                    for key in scalar_fields
                }
            )
    k_max = int(metadata["k_max"])
    xyz = np.full((len(records), k_max, 3), np.nan, dtype="f4")
    radii = np.full((len(records), k_max), np.nan, dtype="f4")
    mask = np.zeros((len(records), k_max), dtype=bool)
    active_probability = np.full((len(records), k_max), np.nan, dtype="f4")
    cardinality_probability = np.full(
        (len(records), k_max + 1), np.nan, dtype="f4"
    )
    for row, record in enumerate(records):
        count = len(record["predicted_radii_mm"])
        indices = np.asarray(
            record.get("predicted_slot_indices", list(range(count))), dtype=np.int64
        )
        if (
            len(indices) != count
            or len(np.unique(indices)) != count
            or (indices < 0).any()
            or (indices >= k_max).any()
        ):
            raise ValueError("invalid predicted slot indices")
        xyz[row, indices] = record["predicted_xyz_mm"]
        radii[row, indices] = record["predicted_radii_mm"]
        mask[row, indices] = True
        if record.get("active_probability") is not None:
            active_probability[row] = record["active_probability"]
        if record.get("cardinality_probability") is not None:
            cardinality_probability[row] = record["cardinality_probability"]
    np.savez_compressed(
        output / "predictions.npz",
        xyz_mm=xyz,
        radii_mm=radii,
        mask=mask,
        active_probability=active_probability,
        cardinality_probability=cardinality_probability,
    )
    if metadata["split"] == "dev":
        protocol = dict(_json_finite(metadata), frozen=True)
        (output / "protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def validate_protocol_matches(protocol, current):
    keys = (
        "mode",
        "sampler_steps",
        "checkpoint_sha256",
        "config_sha256",
        "evaluation_seed",
        "ospa_cutoff_mm",
        "dataset_audit_sha256",
        "heldout_manifest_sha256",
        "anatomy_sha256",
        "evaluator_code_sha256",
        "projection",
        "matching",
        "radius_bounds_mm",
        "draw_count",
    )
    for key in keys:
        if protocol.get(key) != current.get(key):
            raise ValueError(f"heldout setting differs from dev protocol: {key}")


def validate_evaluation_inputs(config, manifest, anatomy_path):
    from .dataset import sha256, source_hash

    expected_anatomy = config["dataset"].get("anatomy_sha256")
    actual_anatomy = sha256(anatomy_path)
    if expected_anatomy != actual_anatomy or manifest.get("anatomy_sha256") != actual_anatomy:
        raise ValueError("evaluation anatomy does not match the configured dataset")
    if manifest.get("config") != config["dataset"]:
        raise ValueError("evaluation config and dataset differ")


def main(argv=None):
    from omegaconf import OmegaConf

    from .dataset import sha256
    from .geometry import CorticalGeometry
    from .train import build_model
    from .v15_dataset import load_evaluation_arrays

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--mode", choices=["count_head", "null_slots"], required=True)
    parser.add_argument("--architecture", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--draw-count", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=20260917)
    parser.add_argument("--ospa-cutoff-mm", type=float, default=30.0)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    configured_mode = config["model"].get("cardinality_mode", "count_head")
    if configured_mode != args.mode:
        raise ValueError("requested mode differs from model configuration")
    steps = args.steps or config["training"].get("sampler_steps", 100)
    evaluation_seed = args.evaluation_seed
    ospa_cutoff_mm = args.ospa_cutoff_mm
    draw_count = args.draw_count
    if draw_count < 1:
        raise ValueError("draw count must be positive")
    protocol = None
    if args.split == "heldout":
        if args.protocol is None:
            raise ValueError("heldout evaluation requires --protocol from dev")
        protocol = load_frozen_protocol(args.protocol)
        steps = protocol["sampler_steps"]
        evaluation_seed = protocol["evaluation_seed"]
        ospa_cutoff_mm = protocol["ospa_cutoff_mm"]
        draw_count = protocol["draw_count"]

    data = load_evaluation_arrays(args.dataset_root / args.split)
    validate_evaluation_inputs(config, data["manifest"], args.anatomy)
    dataset_audit = json.loads((args.dataset_root / "audit.json").read_text())
    geometry, _, anatomy_provenance = CorticalGeometry.load_bundle(args.anatomy)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = build_model(config, args.architecture).to(args.device)
    model.load_state_dict(checkpoint["ema"])
    metadata = dict(
        split=args.split,
        mode=args.mode,
        architecture=args.architecture,
        sampler_steps=steps,
        evaluation_seed=evaluation_seed,
        ospa_cutoff_mm=ospa_cutoff_mm,
        checkpoint_sha256=sha256(args.checkpoint),
        config_sha256=sha256(args.config),
        dataset_manifest_sha256=sha256(args.dataset_root / args.split / "manifest.json"),
        dataset_audit_sha256=sha256(args.dataset_root / "audit.json"),
        heldout_manifest_sha256=dataset_audit["manifest_sha256"]["heldout"],
        anatomy_sha256=sha256(args.anatomy),
        evaluator_code_sha256=source_hash(),
        anatomy_provenance=anatomy_provenance,
        k_max=config["dataset"]["simulation"]["k_max"],
        radius_bounds_mm=[
            config["dataset"]["simulation"]["radius_min"],
            config["dataset"]["simulation"]["radius_max"],
        ],
        projection="nearest_surface_vertex_euclidean",
        matching="hungarian_geodesic_radii_euclidean_centers",
        draw_count=draw_count,
    )
    if protocol is not None:
        validate_protocol_matches(protocol, metadata)
        metadata["dev_protocol_sha256"] = sha256(args.protocol)
    evaluation = evaluate_model(
        model,
        checkpoint,
        data,
        geometry,
        mode=args.mode,
        steps=steps,
        device=args.device,
        evaluation_seed=evaluation_seed,
        draw_count=draw_count,
        radius_bounds=tuple(metadata["radius_bounds_mm"]),
        ospa_cutoff_mm=ospa_cutoff_mm,
    )
    write_evaluation_artifacts(args.output, evaluation, metadata)
    print(json.dumps(_json_finite(evaluation["summary"]), sort_keys=True))


if __name__ == "__main__":
    main()
