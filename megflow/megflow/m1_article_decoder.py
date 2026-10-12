"""Frozen article decoder for the fixed-K=3 M1 comparison."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import torch

from .dataset import sha256, source_hash
from .forward import patch_dictionary, subspace_score


FROZEN_M1_DECODER = {
    "bank_size": 800,
    "euler_steps": 96,
    "initial_noise_scale": 1.5,
    "signal_rank": 6,
    "gibbs_tau": 0.0011995147478874274,
    "refinement_passes": 4,
    "radii_mm": [10.0, 8.0, 10.0],
}


def sample_fixed_k_bank(
    model,
    y,
    mean_mm,
    std_mm,
    *,
    bank_size=FROZEN_M1_DECODER["bank_size"],
    steps=FROZEN_M1_DECODER["euler_steps"],
    initial_noise_scale=FROZEN_M1_DECODER["initial_noise_scale"],
    seed,
):
    """Sample one deterministic S-by-three bank with forward Euler."""
    if not torch.is_tensor(y) or y.ndim != 3 or len(y) != 1:
        raise ValueError("y must contain one pooled recording")
    size, count = int(bank_size), int(steps)
    scale = float(initial_noise_scale)
    if size != bank_size or size <= 0 or count != steps or count <= 0:
        raise ValueError("bank_size and steps must be positive integers")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("initial noise scale must be positive and finite")
    mean = np.asarray(mean_mm, dtype=np.float32)
    std = np.asarray(std_mm, dtype=np.float32)
    if (
        mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("M1 normalization must contain three finite xyz values")
    generator = torch.Generator(device=y.device).manual_seed(int(seed))
    # Draw candidate by candidate so an S-prefix is bitwise independent of the
    # final requested bank size on CPU and CUDA random-number implementations.
    state = scale * torch.stack(
        [
            torch.randn(
                (3, 3), generator=generator, device=y.device, dtype=y.dtype
            )
            for _ in range(size)
        ]
    )
    mask = torch.ones((size, 3), dtype=torch.bool, device=y.device)
    with torch.no_grad():
        condition = model.encoder(y).expand(size, -1, -1)
        for index in range(count):
            t = torch.full(
                (size,), index / count, dtype=state.dtype, device=state.device
            )
            state = state + model.field(t, state, condition, mask) / count
    normalized = state.detach().float().cpu().numpy()
    return normalized * std[None, None, :] + mean[None, None, :]


def gibbs_weights(scores, *, tau=FROZEN_M1_DECODER["gibbs_tau"]):
    """Return stable Gibbs weights for the maximized joint rho score."""
    values = np.asarray(scores, dtype=np.float64)
    temperature = float(tau)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a nonempty finite vector")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("tau must be positive and finite")
    weights = np.exp((values - values.max()) / temperature)
    return weights / weights.sum()


def _normalized_weights(weights, size):
    probabilities = np.asarray(weights, dtype=np.float64)
    if (
        probabilities.shape != (size,)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or probabilities.sum() <= 0
    ):
        raise ValueError("weights must be finite, nonnegative, and match draws")
    return probabilities / probabilities.sum()


def weighted_slot_medoids(draw_vertices, weights, geometry):
    """Apply the article's component-first weighted medoid to every slot."""
    vertices = np.asarray(draw_vertices)
    if vertices.ndim != 2 or not vertices.size:
        raise ValueError("draw_vertices must have shape (draws, sources)")
    if not np.issubdtype(vertices.dtype, np.integer):
        if not np.equal(vertices, np.floor(vertices)).all():
            raise ValueError("draw vertices must be integers")
        vertices = vertices.astype(np.int64)
    if (vertices < 0).any() or (vertices >= len(geometry.vertices)).any():
        raise IndexError("draw vertex is outside geometry")
    probabilities = _normalized_weights(weights, len(vertices))
    result = np.empty(vertices.shape[1], dtype=np.int64)
    for slot in range(vertices.shape[1]):
        draws = vertices[:, slot]
        candidates = np.unique(draws)
        distances = np.asarray(
            geometry.geodesic(candidates[:, None], draws[None, :]), dtype=np.float64
        )
        finite = np.isfinite(distances)
        captured = finite @ probabilities
        numerator = np.where(finite, distances, 0.0) @ probabilities
        within = np.full_like(captured, np.inf)
        np.divide(numerator, captured, out=within, where=captured > 0)
        order = np.lexsort((candidates, within, -captured))
        result[slot] = candidates[int(order[0])]
    return result


def candidate_matrices(dictionaries, proposed_vertices):
    """Gather one slot-specific patch topography for each proposal."""
    patches = np.asarray(dictionaries, dtype=np.float64)
    proposed = np.asarray(proposed_vertices)
    if patches.ndim != 3 or not np.isfinite(patches).all():
        raise ValueError("dictionaries must have shape (sources, sensors, vertices)")
    if proposed.ndim != 2 or proposed.shape[1] != patches.shape[0]:
        raise ValueError("proposals must have shape (draws, sources)")
    if not np.issubdtype(proposed.dtype, np.integer):
        if not np.equal(proposed, np.floor(proposed)).all():
            raise ValueError("proposal vertices must be integers")
        proposed = proposed.astype(np.int64)
    if (proposed < 0).any() or (proposed >= patches.shape[2]).any():
        raise IndexError("proposal vertex is outside dictionaries")
    return np.stack(
        [
            patches[slot][:, proposed[:, slot]].T
            for slot in range(patches.shape[0])
        ],
        axis=-1,
    )


def dominant_signal_subspace(y, *, rank):
    """Return the requested dominant left singular vectors, as in the article."""
    data = np.asarray(y, dtype=np.float64)
    selected_rank = int(rank)
    if (
        data.ndim != 2
        or not data.size
        or not np.isfinite(data).all()
        or selected_rank != rank
        or not 1 <= selected_rank <= min(data.shape)
    ):
        raise ValueError("data and signal rank are invalid")
    basis, singular, _ = np.linalg.svd(data, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(data.shape)
    if singular[0] == 0 or singular[selected_rank - 1] <= tolerance * singular[0]:
        raise ValueError("data do not support the requested signal rank")
    return basis[:, :selected_rank]


def score_proposals(proposed_vertices, dictionaries, y, *, signal_rank=6):
    """Score fixed-K proposals by mean squared principal-cosine alignment."""
    matrices = candidate_matrices(dictionaries, proposed_vertices)
    signal = dominant_signal_subspace(y, rank=signal_rank)
    candidate_basis, singular, _ = np.linalg.svd(matrices, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(matrices.shape[1:])
    supported = singular > tolerance * singular[:, :1]
    overlap = np.einsum("mr,dmk->drk", signal, candidate_basis)
    scores = np.clip(
        np.sum(np.square(overlap) * supported[:, None, :], axis=(1, 2))
        / matrices.shape[2],
        0.0,
        1.0,
    )
    return scores, signal


def align_m1_proposals(
    proposed_vertices, dictionaries, y, geometry, *, signal_rank=6
):
    """Assign the 8 mm slot by rho, then align the two 10 mm slots geodesically."""
    proposed = np.asarray(proposed_vertices, dtype=np.int64)
    if proposed.ndim != 2 or proposed.shape[1] != 3:
        raise ValueError("M1 proposals must have shape (draws, 3)")
    # Each permutation places a different raw token into the unique 8 mm slot.
    permutations = np.asarray([[0, 1, 2], [1, 0, 2], [0, 2, 1]], dtype=np.int64)
    alternatives = np.stack([proposed[:, permutation] for permutation in permutations])
    alternative_scores = []
    signal = None
    for alternative in alternatives:
        scores, current_signal = score_proposals(
            alternative, dictionaries, y, signal_rank=signal_rank
        )
        alternative_scores.append(scores)
        if signal is None:
            signal = current_signal
    alternative_scores = np.stack(alternative_scores)
    selected = np.argmax(alternative_scores, axis=0)
    rows = np.arange(len(proposed))
    canonical = alternatives[selected, rows].copy()
    scores = alternative_scores[selected, rows]

    reference = canonical[int(np.argmax(scores))]
    ten_slots = np.asarray([0, 2], dtype=np.int64)
    for row, candidate in enumerate(canonical):
        distances = np.asarray(
            geometry.geodesic(
                reference[ten_slots, None], candidate[ten_slots][None, :]
            ),
            dtype=np.float64,
        )
        finite = distances[np.isfinite(distances)]
        dominant = (
            (float(finite.max()) + 1.0) * 3 if finite.size else 1.0
        )
        reference_index, candidate_index = linear_sum_assignment(
            np.where(np.isfinite(distances), distances, dominant)
        )
        order = candidate_index[np.argsort(reference_index)]
        canonical[row, ten_slots] = candidate[ten_slots[order]]
    changed = np.any(canonical != proposed, axis=1)
    diagnostics = {
        "permutation_fraction": float(np.mean(changed)),
        "radius_slot_permutation_fraction": float(np.mean(selected != 0)),
    }
    assert signal is not None
    return canonical, scores, diagnostics, signal


def decode_m1_bank(
    xyz_mm,
    score_y,
    dictionaries,
    geometry,
    *,
    signal_rank=FROZEN_M1_DECODER["signal_rank"],
    refinement_passes=FROZEN_M1_DECODER["refinement_passes"],
    tau=FROZEN_M1_DECODER["gibbs_tau"],
):
    """Snap, score, aggregate and refine one fixed-K proposal bank."""
    coordinates = np.asarray(xyz_mm, dtype=np.float64)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (3, 3):
        raise ValueError("M1 candidate coordinates must have shape (S, 3, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("candidate coordinates must be finite")
    projected = cKDTree(geometry.vertices).query(coordinates.reshape(-1, 3), k=1)[1]
    projected = np.asarray(projected, dtype=np.int64).reshape(coordinates.shape[:2])
    aligned, scores, alignment, signal = align_m1_proposals(
        projected, dictionaries, score_y, geometry, signal_rank=signal_rank
    )
    weights = gibbs_weights(scores, tau=tau)
    medoid = weighted_slot_medoids(aligned, weights, geometry)
    duplicate = len(set(medoid.tolist())) != len(medoid)
    if duplicate:
        refined = RefinementResult(
            centroids=medoid.copy(),
            score=float(
                subspace_score(
                    signal,
                    candidate_matrices(dictionaries, medoid[None, :])[0],
                )
            ),
            n_moves=0,
            n_evaluations=0,
            n_passes=0,
            converged=False,
        )
    else:
        refined = refine_centroids(
            medoid,
            dictionaries,
            signal,
            geometry,
            max_passes=refinement_passes,
        )
    positive = weights[weights > 0]
    return {
        "projected_vertices": projected,
        "aligned_vertices": aligned,
        "scores": scores,
        "weights": weights,
        "medoid_centers": medoid,
        "refined_centers": refined.centroids,
        "mean_effective_sample_size": float(1.0 / np.square(weights).sum()),
        "weight_entropy": float(-np.sum(positive * np.log(positive))),
        **alignment,
        "duplicate_initial": duplicate,
        "refinement_score": refined.score,
        "refinement_moves": refined.n_moves,
        "refinement_evaluations": refined.n_evaluations,
        "refinement_passes": refined.n_passes,
        "refinement_converged": refined.converged,
    }


def m1_localization_metrics(
    *,
    true_centers,
    predicted_centers,
    geometry,
    disconnected_penalty_mm=30.0,
    graph_diameter_mm=None,
):
    """Return optimal-assignment M1 metrics without dropping disconnections."""
    truth = np.asarray(true_centers, dtype=np.int64)
    prediction = np.asarray(predicted_centers, dtype=np.int64)
    if truth.shape != (3,) or prediction.shape != (3,):
        raise ValueError("M1 metrics require exactly three true and predicted centers")
    cutoff = float(disconnected_penalty_mm)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("disconnected penalty must be positive and finite")
    distances = np.asarray(
        geometry.geodesic(truth[:, None], prediction[None, :]), dtype=np.float64
    )
    finite_values = distances[np.isfinite(distances)]
    dominant = (
        (float(finite_values.max()) + 1.0) * 4 if finite_values.size else 1.0
    )
    truth_index, prediction_index = linear_sum_assignment(
        np.where(np.isfinite(distances), distances, dominant)
    )
    assigned = distances[truth_index, prediction_index]
    finite = np.isfinite(assigned)
    euclidean = np.linalg.norm(
        geometry.vertices[truth[truth_index]]
        - geometry.vertices[prediction[prediction_index]],
        axis=1,
    )
    if graph_diameter_mm is None:
        all_distances = (
            np.asarray(geometry._distances, dtype=np.float64)
            if geometry._distances is not None
            else np.asarray(
                geometry.geodesic(
                    np.arange(len(geometry.vertices))[:, None],
                    np.arange(len(geometry.vertices))[None, :],
                ),
                dtype=np.float64,
            )
        )
        graph_diameter = float(np.max(all_distances[np.isfinite(all_distances)]))
    else:
        graph_diameter = float(graph_diameter_mm)
        if not np.isfinite(graph_diameter) or graph_diameter <= 0:
            raise ValueError("graph diameter must be positive and finite")
    uncapped = float(np.mean(assigned)) if finite.all() else None
    connected = float(np.mean(assigned[finite])) if finite.any() else None
    return {
        "geodesic_mae_mm": uncapped,
        "geodesic_connected_mae_mm": connected,
        "geodesic_diameter_penalized_mae_mm": float(
            np.mean(np.where(finite, assigned, graph_diameter))
        ),
        "geodesic_mae_at_30_mm": float(
            np.mean(np.where(finite, np.minimum(assigned, cutoff), cutoff))
        ),
        "geodesic_disconnected_count": int((~finite).sum()),
        "euclidean_mae_mm": float(np.mean(euclidean)),
        "assigned_geodesic_mm": assigned.tolist(),
        "truth_assignment": truth_index.tolist(),
        "prediction_assignment": prediction_index.tolist(),
    }


def load_m1_evaluation_arrays(directory):
    """Load a complete immutable M1 dev or held-out split."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("split") not in ("dev", "heldout"):
        raise ValueError("M1 evaluator accepts only dev or heldout")
    fields = ("Y", "Y_pool", "centers", "radii", "record_id")
    chunks = {key: [] for key in fields}
    count = 0
    for shard in manifest.get("shards", []):
        path = directory / shard["file"]
        if sha256(path) != shard["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            missing = set(fields) - set(payload.files)
            if missing:
                raise ValueError(f"M1 shard is missing {sorted(missing)}")
            shard_count = len(payload["Y"])
            if shard_count != int(shard["n_records"]):
                raise ValueError("M1 shard count differs from manifest")
            for key in fields:
                chunks[key].append(np.asarray(payload[key]))
            count += shard_count
    if count != int(manifest.get("n_records", -1)) or count == 0:
        raise ValueError("M1 manifest record count is incomplete")
    return {**{key: np.concatenate(value) for key, value in chunks.items()}, "manifest": manifest}


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_finite(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_m1_artifacts(output, *, summary, records, bank_arrays, metadata):
    """Write immutable M1 banks and a dev-frozen protocol."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(
            {"metadata": _json_finite(metadata), "metrics": _json_finite(summary)},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    (output / "records.json").write_text(
        json.dumps(_json_finite(records), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    np.savez_compressed(output / "bank.npz", **bank_arrays)
    if metadata.get("split") == "dev" and metadata.get("protocol_eligible", True):
        protocol = {**_json_finite(metadata), "frozen": True}
        (output / "protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def load_frozen_m1_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("split") != "dev" or protocol.get("frozen") is not True:
        raise ValueError("heldout evaluation requires a frozen dev protocol")
    for key, value in FROZEN_M1_DECODER.items():
        if protocol.get(key) != value:
            raise ValueError(f"dev protocol differs from frozen {key}")
    return protocol


def validate_m1_protocol_matches(protocol, current):
    keys = (
        "architecture",
        "bank_size",
        "euler_steps",
        "initial_noise_scale",
        "signal_rank",
        "gibbs_tau",
        "refinement_passes",
        "radii_mm",
        "checkpoint_sha256",
        "config_sha256",
        "anatomy_content_sha256",
        "dev_manifest_sha256",
        "heldout_manifest_sha256",
        "sensor_projector_sha256",
        "decoder_code_sha256",
        "evaluation_seed",
    )
    for key in keys:
        if protocol.get(key) != current.get(key):
            raise ValueError(f"heldout setting differs from dev protocol: {key}")


def _estimate_iir_ringing_samples(sos, max_try=100_000):
    sections = np.asarray(sos, dtype=np.float64)
    state = np.zeros((len(sections), 2), dtype=np.float64)
    impulse = np.zeros(1000, dtype=np.float64)
    impulse[0] = 1.0
    last_good = len(impulse)
    threshold = 0.0
    for chunk_index in range(int(np.ceil(max_try / len(impulse)))):
        response, state = sosfilt(sections, impulse, zi=state)
        impulse[0] = 0.0
        magnitude = np.abs(response)
        threshold = max(0.001 * float(np.max(magnitude)), threshold)
        indices = np.flatnonzero(magnitude > threshold)
        if indices.size:
            last_good = int(indices[-1])
        else:
            return int((chunk_index - 1) * len(impulse) + last_good)
    return int(max_try)


def bandpass_sensor_data(y, *, sfreq_hz=1000.0):
    """Apply the article's zero-phase 2--25 Hz order-four Butterworth filter."""
    data = np.asarray(y, dtype=np.float64)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError("sensor data must be a nonempty finite matrix")
    frequency = float(sfreq_hz)
    if not np.isfinite(frequency) or frequency <= 50:
        raise ValueError("sampling frequency must exceed twice the 25 Hz cutoff")
    sos = butter(4, [2.0, 25.0], btype="bandpass", fs=frequency, output="sos")
    padlen = min(_estimate_iir_ringing_samples(sos), data.shape[-1] - 1)
    return np.asarray(sosfiltfilt(sos, data, axis=-1, padlen=padlen), dtype=np.float64)


def anatomy_content_sha256(geometry, leadfield):
    """Hash numerical anatomy used by rho, independent of NPZ metadata."""
    digest = hashlib.sha256()
    for array in (
        geometry.vertices,
        geometry.hemisphere,
        geometry.adjacency.indptr,
        geometry.adjacency.indices,
        geometry.adjacency.data,
        np.asarray(leadfield),
    ):
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def load_validated_m1_anatomy(path, config, manifest):
    """Load either the original M1 bundle or a provenance-linked recovery."""
    from .geometry import CorticalGeometry

    geometry, leadfield, provenance = CorticalGeometry.load_bundle(
        path, allow_fragmented_hemispheres=True
    )
    expected = config["dataset"]["anatomy_sha256"]
    if manifest.get("anatomy_sha256") != expected or manifest.get("config") != config["dataset"]:
        raise ValueError("M1 dataset manifest differs from the configured experiment")
    actual = sha256(path)
    linked = provenance.get("original_anatomy_sha256")
    if actual != expected and linked != expected:
        raise ValueError("M1 anatomy is neither original nor provenance-linked recovery")
    n_components = connected_components(geometry.adjacency, directed=False)[0]
    if (
        leadfield.shape != (102, 4699)
        or len(geometry.vertices) != 4699
        or n_components != 3
        or not np.array_equal(leadfield, leadfield.astype("f4").astype("f8"))
        or config["dataset"]["radii_mm"] != [10.0, 8.0, 10.0]
    ):
        raise ValueError("M1 anatomy does not match legacy numerical invariants")
    return geometry, leadfield, provenance, anatomy_content_sha256(geometry, leadfield)


def load_sensor_projector(path, n_sensors):
    with np.load(path, allow_pickle=False) as payload:
        projector = np.asarray(payload["projector"], dtype=np.float64)
    if projector.shape != (n_sensors, n_sensors) or not np.isfinite(projector).all():
        raise ValueError("sensor projector shape differs from M1 sensors")
    if not np.allclose(projector @ projector, projector, atol=1e-12, rtol=0):
        raise ValueError("sensor projector must be idempotent")
    return projector


def m1_patch_dictionaries(leadfield, geometry, projector):
    dictionaries = {}
    for radius in sorted(set(FROZEN_M1_DECODER["radii_mm"])):
        dictionaries[radius] = projector @ patch_dictionary(
            leadfield, geometry, radius, sigma_ratio=1 / 3
        )
    return np.stack(
        [dictionaries[radius] for radius in FROZEN_M1_DECODER["radii_mm"]]
    )


def _m1_sample_seed(evaluation_seed, split, record_index):
    payload = f"{int(evaluation_seed)}:{split}:{int(record_index)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _prefixed_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def summarize_m1_records(records, *, prefix):
    if not records:
        raise ValueError("at least one M1 record is required")
    disconnected = sum(row[f"{prefix}_geodesic_disconnected_count"] for row in records)
    pair_count = 3 * len(records)
    assigned = [
        value
        for row in records
        for value in row[f"{prefix}_assigned_geodesic_mm"]
    ]
    finite = [value for value in assigned if np.isfinite(value)]
    return {
        "record_count": len(records),
        "geodesic_mae_mm": (
            float(np.mean(assigned)) if disconnected == 0 else None
        ),
        "geodesic_connected_mae_mm": float(np.mean(finite)) if finite else None,
        "geodesic_diameter_penalized_mae_mm": float(
            np.mean([row[f"{prefix}_geodesic_diameter_penalized_mae_mm"] for row in records])
        ),
        "geodesic_mae_at_30_mm": float(
            np.mean([row[f"{prefix}_geodesic_mae_at_30_mm"] for row in records])
        ),
        "geodesic_disconnected_count": int(disconnected),
        "geodesic_disconnected_rate": float(disconnected / pair_count),
        "euclidean_mae_mm": float(
            np.mean([row[f"{prefix}_euclidean_mae_mm"] for row in records])
        ),
    }


def evaluate_m1_model(
    model,
    checkpoint,
    data,
    geometry,
    dictionaries,
    projector,
    *,
    device,
    bank_size,
    evaluation_seed,
    graph_diameter_mm,
):
    """Evaluate all records while preserving each full candidate bank."""
    split = data["manifest"]["split"]
    count = len(data["Y"])
    bank_arrays = {
        "xyz_mm": np.empty((count, bank_size, 3, 3), dtype="f4"),
        "projected_vertices": np.empty((count, bank_size, 3), dtype="i8"),
        "aligned_vertices": np.empty((count, bank_size, 3), dtype="i8"),
        "scores": np.empty((count, bank_size), dtype="f8"),
        "weights": np.empty((count, bank_size), dtype="f8"),
        "medoid_centers": np.empty((count, 3), dtype="i8"),
        "refined_centers": np.empty((count, 3), dtype="i8"),
        "sample_seed": np.empty(count, dtype="u8"),
    }
    device = torch.device(device)
    model.eval()
    records = []
    started = time.monotonic()
    for index in range(count):
        seed = _m1_sample_seed(evaluation_seed, split, index)
        y_pool = torch.from_numpy(np.asarray(data["Y_pool"][index : index + 1])).to(device)
        xyz = sample_fixed_k_bank(
            model,
            y_pool,
            checkpoint["mean_mm"].detach().cpu().numpy(),
            checkpoint["std_mm"].detach().cpu().numpy(),
            bank_size=bank_size,
            steps=FROZEN_M1_DECODER["euler_steps"],
            initial_noise_scale=FROZEN_M1_DECODER["initial_noise_scale"],
            seed=seed,
        )
        score_y = projector @ bandpass_sensor_data(data["Y"][index], sfreq_hz=1000.0)
        decoded = decode_m1_bank(xyz, score_y, dictionaries, geometry)
        medoid_metrics = m1_localization_metrics(
            true_centers=data["centers"][index],
            predicted_centers=decoded["medoid_centers"],
            geometry=geometry,
            graph_diameter_mm=graph_diameter_mm,
        )
        refined_metrics = m1_localization_metrics(
            true_centers=data["centers"][index],
            predicted_centers=decoded["refined_centers"],
            geometry=geometry,
            graph_diameter_mm=graph_diameter_mm,
        )
        records.append(
            {
                "record_index": index,
                "record_id": str(data["record_id"][index]),
                "sample_seed": seed,
                "mean_effective_sample_size": decoded["mean_effective_sample_size"],
                "weight_entropy": decoded["weight_entropy"],
                "permutation_fraction": decoded["permutation_fraction"],
                "radius_slot_permutation_fraction": decoded[
                    "radius_slot_permutation_fraction"
                ],
                "score_min": float(decoded["scores"].min()),
                "score_max": float(decoded["scores"].max()),
                "duplicate_initial": decoded["duplicate_initial"],
                "refinement_moves": decoded["refinement_moves"],
                "refinement_evaluations": decoded["refinement_evaluations"],
                "refinement_passes": decoded["refinement_passes"],
                "refinement_converged": decoded["refinement_converged"],
                **_prefixed_metrics("medoid", medoid_metrics),
                **_prefixed_metrics("refined", refined_metrics),
            }
        )
        bank_arrays["xyz_mm"][index] = xyz
        for key in (
            "projected_vertices",
            "aligned_vertices",
            "scores",
            "weights",
            "medoid_centers",
            "refined_centers",
        ):
            bank_arrays[key][index] = decoded[key]
        bank_arrays["sample_seed"][index] = seed
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(
                json.dumps(
                    {
                        "split": split,
                        "records": index + 1,
                        "total": count,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    summary = {
        "medoid": summarize_m1_records(records, prefix="medoid"),
        "refined": summarize_m1_records(records, prefix="refined"),
        "mean_effective_sample_size": float(
            np.mean([row["mean_effective_sample_size"] for row in records])
        ),
        "mean_weight_entropy": float(np.mean([row["weight_entropy"] for row in records])),
        "mean_refinement_moves": float(np.mean([row["refinement_moves"] for row in records])),
        "mean_permutation_fraction": float(
            np.mean([row["permutation_fraction"] for row in records])
        ),
        "mean_radius_slot_permutation_fraction": float(
            np.mean([row["radius_slot_permutation_fraction"] for row in records])
        ),
        "refinement_converged_fraction": float(
            np.mean([row["refinement_converged"] for row in records])
        ),
        "duplicate_initial_fraction": float(
            np.mean([row["duplicate_initial"] for row in records])
        ),
        "runtime_seconds": time.monotonic() - started,
    }
    return summary, records, bank_arrays


def main(argv=None):
    from omegaconf import OmegaConf

    from .train import build_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--sensor-projector", type=Path, required=True)
    parser.add_argument("--distance-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=("mlp", "transformer"), required=True)
    parser.add_argument("--split", choices=("dev", "heldout"), default="dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-seed", type=int, default=20260923)
    parser.add_argument("--bank-size", type=int, default=FROZEN_M1_DECODER["bank_size"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    dev_manifest_path = args.dataset_root / "dev" / "manifest.json"
    heldout_manifest_path = args.dataset_root / "heldout" / "manifest.json"
    dev_manifest = json.loads(dev_manifest_path.read_text())
    split_manifest = json.loads(
        (args.dataset_root / args.split / "manifest.json").read_text()
    )
    geometry, leadfield, provenance, anatomy_content = load_validated_m1_anatomy(
        args.anatomy, config, split_manifest
    )
    checkpoint_hash = sha256(args.checkpoint)
    metadata = {
        "split": args.split,
        "architecture": args.architecture,
        **FROZEN_M1_DECODER,
        "bank_size": int(args.bank_size),
        "evaluation_seed": int(args.evaluation_seed),
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": sha256(args.config),
        "anatomy_file_sha256": sha256(args.anatomy),
        "anatomy_content_sha256": anatomy_content,
        "dev_manifest_sha256": sha256(dev_manifest_path),
        "heldout_manifest_sha256": sha256(heldout_manifest_path),
        "sensor_projector_sha256": sha256(args.sensor_projector),
        "decoder_code_sha256": source_hash(),
        "anatomy_provenance": provenance,
        "candidate_score": "rank_6_joint_subspace_alignment_rho",
        "recording_filter": "zero_phase_butterworth_order4_2_25_hz",
        "aggregation": "component_first_gibbs_weighted_observed_cortical_medoid",
        "refinement": "strict_rho_one_ring_slot_specific_four_pass",
    }
    if args.split == "heldout":
        if args.protocol is None:
            raise ValueError("heldout evaluation requires --protocol from dev")
        protocol = load_frozen_m1_protocol(args.protocol)
        metadata["bank_size"] = protocol["bank_size"]
        metadata["evaluation_seed"] = protocol["evaluation_seed"]
        validate_m1_protocol_matches(protocol, metadata)
        if args.limit:
            raise ValueError("heldout evaluation cannot use --limit")
    elif args.protocol is not None:
        raise ValueError("--protocol is reserved for heldout")
    if metadata["bank_size"] < 1 or args.limit < 0:
        raise ValueError("bank size must be positive and limit nonnegative")

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if (
        checkpoint.get("architecture") != args.architecture
        or checkpoint.get("config") != config
        or "ema" not in checkpoint
        or "mean_mm" not in checkpoint
        or "std_mm" not in checkpoint
    ):
        raise ValueError("checkpoint does not match the requested M1 experiment")
    model = build_model(config, args.architecture).to(args.device)
    model.load_state_dict(checkpoint["ema"])
    projector = load_sensor_projector(args.sensor_projector, leadfield.shape[0])
    geometry.cache_distances(args.distance_cache)
    graph_diameter = float(np.max(geometry._distances[np.isfinite(geometry._distances)]))
    dictionaries = m1_patch_dictionaries(leadfield, geometry, projector)
    data = load_m1_evaluation_arrays(args.dataset_root / args.split)
    if args.limit:
        limit = min(int(args.limit), len(data["Y"]))
        for key in ("Y", "Y_pool", "centers", "radii", "record_id"):
            data[key] = data[key][:limit]
    metadata["record_count"] = len(data["Y"])
    metadata["protocol_eligible"] = bool(
        args.split == "dev"
        and not args.limit
        and metadata["bank_size"] == FROZEN_M1_DECODER["bank_size"]
    )
    summary, records, arrays = evaluate_m1_model(
        model,
        checkpoint,
        data,
        geometry,
        dictionaries,
        projector,
        device=args.device,
        bank_size=metadata["bank_size"],
        evaluation_seed=metadata["evaluation_seed"],
        graph_diameter_mm=graph_diameter,
    )
    write_m1_artifacts(
        args.output,
        summary=summary,
        records=records,
        bank_arrays=arrays,
        metadata=metadata,
    )
    print(json.dumps(_json_finite({"metadata": metadata, "metrics": summary}), indent=2))


@dataclass(frozen=True)
class RefinementResult:
    centroids: np.ndarray
    score: float
    n_moves: int
    n_evaluations: int
    n_passes: int
    converged: bool


def refine_centroids(
    initial_centroids,
    dictionaries,
    signal_basis,
    geometry,
    *,
    max_passes=FROZEN_M1_DECODER["refinement_passes"],
    improvement_tol=0.0,
):
    """Greedily increase joint rho over one-ring mesh neighbours."""
    centers = np.asarray(initial_centroids, dtype=np.int64).copy()
    patches = np.asarray(dictionaries, dtype=np.float64)
    signal = np.asarray(signal_basis, dtype=np.float64)
    passes_limit = int(max_passes)
    if centers.ndim != 1 or not len(centers):
        raise ValueError("initial centroids must be a nonempty vector")
    if patches.ndim != 3 or patches.shape[0] != len(centers):
        raise ValueError("one patch dictionary is required per source slot")
    if len(set(centers.tolist())) != len(centers):
        raise ValueError("initial centroids must be distinct")
    if passes_limit != max_passes or passes_limit <= 0:
        raise ValueError("max_passes must be a positive integer")
    tolerance = float(improvement_tol)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("improvement tolerance must be finite and nonnegative")

    def score(current):
        return subspace_score(
            signal, candidate_matrices(patches, np.asarray(current)[None, :])[0]
        )

    current_score = score(centers)
    evaluations = 1
    moves = 0
    converged = False
    passes = 0
    for passes in range(1, passes_limit + 1):
        moved = False
        for slot, center in enumerate(centers.copy()):
            occupied = set(centers.tolist()) - {int(center)}
            best_vertex = int(center)
            best_score = current_score
            for candidate in np.asarray(geometry.neighbors(center), dtype=np.int64):
                candidate = int(candidate)
                if candidate in occupied:
                    continue
                proposal = centers.copy()
                proposal[slot] = candidate
                candidate_score = score(proposal)
                evaluations += 1
                if candidate_score > best_score + tolerance:
                    best_vertex, best_score = candidate, candidate_score
            if best_vertex != int(center):
                centers[slot] = best_vertex
                current_score = best_score
                moves += 1
                moved = True
        if not moved:
            converged = True
            break
    return RefinementResult(
        centroids=centers,
        score=float(current_score),
        n_moves=moves,
        n_evaluations=evaluations,
        n_passes=passes,
        converged=converged,
    )
