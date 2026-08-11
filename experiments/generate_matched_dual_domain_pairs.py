#!/usr/bin/env python3
"""Generate matched original/shifted MEG pairs under two persisted lead fields."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.stats import gamma


SHARED_PAYLOAD_KEYS = {
    "G_rebuilt",
    "G_anna",
    "vertices",
    "edge_index",
    "edge_weight_mm",
    "normals",
    "params_spatial",
    "params_temporal",
    "params_noise",
    "noise_level",
    "meta",
}

SENSOR_VIEW_KEYS = (
    "Ys_rebuilt",
    "Ys_rebuilt_shifted",
    "Ys_anna",
    "Ys_anna_shifted",
)


def shift_centroids_anna(
    centroids: np.ndarray,
    vertices_m: np.ndarray,
    *,
    delta_m: np.ndarray = np.array([0.003, 0.003, 0.001]),
    min_dist_mm: float = 30.0,
    k_neighbors: int = 10,
) -> dict[str, Any]:
    """Replay Anna's nearest snap and Cartesian ten-neighbor correction."""
    original = np.asarray(centroids, dtype=np.int64)
    vertices = np.asarray(vertices_m, dtype=np.float64)
    delta = np.asarray(delta_m, dtype=np.float64)
    if original.ndim != 1 or original.size < 2:
        raise ValueError("centroids must be a one-dimensional multi-source array")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or delta.shape != (3,):
        raise ValueError("vertices/delta must use 3-D coordinates")
    if k_neighbors < 1 or k_neighbors > len(vertices):
        raise ValueError("k_neighbors is outside the mesh")

    target_coords = vertices[original] + delta
    tree = cKDTree(vertices)
    _distances, simple = tree.query(target_coords, k=1)
    simple = np.asarray(simple, dtype=np.int64)
    shifted = simple.copy()
    min_dist_m = float(min_dist_mm) / 1000.0
    correction_triggered = bool(np.any(pdist(vertices[simple]) < min_dist_m))
    valid = True
    if correction_triggered:
        _neighbor_distances, neighbor_indices = tree.query(
            target_coords, k=int(k_neighbors)
        )
        if int(k_neighbors) == 1:
            neighbor_indices = np.asarray(neighbor_indices)[:, None]
        best_error = float("inf")
        final_centroids: np.ndarray | None = None
        for combination in itertools.product(*np.asarray(neighbor_indices)):
            candidate = np.asarray(combination, dtype=np.int64)
            candidate_coords = vertices[candidate]
            if np.all(pdist(candidate_coords) >= min_dist_m):
                error = float(
                    np.sum(np.linalg.norm(candidate_coords - target_coords, axis=1))
                )
                if error < best_error:
                    best_error = error
                    final_centroids = candidate
        if final_centroids is None:
            valid = False
        else:
            shifted = final_centroids
    return {
        "shifted_centroids": shifted,
        "simple_centroids": simple,
        "target_coords_m": target_coords,
        "real_delta_m": np.linalg.norm(vertices[original] - vertices[shifted], axis=1),
        "pair_distances_m": pdist(vertices[shifted]),
        "correction_triggered": correction_triggered,
        "valid": valid,
    }


def _standardize_clean(sensor_data: np.ndarray) -> np.ndarray:
    value = np.asarray(sensor_data, dtype=np.float64)
    scale = float(value.reshape(-1).std(ddof=1))
    return value / (scale + 1e-12)


def sensor_pair_from_latents(
    leadfield: np.ndarray,
    spatial_original: np.ndarray,
    spatial_shifted: np.ndarray,
    temporal: np.ndarray,
    sensor_noise: np.ndarray,
    *,
    noise_level: float,
) -> dict[str, np.ndarray]:
    """Project common temporal/noise latents for one lead-field domain."""
    g = np.asarray(leadfield, dtype=np.float64)
    spatial_a = np.asarray(spatial_original, dtype=np.float64)
    spatial_b = np.asarray(spatial_shifted, dtype=np.float64)
    time = np.asarray(temporal, dtype=np.float64)
    noise = np.asarray(sensor_noise, dtype=np.float64)
    if spatial_a.shape != spatial_b.shape:
        raise ValueError("original and shifted spatial profiles differ in shape")
    if time.shape[0] != spatial_a.shape[0] or g.shape[1] != spatial_a.shape[1]:
        raise ValueError("leadfield, spatial profiles, and temporal sources do not align")
    clean_a = _standardize_clean((g @ spatial_a.T) @ time)
    clean_b = _standardize_clean((g @ spatial_b.T) @ time)
    if noise.shape != clean_a.shape:
        raise ValueError("sensor noise shape does not match projected data")
    return {
        "clean_original": clean_a,
        "clean_shifted": clean_b,
        "Y_original": clean_a + float(noise_level) * noise,
        "Y_shifted": clean_b + float(noise_level) * noise,
    }


def generate_noise_latents(
    n_vertices: int, params: dict[str, Any], rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Generate Anna's filtered source-noise latents before lead-field projection."""
    fs = float(params["Fs"])
    ntpoints = int(params["ntpoints"])
    trial_length = int(float(params["window"]) * fs)
    n_noise_sources = int(params["N_noise_sources"])
    pad = int(params["pad"])
    if n_vertices < 1 or trial_length < 1 or n_noise_sources < 1 or pad < 0:
        raise ValueError("noise dimensions must be positive")
    trial_count = int(np.floor(ntpoints / trial_length + 1))
    source_indices = np.empty((trial_count, n_noise_sources), dtype=np.int64)
    source_noise = np.empty(
        (trial_count, n_noise_sources, trial_length), dtype=np.float64
    )
    nyquist = fs / 2.0
    bands = (
        np.array([8.0, 12.0]) / nyquist,
        np.array([15.0, 30.0]) / nyquist,
        np.array([30.0, 50.0]) / nyquist,
        np.array([50.0, 70.0]) / nyquist,
    )
    filters = [butter(3, band, btype="bandpass") for band in bands]
    for trial_index in range(trial_count):
        source_indices[trial_index] = rng.integers(
            0, int(n_vertices), size=n_noise_sources
        )
        white = rng.standard_normal((n_noise_sources, trial_length + 2 * pad))
        filtered = sum(
            filtfilt(b, a, white, axis=1) / (float(np.mean(band)) + 1e-12)
            for band, (b, a) in zip(bands, filters, strict=True)
        )
        source_noise[trial_index] = filtered[:, pad : pad + trial_length]
    return {"source_indices": source_indices, "source_noise": source_noise}


def project_noise_latents(
    leadfield: np.ndarray, latents: dict[str, np.ndarray], params: dict[str, Any]
) -> np.ndarray:
    """Project common source-noise latents through one lead field and standardize."""
    g = np.asarray(leadfield, dtype=np.float64)
    indices = np.asarray(latents["source_indices"], dtype=np.int64)
    source_noise = np.asarray(latents["source_noise"], dtype=np.float64)
    ntpoints = int(params["ntpoints"])
    trial_length = int(float(params["window"]) * float(params["Fs"]))
    n_noise_sources = int(params["N_noise_sources"])
    expected = (len(indices), n_noise_sources, trial_length)
    if source_noise.shape != expected or indices.shape != expected[:2]:
        raise ValueError("source-noise latent shapes do not match noise parameters")
    if np.any(indices < 0) or np.any(indices >= g.shape[1]):
        raise ValueError("source-noise indices are outside the lead field")
    projected = np.empty((g.shape[0], len(indices) * trial_length), dtype=np.float64)
    for trial_index, (trial_indices, trial_noise) in enumerate(
        zip(indices, source_noise, strict=True)
    ):
        start = trial_index * trial_length
        stop = start + trial_length
        projected[:, start:stop] = (
            g[:, trial_indices] @ trial_noise / float(n_noise_sources)
        )
    projected = projected[:, :ntpoints]
    return projected / (projected.reshape(-1).std(ddof=1) + 1e-12)


def _anna_min_snr(values: np.ndarray, params_temporal: dict) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError("sensor data must have shape (samples, sensors, time)")
    power = np.mean(np.square(data), axis=1)
    starts = np.asarray(params_temporal["t_start_sources"], dtype=np.int64)
    stops = np.asarray(params_temporal["t_stop_sources"], dtype=np.int64)
    if starts.shape != (3,) or stops.shape != (3,):
        raise ValueError("Anna SNR rule expects exactly three temporal sources")
    windows = (
        power[:, starts[0] : stops[0]],
        power[:, starts[1] : starts[2]],
        power[:, stops[1] : stops[2]],
    )
    noise_power = np.mean(power[:, stops[2] :], axis=1)
    source_power = [np.percentile(window, 90, axis=1) for window in windows]
    snr = [value / (noise_power + 1e-12) for value in source_power]
    return np.minimum.reduce(snr)


def anna_good_snr_mask(
    views: dict[str, np.ndarray],
    params_temporal: dict,
    *,
    threshold: float = 3.5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not views:
        raise ValueError("at least one sensor-data view is required")
    per_view = {
        name: _anna_min_snr(value, params_temporal) >= float(threshold)
        for name, value in views.items()
    }
    lengths = {len(mask) for mask in per_view.values()}
    if len(lengths) != 1:
        raise ValueError("sensor-data views have different sample counts")
    combined = np.logical_and.reduce(list(per_view.values()))
    return combined, per_view


def deterministic_split_indices(
    n_samples: int, dev_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < int(dev_size) < int(n_samples):
        raise ValueError("dev_size must leave a nonempty held-out split")
    permutation = np.random.default_rng(int(seed)).permutation(int(n_samples))
    return (
        np.asarray(permutation[: int(dev_size)], dtype=np.int64),
        np.asarray(permutation[int(dev_size) :], dtype=np.int64),
    )


def split_payload(
    payload: dict[str, Any], indices: np.ndarray, *, n_samples: int
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= n_samples):
        raise ValueError("split indices are invalid")
    output: dict[str, Any] = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if key in SHARED_PAYLOAD_KEYS or array.ndim == 0:
            output[key] = value
        elif array.shape[0] == int(n_samples):
            output[key] = array[selected]
        else:
            output[key] = value
    output["full_sample_indices"] = selected
    return output


def validate_dual_domain_payload(
    payload: dict[str, Any], *, expected_n: int
) -> tuple[int, int, int]:
    """Validate the locked artifact's four-view and lead-field contract."""
    missing = [key for key in (*SENSOR_VIEW_KEYS, "G_rebuilt", "G_anna") if key not in payload]
    if missing:
        raise KeyError(", ".join(missing))
    shapes = [np.asarray(payload[key]).shape for key in SENSOR_VIEW_KEYS]
    if any(len(shape) != 3 for shape in shapes) or len(set(shapes)) != 1:
        raise ValueError("all four sensor views must share shape (samples, sensors, time)")
    n_samples, n_sensors, n_times = shapes[0]
    if n_samples != int(expected_n):
        raise ValueError(f"expected {expected_n} retained samples, found {n_samples}")
    for key in ("G_rebuilt", "G_anna"):
        leadfield = np.asarray(payload[key])
        if leadfield.ndim != 2 or leadfield.shape[0] != n_sensors:
            raise ValueError(f"{key} does not match the sensor dimension")
    if np.asarray(payload["G_rebuilt"]).shape != np.asarray(payload["G_anna"]).shape:
        raise ValueError("persisted lead fields do not have matching shapes")
    for key in SENSOR_VIEW_KEYS:
        if not np.all(np.isfinite(np.asarray(payload[key]))):
            raise ValueError(f"{key} contains non-finite values")
    return int(n_samples), int(n_sensors), int(n_times)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def write_locked_splits(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    expected_n: int = 700,
    dev_size: int = 350,
    split_seed: int = 20260811,
) -> dict[str, Any]:
    """Write full/dev/held-out NPZs and a content-hashed manifest atomically."""
    validate_dual_domain_payload(payload, expected_n=expected_n)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dev_indices, heldout_indices = deterministic_split_indices(
        expected_n, dev_size, split_seed
    )
    files: dict[str, dict[str, Any]] = {}
    split_specs = (
        ("full", None),
        ("dev", dev_indices),
        ("heldout", heldout_indices),
    )
    for split_name, indices in split_specs:
        split_data = (
            dict(payload)
            if indices is None
            else split_payload(payload, indices, n_samples=expected_n)
        )
        split_data["split_kind"] = np.array(split_name)
        path = destination / f"{split_name}.npz"
        _atomic_savez(path, split_data)
        files[split_name] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "samples": expected_n if split_name == "full" else (
                dev_size if split_name == "dev" else expected_n - dev_size
            ),
        }
    manifest = {
        "schema_version": 1,
        "retained_samples": int(expected_n),
        "split": {
            "seed": int(split_seed),
            "dev_size": int(dev_size),
            "heldout_size": int(expected_n - dev_size),
        },
        "files": files,
    }
    manifest_path = destination / "manifest.json"
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest


def validate_written_bundle(output_dir: str | Path) -> dict[str, Any]:
    """Validate hashes, split isolation, geometry-derived truth, and retained SNR."""
    destination = Path(output_dir)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_n = int(manifest["retained_samples"])
    for split_name, entry in manifest["files"].items():
        path = destination / entry["path"]
        actual = _sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"{split_name} file hash differs from manifest")
    full_path = destination / manifest["files"]["full"]["path"]
    with np.load(full_path, allow_pickle=True) as full:
        validate_dual_domain_payload(full, expected_n=expected_n)
        if str(full["split_kind"].item()) != "full":
            raise ValueError("full artifact has the wrong split marker")
        files = set(full.files)
        truth_keys = {
            "vertices",
            "normals",
            "centroids_idx",
            "centroids_idx_shifted",
            "thetas",
            "thetas_shifted",
            "real_inputs_delta_m",
            "normal_angles_deg",
        }
        if truth_keys <= files:
            vertices = np.asarray(full["vertices"], dtype=np.float64)
            normals = np.asarray(full["normals"], dtype=np.float64)
            original = np.asarray(full["centroids_idx"], dtype=np.int64)
            shifted = np.asarray(full["centroids_idx_shifted"], dtype=np.int64)
            expected_theta = vertices[original].reshape(expected_n, -1)
            expected_shifted_theta = vertices[shifted].reshape(expected_n, -1)
            if not np.allclose(full["thetas"], expected_theta, atol=1e-7, rtol=0.0):
                raise ValueError("original theta does not match persisted centroids")
            if not np.allclose(
                full["thetas_shifted"], expected_shifted_theta, atol=1e-7, rtol=0.0
            ):
                raise ValueError("shifted theta does not match persisted centroids")
            expected_delta = np.linalg.norm(vertices[original] - vertices[shifted], axis=2)
            if not np.allclose(
                full["real_inputs_delta_m"], expected_delta, atol=1e-7, rtol=0.0
            ):
                raise ValueError("realized shift does not match persisted centroids")
            cosines = np.clip(
                np.sum(normals[original] * normals[shifted], axis=2), -1.0, 1.0
            )
            expected_angles = np.degrees(np.arccos(cosines))
            if not np.allclose(
                full["normal_angles_deg"], expected_angles, atol=2e-4, rtol=0.0
            ):
                raise ValueError("normal angles do not match persisted normals")
        threshold = float(manifest.get("generator", {}).get("snr_threshold", -np.inf))
        for key in SENSOR_VIEW_KEYS:
            snr_key = f"min_snr_{key}"
            if snr_key in files and np.any(np.asarray(full[snr_key]) < threshold):
                raise ValueError(f"{snr_key} violates the retained SNR threshold")
        generator = manifest.get("generator", {})
        leadfield_hashes = generator.get("leadfield_array_sha256", {})
        for key in ("G_rebuilt", "G_anna"):
            if key in leadfield_hashes and _array_sha256(full[key]) != leadfield_hashes[key]:
                raise ValueError(f"{key} array hash differs from generator manifest")
    split_indices: dict[str, np.ndarray] = {}
    for split_name in ("dev", "heldout"):
        entry = manifest["files"][split_name]
        with np.load(destination / entry["path"], allow_pickle=True) as split:
            if str(split["split_kind"].item()) != split_name:
                raise ValueError(f"{split_name} artifact has the wrong split marker")
            expected_size = int(entry["samples"])
            validate_dual_domain_payload(split, expected_n=expected_size)
            indices = np.asarray(split["full_sample_indices"], dtype=np.int64)
            if indices.shape != (expected_size,) or len(np.unique(indices)) != expected_size:
                raise ValueError(f"{split_name} indices are not unique and exact")
            split_indices[split_name] = indices
    dev_set = set(split_indices["dev"].tolist())
    heldout_set = set(split_indices["heldout"].tolist())
    if dev_set & heldout_set or dev_set | heldout_set != set(range(expected_n)):
        raise ValueError("dev/held-out indices are not a disjoint full partition")
    return {
        "valid": True,
        "retained_samples": expected_n,
        "split_union_size": len(dev_set | heldout_set),
        "checked_file_hashes": sorted(manifest["files"]),
    }


def build_mesh_adjacency(
    n_vertices: int, edge_index: np.ndarray, edge_weight_mm: np.ndarray
) -> csr_matrix:
    edges = np.asarray(edge_index, dtype=np.int64)
    weights = np.asarray(edge_weight_mm, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] != 2 or weights.shape != (len(edges),):
        raise ValueError("mesh edges and weights have incompatible shapes")
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([weights, weights])
    return coo_matrix(
        (values, (rows, columns)), shape=(int(n_vertices), int(n_vertices))
    ).tocsr()


def spatial_activation_for_centroids(
    centroids: np.ndarray, params: dict[str, Any], adjacency: csr_matrix
) -> np.ndarray:
    centers = np.asarray(centroids, dtype=np.int64)
    radii = np.asarray(params["R_mm"], dtype=np.float64)
    amplitudes = np.asarray(params["A0"], dtype=np.float64)
    sigma = radii * float(params["sigma_mm_scale"])
    if centers.shape != radii.shape or amplitudes.shape != radii.shape:
        raise ValueError("centroids and spatial parameters do not align")
    distances = np.asarray(dijkstra(adjacency, indices=centers, directed=False))
    activation = np.zeros((len(centers), adjacency.shape[0]), dtype=np.float64)
    for source_index in range(len(centers)):
        region = distances[source_index] <= radii[source_index]
        values = distances[source_index, region]
        activation[source_index, region] = amplitudes[source_index] * np.exp(
            -0.5 * np.square(values / sigma[source_index])
        )
    return activation


def random_spatial_activation(
    params: dict[str, Any], adjacency: csr_matrix, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n_sources = int(params["N_src"])
    n_vertices = int(adjacency.shape[0])
    minimum = float(params["min_dist"])
    centers = np.empty(n_sources, dtype=np.int64)
    centers[0] = rng.integers(n_vertices)
    distances = np.asarray(dijkstra(adjacency, indices=int(centers[0]), directed=False))
    eligible = distances >= minimum
    for source_index in range(1, n_sources):
        candidates = np.flatnonzero(eligible)
        if not len(candidates):
            raise RuntimeError("no centroid satisfies the minimum-distance rule")
        centers[source_index] = rng.choice(candidates)
        distances = np.asarray(
            dijkstra(adjacency, indices=int(centers[source_index]), directed=False)
        )
        eligible &= distances >= minimum
    return centers, spatial_activation_for_centroids(centers, params, adjacency)


def temporal_activation(
    params: dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    n_sources = int(params["N_src"])
    ntpoints = int(params["ntpoints"])
    starts = np.asarray(params["t_start_sources"], dtype=np.int64)
    stops = np.asarray(params["t_stop_sources"], dtype=np.int64)
    bands = np.asarray(params["frec_source_band"], dtype=np.float64)
    pad = int(params["pad"])
    shape = float(params["shape"])
    scale = float(params["scale"])
    z_left = gamma.ppf(float(params["qL"]), a=shape, scale=scale)
    z_right = gamma.ppf(float(params["qR"]), a=shape, scale=scale)
    result = np.zeros((n_sources, ntpoints), dtype=np.float64)
    for source_index in range(n_sources):
        start, stop = int(starts[source_index]), int(stops[source_index])
        segment_length = stop - start
        white = rng.standard_normal(segment_length + 2 * pad)
        b, a = butter(3, bands[source_index], btype="bandpass")
        signal = filtfilt(b, a, white) / (float(np.mean(bands[source_index])) + 1e-12)
        time_indices = np.arange(start, stop)
        z = z_left + (z_right - z_left) * (time_indices - start) / segment_length
        envelope = ((z_right - z_left) / segment_length) * (
            gamma.pdf(z, a=shape, scale=scale)
            / (
                gamma.cdf(z_right, a=shape, scale=scale)
                - gamma.cdf(z_left, a=shape, scale=scale)
            )
        )
        envelope /= envelope.std(ddof=1) + 1e-12
        segment = signal[:segment_length]
        segment /= segment.std(ddof=1) + 1e-12
        result[source_index, time_indices] = envelope * segment
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def simulate_dual_domain_candidate(
    *,
    seed: int,
    g_rebuilt: np.ndarray,
    g_anna: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    adjacency: csr_matrix,
    params_spatial: dict[str, Any],
    params_temporal: dict[str, Any],
    params_noise: dict[str, Any],
    noise_level: float,
    snr_threshold: float = 3.5,
) -> dict[str, Any]:
    """Generate one common-latent candidate and retain it only on four-view SNR."""
    rng = np.random.default_rng(int(seed))
    centroids, spatial_original = random_spatial_activation(
        params_spatial, adjacency, rng
    )
    temporal = temporal_activation(params_temporal, rng)
    shifted = shift_centroids_anna(
        centroids,
        vertices,
        min_dist_mm=float(params_spatial["min_dist"]),
        k_neighbors=10,
    )
    if not shifted["valid"]:
        return {"accepted": False, "reason": "shift_min_dist", "seed": int(seed)}
    shifted_centroids = np.asarray(shifted["shifted_centroids"], dtype=np.int64)
    spatial_shifted = spatial_activation_for_centroids(
        shifted_centroids, params_spatial, adjacency
    )
    noise_latents = generate_noise_latents(len(vertices), params_noise, rng)
    rebuilt_noise = project_noise_latents(g_rebuilt, noise_latents, params_noise)
    anna_noise = project_noise_latents(g_anna, noise_latents, params_noise)
    rebuilt_pair = sensor_pair_from_latents(
        g_rebuilt,
        spatial_original,
        spatial_shifted,
        temporal,
        rebuilt_noise,
        noise_level=noise_level,
    )
    anna_pair = sensor_pair_from_latents(
        g_anna,
        spatial_original,
        spatial_shifted,
        temporal,
        anna_noise,
        noise_level=noise_level,
    )
    view_map = {
        "Ys_rebuilt": rebuilt_pair["Y_original"][None, ...],
        "Ys_rebuilt_shifted": rebuilt_pair["Y_shifted"][None, ...],
        "Ys_anna": anna_pair["Y_original"][None, ...],
        "Ys_anna_shifted": anna_pair["Y_shifted"][None, ...],
    }
    combined, _per_view = anna_good_snr_mask(
        view_map, params_temporal, threshold=snr_threshold
    )
    min_snr = {
        key: float(_anna_min_snr(value, params_temporal)[0])
        for key, value in view_map.items()
    }
    if not bool(combined[0]):
        return {
            "accepted": False,
            "reason": "snr",
            "seed": int(seed),
            "min_snr": min_snr,
        }
    original_normals = np.asarray(normals, dtype=np.float64)[centroids]
    shifted_normals = np.asarray(normals, dtype=np.float64)[shifted_centroids]
    cosines = np.clip(np.sum(original_normals * shifted_normals, axis=1), -1.0, 1.0)
    return {
        "accepted": True,
        "seed": int(seed),
        "centroids_idx": centroids,
        "centroids_idx_shifted": shifted_centroids,
        "simple_centroids_idx_shifted": np.asarray(shifted["simple_centroids"]),
        "thetas": np.asarray(vertices)[centroids].reshape(-1),
        "thetas_shifted": np.asarray(vertices)[shifted_centroids].reshape(-1),
        "real_inputs_delta_m": np.asarray(shifted["real_delta_m"]),
        "normal_angles_deg": np.degrees(np.arccos(cosines)),
        "correction_triggered": bool(shifted["correction_triggered"]),
        "min_snr": min_snr,
        "temporal_sha256": _array_sha256(temporal),
        "noise_source_indices_sha256": _array_sha256(noise_latents["source_indices"]),
        "noise_source_timeseries_sha256": _array_sha256(noise_latents["source_noise"]),
        **{
            key: np.asarray(value[0], dtype=np.float32)
            for key, value in view_map.items()
        },
    }


def _git_state(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("numpy", "scipy", "mne", "nibabel", "jax", "equinox"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def map_surface_normals_to_vertices(
    target_vertices: np.ndarray,
    source_vertices: np.ndarray,
    source_normals: np.ndarray,
    mri_to_head: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Transform MRI surface geometry to head coordinates and map persisted vertices."""
    target = np.asarray(target_vertices, dtype=np.float64)
    source = np.asarray(source_vertices, dtype=np.float64)
    normals = np.asarray(source_normals, dtype=np.float64)
    transform = np.asarray(mri_to_head, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target vertices must have shape (N, 3)")
    if source.shape != normals.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source vertices and normals must share shape (M, 3)")
    if transform.shape != (4, 4):
        raise ValueError("MRI-to-head transform must have shape (4, 4)")
    source_homogeneous = np.concatenate(
        [source, np.ones((len(source), 1), dtype=np.float64)], axis=1
    )
    source_head = (source_homogeneous @ transform.T)[:, :3]
    normals_head = normals @ transform[:3, :3].T
    normals_head /= np.linalg.norm(normals_head, axis=1, keepdims=True) + 1e-12
    distances, indices = cKDTree(source_head).query(target, k=1)
    maximum = float(np.max(distances))
    if len(np.unique(indices)) != len(target):
        raise ValueError("target vertices do not map one-to-one to source-space vertices")
    return np.asarray(normals_head[indices], dtype=np.float32), maximum


def load_normals_for_vertices(
    vertices: np.ndarray, *, spacing: str = "ico4"
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map MNE sample surface normals to the persisted forward-model vertices."""
    import mne
    from mne.datasets import sample

    sample_path = Path(sample.data_path(download=False))
    subjects_dir = sample_path / "subjects"
    source_space = mne.setup_source_space(
        "sample", spacing=spacing, subjects_dir=subjects_dir, add_dist=False
    )
    source_vertices = np.concatenate(
        [part["rr"][part["vertno"]] for part in source_space], axis=0
    )
    source_normals = np.concatenate(
        [part["nn"][part["vertno"]] for part in source_space], axis=0
    )
    trans_path = sample_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    head_to_mri = mne.read_trans(trans_path, verbose=False)
    mri_to_head = mne.transforms.invert_transform(head_to_mri)
    mapped_normals, maximum = map_surface_normals_to_vertices(
        vertices, source_vertices, source_normals, mri_to_head["trans"]
    )
    if maximum > 1e-7:
        raise ValueError(
            f"persisted vertices do not map one-to-one to MNE {spacing} normals; max={maximum}"
        )
    surfaces = [subjects_dir / "sample" / "surf" / f"{hemi}.white" for hemi in ("lh", "rh")]
    provenance = {
        "mne_sample_data_path": str(sample_path),
        "spacing": spacing,
        "coordinate_transform": "MRI-to-head from sample_audvis_raw-trans.fif",
        "maximum_vertex_mapping_distance_m": maximum,
        "transform_file": {
            "path": str(trans_path),
            "sha256": _sha256_file(trans_path),
        },
        "surface_files": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)}
            for path in surfaces
        },
    }
    return mapped_normals, provenance


def _load_generation_inputs(
    rebuilt_path: Path, anna_path: Path
) -> dict[str, Any]:
    with np.load(rebuilt_path) as rebuilt:
        rebuilt_values = {key: np.array(rebuilt[key], copy=True) for key in rebuilt.files}
    with np.load(anna_path, allow_pickle=True) as anna:
        geometry = {
            key: np.array(anna[key], copy=True)
            for key in ("vertices", "edge_index", "edge_weight_mm")
        }
        for key in geometry:
            if not np.array_equal(rebuilt_values[key], geometry[key]):
                raise ValueError(f"Anna and rebuilt geometry differ at {key}")
        matrices = np.asarray(anna["G_for_simulation"])
        g_anna = np.array(matrices[0], copy=True)
        if not np.array_equal(matrices, np.broadcast_to(g_anna, matrices.shape)):
            raise ValueError("Anna dG=0 lead field is not constant across samples")
        params = {
            key: dict(anna[key].item())
            for key in ("params_spatial", "params_temporal", "params_noise")
        }
    if rebuilt_values["G"].shape != g_anna.shape:
        raise ValueError("lead fields have different shapes")
    return {
        "G_rebuilt": rebuilt_values["G"],
        "G_anna": g_anna,
        **geometry,
        **params,
    }


def _allocate_staging(
    staging_dir: Path, retained: int, sensors: int, times: int
) -> dict[str, np.memmap]:
    staging_dir.mkdir(parents=True, exist_ok=False)
    return {
        key: np.lib.format.open_memmap(
            staging_dir / f"{key}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(retained, sensors, times),
        )
        for key in SENSOR_VIEW_KEYS
    }


def generate_dataset_bundle(
    *,
    rebuilt_path: Path,
    anna_path: Path,
    output_dir: Path,
    retained: int,
    dev_size: int,
    split_seed: int,
    candidate_seed_start: int,
    max_candidates: int,
    snr_threshold: float,
    noise_level: float,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    start_git = _git_state(repository)
    if start_git["dirty"]:
        raise RuntimeError("generator requires a clean repository at run start")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    inputs = _load_generation_inputs(rebuilt_path, anna_path)
    normals, sample_provenance = load_normals_for_vertices(inputs["vertices"])
    adjacency = build_mesh_adjacency(
        len(inputs["vertices"]), inputs["edge_index"], inputs["edge_weight_mm"]
    )
    staging = _allocate_staging(
        output_dir / ".staging",
        retained,
        inputs["G_rebuilt"].shape[0],
        int(inputs["params_temporal"]["ntpoints"]),
    )
    n_sources = int(inputs["params_spatial"]["N_src"])
    sample_arrays: dict[str, np.ndarray] = {
        "centroids_idx": np.empty((retained, n_sources), dtype=np.int64),
        "centroids_idx_shifted": np.empty((retained, n_sources), dtype=np.int64),
        "simple_centroids_idx_shifted": np.empty((retained, n_sources), dtype=np.int64),
        "thetas": np.empty((retained, n_sources * 3), dtype=np.float32),
        "thetas_shifted": np.empty((retained, n_sources * 3), dtype=np.float32),
        "real_inputs_delta_m": np.empty((retained, n_sources), dtype=np.float32),
        "normal_angles_deg": np.empty((retained, n_sources), dtype=np.float32),
        "correction_triggered": np.empty(retained, dtype=bool),
        "candidate_seeds": np.empty(retained, dtype=np.int64),
        "temporal_sha256": np.empty(retained, dtype="U64"),
        "noise_source_indices_sha256": np.empty(retained, dtype="U64"),
        "noise_source_timeseries_sha256": np.empty(retained, dtype="U64"),
    }
    for key in SENSOR_VIEW_KEYS:
        sample_arrays[f"min_snr_{key}"] = np.empty(retained, dtype=np.float32)
    rejected = {"shift_min_dist": 0, "snr": 0}
    accepted = 0
    attempted = 0
    while accepted < retained and attempted < max_candidates:
        seed = int(candidate_seed_start + attempted)
        candidate = simulate_dual_domain_candidate(
            seed=seed,
            g_rebuilt=inputs["G_rebuilt"],
            g_anna=inputs["G_anna"],
            vertices=inputs["vertices"],
            normals=normals,
            adjacency=adjacency,
            params_spatial=inputs["params_spatial"],
            params_temporal=inputs["params_temporal"],
            params_noise=inputs["params_noise"],
            noise_level=noise_level,
            snr_threshold=snr_threshold,
        )
        attempted += 1
        if not candidate["accepted"]:
            rejected[candidate["reason"]] += 1
            continue
        for key in SENSOR_VIEW_KEYS:
            staging[key][accepted] = candidate[key]
            sample_arrays[f"min_snr_{key}"][accepted] = candidate["min_snr"][key]
        for key in (
            "centroids_idx",
            "centroids_idx_shifted",
            "simple_centroids_idx_shifted",
            "thetas",
            "thetas_shifted",
            "real_inputs_delta_m",
            "normal_angles_deg",
            "correction_triggered",
            "temporal_sha256",
            "noise_source_indices_sha256",
            "noise_source_timeseries_sha256",
        ):
            sample_arrays[key][accepted] = candidate[key]
        sample_arrays["candidate_seeds"][accepted] = seed
        accepted += 1
        if accepted == 1 or accepted % max(1, min(25, retained)) == 0:
            print(
                f"[generation] retained={accepted}/{retained} attempted={attempted}",
                flush=True,
            )
    if accepted != retained:
        raise RuntimeError(
            f"retained only {accepted}/{retained} after {attempted} candidates"
        )
    for array in staging.values():
        array.flush()
    meta = {
        "schema": "matched-dual-domain-pairs-v1",
        "generator_git_commit": start_git["commit"],
        "candidate_seed_start": int(candidate_seed_start),
        "attempted_candidates": int(attempted),
        "retained_candidates": int(accepted),
        "rejected": rejected,
        "snr_threshold": float(snr_threshold),
        "shift_m": [0.003, 0.003, 0.001],
        "min_dist_correction_neighbors": 10,
        "noise_coupling": "common source indices/time series across G; common sensor noise within original/shifted per G",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "packages": _package_versions(),
            "sample_data": sample_provenance,
        },
        "input_files": {
            "rebuilt": {
                "path": str(rebuilt_path.resolve()),
                "sha256": _sha256_file(rebuilt_path),
            },
            "anna_dg0": {
                "path": str(anna_path.resolve()),
                "sha256": _sha256_file(anna_path),
            },
        },
        "leadfield_array_sha256": {
            "G_rebuilt": _array_sha256(inputs["G_rebuilt"]),
            "G_anna": _array_sha256(inputs["G_anna"]),
        },
    }
    payload: dict[str, Any] = {
        **staging,
        **sample_arrays,
        "G_rebuilt": inputs["G_rebuilt"],
        "G_anna": inputs["G_anna"],
        "vertices": inputs["vertices"],
        "edge_index": inputs["edge_index"],
        "edge_weight_mm": inputs["edge_weight_mm"],
        "normals": normals,
        "params_spatial": np.array(inputs["params_spatial"], dtype=object),
        "params_temporal": np.array(inputs["params_temporal"], dtype=object),
        "params_noise": np.array(inputs["params_noise"], dtype=object),
        "noise_level": np.array(noise_level, dtype=np.float32),
        "N_src": np.array(n_sources, dtype=np.int64),
        "Fs": np.array(inputs["params_noise"]["Fs"], dtype=np.float32),
        "meta": np.array(meta, dtype=object),
    }
    manifest = write_locked_splits(
        payload,
        output_dir,
        expected_n=retained,
        dev_size=dev_size,
        split_seed=split_seed,
    )
    end_git = _git_state(repository)
    stable = start_git["commit"] == end_git["commit"] and not end_git["dirty"]
    manifest.update(
        {
            "generator": meta,
            "run_start_git": start_git,
            "artifact_write_git": end_git,
            "repository_changed_during_run": not stable,
        }
    )
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / ".manifest.final.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_manifest.replace(manifest_path)
    report = [
        "# Matched dual-domain dataset generation",
        "",
        f"- Retained pairs: {retained}",
        f"- Attempted candidates: {attempted}",
        f"- Rejected by shift correction: {rejected['shift_min_dist']}",
        f"- Rejected by four-view SNR: {rejected['snr']}",
        f"- Split: dev={dev_size}, held-out={retained - dev_size}, seed={split_seed}",
        f"- Generator commit: `{start_git['commit']}`",
        f"- Repository changed during run: `{not stable}`",
        "- Held-out outcome metrics were not computed or loaded during generation.",
        "- Both persisted lead fields are stored in every split artifact.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report))
    if not stable:
        raise RuntimeError("repository changed during dataset generation")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt-artifact", type=Path, required=True)
    parser.add_argument("--anna-dg0", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retained", type=int, default=700)
    parser.add_argument("--dev-size", type=int, default=350)
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--candidate-seed-start", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--snr-threshold", type=float, default=3.5)
    parser.add_argument("--noise-level", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = generate_dataset_bundle(
        rebuilt_path=args.rebuilt_artifact,
        anna_path=args.anna_dg0,
        output_dir=args.output_dir,
        retained=args.retained,
        dev_size=args.dev_size,
        split_seed=args.split_seed,
        candidate_seed_start=args.candidate_seed_start,
        max_candidates=args.max_candidates,
        snr_threshold=args.snr_threshold,
        noise_level=args.noise_level,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
