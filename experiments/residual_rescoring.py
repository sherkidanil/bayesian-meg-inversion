#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.sparse.csgraph import dijkstra


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _estimate_iir_ringing_samples(sos: np.ndarray, max_try: int = 100_000) -> int:
    """Reproduce the ringing-length estimate used by MNE's IIR filter path."""
    sections = np.asarray(sos, dtype=np.float64)
    state = np.zeros((len(sections), 2), dtype=np.float64)
    chunk_size = 1000
    impulse = np.zeros(chunk_size, dtype=np.float64)
    impulse[0] = 1.0
    last_good = chunk_size
    threshold = 0.0
    for chunk_index in range(int(np.ceil(max_try / chunk_size))):
        response, state = sosfilt(sections, impulse, zi=state)
        impulse[0] = 0.0
        magnitude = np.abs(response)
        threshold = max(0.001 * float(np.max(magnitude)), threshold)
        indices = np.flatnonzero(magnitude > threshold)
        if indices.size:
            last_good = int(indices[-1])
        else:
            return int((chunk_index - 1) * chunk_size + last_good)
    return int(max_try)


def butterworth_bandpass_sensor_data(
    y: np.ndarray,
    *,
    sfreq_hz: float,
    low_hz: float = 2.0,
    high_hz: float = 25.0,
    order: int = 4,
) -> np.ndarray:
    """Apply the zero-phase Butterworth SOS filter used in the RAP MUSIC code."""
    data = _finite_matrix(y, "y")
    sampling_frequency = float(sfreq_hz)
    low = float(low_hz)
    high = float(high_hz)
    filter_order = int(order)
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
        raise ValueError("sfreq_hz must be finite and positive")
    if not 0.0 < low < high:
        raise ValueError("bandpass cutoffs must satisfy 0 < low_hz < high_hz")
    if high >= sampling_frequency / 2.0:
        raise ValueError("high_hz must be below the Nyquist frequency")
    if filter_order <= 0 or filter_order != order:
        raise ValueError("order must be a positive integer")
    if data.shape[-1] < 2:
        raise ValueError("y must contain at least two time samples")
    sos = butter(
        filter_order,
        [low, high],
        btype="bandpass",
        fs=sampling_frequency,
        output="sos",
    )
    padlen = min(_estimate_iir_ringing_samples(sos), data.shape[-1] - 1)
    return np.asarray(sosfiltfilt(sos, data, axis=-1, padlen=padlen), dtype=np.float64)


def orthonormal_column_basis(
    basis: np.ndarray, rtol: float | None = None
) -> np.ndarray:
    matrix = _finite_matrix(basis, "basis")
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.empty((matrix.shape[0], 0), dtype=np.float64)
    if rtol is None:
        tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    else:
        if rtol < 0:
            raise ValueError("rtol must be non-negative")
        tolerance = float(rtol) * singular[0]
    return u[:, singular > tolerance]


def subspace_residual(basis: np.ndarray, y: np.ndarray) -> float:
    matrix = _finite_matrix(basis, "basis")
    data = _finite_matrix(y, "y")
    if matrix.shape[0] != data.shape[0]:
        raise ValueError("basis and y must have the same sensor dimension")
    energy = float(np.sum(np.square(data, dtype=np.float64)))
    if energy <= 0.0:
        raise ValueError("y must contain positive sensor energy")
    q = orthonormal_column_basis(matrix)
    if q.shape[1] == 0:
        return 1.0
    explained = float(np.sum(np.square(q.T @ data, dtype=np.float64)))
    return float(np.sqrt(np.clip(1.0 - explained / energy, 0.0, 1.0)))


def subspace_residual_from_covariance(
    basis: np.ndarray, covariance: np.ndarray
) -> float:
    matrix = _finite_matrix(basis, "basis")
    sensor_covariance = _finite_matrix(covariance, "covariance")
    if sensor_covariance.shape[0] != sensor_covariance.shape[1]:
        raise ValueError("covariance must be square")
    if matrix.shape[0] != sensor_covariance.shape[0]:
        raise ValueError("basis and covariance must have the same sensor dimension")
    energy = float(np.trace(sensor_covariance))
    if energy <= 0.0:
        raise ValueError("covariance must contain positive sensor energy")
    q = orthonormal_column_basis(matrix)
    if q.shape[1] == 0:
        return 1.0
    explained = float(np.trace(q.T @ sensor_covariance @ q))
    return float(np.sqrt(np.clip(1.0 - explained / energy, 0.0, 1.0)))


def gaussian_profile(
    *,
    centroid: int,
    source_slot: int,
    graph,
    n_vertices: int,
    R_mm: np.ndarray,
    sigma_mm_scale: float,
    A0: np.ndarray,
) -> np.ndarray:
    radii = np.asarray(R_mm, dtype=np.float64)
    amplitudes = np.asarray(A0, dtype=np.float64)
    slot = int(source_slot)
    center = int(centroid)
    if radii.ndim != 1 or amplitudes.ndim != 1 or radii.shape != amplitudes.shape:
        raise ValueError("R_mm and A0 must be one-dimensional arrays of equal length")
    if not 0 <= slot < radii.size:
        raise IndexError(f"source_slot {slot} is outside {radii.size} sources")
    if graph.shape != (int(n_vertices), int(n_vertices)):
        raise ValueError("graph shape does not match n_vertices")
    if not 0 <= center < int(n_vertices):
        raise IndexError(f"centroid {center} is outside {n_vertices} vertices")
    radius = float(radii[slot])
    sigma = radius * float(sigma_mm_scale)
    if radius <= 0.0 or sigma <= 0.0:
        raise ValueError("R_mm and sigma_mm_scale must define a positive sigma")
    distances = np.asarray(
        dijkstra(graph, directed=False, indices=center, limit=radius),
        dtype=np.float64,
    )
    support = np.isfinite(distances) & (distances <= radius)
    profile = np.zeros(int(n_vertices), dtype=np.float64)
    profile[support] = float(amplitudes[slot]) * np.exp(
        -0.5 * np.square(distances[support] / sigma)
    )
    return profile


class ProjectedGaussianCache:
    def __init__(self, localization_G: np.ndarray, graph, params_spatial: dict):
        self._leadfield = _finite_matrix(localization_G, "localization_G")
        self._graph = graph
        if graph.shape != (self._leadfield.shape[1], self._leadfield.shape[1]):
            raise ValueError("localization_G vertices do not match graph shape")
        self._radii = np.asarray(params_spatial["R_mm"], dtype=np.float64)
        self._sigma_scale = float(params_spatial["sigma_mm_scale"])
        self._amplitudes = np.asarray(params_spatial["A0"], dtype=np.float64)
        if self._radii.shape != self._amplitudes.shape:
            raise ValueError("R_mm and A0 must have equal shape")
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def column(self, *, source_slot: int, centroid: int) -> np.ndarray:
        key = (int(source_slot), int(centroid))
        if key not in self._cache:
            profile = gaussian_profile(
                centroid=key[1],
                source_slot=key[0],
                graph=self._graph,
                n_vertices=self._leadfield.shape[1],
                R_mm=self._radii,
                sigma_mm_scale=self._sigma_scale,
                A0=self._amplitudes,
            )
            self._cache[key] = self._leadfield @ profile
        return self._cache[key]

    def matrix(self, centroids: np.ndarray) -> np.ndarray:
        vertices = np.asarray(centroids, dtype=np.int64)
        if vertices.shape != self._radii.shape:
            raise ValueError(f"expected centroid shape {self._radii.shape}, got {vertices.shape}")
        return np.column_stack(
            [
                self.column(source_slot=slot, centroid=int(vertex))
                for slot, vertex in enumerate(vertices)
            ]
        )


def residual_rescore_draws(
    theta_draws,
    Y,
    localization_G,
    vertices_m,
    graph,
    params_spatial,
) -> tuple[np.ndarray, int]:
    from metrics import project_coords_to_vertices

    draws = np.asarray(theta_draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[2] != 3 or draws.shape[0] == 0:
        raise ValueError(f"theta_draws must have shape (N, K, 3), got {draws.shape}")
    if not np.all(np.isfinite(draws)):
        raise ValueError("theta_draws contains non-finite values")
    data = _finite_matrix(Y, "Y")
    leadfield = _finite_matrix(localization_G, "localization_G")
    vertices = _finite_matrix(vertices_m, "vertices_m")
    if vertices.shape[1] != 3:
        raise ValueError("vertices_m must have shape (N_vertices, 3)")
    if leadfield.shape != (data.shape[0], vertices.shape[0]):
        raise ValueError("localization_G must match Y sensors and mesh vertices")
    radii = np.asarray(params_spatial["R_mm"])
    if draws.shape[1] != radii.size:
        raise ValueError("theta_draws source count must match params_spatial")

    covariance = data @ data.T
    cache = ProjectedGaussianCache(leadfield, graph, params_spatial)
    residuals = np.empty(draws.shape[0], dtype=np.float64)
    for index, draw in enumerate(draws):
        centroids = project_coords_to_vertices(draw, vertices)
        residuals[index] = subspace_residual_from_covariance(
            cache.matrix(centroids), covariance
        )
    return residuals, int(np.argmin(residuals))


def residual_gibbs_weights(
    residuals: np.ndarray,
    tau: float,
    score_form: str = "quadratic",
) -> np.ndarray:
    """Return stable Gibbs/pseudo-posterior weights from truth-free residuals."""
    values = np.asarray(residuals, dtype=np.float64)
    temperature = float(tau)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("residuals must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("residuals must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("tau must be finite and positive")
    if score_form == "quadratic":
        scores = np.square(values)
        logits = -(scores - np.min(scores)) / (2.0 * temperature**2)
    elif score_form == "linear":
        scores = values
        logits = -(scores - np.min(scores)) / temperature
    else:
        raise ValueError("score_form must be 'quadratic' or 'linear'")
    weights = np.exp(logits)
    normalization = float(np.sum(weights))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise FloatingPointError("Gibbs weights have invalid normalization")
    return weights / normalization


def residual_weighted_mean(
    theta_draws: np.ndarray,
    residuals: np.ndarray,
    tau: float,
    score_form: str = "quadratic",
) -> tuple[np.ndarray, np.ndarray]:
    """Average posterior draws along their first axis using residual Gibbs weights."""
    draws = np.asarray(theta_draws, dtype=np.float64)
    if draws.ndim < 2 or draws.shape[0] == 0 or not np.all(np.isfinite(draws)):
        raise ValueError("theta_draws must be finite with a nonempty draw axis")
    weights = residual_gibbs_weights(residuals, tau=tau, score_form=score_form)
    if draws.shape[0] != len(weights):
        raise ValueError("theta_draws and residuals must have the same draw count")
    return np.tensordot(weights, draws, axes=(0, 0)), weights
