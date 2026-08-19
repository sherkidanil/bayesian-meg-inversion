#!/usr/bin/env python3
"""Greedy classical source scan using Gaussian-patch topographies."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


@dataclass(frozen=True)
class PatchScanResult:
    source_slots: np.ndarray
    centroids: np.ndarray
    residual: float


_WORKER_DICTIONARIES: np.ndarray | None = None


def _initialize_scan_worker(dictionaries: np.ndarray) -> None:
    global _WORKER_DICTIONARIES
    _WORKER_DICTIONARIES = np.asarray(dictionaries, dtype=np.float64)


def _scan_worker(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    if _WORKER_DICTIONARIES is None:
        raise RuntimeError("patch scan worker was not initialized")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        context = None
    else:
        context = threadpool_limits(limits=1)
    started = time.perf_counter()
    if context is None:
        result = greedy_patch_scan_covariance(covariance, _WORKER_DICTIONARIES)
    else:
        with context:
            result = greedy_patch_scan_covariance(covariance, _WORKER_DICTIONARIES)
    return (
        result.source_slots,
        result.centroids,
        result.residual,
        float(time.perf_counter() - started),
    )


def build_gaussian_patch_dictionaries(
    localization_G: np.ndarray,
    graph,
    params_spatial: dict,
) -> np.ndarray:
    """Precompute one ``G @ Phi.T`` dictionary for every source-radius slot."""
    leadfield = np.asarray(localization_G, dtype=np.float64)
    if leadfield.ndim != 2 or not np.all(np.isfinite(leadfield)):
        raise ValueError("localization_G must be a finite matrix")
    if graph.shape != (leadfield.shape[1], leadfield.shape[1]):
        raise ValueError("graph and localization_G vertex counts do not match")
    radii = np.asarray(params_spatial["R_mm"], dtype=np.float64)
    amplitudes = np.asarray(params_spatial["A0"], dtype=np.float64)
    sigma_scale = float(params_spatial["sigma_mm_scale"])
    if radii.ndim != 1 or radii.shape != amplitudes.shape or radii.size == 0:
        raise ValueError("R_mm and A0 must be nonempty aligned vectors")
    if np.any(radii <= 0.0) or sigma_scale <= 0.0:
        raise ValueError("radii and sigma scale must be positive")

    distances = np.asarray(
        dijkstra(graph, directed=False, limit=float(np.max(radii))),
        dtype=np.float64,
    )
    dictionaries = np.empty(
        (radii.size, leadfield.shape[0], leadfield.shape[1]), dtype=np.float64
    )
    for slot, (radius, amplitude) in enumerate(zip(radii, amplitudes, strict=True)):
        support = np.isfinite(distances) & (distances <= float(radius))
        rows, columns = np.nonzero(support)
        sigma = float(radius) * sigma_scale
        values = float(amplitude) * np.exp(
            -0.5 * np.square(distances[rows, columns] / sigma)
        )
        profiles = csr_matrix(
            (values, (rows, columns)), shape=distances.shape, dtype=np.float64
        )
        dictionaries[slot] = np.asarray(profiles @ leadfield.T).T
    return dictionaries


def greedy_patch_scan_covariance(
    covariance: np.ndarray,
    dictionaries: np.ndarray,
) -> PatchScanResult:
    """Greedily add the radius/vertex pair with maximal explained energy.

    Every radius slot is consumed exactly once and a vertex cannot be repeated.
    The vectorized score is algebraically equivalent to QR evaluation of every
    candidate subspace.
    """
    sensor_covariance = np.asarray(covariance, dtype=np.float64)
    patches = np.asarray(dictionaries, dtype=np.float64)
    if patches.ndim != 3 or not np.all(np.isfinite(patches)):
        raise ValueError("dictionaries must have shape (slots, sensors, vertices)")
    slots, sensors, vertices = patches.shape
    if slots == 0 or vertices < slots:
        raise ValueError("there must be at least one distinct vertex per source slot")
    if sensor_covariance.shape != (sensors, sensors):
        raise ValueError("covariance and dictionary sensor dimensions do not match")
    if not np.all(np.isfinite(sensor_covariance)):
        raise ValueError("covariance contains non-finite values")
    energy = float(np.trace(sensor_covariance))
    if energy <= 0.0:
        raise ValueError("covariance must contain positive sensor energy")

    remaining_slots = list(range(slots))
    used_vertices: set[int] = set()
    selected_slots: list[int] = []
    selected_vertices: list[int] = []
    q = np.empty((sensors, 0), dtype=np.float64)
    tolerance = np.finfo(np.float64).eps * max(sensors, vertices)
    for _ in range(slots):
        candidate = np.concatenate([patches[slot] for slot in remaining_slots], axis=1)
        residualized = candidate - q @ (q.T @ candidate)
        denominator = np.sum(np.square(residualized), axis=0)
        numerator = np.sum(
            residualized * (sensor_covariance @ residualized), axis=0
        )
        incremental = np.full(denominator.shape, -np.inf, dtype=np.float64)
        stable = denominator > tolerance
        incremental[stable] = numerator[stable] / denominator[stable]
        for block in range(len(remaining_slots)):
            offset = block * vertices
            for vertex in used_vertices:
                incremental[offset + vertex] = -np.inf
        flat_index = int(np.argmax(incremental))
        if not np.isfinite(incremental[flat_index]):
            raise ValueError("no linearly independent patch candidate remains")
        block, vertex = divmod(flat_index, vertices)
        slot = remaining_slots.pop(block)
        chosen = patches[slot, :, vertex]
        orthogonal = chosen - q @ (q.T @ chosen)
        norm = float(np.linalg.norm(orthogonal))
        if norm <= tolerance:
            raise ValueError("chosen patch is numerically dependent")
        q = np.column_stack([q, orthogonal / norm])
        selected_slots.append(slot)
        selected_vertices.append(vertex)
        used_vertices.add(vertex)

    explained = float(np.trace(q.T @ sensor_covariance @ q))
    residual = float(np.sqrt(np.clip(1.0 - explained / energy, 0.0, 1.0)))
    return PatchScanResult(
        source_slots=np.asarray(selected_slots, dtype=np.int64),
        centroids=np.asarray(selected_vertices, dtype=np.int64),
        residual=residual,
    )


def scan_patch_covariance_batch(
    covariances: np.ndarray,
    dictionaries: np.ndarray,
    *,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    """Scan a paired covariance batch, optionally across forked CPU workers."""
    values = np.asarray(covariances, dtype=np.float64)
    patches = np.asarray(dictionaries, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("covariances must have shape (samples, 2, sensors, sensors)")
    if patches.ndim != 3 or values.shape[2:] != (
        patches.shape[1],
        patches.shape[1],
    ):
        raise ValueError("covariance and dictionary sensor dimensions do not match")
    worker_count = int(workers)
    if worker_count != workers or worker_count <= 0:
        raise ValueError("workers must be a positive integer")
    flat = [values[sample, branch] for sample in range(len(values)) for branch in range(2)]
    if worker_count == 1:
        _initialize_scan_worker(patches)
        rows = [_scan_worker(covariance) for covariance in flat]
    else:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_scan_worker,
            initargs=(patches,),
        ) as pool:
            rows = list(pool.map(_scan_worker, flat, chunksize=1))
    slots = np.stack([row[0] for row in rows]).reshape(len(values), 2, -1)
    centroids = np.stack([row[1] for row in rows]).reshape(len(values), 2, -1)
    residuals = np.asarray([row[2] for row in rows]).reshape(len(values), 2)
    runtime = np.asarray([row[3] for row in rows]).reshape(len(values), 2)
    return {
        "source_slots": slots,
        "centroids": centroids,
        "residuals": residuals,
        "runtime_seconds": runtime,
    }
