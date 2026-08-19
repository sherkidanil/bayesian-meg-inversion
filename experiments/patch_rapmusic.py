"""Fixed-orientation RAP MUSIC over an explicit candidate dictionary."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import time

import numpy as np


_WORKER_DICTIONARY: np.ndarray | None = None
_WORKER_VERTEX_IDS: np.ndarray | None = None
_WORKER_N_SOURCES: int | None = None
_WORKER_SIGNAL_RANK: int | None = None


@dataclass(frozen=True)
class FixedRapMusicResult:
    candidate_indices: np.ndarray
    correlations: np.ndarray


def _initialize_worker(
    dictionary: np.ndarray,
    vertex_ids: np.ndarray,
    n_sources: int,
    signal_rank: int,
) -> None:
    global _WORKER_DICTIONARY, _WORKER_VERTEX_IDS
    global _WORKER_N_SOURCES, _WORKER_SIGNAL_RANK
    _WORKER_DICTIONARY = np.asarray(dictionary, dtype=np.float64)
    _WORKER_VERTEX_IDS = np.asarray(vertex_ids, dtype=np.int64)
    _WORKER_N_SOURCES = int(n_sources)
    _WORKER_SIGNAL_RANK = int(signal_rank)


def _batch_worker(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if (
        _WORKER_DICTIONARY is None
        or _WORKER_VERTEX_IDS is None
        or _WORKER_N_SOURCES is None
        or _WORKER_SIGNAL_RANK is None
    ):
        raise RuntimeError("RAP MUSIC worker was not initialized")
    try:
        from threadpoolctl import threadpool_limits

        context = threadpool_limits(limits=1)
    except (ImportError, AttributeError, OSError):
        # Some mixed BLAS installations expose threadpoolctl but fail during
        # dynamic-library inspection. Runs can still cap BLAS through the
        # documented OPENBLAS/OMP/MKL environment variables.
        context = None
    started = time.perf_counter()
    if context is None:
        result = rap_music_fixed(
            data,
            _WORKER_DICTIONARY,
            n_sources=_WORKER_N_SOURCES,
            signal_rank=_WORKER_SIGNAL_RANK,
            candidate_vertex_ids=_WORKER_VERTEX_IDS,
        )
    else:
        with context:
            result = rap_music_fixed(
                data,
                _WORKER_DICTIONARY,
                n_sources=_WORKER_N_SOURCES,
                signal_rank=_WORKER_SIGNAL_RANK,
                candidate_vertex_ids=_WORKER_VERTEX_IDS,
            )
    return (
        result.candidate_indices,
        result.correlations,
        float(time.perf_counter() - started),
    )


def concatenate_patch_dictionaries(
    dictionaries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate slot dictionaries and retain candidate-to-vertex metadata."""
    patches = np.asarray(dictionaries, dtype=np.float64)
    if patches.ndim != 3 or not np.all(np.isfinite(patches)):
        raise ValueError("dictionaries must have shape (slots, sensors, vertices)")
    slots, _, vertices = patches.shape
    if slots == 0 or vertices == 0:
        raise ValueError("patch dictionaries must be nonempty")
    combined = np.concatenate([patches[slot] for slot in range(slots)], axis=1)
    vertex_ids = np.tile(np.arange(vertices, dtype=np.int64), slots)
    radius_slots = np.repeat(np.arange(slots, dtype=np.int64), vertices)
    return combined, vertex_ids, radius_slots


def rap_music_fixed(
    data: np.ndarray,
    dictionary: np.ndarray,
    *,
    n_sources: int,
    signal_rank: int | None = None,
    candidate_vertex_ids: np.ndarray | None = None,
) -> FixedRapMusicResult:
    """Run recursive applied-and-projected MUSIC for scalar topographies."""
    values = np.asarray(data, dtype=np.float64)
    candidates = np.asarray(dictionary, dtype=np.float64)
    if values.ndim != 2 or candidates.ndim != 2:
        raise ValueError("data and dictionary must be matrices")
    if values.shape[0] != candidates.shape[0]:
        raise ValueError("data and dictionary sensor dimensions do not match")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(candidates)):
        raise ValueError("data and dictionary must be finite")
    source_count = int(n_sources)
    rank = source_count if signal_rank is None else int(signal_rank)
    if source_count <= 0 or rank < source_count:
        raise ValueError("signal_rank must be at least n_sources > 0")
    if rank > min(values.shape):
        raise ValueError("signal_rank exceeds the available data rank")

    if candidate_vertex_ids is None:
        vertex_ids = np.arange(candidates.shape[1], dtype=np.int64)
    else:
        vertex_ids = np.asarray(candidate_vertex_ids, dtype=np.int64)
        if vertex_ids.shape != (candidates.shape[1],):
            raise ValueError("candidate_vertex_ids must align with dictionary columns")
    if np.unique(vertex_ids).size < source_count:
        raise ValueError("fewer unique candidate vertices than requested sources")

    signal_basis, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(values.shape + candidates.shape)
    if singular_values[rank - 1] <= tolerance * singular_values[0]:
        raise ValueError("data do not support the requested signal rank")
    signal_basis = signal_basis[:, :rank]

    selected: list[int] = []
    selected_vertices: set[int] = set()
    correlations: list[float] = []
    q = np.empty((values.shape[0], 0), dtype=np.float64)
    for step in range(source_count):
        projected_candidates = candidates - q @ (q.T @ candidates)
        candidate_norms = np.linalg.norm(projected_candidates, axis=0)

        # RAP MUSIC keeps all original signal-subspace components after
        # projection. Truncating to ``rank - step`` would instead implement
        # TRAP MUSIC and changes the selected candidate set.
        projected_signal = signal_basis - q @ (q.T @ signal_basis)
        overlap = projected_signal.T @ projected_candidates
        scores = np.full(candidates.shape[1], -np.inf, dtype=np.float64)
        stable = candidate_norms > tolerance
        scores[stable] = np.linalg.norm(overlap[:, stable], axis=0) / candidate_norms[
            stable
        ]
        for vertex in selected_vertices:
            scores[vertex_ids == vertex] = -np.inf
        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            raise ValueError("no independent RAP MUSIC candidate remains")

        chosen = projected_candidates[:, index]
        chosen_norm = float(np.linalg.norm(chosen))
        q = np.column_stack([q, chosen / chosen_norm])
        selected.append(index)
        selected_vertices.add(int(vertex_ids[index]))
        correlations.append(float(np.clip(scores[index], 0.0, 1.0)))

    return FixedRapMusicResult(
        candidate_indices=np.asarray(selected, dtype=np.int64),
        correlations=np.asarray(correlations, dtype=np.float64),
    )


def rap_music_fixed_batch(
    data: np.ndarray,
    dictionary: np.ndarray,
    *,
    n_sources: int,
    signal_rank: int,
    candidate_vertex_ids: np.ndarray | None = None,
    workers: int = 1,
) -> dict[str, np.ndarray]:
    """Apply fixed RAP MUSIC to a paired ``(samples, 2, sensors, time)`` batch."""
    values = np.asarray(data, dtype=np.float64)
    candidates = np.asarray(dictionary, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("data must have shape (samples, 2, sensors, time)")
    if candidates.ndim != 2 or candidates.shape[0] != values.shape[2]:
        raise ValueError("dictionary and data sensor dimensions do not match")
    if candidate_vertex_ids is None:
        vertex_ids = np.arange(candidates.shape[1], dtype=np.int64)
    else:
        vertex_ids = np.asarray(candidate_vertex_ids, dtype=np.int64)
    if vertex_ids.shape != (candidates.shape[1],):
        raise ValueError("candidate_vertex_ids must align with dictionary columns")
    worker_count = int(workers)
    if worker_count != workers or worker_count <= 0:
        raise ValueError("workers must be a positive integer")
    flat = [values[sample, branch] for sample in range(len(values)) for branch in range(2)]
    initializer_args = (candidates, vertex_ids, int(n_sources), int(signal_rank))
    if worker_count == 1:
        _initialize_worker(*initializer_args)
        rows = [_batch_worker(item) for item in flat]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("fork"),
            initializer=_initialize_worker,
            initargs=initializer_args,
        ) as pool:
            rows = list(pool.map(_batch_worker, flat, chunksize=1))
    return {
        "candidate_indices": np.stack([row[0] for row in rows]).reshape(
            len(values), 2, int(n_sources)
        ),
        "correlations": np.stack([row[1] for row in rows]).reshape(
            len(values), 2, int(n_sources)
        ),
        "runtime_seconds": np.asarray([row[2] for row in rows], dtype=np.float64).reshape(
            len(values), 2
        ),
    }
