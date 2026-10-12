"""Fixed-K graph-neighbor oracle ascent for M0 (no cross-K score comparison)."""

from dataclasses import dataclass
import numpy as np
from .forward import subspace_score


@dataclass
class AscentResult:
    centroids: np.ndarray
    score: float
    score_history: list[float]
    n_moves: int
    n_passes: int


def coordinate_ascent(centroids, dictionary, e_s, geom, min_separation_mm=30.,
                      n_passes=4, improvement_tol=1e-10):
    centers = np.asarray(centroids, dtype=int).copy()
    if len(set(centers)) != len(centers):
        raise ValueError('duplicate initial centroids')
    for j, center in enumerate(centers):
        others = np.delete(centers, j)
        if (geom.geodesic(center, others) < min_separation_mm).any():
            raise ValueError('initial centroids violate minimum separation')
    score = subspace_score(e_s, dictionary[:, centers])
    history = [score]
    passes = 0
    for passes in range(1, n_passes + 1):
        moved = False
        for j, center in enumerate(centers):
            others = np.delete(centers, j)
            best, best_score = int(center), score
            for candidate in sorted(geom.neighbors(center)):
                if candidate in others:
                    continue
                if (geom.geodesic(candidate, others) < min_separation_mm).any():
                    continue
                trial = centers.copy()
                trial[j] = candidate
                candidate_score = subspace_score(e_s, dictionary[:, trial])
                if candidate_score > best_score + improvement_tol:
                    best, best_score = int(candidate), candidate_score
            if best != center:
                centers[j], score = best, best_score
                history.append(score)
                moved = True
        if not moved:
            break
    return AscentResult(centers, score, history, len(history) - 1, passes)
