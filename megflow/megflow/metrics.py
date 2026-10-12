"""Set-valued localization metrics with explicit hemisphere misses."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def localization_errors(true_c, pred_c, geom):
    true_c, pred_c = np.asarray(true_c, dtype=int), np.asarray(pred_c, dtype=int)
    if len(true_c) != len(pred_c):
        raise ValueError('localization MAE requires equal cardinalities')
    if not len(true_c):
        return np.array([])
    distances = geom.geodesic(true_c[:, None], pred_c[None, :])
    finite = distances[np.isfinite(distances)]
    # Minimize the number of infinite assignments first, then finite distance.
    penalty = (float(finite.max()) + 1) * (len(true_c) + 1) if finite.size else 1.
    rows, cols = linear_sum_assignment(np.where(np.isfinite(distances), distances, penalty))
    return distances[rows, cols]


def ospa(true_c, pred_c, geom, c_cutoff=30., p=1.):
    if c_cutoff <= 0 or p < 1:
        raise ValueError('OSPA requires cutoff>0 and p>=1')
    true_c, pred_c = np.asarray(true_c, dtype=int), np.asarray(pred_c, dtype=int)
    n, m = max(len(true_c), len(pred_c)), min(len(true_c), len(pred_c))
    if not n:
        return 0.
    cost = 0.
    if m:
        distances = geom.geodesic(true_c[:, None], pred_c[None, :])
        matrix = np.minimum(distances, c_cutoff) ** p
        rows, cols = linear_sum_assignment(matrix)
        cost = matrix[rows, cols].sum()
    return float(((cost + c_cutoff ** p * (n - m)) / n) ** (1 / p))
