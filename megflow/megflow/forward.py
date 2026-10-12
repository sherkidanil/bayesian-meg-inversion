"""Gaussian cortical patches and subspace scores, without Gram inversion."""

import numpy as np
from scipy.linalg import eigh


def column_basis(b, eps=1e-10):
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 2 or not np.isfinite(b).all():
        raise ValueError('B must be a finite matrix')
    u, s, _ = np.linalg.svd(b, full_matrices=False)
    threshold = eps * max(b.shape) * s[0] if s.size else 0.
    return u[:, s > threshold]


def patch_matrix(geom, centroids, radii, sigma_ratio=1 / 3):
    centroids = np.asarray(centroids, dtype=int)
    radii = np.broadcast_to(np.asarray(radii, dtype=float), centroids.shape)
    if not np.isfinite(radii).all() or (radii <= 0).any() or sigma_ratio <= 0:
        raise ValueError('patch radius and sigma_ratio must be positive')
    phi = np.zeros((len(centroids), len(geom.vertices)))
    for j, (center, radius) in enumerate(zip(centroids, radii)):
        distances = geom.geodesic_from(center)
        inside = distances <= radius
        phi[j, inside] = np.exp(-.5 * (distances[inside] / (radius * sigma_ratio)) ** 2)
    return phi


def topographies(g, geom, centroids, radii, sigma_ratio=1 / 3):
    return np.asarray(g, dtype=float) @ patch_matrix(geom, centroids, radii, sigma_ratio).T


def patch_dictionary(g, geom, radius, sigma_ratio=1 / 3):
    """Precompute one known-radius topography for every vertex without dense Phi."""
    dictionary = np.empty_like(g, dtype=np.float64)
    for center in range(len(geom.vertices)):
        distances = geom.geodesic_from(center)
        inside = distances <= radius
        weights = np.exp(-.5 * (distances[inside] / (radius * sigma_ratio)) ** 2)
        dictionary[:, center] = g[:, inside] @ weights
    return dictionary


def subspace_score(e_s, b, eps=1e-10):
    if b.shape[1] == 0:
        raise ValueError('score requires at least one source')
    q = column_basis(b, eps)
    return float(np.clip(np.sum((e_s.T @ q) ** 2) / b.shape[1], 0, 1))


def residual_energy(y, b):
    q = column_basis(b)
    residual = y - q @ (q.T @ y)
    return float(np.sum(residual ** 2))


def signal_subspace(y, k, alpha=2., r_max=40):
    """Eigendecompose sensor covariance; fix vector signs deterministically.

    YY^T is the sensor covariance, not the forbidden source Gram matrix B^T B.
    r=2K is a score hyperparameter; the noise-free signal rank is at most K.
    """
    y = np.asarray(y, dtype=np.float64)
    values, vectors = eigh(y @ y.T, check_finite=False, driver='evd')
    values, vectors = values[::-1], vectors[:, ::-1]
    values = np.maximum(values, 0.)
    rank = int(np.sum(values > values[0] * 1e-12)) if values[0] > 0 else 0
    requested = min(max(k, round(alpha * k)), r_max, y.shape[0], y.shape[1])
    e_s = vectors[:, :min(requested, rank)].copy()
    if e_s.shape[1]:
        signs = np.sign(e_s[np.argmax(np.abs(e_s), axis=0), np.arange(e_s.shape[1])])
        e_s *= signs[None, :]
    return e_s, np.sqrt(values)


def rank_diagnostics(b, eps=1e-10):
    s = np.linalg.svd(b, compute_uv=False)
    threshold = eps * max(b.shape) * s[0] if s.size else 0.
    rank = int(np.sum(s > threshold))
    p = s[s > 0] / np.sum(s) if np.sum(s) else np.array([])
    effective = float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.
    condition = float(s[0] / s[-1]) if rank == b.shape[1] else np.inf
    return {'numerical_rank': rank, 'effective_rank': effective,
            'condition_number': condition, 'singular_values': s.tolist()}
