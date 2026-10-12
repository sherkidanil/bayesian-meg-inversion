import numpy as np
import pytest

from megflow.forward import (column_basis, patch_matrix, subspace_score,
                             residual_energy, signal_subspace, rank_diagnostics)
from megflow.geometry import CorticalGeometry


def test_geodesic_infinite_across_hemispheres(geometry):
    assert geometry.geodesic(0, 2) == 4
    assert np.isinf(geometry.geodesic(0, 3))
    np.testing.assert_equal(geometry.geodesic_from(1), [2, 0, 2, np.inf, np.inf, np.inf])


def test_distance_cache_roundtrip_and_corruption(geometry, tmp_path):
    first = geometry.cache_distances(tmp_path)
    assert first.dtype == np.float32
    np.testing.assert_equal(first, geometry.cache_distances(tmp_path))
    path = next(tmp_path.glob('geodesic-*.npy'))
    np.save(path, np.zeros((2, 2)))
    with pytest.raises(ValueError, match='cache'):
        geometry.cache_distances(tmp_path)


def test_geometry_rejects_cross_hemisphere_edges(geometry):
    graph = geometry.adjacency.toarray()
    graph[0, 3] = graph[3, 0] = 10
    with pytest.raises(ValueError, match='hemisphere'):
        CorticalGeometry(geometry.vertices, geometry.normals, graph, geometry.hemisphere)


def test_cache_rejects_plausible_but_wrong_distances(geometry, tmp_path):
    distances = geometry.cache_distances(tmp_path).copy()
    np.save(next(tmp_path.glob('geodesic-*.npy')), distances * 10)
    with pytest.raises(ValueError, match='cache'):
        geometry.cache_distances(tmp_path)


def test_geometry_rejects_disconnected_fragments_within_hemisphere(geometry):
    graph = geometry.adjacency.toarray()
    graph[0, 1] = graph[1, 0] = 0
    with pytest.raises(ValueError, match='connected'):
        CorticalGeometry(geometry.vertices, geometry.normals, graph, geometry.hemisphere)


def test_patch_support_is_geodesic_and_truncated(geometry):
    phi = patch_matrix(geometry, [0, 3], [3., 1.], 1 / 3)
    np.testing.assert_allclose(phi[0], [1, np.exp(-2), 0, 0, 0, 0])
    np.testing.assert_equal(phi[1], [0, 0, 0, 1, 0, 0])


def test_rho_invariance():
    rng = np.random.default_rng(72)
    e = column_basis(rng.normal(size=(10, 5)))
    b = rng.normal(size=(10, 3))
    d = np.array([[2., 1, 0], [0, -3, .5], [0, 0, 4]])
    assert subspace_score(e, b) == pytest.approx(subspace_score(e, b @ d), abs=1e-12)


def test_rho_equals_music_at_k1():
    e = np.eye(3)[:, :2]
    b = np.array([[1.], [2.], [3.]])
    assert subspace_score(e, b) == pytest.approx(5 / 14)


def test_rank_deficient_score_keeps_k_denominator():
    b = np.array([[1., 2.], [0, 0], [0, 0]])
    assert column_basis(b).shape == (3, 1)
    assert subspace_score(np.eye(3)[:, :1], b) == pytest.approx(.5)
    result = rank_diagnostics(b)
    assert result['numerical_rank'] == 1
    assert np.isinf(result['condition_number'])
    assert result['effective_rank'] == pytest.approx(1)


def test_zero_and_full_residual():
    b = np.eye(3)[:, :2]
    assert residual_energy(b @ np.ones((2, 7)), b) == pytest.approx(0, abs=1e-12)
    assert residual_energy(np.eye(3), np.zeros((3, 1))) == pytest.approx(3)


def test_signal_subspace_is_deterministic_and_uses_data_rank():
    y = np.diag([3., 2., 0.]) @ np.ones((3, 8))
    first, spectrum = signal_subspace(y, k=1, alpha=2, r_max=40)
    second, _ = signal_subspace(y, k=1, alpha=2, r_max=40)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 1)
    assert spectrum.shape == (3,)


def test_bundle_roundtrip(geometry, tmp_path):
    g = np.arange(24.).reshape(4, 6)
    path = tmp_path / 'geometry.npz'
    geometry.save_bundle(path, g, {'kind': 'test_fixture'})
    loaded, leadfield, provenance = CorticalGeometry.load_bundle(path)
    np.testing.assert_array_equal(loaded.vertices, geometry.vertices)
    np.testing.assert_array_equal(leadfield, g)
    assert provenance['kind'] == 'test_fixture'
