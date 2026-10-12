import numpy as np
import pytest

from megflow.anatomy import validate_reference_geometry, restrict_surface_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def test_reference_checks_units_vertex_order_and_graph(geometry, tmp_path):
    path = tmp_path / 'reference.npz'
    rows, cols = geometry.adjacency.nonzero()
    unique = rows > cols
    edges = np.stack([rows[unique], cols[unique]], axis=1)
    def write(vertices, weights):
        np.savez(path, vertices=vertices, edge_index=edges, edge_weight_mm=weights)
    write(geometry.vertices / 1000, np.full(len(edges), 2.))
    validate_reference_geometry(geometry, path)
    write(geometry.vertices[::-1] / 1000, np.full(len(edges), 2.))
    with pytest.raises(ValueError, match='vertices'):
        validate_reference_geometry(geometry, path)
    write(geometry.vertices / 1000, np.full(len(edges), 3.))
    with pytest.raises(ValueError, match='edge'):
        validate_reference_geometry(geometry, path)


def test_restriction_preserves_paths_through_excluded_transit_vertices():
    # Retained 0 and 2 have no direct edge; excluded 1 carries the cortical path.
    graph = csr_matrix(np.array([[0., 2, 0, 0, 0], [2, 0, 3, 0, 0],
                                 [0, 3, 0, 0, 0], [0, 0, 0, 0, 4],
                                 [0, 0, 0, 4, 0]]))
    keep = np.array([0, 2, 3, 4])
    reduced = restrict_surface_graph(graph, keep)
    assert reduced[0, 1] == 5
    np.testing.assert_allclose(dijkstra(reduced), dijkstra(graph)[keep][:, keep])


def test_restriction_matches_all_pairs_on_mesh_with_loops():
    rng = np.random.default_rng(9)
    xyz = rng.normal(size=(12, 3))
    distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
    graph = np.where(distances < 2, distances, 0)
    keep = np.array([0, 3, 5, 7, 9, 11])
    reduced = restrict_surface_graph(csr_matrix(graph), keep)
    np.testing.assert_allclose(dijkstra(reduced), dijkstra(graph)[keep][:, keep], atol=1e-12)
