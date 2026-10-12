import numpy as np
import pytest
from scipy.sparse import csr_matrix

from megflow.geometry import CorticalGeometry


@pytest.fixture
def geometry():
    vertices = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0],
                         [0, 10, 0], [2, 10, 0], [4, 10, 0]], dtype=float)
    graph = np.zeros((6, 6))
    for a, b in [(0, 1), (1, 2), (3, 4), (4, 5)]:
        graph[a, b] = graph[b, a] = 2
    return CorticalGeometry(vertices, np.tile([0., 0., 1.], (6, 1)),
                            csr_matrix(graph), np.array([0, 0, 0, 1, 1, 1]))
