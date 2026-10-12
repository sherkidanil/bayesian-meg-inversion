"""Cortical graph geometry. All coordinates and distances are millimetres."""

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra


@dataclass
class CorticalGeometry:
    vertices: np.ndarray
    normals: np.ndarray
    adjacency: sparse.csr_matrix
    hemisphere: np.ndarray
    allow_fragmented_hemispheres: bool = False
    _distances: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.normals = np.asarray(self.normals, dtype=np.float64)
        self.hemisphere = np.asarray(self.hemisphere, dtype=np.int64)
        self.adjacency = sparse.csr_matrix(self.adjacency, dtype=np.float64)
        n = len(self.vertices)
        if self.vertices.shape != (n, 3) or self.normals.shape != (n, 3):
            raise ValueError('vertices and normals must have shape (N, 3)')
        if not np.isfinite(self.vertices).all() or not np.isfinite(self.normals).all():
            raise ValueError('geometry contains non-finite coordinates')
        if self.hemisphere.shape != (n,) or not np.isin(self.hemisphere, [0, 1]).all():
            raise ValueError('hemisphere must contain one 0/1 label per vertex')
        if self.adjacency.shape != (n, n):
            raise ValueError('adjacency shape differs from vertex count')
        self.adjacency.eliminate_zeros()
        if not np.isfinite(self.adjacency.data).all() or (self.adjacency.data <= 0).any():
            raise ValueError('graph weights must be positive and finite')
        if (self.adjacency - self.adjacency.T).nnz:
            raise ValueError('adjacency must be symmetric')
        rows, cols = self.adjacency.nonzero()
        if (self.hemisphere[rows] != self.hemisphere[cols]).any():
            raise ValueError('graph contains cross-hemisphere edges')
        n_components, _ = connected_components(self.adjacency, directed=False)
        if (len(np.unique(self.hemisphere)) != 2 or n_components < 2
                or (n_components != 2 and not self.allow_fragmented_hemispheres)):
            raise ValueError('geometry must have exactly two connected hemispheres')
        lengths = np.linalg.norm(self.normals, axis=1)
        if (lengths == 0).any():
            raise ValueError('surface normals must be nonzero')
        self.normals = self.normals / lengths[:, None]

    @property
    def fingerprint(self):
        digest = hashlib.sha256()
        for array in [self.vertices, self.hemisphere, self.adjacency.indptr,
                      self.adjacency.indices, self.adjacency.data]:
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

    def cache_distances(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'geodesic-{self.fingerprint}.npy'
        checksum_path = path.with_suffix('.sha256')
        def checksum():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest()
        if not path.exists():
            distances = dijkstra(self.adjacency, directed=False).astype(np.float32)
            temporary = path.with_suffix('.tmp.npy')
            np.save(temporary, distances)
            temporary.replace(path)
            checksum_path.write_text(checksum() + '\n')
        if not checksum_path.exists() or checksum_path.read_text().strip() != checksum():
            raise ValueError('geodesic cache checksum mismatch or missing checksum')
        distances = np.load(path, mmap_mode='r', allow_pickle=False)
        n = len(self.vertices)
        if distances.shape != (n, n) or distances.dtype != np.float32:
            raise ValueError('invalid geodesic cache shape or dtype')
        _, labels = connected_components(self.adjacency, directed=False)
        for start in range(0, n, 256):
            block = distances[start:start + 256]
            connected = labels[start:start + 256, None] == labels[None, :]
            if (np.isnan(block).any() or (block < 0).any()
                    or not np.array_equal(np.isfinite(block), connected)
                    or not np.allclose(block, distances[:, start:start + 256].T)):
                raise ValueError('invalid geodesic cache distances')
        if not np.all(np.diag(distances) == 0):
            raise ValueError('invalid geodesic cache diagonal')
        self._distances = distances
        return distances

    def geodesic_from(self, source):
        if self._distances is not None:
            return self._distances[int(source)]
        return dijkstra(self.adjacency, directed=False, indices=int(source))

    def geodesic(self, i, j):
        i, j = np.broadcast_arrays(np.asarray(i, dtype=int), np.asarray(j, dtype=int))
        if self._distances is not None:
            return self._distances[i, j]
        values = np.empty(i.shape, dtype=float)
        for source in np.unique(i):
            selected = i == source
            values[selected] = self.geodesic_from(source)[j[selected]]
        return values.item() if values.ndim == 0 else values

    def neighbors(self, vertex):
        start, end = self.adjacency.indptr[int(vertex):int(vertex) + 2]
        return self.adjacency.indices[start:end]

    def save_bundle(self, path, leadfield, provenance):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vertices_mm=self.vertices, normals=self.normals,
                            hemisphere=self.hemisphere,
                            edge_indptr=self.adjacency.indptr,
                            edge_indices=self.adjacency.indices,
                            edge_weights_mm=self.adjacency.data,
                            leadfield=np.asarray(leadfield, dtype=np.float64),
                            provenance_json=json.dumps(provenance, sort_keys=True))

    @classmethod
    def load_bundle(cls, path, *, allow_fragmented_hemispheres=False):
        with np.load(path, allow_pickle=False) as bundle:
            n = len(bundle['vertices_mm'])
            graph = sparse.csr_matrix((bundle['edge_weights_mm'], bundle['edge_indices'],
                                       bundle['edge_indptr']), shape=(n, n))
            geometry = cls(bundle['vertices_mm'], bundle['normals'], graph,
                           bundle['hemisphere'], allow_fragmented_hemispheres)
            leadfield = bundle['leadfield'].copy()
            provenance = json.loads(str(bundle['provenance_json']))
        if leadfield.ndim != 2 or leadfield.shape[1] != n or not np.isfinite(leadfield).all():
            raise ValueError('leadfield must be finite with one column per vertex')
        return geometry, leadfield, provenance
