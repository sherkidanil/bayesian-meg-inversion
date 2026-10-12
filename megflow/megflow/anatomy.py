"""Export MNE sample anatomy without loading any evaluation recordings."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .geometry import CorticalGeometry


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def restrict_surface_graph(full_graph, retained):
    """Eliminate transit-only vertices with min-plus updates.

    This preserves shortest-path distances between retained source vertices.
    Added edges represent cortical paths, not Euclidean jumps across missing mesh.
    """
    graph = sparse.csr_matrix(full_graph)
    retained = np.asarray(retained, dtype=int)
    if len(set(retained)) != len(retained):
        raise ValueError('retained vertices must be unique')
    neighbors = [dict(zip(graph[i].indices.tolist(), graph[i].data.tolist()))
                 for i in range(graph.shape[0])]
    removed = set(range(graph.shape[0])) - set(retained.tolist())
    while removed:
        vertex = min(removed, key=lambda i: (len(neighbors[i]), i))
        adjacent = list(neighbors[vertex].items())
        for index, (a, wa) in enumerate(adjacent):
            for b, wb in adjacent[index + 1:]:
                distance = wa + wb
                if distance < neighbors[a].get(b, np.inf):
                    neighbors[a][b] = distance
                    neighbors[b][a] = distance
        for a, _ in adjacent:
            neighbors[a].pop(vertex, None)
        neighbors[vertex] = {}
        removed.remove(vertex)
    remap = {int(old): new for new, old in enumerate(retained)}
    rows, cols, weights = [], [], []
    for old in retained:
        for neighbor, weight in neighbors[old].items():
            if neighbor in remap and neighbor != old:
                rows.append(remap[int(old)])
                cols.append(remap[neighbor])
                weights.append(weight)
    return sparse.csr_matrix((weights, (rows, cols)), shape=(len(retained), len(retained)))


def validate_reference_geometry(geometry, path, allow_added_edges=False):
    """Read only geometry members of a legacy NPZ; never access Ys or labels."""
    with np.load(path, allow_pickle=False) as source:
        vertices = np.asarray(source['vertices'], dtype=float) * 1000.
        edges = np.asarray(source['edge_index'], dtype=int)
        weights = np.asarray(source['edge_weight_mm'], dtype=float)
    if vertices.shape != geometry.vertices.shape or not np.allclose(
            vertices, geometry.vertices, rtol=0, atol=1e-4):
        raise ValueError('reference vertices differ in count, order, frame or units')
    graph = sparse.csr_matrix((np.r_[weights, weights],
                               (np.r_[edges[:, 0], edges[:, 1]],
                                np.r_[edges[:, 1], edges[:, 0]])), shape=geometry.adjacency.shape)
    common = graph.multiply(geometry.adjacency.astype(bool))
    if (graph - common).nnz:
        raise ValueError('reference edge topology contains missing edges')
    added_edges = (geometry.adjacency.astype(bool) != graph.astype(bool)).nnz // 2
    if added_edges and not allow_added_edges:
        raise ValueError('reference edge topology differs')
    difference = graph - geometry.adjacency.multiply(graph.astype(bool))
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-4:
        raise ValueError('reference edge lengths differ')
    return {'path': str(Path(path).resolve()),
            'vertices_sha256': hashlib.sha256(vertices.tobytes()).hexdigest(),
            'edges_sha256': hashlib.sha256(edges.tobytes() + weights.tobytes()).hexdigest(),
            'legacy_n_components': int(connected_components(graph, directed=False)[0]),
            'added_cortical_path_edges': added_edges,
            'match_tolerance_mm': 1e-4,
            'evaluation_recordings_accessed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-path', type=Path, required=True)
    parser.add_argument('--reference-dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    import mne
    start = time.monotonic()
    sample_path = args.sample_path.resolve()
    subjects = sample_path / 'subjects'
    raw = sample_path / 'MEG/sample/sample_audvis_raw.fif'
    trans = sample_path / 'MEG/sample/sample_audvis_raw-trans.fif'
    bem_path = subjects / 'sample/bem/sample-5120-bem-sol.fif'
    surface_paths = [subjects / 'sample/surf' / f'{hemi}.{surface}'
                     for hemi in ['lh', 'rh'] for surface in ['white', 'sphere', 'sphere.reg']]
    src = mne.setup_source_space('sample', spacing='ico4', subjects_dir=subjects,
                                 add_dist=False, verbose=False)
    original_src = src.copy()
    full_vertices = np.concatenate([hemi['rr'][hemi['vertno']] for hemi in original_src]) * 1000.
    full_topology = mne.spatial_src_adjacency(original_src, verbose=False).tocsr()
    rows, cols = full_topology.nonzero()
    keep_edges = rows != cols
    rows, cols = rows[keep_edges], cols[keep_edges]
    full_weights = np.linalg.norm(full_vertices[rows] - full_vertices[cols], axis=1)
    full_graph = sparse.csr_matrix((full_weights, (rows, cols)), shape=full_topology.shape)
    bem = mne.read_bem_solution(bem_path, verbose=False)
    forward = mne.make_forward_solution(raw, trans=trans, src=src, bem=bem,
                                       meg=True, eeg=False, mindist=5., n_jobs=1,
                                       verbose=False)
    forward = mne.convert_forward_solution(forward, surf_ori=True, force_fixed=True,
                                           copy=True, verbose=False)
    forward = mne.pick_types_forward(forward, meg='mag', eeg=False)
    src = forward['src']
    vertices = np.concatenate([hemi['rr'][hemi['vertno']] for hemi in src]) * 1000.
    normals = forward['source_nn']
    hemisphere = np.concatenate([np.full(len(hemi['vertno']), i) for i, hemi in enumerate(src)])
    retained = np.concatenate([
        np.searchsorted(original_src[0]['vertno'], src[0]['vertno']),
        len(original_src[0]['vertno']) + np.searchsorted(original_src[1]['vertno'], src[1]['vertno'])])
    graph = restrict_surface_graph(full_graph, retained)
    geometry = CorticalGeometry(vertices, normals, graph, hemisphere)
    g = forward['sol']['data'].astype(np.float64)
    if g.shape != (102, 4699) or connected_components(graph, directed=False)[0] != 2:
        raise ValueError(f'Unexpected sample geometry: G={g.shape}, expected (102,4699) and two components')
    provenance = {'kind': 'mne_sample', 'mne_version': mne.__version__,
                  'spacing': 'ico4', 'mindist_mm': 5., 'orientation': 'fixed_surface_normal',
                  'coordinate_frame': int(forward['coord_frame']),
                  'units': 'mm', 'leadfield_column_normalization': False,
                  'graph_construction': 'original_ico4_min_plus_transit_elimination',
                  'original_mesh_vertices': len(full_vertices),
                  'retained_source_vertices': len(vertices),
                  'n_components': int(connected_components(graph, directed=False)[0]),
                  'channel_names': forward['sol']['row_names'],
                  'created_utc': datetime.now(timezone.utc).isoformat(),
                  'builder_sha256': sha256(__file__),
                  'input_sha256': {str(p): sha256(p) for p in
                                   [raw, trans, bem_path, *surface_paths] if p.exists()}}
    if args.reference_dataset:
        provenance['reference_geometry'] = validate_reference_geometry(
            geometry, args.reference_dataset, allow_added_edges=True)
    provenance['elapsed_seconds'] = time.monotonic() - start
    geometry.save_bundle(args.output, g, provenance)
    print(f'Exported {g.shape}, two hemispheres, bundle SHA256 {sha256(args.output)}', flush=True)


if __name__ == '__main__':
    main()
