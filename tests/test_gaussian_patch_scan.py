from pathlib import Path
import sys

import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_ROOT))

from gaussian_patch_scan import (
    build_gaussian_patch_dictionaries,
    greedy_patch_scan_covariance,
    scan_patch_covariance_batch,
)
from metrics import build_graph_from_edges
from residual_rescoring import ProjectedGaussianCache, subspace_residual_from_covariance


def test_patch_dictionaries_match_projected_gaussian_cache():
    leadfield = np.asarray(
        [
            [1.0, 0.5, -0.5, 0.25],
            [0.0, 1.0, 0.5, -0.25],
            [0.5, -0.5, 1.0, 0.75],
        ]
    )
    edge_index = np.asarray([[0, 1], [1, 2], [2, 3]])
    edge_weight = np.ones(3)
    graph = build_graph_from_edges(edge_index, edge_weight, n_vertices=4)
    params = {
        "R_mm": np.asarray([1.1, 2.1, 1.1]),
        "A0": np.asarray([1.0, 2.0, 1.0]),
        "sigma_mm_scale": 0.5,
    }

    dictionaries = build_gaussian_patch_dictionaries(leadfield, graph, params)
    cache = ProjectedGaussianCache(leadfield, graph, params)

    assert dictionaries.shape == (3, 3, 4)
    for slot in range(3):
        for vertex in range(4):
            np.testing.assert_allclose(
                dictionaries[slot, :, vertex],
                cache.column(source_slot=slot, centroid=vertex),
                rtol=1e-12,
                atol=1e-12,
            )


def _reference_greedy(covariance, dictionaries):
    remaining_slots = list(range(dictionaries.shape[0]))
    selected_slots = []
    selected_vertices = []
    used_vertices = set()
    for _ in range(dictionaries.shape[0]):
        best = None
        for slot in remaining_slots:
            for vertex in range(dictionaries.shape[2]):
                if vertex in used_vertices:
                    continue
                basis = np.column_stack(
                    [
                        dictionaries[s, :, v]
                        for s, v in zip(selected_slots, selected_vertices, strict=True)
                    ]
                    + [dictionaries[slot, :, vertex]]
                )
                residual = subspace_residual_from_covariance(basis, covariance)
                key = (residual, slot, vertex)
                if best is None or key < best:
                    best = key
        _, slot, vertex = best
        selected_slots.append(slot)
        selected_vertices.append(vertex)
        remaining_slots.remove(slot)
        used_vertices.add(vertex)
    basis = np.column_stack(
        [
            dictionaries[s, :, v]
            for s, v in zip(selected_slots, selected_vertices, strict=True)
        ]
    )
    return (
        np.asarray(selected_slots),
        np.asarray(selected_vertices),
        subspace_residual_from_covariance(basis, covariance),
    )


def test_vectorized_greedy_scan_matches_explicit_qr_search():
    rng = np.random.default_rng(4)
    dictionaries = rng.normal(size=(3, 5, 7))
    y = rng.normal(size=(5, 11))
    covariance = y @ y.T

    expected_slots, expected_vertices, expected_residual = _reference_greedy(
        covariance, dictionaries
    )
    result = greedy_patch_scan_covariance(covariance, dictionaries)

    np.testing.assert_array_equal(result.source_slots, expected_slots)
    np.testing.assert_array_equal(result.centroids, expected_vertices)
    assert len(np.unique(result.source_slots)) == 3
    assert len(np.unique(result.centroids)) == 3
    np.testing.assert_allclose(result.residual, expected_residual, rtol=1e-12, atol=1e-12)


def test_batch_scan_preserves_pair_order_and_matches_scalar_results():
    rng = np.random.default_rng(8)
    dictionaries = rng.normal(size=(3, 4, 6))
    y = rng.normal(size=(2, 2, 4, 9))
    covariances = np.einsum("abmt,abnt->abmn", y, y)

    batch = scan_patch_covariance_batch(covariances, dictionaries, workers=1)

    assert batch["centroids"].shape == (2, 2, 3)
    assert batch["source_slots"].shape == (2, 2, 3)
    assert batch["residuals"].shape == (2, 2)
    assert batch["runtime_seconds"].shape == (2, 2)
    for sample in range(2):
        for branch in range(2):
            scalar = greedy_patch_scan_covariance(
                covariances[sample, branch], dictionaries
            )
            np.testing.assert_array_equal(
                batch["centroids"][sample, branch], scalar.centroids
            )
            np.testing.assert_array_equal(
                batch["source_slots"][sample, branch], scalar.source_slots
            )
            assert batch["residuals"][sample, branch] == scalar.residual
            assert batch["runtime_seconds"][sample, branch] >= 0.0
