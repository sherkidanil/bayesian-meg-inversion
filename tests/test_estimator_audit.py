import inspect
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from estimator_audit import (
    align_draws_to_reference,
    posterior_mean_aligned,
    posterior_mean_naive,
    posterior_medoid,
)


def _base_draw():
    return np.array(
        [
            [0.000, 0.000, 0.000],
            [0.010, 0.000, 0.000],
            [0.020, 0.000, 0.000],
        ],
        dtype=np.float64,
    )


def test_align_draws_to_reference_recovers_continuous_source_order():
    reference = _base_draw()
    permuted = reference[[2, 0, 1]] + np.array([0.0001, 0.0, 0.0])
    draws = np.stack([reference, permuted])

    aligned, permutations, nonidentity_fraction = align_draws_to_reference(draws)

    assert np.array_equal(permutations[0], [0, 1, 2])
    assert np.array_equal(permutations[1], [1, 2, 0])
    assert np.allclose(aligned[1], reference + np.array([0.0001, 0.0, 0.0]))
    assert nonidentity_fraction == 1.0


def test_aligned_mean_avoids_slot_collapse_from_cyclic_permutations():
    reference = _base_draw()
    draws = np.stack([reference, reference[[1, 2, 0]], reference[[2, 0, 1]]])

    naive = posterior_mean_naive(draws)
    aligned, nonidentity_fraction = posterior_mean_aligned(draws)

    assert np.allclose(naive[:, 0], [0.010, 0.010, 0.010])
    assert np.allclose(aligned, reference)
    assert nonidentity_fraction == 1.0


def test_posterior_medoid_is_permutation_invariant_and_rejects_outlier():
    reference = _base_draw()
    same_set_permuted = reference[[2, 0, 1]]
    outlier = reference + np.array([0.100, 0.0, 0.0])
    draws = np.stack([reference, same_set_permuted, outlier])

    medoid = posterior_medoid(draws)

    assert np.array_equal(medoid, reference)


def test_posterior_medoid_api_cannot_receive_ground_truth():
    parameters = inspect.signature(posterior_medoid).parameters

    assert list(parameters) == ["theta_draws"]
    assert all("true" not in name.lower() for name in parameters)
