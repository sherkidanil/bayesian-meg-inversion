import numpy as np
import pytest

from megflow.simulator import (PlacementError, sample_centroids, simulate,
                               random_time_courses)
from megflow.decoder import coordinate_ascent
from megflow.forward import column_basis, subspace_score
from megflow.metrics import localization_errors, ospa


def test_separation_sampling_never_relaxes_constraint(geometry):
    with pytest.raises(PlacementError) as error:
        sample_centroids(geometry, 3, 30., np.random.default_rng(3), max_attempts=7)
    assert error.value.attempts == 7
    centers, attempts = sample_centroids(geometry, 2, 30., np.random.default_rng(3))
    assert attempts >= 1
    assert np.isinf(geometry.geodesic(*centers))


def test_random_time_courses_seeded_and_normalized():
    cfg = {'n_samples': 1800, 'sfreq': 1000., 'band_hz': [2., 40.],
           'bandwidth_hz': [2., 6.], 'window_dur_s': [.3, 1.2],
           'correlation': {'mode': 'pairwise', 'rho_range': [0, .95],
                           'p_correlated_pair': 1.}}
    a = random_time_courses(5, cfg, np.random.default_rng(3))
    b = random_time_courses(5, cfg, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a.std(axis=1), 1., atol=1e-12)
    assert np.linalg.matrix_rank(a) == 5


def test_simulation_seed_and_noise_scaling(geometry):
    g = np.random.default_rng(8).normal(size=(10, 6))
    cfg = {'min_separation_mm': 3., 'radius_mm': 1., 'sigma_ratio': 1/3,
           'snr': 3.333, 'time_courses': {'n_samples': 1800, 'sfreq': 1000.,
             'band_hz': [2., 40.], 'bandwidth_hz': [2., 6.],
             'window_dur_s': [.3, 1.2], 'correlation': {'mode': 'none'}}}
    a, b = [simulate(g, geometry, k=2, seed=19, config=cfg) for _ in range(2)]
    np.testing.assert_array_equal(a.Y, b.Y)
    np.testing.assert_array_equal(a.centroids, b.centroids)
    assert a.Y.dtype == np.float32
    assert a.Y_clean.std() == pytest.approx(1, abs=1e-6)
    assert (a.Y - a.Y_clean).std() == pytest.approx(1 / cfg['snr'], rel=.03)


def test_ascent_finds_neighbor_and_respects_occupied_vertices(geometry):
    dictionary = np.array([[1., 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1],
                           [0, 0, 0, 0, 0, 0]])
    e = np.array([[0.], [1.], [0.]])
    result = coordinate_ascent([0], dictionary, e, geometry, min_separation_mm=0)
    assert result.centroids.tolist() == [1]
    assert np.all(np.diff(result.score_history) > 0)
    assert result.score == pytest.approx(1.)
    two = coordinate_ascent([0, 1], dictionary, np.eye(3)[:, :2], geometry,
                            min_separation_mm=2)
    assert len(set(two.centroids)) == 2
    assert geometry.geodesic(*two.centroids) >= 2


def test_noiseless_truth_cannot_improve(geometry):
    dictionary = np.random.default_rng(32).normal(size=(10, 6))
    truth = np.array([0, 3])
    e = column_basis(dictionary[:, truth])
    result = coordinate_ascent(truth, dictionary, e, geometry, min_separation_mm=3)
    np.testing.assert_array_equal(result.centroids, truth)
    assert result.score == pytest.approx(1., abs=1e-12)


def test_ascent_matches_exhaustive_one_slot_search(geometry):
    dictionary = np.random.default_rng(5).normal(size=(8, 6))
    e = column_basis(np.random.default_rng(10).normal(size=(8, 3)))
    result = coordinate_ascent([1], dictionary, e, geometry, n_passes=1,
                               min_separation_mm=0)
    scores = [subspace_score(e, dictionary[:, [i]]) for i in [0, 1, 2]]
    assert result.score == pytest.approx(max(scores), abs=1e-12)


def test_assignment_and_ospa_preserve_infinite_misses(geometry):
    errors = localization_errors([0, 3], [4, 1], geometry)
    np.testing.assert_equal(errors, [2, 2])
    assert np.isinf(localization_errors([0], [3], geometry)[0])
    assert ospa([0], [3], geometry) == 30
    assert ospa([0, 3], [0], geometry) == 15
    assert ospa([], [], geometry) == 0
