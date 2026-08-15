from pathlib import Path
import inspect
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from residual_rescoring import (
    ProjectedGaussianCache,
    butterworth_bandpass_sensor_data,
    gaussian_profile,
    orthonormal_column_basis,
    residual_gibbs_weights,
    residual_rescore_draws,
    residual_weighted_mean,
    subspace_residual,
    subspace_residual_from_covariance,
)


def test_butterworth_bandpass_matches_mne_iir_reference_padding():
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(301)
    y = rng.normal(size=(3, 1800))
    sos = butter(4, [2.0, 25.0], btype="bandpass", fs=1000.0, output="sos")
    expected = sosfiltfilt(sos, y, axis=-1, padlen=1266)

    actual = butterworth_bandpass_sensor_data(
        y, sfreq_hz=1000.0, low_hz=2.0, high_hz=25.0, order=4
    )

    np.testing.assert_array_equal(actual, expected)


def test_bandpass_preserves_candidate_sensor_subspace():
    times = np.arange(1800, dtype=np.float64) / 1000.0
    basis = np.array([[1.0, 0.2], [0.0, 1.0], [0.5, -0.3]])
    temporal = np.stack(
        [np.sin(2.0 * np.pi * 10.0 * times), np.cos(2.0 * np.pi * 18.0 * times)]
    )
    y = basis @ temporal

    filtered = butterworth_bandpass_sensor_data(y, sfreq_hz=1000.0)

    assert subspace_residual(basis, filtered) == pytest.approx(0.0, abs=2e-8)


def test_bandpass_rejects_cutoff_at_or_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        butterworth_bandpass_sensor_data(
            np.ones((2, 100)), sfreq_hz=100.0, low_hz=2.0, high_hz=50.0
        )


def test_residual_is_zero_when_y_lies_in_candidate_subspace():
    basis = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    temporal = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = basis @ temporal

    assert subspace_residual(basis, y) == pytest.approx(0.0, abs=1e-12)


def test_residual_detects_fully_orthogonal_sensor_energy():
    basis = np.array([[1.0], [0.0], [0.0]])
    y = np.array([[0.0], [3.0], [4.0]])

    assert subspace_residual(basis, y) == pytest.approx(1.0)


def test_covariance_residual_matches_direct_projection():
    rng = np.random.default_rng(7)
    basis = rng.normal(size=(8, 3))
    y = rng.normal(size=(8, 20))
    covariance = y @ y.T

    assert subspace_residual_from_covariance(basis, covariance) == pytest.approx(
        subspace_residual(basis, y), abs=1e-12
    )


def test_residual_is_invariant_to_global_and_right_diagonal_scaling():
    rng = np.random.default_rng(11)
    basis = rng.normal(size=(7, 3))
    y = rng.normal(size=(7, 15))
    expected = subspace_residual(basis, y)

    assert subspace_residual(-17.0 * basis, y) == pytest.approx(expected, abs=1e-12)
    assert subspace_residual(basis @ np.diag([0.2, -3.0, 9.0]), y) == pytest.approx(
        expected, abs=1e-12
    )


def test_rank_zero_basis_explains_no_energy():
    y = np.eye(3)

    assert subspace_residual(np.zeros((3, 2)), y) == pytest.approx(1.0)


def test_gaussian_profile_matches_simulator_rule():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 2.0, 0.0],
            ]
        )
    )
    profile = gaussian_profile(
        centroid=1,
        source_slot=0,
        graph=graph,
        n_vertices=4,
        R_mm=np.array([1.5]),
        sigma_mm_scale=0.5,
        A0=np.array([2.0]),
    )
    edge_weight = 2.0 * np.exp(-0.5 * (1.0 / 0.75) ** 2)

    assert np.allclose(profile, [edge_weight, 2.0, edge_weight, 0.0])


def test_column_normalizing_g_can_change_gaussian_patch_projector():
    leadfield = np.diag([2.0, 1.0, 0.5])
    profiles = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    original = leadfield @ profiles.T
    normalized_g = leadfield / np.linalg.norm(leadfield, axis=0, keepdims=True)
    normalized = normalized_g @ profiles.T
    q_original = orthonormal_column_basis(original)
    q_normalized = orthonormal_column_basis(normalized)

    assert not np.allclose(q_original @ q_original.T, q_normalized @ q_normalized.T)


def test_projected_gaussian_cache_reuses_slot_centroid_column():
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    leadfield = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    params = {
        "R_mm": np.array([1.1]),
        "sigma_mm_scale": 0.5,
        "A0": np.array([1.0]),
    }
    cache = ProjectedGaussianCache(leadfield, graph, params)

    first = cache.column(source_slot=0, centroid=1)
    second = cache.column(source_slot=0, centroid=1)

    assert np.array_equal(first, second)
    assert cache.cache_size == 1


def test_residual_rescore_selects_draw_whose_subspace_generated_y():
    vertices_m = np.array(
        [[0.000, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [0.003, 0.0, 0.0]]
    )
    graph = csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
    )
    theta_draws = np.array(
        [
            [[0.000, 0.0, 0.0]],
            [[0.002, 0.0, 0.0]],
        ]
    )
    localization_g = np.eye(4)
    y = localization_g[:, [2]] @ np.array([[1.0, -2.0, 3.0]])
    params = {
        "R_mm": np.array([0.1]),
        "sigma_mm_scale": 0.5,
        "A0": np.array([1.0]),
    }

    residuals, selected = residual_rescore_draws(
        theta_draws, y, localization_g, vertices_m, graph, params
    )

    assert selected == 1
    assert residuals[1] == pytest.approx(0.0, abs=1e-12)
    assert residuals[0] == pytest.approx(1.0)


def test_residual_rescore_public_api_has_no_truth_argument():
    parameters = tuple(inspect.signature(residual_rescore_draws).parameters)

    assert parameters == (
        "theta_draws",
        "Y",
        "localization_G",
        "vertices_m",
        "graph",
        "params_spatial",
    )
    assert not {"centroids_idx", "true_centroids", "true_thetas", "G_for_simulation"} & set(
        parameters
    )


@pytest.mark.parametrize("score_form", ["quadratic", "linear"])
def test_gibbs_weights_are_stable_finite_and_sum_to_one(score_form):
    residuals = np.array([0.1, 0.2, 0.9, 1.0])

    weights = residual_gibbs_weights(residuals, tau=1e-4, score_form=score_form)

    assert weights.shape == residuals.shape
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.argmax(weights) == 0


@pytest.mark.parametrize("tau", [0.0, -1.0, np.inf, np.nan])
def test_gibbs_weights_reject_invalid_temperature(tau):
    with pytest.raises(ValueError, match="tau"):
        residual_gibbs_weights(np.array([0.1, 0.2]), tau=tau)


def test_quadratic_temperature_limits_recover_argmin_and_unweighted_mean():
    residuals = np.array([0.1, 0.2, 0.3])

    cold = residual_gibbs_weights(residuals, tau=1e-8, score_form="quadratic")
    hot = residual_gibbs_weights(residuals, tau=1e12, score_form="quadratic")

    assert np.allclose(cold, [1.0, 0.0, 0.0])
    assert np.allclose(hot, np.full(3, 1.0 / 3.0), atol=1e-14)


def test_residual_weighted_mean_uses_draw_axis_and_returns_weights():
    draws = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]
    )
    residuals = np.array([0.1, 0.1])

    estimate, weights = residual_weighted_mean(
        draws, residuals, tau=0.5, score_form="quadratic"
    )

    assert np.allclose(weights, [0.5, 0.5])
    assert np.allclose(estimate, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_smooth_decoder_public_api_has_no_truth_argument():
    for function in (residual_gibbs_weights, residual_weighted_mean):
        parameters = set(inspect.signature(function).parameters)
        assert not {
            "centroids_idx",
            "true_centroids",
            "true_thetas",
            "G_for_simulation",
        } & parameters
