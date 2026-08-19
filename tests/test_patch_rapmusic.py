from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

from patch_rapmusic import (
    concatenate_patch_dictionaries,
    rap_music_fixed,
    rap_music_fixed_batch,
)


def _orthogonal_problem():
    dictionary = np.eye(6, dtype=np.float64)
    time_courses = np.asarray(
        [
            [1.0, 0.0, 1.0, -1.0, 0.5],
            [0.0, 2.0, -1.0, 0.0, 0.5],
            [1.0, -1.0, 0.0, 2.0, -0.5],
        ]
    )
    truth = np.asarray([0, 2, 4], dtype=np.int64)
    data = dictionary[:, truth] @ time_courses
    return data, dictionary, truth


def test_fixed_rap_music_recovers_exact_signal_dictionary_columns():
    data, dictionary, truth = _orthogonal_problem()

    result = rap_music_fixed(data, dictionary, n_sources=3, signal_rank=3)

    np.testing.assert_array_equal(np.sort(result.candidate_indices), truth)
    assert result.correlations.shape == (3,)
    assert np.all(result.correlations > 1.0 - 1e-12)


def test_fixed_rap_music_is_invariant_to_candidate_column_scaling():
    data, dictionary, truth = _orthogonal_problem()
    scales = np.asarray([0.2, 3.0, 0.7, 4.0, 2.5, 0.1])

    result = rap_music_fixed(
        data,
        dictionary * scales[None, :],
        n_sources=3,
        signal_rank=3,
    )

    np.testing.assert_array_equal(np.sort(result.candidate_indices), truth)


def test_fixed_rap_music_never_selects_one_vertex_through_two_candidates():
    dictionary = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    vertex_ids = np.asarray([7, 7, 8, 9])
    data = dictionary[:, [0, 2]] @ np.asarray([[1.0, 0.5], [0.0, 1.0]])

    result = rap_music_fixed(
        data,
        dictionary,
        n_sources=2,
        signal_rank=2,
        candidate_vertex_ids=vertex_ids,
    )

    selected_vertices = vertex_ids[result.candidate_indices]
    assert len(np.unique(selected_vertices)) == 2
    np.testing.assert_array_equal(np.sort(selected_vertices), [7, 8])


def test_patch_dictionary_concatenation_preserves_vertex_and_radius_mapping():
    dictionaries = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)

    combined, vertices, radius_slots = concatenate_patch_dictionaries(dictionaries)

    assert combined.shape == (3, 8)
    np.testing.assert_array_equal(combined[:, :4], dictionaries[0])
    np.testing.assert_array_equal(combined[:, 4:], dictionaries[1])
    np.testing.assert_array_equal(vertices, [0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(radius_slots, [0, 0, 0, 0, 1, 1, 1, 1])


def test_fixed_rap_music_batch_preserves_sample_and_branch_order():
    data, dictionary, truth = _orthogonal_problem()
    batch = np.stack([[data, 2.0 * data], [0.5 * data, -data]])

    result = rap_music_fixed_batch(
        batch,
        dictionary,
        n_sources=3,
        signal_rank=3,
        workers=1,
    )

    assert result["candidate_indices"].shape == (2, 2, 3)
    assert result["correlations"].shape == (2, 2, 3)
    assert result["runtime_seconds"].shape == (2, 2)
    for sample in range(2):
        for branch in range(2):
            np.testing.assert_array_equal(
                np.sort(result["candidate_indices"][sample, branch]), truth
            )
            assert result["runtime_seconds"][sample, branch] >= 0.0


def test_fixed_rap_music_matches_scalar_mne_rap_projection_rule():
    rng = np.random.default_rng(6613)
    dictionary = rng.normal(size=(8, 20))
    data = (
        dictionary[:, [1, 5, 9]] @ rng.normal(size=(3, 20))
        + 0.3 * rng.normal(size=(8, 20))
    )

    result = rap_music_fixed(data, dictionary, n_sources=3, signal_rank=3)

    # Independent scalar specialization of MNE's _apply_rap_music gives this
    # sequence. In RAP (unlike TRAP), P_perp E_s is not rank-truncated.
    np.testing.assert_array_equal(result.candidate_indices, [9, 5, 3])
