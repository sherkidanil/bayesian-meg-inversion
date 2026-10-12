import numpy as np
import pytest

from megflow.m1_bank_selection import max_score_candidate


def test_max_score_candidate_returns_the_first_highest_scoring_aligned_draw():
    aligned = np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]])
    scores = np.array([0.2, 0.9, 0.9])

    selected = max_score_candidate(aligned, scores)

    np.testing.assert_array_equal(selected, [1, 2, 3])
    selected[0] = 99
    assert aligned[1, 0] == 1


@pytest.mark.parametrize(
    ("aligned", "scores"),
    [
        (np.ones((2, 3, 1)), np.ones(2)),
        (np.ones((2, 3)), np.ones(3)),
        (np.ones((2, 3)), np.array([0.1, np.nan])),
    ],
)
def test_max_score_candidate_rejects_invalid_bank(aligned, scores):
    with pytest.raises(ValueError):
        max_score_candidate(aligned, scores)
