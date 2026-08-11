from pathlib import Path
import sys

import numpy as np
import pytest

TEST_ROOT = Path(__file__).resolve().parent
if TEST_ROOT.name == "tests":
    TEST_ROOT = TEST_ROOT.parent
sys.path.insert(0, str(TEST_ROOT))

from residual_rescoring_dataset import load_dataset_and_leadfield, paired_comparisons


def _write_dataset(path: Path, vertices: np.ndarray) -> None:
    np.savez(
        path,
        Ys=np.zeros((2, 4, 6), dtype=np.float32),
        centroids_idx=np.array([[0, 1, 2], [1, 2, 0]]),
        vertices=vertices,
        edge_index=np.array([[0, 1], [1, 2]]),
        edge_weight_mm=np.array([1.0, 1.0]),
        params_spatial=np.array({"R_mm": [10, 8, 10]}, dtype=object),
    )


def test_load_dataset_and_leadfield_requires_identical_mesh(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    leadfield_path = tmp_path / "leadfield.npz"
    vertices = np.eye(3)
    _write_dataset(dataset_path, vertices)
    np.savez(leadfield_path, G=np.zeros((4, 3)), vertices=vertices + 1.0)

    with pytest.raises(ValueError, match="vertices"):
        load_dataset_and_leadfield(dataset_path, leadfield_path)


def test_load_dataset_and_leadfield_returns_explicit_g(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    leadfield_path = tmp_path / "leadfield.npz"
    vertices = np.eye(3)
    leadfield = np.arange(12).reshape(4, 3)
    _write_dataset(dataset_path, vertices)
    np.savez(leadfield_path, G=leadfield, vertices=vertices)

    loaded = load_dataset_and_leadfield(dataset_path, leadfield_path)

    assert np.array_equal(loaded["localization_G"], leadfield)
    assert loaded["dataset"]["Ys"].shape == (2, 4, 6)


def test_paired_comparisons_use_same_repeat_and_report_wins():
    runs = [
        {"model_key": "a", "repeat": 1, "data_seed": 10, "mae_geo_mm": 4.0},
        {"model_key": "a", "repeat": 2, "data_seed": 11, "mae_geo_mm": 8.0},
        {"model_key": "b", "repeat": 1, "data_seed": 10, "mae_geo_mm": 5.0},
        {"model_key": "b", "repeat": 2, "data_seed": 11, "mae_geo_mm": 7.0},
    ]

    result = paired_comparisons(runs, metric="mae_geo_mm")

    assert result == [
        {
            "model_a": "a",
            "model_b": "b",
            "differences_a_minus_b": [-1.0, 1.0],
            "mean_difference_a_minus_b": 0.0,
            "a_wins": 1,
            "b_wins": 1,
            "ties": 0,
            "data_seeds": [10, 11],
        }
    ]
