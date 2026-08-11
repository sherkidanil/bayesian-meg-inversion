from pathlib import Path
import sys

import numpy as np
import pytest

MODULE_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "tests"
    else Path(__file__).resolve().parent
)
sys.path.insert(0, str(MODULE_ROOT))

from historical_leadfield_provenance import (
    classify_paired_preference,
    configure_project_import_path,
    historical_dependency,
    read_npy_prefix_from_npz,
)


def test_npz_prefix_reader_returns_requested_rows(tmp_path):
    path = tmp_path / "large.npz"
    values = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
    np.savez_compressed(path, Ys=values)

    loaded, full_shape = read_npy_prefix_from_npz(path, "Ys", 2)

    assert full_shape == (5, 3, 4)
    assert np.array_equal(loaded, values[:2])


def test_paired_preference_requires_calibration_and_decisive_fraction():
    differences = np.r_[-np.ones(298), np.ones(2)]

    result = classify_paired_preference(
        differences, anna_calibration_fraction=313 / 315
    )

    assert result["classification"] == "rebuilt_like"
    assert result["fraction_rebuilt_lower"] == pytest.approx(298 / 300)
    assert classify_paired_preference(
        differences, anna_calibration_fraction=0.70
    )["classification"] == "calibration_failed"
    assert classify_paired_preference(
        np.r_[-np.ones(55), np.ones(45)], anna_calibration_fraction=0.99
    )["classification"] == "unidentified"


def test_historical_dependency_maps_named_project_numbers():
    assert "4.589" in historical_dependency(
        "meg_flow_dataset_10000_new_leadfield_error.npz"
    )
    assert "three-model" in historical_dependency("test_dataset_good_idxs_mesh.npz")
    assert "training" in historical_dependency("meg_flow_dataset_10000_new.npz")


def test_standalone_cli_adds_worktree_root_to_import_path():
    search_path = ["existing"]

    root = configure_project_import_path(
        "/repo/new_version_mesh_experiments/historical_leadfield_provenance.py",
        search_path,
    )

    assert root == Path("/repo")
    assert search_path[0] == "/repo"
