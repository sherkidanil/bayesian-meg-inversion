from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent if HERE.name == "tests" else HERE
sys.path.insert(0, str(MODULE_ROOT))

from best_model_alias import artifact_hashes, assert_artifacts_match, switch_alias_atomically


def _write_bundle(directory: Path, prefix: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("model.eqx", "stats.npz", "result.json"):
        (directory / name).write_bytes(f"{prefix}:{name}".encode())


def test_switch_archives_previous_and_resolves_exactly_to_source(tmp_path):
    alias = tmp_path / "alias"
    source = tmp_path / "honest"
    archive = tmp_path / "archive" / "push"
    _write_bundle(alias, "push")
    _write_bundle(source, "honest")
    previous = artifact_hashes(alias)

    result = switch_alias_atomically(alias, source, archive)

    assert result["previous"] == previous
    assert artifact_hashes(archive) == previous
    assert result["current"] == artifact_hashes(source)
    assert_artifacts_match(alias, source)


def test_switch_refuses_to_overwrite_conflicting_archive(tmp_path):
    alias = tmp_path / "alias"
    source = tmp_path / "honest"
    archive = tmp_path / "archive" / "push"
    _write_bundle(alias, "push")
    _write_bundle(source, "honest")
    _write_bundle(archive, "different")

    with pytest.raises(ValueError, match="archive already contains a different"):
        switch_alias_atomically(alias, source, archive)

    assert (alias / "model.eqx").read_bytes() == b"push:model.eqx"
