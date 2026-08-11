#!/usr/bin/env python3
"""Hash-verified, recoverable switching of the canonical best-model alias."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable


ARTIFACT_NAMES = ("model.eqx", "stats.npz", "result.json")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_hashes(
    directory: str | Path, names: Iterable[str] = ARTIFACT_NAMES
) -> dict[str, str]:
    root = Path(directory)
    return {name: sha256_file(root / name) for name in names}


def assert_artifacts_match(
    actual: str | Path,
    expected: str | Path,
    names: Iterable[str] = ARTIFACT_NAMES,
) -> dict[str, str]:
    actual_hashes = artifact_hashes(actual, names)
    expected_hashes = artifact_hashes(expected, names)
    if actual_hashes != expected_hashes:
        raise ValueError(
            "artifact hash mismatch: "
            + json.dumps({"actual": actual_hashes, "expected": expected_hashes})
        )
    return actual_hashes


def switch_alias_atomically(
    alias_dir: str | Path,
    source_dir: str | Path,
    archive_dir: str | Path,
    names: Iterable[str] = ARTIFACT_NAMES,
) -> dict[str, dict[str, str]]:
    """Archive the current alias, then atomically replace each verified artifact."""
    alias = Path(alias_dir)
    source = Path(source_dir)
    archive = Path(archive_dir)
    selected_names = tuple(names)
    for name in selected_names:
        if not (alias / name).is_file():
            raise FileNotFoundError(alias / name)
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    archive.mkdir(parents=True, exist_ok=True)
    previous = artifact_hashes(alias, selected_names)
    for name in selected_names:
        destination = archive / name
        if destination.exists() and sha256_file(destination) != previous[name]:
            raise ValueError(f"archive already contains a different {name}")
        if not destination.exists():
            shutil.copy2(alias / name, destination)
    if artifact_hashes(archive, selected_names) != previous:
        raise RuntimeError("archived alias failed hash verification")

    with tempfile.TemporaryDirectory(prefix="best-model-switch-", dir=alias) as tmp:
        staging = Path(tmp)
        for name in selected_names:
            shutil.copy2(source / name, staging / name)
            if sha256_file(staging / name) != sha256_file(source / name):
                raise RuntimeError(f"staging copy failed verification for {name}")
        for name in selected_names:
            os.replace(staging / name, alias / name)

    current = assert_artifacts_match(alias, source, selected_names)
    return {"previous": previous, "current": current}

