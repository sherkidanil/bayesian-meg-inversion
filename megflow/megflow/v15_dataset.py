"""Independent variable-K shards and memory-mapped pooled observations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import json
from importlib.metadata import PackageNotFoundError, version
import multiprocessing as mp
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from omegaconf import OmegaConf
from scipy.sparse.csgraph import dijkstra

from .dataset import SPLITS, digest_json, record_seed, sha256, source_hash, write_json
from .geometry import CorticalGeometry
from .v15 import simulate, validate_windows

FIELDS = (
    "theta",
    "centers",
    "radii",
    "band_idx",
    "t_start",
    "t_stop",
    "snr_per_source",
    "mask",
    "K",
    "seed",
    "placement_attempts",
    "leadfield_error_mean",
)


def runtime_versions():
    try:
        mne_version = version("mne")
    except PackageNotFoundError:
        mne_version = (
            None  # Unit tests can exercise unfiltered array storage without MNE.
        )
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        mne=mne_version,
    )


def validate_labels(data, config):
    mask, k = data["mask"], data["K"]
    c = config["simulation"]
    if (
        mask.dtype != bool
        or mask.shape != (len(k), c["k_max"])
        or not np.array_equal(mask, np.arange(c["k_max"])[None, :] < k[:, None])
        or (k < c["k_min"]).any()
        or (k > c["k_max"]).any()
    ):
        raise ValueError("invalid K or source mask")
    if data["theta"].shape != (*mask.shape, 3):
        raise ValueError("invalid theta shape")
    for name in ["theta", "radii", "snr_per_source"]:
        if (
            not np.isfinite(data[name][mask]).all()
            or not np.isnan(data[name][~mask]).all()
        ):
            raise ValueError(f"invalid active values or padding in {name}")
    for name in ["centers", "band_idx", "t_start", "t_stop"]:
        if (data[name][mask] < 0).any() or (data[name][~mask] != -1).any():
            raise ValueError(f"invalid integer labels: {name}")
    if (
        (data["radii"][mask] < c["radius_min"]).any()
        or (data["radii"][mask] > c["radius_max"]).any()
        or (data["t_stop"][mask] > c["n_samples"]).any()
        or not ((data["t_stop"] - data["t_start"])[mask] == c["duration_samples"]).all()
    ):
        raise ValueError("radii or activation windows outside configured support")


def write_split(
    records, directory, split, config, anatomy_hash, code_hash, *, require_filtered=True
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    count = config["counts"][split]
    pool = config["pool"]
    c = config["simulation"]
    if c["n_samples"] % pool:
        raise ValueError("pool must divide recording length")
    versions = runtime_versions()
    fingerprint = digest_json(
        dict(
            config=config,
            anatomy_sha256=anatomy_hash,
            code_sha256=code_hash,
            split=split,
            versions=versions,
        )
    )
    manifest = dict(
        format_version=2,
        split=split,
        fingerprint=fingerprint,
        config=config,
        anatomy_sha256=anatomy_hash,
        code_sha256=code_hash,
        n_records=count,
        units="mm",
        created_utc=datetime.now(timezone.utc).isoformat(),
        shards=[],
        versions=versions,
        filtered=require_filtered,
    )
    labels = {key: [] for key in FIELDS}
    states = []
    identities = []
    geometries = []
    chunk = []
    pooled = None
    offset = 0

    def flush():
        nonlocal offset
        if not chunk:
            return
        y = np.stack(chunk)
        stop = offset + len(y)
        pooled[offset:stop] = y.reshape(*y.shape[:-1], -1, pool).mean(-1)
        path = (
            directory / f'{split}-{fingerprint[:16]}-{len(manifest["shards"]):04d}.npz'
        )
        np.savez(
            path, Y=y, seed=np.asarray(labels["seed"][offset:stop], dtype="uint64")
        )
        manifest["shards"].append(
            dict(
                file=path.name,
                sha256=sha256(path),
                n_records=len(y),
                start=offset,
                stop=stop,
            )
        )
        offset = stop
        chunk.clear()
        print(
            json.dumps(
                dict(
                    split=split,
                    generated=offset,
                    seconds=round(time.monotonic() - start, 1),
                )
            ),
            flush=True,
        )

    for i, r in enumerate(records):
        if i >= count:
            raise ValueError("more records than configured")
        if require_filtered and r["filtered"] is not True:
            raise ValueError("unfiltered record")
        if (
            r["Y"].ndim != 2
            or r["Y"].shape[-1] != c["n_samples"]
            or not np.isfinite(r["Y"]).all()
        ):
            raise ValueError("invalid sensor recording")
        if pooled is None:
            pooled = np.lib.format.open_memmap(
                directory / "Y_pool.npy",
                mode="w+",
                dtype="float32",
                shape=(count, r["Y"].shape[0], c["n_samples"] // pool),
            )
        for key in FIELDS:
            labels[key].append(r[key])
        states.append(json.dumps(r["leadfield_rng_state"], sort_keys=True))
        gid = digest_json(
            dict(
                centers=r["centers"][r["mask"]].tolist(),
                radii=r["radii"][r["mask"]].tolist(),
            )
        )
        geometries.append(gid)
        identities.append(digest_json(dict(seed=r["seed"], geometry_id=gid)))
        chunk.append(r["Y"])
        if len(chunk) == config["shard_size"]:
            flush()
    flush()
    if offset != count or not count:
        raise ValueError("fewer records than configured")
    pooled.flush()
    del pooled
    data = {
        key: np.asarray(values, dtype="uint64" if key == "seed" else None)
        for key, values in labels.items()
    }
    validate_labels(data, config)
    data["leadfield_rng_state"] = np.asarray(states)
    np.savez(directory / "labels.npz", **data)
    manifest.update(
        record_ids=identities,
        geometry_ids=geometries,
        seeds=data["seed"].tolist(),
        files={name: sha256(directory / name) for name in ["Y_pool.npy", "labels.npz"]},
        k_counts={
            str(k): int(np.sum(data["K"] == k))
            for k in range(c["k_min"], c["k_max"] + 1)
        },
        placement_max_attempts=int(data["placement_attempts"].max()),
        elapsed_seconds=time.monotonic() - start,
    )
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_training_arrays(directory):
    directory = Path(directory)
    m = json.loads((directory / "manifest.json").read_text())
    if m["split"] not in ("train", "val"):
        raise ValueError("trainer can only load train or val")
    for name, digest in m["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"dataset checksum mismatch: {name}")
    with np.load(directory / "labels.npz", allow_pickle=False) as f:
        data = {key: f[key] for key in ["theta", "radii", "mask", "K"]}
    data.update(
        Y=np.load(directory / "Y_pool.npy", mmap_mode="r", allow_pickle=False),
        manifest=m,
    )
    return data


def load_evaluation_arrays(directory):
    """Load a frozen dev/heldout split without relaxing the training guard."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["split"] not in ("dev", "heldout"):
        raise ValueError("evaluator can only load dev or heldout")
    expected_fingerprint = digest_json(
        dict(
            config=manifest["config"],
            anatomy_sha256=manifest["anatomy_sha256"],
            code_sha256=manifest["code_sha256"],
            split=manifest["split"],
            versions=manifest["versions"],
        )
    )
    if (
        manifest.get("format_version") != 2
        or manifest.get("fingerprint") != expected_fingerprint
    ):
        raise ValueError("invalid evaluation manifest fingerprint")
    audit_path = directory.parent / "audit.json"
    if not audit_path.is_file():
        raise ValueError("evaluation requires the root dataset audit")
    audit = json.loads(audit_path.read_text())
    if audit.get("manifest_sha256", {}).get(manifest["split"]) != sha256(
        directory / "manifest.json"
    ):
        raise ValueError("evaluation manifest is not frozen by the root audit")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"dataset checksum mismatch: {name}")
    with np.load(directory / "labels.npz", allow_pickle=False) as labels:
        data = {
            key: labels[key]
            for key in ["theta", "radii", "centers", "mask", "K"]
        }
    data.update(
        Y=np.load(directory / "Y_pool.npy", mmap_mode="r", allow_pickle=False),
        manifest=manifest,
    )
    return data


def audit_splits(root):
    root = Path(root)
    manifests = {
        s: json.loads((root / s / "manifest.json").read_text()) for s in SPLITS
    }
    for split, m in manifests.items():
        expected = digest_json(
            dict(
                config=m["config"],
                anatomy_sha256=m["anatomy_sha256"],
                code_sha256=m["code_sha256"],
                split=split,
                versions=m["versions"],
            )
        )
        if (
            m["format_version"] != 2
            or m["split"] != split
            or m["fingerprint"] != expected
        ):
            raise ValueError("invalid split fingerprint")
        if m["n_records"] != m["config"]["counts"][split] or m["n_records"] != sum(
            s["n_records"] for s in m["shards"]
        ):
            raise ValueError("split count mismatch")
        for key in ["record_ids", "geometry_ids", "seeds"]:
            if len(m[key]) != m["n_records"] or len(set(m[key])) != m["n_records"]:
                raise ValueError(f"duplicate or missing {key}")
        files = dict(m["files"], **{s["file"]: s["sha256"] for s in m["shards"]})
        for name, digest in files.items():
            if sha256(root / split / name) != digest:
                raise ValueError(f"dataset checksum mismatch: {split}/{name}")
        with np.load(root / split / "labels.npz", allow_pickle=False) as labels:
            validate_labels(labels, m["config"])
            if labels["seed"].tolist() != m["seeds"]:
                raise ValueError("label seeds do not match manifest")
        pooled = np.load(root / split / "Y_pool.npy", mmap_mode="r", allow_pickle=False)
        if (
            pooled.shape[0] != m["n_records"]
            or pooled.shape[-1]
            != m["config"]["simulation"]["n_samples"] // m["config"]["pool"]
        ):
            raise ValueError("pooled shape mismatch")
        for start in range(0, len(pooled), 250):
            if not np.isfinite(pooled[start : start + 250]).all():
                raise ValueError("nonfinite observations")
    for key in ["config", "code_sha256", "anatomy_sha256", "filtered", "versions"]:
        if len({digest_json(m[key]) for m in manifests.values()}) != 1:
            raise ValueError(f"split {key} mismatch")
    for key in ["record_ids", "geometry_ids", "seeds"]:
        for i, a in enumerate(SPLITS):
            for b in list(SPLITS)[i + 1 :]:
                if set(manifests[a][key]) & set(manifests[b][key]):
                    raise ValueError(f"split overlap in {key}")
    report = dict(
        counts={s: m["n_records"] for s, m in manifests.items()},
        disjoint_seeds=True,
        disjoint_record_ids=True,
        disjoint_geometry_ids=True,
        shard_checksums_verified=True,
        simulator_and_config_match=True,
        manifest_sha256={s: sha256(root / s / "manifest.json") for s in SPLITS},
        filtered=next(iter(manifests.values()))["filtered"],
    )
    write_json(root / "audit.json", report)
    return report


def cache_distances_float64(geometry, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"v15-float64-{geometry.fingerprint}.npy"
    checksum = path.with_suffix(".sha256")
    if not path.exists():
        distance = dijkstra(geometry.adjacency, directed=False)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, distance)
        temporary.replace(path)
        checksum.write_text(sha256(path) + "\n")
    if not checksum.exists() or checksum.read_text().strip() != sha256(path):
        raise ValueError("distance cache checksum mismatch")
    distance = np.load(path, mmap_mode="r", allow_pickle=False)
    if distance.dtype != np.float64 or distance.shape != (len(geometry.vertices),) * 2:
        raise ValueError("distance cache shape/dtype")
    geometry._distances = distance


_WORKER = None


def _init_worker(anatomy, cache, config):
    global _WORKER
    if sha256(anatomy) != config["anatomy_sha256"]:
        raise ValueError("anatomy checksum mismatch")
    geom, g, provenance = CorticalGeometry.load_bundle(
        anatomy, allow_fragmented_hemispheres=True
    )
    if provenance["graph_construction"] != "supplied_v15_cropped_oct6":
        raise ValueError("expected new oct6 anatomy")
    cache_distances_float64(geom, cache)
    _WORKER = geom, g, config


def _generate(index, split):
    geom, g, c = _WORKER
    return simulate(g, geom, record_seed(c["seed"], split, index), c["simulation"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--anatomy", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--group", choices=["train", "eval", "audit"], required=True)
    args = p.parse_args(argv)
    if args.group == "audit":
        print(json.dumps(audit_splits(args.output), indent=2))
        return
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)["dataset"]
    c = cfg["simulation"]
    validate_windows(
        c["k_min"], c["k_max"], c["n_samples"], c["duration_samples"], c["overlap_frac"]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "geometry-cache"
    _init_worker(args.anatomy, cache, cfg)
    group = ["train", "val"] if args.group == "train" else ["dev", "heldout"]
    with ProcessPoolExecutor(
        args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(args.anatomy, cache, cfg),
    ) as workers:
        for split in group:

            def records():
                for start in range(0, cfg["counts"][split], args.workers * 2):
                    yield from workers.map(
                        partial(_generate, split=split),
                        range(
                            start, min(start + args.workers * 2, cfg["counts"][split])
                        ),
                    )

            write_split(
                records(),
                args.output / split,
                split,
                cfg,
                sha256(args.anatomy),
                source_hash(),
            )


if __name__ == "__main__":
    main()
