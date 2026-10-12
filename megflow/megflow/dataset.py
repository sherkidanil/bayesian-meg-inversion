"""Immutable, independently seeded M1 shards; the trainer cannot load evaluation splits."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from omegaconf import OmegaConf

from .geometry import CorticalGeometry
from .prototype_simulator import simulate_fixed


SPLITS = {'train': 101, 'val': 102, 'dev': 201, 'heldout': 202}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hash():
    root = Path(__file__).parent
    return digest_json({p.name: sha256(p) for p in sorted(root.glob('*.py'))})


def record_seed(seed, split, index):
    if split not in SPLITS or index < 0:
        raise ValueError('unknown split or negative record index')
    return int(np.random.SeedSequence([seed, SPLITS[split], index]).generate_state(
        1, dtype=np.uint64)[0])


def xyz_statistics(theta):
    theta = np.asarray(theta, dtype=np.float64)
    return theta.mean(axis=(0, 1)).astype('f4'), (theta.std(axis=(0, 1)) + 1e-3).astype('f4')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def write_split(records, directory, split, config, anatomy_hash, code_hash):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    fingerprint = digest_json(dict(config=config, anatomy_sha256=anatomy_hash,
                                   code_sha256=code_hash, split=split))
    manifest = dict(format_version=1, split=split, fingerprint=fingerprint, config=config,
                    anatomy_sha256=anatomy_hash, code_sha256=code_hash, shards=[], n_records=0,
                    created_utc=datetime.now(timezone.utc).isoformat(), units='mm',
                    versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    chunk, identities, geometry_ids, seeds, snrs = [], [], [], [], []

    def flush():
        if not chunk:
            return
        ys = np.stack([r['Y'] for r in chunk])
        pool = int(config['pool'])
        if ys.shape[-1] % pool:
            raise ValueError('time axis must be divisible by pool')
        payload = dict(Y=ys, Y_pool=ys.reshape(*ys.shape[:-1], -1, pool).mean(-1),
                       theta=np.stack([r['theta'] for r in chunk]),
                       centers=np.stack([r['centers'] for r in chunk]),
                       radii=np.stack([r['radii'] for r in chunk]),
                       seed=np.array([r['seed'] for r in chunk], dtype='uint64'),
                       snr_db=np.array([r['snr_db'] for r in chunk]),
                       K=np.full(len(chunk), 3, dtype='int16'),
                       placement_attempts=np.array([r['placement_attempts'] for r in chunk]),
                       record_id=np.array(identities[-len(chunk):]))
        name = f'{split}-{fingerprint[:16]}-{len(manifest["shards"]):04d}.npz'
        path = directory / name
        np.savez(path, **payload)
        manifest['shards'].append(dict(file=name, sha256=sha256(path), n_records=len(chunk)))
        manifest['n_records'] += len(chunk)
        print(json.dumps(dict(split=split, generated=manifest['n_records'],
                              seconds=round(time.monotonic() - start, 1))), flush=True)
        chunk.clear()

    for record in records:
        if not np.isfinite(record['Y']).all() or not np.isfinite(record['theta']).all():
            raise ValueError('non-finite simulated recording')
        geometry_id = digest_json(dict(centers=np.asarray(record['centers']).tolist(),
                                      radii=np.asarray(record['radii']).tolist()))
        identity = digest_json(dict(geometry_id=geometry_id, seed=int(record['seed'])))
        geometry_ids.append(geometry_id)
        identities.append(identity)
        seeds.append(int(record['seed']))
        snrs.append(float(record['snr_db']))
        chunk.append(record)
        if len(chunk) == config['shard_size']:
            flush()
    flush()
    if not identities or len(set(identities)) != len(identities) or len(set(seeds)) != len(seeds):
        raise ValueError('empty dataset or duplicate record identity/seed')
    manifest.update(record_ids=identities, geometry_ids=geometry_ids, seeds=seeds,
                    elapsed_seconds=time.monotonic() - start,
                    snr_db=dict(mean=float(np.mean(snrs)), minimum=min(snrs), maximum=max(snrs)))
    write_json(directory / 'manifest.json', manifest)
    return manifest


def load_training_arrays(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['split'] not in ('train', 'val'):
        raise ValueError('trainer can only load train or val')
    ys, theta = [], []
    for shard in manifest['shards']:
        path = directory / shard['file']
        if sha256(path) != shard['sha256']:
            raise ValueError(f'dataset checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            ys.append(data['Y_pool'])
            theta.append(data['theta'])
    return dict(Y=np.concatenate(ys), theta=np.concatenate(theta), manifest=manifest)


def audit_splits(root):
    root = Path(root)
    manifests = {split: json.loads((root / split / 'manifest.json').read_text()) for split in SPLITS}
    for split, manifest in manifests.items():
        expected = digest_json(dict(config=manifest['config'], anatomy_sha256=manifest['anatomy_sha256'],
                                    code_sha256=manifest['code_sha256'], split=split))
        if manifest['split'] != split or manifest['fingerprint'] != expected:
            raise ValueError('split fingerprint/config mismatch')
        count = manifest['n_records']
        if count != manifest['config']['counts'][split] or count != sum(s['n_records'] for s in manifest['shards']):
            raise ValueError('split count mismatch')
        for key in ('record_ids', 'seeds', 'geometry_ids'):
            if len(manifest[key]) != count or len(set(manifest[key])) != count:
                raise ValueError(f'duplicate or missing {key} in {split}')
        for shard in manifest['shards']:
            if sha256(root / split / shard['file']) != shard['sha256']:
                raise ValueError(f'dataset checksum mismatch: {split}/{shard["file"]}')
    for key in ('config', 'code_sha256'):
        if len({digest_json(m[key]) for m in manifests.values()}) != 1:
            raise ValueError(f'split {key} mismatch')
    for key in ('record_ids', 'seeds', 'geometry_ids'):
        for i, a in enumerate(SPLITS):
            for b in list(SPLITS)[i + 1:]:
                if set(manifests[a][key]) & set(manifests[b][key]):
                    raise ValueError(f'split overlap in {key}: {a}, {b}')
    if len({m['anatomy_sha256'] for m in manifests.values()}) != 1:
        raise ValueError('split anatomies differ')
    report = dict(counts={k: v['n_records'] for k, v in manifests.items()},
                  disjoint_seeds=True, disjoint_record_ids=True, disjoint_geometry_ids=True,
                  shard_checksums_verified=True, simulator_and_config_match=True,
                  manifest_sha256={k: sha256(root / k / 'manifest.json') for k in SPLITS})
    write_json(root / 'audit.json', report)
    return report


_WORKER = None


def _init_worker(anatomy, cache, config):
    global _WORKER
    if sha256(anatomy) != config['anatomy_sha256']:
        raise ValueError('M1 anatomy checksum differs from the frozen legacy bundle')
    geom, leadfield, provenance = CorticalGeometry.load_bundle(anatomy, allow_fragmented_hemispheres=True)
    if (provenance.get('graph_construction') != 'legacy_cropped_mesh_explicit_M1_compatibility'
            or provenance.get('n_components') != 3 or leadfield.shape != (102, 4699)
            or not np.array_equal(leadfield, leadfield.astype('f4').astype('f8'))):
        raise ValueError('M1 requires the explicitly exported prototype anatomy')
    geom.cache_distances(cache)
    _WORKER = geom, leadfield, config


def _generate(index, split):
    geom, leadfield, config = _WORKER
    return simulate_fixed(leadfield, geom, record_seed(config['seed'], split, index),
                          min_separation_mm=config['min_separation_mm'], noise_level=config['noise_level'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--anatomy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--group', choices=['train', 'eval', 'audit'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.group == 'audit':
        print(json.dumps(audit_splits(args.output), indent=2))
        return
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)['dataset']
    if (cfg['simulator'] != 'prototype_fixed_k3_v1' or cfg['radii_mm'] != [10., 8., 10.]
            or cfg['n_samples'] != 1800 or cfg['sfreq'] != 1000
            or cfg['geometry_mode'] != 'legacy_fragmented'):
        raise ValueError('configuration does not describe the implemented M1 prototype simulator')
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / 'geometry-cache'
    # Build once in parent; workers only open the verified immutable memory map.
    _init_worker(args.anatomy, cache, cfg)
    group = ['train', 'val'] if args.group == 'train' else ['dev', 'heldout']
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                             initializer=_init_worker, initargs=(args.anatomy, cache, cfg)) as pool:
        for split in group:
            # Bounded submissions avoid retaining thousands of full recordings in futures.
            def records():
                count = cfg['counts'][split]
                for start in range(0, count, args.workers * 4):
                    yield from pool.map(partial(_generate, split=split),
                                        range(start, min(start + args.workers * 4, count)))
            write_split(records(), args.output / split, split, cfg, sha256(args.anatomy), source_hash())


if __name__ == '__main__':
    main()
