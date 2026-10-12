#!/usr/bin/env python3
"""Append train records using the original immutable generator on PYTHONPATH.

Existing shard bytes are hard-linked into a fresh dataset. This script's hash is
recorded separately from the unchanged data-generation package hash.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime, timezone
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy
from omegaconf import OmegaConf

from megflow.dataset import (SPLITS, _generate, _init_worker, audit_splits, digest_json,
                             sha256, source_hash, write_json, write_split)


def validate_parent(parent, config):
    parent = Path(parent)
    audit = json.loads((parent / 'audit.json').read_text())
    for key in ['disjoint_seeds', 'disjoint_record_ids', 'disjoint_geometry_ids',
                'shard_checksums_verified', 'simulator_and_config_match']:
        if audit.get(key) is not True:
            raise ValueError(f'parent audit missing successful {key}')
    manifests = {}
    runtime = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    generator_hash = source_hash()
    for split in SPLITS:
        path = parent / split / 'manifest.json'
        if sha256(path) != audit['manifest_sha256'][split]:
            raise ValueError('parent manifest checksum mismatch')
        m = json.loads(path.read_text())
        expected = copy.deepcopy(m['config'])
        expected['counts']['train'] = config['counts']['train']
        if 'train_index_exclusions' in config:
            expected['train_index_exclusions'] = config['train_index_exclusions']
        if config != expected:
            raise ValueError('only the training count and explicit duplicate exclusions may change')
        if m['code_sha256'] != generator_hash:
            raise ValueError('extension must use the original frozen generator package')
        if m['versions'] != runtime:
            raise ValueError('extension must use the original generator runtime versions')
        for shard in m['shards']:
            if sha256(parent / split / shard['file']) != shard['sha256']:
                raise ValueError(f'parent shard checksum mismatch: {split}/{shard["file"]}')
        manifests[split] = m
    if config['counts']['train'] <= manifests['train']['n_records']:
        raise ValueError('new training count must exceed the original count')
    return manifests


def extension_indices(start, count, excluded):
    if (len(set(excluded)) != len(excluded)
            or any(type(i) is not int or i < start for i in excluded)):
        raise ValueError('excluded indices must be unique integers after the existing train prefix')
    excluded = set(excluded)
    indices, index = [], start
    while len(indices) < count:
        if index not in excluded:
            indices.append(index)
        index += 1
    if any(i >= index for i in excluded):
        raise ValueError('excluded index is outside the generated candidate interval')
    return indices


def extend_dataset(parent, output, config, new_records):
    """new_records(indices) generates the explicit continuation of train seeds."""
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifests = validate_parent(parent, config)
    start_time = time.monotonic()
    output.mkdir(parents=True)
    assembly_hash = sha256(__file__)
    start = manifests['train']['n_records']
    stop = config['counts']['train']
    indices = extension_indices(start, stop - start, config.get('train_index_exclusions', []))
    lineage = output / 'lineage'
    lineage.mkdir()
    for split in SPLITS:
        (lineage / split).mkdir()
        shutil.copyfile(parent / split / 'manifest.json', lineage / split / 'manifest.json')
        (output / split).mkdir()
    # Link parent shards before generating; EXDEV must fail early, never silently copy 80 GB.
    assembled = {}
    for split, original in manifests.items():
        m = copy.deepcopy(original)
        m['config'] = copy.deepcopy(config)
        m['fingerprint'] = digest_json(dict(config=config, anatomy_sha256=m['anatomy_sha256'],
                                           code_sha256=m['code_sha256'], split=split))
        m['shards'] = []
        for index, shard in enumerate(original['shards']):
            name = f'{split}-{m["fingerprint"][:16]}-{index:04d}.npz'
            os.link(parent / split / shard['file'], output / split / name)
            m['shards'].append(dict(shard, file=name))
        m.update(assembly_script_sha256=assembly_hash,
                 assembled_utc=datetime.now(timezone.utc).isoformat(),
                 parent_manifest_sha256=sha256(parent / split / 'manifest.json'),
                 parent_records=original['n_records'])
        assembled[split] = m
    delta_dir = output / 'delta-train'
    delta = write_split(new_records(indices), delta_dir, 'train', config,
                        manifests['train']['anatomy_sha256'], manifests['train']['code_sha256'])
    if delta['n_records'] != stop - start:
        raise ValueError('new-record count does not match requested train extension')
    # Verify the seed interval rather than accepting arbitrary new examples.
    from megflow.dataset import record_seed
    expected_seeds = [record_seed(config['seed'], 'train', i) for i in indices]
    if delta['seeds'] != expected_seeds:
        raise ValueError('new records do not use the declared train seed interval')
    m = assembled['train']
    for shard in delta['shards']:
        name = f'train-{m["fingerprint"][:16]}-{len(m["shards"]):04d}.npz'
        os.link(delta_dir / shard['file'], output / 'train' / name)
        m['shards'].append(dict(shard, file=name))
    for key in ['record_ids', 'geometry_ids', 'seeds']:
        m[key] += delta[key]
    m['n_records'] = stop
    a, b = manifests['train']['snr_db'], delta['snr_db']
    m['snr_db'] = dict(mean=(a['mean'] * start + b['mean'] * (stop - start)) / stop,
                       minimum=min(a['minimum'], b['minimum']), maximum=max(a['maximum'], b['maximum']))
    m['elapsed_seconds'] += delta['elapsed_seconds']
    m.update(added_records=stop - start, delta_manifest_sha256=sha256(delta_dir / 'manifest.json'),
             candidate_train_index_range=[start, indices[-1] + 1],
             excluded_train_indices=config.get('train_index_exclusions', []),
             extension_generation_seconds=delta['elapsed_seconds'])
    for split, manifest in assembled.items():
        write_json(output / split / 'manifest.json', manifest)
    audit = audit_splits(output)
    report = dict(audit, parent=str(parent.resolve()), inherited_train_records=start,
                   added_train_records=stop - start, evaluation_shards_byte_identical=True,
                   inherited_shards_hardlinked=True, assembly_script_sha256=assembly_hash,
                   elapsed_seconds=time.monotonic() - start_time)
    write_json(output / 'extension.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anatomy', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)['dataset']
    if args.workers < 1:
        raise ValueError('worker count must be positive')
    cache = args.output.parent / 'extension-geometry-cache'
    _init_worker(args.anatomy, cache, config)

    def new_records(indices):
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=_init_worker, initargs=(args.anatomy, cache, config)) as pool:
            for offset in range(0, len(indices), args.workers * 4):
                yield from pool.map(partial(_generate, split='train'),
                                    indices[offset:offset + args.workers * 4])

    report = extend_dataset(args.parent, args.output, config, new_records)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
