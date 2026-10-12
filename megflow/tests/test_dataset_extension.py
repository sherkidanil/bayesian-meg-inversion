import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from megflow.dataset import SPLITS, audit_splits, record_seed, sha256, source_hash, write_split


def extension_module():
    path = Path(__file__).parents[1] / 'scripts/extend_dataset.py'
    spec = importlib.util.spec_from_file_location('extend_dataset', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_record(index, split):
    offset = list(SPLITS).index(split) * 100
    return dict(Y=np.full((2, 24), index + offset, dtype='f4'),
                theta=np.full((3, 3), index + offset, dtype='f4'),
                centers=np.array([index + offset, 500, 600]), radii=np.array([10., 8., 10.]),
                seed=record_seed(4, split, index), snr_db=10. + index / 100, placement_attempts=1)


def make_parent(root):
    config = dict(seed=4, pool=12, shard_size=2, counts={s: 3 for s in SPLITS}, noise_level=.3)
    for split in SPLITS:
        write_split((toy_record(i, split) for i in range(3)), root / split, split,
                    config, 'anatomy', source_hash())
    audit_splits(root)
    return config


def test_extension_preserves_prefix_and_evaluation_bytes(tmp_path):
    extension = extension_module()
    old = make_parent(tmp_path / 'old')
    config = copy.deepcopy(old)
    config['counts']['train'] = 7
    requested = []
    def new_records(indices):
        requested.extend(indices)
        return (toy_record(i, 'train') for i in indices)
    report = extension.extend_dataset(tmp_path / 'old', tmp_path / 'new', config, new_records)
    assert requested == [3, 4, 5, 6]
    assert report['counts'] == dict(train=7, val=3, dev=3, heldout=3)
    import json
    for split in SPLITS:
        a = json.loads((tmp_path / 'old' / split / 'manifest.json').read_text())
        b = json.loads((tmp_path / 'new' / split / 'manifest.json').read_text())
        assert b['record_ids'][:3] == a['record_ids']
        for old_shard, new_shard in zip(a['shards'], b['shards']):
            src = tmp_path / 'old' / split / old_shard['file']
            dst = tmp_path / 'new' / split / new_shard['file']
            assert sha256(src) == sha256(dst)
            assert src.stat().st_ino == dst.stat().st_ino  # no raw-data copy
        lineage = tmp_path / 'new' / 'lineage' / split / 'manifest.json'
        assert sha256(lineage) == sha256(tmp_path / 'old' / split / 'manifest.json')
    with pytest.raises(FileExistsError):
        extension.extend_dataset(tmp_path / 'old', tmp_path / 'new', config, new_records)


@pytest.mark.parametrize('mutation', ['noise', 'val_count', 'source_code', 'smaller'])
def test_extension_rejects_incompatible_inputs_before_generation(tmp_path, mutation):
    extension = extension_module()
    config = make_parent(tmp_path / 'old')
    config = copy.deepcopy(config)
    config['counts']['train'] = 7
    if mutation == 'noise':
        config['noise_level'] = .8
    elif mutation == 'val_count':
        config['counts']['val'] = 4
    elif mutation == 'smaller':
        config['counts']['train'] = 2
    else:
        import json
        p = tmp_path / 'old' / 'train' / 'manifest.json'
        m = json.loads(p.read_text()); m['code_sha256'] = 'wrong'
        p.write_text(json.dumps(m))
    def must_not_generate(*args):
        pytest.fail('generation started before preflight rejection')
    with pytest.raises(ValueError):
        extension.extend_dataset(tmp_path / 'old', tmp_path / 'new', config, must_not_generate)


def test_extension_overlap_is_rejected(tmp_path):
    extension = extension_module()
    config = make_parent(tmp_path / 'old')
    config['counts']['train'] = 4
    repeated = toy_record(0, 'heldout')
    repeated['seed'] = record_seed(4, 'train', 3)
    with pytest.raises(ValueError, match='duplicate|overlap'):
        extension.extend_dataset(tmp_path / 'old', tmp_path / 'new', config,
                                 lambda indices: iter([repeated]))


def test_excluded_candidate_is_replaced_and_recorded(tmp_path):
    extension = extension_module()
    config = make_parent(tmp_path / 'old')
    config['counts']['train'] = 7
    config['train_index_exclusions'] = [4]
    requested = []
    def records(indices):
        requested.extend(indices)
        return (toy_record(i, 'train') for i in indices)
    report = extension.extend_dataset(tmp_path / 'old', tmp_path / 'new', config, records)
    assert requested == [3, 5, 6, 7]
    assert report['counts']['train'] == 7
    import json
    m = json.loads((tmp_path / 'new/train/manifest.json').read_text())
    assert m['candidate_train_index_range'] == [3, 8]
    assert m['excluded_train_indices'] == [4]
