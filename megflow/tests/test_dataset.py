import json

import numpy as np
import pytest

from megflow import dataset, prototype_simulator
from megflow.geometry import CorticalGeometry


def test_fixed_temporal_windows_and_reproducibility():
    a = prototype_simulator.fixed_time_courses(np.random.default_rng(4))
    b = prototype_simulator.fixed_time_courses(np.random.default_rng(4))
    assert a.shape == (3, 1800)
    np.testing.assert_array_equal(a, b)
    assert np.count_nonzero(a[0, 600:]) == 0
    assert np.count_nonzero(a[1, :600]) == np.count_nonzero(a[1, 1200:]) == 0
    assert np.count_nonzero(a[2, :1000]) == np.count_nonzero(a[2, 1700:]) == 0
    assert np.isfinite(a).all() and np.std(a) > 0


def test_cortical_noise_is_deterministic_and_unit_std():
    g = np.random.default_rng(0).normal(size=(4, 6))
    a = prototype_simulator.brain_noise(g, np.random.default_rng(8), n_noise_sources=5)
    b = prototype_simulator.brain_noise(g, np.random.default_rng(8), n_noise_sources=5)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (4, 1800)
    assert np.std(a, ddof=1) == pytest.approx(1, abs=1e-10)


def test_legacy_geometry_requires_explicit_opt_in(geometry):
    g = geometry.adjacency.copy().tolil()
    g[0, 1] = g[1, 0] = 0
    args = (geometry.vertices, geometry.normals, g.tocsr(), geometry.hemisphere)
    with pytest.raises(ValueError, match='two connected'):
        CorticalGeometry(*args)
    legacy = CorticalGeometry(*args, allow_fragmented_hemispheres=True)
    assert np.isinf(legacy.geodesic(0, 1))


def test_split_seeds_are_disjoint():
    seeds = [dataset.record_seed(20260914, split, i)
             for split in dataset.SPLITS for i in range(1000)]
    assert len(set(seeds)) == 4000
    with pytest.raises(ValueError):
        dataset.record_seed(1, 'heldout_typo', 0)


def test_shards_roundtrip_and_corruption_rejected(tmp_path):
    rng = np.random.default_rng(3)
    records = []
    for i in range(5):
        records.append(dict(Y=rng.normal(size=(4, 24)).astype('f4'),
                            theta=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='f4') + i,
                            centers=np.array([0, 2, 4]), radii=np.array([10, 8, 10], dtype='f4'),
                            seed=i, snr_db=10., placement_attempts=1))
    manifest = dataset.write_split(iter(records), tmp_path / 'train', 'train',
                                   dict(seed=4, pool=12, shard_size=2), 'anatomy-hash', 'code-hash')
    assert manifest['n_records'] == 5
    assert [x['n_records'] for x in manifest['shards']] == [2, 2, 1]
    loaded = dataset.load_training_arrays(tmp_path / 'train')
    assert loaded['Y'].shape == (5, 4, 2)
    np.testing.assert_allclose(loaded['Y'][0], records[0]['Y'].reshape(4, 2, 12).mean(-1))
    np.testing.assert_array_equal(loaded['theta'][0], records[0]['theta'])
    mean, std = dataset.xyz_statistics(loaded['theta'])
    assert mean.shape == std.shape == (3,)
    np.testing.assert_allclose(mean, loaded['theta'].mean(axis=(0, 1)))
    with pytest.raises(FileExistsError):
        dataset.write_split(iter(records), tmp_path / 'train', 'train', {}, '', '')
    shard = tmp_path / 'train' / manifest['shards'][0]['file']
    with shard.open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        dataset.load_training_arrays(tmp_path / 'train')


def test_training_loader_rejects_heldout(tmp_path):
    tmp_path.joinpath('manifest.json').write_text(json.dumps({'split': 'heldout'}))
    with pytest.raises(ValueError, match='train or val'):
        dataset.load_training_arrays(tmp_path)


def test_audit_rejects_changed_simulator_and_corrupt_eval(tmp_path):
    config = dict(seed=4, pool=12, shard_size=2, counts={s: 1 for s in dataset.SPLITS})
    for i, split in enumerate(dataset.SPLITS):
        r = dict(Y=np.zeros((4, 24), dtype='f4'), theta=np.ones((3, 3), dtype='f4'),
                 centers=np.array([i, i + 1, i + 2]), radii=np.array([10, 8, 10]),
                 seed=i, snr_db=10., placement_attempts=1)
        dataset.write_split(iter([r]), tmp_path / split, split, config, 'anatomy', 'code')
    assert dataset.audit_splits(tmp_path)['counts']['heldout'] == 1
    path = tmp_path / 'heldout' / 'manifest.json'
    original = path.read_text()
    m = json.loads(original)
    m['config']['noise_level'] = .8
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='config|fingerprint'):
        dataset.audit_splits(tmp_path)
    path.write_text(original)
    with (tmp_path / 'heldout' / m['shards'][0]['file']).open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        dataset.audit_splits(tmp_path)
