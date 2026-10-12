import copy
import numpy as np
import pytest

from megflow.milestone import (config_hash, load_config, run_cases, summarize,
                               acceptance_gate)


def test_config_hash_ignores_key_order_but_tracks_simulator_changes():
    assert config_hash({'a': 1, 'b': 2}) == config_hash({'b': 2, 'a': 1})
    assert config_hash({'radius': 10}) != config_hash({'radius': 8})


def test_failed_placements_keep_requested_denominator(geometry):
    cfg = load_config()
    cfg.update({'k_values': [3], 'seeds': [1], 'n_per_k': 3, 'smoke': True})
    g = np.ones((4, 6))
    rows = list(run_cases(cfg, geometry, g))
    assert len(rows) == 3
    assert all(row['status'] == 'placement_failed' for row in rows)
    assert all(row['placement_attempts'] == 100 for row in rows)
    stats = summarize(rows, cfg)
    assert stats[0]['n_requested'] == 3
    assert stats[0]['n_simulated'] == 0
    assert stats[0]['placement_success'] == 0
    assert stats[0]['rho_median'] is None


def good_stats(cfg):
    return [{'seed': seed, 'k': k, 'n_requested': 200, 'n_simulated': 200,
             'placement_success': 1., 'full_rank_fraction': 1.,
             'rho_median': .99, 'drift_p90_mm': 1., 'noiseless_gain_max': 0.}
            for seed in cfg['seeds'] for k in cfg['k_values']]


def test_gate_rejects_smoke_and_synthetic_data():
    cfg = load_config()
    stats = good_stats(cfg)
    assert not acceptance_gate(stats, cfg, {'kind': 'test_fixture'}, (102, 4699))['accepted']
    cfg['smoke'] = True
    assert not acceptance_gate(stats, cfg, {'kind': 'mne_sample'}, (102, 4699))['accepted']


def test_gate_uses_contiguous_tested_prefix_and_all_seeds():
    cfg = load_config()
    stats = good_stats(cfg)
    next(s for s in stats if s['k'] == 8 and s['seed'] == cfg['seeds'][1])['drift_p90_mm'] = 7
    gate = acceptance_gate(stats, cfg, {'kind': 'mne_sample'}, (102, 4699))
    assert gate['accepted']
    assert gate['recommended_k_max'] == 5
    assert gate['untested_intermediate_k'] == [4]


def test_gate_rejects_missing_or_duplicate_seed_k_cells():
    cfg = load_config()
    stats = good_stats(cfg)
    assert not acceptance_gate(stats[:-1], cfg, {'kind': 'mne_sample'}, (102, 4699))['accepted']
    assert not acceptance_gate(stats + stats[:1], cfg, {'kind': 'mne_sample'}, (102, 4699))['accepted']


def test_gate_rejects_changed_experiment_even_with_perfect_scores():
    cfg = load_config()
    cfg['score'] = {'alpha': 102., 'r_max': 102}
    cfg['decoder']['n_passes'] = 0
    cfg['simulator']['min_separation_mm'] = 0
    gate = acceptance_gate(good_stats(cfg), cfg, {'kind': 'mne_sample'}, (102, 4699))
    assert not gate['accepted']
    assert gate['recommended_k_max'] is None
    assert any('frozen' in reason for reason in gate['reasons'])


def test_case_seeds_reproducible_and_noiseless_control(geometry):
    cfg = load_config()
    cfg.update({'k_values': [1, 2], 'seeds': [1], 'n_per_k': 2, 'smoke': True})
    cfg['simulator']['min_separation_mm'] = 2.
    cfg['simulator']['radius_mm'] = 1.
    g = np.random.default_rng(3).normal(size=(10, 6))
    rows = list(run_cases(cfg, geometry, g))
    again = list(run_cases(copy.deepcopy(cfg), geometry, g))
    assert rows == again
    assert len(set(r['recording_seed'] for r in rows)) == 4
    assert all(r['clean_score_gain'] <= 1e-10 for r in rows)
    assert all(r['noisy_score_after'] >= r['noisy_score_before'] for r in rows)
