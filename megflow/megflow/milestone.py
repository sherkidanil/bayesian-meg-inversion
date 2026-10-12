"""M0 execution, reproducibility records, and explicit acceptance gates."""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import wilcoxon

from .decoder import coordinate_ascent
from .forward import patch_dictionary, rank_diagnostics, signal_subspace, subspace_score
from .geometry import CorticalGeometry
from .metrics import localization_errors
from .simulator import PlacementError, simulate


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_K = [1, 2, 3, 5, 8, 12, 20]
# Preregistered before results; custom configurations remain exploratory.
FROZEN_M0_CONFIG_SHA256 = 'c415b5028a740b86faa9b706d1ebde8b556a2fc2e3dee64f6a2a946b04326f71'


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path=None):
    base = OmegaConf.load(PACKAGE_ROOT / 'config/base.yaml')
    stage = OmegaConf.load(path or PACKAGE_ROOT / 'config/m0.yaml')
    return OmegaConf.to_container(OmegaConf.merge(base, stage), resolve=True)


def validate_config(config):
    if config.get('milestone') != 'M0':
        raise ValueError('only M0 is implemented; later milestones are gated')
    for key in ['k_values', 'seeds']:
        values = config[key]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{key} must be nonempty and unique')
    if config['n_per_k'] < 1 or min(config['k_values']) < 1:
        raise ValueError('sample count and source counts must be positive')
    if config['simulator'].get('noise_mode') != 'white':
        raise ValueError('M0 currently implements the declared white-noise protocol only')
    if config['simulator']['time_courses'].get('mode') != 'random':
        raise ValueError('arbitrary-K M0 requires randomized time courses')


def run_cases(config, geometry, leadfield):
    validate_config(config)
    sim = config['simulator']
    dictionary = patch_dictionary(leadfield, geometry, sim['radius_mm'], sim['sigma_ratio'])
    for seed in config['seeds']:
        random.seed(seed)
        np.random.seed(seed)
        # No torch operations occur in M0. Its seed is set by future training entrypoints.
        for k in config['k_values']:
            for index in range(config['n_per_k']):
                recording_seed = int(np.random.SeedSequence([seed, k, index]).generate_state(
                    1, dtype=np.uint64)[0])
                row = {'seed': seed, 'k': k, 'index': index,
                       'recording_seed': recording_seed, 'status': 'ok'}
                try:
                    recording = simulate(leadfield, geometry, k, recording_seed, sim, dictionary)
                except PlacementError as error:
                    row.update(status='placement_failed', placement_attempts=error.attempts)
                    yield row
                    continue
                row['placement_attempts'] = recording.placement_attempts
                row['centroids_true'] = recording.centroids.tolist()
                diagnostic = rank_diagnostics(dictionary[:, recording.centroids])
                row.update(diagnostic)
                for label, y in [('noisy', recording.Y), ('clean', recording.Y_clean)]:
                    e_s, singular = signal_subspace(y, k, **config['score'])
                    initial = subspace_score(e_s, dictionary[:, recording.centroids])
                    result = coordinate_ascent(recording.centroids, dictionary, e_s, geometry,
                                               min_separation_mm=sim['min_separation_mm'],
                                               **config['decoder'])
                    drift = localization_errors(recording.centroids, result.centroids, geometry)
                    row.update({f'{label}_score_before': initial,
                                f'{label}_score_after': result.score,
                                f'{label}_score_gain': result.score - initial,
                                f'{label}_drift_mm': float(np.mean(drift)),
                                f'{label}_hemisphere_misses': int(np.isinf(drift).sum()),
                                f'{label}_moved': int(result.n_moves > 0),
                                f'{label}_n_moves': result.n_moves,
                                f'{label}_signal_rank': e_s.shape[1],
                                f'{label}_sensor_singular_values': singular.tolist(),
                                f'{label}_centroids_after': result.centroids.tolist()})
                yield row


def paired_diagnostics(rows, seed):
    if not rows:
        return None
    result = {}
    rng = np.random.default_rng([seed, 9871])
    for metric in ['score_before', 'drift_mm']:
        differences = np.array([r[f'noisy_{metric}'] - r[f'clean_{metric}'] for r in rows])
        if not np.isfinite(differences).all():
            result[metric] = {'mean_difference': None, 'reason': 'non-finite pair',
                              'n_nonfinite': int((~np.isfinite(differences)).sum())}
            continue
        estimates = differences[rng.integers(len(rows), size=(2000, len(rows)))].mean(axis=1)
        p = 1. if np.all(differences == 0) else float(wilcoxon(differences).pvalue)
        result[metric] = {'mean_difference': float(differences.mean()),
                          'bootstrap_ci95': np.quantile(estimates, [.025, .975]).tolist(),
                          'wilcoxon_two_sided_p': p, 'n_pairs': len(rows)}
    return result


def summarize(rows, config):
    result = []
    for seed in config['seeds']:
        for k in config['k_values']:
            requested = [r for r in rows if r['seed'] == seed and r['k'] == k]
            usable = [r for r in requested if r['status'] == 'ok']
            n = len(usable)
            stat = {'seed': seed, 'k': k, 'n_requested': len(requested), 'n_simulated': n,
                    'placement_success': n / len(requested) if requested else 0.}
            def aggregate(key, mode='mean'):
                if not n:
                    return None
                values = np.asarray([r[key] for r in usable])
                if mode == 'median':
                    return float(np.quantile(values, .5, method='inverted_cdf'))
                if mode == 'p90':
                    return float(np.quantile(values, .9, method='inverted_cdf'))
                if mode == 'max':
                    return float(np.max(values))
                return float(np.mean(values))
            stat.update(rho_mean=aggregate('noisy_score_before'),
                        rho_median=aggregate('noisy_score_before', 'median'),
                        condition_median=aggregate('condition_number', 'median'),
                        effective_rank_mean=aggregate('effective_rank'),
                        numerical_rank_mean=aggregate('numerical_rank'),
                        full_rank_fraction=sum(r['numerical_rank'] == k for r in usable) / n if n else None,
                        move_fraction=aggregate('noisy_moved'),
                        drift_mean_mm=aggregate('noisy_drift_mm'),
                        drift_p90_mm=aggregate('noisy_drift_mm', 'p90'),
                        score_gain_mean=aggregate('noisy_score_gain'),
                        noiseless_gain_max=aggregate('clean_score_gain', 'max'),
                        hemisphere_misses=sum(r['noisy_hemisphere_misses'] for r in usable),
                        paired=paired_diagnostics(usable, seed))
            result.append(stat)
    return result


def acceptance_gate(stats, config, provenance, shape):
    reasons = []
    if config_hash(config) != FROZEN_M0_CONFIG_SHA256:
        reasons.append('configuration differs from the frozen M0 protocol')
    if config.get('smoke'):
        reasons.append('smoke run cannot establish acceptance')
    if provenance.get('kind') not in ['mne_sample', 'verified_historical_mne_sample']:
        reasons.append('real MNE sample anatomy provenance is required')
    if tuple(shape) != (102, 4699):
        reasons.append('requires 102 sensors and 4699 vertices')
    expected = {(s, k) for s in config['seeds'] for k in config['k_values']}
    observed = [(s['seed'], s['k']) for s in stats]
    if set(observed) != expected or len(observed) != len(expected):
        reasons.append('incomplete or duplicate seed/K cells')
    if (config['k_values'] != REQUIRED_K or len(set(config['seeds'])) < 3
            or config['n_per_k'] < 200
            or any(s['n_requested'] != config['n_per_k'] for s in stats)):
        reasons.append('requires all seven K values and >=200 requests/K for >=3 seeds')
    thresholds = config['acceptance']
    by_k = {}
    recommended = None
    prefix_ok = True
    for k in sorted(config['k_values']):
        cells = [s for s in stats if s['k'] == k]
        passed = bool(cells) and len(cells) == len(config['seeds'])
        for cell in cells:
            for name, threshold, greater in [
                ('placement_success', thresholds['min_placement_success'], True),
                ('full_rank_fraction', thresholds['min_full_rank_fraction'], True),
                ('rho_median', thresholds['min_median_rho'], True),
                ('drift_p90_mm', thresholds['max_p90_drift_mm'], False),
                ('noiseless_gain_max', thresholds['max_noiseless_gain'], False)]:
                value = cell[name]
                passed &= value is not None and np.isfinite(value) and (
                    value >= threshold if greater else value <= threshold)
        by_k[str(k)] = bool(passed)
        prefix_ok &= passed
        if prefix_ok:
            recommended = k
    if recommended is None:
        reasons.append('no tested prefix passes the frozen operational tolerances')
    return {'accepted': not reasons, 'recommended_k_max': recommended if not reasons else None,
            'per_k_pass': by_k, 'reasons': reasons,
            'untested_intermediate_k': [k for k in range(1, (recommended or 0) + 1)
                                       if k not in config['k_values']],
            'interpretation': 'operational white-noise local-oracle limit; not a proof of identifiability'}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return 'inf' if value == np.inf else '-inf' if value == -np.inf else None
    return value


def report_markdown(stats, gate, manifest):
    def fmt(x):
        return 'undefined' if x is None else f'{x:.4g}'
    def mean_std(cells, name):
        values = [s[name] for s in cells]
        if any(v is None for v in values):
            return 'undefined'
        if not np.isfinite(values).all():
            return 'inf (not filtered)'
        return f'{np.mean(values):.4g} ± {np.std(values, ddof=1) if len(values)>1 else 0:.3g}'
    text = ['# M0 oracle stability report', '',
            f"Code commit: `{manifest['code_commit']}`; dirty: `{manifest['code_dirty']}`.",
            f"Config SHA-256: `{manifest['config_sha256']}`.",
            f"Geometry SHA-256: `{manifest['geometry_sha256']}`.",
            f"Started: {manifest['started_utc']}; elapsed: {manifest['elapsed_seconds']:.1f} s.",
            f"Seeds: {manifest['config']['seeds']}; requests per K/seed: {manifest['config']['n_per_k']}.",
            '', '## Across seeds: mean ± sample standard deviation', '',
            '| K | Requested / simulated | Placement | Oracle rho | cond(B), median | Effective rank | Move fraction | Drift mm | Drift p90 mm |',
            '|---|---|---|---|---|---|---|---|---|']
    for k in manifest['config']['k_values']:
        cells = [s for s in stats if s['k'] == k]
        numbers = [mean_std(cells, key) for key in ['placement_success', 'rho_mean',
                    'condition_median', 'effective_rank_mean', 'move_fraction',
                    'drift_mean_mm', 'drift_p90_mm']]
        text.append(f"| {k} | {sum(s['n_requested'] for s in cells)} / {sum(s['n_simulated'] for s in cells)} | "
                    + ' | '.join(numbers) + ' |')
    text += ['', '## Per-seed gate evidence', '',
             '| Seed | K | Full rank fraction | Median rho | Drift p90 mm | Max clean score gain | Hemisphere misses |',
             '|---|---|---|---|---|---|---|']
    for s in stats:
        text.append(f"| {s['seed']} | {s['k']} | " + ' | '.join(fmt(s[key]) for key in
                    ['full_rank_fraction', 'rho_median', 'drift_p90_mm', 'noiseless_gain_max',
                     'hemisphere_misses']) + ' |')
    text += ['', '## Acceptance', '', f"Accepted: **{gate['accepted']}**.",
             f"Recommended K_max: **{gate['recommended_k_max']}**.",
             f"Untested intermediate counts: {gate['untested_intermediate_k']}.",
             f"Reasons: {gate['reasons']}.", '',
             'Gate uses every seed separately; tolerances are saved in config.acceptance.',
             'Scores are compared only within fixed K. Local ascent cannot prove global uniqueness.',
             'White noise, randomized correlated courses and known 10 mm radii define this experiment.',
             'Placement failures remain in the requested denominator. Score/rank quantities exist only',
             'for generated recordings; simulated counts are explicit. Infinite localization errors',
             'are never removed. Paired noisy-minus-clean bootstrap intervals and Wilcoxon tests',
             'are in summary.json (exploratory, not used to tune the gate).', '']
    return '\n'.join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=PACKAGE_ROOT / 'cache')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--n-per-k', type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke:
        cfg.update(smoke=True, seeds=cfg['seeds'][:1], n_per_k=2)
    if args.n_per_k is not None:
        cfg['n_per_k'] = args.n_per_k
    validate_config(cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    repo = PACKAGE_ROOT.parent
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip())
    geometry, g, provenance = CorticalGeometry.load_bundle(args.geometry)
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
                'config': cfg, 'config_sha256': config_hash(cfg),
                'simulator_sha256': config_hash(cfg['simulator']),
                'geometry_sha256': sha256_file(args.geometry),
                'geometry_path': str(args.geometry.resolve()), 'provenance': provenance,
                'code_commit': commit, 'code_dirty': dirty,
                'python': platform.python_version(), 'platform': platform.platform(),
                'thread_environment': {name: os.environ.get(name) for name in
                    ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS']},
                'packages': {name: importlib.metadata.version(name)
                             for name in ['numpy', 'scipy', 'omegaconf']}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    geometry.cache_distances(args.cache_dir)
    print(f'M0: {g.shape}, config {manifest["config_sha256"][:12]}', flush=True)
    rows = []
    raw_path = args.output / f'cases-{manifest["simulator_sha256"][:16]}.jsonl'
    with raw_path.open('x') as raw:
        for row in run_cases(cfg, geometry, g):
            raw.write(json.dumps(json_safe(row), allow_nan=False) + '\n')
            raw.flush()
            rows.append(row)
            if (row['index'] + 1) % 25 == 0 or row['index'] + 1 == cfg['n_per_k']:
                print(f"seed={row['seed']} K={row['k']} {row['index']+1}/{cfg['n_per_k']} "
                      f"elapsed={time.monotonic()-start:.1f}s", flush=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output / 'cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})
    stats = summarize(rows, cfg)
    gate = acceptance_gate(stats, cfg, provenance, g.shape)
    if dirty:
        gate['accepted'] = False
        gate['recommended_k_max'] = None
        gate['reasons'].append('scientific execution requires a clean committed code snapshot')
    manifest.update(elapsed_seconds=time.monotonic() - start,
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                    n_rows=len(rows), completed=True, raw_sha256=sha256_file(raw_path))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(json_safe(
        {'stats': stats, 'gate': gate}), indent=2, allow_nan=False) + '\n')
    (args.output / 'REPORT.md').write_text(report_markdown(stats, gate, manifest))
    print(json.dumps(gate), flush=True)
    return 0
