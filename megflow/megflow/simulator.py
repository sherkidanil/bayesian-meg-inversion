"""Seeded M0 simulation: random temporal signatures and explicit white noise."""

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from .forward import topographies


class PlacementError(RuntimeError):
    def __init__(self, k, attempts):
        self.attempts = attempts
        super().__init__(f'Could not place K={k} sources after {attempts} attempts')


def sample_centroids(geom, k, min_separation_mm, rng, max_attempts=100):
    if k < 1 or max_attempts < 1 or min_separation_mm < 0:
        raise ValueError('invalid placement configuration')
    n = len(geom.vertices)
    for attempt in range(1, max_attempts + 1):
        available = np.ones(n, dtype=bool)
        centers = []
        for _ in range(k):
            choices = np.flatnonzero(available)
            if not choices.size:
                break
            center = int(rng.choice(choices))
            centers.append(center)
            available &= geom.geodesic_from(center) >= min_separation_mm
            available[center] = False
        if len(centers) == k:
            return np.asarray(centers, dtype=np.int64), attempt
    raise PlacementError(k, max_attempts)


def random_time_courses(k, config, rng):
    n = int(config.get('n_samples', 1800))
    fs = float(config.get('sfreq', 1000))
    band = config.get('band_hz', [2., 40.])
    bandwidth = config.get('bandwidth_hz', [2., 6.])
    duration = config.get('window_dur_s', [.3, 1.2])
    if not (0 < band[0] < band[1] < fs / 2 and 0 < duration[0] <= duration[1] <= n / fs):
        raise ValueError('invalid temporal frequency or duration range')
    courses = np.zeros((k, n))
    pad = int(3 * fs)
    for j in range(k):
        width = float(rng.uniform(*bandwidth))
        width = min(width, band[1] - band[0])
        center = rng.uniform(band[0] + width / 2, band[1] - width / 2)
        sos = butter(3, [center - width / 2, center + width / 2], fs=fs,
                     btype='bandpass', output='sos')
        course = sosfiltfilt(sos, rng.standard_normal(n + 2 * pad))[pad:pad + n]
        length = min(n, max(3, round(rng.uniform(*duration) * fs)))
        start = int(rng.integers(n - length + 1))
        courses[j, start:start + length] = course[start:start + length] * tukey(length, .25)
    courses /= np.maximum(courses.std(axis=1, keepdims=True), 1e-15)
    correlation = config.get('correlation', {'mode': 'none'})
    mode = correlation.get('mode', 'none')
    if mode not in ('none', 'pairwise'):
        raise ValueError(f'unsupported M0 correlation mode: {mode}')
    if (mode == 'pairwise' and k >= 2
            and rng.random() < correlation.get('p_correlated_pair', .3)):
        a, b = rng.choice(k, 2, replace=False)
        rho = float(rng.uniform(*correlation.get('rho_range', [0., .95])))
        if not 0 <= rho <= 1:
            raise ValueError('pairwise correlation must be in [0,1]')
        courses[b] = rho * courses[a] + np.sqrt(1 - rho ** 2) * courses[b]
        courses[b] /= max(courses[b].std(), 1e-15)
    return courses


@dataclass
class Recording:
    Y: np.ndarray
    Y_clean: np.ndarray
    centroids: np.ndarray
    radii: np.ndarray
    K: int
    time_courses: np.ndarray
    snr: float
    seed: int
    placement_attempts: int


def simulate(g, geom, k, seed, config, dictionary=None):
    rng = np.random.default_rng(seed)
    centers, attempts = sample_centroids(geom, k, config['min_separation_mm'], rng,
                                         config.get('max_placement_attempts', 100))
    courses = random_time_courses(k, config['time_courses'], rng)
    radius = float(config['radius_mm'])
    b = (topographies(g, geom, centers, radius, config['sigma_ratio'])
         if dictionary is None else dictionary[:, centers])
    y_clean = b @ courses
    scale = y_clean.std()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('simulated clean sensor signal has zero/nonfinite variance')
    y_clean /= scale
    snr = float(config['snr'])
    if not np.isfinite(snr) or snr <= 0:
        raise ValueError('snr must be positive and finite')
    y = y_clean + rng.standard_normal(y_clean.shape) / snr
    return Recording(y.astype(np.float32), y_clean, centers, np.full(k, radius),
                     k, courses, snr, int(seed), attempts)
