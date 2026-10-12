"""M1 compatibility simulator: fixed signatures and structured cortical noise.

Ported from generate_dataset.py, sha256
9b4823759f101a40d01804a83658270d00c11920103f265cd332aa6579788481.
Retains its filtering, gamma envelope, padding convention and RNG consumption.
"""

from functools import lru_cache

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import gamma

from .forward import topographies
from .simulator import sample_centroids


RADII = np.array([10., 8., 10.])


@lru_cache(maxsize=1)
def _temporal_parameters():
    start_stop = [(0, 600), (600, 1200), (1000, 1700)]
    bands = np.array([[8., 12.], [15., 20.], [4., 8.]]) / 500.
    zl, zr = gamma.ppf([1e-5, .999], a=6., scale=4.)
    result = []
    for (start, stop), band in zip(start_stop, bands):
        n = stop - start
        z = zl + (zr - zl) * np.arange(n) / n
        envelope = ((zr - zl) / n) * gamma.pdf(z, a=6., scale=4.) / (
            gamma.cdf(zr, a=6., scale=4.) - gamma.cdf(zl, a=6., scale=4.))
        envelope /= envelope.std(ddof=1) + 1e-12
        result.append((start, stop, butter(3, band, btype='bandpass'), band.mean(), envelope))
    return result


def fixed_time_courses(rng):
    courses = np.zeros((3, 1800))
    for j, (start, stop, (b, a), frequency, envelope) in enumerate(_temporal_parameters()):
        n = stop - start
        q = rng.standard_normal(n + 600)
        s = filtfilt(b, a, q) / (frequency + 1e-12)
        segment = s[:n]  # Original prototype uses the beginning, not the padded centre.
        courses[j, start:stop] = envelope * segment / (segment.std(ddof=1) + 1e-12)
    return courses


@lru_cache(maxsize=1)
def _noise_filters():
    bands = np.array([[8., 12.], [15., 30.], [30., 50.], [50., 70.]]) / 500.
    return [(butter(3, band, btype='bandpass'), band.mean()) for band in bands]


def brain_noise(g, rng, n_noise_sources=100):
    noise = np.zeros((g.shape[0], 1900))
    for trial in range(19):  # Retain the final discarded trial's RNG consumption.
        sources = rng.integers(g.shape[1], size=n_noise_sources)
        q = rng.standard_normal((n_noise_sources, 700))
        source_noise = sum(filtfilt(b, a, q, axis=1) / (frequency + 1e-12)
                           for (b, a), frequency in _noise_filters())
        noise[:, trial * 100:(trial + 1) * 100] = (
            g[:, sources] @ source_noise[:, 300:400] / n_noise_sources)
    noise = noise[:, :1800]
    return noise / (noise.std(ddof=1) + 1e-12)


def simulate_fixed(g, geometry, seed, *, min_separation_mm=30., noise_level=.3):
    rng = np.random.default_rng(seed)
    centers, attempts = sample_centroids(geometry, 3, min_separation_mm, rng, max_attempts=100)
    courses = fixed_time_courses(rng)
    clean = topographies(g, geometry, centers, RADII, sigma_ratio=.3333333333333) @ courses
    clean /= clean.std(ddof=1) + 1e-12
    noise = noise_level * brain_noise(g, rng)
    return dict(Y=(clean + noise).astype('f4'), theta=geometry.vertices[centers].astype('f4'),
                centers=centers, radii=RADII.astype('f4'), seed=int(seed),
                snr_db=float(10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))),
                placement_attempts=attempts)
