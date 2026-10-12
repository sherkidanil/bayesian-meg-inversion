"""Seeded, padded adapter of the collaborator's 15 September generator (known variable K)."""

import copy
import numpy as np
from scipy.stats import truncnorm

from .v15_primitives import (
    Modeling_Liedfield_Error as perturb_leadfield,
    Sample_Activation_Windows as sample_windows,
    Temporal_Activation,
    Generate_Brain_Noise,
)


def default_simulation_config():
    return dict(
        k_min=1,
        k_max=5,
        n_samples=2000,
        sfreq=1000,
        duration_samples=500,
        overlap_frac=0.4,
        min_separation_mm=50.0,
        radius_min=8.0,
        radius_max=25.0,
        radius_mu=14.515,
        radius_sigma=3.80,
        sigma_mm_scale=1 / 3,
        freq_bands_hz=[[4.0, 8.0], [8.0, 12.0], [12.0, 16.0], [16.0, 20.0]],
        noise_level=0.3,
        beta_rad=0.13,
        beta_tan=0.001,
        max_attempts=20,
        temporal=dict(pad=300, shape=6.0, scale=4.0, qL=1e-5, qR=0.999),
        noise=dict(window=0.1, N_noise_sources=100, pad=300),
        filter=dict(l_freq=2.0, h_freq=40.0, order=4, ftype="butter"),
    )


def validate_windows(k_min, k_max, length, duration, overlap):
    if not (1 <= k_min <= k_max and 0 < duration <= length and 0 <= overlap <= 1):
        raise ValueError("invalid activation window configuration")
    for k in range(k_min, k_max + 1):
        slot, extension = length // k, overlap * duration / 2
        for j in range(k):
            lo = max(0, int(round(j * slot - extension)))
            hi = min(
                length - duration, int(round((j + 1) * slot + extension - duration))
            )
            if hi < lo:
                raise ValueError(
                    f"activation window does not fit for K={k}, source={j}"
                )


def spatial_activation(params, geometry, rng):
    """Original greedy placement, with explicit vertex count and graph distance API."""
    k, n = params["N_src"], len(geometry.vertices)
    centers, activation = np.zeros(k, dtype=int), np.zeros((k, n))
    candidates = np.ones(n, dtype=bool)
    for j in range(k):
        choices = np.flatnonzero(candidates)
        if not len(choices):
            raise RuntimeError("no vertices satisfy min_dist")
        centers[j] = rng.integers(n) if j == 0 else rng.choice(choices)
        distance = geometry.geodesic_from(centers[j])
        region = distance <= params["R_mm"][j]
        sigma = params["R_mm"][j] * params["sigma_mm_scale"]
        activation[j, region] = params["A0"][j] * np.exp(
            -0.5 * (distance[region] / sigma) ** 2
        )
        candidates &= distance >= params["min_dist"]
    return centers, activation


def filter_recording(y, config):
    import mne

    f = config["filter"]
    return mne.filter.filter_data(
        y,
        config["sfreq"],
        f["l_freq"],
        f["h_freq"],
        method="iir",
        iir_params=dict(order=f["order"], ftype=f["ftype"], output="sos"),
        verbose=False,
    )


def simulate(g, geometry, seed, config, *, apply_filter=True):
    c = config
    validate_windows(
        c["k_min"], c["k_max"], c["n_samples"], c["duration_samples"], c["overlap_frac"]
    )
    rng = np.random.default_rng(seed)
    a = (c["radius_min"] - c["radius_mu"]) / c["radius_sigma"]
    b = (c["radius_max"] - c["radius_mu"]) / c["radius_sigma"]
    radius_dist = truncnorm(a, b, loc=c["radius_mu"], scale=c["radius_sigma"])
    for attempt in range(c["max_attempts"]):
        k = int(rng.integers(c["k_min"], c["k_max"] + 1))
        radii = radius_dist.rvs(size=k, random_state=rng)
        bands = rng.integers(0, len(c["freq_bands_hz"]), size=k)
        start, stop = sample_windows(
            k, c["n_samples"], c["duration_samples"], c["overlap_frac"], rng
        )
        spatial = dict(
            N_src=k,
            R_mm=radii,
            A0=np.ones(k),
            min_dist=c["min_separation_mm"],
            sigma_mm_scale=c["sigma_mm_scale"],
        )
        try:
            centers, activation = spatial_activation(spatial, geometry, rng)
            break
        except RuntimeError:
            if attempt == c["max_attempts"] - 1:
                raise RuntimeError(
                    f"seed {seed}: centroid placement failed after {attempt+1} attempts"
                )
    temporal = dict(
        c["temporal"],
        N_src=k,
        ntpoints=c["n_samples"],
        t_start_sources=start,
        t_stop_sources=stop,
        frec_source_band=np.array(c["freq_bands_hz"])[bands] / (c["sfreq"] / 2),
    )
    courses = Temporal_Activation(temporal, rng)
    g_rng_state = copy.deepcopy(rng.bit_generator.state)
    perturbed, rad, tan, total = perturb_leadfield(g, c["beta_rad"], c["beta_tan"], rng)
    # Associative product avoids materializing a (vertices,time) activation matrix.
    topographies = perturbed @ activation.T
    clean = topographies @ courses
    noise = Generate_Brain_Noise(
        perturbed, dict(c["noise"], Fs=c["sfreq"], ntpoints=c["n_samples"]), rng
    )
    scale = 1 / (clean.reshape(-1).std(ddof=1) + 1e-12)
    y = clean * scale + c["noise_level"] * noise
    if apply_filter:
        y = filter_recording(y, c)
    snr = []
    for j in range(k):
        win = slice(start[j], stop[j])
        signal_power = (
            scale**2
            * np.sum(topographies[:, j] ** 2)
            * np.mean(courses[j, win] ** 2)
            / len(g)
        )
        noise_power = c["noise_level"] ** 2 * np.mean(noise[:, win] ** 2)
        snr.append(signal_power / (noise_power + 1e-12))

    def pad(values, fill):
        values = np.asarray(values)
        out = np.full((c["k_max"], *values.shape[1:]), fill, dtype=values.dtype)
        out[:k] = values
        return out

    return dict(
        Y=y.astype("f4"),
        theta=pad(geometry.vertices[centers], np.nan).astype("f4"),
        centers=pad(centers, -1),
        radii=pad(radii, np.nan),
        band_idx=pad(bands, -1),
        t_start=pad(start, -1),
        t_stop=pad(stop, -1),
        snr_per_source=pad(snr, np.nan),
        mask=np.arange(c["k_max"]) < k,
        K=k,
        seed=int(seed),
        placement_attempts=attempt + 1,
        leadfield_rng_state=g_rng_state,
        leadfield_error_mean=np.array([rad.mean(), tan.mean(), total.mean()]),
        filtered=apply_filter,
    )


def reconstruct_leadfield(g, record, config):
    rng = np.random.default_rng()
    rng.bit_generator.state = record["leadfield_rng_state"]
    return perturb_leadfield(g, config["beta_rad"], config["beta_tan"], rng)[0]


def xyz_statistics_masked(theta, mask):
    points = np.asarray(theta, dtype="f8")[mask]
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("active coordinates must be finite and nonempty")
    return points.mean(0).astype("f4"), (points.std(0) + 1e-3).astype("f4")


def normalize_theta(theta, mask, mean, std):
    return np.where(mask[..., None], (theta - mean) / std, 0.0).astype("f4")


def source_state_statistics(theta, radii, mask):
    """Train-only component statistics for ``(x, y, z, log radius)``."""
    theta = np.asarray(theta, dtype="f8")
    radii = np.asarray(radii, dtype="f8")
    mask = np.asarray(mask, dtype=bool)
    if theta.shape != (*mask.shape, 3) or radii.shape != mask.shape:
        raise ValueError("source state and mask shapes do not match")
    if (
        not mask.any()
        or not np.isfinite(theta[mask]).all()
        or not np.isfinite(radii[mask]).all()
        or (radii[mask] <= 0).any()
    ):
        raise ValueError(
            "active source coordinates and radii must be finite and positive"
        )
    points = np.concatenate([theta[mask], np.log(radii[mask])[:, None]], axis=1)
    return points.mean(0).astype("f4"), (points.std(0) + 1e-3).astype("f4")


def normalize_source_state(theta, radii, mask, mean, std):
    """Return finite padded tokens with a normalized log-radius component."""
    theta = np.asarray(theta, dtype="f4")
    radii = np.asarray(radii, dtype="f4")
    mask = np.asarray(mask, dtype=bool)
    mean, std = np.asarray(mean, dtype="f4"), np.asarray(std, dtype="f4")
    if (
        theta.shape != (*mask.shape, 3)
        or radii.shape != mask.shape
        or mean.shape != (4,)
        or std.shape != (4,)
        or (std <= 0).any()
        or (radii[mask] <= 0).any()
    ):
        raise ValueError("invalid source state normalization inputs")
    safe_radii = np.where(mask, radii, 1.0)
    state = np.concatenate([theta, np.log(safe_radii)[..., None]], axis=-1)
    return np.where(mask[..., None], (state - mean) / std, 0.0).astype("f4")


def denormalize_source_state(state, mask, mean, std, *, radius_bounds=None):
    """Decode normalized tokens into xyz millimetres and positive radii."""
    state = np.asarray(state, dtype="f4")
    mask = np.asarray(mask, dtype=bool)
    mean, std = np.asarray(mean, dtype="f4"), np.asarray(std, dtype="f4")
    if state.shape != (*mask.shape, 4) or mean.shape != (4,) or std.shape != (4,):
        raise ValueError("invalid normalized source state")
    decoded = state * std + mean
    theta = np.where(mask[..., None], decoded[..., :3], np.nan).astype("f4")
    radii = np.exp(decoded[..., 3])
    if radius_bounds is not None:
        lo, hi = radius_bounds
        if not 0 < lo < hi:
            raise ValueError("invalid radius bounds")
        radii = np.clip(radii, lo, hi)
    radii = np.where(mask, radii, np.nan).astype("f4")
    return theta, radii


def unknown_k_flow_batch(theta, mask, generator, permute):
    import torch

    theta = theta.masked_fill(~mask[..., None], 0)
    if permute:
        keys = torch.rand(mask.shape, generator=generator).masked_fill(~mask, 2.0)
        permutation = keys.argsort(dim=1)
        theta = torch.gather(theta, 1, permutation[..., None].expand_as(theta))
    theta0 = torch.randn(
        theta.shape, generator=generator, device=theta.device, dtype=theta.dtype
    )
    t = torch.rand(
        theta.shape[0], generator=generator, device=theta.device, dtype=theta.dtype
    )
    active_path = (1 - t[:, None, None]) * theta0 + t[:, None, None] * theta
    xt = torch.where(mask[..., None], active_path, theta0)
    target = torch.where(mask[..., None], theta - theta0, 0.0)
    return t, xt, target
