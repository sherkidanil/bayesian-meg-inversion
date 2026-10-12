#!/usr/bin/env python3
"""Numerically compare the adapter with reviewed function bodies of the supplied file."""
import argparse
import ast
import json
from pathlib import Path
import time

import mne
import networkx as nx
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import gamma, truncnorm

from megflow.dataset import sha256, write_json
from megflow.geometry import CorticalGeometry
from megflow.v15 import default_simulation_config, simulate
from megflow.v15_dataset import cache_distances_float64


def reference(g, geom, seed, c, functions):
    rng = np.random.default_rng(seed)
    dist = truncnorm(
        (c["radius_min"] - c["radius_mu"]) / c["radius_sigma"],
        (c["radius_max"] - c["radius_mu"]) / c["radius_sigma"],
        loc=c["radius_mu"],
        scale=c["radius_sigma"],
    )
    graph = nx.from_scipy_sparse_array(geom.adjacency)
    for attempt in range(c["max_attempts"]):
        k = int(rng.integers(c["k_min"], c["k_max"] + 1))
        radii = dist.rvs(size=k, random_state=rng)
        bands = rng.integers(0, len(c["freq_bands_hz"]), size=k)
        start, stop = functions["Sample_Activation_Windows"](
            k, c["n_samples"], c["duration_samples"], c["overlap_frac"], rng
        )
        try:
            centers, spatial = functions["Spatial_Activation"](
                dict(
                    N_src=k,
                    R_mm=radii,
                    A0=np.ones(k),
                    min_dist=c["min_separation_mm"],
                    sigma_mm_scale=c["sigma_mm_scale"],
                ),
                graph,
                rng,
            )
            break
        except RuntimeError:
            if attempt == c["max_attempts"] - 1:
                raise
    temporal = functions["Temporal_Activation"](
        dict(
            c["temporal"],
            N_src=k,
            ntpoints=c["n_samples"],
            t_start_sources=start,
            t_stop_sources=stop,
            frec_source_band=np.array(c["freq_bands_hz"])[bands] / (c["sfreq"] / 2),
        ),
        rng,
    )
    activation = temporal.T @ spatial
    perturbed, *_ = functions["Modeling_Liedfield_Error"](
        g, c["beta_rad"], c["beta_tan"], rng
    )
    clean = perturbed @ activation.T
    noise = functions["Generate_Brain_Noise"](
        perturbed, dict(c["noise"], Fs=c["sfreq"], ntpoints=c["n_samples"]), rng
    )
    y = clean / (clean.reshape(-1).std(ddof=1) + 1e-12) + c["noise_level"] * noise
    raw = mne.io.RawArray(
        y,
        mne.create_info([f"M{i}" for i in range(len(g))], c["sfreq"], ch_types="mag"),
        verbose=False,
    )
    f = c["filter"]
    raw.filter(
        f["l_freq"],
        f["h_freq"],
        method="iir",
        iir_params=dict(order=f["order"], ftype=f["ftype"], output="sos"),
        verbose=False,
    )
    snr = []
    scale = 1 / (clean.reshape(-1).std(ddof=1) + 1e-12)
    for j in range(k):
        b = perturbed @ spatial[j]
        win = slice(start[j], stop[j])
        p_sig = scale**2 * np.sum(b**2) * np.mean(temporal[j, win] ** 2) / len(g)
        p_noi = c["noise_level"] ** 2 * np.mean(noise[:, win] ** 2)
        snr.append(p_sig / (p_noi + 1e-12))
    return dict(
        Y=raw.get_data().astype("f4"),
        K=k,
        radii=radii,
        centers=centers,
        band_idx=bands,
        t_start=start,
        t_stop=stop,
        snr_per_source=np.array(snr),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anatomy", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        sha256(args.reference)
        != "e43a62983f5d8884040c24b3b25bb55060cff1b43aaf94a6502060da598a0e84"
    ):
        raise ValueError("reference checksum")
    geom, g, provenance = CorticalGeometry.load_bundle(
        args.anatomy, allow_fragmented_hemispheres=True
    )
    cache_distances_float64(geom, args.output.parent / "parity-cache")
    tree = ast.parse(args.reference.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(
        np=np,
        nx=nx,
        butter=butter,
        filtfilt=filtfilt,
        gamma=gamma,
        N_vert=len(geom.vertices),
    )
    exec(compile(tree, str(args.reference), "exec"), ns)
    rows = []
    for seed, kmax, length in [
        (0, 5, 2000),
        (17, 5, 2000),
        (303, 5, 2000),
        (0, 10, 4000),
    ]:
        cfg = default_simulation_config()
        cfg.update(k_max=kmax, n_samples=length)
        t = time.monotonic()
        a = simulate(g, geom, seed, cfg)
        fast = time.monotonic() - t
        t = time.monotonic()
        b = reference(g, geom, seed, cfg, ns)
        slow = time.monotonic() - t
        assert a["K"] == b["K"]
        for key in ["centers", "band_idx", "t_start", "t_stop", "radii"]:
            np.testing.assert_array_equal(a[key][a["mask"]], b[key])
        np.testing.assert_allclose(
            a["snr_per_source"][a["mask"]], b["snr_per_source"], rtol=1e-11, atol=1e-11
        )
        np.testing.assert_allclose(a["Y"], b["Y"], rtol=0, atol=2e-6)
        row = dict(
            seed=seed,
            k_max=kmax,
            n_samples=length,
            K=a["K"],
            max_abs_y=float(abs(a["Y"] - b["Y"]).max()),
            adapter_seconds=fast,
            reference_seconds=slow,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_json(
        args.output,
        dict(
            passed=True,
            anatomy_sha256=sha256(args.anatomy),
            reference_sha256=sha256(args.reference),
            checks=rows,
            anatomy=provenance,
            numpy=np.__version__,
            mne=mne.__version__,
        ),
    )


if __name__ == "__main__":
    main()
