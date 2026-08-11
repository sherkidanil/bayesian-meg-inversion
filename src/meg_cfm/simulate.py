#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_dataset.py

Generate MEG simulation dataset (MNE sample) and SAVE mesh-adjacency edges
for proper geodesic distance computation on the cortical surface.

Output .npz contains:
  - Ys:            [N, M, T] float32
  - thetas:        [N, 3*K]  float32 (centroid coords in meters)
  - centroids_idx: [N, K]    int64   (true centroid vertex indices)
  - vertices:      [N_vert,3] float32 (meters)
  - edge_index:    [E,2] int32        (undirected edges saved once, i>j)
  - edge_weight_mm:[E]   float32      (edge lengths in mm)
  - params_spatial / params_temporal / params_noise (pickled dicts)
  - N_src, Fs, meta

Requirements:
  pip install numpy scipy mne networkx pyyaml tqdm
"""

import os
import argparse
from pathlib import Path
import numpy as np
import networkx as nx
import yaml
from tqdm import tqdm

from scipy.signal import butter, filtfilt
from scipy.stats import gamma

import mne
from mne.datasets import sample
from mne import spatial_src_adjacency


# ============================================================
# 1) Simulation functions (same logic as your code)
# ============================================================

def Spatial_Activation(params, Graph, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    N_src = params["N_src"]
    min_dist = params["min_dist"]
    R_mm = params["R_mm"]
    sigma_mm = R_mm * params["sigma_mm_scale"]
    A0 = params["A0"]

    N_vert = len(Graph)  # must include ALL vertices as nodes

    centroids = np.zeros(N_src, dtype=int)
    spatial_activation = np.zeros((N_src, N_vert), dtype=float)

    # --- first source ---
    centroids[0] = rng.integers(N_vert)
    d = nx.single_source_dijkstra_path_length(Graph, centroids[0], weight="weight")
    d_vec = np.full(N_vert, np.inf, dtype=float)
    for k, v in d.items():
        d_vec[k] = v

    idx_region = d_vec <= R_mm[0]
    r = d_vec[idx_region]
    spatial_activation[0, idx_region] = A0[0] * np.exp(-0.5 * (r / sigma_mm[0]) ** 2)

    # --- other sources ---
    for j in range(1, N_src):
        if j == 1:
            idx_centroids = d_vec >= min_dist
        else:
            idx_centroids = idx_centroids & (d_vec >= min_dist)

        candidates = np.where(idx_centroids)[0]
        if candidates.size == 0:
            candidates = np.arange(N_vert, dtype=int)

        centroids[j] = rng.choice(candidates)

        d = nx.single_source_dijkstra_path_length(Graph, centroids[j], weight="weight")
        d_vec = np.full(N_vert, np.inf, dtype=float)
        for k, v in d.items():
            d_vec[k] = v

        idx_region = d_vec <= R_mm[j]
        r = d_vec[idx_region]
        spatial_activation[j, idx_region] = A0[j] * np.exp(-0.5 * (r / sigma_mm[j]) ** 2)

    return centroids, spatial_activation


def Temporal_Activation(params, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    N_src = params["N_src"]
    ntpoints = params["ntpoints"]
    t_start_sources = params["t_start_sources"]
    t_stop_sources = params["t_stop_sources"]
    frec_source_band = params["frec_source_band"]  # normalized to Nyquist
    pad = params["pad"]
    shape = params["shape"]
    scale = params["scale"]
    qL = params["qL"]
    qR = params["qR"]

    zL = gamma.ppf(qL, a=shape, scale=scale)
    zR = gamma.ppf(qR, a=shape, scale=scale)

    el_time_activation = np.zeros((N_src, ntpoints), dtype=float)

    for j in range(N_src):
        start = int(t_start_sources[j])
        stop = int(t_stop_sources[j])
        nseg = stop - start
        if nseg <= 1:
            continue

        q = rng.standard_normal(nseg + 2 * pad)

        b, a = butter(3, frec_source_band[j, :], btype="bandpass")
        s = filtfilt(b, a, q)
        s = s / (np.mean(frec_source_band[j, :]) + 1e-12)

        x = np.arange(start, stop)
        z = zL + (zR - zL) * (x - start) / nseg
        mask = ((zR - zL) / nseg) * (
            gamma.pdf(z, a=shape, scale=scale)
            / (gamma.cdf(zR, a=shape, scale=scale) - gamma.cdf(zL, a=shape, scale=scale))
        )

        mask = mask / (mask.std(ddof=1) + 1e-12)
        s_seg = s[:nseg]
        s_seg = s_seg / (s_seg.std(ddof=1) + 1e-12)

        el_time_activation[j, x] = mask * s_seg

    return el_time_activation


def Generate_Brain_Noise(G, params, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    Fs = float(params["Fs"])
    ntpoints = int(params["ntpoints"])
    window = float(params["window"])
    N_noise_sources = int(params["N_noise_sources"])
    pad = int(params["pad"])

    def Generate_noise_trial(G, noise_trial_length, add_len, N_noise_sources, Fs, rng):
        Nsrc = G.shape[1]
        SrcIndex = rng.integers(0, Nsrc, size=N_noise_sources)
        q = rng.standard_normal((N_noise_sources, noise_trial_length + 2 * add_len))

        nyq = Fs / 2.0
        alpha_band = np.array([8.0, 12.0]) / nyq
        beta_band = np.array([15.0, 30.0]) / nyq
        gamma1_band = np.array([30.0, 50.0]) / nyq
        gamma2_band = np.array([50.0, 70.0]) / nyq

        b_alpha, a_alpha = butter(3, alpha_band, btype="bandpass")
        b_beta, a_beta = butter(3, beta_band, btype="bandpass")
        b_gamma1, a_gamma1 = butter(3, gamma1_band, btype="bandpass")
        b_gamma2, a_gamma2 = butter(3, gamma2_band, btype="bandpass")

        SourceNoise = (
            filtfilt(b_alpha, a_alpha, q, axis=1) / (np.mean(alpha_band) + 1e-12)
            + filtfilt(b_beta, a_beta, q, axis=1) / (np.mean(beta_band) + 1e-12)
            + filtfilt(b_gamma1, a_gamma1, q, axis=1) / (np.mean(gamma1_band) + 1e-12)
            + filtfilt(b_gamma2, a_gamma2, q, axis=1) / (np.mean(gamma2_band) + 1e-12)
        )

        BrainNoiseTrial = (
            G[:, SrcIndex] @ SourceNoise[:, add_len:add_len + noise_trial_length] / max(1, N_noise_sources)
        )
        return BrainNoiseTrial

    noise_trial_length = int(window * Fs)
    noise_trials_count = int(np.floor(ntpoints / noise_trial_length + 1))

    Noise = np.zeros((G.shape[0], noise_trials_count * noise_trial_length), dtype=float)

    for ntidx in range(noise_trials_count):
        start = ntidx * noise_trial_length
        stop = start + noise_trial_length
        Noise[:, start:stop] = Generate_noise_trial(
            G, noise_trial_length, pad, N_noise_sources, Fs, rng
        )

    Noise = Noise[:, :ntpoints]
    Noise = Noise / (Noise.reshape(-1).std(ddof=1) + 1e-12)
    return Noise


def simulate_single_sample(G, Graph, vertices, params_spatial, params_temporal, params_noise,
                           noise_level=0.3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    centroids_idx, spatial_activation = Spatial_Activation(params_spatial, Graph, rng=rng)
    el_time_activation = Temporal_Activation(params_temporal, rng=rng)

    # [ntpoints, N_vert]
    spatio_temporal_activation = el_time_activation.T @ spatial_activation

    # [M, ntpoints]
    Y_unnoise = G @ spatio_temporal_activation.T

    Noise = Generate_Brain_Noise(G, params_noise, rng=rng)

    Y_unnoise_norm = Y_unnoise / (Y_unnoise.reshape(-1).std(ddof=1) + 1e-12)
    Y = Y_unnoise_norm + float(noise_level) * Noise

    # theta = centroid coords (meters) flattened
    centroids_coords = vertices[centroids_idx, :]   # [K,3] meters
    theta = centroids_coords.reshape(-1)            # [3K]

    return {
        "Y": Y.astype(np.float32),
        "theta": theta.astype(np.float32),
        "centroids_idx": centroids_idx.astype(np.int64),
    }


# ============================================================
# 2) Forward model + mesh edges (for geodesic)
# ============================================================

def build_forward_vertices_edges(spacing="ico4", mindist=5.0):
    """Return leadfield G (magnetometers), vertices (meters), and mesh edges+weights (mm).
    Edge list corresponds to true mesh-adjacency (spatial_src_adjacency)."""

    print("[MNE] Building forward model + mesh adjacency...")
    data_path = Path(sample.data_path())
    subjects_dir = data_path / "subjects"
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    trans_fname = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"

    src = mne.setup_source_space("sample", spacing=spacing, subjects_dir=subjects_dir, add_dist=False)
    bem = mne.read_bem_solution(str(data_path / "subjects" / "sample" / "bem" / "sample-5120-bem-sol.fif"))

    fwd = mne.make_forward_solution(
        str(raw_fname),
        trans=str(trans_fname),
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=float(mindist),
    )
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, copy=True)
    src = fwd["src"]

    # magnetometers only
    fwd_mag = mne.pick_types_forward(fwd, meg="mag", eeg=False)
    G = fwd_mag["sol"]["data"].astype(np.float32)  # [M, N_vert]

    # vertices in the same ordering as forward/src: LH used vertices then RH used vertices
    rr_lh = src[0]["rr"][src[0]["vertno"]]
    rr_rh = src[1]["rr"][src[1]["vertno"]]
    vertices = np.concatenate([rr_lh, rr_rh], axis=0).astype(np.float32)  # meters
    N_vert = vertices.shape[0]

    # mesh adjacency edges
    adj = spatial_src_adjacency(src)  # sparse (N_vert,N_vert), adjacency on cortical mesh
    ii, jj = adj.nonzero()

    # keep each undirected edge once (i > j)
    mask = ii > jj
    ii = ii[mask].astype(np.int32)
    jj = jj[mask].astype(np.int32)

    # weights = Euclidean edge length in mm (since edges are mesh-neighbors)
    w_mm = (np.linalg.norm(vertices[ii] - vertices[jj], axis=1) * 1000.0).astype(np.float32)

    edge_index = np.stack([ii, jj], axis=1).astype(np.int32)  # [E,2]
    edge_weight_mm = w_mm                                     # [E]

    print(f"[MNE] G shape: {G.shape}")
    print(f"[MNE] vertices: {vertices.shape} (meters)")
    print(f"[MNE] edges: {edge_index.shape} (unique undirected)")
    return G, vertices, edge_index, edge_weight_mm


def build_networkx_graph(N_vert, edge_index, edge_weight_mm):
    """Build weighted NetworkX graph for Spatial_Activation (Dijkstra in mm)."""
    Graph = nx.Graph()
    Graph.add_nodes_from(range(int(N_vert)))
    u = edge_index[:, 0].astype(int)
    v = edge_index[:, 1].astype(int)
    w = edge_weight_mm.astype(float)
    for ui, vi, wi in zip(u, v, w):
        Graph.add_edge(ui, vi, weight=wi)
    return Graph


# ============================================================
# 3) Dataset generation + saving (WITH edges for geodesic)
# ============================================================

def generate_and_save_dataset(
    dataset_path,
    n_samples,
    G,
    Graph,
    vertices,
    edge_index,
    edge_weight_mm,
    params_spatial,
    params_temporal,
    params_noise,
    noise_level=0.3,
    seed=0,
    meta=None,
):
    rng = np.random.default_rng(int(seed))

    first = simulate_single_sample(
        G, Graph, vertices,
        params_spatial, params_temporal, params_noise,
        noise_level=noise_level, rng=rng
    )
    Y0, th0, c0 = first["Y"], first["theta"], first["centroids_idx"]
    M, T = Y0.shape
    theta_dim = th0.shape[0]
    K = c0.shape[0]

    print(f"[Dataset] First sample shapes: Y={Y0.shape}, theta={th0.shape}, K={K}")

    # NOTE: this is RAM-hungry for large N. If needed, move to memmap.
    Ys = np.empty((n_samples, M, T), dtype=np.float32)
    thetas = np.empty((n_samples, theta_dim), dtype=np.float32)
    centroids_idx_all = np.empty((n_samples, K), dtype=np.int64)

    Ys[0] = Y0
    thetas[0] = th0
    centroids_idx_all[0] = c0

    for k in tqdm(range(1, n_samples), desc="Simulating"):
        sample_k = simulate_single_sample(
            G, Graph, vertices,
            params_spatial, params_temporal, params_noise,
            noise_level=noise_level, rng=rng
        )
        Ys[k] = sample_k["Y"]
        thetas[k] = sample_k["theta"]
        centroids_idx_all[k] = sample_k["centroids_idx"]

    meta = {} if meta is None else dict(meta)
    meta.update({
        "vertices_unit": "m",
        "edge_weight_unit": "mm",
        "edge_type": "mesh_adjacency",
    })

    print(f"[Dataset] Saving -> {dataset_path}")
    np.savez_compressed(
        dataset_path,
        Ys=Ys,
        thetas=thetas,
        centroids_idx=centroids_idx_all,
        vertices=vertices,
        edge_index=edge_index,
        edge_weight_mm=edge_weight_mm,
        params_spatial=params_spatial,
        params_temporal=params_temporal,
        params_noise=params_noise,
        N_src=np.array(params_spatial["N_src"], dtype=np.int64),
        Fs=np.array(params_temporal.get("Fs", 0.0), dtype=np.float32),
        meta=meta,
    )
    print("[Dataset] Done.")


# ============================================================
# 4) Config parsing
# ============================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml", help="YAML config")
    ap.add_argument("--out", type=str, default=None, help="Override output npz path")
    ap.add_argument("--force", action="store_true", help="Overwrite existing file")
    ap.add_argument("--spacing", type=str, default="ico4", help="MNE source space spacing")
    ap.add_argument("--mindist", type=float, default=5.0, help="MNE mindist in mm")
    args = ap.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    sim_cfg = cfg.get("simulation", {})
    spatial_cfg = sim_cfg.get("spatial", {})
    temporal_cfg = sim_cfg.get("temporal", {})
    noise_cfg = sim_cfg.get("noise", {})

    n_samples = int(data_cfg.get("n_samples", 10_000))
    N_src = int(data_cfg.get("N_src", 3))
    duration = float(data_cfg.get("duration", 1.8))
    Fs = float(data_cfg.get("Fs", 1000.0))
    noise_level = float(data_cfg.get("noise_level", 0.3))
    seed = int(cfg.get("training", {}).get("seed", 42))

    ntpoints = int(round(Fs * duration))

    # ---- spatial params ----
    R_mm = np.array(spatial_cfg.get("R_mm", [10.0, 8.0, 10.0]), dtype=float)
    A0 = np.array(spatial_cfg.get("A0", [1.0, 1.0, 1.0]), dtype=float)
    if len(R_mm) != N_src or len(A0) != N_src:
        raise ValueError("R_mm and A0 must have length N_src")

    params_spatial = {
        "N_src": N_src,
        "min_dist": float(spatial_cfg.get("min_dist", 30.0)),
        "R_mm": R_mm,
        "sigma_mm_scale": float(spatial_cfg.get("sigma_mm_scale", 1.0/3.0)),
        "A0": A0,
    }

    # ---- temporal params ----
    t_start_sec = np.array(temporal_cfg.get("t_start_sources", [0.0, 0.6, 1.0]), dtype=float)
    t_stop_sec = np.array(temporal_cfg.get("t_stop_sources", [0.6, 1.2, 1.7]), dtype=float)
    frec_band_hz = np.array(
        temporal_cfg.get("frec_source_band", [[8.0, 12.0], [15.0, 20.0], [4.0, 8.0]]),
        dtype=float,
    )
    if t_start_sec.shape[0] != N_src or t_stop_sec.shape[0] != N_src or frec_band_hz.shape[0] != N_src:
        raise ValueError("Temporal arrays must have first dimension N_src")

    params_temporal = {
        "N_src": N_src,
        "ntpoints": ntpoints,
        "t_start_sources": (t_start_sec * Fs).astype(int),
        "t_stop_sources": (t_stop_sec * Fs).astype(int),
        "frec_source_band": frec_band_hz / (Fs / 2.0),  # normalized
        "pad": int(temporal_cfg.get("pad", 300)),
        "shape": float(temporal_cfg.get("shape", 6.0)),
        "scale": float(temporal_cfg.get("scale", 4.0)),
        "qL": float(temporal_cfg.get("qL", 1e-5)),
        "qR": float(temporal_cfg.get("qR", 0.999)),
        "Fs": Fs,  # store explicitly for convenience
    }

    # ---- noise params ----
    params_noise = {
        "Fs": Fs,
        "ntpoints": ntpoints,
        "window": float(noise_cfg.get("window", 0.1)),
        "N_noise_sources": int(noise_cfg.get("N_noise_sources", 100)),
        "pad": int(noise_cfg.get("pad", 300)),
    }

    # output path
    dataset_path = args.out or data_cfg.get("dataset_path", f"meg_flow_dataset_{n_samples}_with_edges.npz")
    dataset_path = str(dataset_path)

    if os.path.exists(dataset_path) and not args.force:
        print(f"[Main] Exists: {dataset_path} (use --force to overwrite)")
        return

    # build forward + vertices + mesh edges
    G, vertices, edge_index, edge_weight_mm = build_forward_vertices_edges(
        spacing=args.spacing, mindist=args.mindist
    )
    Graph = build_networkx_graph(vertices.shape[0], edge_index, edge_weight_mm)

    meta = {
        "seed": seed,
        "n_samples": n_samples,
        "spacing": args.spacing,
        "mindist": args.mindist,
        "note": "Dataset generated with mesh-adjacency edges for true surface geodesics.",
    }

    generate_and_save_dataset(
        dataset_path=dataset_path,
        n_samples=n_samples,
        G=G,
        Graph=Graph,
        vertices=vertices,
        edge_index=edge_index,
        edge_weight_mm=edge_weight_mm,
        params_spatial=params_spatial,
        params_temporal=params_temporal,
        params_noise=params_noise,
        noise_level=noise_level,
        seed=seed,
        meta=meta,
    )


if __name__ == "__main__":
    main()