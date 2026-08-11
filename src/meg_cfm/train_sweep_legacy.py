"""Historical Optuna sweep — RETAINED FOR THE RECORD, DO NOT REUSE AS-IS.

The validation metric written to ``trial.set_user_attr("eval_mae_geo_mm", ...)``
below selects, among several posterior draws, the one closest to GROUND TRUTH.
That is an oracle upper bound, not a localiser score. Every historical number
derived from this attribute is contaminated; see docs/findings.md section 1.

The search objective itself (validation loss) is honest, so trial rankings by
objective value remain usable. Kept unmodified so the contamination stays
auditable.
"""

import os
import argparse
import time
import numpy as np
from tqdm import tqdm

import optuna

# CUDA_VISIBLE_DEVICES should be set by the shell script before running Python
# Check it here for debugging
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
print(f"[Worker] CUDA_VISIBLE_DEVICES (from env): {cuda_visible}")

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from metrics import (
    build_knn_geodesic,
    build_graph_from_edges,
    project_coords_to_vertices,
    compute_metrics_one,
    aggregate_metrics,
)

# -----------------------------
# Data helpers
# -----------------------------
def pool_time(Y: np.ndarray, pool: int) -> np.ndarray:
    if pool == 1:
        return Y
    B, C, T = Y.shape
    assert T % pool == 0
    return Y.reshape(B, C, T // pool, pool).mean(axis=-1)


def make_batch(
    rng: np.random.Generator,
    Ys: np.ndarray,
    thetas: np.ndarray,
    idx_pool: np.ndarray,
    batch_size: int,
    pool: int,
    permute_sources: bool,
    n_src: int = 3,
):
    idx = rng.choice(idx_pool, size=batch_size, replace=False)
    Y = np.asarray(Ys[idx], dtype=np.float32)               # [B,102,1800]
    Yp = pool_time(Y, pool=pool).astype(np.float32)         # [B,102,T']
    theta = np.asarray(thetas[idx], dtype=np.float32)       # [B,3*n_src]

    if permute_sources:
        B = theta.shape[0]
        blocks = theta.reshape(B, n_src, 3)
        perms = np.array([rng.permutation(n_src) for _ in range(B)], dtype=np.int64)
        blocks = blocks[np.arange(B)[:, None], perms]
        theta = blocks.reshape(B, 3 * n_src)

    return Yp, theta


def get_act(name: str):
    name = str(name).lower()
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    if name == "gelu":
        return jax.nn.gelu
    if name == "tanh":
        return jnp.tanh
    raise ValueError(f"Unknown activation: {name}")


# -----------------------------
# Model
# -----------------------------
class CNNEncoder(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d
    conv3: eqx.nn.Conv1d
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    norm3: eqx.nn.LayerNorm
    proj: eqx.nn.Linear
    act: callable

    def __init__(self, *, key, in_ch, cond_dim, width1, width2, kernel, stride, act):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        pad = kernel // 2
        self.conv1 = eqx.nn.Conv1d(in_ch, width1, kernel_size=kernel, stride=stride, padding=pad, key=k1)
        self.conv2 = eqx.nn.Conv1d(width1, width2, kernel_size=kernel, stride=stride, padding=pad, key=k2)
        self.conv3 = eqx.nn.Conv1d(width2, width2, kernel_size=kernel, stride=stride, padding=pad, key=k3)

        self.norm1 = eqx.nn.LayerNorm(width1, use_weight=True, use_bias=True)
        self.norm2 = eqx.nn.LayerNorm(width2, use_weight=True, use_bias=True)
        self.norm3 = eqx.nn.LayerNorm(width2, use_weight=True, use_bias=True)

        self.proj = eqx.nn.Linear(width2, cond_dim, key=k4)
        self.act = get_act(act)

    def _norm_time(self, x, norm):
        x = jnp.swapaxes(x, 0, 1)
        x = jax.vmap(norm)(x)
        return jnp.swapaxes(x, 0, 1)

    def __call__(self, Y):
        def f(y):
            x = self.conv1(y); x = self._norm_time(x, self.norm1); x = self.act(x)
            x = self.conv2(x); x = self._norm_time(x, self.norm2); x = self.act(x)
            x = self.conv3(x); x = self._norm_time(x, self.norm3); x = self.act(x)
            x = x.mean(axis=-1)
            return self.proj(x)
        return jax.vmap(f)(Y)


class VectorField(eqx.Module):
    mlp: eqx.nn.MLP
    def __call__(self, t, x, cond):
        inp = jnp.concatenate([x, cond, t[:, None]], axis=-1)
        return jax.vmap(self.mlp)(inp)


class ConditionalFlow(eqx.Module):
    enc: CNNEncoder
    vf: VectorField
    def __call__(self, t, x, Y):
        cond = self.enc(Y)
        return self.vf(t, x, cond)


# -----------------------------
# Flow matching loss with optional min-permutation
# -----------------------------
_PERMS_3 = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]], dtype=np.int32)

def fm_loss(model, key, Y, theta, theta_mean, theta_std, min_perm_loss: bool, n_src=3):
    key0, keyt = jax.random.split(key, 2)

    # base noise
    x0 = jax.random.normal(key0, shape=theta.shape)

    B = theta.shape[0]
    t = jax.random.uniform(keyt, shape=(B,), minval=0.0, maxval=1.0)

    def mse_for_theta(th):
        x1 = (th - theta_mean) / theta_std
        xt = (1.0 - t)[:, None] * x0 + t[:, None] * x1
        target = x1 - x0
        v = model(t, xt, Y)
        return jnp.mean((v - target) ** 2)

    if not min_perm_loss:
        return mse_for_theta(theta)

    assert n_src == 3 and theta.shape[1] == 9
    blocks = theta.reshape(B, 3, 3)
    losses = []
    for p in _PERMS_3:
        thp = blocks[:, p, :].reshape(B, 9)
        losses.append(mse_for_theta(thp))
    return jnp.min(jnp.stack(losses))


def make_train_step(tx, theta_mean, theta_std, min_perm_loss, n_src):
    @eqx.filter_jit
    def step(model, opt_state, key, Y, theta):
        key, subkey = jax.random.split(key, 2)
        loss, grads = eqx.filter_value_and_grad(fm_loss)(
            model, subkey, Y, theta, theta_mean, theta_std, min_perm_loss, n_src
        )
        updates, opt_state = tx.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, key, loss
    return step


@eqx.filter_jit
def eval_loss(model, key, Y, theta, theta_mean, theta_std, min_perm_loss, n_src):
    return fm_loss(model, key, Y, theta, theta_mean, theta_std, min_perm_loss, n_src)


# -----------------------------
# Inference (fast Euler) for metrics subset
# -----------------------------
def euler_sample(model: ConditionalFlow, Y: jnp.ndarray, key: jax.Array, theta_dim: int, steps: int):
    dt = 1.0 / steps
    x = jax.random.normal(key, (1, theta_dim))
    for i in range(steps):
        t = jnp.full((1,), i * dt, dtype=jnp.float32)
        v = model(t, x, Y)
        x = x + dt * v
    return x  # normalized theta


# -----------------------------
# Objective factory
# -----------------------------
def objective_factory(
    Ys, thetas, centroids, vertices, R_mm, A0,
    train_idx, val_idx,
    seed_np=42,
    knn_graph=None,
    thresholds_mm=(5, 10, 15, 20, 30),
):
    rng_np = np.random.default_rng(seed_np)

    theta_mean = np.asarray(thetas[train_idx].mean(axis=0), dtype=np.float32)
    theta_std = np.asarray(thetas[train_idx].std(axis=0) + 1e-6, dtype=np.float32)
    theta_mean_j = jnp.asarray(theta_mean)
    theta_std_j = jnp.asarray(theta_std)

    def objective(trial: optuna.Trial):
        # ---- search space (wide) ----
        pool = trial.suggest_categorical("pool", [5, 6, 10, 12, 15])
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 192, 256])
        steps = trial.suggest_categorical("steps", [2000, 3000, 5000, 8000, 12000])

        cond_dim = trial.suggest_categorical("cond_dim", [128, 192, 256, 384, 512])
        cnn_width1 = trial.suggest_categorical("cnn_width1", [64, 96, 128, 160])
        cnn_width2 = trial.suggest_categorical("cnn_width2", [128, 192, 256, 320, 384])
        cnn_kernel = trial.suggest_categorical("cnn_kernel", [7, 9, 15])
        cnn_stride = trial.suggest_categorical("cnn_stride", [1, 2])
        enc_act = trial.suggest_categorical("enc_act", ["relu", "silu", "gelu"])

        vf_width = trial.suggest_categorical("vf_width", [192, 256, 384, 512, 768])
        vf_depth = trial.suggest_categorical("vf_depth", [1, 2, 3, 4])
        vf_act = trial.suggest_categorical("vf_act", ["tanh", "silu", "gelu"])

        permute_sources = trial.suggest_categorical("permute_sources", [True, False])
        min_perm_loss = trial.suggest_categorical("min_perm_loss", [True, False])

        use_wd = trial.suggest_categorical("use_weight_decay", [False, True])
        wd = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True) if use_wd else 0.0
        clip_norm = trial.suggest_categorical("clip_norm", [0.0, 1.0, 2.0, 5.0])

        # eval settings (not optimized, just to compute extra metrics occasionally)
        ode_steps = trial.suggest_categorical("ode_steps", [32, 64, 96])
        post_samples = trial.suggest_categorical("n_post_samples", [1, 3, 5])

        # ---- build model ----
        key = jax.random.PRNGKey(777 + trial.number)
        k1, k2 = jax.random.split(key, 2)

        enc = CNNEncoder(
            key=k1,
            in_ch=102,
            cond_dim=cond_dim,
            width1=cnn_width1,
            width2=cnn_width2,
            kernel=cnn_kernel,
            stride=cnn_stride,
            act=enc_act,
        )

        vf = VectorField(
            mlp=eqx.nn.MLP(
                in_size=9 + cond_dim + 1,
                out_size=9,
                width_size=vf_width,
                depth=vf_depth,
                activation=get_act(vf_act),
                key=k2,
            )
        )

        model = ConditionalFlow(enc=enc, vf=vf)

        # optimizer (clip correctly through chain)
        if clip_norm > 0:
            tx = optax.chain(
                optax.clip_by_global_norm(clip_norm),
                optax.adamw(lr, weight_decay=wd),
            )
        else:
            tx = optax.adamw(lr, weight_decay=wd)

        opt_state = tx.init(eqx.filter(model, eqx.is_array))
        train_step = make_train_step(tx, theta_mean_j, theta_std_j, min_perm_loss, n_src=3)

        # ---- training ----
        key_train = jax.random.PRNGKey(999 + trial.number)
        eval_every = max(200, steps // 10)
        val_batches = 5

        for step in range(1, steps + 1):
            Y_np, th_np = make_batch(
                rng_np, Ys, thetas, train_idx,
                batch_size=batch_size, pool=pool,
                permute_sources=permute_sources, n_src=3
            )
            model, opt_state, key_train, _ = train_step(
                model, opt_state, key_train,
                jnp.asarray(Y_np), jnp.asarray(th_np)
            )

            if step % eval_every == 0:
                # quick val estimate
                vkey = key_train
                vls = []
                for _ in range(val_batches):
                    Yv_np, tv_np = make_batch(
                        rng_np, Ys, thetas, val_idx,
                        batch_size=batch_size, pool=pool,
                        permute_sources=False, n_src=3
                    )
                    vkey, sub = jax.random.split(vkey, 2)
                    v = eval_loss(
                        model, sub,
                        jnp.asarray(Yv_np), jnp.asarray(tv_np),
                        theta_mean_j, theta_std_j,
                        min_perm_loss=False, n_src=3
                    )
                    vls.append(float(v))
                vloss = float(np.mean(vls))
                
                # Report intermediate value for pruning and visualization
                trial.report(vloss, step)
                
                # Log intermediate metrics
                print(f"[Trial {trial.number}] Step {step}/{steps}: val_loss={vloss:.6f}")
                
                if trial.should_prune():
                    print(f"[Trial {trial.number}] Pruned at step {step}")
                    raise optuna.TrialPruned()

        # ---- final val (more stable) ----
        vkey = key_train
        vls = []
        for _ in range(15):
            Yv_np, tv_np = make_batch(
                rng_np, Ys, thetas, val_idx,
                batch_size=batch_size, pool=pool,
                permute_sources=False, n_src=3
            )
            vkey, sub = jax.random.split(vkey, 2)
            v = eval_loss(
                model, sub, jnp.asarray(Yv_np), jnp.asarray(tv_np),
                theta_mean_j, theta_std_j,
                min_perm_loss=False, n_src=3
            )
            vls.append(float(v))
        vloss = float(np.mean(vls))

        # ---- extra metrics on a small subset (store in user_attrs) ----
        # Keep it small so sweep stays fast
        subset_n = 30
        subset_ids = rng_np.choice(val_idx, size=min(subset_n, len(val_idx)), replace=False)
        ms = []
        key_inf = jax.random.PRNGKey(12345 + trial.number)

        for sid in subset_ids:
            # true indices
            C_true = centroids[sid].astype(int)

            # sample posterior and take first sample only (or best-of-S)
            Y0 = Ys[sid:sid+1].astype(np.float32)
            Yp = pool_time(Y0, pool=pool).astype(np.float32)
            Yj = jnp.asarray(Yp)

            best_m = None
            best_score = float("inf")

            for s in range(post_samples):
                try:
                    key_inf, sub = jax.random.split(key_inf, 2)
                    x = euler_sample(model, Yj, sub, theta_dim=9, steps=ode_steps)[0]
                    th = (np.array(x).reshape(-1) * theta_std + theta_mean).reshape(3, 3)  # meters
                    C_pred = project_coords_to_vertices(th, vertices)

                    m = compute_metrics_one(
                        C_true=C_true,
                        C_pred=C_pred,
                        vertices_m=vertices,
                        knn_graph=knn_graph,
                        R_mm=R_mm,
                        A0=A0,
                        thresholds_mm=thresholds_mm,
                    )

                    if m is not None and np.isfinite(m.get("mae_geo_mm", np.inf)):
                        score = m["mae_geo_mm"]  # choose geo as primary
                        if score < best_score:
                            best_score = score
                            best_m = m
                except Exception as e:
                    print(f"[Trial {trial.number}] Warning: Failed to compute metrics for sample {sid}, attempt {s}: {e}")
                    continue

            # Only append if we got valid metrics
            if best_m is not None:
                ms.append(best_m)
            else:
                print(f"[Trial {trial.number}] Warning: No valid metrics for sample {sid}, skipping")

        # Check if we have any valid metrics before aggregating
        if not ms:
            print(f"[Trial {trial.number}] ERROR: No valid metrics computed, skipping eval metrics")
            # Return a high loss to indicate failure
            trial.set_user_attr("eval_failed", True)
            return vloss
        
        # Additional safety check: filter out any None values that might have slipped through
        ms = [m for m in ms if m is not None and isinstance(m, dict)]
        
        if not ms:
            print(f"[Trial {trial.number}] ERROR: No valid metrics after filtering, skipping eval metrics")
            trial.set_user_attr("eval_failed", True)
            return vloss
        
        agg = aggregate_metrics(ms)
        
        # Log all metrics
        print(f"[Trial {trial.number}] Final val_loss={vloss:.6f}")
        print(f"[Trial {trial.number}] Eval metrics:")
        print(f"  mae_geo_mm: {agg['mae_geo_mm']:.4f}")
        print(f"  mae_euc_mm: {agg['mae_euc_mm']:.4f}")
        print(f"  err_norm_geo_over_R: {agg['err_norm_geo_over_R']:.4f}")
        print(f"  err_excess_geo_mm: {agg['err_excess_geo_mm']:.4f}")
        print(f"  err_strong_geo_mm: {agg['err_strong_geo_mm']:.4f}")
        print(f"  err_weak_geo_mm: {agg['err_weak_geo_mm']:.4f}")
        for thr in thresholds_mm:
            print(f"  hit_geo@{int(thr)}mm: {agg[f'hit_geo@{int(thr)}mm']:.4f}")
            print(f"  hit_euc@{int(thr)}mm: {agg[f'hit_euc@{int(thr)}mm']:.4f}")
        
        # Store all metrics as user attributes for later analysis
        trial.set_user_attr("eval_mae_geo_mm", agg["mae_geo_mm"])
        trial.set_user_attr("eval_mae_euc_mm", agg["mae_euc_mm"])
        trial.set_user_attr("eval_err_norm_geo_over_R", agg["err_norm_geo_over_R"])
        trial.set_user_attr("eval_err_excess_geo_mm", agg["err_excess_geo_mm"])
        trial.set_user_attr("eval_err_strong_geo_mm", agg["err_strong_geo_mm"])
        trial.set_user_attr("eval_err_weak_geo_mm", agg["err_weak_geo_mm"])
        # store hit@ thresholds
        for thr in thresholds_mm:
            trial.set_user_attr(f"eval_hit_geo@{int(thr)}mm", agg[f"hit_geo@{int(thr)}mm"])
            trial.set_user_attr(f"eval_hit_euc@{int(thr)}mm", agg[f"hit_euc@{int(thr)}mm"])

        return vloss

    return objective


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--storage", type=str, default="sqlite:///optuna_meg_fm_mesh.db")
    p.add_argument("--study", type=str, default="meg_fm_cnn_mesh")
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--dataset_npz", type=str, default="/mnt/local/tester/meg/meg_flow_dataset_10000_new.npz")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # CUDA_VISIBLE_DEVICES should already be set by shell script before Python started
    # Verify it matches the --gpu argument
    env_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env_gpu != str(args.gpu):
        print(f"[Worker] WARNING: CUDA_VISIBLE_DEVICES mismatch!")
        print(f"[Worker] Environment: {env_gpu}, Argument: {args.gpu}")
        print(f"[Worker] Setting CUDA_VISIBLE_DEVICES={args.gpu} (but JAX may have already initialized)")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print(f"[Worker] CUDA_VISIBLE_DEVICES correctly set to {args.gpu}")
    
    # Verify JAX sees only one device
    try:
        devices = jax.devices()
        print(f"[Worker] JAX devices: {devices}")
        if len(devices) > 1:
            print(f"[Worker] ERROR: JAX sees {len(devices)} devices!")
            print(f"[Worker] This means CUDA_VISIBLE_DEVICES was not set before JAX import.")
            print(f"[Worker] All workers may be using the same GPU!")
        elif len(devices) == 1:
            print(f"[Worker] JAX correctly sees 1 device: {devices[0]}")
        else:
            print(f"[Worker] WARNING: JAX sees {len(devices)} devices (expected 1)")
    except Exception as e:
        print(f"[Worker] Could not check JAX devices: {e}")

    data = np.load(args.dataset_npz, allow_pickle=True)
    Ys = data["Ys"]
    thetas = data["thetas"]
    centroids = data["centroids_idx"]
    vertices = data["vertices"]
    params_spatial = data["params_spatial"].item() if "params_spatial" in data.files else {}
    R_mm = params_spatial.get("R_mm", [10.0, 8.0, 10.0])
    A0 = params_spatial.get("A0", None)
    
    # Load graph edges if available
    edge_index = data.get("edge_index", None)
    edge_weight_mm = data.get("edge_weight_mm", None)
    n_vertices = vertices.shape[0]

    N = Ys.shape[0]
    rng_np = np.random.default_rng(args.seed)
    perm = rng_np.permutation(N)
    n_val = max(1, int(0.1 * N))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Use true graph from dataset if available, otherwise fallback to kNN
    if edge_index is not None and edge_weight_mm is not None:
        print("[Worker] Building graph from dataset edges (TRUE mesh-geodesic)...")
        knn_graph = build_graph_from_edges(edge_index, edge_weight_mm, n_vertices)
        print(f"[Worker] Graph shape: {knn_graph.shape}, nnz: {knn_graph.nnz}")
        print("[Worker] Using TRUE mesh-geodesic distances from cortical surface!")
    else:
        print("[Worker] WARNING: No mesh edges found, using kNN fallback...")
        knn_graph = build_knn_geodesic(vertices, k=10)

    study = optuna.load_study(study_name=args.study, storage=args.storage)
    print(f"[Worker] Loaded study '{args.study}' with {len(study.trials)} existing trials")

    objective = objective_factory(
        Ys=Ys,
        thetas=thetas,
        centroids=centroids,
        vertices=vertices,
        R_mm=R_mm,
        A0=A0,
        train_idx=train_idx,
        val_idx=val_idx,
        seed_np=args.seed,
        knn_graph=knn_graph,
        thresholds_mm=(5, 10, 15, 20, 30),
    )

    print(f"[Worker] Starting optimization: {args.n_trials} trials")
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\n" + "="*60)
    print(f"[Worker] Optimization completed!")
    print(f"[Worker] Best value: {study.best_value:.6f}")
    print(f"[Worker] Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    best = study.best_trial
    print(f"[Worker] Best trial user attrs (metrics):")
    for k, v in sorted(best.user_attrs.items()):
        print(f"  {k}: {v}")
    print("="*60)


if __name__ == "__main__":
    main()
