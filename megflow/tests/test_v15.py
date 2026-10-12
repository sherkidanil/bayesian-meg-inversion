"""Contracts for the supplied 15 September simulator and variable-K training."""

import ast
import importlib
import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.signal import butter, filtfilt
from scipy.stats import gamma


def module():
    assert importlib.util.find_spec("megflow.v15") is not None, "new simulator missing"
    return importlib.import_module("megflow.v15")


def test_null_slot_config_reuses_dataset_and_removes_count_head():
    from omegaconf import OmegaConf

    root = Path(__file__).parents[1] / "config"
    baseline = OmegaConf.to_container(
        OmegaConf.load(root / "v15_k10_4s.yaml"), resolve=True
    )
    null = OmegaConf.to_container(
        OmegaConf.load(root / "v15_null_slots.yaml"), resolve=True
    )
    assert null["dataset"] == baseline["dataset"]
    assert null["encoder"] == baseline["encoder"]
    assert null["model"]["cardinality_mode"] == "null_slots"
    assert null["dataset"]["simulation"]["k_max"] == 10
    assert null["model"]["target"] == "xyz_radius"
    assert "count_loss_weight" not in null["training"]


def reference_functions(n_vertices):
    path = Path(__file__).parents[1] / "references/Dataset_generation_15_09_2026.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = dict(
        np=np, nx=nx, butter=butter, filtfilt=filtfilt, gamma=gamma, N_vert=n_vertices
    )
    exec(compile(tree, str(path), "exec"), ns)
    return ns


def test_window_feasibility_includes_boundary_slots():
    m = module()
    # Original aggregate test admits K6; first/last slots actually fail.
    with pytest.raises(ValueError, match="window"):
        m.validate_windows(1, 6, 2000, 500, 0.4)
    for length, duration, kmax in [(2000, 500, 5), (4000, 500, 10), (2000, 250, 10)]:
        m.validate_windows(1, kmax, length, duration, 0.4)
        for k in range(1, kmax + 1):
            a, b = m.sample_windows(k, length, duration, 0.4, np.random.default_rng(7))
            ref = reference_functions(6)["Sample_Activation_Windows"](
                k, length, duration, 0.4, np.random.default_rng(7)
            )
            np.testing.assert_array_equal(a, ref[0])
            np.testing.assert_array_equal(b, ref[1])
            assert (
                (a >= 0).all() and (b <= length).all() and ((b - a) == duration).all()
            )


def test_primitives_match_supplied_functions(geometry):
    m = module()
    ns = reference_functions(6)
    g = np.random.default_rng(7).normal(size=(4, 6)) * 1e-7
    for seed in [0, 17, 303]:
        actual = m.perturb_leadfield(g, 0.13, 0.001, np.random.default_rng(seed))
        expected = ns["Modeling_Liedfield_Error"](
            g, 0.13, 0.001, np.random.default_rng(seed)
        )
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        params = dict(
            N_src=3,
            N_vert=6,
            min_dist=2.0,
            R_mm=np.array([2.0, 3.0, 4.0]),
            sigma_mm_scale=1 / 3,
            A0=np.ones(3),
        )
        actual = m.spatial_activation(params, geometry, np.random.default_rng(seed))
        expected = ns["Spatial_Activation"](
            params,
            nx.from_scipy_sparse_array(geometry.adjacency),
            np.random.default_rng(seed),
        )
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)


def test_seeded_simulation_preserves_latents_and_reconstructs_g(geometry):
    m = module()
    cfg = m.default_simulation_config()
    cfg.update(k_max=3, min_separation_mm=2.0)
    g = np.random.default_rng(4).normal(size=(4, 6)) * 1e-7
    a = m.simulate(g, geometry, 123, cfg, apply_filter=False)
    b = m.simulate(g, geometry, 123, cfg, apply_filter=False)
    for key in [
        "Y",
        "theta",
        "centers",
        "radii",
        "band_idx",
        "t_start",
        "t_stop",
        "snr_per_source",
        "mask",
    ]:
        np.testing.assert_array_equal(a[key], b[key])
    assert 1 <= a["K"] <= 3 and np.isfinite(a["Y"]).all()
    assert a["mask"].sum() == a["K"] and np.isnan(a["theta"][~a["mask"]]).all()
    assert ((a["radii"][a["mask"]] >= 8) & (a["radii"][a["mask"]] <= 25)).all()
    restored = np.random.default_rng()
    restored.bit_generator.state = a["leadfield_rng_state"]
    expected = m.perturb_leadfield(g, 0.13, 0.001, restored)[0]
    np.testing.assert_array_equal(m.reconstruct_leadfield(g, a, cfg), expected)


def test_masked_normalization_and_unknown_k_flow_hide_padding():
    m = module()
    import torch

    theta = np.array(
        [[[1.0, 2.0, 3.0], [np.nan] * 3], [[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]],
        dtype="f4",
    )
    mask = np.isfinite(theta).all(-1)
    mean, std = m.xyz_statistics_masked(theta, mask)
    np.testing.assert_array_equal(mean, [3, 4, 5])
    x = m.normalize_theta(theta, mask, mean, std)
    assert np.isfinite(x).all() and (x[~mask] == 0).all()
    t, xt, v = m.unknown_k_flow_batch(
        torch.from_numpy(x),
        torch.from_numpy(mask),
        torch.Generator().manual_seed(3),
        True,
    )
    assert torch.count_nonzero(xt[~torch.from_numpy(mask)]) > 0
    assert torch.count_nonzero(v[~torch.from_numpy(mask)]) == 0
    # At t=1, the valid target set is preserved by the random permutation.
    recovered = xt + (1 - t[:, None, None]) * v
    for i in range(2):
        np.testing.assert_allclose(
            np.sort(recovered[i][mask[i]].numpy(), axis=0),
            np.sort(x[i][mask[i]], axis=0),
            atol=1e-6,
        )


def test_source_state_uses_log_radius_and_round_trips():
    m = module()
    theta = np.array(
        [[[1.0, 2.0, 3.0], [np.nan] * 3], [[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]],
        dtype="f4",
    )
    radii = np.array([[8.0, np.nan], [16.0, 25.0]], dtype="f4")
    mask = np.isfinite(theta).all(-1)

    mean, std = m.source_state_statistics(theta, radii, mask)
    state = m.normalize_source_state(theta, radii, mask, mean, std)
    restored_theta, restored_radii = m.denormalize_source_state(
        state, mask, mean, std, radius_bounds=(8.0, 25.0)
    )

    assert mean.shape == std.shape == (4,)
    assert np.isfinite(state).all() and (state[~mask] == 0).all()
    np.testing.assert_allclose(restored_theta[mask], theta[mask], atol=1e-6)
    np.testing.assert_allclose(restored_radii[mask], radii[mask], atol=1e-5)
    assert np.isnan(restored_theta[~mask]).all()
    assert np.isnan(restored_radii[~mask]).all()


@pytest.mark.parametrize("architecture", ["mlp", "transformer"])
def test_unknown_k_models_do_not_accept_truth_mask(architecture):
    import torch
    from megflow.encoder import RawEncoder
    from megflow.v15_model import UnknownKConditionalFlow

    model = UnknownKConditionalFlow(
        RawEncoder(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        dict(
            k_max=5,
            d_in=4,
            d_cond=12,
            width=32,
            depth=2,
            d_model=32,
            n_layers=2,
            n_heads=4,
        ),
        architecture,
    )
    x, y, t = torch.randn(3, 5, 4), torch.randn(3, 4, 40), torch.rand(3)
    velocity, logits = model(t, x, y)
    assert velocity.shape == x.shape and logits.shape == (3, 5)
    cond, encoded_logits = model.encode(y)
    for k in [1, 3, 5]:
        assert model.velocity(t, x[:, :k], cond).shape == (3, k, 4)
    torch.testing.assert_close(logits, encoded_logits)
    assert torch.isfinite(velocity).all() and torch.isfinite(logits).all()
    (velocity.square().mean() + logits.square().mean()).backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_disk_backed_dataset_audit_and_tamper_detection(tmp_path, geometry):
    assert (
        importlib.util.find_spec("megflow.v15_dataset") is not None
    ), "variable dataset missing"
    ds = importlib.import_module("megflow.v15_dataset")
    m = module()
    from megflow.dataset import SPLITS

    cfg = dict(
        pool=10,
        shard_size=2,
        counts={s: 3 for s in SPLITS},
        simulation=m.default_simulation_config(),
    )
    cfg["simulation"].update(k_max=3, min_separation_mm=2.0)
    g = np.random.default_rng(4).normal(size=(4, 6)) * 1e-7
    for j, split in enumerate(SPLITS):
        records = [
            m.simulate(g, geometry, 100 * j + i, cfg["simulation"], apply_filter=False)
            for i in range(3)
        ]
        ds.write_split(
            iter(records),
            tmp_path / split,
            split,
            cfg,
            "anatomy",
            "code",
            require_filtered=False,
        )
    report = ds.audit_splits(tmp_path)
    assert report["counts"] == cfg["counts"]
    data = ds.load_training_arrays(tmp_path / "train")
    assert isinstance(data["Y"], np.memmap) and data["Y"].shape == (3, 4, 200)
    assert data["mask"].shape == (3, 3)
    assert data["radii"].shape == (3, 3)
    with pytest.raises(ValueError, match="train or val"):
        ds.load_training_arrays(tmp_path / "heldout")
    path = tmp_path / "train" / "Y_pool.npy"
    with path.open("r+b") as f:
        f.seek(-1, 2)
        x = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([x[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        ds.load_training_arrays(tmp_path / "train")


@pytest.mark.parametrize("architecture", ["mlp", "transformer"])
def test_variable_k_training_end_to_end(tmp_path, geometry, architecture):
    import torch
    from megflow.dataset import SPLITS
    from megflow.v15_dataset import write_split, audit_splits
    from megflow.train import run_training, build_model

    m = module()
    sim = m.default_simulation_config()
    sim.update(k_max=3, min_separation_mm=2.0)
    config = dict(
        milestone="V15",
        dataset=dict(
            pool=10, shard_size=2, counts={s: 3 for s in SPLITS}, simulation=sim
        ),
        encoder=dict(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        model=dict(
            target="xyz_radius",
            radius_transform="log",
            mlp_width=32,
            mlp_depth=2,
            d_model=32,
            n_layers=2,
            n_heads=4,
        ),
        training=dict(
            seed=303,
            coupling="none",
            permute_sources=True,
            count_loss_weight=0.1,
            normalize="shared_source_train_only",
            steps=3,
            batch_size=3,
            lr=0.001,
            warmup=1,
            weight_decay=0.01,
            clip_norm=1.0,
            ema_decay=0.9,
            bf16=False,
            eval_every=3,
            log_every=3,
            max_seconds=60,
        ),
    )
    g = np.random.default_rng(4).normal(size=(4, 6)) * 1e-7
    train_records = None
    for j, s in enumerate(SPLITS):
        records = [
            m.simulate(g, geometry, 100 * j + i, sim, apply_filter=False)
            for i in range(3)
        ]
        # Isolate training plumbing from expensive MNE filter; simulator filter parity is separate.
        for r in records:
            r["filtered"] = True
        if s == "train":
            train_records = records
        write_split(
            iter(records),
            tmp_path / "data" / s,
            s,
            config["dataset"],
            "anatomy",
            "code",
        )
    audit_splits(tmp_path / "data")
    out, result = run_training(
        config, architecture, tmp_path / "data", tmp_path / "runs", device="cpu"
    )
    assert result["status"] == "complete" and result["step"] == 3
    assert 0 <= result["count_accuracy"] <= 1
    assert set(result["flow_loss_by_k"]) <= {"1", "2", "3"}
    saved = torch.load(out / "last.pt", weights_only=False)
    mean, _ = m.source_state_statistics(
        np.stack([r["theta"] for r in train_records]),
        np.stack([r["radii"] for r in train_records]),
        np.stack([r["mask"] for r in train_records]),
    )
    np.testing.assert_array_equal(saved["state_mean"].numpy(), mean)
    model = build_model(config, architecture)
    model.load_state_dict(saved["ema"])
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_unknown_k_transformer_is_equivariant():
    import torch
    from megflow.encoder import RawEncoder
    from megflow.v15_model import UnknownKConditionalFlow

    model = UnknownKConditionalFlow(
        RawEncoder(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        dict(
            k_max=5,
            d_in=4,
            d_cond=12,
            width=32,
            depth=2,
            d_model=32,
            n_layers=2,
            n_heads=4,
        ),
        "transformer",
    )
    x = torch.randn(4, 5, 4)
    y = torch.randn(4, 4, 40)
    t = torch.rand(4)
    p = torch.tensor([4, 1, 3, 0, 2])
    a, logits = model(t, x, y)
    b, permuted_logits = model(t, x[:, p], y)
    torch.testing.assert_close(b, a[:, p], rtol=0, atol=1e-5)
    torch.testing.assert_close(logits, permuted_logits, rtol=0, atol=0)


def test_unknown_k_sampler_allocates_only_predicted_tokens():
    import torch
    from megflow.v15_model import sample_unknown_k

    class FakeModel:
        k_max = 3
        d_in = 4

        def __init__(self):
            self.seen = []

        def encode(self, y):
            logits = torch.full((len(y), 3), -10.0)
            predicted = y[:, 0, 0].long()
            logits[torch.arange(len(y)), predicted - 1] = 10.0
            return torch.zeros(len(y), 1, 2), logits

        def velocity(self, t, theta, cond):
            self.seen.append(theta.shape[1])
            return torch.zeros_like(theta)

    model = FakeModel()
    y = torch.zeros(3, 1, 2)
    y[:, 0, 0] = torch.tensor([1, 3, 1])
    theta, mask, logits = sample_unknown_k(
        model, y, steps=4, generator=torch.Generator().manual_seed(9)
    )

    assert theta.shape == (3, 3, 4)
    assert model.seen == [1] * 4 + [3] * 4
    assert logits.argmax(1).add(1).tolist() == [1, 3, 1]
    assert mask.sum(1).tolist() == [1, 3, 1]
    assert torch.isfinite(theta[mask]).all() and torch.isnan(theta[~mask]).all()


def test_grouped_unknown_k_loss_backpropagates_through_field():
    import torch
    from torch.nn import functional as F
    from megflow.encoder import RawEncoder
    from megflow.train import unknown_k_flow_errors
    from megflow.v15_model import UnknownKConditionalFlow

    model = UnknownKConditionalFlow(
        RawEncoder(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        dict(
            k_max=3,
            d_in=4,
            d_cond=12,
            width=32,
            depth=2,
            d_model=32,
            n_layers=2,
            n_heads=4,
        ),
        "transformer",
    )
    theta, target = torch.randn(4, 3, 4), torch.randn(4, 3, 4)
    mask = torch.arange(3)[None] < torch.tensor([1, 2, 3, 2])[:, None]
    y, t = torch.randn(4, 4, 40), torch.rand(4)
    errors, logits = unknown_k_flow_errors(model, t, theta, target, y, mask)
    (errors.mean() + 0.1 * F.cross_entropy(logits, mask.sum(1) - 1)).backward()

    assert errors.shape == (4,) and torch.isfinite(errors).all()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for p in model.field.parameters()
    )


def test_runtime_is_fingerprinted_and_checked_between_splits(tmp_path, geometry):
    import json
    from megflow.dataset import SPLITS, write_json, digest_json
    from megflow.v15_dataset import write_split, audit_splits

    m = module()
    sim = m.default_simulation_config()
    sim.update(k_max=1, min_separation_mm=2.0)
    cfg = dict(pool=10, shard_size=1, counts={s: 1 for s in SPLITS}, simulation=sim)
    g = np.random.default_rng(4).normal(size=(4, 6)) * 1e-7
    manifests = {}
    for j, s in enumerate(SPLITS):
        r = m.simulate(g, geometry, j, sim, apply_filter=False)
        manifests[s] = write_split(
            iter([r]), tmp_path / s, s, cfg, "anatomy", "code", require_filtered=False
        )
    assert "mne" in manifests["train"]["versions"]
    val = manifests["val"]
    val["versions"]["mne"] = "different-runtime"
    # A valid fingerprint in a different runtime must still fail the cross-split audit.
    val["fingerprint"] = digest_json(
        dict(
            config=cfg,
            anatomy_sha256="anatomy",
            code_sha256="code",
            split="val",
            versions=val["versions"],
        )
    )
    write_json(tmp_path / "val" / "manifest.json", val)
    with pytest.raises(ValueError, match="versions"):
        audit_splits(tmp_path)
