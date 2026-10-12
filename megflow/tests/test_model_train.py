import copy
import io

import numpy as np
import pytest
import torch

from megflow.model import SetVelocityField, ConditionalFlow, MLPVelocityField
from megflow.encoder import RawEncoder
from megflow.train import (
    initialize_from_checkpoint,
    masked_flow_loss,
    masked_flow_loss_by_k,
    flow_batch,
    learning_rate,
)


torch.set_num_threads(1)


@pytest.fixture
def field():
    torch.manual_seed(10)
    return SetVelocityField(d_model=32, n_layers=2, n_heads=4, d_cond=16)


def test_equivariance_and_padding(field):
    theta, cond, t = torch.randn(4, 3, 3), torch.randn(4, 5, 16), torch.rand(4)
    mask = torch.ones(4, 3, dtype=torch.bool)
    output = field(t, theta, cond, mask)
    p = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        field(t, theta[:, p], cond, mask), output[:, p], atol=1e-5, rtol=0
    )
    padded = torch.cat([theta, torch.randn(4, 17, 3) * 100], dim=1)
    padded.requires_grad_()
    padded_mask = torch.arange(20)[None].expand(4, -1) < 3
    padded_output = field(t, padded, cond, padded_mask)
    torch.testing.assert_close(padded_output[:, :3], output, atol=1e-5, rtol=0)
    assert torch.count_nonzero(padded_output[:, 3:]) == 0
    padded_output.sum().backward()
    assert torch.count_nonzero(padded.grad[:, 3:]) == 0


def test_activation_scale_does_not_grow_with_k(field):
    # Large batch reduces sampling variability of independently drawn token sets.
    t, cond = torch.rand(1024), torch.randn(1024, 5, 16)
    scales = []
    with torch.no_grad():
        for k in range(1, 11):
            y = field(
                t, torch.randn(1024, k, 3), cond, torch.ones(1024, k, dtype=torch.bool)
            )
            scales.append(y.square().mean().sqrt().item())
    assert max(scales) / min(scales) < 1.2


def test_loss_ignores_padding_and_normalizes_per_sample():
    pred = torch.zeros(2, 4, 3, requires_grad=True)
    target = torch.tensor([[[2.0] * 3] * 4, [[4.0] * 3] * 4])
    mask = torch.tensor([[True, False, False, False], [True, True, True, False]])
    target[~mask] = float("nan")
    loss = masked_flow_loss(pred, target, mask)
    assert loss.item() == pytest.approx((4 + 16) / 2)
    loss.backward()
    assert torch.count_nonzero(pred.grad[~mask]) == 0


def test_loss_by_k_reports_each_group_and_weighted_total():
    pred = torch.zeros(4, 3, 3)
    target = torch.zeros_like(pred)
    mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [True, True, False],
            [True, True, True],
        ]
    )
    target[0, 0] = 1
    target[1, 0] = 3
    target[2, :2] = 2
    target[3, :3] = 4

    overall, by_k = masked_flow_loss_by_k(pred, target, mask)

    assert overall.item() == pytest.approx((1 + 9 + 4 + 16) / 4)
    assert by_k == pytest.approx({1: 5.0, 2: 4.0, 3: 16.0})


def test_flow_batch_stream_reproducible():
    x = torch.randn(5, 3, 3)
    a = flow_batch(x, torch.Generator().manual_seed(99))
    b = flow_batch(x, torch.Generator().manual_seed(99))
    for u, v in zip(a, b):
        torch.testing.assert_close(u, v, rtol=0, atol=0)
    t, xt, velocity = a
    torch.testing.assert_close(xt + (1 - t[:, None, None]) * velocity, x)


def test_raw_encoder_and_mlp_checkpoint_roundtrip():
    model = ConditionalFlow(
        RawEncoder(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        MLPVelocityField(d_cond=12, width=32, depth=3),
    )
    x, y, t = torch.randn(2, 3, 3), torch.randn(2, 4, 24), torch.rand(2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    out = model(t, x, y, mask)
    assert out.shape == x.shape
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    model.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(model(t, x, y, mask), out, rtol=0, atol=0)


def test_real_optimizer_reduces_flow_loss(field):
    x = torch.randn(32, 3, 3) * 0.1 + 1.0
    t, xt, velocity = flow_batch(x, torch.Generator().manual_seed(42))
    cond, mask = torch.randn(32, 2, 16), torch.ones(32, 3, dtype=torch.bool)
    optimizer = torch.optim.AdamW(field.parameters(), lr=0.005)
    before = masked_flow_loss(field(t, xt, cond, mask), velocity, mask).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = masked_flow_loss(field(t, xt, cond, mask), velocity, mask)
        loss.backward()
        optimizer.step()
    after = masked_flow_loss(field(t, xt, cond, mask), velocity, mask).item()
    assert after < before * 0.2


def test_learning_rate_warmup_and_decay():
    assert learning_rate(0, 0.001, 10, 100) == pytest.approx(0.0001)
    assert learning_rate(9, 0.001, 10, 100) == pytest.approx(0.001)
    assert learning_rate(99, 0.001, 10, 100) < 1e-6


def test_paired_models_have_identical_encoder_initialization():
    from megflow.train import build_model

    config = dict(
        training=dict(seed=303),
        encoder=dict(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        model=dict(mlp_width=32, mlp_depth=3, d_model=32, n_layers=2, n_heads=4),
    )
    a = build_model(config, "mlp").encoder.state_dict()
    b = build_model(config, "transformer").encoder.state_dict()
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_initialize_from_checkpoint_loads_ema_and_rejects_incompatible_config(tmp_path):
    from megflow.train import build_model

    config = dict(
        milestone="M1",
        dataset=dict(simulator="fixed"),
        training=dict(seed=303, lr=3e-4),
        encoder=dict(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        model=dict(mlp_width=32, mlp_depth=3, d_model=32, n_layers=2, n_heads=4),
    )
    parent = build_model(config, "transformer")
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.add_(0.25)
    path = tmp_path / "parent.pt"
    torch.save(
        dict(
            architecture="transformer",
            config=config,
            ema=parent.state_dict(),
            step=20_000,
            fingerprint="parent-fingerprint",
            mean_mm=torch.tensor([1.0, 2.0, 3.0]),
            std_mm=torch.tensor([4.0, 5.0, 6.0]),
        ),
        path,
    )
    child_config = copy.deepcopy(config)
    child_config["training"]["lr"] = 3e-5
    child = build_model(child_config, "transformer")

    metadata = initialize_from_checkpoint(
        child,
        path,
        child_config,
        "transformer",
        np.array([1.0, 2.0, 3.0], dtype="f4"),
        np.array([4.0, 5.0, 6.0], dtype="f4"),
        variable=False,
    )

    for actual, expected in zip(child.parameters(), parent.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata["initialization_step"] == 20_000
    assert metadata["initialization_fingerprint"] == "parent-fingerprint"
    assert len(metadata["initialization_checkpoint_sha256"]) == 64

    incompatible = copy.deepcopy(child_config)
    incompatible["model"]["n_layers"] = 3
    with pytest.raises(ValueError, match="model differs"):
        initialize_from_checkpoint(
            child,
            path,
            incompatible,
            "transformer",
            np.array([1.0, 2.0, 3.0], dtype="f4"),
            np.array([4.0, 5.0, 6.0], dtype="f4"),
            variable=False,
        )


@pytest.mark.parametrize("interrupt", [False, True])
def test_end_to_end_training_uses_train_only_and_saves_valid_checkpoint(
    tmp_path, monkeypatch, interrupt
):
    import threading
    import numpy as np
    from megflow.dataset import SPLITS, write_split, audit_splits
    from megflow.train import run_training, build_model

    config = dict(
        milestone="M1",
        dataset=dict(pool=12, shard_size=2, counts={s: 4 for s in SPLITS}),
        encoder=dict(
            n_sensors=4, widths=[8, 16, 16], kernel_size=3, n_queries=4, cond_dim=12
        ),
        model=dict(mlp_width=32, mlp_depth=3, d_model=32, n_layers=2, n_heads=4),
        training=dict(
            seed=303,
            coupling="none",
            permute_sources=False,
            normalize="shared_xyz_train_only",
            steps=6,
            batch_size=4,
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
    rng = np.random.default_rng(44)
    for j, split in enumerate(SPLITS):
        records = [
            dict(
                Y=rng.normal(size=(4, 24)).astype("f4"),
                theta=np.full((3, 3), j * 100 + i, dtype="f4"),
                centers=np.array([i + 10 * j, 80, 90]),
                radii=np.array([10.0, 8.0, 10.0]),
                seed=i + 10 * j,
                snr_db=10.0,
                placement_attempts=1,
            )
            for i in range(4)
        ]
        write_split(
            iter(records),
            tmp_path / "data" / split,
            split,
            config["dataset"],
            "anatomy",
            "code",
        )
    audit_splits(tmp_path / "data")
    stop = threading.Event()
    if interrupt:
        original_step = torch.optim.AdamW.step

        def stop_after_update(optimizer, *args, **kwargs):
            value = original_step(optimizer, *args, **kwargs)
            stop.set()
            return value

        monkeypatch.setattr(torch.optim.AdamW, "step", stop_after_update)
    path, result = run_training(
        config,
        "mlp",
        tmp_path / "data",
        tmp_path / "runs",
        device="cpu",
        stop_event=stop,
    )
    assert result["status"] == ("interrupted" if interrupt else "complete")
    assert result["step"] == (1 if interrupt else 6)
    assert result["validation_step"] == (0 if interrupt else 6)
    assert result["validation_matches_checkpoint"] is not interrupt
    saved = torch.load(path / "last.pt", weights_only=False)
    assert saved["step"] == result["step"]
    torch.testing.assert_close(saved["mean_mm"], torch.full((3,), 1.5))
    loaded = build_model(config, "mlp")
    loaded.load_state_dict(saved["ema"])
    assert all(torch.isfinite(p).all() for p in loaded.parameters())
    assert result["acceptance_evaluated"] is False
