"""Paired M1 training with immutable inputs, fixed validation probes and bounded runs."""

import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import signal
import threading
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from .dataset import (
    digest_json,
    load_training_arrays,
    sha256,
    source_hash,
    write_json,
    xyz_statistics,
)
from .encoder import RawEncoder
from .model import ConditionalFlow, MLPVelocityField, SetVelocityField


def masked_flow_errors(prediction, target, mask):
    difference = torch.where(mask[..., None], prediction.float() - target.float(), 0.0)
    return difference.square().sum(dim=(1, 2)) / (
        mask.sum(1).clamp_min(1) * prediction.shape[-1]
    )


def masked_flow_loss(prediction, target, mask):
    return masked_flow_errors(prediction, target, mask).mean()


def masked_flow_loss_by_k(prediction, target, mask):
    per_sample = masked_flow_errors(prediction, target, mask)
    counts = mask.sum(1)
    by_k = {
        int(k): float(per_sample[counts == k].mean())
        for k in counts.unique(sorted=True)
    }
    return per_sample.mean(), by_k


def unknown_k_flow_errors(model, t, theta_t, target, y, mask):
    """Predict K from Y and teacher-force only the target set size into the field.

    The count head never receives K. During inference its argmax determines the
    number of tokens allocated to ``model.velocity``.
    """
    cond, logits = model.encode(y)
    counts = mask.sum(1)
    errors = torch.empty(len(theta_t), device=theta_t.device, dtype=torch.float32)
    for k in counts.unique(sorted=True):
        selected = counts == k
        prediction = model.velocity(t[selected], theta_t[selected, :k], cond[selected])
        errors[selected] = (
            (prediction.float() - target[selected, :k].float())
            .square()
            .mean(dim=(1, 2))
        )
    return errors, logits


def null_slot_flow_errors(
    prediction,
    target,
    status_logits,
    status_target,
    status_input,
    active_mask,
):
    """Return per-record active, null and masked-status errors."""
    if prediction.shape != target.shape or active_mask.shape != prediction.shape[:2]:
        raise ValueError("null-slot velocity tensors have incompatible shapes")
    if status_logits.shape != (*active_mask.shape, 2):
        raise ValueError("null-slot status logits have incompatible shape")
    difference = prediction.float() - target.float()
    active_squared = torch.where(
        active_mask[..., None], difference.square(), 0.0
    ).sum((1, 2))
    null_mask = ~active_mask
    null_squared = torch.where(
        null_mask[..., None], difference.square(), 0.0
    ).sum((1, 2))
    d_state = prediction.shape[-1]
    active_errors = active_squared / (active_mask.sum(1).clamp_min(1) * d_state)
    null_errors = null_squared / (null_mask.sum(1).clamp_min(1) * d_state)
    masked = status_input == 2
    cross_entropy = F.cross_entropy(
        status_logits.float().reshape(-1, 2),
        status_target.reshape(-1),
        reduction="none",
    ).reshape_as(status_target)
    masked_count = masked.sum(1)
    status_errors = torch.where(masked, cross_entropy, 0.0).sum(1) / masked_count.clamp_min(1)
    masked_correct = torch.where(
        masked, status_logits.argmax(-1) == status_target, False
    ).sum(1)
    return active_errors, null_errors, status_errors, masked_correct, masked_count


def null_slot_inference_logits(model, t, state, y):
    """Predict slot existence with no ground-truth statuses revealed."""
    from .null_slots import MASK

    status = torch.full(
        state.shape[:2], MASK, dtype=torch.long, device=state.device
    )
    return model(t, state, status, y)[1]


def flow_batch(theta1, generator):
    theta0 = torch.randn(
        theta1.shape, generator=generator, device=theta1.device, dtype=theta1.dtype
    )
    t = torch.rand(
        theta1.shape[0], generator=generator, device=theta1.device, dtype=theta1.dtype
    )
    return (
        t,
        (1 - t[:, None, None]) * theta0 + t[:, None, None] * theta1,
        theta1 - theta0,
    )


def learning_rate(step, peak, warmup, steps):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup - 1)
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def build_model(config, architecture):
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    encoder = RawEncoder(**config["encoder"])
    torch.manual_seed(seed + 1)
    cfg = config["model"]
    if config.get("milestone") == "V15":
        target = cfg.get("target", "xyz")
        if target not in ("xyz", "xyz_radius") or (
            target == "xyz_radius" and cfg.get("radius_transform") != "log"
        ):
            raise ValueError("V15 target must be xyz or xyz_radius with log radii")
        cardinality_mode = cfg.get("cardinality_mode", "count_head")
        if cardinality_mode == "null_slots":
            if architecture != "transformer":
                raise ValueError("null-slot cardinality requires the Transformer")
            from .null_slot_model import NullSlotConditionalFlow

            return NullSlotConditionalFlow(
                encoder,
                dict(
                    k_max=config["dataset"]["simulation"]["k_max"],
                    d_in=4 if target == "xyz_radius" else 3,
                    d_cond=config["encoder"]["cond_dim"],
                    d_model=cfg["d_model"],
                    n_layers=cfg["n_layers"],
                    n_heads=cfg["n_heads"],
                ),
            )
        if cardinality_mode != "count_head":
            raise ValueError(f"unknown V15 cardinality mode: {cardinality_mode}")
        from .v15_model import UnknownKConditionalFlow

        return UnknownKConditionalFlow(
            encoder,
            dict(
                k_max=config["dataset"]["simulation"]["k_max"],
                d_in=4 if target == "xyz_radius" else 3,
                d_cond=config["encoder"]["cond_dim"],
                width=cfg["mlp_width"],
                depth=cfg["mlp_depth"],
                d_model=cfg["d_model"],
                n_layers=cfg["n_layers"],
                n_heads=cfg["n_heads"],
            ),
            architecture,
        )
    elif architecture == "mlp":
        field = MLPVelocityField(
            d_cond=config["encoder"]["cond_dim"],
            width=cfg["mlp_width"],
            depth=cfg["mlp_depth"],
        )
    elif architecture == "transformer":
        field = SetVelocityField(
            d_cond=config["encoder"]["cond_dim"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
        )
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    return ConditionalFlow(encoder, field)


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def initialize_from_checkpoint(
    model,
    checkpoint_path,
    config,
    architecture,
    mean,
    std,
    *,
    variable,
):
    """Warm-start a new run from validated EMA weights.

    Training settings may change, while the data representation and model
    topology must remain identical. A fresh optimizer and random streams are
    intentionally used so the child run has its own auditable schedule.
    """
    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != architecture:
        raise ValueError("initialization checkpoint architecture differs")
    parent_config = checkpoint.get("config")
    if not isinstance(parent_config, dict):
        raise ValueError("initialization checkpoint has no configuration")
    for section in ("milestone", "dataset", "encoder", "model"):
        if parent_config.get(section) != config.get(section):
            raise ValueError(f"initialization checkpoint {section} differs")
    mean_key, std_key = (
        ("state_mean", "state_std") if variable else ("mean_mm", "std_mm")
    )
    if mean_key not in checkpoint or std_key not in checkpoint:
        raise ValueError("initialization checkpoint has no normalization")
    parent_mean = checkpoint[mean_key].detach().cpu().numpy()
    parent_std = checkpoint[std_key].detach().cpu().numpy()
    if not np.array_equal(parent_mean, np.asarray(mean)) or not np.array_equal(
        parent_std, np.asarray(std)
    ):
        raise ValueError("initialization checkpoint normalization differs")
    if "ema" not in checkpoint:
        raise ValueError("initialization checkpoint has no EMA weights")
    model.load_state_dict(checkpoint["ema"])
    return {
        "initialization_checkpoint_sha256": sha256(path),
        "initialization_step": int(checkpoint.get("step", 0)),
        "initialization_fingerprint": checkpoint.get("fingerprint"),
        "initialization_source": "ema",
    }


def run_training(
    config,
    architecture,
    dataset_root,
    output_root,
    device="cuda",
    code_commit=None,
    stop_event=None,
    initialization_checkpoint=None,
):
    started = time.monotonic()
    cfg = config["training"]
    variable = config["milestone"] == "V15"
    cardinality_mode = (
        config["model"].get("cardinality_mode", "count_head")
        if variable
        else "fixed"
    )
    null_slots = cardinality_mode == "null_slots"
    expected_normalize = (
        "shared_source_train_only" if variable else "shared_xyz_train_only"
    )
    expected_coupling = "rectangular_hungarian" if null_slots else "none"
    if (
        config["milestone"] not in ("M1", "V15")
        or cfg["coupling"] != expected_coupling
        or (cfg.get("permute_sources", False) and not variable)
        or cfg["normalize"] != expected_normalize
    ):
        raise ValueError("unsupported flow training configuration")
    if cfg["steps"] < 1 or cfg["batch_size"] < 1 or not 0 < cfg["max_seconds"] <= 6900:
        raise ValueError("invalid training length, batch size or runtime limit")
    if variable and not null_slots and cfg.get("count_loss_weight", 0) <= 0:
        raise ValueError("unknown-K training requires a positive count loss weight")
    if null_slots and (
        architecture != "transformer"
        or cfg.get("null_flow_weight", 0) <= 0
        or cfg.get("existence_loss_weight", 0) <= 0
    ):
        raise ValueError("null-slot training requires positive loss weights and a Transformer")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 1
    ):
        raise ValueError("CUDA training requires exactly one visible GPU")
    dataset_root, output_root = Path(dataset_root), Path(output_root)
    audit = json.loads((dataset_root / "audit.json").read_text())
    for key in [
        "disjoint_seeds",
        "disjoint_record_ids",
        "disjoint_geometry_ids",
        "shard_checksums_verified",
        "simulator_and_config_match",
    ]:
        if audit.get(key) is not True:
            raise ValueError(f"dataset audit did not pass: {key}")
    for split in ("train", "val"):
        if (
            sha256(dataset_root / split / "manifest.json")
            != audit["manifest_sha256"][split]
        ):
            raise ValueError("training manifest changed since split audit")
    loader = load_training_arrays
    if variable:
        from .v15_dataset import load_training_arrays as loader
        from .v15 import (
            normalize_source_state,
            source_state_statistics,
            unknown_k_flow_batch,
        )
        if null_slots:
            from .null_slots import null_slot_flow_batch

        if audit.get("filtered") is not True:
            raise ValueError("V15 training requires filtered observations")
        if (
            config["model"].get("target") != "xyz_radius"
            or config["model"].get("radius_transform") != "log"
        ):
            raise ValueError("V15 training requires xyz_radius with log radii")
    train = loader(dataset_root / "train")
    val = loader(dataset_root / "val")
    k_max = config["dataset"]["simulation"]["k_max"] if variable else 3
    for data in (train, val):
        if data["manifest"]["config"] != config["dataset"]:
            raise ValueError("training config and generated data differ")
        if (
            data["theta"].shape[1:] != (k_max, 3)
            or (variable and data["radii"].shape[1:] != (k_max,))
            or data["Y"].shape[1] != config["encoder"]["n_sensors"]
        ):
            raise ValueError("unexpected training tensor dimensions")
    mean, std = (
        source_state_statistics(train["theta"], train["radii"], train["mask"])
        if variable
        else xyz_statistics(train["theta"])
    )
    model = build_model(config, architecture).to(device)
    initialization = (
        initialize_from_checkpoint(
            model,
            initialization_checkpoint,
            config,
            architecture,
            mean,
            std,
            variable=variable,
        )
        if initialization_checkpoint is not None
        else None
    )
    fingerprint_inputs = dict(
        config=config,
        architecture=architecture,
        code_sha256=source_hash(),
        train=train["manifest"]["fingerprint"],
        val=val["manifest"]["fingerprint"],
    )
    if initialization is not None:
        fingerprint_inputs["initialization_checkpoint_sha256"] = initialization[
            "initialization_checkpoint_sha256"
        ]
    fingerprint = digest_json(fingerprint_inputs)
    output = output_root / f"{architecture}-{fingerprint[:16]}"
    output.mkdir(parents=True, exist_ok=False)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    if variable:
        train_y = train["Y"]  # Read only selected batches from the NPY memory map.
        val_y = torch.from_numpy(np.array(val["Y"]))
        train_theta = torch.from_numpy(
            normalize_source_state(
                train["theta"], train["radii"], train["mask"], mean, std
            )
        )
        val_theta = torch.from_numpy(
            normalize_source_state(val["theta"], val["radii"], val["mask"], mean, std)
        )
        train_mask, val_mask = torch.from_numpy(train["mask"]), torch.from_numpy(
            val["mask"]
        )
    else:
        train_y, train_theta = torch.from_numpy(train["Y"]), torch.from_numpy(
            (train["theta"] - mean) / std
        )
        val_y, val_theta = torch.from_numpy(val["Y"]), torch.from_numpy(
            (val["theta"] - mean) / std
        )
        train_mask = torch.ones(train_theta.shape[:2], dtype=torch.bool)
        val_mask = torch.ones(val_theta.shape[:2], dtype=torch.bool)
    if device.type == "cuda" and not variable:
        train_y = train_y.pin_memory()
    # CPU streams make paired batches/noise independent of parameter count and CUDA kernels.
    batch_rng = np.random.default_rng(cfg["seed"] + 1000)
    flow_rng = torch.Generator().manual_seed(cfg["seed"] + 2000)

    def make_flow(theta, mask, generator):
        if null_slots:
            return null_slot_flow_batch(theta, mask, generator)
        if variable:
            return unknown_k_flow_batch(
                theta, mask, generator, cfg["permute_sources"]
            )
        return flow_batch(theta, generator)

    val_probe = make_flow(
        val_theta, val_mask, torch.Generator().manual_seed(cfg["seed"] + 3000)
    )
    amp_enabled = bool(cfg["bf16"] and device.type == "cuda")

    def amp():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp_enabled
            else nullcontext()
        )

    @torch.no_grad()
    def validate(net):
        net.eval()
        total = flow_total = count_total = correct = 0.0
        active_total = null_total = status_total = 0.0
        status_correct = status_count = count_absolute = zero_active = 0.0
        by_k_total = torch.zeros(k_max, dtype=torch.float64)
        by_k_count = torch.zeros(k_max, dtype=torch.int64)
        confusion = torch.zeros(
            (k_max, k_max + 1 if null_slots else k_max), dtype=torch.int64
        )
        for start in range(0, len(val_y), cfg["batch_size"]):
            stop = min(start + cfg["batch_size"], len(val_y))
            mask = val_mask[start:stop].to(device)
            if null_slots:
                t, xt, target, status_input, status_target, active_mask = [
                    value[start:stop].to(device) for value in val_probe
                ]
                y_validation = val_y[start:stop].to(device)
                with amp():
                    prediction, status_logits = net(
                        t, xt, status_input, y_validation
                    )
                    (
                        active_errors,
                        null_errors,
                        status_errors,
                        batch_status_correct,
                        batch_status_count,
                    ) = null_slot_flow_errors(
                        prediction,
                        target,
                        status_logits,
                        status_target,
                        status_input,
                        active_mask,
                    )
                    errors = active_errors + cfg["null_flow_weight"] * null_errors
                    losses = errors + cfg["existence_loss_weight"] * status_errors
                    inference_logits = null_slot_inference_logits(
                        net, t, xt, y_validation
                    )
                labels = active_mask.sum(1)
                predicted = (inference_logits.argmax(-1) == 0).sum(1)
                active_total += float(active_errors.sum())
                null_total += float(null_errors.sum())
                status_total += float(status_errors.sum())
                status_correct += int(batch_status_correct.sum())
                status_count += int(batch_status_count.sum())
                count_absolute += int((predicted - labels).abs().sum())
                zero_active += int((predicted == 0).sum())
                correct += int((predicted == labels).sum())
                for truth, guess, error in zip(
                    labels.cpu(), predicted.cpu(), errors.cpu()
                ):
                    by_k_total[truth - 1] += error.double()
                    by_k_count[truth - 1] += 1
                    confusion[truth - 1, guess] += 1
            else:
                t, xt, target = [a[start:stop].to(device) for a in val_probe]
                with amp():
                    if variable:
                        errors, logits = unknown_k_flow_errors(
                            net, t, xt, target, val_y[start:stop].to(device), mask
                        )
                        labels = mask.sum(1) - 1
                        count_errors = F.cross_entropy(
                            logits.float(), labels, reduction="none"
                        )
                        losses = errors + cfg["count_loss_weight"] * count_errors
                    else:
                        errors = masked_flow_errors(
                            net(t, xt, val_y[start:stop].to(device), mask), target, mask
                        )
                        losses = errors
                if variable:
                    count_total += float(count_errors.sum())
                    predicted = logits.argmax(1)
                    correct += int((predicted == labels).sum())
                    for truth, guess, error in zip(
                        labels.cpu(), predicted.cpu(), errors.cpu()
                    ):
                        by_k_total[truth] += error.double()
                        by_k_count[truth] += 1
                        confusion[truth, guess] += 1
            total += (
                float(losses.sum())
                if variable
                else float(losses.mean()) * (stop - start)
            )
            flow_total += float(errors.sum())
        net.train()
        metrics = dict(loss=total / len(val_y), flow_loss=flow_total / len(val_y))
        if null_slots:
            metrics.update(
                active_flow_loss=active_total / len(val_y),
                null_flow_loss=null_total / len(val_y),
                status_loss=status_total / len(val_y),
                status_accuracy=status_correct / max(1, status_count),
                count_accuracy=correct / len(val_y),
                count_mae=count_absolute / len(val_y),
                zero_active_rate=zero_active / len(val_y),
                flow_loss_by_k={
                    str(i + 1): float(by_k_total[i] / by_k_count[i])
                    for i in range(k_max)
                    if by_k_count[i]
                },
                count_confusion=confusion.tolist(),
            )
        elif variable:
            metrics.update(
                count_loss=count_total / len(val_y),
                count_accuracy=correct / len(val_y),
                flow_loss_by_k={
                    str(i + 1): float(by_k_total[i] / by_k_count[i])
                    for i in range(k_max)
                    if by_k_count[i]
                },
                count_confusion=confusion.tolist(),
            )
        return metrics

    manifest = dict(
        architecture=architecture,
        config=config,
        fingerprint=fingerprint,
        code_sha256=source_hash(),
        code_commit=code_commit,
        dataset_audit_sha256=sha256(dataset_root / "audit.json"),
        train_fingerprint=train["manifest"]["fingerprint"],
        val_fingerprint=val["manifest"]["fingerprint"],
        parameter_count=sum(p.numel() for p in model.parameters()),
        encoder_parameter_count=sum(p.numel() for p in model.encoder.parameters()),
        device=str(device),
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        amp_dtype="bfloat16" if amp_enabled else "float32",
        created_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_recordings_accessed=False,
        acceptance_evaluated=False,
    )
    if variable:
        manifest.update(
            known_k=False,
            predicts_k=True,
            predicts_radii=True,
            cardinality_mode=cardinality_mode,
            count_head=not null_slots,
            target_components=["x_mm", "y_mm", "z_mm", "log_radius_mm"],
            normalisation_mean=mean.tolist(),
            normalisation_std=std.tolist(),
            radius_bounds_mm=[
                config["dataset"]["simulation"]["radius_min"],
                config["dataset"]["simulation"]["radius_max"],
            ],
            validation_flow_count_mode=(
                "joint_status_canvas" if null_slots else "teacher_forced"
            ),
            dataset_loading="memory_mapped_batches",
            source_label_permutation=cfg.get("permute_sources", False),
        )
    else:
        manifest.update(
            normalisation_mean_mm=mean.tolist(),
            normalisation_std_mm=std.tolist(),
        )
    if initialization is not None:
        manifest.update(initialization)
    write_json(output / "manifest.json", manifest)
    initial_metrics = validate(model)
    initial_val = initial_metrics["loss"]
    write_json(
        output / "status.json",
        dict(status="running", step=0, initial_val_loss=initial_val),
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                parameters=manifest["parameter_count"],
                initial_val_loss=initial_val,
            )
        ),
        flush=True,
    )
    step, best_val = 0, math.inf
    latest_val, latest_ema_val = initial_val, initial_val
    latest_metrics = latest_ema_metrics = initial_metrics
    validation_step = 0
    history = output / "history.jsonl"

    def checkpoint():
        payload = dict(
            model=model.state_dict(),
            ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),
            step=step,
            fingerprint=fingerprint,
            config=config,
            architecture=architecture,
            batch_rng=batch_rng.bit_generator.state,
            flow_rng=flow_rng.get_state(),
            val_loss=latest_val,
            ema_val_loss=latest_ema_val,
            validation_step=validation_step,
        )
        if initialization is not None:
            payload.update(initialization)
        payload.update(
            dict(state_mean=torch.from_numpy(mean), state_std=torch.from_numpy(std))
            if variable
            else dict(mean_mm=torch.from_numpy(mean), std_mm=torch.from_numpy(std))
        )
        atomic_checkpoint(output / "last.pt", payload)

    losses, gradients = [], []
    try:
        model.train()
        for index in range(cfg["steps"]):
            if (
                stop_event is not None and stop_event.is_set()
            ) or time.monotonic() - started >= cfg["max_seconds"]:
                break
            idx = batch_rng.choice(
                len(train_y), min(cfg["batch_size"], len(train_y)), replace=False
            )
            mask_cpu = train_mask[idx]
            probe = [
                value.to(device)
                for value in make_flow(train_theta[idx], mask_cpu, flow_rng)
            ]
            if null_slots:
                t, xt, target, status_input, status_target, active_mask = probe
            else:
                t, xt, target = probe
            y_batch = torch.from_numpy(train_y[idx]) if variable else train_y[idx]
            y = y_batch.to(device, non_blocking=True)
            mask = mask_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(index, cfg["lr"], cfg["warmup"], cfg["steps"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            with amp():
                if null_slots:
                    prediction, status_logits = model(t, xt, status_input, y)
                    (
                        active_errors,
                        null_errors,
                        status_errors,
                        _,
                        _,
                    ) = null_slot_flow_errors(
                        prediction,
                        target,
                        status_logits,
                        status_target,
                        status_input,
                        active_mask,
                    )
                    loss = (
                        active_errors
                        + cfg["null_flow_weight"] * null_errors
                        + cfg["existence_loss_weight"] * status_errors
                    ).mean()
                elif variable:
                    flow_errors, count_logits = unknown_k_flow_errors(
                        model, t, xt, target, y, mask
                    )
                    flow_loss = flow_errors.mean()
                    count_loss = F.cross_entropy(count_logits.float(), mask.sum(1) - 1)
                    loss = flow_loss + cfg["count_loss_weight"] * count_loss
                else:
                    loss = masked_flow_loss(model(t, xt, y, mask), target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {index}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1 - cfg["ema_decay"])
            step = index + 1
            losses.append(float(loss.detach()))
            gradients.append(float(norm))
            evaluate = step % cfg["eval_every"] == 0 or step == cfg["steps"]
            if evaluate:
                latest_metrics, latest_ema_metrics = validate(model), validate(ema)
                latest_val, latest_ema_val = (
                    latest_metrics["loss"],
                    latest_ema_metrics["loss"],
                )
                validation_step = step
                if not math.isfinite(latest_val + latest_ema_val):
                    raise FloatingPointError("non-finite validation loss")
                checkpoint()
                validated_payload = dict(
                    ema=ema.state_dict(),
                    step=step,
                    fingerprint=fingerprint,
                    config=config,
                    architecture=architecture,
                )
                if initialization is not None:
                    validated_payload.update(initialization)
                validated_payload.update(
                    dict(
                        state_mean=torch.from_numpy(mean),
                        state_std=torch.from_numpy(std),
                    )
                    if variable
                    else dict(
                        mean_mm=torch.from_numpy(mean),
                        std_mm=torch.from_numpy(std),
                    )
                )
                atomic_checkpoint(output / "validated-ema.pt", validated_payload)
                if latest_ema_val < best_val:
                    best_val = latest_ema_val
                    atomic_checkpoint(output / "best-ema.pt", validated_payload)
            if step % cfg["log_every"] == 0 or evaluate:
                row = dict(
                    step=step,
                    train_loss=float(np.mean(losses)),
                    grad_norm=float(np.mean(gradients)),
                    lr=lr,
                    elapsed_seconds=time.monotonic() - started,
                    gpu_peak_memory_gb=(
                        torch.cuda.max_memory_allocated() / 1e9
                        if device.type == "cuda"
                        else 0.0
                    ),
                )
                if evaluate:
                    row.update(
                        val_loss=latest_val,
                        ema_val_loss=latest_ema_val,
                        validation_step=validation_step,
                    )
                    if variable:
                        if null_slots:
                            for prefix, metrics in (
                                ("val", latest_metrics),
                                ("ema_val", latest_ema_metrics),
                            ):
                                for key in (
                                    "flow_loss",
                                    "active_flow_loss",
                                    "null_flow_loss",
                                    "status_loss",
                                    "status_accuracy",
                                    "count_accuracy",
                                    "count_mae",
                                    "zero_active_rate",
                                    "flow_loss_by_k",
                                    "count_confusion",
                                ):
                                    row[f"{prefix}_{key}"] = metrics[key]
                        else:
                            row.update(
                                val_flow_loss=latest_metrics["flow_loss"],
                                ema_val_flow_loss=latest_ema_metrics["flow_loss"],
                                val_count_loss=latest_metrics["count_loss"],
                                ema_val_count_loss=latest_ema_metrics["count_loss"],
                                val_count_accuracy=latest_metrics["count_accuracy"],
                                ema_val_count_accuracy=latest_ema_metrics["count_accuracy"],
                                val_flow_loss_by_k=latest_metrics["flow_loss_by_k"],
                                ema_val_flow_loss_by_k=latest_ema_metrics["flow_loss_by_k"],
                                val_count_confusion=latest_metrics["count_confusion"],
                                ema_val_count_confusion=latest_ema_metrics[
                                    "count_confusion"
                                ],
                            )
                with history.open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
                write_json(output / "status.json", dict(status="running", **row))
                losses.clear()
                gradients.clear()
        checkpoint()
        status = (
            "complete"
            if step == cfg["steps"]
            else (
                "interrupted"
                if stop_event is not None and stop_event.is_set()
                else "budget_exhausted"
            )
        )
        result = dict(
            status=status,
            step=step,
            initial_val_loss=initial_val,
            val_loss=latest_val,
            ema_val_loss=latest_ema_val,
            validation_step=validation_step,
            validation_matches_checkpoint=validation_step == step,
            elapsed_seconds=time.monotonic() - started,
            acceptance_evaluated=False,
            checkpoint_sha256=sha256(output / "last.pt"),
        )
        if initialization is not None:
            result.update(initialization)
        if validation_step:
            result.update(
                validated_checkpoint_step=validation_step,
                validated_checkpoint_sha256=sha256(output / "validated-ema.pt"),
            )
        if null_slots:
            result.update(
                flow_loss=latest_ema_metrics["flow_loss"],
                active_flow_loss=latest_ema_metrics["active_flow_loss"],
                null_flow_loss=latest_ema_metrics["null_flow_loss"],
                status_loss=latest_ema_metrics["status_loss"],
                status_accuracy=latest_ema_metrics["status_accuracy"],
                count_accuracy=latest_ema_metrics["count_accuracy"],
                count_mae=latest_ema_metrics["count_mae"],
                zero_active_rate=latest_ema_metrics["zero_active_rate"],
                flow_loss_by_k=latest_ema_metrics["flow_loss_by_k"],
                count_confusion=latest_ema_metrics["count_confusion"],
                metrics_source="ema",
                flow_count_mode="joint_status_canvas",
            )
        elif variable:
            result.update(
                flow_loss=latest_ema_metrics["flow_loss"],
                flow_loss_by_k=latest_ema_metrics["flow_loss_by_k"],
                count_loss=latest_ema_metrics["count_loss"],
                count_accuracy=latest_ema_metrics["count_accuracy"],
                count_confusion=latest_ema_metrics["count_confusion"],
                metrics_source="ema",
                flow_count_mode="teacher_forced",
            )
        write_json(output / "result.json", result)
        write_json(output / "status.json", result)
        return output, result
    except BaseException as error:
        write_json(
            output / "status.json", dict(status="failed", step=step, error=repr(error))
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architecture", choices=["mlp", "transformer"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--code-commit")
    parser.add_argument("--initialize-from", type=Path)
    args = parser.parse_args(argv)
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.steps:
        config["training"].update(
            steps=args.steps,
            warmup=min(500, max(1, args.steps // 10)),
            eval_every=min(500, args.steps),
            log_every=min(100, args.steps),
        )
    stop_event = threading.Event()
    previous = {
        sig: signal.signal(sig, lambda *_: stop_event.set())
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        output, result = run_training(
            config,
            args.architecture,
            args.dataset,
            args.output,
            args.device,
            args.code_commit,
            stop_event,
            args.initialize_from,
        )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(dict(output=str(output), **result)), flush=True)
    if result["status"] != "complete":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
