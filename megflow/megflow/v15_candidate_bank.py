"""Candidate-bank scoring and geodesic aggregation for V15 source sets."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from .forward import signal_subspace, subspace_score, topographies
from .v15_evaluate import _json_finite, nearest_vertex_indices, sample_sources


def validate_prefixes(prefixes, bank_size):
    prefixes = tuple(int(value) for value in prefixes)
    if (
        not isinstance(bank_size, int)
        or bank_size < 1
        or not prefixes
        or prefixes != tuple(sorted(set(prefixes)))
        or prefixes[0] < 1
        or prefixes[-1] > bank_size
    ):
        raise ValueError("prefixes must be increasing, unique, positive, and within bank")
    return prefixes


class RawRecordingReader:
    """Read full-resolution V15 observations while caching one shard."""

    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.manifest = manifest
        self._loaded_file = None
        self._loaded_arrays = None
        self._loaded_shard = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._loaded_file is not None:
            self._loaded_file.close()
        self._loaded_file = None
        self._loaded_arrays = None
        self._loaded_shard = None

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= int(self.manifest["n_records"]):
            raise IndexError(index)
        shard = next(
            (
                item
                for item in self.manifest["shards"]
                if int(item["start"]) <= index < int(item["stop"])
            ),
            None,
        )
        if shard is None:
            raise ValueError(f"record {index} is not covered by a data shard")
        if self._loaded_shard != shard["file"]:
            self.close()
            self._loaded_file = np.load(
                self.directory / shard["file"], allow_pickle=False
            )
            self._loaded_arrays = self._loaded_file["Y"]
            self._loaded_shard = shard["file"]
        return np.asarray(self._loaded_arrays[index - int(shard["start"])])


def load_candidate_bank_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("split") != "dev" or protocol.get("frozen") is not True:
        raise ValueError("heldout evaluation requires a frozen dev protocol")
    if not isinstance(protocol.get("bank_size"), int) or protocol["bank_size"] < 1:
        raise ValueError("dev protocol has invalid candidate bank size")
    return protocol


def write_candidate_bank_artifacts(
    output, *, summaries, records, bank_arrays, metadata
):
    """Write reviewable candidate banks and their nested-prefix summaries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        metadata=_json_finite(metadata),
        metrics_by_prefix=_json_finite(summaries),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "records.json").write_text(
        json.dumps(_json_finite(records), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    np.savez_compressed(output / "bank.npz", **bank_arrays)
    if metadata["split"] == "dev":
        protocol = dict(_json_finite(metadata), frozen=True)
        (output / "protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def select_bank_cardinality(cardinality_probability, observed_counts):
    """Choose K from model probabilities, restricted to counts present in the bank."""
    probabilities = np.asarray(cardinality_probability, dtype=float)
    observed = np.asarray(observed_counts, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(observed):
        raise ValueError("cardinality probabilities and observed counts must align")
    if probabilities.shape[1] < 2 or not np.isfinite(probabilities).all():
        raise ValueError("cardinality probabilities must be finite with K classes")
    if (probabilities < 0).any() or not np.allclose(
        probabilities.sum(1), 1.0, atol=1e-6
    ):
        raise ValueError("cardinality probabilities must be normalized")
    if not np.allclose(probabilities[:, 0], 0.0, atol=1e-7):
        raise ValueError("cardinality class zero must have zero probability")
    k_max = probabilities.shape[1] - 1
    if not len(observed) or (observed < 1).any() or (observed > k_max).any():
        raise ValueError("observed source counts must lie in 1..K_max")

    mean_probability = probabilities.mean(0)
    selected = int(np.argmax(mean_probability[1:]) + 1)
    present = np.unique(observed)
    if selected in present:
        return selected, False
    fallback = int(present[np.argmax(mean_probability[present])])
    return fallback, True


def _validated_weights(weights, size):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (size,) or not np.isfinite(weights).all():
        raise ValueError("weights must be one finite value per candidate")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive mass")
    return weights / weights.sum()


def gibbs_weights(scores, *, tau=1.2e-3):
    """Return stable Gibbs weights for within-cardinality subspace scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite vector")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("Gibbs temperature must be finite and positive")
    logits = (scores - scores.max()) / tau
    weights = np.exp(logits)
    return weights / weights.sum()


def align_candidate_bank(centers, radii, scores, geometry):
    """Align exchangeable candidates to the maximum-score reference."""
    centers = np.asarray(centers, dtype=np.int64)
    radii = np.asarray(radii, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if (
        centers.ndim != 2
        or not centers.size
        or radii.shape != centers.shape
        or scores.shape != (len(centers),)
        or not np.isfinite(radii).all()
        or not np.isfinite(scores).all()
        or (radii <= 0).any()
    ):
        raise ValueError("candidate centers, radii, and scores are incompatible")
    reference = centers[int(np.argmax(scores))]
    aligned_centers = np.empty_like(centers)
    aligned_radii = np.empty_like(radii)
    nonidentity = 0
    identity = np.arange(centers.shape[1])
    for row, candidate in enumerate(centers):
        distances = np.asarray(
            geometry.geodesic(reference[:, None], candidate[None, :]), dtype=float
        )
        finite = distances[np.isfinite(distances)]
        dominant = (
            (float(finite.max()) + 1.0) * (centers.shape[1] + 1)
            if finite.size
            else 1.0
        )
        reference_index, candidate_index = linear_sum_assignment(
            np.where(np.isfinite(distances), distances, dominant)
        )
        permutation = candidate_index[np.argsort(reference_index)]
        aligned_centers[row] = candidate[permutation]
        aligned_radii[row] = radii[row, permutation]
        nonidentity += int(not np.array_equal(permutation, identity))
    return aligned_centers, aligned_radii, nonidentity / len(centers)


def weighted_geodesic_medoid(vertices, weights, geometry, *, cutoff_mm=30.0):
    """Choose an observed vertex minimizing capped weighted graph distance."""
    vertices = np.asarray(vertices, dtype=np.int64)
    if vertices.ndim != 1 or not len(vertices):
        raise ValueError("medoid vertices must be a nonempty vector")
    if not np.isfinite(cutoff_mm) or cutoff_mm <= 0:
        raise ValueError("medoid cutoff must be finite and positive")
    weights = _validated_weights(weights, len(vertices))
    distances = np.asarray(
        geometry.geodesic(vertices[:, None], vertices[None, :]), dtype=float
    )
    costs = np.where(np.isfinite(distances), np.minimum(distances, cutoff_mm), cutoff_mm)
    objective = costs @ weights
    return int(vertices[int(np.argmin(objective))])


def aggregate_aligned_bank(
    centers, radii, weights, geometry, *, cutoff_mm=30.0
):
    """Aggregate aligned centers by medoid and radii with the same weights."""
    centers = np.asarray(centers, dtype=np.int64)
    radii = np.asarray(radii, dtype=float)
    if (
        centers.ndim != 2
        or not centers.size
        or radii.shape != centers.shape
        or not np.isfinite(radii).all()
        or (radii <= 0).any()
    ):
        raise ValueError("aligned centers and radii are incompatible")
    weights = _validated_weights(weights, len(centers))
    aggregated_centers = np.asarray(
        [
            weighted_geodesic_medoid(
                centers[:, slot], weights, geometry, cutoff_mm=cutoff_mm
            )
            for slot in range(centers.shape[1])
        ],
        dtype=np.int64,
    )
    aggregated_radii = np.average(radii, axis=0, weights=weights)
    return aggregated_centers, aggregated_radii


def score_candidate_bank(*, centers, radii, y, leadfield, geometry):
    """Score fixed-cardinality candidates with their predicted patch radii."""
    centers = np.asarray(centers, dtype=np.int64)
    radii = np.asarray(radii, dtype=float)
    y = np.asarray(y, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    if (
        centers.ndim != 2
        or not centers.size
        or radii.shape != centers.shape
        or not np.isfinite(radii).all()
        or (radii <= 0).any()
    ):
        raise ValueError("fixed-cardinality candidates and radii are invalid")
    if (
        y.ndim != 2
        or leadfield.ndim != 2
        or y.shape[0] != leadfield.shape[0]
        or leadfield.shape[1] != len(geometry.vertices)
        or not np.isfinite(y).all()
        or not np.isfinite(leadfield).all()
    ):
        raise ValueError("recording and leadfield are incompatible")
    e_s, _ = signal_subspace(y, centers.shape[1], alpha=2.0, r_max=40)
    return np.asarray(
        [
            subspace_score(
                e_s,
                topographies(leadfield, geometry, candidate, candidate_radii),
            )
            for candidate, candidate_radii in zip(centers, radii)
        ],
        dtype=float,
    )


def decode_bank_prefix(
    *,
    xyz_mm,
    radii_mm,
    mask,
    cardinality_probability,
    y,
    leadfield,
    geometry,
    tau=1.2e-3,
    cutoff_mm=30.0,
):
    """Select K, score its candidate configurations, and aggregate the prefix."""
    xyz_mm = np.asarray(xyz_mm, dtype=float)
    radii_mm = np.asarray(radii_mm, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    probabilities = np.asarray(cardinality_probability, dtype=float)
    if (
        xyz_mm.ndim != 3
        or xyz_mm.shape[2] != 3
        or radii_mm.shape != xyz_mm.shape[:2]
        or mask.shape != xyz_mm.shape[:2]
        or probabilities.shape[0] != len(xyz_mm)
        or not len(xyz_mm)
    ):
        raise ValueError("candidate bank arrays have incompatible shapes")
    if not np.isfinite(xyz_mm[mask]).all() or not np.isfinite(radii_mm[mask]).all():
        raise ValueError("active candidate states must be finite")
    if (radii_mm[mask] <= 0).any():
        raise ValueError("active candidate radii must be positive")
    counts = mask.sum(1)
    selected_k, fallback = select_bank_cardinality(probabilities, counts)
    eligible_index = np.flatnonzero(counts == selected_k)
    if not len(eligible_index):
        raise RuntimeError("selected cardinality has no eligible candidates")
    centers = np.stack(
        [
            nearest_vertex_indices(xyz_mm[index, mask[index]], geometry.vertices)
            for index in eligible_index
        ]
    )
    radii = np.stack(
        [radii_mm[index, mask[index]] for index in eligible_index]
    )
    scores = score_candidate_bank(
        centers=centers,
        radii=radii,
        y=y,
        leadfield=leadfield,
        geometry=geometry,
    )
    weights = gibbs_weights(scores, tau=tau)
    aligned_centers, aligned_radii, permutation_rate = align_candidate_bank(
        centers, radii, scores, geometry
    )
    aggregate_centers, aggregate_radii = aggregate_aligned_bank(
        aligned_centers,
        aligned_radii,
        weights,
        geometry,
        cutoff_mm=cutoff_mm,
    )
    return dict(
        centers=aggregate_centers,
        radii=aggregate_radii,
        selected_k=int(selected_k),
        cardinality_fallback=bool(fallback),
        eligible_candidate_count=int(len(eligible_index)),
        eligible_candidate_indices=eligible_index,
        scores=scores,
        weights=weights,
        effective_sample_size=float(1.0 / np.square(weights).sum()),
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        score_mean=float(scores.mean()),
        permutation_rate=float(permutation_rate),
    )


def sample_candidate_bank(
    model,
    checkpoint,
    y,
    *,
    mode,
    steps,
    bank_size,
    record_seed,
    radius_bounds=(8.0, 25.0),
):
    """Sample one record as a batch so every bank prefix is reproducible."""
    import torch

    if not isinstance(y, torch.Tensor) or y.ndim != 3 or len(y) != 1:
        raise ValueError("candidate sampling requires one batched recording")
    if not isinstance(bank_size, int) or bank_size < 1:
        raise ValueError("candidate bank size must be positive")
    if not isinstance(record_seed, int) or record_seed < 0:
        raise ValueError("record seed must be a nonnegative integer")
    batch = y.repeat(bank_size, 1, 1)
    generator = torch.Generator(device=y.device.type).manual_seed(record_seed)
    return sample_sources(
        model,
        checkpoint,
        batch,
        mode=mode,
        steps=steps,
        generator=generator,
        radius_bounds=radius_bounds,
    )


def record_tensor(values, device):
    """Copy a possibly read-only memory map before exposing it to PyTorch."""
    import torch

    return torch.from_numpy(np.array(values, copy=True)).to(device)


def _summarize_candidate_records(records, *, k_max):
    from .v15_evaluate import aggregate_record_metrics

    summary = aggregate_record_metrics(records, k_max=k_max)

    def mean(key):
        return float(np.mean([record[key] for record in records]))

    summary["candidate_bank"] = dict(
        eligible_candidate_count_mean=mean("eligible_candidate_count"),
        eligible_candidate_count_min=int(
            min(record["eligible_candidate_count"] for record in records)
        ),
        effective_sample_size_mean=mean("effective_sample_size"),
        cardinality_fallback_rate=mean("cardinality_fallback"),
        permutation_rate_mean=mean("permutation_rate"),
        score_range_mean=float(
            np.mean(
                [record["score_max"] - record["score_min"] for record in records]
            )
        ),
        forced_nonempty_candidate_rate=mean("forced_nonempty_candidate_rate"),
        variable_sampled_cardinality_rate=float(
            np.mean([record["unique_sampled_k"] > 1 for record in records])
        ),
    )
    return summary


def validate_candidate_bank_protocol_matches(protocol, current):
    keys = (
        "mode",
        "architecture",
        "sampler_steps",
        "evaluation_seed",
        "bank_size",
        "prefixes",
        "gibbs_tau",
        "ospa_cutoff_mm",
        "checkpoint_sha256",
        "config_sha256",
        "dataset_audit_sha256",
        "heldout_manifest_sha256",
        "anatomy_sha256",
        "decoder_code_sha256",
        "radius_bounds_mm",
    )
    for key in keys:
        if protocol.get(key) != current.get(key):
            raise ValueError(f"heldout candidate-bank setting differs from dev: {key}")


def main(argv=None):
    """Run a frozen candidate-bank decoder on V15 dev or heldout."""
    from omegaconf import OmegaConf
    import torch

    from .dataset import sha256, source_hash
    from .geometry import CorticalGeometry
    from .train import build_model
    from .v15_dataset import (
        cache_distances_float64,
        load_evaluation_arrays,
    )
    from .v15_evaluate import (
        _sample_seed,
        evaluate_record,
        validate_evaluation_inputs,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--distance-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--mode", choices=["count_head", "null_slots"], required=True)
    parser.add_argument(
        "--architecture", choices=["mlp", "transformer"], default="transformer"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--bank-size", type=int, default=100)
    parser.add_argument("--prefixes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--evaluation-seed", type=int, default=20260923)
    parser.add_argument("--gibbs-tau", type=float, default=1.2e-3)
    parser.add_argument("--ospa-cutoff-mm", type=float, default=30.0)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args(argv)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    configured_mode = config["model"].get("cardinality_mode", "count_head")
    if configured_mode != args.mode:
        raise ValueError("requested mode differs from model configuration")
    steps = args.steps or config["training"].get("sampler_steps", 100)
    bank_size = args.bank_size
    prefixes = validate_prefixes(args.prefixes, bank_size)
    evaluation_seed = args.evaluation_seed
    gibbs_tau = args.gibbs_tau
    ospa_cutoff_mm = args.ospa_cutoff_mm
    frozen_protocol = None
    if args.split == "heldout":
        if args.protocol is None:
            raise ValueError("heldout candidate-bank evaluation requires --protocol")
        frozen_protocol = load_candidate_bank_protocol(args.protocol)
        steps = frozen_protocol["sampler_steps"]
        bank_size = frozen_protocol["bank_size"]
        prefixes = validate_prefixes(frozen_protocol["prefixes"], bank_size)
        evaluation_seed = frozen_protocol["evaluation_seed"]
        gibbs_tau = frozen_protocol["gibbs_tau"]
        ospa_cutoff_mm = frozen_protocol["ospa_cutoff_mm"]
    if steps < 1 or not np.isfinite(gibbs_tau) or gibbs_tau <= 0:
        raise ValueError("sampler steps and Gibbs temperature must be positive")

    data = load_evaluation_arrays(args.dataset_root / args.split)
    validate_evaluation_inputs(config, data["manifest"], args.anatomy)
    audit_path = args.dataset_root / "audit.json"
    dataset_audit = json.loads(audit_path.read_text())
    for shard in data["manifest"]["shards"]:
        if sha256(args.dataset_root / args.split / shard["file"]) != shard["sha256"]:
            raise ValueError(f"dataset shard checksum mismatch: {shard['file']}")
    geometry, leadfield, anatomy_provenance = CorticalGeometry.load_bundle(args.anatomy)
    cache_distances_float64(geometry, args.distance_cache)
    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    model = build_model(config, args.architecture).to(args.device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()

    radius_bounds = (
        config["dataset"]["simulation"]["radius_min"],
        config["dataset"]["simulation"]["radius_max"],
    )
    metadata = dict(
        split=args.split,
        mode=args.mode,
        architecture=args.architecture,
        sampler_steps=int(steps),
        evaluation_seed=int(evaluation_seed),
        bank_size=int(bank_size),
        prefixes=list(prefixes),
        gibbs_tau=float(gibbs_tau),
        ospa_cutoff_mm=float(ospa_cutoff_mm),
        checkpoint_sha256=sha256(args.checkpoint),
        config_sha256=sha256(args.config),
        dataset_manifest_sha256=sha256(
            args.dataset_root / args.split / "manifest.json"
        ),
        dataset_audit_sha256=sha256(audit_path),
        heldout_manifest_sha256=dataset_audit["manifest_sha256"]["heldout"],
        anatomy_sha256=sha256(args.anatomy),
        decoder_code_sha256=source_hash(),
        anatomy_provenance=anatomy_provenance,
        k_max=int(config["dataset"]["simulation"]["k_max"]),
        radius_bounds_mm=list(radius_bounds),
        cardinality_selection="argmax_mean_model_probability_observed_fallback",
        candidate_score="rho_with_predicted_variable_radii_fixed_k",
        weighting="gibbs",
        alignment="geodesic_hungarian_to_max_rho",
        aggregation="weighted_capped_geodesic_medoid",
        recording_for_score="unpooled_filtered",
    )
    if frozen_protocol is not None:
        validate_candidate_bank_protocol_matches(frozen_protocol, metadata)
        metadata["dev_protocol_sha256"] = sha256(args.protocol)

    n_records = len(data["Y"])
    k_max = metadata["k_max"]
    bank_arrays = dict(
        xyz_mm=np.full((n_records, bank_size, k_max, 3), np.nan, dtype="f4"),
        radii_mm=np.full((n_records, bank_size, k_max), np.nan, dtype="f4"),
        mask=np.zeros((n_records, bank_size, k_max), dtype=bool),
        active_probability=np.full(
            (n_records, bank_size, k_max), np.nan, dtype="f4"
        ),
        cardinality_probability=np.full(
            (n_records, bank_size, k_max + 1), np.nan, dtype="f4"
        ),
        forced_nonempty=np.zeros((n_records, bank_size), dtype=bool),
        record_seed=np.zeros(n_records, dtype="uint64"),
    )
    records = {str(prefix): [] for prefix in prefixes}
    started = time.monotonic()
    device = torch.device(args.device)
    with RawRecordingReader(
        args.dataset_root / args.split, data["manifest"]
    ) as raw_reader:
        for index in range(n_records):
            record_seed = _sample_seed(
                evaluation_seed, args.split, index, draw=0
            )
            pooled_y = record_tensor(data["Y"][index : index + 1], device)
            decoded = sample_candidate_bank(
                model,
                checkpoint,
                pooled_y,
                mode=args.mode,
                steps=steps,
                bank_size=bank_size,
                record_seed=record_seed,
                radius_bounds=radius_bounds,
            )
            bank_arrays["xyz_mm"][index] = decoded.xyz_mm
            bank_arrays["radii_mm"][index] = decoded.radii_mm
            bank_arrays["mask"][index] = decoded.mask
            bank_arrays["cardinality_probability"][index] = (
                decoded.cardinality_probability
            )
            bank_arrays["forced_nonempty"][index] = decoded.forced_nonempty
            bank_arrays["record_seed"][index] = record_seed
            if decoded.active_probability is not None:
                bank_arrays["active_probability"][index] = decoded.active_probability

            raw_y = raw_reader[index]
            truth_mask = np.asarray(data["mask"][index], dtype=bool)
            for prefix in prefixes:
                aggregated = decode_bank_prefix(
                    xyz_mm=decoded.xyz_mm[:prefix],
                    radii_mm=decoded.radii_mm[:prefix],
                    mask=decoded.mask[:prefix],
                    cardinality_probability=decoded.cardinality_probability[:prefix],
                    y=raw_y,
                    leadfield=leadfield,
                    geometry=geometry,
                    tau=gibbs_tau,
                    cutoff_mm=ospa_cutoff_mm,
                )
                record = evaluate_record(
                    true_centers=data["centers"][index, truth_mask],
                    true_radii=data["radii"][index, truth_mask],
                    predicted_xyz=geometry.vertices[aggregated["centers"]],
                    predicted_radii=aggregated["radii"],
                    geometry=geometry,
                    ospa_cutoff_mm=ospa_cutoff_mm,
                )
                sampled_counts = decoded.mask[:prefix].sum(1)
                record.update(
                    record_index=index,
                    prefix=int(prefix),
                    sample_seed=int(record_seed),
                    predicted_centers=aggregated["centers"].tolist(),
                    predicted_xyz_mm=geometry.vertices[
                        aggregated["centers"]
                    ].tolist(),
                    predicted_radii_mm=aggregated["radii"].tolist(),
                    cardinality_probability=decoded.cardinality_probability[
                        :prefix
                    ].mean(0).tolist(),
                    calibration_kind=decoded.calibration_kind,
                    selected_k=aggregated["selected_k"],
                    eligible_candidate_count=aggregated[
                        "eligible_candidate_count"
                    ],
                    effective_sample_size=aggregated["effective_sample_size"],
                    cardinality_fallback=aggregated["cardinality_fallback"],
                    permutation_rate=aggregated["permutation_rate"],
                    score_min=aggregated["score_min"],
                    score_max=aggregated["score_max"],
                    score_mean=aggregated["score_mean"],
                    forced_nonempty_candidate_rate=float(
                        decoded.forced_nonempty[:prefix].mean()
                    ),
                    unique_sampled_k=int(len(np.unique(sampled_counts))),
                )
                records[str(prefix)].append(record)
            if (index + 1) % 10 == 0 or index + 1 == n_records:
                print(
                    json.dumps(
                        dict(
                            evaluated=index + 1,
                            total=n_records,
                            elapsed_seconds=round(time.monotonic() - started, 1),
                        )
                    ),
                    flush=True,
                )

    summaries = {
        str(prefix): _summarize_candidate_records(
            records[str(prefix)], k_max=k_max
        )
        for prefix in prefixes
    }
    metadata["runtime_seconds"] = time.monotonic() - started
    metadata["runtime_per_record_seconds"] = metadata["runtime_seconds"] / n_records
    write_candidate_bank_artifacts(
        args.output,
        summaries=summaries,
        records=records,
        bank_arrays=bank_arrays,
        metadata=metadata,
    )
    print(json.dumps(_json_finite(summaries), sort_keys=True))


if __name__ == "__main__":
    main()
