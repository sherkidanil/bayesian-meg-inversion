"""Evaluate a frozen M1 candidate bank with maximum-score aggregation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf

from .dataset import sha256, write_json
from .m1_article_decoder import (
    FROZEN_M1_DECODER,
    _json_finite,
    _prefixed_metrics,
    bandpass_sensor_data,
    dominant_signal_subspace,
    load_m1_evaluation_arrays,
    load_sensor_projector,
    load_validated_m1_anatomy,
    m1_localization_metrics,
    m1_patch_dictionaries,
    refine_centroids,
    summarize_m1_records,
)


def max_score_candidate(aligned_vertices, scores):
    """Select the first highest-scoring aligned candidate without truth access."""
    aligned = np.asarray(aligned_vertices)
    values = np.asarray(scores, dtype=np.float64)
    if aligned.ndim != 2 or not aligned.size or values.shape != (len(aligned),):
        raise ValueError("aligned candidates and scores have incompatible shapes")
    if not np.isfinite(values).all():
        raise ValueError("candidate scores must be finite")
    if not np.issubdtype(aligned.dtype, np.integer):
        if not np.equal(aligned, np.floor(aligned)).all():
            raise ValueError("aligned candidates must contain integer vertices")
        aligned = aligned.astype(np.int64)
    return np.asarray(aligned[int(np.argmax(values))], dtype=np.int64).copy()


def evaluate_max_score_bank(
    bank,
    data,
    geometry,
    dictionaries,
    projector,
    *,
    graph_diameter_mm,
    refinement_passes=FROZEN_M1_DECODER["refinement_passes"],
):
    """Score and refine one max-rho candidate per record from a frozen bank."""
    aligned = np.asarray(bank["aligned_vertices"])
    scores = np.asarray(bank["scores"])
    count = len(data["Y"])
    if aligned.ndim != 3 or aligned.shape[0] != count or scores.shape != aligned.shape[:2]:
        raise ValueError("candidate bank does not match evaluation records")
    records = []
    started = time.monotonic()
    for index in range(count):
        initial = max_score_candidate(aligned[index], scores[index])
        score_y = projector @ bandpass_sensor_data(data["Y"][index], sfreq_hz=1000.0)
        signal = dominant_signal_subspace(
            score_y, rank=FROZEN_M1_DECODER["signal_rank"]
        )
        duplicate = len(set(initial.tolist())) != len(initial)
        refined = (
            initial
            if duplicate
            else refine_centroids(
                initial,
                dictionaries,
                signal,
                geometry,
                max_passes=refinement_passes,
            ).centroids
        )
        initial_metrics = m1_localization_metrics(
            true_centers=data["centers"][index],
            predicted_centers=initial,
            geometry=geometry,
            graph_diameter_mm=graph_diameter_mm,
        )
        refined_metrics = m1_localization_metrics(
            true_centers=data["centers"][index],
            predicted_centers=refined,
            geometry=geometry,
            graph_diameter_mm=graph_diameter_mm,
        )
        records.append(
            {
                "record_index": index,
                "record_id": str(data["record_id"][index]),
                "selected_candidate_index": int(np.argmax(scores[index])),
                "duplicate_initial": duplicate,
                **_prefixed_metrics("initial", initial_metrics),
                **_prefixed_metrics("refined", refined_metrics),
            }
        )
    return {
        "initial": summarize_m1_records(records, prefix="initial"),
        "refined": summarize_m1_records(records, prefix="refined"),
        "duplicate_initial_fraction": float(
            np.mean([row["duplicate_initial"] for row in records])
        ),
        "refinement_gain_geo_at_30_mm": float(
            np.mean([row["initial_geodesic_mae_at_30_mm"] for row in records])
            - np.mean([row["refined_geodesic_mae_at_30_mm"] for row in records])
        ),
        "runtime_seconds": time.monotonic() - started,
    }, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--sensor-projector", type=Path, required=True)
    parser.add_argument("--distance-cache", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "heldout"), required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    manifest_path = args.dataset_root / args.split / "manifest.json"
    split_manifest = json.loads(manifest_path.read_text())
    geometry, leadfield, _, anatomy_content = load_validated_m1_anatomy(
        args.anatomy, config, split_manifest
    )
    geometry.cache_distances(args.distance_cache)
    graph_diameter = float(
        np.max(geometry._distances[np.isfinite(geometry._distances)])
    )
    projector = load_sensor_projector(args.sensor_projector, leadfield.shape[0])
    dictionaries = m1_patch_dictionaries(leadfield, geometry, projector)
    data = load_m1_evaluation_arrays(args.dataset_root / args.split)
    source_summary = json.loads((args.bank_dir / "summary.json").read_text())
    source_metadata = source_summary.get("metadata", {})
    if source_metadata.get("split") != args.split:
        raise ValueError("candidate bank split differs from requested split")
    with np.load(args.bank_dir / "bank.npz", allow_pickle=False) as payload:
        bank = {key: np.asarray(payload[key]) for key in ("aligned_vertices", "scores")}
    metrics, records = evaluate_max_score_bank(
        bank,
        data,
        geometry,
        dictionaries,
        projector,
        graph_diameter_mm=graph_diameter,
    )
    metadata = {
        "schema": "m1-max-score-bank-evaluation-v1",
        "split": args.split,
        "aggregation": "maximum_rho_candidate_first_tie",
        "refinement": "strict_rho_one_ring_slot_specific_four_pass",
        "truth_free_selection_and_refinement": True,
        "bank_sha256": sha256(args.bank_dir / "bank.npz"),
        "bank_summary_sha256": sha256(args.bank_dir / "summary.json"),
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(manifest_path),
        "anatomy_content_sha256": anatomy_content,
        "sensor_projector_sha256": sha256(args.sensor_projector),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", _json_finite({"metadata": metadata, "metrics": metrics}))
    write_json(args.output / "records.json", _json_finite(records))
    print(json.dumps(_json_finite({"metadata": metadata, "metrics": metrics}), indent=2))


if __name__ == "__main__":
    main()
