#!/usr/bin/env python3
"""Recompute finite geodesic metrics from saved V15 candidate predictions."""

import argparse
import json
from pathlib import Path

from megflow.dataset import sha256, source_hash
from megflow.geometry import CorticalGeometry
from megflow.v15_dataset import load_evaluation_arrays
from megflow.v15_evaluate import recompute_geodesic_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anatomy", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--disconnected-penalty-mm", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    records = json.loads(args.records.read_text())
    if not isinstance(records, list):
        raise ValueError("records JSON must contain a list")
    data = load_evaluation_arrays(args.dataset_root / args.split)
    geometry, _, anatomy_provenance = CorticalGeometry.load_bundle(args.anatomy)
    metrics = recompute_geodesic_summary(
        records,
        data,
        geometry,
        disconnected_penalty_mm=args.disconnected_penalty_mm,
        draw=args.draw,
    )
    result = dict(
        metadata=dict(
            records_sha256=sha256(args.records),
            dataset_manifest_sha256=sha256(
                args.dataset_root / args.split / "manifest.json"
            ),
            anatomy_sha256=sha256(args.anatomy),
            evaluator_code_sha256=source_hash(),
            anatomy_provenance=anatomy_provenance,
        ),
        metrics=metrics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
