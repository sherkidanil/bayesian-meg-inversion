#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/local/tester/meg")
DATASETS = {
    "dG=0": ROOT / "data_anna_shift/anna_dg0.npz",
    "dG!=0": ROOT / "data_anna_shift/anna_dgnz.npz",
}
REFERENCE = ROOT / "new_version_mesh_experiments/best_examples/test_dataset_mesh.npz"


def describe(value):
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def item_or_repr(value):
    arr = np.asarray(value)
    if arr.shape == ():
        try:
            return arr.item()
        except ValueError:
            pass
    return repr(value)


def exact_sample_hashes(array):
    return [hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest() for x in array]


def stats_mm(values_m):
    values = np.asarray(values_m, dtype=np.float64) * 1000.0
    return {
        "shape": list(values.shape),
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "std_mm": float(np.std(values)),
        "min_mm": float(np.min(values)),
        "max_mm": float(np.max(values)),
    }


def main():
    ref = np.load(REFERENCE, allow_pickle=True)
    reference_shapes = {key: describe(ref[key]) for key in ref.files}
    result = {"reference": str(REFERENCE), "reference_shapes": reference_shapes, "datasets": {}}

    for label, path in DATASETS.items():
        d = np.load(path, allow_pickle=True)
        entry = {
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "keys": list(d.files),
            "arrays": {},
        }
        for key in d.files:
            entry["arrays"][key] = describe(d[key])

        for key in ("params_spatial", "params_temporal", "params_noise", "N_src", "Fs", "meta"):
            if key in d.files:
                entry.setdefault("metadata", {})[key] = item_or_repr(d[key])

        vertices = np.asarray(d["vertices"])
        edge_index = np.asarray(d["edge_index"])
        edge_weight = np.asarray(d["edge_weight_mm"])
        entry["mesh_compatibility"] = {
            "vertices_shape_equal": vertices.shape == ref["vertices"].shape,
            "vertices_exact_equal": bool(np.array_equal(vertices, ref["vertices"])),
            "vertices_max_abs_diff_m": float(np.max(np.abs(vertices - ref["vertices"]))) if vertices.shape == ref["vertices"].shape else None,
            "edge_index_shape_equal": edge_index.shape == ref["edge_index"].shape,
            "edge_index_exact_equal": bool(np.array_equal(edge_index, ref["edge_index"])),
            "edge_weight_shape_equal": edge_weight.shape == ref["edge_weight_mm"].shape,
            "edge_weight_exact_equal": bool(np.array_equal(edge_weight, ref["edge_weight_mm"])),
            "edge_weight_max_abs_diff_mm": float(np.max(np.abs(edge_weight - ref["edge_weight_mm"]))) if edge_weight.shape == ref["edge_weight_mm"].shape else None,
        }

        c = np.asarray(d["centroids_idx"], dtype=np.int64)
        cs = np.asarray(d["centroids_idx_shifted"], dtype=np.int64)
        coordinate_delta_m = np.linalg.norm(vertices[c] - vertices[cs], axis=-1)
        stored_delta_m = np.asarray(d["real_inputs_delta_m"], dtype=np.float64)
        entry["input_delta_stored"] = stats_mm(stored_delta_m)
        entry["input_delta_recomputed"] = stats_mm(coordinate_delta_m)
        entry["stored_vs_recomputed"] = {
            "shape_equal": stored_delta_m.shape == coordinate_delta_m.shape,
            "max_abs_diff_m": float(np.max(np.abs(stored_delta_m - coordinate_delta_m))) if stored_delta_m.shape == coordinate_delta_m.shape else None,
        }

        theta = np.asarray(d["thetas"]).reshape(c.shape[0], c.shape[1], 3)
        theta_s = np.asarray(d["thetas_shifted"]).reshape(cs.shape[0], cs.shape[1], 3)
        entry["theta_consistency"] = {
            "theta_max_abs_diff_from_vertex_m": float(np.max(np.abs(theta - vertices[c]))),
            "theta_shifted_max_abs_diff_from_vertex_m": float(np.max(np.abs(theta_s - vertices[cs]))),
        }

        ys = np.asarray(d["Ys"])
        ys_shifted = np.asarray(d["Ys_shifted"])
        entry["sensor_inputs"] = {
            "n_samples": int(ys.shape[0]),
            "reference_shape_per_sample": list(ref["Ys"].shape[1:]),
            "shape_per_sample_equal_reference": ys.shape[1:] == ref["Ys"].shape[1:],
            "time_samples_divisible_by_pool_12": bool(ys.shape[-1] % 12 == 0),
            "paired_shape_equal": ys.shape == ys_shifted.shape,
            "all_pairs_different": bool(np.all(np.any(ys != ys_shifted, axis=tuple(range(1, ys.ndim))))),
            "n_identical_pairs": int(np.sum(np.all(ys == ys_shifted, axis=tuple(range(1, ys.ndim))))),
            "mean_pair_l2": float(np.mean(np.linalg.norm((ys - ys_shifted).reshape(ys.shape[0], -1), axis=1))),
        }
        zero_shift = coordinate_delta_m == 0.0
        entry["zero_shift_diagnostics"] = {
            "n_zero_source_shifts": int(np.sum(zero_shift)),
            "n_samples_with_any_zero_shift": int(np.sum(np.any(zero_shift, axis=1))),
            "n_samples_with_all_zero_shifts": int(np.sum(np.all(zero_shift, axis=1))),
        }

        g = np.asarray(d["G_for_simulation"])
        hashes = exact_sample_hashes(g)
        entry["lead_fields"] = {
            "n_matrices": int(g.shape[0]),
            "n_unique_exact_sha256": len(set(hashes)),
            "first_hash": hashes[0],
            "all_equal_first": all(x == hashes[0] for x in hashes),
        }
        result["datasets"][label] = entry
        del g, ys, ys_shifted, d

    d0 = np.load(DATASETS["dG=0"], allow_pickle=True)
    d1 = np.load(DATASETS["dG!=0"], allow_pickle=True)
    result["cross_dataset"] = {
        key: bool(np.array_equal(d0[key], d1[key]))
        for key in (
            "centroids_idx",
            "centroids_idx_shifted",
            "real_inputs_delta_m",
            "thetas",
            "thetas_shifted",
            "vertices",
            "edge_index",
            "edge_weight_mm",
        )
    }

    out = ROOT / "reports/shift_paired_setup_20260810_135708/dataset_validation.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
