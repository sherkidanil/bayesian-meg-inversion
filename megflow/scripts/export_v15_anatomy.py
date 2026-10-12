#!/usr/bin/env python3
"""Export oct6 anatomy using the supplied script's exact cropped mesh semantics."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from megflow.geometry import CorticalGeometry
from megflow.anatomy import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import mne

    path = args.sample_path
    raw = path / "MEG/sample/sample_audvis_raw.fif"
    trans = path / "MEG/sample/sample_audvis_raw-trans.fif"
    bempath = path / "subjects/sample/bem/sample-5120-bem-sol.fif"
    # Preserve the reference default add_dist=True: patch metadata affects CPS normals.
    src = mne.setup_source_space(
        "sample", spacing="oct6", subjects_dir=path / "subjects", verbose=False
    )
    fwd = mne.make_forward_solution(
        raw,
        trans=trans,
        src=src,
        bem=mne.read_bem_solution(bempath, verbose=False),
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )
    fwd = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, copy=True, verbose=False
    )
    mag = mne.pick_types_forward(fwd, meg="mag", eeg=False)
    src = fwd["src"]
    vertices = np.concatenate([s["rr"][s["vertno"]] for s in src])
    normals = np.concatenate([s["nn"][s["vertno"]] for s in src])
    hemi = np.concatenate([np.full(len(s["vertno"]), i) for i, s in enumerate(src)])
    adjacency = mne.spatial_src_adjacency(src, verbose=False)
    ii, jj = adjacency.nonzero()
    keep = ii > jj
    ii, jj = ii[keep], jj[keep]
    weights = np.linalg.norm(vertices[ii] - vertices[jj], axis=1) * 1000.0
    graph = sparse.csr_matrix(
        (np.r_[weights, weights], (np.r_[ii, jj], np.r_[jj, ii])),
        shape=(len(vertices),) * 2,
    )
    components, labels = connected_components(graph, directed=False)
    geometry = CorticalGeometry(
        vertices * 1000, normals, graph, hemi, allow_fragmented_hemispheres=True
    )
    g = mag["sol"]["data"]
    if g.shape[0] != 102 or np.any(np.linalg.norm(g, axis=0) == 0):
        raise ValueError("invalid leadfield")
    # Verify the reference RawArray filtering order matches the forward channels.
    info = mne.io.read_info(raw, verbose=False)
    channels = [
        info["ch_names"][i]
        for i in mne.pick_types(info, meg="mag", eeg=False, exclude=[])
    ]
    if channels != mag["sol"]["row_names"]:
        raise ValueError("forward/raw magnetometer channel mismatch")
    inputs = [raw, trans, bempath] + [
        path / "subjects/sample/surf" / f"{h}.{s}"
        for h in ["lh", "rh"]
        for s in ["white", "sphere", "sphere.reg"]
    ]
    provenance = dict(
        spacing="oct6",
        add_dist=True,
        mindist_mm=5.0,
        orientation="fixed_surface_normal",
        graph_construction="supplied_v15_cropped_oct6",
        n_components=int(components),
        component_sizes=np.bincount(labels).tolist(),
        isolated_vertices=np.flatnonzero(np.diff(graph.indptr) == 0).tolist(),
        coordinate_frame=int(fwd["coord_frame"]),
        channel_names=channels,
        mne_version=mne.__version__,
        units="mm",
        leadfield_column_normalization=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        input_sha256={str(x): sha256(x) for x in inputs},
        builder_sha256=sha256(__file__),
        reference_sha256="e43a62983f5d8884040c24b3b25bb55060cff1b43aaf94a6502060da598a0e84",
    )
    geometry.save_bundle(args.output, g, provenance)
    print(
        dict(
            shape=g.shape,
            components=provenance["component_sizes"],
            sha256=sha256(args.output),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
