#!/usr/bin/env python
"""kidney_annotation step 3c: detection of endothelial marker proteins in selected kidney cell groups (descriptive).

Follow-up of step 3b (14 Oct 2026): with the V3 + atlas
markers, the stored-graph Leiden 2.0 cluster that holds the 19 Fig 2A 'MD' cells was labelled AVR. This script
tabulates, per protein, the fraction of cells with intensity > 0 and the mean canonical z in a few groups. No test,
no label is assigned.

Proteins: the atlas-derived markers selected for Endo, AVR and DVR (kidney_annotation_atlas_markers.csv) and the V3 DVR /
AVR markers quantified in the matrix.
Groups: all 712 cells; Fig 2A MD, DVR and JG; stored graph Leiden 2.0 cluster 20; depth-regressed graph Leiden 2.0
cluster 11 (seed 42; kidney_annotation_scan_clusters_per_cell.csv).

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation_atlas_markers.csv; KIDNEY_markers_V3.xlsx;
  kidney_annotation_scan_clusters_per_cell.csv.
Output: results/tables/kidney_annotation/kidney_annotation_extmarkers_endothelial_detection.csv

Command (project root):  python scripts/kidney_annotation_endothelial_detection.py
"""

from __future__ import annotations
# Select the project environment before importing analysis libraries.
if __name__ == "__main__":
    import sys as _runtime_sys
    from pathlib import Path as _RuntimePath
    _runtime_root = next(
        parent for parent in _RuntimePath(__file__).resolve().parents
        if (parent / "_scanpy_runtime.py").is_file()
    )
    _runtime_sys.path.insert(0, str(_runtime_root))
    from _scanpy_runtime import ensure_scanpy_environment as _select_project_python
    _select_project_python(__file__)


import argparse

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import kidney_annotation_scan_lib as s


def main() -> None:
    parser = argparse.ArgumentParser(description="kidney_annotation step 3c: endothelial marker detection (descriptive).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(s.TASK, "endothelial_detection", args)
    adata, labels = t5.load(run, s.DATASET)
    var = adata.var_names.astype(str)
    det = pd.DataFrame(cc.dense(adata.layers["intensity"]) > 0, index=adata.obs_names.astype(str), columns=var)
    z = pd.DataFrame(cc.dense(adata.X), index=adata.obs_names.astype(str), columns=var)
    atlas = s.read_table(run, "atlas_markers", keep_default_na=False)
    v3 = t5.read_markers(run, s.DATASET, var)
    scan = s.read_table(run, "scan_clusters_per_cell", dtype=str).set_index("cell_id").reindex(det.index)
    proteins = list(dict.fromkeys(list(atlas.loc[atlas["scp_type"].isin(["Endo", "AVR", "DVR"]), "protein"])
                                  + v3.present.get("DVR", []) + v3.present.get("AVR", [])))
    groups = {
        "all cells": np.ones(len(det), bool),
        "Fig 2A MD": (labels["kidney_pre_label"] == "MD").to_numpy(),
        "Fig 2A DVR": (labels["kidney_pre_label"] == "DVR").to_numpy(),
        "Fig 2A JG": (labels["kidney_pre_label"] == "JG").to_numpy(),
        "(a) stored, Leiden 2.0, cluster 20": (scan["stored__r2.0__cluster"] == "20").to_numpy(),
        "(b) regressed, Leiden 2.0, cluster 11": (scan["regressed__r2.0__cluster"] == "11").to_numpy(),
    }
    rows = []
    for p in proteins:
        row = {"protein": p, "atlas_marker_for": ";".join(atlas.loc[atlas["protein"] == p, "scp_type"]),
               "v3_listed_for": ";".join(sorted(v3.types_by_gene.get(p, set())))}
        for gname, m in groups.items():
            row[f"{gname}: n cells"] = int(m.sum())
            row[f"{gname}: detected"] = float(det.loc[m, p].mean())
            row[f"{gname}: mean z"] = float(z.loc[m, p].mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("Fig 2A MD: detected", ascending=False).reset_index(drop=True)
    same = bool((groups["Fig 2A MD"] == groups["(a) stored, Leiden 2.0, cluster 20"]).all())
    run.note(f"Fig 2A MD cells and stored-graph Leiden 2.0 cluster 20 are the same cells: {same}")
    run.table(out, "extmarkers_endothelial_detection")
    run.finish({"groups": {k: int(v.sum()) for k, v in groups.items()}, "md_equals_stored_cluster20": same})


if __name__ == "__main__":
    main()
