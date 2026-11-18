#!/usr/bin/env python
"""kidney_annotation step 3d: V3 markers plus one endothelial row, on graphs (a) and (b) at Leiden 2.0 (kidney).

Requested by the authors on 14 Oct 2026 after step 3c showed
endothelial proteins in the 19 Fig 2A 'MD' cells. The endothelial row (Pecam1, Emcn, Eng, Esam, Icam2, Fabp4, Ehd4;
atlas Endo / AVR markers, none listed in V3) was proposed after that check, i.e. not fixed in advance; this is a
declared extension of the V3 workbook ('kidney V3 markers only', the project notes).

Marker sets: `v3` (re-run as a check against the rank-free tables) and `v3_endo` (V3 plus the 'Endothelial' row;
strict shared rule). Clusterings: stored graph (a) and depth-regressed graph (b) at Leiden 2.0, seeds 42 and 0-9.
Rules: top-25 (cluster_de_lib) and rank-free rescue (kidney_annotation_rankfree_sensitivity.rank_free). Minimum cluster
size 5.

Inputs (read-only): kidney_object canonical kidney object and labels; KIDNEY_markers_V3.xlsx; kidney_annotation graphs h5ad;
  kidney_annotation_scan_clusters_per_cell.csv, kidney_annotation_scan_seed_clusters_per_cell.csv,
  kidney_annotation_rankfree_clusters_per_cell.csv (reproduction checks).
Outputs: results/tables/kidney_annotation/kidney_annotation_v3endo_*.csv

Command (project root):  python scripts/kidney_annotation_v3_endothelial.py
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
import time

import numpy as np
import pandas as pd

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import kidney_annotation_scan_lib as s
from kidney_annotation_extended_markers import marker_set
from kidney_annotation_rankfree_sensitivity import as_assignment, cluster_rows, rank_free

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
GRAPHS = ("stored", "regressed")
RESOLUTION = 2.0
SETS = ("v3", "v3_endo")
RULES = ("top25", "rankfree")


def main() -> None:
    parser = argparse.ArgumentParser(description="kidney_annotation step 3d: V3 + endothelial row at Leiden 2.0, graphs a and b.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(s.TASK, "v3_endothelial", args)
    t_start = time.time()
    adata, labels = t5.load(run, s.DATASET)
    store = s.load_graphs(run)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    x_z = pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    cache = t5.rank_cache(np.log1p(intensity), detected)
    v3 = t5.read_markers(run, s.DATASET, var_names)
    sets = {"v3": v3, "v3_endo": marker_set({**v3.lists, s.ENDO_TYPE: list(s.ENDOTHELIAL_MARKERS)}, var_names)}
    specific = {k: {t: ms.specific_present(t) for t in ms.lists if ms.specific_present(t)} for k, ms in sets.items()}
    scan42 = s.read_table(run, "scan_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    seeds_tab = s.read_table(run, "scan_seed_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    rf42 = s.read_table(run, "rankfree_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    checks = t5.Checks(s.DATASET)
    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in
                                                               t5.EXPECTED_COUNTS[col]),
                   observed == t5.EXPECTED_COUNTS[col])
    missing = [g for g in s.ENDOTHELIAL_MARKERS if g not in set(var_names)]
    checks.add("endothelial row: every marker is quantified", ", ".join(missing) or "none", not missing)
    in_v3 = [g for g in s.ENDOTHELIAL_MARKERS if g in v3.types_by_gene]
    checks.add("endothelial row: no marker is listed in V3 (all type-specific)", ", ".join(in_v3) or "none", not in_v3)
    det_all = det_frame[list(s.ENDOTHELIAL_MARKERS)].mean()
    checks.add("endothelial row: detection in all 712 cells", "; ".join(f"{g} {v:.1%}" for g, v in det_all.items()), None)
    if not checks.ok:
        run.table(checks.frame(), "v3endo_checks")
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("A check failed; see kidney_annotation_v3endo_checks.csv.")

    per_cell = labels[[*s.LABEL_COLUMNS, "n_detected_proteins"]].copy()
    ids = labels["n_detected_proteins"].to_numpy(float)
    seed_rows, clus_all = [], []
    mism = {"clusters": 0, "v3_top25_seed42": 0, "v3_rankfree_seed42": 0, "v3_top25_seeds": 0}
    for graph in GRAPHS:
        g = s.graph_adata(store, graph)
        ref = {}
        for seed in s.SEEDS:
            name = s.clustering_name(graph, RESOLUTION, seed)
            cl = t5.leiden(g, RESOLUTION, "t15_endo_tmp", seed=seed).to_numpy().astype(str)
            src = scan42 if seed == s.SEED else seeds_tab
            mism["clusters"] += int((src[f"{name}__cluster"].to_numpy() != cl).sum())
            de_all = t5.de_one_vs_rest(cache, cl, var_names, name)
            row = {"graph": graph, "resolution": RESOLUTION, "seed": seed, "n_clusters": len(set(cl))}
            for m in SETS:
                de, _, asg = s.annotate_clustering(cache, cl, var_names, name, sets[m], x_z, det_frame, de=de_all)
                rf = rank_free(asg, de, specific[m])
                rf["top25_overlap"] = rf["cluster"].map(asg.set_index("cluster")["overlap"]).fillna("")
                labs = {r: s.per_cell_labels(cl, as_assignment(rf, r)).astype(str) for r in RULES}
                confs = {r: s.per_cell_labels(cl, as_assignment(rf, r), "confidence").astype(str) for r in RULES}
                if m == "v3" and seed == s.SEED:
                    mism["v3_top25_seed42"] += int((rf42[f"{name}__top25_label"].to_numpy() != labs["top25"]).sum())
                    mism["v3_rankfree_seed42"] += int((rf42[f"{name}__rankfree_label"].to_numpy() != labs["rankfree"]).sum())
                elif m == "v3":
                    mism["v3_top25_seeds"] += int((src[f"{name}__label"].to_numpy() != labs["top25"]).sum())
                if seed == s.SEED:
                    ref[m] = labs
                    if m == "v3_endo":
                        per_cell[f"{name}__cluster"] = cl
                        for r in RULES:
                            per_cell[f"{name}__{r}_label"] = labs[r]
                            per_cell[f"{name}__{r}_confidence"] = confs[r]
                    cr = cluster_rows(graph, RESOLUTION, cl, rf, labels, ids)
                    cr.insert(3, "marker_set", m)
                    clus_all.append(cr)
                for r in RULES:
                    types = s.types_resolved(as_assignment(rf, r))
                    row[f"{m}__{r}__n_types"] = len(types)
                    row[f"{m}__{r}__types"] = ";".join(f"{t}({c})" for t, c in types.items())
                    row[f"{m}__{r}__fraction_unknown"] = float((labs[r] == s.UNKNOWN).mean())
                    row[f"{m}__{r}__same_label_as_seed42"] = float((ref[m][r] == labs[r]).mean())
                    row[f"{m}__{r}__n_cells_endothelial"] = int((labs[r] == s.ENDO_TYPE).sum())
            seed_rows.append(row)
        r42 = seed_rows[-len(s.SEEDS)]
        print(f"[kidney] {graph} r2.0: v3 {r42['v3__rankfree__n_types']} / v3_endo {r42['v3_endo__rankfree__n_types']} "
              f"types (rank-free); {r42['v3_endo__rankfree__types']}", flush=True)
    for k, v in mism.items():
        checks.add(f"reproduction of the scan / rank-free tables: {k}", f"{v} mismatching cell entries", v == 0, "= 0")
    seeds = pd.DataFrame(seed_rows)
    rows = []
    for graph, grp in seeds.groupby("graph", sort=False):
        r42 = grp[grp["seed"] == s.SEED].iloc[0]
        other = grp[grp["seed"] != s.SEED]
        for m in SETS:
            for r in RULES:
                p = f"{m}__{r}__"
                rows.append({"graph": graph, "display": s.graph_display(graph), "resolution": RESOLUTION, "marker_set": m,
                             "rule": r, "n_clusters": int(r42["n_clusters"]), "n_types": int(r42[p + "n_types"]),
                             "types": r42[p + "types"], "n_types_all_seeds_min": int(grp[p + "n_types"].min()),
                             "n_types_all_seeds_max": int(grp[p + "n_types"].max()),
                             "fraction_unknown": float(r42[p + "fraction_unknown"]),
                             "n_cells_endothelial": int(r42[p + "n_cells_endothelial"]),
                             "seeds_with_endothelial_label": int((grp[p + "n_cells_endothelial"] > 0).sum()),
                             "same_label_as_seed42_median": float(other[p + "same_label_as_seed42"].median())})
    summ = pd.DataFrame(rows)
    for r in summ[summ["marker_set"] == "v3_endo"].itertuples(index=False):
        checks.add(f"v3_endo {r.graph} 2.0 {r.rule}: types (seeds); Endothelial cells (seeds with the label)",
                   f"{r.n_types} ({r.n_types_all_seeds_min}-{r.n_types_all_seeds_max}); {r.n_cells_endothelial} "
                   f"({r.seeds_with_endothelial_label}/11)", None)
    run.table(checks.frame(), "v3endo_checks")
    run.table(summ, "v3endo_summary")
    run.table(seeds, "v3endo_seed_stability")
    run.table(pd.concat(clus_all, ignore_index=True), "v3endo_cluster_annotation")
    run.table(per_cell, "v3endo_clusters_per_cell", index=True)
    run.finish({"endothelial_markers": list(s.ENDOTHELIAL_MARKERS), "graphs": list(GRAPHS), "resolution": RESOLUTION,
                "seeds": list(s.SEEDS), "rules": list(RULES), "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A reproduction check failed; see kidney_annotation_v3endo_checks.csv.")


if __name__ == "__main__":
    main()
