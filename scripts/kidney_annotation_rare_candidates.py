#!/usr/bin/env python
"""kidney_annotation step 2, plan step 4: rare-type candidate cells from type-specific quantified markers (kidney).

Plan:
the analysis plan.

Flags, not labels. A V3 marker is type-specific if only one workbook type lists it (strict cluster_de rule). Per cell and
flagged type: the type-specific quantified markers detected (intensity > 0); a cell is a candidate if >= 1 is
detected ('>= 2' also reported for types with >= 2 such markers). Flagged types: JG, ATL, DCT1, PC, IC-A, IC-B, Podo,
plus PCT, PST and TAL (plan). Candidates are located on both submitted label columns, the stored Leiden 2.0
clusters and every seed-42 clustering of the resolution scan. Types without a type-specific quantified marker are
listed as a coverage limit.

Inputs (read-only): kidney_object canonical kidney object and labels; data/KIDNEY/KIDNEY_markers_V3.xlsx;
  kidney_annotation_scan_clusters_per_cell.csv (seed-42 clusters and labels of the scan).
Outputs: results/tables/kidney_annotation/kidney_annotation_rare_*.csv

Command (project root):  python scripts/kidney_annotation_rare_candidates.py
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

FLAGGED = (*s.RARE_TYPES, *s.ALSO_FLAGGED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation step 2 / plan step 4: rare-type candidate flags.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "rare_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_annotation_rare_checks.csv (nothing else written).")


def counts_text(values) -> str:
    vc = pd.Series(values).astype(str).value_counts()
    return "; ".join(f"{k} {v}" for k, v in vc.items())


def locate(cand: np.ndarray, clusters: np.ndarray, cl_label: pd.Series, cl_conf: pd.Series, cell_type: str) -> dict:
    """Where the candidate cells of one type sit in one clustering."""
    clusters = np.asarray(clusters).astype(str)
    sizes = pd.Series(clusters).value_counts()
    n_cand = int(cand.sum())
    out = {"n_candidates": n_cand}
    if n_cand == 0:
        return out
    in_cl = pd.Series(clusters[cand]).value_counts()
    top = str(in_cl.index[0])
    frac_of_cluster = (in_cl / sizes.reindex(in_cl.index)).sort_values(ascending=False)
    eligible = frac_of_cluster[sizes.reindex(frac_of_cluster.index) >= s.MIN_CLUSTER_SIZE]
    lab_of_cand = pd.Series(clusters[cand]).map(cl_label)
    out.update({
        "n_clusters_spanned": int(len(in_cl)),
        "top_cluster": top, "top_cluster_size": int(sizes[top]), "n_candidates_in_top_cluster": int(in_cl.iloc[0]),
        "fraction_of_candidates_in_top_cluster": float(in_cl.iloc[0] / n_cand),
        "fraction_of_top_cluster_candidates": float(in_cl.iloc[0] / sizes[top]),
        "top_cluster_label": str(cl_label.get(top, "")), "top_cluster_confidence": str(cl_conf.get(top, "")),
        "max_candidate_fraction_cluster": str(eligible.index[0]) if len(eligible) else "",
        "max_candidate_fraction": float(eligible.iloc[0]) if len(eligible) else np.nan,
        "max_candidate_fraction_cluster_size": int(sizes[eligible.index[0]]) if len(eligible) else np.nan,
        "max_candidate_fraction_cluster_label": str(cl_label.get(eligible.index[0], "")) if len(eligible) else "",
        "n_candidates_in_clusters_labelled_type": int((lab_of_cand == cell_type).sum()),
        "n_candidates_in_below_min_clusters": int((lab_of_cand == s.BELOW_MIN).sum()),
        "candidate_cluster_labels": counts_text(lab_of_cand),
    })
    return out


def main() -> None:
    args = parse_args()
    run = cc.Run(s.TASK, "rare_candidates", args)
    adata, labels = t5.load(run, s.DATASET)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = pd.DataFrame(cc.dense(adata.layers["intensity"]).astype(np.float64), index=cell_ids, columns=var_names)
    markers = t5.read_markers(run, s.DATASET, var_names)
    scan = s.read_table(run, "scan_clusters_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    checks = t5.Checks(s.DATASET)

    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in
                                                               t5.EXPECTED_COUNTS[col]),
                   observed == t5.EXPECTED_COUNTS[col])
    same = list(scan.index) == list(cell_ids)
    checks.add("scan per-cell table has the canonical cells in order", f"{same}", same)
    for t, expected in s.BRIEF_TYPE_SPECIFIC.items():
        got = tuple(markers.specific_present(t))
        checks.add(f"{t}: type-specific quantified markers equal the plan", ";".join(got) or "none",
                   set(got) == set(expected), ";".join(expected))
    for t in s.BRIEF_NO_SPECIFIC:
        checks.add(f"{t}: no type-specific quantified marker (plan)",
                   f"specific {';'.join(markers.specific_present(t)) or 'none'}; quantified "
                   f"{';'.join(markers.present[t])}", not markers.specific_present(t))
    for t in s.ALSO_FLAGGED:
        checks.add(f"{t}: type-specific quantified markers (information)", ";".join(markers.specific_present(t)), None)
    checks.add("AVR: quantified markers", ";".join(markers.present.get("AVR", [])) or "none", None)
    other = [t for t in markers.lists if t not in (*FLAGGED, *s.BRIEF_NO_SPECIFIC, "AVR") and markers.specific_present(t)]
    checks.add("no other workbook type has a type-specific quantified marker", ";".join(other) or "none", not other)
    if not checks.ok:
        stop(run, checks)

    # Per cell ----------------------------------------------------------------------------------
    per_cell = labels[[*s.LABEL_COLUMNS, "n_detected_proteins", "stored_leiden_res_2.00"]].copy()
    marker_rows, summary_rows, by_label_rows = [], [], []
    cand = {}
    for t in FLAGGED:
        genes = markers.specific_present(t)
        det = intensity[genes] > 0
        for gname in genes:
            per_cell[f"{t}__{gname}__detected"] = det[gname].astype(int).to_numpy()
            d = det[gname].to_numpy()
            marker_rows.append({"cell_type": t, "marker": gname, "n_cells_detected": int(d.sum()),
                                "detection_rate": float(d.mean()),
                                "median_ln1p_detected": float(np.log1p(intensity.loc[d, gname]).median()) if d.any() else np.nan,
                                "detected_by_kidney_reannot_label": counts_text(labels.loc[d, "kidney_reannot_label"]),
                                "detected_by_kidney_pre_label": counts_text(labels.loc[d, "kidney_pre_label"])})
        n = det.sum(axis=1).to_numpy()
        per_cell[f"{t}__n_specific_detected"] = n
        per_cell[f"{t}__candidate"] = (n >= 1).astype(int)
        cand[t] = n >= 1
        row = {"cell_type": t, "n_specific_markers": len(genes), "specific_markers": ";".join(genes),
               "n_candidates_ge1": int((n >= 1).sum()), "n_candidates_ge2": int((n >= 2).sum()) if len(genes) >= 2 else np.nan}
        for col in s.LABEL_COLUMNS:
            lab = labels[col].astype(str).to_numpy()
            short = "reannot" if col == "kidney_reannot_label" else "pre"
            row[f"n_cells_labelled_{short}"] = int((lab == t).sum())
            row[f"n_labelled_{short}_that_are_candidates"] = int(((lab == t) & (n >= 1)).sum())
            row[f"candidates_by_{short}_label"] = counts_text(lab[n >= 1])
            for lab_value, k in pd.Series(lab[n >= 1]).value_counts().items():
                by_label_rows.append({"cell_type": t, "label_column": col, "label": lab_value, "n_candidates": int(k),
                                      "n_cells_with_label": int((lab == lab_value).sum())})
        stored = labels["stored_leiden_res_2.00"].astype(str).to_numpy()
        in_cl = pd.Series(stored[n >= 1]).value_counts()
        row["stored_leiden2_clusters_spanned"] = int(len(in_cl))
        row["stored_leiden2_top_cluster"] = str(in_cl.index[0]) if len(in_cl) else ""
        row["stored_leiden2_top_cluster_n_candidates"] = int(in_cl.iloc[0]) if len(in_cl) else 0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    for r in summary.itertuples(index=False):
        checks.add(f"{r.cell_type}: candidates (>= 1 specific marker detected)",
                   f"{r.n_candidates_ge1} cells; labelled {r.cell_type}: reannot {r.n_labelled_reannot_that_are_candidates}"
                   f"/{r.n_cells_labelled_reannot}, pre {r.n_labelled_pre_that_are_candidates}/{r.n_cells_labelled_pre}",
                   None)

    # Location in every seed-42 clustering ------------------------------------------------------
    loc_rows = []
    pre = labels["kidney_pre_label"].astype(str)
    stored = labels["stored_leiden_res_2.00"].astype(str).to_numpy()
    stored_lab = pd.Series(pre.to_numpy()).groupby(stored).first()
    for t in FLAGGED:
        loc_rows.append({"graph": "submitted", "resolution": 2.0, "clustering": "stored Leiden 2.0 (label = "
                         "kidney_pre_label)", "cell_type": t,
                         **locate(cand[t], stored, stored_lab, pd.Series(dtype=str), t)})
        for graph in s.GRAPH_NAMES:
            for res in s.RESOLUTIONS:
                name = s.clustering_name(graph, res)
                cl = scan[f"{name}__cluster"].astype(str).to_numpy()
                frame = pd.DataFrame({"c": cl, "l": scan[f"{name}__label"].astype(str).to_numpy(),
                                      "f": scan[f"{name}__confidence"].astype(str).to_numpy()})
                first = frame.groupby("c").first()
                loc_rows.append({"graph": graph, "resolution": res, "clustering": name, "cell_type": t,
                                 **locate(cand[t], cl, first["l"], first["f"], t)})
    location = pd.DataFrame(loc_rows)

    # Coverage limits -------------------------------------------------------------------------
    cov_rows = []
    for t in (*s.BRIEF_NO_SPECIFIC, "AVR"):
        present = markers.present.get(t, [])
        cov_rows.append({"cell_type": t, "markers_listed": ";".join(markers.lists.get(t, [])),
                         "n_listed": len(markers.lists.get(t, [])), "markers_quantified": ";".join(present),
                         "n_quantified": len(present), "n_type_specific_quantified": len(markers.specific_present(t)),
                         "quantified_markers_shared_with": "; ".join(
                             f"{g}: {','.join(sorted(markers.types_by_gene[g] - {t}))}" for g in present),
                         "limit": ("no marker quantified: no flag possible" if not present else
                                   "quantified markers are all listed for other types too: a detected marker does "
                                   "not single out this type")})
    coverage = pd.DataFrame(cov_rows)

    run.table(checks.frame(), "rare_checks")
    run.table(pd.DataFrame(marker_rows), "rare_marker_detection")
    run.table(summary, "rare_candidates_summary")
    run.table(pd.DataFrame(by_label_rows), "rare_candidates_by_label")
    run.table(location, "rare_candidates_location")
    run.table(coverage, "rare_coverage_limits")
    run.table(per_cell, "rare_candidates_per_cell", index=True)
    run.finish({"flagged_types": list(FLAGGED), "specific_rule": "listed for exactly one type in the V3 workbook "
                "(cluster_de_lib.MarkerSets.is_shared)", "detected": "layers['intensity'] > 0",
                "candidate": ">= 1 type-specific quantified marker detected (>= 2 also reported)",
                "min_cluster_size": s.MIN_CLUSTER_SIZE, "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
