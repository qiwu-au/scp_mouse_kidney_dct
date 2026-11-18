#!/usr/bin/env python
"""pt_tal step 6: power note: cell numbers, marker coverage, detectable effects, what cannot be resolved.

Analysis plan, step 6.

Contents
--------
- Cell numbers and median protein IDs per SCP group under each label set (PCT, PST, TAL, MD) and for the cells with
  the same PT label in both sets; atlas cells per class.
- Smallest effect detectable with 80% power at two-sided alpha 0.05 for the group sizes used in step 3 and for the
  Fig 2A MD vs TAL comparison: AUC by Noether's approximation for the Mann-Whitney test; Hedges' g (normal
  approximation). These are design quantities, not results.
- Panel and literature coverage (from steps 1-2).
- Items that cannot be resolved with the available data, each with the observed numbers behind it (read from the step
  tables; no interpretation).

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation labels; pt_tal step-0 to step-4 tables.
Outputs: results/tables/pt_tal/pt_tal_power_groups.csv, pt_tal_power_detectable.csv,
         pt_tal_power_unresolved.csv, pt_tal_power_checks.csv

Command (project root):  python scripts/pt_tal_power.py
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
import pt_tal_lib as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pt_tal step 6: power note.")
    cc.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "power", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, _, _, det = T.load_scp(run, checks)
    cov = T.read_table(run, "panel_coverage")
    lit = T.read_table(run, "literature")
    de = T.read_table(run, "atlas_de")
    batches = T.read_table(run, "atlas_batches")
    tests = T.read_table(run, "scores_pct_vs_pst")
    scan = T.read_table(run, "sub_scan")
    lc = T.read_table(run, "label_changes")
    ids = labels["n_detected_proteins"].astype(float)

    # groups ------------------------------------------------------------------------------------------------
    grp_rows = []
    for col in T.LABELS:
        for g in (*T.PT_LABELS, T.TAL_LABEL, T.MD_LABEL):
            m = labels[col].eq(g)
            if m.any():
                grp_rows.append({"grouping": T.LABEL_DISPLAY[col], "group": g, "n_cells": int(m.sum()),
                                 "median_protein_ids": float(ids[m].median()),
                                 "iqr_protein_ids": f"{ids[m].quantile(0.25):.0f}-{ids[m].quantile(0.75):.0f}"})
    cons = T.pt_groupings(labels)[T.CONSENSUS]
    for g in T.PT_LABELS:
        m = cons.eq(g)
        grp_rows.append({"grouping": T.GROUPING_TEXT[T.CONSENSUS], "group": g, "n_cells": int(m.sum()),
                         "median_protein_ids": float(ids[m].median()),
                         "iqr_protein_ids": f"{ids[m].quantile(0.25):.0f}-{ids[m].quantile(0.75):.0f}"})
    for c in T.REGION_CLASSES:
        b = batches[batches["Celltype"] == c]
        elig = b[b["n_cells"] >= T.BATCH_MIN_CELLS]
        grp_rows.append({"grouping": "atlas Celltype", "group": c, "n_cells": int(b["n_cells"].sum()),
                         "n_nuclei": int(b.loc[b["Source"] == "Nuclei", "n_cells"].sum()),
                         "n_studies": int(b["Origin"].nunique()),
                         "n_batches_ge_20_cells": int(len(elig))})
    groups = pd.DataFrame(grp_rows)

    # detectable effects ---------------------------------------------------------------------------------------
    det_rows = []
    pairs = [(T.GROUPING_TEXT[g], "PCT vs PST", *[int(T.pt_groupings(labels)[g].eq(k).sum()) for k in T.PT_LABELS])
             for g in ("t15", "pre", T.CONSENSUS)]
    pairs.append(("Fig 2A labels", "MD vs TAL", int(labels["kidney_pre_label"].eq("MD").sum()),
                  int(labels["kidney_pre_label"].eq("TAL").sum())))
    for grouping, comp, n1, n2 in pairs:
        rec = {"grouping": grouping, "comparison": comp, "n_1": n1, "n_2": n2,
               "min_detectable_auc_80pct_power": T.min_detectable_auc(n1, n2),
               "min_detectable_hedges_g_80pct_power": T.min_detectable_g(n1, n2)}
        if comp == "PCT vs PST":
            g = {v: k for k, v in T.GROUPING_TEXT.items()}[grouping]
            t = tests[(tests["grouping"] == g) & (tests["score"] == "S1/S2 - S3") & (tests["gene_set"] == "all")
                      & (tests["method"] == "mean_z")].iloc[0]
            rec.update(observed_auc_s1s2_minus_s3=float(t["auc"]), observed_hedges_g_s1s2_minus_s3=float(t["hedges_g"]))
        det_rows.append(rec)
        checks.add(f"{grouping} {comp}: n; smallest detectable AUC (80% power, alpha 0.05)",
                   f"{n1} / {n2}; {rec['min_detectable_auc_80pct_power']:.3f}", None)
    detectable = pd.DataFrame(det_rows)

    # unresolved items --------------------------------------------------------------------------------------------
    def cv(col, panel, key):
        return cov[(cov["label_column"] == col) & (cov["panel"] == panel)][key].iloc[0]

    def litr(g):
        return lit[lit["gene"] == g].iloc[0]

    md = labels["kidney_pre_label"].eq("MD")
    pts2 = de[de["contrast"] == "PTS2"]
    md_b = batches[batches["Celltype"] == "MD"]
    pt_lc = lc.set_index("label")
    n_same = int((labels["kidney_pre_label"].isin(T.PT_LABELS)
                  & labels["kidney_pre_label"].eq(labels["kidney_t15_label"])).sum())
    n_pre_pt = int(labels["kidney_pre_label"].isin(T.PT_LABELS).sum())
    cons_ids = tests[(tests["grouping"] == T.CONSENSUS) & (tests["score"] == "ln(protein IDs)")].iloc[0]
    sc_pt = scan[(scan["segment"] == "PT") & (scan["graph"] == "primary")].sort_values("resolution").iloc[0]
    sc_tal = scan[(scan["segment"] == "TAL") & (scan["graph"] == "primary")].sort_values("resolution").iloc[0]
    items = [
        ("S2 (atlas PTS2) panel", "no protein panel",
         f"atlas PTS2 vs other PT classes: {int(((pts2['q_bh'] < T.Q_MAX) & (pts2['log2fc'] > 0)).sum())} genes up at "
         f"q < {T.Q_MAX}; {int(cv(T.PRIMARY, 'PTS2', 'n_pass_q_lfc_detection'))} also with log2 FC >= {T.LFC_MIN:g} "
         f"and detection >= {T.PCT_MIN:.0%}; {int(cv(T.PRIMARY, 'PTS2', 'n_pass_rna_filters_incl_batch'))} after the "
         "batch-sign rule"),
        ("MTAL panel", "one protein (kidney_annotation) / none (Fig 2A)",
         f"{int(cv(T.PRIMARY, 'MTAL', 'n_pass_rna_filters_incl_batch'))} genes pass the RNA filters; with SCP detection "
         f">= {T.SCP_DET_MIN:.0%} of TAL cells: {int(cv(T.PRIMARY, 'MTAL', 'n_in_panel'))} under kidney_annotation "
         f"({cv(T.PRIMARY, 'MTAL', 'panel_genes')}), {int(cv('kidney_pre_label', 'MTAL', 'n_in_panel'))} under Fig 2A"),
        ("CTAL vs MTAL in SCP", "no SCP TAL subgroups",
         f"TAL is one label in both sets (kidney_annotation {int(labels['kidney_t15_label'].eq('TAL').sum())}, Fig 2A "
         f"{int(labels['kidney_pre_label'].eq('TAL').sum())} cells); TAL subclusters (primary graph, r "
         f"{sc_tal['resolution']}): eta2 of ln(protein IDs) {sc_tal['eta2_ln_ids']:.2f}; CTAL panel "
         f"{int(cv(T.PRIMARY, 'CTAL', 'n_in_panel'))} proteins"),
        ("MD", "not resolved",
         f"Nos1 detected in {int(det.loc[md, 'Nos1'].sum())} of {int(md.sum())} Fig 2A MD cells and "
         f"{int(det['Nos1'].sum())} of {len(det)} cells overall; Ptgs2 not quantified; the 19 Fig 2A MD cells are kidney_annotation "
         f"Endothelial; atlas MD = {int(md_b['n_cells'].sum())} nuclei from {md_b['Origin'].nunique()} studies "
         f"({int((md_b['n_cells'] >= T.BATCH_MIN_CELLS).sum())} batch with >= {T.BATCH_MIN_CELLS} cells)"),
        ("Literature S3 candidate Slc7a13", "not quantified as protein",
         f"in atlas: {bool(litr('Slc7a13')['in_atlas'])}"),
        ("Literature S1/S2 candidate Slc22a6", "not an atlas gene",
         f"protein detected in {int(litr('Slc22a6')['n_cells_detected_all_712'])} of 712 cells"),
        ("Literature TAL candidate Cldn16", "not quantified as protein and not an atlas gene", ""),
        ("Literature TAL candidates Cldn10 / Cldn19", "low protein detection",
         f"Cldn10 {int(litr('Cldn10')['n_cells_detected_all_712'])}, Cldn19 "
         f"{int(litr('Cldn19')['n_cells_detected_all_712'])} of 712 cells"),
        ("PCT / PST label sets", "labels differ between the sets",
         f"{n_same} of {n_pre_pt} Fig 2A PCT / PST cells keep the same PT label under kidney_annotation; kidney_annotation PST cells with another "
         f"Fig 2A label: {pt_lc.loc['PST', 't15_cells_from_fig2a']}; kidney_annotation PCT cells with another Fig 2A label: "
         f"{pt_lc.loc['PCT', 't15_cells_from_fig2a']}"),
        ("Cells with the same PT label in both sets", "PCT and PST separated by protein IDs",
         f"PCT {int(cons.eq('PCT').sum())} vs PST {int(cons.eq('PST').sum())} cells; AUC of ln(protein IDs) PCT vs "
         f"PST {cons_ids['auc']:.3f}"),
        ("PT subclusters", "depth-associated",
         f"primary graph r {sc_pt['resolution']}: eta2 of ln(protein IDs) {sc_pt['eta2_ln_ids']:.2f}; ARI with kidney_annotation / "
         f"Fig 2A labels {sc_pt['ari_t15']:.2f} / {sc_pt['ari_pre']:.2f}"),
    ]
    unresolved = pd.DataFrame(items, columns=["item", "status", "observed"])
    for r in unresolved.itertuples():
        checks.add(f"unresolved: {r.item} ({r.status})", r.observed or "-", None)
    run.table(groups, "power_groups")
    run.table(detectable, "power_detectable")
    run.table(unresolved, "power_unresolved")
    run.table(checks.frame(), "power_checks")
    run.finish({"alpha": 0.05, "power": 0.80, "auc_formula": "Noether: AUC - 0.5 = (z_a/2 + z_b) / sqrt(12 c (1-c) N)",
                "g_formula": "(z_a/2 + z_b) sqrt(1/n1 + 1/n2)", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")


if __name__ == "__main__":
    main()
