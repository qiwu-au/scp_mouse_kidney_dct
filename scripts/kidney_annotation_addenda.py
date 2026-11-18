#!/usr/bin/env python
"""kidney_annotation addenda: reviewer_core B.6, qc_mito kidney per-type tables and the Fig 2C caption under kidney_t15_label.

Requested by the authors on 14 Oct 2026, after adopting the kidney_annotation
(a) labels (`kidney_t15_label`). The completed tasks are not re-run; their kidney per-type summaries are regrouped by
the new label column with the original code (functions imported, nothing copied):
  - reviewer_core B.6: Mki67 / Top2a detection per kidney type (reviewer_core_kidney_background.detection_frame / summarize);
  - qc_mito: MitoCarta / OXPHOS summaries, within-label outlier flags and counts, variance explained, global-cutoff
    simulation, within-type depth correlation (qc_mito_mito_qc, qc_mito_mito_tests), and the dissociation-stress
    summaries, detection and type tests (qc_mito_lib, qc_mito_stress), on qc_mito's own per-cell tables;
  - Fig 2C: draft caption for the kidney_annotation dotplot (kidney_annotation_t15_dotplot_*_a_endo).
Each regrouping is also run for kidney_pre_label; those rows must equal the original reviewer_core / qc_mito tables (check).

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation_cluster_labels_per_cell_a_endo.csv and the
  a_endo dotplot tables; qc_mito kidney per-cell QC and stress tables and its summary tables; reviewer_core kidney_bg_by_label.
Outputs: results/tables/kidney_annotation/kidney_annotation_addendum_*.csv / .md

Command (project root):  python scripts/kidney_annotation_addenda.py
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
import qc_mito_lib as t3
import qc_mito_mito_qc as mq
import qc_mito_mito_tests as mt
import qc_mito_stress as st
import reviewer_core_kidney_background as kb
import cluster_de_lib as t5
import kidney_annotation_scan_lib as s

COL = "kidney_t15_label"
PRE = "kidney_pre_label"
KIDNEY_ANNOTATION_LABELS = cc.OUT_ROOTS["tables"] / s.TASK / "kidney_annotation_cluster_labels_per_cell_a_endo.csv"
QC_MITO = cc.OUT_ROOTS["tables"] / "qc_mito"
REVIEWER_CORE = cc.OUT_ROOTS["tables"] / "reviewer_core"
TOL = 1e-9


def compare(checks: t5.Checks, name: str, new: pd.DataFrame, old: pd.DataFrame, keys: list[str],
            cols: list[str]) -> None:
    """kidney_pre_label rows of a regrouped table must equal the original table (deterministic columns)."""
    a = new[new["label_column"] == PRE].set_index(keys)[cols]
    b = old[(old["label_column"] == PRE) & (old.get("dataset", "kidney") == "kidney")].set_index(keys)[cols]
    a, b = a.sort_index(), b.sort_index()
    same_rows = list(a.index) == list(b.index)
    diff = float(np.nanmax(np.abs(a.to_numpy(float) - b.reindex(a.index).to_numpy(float)))) if same_rows and len(a) else np.inf
    checks.add(f"{name}: kidney_pre_label rows equal the original table", f"{len(a)} rows; max abs diff {diff:.3g}",
               same_rows and diff < TOL, f"< {TOL}")


def caption(run: cc.Run, labels: pd.Series) -> str:
    order = pd.read_csv(run.use_input(cc.out_path(s.TASK, "tables", "t15_dotplot_marker_order_a_endo", ".csv"),
                                      "kidney_annotation a_endo dotplot marker order"), keep_default_na=False)
    sel = pd.read_csv(run.use_input(cc.out_path(s.TASK, "tables", "t15_dotplot_main_selection_a_endo", ".csv"),
                                    "kidney_annotation a_endo dotplot main selection"))
    values = pd.read_csv(run.use_input(cc.out_path(s.TASK, "tables", "t15_dotplot_values_a_endo", ".csv"),
                                       "kidney_annotation a_endo dotplot values (row order)"))
    counts = labels.value_counts()
    full = order[(order["version"] == "full") & (order["plotted"].astype(str) == "True")]
    main = order[(order["version"] == "main") & (order["plotted"].astype(str) == "True")]
    rep_full = full[full["repeat"].astype(str) == "True"]
    missing = order[(order["version"] == "full") & (order["in_protein_matrix"].astype(str) == "False")]
    miss_by = missing.groupby("group")["marker"].apply(lambda m: ", ".join(m))
    fb = sel[sel["selected"] & sel["selection"].str.startswith("fallback")]
    rows = values.drop_duplicates("group").sort_values("group_position")["group"].tolist()   # the figure's row order
    row_txt = ", ".join(f"{r} ({int(counts[r])})" for r in rows if r in counts.index)
    enc = ("Dot size: fraction of the group's cells in which the protein was quantified (intensity > 0). Colour: group "
           "mean of the per-protein z-scored ln(1 + intensity), rescaled from 0 (lowest group) to 1 (highest group) "
           "for each protein across the groups shown. No dot: not quantified in any cell of the group.")
    lab = ("Rows: kidney cell types from cluster-level annotation (stored neighbour graph, Leiden resolution 2.0; a "
           "cluster takes a cell type when a marker listed only for that type is significantly enriched in it, "
           "Methods); numbers in brackets = cells. Unknown: clusters without such a marker. Rows shown: "
           f"{row_txt}.")
    place = ("Proteins: curated markers (Supplementary Table 1, kidney tab, plus the added endothelial row: "
             f"{', '.join(s.ENDOTHELIAL_MARKERS)}) quantified in the 4,054-protein matrix. Each protein is placed "
             "under the cell type with the highest group mean among the types that list it; groups follow the row "
             "order.")
    rep = ("" if rep_full.empty else
           " † marks a listed marker repeated so that every cell type has a column ("
           + "; ".join(f"{r.group}: {r.marker}" for r in rep_full.itertuples(index=False)) + ").")
    fbt = ("" if fb.empty else " Types without a marker quantified in >= 25% of their cells show the marker with the "
           "highest specificity (" + "; ".join(f"{r.group}: {r.marker}" for r in fb.itertuples(index=False)) + ").")
    L = ["# kidney_annotation addendum: draft captions for the revised Fig 2C (kidney_annotation labels, option (a))", "",
         f"(`scripts/kidney_annotation_addenda.py`, {cc.now_utc()}). Draft for the authors; "
         "objective description only. Figures: `results/figures/kidney_annotation/kidney_annotation_t15_dotplot_"
         "{main,full}_a_endo` (PDF, 600-dpi PNG, TIFF). Row names follow `kidney_t15_label` (DCT1, PC); if the "
         "authors adopt other display names (e.g. DCT, CNT/PC), the figure must be redrawn with them.", "",
         "## Main version (Fig 2C replacement)", "",
         f"Marker proteins per kidney cell type. {lab} {enc} {place} Up to three proteins per type are shown "
         f"({len(main)} columns), chosen among those quantified in >= 25% of the type's cells and ranked by "
         f"specificity (type mean minus the highest mean among the other types).{fbt}", "",
         "## Full version (supplement)", "",
         f"All {len(full) - len(rep_full)} curated marker proteins quantified for the cell types shown ({len(full)} "
         f"columns).{rep} {enc} {place} Listed markers not quantified: "
         + "; ".join(f"{g}: {m}" for g, m in miss_by.items()) + ".", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="kidney_annotation addenda: reviewer_core B.6, qc_mito kidney tables, Fig 2C caption.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(s.TASK, "addenda", args)
    checks = t5.Checks(s.DATASET)
    kidney, klabels = t5.load(run, s.DATASET)
    t15 = pd.read_csv(run.use_input(KIDNEY_ANNOTATION_LABELS, "kidney_annotation adopted kidney labels (a)"), dtype={"cell_id": str}).set_index("cell_id")
    same = list(t15.index) == list(klabels.index)
    checks.add("kidney_annotation label table has the canonical cells in order", f"{same}", same)
    lab = t15[COL].astype(str)
    checks.add(f"{COL} counts", "; ".join(f"{k} {v}" for k, v in lab.value_counts().items()), None)

    # Point the qc_mito kidney label columns at the kidney_annotation labels (plus pre for the reproduction check).
    t3.DATASETS["kidney"] = {**t3.DATASETS["kidney"], "label_columns": (COL, PRE), "plot_columns": (COL, PRE)}
    t3.SHORT_COLUMN[COL] = "t15"
    t3.LABEL_DISPLAY[COL] = "kidney_annotation cluster label (option a)"

    # ---- qc_mito mito tables --------------------------------------------------------------------
    qc = pd.read_csv(run.use_input(QC_MITO / "qc_mito_kidney_qc_per_cell.csv", "qc_mito kidney per-cell QC"),
                     dtype={"cell_id": str}).set_index("cell_id").reindex(klabels.index)
    keep = [c for c in qc.columns if not c.startswith(("outlier__", "flag__", "evaluated__"))
            and c != "kidney_reannot_label"]
    frame = qc[keep].copy()
    frame[COL] = lab.to_numpy()
    frame, thr = mq.add_outlier_flags(frame, "kidney")
    counts, _ = mq.outlier_counts(frame, "kidney")
    summ = pd.concat([t3.summarise(frame, c, mq.SUMMARY_METRICS, "kidney") for c in (COL, PRE)], ignore_index=True)
    ve = mt.variance_explained({"kidney": frame})
    sim, tests = mt.global_cutoffs({"kidney": frame})
    wt = mt.within_type({"kidney": frame})
    o_summ = pd.read_csv(QC_MITO / "qc_mito_qc_summary_by_label.csv")
    o_cnt = pd.read_csv(QC_MITO / "qc_mito_outlier_counts.csv")
    o_ve = pd.read_csv(QC_MITO / "qc_mito_variance_explained.csv")
    o_sim = pd.read_csv(QC_MITO / "qc_mito_global_cutoff_simulation.csv")
    o_wt = pd.read_csv(QC_MITO / "qc_mito_within_type_depth_correlation.csv")
    for p in ("qc_summary_by_label", "outlier_counts", "variance_explained", "global_cutoff_simulation",
              "within_type_depth_correlation"):
        run.use_input(QC_MITO / f"qc_mito_{p}.csv", f"qc_mito {p} (reproduction check)")
    num = [c for c in summ.columns if c not in ("dataset", "label_column", "label", "metric")
           and pd.api.types.is_numeric_dtype(summ[c])]
    compare(checks, "qc_mito qc_summary_by_label", summ, o_summ, ["label", "metric"], num)
    compare(checks, "qc_mito outlier_counts", counts, o_cnt, ["label"], ["n_cells", "n_any", "n_mito_any", "n_mito_high"])
    compare(checks, "qc_mito variance_explained", ve, o_ve, ["metric", "scale", "depth_variable"],
            ["r2_label", "r2_depth", "r2_label_plus_depth"])
    compare(checks, "qc_mito global_cutoff_simulation", sim, o_sim, ["metric", "cutoff", "label"],
            ["n_cells", "n_removed", "percent_of_label_removed"])
    compare(checks, "qc_mito within_type_depth_correlation", wt, o_wt, ["metric", "depth_variable", "label"], ["rho"])

    # ---- qc_mito stress tables --------------------------------------------------------------------
    stress = pd.read_csv(run.use_input(QC_MITO / "qc_mito_kidney_stress_per_cell.csv", "qc_mito kidney stress per cell"),
                         dtype={"cell_id": str}).set_index("cell_id").reindex(klabels.index)
    stress[COL] = lab.to_numpy()
    _, present = st.coverage(kidney, "kidney")
    intensity = cc.dense(kidney.layers["intensity"])
    names = list(kidney.var_names.astype(str))
    s_summ, s_det, s_tests = [], [], []
    for c in (COL, PRE):
        s_summ.append(t3.summarise(stress, c, t3.STRESS_SCORES, "kidney"))
        for label in t3.label_order(stress[c], "kidney"):
            m = (stress[c].astype(str) == label).to_numpy()
            for p in present:
                d = intensity[m, names.index(p)] > 0
                s_det.append({"dataset": "kidney", "label_column": c, "label": label, "matrix_protein": p,
                              "n_cells": int(m.sum()), "n_detected": int(d.sum()), "percent_detected": 100 * d.mean()})
        for score in t3.STRESS_SCORES:
            summary, _ = t3.type_tests(stress, c, score, "PCT")
            s_tests.append({"dataset": "kidney", "variant": "all present proteins", **summary})
    s_summ = pd.concat(s_summ, ignore_index=True)
    o_ss = pd.read_csv(run.use_input(QC_MITO / "qc_mito_stress_summary_by_label.csv", "qc_mito stress summary (check)"))
    num = [c for c in s_summ.columns if c not in ("dataset", "label_column", "label", "metric")
           and pd.api.types.is_numeric_dtype(s_summ[c])]
    compare(checks, "qc_mito stress_summary_by_label", s_summ, o_ss, ["label", "metric"], num)

    # ---- reviewer_core B.6 ----------------------------------------------------------------------------
    kdet = kb.detection_frame(kidney, klabels)
    rows = []
    for by_bin in (False, True):
        rows += kb.summarize(kdet, pd.Series("all", index=klabels.index), "KIDNEY", "all_cells", by_bin)
        rows += kb.summarize(kdet, lab, "KIDNEY", COL, by_bin)
        rows += kb.summarize(kdet, klabels[PRE], "KIDNEY", PRE, by_bin)
    bg = pd.DataFrame(rows)
    o_bg = pd.read_csv(run.use_input(REVIEWER_CORE / "reviewer_core_kidney_bg_by_label.csv", "reviewer_core B.6 kidney table (check)"))
    o_bg = o_bg[o_bg["dataset"] == "KIDNEY"]
    compare(checks, "reviewer_core kidney_bg_by_label", bg[bg["depth_bin"] == "all"], o_bg.assign(dataset="kidney"), ["label"],
            ["n_cells", "n_Mki67_detected", "n_Top2a_detected", "n_mki67_or_top2a_detected"])

    if not checks.ok:
        run.table(checks.frame(), "addendum_checks")
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("A reproduction check failed; see kidney_annotation_addendum_checks.csv.")

    run.table(checks.frame(), "addendum_checks")
    run.table(summ, "addendum_t03_qc_summary_by_label")
    run.table(thr, "addendum_t03_outlier_thresholds")
    run.table(counts, "addendum_t03_outlier_counts")
    run.table(ve, "addendum_t03_variance_explained")
    run.table(sim, "addendum_t03_global_cutoff_simulation")
    run.table(tests, "addendum_t03_global_cutoff_tests")
    run.table(wt, "addendum_t03_within_type_depth_correlation")
    run.table(s_summ, "addendum_t03_stress_summary_by_label")
    run.table(pd.DataFrame(s_det), "addendum_t03_stress_detection_by_label")
    run.table(pd.DataFrame(s_tests), "addendum_t03_stress_type_tests")
    run.table(bg[bg["depth_bin"] == "all"], "addendum_t02_kidney_bg_by_label")
    run.table(bg[bg["depth_bin"] != "all"], "addendum_t02_kidney_bg_by_label_and_depth_bin")
    run.text(caption(run, lab), "addendum_fig2c_captions", ".md")
    run.finish({"label_column": COL, "comparison_column": PRE, "t03_functions": "qc_mito_mito_qc.add_outlier_flags / "
                "outlier_counts; qc_mito_lib.summarise / type_tests; qc_mito_mito_tests.variance_explained / "
                "global_cutoffs / within_type; qc_mito_stress.coverage", "t02_functions":
                "reviewer_core_kidney_background.detection_frame / summarize", "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
