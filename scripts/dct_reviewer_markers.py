#!/usr/bin/env python
"""dct_reviewer step 1: reviewer_core A.1 Task 3 marker panels per DCT condition (combined search, dct_labels labels).

Calls the ctrl_reviewer_core step-1 functions
(``ctrl_reviewer_core_markers.score_task3``, ``marker_summary``; unchanged) once per condition with the
condition's label columns (``dct_reviewer_lib.bound``); the ctrl_reviewer_core file is not edited.

Inputs (read-only): dct_labels labelled objects and canonical labels; data/DCT/DCT_markers_V1.xlsx (ProLIF
row, legacy ProLIF context score).
Outputs:
  results/tables/dct_reviewer/dct_reviewer_task3_*.csv (both conditions; ``condition`` column)
  results/figures/dct_reviewer/dct_reviewer_task3_score_violins_<role>.pdf/.png/.tif
  results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_task3_scored.h5ad (dct_labels object + Task 3 scores and flags)

Within each condition (frozen ctrl_reviewer_core settings): panels with Umod in the TAL panel; score_genes on
the scaled .X (random_state 42, use_raw False); ambiguity flag = score > 90th percentile of that
condition's cells for the TAL, CNT/CD and interstitial panels. Detection = intensity > 0;
abundance = ln(1 + intensity) in detected cells.

Command (project root):  python scripts/dct_reviewer_markers.py
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
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import ctrl_reviewer_core_markers as m19
import dct_reviewer_lib as lib

ensure_scanpy_compat()
lib.style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "dct_reviewer_markers.py"
GROUPS = [("Positive DCT panels", ["DCT_core_score", "DCT1_associated_score", "DCT2_associated_score"]),
          ("Competing panels", ["TAL_score", "CNT_CD_score", "Interstitial_score"]),
          ("Context", ["cell_cycle_score", "ProLIF_legacy_score"])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 1: Task 3 marker panels per DCT condition.")
    parser.add_argument("--ambiguity-quantile", type=float, default=0.90)
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str, legacy: list[str], quantile: float) -> dict:
    s = lib.SETTINGS[cond]
    adata, labels = lib.load(run, cond)
    log_int = t2.ln1p_frame(adata)
    with lib.bound(s, m19):
        coverage, thresholds, score_cols, positive_cols, negative_cols, flag_cols = m19.score_task3(
            adata, legacy, quantile)
    keep = ["task3_dct_positive_label", "task3_dct_positive_score", "task3_dct_positive_margin", *score_cols,
            "max_tal_cnt_interstitial_score", "dominant_contamination_panel", *flag_cols, "ambiguous_tal_cnt_interstitial"]
    per_cell = labels[["cell_key", "plate", *s.LABEL_COLUMNS, "n_detected_proteins"]].join(
        adata.obs[keep].set_axis(labels.index, axis=0))

    summary_scores = score_cols + ["task3_dct_positive_score", "task3_dct_positive_margin", "max_tal_cnt_interstitial_score"]
    summary_flags = flag_cols + ["ambiguous_tal_cnt_interstitial"]
    frames = []
    for col in (*s.LABEL_COLUMNS, "task3_dct_positive_label"):
        summary = t2.summary_by(per_cell, col, summary_scores, summary_flags).rename(columns={col: "label"})
        summary.insert(0, "label_column", col)
        frames.append(summary)

    rows = []
    for col in s.LABEL_COLUMNS:
        table = pd.crosstab(per_cell[col], per_cell["task3_dct_positive_label"].astype(str))
        for label in cc.ordered_labels(table.index):
            for positive in t2.POSITIVE_PANELS:
                n = int(table.loc[label, positive]) if positive in table.columns else 0
                rows.append({"label_column": col, "label": label, "task3_dct_positive_label": positive,
                             "n_cells": n, "percent_of_label": 100 * n / table.loc[label].sum()})

    X = pd.DataFrame(cc.dense(adata.X), index=labels.index, columns=adata.var_names)
    with lib.bound(s, m19):
        marker_cov, marker_table = m19.marker_summary(log_int, X, labels)

    for col in ("task3_dct_positive_label", "dominant_contamination_panel"):
        adata.obs[col] = adata.obs[col].astype(str).astype("category")
    run.h5ad(adata, f"DCT_{s.KEY}_task3_scored")
    return {"coverage": pd.DataFrame(coverage), "thresholds": pd.DataFrame(thresholds),
            "per_cell": lib.to_generic(per_cell, cond).reset_index(),
            "summary": pd.concat(frames, ignore_index=True), "composition": pd.DataFrame(rows),
            "marker_cov": marker_cov, "marker_table": marker_table, "_per_cell": per_cell,
            "_thr": {t["score_column"]: t["threshold"] for t in thresholds}, "_legacy": legacy}


def plot_violins(results: dict, role: str, figs: lib.Figures) -> None:
    scores = [c for _, cols in GROUPS for c in cols]
    cols_of = {c: lib.SETTINGS[c].COLUMN_OF[role] for c in lib.CONDITIONS}
    order = lib.union_order(*(results[c]["_per_cell"][cols_of[c]] for c in lib.CONDITIONS))
    n_pairs = 2
    n_rows = int(np.ceil(len(scores) / n_pairs))
    fig = lib.new_figure(lib.FULL_MM, 54 * n_rows + 10)
    gs = fig.add_gridspec(n_rows, 2 * n_pairs, left=0.075, right=0.99, top=0.955, bottom=0.095, hspace=1.05,
                          wspace=0.32)
    for i, score in enumerate(scores):
        r, p = divmod(i, n_pairs)
        axes = [fig.add_subplot(gs[r, 2 * p])]
        axes.append(fig.add_subplot(gs[r, 2 * p + 1], sharey=axes[0]))
        for ax, cond in zip(axes, lib.CONDITIONS):
            pc = results[cond]["_per_cell"]
            lab = pc[cols_of[cond]].astype(str)
            groups = {l: pc.loc[lab == l, score].to_numpy(float) for l in order}
            lib.violins(ax, groups, order)
            thr = results[cond]["_thr"].get(score, np.nan)
            if np.isfinite(thr):
                ax.axhline(thr, color="#52514e", linestyle="--", linewidth=0.6)
            lib.count_ticks(ax, order, lab.value_counts().to_dict())
            lib.grid_y(ax)
            ax.set_title(f"{score.replace('_score', '')}: {cond}", fontsize=7)
            if ax is axes[1]:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel("score")
    fig.text(0.005, 0.995, f"Task 3 panel scores by {lib.ROLE_DISPLAY[role]} per condition (dot = median, bar = IQR; "
             "dashed = condition's 90th percentile flag threshold)", ha="left", va="top", fontsize=7)
    figs.save(fig, f"task3_score_violins_{role}", f"Task 3 panel scores by the {role} label, Ctrl and ZeroK",
              label_role=role)


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "task3_markers", args)
    lib.ported_from(run, "ctrl_reviewer_core_markers.py")
    legacy = cc.read_marker_workbook(run.use_input(cc.INPUTS["dct"]["markers"],
                                                   "DCT marker workbook (Supp Table 1 DCT tab)")).get("ProLIF", [])
    results = {c: one_condition(run, c, legacy, args.ambiguity_quantile) for c in lib.CONDITIONS}
    for key, name in (("coverage", "task3_marker_coverage"), ("thresholds", "task3_ambiguity_thresholds"),
                      ("per_cell", "task3_marker_scores_per_cell"), ("summary", "task3_marker_summary_by_label"),
                      ("composition", "task3_positive_label_composition"),
                      ("marker_cov", "task3_individual_marker_coverage"),
                      ("marker_table", "task3_individual_markers_by_label")):
        run.table(lib.stack({c: results[c][key] for c in lib.CONDITIONS}), name)
    run.note("Per-cell table: generic columns primary_label / second_label = cs_ctrl_deg_label / cs_ctrl_cluster_label "
             "(Ctrl) and cs_zerok_pipeline_label / cs_zerok_cluster_label (ZeroK).")
    run.note("Detection = intensity > 0 (dct_labels intensity layer); abundance = ln(1 + intensity) in detected cells. "
             "Thresholds are the 90th percentile within each condition.")
    figs = lib.Figures(run, "task3_markers")
    for role in lib.ROLES:
        plot_violins(results, role, figs)
    figs.finish()
    run.finish({"legacy_prolif_markers": legacy, "individual_markers": [m for m, _ in t2.INDIVIDUAL_MARKERS],
                "n_flagged_any": {c: int(results[c]["_per_cell"]["ambiguous_tal_cnt_interstitial"].sum())
                                  for c in lib.CONDITIONS},
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_task3_markers.csv.")


if __name__ == "__main__":
    main()
