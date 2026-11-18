#!/usr/bin/env python
"""ctrl_reviewer_core step 3: reviewer_core A.3 ALDO score sensitivity on the control data (labels_core labels).

Port of
reviewer_core_aldo.py (unchanged method; inputs and label columns replaced; the
expression source 'V1_X' is named 'ctrl_X'). The labels_core object carries no stored
submission ALDO scores, so the reviewer_core 'submission' rows do not exist here.

Inputs (read-only): results/h5ad/ctrl_reviewer_core_DCT_ctrl_task3_scored.h5ad (step 1;
scaled .X, Task 3 flags) and results/h5ad/ctrl_reviewer_core_DCT_ctrl_cc_regressed.h5ad
(step 2, cell_cycle_score regressed); labels_core canonical labels.
Outputs:
  results/tables/ctrl_reviewer_core/ctrl_reviewer_core_aldo_*.csv
  results/figures/ctrl_reviewer_core/ctrl_reviewer_core_aldo_*.pdf/.png

Score families (ALDO_activity, ALDO_signed, ALDO_machinery) x variants (base,
no_Slc12a3, no_cell_cycle, no_Slc12a3_no_cell_cycle) = mean protein z-score
(ddof=0) of the UP panel minus the DOWN panel, computed on two expression
sources: the control .X ("ctrl_X") and cell-cycle-regressed expression ("cc_regressed_3a").
Contrasts: DCT2 vs DCT1, ProLIF vs DCT1, ProLIF vs DCT2 (Mann-Whitney), in all
cells and after excluding TAL/CNT/interstitial-flagged cells; BH-FDR within
label column x expression source.

Command (project root):  python scripts/ctrl_reviewer_core_aldo.py
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
import reviewer_core_lib as t2
import ctrl_reviewer_core_lib as lib

t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "ctrl_reviewer_core_aldo.py"
SOURCES = ("ctrl_X", "cc_regressed_3a")
COMPARISON_COLORS = {"DCT2_vs_DCT1": t2.SERIES[0], "ProLIF_vs_DCT1": t2.SERIES[1], "ProLIF_vs_DCT2": t2.SERIES[2]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 3: ALDO score sensitivity on the control data.")
    cc.add_common_args(parser)
    return parser.parse_args()


def summarize(per_cell: pd.DataFrame, label_col: str, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    cohorts = {"all_cells": np.ones(len(per_cell), dtype=bool),
               "exclude_TAL_CNT_interstitial_flagged": ~per_cell["exclude_tal_cnt_interstitial"].to_numpy(bool)}
    for cohort, mask in cohorts.items():
        for label, group in per_cell.loc[mask].groupby(label_col, observed=True):
            for score_col in score_cols:
                values = pd.to_numeric(group[score_col], errors="coerce").dropna()
                rows.append({"label_column": label_col, "cohort": cohort, "cell_type": str(label), "score": score_col,
                             "n_cells": len(values), "mean": values.mean(), "sd": values.std(ddof=1),
                             "median": values.median(), "q25": values.quantile(.25), "q75": values.quantile(.75)})
    return pd.DataFrame(rows)


def plot_forest(contrast: pd.DataFrame, label_col: str, run: cc.Run) -> None:
    sub = contrast[(contrast["label_column"] == label_col) & contrast["source"].isin(SOURCES)]
    keys = [f"{f}__{v}" for f in t2.ALDO_FAMILIES for v in t2.ALDO_VARIANTS]
    cohorts = ["all_cells", "exclude_TAL_CNT_interstitial_flagged"]
    cohort_title = {"all_cells": "all cells", "exclude_TAL_CNT_interstitial_flagged": "TAL/CNT/interstitial-flagged excluded"}
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6), sharey=True, sharex=True)
    for r, cohort in enumerate(cohorts):
        for c, source in enumerate(SOURCES):
            ax = axes[r, c]
            part = sub[(sub["cohort"] == cohort) & (sub["source"] == source)]
            for offset, comparison in zip((-0.25, 0, 0.25), COMPARISON_COLORS):
                p = (part[part["comparison"] == comparison].assign(key=lambda d: d["family"] + "__" + d["variant"])
                     .set_index("key").reindex(keys))
                y = np.arange(len(keys)) + offset
                sig = (p["p_fdr_bh"] < 0.05).to_numpy()
                color = COMPARISON_COLORS[comparison]
                ax.scatter(p["median_difference"][sig], y[sig], color=color, s=12, label=f"{comparison}, q < 0.05")
                ax.scatter(p["median_difference"][~sig], y[~sig], facecolors="white", edgecolors=color, s=12,
                           linewidths=0.8, label=f"{comparison}, q >= 0.05")
            ax.axvline(0, color=t2.INK["muted"], linewidth=0.6)
            ax.set_yticks(range(len(keys)), keys, fontsize=5.5)
            ax.invert_yaxis() if r == 0 and c == 0 else None
            ax.set_title(f"{source} · {cohort_title[cohort]}", fontsize=6.8)
            ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.5)
    fig.supxlabel("Median difference (group minus reference)", fontsize=7, color=t2.INK["secondary"])
    lines = []
    for cohort in cohorts:
        n = sub[sub["cohort"] == cohort].groupby("comparison")[["n_group", "n_reference"]].first()
        lines.append(f"{cohort_title[cohort]}: " + "; ".join(f"{k} n = {int(v.n_group)} vs {int(v.n_reference)}"
                                                             for k, v in n.iterrows()))
    fig.suptitle(f"ALDO contrasts by {lib.LABEL_DISPLAY[label_col]} (filled: BH q < 0.05 within label column x source)",
                 x=0.01, ha="left", fontsize=7.5, fontweight="bold")
    fig.text(0.01, 0.955, "\n".join(lines), ha="left", va="top", fontsize=5.6, color=t2.INK["secondary"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"aldo_contrasts_forest_by_{label_col}", "ALDO score contrasts between labels_core labels (control data)")


def plot_boxes(per_cell: pd.DataFrame, label_col: str, source: str, run: cc.Run) -> None:
    order = list(t2.CORE_LABELS)
    variants = ["base", "no_Slc12a3"]
    fig, axes = plt.subplots(len(t2.ALDO_FAMILIES), 2, figsize=(5.6, 1.9 * len(t2.ALDO_FAMILIES) + 0.5), sharey="row")
    excl = per_cell["exclude_tal_cnt_interstitial"].to_numpy(bool)
    for r, family in enumerate(t2.ALDO_FAMILIES):
        for c, (cohort, keep) in enumerate((("all cells", np.ones(len(per_cell), bool)),
                                            ("excluding TAL/CNT/interstitial flags", ~excl))):
            ax = axes[r, c]
            for i, label in enumerate(order):
                for j, variant in enumerate(variants):
                    vals = per_cell.loc[keep & (per_cell[label_col] == label).to_numpy(),
                                        f"{source}__{family}__{variant}"].dropna().to_numpy(float)
                    pos = i + (j - 0.5) * 0.36
                    bp = ax.boxplot(vals, positions=[pos], widths=0.3, showfliers=False, patch_artist=True)
                    for patch in bp["boxes"]:
                        patch.set_facecolor(t2.LABEL_PALETTE[label] if j == 0 else "white")
                        patch.set_edgecolor(t2.LABEL_PALETTE[label])
                    for part in ("medians", "whiskers", "caps"):
                        for line in bp[part]:
                            line.set_color(t2.INK["primary"] if part == "medians" else t2.LABEL_PALETTE[label])
                            line.set_linewidth(0.8)
            ax.set_xticks(range(len(order)), [f"{l}\n{int((keep & (per_cell[label_col] == l).to_numpy()).sum())}"
                                              for l in order])
            ax.set_title(f"{family} · {cohort}", fontsize=6.5)
            ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel("UP minus DOWN mean z" if family == "ALDO_signed" else "mean protein z-score")
    fig.suptitle(f"ALDO scores ({source}) by {lib.LABEL_DISPLAY[label_col]}; filled box = base, open box = no_Slc12a3; "
                 "numbers = cells", x=0.01, ha="left", fontsize=7, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"aldo_scores_{source}_by_{label_col}", f"ALDO scores ({source}) by labels_core label (control data)")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "aldo", args)
    lib.ported_from(run, "reviewer_core_aldo.py")
    adata = lib.read_h5ad(run, "DCT_ctrl_task3_scored")
    regressed = lib.read_h5ad(run, "DCT_ctrl_cc_regressed")
    if list(regressed.obs_names) != list(adata.obs_names) or list(regressed.var_names) != list(adata.var_names):
        raise ValueError("Regressed object does not share cells/proteins with the A.1 object.")
    _, labels = lib.load_ctrl(run)

    per_cell = labels[["plate", *lib.LABEL_COLUMNS]].copy()
    for col in t2.FLAG_COLUMNS:
        per_cell[col] = adata.obs[col].astype(bool).to_numpy()
    per_cell["exclude_tal_cnt_interstitial"] = per_cell[list(t2.FLAG_COLUMNS)].any(axis=1)
    frames = [t2.aldo_scores(adata, "ctrl_X"), t2.aldo_scores(regressed, "cc_regressed_3a")]
    per_cell = per_cell.join(pd.concat(frames, axis=1))
    score_cols = [c for f in frames for c in f.columns]
    for name in t2.ALDO_SUBMISSION_SCORES:
        if name in adata.obs:
            per_cell[f"submission__{name}"] = adata.obs[name].to_numpy(float)
            score_cols.append(f"submission__{name}")
    run.note("The labels_core object has no stored submission ALDO columns (" + ", ".join(t2.ALDO_SUBMISSION_SCORES)
             + "); the reviewer_core 'submission' rows do not apply.")

    run.table(t2.aldo_coverage(adata.var_names), "aldo_marker_coverage")
    run.table(per_cell.reset_index(), "aldo_scores_per_cell")

    summary = pd.concat([summarize(per_cell, col, score_cols) for col in lib.LABEL_COLUMNS], ignore_index=True)
    contrast = pd.concat([t2.aldo_contrast_table(per_cell, col, score_cols) for col in lib.LABEL_COLUMNS],
                         ignore_index=True)
    run.note("BH-FDR is applied within each label column x expression source, across cohorts, comparisons and scores.")
    run.table(summary, "aldo_summary_by_label")
    run.table(contrast, "aldo_contrasts")
    headline = contrast[contrast["variant"].isin(["base", "no_Slc12a3", "submission"])]
    run.table(headline[["label_column", "source", "family", "variant", "cohort", "comparison", "n_group", "n_reference",
                        "median_group", "median_reference", "median_difference", "cohens_d", "p_value", "p_fdr_bh"]],
              "aldo_contrasts_headline")

    for col in lib.LABEL_COLUMNS:
        for source in SOURCES:
            plot_boxes(per_cell, col, source, run)
        plot_forest(contrast, col, run)
    run.finish({"pairs": t2.ALDO_PAIRS, "expression_sources": [*SOURCES, "submission (stored obs)"]})


if __name__ == "__main__":
    main()
