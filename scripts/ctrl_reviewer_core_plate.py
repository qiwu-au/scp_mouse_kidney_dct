#!/usr/bin/env python
"""ctrl_reviewer_core step 5: reviewer_core A.5 (first part) plate stratification on the control data (labels_core labels).

Port of
reviewer_core_plate.py (unchanged method; inputs and label columns replaced). The
reviewer_core A.5 reconciliation with the V2 labels is dropped (ctrl_reviewer_core plan).

Inputs (read-only): results/h5ad/ctrl_reviewer_core_DCT_ctrl_task3_scored.h5ad (step 1); labels_core canonical labels.
Outputs:
  results/tables/ctrl_reviewer_core/ctrl_reviewer_core_plate_*.csv
  results/figures/ctrl_reviewer_core/ctrl_reviewer_core_plate_*.pdf/.png

Per plate (P1/P2 from the cell ID; pooled sort plates, a technical label only):
label fractions and chi-square, detected proteins per cell, Task 3 scores, and
pseudobulk Spearman correlation P1 vs P2 within each label (pseudobulk =
ln(1 + mean intensity), non-detected = 0; proteins with zero pseudobulk in
either plate are left out of that correlation).

Command (project root):  python scripts/ctrl_reviewer_core_plate.py
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
from scipy import stats

import scp_common as cc
import reviewer_core_lib as t2
import ctrl_reviewer_core_lib as lib

t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "ctrl_reviewer_core_plate.py"
SCORES = ("DCT_core_score", "DCT1_associated_score", "DCT2_associated_score", "TAL_score", "CNT_CD_score",
          "Interstitial_score", "ProLIF_legacy_score", "cell_cycle_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 5: plate stratification on the control data.")
    cc.add_common_args(parser)
    return parser.parse_args()


def groups(labels: pd.DataFrame):
    out = [("all_cells", "all", np.ones(len(labels), bool))]
    for col in lib.LABEL_COLUMNS:
        out += [(col, label, (labels[col] == label).to_numpy()) for label in cc.ordered_labels(labels[col])]
    return out


def pseudobulk_correlations(intensity: np.ndarray, labels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, points = [], {}
    plates = labels["plate"].to_numpy()
    for col, label, mask in groups(labels):
        p1, p2 = mask & (plates == "P1"), mask & (plates == "P2")
        row = {"label_column": col, "label": label, "n_cells_P1": int(p1.sum()), "n_cells_P2": int(p2.sum())}
        if p1.sum() and p2.sum():
            a = np.log1p(intensity[p1].mean(axis=0))
            b = np.log1p(intensity[p2].mean(axis=0))
            keep = (a > 0) & (b > 0)
            rho, p = stats.spearmanr(a[keep], b[keep])
            row.update(n_proteins=int(keep.sum()), spearman_rho=rho, p_value=p,
                       n_proteins_only_P1=int(((a > 0) & (b == 0)).sum()), n_proteins_only_P2=int(((b > 0) & (a == 0)).sum()))
            points[(col, label)] = (a[keep], b[keep], rho)
        rows.append(row)
    return pd.DataFrame(rows), points


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "plate", args)
    lib.ported_from(run, "reviewer_core_plate.py")
    adata = lib.read_h5ad(run, "DCT_ctrl_task3_scored")
    _, labels = lib.load_ctrl(run)
    scores = adata.obs[[s for s in SCORES if s in adata.obs]].set_axis(labels.index, axis=0)
    run.note("Control plates P1/P2 are sort plates of one pooled sample of 3 mice (authors, 28 Oct 2026); plate is a "
             "technical label only.")

    rows, tests = [], []
    for col in lib.LABEL_COLUMNS:
        table = pd.crosstab(labels["plate"], labels[col])
        for plate, counts in table.iterrows():
            for label in cc.ordered_labels(table.columns):
                rows.append({"label_column": col, "plate": plate, "label": label, "n_cells": int(counts[label]),
                             "fraction_of_plate": counts[label] / counts.sum()})
        chi2, p, dof, _ = stats.chi2_contingency(table.to_numpy())
        tests.append({"label_column": col, "test": "chi-square plate x label", "statistic": chi2, "dof": dof,
                      "p_value": p, "note": (f"includes Unknown (n={int(table['Unknown'].sum())})"
                                                   if "Unknown" in table.columns else "")})
    label_fractions = pd.DataFrame(rows)
    run.table(label_fractions, "plate_label_fractions")
    run.table(pd.DataFrame(tests), "plate_label_tests")

    rows = []
    nd = labels["n_detected_proteins"]
    for col, label, mask in groups(labels):
        p1 = nd[mask & (labels["plate"] == "P1").to_numpy()]
        p2 = nd[mask & (labels["plate"] == "P2").to_numpy()]
        row = {"label_column": col, "label": label, "n_P1": len(p1), "n_P2": len(p2),
               "median_P1": p1.median(), "median_P2": p2.median(), "q25_P1": p1.quantile(0.25), "q75_P1": p1.quantile(0.75),
               "q25_P2": p2.quantile(0.25), "q75_P2": p2.quantile(0.75)}
        result = cc.compare_groups(p2, p1)
        row.update(mannwhitney_p_P2_vs_P1=result["p_value"], rank_biserial_P2_vs_P1=result["rank_biserial"])
        rows.append(row)
    run.table(pd.DataFrame(rows), "plate_detected_proteins")

    rows = []
    frame = labels.join(scores)
    for col in lib.LABEL_COLUMNS:
        for label in cc.ordered_labels(labels[col]):
            sub = frame[frame[col] == label]
            for score_col in scores.columns:
                p1, p2 = sub.loc[sub["plate"] == "P1", score_col], sub.loc[sub["plate"] == "P2", score_col]
                row = {"label_column": col, "label": label, "score": score_col, "n_P1": len(p1), "n_P2": len(p2),
                       "median_P1": p1.median(), "median_P2": p2.median(), "mean_P1": p1.mean(), "mean_P2": p2.mean()}
                result = cc.compare_groups(p2, p1)
                row.update(median_difference_P2_minus_P1=result["median_difference"], p_value=result["p_value"])
                rows.append(row)
    run.table(cc.add_bh(pd.DataFrame(rows), ["label_column"]), "plate_marker_scores")

    intensity = np.expm1(t2.ln1p_frame(adata).to_numpy())
    corr, points = pseudobulk_correlations(intensity, labels)
    run.table(corr, "plate_pseudobulk_spearman")

    # Figures: label fractions and detected proteins by plate; pseudobulk scatter.
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.4, 1.4]})
    for ax, col in zip(axes[:2], lib.LABEL_COLUMNS):
        sub = label_fractions[label_fractions["label_column"] == col].pivot(index="plate", columns="label",
                                                                              values="fraction_of_plate")
        order = cc.ordered_labels(sub.columns)
        bottom = np.zeros(len(sub))
        for label in order:
            vals = 100 * sub[label].to_numpy(float)
            ax.bar(range(len(sub)), vals, bottom=bottom, color=t2.LABEL_PALETTE.get(label, "0.5"), width=0.7,
                   edgecolor="white", linewidth=0.8, label=label)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 6:
                    ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=5.2, color="white")
            bottom += vals
        n = labels["plate"].value_counts()
        ax.set_xticks(range(len(sub)), [f"{p}\n(n={n[p]})" for p in sub.index])
        ax.set_title(lib.LABEL_DISPLAY[col], fontsize=6.8)
        ax.set_ylabel("% of plate" if col == lib.LABEL_COLUMNS[0] else "")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=5.5)
    for ax, col in zip(axes[2:], lib.LABEL_COLUMNS):
        order = cc.ordered_labels(labels[col])
        for i, label in enumerate(order):
            for j, plate in enumerate(("P1", "P2")):
                vals = labels.loc[(labels[col] == label) & (labels["plate"] == plate), "n_detected_proteins"]
                bp = ax.boxplot(vals.to_numpy(float), positions=[i + (j - 0.5) * 0.38], widths=0.32, showfliers=False,
                                patch_artist=True)
                bp["boxes"][0].set_facecolor(cc.PLATE_PALETTE[plate])
                bp["boxes"][0].set_edgecolor(cc.PLATE_PALETTE[plate])
                for part in ("whiskers", "caps"):
                    for line in bp[part]:
                        line.set_color(cc.PLATE_PALETTE[plate])
                bp["medians"][0].set_color("white")
        ax.set_xticks(range(len(order)), order)
        ax.set_title(f"Detected proteins, {lib.LABEL_DISPLAY[col]}", fontsize=6.8)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[2].set_ylabel("Protein IDs per cell")
    from matplotlib.patches import Patch
    axes[3].legend(handles=[Patch(color=cc.PLATE_PALETTE[p], label=p) for p in ("P1", "P2")], loc="upper right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    t2.footer(fig, SCRIPT, "plates are pooled sort plates (technical label)")
    run.figure(fig, "plate_labels_and_detected_proteins", "Label fractions and protein IDs by plate")

    keys = [k for k in points]
    ncols = 4
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 1.9 * nrows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        a, b, rho = points[key]
        ax.scatter(a, b, s=1, alpha=0.35, color=t2.INK["secondary"], linewidths=0, rasterized=True)
        lim = [0, max(a.max(), b.max()) * 1.03]
        ax.plot(lim, lim, color=t2.INK["muted"], linewidth=0.5)
        ax.set_title(f"{key[1]} ({lib.LABEL_DISPLAY.get(key[0], 'all cells')})\nrho = {rho:.3f}, n = {len(a)}", fontsize=5.6)
        ax.set_xlabel("P1 ln(1 + mean intensity)", fontsize=5.5)
        ax.set_ylabel("P2 ln(1 + mean intensity)", fontsize=5.5)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "plate_pseudobulk_scatter", "Pseudobulk P1 versus P2 within each label")
    run.finish()


if __name__ == "__main__":
    main()
