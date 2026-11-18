#!/usr/bin/env python
"""dct_reviewer step 5: reviewer_core A.5 (first part) plate stratification per DCT condition (combined search, dct_labels labels).

Port of ctrl_reviewer_core_plate.py (unchanged method),
run once per condition; ``groups`` and ``pseudobulk_correlations`` are the ctrl_reviewer_core functions called with
the condition's label columns (``dct_reviewer_lib.bound``).

Inputs (read-only): results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_task3_scored.h5ad (step 1); dct_labels labels.
Outputs:
  results/tables/dct_reviewer/dct_reviewer_plate_*.csv (both conditions; ``condition`` column)
  results/figures/dct_reviewer/dct_reviewer_plate_labels_and_detected_proteins.pdf/.png/.tif
  results/figures/dct_reviewer/dct_reviewer_plate_pseudobulk_scatter.pdf/.png/.tif

Per plate (P1/P2 from the cell ID; sort plates of one pooled sample per condition, a technical label
only): label fractions and chi-square, detected proteins per cell, Task 3 scores (BH within condition x
label column), and pseudobulk Spearman correlation P1 vs P2 within each label (pseudobulk =
ln(1 + mean intensity), non-detected = 0; proteins with zero pseudobulk in either plate are left out).

Command (project root):  python scripts/dct_reviewer_plate.py
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
import ctrl_reviewer_core_plate as m19
import dct_reviewer_lib as lib

lib.style()
from matplotlib.patches import Patch  # noqa: E402

SCRIPT = "dct_reviewer_plate.py"
SCORES = m19.SCORES
PLATES = ("P1", "P2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 5: plate stratification per DCT condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str) -> dict:
    s = lib.SETTINGS[cond]
    adata = lib.read_h5ad(run, f"DCT_{s.KEY}_task3_scored")
    _, labels = lib.load(run, cond)
    if list(labels.index) != list(adata.obs_names.astype(str)):
        raise ValueError(f"{cond}: label and object cell order differ.")
    scores = adata.obs[[c for c in SCORES if c in adata.obs]].set_axis(labels.index, axis=0)

    rows, tests = [], []
    for col in s.LABEL_COLUMNS:
        table = pd.crosstab(labels["plate"], labels[col])
        for plate, counts in table.iterrows():
            for label in cc.ordered_labels(table.columns):
                rows.append({"label_column": col, "plate": plate, "label": label, "n_cells": int(counts[label]),
                             "fraction_of_plate": counts[label] / counts.sum()})
        chi2, p, dof, _ = stats.chi2_contingency(table.to_numpy())
        tests.append({"label_column": col, "test": "chi-square plate x label", "statistic": chi2, "dof": dof,
                      "p_value": p, "note": (f"includes Unknown (n={int(table['Unknown'].sum())})"
                                             if "Unknown" in table.columns else "")})

    with lib.bound(s, m19):
        group_list = m19.groups(labels)
    det = []
    nd = labels["n_detected_proteins"]
    for col, label, mask in group_list:
        p1 = nd[mask & (labels["plate"] == "P1").to_numpy()]
        p2 = nd[mask & (labels["plate"] == "P2").to_numpy()]
        row = {"label_column": col, "label": label, "n_P1": len(p1), "n_P2": len(p2),
               "median_P1": p1.median(), "median_P2": p2.median(), "q25_P1": p1.quantile(0.25), "q75_P1": p1.quantile(0.75),
               "q25_P2": p2.quantile(0.25), "q75_P2": p2.quantile(0.75)}
        result = cc.compare_groups(p2, p1)
        row.update(mannwhitney_p_P2_vs_P1=result["p_value"], rank_biserial_P2_vs_P1=result["rank_biserial"])
        det.append(row)

    marker_rows = []
    frame = labels.join(scores)
    for col in s.LABEL_COLUMNS:
        for label in cc.ordered_labels(labels[col]):
            sub = frame[frame[col] == label]
            for score_col in scores.columns:
                p1, p2 = sub.loc[sub["plate"] == "P1", score_col], sub.loc[sub["plate"] == "P2", score_col]
                row = {"label_column": col, "label": label, "score": score_col, "n_P1": len(p1), "n_P2": len(p2),
                       "median_P1": p1.median(), "median_P2": p2.median(), "mean_P1": p1.mean(), "mean_P2": p2.mean()}
                result = cc.compare_groups(p2, p1)
                row.update(median_difference_P2_minus_P1=result["median_difference"], p_value=result["p_value"])
                marker_rows.append(row)

    intensity = np.expm1(t2.ln1p_frame(adata).to_numpy())
    with lib.bound(s, m19):
        corr, points = m19.pseudobulk_correlations(intensity, labels)
    return {"fractions": pd.DataFrame(rows), "tests": pd.DataFrame(tests), "detected": pd.DataFrame(det),
            "markers": cc.add_bh(pd.DataFrame(marker_rows), ["label_column"]), "corr": corr, "_points": points,
            "_labels": labels}


def plot_labels_and_ids(results: dict, figs: lib.Figures) -> None:
    panels = [(role, c) for role in lib.ROLES for c in lib.CONDITIONS]
    fig = lib.new_figure(lib.FULL_MM, 125)
    gs = fig.add_gridspec(2, 4, left=0.075, right=0.99, top=0.82, bottom=0.12, hspace=0.6, wspace=0.12)
    all_labels = lib.union_order(*(results[c]["_labels"][lib.SETTINGS[c].COLUMN_OF[r]] for r, c in panels))
    first_top = first_bottom = None
    for k, (role, cond) in enumerate(panels):
        labels = results[cond]["_labels"]
        col = lib.SETTINGS[cond].COLUMN_OF[role]
        # label fractions per plate
        ax = fig.add_subplot(gs[0, k], sharey=first_top)
        first_top = first_top or ax
        fr = results[cond]["fractions"]
        sub = fr[fr["label_column"] == col].pivot(index="plate", columns="label", values="fraction_of_plate")
        bottom = np.zeros(len(sub))
        for label in cc.ordered_labels(sub.columns):
            vals = 100 * sub[label].to_numpy(float)
            ax.bar(range(len(sub)), vals, bottom=bottom, color=t2.LABEL_PALETTE.get(label, "0.5"), width=0.7,
                   edgecolor="white", linewidth=0.8)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v >= 9:
                    ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7, color="white")
            bottom += vals
        n = labels["plate"].value_counts()
        ax.set_xticks(range(len(sub)), [f"{p} ({n[p]})" for p in sub.index])
        ax.set_title(f"{cond}: {col}", fontsize=7)
        if k == 0:
            ax.set_ylabel("% of plate")
        else:
            ax.tick_params(labelleft=False)
        # detected proteins per plate within label
        axb = fig.add_subplot(gs[1, k], sharey=first_bottom)
        first_bottom = first_bottom or axb
        order = cc.ordered_labels(labels[col])
        for i, label in enumerate(order):
            for j, plate in enumerate(PLATES):
                vals = labels.loc[(labels[col] == label) & (labels["plate"] == plate), "n_detected_proteins"]
                if vals.empty:
                    continue
                bp = axb.boxplot(vals.to_numpy(float), positions=[i + (j - 0.5) * 0.38], widths=0.32,
                                 showfliers=False, patch_artist=True)
                bp["boxes"][0].set_facecolor(cc.PLATE_PALETTE[plate])
                bp["boxes"][0].set_edgecolor(cc.PLATE_PALETTE[plate])
                for part in ("whiskers", "caps"):
                    for line in bp[part]:
                        line.set_color(cc.PLATE_PALETTE[plate])
                bp["medians"][0].set_color("white")
        axb.set_xticks(range(len(order)), order, rotation=90)
        axb.set_xlim(-0.6, len(order) - 0.4)
        axb.set_title(f"{cond}: protein IDs", fontsize=7)
        axb.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        axb.set_axisbelow(True)
        if k == 0:
            axb.set_ylabel("protein IDs per cell")
        else:
            axb.tick_params(labelleft=False)
    handles = [Patch(color=t2.LABEL_PALETTE.get(l, "0.5"), label=l) for l in all_labels]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.935), ncol=len(handles), frameon=False,
               title="label (top row)", title_fontsize=7)
    plates = [Patch(color=cc.PLATE_PALETTE[p], label=p) for p in PLATES]
    fig.legend(handles=plates, loc="upper right", bbox_to_anchor=(0.99, 0.935), ncol=len(plates), frameon=False,
               title="plate (bottom row)", title_fontsize=7)
    fig.text(0.005, 0.995, "Label fractions and protein IDs by plate (columns: primary Ctrl, primary ZeroK, second Ctrl, "
             "second ZeroK)\nPlates are sort plates of one pooled sample per condition; plate (n cells)",
             ha="left", va="top", fontsize=7)
    figs.save(fig, "plate_labels_and_detected_proteins", "Label fractions and protein IDs by plate, Ctrl and ZeroK")


def plot_pseudobulk(results: dict, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 82)
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.99, top=0.83, bottom=0.17, wspace=0.55,
                          width_ratios=[1, 1, 1.25])
    lim = max(float(max(results[c]["_points"][("all_cells", "all")][0].max(),
                        results[c]["_points"][("all_cells", "all")][1].max())) for c in lib.CONDITIONS) * 1.03
    for k, cond in enumerate(lib.CONDITIONS):
        ax = fig.add_subplot(gs[0, k])
        a, b, rho = results[cond]["_points"][("all_cells", "all")]
        ax.scatter(a, b, s=1, alpha=0.35, color=t2.INK["secondary"], linewidths=0, rasterized=True)
        ax.plot([0, lim], [0, lim], color=t2.INK["muted"], linewidth=0.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{cond}: all cells\nrho = {rho:.3f}, n = {len(a)} proteins", fontsize=7)
        ax.set_xlabel("P1 ln(1 + mean intensity)")
        ax.set_ylabel("P2 ln(1 + mean intensity)")
    ax = fig.add_subplot(gs[0, 2])
    rows = []
    for cond in lib.CONDITIONS:
        s = lib.SETTINGS[cond]
        corr = results[cond]["corr"]
        for _, r in corr.iterrows():
            role = "all cells" if r["label_column"] == "all_cells" else s.role(r["label_column"])
            rows.append({"condition": cond, "group": "all cells" if role == "all cells" else f"{role}: {r['label']}",
                         "rho": r.get("spearman_rho", np.nan)})
    dots = pd.DataFrame(rows)
    order = list(dict.fromkeys(dots["group"]))
    yy = {g: i for i, g in enumerate(order)}
    for j, cond in enumerate(lib.CONDITIONS):
        d = dots[dots["condition"] == cond]
        ax.scatter(d["rho"], [yy[g] + (j - 0.5) * 0.3 for g in d["group"]], color=lib.COND_COLORS[cond], s=12,
                   label=cond, zorder=3)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("Spearman rho, P1 vs P2 pseudobulk")
    ax.set_title("rho per label group", fontsize=7)
    ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.06), ncol=2, frameon=False, borderaxespad=0)
    fig.text(0.005, 0.995, "Pseudobulk P1 versus P2 (proteins with non-zero pseudobulk on both plates; groups with cells "
             "on both plates)", ha="left", va="top", fontsize=7)
    figs.save(fig, "plate_pseudobulk_scatter", "Pseudobulk P1 versus P2 per condition and rho per label group")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "plate", args)
    lib.ported_from(run, "ctrl_reviewer_core_plate.py")
    run.note("Plates P1/P2 are sort plates of one pooled sample of 3 mice per condition (authors, 28 Oct 2026); plate is "
             "a technical label only.")
    results = {c: one_condition(run, c) for c in lib.CONDITIONS}
    for key, name in (("fractions", "plate_label_fractions"), ("tests", "plate_label_tests"),
                      ("detected", "plate_detected_proteins"), ("markers", "plate_marker_scores"),
                      ("corr", "plate_pseudobulk_spearman")):
        run.table(lib.stack({c: results[c][key] for c in lib.CONDITIONS}), name)
    run.note("plate_marker_scores: BH within condition x label column. Detected proteins: rows with "
             "label_column 'all_cells' are all cells of the condition.")
    figs = lib.Figures(run, "plate")
    plot_labels_and_ids(results, figs)
    plot_pseudobulk(results, figs)
    figs.finish()
    run.finish({"figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_plate.csv.")


if __name__ == "__main__":
    main()
