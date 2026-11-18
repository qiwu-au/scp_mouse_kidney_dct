#!/usr/bin/env python
"""ctrl_reviewer_core step 7: reviewer_core C Umod and neighbouring-segment markers per labels_core label (Editors' point 3; control data).

Port of
reviewer_core_umod.py (unchanged method; inputs and label columns replaced).

Inputs (read-only): labels_core labelled control object and canonical labels (intensity layer).
Outputs:
  results/tables/ctrl_reviewer_core/ctrl_reviewer_core_umod_markers_by_label.csv
  results/tables/ctrl_reviewer_core/ctrl_reviewer_core_umod_slc12a1_codetection.csv
  results/tables/ctrl_reviewer_core/ctrl_reviewer_core_umod_dct1_vs_dct2_tests.csv
  results/figures/ctrl_reviewer_core/ctrl_reviewer_core_umod_markers_<label_column>.pdf/.png (dotplot + violins)

Per label (both labels_core label columns): detection fraction (intensity > 0, Wilson
95% CI) and median ln(1 + intensity) in detected cells for Umod, Slc12a1,
Pvalb, Egf, Slc12a3, Calb1, Slc8a1, S100g and Klk1; Umod/Slc12a1
co-detection counts; Umod DCT1 vs DCT2: Mann-Whitney U on detected cells only
and on all cells (non-detected = 0), and Fisher's exact test on detection;
BH-FDR within label column.

Command (project root):  python scripts/ctrl_reviewer_core_umod.py
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "ctrl_reviewer_core_umod.py"
MARKERS = [("Umod", "TAL"), ("Slc12a1", "TAL"), ("Pvalb", "DCT1"), ("Egf", "DCT1"), ("Slc12a3", "DCT1 and DCT2"),
           ("Calb1", "DCT2/CNT"), ("Slc8a1", "DCT2/CNT"), ("S100g", "DCT2/CNT"), ("Klk1", "DCT2/CNT")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 7: Umod and neighbouring markers per labels_core label (control data).")
    cc.add_common_args(parser)
    return parser.parse_args()


def plot_markers(table: pd.DataFrame, ln1p: pd.DataFrame, labels: pd.DataFrame, col: str, run: cc.Run) -> None:
    sub = table[table["label_column"] == col]
    order = cc.ordered_labels(labels[col])
    markers = [m for m, _ in MARKERS]
    seg = dict(MARKERS)
    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.45, 1, 1, 1], hspace=0.75, wspace=0.42,
                          left=0.12, right=0.99, top=0.89, bottom=0.2)
    ax = fig.add_subplot(gs[:, 0])
    vmax = np.nanmax(sub["median_ln1p_detected"].to_numpy(float))
    vmin = np.nanmin(sub["median_ln1p_detected"].to_numpy(float))
    cmap = LinearSegmentedColormap.from_list("blue", t2.BLUE_RAMP)
    for i, marker in enumerate(markers):
        rows = sub[sub["marker"] == marker].set_index("label").reindex(order)
        med = rows["median_ln1p_detected"].to_numpy(float)
        sizes = np.maximum(rows["detection_fraction"].to_numpy(float) * 200, 1.0)
        colors = [cmap((m - vmin) / (vmax - vmin)) if np.isfinite(m) else t2.INK["grid"] for m in med]
        ax.scatter(np.arange(len(order)), np.full(len(order), i), s=sizes, c=colors, edgecolors=t2.INK["secondary"],
                   linewidths=0.3)
    ax.set_yticks(range(len(markers)), [f"{m} ({seg[m]})" for m in markers], fontsize=6)
    ax.set_xticks(range(len(order)), [f"{l}\nn={int(sub[sub['label'] == l]['n_cells'].iloc[0])}" for l in order],
                  fontsize=5.8)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(len(markers) - 0.5, -0.5)
    ax.set_title("Detection (dot size) and median ln(1+intensity)\nin detected cells (colour, common scale)", fontsize=6.4)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(f * 200), markerfacecolor=t2.INK["grid"],
                      markeredgecolor=t2.INK["secondary"], label=f"{int(f * 100)}%") for f in (0.1, 0.5, 1.0)]
    ax.legend(handles=handles, title="Detected", loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              title_fontsize=5.8, fontsize=5.6, columnspacing=1.2)
    cax = ax.inset_axes([0.08, -0.2, 0.84, 0.022])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5)
    cb.set_label("median ln(1+intensity), detected cells", fontsize=5.5)
    colors = t2.label_colors(order)
    for k, marker in enumerate(markers):
        vax = fig.add_subplot(gs[k // 3, 1 + k % 3])
        data = [ln1p.loc[(labels[col] == l).to_numpy(), marker] for l in order]
        det = [d[d > 0].to_numpy() for d in data]
        for i, (d, lab) in enumerate(zip(det, order)):
            if len(d) >= 3:
                parts = vax.violinplot([d], positions=[i], widths=0.8, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(colors[lab])
                    body.set_edgecolor("white")
                    body.set_alpha(0.85)
            elif len(d):
                vax.scatter(np.full(len(d), i), d, color=colors[lab], s=6, zorder=2, linewidths=0)
            if len(d):
                vax.scatter([i], [np.median(d)], color="white", edgecolor=t2.INK["primary"], s=7, zorder=3, linewidths=0.5)
        vax.set_xticks(range(len(order)), [f"{l if l != 'Unknown' else 'Unk.'}\n{len(d)}/{len(f)}"
                                           for l, d, f in zip(order, det, data)], fontsize=4.8)
        vax.set_title(marker, fontsize=6.4)
        vax.tick_params(axis="y", labelsize=5)
        vax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.4)
        vax.set_axisbelow(True)
        if k % 3 == 0:
            vax.set_ylabel("ln(1+intensity)", fontsize=5.5)
    fig.suptitle(f"Umod and neighbouring markers by {lib.LABEL_DISPLAY[col]}", x=0.01, ha="left", fontsize=7.5,
                 fontweight="bold")
    fig.text(0.01, 0.935, "Violins: ln(1+intensity) in detected cells (dot = median; fewer than 3 detected cells drawn "
             "as points). Tick labels: detected/total cells. Unk. = Unknown.", fontsize=5.8, color=t2.INK["secondary"])
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"umod_markers_{col}", "Umod and neighbouring markers per labels_core label (dotplot and violins; control data)")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "umod", args)
    lib.ported_from(run, "reviewer_core_umod.py")
    adata, labels = lib.load_ctrl(run)
    ln1p = t2.ln1p_frame(adata)
    missing = [m for m, _ in MARKERS if m not in ln1p.columns]
    if missing:
        run.note(f"Markers absent from the protein matrix: {missing}")

    rows, codet, tests = [], [], []
    for col in lib.LABEL_COLUMNS:
        for label in cc.ordered_labels(labels[col]):
            mask = (labels[col] == label).to_numpy()
            for order, (marker, segment) in enumerate(MARKERS, start=1):
                if marker not in ln1p.columns:
                    continue
                v = ln1p.loc[mask, marker]
                d = v[v > 0]
                lo, hi = t2.wilson_ci(len(d), len(v))
                rows.append({"label_column": col, "label": label, "order": order, "marker": marker,
                             "expected_segment": segment, "n_cells": len(v), "n_detected": len(d),
                             "detection_fraction": len(d) / len(v), "detection_ci_low": lo, "detection_ci_high": hi,
                             "median_ln1p_detected": d.median() if len(d) else np.nan,
                             "q25_ln1p_detected": d.quantile(0.25) if len(d) else np.nan,
                             "q75_ln1p_detected": d.quantile(0.75) if len(d) else np.nan,
                             "median_ln1p_all_cells": v.median(), "mean_ln1p_all_cells": v.mean()})
            u, s = ln1p.loc[mask, "Umod"] > 0, ln1p.loc[mask, "Slc12a1"] > 0
            codet.append({"label_column": col, "label": label, "n_cells": int(mask.sum()), "n_umod_detected": int(u.sum()),
                          "n_slc12a1_detected": int(s.sum()), "n_umod_and_slc12a1": int((u & s).sum()),
                          "n_umod_without_slc12a1": int((u & ~s).sum()), "n_slc12a1_without_umod": int((~u & s).sum()),
                          "n_neither": int((~u & ~s).sum())})
        a = ln1p.loc[(labels[col] == "DCT1").to_numpy(), "Umod"]
        b = ln1p.loc[(labels[col] == "DCT2").to_numpy(), "Umod"]
        for scope, x, y in (("detected cells only", a[a > 0], b[b > 0]), ("all cells, non-detected = 0", a, b)):
            res = cc.compare_groups(x, y)
            tests.append({"label_column": col, "marker": "Umod", "comparison": "DCT1_vs_DCT2", "scope": scope,
                          "test": "Mann-Whitney U (two-sided)", "n_DCT1": res["n_group"], "n_DCT2": res["n_reference"],
                          "median_DCT1": res["median_group"], "median_DCT2": res["median_reference"],
                          "median_difference": res["median_difference"], "rank_biserial": res["rank_biserial"],
                          "statistic": res["mannwhitney_u"], "p_value": res["p_value"]})
        table = [[int((a > 0).sum()), int((a == 0).sum())], [int((b > 0).sum()), int((b == 0).sum())]]
        odds, p = stats.fisher_exact(table)
        tests.append({"label_column": col, "marker": "Umod", "comparison": "DCT1_vs_DCT2", "scope": "detection",
                      "test": "Fisher exact (detected vs not)", "n_DCT1": len(a), "n_DCT2": len(b),
                      "median_DCT1": float((a > 0).mean()), "median_DCT2": float((b > 0).mean()),
                      "median_difference": float((a > 0).mean() - (b > 0).mean()), "statistic": odds, "p_value": p,
                      "note": "median_* columns hold detection fractions; statistic = odds ratio"})
    table = pd.DataFrame(rows)
    run.table(table, "umod_markers_by_label")
    run.table(pd.DataFrame(codet), "umod_slc12a1_codetection")
    run.table(cc.add_bh(pd.DataFrame(tests), ["label_column"]), "umod_dct1_vs_dct2_tests")
    for col in lib.LABEL_COLUMNS:
        plot_markers(table, ln1p, labels, col, run)
    run.finish({"markers": [m for m, _ in MARKERS]})


if __name__ == "__main__":
    main()
