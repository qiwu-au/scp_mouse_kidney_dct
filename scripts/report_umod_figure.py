#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S17 - uromodulin (Umod) and parvalbumin (Pvalb) in DCT1 and DCT2
cells of the DCT data, with the total-kidney data and the Mouse Kidney Atlas for comparison.

(authors, 12 Nov 2026: "Yes, go ahead" on the the response to the reviewers plan;
Editors' comment 3). One reader-facing figure, 180 mm wide, with panel letters. No new statistics: the violins and
points are drawn from the stored per-cell values, and every number is read from a stored table.

Rows 1 and 2 = control and zero-K cells (within condition only; nothing compares the conditions):
  A  Umod, ln(1 + intensity) in the cells with Umod detected, by cluster label (violin, median, interquartile
     range); below each label: cells with Umod detected / cells of the label.
  B  The same for Pvalb.
  C  Umod, ln(1 + intensity), against the number of proteins identified per cell (cells with Umod detected),
     coloured by cluster label.
Row 3:
  D  Control cells: DCT2 - DCT1 difference of mean ln(1 + intensity) (non-detected = 0) for the DCT list markers,
     Umod and Slc12a1; open circle = unadjusted, filled circle = adjusted for ln(proteins per cell) and sort plate.
  E  Total kidney: cells with Umod or Slc12a1 detected, by cell type (the cell types of Figure 2A).
  F  Mouse Kidney Atlas: cells with Umod or Pvalb transcripts, by atlas cell type (the rows of the RNA comparator
     of the response to the reviewers), with the derived DCT1-like and DCT2-like groups of the atlas DCT cells (not atlas labels)
     after the DCT row.
The cluster label value 'Unknown' of the DCT data is shown as "not resolved" (DCT, subtype not resolved; authors,
12 Nov 2026); the kidney label Unknown keeps its name. dotplot / dct_labels format: Arial >= 7 pt, exact 180 mm width,
PDF + 600-dpi PNG + TIFF; no title, footer or internal names in the figure.

Inputs (read-only): results/tables/report/report_09_negative_marker_per_cell.csv (per-cell values);
  results/tables/report/report_10_marker_dct2_vs_dct1.csv (report_umod_evidence.py; from dct_rna);
  results/tables/dct_reviewer/dct_reviewer_umod_markers_by_label.csv;
  results/tables/kidney_annotation/kidney_annotation_t15_dotplot_values_a_endo.csv;
  results/tables/dct_rna/dct_rna_dotplot_rna_values.csv
Outputs: results/figures/report/report_10_S10a_umod_dct.{pdf,png,tif};
  results/tables/report/report_10_s10a_figure_{values,checks}.csv;
  results/tables/report/report_step_10_s10a_figure_manifest.json

Command (project root), after report_umod_evidence.py:
  python scripts/report_umod_figure.py --overwrite
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
import dct_labels_figures as t27f
import dct_reviewer_lib as lib

lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TASK = "report"
STEP = "10_s10a_figure"
W, H = 180.0, 176.0                                   # mm
AX_H = 32.0                                           # panels A-C
ROW_Y = {"Ctrl": 135.0, "ZeroK": 86.0}
COL = {"A": (17.0, 34.0), "B": (63.0, 34.0), "C": (112.0, 61.0)}     # left, width (mm)
D_BOX = (20.0, 22.0, 46.0, 42.0)                      # left, bottom, width, height (mm)
E_BOX = (93.0, 30.0, 32.0, 34.0)
F_BOX = (142.0, 30.0, 31.0, 34.0)
BASELINE, LINE = 1.6, 3.2                             # header: last baseline above the axis top; line pitch (mm)
TICK_TOP, TICK_LINE = 1.4, 3.0                        # x tick label lines below the axis (mm)
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
ORDER = ("DCT1", "DCT2", "Unknown")                   # dct_labels label values; the empty DCT2 slot is kept for zero-K
LABEL_LINES = {"DCT1": ("DCT1", ""), "DCT2": ("DCT2", ""), "Unknown": ("not", "resolved")}
LEGEND_NAME = {"DCT1": "DCT1", "DCT2": "DCT2", "Unknown": "DCT, subtype not resolved"}
INK, DARK, BAR = "#0b0b0b", "#52514e", "#3d3d3a"
EXPECTED = {"Ctrl": {"DCT1": 382, "DCT2": 85, "Unknown": 262}, "ZeroK": {"DCT1": 449, "Unknown": 253}}
MARKER_GROUPS = ("core DCT", "DCT1", "DCT2", "TAL")
GROUP_SHORT = {"core DCT": "core", "DCT1": "DCT1", "DCT2": "DCT2", "TAL": "TAL"}
KIDNEY_ORDER = ("Endothelial", "JG", "PCT", "PST", "TAL", "DCT1", "PC", "IC-B", "Unknown")
KIDNEY_DISPLAY = {"DCT1": "DCT", "PC": "CNT/PC"}
ATLAS_ORDER = ("DCT", "DCT1 (atlas_dct)", "DCT2 (atlas_dct)", "DCT-CNT", "CNT", "CTAL", "MTAL")
ATLAS_DISPLAY = {"DCT1 (atlas_dct)": "DCT1-like", "DCT2 (atlas_dct)": "DCT2-like"}
# Internal names that must not appear in the figure (reader-facing). 'Unknown' is allowed only as the kidney label.
BANNED = ("ctrl", "zerok", "t08", "t28", "t29", "task", "generated", "primary", "second", "submitted",
          "re-annotation", "revised", "_", ".csv")


def axes_mm(fig, left: float, bottom: float, width: float, height: float):
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    return ax


def header(ax, letter: str, lines: tuple[str, ...], height_mm: float, width_mm: float, n_lines: int = 2) -> None:
    """Panel letter (bold) and title lines above an axis; the first title line and the letter share a baseline."""
    for k, text in enumerate(lines):
        y = 1.0 + (BASELINE + (n_lines - 1 - k) * LINE) / height_mm
        if text:
            ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="baseline", fontsize=7.5)
        if k == 0:
            ax.text(-4.6 / width_mm, y, letter, transform=ax.transAxes, ha="left", va="baseline", fontsize=8.5,
                    fontweight="bold")


def label_ticks(ax, third_line: dict[str, str]) -> None:
    """Horizontal x tick labels, one text per line: label (two lines for 'not resolved'), then the cell numbers."""
    ax.set_xticks(range(len(ORDER)), [""] * len(ORDER))
    colour = plt.rcParams["xtick.color"]
    trans = ax.get_xaxis_transform()
    for i, lab in enumerate(ORDER):
        for k, text in enumerate((*LABEL_LINES[lab], third_line[lab])):
            if text:
                ax.text(i, -(TICK_TOP + k * TICK_LINE) / AX_H, text, transform=trans, ha="center", va="top",
                        fontsize=7, color=colour)


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: figure S-10-a.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")

    def read(rel: str, label: str) -> pd.DataFrame:
        return pd.read_csv(run.use_input(cc.PROJECT_ROOT / rel, label), float_precision="round_trip")

    pc = read("results/tables/report/report_09_negative_marker_per_cell.csv", "per-cell values")
    md = read("results/tables/report/report_10_marker_dct2_vs_dct1.csv", "marker differences")
    bl = read("results/tables/dct_reviewer/dct_reviewer_umod_markers_by_label.csv", "dct_reviewer markers by label")
    kid = read("results/tables/kidney_annotation/kidney_annotation_t15_dotplot_values_a_endo.csv", "kidney_annotation kidney values")
    rna = read("results/tables/dct_rna/dct_rna_dotplot_rna_values.csv", "dct_rna atlas values")
    bl = bl[bl["label_role"] == "primary"].set_index(["condition", "label", "marker"])
    conds = [c for c in lib.CONDITIONS if c in set(pc["condition"])]
    checks.add("per-cell table holds both conditions", "; ".join(conds), conds == list(lib.CONDITIONS))
    for cond in conds:
        got = pc[pc["condition"] == cond]["primary_label"].astype(str).value_counts().to_dict()
        checks.add(f"{cond}: cells per cluster label = canonical counts",
                   "; ".join(f"{k} {got[k]}" for k in ORDER if k in got), got == EXPECTED[cond], str(EXPECTED[cond]))

    colors = lib.label_colors(ORDER)
    values = []
    fig = lib.new_figure(W, H)

    # --- panels A-C ---------------------------------------------------------------------------------------------
    ylim = {}
    for protein in ("Umod", "Pvalb"):
        v = pc.loc[pc[f"det_{protein}"].astype(bool), f"ln1p_{protein}"].to_numpy(float)
        span = float(v.max() - v.min())
        ylim[protein] = (float(v.min()) - 0.05 * span, float(v.max()) + 0.05 * span)
    xmax = float(pc["n_detected_proteins"].max())
    med_diff, det_ok = 0.0, True
    for cond in conds:
        top = cond == conds[0]
        sub = pc[pc["condition"] == cond]
        lab = sub["primary_label"].astype(str)
        for letter, protein, title in (("A", "Umod", ("Umod (uromodulin)", "cells with Umod detected")),
                                       ("B", "Pvalb", ("Pvalb (parvalbumin)", "cells with Pvalb detected"))):
            left, width = COL[letter]
            ax = axes_mm(fig, left, ROW_Y[cond], width, AX_H)
            lib.grid_y(ax)
            det = sub[f"det_{protein}"].astype(bool)
            groups = {l_: sub.loc[(lab == l_) & det, f"ln1p_{protein}"].to_numpy(float) for l_ in ORDER}
            lib.violins(ax, groups, list(ORDER))
            for body in ax.collections:        # outline in the label colour: a white outline hides the thin tails
                if isinstance(body, PolyCollection):
                    body.set_edgecolor(body.get_facecolor())
                    body.set_linewidth(0.5)
            ax.set_xlim(-0.5, len(ORDER) - 0.5)
            ax.set_ylim(*ylim[protein])
            third = {}
            for l_ in ORDER:
                n_cells = int((lab == l_).sum())
                if not n_cells:
                    third[l_] = "none"
                    continue
                v = groups[l_]
                s = bl.loc[(cond, l_, protein)]
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                med_diff = max(med_diff, abs(float(med) - float(s["median_ln1p_detected"])))
                det_ok &= int(s["n_detected"]) == int(v.size) and int(s["n_cells"]) == n_cells
                third[l_] = f"{v.size}/{n_cells}"
                values.append({"panel": letter, "condition": COND_DISPLAY[cond], "group": LEGEND_NAME[l_],
                               "series": protein, "n_cells": n_cells, "n_detected": int(v.size),
                               "value": float(s["median_ln1p_detected"]), "value_name":
                               "median ln(1 + intensity), detected cells", "q25": float(q1), "q75": float(q3),
                               "source": "dct_reviewer_umod_markers_by_label.csv (median, counts); violins from "
                                         "report_09_negative_marker_per_cell.csv"})
            label_ticks(ax, third)
            if letter == "A":
                ax.set_ylabel("ln(1 + intensity)")
            if top:
                header(ax, letter, title, AX_H, width)
        left, width = COL["C"]
        ax = axes_mm(fig, left, ROW_Y[cond], width, AX_H)
        lib.grid_y(ax)
        det = sub["det_Umod"].astype(bool)
        for l_ in ("Unknown", "DCT1", "DCT2"):             # the smallest group is drawn last
            m = (lab == l_) & det
            ax.scatter(sub.loc[m, "n_detected_proteins"], sub.loc[m, "ln1p_Umod"], s=2.2, color=colors[l_],
                       alpha=0.75, linewidths=0, zorder=3 if l_ == "DCT2" else 2)
            if int(m.sum()):
                values.append({"panel": "C", "condition": COND_DISPLAY[cond], "group": LEGEND_NAME[l_],
                               "series": "Umod", "n_cells": int((lab == l_).sum()), "n_detected": int(m.sum()),
                               "value": float(sub.loc[m, "n_detected_proteins"].median()), "value_name":
                               "median proteins per cell of the plotted cells", "q25": np.nan, "q75": np.nan,
                               "source": "report_09_negative_marker_per_cell.csv (one point per cell)"})
        ax.set_xlim(0, 1.03 * xmax)
        ax.set_xticks(range(0, int(xmax) + 1, 1000), [f"{t:,}" for t in range(0, int(xmax) + 1, 1000)])
        ax.set_ylim(*ylim["Umod"])
        ax.set_ylabel("Umod, ln(1 + intensity)")
        if cond == conds[-1]:
            ax.set_xlabel("proteins identified per cell")
        if top:
            header(ax, "C", ("Umod by proteins per cell", ""), AX_H, width)
            handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=colors[l_], markeredgewidth=0,
                              markersize=3.4, label=LEGEND_NAME[l_]) for l_ in ORDER]
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False,
                      fontsize=7, handlelength=0.8, handletextpad=0.3, columnspacing=1.2, borderaxespad=0,
                      borderpad=0.15)
        fig.text(3.0 / W, (ROW_Y[cond] + AX_H / 2) / H, COND_DISPLAY[cond], rotation=90, ha="center", va="center",
                 fontsize=7.5, fontweight="bold")
    checks.add("A, B: median dots = the stored medians per label (largest absolute difference); cells and detected "
               "cells = the stored counts", f"{med_diff:.2e}; counts equal {bool(det_ok)}",
               med_diff < 1e-9 and bool(det_ok), "< 1e-9; True")
    n_points = {c: int(pc[(pc["condition"] == c)]["det_Umod"].astype(bool).sum()) for c in conds}
    checks.add("C: points = cells with Umod detected (control 712, zero-K 690)", n_points,
               n_points == {"Ctrl": 712, "ZeroK": 690})

    # --- panel D: DCT2 - DCT1 per marker, control cells, cluster annotation -------------------------------------
    d = md[(md["condition"] == "Ctrl") & (md["label_role"] == "primary")].reset_index(drop=True)
    checks.add("D: 11 markers, control cells, cluster annotation, all compared (DCT1 382, DCT2 85)",
               f"{len(d)} markers; {sorted(set(d['status']))}; DCT1 {int(d['n_DCT1'].iloc[0])}, DCT2 "
               f"{int(d['n_DCT2'].iloc[0])}",
               len(d) == 11 and set(d["status"]) == {"compared"} and int(d["n_DCT1"].iloc[0]) == 382
               and int(d["n_DCT2"].iloc[0]) == 85 and list(d["marker_group"].drop_duplicates()) == list(MARKER_GROUPS))
    left, bottom, width, height = D_BOX
    ax = axes_mm(fig, left, bottom, width, height)
    y = np.arange(len(d))[::-1].astype(float)
    ax.axvline(0, color=DARK, linewidth=0.6, zorder=1)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    raw = d["difference_mean_ln1p_DCT2_minus_DCT1"].to_numpy(float)
    adj = d["difference_adjusted_DCT2_minus_DCT1"].to_numpy(float)
    ax.hlines(y, np.minimum(raw, adj), np.maximum(raw, adj), color="#a8a69e", linewidth=0.6, zorder=2)
    ax.scatter(raw, y, s=13, facecolor="white", edgecolor=INK, linewidths=0.7, zorder=3)
    ax.scatter(adj, y, s=13, facecolor=INK, edgecolor=INK, linewidths=0.7, zorder=4)
    ax.set_yticks(y, list(d["marker"]))
    ax.set_ylim(-0.7, len(d) - 0.3)
    lo, hi = float(min(raw.min(), adj.min())), float(max(raw.max(), adj.max()))
    ax.set_xlim(np.floor(lo) - 0.3, np.ceil(hi) + 0.3)
    ax.set_xticks(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))
    ax.set_xlabel("DCT2 − DCT1, difference of mean ln(1 + intensity)")
    trans = ax.get_yaxis_transform()
    for group in MARKER_GROUPS:
        ys = y[(d["marker_group"] == group).to_numpy()]
        x_bar = 1.0 + 1.6 / width
        ax.plot([x_bar, x_bar], [ys.min() - 0.3, ys.max() + 0.3], transform=trans, color=DARK, linewidth=0.7,
                clip_on=False)
        ax.text(1.0 + 2.6 / width, float(ys.mean()), GROUP_SHORT[group], transform=trans, ha="left", va="center",
                fontsize=7, color=INK)
    header(ax, "D", ("DCT2 against DCT1, control cells", ""), height, width, n_lines=1)
    for _, r in d.iterrows():
        values.append({"panel": "D", "condition": "Control", "group": r["marker_group"], "series": r["marker"],
                       "n_cells": int(r["n_DCT1"] + r["n_DCT2"]), "n_detected": np.nan,
                       "value": float(r["difference_mean_ln1p_DCT2_minus_DCT1"]),
                       "value_name": "DCT2 - DCT1, difference of mean ln(1 + intensity), non-detected = 0 "
                                     "(value); adjusted for ln(proteins per cell) and sort plate (value_adjusted)",
                       "value_adjusted": float(r["difference_adjusted_DCT2_minus_DCT1"]),
                       "q_unadjusted": float(r["q_mannwhitney_bh"]), "p_adjusted": float(r["p_adjusted"]),
                       "source": "report_10_marker_dct2_vs_dct1.csv (from dct_rna_fc_genes.csv)"})
    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=INK,
                      markeredgewidth=0.7, markersize=3.6, label="unadjusted"),
               Line2D([], [], marker="o", linestyle="none", markerfacecolor=INK, markeredgecolor=INK,
                      markeredgewidth=0.7, markersize=3.6, label="adjusted for proteins per cell and sort plate")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left / W, 5.0 / H), ncol=1, frameon=False,
               fontsize=7, handlelength=1.0, handletextpad=0.4, borderaxespad=0, labelspacing=0.3)

    # --- panel E: total kidney, Umod and Slc12a1 by cell type ---------------------------------------------------
    checks.add("E: kidney values use the kidney cell types (one annotation)", "; ".join(sorted(set(kid["label_column"]))),
               set(kid["label_column"]) == {"kidney_t15_label"})
    left, bottom, width, height = E_BOX
    ax = axes_mm(fig, left, bottom, width, height)
    lib.grid_y(ax)
    k_ok = True
    for j, (protein, face) in enumerate((("Umod", BAR), ("Slc12a1", "white"))):
        k = kid[kid["marker"] == protein].set_index("group").loc[list(KIDNEY_ORDER)]
        k_ok &= bool(np.allclose(k["n_detected"] / k["n_cells"], k["frac_detected"]))
        ax.bar(np.arange(len(k)) + (j - 0.5) * 0.4, 100 * k["frac_detected"].to_numpy(float), width=0.4, color=face,
               edgecolor=BAR, linewidth=0.6, zorder=3)
        for g, r in k.iterrows():
            values.append({"panel": "E", "condition": "Total kidney", "group": KIDNEY_DISPLAY.get(g, g),
                           "series": protein, "n_cells": int(r["n_cells"]), "n_detected": int(r["n_detected"]),
                           "value": 100 * float(r["frac_detected"]), "value_name": "cells with the protein detected (%)",
                           "source": "kidney_annotation_t15_dotplot_values_a_endo.csv"})
    n_kid = kid[kid["marker"] == "Umod"].set_index("group").loc[list(KIDNEY_ORDER), "n_cells"].astype(int)
    checks.add("E: cells per kidney cell type = canonical counts (712 cells), fractions = detected / cells",
               "; ".join(f"{KIDNEY_DISPLAY.get(g, g)} {n}" for g, n in n_kid.items()),
               bool(k_ok) and n_kid.to_dict() == {"Endothelial": 19, "JG": 24, "PCT": 160, "PST": 158, "TAL": 138,
                                                  "DCT1": 58, "PC": 38, "IC-B": 8, "Unknown": 109})
    ax.set_xticks(range(len(KIDNEY_ORDER)), [f"{KIDNEY_DISPLAY.get(g, g)} ({n_kid[g]})" for g in KIDNEY_ORDER],
                  rotation=90)
    ax.set_xlim(-0.6, len(KIDNEY_ORDER) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("cells with the protein detected (%)")
    ax.legend(handles=[Patch(facecolor=BAR, edgecolor=BAR, linewidth=0.6, label="Umod"),
                       Patch(facecolor="white", edgecolor=BAR, linewidth=0.6, label="Slc12a1")],
              loc="upper left", frameon=False, fontsize=7, handlelength=0.9, handleheight=0.7, handletextpad=0.4,
              borderaxespad=0.2, labelspacing=0.25)
    header(ax, "E", ("Total kidney, protein", ""), height, width, n_lines=1)

    # --- panel F: Mouse Kidney Atlas, Umod and Pvalb transcripts by atlas group ---------------------------------
    # The atlas cell types (as in the RNA comparator of the response to the reviewers), with the two derived groups of the DCT split
    # placed after the atlas DCT row; the other rows are the same in both row sets of the dct_rna table.
    r = pd.concat([rna[(rna["level"] == "published") & (rna["rna_source"] == "all") & (rna["group"] == "DCT")],
                   rna[(rna["level"] == "derived") & (rna["rna_source"] == "all")]], ignore_index=True)
    left, bottom, width, height = F_BOX
    ax = axes_mm(fig, left, bottom, width, height)
    lib.grid_y(ax)
    for j, (gene, face) in enumerate((("Umod", BAR), ("Pvalb", "white"))):
        g = r[r["marker"] == gene].set_index("group").loc[list(ATLAS_ORDER)]
        ax.bar(np.arange(len(g)) + (j - 0.5) * 0.4, 100 * g["frac_detected"].to_numpy(float), width=0.4, color=face,
               edgecolor=BAR, linewidth=0.6, zorder=3)
        for name, row in g.iterrows():
            values.append({"panel": "F", "condition": "Mouse Kidney Atlas, cells and nuclei",
                           "group": ATLAS_DISPLAY.get(name, name), "series": gene,
                           "n_cells": int(row["n_atlas_cells"]), "n_detected": np.nan,
                           "value": 100 * float(row["frac_detected"]),
                           "value_name": "atlas cells with at least one count (%)",
                           "source": "dct_rna_dotplot_rna_values.csv (all atlas cells; DCT from the atlas cell "
                                     "types, the other rows from the row set with the derived DCT split)"})
    n_at = r[r["marker"] == "Umod"].set_index("group").loc[list(ATLAS_ORDER), "n_atlas_cells"].astype(int)
    checks.add("F: atlas groups and cells (atlas cell types, and the derived split of the DCT cells; all atlas cells)",
               "; ".join(f"{ATLAS_DISPLAY.get(g, g)} {n}" for g, n in n_at.items()),
               n_at.to_dict() == {"DCT": 16381, "DCT1 (atlas_dct)": 8864, "DCT2 (atlas_dct)": 2752, "DCT-CNT": 810, "CNT": 3903,
                                  "CTAL": 3600, "MTAL": 3840})
    ax.set_xticks(range(len(ATLAS_ORDER)), [f"{ATLAS_DISPLAY.get(g, g)} ({n_at[g]:,})" for g in ATLAS_ORDER],
                  rotation=90)
    ax.set_xlim(-0.6, len(ATLAS_ORDER) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("cells with transcripts (%)")
    ax.legend(handles=[Patch(facecolor=BAR, edgecolor=BAR, linewidth=0.6, label="Umod"),
                       Patch(facecolor="white", edgecolor=BAR, linewidth=0.6, label="Pvalb")],
              loc="upper left", frameon=False, fontsize=7, handlelength=0.9, handleheight=0.7, handletextpad=0.4,
              borderaxespad=0.2, labelspacing=0.25, ncol=2, columnspacing=1.0)
    header(ax, "F", ("Mouse Kidney Atlas, RNA", ""), height, width, n_lines=1)

    # --- reader-facing text, format, outputs --------------------------------------------------------------------
    texts = [t.get_text() for t in t27f.drawn_texts(fig)]
    bad = [t for t in texts if any(b in t.lower() for b in BANNED)]
    unknown = [t for t in texts if "unknown" in t.lower()]
    checks.add("reader-facing text: no internal names (Ctrl, ZeroK, task codes, column or file names, 'Generated "
               "by', words for the revision process); 'Unknown' only as the kidney label of panel E",
               f"{len(texts)} texts; {len(bad)} with an internal name" + (f": {bad}" if bad else "")
               + f"; texts with Unknown: {unknown}", not bad and unknown == ["Unknown (109)"])
    figs = lib.Figures(run, STEP)
    row = figs.save(fig, "10_S10a_umod_dct",
                    "S-10-a: Umod and Pvalb in the DCT data by cluster label, control and zero-K (A, B), Umod by "
                    "proteins per cell (C), DCT2 - DCT1 difference per marker in control, unadjusted and adjusted "
                    "(D), Umod and Slc12a1 by cell type in total kidney (E), Umod and Pvalb transcripts by atlas "
                    "group (F)", lib.FULL_MM)
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in PDF)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps "
               f"{row['n_text_overlaps']}; outside {row['n_text_outside_figure']}", bool(row["pass"]))
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"label_columns": {c: lib.SETTINGS[c].PRIMARY for c in conds},
                "label_display": {"Unknown": "not resolved (DCT, subtype not resolved)"},
                "label_colours": colors, "kidney_display": KIDNEY_DISPLAY, "atlas_display": ATLAS_DISPLAY,
                "width_mm": W, "height_mm": H, "y_limits": {p: [round(a, 4), round(b, 4)] for p, (a, b) in ylim.items()},
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
