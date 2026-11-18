#!/usr/bin/env python
"""The response to the reviewers: supplementary figure S-09-a - Slc12a1 and Aqp2 in the DCT data (control and zero-K side
by side): detection by label and by protein depth, and the annotation after excluding the cells in which they are
detected, against random removals.

Drawn from the tables of
report_negative_markers_exclusion.py; no statistic is computed here. Rows = condition (within condition
only; nothing compares the conditions). Panels:
  A  cells with Slc12a1 / Aqp2 detected, by primary label (cs_ctrl_deg_label, cs_zerok_pipeline_label), in %;
     the number above each bar is the number of cells; an absent label keeps an empty slot;
  B  the same by quartile of protein IDs per cell (within condition);
  C  fraction of the remaining cells that keep their primary label after the cells with Slc12a1, Aqp2 or either
     detected are removed (frozen dct_labels rule on the stored Leiden 1.0 clusters): observed (dot) against 100 random
     removals of the same number of cells (median and central 95%), drawn within the primary label (light bar) and
     within primary label x protein-ID quartile (dark bar; report_negative_markers_depth_null.py);
  D  the same with Leiden 1.0 rerun on the rebuilt graph (seed 0).
dotplot / dct_labels format: Arial >= 7 pt, exact 180 mm width, PDF + 600-dpi PNG + TIFF; one assembled file with panel
letters (as the S-07 figures, authors 11 Nov 2026).

Inputs (read-only): results/tables/report/report_09_negative_marker_{by_label,depth,exclusion_vs_random,
  depth_null_vs_random}.csv
Outputs: results/figures/report/report_09_negative_markers.{pdf,png,tif};
  results/tables/report/report_09_negative_marker_figure_{values,checks}.csv;
  results/tables/report/report_step_09_negative_marker_figure_manifest.json

Command (project root):  python scripts/report_negative_markers_figure.py
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
import cluster_de_lib as t5
import dct_reviewer_lib as lib

lib.style()
from matplotlib.lines import Line2D  # noqa: E402

TASK = "report"
STEP = "09_negative_marker_figure"
P = "09_negative_marker"
W, H = 180.0, 118.0                       # mm
COL_X = [21.0, 58.0, 110.0, 146.0]        # left edge of each panel column, mm
AX_W, AX_H = 32.0, 33.0
ROW_Y = {"Ctrl": 76.0, "ZeroK": 26.0}
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
# Display name of the dct_labels label value 'Unknown' (authors, 12 Nov 2026, the response to the reviewers: "Use not resolved"): these cells are
# "DCT, subtype not resolved" in the letter and the legend; the tables keep the dct_labels value.
LABEL_DISPLAY = {"Unknown": "not\nresolved"}     # two lines: one line overlaps the DCT2 tick label
COLOR = {"Slc12a1": t3.SERIES[0], "Aqp2": t3.SERIES[1]}
SETS = [("slc12a1_detected", "Slc12a1"), ("aqp2_detected", "Aqp2"), ("slc12a1_or_aqp2_detected", "either")]
OUTCOME = {"C": "deg_stored_agree", "D": "deg_rerun_agree_s0"}
TITLES = {"A": "by cluster label", "B": "by protein-ID quartile", "C": "stored clusters", "D": "Leiden 1.0 rerun"}
INK, GREY, DARK = "#0b0b0b", "#b9b7b0", "#52514e"   # ink; label-matched null; label- and depth-matched null


def table(run: cc.Run, name: str) -> pd.DataFrame:
    path = cc.out_path(TASK, "tables", f"{P}_{name}", ".csv")
    return pd.read_csv(run.use_input(path, f"The response to the reviewers table: {name}"))


def new_axes(fig, col: int, cond: str):
    return fig.add_axes([COL_X[col] / W, ROW_Y[cond] / H, AX_W / W, AX_H / H])


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    lib.grid_y(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: figure S-09-a.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    lab = table(run, "by_label")
    depth = table(run, "depth")
    vsr = table(run, "exclusion_vs_random")
    dnull = table(run, "depth_null_vs_random")
    conds = [c for c in lib.CONDITIONS if c in set(lab["condition"])]
    checks.add("tables hold both conditions", "; ".join(conds), conds == list(lib.CONDITIONS))

    prim = lab[(lab["label_role"] == "primary") & lab["marker_set"].isin(COLOR)]
    order = cc.ordered_labels(set(prim["label"]))
    values = []
    fig = lib.new_figure(W, H)
    for cond in conds:
        top = cond == conds[0]
        # --- A: by primary label ---------------------------------------------------------------------------------
        ax_a = new_axes(fig, 0, cond)
        sub = prim[prim["condition"] == cond].set_index(["label", "marker_set"])
        n_of = {lb: int(sub.loc[(lb, "Slc12a1"), "n_cells"]) if (lb, "Slc12a1") in sub.index else 0 for lb in order}
        ymax = 0.0
        for j, gene in enumerate(COLOR):
            for i, lb in enumerate(order):
                if (lb, gene) not in sub.index:
                    continue
                r = sub.loc[(lb, gene)]
                pct = 100 * float(r["fraction_detected"])
                x = i + (-0.2 if j == 0 else 0.2)
                ax_a.bar(x, pct, width=0.38, color=COLOR[gene], linewidth=0)
                ax_a.text(x, pct + 0.6, str(int(r["n_detected"])), ha="center", va="bottom", fontsize=7, color=INK)
                ymax = max(ymax, pct)
                values.append({"panel": "A", "condition": cond, "x": lb, "series": gene, "n_cells": int(r["n_cells"]),
                               "n": int(r["n_detected"]), "value": pct, "unit": "% of cells"})
        ax_a.set_xticks(range(len(order)), [f"{LABEL_DISPLAY.get(lb, lb)}\n({n_of[lb]})" for lb in order])
        ax_a.set_xlim(-0.6, len(order) - 0.4)
        ax_a.set_ylabel("cells with protein detected (%)")
        # --- B: by protein-ID quartile ---------------------------------------------------------------------------
        ax_b = new_axes(fig, 1, cond)
        dsub = depth[(depth["condition"] == cond) & depth["marker_set"].isin(COLOR)]
        quartiles = ["Q1", "Q2", "Q3", "Q4"]
        for gene in COLOR:
            g = dsub[dsub["marker_set"] == gene].set_index("protein_id_quartile").reindex(quartiles)
            pct = 100 * g["fraction_detected"].to_numpy(float)
            ax_b.plot(range(4), pct, color=COLOR[gene], marker="o", markersize=3, linewidth=1.0)
            ymax = max(ymax, float(np.nanmax(pct)))
            for q, v, k, n in zip(quartiles, pct, g["n_detected"], g["n_cells"]):
                values.append({"panel": "B", "condition": cond, "x": q, "series": gene, "n_cells": int(n),
                               "n": int(k), "value": float(v), "unit": "% of cells"})
        ax_b.set_xticks(range(4), quartiles)
        ax_b.set_xlim(-0.4, 3.4)
        ax_b.set_xlabel("protein-ID quartile", labelpad=2)
        for ax in (ax_a, ax_b):
            ax.set_ylim(0, ymax * 1.22 + 1)
            tidy(ax)
        ax_b.tick_params(labelleft=False)
        # --- C, D: exclusion against random removals -------------------------------------------------------------
        vsub = vsr[vsr["condition"] == cond]
        axes_cd, lo_all = [], 1.0
        for col, letter in ((2, "C"), (3, "D")):
            ax = new_axes(fig, col, cond)
            v = vsub[vsub["outcome"] == OUTCOME[letter]].set_index("exclusion_set")
            vd = dnull[(dnull["condition"] == cond) & (dnull["outcome"] == OUTCOME[letter])].set_index("exclusion_set")
            ticks = []
            for i, (key, name) in enumerate(SETS):
                if key not in v.index:
                    ticks.append(f"{name}\n(0)")
                    continue
                r, d = v.loc[key], vd.loc[key]
                for x, null, colour in ((i - 0.26, r, GREY), (i, d, DARK)):
                    ax.vlines(x, null["random_q025"], null["random_q975"], color=colour, linewidth=2.6)
                    ax.plot([x], [null["random_median"]], marker="_", color=INK, markersize=6, markeredgewidth=1.0)
                ax.plot([i + 0.26], [r["observed"]], marker="o", color=INK, markersize=3.5, linestyle="none")
                lo_all = min(lo_all, float(r["random_q025"]), float(d["random_q025"]), float(r["observed"]))
                ticks.append(f"{name}\n({int(r['n_removed'])})")
                values.append({"panel": letter, "condition": cond, "x": key, "series": OUTCOME[letter],
                               "n_cells": int(r["n_remaining"]), "n": int(r["n_removed"]),
                               "value": float(r["observed"]), "random_median": float(r["random_median"]),
                               "random_q025": float(r["random_q025"]), "random_q975": float(r["random_q975"]),
                               "depth_matched_random_median": float(d["random_median"]),
                               "depth_matched_random_q025": float(d["random_q025"]),
                               "depth_matched_random_q975": float(d["random_q975"]),
                               "unit": "fraction of remaining cells keeping the primary label"})
            ax.set_xticks(range(len(SETS)), ticks)
            ax.set_xlim(-0.6, len(SETS) - 0.4)
            tidy(ax)
            axes_cd.append(ax)
        span = max(1.0 - lo_all, 0.02)
        for ax in axes_cd:
            ax.set_ylim(lo_all - 0.12 * span, 1.0 + 0.08 * span)
            ax.yaxis.set_major_formatter(lambda y, _pos: f"{y:.2f}")
        axes_cd[0].set_ylabel("cells keeping their label")
        axes_cd[1].tick_params(labelleft=False)
        fig.text(3.0 / W, (ROW_Y[cond] + AX_H / 2) / H, COND_DISPLAY[cond], rotation=90, ha="center", va="center",
                 fontsize=7.5, fontweight="bold")
        if top:
            for ax, letter in zip((ax_a, ax_b, *axes_cd), "ABCD"):
                ax.set_title(TITLES[letter], loc="left", fontsize=7.5, pad=3)
                ax.text(-5.0 / AX_W, 1.0 + 1.2 / AX_H, letter, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=8.5, fontweight="bold")
    left = [Line2D([], [], marker="s", linestyle="none", color=COLOR[g], markersize=5, label=g) for g in COLOR]
    right = [Line2D([], [], marker="o", linestyle="none", color=INK, markersize=3.5, label="marker-detected cells removed"),
             Line2D([], [], color=GREY, linewidth=2.6, label="100 random removals, same labels (median, 95%)"),
             Line2D([], [], color=DARK, linewidth=2.6, label="100 random removals, same labels and depth")]
    for handles, x, ncol in ((left, COL_X[0], 2), (right, COL_X[2], 1)):
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x / W, 1.5 / H), ncol=ncol, frameon=False,
                   fontsize=7, handlelength=1.4, handletextpad=0.5, columnspacing=1.4, labelspacing=0.3,
                   borderaxespad=0)

    figs = lib.Figures(run, STEP)
    row = figs.save(fig, "09_negative_markers",
                    "S-09-a: Slc12a1 and Aqp2 in the DCT data, control and zero-K: detection by primary label (A) and "
                    "protein-ID quartile (B); primary label agreement after excluding the cells with the protein "
                    "detected against 100 random removals, stored clusters (C) and Leiden 1.0 rerun (D)", lib.FULL_MM)
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in PDF)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps "
               f"{row['n_text_overlaps']}; outside {row['n_text_outside_figure']}", bool(row["pass"]))
    run.table(pd.DataFrame(values), f"{P}_figure_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{P}_figure_checks")
    run.finish({"panels": TITLES, "outcomes": OUTCOME, "sets": dict(SETS), "width_mm": W, "height_mm": H,
                "colours": COLOR, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{P}_figure_checks.csv.")


if __name__ == "__main__":
    main()
