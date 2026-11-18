#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S-09-b - panel scores by cluster label in the DCT data, control and
zero-K, with the 90th-percentile flag lines.

The authors asked on 12 Nov 2026 to redraw the two
supplementary figures that existed only as analysis figures ("Yes, redraw them"). This script redraws the dct_reviewer
analysis figure (dct_reviewer_markers.py, dct_reviewer_task3_score_violins_primary) as one reader-facing figure,
following the re-render checklist of the response to the reviewers, section 5: no in-figure title; panels named DCT core,
DCT1-associated, DCT2-associated, TAL, CNT and collecting duct, Interstitial; the cell-cycle and legacy ProLIF
panels dropped; Ctrl -> Control, ZeroK -> Zero-K; the empty DCT2 slot kept for
zero-K; the label value 'Unknown' shown as "not resolved", as in Figure S-09-a. No new statistics: the violins,
median dots and interquartile bars are drawn from the stored per-cell scores (as dct_reviewer_lib.violins drew the
analysis figure), the lines are the stored thresholds and the numbers are the stored flag counts.

Rows = condition (within condition only; nothing compares the conditions). Panels A-F = the six panel scores
(score_genes on the scaled matrix, dct_reviewer step 1), by cluster label (cs_ctrl_deg_label, cs_zerok_pipeline_label):
  A  DCT core (Slc12a3, Pvalb)          D  TAL (Slc12a1, Umod)
  B  DCT1-associated (Pvalb, Egf)       E  CNT and collecting duct (Aqp2, Klk1)
  C  DCT2-associated (Slc8a1, Calb1,    F  Interstitial (Col1a1, Col1a2)
     Klk1)
D-F: dashed line = 90th percentile of the score over the cells of the condition (the stored flag threshold);
number above each violin = the cells of that label above the line. The proteins of each panel are named in the
legend, not in the figure. The y axis of a panel is the same in both rows.
dotplot / dct_labels format: Arial >= 7 pt, exact 180 mm width, PDF + 600-dpi PNG (+ TIFF); one assembled file with panel
letters; no title, footer or internal names in the figure. Each line of the x tick labels is its own text (the
label on top, "resolved" below it, the cell numbers on the third line), so that 22.8 mm axes hold horizontal labels.

Inputs (read-only): results/tables/dct_reviewer/dct_reviewer_task3_{marker_scores_per_cell,
  ambiguity_thresholds,marker_coverage,marker_summary_by_label}.csv
Outputs: results/figures/report/report_09_S09b_panel_scores.{pdf,png,tif};
  results/tables/report/report_09_s09b_figure_{values,checks}.csv;
  results/tables/report/report_step_09_s09b_figure_manifest.json

Command (project root):  python scripts/report_dct_marker_scores_figure.py --overwrite
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

TASK = "report"
STEP = "09_s09b_figure"
W, H = 180.0, 107.0                       # mm
LEFT, AX_W, GAP = 14.5, 22.8, 5.2         # first axis; axis width; gap between axes (mm)
AX_H = 32.0
ROW_Y = {"Ctrl": 65.0, "ZeroK": 19.0}
BASELINE, LINE = 1.6, 3.2                 # header: last baseline above the axis top; line pitch (mm)
TICK_TOP, TICK_LINE = 1.4, 3.0            # x tick label lines below the axis (mm)
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
ORDER = ("DCT1", "DCT2", "Unknown")       # dct_labels label values; the empty DCT2 slot is kept for zero-K
# Display name of the dct_labels label value 'Unknown' (authors, 12 Nov 2026, the response to the reviewers: "Use not resolved"), on two lines
# as in Figure S-09-a; the tables keep the dct_labels value.
LABEL_LINES = {"DCT1": ("DCT1", ""), "DCT2": ("DCT2", ""), "Unknown": ("not", "resolved")}
INK, DARK = "#0b0b0b", "#52514e"
# letter, panel (table value), score column, title lines, proteins (legend), flag column, flag-count column
PANELS = (
    ("A", "DCT_core", "DCT_core_score", ("DCT core",), ("Slc12a3", "Pvalb"), None, None),
    ("B", "DCT1_associated", "DCT1_associated_score", ("DCT1-", "associated"), ("Pvalb", "Egf"), None, None),
    ("C", "DCT2_associated", "DCT2_associated_score", ("DCT2-", "associated"), ("Slc8a1", "Calb1", "Klk1"), None,
     None),
    ("D", "TAL", "TAL_score", ("TAL",), ("Slc12a1", "Umod"), "tal_like_flag", "n_tal_like_flag"),
    ("E", "CNT_CD", "CNT_CD_score", ("CNT and", "collecting duct"), ("Aqp2", "Klk1"), "cnt_cd_like_flag",
     "n_cnt_cd_like_flag"),
    ("F", "Interstitial", "Interstitial_score", ("Interstitial",), ("Col1a1", "Col1a2"), "interstitial_like_flag",
     "n_interstitial_like_flag"),
)
# Canonical counts (the project notes, dct_labels) and the flag numbers of the the response to the reviewers text (section 2).
EXPECTED = {
    "Ctrl": {"n_cells": 729, "labels": {"DCT1": 382, "DCT2": 85, "Unknown": 262}, "n_flagged": 73,
             "flags": {"TAL": {"DCT1": 24, "DCT2": 10, "Unknown": 39}, "CNT_CD": {"DCT1": 21, "DCT2": 12, "Unknown": 40},
                       "Interstitial": {"DCT1": 42, "DCT2": 1, "Unknown": 30}}},
    "ZeroK": {"n_cells": 702, "labels": {"DCT1": 449, "Unknown": 253}, "n_flagged": 71,
              "flags": {"TAL": {"DCT1": 30, "Unknown": 41}, "CNT_CD": {"DCT1": 55, "Unknown": 16},
                        "Interstitial": {"DCT1": 32, "Unknown": 39}}},
}
# Internal names that must not appear in the figure (reader-facing).
BANNED = ("ctrl", "zerok", "unknown", "t28", "task", "generated", "prolif", "cell_cycle", "_", ".csv")


def name_of(lines: tuple[str, ...]) -> str:
    """Panel name on one line: the title lines joined, with no space after a hyphen at a line end."""
    return "".join(s if s.endswith("-") else f"{s} " for s in lines).strip()


def t28(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = cc.out_path(lib.TASK, "tables", name, ".csv")
    return pd.read_csv(run.use_input(path, f"dct_reviewer table: {name}"), **kwargs)


def new_axes(fig, col: int, cond: str):
    ax = fig.add_axes([(LEFT + col * (AX_W + GAP)) / W, ROW_Y[cond] / H, AX_W / W, AX_H / H])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    lib.grid_y(ax)
    return ax


def header(ax, letter: str, lines: tuple[str, ...], n_lines: int = 2) -> None:
    """Panel letter (bold) and title above an axis; the first title line and the letter share the top baseline."""
    for k, text in enumerate(lines):
        y = 1.0 + (BASELINE + (n_lines - 1 - k) * LINE) / AX_H
        ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="baseline", fontsize=7.5)
        if k == 0:
            ax.text(-4.6 / AX_W, y, letter, transform=ax.transAxes, ha="left", va="baseline", fontsize=8.5,
                    fontweight="bold")


def label_ticks(ax, counts: dict[str, int]) -> None:
    """Horizontal x tick labels, one text per line: label (two lines for 'not resolved'), then the cell number."""
    ax.set_xticks(range(len(ORDER)), [""] * len(ORDER))
    colour = plt.rcParams["xtick.color"]
    trans = ax.get_xaxis_transform()
    for i, lab in enumerate(ORDER):
        for k, text in enumerate((*LABEL_LINES[lab], f"({counts.get(lab, 0)})")):
            if text:
                ax.text(i, -(TICK_TOP + k * TICK_LINE) / AX_H, text, transform=trans, ha="center", va="top",
                        fontsize=7, color=colour)


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: figure S-09-b.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    pc = t28(run, "task3_marker_scores_per_cell", keep_default_na=False, na_values=[""])
    thr = t28(run, "task3_ambiguity_thresholds").set_index(["condition", "panel"])
    cov = t28(run, "task3_marker_coverage", keep_default_na=False).set_index(["condition", "panel"])
    summ = t28(run, "task3_marker_summary_by_label", keep_default_na=False, na_values=[""])
    summ = summ[summ["label_role"] == "primary"].set_index(["condition", "label"])
    conds = [c for c in lib.CONDITIONS if c in set(pc["condition"])]
    checks.add("tables hold both conditions", "; ".join(conds), conds == list(lib.CONDITIONS))

    # --- cells and labels ---------------------------------------------------------------------------------------
    for cond in conds:
        sub = pc[pc["condition"] == cond]
        got = sub["primary_label"].astype(str).value_counts().to_dict()
        stored = {lab: int(n) for (c, lab), n in summ["n_cells"].items() if c == cond}
        checks.add(f"{cond}: cells and cluster labels = canonical counts and the stored summary",
                   f"{len(sub)} cells; " + "; ".join(f"{k} {got[k]}" for k in ORDER if k in got),
                   len(sub) == EXPECTED[cond]["n_cells"] and got == EXPECTED[cond]["labels"] == stored,
                   f"{EXPECTED[cond]['n_cells']}; {EXPECTED[cond]['labels']}")
        checks.add(f"{cond}: label column of the stored summary is the primary cluster annotation",
                   "; ".join(sorted(set(summ.loc[cond, "label_column"]))),
                   set(summ.loc[cond, "label_column"]) == {lib.SETTINGS[cond].PRIMARY})
    # --- proteins of each panel (named in the legend) -----------------------------------------------------------
    for _letter, panel, score, title, proteins, _flag, _n in PANELS:
        got = {c: tuple(str(cov.loc[(c, panel), "present"]).split(";")) for c in conds}
        checks.add(f"panel {name_of(title)}: proteins scored = the legend, in both conditions",
                   "; ".join(f"{c}: {', '.join(got[c])}" for c in conds),
                   all(got[c] == proteins and cov.loc[(c, panel), "score_column"] == score for c in conds),
                   ", ".join(proteins))

    # --- y range per panel: the same in both rows; D-F keep room above the violins for the flag numbers, and no
    #     tick or grid line in that room -----------------------------------------------------------------------
    ylim, yticks = {}, {}
    for _letter, panel, score, _title, _proteins, flag, _n in PANELS:
        lo, hi = float(pc[score].min()), float(pc[score].max())
        span = hi - lo
        ylim[panel] = (lo - 0.05 * span, hi + (0.20 if flag else 0.05) * span)
        last = hi if flag else ylim[panel][1]
        yticks[panel] = [t for t in range(-10, 11, 2) if ylim[panel][0] <= t <= last]

    values = []
    fig = lib.new_figure(W, H)
    med_diff, flag_ok, thr_ok, n_flag_ok = 0.0, True, True, True
    for cond in conds:
        top = cond == conds[0]
        sub = pc[pc["condition"] == cond]
        lab = sub["primary_label"].astype(str)
        counts = lab.value_counts().to_dict()
        for col, (letter, panel, score, title, proteins, flag, n_col) in enumerate(PANELS):
            ax = new_axes(fig, col, cond)
            groups = {l_: sub.loc[lab == l_, score].to_numpy(float) for l_ in ORDER}
            lib.violins(ax, groups, list(ORDER))
            for body in ax.collections:        # outline in the label colour: a white outline hides the thin tails
                if isinstance(body, PolyCollection):
                    body.set_edgecolor(body.get_facecolor())
                    body.set_linewidth(0.5)
            ax.set_xlim(-0.5, len(ORDER) - 0.5)
            ax.set_ylim(*ylim[panel])
            ax.set_yticks(yticks[panel])
            label_ticks(ax, counts)
            line = np.nan
            if flag:
                t = thr.loc[(cond, panel)]
                line = float(t["threshold"])
                ax.axhline(line, color=DARK, linestyle=(0, (3, 2)), linewidth=0.7, zorder=2.5)
                above = sub[score].to_numpy(float) > line
                stored_flag = sub[flag].astype(str).str.lower().eq("true").to_numpy()
                flag_ok &= bool(np.array_equal(above, stored_flag))
                thr_ok &= (t["score_column"] == score and t["flag_column"] == flag and float(t["quantile"]) == 0.9)
                n_flag_ok &= int(above.sum()) == int(t["n_flagged"]) == EXPECTED[cond]["n_flagged"]
            for i, l_ in enumerate(ORDER):
                v = groups[l_]
                if not v.size:
                    continue
                s = summ.loc[(cond, l_)]
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                med_diff = max(med_diff, abs(float(med) - float(s[f"{score}_median"])))
                n_above = np.nan
                if flag:
                    n_above = int(s[n_col])
                    flag_ok &= n_above == int((v > line).sum()) == EXPECTED[cond]["flags"][panel][l_]
                    ax.text(i, ylim[panel][1] - 0.02 * (ylim[panel][1] - ylim[panel][0]), str(n_above), ha="center",
                            va="top", fontsize=7, color=INK)
                values.append({"panel": letter, "panel_name": name_of(title), "proteins": "; ".join(proteins),
                               "condition": cond, "condition_display": COND_DISPLAY[cond], "label": l_,
                               "label_display": " ".join(x for x in LABEL_LINES[l_] if x), "n_cells": int(v.size),
                               "median": float(s[f"{score}_median"]), "q25": float(q1), "q75": float(q3),
                               "score_min": float(v.min()), "score_max": float(v.max()), "flag_line": line,
                               "n_cells_above_line": n_above, "score_column": score,
                               "median_source": "dct_reviewer_task3_marker_summary_by_label.csv",
                               "violin_and_quartile_source": "dct_reviewer_task3_marker_scores_per_cell.csv "
                                                             "(drawn from the stored per-cell scores)",
                               "line_source": "dct_reviewer_task3_ambiguity_thresholds.csv" if flag else ""})
            if col == 0:
                ax.set_ylabel("score")
            if top:
                header(ax, letter, title)
        fig.text(3.0 / W, (ROW_Y[cond] + AX_H / 2) / H, COND_DISPLAY[cond], rotation=90, ha="center", va="center",
                 fontsize=7.5, fontweight="bold")

    checks.add("median dots = the stored medians per label and panel (largest absolute difference)",
               f"{med_diff:.2e}", med_diff < 1e-9, "< 1e-9")
    checks.add("dashed lines = the stored thresholds (score and flag columns, quantile 0.9)",
               "; ".join(f"{c} {p}: {float(thr.loc[(c, p), 'threshold']):.4f}" for c in conds
                         for _l, p, _s, _t, _g, f, _n in PANELS if f), bool(thr_ok))
    checks.add("cells above each line = the stored flags (per cell and per label), the stored summary and the "
               "the response to the reviewers text", "; ".join(
                   f"{c} {p}: " + " / ".join(str(int(summ.loc[(c, l_), n])) for l_ in ORDER if (c, l_) in summ.index)
                   for c in conds for _l, p, _s, _t, _g, f, n in PANELS if f), bool(flag_ok))
    checks.add("flagged cells per panel: 73 in control, 71 in zero-K (stored thresholds table)",
               "; ".join(f"{c}: " + ", ".join(str(int(thr.loc[(c, p), 'n_flagged'])) for _l, p, _s, _t, _g, f, _n
                                               in PANELS if f) for c in conds), bool(n_flag_ok))
    checks.add("share of cells above a line (by construction the highest tenth within the condition)",
               "; ".join(f"{c}: {EXPECTED[c]['n_flagged']} of {EXPECTED[c]['n_cells']} = "
                         f"{100 * EXPECTED[c]['n_flagged'] / EXPECTED[c]['n_cells']:.1f}%" for c in conds), None)

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=INK,
                      markeredgewidth=0.6, markersize=3.2, label="median"),
               Line2D([], [], marker="|", linestyle="none", color=INK, markersize=7, markeredgewidth=1.0,
                      label="interquartile range"),
               Line2D([], [], color=DARK, linewidth=0.7, linestyle=(0, (3, 2)),
                      label="90th percentile of the score within the condition (numbers: cells above it)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(LEFT / W, 1.5 / H), ncol=3, frameon=False,
               fontsize=7, handlelength=1.6, handletextpad=0.5, columnspacing=2.0, borderaxespad=0)

    texts = [t.get_text() for t in t27f.drawn_texts(fig)]
    bad = [t for t in texts if any(b in t.lower() for b in BANNED)]
    checks.add("reader-facing text: no internal names (Ctrl, ZeroK, Unknown, task codes, column or file names, "
               "'Generated by'); no cell-cycle or legacy ProLIF panel",
               f"{len(texts)} texts; {len(bad)} with an internal name" + (f": {bad}" if bad else ""), not bad)
    figs = lib.Figures(run, STEP)
    row = figs.save(fig, "09_S09b_panel_scores",
                    "S-09-b: panel scores by cluster label in the DCT data, control and zero-K: DCT core (A), "
                    "DCT1-associated (B), DCT2-associated (C), TAL (D), CNT and collecting duct (E), Interstitial (F); "
                    "dashed lines in D-F = 90th percentile within the condition, numbers = cells above the line",
                    lib.FULL_MM)
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in PDF)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps "
               f"{row['n_text_overlaps']}; outside {row['n_text_outside_figure']}", bool(row["pass"]))
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"panels": {letter: {"name": name_of(title), "score_column": score, "proteins": list(proteins)}
                           for letter, _p, score, title, proteins, _f, _n in PANELS},
                "label_columns": {c: lib.SETTINGS[c].PRIMARY for c in conds},
                "label_display": {"Unknown": "not resolved"}, "label_colours": lib.label_colors(ORDER),
                "width_mm": W, "height_mm": H, "y_limits": {p: [round(a, 4), round(b, 4)] for p, (a, b) in ylim.items()},
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
