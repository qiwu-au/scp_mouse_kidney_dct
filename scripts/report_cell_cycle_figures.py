#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figures S23, S18 and S24 (ProLIF and the cell cycle, DCT data).

. Three reader-facing figures, each one assembled file with panel letters, drawn from stored
tables only (no new statistics here): the dct_reviewer tables, the dct_prolif tables and the the response to the reviewers tables written by
report_prolif_percell_tabulation.py and report_aldo_random_sets.py. Control and zero-K are shown side
by side, each analysed on its own; nothing compares the conditions.

  S23  Cell-cycle correction: (A) proteins per cell of the two ProLIF definitions; (B) largest share of one
          Leiden cluster made up of ProLIF cells of the per-cell test, by matrix; (C) same-label neighbour fraction
          of the ProLIF cells of the marker-score label; (D) pseudotime medians, per-cell test, with the reference
          of random cells from the same clusters; (E) pseudotime medians, marker-score label of the clusters.
  S18  The per-cell proliferation test: (A) DCT2 - DCT1 marker axis against the proliferation score; (B) cells
          with Mki67 or Top2a detected; (C) ProLIF cells per Leiden cluster of the cluster annotation; (D) proteins
          per cell.
  S24  ALDO scores of the ProLIF cells against DCT1 and DCT2 (Cohen's d), with and without Slc12a3, on the main
          matrix and after regression of the cell-cycle score, against 1,000 random protein sets.

Reader-facing names (list of items, decisions of 12 Nov 2026): "Control" / "Zero-K"; "not resolved" for the label value
Unknown of the cluster annotation; "marker-score label" for cs_<cond>_cluster_label; no task code, file or column
name, and no word for the revision process in a figure (checked).
Format (dotplot / dct_labels): Arial >= 7 pt, exact 180 mm width, PDF + 600-dpi PNG + TIFF, no in-figure title or footer.

Inputs (read-only): results/tables/report/report_12_percell_prolif_{per_cell,groups,cluster_spread,
  pseudotime_by_group,pseudotime_same_cluster_reference}.csv, report_aldo_random_sets.csv;
  results/tables/dct_reviewer/dct_reviewer_{task4_knn_label_purity,task4_dpt_summary}.csv;
  results/tables/dct_prolif/dct_prolif_prolif_overlay_by_t27_cluster.csv
Outputs: results/figures/report/report_12_S12a_cell_cycle_correction, _S12b_percell_test,
  _S12c_aldo_scores (.pdf, .png, .tif); results/tables/report/report_12_s12_figures_{values,checks}.csv;
  results/tables/report/report_step_12_s12_figures_manifest.json

Command (project root):  python scripts/report_cell_cycle_figures.py --overwrite
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TASK = "report"
STEP = "12_s12_figures"
DCT_REVIEWER, DCT_PROLIF = "dct_reviewer", "dct_prolif"
W = 180.0
CONDS = ("Ctrl", "ZeroK")
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
INK, DARK, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
COL = {"DCT1": "#1f77b4", "DCT2": "#ff7f0e", "ProLIF": "#2ca02c", "Unknown": "#B3B3B3", "score": "#8e44ad",
       "other": "#B3B3B3", "reference": "#c9c8c2"}
GROUPS = ("DCT1", "DCT2", "ProLIF", "Unknown")
GROUP_DISPLAY = {"DCT1": "DCT1", "DCT2": "DCT2", "ProLIF": "ProLIF", "Unknown": "not resolved"}
OBJECTS = ("uncorrected", "cc_removed", "cc_regressed", "tirosh_removed", "tirosh_regressed", "corr_removed",
           "corr_regressed")
OBJECT_TICK = ("none", "removed", "regressed", "removed", "regressed", "removed", "regressed")
OBJECT_GROUPS = ((1.5, ("cell-cycle", "panel")), (3.5, ("S and G2/M", "lists")), (5.5, ("correlated", "proteins")))
BANNED = ("ctrl", "zerok", "unknown", "t27", "t28", "t31", "generated", "submitted", "submission",
          "re-annotation", "cs_", ".csv", "scaled_x", "cc_", "tirosh", "route", "nofloor")
# Numbers of the the response to the reviewers letter (section 2) that the figures show.
LETTER = {
    "Ctrl": {"n_score": 119, "n_call": 37, "ids_score": 516, "ids_call": 1913, "max_share_r1_none": 0.18,
             "max_share_r3_none": 1.0, "max_share_r3_removed": (0.4, 0.4, 1 / 6), "pt_call": 0.39, "pt_dct1": 0.41,
             "pt_dct2": 0.30, "pt_score": 0.82, "mki_call": (14, 37)},
    "ZeroK": {"n_score": 98, "n_call": 63, "ids_score": 599, "ids_call": 1919, "max_share_r1_none": 0.20,
              "max_share_r3_none": 1.0, "pt_call": 0.21, "pt_dct1": 0.18, "pt_score": 0.67, "mki_call": (37, 63)},
}


def mm_axes(fig, h_mm: float, x: float, y: float, w: float, h: float):
    ax = fig.add_axes([x / W, y / h_mm, w / W, h / h_mm])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    return ax


def letter_title(fig, h_mm: float, x: float, y: float, letter: str, title: str) -> None:
    fig.text(x / W, y / h_mm, letter, fontsize=8.5, fontweight="bold", ha="left", va="baseline")
    fig.text((x + 4.5) / W, y / h_mm, title, fontsize=7.5, ha="left", va="baseline")


def violin(ax, pos: float, values, colour: str, width: float = 0.75) -> None:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size >= 2 and np.nanstd(v) > 0:
        parts = ax.violinplot([v], positions=[pos], widths=width, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_linewidth(0.5)
            body.set_alpha(0.85)
    if v.size:
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(pos, q1, q3, color=INK, linewidth=1.0)
        ax.scatter([pos], [med], color="white", edgecolor=INK, s=8, zorder=3, linewidths=0.6)


def tick_lines(ax, ax_h: float, positions, lines_per_tick, top: float = 1.4, pitch: float = 3.0) -> None:
    """Horizontal x tick labels, one text per line (so that every line is checked for overlap on its own)."""
    ax.set_xticks(list(positions), [""] * len(positions))
    trans = ax.get_xaxis_transform()
    for x, lines in zip(positions, lines_per_tick):
        for k, text in enumerate(lines):
            if text:
                ax.text(x, -(top + k * pitch) / ax_h, text, transform=trans, ha="center", va="top", fontsize=7)


def object_ticks(ax, ax_h: float, labelled: bool) -> None:
    ax.set_xlim(-0.6, 6.6)
    ax.set_xticks(range(7), OBJECT_TICK if labelled else [""] * 7, rotation=90)
    if labelled:
        trans = ax.get_xaxis_transform()
        for x, lines in ((0, ("no", "correction")), *OBJECT_GROUPS):
            for k, text in enumerate(lines):
                ax.text(x, -(15.5 + k * 3.0) / ax_h, text, transform=trans, ha="center", va="top", fontsize=7)
        for x0, x1 in ((0.7, 2.3), (2.7, 4.3), (4.7, 6.3)):
            ax.plot([x0, x1], [-14.3 / ax_h] * 2, transform=trans, color=DARK, linewidth=0.6, clip_on=False)


def legend_row(fig, h_mm: float, x: float, y: float, handles, ncol: int | None = None) -> None:
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x / W, y / h_mm), ncol=ncol or len(handles),
               frameon=False, fontsize=7, handlelength=1.4, handletextpad=0.4, columnspacing=1.4, borderaxespad=0)


def dot(colour: str, label: str, marker: str = "o", filled: bool = True, size: float = 4.0) -> Line2D:
    return Line2D([], [], marker=marker, linestyle="none", markersize=size, markeredgewidth=0.8,
                  markeredgecolor=colour, markerfacecolor=colour if filled else "white", label=label)


def check_text(checks: t5.Checks, fig, name: str) -> None:
    texts = [t.get_text() for t in t27f.drawn_texts(fig)]
    bad = [t for t in texts if any(b in t.lower() for b in BANNED)]
    checks.add(f"{name}: reader-facing text only (no internal label, task, file or column name; no word for the "
               "revision process)", f"{len(texts)} texts; {len(bad)} with an internal name" + (f": {bad}" if bad else ""),
               not bad)


def save(checks: t5.Checks, figs: lib.Figures, fig, name: str, description: str) -> None:
    check_text(checks, fig, name)
    row = figs.save(fig, name, description, lib.FULL_MM)
    checks.add(f"{name}: figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in "
               "the PDF)", f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps "
               f"{row['n_text_overlaps']}; outside {row['n_text_outside_figure']}", bool(row["pass"]))


# =====================================================================================================================
# S23
# =====================================================================================================================
def figure_a(run, checks, figs, values, per_cell, spread, knn, pt_group, pt_ref, dpt_summary) -> None:
    h_mm = 209.0
    col_x, ax_w = {"Ctrl": 16.0, "ZeroK": 104.0}, 70.0
    rows = {"A": (179.0, 20.0), "B": (132.0, 22.0), "C": (100.0, 18.0), "D": (64.0, 22.0), "E": (28.0, 22.0)}
    fig = lib.new_figure(W, h_mm)
    for cond in CONDS:
        fig.text((col_x[cond] + ax_w / 2) / W, 205.5 / h_mm, COND_DISPLAY[cond], fontsize=8, fontweight="bold",
                 ha="center", va="baseline")
        pc = per_cell[per_cell["condition"] == cond]
        exp = LETTER[cond]

        # A: proteins per cell, the two ProLIF definitions
        y0, ah = rows["A"]
        ax = mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        score = pc["second_label"].eq("ProLIF")
        call = pc["is_prolif"].astype(str).str.lower().eq("true")
        sets = (("ProLIF", "marker-score label", pc.loc[score, "n_detected"], COL["score"]),
                ("other cells", "", pc.loc[~score, "n_detected"], COL["other"]),
                ("ProLIF", "per-cell test", pc.loc[call, "n_detected"], COL["ProLIF"]),
                ("other cells", "", pc.loc[~call, "n_detected"], COL["other"]))
        positions = (0, 1, 2.6, 3.6)
        for pos, (name, sub, v, colour) in zip(positions, sets):
            violin(ax, pos, v, colour)
            values.append({"figure": "S-12-a", "panel": "A", "condition": cond, "series": f"{name} {sub}".strip(),
                           "x": pos, "n": int(len(v)), "value": float(np.median(v)), "value_name": "median proteins per cell",
                           "source": "report_12_percell_prolif_per_cell.csv"})
        ax.set_xlim(-0.6, 4.2)
        ax.set_ylim(0, 5000)
        ax.set_yticks([0, 2000, 4000])
        tick_lines(ax, ah, positions, [(n, f"({len(v)})") for n, _s, v, _ in sets])
        trans = ax.get_xaxis_transform()
        for xc, text in ((0.5, "marker-score label"), (3.1, "per-cell test")):
            ax.plot([xc - 0.9, xc + 0.9], [-8.2 / ah] * 2, transform=trans, color=DARK, linewidth=0.6, clip_on=False)
            ax.text(xc, -9.2 / ah, text, transform=trans, ha="center", va="top", fontsize=7)
        if cond == "Ctrl":
            ax.set_ylabel("proteins per cell")
        checks.add(f"S-12-a A, {cond}: cells and median proteins per cell of the two ProLIF definitions = the letter",
                   f"marker-score {int(score.sum())} cells, median {np.median(pc.loc[score, 'n_detected']):.0f}; "
                   f"per-cell test {int(call.sum())} cells, median {np.median(pc.loc[call, 'n_detected']):.0f}",
                   int(score.sum()) == exp["n_score"] and int(call.sum()) == exp["n_call"]
                   and round(float(np.median(pc.loc[score, "n_detected"]))) == exp["ids_score"]
                   and round(float(np.median(pc.loc[call, "n_detected"]))) == exp["ids_call"])

        # B: largest share of one cluster made up of per-cell ProLIF cells
        y0, ah = rows["B"]
        ax = mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        sp = spread[(spread["condition"] == cond) & spread["object"].isin(OBJECTS)]
        for res, filled, dx in (("1.00", True, -0.16), ("3.00", False, 0.16)):
            s = sp[sp["resolution"].astype(float).map("{:.2f}".format) == res].set_index("object").loc[list(OBJECTS)]
            y = s["max_prolif_fraction_of_a_cluster"].to_numpy(float)
            ax.plot(np.arange(7) + dx, y, linestyle="none", marker="o", markersize=4, markeredgewidth=0.8,
                    markeredgecolor=COL["ProLIF"], markerfacecolor=COL["ProLIF"] if filled else "white", zorder=3)
            for i, obj in enumerate(OBJECTS):
                values.append({"figure": "S-12-a", "panel": "B", "condition": cond, "series": f"resolution {res[0]}",
                               "x": obj, "n": int(s.loc[obj, "cluster_of_max_fraction_n_cells"]), "value": float(y[i]),
                               "value_name": "largest ProLIF share of one cluster (n = cells of that cluster)",
                               "source": "report_12_percell_prolif_cluster_spread.csv"})
            if res == "1.00":
                ok = abs(round(float(y[0]), 2) - exp["max_share_r1_none"]) < 1e-9
                checks.add(f"S-12-a B, {cond}: largest ProLIF share of a cluster, no correction, resolution 1",
                           f"{y[0]:.2f}", ok, str(exp["max_share_r1_none"]))
            else:
                checks.add(f"S-12-a B, {cond}: largest ProLIF share of a cluster at resolution 3: no correction; "
                           "after removal of the three protein sets",
                           f"{y[0]:.2f}; " + ", ".join(f"{y[i]:.2f}" for i in (1, 3, 5)),
                           abs(y[0] - exp["max_share_r3_none"]) < 1e-9
                           and ("max_share_r3_removed" not in exp
                                or np.allclose([y[1], y[3], y[5]], exp["max_share_r3_removed"])))
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        object_ticks(ax, ah, False)
        if cond == "Ctrl":
            ax.set_ylabel("share of a cluster")

        # C: same-label neighbour fraction of the marker-score ProLIF cells
        y0, ah = rows["C"]
        ax = mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        k = knn[(knn["condition"] == cond) & (knn["label_role"] == "second") & (knn["label"] == "ProLIF")]
        k = k.set_index("object").loc[list(OBJECTS)]
        ax.plot(range(7), k["observed_same_label_fraction"], linestyle="none", marker="o", markersize=4,
                color=COL["score"], zorder=3)
        ax.hlines(k["null_mean"], np.arange(7) - 0.3, np.arange(7) + 0.3, color=DARK, linewidth=0.9, zorder=2)
        for obj in OBJECTS:
            values.append({"figure": "S-12-a", "panel": "C", "condition": cond, "series": "ProLIF, marker-score label",
                           "x": obj, "n": int(k.loc[obj, "n_cells"]), "value": float(k.loc[obj, "observed_same_label_fraction"]),
                           "value_name": "same-label neighbour fraction (k = 10)", "reference": float(k.loc[obj, "null_mean"]),
                           "source": "dct_reviewer_task4_knn_label_purity.csv"})
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        object_ticks(ax, ah, False)
        if cond == "Ctrl":
            ax.set_ylabel("neighbour fraction")

        # D: pseudotime, groups of the per-cell test
        y0, ah = rows["D"]
        ax = mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        g = pt_group[pt_group["condition"] == cond].set_index(["object", "group"])
        r = pt_ref[pt_ref["condition"] == cond].set_index("object").loc[list(OBJECTS)]
        ax.bar(np.arange(7), r["reference_q975"] - r["reference_q025"], bottom=r["reference_q025"], width=0.84,
               color=COL["reference"], linewidth=0, zorder=1)
        for grp, dx, marker in (("DCT1", -0.22, "D"), ("DCT2", 0.22, "D"), ("ProLIF", 0.0, "o")):
            if (OBJECTS[0], grp) not in g.index:
                continue
            y = np.array([g.loc[(o, grp), "median"] for o in OBJECTS], float)
            ax.plot(np.arange(7) + dx, y, linestyle="none", marker=marker, markersize=3.6 if marker == "D" else 4.2,
                    color=COL[grp], zorder=3)
            for i, obj in enumerate(OBJECTS):
                values.append({"figure": "S-12-a", "panel": "D", "condition": cond, "series": f"{grp}, per-cell test groups",
                               "x": obj, "n": int(g.loc[(obj, grp), "n"]), "value": float(y[i]),
                               "value_name": "median pseudotime",
                               "reference": f"{r.loc[obj, 'reference_q025']:.4f}-{r.loc[obj, 'reference_q975']:.4f}"
                               if grp == "ProLIF" else "", "source": "report_12_percell_prolif_pseudotime_by_group.csv; "
                               "report_12_percell_prolif_pseudotime_same_cluster_reference.csv"})
        med = {grp: float(g.loc[("uncorrected", grp), "median"]) for grp in ("DCT1", "DCT2", "ProLIF")
               if ("uncorrected", grp) in g.index}
        checks.add(f"S-12-a D, {cond}: median pseudotime without correction (per-cell test groups) = the letter",
                   "; ".join(f"{k_} {v:.2f}" for k_, v in med.items()),
                   round(med["ProLIF"], 2) == exp["pt_call"] and round(med["DCT1"], 2) == exp["pt_dct1"]
                   and ("pt_dct2" not in exp or round(med["DCT2"], 2) == exp["pt_dct2"]))
        ax.set_ylim(0.1, 0.5)
        ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
        object_ticks(ax, ah, False)
        if cond == "Ctrl":
            ax.set_ylabel("pseudotime")

        # E: pseudotime, marker-score label of the clusters
        y0, ah = rows["E"]
        ax = mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        d = dpt_summary[(dpt_summary["condition"] == cond) & (dpt_summary["label_role"] == "second")]
        d = d.set_index("object").loc[list(OBJECTS)]
        for grp, dx, marker, colour in (("DCT1", -0.22, "D", COL["DCT1"]), ("DCT2", 0.22, "D", COL["DCT2"]),
                                        ("ProLIF", 0.0, "o", COL["score"])):
            y = d[f"original_median_{grp}"].to_numpy(float)
            ax.plot(np.arange(7) + dx, y, linestyle="none", marker=marker, markersize=3.6 if marker == "D" else 4.2,
                    color=colour, zorder=3)
            for i, obj in enumerate(OBJECTS):
                values.append({"figure": "S-12-a", "panel": "E", "condition": cond, "series": f"{grp}, marker-score label",
                               "x": obj, "value": float(y[i]), "value_name": "median pseudotime",
                               "source": "dct_reviewer_task4_dpt_summary.csv"})
        checks.add(f"S-12-a E, {cond}: median pseudotime of the marker-score ProLIF cells without correction; not "
                   "between the DCT1 and DCT2 medians in any of the seven matrices",
                   f"{float(d.loc['uncorrected', 'original_median_ProLIF']):.2f}; between in "
                   f"{int(d['original_prolif_between'].astype(str).str.lower().eq('true').sum())} of 7",
                   round(float(d.loc["uncorrected", "original_median_ProLIF"]), 2) == exp["pt_score"]
                   and not d["original_prolif_between"].astype(str).str.lower().eq("true").any())
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1.0])
        object_ticks(ax, ah, True)
        if cond == "Ctrl":
            ax.set_ylabel("pseudotime")

    x0 = col_x["Ctrl"] - 11.0
    letter_title(fig, h_mm, x0, 201.2, "A", "Proteins identified per cell: ProLIF cells of the two definitions (number of cells)")
    letter_title(fig, h_mm, x0, 160.6, "B", "ProLIF cells of the per-cell test: largest share of one Leiden cluster")
    legend_row(fig, h_mm, col_x["Ctrl"], 155.6, [dot(COL["ProLIF"], "resolution 1"),
                                                 dot(COL["ProLIF"], "resolution 3", filled=False)])
    letter_title(fig, h_mm, x0, 124.6, "C", "ProLIF cells of the marker-score label: neighbours with the same label")
    legend_row(fig, h_mm, col_x["Ctrl"], 119.6, [dot(COL["score"], "observed"),
                                                 Line2D([], [], color=DARK, linewidth=0.9, label="mean of 1,000 label permutations")])
    letter_title(fig, h_mm, x0, 92.6, "D", "Median pseudotime: ProLIF cells of the per-cell test, DCT1 and DCT2 (cluster annotation)")
    legend_row(fig, h_mm, col_x["Ctrl"], 87.6,
               [dot(COL["DCT1"], "DCT1", "D", size=3.6), dot(COL["DCT2"], "DCT2", "D", size=3.6),
                dot(COL["ProLIF"], "ProLIF"),
                Patch(facecolor=COL["reference"], linewidth=0, label="median of random cells of the same clusters (central 95%)")])
    letter_title(fig, h_mm, x0, 56.6, "E", "Median pseudotime: ProLIF, DCT1 and DCT2 of the marker-score label")
    legend_row(fig, h_mm, col_x["Ctrl"], 51.6, [dot(COL["DCT1"], "DCT1", "D", size=3.6),
                                                dot(COL["DCT2"], "DCT2", "D", size=3.6), dot(COL["score"], "ProLIF")])
    save(checks, figs, fig, "12_S12a_cell_cycle_correction",
         "S-12-a: cell-cycle correction in the DCT data, control and zero-K: proteins per cell of the two ProLIF "
         "definitions (A); largest ProLIF share of a Leiden cluster by matrix (B); same-label neighbour fraction of "
         "the marker-score ProLIF cells (C); pseudotime medians, per-cell test (D) and marker-score label (E)")


# =====================================================================================================================
# S18
# =====================================================================================================================
def figure_b(run, checks, figs, values, per_cell, groups, by_cluster) -> None:
    h_mm = 122.0
    xs = {"A": (14.0, 44.0), "B": (70.0, 21.0), "C": (104.0, 42.0), "D": (158.0, 18.0)}
    row_y, ah = {"Ctrl": 76.0, "ZeroK": 22.0}, 38.0
    fig = lib.new_figure(W, h_mm)
    for cond in CONDS:
        bottom = cond == "ZeroK"
        pc = per_cell[per_cell["condition"] == cond]
        gr = groups[groups["condition"] == cond].set_index("group")
        exp = LETTER[cond]
        fig.text(3.0 / W, (row_y[cond] + ah / 2) / h_mm, COND_DISPLAY[cond], rotation=90, ha="center", va="center",
                 fontsize=8, fontweight="bold")

        # A: marker axis against the proliferation score
        ax = mm_axes(fig, h_mm, xs["A"][0], row_y[cond], xs["A"][1], ah)
        shown = pc[pc["axis"].notna() & pc["prolif_z"].notna()]
        for grp in ("Unknown", "DCT1", "DCT2", "ProLIF"):
            s = shown[shown["group"] == grp]
            ax.scatter(s["axis"], s["prolif_z"], s=9 if grp == "ProLIF" else 3, color=COL[grp], linewidths=0,
                       zorder=4 if grp == "ProLIF" else 2, rasterized=True)
            values.append({"figure": "S-12-b", "panel": "A", "condition": cond, "series": GROUP_DISPLAY[grp],
                           "n": int(len(s)), "value": float(s["prolif_z"].median()) if len(s) else np.nan,
                           "value_name": "cells plotted (n); median proliferation score",
                           "source": "report_12_percell_prolif_per_cell.csv"})
        ax.set_xlim(-4.2, 6.2)
        ax.set_ylim(-4, 16)
        ax.set_yticks([0, 5, 10, 15])
        ax.xaxis.grid(False)
        if cond == "Ctrl":
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("DCT2 minus DCT1 marker score")
        ax.set_ylabel("proliferation score (z)")
        n_p = int((shown["group"] == "ProLIF").sum())
        checks.add(f"S-12-b A, {cond}: cells with a marker axis plotted; ProLIF cells among them",
                   f"{len(shown)} cells; ProLIF {n_p} of {exp['n_call']}",
                   (len(shown), n_p) == {"Ctrl": (612, 34), "ZeroK": (576, 60)}[cond])

        # B: Mki67 or Top2a detected
        ax = mm_axes(fig, h_mm, xs["B"][0], row_y[cond], xs["B"][1], ah)
        for i, grp in enumerate(GROUPS):
            if grp not in gr.index:
                continue
            n, tot = int(gr.loc[grp, "n_mki67_or_top2a_detected"]), int(gr.loc[grp, "n_cells"])
            ax.bar(i, 100 * n / tot, width=0.7, color=COL[grp])
            ax.text(i, 100 * n / tot + 2.0, str(n), ha="center", va="bottom", fontsize=7)
            values.append({"figure": "S-12-b", "panel": "B", "condition": cond, "series": GROUP_DISPLAY[grp], "n": tot,
                           "value": n, "value_name": "cells with Mki67 or Top2a detected (n = cells of the group)",
                           "source": "report_12_percell_prolif_groups.csv"})
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, 75)
        ax.set_yticks([0, 25, 50, 75])
        ax.set_xticks(range(4), [GROUP_DISPLAY[g_] for g_ in GROUPS] if bottom else [""] * 4, rotation=90)
        ax.set_ylabel("% of cells")
        got = (int(gr.loc["ProLIF", "n_mki67_or_top2a_detected"]), int(gr.loc["ProLIF", "n_cells"]))
        checks.add(f"S-12-b B, {cond}: ProLIF cells with Mki67 or Top2a detected = the letter", f"{got[0]} of {got[1]}",
                   got == exp["mki_call"])

        # C: ProLIF cells per cluster
        ax = mm_axes(fig, h_mm, xs["C"][0], row_y[cond], xs["C"][1], ah)
        bc = by_cluster[(by_cluster["run"] == "nofloor") & (by_cluster["condition"] == cond)].copy()
        bc["order"] = bc["t27_label"].map({"DCT1": 0, "DCT2": 1, "Unknown": 2})
        bc = bc.sort_values(["order", "n_prolif", "t27_cluster"], ascending=[True, False, True]).reset_index(drop=True)
        for i, r_ in bc.iterrows():
            ax.bar(i, r_["n_cells"], width=0.78, color=COL[r_["t27_label"]], alpha=0.45, linewidth=0)
            ax.bar(i, r_["n_prolif"], width=0.78, color=COL["ProLIF"], linewidth=0)
            if r_["n_prolif"]:
                ax.text(i, r_["n_cells"] + 1.5, str(int(r_["n_prolif"])), ha="center", va="bottom", fontsize=7)
            values.append({"figure": "S-12-b", "panel": "C", "condition": cond, "series": GROUP_DISPLAY[r_["t27_label"]],
                           "x": f"cluster {int(r_['t27_cluster'])}", "n": int(r_["n_cells"]), "value": int(r_["n_prolif"]),
                           "value_name": "ProLIF cells in the cluster (n = cells of the cluster)",
                           "source": "dct_prolif_prolif_overlay_by_t27_cluster.csv"})
        ax.set_xlim(-0.7, 12.7)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks([])
        ax.set_ylabel("cells per cluster")
        if bottom:
            ax.set_xlabel("Leiden clusters (resolution 1)")
        checks.add(f"S-12-b C, {cond}: 13 clusters; clusters with ProLIF cells; ProLIF cells in all",
                   f"{len(bc)}; {int((bc['n_prolif'] > 0).sum())}; {int(bc['n_prolif'].sum())}",
                   len(bc) == 13 and int((bc["n_prolif"] > 0).sum()) == {"Ctrl": 8, "ZeroK": 9}[cond]
                   and int(bc["n_prolif"].sum()) == exp["n_call"])

        # D: proteins per cell
        ax = mm_axes(fig, h_mm, xs["D"][0], row_y[cond], xs["D"][1], ah)
        for i, grp in enumerate(GROUPS):
            v = pc.loc[pc["group"] == grp, "n_detected"]
            if len(v):
                violin(ax, i, v, COL[grp])
                values.append({"figure": "S-12-b", "panel": "D", "condition": cond, "series": GROUP_DISPLAY[grp],
                               "n": int(len(v)), "value": float(v.median()), "value_name": "median proteins per cell",
                               "source": "report_12_percell_prolif_per_cell.csv"})
        ax.set_xlim(-0.6, 3.6)
        ax.set_ylim(0, 5000)
        ax.set_yticks([0, 2000, 4000])
        ax.set_xticks(range(4), [GROUP_DISPLAY[g_] for g_ in GROUPS] if bottom else [""] * 4, rotation=90)
        ax.set_ylabel("proteins per cell")

    top = row_y["Ctrl"] + ah
    for key, title in (("A", "Marker axis and proliferation score"), ("B", "Mki67 or Top2a"),
                       ("C", "ProLIF cells per cluster (number)"), ("D", "Proteins")):
        letter_title(fig, h_mm, xs[key][0] - (9.0 if key != "A" else 8.0), top + 3.0, key, title)
    handles = [dot(COL[g_], GROUP_DISPLAY[g_], size=3.4) for g_ in ("DCT1", "DCT2", "Unknown")]
    handles.append(dot(COL["ProLIF"], "ProLIF (per-cell test)", size=4.4))
    handles.append(Patch(facecolor=COL["DCT1"], alpha=0.45, linewidth=0, label="cells of a cluster, by its annotation (C)"))
    legend_row(fig, h_mm, xs["A"][0], 1.5, handles)
    save(checks, figs, fig, "12_S12b_percell_test",
         "S-12-b: per-cell proliferation test in the DCT data, control and zero-K: DCT2 - DCT1 marker axis against "
         "the proliferation score (A); cells with Mki67 or Top2a detected (B); ProLIF cells per Leiden cluster of "
         "the cluster annotation (C); proteins per cell (D)")


# =====================================================================================================================
# S24
# =====================================================================================================================
def figure_c(run, checks, figs, values, rnd) -> None:
    h_mm = 132.0
    ax_w, gap, left = 35.0, 6.5, 15.0
    row_y, ah = {"Ctrl": 82.0, "ZeroK": 31.0}, 38.0
    panels = (("A", "per_cell_call", "ProLIF_vs_DCT1", ("ProLIF (per-cell test)", "minus DCT1")),
              ("B", "per_cell_call", "ProLIF_vs_DCT2", ("ProLIF (per-cell test)", "minus DCT2")),
              ("C", "marker_score_clusters", "ProLIF_vs_DCT1", ("ProLIF (marker-score label)", "minus DCT1")),
              ("D", "marker_score_clusters", "ProLIF_vs_DCT2", ("ProLIF (marker-score label)", "minus DCT2")))
    scores = (("ALDO_activity", "base"), ("ALDO_activity", "no_Slc12a3"), ("ALDO_signed", "base"),
              ("ALDO_signed", "no_Slc12a3"), ("ALDO_machinery", "base"))
    ticks = ("+ Slc12a3", "− Slc12a3", "+ Slc12a3", "− Slc12a3", "machinery")
    sources = (("scaled_X", -0.19, True), ("cc_regressed_3a", 0.19, False))
    fig = lib.new_figure(W, h_mm)
    inside = {c: [0, 0] for c in CONDS}
    for cond in CONDS:
        bottom = cond == "ZeroK"
        fig.text(3.0 / W, (row_y[cond] + ah / 2) / h_mm, COND_DISPLAY[cond], rotation=90, ha="center", va="center",
                 fontsize=8, fontweight="bold")
        for j, (letter, definition, comp, title) in enumerate(panels):
            x0 = left + j * (ax_w + gap)
            ax = mm_axes(fig, h_mm, x0, row_y[cond], ax_w, ah)
            sub = rnd[(rnd["condition"] == cond) & (rnd["prolif_definition"] == definition) & (rnd["comparison"] == comp)]
            if cond == "Ctrl":
                for k, text in enumerate(title):
                    ax.text(0.0, 1.0 + (5.2 - k * 3.2) / ah, text, transform=ax.transAxes, ha="left", va="baseline",
                            fontsize=7.5)
                ax.text(-5.0 / ax_w, 1.0 + 5.2 / ah, letter, transform=ax.transAxes, ha="left", va="baseline",
                        fontsize=8.5, fontweight="bold")
            ax.set_xlim(-0.6, 4.6)
            ax.set_ylim(-2.9, 2.3)
            ax.set_yticks([-2, -1, 0, 1, 2])
            if j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Cohen's d")
            ax.set_xticks(range(5), ticks if bottom else [""] * 5, rotation=90)
            if bottom:
                trans = ax.get_xaxis_transform()
                for xc, name in ((0.5, "activity"), (2.5, "signed")):
                    ax.plot([xc - 0.8, xc + 0.8], [-15.6 / ah] * 2, transform=trans, color=DARK, linewidth=0.6, clip_on=False)
                    ax.text(xc, -16.6 / ah, name, transform=trans, ha="center", va="top", fontsize=7)
            if sub.empty:
                ax.text(0.5, 0.5, "no DCT2 cluster", transform=ax.transAxes, ha="center", va="center", fontsize=7,
                        color=DARK)
                continue
            ax.axhline(0, color=DARK, linewidth=0.6, zorder=1)
            for i, (score, variant) in enumerate(scores):
                for source, dx, filled in sources:
                    r_ = sub[(sub["source"] == source) & (sub["score"] == score) & (sub["variant"] == variant)].iloc[0]
                    lo, hi, mid = (float(r_[f"cohens_d_random_{q}"]) for q in ("q025", "q975", "median"))
                    obs = float(r_["cohens_d_observed"])
                    ax.vlines(i + dx, lo, hi, color=COL["reference"], linewidth=3.2, zorder=2)
                    ax.hlines(mid, i + dx - 0.13, i + dx + 0.13, color=DARK, linewidth=0.7, zorder=2.5)
                    ax.plot([i + dx], [obs], linestyle="none", marker="o", markersize=3.6, markeredgewidth=0.8,
                            markeredgecolor=INK, markerfacecolor=INK if filled else "white", zorder=4)
                    inside[cond][0] += int(lo <= obs <= hi)
                    inside[cond][1] += 1
                    values.append({"figure": "S-12-c", "panel": letter, "condition": cond,
                                   "series": "main matrix" if filled else "cell-cycle score regressed",
                                   "x": f"{score} ({variant})", "n": int(r_["n_prolif"]), "value": obs,
                                   "value_name": "Cohen's d, ProLIF minus reference (n = ProLIF cells)",
                                   "reference": f"{lo:.4f}-{hi:.4f} (median {mid:.4f})",
                                   "source": "report_aldo_random_sets.csv"})
    for cond in CONDS:
        checks.add(f"S-12-c, {cond}: contrasts drawn; observed value inside the central 95% of the random sets",
                   f"{inside[cond][1]} drawn; {inside[cond][0]} inside", inside[cond][1] == {"Ctrl": 40, "ZeroK": 30}[cond])
    # letter numbers: per-cell call against DCT1 after regression, ALDO activity (base)
    for cond, exp_obs, exp_lo, exp_hi in (("Ctrl", 1.18, 1.39, 1.94), ("ZeroK", 0.84, 0.88, 1.22)):
        r_ = rnd[(rnd["condition"] == cond) & (rnd["prolif_definition"] == "per_cell_call")
                 & (rnd["comparison"] == "ProLIF_vs_DCT1") & (rnd["source"] == "cc_regressed_3a")
                 & (rnd["score"] == "ALDO_activity") & (rnd["variant"] == "base")].iloc[0]
        got = (round(float(r_["cohens_d_observed"]), 2), round(float(r_["cohens_d_random_q025"]), 2),
               round(float(r_["cohens_d_random_q975"]), 2))
        checks.add(f"S-12-c A, {cond}: ALDO activity after regression, per-cell test against DCT1: d and the central "
                   "95% of the random sets = the letter", f"{got[0]}; {got[1]} to {got[2]}", got == (exp_obs, exp_lo, exp_hi))
    handles = [dot(INK, "main matrix", size=3.6), dot(INK, "cell-cycle score regressed out", filled=False, size=3.6),
               Line2D([], [], color=COL["reference"], linewidth=3.2, label="1,000 random protein sets (central 95%; line, median)")]
    legend_row(fig, h_mm, left, 1.5, handles)
    save(checks, figs, fig, "12_S12c_aldo_scores",
         "S-12-c: ALDO scores of the ProLIF cells against DCT1 and DCT2 (Cohen's d) in the DCT data, control and "
         "zero-K, with and without Slc12a3, on the main matrix and after regression of the cell-cycle score, against "
         "1,000 random protein sets of the same size and detection rate")


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figures S-12-a, S-12-b, S-12-c.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")

    def table(task: str, name: str) -> pd.DataFrame:
        path = cc.out_path(task, "tables", name, ".csv")
        return pd.read_csv(run.use_input(path, f"{task}: {name}"), keep_default_na=False, na_values=[""])

    per_cell = table(TASK, "12_percell_prolif_per_cell")
    groups = table(TASK, "12_percell_prolif_groups")
    spread = table(TASK, "12_percell_prolif_cluster_spread")
    pt_group = table(TASK, "12_percell_prolif_pseudotime_by_group")
    pt_ref = table(TASK, "12_percell_prolif_pseudotime_same_cluster_reference")
    rnd = table(TASK, "12_aldo_random_sets")
    knn = table(DCT_REVIEWER, "task4_knn_label_purity")
    dpt_summary = table(DCT_REVIEWER, "task4_dpt_summary")
    by_cluster = table(DCT_PROLIF, "prolif_overlay_by_t27_cluster")
    rnd = rnd[rnd["score"] != "all_proteins"]

    figs = lib.Figures(run, STEP)
    values: list[dict] = []
    figure_a(run, checks, figs, values, per_cell, spread, knn, pt_group, pt_ref, dpt_summary)
    figure_b(run, checks, figs, values, per_cell, groups, by_cluster)
    figure_c(run, checks, figs, values, rnd)
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"figures": ["12_S12a_cell_cycle_correction", "12_S12b_percell_test", "12_S12c_aldo_scores"],
                "width_mm": W, "colours": COL, "label_display": GROUP_DISPLAY, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
