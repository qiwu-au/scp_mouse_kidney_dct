#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S-01-b - proteins per cell in the DCT data, control and zero-K.

The authors asked on 12 Nov 2026 to redraw the two
supplementary figures that existed only as analysis figures ("Yes, redraw them"). The response to the reviewers, section 5, fixes the
content of S-01-b: the proteins per cell of all control and zero-K cells (removed cells ringed; line at 100
proteins) and of the retained cells over the proteins retained in both conditions, by condition and sort plate.
This script redraws the dct_intake analysis figure (dct_intake_qc.py, dct_intake_proteins_per_cell) as one reader-facing
figure. No new statistics: the points are the stored per-cell counts and the boxes are the stored quartiles.

Rows = condition (within condition only; nothing compares the conditions). Panels:
  A  all sorted cells, proteins counted over all protein groups of the search; cells removed by the filter (fewer
     than 100 proteins) are ringed; dashed line at 100 proteins. Left: the full range. Right: the same cells
     between 0 and 200 proteins, enlarged (on the full range 100 proteins are 0.7 mm, so the ringed cells and
     the line cannot be told apart there);
  B  retained cells, proteins counted over the protein groups retained in both conditions (--retained-proteins
     joint, the plan of the response to the reviewers) or over the protein groups retained in the condition (--retained-proteins
     condition; the counts behind the ranges of the letter, 100 to 4,715 and 105 to 4,466).
Points: one cell, coloured by sort plate (plate 1, plate 2: technical sort plates of the pooled cells, never mice
or replicates; the plate colours of the dct_intake figure). Box: quartiles and median as stored in the dct_intake tables (no
whiskers: the 1.5 x IQR whiskers of the analysis figure are not in a stored table).
dotplot / dct_labels format: Arial >= 7 pt, exact 180 mm width, PDF + 600-dpi PNG (+ TIFF); one assembled file with panel
letters; no title, footer or internal names in the figure.

Inputs (read-only): results/tables/dct_intake/dct_intake_{dct_cells,qc_per_cell,filtering_summary,
  proteins_per_cell_range,qc_by_condition_plate,filtering_removed_cells,proteins}.csv
Outputs: results/figures/report/report_01_S01b_proteins_per_cell.{pdf,png,tif};
  results/tables/report/report_01_s01b_figure_{values,checks}.csv;
  results/tables/report/report_step_01_s01b_figure_manifest.json

Command (project root):  python scripts/report_proteins_per_cell_figure.py --overwrite
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
import re

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dct_labels_figures as t27f
import dct_reviewer_lib as lib

lib.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

TASK = "report"
STEP = "01_s01b_figure"
DCT_INTAKE_TASK = "dct_intake"
W, H = 180.0, 116.0                        # mm
AXES = {"A": (21.0, 52.0), "zoom": (81.0, 38.0), "B": (129.0, 49.0)}     # left edge and width of each axis, mm
AX_H = 38.0
ROW_Y = {"Ctrl": 67.0, "ZeroK": 16.0}
BASELINE, LINE = 1.6, 3.2                  # header: last baseline above the axis top; line pitch (mm)
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
PLATES = ("P1", "P2")
PLATE_DISPLAY = {"P1": "plate 1", "P2": "plate 2"}
INK, DARK = "#0b0b0b", "#52514e"
JITTER, BOX_HALF = 0.26, 0.32
MIN_PROTEINS = cc.MIN_CELL_PROTEINS        # 100: the cell filter of every dataset
ZOOM_MAX = 200                             # upper end of the enlarged range of panel A
QUANTS = ("min", "q25", "median", "q75", "max")
# Canonical counts (the project notes; dct_intake README) and the numbers of the the response to the reviewers letter text (section 2).
EXPECTED = {
    "Ctrl": {"sorted": 768, "retained": 729, "removed": 39, "removed_range": (2, 92),
             "retained_range": (100, 4715), "retained_median": 1767.0, "retained_proteins": 6078},
    "ZeroK": {"sorted": 767, "retained": 702, "removed": 65, "removed_range": (0, 76),
              "retained_range": (105, 4466), "retained_median": 1780.5, "retained_proteins": 5953},
}
N_GROUPS, N_JOINT = 6208, 5897
# Panel B: which protein set the retained cells are counted over.
B_SETS = {
    "joint": {"column": "n_detected_joint", "protein_set": "joint proteins (retained in both conditions)",
              "title": (f"retained cells, {N_JOINT:,} proteins", "retained in both conditions"),
              "counted_over": f"the {N_JOINT:,} protein groups retained in both conditions"},
    "condition": {"column": "n_detected", "protein_set": "own retained proteins",
                  "title": ("retained cells, proteins", "retained in the condition"),
                  "counted_over": "the protein groups retained in the condition (control 6,078; zero-K 5,953)"},
}
TITLES = {"A": ("all sorted cells,", f"all {N_GROUPS:,} protein groups"),
          "zoom": (f"0 to {ZOOM_MAX} proteins,", "enlarged")}
# Internal names that must not appear in the figure (reader-facing).
BANNED = ("ctrl", "zerok", "t26", "t28", "task", "generated", "unknown", "ids", "_", ".csv", "joint")


def t26(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = cc.out_path(DCT_INTAKE_TASK, "tables", name, ".csv")
    return pd.read_csv(run.use_input(path, f"dct_intake table: {name}"), **kwargs)


def five(values) -> dict[str, float]:
    """Minimum, quartiles and maximum of the per-cell counts, to compare with the stored rows (not plotted)."""
    v = np.asarray(values, dtype=float)
    q = np.percentile(v, [25, 50, 75])
    return {"min": float(v.min()), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]),
            "max": float(v.max())}


def same(per_cell: dict[str, float], stored: pd.Series, prefix: str = "") -> bool:
    return all(abs(per_cell[k] - float(stored[f"{prefix}{k}"])) < 1e-9 for k in QUANTS)


def new_axes(fig, key: str, cond: str):
    x, w = AXES[key]
    ax = fig.add_axes([x / W, ROW_Y[cond] / H, w / W, AX_H / H])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5)
    lib.grid_y(ax)
    ax.set_xlim(-0.6, len(PLATES) - 0.4)
    return ax


def header(ax, key: str, lines: tuple[str, ...], letter: str = "") -> None:
    """Panel letter (bold) and title lines above an axis; the letter sits on the baseline of the first line."""
    w = AXES[key][1]
    for k, text in enumerate(lines):
        y = 1.0 + (BASELINE + (len(lines) - 1 - k) * LINE) / AX_H
        ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="baseline", fontsize=7.5)
        if k == 0 and letter:
            ax.text(-5.0 / w, y, letter, transform=ax.transAxes, ha="left", va="baseline", fontsize=8.5,
                    fontweight="bold")


def jitter(frame: pd.DataFrame) -> np.ndarray:
    """Horizontal position of each cell: plate slot plus a fixed random offset (seed 42), reused in every view."""
    rng = np.random.default_rng(cc.RANDOM_SEED)
    x = np.zeros(len(frame))
    plate = frame["plate"].to_numpy()
    for i, p in enumerate(PLATES):
        m = plate == p
        x[m] = i + rng.uniform(-JITTER, JITTER, int(m.sum()))
    return x


def points(ax, frame: pd.DataFrame, x: np.ndarray, column: str, ring: str | None, ring_size: float = 9) -> None:
    """One point per cell (stored per-cell counts), coloured by sort plate; removed cells ringed."""
    v = frame[column].to_numpy(float)
    colours = frame["plate"].map(cc.PLATE_PALETTE).to_numpy()
    ax.scatter(x, v, s=3.5, c=list(colours), alpha=0.75, linewidths=0, zorder=2)
    if ring is not None:
        m = frame[ring].to_numpy(bool)
        ax.scatter(x[m], v[m], s=ring_size, facecolors="none", edgecolors=INK, linewidths=0.5, zorder=5)


def boxes(ax, stored: dict[str, pd.Series], prefix: str) -> None:
    """Quartile box and median line per plate, as stored."""
    for i, plate in enumerate(PLATES):
        q25, med, q75 = (float(stored[plate][f"{prefix}{k}"]) for k in ("q25", "median", "q75"))
        ax.add_patch(Rectangle((i - BOX_HALF, q25), 2 * BOX_HALF, q75 - q25, facecolor="none", edgecolor=DARK,
                               linewidth=0.7, zorder=3))
        ax.plot([i - BOX_HALF, i + BOX_HALF], [med, med], color=INK, linewidth=1.2, solid_capstyle="butt", zorder=4)


def full_range(ax) -> None:
    ax.set_ylim(-140, 5000)
    ax.set_yticks(range(0, 5001, 1000))
    ax.yaxis.set_major_formatter(lambda y, _pos: f"{y:,.0f}")


def threshold_line(ax) -> None:
    ax.axhline(MIN_PROTEINS, color=DARK, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: figure S-01-b.")
    parser.add_argument("--retained-proteins", choices=tuple(B_SETS), default="joint",
                        help="Protein set of panel B: joint = retained in both conditions; "
                             "condition = retained in the condition (the ranges of the letter).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    bset = B_SETS[args.retained_proteins]
    titles = {**TITLES, "B": bset["title"]}

    cells = t26(run, "dct_cells", dtype={"well": str})                 # all sorted cells
    cells["removed"] = ~cells["retained"].astype(str).str.lower().eq("true")
    qc = t26(run, "qc_per_cell", dtype={"well": str})                  # retained cells
    summ = t26(run, "filtering_summary").set_index(["condition", "plate", "stage"])
    ranges = t26(run, "proteins_per_cell_range").set_index(["condition", "plate"])
    byplate = t26(run, "qc_by_condition_plate")
    removed_list = t26(run, "filtering_removed_cells", dtype={"well": str})
    prot = t26(run, "proteins", usecols=["retained_ctrl", "retained_zerok", "in_joint"])
    conds = [c for c in lib.CONDITIONS if c in set(cells["condition"])]
    checks.add("tables hold both conditions", "; ".join(conds), conds == list(lib.CONDITIONS))

    # --- protein sets -------------------------------------------------------------------------------------------
    n_ret = {"Ctrl": int(prot["retained_ctrl"].sum()), "ZeroK": int(prot["retained_zerok"].sum())}
    checks.add("protein groups: in the search; retained in control, in zero-K, in both (protein table)",
               f"{len(prot)}; {n_ret['Ctrl']}, {n_ret['ZeroK']}, {int(prot['in_joint'].sum())}",
               len(prot) == N_GROUPS and int(prot["in_joint"].sum()) == N_JOINT
               and all(n_ret[c] == EXPECTED[c]["retained_proteins"] for c in conds),
               f"{N_GROUPS}; 6078, 5953, {N_JOINT}")
    checks.add("protein groups in the filtering summary (all rows; retained per condition)",
               "; ".join(f"{c} {int(summ.loc[(c, 'all', 'raw_all_cells'), 'n_protein_groups'])}, "
                         f"{int(summ.loc[(c, 'all', 'retained_cells_retained_proteins'), 'n_protein_groups'])}"
                         for c in conds),
               all(int(summ.loc[(c, "all", "raw_all_cells"), "n_protein_groups"]) == N_GROUPS
                   and int(summ.loc[(c, "all", "retained_cells_retained_proteins"), "n_protein_groups"])
                   == EXPECTED[c]["retained_proteins"] for c in conds))

    # --- the two per-cell tables hold the same retained cells and counts ----------------------------------------
    kept = cells[~cells["removed"]].set_index("cell_id")
    q = qc.set_index("cell_id")
    same_cells = set(kept.index) == set(q.index) and len(q) == len(kept)
    same_counts = same_cells and all(bool((kept.loc[q.index, c] == q[c]).all())
                                     for c in ("n_detected", "n_detected_joint", "plate", "condition"))
    checks.add("retained cells: the per-cell QC table and the cell table hold the same cells and counts",
               f"{len(q)} cells; same cells {same_cells}; same counts {same_counts}", same_counts)
    checks.add("removed cells: the ringed cells are the cells of the removed-cell list",
               f"{int(cells['removed'].sum())} ringed; list {len(removed_list)}",
               set(cells.loc[cells["removed"], "cell_key"]) == set(removed_list["cell_key"]))
    below = cells["n_detected_all_rows"] < MIN_PROTEINS
    checks.add(f"filter: every ringed cell has fewer than {MIN_PROTEINS} proteins (all protein groups) and every "
               "other cell has at least that many", f"{int((below & cells['removed']).sum())} ringed below; "
               f"{int((below != cells['removed']).sum())} cells disagree", bool((below == cells["removed"]).all()))

    values = []
    fig = lib.new_figure(W, H)
    for cond in conds:
        top = cond == conds[0]
        exp = EXPECTED[cond]
        allc = cells[cells["condition"] == cond]
        ret = qc[qc["condition"] == cond]
        n_plate = allc.groupby("plate").size().to_dict()
        n_removed = allc.groupby("plate")["removed"].sum().astype(int).to_dict()
        n_kept = ret.groupby("plate").size().to_dict()
        base = {"condition": cond, "condition_display": COND_DISPLAY[cond]}

        # --- checks: cell numbers ------------------------------------------------------------------------------
        got = (len(allc), len(ret), int(allc["removed"].sum()))
        stored = (int(summ.loc[(cond, "all", "raw_all_cells"), "n_cells"]),
                  int(summ.loc[(cond, "all", "retained_cells_retained_proteins"), "n_cells"]),
                  int(summ.loc[(cond, "all", "removed_cells"), "n_cells"]))
        checks.add(f"{cond}: cells sorted, retained, removed = canonical counts and the filtering summary",
                   f"{got[0]}, {got[1]}, {got[2]}; summary {stored[0]}, {stored[1]}, {stored[2]}",
                   got == stored == (exp["sorted"], exp["retained"], exp["removed"])
                   and int(ranges.loc[(cond, "all"), "n_retained_cells"]) == exp["retained"],
                   f"{exp['sorted']}, {exp['retained']}, {exp['removed']}")
        per_plate_ok = all(
            int(summ.loc[(cond, p, "raw_all_cells"), "n_cells"]) == n_plate[p]
            and int(summ.loc[(cond, p, "removed_cells"), "n_cells"]) == n_removed[p]
            and int(summ.loc[(cond, p, "retained_cells_retained_proteins"), "n_cells"]) == n_kept[p]
            and int(ranges.loc[(cond, p), "n_retained_cells"]) == n_kept[p] for p in PLATES)
        checks.add(f"{cond}: cells per sort plate (sorted, removed, retained) = the filtering summary and range table",
                   "; ".join(f"{PLATE_DISPLAY[p]} {n_plate[p]}, {n_removed[p]}, {n_kept[p]}" for p in PLATES),
                   per_plate_ok)

        # --- checks: removed cells and the ranges of the letter (retained cells, retained proteins) -------------
        rem = five(allc.loc[allc["removed"], "n_detected_all_rows"])
        rs = summ.loc[(cond, "all", "removed_cells")]
        checks.add(f"{cond}: proteins per removed cell (range) = the filtering summary and the the response to the reviewers text",
                   f"{rem['min']:.0f} to {rem['max']:.0f}; summary {int(rs['protein_ids_min'])} to "
                   f"{int(rs['protein_ids_max'])}",
                   (rem["min"], rem["max"]) == (rs["protein_ids_min"], rs["protein_ids_max"]) == exp["removed_range"],
                   f"{exp['removed_range'][0]} to {exp['removed_range'][1]}")
        own_ok, own_text = True, []
        for p in ("all", *PLATES):
            v = five(ret["n_detected"] if p == "all" else ret.loc[ret["plate"] == p, "n_detected"])
            own_ok &= same(v, summ.loc[(cond, p, "retained_cells_retained_proteins")], "protein_ids_")
            own_ok &= same(v, ranges.loc[(cond, p)], "protein_ids_")
            own_text.append(f"{PLATE_DISPLAY.get(p, p)} {v['min']:.0f} to {v['max']:.0f} (median {v['median']:g})")
        checks.add(f"{cond}: proteins per retained cell over the retained proteins of the condition (minimum, "
                   "quartiles, maximum) = the filtering summary and the range table", "; ".join(own_text), own_ok)
        v = five(ret["n_detected"])
        checks.add(f"{cond}: range and median of the the response to the reviewers text (retained cells, retained proteins)",
                   f"{v['min']:.0f} to {v['max']:.0f}, median {v['median']:g}",
                   (v["min"], v["max"]) == exp["retained_range"] and v["median"] == exp["retained_median"],
                   f"{exp['retained_range'][0]} to {exp['retained_range'][1]}, median {exp['retained_median']:g}")

        # --- A: all sorted cells, all protein groups (full range, and 0 to ZOOM_MAX enlarged) --------------------
        x_all = jitter(allc)
        stored_a = {p: summ.loc[(cond, p, "raw_all_cells")] for p in PLATES}
        ax_a = new_axes(fig, "A", cond)
        points(ax_a, allc, x_all, "n_detected_all_rows", "removed")
        boxes(ax_a, stored_a, "protein_ids_")
        threshold_line(ax_a)
        full_range(ax_a)
        ax_a.set_xticks(range(len(PLATES)), [f"{PLATE_DISPLAY[p]}\n({n_plate[p]} cells)" for p in PLATES])
        ax_a.set_ylabel("proteins per cell")
        ax_z = new_axes(fig, "zoom", cond)
        in_view = (allc["n_detected_all_rows"] <= ZOOM_MAX).to_numpy()
        points(ax_z, allc[in_view], x_all[in_view], "n_detected_all_rows", "removed", ring_size=12)
        threshold_line(ax_z)
        ax_z.set_ylim(-9, ZOOM_MAX + 6)
        ax_z.set_yticks(range(0, ZOOM_MAX + 1, 50))
        ax_z.set_xticks(range(len(PLATES)), [f"{PLATE_DISPLAY[p]}\n({n_removed[p]} removed)" for p in PLATES])
        ok_a = True
        for p in PLATES:
            on_plate = (allc["plate"] == p).to_numpy()
            v = five(allc.loc[on_plate, "n_detected_all_rows"])
            ok_a &= same(v, stored_a[p], "protein_ids_")
            values.append({"panel": "A", "view": "full range", **base, "plate": p, "plate_display": PLATE_DISPLAY[p],
                           "n_cells_plotted": n_plate[p], "n_cells_ringed_removed": n_removed[p],
                           "box_q25": float(stored_a[p]["protein_ids_q25"]),
                           "box_median": float(stored_a[p]["protein_ids_median"]),
                           "box_q75": float(stored_a[p]["protein_ids_q75"]), "points_min": v["min"],
                           "points_max": v["max"], "dashed_line": MIN_PROTEINS,
                           "counted_over": f"all {N_GROUPS:,} protein groups of the search",
                           "per_cell_source": "dct_intake_dct_cells.csv: n_detected_all_rows, retained",
                           "box_source": "dct_intake_filtering_summary.csv: stage raw_all_cells"})
            shown = allc.loc[on_plate & in_view, "n_detected_all_rows"]
            values.append({"panel": "A", "view": f"0 to {ZOOM_MAX} proteins, enlarged", **base, "plate": p,
                           "plate_display": PLATE_DISPLAY[p], "n_cells_plotted": int(len(shown)),
                           "n_cells_ringed_removed": n_removed[p], "box_q25": np.nan, "box_median": np.nan,
                           "box_q75": np.nan, "points_min": float(shown.min()), "points_max": float(shown.max()),
                           "dashed_line": MIN_PROTEINS,
                           "counted_over": f"all {N_GROUPS:,} protein groups of the search",
                           "per_cell_source": "dct_intake_dct_cells.csv: n_detected_all_rows, retained (the cells "
                                              f"of the full range with at most {ZOOM_MAX} proteins)",
                           "box_source": "no box (the quartiles lie above the enlarged range)"})
        checks.add(f"{cond} A: plotted cells per plate (minimum, quartiles, maximum) = the filtering summary, all "
                   "cells", "; ".join(f"{PLATE_DISPLAY[p]} median {float(stored_a[p]['protein_ids_median']):g}"
                                      for p in PLATES), ok_a)
        checks.add(f"{cond} A, enlarged range: every removed cell is inside it (0 to {ZOOM_MAX} proteins)",
                   f"{int((allc['removed'].to_numpy() & in_view).sum())} of {exp['removed']} removed cells shown",
                   int((allc["removed"].to_numpy() & in_view).sum()) == exp["removed"])

        # --- B: retained cells -----------------------------------------------------------------------------------
        ax_b = new_axes(fig, "B", cond)
        col = bset["column"]
        bp = byplate[(byplate["condition"] == cond) & (byplate["protein_set"] == bset["protein_set"])
                     & (byplate["metric"] == col)].set_index("plate")
        stored_b = {p: bp.loc[p] for p in PLATES}
        points(ax_b, ret, jitter(ret), col, None)
        boxes(ax_b, stored_b, "")
        full_range(ax_b)
        ax_b.set_xticks(range(len(PLATES)), [f"{PLATE_DISPLAY[p]}\n({n_kept[p]} cells)" for p in PLATES])
        ok_b = True
        for p in PLATES:
            v = five(ret.loc[ret["plate"] == p, col])
            ok_b &= same(v, stored_b[p]) and int(stored_b[p]["n_cells"]) == n_kept[p]
            values.append({"panel": "B", "view": "full range", **base, "plate": p, "plate_display": PLATE_DISPLAY[p],
                           "n_cells_plotted": n_kept[p], "n_cells_ringed_removed": 0,
                           "box_q25": float(stored_b[p]["q25"]), "box_median": float(stored_b[p]["median"]),
                           "box_q75": float(stored_b[p]["q75"]), "points_min": v["min"], "points_max": v["max"],
                           "dashed_line": np.nan, "counted_over": bset["counted_over"],
                           "per_cell_source": f"dct_intake_qc_per_cell.csv: {col}",
                           "box_source": f"dct_intake_qc_by_condition_plate.csv: {bset['protein_set']}, {col}"})
        v_all = five(ret[col])
        ok_b &= same(v_all, bp.loc["all"]) and int(bp.loc["all", "n_cells"]) == exp["retained"]
        checks.add(f"{cond} B: plotted retained cells (minimum, quartiles, maximum; all and per plate) = the stored "
                   f"QC summary, {bset['protein_set']}",
                   f"all {v_all['min']:.0f} to {v_all['max']:.0f} (median {v_all['median']:g}); "
                   + "; ".join(f"{PLATE_DISPLAY[p]} median {float(stored_b[p]['median']):g}" for p in PLATES), ok_b)
        values.append({"panel": "B", "view": "full range", **base, "plate": "all", "plate_display": "both plates",
                       "n_cells_plotted": exp["retained"], "n_cells_ringed_removed": 0,
                       "box_q25": float(bp.loc["all", "q25"]), "box_median": float(bp.loc["all", "median"]),
                       "box_q75": float(bp.loc["all", "q75"]), "points_min": v_all["min"], "points_max": v_all["max"],
                       "dashed_line": np.nan, "counted_over": bset["counted_over"],
                       "per_cell_source": f"dct_intake_qc_per_cell.csv: {col}",
                       "box_source": "no box drawn for both plates together (numbers for the legend): "
                                     "dct_intake_qc_by_condition_plate.csv"})

        fig.text(3.0 / W, (ROW_Y[cond] + AX_H / 2) / H, COND_DISPLAY[cond], rotation=90, ha="center", va="center",
                 fontsize=7.5, fontweight="bold")
        if top:
            header(ax_a, "A", titles["A"], "A")
            header(ax_z, "zoom", titles["zoom"])
            header(ax_b, "B", titles["B"], "B")

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor=INK,
                      markeredgewidth=0.5, markersize=3.6, label=f"removed cell (fewer than {MIN_PROTEINS} proteins)"),
               Line2D([], [], color=DARK, linewidth=0.7, linestyle=(0, (3, 2)), label=f"{MIN_PROTEINS} proteins"),
               Patch(facecolor="none", edgecolor=DARK, linewidth=0.7, label="quartiles (box) and median (line)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(AXES["A"][0] / W, 1.5 / H), ncol=3, frameon=False,
               fontsize=7, handlelength=1.6, handletextpad=0.5, columnspacing=2.0, borderaxespad=0)

    texts = [t.get_text() for t in t27f.drawn_texts(fig)]
    bad = [t for t in texts if any(b in t.lower() for b in BANNED) or re.search(r"\bP[12]\b", t)]
    checks.add("reader-facing text: no internal names (Ctrl, ZeroK, P1 / P2, task codes, column or file names, "
               "'Generated by')", f"{len(texts)} texts; {len(bad)} with an internal name" + (f": {bad}" if bad else ""),
               not bad)
    figs = lib.Figures(run, STEP)
    row = figs.save(fig, "01_S01b_proteins_per_cell",
                    "S-01-b: proteins per cell in the DCT data, control and zero-K, by sort plate: all sorted cells "
                    f"over all protein groups, removed cells ringed, line at {MIN_PROTEINS} proteins, with the range "
                    f"0 to {ZOOM_MAX} enlarged (A); retained cells over {bset['counted_over']} (B)", lib.FULL_MM)
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in PDF)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps "
               f"{row['n_text_overlaps']}; outside {row['n_text_outside_figure']}", bool(row["pass"]))
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"panels": {k: " ".join(v) for k, v in titles.items()}, "panel_b_protein_set": bset["protein_set"],
                "panel_b_column": bset["column"], "enlarged_range_max": ZOOM_MAX, "width_mm": W, "height_mm": H,
                "plate_colours": {PLATE_DISPLAY[p]: cc.PLATE_PALETTE[p] for p in PLATES},
                "jitter_seed": cc.RANDOM_SEED, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
