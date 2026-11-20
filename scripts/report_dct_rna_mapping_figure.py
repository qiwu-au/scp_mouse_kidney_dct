#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S22 - nearest Mouse Kidney Atlas class of each DCT cell, by cluster
label and by protein depth, and cells with cycling markers in the atlas and in the SCP-MS data.

Editors' comment 1 (no scRNA-seq comparator for the DCT
cells); plan approved by the authors on 12 Nov 2026 ("OK as proposed"). One reader-facing figure with panel letters,
redrawn from the stored dct_rna tables. No new statistics; the 'all cells' rows of panel A are read from the stored
summary. The dct_rna analysis figures (dct_rna_mapping_confusion, dct_rna_cycling) carry internal names and
footers and are not used.

(A) Per SCP-MS group (cluster annotation; control and zero-K): percentage of cells whose protein profile is nearest
    to each atlas class (cosine similarity to class centroids on up to 50 informative genes per class), and the
    percentage of low-confidence cells. Source: dct_rna_mapping_summary.csv (level 'published', primary variant
    top50_equal_weight_z, primary label column).
(B) The same by quartile of proteins identified per cell within the condition. Source:
    dct_rna_mapping_plate_depth.csv (level 'published', by 'depth_bin').
(C) Fraction of cells with Mki67, Top2a or either detected: atlas classes (at least one count) and SCP-MS groups
    (intensity > 0), with Wilson 95% intervals. Source: dct_rna_cycling_side_by_side.csv.
Control and zero-K are shown one after the other; nothing compares them.

dotplot / dct_labels format: Arial >= 7 pt, exact 180 mm width, PDF + 600-dpi PNG + TIFF; no title, footer or internal names.

Inputs (read-only): results/tables/dct_rna/dct_rna_{mapping_summary,mapping_plate_depth,
  cycling_side_by_side}.csv
Outputs: results/figures/report/report_11_S11b_mapping_cycling.{pdf,png,tif};
  results/tables/report/report_11_s11b_figure_{values,checks}.csv;
  results/tables/report/report_step_11_s11b_figure_manifest.json

Command (project root):  python scripts/report_dct_rna_mapping_figure.py
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
import dotplot_dotplots as t6d
import atlas_dct_lib as t8
import dct_labels_figures as t27f
import dct_reviewer_lib as lib
import report_dct_rna_lib as L

TASK = "report"
STEP = "11_s11b_figure"
DCT_RNA = cc.OUT_ROOTS["tables"] / "dct_rna"
CONDS = ("Ctrl", "ZeroK")
LABEL_ROWS = {"Ctrl": ["DCT1", "DCT2", "Unknown", "all"], "ZeroK": ["DCT1", "Unknown", "all"]}
SHOWN = {"DCT1": "DCT1", "DCT2": "DCT2", "Unknown": L.UNRESOLVED_SHORT, "all": "all cells"}
N_CELLS = {"Ctrl": {"DCT1": 382, "DCT2": 85, "Unknown": 262, "all": 729},
           "ZeroK": {"DCT1": 449, "Unknown": 253, "all": 702}}
QUARTILES = ("Q1 (lowest)", "Q2", "Q3", "Q4 (highest)")
CLASS_COLS = [f"frac_top_{c}" for c in L.ATLAS_ROWS]
MEASURES = (("Mki67", "Mki67", "o", "#2a78d6"), ("Top2a", "Top2a", "s", "#eb6834"),
            ("Mki67_or_Top2a", "Mki67 or Top2a", "D", "#1baf7a"))
RNA_POPS = (("DCT", "Atlas DCT, all"), ("DCT, Nuclei", "Atlas DCT, nuclei"), ("DCT, Cell", "Atlas DCT, cells"),
            ("DCT-CNT", "Atlas DCT-CNT"), ("CNT", "Atlas CNT"), ("CTAL", "Atlas CTAL"), ("MTAL", "Atlas MTAL"))
SCP_POPS = (("all cells", "all cells"), ("DCT1 (primary)", "DCT1"), ("DCT2 (primary)", "DCT2"),
            ("Unknown (primary)", L.UNRESOLVED_SHORT))
W = 180.0
X_MIN = 0.001
# Numbers of the the response to the reviewers text (section 2).
LETTER = {("A", "Ctrl", "all", "DCT"): 40, ("A", "ZeroK", "all", "DCT"): 32, ("A", "Ctrl", "DCT1", "DCT"): 47,
          ("A", "Ctrl", "DCT2", "DCT"): 18, ("A", "ZeroK", "DCT1", "DCT"): 36, ("B", "Ctrl", "Q1 (lowest)", "CNT"): 70,
          ("B", "Ctrl", "Q4 (highest)", "DCT-CNT"): 57, ("B", "Ctrl", "Q4 (highest)", "CTAL"): 42,
          ("B", "ZeroK", "Q1 (lowest)", "CNT"): 82, ("B", "ZeroK", "Q4 (highest)", "DCT-CNT"): 74}
LETTER_K = {("RNA", "DCT", "Mki67_or_Top2a"): (217, 16381), ("RNA", "DCT", "Mki67"): (118, 16381),
            ("RNA", "DCT", "Top2a"): (113, 16381), ("Ctrl", "all cells", "Mki67_or_Top2a"): (21, 729),
            ("Ctrl", "all cells", "Mki67"): (15, 729), ("Ctrl", "all cells", "Top2a"): (16, 729),
            ("ZeroK", "all cells", "Mki67_or_Top2a"): (56, 702), ("ZeroK", "all cells", "Mki67"): (52, 702),
            ("ZeroK", "all cells", "Top2a"): (35, 702)}


def heat_block(cv, x0, y0, letter, title_lines, row_groups, mats, lows, cell_w, cell_h, label_w, cmap):
    """One heat map: condition header rows, atlas class columns, a text column with the low-confidence share."""
    L.panel_letter(cv, x0, y0, letter)
    tx = x0 + 5.0
    for k, line in enumerate(title_lines):
        cv.text(tx, y0 + k * t6d.LINE_H * 1.15, line, t6d.TITLE_FONT if k == 0 else t6d.FONT, ha="left", va="top",
                fontweight="bold" if k == 0 else "normal")
    gx = tx + label_w
    y = y0 + len(title_lines) * t6d.LINE_H * 1.15 + 1.5
    head_h = t6d.text_mm("DCT-CNT")[0] + 1.0
    for j, c in enumerate(L.ATLAS_ROWS):
        cv.text(gx + (j + 0.5) * cell_w, y + head_h - 0.6, c, rotation=90, ha="center", va="bottom")
    lx = gx + 5 * cell_w + 2.0
    for k, text in enumerate(("low", "confidence", "(%)")):
        cv.text(lx, y + head_h - 0.6 - (2 - k) * t6d.LINE_H * 1.15, text, ha="left", va="bottom")
    y += head_h
    for (cond, labels), mat, low in zip(row_groups, mats, lows):
        cv.text(tx, y + 0.4, L.COND_DISPLAY[cond], ha="left", va="top", fontweight="bold")
        y += t6d.LINE_H * 1.15 + 0.8
        L.heat_grid(cv, gx, y, cell_w, cell_h, mat, cmap, 0.0, 1.0, [[f"{100 * v:.0f}" for v in row] for row in mat])
        for i, lab in enumerate(labels):
            yc = y + (i + 0.5) * cell_h
            cv.text(gx - 1.2, yc, lab, ha="right", va="center_baseline")
            cv.text(lx + 3.0, yc, f"{100 * low[i]:.0f}", ha="center", va="center_baseline")
        y += len(labels) * cell_h + 1.5
    return y, gx


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-11-b.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    ms = pd.read_csv(run.use_input(DCT_RNA / "dct_rna_mapping_summary.csv", "dct_rna mapping summary"))
    pdp = pd.read_csv(run.use_input(DCT_RNA / "dct_rna_mapping_plate_depth.csv", "dct_rna mapping by plate and depth"))
    cyc = pd.read_csv(run.use_input(DCT_RNA / "dct_rna_cycling_side_by_side.csv", "dct_rna cycling, RNA and SCP"))
    lib.style()
    values = []

    # --- panels A and B: values -----------------------------------------------------------------------------------
    ms = ms[(ms["level"] == "published") & ms["primary_variant"].astype(bool) & (ms["label_role"] == "primary")
            & (ms["status"] == "computed")]
    checks.add("panel A: mapping variant (top 50 informative genes per atlas class, equal weights, z-scored)",
               "; ".join(sorted(ms["variant"].unique())), set(ms["variant"]) == {"top50_equal_weight_z"})
    a_groups, a_mats, a_lows, worst = [], [], [], 0.0
    for cond in CONDS:
        s = ms[ms["condition"] == cond].set_index("scp_label").reindex(LABEL_ROWS[cond])
        n = s["n_cells"].astype(int).to_dict()
        checks.add(f"panel A, {cond}: cells per row", "; ".join(f"{k} {v}" for k, v in n.items()), n == N_CELLS[cond])
        mat = s[CLASS_COLS].to_numpy(float)
        checks.add(f"panel A, {cond}: the five class fractions of each row sum to 1",
                   f"largest deviation {float(np.abs(mat.sum(axis=1) - 1).max()):.1e}",
                   bool(np.allclose(mat.sum(axis=1), 1, atol=1e-9)))
        a_groups.append((cond, [f"{SHOWN[g]} ({n[g]})" for g in LABEL_ROWS[cond]]))
        a_mats.append(mat)
        a_lows.append(s["frac_low_confidence"].to_numpy(float))
        for i, g in enumerate(LABEL_ROWS[cond]):
            for j, c in enumerate(L.ATLAS_ROWS):
                values.append({"panel": "A", "condition": L.COND_DISPLAY[cond], "row": g, "row_display": SHOWN[g],
                               "n_cells": n[g], "measure": f"nearest atlas class {c}", "fraction": mat[i, j],
                               "source": "dct_rna_mapping_summary.csv"})
                if ("A", cond, g, c) in LETTER:
                    worst = max(worst, abs(round(100 * mat[i, j]) - LETTER[("A", cond, g, c)]))
            values.append({"panel": "A", "condition": L.COND_DISPLAY[cond], "row": g, "row_display": SHOWN[g],
                           "n_cells": n[g], "measure": "low confidence", "fraction": a_lows[-1][i],
                           "source": "dct_rna_mapping_summary.csv"})
    dp = pdp[(pdp["level"] == "published") & (pdp["by"] == "depth_bin")]
    b_groups, b_mats, b_lows = [], [], []
    for cond in CONDS:
        s = dp[dp["condition"] == cond].set_index("value").reindex(QUARTILES)
        checks.add(f"panel B, {cond}: cells and median proteins per quartile",
                   "; ".join(f"{q.split(' ')[0]} {int(r.n_cells)} / {r.median_protein_ids:g}" for q, r in s.iterrows()),
                   int(s["n_cells"].sum()) == N_CELLS[cond]["all"])
        mat = s[CLASS_COLS].to_numpy(float)
        b_groups.append((cond, [f"{q.split(' ')[0]} ({int(r.n_cells)}; {r.median_protein_ids:,.0f})"
                                for q, r in s.iterrows()]))
        b_mats.append(mat)
        b_lows.append(s["frac_low_confidence"].to_numpy(float))
        for i, qn in enumerate(QUARTILES):
            for j, c in enumerate(L.ATLAS_ROWS):
                values.append({"panel": "B", "condition": L.COND_DISPLAY[cond], "row": qn, "row_display": qn,
                               "n_cells": int(s["n_cells"].iloc[i]), "median_protein_ids": s["median_protein_ids"].iloc[i],
                               "measure": f"nearest atlas class {c}", "fraction": mat[i, j],
                               "source": "dct_rna_mapping_plate_depth.csv"})
                if ("B", cond, qn, c) in LETTER:
                    worst = max(worst, abs(round(100 * mat[i, j]) - LETTER[("B", cond, qn, c)]))
            values.append({"panel": "B", "condition": L.COND_DISPLAY[cond], "row": qn, "row_display": qn,
                           "n_cells": int(s["n_cells"].iloc[i]), "measure": "low confidence", "fraction": b_lows[-1][i],
                           "source": "dct_rna_mapping_plate_depth.csv"})
    checks.add("panels A and B: percentages of the the response to the reviewers text (section 2) = the plotted values",
               f"{len(LETTER)} values; largest difference {worst:g}", worst == 0)

    # --- panel C: values --------------------------------------------------------------------------------------------
    cyc = cyc[cyc["status"] == "computed"]
    c_rows = []                                          # (group header or None, label, {measure: (k, n, f, lo, hi)})
    rna = cyc[cyc["modality"].str.startswith("RNA")]
    for pop, shown in RNA_POPS:
        s = rna[rna["population"] == pop].set_index("measure")
        c_rows.append(("RNA", pop, f"{shown} ({int(s['n_cells'].iloc[0]):,})", s))
    for cond in CONDS:
        scp = cyc[(cyc["condition"] == cond) & cyc["modality"].str.startswith("SCP")]
        for pop, shown in SCP_POPS:
            s = scp[scp["population"] == pop].set_index("measure")
            if s.empty:
                continue                                 # no DCT2 group in zero-K
            c_rows.append((cond, pop, f"{L.COND_DISPLAY[cond]}, {shown} ({int(s['n_cells'].iloc[0])})", s))
    checks.add("panel C: rows (7 atlas rows, 4 control rows, 3 zero-K rows)", f"{len(c_rows)} rows", len(c_rows) == 14)
    bad_k, bad_ci = [], 0.0
    for src, pop, shown, s in c_rows:
        for key, name, _mk, _col in MEASURES:
            r = s.loc[key]
            lo, hi = L.wilson(float(r["k"]), float(r["n_cells"]))
            bad_ci = max(bad_ci, abs(lo - float(r["ci_low"])), abs(hi - float(r["ci_high"])))
            values.append({"panel": "C", "condition": {"RNA": "atlas"}.get(src, L.COND_DISPLAY.get(src, src)),
                           "row": pop, "row_display": shown, "n_cells": int(r["n_cells"]), "measure": name,
                           "k": int(r["k"]), "fraction": float(r["fraction"]), "ci_low": float(r["ci_low"]),
                           "ci_high": float(r["ci_high"]), "source": "dct_rna_cycling_side_by_side.csv"})
            exp = LETTER_K.get((src, pop, key))
            if exp and (int(r["k"]), int(r["n_cells"])) != exp:
                bad_k.append(f"{src} {pop} {key}: {int(r['k'])}/{int(r['n_cells'])}")
    checks.add("panel C: counts of the the response to the reviewers text (section 2) = the stored counts", bad_k or f"{len(LETTER_K)} "
               "counts equal", not bad_k)
    checks.add("panel C: stored intervals are Wilson 95% intervals of the stored counts (largest difference)",
               f"{bad_ci:.1e}", bad_ci < 1e-6, "< 1e-6")

    # --- figure -----------------------------------------------------------------------------------------------------
    m = t6d.MARGIN
    cmap = t8.sequential_cmap()
    cell_w, cell_h = 8.2, 4.8
    a_label = max(t6d.text_mm(s)[0] for _, labs in a_groups for s in labs) + 2.0
    b_label = max(t6d.text_mm(s)[0] for _, labs in b_groups for s in labs) + 2.0
    a_w = 5.0 + a_label + 5 * cell_w + 2.0 + t6d.text_mm("confidence")[0]
    bx = m + a_w + 8.0
    b_w = 5.0 + b_label + 5 * cell_w + 2.0 + t6d.text_mm("confidence")[0]
    if bx + b_w > W - m:
        raise SystemExit(f"panel B does not fit: right edge {bx + b_w:.1f} mm")
    n_title = 3
    top_h = (n_title * t6d.LINE_H * 1.15 + 1.5 + t6d.text_mm("DCT-CNT")[0] + 1.0
             + 2 * (t6d.LINE_H * 1.15 + 0.8 + 1.5) + 8 * cell_h)
    row_h, c_top = 4.3, m + top_h + 12.0
    c_h = len(c_rows) * row_h
    cv = t6d.Canvas(W, c_top + c_h + 11.0 + m)
    ya, _ = heat_block(cv, m, m, "A", ("Nearest atlas class of each cell,", "by cluster label (cells);",
                                         "% of the cells of the row"),
                       a_groups, a_mats, a_lows, cell_w, cell_h, a_label, cmap)
    yb, gxb = heat_block(cv, bx, m, "B", ("By quartile of proteins per cell", "(cells; median proteins per cell);",
                                           "% of the cells of the row"),
                         b_groups, b_mats, b_lows, cell_w, cell_h, b_label, cmap)
    L.colour_bar(cv, m + 5.0 + a_label, ya + 1.0, 5 * cell_w, 2.2, cmap, (0, 0.5, 1), ("0", "50", "100"), "% of the row")

    L.panel_letter(cv, m, c_top - 6.5, "C")
    cv.text(m + 5.0, c_top - 6.5, "Cells with cycling markers: transcripts in the atlas, proteins in the SCP-MS data",
            t6d.TITLE_FONT, ha="left", va="top", fontweight="bold")
    label_w = max(t6d.text_mm(s)[0] for _, _, s, _ in c_rows) + 2.0
    ax_x, ax_w = m + 5.0 + label_w, 88.0
    ax = cv.fig.add_axes([ax_x / W, 1 - (c_top + c_h) / cv.H, ax_w / W, c_h / cv.H])
    ax.set_xscale("log")
    ax.set_xlim(X_MIN, 0.5)
    ax.set_ylim(len(c_rows) - 0.5, -0.5)
    ax.set_yticks(range(len(c_rows)), [s for _, _, s, _ in c_rows])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.set_xticks([0.001, 0.01, 0.1], ["0.1", "1", "10"])
    ax.set_xlabel("Cells with the marker detected (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    for edge in (6.5, 10.5):
        ax.axhline(edge, color="#898781", linewidth=0.5)
    for i, (_src, _pop, _shown, s) in enumerate(c_rows):
        for d, (key, _name, marker, colour) in zip((-0.27, 0.0, 0.27), MEASURES):
            r = s.loc[key]
            f = float(r["fraction"])
            ax.plot([max(float(r["ci_low"]), X_MIN), float(r["ci_high"])], [i + d, i + d], color=colour, linewidth=0.8,
                    solid_capstyle="butt", zorder=2)
            ax.plot([max(f, X_MIN)], [i + d], marker=marker, markersize=3.2, linestyle="none", color=colour,
                    markerfacecolor=colour if f > 0 else "white", markeredgewidth=0.6, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=mk, linestyle="-", linewidth=0.8, color=col, markersize=3.2, label=name)
               for _key, name, mk, col in MEASURES]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.0), frameon=False, fontsize=7,
              handlelength=1.8, handletextpad=0.6, borderaxespad=0, labelspacing=0.5,
              title="Symbol, fraction;\nline, Wilson 95% interval", title_fontsize=7, alignment="left")

    bad = L.banned_texts(t27f.drawn_texts(cv.fig))
    checks.add("reader-facing text: no internal names in the figure", f"{len(bad)} texts with an internal name"
               + (f": {bad}" if bad else ""), not bad)
    figs = lib.Figures(run, STEP)
    row = figs.save(cv.fig, "11_S11b_mapping_cycling",
                    "S-11-b: (A) nearest atlas class per DCT cell by cluster label, control and zero-K; (B) by "
                    "quartile of proteins per cell; (C) cells with Mki67 or Top2a in the atlas (transcripts) and in "
                    "the SCP-MS data (proteins), with Wilson 95% intervals", lib.FULL_MM)
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in PDF)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps "
               f"{row['n_text_overlaps']} {row['overlaps'][:200]}; outside {row['n_text_outside_figure']} "
               f"{row['outside'][:100]}", bool(row["pass"]))
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"mapping": "level published (atlas Celltype classes); variant top50_equal_weight_z; primary label "
                           "column; low confidence = score < 0.20 or margin < 0.05",
                "depth": "quartiles of protein IDs per cell within the condition", "cycling_measures":
                [name for _k, name, _m, _c in MEASURES], "label_display": SHOWN, "width_mm": W,
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
