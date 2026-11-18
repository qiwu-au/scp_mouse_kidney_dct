#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S-11-a - DCT marker proteins and their transcripts in the Mouse Kidney
Atlas, and the correlation of group means, control and zero-K.

Editors' comment 1 (no scRNA-seq comparator for the DCT
cells); plan approved by the authors on 12 Nov 2026 ("OK as proposed"). One reader-facing figure with panel letters,
redrawn from the stored dct_labels and dct_rna tables. No new statistics. The dct_rna analysis figures (dct_rna_dotplot_panel_*,
dct_rna_pseudobulk_rho_*) carry internal column names and footers and are not used.

(A) Marker dot plot, same 13 columns in every block (the panel markers of Figure 3B):
    protein, control cells by the cluster annotation (cs_ctrl_deg_label: DCT1, DCT2, Unknown shown as "DCT, subtype
    not resolved"); protein, zero-K cells (cs_zerok_pipeline_label: DCT1, Unknown); RNA, atlas classes DCT, DCT-CNT,
    CNT, CTAL, MTAL (atlas level Celltype, all cells and nuclei). Values: dct_labels_dotplot_values.csv and
    dct_rna_dotplot_rna_values.csv. Colour is scaled 0-1 per marker across the rows of each block.
(B) Spearman correlation across genes between the mean RNA level of each atlas class (rows) and the mean protein
    level of each SCP-MS group (columns), per condition: top, group means; bottom, z-scores of each gene across the
    groups within its data set. Values: dct_rna_pseudobulk_rho.csv (level 'published', rna_source 'all', primary
    label column, scp_set 'with_unknown', threshold 0.10). Control and zero-K are side by side; nothing compares them.

dotplot / dct_labels format: Arial >= 7 pt, exact 180 mm width, PDF + 600-dpi PNG + TIFF; no title, footer or internal names.

Inputs (read-only): results/tables/dct_labels/dct_labels_{dotplot_values,marker_order_dct}.csv;
  results/tables/dct_rna/dct_rna_{dotplot_rna_values,pseudobulk_rho}.csv
Outputs: results/figures/report/report_11_S11a_markers_correlation.{pdf,png,tif};
  results/tables/report/report_11_s11a_figure_{values,checks}.csv;
  results/tables/report/report_step_11_s11a_figure_manifest.json

Command (project root):  python scripts/report_dct_rna_markers_figure.py
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
STEP = "11_s11a_figure"
DCT_LABELS = cc.OUT_ROOTS["tables"] / "dct_labels"
DCT_RNA = cc.OUT_ROOTS["tables"] / "dct_rna"
PRIMARY = {"Ctrl": "cs_ctrl_deg_label", "ZeroK": "cs_zerok_pipeline_label"}
ROWS = {"Ctrl": ["DCT1", "DCT2", "Unknown"], "ZeroK": ["DCT1", "Unknown"]}
EXPECTED = {"Ctrl": {"DCT1": 382, "DCT2": 85, "Unknown": 262}, "ZeroK": {"DCT1": 449, "Unknown": 253}}
N_GENES = {"Ctrl": 2022, "ZeroK": 1791}
W = 180.0
PITCH = 4.2
# Numbers of the the response to the reviewers text (section 2): rho of the group means with the atlas DCT class.
LETTER_RHO = {("Ctrl", "raw", "DCT", "DCT1"): 0.50, ("Ctrl", "raw", "DCT", "DCT2"): 0.47,
              ("Ctrl", "raw", "DCT", "Unknown"): 0.46, ("ZeroK", "raw", "DCT", "DCT1"): 0.50,
              ("ZeroK", "raw", "DCT", "Unknown"): 0.46, ("Ctrl", "z", "DCT", "DCT1"): 0.37,
              ("Ctrl", "z", "DCT", "DCT2"): -0.00, ("Ctrl", "z", "DCT", "Unknown"): -0.35,
              ("ZeroK", "z", "DCT", "DCT1"): 0.18, ("ZeroK", "z", "DCT", "Unknown"): -0.16}


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-11-a.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")

    order = pd.read_csv(run.use_input(DCT_LABELS / "dct_labels_marker_order_dct.csv", "dct_labels marker order (DCT)"))
    order = order[(order["version"] == "panel") & order["plotted"]].sort_values("plotted_position")
    markers = order["marker"].tolist()
    groups = L.marker_groups(order["group"].tolist())
    checks.add("columns: the 13 panel markers of Figure 3B, plotted order", ", ".join(markers), len(markers) == 13)
    pv = pd.read_csv(run.use_input(DCT_LABELS / "dct_labels_dotplot_values.csv", "dct_labels dotplot values (protein)"))
    rv = pd.read_csv(run.use_input(DCT_RNA / "dct_rna_dotplot_rna_values.csv", "dct_rna RNA dotplot values"))
    rho = pd.read_csv(run.use_input(DCT_RNA / "dct_rna_pseudobulk_rho.csv", "dct_rna pseudobulk Spearman rho"))

    # --- panel A: dot plots -------------------------------------------------------------------------------------
    lib.style()
    t6d.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6d.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    panels, values = [], []
    for cond, rows in ROWS.items():
        s = pv[(pv["condition"] == cond) & (pv["label_column"] == PRIMARY[cond]) & pv["marker"].isin(markers)]
        n = s.drop_duplicates("group").set_index("group")["n_cells"].to_dict()
        checks.add(f"{cond}: rows and cells of the cluster annotation; values per row x marker",
                   "; ".join(f"{g} {n.get(g)}" for g in rows) + f"; {len(s)} values",
                   n == EXPECTED[cond] and len(s) == len(rows) * 13, str(EXPECTED[cond]))
        frac = s.pivot(index="group", columns="marker", values="frac_detected").reindex(index=rows, columns=markers)
        colr = s.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=rows, columns=markers)
        shown = [L.UNRESOLVED if g == "Unknown" else g for g in rows]
        panels.append(L.FixedPanel(f"Protein (SCP-MS), {'control' if cond == 'Ctrl' else 'zero-K'} cells", shown,
                                   [n[g] for g in rows], [L.LABEL_COLOURS[g] for g in rows], markers, groups,
                                   frac.to_numpy(float), colr.to_numpy(float), [False] * 13))
        for i, g in enumerate(rows):
            for j, mk in enumerate(markers):
                values.append({"panel": "A", "block": f"protein, {L.COND_DISPLAY[cond]}", "row": g,
                               "row_display": shown[i], "n_cells_row": n[g], "column": mk,
                               "frac_detected": frac.iloc[i, j], "colour_scaled_0_1": colr.iloc[i, j],
                               "source": "dct_labels_dotplot_values.csv"})
    rna = L.rna_panel_values(rv, markers, "published", "all", L.ATLAS_ROWS)
    checks.add("RNA rows: atlas classes and cells (level Celltype, all cells and nuclei); grey column",
               "; ".join(f"{r} {n:,}" for r, n in zip(L.ATLAS_ROWS, rna["n"])) + "; grey: "
               + ", ".join(m for m, g in zip(markers, rna["grey_cols"]) if g),
               rna["n"] == [16381, 810, 3903, 3600, 3840] and sum(rna["grey_cols"]) == 1)
    panels.append(L.FixedPanel(L.RNA_TITLE["all"], list(L.ATLAS_ROWS), rna["n"], rna["colours"], markers, groups,
                               rna["frac"].to_numpy(float), rna["colour"].to_numpy(float), [False] * 13,
                               grey_rows=[False] * 5, grey_cols=rna["grey_cols"], n_text=[f"{n:,}" for n in rna["n"]]))
    for i, r in enumerate(L.ATLAS_ROWS):
        for j, mk in enumerate(markers):
            values.append({"panel": "A", "block": "RNA, Mouse Kidney Atlas", "row": r, "row_display": r,
                           "n_cells_row": rna["n"][i], "column": mk, "frac_detected": rna["frac"].iloc[i, j],
                           "colour_scaled_0_1": rna["colour"].iloc[i, j], "grey": bool(rna["grey_cols"][j]),
                           "source": "dct_rna_dotplot_rna_values.csv"})

    m, vgap = t6d.MARGIN, 2.5
    body_w, body_h = L.stack_size(panels, PITCH, vgap)
    sw, sh, _ = t6d.size_legend_dims(PITCH)
    cw_leg, ch_leg = t6d.color_legend_dims()
    ax0 = m + 5.0                                        # left edge of panel A (after the panel letter)
    a_h = body_h + 2.5 + max(sh, ch_leg)

    # --- panel B: correlation heat maps (geometry) ---------------------------------------------------------------
    cell_w, cell_h, block_gap, label_w = 8.6, 5.2, 2.5, 17.0
    bx = ax0 + body_w + 9.0                              # left edge of panel B
    grid_x = bx + label_w
    n_cols = sum(len(r) for r in ROWS.values())
    b_w = label_w + n_cols * cell_w + block_gap
    head_h = 4 * t6d.LINE_H * 1.15 + 1.0                 # condition, label (two lines), cell number
    title_h = t6d.LINE_H * 1.15 * 2 + 1.5
    heat_h = title_h + head_h + 5 * cell_h
    b_h = 2 * heat_h + 5.0 + 9.0                         # two heat maps, gap, colour bar with ticks
    if bx + b_w > W - m:
        raise SystemExit(f"panel B does not fit: right edge {bx + b_w:.1f} mm")
    cv = t6d.Canvas(W, 2 * m + max(a_h, b_h))
    L.panel_letter(cv, m, m, "A")
    geo = L.draw_stack(cv, panels, ax0, m, PITCH, vgap)
    t6d.draw_legends(cv, ax0, geo[-1]["bottom"] + 2.5, PITCH, stacked=False)

    L.panel_letter(cv, bx - 4.5, m, "B")
    q = rho[(rho["level"] == "published") & (rho["rna_source"] == "all") & (rho["label_role"] == "primary")
            & (rho["scp_set"] == "with_unknown") & np.isclose(rho["threshold"], 0.10) & (rho["status"] == "computed")]
    genes = q.groupby("condition")["n_genes"].agg(["min", "max"])
    checks.add("panel B: genes compared per condition (detected in >= 10% of at least one group in each data set)",
               "; ".join(f"{c} {int(genes.loc[c, 'min'])}" for c in ROWS),
               all(int(genes.loc[c, "min"]) == int(genes.loc[c, "max"]) == N_GENES[c] for c in ROWS), str(N_GENES))
    cmap = t8.diverging_cmap()
    titles = {"raw": ("Spearman ρ across genes,", "mean level per group"),
              "z": ("Spearman ρ across genes, level relative", "to the other groups (z-score per gene)")}
    y = m
    worst = 0.0
    for vt in ("raw", "z"):
        for k, line in enumerate(titles[vt]):
            cv.text(bx, y + k * t6d.LINE_H * 1.15, line, t6d.TITLE_FONT if k == 0 else t6d.FONT, ha="left", va="top",
                    fontweight="bold" if k == 0 else "normal")
        gy = y + title_h + head_h
        x = grid_x
        for cond, rows in ROWS.items():
            cx = x + len(rows) * cell_w / 2
            cv.text(cx, y + title_h, L.COND_DISPLAY[cond], ha="center", va="top", fontweight="bold")
            mat = np.full((5, len(rows)), np.nan)
            for j, g in enumerate(rows):
                lines = ("not", "resolved") if g == "Unknown" else (g, "")
                for k, text in enumerate((*lines, f"({EXPECTED[cond][g]})")):
                    if text:
                        cv.text(x + (j + 0.5) * cell_w, y + title_h + (k + 1) * t6d.LINE_H * 1.15, text, ha="center",
                                va="top")
                for i, r in enumerate(L.ATLAS_ROWS):
                    sel = q[(q["condition"] == cond) & (q["value_type"] == vt) & (q["rna_group"] == r)
                            & (q["scp_group"] == g)]
                    if len(sel) != 1:
                        raise SystemExit(f"rho row missing or repeated: {cond} {vt} {r} {g}")
                    mat[i, j] = float(sel["rho"].iloc[0])
                    values.append({"panel": "B", "block": f"{vt}, {L.COND_DISPLAY[cond]}", "row": r, "row_display": r,
                                   "n_cells_row": int(sel["n_rna_cells"].iloc[0]), "column": g,
                                   "column_display": L.UNRESOLVED_SHORT if g == "Unknown" else g,
                                   "n_cells_column": int(sel["n_scp_cells"].iloc[0]), "rho": mat[i, j],
                                   "n_genes": int(sel["n_genes"].iloc[0]), "source": "dct_rna_pseudobulk_rho.csv"})
                    key = (cond, vt, r, g)
                    if key in LETTER_RHO:
                        worst = max(worst, abs(round(mat[i, j], 2) - LETTER_RHO[key]))
            L.heat_grid(cv, x, gy, cell_w, cell_h, mat, cmap, -1.0, 1.0,
                        [[f"{v:.2f}".replace("-0.00", "0.00").replace("-", "−") for v in row] for row in mat])
            x += len(rows) * cell_w + block_gap
        for i, r in enumerate(L.ATLAS_ROWS):
            yc = gy + (i + 0.5) * cell_h
            cv.rect(grid_x - 1.0 - t6d.CHIP_W, yc - 0.8, t6d.CHIP_W, 1.6, facecolor=t8.RNA_PALETTE[r], edgecolor="none")
            cv.text(grid_x - 1.0 - t6d.CHIP_W - 0.8, yc, r, ha="right", va="center_baseline")
        y = gy + 5 * cell_h + 5.0
    L.colour_bar(cv, grid_x, y - 1.5, 26.0, 2.2, cmap, (0, 0.5, 1), ("−1", "0", "1"), "Spearman ρ")
    checks.add("panel B: rho of the the response to the reviewers text (section 2) = the plotted values at two decimals",
               f"{len(LETTER_RHO)} values; largest difference {worst:.3f}", worst < 0.005, "< 0.005")

    bad = L.banned_texts(t27f.drawn_texts(cv.fig))
    checks.add("reader-facing text: no internal names in the figure", f"{len(bad)} texts with an internal name"
               + (f": {bad}" if bad else ""), not bad)
    figs = lib.Figures(run, STEP)
    row = figs.save(cv.fig, "11_S11a_markers_correlation",
                    "S-11-a: (A) DCT marker proteins by cluster annotation, control and zero-K, above their "
                    "transcripts in the Mouse Kidney Atlas classes; (B) Spearman correlation of atlas class means "
                    "with SCP-MS group means, raw and z-scored, per condition", lib.FULL_MM)
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in PDF)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps "
               f"{row['n_text_overlaps']} {row['overlaps'][:150]}; outside {row['n_text_outside_figure']} "
               f"{row['outside'][:100]}", bool(row["pass"]))
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"label_columns": PRIMARY, "label_display": {"Unknown": L.UNRESOLVED}, "markers": markers,
                "rna": "atlas level Celltype, all cells and nuclei", "rho_filter": "level published; rna_source all; "
                "primary label column; scp_set with_unknown; threshold 0.10", "n_genes": N_GENES,
                "width_mm": W, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
