#!/usr/bin/env python
"""The response to the reviewers: Figure 4f with RNA rows - the DCT marker proteins of Figure 4f above their transcripts in
the Mouse Kidney Atlas.

Editors' comment 1 (no scRNA-seq comparator for the DCT
cells). Plan approved by the authors on 12 Nov 2026 ("OK as proposed"): Figure 4f gains RNA rows under the protein
rows, with the same 13 markers, as Figure 2B does for the kidney. This is a new the response to the reviewers file; the the response to the reviewers panel
(report_06_fig3b_dotplot_85mm) is not changed and stays the fallback.

No new statistics: both halves are drawn from stored values.
- Protein (top): the four groups of Figure 4e (control cells of the combined search) and the 13 panel markers, read
  from results/tables/report/report_06_fig3b_dotplot_values.csv. The row of the dct_labels label value
  'Unknown' is shown as "DCT, subtype not resolved" (authors, 12 Nov 2026); the values keep 'Unknown'.
- RNA (bottom): the atlas classes DCT, DCT-CNT, CNT, CTAL and MTAL (atlas level Celltype; all cells and nuclei), the
  same 13 columns, read from results/tables/dct_rna/dct_rna_dotplot_rna_values.csv (level 'published',
  rna_source 'all'). Trpv5 is not among the atlas genes (grey column).
Encoding as Figure 2B (kidney_figure2_fig2b_full.py): dot area = fraction of cells with intensity > 0 (protein) or at
least one count (RNA); colour = group mean scaled 0-1 per marker across the rows of its own half. Arial >= 7 pt,
exact 85 mm width (Figure 4e is 85 mm), PDF + 600-dpi PNG + LZW TIFF, no panel letter, no footer.

Inputs (read-only): the two value tables above.
Outputs: results/figures/report/report_11_fig3b_protein_rna_85mm.{pdf,png,tif};
  results/tables/report/report_11_fig3b_protein_rna_{values,checks}.csv;
  results/tables/report/report_step_11_fig3b_protein_rna_manifest.json

Command (project root):  python scripts/report_dct_dotplot_protein_rna.py
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
import dotplot_dotplots as t6d   # sets the dotplot figure style (Arial) on import
from report_dct_rna_lib import (ATLAS_ROWS, BANNED, RNA_TITLE, FixedPanel, UNRESOLVED, banned_texts, marker_groups,
                               rna_panel_values, stacked_below)
from dct_labels_figures import FORMATS, arial

TASK = "report"
STEP = "11_fig3b_protein_rna"
PROTEIN_VALUES = cc.OUT_ROOTS["tables"] / TASK / "report_06_fig3b_dotplot_values.csv"
RNA_VALUES = cc.OUT_ROOTS["tables"] / "dct_rna" / "dct_rna_dotplot_rna_values.csv"
ROWS = ["DCT1", "DCT2", "ProLIF", "Unknown"]                        # Figure 4e groups
EXPECTED = {"DCT1": 362, "DCT2": 79, "ProLIF": 37, "Unknown": 251}
ROW_COLOURS = {"DCT1": "#1f77b4", "DCT2": "#ff7f0e", "ProLIF": "#2ca02c", "Unknown": "#B3B3B3"}   # Figure 4e
WIDTH = 85.0


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Figure 3B, protein above RNA.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_ctrl_combined")

    # --- protein half: the stored the response to the reviewers values ------------------------------------------------------------
    pv = pd.read_csv(run.use_input(PROTEIN_VALUES, "the response to the reviewers Figure 3B values (protein half)"))
    markers = pv.drop_duplicates("marker")["marker"].tolist()
    n_rows = pv.drop_duplicates("group").set_index("group")["n_cells"].to_dict()
    checks.add("protein half: rows, cells per row and markers of the the response to the reviewers Figure 3B",
               "; ".join(f"{g} {n_rows.get(g)}" for g in ROWS) + f"; {len(markers)} markers; {len(pv)} values",
               n_rows == EXPECTED and len(markers) == 13 and len(pv) == 52, "362, 79, 37, 251; 13 markers; 52 values")
    shown = pv.drop_duplicates("group").set_index("group")["row_display"].to_dict()
    checks.add("protein half: row names shown (the dct_labels value 'Unknown' is shown as 'DCT, subtype not resolved')",
               "; ".join(f"{g} -> {shown[g]}" for g in ROWS if shown[g] != g), shown["Unknown"] == UNRESOLVED)
    groups = marker_groups(pv.drop_duplicates("marker")["marker_group"].tolist())
    frac = pv.pivot(index="group", columns="marker", values="frac_detected").reindex(index=ROWS, columns=markers)
    colr = pv.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=ROWS, columns=markers)
    checks.add("protein half: colour scaled 0-1 per marker across the four rows (min 0, max 1 in every column)",
               f"min {float(colr.min().max()):.3g}; max {float(colr.max().min()):.3g}",
               bool(np.allclose(colr.min(), 0) and np.allclose(colr.max(), 1)))

    # --- RNA half: the stored dct_rna values ------------------------------------------------------------------------
    rv = pd.read_csv(run.use_input(RNA_VALUES, "dct_rna RNA dotplot values (RNA half)"))
    rna = rna_panel_values(rv, markers, "published", "all", ATLAS_ROWS)
    checks.add("RNA half: atlas classes and cells per row (level Celltype, all cells and nuclei)",
               "; ".join(f"{r} {n:,}" for r, n in zip(ATLAS_ROWS, rna["n"])),
               rna["n"] == [16381, 810, 3903, 3600, 3840], "16,381; 810; 3,903; 3,600; 3,840")
    grey = [m for m, g in zip(markers, rna["grey_cols"]) if g]
    checks.add("RNA half: markers that are not atlas genes (grey column)", ", ".join(grey) or "none", grey == ["Trpv5"],
               "Trpv5")
    for mk, row, expected in (("Slc12a3", "DCT", 0.84), ("Pvalb", "DCT", 0.45), ("Umod", "DCT", 0.48),
                              ("Slc12a1", "CTAL", 0.98), ("Mki67", "DCT", 0.01)):
        v = float(rna["frac"].loc[row, mk])
        checks.add(f"RNA half: fraction of atlas {row} cells with {mk} counts", f"{v:.4f}",
                   abs(v - expected) < 0.005, f"{expected:.2f}")

    # --- figure ---------------------------------------------------------------------------------------------------
    arial()
    t6d.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6d.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    prot = FixedPanel("Protein (SCP-MS)", [shown[g] for g in ROWS], [n_rows[g] for g in ROWS],
                      [ROW_COLOURS[g] for g in ROWS], markers, groups, frac.to_numpy(float), colr.to_numpy(float),
                      [False] * len(markers))
    rnap = FixedPanel(RNA_TITLE["all"], list(ATLAS_ROWS), rna["n"], rna["colours"], markers, groups,
                      rna["frac"].to_numpy(float), rna["colour"].to_numpy(float), [False] * len(markers),
                      grey_rows=[False] * len(ATLAS_ROWS), grey_cols=rna["grey_cols"],
                      n_text=[f"{n:,}" for n in rna["n"]])
    cv, pitch = stacked_below([prot, rnap], WIDTH)
    tc = t6d.text_checks(cv)
    bad = banned_texts(cv.texts, BANNED)
    checks.add("reader-facing text: no internal names in the figure", f"{tc['n_texts']} texts; {len(bad)} with an "
               f"internal name" + (f": {bad}" if bad else ""), not bad)
    paths = run.figure(cv.fig, f"{STEP}_85mm",
                       "Figure 3B with RNA rows: DCT marker proteins in the Figure 3A groups (control cells) above "
                       "their transcripts in the Mouse Kidney Atlas classes DCT, DCT-CNT, CNT, CTAL and MTAL",
                       formats=FORMATS, tight=False)
    fc = t6d.file_checks(paths, cv.W, cv.H)
    ok_fig = (tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
              and "arial" in tc["fonts"] and fc["width_ok"] and fc.get("pdf_truetype_embedded", False)
              and abs(cv.W - WIDTH) < 0.05)
    fig_row = {"figure": f"report_{STEP}_85mm", "width_mm": round(cv.W, 2), "height_mm": round(cv.H, 2),
               "pitch_mm": round(pitch, 3), **tc, **fc, "pass": ok_fig}
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 85 mm, TrueType in PDF)",
               f"{fig_row['width_mm']} x {fig_row['height_mm']} mm; pitch {pitch:.2f} mm; min font "
               f"{tc['min_font_pt']} pt; overlaps {tc['n_text_overlaps']} {tc['overlaps'][:120]}; outside "
               f"{tc['n_text_outside_figure']} {tc['outside'][:120]}", ok_fig)

    out = []
    for half, panel, rows, n in (("protein", prot, ROWS, prot.row_n), ("RNA", rnap, ATLAS_ROWS, rna["n"])):
        for i, r in enumerate(rows):
            for j, mk in enumerate(markers):
                out.append({"half": half, "row": r, "row_display": panel.rows[i], "n_cells_row": int(n[i]),
                            "marker": mk, "marker_group": next(g for g, a, b in groups if a <= j <= b),
                            "frac_detected": panel.frac[i, j], "colour_scaled_0_1": panel.color[i, j],
                            "grey": half == "RNA" and bool(rna["grey_cols"][j])})
    run.table(pd.DataFrame(out), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame([fig_row])], ignore_index=True), f"{STEP}_checks")
    run.finish({"request": "authors, 12 Nov 2026: Figure 3B gains RNA rows",
                "protein_rows": ROWS, "protein_row_display": shown, "rna_rows": list(ATLAS_ROWS),
                "rna_level": "atlas Celltype, all cells and nuclei", "markers": markers, "width_mm": WIDTH,
                "fallback": "results/figures/report/report_06_fig3b_dotplot_85mm",
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
