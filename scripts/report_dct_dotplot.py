#!/usr/bin/env python
"""The response to the reviewers: revised Figure 4f, DCT marker dotplot with the groups of the revised Figure 4e.

Drawn at the authors' request (10 Nov 2026, the response to the reviewers: "Draw 3B
with 3A groups") so that Figure 4f shows the same cell groups as the revised Figure 4e, which the authors chose on 10 Nov
2026: results/figures/dct_prolif/dct_prolif_prolif_overlay_nofloor_ctrl_85mm.

Rows (the Figure 4e groups, control cells of the combined search, 729 cells):
  each cell's dct_labels primary label (cs_ctrl_deg_label: DCT1, DCT2, Unknown), except the 37 cells that the dct_prolif route-A
  per-cell call of the no-floor run (cs_t31nf_label) labels ProLIF, which form their own row. The groups are read from
  results/tables/dct_prolif/dct_prolif_prolif_overlay_per_cell.csv (run 'nofloor', condition 'Ctrl', column
  overlay_group); nothing is refitted and no label changes. dct_prolif is not adopted: the dct_labels labels stay primary
  (the project notes). The row colours are those of Figure 4e (dct_prolif_lib.LABEL_COLORS).
Row display name (authors, 12 Nov 2026, via the response to the reviewers: "Yes, rename there"): the dct_labels label value 'Unknown' (cells of
  clusters without marker support) is shown as "DCT, subtype not resolved" (dct_prolif_lib.UNRESOLVED). Display only:
  the groups, the checks and the values file keep 'Unknown'; the values file adds the column 'row_display'.
Columns: the dct_labels panel markers in the dct_labels plotted order (dct_labels_marker_order_dct.csv, version 'panel'):
  Core Slc12a3 | DCT1 Pvalb, Egf, Ptgfr | DCT2 Calb1, Slc8a1, S100g, Klk1, Trpv5 | TAL Umod, Slc12a1 |
  Cycling Mki67, Top2a. Group names are marker groups, not cell labels.
Encoding: as the dct_labels dotplots (dct_labels_figures.py, dotplot format): dot area = fraction of the row's cells with
  intensity > 0 (layers['intensity']); colour = row mean of the scaled .X of the dct_labels labelled object (per-protein
  z-scored ln(1 + intensity), submitted pipeline), rescaled 0-1 per protein across the four rows; Arial >= 7 pt;
  exact 85 mm width (Figure 4e is 85 mm); PDF + 600-dpi PNG + LZW TIFF. No title, no panel letter, no footer (one
  plot per file, as the kidney_figure2 Figure 2 files; the authors assemble the figure).

Check before drawing: the same code reproduces the dct_labels dotplot values of cs_ctrl_deg_label (rows DCT1, DCT2, Unknown)
from the dct_labels labelled object.

Inputs (read-only): results/h5ad/dct_labels_DCT_ctrl_labelled.h5ad;
  results/tables/dct_labels/dct_labels_dct_canonical_labels.csv, dct_labels_marker_order_dct.csv,
  dct_labels_dotplot_values.csv;
  results/tables/dct_prolif/dct_prolif_prolif_overlay_per_cell.csv, dct_prolif_dct_labels.csv.
Outputs: results/figures/report/report_06_fig3b_dotplot_85mm.{pdf,png,tif};
  results/tables/report/report_06_fig3b_dotplot_{values,checks}.csv;
  results/tables/report/report_step_06_fig3b_dotplot_manifest.json

Command (project root):  python scripts/report_dct_dotplot.py
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

import anndata as ad
import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dotplot_lib as t6
import labels_core_lib as t18
import dct_prolif_lib as t31
import dotplot_dotplots as t6d   # sets the dotplot figure style (Arial) on import
from dct_labels_figures import FORMATS, arial

TASK = "report"
STEP = "06_fig3b_dotplot"
DCT_LABELS_DIR = cc.OUT_ROOTS["tables"] / "dct_labels"
DCT_PROLIF_DIR = cc.OUT_ROOTS["tables"] / "dct_prolif"
DCT_LABELS_H5AD = cc.OUT_ROOTS["h5ad"] / "dct_labels_DCT_ctrl_labelled.h5ad"
DCT_LABELS_CANON = DCT_LABELS_DIR / "dct_labels_dct_canonical_labels.csv"
DCT_LABELS_ORDER = DCT_LABELS_DIR / "dct_labels_marker_order_dct.csv"
DCT_LABELS_VALUES = DCT_LABELS_DIR / "dct_labels_dotplot_values.csv"
DCT_PROLIF_OVERLAY = DCT_PROLIF_DIR / "dct_prolif_prolif_overlay_per_cell.csv"
DCT_PROLIF_LABELS = DCT_PROLIF_DIR / "dct_prolif_dct_labels.csv"

PRIMARY = "cs_ctrl_deg_label"           # dct_labels Ctrl primary (the project notes, DCT labels, combined search)
DCT_PROLIF_COL = "cs_t31nf_label"              # dct_prolif primary run (no floor); route-A ProLIF only is used
GROUP_COL = "figure_3a_group"
ROWS = ["DCT1", "DCT2", "ProLIF", "Unknown"]                       # Figure 4e legend order
EXPECTED = {"DCT1": 362, "DCT2": 79, "ProLIF": 37, "Unknown": 251}  # Figure 4e (dct_prolif README section 16; the response to the reviewers)
ROW_DISPLAY = {"Unknown": t31.UNRESOLVED}   # display only (authors, 12 Nov 2026); data keep the dct_labels value 'Unknown'
WIDTH = 85.0
PITCH_MAX = 4.5


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: revised Figure 3B with the Figure 3A groups.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_ctrl_combined")

    # --- Figure 4e groups (dct_prolif overlay table) -------------------------------------------------------------------
    ov = pd.read_csv(run.use_input(DCT_PROLIF_OVERLAY, "dct_prolif overlay per cell (Figure 3A groups)"), dtype={"cell_id": str})
    ov = ov[(ov["run"] == "nofloor") & (ov["condition"] == "Ctrl")].set_index("cell_id")
    counts = ov["overlay_group"].value_counts().to_dict()
    checks.add("Figure 3A groups (overlay, no-floor run, Ctrl): counts equal the Figure 3A figure",
               "; ".join(f"{g} {counts.get(g, 0)}" for g in ROWS) + f"; total {len(ov)}",
               counts == EXPECTED and len(ov) == 729, "DCT1 362; DCT2 79; ProLIF 37; Unknown 251; total 729")

    # --- dct_labels labelled object and canonical labels -----------------------------------------------------------------
    adata = ad.read_h5ad(run.use_input(DCT_LABELS_H5AD, "dct_labels labelled object (Ctrl)"))
    ids = pd.Index(adata.obs_names.astype(str))
    same = set(ids) == set(ov.index) and len(ids) == len(ov)
    checks.add("dct_labels Ctrl object and the overlay table hold the same cells (matched on cell_id)",
               f"{len(ids)} object cells; {len(ov)} overlay cells; same set {same}", same)
    if not same:
        raise SystemExit("Cell sets differ; see the checks above.")
    ov = ov.reindex(ids)
    canon = pd.read_csv(run.use_input(DCT_LABELS_CANON, "dct_labels canonical DCT labels"), dtype={"cell_id": str})
    canon = canon[canon["condition"] == "Ctrl"].set_index("cell_id").reindex(ids)
    obs_primary = adata.obs[PRIMARY].astype(str).to_numpy()
    eq_canon = bool((canon[PRIMARY].astype(str).to_numpy() == obs_primary).all())
    eq_overlay = bool((ov["t27_label"].astype(str).to_numpy() == obs_primary).all())
    checks.add(f"{PRIMARY}: object = dct_labels canonical table = overlay t27_label", f"{eq_canon} / {eq_overlay}",
               eq_canon and eq_overlay)
    t31_lab = pd.read_csv(run.use_input(DCT_PROLIF_LABELS, "dct_prolif canonical labels"), dtype={"cell_id": str})
    t31_lab = t31_lab[t31_lab["condition"] == "Ctrl"].set_index("cell_id").reindex(ids)
    prolif_t31 = t31_lab[DCT_PROLIF_COL].astype(str).to_numpy() == t31.PROLIF
    prolif_ov = ov["overlay_group"].astype(str).to_numpy() == t31.PROLIF
    checks.add(f"ProLIF row = cells with {DCT_PROLIF_COL} ProLIF (dct_prolif_dct_labels.csv)",
               f"{int(prolif_ov.sum())} overlay; {int(prolif_t31.sum())} dct_prolif; equal {bool((prolif_ov == prolif_t31).all())}",
               bool((prolif_ov == prolif_t31).all()) and int(prolif_ov.sum()) == 37)
    rest = ~prolif_ov
    ok_rest = bool((ov["overlay_group"].astype(str).to_numpy()[rest] == obs_primary[rest]).all())
    checks.add(f"every other cell's group = its {PRIMARY}", f"{int(rest.sum())} cells; equal {ok_rest}", ok_rest)
    moved = pd.crosstab(pd.Series(obs_primary[prolif_ov], name=PRIMARY), "ProLIF row")["ProLIF row"].to_dict()
    checks.add(f"ProLIF row by {PRIMARY}", "; ".join(f"{k} {v}" for k, v in moved.items()), None)
    groups = pd.Series(ov["overlay_group"].astype(str).to_numpy(), index=ids, name=GROUP_COL)

    # --- markers (dct_labels panel order) --------------------------------------------------------------------------------
    order = pd.read_csv(run.use_input(DCT_LABELS_ORDER, "dct_labels marker order (DCT)"))
    order = (order[(order["version"] == "panel") & order["plotted"]].sort_values("plotted_position")
             .reset_index(drop=True))
    markers = order["marker"].tolist()
    checks.add("columns: dct_labels panel markers, plotted order", ", ".join(markers), len(markers) == 13, "13 markers")
    if "ln1p_intensity" not in adata.layers:
        adata.layers["ln1p_intensity"] = np.log1p(cc.dense(adata.layers["intensity"]).astype(np.float64))
        checks.add("layer ln1p_intensity computed from layers['intensity'] (absent from the object)", "True", None)

    # --- reproduction of the dct_labels dotplot values (same code, dct_labels primary rows) -------------------------------------
    t27v = pd.read_csv(run.use_input(DCT_LABELS_VALUES, "dct_labels dotplot values"))
    t27v = t27v[(t27v["condition"] == "Ctrl") & (t27v["label_column"] == PRIMARY) & t27v["marker"].isin(markers)]
    rep = t6.group_values(adata, pd.Series(obs_primary, index=ids), ["DCT1", "DCT2", "Unknown"], markers)
    rep.insert(0, "label_column", PRIMARY)
    rep["mean_z_scaled_0_1"] = t6.scaled_01(rep)
    m = rep.merge(t27v, on=["group", "marker"], suffixes=("", "_t27"))
    diffs = {c: float(np.max(np.abs(m[c] - m[f"{c}_t27"]))) for c in ("frac_detected", "mean_z", "mean_z_scaled_0_1")}
    ok_rep = len(m) == len(t27v) == 3 * len(markers) and max(diffs.values()) < 1e-9
    checks.add(f"same code reproduces the dct_labels dotplot values of {PRIMARY} (rows x markers; max abs diff)",
               f"{len(m)} of {len(t27v)}; " + "; ".join(f"{k} {v:.1e}" for k, v in diffs.items()), ok_rep, "< 1e-9")

    # --- Figure 4f values -----------------------------------------------------------------------------------------
    vals = t6.group_values(adata, groups, ROWS, markers)
    vals.insert(0, "label_column", GROUP_COL)
    vals["mean_z_scaled_0_1"] = t6.scaled_01(vals)
    vals = vals.merge(order[["marker", "group"]].rename(columns={"group": "marker_group"}), on="marker")
    vals.insert(vals.columns.get_loc("group") + 1, "row_display", vals["group"].map(lambda g: ROW_DISPLAY.get(g, g)))
    n_rows = vals.drop_duplicates("group").set_index("group")["n_cells"].to_dict()
    checks.add("cells per row", "; ".join(f"{g} {n_rows[g]}" for g in ROWS), n_rows == EXPECTED)
    top = vals.loc[vals.groupby("marker")["mean_z_scaled_0_1"].idxmax(), ["marker", "group"]].set_index("marker")
    checks.add("row with the highest colour per marker (1.0)",
               "; ".join(f"{mk} {top.loc[mk, 'group']}" for mk in markers), None)
    for mk in ("Mki67", "Top2a", "Slc12a3"):
        v = vals[vals["marker"] == mk].set_index("group")
        checks.add(f"{mk}: cells detected per row",
                   "; ".join(f"{g} {int(v.loc[g, 'n_detected'])}/{int(v.loc[g, 'n_cells'])}" for g in ROWS), None)
    med = ov.groupby("overlay_group")["n_detected"].median()
    checks.add("median protein IDs per cell, per row (overlay table n_detected)",
               "; ".join(f"{g} {med[g]:g}" for g in ROWS), None)
    colours = {g: t31.LABEL_COLORS[g] for g in ROWS}
    same_col = all(t18.LABEL_COLORS.get(g) == colours[g] for g in ROWS)
    checks.add("row colours (Figure 3A, dct_prolif_lib) equal the dct_labels dotplot colours (labels_core_lib)",
               "; ".join(f"{g} {c}" for g, c in colours.items()) + f"; equal {same_col}", None)

    # --- figure ---------------------------------------------------------------------------------------------------
    arial()
    frac = vals.pivot(index="group", columns="marker", values="frac_detected").reindex(index=ROWS, columns=markers)
    colr = vals.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=ROWS, columns=markers)
    shown = [ROW_DISPLAY.get(g, g) for g in ROWS]
    checks.add("row names shown in the figure (display only; data keep the dct_labels value)",
               "; ".join(f"{g} -> {s}" for g, s in zip(ROWS, shown) if g != s), None)
    panel = t6d.Panel("", shown, [n_rows[g] for g in ROWS], [colours[g] for g in ROWS], markers,
                      t6d.groups_of(order), frac.to_numpy(float), colr.to_numpy(float), [False] * len(markers))
    cv, pitch = t6d.build_figure([panel], WIDTH, "below", PITCH_MAX)
    tc = t6d.text_checks(cv)
    paths = run.figure(cv.fig, f"{STEP}_85mm",
                       "Revised Figure 3B: DCT control marker dotplot, rows = Figure 3A groups (dct_labels primary label with "
                       "the dct_prolif route-A ProLIF cells as their own row); dct_labels panel markers", formats=FORMATS, tight=False)
    fc = t6d.file_checks(paths, cv.W, cv.H)
    ok_fig = (tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
              and "arial" in tc["fonts"] and fc["width_ok"] and fc.get("pdf_truetype_embedded", False)
              and abs(cv.W - WIDTH) < 0.05)
    fig_row = {"figure": f"report_{STEP}_85mm", "width_mm": round(cv.W, 2), "height_mm": round(cv.H, 2),
               "pitch_mm": round(pitch, 3), **tc, **fc, "pass": ok_fig}
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 85 mm, TrueType in PDF)",
               f"{fig_row['width_mm']} x {fig_row['height_mm']} mm; min font {tc['min_font_pt']} pt; "
               f"overlaps {tc['n_text_overlaps']}; outside {tc['n_text_outside_figure']}", ok_fig)

    run.table(vals, f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame([fig_row])], ignore_index=True), f"{STEP}_checks")
    run.finish({"request": "authors, 10 Nov 2026: 'Draw 3B with 3A groups'",
                "figure_3a": "results/figures/dct_prolif/dct_prolif_prolif_overlay_nofloor_ctrl_85mm",
                "rows": ROWS, "expected_rows": EXPECTED, "primary_label": PRIMARY, "prolif_source": DCT_PROLIF_COL,
                "row_display": ROW_DISPLAY,
                "row_display_request": "authors, 12 Nov 2026 (via the response to the reviewers): 'Yes, rename there'",
                "markers": markers, "row_colours": colours, "width_mm": WIDTH,
                "encoding": "dct_labels / dotplot: area = fraction with intensity > 0; colour = row mean of the scaled .X, "
                            "rescaled 0-1 per protein across the four rows",
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit("A check failed; see report_06_fig3b_dotplot_checks.csv.")


if __name__ == "__main__":
    main()
