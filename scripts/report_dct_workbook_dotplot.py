#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S-06-a, every quantified marker of the DCT list in the control cells.

Drawn at the authors' request (12 Nov 2026: "Yes, redraw
them"; the response to the reviewers, section 5 and open item 3). It replaces, as a reader-facing file, the dct_labels analysis figure
results/figures/dct_labels/dct_labels_dotplot_workbook_cs_ctrl_deg_label_85mm (in-figure title, row
"Unknown (262)"); the dct_labels file is not edited. No new statistics: the values are computed from the stored dct_labels labelled
object exactly as report_dct_dotplot.py does, and must equal the stored dct_labels dotplot values.

Rows: the cluster annotation of the control cells of the combined search (cs_ctrl_deg_label; 729 cells): DCT1 382,
  DCT2 85, Unknown 262. The dct_labels label value 'Unknown' (cells of clusters without marker support) is shown as
  "DCT, subtype not resolved" (dct_prolif_lib.UNRESOLVED; the row name of the redrawn Figure 3B; authors, 12 Nov 2026,
  via the response to the reviewers: "Yes, rename there"). Display only: the checks and the values file keep 'Unknown'; the values file
  has the column 'row_display'. No ProLIF row: the rows are the cluster annotation, not the groups of Figure 3A.
Columns: every marker of the DCT list (data/DCT/DCT_markers_V1.xlsx = Supplemental Table 1, DCT tab) that is quantified,
  each once under the first cell type of the list that names it, in the dct_labels plotted order
  (dct_labels_marker_order_dct.csv, version 'workbook'; 13 markers):
  DCT1 Slc12a3, Pvalb, Egf, Umod, Trpm7, Ptgfr | DCT2 Slc8a1, Calb1, S100g, Klk1, Trpv5 | ProLIF Top2a, Mki67.
  Col5a2 (DCT1) and Cenpf (ProLIF) are not quantified. The bracket names are categories of the marker list, not cell
  labels.
Encoding: as the dct_labels dotplots and Figure 3B (dotplot format): dot area = fraction of the row's cells with intensity > 0;
  colour = row mean of the scaled .X of the dct_labels labelled object (per-protein z-scored ln(1 + intensity)), rescaled
  0-1 per protein across the three rows; Arial >= 7 pt; exact 85 mm width (as Figure 3B); PDF + 600-dpi PNG + LZW
  TIFF. No title, no panel letter, no footer.

Inputs (read-only): results/h5ad/dct_labels_DCT_ctrl_labelled.h5ad;
  results/tables/dct_labels/dct_labels_dct_canonical_labels.csv, dct_labels_marker_order_dct.csv,
  dct_labels_dotplot_values.csv.
Outputs: results/figures/report/report_06_S06a_dct_markers.{pdf,png,tif};
  results/tables/report/report_06_s06a_figure_{values,checks}.csv;
  results/tables/report/report_step_06_s06a_figure_manifest.json

Command (project root):  python scripts/report_dct_workbook_dotplot.py
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
STEP = "06_s06a_figure"
FIGURE = "06_S06a_dct_markers"
DCT_LABELS_DIR = cc.OUT_ROOTS["tables"] / "dct_labels"
DCT_LABELS_H5AD = cc.OUT_ROOTS["h5ad"] / "dct_labels_DCT_ctrl_labelled.h5ad"
DCT_LABELS_CANON = DCT_LABELS_DIR / "dct_labels_dct_canonical_labels.csv"
DCT_LABELS_ORDER = DCT_LABELS_DIR / "dct_labels_marker_order_dct.csv"
DCT_LABELS_VALUES = DCT_LABELS_DIR / "dct_labels_dotplot_values.csv"

PRIMARY = "cs_ctrl_deg_label"                         # dct_labels Ctrl primary (the project notes, DCT labels, combined search)
ROWS = ["DCT1", "DCT2", "Unknown"]
EXPECTED = {"DCT1": 382, "DCT2": 85, "Unknown": 262}  # the project notes; the response to the reviewers section 5
ROW_DISPLAY = {"Unknown": t31.UNRESOLVED}             # display only, as the redrawn Figure 3B
# The response to the reviewers, section 5 (S-06-a) and the Supplementary Methods paragraph: the 13 markers in this order.
EXPECTED_MARKERS = ["Slc12a3", "Pvalb", "Egf", "Umod", "Trpm7", "Ptgfr", "Slc8a1", "Calb1", "S100g", "Klk1", "Trpv5",
                    "Top2a", "Mki67"]
EXPECTED_GROUPS = [("DCT1", 0, 5), ("DCT2", 6, 10), ("ProLIF", 11, 12)]
EXPECTED_ABSENT = ["Col5a2", "Cenpf"]                 # The response to the reviewers, Figure 3 legend (B)
WIDTH = 85.0
PITCH_MAX = 4.5


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-06-a.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_ctrl_combined")

    # --- dct_labels labelled object and canonical labels -----------------------------------------------------------------
    adata = ad.read_h5ad(run.use_input(DCT_LABELS_H5AD, "dct_labels labelled object (Ctrl)"))
    ids = pd.Index(adata.obs_names.astype(str))
    canon = pd.read_csv(run.use_input(DCT_LABELS_CANON, "dct_labels canonical DCT labels"), dtype={"cell_id": str})
    canon = canon[canon["condition"] == "Ctrl"].set_index("cell_id")
    same = set(ids) == set(canon.index) and len(ids) == len(canon) == 729
    checks.add("dct_labels Ctrl object and the canonical table hold the same 729 cells (matched on cell_id)",
               f"{len(ids)} object cells; {len(canon)} canonical cells; same set {same}", same)
    if not same:
        raise SystemExit("Cell sets differ; see the checks above.")
    canon = canon.reindex(ids)
    obs_primary = adata.obs[PRIMARY].astype(str).to_numpy()
    eq = bool((canon[PRIMARY].astype(str).to_numpy() == obs_primary).all())
    checks.add(f"{PRIMARY}: object = dct_labels canonical table", f"{eq}", eq)
    labels = pd.Series(obs_primary, index=ids, name=PRIMARY)
    counts = labels.value_counts().to_dict()
    checks.add("cells per row equal the canonical counts (the project notes; the response to the reviewers section 5)",
               "; ".join(f"{g} {counts.get(g, 0)}" for g in ROWS) + f"; total {len(labels)}", counts == EXPECTED,
               "DCT1 382; DCT2 85; Unknown 262; total 729")

    # --- markers (dct_labels workbook order: every quantified marker of the DCT list, first listing type) ----------------
    order_all = pd.read_csv(run.use_input(DCT_LABELS_ORDER, "dct_labels marker order (DCT)"))
    wb = order_all[order_all["version"] == "workbook"]
    order = wb[wb["plotted"]].sort_values("plotted_position").reset_index(drop=True)
    markers = order["marker"].tolist()
    checks.add("columns: the quantified markers of the DCT list in the dct_labels plotted order = the response to the reviewers (13, with Trpm7)",
               ", ".join(markers), markers == EXPECTED_MARKERS, "; ".join(EXPECTED_MARKERS))
    groups = t6d.groups_of(order)
    checks.add("column groups: the first cell type of the marker list that names the marker",
               "; ".join(f"{g} {j1 - j0 + 1}" for g, j0, j1 in groups), groups == EXPECTED_GROUPS,
               "DCT1 6; DCT2 5; ProLIF 2")
    first = bool((order["group"] == order["first_listing_type"]).all())
    checks.add("each marker's group = its first listing type in the marker list (dct_labels order table)", f"{first}", first)
    absent = wb.loc[~wb["plotted"], "marker"].tolist()
    checks.add("markers of the DCT list not quantified (not drawn)", ", ".join(absent) or "none",
               absent == EXPECTED_ABSENT, ", ".join(EXPECTED_ABSENT))
    if "ln1p_intensity" not in adata.layers:
        adata.layers["ln1p_intensity"] = np.log1p(cc.dense(adata.layers["intensity"]).astype(np.float64))
        checks.add("layer ln1p_intensity computed from layers['intensity'] (absent from the object)", "True", None)

    # --- values: computed as Figure 3B, compared with the stored dct_labels values ----------------------------------------
    vals = t6.group_values(adata, labels, ROWS, markers)
    vals.insert(0, "label_column", PRIMARY)
    vals["mean_z_scaled_0_1"] = t6.scaled_01(vals)
    t27v = pd.read_csv(run.use_input(DCT_LABELS_VALUES, "dct_labels dotplot values"))
    t27v = t27v[(t27v["condition"] == "Ctrl") & (t27v["label_column"] == PRIMARY) & t27v["marker"].isin(markers)]
    m = vals.merge(t27v, on=["group", "marker"], suffixes=("", "_t27"))
    diffs = {c: float(np.max(np.abs(m[c] - m[f"{c}_t27"]))) for c in ("frac_detected", "mean_z", "mean_z_scaled_0_1")}
    n_eq = bool((m["n_cells"] == m["n_cells_t27"]).all() and (m["n_detected"] == m["n_detected_t27"]).all())
    ok_rep = len(m) == len(t27v) == len(ROWS) * len(markers) and max(diffs.values()) < 1e-9 and n_eq
    checks.add("plotted values equal the stored dct_labels dotplot values (rows x markers; max abs diff; cell counts equal)",
               f"{len(m)} of {len(t27v)}; " + "; ".join(f"{k} {v:.1e}" for k, v in diffs.items()) + f"; counts {n_eq}",
               ok_rep, "39 of 39; < 1e-9")
    vals = vals.merge(order[["marker", "group"]].rename(columns={"group": "marker_group"}), on="marker")
    vals.insert(vals.columns.get_loc("group") + 1, "row_display", vals["group"].map(lambda g: ROW_DISPLAY.get(g, g)))
    n_rows = vals.drop_duplicates("group").set_index("group")["n_cells"].to_dict()
    top = vals.loc[vals.groupby("marker")["mean_z_scaled_0_1"].idxmax(), ["marker", "group"]].set_index("marker")
    checks.add("row with the highest colour per marker (1.0)",
               "; ".join(f"{mk} {top.loc[mk, 'group']}" for mk in markers), None)
    for mk in ("Trpm7", "Mki67", "Top2a"):
        v = vals[vals["marker"] == mk].set_index("group")
        checks.add(f"{mk}: cells detected per row",
                   "; ".join(f"{g} {int(v.loc[g, 'n_detected'])}/{int(v.loc[g, 'n_cells'])}" for g in ROWS), None)
    colours = {g: t18.LABEL_COLORS[g] for g in ROWS}
    same_col = all(t31.LABEL_COLORS.get(g) == colours[g] for g in ROWS)
    checks.add("row colours: the dct_labels dotplot colours (labels_core_lib) = the Figure 3A / 3B colours (dct_prolif_lib)",
               "; ".join(f"{g} {c}" for g, c in colours.items()) + f"; equal {same_col}", same_col)

    # --- figure ---------------------------------------------------------------------------------------------------
    arial()
    frac = vals.pivot(index="group", columns="marker", values="frac_detected").reindex(index=ROWS, columns=markers)
    colr = vals.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=ROWS, columns=markers)
    shown = [ROW_DISPLAY.get(g, g) for g in ROWS]
    checks.add("row names shown in the figure (display only; data keep the dct_labels value) = the redrawn Figure 3B row",
               "; ".join(f"{g} -> {s}" for g, s in zip(ROWS, shown) if g != s),
               shown == ["DCT1", "DCT2", "DCT, subtype not resolved"])
    panel = t6d.Panel("", shown, [n_rows[g] for g in ROWS], [colours[g] for g in ROWS], markers, groups,
                      frac.to_numpy(float), colr.to_numpy(float), [False] * len(markers))
    cv, pitch = t6d.build_figure([panel], WIDTH, "below", PITCH_MAX)
    drawn = [t.get_text() for t in cv.texts]
    banned = ("Unknown", "Generated", "cs_", "Ctrl", "rank-free", "Leiden")
    hits = sorted({b for b in banned for s in drawn if b in s})
    checks.add("no title, footer, task code, internal name or 'Unknown' among the drawn texts",
               f"{len(drawn)} texts; found: {', '.join(hits) or 'none'}", not hits)
    tc = t6d.text_checks(cv)
    paths = run.figure(cv.fig, FIGURE,
                       "Supplementary Figure S-06-a: DCT control cells, every quantified marker of the DCT list under "
                       "the first cell type that lists it; rows = cluster annotation", formats=FORMATS, tight=False)
    fc = t6d.file_checks(paths, cv.W, cv.H)
    ok_fig = (tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
              and tc["fonts"] == "arial.ttf" and fc["width_ok"] and fc.get("pdf_truetype_embedded", False)
              and abs(cv.W - WIDTH) < 0.05)
    fig_row = {"figure": f"report_{FIGURE}", "width_mm": round(cv.W, 2), "height_mm": round(cv.H, 2),
               "pitch_mm": round(pitch, 3), **tc, **fc, "pass": ok_fig}
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 85 mm, TrueType in PDF, 600 dpi)",
               f"{fig_row['width_mm']} x {fig_row['height_mm']} mm; min font {tc['min_font_pt']} pt; fonts "
               f"{tc['fonts']}; overlaps {tc['n_text_overlaps']}; outside {tc['n_text_outside_figure']}; png "
               f"{fc.get('png_dpi')}", ok_fig)

    run.table(vals, f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame([fig_row])], ignore_index=True), f"{STEP}_checks")
    run.finish({"request": "authors, 12 Nov 2026: 'Yes, redraw them'",
                "replaces_for_readers": "results/figures/dct_labels/"
                                        "dct_labels_dotplot_workbook_cs_ctrl_deg_label_85mm",
                "rows": ROWS, "expected_rows": EXPECTED, "label_column": PRIMARY, "row_display": ROW_DISPLAY,
                "markers": markers, "marker_groups": groups, "not_quantified": absent, "row_colours": colours,
                "width_mm": WIDTH,
                "encoding": "dct_labels / dotplot: area = fraction with intensity > 0; colour = row mean of the scaled .X, "
                            "rescaled 0-1 per protein across the three rows",
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit("A check failed; see report_06_s06a_figure_checks.csv.")


if __name__ == "__main__":
    main()
