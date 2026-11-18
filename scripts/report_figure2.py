#!/usr/bin/env python
"""The response to the reviewers: Figure 2 (panels a-f) compiled as one figure in the journal's style.

(authors, 15 Nov 2026: "Yes, redraw"; the journal's "Checklist
for Style": lower-case panel letters, numbers of four digits closed up, one image per figure, 1000 dpi).

What changes against report_fig2_compiled (report_figure2_layout.py)
  letters   A, B, C, D, E, F -> a, b, c, d, e, f (same font, size and place)
  panel b   RNA row labels "TAL (7,440)", "CNT/PC (4,975)", "IC-B (2,447)" -> "TAL (7440)", "CNT/PC (4975)",
            "IC-B (2447)". The labels with five digits keep the comma: "Endothelial (16,141)", "PCT (50,741)",
            "PST (22,091)", "DCT (16,381)".
  nothing else: the layout rules, the panel order and the six other panel files are those of the current figure.

Panel b is drawn again as kidney_figure2_fig2b_full.py draws it (``stacked`` of that script, the dotplot / kidney_rna panel classes,
the same stored kidney_annotation and kidney_rna values; RNA of all atlas cells and nuclei), compared with the current panel PNG pixel for
pixel, and then the three labels are changed in the drawn figure. It is written as its own file too
(report_27_fig2b_ki_style).
The other panels (a, c top and bottom, d, e, f) are not changed: the PNG and the PDF of the compiled figure use the
kidney_figure2 panel files as they are. For the 1000-dpi TIFF each of them is drawn again by its kidney_figure2 function (``fig2a`` of
kidney_figure2_fig2ab.py; ``fig2c``, ``fig2d``, ``fig2e``, ``fig2f`` of kidney_figure2_panels.py; imported) from the kidney_object object
and the stored kidney_figure2 tables, with a stand-in for the run object that takes the figure instead of writing files. Each
drawing is first compared with the current panel PNG at 600 dpi, pixel for pixel, and then drawn at dpi 1000.
  PNG (600 dpi): the 600-dpi panel files placed pixel for pixel. TIFF (1000 dpi): the panels drawn by matplotlib at
  dpi 1000, placed pixel for pixel by the same rules (nothing enlarged; panels saved with a tight box differ from
  their 600-dpi size by up to 0.2 mm, so the page height differs by under 0.1 mm). PDF: the panel PDFs merged with
  pypdf (vector, fonts embedded).
No new statistics (no test is run again; the top-10 proteins, the core sets and the Jaccard values are read from the
stored kidney_figure2 tables); no value, colour, position or size is changed. No kidney_figure2 or report file of another thread is written.

Inputs (read-only): kidney_object kidney object and labels with the kidney_annotation columns; kidney_annotation dot-plot values and marker order; kidney_rna RNA
  dot-plot values; results/tables/kidney_figure2/kidney_figure2_t15_{per_cell_ids_dynamic_range, exact_combinations,
  core_jaccard_matrix, de_top10, fig2f_medians}.csv and kidney_figure2_fig2b_full_dotplot_values.csv;
  results/figures/kidney_figure2/ (the seven panel PDF and PNG files);
  results/figures/report/report_fig2_compiled.{pdf,png} (comparison).
Outputs: results/figures/report/report_27_fig2b_ki_style.{pdf,png};
  results/figures/report/report_27_fig2_ki_style.pdf (vector), .png (600 dpi), .tif (1000 dpi; RGB, LZW);
  results/tables/report/report_27_redraw_fig2_checks.csv, report_27_redraw_fig2_layout.csv;
  results/tables/report/report_step_27_redraw_fig2_manifest.json

Command (project root):  python scripts/report_figure2.py --overwrite
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
import kidney_figure2_fig2b_full as fb               # Figure 2B drawing (kidney_figure2 step fig2b_full)
import kidney_figure2_fig2ab as fa                   # Figure 2A drawing (kidney_figure2 step fig2ab)
import kidney_figure2_panels as fp                   # Figure 2C-2F drawing (kidney_figure2 step panels)
import report_figure_style_lib as R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T, t6dot, t7dot, t7, t15c, c2 = fb.T, fb.t6dot, fb.t7dot, fb.t7, fb.t15c, R.c2
TASK = R.TASK
STEP = "27_redraw_fig2"
NAME, NAME_B = "27_fig2_ki_style", "27_fig2b_ki_style"
FIG_DIR = cc.OUT_ROOTS["figures"] / TASK
KIDNEY_FIGURE2_FIG = cc.OUT_ROOTS["figures"] / T.TASK
KIDNEY_FIGURE2_TAB = cc.OUT_ROOTS["tables"] / T.TASK
OLD = FIG_DIR / "report_fig2_compiled"
# panel key of report_figure2_layout.PANELS -> name under which the kidney_figure2 function saves the panel
DRAWN_AS = {"A": "fig2a_umap", "C_top": "fig2c_ids_per_cell", "C_bottom": "fig2c_dynamic_range",
            "D": "fig2d_core_combinations", "E": "fig2e_core_jaccard", "F": "fig2f_stacked_violin_top10"}
EXPECTED_B = [("TAL (7,440)", "TAL (7440)"), ("CNT/PC (4,975)", "CNT/PC (4975)"), ("IC-B (2,447)", "IC-B (2447)")]
KEPT_B = ["Endothelial (16,141)", "PCT (50,741)", "PST (22,091)", "DCT (16,381)"]
EXPECTED_LETTERS = [("A", "a"), ("B", "b"), ("C D", "c d"), ("E F", "e f")]      # lines of the PDF text (pypdf)


class Capture:
    """Stands in for the run object in the kidney_figure2 drawing functions: takes each figure (drawn at 600 and at 1000 dpi
    with the keyword arguments of scp_common.save_figure) instead of writing files. Inputs go to the real run."""

    def __init__(self, run: cc.Run):
        self.run, self.images = run, {}

    def use_input(self, path, label: str = "", **kwargs):
        return self.run.use_input(path, label, **kwargs)

    def figure(self, fig, name: str, description: str = "", subdir: str = "", formats=(".pdf", ".png"),
               tight: bool = True) -> list:
        self.images[name] = {dpi: R.opaque_rgb(R.render(fig, dpi, tight), f"{name} at {dpi} dpi")
                             for dpi in (R.DPI_PNG, R.DPI_TIF)}
        plt.close(fig)
        return []


def layout(w: dict, h: dict, width, gap):
    """Positions (x from the left, y from the top) and page height; the rule of report_figure2_layout.py; any unit."""
    y_b = h["A"] + gap
    y_c = y_b + h["B"] + gap
    y_e = y_c + max(h["C_top"] + h["C_bottom"], h["D"]) + gap
    pos = {"A": (0, 0), "B": (0, y_b), "C_top": (0, y_c), "C_bottom": (0, y_c + h["C_top"]),
           "D": (width - w["D"], y_c), "E": (0, y_e), "F": (width - w["F"], y_e)}
    return pos, y_e + max(h["E"], h["F"])


def draw_b(run, checks, adata, labels, rows, colors):
    """Figure 2B as kidney_figure2_fig2b_full.main draws it for the RNA of all atlas cells and nuclei."""
    vals = pd.read_csv(run.use_input(t7.KIDNEY_ANNOTATION_DOTPLOT_VALUES, "kidney_annotation a_endo dotplot values (protein half)"))
    order_all = pd.read_csv(run.use_input(t7.KIDNEY_ANNOTATION_MARKER_ORDER, "kidney_annotation a_endo dotplot marker order"))
    rv = pd.read_csv(run.use_input(cc.out_path(t7.TASK, "tables", "dotplot_rna_values", ".csv"),
                                   "kidney_rna RNA dotplot values (RNA half)"))
    order = t6dot.plotted_order(order_all, fb.VERSION)
    markers = order["marker"].tolist()
    rows_v = vals.drop_duplicates("group").sort_values("group_position")["group"].tolist()
    checks.add("panel b: rows = kidney_annotation row order (Unknown last); columns = the 45 quantified markers, none repeated",
               "; ".join(rows_v) + f" | {len(markers)} markers",
               rows_v == rows and len(markers) == fb.N_MARKERS and len(set(markers)) == fb.N_MARKERS)
    disp = [T.display(r) for r in rows]
    vc = labels[fb.COL].value_counts()
    lo = pd.DataFrame({"label_column": fb.COL, "label": disp, "position": range(1, len(rows) + 1), "plotted": True,
                       "n_cells": [int(vc[r]) for r in rows], "colour": [colors[r] for r in rows]})
    prot = t6dot.make_panel(fb.PROT_TITLE, vals.assign(group=vals["group"].map(T.display)), lo, fb.COL,
                            order.assign(group=order["group"].map(T.display)))
    t6dot.plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    src = "all"
    sub = rv[(rv["label_column"] == fb.COL) & (rv["rna_source"] == src)].copy()
    sub["group"] = sub["group"].map(T.display)
    frac = sub.pivot(index="group", columns="marker", values="frac_detected").reindex(index=disp, columns=markers)
    colr = sub.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(index=disp, columns=markers)
    nrow = sub.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(disp)
    in_atlas = sub.drop_duplicates("marker").set_index("marker")["gene_in_atlas"].astype(bool)
    grey_rows = [int(n) == 0 for n in nrow]
    grey_cols = [not bool(in_atlas.get(m, False)) for m in markers]
    rna = t7dot.GreyPanel(f"RNA (Mouse Kidney Atlas, {t7dot.RNA_SOURCES[src][1]})", disp, [int(n) for n in nrow],
                          prot.row_colors, markers, prot.groups, frac.to_numpy(float), colr.to_numpy(float),
                          prot.repeat, grey_rows=grey_rows, grey_cols=grey_cols,
                          n_text=[f"{int(n):,}" if n > 0 else "–" for n in nrow])
    cv, pitch = fb.stacked([prot, rna])
    # values drawn = values the kidney_figure2 step stored for this panel
    out = []
    for panel, f, c, n in (("protein", prot.frac, prot.color, prot.row_n), ("RNA", rna.frac, rna.color, list(nrow))):
        for i, r in enumerate(disp):
            for j, mk in enumerate(markers):
                out.append({"panel": panel, "row": r, "marker": mk, "n_cells_row": int(n[i]),
                            "frac_detected": f[i, j], "colour_scaled_0_1": c[i, j],
                            "grey": panel == "RNA" and (grey_rows[i] or grey_cols[j])})
    stored = pd.read_csv(run.use_input(KIDNEY_FIGURE2_TAB / "kidney_figure2_fig2b_full_dotplot_values.csv", "kidney_figure2 Figure 2B values"))
    stored = stored[(stored["figure"] == "kidney_figure2_fig2b_marker_dotplot_full") & (stored["rna_source"] == src)]
    m = pd.DataFrame(out).merge(stored, on=["panel", "row", "marker"], suffixes=("", "_stored"))
    ok = (len(m) == len(out) == len(stored) == 2 * len(disp) * len(markers)
          and np.allclose(m["frac_detected"], m["frac_detected_stored"], rtol=0, atol=1e-12)
          and np.allclose(m["colour_scaled_0_1"], m["colour_scaled_0_1_stored"], rtol=0, atol=1e-12)
          and (m["n_cells_row"] == m["n_cells_row_stored"]).all() and (m["grey"] == m["grey_stored"].astype(bool)).all())
    checks.add("panel b: dot sizes, colours, cells per row and grey cells equal the stored kidney_figure2 values",
               f"{len(m)} of {len(out)} values matched; equal {bool(ok)}", bool(ok))
    return cv, pitch


def draw_others(cap: Capture, checks, adata, labels, rows, types, colors) -> None:
    """Panels a, c, d, e, f by their kidney_figure2 functions, from the kidney_object object and the stored kidney_figure2 tables."""
    lab_s = labels[T.PRIMARY].astype(str)
    lab = lab_s.to_numpy()
    n_cells = {r: int((lab == r).sum()) for r in rows}
    # a: as kidney_figure2_fig2ab.main
    conf = labels["kidney_t15_confidence"].astype(str).to_numpy()
    clusters = labels["t15_cluster"].astype(str).to_numpy()
    fa.fig2a(cap, np.asarray(adata.obsm["X_umap"]), lab, conf, clusters, rows, colors, n_cells, [])
    # c: per-cell values stored by kidney_figure2_panels.py
    per_cell = pd.read_csv(cap.use_input(KIDNEY_FIGURE2_TAB / "kidney_figure2_t15_per_cell_ids_dynamic_range.csv",
                                         "kidney_figure2 per-cell protein IDs and dynamic range"), dtype={"cell_id": str})
    same = (list(per_cell["cell_id"]) == list(labels.index.astype(str))
            and bool((per_cell["label"].astype(str).to_numpy() == lab).all()))
    checks.add("panel c: stored per-cell table holds the 712 cells in object order with the same labels",
               f"{len(per_cell)} rows; {same}", same and len(per_cell) == 712)
    fp.fig2c(cap, per_cell, rows, colors, [])
    # d: the 20 plotted core-set combinations
    comb = pd.read_csv(cap.use_input(KIDNEY_FIGURE2_TAB / "kidney_figure2_t15_exact_combinations.csv", "kidney_figure2 exact core-set combinations"))
    top = comb[comb["in_top20"].astype(bool)]
    checks.add("panel d: combinations plotted (stored top 20) and their protein counts",
               f"{len(top)}: {', '.join(map(str, top['count']))}", len(top) == T.UPSET_TOPN)
    fp.fig2d(cap, top, types, n_cells, [])
    # e: Jaccard matrix
    jm = pd.read_csv(cap.use_input(KIDNEY_FIGURE2_TAB / "kidney_figure2_t15_core_jaccard_matrix.csv", "kidney_figure2 core Jaccard matrix"))
    names = [T.display(t) for t in types]
    checks.add("panel e: stored Jaccard matrix has the eight cell types in the panel order",
               "; ".join(jm["type"]), list(jm["type"]) == names and list(jm.columns[1:]) == names)
    fp.fig2e(cap, jm[names].to_numpy(float), types, n_cells, [])
    # f: the stored top-10 proteins; intensities of the kidney_object object scaled as kidney_figure2_panels.main does
    top10 = pd.read_csv(cap.use_input(KIDNEY_FIGURE2_TAB / "kidney_figure2_t15_de_top10.csv", "kidney_figure2 top-10 proteins")
                        ).sort_values("rank")["protein"].tolist()
    var_names = list(adata.var_names.astype(str))
    ad_de = T.de_adata(T.intensity(adata), lab_s, var_names, order=types)     # known groups; no test is run
    scaled = T.scale_var(np.asarray(ad_de.X)[:, [var_names.index(g) for g in top10]])
    med = fp.fig2f(cap, scaled, ad_de.obs["group"].astype(str).to_numpy(), types, n_cells, top10, [])
    stored = pd.read_csv(cap.use_input(KIDNEY_FIGURE2_TAB / "kidney_figure2_t15_fig2f_medians.csv", "kidney_figure2 Figure 2F medians"))
    d = float(np.abs(stored[top10].to_numpy(float) - med).max()) if list(stored["type"]) == names else np.inf
    checks.add("panel f: group medians of the drawn violins equal the stored kidney_figure2 medians (max abs difference)",
               f"{d:.1e}; proteins {', '.join(top10)}", d < 1e-12, "< 1e-12")


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Figure 2 compiled (a-f) in the journal's style.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    R.stop_if_outputs_exist(run, {NAME: (".pdf", ".png", ".tif"), NAME_B: (".pdf", ".png")},
                            (f"{STEP}_checks", f"{STEP}_layout"))
    checks = t5.Checks(T.DATASET)
    np.random.seed(T.SEED)
    keys = [k for k, _, _, _ in c2.PANELS]
    stems = {k: stem for k, _, stem, _ in c2.PANELS}
    letter_of = {k: letter.lower() for k, letter, _, _ in c2.PANELS if letter}
    files = {k: {ext: run.use_input(KIDNEY_FIGURE2_FIG / f"{stems[k]}{ext}", f"current panel {k} ({ext[1:]})")
                 for ext in (".pdf", ".png")} for k in keys}
    old_png_path = run.use_input(OLD.with_suffix(".png"), "current compiled Figure 2 (PNG, 600 dpi)")
    old_pdf_path = run.use_input(OLD.with_suffix(".pdf"), "current compiled Figure 2 (PDF)")

    adata, labels = T.load(run, checks)
    lab = labels[T.PRIMARY].astype(str).to_numpy()
    types, rows = t15c.t15_rows(np.unique(lab))
    colors = T.palette(adata, rows, checks)      # kidney_annotation palette, PST #8c564b (as the kidney_figure2 panels)

    # --- panel b: drawn again, then the three row labels ------------------------------------------------------------------
    cv, pitch = draw_b(run, checks, adata, labels, rows, colors)
    old_b, facts_b = R.read_image(files["B"][".png"])
    ref = R.opaque_rgb(R.render(cv.fig, R.DPI_PNG), "panel b")
    n = int((ref != old_b).any(axis=2).sum()) if ref.shape == old_b.shape else -1
    checks.add("panel b: drawing with the current strings equals the current panel PNG pixel for pixel (600 dpi)",
               f"{ref.shape[1]} x {ref.shape[0]} px; {n} pixels differ", n == 0, "0")
    changes = R.edit_texts(cv.fig, R.close_up)
    checks.add("panel b: texts changed (old -> new)", R.changes_text(changes),
               sorted((c["old"], c["new"]) for c in changes) == sorted(EXPECTED_B), f"{len(EXPECTED_B)} row labels")
    drawn = [t.get_text() for t in cv.texts]
    checks.add("panel b: the labels with five digits keep their comma", "; ".join(k for k in KEPT_B if k in drawn),
               all(k in drawn for k in KEPT_B), "; ".join(KEPT_B))
    tc = t6dot.text_checks(cv)
    checks.add("panel b: format after the change (texts >= 7 pt, Arial, no text overlap, no text outside)",
               f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; {tc['fonts']}; overlaps {tc['n_text_overlaps']}; "
               f"outside {tc['n_text_outside_figure']}", tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0
               and tc["n_text_outside_figure"] == 0
               and all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";")))
    b600 = R.opaque_rgb(R.render(cv.fig, R.DPI_PNG), "panel b at 600 dpi")
    b1000 = R.opaque_rgb(R.render(cv.fig, R.DPI_TIF), "panel b at 1000 dpi")
    desc_b = ("Figure 2B in the journal's style (numbers of four digits closed up): paired protein / RNA marker "
              "dotplot, all 45 quantified markers; values and layout of kidney_figure2_fig2b_marker_dotplot_full")
    pdf_b, png_b = run.figure(cv.fig, NAME_B, desc_b, formats=(".pdf", ".png"), tight=False)
    new_b, _ = R.read_image(png_b)
    checks.add("panel b: written PNG equals the 600-dpi drawing", f"{int((new_b != b600).any(axis=2).sum())} pixels "
               "differ", new_b.shape == b600.shape and bool((new_b == b600).all()), "0")
    d = R.diff_outside(old_b, new_b, R.change_boxes(changes))
    checks.add("panel b: new PNG against the current panel PNG: pixels that differ; of these outside the boxes of the "
               "changed labels", f"{d['n_diff']} differ; {d['n_outside']} outside {d['outside_box']}",
               d["same_size"] and d["n_outside"] == 0 and d["n_diff"] > 0, "0 outside")
    old_bf = R.pdf_facts(files["B"][".pdf"])
    R.add_file_checks(checks, "panel b", pdf=pdf_b, png=png_b, tif=None,
                      size_mm=(old_bf["width_mm"], old_bf["height_mm"]),
                      png_px=(facts_b["width_px"], facts_b["height_px"]), tif_px=None,
                      size_from="the size of the current panel")
    R.add_text_checks(checks, "panel b", old_bf["lines"], R.pdf_facts(pdf_b)["lines"], EXPECTED_B,
                      "the current panel PDF")

    # --- the other panels: drawn again for the 1000-dpi image; nothing changed ---------------------------------------------
    cap = Capture(run)
    draw_others(cap, checks, adata, labels, rows, types, colors)
    current = {"B": new_b}
    for k, name in DRAWN_AS.items():
        current[k], facts = R.read_image(files[k][".png"])
        ref = cap.images[name][R.DPI_PNG]
        n = int((ref != current[k]).any(axis=2).sum()) if ref.shape == current[k].shape else -1
        big = cap.images[name][R.DPI_TIF]
        checks.add(f"panel {k}: drawing equals the current panel PNG pixel for pixel (600 dpi); the same figure at "
                   "1000 dpi", f"{ref.shape[1]} x {ref.shape[0]} px, {n} pixels differ; {big.shape[1]} x "
                   f"{big.shape[0]} px = {R.mm(big.shape[1], R.DPI_TIF):.2f} x {R.mm(big.shape[0], R.DPI_TIF):.2f} mm "
                   f"(600 dpi: {R.mm(ref.shape[1], R.DPI_PNG):.2f} x {R.mm(ref.shape[0], R.DPI_PNG):.2f} mm)",
                   n == 0 and facts["alpha_min"] == 255, "0")
    panels = {R.DPI_PNG: current, R.DPI_TIF: {"B": b1000, **{k: cap.images[name][R.DPI_TIF] for k, name in DRAWN_AS.items()}}}
    pdfs = {k: (pdf_b if k == "B" else files[k][".pdf"]) for k in keys}

    # --- compiled figure ------------------------------------------------------------------------------------------------
    out_rows, pages, letters_at = [], {}, {}
    for dpi in (R.DPI_PNG, R.DPI_TIF):
        img = panels[dpi]
        w = {k: img[k].shape[1] for k in keys}
        h = {k: img[k].shape[0] for k in keys}
        width = R.px(R.PAGE_MM, dpi)
        pos, height = layout(w, h, width, R.px(R.ROW_GAP_MM, dpi))
        inside, overlaps = R.layout_problems(pos, w, h, width, height)
        checks.add(f"{dpi} dpi: every panel inside the page; panels that overlap", f"{inside}; {overlaps or 'none'}",
                   inside and not overlaps)
        page = R.compose(img, pos, width, height)
        letters = [(letter_of[k], pos[k]) for k in keys if k in letter_of]
        blank = R.letters_blank(page, letters, dpi)
        checks.add(f"{dpi} dpi: the panel corners are blank where the letters go (letter with a rim of "
                   f"{R.LETTER_PAD_MM} mm)", "; ".join(f"{k}: {'blank' if v else 'NOT blank'}" for k, v in blank.items()),
                   all(blank.values()))
        if not all(blank.values()):
            raise SystemExit("a letter would cover drawn content")
        lettered, boxes = R.draw_letters(page, letters, dpi)
        n_bad = 0
        for k, (x, y) in pos.items():
            diff = (lettered[y:y + h[k], x:x + w[k]] != img[k]).any(axis=2)
            if k in letter_of:
                x0, y0, x1, y1 = boxes[letter_of[k]]
                diff[max(y0 - y, 0):y1 - y, max(x0 - x, 0):x1 - x] = False
            n_bad += int(diff.sum())
            out_rows.append({"image": f"{dpi} dpi", "panel": k, "letter": letter_of.get(k, ""), "x_px": x, "y_px": y,
                             "width_px": w[k], "height_px": h[k], "x_mm": round(R.mm(x, dpi), 2),
                             "y_mm": round(R.mm(y, dpi), 2), "width_mm": round(R.mm(w[k], dpi), 2),
                             "height_mm": round(R.mm(h[k], dpi), 2), "scaling": "none (1:1)"})
        checks.add(f"{dpi} dpi: the seven panels are placed pixel for pixel (letter boxes left out; no scaling)",
                   f"{n_bad} pixels differ from the panel images", n_bad == 0, "0")
        pages[dpi] = (lettered, pos, (width, height), boxes)
        letters_at[dpi] = letters
    order = R.reading_order(letters_at[R.DPI_PNG])
    checks.add("letters in reading order (top to bottom, left to right)", order, order == "abcdef", "abcdef")

    desc = ("Figure 2 compiled in the journal's style (lower-case letters a-f; numbers of four digits closed up in "
            "panel b); panels and layout of report_fig2_compiled")
    png, tif, pdf = (run.out("figures", NAME, ext) for ext in (".png", ".tif", ".pdf"))
    R.save_png(png, pages[R.DPI_PNG][0], desc)
    R.save_tif(tif, pages[R.DPI_TIF][0], desc)
    facts = {k: R.pdf_facts(p) for k, p in pdfs.items()}
    wp = {k: f["width_mm"] * R.PT_PER_MM for k, f in facts.items()}
    hp = {k: f["height_mm"] * R.PT_PER_MM for k, f in facts.items()}
    width_pt = R.PAGE_MM * R.PT_PER_MM
    pos_pt, height_pt = layout(wp, hp, width_pt, R.ROW_GAP_MM * R.PT_PER_MM)
    merged = R.merge_pdf(pdf, [(pdfs[k], x, y) for k, (x, y) in pos_pt.items()], width_pt, height_pt,
                         [(letter_of[k], pos_pt[k]) for k in keys if k in letter_of], "Figure 2")
    pos600 = pages[R.DPI_PNG][1]
    worst = max(max(abs(x / R.PT_PER_MM - R.mm(pos600[k][0], R.DPI_PNG)), abs(y / R.PT_PER_MM - R.mm(pos600[k][1], R.DPI_PNG)))
                for k, (x, y) in pos_pt.items())
    checks.add("PDF: panel positions agree with the 600-dpi image", f"largest difference {worst:.2f} mm", worst <= 0.2,
               "<= 0.2 mm")
    checks.add("PDF: merging added no raster image (the panel PDFs hold raster parts of their own: rasterized points, "
               "colour bars, the heat map)", f"{merged['n_images_page']} on the page; {merged['n_images_panels']} in "
               "the panels", merged["n_images_page"] == merged["n_images_panels"])
    checks.add("PDF: letters written as text", merged["letters_text"], sorted(merged["letters_text"]) == list("abcdef"),
               "abcdef")

    # --- checks on the written files, and against the current compiled figure ---------------------------------------------
    old_rgb, old_png = R.read_image(old_png_path)
    old = R.pdf_facts(old_pdf_path)
    size600, size1000 = pages[R.DPI_PNG][2], pages[R.DPI_TIF][2]
    R.add_file_checks(checks, "Figure 2", pdf=pdf, png=png, tif=tif, size_mm=(old["width_mm"], old["height_mm"]),
                      png_px=(old_png["width_px"], old_png["height_px"]), tif_px=size1000,
                      size_from="the size of the current figure")
    png_rgb, _ = R.read_image(png)
    tif_rgb, _ = R.read_image(tif)
    checks.add("written PNG and TIFF equal the composed pages",
               f"PNG {int((png_rgb != pages[R.DPI_PNG][0]).any(axis=2).sum())}, TIFF "
               f"{int((tif_rgb != pages[R.DPI_TIF][0]).any(axis=2).sum())} pixels differ",
               bool((png_rgb == pages[R.DPI_PNG][0]).all()) and bool((tif_rgb == pages[R.DPI_TIF][0]).all()), "0")
    allowed = R.change_boxes(changes, offset=pos600["B"])
    for letter, cell in letters_at[R.DPI_PNG]:
        allowed += [R.letter_box(letter, cell, R.DPI_PNG)[1], R.letter_box(letter.upper(), cell, R.DPI_PNG)[1]]
    d = R.diff_outside(old_rgb, png_rgb, allowed)
    checks.add("new PNG against the current compiled PNG: pixels that differ; of these outside the letter boxes and "
               "the boxes of the three changed labels of panel b",
               f"{d['n_diff']} differ; {d['n_outside']} outside {d['outside_box']}",
               d["same_size"] and d["n_outside"] == 0 and d["n_diff"] > 0, "0 outside")
    R.add_tif_png_check(checks, "Figure 2", tif_rgb, png_rgb)
    R.add_text_checks(checks, "Figure 2", old["lines"], R.pdf_facts(pdf)["lines"], EXPECTED_B + EXPECTED_LETTERS,
                      "the current compiled PDF")

    run.table(pd.DataFrame(out_rows), f"{STEP}_layout")
    run.table(checks.frame(), f"{STEP}_checks")
    n_fail = int((~checks.frame()["pass"]).sum())
    print(f"Figure 2: PDF {R.PAGE_MM:.1f} x {height_pt / R.PT_PER_MM:.1f} mm; PNG {size600[0]} x {size600[1]} px; "
          f"TIFF {size1000[0]} x {size1000[1]} px = {R.mm(size1000[0], R.DPI_TIF):.1f} x "
          f"{R.mm(size1000[1], R.DPI_TIF):.1f} mm")
    print(f"Checks: {len(checks.rows) - n_fail} passed or information, {n_fail} failed")
    run.finish({"figure": "Figure 2 (a-f)", "panels": {k: stems[k] for k in keys},
                "panel_b": "kidney_figure2_fig2b_marker_dotplot_full drawn again (kidney_figure2_fig2b_full.stacked); three "
                           "row labels closed up",
                "other_panels": "kidney_figure2 files as they are (PNG, PDF); drawn again at dpi 1000 by kidney_figure2_fig2ab.fig2a "
                                "and kidney_figure2_panels.fig2c-fig2f for the TIFF",
                "strings_changed": [f"{c['old']} -> {c['new']}" for c in changes] + ["letters A-F -> a-f"],
                "page_width_mm": R.PAGE_MM, "row_gap_mm": R.ROW_GAP_MM, "letter_pt": R.LETTER_PT,
                "letter_font": "Arial Bold", "letter_offset_mm": R.LETTER_OFFSET_MM,
                "size_mm_pdf": [R.PAGE_MM, round(height_pt / R.PT_PER_MM, 2)],
                "size_mm_tif": [round(R.mm(size1000[0], R.DPI_TIF), 2), round(R.mm(size1000[1], R.DPI_TIF), 2)],
                "png_dpi": R.DPI_PNG, "tif_dpi": R.DPI_TIF,
                "tif": "panels drawn by matplotlib at dpi 1000 and placed pixel for pixel (not enlarged)",
                "pdf": "panel PDFs merged with pypdf (vector)", "new_statistics": "none", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
