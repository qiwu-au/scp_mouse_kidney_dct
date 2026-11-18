#!/usr/bin/env python
"""The response to the reviewers: Figure 3 (panels a and b) compiled as one figure in the journal's style.

(authors, 15 Nov 2026: "Yes, redraw"; the journal's "Checklist
for Style": one image per figure, lower-case panel letters, numbers of four digits closed up).

Panels
  a  Figure 3A, report_05_fig3a_umap_85mm (85 x 80 mm): no text is changed. The PNG and PDF of the response to the reviewers are
     used as they are. For the 1000-dpi TIFF the panel is drawn again by ``panel`` of report_dct_umap.py
     (imported) from the same inputs; that drawing is first compared with the current PNG at 600 dpi, pixel for pixel.
  b  Figure 3B, report_11_fig3b_protein_rna_85mm (85 x 89.3 mm): drawn again as report_dct_dotplot_protein_rna.py
     draws it (``FixedPanel``, ``stacked_below`` of report_dct_rna_lib.py; the same stored values), compared with the
     current PNG pixel for pixel; then the RNA row labels "CNT (3,903)", "CTAL (3,600)", "MTAL (3,840)" become
     "CNT (3903)", "CTAL (3600)", "MTAL (3840)". "DCT (16,381)" keeps the comma. Nothing else is changed. Written
     as its own file too (report_27_fig3b_ki_style).
Compiled figure: 180 mm wide; the panels at their drawn size (no scaling), top-aligned, a at the left edge, b ending
  at the right edge; letters a and b in Arial Bold 10 pt at 0.5 / 0.4 mm from the left / top of each panel cell, as
  report_figure2_layout.py places its letters. Where a letter would cover drawn content (the title of panel b
  starts 1.5 mm below the top of the panel), a top margin of TOP_MARGIN_MM is added above both panels and the letters
  stand in it; nothing is made smaller.
  PNG (600 dpi): the 600-dpi panel files, pixel for pixel. TIFF (1000 dpi): the two panels drawn by matplotlib at
  dpi 1000, pixel for pixel (nothing enlarged). PDF: the two panel PDFs merged with pypdf (vector, fonts embedded).
No new statistics; no value, colour, position or size is changed. The the response to the reviewers and the response to the reviewers files are not written.

Inputs (read-only): as report_dct_umap.py (dct_prolif overlay table; dct_labels labelled control object) and
  report_dct_dotplot_protein_rna.py;
  results/tables/report/report_11_fig3b_protein_rna_values.csv (values check);
  results/figures/report/report_05_fig3a_umap_85mm.{pdf,png}, report_11_fig3b_protein_rna_85mm.{pdf,png}.
Outputs: results/figures/report/report_27_fig3b_ki_style.{pdf,png};
  results/figures/report/report_27_fig3_ki_style.pdf (vector), .png (600 dpi), .tif (1000 dpi; RGB, LZW);
  results/tables/report/report_27_redraw_fig3_checks.csv, report_27_redraw_fig3_layout.csv;
  results/tables/report/report_step_27_redraw_fig3_manifest.json

Command (project root):  python scripts/report_figure4.py --overwrite
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
import report_dct_umap as a3          # Figure 3A drawing
import report_dct_dotplot_protein_rna as b3      # Figure 3B drawing
import report_figure_style_lib as R

import matplotlib.pyplot as plt

t6d = b3.t6d
TASK = R.TASK
STEP = "27_redraw_fig3"
NAME, NAME_B = "27_fig3_ki_style", "27_fig3b_ki_style"
FIG_DIR = cc.OUT_ROOTS["figures"] / TASK
OLD_A = FIG_DIR / "report_05_fig3a_umap_85mm"
OLD_B = FIG_DIR / "report_11_fig3b_protein_rna_85mm"
OLD_B_VALUES = cc.OUT_ROOTS["tables"] / TASK / "report_11_fig3b_protein_rna_values.csv"
TOP_MARGIN_MM = 3.0            # used only if a letter would cover drawn content (as the room above the F-19 plots)
EXPECTED_B = [("CNT (3,903)", "CNT (3903)"), ("CTAL (3,600)", "CTAL (3600)"), ("MTAL (3,840)", "MTAL (3840)")]
KEPT_B = "DCT (16,381)"


def layout(w: dict, h: dict, width, top):
    """Positions (x from the left, y from the top) and page height; any unit."""
    return {"A": (0, top), "B": (width - w["B"], top)}, top + max(h["A"], h["B"])


def draw_a(run, checks):
    """Figure 3A as report_dct_umap.main draws it (inputs, style, panel)."""
    ov = pd.read_csv(run.use_input(a3.DCT_PROLIF_OVERLAY, "dct_prolif overlay per cell (Figure 3A groups)"), dtype={"cell_id": str})
    ov = ov[(ov["run"] == "nofloor") & (ov["condition"] == "Ctrl")].set_index("cell_id")
    adata = ad.read_h5ad(run.use_input(a3.DCT_LABELS_H5AD, "dct_labels labelled object (Ctrl): UMAP coordinates"))
    ids = pd.Index(adata.obs_names.astype(str))
    if set(ids) != set(ov.index) or len(ids) != len(ov):
        raise SystemExit("dct_labels object and the overlay table hold different cells")
    ov = ov.reindex(ids)
    group = ov["overlay_group"].astype(str).to_numpy()
    counts = pd.Series(group).value_counts().to_dict()
    checks.add("panel a: groups of the overlay table equal the the response to the reviewers figure",
               "; ".join(f"{g} {counts.get(g, 0)}" for g in a3.EXPECTED), counts == a3.EXPECTED,
               "DCT1 362; DCT2 79; ProLIF 37; Unknown 251")
    a3.t28.style()
    fig, _, ncol, labels = a3.panel(np.asarray(adata.obsm["X_umap"], float), group)
    checks.add("panel a: key labels (not changed)", "; ".join(labels) + f"; {ncol} columns",
               labels == ["DCT1 (362)", "DCT2 (79)", "ProLIF (37)", f"{a3.t31.UNRESOLVED} (251)"])
    return fig


def draw_b(run, checks):
    """Figure 3B as report_dct_dotplot_protein_rna.main draws it (stored values, style, panels, canvas)."""
    pv = pd.read_csv(run.use_input(b3.PROTEIN_VALUES, "the response to the reviewers Figure 3B values (protein half)"))
    markers = pv.drop_duplicates("marker")["marker"].tolist()
    n_rows = pv.drop_duplicates("group").set_index("group")["n_cells"].to_dict()
    shown = pv.drop_duplicates("group").set_index("group")["row_display"].to_dict()
    groups = b3.marker_groups(pv.drop_duplicates("marker")["marker_group"].tolist())
    frac = pv.pivot(index="group", columns="marker", values="frac_detected").reindex(index=b3.ROWS, columns=markers)
    colr = pv.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=b3.ROWS, columns=markers)
    rv = pd.read_csv(run.use_input(b3.RNA_VALUES, "dct_rna RNA dotplot values (RNA half)"))
    rna = b3.rna_panel_values(rv, markers, "published", "all", b3.ATLAS_ROWS)
    checks.add("panel b: rows, cells per row and markers (protein); atlas cells per row (RNA)",
               "; ".join(f"{g} {n_rows.get(g)}" for g in b3.ROWS) + f"; {len(markers)} markers | "
               + "; ".join(f"{r} {n}" for r, n in zip(b3.ATLAS_ROWS, rna["n"])),
               n_rows == b3.EXPECTED and len(markers) == 13 and rna["n"] == [16381, 810, 3903, 3600, 3840])
    b3.arial()
    t6d.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6d.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    prot = b3.FixedPanel("Protein (SCP-MS)", [shown[g] for g in b3.ROWS], [n_rows[g] for g in b3.ROWS],
                         [b3.ROW_COLOURS[g] for g in b3.ROWS], markers, groups, frac.to_numpy(float),
                         colr.to_numpy(float), [False] * len(markers))
    rnap = b3.FixedPanel(b3.RNA_TITLE["all"], list(b3.ATLAS_ROWS), rna["n"], rna["colours"], markers, groups,
                         rna["frac"].to_numpy(float), rna["colour"].to_numpy(float), [False] * len(markers),
                         grey_rows=[False] * len(b3.ATLAS_ROWS), grey_cols=rna["grey_cols"],
                         n_text=[f"{n:,}" for n in rna["n"]])
    cv, pitch = b3.stacked_below([prot, rnap], b3.WIDTH)
    # values drawn = values the the response to the reviewers script stored
    out = []
    for half, panel, rows, n in (("protein", prot, b3.ROWS, prot.row_n), ("RNA", rnap, b3.ATLAS_ROWS, rna["n"])):
        for i, r in enumerate(rows):
            for j, mk in enumerate(markers):
                out.append({"half": half, "row": r, "marker": mk, "n_cells_row": int(n[i]),
                            "frac_detected": panel.frac[i, j], "colour_scaled_0_1": panel.color[i, j],
                            "grey": half == "RNA" and bool(rna["grey_cols"][j])})
    stored = pd.read_csv(run.use_input(OLD_B_VALUES, "the response to the reviewers values table"))
    m = pd.DataFrame(out).merge(stored, on=["half", "row", "marker"], suffixes=("", "_stored"))
    ok = (len(m) == len(out) == len(stored) == 117
          and np.allclose(m["frac_detected"], m["frac_detected_stored"], rtol=0, atol=1e-12)
          and np.allclose(m["colour_scaled_0_1"], m["colour_scaled_0_1_stored"], rtol=0, atol=1e-12)
          and (m["n_cells_row"] == m["n_cells_row_stored"]).all() and (m["grey"] == m["grey_stored"].astype(bool)).all())
    checks.add("panel b: dot sizes, colours, cells per row and grey columns equal the stored the response to the reviewers values",
               f"{len(m)} of {len(out)} values matched; equal {bool(ok)}", bool(ok))
    return cv, pitch


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Figure 3 compiled (a, b) in the journal's style.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    R.stop_if_outputs_exist(run, {NAME: (".pdf", ".png", ".tif"), NAME_B: (".pdf", ".png")},
                            (f"{STEP}_checks", f"{STEP}_layout"))
    checks = t5.Checks("dct_ctrl_combined")
    files = {k: {ext: run.use_input(base.with_suffix(ext), f"current Figure 3{k} ({ext[1:]})") for ext in (".pdf", ".png")}
             for k, base in (("A", OLD_A), ("B", OLD_B))}

    # --- panel a: drawn again for the 1000-dpi image; no text changed ------------------------------------------------
    fig_a = draw_a(run, checks)
    old_a, facts_a = R.read_image(files["A"][".png"])
    ref = R.opaque_rgb(R.render(fig_a, R.DPI_PNG), "panel a")
    n = int((ref != old_a).any(axis=2).sum()) if ref.shape == old_a.shape else -1
    checks.add("panel a: drawing equals the current PNG pixel for pixel (600 dpi)",
               f"{ref.shape[1]} x {ref.shape[0]} px; {n} pixels differ", n == 0, "0")
    a1000 = R.opaque_rgb(R.render(fig_a, R.DPI_TIF), "panel a at 1000 dpi")
    plt.close(fig_a)

    # --- panel b: drawn again, then the three row labels ------------------------------------------------------------------
    cv, pitch = draw_b(run, checks)
    old_b, facts_b = R.read_image(files["B"][".png"])
    ref = R.opaque_rgb(R.render(cv.fig, R.DPI_PNG), "panel b")
    n = int((ref != old_b).any(axis=2).sum()) if ref.shape == old_b.shape else -1
    checks.add("panel b: drawing with the current strings equals the current PNG pixel for pixel (600 dpi)",
               f"{ref.shape[1]} x {ref.shape[0]} px; {n} pixels differ", n == 0, "0")
    changes = R.edit_texts(cv.fig, R.close_up)
    checks.add("panel b: texts changed (old -> new)", R.changes_text(changes),
               sorted((c["old"], c["new"]) for c in changes) == sorted(EXPECTED_B), f"{len(EXPECTED_B)} row labels")
    drawn = [t.get_text() for t in cv.texts]
    checks.add("panel b: the label with five digits keeps its comma", KEPT_B if KEPT_B in drawn else "not found",
               KEPT_B in drawn)
    tc = t6d.text_checks(cv)
    checks.add("panel b: format after the change (texts >= 7 pt, Arial, no text overlap, no text outside)",
               f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; {tc['fonts']}; overlaps {tc['n_text_overlaps']}; "
               f"outside {tc['n_text_outside_figure']}", tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0
               and tc["n_text_outside_figure"] == 0
               and all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";")))
    b600 = R.opaque_rgb(R.render(cv.fig, R.DPI_PNG), "panel b at 600 dpi")
    b1000 = R.opaque_rgb(R.render(cv.fig, R.DPI_TIF), "panel b at 1000 dpi")
    desc_b = ("Figure 3B in the journal's style (numbers of four digits closed up): DCT marker proteins in the "
              "Figure 3A groups above their transcripts in the Mouse Kidney Atlas; values and layout of "
              "report_11_fig3b_protein_rna_85mm")
    pdf_b, png_b = run.figure(cv.fig, NAME_B, desc_b, formats=(".pdf", ".png"), tight=False)
    new_b, _ = R.read_image(png_b)
    checks.add("panel b: written PNG equals the 600-dpi drawing", f"{int((new_b != b600).any(axis=2).sum())} pixels "
               "differ", new_b.shape == b600.shape and bool((new_b == b600).all()), "0")
    d = R.diff_outside(old_b, new_b, R.change_boxes(changes))
    checks.add("panel b: new PNG against the current PNG: pixels that differ; of these outside the boxes of the "
               "changed labels", f"{d['n_diff']} differ; {d['n_outside']} outside {d['outside_box']}",
               d["same_size"] and d["n_outside"] == 0 and d["n_diff"] > 0, "0 outside")
    old_bf = R.pdf_facts(files["B"][".pdf"])
    R.add_file_checks(checks, "panel b", pdf=pdf_b, png=png_b, tif=None, size_mm=(old_bf["width_mm"], old_bf["height_mm"]),
                      png_px=(facts_b["width_px"], facts_b["height_px"]), tif_px=None,
                      size_from="the size of the current panel")
    R.add_text_checks(checks, "panel b", old_bf["lines"], R.pdf_facts(pdf_b)["lines"], EXPECTED_B,
                      "the current panel PDF")

    # --- compiled figure ------------------------------------------------------------------------------------------------
    panels = {R.DPI_PNG: {"A": old_a, "B": new_b}, R.DPI_TIF: {"A": a1000, "B": b1000}}
    pdfs = {"A": files["A"][".pdf"], "B": pdf_b}

    def page_for(dpi: int, top_mm: float):
        img = panels[dpi]
        w = {k: v.shape[1] for k, v in img.items()}
        h = {k: v.shape[0] for k, v in img.items()}
        width = R.px(R.PAGE_MM, dpi)
        pos, height = layout(w, h, width, R.px(top_mm, dpi))
        letters = [("a", (pos["A"][0], 0)), ("b", (pos["B"][0], 0))]
        return R.compose(img, pos, width, height), pos, w, h, (width, height), letters

    page, pos, w, h, size, letters = page_for(R.DPI_PNG, 0.0)
    blank0 = R.letters_blank(page, letters, R.DPI_PNG)
    top_mm = 0.0 if all(blank0.values()) else TOP_MARGIN_MM
    checks.add("letters at the top-left corner of the panels without a margin: is the place blank?",
               "; ".join(f"{k}: {'blank' if v else 'covers drawn content'}" for k, v in blank0.items())
               + f" -> top margin {top_mm:g} mm", None)
    out_rows, pages, boxes = [], {}, {}
    for dpi in (R.DPI_PNG, R.DPI_TIF):
        page, pos, w, h, size, letters = page_for(dpi, top_mm)
        inside, overlaps = R.layout_problems(pos, w, h, *size)
        checks.add(f"{dpi} dpi: every panel inside the page; panels that overlap", f"{inside}; {overlaps or 'none'}",
                   inside and not overlaps)
        blank = R.letters_blank(page, letters, dpi)
        checks.add(f"{dpi} dpi: the page is blank where the letters go (letter with a rim of {R.LETTER_PAD_MM} mm)",
                   "; ".join(f"{k}: {'blank' if v else 'NOT blank'}" for k, v in blank.items()), all(blank.values()))
        if not all(blank.values()):
            raise SystemExit("a letter would cover drawn content")
        lettered, boxes[dpi] = R.draw_letters(page, letters, dpi)
        pages[dpi] = (lettered, pos, w, h, size)
        for k, (x, y) in pos.items():
            same = bool((lettered[y:y + h[k], x:x + w[k]] == panels[dpi][k]).all())
            checks.add(f"{dpi} dpi: panel {k.lower()} is placed pixel for pixel ({w[k]} x {h[k]} px at {x}, {y}; no "
                       "scaling)", "identical to the panel image" if same else "differs", same)
            out_rows.append({"image": f"{dpi} dpi", "panel": k.lower(), "x_px": x, "y_px": y, "width_px": w[k],
                             "height_px": h[k], "x_mm": round(R.mm(x, dpi), 2), "y_mm": round(R.mm(y, dpi), 2),
                             "width_mm": round(R.mm(w[k], dpi), 2), "height_mm": round(R.mm(h[k], dpi), 2),
                             "scaling": "none (1:1)"})
        used = np.zeros(lettered.shape[:2], dtype=bool)
        for k, (x, y) in pos.items():
            used[y:y + h[k], x:x + w[k]] = True
        for x0, y0, x1, y1 in boxes[dpi].values():
            used[y0:y1, x0:x1] = True
        rest = int(((lettered < 255).any(axis=2) & ~used).sum())
        checks.add(f"{dpi} dpi: outside the panels and the letters the page is white", f"{rest} pixels not white",
                   rest == 0, "0")
    checks.add("letters in reading order", R.reading_order(letters), R.reading_order(letters) == "ab", "ab")

    desc = ("Figure 3 compiled in the journal's style: (a) control DCT cells on the UMAP by cluster annotation with the "
            "ProLIF cells; (b) DCT marker proteins above their transcripts in the Mouse Kidney Atlas; panels at their "
            "drawn size")
    png, tif, pdf = (run.out("figures", NAME, ext) for ext in (".png", ".tif", ".pdf"))
    R.save_png(png, pages[R.DPI_PNG][0], desc)
    R.save_tif(tif, pages[R.DPI_TIF][0], desc)
    # PDF: same rule in points
    facts = {k: R.pdf_facts(p) for k, p in pdfs.items()}
    wp = {k: f["width_mm"] * R.PT_PER_MM for k, f in facts.items()}
    hp = {k: f["height_mm"] * R.PT_PER_MM for k, f in facts.items()}
    width_pt, top_pt = R.PAGE_MM * R.PT_PER_MM, top_mm * R.PT_PER_MM
    pos_pt, height_pt = layout(wp, hp, width_pt, top_pt)
    merged = R.merge_pdf(pdf, [(pdfs[k], x, y) for k, (x, y) in pos_pt.items()], width_pt, height_pt,
                         [("a", (pos_pt["A"][0], 0.0)), ("b", (pos_pt["B"][0], 0.0))], "Figure 3")
    pos600 = pages[R.DPI_PNG][1]
    worst = max(max(abs(x / R.PT_PER_MM - R.mm(pos600[k][0], R.DPI_PNG)), abs(y / R.PT_PER_MM - R.mm(pos600[k][1], R.DPI_PNG)))
                for k, (x, y) in pos_pt.items())
    checks.add("PDF: panel positions agree with the 600-dpi image", f"largest difference {worst:.2f} mm", worst <= 0.2,
               "<= 0.2 mm")
    checks.add("PDF: merging added no raster image (the panel PDFs hold raster parts of their own: the points of "
               "panel a are drawn rasterized, the colour bar of panel b is an image)",
               f"{merged['n_images_page']} on the page; {merged['n_images_panels']} in the panels",
               merged["n_images_page"] == merged["n_images_panels"])
    checks.add("PDF: letters written as text", merged["letters_text"], merged["letters_text"] == "ab", "ab")

    # --- checks on the written files -------------------------------------------------------------------------------------
    size600, size1000 = pages[R.DPI_PNG][4], pages[R.DPI_TIF][4]
    R.add_file_checks(checks, "Figure 3", pdf=pdf, png=png, tif=tif,
                      size_mm=(R.PAGE_MM, height_pt / R.PT_PER_MM), png_px=size600, tif_px=size1000,
                      size_from=f"180 mm width and the height of the taller panel plus the top margin of {top_mm:g} mm")
    png_rgb, _ = R.read_image(png)
    tif_rgb, _ = R.read_image(tif)
    checks.add("written PNG and TIFF equal the composed pages",
               f"PNG {int((png_rgb != pages[R.DPI_PNG][0]).any(axis=2).sum())}, TIFF "
               f"{int((tif_rgb != pages[R.DPI_TIF][0]).any(axis=2).sum())} pixels differ",
               bool((png_rgb == pages[R.DPI_PNG][0]).all()) and bool((tif_rgb == pages[R.DPI_TIF][0]).all()), "0")
    R.add_tif_png_check(checks, "Figure 3", tif_rgb, png_rgb)
    old_lines = R.pdf_facts(files["A"][".pdf"])["lines"] + old_bf["lines"]
    R.add_text_checks(checks, "Figure 3", old_lines, R.pdf_facts(pdf)["lines"], EXPECTED_B + [("", "a b")],
                      "the two current panel PDFs")

    run.table(pd.DataFrame(out_rows), f"{STEP}_layout")
    run.table(checks.frame(), f"{STEP}_checks")
    n_fail = int((~checks.frame()["pass"]).sum())
    print(f"Figure 3: {R.mm(size600[0], R.DPI_PNG):.1f} x {R.mm(size600[1], R.DPI_PNG):.1f} mm; PNG {size600[0]} x "
          f"{size600[1]} px; TIFF {size1000[0]} x {size1000[1]} px")
    print(f"Checks: {len(checks.rows) - n_fail} passed or information, {n_fail} failed")
    run.finish({"figure": "Figure 3 (a, b)", "panel_a": "report_05_fig3a_umap_85mm (unchanged; drawn again at "
                "dpi 1000 by report_dct_umap.panel)", "panel_b": "report_11_fig3b_protein_rna_85mm "
                "drawn again (report_dct_rna_lib.FixedPanel, stacked_below); three row labels closed up",
                "strings_changed": [f"{c['old']} -> {c['new']}" for c in changes], "page_width_mm": R.PAGE_MM,
                "top_margin_mm": top_mm, "letter_pt": R.LETTER_PT, "letter_font": "Arial Bold",
                "letter_offset_mm": R.LETTER_OFFSET_MM, "size_mm": [R.PAGE_MM, round(height_pt / R.PT_PER_MM, 2)],
                "png_dpi": R.DPI_PNG, "tif_dpi": R.DPI_TIF,
                "tif": "panels drawn by matplotlib at dpi 1000 and placed pixel for pixel (not enlarged)",
                "pdf": "panel PDFs merged with pypdf (vector)", "new_statistics": "none", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
