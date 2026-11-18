#!/usr/bin/env python
"""report: Figure 2 compiled as one main figure from the seven kidney_figure2 panel files, with panel letters A-F.

at the authors' request (12 Nov 2026: "Compile Figure 2 as one
final main figure for me"; then "Yes, install pypdf" for a vector PDF). No analysis and no redrawing: the kidney_figure2 panel
files are placed at their drawn size (no scaling) on a white page 180 mm wide, and a bold letter is written into the
blank top-left corner of each panel. The kidney_figure2 files are not edited.
  PNG and TIFF: the 600-dpi PNG panels, pixel for pixel (no resampling).
  PDF: the PDF panels merged with pypdf, so lines and text stay vector and the Arial fonts stay embedded.

Panels (results/figures/kidney_figure2; the same files as revision/Figure 2A.png ... Figure 2F.png):
  A  kidney_figure2_fig2a_umap                       UMAP of the 712 retained cells, kidney annotation
  B  kidney_figure2_fig2b_marker_dotplot_full        marker dotplot, protein above RNA, 45 quantified markers
  C  kidney_figure2_fig2c_ids_per_cell (top)         protein IDs per cell by cell type
     kidney_figure2_fig2c_dynamic_range (bottom)     dynamic range per cell by cell type
  D  kidney_figure2_fig2d_core_combinations          exact core-set membership (UpSet layout)
  E  kidney_figure2_fig2e_core_jaccard               Jaccard index of the core proteomes
  F  kidney_figure2_fig2f_stacked_violin_top10       top-10 proteins, stacked violins
Layout (reading order A-F; every panel at its drawn size):
  row 1  A (left)
  row 2  B (full width)
  row 3  C, its two plots stacked (left) | D (right, top-aligned)
  row 4  E (left) | F (right)
  Rows are separated by ROW_GAP_MM; in a row the left panel starts at the left edge and the right panel ends at the
  right edge of the page.
Letters: Arial Bold, LETTER_PT points, black, at the top-left corner of the panel. The script stops if the letter
  would cover anything drawn in the panel.

Outputs: results/figures/report/report_fig2_compiled.png (600 dpi), .tif (600 dpi, LZW) and .pdf (vector).
  The same layout rules give the PDF and the image; the panel PDFs and PNGs differ by at most 0.1 mm in size, so
  panel positions agree to within 0.2 mm (checked).
  results/tables/report/report_fig2_compiled_{layout,checks}.csv;
  results/tables/report/report_step_fig2_compiled_manifest.json

Command (project root):  python scripts/report_figure2_layout.py
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
import hashlib
import json
import platform
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import pypdf
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter, Transformation

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "_scanpy_runtime.py").is_file())
SRC = ROOT / "results" / "figures" / "kidney_figure2"
FIG_DIR = ROOT / "results" / "figures" / "report"
TAB_DIR = ROOT / "results" / "tables" / "report"
REVISION = ROOT / "revision"
NAME = "report_fig2_compiled"

DPI = 600
PAGE_MM = 180.0
ROW_GAP_MM = 3.0
LETTER_PT = 10.0
LETTER_OFFSET_MM = (0.5, 0.4)          # from the left and the top edge of the panel
ARIAL_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")

# key, letter, file stem, the authors' copy in revision/
PANELS = [
    ("A", "A", "kidney_figure2_fig2a_umap", "Figure 2A.png"),
    ("B", "B", "kidney_figure2_fig2b_marker_dotplot_full", "Figure 2B.png"),
    ("C_top", "C", "kidney_figure2_fig2c_ids_per_cell", "Figure 2C (top).png"),
    ("C_bottom", None, "kidney_figure2_fig2c_dynamic_range", "Figure 2C (bottom).png"),
    ("D", "D", "kidney_figure2_fig2d_core_combinations", "Figure 2D.png"),
    ("E", "E", "kidney_figure2_fig2e_core_jaccard", "Figure 2E.png"),
    ("F", "F", "kidney_figure2_fig2f_stacked_violin_top10", "Figure 2F.png"),
]


def px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def mm(pixels: float) -> float:
    return pixels / DPI * 25.4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


PT_PER_MM = 72 / 25.4
PDF_TEXT = {       # words that must be readable as text in the merged PDF, per panel
    "A": "UMAP 1", "B": "Protein (SCP-MS)", "C_top": "Protein IDs per cell", "C_bottom": "Dynamic range",
    "D": "core-set membership", "E": "Jaccard index", "F": "Median in group",
}


def page_fonts(page) -> list[tuple[str, bool]]:
    """(font name, embedded?) for every font in the page resources."""
    out = []
    fonts = page["/Resources"].get("/Font", {})
    for _, ref in fonts.items():
        f = ref.get_object()
        d = f["/DescendantFonts"][0].get_object() if "/DescendantFonts" in f else f
        desc = d.get("/FontDescriptor")
        desc = desc.get_object() if desc is not None else {}
        out.append((str(f.get("/BaseFont")), any(k in desc for k in ("/FontFile", "/FontFile2", "/FontFile3"))))
    return out


def count_images(page) -> int:
    xobjects = page["/Resources"].get("/XObject")
    if xobjects is None:
        return 0
    xobjects = xobjects.get_object()
    return sum(1 for k in xobjects if xobjects[k].get_object().get("/Subtype") == "/Image")


def build_vector_pdf(out: Path, ascent_pt: float, raster_mm: dict, check) -> dict:
    """Merge the panel PDFs on one page with the layout rules of the image, add the letters as text."""
    pages = {}
    for key, _, stem, _ in PANELS:
        reader = PdfReader(str(SRC / f"{stem}.pdf"))
        p = reader.pages[0]
        plain = (len(reader.pages) == 1 and float(p.mediabox.left) == 0 and float(p.mediabox.bottom) == 0
                 and p.get("/Rotate") in (None, 0) and p.cropbox == p.mediabox)
        check(f"pdf {key}: source is one unrotated page with its origin at 0, 0", plain, True, plain)
        pages[key] = p
    w = {k: float(p.mediabox.width) for k, p in pages.items()}
    h = {k: float(p.mediabox.height) for k, p in pages.items()}
    width, gap = PAGE_MM * PT_PER_MM, ROW_GAP_MM * PT_PER_MM
    y_b = h["A"] + gap
    y_c = y_b + h["B"] + gap
    y_e = y_c + max(h["C_top"] + h["C_bottom"], h["D"]) + gap
    height = y_e + max(h["E"], h["F"])
    top = {          # x from the left, y from the top of the page (points)
        "A": (0.0, 0.0), "B": (0.0, y_b), "C_top": (0.0, y_c), "C_bottom": (0.0, y_c + h["C_top"]),
        "D": (width - w["D"], y_c), "E": (0.0, y_e), "F": (width - w["F"], y_e),
    }
    fits = all(x >= -0.01 and x + w[k] <= width + 0.01 and y + h[k] <= height + 0.01 for k, (x, y) in top.items())
    check("pdf: every panel lies inside the page", fits, True, fits)
    worst = max(max(abs(x / PT_PER_MM - raster_mm[k][0]), abs(y / PT_PER_MM - raster_mm[k][1]))
                for k, (x, y) in top.items())
    check("pdf: panel positions agree with the image", f"largest difference {worst:.2f} mm", "<= 0.2 mm",
          worst <= 0.2)

    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    for key, (x, y) in top.items():
        page.merge_transformed_page(pages[key], Transformation().translate(tx=x, ty=height - y - h[key]))

    bold = Path(font_manager.findfont(font_manager.FontProperties(family="Arial", weight="bold"),
                                      fallback_to_default=False))
    check("pdf letters: font file", bold.name.lower(), "arialbd.ttf", bold.name.lower() == "arialbd.ttf")
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig = plt.figure(figsize=(width / 72, height / 72))
        for key, letter, _, _ in PANELS:
            if letter is None:
                continue
            x, y = top[key]
            base = y + LETTER_OFFSET_MM[1] * PT_PER_MM + ascent_pt        # baseline, from the top of the page
            fig.text((x + LETTER_OFFSET_MM[0] * PT_PER_MM) / width, 1 - base / height, letter, ha="left",
                     va="baseline", fontsize=LETTER_PT, fontweight="bold", fontfamily="Arial", color="black")
        buf = BytesIO()
        fig.savefig(buf, format="pdf", transparent=True)
        plt.close(fig)
    buf.seek(0)
    page.merge_page(PdfReader(buf).pages[0])
    writer.add_metadata({"/Title": "Figure 2", "/Creator": "report_figure2_layout.py (pypdf)"})
    with open(out, "wb") as fh:
        writer.write(fh)

    # ---- read the written file back -------------------------------------------------------------------------
    back = PdfReader(str(out))
    p = back.pages[0]
    got = (float(p.mediabox.width), float(p.mediabox.height))
    check("pdf: one page of the expected size", f"{len(back.pages)} page, {got[0]:.2f} x {got[1]:.2f} pt",
          f"1 page, {width:.2f} x {height:.2f} pt",
          len(back.pages) == 1 and abs(got[0] - width) < 0.01 and abs(got[1] - height) < 0.01)
    text = " ".join((p.extract_text() or "").split())
    for key, words in PDF_TEXT.items():
        check(f"pdf: text of panel {key} is kept as text", words if words in text else "not found", words,
              words in text)
    fonts = page_fonts(p)
    names = sorted({n.split("+")[-1] for n, _ in fonts})
    check("pdf: every font is embedded", f"{sum(e for _, e in fonts)} of {len(fonts)} fonts ({', '.join(names)})",
          "all embedded, Arial only",
          bool(fonts) and all(e for _, e in fonts) and all(n.startswith("Arial") for n in names))
    buf.seek(0)
    letters_text = "".join(ch for ch in (PdfReader(buf).pages[0].extract_text() or "") if ch.isalpha())
    check("pdf: letters written as text", letters_text, "ABCDEF", sorted(letters_text) == list("ABCDEF"))
    n_src = sum(count_images(q) for q in pages.values())
    n_out = count_images(p)
    check("pdf: merging added no raster image (the panels' own colour bars and heat-map cells are images)",
          f"{n_out} images on the page", f"{n_src} in the seven panels", n_out == n_src)
    return {"width_pt": round(width, 2), "height_pt": round(height, 2), "width_mm": round(width / PT_PER_MM, 2),
            "height_mm": round(height / PT_PER_MM, 2),
            "panels_pt": {k: {"x": round(x, 2), "y_from_top": round(y, 2), "width": round(w[k], 2),
                              "height": round(h[k], 2)} for k, (x, y) in top.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="report: Figure 2 compiled from the kidney_figure2 panels (letters A-F).")
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args()
    outputs = [FIG_DIR / f"{NAME}.{ext}" for ext in ("png", "tif", "pdf")]
    if not args.overwrite and any(p.exists() for p in outputs):
        sys.exit("outputs exist; use --overwrite")

    checks: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})

    # ---- read the panels ----------------------------------------------------------------------------------
    img, info = {}, {}
    for key, letter, stem, user_copy in PANELS:
        path = SRC / f"{stem}.png"
        with Image.open(path) as im:
            dpi = float(im.info.get("dpi", (0, 0))[0])
            rgba = np.asarray(im.convert("RGBA"))
        check(f"{key}: source file is 600 dpi", f"{dpi:.2f}", "600", abs(dpi - DPI) < 0.5)
        check(f"{key}: source file has no transparent pixel", int(rgba[..., 3].min()), 255,
              int(rgba[..., 3].min()) == 255)
        img[key] = np.ascontiguousarray(rgba[..., :3])
        digest = sha256(path)
        copy = REVISION / user_copy
        same = copy.is_file() and sha256(copy) == digest
        check(f"{key}: same file as revision/{user_copy}", "identical" if same else "differs or absent",
              "identical", same)
        info[key] = {"letter": letter, "file": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest,
                     "width_px": rgba.shape[1], "height_px": rgba.shape[0]}

    # ---- layout (pixels at 600 dpi) -------------------------------------------------------------------------
    width, gap = px(PAGE_MM), px(ROW_GAP_MM)
    h = {k: v.shape[0] for k, v in img.items()}
    w = {k: v.shape[1] for k, v in img.items()}
    y_a = 0
    y_b = y_a + h["A"] + gap
    y_c = y_b + h["B"] + gap
    y_e = y_c + max(h["C_top"] + h["C_bottom"], h["D"]) + gap
    height = y_e + max(h["E"], h["F"])
    pos = {
        "A": (0, y_a),
        "B": (0, y_b),
        "C_top": (0, y_c),
        "C_bottom": (0, y_c + h["C_top"]),
        "D": (width - w["D"], y_c),
        "E": (0, y_e),
        "F": (width - w["F"], y_e),
    }
    check("page width", f"{width} px = {mm(width):.2f} mm", "180 mm", abs(mm(width) - PAGE_MM) < 0.05)
    inside = all(x >= 0 and y >= 0 and x + w[k] <= width and y + h[k] <= height for k, (x, y) in pos.items())
    check("every panel lies inside the page", inside, True, inside)
    keys = list(pos)
    overlaps = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            (xa, ya), (xb, yb) = pos[a], pos[b]
            if xa < xb + w[b] and xb < xa + w[a] and ya < yb + h[b] and yb < ya + h[a]:
                overlaps.append(f"{a}/{b}")
    check("no two panels overlap", overlaps or "none", "none", not overlaps)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for k, (x, y) in pos.items():
        canvas[y:y + h[k], x:x + w[k]] = img[k]
    page = Image.fromarray(canvas, "RGB")

    # ---- letters ------------------------------------------------------------------------------------------
    font = ImageFont.truetype(str(ARIAL_BOLD), int(round(LETTER_PT * DPI / 72)))
    draw = ImageDraw.Draw(page)
    letter_boxes, order = {}, []
    for key, letter, _, _ in PANELS:
        if letter is None:
            continue
        x, y = pos[key]
        at = (x + px(LETTER_OFFSET_MM[0]), y + px(LETTER_OFFSET_MM[1]))
        box = draw.textbbox(at, letter, font=font, anchor="lt")
        pad = px(0.3)
        x0, y0, x1, y1 = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
        region = canvas[max(y0, 0):y1, max(x0, 0):x1]
        blank = bool((region >= 250).all())
        check(f"{letter}: corner is blank where the letter goes ({mm(x1 - x0):.1f} x {mm(y1 - y0):.1f} mm)",
              "blank" if blank else "not blank", "blank", blank)
        if not blank:
            sys.exit(f"letter {letter} would cover drawn content")
        draw.text(at, letter, font=font, fill=(0, 0, 0), anchor="lt")
        letter_boxes[key] = (max(x0, 0), max(y0, 0), x1, y1)
        order.append((y, x, letter))
    reading = "".join(l for _, _, l in sorted(order))
    check("letters in reading order (top to bottom, left to right)", reading, "ABCDEF", reading == "ABCDEF")
    cap_mm = mm(draw.textbbox((0, 0), "A", font=font, anchor="lt")[3] - draw.textbbox((0, 0), "A", font=font,
                                                                                        anchor="lt")[1])
    check("letter: Arial Bold, capital height", f"{LETTER_PT:g} pt, {cap_mm:.2f} mm", "about 2.5 mm",
          2.2 <= cap_mm <= 2.8)

    # ---- save ---------------------------------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    png, tif, pdf = outputs
    page.save(png, dpi=(DPI, DPI))
    page.save(tif, dpi=(DPI, DPI), compression="tiff_lzw")
    raster_mm = {k: (mm(x), mm(y)) for k, (x, y) in pos.items()}
    pdf_layout = build_vector_pdf(pdf, font.getmetrics()[0] * 72 / DPI, raster_mm, check)

    # ---- check the saved files ----------------------------------------------------------------------------
    for path in (png, tif):
        with Image.open(path) as im:
            saved = np.asarray(im.convert("RGB"))
            dpi = float(im.info.get("dpi", (0, 0))[0])
        check(f"{path.suffix[1:]}: size and resolution", f"{saved.shape[1]} x {saved.shape[0]} px, {dpi:.1f} dpi",
              f"{width} x {height} px, 600 dpi", saved.shape[:2] == (height, width) and abs(dpi - DPI) < 0.5)
        for k, (x, y) in pos.items():
            diff = (saved[y:y + h[k], x:x + w[k]] != img[k]).any(axis=2)
            if k in letter_boxes:
                x0, y0, x1, y1 = letter_boxes[k]
                diff[y0 - y:y1 - y, x0 - x:x1 - x] = False
            n = int(diff.sum())
            check(f"{path.suffix[1:]}: panel {k} equals its source pixel for pixel (letter box excluded)",
                  f"{n} pixels differ", "0", n == 0)
    layout = pd.DataFrame([{
        "panel": k, "letter": info[k]["letter"] or "", "source_file": info[k]["file"],
        "x_mm": round(mm(x), 2), "y_mm": round(mm(y), 2),
        "width_mm": round(mm(w[k]), 2), "height_mm": round(mm(h[k]), 2),
        "x_px": x, "y_px": y, "width_px": w[k], "height_px": h[k], "scaling": "none (1:1)",
    } for k, (x, y) in pos.items()])
    layout.to_csv(TAB_DIR / f"{NAME}_layout.csv", index=False)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(TAB_DIR / f"{NAME}_checks.csv", index=False)

    manifest = {
        "task": "report", "step": "fig2_compiled", "created": datetime.now().isoformat(timespec="seconds"),
        "request": "authors, 12 Nov 2026: 'Compile Figure 2 as one final main figure for me'; 'Yes, install pypdf'",
        "method": "PNG/TIFF: panel PNG files placed 1:1 on a white page, letters drawn with Pillow. PDF: panel PDF "
                  "files merged 1:1 with pypdf, letters written as Arial Bold text. No analysis, no seed",
        "parameters": {"dpi": DPI, "page_width_mm": PAGE_MM, "row_gap_mm": ROW_GAP_MM, "letter_pt": LETTER_PT,
                       "letter_font": "Arial Bold", "letter_offset_mm": LETTER_OFFSET_MM},
        "figure_size": {"width_px": width, "height_px": height, "width_mm": round(mm(width), 2),
                        "height_mm": round(mm(height), 2)},
        "inputs": info,
        "outputs": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in outputs},
        "pdf": {"note": "vector: the seven panel PDFs merged on one page; fonts embedded; no raster image",
                **pdf_layout},
        "versions": {"python": platform.python_version(), "Pillow": PIL.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "pypdf": pypdf.__version__},
    }
    (TAB_DIR / "report_step_fig2_compiled_manifest.json").write_text(json.dumps(manifest, indent=1),
                                                                        encoding="utf-8")
    failed = int((~checks_df["pass"]).sum())
    print(f"Figure 2: {mm(width):.1f} x {mm(height):.1f} mm ({width} x {height} px at {DPI} dpi)")
    print(f"checks: {len(checks_df)} rows, {failed} failed")
    for row in checks_df[~checks_df["pass"]].itertuples():
        print("  FAILED:", row.check, "|", row.observed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
