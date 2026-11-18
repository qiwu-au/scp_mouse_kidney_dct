"""The response to the reviewers: shared helpers of the four figure redraws in the journal's style ("Checklist for Style").

(authors, 15 Nov 2026: "Yes, redraw"). Imported by
report_figure2.py, report_figure4.py, report_figure3.py and
report_figure5.py. Runs no analysis and writes nothing by itself.

What the four scripts share
- Every figure is drawn by the drawing code of its own thread (imported, not copied), with the strings of the current
  figure. That drawing is compared with the current 600-dpi PNG pixel for pixel; only then are the listed strings
  changed in the drawn figure (``edit_texts``). Values, colours, sizes and positions are not touched.
- PNG at 600 dpi and PDF as before; TIFF at 1000 dpi: the same figure drawn by matplotlib at dpi 1000 (``render``), or,
  for a compiled figure, its panels drawn at dpi 1000 and placed pixel for pixel. No image is enlarged.
- Compiled figures: the layout rules and the letter placement of report_figure2_layout.py (constants imported from
  it): panels at their drawn size, Arial Bold 10 pt letters at 0.5 / 0.4 mm from the left / top of the panel cell;
  PDF = the panel PDFs merged with pypdf.
- Checks: text lines of the old and the new PDF (pypdf), pixels outside the changed texts, page size, fonts, files.

Journal rules applied (the calling scripts list the strings): lower-case panel letters; axis labels in sentence case;
numbers of four digits closed up, five or more digits with a comma; P value with a capital italic P.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from pypdf import PdfReader, PdfWriter, Transformation

import scp_common as cc
import report_figure2_layout as c2      # layout constants and PDF helpers of the compiled Figure 2 (writes nothing on import)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text

Image.MAX_IMAGE_PIXELS = None             # Figure 2 at 1000 dpi is 7087 x 12492 pixels

TASK = "report"
PREFIX = "report_27_"                 # every file written by the the response to the reviewers redraw scripts starts with this
DPI_PNG, DPI_TIF = 600, 1000
PT_PER_MM = c2.PT_PER_MM
PAGE_MM = c2.PAGE_MM                      # 180 mm
ROW_GAP_MM = c2.ROW_GAP_MM                # 3 mm between rows of panels
LETTER_PT = c2.LETTER_PT                  # 10 pt
LETTER_OFFSET_MM = c2.LETTER_OFFSET_MM    # (0.5, 0.4) mm from the left and the top edge of the panel cell
LETTER_PAD_MM = 0.3                       # blank rim required around a letter (as report_figure2_layout.py)
ARIAL_BOLD = c2.ARIAL_BOLD

FOUR_DIGITS_COMMA = re.compile(r"(?<![\d,.])\d,\d{3}(?![\d,])")       # 1,234 but not 12,345 or 1,234,567
FIVE_DIGITS_COMMA = re.compile(r"(?<![\d,.])\d{2,3},\d{3}(?![\d,])")  # 12,345 and 123,456
LOWER_P = re.compile(r"(?<![A-Za-z])p\s*[=<>≤≥]")                      # "p =", "p <" ...


# ---------------------------------------------------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------------------------------------------------
def close_up(s: str) -> str:
    """Numbers of four digits lose the comma (7,440 -> 7440); numbers of five or more digits keep it."""
    return re.sub(r"(?<![\d,.])(\d),(\d{3})(?![\d,])", r"\1\2", s)


def replace_exact(mapping: dict[str, str]):
    """A converter that replaces whole strings only."""
    return lambda s: mapping.get(s, s)


# ---------------------------------------------------------------------------------------------------------------------
# Drawing and files
# ---------------------------------------------------------------------------------------------------------------------
def px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def mm(pixels: float, dpi: int) -> float:
    return pixels / dpi * 25.4


def render(fig, dpi: int, tight: bool = False) -> np.ndarray:
    """The figure drawn by matplotlib (Agg) at ``dpi`` as an RGBA array; keyword arguments as scp_common.save_figure."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else None)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGBA"))


def opaque_rgb(rgba: np.ndarray, what: str) -> np.ndarray:
    """RGB of an image without a transparent pixel (stops otherwise: dropping alpha would change colours)."""
    if rgba.shape[2] == 4:
        if int(rgba[..., 3].min()) != 255:
            raise SystemExit(f"{what}: transparent pixels found; not written.")
        rgba = rgba[..., :3]
    return np.ascontiguousarray(rgba)


def read_image(path: Path) -> tuple[np.ndarray, dict]:
    """RGB pixels of a PNG or TIFF file and what the file says about itself."""
    with Image.open(path) as im:
        facts = {"width_px": im.size[0], "height_px": im.size[1], "mode": im.mode,
                 "dpi": float(im.info.get("dpi", (0, 0))[0]), "compression": im.info.get("compression", ""),
                 "bands": "".join(im.getbands())}
        rgba = np.asarray(im.convert("RGBA"))
    facts["alpha_min"] = int(rgba[..., 3].min())
    return np.ascontiguousarray(rgba[..., :3]), facts


def _script() -> str:
    return cc.rel(Path(sys.argv[0])) if sys.argv and sys.argv[0] else "interactive"


def guard(path: Path) -> Path:
    """Only files of this thread's redraw are written (new names; nothing of another thread is replaced)."""
    path = Path(path)
    if not path.name.startswith(PREFIX):
        raise PermissionError(f"Refusing to write {path.name}: the response to the reviewers redraw files start with {PREFIX}")
    return path


def stop_if_outputs_exist(run: cc.Run, figures: dict[str, tuple[str, ...]], tables: tuple[str, ...]) -> None:
    """Before any work: stop if an output of the step exists and --overwrite was not given.

    ``figures``: name -> extensions; ``tables``: names of the CSV tables (the manifest is checked too)."""
    paths = [cc.out_path(run.task, "figures", name, ext) for name, exts in figures.items() for ext in exts]
    paths += [cc.out_path(run.task, "tables", name, ".csv") for name in tables]
    for p in paths:
        guard(p)                                  # figure and table names of this thread only
    paths.append(run.manifest_path)               # report_step_27_redraw_..._manifest.json (scp_common's name)
    found = [cc.rel(p) for p in paths if p.exists()]
    if found and not run.overwrite:
        raise SystemExit(f"{len(found)} outputs of this step exist (first: {found[0]}); rerun with --overwrite to "
                         "replace this step's own files.")


def save_png(path: Path, rgb: np.ndarray, description: str) -> None:
    info = PngInfo()
    info.add_text("Software", f"{cc.GENERATED_BY}; {_script()}")
    info.add_text("Description", description)
    info.add_text("Title", Path(path).stem)
    Image.fromarray(rgb).save(guard(path), dpi=(DPI_PNG, DPI_PNG), pnginfo=info)
    print(f"Saved PNG: {cc.rel(path)}")


def save_tif(path: Path, rgb: np.ndarray, description: str) -> None:
    """RGB without alpha, LZW, resolution tag 1000 dpi."""
    Image.fromarray(rgb).save(guard(path), compression="tiff_lzw", dpi=(DPI_TIF, DPI_TIF),
                              description=f"{cc.GENERATED_BY}; {_script()}; {description}")
    print(f"Saved TIFF: {cc.rel(path)}")


# ---------------------------------------------------------------------------------------------------------------------
# Changing strings in a drawn figure
# ---------------------------------------------------------------------------------------------------------------------
def _box(bbox, height_px: int, scale: float) -> tuple[int, int, int, int]:
    """Display box (origin bottom left) -> pixel box of the 600-dpi image (x0, y0, x1, y1; origin top left)."""
    return (int(np.floor(bbox.x0 * scale)), int(np.floor(height_px - bbox.y1 * scale)),
            int(np.ceil(bbox.x1 * scale)), int(np.ceil(height_px - bbox.y0 * scale)))


def edit_texts(fig, convert, skip=()) -> list[dict]:
    """Apply ``convert`` (str -> str) to every visible text of the figure; nothing else is changed.

    Tick labels are changed through their axis (matplotlib writes them again at every draw). Returns one row per
    changed text: kind, old, new and the pixel boxes of the old and the new text in the 600-dpi image.
    """
    scale = DPI_PNG / fig.dpi
    height_px = int(fig.bbox.height * scale)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pending = []
    tick_ids = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            ticks = list(axis.get_major_ticks()) + list(axis.get_minor_ticks())
            tick_ids.update(id(lab) for t in ticks for lab in (t.label1, t.label2))
            major = list(axis.get_major_ticks(len(axis.get_majorticklocs())))
            old = [t.label1.get_text() for t in major]
            new = [convert(s) for s in old]
            if new != old:
                boxes = [_box(t.label1.get_window_extent(r), height_px, scale) for t in major]
                axis.set_ticklabels(new)                      # locations, limits and tick style stay
                pending += [("tick label", major[i].label1, old[i], new[i], boxes[i])
                            for i in range(len(old)) if old[i] != new[i] and major[i].label1.get_visible()]
    for t in fig.findobj(Text):
        if id(t) in tick_ids or any(t is s for s in skip) or not t.get_visible():
            continue
        old = t.get_text()
        new = convert(old)
        if new != old:
            box = _box(t.get_window_extent(r), height_px, scale)
            t.set_text(new)
            pending.append(("text", t, old, new, box))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    return [{"kind": kind, "old": old, "new": new, "box_old": box,
             "box_new": _box(t.get_window_extent(r), height_px, scale)} for kind, t, old, new, box in pending]


def change_boxes(changes: list[dict], offset=(0, 0)) -> list[tuple[int, int, int, int]]:
    out = []
    for c in changes:
        for key in ("box_old", "box_new"):
            x0, y0, x1, y1 = c[key]
            out.append((x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]))
    return out


def changes_text(changes: list[dict]) -> str:
    return "; ".join(f"{c['old']!r} -> {c['new']!r}" for c in changes) or "none"


def diff_outside(old: np.ndarray, new: np.ndarray, boxes, pad: int = 3) -> dict:
    """Pixels that differ between two images of the same size, and how many of them lie outside the boxes."""
    if old.shape != new.shape:
        return {"same_size": False, "n_diff": -1, "n_outside": -1, "outside_box": ""}
    d = (old != new).any(axis=2)
    allowed = np.zeros(d.shape, dtype=bool)
    for x0, y0, x1, y1 in boxes:
        allowed[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    out = d & ~allowed
    ys, xs = np.nonzero(out)
    where = f"x {xs.min()}-{xs.max()}, y {ys.min()}-{ys.max()}" if len(ys) else ""
    return {"same_size": True, "n_diff": int(d.sum()), "n_outside": int(out.sum()), "outside_box": where,
            "n_allowed": int(allowed.sum())}


def text_layout(fig, texts) -> dict:
    """Font sizes, font files, overlaps and texts outside the figure (as atlas_dct_lib.text_checks) for ``texts``."""
    from matplotlib import font_manager

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes, sizes, fonts, outside = [], [], set(), []
    for t in texts:
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width == 0:
            continue
        boxes.append((t.get_text(), bb))
        sizes.append(t.get_fontsize())
        fonts.add(Path(font_manager.findfont(t.get_fontproperties())).name.lower())
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            outside.append(t.get_text()[:30])
    shrink = 0.05 / 25.4 * fig.dpi
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > shrink and min(a.y1, b.y1) - max(a.y0, b.y0) > shrink:
                overlaps.append(f"{boxes[i][0]!r}/{boxes[j][0]!r}")
    return {"n_texts": len(boxes), "min_font_pt": min(sizes) if sizes else np.nan, "fonts": ";".join(sorted(fonts)),
            "n_text_overlaps": len(overlaps), "overlaps": "; ".join(overlaps[:8]),
            "n_text_outside_figure": len(outside), "outside": "; ".join(outside[:8])}


# ---------------------------------------------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------------------------------------------
def pdf_facts(path: Path) -> dict:
    """Pages, page size (mm), fonts (name, embedded), raster images and text lines (white space normalized) of a PDF."""
    reader = PdfReader(str(path))
    page = reader.pages[0]
    lines = [" ".join(line.split()) for line in (page.extract_text() or "").splitlines()]
    return {"pages": len(reader.pages), "width_mm": float(page.mediabox.width) / PT_PER_MM,
            "height_mm": float(page.mediabox.height) / PT_PER_MM,
            "fonts": [(name.split("+")[-1], emb) for name, emb in c2.page_fonts(page)],
            "n_images": c2.count_images(page), "lines": [line for line in lines if line]}


def line_changes(old: list[str], new: list[str]) -> tuple[list[str], list[str]]:
    """Lines only in the old text and lines only in the new text (with multiplicity, in order of appearance)."""
    def only(a, b):
        left = Counter(a) - Counter(b)
        out = []
        for line in a:
            if left[line] > 0:
                out.append(line)
                left[line] -= 1
        return out
    return only(old, new), only(new, old)


def add_text_checks(checks, label: str, old_lines: list[str], new_lines: list[str], expected: list[tuple[str, str]],
                    old_name: str) -> None:
    """Check rows: the PDF text differs from the old one by the expected lines only; the journal's number and P rules.

    ``expected``: (old line, new line); an empty old line stands for a line that is new (panel letters of a figure
    compiled for the first time)."""
    removed, added = line_changes(old_lines, new_lines)
    ok = (Counter(removed) == Counter(o for o, _ in expected if o)
          and Counter(added) == Counter(n for _, n in expected if n))
    pairs = "; ".join(f"{o!r} -> {n!r}" if o else f"new line {n!r}" for o, n in expected)
    extra = ""
    if not ok:
        extra = f" | OBSERVED only in the old text: {removed} | only in the new text: {added}"
    checks.add(f"{label}: text lines of the new PDF differ from {old_name} by the listed changes only (pypdf; "
               f"{len(old_lines)} old and {len(new_lines)} new lines)",
               f"{len(removed)} lines only in the old text, {len(added)} only in the new: {pairs}{extra}", ok,
               f"{len(expected)} changed lines")
    four = [line for line in new_lines if FOUR_DIGITS_COMMA.search(line)]
    checks.add(f"{label}: no number of four digits with a comma in the PDF text", f"{len(four)} lines"
               + (f": {four}" if four else ""), not four, "0")
    five = sorted({m.group(0) for line in new_lines for m in FIVE_DIGITS_COMMA.finditer(line)})
    checks.add(f"{label}: numbers of five or more digits written with a comma (information)",
               ", ".join(five) or "none in this figure", None)
    low = [line for line in new_lines if LOWER_P.search(line)]
    checks.add(f"{label}: no lower-case 'p =' in the PDF text", f"{len(low)} lines" + (f": {low}" if low else ""),
               not low, "0")


def add_file_checks(checks, label: str, *, pdf: Path, png: Path, tif: Path | None, size_mm: tuple[float, float],
                    png_px: tuple[int, int], tif_px: tuple[int, int] | None, size_from: str) -> dict:
    """Check rows on the written files: page size, fonts, resolution, pixel size, TIFF mode and compression."""
    f = pdf_facts(pdf)
    checks.add(f"{label}: PDF is one page of {size_from} (mm)",
               f"{f['pages']} page; {f['width_mm']:.2f} x {f['height_mm']:.2f}",
               f["pages"] == 1 and abs(f["width_mm"] - size_mm[0]) < 0.02 and abs(f["height_mm"] - size_mm[1]) < 0.02,
               f"{size_mm[0]:.2f} x {size_mm[1]:.2f}")
    names = sorted({n for n, _ in f["fonts"]})
    checks.add(f"{label}: every font of the PDF is embedded and of the Arial family",
               f"{sum(e for _, e in f['fonts'])} of {len(f['fonts'])} embedded; {', '.join(names)}",
               bool(f["fonts"]) and all(e for _, e in f["fonts"]) and all(n.startswith("Arial") for n in names),
               "all embedded; Arial only")
    _, p = read_image(png)
    checks.add(f"{label}: PNG pixel size and resolution",
               f"{p['width_px']} x {p['height_px']} px; {p['dpi']:.1f} dpi",
               (p["width_px"], p["height_px"]) == tuple(png_px) and abs(p["dpi"] - DPI_PNG) < 0.5,
               f"{png_px[0]} x {png_px[1]} px; 600 dpi")
    out = {"pdf": f, "png": p}
    if tif is not None:
        _, t = read_image(tif)
        checks.add(f"{label}: TIFF pixel size, resolution tag, mode and compression",
                   f"{t['width_px']} x {t['height_px']} px; {t['dpi']:.1f} dpi; {t['mode']}; {t['compression']}",
                   (t["width_px"], t["height_px"]) == tuple(tif_px) and abs(t["dpi"] - DPI_TIF) < 0.5
                   and t["mode"] == "RGB" and t["compression"] == "tiff_lzw",
                   f"{tif_px[0]} x {tif_px[1]} px; 1000 dpi; RGB; tiff_lzw")
        checks.add(f"{label}: TIFF has no alpha channel", f"bands {t['bands']}", t["bands"] == "RGB", "RGB")
        w_mm, h_mm = mm(t["width_px"], DPI_TIF), mm(t["height_px"], DPI_TIF)
        checks.add(f"{label}: TIFF size in mm at its resolution against the PDF page",
                   f"{w_mm:.2f} x {h_mm:.2f} mm; PDF {f['width_mm']:.2f} x {f['height_mm']:.2f} mm",
                   abs(w_mm - f["width_mm"]) < 0.2 and abs(h_mm - f["height_mm"]) < 0.2, "within 0.2 mm")
        out["tif"] = t
    return out


def _block_means(tif_rgb: np.ndarray, png_rgb: np.ndarray, per_inch: int) -> tuple[np.ndarray, np.ndarray]:
    """Grey block averages of the two images on the same grid (blocks of 25.4 / per_inch mm)."""
    kt, kp = DPI_TIF // per_inch, DPI_PNG // per_inch
    h = min(tif_rgb.shape[0] // kt, png_rgb.shape[0] // kp)
    w = min(tif_rgb.shape[1] // kt, png_rgb.shape[1] // kp)
    a = Image.fromarray(tif_rgb).reduce(kt, box=(0, 0, w * kt, h * kt)).convert("L")
    b = Image.fromarray(png_rgb).reduce(kp, box=(0, 0, w * kp, h * kp)).convert("L")
    return np.asarray(a).astype(np.float32), np.asarray(b).astype(np.float32)


def add_tif_png_check(checks, label: str, tif_rgb: np.ndarray, png_rgb: np.ndarray) -> None:
    """The 1000-dpi image and the 600-dpi image show the same figure at the same place.

    Both are averaged over blocks on one grid. (1) Correlation of the 1.27 mm block averages. (2) With 0.254 mm
    blocks, the mean absolute difference is smallest when the two images are not shifted against each other (shifts of
    one block in the eight directions are tried). The two drawings differ in their pixels (text and thin lines are
    fitted to the pixel grid of each resolution), so no equality is expected. The averages are used for nothing else.
    """
    a, b = _block_means(tif_rgb, png_rgb, 20)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    a, b = _block_means(tif_rgb, png_rgb, 100)
    h, w = a.shape
    shift = {(dy, dx): float(np.abs(a[1:h - 1, 1:w - 1] - b[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]).mean())
             for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    best = min(shift, key=shift.get)
    other = min(v for k, v in shift.items() if k != (0, 0))
    ink_t = float((tif_rgb.min(axis=2) < 128).mean())
    ink_p = float((png_rgb.min(axis=2) < 128).mean())
    checks.add(f"{label}: the 1000-dpi TIFF and the 600-dpi PNG show the same figure at the same place (block "
               f"averages on one grid; correlation at 1.27 mm; mean absolute difference at 0.254 mm without a shift "
               f"and the smallest with a shift of one block; share of dark pixels TIFF / PNG)",
               f"r = {corr:.4f}; {shift[(0, 0)]:.2f} against {other:.2f}; {100 * ink_t:.2f}% / {100 * ink_p:.2f}%",
               corr > 0.98 and best == (0, 0), "r > 0.98; smallest without a shift")


# ---------------------------------------------------------------------------------------------------------------------
# Compiled figures (rules of report_figure2_layout.py)
# ---------------------------------------------------------------------------------------------------------------------
def compose(panels: dict[str, np.ndarray], pos: dict[str, tuple[int, int]], width: int, height: int) -> np.ndarray:
    """White page with every panel at its position, pixel for pixel."""
    page = np.full((height, width, 3), 255, dtype=np.uint8)
    for key, (x, y) in pos.items():
        a = panels[key]
        page[y:y + a.shape[0], x:x + a.shape[1]] = a
    return page


def layout_problems(pos: dict, w: dict, h: dict, width, height) -> tuple[bool, list[str]]:
    """Every panel inside the page; pairs of panels that overlap."""
    inside = all(x >= 0 and y >= 0 and x + w[k] <= width + 1e-6 and y + h[k] <= height + 1e-6
                 for k, (x, y) in pos.items())
    keys, overlaps = list(pos), []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            (xa, ya), (xb, yb) = pos[a], pos[b]
            if xa < xb + w[b] and xb < xa + w[a] and ya < yb + h[b] and yb < ya + h[a]:
                overlaps.append(f"{a}/{b}")
    return inside, overlaps


def letter_font(dpi: int):
    return ImageFont.truetype(str(ARIAL_BOLD), int(round(LETTER_PT * dpi / 72)))


def letter_box(letter: str, cell_xy: tuple[int, int], dpi: int) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Anchor of a letter (left, ascender line) for a panel cell, and its ink box with the blank rim around it."""
    at = (cell_xy[0] + px(LETTER_OFFSET_MM[0], dpi), cell_xy[1] + px(LETTER_OFFSET_MM[1], dpi))
    draw = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    b = draw.textbbox(at, letter, font=letter_font(dpi), anchor="lt")
    pad = px(LETTER_PAD_MM, dpi)
    return at, (max(b[0] - pad, 0), max(b[1] - pad, 0), b[2] + pad, b[3] + pad)


def letters_blank(page: np.ndarray, letters: list[tuple[str, tuple[int, int]]], dpi: int) -> dict[str, bool]:
    """Is the page blank (all channels >= 250) where each letter and its rim would go?"""
    out = {}
    for letter, cell in letters:
        _, (x0, y0, x1, y1) = letter_box(letter, cell, dpi)
        out[letter] = bool((page[y0:y1, x0:x1] >= 250).all())
    return out


def draw_letters(page: np.ndarray, letters: list[tuple[str, tuple[int, int]]], dpi: int):
    """Write the letters on the page (Pillow, Arial Bold); returns the page as an array and the box of each letter."""
    im = Image.fromarray(page)
    draw = ImageDraw.Draw(im)
    font = letter_font(dpi)
    boxes = {}
    for letter, cell in letters:
        at, box = letter_box(letter, cell, dpi)
        draw.text(at, letter, font=font, fill=(0, 0, 0), anchor="lt")
        boxes[letter] = box
    return np.asarray(im), boxes


def reading_order(letters: list[tuple[str, tuple[int, int]]]) -> str:
    return "".join(letter for _, _, letter in sorted((y, x, letter) for letter, (x, y) in letters))


def merge_pdf(out: Path, placed: list[tuple[Path, float, float]], width_pt: float, height_pt: float,
              letters: list[tuple[str, tuple[float, float]]], title: str) -> dict:
    """Panel PDFs merged on one page (vector; fonts stay embedded) and the letters written as Arial Bold text.

    ``placed``: (panel PDF, x from the left, y from the top) in points; ``letters``: (letter, top-left corner of its
    panel cell in points). The letter baseline lies one font ascent (Pillow metrics of the 600-dpi font, as the
    image) below the offset from the top of the cell. Returns the number of raster images in the panels and on the
    page.
    """
    ascent_pt = letter_font(DPI_PNG).getmetrics()[0] * 72 / DPI_PNG
    writer = PdfWriter()
    page = writer.add_blank_page(width=width_pt, height=height_pt)
    readers, n_src = [], 0
    for path, x, y in placed:
        reader = PdfReader(str(path))
        readers.append(reader)
        p = reader.pages[0]
        plain = (len(reader.pages) == 1 and float(p.mediabox.left) == 0 and float(p.mediabox.bottom) == 0
                 and p.get("/Rotate") in (None, 0) and p.cropbox == p.mediabox)
        if not plain:
            raise SystemExit(f"{cc.rel(path)}: not one unrotated page with its origin at 0, 0")
        n_src += c2.count_images(p)
        page.merge_transformed_page(p, Transformation().translate(tx=x, ty=height_pt - y - float(p.mediabox.height)))
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig = plt.figure(figsize=(width_pt / 72, height_pt / 72))
        for letter, (x, y) in letters:
            base = y + LETTER_OFFSET_MM[1] * PT_PER_MM + ascent_pt             # baseline, from the top of the page
            fig.text((x + LETTER_OFFSET_MM[0] * PT_PER_MM) / width_pt, 1 - base / height_pt, letter, ha="left",
                     va="baseline", fontsize=LETTER_PT, fontweight="bold", fontfamily="Arial", color="black")
        buf = BytesIO()
        fig.savefig(buf, format="pdf", transparent=True)
        plt.close(fig)
    buf.seek(0)
    overlay = PdfReader(buf)
    page.merge_page(overlay.pages[0])
    writer.add_metadata({"/Title": title, "/Creator": f"{cc.GENERATED_BY}; {_script()} (pypdf)"})
    with open(guard(out), "wb") as fh:
        writer.write(fh)
    print(f"Saved PDF: {cc.rel(out)}")
    buf.seek(0)
    letters_text = "".join(ch for ch in (PdfReader(buf).pages[0].extract_text() or "") if ch.isalpha())
    return {"n_images_panels": n_src, "n_images_page": c2.count_images(PdfReader(str(out)).pages[0]),
            "letters_text": letters_text}
