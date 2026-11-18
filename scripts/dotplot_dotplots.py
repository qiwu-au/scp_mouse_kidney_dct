#!/usr/bin/env python
"""dotplot step 2: marker dotplots (Fig 2C and Fig 3B replacements), laid out like SISKA 1.0 Fig 1c.

Rules: ``dotplot_lib.py``; values from step 1.

Encoding (as SISKA Fig 1c): dot area = fraction of the group's cells with intensity > 0; colour = group
mean of the per-protein z-scored ln(1 + intensity) (canonical ``.X``), rescaled 0-1 per protein across the
rows shown (white -> dark blue). Rows top to bottom follow the nephron; marker groups follow the rows, so the
plot reads as a diagonal; thin brackets above the columns name each group.

Format: exact figure width (180 mm; DCT panel also 85 mm), Arial, every text >= 7 pt, vector PDF (TrueType
fonts embedded) plus 600-dpi PNG and LZW TIFF. Every figure is laid out on a millimetre canvas; the script
checks font sizes, font family, text overlap, text inside the figure and the saved file dimensions.

Inputs (read-only; results/tables/dotplot/): label_order, marker_order_kidney, marker_order_dct,
  kidney_dotplot_values, dct_dotplot_values, kidney_marker_coverage, kidney_main_selection
Outputs:
  results/figures/dotplot/dotplot_kidney_dotplot_{main,full}_{reannot,pre}.pdf/.png/.tif
  results/figures/dotplot/dotplot_dct_dotplot_panel_180mm.pdf/.png/.tif         (both V1 label columns)
  results/figures/dotplot/dotplot_dct_dotplot_panel_{v1_cell_label,v1_cluster_label}_85mm.*
  results/figures/dotplot/dotplot_dct_dotplot_workbook_180mm.*                  (V1 workbook markers)
  results/tables/dotplot/dotplot_figure_checks.csv, dotplot_figure_captions.md
  results/tables/dotplot/dotplot_step_dotplots_manifest.json

Command (project root):
  python scripts/dotplot_dotplots.py
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import qc_mito_lib as t3
import dotplot_lib as t6

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none"})

TASK = t6.TASK
SCRIPT = "dotplot_dotplots.py"
DPI = 600
PT = 72 / 25.4                       # points per mm
FONT = 7.0                           # every text (journal minimum 7 pt at final size)
TITLE_FONT = 7.5
INK = t3.INK
EDGE = "#9a9891"                     # dot outline (keeps the lightest dots visible)
LEGEND_DOT = "#b5b4ad"               # neutral grey for the size legend (size carries no colour value)
FRAME = INK["muted"]
SEQUENTIAL = LinearSegmentedColormap.from_list("t06_blue", ["#eef4fc", *t3.BLUE_RAMP])
DOT_SCALE = 0.9                      # dot diameter at 100% detected, as a fraction of the cell pitch
SIZE_STEPS = (0.2, 0.4, 0.6, 0.8, 1.0)
MARGIN = 1.5                         # mm around the content
CHIP_W = 1.6                         # mm, row colour chip
SIZE_TITLE = "Fraction of cells with\nintensity > 0 (%)"
COLOR_TITLE = "Mean in group\n(scaled 0–1 per protein)"
DAGGER = "†"
REPEAT_NOTE = (f"{DAGGER} Shown so that each of the 14 re-annotated cell types has a column: a marker listed for "
               "this cell type whose group mean is higher in another cell type that lists it.")


# ---------------------------------------------------------------------------
# Text measurement (mm at 600 dpi)
# ---------------------------------------------------------------------------
_MEASURE = {}


def text_mm(s: str, size: float = FONT, weight: str = "normal", rotation: float = 0) -> tuple[float, float]:
    if "fig" not in _MEASURE:
        _MEASURE["fig"] = plt.figure(dpi=DPI)
        _MEASURE["r"] = _MEASURE["fig"].canvas.get_renderer()
    t = _MEASURE["fig"].text(0, 0, s, fontsize=size, fontweight=weight, rotation=rotation)
    bb = t.get_window_extent(_MEASURE["r"])
    t.remove()
    return bb.width / DPI * 25.4, bb.height / DPI * 25.4


LINE_H = text_mm("Hg")[1]


# ---------------------------------------------------------------------------
# Millimetre canvas
# ---------------------------------------------------------------------------
class Canvas:
    """A figure of exact size with one overlay axes whose data units are mm (y downwards)."""

    def __init__(self, width: float, height: float):
        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=DPI)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.set_axis_off()
        self.texts = []

    def text(self, x, y, s, size=FONT, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=kw.pop("color", INK["primary"]), clip_on=False, **kw)
        self.texts.append(t)
        return t

    def line(self, xs, ys, lw=0.5, color=FRAME):
        self.ax.plot(xs, ys, lw=lw, color=color, solid_capstyle="butt", clip_on=False)

    def rect(self, x, y, w, h, **kw):
        self.ax.add_patch(Rectangle((x, y), w, h, clip_on=False, **kw))

    def dots(self, xs, ys, frac, color, pitch):
        """Dots of area proportional to ``frac``; ``color`` is 0-1 values (colour map) or one fixed colour."""
        frac = np.asarray(frac, float)
        keep = frac > 0
        size = (DOT_SCALE * pitch * PT * np.sqrt(frac[keep])) ** 2
        paint = {"color": color} if isinstance(color, str) else {
            "c": np.asarray(color)[keep], "cmap": SEQUENTIAL, "vmin": 0, "vmax": 1}
        self.ax.scatter(np.asarray(xs)[keep], np.asarray(ys)[keep], s=size, edgecolors=EDGE, linewidths=0.3,
                        clip_on=False, zorder=3, **paint)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
@dataclass
class Panel:
    title: str
    rows: list[str]
    row_n: list[int]
    row_colors: list[str]
    markers: list[str]
    groups: list[tuple[str, int, int]]      # (label, first column, last column)
    frac: np.ndarray                        # rows x markers
    color: np.ndarray                       # rows x markers, 0-1
    repeat: list[bool]                      # column repeated so that its row type has a column (dagger)

    def row_labels(self) -> list[str]:
        return [f"{r} ({n})" for r, n in zip(self.rows, self.row_n)]

    def col_labels(self) -> list[str]:
        return [m + (DAGGER if r else "") for m, r in zip(self.markers, self.repeat)]

    def row_label_w(self) -> float:
        return max(text_mm(s)[0] for s in self.row_labels()) + 0.8 + CHIP_W + 0.8 + 0.5

    def gene_h(self) -> float:
        return max(text_mm(m)[0] for m in self.col_labels()) + 0.5 + 0.8

    def bracket_levels(self, pitch: float) -> list[int]:
        levels, right = [], {}
        for label, j0, j1 in self.groups:
            c = (j0 + j1 + 1) / 2 * pitch
            w = text_mm(label)[0]
            lvl = 0
            while right.get(lvl, -np.inf) + 0.8 > c - w / 2:
                lvl += 1
            right[lvl] = c + w / 2
            levels.append(lvl)
        return levels

    def bracket_h(self, pitch: float) -> float:
        n = max(self.bracket_levels(pitch), default=0) + 1
        return 1.3 + n * (LINE_H + 0.3)

    def title_h(self) -> float:
        return text_mm(self.title, TITLE_FONT, "bold")[1] + 0.8

    def size(self, pitch: float) -> tuple[float, float]:
        w = self.row_label_w() + len(self.markers) * pitch
        h = self.title_h() + self.bracket_h(pitch) + len(self.rows) * pitch + self.gene_h()
        return w, h

    def draw(self, cv: Canvas, x0: float, y0: float, pitch: float) -> dict[str, float]:
        gl = x0 + self.row_label_w()
        gt = y0 + self.title_h() + self.bracket_h(pitch)
        nr, nc = len(self.rows), len(self.markers)
        cv.text(x0, y0, self.title, TITLE_FONT, fontweight="bold", ha="left", va="top")
        # brackets and group labels
        for (label, j0, j1), lvl in zip(self.groups, self.bracket_levels(pitch)):
            xa, xb = gl + j0 * pitch + 0.25 * pitch, gl + (j1 + 1) * pitch - 0.25 * pitch
            yb = gt - 1.0
            cv.line([xa, xa, xb, xb], [gt - 0.35, yb, yb, gt - 0.35], lw=0.5, color=INK["secondary"])
            cv.text((xa + xb) / 2, yb - 0.3 - lvl * (LINE_H + 0.3), label, ha="center", va="bottom",
                    color=INK["primary"])
        # frame, ticks, labels
        cv.rect(gl, gt, nc * pitch, nr * pitch, fill=False, lw=0.5, edgecolor=FRAME)
        for i, (lab, col) in enumerate(zip(self.row_labels(), self.row_colors)):
            yc = gt + (i + 0.5) * pitch
            cv.line([gl - 0.5, gl], [yc, yc], lw=0.4)
            chip = min(CHIP_W, 0.6 * pitch)
            cv.rect(gl - 0.5 - 0.8 - CHIP_W, yc - chip / 2, CHIP_W, chip, facecolor=col or INK["muted"],
                    edgecolor="none")
            cv.text(gl - 0.5 - 0.8 - CHIP_W - 0.8, yc, lab, ha="right", va="center_baseline")
        gb = gt + nr * pitch
        for j, m in enumerate(self.col_labels()):
            xc = gl + (j + 0.5) * pitch
            cv.line([xc, xc], [gb, gb + 0.5], lw=0.4)
            cv.text(xc, gb + 0.5 + 0.8, m, rotation=90, ha="center", va="top", rotation_mode="default")
        jj, ii = np.meshgrid(np.arange(nc), np.arange(nr))
        cv.dots(gl + (jj.ravel() + 0.5) * pitch, gt + (ii.ravel() + 0.5) * pitch, self.frac.ravel(),
                self.color.ravel(), pitch)
        return {"grid_left": gl, "grid_top": gt, "grid_right": gl + nc * pitch, "grid_bottom": gb,
                "bottom": gb + self.gene_h()}


# ---------------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------------
def size_legend_dims(pitch: float) -> tuple[float, float, float]:
    dmax = DOT_SCALE * pitch
    step = max(dmax + 1.2, text_mm("100")[0] + 1.2)
    w = max(len(SIZE_STEPS) * step, max(text_mm(s)[0] for s in SIZE_TITLE.split("\n")))
    h = 2 * LINE_H + 0.8 + dmax + 0.6 + LINE_H
    return w, h, step


def color_legend_dims() -> tuple[float, float]:
    w = max(24.0, max(text_mm(s)[0] for s in COLOR_TITLE.split("\n")))
    return w, 2 * LINE_H + 0.8 + 2.2 + 0.6 + LINE_H


def draw_legends(cv: Canvas, x: float, y: float, pitch: float, stacked: bool) -> tuple[float, float]:
    """Size legend then colour legend; stacked (one above the other) or side by side. Returns (w, h)."""
    sw, sh, step = size_legend_dims(pitch)
    cw, ch = color_legend_dims()
    dmax = DOT_SCALE * pitch
    cv.text(x, y, SIZE_TITLE, ha="left", va="top", linespacing=1.15)
    ty = y + 2 * LINE_H + 0.8 + dmax / 2
    xs = x + step / 2 + np.arange(len(SIZE_STEPS)) * step
    cv.dots(xs, np.full(len(xs), ty), np.array(SIZE_STEPS), LEGEND_DOT, pitch)
    for xi, f in zip(xs, SIZE_STEPS):
        cv.text(xi, ty + dmax / 2 + 0.6, f"{int(round(f * 100))}", ha="center", va="top")
    cx, cy = (x, y + sh + 3.0) if stacked else (x + sw + 6.0, y)
    cv.text(cx, cy, COLOR_TITLE, ha="left", va="top", linespacing=1.15)
    by = cy + 2 * LINE_H + 0.8
    bw = 22.0
    grad = np.linspace(0, 1, 256)[None, :]
    cv.ax.imshow(grad, cmap=SEQUENTIAL, aspect="auto", extent=(cx, cx + bw, by + 2.2, by), interpolation="bilinear",
                 zorder=2)
    cv.ax.set_xlim(0, cv.W)
    cv.ax.set_ylim(cv.H, 0)
    cv.rect(cx, by, bw, 2.2, fill=False, lw=0.4, edgecolor=FRAME)
    for v in (0, 0.5, 1):
        xv = cx + v * bw
        cv.line([xv, xv], [by + 2.2, by + 2.7], lw=0.4)
        cv.text(xv, by + 2.7 + 0.3, f"{v:g}", ha="center", va="top")
    if stacked:
        return max(sw, cw), sh + 3.0 + ch
    return sw + 6.0 + cw, max(sh, ch)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def solve_pitch(panels: list[Panel], width: float, legend: str, p_max: float, gap: float = 8.0) -> float:
    lw = max(size_legend_dims(p_max)[0], color_legend_dims()[0]) if legend == "right" else 0.0
    fixed = 2 * MARGIN + sum(p.row_label_w() for p in panels) + gap * (len(panels) - 1) + (
        5.0 + lw if legend == "right" else 0.0)
    cols = sum(len(p.markers) for p in panels)
    return min(p_max, (width - fixed) / cols)


def wrap_mm(text: str, width: float) -> list[str]:
    """Greedy word wrap so that each line fits ``width`` mm at the figure font size."""
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_mm(trial)[0] > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def build_figure(panels: list[Panel], width: float, legend: str, p_max: float, gap: float = 8.0):
    pitch = solve_pitch(panels, width, legend, p_max, gap)
    note = wrap_mm(REPEAT_NOTE, width - 2 * MARGIN) if any(any(p.repeat) for p in panels) else []
    note_h = (1.5 + len(note) * LINE_H * 1.15) if note else 0.0
    sizes = [p.size(pitch) for p in panels]
    sw, sh, _ = size_legend_dims(pitch)
    cw, ch = color_legend_dims()
    if legend == "right":
        leg_w, leg_h = max(sw, cw), sh + 3.0 + ch
        content_w = sum(s[0] for s in sizes) + gap * (len(panels) - 1) + 5.0 + leg_w
        top_offset = max(p.title_h() + p.bracket_h(pitch) for p in panels)
        height = 2 * MARGIN + max(max(s[1] for s in sizes), top_offset + leg_h)
    else:
        leg_w, leg_h = sw + 6.0 + cw, max(sh, ch)
        content_w = max(sum(s[0] for s in sizes) + gap * (len(panels) - 1), leg_w)
        height = 2 * MARGIN + max(s[1] for s in sizes) + 2.5 + leg_h
    if content_w > width - 2 * MARGIN + 1e-6:
        raise ValueError(f"content {content_w:.1f} mm wider than {width} mm")
    cv = Canvas(width, height + note_h)
    x0 = x = (width - content_w) / 2
    geo = []
    for p, (w, _) in zip(panels, sizes):
        geo.append(p.draw(cv, x, MARGIN, pitch))
        x += w + gap
    if legend == "right":
        lx = geo[-1]["grid_right"] + 5.0
        draw_legends(cv, lx, geo[-1]["grid_top"], pitch, stacked=True)
    else:
        draw_legends(cv, geo[0]["grid_left"] - panels[0].row_label_w() + 0.0,
                     max(g["bottom"] for g in geo) + 2.5, pitch, stacked=False)
    if note:
        nx = min(x0, (width - max(text_mm(s)[0] for s in note)) / 2)
        cv.text(max(nx, MARGIN), height - MARGIN + 1.5, "\n".join(note), ha="left", va="top", linespacing=1.15,
                color=INK["secondary"])
    return cv, pitch


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def text_checks(cv: Canvas) -> dict:
    cv.fig.canvas.draw()
    r = cv.fig.canvas.get_renderer()
    fig_bb = cv.fig.bbox
    boxes, sizes, fonts, outside = [], [], set(), []
    for t in cv.texts:
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        boxes.append((t.get_text(), bb))
        sizes.append(t.get_fontsize())
        fonts.add(Path(font_manager.findfont(t.get_fontproperties())).name.lower())
        if bb.x0 < fig_bb.x0 - 0.5 or bb.y0 < fig_bb.y0 - 0.5 or bb.x1 > fig_bb.x1 + 0.5 or bb.y1 > fig_bb.y1 + 0.5:
            outside.append(t.get_text())
    shrink = 0.05 / 25.4 * DPI       # 0.05 mm tolerance
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if (min(a.x1, b.x1) - max(a.x0, b.x0) > shrink) and (min(a.y1, b.y1) - max(a.y0, b.y0) > shrink):
                overlaps.append(f"{boxes[i][0]!r}/{boxes[j][0]!r}")
    return {"n_texts": len(boxes), "min_font_pt": min(sizes), "fonts": ";".join(sorted(fonts)),
            "n_text_overlaps": len(overlaps), "overlaps": "; ".join(overlaps[:10]),
            "n_text_outside_figure": len(outside), "outside": "; ".join(outside[:10])}


def file_checks(paths: list[Path], width_mm: float, height_mm: float) -> dict:
    from PIL import Image

    out = {}
    for p in paths:
        if p.suffix == ".png":
            with Image.open(p) as im:
                out.update(png_px=f"{im.size[0]}x{im.size[1]}", png_dpi=str(tuple(round(d) for d in im.info.get("dpi", (0, 0)))),
                           png_width_mm=round(im.size[0] / DPI * 25.4, 2))
        elif p.suffix == ".tif":
            with Image.open(p) as im:
                out.update(tif_px=f"{im.size[0]}x{im.size[1]}", tif_mode=im.mode,
                           tif_dpi=str(tuple(round(float(d)) for d in im.info.get("dpi", (0, 0)))),
                           tif_compression=im.info.get("compression", ""))
        elif p.suffix == ".pdf":
            raw = p.read_bytes()
            m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
            out.update(pdf_width_mm=round(float(m.group(1)) / PT, 2) if m else np.nan,
                       pdf_height_mm=round(float(m.group(2)) / PT, 2) if m else np.nan,
                       pdf_truetype_embedded=b"/FontFile2" in raw, pdf_arial=b"Arial" in raw)
    exp_px = round(width_mm / 25.4 * DPI)
    png_w = int(out.get("png_px", "0x0").split("x")[0])
    out["width_ok"] = bool(abs(png_w - exp_px) <= 1 and abs(out.get("pdf_width_mm", 0) - width_mm) < 0.1)
    return out


# ---------------------------------------------------------------------------
# Data -> panels
# ---------------------------------------------------------------------------
def groups_of(order: pd.DataFrame) -> list[tuple[str, int, int]]:
    out = []
    for j, g in enumerate(order["group"]):
        if out and out[-1][0] == g:
            out[-1] = (g, out[-1][1], j)
        else:
            out.append((g, j, j))
    return out


def make_panel(title, values, label_order, col, order) -> Panel:
    lo = label_order[(label_order["label_column"] == col) & label_order["plotted"]].sort_values("position")
    rows = lo["label"].tolist()
    markers = order["marker"].tolist()       # a repeated marker appears more than once
    v = values[values["label_column"] == col]
    frac = v.pivot(index="group", columns="marker", values="frac_detected").reindex(index=rows, columns=markers)
    colr = v.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=rows, columns=markers)
    if frac.isna().any().any() or colr.isna().any().any():
        raise ValueError(f"missing values for {col}")
    repeat = order["repeat"].fillna(False).astype(bool).tolist() if "repeat" in order else [False] * len(markers)
    return Panel(title, rows, lo["n_cells"].astype(int).tolist(), lo["colour"].fillna("").tolist(), markers,
                 groups_of(order), frac.to_numpy(float), colr.to_numpy(float), repeat)


def plotted_order(order: pd.DataFrame, version: str) -> pd.DataFrame:
    o = order[(order["version"] == version) & order["plotted"]].copy()
    return o.sort_values("position").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------
def kidney_caption_facts(cov: pd.DataFrame, order_k: pd.DataFrame, sel: pd.DataFrame, kvals: pd.DataFrame) -> dict:
    rows = list(t6.KIDNEY_ROW_ORDER)
    c = cov[cov["type_has_cells"] & cov["workbook_position"].notna()]
    unique = c.drop_duplicates("marker")
    missing = unique[~unique["in_protein_matrix"]]
    miss_by_type = c[~c["in_protein_matrix"]].groupby("type")["marker"].apply(lambda s: ", ".join(s))
    full = plotted_order(order_k, "full")
    placed = full[~full["repeat"].astype(bool)]
    reps = full[full["repeat"].astype(bool)]
    v = kvals[kvals["label_column"] == "kidney_reannot_label"].set_index(["group", "marker"])
    n_all = int(v.xs(reps["marker"].iloc[0], level="marker")["n_cells"].sum()) if len(reps) else 0
    rep_txt = []
    for r in reps.itertuples():
        x = v.loc[(r.group, r.marker)]
        tot = int(v.xs(r.marker, level="marker")["n_detected"].sum())
        home = placed.loc[placed["marker"] == r.marker, "group"].iloc[0]
        rep_txt.append(f"{r.group}: {r.marker} (detected in {int(x['n_detected'])} of {int(x['n_cells'])} {r.group} "
                       f"cells; placed under {home}; detected in {tot} of {n_all} cells overall)")
    fallback = sel[sel["selected"] & sel["selection"].str.startswith("fallback")]
    no_cells = cov[~cov["type_has_cells"]].groupby("type")["marker"].apply(lambda s: ", ".join(s))
    return {"n_unique": len(unique), "n_present": int(unique["in_protein_matrix"].sum()), "n_missing": len(missing),
            "miss_by_type": "; ".join(f"{t}: {miss_by_type[t]}" for t in rows if t in miss_by_type.index),
            "repeats": "; ".join(rep_txt), "fallback": fallback, "no_cells": no_cells,
            "n_full_markers": int(placed.shape[0]), "n_full_cols": len(full),
            "n_main_cols": len(plotted_order(order_k, "main"))}


def write_captions(run, facts, figs: dict[str, str], dct_unknown: int) -> None:
    f = facts
    enc = ("Dot size: fraction of the group's cells in which the protein was quantified (intensity > 0). "
           "Colour: group mean of the per-protein z-scored ln(1 + intensity), rescaled from 0 (lowest group) "
           "to 1 (highest group) for each protein across the groups shown. No dot: not quantified in any cell "
           "of the group. Numbers in brackets: cells per group.")
    place = ("Proteins: curated V3 markers (Supplementary Table 1) quantified in the 4,054-protein matrix. "
             "Each protein is placed under the cell type with the highest group mean among the types that "
             "list it as a marker (re-annotated labels); groups follow the row order. Every cell type has at "
             "least one column: † marks a listed marker shown for a cell type whose quantified listed "
             f"markers all have a higher mean in another listing type ({f['repeats']}).")
    fb = "; ".join(f"{r.group}: {r.marker} (no marker quantified in >= 25% of {r.group} cells)"
                   for r in f["fallback"].itertuples())
    L = ["# dotplot draft figure captions", "",
         f"(`scripts/{SCRIPT}`, {cc.now_utc()}). Drafts for the authors; "
         "objective descriptions only. Figure files: `results/figures/dotplot/`.", ""]
    L += ["## Kidney, main version (Fig 2C replacement)", "",
          f"Files: `{figs['kidney_main_reannot']}` (re-annotated labels, Fig 2B; primary) and "
          f"`{figs['kidney_main_pre']}` (pre-annotation labels, Fig 2A; same columns).", "",
          f"Marker proteins per kidney cell type. {enc} {place} Up to three proteins per type are shown, chosen "
          f"among those quantified in >= 25% of the type's cells, ranked by specificity (type mean minus the "
          f"highest mean among the other types). {fb}. "
          f"{f['n_missing']} of the {f['n_unique']} V3 markers listed for these cell types were not quantified "
          f"(Supplementary Table; dotplot coverage table).", "",
          "## Kidney, full version (supplement)", "",
          f"Files: `{figs['kidney_full_reannot']}` and `{figs['kidney_full_pre']}`.", "",
          f"All {f['n_full_markers']} V3 marker proteins quantified for the kidney cell types "
          f"({f['n_full_cols']} columns including the † repeats). {enc} {place} "
          f"Not quantified ({f['n_missing']}): {f['miss_by_type']}. Markers of types "
          f"without cells in this dataset (not shown): "
          + "; ".join(f"{t}: {m}" for t, m in f["no_cells"].items()) + ".", "",
          "## DCT panel (Fig 3B replacement)", "",
          f"Files: `{figs['dct_panel_180mm']}` (both V1 label columns), `{figs['dct_panel_v1_cell_label_85mm']}`, "
          f"`{figs['dct_panel_v1_cluster_label_85mm']}`.", "",
          f"Selected DCT, TAL and cycling marker proteins in the DCT-enriched dataset. {enc} Groups of columns: "
          "core DCT (Slc12a3), DCT1 (Pvalb, Egf), DCT2 (Calb1, Slc8a1, S100g, Klk1), TAL (Umod, Slc12a1), "
          "cycling (Mki67, Top2a). Rows: V1 per-cell labels (Fig 3B, D, G) or V1 cluster labels (Fig 3A); "
          f"the {dct_unknown} cells with the cluster label Unknown are not shown. With three groups per plot, "
          "the 0-1 colour scale stretches small differences between group means (raw means: "
          "`dotplot_dct_dotplot_values.csv`).", "",
          "## DCT, V1 workbook markers (supplement)", "",
          f"File: `{figs['dct_workbook_180mm']}`.", "",
          f"V1 marker proteins (Supplementary Table 1, DCT tab) quantified in the 2,561-protein matrix. {enc} "
          "Each protein is shown once, under the label with the highest group mean among the labels that list "
          "it (V1 per-cell labels). Not quantified: Col5a2, Ptgfr (DCT1), Trpv5 (DCT2), Cenpf (ProLIF).", ""]
    run.text("\n".join(L), "figure_captions", ".md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="dotplot step 2: marker dotplots.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, "dotplots", args)

    def tbl(name):
        return pd.read_csv(run.use_input(t6.table_path(name), f"dotplot step 1: {name}"))

    label_order = tbl("label_order")
    order_k, order_d = tbl("marker_order_kidney"), tbl("marker_order_dct")
    kvals, dvals = tbl("kidney_dotplot_values"), tbl("dct_dotplot_values")
    cov, sel = tbl("kidney_marker_coverage"), tbl("kidney_main_selection")

    specs = []   # (name, panels, width, legend, p_max, description)
    for version, vtxt in (("main", "V3 marker proteins, up to 3 per cell type"),
                          ("full", "all V3 marker proteins quantified")):
        order = plotted_order(order_k, version)
        for col in t6.LABEL_COLUMNS["kidney"]:
            lab = "re-annotated labels (Fig 2B)" if col == "kidney_reannot_label" else "pre-annotation labels (Fig 2A)"
            panel = make_panel(f"Kidney, {lab}: {vtxt}", kvals, label_order, col, order)
            specs.append((f"kidney_dotplot_{version}_{t6.LABEL_SHORT[col]}", [panel], 180.0, "right", 4.5,
                          f"kidney marker dotplot, {version} version, {col}"))
    porder = plotted_order(order_d, "panel")
    wb_order = plotted_order(order_d, "workbook")
    dpanels = {col: make_panel(f"DCT, {t6.LABEL_DISPLAY[col]}", dvals, label_order, col, porder)
               for col in t6.LABEL_COLUMNS["dct"]}
    specs.append(("dct_dotplot_panel_180mm", list(dpanels.values()), 180.0, "right", 4.5,
                  "DCT marker panel (Fig 3B replacement), both V1 label columns"))
    for col, panel in dpanels.items():
        specs.append((f"dct_dotplot_panel_{col}_85mm", [panel], 85.0, "below", 4.5,
                      f"DCT marker panel (Fig 3B replacement), {col}, 85 mm"))
    wpanels = [make_panel(f"DCT, {t6.LABEL_DISPLAY[col]}: V1 workbook markers", dvals, label_order, col, wb_order)
               for col in t6.LABEL_COLUMNS["dct"]]
    specs.append(("dct_dotplot_workbook_180mm", wpanels, 180.0, "right", 4.5,
                  "DCT V1 workbook markers, both V1 label columns"))

    checks, figs = [], {}
    for name, panels, width, legend, p_max, desc in specs:
        cv, pitch = build_figure(panels, width, legend, p_max)
        tc = text_checks(cv)
        paths = run.figure(cv.fig, name, desc, formats=(".pdf", ".png", ".tif"), tight=False)
        fc = file_checks(paths, cv.W, cv.H)
        figs[name] = cc.rel(paths[0].with_suffix(""))
        row = {"figure": name, "width_mm": round(cv.W, 2), "height_mm": round(cv.H, 2), "pitch_mm": round(pitch, 2),
               "n_rows": "+".join(str(len(p.rows)) for p in panels),
               "n_columns": "+".join(str(len(p.markers)) for p in panels), **tc, **fc}
        arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
        row["pass"] = bool(tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0
                           and tc["n_text_outside_figure"] == 0 and fc["width_ok"] and fc.get("tif_mode") == "RGB"
                           and fc.get("pdf_truetype_embedded") and fc.get("pdf_arial"))
        checks.append(row)
        print(f"  {name}: {cv.W:.0f} x {cv.H:.1f} mm, pitch {pitch:.2f} mm, min font {tc['min_font_pt']} pt, "
              f"overlaps {tc['n_text_overlaps']}, outside {tc['n_text_outside_figure']}, pass {row['pass']}")
    run.table(pd.DataFrame(checks), "figure_checks")

    facts = kidney_caption_facts(cov, order_k, sel, kvals)
    dct_unknown = int(label_order.loc[(label_order["label_column"] == "v1_cluster_label")
                                      & ~label_order["plotted"], "n_cells"].sum())
    write_captions(run, facts, {k.replace("dct_dotplot_", "dct_").replace("kidney_dotplot_", "kidney_"): v
                                for k, v in figs.items()}, dct_unknown)
    n_fail = sum(not c["pass"] for c in checks)
    run.note(f"{len(checks)} figures; {n_fail} failed the format checks.")
    run.finish({"dpi": DPI, "font": "Arial", "min_font_pt": FONT, "dot_scale": DOT_SCALE,
                "size_legend_steps": list(SIZE_STEPS), "colormap": ["#eef4fc", *t3.BLUE_RAMP],
                "dot_edge": EDGE, "encoding_reference": "SISKA 1.0 Fig 1c (reviewer's reference, shown by the authors)"},
               status="completed" if n_fail == 0 else "completed with failed format checks")


if __name__ == "__main__":
    main()
