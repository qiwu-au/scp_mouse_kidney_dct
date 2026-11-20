#!/usr/bin/env python
"""The response to the reviewers: shared helpers of the the response to the reviewers figures (Editors' comment 1, RNA comparator for DCT).

(12 Nov 2026). Used by report_dct_dotplot_protein_rna.py,
report_dct_rna_markers_figure.py and report_dct_rna_mapping_figure.py. Drawing helpers only: every plotted value is read
from a stored table by the calling script. This module writes nothing by itself.

- FixedPanel: the dotplot / kidney_rna dot-plot panel with a fixed row-label width, so that stacked panels share one grid and
  their titles start at the same left edge.
- stacked_below / draw_stack: panels stacked with the same columns; the size and colour legends below or beside.
- rna_panel_values: the RNA rows of a dot plot from the stored dct_rna table dct_rna_dotplot_rna_values.csv.
- reader-facing names and the check that no internal name is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import dotplot_dotplots as t6d
import kidney_rna_dotplots as d7
import atlas_dct_lib as t8

UNRESOLVED = "DCT, subtype not resolved"            # display name of the dct_labels label value 'Unknown' (authors, 12 Nov 2026)
UNRESOLVED_SHORT = "not resolved"
ATLAS_ROWS = ("DCT", "DCT-CNT", "CNT", "CTAL", "MTAL")           # atlas level Celltype (authors, atlas_dct / pt_tal / the response to the reviewers)
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
RNA_TITLE = {"all": "RNA (Mouse Kidney Atlas, all cells and nuclei)"}
LABEL_COLOURS = {"DCT1": "#1f77b4", "DCT2": "#ff7f0e", "ProLIF": "#2ca02c", "Unknown": "#B3B3B3"}   # Figure 4e
# Internal names that must not be drawn (reader-facing figures; authors, 12 Nov 2026).
BANNED = ("ctrl", "zerok", "unknown", "t08", "t27", "t28", "t29", "cs_", "primary", "second", "published", "derived",
          "generated", ".csv", "submitted", "re-annotation", "does not apply")


@dataclass
class FixedPanel(d7.GreyPanel):
    """GreyPanel whose row-label width can be set, so that stacked panels share the grid and the title edge."""

    label_w: float = 0.0

    def natural_label_w(self) -> float:
        return d7.GreyPanel.row_label_w(self)

    def row_label_w(self) -> float:
        return self.label_w or self.natural_label_w()


def marker_groups(names: list[str]) -> list[tuple[str, int, int]]:
    """(group, first column, last column) for consecutive columns of the same marker group."""
    out: list[tuple[str, int, int]] = []
    for j, g in enumerate(names):
        if out and out[-1][0] == g:
            out[-1] = (g, out[-1][1], j)
        else:
            out.append((g, j, j))
    return out


def rna_panel_values(rv: pd.DataFrame, markers: list[str], level: str, source: str, rows) -> dict:
    """RNA rows of a dot plot from the stored dct_rna values (fraction with counts, colour scaled 0-1 per gene)."""
    s = rv[(rv["level"] == level) & (rv["rna_source"] == source) & rv["marker"].isin(markers)]
    frac = s.pivot(index="group", columns="marker", values="frac_detected").reindex(index=list(rows), columns=markers)
    colour = s.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(index=list(rows),
                                                                                         columns=markers)
    in_atlas = s.drop_duplicates("marker").set_index("marker")["gene_in_atlas"].astype(bool)
    n = s.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(list(rows))
    if n.isna().any() or len(s) != len(rows) * len(markers):
        raise ValueError(f"RNA values missing for {level} / {source}")
    grey = [not bool(in_atlas.get(m, False)) for m in markers]
    return {"frac": frac.fillna(0.0), "colour": colour.fillna(0.0), "n": [int(v) for v in n],
            "colours": [t8.RNA_PALETTE[r] for r in rows], "grey_cols": grey}


def draw_stack(cv: t6d.Canvas, panels: list[FixedPanel], x0: float, y0: float, pitch: float, vgap: float) -> list[dict]:
    """Draw the panels one under the other from (x0, y0); returns their geometry."""
    lw = max(p.natural_label_w() for p in panels)
    geo, y = [], y0
    for p in panels:
        p.label_w = lw
        geo.append(p.draw(cv, x0, y, pitch))
        y += p.size(pitch)[1] + vgap
    return geo


def stack_size(panels: list[FixedPanel], pitch: float, vgap: float) -> tuple[float, float]:
    lw = max(p.natural_label_w() for p in panels)
    for p in panels:
        p.label_w = lw
    sizes = [p.size(pitch) for p in panels]
    return max(w for w, _ in sizes), sum(h for _, h in sizes) + vgap * (len(panels) - 1)


def stacked_below(panels: list[FixedPanel], width: float, p_max: float = 4.5, vgap: float = 2.5):
    """Stacked panels over the full width, legends side by side below (no note). Returns (canvas, pitch)."""
    m = t6d.MARGIN
    ncol = len(panels[0].markers)
    if any(len(p.markers) != ncol for p in panels):
        raise ValueError("stacked panels need the same columns")
    lw = max(p.natural_label_w() for p in panels)
    pitch = min(p_max, (width - 2 * m - lw) / ncol)
    body_w, body_h = stack_size(panels, pitch, vgap)
    sw, sh, _ = t6d.size_legend_dims(pitch)
    cw, ch = t6d.color_legend_dims()
    leg_w, leg_h = sw + 6.0 + cw, max(sh, ch)
    if max(body_w, leg_w) > width - 2 * m + 1e-6:
        raise ValueError(f"content {max(body_w, leg_w):.1f} mm wider than {width} mm")
    cv = t6d.Canvas(width, 2 * m + body_h + 2.5 + leg_h)
    x0 = (width - body_w) / 2
    geo = draw_stack(cv, panels, x0, m, pitch, vgap)
    t6d.draw_legends(cv, x0, geo[-1]["bottom"] + 2.5, pitch, stacked=False)
    return cv, pitch


def banned_texts(texts, banned=BANNED) -> list[str]:
    return [t.get_text() for t in texts if t.get_text().strip()
            and any(b in t.get_text().lower() for b in banned)]


def panel_letter(cv: t6d.Canvas, x: float, y: float, letter: str) -> None:
    cv.text(x, y, letter, 8.5, fontweight="bold", ha="left", va="top")


def heat_grid(cv: t6d.Canvas, x: float, y: float, cw: float, ch: float, values: np.ndarray, cmap, vmin: float,
              vmax: float, texts: list[list[str]]) -> None:
    """A grid of coloured cells with a number in each (mm canvas; x, y = top-left corner of the grid)."""
    nr, nc = values.shape
    for i in range(nr):
        for j in range(nc):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            rgba = cmap((v - vmin) / (vmax - vmin))
            cv.rect(x + j * cw, y + i * ch, cw, ch, facecolor=rgba, edgecolor="white", lw=0.5)
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            cv.text(x + (j + 0.5) * cw, y + (i + 0.5) * ch, texts[i][j], ha="center", va="center_baseline",
                    color="white" if lum < 0.45 else t6d.INK["primary"])


def colour_bar(cv: t6d.Canvas, x: float, y: float, w: float, h: float, cmap, ticks, labels, title: str) -> None:
    """Horizontal colour bar with ticks below and the title to its right (mm canvas)."""
    grad = np.linspace(0, 1, 256)[None, :]
    cv.ax.imshow(grad, cmap=cmap, aspect="auto", extent=(x, x + w, y + h, y), interpolation="bilinear", zorder=2)
    cv.ax.set_xlim(0, cv.W)
    cv.ax.set_ylim(cv.H, 0)
    cv.rect(x, y, w, h, fill=False, lw=0.4, edgecolor=t6d.FRAME)
    for frac, lab in zip(ticks, labels):
        xv = x + frac * w
        cv.line([xv, xv], [y + h, y + h + 0.5], lw=0.4)
        cv.text(xv, y + h + 0.8, lab, ha="center", va="top")
    cv.text(x + w + 2.0, y + h / 2, title, ha="left", va="center_baseline")


def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95% interval (as atlas_dct_lib.wilson; used only to check stored intervals)."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
