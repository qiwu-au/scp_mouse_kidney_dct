#!/usr/bin/env python
"""kidney_rna step 5b: direct direct RNA vs protein figures under the updated kidney labels.

Requested by the authors on 15 Oct 2026 after seeing the earlier
figures (tables/rna_protein_concordance_mka_scp_minScore0p20_margin0p00_final_featureCoverage/, the later
'_pairedHeatmap_..._v2' heatmap, and integrate_scp_with_mka.py's marker_pseudobulk_heatmaps): the same figure types,
redrawn from the kidney_rna step-5 tables (approved mapping, same detection rule for both modalities) under
kidney_t15_label (primary) and kidney_pre_label. Plan approved by the authors (15 Oct 2026): 180 mm width, threshold 10%,
all atlas cells primary, nuclei-only supplements.

Figures (results/figures/kidney_rna/, PDF + 600-dpi PNG):
1. kidney_rna_paired_top_{t15,pre}_{all,nuclei}: compartments x gene pairs (RNA | protein side by side), each
   gene z-scored within each modality across the compartments (ddof 0, as the earlier analysis); strip = per-gene Spearman rho
   across compartments. Genes (the earlier analysis rule): concordant at 10%; V3 / endothelial-row markers first, then the top 30
   by rho; at most 36, ordered by rho.
2. kidney_rna_paired_markers_{t15,pre}_{all,nuclei}: same layout, every V3 / endothelial-row marker detected in both
   modalities at 10% (any rho), in the dotplot column order (kidney_annotation a_endo / dotplot), grouped by dotplot column group.
3. kidney_rna_paired_marker_rho_{t15,pre}: per V3 / endothelial-row marker (each once, under its first listing
   type), rho across compartments; colour = coverage (in protein matrix and atlas / atlas only / protein only /
   neither); filled = detected in both at 10%; area = mean of the highest RNA and protein detection fractions;
   grey band = central 95% of rho for unrelated rankings (all permutations); dotted lines at 0.2 and 0.5.
4. kidney_rna_paired_compartments_{all,nuclei}: per compartment, rho across features detected in both at 10%
   (bar, n) and across all 3,569 shared features (open circle); kidney_annotation and Fig 2A panels.
5. kidney_rna_paired_marker_heatmaps_{t15,pre}: marker columns (quantified and in the atlas; dotplot order) for
   RNA (all 35 atlas Celltypes; chips = the SCP label each Celltype is mapped to) above protein (SCP labels), each
   marker z-scored within its modality across the rows (as integrate_scp_with_mka.py).
No new statistics beyond the step-5 tables, apart from the permutation band of figure 3.

Inputs (read-only): kidney_rna step-5 tables (concordance_long / features / markers / compartments), kidney_rna mapping table and
atlas pseudobulk, kidney_annotation a_endo and dotplot marker orders, dotplot label order, kidney_object canonical kidney object and kidney_annotation labels.
Outputs: the figures above; results/tables/kidney_rna/kidney_rna_paired_{heatmap_values,marker_rho,compartments,
         marker_heatmap_values,figure_checks,checks}.csv

Command (project root):  python scripts/kidney_rna_paired_figures.py
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
import math
from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import qc_mito_lib as t3
import cluster_de_lib as t5
import cluster_de_figures as t5fig
import dotplot_dotplots as t6dot
import kidney_rna_concordance as c7
import kidney_rna_lib as L
import kidney_annotation_chosen_labels as t15c

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FONT = 7.0
TITLE = 7.5
plt.rcParams.update({"font.family": "Arial", "font.size": FONT, "axes.titlesize": TITLE, "axes.labelsize": FONT,
                     "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

W = 180.0
M = 3.0
THR = 0.10
TOP_N, MAX_GENES = 30, 36            # the earlier analysis rule
MAX_PAIRS_PER_BLOCK = 20
Z_LIM = 2.5
CMAP = t5fig.DIVERGING
INK = t3.INK
SHORT = {"kidney_t15_label": "t15", "kidney_pre_label": "pre"}
DISPLAY = {"kidney_t15_label": "kidney_annotation labels", "kidney_pre_label": "Fig 2A labels"}
SRC_TEXT = {"all": "all atlas cells", "nuclei": "atlas nuclei (snRNA-seq)"}
DOTPLOT_DIR = cc.OUT_ROOTS["tables"] / "dotplot"
MAPPING = cc.out_path(L.TASK, "tables", "celltype_mapping", ".csv")
COVERAGE = {"protein and atlas": t3.SERIES[0], "atlas only (protein not quantified)": t3.SERIES[1],
            "protein only (gene not in atlas)": t3.SERIES[2], "neither": INK["muted"]}
LH = t6dot.text_mm("Hg", FONT)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_rna step 5b: direct RNA vs protein figures.")
    cc.add_common_args(parser)
    return parser.parse_args()


def tw(s: str, size: float = FONT, weight: str = "normal") -> float:
    return t6dot.text_mm(s, size, weight)[0]


def note_block(note: str) -> tuple[str, float]:
    """Wrapped note text and its rendered height (mm) at linespacing 1.15, plus a 1.5 mm gap above."""
    text = "\n".join(t6dot.wrap_mm(note, W - 2 * M))
    fig = plt.figure(dpi=600)
    t = fig.text(0, 0, text, fontsize=FONT, linespacing=1.15)
    h = t.get_window_extent(fig.canvas.get_renderer()).height / 600 * 25.4
    plt.close(fig)
    return text, h + 1.5


def draw_note(sh, text: str, note_h: float) -> None:
    sh.text(M, sh.H - M - note_h + 1.5, text, va="top", linespacing=1.15, color=INK["secondary"])


class Sheet:
    """Figure of exact size W x H mm with helpers that place axes and text in mm (y downwards)."""

    def __init__(self, height: float):
        self.H = height
        self.fig = plt.figure(figsize=(W / 25.4, height / 25.4), dpi=600)   # checks measure at the saved dpi

    def ax(self, x, y, w, h):
        return self.fig.add_axes((x / W, 1 - (y + h) / self.H, w / W, h / self.H))

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FONT)
        kw.setdefault("color", INK["primary"])
        return self.fig.text(x / W, 1 - y / self.H, s, **kw)

    def line(self, xs, ys, **kw):
        kw.setdefault("lw", 0.5)
        kw.setdefault("color", INK["secondary"])
        self.fig.add_artist(matplotlib.lines.Line2D([x / W for x in xs], [1 - y / self.H for y in ys], **kw))

    def rect(self, x, y, w, h, **kw):
        self.fig.add_artist(Rectangle((x / W, 1 - (y + h) / self.H), w / W, h / self.H, **kw))


def zscore(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else np.zeros_like(v)


def text_checks(fig) -> dict:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes, sizes, fonts, outside = [], [], set(), []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width == 0:
            continue
        boxes.append((t.get_text(), bb))
        sizes.append(t.get_fontsize())
        fonts.add(Path(font_manager.findfont(t.get_fontproperties())).name.lower())
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            outside.append(f"{t.get_text()[:30]} [x {bb.x0:.0f}-{bb.x1:.0f} of {fb.x1:.0f}; "
                           f"y {bb.y0:.0f}-{bb.y1:.0f} of {fb.y1:.0f}]")
    shrink = 0.05 / 25.4 * fig.dpi
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > shrink and min(a.y1, b.y1) - max(a.y0, b.y0) > shrink:
                overlaps.append(f"{boxes[i][0]!r}/{boxes[j][0]!r}")
    return {"n_texts": len(boxes), "min_font_pt": min(sizes), "fonts": ";".join(sorted(fonts)),
            "n_text_overlaps": len(overlaps), "overlaps": "; ".join(overlaps[:8]),
            "n_text_outside_figure": len(outside), "outside": "; ".join(outside[:8])}


def save(run, fig, name: str, desc: str, rows: list[dict], height: float) -> None:
    tc = text_checks(fig)
    paths = run.figure(fig, name, desc, formats=(".pdf", ".png"), tight=False)
    fc = t6dot.file_checks(paths, W, height)
    arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
    ok = bool(tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
              and fc["width_ok"] and fc.get("pdf_truetype_embedded") and fc.get("pdf_arial"))
    rows.append({"figure": name, "width_mm": W, "height_mm": round(height, 1), **tc, **fc, "pass": ok})
    print(f"  {name}: {W:.0f} x {height:.1f} mm, min font {tc['min_font_pt']}, overlaps {tc['n_text_overlaps']}, "
          f"outside {tc['n_text_outside_figure']} {tc['outside'][:80]!r}, pass {ok}", flush=True)
    plt.close(fig)


def bracket_levels(spans: list[tuple[str, float, float]]) -> list[int]:
    levels, right = [], {}
    for label, x0, x1 in spans:
        c, w = (x0 + x1) / 2, tw(label)
        lvl = 0
        while right.get(lvl, -np.inf) + 0.8 > c - w / 2:
            lvl += 1
        right[lvl] = c + w / 2
        levels.append(lvl)
    return levels


def colorbar(sh: Sheet, x: float, y: float, h: float, label: str) -> None:
    cax = sh.ax(x, y, 2.4, h)
    grad = np.linspace(Z_LIM, -Z_LIM, 256)[:, None]
    cax.imshow(grad, cmap=CMAP, vmin=-Z_LIM, vmax=Z_LIM, aspect="auto", extent=(0, 1, -Z_LIM, Z_LIM))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([-2, 0, 2])
    cax.tick_params(length=1.5, width=0.4, labelsize=FONT, pad=1)
    for s in cax.spines.values():
        s.set_linewidth(0.4)
        s.set_visible(True)
    sh.text(x + 2.4 + 5.2, y + h / 2, label, rotation=90, ha="left", va="center", color=INK["secondary"])


# ---------------------------------------------------------------------------
# Figures 1 and 2: paired heatmaps
# ---------------------------------------------------------------------------
def paired_heatmap(run, name: str, title: str, rows: list[str], row_colors: list[str], pairs: list[dict], note: str,
                   fig_rows: list[dict]) -> None:
    n_pairs = len(pairs)
    nb = max(1, math.ceil(n_pairs / MAX_PAIRS_PER_BLOCK))
    ppb = math.ceil(n_pairs / nb)
    blocks = [pairs[i * ppb:(i + 1) * ppb] for i in range(nb)]
    lw = max(tw(r) for r in rows) + 1.0 + 1.6 + 1.0
    right = 16.0
    cw = (W - 2 * M - lw - right) / (2 * ppb)
    RH, STRIP, RP = 3.4, 3.4, LH + 0.8
    gene_h = max(tw(p["label"]) for p in pairs) + 1.2
    has_groups = any(p.get("group") for p in pairs)
    title_h = t6dot.text_mm(title, TITLE, "bold")[1] + 1.5
    geo = []
    for blk in blocks:
        spans = []
        if has_groups:
            for j, p in enumerate(blk):
                if spans and spans[-1][0] == p["group"]:
                    spans[-1] = (p["group"], spans[-1][1], j)
                else:
                    spans.append((p["group"], j, j))
        span_mm = [(g, M + lw + 2 * a * cw, M + lw + 2 * (b + 1) * cw) for g, a, b in spans]
        lv = bracket_levels(span_mm) if span_mm else []
        grp_h = (max(lv) + 1) * (LH + 0.3) + 1.6 if lv else 0.0
        geo.append((blk, span_mm, lv, grp_h))
    note_text, note_h = note_block(note)
    block_h = [g[3] + STRIP + 0.6 + len(rows) * RH + RP + gene_h for g in geo]
    H = M + title_h + sum(block_h) + 4.0 * (nb - 1) + note_h + M
    sh = Sheet(H)
    sh.text(M, M, title, fontsize=TITLE, fontweight="bold", va="top")
    y = M + title_h
    for bi, ((blk, span_mm, lv, grp_h), bh) in enumerate(zip(geo, block_h)):
        x0 = M + lw
        ncol = 2 * len(blk)
        for (g, a, b), level in zip(span_mm, lv):
            yb = y + grp_h - 0.6
            sh.line([a + 0.4, a + 0.4, b - 0.4, b - 0.4], [yb + 0.5, yb, yb, yb + 0.5])
            sh.text((a + b) / 2, yb - 0.3 - level * (LH + 0.3), g, ha="center", va="bottom")
        ys = y + grp_h
        for j, p in enumerate(blk):
            xa = x0 + 2 * j * cw
            sh.rect(xa + 0.15, ys, 2 * cw - 0.3, STRIP, facecolor="#f4f3ef", edgecolor=INK["grid"], lw=0.4)
            sh.text(xa + cw, ys + STRIP / 2, f"{p['rho']:.2f}", ha="center", va="center_baseline")
        sh.text(x0 - 1.0, ys + STRIP / 2, "rho", ha="right", va="center_baseline", color=INK["secondary"])
        yh = ys + STRIP + 0.6
        mat = np.column_stack([np.column_stack([p["rna_z"], p["prot_z"]]) for p in blk])
        ax = sh.ax(x0, yh, ncol * cw, len(rows) * RH)
        ax.imshow(mat, cmap=CMAP, vmin=-Z_LIM, vmax=Z_LIM, aspect="auto", interpolation="nearest",
                  extent=(0, ncol, len(rows), 0))
        for k in range(1, ncol):
            ax.axvline(k, color="white", lw=1.4 if k % 2 == 0 else 0.3)
        for k in range(1, len(rows)):
            ax.axhline(k, color="white", lw=0.3)
        ax.set_xlim(0, ncol)
        ax.set_ylim(len(rows), 0)
        ax.set_axis_off()
        for i, (r, col) in enumerate(zip(rows, row_colors)):
            yc = yh + (i + 0.5) * RH
            sh.rect(x0 - 1.0 - 1.6, yc - 0.8, 1.6, 1.6, facecolor=col, edgecolor="none")
            sh.text(x0 - 1.0 - 1.6 - 1.0, yc, r, ha="right", va="center_baseline")
        yb = yh + len(rows) * RH
        for j, p in enumerate(blk):
            xa = x0 + 2 * j * cw
            sh.text(xa + cw / 2, yb + 0.4, "R", ha="center", va="top", color=INK["secondary"])
            sh.text(xa + 1.5 * cw, yb + 0.4, "P", ha="center", va="top", color=INK["secondary"])
            sh.text(xa + cw, yb + RP + 0.2, p["label"], rotation=90, ha="center", va="top")
        if bi == 0:
            colorbar(sh, x0 + ncol * cw + 2.5, yh, min(len(rows) * RH, 24.0), "z-score within modality")
        y += bh + 4.0
    draw_note(sh, note_text, note_h)
    save(run, sh.fig, name, title, fig_rows, H)


# ---------------------------------------------------------------------------
# Figure 3: marker rho (the earlier analysis canonical marker plot)
# ---------------------------------------------------------------------------
def marker_rho_figure(run, name: str, col: str, mk: pd.DataFrame, n_comp: int, fig_rows: list[dict]) -> pd.DataFrame:
    type_order = {t: i for i, t in enumerate((*L.ROW_ORDER, "AVR", "Podo"))}
    d = mk.copy()
    d["type_pos"] = d["listed_for"].map(type_order).fillna(99)
    d["wb_pos"] = d.groupby("listed_for").cumcount()
    d = d.sort_values(["type_pos", "wb_pos"]).drop_duplicates("marker").reset_index(drop=True)
    d["coverage"] = np.select([d["in_protein_matrix"] & d["in_atlas"], d["in_atlas"], d["in_protein_matrix"]],
                              list(COVERAGE)[:3], default="neither")
    d["detected_both"] = d["class"].astype(str).str.startswith("Detected in both")
    det = d[["max_rna_detection", "max_protein_detection"]].mean(axis=1, skipna=True).fillna(0)
    d["mean_detection"] = det
    d["label"] = d["listed_for"] + " | " + d["marker"]
    null = c7.null_rho(n_comp)
    lo, hi = np.quantile(null, [0.025, 0.975])
    half = math.ceil(len(d) / 2)
    parts = [d.iloc[:half], d.iloc[half:]]
    RH = 2.9
    lab_w = max(tw(s) for s in d["label"]) + 1.0
    title = f"V3 + endothelial-row markers: RNA-protein rho across {n_comp} compartments ({DISPLAY[col]}, all atlas cells)"
    title_h = t6dot.text_mm(title, TITLE, "bold")[1] + 1.5
    leg_h = 2 * LH + 3.0
    axis_h = 2 * LH + 2.5
    note = (f"Each marker once, under its first listing type. Colour: coverage; filled: detected in both modalities "
            f"at 10% in at least one compartment; area: mean of the highest RNA and protein detection fractions. "
            f"Markers without a rho are placed at the left. Grey band: central 95% of rho for unrelated rankings of "
            f"{n_comp} compartments ({lo:.2f} to {hi:.2f}); dotted lines at 0.2 and 0.5 (class limits).")
    note_text, note_h = note_block(note)
    body = half * RH
    H = M + title_h + leg_h + body + axis_h + note_h + M
    sh = Sheet(H)
    sh.text(M, M, title, fontsize=TITLE, fontweight="bold", va="top")
    # legend
    ly = M + title_h + 1.0
    x = M
    for lab, colr in COVERAGE.items():
        sh.fig.add_artist(matplotlib.lines.Line2D([(x + 1.2) / W], [1 - (ly + 1.0) / H], marker="o", ms=4,
                                                  mfc=colr, mec=colr, lw=0))
        sh.text(x + 2.6, ly + 1.0, lab, va="center_baseline")
        x += 2.6 + tw(lab) + 4.0
    x = M
    for lab, filled in (("detected in both at 10%", True), ("not detected in both", False)):
        sh.fig.add_artist(matplotlib.lines.Line2D([(x + 1.2) / W], [1 - (ly + LH + 2.0) / H], marker="o", ms=4,
                                                  mfc=INK["secondary"] if filled else "white", mec=INK["secondary"],
                                                  lw=0))
        sh.text(x + 2.6, ly + LH + 2.0, lab, va="center_baseline")
        x += 2.6 + tw(lab) + 4.0
    pw = (W - 2 * M - 2 * lab_w - 6.0) / 2
    y0 = M + title_h + leg_h
    for k, part in enumerate(parts):
        x0 = M + lab_w + k * (pw + 6.0 + lab_w)
        ax = sh.ax(x0, y0, pw, body)
        n = len(part)
        ax.set_ylim(half - 0.5, -0.5)
        ax.set_xlim(-1.35, 1.05)
        ax.axvspan(lo, hi, color=INK["grid"], alpha=0.6, lw=0, zorder=0)
        for v in (0.2, 0.5):
            ax.axvline(v, color=INK["secondary"], lw=0.5, ls=":", zorder=1)
        ax.axvline(0, color=INK["muted"], lw=0.4, zorder=1)
        ax.axvline(-1.1, color=INK["grid"], lw=0.4, zorder=1)
        for i, r in enumerate(part.itertuples()):
            xv = r.spearman_rho if np.isfinite(r.spearman_rho) else -1.22
            colr = COVERAGE[r.coverage]
            ax.scatter(xv, i, s=4 + 34 * float(r.mean_detection), facecolors=colr if r.detected_both else "white",
                       edgecolors=colr, linewidths=0.7, zorder=3)
            ax.axhline(i, color=INK["grid"], lw=0.3, zorder=0)
        ax.set_yticks(range(n), part["label"])
        ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"])
        ax.set_xticks([-1.22, -0.5, 0, 0.5, 1.0], ["no rho", "-0.5", "0", "0.5", "1"])
        ax.tick_params(axis="x", labelsize=FONT, length=1.5, width=0.4)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.4)
        ax.set_xlabel("Spearman rho, RNA vs protein across compartments", fontsize=FONT, labelpad=1.5)
    draw_note(sh, note_text, note_h)
    save(run, sh.fig, name, title, fig_rows, H)
    d["chance_q025"], d["chance_q975"] = lo, hi
    return d


# ---------------------------------------------------------------------------
# Figure 4: compartment bars
# ---------------------------------------------------------------------------
def compartment_figure(run, name: str, comp: pd.DataFrame, src: str, fig_rows: list[dict]) -> pd.DataFrame:
    sub = comp[(comp["rna_source"] == src) & np.isclose(comp["threshold"], THR)].copy()
    cols = ("kidney_t15_label", "kidney_pre_label")
    nmax = max(len(sub[sub["label_column"] == c]) for c in cols)
    RH = 5.0
    title = f"RNA-protein agreement per compartment ({SRC_TEXT[src]}; features detected in both at 10%)"
    title_h = t6dot.text_mm(title, TITLE, "bold")[1] + 1.5
    note = ("Bar: Spearman rho across the features detected in both modalities (>= 10%) in that compartment, n = "
            "features; open circle: rho across all 3,569 shared features. RNA: atlas cells of the Celltypes mapped to "
            "the label (kidney_rna_celltype_mapping.csv).")
    note_text, note_h = note_block(note)
    sub_h = LH + 1.5
    H = M + title_h + sub_h + nmax * RH + 2 * LH + 3.0 + note_h + M
    sh = Sheet(H)
    sh.text(M, M, title, fontsize=TITLE, fontweight="bold", va="top")
    lab_w = max(tw(s) for s in sub["compartment"]) + 1.5
    pw = (W - 2 * M - 2 * lab_w - 8.0) / 2
    y0 = M + title_h + sub_h
    for k, col in enumerate(cols):
        s = sub[sub["label_column"] == col].reset_index(drop=True)
        x0 = M + lab_w + k * (pw + 8.0 + lab_w)
        sh.text(x0 - lab_w, y0 - 1.0, DISPLAY[col], fontweight="bold", va="bottom")
        ax = sh.ax(x0, y0, pw, nmax * RH)
        yy = np.arange(len(s))
        ax.barh(yy, s["spearman_rho_detected_both"], height=0.62, color=t3.BLUE_RAMP[4], zorder=2)
        ax.scatter(s["spearman_rho_all_shared"], yy, s=10, facecolors="white", edgecolors=INK["primary"],
                   linewidths=0.6, zorder=3)
        for i, r in s.iterrows():
            xv = max(r["spearman_rho_detected_both"], r["spearman_rho_all_shared"], 0)
            ax.text(xv + 0.02, i, f"n = {int(r['n_detected_both'])}", va="center", fontsize=FONT,
                    color=INK["secondary"])
        ax.set_yticks(yy, s["compartment"])
        ax.set_ylim(nmax - 0.5, -0.5)
        xmin = min(-0.15, float(s[["spearman_rho_detected_both", "spearman_rho_all_shared"]].min().min()) - 0.05)
        ax.set_xlim(xmin, 0.85)
        ax.set_xticks([v for v in (-0.2, 0, 0.2, 0.4, 0.6, 0.8) if v >= xmin])
        ax.axvline(0, color=INK["muted"], lw=0.5)
        ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"])
        ax.tick_params(axis="x", labelsize=FONT, length=1.5, width=0.4)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.4)
        ax.set_xlabel("Spearman rho across features", fontsize=FONT, labelpad=1.5)
    draw_note(sh, note_text, note_h)
    save(run, sh.fig, name, title, fig_rows, H)
    return sub


# ---------------------------------------------------------------------------
# Figure 5: stacked marker heatmaps (integrate_scp_with_mka.py style)
# ---------------------------------------------------------------------------
def marker_heatmaps(run, name: str, col: str, rna_rows: list[str], rna_chip: list[str], prot_rows: list[str],
                    prot_colors: list[str], markers: list[str], groups: list[str], rna_z: np.ndarray,
                    prot_z: np.ndarray, fig_rows: list[dict]) -> None:
    RH = 2.9
    title = f"Marker pseudobulk, RNA (atlas Celltypes) and protein ({DISPLAY[col]}), z-scored within modality"
    title_h = t6dot.text_mm(title, TITLE, "bold")[1] + 1.5
    lw = max(tw(r) for r in rna_rows + prot_rows) + 1.0 + 1.6 + 1.0
    right = 16.0
    ncol = len(markers)
    cw = (W - 2 * M - lw - right) / ncol
    spans = []
    for j, g in enumerate(groups):
        if spans and spans[-1][0] == g:
            spans[-1] = (g, spans[-1][1], j)
        else:
            spans.append((g, j, j))
    span_mm = [(g, M + lw + a * cw, M + lw + (b + 1) * cw) for g, a, b in spans]
    lv = bracket_levels(span_mm)
    grp_h = (max(lv) + 1) * (LH + 0.3) + 1.6
    gene_h = max(tw(m) for m in markers) + 1.2
    sub_h = LH + 1.2
    note = ("Columns: V3 / endothelial-row marker proteins quantified in the SCP matrix and present in the atlas, in "
            "the dotplot column order (brackets = dotplot column groups). RNA: mean log-normalized expression per atlas "
            "Celltype (all cells; Unknown not shown); chip = the SCP label the Celltype is mapped to. Protein: mean "
            "ln(1 + intensity) per SCP label (missing = 0). Each marker z-scored within its modality across the rows.")
    note_text, note_h = note_block(note)
    H = M + title_h + grp_h + sub_h + len(rna_rows) * RH + 3.0 + sub_h + len(prot_rows) * RH + gene_h + note_h + M
    sh = Sheet(H)
    sh.text(M, M, title, fontsize=TITLE, fontweight="bold", va="top")
    x0 = M + lw
    y = M + title_h
    for (g, a, b), level in zip(span_mm, lv):
        yb = y + grp_h - 0.6
        sh.line([a + 0.3, a + 0.3, b - 0.3, b - 0.3], [yb + 0.5, yb, yb, yb + 0.5])
        sh.text((a + b) / 2, yb - 0.3 - level * (LH + 0.3), g, ha="center", va="bottom")
    y += grp_h
    for k, (rows, chips, mat, lab) in enumerate(((rna_rows, rna_chip, rna_z, "RNA: atlas Celltype (all cells)"),
                                                  (prot_rows, prot_colors, prot_z, f"Protein: {DISPLAY[col]}"))):
        sh.text(M, y + sub_h - 0.4, lab, fontweight="bold", va="bottom")
        y += sub_h
        ax = sh.ax(x0, y, ncol * cw, len(rows) * RH)
        ax.imshow(mat, cmap=CMAP, vmin=-Z_LIM, vmax=Z_LIM, aspect="auto", interpolation="nearest",
                  extent=(0, ncol, len(rows), 0))
        for j in range(1, ncol):
            ax.axvline(j, color="white", lw=0.3)
        for i in range(1, len(rows)):
            ax.axhline(i, color="white", lw=0.3)
        ax.set_xlim(0, ncol)
        ax.set_ylim(len(rows), 0)
        ax.set_axis_off()
        for i, (r, c_) in enumerate(zip(rows, chips)):
            yc = y + (i + 0.5) * RH
            if c_:
                sh.rect(x0 - 1.0 - 1.6, yc - 0.8, 1.6, 1.6, facecolor=c_, edgecolor="none")
            sh.text(x0 - 1.0 - 1.6 - 1.0, yc, r, ha="right", va="center_baseline")
        if k == 0:
            colorbar(sh, x0 + ncol * cw + 2.5, y, 24.0, "z-score within modality")
        y += len(rows) * RH + (3.0 if k == 0 else 0.0)
    for j, m in enumerate(markers):
        sh.text(x0 + (j + 0.5) * cw, y + 0.6, m, rotation=90, ha="center", va="top")
    draw_note(sh, note_text, note_h)
    save(run, sh.fig, name, title, fig_rows, H)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "paired_figures", args)
    checks = t5.Checks(L.DATASET)
    adata, labels = L.load_kidney(run, checks)

    def t07(name: str) -> pd.DataFrame:
        return pd.read_csv(run.use_input(cc.out_path(L.TASK, "tables", name, ".csv"), f"kidney_rna step 5 {name}"))

    long_df, feat, mk, comp = (t07("concordance_long"), t07("concordance_features"), t07("concordance_markers"),
                               t07("concordance_compartments"))
    mapping = pd.read_csv(run.use_input(MAPPING, "kidney_rna approved SCP label -> atlas mapping (step 2)"))
    pb = L.read_pseudobulk(run)
    orders = {"kidney_t15_label": pd.read_csv(run.use_input(L.KIDNEY_ANNOTATION_MARKER_ORDER, "kidney_annotation a_endo dotplot marker order")),
              "kidney_pre_label": pd.read_csv(run.use_input(DOTPLOT_DIR / "dotplot_marker_order_kidney.csv",
                                                            "dotplot kidney marker order"))}
    t06_lo = pd.read_csv(run.use_input(DOTPLOT_DIR / "dotplot_label_order.csv", "dotplot label order"))
    t15_rows = L.row_order(labels["kidney_t15_label"].unique())
    pal_t15 = t15c.palette(adata, t15_rows)
    pal_pre = dict(zip(t06_lo.loc[t06_lo["label_column"] == "kidney_pre_label", "label"],
                       t06_lo.loc[t06_lo["label_column"] == "kidney_pre_label", "colour"]))
    palettes = {"kidney_t15_label": pal_t15, "kidney_pre_label": pal_pre}
    plt.rcParams.update({"font.family": "Arial", "font.size": FONT, "xtick.labelsize": FONT,
                         "ytick.labelsize": FONT, "axes.labelsize": FONT, "legend.fontsize": FONT})

    fig_rows: list[dict] = []
    hv_rows: list[pd.DataFrame] = []
    for col in ("kidney_t15_label", "kidney_pre_label"):
        order = t6dot.plotted_order(orders[col], "full")
        order = order[~order["repeat"].fillna(False).astype(bool)]
        dot_pos = {m: i for i, m in enumerate(order["marker"])}
        dot_grp = dict(zip(order["marker"], order["group"]))
        for src in ("all", "nuclei"):
            lg = long_df[(long_df["label_column"] == col) & (long_df["rna_source"] == src)]
            rows = [r for r in L.row_order(lg["compartment"].unique())]
            f = feat[(feat["label_column"] == col) & (feat["rna_source"] == src) & np.isclose(feat["threshold"], THR)]
            f = f.set_index("gene")

            def pairs_for(genes: list[str], group_of=None) -> list[dict]:
                out = []
                for g in genes:
                    s = lg[lg["gene"] == g].set_index("compartment").reindex(rows)
                    prot = f.loc[g, "protein"]
                    is_marker = isinstance(f.loc[g, "marker_listed_for"], str) and f.loc[g, "marker_listed_for"] != ""
                    out.append({"gene": g, "protein": prot, "rho": float(f.loc[g, "spearman_rho"]),
                                "class": f.loc[g, "class"], "group": group_of(prot) if group_of else "",
                                "label": prot + ("*" if (is_marker and not group_of) else ""),
                                "rna_mean": s["rna_mean_lognorm"].to_numpy(), "prot_mean": s["protein_mean_ln1p"].to_numpy(),
                                "rna_z": zscore(s["rna_mean_lognorm"].to_numpy()),
                                "prot_z": zscore(s["protein_mean_ln1p"].to_numpy())})
                return out

            conc = f[f["class"] == "Detected in both; concordant"]
            canonical = conc[conc["marker_listed_for"].fillna("") != ""].index.tolist()
            top = conc.sort_values("spearman_rho", ascending=False).index[:TOP_N].tolist()
            cand = list(dict.fromkeys(canonical + top))
            genes = (conc.loc[cand].reset_index().sort_values(["spearman_rho", "gene"], ascending=[False, True])
                     ["gene"].head(MAX_GENES).tolist())
            checks.add(f"{SHORT[col]} / {src}: top-concordant genes (markers among them; the earlier analysis rule)",
                       f"{len(genes)} ({sum(g in canonical for g in genes)})", len(genes) > 0)
            pr = pairs_for(genes)
            colors = [palettes[col].get(r, INK["muted"]) for r in rows]
            note = (f"Rows: compartments (SCP label; RNA = atlas cells of its mapped Celltypes, {SRC_TEXT[src]}). "
                    "Each gene: RNA (R) and protein (P) side by side, each z-scored across the compartments within its "
                    "modality (RNA: mean log-normalized expression; protein: mean ln(1 + intensity), missing = 0). "
                    "Strip: Spearman rho of RNA vs protein means across the compartments. Genes: concordant at 10% "
                    f"(rho >= 0.5); V3 / endothelial-row markers (*) first, then the top {TOP_N} by rho, at most "
                    f"{MAX_GENES}, ordered by rho (the earlier rule).")
            paired_heatmap(run, f"paired_top_{SHORT[col]}_{src}",
                           f"Top concordant RNA-protein genes across compartments ({DISPLAY[col]}, {SRC_TEXT[src]})",
                           rows, colors, pr, note, fig_rows)
            mk_genes = [g for g in f.index if f.loc[g, "protein"] in dot_pos
                        and str(f.loc[g, "class"]).startswith("Detected in both")]
            mk_genes = sorted(mk_genes, key=lambda g: dot_pos[f.loc[g, "protein"]])
            checks.add(f"{SHORT[col]} / {src}: dotplot markers detected in both at 10% (of quantified markers in the "
                       "atlas)", f"{len(mk_genes)} of {sum(p in dot_pos for p in f['protein'])}", len(mk_genes) > 0)
            pm = pairs_for(mk_genes, group_of=lambda p: dot_grp[p])
            note_m = (f"Rows: compartments (SCP label; RNA = atlas cells of its mapped Celltypes, {SRC_TEXT[src]}). "
                      "Columns: the dotplot marker proteins detected in both modalities at 10% (any rho), in the dotplot "
                      "column order; brackets = dotplot column groups. Each gene: RNA (R) and protein (P), each z-scored "
                      "across the compartments within its modality. Strip: Spearman rho across the compartments.")
            paired_heatmap(run, f"paired_markers_{SHORT[col]}_{src}",
                           f"Marker genes, RNA vs protein across compartments ({DISPLAY[col]}, {SRC_TEXT[src]})",
                           rows, colors, pm, note_m, fig_rows)
            for sel, prs in (("top_concordant", pr), ("markers", pm)):
                for pos, p in enumerate(prs, start=1):
                    for mod, mean_k, z_k in (("RNA", "rna_mean", "rna_z"), ("protein", "prot_mean", "prot_z")):
                        hv_rows.append(pd.DataFrame({
                            "label_column": col, "rna_source": src, "selection": sel, "pair_position": pos,
                            "gene": p["gene"], "protein": p["protein"], "group": p["group"], "class": p["class"],
                            "spearman_rho": p["rho"], "compartment": rows, "modality": mod, "mean": p[mean_k],
                            "z_within_modality": p[z_k]}))
    hv = pd.concat(hv_rows, ignore_index=True)
    zm = hv.groupby(["label_column", "rna_source", "selection", "gene", "modality"])["z_within_modality"].mean()
    checks.add("plotted z-scores have mean 0 per gene and modality (max abs)", f"{float(zm.abs().max()):.1e}",
               float(zm.abs().max()) < 1e-9)

    # figure 3
    rho_rows = []
    for col in ("kidney_t15_label", "kidney_pre_label"):
        m = mk[(mk["label_column"] == col) & (mk["rna_source"] == "all") & np.isclose(mk["threshold"], THR)]
        n_comp = int(feat.loc[(feat["label_column"] == col) & (feat["rna_source"] == "all"), "n_compartments"].iloc[0])
        rho_rows.append(marker_rho_figure(run, f"paired_marker_rho_{SHORT[col]}", col, m, n_comp, fig_rows))
    rho_df = pd.concat(rho_rows, ignore_index=True)
    checks.add("figure 3: markers plotted per label set (unique)", "; ".join(
        f"{SHORT[c]} {int((rho_df['label_column'] == c).sum())}" for c in ("kidney_t15_label", "kidney_pre_label")),
        None)

    # figure 4
    comp_rows = [compartment_figure(run, f"paired_compartments_{src}", comp, src, fig_rows) for src in ("all", "nuclei")]

    # figure 5
    mh_rows = []
    look = L.atlas_gene_lookup(pb.var_names)
    celltypes = [c for c in L.atlas_order(pb.obs["Celltype"].astype(str).unique()) if c != L.ATLAS_UNKNOWN]
    pidx_all = pd.Index(adata.var_names.astype(str))
    ln1p = cc.dense(adata.layers["ln1p_intensity"])
    for col in ("kidney_t15_label", "kidney_pre_label"):
        order = t6dot.plotted_order(orders[col], "full")
        order = order[~order["repeat"].fillna(False).astype(bool)]
        markers = [m for m in order["marker"] if m.lower() in look]
        groups = [dict(zip(order["marker"], order["group"]))[m] for m in markers]
        genes = [look[m.lower()] for m in markers]
        rna = np.vstack([L.aggregate(pb, L.group_mask(pb, Celltype=ct), genes)["mean_lognorm"].to_numpy()
                         for ct in celltypes])
        rows = L.row_order(labels[col].unique())
        lab = labels[col].astype(str).to_numpy()
        pos = pidx_all.get_indexer(markers)
        prot = np.vstack([ln1p[lab == r][:, pos].mean(axis=0) for r in rows])
        rna_z = np.column_stack([zscore(rna[:, j]) for j in range(len(markers))])
        prot_z = np.column_stack([zscore(prot[:, j]) for j in range(len(markers))])
        mp = mapping[mapping["label_column"] == col]
        chip_of = {}
        for r in mp.itertuples():
            if isinstance(r.atlas_celltype_expected, str):
                for ct in r.atlas_celltype_expected.split(";"):
                    chip_of[ct] = palettes[col].get(r.scp_label, INK["muted"])
        marker_heatmaps(run, f"paired_marker_heatmaps_{SHORT[col]}", col, celltypes,
                        [chip_of.get(ct, "") for ct in celltypes], rows,
                        [palettes[col].get(r, INK["muted"]) for r in rows], markers, groups, rna_z, prot_z, fig_rows)
        for mod, rws, mean_m, z_m in (("RNA", celltypes, rna, rna_z), ("protein", rows, prot, prot_z)):
            for i, rw in enumerate(rws):
                mh_rows.append(pd.DataFrame({"label_column": col, "modality": mod, "row": rw, "marker": markers,
                                             "group": groups, "mean": mean_m[i], "z_within_modality": z_m[i]}))
        absent = [m for m in order["marker"] if m.lower() not in look]
        checks.add(f"figure 5 {SHORT[col]}: marker columns (quantified, in atlas); quantified but not in atlas",
                   f"{len(markers)}; {', '.join(absent) or 'none'}", len(markers) > 0)

    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks (180 mm, Arial >= 7 pt, no overlap, PDF + PNG)",
               f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(hv, "paired_heatmap_values")
    run.table(rho_df, "paired_marker_rho")
    run.table(pd.concat(comp_rows, ignore_index=True), "paired_compartments")
    run.table(pd.concat(mh_rows, ignore_index=True), "paired_marker_heatmap_values")
    run.table(figs, "paired_figure_checks")
    run.table(checks.frame(), "paired_checks")
    run.finish({"threshold": THR, "top_n": TOP_N, "max_genes": MAX_GENES, "max_pairs_per_block": MAX_PAIRS_PER_BLOCK,
                "z_score": "per gene within modality across rows, ddof 0 (as the earlier analysis)", "z_colour_limit": Z_LIM,
                "width_mm": W, "request": "authors, 15 Oct 2026: direct direct RNA vs protein figures",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see kidney_rna_paired_checks.csv.")


if __name__ == "__main__":
    main()
