#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figures S-07-a to S-07-c (PT / TAL heterogeneity), revised annotation only.

The authors decided on 30 Oct 2026 that the kidney_annotation revised
annotation (kidney_t15_label) is the only kidney annotation in the manuscript, letter and figures, and asked on
10 Nov 2026 for the S-07 panels to be redrawn with it ("Yes, redraw them"). On 11 Nov 2026 the authors asked for each
supplementary figure as one assembled figure instead of separate panel files ("make Supplemental Figure S-07-a as one
figure (not as separate panels a b c). Same for 07-b and 07-c"). No new statistics: every plotted value is read from
the stored pt_tal / rat_segments tables.

Style: one file per supplementary figure, 180 mm wide, panels marked A, B (C) in bold; no title or footnote (the
legends are in the the response to the reviewers file, section 5); Arial >= 7 pt; row colours of the revised Figure 2A (kidney_figure2 colour key,
where PST is #8c564b since 10 Nov 2026).

Figures (results/figures/report/, PDF + 600-dpi PNG):
- report_07_S07a_atlas_markers: A, atlas subtype panel proteins and literature markers, protein above RNA;
  B, nearest atlas subtype per cell type.
- report_07_S07b_scores_depth: A, PCT vs PST on the atlas subtype scores; B, PT and TAL subsets by cell type and
  ln(protein IDs).
- report_07_S07c_rat_segments: A, PCT vs PST on the rat segment scores; B, pseudobulk PCT - PST vs rat segment
  ratios; C, pseudobulk SCP-MS groups vs rat segments.

Inputs (read-only): pt_tal tables dotplot_columns, dotplot_values, map_confusion, scores_pct_vs_pst, subsets_per_cell;
                    rat_segments tables scores_pct_vs_pst, pseudobulk_pt_ratio, pseudobulk_matrix; kidney_figure2 fig2a_umap_key.
Outputs: the figures above; results/tables/report/report_07_s07_panels_{values,checks,figure_checks}.csv;
         results/tables/report/report_step_07_s07_panels_manifest.json

Command (project root):  python scripts/report_pt_tal_figures.py --overwrite
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
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

import scp_common as cc
import dotplot_dotplots as t6dot          # Panel, Canvas, legends
import kidney_rna_dotplots as t7dot          # GreyPanel, P_MAX (as the kidney_rna / pt_tal paired dotplot)
import atlas_dct_lib as T8                  # Sheet, save_checked, styles
import pt_tal_lib as T9                  # legend and UMAP helpers, RNA palette
import rat_segments_lib as RAT_SEGMENTS

TASK = "report"
STEP = "07_s07_panels"
COL = "kidney_t15_label"
MAP_VARIANT = "primary|panels|subset_z"      # pt_tal primary mapping (pt_tal_subclusters.py)
SERIES_COLOR = "#184f95"                     # pt_tal / rat_segments colour of the kidney_annotation series
PST_COLOR = "#8c564b"                        # kidney_figure2 key since 10 Nov 2026 (was #7f7f7f; authors: "Change PST colour")
MINUS = "−"
LETTER_PT = 9.0                              # panel letters, bold
ROW_GAP, COL_GAP = 4.0, 14.0                 # mm between panel rows / between panels in one row
LIT_GROUP = {"Literature (plan)": "Literature markers"}
ATLAS_ROWS = ("PTS1", "PTS3", "PTS3T2", "S1/S2", "S3", "S1/S2 - S3", "PTS1 - PTS3", "ln(protein IDs)")
RAT_ROWS = ("S1", "S2", "S3", "S1/S2", "S1/S2 - S3", "S1 - S3", "pt_tal atlas: S1/S2 - S3", "ln(protein IDs)")
ROW_TEXT = {"pt_tal atlas: S1/S2 - S3": "atlas S1/S2 - S3"}
EXPECTED = {
    "dotplot_rows": {"PCT": 160, "PST": 158, "TAL": 138},
    "dotplot_columns": 26,
    "slc5a2_detected": {"PCT": 0.74375, "PST": 0.5},
    "map_pct": (63, 34, 33, 30),
    "map_pct_s1s2": 97,
    "map_pst_s3": 86,
    "map_tal": {"TAL": (60, 78), "Unknown": (2, 12)},
    "subset_pt": {"PCT": 160, "PST": 158, "Unknown": 38},
    "subset_tal": {"TAL": 138, "Unknown": 14, "PST": 1, "JG": 1},
    "atlas_auc_s1s2_s3": 0.6203,
    "auc_ln_ids": 0.5221,
    "rat_auc_s1s2_s3": 0.6413,
    "pb_rho_s1s2_s3": 0.2263,
    "pb_n_ratio_proteins": 1417,
    "pb_n_proteins": 1351,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: S-07 figures, revised annotation only.")
    cc.add_common_args(parser)
    return parser.parse_args()


def txt(s: str) -> str:
    """Display text: renamed rows, true minus signs between scores."""
    return ROW_TEXT.get(s, s).replace(" - ", f" {MINUS} ")


def check(checks: list[dict], name: str, observed, expected, ok: bool) -> None:
    checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})


# ---------------------------------------------------------------------------
# Assembly: one figure per supplementary figure, panels placed in mm
# ---------------------------------------------------------------------------
class Board(T8.Sheet):
    """One figure of W x H mm: a dotplot dotplot Canvas (mm overlay axes) that also takes Sheet axes and text."""

    def __init__(self, height: float, width: float = T8.W):
        self.cv = t6dot.Canvas(width, height)
        self.fig, self.W, self.H = self.cv.fig, width, height


@dataclass
class Block:
    """One panel: its size in mm and a function that draws it with its top-left corner at (x, y)."""
    w: float
    h: float
    draw: Callable[[Board, float, float], None]
    values: list[dict] = field(default_factory=list)


def assemble(run, fig_rows: list[dict], name: str, desc: str, rows: list[list[tuple[str, Block]]]) -> list[dict]:
    """Panels of one row sit side by side (top-aligned), rows below each other; each panel has a bold letter."""
    letter_h = T8.text_mm("A", LETTER_PT, "bold")[1] + 1.0
    avail = T8.W - 2 * T8.M
    for row in rows:
        w = sum(b.w for _, b in row) + COL_GAP * (len(row) - 1)
        if w > avail + 1e-6:
            raise ValueError(f"{name}: a panel row is {w:.1f} mm wide, more than {avail:.1f} mm")
    height = 2 * T8.M + sum(letter_h + max(b.h for _, b in row) for row in rows) + ROW_GAP * (len(rows) - 1)
    sh = Board(height)
    out = []
    y = T8.M
    for row in rows:
        x = T8.M
        for letter, b in row:
            sh.text(x, y, letter, fontsize=LETTER_PT, fontweight="bold", va="top")
            b.draw(sh, x, y + letter_h)
            out += [{"figure": name, "panel_letter": letter, **v} for v in b.values]
            x += b.w + COL_GAP
        y += letter_h + max(b.h for _, b in row) + ROW_GAP
    T8.save_checked(run, sh, name, desc, fig_rows)
    return out


# ---------------------------------------------------------------------------
# S-07-a A: paired dotplot (protein above RNA), revised annotation rows only
# ---------------------------------------------------------------------------
def dotplot_block(checks, cols: pd.DataFrame, vals: pd.DataFrame, colors: dict, avail: float) -> Block:
    cols = cols[cols["version"] == "main"].sort_values("position")
    markers = cols["marker"].tolist()
    groups = t6dot.groups_of(pd.DataFrame({"group": cols["group"].replace(LIT_GROUP)}))
    check(checks, "S-07-a A dotplot: columns (pt_tal main version)", len(markers), EXPECTED["dotplot_columns"],
          len(markers) == EXPECTED["dotplot_columns"])
    rows = list(EXPECTED["dotplot_rows"])
    p = vals[vals["panel"] == "protein t15"]
    quant = dict(zip(cols["marker"], cols["quantified_protein"].astype(bool)))
    frac = np.zeros((len(rows), len(markers)))
    colr = np.zeros((len(rows), len(markers)))
    for j, mk in enumerate(markers):
        if quant[mk]:
            s = p[p["marker"] == mk].set_index("row").reindex(rows)
            frac[:, j] = s["frac_detected"].to_numpy(float)
            colr[:, j] = s["mean_scaled_0_1"].to_numpy(float)
    n = p.drop_duplicates("row").set_index("row")["n"].reindex(rows).astype(int).tolist()
    check(checks, "S-07-a A dotplot: protein rows (n cells)", dict(zip(rows, n)), EXPECTED["dotplot_rows"],
          dict(zip(rows, n)) == EXPECTED["dotplot_rows"])
    j5 = markers.index("Slc5a2")
    obs = {r: round(float(frac[i, j5]), 5) for i, r in enumerate(rows) if r != "TAL"}
    check(checks, "S-07-a A dotplot: Slc5a2 detected fraction (PCT, PST)", obs, EXPECTED["slc5a2_detected"],
          obs == EXPECTED["slc5a2_detected"])
    prot = t7dot.GreyPanel("Protein (SCP-MS)", rows, n, [colors[r] for r in rows], markers, groups, frac, colr,
                           [False] * len(markers), grey_rows=[False] * len(rows),
                           grey_cols=[not quant[mk] for mk in markers])
    rna_rows = list(T9.REGION_CLASSES)
    r_ = vals[vals["panel"] == "rna all"]
    rfrac = np.array([[float(r_[(r_["row"] == r) & (r_["marker"] == mk)]["frac_detected"].iloc[0]) for mk in markers]
                      for r in rna_rows])
    rcol = np.array([[float(r_[(r_["row"] == r) & (r_["marker"] == mk)]["mean_scaled_0_1"].iloc[0]) for mk in markers]
                     for r in rna_rows])
    rn = [int(r_[r_["row"] == r]["n"].iloc[0]) for r in rna_rows]
    in_atlas = dict(zip(cols["marker"], cols["atlas_gene"].fillna("").astype(str) != ""))
    rna = t7dot.GreyPanel("RNA (Mouse Kidney Atlas, all cells and nuclei)", rna_rows, rn,
                          [T9.RNA_PALETTE[r] for r in rna_rows], markers, groups, rfrac, rcol, [False] * len(markers),
                          grey_rows=[x == 0 for x in rn], grey_cols=[not in_atlas[mk] for mk in markers],
                          n_text=[f"{x:,}" for x in rn])
    check(checks, "S-07-a A dotplot: grey columns, protein (not quantified) | RNA (not an atlas gene)",
          ", ".join(m for m in markers if not quant[m]) + " | " + ", ".join(m for m in markers if not in_atlas[m]),
          "Slc7a13, Cldn16, Ptgs2 | Slc22a6, Cldn16",
          [m for m in markers if not quant[m]] == ["Slc7a13", "Cldn16", "Ptgs2"]
          and [m for m in markers if not in_atlas[m]] == ["Slc22a6", "Cldn16"])

    # geometry as kidney_rna_dotplots.build_paired (layout "stacked"), without the in-figure note
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    panels, vgap = [prot, rna], 3.0
    cw, ch = t6dot.color_legend_dims()
    lw = max(q.row_label_w() for q in panels)
    leg_w = max(t6dot.size_legend_dims(t7dot.P_MAX)[0], cw)
    pitch = min(t7dot.P_MAX, (avail - lw - 5.0 - leg_w) / len(markers))
    sw, sh_, _ = t6dot.size_legend_dims(pitch)
    leg_w, leg_h = max(sw, cw), sh_ + 3.0 + ch
    sizes = [q.size(pitch) for q in panels]
    body_h = sum(h for _, h in sizes) + vgap * (len(panels) - 1)
    top_offset = panels[0].title_h() + panels[0].bracket_h(pitch)

    def draw(sh: Board, x: float, y: float) -> None:
        yy, geo = y, []
        for q, (_, h) in zip(panels, sizes):
            geo.append(q.draw(sh.cv, x + lw - q.row_label_w(), yy, pitch))
            yy += h + vgap
        t6dot.draw_legends(sh.cv, geo[0]["grid_right"] + 5.0, geo[0]["grid_top"], pitch, stacked=True)

    out = []
    for panel, rr, f_, c_ in (("protein", rows, frac, colr), ("RNA", rna_rows, rfrac, rcol)):
        for i, r in enumerate(rr):
            for j, mk in enumerate(markers):
                out.append({"panel": panel, "row": r, "column": mk, "frac_detected": f_[i, j],
                            "colour_scaled_0_1": c_[i, j]})
    return Block(lw + len(markers) * pitch + 5.0 + leg_w, max(body_h, top_offset + leg_h), draw, out)


# ---------------------------------------------------------------------------
# S-07-a B: nearest atlas subtype per cell type
# ---------------------------------------------------------------------------
def mapping_block(checks, conf: pd.DataFrame) -> Block:
    LH = T9.LH
    cmap = T8.sequential_cmap()
    c = conf[(conf["variant"] == MAP_VARIANT) & (conf["label_column"] == COL) & (conf["n_label"] >= 5)]
    cw, rh, gap = 16.0, 4.6, 10.0
    blocks, out = [], []
    for seg, classes, head in (("PT", T9.PT_CLASSES, "Proximal tubule cells"), ("TAL", T9.TAL_CLASSES, "TAL cells")):
        s = c[c["segment"] == seg]
        labs = T9.order_labels(s["label"])
        counts = np.array([[int(s[(s["label"] == l_) & (s["class"] == k)]["n_top"].iloc[0]) for k in classes]
                           for l_ in labs])
        confn = np.array([[int(s[(s["label"] == l_) & (s["class"] == k)]["n_top_confident"].iloc[0]) for k in classes]
                          for l_ in labs])
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        blocks.append((list(classes), head, labs, counts, confn, frac))
        for a, l_ in enumerate(labs):
            for b, k in enumerate(classes):
                out.append({"panel": seg, "row": l_, "column": k, "n_cells": counts[a, b], "n_confident": confn[a, b],
                            "fraction": frac[a, b]})
    lw = max(T8.text_mm(f"{l_} ({int(cn.sum(axis=1)[i])})")[0] for _, _, labs, cn, _, _ in blocks
             for i, l_ in enumerate(labs)) + 2.0
    nmax = max(len(b[2]) for b in blocks)

    def draw(sh: Board, x: float, y: float) -> None:
        xx = x + lw
        y0 = y + LH + 1.0
        for classes, head, labs, counts, confn, frac in blocks:
            sh.text(xx - lw, y, head, va="top")
            ax = sh.ax(xx, y0, len(classes) * cw, len(labs) * rh)
            ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                      extent=(0, len(classes), len(labs), 0))
            ax.set_axis_off()
            for a, l_ in enumerate(labs):
                sh.text(xx - 1.0, y0 + (a + 0.5) * rh, f"{l_} ({int(counts[a].sum())})", ha="right",
                        va="center_baseline")
                for b in range(len(classes)):
                    sh.text(xx + (b + 0.5) * cw, y0 + (a + 0.5) * rh, f"{counts[a, b]} ({confn[a, b]})", ha="center",
                            va="center_baseline", color="white" if frac[a, b] > 0.55 else T9.INK["primary"])
            for b, k in enumerate(classes):
                sh.text(xx + (b + 0.5) * cw, y0 + len(labs) * rh + 0.8, k, ha="center", va="top")
            xx += len(classes) * cw + gap + lw

    pt, tal = blocks[0], blocks[1]
    got_pct = tuple(int(v) for v in pt[3][pt[2].index("PCT")])
    check(checks, "S-07-a B mapping: PCT cells nearest PTS1 / PTS2 / PTS3 / PTS3T2", got_pct, EXPECTED["map_pct"],
          got_pct == EXPECTED["map_pct"])
    s12 = int(got_pct[0] + got_pct[1])
    s3 = int(pt[3][pt[2].index("PST")][2:].sum())
    check(checks, "S-07-a B mapping: PCT nearest S1/S2; PST nearest PTS3/PTS3T2",
          f"{s12}/160; {s3}/158", f"{EXPECTED['map_pct_s1s2']}/160; {EXPECTED['map_pst_s3']}/158",
          s12 == EXPECTED["map_pct_s1s2"] and s3 == EXPECTED["map_pst_s3"])
    got_rows = {lab: int(pt[3][i].sum()) for i, lab in enumerate(pt[2])}
    check(checks, "S-07-a B mapping: PT rows (cells)", got_rows, EXPECTED["subset_pt"],
          got_rows == EXPECTED["subset_pt"])
    got_tal = {lab: tuple(int(v) for v in tal[3][i]) for i, lab in enumerate(tal[2])}
    check(checks, "S-07-a B mapping: TAL rows nearest CTAL / MTAL", got_tal, EXPECTED["map_tal"],
          got_tal == EXPECTED["map_tal"])
    w = sum(lw + len(b[0]) * cw for b in blocks) + gap * (len(blocks) - 1)
    return Block(w, LH + 1.0 + nmax * rh + 0.8 + LH, draw, out)


# ---------------------------------------------------------------------------
# S-07-b A and S-07-c A: PCT vs PST effects, one series
# ---------------------------------------------------------------------------
def effects_block(tests: pd.DataFrame, rows: tuple[str, ...], split_before: str, avail: float) -> Block:
    LH = T9.LH
    rh, gap = 5.0, 12.0
    labels = [txt(r) for r in rows]
    lw = max(T8.text_mm(r)[0] for r in labels) + 2.0
    pw = (avail - lw - gap) / 2
    heads = ("AUC (PCT vs PST)", f"PCT {MINUS} PST (SD), adjusted for ln(protein IDs)")
    lim = np.nanmax(np.abs(tests[["ci_sd_low_adjusted_ln_ids", "ci_sd_high_adjusted_ln_ids"]].to_numpy(float)))
    lim = float(np.ceil(lim * 2) / 2)
    i_split = rows.index(split_before)
    out = []
    for r in rows:
        t_ = tests[tests["score"] == r].iloc[0]
        out.append({"panel": "auc", "row": r, "value": t_.auc, "ci_low": t_.auc_ci_low, "ci_high": t_.auc_ci_high})
        if r != "ln(protein IDs)":
            out.append({"panel": "coefficient_adjusted", "row": r, "value": t_.coef_sd_adjusted_ln_ids,
                        "ci_low": t_.ci_sd_low_adjusted_ln_ids, "ci_high": t_.ci_sd_high_adjusted_ln_ids})

    def draw(sh: Board, x: float, y: float) -> None:
        y0 = y + LH + 1.0
        for k, (what, head) in enumerate(zip(("auc", "coef"), heads)):
            x0 = x + lw + k * (pw + gap)
            sh.text(x0, y, head, va="top")
            ax = sh.ax(x0, y0, pw, len(rows) * rh)
            for i, r in enumerate(rows):
                t_ = tests[tests["score"] == r].iloc[0]
                yy = i + 0.5
                if what == "auc":
                    ax.plot([t_.auc_ci_low, t_.auc_ci_high], [yy, yy], color=SERIES_COLOR, lw=0.8)
                    ax.plot(t_.auc, yy, "o", color=SERIES_COLOR, ms=2.6)
                elif r != "ln(protein IDs)":
                    ax.plot([t_.ci_sd_low_adjusted_ln_ids, t_.ci_sd_high_adjusted_ln_ids], [yy, yy],
                            color=SERIES_COLOR, lw=0.8)
                    ax.plot(t_.coef_sd_adjusted_ln_ids, yy, "o", color=SERIES_COLOR, ms=2.6)
            ax.set_ylim(len(rows), 0)
            ax.set_yticks([])
            ax.axvline(0.5 if what == "auc" else 0.0, color=T9.INK["muted"], lw=0.5, ls="--")
            if what == "auc":
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            else:
                ax.set_xlim(-lim, lim)
                ax.set_xticks(np.linspace(-lim, lim, 5))
            for i in range(1, len(rows)):
                ax.axhline(i, color=T9.INK["grid"], lw=0.8 if i == i_split else 0.4)
            T8.style_axes(ax)
            ax.spines["left"].set_visible(False)
        for i, r in enumerate(labels):
            sh.text(x + lw - 1.0, y0 + (i + 0.5) * rh, r, ha="right", va="center_baseline")

    return Block(avail, LH + 1.0 + len(rows) * rh + 5.5, draw, out)


# ---------------------------------------------------------------------------
# S-07-b B: subset UMAPs
# ---------------------------------------------------------------------------
def umaps_block(checks, per_cell: pd.DataFrame, colors: dict, avail: float) -> Block:
    from matplotlib import colormaps

    LH = T9.LH
    gap, ph, vgap = 14.0, 44.0, 3.0
    pw = (avail - gap) / 2
    blocks, out = [], []
    for seg, head, exp in (("PT", "Proximal tubule subset", EXPECTED["subset_pt"]),
                           ("TAL", "TAL subset", EXPECTED["subset_tal"])):
        d = per_cell[per_cell["segment"] == seg]
        got = d[COL].astype(str).value_counts().to_dict()
        check(checks, f"S-07-b B UMAP: {seg} subset cells by cell type", got, exp, got == exp)
        order = T9.order_labels(d[COL])
        ent = [(f"{k} ({int((d[COL] == k).sum())})", "") for k in order]
        leg_h = max(len(T9.legend_rows(ent, pw)) * (LH + 0.4), 1.8 + 0.5 + LH + 1.0)
        blocks.append((head, d, order, LH + 1.5 + LH + 0.8 + ph + 0.8 + leg_h))
        out += [{"panel": seg, "row": k, "n_cells": int((d[COL] == k).sum())} for k in order]

    def draw(sh: Board, x: float, y: float) -> None:
        yy = y
        for head, d, order, h in blocks:
            sh.text(x, yy, f"{head} ({len(d)} cells)", fontweight="bold", va="top")
            xy = d[["umap1_primary", "umap2_primary"]].to_numpy(float)
            T9.umap_categorical(sh, x, yy + LH + 1.5, pw, ph, xy, d[COL], order, colors, "Cell type")
            T9.umap_continuous(sh, x + pw + gap, yy + LH + 1.5, pw, ph, xy,
                               np.log(d["n_detected_proteins"].astype(float)), "ln(protein IDs)", colormaps["viridis"])
            yy += h + vgap

    return Block(avail, sum(b[3] for b in blocks) + vgap * (len(blocks) - 1), draw, out)


# ---------------------------------------------------------------------------
# S-07-c B and C: pseudobulk vs the rat segments
# ---------------------------------------------------------------------------
def ratios_block(checks, ratio: pd.DataFrame) -> Block:
    """SCP-MS PCT - PST difference vs each rat log2 ratio, with the depth-stratified permutation range."""
    LH = T9.LH
    d = ratio[(ratio["protein_set"] == "all_shared") & (ratio["label_column"] == COL)]
    rows = [(v, r) for v in ("raw", "depth_adjusted") for r in RAT_SEGMENTS.PT_RATIOS]
    rl = [f"{'adjusted' if v == 'depth_adjusted' else 'raw'}: {txt(r)}" for v, r in rows]
    lw = max(T8.text_mm(s)[0] for s in rl) + 2.0
    rh, pw = 4.6, 60.0
    leg = T9.legend_rows([("observed", SERIES_COLOR), ("permutation 95% range", T9.INK["grid"])], lw + pw)
    out = []
    for v, rn in rows:
        t_ = d[(d["scp_ratio"] == v) & (d["limbutara_ratio"] == rn)].iloc[0]
        out.append({"panel": v, "row": rn, "value": t_.rho, "ci_low": t_.null_q025, "ci_high": t_.null_q975,
                    "n_proteins": int(t_.n_proteins)})

    def draw(sh: Board, x: float, y: float) -> None:
        x0 = x + lw
        ax = sh.ax(x0, y, pw, len(rows) * rh)
        for i, (v, rn) in enumerate(rows):
            t_ = d[(d["scp_ratio"] == v) & (d["limbutara_ratio"] == rn)].iloc[0]
            ax.plot([t_.null_q025, t_.null_q975], [i + 0.5, i + 0.5], color=T9.INK["grid"], lw=2.2,
                    solid_capstyle="butt")
            ax.plot(t_.rho, i + 0.5, "o", color=SERIES_COLOR, ms=2.6)
        ax.set_ylim(len(rows), 0)
        ax.set_yticks([])
        ax.axvline(0, color=T9.INK["muted"], lw=0.5, ls="--")
        ax.set_xlim(-0.3, 0.3)
        ax.set_xticks([-0.3, -0.15, 0, 0.15, 0.3])
        ax.set_xlabel("Spearman ρ")
        for i in range(1, len(rows)):
            ax.axhline(i, color=T9.INK["grid"], lw=0.8 if i == len(RAT_SEGMENTS.PT_RATIOS) else 0.4)
        T8.style_axes(ax)
        ax.spines["left"].set_visible(False)
        for i, s in enumerate(rl):
            sh.text(x0 - 1.0, y + (i + 0.5) * rh, s, ha="right", va="center_baseline")
        T9.draw_legend(sh, x, y + len(rows) * rh + 9.0, leg)

    raw = d[(d["scp_ratio"] == "raw") & (d["limbutara_ratio"] == "S1/S2 - S3")].iloc[0]
    check(checks, "S-07-c B pseudobulk: raw S1/S2 - S3 rho; above the permutation range; proteins",
          f"{raw.rho:.4f}; {raw.rho > raw.null_q975}; {int(raw.n_proteins)}",
          f"{EXPECTED['pb_rho_s1s2_s3']}; True; {EXPECTED['pb_n_ratio_proteins']}",
          round(float(raw.rho), 4) == EXPECTED["pb_rho_s1s2_s3"] and raw.rho > raw.null_q975
          and int(raw.n_proteins) == EXPECTED["pb_n_ratio_proteins"])
    return Block(lw + pw, len(rows) * rh + 9.0 + len(leg) * (LH + 0.4), draw, out)


def matrix_block(checks, mat: pd.DataFrame) -> Block:
    """SCP-MS group means vs rat segment means (z-scored per protein within each data type)."""
    LH = T9.LH
    groups, segs = ("PCT", "PST", "TAL"), RAT_SEGMENTS.STUDY_SEGMENTS
    m = mat[(mat["label_column"] == COL) & (mat["values"] == "z")]
    arr = np.array([[float(m[(m["scp_group"] == g) & (m["limbutara_segment"] == s)]["rho"].iloc[0]) for s in segs]
                    for g in groups])
    n = int(m["n_proteins"].iloc[0])
    check(checks, "S-07-c C pseudobulk matrix: proteins; PCT-S2, PST-S3, TAL-mTAL rho",
          f"{n}; {arr[0, 1]:+.2f}, {arr[1, 2]:+.2f}, {arr[2, 4]:+.2f}", f"{EXPECTED['pb_n_proteins']}; +0.23, +0.25, "
          "+0.48", n == EXPECTED["pb_n_proteins"] and f"{arr[0, 1]:+.2f}, {arr[1, 2]:+.2f}, {arr[2, 4]:+.2f}"
          == "+0.23, +0.25, +0.48")
    cw, rh = 9.0, 4.6
    lw = max(T8.text_mm(g)[0] for g in groups) + 2.0
    out = [{"panel": "z", "row": g, "column": s, "value": arr[a, b], "n_proteins": n}
           for a, g in enumerate(groups) for b, s in enumerate(segs)]

    def draw(sh: Board, x: float, y: float) -> None:
        x0, y0 = x + lw, y + LH + 1.0
        sh.text(x0, y, "Rat segment", va="top")
        ax = sh.ax(x0, y0, len(segs) * cw, len(groups) * rh)
        ax.imshow(arr, cmap=T8.diverging_cmap(), vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
                  extent=(0, len(segs), len(groups), 0))
        ax.set_axis_off()
        for a, g in enumerate(groups):
            sh.text(x0 - 1.0, y0 + (a + 0.5) * rh, g, ha="right", va="center_baseline")
            for b in range(len(segs)):
                sh.text(x0 + (b + 0.5) * cw, y0 + (a + 0.5) * rh, f"{arr[a, b]:+.2f}".replace("-", MINUS),
                        ha="center", va="center_baseline",
                        color="white" if abs(arr[a, b]) > 0.55 else T9.INK["primary"])
        for b, s in enumerate(segs):
            sh.text(x0 + (b + 0.5) * cw, y0 + len(groups) * rh + 0.6, s, ha="center", va="top")

    return Block(lw + len(segs) * cw, LH + 1.0 + len(groups) * rh + 0.6 + LH, draw, out)


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)

    def tab(task: str, name: str) -> pd.DataFrame:
        return pd.read_csv(run.use_input(cc.out_path(task, "tables", name, ".csv"), f"{task} {name}"))

    cols, vals = tab(T9.TASK, "dotplot_columns"), tab(T9.TASK, "dotplot_values")
    conf, t9_tests = tab(T9.TASK, "map_confusion"), tab(T9.TASK, "scores_pct_vs_pst")
    per_cell = tab(T9.TASK, "subsets_per_cell")
    t24_tests = tab(RAT_SEGMENTS.TASK, "scores_pct_vs_pst")
    ratio, mat = tab(RAT_SEGMENTS.TASK, "pseudobulk_pt_ratio"), tab(RAT_SEGMENTS.TASK, "pseudobulk_matrix")
    key = tab("kidney_figure2", "fig2a_umap_key")
    colors = dict(zip(key["label"], key["colour"]))

    checks: list[dict] = []
    used = {k: colors.get(k) for k in ("PCT", "PST", "TAL", "JG", "Unknown")}
    check(checks, "row colours = revised Figure 2A (kidney_figure2 key): PCT, PST, TAL, JG, Unknown; PST brown",
          used, f"all in the kidney_figure2 colour key; PST {PST_COLOR}",
          all(v is not None for v in used.values()) and str(used["PST"]).lower() == PST_COLOR)
    sel9 = t9_tests[(t9_tests["grouping"] == "t15") & t9_tests["gene_set"].isin(["all", "-"])
                    & t9_tests["method"].isin(["mean_z", "-"]) & t9_tests["score"].isin(ATLAS_ROWS)]
    sel24 = t24_tests[(t24_tests["grouping"] == "t15") & t24_tests["threshold"].isin([RAT_SEGMENTS.PRIMARY_THR, "-"])
                      & t24_tests["gene_set"].isin(["all", "-"]) & t24_tests["method"].isin(["mean_z", "-"])
                      & t24_tests["score"].isin(RAT_ROWS)]
    for nm, sel, rows in (("atlas", sel9, ATLAS_ROWS), ("rat", sel24, RAT_ROWS)):
        dup = sel.groupby("score").size()
        check(checks, f"{nm} scores: one revised-annotation row per plotted score", f"{len(dup)} scores, max "
              f"{int(dup.max())} row(s)", f"{len(rows)} scores, 1 row", len(dup) == len(rows) and int(dup.max()) == 1)
    a = sel9.set_index("score")
    r = sel24.set_index("score")
    check(checks, "S1/S2 - S3 AUC: atlas, rat, atlas reference row in the rat table; ln(protein IDs)",
          f"{a.loc['S1/S2 - S3', 'auc']:.4f}, {r.loc['S1/S2 - S3', 'auc']:.4f}, "
          f"{r.loc['pt_tal atlas: S1/S2 - S3', 'auc']:.4f}; {a.loc['ln(protein IDs)', 'auc']:.4f}",
          f"{EXPECTED['atlas_auc_s1s2_s3']}, {EXPECTED['rat_auc_s1s2_s3']}, {EXPECTED['atlas_auc_s1s2_s3']}; "
          f"{EXPECTED['auc_ln_ids']}",
          round(a.loc["S1/S2 - S3", "auc"], 4) == EXPECTED["atlas_auc_s1s2_s3"]
          and round(r.loc["S1/S2 - S3", "auc"], 4) == EXPECTED["rat_auc_s1s2_s3"]
          and round(r.loc["pt_tal atlas: S1/S2 - S3", "auc"], 4) == EXPECTED["atlas_auc_s1s2_s3"]
          and round(a.loc["ln(protein IDs)", "auc"], 4) == EXPECTED["auc_ln_ids"])

    T9.setup()                                # Arial 7 pt style; sets T9.LH
    avail = T8.W - 2 * T8.M
    fig_rows: list[dict] = []
    out = assemble(run, fig_rows, "07_S07a_atlas_markers",
                   "Supplementary Figure S-07-a: atlas subtype markers (A) and nearest atlas subtype (B), revised "
                   "annotation",
                   [[("A", dotplot_block(checks, cols, vals, colors, avail))], [("B", mapping_block(checks, conf))]])
    out += assemble(run, fig_rows, "07_S07b_scores_depth",
                    "Supplementary Figure S-07-b: PCT vs PST on the atlas subtype scores (A) and subset UMAPs (B), "
                    "revised annotation",
                    [[("A", effects_block(sel9, ATLAS_ROWS, "ln(protein IDs)", avail))],
                     [("B", umaps_block(checks, per_cell, colors, avail))]])
    out += assemble(run, fig_rows, "07_S07c_rat_segments",
                    "Supplementary Figure S-07-c: PCT vs PST on the rat segment scores (A), pseudobulk vs rat "
                    "segment ratios (B) and segments (C), revised annotation",
                    [[("A", effects_block(sel24, RAT_ROWS, "pt_tal atlas: S1/S2 - S3", avail))],
                     [("B", ratios_block(checks, ratio)), ("C", matrix_block(checks, mat))]])
    figs = pd.DataFrame(fig_rows)
    check(checks, "figures pass the format checks (Arial >= 7 pt, no text overlap, text inside, PDF + PNG, 180 mm)",
          f"{int(figs['pass'].sum())} of {len(figs)}", "3 of 3", bool(figs["pass"].all()) and len(figs) == 3)
    check(checks, "figure heights (mm; one A4 page with the legend: at most 230)",
          ", ".join(f"{h:.0f}" for h in figs["height_mm"]), "each <= 230", bool((figs["height_mm"] <= 230).all()))
    for c in checks:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['observed']}")
    run.table(pd.DataFrame(out), f"{STEP}_values")
    run.table(figs, f"{STEP}_figure_checks")
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    all_ok = all(c["pass"] for c in checks)
    run.finish({"label_column": COL, "map_variant": MAP_VARIANT, "rat_threshold": RAT_SEGMENTS.PRIMARY_THR,
                "score_method": "mean_z", "gene_set": "all", "pseudobulk_protein_set": "all_shared",
                "style": "one file per supplementary figure, 180 mm wide, panel letters A-C, no title or footnote",
                "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit("A check failed; see report_07_s07_panels_checks.csv.")


if __name__ == "__main__":
    main()
