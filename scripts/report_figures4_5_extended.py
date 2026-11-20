#!/usr/bin/env python
"""The response to the reviewers: Figures 4 and 5 extended, and Supplementary Figure S26 without its first panel.

The authors asked on 19 Nov 2026 for more in Figures 4 and 5
(final numbering: Figure 4 = the DCT figure, Figure 5 = the zero-K panel figure) and chose, after the stored seed
scan was shown (a zero-K DCT2 cluster is annotated in 9 of 11 Leiden seeds at resolution 1): "lower DCT2 markers,
The current panels stay as further panels".

Figure 4 (six panels, 180 mm wide; panels at their drawn size, placed pixel for pixel as in the response to the reviewers):
  a  control cells on their UMAP, cluster annotation (no ProLIF cells drawn)
  b  zero-K cells on their UMAP, cluster annotation at the seed of the study
  c  marker proteins in the five annotated groups of both diets, one colour scale over both diets (mean z-score of
     the matrix scaled over all 1,431 cells; the dotplot dot-plot rule, dotplot_lib.group_values / scaled_01)
  d  cells annotated DCT2 at each of the 11 Leiden seeds of the stored scan (resolution 1, rank-free rule)
  e  the current panel a (control UMAP with the 37 ProLIF cells), drawn by the response to the reviewers's function, unchanged
  f  the current panel b (protein above RNA dot plot), drawn by the response to the reviewers's function, unchanged
Figure 5 (180 mm): panels a and b as in the response to the reviewers (drawn by the the response to the reviewers functions, journal-style strings), plus
  c  the first panel of Supplementary Figure S26: zero-K against control in DCT cells against the bulk
     whole-kidney proteome on a 0% or 1% potassium diet
Supplementary Figure S26, second version: its two other panels, relettered A and B.
Panel letters: lower case, Arial Bold 15 pt (the size of the main-figure TIFFs since 19 Nov 2026).
Journal style: sentence-case axis labels, four-digit numbers without a comma, capital italic P.

No new test. New descriptive values: the group means of panel c over both diets (stored object, dotplot rule). All
other values are read from stored tables. No output of an earlier step is written.

Inputs (read-only): results/h5ad/dct_labels_DCT_{ctrl,zerok,joint}_labelled.h5ad;
  results/tables/dct_labels/dct_labels_{marker_order_dct,dotplot_values,scan_seed_stability}.csv;
  results/tables/zerok_vs_control/zerok_vs_control_{panel_results,specificity_matched_null,
  specificity_null_draws_summary,bulk_matched,bulk_concordance,bulk_plate_null,bulk_sign_tables}.csv;
  the inputs of the functions that draw panels e and f (dct_prolif overlay table, the response to the reviewers and dct_rna dot-plot values);
  results/figures/report/report_05_fig3a_umap_85mm.{pdf,png}, report_27_fig3b_ki_style.{pdf,png},
  report_27_figF19_ki_style.png (comparison).
Outputs: results/figures/report/report_19_fig4_extended.{pdf,png,tif},
  report_19_fig4_panel_{a,b,c,d}.{pdf,png}, report_19_fig5_extended.{pdf,png,tif},
  report_19_S19d_bulk_benchmark_v2.{pdf,png,tif};
  results/tables/report/report_19_fig45_extended_{values,checks,layout}.csv;
  results/tables/report/report_step_19_fig45_extended_manifest.json

Command (project root):  python scripts/report_figures4_5_extended.py --overwrite
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
from contextlib import contextmanager
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import report_zerok_figures as z19          # The response to the reviewers drawing functions (set the atlas_dct figure style on import)

import matplotlib

RC_CTRL_REVIEWER_CORE = matplotlib.rcParams.copy()                 # the style in which the response to the reviewers draws (restored before its plots)

import dotplot_lib as t6                        # noqa: E402
import report_figure4 as f3             # noqa: E402  the response to the reviewers: current Figure 4 panels (draw_a, draw_b)
import report_figure5 as F19          # noqa: E402  the response to the reviewers: journal-style strings and the italic P
import report_figure_style_lib as R               # noqa: E402

import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402
from matplotlib.text import Text                   # noqa: E402

a3, b3, t6d = f3.a3, f3.b3, f3.t6d
ov31, t31 = a3.ov31, a3.t31
t8, lib30, t22, pick = z19.t8, z19.lib30, z19.t22, z19.pick
INK, FONT, COND, PRI = z19.INK, z19.FONT, z19.COND, z19.PRI

TASK, STEP = "report", "19_fig45_extended"
PREFIX = "report_19_"
LETTER_PT = 15.0
FIG_DIR = cc.OUT_ROOTS["figures"] / TASK
DCT_LABELS = cc.OUT_ROOTS["tables"] / "dct_labels"
H5 = {k: cc.OUT_ROOTS["h5ad"] / f"dct_labels_DCT_{k}_labelled.h5ad" for k in ("ctrl", "zerok", "joint")}
OLD_E = FIG_DIR / "report_05_fig3a_umap_85mm"
OLD_F = FIG_DIR / "report_27_fig3b_ki_style"
OLD_F19 = FIG_DIR / "report_27_figF19_ki_style.png"
STRIP_MM, ROW_GAP_MM = 5.0, 2.0          # blank strip above each row for the 15 pt letters; gap between rows
W_C, W_D = 110.0, 67.0                   # widths of panels c and d (mm); 110 + 3 + 67 = 180
LABELS = {"Ctrl": "cs_ctrl_deg_label", "ZeroK": "cs_zerok_pipeline_label"}
EXPECTED = {"Ctrl": {"DCT1": 382, "DCT2": 85, "Unknown": 262}, "ZeroK": {"DCT1": 449, "Unknown": 253}}
ROWS_C = [("Ctrl", "DCT1"), ("Ctrl", "DCT2"), ("Ctrl", "Unknown"), ("ZeroK", "DCT1"), ("ZeroK", "Unknown")]
ROW_NAME = {("Ctrl", "DCT1"): "Control, DCT1", ("Ctrl", "DCT2"): "Control, DCT2",
            ("Ctrl", "Unknown"): "Control, not resolved", ("ZeroK", "DCT1"): "Zero-K, DCT1",
            ("ZeroK", "Unknown"): "Zero-K, not resolved"}
BANNED = ("ctrl", "zerok", "unknown", "generated", "pipeline", "primary", "alternative", "submitted",
          "label_free", "_", ".csv", "t27", "t30", "frozen", "seed 42")
VALUES: list[dict] = []


def val(figure, panel, row, series, value, low=np.nan, high=np.nan, source="") -> None:
    VALUES.append({"figure": figure, "panel": panel, "row": row, "series": series, "value": value, "low": low,
                   "high": high, "source": source})


@contextmanager
def lib_letters(pt: float):
    """report_figure_style_lib with its letter size set to ``pt`` and its writer accepting this script's names."""
    old_pt, old_guard = R.LETTER_PT, R.guard
    R.LETTER_PT = pt
    R.guard = lambda p: Path(p) if Path(p).name.startswith(PREFIX) else old_guard(p)
    try:
        yield
    finally:
        R.LETTER_PT, R.guard = old_pt, old_guard


def t19_style() -> None:
    matplotlib.rcParams.update(RC_CTRL_REVIEWER_CORE)


def banned(fig) -> list[str]:
    fig.canvas.draw()
    texts = [t.get_text() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return sorted({t for t in texts if any(b in t.lower() for b in BANNED)})


def both(fig, what: str) -> dict[int, np.ndarray]:
    return {dpi: R.opaque_rgb(R.render(fig, dpi), f"{what} at {dpi} dpi") for dpi in (R.DPI_PNG, R.DPI_TIF)}


# ---------------------------------------------------------------------------------------------------------------------
# Figure 4, new panels
# ---------------------------------------------------------------------------------------------------------------------
def umap_panel(umap: np.ndarray, label: np.ndarray, title: str):
    """UMAP by cluster annotation, drawn as the current panel (report_dct_umap.panel) without ProLIF."""
    a3.t28.style()
    fig, ax = plt.subplots(figsize=(ov31.W85 * a3.t28.MM, ov31.H80 * a3.t28.MM))
    counts = {g: int((label == g).sum()) for g in ov31.LEGEND_ORDER if g != t31.PROLIF}
    for g in ov31.DRAW_ORDER:
        m = label == g
        if g != t31.PROLIF and m.any():
            ax.scatter(umap[m, 0], umap[m, 1], s=ov31.POINT_SIZE[g], color=ov31.GROUP_COLORS[g], lw=0,
                       rasterized=True, zorder=2)
    shown = [g for g in ov31.LEGEND_ORDER if counts.get(g)]
    labels = [f"{a3.DISPLAY.get(g, g)} ({counts[g]})" for g in shown]
    handles = [Line2D([], [], marker="o", ls="", ms=3.0, color=ov31.GROUP_COLORS[g], label=lab)
               for g, lab in zip(shown, labels)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2 if len(shown) > 2 else 1,
              frameon=False, handletextpad=0.1, columnspacing=0.6, borderaxespad=0)
    a3.t18f.clean(ax)
    ax.set_title(title, fontsize=7.5, fontweight="bold", loc="center", pad=2)
    fig.tight_layout()
    return fig, counts, labels


def dot_values(run, checks) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group values of panel c over both diets (dotplot rule on the matrix scaled over all cells)."""
    joint = ad.read_h5ad(run.use_input(H5["joint"], "dct_labels labelled joint object (z-scored over all 1,431 cells)"))
    cond = joint.obs["condition"].astype(str).to_numpy()
    lab = np.where(cond == "Ctrl", joint.obs[LABELS["Ctrl"]].astype(str), joint.obs[LABELS["ZeroK"]].astype(str))
    keys = [f"{c}|{g}" for c, g in ROWS_C]
    groups = pd.Series([f"{c}|{g}" for c, g in zip(cond, lab)], index=joint.obs_names.astype(str))
    order = pd.read_csv(run.use_input(DCT_LABELS / "dct_labels_marker_order_dct.csv", "dct_labels marker order (DCT)"))
    order = order[(order["version"] == "panel") & order["plotted"]].sort_values("plotted_position").reset_index(drop=True)
    markers = order["marker"].tolist()
    vals = t6.group_values(joint, groups, keys, markers)
    vals.insert(0, "label_column", "cluster annotation, both diets")
    vals["mean_z_scaled_0_1"] = t6.scaled_01(vals)
    n = vals.drop_duplicates("group").set_index("group")["n_cells"].to_dict()
    want = {f"{c}|{g}": EXPECTED[c][g] for c, g in ROWS_C}
    checks.add("panel c: cells per row (control DCT1, DCT2, not resolved; zero-K DCT1, not resolved); markers",
               "; ".join(f"{k} {n[k]}" for k in keys) + f"; {len(markers)} markers", n == want and len(markers) == 13,
               "382; 85; 262; 449; 253; 13")
    z = cc.dense(joint.X[:, pd.Index(joint.var_names.astype(str)).get_indexer(markers)]).astype(float)
    checks.add("panel c: the matrix is z-scored over the cells of both diets and clipped at +/-10 (mean and SD of the "
               "13 markers; clipping moves them slightly from 0 and 1)",
               f"largest |mean| {np.abs(z.mean(axis=0)).max():.1e}; SD {z.std(axis=0, ddof=1).min():.3f}-"
               f"{z.std(axis=0, ddof=1).max():.3f}; largest |z| {np.abs(z).max():.1f}; {joint.n_obs} cells",
               np.abs(z.mean(axis=0)).max() < 0.02 and 0.85 < z.std(axis=0, ddof=1).min()
               and z.std(axis=0, ddof=1).max() < 1.001 and np.abs(z).max() <= 10.0 and joint.n_obs == 1431)
    stored = pd.read_csv(run.use_input(DCT_LABELS / "dct_labels_dotplot_values.csv", "dct_labels dot-plot values per condition"))
    worst, n_cmp = 0.0, 0
    for c, g in ROWS_C:
        s = stored[(stored["condition"] == c) & (stored["label_column"] == LABELS[c]) & (stored["group"] == g)
                   & stored["marker"].isin(markers)].set_index("marker")["frac_detected"]
        mine = vals[vals["group"] == f"{c}|{g}"].set_index("marker")["frac_detected"]
        worst = max(worst, float((mine - s.reindex(mine.index)).abs().max()))
        n_cmp += int(s.reindex(mine.index).notna().sum())
    checks.add("panel c: fraction of cells with the protein detected equals the stored dct_labels values per condition",
               f"{n_cmp} of {len(vals)} values compared; largest difference {worst:.1e}",
               n_cmp == len(vals) == 65 and worst < 1e-12, "65; < 1e-12")
    return vals, order


def dot_panel(vals: pd.DataFrame, order: pd.DataFrame):
    b3.arial()
    t6d.SIZE_TITLE = "Fraction of cells (%) with\nintensity > 0"
    t6d.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein)"
    markers = order["marker"].tolist()
    keys = [f"{c}|{g}" for c, g in ROWS_C]
    frac = vals.pivot(index="group", columns="marker", values="frac_detected").reindex(index=keys, columns=markers)
    colr = vals.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=keys, columns=markers)
    n = vals.drop_duplicates("group").set_index("group")["n_cells"].to_dict()
    panel = t6d.Panel("", [ROW_NAME[k] for k in ROWS_C], [n[k] for k in keys],
                      [b3.ROW_COLOURS[g] for _, g in ROWS_C], markers, t6d.groups_of(order), frac.to_numpy(float),
                      colr.to_numpy(float), [False] * len(markers))
    return t6d.build_figure([panel], W_C, "below", 4.5)


def seed_values(run, checks) -> pd.DataFrame:
    s = pd.read_csv(run.use_input(DCT_LABELS / "dct_labels_scan_seed_stability.csv", "dct_labels scan: annotation per Leiden seed"))
    s = s[(s["rule"] == "rankfree") & np.isclose(s["resolution"].astype(float), 1.0)].copy()
    s["n_dct2"] = [int(m.group(1)) if (m := re.search(r"DCT2 (\d+)", str(x))) else 0 for x in s["label_counts"]]
    s["n_cells"] = s["condition"].map({"Ctrl": 729, "ZeroK": 702})
    s["pct_dct2"] = 100 * s["n_dct2"] / s["n_cells"]
    got = {c: (int((d["n_dct2"] > 0).sum()), len(d), int(d.loc[d["seed"] == 42, "n_dct2"].iloc[0]))
           for c, d in s.groupby("condition")}
    checks.add("panel d: seeds with a DCT2 cluster / seeds scanned / DCT2 cells at the seed of the study, control; "
               "zero-K", f"{got['Ctrl']}; {got['ZeroK']}", got == {"Ctrl": (11, 11, 85), "ZeroK": (9, 11, 0)},
               "(11, 11, 85); (9, 11, 0)")
    rng = {c: (int(d.loc[d["n_dct2"] > 0, "n_dct2"].min()), int(d["n_dct2"].max())) for c, d in s.groupby("condition")}
    checks.add("panel d: DCT2 cells in the seeds with a DCT2 cluster (min, max), control; zero-K",
               f"{rng['Ctrl']}; {rng['ZeroK']}", rng == {"Ctrl": (58, 103), "ZeroK": (51, 152)}, "(58, 103); (51, 152)")
    return s


def seed_panel(s: pd.DataFrame, height: float):
    t19_style()
    sheet = t8.Sheet(height, width=W_D)
    ax = sheet.ax(15.0, 4.0, W_D - 15.0 - 4.0, height - 4.0 - 17.0)
    for i, cond in enumerate(("Ctrl", "ZeroK")):
        d = s[s["condition"] == cond].sort_values(["pct_dct2", "seed"]).reset_index(drop=True)
        off = np.array([((k % 5) - 2) * 0.09 for k in range(len(d))])
        used = d["seed"].to_numpy() == 42
        ax.plot(i + off[~used], d.loc[~used, "pct_dct2"], "o", ms=3.6, mfc="white", mec=COND[cond], mew=0.8, ls="")
        ax.plot(i + off[used], d.loc[used, "pct_dct2"], "o", ms=4.6, mfc=COND[cond], mec=INK["primary"], mew=0.6, ls="",
                zorder=3)
        for r in d.itertuples():
            val("Figure 4", "d", z19.COND_NAME[cond], f"Leiden seed {r.seed}: cells annotated DCT2 (%); cells in low",
                float(r.pct_dct2), int(r.n_dct2), int(r.n_cells), "dct_labels_scan_seed_stability.csv")
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([z19.COND_NAME["Ctrl"], z19.COND_NAME["ZeroK"]])
    ax.set_ylim(-1.5, 25)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_ylabel("Cells annotated DCT2 (%)", labelpad=1.5)
    z19.style(ax)
    handles = [Line2D([], [], marker="o", ls="", ms=4.6, mfc=INK["muted"], mec=INK["primary"], mew=0.6,
                      label="Clustering of this study"),
               Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=INK["muted"], mew=0.8,
                      label="10 other random starts")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.22, -0.26), ncol=1, frameon=False,
              handletextpad=0.2, borderaxespad=0, labelspacing=0.3)
    return sheet


# ---------------------------------------------------------------------------------------------------------------------
# Figure 5, new panel; Supplementary Figure S26 without its first panel
# ---------------------------------------------------------------------------------------------------------------------
def draw_bulk(sheet, T, checks, side: float = 46.0) -> None:
    """Panel A of Supplementary Figure S26, journal style; leader lines drawn as lines."""
    d = pick(T["bulk_matched"], label_column=PRI, group="all_DCT", variant="none")
    ax = sheet.ax(16.0, 4.0, side, side)
    x, y = d["bulk_log2fc"].to_numpy(float), d["pseudobulk_log2fc_pooled"].to_numpy(float)
    sig = (d["bulk_q"] < 0.05).to_numpy()
    enr = d["dct_enriched"].astype(str).str.lower().eq("true").to_numpy()
    ax.scatter(x[~sig], y[~sig], s=1.2, c=INK["grid"], linewidths=0, rasterized=True)
    ax.scatter(x[sig], y[sig], s=1.6, c=INK["muted"], linewidths=0, rasterized=True)
    ax.scatter(x[enr], y[enr], s=5, c=t8.SERIES[0], linewidths=0, zorder=3)
    lx = float(np.nanquantile(np.abs(x), 0.999)) * 1.05
    ly = float(np.nanquantile(np.abs(y[np.isfinite(y)]), 0.995)) * 1.05
    ax.set_xlim(-lx, lx)
    ax.set_ylim(-ly, ly)
    ax.axhline(0, color=INK["muted"], lw=0.5, zorder=0)
    ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    pts = []
    for g in z19.PROT:
        r = d.loc[d["protein"] == g]
        if len(r) and np.isfinite(r["pseudobulk_log2fc_pooled"].iloc[0]):
            px = float(np.clip(r["bulk_log2fc"].iloc[0], -lx, lx))
            py = float(np.clip(r["pseudobulk_log2fc_pooled"].iloc[0], -ly, ly))
            ax.scatter([px], [py], s=16, facecolors="none", edgecolors=INK["primary"], linewidths=0.6, zorder=4,
                       clip_on=False)
            pts.append((g, px, py))
            val("Figure 5", "c", g, "ring: bulk log2 fold change (value), DCT-cell log2 ratio (low)",
                float(r["bulk_log2fc"].iloc[0]), float(r["pseudobulk_log2fc_pooled"].iloc[0]),
                source="zerok_vs_control_bulk_matched.csv")
    line = dict(color=INK["muted"], lw=0.4, clip_on=False, zorder=3.5)
    step = 3.6 / side * (2 * ly)                         # 3.6 mm in data units
    right = sorted([p for p in pts if p[1] >= 0], key=lambda t: t[2])
    ys = [min(max(py, -0.94 * ly), 0.94 * ly) for _, _, py in right]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + step)
    over = (ys[-1] - 0.96 * ly) if ys else 0.0
    if over > 0:
        ys = [v - over for v in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - step)
    for (g, px, py), ty in zip(right, ys):
        ax.plot([px, 1.07 * lx], [py, ty], **line)
        ax.text(1.09 * lx, ty, g, va="center", ha="left", fontsize=FONT, clip_on=False)
    left = sorted([p for p in pts if p[1] < 0], key=lambda t: -t[1])
    for i, (g, px, py) in enumerate(left):
        ty = -0.35 * ly - i * step
        ax.plot([px, -0.64 * lx], [py, ty], **line)
        ax.text(-0.62 * lx, ty, g, va="center", ha="left", fontsize=FONT)
    c = pick(T["bulk_conc"], label_column=PRI, group="all_DCT", variant="none", scp_effect="pseudobulk",
             protein_set="all shared").iloc[0]
    ax.text(0.03, 0.97, f"ρ = {c['spearman']:.2f}\nn = {int(c['n'])}", transform=ax.transAxes, va="top", fontsize=FONT)
    ax.set_xlabel("Whole kidney, 0% vs 1% potassium diet\n(log2 fold change)", labelpad=1.5)
    ax.set_ylabel("DCT cells, zero-K vs control\n(log2 ratio)", labelpad=1.5)
    z19.style(ax)
    val("Figure 5", "c", "all shared proteins", "points drawn (low: bulk q < 0.05; high: DCT-enriched)", int(len(d)),
        int(sig.sum()), int(enr.sum()), "zerok_vs_control_bulk_matched.csv")
    val("Figure 5", "c", "all shared proteins", "Spearman rho (n in low)", float(c["spearman"]), int(c["n"]),
        source="zerok_vs_control_bulk_concordance.csv")
    checks.add("Figure 5c: shared proteins drawn; Spearman rho; rings (panel proteins found in the bulk data)",
               f"{len(d)}; {c['spearman']:.2f} (n {int(c['n'])}); {', '.join(g for g, _, _ in pts)}",
               (len(d), round(float(c["spearman"]), 2), int(c["n"])) == (3445, 0.13, 3445) and len(pts) == 7,
               "3445; 0.13 (n 3445); 7 proteins (Klhl3 is not in the bulk data)")


def figure5(run, T, checks) -> dict:
    t19_style()
    name = "19_fig5_extended"
    pad, gap = 3.0, 3.0
    y_b = pad + z19.PANEL_H + gap
    side = 46.0
    # Height: the current figure (127.2 mm) plus a whole number of 600-dpi pixels, so that panels a and b fall on
    # the same pixels as in the current figure (matplotlib measures from the bottom of the canvas).
    base = y_b + pad + z19.MATCHED_H
    extra_px = int(np.ceil((y_b + pad + 4.0 + side + 11.0 - base) * R.DPI_PNG / 25.4))
    sheet = t8.Sheet(base + extra_px * 25.4 / R.DPI_PNG)
    scratch = t5.Checks("dct_joint")
    z19.RECORD = False
    try:
        z19.draw_panel(z19.Sub(sheet, 0.0, pad), T, name, scratch)
        z19.draw_matched(z19.Sub(sheet, 0.0, y_b + pad), T, name, scratch)
    finally:
        z19.RECORD = True
    draw_bulk(z19.Sub(sheet, 90.0, y_b + pad), T, checks, side)
    fig = sheet.fig
    z19.clean_ticks(fig)
    checks.add("Figure 5a, 5b: the response to the reviewers checks of the two plots", f"{sum(r['pass'] for r in scratch.rows)} of "
               f"{len(scratch.rows)} pass", scratch.ok)
    strings = {k: v for k, v in F19.STRINGS.items() if k not in ("A", "B")}
    changes = R.edit_texts(fig, R.replace_exact(strings))
    checks.add("Figure 5: axis labels in sentence case", R.changes_text(changes),
               sorted((c["old"], c["new"]) for c in changes) == sorted(strings.items()), f"{len(strings)} labels")
    p = F19.italic_p(fig)
    checks.add("Figure 5b: P value written with a capital italic P", "".join(t.get_text() for t in p["runs"]) + "; "
               + p["italic_file"], p["italic_file"] == "ariali.ttf"
               and "".join(t.get_text() for t in p["runs"]) == "empirical P = 0.0001")
    cells = {"a": (0.0, 0.0), "b": (0.0, y_b), "c": (90.0, y_b)}
    for letter, (x, y) in cells.items():
        sheet.text(x + 1.0, y + 0.8, letter, fontsize=LETTER_PT, fontweight="bold", va="top", ha="left")
    texts = [t for t in fig.findobj(Text) if t is not p["annotation"]]
    tc = R.text_layout(fig, texts)
    lone = [o for o in tc["overlaps"].split("; ") if o and "empirical " not in o and "'P'" not in o]
    arial = all(f in ("arial.ttf", "arialbd.ttf", "ariali.ttf") for f in tc["fonts"].split(";"))
    checks.add("Figure 5: format (texts >= 7 pt, Arial, no text overlap other than the runs of the P-value line, no "
               "text outside)", f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; {tc['fonts']}; other overlaps "
               f"{lone or 'none'}; outside {tc['n_text_outside_figure']}",
               tc["min_font_pt"] >= 7 and arial and not lone and tc["n_text_outside_figure"] == 0)
    bad = banned(fig)
    checks.add("Figure 5: reader-facing text only", f"{len(bad)} texts with an internal name {bad[:4]}", not bad)
    img = both(fig, "Figure 5")
    old, _ = R.read_image(run.use_input(OLD_F19, "current Figure 5"))
    new = img[R.DPI_PNG]
    h = min(old.shape[0], new.shape[0])
    diff = (old[:h] != new[:h, :old.shape[1]]).any(axis=2)
    keep = np.ones_like(diff)
    for x, y in cells.values():                              # letters (old 8.5 pt and new 15 pt)
        keep[R.px(y, R.DPI_PNG):R.px(y + 6.5, R.DPI_PNG), R.px(x, R.DPI_PNG):R.px(x + 9.0, R.DPI_PNG)] = False
    keep[R.px(y_b, R.DPI_PNG):, R.px(88.0, R.DPI_PNG):] = False          # the new panel c
    n_out = int((diff & keep).sum())
    checks.add("Figure 5: panels a and b equal the current figure pixel for pixel (600 dpi), outside the letters and "
               "the place of the new panel c", f"{n_out} pixels differ; old {old.shape[1]} x {old.shape[0]} px, new "
               f"{new.shape[1]} x {new.shape[0]} px", n_out == 0 and old.shape[1] == new.shape[1], "0")
    desc = ("Figure 5 extended: (a) pre-specified panel proteins, zero-K against control; (b) the 7 directional "
            "proteins against 10,000 matched random protein sets; (c) DCT-cell effects against a bulk whole-kidney "
            "proteome on a 0% or 1% potassium diet")
    with lib_letters(LETTER_PT):
        R.save_tif(run.out("figures", name, ".tif"), img[R.DPI_TIF], desc)
    size = (sheet.W, sheet.H)
    pdf, png = run.figure(fig, name, desc, formats=(".pdf", ".png"), tight=False)
    return {"name": name, "size_mm": size, "png": png, "pdf": pdf}


def s19d_v2(run, T, checks) -> dict:
    """Supplementary Figure S26 without its first panel: correlations (A) and direction (B), as the response to the reviewers drew them."""
    t19_style()
    name = "19_S19d_bulk_benchmark_v2"
    nulls, signs = T["bulk_null"], T["bulk_signs"]
    rows5 = z19.ROWS5
    lab5 = [z19.GROUP[k] for k in rows5]
    sheet = t8.Sheet(84.0)
    marks = {"ZeroK vs Ctrl (pooled)": ("o", INK["primary"], INK["primary"]),
             "ZeroK vs Ctrl, P1 only": ("^", z19.PLATE["P1"], z19.PLATE["P1"]),
             "ZeroK vs Ctrl, P2 only": ("s", z19.PLATE["P2"], z19.PLATE["P2"]),
             "Ctrl P2 vs P1 (technical)": ("o", COND["Ctrl"], "white"),
             "ZeroK P2 vs P1 (technical)": ("o", COND["ZeroK"], "white")}
    z19.letter(sheet, 3.0, 3.0, "A")
    for k, (pset, title) in enumerate((("all shared", "all shared proteins"),
                                       ("bulk q < 0.05", "proteins with q < 0.05 in whole kidney"))):
        ax = sheet.ax(30.0, 24.0 + k * 30.0, 48.0, 21.0)
        s0 = pick(nulls, variant="none", protein_set=pset)
        for (cname, (mk, ec, fc)), off in zip(marks.items(), np.linspace(-0.28, 0.28, len(marks))):
            s = pick(s0, contrast=cname).set_index(["label_column", "group"]).reindex(rows5)
            yy = np.arange(len(rows5)) + off
            ax.hlines(yy, s["ci_low"], s["ci_high"], color=ec, lw=0.6)
            ax.plot(s["spearman"], yy, mk, ms=3, mec=ec, mfc=fc, mew=0.6, ls="", label=z19.CONTRAST[cname])
            for key, r in s.iterrows():
                val("S-19-d v2", "A", z19.GROUP[key], f"{title}; {z19.CONTRAST[cname]}: Spearman rho (Fisher-z 95% "
                    f"interval); n = {int(r['n'])}", float(r["spearman"]), float(r["ci_low"]), float(r["ci_high"]),
                    "zerok_vs_control_bulk_plate_null.csv")
        ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
        ax.set_yticks(range(len(rows5)))
        ax.set_yticklabels(lab5)
        ax.set_ylim(len(rows5) - 0.5, -0.5)
        ax.set_title(title, fontsize=FONT, pad=2)
        if k == 1:
            ax.set_xlabel("Spearman ρ with whole kidney", labelpad=1.5)
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(-0.52, 1.16), ncol=2, frameon=False, handletextpad=0.2,
                      columnspacing=0.6, borderaxespad=0)
        z19.style(ax)
    z19.letter(sheet, 98.0, 3.0, "B")
    ax = sheet.ax(128.0, 24.0, 44.0, 21.0)
    for k, (pset, colr, lab) in enumerate((("all shared", INK["primary"], "all shared proteins"),
                                           ("DCT-enriched", t8.SERIES[0], "DCT-enriched proteins"))):
        s = pick(signs, variant="none", protein_set=pset).set_index(["label_column", "group"]).reindex(rows5)
        yy = np.arange(len(rows5)) + (k - 0.5) * 0.3
        ax.hlines(yy, 100 * s["agreement_ci_low"], 100 * s["agreement_ci_high"], color=colr, lw=0.6)
        ax.plot(100 * s["agreement"], yy, "o", ms=3, color=colr, ls="", label=lab)
        for key, r in s.iterrows():
            val("S-19-d v2", "B", z19.GROUP[key], f"{lab}: same direction (%), Wilson 95% interval; proteins "
                f"significant in both: {int(r['n_both_significant'])}", 100 * float(r["agreement"]),
                100 * float(r["agreement_ci_low"]), 100 * float(r["agreement_ci_high"]), "zerok_vs_control_bulk_sign_tables.csv")
    ax.axvline(50, color=INK["muted"], lw=0.5, zorder=0)
    ax.set_yticks(range(len(rows5)))
    ax.set_yticklabels(lab5)
    ax.set_ylim(len(rows5) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("same direction (%)", labelpad=1.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=1, frameon=False, handletextpad=0.2, borderaxespad=0)
    z19.style(ax)
    fig = sheet.fig
    z19.clean_ticks(fig)
    tc = t8.text_checks(fig)
    bad = banned(fig)
    checks.add("S-19-d second version: format (texts >= 7 pt, no overlap, none outside) and reader-facing text",
               f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; overlaps {tc['n_text_overlaps']} {tc['overlaps'][:80]}; "
               f"outside {tc['n_text_outside_figure']}; internal names {bad[:3] or 'none'}",
               tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0 and not bad)
    desc = ("S-19-d (second version, without the per-protein panel that moved to Figure 5): correlations with the "
            "bulk whole-kidney proteome (A) and direction of the proteins significant in both (B)")
    img = both(fig, "S-19-d v2")
    with lib_letters(LETTER_PT):
        R.save_tif(run.out("figures", name, ".tif"), img[R.DPI_TIF], desc)
    size = (sheet.W, sheet.H)
    pdf, png = run.figure(fig, name, desc, formats=(".pdf", ".png"), tight=False)
    return {"name": name, "size_mm": size, "png": png, "pdf": pdf}


# ---------------------------------------------------------------------------------------------------------------------
# Figure 4, compiled
# ---------------------------------------------------------------------------------------------------------------------
def figure4(run, checks) -> dict:
    name = "19_fig4_extended"
    imgs: dict[str, dict[int, np.ndarray]] = {}
    pdfs: dict[str, Path] = {}
    # a, b: UMAPs by cluster annotation
    for key, cond, title in (("a", "Ctrl", "Control"), ("b", "ZeroK", "Zero-K")):
        obj = ad.read_h5ad(run.use_input(H5["ctrl" if cond == "Ctrl" else "zerok"], f"dct_labels labelled object ({cond})"))
        lab = obj.obs[LABELS[cond]].astype(str).to_numpy()
        fig, counts, labels = umap_panel(np.asarray(obj.obsm["X_umap"], float), lab, title)
        checks.add(f"panel {key}: {title} cells per cluster annotation; key", "; ".join(labels),
                   {g: n for g, n in counts.items() if n} == EXPECTED[cond]
                   and obj.n_obs == sum(EXPECTED[cond].values()), str(EXPECTED[cond]))
        bad = banned(fig)
        checks.add(f"panel {key}: reader-facing text only", f"{bad or 'no internal name'}", not bad)
        for g, n in counts.items():
            val("Figure 4", key, a3.DISPLAY.get(g, g), f"{title}: cells", int(n),
                source=f"dct_labels_DCT_{'ctrl' if cond == 'Ctrl' else 'zerok'}_labelled.h5ad ({LABELS[cond]}; X_umap)")
        imgs[key] = both(fig, f"panel {key}")
        pdfs[key], _ = run.figure(fig, f"19_fig4_panel_{key}", f"Figure 4{key}: {title} DCT cells on their UMAP by "
                                  "cluster annotation", formats=(".pdf", ".png"), tight=False)
    # c: markers in both diets
    vals, order = dot_values(run, checks)
    cv, pitch = dot_panel(vals, order)
    tc = t6d.text_checks(cv)
    bad = banned(cv.fig)
    checks.add("panel c: format (texts >= 7 pt, Arial, no overlap, none outside); width; reader-facing text",
               f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; {tc['fonts']}; overlaps {tc['n_text_overlaps']}; "
               f"outside {tc['n_text_outside_figure']}; {cv.W:.1f} x {cv.H:.1f} mm; pitch {pitch:.2f} mm; "
               f"internal names {bad or 'none'}", tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0
               and tc["n_text_outside_figure"] == 0 and abs(cv.W - W_C) < 0.05 and not bad)
    for r in vals.itertuples():
        c, g = r.group.split("|")
        val("Figure 4", "c", ROW_NAME[(c, g)], f"{r.marker}: colour (mean z scaled 0-1 over the five rows); low = "
            "fraction detected; high = mean z", float(r.mean_z_scaled_0_1), float(r.frac_detected), float(r.mean_z),
            "dct_labels_DCT_joint_labelled.h5ad (dotplot rule)")
    top = vals.loc[vals.groupby("marker")["mean_z_scaled_0_1"].idxmax()].set_index("marker")["group"]
    checks.add("panel c: row with the highest colour per DCT2 marker", "; ".join(
        f"{m} {ROW_NAME[tuple(top[m].split('|'))]}" for m in ("Calb1", "Slc8a1", "S100g", "Klk1", "Trpv5")), None)
    h_c = cv.H
    imgs["c"] = both(cv.fig, "panel c")
    pdfs["c"], _ = run.figure(cv.fig, "19_fig4_panel_c", "Figure 4c: DCT marker proteins in the annotated groups of "
                              "both diets, one colour scale", formats=(".pdf", ".png"), tight=False)
    # d: DCT2 cells per seed
    s = seed_values(run, checks)
    sheet = seed_panel(s, h_c)
    tc = t8.text_checks(sheet.fig)
    bad = banned(sheet.fig)
    checks.add("panel d: format (texts >= 7 pt, no overlap, none outside); reader-facing text",
               f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; overlaps {tc['n_text_overlaps']} {tc['overlaps'][:80]}; "
               f"outside {tc['n_text_outside_figure']}; internal names {bad or 'none'}",
               tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0 and not bad)
    imgs["d"] = both(sheet.fig, "panel d")
    pdfs["d"], _ = run.figure(sheet.fig, "19_fig4_panel_d", "Figure 4d: cells annotated DCT2 at each Leiden seed of "
                              "the stored scan, control and zero-K", formats=(".pdf", ".png"), tight=False)
    # e, f: the current panels, drawn by the response to the reviewers's functions
    fig_e = f3.draw_a(run, checks)
    imgs["e"] = both(fig_e, "panel e")
    plt.close(fig_e)
    old_e, _ = R.read_image(run.use_input(OLD_E.with_suffix(".png"), "current panel a"))
    checks.add("panel e: equals the current panel a pixel for pixel (600 dpi)",
               f"{int((imgs['e'][R.DPI_PNG] != old_e).any(axis=2).sum()) if imgs['e'][R.DPI_PNG].shape == old_e.shape else -1} "
               "pixels differ", imgs["e"][R.DPI_PNG].shape == old_e.shape and bool((imgs["e"][R.DPI_PNG] == old_e).all()), "0")
    cv_f, _ = f3.draw_b(run, checks)
    R.edit_texts(cv_f.fig, R.close_up)
    imgs["f"] = both(cv_f.fig, "panel f")
    plt.close(cv_f.fig)
    old_f, _ = R.read_image(run.use_input(OLD_F.with_suffix(".png"), "current panel b"))
    checks.add("panel f: equals the current panel b pixel for pixel (600 dpi)",
               f"{int((imgs['f'][R.DPI_PNG] != old_f).any(axis=2).sum()) if imgs['f'][R.DPI_PNG].shape == old_f.shape else -1} "
               "pixels differ", imgs["f"][R.DPI_PNG].shape == old_f.shape and bool((imgs["f"][R.DPI_PNG] == old_f).all()), "0")
    pdfs["e"] = run.use_input(OLD_E.with_suffix(".pdf"), "current panel a")
    pdfs["f"] = run.use_input(OLD_F.with_suffix(".pdf"), "current panel b")

    # --- compose ----------------------------------------------------------------------------------------------------
    rows = (("a", "b"), ("c", "d"), ("e", "f"))
    layout_rows, pages = [], {}
    with lib_letters(LETTER_PT):
        for dpi in (R.DPI_PNG, R.DPI_TIF):
            w = {k: v[dpi].shape[1] for k, v in imgs.items()}
            h = {k: v[dpi].shape[0] for k, v in imgs.items()}
            width, strip, gap = R.px(R.PAGE_MM, dpi), R.px(STRIP_MM, dpi), R.px(ROW_GAP_MM, dpi)
            pos, cells, y = {}, [], 0
            for left, right in rows:
                pos[left], pos[right] = (0, y + strip), (width - w[right], y + strip)
                cells += [(left, (0, y)), (right, (width - w[right], y))]
                y += strip + max(h[left], h[right]) + gap
            height = y - gap
            panels = {k: v[dpi] for k, v in imgs.items()}
            page = R.compose(panels, pos, width, height)
            inside, overlaps = R.layout_problems(pos, w, h, width, height)
            blank = R.letters_blank(page, cells, dpi)
            checks.add(f"{dpi} dpi: panels inside the page; overlapping panels; page blank where the letters go",
                       f"{inside}; {overlaps or 'none'}; {'all blank' if all(blank.values()) else blank}",
                       inside and not overlaps and all(blank.values()))
            lettered, _ = R.draw_letters(page, cells, dpi)
            same = all(bool((lettered[yy:yy + h[k], xx:xx + w[k]] == panels[k]).all()) for k, (xx, yy) in pos.items())
            checks.add(f"{dpi} dpi: every panel placed pixel for pixel (no scaling)", str(same), same)
            pages[dpi] = (lettered, pos, cells, (width, height))
            for k, (xx, yy) in pos.items():
                layout_rows.append({"image": f"{dpi} dpi", "panel": k, "x_mm": round(R.mm(xx, dpi), 2),
                                    "y_mm": round(R.mm(yy, dpi), 2), "width_mm": round(R.mm(w[k], dpi), 2),
                                    "height_mm": round(R.mm(h[k], dpi), 2), "scaling": "none (1:1)"})
        checks.add("letters in reading order", R.reading_order(pages[R.DPI_PNG][2]),
                   R.reading_order(pages[R.DPI_PNG][2]) == "abcdef", "abcdef")
        desc = ("Figure 4 extended: (a, b) control and zero-K DCT cells on their UMAPs by cluster annotation; (c) marker "
                "proteins in the annotated groups of both diets; (d) cells annotated DCT2 per Leiden seed; (e) control "
                "cells with the ProLIF cells; (f) marker proteins above their transcripts in the Mouse Kidney Atlas")
        png, tif, pdf = (run.out("figures", name, ext) for ext in (".png", ".tif", ".pdf"))
        R.save_png(png, pages[R.DPI_PNG][0], desc)
        R.save_tif(tif, pages[R.DPI_TIF][0], desc)
        _, pos600, cells600, (w600, h600) = pages[R.DPI_PNG]
        to_pt = 72.0 / R.DPI_PNG
        merged = R.merge_pdf(pdf, [(pdfs[k], x * to_pt, y * to_pt) for k, (x, y) in pos600.items()], w600 * to_pt,
                             h600 * to_pt, [(k, (x * to_pt, y * to_pt)) for k, (x, y) in cells600], "Figure 4")
    checks.add("PDF: letters written as text; merging added no raster image", f"{merged['letters_text']}; "
               f"{merged['n_images_page']} on the page, {merged['n_images_panels']} in the panels",
               merged["letters_text"] == "abcdef" and merged["n_images_page"] == merged["n_images_panels"])
    run.table(pd.DataFrame(layout_rows), f"{STEP}_layout")
    return {"name": name, "size_mm": (round(R.mm(w600, R.DPI_PNG), 1), round(R.mm(h600, R.DPI_PNG), 1)), "png": png,
            "pdf": pdf}


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Figures 4 and 5 extended; S-19-d second version.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_joint")
    lib30.check_inputs(run, checks)
    rd = lambda n, **kw: lib30.read_table(run, n, **kw)  # noqa: E731
    panel = rd("panel_results")
    T = {"panel": panel[panel["status"] == "tested"], "spec_null": rd("specificity_matched_null"),
         "spec_draws": rd("specificity_null_draws_summary"), "bulk_matched": rd("bulk_matched"),
         "bulk_conc": rd("bulk_concordance"), "bulk_null": rd("bulk_plate_null"), "bulk_signs": rd("bulk_sign_tables")}
    out = [figure5(run, T, checks), s19d_v2(run, T, checks), figure4(run, checks)]
    run.table(pd.DataFrame(VALUES), f"{STEP}_values")
    run.table(checks.frame(), f"{STEP}_checks")
    for o in out:
        print(f"{o['name']}: {o['size_mm'][0]} x {o['size_mm'][1]} mm")
    n_fail = int((~checks.frame()["pass"]).sum())
    print(f"Checks: {len(checks.rows) - n_fail} passed or information, {n_fail} failed")
    run.finish({"figures": {o["name"]: list(o["size_mm"]) for o in out}, "letter_pt": LETTER_PT,
                "letter_font": "Arial Bold, lower case (supplementary figure: capitals, 8.5 pt as the response to the reviewers)",
                "png_dpi": R.DPI_PNG, "tif_dpi": R.DPI_TIF, "figure4_strip_mm": STRIP_MM, "figure4_row_gap_mm": ROW_GAP_MM,
                "new_statistics": "none; new descriptive group means for Figure 4c (dotplot rule, joint matrix)",
                "installed_in_revision": False, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        failed = [r["check"] for r in checks.rows if not r["pass"]]
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv: {failed}")


if __name__ == "__main__":
    main()
