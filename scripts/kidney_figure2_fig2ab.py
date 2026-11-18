#!/usr/bin/env python
"""kidney_figure2 step "fig2ab": clean single-panel files for the revised Figure 2A and 2B.

Task kidney_figure2 (authors, 31 Oct 2026).

The submitted Figure 2B (re-annotation UMAP) is removed, so the revised Figure 2 is lettered A-F (authors, 31 Oct 2026):
- 2A: UMAP of the 712 retained kidney cells (stored ``X_umap`` of the kidney_object object, the submitted Figure 2A layout),
  coloured by ``kidney_t15_label`` (display names DCT, CNT/PC). Leiden cluster numbers are kept, small (link to the
  per-cluster evidence table S-05-a), and the one cluster labelled from shared markers only (cluster 21, PST) is drawn
  hollow (both authors, 31 Oct 2026). Unknown cells are small grey dots. Redrawn here because the kidney_annotation figure holds three
  panels, a title and a footer, and its numbers and key are below 5 pt at 88 mm.
- 2B (was 2C): the paired protein / RNA marker dotplot of kidney_rna (``dotplot_t15_main_all``), redrawn from the stored
  kidney_annotation and kidney_rna value tables (no atlas reload) with display names, the panel titles "Protein (SCP-MS)" and "RNA (Mouse
  Kidney Atlas, all cells and nuclei)" and no in-figure note (its content is in the legend). Layout, marker selection,
  colours and checks as kidney_rna (180 mm, Arial >= 7 pt, no text overlap).
No panel letters or figure titles: the authors assemble Figure 2 in PowerPoint (authors, 31 Oct 2026).
PST is drawn in tab10 brown #8c564b instead of the kidney_annotation palette grey #7f7f7f (authors, 10 Nov 2026; kidney_figure2_lib.palette),
in the 2A dots and key and the 2B row chip.

Inputs (read-only): kidney_object kidney object (X_umap; protein values re-check) and labels with the kidney_annotation columns
                    (kidney_rna_lib.load_kidney); kidney_annotation dotplot values and marker order (a_endo); kidney_rna RNA dotplot values
                    and marker coverage.
Outputs: results/figures/kidney_figure2/kidney_figure2_fig2a_umap, kidney_figure2_fig2b_marker_dotplot (.pdf/.png);
         results/tables/kidney_figure2/kidney_figure2_fig2a_umap_key.csv, kidney_figure2_fig2b_dotplot_values.csv,
         kidney_figure2_fig2ab_figure_checks.csv, kidney_figure2_fig2ab_checks.csv, kidney_figure2_step_fig2ab_manifest.json

Command (project root):  python scripts/kidney_figure2_fig2ab.py
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
import qc_mito_lib as t3
import cluster_de_lib as t5
import dotplot_lib as t6
import dotplot_dotplots as t6dot          # Panel, Canvas, legends, text / file checks
import kidney_rna_dotplots as t7dot          # GreyPanel, WIDTH, P_MAX (as the kidney_rna paired dotplot)
import kidney_rna_lib as t7
import kidney_annotation_chosen_labels as t15c      # kidney_annotation row order and palette
import kidney_figure2_lib as T

STEP = "fig2ab"
COL = T.PRIMARY
LOW = "low_shared_only"                      # kidney_annotation confidence of the shared-marker-only cluster (drawn hollow)
UMAP_W_MM, UMAP_H_MM = 86.0, 50.0
RNA_SOURCE = "all"
PROT_TITLE = "Protein (SCP-MS)"
RNA_TITLE = "RNA (Mouse Kidney Atlas, all cells and nuclei)"
VALUE_TOL = 1e-9


# ---------------------------------------------------------------------------
# Figure 2A
# ---------------------------------------------------------------------------
def fig2a(run, xy, lab, conf, clusters, rows, colors, n_cells, fig_rows) -> pd.DataFrame:
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    T.setup_style()
    ink = t3.INK
    fig = plt.figure(figsize=(UMAP_W_MM * T.MM, UMAP_H_MM * T.MM))
    ax = fig.add_axes((0.04, 0.06, 0.62, 0.92))
    for r in rows:
        m = lab == r
        if r == T.UNKNOWN:      # recedes behind the labelled cells, as in the kidney_annotation panel
            ax.scatter(xy[m, 0], xy[m, 1], s=1.6, color=T.UNKNOWN_COLOR, linewidths=0, rasterized=True, zorder=0)
            continue
        hollow = m & (conf == LOW)
        filled = m & ~hollow
        ax.scatter(xy[filled, 0], xy[filled, 1], s=3.5, color=colors[r], linewidths=0, rasterized=True, zorder=2)
        ax.scatter(xy[hollow, 0], xy[hollow, 1], s=4.5, facecolors="none", edgecolors=colors[r], linewidths=0.45,
                   rasterized=True, zorder=2)
    for c in t5.sort_clusters(clusters):
        centre = np.median(xy[clusters == c], axis=0)
        ax.text(centre[0], centre[1], c, fontsize=5.0, ha="center", va="center", color=ink["primary"], zorder=4,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("UMAP 1", fontsize=6, labelpad=1)
    ax.set_ylabel("UMAP 2", fontsize=6, labelpad=1)
    key = [f"{T.display(r)} ({n_cells[r]})" for r in rows]
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markeredgewidth=0,
                      markerfacecolor=(T.UNKNOWN_COLOR if r == T.UNKNOWN else colors[r]), label=k)
               for r, k in zip(rows, key)]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor="none",
                          markeredgecolor=ink["secondary"], markeredgewidth=0.6, label="shared markers only"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.67, 0.52), frameon=False, fontsize=6,
               handletextpad=0.3, labelspacing=0.5, borderaxespad=0)
    paths = run.figure(fig, "fig2a_umap", "Figure 2A: UMAP coloured by kidney_t15_label (cluster numbers; "
                                          "shared-marker cluster hollow)")
    fig_rows.append({**T.figure_checks(fig, "fig2a_umap", paths, key + ["shared markers only"]),
                     "min_font_rule_pt": T.MIN_FONT_PT})
    plt.close(fig)
    return pd.DataFrame({
        "label": rows, "display_label": [T.display(r) for r in rows], "n_cells": [n_cells[r] for r in rows],
        "colour": [T.UNKNOWN_COLOR if r == T.UNKNOWN else colors[r] for r in rows],
        "drawn_as": ["small grey dots" if r == T.UNKNOWN else
                     ("filled; hollow for the shared-marker-only cluster" if ((lab == r) & (conf == LOW)).any()
                      else "filled") for r in rows],
        "n_hollow": [int(((lab == r) & (conf == LOW)).sum()) for r in rows],
        "leiden_clusters": [", ".join(t5.sort_clusters(np.unique(clusters[lab == r]))) for r in rows]})


# ---------------------------------------------------------------------------
# Figure 2B (paired dotplot)
# ---------------------------------------------------------------------------
def paired_side(panels, gap: float = 8.0):
    """``kidney_rna_dotplots.build_paired`` (layout "side") without the in-figure note."""
    m = t6dot.MARGIN
    cw, ch = t6dot.color_legend_dims()
    pitch = t6dot.solve_pitch(panels, t7dot.WIDTH, "below", t7dot.P_MAX, gap)
    sizes = [p.size(pitch) for p in panels]
    sw, sh, _ = t6dot.size_legend_dims(pitch)
    leg_w, leg_h = sw + 6.0 + cw, max(sh, ch)
    content_w = max(sum(w for w, _ in sizes) + gap * (len(panels) - 1), leg_w)
    body_h = max(h for _, h in sizes)
    cv = t6dot.Canvas(t7dot.WIDTH, 2 * m + body_h + 2.5 + leg_h)
    x0 = x = (t7dot.WIDTH - content_w) / 2
    geo = []
    for p, (w, _) in zip(panels, sizes):
        geo.append(p.draw(cv, x, m, pitch))
        x += w + gap
    t6dot.draw_legends(cv, x0, max(g["bottom"] for g in geo) + 2.5, pitch, stacked=False)
    return cv, pitch


def fig2b(run, adata, labels, rows, colors, checks, fig_rows) -> pd.DataFrame:
    vals = pd.read_csv(run.use_input(t7.KIDNEY_ANNOTATION_DOTPLOT_VALUES, "kidney_annotation a_endo dotplot values (protein half)"))
    order_all = pd.read_csv(run.use_input(t7.KIDNEY_ANNOTATION_MARKER_ORDER, "kidney_annotation a_endo dotplot marker order"))
    rv = pd.read_csv(run.use_input(cc.out_path(t7.TASK, "tables", "dotplot_rna_values", ".csv"),
                                   "kidney_rna RNA dotplot values (RNA half)"))
    cov = pd.read_csv(run.use_input(cc.out_path(t7.TASK, "tables", "dotplot_marker_coverage", ".csv"),
                                    "kidney_rna dotplot marker coverage (markers of dotplot_t15_main_*)"))
    order = t6dot.plotted_order(order_all, "main")
    markers = order["marker"].tolist()
    rows_v = vals.drop_duplicates("group").sort_values("group_position")["group"].tolist()
    checks.add("dotplot rows = kidney_annotation row order", "; ".join(rows_v), rows_v == rows)
    t07_main = sorted(cov.loc[cov["plotted_t15_main"].astype(bool), "marker"])
    checks.add("dotplot columns = the markers of kidney_rna dotplot_t15_main_all", f"{len(markers)} markers",
               sorted(markers) == t07_main)

    # protein values re-checked against the kidney_object object (as kidney_rna)
    recomputed = t6.group_values(adata, labels[COL], rows, sorted(set(markers)))
    merged = vals.merge(recomputed, on=["group", "marker"], suffixes=("_stored", "_recomputed"))
    d = max(float(np.max(np.abs(merged["frac_detected_stored"] - merged["frac_detected_recomputed"]))),
            float(np.max(np.abs(merged["mean_z_stored"] - merged["mean_z_recomputed"]))))
    n_exp = len(rows) * len(set(markers))
    checks.add("protein dotplot values reproduce from the kidney_object object (rows x markers; max abs diff)",
               f"{len(merged)} of {n_exp}; {d:.1e}", len(merged) == n_exp and d < VALUE_TOL, f"< {VALUE_TOL:g}")

    # display names (DCT1 -> DCT, PC -> CNT/PC) in rows, values and column brackets
    disp = [T.display(r) for r in rows]
    vc = labels[COL].value_counts()
    lo = pd.DataFrame({"label_column": COL, "label": disp, "position": range(1, len(rows) + 1), "plotted": True,
                       "n_cells": [int(vc[r]) for r in rows], "colour": [colors[r] for r in rows]})
    prot = t6dot.make_panel(PROT_TITLE, vals.assign(group=vals["group"].map(T.display)), lo, COL,
                            order.assign(group=order["group"].map(T.display)))
    sub = rv[(rv["label_column"] == COL) & (rv["rna_source"] == RNA_SOURCE)].copy()
    sub["group"] = sub["group"].map(T.display)
    frac = sub.pivot(index="group", columns="marker", values="frac_detected").reindex(index=disp, columns=markers)
    colr = sub.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(index=disp, columns=markers)
    nrow = sub.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(disp)
    in_atlas = sub.drop_duplicates("marker").set_index("marker")["gene_in_atlas"].astype(bool)
    grey_rows = [int(n) == 0 for n in nrow]
    grey_cols = [not bool(in_atlas.get(m, False)) for m in markers]
    rna = t7dot.GreyPanel(RNA_TITLE, disp, [int(n) for n in nrow], prot.row_colors, markers, prot.groups,
                          frac.to_numpy(float), colr.to_numpy(float), prot.repeat, grey_rows=grey_rows,
                          grey_cols=grey_cols, n_text=[f"{int(n):,}" if n > 0 else "–" for n in nrow])
    checks.add("grey RNA rows (no atlas counterpart)", ", ".join(r for r, g in zip(disp, grey_rows) if g) or "none",
               None)
    checks.add("grey RNA columns (gene not in the atlas)", ", ".join(m for m, g in zip(markers, grey_cols) if g)
               or "none", None)

    t6dot.plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    cv, pitch = paired_side([prot, rna])
    tc = t6dot.text_checks(cv)
    paths = run.figure(cv.fig, "fig2b_marker_dotplot", "Figure 2B (was 2C): paired protein / RNA marker dotplot, "
                                                       "kidney_t15_label, display names, no note", tight=False)
    fc = t6dot.file_checks(paths, cv.W, cv.H)
    base = T.figure_checks(cv.fig, "fig2b_marker_dotplot", paths, prot.row_labels(), max_w_mm=t7dot.WIDTH)
    arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
    ok = bool(base["pass"] and tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0
              and tc["n_text_outside_figure"] == 0 and fc["width_ok"])
    fig_rows.append({**base, "min_font_rule_pt": 7, "pitch_mm": round(pitch, 2), "n_text_overlaps":
                     tc["n_text_overlaps"], "n_text_outside_figure": tc["n_text_outside_figure"], "fonts": tc["fonts"],
                     "pass": ok})
    t6dot.plt.close(cv.fig)

    out = []
    for panel, f, c, n in (("protein", prot.frac, prot.color, prot.row_n), ("RNA", rna.frac, rna.color, list(nrow))):
        for i, r in enumerate(disp):
            for j, mk in enumerate(markers):
                out.append({"panel": panel, "row": r, "n_cells_row": int(n[i]), "marker": mk,
                            "bracket": order["group"].map(T.display).iloc[j], "frac_detected": f[i, j],
                            "colour_scaled_0_1": c[i, j],
                            "grey": panel == "RNA" and (grey_rows[i] or grey_cols[j])})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="kidney_figure2: clean Figure 2A (UMAP) and 2B (paired dotplot) files.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(T.TASK, STEP, args)
    checks = t5.Checks(T.DATASET)
    np.random.seed(T.SEED)
    import matplotlib
    matplotlib.use("Agg")

    adata, labels = T.load(run, checks)
    lab = labels[COL].astype(str).to_numpy()
    conf = labels["kidney_t15_confidence"].astype(str).to_numpy()
    clusters = labels["t15_cluster"].astype(str).to_numpy()
    types, rows = t15c.t15_rows(np.unique(lab))
    colors = T.palette(adata, rows, checks)      # kidney_annotation palette, PST #8c564b (authors, 10 Nov 2026)
    n_cells = {r: int((lab == r).sum()) for r in rows}
    checks.add("rows in the revised Figure 2 order (Unknown last)", "; ".join(map(T.display, rows)),
               len(types) == 8 and rows[-1] == T.UNKNOWN)
    xy = np.asarray(adata.obsm["X_umap"])
    checks.add("stored UMAP of the kidney_object object (cells x 2)", f"{xy.shape}", xy.shape == (len(lab), 2))
    n_cl = len(np.unique(clusters))
    checks.add("Leiden clusters numbered on 2A", f"{n_cl}", n_cl == 23)
    low = conf == LOW
    low_cl, low_lab = sorted(set(clusters[low])), sorted(set(lab[low]))
    checks.add("hollow cells = the one shared-marker-only cluster", f"{int(low.sum())} cells; cluster "
               f"{', '.join(low_cl)}; {', '.join(low_lab)}", int(low.sum()) == 24 and low_cl == ["21"]
               and low_lab == ["PST"])

    fig_rows: list[dict] = []
    key = fig2a(run, xy, lab, conf, clusters, rows, colors, n_cells, fig_rows)
    run.table(key, "fig2a_umap_key")
    dot = fig2b(run, adata, labels, rows, colors, checks, fig_rows)
    run.table(dot, "fig2b_dotplot_values")
    figs = pd.DataFrame(fig_rows)
    run.table(figs, "fig2ab_figure_checks")
    checks.add("figures passing the format checks (2A <= 90 mm, >= 5 pt; 2B 180 mm, Arial >= 7 pt, no overlap)",
               f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(checks.frame(), "fig2ab_checks")
    run.finish({"display": T.DISPLAY, "cluster_numbers": "kept, small (authors, 31 Oct 2026)",
                "shared_marker_cluster": "hollow (authors, 31 Oct 2026)", "rna_source": RNA_SOURCE,
                "lettering": "submitted 2B removed; 2A UMAP, 2B paired dotplot (was 2C) (authors, 31 Oct 2026)",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see kidney_figure2_fig2ab_checks.csv.")


if __name__ == "__main__":
    main()
