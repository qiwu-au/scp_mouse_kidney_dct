#!/usr/bin/env python
"""kidney_figure2 step "fig2b_full": the revised Figure 2B marker dotplot with all 45 quantified markers.

Task kidney_figure2; request from the response to the reviewers (D8, 10 Nov 2026).

The authors chose all 45 quantified markers for the main Figure 2B and asked for
the nuclei-only version too ("Go, with nuclei", 10 Nov 2026):
- 2B: the kidney_rna "full" paired dotplot (``dotplot_t15_full_all``): the 45 quantified V3 + endothelial-row markers of the
  eight kidney_annotation types in the kidney_annotation a_endo order, protein above RNA (the kidney_rna stacked layout; 2 x 45 columns do not fit side
  by side at 180 mm). Redrawn from the stored kidney_annotation and kidney_rna value tables (no atlas reload) in the style of
  ``kidney_figure2_fig2ab.py``: display names, the panel title "Protein (SCP-MS)", no in-figure note, grey RNA rows and
  columns, Arial >= 7 pt, no text overlap.
- Supplementary Figure S5: the same panel with the RNA half from the atlas nuclei only (``dotplot_t15_full_nuclei``).
The 19-marker file of step fig2ab (``kidney_figure2_fig2b_marker_dotplot``) is kept; this step writes new files only.
PST row chip in tab10 brown #8c564b instead of the kidney_annotation palette grey #7f7f7f (authors, 10 Nov 2026; kidney_figure2_lib.palette).
Also tabulates, per marker, the row with the highest plotted colour (Unknown row included) against the marker's own
cell type.

Inputs (read-only): kidney_object kidney object (protein values re-check) and labels with the kidney_annotation columns
                    (kidney_rna_lib.load_kidney); kidney_annotation dotplot values and marker order (a_endo); kidney_rna RNA dotplot values
                    and marker coverage.
Outputs: results/figures/kidney_figure2/kidney_figure2_fig2b_marker_dotplot_full, kidney_figure2_fig2b_marker_dotplot_full_nuclei
         (.pdf/.png);
         results/tables/kidney_figure2/kidney_figure2_fig2b_full_dotplot_values.csv, kidney_figure2_fig2b_full_own_row.csv,
         kidney_figure2_fig2b_full_figure_checks.csv, kidney_figure2_fig2b_full_checks.csv,
         kidney_figure2_step_fig2b_full_manifest.json

Command (project root):  python scripts/kidney_figure2_fig2b_full.py
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
import dotplot_lib as t6
import dotplot_dotplots as t6dot          # Panel, Canvas, legends, text / file checks
import kidney_rna_dotplots as t7dot          # GreyPanel, WIDTH, P_MAX, RNA_SOURCES (as the kidney_rna paired dotplot)
import kidney_rna_lib as t7
import kidney_annotation_chosen_labels as t15c      # kidney_annotation row order and palette
import kidney_figure2_lib as T

STEP = "fig2b_full"
COL = T.PRIMARY
VERSION = "full"
SOURCES = {"all": "fig2b_marker_dotplot_full", "nuclei": "fig2b_marker_dotplot_full_nuclei"}
PROT_TITLE = "Protein (SCP-MS)"
N_MARKERS = 45
VALUE_TOL = 1e-9


def stacked(panels, vgap: float = 3.0):
    """``kidney_rna_dotplots.build_paired`` (layout "stacked") without the in-figure note."""
    m = t6dot.MARGIN
    cw, ch = t6dot.color_legend_dims()
    ncol = len(panels[0].markers)
    lw = max(p.row_label_w() for p in panels)
    leg_w = max(t6dot.size_legend_dims(t7dot.P_MAX)[0], cw)
    pitch = min(t7dot.P_MAX, (t7dot.WIDTH - 2 * m - lw - 5.0 - leg_w) / ncol)
    sw, sh, _ = t6dot.size_legend_dims(pitch)
    leg_w, leg_h = max(sw, cw), sh + 3.0 + ch
    sizes = [p.size(pitch) for p in panels]
    body_h = sum(h for _, h in sizes) + vgap * (len(panels) - 1)
    top_offset = panels[0].title_h() + panels[0].bracket_h(pitch)
    content_w = lw + ncol * pitch + 5.0 + leg_w
    if content_w > t7dot.WIDTH - 2 * m + 1e-6:
        raise ValueError(f"content {content_w:.1f} mm wider than {t7dot.WIDTH} mm")
    cv = t6dot.Canvas(t7dot.WIDTH, 2 * m + max(body_h, top_offset + leg_h))
    x0 = (t7dot.WIDTH - content_w) / 2
    y = m
    geo = []
    for p, (_, h) in zip(panels, sizes):
        geo.append(p.draw(cv, x0 + lw - p.row_label_w(), y, pitch))
        y += h + vgap
    t6dot.draw_legends(cv, geo[0]["grid_right"] + 5.0, geo[0]["grid_top"], pitch, stacked=True)
    return cv, pitch


def top_row(color: np.ndarray, rows: list[str], skip: list[bool] | None = None) -> tuple[str, int]:
    """Row with the highest plotted colour in one column (grey rows skipped) and the number of rows tied at it."""
    c = np.asarray(color, dtype=float).copy()
    if skip is not None:
        c[np.asarray(skip)] = np.nan
    if np.all(np.isnan(c)):
        return "", 0
    return rows[int(np.nanargmax(c))], int(np.sum(c == np.nanmax(c)))


def main() -> None:
    parser = argparse.ArgumentParser(description="kidney_figure2: Figure 2B with all 45 quantified markers (stacked).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(T.TASK, STEP, args)
    checks = t5.Checks(T.DATASET)
    np.random.seed(T.SEED)
    import matplotlib
    matplotlib.use("Agg")

    adata, labels = T.load(run, checks)
    lab = labels[COL].astype(str).to_numpy()
    types, rows = t15c.t15_rows(np.unique(lab))
    colors = T.palette(adata, rows, checks)      # kidney_annotation palette, PST #8c564b (authors, 10 Nov 2026)

    vals = pd.read_csv(run.use_input(t7.KIDNEY_ANNOTATION_DOTPLOT_VALUES, "kidney_annotation a_endo dotplot values (protein half)"))
    order_all = pd.read_csv(run.use_input(t7.KIDNEY_ANNOTATION_MARKER_ORDER, "kidney_annotation a_endo dotplot marker order"))
    rv = pd.read_csv(run.use_input(cc.out_path(t7.TASK, "tables", "dotplot_rna_values", ".csv"),
                                   "kidney_rna RNA dotplot values (RNA half)"))
    cov = pd.read_csv(run.use_input(cc.out_path(t7.TASK, "tables", "dotplot_marker_coverage", ".csv"),
                                    "kidney_rna dotplot marker coverage (markers of dotplot_t15_full_*)"))
    order = t6dot.plotted_order(order_all, VERSION)
    markers = order["marker"].tolist()
    repeat = order["repeat"].fillna(False).astype(bool)
    rows_v = vals.drop_duplicates("group").sort_values("group_position")["group"].tolist()
    checks.add("dotplot rows = kidney_annotation row order (Unknown last)", "; ".join(rows_v),
               rows_v == rows and len(types) == 8 and rows[-1] == T.UNKNOWN)
    t07_full = sorted(cov.loc[cov["plotted_t15_full"].astype(bool), "marker"])
    checks.add("dotplot columns = the markers of kidney_rna dotplot_t15_full_all (kidney_annotation a_endo order, none repeated)",
               f"{len(markers)} markers; {int(repeat.sum())} repeated",
               len(markers) == N_MARKERS and sorted(markers) == t07_full and not repeat.any())

    # protein values re-checked against the kidney_object object (as kidney_rna)
    recomputed = t6.group_values(adata, labels[COL], rows, sorted(set(markers)))
    merged = vals.merge(recomputed, on=["group", "marker"], suffixes=("_stored", "_recomputed"))
    merged = merged[merged["marker"].isin(markers)]
    d = max(float(np.max(np.abs(merged["frac_detected_stored"] - merged["frac_detected_recomputed"]))),
            float(np.max(np.abs(merged["mean_z_stored"] - merged["mean_z_recomputed"]))))
    n_exp = len(rows) * len(set(markers))
    checks.add("protein dotplot values reproduce from the kidney_object object (rows x markers; max abs diff)",
               f"{len(merged)} of {n_exp}; {d:.1e}", len(merged) == n_exp and d < VALUE_TOL, f"< {VALUE_TOL:g}")
    if not checks.ok:
        run.table(checks.frame(), "fig2b_full_checks")
        raise SystemExit("A check failed; see kidney_figure2_fig2b_full_checks.csv.")

    # display names (DCT1 -> DCT, PC -> CNT/PC) in rows, values and column brackets
    disp = [T.display(r) for r in rows]
    own = order["group"].map(T.display).tolist()
    vc = labels[COL].value_counts()
    lo = pd.DataFrame({"label_column": COL, "label": disp, "position": range(1, len(rows) + 1), "plotted": True,
                       "n_cells": [int(vc[r]) for r in rows], "colour": [colors[r] for r in rows]})
    prot = t6dot.make_panel(PROT_TITLE, vals.assign(group=vals["group"].map(T.display)), lo, COL,
                            order.assign(group=order["group"].map(T.display)))

    t6dot.plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    out, fig_rows, rna_top = [], [], {}
    for src, name in SOURCES.items():
        sub = rv[(rv["label_column"] == COL) & (rv["rna_source"] == src)].copy()
        sub["group"] = sub["group"].map(T.display)
        frac = sub.pivot(index="group", columns="marker", values="frac_detected").reindex(index=disp, columns=markers)
        colr = sub.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(index=disp, columns=markers)
        nrow = sub.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(disp)
        in_atlas = sub.drop_duplicates("marker").set_index("marker")["gene_in_atlas"].astype(bool)
        grey_rows = [int(n) == 0 for n in nrow]
        grey_cols = [not bool(in_atlas.get(m, False)) for m in markers]
        rna = t7dot.GreyPanel(f"RNA (Mouse Kidney Atlas, {t7dot.RNA_SOURCES[src][1]})", disp,
                              [int(n) for n in nrow], prot.row_colors, markers, prot.groups, frac.to_numpy(float),
                              colr.to_numpy(float), prot.repeat, grey_rows=grey_rows, grey_cols=grey_cols,
                              n_text=[f"{int(n):,}" if n > 0 else "–" for n in nrow])
        checks.add(f"RNA {src}: atlas cells per row; grey rows (no atlas counterpart)",
                   "; ".join(f"{r} {int(n):,}" for r, n in zip(disp, nrow)) + " | grey: "
                   + (", ".join(r for r, g in zip(disp, grey_rows) if g) or "none"), None)
        checks.add(f"RNA {src}: grey columns (gene not in the atlas)",
                   ", ".join(m for m, g in zip(markers, grey_cols) if g) or "none", None)

        cv, pitch = stacked([prot, rna])
        tc = t6dot.text_checks(cv)
        paths = run.figure(cv.fig, name, f"Figure 2B{'' if src == 'all' else ' / Supplementary Figure S-05-e'}: "
                                         f"paired protein / RNA marker dotplot, all {N_MARKERS} quantified markers, "
                                         f"stacked, RNA {src}, kidney_t15_label, display names, no note", tight=False)
        fc = t6dot.file_checks(paths, cv.W, cv.H)
        base = T.figure_checks(cv.fig, name, paths, prot.row_labels(), max_w_mm=t7dot.WIDTH)
        arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
        ok = bool(base["pass"] and tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0
                  and tc["n_text_outside_figure"] == 0 and fc["width_ok"])
        fig_rows.append({**base, "rna_source": src, "layout": "stacked", "n_columns": len(markers),
                         "min_font_rule_pt": 7, "pitch_mm": round(pitch, 2), "n_text_overlaps": tc["n_text_overlaps"],
                         "n_text_outside_figure": tc["n_text_outside_figure"], "fonts": tc["fonts"], "pass": ok})
        print(f"  {name}: {cv.W:.0f} x {cv.H:.1f} mm, pitch {pitch:.2f} mm, overlaps {tc['n_text_overlaps']}, "
              f"outside {tc['n_text_outside_figure']}, pass {ok}", flush=True)
        t6dot.plt.close(cv.fig)

        for panel, f, c, n in (("protein", prot.frac, prot.color, prot.row_n), ("RNA", rna.frac, rna.color, list(nrow))):
            for i, r in enumerate(disp):
                for j, mk in enumerate(markers):
                    out.append({"figure": f"kidney_figure2_{name}", "panel": panel, "rna_source": src, "row": r,
                                "n_cells_row": int(n[i]), "marker": mk, "bracket": own[j], "frac_detected": f[i, j],
                                "colour_scaled_0_1": c[i, j],
                                "grey": panel == "RNA" and (grey_rows[i] or grey_cols[j])})
        grey_row_of = dict(zip(disp, grey_rows))
        rna_top[src] = [("", 0) if grey_cols[j] or grey_row_of[own[j]] else top_row(rna.color[:, j], disp, grey_rows)
                        for j in range(len(markers))]   # no RNA comparison where the column or the own row is grey
    values = pd.DataFrame(out)
    values = values[~((values["panel"] == "protein") & (values["rna_source"] != "all"))]   # protein half once

    # highest plotted colour per marker: own row or not (protein; Unknown row included). RNA as information.
    t15_flag = order.set_index("marker")["assigned_is_top_of_all_rows"].astype(bool)
    rec = []
    for j, mk in enumerate(markers):
        p_top, p_tie = top_row(prot.color[:, j], disp)
        rec.append({"position": j + 1, "marker": mk, "own_type": own[j], "placement": order["placement"].iloc[j],
                    "protein_top_row": p_top, "protein_rows_tied_at_top": p_tie, "protein_own_row_top": p_top == own[j],
                    "t15_assigned_is_top_of_all_rows": bool(t15_flag[mk]),
                    "t15_top_row_all_types": T.display(order["top_row_all_types"].iloc[j]),
                    **{f"rna_{s}_top_row": rna_top[s][j][0] or "grey column or own row" for s in SOURCES},
                    **{f"rna_{s}_own_row_top": (rna_top[s][j][0] == own[j]) if rna_top[s][j][0] else np.nan
                       for s in SOURCES}})
    own_tab = pd.DataFrame(rec)
    n_own = int(own_tab["protein_own_row_top"].sum())
    not_own = own_tab[~own_tab["protein_own_row_top"]]
    checks.add("protein: markers whose highest plotted colour is in their own row (Unknown row included); others "
               "(row with the highest colour)",
               f"{n_own} of {len(markers)}; " + ", ".join(f"{r.marker} ({r.protein_top_row})" for r in not_own.itertuples()),
               bool((own_tab["protein_own_row_top"] == own_tab["t15_assigned_is_top_of_all_rows"]).all()
                    and (own_tab["protein_rows_tied_at_top"] == 1).all()),
               "same markers as kidney_annotation assigned_is_top_of_all_rows; no ties")
    for s in SOURCES:
        col_ = own_tab[f"rna_{s}_own_row_top"]
        checks.add(f"RNA {s} (information): markers whose highest colour among non-grey rows is in their own row",
                   f"{int((col_ == True).sum())} of {int(col_.notna().sum())} markers with an atlas gene and an "
                   f"atlas row for the own type", None)

    run.table(values, "fig2b_full_dotplot_values")
    run.table(own_tab, "fig2b_full_own_row")
    figs = pd.DataFrame(fig_rows)
    run.table(figs, "fig2b_full_figure_checks")
    checks.add("figures passing the format checks (180 mm, Arial >= 7 pt, no overlap, no text outside)",
               f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(checks.frame(), "fig2b_full_checks")
    run.finish({"display": T.DISPLAY, "version": VERSION, "n_markers": len(markers), "layout": "stacked (kidney_rna full)",
                "rna_sources": list(SOURCES),
                "marker_set": "all 45 quantified markers (authors, the response to the reviewers Nov 2026: \"Use all 45\")",
                "nuclei_version": "rendered for Supplementary Figure S-05-e (authors, 10 Nov 2026: \"Go, with nuclei\")",
                "kept": "kidney_figure2_fig2b_marker_dotplot (19 markers, step fig2ab) unchanged",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see kidney_figure2_fig2b_full_checks.csv.")


if __name__ == "__main__":
    main()
