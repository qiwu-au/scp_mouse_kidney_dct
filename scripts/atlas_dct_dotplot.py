#!/usr/bin/env python
"""atlas_dct step 5: paired DCT marker dotplot, RNA atlas groups next to the SCP V1 labels (dotplot order).

Analysis plan, step 5.

Panels (same columns, stacked; dotplot / kidney_rna building blocks and encoding)
---------------------------------------------------------------------
- Protein, V1 per-cell labels and Protein, V1 cluster labels: the published dotplot DCT dotplot values (re-checked
  against the kidney_object object): dot = fraction of the label's cells with intensity > 0; colour = group mean of the
  per-protein z-scored ln(1 + intensity), rescaled 0-1 per protein across the three rows. Unknown (n = 4) not shown
  (as dotplot).
- RNA: atlas groups (published DCT, DCT-CNT, CNT, CTAL, MTAL; or the atlas_dct-derived DCT1 (atlas_dct), DCT2 (atlas_dct) with
  DCT-CNT, CNT, CTAL, MTAL): dot = fraction with counts > 0; colour = mean log-normalized expression (10,000 per cell,
  log1p), rescaled 0-1 per gene across the RNA rows.
- Columns: dotplot 'panel' order (Core | DCT1 | DCT2 | TAL | Cycling; 11 markers; main) and dotplot 'workbook' order (V1
  workbook markers placed by dotplot; supplement).
- RNA subsets: all atlas cells (primary); nuclei only and cells only (published groups, panel order).

Inputs (read-only): dotplot DCT marker order, dotplot values and label order; kidney_object canonical DCT object and labels;
                    atlas_dct region subset and split table.
Outputs: results/figures/atlas_dct/atlas_dct_dotplot_{panel,workbook}_{published,derived}_{all,nuclei,cells}
         (.pdf/.png/.tif); results/tables/atlas_dct/atlas_dct_dotplot_rna_values.csv, atlas_dct_dotplot_checks.csv,
         atlas_dct_dotplot_figure_checks.csv

Command (project root):  python scripts/atlas_dct_dotplot.py
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
import dotplot_dotplots as t6dot
import kidney_rna_dotplots as d7            # GreyPanel, build_paired (180 mm, stacked / side by side)
import kidney_rna_lib as L7
import atlas_dct_lib as T

VALUE_TOL = 1e-9
FIGURES = [("panel", False, "all"), ("panel", False, "nuclei"), ("panel", False, "cells"), ("panel", True, "all"),
           ("workbook", False, "all"), ("workbook", True, "all")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atlas_dct step 5: paired RNA / protein DCT marker dotplot.")
    cc.add_common_args(parser)
    return parser.parse_args()


def rna_values(region, ln, markers: list[str], gene_of: dict[str, str]) -> pd.DataFrame:
    out = []
    genes = [gene_of[m] for m in markers if gene_of[m]]
    pos = region.var_names.get_indexer(genes)
    sub_ln, sub_c = ln[:, pos], region.X[:, pos]
    for derived in (False, True):
        for src in T.RNA_SOURCES:
            st = T.group_stats(sub_ln, sub_c, region.obs[T.group_column(derived)], T.rna_groups(derived),
                               pd.Index(genes), T.source_mask(region.obs, src))
            for gpos, g in enumerate(T.rna_groups(derived), start=1):
                n = int(st["n"][g])
                for m in markers:
                    ag = gene_of[m]
                    has = bool(ag) and n > 0
                    out.append({"level": T.LEVEL_SHORT[derived], "rna_source": src, "group": g, "group_position": gpos,
                                "n_atlas_cells": n, "marker": m, "atlas_gene": ag, "gene_in_atlas": bool(ag),
                                "frac_detected": float(st["frac"].loc[g, ag]) if has else 0.0,
                                "mean_lognorm": float(st["mean"].loc[g, ag]) if has else np.nan})
    v = pd.DataFrame(out)
    grp = v.groupby(["level", "rna_source", "marker"])["mean_lognorm"]
    lo, hi = grp.transform("min"), grp.transform("max")
    v["mean_scaled_0_1"] = ((v["mean_lognorm"] - lo) / (hi - lo).where(hi > lo)).fillna(0.0)
    return v


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "dotplot", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, _, _ = T.load_scp(run, checks)
    order = pd.read_csv(run.use_input(T.DOTPLOT_DIR / "dotplot_marker_order_dct.csv", "dotplot DCT marker order"))
    vals = pd.read_csv(run.use_input(T.DOTPLOT_DIR / "dotplot_dct_dotplot_values.csv", "dotplot DCT dotplot values"))
    lo = pd.read_csv(run.use_input(T.DOTPLOT_DIR / "dotplot_label_order.csv", "dotplot label order"))
    region = T.load_region(run, with_split=True)

    # protein values: re-check against the canonical object
    markers_all = sorted(set(vals["marker"]))
    for col in T.LABEL_COLUMNS:
        rows = lo[(lo["label_column"] == col) & lo["plotted"]].sort_values("position")["label"].tolist()
        rec = t6.group_values(adata, labels[col], rows, markers_all)
        mg = vals[vals["label_column"] == col].merge(rec, on=["group", "marker"], suffixes=("_stored", "_new"))
        d_f = float(np.max(np.abs(mg["frac_detected_stored"] - mg["frac_detected_new"])))
        d_z = float(np.max(np.abs(mg["mean_z_stored"] - mg["mean_z_new"])))
        checks.add(f"{col}: dotplot DCT dotplot values reproduce from the kidney_object object (entries; max abs difference of "
                   "detection, mean z)", f"{len(mg)} of {len(rows) * len(markers_all)}; {d_f:.1e}; {d_z:.1e}",
                   len(mg) == len(rows) * len(markers_all) and max(d_f, d_z) < VALUE_TOL, f"< {VALUE_TOL:g}")
    look = L7.atlas_gene_lookup(region.var_names)
    plotted = {v: t6dot.plotted_order(order, v) for v in ("panel", "workbook")}
    markers = sorted(set().union(*[set(o["marker"]) for o in plotted.values()]))
    gene_of = {m: look.get(m.lower(), "") for m in markers}
    absent = [m for m in markers if not gene_of[m]]
    checks.add("plotted markers absent from the atlas genes (grey RNA columns)", ", ".join(absent) or "none", None)
    if not checks.ok:
        run.table(checks.frame(), "dotplot_checks")
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("A check failed; see atlas_dct_dotplot_checks.csv.")
    ln = T.lognorm(region.X)
    rv = rna_values(region, ln, markers, gene_of)
    for (lvl, src), s in rv.groupby(["level", "rna_source"]):
        d = s.drop_duplicates("group")
        checks.add(f"RNA {lvl} / {src}: atlas cells per row", "; ".join(f"{r.group} {r.n_atlas_cells}"
                                                                       for r in d.itertuples()), None)

    t6dot.plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    fig_rows = []
    for version, derived, src in FIGURES:
        o = plotted[version]
        mk = o["marker"].tolist()
        prots = [t6dot.make_panel(f"Protein (SCP-MS), {T.LABEL_DISPLAY[col]}", vals, lo, col, o)
                 for col in T.LABEL_COLUMNS]
        lvl = T.LEVEL_SHORT[derived]
        s = rv[(rv["level"] == lvl) & (rv["rna_source"] == src)]
        rows = list(T.rna_groups(derived))
        frac = s.pivot(index="group", columns="marker", values="frac_detected").reindex(index=rows, columns=mk)
        colr = s.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(index=rows, columns=mk)
        nrow = s.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(rows)
        rna = d7.GreyPanel(f"RNA (Mouse Kidney Atlas, {T.SOURCE_TEXT[src].replace('atlas ', '')})", rows,
                           [int(n) for n in nrow], [T.RNA_PALETTE[r] for r in rows], mk, prots[0].groups,
                           frac.to_numpy(float), colr.to_numpy(float), [False] * len(mk),
                           grey_rows=[n == 0 for n in nrow], grey_cols=[not gene_of[m] for m in mk],
                           n_text=[f"{int(n):,}" if n > 0 else "–" for n in nrow])
        vtxt = ("dotplot panel markers (Fig 3B replacement)" if version == "panel"
                else "DCT V1 workbook markers in dotplot placement")
        note = (f"{vtxt}; columns as the dotplot DCT dotplot. Protein: dot = fraction of the label's cells with intensity "
                "> 0; colour = mean z-scored ln(1 + intensity), scaled 0–1 per protein across the three labels (Unknown, "
                "n = 4, not shown). RNA: atlas Celltype cells" + (" (DCT1 (atlas_dct) / DCT2 (atlas_dct): atlas_dct-derived split of the "
                "atlas DCT cells, not atlas labels)" if derived else "") + "; dot = fraction with counts > 0; colour = "
                "mean log-normalized expression scaled 0–1 per gene across the RNA rows. Row numbers = cells. RNA and "
                "protein come from different animals and studies.")
        cv, pitch = d7.build_paired([*prots, rna], "stacked", note)
        tc = t6dot.text_checks(cv)
        name = f"dotplot_{version}_{lvl}_{src}"
        paths = run.figure(cv.fig, name, f"paired RNA / protein DCT dotplot, {version}, {lvl}, RNA {src}",
                           formats=(".pdf", ".png", ".tif"), tight=False)
        fc = t6dot.file_checks(paths, cv.W, cv.H)
        arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
        ok = bool(tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
                  and fc["width_ok"] and fc.get("tif_mode") == "RGB" and fc.get("pdf_truetype_embedded")
                  and fc.get("pdf_arial"))
        fig_rows.append({"figure": name, "version": version, "level": lvl, "rna_source": src,
                         "primary": version == "panel" and src == "all", "width_mm": round(cv.W, 2),
                         "height_mm": round(cv.H, 2), "pitch_mm": round(pitch, 2), "n_columns": len(mk), **tc, **fc,
                         "pass": ok})
        print(f"  {name}: {cv.W:.0f} x {cv.H:.1f} mm, pitch {pitch:.2f}, overlaps {tc['n_text_overlaps']} "
              f"{tc['overlaps'][:100]!r}, outside {tc['n_text_outside_figure']}, pass {ok}", flush=True)
        t6dot.plt.close(cv.fig)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks (180 mm, Arial >= 7 pt, no overlap, PDF/PNG/TIFF)",
               f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(rv, "dotplot_rna_values")
    run.table(figs, "dotplot_figure_checks")
    run.table(checks.frame(), "dotplot_checks")
    run.finish({"layout": "stacked: protein V1 per-cell, protein V1 cluster, RNA (same columns)",
                "rna_colour": "mean log-normalized, min-max 0-1 per gene across RNA rows", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see atlas_dct_dotplot_checks.csv.")


if __name__ == "__main__":
    main()
