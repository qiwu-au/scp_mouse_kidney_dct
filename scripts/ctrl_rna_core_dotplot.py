#!/usr/bin/env python
"""ctrl_rna_core step 4: paired DCT marker dotplot, RNA atlas groups next to the control SCP labels (labels_core order).

Analysis plan,
step 4 (port of atlas_dct_dotplot.py; ``rna_values`` imported unchanged; protein panels with
``labels_core_figures.panel_for``).

Panels (same columns, stacked; dotplot / kidney_rna building blocks and encoding, SISKA Fig 1c layout)
-------------------------------------------------------------------------------------------
- Protein, DE-rule labels (primary) and Protein, submitted cluster labels: the labels_core dotplot values (re-checked
  against the labels_core object): dot = fraction of the label's cells with intensity > 0; colour = group mean of the
  per-protein z-scored ln(1 + intensity), rescaled 0-1 per protein across the four rows DCT1, DCT2, ProLIF, Unknown
  (Unknown its own row, as labels_core).
- RNA: atlas groups (published DCT, DCT-CNT, CNT, CTAL, MTAL; or the atlas_dct-derived DCT1 (atlas_dct), DCT2 (atlas_dct) with
  DCT-CNT, CNT, CTAL, MTAL): dot = fraction with counts > 0; colour = mean log-normalized expression (10,000 per cell,
  log1p), rescaled 0-1 per gene across the RNA rows.
- Columns: labels_core 'panel' order (Core | DCT1 | DCT2 | TAL | Cycling; 12 markers incl. Col5a2; main) and labels_core 'workbook'
  order (DCT V1 workbook markers quantified in the control data, under the first workbook type listing them).
- RNA subsets: all atlas cells (primary); nuclei only and cells only (published groups, panel order).

Inputs (read-only): labels_core marker order and dotplot values; labels_core labelled control object and canonical labels;
                    atlas_dct region subset and split table.
Outputs: results/figures/ctrl_rna_core/ctrl_rna_core_dotplot_{panel,workbook}_{published,derived}_{all,nuclei,cells}
         (.pdf/.png/.tif); results/tables/ctrl_rna_core/ctrl_rna_core_dotplot_rna_values.csv, ctrl_rna_core_dotplot_checks.csv,
         ctrl_rna_core_dotplot_figure_checks.csv

Command (project root):  python scripts/ctrl_rna_core_dotplot.py
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
import kidney_rna_dotplots as d7            # GreyPanel, build_paired (180 mm, stacked)
import kidney_rna_lib as L7
import atlas_dct_lib as T
import ctrl_rna_core_lib as L
from atlas_dct_dotplot import FIGURES, VALUE_TOL, rna_values
from labels_core_figures import panel_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_rna_core step 4: paired RNA / control protein DCT marker dotplot.")
    cc.add_common_args(parser)
    return parser.parse_args()


def plotted(order: pd.DataFrame, version: str) -> pd.DataFrame:
    """labels_core marker order of one version, plotted markers only, in labels_core plotting order."""
    o = order[(order["version"] == version) & order["plotted"].astype(bool)]
    return o.sort_values("plotted_position").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "dotplot", args)
    L.ported_from(run, "atlas_dct_dotplot.py")
    checks = t5.Checks(L.DATASET)
    adata, labels, _, _ = L.load_scp(run, checks)
    order = L.read_table(run, "marker_order_dct", L.LABELS_CORE_TASK)
    vals = L.read_table(run, "dotplot_values", L.LABELS_CORE_TASK)
    region = T.load_region(run, with_split=True)

    # protein values: re-check the labels_core values against the labels_core labelled object
    markers_all = sorted(set(vals["marker"]))
    rows = list(L.SCP_GROUPS)
    for col in L.LABEL_COLUMNS:
        rec = t6.group_values(adata, labels[col], rows, markers_all)
        mg = vals[vals["label_column"] == col].merge(rec, on=["group", "marker"], suffixes=("_stored", "_new"))
        d_f = float(np.max(np.abs(mg["frac_detected_stored"] - mg["frac_detected_new"])))
        d_z = float(np.max(np.abs(mg["mean_z_stored"] - mg["mean_z_new"])))
        n_ok = bool((mg["n_cells_stored"] == mg["n_cells_new"]).all())
        checks.add(f"{col}: labels_core dotplot values reproduce from the labels_core object (entries; cells per row equal; max abs "
                   "difference of detection, mean z)", f"{len(mg)} of {len(rows) * len(markers_all)}; {n_ok}; "
                   f"{d_f:.1e}; {d_z:.1e}", len(mg) == len(rows) * len(markers_all) and n_ok and max(d_f, d_z) < VALUE_TOL,
                   f"< {VALUE_TOL:g}")
    look = L7.atlas_gene_lookup(region.var_names)
    orders = {v: plotted(order, v) for v in ("panel", "workbook")}
    markers = sorted(set().union(*[set(o["marker"]) for o in orders.values()]))
    gene_of = {m: look.get(m.lower(), "") for m in markers}
    absent = [m for m in markers if not gene_of[m]]
    checks.add("plotted markers absent from the atlas genes (grey RNA columns)", ", ".join(absent) or "none", None)
    unplotted = order.loc[~order["plotted"].astype(bool), "marker"].drop_duplicates().tolist()
    checks.add("workbook markers not quantified in the control data (not plotted; labels_core)", ", ".join(unplotted) or "none",
               None)
    if not checks.ok:
        run.table(checks.frame(), "dotplot_checks")
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("A check failed; see ctrl_rna_core_dotplot_checks.csv.")
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
        o = orders[version]
        mk = o["marker"].tolist()
        prots = [panel_for(vals, col, rows, o, f"Protein (SCP-MS, control DCT cells), {L.LABEL_DISPLAY[col]}")
                 for col in L.LABEL_COLUMNS]
        lvl = T.LEVEL_SHORT[derived]
        s = rv[(rv["level"] == lvl) & (rv["rna_source"] == src)]
        rrows = list(T.rna_groups(derived))
        frac = s.pivot(index="group", columns="marker", values="frac_detected").reindex(index=rrows, columns=mk)
        colr = s.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(index=rrows, columns=mk)
        nrow = s.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(rrows)
        rna = d7.GreyPanel(f"RNA (Mouse Kidney Atlas, {T.SOURCE_TEXT[src].replace('atlas ', '')})", rrows,
                           [int(n) for n in nrow], [T.RNA_PALETTE[r] for r in rrows], mk, prots[0].groups,
                           frac.to_numpy(float), colr.to_numpy(float), [False] * len(mk),
                           grey_rows=[n == 0 for n in nrow], grey_cols=[not gene_of[m] for m in mk],
                           n_text=[f"{int(n):,}" if n > 0 else "–" for n in nrow])
        vtxt = ("labels_core panel markers (Fig 3B replacement, with Col5a2)" if version == "panel"
                else "DCT V1 workbook markers quantified in the control data, under the first workbook type listing them")
        note = (f"{vtxt}; columns as the labels_core DCT dotplot. Protein: dot = fraction of the label's cells with intensity "
                "> 0; colour = mean z-scored ln(1 + intensity), scaled 0–1 per protein across the four rows (Unknown "
                "included). RNA: atlas Celltype cells" + (" (" + L.ATLAS_DCT_SPLIT_CAVEAT.rstrip(".") + ")" if derived else "")
                + "; dot = fraction with counts > 0; colour = mean log-normalized expression scaled 0–1 per gene across "
                "the RNA rows" + (f"; grey columns: not an atlas gene ({', '.join(m for m in mk if not gene_of[m])})"
                                  if any(not gene_of[m] for m in mk) else "")
                + ". Row numbers = cells. " + L.UNPAIRED_NOTE)
        cv, pitch = d7.build_paired([*prots, rna], "stacked", note)
        tc = t6dot.text_checks(cv)
        name = f"dotplot_{version}_{lvl}_{src}"
        paths = run.figure(cv.fig, name, f"paired RNA / control protein DCT dotplot, {version}, {lvl}, RNA {src}",
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
    run.finish({"layout": "stacked: protein DE-rule labels, protein submitted cluster labels, RNA (same columns)",
                "protein_rows": rows, "rna_colour": "mean log-normalized, min-max 0-1 per gene across RNA rows",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see ctrl_rna_core_dotplot_checks.csv.")


if __name__ == "__main__":
    main()
