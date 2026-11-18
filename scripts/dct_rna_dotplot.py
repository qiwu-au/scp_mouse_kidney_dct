#!/usr/bin/env python
"""dct_rna step 4: paired DCT marker dotplot, RNA atlas groups below the SCP labels of both conditions (dct_labels order).

Analysis plan,
step 4 (port of ctrl_rna_core_dotplot.py; ``rna_values``, ``FIGURES`` and ``VALUE_TOL`` of atlas_dct_dotplot,
``plotted`` of ctrl_rna_core_dotplot and ``panel_for`` of labels_core_figures imported unchanged).

Panels (same columns, stacked; dotplot / kidney_rna building blocks and encoding, SISKA Fig 1c layout)
-------------------------------------------------------------------------------------------
- Protein, Ctrl primary / Ctrl second / ZeroK primary / ZeroK second: the dct_labels dotplot values (re-checked against the
  dct_labels objects): dot = fraction of the label's cells with intensity > 0; colour = group mean of the per-protein
  z-scored ln(1 + intensity), rescaled 0-1 per protein across the labels present under the column (dct_labels; Unknown its
  own row). Rows: the labels present (as dct_labels Fig 3B).
- RNA: atlas groups (published DCT, DCT-CNT, CNT, CTAL, MTAL; or the atlas_dct-derived DCT1 (atlas_dct), DCT2 (atlas_dct) with
  DCT-CNT, CNT, CTAL, MTAL): dot = fraction with counts > 0; colour = mean log-normalized expression (10,000 per cell,
  log1p), rescaled 0-1 per gene across the RNA rows.
- Columns: dct_labels 'panel' order (Core | DCT1 | DCT2 | TAL | Cycling; main) and dct_labels 'workbook' order (DCT V1 workbook
  markers quantified in both conditions, under the first workbook type listing them).
- RNA subsets: all atlas cells (primary); nuclei only and cells only (published groups, panel order).

Inputs (read-only): dct_labels marker order and dotplot values; dct_labels labelled objects and canonical labels; atlas_dct region subset
                    and split table.
Outputs: results/figures/dct_rna/dct_rna_dotplot_{panel,workbook}_{published,derived}_{all,nuclei,cells}
         (.pdf/.png/.tif); results/tables/dct_rna/dct_rna_dotplot_rna_values.csv,
         dct_rna_dotplot_protein_recheck.csv, dct_rna_dotplot_checks.csv, dct_rna_dotplot_figure_checks.csv

Command (project root):  python scripts/dct_rna_dotplot.py
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
import dotplot_lib as t6
import dotplot_dotplots as t6dot
import kidney_rna_dotplots as d7            # GreyPanel, build_paired (180 mm, stacked)
import kidney_rna_lib as L7
import atlas_dct_lib as T
import dct_rna_lib as L
from atlas_dct_dotplot import FIGURES, VALUE_TOL, rna_values
from labels_core_figures import panel_for
from ctrl_rna_core_dotplot import plotted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_rna step 4: paired RNA / protein DCT marker dotplot, both conditions.")
    cc.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "dotplot", args)
    L.ported_from(run, "ctrl_rna_core_dotplot.py", "atlas_dct_dotplot.py")
    checks = L.Checks(L.DATASET)
    order = L.read_table(run, "marker_order_dct", L.DCT_LABELS_TASK)
    vals = L.read_table(run, "dotplot_values", L.DCT_LABELS_TASK)
    region = T.load_region(run, with_split=True)

    # protein values: re-check the dct_labels values against the dct_labels labelled objects ----------------------------------
    rows_of, recheck = {}, []
    for cond in L.CONDITIONS:
        s = L.SETTINGS[cond]
        adata, labels, _, _ = L.load_scp(run, cond, checks)
        v = vals[vals["condition"] == cond]
        markers_all = sorted(set(v["marker"]))
        for col in s.LABEL_COLUMNS:
            rows = L.present(labels, col)
            rows_of[(cond, col)] = rows
            stored_rows = list(v[v["label_column"] == col].drop_duplicates("group").sort_values("group_position")["group"])
            rec = t6.group_values(adata, labels[col], rows, markers_all)
            mg = v[v["label_column"] == col].merge(rec, on=["group", "marker"], suffixes=("_stored", "_new"))
            d_f = float(np.max(np.abs(mg["frac_detected_stored"] - mg["frac_detected_new"])))
            d_z = float(np.max(np.abs(mg["mean_z_stored"] - mg["mean_z_new"])))
            n_ok = bool((mg["n_cells_stored"] == mg["n_cells_new"]).all())
            ok = (stored_rows == rows and len(mg) == len(rows) * len(markers_all) and n_ok
                  and max(d_f, d_z) < VALUE_TOL)
            checks.add(f"{col} ({L.role_of(cond, col)}): dct_labels dotplot values reproduce from the dct_labels object (rows; entries; "
                       "cells per row equal; max abs difference of detection, mean z)",
                       f"{', '.join(rows)}; {len(mg)} of {len(rows) * len(markers_all)}; {n_ok}; {d_f:.1e}; {d_z:.1e}",
                       ok, f"< {VALUE_TOL:g}", cond)
            recheck.append({"condition": cond, "label_role": L.role_of(cond, col), "label_column": col,
                            "rows": ";".join(rows), "absent_labels": ";".join(L.absent(labels, col)),
                            "n_entries": len(mg), "max_abs_diff_detection": d_f, "max_abs_diff_mean_z": d_z,
                            "cells_equal": n_ok, "pass": ok})
        del adata
    look = L7.atlas_gene_lookup(region.var_names)
    orders = {ver: plotted(order, ver) for ver in ("panel", "workbook")}
    markers = sorted(set().union(*[set(o["marker"]) for o in orders.values()]))
    gene_of = {m: look.get(m.lower(), "") for m in markers}
    absent_genes = [m for m in markers if not gene_of[m]]
    checks.add("plotted markers absent from the atlas genes (grey RNA columns)", ", ".join(absent_genes) or "none", None)
    unplotted = order.loc[~order["plotted"].astype(bool), "marker"].drop_duplicates().tolist()
    checks.add("workbook markers not quantified (not plotted; dct_labels)", ", ".join(unplotted) or "none", None)
    L.stop_if_failed(run, checks, "dotplot_checks")
    ln = T.lognorm(region.X)
    rv = rna_values(region, ln, markers, gene_of)
    for (lvl, src), s_ in rv.groupby(["level", "rna_source"]):
        d = s_.drop_duplicates("group")
        checks.add(f"RNA {lvl} / {src}: atlas cells per row", "; ".join(f"{r.group} {r.n_atlas_cells}"
                                                                       for r in d.itertuples()), None)

    t6dot.plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    fig_rows = []
    for version, derived, src in FIGURES:
        o = orders[version]
        mk = o["marker"].tolist()
        prots = []
        for cond in L.CONDITIONS:
            for col in L.SETTINGS[cond].LABEL_COLUMNS:
                prots.append(panel_for(vals[vals["condition"] == cond], col, rows_of[(cond, col)], o,
                                       f"Protein (SCP-MS), {cond} DCT cells, {col} ({L.role_of(cond, col)})"))
        lvl = T.LEVEL_SHORT[derived]
        s_ = rv[(rv["level"] == lvl) & (rv["rna_source"] == src)]
        rrows = list(T.rna_groups(derived))
        frac = s_.pivot(index="group", columns="marker", values="frac_detected").reindex(index=rrows, columns=mk)
        colr = s_.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(index=rrows, columns=mk)
        nrow = s_.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(rrows)
        rna = d7.GreyPanel(f"RNA (Mouse Kidney Atlas, {T.SOURCE_TEXT[src].replace('atlas ', '')})", rrows,
                           [int(n) for n in nrow], [T.RNA_PALETTE[r] for r in rrows], mk, prots[0].groups,
                           frac.to_numpy(float), colr.to_numpy(float), [False] * len(mk),
                           grey_rows=[n == 0 for n in nrow], grey_cols=[not gene_of[m] for m in mk],
                           n_text=[f"{int(n):,}" if n > 0 else "–" for n in nrow])
        vtxt = ("dct_labels panel markers (Fig 3B replacement; Ptgfr and Trpv5 added, Col5a2 not quantified)"
                if version == "panel" else "DCT V1 workbook markers quantified in both conditions, under the first "
                "workbook type listing them (dct_labels)")
        note = (f"{vtxt}; columns as the dct_labels DCT dotplot. Protein: rows = labels with cells under the column (primary "
                "columns: no ProLIF; ZeroK primary: no DCT2, does not apply); dot = fraction of the label's cells with "
                "intensity > 0; colour = mean z-scored ln(1 + intensity), scaled 0–1 per protein across the rows of "
                "the column (two rows under the ZeroK primary column, so 0 or 1). RNA: atlas Celltype cells"
                + (" (" + L.ATLAS_DCT_SPLIT_CAVEAT.rstrip(".") + ")" if derived else "")
                + "; dot = fraction with counts > 0; colour = mean log-normalized expression scaled 0–1 per gene across "
                "the RNA rows" + (f"; grey columns: not an atlas gene ({', '.join(m for m in mk if not gene_of[m])})"
                                  if any(not gene_of[m] for m in mk) else "")
                + ". Row numbers = cells. " + L.UNPAIRED_NOTE + " " + L.ZEROK_NOTE)
        cv, pitch = d7.build_paired([*prots, rna], "stacked", note)
        tc = t6dot.text_checks(cv)
        name = f"dotplot_{version}_{lvl}_{src}"
        paths = run.figure(cv.fig, name, f"paired RNA / protein DCT dotplot, Ctrl and ZeroK, {version}, {lvl}, RNA {src}",
                           formats=L.FORMATS, tight=False)
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
    figs = L.figure_rows_frame(fig_rows, "dotplot")
    checks.add("figures passing the format checks (180 mm, Arial >= 7 pt, no overlap, PDF/PNG/TIFF)",
               f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    rv.insert(0, "condition", "both")                     # atlas values: the same RNA rows sit under both conditions
    run.table(rv, "dotplot_rna_values")
    run.table(pd.DataFrame(recheck), "dotplot_protein_recheck")
    run.table(figs, "dotplot_figure_checks")
    run.table(checks.frame(), "dotplot_checks")
    run.finish({"layout": "stacked: protein Ctrl primary, Ctrl second, ZeroK primary, ZeroK second, RNA (same columns)",
                "protein_rows": {f"{c}:{col}": r for (c, col), r in rows_of.items()},
                "protein_values": "dct_labels dct_labels_dotplot_values.csv (re-checked)",
                "rna_colour": "mean log-normalized, min-max 0-1 per gene across RNA rows", "formats": list(L.FORMATS),
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see dct_rna_dotplot_checks.csv.")


if __name__ == "__main__":
    main()
