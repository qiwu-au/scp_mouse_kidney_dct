#!/usr/bin/env python
"""pt_tal step 5: paired dotplot, RNA atlas subtypes next to the SCP groups, for the subtype markers.

Analysis plan, step 5.
Layout and encoding as dotplot / kidney_rna / atlas_dct (SISKA Fig 1c style; ``kidney_rna_dotplots.build_paired``, stacked panels with
the same columns): protein kidney_annotation labels, protein Fig 2A labels, RNA atlas classes.

Panels
------
- Protein (kidney_annotation labels): PCT, PST, TAL; protein (Fig 2A labels): PCT, PST, TAL, MD. Dot = fraction of the label's cells
  with intensity > 0; colour = group mean of the canonical z-scored ln(1 + intensity), scaled 0-1 per protein across
  the panel's rows. Grey columns: not quantified as protein.
- RNA: atlas Celltype PTS1, PTS2, PTS3, PTS3T2, CTAL, MTAL, MD (step-1 class statistics). Dot = fraction with counts
  > 0; colour = mean log-normalized expression scaled 0-1 per gene across the RNA rows. Grey: gene not in the atlas
  (column) or no atlas cells in the subset (row).
- Columns, main: the top 3 proteins (step-1 rank, kidney_annotation panels) of each one-vs-rest panel (PTS1, PTS2 (empty), PTS3,
  PTS3T2, CTAL, MTAL, MD), each protein once, under the class with the highest atlas mean among the panels listing it;
  then the plan's literature candidates not already shown. RNA: all atlas cells (primary), nuclei only, cells only.
- Columns, full (all atlas cells; four figures of <= 45 columns): PTS1 + PTS3 panels; PTS3T2 panel + PT literature;
  S1/S2 + S3 grouped panels; CTAL + MTAL + MD panels + TAL / MD literature.

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation labels; pt_tal step-1/2 tables (panels,
                    atlas_class_stats, literature).
Outputs: results/figures/pt_tal/pt_tal_dotplot_main_{all,nuclei,cells}, pt_tal_dotplot_full_{pt_s1_s3,
         pt_s3t2,pt_grouped,tal_md} (.pdf/.png/.tif); results/tables/pt_tal/pt_tal_dotplot_columns.csv,
         pt_tal_dotplot_values.csv, pt_tal_dotplot_figure_checks.csv, pt_tal_dotplot_checks.csv

Command (project root):  python scripts/pt_tal_dotplot.py
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
import dotplot_dotplots as t6dot
import dotplot_lib as t6
import kidney_rna_dotplots as d7
import pt_tal_lib as T

MAIN_TOP = 3
MAX_COLUMNS = 45                     # kidney_rna full dotplots: 45 columns passed the format checks at 180 mm
LIT_GROUP = "Literature (plan)"
PROTEIN_ROWS = {"kidney_t15_label": ["PCT", "PST", "TAL"], "kidney_pre_label": ["PCT", "PST", "TAL", "MD"]}
RNA_ROWS = list(T.REGION_CLASSES)
ONE_VS_REST = ("PTS1", "PTS2", "PTS3", "PTS3T2", "CTAL", "MTAL", "MD")
FULL = {
    "pt_s1_s3": (("PTS1", "PTS3"), None),
    "pt_s3t2": (("PTS3T2",), "PT"),
    "pt_grouped": (("S1/S2", "S3"), None),
    "tal_md": (("CTAL", "MTAL", "MD"), "TAL"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pt_tal step 5: paired RNA / protein subtype dotplot.")
    cc.add_common_args(parser)
    return parser.parse_args()


def place(panels: pd.DataFrame, contrasts, top: int | None, rna_mean: pd.DataFrame) -> list[tuple[str, str]]:
    """(group, protein) columns: panel proteins (rank <= top), each once, under the contrast whose group classes have
    the highest atlas mean; groups in contrast order."""
    p = panels[(panels["label_column"] == T.PRIMARY) & panels["panel"].isin(contrasts)]
    if top:
        p = p[p["rank"] <= top]
    listing = p.groupby("protein")["panel"].agg(list)
    gene_of = dict(zip(p["protein"], p["atlas_gene"]))
    chosen = {}
    for prot, cons in listing.items():
        def m(c):
            return float(rna_mean.loc[list(T.CONTRASTS[c][1]), gene_of[prot]].mean())
        chosen[prot] = max(cons, key=lambda c: (m(c), -contrasts.index(c)))
    cols = []
    for c in contrasts:
        prots = p[p["panel"] == c].sort_values("rank")["protein"]
        cols += [(c, x) for x in prots if chosen[x] == c and (c, x) not in cols]
    return cols


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "dotplot", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, _, _, _ = T.load_scp(run, checks)
    panels = T.read_table(run, "panels")
    cs = T.read_table(run, "atlas_class_stats")
    lit = T.read_table(run, "literature")
    colors = T.label_colors(adata)
    names = set(adata.var_names.astype(str))
    rna_all = cs[cs["rna_source"] == "all"].pivot(index="class", columns="atlas_gene", values="mean_lognorm")
    gene_lookup = {g.lower(): g for g in cs["atlas_gene"].unique()}

    # columns --------------------------------------------------------------------------------------------------
    lit_genes = [g for g, _, _ in T.LITERATURE]
    lit_seg = {g: ("PT" if grp.startswith("S") else "TAL") for g, grp, _ in T.LITERATURE}
    versions = {"main": place(panels, list(ONE_VS_REST), MAIN_TOP, rna_all)}
    shown = {p for _, p in versions["main"]}
    versions["main"] += [(LIT_GROUP, g) for g in lit_genes if g not in shown]
    for name, (contrasts, lit_part) in FULL.items():
        cols = place(panels, list(contrasts), None, rna_all)
        if lit_part:
            shown_ = {p for _, p in cols}
            cols += [(LIT_GROUP, g) for g in lit_genes if lit_seg[g] == lit_part and g not in shown_]
        versions[f"full_{name}"] = cols
    col_rows = []
    for v, cols in versions.items():
        checks.add(f"dotplot {v}: columns (<= {MAX_COLUMNS})", f"{len(cols)}", len(cols) <= MAX_COLUMNS)
        for j, (g, p) in enumerate(cols, start=1):
            col_rows.append({"version": v, "position": j, "group": g, "marker": p,
                             "quantified_protein": p in names, "atlas_gene": gene_lookup.get(p.lower(), ""),
                             "literature_candidate": p in lit_genes})
    columns = pd.DataFrame(col_rows)
    absent = columns.drop_duplicates("marker")
    checks.add("plotted markers not quantified as protein (grey protein columns)",
               ", ".join(absent.loc[~absent["quantified_protein"], "marker"]) or "none", None)
    checks.add("plotted markers not among the atlas genes (grey RNA columns)",
               ", ".join(absent.loc[absent["atlas_gene"] == "", "marker"]) or "none", None)
    if not checks.ok:
        run.table(checks.frame(), "dotplot_checks")
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("A check failed; see pt_tal_dotplot_checks.csv.")

    # values ----------------------------------------------------------------------------------------------------
    markers = sorted(set(columns["marker"]))
    quant = [m for m in markers if m in names]
    val_rows = []
    for col_, rows in PROTEIN_ROWS.items():
        gv = t6.group_values(adata, labels[col_], rows, quant)
        gv.insert(0, "panel", f"protein {T.LABEL_SHORT[col_]}")
        gv["label_column"] = col_
        gv["mean_scaled_0_1"] = t6.scaled_01(gv)
        val_rows.append(gv.rename(columns={"group": "row", "n_cells": "n"}))
    for src in T.RNA_SOURCES:
        s = cs[cs["rna_source"] == src]
        for m in markers:
            g = gene_lookup.get(m.lower(), "")
            for c in RNA_ROWS:
                r = s[(s["class"] == c) & (s["atlas_gene"] == g)] if g else s.iloc[0:0]
                n = int(s[s["class"] == c]["n_cells"].iloc[0])
                val_rows.append(pd.DataFrame([{
                    "panel": f"rna {src}", "row": c, "n": n, "marker": m,
                    "frac_detected": float(r["frac_detected"].iloc[0]) if len(r) and n else 0.0,
                    "mean_lognorm": float(r["mean_lognorm"].iloc[0]) if len(r) and n else np.nan}]))
    vals = pd.concat(val_rows, ignore_index=True)
    rna = vals["panel"].str.startswith("rna")
    grp = vals[rna].groupby(["panel", "marker"])["mean_lognorm"]
    lo, hi = grp.transform("min"), grp.transform("max")
    vals.loc[rna, "mean_scaled_0_1"] = ((vals.loc[rna, "mean_lognorm"] - lo) / (hi - lo).where(hi > lo)).fillna(0.0)
    vals.loc[rna & vals["mean_lognorm"].isna(), "mean_scaled_0_1"] = 0.0

    # figures ---------------------------------------------------------------------------------------------------
    t6dot.plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    figs_todo = [("main", src) for src in T.RNA_SOURCES] + [(v, "all") for v in versions if v.startswith("full")]
    fig_rows = []
    for v, src in figs_todo:
        cols = versions[v]
        mk = [p for _, p in cols]
        groups = t6dot.groups_of(pd.DataFrame({"group": [g for g, _ in cols]}))
        pans = []
        for col_, rows in PROTEIN_ROWS.items():
            s = vals[vals["panel"] == f"protein {T.LABEL_SHORT[col_]}"]
            frac = np.zeros((len(rows), len(mk)))
            colr = np.zeros((len(rows), len(mk)))
            for j, m in enumerate(mk):
                if m in names:
                    sm = s[s["marker"] == m].set_index("row").reindex(rows)
                    frac[:, j] = sm["frac_detected"].to_numpy()
                    colr[:, j] = sm["mean_scaled_0_1"].to_numpy()
            n = [int((labels[col_] == r).sum()) for r in rows]
            pans.append(d7.GreyPanel(f"Protein (SCP-MS), {T.LABEL_DISPLAY[col_]}", rows, n,
                                     [colors.get(r, T.INK["muted"]) for r in rows], mk, groups, frac, colr,
                                     [False] * len(mk), grey_rows=[False] * len(rows),
                                     grey_cols=[m not in names for m in mk]))
        s = vals[vals["panel"] == f"rna {src}"]
        frac = np.array([[float(s[(s["row"] == r) & (s["marker"] == m)]["frac_detected"].iloc[0]) for m in mk]
                         for r in RNA_ROWS])
        colr = np.array([[float(s[(s["row"] == r) & (s["marker"] == m)]["mean_scaled_0_1"].iloc[0]) for m in mk]
                         for r in RNA_ROWS])
        nrow = [int(s[s["row"] == r]["n"].iloc[0]) for r in RNA_ROWS]
        pans.append(d7.GreyPanel(f"RNA (Mouse Kidney Atlas, {T.SOURCE_TEXT[src].replace('atlas ', '')})", RNA_ROWS,
                                 nrow, [T.RNA_PALETTE[r] for r in RNA_ROWS], mk, groups, frac, colr, [False] * len(mk),
                                 grey_rows=[n_ == 0 for n_ in nrow],
                                 grey_cols=[not gene_lookup.get(m.lower(), "") for m in mk],
                                 n_text=[f"{n_:,}" if n_ > 0 else "–" for n_ in nrow]))
        what = ("top 3 proteins of each atlas one-vs-rest panel (kidney_annotation panels; PTS2 panel empty), each once, then the "
                "plan's literature candidates" if v == "main" else
                f"all proteins of the panels {', '.join(FULL[v[5:]][0])} (kidney_annotation panels)" +
                (" and the remaining literature candidates" if FULL[v[5:]][1] else ""))
        note = (f"Columns: {what}; brackets = atlas class whose panel lists the protein (highest atlas mean if several). "
                "Protein: dot = fraction of the label's cells with intensity > 0; colour = mean z-scored ln(1 + "
                "intensity), scaled 0–1 per protein across the panel's rows. RNA: atlas Celltype; dot = fraction with "
                "counts > 0; colour = mean log-normalized expression scaled 0–1 per gene across the RNA rows. Grey: "
                "not quantified as protein / not an atlas gene / no atlas cells. Row numbers = cells. RNA and protein "
                "come from different animals and studies. Fig 2A MD = kidney_annotation Endothelial.")
        cv, pitch = d7.build_paired(pans, "stacked", note)
        tc = t6dot.text_checks(cv)
        name = f"dotplot_{v}_{src}" if v == "main" else f"dotplot_{v}"
        paths = run.figure(cv.fig, name, f"paired RNA / protein PT-TAL subtype dotplot, {v}, RNA {src}",
                           formats=(".pdf", ".png", ".tif"), tight=False)
        fc = t6dot.file_checks(paths, cv.W, cv.H)
        arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
        ok = bool(tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
                  and fc["width_ok"] and fc.get("tif_mode") == "RGB" and fc.get("pdf_truetype_embedded")
                  and fc.get("pdf_arial"))
        fig_rows.append({"figure": name, "version": v, "rna_source": src, "primary": v == "main" and src == "all",
                         "width_mm": round(cv.W, 2), "height_mm": round(cv.H, 2), "pitch_mm": round(pitch, 2),
                         "n_columns": len(mk), **tc, **fc, "pass": ok})
        print(f"  {name}: {cv.W:.0f} x {cv.H:.1f} mm, pitch {pitch:.2f}, overlaps {tc['n_text_overlaps']} "
              f"{tc['overlaps'][:100]!r}, outside {tc['n_text_outside_figure']}, pass {ok}", flush=True)
        t6dot.plt.close(cv.fig)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks (180 mm, Arial >= 7 pt, no overlap, PDF/PNG/TIFF)",
               f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(columns, "dotplot_columns")
    run.table(vals, "dotplot_values")
    run.table(figs, "dotplot_figure_checks")
    run.table(checks.frame(), "dotplot_checks")
    run.finish({"main_top": MAIN_TOP, "max_columns": MAX_COLUMNS, "layout": "stacked: protein kidney_annotation, protein Fig 2A, RNA",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see pt_tal_dotplot_checks.csv.")


if __name__ == "__main__":
    main()
