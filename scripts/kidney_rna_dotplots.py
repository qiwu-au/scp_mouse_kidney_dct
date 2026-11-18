#!/usr/bin/env python
"""kidney_rna step 3: paired protein (SCP-MS) and RNA (Mouse Kidney Atlas) marker dotplots.

Analysis plan,
step 3 (reviewer R1: "dotplot based on SN_Seq for the marker genes used for annotation").

Panels (same rows, same columns, same encoding)
-----------------------------------------------
- Protein panel: the published kidney_annotation / dotplot dotplot values, re-read and checked against the kidney_object canonical object
  (dot = fraction of the label's cells with intensity > 0; colour = group mean of the per-protein z-scored
  ln(1 + intensity), rescaled 0-1 per protein across the rows).
  kidney_t15_label: kidney_annotation (a_endo) marker order and values; kidney_pre_label: dotplot kidney marker order and values.
- RNA panel: for each row, the atlas cells of the row's expected Celltypes in the approved mapping
  (kidney_rna_celltype_mapping.csv; mixed classes not included), from the kidney_rna atlas pseudobulk.
  Dot = fraction of those cells with counts > 0; colour = mean log-normalized expression (10,000 per cell, log1p),
  rescaled 0-1 per gene across the rows that have atlas cells (as the protein panel).
  Rows without an atlas counterpart (JG, Unknown) or without atlas cells in the subset, and genes absent from the
  atlas, are shaded grey.
- RNA subsets (authors, 14 Oct 2026): all atlas cells (primary); nuclei only (snRNA-seq) and cells only (scRNA-seq)
  as supplements (atlas obs 'Source').
- Layout (authors, 14 Oct 2026): main versions side by side; full versions stacked (protein above RNA, columns
  aligned). Format checks as dotplot (180 mm, Arial >= 7 pt, no text overlap, PDF + 600-dpi PNG + LZW TIFF).

Inputs (read-only): kidney_rna mapping table and atlas pseudobulk; kidney_annotation a_endo marker order and dotplot values; dotplot kidney
marker order, dotplot values and label order; kidney_object canonical kidney object (value check) and kidney_annotation labels.
Outputs: results/figures/kidney_rna/kidney_rna_dotplot_{t15,pre}_{main,full}_{all,nuclei,cells}.pdf/.png/.tif
         results/tables/kidney_rna/kidney_rna_dotplot_rna_values.csv, kidney_rna_dotplot_protein_check.csv,
         kidney_rna_dotplot_marker_coverage.csv, kidney_rna_dotplot_figure_checks.csv, kidney_rna_dotplot_checks.csv

Command (project root):  python scripts/kidney_rna_dotplots.py
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

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dotplot_lib as t6
import dotplot_dotplots as t6dot          # Panel, Canvas, legends, checks (sets Arial, 7 pt)
import kidney_rna_lib as L
import kidney_annotation_chosen_labels as t15c      # kidney_annotation row order and palette

DOTPLOT_DIR = cc.OUT_ROOTS["tables"] / "dotplot"
MAPPING = cc.out_path(L.TASK, "tables", "celltype_mapping", ".csv")
RNA_SOURCES = {"all": (None, "all cells and nuclei"), "nuclei": ("Nuclei", "nuclei, snRNA-seq"),
               "cells": ("Cell", "cells, scRNA-seq")}
PRIMARY_SOURCE = "all"
LABEL_SHORT = {"kidney_t15_label": "t15", "kidney_pre_label": "pre"}
LABEL_TITLE = {"kidney_t15_label": "kidney_annotation labels", "kidney_pre_label": "Fig 2A labels"}
VERSION_TEXT = {
    "kidney_t15_label": {"main": "V3 + endothelial-row marker proteins, up to 3 per cell type (kidney_annotation selection)",
                         "full": "all quantified V3 + endothelial-row marker proteins"},
    "kidney_pre_label": {"main": "V3 marker proteins, up to 3 per cell type (dotplot selection)",
                         "full": "all quantified V3 marker proteins"},
}
GREY = "#e9e9e6"
WIDTH = 180.0
P_MAX = 4.5
VALUE_TOL = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_rna step 3: paired protein / RNA marker dotplots.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "dotplot_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_rna_dotplot_checks.csv.")


@dataclass
class GreyPanel(t6dot.Panel):
    """dotplot panel with grey rows / columns (no RNA data) and free row-count text."""

    grey_rows: list = field(default_factory=list)
    grey_cols: list = field(default_factory=list)
    n_text: list = field(default_factory=list)

    def row_labels(self) -> list[str]:
        if self.n_text:
            return [f"{r} ({t})" for r, t in zip(self.rows, self.n_text)]
        return super().row_labels()

    def draw(self, cv, x0, y0, pitch):
        geo = super().draw(cv, x0, y0, pitch)
        gl, gt = geo["grid_left"], geo["grid_top"]
        nr, nc = len(self.rows), len(self.markers)
        for i, g in enumerate(self.grey_rows):
            if g:
                cv.rect(gl, gt + i * pitch, nc * pitch, pitch, facecolor=GREY, edgecolor="none", zorder=0.5)
        for j, g in enumerate(self.grey_cols):
            if g:
                cv.rect(gl + j * pitch, gt, pitch, nr * pitch, facecolor=GREY, edgecolor="none", zorder=0.5)
        return geo


# ---------------------------------------------------------------------------
# Figure assembly (side by side or stacked; dotplot building blocks)
# ---------------------------------------------------------------------------
def build_paired(panels: list[t6dot.Panel], layout: str, note: str, gap: float = 8.0, vgap: float = 3.0):
    M, LH = t6dot.MARGIN, t6dot.LINE_H
    lines = t6dot.wrap_mm(note, WIDTH - 2 * M)
    note_h = 1.5 + len(lines) * LH * 1.15
    cw, ch = t6dot.color_legend_dims()
    if layout == "side":
        pitch = t6dot.solve_pitch(panels, WIDTH, "below", P_MAX, gap)
        sizes = [p.size(pitch) for p in panels]
        sw, sh, _ = t6dot.size_legend_dims(pitch)
        leg_w, leg_h = sw + 6.0 + cw, max(sh, ch)
        content_w = max(sum(w for w, _ in sizes) + gap * (len(panels) - 1), leg_w)
        body_h = max(h for _, h in sizes)
        height = 2 * M + body_h + 2.5 + leg_h + note_h
        cv = t6dot.Canvas(WIDTH, height)
        x0 = x = (WIDTH - content_w) / 2
        geo = []
        for p, (w, _) in zip(panels, sizes):
            geo.append(p.draw(cv, x, M, pitch))
            x += w + gap
        leg_y = max(g["bottom"] for g in geo) + 2.5
        t6dot.draw_legends(cv, x0, leg_y, pitch, stacked=False)
        note_y = leg_y + leg_h + 1.5
    else:
        ncols = {len(p.markers) for p in panels}
        if len(ncols) != 1:
            raise ValueError("stacked panels need the same columns")
        ncol = ncols.pop()
        lw = max(p.row_label_w() for p in panels)
        leg_w = max(t6dot.size_legend_dims(P_MAX)[0], cw)
        pitch = min(P_MAX, (WIDTH - 2 * M - lw - 5.0 - leg_w) / ncol)
        sw, sh, _ = t6dot.size_legend_dims(pitch)
        leg_w, leg_h = max(sw, cw), sh + 3.0 + ch
        sizes = [p.size(pitch) for p in panels]
        body_h = sum(h for _, h in sizes) + vgap * (len(panels) - 1)
        top_offset = panels[0].title_h() + panels[0].bracket_h(pitch)
        content_w = lw + ncol * pitch + 5.0 + leg_w
        if content_w > WIDTH - 2 * M + 1e-6:
            raise ValueError(f"content {content_w:.1f} mm wider than {WIDTH} mm")
        height = 2 * M + max(body_h, top_offset + leg_h) + note_h
        cv = t6dot.Canvas(WIDTH, height)
        x0 = (WIDTH - content_w) / 2
        y = M
        geo = []
        for p, (_, h) in zip(panels, sizes):
            geo.append(p.draw(cv, x0 + lw - p.row_label_w(), y, pitch))
            y += h + vgap
        t6dot.draw_legends(cv, geo[0]["grid_right"] + 5.0, geo[0]["grid_top"], pitch, stacked=True)
        note_y = M + max(body_h, top_offset + leg_h) + 1.5
    nx = max(M, min((WIDTH - max(t6dot.text_mm(s)[0] for s in lines)) / 2, 2 * M))
    cv.text(nx, note_y, "\n".join(lines), ha="left", va="top", linespacing=1.15, color=t6dot.INK["secondary"])
    return cv, pitch


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------
def rna_values(pb, mapping: pd.DataFrame, col: str, rows: list[str], markers: list[str],
               gene_of: dict[str, str]) -> pd.DataFrame:
    """Long table per RNA subset x row x marker: n atlas cells, detection, mean log-normalized, scaled 0-1."""
    out = []
    mp = mapping[mapping["label_column"] == col].set_index("scp_label")
    in_atlas = [m for m in markers if gene_of.get(m)]
    genes = [gene_of[m] for m in in_atlas]
    for src, (source, _) in RNA_SOURCES.items():
        for pos, r in enumerate(rows, start=1):
            exp = mp.loc[r, "atlas_celltype_expected"]
            classes = [] if pd.isna(exp) or exp == "" else str(exp).split(";")
            crit = {"Celltype": classes}
            if source:
                crit["Source"] = source
            agg = L.aggregate(pb, L.group_mask(pb, **crit), genes) if classes else None
            n = int(agg["n_cells"].iloc[0]) if agg is not None else 0
            for m in markers:
                g = gene_of.get(m, "")
                has = bool(g) and n > 0
                rec = agg.loc[g] if has else None
                out.append({"label_column": col, "rna_source": src, "group": r, "group_position": pos,
                            "atlas_celltypes": ";".join(classes), "n_atlas_cells": n, "marker": m, "atlas_gene": g,
                            "gene_in_atlas": bool(g), "row_has_atlas_cells": n > 0,
                            "frac_detected": float(rec["frac_detected"]) if has else 0.0,
                            "mean_lognorm": float(rec["mean_lognorm"]) if has else np.nan})
    v = pd.DataFrame(out)
    grp = v.groupby(["label_column", "rna_source", "marker"])["mean_lognorm"]
    lo, hi = grp.transform("min"), grp.transform("max")
    v["mean_scaled_0_1"] = ((v["mean_lognorm"] - lo) / (hi - lo).where(hi > lo)).fillna(0.0)
    v.loc[v["mean_lognorm"].isna(), "mean_scaled_0_1"] = 0.0
    return v


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "dotplots", args)
    checks = t5.Checks(L.DATASET)
    adata, labels = L.load_kidney(run, checks)
    mapping = pd.read_csv(run.use_input(MAPPING, "kidney_rna approved SCP label -> atlas mapping (step 2)"))
    approved = mapping["status"].astype(str).str.contains("approved by the authors").all()
    checks.add("mapping table approved by the authors", f"{approved}", approved)
    pb = L.read_pseudobulk(run)
    if not checks.ok:
        stop(run, checks)

    t15_order = pd.read_csv(run.use_input(L.KIDNEY_ANNOTATION_MARKER_ORDER, "kidney_annotation a_endo dotplot marker order"))
    t15_vals = pd.read_csv(run.use_input(L.KIDNEY_ANNOTATION_DOTPLOT_VALUES, "kidney_annotation a_endo dotplot values"))
    t06_order = pd.read_csv(run.use_input(DOTPLOT_DIR / "dotplot_marker_order_kidney.csv", "dotplot kidney marker order"))
    t06_vals = pd.read_csv(run.use_input(DOTPLOT_DIR / "dotplot_kidney_dotplot_values.csv", "dotplot kidney dotplot values"))
    t06_lo = pd.read_csv(run.use_input(DOTPLOT_DIR / "dotplot_label_order.csv", "dotplot label order"))

    # row orders and colours (as the published figures)
    t15_rows = (t15_vals.drop_duplicates("group").sort_values("group_position")["group"].tolist())
    _, t15_expected = t15c.t15_rows(labels["kidney_t15_label"].unique())
    checks.add("kidney_annotation dotplot rows equal the kidney_annotation row order", "; ".join(t15_rows), t15_rows == t15_expected)
    colors = t15c.palette(adata, t15_rows)
    vc = labels["kidney_t15_label"].value_counts()
    lo_t15 = pd.DataFrame({"label_column": "kidney_t15_label", "label": t15_rows, "position": range(1, len(t15_rows) + 1),
                           "plotted": True, "n_cells": [int(vc[r]) for r in t15_rows],
                           "colour": [colors.get(r, t6dot.INK["muted"]) for r in t15_rows]})
    lo_pre = t06_lo[(t06_lo["label_column"] == "kidney_pre_label") & t06_lo["plotted"]].copy()
    spec = {
        "kidney_t15_label": {"order": t15_order, "values": t15_vals, "label_order": lo_t15,
                             "rows": t15_rows},
        "kidney_pre_label": {"order": t06_order, "values": t06_vals[t06_vals["label_column"] == "kidney_pre_label"],
                             "label_order": lo_pre,
                             "rows": lo_pre.sort_values("position")["label"].tolist()},
    }

    # protein values: re-check against the canonical object
    prot_rows = []
    for col, sp in spec.items():
        markers = sorted(set(sp["values"]["marker"]))
        recomputed = t6.group_values(adata, labels[col], sp["rows"], markers)
        merged = sp["values"].merge(recomputed, on=["group", "marker"], suffixes=("_stored", "_recomputed"))
        d_frac = float(np.max(np.abs(merged["frac_detected_stored"] - merged["frac_detected_recomputed"])))
        d_z = float(np.max(np.abs(merged["mean_z_stored"] - merged["mean_z_recomputed"])))
        n_exp = len(sp["rows"]) * len(markers)
        ok = len(merged) == n_exp and max(d_frac, d_z) < VALUE_TOL
        checks.add(f"{col}: stored protein dotplot values reproduce from the kidney_object object (rows x markers; max abs "
                   "difference of detection, mean z)", f"{len(merged)} of {n_exp}; {d_frac:.1e}; {d_z:.1e}", ok,
                   f"< {VALUE_TOL:g}")
        prot_rows.append({"label_column": col, "n_rows": len(sp["rows"]), "n_markers": len(markers),
                          "n_matched": len(merged), "max_abs_diff_frac": d_frac, "max_abs_diff_mean_z": d_z,
                          "pass": ok})
        for version in ("main", "full"):
            o = t6dot.plotted_order(sp["order"], version)
            missing = sorted(set(o["marker"]) - set(markers))
            checks.add(f"{col} {version}: plotted markers all have stored values", ", ".join(missing) or "all",
                       not missing)
    if not checks.ok:
        stop(run, checks)

    # RNA values
    look = L.atlas_gene_lookup(pb.var_names)
    all_markers = sorted(set().union(*[set(t6dot.plotted_order(sp["order"], v)["marker"])
                                       for sp in spec.values() for v in ("main", "full")]))
    gene_of = {m: look.get(m.lower(), "") for m in all_markers}
    rv = pd.concat([rna_values(pb, mapping, col, sp["rows"], all_markers, gene_of) for col, sp in spec.items()],
                   ignore_index=True)
    absent = sorted(m for m, g in gene_of.items() if not g)
    checks.add("plotted marker proteins absent from the atlas genes (grey RNA columns)",
               f"{len(absent)} of {len(all_markers)}: {', '.join(absent) or 'none'}", None)
    for col, sp in spec.items():
        for src in RNA_SOURCES:
            sub = rv[(rv["label_column"] == col) & (rv["rna_source"] == src)].drop_duplicates("group")
            empty = sub.loc[~sub["row_has_atlas_cells"], "group"].tolist()
            checks.add(f"{col} / RNA {src}: atlas cells per row; rows without atlas cells",
                       "; ".join(f"{r.group} {r.n_atlas_cells}" for r in sub.itertuples()) +
                       f" | grey: {', '.join(empty) or 'none'}", None)
    cov = pd.DataFrame([{"marker": m, "atlas_gene": gene_of[m], "gene_in_atlas": bool(gene_of[m]),
                         **{f"plotted_{LABEL_SHORT[col]}_{v}": m in set(t6dot.plotted_order(sp["order"], v)["marker"])
                            for col, sp in spec.items() for v in ("main", "full")}} for m in all_markers])

    # figures (Arial again: importing kidney_annotation_chosen_labels re-applies the qc_mito style, as in kidney_annotation)
    t6dot.plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    t6dot.SIZE_TITLE = "Fraction of cells (%) with\nintensity / counts > 0"
    t6dot.COLOR_TITLE = "Mean in group (scaled\n0–1 per protein / gene)"
    fig_rows = []
    for col, sp in spec.items():
        for version in ("main", "full"):
            order = t6dot.plotted_order(sp["order"], version)
            prot = t6dot.make_panel(f"Protein (SCP-MS), {LABEL_TITLE[col]}", sp["values"], sp["label_order"], col,
                                    order)
            markers = order["marker"].tolist()
            repeat = order["repeat"].fillna(False).astype(bool).tolist() if "repeat" in order else [False] * len(markers)
            for src, (_, src_text) in RNA_SOURCES.items():
                sub = rv[(rv["label_column"] == col) & (rv["rna_source"] == src)]
                frac = sub.pivot(index="group", columns="marker", values="frac_detected").reindex(
                    index=sp["rows"], columns=markers)
                colr = sub.pivot(index="group", columns="marker", values="mean_scaled_0_1").reindex(
                    index=sp["rows"], columns=markers)
                nrow = sub.drop_duplicates("group").set_index("group")["n_atlas_cells"].reindex(sp["rows"])
                rna = GreyPanel(f"RNA (Mouse Kidney Atlas, {src_text})", sp["rows"], [int(n) for n in nrow],
                                prot.row_colors, markers, prot.groups, frac.to_numpy(float), colr.to_numpy(float),
                                repeat, grey_rows=[n == 0 for n in nrow], grey_cols=[not gene_of[m] for m in markers],
                                n_text=[f"{int(n):,}" if n > 0 else "–" for n in nrow])
                dagger = (f" {t6dot.DAGGER} A marker listed for this cell type whose group mean is higher in another "
                          "cell type that lists it, repeated so that each cell type has a column." if any(repeat) else "")
                vt = VERSION_TEXT[col][version]
                note = (f"{vt[0].upper() + vt[1:]}; columns and rows as the protein dotplot. Protein: "
                        "dot = fraction of the label's cells with intensity > 0; colour = mean z-scored ln(1 + "
                        "intensity). RNA: atlas cells of the Celltypes mapped to each label (kidney_rna_celltype_"
                        "mapping.csv; mixed classes not included); dot = fraction with counts > 0; colour = mean "
                        "log-normalized expression. Colours scaled 0–1 per protein / gene across rows; row numbers = "
                        "cells. Grey: no atlas counterpart or no atlas cells in this subset (row); gene not in the "
                        "atlas (column)." + dagger)
                layout = "side" if version == "main" else "stacked"
                cv, pitch = build_paired([prot, rna], layout, note)
                tc = t6dot.text_checks(cv)
                name = f"dotplot_{LABEL_SHORT[col]}_{version}_{src}"
                paths = run.figure(cv.fig, name, f"paired protein / RNA dotplot, {col}, {version}, RNA {src}",
                                   formats=(".pdf", ".png", ".tif"), tight=False)
                fc = t6dot.file_checks(paths, cv.W, cv.H)
                arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
                ok = bool(tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0
                          and tc["n_text_outside_figure"] == 0 and fc["width_ok"] and fc.get("tif_mode") == "RGB"
                          and fc.get("pdf_truetype_embedded") and fc.get("pdf_arial"))
                fig_rows.append({"figure": name, "label_column": col, "version": version, "rna_source": src,
                                 "primary": src == PRIMARY_SOURCE, "layout": layout, "width_mm": round(cv.W, 2),
                                 "height_mm": round(cv.H, 2), "pitch_mm": round(pitch, 2), "n_rows": len(sp["rows"]),
                                 "n_columns": len(markers), **tc, **fc, "pass": ok})
                print(f"  {name}: {cv.W:.0f} x {cv.H:.1f} mm, pitch {pitch:.2f} mm, overlaps {tc['n_text_overlaps']}, "
                      f"outside {tc['n_text_outside_figure']}, pass {ok}", flush=True)
                t6dot.plt.close(cv.fig)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks (180 mm, Arial >= 7 pt, no overlap, PDF/PNG/TIFF)",
               f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))

    run.table(rv, "dotplot_rna_values")
    run.table(pd.DataFrame(prot_rows), "dotplot_protein_check")
    run.table(cov, "dotplot_marker_coverage")
    run.table(figs, "dotplot_figure_checks")
    run.table(checks.frame(), "dotplot_checks")
    run.finish({"rna_sources": {k: v[1] for k, v in RNA_SOURCES.items()}, "primary_rna_source": PRIMARY_SOURCE,
                "layout": "main side by side; full stacked (authors, 14 Oct 2026)", "width_mm": WIDTH,
                "max_pitch_mm": P_MAX, "rna_rows": "expected Celltypes of the approved mapping (mixed not included)",
                "rna_colour": "mean log-normalized expression (10,000 per cell, log1p), min-max 0-1 per gene "
                              "across rows with atlas cells", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see kidney_rna_dotplot_checks.csv.")


if __name__ == "__main__":
    main()
