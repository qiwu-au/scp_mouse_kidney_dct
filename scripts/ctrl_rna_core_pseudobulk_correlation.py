#!/usr/bin/env python
"""ctrl_rna_core step 2: pseudobulk Spearman correlation, RNA atlas groups x control SCP labels.

Analysis plan,
step 2 (port of atlas_dct_pseudobulk_correlation.py; ``rho_rows`` imported unchanged).

Groups
------
- RNA (atlas Celltype): published classes DCT, DCT-CNT, CNT, CTAL, MTAL; and the atlas_dct-derived level DCT1 (atlas_dct),
  DCT2 (atlas_dct), DCT-CNT, CNT, CTAL, MTAL (DCT unassigned (atlas_dct) left out). RNA subsets: all atlas cells (primary),
  nuclei only, cells only (obs 'Source').
- SCP: DCT1, DCT2, ProLIF and Unknown under ctrl_deg_label (primary) and ctrl_cluster_label (Unknown its own row,
  authors 28 Oct 2026). Sensitivity in the CSV: the three core labels only (Unknown left out of the gene selection and
  the z-scores, as atlas_dct); the labels split by sort plate (P1 / P2).

Values
------
- RNA: mean log-normalized expression (10,000 per cell, natural log1p); detection = counts > 0.
- Protein: mean ln(1 + intensity) with missing = 0 (labels_core intensity layer); detection = intensity > 0.
- Genes: the shared genes (step 1) detected in both modalities: detection >= t in at least one RNA group of the
  set and >= t in at least one SCP group of the label column; t = 10% (primary), 5% and 20%.
- Spearman rho across those genes for each RNA group x SCP group, on (a) the raw means and (b) z-scores of each
  gene across the groups within its modality (ddof 0).

Inputs (read-only): labels_core labelled control object and canonical labels; atlas_dct region subset and split table; ctrl_rna_core step 1.
Outputs: results/tables/ctrl_rna_core/ctrl_rna_core_pseudobulk_rho.csv, ctrl_rna_core_pseudobulk_rho_plate.csv,
         ctrl_rna_core_pseudobulk_genes.csv, ctrl_rna_core_pseudobulk_means.csv, ctrl_rna_core_pseudobulk_checks.csv,
         ctrl_rna_core_pseudobulk_figure_checks.csv;
         results/figures/ctrl_rna_core/ctrl_rna_core_pseudobulk_rho_{published,derived}_{all,nuclei,cells}.pdf/.png

Command (project root):  python scripts/ctrl_rna_core_pseudobulk_correlation.py
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
import atlas_dct_lib as T
import ctrl_rna_core_lib as L
from atlas_dct_pseudobulk_correlation import VALUE_TYPES, rho_rows

SCP_SETS = {"with_unknown": L.SCP_GROUPS, "core_only": L.CORE_GROUPS}   # primary first
PRIMARY_SET = "with_unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_rna_core step 2: pseudobulk Spearman correlation RNA x control SCP.")
    cc.add_common_args(parser)
    return parser.parse_args()


def heatmap_figure(run, rho: pd.DataFrame, derived: bool, src: str, fig_rows: list[dict]) -> None:
    T.setup_matplotlib()
    cmap = T.diverging_cmap()
    LH = T.text_mm("Hg")[1]
    sub = rho[(rho["level"] == T.LEVEL_SHORT[derived]) & (rho["rna_source"] == src) & (rho["threshold"] == T.DETECTION)
              & (rho["scp_set"] == PRIMARY_SET)]
    rows = [g for g in T.rna_groups(derived) if g in set(sub["rna_group"])]
    cols = list(L.SCP_GROUPS)
    cw, rh = 11.0, 5.0
    lw = max(T.text_mm(r)[0] for r in rows) + 1.2 + 1.6 + 1.0
    pw = lw + len(cols) * cw
    gapx, gapy = 8.0, 4.0
    title = f"Spearman rho, RNA ({T.SOURCE_TEXT[src]}; {T.LEVEL_TEXT[derived]}) x control SCP labels"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    ph_title = LH + 1.0
    col_h = 2 * LH + 1.5                             # label name and n cells on two lines
    panel_h = ph_title + col_h + len(rows) * rh
    n_genes = {(c, vt): int(sub[(sub["label_column"] == c) & (sub["value_type"] == vt)]["n_genes"].iloc[0])
               for c in L.LABEL_COLUMNS for vt in VALUE_TYPES}
    n_scp = {(c, g): int(sub[(sub["label_column"] == c) & (sub["scp_group"] == g)]["n_scp_cells"].iloc[0])
             for c in L.LABEL_COLUMNS for g in cols}
    note = (f"Genes shared by the atlas and the control DCT protein matrix and detected in >= {T.DETECTION:.0%} of at "
            "least one group in each modality (n per panel). RNA: mean log-normalized expression; protein: mean ln(1 + "
            "intensity), missing = 0. z: each gene z-scored across the groups within its modality (SCP: the four rows "
            "including Unknown). Column numbers: SCP cells. " + L.UNPAIRED_NOTE
            + (" " + L.ATLAS_DCT_SPLIT_CAVEAT if derived else ""))
    note_text, note_h = T.note_block(note)
    H = T.M + title_h + 2 * panel_h + gapy + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    for i, vt in enumerate(VALUE_TYPES):
        for j, col in enumerate(L.LABEL_COLUMNS):
            x = T.M + j * (pw + gapx)
            y = T.M + title_h + i * (panel_h + gapy)
            sh.text(x, y, f"{L.LABEL_DISPLAY[col]}, {VALUE_TYPES[vt]} (n = {n_genes[(col, vt)]:,})", va="top")
            x0, y0 = x + lw, y + ph_title + col_h
            p = sub[(sub["label_column"] == col) & (sub["value_type"] == vt)]
            mat = p.pivot(index="rna_group", columns="scp_group", values="rho").reindex(index=rows, columns=cols)
            ax = sh.ax(x0, y0, len(cols) * cw, len(rows) * rh)
            ax.imshow(mat.to_numpy(float), cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
                      extent=(0, len(cols), len(rows), 0))
            ax.set_axis_off()
            for a in range(len(rows)):
                for b in range(len(cols)):
                    v = mat.iloc[a, b]
                    sh.text(x0 + (b + 0.5) * cw, y0 + (a + 0.5) * rh, f"{v:.2f}", ha="center", va="center_baseline",
                            color="white" if abs(v) > 0.6 else T.INK["primary"])
            for b, c in enumerate(cols):
                xc = x0 + (b + 0.5) * cw
                sh.text(xc, y0 - 0.8 - LH - 0.3, c, ha="center", va="bottom", color=L.SCP_PALETTE[c])
                sh.text(xc, y0 - 0.8, f"n = {n_scp[(col, c)]}", ha="center", va="bottom", color=T.INK["secondary"])
            for a, r in enumerate(rows):
                yc = y0 + (a + 0.5) * rh
                sh.rect(x0 - 1.0 - 1.6, yc - 0.8, 1.6, 1.6, facecolor=T.RNA_PALETTE[r], edgecolor="none")
                sh.text(x0 - 1.0 - 1.6 - 1.0, yc, r, ha="right", va="center_baseline")
    cx = T.M + 2 * pw + gapx + 5.0
    cy = T.M + title_h + ph_title + col_h
    cax = sh.ax(cx, cy, 2.4, 22.0)
    cax.imshow(np.linspace(1, -1, 256)[:, None], cmap=cmap, vmin=-1, vmax=1, aspect="auto", extent=(0, 1, -1, 1))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([-1, 0, 1])
    cax.tick_params(length=1.5, width=0.4, labelsize=T.FONT, pad=1)
    sh.text(cx + 2.4 + 5.5, cy + 11.0, "Spearman rho", rotation=90, ha="left", va="center")
    sh.note(note_text, note_h)
    name = f"pseudobulk_rho_{T.LEVEL_SHORT[derived]}_{src}"
    T.save_checked(run, sh, name, title, fig_rows)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "pseudobulk", args)
    L.ported_from(run, "atlas_dct_pseudobulk_correlation.py")
    checks = t5.Checks(L.DATASET)
    adata, labels, ln1p, det = L.load_scp(run, checks)
    region = T.load_region(run, with_split=True)
    sf = L.read_table(run, "shared_features")
    sf = sf[sf["shared"]]
    genes_rna, genes_prot = sf["atlas_gene"].tolist(), sf["protein"].tolist()
    checks.add("shared genes (step 1)", f"{len(sf)}", len(sf) == 5644)
    pos = region.var_names.get_indexer(genes_rna)
    ln = T.lognorm(region.X)[:, pos]
    counts = region.X[:, pos]
    genes_idx = pd.Index(genes_rna)

    rna_stats = {}
    for derived in (False, True):
        grp = region.obs[T.group_column(derived)]
        for src in T.RNA_SOURCES:
            st = T.group_stats(ln, counts, grp, T.rna_groups(derived), genes_idx, T.source_mask(region.obs, src))
            rna_stats[(derived, src)] = st
            checks.add(f"RNA {T.LEVEL_SHORT[derived]} / {src}: cells per group",
                       "; ".join(f"{g} {n}" for g, n in st["n"].items()), None)
    scp_stats = {(col, s): T.scp_group_stats(ln1p[genes_prot], det[genes_prot], labels[col], groups)
                 for col in L.LABEL_COLUMNS for s, groups in SCP_SETS.items()}
    # plate-split SCP groups (CSV only)
    plate_stats = {}
    for col in L.LABEL_COLUMNS:
        lp = labels[col].astype(str) + " " + labels["plate"].astype(str)
        order = [f"{g} {p}" for g in L.SCP_GROUPS for p in ("P1", "P2")]
        plate_stats[col] = T.scp_group_stats(ln1p[genes_prot], det[genes_prot], lp, order)
        checks.add(f"{col}: SCP cells per label x plate", "; ".join(f"{g} {n}" for g, n in plate_stats[col]["n"].items()),
                   None)

    rows, plate_rows, gene_rows = [], [], []
    for (derived, src), st in rna_stats.items():
        for col in L.LABEL_COLUMNS:
            for s in SCP_SETS:
                for thr in T.DETECTION_SENS:
                    meta = {"level": T.LEVEL_SHORT[derived], "rna_source": src, "label_column": col, "scp_set": s}
                    r, gp = rho_rows(st, scp_stats[(col, s)], genes_rna, genes_prot, thr, meta)
                    rows += r
                    if thr == T.DETECTION and src == "all" and s == PRIMARY_SET:
                        gene_rows += [{**meta, "threshold": thr, "protein": g} for g in gp]
                        pr, _ = rho_rows(st, plate_stats[col], genes_rna, genes_prot, thr, meta)
                        plate_rows += pr
    rho = pd.DataFrame(rows)
    checks.add("rho rows (levels x RNA subsets x label columns x SCP sets x thresholds x value types x groups)",
               f"{len(rho)}", bool(rho["rho"].notna().all()))
    for derived in (False, True):
        for col in L.LABEL_COLUMNS:
            for s in SCP_SETS:
                sel = rho[(rho["level"] == T.LEVEL_SHORT[derived]) & (rho["rna_source"] == "all")
                          & (rho["label_column"] == col) & (rho["threshold"] == T.DETECTION) & (rho["scp_set"] == s)]
                for vt in VALUE_TYPES:
                    v = sel[sel["value_type"] == vt]
                    best = v.loc[v.groupby("scp_group", sort=False)["rho"].idxmax()]
                    checks.add(f"{T.LEVEL_SHORT[derived]} / all / {col} / {s} / {vt}: n genes; RNA group with the highest "
                               "rho per SCP label", f"{int(v['n_genes'].iloc[0])}; " + "; ".join(
                                   f"{b.scp_group} -> {b.rna_group} ({b.rho:.2f})" for b in best.itertuples()), None)
    vals = rho[(rho["threshold"] == T.DETECTION) & (rho["scp_set"] == PRIMARY_SET)]
    for vt in VALUE_TYPES:
        v = vals[vals["value_type"] == vt]["rho"]
        checks.add(f"rho range, all levels / RNA subsets / both columns ({vt}, threshold {T.DETECTION:.0%})",
                   f"{v.min():.2f} to {v.max():.2f}", None)

    means = []
    for (derived, src), st in rna_stats.items():
        if src != "all":
            continue
        for g in st["mean"].index:
            means.append(pd.DataFrame({"modality": "RNA", "level": T.LEVEL_SHORT[derived], "group": g,
                                       "n_cells": int(st["n"][g]), "gene": genes_rna, "protein": genes_prot,
                                       "mean": st["mean"].loc[g].to_numpy(), "detection": st["frac"].loc[g].to_numpy()}))
    for (col, s), st in scp_stats.items():
        if s != PRIMARY_SET:
            continue
        for g in st["mean"].index:
            means.append(pd.DataFrame({"modality": "protein", "level": col, "group": g, "n_cells": int(st["n"][g]),
                                       "gene": genes_rna, "protein": genes_prot, "mean": st["mean"].loc[g].to_numpy(),
                                       "detection": st["frac"].loc[g].to_numpy()}))
    means = pd.concat(means, ignore_index=True)

    fig_rows = []
    for derived in (False, True):
        for src in T.RNA_SOURCES:
            heatmap_figure(run, rho, derived, src, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(rho, "pseudobulk_rho")
    run.table(pd.DataFrame(plate_rows), "pseudobulk_rho_plate")
    run.table(pd.DataFrame(gene_rows), "pseudobulk_genes")
    run.table(means, "pseudobulk_means")
    run.table(figs, "pseudobulk_figure_checks")
    run.table(checks.frame(), "pseudobulk_checks")
    run.finish({"thresholds": list(T.DETECTION_SENS), "primary_threshold": T.DETECTION,
                "scp_sets": {k: list(v) for k, v in SCP_SETS.items()}, "primary_scp_set": PRIMARY_SET,
                "rna_value": "mean log-normalized (10,000 per cell, log1p)", "protein_value": "mean ln(1 + intensity), "
                "missing = 0", "z": "per gene across the groups within each modality, ddof 0",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see ctrl_rna_core_pseudobulk_checks.csv.")


if __name__ == "__main__":
    main()
