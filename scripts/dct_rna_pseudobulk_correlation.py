#!/usr/bin/env python
"""dct_rna step 2: pseudobulk Spearman correlation, RNA atlas groups x SCP labels, per condition.

Analysis plan,
step 2 (port of ctrl_rna_core_pseudobulk_correlation.py with a condition argument; ``rho_rows`` and ``VALUE_TYPES`` of
atlas_dct_pseudobulk_correlation imported unchanged).

Groups
------
- RNA (atlas Celltype): published DCT, DCT-CNT, CNT, CTAL, MTAL; and the atlas_dct-derived level DCT1 (atlas_dct), DCT2 (atlas_dct),
  DCT-CNT, CNT, CTAL, MTAL (DCT unassigned (atlas_dct) left out). RNA subsets: all atlas cells (primary), nuclei only,
  cells only (obs 'Source').
- SCP, per condition (Ctrl first): the labels present under the primary and the second column (Unknown its own
  row, as ctrl_rna_core). Sensitivity: core labels only (Unknown left out of the gene selection and the z-scores, as atlas_dct); the
  labels split by sort plate (P1 / P2).

Values (ctrl_rna_core, unchanged)
-----------------------
- RNA: mean log-normalized expression (10,000 per cell, natural log1p); detection = counts > 0.
- Protein: mean ln(1 + intensity) with missing = 0 (dct_labels intensity layer); detection = intensity > 0.
- Genes: the condition's shared genes (step 1) detected >= t in at least one RNA group of the set and >= t in at
  least one SCP group of the label column; t = 10% (primary), 5% and 20%.
- Spearman rho across those genes on (a) the raw means and (b) z-scores of each gene across the groups within its
  modality (ddof 0; zero SD -> 0).

Absent labels: no row for a label without cells. z-scores across a single SCP group are all 0, so z rows of an SCP
set with one group (ZeroK primary, core labels only: DCT1) are 'does not apply' (rho left empty). With two SCP groups
(Ctrl primary core only; ZeroK primary with Unknown) each z-score is -1, 0 or +1 (reported as a coverage limit).

Inputs (read-only): dct_labels labelled objects and canonical labels; atlas_dct region subset and split table; dct_rna step 1.
Outputs: results/tables/dct_rna/dct_rna_pseudobulk_rho.csv, _rho_plate.csv, _genes.csv, _means.csv,
         _checks.csv, _figure_checks.csv;
         results/figures/dct_rna/dct_rna_pseudobulk_rho_{published,derived}_{all,nuclei,cells}_{primary,second}

Command (project root):  python scripts/dct_rna_pseudobulk_correlation.py
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
import warnings

import numpy as np
import pandas as pd

import scp_common as cc
import atlas_dct_lib as T
import dct_rna_lib as L
from atlas_dct_pseudobulk_correlation import VALUE_TYPES, rho_rows

SCP_SETS = ("with_unknown", "core_only")      # primary first (ctrl_rna_core)
PRIMARY_SET = "with_unknown"
ONE_GROUP = "does not apply (one SCP group: every z-score is 0)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_rna step 2: pseudobulk Spearman correlation RNA x SCP, per condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def scp_groups(labels: pd.DataFrame, col: str, scp_set: str) -> list[str]:
    groups = L.present(labels, col)
    return groups if scp_set == "with_unknown" else [g for g in groups if g in L.CORE_GROUPS]


def one_condition(run, cond, region, ln_full, checks):
    s = L.SETTINGS[cond]
    adata, labels, ln1p, det = L.load_scp(run, cond, checks)
    del adata
    sf = L.read_table(run, "shared_features")
    sf = sf[(sf["condition"] == cond) & sf["shared"]]
    genes_rna, genes_prot = sf["atlas_gene"].tolist(), sf["protein"].tolist()
    checks.add("shared genes (step 1)", f"{len(sf)}", len(sf) > 0 and sf["protein"].is_unique, "", cond)
    pos = region.var_names.get_indexer(genes_rna)
    ln = ln_full[:, pos]
    counts = region.X[:, pos]
    genes_idx = pd.Index(genes_rna)

    rna_stats = {}
    for derived in (False, True):
        grp = region.obs[T.group_column(derived)]
        for src in T.RNA_SOURCES:
            st = T.group_stats(ln, counts, grp, T.rna_groups(derived), genes_idx, T.source_mask(region.obs, src))
            rna_stats[(derived, src)] = st
            if cond == "Ctrl":
                checks.add(f"RNA {T.LEVEL_SHORT[derived]} / {src}: cells per group",
                           "; ".join(f"{g} {n}" for g, n in st["n"].items()), None)
    scp_stats, groups_of = {}, {}
    for col in s.LABEL_COLUMNS:
        for ss in SCP_SETS:
            groups = scp_groups(labels, col, ss)
            groups_of[(col, ss)] = groups
            scp_stats[(col, ss)] = T.scp_group_stats(ln1p[genes_prot], det[genes_prot], labels[col], groups)
            checks.add(f"{col} ({L.role_of(cond, col)}) / {ss}: SCP groups (cells)",
                       "; ".join(f"{g} {n}" for g, n in scp_stats[(col, ss)]["n"].items()), None, "", cond)
    plate_stats = {}
    for col in s.LABEL_COLUMNS:
        lp = labels[col].astype(str) + " " + labels["plate"].astype(str)
        order = [f"{g} {p}" for g in L.present(labels, col) for p in ("P1", "P2") if (lp == f"{g} {p}").any()]
        plate_stats[col] = T.scp_group_stats(ln1p[genes_prot], det[genes_prot], lp, order)
        checks.add(f"{col}: SCP cells per label x plate",
                   "; ".join(f"{g} {n}" for g, n in plate_stats[col]["n"].items()), None, "", cond)

    rows, plate_rows, gene_rows = [], [], []
    for (derived, src), st in rna_stats.items():
        for col in s.LABEL_COLUMNS:
            for ss in SCP_SETS:
                for thr in T.DETECTION_SENS:
                    meta = {"level": T.LEVEL_SHORT[derived], "rna_source": src, "label_column": col, "scp_set": ss}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")      # constant z vectors of a one-group set (marked below)
                        r, gp = rho_rows(st, scp_stats[(col, ss)], genes_rna, genes_prot, thr, meta)
                    n_groups = len(groups_of[(col, ss)])
                    for row in r:
                        row["n_scp_groups"] = n_groups
                        if row["value_type"] == "z" and n_groups < 2:
                            row["status"], row["rho"], row["p_value"] = ONE_GROUP, np.nan, np.nan
                        else:
                            row["status"] = "computed"
                    rows += r
                    if thr == T.DETECTION and src == "all" and ss == PRIMARY_SET:
                        gene_rows += [{**meta, "threshold": thr, "protein": g} for g in gp]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pr, _ = rho_rows(st, plate_stats[col], genes_rna, genes_prot, thr, meta)
                        plate_rows += pr
    rho = pd.DataFrame(rows)
    comp = rho["status"] == "computed"
    checks.add("rho rows (levels x RNA subsets x label columns x SCP sets x thresholds x value types x groups); "
               "computed; does not apply", f"{len(rho)}; {int(comp.sum())}; {int((~comp).sum())}",
               bool(rho.loc[comp, "rho"].notna().all()), "rho finite in every computed row", cond)
    for col in s.LABEL_COLUMNS:
        for ss in SCP_SETS:
            if len(groups_of[(col, ss)]) == 2:
                checks.add(f"{col} / {ss}: two SCP groups, so each z-score is -1, 0 or +1 (coverage limit)",
                           ", ".join(groups_of[(col, ss)]), None, "", cond)
    for derived in (False, True):
        for col in s.LABEL_COLUMNS:
            for ss in SCP_SETS:
                sel = rho[(rho["level"] == T.LEVEL_SHORT[derived]) & (rho["rna_source"] == "all")
                          & (rho["label_column"] == col) & (rho["threshold"] == T.DETECTION) & (rho["scp_set"] == ss)
                          & (rho["status"] == "computed")]
                for vt in VALUE_TYPES:
                    v = sel[sel["value_type"] == vt]
                    if v.empty:
                        checks.add(f"{T.LEVEL_SHORT[derived]} / all / {col} / {ss} / {vt}", ONE_GROUP, None, "", cond)
                        continue
                    best = v.loc[v.groupby("scp_group", sort=False)["rho"].idxmax()]
                    checks.add(f"{T.LEVEL_SHORT[derived]} / all / {col} / {ss} / {vt}: n genes; RNA group with the "
                               "highest rho per SCP label", f"{int(v['n_genes'].iloc[0])}; " + "; ".join(
                                   f"{b.scp_group} -> {b.rna_group} ({b.rho:.2f})" for b in best.itertuples()), None,
                               "", cond)
    vals = rho[(rho["threshold"] == T.DETECTION) & (rho["scp_set"] == PRIMARY_SET) & comp]
    for vt in VALUE_TYPES:
        v = vals[vals["value_type"] == vt]["rho"]
        checks.add(f"rho range, all levels / RNA subsets / both columns ({vt}, threshold {T.DETECTION:.0%})",
                   f"{v.min():.2f} to {v.max():.2f}", None, "", cond)

    means = []
    for (derived, src), st in rna_stats.items():
        if src != "all":
            continue
        for g in st["mean"].index:
            means.append(pd.DataFrame({"modality": "RNA", "level": T.LEVEL_SHORT[derived], "group": g,
                                       "n_cells": int(st["n"][g]), "gene": genes_rna, "protein": genes_prot,
                                       "mean": st["mean"].loc[g].to_numpy(), "detection": st["frac"].loc[g].to_numpy()}))
    for (col, ss), st in scp_stats.items():
        if ss != PRIMARY_SET:
            continue
        for g in st["mean"].index:
            means.append(pd.DataFrame({"modality": "protein", "level": col, "group": g, "n_cells": int(st["n"][g]),
                                       "gene": genes_rna, "protein": genes_prot, "mean": st["mean"].loc[g].to_numpy(),
                                       "detection": st["frac"].loc[g].to_numpy()}))
    n_cells = {col: labels[col].astype(str).value_counts().to_dict() for col in s.LABEL_COLUMNS}
    return rho, pd.DataFrame(plate_rows), pd.DataFrame(gene_rows), pd.concat(means, ignore_index=True), n_cells


def heatmap_figure(run, rho: pd.DataFrame, derived: bool, src: str, role: str, n_cells: dict, fig_rows: list[dict]):
    """ctrl_rna_core_pseudobulk_correlation.heatmap_figure with the conditions as panel columns (one label role)."""
    T.setup_matplotlib()
    cmap = T.diverging_cmap()
    LH = T.text_mm("Hg")[1]
    sub = rho[(rho["level"] == T.LEVEL_SHORT[derived]) & (rho["rna_source"] == src) & (rho["threshold"] == T.DETECTION)
              & (rho["scp_set"] == PRIMARY_SET) & (rho["label_role"] == role)]
    rows = [g for g in T.rna_groups(derived) if g in set(sub["rna_group"])]
    cols = list(L.SCP_GROUPS)
    cw, rh = 11.0, 5.0
    lw = max(T.text_mm(r)[0] for r in rows) + 1.2 + 1.6 + 1.0
    pw = lw + len(cols) * cw
    gapx, gapy = 8.0, 4.0
    title = L.fit_title(f"Spearman rho, RNA ({T.SOURCE_TEXT[src]}; {T.LEVEL_TEXT[derived]}) x SCP labels, "
                        f"{role} label columns, Ctrl and ZeroK")
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    ph_title = 2 * LH + 1.0
    col_h = 2 * LH + 1.5
    panel_h = ph_title + col_h + len(rows) * rh
    note = (f"Genes shared by the atlas and the condition's DCT protein matrix and detected in >= {T.DETECTION:.0%} of "
            "at least one group in each modality (n per panel). RNA: mean log-normalized expression; protein: mean ln(1 "
            "+ intensity), missing = 0. z: each gene z-scored across the groups within its modality (SCP: the labels "
            "present, Unknown included). Column numbers: SCP cells; grey columns: label absent under this column (does "
            "not apply). " + L.UNPAIRED_NOTE + " " + L.ZEROK_NOTE + (" " + L.ATLAS_DCT_SPLIT_CAVEAT if derived else ""))
    note_text, note_h = T.note_block(note)
    H = T.M + title_h + 2 * panel_h + gapy + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    for i, vt in enumerate(VALUE_TYPES):
        for j, cond in enumerate(L.CONDITIONS):
            col = L.column_of(cond, role)
            p = sub[(sub["condition"] == cond) & (sub["value_type"] == vt)]
            x = T.M + j * (pw + gapx)
            y = T.M + title_h + i * (panel_h + gapy)
            ng = int(p["n_genes"].iloc[0])
            sh.text(x, y, f"{cond}: {col}\n{VALUE_TYPES[vt]} (n = {ng:,} genes)", va="top", linespacing=1.15)
            x0, y0 = x + lw, y + ph_title + col_h
            mat = p.pivot(index="rna_group", columns="scp_group", values="rho").reindex(index=rows, columns=cols)
            ax = sh.ax(x0, y0, len(cols) * cw, len(rows) * rh)
            ax.imshow(mat.to_numpy(float), cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
                      extent=(0, len(cols), len(rows), 0))
            ax.set_axis_off()
            for b, c in enumerate(cols):
                xc = x0 + (b + 0.5) * cw
                n = int(n_cells[cond][col].get(c, 0))
                if n == 0:
                    for a in range(len(rows)):
                        L.absent_cell(sh, x0 + b * cw, y0 + a * rh, cw, rh, "–" if a == 0 else "")
                else:
                    for a in range(len(rows)):
                        v = mat.iloc[a, b]
                        sh.text(xc, y0 + (a + 0.5) * rh, f"{v:.2f}", ha="center", va="center_baseline",
                                color="white" if abs(v) > 0.6 else T.INK["primary"])
                sh.text(xc, y0 - 0.8 - LH - 0.3, c, ha="center", va="bottom",
                        color=L.SCP_PALETTE[c] if n else T.INK["muted"])
                sh.text(xc, y0 - 0.8, f"n = {n}", ha="center", va="bottom", color=T.INK["secondary"])
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
    name = f"pseudobulk_rho_{T.LEVEL_SHORT[derived]}_{src}_{role}"
    L.save_checked(run, sh, name, title.replace("\n", " "), fig_rows, label_role=role)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "pseudobulk", args)
    L.ported_from(run, "ctrl_rna_core_pseudobulk_correlation.py", "atlas_dct_pseudobulk_correlation.py")
    checks = L.Checks(L.DATASET)
    region = T.load_region(run, with_split=True)
    ln_full = T.lognorm(region.X)
    out = {c: one_condition(run, c, region, ln_full, checks) for c in L.CONDITIONS}
    rho = L.stack({c: out[c][0] for c in L.CONDITIONS})
    n_cells = {c: out[c][4] for c in L.CONDITIONS}
    fig_rows = []
    for derived in (False, True):
        for src in T.RNA_SOURCES:
            for role in L.ROLES:
                heatmap_figure(run, rho, derived, src, role, n_cells, fig_rows)
    figs = L.figure_rows_frame(fig_rows, "pseudobulk")
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(rho, "pseudobulk_rho")
    run.table(L.stack({c: out[c][1] for c in L.CONDITIONS}), "pseudobulk_rho_plate")
    run.table(L.stack({c: out[c][2] for c in L.CONDITIONS}), "pseudobulk_genes")
    run.table(L.stack({c: out[c][3] for c in L.CONDITIONS}), "pseudobulk_means")
    run.table(figs, "pseudobulk_figure_checks")
    run.table(checks.frame(), "pseudobulk_checks")
    run.finish({"thresholds": list(T.DETECTION_SENS), "primary_threshold": T.DETECTION, "scp_sets": list(SCP_SETS),
                "scp_set_rule": "with_unknown = labels present under the column; core_only = present labels other than "
                "Unknown", "primary_scp_set": PRIMARY_SET, "one_group_z": ONE_GROUP,
                "rna_value": "mean log-normalized (10,000 per cell, log1p)", "protein_value": "mean ln(1 + intensity), "
                "missing = 0", "z": "per gene across the groups within each modality, ddof 0", "formats": list(L.FORMATS),
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see dct_rna_pseudobulk_checks.csv.")


if __name__ == "__main__":
    main()
