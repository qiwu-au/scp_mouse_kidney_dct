#!/usr/bin/env python
"""atlas_dct step 6: DCT2-vs-DCT1 fold-change concordance, RNA (atlas_dct-derived split) vs protein (V1 labels).

Analysis plan, step 6.

Per gene (shared genes, step 2)
-------------------------------
- RNA: DCT2 (atlas_dct) minus DCT1 (atlas_dct) mean log-normalized expression (natural log; 10,000 per cell). Primary: all atlas
  cells and nuclei, split at the naming threshold 0.25. Sensitivity: cells only, nuclei only (DCT2 (atlas_dct) holds few
  nuclei), and the split at threshold 0.10 (DCT1 = subclusters named DCT1 at 0.10).
- Protein, per V1 label column: DCT2 minus DCT1 mean ln(1 + intensity) (missing = 0); Mann-Whitney U (BH within
  label column); and a depth- and plate-adjusted difference: OLS ln(1 + intensity) ~ DCT2 + ln(protein IDs) + plate
  on the DCT1 and DCT2 cells (coefficient of DCT2).
- Genes compared: detected in >= 10% of DCT1 or DCT2 cells in each modality (5% and 20% for the primary in the CSV).
- Spearman rho of RNA vs protein difference; fraction of genes with the same sign (two-sided binomial test vs 0.5);
  for all compared genes, without the naming genes, and without every DCT V1 workbook marker.
- Transporters and channels by a symbol rule (CLASS_RULES below; no annotation download): direction concordant if
  the RNA and protein differences have the same sign; |difference| < 0.1 in either modality is marked 'small'.

Inputs (read-only): kidney_object canonical DCT object and labels; atlas_dct region subset, split table, shared features.
Outputs: results/tables/atlas_dct/atlas_dct_fc_genes.csv, atlas_dct_fc_concordance.csv,
         atlas_dct_fc_transporters_channels.csv, atlas_dct_fc_checks.csv, atlas_dct_fc_figure_checks.csv;
         results/figures/atlas_dct/atlas_dct_fc_scatter.pdf/.png

Command (project root):  python scripts/atlas_dct_fold_change.py
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

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc
import cluster_de_lib as t5
import atlas_dct_lib as T

SMALL = 0.10
CLASS_RULES = (   # (class, regex on the mouse symbol); first match wins
    ("SLC transporter", r"^Slc\d"),
    ("ABC transporter", r"^Abc[a-g]\d"),
    ("ATPase pump (P-type / V-type)", r"^Atp(1a|1b|2a|2b|2c|4a|4b|7a|7b|8a|8b|9a|9b|10|11|12a|13a)|^Atp6v"),
    ("ion / water channel", r"^(Kcn|Scn\d|Scnn1|Cacn|Clcn|Clic|Trp[acmv]|Pkd2|Aqp|Ano\d|Best\d|Hcn|Piezo|Asic|Ryr|"
                            r"Itpr|Orai|Tpcn|Mcoln|Lrrc8|Cftr|Gja|Gjb|Hvcn|Tmem63|Ttyh|Vdac|Mcu$)"),
    ("claudin (paracellular)", r"^Cldn\d"),
)
RNA_VARIANTS = {   # name -> (source, split column)
    "all_thr0.25": ("all", "split_label"), "cells_thr0.25": ("cells", "split_label"),
    "nuclei_thr0.25": ("nuclei", "split_label"), "all_thr0.10": ("all", "split_label_thr_0.10"),
}
PRIMARY_RNA = "all_thr0.25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atlas_dct step 6: DCT2-vs-DCT1 fold-change concordance.")
    cc.add_common_args(parser)
    return parser.parse_args()


def gene_class(g: str) -> str:
    return next((c for c, rx in CLASS_RULES if re.search(rx, g)), "")


def ols_effect(Y: np.ndarray, is2: np.ndarray, ln_ids: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(len(is2)), is2, ln_ids, p2]).astype(float)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ Y
    resid = Y - X @ beta
    dof = X.shape[0] - X.shape[1]
    s2 = (resid ** 2).sum(axis=0) / dof
    se = np.sqrt(s2 * xtx_inv[1, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta[1] / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta[1], p


def concord(x: np.ndarray, y: np.ndarray) -> dict:
    r = stats.spearmanr(x, y)
    same = np.sign(x) == np.sign(y)
    nz = (x != 0) & (y != 0)
    b = stats.binomtest(int(same[nz].sum()), int(nz.sum()), 0.5) if nz.sum() else None
    return {"n_genes": len(x), "spearman_rho": float(r.statistic), "spearman_p": float(r.pvalue),
            "fraction_same_sign": float(same[nz].mean()) if nz.sum() else np.nan,
            "binomial_p_same_sign": float(b.pvalue) if b else np.nan}


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "fold_change", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, ln1p, det = T.load_scp(run, checks)
    region = T.load_region(run, with_split=True)
    split = pd.read_csv(run.use_input(T.SPLIT_CSV, "atlas_dct-derived split"), dtype={"atlas_cell": str}).set_index("atlas_cell")
    sf = T.read_table(run, "shared_features")
    sf = sf[sf["shared"]]
    genes_rna, genes_prot = sf["atlas_gene"].to_numpy(), sf["protein"].to_numpy()
    naming = T.naming_lists(run)
    naming_set = {g for v in naming.values() for g in v}
    workbook = T.workbook_markers(run)

    # RNA differences --------------------------------------------------------------------------------------
    pos = region.var_names.get_indexer(genes_rna)
    ln = T.lognorm(region.X)[:, pos]
    cnt = region.X[:, pos]
    dct = (region.obs["rna_group"] == "DCT").to_numpy()
    rna = {}
    for vname, (src, scol) in RNA_VARIANTS.items():
        lab = pd.Series("", index=region.obs_names)
        lab[dct] = split.loc[region.obs_names[dct], scol].astype(str).to_numpy()
        st = T.group_stats(ln, cnt, lab, (T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT), pd.Index(genes_rna), T.source_mask(region.obs, src))
        rna[vname] = st
        checks.add(f"RNA {vname}: cells DCT1 (atlas_dct); DCT2 (atlas_dct)", f"{st['n'][T.DCT1_ATLAS_DCT]}; {st['n'][T.DCT2_ATLAS_DCT]}",
                   None)

    # protein differences ------------------------------------------------------------------------------------
    prot = {}
    ln_ids = np.log(labels["n_detected_proteins"].to_numpy(float))
    p2 = (labels["plate"].astype(str) == "P2").to_numpy(float)
    L = ln1p[genes_prot].to_numpy()
    D = det[genes_prot].to_numpy()
    for col in T.LABEL_COLUMNS:
        lab = labels[col].astype(str).to_numpy()
        m1, m2 = lab == "DCT1", lab == "DCT2"
        diff = L[m2].mean(axis=0) - L[m1].mean(axis=0)
        mw = np.array([stats.mannwhitneyu(L[m2, j], L[m1, j]).pvalue if (L[m2, j].any() or L[m1, j].any())
                       else np.nan for j in range(L.shape[1])])
        mw_q = np.full(len(mw), np.nan)
        ok = np.isfinite(mw)
        mw_q[ok] = stats.false_discovery_control(mw[ok], method="bh")
        keep = m1 | m2
        adj, adj_p = ols_effect(L[keep], m2[keep].astype(float), ln_ids[keep], p2[keep])
        prot[col] = {"diff": diff, "adj": adj, "adj_p": adj_p, "mw_p": mw, "mw_q": mw_q,
                     "det1": D[m1].mean(axis=0), "det2": D[m2].mean(axis=0),
                     "ids": (np.median(labels.loc[m1, "n_detected_proteins"]), np.median(labels.loc[m2, "n_detected_proteins"]))}
        checks.add(f"protein {col}: DCT1 / DCT2 cells; median protein IDs", f"{int(m1.sum())} / {int(m2.sum())}; "
                   f"{prot[col]['ids'][0]:.0f} / {prot[col]['ids'][1]:.0f}", None)

    # per-gene table (primary RNA variant) --------------------------------------------------------------------
    r0 = rna[PRIMARY_RNA]
    genes = pd.DataFrame({"atlas_gene": genes_rna, "protein": genes_prot,
                          "class": [gene_class(g) for g in genes_prot],
                          "naming_gene": [g in naming_set for g in genes_prot],
                          "v1_workbook_marker": [g in workbook for g in genes_prot],
                          "rna_det_DCT1_atlas_dct": r0["frac"].loc[T.DCT1_ATLAS_DCT].to_numpy(),
                          "rna_det_DCT2_atlas_dct": r0["frac"].loc[T.DCT2_ATLAS_DCT].to_numpy(),
                          "rna_diff_DCT2_minus_DCT1": (r0["mean"].loc[T.DCT2_ATLAS_DCT] - r0["mean"].loc[T.DCT1_ATLAS_DCT]).to_numpy()})
    for vname, st in rna.items():
        if vname != PRIMARY_RNA:
            genes[f"rna_diff_{vname}"] = (st["mean"].loc[T.DCT2_ATLAS_DCT] - st["mean"].loc[T.DCT1_ATLAS_DCT]).to_numpy()
    for col in T.LABEL_COLUMNS:
        s = T.LABEL_SHORT[col]
        p = prot[col]
        genes[f"prot_det_DCT1_{s}"] = p["det1"]
        genes[f"prot_det_DCT2_{s}"] = p["det2"]
        genes[f"prot_diff_{s}"] = p["diff"]
        genes[f"prot_mw_q_{s}"] = p["mw_q"]
        genes[f"prot_diff_adjusted_{s}"] = p["adj"]
        genes[f"prot_adjusted_p_{s}"] = p["adj_p"]

    # concordance ------------------------------------------------------------------------------------------
    rows = []
    for vname, st in rna.items():
        rdet = st["frac"].loc[[T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT]].max(axis=0).to_numpy()
        rdiff = (st["mean"].loc[T.DCT2_ATLAS_DCT] - st["mean"].loc[T.DCT1_ATLAS_DCT]).to_numpy()
        for col in T.LABEL_COLUMNS:
            p = prot[col]
            pdet = np.maximum(p["det1"], p["det2"])
            for thr in (T.DETECTION_SENS if vname == PRIMARY_RNA else (T.DETECTION,)):
                both = (rdet >= thr) & (pdet >= thr)
                for gset, gmask in (("all compared genes", np.ones(len(both), bool)),
                                    ("without naming genes", ~genes["naming_gene"].to_numpy()),
                                    ("without V1 workbook markers", ~genes["v1_workbook_marker"].to_numpy()),
                                    ("transporters and channels", genes["class"].ne("").to_numpy())):
                    m = both & gmask
                    for ptype, pv in (("raw", p["diff"]), ("adjusted for depth and plate", p["adj"])):
                        rows.append({"rna_variant": vname, "primary_rna": vname == PRIMARY_RNA, "label_column": col,
                                     "threshold": thr, "gene_set": gset, "protein_difference": ptype,
                                     **concord(rdiff[m], pv[m])})
    conc = pd.DataFrame(rows)
    prim = conc[conc["primary_rna"] & (conc["threshold"] == T.DETECTION)]
    for col in T.LABEL_COLUMNS:
        for ptype in ("raw", "adjusted for depth and plate"):
            s = prim[(prim["label_column"] == col) & (prim["protein_difference"] == ptype)].set_index("gene_set")
            checks.add(f"{col} / protein {ptype}: rho (n; fraction same sign) for all / without naming / without "
                       "workbook / transporters+channels",
                       " | ".join(f"{s.loc[g, 'spearman_rho']:.3f} ({int(s.loc[g, 'n_genes'])}; "
                                  f"{s.loc[g, 'fraction_same_sign']:.2f})" for g in s.index), None)

    # transporters and channels -------------------------------------------------------------------------------
    rdet0 = np.maximum(genes["rna_det_DCT1_atlas_dct"], genes["rna_det_DCT2_atlas_dct"])
    tc_rows = []
    for col in T.LABEL_COLUMNS:
        s = T.LABEL_SHORT[col]
        pdet = np.maximum(genes[f"prot_det_DCT1_{s}"], genes[f"prot_det_DCT2_{s}"])
        m = genes["class"].ne("") & (rdet0 >= T.DETECTION) & (pdet >= T.DETECTION)
        for _, g in genes[m].iterrows():
            rd, pdf, pa = g["rna_diff_DCT2_minus_DCT1"], g[f"prot_diff_{s}"], g[f"prot_diff_adjusted_{s}"]
            small = abs(rd) < SMALL or abs(pdf) < SMALL
            tc_rows.append({"label_column": col, "protein": g["protein"], "class": g["class"],
                            "naming_gene": g["naming_gene"], "v1_workbook_marker": g["v1_workbook_marker"],
                            "rna_diff_DCT2_minus_DCT1": rd, "prot_diff": pdf, "prot_mw_q": g[f"prot_mw_q_{s}"],
                            "prot_diff_adjusted": pa, "prot_adjusted_p": g[f"prot_adjusted_p_{s}"],
                            "direction_raw": "concordant" if np.sign(rd) == np.sign(pdf) else "discordant",
                            "direction_adjusted": "concordant" if np.sign(rd) == np.sign(pa) else "discordant",
                            "small_difference": small,
                            "rna_det_DCT1_atlas_dct": g["rna_det_DCT1_atlas_dct"], "rna_det_DCT2_atlas_dct": g["rna_det_DCT2_atlas_dct"],
                            "prot_det_DCT1": g[f"prot_det_DCT1_{s}"], "prot_det_DCT2": g[f"prot_det_DCT2_{s}"]})
    tc = pd.DataFrame(tc_rows).sort_values(["label_column", "direction_raw", "class", "protein"])
    for col in T.LABEL_COLUMNS:
        s = tc[tc["label_column"] == col]
        nb = s[~s["small_difference"]]
        checks.add(f"{col}: transporters / channels compared; concordant / discordant (raw); of these with |diff| >= "
                   f"{SMALL} in both", f"{len(s)}; {int((s['direction_raw'] == 'concordant').sum())} / "
                   f"{int((s['direction_raw'] == 'discordant').sum())}; {int((nb['direction_raw'] == 'concordant').sum())}"
                   f" / {int((nb['direction_raw'] == 'discordant').sum())}", None)

    fig_rows = []
    scatter(run, genes, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(genes, "fc_genes")
    run.table(conc, "fc_concordance")
    run.table(tc, "fc_transporters_channels")
    run.table(pd.DataFrame([{"class": c, "regex": rx} for c, rx in CLASS_RULES]), "fc_class_rules")
    run.table(figs, "fc_figure_checks")
    run.table(checks.frame(), "fc_checks")
    run.finish({"rna_variants": RNA_VARIANTS, "primary_rna_variant": PRIMARY_RNA, "detection": T.DETECTION,
                "small_difference": SMALL, "class_rules": dict(CLASS_RULES),
                "adjusted_model": "OLS ln1p ~ DCT2 + ln(protein IDs) + plate (DCT1 and DCT2 cells)",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see atlas_dct_fc_checks.csv.")


def place_labels(sh, ax, x0, y0, w, h, pts, xlim, ylim):
    """Greedy label placement in mm: candidate offsets around each point; skip overlaps with placed labels."""
    placed = []
    LH = T.text_mm("Hg")[1]
    for name, px, py in pts:
        mx = x0 + (px - xlim[0]) / (xlim[1] - xlim[0]) * w
        my = y0 + (ylim[1] - py) / (ylim[1] - ylim[0]) * h
        tw = T.text_mm(name)[0]
        for dx, dy in ((1.2, 0), (-1.2 - tw, 0), (1.0, -LH), (1.0, LH), (-1.0 - tw, -LH), (-1.0 - tw, LH),
                       (1.0, -2 * LH), (1.0, 2 * LH), (-1.0 - tw, -2 * LH), (-1.0 - tw, 2 * LH),
                       (-tw / 2, 1.5 * LH), (-tw / 2, -1.5 * LH), (-tw, 2.5 * LH), (-tw, -2.5 * LH)):
            bx, by = mx + dx, my + dy - LH / 2
            box = (bx - 0.3, by - 0.3, bx + tw + 0.3, by + LH + 0.3)
            inside = box[0] >= x0 and box[2] <= x0 + w and box[1] >= y0 and box[3] <= y0 + h
            if inside and not any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
                placed.append(box)
                sh.text(bx, by + LH / 2, name, va="center_baseline", color=T.INK["primary"])
                break


def scatter(run, genes: pd.DataFrame, fig_rows: list[dict]) -> None:
    plt = T.setup_matplotlib()
    title = "DCT2 minus DCT1: RNA (atlas_dct-derived split) vs protein (V1 labels), per gene"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    LH = T.text_mm("Hg")[1]
    pw, ph, gx, gy = 68.0, 52.0, 22.0, 14.0
    note = (f"Genes detected in >= {T.DETECTION:.0%} of DCT1 or DCT2 cells in each modality. RNA: difference of mean "
            "log-normalized expression, all atlas cells and nuclei, DCT2 (atlas_dct) - DCT1 (atlas_dct) (atlas_dct-derived split, not atlas "
            "labels). Protein: difference of mean ln(1 + intensity), missing = 0; adjusted: OLS coefficient with ln(protein "
            "IDs) and plate. Orange: transporters and channels (symbol rule); blue, labelled: naming genes of the split. "
            "Unpaired comparison: different animals and studies.")
    note_text, note_h = T.note_block(note)
    hh = LH + 3.0                                  # panel header + gap above the axes (top tick label)
    H = T.M + title_h + 2 * (hh + ph) + gy + 8.0 + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    rdet = np.maximum(genes["rna_det_DCT1_atlas_dct"], genes["rna_det_DCT2_atlas_dct"])
    for i, col in enumerate(T.LABEL_COLUMNS):
        s = T.LABEL_SHORT[col]
        pdet = np.maximum(genes[f"prot_det_DCT1_{s}"], genes[f"prot_det_DCT2_{s}"])
        g = genes[(rdet >= T.DETECTION) & (pdet >= T.DETECTION)]
        for j, (ycol, ylab) in enumerate(((f"prot_diff_{s}", "Protein difference"),
                                          (f"prot_diff_adjusted_{s}", "Protein difference, adjusted"))):
            x0 = T.M + 14.0 + j * (pw + gx)
            y0 = T.M + title_h + i * (hh + ph + gy) + hh
            x, y = g["rna_diff_DCT2_minus_DCT1"].to_numpy(), g[ycol].to_numpy()
            r = stats.spearmanr(x, y).statistic
            sh.text(x0 - 12.0, y0 - hh, f"{T.LABEL_DISPLAY[col]}: rho = {r:.2f} (n = {len(g):,})", va="top")
            ax = sh.ax(x0, y0, pw, ph)
            base = g["class"].eq("") & ~g["naming_gene"]
            ax.scatter(x[base], y[base], s=2.5, color=T.INK["background_cells"], linewidths=0, rasterized=True)
            tcm = g["class"].ne("") & ~g["naming_gene"]
            ax.scatter(x[tcm], y[tcm], s=5, color=T.SERIES[1], linewidths=0)
            nm = g["naming_gene"].to_numpy()
            ax.scatter(x[nm], y[nm], s=9, color=T.SERIES[0], linewidths=0)
            xl = np.nanmax(np.abs(x)) * 1.12
            yl = np.nanmax(np.abs(y)) * 1.12
            ax.set_xlim(-xl, xl)
            ax.set_ylim(-yl, yl)
            ax.axhline(0, color=T.INK["muted"], lw=0.4)
            ax.axvline(0, color=T.INK["muted"], lw=0.4)
            ax.set_xlabel("RNA difference (DCT2 (atlas_dct) - DCT1 (atlas_dct))", labelpad=1)
            ax.set_ylabel(ylab, labelpad=1)
            T.style_axes(ax)
            T.trim_ticks(ax)
            pts = [(p, a, b) for p, a, b in zip(g.loc[nm, "protein"], x[nm], y[nm])]
            place_labels(sh, ax, x0, y0, pw, ph, pts, (-xl, xl), (-yl, yl))
    sh.note(note_text, note_h)
    T.save_checked(run, sh, "fc_scatter", title, fig_rows)


if __name__ == "__main__":
    main()
