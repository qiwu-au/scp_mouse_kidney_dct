#!/usr/bin/env python
"""ctrl_rna_core step 5: DCT2-vs-DCT1 fold-change concordance, RNA (atlas_dct-derived split) vs control protein labels.

Analysis plan,
step 5 (port of atlas_dct_fold_change.py; ``gene_class``, ``ols_effect``, ``concord``, ``place_labels``, the class
rules and the RNA variants imported unchanged).

Per gene (shared genes, step 1)
-------------------------------
- RNA: DCT2 (atlas_dct) minus DCT1 (atlas_dct) mean log-normalized expression (natural log; 10,000 per cell). Primary: all atlas
  cells and nuclei, split at the naming threshold 0.25. Sensitivity: cells only, nuclei only (DCT2 (atlas_dct) holds 39
  nuclei), and the split at threshold 0.10.
- Protein, per label column (ctrl_deg_label, ctrl_cluster_label; both have DCT1 and DCT2): DCT2 minus DCT1 mean
  ln(1 + intensity) (missing = 0); Mann-Whitney U (BH within label column); and a depth- and plate-adjusted
  difference: OLS ln(1 + intensity) ~ DCT2 + ln(protein IDs) + plate on the DCT1 and DCT2 cells (coefficient of
  DCT2). ProLIF and Unknown cells are not used.
- Genes compared: detected in >= 10% of DCT1 or DCT2 cells in each modality (5% and 20% for the primary in the CSV).
- Spearman rho of RNA vs protein difference; fraction of genes with the same sign (two-sided binomial test vs 0.5);
  for all compared genes, without the naming genes of the split, and without every DCT V1 workbook marker.
- Transporters and channels by the atlas_dct symbol rule (no annotation download): direction concordant if the RNA and
  protein differences have the same sign; |difference| < 0.1 in either modality is marked 'small'.

Inputs (read-only): labels_core labelled control object and canonical labels; atlas_dct region subset and split table; DCT V1
                    marker workbook; ctrl_rna_core step 1.
Outputs: results/tables/ctrl_rna_core/ctrl_rna_core_fc_genes.csv, ctrl_rna_core_fc_concordance.csv,
         ctrl_rna_core_fc_transporters_channels.csv, ctrl_rna_core_fc_class_rules.csv, ctrl_rna_core_fc_checks.csv,
         ctrl_rna_core_fc_figure_checks.csv; results/figures/ctrl_rna_core/ctrl_rna_core_fc_scatter.pdf/.png

Command (project root):  python scripts/ctrl_rna_core_fold_change.py
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
from scipy import stats

import scp_common as cc
import cluster_de_lib as t5
import atlas_dct_lib as T
import ctrl_rna_core_lib as L
from atlas_dct_fold_change import (CLASS_RULES, PRIMARY_RNA, RNA_VARIANTS, SMALL, concord, gene_class, ols_effect,
                                    place_labels)

ADJUSTED = "adjusted for depth and plate"
GENE_SETS = ("all compared genes", "without naming genes", "without DCT V1 workbook markers", "transporters and channels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_rna_core step 5: DCT2-vs-DCT1 fold-change concordance (control data).")
    cc.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "fold_change", args)
    L.ported_from(run, "atlas_dct_fold_change.py")
    checks = t5.Checks(L.DATASET)
    adata, labels, ln1p, det = L.load_scp(run, checks)
    region = T.load_region(run, with_split=True)
    split = pd.read_csv(run.use_input(T.SPLIT_CSV, "atlas_dct-derived split"), dtype={"atlas_cell": str}).set_index("atlas_cell")
    sf = L.read_table(run, "shared_features")
    sf = sf[sf["shared"]]
    genes_rna, genes_prot = sf["atlas_gene"].to_numpy(), sf["protein"].to_numpy()
    naming = T.naming_lists(run, checks)
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
    Lm = ln1p[genes_prot].to_numpy()
    D = det[genes_prot].to_numpy()
    for col in L.LABEL_COLUMNS:
        lab = labels[col].astype(str).to_numpy()
        m1, m2 = lab == "DCT1", lab == "DCT2"
        diff = Lm[m2].mean(axis=0) - Lm[m1].mean(axis=0)
        mw = np.array([stats.mannwhitneyu(Lm[m2, j], Lm[m1, j]).pvalue if (Lm[m2, j].any() or Lm[m1, j].any())
                       else np.nan for j in range(Lm.shape[1])])
        mw_q = np.full(len(mw), np.nan)
        ok = np.isfinite(mw)
        mw_q[ok] = stats.false_discovery_control(mw[ok], method="bh")
        keep = m1 | m2
        adj, adj_p = ols_effect(Lm[keep], m2[keep].astype(float), ln_ids[keep], p2[keep])
        ids = (np.median(labels.loc[m1, "n_detected_proteins"]), np.median(labels.loc[m2, "n_detected_proteins"]))
        prot[col] = {"diff": diff, "adj": adj, "adj_p": adj_p, "mw_p": mw, "mw_q": mw_q,
                     "det1": D[m1].mean(axis=0), "det2": D[m2].mean(axis=0), "ids": ids}
        checks.add(f"protein {col}: DCT1 / DCT2 cells; median protein IDs; P2 share", f"{int(m1.sum())} / "
                   f"{int(m2.sum())}; {ids[0]:.0f} / {ids[1]:.0f}; {p2[m1].mean():.2f} / {p2[m2].mean():.2f}", None)

    # per-gene table (primary RNA variant) --------------------------------------------------------------------
    r0 = rna[PRIMARY_RNA]
    genes = pd.DataFrame({"atlas_gene": genes_rna, "protein": genes_prot,
                          "class": [gene_class(g) for g in genes_prot],
                          "naming_gene": [g in naming_set for g in genes_prot],
                          "workbook_marker": [g in workbook for g in genes_prot],
                          "rna_det_DCT1_atlas_dct": r0["frac"].loc[T.DCT1_ATLAS_DCT].to_numpy(),
                          "rna_det_DCT2_atlas_dct": r0["frac"].loc[T.DCT2_ATLAS_DCT].to_numpy(),
                          "rna_diff_DCT2_minus_DCT1": (r0["mean"].loc[T.DCT2_ATLAS_DCT] - r0["mean"].loc[T.DCT1_ATLAS_DCT]).to_numpy()})
    for vname, st in rna.items():
        if vname != PRIMARY_RNA:
            genes[f"rna_diff_{vname}"] = (st["mean"].loc[T.DCT2_ATLAS_DCT] - st["mean"].loc[T.DCT1_ATLAS_DCT]).to_numpy()
    for col in L.LABEL_COLUMNS:
        s = L.LABEL_SHORT[col]
        p = prot[col]
        genes[f"prot_det_DCT1_{s}"] = p["det1"]
        genes[f"prot_det_DCT2_{s}"] = p["det2"]
        genes[f"prot_diff_{s}"] = p["diff"]
        genes[f"prot_mw_q_{s}"] = p["mw_q"]
        genes[f"prot_diff_adjusted_{s}"] = p["adj"]
        genes[f"prot_adjusted_p_{s}"] = p["adj_p"]
    checks.add("naming genes and DCT V1 workbook markers among the shared genes",
               f"{', '.join(genes.loc[genes['naming_gene'], 'protein'])}; "
               f"{', '.join(genes.loc[genes['workbook_marker'], 'protein'])}", None)

    # concordance ------------------------------------------------------------------------------------------
    rows = []
    for vname, st in rna.items():
        rdet = st["frac"].loc[[T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT]].max(axis=0).to_numpy()
        rdiff = (st["mean"].loc[T.DCT2_ATLAS_DCT] - st["mean"].loc[T.DCT1_ATLAS_DCT]).to_numpy()
        for col in L.LABEL_COLUMNS:
            p = prot[col]
            pdet = np.maximum(p["det1"], p["det2"])
            for thr in (T.DETECTION_SENS if vname == PRIMARY_RNA else (T.DETECTION,)):
                both = (rdet >= thr) & (pdet >= thr)
                masks = (np.ones(len(both), bool), ~genes["naming_gene"].to_numpy(),
                         ~genes["workbook_marker"].to_numpy(), genes["class"].ne("").to_numpy())
                for gset, gmask in zip(GENE_SETS, masks):
                    m = both & gmask
                    for ptype, pv in (("raw", p["diff"]), (ADJUSTED, p["adj"])):
                        rows.append({"rna_variant": vname, "primary_rna": vname == PRIMARY_RNA, "label_column": col,
                                     "threshold": thr, "gene_set": gset, "protein_difference": ptype,
                                     **concord(rdiff[m], pv[m])})
    conc = pd.DataFrame(rows)
    prim = conc[conc["primary_rna"] & (conc["threshold"] == T.DETECTION)]
    for col in L.LABEL_COLUMNS:
        for ptype in ("raw", ADJUSTED):
            s = prim[(prim["label_column"] == col) & (prim["protein_difference"] == ptype)].set_index("gene_set")
            checks.add(f"{col} / protein {ptype}: rho (n; fraction same sign) for all / without naming / without "
                       "workbook / transporters+channels",
                       " | ".join(f"{s.loc[g, 'spearman_rho']:.3f} ({int(s.loc[g, 'n_genes'])}; "
                                  f"{s.loc[g, 'fraction_same_sign']:.2f})" for g in s.index), None)

    # transporters and channels -------------------------------------------------------------------------------
    rdet0 = np.maximum(genes["rna_det_DCT1_atlas_dct"], genes["rna_det_DCT2_atlas_dct"])
    tc_rows = []
    for col in L.LABEL_COLUMNS:
        s = L.LABEL_SHORT[col]
        pdet = np.maximum(genes[f"prot_det_DCT1_{s}"], genes[f"prot_det_DCT2_{s}"])
        m = genes["class"].ne("") & (rdet0 >= T.DETECTION) & (pdet >= T.DETECTION)
        for _, g in genes[m].iterrows():
            rd, pdf, pa = g["rna_diff_DCT2_minus_DCT1"], g[f"prot_diff_{s}"], g[f"prot_diff_adjusted_{s}"]
            small = abs(rd) < SMALL or abs(pdf) < SMALL
            tc_rows.append({"label_column": col, "protein": g["protein"], "class": g["class"],
                            "naming_gene": g["naming_gene"], "workbook_marker": g["workbook_marker"],
                            "rna_diff_DCT2_minus_DCT1": rd, "prot_diff": pdf, "prot_mw_q": g[f"prot_mw_q_{s}"],
                            "prot_diff_adjusted": pa, "prot_adjusted_p": g[f"prot_adjusted_p_{s}"],
                            "direction_raw": "concordant" if np.sign(rd) == np.sign(pdf) else "discordant",
                            "direction_adjusted": "concordant" if np.sign(rd) == np.sign(pa) else "discordant",
                            "small_difference": small,
                            "rna_det_DCT1_atlas_dct": g["rna_det_DCT1_atlas_dct"], "rna_det_DCT2_atlas_dct": g["rna_det_DCT2_atlas_dct"],
                            "prot_det_DCT1": g[f"prot_det_DCT1_{s}"], "prot_det_DCT2": g[f"prot_det_DCT2_{s}"]})
    tc = pd.DataFrame(tc_rows).sort_values(["label_column", "direction_raw", "class", "protein"])
    for col in L.LABEL_COLUMNS:
        s = tc[tc["label_column"] == col]
        nb = s[~s["small_difference"]]
        checks.add(f"{col}: transporters / channels compared; concordant / discordant (raw); of these with |diff| >= "
                   f"{SMALL} in both", f"{len(s)}; {int((s['direction_raw'] == 'concordant').sum())} / "
                   f"{int((s['direction_raw'] == 'discordant').sum())}; {int((nb['direction_raw'] == 'concordant').sum())}"
                   f" / {int((nb['direction_raw'] == 'discordant').sum())}", None)
        for d in ("concordant", "discordant"):
            x = nb[nb["direction_raw"] == d]
            checks.add(f"{col}: {d} (raw), |diff| >= {SMALL} in both (RNA; protein raw; protein adjusted)",
                       "; ".join(f"{r.protein} ({r.rna_diff_DCT2_minus_DCT1:+.2f}; {r.prot_diff:+.2f}; "
                                 f"{r.prot_diff_adjusted:+.2f})" for r in x.itertuples()) or "none", None)

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
                "label_columns": list(L.LABEL_COLUMNS), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see ctrl_rna_core_fc_checks.csv.")


def scatter(run, genes: pd.DataFrame, fig_rows: list[dict]) -> None:
    """atlas_dct_fold_change.scatter with the ctrl_rna_core label columns."""
    T.setup_matplotlib()
    title = "DCT2 minus DCT1: RNA (atlas_dct-derived split) vs control protein labels, per gene"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    LH = T.text_mm("Hg")[1]
    pw, ph, gx, gy = 68.0, 52.0, 22.0, 14.0
    note = (f"Genes detected in >= {T.DETECTION:.0%} of DCT1 or DCT2 cells in each modality. RNA: difference of mean "
            "log-normalized expression, all atlas cells and nuclei, DCT2 (atlas_dct) - DCT1 (atlas_dct). Protein: difference of mean "
            "ln(1 + intensity), missing = 0; adjusted: OLS coefficient with ln(protein IDs) and plate. Orange: "
            "transporters and channels (symbol rule); blue, labelled: naming genes of the split. "
            + L.ATLAS_DCT_SPLIT_CAVEAT + " " + L.UNPAIRED_NOTE)
    note_text, note_h = T.note_block(note)
    hh = LH + 3.0
    H = T.M + title_h + 2 * (hh + ph) + gy + 8.0 + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    rdet = np.maximum(genes["rna_det_DCT1_atlas_dct"], genes["rna_det_DCT2_atlas_dct"])
    for i, col in enumerate(L.LABEL_COLUMNS):
        s = L.LABEL_SHORT[col]
        pdet = np.maximum(genes[f"prot_det_DCT1_{s}"], genes[f"prot_det_DCT2_{s}"])
        g = genes[(rdet >= T.DETECTION) & (pdet >= T.DETECTION)]
        for j, (ycol, ylab) in enumerate(((f"prot_diff_{s}", "Protein difference"),
                                          (f"prot_diff_adjusted_{s}", "Protein difference, adjusted"))):
            x0 = T.M + 14.0 + j * (pw + gx)
            y0 = T.M + title_h + i * (hh + ph + gy) + hh
            x, y = g["rna_diff_DCT2_minus_DCT1"].to_numpy(), g[ycol].to_numpy()
            r = stats.spearmanr(x, y).statistic
            sh.text(x0 - 12.0, y0 - hh, f"{L.LABEL_DISPLAY[col]}: rho = {r:.2f} (n = {len(g):,})", va="top")
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
