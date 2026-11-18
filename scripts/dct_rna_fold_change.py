#!/usr/bin/env python
"""dct_rna step 5: DCT2-vs-DCT1 fold-change concordance, RNA (atlas_dct-derived split) vs protein, per condition.

Analysis plan,
step 5 (port of ctrl_rna_core_fold_change.py with a condition argument; ``CLASS_RULES``, ``PRIMARY_RNA``,
``RNA_VARIANTS``, ``SMALL``, ``concord``, ``gene_class`` and ``ols_effect`` of atlas_dct_fold_change imported
unchanged; ``place_labels`` is the atlas_dct function plus a leader line for labels placed away from their point).

Method (ctrl_rna_core, unchanged)
-----------------------
- RNA: DCT2 (atlas_dct) - DCT1 (atlas_dct) difference of mean log-normalized expression (all atlas cells; primary split; RNA
  variants as atlas_dct / ctrl_rna_core).
- Protein: DCT2 - DCT1 difference of mean ln(1 + intensity), missing = 0, and the coefficient of DCT2 in OLS
  ln(1 + intensity) ~ DCT2 + ln(protein IDs) + plate on the DCT1 and DCT2 cells of the column.
- Genes: the condition's shared genes detected in >= 10% of DCT1 or DCT2 cells in each modality (5% / 20% in the CSV).
- Gene sets: all; without the naming genes of the split; without DCT V1 workbook markers; transporters and channels.

Applicability: the step needs DCT1 and DCT2 cells under the column. It runs for cs_ctrl_deg_label (primary),
cs_ctrl_cluster_label (Ctrl second) and cs_zerok_cluster_label (ZeroK second). cs_zerok_pipeline_label (ZeroK primary)
has no DCT2: its rows are 'does not apply' and the ZeroK second column's result is reported beside them.

Inputs (read-only): dct_labels labelled objects and canonical labels; atlas_dct region subset, split table and naming lists; DCT
                    V1 marker workbook; dct_rna step 1.
Outputs: results/tables/dct_rna/dct_rna_fc_genes.csv, _concordance.csv, _transporters_channels.csv,
         _class_rules.csv, _checks.csv, _figure_checks.csv;
         results/figures/dct_rna/dct_rna_fc_scatter_{ctrl,zerok} (.pdf/.png/.tif)

Command (project root):  python scripts/dct_rna_fold_change.py
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
import atlas_dct_lib as T
import dct_rna_lib as L
from atlas_dct_fold_change import CLASS_RULES, PRIMARY_RNA, RNA_VARIANTS, SMALL, concord, gene_class, ols_effect
from matplotlib.lines import Line2D
from ctrl_rna_core_fold_change import ADJUSTED, GENE_SETS

NA_REASON = "no DCT2 cells under this column (dct_labels); see the second column of the same condition"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_rna step 5: DCT2-vs-DCT1 fold-change concordance, per condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def applicable(labels: pd.DataFrame, col: str) -> bool:
    lab = labels[col].astype(str)
    return bool((lab == "DCT1").any() and (lab == "DCT2").any())


def one_condition(run, cond, region, split, ln_full, naming_set, workbook, checks):
    s = L.SETTINGS[cond]
    adata, labels, ln1p, det = L.load_scp(run, cond, checks)
    del adata
    sf = L.read_table(run, "shared_features")
    sf = sf[(sf["condition"] == cond) & sf["shared"]]
    genes_rna, genes_prot = sf["atlas_gene"].to_numpy(), sf["protein"].to_numpy()

    pos = region.var_names.get_indexer(genes_rna)
    ln = ln_full[:, pos]
    cnt = region.X[:, pos]
    dct = (region.obs["rna_group"] == "DCT").to_numpy()
    rna = {}
    for vname, (src, scol) in RNA_VARIANTS.items():
        lab = pd.Series("", index=region.obs_names)
        lab[dct] = split.loc[region.obs_names[dct], scol].astype(str).to_numpy()
        st = T.group_stats(ln, cnt, lab, (T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT), pd.Index(genes_rna), T.source_mask(region.obs, src))
        rna[vname] = st
        if cond == "Ctrl":
            checks.add(f"RNA {vname}: cells DCT1 (atlas_dct); DCT2 (atlas_dct)", f"{st['n'][T.DCT1_ATLAS_DCT]}; {st['n'][T.DCT2_ATLAS_DCT]}",
                       None)

    prot, cols = {}, [c for c in s.LABEL_COLUMNS if applicable(labels, c)]
    ln_ids = np.log(labels["n_detected_proteins"].to_numpy(float))
    p2 = (labels["plate"].astype(str) == "P2").to_numpy(float)
    Lm = ln1p[genes_prot].to_numpy()
    D = det[genes_prot].to_numpy()
    for col in s.LABEL_COLUMNS:
        if col not in cols:
            checks.add(f"protein {col} ({L.role_of(cond, col)}): DCT1 / DCT2 cells", f"{L.counts_text(labels[col])}; "
                       f"{L.NOT_APPLICABLE}", None, "", cond)
            continue
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
        checks.add(f"protein {col} ({L.role_of(cond, col)}): DCT1 / DCT2 cells; median protein IDs; P2 share",
                   f"{int(m1.sum())} / {int(m2.sum())}; {ids[0]:.0f} / {ids[1]:.0f}; {p2[m1].mean():.2f} / "
                   f"{p2[m2].mean():.2f}", None, "", cond)

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
    for col in s.LABEL_COLUMNS:
        r = L.role_of(cond, col)
        p = prot.get(col)
        for key, out in (("det1", "prot_det_DCT1"), ("det2", "prot_det_DCT2"), ("diff", "prot_diff"),
                         ("mw_q", "prot_mw_q"), ("adj", "prot_diff_adjusted"), ("adj_p", "prot_adjusted_p")):
            genes[f"{out}_{r}"] = p[key] if p is not None else np.nan
    checks.add("naming genes and DCT V1 workbook markers among the shared genes",
               f"{', '.join(genes.loc[genes['naming_gene'], 'protein'])}; "
               f"{', '.join(genes.loc[genes['workbook_marker'], 'protein'])}", None, "", cond)

    rows, keys = [], None
    for vname, st in rna.items():
        rdet = st["frac"].loc[[T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT]].max(axis=0).to_numpy()
        rdiff = (st["mean"].loc[T.DCT2_ATLAS_DCT] - st["mean"].loc[T.DCT1_ATLAS_DCT]).to_numpy()
        for col in cols:
            p = prot[col]
            pdet = np.maximum(p["det1"], p["det2"])
            for thr in (T.DETECTION_SENS if vname == PRIMARY_RNA else (T.DETECTION,)):
                both = (rdet >= thr) & (pdet >= thr)
                masks = (np.ones(len(both), bool), ~genes["naming_gene"].to_numpy(),
                         ~genes["workbook_marker"].to_numpy(), genes["class"].ne("").to_numpy())
                for gset, gmask in zip(GENE_SETS, masks):
                    m = both & gmask
                    for ptype, pv in (("raw", p["diff"]), (ADJUSTED, p["adj"])):
                        res = concord(rdiff[m], pv[m])
                        keys = keys or list(res)
                        rows.append({"rna_variant": vname, "primary_rna": vname == PRIMARY_RNA, "label_column": col,
                                     "threshold": thr, "gene_set": gset, "protein_difference": ptype,
                                     "status": "computed", **res})
    for col in s.LABEL_COLUMNS:
        if col in cols:
            continue
        for thr in T.DETECTION_SENS:
            for gset in GENE_SETS:
                for ptype in ("raw", ADJUSTED):
                    rows.append({"rna_variant": PRIMARY_RNA, "primary_rna": True, "label_column": col, "threshold": thr,
                                 "gene_set": gset, "protein_difference": ptype, "status": L.NOT_APPLICABLE,
                                 "reason": NA_REASON})
    conc = pd.DataFrame(rows)
    prim = conc[conc["primary_rna"] & (conc["threshold"] == T.DETECTION) & (conc["status"] == "computed")]
    for col in cols:
        for ptype in ("raw", ADJUSTED):
            x = prim[(prim["label_column"] == col) & (prim["protein_difference"] == ptype)].set_index("gene_set")
            checks.add(f"{col} / protein {ptype}: rho (n; fraction same sign) for all / without naming / without "
                       "workbook / transporters+channels",
                       " | ".join(f"{x.loc[g, 'spearman_rho']:.3f} ({int(x.loc[g, 'n_genes'])}; "
                                  f"{x.loc[g, 'fraction_same_sign']:.2f})" for g in x.index), None, "", cond)

    rdet0 = np.maximum(genes["rna_det_DCT1_atlas_dct"], genes["rna_det_DCT2_atlas_dct"])
    tc_rows = []
    for col in cols:
        r = L.role_of(cond, col)
        pdet = np.maximum(genes[f"prot_det_DCT1_{r}"], genes[f"prot_det_DCT2_{r}"])
        m = genes["class"].ne("") & (rdet0 >= T.DETECTION) & (pdet >= T.DETECTION)
        for _, g in genes[m].iterrows():
            rd, pdf, pa = g["rna_diff_DCT2_minus_DCT1"], g[f"prot_diff_{r}"], g[f"prot_diff_adjusted_{r}"]
            small = abs(rd) < SMALL or abs(pdf) < SMALL
            tc_rows.append({"label_column": col, "protein": g["protein"], "class": g["class"],
                            "naming_gene": g["naming_gene"], "workbook_marker": g["workbook_marker"],
                            "rna_diff_DCT2_minus_DCT1": rd, "prot_diff": pdf, "prot_mw_q": g[f"prot_mw_q_{r}"],
                            "prot_diff_adjusted": pa, "prot_adjusted_p": g[f"prot_adjusted_p_{r}"],
                            "direction_raw": "concordant" if np.sign(rd) == np.sign(pdf) else "discordant",
                            "direction_adjusted": "concordant" if np.sign(rd) == np.sign(pa) else "discordant",
                            "small_difference": small,
                            "rna_det_DCT1_atlas_dct": g["rna_det_DCT1_atlas_dct"], "rna_det_DCT2_atlas_dct": g["rna_det_DCT2_atlas_dct"],
                            "prot_det_DCT1": g[f"prot_det_DCT1_{r}"], "prot_det_DCT2": g[f"prot_det_DCT2_{r}"]})
    tc = pd.DataFrame(tc_rows).sort_values(["label_column", "direction_raw", "class", "protein"])
    for col in cols:
        x = tc[tc["label_column"] == col]
        nb = x[~x["small_difference"]]
        checks.add(f"{col}: transporters / channels compared; concordant / discordant (raw); of these with |diff| >= "
                   f"{SMALL} in both", f"{len(x)}; {int((x['direction_raw'] == 'concordant').sum())} / "
                   f"{int((x['direction_raw'] == 'discordant').sum())}; {int((nb['direction_raw'] == 'concordant').sum())}"
                   f" / {int((nb['direction_raw'] == 'discordant').sum())}", None, "", cond)
        for d in ("concordant", "discordant"):
            y = nb[nb["direction_raw"] == d]
            checks.add(f"{col}: {d} (raw), |diff| >= {SMALL} in both (RNA; protein raw; protein adjusted)",
                       "; ".join(f"{q.protein} ({q.rna_diff_DCT2_minus_DCT1:+.2f}; {q.prot_diff:+.2f}; "
                                 f"{q.prot_diff_adjusted:+.2f})" for q in y.itertuples()) or "none", None, "", cond)
    return genes, conc, tc


def place_labels(sh, ax, x0, y0, w, h, pts, xlim, ylim):
    """atlas_dct_fold_change.place_labels (same candidates and order) with two additions, so a label is never read
    as another point's: a label box may not cover another labelled point (0.8 mm margin), and a label placed
    diagonally (not on its point's line) gets a thin leader line to its point."""
    placed, unplaced = [], []
    LH = T.text_mm("Hg")[1]
    mm = [(x0 + (px - xlim[0]) / (xlim[1] - xlim[0]) * w, y0 + (ylim[1] - py) / (ylim[1] - ylim[0]) * h)
          for _, px, py in pts]
    for k, (name, px, py) in enumerate(pts):
        mx, my = mm[k]
        tw = T.text_mm(name)[0]
        others = [q for i, q in enumerate(mm) if i != k]
        for dx, dy in ((1.2, 0), (-1.2 - tw, 0), (1.0, -LH), (1.0, LH), (-1.0 - tw, -LH), (-1.0 - tw, LH),
                       (1.0, -2 * LH), (1.0, 2 * LH), (-1.0 - tw, -2 * LH), (-1.0 - tw, 2 * LH),
                       (-tw / 2, 1.5 * LH), (-tw / 2, -1.5 * LH), (-tw, 2.5 * LH), (-tw, -2.5 * LH)):
            bx, by = mx + dx, my + dy - LH / 2
            box = (bx - 0.3, by - 0.3, bx + tw + 0.3, by + LH + 0.3)
            inside = box[0] >= x0 and box[2] <= x0 + w and box[1] >= y0 and box[3] <= y0 + h
            covers = any(box[0] - 0.8 <= qx <= box[2] + 0.8 and box[1] - 0.8 <= qy <= box[3] + 0.8 for qx, qy in others)
            if inside and not covers and not any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                                                 for b in placed):
                placed.append(box)
                sh.text(bx, by + LH / 2, name, va="center_baseline", color=T.INK["primary"])
                if dy != 0:
                    ex = min(max(mx, bx), bx + tw)
                    ey = by + LH + 0.2 if dy < 0 else by - 0.2
                    sh.fig.add_artist(Line2D([mx / sh.W, ex / sh.W], [1 - my / sh.H, 1 - ey / sh.H],
                                             color=T.INK["secondary"], lw=0.4))
                break
        else:
            unplaced.append(name)
    return unplaced


def scatter(run, cond: str, genes: pd.DataFrame, fig_rows: list[dict]) -> None:
    """ctrl_rna_core_fold_change.scatter for one condition (rows: primary, second; 'does not apply' panel when needed)."""
    s = L.SETTINGS[cond]
    T.setup_matplotlib()
    title = L.fit_title(f"DCT2 minus DCT1: RNA (atlas_dct-derived split) vs {cond} protein labels, per gene")
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    LH = T.text_mm("Hg")[1]
    pw, ph, gx, gy = 68.0, 52.0, 22.0, 14.0
    note = (f"Genes detected in >= {T.DETECTION:.0%} of DCT1 or DCT2 cells in each modality. RNA: difference of mean "
            "log-normalized expression, all atlas cells and nuclei, DCT2 (atlas_dct) - DCT1 (atlas_dct). Protein: difference of mean "
            "ln(1 + intensity), missing = 0; adjusted: OLS coefficient with ln(protein IDs) and plate. Orange: "
            "transporters and channels (symbol rule); blue, labelled: naming genes of the split. "
            + L.ATLAS_DCT_SPLIT_CAVEAT + " " + L.UNPAIRED_NOTE + (" " + L.ZEROK_NOTE if cond == "ZeroK" else ""))
    note_text, note_h = T.note_block(note)
    hh = LH + 3.0
    absent_h = 16.0
    heights = [absent_h if genes[f"prot_diff_{L.role_of(cond, c)}"].isna().all() else ph for c in s.LABEL_COLUMNS]
    H = T.M + title_h + sum(hh + h for h in heights) + gy + 8.0 + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    rdet = np.maximum(genes["rna_det_DCT1_atlas_dct"], genes["rna_det_DCT2_atlas_dct"])
    for i, col in enumerate(s.LABEL_COLUMNS):
        r = L.role_of(cond, col)
        y_top = T.M + title_h + sum(hh + h + gy for h in heights[:i])
        if genes[f"prot_diff_{r}"].isna().all():
            sh.text(T.M + 2.0, y_top, f"{col} ({r})", va="top")
            L.absent_cell(sh, T.M + 14.0, y_top + hh, 2 * pw + gx, absent_h, "")
            sh.text(T.M + 14.0 + pw + gx / 2, y_top + hh + absent_h / 2,
                    f"Does not apply: {col} has no DCT2 cells (dct_labels).\nSee the second column ({s.SECOND}) below.",
                    ha="center", va="center", color=T.INK["secondary"], linespacing=1.3)
            continue
        pdet = np.maximum(genes[f"prot_det_DCT1_{r}"], genes[f"prot_det_DCT2_{r}"])
        g = genes[(rdet >= T.DETECTION) & (pdet >= T.DETECTION)]
        for j, (ycol, ylab) in enumerate(((f"prot_diff_{r}", "Protein difference"),
                                          (f"prot_diff_adjusted_{r}", "Protein difference, adjusted"))):
            x0 = T.M + 14.0 + j * (pw + gx)
            y0 = y_top + hh
            x, y = g["rna_diff_DCT2_minus_DCT1"].to_numpy(), g[ycol].to_numpy()
            rho = stats.spearmanr(x, y).statistic
            sh.text(x0 - 12.0, y0 - hh, f"{col} ({r}): rho = {rho:.2f} (n = {len(g):,})", va="top")
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
            unplaced = place_labels(sh, ax, x0, y0, pw, ph, pts, (-xl, xl), (-yl, yl))
            print(f"  fc_scatter_{L.KEY[cond]} {col} panel {j + 1}: labels not placed: {', '.join(unplaced) or 'none'}")
    sh.note(note_text, note_h)
    L.save_checked(run, sh, f"fc_scatter_{L.KEY[cond]}", title.replace("\n", " "), fig_rows, condition=cond)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "fold_change", args)
    L.ported_from(run, "ctrl_rna_core_fold_change.py", "atlas_dct_fold_change.py")
    checks = L.Checks(L.DATASET)
    region = T.load_region(run, with_split=True)
    split = pd.read_csv(run.use_input(T.SPLIT_CSV, "atlas_dct-derived split"), dtype={"atlas_cell": str}).set_index("atlas_cell")
    naming = T.naming_lists(run, checks)
    naming_set = {g for v in naming.values() for g in v}
    workbook = T.workbook_markers(run)
    ln_full = T.lognorm(region.X)
    res = {c: one_condition(run, c, region, split, ln_full, naming_set, workbook, checks) for c in L.CONDITIONS}
    fig_rows = []
    for cond in L.CONDITIONS:
        scatter(run, cond, res[cond][0], fig_rows)
    figs = L.figure_rows_frame(fig_rows, "fold_change")
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(L.stack({c: res[c][0] for c in L.CONDITIONS}), "fc_genes")
    run.table(L.stack({c: res[c][1] for c in L.CONDITIONS}), "fc_concordance")
    run.table(L.stack({c: res[c][2] for c in L.CONDITIONS}), "fc_transporters_channels")
    run.table(pd.DataFrame([{"condition": "both", "class": c, "regex": rx} for c, rx in CLASS_RULES]), "fc_class_rules")
    run.table(figs, "fc_figure_checks")
    run.table(checks.frame(), "fc_checks")
    run.finish({"rna_variants": RNA_VARIANTS, "primary_rna_variant": PRIMARY_RNA, "detection": T.DETECTION,
                "small_difference": SMALL, "class_rules": dict(CLASS_RULES),
                "adjusted_model": "OLS ln1p ~ DCT2 + ln(protein IDs) + plate (DCT1 and DCT2 cells)",
                "label_columns": {c: list(L.SETTINGS[c].LABEL_COLUMNS) for c in L.CONDITIONS},
                "not_applicable": {"ZeroK": [L.SETTINGS["ZeroK"].PRIMARY, NA_REASON]}, "formats": list(L.FORMATS),
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see dct_rna_fc_checks.csv.")


if __name__ == "__main__":
    main()
