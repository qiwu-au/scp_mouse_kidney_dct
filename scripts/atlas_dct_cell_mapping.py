#!/usr/bin/env python
"""atlas_dct step 4: cell-level mapping of the SCP DCT cells to RNA DCT / CNT / TAL centroids (cosine).

Analysis plan, step 4.

Method (fixed in the approved plan)
-----------------------------------
- Reference groups (atlas cells and nuclei, all Sources): published DCT, DCT-CNT, CNT, CTAL, MTAL (primary); and the
  atlas_dct-derived level DCT1 (atlas_dct), DCT2 (atlas_dct), DCT-CNT, CNT, CTAL, MTAL (DCT unassigned (atlas_dct) not a centroid).
- Informative genes: Wilcoxon one-vs-rest among the reference groups (scanpy rank_genes_groups on log-normalized
  expression of the 2,172 shared genes), kept if BH q < 0.05, log2 fold change >= 1, detected (counts > 0) in >= 25%
  of the group's cells, and the protein detected (intensity > 0) in >= 10% of the 750 SCP cells; the top 50 per
  group by Wilcoxon score; the union is the informative set (all listed in the CSV). Sensitivity: top 25 / top 100.
- Centroids: group means of log-normalized expression, z-scored per gene across the groups (each group weighted
  equally). Sensitivity ('kidney_rna style'): expression standardized per gene by the mean and SD over all reference cells,
  centroid = group mean of the standardized values (as kidney_rna_reference_mapping.py).
- SCP cells: canonical .X (per-protein z-scored ln(1 + intensity), clipped at +/-10) on the informative proteins.
- Cosine similarity to each centroid; top class, top score, second, margin (top - second). Low confidence: score <
  0.20 or margin < 0.05 (kidney_rna thresholds). Predictions do not use the SCP labels.

Reported: confusion against both V1 label columns, margins, the fraction mapped to TAL (CTAL or MTAL), CNT and
DCT-CNT; cross-tabulation with the reviewer_core TAL / CNT_CD / interstitial flags; by plate and by protein-ID bin (reviewer_core bins);
Spearman rho of the top score and margin with protein IDs.

Inputs (read-only): kidney_object canonical DCT object and labels; reviewer_core per-cell flags (reviewer_core_task3_marker_scores_per_cell.csv);
                    atlas_dct region subset, split table and shared features.
Outputs: results/tables/atlas_dct/atlas_dct_mapping_*.csv;
         results/figures/atlas_dct/atlas_dct_mapping_confusion.pdf/.png, atlas_dct_mapping_scores.pdf/.png

Command (project root):  python scripts/atlas_dct_cell_mapping.py
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

TOP_N = 50
TOP_N_SENS = (25, 50, 100)
Q_MAX, LFC_MIN, PCT_MIN, SCP_DET_MIN = 0.05, 1.0, 0.25, 0.10
SCORE_MIN, MARGIN_MIN = 0.20, 0.05
FLAGS = ("tal_like_flag", "cnt_cd_like_flag", "interstitial_like_flag", "ambiguous_tal_cnt_interstitial")
BIN_EDGES = [100, 300, 500, 700, 900, 1100, 1300, np.inf]          # reviewer_core B.6 protein-ID bins
BIN_LABELS = ["100-299", "300-499", "500-699", "700-899", "900-1099", "1100-1299", "1300+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atlas_dct step 4: SCP cell -> RNA centroid mapping (cosine).")
    cc.add_common_args(parser)
    return parser.parse_args()


def de_table(ln_sub, groups: np.ndarray, order, genes, scp_det: pd.Series) -> pd.DataFrame:
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=ln_sub, obs=pd.DataFrame({"g": pd.Categorical(groups, categories=list(order))}),
                   var=pd.DataFrame(index=list(genes)))
    sc.tl.rank_genes_groups(a, "g", method="wilcoxon", use_raw=False, tie_correct=True)
    detected = ln_sub > 0
    out = []
    for g in order:
        d = sc.get.rank_genes_groups_df(a, group=g)
        d = d[["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]].copy()
        m = groups == g
        fg = pd.Series(np.asarray(detected[np.flatnonzero(m)].mean(axis=0)).ravel(), index=list(genes))
        fr = pd.Series(np.asarray(detected[np.flatnonzero(~m)].mean(axis=0)).ravel(), index=list(genes))
        d["frac_group"] = fg.reindex(d["names"]).to_numpy()
        d["frac_rest"] = fr.reindex(d["names"]).to_numpy()
        d.insert(0, "group", g)
        out.append(d)
    d = pd.concat(out, ignore_index=True).rename(columns={"names": "atlas_gene", "scores": "wilcoxon_score",
                                                          "logfoldchanges": "log2fc", "pvals": "p_value",
                                                          "pvals_adj": "q_bh"})
    d["scp_detection_all"] = scp_det.reindex(d["atlas_gene"]).to_numpy()
    d["passes_filters"] = ((d["q_bh"] < Q_MAX) & (d["log2fc"] >= LFC_MIN) & (d["frac_group"] >= PCT_MIN)
                           & (d["scp_detection_all"] >= SCP_DET_MIN))
    d["rank_in_group"] = np.nan
    for g in order:
        m = (d["group"] == g) & d["passes_filters"]
        d.loc[m, "rank_in_group"] = d.loc[m, "wilcoxon_score"].rank(ascending=False, method="first").to_numpy()
    return d


def informative(de: pd.DataFrame, n: int) -> list[str]:
    sel = de[de["rank_in_group"] <= n]
    return list(dict.fromkeys(sel.sort_values(["group", "rank_in_group"])["atlas_gene"]))


def cosine_map(cell_x: np.ndarray, cent: pd.DataFrame) -> pd.DataFrame:
    """cell_x: cells x genes; cent: groups x genes (same gene order). Returns scores per group."""
    c = cent.to_numpy(float)
    num = cell_x @ c.T
    den = np.linalg.norm(cell_x, axis=1)[:, None] * np.linalg.norm(c, axis=1)[None, :]
    return pd.DataFrame(num / np.where(den > 0, den, np.nan), columns=cent.index)


def summarize(scores: pd.DataFrame, index) -> pd.DataFrame:
    s = scores.to_numpy()
    o = np.argsort(-s, axis=1)
    top = scores.columns.to_numpy()[o[:, 0]]
    sec = scores.columns.to_numpy()[o[:, 1]]
    ts, ss = s[np.arange(len(s)), o[:, 0]], s[np.arange(len(s)), o[:, 1]]
    out = pd.DataFrame({"top_class": top, "top_score": ts, "second_class": sec, "second_score": ss,
                        "margin": ts - ss}, index=index)
    out["low_confidence"] = (out["top_score"] < SCORE_MIN) | (out["margin"] < MARGIN_MIN)
    return out


def class_fractions(pred: pd.DataFrame, labels: pd.Series, classes) -> pd.DataFrame:
    rows = []
    lab = labels.astype(str)
    for g in [*T.SCP_GROUPS, "all"]:
        m = np.ones(len(lab), bool) if g == "all" else (lab == g).to_numpy()
        if not m.any():
            continue
        p = pred.loc[m]
        rec = {"scp_label": g, "n_cells": int(m.sum()), "median_top_score": float(p["top_score"].median()),
               "median_margin": float(p["margin"].median()), "frac_low_confidence": float(p["low_confidence"].mean())}
        for c in classes:
            rec[f"frac_top_{c}"] = float((p["top_class"] == c).mean())
        rec["frac_top_TAL"] = float(p["top_class"].isin(T.TAL_CLASSES).mean())
        rec["frac_top_TAL_confident"] = float((p["top_class"].isin(T.TAL_CLASSES) & ~p["low_confidence"]).mean())
        rec["frac_top_CNT_confident"] = float(((p["top_class"] == "CNT") & ~p["low_confidence"]).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "cell_mapping", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, ln1p, det = T.load_scp(run, checks)
    flags = pd.read_csv(run.use_input(T.REVIEWER_CORE_DIR / "reviewer_core_task3_marker_scores_per_cell.csv",
                                      "reviewer_core per-cell TAL / CNT_CD / interstitial flags"), dtype={"cell_id": str})
    flags = flags.set_index("cell_id").reindex(labels.index)
    same = flags[list(T.LABEL_COLUMNS)].astype(str).equals(labels[list(T.LABEL_COLUMNS)].astype(str))
    checks.add("reviewer_core flag table: same 750 cells and labels as kidney_object", f"{same}", same)
    region = T.load_region(run, with_split=True)
    sf = T.read_table(run, "shared_features")
    sf = sf[sf["shared"]]
    gene_to_prot = dict(zip(sf["atlas_gene"], sf["protein"]))
    pos = region.var_names.get_indexer(sf["atlas_gene"])
    ln_all = T.lognorm(region.X)[:, pos]
    genes = pd.Index(sf["atlas_gene"])
    scp_det = pd.Series(det[sf["protein"]].mean(axis=0).to_numpy(), index=sf["atlas_gene"].to_numpy())
    Xz = pd.DataFrame(cc.dense(adata.X), index=labels.index, columns=adata.var_names.astype(str))
    ids = labels["n_detected_proteins"].astype(float)

    de_all, per_cell, conf_rows, summ_rows, var_rows, gene_rows = [], [], [], [], [], []
    primary_pred, all_preds = {}, {}
    for derived in (False, True):
        lvl = T.LEVEL_SHORT[derived]
        order = list(T.rna_groups(derived))
        grp = region.obs[T.group_column(derived)].astype(str).to_numpy()
        ref = np.isin(grp, order)
        ln_ref = ln_all[np.flatnonzero(ref)]
        print(f"[mapping] Wilcoxon one-vs-rest, {lvl}: {int(ref.sum())} cells x {len(genes)} genes", flush=True)
        de = de_table(ln_ref, grp[ref], order, genes, scp_det)
        de.insert(0, "level", lvl)
        de_all.append(de)
        n_pass = de[de["passes_filters"]].groupby("group").size().reindex(order, fill_value=0)
        checks.add(f"{lvl}: genes passing the filters per group", "; ".join(f"{g} {n}" for g, n in n_pass.items()),
                   None)
        # centroids
        stats_ = T.group_stats(ln_ref, region.X[np.flatnonzero(ref)][:, pos], pd.Series(grp[ref]), order, genes)
        mean_ref = np.asarray(ln_ref.mean(axis=0)).ravel()
        sd_ref = np.sqrt(np.maximum(np.asarray(ln_ref.multiply(ln_ref).mean(axis=0)).ravel() - mean_ref ** 2, 0))
        for n in TOP_N_SENS:
            inf = informative(de, n)
            for style in ("equal_weight_z", "t07_style"):
                if style == "equal_weight_z":
                    cent = T.zscore_rows(stats_["mean"][inf])
                else:
                    j = genes.get_indexer(inf)
                    cent = (stats_["mean"][inf] - mean_ref[j]) / np.where(sd_ref[j] > 0, sd_ref[j], 1.0)
                prots = [gene_to_prot[g] for g in inf]
                cx = Xz[prots].to_numpy(float)
                scores = cosine_map(cx, cent)
                pred = summarize(scores, labels.index)
                variant = f"top{n}_{style}"
                primary = n == TOP_N and style == "equal_weight_z"
                if primary:
                    primary_pred[derived] = (pred, scores, inf)
                    gene_rows += [{"level": lvl, "atlas_gene": g, "protein": gene_to_prot[g],
                                   "group": de[(de["atlas_gene"] == g) & (de["rank_in_group"] <= n)]["group"].iloc[0],
                                   **{f"centroid_z_{c}": float(cent.loc[c, g]) for c in order}} for g in inf]
                for col in T.LABEL_COLUMNS:
                    fr = class_fractions(pred, labels[col], order)
                    fr.insert(0, "label_column", col)
                    fr.insert(0, "variant", variant)
                    fr.insert(0, "level", lvl)
                    fr["n_informative_genes"] = len(inf)
                    fr["primary_variant"] = primary
                    summ_rows.append(fr)
                    ct = pd.crosstab(labels[col].astype(str), pred["top_class"]).reindex(columns=order, fill_value=0)
                    for sg in ct.index:
                        for c in order:
                            conf_rows.append({"level": lvl, "variant": variant, "label_column": col, "scp_label": sg,
                                              "rna_class": c, "n_cells": int(ct.loc[sg, c]),
                                              "fraction_of_label": float(ct.loc[sg, c] / ct.loc[sg].sum())})
                all_preds[(derived, variant)] = (pred, len(inf))
        p, s, inf = primary_pred[derived]
        for (dv, variant), (pred, n_inf) in all_preds.items():
            if dv == derived and variant != f"top{TOP_N}_equal_weight_z":
                var_rows.append({"level": lvl, "variant": variant, "n_informative_genes": n_inf,
                                 "fraction_same_top_class_as_primary": float((pred["top_class"] == p["top_class"]).mean()),
                                 "fraction_low_confidence": float(pred["low_confidence"].mean())})
        checks.add(f"{lvl} primary: informative genes (top {TOP_N} per group); cells low confidence",
                   f"{len(inf)}; {int(p['low_confidence'].sum())} of {len(p)}", None)
        pc = p.add_prefix(f"{lvl}_")
        for c in s.columns:
            pc[f"{lvl}_score_{c}"] = s[c].to_numpy()
        per_cell.append(pc)

    pc = pd.concat([labels[["plate", *T.LABEL_COLUMNS, "n_detected_proteins"]], flags[list(FLAGS)], *per_cell],
                   axis=1).reset_index()
    pc["depth_bin"] = pd.cut(pc["n_detected_proteins"], BIN_EDGES, right=False, labels=BIN_LABELS).astype(str)
    summ = pd.concat(summ_rows, ignore_index=True)
    conf = pd.DataFrame(conf_rows)

    # flags, plate, depth (primary variants) -----------------------------------------------------------------
    flag_rows, pd_rows = [], []
    for derived in (False, True):
        lvl = T.LEVEL_SHORT[derived]
        top = pc[f"{lvl}_top_class"]
        mapped = {"TAL": top.isin(T.TAL_CLASSES), "CNT": top == "CNT", "DCT-CNT": top == "DCT-CNT",
                  "TAL or CNT": top.isin([*T.TAL_CLASSES, "CNT"])}
        for mname, mv in mapped.items():
            for conf_only in (False, True):
                mm = mv & ~pc[f"{lvl}_low_confidence"] if conf_only else mv
                for fl in FLAGS:
                    f_ = pc[fl].astype(bool)
                    tab = np.array([[int((mm & f_).sum()), int((mm & ~f_).sum())],
                                    [int((~mm & f_).sum()), int((~mm & ~f_).sum())]])
                    odds, p = stats.fisher_exact(tab)
                    flag_rows.append({"level": lvl, "mapped_to": mname, "confident_only": conf_only, "t02_flag": fl,
                                      "n_mapped": int(mm.sum()), "n_flagged": int(f_.sum()),
                                      "n_mapped_and_flagged": int(tab[0, 0]),
                                      "frac_mapped_that_are_flagged": tab[0, 0] / max(int(mm.sum()), 1),
                                      "frac_flagged_that_are_mapped": tab[0, 0] / max(int(f_.sum()), 1),
                                      "fisher_odds_ratio": odds, "fisher_p": p})
        for by in ("plate", "depth_bin"):
            for key, sub in pc.groupby(by):
                t_ = sub[f"{lvl}_top_class"]
                rec = {"level": lvl, "by": by, "value": key, "n_cells": len(sub),
                       "frac_low_confidence": float(sub[f"{lvl}_low_confidence"].mean()),
                       "median_top_score": float(sub[f"{lvl}_top_score"].median())}
                for c in T.rna_groups(derived):
                    rec[f"frac_top_{c}"] = float((t_ == c).mean())
                rec["frac_top_TAL"] = float(t_.isin(T.TAL_CLASSES).mean())
                pd_rows.append(rec)
        for q in ("top_score", "margin"):
            r = stats.spearmanr(pc[f"{lvl}_{q}"], pc["n_detected_proteins"])
            checks.add(f"{lvl} primary: Spearman rho of {q} with protein IDs per cell", f"{r.statistic:.3f} "
                       f"(p {r.pvalue:.1e})", None)
            pd_rows.append({"level": lvl, "by": f"spearman_{q}_vs_protein_ids", "value": "all", "n_cells": len(pc),
                            "rho": float(r.statistic), "p_value": float(r.pvalue)})
        for col in T.LABEL_COLUMNS:
            s = summ[(summ["level"] == lvl) & summ["primary_variant"] & (summ["label_column"] == col)]
            checks.add(f"{lvl} primary / {col}: fraction mapped to TAL; CNT; DCT-CNT; low confidence, per label",
                       "; ".join(f"{r.scp_label} {r.frac_top_TAL:.2f} / {r['frac_top_CNT']:.2f} / "
                                 f"{r['frac_top_DCT-CNT']:.2f} / {r.frac_low_confidence:.2f}"
                                 for _, r in s.iterrows()), None)

    fig_rows = []
    figures(run, conf, pc, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(pd.concat(de_all, ignore_index=True), "mapping_de_genes")
    run.table(pd.DataFrame(gene_rows), "mapping_informative_genes")
    run.table(pc, "mapping_per_cell")
    run.table(conf, "mapping_confusion")
    run.table(summ, "mapping_summary")
    run.table(pd.DataFrame(var_rows), "mapping_variants")
    run.table(pd.DataFrame(flag_rows), "mapping_flags")
    run.table(pd.DataFrame(pd_rows), "mapping_plate_depth")
    run.table(figs, "mapping_figure_checks")
    run.table(checks.frame(), "mapping_checks")
    run.finish({"top_n": TOP_N, "top_n_sensitivity": list(TOP_N_SENS), "q_max": Q_MAX, "log2fc_min": LFC_MIN,
                "frac_group_min": PCT_MIN, "scp_detection_min": SCP_DET_MIN, "score_min": SCORE_MIN,
                "margin_min": MARGIN_MIN, "centroids": "equal-weight z across groups (primary); kidney_rna style (sensitivity)",
                "scp_matrix": "canonical .X (z-scored ln(1 + intensity), clipped at 10)", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see atlas_dct_mapping_checks.csv.")


def figures(run, conf: pd.DataFrame, pc: pd.DataFrame, fig_rows: list[dict]) -> None:
    plt = T.setup_matplotlib()
    cmap = T.sequential_cmap()
    LH = T.text_mm("Hg")[1]
    variant = f"top{TOP_N}_equal_weight_z"
    # 1. confusion -------------------------------------------------------------------------------------
    rows = [(col, g) for col in T.LABEL_COLUMNS for g in T.SCP_GROUPS]
    row_txt = [f"{g} ({T.LABEL_SHORT[c]})" for c, g in rows]
    lw = max(T.text_mm(r)[0] for r in row_txt) + 1.0 + 1.6 + 1.0
    rh = 4.6
    title = "SCP DCT cells mapped to the nearest RNA centroid (cosine; fraction of each SCP label)"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    panels = [(False, "Atlas classes"), (True, "atlas_dct-derived split + atlas classes")]
    cw = 9.5
    gene_h = max(T.text_mm(c)[0] for c in (*T.DERIVED_GROUPS, *T.PUBLISHED_GROUPS)) + 1.5
    note = (f"Top {TOP_N} informative genes per RNA group (Wilcoxon one-vs-rest among the groups; q < {Q_MAX}, log2 FC >= "
            f"{LFC_MIN:g}, detected in >= {PCT_MIN:.0%} of the group and in >= {SCP_DET_MIN:.0%} of SCP cells); centroids "
            "z-scored per gene across the groups; SCP cells: canonical z-scored ln(1 + intensity). Numbers: cells. "
            "cell / cluster = V1 per-cell / cluster labels. DCT1 (atlas_dct) / DCT2 (atlas_dct): atlas_dct-derived split, not atlas labels.")
    note_text, note_h = T.note_block(note)
    H = T.M + title_h + LH + 1.0 + len(rows) * rh + gene_h + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    x = T.M
    for derived, ptitle in panels:
        order = list(T.rna_groups(derived))
        y = T.M + title_h
        sh.text(x, y, ptitle, va="top")
        x0, y0 = x + lw, y + LH + 1.0
        c = conf[(conf["level"] == T.LEVEL_SHORT[derived]) & (conf["variant"] == variant)]
        frac = np.array([[c[(c["label_column"] == col) & (c["scp_label"] == g) & (c["rna_class"] == k)]
                          ["fraction_of_label"].iloc[0] for k in order] for col, g in rows])
        nn = np.array([[c[(c["label_column"] == col) & (c["scp_label"] == g) & (c["rna_class"] == k)]
                        ["n_cells"].iloc[0] for k in order] for col, g in rows])
        ax = sh.ax(x0, y0, len(order) * cw, len(rows) * rh)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=(0, len(order), len(rows), 0))
        ax.axhline(len(T.SCP_GROUPS), color="white", lw=1.5)
        ax.set_axis_off()
        for a in range(len(rows)):
            for b in range(len(order)):
                sh.text(x0 + (b + 0.5) * cw, y0 + (a + 0.5) * rh, f"{nn[a, b]}", ha="center", va="center_baseline",
                        color="white" if frac[a, b] > 0.55 else T.INK["primary"])
        for a, ((col, g), t_) in enumerate(zip(rows, row_txt)):
            yc = y0 + (a + 0.5) * rh
            sh.rect(x0 - 1.0 - 1.6, yc - 0.8, 1.6, 1.6, facecolor=T.SCP_PALETTE[g], edgecolor="none")
            sh.text(x0 - 1.0 - 1.6 - 1.0, yc, t_, ha="right", va="center_baseline")
        for b, k in enumerate(order):
            sh.text(x0 + (b + 0.5) * cw, y0 + len(rows) * rh + 0.8, k, rotation=90, ha="center", va="top")
        x = x0 + len(order) * cw + 8.0
    cax = sh.ax(x, T.M + title_h + LH + 1.0, 2.4, 20.0)
    cax.imshow(np.linspace(1, 0, 256)[:, None], cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 0.5, 1])
    cax.tick_params(length=1.5, width=0.4, labelsize=T.FONT, pad=1)
    sh.text(x + 2.4 + 6.5, T.M + title_h + LH + 1.0 + 10.0, "Fraction of label", rotation=90, ha="left", va="center")
    sh.note(note_text, note_h)
    T.save_checked(run, sh, "mapping_confusion", title, fig_rows)

    # 2. top score and margin per SCP label (primary, atlas classes) ------------------------------------------
    title = "Cosine top score and margin per SCP label (atlas classes, primary variant)"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    note = (f"Dashed lines: low-confidence thresholds (score {SCORE_MIN}, margin {MARGIN_MIN}). Boxes: median and "
            "quartiles; whiskers 5th-95th percentile. cell / cluster = V1 per-cell / cluster labels.")
    note_text, note_h = T.note_block(note)
    ph = 40.0
    H = T.M + title_h + 3.0 + ph + 10.0 + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    for k, (q, ylab, thr) in enumerate((("top_score", "Top cosine score", SCORE_MIN),
                                        ("margin", "Margin (top - second)", MARGIN_MIN))):
        ax = sh.ax(T.M + 12.0 + k * 88.0, T.M + title_h + 3.0, 72.0, ph)
        data, ticks, cols = [], [], []
        for col in T.LABEL_COLUMNS:
            for g in T.SCP_GROUPS:
                data.append(pc.loc[pc[col] == g, f"published_{q}"].to_numpy(float))
                ticks.append(f"{g}\n{T.LABEL_SHORT[col]}")
                cols.append(T.SCP_PALETTE[g])
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.6, patch_artist=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, alpha=0.55, edgecolor=T.INK["secondary"], linewidth=0.5)
        for part in ("whiskers", "caps", "medians"):
            for l_ in bp[part]:
                l_.set(color=T.INK["secondary"], linewidth=0.6)
        ax.axhline(thr, color=T.INK["muted"], lw=0.6, ls="--")
        ax.set_xticks(range(1, len(ticks) + 1))
        ax.set_xticklabels(ticks)
        ax.set_ylabel(ylab)
        T.style_axes(ax)
    sh.note(note_text, note_h)
    T.save_checked(run, sh, "mapping_scores", title, fig_rows)


if __name__ == "__main__":
    main()
