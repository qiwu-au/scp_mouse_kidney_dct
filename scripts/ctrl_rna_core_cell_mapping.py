#!/usr/bin/env python
"""ctrl_rna_core step 3: cell-level mapping of the control SCP DCT cells to RNA DCT / CNT / TAL centroids (cosine).

Analysis plan,
step 3 (port of atlas_dct_cell_mapping.py; ``de_table``, ``informative``, ``cosine_map``, ``summarize`` and the
thresholds imported unchanged).

Method (atlas_dct step 4)
-------------------
- Reference groups (atlas cells and nuclei, all Sources): published DCT, DCT-CNT, CNT, CTAL, MTAL; and the
  atlas_dct-derived level DCT1 (atlas_dct), DCT2 (atlas_dct), DCT-CNT, CNT, CTAL, MTAL (DCT unassigned (atlas_dct) not a centroid).
- Informative genes: Wilcoxon one-vs-rest among the reference groups (scanpy rank_genes_groups on log-normalized
  expression of the shared genes of step 1), kept if BH q < 0.05, log2 fold change >= 1, detected (counts > 0) in
  >= 25% of the group's cells, and the protein detected (intensity > 0) in >= 10% of the 727 control cells; the top
  50 per group by Wilcoxon score; the union is the informative set (all listed in the CSV). Sensitivity: top 25 /
  top 100, and kidney_rna-style centroids.
- Centroids: group means of log-normalized expression, z-scored per gene across the groups (equal weight).
- SCP cells: labels_core ``.X`` (per-protein z-scored ln(1 + intensity), clipped at +/-10) on the informative proteins.
- Cosine similarity to each centroid; top class, score, second, margin. Low confidence: score < 0.20 or margin <
  0.05 (kidney_rna thresholds). Predictions do not use the SCP labels.

Reported: confusion against both label columns (DCT1, DCT2, ProLIF, Unknown), margins, the fraction mapped to TAL
(CTAL or MTAL), CNT and DCT-CNT; cross-tabulation with the ctrl_reviewer_core TAL / CNT_CD / interstitial flags; by plate and by
protein-ID quartile of the control cells (atlas_dct used the reviewer_core B.6 bins, set for the retired data's depth range);
Spearman rho of the top score and margin with protein IDs.

Inputs (read-only): labels_core labelled control object and canonical labels; ctrl_reviewer_core per-cell flags
                    (ctrl_reviewer_core_task3_marker_scores_per_cell.csv); atlas_dct region subset and split table; ctrl_rna_core step 1.
Outputs: results/tables/ctrl_rna_core/ctrl_rna_core_mapping_*.csv;
         results/figures/ctrl_rna_core/ctrl_rna_core_mapping_confusion.pdf/.png, ctrl_rna_core_mapping_scores.pdf/.png

Command (project root):  python scripts/ctrl_rna_core_cell_mapping.py
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
from atlas_dct_cell_mapping import (FLAGS, LFC_MIN, MARGIN_MIN, PCT_MIN, Q_MAX, SCORE_MIN, SCP_DET_MIN, TOP_N,
                                     TOP_N_SENS, cosine_map, de_table, informative, summarize)

LABELS_CORE_QUARTILE_EDGES = (108.0, 1372.5, 2096.0, 2786.0, 5058.0)      # labels_core README section 4b
BIN_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_rna_core step 3: control SCP cell -> RNA centroid mapping (cosine).")
    cc.add_common_args(parser)
    return parser.parse_args()


def class_fractions(pred: pd.DataFrame, labels: pd.Series, classes) -> pd.DataFrame:
    """atlas_dct_cell_mapping.class_fractions with the ctrl_rna_core SCP groups (Unknown its own row)."""
    rows = []
    lab = labels.astype(str)
    for g in [*L.SCP_GROUPS, "all"]:
        m = np.ones(len(lab), bool) if g == "all" else (lab == g).to_numpy()
        if not m.any():
            continue
        p = pred.loc[m]
        rec = {"scp_label": g, "n_cells": int(m.sum()), "median_top_score": float(p["top_score"].median()),
               "median_margin": float(p["margin"].median()), "frac_low_confidence": float(p["low_confidence"].mean())}
        for c in classes:
            rec[f"frac_top_{c}"] = float((p["top_class"] == c).mean())
        rec["frac_top_TAL"] = float(p["top_class"].isin(T.TAL_CLASSES).mean())
        rec["frac_top_TAL_or_CNT"] = float(p["top_class"].isin([*T.TAL_CLASSES, "CNT"]).mean())
        rec["frac_top_TAL_confident"] = float((p["top_class"].isin(T.TAL_CLASSES) & ~p["low_confidence"]).mean())
        rec["frac_top_CNT_confident"] = float(((p["top_class"] == "CNT") & ~p["low_confidence"]).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "cell_mapping", args)
    L.ported_from(run, "atlas_dct_cell_mapping.py")
    checks = t5.Checks(L.DATASET)
    adata, labels, ln1p, det = L.load_scp(run, checks)
    flags = L.per_cell_table(run, L.CTRL_REVIEWER_CORE_TASK, "task3_marker_scores_per_cell", labels, checks)
    region = T.load_region(run, with_split=True)
    sf = L.read_table(run, "shared_features")
    sf = sf[sf["shared"]]
    gene_to_prot = dict(zip(sf["atlas_gene"], sf["protein"]))
    pos = region.var_names.get_indexer(sf["atlas_gene"])
    ln_all = T.lognorm(region.X)[:, pos]
    genes = pd.Index(sf["atlas_gene"])
    scp_det = pd.Series(det[sf["protein"]].mean(axis=0).to_numpy(), index=sf["atlas_gene"].to_numpy())
    Xz = pd.DataFrame(cc.dense(adata.X), index=labels.index, columns=adata.var_names.astype(str))
    ids = labels["n_detected_proteins"].astype(float)
    q = np.quantile(ids, [0, 0.25, 0.5, 0.75, 1.0])
    checks.add("protein-ID quartile edges equal labels_core section 4b", "; ".join(f"{v:g}" for v in q),
               bool(np.allclose(q, LABELS_CORE_QUARTILE_EDGES)), "; ".join(f"{v:g}" for v in LABELS_CORE_QUARTILE_EDGES))

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
                scores = cosine_map(Xz[prots].to_numpy(float), cent)
                pred = summarize(scores, labels.index)
                variant = f"top{n}_{style}"
                primary = n == TOP_N and style == "equal_weight_z"
                if primary:
                    primary_pred[derived] = (pred, scores, inf)
                    gene_rows += [{"level": lvl, "atlas_gene": g, "protein": gene_to_prot[g],
                                   "group": de[(de["atlas_gene"] == g) & (de["rank_in_group"] <= n)]["group"].iloc[0],
                                   "scp_detection_all": float(scp_det[g]),
                                   **{f"centroid_z_{c}": float(cent.loc[c, g]) for c in order}} for g in inf]
                for col in L.LABEL_COLUMNS:
                    fr = class_fractions(pred, labels[col], order)
                    fr.insert(0, "label_column", col)
                    fr.insert(0, "variant", variant)
                    fr.insert(0, "level", lvl)
                    fr["n_informative_genes"] = len(inf)
                    fr["primary_variant"] = primary
                    summ_rows.append(fr)
                    ct = (pd.crosstab(labels[col].astype(str), pred["top_class"])
                          .reindex(index=list(L.SCP_GROUPS), columns=order, fill_value=0))
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
        per_group = pd.Series([r["group"] for r in gene_rows if r["level"] == lvl]).value_counts().reindex(order,
                                                                                                         fill_value=0)
        checks.add(f"{lvl} primary: informative genes (top {TOP_N} per group; per group); cells low confidence",
                   f"{len(inf)} ({'; '.join(f'{g} {n}' for g, n in per_group.items())}); "
                   f"{int(p['low_confidence'].sum())} of {len(p)}", None)
        pc = p.add_prefix(f"{lvl}_")
        for c in s.columns:
            pc[f"{lvl}_score_{c}"] = s[c].to_numpy()
        per_cell.append(pc)

    pc = pd.concat([labels[["plate", *L.LABEL_COLUMNS, "n_detected_proteins"]], flags[list(FLAGS)], *per_cell],
                   axis=1).reset_index()
    pc["depth_bin"] = pd.cut(pc["n_detected_proteins"], list(LABELS_CORE_QUARTILE_EDGES), labels=BIN_LABELS,
                             include_lowest=True).astype(str)
    checks.add("cells per protein-ID quartile", "; ".join(f"{k} {v}" for k, v in
                                                         pc["depth_bin"].value_counts().reindex(BIN_LABELS).items()),
               bool(pc["depth_bin"].isin(BIN_LABELS).all()))
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
                    odds, pv = stats.fisher_exact(tab)
                    flag_rows.append({"level": lvl, "mapped_to": mname, "confident_only": conf_only, "t19_flag": fl,
                                      "n_mapped": int(mm.sum()), "n_flagged": int(f_.sum()),
                                      "n_mapped_and_flagged": int(tab[0, 0]),
                                      "frac_mapped_that_are_flagged": tab[0, 0] / max(int(mm.sum()), 1),
                                      "frac_flagged_that_are_mapped": tab[0, 0] / max(int(f_.sum()), 1),
                                      "fisher_odds_ratio": odds, "fisher_p": pv})
        for by in ("plate", "depth_bin"):
            for key, sub in pc.groupby(by):
                t_ = sub[f"{lvl}_top_class"]
                rec = {"level": lvl, "by": by, "value": key, "n_cells": len(sub),
                       "median_protein_ids": float(sub["n_detected_proteins"].median()),
                       "frac_low_confidence": float(sub[f"{lvl}_low_confidence"].mean()),
                       "median_top_score": float(sub[f"{lvl}_top_score"].median())}
                for c in T.rna_groups(derived):
                    rec[f"frac_top_{c}"] = float((t_ == c).mean())
                rec["frac_top_TAL"] = float(t_.isin(T.TAL_CLASSES).mean())
                pd_rows.append(rec)
        for qn in ("top_score", "margin"):
            r = stats.spearmanr(pc[f"{lvl}_{qn}"], pc["n_detected_proteins"])
            checks.add(f"{lvl} primary: Spearman rho of {qn} with protein IDs per cell", f"{r.statistic:.3f} "
                       f"(p {r.pvalue:.1e})", None)
            pd_rows.append({"level": lvl, "by": f"spearman_{qn}_vs_protein_ids", "value": "all", "n_cells": len(pc),
                            "rho": float(r.statistic), "p_value": float(r.pvalue)})
        for col in L.LABEL_COLUMNS:
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
                "scp_matrix": "labels_core .X (z-scored ln(1 + intensity), clipped at 10)",
                "depth_bins": "control protein-ID quartiles (labels_core edges)", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see ctrl_rna_core_mapping_checks.csv.")


def figures(run, conf: pd.DataFrame, pc: pd.DataFrame, fig_rows: list[dict]) -> None:
    T.setup_matplotlib()
    cmap = T.sequential_cmap()
    LH = T.text_mm("Hg")[1]
    variant = f"top{TOP_N}_equal_weight_z"
    # 1. confusion -------------------------------------------------------------------------------------
    rows = [r for col in L.LABEL_COLUMNS for r in L.label_rows(col)]
    row_txt = [f"{g} ({L.LABEL_SHORT[c]})" for c, g in rows]
    lw = max(T.text_mm(r)[0] for r in row_txt) + 1.0 + 1.6 + 1.0
    rh = 4.6
    title = "Control SCP DCT cells mapped to the nearest RNA centroid (cosine; fraction of each SCP label)"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    panels = [(False, "Atlas classes"), (True, "atlas_dct-derived split + atlas classes")]
    cw = 8.5                                         # atlas_dct 9.5; the 'Unknown (cluster)' row labels need the width
    gene_h = max(T.text_mm(c)[0] for c in (*T.DERIVED_GROUPS, *T.PUBLISHED_GROUPS)) + 1.5
    note = (f"Top {TOP_N} informative genes per RNA group (Wilcoxon one-vs-rest among the groups; q < {Q_MAX}, log2 FC >= "
            f"{LFC_MIN:g}, detected in >= {PCT_MIN:.0%} of the group and in >= {SCP_DET_MIN:.0%} of control cells); "
            "centroids z-scored per gene across the groups; SCP cells: z-scored ln(1 + intensity) (labels_core). Numbers: cells. "
            "deg = DE-rule labels (primary); cluster = submitted cluster labels. " + L.ATLAS_DCT_SPLIT_CAVEAT + " "
            + L.UNPAIRED_NOTE)
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
        ax.axhline(len(L.SCP_GROUPS), color="white", lw=1.5)
        ax.set_axis_off()
        for a in range(len(rows)):
            for b in range(len(order)):
                sh.text(x0 + (b + 0.5) * cw, y0 + (a + 0.5) * rh, f"{nn[a, b]}", ha="center", va="center_baseline",
                        color="white" if frac[a, b] > 0.55 else T.INK["primary"])
        for a, ((col, g), t_) in enumerate(zip(rows, row_txt)):
            yc = y0 + (a + 0.5) * rh
            sh.rect(x0 - 1.0 - 1.6, yc - 0.8, 1.6, 1.6, facecolor=L.SCP_PALETTE[g], edgecolor="none")
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

    # 2. top score and margin per SCP label (primary, atlas classes); panels stacked -------------------------------
    title = "Cosine top score and margin per control SCP label (atlas classes, primary variant)"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    note = (f"Dashed lines: low-confidence thresholds (score {SCORE_MIN}, margin {MARGIN_MIN}). Boxes: median and "
            "quartiles; whiskers 5th-95th percentile. deg = DE-rule labels (primary); cluster = submitted cluster labels.")
    note_text, note_h = T.note_block(note)
    ph, tick_h, gap, top_gap = 34.0, 2 * LH + 2.0, 4.0, 2.5
    H = T.M + title_h + top_gap + 2 * (ph + tick_h) + gap + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    for k, (qn, ylab, thr) in enumerate((("top_score", "Top cosine score", SCORE_MIN),
                                         ("margin", "Margin (top - second)", MARGIN_MIN))):
        ax = sh.ax(T.M + 14.0, T.M + title_h + top_gap + k * (ph + tick_h + gap), T.W - 2 * T.M - 16.0, ph)
        data, ticks, cols = [], [], []
        for col in L.LABEL_COLUMNS:
            for g in L.SCP_GROUPS:
                data.append(pc.loc[pc[col] == g, f"published_{qn}"].to_numpy(float))
                ticks.append(f"{g}\n{L.LABEL_SHORT[col]}")
                cols.append(L.SCP_PALETTE[g])
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.6, patch_artist=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, alpha=0.55, edgecolor=T.INK["secondary"], linewidth=0.5)
        for part in ("whiskers", "caps", "medians"):
            for l_ in bp[part]:
                l_.set(color=T.INK["secondary"], linewidth=0.6)
        ax.axhline(thr, color=T.INK["muted"], lw=0.6, ls="--")
        ax.axvline(len(L.SCP_GROUPS) + 0.5, color=T.INK["grid"], lw=0.6)
        ax.set_xticks(range(1, len(ticks) + 1))
        ax.set_xticklabels(ticks)
        ax.set_ylabel(ylab)
        T.style_axes(ax)
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9])   # as T.trim_ticks, y only
    sh.note(note_text, note_h)
    T.save_checked(run, sh, "mapping_scores", title, fig_rows)


if __name__ == "__main__":
    main()
