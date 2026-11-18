#!/usr/bin/env python
"""rat_segments step 4a: SCP scores on the Limbutara panels; PCT vs PST; TAL; nearest Limbutara segment.

Task rat_segments (authors, 30 Oct 2026).

What it does (pt_tal step 3 / step 4 methods, applied to the rat_segments panels)
--------------------------------------------------------------------
1. Per-cell scores for every panel (rat_segments_panels.py), label set (panels filtered with ``kidney_t15_label`` or
   ``kidney_pre_label``) and threshold (fc2 primary, fc1.5 sensitivity): mean of the canonical z-scored ln(1 +
   intensity) (``mean_z``, primary), scanpy ``score_genes`` on ln(1 + intensity) (ctrl 50, 25 bins, seed 42) and the
   detected fraction; gene sets: all panel proteins, without V3 markers, without ribosomal / MitoCarta proteins.
   Differences: S1/S2 - S3, S1 - S3, cTAL - mTAL.
2. PCT vs PST (pt_tal groupings: kidney_annotation labels with kidney_annotation panels, Fig 2A labels with Fig 2A panels, cells with the same PT
   label in both sets with kidney_annotation panels; the last is reported but not used as evidence, the response to the reviewers D3): AUC = P(PCT >
   PST) with bootstrap 95% CI (1,000 resamples, mean_z only), Hedges g, Mann-Whitney p (BH within grouping x
   threshold x gene set x method), OLS of the score on PCT with and without ln(protein IDs), Spearman rho of the
   score with ln(protein IDs). Reference row: the pt_tal atlas S1/S2 - S3 score (pt_tal per-cell table), same tests.
3. TAL cells: score summaries, rho with ln(protein IDs), rho between the cTAL and mTAL scores; Fig 2A MD (kidney_annotation
   Endothelial) vs Fig 2A TAL cells (descriptive), as pt_tal.
4. Nearest Limbutara segment per cell of the pt_tal union subsets (PT 356 -> S1 / S2 / S3; TAL 154 -> cTAL / mTAL):
   cosine to segment centroids on the union of the segment's one-vs-rest panel proteins (both label sets);
   centroid = (segment mean - mean over the samples of the segment's classes) / SD over those samples, on the
   centred log2 copies per cell; SCP ln(1 + intensity) z-scored across the subset (clip 10; sensitivity: canonical
   .X); low confidence: top score < 0.20 or margin < 0.05 (pt_tal). Confusion with both label sets; crosstab with the
   pt_tal atlas mapping (primary variant).

Inputs (read-only): kidney_object kidney object and labels; kidney_annotation labels; rat_segments panels, intake / panel tables; KTEA per-sample
                    file; V3 workbook; qc_mito MitoCarta annotation; pt_tal subsets_per_cell, scores_per_cell,
                    scores_pct_vs_pst, map_per_cell.
Outputs: results/tables/rat_segments/rat_segments_scores_per_cell.csv, rat_segments_score_sets.csv,
         rat_segments_scores_pct_vs_pst.csv, rat_segments_scores_tal.csv, rat_segments_map_per_cell.csv,
         rat_segments_map_summary.csv, rat_segments_map_confusion.csv, rat_segments_map_pt_agreement.csv,
         rat_segments_map_vs_t09.csv, rat_segments_scores_checks.csv, rat_segments_scores_figure_checks.csv,
         rat_segments_step_scoring_manifest.json;
         results/figures/rat_segments/rat_segments_scores_effects_pt, rat_segments_map_confusion (.pdf, .png)

Command (project root):  python scripts/rat_segments_scoring.py
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
import atlas_dct_lib as T8
import pt_tal_lib as T9
import rat_segments_lib as T

REF_ROW = "pt_tal atlas: S1/S2 - S3"
LN_IDS = "ln(protein IDs)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rat_segments step 4a: scores, PCT vs PST, TAL, nearest segment.")
    cc.add_common_args(parser)
    return parser.parse_args()


def cosine(cell_x: np.ndarray, cent: pd.DataFrame) -> pd.DataFrame:
    c = cent.to_numpy(float)
    num = cell_x @ c.T
    den = np.linalg.norm(cell_x, axis=1)[:, None] * np.linalg.norm(c, axis=1)[None, :]
    return pd.DataFrame(num / np.where(den > 0, den, np.nan), columns=cent.index)


def summarize(scores: pd.DataFrame, index) -> pd.DataFrame:
    s = scores.to_numpy()
    o = np.argsort(-np.nan_to_num(s, nan=-9), axis=1)
    ar = np.arange(len(s))
    out = pd.DataFrame({"top_class": scores.columns.to_numpy()[o[:, 0]], "top_score": s[ar, o[:, 0]],
                        "second_class": scores.columns.to_numpy()[o[:, 1]], "second_score": s[ar, o[:, 1]]}, index=index)
    out["margin"] = out["top_score"] - out["second_score"]
    out["low_confidence"] = (out["top_score"] < T9.SCORE_MIN) | (out["margin"] < T9.MARGIN_MIN) | out["top_score"].isna()
    return out


def test_row(v, a, b, pt, ln_ids, n_boot) -> dict:
    e = T9.effect_sizes(v[a], v[b], n_boot=n_boot)
    o = T9.ols_group(v[pt], a[pt], ln_ids[pt])
    r = stats.spearmanr(v[pt], ln_ids[pt], nan_policy="omit")
    return {**e, **o, "rho_score_ln_ids": float(r.statistic), "p_rho_score_ln_ids": float(r.pvalue)}


def main() -> None:
    import anndata as ad
    import scanpy as sc

    args = parse_args()
    run = cc.Run(T.TASK, "scoring", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, xz, ln1p, det = T9.load_scp(run, checks)
    panels = T.read_table(run, "panels")
    sub = T9.read_table(run, "subsets_per_cell", dtype={"cell_id": str})
    t09_pc = T9.read_table(run, "scores_per_cell", dtype={"cell_id": str}).set_index("cell_id").reindex(labels.index)
    t09_tests = T9.read_table(run, "scores_pct_vs_pst")
    t09_map = T9.read_table(run, "map_per_cell", dtype={"cell_id": str})
    v3 = T9.v3_marker_genes(run)
    fl = T9.protein_flags(run, adata.var_names.astype(str)).set_index("protein")
    ln_ids = np.log(labels["n_detected_proteins"].astype(float)).to_numpy()
    a_ln = ad.AnnData(X=ln1p.to_numpy(np.float32), obs=pd.DataFrame(index=labels.index),
                      var=pd.DataFrame(index=ln1p.columns))

    # 1. scores -----------------------------------------------------------------------------------------------------
    set_rows, scores = [], {}
    for lab_col in T9.LABELS:
        s = T9.LABEL_SHORT[lab_col]
        for thr in T.THRESHOLDS:
            for panel in T.CONTRASTS:
                prots = panels[(panels["label_column"] == lab_col) & (panels["threshold"] == thr)
                               & (panels["panel"] == panel)].sort_values("rank")["protein"].tolist()
                sets = {"all": prots, "no_v3": [p for p in prots if p not in v3],
                        "no_ribo_mito": [p for p in prots if not (fl.loc[p, "ribosomal"] or fl.loc[p, "mitocarta"])]}
                for gs, genes in sets.items():
                    set_rows.append({"label_column": lab_col, "threshold": thr, "panel": panel, "gene_set": gs,
                                     "n_proteins": len(genes), "proteins": ", ".join(genes)})
                    for method in T.METHODS:
                        key = T.score_col(s, thr, panel, gs, method)
                        if not genes:
                            scores[key] = np.full(len(labels), np.nan)
                        elif method == "mean_z":
                            scores[key] = xz[genes].mean(axis=1).to_numpy()
                        elif method == "detected_fraction":
                            scores[key] = det[genes].mean(axis=1).to_numpy()
                        else:
                            sc.tl.score_genes(a_ln, genes, ctrl_size=50, n_bins=25, random_state=T.SEED,
                                              score_name="tmp", use_raw=False)
                            scores[key] = a_ln.obs["tmp"].to_numpy(float)
            for dname, (p1, p2) in T.DIFFS.items():
                for gs in T.GENE_SETS:
                    for method in T.METHODS:
                        scores[T.score_col(s, thr, dname, gs, method)] = (scores[T.score_col(s, thr, p1, gs, method)]
                                                                          - scores[T.score_col(s, thr, p2, gs, method)])
    score_sets = pd.DataFrame(set_rows)
    per_cell = pd.concat([labels[["kidney_t15_label", "kidney_pre_label", "n_detected_proteins", "t15_cluster"]],
                          pd.DataFrame(scores, index=labels.index)], axis=1)
    masks = T9.union_masks(labels)
    per_cell.insert(3, "in_pt_union", masks["PT"].to_numpy())
    per_cell.insert(4, "in_tal_union", masks["TAL"].to_numpy())
    per_cell.insert(5, "md_fig2a", masks["MD_pre"].to_numpy())
    for r in score_sets[score_sets["gene_set"] != "all"].itertuples():
        if r.n_proteins == 0:
            checks.add(f"empty gene set, no score: {T9.LABEL_SHORT[r.label_column]} {r.threshold} {r.panel} {r.gene_set}",
                       "0 proteins", None)
    checks.add("all-protein panels non-empty (every label set x threshold x panel)",
               int((score_sets[score_sets["gene_set"] == "all"]["n_proteins"] > 0).sum()),
               bool((score_sets[score_sets["gene_set"] == "all"]["n_proteins"] > 0).all()))

    # 2. PCT vs PST -----------------------------------------------------------------------------------------------------
    groupings = T9.pt_groupings(labels)
    test_rows = []
    for g, lab in groupings.items():
        s = T9.LABEL_SHORT[T.GROUPING_PANELS[g]]
        a, b = lab.eq("PCT").to_numpy(), lab.eq("PST").to_numpy()
        pt = a | b
        for thr in T.THRESHOLDS:
            for score in T.PT_SCORES:
                for gs in T.GENE_SETS:
                    for method in T.METHODS:
                        v = scores[T.score_col(s, thr, score, gs, method)]
                        if np.all(~np.isfinite(v[pt])):
                            continue
                        test_rows.append({"grouping": g, "grouping_text": T9.GROUPING_TEXT[g], "panels_from": s,
                                          "threshold": thr, "score": score, "gene_set": gs, "method": method,
                                          **test_row(v, a, b, pt, ln_ids, T.N_BOOT if method == "mean_z" else 0)})
        # reference rows (threshold "-"): the pt_tal atlas S1/S2 - S3 score (mean_z, all), and ln(protein IDs)
        v = t09_pc[f"score__{s}__S1S2_minus_S3__all__mean_z"].to_numpy(float)
        test_rows.append({"grouping": g, "grouping_text": T9.GROUPING_TEXT[g], "panels_from": s, "threshold": "-",
                          "score": REF_ROW, "gene_set": "all", "method": "mean_z",
                          **test_row(v, a, b, pt, ln_ids, T.N_BOOT)})
        e = T9.effect_sizes(ln_ids[a], ln_ids[b], n_boot=T.N_BOOT)
        test_rows.append({"grouping": g, "grouping_text": T9.GROUPING_TEXT[g], "panels_from": s, "threshold": "-",
                          "score": LN_IDS, "gene_set": "-", "method": "-", **e})
        t9 = t09_tests[(t09_tests["grouping"] == g) & (t09_tests["score"] == "S1/S2 - S3")
                       & (t09_tests["gene_set"] == "all") & (t09_tests["method"] == "mean_z")].iloc[0]
        mine = test_rows[-2]
        checks.add(f"{T9.GROUPING_TEXT[g]}: pt_tal S1/S2 - S3 AUC / adjusted coef reproduced from the pt_tal per-cell table",
                   f"{mine['auc']:.4f} vs {t9.auc:.4f}; {mine['coef_sd_adjusted_ln_ids']:+.3f} vs "
                   f"{t9.coef_sd_adjusted_ln_ids:+.3f}",
                   bool(abs(mine["auc"] - t9.auc) < 1e-9 and abs(mine["coef_sd_adjusted_ln_ids"]
                                                                 - t9.coef_sd_adjusted_ln_ids) < 1e-9))
    tests = pd.DataFrame(test_rows)
    tests = cc.add_bh(tests, ["grouping", "threshold", "gene_set", "method"], p_col="p_mannwhitney",
                      out_col="q_bh_mannwhitney")
    for g in groupings:
        r = tests[(tests["grouping"] == g) & (tests["score"] == "S1/S2 - S3") & (tests["threshold"] == T.PRIMARY_THR)
                  & (tests["gene_set"] == "all") & (tests["method"] == "mean_z")].iloc[0]
        checks.add(f"{T9.GROUPING_TEXT[g]}: n PCT / PST; Limbutara S1/S2 - S3 (fc2, mean_z) AUC [95% CI]; adjusted coef "
                   f"(SD); rho with ln(IDs)",
                   f"{r.n_1} / {r.n_2}; {r.auc:.3f} [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]; "
                   f"{r.coef_sd_adjusted_ln_ids:+.2f} (p {r.p_adjusted_ln_ids:.1e}); {r.rho_score_ln_ids:+.2f}", None)

    # 3. TAL ----------------------------------------------------------------------------------------------------------
    tal_rows = []
    md, ptal = masks["MD_pre"].to_numpy(), labels["kidney_pre_label"].eq("TAL").to_numpy()
    for lab_col in T9.LABELS:
        s = T9.LABEL_SHORT[lab_col]
        tal = labels[lab_col].eq(T9.TAL_LABEL).to_numpy()
        for thr in T.THRESHOLDS:
            for score in T.TAL_SCORES:
                for gs in T.GENE_SETS:
                    for method in T.METHODS:
                        v = scores[T.score_col(s, thr, score, gs, method)]
                        if np.all(~np.isfinite(v[tal])):
                            continue
                        r = stats.spearmanr(v[tal], ln_ids[tal], nan_policy="omit")
                        rec = {"label_column": lab_col, "panels_from": s, "threshold": thr, "score": score,
                               "gene_set": gs, "method": method, "n_tal": int(tal.sum()),
                               "median_tal": float(np.nanmedian(v[tal])),
                               "q25_tal": float(np.nanquantile(v[tal], 0.25)),
                               "q75_tal": float(np.nanquantile(v[tal], 0.75)),
                               "rho_ln_ids_tal": float(r.statistic), "p_rho_ln_ids_tal": float(r.pvalue)}
                        if score in T.TAL_PANELS:
                            other = [x for x in T.TAL_PANELS if x != score][0]
                            w = scores[T.score_col(s, thr, other, gs, method)]
                            if np.any(np.isfinite(w[tal])):
                                r2 = stats.spearmanr(v[tal], w[tal], nan_policy="omit")
                                rec.update(rho_with_other_tal_score=float(r2.statistic), p_rho_other=float(r2.pvalue))
                        e = T9.effect_sizes(v[md], v[ptal])
                        rec.update({f"md_vs_fig2a_tal_{k}": e[k] for k in ("n_1", "n_2", "median_1", "median_2", "auc",
                                                                           "hedges_g", "p_mannwhitney")})
                        tal_rows.append(rec)
    tal_tab = pd.DataFrame(tal_rows)
    for r in tal_tab[(tal_tab["gene_set"] == "all") & (tal_tab["method"] == "mean_z")
                     & (tal_tab["threshold"] == T.PRIMARY_THR)].itertuples():
        checks.add(f"TAL cells ({T9.LABEL_DISPLAY[r.label_column]}), {r.score} (fc2, mean_z): median [IQR]; rho with "
                   f"ln(IDs); Fig 2A MD vs TAL AUC",
                   f"{r.median_tal:.2f} [{r.q25_tal:.2f}, {r.q75_tal:.2f}]; {r.rho_ln_ids_tal:+.2f}; "
                   f"{r.md_vs_fig2a_tal_auc:.2f}", None)

    # 4. nearest Limbutara segment ------------------------------------------------------------------------------------
    ann, vals, seg = T.read_samples()
    T.check_sources(run)
    lgc, _, _ = T.centred_log2(vals, seg)
    map_rows, map_conf, map_sum, agree_rows, vs_rows = [], [], [], [], []
    mapped = {}
    t09_pv = t09_map[t09_map["variant"] == "primary|panels|subset_z"].set_index("cell_id")
    t09_to_lim = {"PTS1": "S1/S2", "PTS2": "S1/S2", "PTS3": "S3", "PTS3T2": "S3", "CTAL": "cTAL", "MTAL": "mTAL"}
    for thr in T.THRESHOLDS:
        for sg, classes in T.MAP_CLASSES.items():
            cells = sub[sub["segment"] == sg]["cell_id"].tolist()
            pr = panels[(panels["threshold"] == thr) & panels["panel"].isin(classes)].sort_values(["panel", "rank"])
            prot = list(dict.fromkeys(pr["protein"]))
            rat = pr.drop_duplicates("protein").set_index("protein").loc[prot, "rat_id"].astype(str).to_numpy()
            samp = seg.index[seg.isin(classes)]
            x_r = lgc.loc[rat, samp]
            if x_r.isna().all(axis=1).any():
                raise ValueError("a mapping protein has no quantified sample in the segment's classes")
            mu, sd_ = x_r.mean(axis=1).to_numpy(), x_r.std(axis=1, ddof=1).to_numpy()
            cent = pd.DataFrame([(lgc.loc[rat, seg.index[seg == c]].mean(axis=1).to_numpy() - mu)
                                 / np.where(sd_ > 0, sd_, 1.0) for c in classes], index=list(classes), columns=prot)
            for scp_matrix in ("subset_z", "canonical"):
                x = (cc.scale_like_scanpy(ln1p.loc[cells, prot].to_numpy()) if scp_matrix == "subset_z"
                     else xz.loc[cells, prot].to_numpy())
                sc_ = cosine(x, cent)
                pred = summarize(sc_, pd.Index(cells, name="cell_id"))
                variant = f"{thr}|panels|{scp_matrix}"
                mapped[(sg, variant)] = pred
                rho = stats.spearmanr(pred["top_score"], np.log(labels.loc[cells, "n_detected_proteins"].astype(float)),
                                      nan_policy="omit")
                map_sum.append({"segment": sg, "classes": ";".join(classes), "threshold": thr, "variant": variant,
                                "primary": thr == T.PRIMARY_THR and scp_matrix == "subset_z",
                                "n_informative_proteins": len(prot), "n_cells": len(cells),
                                "frac_low_confidence": float(pred["low_confidence"].mean()),
                                "median_top_score": float(pred["top_score"].median()),
                                "median_margin": float(pred["margin"].median()),
                                "rho_top_score_ln_ids": float(rho.statistic), "p_rho": float(rho.pvalue),
                                **{f"frac_top_{c}": float((pred["top_class"] == c).mean()) for c in classes}})
                out = pred.copy()
                out.insert(0, "variant", variant)
                out.insert(0, "segment", sg)
                for c in classes:
                    out[f"score_{c}"] = sc_[c].to_numpy()
                map_rows.append(out.reset_index())
                for lab_col in T9.LABELS:
                    for l_, gidx in labels.loc[cells, lab_col].groupby(labels.loc[cells, lab_col]).groups.items():
                        p_ = pred.loc[gidx]
                        for c in classes:
                            map_conf.append({"segment": sg, "variant": variant, "label_column": lab_col, "label": l_,
                                             "n_label": len(gidx), "class": c,
                                             "n_top": int((p_["top_class"] == c).sum()),
                                             "n_top_confident": int(((p_["top_class"] == c) & ~p_["low_confidence"]).sum()),
                                             "frac_top": float((p_["top_class"] == c).mean())})
                # crosstab with the pt_tal atlas mapping (primary pt_tal variant), atlas classes collapsed to S1/S2, S3, cTAL, mTAL
                t9c = t09_pv[t09_pv["segment"] == sg].reindex(cells)["top_class"].dropna().map(t09_to_lim)
                lim = pred.loc[t9c.index, "top_class"].map(lambda c: "S1/S2" if c in ("S1", "S2") else c)
                ct = pd.crosstab(t9c.rename("t09_atlas"), lim.rename("limbutara"))
                for i_ in ct.index:
                    for j_ in ct.columns:
                        vs_rows.append({"segment": sg, "variant": variant, "t09_atlas_top": i_, "limbutara_top": j_,
                                        "n_cells": int(ct.loc[i_, j_])})
                vs_rows.append({"segment": sg, "variant": variant, "t09_atlas_top": "agreement", "limbutara_top": "-",
                                "n_cells": int((t9c == lim).sum()), "frac": float((t9c == lim).mean()),
                                "cohen_kappa": float(_kappa(t9c, lim))})
            checks.add(f"{sg} mapping ({thr}): informative proteins; low confidence (subset z)",
                       f"{len(prot)}; {mapped[(sg, f'{thr}|panels|subset_z')]['low_confidence'].mean():.2f}", None)
    map_conf = pd.DataFrame(map_conf)
    for thr in T.THRESHOLDS:
        pred = mapped[("PT", f"{thr}|panels|subset_z")]
        for g, lab in groupings.items():
            l_ = lab.reindex(pred.index)
            for k, cls_ in T.EXPECTED_MAP.items():
                m = l_.eq(k)
                top = pred.loc[m, "top_class"]
                conf = ~pred.loc[m, "low_confidence"]
                agree_rows.append({"threshold": thr, "grouping": g, "label": k, "n_cells": int(m.sum()),
                                   "expected_classes": ";".join(cls_), "frac_top_expected": float(top.isin(cls_).mean()),
                                   "frac_top_expected_confident": float((top.isin(cls_) & conf).mean()),
                                   "frac_low_confidence": float((~conf).mean())})
    agree = pd.DataFrame(agree_rows)
    for r in agree[agree["threshold"] == T.PRIMARY_THR].itertuples():
        checks.add(f"PT mapping (fc2), {T9.GROUPING_TEXT[r.grouping]}: {r.label} cells nearest {r.expected_classes} "
                   "(all; confident)", f"{r.frac_top_expected:.2f}; {r.frac_top_expected_confident:.2f} of {r.n_cells}",
                   None)
    vs = pd.DataFrame(vs_rows)
    for r in vs[(vs["t09_atlas_top"] == "agreement") & vs["variant"].str.endswith("subset_z")].itertuples():
        checks.add(f"{r.segment} ({r.variant}): nearest Limbutara vs nearest atlas class (collapsed) agree; kappa",
                   f"{r.frac:.2f}; {r.cohen_kappa:.2f}", None)

    fig_rows = []
    figures(run, tests, map_conf, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(per_cell.reset_index(), "scores_per_cell")
    run.table(score_sets, "score_sets")
    run.table(tests, "scores_pct_vs_pst")
    run.table(tal_tab, "scores_tal")
    run.table(pd.concat(map_rows, ignore_index=True), "map_per_cell")
    run.table(pd.DataFrame(map_sum), "map_summary")
    run.table(map_conf, "map_confusion")
    run.table(agree, "map_pt_agreement")
    run.table(vs, "map_vs_t09")
    run.table(figs, "scores_figure_checks")
    run.table(checks.frame(), "scores_checks")
    run.finish({"methods": list(T.METHODS), "gene_sets": list(T.GENE_SETS), "n_boot": T.N_BOOT,
                "thresholds": T.THRESHOLDS, "primary_threshold": T.PRIMARY_THR,
                "score_genes": {"ctrl_size": 50, "n_bins": 25, "random_state": T.SEED, "matrix": "ln(1 + intensity)"},
                "groupings": T9.GROUPING_TEXT, "mapping": {"score_min": T9.SCORE_MIN, "margin_min": T9.MARGIN_MIN,
                                                           "centroids": "(segment mean - mean over the class samples) / "
                                                                        "SD over the class samples; centred log2 copies"},
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see rat_segments_scores_checks.csv.")


def _kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    ct = pd.crosstab(a, b).reindex(index=cats, columns=cats, fill_value=0).to_numpy(float)
    n = ct.sum()
    po = np.trace(ct) / n
    pe = float((ct.sum(axis=1) * ct.sum(axis=0)).sum()) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def figures(run, tests, map_conf, fig_rows) -> None:
    T9.setup()
    LH = T9.LH

    # 1. effects (PT), fc2 ----------------------------------------------------------------------------------------
    gcol = {"t15": "#184f95", "pre": "#6da7ec"}
    gtxt = {"t15": "kidney_annotation labels (kidney_annotation panels)", "pre": "Fig 2A labels (Fig 2A panels)"}
    rows = [*T.PT_SCORES, REF_ROW, LN_IDS]
    title = "PCT vs PST along the Limbutara segment scores: AUC and depth-adjusted difference"
    note = ("AUC = P(score of a PCT cell > score of a PST cell); bars: bootstrap 95% CI (1,000 resamples). Right: OLS "
            "coefficient of PCT (vs PST) in score-SD units, adjusted for ln(protein IDs), with 95% CI. Scores: mean z "
            "over the panel proteins (Limbutara et al. 2020 rat segment panels, 2-fold rule; all panel proteins). "
            "Reference rows: the pt_tal atlas S1/S2 - S3 score and ln(protein IDs) itself (no adjustment). Cells with the "
            "same PT label in both sets: table only.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    rh = 5.0
    lw = max(T8.text_mm(r)[0] for r in rows) + 2.0
    leg = T9.legend_rows([(gtxt[g], gcol[g]) for g in gcol], 150)
    pw = (180 - 2 * T8.M - lw - 12.0) / 2
    H = T8.M + title_h + LH + 1.0 + len(rows) * rh + 9.0 + len(leg) * (LH + 0.4) + 2.0 + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    y0 = T8.M + title_h + LH + 1.0
    sel = tests[tests["threshold"].isin([T.PRIMARY_THR, "-"]) & tests["gene_set"].isin(["all", "-"])
                & tests["method"].isin(["mean_z", "-"]) & tests["grouping"].isin(list(gcol))]
    lim = np.nanmax(np.abs(sel[["ci_sd_low_adjusted_ln_ids", "ci_sd_high_adjusted_ln_ids"]].to_numpy()))
    lim = float(np.ceil(lim * 2) / 2)
    for k, (what, xl) in enumerate((("auc", "AUC (PCT vs PST)"), ("coef", "PCT coefficient (SD), adj. ln(IDs)"))):
        x0 = T8.M + lw + k * (pw + 12.0)
        sh.text(x0, T8.M + title_h, xl, va="top")
        ax = sh.ax(x0, y0, pw, len(rows) * rh)
        for i, r in enumerate(rows):
            for j, g in enumerate(gcol):
                t_ = sel[(sel["grouping"] == g) & (sel["score"] == r)]
                if t_.empty:
                    continue
                t_ = t_.iloc[0]
                yy = i + 0.5 + (j - 0.5) * 0.3
                if what == "auc":
                    ax.plot([t_.auc_ci_low, t_.auc_ci_high], [yy, yy], color=gcol[g], lw=0.8)
                    ax.plot(t_.auc, yy, "o", color=gcol[g], ms=2.6)
                elif r != LN_IDS:
                    ax.plot([t_.ci_sd_low_adjusted_ln_ids, t_.ci_sd_high_adjusted_ln_ids], [yy, yy], color=gcol[g],
                            lw=0.8)
                    ax.plot(t_.coef_sd_adjusted_ln_ids, yy, "o", color=gcol[g], ms=2.6)
        ax.set_ylim(len(rows), 0)
        ax.set_yticks([])
        ax.axvline(0.5 if what == "auc" else 0.0, color=T.INK["muted"], lw=0.5, ls="--")
        if what == "auc":
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        else:
            ax.set_xlim(-lim, lim)
            ax.set_xticks(np.linspace(-lim, lim, 5))
        for i in range(1, len(rows)):
            ax.axhline(i, color=T.INK["grid"], lw=0.4 if i != len(T.PT_SCORES) else 0.8)
        T8.style_axes(ax)
        ax.spines["left"].set_visible(False)
    for i, r in enumerate(rows):
        sh.text(T8.M + lw - 1.0, y0 + (i + 0.5) * rh, r, ha="right", va="center_baseline")
    T9.draw_legend(sh, T8.M + lw, y0 + len(rows) * rh + 9.0, leg)
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "scores_effects_pt", title, fig_rows)

    # 2. mapping confusion (fc2, subset z) --------------------------------------------------------------------------
    cmap = T8.sequential_cmap()
    pv = f"{T.PRIMARY_THR}|panels|subset_z"
    title = "Nearest Limbutara segment centroid per SCP label (fraction of the label; 2-fold panels)"
    note = ("Fill and numbers: cells of the label whose top segment is the column (in brackets: confident, score >= "
            f"{T9.SCORE_MIN} and margin >= {T9.MARGIN_MIN}). PT union subset: S1 / S2 / S3; TAL union subset: cTAL / "
            "mTAL. Centroids: rat segment means of the centred log2 copies per cell, standardized over the samples of "
            "the segment's classes; SCP: ln(1 + intensity) z-scored across the subset. Labels with >= 5 cells.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    cw = 14.0
    rows_all = []
    for sg in ("PT", "TAL"):
        for lab_col in T9.LABELS:
            c = map_conf[(map_conf["segment"] == sg) & (map_conf["variant"] == pv)
                         & (map_conf["label_column"] == lab_col) & (map_conf["n_label"] >= 5)]
            for l_ in T9.order_labels(c["label"].unique()):
                rows_all.append((sg, lab_col, l_))
    rl = [f"{l_} ({T9.GROUPING_SHORT[T9.LABEL_SHORT[c_]]}, {s})" for s, c_, l_ in rows_all]
    lw = max(T8.text_mm(r)[0] for r in rl) + 2.0
    n_pt = sum(1 for r in rows_all if r[0] == "PT")
    H = T8.M + title_h + LH + 1.0 + max(n_pt, len(rows_all) - n_pt) * 4.6 + 4.0 + 6.0 + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    y0 = T8.M + title_h + LH + 1.0
    for sg, x_off in (("PT", 0.0), ("TAL", 3 * cw + 16.0)):
        classes = list(T.MAP_CLASSES[sg])
        idx = [i for i, r in enumerate(rows_all) if r[0] == sg]
        counts, confn = [], []
        for i in idx:
            s, c_, l_ = rows_all[i]
            c = map_conf[(map_conf["segment"] == s) & (map_conf["variant"] == pv) & (map_conf["label_column"] == c_)
                         & (map_conf["label"] == l_)].set_index("class")
            counts.append([int(c.loc[k, "n_top"]) for k in classes])
            confn.append([int(c.loc[k, "n_top_confident"]) for k in classes])
        counts, confn = np.array(counts), np.array(confn)
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        x0 = T8.M + lw + x_off
        ax = sh.ax(x0, y0, len(classes) * cw, len(idx) * 4.6)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=(0, len(classes), len(idx), 0))
        ax.set_axis_off()
        for a in range(len(idx)):
            for b in range(len(classes)):
                sh.text(x0 + (b + 0.5) * cw, y0 + (a + 0.5) * 4.6, f"{counts[a, b]} ({confn[a, b]})", ha="center",
                        va="center_baseline", color="white" if frac[a, b] > 0.55 else T.INK["primary"])
        for a, i in enumerate(idx):
            if sg == "PT":
                sh.text(x0 - 1.0, y0 + (a + 0.5) * 4.6, rl[i], ha="right", va="center_baseline")
            else:
                sh.text(x0 + len(classes) * cw + 1.0, y0 + (a + 0.5) * 4.6, rl[i], ha="left", va="center_baseline")
        for b, k in enumerate(classes):
            sh.text(x0 + (b + 0.5) * cw, y0 + len(idx) * 4.6 + 0.8, k, ha="center", va="top")
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "map_confusion", title, fig_rows)


if __name__ == "__main__":
    main()
