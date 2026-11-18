#!/usr/bin/env python
"""pt_tal step 3: SCP scores on the atlas-derived subtype panels; PCT vs PST along the S1/S2 vs S3 scores.

Analysis plan, step 3.

Scores (all 712 cells; per label set with that label set's panels from step 1)
-------------------------------------------------------------------------------
- Methods: ``mean_z`` (primary) = mean of the canonical .X (per-protein z-scored ln(1 + intensity), missing = 0) over
  the panel proteins; ``score_genes`` = scanpy score_genes on ln(1 + intensity) (control genes, 25 bins, 50 controls,
  seed 42); ``detected_fraction`` = fraction of the panel proteins detected (intensity > 0).
- Gene sets: the panel (``all``); the panel without kidney V3 / kidney_annotation endothelial-row markers (``no_v3``; the PCT / PST
  labels were assigned with V3 markers); the panel without ribosomal and MitoCarta3.0 proteins (``no_ribo_mito``;
  rule of kidney_annotation_atlas_reference.py; added in this step because the S1/S2 panel holds ribosomal proteins).
- Panels: PT one-vs-rest (PTS1, PTS2, PTS3, PTS3T2; PTS2 is empty), grouped (S1/S2, S3); TAL (CTAL, MTAL); MD.
  Differences: S1/S2 - S3 and PTS1 - PTS3 (PT); CTAL - MTAL (TAL).

Tests
-----
- PCT vs PST for three groupings: kidney_annotation labels (kidney_annotation panels), Fig 2A labels (Fig 2A panels) and cells with the same PT
  label in both sets (kidney_annotation panels). Effect sizes: AUC = P(PCT score > PST score) (+ bootstrap 95% CI, 1,000 resamples,
  for mean_z), Cliff's delta, Hedges' g, Mann-Whitney p (BH within grouping x gene set x method); OLS score ~ PCT +
  ln(protein IDs) (the project notes: depth is a confounder): group coefficient in score-SD units, unadjusted and adjusted.
  Spearman rho of each score with ln(protein IDs) within the grouping's PT cells. ln(protein IDs) itself is
  compared as a reference row.
- TAL: no SCP TAL subgroups exist; TAL cells' score distributions, rho with ln(protein IDs) and between the CTAL and
  MTAL scores; Fig 2A MD (19 cells; kidney_annotation Endothelial) vs Fig 2A TAL cells on the TAL / MD panel scores (descriptive).

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation labels; KIDNEY_markers_V3.xlsx; qc_mito MitoCarta
                    annotation; pt_tal step-0/1 tables (subsets_per_cell, panels).
Outputs: results/tables/pt_tal/pt_tal_scores_per_cell.csv, pt_tal_score_sets.csv,
         pt_tal_scores_pct_vs_pst.csv, pt_tal_scores_tal.csv, pt_tal_scores_checks.csv;
         results/figures/pt_tal/pt_tal_scores_umap_{pt,tal}_{t15,pre}, pt_tal_scores_effects_pt,
         pt_tal_scores_by_label (.pdf/.png)

Command (project root):  python scripts/pt_tal_scoring.py
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
import pt_tal_lib as T

METHODS = ("mean_z", "score_genes", "detected_fraction")
GENE_SETS = ("all", "no_v3", "no_ribo_mito")
GENE_SET_TEXT = {"all": "panel", "no_v3": "panel without V3 markers", "no_ribo_mito": "panel without ribosomal / MitoCarta"}
N_BOOT = 1000
DIFFS = {"S1/S2 - S3": ("S1/S2", "S3"), "PTS1 - PTS3": ("PTS1", "PTS3"), "CTAL - MTAL": ("CTAL", "MTAL")}
PT_SCORES = (*T.PT_ONE_VS_REST, *T.PT_GROUPED, "S1/S2 - S3", "PTS1 - PTS3")
TAL_SCORES = (*T.TAL_PANELS, "CTAL - MTAL", "MD")
GROUPING_PANELS = {"t15": "kidney_t15_label", "pre": "kidney_pre_label", T.CONSENSUS: "kidney_t15_label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pt_tal step 3: SCP subtype scores and PCT vs PST effect sizes.")
    cc.add_common_args(parser)
    return parser.parse_args()


def col(s: str, score: str, gs: str, method: str) -> str:
    return f"score__{s}__{score.replace('/', '').replace(' - ', '_minus_')}__{gs}__{method}"


def main() -> None:
    import anndata as ad
    import scanpy as sc

    args = parse_args()
    run = cc.Run(T.TASK, "scoring", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, xz, ln1p, det = T.load_scp(run, checks)
    panels = T.read_table(run, "panels")
    sub = T.read_table(run, "subsets_per_cell", dtype={"cell_id": str})
    v3 = T.v3_marker_genes(run)
    fl = T.protein_flags(run, adata.var_names.astype(str)).set_index("protein")
    ln_ids = np.log(labels["n_detected_proteins"].astype(float))
    a_ln = ad.AnnData(X=ln1p.to_numpy(np.float32), obs=pd.DataFrame(index=labels.index),
                      var=pd.DataFrame(index=ln1p.columns))

    # score sets and per-cell scores ------------------------------------------------------------------------
    set_rows, scores = [], {}
    for lab_col in T.LABELS:
        s = T.LABEL_SHORT[lab_col]
        for panel in T.CONTRASTS:
            prots = panels[(panels["label_column"] == lab_col) & (panels["panel"] == panel)].sort_values("rank")
            prots = prots["protein"].tolist()
            sets = {"all": prots, "no_v3": [p for p in prots if p not in v3],
                    "no_ribo_mito": [p for p in prots if not (fl.loc[p, "ribosomal"] or fl.loc[p, "mitocarta"])]}
            for gs, genes in sets.items():
                set_rows.append({"label_column": lab_col, "panel": panel, "gene_set": gs, "n_proteins": len(genes),
                                 "proteins": ", ".join(genes)})
                for method in METHODS:
                    key = col(s, panel, gs, method)
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
        for dname, (p1, p2) in DIFFS.items():
            for gs in GENE_SETS:
                for method in METHODS:
                    scores[col(s, dname, gs, method)] = scores[col(s, p1, gs, method)] - scores[col(s, p2, gs, method)]
    score_sets = pd.DataFrame(set_rows)
    per_cell = pd.concat([labels[["kidney_t15_label", "kidney_pre_label", "n_detected_proteins", "t15_cluster"]],
                          pd.DataFrame(scores, index=labels.index)], axis=1)
    masks = T.union_masks(labels)
    per_cell.insert(3, "in_pt_union", masks["PT"].to_numpy())
    per_cell.insert(4, "in_tal_union", masks["TAL"].to_numpy())
    per_cell.insert(5, "md_fig2a", masks["MD_pre"].to_numpy())
    for r in score_sets[score_sets["gene_set"] == "all"].itertuples():
        if r.n_proteins == 0:
            checks.add(f"empty panel, no score: {T.LABEL_SHORT[r.label_column]} {r.panel}", "0 proteins", None)

    # PCT vs PST ---------------------------------------------------------------------------------------------------
    groupings = T.pt_groupings(labels)
    test_rows = []
    for g, lab in groupings.items():
        s = T.LABEL_SHORT[GROUPING_PANELS[g]]
        a, b = lab.eq("PCT").to_numpy(), lab.eq("PST").to_numpy()
        pt = a | b
        for score in (*PT_SCORES, "ln(protein IDs)"):
            for gs in (GENE_SETS if score != "ln(protein IDs)" else ("-",)):
                for method in (METHODS if score != "ln(protein IDs)" else ("-",)):
                    v = ln_ids.to_numpy() if score == "ln(protein IDs)" else scores[col(s, score, gs, method)]
                    if np.all(~np.isfinite(v[pt])):
                        continue
                    e = T.effect_sizes(v[a], v[b], n_boot=N_BOOT if method in ("mean_z", "-") else 0)
                    o = T.ols_group(v[pt], a[pt], ln_ids.to_numpy()[pt]) if score != "ln(protein IDs)" else {}
                    r = stats.spearmanr(v[pt], ln_ids.to_numpy()[pt]) if score != "ln(protein IDs)" else None
                    test_rows.append({"grouping": g, "grouping_text": T.GROUPING_TEXT[g], "panels_from": s,
                                      "score": score, "gene_set": gs, "method": method, **e, **o,
                                      "rho_score_ln_ids": float(r.statistic) if r is not None else np.nan,
                                      "p_rho_score_ln_ids": float(r.pvalue) if r is not None else np.nan})
    tests = pd.DataFrame(test_rows)
    tests = cc.add_bh(tests, ["grouping", "gene_set", "method"], p_col="p_mannwhitney", out_col="q_bh_mannwhitney")
    for g in groupings:
        r = tests[(tests["grouping"] == g) & (tests["score"] == "S1/S2 - S3") & (tests["gene_set"] == "all")
                  & (tests["method"] == "mean_z")].iloc[0]
        rd = tests[(tests["grouping"] == g) & (tests["score"] == "ln(protein IDs)")].iloc[0]
        checks.add(f"{T.GROUPING_TEXT[g]}: n PCT / PST; S1/S2 - S3 (mean_z) AUC [95% CI]; adjusted coef (SD); "
                   f"rho with ln(IDs); ln(IDs) AUC",
                   f"{r.n_1} / {r.n_2}; {r.auc:.3f} [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]; "
                   f"{r.coef_sd_adjusted_ln_ids:+.2f} (p {r.p_adjusted_ln_ids:.1e}); {r.rho_score_ln_ids:+.2f}; "
                   f"{rd.auc:.3f}", None)

    # TAL -------------------------------------------------------------------------------------------------------------
    tal_rows = []
    for lab_col in T.LABELS:
        s = T.LABEL_SHORT[lab_col]
        tal = labels[lab_col].eq(T.TAL_LABEL).to_numpy()
        for score in TAL_SCORES:
            for gs in GENE_SETS:
                for method in METHODS:
                    v = scores[col(s, score, gs, method)]
                    if np.all(~np.isfinite(v[tal])):
                        continue
                    r = stats.spearmanr(v[tal], ln_ids.to_numpy()[tal])
                    rec = {"label_column": lab_col, "panels_from": s, "score": score, "gene_set": gs, "method": method,
                           "n_tal": int(tal.sum()), "median_tal": float(np.median(v[tal])),
                           "q25_tal": float(np.quantile(v[tal], 0.25)), "q75_tal": float(np.quantile(v[tal], 0.75)),
                           "rho_ln_ids_tal": float(r.statistic), "p_rho_ln_ids_tal": float(r.pvalue)}
                    if score in T.TAL_PANELS:
                        other = [x for x in T.TAL_PANELS if x != score][0]
                        w = scores[col(s, other, gs, method)]
                        if np.any(np.isfinite(w[tal])):
                            r2 = stats.spearmanr(v[tal], w[tal])
                            rec.update(rho_with_other_tal_score=float(r2.statistic), p_rho_other=float(r2.pvalue))
                    md, ptal = masks["MD_pre"].to_numpy(), labels["kidney_pre_label"].eq("TAL").to_numpy()
                    e = T.effect_sizes(v[md], v[ptal])
                    rec.update({f"md_vs_fig2a_tal_{k}": e[k] for k in ("n_1", "n_2", "median_1", "median_2", "auc",
                                                                       "hedges_g", "p_mannwhitney")})
                    tal_rows.append(rec)
    tal_tab = pd.DataFrame(tal_rows)
    for r in tal_tab[(tal_tab["gene_set"] == "all") & (tal_tab["method"] == "mean_z")
                     & (tal_tab["label_column"] == T.PRIMARY)].itertuples():
        checks.add(f"TAL cells (kidney_annotation panels), {r.score}: median [IQR]; rho with ln(IDs); Fig 2A MD vs TAL AUC",
                   f"{r.median_tal:.2f} [{r.q25_tal:.2f}, {r.q75_tal:.2f}]; {r.rho_ln_ids_tal:+.2f}; "
                   f"{r.md_vs_fig2a_tal_auc:.2f}", None)

    fig_rows = []
    figures(run, adata, labels, per_cell, sub, tests, tal_tab, score_sets, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(per_cell.reset_index(), "scores_per_cell")
    run.table(score_sets, "score_sets")
    run.table(tests, "scores_pct_vs_pst")
    run.table(tal_tab, "scores_tal")
    run.table(figs, "scores_figure_checks")
    run.table(checks.frame(), "scores_checks")
    run.finish({"methods": list(METHODS), "gene_sets": list(GENE_SETS), "n_boot": N_BOOT,
                "score_genes": {"ctrl_size": 50, "n_bins": 25, "random_state": T.SEED, "matrix": "ln(1 + intensity)"},
                "groupings": T.GROUPING_TEXT, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see pt_tal_scores_checks.csv.")


def figures(run, adata, labels, per_cell, sub, tests, tal_tab, score_sets, fig_rows) -> None:
    plt = T.setup()
    from matplotlib import colormaps

    LH = T.LH
    colors = T.label_colors(adata)
    viridis = colormaps["viridis"]
    div = T8.diverging_cmap()
    pc = per_cell.copy()
    pc.index = pc.index.astype(str)

    # 1. UMAPs ---------------------------------------------------------------------------------------------------
    order_labels = T.order_labels
    for seg, seg_scores in (("PT", ("PTS1", "PTS2", "PTS3", "PTS3T2", "S1/S2", "S3", "S1/S2 - S3")),
                            ("TAL", ("CTAL", "MTAL", "CTAL - MTAL", "MD"))):
        d = sub[sub["segment"] == seg].set_index("cell_id")
        xy = d[["umap1_primary", "umap2_primary"]].to_numpy(float)
        for lab_col in T.LABELS:
            s = T.LABEL_SHORT[lab_col]
            items = [("label", c) for c in T.LABELS] + [("ids", None)]
            empty = []
            for sc_ in seg_scores:
                v = pc.loc[d.index, col(s, sc_, "all", "mean_z")].to_numpy(float)
                (items.append(("score", sc_)) if np.isfinite(v).any() else empty.append(sc_))
            title = (f"{seg} union subset ({len(d)} cells): panel scores (panels filtered with the "
                     f"{T.LABEL_DISPLAY[lab_col]}), subset UMAP")
            note = ("Score = mean of the canonical z-scored ln(1 + intensity) over the panel proteins (step 1; atlas "
                    "Celltype contrasts within the segment; protein detected in >= 10% of the segment's SCP cells under "
                    f"the {T.LABEL_DISPLAY[lab_col]}). Colour limits: 2nd-98th percentile; differences centred at 0. "
                    + (f"Empty panels (no score): {', '.join(empty)}. " if empty else "")
                    + "UMAP: subset embedding of step 0 (primary).")
            note_text, note_h = T8.note_block(note)
            title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
            pw, ph, gap = 52.0, 36.0, 7.0
            nrow = int(np.ceil(len(items) / 3))
            leg_h = 0.0
            for c in T.LABELS:
                ent = [(f"{k} ({int((d[c] == k).sum())})", "") for k in order_labels(d[c])]
                leg_h = max(leg_h, len(T.legend_rows(ent, pw)) * (LH + 0.4))
            rh = LH + 0.8 + ph + 0.8 + max(leg_h, 1.8 + 0.5 + LH + 0.2) + 3.0
            H = T8.M + title_h + nrow * rh + note_h + T8.M
            sh = T8.Sheet(H)
            sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
            for k, (kind, what) in enumerate(items):
                x = T8.M + (k % 3) * (pw + gap)
                y = T8.M + title_h + (k // 3) * rh
                if kind == "label":
                    T.umap_categorical(sh, x, y, pw, ph, xy, d[what], order_labels(d[what]), colors,
                                       T.LABEL_DISPLAY[what])
                elif kind == "ids":
                    T.umap_continuous(sh, x, y, pw, ph, xy, np.log(d["n_detected_proteins"].astype(float)),
                                      "ln(protein IDs)", viridis)
                else:
                    v = pc.loc[d.index, col(s, what, "all", "mean_z")].to_numpy(float)
                    n = int(score_sets[(score_sets["label_column"] == lab_col) & (score_sets["gene_set"] == "all")
                                       & (score_sets["panel"] == what)]["n_proteins"].iloc[0]) if " - " not in what else 0
                    ttl = f"{what} score ({n} proteins)" if n else f"{what} score"
                    if " - " in what:
                        lim = float(np.nanquantile(np.abs(v), 0.98))
                        T.umap_continuous(sh, x, y, pw, ph, xy, v, ttl, div, -lim, lim)
                    else:
                        T.umap_continuous(sh, x, y, pw, ph, xy, v, ttl, viridis)
            sh.note(note_text, note_h)
            T8.save_checked(run, sh, f"scores_umap_{seg.lower()}_{s}", title, fig_rows)

    # 2. effects (PT) -------------------------------------------------------------------------------------------
    gcol = {"t15": "#184f95", "pre": "#6da7ec", T.CONSENSUS: "#eb6834"}
    rows = [r for r in (*PT_SCORES, "ln(protein IDs)") if r != "PTS2"]
    title = "PCT vs PST along the PT subtype scores: AUC and depth-adjusted difference"
    note = ("AUC = P(score of a PCT cell > score of a PST cell); bars: bootstrap 95% CI (1,000 resamples). Right: OLS "
            "coefficient of PCT (vs PST) in score-SD units, adjusted for ln(protein IDs), with 95% CI. Scores: mean "
            "z over the panel proteins (all panel proteins). Groupings: kidney_annotation labels (kidney_annotation panels), Fig 2A labels (Fig 2A "
            "panels), cells with the same PT label in both sets (kidney_annotation panels). PTS2 panel empty. Bottom row: ln(protein "
            "IDs) itself (no adjustment).")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    rh = 5.0
    lw = max(T8.text_mm(r)[0] for r in rows) + 2.0
    leg = T.legend_rows([(T.GROUPING_TEXT[g], gcol[g]) for g in gcol], 150)
    pw = (180 - 2 * T8.M - lw - 12.0) / 2
    H = T8.M + title_h + LH + 1.0 + len(rows) * rh + 9.0 + len(leg) * (LH + 0.4) + 2.0 + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    y0 = T8.M + title_h + LH + 1.0
    for k, (what, xl) in enumerate((("auc", "AUC (PCT vs PST)"), ("coef", "PCT coefficient (SD), adj. ln(IDs)"))):
        x0 = T8.M + lw + k * (pw + 12.0)
        sh.text(x0, T8.M + title_h, xl, va="top")
        ax = sh.ax(x0, y0, pw, len(rows) * rh)
        for i, r in enumerate(rows):
            for j, g in enumerate(gcol):
                t_ = tests[(tests["grouping"] == g) & (tests["score"] == r)
                           & tests["gene_set"].isin(["all", "-"]) & tests["method"].isin(["mean_z", "-"])]
                if t_.empty:
                    continue
                t_ = t_.iloc[0]
                yy = i + 0.5 + (j - 1) * 0.25
                if what == "auc":
                    ax.plot([t_.auc_ci_low, t_.auc_ci_high], [yy, yy], color=gcol[g], lw=0.8)
                    ax.plot(t_.auc, yy, "o", color=gcol[g], ms=2.6)
                elif r != "ln(protein IDs)":
                    ax.plot([t_.ci_sd_low_adjusted_ln_ids, t_.ci_sd_high_adjusted_ln_ids], [yy, yy], color=gcol[g], lw=0.8)
                    ax.plot(t_.coef_sd_adjusted_ln_ids, yy, "o", color=gcol[g], ms=2.6)
        ax.set_ylim(len(rows), 0)
        ax.set_yticks([])
        ax.axvline(0.5 if what == "auc" else 0.0, color=T.INK["muted"], lw=0.5, ls="--")
        if what == "auc":
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        else:
            lim = np.nanmax(np.abs(tests[tests["method"] == "mean_z"][["ci_sd_low_adjusted_ln_ids",
                                                                     "ci_sd_high_adjusted_ln_ids"]].to_numpy()))
            lim = float(np.ceil(lim * 2) / 2)
            ax.set_xlim(-lim, lim)
            ax.set_xticks(np.linspace(-lim, lim, 5))
        for i in range(1, len(rows)):
            ax.axhline(i, color=T.INK["grid"], lw=0.4)
        T8.style_axes(ax)
        ax.spines["left"].set_visible(False)
    for i, r in enumerate(rows):
        sh.text(T8.M + lw - 1.0, y0 + (i + 0.5) * rh, r, ha="right", va="center_baseline")
    T.draw_legend(sh, T8.M + lw, y0 + len(rows) * rh + 9.0, leg)
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "scores_effects_pt", title, fig_rows)

    # 3. by label: PT difference boxes + scatter; TAL scores ---------------------------------------------------------
    title = "S1/S2 - S3 score by PT label, and TAL / MD panel scores (mean z)"
    note = ("A: S1/S2 - S3 score of PCT and PST cells under each grouping (boxes: median, quartiles; whiskers 5th-95th "
            "percentile; points: cells). B: the same score against ln(protein IDs), PT union cells, kidney_annotation labels. C: CTAL "
            "- MTAL score of kidney_annotation TAL cells against ln(protein IDs) (MTAL panel: Cldn10 only). D: MD panel score of the "
            "Fig 2A MD cells (kidney_annotation Endothelial) and the Fig 2A TAL cells (descriptive). kidney_annotation panels unless stated.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    ph = 38.0
    H = T8.M + title_h + 2 * (LH + 1.0 + ph + 12.0) + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    groupings = T.pt_groupings(labels)
    rng = np.random.default_rng(T.SEED)

    def boxes(ax, data, cols_, ticks):
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True)
        for patch, c in zip(bp["boxes"], cols_):
            patch.set(facecolor=c, alpha=0.35, edgecolor=T.INK["secondary"], linewidth=0.5)
        for part in ("whiskers", "caps", "medians"):
            for l_ in bp[part]:
                l_.set(color=T.INK["secondary"], linewidth=0.6)
        for i, (v, c) in enumerate(zip(data, cols_), start=1):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=1.2, color=c, linewidths=0, rasterized=True)
        ax.set_xticks(range(1, len(ticks) + 1))
        ax.set_xticklabels(ticks)
        T8.style_axes(ax)

    yA = T8.M + title_h
    sh.text(T8.M, yA, "A", fontweight="bold", va="top")
    ax = sh.ax(T8.M + 12.0, yA + LH + 1.0, 72.0, ph)
    data, cols_, ticks = [], [], []
    for g, lab in groupings.items():
        s = T.LABEL_SHORT[GROUPING_PANELS[g]]
        v = pc[col(s, "S1/S2 - S3", "all", "mean_z")].to_numpy(float)
        for k in T.PT_LABELS:
            m = lab.eq(k).to_numpy()
            data.append(v[m])
            cols_.append(colors[k])
            ticks.append(f"{k}\n{T.GROUPING_SHORT[g]}")
    boxes(ax, data, cols_, ticks)
    ax.set_ylabel("S1/S2 - S3 score")
    sh.text(T8.M + 100.0, yA, "B", fontweight="bold", va="top")
    ax = sh.ax(T8.M + 112.0, yA + LH + 1.0, 60.0, ph)
    d = pc[pc["in_pt_union"]]
    for k in ("Unknown", *T.PT_LABELS):
        m = d["kidney_t15_label"].eq(k).to_numpy()
        ax.scatter(np.log(d.loc[m, "n_detected_proteins"].astype(float)), d.loc[m, col("t15", "S1/S2 - S3", "all", "mean_z")],
                   s=2.0, color=colors.get(k, T.INK["muted"]), linewidths=0, rasterized=True, label=k)
    ax.set_xlabel("ln(protein IDs)")
    ax.set_ylabel("S1/S2 - S3 score")
    T8.style_axes(ax)
    T8.trim_ticks(ax)
    ax.legend(loc="upper left", fontsize=7, handletextpad=0.2, borderaxespad=0.2, markerscale=2.5)
    yC = yA + LH + 1.0 + ph + 12.0
    sh.text(T8.M, yC, "C", fontweight="bold", va="top")
    ax = sh.ax(T8.M + 12.0, yC + LH + 1.0, 60.0, ph)
    m = labels["kidney_t15_label"].eq("TAL").to_numpy()
    ax.scatter(np.log(pc.loc[m, "n_detected_proteins"].astype(float)), pc.loc[m, col("t15", "CTAL - MTAL", "all", "mean_z")],
               s=2.0, color=colors["TAL"], linewidths=0, rasterized=True)
    ax.set_xlabel("ln(protein IDs)")
    ax.set_ylabel("CTAL - MTAL score")
    T8.style_axes(ax)
    T8.trim_ticks(ax)
    sh.text(T8.M + 100.0, yC, "D", fontweight="bold", va="top")
    ax = sh.ax(T8.M + 112.0, yC + LH + 1.0, 60.0, ph)
    v = pc[col("t15", "MD", "all", "mean_z")].to_numpy(float)
    md = labels["kidney_pre_label"].eq("MD").to_numpy()
    tal = labels["kidney_pre_label"].eq("TAL").to_numpy()
    boxes(ax, [v[md], v[tal]], [colors["MD"], colors["TAL"]], [f"Fig 2A MD\n({int(md.sum())})",
                                                              f"Fig 2A TAL\n({int(tal.sum())})"])
    ax.set_ylabel("MD panel score")
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "scores_by_label", title, fig_rows)


if __name__ == "__main__":
    main()
