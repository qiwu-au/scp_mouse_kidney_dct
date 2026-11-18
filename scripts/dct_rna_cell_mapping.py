#!/usr/bin/env python
"""dct_rna step 3: cell-level mapping of the SCP DCT cells to RNA DCT / CNT / TAL centroids (cosine), per condition.

Analysis plan,
step 3 (port of ctrl_rna_core_cell_mapping.py with a condition argument; ``de_table``, ``informative``, ``cosine_map``,
``summarize`` and the thresholds of atlas_dct_cell_mapping and ``class_fractions`` of ctrl_rna_core_cell_mapping
imported unchanged; ``class_fractions`` runs under ``dct_rna_lib.bound``).

Method (atlas_dct step 4 / ctrl_rna_core step 3, unchanged)
-------------------------------------------
- Reference groups (atlas cells and nuclei, all Sources): published DCT, DCT-CNT, CNT, CTAL, MTAL; and the
  atlas_dct-derived level DCT1 (atlas_dct), DCT2 (atlas_dct), DCT-CNT, CNT, CTAL, MTAL (DCT unassigned (atlas_dct) not a centroid).
- Informative genes: Wilcoxon one-vs-rest among the reference groups on log-normalized expression of the condition's
  shared genes (step 1), kept if BH q < 0.05, log2 FC >= 1, detected in >= 25% of the group's cells, and the protein
  detected (intensity > 0) in >= 10% of the condition's cells; top 50 per group by Wilcoxon score (union).
  Sensitivity: top 25 / top 100, and kidney_rna-style centroids. The protein filter uses each condition's own cells, so
  each condition has its own informative gene set.
- Centroids: group means of log-normalized expression, z-scored per gene across the groups (equal weight).
- SCP cells: dct_labels ``.X`` (per-protein z-scored ln(1 + intensity), clipped at 10, each condition scaled on its own).
- Cosine similarity to each centroid; top class, score, second, margin. Low confidence: score < 0.20 or margin <
  0.05. Predictions do not use the SCP labels.

Reported per condition (Ctrl first): confusion against the primary and second columns (labels present; absent
labels 'does not apply'), margins, the fraction mapped to TAL (CTAL or MTAL), CNT and DCT-CNT; cross-tabulation with
the dct_reviewer TAL / CNT_CD / interstitial flags; by plate and by the condition's protein-ID quartile; Spearman rho of the
top score and margin with protein IDs.

Inputs (read-only): dct_labels labelled objects and canonical labels; dct_reviewer per-cell flags
                    (dct_reviewer_task3_marker_scores_per_cell.csv); dct_intake filtering summary; atlas_dct region subset and split
                    table; dct_rna step 1.
Outputs: results/tables/dct_rna/dct_rna_mapping_*.csv;
         results/figures/dct_rna/dct_rna_mapping_confusion, dct_rna_mapping_scores (.pdf/.png/.tif)

Command (project root):  python scripts/dct_rna_cell_mapping.py
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
import ctrl_rna_core_cell_mapping as M20
import dct_rna_lib as L
from atlas_dct_cell_mapping import (FLAGS, LFC_MIN, MARGIN_MIN, PCT_MIN, Q_MAX, SCORE_MIN, SCP_DET_MIN, TOP_N,
                                     TOP_N_SENS, cosine_map, de_table, informative, summarize)

PRIMARY_VARIANT = f"top{TOP_N}_equal_weight_z"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_rna step 3: SCP cell -> RNA centroid mapping (cosine), per condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run, cond, region, ln_full, checks):
    s = L.SETTINGS[cond]
    adata, labels, ln1p, det = L.load_scp(run, cond, checks)
    del ln1p
    flags = L.t28_per_cell(run, "task3_marker_scores_per_cell", cond, labels, checks)
    sf = L.read_table(run, "shared_features")
    sf = sf[(sf["condition"] == cond) & sf["shared"]]
    gene_to_prot = dict(zip(sf["atlas_gene"], sf["protein"]))
    pos = region.var_names.get_indexer(sf["atlas_gene"])
    ln_all = ln_full[:, pos]
    genes = pd.Index(sf["atlas_gene"])
    scp_det = pd.Series(det[sf["protein"]].mean(axis=0).to_numpy(), index=sf["atlas_gene"].to_numpy())
    Xz = pd.DataFrame(cc.dense(adata.X), index=labels.index, columns=adata.var_names.astype(str))
    checks.add("dct_labels .X: finite; max |value| (scaled, clip 10)", f"{bool(np.isfinite(Xz.to_numpy()).all())}; "
               f"{np.abs(Xz.to_numpy()).max():.2f}", bool(np.isfinite(Xz.to_numpy()).all())
               and np.abs(Xz.to_numpy()).max() <= 10 + 1e-9, "<= 10", cond)
    edges = L.depth_quartiles(run, cond, labels, checks)

    de_all, per_cell, conf_rows, summ_rows, var_rows, gene_rows = [], [], [], [], [], []
    for derived in (False, True):
        lvl = T.LEVEL_SHORT[derived]
        order = list(T.rna_groups(derived))
        grp = region.obs[T.group_column(derived)].astype(str).to_numpy()
        ref = np.isin(grp, order)
        ln_ref = ln_all[np.flatnonzero(ref)]
        print(f"[mapping] {cond}: Wilcoxon one-vs-rest, {lvl}: {int(ref.sum())} cells x {len(genes)} genes", flush=True)
        de = de_table(ln_ref, grp[ref], order, genes, scp_det)
        de.insert(0, "level", lvl)
        de_all.append(de)
        n_pass = de[de["passes_filters"]].groupby("group").size().reindex(order, fill_value=0)
        checks.add(f"{lvl}: genes passing the filters per group", "; ".join(f"{g} {n}" for g, n in n_pass.items()),
                   None, "", cond)
        stats_ = T.group_stats(ln_ref, region.X[np.flatnonzero(ref)][:, pos], pd.Series(grp[ref]), order, genes)
        mean_ref = np.asarray(ln_ref.mean(axis=0)).ravel()
        sd_ref = np.sqrt(np.maximum(np.asarray(ln_ref.multiply(ln_ref).mean(axis=0)).ravel() - mean_ref ** 2, 0))
        preds = {}
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
                primary = variant == PRIMARY_VARIANT
                preds[variant] = (pred, scores, inf)
                if primary:
                    gene_rows += [{"level": lvl, "atlas_gene": g, "protein": gene_to_prot[g],
                                   "group": de[(de["atlas_gene"] == g) & (de["rank_in_group"] <= n)]["group"].iloc[0],
                                   "scp_detection_all": float(scp_det[g]),
                                   **{f"centroid_z_{c}": float(cent.loc[c, g]) for c in order}} for g in inf]
                for col in s.LABEL_COLUMNS:
                    with L.bound(s, M20):
                        fr = M20.class_fractions(pred, labels[col], order)
                    fr.insert(0, "label_column", col)
                    fr.insert(0, "variant", variant)
                    fr.insert(0, "level", lvl)
                    fr["n_informative_genes"] = len(inf)
                    fr["primary_variant"] = primary
                    fr["status"] = "computed"
                    na = [{"level": lvl, "variant": variant, "label_column": col, "scp_label": g, "n_cells": 0,
                           "n_informative_genes": len(inf), "primary_variant": primary, "status": L.NOT_APPLICABLE}
                          for g in L.absent(labels, col)]
                    summ_rows.append(pd.concat([fr, pd.DataFrame(na)], ignore_index=True) if na else fr)
                    lab = labels[col].astype(str)
                    ct = pd.crosstab(lab, pred["top_class"]).reindex(index=list(L.SCP_GROUPS), columns=order,
                                                                     fill_value=0)
                    for sg in ct.index:
                        tot = int(ct.loc[sg].sum())
                        for c in order:
                            conf_rows.append({"level": lvl, "variant": variant, "label_column": col, "scp_label": sg,
                                              "rna_class": c, "n_cells": int(ct.loc[sg, c]),
                                              "fraction_of_label": float(ct.loc[sg, c] / tot) if tot else np.nan,
                                              "status": "computed" if tot else L.NOT_APPLICABLE})
        p, sc_, inf = preds[PRIMARY_VARIANT]
        for variant, (pred, _, inf_v) in preds.items():
            if variant != PRIMARY_VARIANT:
                var_rows.append({"level": lvl, "variant": variant, "n_informative_genes": len(inf_v),
                                 "fraction_same_top_class_as_primary": float((pred["top_class"] == p["top_class"]).mean()),
                                 "fraction_low_confidence": float(pred["low_confidence"].mean())})
        per_group = pd.Series([r["group"] for r in gene_rows if r["level"] == lvl]).value_counts().reindex(
            order, fill_value=0)
        checks.add(f"{lvl} primary: informative genes (top {TOP_N} per group; per group); cells low confidence",
                   f"{len(inf)} ({'; '.join(f'{g} {n}' for g, n in per_group.items())}); "
                   f"{int(p['low_confidence'].sum())} of {len(p)}", None, "", cond)
        pc = p.add_prefix(f"{lvl}_")
        for c in sc_.columns:
            pc[f"{lvl}_score_{c}"] = sc_[c].to_numpy()
        per_cell.append(pc)

    base = labels[["cell_key", "plate", *s.LABEL_COLUMNS, "n_detected_proteins"]].rename(
        columns={s.PRIMARY: "primary_label", s.SECOND: "second_label"})
    pc = pd.concat([base, flags[list(FLAGS)], *per_cell], axis=1).reset_index()
    pc["depth_bin"] = L.quartile_bins(pc["n_detected_proteins"], edges).to_numpy()
    checks.add("cells per protein-ID quartile", "; ".join(f"{k} {v}" for k, v in
                                                         pc["depth_bin"].value_counts().reindex(L.QUARTILE_LABELS).items()),
               bool(pc["depth_bin"].isin(L.QUARTILE_LABELS).all()), "", cond)
    summ = pd.concat(summ_rows, ignore_index=True)
    conf = pd.DataFrame(conf_rows)

    # flags, plate, depth (primary variants) ------------------------------------------------------------------------
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
                    flag_rows.append({"level": lvl, "mapped_to": mname, "confident_only": conf_only, "t28_flag": fl,
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
            checks.add(f"{lvl} primary: Spearman rho of {qn} with protein IDs per cell",
                       f"{r.statistic:.3f} (p {r.pvalue:.1e})", None, "", cond)
            pd_rows.append({"level": lvl, "by": f"spearman_{qn}_vs_protein_ids", "value": "all", "n_cells": len(pc),
                            "rho": float(r.statistic), "p_value": float(r.pvalue)})
        for col in s.LABEL_COLUMNS:
            x = summ[(summ["level"] == lvl) & summ["primary_variant"] & (summ["label_column"] == col)
                     & (summ["status"] == "computed")]
            checks.add(f"{lvl} primary / {col}: fraction mapped to TAL; CNT; DCT-CNT; low confidence, per label",
                       "; ".join(f"{r.scp_label} {r.frac_top_TAL:.2f} / {r['frac_top_CNT']:.2f} / "
                                 f"{r['frac_top_DCT-CNT']:.2f} / {r.frac_low_confidence:.2f}"
                                 for _, r in x.iterrows()), None, "", cond)
    de_all = pd.concat(de_all, ignore_index=True)
    return {"de": de_all, "genes": pd.DataFrame(gene_rows), "per_cell": pc, "conf": conf, "summ": summ,
            "variants": pd.DataFrame(var_rows), "flags": pd.DataFrame(flag_rows), "plate_depth": pd.DataFrame(pd_rows)}


# ---------------------------------------------------------------------------
# Figures (ports of ctrl_rna_core_cell_mapping.figures: both conditions in one figure)
# ---------------------------------------------------------------------------
def block_rows():
    """(condition, role, label) rows of the confusion figure, Ctrl first."""
    return [(c, r, g) for c in L.CONDITIONS for r in L.ROLES for g in L.SCP_GROUPS]


def confusion_figure(run, conf: pd.DataFrame, fig_rows: list[dict]) -> None:
    T.setup_matplotlib()
    cmap = T.sequential_cmap()
    LH = T.text_mm("Hg")[1]
    rows = block_rows()
    row_txt = [f"{g} ({c} {r})" for c, r, g in rows]
    lw = max(T.text_mm(t)[0] for t in row_txt) + 1.0 + 1.6 + 1.0
    rh = 4.2
    title = L.fit_title("SCP DCT cells mapped to the nearest RNA centroid (cosine; fraction of each SCP label), Ctrl "
                        "and ZeroK")
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    panels = [(False, "Atlas classes"), (True, "atlas_dct-derived split + atlas classes")]
    cw = 8.5
    gene_h = max(T.text_mm(c)[0] for c in (*T.DERIVED_GROUPS, *T.PUBLISHED_GROUPS)) + 1.5
    note = (f"Top {TOP_N} informative genes per RNA group (Wilcoxon one-vs-rest among the groups; q < {Q_MAX}, log2 FC >= "
            f"{LFC_MIN:g}, detected in >= {PCT_MIN:.0%} of the group and in >= {SCP_DET_MIN:.0%} of the condition's "
            "cells); centroids z-scored per gene across the groups; SCP cells: z-scored ln(1 + intensity) (dct_labels, each "
            "condition scaled on its own). Numbers: cells. Rows: label (condition, label column role); primary = "
            "cs_ctrl_deg_label / cs_zerok_pipeline_label, second = cs_ctrl_cluster_label / cs_zerok_cluster_label; grey "
            "rows: label absent under this column (does not apply). " + L.ATLAS_DCT_SPLIT_CAVEAT + " " + L.UNPAIRED_NOTE + " "
            + L.ZEROK_NOTE)
    note_text, note_h = T.note_block(note)
    H = T.M + title_h + LH + 1.0 + len(rows) * rh + gene_h + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    x = T.M + lw
    y0 = T.M + title_h + LH + 1.0
    nblock = len(L.SCP_GROUPS)
    for k, (derived, ptitle) in enumerate(panels):
        order = list(T.rna_groups(derived))
        sh.text(x if k else T.M, y0 - LH - 1.0, ptitle, va="top")
        c = conf[(conf["level"] == T.LEVEL_SHORT[derived]) & (conf["variant"] == PRIMARY_VARIANT)]
        frac = np.full((len(rows), len(order)), np.nan)
        nn = np.zeros((len(rows), len(order)), int)
        for a, (cond, role, g) in enumerate(rows):
            col = L.column_of(cond, role)
            sel = c[(c["condition"] == cond) & (c["label_column"] == col) & (c["scp_label"] == g)].set_index("rna_class")
            frac[a] = sel["fraction_of_label"].reindex(order).to_numpy(float)
            nn[a] = sel["n_cells"].reindex(order).to_numpy(int)
        ax = sh.ax(x, y0, len(order) * cw, len(rows) * rh)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=(0, len(order), len(rows), 0))
        for b in range(1, len(rows) // nblock):
            ax.axhline(b * nblock, color="white" if b % 2 else T.INK["secondary"], lw=1.5 if b % 2 else 1.0)
        ax.set_axis_off()
        for a in range(len(rows)):
            if np.isnan(frac[a]).all():
                L.absent_cell(sh, x, y0 + a * rh, len(order) * cw, rh, "does not apply" if k == 0 else "")
                continue
            for b in range(len(order)):
                sh.text(x + (b + 0.5) * cw, y0 + (a + 0.5) * rh, f"{nn[a, b]}", ha="center", va="center_baseline",
                        color="white" if frac[a, b] > 0.55 else T.INK["primary"])
        if k == 0:
            for a, ((cond, role, g), t_) in enumerate(zip(rows, row_txt)):
                yc = y0 + (a + 0.5) * rh
                sh.rect(x - 1.0 - 1.6, yc - 0.8, 1.6, 1.6, facecolor=L.SCP_PALETTE[g], edgecolor="none")
                sh.text(x - 1.0 - 1.6 - 1.0, yc, t_, ha="right", va="center_baseline")
        for b, kk in enumerate(order):
            sh.text(x + (b + 0.5) * cw, y0 + len(rows) * rh + 0.8, kk, rotation=90, ha="center", va="top")
        x = x + len(order) * cw + 8.0
    cax = sh.ax(x, y0, 2.4, 20.0)
    cax.imshow(np.linspace(1, 0, 256)[:, None], cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 0.5, 1])
    cax.tick_params(length=1.5, width=0.4, labelsize=T.FONT, pad=1)
    sh.text(x + 2.4 + 6.5, y0 + 10.0, "Fraction of label", rotation=90, ha="left", va="center")
    sh.note(note_text, note_h)
    L.save_checked(run, sh, "mapping_confusion", title.replace("\n", " "), fig_rows)


def scores_figure(run, pc: pd.DataFrame, fig_rows: list[dict]) -> None:
    T.setup_matplotlib()
    LH = T.text_mm("Hg")[1]
    title = L.fit_title("Cosine top score and margin per SCP label (atlas classes, primary variant), Ctrl and ZeroK")
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    note = (f"Dashed lines: low-confidence thresholds (score {SCORE_MIN}, margin {MARGIN_MIN}). Boxes: median and "
            "quartiles; whiskers 5th-95th percentile. Groups: condition and label column role (primary = "
            "cs_ctrl_deg_label / cs_zerok_pipeline_label; second = cs_ctrl_cluster_label / cs_zerok_cluster_label); only "
            "labels with cells are shown. " + L.ZEROK_NOTE)
    note_text, note_h = T.note_block(note)
    groups = []
    for cond in L.CONDITIONS:
        sub = pc[pc["condition"] == cond]
        for role in L.ROLES:
            lab = sub[f"{role}_label"].astype(str)
            groups.append((cond, role, [g for g in L.SCP_GROUPS if (lab == g).any()]))
    ph, head_h, tick_h, gap = 30.0, LH + 1.5, LH + 2.0, 4.0
    H = T.M + title_h + 2 * (head_h + ph + tick_h) + gap + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    ax_x, ax_w = T.M + 14.0, T.W - 2 * T.M - 16.0
    for k, (qn, ylab, thr) in enumerate((("top_score", "Top cosine score", SCORE_MIN),
                                         ("margin", "Margin (top - second)", MARGIN_MIN))):
        ay = T.M + title_h + k * (head_h + ph + tick_h + gap) + head_h
        ax = sh.ax(ax_x, ay, ax_w, ph)
        data, ticks, cols, spans = [], [], [], []
        for cond, role, labs in groups:
            sub = pc[pc["condition"] == cond]
            start = len(data) + 1
            for g in labs:
                data.append(sub.loc[sub[f"{role}_label"].astype(str) == g, f"published_{qn}"].to_numpy(float))
                ticks.append(g)
                cols.append(L.SCP_PALETTE[g])
            spans.append((cond, role, start, len(data)))
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.6, patch_artist=True)
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, alpha=0.55, edgecolor=T.INK["secondary"], linewidth=0.5)
        for part in ("whiskers", "caps", "medians"):
            for l_ in bp[part]:
                l_.set(color=T.INK["secondary"], linewidth=0.6)
        ax.axhline(thr, color=T.INK["muted"], lw=0.6, ls="--")
        n = len(data)
        ax.set_xlim(0.5, n + 0.5)
        for i, (cond, role, a, b) in enumerate(spans):
            if i:
                ax.axvline(a - 0.5, color=T.INK["secondary"] if role == "primary" else T.INK["grid"], lw=0.6)
            xc = ax_x + ((a + b) / 2 - 0.5) / n * ax_w
            sh.text(xc, ay - 0.8, f"{cond} {role}", ha="center", va="bottom", color=T.INK["secondary"])
        ax.set_xticks(range(1, n + 1))
        ax.set_xticklabels(ticks)
        ax.set_ylabel(ylab)
        T.style_axes(ax)
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9])
    sh.note(note_text, note_h)
    L.save_checked(run, sh, "mapping_scores", title.replace("\n", " "), fig_rows)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "cell_mapping", args)
    L.ported_from(run, "ctrl_rna_core_cell_mapping.py", "atlas_dct_cell_mapping.py")
    checks = L.Checks(L.DATASET)
    region = T.load_region(run, with_split=True)
    ln_full = T.lognorm(region.X)
    res = {c: one_condition(run, c, region, ln_full, checks) for c in L.CONDITIONS}
    out = {k: L.stack({c: res[c][k] for c in L.CONDITIONS}) for k in res["Ctrl"]}
    fig_rows = []
    confusion_figure(run, out["conf"], fig_rows)
    scores_figure(run, out["per_cell"], fig_rows)
    figs = L.figure_rows_frame(fig_rows, "cell_mapping")
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(out["de"], "mapping_de_genes")
    run.table(out["genes"], "mapping_informative_genes")
    run.table(out["per_cell"], "mapping_per_cell")
    run.table(out["conf"], "mapping_confusion")
    run.table(out["summ"], "mapping_summary")
    run.table(out["variants"], "mapping_variants")
    run.table(out["flags"], "mapping_flags")
    run.table(out["plate_depth"], "mapping_plate_depth")
    run.table(figs, "mapping_figure_checks")
    run.table(checks.frame(), "mapping_checks")
    run.finish({"top_n": TOP_N, "top_n_sensitivity": list(TOP_N_SENS), "q_max": Q_MAX, "log2fc_min": LFC_MIN,
                "frac_group_min": PCT_MIN, "scp_detection_min": SCP_DET_MIN, "scp_detection_cells": "the condition's "
                "own cells", "score_min": SCORE_MIN, "margin_min": MARGIN_MIN,
                "centroids": "equal-weight z across groups (primary); kidney_rna style (sensitivity)",
                "scp_matrix": "dct_labels .X (z-scored ln(1 + intensity), clipped at 10; each condition scaled on its own)",
                "depth_bins": "protein-ID quartiles per condition (dct_intake filtering summary)",
                "per_cell_columns": {c: {"primary_label": L.SETTINGS[c].PRIMARY, "second_label": L.SETTINGS[c].SECOND}
                                     for c in L.CONDITIONS}, "formats": list(L.FORMATS), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see dct_rna_mapping_checks.csv.")


if __name__ == "__main__":
    main()
