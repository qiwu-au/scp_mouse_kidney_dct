#!/usr/bin/env python
"""rat_segments step 4b: pseudobulk check, SCP PCT / PST / TAL vs Limbutara S1 / S2 / S3 / cTAL / mTAL.

Task rat_segments (authors, 30 Oct 2026).

What it does
------------
SCP pseudobulk = mean ln(1 + intensity) per label with missing = 0 (as atlas_dct); Limbutara = segment mean of the centred
log2 copies per cell (rat_segments_panels.py; samples quantified only). Matched proteins only (rat_segments_intake.py).
1. PT, within segment. With two SCP groups, centring within PT reduces both sides to log ratios: Spearman rho of the
   SCP ratio PCT - PST with the Limbutara ratios S1 - S3, S2 - S3, S1 - S2 and mean(S1, S2) - S3.
   - SCP ratio: raw (difference of the group means) and depth-adjusted (per-protein OLS of ln(1 + intensity) on PCT
     and ln(protein IDs) over the PCT + PST cells; the PCT coefficient).
   - Proteins: quantified (>= 2/3 samples) in S1, S2 and S3 and detected in >= 10% of the PCT or the PST cells of the
     label set; sensitivity: only the Limbutara-variable proteins (largest pairwise |difference| among S1-S3 >= 1).
   - Null: PCT / PST labels permuted within ln(protein IDs) deciles of the PCT + PST cells (1,000 permutations,
     seed 42), same rho recomputed; two-sided permutation p = 2 x the smaller tail of the null, (1 + #tail) / 1,001
     (capped at 1); the null keeps each label's depth composition and is not centred at 0.
2. Across segments (a within-TAL ratio is not possible: SCP has one TAL label): SCP PCT, PST, TAL x Limbutara S1, S2,
   S3, cTAL, mTAL; proteins quantified in all five segments and detected in >= 10% of the cells of at least one of
   the three SCP groups; Spearman rho on the raw values and on z-scores of each protein across the groups within its
   data type (ddof 0, as atlas_dct).
Both label sets (``kidney_t15_label``, ``kidney_pre_label``).

Inputs (read-only): kidney_object kidney object and labels; kidney_annotation labels; rat_segments intake / panel tables.
Outputs: results/tables/rat_segments/rat_segments_pseudobulk_pt_ratio.csv, rat_segments_pseudobulk_pt_proteins.csv,
         rat_segments_pseudobulk_matrix.csv, rat_segments_pseudobulk_means.csv, rat_segments_pseudobulk_checks.csv,
         rat_segments_pseudobulk_figure_checks.csv, rat_segments_step_pseudobulk_manifest.json;
         results/figures/rat_segments/rat_segments_pseudobulk (.pdf, .png)

Command (project root):  python scripts/rat_segments_pseudobulk.py
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

SCP_GROUPS = ("PCT", "PST", "TAL")
VERSIONS = ("raw", "depth_adjusted")
PROTEIN_SETS = ("all_shared", "limbutara_variable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rat_segments step 4b: pseudobulk SCP vs Limbutara.")
    cc.add_common_args(parser)
    return parser.parse_args()


def spearman_rank(rank_a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman rho of a fixed ranked vector with each row of b (rows ranked here)."""
    rb = stats.rankdata(b, axis=-1)
    ra = (rank_a - rank_a.mean()) / rank_a.std()
    rb = (rb - rb.mean(axis=-1, keepdims=True)) / rb.std(axis=-1, keepdims=True)
    return (rb * ra).mean(axis=-1)


def perm_p(null: np.ndarray, obs: float) -> float:
    """Two-sided permutation p relative to the null distribution (which is not centred at 0 here):
    2 x the smaller tail, (1 + #tail) / (N + 1), capped at 1."""
    n = len(null)
    hi = (1 + np.sum(null >= obs - 1e-12)) / (n + 1)
    lo = (1 + np.sum(null <= obs + 1e-12)) / (n + 1)
    return float(min(1.0, 2 * min(hi, lo)))


def scp_ratios(y: np.ndarray, is_pct: np.ndarray, ln_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw (mean PCT - mean PST) and depth-adjusted (OLS PCT coefficient with ln(protein IDs)) per protein."""
    raw = y[is_pct].mean(axis=0) - y[~is_pct].mean(axis=0)
    design = np.column_stack([np.ones(len(y)), is_pct.astype(float), ln_ids])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return raw, beta[1]


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "pseudobulk", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, xz, ln1p, det = T9.load_scp(run, checks)
    lm = T.read_table(run, "limbutara_mouse").set_index("protein")
    stc = T.read_table(run, "ktea_segment_stats_centred", dtype={"rat_id": str}).set_index("rat_id")
    lim = stc.reindex(lm["rat_id"].astype(str).to_numpy())
    lim.index = lm.index
    ln_ids_all = np.log(labels["n_detected_proteins"].astype(float)).to_numpy()
    rng = np.random.default_rng(T.SEED)

    means_rows, ratio_rows, prot_rows, mat_rows, null_store = [], [], [], [], {}
    for lab_col in T9.LABELS:
        s = T9.LABEL_SHORT[lab_col]
        lab = labels[lab_col].astype(str)
        gm = pd.DataFrame({g: ln1p[lab.eq(g).to_numpy()].mean(axis=0) for g in SCP_GROUPS})
        gd = pd.DataFrame({g: det[lab.eq(g).to_numpy()].mean(axis=0) for g in SCP_GROUPS})
        for g in SCP_GROUPS:
            means_rows.append(pd.DataFrame({"label_column": lab_col, "group": g, "protein": gm.index,
                                            "mean_ln1p": gm[g].to_numpy(), "detection": gd[g].to_numpy(),
                                            "n_cells": int(lab.eq(g).sum())}))

        # 1. PT ratio -------------------------------------------------------------------------------------------------
        q_pt = lim[[f"quant_{x}" for x in T.PT_SEGMENTS]].astype(bool).all(axis=1)
        d_pt = gd[["PCT", "PST"]].reindex(lim.index).max(axis=1) >= T.SCP_DET_MIN
        prots = lim.index[q_pt & d_pt].tolist()
        L = lim.loc[prots, [f"log2_{x}" for x in T.PT_SEGMENTS]].set_axis(list(T.PT_SEGMENTS), axis=1)
        diffs = L[["S1", "S2", "S3"]].to_numpy()
        variable = (np.abs(diffs[:, :, None] - diffs[:, None, :]).max(axis=(1, 2)) >= T.VARIABLE_MIN)
        lim_ratios = {name: L[list(a)].mean(axis=1).to_numpy() - L[list(b)].mean(axis=1).to_numpy()
                      for name, (a, b) in T.PT_RATIOS.items()}
        pt = lab.isin(["PCT", "PST"]).to_numpy()
        y = ln1p.loc[pt, prots].to_numpy(np.float64)
        is_pct = lab[pt].eq("PCT").to_numpy()
        z = ln_ids_all[pt]
        obs_raw, obs_adj = scp_ratios(y, is_pct, z)
        bins = pd.qcut(pd.Series(z).rank(method="first"), T.DEPTH_BINS, labels=False).to_numpy()
        perm_raw = np.empty((T.N_PERM, len(prots)))
        perm_adj = np.empty((T.N_PERM, len(prots)))
        for k in range(T.N_PERM):
            lab_p = is_pct.copy()
            for b_ in range(T.DEPTH_BINS):
                idx = np.flatnonzero(bins == b_)
                lab_p[idx] = rng.permutation(is_pct[idx])
            perm_raw[k], perm_adj[k] = scp_ratios(y, lab_p, z)
        checks.add(f"{T9.LABEL_DISPLAY[lab_col]}: PT proteins (quantified in S1-S3, >= 10% of PCT or PST cells); "
                   "Limbutara-variable", f"{len(prots)}; {int(variable.sum())}", len(prots) > 100)
        checks.add(f"{T9.LABEL_DISPLAY[lab_col]}: permutation keeps the PCT count within every depth decile",
                   f"{int(is_pct.sum())} PCT of {len(is_pct)}", True)
        for pset in PROTEIN_SETS:
            keep = np.ones(len(prots), bool) if pset == "all_shared" else variable
            for ver, obs, perm in (("raw", obs_raw, perm_raw), ("depth_adjusted", obs_adj, perm_adj)):
                for rname, lr in lim_ratios.items():
                    r = stats.spearmanr(obs[keep], lr[keep])
                    null = spearman_rank(stats.rankdata(lr[keep]), perm[:, keep])
                    null_store[(s, pset, ver, rname)] = null
                    ratio_rows.append({"label_column": lab_col, "protein_set": pset, "scp_ratio": ver,
                                       "limbutara_ratio": rname, "n_proteins": int(keep.sum()),
                                       "n_pct": int(is_pct.sum()), "n_pst": int((~is_pct).sum()),
                                       "rho": float(r.statistic), "p_spearman": float(r.pvalue),
                                       "null_median": float(np.median(null)),
                                       "null_q025": float(np.quantile(null, 0.025)),
                                       "null_q975": float(np.quantile(null, 0.975)),
                                       "frac_null_ge_observed": float(np.mean(null >= r.statistic - 1e-12)),
                                       "p_perm_two_sided": perm_p(null, float(r.statistic))})
        prot_rows.append(pd.DataFrame({"label_column": lab_col, "protein": prots, "rat_id": lm.loc[prots, "rat_id"],
                                       "scp_ratio_raw": obs_raw, "scp_ratio_depth_adjusted": obs_adj,
                                       "limbutara_variable": variable,
                                       **{f"log2_{x}": L[x].to_numpy() for x in T.PT_SEGMENTS},
                                       **{f"limbutara_{T.safe(k)}": v for k, v in lim_ratios.items()},
                                       "scp_det_pct": gd.loc[prots, "PCT"].to_numpy(),
                                       "scp_det_pst": gd.loc[prots, "PST"].to_numpy()}))

        # 2. 3 x 5 matrix ---------------------------------------------------------------------------------------------
        q5 = lim[[f"quant_{x}" for x in T.STUDY_SEGMENTS]].astype(bool).all(axis=1)
        d3 = gd.reindex(lim.index).max(axis=1) >= T.SCP_DET_MIN
        p5 = lim.index[q5 & d3].tolist()
        A = gm.loc[p5, list(SCP_GROUPS)]
        B = lim.loc[p5, [f"log2_{x}" for x in T.STUDY_SEGMENTS]].set_axis(list(T.STUDY_SEGMENTS), axis=1)
        # T8.zscore_rows standardizes each column across rows (groups x genes in atlas_dct): transpose so that each
        # protein is z-scored across the groups of its data type
        vals = {"raw": (A, B), "z": (T8.zscore_rows(A.T).T, T8.zscore_rows(B.T).T)}
        zA = vals["z"][0]
        checks.add(f"{T9.LABEL_DISPLAY[lab_col]}: z-scores per protein across the SCP groups (row means 0)",
                   f"max |row mean| {float(zA.mean(axis=1).abs().max()):.1e}", bool(zA.mean(axis=1).abs().max() < 1e-9))
        checks.add(f"{T9.LABEL_DISPLAY[lab_col]}: proteins for the 3 x 5 matrix", len(p5), len(p5) > 100)
        for vt, (a_, b_) in vals.items():
            for g in SCP_GROUPS:
                for x in T.STUDY_SEGMENTS:
                    ok = np.isfinite(a_[g].to_numpy()) & np.isfinite(b_[x].to_numpy())
                    r = stats.spearmanr(a_.loc[ok, g], b_.loc[ok, x])
                    mat_rows.append({"label_column": lab_col, "values": vt, "scp_group": g, "limbutara_segment": x,
                                     "n_proteins": int(ok.sum()), "rho": float(r.statistic), "p": float(r.pvalue)})

    ratio = pd.DataFrame(ratio_rows)
    mat = pd.DataFrame(mat_rows)
    for r in ratio[(ratio["protein_set"] == "all_shared")].itertuples():
        checks.add(f"{T9.LABEL_DISPLAY[r.label_column]}, SCP {r.scp_ratio} PCT - PST vs Limbutara {r.limbutara_ratio}: "
                   "rho; null 95%; p perm", f"{r.rho:+.3f}; [{r.null_q025:+.3f}, {r.null_q975:+.3f}]; "
                   f"{r.p_perm_two_sided:.3f}", None)
    for r in mat[(mat["values"] == "z")].groupby(["label_column", "scp_group"]):
        (lc, g), d = r
        top = d.sort_values("rho", ascending=False).iloc[0]
        checks.add(f"{T9.LABEL_DISPLAY[lc]}, 3 x 5 (z): SCP {g} highest rho with", f"{top.limbutara_segment} "
                   f"({top.rho:+.2f}); " + ", ".join(f"{x} {v:+.2f}" for x, v in zip(d.limbutara_segment, d.rho)), None)

    fig_rows = []
    figure(run, ratio, mat, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(ratio, "pseudobulk_pt_ratio")
    run.table(pd.concat(prot_rows, ignore_index=True), "pseudobulk_pt_proteins")
    run.table(mat, "pseudobulk_matrix")
    run.table(pd.concat(means_rows, ignore_index=True), "pseudobulk_means")
    run.table(figs, "pseudobulk_figure_checks")
    run.table(checks.frame(), "pseudobulk_checks")
    run.finish({"n_perm": T.N_PERM, "depth_bins": T.DEPTH_BINS, "variable_min_log2": T.VARIABLE_MIN,
                "pt_ratios": {k: [list(a), list(b)] for k, (a, b) in T.PT_RATIOS.items()},
                "scp_values": "mean ln(1 + intensity), missing = 0", "limbutara_values": T.CENTRING_TEXT,
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see rat_segments_pseudobulk_checks.csv.")


def figure(run, ratio, mat, fig_rows) -> None:
    T9.setup()
    LH = T9.LH
    gcol = {"kidney_t15_label": "#184f95", "kidney_pre_label": "#6da7ec"}
    title = "Pseudobulk: SCP PCT - PST vs Limbutara segment ratios, and SCP groups vs Limbutara segments"
    note = ("A: Spearman rho of the SCP PCT - PST difference (mean ln(1 + intensity); raw and adjusted for ln(protein "
            "IDs)) with the Limbutara log2 ratios, over the shared PT proteins (quantified in S1-S3; >= 10% of PCT or "
            "PST cells). Grey bars: 95% range of rho with PCT / PST labels permuted within ln(protein IDs) deciles "
            "(1,000 permutations). B: Spearman rho of SCP group means and Limbutara segment means, each z-scored per "
            "protein across the groups of its data type (proteins quantified in all five segments; >= 10% of the cells "
            "of one SCP group). Limbutara values: log2 copies per cell centred per sample.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    rows = [(v, r) for v in VERSIONS for r in T.PT_RATIOS]
    rl = [f"{'adjusted' if v == 'depth_adjusted' else 'raw'}: {r}" for v, r in rows]
    lw = max(T8.text_mm(x)[0] for x in rl) + 2.0
    rh = 4.6
    pw = 52.0
    hm_cw, hm_rh = 7.5, 4.6
    H = T8.M + title_h + LH + 1.0 + max(len(rows) * rh + 9.0 + 2 * (LH + 0.4), 2 * (3 * hm_rh + LH + 6.0)) + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    y0 = T8.M + title_h + LH + 1.0
    sh.text(T8.M, T8.M + title_h, "A", fontweight="bold", va="top")
    x0 = T8.M + lw
    ax = sh.ax(x0, y0, pw, len(rows) * rh)
    d = ratio[ratio["protein_set"] == "all_shared"]
    for i, (v, rn) in enumerate(rows):
        for j, lc in enumerate(gcol):
            t_ = d[(d["label_column"] == lc) & (d["scp_ratio"] == v) & (d["limbutara_ratio"] == rn)].iloc[0]
            yy = i + 0.5 + (j - 0.5) * 0.36
            ax.plot([t_.null_q025, t_.null_q975], [yy, yy], color=T.INK["grid"], lw=2.2, solid_capstyle="butt")
            ax.plot(t_.rho, yy, "o", color=gcol[lc], ms=2.6)
    ax.set_ylim(len(rows), 0)
    ax.set_yticks([])
    ax.axvline(0, color=T.INK["muted"], lw=0.5, ls="--")
    lim = float(np.ceil(max(0.1, np.nanmax(np.abs(d[["rho", "null_q025", "null_q975"]].to_numpy()))) * 10) / 10)
    ax.set_xlim(-lim, lim)
    ax.set_xticks(np.round(np.linspace(-lim, lim, 5), 2))
    ax.set_xlabel("Spearman rho")
    for i in range(1, len(rows)):
        ax.axhline(i, color=T.INK["grid"], lw=0.4 if i != len(T.PT_RATIOS) else 0.8)
    T8.style_axes(ax)
    ax.spines["left"].set_visible(False)
    for i, x in enumerate(rl):
        sh.text(x0 - 1.0, y0 + (i + 0.5) * rh, x, ha="right", va="center_baseline")
    leg = T9.legend_rows([(T9.LABEL_DISPLAY[k], c) for k, c in gcol.items()] + [("permutation 95%", T.INK["grid"])],
                         pw + lw)
    T9.draw_legend(sh, T8.M, y0 + len(rows) * rh + 9.0, leg)

    cmap = T8.diverging_cmap()
    xb = x0 + pw + 22.0
    sh.text(xb, T8.M + title_h, "B", fontweight="bold", va="top")
    yb = y0
    for lc in T9.LABELS:
        m = mat[(mat["label_column"] == lc) & (mat["values"] == "z")]
        arr = np.array([[m[(m["scp_group"] == g) & (m["limbutara_segment"] == x)]["rho"].iloc[0]
                         for x in T.STUDY_SEGMENTS] for g in SCP_GROUPS])
        n = int(m["n_proteins"].iloc[0])
        sh.text(xb + 12.0, yb, f"{T9.LABEL_DISPLAY[lc]} ({n} proteins)", va="top")
        yh = yb + LH + 0.8
        axh = sh.ax(xb + 12.0, yh, len(T.STUDY_SEGMENTS) * hm_cw, 3 * hm_rh)
        axh.imshow(arr, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest",
                   extent=(0, len(T.STUDY_SEGMENTS), 3, 0))
        axh.set_axis_off()
        for a in range(3):
            sh.text(xb + 11.0, yh + (a + 0.5) * hm_rh, SCP_GROUPS[a], ha="right", va="center_baseline")
            for b in range(len(T.STUDY_SEGMENTS)):
                sh.text(xb + 12.0 + (b + 0.5) * hm_cw, yh + (a + 0.5) * hm_rh, f"{arr[a, b]:+.2f}", ha="center",
                        va="center_baseline", color="white" if abs(arr[a, b]) > 0.55 else T.INK["primary"])
        for b, x in enumerate(T.STUDY_SEGMENTS):
            sh.text(xb + 12.0 + (b + 0.5) * hm_cw, yh + 3 * hm_rh + 0.6, x, ha="center", va="top")
        yb = yh + 3 * hm_rh + LH + 4.0
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "pseudobulk", title, fig_rows)


if __name__ == "__main__":
    main()
