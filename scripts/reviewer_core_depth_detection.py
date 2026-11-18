#!/usr/bin/env python
"""reviewer_core B.1-B.3: depth and plate confounding of ProLIF - detection, logistic models, matching.



Inputs (read-only): kidney_object canonical DCT object (intensity layer, labels, plate,
n_detected_proteins); reviewer_core outputs reviewer_core_task3_marker_scores_per_cell.csv
(cell_cycle_score), reviewer_core_aldo_scores_per_cell.csv (V1_X ALDO scores) and
reviewer_core_task4_mki67_top2a_correlated_proteins.csv (circular cell-cycle set).
Outputs:
  results/tables/reviewer_core/reviewer_core_depth_*.csv
  results/figures/reviewer_core/reviewer_core_depth_*.pdf/.png

B.1 Detection (intensity > 0) against depth (protein IDs per cell): binned
    (deciles of IDs over all 750 cells) and logistic fits detected ~ ln(IDs),
    per label, for Mki67, Top2a, Slc12a3, Pvalb and Calb1, and for a reference
    set of proteins detected in 40-80% of cells that are not in the 18-protein
    cell-cycle panel, the Tirosh S/G2M lists or the Mki67/Top2a-correlated set.
    Also detected ~ ln(IDs) + label + plate per protein (label odds ratios
    adjusted for depth and plate).
B.2 ProLIF (each label column) ~ z(ln IDs) + plate, and the same with
    z(cell_cycle_score) added; odds ratios with Wald 95% CI.
B.3 Depth-matched comparison, ProLIF versus DCT1+DCT2 (cluster 'Unknown'
    cells excluded): (i) 1:1 nearest neighbour on ln(IDs) within plate, without
    replacement, random order (seed 42), caliper 0.05 ln units; paired tests
    (McNemar exact for detection, Wilcoxon signed-rank for scores);
    (ii) strata = plate x IDs decile, ProLIF-weighted mean difference with a
    within-stratum label permutation test (2,000 permutations); and
    (iii) the unmatched comparison for reference. BH-FDR within label column x method.

Command (project root):  python scripts/reviewer_core_depth_detection.py
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
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

import scp_common as cc
import reviewer_core_lib as t2

t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "reviewer_core_depth_detection.py"
MARKERS = ["Mki67", "Top2a", "Slc12a3", "Pvalb", "Calb1"]
BINARY_METRICS = ["Slc12a3_detected", "Mki67_detected", "Top2a_detected", "Mki67_or_Top2a_detected"]
CONTINUOUS_METRICS = ["Slc12a3_ln1p_all_cells", "cell_cycle_score",
                      "V1_X__ALDO_activity__base", "V1_X__ALDO_activity__no_Slc12a3",
                      "V1_X__ALDO_signed__base", "V1_X__ALDO_signed__no_Slc12a3", "V1_X__ALDO_machinery__base"]
METHOD_COLORS = {"unmatched_all_cells": t2.INK["muted"], "nn_matched_within_plate": t2.SERIES[0],
                 "depth_decile_strata_within_plate": t2.SERIES[1]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reviewer_core B.1-B.3: depth/plate confounding of ProLIF.")
    parser.add_argument("--reference-min", type=float, default=0.40)
    parser.add_argument("--reference-max", type=float, default=0.80)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--caliper", type=float, default=0.05, help="Maximum |ln(IDs) difference| within a matched pair.")
    parser.add_argument("--n-permutations", type=int, default=2000)
    cc.add_common_args(parser)
    return parser.parse_args()


def logit_fit(y: np.ndarray, X: pd.DataFrame):
    """statsmodels Logit; returns (fit, note). Separation or non-convergence is reported, not raised."""
    if y.min() == y.max():
        return None, "outcome constant"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.Logit(y.astype(float), sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)
        except Exception as exc:  # perfect separation etc.
            return None, type(exc).__name__ + ": " + str(exc)[:120]
    note = "" if fit.mle_retvals.get("converged", True) else "not converged"
    return fit, note


# ---------------------------------------------------------------------------
# B.1
# ---------------------------------------------------------------------------
def detection_fits(detected: pd.DataFrame, proteins: list[str], protein_set: str, lx: np.ndarray,
                   labels: pd.DataFrame, depth_points: dict[str, float]) -> pd.DataFrame:
    rows = []
    groups = [("all_cells", "all", np.ones(len(labels), bool))]
    for col in t2.LABEL_COLUMNS:
        groups += [(col, lab, (labels[col] == lab).to_numpy()) for lab in t2.CORE_LABELS]
    for protein in proteins:
        y_all = detected[protein].to_numpy()
        for col, lab, mask in groups:
            y = y_all[mask]
            row = {"protein_set": protein_set, "protein": protein, "label_column": col, "label": lab,
                   "n_cells": int(mask.sum()), "n_detected": int(y.sum()), "fraction_detected": float(y.mean())}
            fit, note = logit_fit(y, pd.DataFrame({"ln_ids": lx[mask]}))
            row["note"] = note
            if fit is not None:
                ci = fit.conf_int().loc["ln_ids"]
                b0, b1 = fit.params["const"], fit.params["ln_ids"]
                row.update(intercept=b0, slope_per_ln_ids=b1, slope_ci_low=ci[0], slope_ci_high=ci[1],
                           p_value=fit.pvalues["ln_ids"], or_per_doubling_ids=np.exp(b1 * np.log(2)))
                for name, ids in depth_points.items():
                    row[f"p_detect_at_{name}"] = 1 / (1 + np.exp(-(b0 + b1 * np.log(ids))))
            rows.append(row)
    return pd.DataFrame(rows)


def adjusted_label_effects(detected: pd.DataFrame, proteins: list[str], protein_set: str, lx: np.ndarray,
                           labels: pd.DataFrame) -> pd.DataFrame:
    """detected ~ ln(IDs) + label (ref DCT1) + plate, per protein and label column."""
    rows = []
    plate_p2 = (labels["plate"] == "P2").astype(float).to_numpy()
    for col in t2.LABEL_COLUMNS:
        keep = labels[col].isin(t2.CORE_LABELS).to_numpy()
        lab = labels.loc[keep, col].to_numpy()
        X = pd.DataFrame({"ln_ids": lx[keep], "label_DCT2": (lab == "DCT2").astype(float),
                          "label_ProLIF": (lab == "ProLIF").astype(float), "plate_P2": plate_p2[keep]})
        for protein in proteins:
            y = detected[protein].to_numpy()[keep]
            fit, note = logit_fit(y, X)
            for term in X.columns:
                row = {"protein_set": protein_set, "protein": protein, "label_column": col, "term": term,
                       "n_cells": int(keep.sum()), "n_detected": int(y.sum()), "note": note}
                if fit is not None:
                    ci = fit.conf_int().loc[term]
                    row.update(coef=fit.params[term], odds_ratio=np.exp(fit.params[term]), or_ci_low=np.exp(ci[0]),
                               or_ci_high=np.exp(ci[1]), p_value=fit.pvalues[term])
                rows.append(row)
    return pd.DataFrame(rows)


def binned_detection(detected: pd.DataFrame, proteins: list[str], bins: pd.Series, ids: np.ndarray,
                     labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all_cells", "all", np.ones(len(labels), bool))]
    for col in t2.LABEL_COLUMNS:
        groups += [(col, lab, (labels[col] == lab).to_numpy()) for lab in cc.ordered_labels(labels[col])]
    codes = bins.cat.codes.to_numpy()
    for protein in proteins:
        y = detected[protein].to_numpy()
        for col, lab, mask in groups:
            for b, interval in enumerate(bins.cat.categories):
                m = mask & (codes == b)
                n, k = int(m.sum()), int(y[m].sum())
                lo, hi = t2.wilson_ci(k, n)
                rows.append({"protein": protein, "label_column": col, "label": lab, "ids_bin": b + 1,
                             "ids_bin_interval": str(interval), "median_ids_in_bin": float(np.median(ids[m])) if n else np.nan,
                             "n_cells": n, "n_detected": k, "fraction_detected": k / n if n else np.nan,
                             "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def plot_detection(binned: pd.DataFrame, fits: pd.DataFrame, ref_fits: pd.DataFrame, ref_binned: pd.DataFrame,
                   n_ref: int, labels: pd.DataFrame, run: cc.Run) -> None:
    ids = labels["n_detected_proteins"].to_numpy(float)
    grid = np.linspace(ids.min(), ids.max(), 200)
    for col in t2.LABEL_COLUMNS:
        fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.9), sharex=True, sharey=True)
        for ax, protein in zip(axes.flat, [*MARKERS, "reference"]):
            for lab in t2.CORE_LABELS:
                color = t2.LABEL_PALETTE[lab]
                lab_ids = ids[(labels[col] == lab).to_numpy()]
                lgrid = grid[(grid >= lab_ids.min()) & (grid <= lab_ids.max())]
                if protein == "reference":
                    part = ref_fits[(ref_fits["label_column"] == col) & (ref_fits["label"] == lab)].dropna(
                        subset=["slope_per_ln_ids"])
                    curves = 1 / (1 + np.exp(-(part["intercept"].to_numpy()[:, None]
                                               + part["slope_per_ln_ids"].to_numpy()[:, None] * np.log(lgrid)[None, :])))
                    q25, q50, q75 = np.percentile(curves, [25, 50, 75], axis=0)
                    ax.fill_between(lgrid, q25, q75, color=color, alpha=0.18, linewidth=0)
                    ax.plot(lgrid, q50, color=color, linewidth=1.3, label=lab)
                    pts = ref_binned[(ref_binned["label_column"] == col) & (ref_binned["label"] == lab)]
                    ax.scatter(pts["median_ids_in_bin"], pts["median_fraction"], s=9, color=color,
                               edgecolors="white", linewidths=0.4, zorder=3)
                else:
                    pts = binned[(binned["protein"] == protein) & (binned["label_column"] == col)
                                 & (binned["label"] == lab) & (binned["n_cells"] >= 5)]
                    ax.errorbar(pts["median_ids_in_bin"], pts["fraction_detected"],
                                yerr=[pts["fraction_detected"] - pts["ci_low"], pts["ci_high"] - pts["fraction_detected"]],
                                fmt="o", ms=2.8, color=color, elinewidth=0.5, capsize=0, mec="white", mew=0.3, zorder=3)
                    f = fits[(fits["protein"] == protein) & (fits["label_column"] == col) & (fits["label"] == lab)]
                    if len(f) and np.isfinite(f["slope_per_ln_ids"].iloc[0]):
                        b0, b1 = f["intercept"].iloc[0], f["slope_per_ln_ids"].iloc[0]
                        ax.plot(lgrid, 1 / (1 + np.exp(-(b0 + b1 * np.log(lgrid)))), color=color, linewidth=1.3, label=lab)
            title = (f"Reference set: {n_ref} proteins detected in 40-80% of cells,\nnot cell-cycle related "
                     "(line = median fit, band = IQR; dots = median binned fraction)") if protein == "reference" else protein
            ax.set_title(title, fontsize=6.8 if protein != "reference" else 5.6)
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
        for ax in axes[1]:
            ax.set_xlabel("Protein IDs per cell")
        for ax in axes[:, 0]:
            ax.set_ylabel("P(detected)")
        axes[0, 0].legend(loc="upper left", fontsize=5.5)
        fig.suptitle(f"Detection against depth by {t2.LABEL_DISPLAY[col]} (dots: deciles of IDs over all cells, "
                     "bins with >= 5 cells, Wilson 95% CI; lines: logistic fit on ln(IDs) within label)",
                     x=0.01, ha="left", fontsize=6.8, fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        t2.footer(fig, SCRIPT)
        run.figure(fig, f"depth_detection_vs_ids_by_{col}", "Detection probability against protein IDs per cell")


# ---------------------------------------------------------------------------
# B.2
# ---------------------------------------------------------------------------
def prolif_models(labels: pd.DataFrame, cc_score: np.ndarray) -> pd.DataFrame:
    lx = np.log(labels["n_detected_proteins"].to_numpy(float))
    sd_lx, sd_cc = lx.std(ddof=1), cc_score.std(ddof=1)
    base = pd.DataFrame({"z_ln_ids": (lx - lx.mean()) / sd_lx,
                         "plate_P2": (labels["plate"] == "P2").astype(float).to_numpy()})
    with_cc = base.assign(z_cell_cycle_score=(cc_score - cc_score.mean()) / sd_cc)
    unit_sd = {"z_ln_ids": sd_lx, "z_cell_cycle_score": sd_cc, "plate_P2": 1.0}
    rows = []
    for col in t2.LABEL_COLUMNS:
        y = (labels[col] == "ProLIF").to_numpy().astype(float)
        for model, X in (("ProLIF ~ ln_ids + plate", base), ("ProLIF ~ ln_ids + plate + cell_cycle_score", with_cc)):
            fit, note = logit_fit(y, X)
            for term in X.columns:
                row = {"label_column": col, "model": model, "term": term, "n": len(y), "n_prolif": int(y.sum()),
                       "note": note, "sd_of_unscaled_variable": unit_sd[term]}
                if fit is not None:
                    ci = fit.conf_int().loc[term]
                    row.update(coef=fit.params[term], se=fit.bse[term], odds_ratio=np.exp(fit.params[term]),
                               or_ci_low=np.exp(ci[0]), or_ci_high=np.exp(ci[1]), p_value=fit.pvalues[term],
                               aic=fit.aic, mcfadden_pseudo_r2=fit.prsquared,
                               or_per_unscaled_unit=np.exp(fit.params[term] / unit_sd[term]))
                rows.append(row)
    return pd.DataFrame(rows)


def plot_models(models: pd.DataFrame, run: cc.Run) -> None:
    terms = ["z_ln_ids", "plate_P2", "z_cell_cycle_score"]
    term_text = {"z_ln_ids": "ln(protein IDs), per SD", "plate_P2": "plate P2 vs P1",
                 "z_cell_cycle_score": "cell_cycle_score, per SD"}
    model_colors = dict(zip(models["model"].unique(), (t2.SERIES[0], t2.SERIES[1])))
    fig, axes = plt.subplots(1, len(t2.LABEL_COLUMNS), figsize=(7.0, 2.4), sharey=True, sharex=True)
    for ax, col in zip(axes, t2.LABEL_COLUMNS):
        sub = models[models["label_column"] == col]
        for k, (model, color) in enumerate(model_colors.items()):
            part = sub[sub["model"] == model].set_index("term").reindex(terms)
            y = np.arange(len(terms)) + (k - 0.5) * 0.3
            ok = part["odds_ratio"].notna().to_numpy()
            ax.errorbar(part["odds_ratio"][ok], y[ok], xerr=[(part["odds_ratio"] - part["or_ci_low"])[ok],
                                                             (part["or_ci_high"] - part["odds_ratio"])[ok]],
                        fmt="o", ms=3.5, color=color, elinewidth=0.9, capsize=0, label=model)
            for yy, (o, lo, hi) in zip(y[ok], part.loc[ok, ["odds_ratio", "or_ci_low", "or_ci_high"]].to_numpy()):
                ax.text(hi * 1.08, yy, f"{o:.2f} ({lo:.2f}-{hi:.2f})", va="center", fontsize=5.2, color=t2.INK["secondary"])
        ax.axvline(1, color=t2.INK["muted"], linewidth=0.6)
        ax.set_xscale("log")
        ax.set_yticks(range(len(terms)), [term_text[t] for t in terms])
        ax.invert_yaxis() if col == t2.LABEL_COLUMNS[0] else None
        n = int(sub["n_prolif"].iloc[0])
        ax.set_title(f"{t2.LABEL_DISPLAY[col]} (ProLIF n = {n} of {int(sub['n'].iloc[0])})", fontsize=7)
        ax.set_xlabel("Odds ratio for ProLIF label (95% CI, log scale)")
        ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.32), ncol=2, fontsize=5.8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "depth_prolif_logistic_models", "Logistic models of the ProLIF label on depth, plate and cell-cycle score")


# ---------------------------------------------------------------------------
# B.3
# ---------------------------------------------------------------------------
def nn_match(lx: np.ndarray, plate: np.ndarray, treated: np.ndarray, control: np.ndarray, caliper: float,
             rng: np.random.Generator) -> tuple[list[tuple[int, int]], list[int]]:
    pairs, unmatched = [], []
    for p in sorted(set(plate)):
        t_idx = np.flatnonzero(treated & (plate == p))
        c_idx = list(np.flatnonzero(control & (plate == p)))
        for t in rng.permutation(t_idx):
            if not c_idx:
                unmatched.append(int(t))
                continue
            d = np.abs(lx[c_idx] - lx[t])
            j = int(np.argmin(d))
            if d[j] <= caliper:
                pairs.append((int(t), int(c_idx.pop(j))))
            else:
                unmatched.append(int(t))
    return pairs, unmatched


def smd(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def strata_test(values: np.ndarray, is_t: np.ndarray, strata: np.ndarray, n_perm: int, rng) -> dict:
    keep_strata = [s for s in np.unique(strata) if is_t[strata == s].any() and (~is_t[strata == s]).any()]
    mask = np.isin(strata, keep_strata)
    v, t, s = values[mask], is_t[mask], strata[mask]

    def stat(tt):
        num = den = 0.0
        for st in keep_strata:
            m = s == st
            w = tt[m].sum()
            num += w * (v[m][tt[m]].mean() - v[m][~tt[m]].mean())
            den += w
        return num / den

    observed = stat(t)
    perm = np.empty(n_perm)
    groups = [np.flatnonzero(s == st) for st in keep_strata]
    for b in range(n_perm):
        tt = t.copy()
        for g in groups:
            tt[g] = rng.permutation(t[g])
        perm[b] = stat(tt)
    return {"weighted_difference": observed, "p_value": (1 + (np.abs(perm) >= abs(observed)).sum()) / (n_perm + 1),
            "n_strata_used": len(keep_strata), "n_treated_used": int(t.sum()), "n_control_used": int((~t).sum())}


def matched_comparisons(metrics: pd.DataFrame, labels: pd.DataFrame, args, rng_seed: int):
    ids = labels["n_detected_proteins"].to_numpy(float)
    lx = np.log(ids)
    plate = labels["plate"].to_numpy()
    deciles = pd.qcut(ids, args.n_bins, labels=False, duplicates="drop")
    strata = np.array([f"{p}_d{d}" for p, d in zip(plate, deciles)])
    rows, balance, pair_rows = [], [], []
    for col in t2.LABEL_COLUMNS:
        lab = labels[col].to_numpy()
        treated, control = lab == "ProLIF", np.isin(lab, ["DCT1", "DCT2"])
        rng = np.random.default_rng(rng_seed)
        pairs, unmatched = nn_match(lx, plate, treated, control, args.caliper, rng)
        t_idx = np.array([p[0] for p in pairs])
        c_idx = np.array([p[1] for p in pairs])
        for t, c in pairs:
            pair_rows.append({"label_column": col, "prolif_cell_id": labels.index[t], "matched_cell_id": labels.index[c],
                              "matched_label": lab[c], "plate": plate[t], "ids_prolif": int(ids[t]),
                              "ids_matched": int(ids[c]), "abs_ln_ids_difference": abs(lx[t] - lx[c])})
        balance.append({"label_column": col, "n_prolif": int(treated.sum()), "n_control_pool": int(control.sum()),
                        "n_pairs": len(pairs), "n_prolif_unmatched": len(unmatched),
                        "n_unmatched_P1": int(sum(plate[u] == "P1" for u in unmatched)),
                        "n_unmatched_P2": int(sum(plate[u] == "P2" for u in unmatched)),
                        "median_ids_unmatched_prolif": float(np.median(ids[unmatched])) if unmatched else np.nan,
                        "median_ids_prolif_all": float(np.median(ids[treated])),
                        "median_ids_control_all": float(np.median(ids[control])),
                        "median_ids_prolif_matched": float(np.median(ids[t_idx])),
                        "median_ids_control_matched": float(np.median(ids[c_idx])),
                        "smd_ln_ids_before": smd(lx[treated], lx[control]),
                        "smd_ln_ids_after": smd(lx[t_idx], lx[c_idx]),
                        "max_abs_ln_ids_difference_in_pair": float(np.max(np.abs(lx[t_idx] - lx[c_idx]))),
                        "matched_controls_DCT1": int((lab[c_idx] == "DCT1").sum()),
                        "matched_controls_DCT2": int((lab[c_idx] == "DCT2").sum()),
                        "caliper_ln_units": args.caliper})
        perm_rng = np.random.default_rng(rng_seed)
        for metric in [*BINARY_METRICS, *CONTINUOUS_METRICS]:
            values = metrics[metric].to_numpy(float)
            binary = metric in BINARY_METRICS
            base = {"label_column": col, "metric": metric, "kind": "binary" if binary else "continuous"}
            # unmatched
            row = {**base, "method": "unmatched_all_cells", "n_prolif": int(treated.sum()), "n_comparator": int(control.sum())}
            a, b = values[treated], values[control]
            if binary:
                table = [[int(a.sum()), int(len(a) - a.sum())], [int(b.sum()), int(len(b) - b.sum())]]
                row.update(value_prolif=a.mean(), value_comparator=b.mean(), difference=a.mean() - b.mean(),
                           test="Fisher exact", p_value=stats.fisher_exact(table)[1])
            else:
                res = cc.compare_groups(a, b)
                row.update(value_prolif=res["median_group"], value_comparator=res["median_reference"],
                           difference=res["median_difference"], test="Mann-Whitney U", p_value=res["p_value"])
            rows.append(row)
            # nearest-neighbour matched (paired)
            a, b = values[t_idx], values[c_idx]
            row = {**base, "method": "nn_matched_within_plate", "n_prolif": len(t_idx), "n_comparator": len(c_idx)}
            if binary:
                n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
                n11, n00 = int(((a == 1) & (b == 1)).sum()), int(((a == 0) & (b == 0)).sum())
                row.update(value_prolif=a.mean(), value_comparator=b.mean(), difference=a.mean() - b.mean(),
                           test="McNemar exact", p_value=mcnemar([[n11, n10], [n01, n00]], exact=True).pvalue,
                           discordant_prolif_only=n10, discordant_comparator_only=n01)
            else:
                diff = a - b
                row.update(value_prolif=float(np.median(a)), value_comparator=float(np.median(b)),
                           difference=float(np.median(diff)), test="Wilcoxon signed-rank (median of paired differences)",
                           p_value=float(stats.wilcoxon(a, b).pvalue))
            rows.append(row)
            # strata
            # Cells outside ProLIF/DCT1/DCT2 go to a stratum without ProLIF cells, which strata_test drops.
            st = strata_test(values, treated, np.where(treated | control, strata, "excluded"), args.n_permutations,
                             perm_rng)
            rows.append({**base, "method": "depth_decile_strata_within_plate", "n_prolif": st["n_treated_used"],
                         "n_comparator": st["n_control_used"], "difference": st["weighted_difference"],
                         "value_prolif": np.nan, "value_comparator": np.nan,
                         "test": f"within-stratum label permutation ({args.n_permutations})",
                         "p_value": st["p_value"], "n_strata_used": st["n_strata_used"]})
    out = cc.add_bh(pd.DataFrame(rows), ["label_column", "method"])
    return out, pd.DataFrame(balance), pd.DataFrame(pair_rows)


def plot_matching(comp: pd.DataFrame, pairs: pd.DataFrame, labels: pd.DataFrame, run: cc.Run) -> None:
    ids = labels["n_detected_proteins"].to_numpy(float)
    fig, axes = plt.subplots(len(t2.LABEL_COLUMNS), 3, figsize=(8.2, 5.4), gridspec_kw={"width_ratios": [1, 1.1, 1.3]})
    bins = np.linspace(np.log(ids.min()), np.log(ids.max()), 30)
    for r, col in enumerate(t2.LABEL_COLUMNS):
        lab = labels[col].to_numpy()
        ax = axes[r, 0]
        p = pairs[pairs["label_column"] == col]
        ax.hist(np.log(ids[lab == "ProLIF"]), bins=bins, histtype="step", color=t2.LABEL_PALETTE["ProLIF"],
                linewidth=1.0, label="ProLIF, all")
        ax.hist(np.log(ids[np.isin(lab, ["DCT1", "DCT2"])]), bins=bins, histtype="step", color=t2.INK["muted"],
                linewidth=1.0, label="DCT1+DCT2, all")
        ax.hist(np.log(p["ids_matched"].to_numpy(float)), bins=bins, histtype="stepfilled", color=t2.SERIES[0],
                alpha=0.35, linewidth=0, label=f"matched DCT1+DCT2 (n={len(p)})")
        ax.set_xlabel("ln(protein IDs per cell)")
        ax.set_ylabel("Cells")
        ax.set_title(f"{t2.LABEL_DISPLAY[col]}: depth before / after matching", fontsize=6.6)
        ax.legend(fontsize=5, loc="upper left")
        for c, kind in ((1, "binary"), (2, "continuous")):
            ax = axes[r, c]
            sub = comp[(comp["label_column"] == col) & (comp["kind"] == kind)]
            metrics = list(dict.fromkeys(sub["metric"]))
            for k, (method, color) in enumerate(METHOD_COLORS.items()):
                part = sub[sub["method"] == method].set_index("metric").reindex(metrics)
                y = np.arange(len(metrics)) + (k - 1) * 0.25
                sig = (part["p_fdr_bh"] < 0.05).to_numpy()
                ax.scatter(part["difference"][sig], y[sig], color=color, s=12, label=f"{method}, q < 0.05")
                ax.scatter(part["difference"][~sig], y[~sig], facecolors="white", edgecolors=color, s=12, linewidths=0.8,
                           label=f"{method}, q >= 0.05")
            ax.axvline(0, color=t2.INK["muted"], linewidth=0.6)
            ax.set_yticks(range(len(metrics)), [m.replace("V1_X__", "") for m in metrics], fontsize=5.5)
            ax.invert_yaxis()
            ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_title("Detection: difference in fraction\n(ProLIF minus DCT1+DCT2)" if kind == "binary"
                         else "Scores: median (paired) difference;\nstrata: ProLIF-weighted mean difference", fontsize=5.8)
    axes[0, 2].legend(fontsize=4.8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("ProLIF versus DCT1+DCT2 before and after depth matching within plate", x=0.01, ha="left",
                 fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "depth_matched_comparison", "ProLIF versus DCT1+DCT2 after depth matching")


def main() -> None:
    args = parse_args()
    run = cc.Run(t2.TASK, "depth_detection", args)
    adata, labels = t2.load_dct(run)
    task3 = t2.read_table(run, "task3_marker_scores_per_cell").set_index("cell_id").reindex(labels.index)
    aldo = t2.read_table(run, "aldo_scores_per_cell").set_index("cell_id").reindex(labels.index)
    corr = t2.read_table(run, "task4_mki67_top2a_correlated_proteins")
    ln1p = t2.ln1p_frame(adata)
    detected = ln1p > 0
    ids = labels["n_detected_proteins"].to_numpy(float)
    lx = np.log(ids)
    if not np.array_equal(detected.sum(axis=1).to_numpy(), ids.astype(int)):
        raise ValueError("n_detected_proteins does not equal the number of detected proteins in the intensity layer.")

    # B.1 reference set.
    frac = detected.mean(axis=0)
    excluded = set(t2.CELL_CYCLE_MARKERS) | set(t2.TIROSH_ALL) | set(corr.loc[corr["selected"], "protein"]) | set(MARKERS)
    in_range = (frac >= args.reference_min) & (frac <= args.reference_max)
    reference = [p for p in detected.columns if in_range[p] and p not in excluded]
    ref_table = pd.DataFrame({"protein": detected.columns, "fraction_detected": frac.to_numpy(),
                              "in_40_80_range": in_range.to_numpy(),
                              "excluded_cell_cycle_related": [p in excluded for p in detected.columns]})
    ref_table["in_reference_set"] = ref_table["protein"].isin(reference)
    run.table(ref_table[ref_table["in_40_80_range"]], "depth_reference_protein_set")
    run.note(f"Reference set: {len(reference)} proteins detected in {args.reference_min:.0%}-{args.reference_max:.0%} "
             f"of cells ({int(in_range.sum())} in range; {int((in_range & ref_table['excluded_cell_cycle_related'].to_numpy()).sum())} "
             "excluded as cell-cycle related or as the five markers).")
    depth_points = {"500_ids": 500.0, "median_ids": float(np.median(ids)), "1200_ids": 1200.0}
    bins = pd.Series(pd.qcut(ids, args.n_bins), index=labels.index)
    marker_binned = binned_detection(detected, MARKERS, bins, ids, labels)
    marker_fits = detection_fits(detected, MARKERS, "marker", lx, labels, depth_points)
    ref_fits = detection_fits(detected, reference, "reference", lx, labels, depth_points)
    ref_binned_long = binned_detection(detected, reference, bins, ids, labels)
    ref_binned = (ref_binned_long.groupby(["label_column", "label", "ids_bin", "ids_bin_interval"], sort=False)
                  .agg(n_proteins=("protein", "nunique"), n_cells=("n_cells", "first"),
                       median_ids_in_bin=("median_ids_in_bin", "first"),
                       median_fraction=("fraction_detected", "median"),
                       q25_fraction=("fraction_detected", lambda s: s.quantile(0.25)),
                       q75_fraction=("fraction_detected", lambda s: s.quantile(0.75))).reset_index())
    ref_summary = (ref_fits.dropna(subset=["slope_per_ln_ids"])
                   .assign(positive_slope_p05=lambda d: (d["slope_per_ln_ids"] > 0) & (d["p_value"] < 0.05))
                   .groupby(["label_column", "label"], sort=False)
                   .agg(n_proteins=("protein", "nunique"), median_slope=("slope_per_ln_ids", "median"),
                        q25_slope=("slope_per_ln_ids", lambda s: s.quantile(0.25)),
                        q75_slope=("slope_per_ln_ids", lambda s: s.quantile(0.75)),
                        median_or_per_doubling=("or_per_doubling_ids", "median"),
                        fraction_positive_slope_p05=("positive_slope_p05", "mean")).reset_index())
    run.table(marker_binned, "depth_detection_binned_markers")
    run.table(ref_binned, "depth_detection_binned_reference_summary")
    run.table(pd.concat([marker_fits, ref_fits], ignore_index=True), "depth_detection_logistic_fits")
    run.table(ref_summary, "depth_detection_reference_slope_summary")
    adj_markers = adjusted_label_effects(detected, MARKERS, "marker", lx, labels)
    adj_ref = adjusted_label_effects(detected, reference, "reference", lx, labels)
    adj_ref_summary = (adj_ref.dropna(subset=["odds_ratio"]).groupby(["label_column", "term"], sort=False)
                       .agg(n_proteins=("protein", "nunique"), median_odds_ratio=("odds_ratio", "median"),
                            q25_odds_ratio=("odds_ratio", lambda s: s.quantile(0.25)),
                            q75_odds_ratio=("odds_ratio", lambda s: s.quantile(0.75))).reset_index())
    adj_ref_summary.insert(0, "protein", "reference_set_median")
    run.table(pd.concat([adj_markers, adj_ref_summary], ignore_index=True), "depth_detection_adjusted_label_effects")
    run.note("Adjusted label effects: detected ~ ln(IDs) + label (DCT1 reference) + plate, cells with DCT1/DCT2/ProLIF "
             "labels only. For v1_cell_label, Slc12a3, Top2a and Mki67 are the ProLIF marker list itself.")
    plot_detection(marker_binned, marker_fits, ref_fits, ref_binned, len(reference), labels, run)

    # B.2
    cc_score = task3["cell_cycle_score"].to_numpy(float)
    models = prolif_models(labels, cc_score)
    run.table(models, "depth_prolif_logistic_models")
    plot_models(models, run)

    # B.3
    metrics = pd.DataFrame(index=labels.index)
    for marker in ("Slc12a3", "Mki67", "Top2a"):
        metrics[f"{marker}_detected"] = detected[marker].astype(float).to_numpy()
    metrics["Mki67_or_Top2a_detected"] = (detected["Mki67"] | detected["Top2a"]).astype(float).to_numpy()
    metrics["Slc12a3_ln1p_all_cells"] = ln1p["Slc12a3"].to_numpy()
    metrics["cell_cycle_score"] = cc_score
    for col in CONTINUOUS_METRICS:
        if col.startswith("V1_X__"):
            metrics[col] = aldo[col].to_numpy(float)
    comp, balance, pairs = matched_comparisons(metrics, labels, args, t2.SEED)
    run.table(balance, "depth_matching_balance")
    run.table(pairs, "depth_matched_pairs")
    run.table(comp, "depth_matched_comparisons")
    plot_matching(comp, pairs, labels, run)
    run.finish({"markers": MARKERS, "n_reference_proteins": len(reference), "depth_points": depth_points})


if __name__ == "__main__":
    main()
