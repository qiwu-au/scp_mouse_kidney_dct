#!/usr/bin/env python
"""dct_reviewer step 6a: reviewer_core B.1-B.3 depth and plate confounding of ProLIF per DCT condition (dct_labels labels).

Port of ctrl_reviewer_core_depth_detection.py (unchanged
method), run once per condition. Called unchanged under ``dct_reviewer_lib.bound``: ``binned_detection``,
``prolif_models``, ``logit_fit``, ``nn_match``, ``smd`` and ``strata_test``. Copied with changes:
``detection_fits`` and ``adjusted_label_effects`` (labels absent under a column get 'does not apply'
rows instead of a fit; the label terms of absent labels are left out of the adjusted model) and
``matched_comparisons`` (metric list passed in; the ALDO source 'ctrl_X' is named 'scaled_X').

Inputs (read-only): dct_labels labelled objects (intensity layer) and canonical labels; dct_reviewer outputs
dct_reviewer_task3_marker_scores_per_cell.csv (cell_cycle_score), dct_reviewer_aldo_scores_per_cell.csv
(scaled_X ALDO scores) and dct_reviewer_task4_mki67_top2a_correlated_proteins.csv (circular set).
Outputs:
  results/tables/dct_reviewer/dct_reviewer_depth_*.csv (both conditions; ``condition`` column)
  results/figures/dct_reviewer/dct_reviewer_depth_detection_vs_ids_<role>.pdf/.png/.tif
  results/figures/dct_reviewer/dct_reviewer_depth_prolif_logistic_models.pdf/.png/.tif
  results/figures/dct_reviewer/dct_reviewer_depth_matched_comparison.pdf/.png/.tif

B.1 (all label columns) detection (intensity > 0) against protein IDs per cell: deciles of IDs within the
    condition, logistic fits detected ~ ln(IDs) per label for Mki67, Top2a, Slc12a3, Pvalb, Calb1 and a
    reference set (proteins detected in 40-80% of the condition's cells, not cell-cycle related); fitted
    P(detected) at the condition's quartiles of IDs; detected ~ ln(IDs) + label + plate per protein.
B.2 ProLIF ~ z(ln IDs) + plate (+ z(cell_cycle_score)): label columns with ProLIF cells only.
B.3 ProLIF vs DCT1+DCT2, 1:1 nearest neighbour on ln(IDs) within plate (caliper 0.05 ln units, seed 42),
    depth-decile strata within plate (2,000 permutations) and unmatched; label columns with ProLIF only.
    BH-FDR within condition x label column x method.

Command (project root):  python scripts/dct_reviewer_depth_detection.py
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
from statsmodels.stats.contingency_tables import mcnemar

import scp_common as cc
import reviewer_core_lib as t2
import ctrl_reviewer_core_depth_detection as m19
import dct_reviewer_lib as lib

lib.style()
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "dct_reviewer_depth_detection.py"
MARKERS = m19.MARKERS
BINARY_METRICS = m19.BINARY_METRICS
CONTINUOUS_METRICS = [m.replace("ctrl_X__", "scaled_X__") for m in m19.CONTINUOUS_METRICS]
METHOD_COLORS = m19.METHOD_COLORS
METHOD_DISPLAY = {"unmatched_all_cells": "unmatched", "nn_matched_within_plate": "1:1 matched within plate",
                  "depth_decile_strata_within_plate": "depth-decile strata within plate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 6a: depth/plate confounding of ProLIF per DCT condition.")
    parser.add_argument("--reference-min", type=float, default=0.40)
    parser.add_argument("--reference-max", type=float, default=0.80)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--caliper", type=float, default=0.05, help="Maximum |ln(IDs) difference| within a matched pair.")
    parser.add_argument("--n-permutations", type=int, default=2000)
    cc.add_common_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# B.1 (copies of the ctrl_reviewer_core functions; absent labels give 'does not apply' rows)
# ---------------------------------------------------------------------------
def detection_fits(s: lib.Settings, detected: pd.DataFrame, proteins: list[str], protein_set: str, lx: np.ndarray,
                   labels: pd.DataFrame, depth_points: dict[str, float]) -> pd.DataFrame:
    rows = []
    groups = [("all_cells", "all", np.ones(len(labels), bool))]
    for col in s.LABEL_COLUMNS:
        groups += [(col, lab, (labels[col] == lab).to_numpy()) for lab in t2.CORE_LABELS]
    for protein in proteins:
        y_all = detected[protein].to_numpy()
        for col, lab, mask in groups:
            y = y_all[mask]
            row = {"protein_set": protein_set, "protein": protein, "label_column": col, "label": lab,
                   "n_cells": int(mask.sum()), "n_detected": int(y.sum()),
                   "fraction_detected": float(y.mean()) if mask.any() else np.nan}
            if not mask.any():
                row["note"] = f"{lib.NOT_APPLICABLE}: no {lab} cell under {col}"
                rows.append(row)
                continue
            fit, note = m19.logit_fit(y, pd.DataFrame({"ln_ids": lx[mask]}))
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


def adjusted_label_effects(s: lib.Settings, detected: pd.DataFrame, proteins: list[str], protein_set: str,
                           lx: np.ndarray, labels: pd.DataFrame) -> pd.DataFrame:
    """detected ~ ln(IDs) + label (ref DCT1) + plate, per protein and label column; absent labels have no term."""
    rows = []
    plate_p2 = (labels["plate"] == "P2").astype(float).to_numpy()
    for col in s.LABEL_COLUMNS:
        keep = labels[col].isin(t2.CORE_LABELS).to_numpy()
        lab = labels.loc[keep, col].to_numpy()
        terms = {"ln_ids": lx[keep]}
        absent = []
        for other in ("DCT2", "ProLIF"):
            if (lab == other).any():
                terms[f"label_{other}"] = (lab == other).astype(float)
            else:
                absent.append(f"label_{other}")
        terms["plate_P2"] = plate_p2[keep]
        X = pd.DataFrame(terms)
        for protein in proteins:
            y = detected[protein].to_numpy()[keep]
            fit, note = m19.logit_fit(y, X)
            for term in X.columns:
                row = {"protein_set": protein_set, "protein": protein, "label_column": col, "term": term,
                       "n_cells": int(keep.sum()), "n_detected": int(y.sum()), "note": note}
                if fit is not None:
                    ci = fit.conf_int().loc[term]
                    row.update(coef=fit.params[term], odds_ratio=np.exp(fit.params[term]), or_ci_low=np.exp(ci[0]),
                               or_ci_high=np.exp(ci[1]), p_value=fit.pvalues[term])
                rows.append(row)
            for term in absent:
                rows.append({"protein_set": protein_set, "protein": protein, "label_column": col, "term": term,
                             "n_cells": int(keep.sum()), "n_detected": int(y.sum()),
                             "note": f"{lib.NOT_APPLICABLE}: no {term.removeprefix('label_')} cell under {col}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B.3 (copy of ctrl_reviewer_core matched_comparisons with the metric list passed in)
# ---------------------------------------------------------------------------
def matched_comparisons(s: lib.Settings, metrics: pd.DataFrame, labels: pd.DataFrame, args, rng_seed: int,
                        continuous: list[str]):
    ids = labels["n_detected_proteins"].to_numpy(float)
    lx = np.log(ids)
    plate = labels["plate"].to_numpy()
    deciles = pd.qcut(ids, args.n_bins, labels=False, duplicates="drop")
    strata = np.array([f"{p}_d{d}" for p, d in zip(plate, deciles)])
    rows, balance, pair_rows = [], [], []
    for col in s.LABEL_COLUMNS:
        lab = labels[col].to_numpy()
        treated, control = lab == "ProLIF", np.isin(lab, ["DCT1", "DCT2"])
        rng = np.random.default_rng(rng_seed)
        pairs, unmatched = m19.nn_match(lx, plate, treated, control, args.caliper, rng)
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
                        "smd_ln_ids_before": m19.smd(lx[treated], lx[control]),
                        "smd_ln_ids_after": m19.smd(lx[t_idx], lx[c_idx]),
                        "max_abs_ln_ids_difference_in_pair": float(np.max(np.abs(lx[t_idx] - lx[c_idx]))),
                        "matched_controls_DCT1": int((lab[c_idx] == "DCT1").sum()),
                        "matched_controls_DCT2": int((lab[c_idx] == "DCT2").sum()),
                        "caliper_ln_units": args.caliper})
        perm_rng = np.random.default_rng(rng_seed)
        for metric in [*BINARY_METRICS, *continuous]:
            values = metrics[metric].to_numpy(float)
            binary = metric in BINARY_METRICS
            base = {"label_column": col, "metric": metric, "kind": "binary" if binary else "continuous"}
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
            st = m19.strata_test(values, treated, np.where(treated | control, strata, "excluded"), args.n_permutations,
                                 perm_rng)
            rows.append({**base, "method": "depth_decile_strata_within_plate", "n_prolif": st["n_treated_used"],
                         "n_comparator": st["n_control_used"], "difference": st["weighted_difference"],
                         "value_prolif": np.nan, "value_comparator": np.nan,
                         "test": f"within-stratum label permutation ({args.n_permutations})",
                         "p_value": st["p_value"], "n_strata_used": st["n_strata_used"]})
    out = cc.add_bh(pd.DataFrame(rows), ["label_column", "method"])
    return out, pd.DataFrame(balance), pd.DataFrame(pair_rows)


def with_na(frame: pd.DataFrame, cond: str, missing: list[str], analysis: str) -> pd.DataFrame:
    """Computed rows (status 'computed') plus one 'does not apply' row per label column without ProLIF."""
    na = lib.na_rows(cond, missing, analysis, "no ProLIF cell under this column").drop(columns=["condition", "label_role"])
    return pd.concat([frame.assign(status="computed"), na], ignore_index=True)


def one_condition(run: cc.Run, cond: str, args) -> dict:
    s = lib.SETTINGS[cond]
    adata, labels = lib.load(run, cond)
    task3 = lib.read_table(run, "task3_marker_scores_per_cell", dtype={"cell_id": str})
    task3 = task3[task3["condition"] == cond].set_index("cell_id").reindex(labels.index)
    aldo = lib.read_table(run, "aldo_scores_per_cell", dtype={"cell_id": str})
    aldo = aldo[aldo["condition"] == cond].set_index("cell_id").reindex(labels.index)
    corr = lib.read_table(run, "task4_mki67_top2a_correlated_proteins")
    corr = corr[corr["condition"] == cond]
    if task3["cell_cycle_score"].isna().any() or aldo[CONTINUOUS_METRICS[2]].isna().any():
        raise ValueError(f"{cond}: dct_reviewer step 1 or step 3 per-cell table does not cover every cell.")
    ln1p = t2.ln1p_frame(adata)
    detected = ln1p > 0
    ids = labels["n_detected_proteins"].to_numpy(float)
    lx = np.log(ids)
    if not np.array_equal(detected.sum(axis=1).to_numpy(), ids.astype(int)):
        raise ValueError(f"{cond}: n_detected_proteins does not equal the number of detected proteins in the "
                         "intensity layer.")
    out = {"labels": labels}

    frac = detected.mean(axis=0)
    excluded = set(t2.CELL_CYCLE_MARKERS) | set(t2.TIROSH_ALL) | set(corr.loc[corr["selected"], "protein"]) | set(MARKERS)
    in_range = (frac >= args.reference_min) & (frac <= args.reference_max)
    reference = [p for p in detected.columns if in_range[p] and p not in excluded]
    ref_table = pd.DataFrame({"protein": detected.columns, "fraction_detected": frac.to_numpy(),
                              "in_40_80_range": in_range.to_numpy(),
                              "excluded_cell_cycle_related": [p in excluded for p in detected.columns]})
    ref_table["in_reference_set"] = ref_table["protein"].isin(reference)
    out["ref_table"] = ref_table[ref_table["in_40_80_range"]]
    out["n_ref"], out["n_in_range"] = len(reference), int(in_range.sum())
    out["n_ref_excluded"] = int((in_range & ref_table["excluded_cell_cycle_related"].to_numpy()).sum())
    depth_points = {"q25_ids": float(np.quantile(ids, 0.25)), "median_ids": float(np.median(ids)),
                    "q75_ids": float(np.quantile(ids, 0.75))}
    out["depth_points"] = depth_points
    bins = pd.Series(pd.qcut(ids, args.n_bins), index=labels.index)
    with lib.bound(s, m19):
        marker_binned = m19.binned_detection(detected, MARKERS, bins, ids, labels)
        ref_binned_long = m19.binned_detection(detected, reference, bins, ids, labels)
    marker_fits = detection_fits(s, detected, MARKERS, "marker", lx, labels, depth_points)
    ref_fits = detection_fits(s, detected, reference, "reference", lx, labels, depth_points)
    out["ref_binned"] = (ref_binned_long.groupby(["label_column", "label", "ids_bin", "ids_bin_interval"], sort=False)
                         .agg(n_proteins=("protein", "nunique"), n_cells=("n_cells", "first"),
                              median_ids_in_bin=("median_ids_in_bin", "first"),
                              median_fraction=("fraction_detected", "median"),
                              q25_fraction=("fraction_detected", lambda v: v.quantile(0.25)),
                              q75_fraction=("fraction_detected", lambda v: v.quantile(0.75))).reset_index())
    out["ref_summary"] = (ref_fits.dropna(subset=["slope_per_ln_ids"])
                          .assign(positive_slope_p05=lambda d: (d["slope_per_ln_ids"] > 0) & (d["p_value"] < 0.05))
                          .groupby(["label_column", "label"], sort=False)
                          .agg(n_proteins=("protein", "nunique"), median_slope=("slope_per_ln_ids", "median"),
                               q25_slope=("slope_per_ln_ids", lambda v: v.quantile(0.25)),
                               q75_slope=("slope_per_ln_ids", lambda v: v.quantile(0.75)),
                               median_or_per_doubling=("or_per_doubling_ids", "median"),
                               fraction_positive_slope_p05=("positive_slope_p05", "mean")).reset_index())
    out["marker_binned"], out["marker_fits"], out["ref_fits"] = marker_binned, marker_fits, ref_fits
    out["fits"] = pd.concat([marker_fits, ref_fits], ignore_index=True)
    adj_markers = adjusted_label_effects(s, detected, MARKERS, "marker", lx, labels)
    adj_ref = adjusted_label_effects(s, detected, reference, "reference", lx, labels)
    adj_ref_summary = (adj_ref.dropna(subset=["odds_ratio"]).groupby(["label_column", "term"], sort=False)
                       .agg(n_proteins=("protein", "nunique"), median_odds_ratio=("odds_ratio", "median"),
                            q25_odds_ratio=("odds_ratio", lambda v: v.quantile(0.25)),
                            q75_odds_ratio=("odds_ratio", lambda v: v.quantile(0.75))).reset_index())
    adj_ref_summary.insert(0, "protein", "reference_set_median")
    out["adjusted"] = pd.concat([adj_markers, adj_ref_summary], ignore_index=True)

    # B.2 and B.3: label columns with ProLIF cells only.
    with_prolif = lib.columns_with(labels, s, "ProLIF")
    missing = [c for c in s.LABEL_COLUMNS if c not in with_prolif]
    out["with_prolif"], out["missing"] = with_prolif, missing
    sub = s.subset(with_prolif)
    cc_score = task3["cell_cycle_score"].to_numpy(float)
    with lib.bound(sub, m19):
        models = m19.prolif_models(labels, cc_score)
    out["models"] = with_na(models, cond, missing, "B.2 ProLIF logistic models")
    metrics = pd.DataFrame(index=labels.index)
    for marker in ("Slc12a3", "Mki67", "Top2a"):
        metrics[f"{marker}_detected"] = detected[marker].astype(float).to_numpy()
    metrics["Mki67_or_Top2a_detected"] = (detected["Mki67"] | detected["Top2a"]).astype(float).to_numpy()
    metrics["Slc12a3_ln1p_all_cells"] = ln1p["Slc12a3"].to_numpy()
    metrics["cell_cycle_score"] = cc_score
    for col in CONTINUOUS_METRICS:
        if col.startswith("scaled_X__"):
            metrics[col] = aldo[col].to_numpy(float)
    comp, balance, pairs = matched_comparisons(sub, metrics, labels, args, t2.SEED, CONTINUOUS_METRICS)
    out["comp"] = with_na(comp, cond, missing, "B.3 depth-matched comparisons")
    out["balance"] = with_na(balance, cond, missing, "B.3 depth matching")
    out["pairs"] = pairs
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_detection(res: dict, role: str, figs: lib.Figures) -> None:
    proteins = [*MARKERS, "reference"]
    n_pairs = 2
    n_rows = int(np.ceil(len(proteins) / n_pairs))
    fig = lib.new_figure(lib.FULL_MM, 58 * n_rows + 16)
    gs = fig.add_gridspec(n_rows, 2 * n_pairs, left=0.075, right=0.99, top=0.855, bottom=0.07, hspace=0.62,
                          wspace=0.12)
    all_ids = np.concatenate([res[c]["labels"]["n_detected_proteins"].to_numpy(float) for c in lib.CONDITIONS])
    xlim = (0, all_ids.max() * 1.03)
    for i, protein in enumerate(proteins):
        r, p = divmod(i, n_pairs)
        for j, cond in enumerate(lib.CONDITIONS):
            ax = fig.add_subplot(gs[r, 2 * p + j])
            d, col = res[cond], lib.SETTINGS[cond].COLUMN_OF[role]
            labels = d["labels"]
            ids = labels["n_detected_proteins"].to_numpy(float)
            grid = np.linspace(ids.min(), ids.max(), 200)
            for lab in t2.CORE_LABELS:
                in_lab = (labels[col] == lab).to_numpy()
                if not in_lab.any():
                    continue
                color = t2.LABEL_PALETTE[lab]
                lgrid = grid[(grid >= ids[in_lab].min()) & (grid <= ids[in_lab].max())]
                if protein == "reference":
                    part = d["ref_fits"][(d["ref_fits"]["label_column"] == col) & (d["ref_fits"]["label"] == lab)].dropna(
                        subset=["slope_per_ln_ids"])
                    curves = 1 / (1 + np.exp(-(part["intercept"].to_numpy()[:, None]
                                               + part["slope_per_ln_ids"].to_numpy()[:, None] * np.log(lgrid)[None, :])))
                    q25, q50, q75 = np.percentile(curves, [25, 50, 75], axis=0)
                    ax.fill_between(lgrid, q25, q75, color=color, alpha=0.18, linewidth=0)
                    ax.plot(lgrid, q50, color=color, linewidth=1.2)
                    pts = d["ref_binned"][(d["ref_binned"]["label_column"] == col) & (d["ref_binned"]["label"] == lab)]
                    ax.scatter(pts["median_ids_in_bin"], pts["median_fraction"], s=8, color=color,
                               edgecolors="white", linewidths=0.4, zorder=3)
                else:
                    b = d["marker_binned"]
                    pts = b[(b["protein"] == protein) & (b["label_column"] == col) & (b["label"] == lab)
                            & (b["n_cells"] >= 5)]
                    ax.errorbar(pts["median_ids_in_bin"], pts["fraction_detected"],
                                yerr=[pts["fraction_detected"] - pts["ci_low"], pts["ci_high"] - pts["fraction_detected"]],
                                fmt="o", ms=2.6, color=color, elinewidth=0.5, capsize=0, mec="white", mew=0.3, zorder=3)
                    f = d["marker_fits"]
                    f = f[(f["protein"] == protein) & (f["label_column"] == col) & (f["label"] == lab)]
                    if len(f) and np.isfinite(f["slope_per_ln_ids"].iloc[0]):
                        b0, b1 = f["intercept"].iloc[0], f["slope_per_ln_ids"].iloc[0]
                        ax.plot(lgrid, 1 / (1 + np.exp(-(b0 + b1 * np.log(lgrid)))), color=color, linewidth=1.2)
            name = f"reference set ({d['n_ref']})" if protein == "reference" else protein
            ax.set_title(f"{name}: {cond}", fontsize=7)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.02, 1.02)
            lib.grid_y(ax)
            if j == 0 and p == 0:
                ax.set_ylabel("P(detected)")
            else:
                ax.tick_params(labelleft=False)
            if r == n_rows - 1:
                ax.set_xlabel("protein IDs per cell")
    present = lib.union_order(*(res[c]["labels"][lib.SETTINGS[c].COLUMN_OF[role]] for c in lib.CONDITIONS),
                              order=list(t2.CORE_LABELS))
    handles = [Line2D([], [], color=t2.LABEL_PALETTE[l], marker="o", markersize=3, linewidth=1.2, label=l)
               for l in present if l in t2.CORE_LABELS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.07, 0.945), ncol=len(handles), frameon=False)
    fig.text(0.005, 0.995, f"Detection against depth by {lib.ROLE_DISPLAY[role]}\ndots = deciles of IDs within the "
             "condition (bins >= 5 cells, Wilson 95% CI); lines = logistic fit on ln(IDs) within label\nreference set "
             "= proteins detected in 40-80% of the condition's cells, not cell-cycle related (line = median fit, band = "
             "IQR)", ha="left", va="top", fontsize=7)
    figs.save(fig, f"depth_detection_vs_ids_{role}", f"Detection against protein IDs, {role} label, Ctrl and ZeroK",
              label_role=role)


def plot_models(res: dict, figs: lib.Figures) -> None:
    terms = ["z_ln_ids", "plate_P2", "z_cell_cycle_score"]
    term_text = {"z_ln_ids": "ln(protein IDs), per SD", "plate_P2": "plate P2 vs P1",
                 "z_cell_cycle_score": "cell_cycle_score, per SD"}
    models = pd.concat([res[c]["models"].assign(condition=c) for c in lib.CONDITIONS])
    models = models[models["status"] == "computed"]
    model_colors = dict(zip(models["model"].unique(), (t2.SERIES[0], t2.SERIES[1])))
    lo = models["or_ci_low"].min()
    hi = models["or_ci_high"].max()
    fig = lib.new_figure(lib.FULL_MM, 72)
    gs = fig.add_gridspec(1, 2, left=0.215, right=0.99, top=0.8, bottom=0.3, wspace=0.08)
    for k, cond in enumerate(lib.CONDITIONS):
        ax = fig.add_subplot(gs[0, k])
        sub = models[models["condition"] == cond]
        for m, (model, color) in enumerate(model_colors.items()):
            part = sub[sub["model"] == model].set_index("term").reindex(terms)
            y = np.arange(len(terms)) + (m - 0.5) * 0.3
            ok = part["odds_ratio"].notna().to_numpy()
            ax.errorbar(part["odds_ratio"][ok], y[ok], xerr=[(part["odds_ratio"] - part["or_ci_low"])[ok],
                                                             (part["or_ci_high"] - part["odds_ratio"])[ok]],
                        fmt="o", ms=3.2, color=color, elinewidth=0.9, capsize=0, label=model)
        ax.axvline(1, color=t2.INK["muted"], linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlim(lo / 1.3, hi * 1.3)
        ax.set_yticks(range(len(terms)), [term_text[t] for t in terms] if k == 0 else [""] * len(terms))
        ax.set_ylim(len(terms) - 0.5, -0.5)
        col = sub["label_column"].iloc[0]
        ax.set_title(f"{cond}: {col} (ProLIF {int(sub['n_prolif'].iloc[0])} of {int(sub['n'].iloc[0])})", fontsize=7)
        ax.set_xlabel("odds ratio for the ProLIF label (95% CI, log scale)")
        ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        if k == 0:
            handles, labs = ax.get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower left", bbox_to_anchor=(0.16, 0.0), ncol=2, frameon=False)
    na = "; ".join(f"{c}: {', '.join(res[c]['missing'])}" for c in lib.CONDITIONS if res[c]["missing"])
    fig.text(0.005, 0.995, "Logistic models of the ProLIF label on depth, plate and cell-cycle score (second label "
             f"column; values in dct_reviewer_depth_prolif_logistic_models.csv)\n{lib.NOT_APPLICABLE} (no ProLIF cell): "
             f"{na}", ha="left", va="top", fontsize=7)
    figs.save(fig, "depth_prolif_logistic_models", "Logistic models of the ProLIF label, Ctrl and ZeroK (second label)")


def plot_matching(res: dict, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 170)
    gs = fig.add_gridspec(3, 2, left=0.205, right=0.99, top=0.9, bottom=0.12, hspace=0.6, wspace=0.08,
                          height_ratios=[1.1, 0.75, 1.15])
    all_ids = np.concatenate([res[c]["labels"]["n_detected_proteins"].to_numpy(float) for c in lib.CONDITIONS])
    bins = np.linspace(np.log(all_ids.min()), np.log(all_ids.max()), 30)
    comp = pd.concat([res[c]["comp"].assign(condition=c) for c in lib.CONDITIONS])
    comp = comp[comp["status"] == "computed"]
    xl = {kind: (comp.loc[comp["kind"] == kind, "difference"].min(), comp.loc[comp["kind"] == kind, "difference"].max())
          for kind in ("binary", "continuous")}
    first_hist = None
    for k, cond in enumerate(lib.CONDITIONS):
        d = res[cond]
        col = d["with_prolif"][0]
        labels = d["labels"]
        ids = labels["n_detected_proteins"].to_numpy(float)
        lab = labels[col].to_numpy()
        p = d["pairs"][d["pairs"]["label_column"] == col]
        ax = fig.add_subplot(gs[0, k], sharey=first_hist)
        first_hist = first_hist or ax
        ax.hist(np.log(ids[lab == "ProLIF"]), bins=bins, histtype="step", color=t2.LABEL_PALETTE["ProLIF"],
                linewidth=1.0, label="ProLIF, all")
        ax.hist(np.log(ids[np.isin(lab, ["DCT1", "DCT2"])]), bins=bins, histtype="step", color=t2.INK["muted"],
                linewidth=1.0, label="DCT1+DCT2, all")
        ax.hist(np.log(p["ids_matched"].to_numpy(float)), bins=bins, histtype="stepfilled", color=t2.SERIES[0],
                alpha=0.35, linewidth=0, label="matched DCT1+DCT2")
        ax.set_xlabel("ln(protein IDs per cell)")
        if k == 0:
            ax.set_ylabel("cells")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"{cond}: {col}\nProLIF {int((lab == 'ProLIF').sum())}, pairs {len(p)}", fontsize=7)
        if k == 1:
            ax.legend(loc="upper left", frameon=False)
        for r, kind in ((1, "binary"), (2, "continuous")):
            ax = fig.add_subplot(gs[r, k])
            sub = comp[(comp["condition"] == cond) & (comp["label_column"] == col) & (comp["kind"] == kind)]
            metrics = list(dict.fromkeys(sub["metric"]))
            for m, (method, color) in enumerate(METHOD_COLORS.items()):
                part = sub[sub["method"] == method].set_index("metric").reindex(metrics)
                y = np.arange(len(metrics)) + (m - 1) * 0.25
                sig = (part["p_fdr_bh"] < 0.05).to_numpy()
                x = part["difference"].to_numpy(float)
                ax.scatter(x[sig], y[sig], color=color, s=10, linewidths=0)
                ax.scatter(x[~sig], y[~sig], facecolors="white", edgecolors=color, s=10, linewidths=0.8)
            ax.axvline(0, color=t2.INK["muted"], linewidth=0.6)
            ax.set_yticks(range(len(metrics)), [mm.replace("scaled_X__", "") for mm in metrics] if k == 0
                          else [""] * len(metrics))
            ax.set_ylim(len(metrics) - 0.5, -0.5)
            pad = 0.06 * (xl[kind][1] - xl[kind][0])
            ax.set_xlim(min(xl[kind][0], 0) - pad, max(xl[kind][1], 0) + pad)
            ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{cond}: detection, difference in fraction" if kind == "binary"
                         else f"{cond}: scores, median (paired) difference", fontsize=7)
            if r == 2:
                ax.set_xlabel("ProLIF minus DCT1+DCT2")
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=4, label=METHOD_DISPLAY[k])
               for k, c in METHOD_COLORS.items()]
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#52514e",
                          markersize=4, label="open: BH q >= 0.05"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=4, frameon=False)
    na = "; ".join(f"{c}: {', '.join(res[c]['missing'])}" for c in lib.CONDITIONS if res[c]["missing"])
    fig.text(0.005, 0.995, "ProLIF versus DCT1+DCT2 before and after depth matching within plate (second label column; "
             "strata: ProLIF-weighted mean difference)\n"
             f"filled: BH q < 0.05 within condition x label column x method; {lib.NOT_APPLICABLE} (no ProLIF cell): {na}",
             ha="left", va="top", fontsize=7)
    figs.save(fig, "depth_matched_comparison", "ProLIF versus DCT1+DCT2 after depth matching, Ctrl and ZeroK")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "depth_detection", args)
    lib.ported_from(run, "ctrl_reviewer_core_depth_detection.py")
    res = {c: one_condition(run, c, args) for c in lib.CONDITIONS}
    for cond in lib.CONDITIONS:
        d = res[cond]
        run.note(f"{cond}: reference set {d['n_ref']} proteins detected in {args.reference_min:.0%}-"
                 f"{args.reference_max:.0%} of cells ({d['n_in_range']} in range; {d['n_ref_excluded']} excluded as "
                 "cell-cycle related or as the five markers). Depth points (IDs): "
                 + ", ".join(f"{k} {v:.0f}" for k, v in d["depth_points"].items()) + ".")
        run.note(f"{cond}: B.2 and B.3 run on {d['with_prolif'] or 'no column'}; {lib.NOT_APPLICABLE} under "
                 f"{d['missing'] or 'no column'} (no ProLIF cell).")
    for key, name in (("ref_table", "depth_reference_protein_set"), ("marker_binned", "depth_detection_binned_markers"),
                      ("ref_binned", "depth_detection_binned_reference_summary"),
                      ("fits", "depth_detection_logistic_fits"), ("ref_summary", "depth_detection_reference_slope_summary"),
                      ("adjusted", "depth_detection_adjusted_label_effects"), ("models", "depth_prolif_logistic_models"),
                      ("balance", "depth_matching_balance"), ("pairs", "depth_matched_pairs"),
                      ("comp", "depth_matched_comparisons")):
        run.table(lib.stack({c: res[c][key] for c in lib.CONDITIONS}), name)
    run.table(pd.DataFrame([{"condition": c, **res[c]["depth_points"]} for c in lib.CONDITIONS]), "depth_points")
    run.note("Adjusted label effects: detected ~ ln(IDs) + label (DCT1 reference) + plate, cells with DCT1/DCT2/ProLIF "
             "labels only; a label without cells under a column has no term ('does not apply' rows).")
    figs = lib.Figures(run, "depth_detection")
    for role in lib.ROLES:
        plot_detection(res, role, figs)
    plot_models(res, figs)
    plot_matching(res, figs)
    figs.finish()
    run.finish({"markers": MARKERS, "n_reference_proteins": {c: res[c]["n_ref"] for c in lib.CONDITIONS},
                "depth_points": {c: res[c]["depth_points"] for c in lib.CONDITIONS},
                "b2_b3_columns": {c: res[c]["with_prolif"] for c in lib.CONDITIONS}, "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_depth_detection.csv.")


if __name__ == "__main__":
    main()
