#!/usr/bin/env python
"""qc_mito tests A-D: does the mitochondrial fraction behave like a cell-quality metric?

Added to the qc_mito plan
with the authors (13 Oct 2026) to test, objectively, the premises of using a
mitochondrial protein fraction as a QC metric as in scRNA-seq.

Inputs (read-only):
  results/tables/qc_mito/qc_mito_{kidney,dct}_qc_per_cell.csv (step mito_qc)
  results/tables/qc_mito/qc_mito_mitocarta_mouse_list.csv and *_protein_annotation.csv (step mitocarta)
  data/DCT/DCT_758cells.csv, data/KIDNEY/KIDNEY_768cells.csv (raw matrices, for test D only)
Outputs (results/tables/qc_mito/, results/figures/qc_mito/):
  A  qc_mito_variance_explained.csv          R2 of label, depth and both (OLS; raw and rank scale)
  B  qc_mito_global_cutoff_simulation.csv     share of each label removed by global cutoffs
     qc_mito_global_cutoff_tests.csv          label x removed: chi-square with permutation p
  C  qc_mito_within_type_depth_correlation.csv  Spearman of mito metrics with depth inside each label
  D  qc_mito_all_raw_cells_mito.csv, qc_mito_removed_vs_retained.csv
  figures: qc_mito_variance_explained, *_global_cutoff, *_within_type_depth_correlation,
           qc_mito_removed_vs_retained

Definitions
  depth: log10(protein IDs) (primary) and log10(total intensity).
  B cutoffs (high side): global median + 3 x MAD (scaled), global top 5%, global top 10%;
    compared with the within-label 3 x MAD high flags of step mito_qc.
  B permutation p: removal flags permuted 10,000 times (seed 42); statistic = Pearson chi-square.
  D: metrics over ALL protein rows of the raw CSV (KIDNEY 4,144; DCT 2,600) for all raw cells, so cells
     removed by the >= 100-ID filter and retained cells are measured on the same protein set.

Command (project root):  python scripts/qc_mito_mito_tests.py
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
import qc_mito_lib as t3

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "qc_mito_mito_tests.py"
DEPTH_VARS = ("log10_n_detected", "log10_total_intensity")
DEPTH_DISPLAY = {"log10_n_detected": "log10 protein IDs", "log10_total_intensity": "log10 total intensity"}
CUTOFF_METRICS = ("mitocarta_intensity_fraction", "mitocarta_id_fraction")
CUTOFFS = ("global_median_plus_3MAD", "global_top5pct", "global_top10pct", "within_label_3MAD_high")
CUTOFF_DISPLAY = {"global_median_plus_3MAD": "global median + 3 MAD", "global_top5pct": "global top 5%",
                  "global_top10pct": "global top 10%", "within_label_3MAD_high": "within-label 3 MAD (high)"}
N_PERM = 10_000
D_METRICS = ("mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
             "mt_encoded_intensity_fraction", "n_detected_mitocarta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qc_mito tests A-D on the mitochondrial QC metrics.")
    cc.add_common_args(parser)
    return parser.parse_args()


def read_per_cell(run: cc.Run, dataset: str) -> pd.DataFrame:
    path = cc.out_path(t3.TASK, "tables", f"{dataset}_qc_per_cell", ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run qc_mito_mito_qc.py first.")
    return pd.read_csv(run.use_input(path, f"qc_mito per-cell QC ({dataset})"), dtype={"cell_id": str}).set_index("cell_id")


def r2(y: np.ndarray, columns: dict[str, np.ndarray]) -> tuple[float, float]:
    import statsmodels.api as sm

    design = pd.DataFrame(columns) if columns else pd.DataFrame(index=range(len(y)))
    design.insert(0, "const", 1.0)
    fit = sm.OLS(y, design.astype(float)).fit()
    return float(fit.rsquared), float(fit.rsquared_adj)


def dummies(values: pd.Series, prefix: str) -> dict[str, np.ndarray]:
    cats = sorted(values.astype(str).unique())
    return {f"{prefix}_{c}": (values.astype(str) == c).to_numpy(float) for c in cats[1:]}


# ---------------------------------------------------------------------------
# A. Variance explained
# ---------------------------------------------------------------------------
def variance_explained(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for col in t3.DATASETS[dataset]["label_columns"]:
            lab = dummies(frame[col], "label")
            plate = dummies(frame["plate"], "plate") if dataset == "dct" else {}
            for metric in t3.MITO_METRICS:
                for scale in ("raw", "rank"):
                    y = frame[metric].to_numpy(float)
                    if scale == "rank":
                        y = stats.rankdata(y) / len(y)
                    for depth_var in DEPTH_VARS:
                        d = {"depth": frame[depth_var].to_numpy(float)}
                        r_lab, a_lab = r2(y, lab)
                        r_dep, a_dep = r2(y, d)
                        r_both, a_both = r2(y, {**lab, **d})
                        row = {"dataset": dataset, "label_column": col, "metric": metric, "scale": scale,
                               "depth_variable": depth_var, "n_cells": len(y), "n_labels": len(lab) + 1,
                               "r2_label": r_lab, "r2_depth": r_dep, "r2_label_plus_depth": r_both,
                               "adj_r2_label": a_lab, "adj_r2_depth": a_dep, "adj_r2_label_plus_depth": a_both,
                               "unique_r2_label": r_both - r_dep, "unique_r2_depth": r_both - r_lab,
                               "shared_r2": r_lab + r_dep - r_both,
                               "partial_r2_label_after_depth": (r_both - r_dep) / (1 - r_dep)}
                        if plate:
                            r_pl, _ = r2(y, plate)
                            r_all, a_all = r2(y, {**lab, **d, **plate})
                            row.update(r2_plate=r_pl, r2_label_plus_depth_plus_plate=r_all,
                                       adj_r2_label_plus_depth_plus_plate=a_all)
                        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. Global cutoff simulation
# ---------------------------------------------------------------------------
def chi2_stat(codes: np.ndarray, removed: np.ndarray, n_labels: int) -> float:
    n = len(codes)
    rem = np.bincount(codes, weights=removed.astype(float), minlength=n_labels)
    tot = np.bincount(codes, minlength=n_labels).astype(float)
    k = removed.sum()
    exp_rem = tot * k / n
    exp_keep = tot * (n - k) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(exp_rem > 0, (rem - exp_rem) ** 2 / exp_rem, 0) + \
            np.where(exp_keep > 0, ((tot - rem) - exp_keep) ** 2 / exp_keep, 0)
    return float(terms.sum())


def cutoff_masks(frame: pd.DataFrame, metric: str, col: str) -> dict[str, tuple[np.ndarray, float]]:
    x = frame[metric].to_numpy(float)
    med = np.nanmedian(x)
    mad = stats.median_abs_deviation(x[np.isfinite(x)], scale="normal")
    out = {"global_median_plus_3MAD": (x > med + t3.N_MAD * mad, med + t3.N_MAD * mad)}
    for q, name in ((0.95, "global_top5pct"), (0.90, "global_top10pct")):
        thr = float(np.nanquantile(x, q))
        out[name] = (x >= thr, thr)
    if col != "plate":
        out["within_label_3MAD_high"] = ((frame[f"outlier__{t3.SHORT_COLUMN[col]}__{metric}"] == "high").to_numpy(),
                                         np.nan)
    return out


def global_cutoffs(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tests = [], []
    rng = np.random.default_rng(t3.SEED)
    for dataset, frame in frames.items():
        for col in t3.DATASETS[dataset]["plot_columns"]:
            order = t3.label_order(frame[col], dataset)
            labels = frame[col].astype(str).to_numpy()
            codes = pd.Categorical(labels, categories=order).codes.astype(int)
            for metric in CUTOFF_METRICS:
                for cutoff, (removed, thr) in cutoff_masks(frame, metric, col).items():
                    n_rem = int(removed.sum())
                    for label in ["all_cells", *order]:
                        mask = np.ones(len(frame), bool) if label == "all_cells" else labels == label
                        k = int((removed & mask).sum())
                        rows.append({"dataset": dataset, "label_column": col, "metric": metric, "cutoff": cutoff,
                                     "threshold": thr, "label": label, "n_cells": int(mask.sum()), "n_removed": k,
                                     "percent_of_label_removed": 100 * k / mask.sum(),
                                     "share_of_all_cells": mask.mean(),
                                     "share_of_removed": k / n_rem if n_rem else np.nan,
                                     "ratio_share_removed_to_share_cells": (k / n_rem) / mask.mean() if n_rem else np.nan})
                    observed = chi2_stat(codes, removed, len(order)) if n_rem else np.nan
                    if n_rem:
                        null = np.array([chi2_stat(codes, rng.permutation(removed), len(order)) for _ in range(N_PERM)])
                        p_perm = (1 + (null >= observed - 1e-12).sum()) / (N_PERM + 1)
                        table = pd.crosstab(labels, removed).to_numpy()
                        p_asym = stats.chi2_contingency(table)[1] if table.shape[1] == 2 else np.nan
                    else:
                        p_perm = p_asym = np.nan
                    tests.append({"dataset": dataset, "label_column": col, "metric": metric, "cutoff": cutoff,
                                  "threshold": thr, "n_cells": len(frame), "n_removed": n_rem,
                                  "n_labels": len(order), "chi2": observed, "p_asymptotic": p_asym,
                                  "p_permutation": p_perm, "n_permutations": N_PERM if n_rem else 0})
    return pd.DataFrame(rows), pd.DataFrame(tests)


# ---------------------------------------------------------------------------
# C. Within-label association with depth
# ---------------------------------------------------------------------------
def within_type(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, frame in frames.items():
        for col in t3.DATASETS[dataset]["label_columns"]:
            order = t3.label_order(frame[col], dataset)
            counts = frame[col].astype(str).value_counts()
            for metric in t3.MITO_METRICS:
                for depth_var in DEPTH_VARS:
                    block = []
                    for label in ["all_cells", *order]:
                        mask = np.ones(len(frame), bool) if label == "all_cells" else (frame[col].astype(str) == label).to_numpy()
                        row = {"dataset": dataset, "label_column": col, "metric": metric, "depth_variable": depth_var,
                               "label": label, "n_cells": int(mask.sum())}
                        if label != "all_cells" and counts[label] < t3.MIN_GROUP:
                            row.update(n=int(mask.sum()), rho=np.nan, ci_low=np.nan, ci_high=np.nan, p_value=np.nan,
                                       note=f"not evaluated (n < {t3.MIN_GROUP})")
                        else:
                            row.update(t3.spearman_ci(frame.loc[mask, depth_var], frame.loc[mask, metric]), note="")
                        block.append(row)
                    block = pd.DataFrame(block)
                    types = (block["label"] != "all_cells") & block["p_value"].notna()
                    block["p_fdr_bh_within_labels"] = np.nan
                    if types.any():
                        block.loc[types, "p_fdr_bh_within_labels"] = stats.false_discovery_control(
                            block.loc[types, "p_value"].to_numpy(float), method="bh")
                    rows.append(block)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# D. Cells removed by the >= 100-ID filter
# ---------------------------------------------------------------------------
def removed_cells(run: cc.Run, mitocarta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    per_cell, rows, checks = [], [], []
    for dataset in ("kidney", "dct"):
        rb = cc.rebuild_ln_intensity(dataset, run)
        meta_all = rb.protein_meta_all.set_index("var_name")
        meta = pd.DataFrame({"pg_genes": meta_all["gene_symbol"], "protein_group": meta_all["protein_group"],
                             "first_symbol": meta_all.index.astype(str),
                             "protein_description": meta_all["protein_description"]}, index=meta_all.index)
        flags_all = t3.annotate_proteins(meta, mitocarta)
        retained_annotation = t3.read_annotation(run, dataset, rb.intensity.columns)
        kept = flags_all.loc[rb.keep_proteins]
        for name in t3.SETS:
            same = np.array_equal(kept[name].to_numpy(bool), retained_annotation[name].to_numpy(bool))
            checks.append({"dataset": dataset, "check": f"set '{name}' on retained rows equals step-1 annotation",
                           "pass": bool(same)})
        metrics = t3.mito_metrics(rb.raw_intensity.to_numpy(), flags_all, rb.raw_intensity.index)
        metrics.insert(0, "retained", rb.keep_cells)
        metrics.insert(0, "dataset", dataset)
        per_cell.append(metrics)
        ids = metrics["n_detected_proteins"]
        retained = metrics["retained"].to_numpy(bool)
        low_cut = float(np.quantile(ids[retained], 0.10))
        groups = {"retained_all": retained, f"retained_lowest_decile_ids (<= {low_cut:.0f} IDs)": retained & (ids <= low_cut).to_numpy()}
        removed = ~retained
        for metric in D_METRICS:
            x = metrics[metric]
            for ref_name, ref in groups.items():
                row = {"dataset": dataset, "metric": metric, "comparison": f"removed_vs_{ref_name}",
                       "n_removed_cells": int(removed.sum()),
                       "n_removed_with_0_ids": int((removed & (ids == 0).to_numpy()).sum()),
                       "removed_ids_min": int(ids[removed].min()), "removed_ids_max": int(ids[removed].max()),
                       "reference_ids_min": int(ids[ref].min()), "reference_ids_max": int(ids[ref].max())}
                row.update(cc.compare_groups(x[removed], x[ref]))
                rows.append(row)
        run.note(f"{dataset}: test D over {rb.raw_intensity.shape[1]} protein rows; {int(removed.sum())} removed cells "
                 f"({int((removed & (ids == 0).to_numpy()).sum())} with 0 IDs); MitoCarta rows "
                 f"{int(flags_all['mitocarta'].sum())}.")
    return pd.concat(per_cell), pd.DataFrame(rows), checks


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_variance(run, ve: pd.DataFrame) -> None:
    sub = ve[(ve["scale"] == "raw") & (ve["depth_variable"] == "log10_n_detected")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharex=True)
    bars = [("r2_depth", "depth only (log10 IDs)", t3.SERIES[0]), ("r2_label", "label only", t3.SERIES[1]),
            ("r2_label_plus_depth", "label + depth", t3.SERIES[2])]
    for ax, dataset in zip(axes, ("kidney", "dct")):
        d = sub[sub["dataset"] == dataset].reset_index(drop=True)
        ylabels = [f"{t3.METRIC_DISPLAY[m].replace(' fraction of', ',').replace('summed ', '')}\n"
                   f"{t3.LABEL_DISPLAY[c]}" for m, c in zip(d["metric"], d["label_column"])]
        y = np.arange(len(d))
        h = 0.26
        for j, (key, name, color) in enumerate(bars):
            ax.barh(y + (j - 1) * h, d[key], height=h * 0.92, color=color, label=name)
            for yi, v in zip(y + (j - 1) * h, d[key]):
                ax.text(v + 0.005, yi, f"{v:.2f}", va="center", fontsize=4.8, color=t3.INK["secondary"])
        ax.set_yticks(y, ylabels, fontsize=5.2)
        ax.invert_yaxis()
        ax.set_xlabel("R² (OLS, all cells)")
        ax.set_title(t3.DATASETS[dataset]["display"], fontsize=7)
        ax.xaxis.grid(True, color=t3.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[1].legend(loc="lower right", fontsize=5.6)
    fig.suptitle("Variance of mitochondrial fractions explained by cell label and by depth", x=0.01, ha="left",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t3.footer(fig, SCRIPT, "rank-scale and log10 total-intensity versions in qc_mito_variance_explained.csv")
    run.figure(fig, "variance_explained", "R2 of label and depth for mito fractions")


def fig_cutoffs(run, sim: pd.DataFrame, dataset: str) -> None:
    metric = "mitocarta_intensity_fraction"
    cols = t3.DATASETS[dataset]["label_columns"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(7.4, 2.3 * len(cols)), squeeze=False)
    for ax, col in zip(axes[:, 0], cols):
        d = sim[(sim["dataset"] == dataset) & (sim["label_column"] == col) & (sim["metric"] == metric)]
        order = [lab for lab in t3.label_order(d.loc[d["label"] != "all_cells", "label"], dataset)]
        x = np.arange(len(order))
        w = 0.2
        for j, cutoff in enumerate(CUTOFFS):
            dd = d[d["cutoff"] == cutoff].set_index("label")
            if dd.empty:
                continue
            vals = dd.loc[order, "percent_of_label_removed"].to_numpy(float)
            ax.bar(x + (j - 1.5) * w, vals, width=w * 0.92, color=t3.SERIES[j], label=CUTOFF_DISPLAY[cutoff])
            overall = float(dd.loc["all_cells", "percent_of_label_removed"])
            ax.axhline(overall, color=t3.SERIES[j], linewidth=0.6, linestyle=(0, (3, 2)))
        n = d[d["cutoff"] == CUTOFFS[0]].set_index("label")["n_cells"]
        ax.set_xticks(x, [f"{lab}\n{int(n[lab])}" for lab in order], fontsize=5.6)
        ax.set_ylabel("% of label removed")
        ax.set_title(f"{t3.LABEL_DISPLAY[col]} (dashed: % of all cells removed)", fontsize=6.6)
        t3.ygrid(ax)
    axes[0, 0].legend(loc="upper right", fontsize=5.6, ncol=2)
    fig.suptitle(f"{t3.DATASETS[dataset]['display']}: cells removed by cutoffs on the MitoCarta fraction of summed "
                 "intensity (high side)", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t3.footer(fig, SCRIPT, "simulation only; no cell was excluded; within-label rule evaluates labels with >= 10 cells")
    run.figure(fig, f"{dataset}_global_cutoff", f"{dataset}: global vs within-label mito cutoffs")


def fig_within(run, wt: pd.DataFrame, dataset: str) -> None:
    cols = t3.DATASETS[dataset]["label_columns"]
    fig, axes = plt.subplots(len(DEPTH_VARS), len(cols), figsize=(7.4, 5.0), squeeze=False, sharex=True)
    for i, depth_var in enumerate(DEPTH_VARS):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            d = wt[(wt["dataset"] == dataset) & (wt["label_column"] == col) & (wt["depth_variable"] == depth_var)]
            labels = [lab for lab in ["all_cells", *t3.label_order(d.loc[d["label"] != "all_cells", "label"], dataset)]
                      if d.loc[d["label"] == lab, "rho"].notna().any()]
            y = np.arange(len(labels))
            for k, metric in enumerate(t3.MITO_METRICS):
                dd = d[d["metric"] == metric].set_index("label").loc[labels]
                yy = y + (k - 1) * 0.22
                ax.errorbar(dd["rho"], yy, xerr=[dd["rho"] - dd["ci_low"], dd["ci_high"] - dd["rho"]], fmt="o",
                            markersize=2.8, color=t3.SERIES[k], elinewidth=0.7, capsize=0,
                            label=t3.METRIC_DISPLAY[metric])
            ax.axvline(0, color=t3.INK["muted"], linewidth=0.6)
            n = d[d["metric"] == t3.MITO_METRICS[0]].set_index("label")["n_cells"]
            ax.set_yticks(y, [f"{lab} ({int(n[lab])})" for lab in labels], fontsize=5.6)
            ax.invert_yaxis()
            ax.set_xlim(-1, 1)
            ax.xaxis.grid(True, color=t3.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{t3.LABEL_DISPLAY[col]} vs {DEPTH_DISPLAY[depth_var]}", fontsize=6.4)
            if i == len(DEPTH_VARS) - 1:
                ax.set_xlabel("Spearman rho (95% CI)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper right", ncol=3, fontsize=5.6, bbox_to_anchor=(0.995, 0.955))
    fig.suptitle(f"{t3.DATASETS[dataset]['display']}: mitochondrial fractions vs depth within each label "
                 f"(labels with >= {t3.MIN_GROUP} cells)", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    t3.footer(fig, SCRIPT, "Fisher-z interval, SE = 1.06/sqrt(n-3)")
    run.figure(fig, f"{dataset}_within_type_depth_correlation", f"{dataset}: within-label Spearman with depth")


def fig_removed(run, per_cell: pd.DataFrame) -> None:
    metric = "mitocarta_intensity_fraction"
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), gridspec_kw={"width_ratios": [1.6, 1]})
    for row, dataset in enumerate(("kidney", "dct")):
        d = per_cell[per_cell["dataset"] == dataset]
        ids = d["n_detected_proteins"].to_numpy(float)
        x = d[metric].to_numpy(float)
        ret = d["retained"].to_numpy(bool)
        ax = axes[row, 0]
        ok = ids > 0
        ax.scatter(ids[ok & ret], x[ok & ret], s=3, color=t3.INK["background_cells"], linewidths=0, rasterized=True,
                   label=f"retained ({int(ret.sum())})")
        ax.scatter(ids[ok & ~ret], x[ok & ~ret], s=10, color=t3.SERIES[1], linewidths=0.3, edgecolors="white",
                   label=f"removed, < 100 IDs ({int((~ret).sum())}; {int((~ret & ~ok).sum())} with 0 IDs not shown)")
        ax.axvline(100, color=t3.INK["muted"], linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlabel("Protein IDs per cell (all protein rows; log scale)")
        ax.set_ylabel(t3.METRIC_DISPLAY[metric], fontsize=6.2)
        ax.set_title(f"{t3.DATASETS[dataset]['display']}: all raw cells", fontsize=6.8)
        ax.set_ylim(-0.03, np.nanmax(x[ok]) * 1.3)
        ax.legend(loc="upper right", fontsize=5.4, markerscale=1.6)
        t3.ygrid(ax)
        ax = axes[row, 1]
        low_cut = np.quantile(ids[ret], 0.10)
        groups = [("removed", ~ret & ok, t3.SERIES[1]), ("retained,\nlowest 10% IDs", ret & (ids <= low_cut), t3.SERIES[0]),
                  ("retained,\nall", ret, t3.INK["secondary"])]
        # The two retained groups overlap; each is drawn from its own mask.
        rng = np.random.default_rng(t3.SEED)
        for i, (name, mask, color) in enumerate(groups):
            vals = x[mask & np.isfinite(x)]
            bp = ax.boxplot(vals, positions=[i], widths=0.55, showfliers=False, patch_artist=True)
            bp["boxes"][0].set_facecolor("none")
            bp["boxes"][0].set_edgecolor(t3.INK["secondary"])
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set_color(t3.INK["secondary"])
            bp["medians"][0].set_color(t3.INK["primary"])
            ax.scatter(i + rng.uniform(-0.2, 0.2, vals.size), vals, s=3, color=color, alpha=0.6, linewidths=0,
                       rasterized=True)
        ax.set_xticks(range(len(groups)), [f"{g[0]}\n(n={int((g[1] & np.isfinite(x)).sum())})" for g in groups],
                      fontsize=5.4)
        ax.set_title(f"lowest 10% = <= {low_cut:.0f} IDs", fontsize=6.4)
        t3.ygrid(ax)
    fig.suptitle("Cells removed by the >= 100-ID filter: MitoCarta fraction of summed intensity", x=0.01, ha="left",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t3.footer(fig, SCRIPT, "metrics over all protein rows of the raw CSV; removed cells have few IDs, so fractions are noisy")
    run.figure(fig, "removed_vs_retained", "Removed vs retained cells, MitoCarta fraction")


def main() -> None:
    args = parse_args()
    run = cc.Run(t3.TASK, "mito_tests", args)
    frames = {ds: read_per_cell(run, ds) for ds in ("kidney", "dct")}
    mitocarta = t3.read_mitocarta(run)

    ve = variance_explained(frames)
    run.table(ve, "variance_explained")
    sim, tests = global_cutoffs(frames)
    run.table(sim, "global_cutoff_simulation")
    run.table(tests, "global_cutoff_tests")
    wt = within_type(frames)
    run.table(wt, "within_type_depth_correlation")
    per_cell, removed, checks = removed_cells(run, mitocarta)
    run.table(per_cell.reset_index(), "all_raw_cells_mito")
    run.table(removed, "removed_vs_retained")
    checks = pd.DataFrame(checks)
    run.table(checks, "mito_tests_checks")

    fig_variance(run, ve)
    for dataset in ("kidney", "dct"):
        fig_cutoffs(run, sim, dataset)
        fig_within(run, wt, dataset)
    fig_removed(run, per_cell)
    ok = bool(checks["pass"].all())
    run.finish({"depth_variables": list(DEPTH_VARS), "cutoffs": list(CUTOFFS), "n_permutations": N_PERM,
                "min_group": t3.MIN_GROUP, "n_mad": t3.N_MAD, "checks_pass": ok},
               status="completed" if ok else "completed_with_failed_checks")


if __name__ == "__main__":
    main()
