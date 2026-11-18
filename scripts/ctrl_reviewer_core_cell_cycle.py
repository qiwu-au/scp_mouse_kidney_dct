#!/usr/bin/env python
"""ctrl_reviewer_core step 2: reviewer_core A.2 Task 4 cell-cycle correction on the control data (labels_core labels).

Port of
reviewer_core_cell_cycle.py (unchanged method; inputs, label columns, the stored
clusters and the DPT root rule replaced as below).

Inputs (read-only): results/h5ad/ctrl_reviewer_core_DCT_ctrl_task3_scored.h5ad (step 1);
labels_core canonical labels; results/tables/labels_core/labels_core_submitted_dpt_roots.csv (root check).
Outputs:
  results/tables/ctrl_reviewer_core/ctrl_reviewer_core_task4_*.csv
  results/figures/ctrl_reviewer_core/ctrl_reviewer_core_task4_*.pdf/.png
  results/h5ad/ctrl_reviewer_core_DCT_ctrl_<branch>.h5ad for the six corrected branches (no intensity layers)

ctrl_reviewer_core changes: label columns ctrl_deg_label and ctrl_cluster_label; the stored labels_core
clusters (ctrl_leiden3_cluster, Leiden 3.0; ctrl_deg_cluster, Leiden 1.0) are
added as two extra 'uncorrected' clusterings; DPT root = first ctrl_cell_label
DCT1 cell (the labels_core root rule), random roots drawn as in reviewer_core/labels_core and checked
against the labels_core roots.

(a) Original 18-protein panel: remove the detected proteins, or regress
    cell_cycle_score; rescale, PCA, neighbours (10, 20 PCs), Leiden 3.0, UMAP.
(b) Tirosh S/G2M: removal of the union with the original panel; regression of
    S_score + G2M_score. Circular set: proteins Spearman-correlated with Mki67
    or Top2a (rho >= 0.3, BH q < 0.05) on the scaled matrix.
(c) Leiden at 0.3, 0.5, 0.8, 1.0 and 3.0 on every graph (uncorrected = stored
    labels_core graph); ARI/NMI against both label columns.
(d) kNN label purity (k = 10, 20 PCs) against 1,000 label permutations.
(e) ProLIF versus detected proteins per cell and plate.
(f) DPT from the original root (first per-cell DCT1 cell) and 20 random DCT1 roots.

Command (project root):  python scripts/ctrl_reviewer_core_cell_cycle.py
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
import scanpy as sc
import statsmodels.api as sm
from scipy import stats
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import ctrl_reviewer_core_lib as lib

ensure_scanpy_compat()
t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "ctrl_reviewer_core_cell_cycle.py"
OBJECT_INFO = {"uncorrected": ("none", "stored labels_core graph"), **t2.CC_BRANCHES}
ROOT_COLUMN = "ctrl_cell_label"   # labels_core DPT root rule: first per-cell DCT1 cell (scp_dct_upload.py lines 272-278)
STORED = {f"stored_{c}": c for c, _ in lib.STORED_CLUSTERS.values()}
STORED_TICK = {"stored_ctrl_leiden3_cluster": "stored 3.0 (labels_core)", "stored_ctrl_deg_cluster": "stored 1.0 (labels_core)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 2: Task 4 cell-cycle correction on the control data.")
    parser.add_argument("--resolutions", type=float, nargs="+", default=list(t2.RESOLUTIONS))
    parser.add_argument("--corr-min-rho", type=float, default=0.30)
    parser.add_argument("--corr-max-fdr", type=float, default=0.05)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-random-roots", type=int, default=20)
    cc.add_common_args(parser)
    return parser.parse_args()


def spearman_against(matrix: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spearman rho and two-sided p (t approximation, as scipy.stats.spearmanr) of each column with anchor."""
    ranks = stats.rankdata(matrix, axis=0)
    ranks -= ranks.mean(axis=0)
    a = stats.rankdata(anchor)
    a -= a.mean()
    denom = np.sqrt((ranks ** 2).sum(axis=0) * (a ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (ranks * a[:, None]).sum(axis=0) / denom
        t = rho * np.sqrt((len(a) - 2) / np.clip(1 - rho ** 2, 1e-300, None))
    return rho, 2 * stats.t.sf(np.abs(t), len(a) - 2)


def correlated_proteins(adata, n_detected: np.ndarray, min_rho: float, max_fdr: float) -> tuple[pd.DataFrame, list[str]]:
    X = cc.dense(adata.X).astype(float)
    names = list(adata.var_names)
    table = pd.DataFrame({"protein": names})
    selected = np.zeros(len(names), dtype=bool)
    anchors = [a for a in ("Mki67", "Top2a") if a in names]
    for anchor in anchors:
        rho, p = spearman_against(X, X[:, names.index(anchor)])
        p[names.index(anchor)] = np.nan
        q = np.full(len(p), np.nan)
        ok = np.isfinite(p)
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
        table[f"rho_{anchor}"], table[f"p_{anchor}"], table[f"q_{anchor}"] = rho, p, q
        selected |= (rho >= min_rho) & (q < max_fdr)
    for anchor in anchors:
        selected[names.index(anchor)] = True
    table["rho_n_detected_proteins"] = spearman_against(X, n_detected.astype(float))[0]
    table["in_original_18_panel"] = table["protein"].isin(t2.CELL_CYCLE_MARKERS)
    table["in_tirosh_S_G2M"] = table["protein"].isin(t2.TIROSH_ALL)
    table["selected"] = selected
    return table, [n for n, s in zip(names, selected) if s]


def removed_branch(adata, genes: list[str], key: str):
    keep = [g for g in adata.var_names if g not in set(genes)]
    return t2.recluster(adata[:, keep].copy(), key, t2.ORIGINAL_RESOLUTION)


def regressed_branch(adata, keys: list[str], key: str):
    out = adata.copy()
    if all(np.nanstd(out.obs[k].astype(float)) > 0 for k in keys):
        sc.pp.regress_out(out, keys=list(keys), n_jobs=1)
    return t2.recluster(out, key, t2.ORIGINAL_RESOLUTION)


def confounding(labels: pd.DataFrame, scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ndet = labels["n_detected_proteins"].to_numpy(float)
    rows, logit_rows = [], []
    for col in lib.LABEL_COLUMNS:
        is_p = (labels[col] == "ProLIF").to_numpy()
        row = {"label_column": col, "n_prolif": int(is_p.sum()), "n_other": int((~is_p).sum())}
        row.update({f"ids_{k}": v for k, v in cc.compare_groups(ndet[is_p], ndet[~is_p]).items()})
        table = pd.crosstab(pd.Series(is_p, name="ProLIF"), labels["plate"].to_numpy()).reindex(
            index=[False, True], columns=["P1", "P2"], fill_value=0)
        odds, p = stats.fisher_exact(table.to_numpy())
        row.update({"P1_n_prolif": int(table.loc[True, "P1"]), "P1_n_cells": int(table["P1"].sum()),
                    "P2_n_prolif": int(table.loc[True, "P2"]), "P2_n_cells": int(table["P2"].sum()),
                    "P1_fraction_prolif": table.loc[True, "P1"] / table["P1"].sum(),
                    "P2_fraction_prolif": table.loc[True, "P2"] / table["P2"].sum(),
                    "fisher_odds_ratio_P2_vs_P1": odds, "fisher_p": p})
        rows.append(row)
        design = pd.DataFrame({"z_n_detected_proteins": (ndet - ndet.mean()) / ndet.std(ddof=1),
                               "plate_P2": (labels["plate"] == "P2").astype(float).to_numpy()})
        try:
            fit = sm.Logit(is_p.astype(float), sm.add_constant(design)).fit(disp=0)
            ci = fit.conf_int()
            for term in fit.params.index:
                logit_rows.append({"label_column": col, "term": term, "coef": fit.params[term], "se": fit.bse[term],
                                   "odds_ratio": np.exp(fit.params[term]), "or_ci_low": np.exp(ci.loc[term, 0]),
                                   "or_ci_high": np.exp(ci.loc[term, 1]), "p_value": fit.pvalues[term], "n": len(is_p)})
        except Exception as exc:  # report instead of failing (e.g. separation)
            logit_rows.append({"label_column": col, "term": "fit_failed", "coef": np.nan, "note": str(exc)})
    corr_rows = []
    for score_col in scores.columns:
        rho, p = stats.spearmanr(scores[score_col].to_numpy(float), ndet, nan_policy="omit")
        corr_rows.append({"score": score_col, "vs": "n_detected_proteins", "spearman_rho": rho, "p_value": p,
                          "n": len(ndet)})
    return pd.DataFrame(rows), pd.DataFrame(logit_rows), pd.DataFrame(corr_rows)


def run_dpt(objects: dict, labels: pd.DataFrame, n_random: int):
    dct1 = np.flatnonzero(labels[ROOT_COLUMN].to_numpy() == "DCT1")
    original = int(dct1[0])
    rng = np.random.default_rng(t2.SEED)
    random_roots = rng.choice(dct1[1:], size=min(n_random, len(dct1) - 1), replace=False)
    roots = [("original", original)] + [(f"random_{i + 1:02d}", int(r)) for i, r in enumerate(random_roots)]
    rows, per_cell = [], pd.DataFrame(index=labels.index)
    for name, obj in objects.items():
        sc.tl.diffmap(obj)
        for root_name, root in roots:
            obj.uns["iroot"] = root
            sc.tl.dpt(obj)
            pt = obj.obs["dpt_pseudotime"].to_numpy(float).copy()
            if root_name == "original":
                per_cell[f"{name}__dpt_original_root"] = pt
            finite = np.isfinite(pt)
            for col in lib.LABEL_COLUMNS:
                lab = labels[col].to_numpy()
                med = {l: float(np.median(pt[finite & (lab == l)])) if (finite & (lab == l)).any() else np.nan
                       for l in t2.CORE_LABELS}
                lo, hi = sorted((med["DCT1"], med["DCT2"]))
                rows.append({"object": name, "root": root_name, "root_index": root, "root_cell_id": labels.index[root],
                             "label_column": col, "median_DCT1": med["DCT1"], "median_DCT2": med["DCT2"],
                             "median_ProLIF": med["ProLIF"], "prolif_between_dct1_dct2": bool(lo < med["ProLIF"] < hi),
                             "n_infinite": int((~finite).sum())})
        obj.uns["iroot"] = original
        sc.tl.dpt(obj)
    detail = pd.DataFrame(rows)
    summary = []
    for (name, col), sub in detail.groupby(["object", "label_column"], sort=False):
        orig = sub[sub["root"] == "original"].iloc[0]
        rnd = sub[sub["root"] != "original"]
        summary.append({"object": name, "label_column": col, "original_root_cell_id": orig["root_cell_id"],
                        "original_median_DCT1": orig["median_DCT1"], "original_median_DCT2": orig["median_DCT2"],
                        "original_median_ProLIF": orig["median_ProLIF"],
                        "original_prolif_between": orig["prolif_between_dct1_dct2"],
                        "n_random_roots": len(rnd), "n_random_prolif_between": int(rnd["prolif_between_dct1_dct2"].sum()),
                        "fraction_random_prolif_between": rnd["prolif_between_dct1_dct2"].mean()})
    return detail, pd.DataFrame(summary), per_cell


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_object_umaps(name: str, obj, labels: pd.DataFrame, score_col: str, run: cc.Run) -> None:
    umap = obj.obsm["X_umap"]
    panels = [*((c, "cat") for c in lib.LABEL_COLUMNS), ("leiden_r3.00", "cluster"),
              (score_col, "num"), ("dpt_pseudotime", "num")]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.15 * len(panels), 2.45))
    for ax, (col, kind) in zip(axes, panels):
        if kind == "cat":
            values = labels[col].to_numpy()
            for cat in cc.ordered_labels(values):
                mask = values == cat
                ax.scatter(umap[mask, 0], umap[mask, 1], s=2.5, color=t2.LABEL_PALETTE.get(cat, "0.5"), linewidths=0,
                           label=f"{cat} ({mask.sum()})", rasterized=True)
            ax.legend(fontsize=5, markerscale=2.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
                      handletextpad=0.1, columnspacing=0.6)
            ax.set_title(lib.LABEL_DISPLAY[col], fontsize=6.8)
        elif kind == "cluster":
            codes = pd.Categorical(obj.obs[col].astype(str)).codes
            ax.scatter(umap[:, 0], umap[:, 1], c=codes, cmap="tab20", s=2.5, linewidths=0, rasterized=True)
            ax.set_title(f"Leiden 3.0 ({obj.obs[col].nunique()} clusters)", fontsize=6.8)
        else:
            values = obj.obs[col].to_numpy(float) if col in obj.obs else np.full(obj.n_obs, np.nan)
            sca = ax.scatter(umap[:, 0], umap[:, 1], c=values, cmap="viridis", s=2.5, linewidths=0, rasterized=True)
            cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=5)
            ax.set_title(col if col != "dpt_pseudotime" else "DPT (original root)", fontsize=6.8)
        t2.clean_axes(ax)
    info = OBJECT_INFO[name]
    fig.suptitle(f"{name}: {info[0]}, {info[1]}", x=0.01, ha="left", fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"task4_umap_{name}", f"UMAP of the {name} object")


def plot_heatmaps(ari: pd.DataFrame, metric: str, run: cc.Run) -> None:
    fig, axes = plt.subplots(1, len(lib.LABEL_COLUMNS), figsize=(3.4 * len(lib.LABEL_COLUMNS), 2.9))
    for ax, col in zip(axes, lib.LABEL_COLUMNS):
        sub = ari[ari["label_column"] == col].pivot(index="object", columns="resolution", values=metric)
        sub = sub.reindex(index=[o for o in OBJECT_INFO if o in sub.index])
        vmax = max(0.5, float(np.nanmax(sub.to_numpy())))
        im = ax.imshow(sub.to_numpy(float), cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", t2.BLUE_RAMP),
                       vmin=0, vmax=vmax, aspect="auto")
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                v = sub.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 0.6 * vmax else t2.INK["primary"])
        ax.set_xticks(range(sub.shape[1]), [STORED_TICK.get(c, c) for c in sub.columns],
                      rotation=30, ha="right")
        ax.set_yticks(range(sub.shape[0]), sub.index)
        ax.set_title(f"{metric.upper()} vs {lib.LABEL_DISPLAY[col]}", fontsize=7)
        ax.set_xlabel("Leiden resolution")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"task4_{metric}_clusters_vs_labels", f"{metric.upper()} of clusters versus labels_core labels")


def plot_concentration(conc: pd.DataFrame, run: cc.Run) -> None:
    sub = conc[np.isclose(conc["resolution_numeric"], t2.ORIGINAL_RESOLUTION) & conc["object"].isin(OBJECT_INFO)]
    objects = [o for o in OBJECT_INFO if o in set(sub["object"])]
    fig, axes = plt.subplots(1, len(lib.LABEL_COLUMNS), figsize=(3.4 * len(lib.LABEL_COLUMNS), 2.6), sharey=True)
    for ax, col in zip(axes, lib.LABEL_COLUMNS):
        part = sub[sub["label_column"] == col].set_index("object").reindex(objects)
        y = np.arange(len(objects))
        ax.barh(y - 0.18, part["top1_percent"], height=0.34, color=t2.SERIES[0], label="largest cluster")
        ax.barh(y + 0.18, part["top3_percent"], height=0.34, color=t2.SERIES[1], label="top 3 clusters")
        for yy, (a, b) in enumerate(zip(part["top1_percent"], part["top3_percent"])):
            ax.text(a + 0.8, yy - 0.18, f"{a:.1f}", va="center", fontsize=5.2, color=t2.INK["secondary"])
            ax.text(b + 0.8, yy + 0.18, f"{b:.1f}", va="center", fontsize=5.2, color=t2.INK["secondary"])
        ax.set_yticks(y, objects)
        ax.invert_yaxis()
        n = int(part["n_prolif"].iloc[0])
        ax.set_title(f"{lib.LABEL_DISPLAY[col]} (ProLIF n = {n})", fontsize=7)
        ax.set_xlabel("% of ProLIF cells in largest / top-3 Leiden 3.0 clusters")
        ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "task4_prolif_concentration_r3.00", "ProLIF concentration in Leiden 3.0 clusters")


def plot_purity(table: pd.DataFrame, run: cc.Run, name: str, objects: list[str], title: str) -> None:
    fig, axes = plt.subplots(1, len(lib.LABEL_COLUMNS), figsize=(3.6 * len(lib.LABEL_COLUMNS), 2.8), sharey=True)
    for ax, col in zip(axes, lib.LABEL_COLUMNS):
        sub = table[(table["label_column"] == col) & table["label"].isin(t2.CORE_LABELS)]
        for offset, label in zip((-0.22, 0, 0.22), t2.CORE_LABELS):
            part = sub[sub["label"] == label].set_index("object").reindex(objects)
            x = np.arange(len(objects)) + offset
            ax.vlines(x, part["null_q025"], part["null_q975"], color=t2.INK["background_cells"], linewidth=3.2)
            ax.scatter(x, part["observed_same_label_fraction"], color=t2.LABEL_PALETTE[label], s=14, zorder=3,
                       label=label, edgecolors="white", linewidths=0.4)
        ax.set_xticks(range(len(objects)), objects, rotation=35, ha="right")
        ax.set_title(f"kNN label purity, {lib.LABEL_DISPLAY[col]}", fontsize=7)
        ax.set_ylabel("Mean same-label fraction of 10 neighbours")
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(title="dot = observed; grey bar = permutation 95%", title_fontsize=5.5, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t2.footer(fig, SCRIPT)
    run.figure(fig, name, title)


def plot_dpt(detail: pd.DataFrame, label_col: str, run: cc.Run) -> None:
    sub = detail[detail["label_column"] == label_col]
    objects = [o for o in OBJECT_INFO if o in set(sub["object"])]
    fig, axes = plt.subplots(1, len(objects), figsize=(1.35 * len(objects) + 0.4, 2.6), sharey=True)
    order = ["DCT1", "ProLIF", "DCT2"]
    rng = np.random.default_rng(0)  # jitter only
    for ax, name in zip(axes, objects):
        part = sub[sub["object"] == name]
        rnd = part[part["root"] != "original"]
        orig = part[part["root"] == "original"].iloc[0]
        for i, label in enumerate(order):
            vals = rnd[f"median_{label}"].to_numpy(float)
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=5, color=t2.LABEL_PALETTE[label], alpha=0.7,
                       linewidths=0)
            ax.scatter([i], [orig[f"median_{label}"]], marker="D", color=t2.INK["primary"], s=12, zorder=4)
        n_between = int(rnd["prolif_between_dct1_dct2"].sum())
        ax.set_xticks(range(3), order, rotation=30)
        ax.set_title(f"{name}\nbetween: {n_between}/{len(rnd)}", fontsize=6)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Median DPT per label")
    fig.suptitle(f"DPT medians by {lib.LABEL_DISPLAY[label_col]} (diamond = original root; dots = 20 random DCT1 roots; "
                 "'between' = ProLIF median between DCT1 and DCT2)", x=0.01, ha="left", fontsize=7, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"task4_dpt_medians_by_{label_col}", "DPT medians per label and root")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "cell_cycle", args)
    lib.ported_from(run, "reviewer_core_cell_cycle.py")
    adata = lib.read_h5ad(run, "DCT_ctrl_task3_scored")
    _, labels = lib.load_ctrl(run)
    if list(adata.obs_names) != list(labels.index):
        raise ValueError("Cell order of the A.1 object differs from the canonical labels.")
    stored_dpt = adata.obs["dpt_pseudotime"].to_numpy(float).copy()
    n_detected = labels["n_detected_proteins"].to_numpy()

    # (a) Original 18-protein panel (cell_cycle_score was computed in A.1 with the same call).
    present_orig = [g for g in t2.CELL_CYCLE_MARKERS if g in adata.var_names]
    missing_orig = [g for g in t2.CELL_CYCLE_MARKERS if g not in adata.var_names]
    objects: dict[str, object] = {"uncorrected": adata}
    r3_keys: dict[str, str] = {}
    objects["cc_removed"] = removed_branch(adata, present_orig, "leiden_cc_removed_res_3.00")
    objects["cc_regressed"] = regressed_branch(adata, ["cell_cycle_score"], "leiden_cc_regressed_res_3.00")
    r3_keys.update(cc_removed="leiden_cc_removed_res_3.00", cc_regressed="leiden_cc_regressed_res_3.00")

    # (b) Tirosh/Seurat S and G2M sets.
    s_rows, s_present = t2.match_genes(t2.TIROSH_S, adata.var_names, "S")
    g_rows, g_present = t2.match_genes(t2.TIROSH_G2M, adata.var_names, "G2M")
    o_rows, _ = t2.match_genes(t2.CELL_CYCLE_MARKERS, adata.var_names, "original_18_panel")
    run.table(pd.DataFrame(s_rows + g_rows + o_rows), "task4_cell_cycle_gene_coverage")
    order = list(adata.var_names)
    expanded = sorted(set(present_orig) | set(s_present) | set(g_present), key=order.index)
    if not (s_present and g_present):
        raise RuntimeError("Tirosh S or G2M set has no detected genes; cannot run the expanded branches.")
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g_present, random_state=t2.SEED, use_raw=False)
    objects["tirosh_removed"] = removed_branch(adata, expanded, "leiden_tirosh_removed_res_3.00")
    objects["tirosh_regressed"] = regressed_branch(adata, ["S_score", "G2M_score"], "leiden_tirosh_regressed_res_3.00")
    r3_keys.update(tirosh_removed="leiden_tirosh_removed_res_3.00", tirosh_regressed="leiden_tirosh_regressed_res_3.00")

    corr_table, corr_selected = correlated_proteins(adata, n_detected, args.corr_min_rho, args.corr_max_fdr)
    run.table(corr_table.sort_values("selected", ascending=False, kind="stable"), "task4_mki67_top2a_correlated_proteins")
    run.note(f"Circular set: Spearman rho >= {args.corr_min_rho} and BH q < {args.corr_max_fdr} with Mki67 or Top2a "
             f"on scaled X ({adata.n_obs} cells); {len(corr_selected)} proteins incl. anchors.")
    t2.score(adata, corr_selected, "cc_corr_score")
    objects["corr_removed"] = removed_branch(adata, corr_selected, "leiden_corr_removed_res_3.00")
    objects["corr_regressed"] = regressed_branch(adata, ["cc_corr_score"], "leiden_corr_regressed_res_3.00")
    r3_keys.update(corr_removed="leiden_corr_removed_res_3.00", corr_regressed="leiden_corr_regressed_res_3.00")

    sets = pd.DataFrame([
        {"object": name, "gene_set": OBJECT_INFO[name][0], "correction": OBJECT_INFO[name][1],
         "n_proteins_after": obj.n_vars,
         "proteins_removed": ";".join(sorted(set(adata.var_names) - set(obj.var_names), key=order.index)),
         "regressed_keys": {"cc_regressed": "cell_cycle_score", "tirosh_regressed": "S_score;G2M_score",
                            "corr_regressed": "cc_corr_score"}.get(name, "")}
        for name, obj in objects.items()])
    run.table(sets, "task4_correction_definitions")
    score_cols = [c for c in ("cell_cycle_score", "S_score", "G2M_score", "cc_corr_score") if c in adata.obs]
    scores = adata.obs[score_cols].set_axis(labels.index, axis=0)

    # (c) Leiden at several resolutions on every graph; ARI/NMI against the labels.
    resolutions = sorted(set(args.resolutions) | {t2.ORIGINAL_RESOLUTION})
    clusters = pd.DataFrame(index=labels.index)
    for name, obj in objects.items():
        if list(obj.obs_names) != list(adata.obs_names):
            raise ValueError(f"Cell order differs in {name}")
        for res in resolutions:
            key = f"leiden_r{res:.2f}"
            if name in r3_keys and np.isclose(res, t2.ORIGINAL_RESOLUTION):
                obj.obs[key] = obj.obs[r3_keys[name]].astype(str).astype("category")
            else:
                t2.leiden(obj, res, key)
            clusters[f"{name}__r{res:.2f}"] = obj.obs[key].astype(str).to_numpy()
    for key, col in STORED.items():
        clusters[f"uncorrected__{key}"] = labels[col].astype(str).to_numpy()
    checks = []
    for col, res in lib.STORED_CLUSTERS.values():
        same = int((clusters[f"uncorrected__r{res:.2f}"].to_numpy() == labels[col].astype(str).to_numpy()).sum())
        checks.append({"check": f"Leiden {res} (seed 42) on the stored labels_core graph equals {col}",
                       "value": f"{same}/{len(labels)}", "pass": same == len(labels)})
    run.table(clusters.reset_index(), "task4_clusters_per_cell")

    ari = t2.ari_nmi_rows(clusters, labels, lib.LABEL_COLUMNS)
    conc = t2.concentration_rows(clusters, labels, lib.LABEL_COLUMNS)
    comp_rows = []
    for column in clusters.columns:
        name, res = column.split("__")
        for col in lib.LABEL_COLUMNS:
            table = pd.crosstab(clusters[column], labels[col])
            for cluster, counts in table.iterrows():
                for label, n in counts.items():
                    comp_rows.append({"object": name, "resolution": res.removeprefix("r"), "cluster": cluster,
                                      "label_column": col, "label": label, "n_cells": int(n),
                                      "fraction_of_cluster": n / counts.sum()})
    run.table(ari, "task4_ari_nmi_clusters_vs_labels")
    run.table(conc, "task4_prolif_concentration")
    run.table(pd.DataFrame(comp_rows), "task4_cluster_composition_long")

    task3_cols = [c for c in ("ambiguous_tal_cnt_interstitial", "DCT_core_score", "DCT1_associated_score",
                              "DCT2_associated_score", "TAL_score") if c in adata.obs]
    context = adata.obs[task3_cols].set_axis(labels.index, axis=0).join(scores)
    summary_rows = []
    for name in objects:
        column = f"{name}__r{t2.ORIGINAL_RESOLUTION:.2f}"
        for cluster, idx in labels.groupby(clusters[column]).groups.items():
            row = {"object": name, "resolution": f"{t2.ORIGINAL_RESOLUTION:.2f}", "cluster": cluster, "n_cells": len(idx)}
            for col in lib.LABEL_COLUMNS:
                row[f"n_prolif_{col}"] = int((labels.loc[idx, col] == "ProLIF").sum())
                row[f"fraction_prolif_{col}"] = row[f"n_prolif_{col}"] / len(idx)
            for c in context.columns:
                vals = context.loc[idx, c]
                row[f"fraction_{c}" if vals.dtype == bool else f"{c}_median"] = vals.mean() if vals.dtype == bool else vals.median()
            summary_rows.append(row)
    run.table(pd.DataFrame(summary_rows), f"task4_cluster_summary_r{t2.ORIGINAL_RESOLUTION:.2f}")

    # (d) kNN label purity.
    purity = t2.knn_purity({name: obj.obsm["X_pca"][:, :t2.N_PCS] for name, obj in objects.items()},
                           labels, lib.LABEL_COLUMNS, args.knn_k, args.n_permutations)
    run.table(purity, "task4_knn_label_purity")

    # (e) Confounding with detected proteins and plate.
    conf, logit, corr = confounding(labels, scores)
    run.table(conf, "task4_confounding_prolif_vs_ids_and_plate")
    run.table(logit, "task4_confounding_logistic_prolif")
    run.table(corr, "task4_confounding_scores_vs_ids")

    # (f) Diffusion pseudotime on each graph.
    dpt_detail, dpt_summary, dpt_cells = run_dpt(objects, labels, args.n_random_roots)
    dpt_diff = float(np.nanmax(np.abs(dpt_cells["uncorrected__dpt_original_root"].to_numpy() - stored_dpt)))
    run.note(f"Uncorrected DPT, original root: max |recomputed - stored labels_core dpt_pseudotime| = {dpt_diff:.3g}.")
    checks.append({"check": "uncorrected DPT from the original root reproduces the stored labels_core dpt_pseudotime",
                   "value": f"max abs diff {dpt_diff:.3g}", "pass": dpt_diff < 1e-5})
    t18_roots = pd.read_csv(run.use_input(cc.out_path(lib.LABELS_CORE_TASK, "tables", "submitted_dpt_roots", ".csv"),
                                          "labels_core DPT roots (check)"))
    ours = dpt_detail.loc[(dpt_detail["object"] == "uncorrected") & (dpt_detail["label_column"] == lib.PRIMARY),
                          "root_index"].astype(int).tolist()
    same_roots = ours == t18_roots["root_index"].astype(int).tolist()
    checks.append({"check": "original and 20 random roots equal the labels_core roots (labels_core_submitted_dpt_roots.csv)",
                   "value": f"{same_roots}", "pass": same_roots})
    run.table(pd.DataFrame(checks), "task4_checks")
    if not all(c["pass"] for c in checks):
        run.finish(status="stopped_failed_checks")
        raise SystemExit("A check failed; see ctrl_reviewer_core_task4_checks.csv.")
    run.table(dpt_detail, "task4_dpt_medians_per_root")
    run.table(dpt_summary, "task4_dpt_summary")
    run.table(dpt_cells.reset_index(), "task4_dpt_per_cell_original_root")
    run.table(labels[["plate", *lib.LABEL_COLUMNS, "n_detected_proteins"]].join(scores).reset_index(),
              "task4_cell_cycle_scores_per_cell")

    for name, obj in objects.items():
        score_col = {"tirosh_removed": "G2M_score", "tirosh_regressed": "G2M_score", "corr_removed": "cc_corr_score",
                     "corr_regressed": "cc_corr_score"}.get(name, "cell_cycle_score")
        if score_col not in obj.obs:
            obj.obs[score_col] = scores[score_col].to_numpy()
        plot_object_umaps(name, obj, labels, score_col, run)
    plot_heatmaps(ari, "ari", run)
    plot_heatmaps(ari, "nmi", run)
    plot_concentration(conc, run)
    plot_purity(purity, run, "task4_knn_label_purity", list(OBJECT_INFO),
                "kNN label purity (k = 10, 20 PCs) versus 1,000 label permutations")
    for col in lib.LABEL_COLUMNS:
        plot_dpt(dpt_detail, col, run)

    for name, obj in objects.items():
        if name == "uncorrected":
            continue
        for col in ("plate", *lib.LABEL_COLUMNS):
            obj.obs[col] = pd.Categorical(labels[col].to_numpy())
        for col in score_cols:
            obj.obs[col] = scores[col].to_numpy()
        t2.write_h5ad_light(run, obj, f"DCT_ctrl_{name}")
    run.finish({"cell_cycle_original_present": present_orig, "cell_cycle_original_missing": missing_orig,
                "tirosh_S_present": s_present, "tirosh_G2M_present": g_present,
                "expanded_removed_set": expanded, "correlated_set": corr_selected,
                "dpt_max_abs_diff_vs_stored": dpt_diff})


if __name__ == "__main__":
    main()
