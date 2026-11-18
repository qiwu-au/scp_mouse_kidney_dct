#!/usr/bin/env python
"""ctrl_reviewer_core step 12b: preprocessing steps 4-5 on the control DCT data - heatmaps, UMAPs and the variant ranking.

Port of
preprocessing_summary_figures.py, DCT only. ctrl_reviewer_core changes (agreed in the ctrl_reviewer_core plan, authors 28 Oct 2026):
label columns ctrl_deg_label and ctrl_cluster_label; macro averages over DCT1, DCT2
and ProLIF only (ctrl_reviewer_core_lib.macro_core; Unknown reported per label); the UMAP
alternative is chosen on ctrl_deg_label (the primary column) and the both-column
ranking is kept in the table; the third UMAP row none__z10__hvg2000 is kept.
The preprocessing text below describes the original; outputs carry the ctrl_reviewer_core_ prefix.

Reads the per-dataset tables of preprocessing_sensitivity.py and writes:
  - the variant grid (definitions of the 33 pipelines);
  - per dataset, one summary row per pipeline x label column and a ranking:
      label preservation = mean of the ranks of best-resolution ARI, best-resolution NMI,
        macro kNN-purity enrichment and macro silhouette (macro = labels with >= 10 cells);
      depth independence = rank of max(|rho PC1|, |rho PC2|) with ln(protein IDs);
      combined = mean of the two ranks.
    KIDNEY: per label column; the UMAP alternative is chosen on kidney_reannot_label (primary).
    DCT: per label column and for both V1 columns together (mean of the two label-preservation
    ranks); the UMAP alternative is chosen on the latter.
    The alternative is the best combined rank among the non-baseline pipelines (ties: depth rank,
    then grid order). The ranking orders observed values; it is not a recommendation.
    DCT also gets a third UMAP row, none__z10__hvg2000 (named by the authors, 13 Oct 2026; EXTRA_UMAP).
  - heatmaps (clustering agreement, kNN purity, silhouette, PC-depth rho), a rank scatter, an
    HVG detection-rate plot, and UMAPs of the baseline and the chosen alternative.

Inputs (read-only): results/tables/preprocessing/preprocessing_<dataset>_*.csv; kidney_object canonical objects
  (label colours) and labels.
Outputs: results/tables/preprocessing/preprocessing_variant_grid.csv, preprocessing_<dataset>_variant_summary.csv,
  preprocessing_<dataset>_ranking.csv; results/figures/preprocessing/preprocessing_<dataset>_*.pdf/.png

Command (project root):  python scripts/preprocessing_summary_figures.py
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

import scp_common as cc
import qc_mito_lib as t3
import preprocessing_lib as t4
import ctrl_reviewer_core_lib as lib

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SCRIPT = "ctrl_reviewer_core_preprocessing_figures.py"
BOTH = "both_label_columns"
SELECTION_SCOPE = {"dct": lib.PRIMARY}
# Extra UMAP row requested by the authors (13 Oct 2026): a pipeline that changes the preprocessing, because the DCT
# alternative chosen by the rule (k = 15) shares the baseline PCA. Named by the authors, not chosen by the rule.
EXTRA_UMAP = {"dct": "none__z10__hvg2000"}
NORM_DISPLAY = {"none": "none", "median": "median", "total": "total"}
DIVERGING = LinearSegmentedColormap.from_list("t04_div", ["#e34948", "#f0efec", "#2a78d6"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 12b: preprocessing summary, ranking and figures (control DCT).")
    cc.add_common_args(parser)
    return parser.parse_args()


def read(run: cc.Run, ds: str, name: str, **kwargs) -> pd.DataFrame:
    path = cc.out_path(lib.TASK, "tables", f"{ds}_{name}", ".csv")
    return pd.read_csv(run.use_input(path, f"preprocessing {ds} {name}"), **kwargs)


def variant_label(v: t4.Variant) -> str:
    text = f"{v.normalization} · {v.scaling} · {v.features}"
    if v.block == "pca_test":
        text += f" · {v.n_pcs} PCs"
    elif v.block == "knn_test":
        text += f" · k={v.k}"
    return text + ("  (baseline)" if v.is_baseline else "")


# ---------------------------------------------------------------------------
# Summary and ranking
# ---------------------------------------------------------------------------
def summarise(ds: str, clus: pd.DataFrame, purity: pd.DataFrame, sil: pd.DataFrame, depth: pd.DataFrame,
              feat: pd.DataFrame) -> pd.DataFrame:
    grid = {v.variant_id: v for v in t4.variant_grid()}
    n_prot = feat.set_index(["normalization", "features"])["n_proteins"]
    rows = []
    for vid, v in grid.items():
        d = depth[depth["variant_id"] == vid].set_index("pc")
        r1, r2 = float(d.at[1, "spearman_rho_ln_n_detected"]), float(d.at[2, "spearman_rho_ln_n_detected"])
        for col in lib.PREPROCESSING_DATASETS[ds]["label_columns"]:
            c = clus[(clus["variant_id"] == vid) & (clus["label_column"] == col)].set_index("resolution")
            p = purity[(purity["variant_id"] == vid) & (purity["label_column"] == col)]
            s = sil[(sil["variant_id"] == vid) & (sil["label_column"] == col)]
            row = {"dataset": ds, "label_column": col, **v.as_row(),
                   "n_proteins": int(n_prot.loc[(v.normalization, v.features)]),
                   "best_ari": float(c["ari"].max()), "best_ari_resolution": float(c["ari"].idxmax()),
                   "best_nmi": float(c["nmi"].max()), "best_nmi_resolution": float(c["nmi"].idxmax()),
                   "best_majority_accuracy": float(c["majority_accuracy"].max()),
                   "macro_knn_purity": lib.macro_core(p, "observed_same_label_fraction"),
                   "macro_knn_purity_enrichment": lib.macro_core(p, "enrichment"),
                   "all_cells_knn_purity": float(p.loc[p["label"] == "all_cells", "observed_same_label_fraction"].iloc[0]),
                   "macro_silhouette": lib.macro_core(s, "mean_silhouette"),
                   "all_cells_silhouette": float(s.loc[s["label"] == "all_cells", "mean_silhouette"].iloc[0]),
                   "rho_pc1_ln_n_detected": r1, "rho_pc2_ln_n_detected": r2,
                   "max_abs_rho_pc12": max(abs(r1), abs(r2)),
                   "pc1_variance_ratio": float(d.at[1, "variance_ratio"]),
                   "pc2_variance_ratio": float(d.at[2, "variance_ratio"])}
            for res in t4.RESOLUTIONS:
                row[f"n_clusters_r{res:.1f}"] = int(c.at[res, "n_clusters"])
                row[f"ari_r{res:.1f}"] = float(c.at[res, "ari"])
                row[f"nmi_r{res:.1f}"] = float(c.at[res, "nmi"])
                row[f"ari_vs_stored_leiden_r{res:.1f}"] = float(c.at[res, "ari_vs_stored_leiden"])
            rows.append(row)
    return pd.DataFrame(rows)


LP_METRICS = ("best_ari", "best_nmi", "macro_knn_purity_enrichment", "macro_silhouette")


def rank_scope(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    for m in LP_METRICS:
        out[f"rank_{m}"] = out[m].rank(ascending=False, method="average")
    out["label_preservation_mean_rank"] = out[[f"rank_{m}" for m in LP_METRICS]].mean(axis=1)
    out["label_preservation_rank"] = out["label_preservation_mean_rank"].rank(method="min")
    out["depth_independence_rank"] = out["max_abs_rho_pc12"].rank(method="min")
    out["combined_score"] = (out["label_preservation_rank"] + out["depth_independence_rank"]) / 2
    out["combined_rank"] = out["combined_score"].rank(method="min")
    out.insert(1, "scope", scope)
    return out


def ranking(ds: str, summary: pd.DataFrame) -> pd.DataFrame:
    order = [v.variant_id for v in t4.variant_grid()]
    parts = [rank_scope(summary[summary["label_column"] == col], col) for col in lib.PREPROCESSING_DATASETS[ds]["label_columns"]]
    if ds == "dct":
        a, b = parts
        both = a[["dataset", *[c for c in a.columns if c in t4.Variant.__dataclass_fields__ or c in
                                ("variant_id", "block", "n_pcs", "k_neighbors", "is_baseline", "embedding_key",
                                 "n_proteins", "max_abs_rho_pc12", "rho_pc1_ln_n_detected", "rho_pc2_ln_n_detected")]]].copy()
        both = both.loc[:, ~both.columns.duplicated()]
        lp = (a.set_index("variant_id")["label_preservation_rank"] + b.set_index("variant_id")["label_preservation_rank"]) / 2
        both["label_preservation_mean_rank"] = both["variant_id"].map(lp).to_numpy()
        both["label_preservation_rank"] = both["label_preservation_mean_rank"].rank(method="min")
        both["depth_independence_rank"] = both["max_abs_rho_pc12"].rank(method="min")
        both["combined_score"] = (both["label_preservation_rank"] + both["depth_independence_rank"]) / 2
        both["combined_rank"] = both["combined_score"].rank(method="min")
        both.insert(1, "scope", BOTH)
        parts.append(both)
    out = pd.concat(parts, ignore_index=True)
    out["grid_order"] = out["variant_id"].map({v: i for i, v in enumerate(order)})
    sel = out[(out["scope"] == SELECTION_SCOPE[ds]) & ~out["is_baseline"].astype(bool)]
    chosen = sel.sort_values(["combined_score", "depth_independence_rank", "grid_order"]).iloc[0]["variant_id"]
    out["selected_for_umap"] = (out["variant_id"] == chosen) & (out["scope"] == SELECTION_SCOPE[ds])
    out["extra_umap_row_user"] = (out["variant_id"] == EXTRA_UMAP.get(ds)) & (out["scope"] == SELECTION_SCOPE[ds])
    return out.sort_values(["scope", "combined_rank", "grid_order"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def heat(ax, values: np.ndarray, cmap, vmin: float, vmax: float, fmt: str = "{:.2f}", norm=None,
         fontsize: float = 4.2) -> None:
    norm = norm or plt.Normalize(vmin, vmax)
    ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    for (i, j), val in np.ndenumerate(values):
        if np.isfinite(val):
            shade = norm(val)
            light = (abs(shade - 0.5) > 0.32) if norm.__class__ is TwoSlopeNorm else shade > 0.62
            ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=fontsize,
                    color="white" if light else t3.INK["primary"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="y", length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def row_labels(ax, grid: list[t4.Variant], highlight: str | None = None) -> None:
    ax.set_yticks(range(len(grid)), [variant_label(v) for v in grid], fontsize=4.8)
    for tick, v in zip(ax.get_yticklabels(), grid):
        if v.is_baseline or v.variant_id == highlight:
            tick.set_fontweight("bold")
        tick.set_color(t3.INK["primary"] if v.is_baseline or v.variant_id == highlight else t3.INK["secondary"])
    ax.tick_params(axis="y", length=0)


def block_lines(ax, grid: list[t4.Variant], ncols: int) -> None:
    for i in range(1, len(grid)):
        if grid[i].block != grid[i - 1].block or (grid[i].block == "core_factorial"
                                                    and grid[i].normalization != grid[i - 1].normalization):
            ax.axhline(i - 0.5, color="white", linewidth=1.6)


def fig_clustering(run, ds: str, clus: pd.DataFrame, chosen: str) -> None:
    grid = t4.variant_grid()
    cols = lib.PREPROCESSING_DATASETS[ds]["label_columns"]
    panels = [("n_clusters", None)] + [(m, c) for c in cols for m in ("ari", "nmi", "majority_accuracy")]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.0 * len(panels) + 1.9, 5.6), sharey=True)
    for ax, (metric, col) in zip(axes, panels):
        sub = clus[clus["label_column"] == (col or cols[0])]
        mat = sub.pivot(index="variant_id", columns="resolution", values=metric).reindex([v.variant_id for v in grid])
        values = mat.to_numpy(float)
        if metric == "n_clusters":
            heat(ax, values, t3.sequential_cmap(), 0, np.nanmax(values), fmt="{:.0f}")
            ax.set_title("clusters (n)", fontsize=6)
        else:
            heat(ax, values, t3.sequential_cmap(), 0, 1)
            name = {"ari": "ARI", "nmi": "NMI", "majority_accuracy": "majority acc."}[metric]
            ax.set_title(f"{name}\n{lib.LABEL_DISPLAY[col]}", fontsize=5.4)
        block_lines(ax, grid, values.shape[1])
        ax.set_xticks(range(len(t4.RESOLUTIONS)), [f"{r:g}" for r in t4.RESOLUTIONS], fontsize=5)
        ax.set_xlabel("Leiden res.", fontsize=5.5)
    row_labels(axes[0], grid, chosen)
    fig.suptitle(f"{lib.PREPROCESSING_DATASETS[ds]['display']}: clustering agreement with the submitted labels across 33 pipelines",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    t3.footer(fig, SCRIPT, "bold rows: baseline (submitted pipeline) and the UMAP alternative; colour scale 0-1 "
                           "(clusters: 0-max)")
    run.figure(fig, f"{ds}_heatmap_clustering", f"{ds}: ARI, NMI, majority accuracy and cluster counts")


def label_matrix(frame: pd.DataFrame, col: str, value: str, labels_order: list[str]) -> np.ndarray:
    sub = frame[frame["label_column"] == col]
    return sub.pivot(index="variant_id", columns="label", values=value).reindex(
        index=[v.variant_id for v in t4.variant_grid()], columns=labels_order).to_numpy(float)


def label_header(ds: str, frame: pd.DataFrame, col: str) -> list[str]:
    sub = frame[(frame["label_column"] == col) & (frame["variant_id"] == t4.BASELINE_ID)]
    counts = dict(zip(sub["label"], sub["n_cells"]))
    order = [lab for lab in cc.ordered_labels(list(counts), lib.PREPROCESSING_DATASETS[ds]["order"]) if lab != "all_cells"]
    return [*order, "all_cells"], counts


def fig_label_metric(run, ds: str, frame: pd.DataFrame, value: str, name: str, title: str, chosen: str,
                     diverging: bool = False) -> None:
    grid = t4.variant_grid()
    cols = lib.PREPROCESSING_DATASETS[ds]["label_columns"]
    headers = [label_header(ds, frame, c) for c in cols]
    widths = [len(h[0]) for h in headers]
    fig, axes = plt.subplots(1, 2, figsize=(0.32 * sum(widths) + 2.4, 5.8), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    for ax, col, (order, counts) in zip(axes, cols, headers):
        values = label_matrix(frame, col, value, order)
        if diverging:
            lim = max(0.05, float(np.nanmax(np.abs(values))))
            heat(ax, values, DIVERGING, -lim, lim, norm=TwoSlopeNorm(0, -lim, lim))
        else:
            heat(ax, values, t3.sequential_cmap(), 0, 1)
        block_lines(ax, grid, values.shape[1])
        ax.set_xticks(range(len(order)), [f"{lab}\n({counts[lab]})" for lab in order], fontsize=4.6, rotation=90)
        ax.set_title(lib.LABEL_DISPLAY[col], fontsize=6)
    row_labels(axes[0], grid, chosen)
    fig.suptitle(f"{lib.PREPROCESSING_DATASETS[ds]['display']}: {title}", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    note = ("diverging scale centred at 0" if diverging else "colour scale 0-1") + \
        "; labels with < 10 cells are shown but left out of the macro averages"
    t3.footer(fig, SCRIPT, note)
    run.figure(fig, f"{ds}_heatmap_{name}", f"{ds}: {title}")


def fig_depth(run, ds: str, depth: pd.DataFrame, chosen: str) -> None:
    grid = t4.variant_grid()
    ids = [v.variant_id for v in grid]
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 5.6), sharey=True, gridspec_kw={"width_ratios": [5, 5, 2.4]})
    for ax, col, title in ((axes[0], "spearman_rho_ln_n_detected", "rho with ln(protein IDs)"),
                           (axes[1], "spearman_rho_ln_total_intensity", "rho with ln(total intensity)")):
        values = depth.pivot(index="variant_id", columns="pc", values=col).reindex(ids).to_numpy(float)
        heat(ax, values, DIVERGING, -1, 1, norm=TwoSlopeNorm(0, -1, 1))
        block_lines(ax, grid, values.shape[1])
        ax.set_xticks(range(values.shape[1]), [f"PC{i + 1}" for i in range(values.shape[1])], fontsize=5)
        ax.set_title(title, fontsize=6)
    var = depth.pivot(index="variant_id", columns="pc", values="variance_ratio").reindex(ids)[[1, 2]].to_numpy(float)
    heat(axes[2], var, t3.sequential_cmap(), 0, max(0.05, float(np.nanmax(var))), fmt="{:.2f}")
    axes[2].set_xticks([0, 1], ["PC1", "PC2"], fontsize=5)
    axes[2].set_title("variance\nratio", fontsize=6)
    block_lines(axes[2], grid, 2)
    row_labels(axes[0], grid, chosen)
    fig.suptitle(f"{lib.PREPROCESSING_DATASETS[ds]['display']}: Spearman rho of PC scores with depth", x=0.01, ha="left",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    t3.footer(fig, SCRIPT, "PC signs are arbitrary; the ranking uses max(|rho PC1|, |rho PC2|) with ln(protein IDs)")
    run.figure(fig, f"{ds}_heatmap_pc_depth", f"{ds}: PC-depth Spearman rho")


def fig_rank_scatter(run, ds: str, rank: pd.DataFrame) -> None:
    scopes = list(lib.PREPROCESSING_DATASETS[ds]["label_columns"]) + ([BOTH] if ds == "dct" else [])
    from matplotlib.lines import Line2D

    chosen = rank.loc[rank["selected_for_umap"].astype(bool), "variant_id"].iloc[0]
    fig, axes = plt.subplots(1, len(scopes), figsize=(2.6 * len(scopes), 2.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, scope in zip(axes, scopes):
        sub = rank[rank["scope"] == scope]
        ax.scatter(sub["max_abs_rho_pc12"], sub["label_preservation_mean_rank"], s=12, color=t3.SERIES[0],
                   alpha=0.75, linewidths=0)
        marks = [(chosen, "s", 34), (t4.BASELINE_ID, "o", 26)]
        if ds in EXTRA_UMAP:
            marks.append((EXTRA_UMAP[ds], "D", 26))
        for vid, marker, size in marks:
            r = sub[sub["variant_id"] == vid].iloc[0]
            ax.scatter([r["max_abs_rho_pc12"]], [r["label_preservation_mean_rank"]], s=size, marker=marker,
                       facecolors="none", edgecolors=t3.INK["primary"], linewidths=0.7, zorder=4)
        ax.invert_yaxis()
        ax.set_xlabel("max |rho| PC1/PC2 vs ln(protein IDs)")
        ax.set_ylabel("label-preservation mean rank\n(1 = best)")
        ax.set_title(lib.LABEL_DISPLAY.get(scope, "both label columns"), fontsize=6.2)
        t3.ygrid(ax)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4, markerfacecolor=t3.SERIES[0],
                      markeredgewidth=0, label="pipeline"),
               Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor="none",
                      markeredgecolor=t3.INK["primary"], label="baseline (submitted pipeline)"),
               Line2D([0], [0], marker="s", linestyle="", markersize=5, markerfacecolor="none",
                      markeredgecolor=t3.INK["primary"], label=f"UMAP alternative: {chosen.replace('__', ' · ')}")]
    if ds in EXTRA_UMAP:
        handles.append(Line2D([0], [0], marker="D", linestyle="", markersize=4.5, markerfacecolor="none",
                              markeredgecolor=t3.INK["primary"],
                              label=f"third UMAP row (authors): {EXTRA_UMAP[ds].replace('__', ' · ')}"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 0.045), fontsize=6)
    fig.suptitle(f"{lib.PREPROCESSING_DATASETS[ds]['display']}: label preservation against depth association (33 pipelines)",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.92))
    t3.footer(fig, SCRIPT, "names and ranks of all pipelines: preprocessing_" + ds + "_ranking.csv")
    run.figure(fig, f"{ds}_rank_scatter", f"{ds}: label preservation vs depth association")


def fig_feature_detection(run, ds: str, feat_sets: pd.DataFrame) -> None:
    det = feat_sets["detection_rate"].to_numpy(float)
    groups = [("all proteins", np.ones(len(det), bool))]
    for norm in ("none", "median", "total"):
        for f in ("hvg500", "hvg1000", "hvg2000", "highcov50"):
            col = f"{norm}__{f}"
            if col in feat_sets:
                groups.append((f"{norm} · {f}", feat_sets[col].astype(bool).to_numpy()))
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for i, (name, mask) in enumerate(groups):
        vals = det[mask]
        bp = ax.boxplot(vals, positions=[i], widths=0.6, showfliers=False, patch_artist=True)
        bp["boxes"][0].set_facecolor("none")
        for part in ("boxes", "whiskers", "caps"):
            for line in bp[part]:
                line.set_color(t3.INK["secondary"])
                line.set_linewidth(0.6)
        bp["medians"][0].set_color(t3.SERIES[0])
        bp["medians"][0].set_linewidth(1.2)
    ax.set_xticks(range(len(groups)), [f"{n}\n({int(m.sum())})" for n, m in groups], fontsize=4.8, rotation=60,
                  ha="right")
    ax.set_ylabel("fraction of cells detecting the protein")
    ax.set_ylim(0, 1.02)
    t3.ygrid(ax)
    fig.suptitle(f"{lib.PREPROCESSING_DATASETS[ds]['display']}: detection rate of the selected proteins per feature set",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    t3.footer(fig, SCRIPT, f"HVG flavor '{t4.HVG_FLAVOR}' on each normalized log matrix (missing = 0)")
    run.figure(fig, f"{ds}_feature_detection", f"{ds}: detection rate of HVG / high-coverage feature sets")


def fig_umaps(run, ds: str, adata, labels: pd.DataFrame, coords: pd.DataFrame, chosen: str) -> None:
    cols = lib.PREPROCESSING_DATASETS[ds]["label_columns"]
    lx = np.log(labels["n_detected_proteins"].to_numpy(float))
    variants = {v.variant_id: v for v in t4.variant_grid()}
    rows = [(t4.BASELINE_ID, "baseline (submitted pipeline)"), (chosen, "alternative: " + variant_label(variants[chosen]))]
    if ds in EXTRA_UMAP:
        rows.append((EXTRA_UMAP[ds], "author-selected: " + variant_label(variants[EXTRA_UMAP[ds]])))
    last = len(rows) - 1
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.4, 2.5 * len(rows) + 0.4))
    for r, (vid, row_title) in enumerate(rows):
        xy = coords[[f"{vid}__umap1", f"{vid}__umap2"]].to_numpy(float)
        for c, col in enumerate(cols):
            values = labels[col].astype(str).to_numpy()
            order = cc.ordered_labels(values, lib.PREPROCESSING_DATASETS[ds]["order"])
            colors = lib.label_colors(col)
            t3.umap_categorical(axes[r, c], xy, values, order, colors, lib.LABEL_DISPLAY[col])
            if r == last:
                t3.legend_with_counts(axes[r, c], values, order, colors, loc="upper center",
                                      bbox_to_anchor=(0.5, -0.04), ncol=3 if len(order) > 6 else 2,
                                      handletextpad=0.2, columnspacing=0.6)
        t3.umap_continuous(fig, axes[r, 2], xy, lx, "ln(protein IDs)")
        axes[r, 0].set_ylabel(row_title, fontsize=6.2)
        axes[r, 0].yaxis.set_label_coords(-0.04, 0.5)
    title = "the top-ranked alternative" + (" and a author-selected pipeline" if ds in EXTRA_UMAP else " pipeline")
    fig.suptitle(f"{lib.PREPROCESSING_DATASETS[ds]['display']}: UMAP of the baseline, {title}", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    t3.footer(fig, SCRIPT, "all UMAPs recomputed here with the same settings (seed 42); colours = submitted labels")
    run.figure(fig, f"{ds}_umap_baseline_vs_alternative", f"{ds}: UMAP of baseline and alternative")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "preprocessing_figures", args)
    lib.ported_from(run, "preprocessing_summary_figures.py")
    run.table(t4.grid_table(), "variant_grid")
    for ds in lib.QC_MITO_RUN:
        clus = read(run, ds, "clustering_metrics")
        purity = read(run, ds, "knn_purity")
        sil = read(run, ds, "silhouette")
        depth = read(run, ds, "pc_depth")
        feat = read(run, ds, "feature_set_summary")
        feat_sets = read(run, ds, "feature_sets")
        coords = read(run, ds, "umap_coords", dtype={"cell_id": str}).set_index("cell_id")
        adata, labels = lib.load_ctrl(run)
        if list(coords.index) != list(labels.index):
            raise ValueError(f"{ds}: UMAP coordinate order differs from the canonical labels.")
        summary = summarise(ds, clus, purity, sil, depth, feat)
        rank = ranking(ds, summary)
        chosen = rank.loc[rank["selected_for_umap"], "variant_id"].iloc[0]
        run.note(f"{ds}: UMAP alternative = {chosen} (scope {SELECTION_SCOPE[ds]}).")
        run.table(summary, f"{ds}_variant_summary")
        run.table(rank, f"{ds}_ranking")
        fig_clustering(run, ds, clus, chosen)
        fig_label_metric(run, ds, purity, "observed_same_label_fraction", "knn_purity",
                         "kNN label purity (k = 10, same-label fraction of neighbours) per submitted label", chosen)
        fig_label_metric(run, ds, sil, "mean_silhouette", "silhouette",
                         "mean silhouette per submitted label (pipeline PC space)", chosen, diverging=True)
        fig_depth(run, ds, depth, chosen)
        fig_rank_scatter(run, ds, rank)
        fig_feature_detection(run, ds, feat_sets)
        fig_umaps(run, ds, adata, labels, coords, chosen)
    run.finish({"label_preservation_metrics": list(LP_METRICS), "selection_scope": SELECTION_SCOPE,
                "extra_umap_row_user": EXTRA_UMAP,
                "min_label_n_for_macro": t4.MIN_LABEL_N, "macro_labels": list(lib.CORE_LABELS)})


if __name__ == "__main__":
    main()
