#!/usr/bin/env python
"""dct_reviewer step 12b: preprocessing steps 4-5 per DCT condition (combined search) - variant summary, ranking, heatmaps,
rank scatter, feature detection and UMAPs.

Port of ctrl_reviewer_core_preprocessing_figures.py. summarise and
ranking are called unchanged under dct_reviewer_lib.bound (dct_reviewer_lib.part2_settings: preprocessing_DATASETS, macro_core),
with the module's SELECTION_SCOPE set to each condition's primary column while it runs. The figures are redrawn in
the dotplot format (Arial >= 7 pt, 180 mm wide; ctrl_reviewer_core drew one dataset per figure at 4-6 pt), Ctrl and ZeroK side by side.

Ranking (as ctrl_reviewer_core/preprocessing): label preservation = mean of the ranks of best-resolution ARI, best-resolution NMI, macro
kNN-purity enrichment and macro silhouette (macro over DCT1, DCT2 and ProLIF with >= 10 cells; Unknown left out);
depth independence = rank of max(|rho PC1|, |rho PC2|) with ln(protein IDs); combined = mean of the two ranks. Scopes:
each label column, and both columns together (mean of the two label-preservation ranks). The UMAP alternative is the
best combined rank on the primary column among the non-baseline pipelines (ties: depth rank, then grid order). The
ranking orders observed values; it is not a recommendation. Third UMAP row: none__z10__hvg2000 (named by the authors,
13 Oct 2026; kept as in ctrl_reviewer_core).

Inputs (read-only): dct_reviewer_prep_*.csv (step 12a); dct_labels canonical labels.
Outputs (results/tables/dct_reviewer/, results/figures/dct_reviewer/; every table has `condition`):
  dct_reviewer_prep_variant_grid.csv, dct_reviewer_prep_variant_summary.csv, dct_reviewer_prep_ranking.csv
  figures: dct_reviewer_prep_heatmap_clustering_{primary,second}, dct_reviewer_prep_heatmap_knn_purity,
           dct_reviewer_prep_heatmap_silhouette, dct_reviewer_prep_heatmap_pc_depth, dct_reviewer_prep_rank_scatter,
           dct_reviewer_prep_feature_detection, dct_reviewer_prep_umap_baseline_vs_alternative

Command (project root):  python scripts/dct_reviewer_preprocessing_figures.py
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
import reviewer_core_lib as t2
import qc_mito_lib as t3
import preprocessing_lib as t4
import ctrl_reviewer_core_preprocessing_figures as p19f
import dct_reviewer_lib as lib

lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

SCRIPT = "dct_reviewer_preprocessing_figures.py"
BOTH = p19f.BOTH
EXTRA_UMAP = p19f.EXTRA_UMAP["dct"]
DIVERGING = p19f.DIVERGING
SCOPE_DISPLAY = {"primary": "primary label", "second": "second label", BOTH: "both label columns"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 12b: preprocessing summary, ranking and figures.")
    cc.add_common_args(parser)
    return parser.parse_args()


def read(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = lib.table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run dct_reviewer_preprocessing.py first.")
    return pd.read_csv(run.use_input(path, f"dct_reviewer step 12a {name}"), **kwargs)


def short_label(v: t4.Variant) -> str:
    text = f"{v.normalization}/{v.scaling}/{v.features}"
    if v.block == "pca_test":
        text += f" {v.n_pcs} PCs"
    elif v.block == "knn_test":
        text += f" k={v.k}"
    return text + (" (baseline)" if v.is_baseline else "")


def scope_role(cond: str, scope: str) -> str:
    return BOTH if scope == BOTH else lib.SETTINGS[cond].role(scope)


# ---------------------------------------------------------------------------
# Summary and ranking (ctrl_reviewer_core functions per condition)
# ---------------------------------------------------------------------------
def summary_and_rank(cond: str, tabs: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    s = lib.part2_settings(cond)
    t = {k: v[v["condition"] == cond].drop(columns="condition") for k, v in tabs.items()}
    old = p19f.SELECTION_SCOPE
    try:
        p19f.SELECTION_SCOPE = {"dct": s.PRIMARY}
        with lib.bound(s, p19f):
            summary = p19f.summarise("dct", t["clus"], t["purity"], t["sil"], t["depth"], t["feat"])
            rank = p19f.ranking("dct", summary)
    finally:
        p19f.SELECTION_SCOPE = old
    rank.insert(int(rank.columns.get_loc("scope")) + 1, "scope_role", [scope_role(cond, x) for x in rank["scope"]])
    chosen = rank.loc[rank["selected_for_umap"].astype(bool), "variant_id"].iloc[0]
    return summary, rank, chosen


# ---------------------------------------------------------------------------
# Figure helpers (dotplot format)
# ---------------------------------------------------------------------------
def compact(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if abs(abs(v) - 1) < 0.005:
        return "1" if v > 0 else "-1"
    return f"{v:.2f}".replace("0.", ".", 1)


def heat(ax, values: np.ndarray, cmap, vmin: float, vmax: float, integer: bool = False, norm=None) -> None:
    norm = norm or plt.Normalize(vmin, vmax)
    ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    for (i, j), val in np.ndenumerate(values):
        if np.isfinite(val):
            shade = norm(val)
            light = (abs(shade - 0.5) > 0.32) if isinstance(norm, TwoSlopeNorm) else shade > 0.62
            ax.text(j, i, f"{val:.0f}" if integer else compact(val), ha="center", va="center", fontsize=7,
                    color="white" if light else t2.INK["primary"])
    ax.set_yticks([])
    ax.tick_params(axis="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def row_labels(ax, grid: list[t4.Variant]) -> None:
    ax.set_yticks(range(len(grid)), [short_label(v) for v in grid])
    for tick, v in zip(ax.get_yticklabels(), grid):
        tick.set_fontweight("bold" if v.is_baseline else "normal")
        tick.set_color(t2.INK["primary"] if v.is_baseline else t2.INK["secondary"])
    ax.tick_params(axis="y", length=0)


def mark_row(ax, row: int, ncols: int) -> None:
    ax.add_patch(Rectangle((-0.5, row - 0.5), ncols, 1, fill=False, edgecolor=t2.INK["primary"], linewidth=0.9,
                           clip_on=False, zorder=5))


def group_header(fig, axes: list, text: str, y: float) -> None:
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    fig.text((x0 + x1) / 2, y, text, ha="center", va="bottom", fontsize=7, fontweight="bold")


def label_order_counts(frame: pd.DataFrame, cond: str, col: str) -> tuple[list[str], dict]:
    sub = frame[(frame["condition"] == cond) & (frame["label_column"] == col) & (frame["variant_id"] == t4.BASELINE_ID)]
    counts = dict(zip(sub["label"], sub["n_cells"]))
    order = [lab for lab in cc.ordered_labels(list(counts), lib.LABEL_ORDER) if lab != "all_cells"]
    return [*order, "all_cells"], counts


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_clustering(clus: pd.DataFrame, chosen: dict, role: str, figs: lib.Figures) -> None:
    grid = t4.variant_grid()
    ids = [v.variant_id for v in grid]
    metrics = [("n_clusters", "clusters (n)"), ("ari", "ARI"), ("nmi", "NMI"), ("majority_accuracy", "maj. acc.")]
    nmax = float(clus["n_clusters"].max())
    fig = lib.new_figure(lib.FULL_MM, 158)
    gs = fig.add_gridspec(1, 8, left=0.205, right=0.99, top=0.875, bottom=0.075, wspace=0.12)
    for ci, cond in enumerate(lib.CONDITIONS):
        col = lib.SETTINGS[cond].COLUMN_OF[role]
        sub = clus[(clus["condition"] == cond) & (clus["label_column"] == col)]
        axes = []
        for k, (metric, name) in enumerate(metrics):
            ax = fig.add_subplot(gs[0, 4 * ci + k])
            mat = sub.pivot(index="variant_id", columns="resolution", values=metric).reindex(ids).to_numpy(float)
            if metric == "n_clusters":
                heat(ax, mat, t3.sequential_cmap(), 0, nmax, integer=True)
            else:
                heat(ax, mat, t3.sequential_cmap(), 0, 1)
            p19f.block_lines(ax, grid, mat.shape[1])
            mark_row(ax, ids.index(chosen[cond]), mat.shape[1])
            ax.set_xticks(range(len(t4.RESOLUTIONS)), [f"{r:g}" for r in t4.RESOLUTIONS])
            ax.set_title(name, fontsize=7)
            if ci == 0 and k == 0:
                row_labels(ax, grid)
            axes.append(ax)
        group_header(fig, axes, f"{cond}: {lib.SETTINGS[cond].COLUMN_OF[role]}", 0.91)
        fig.text((axes[0].get_position().x0 + axes[-1].get_position().x1) / 2, 0.035, "Leiden resolution",
                 ha="center", va="top", fontsize=7)
    fig.text(0.005, 0.995, f"Clustering agreement with the {lib.ROLE_DISPLAY[role]} across 33 pipelines, per condition"
             "\n(colour scale 0-1, clusters 0-max; bold row: baseline = submitted pipeline; boxed row: the condition's "
             "UMAP alternative)", ha="left", va="top", fontsize=7)
    figs.save(fig, f"prep_heatmap_clustering_{role}", f"ARI, NMI, majority accuracy and cluster counts ({role} label)",
              label_role=role)


def fig_label_metric(frame: pd.DataFrame, value: str, name: str, title: str, chosen: dict, figs: lib.Figures,
                     diverging: bool = False) -> None:
    grid = t4.variant_grid()
    ids = [v.variant_id for v in grid]
    panels = [(cond, role) for cond in lib.CONDITIONS for role in lib.ROLES]
    heads = {p: label_order_counts(frame, p[0], lib.SETTINGS[p[0]].COLUMN_OF[p[1]]) for p in panels}
    mats = {}
    for cond, role in panels:
        col = lib.SETTINGS[cond].COLUMN_OF[role]
        sub = frame[(frame["condition"] == cond) & (frame["label_column"] == col)]
        mats[(cond, role)] = sub.pivot(index="variant_id", columns="label", values=value).reindex(
            index=ids, columns=heads[(cond, role)][0]).to_numpy(float)
    lim = max(0.05, max(float(np.nanmax(np.abs(m))) for m in mats.values()))
    fig = lib.new_figure(lib.FULL_MM, 172)
    gs = fig.add_gridspec(1, 4, width_ratios=[len(heads[p][0]) for p in panels], left=0.205, right=0.99, top=0.88,
                          bottom=0.135, wspace=0.06)
    for k, (cond, role) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        order, counts = heads[(cond, role)]
        values = mats[(cond, role)]
        if diverging:
            heat(ax, values, DIVERGING, -lim, lim, norm=TwoSlopeNorm(0, -lim, lim))
        else:
            heat(ax, values, t3.sequential_cmap(), 0, 1)
        p19f.block_lines(ax, grid, values.shape[1])
        mark_row(ax, ids.index(chosen[cond]), values.shape[1])
        ax.set_xticks(range(len(order)), [f"{'all cells' if lab == 'all_cells' else lab} ({int(counts[lab])})"
                                          for lab in order], rotation=90)
        ax.set_title(f"{cond}\n{role} label", fontsize=7)
        if k == 0:
            row_labels(ax, grid)
    scale = f"diverging scale centred at 0 (+/- {lim:.2f})" if diverging else "colour scale 0-1"
    fig.text(0.005, 0.995, f"{title}, per condition\n({scale}; labels with < 10 cells are left out of the macro "
             "averages;\nbold row: baseline; boxed row: the condition's UMAP alternative)", ha="left", va="top",
             fontsize=7)
    figs.save(fig, f"prep_heatmap_{name}", title)


def fig_pc_depth(depth: pd.DataFrame, chosen: dict, figs: lib.Figures) -> None:
    grid = t4.variant_grid()
    ids = [v.variant_id for v in grid]
    blocks = [("spearman_rho_ln_n_detected", "rho, ln(IDs)", [1, 2, 3, 4, 5], True),
              ("spearman_rho_ln_total_intensity", "rho, ln(total int.)", [1, 2, 3, 4, 5], True),
              ("variance_ratio", "variance\nratio", [1, 2], False)]
    vmax_var = max(0.05, float(depth[depth["pc"].isin([1, 2])]["variance_ratio"].max()))
    fig = lib.new_figure(lib.FULL_MM, 158)
    gs = fig.add_gridspec(1, 6, width_ratios=[5, 5, 2, 5, 5, 2], left=0.205, right=0.99, top=0.85, bottom=0.075,
                          wspace=0.1)
    for ci, cond in enumerate(lib.CONDITIONS):
        sub = depth[depth["condition"] == cond]
        axes = []
        for k, (col, title, pcs, div) in enumerate(blocks):
            ax = fig.add_subplot(gs[0, 3 * ci + k])
            values = sub.pivot(index="variant_id", columns="pc", values=col).reindex(ids)[pcs].to_numpy(float)
            if div:
                heat(ax, values, DIVERGING, -1, 1, norm=TwoSlopeNorm(0, -1, 1))
            else:
                heat(ax, values, t3.sequential_cmap(), 0, vmax_var)
            p19f.block_lines(ax, grid, values.shape[1])
            mark_row(ax, ids.index(chosen[cond]), values.shape[1])
            ax.set_xticks(range(len(pcs)), [str(p) for p in pcs])
            ax.set_title(title, fontsize=7)
            if ci == 0 and k == 0:
                row_labels(ax, grid)
            axes.append(ax)
        group_header(fig, axes, cond, 0.91)
        fig.text((axes[0].get_position().x0 + axes[-1].get_position().x1) / 2, 0.035, "principal component",
                 ha="center", va="top", fontsize=7)
    fig.text(0.005, 0.995, "Spearman rho of PC scores with depth, per condition (PC signs are arbitrary; the ranking "
             "uses max(|rho PC1|, |rho PC2|)\nwith ln(protein IDs); bold row: baseline; boxed row: the condition's UMAP "
             "alternative)", ha="left", va="top", fontsize=7)
    figs.save(fig, "prep_heatmap_pc_depth", "PC-depth Spearman rho, Ctrl and ZeroK")


def fig_rank_scatter(rank: pd.DataFrame, chosen: dict, figs: lib.Figures) -> None:
    scopes = ["primary", "second", BOTH]
    fig = lib.new_figure(lib.FULL_MM, 128)
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.99, top=0.84, bottom=0.17, hspace=0.6, wspace=0.3)
    first = None
    for r, cond in enumerate(lib.CONDITIONS):
        for c, scope in enumerate(scopes):
            ax = fig.add_subplot(gs[r, c], sharex=first)
            first = first or ax
            sub = rank[(rank["condition"] == cond) & (rank["scope_role"] == scope)]
            ax.scatter(sub["max_abs_rho_pc12"], sub["label_preservation_mean_rank"], s=10, color=t3.SERIES[0],
                       alpha=0.75, linewidths=0)
            for vid, marker, size in ((chosen[cond], "s", 30), (t4.BASELINE_ID, "o", 24), (EXTRA_UMAP, "D", 22)):
                row = sub[sub["variant_id"] == vid].iloc[0]
                ax.scatter([row["max_abs_rho_pc12"]], [row["label_preservation_mean_rank"]], s=size, marker=marker,
                           facecolors="none", edgecolors=t2.INK["primary"], linewidths=0.8, zorder=4)
            ax.set_ylim(34, 0)
            ax.set_title(f"{cond}: {SCOPE_DISPLAY[scope]}", fontsize=7)
            if r == 1:
                ax.set_xlabel("max |rho| PC1/PC2 vs ln(IDs)")
            if c == 0:
                ax.set_ylabel("label-preservation\nmean rank (1 = best)")
            lib.grid_y(ax)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4, markerfacecolor=t3.SERIES[0],
                      markeredgewidth=0, label="pipeline"),
               Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor="none",
                      markeredgecolor=t2.INK["primary"], label="baseline"),
               Line2D([0], [0], marker="s", linestyle="", markersize=5, markerfacecolor="none",
                      markeredgecolor=t2.INK["primary"], label="UMAP alternative (per condition)"),
               Line2D([0], [0], marker="D", linestyle="", markersize=4.5, markerfacecolor="none",
                      markeredgecolor=t2.INK["primary"], label=f"third UMAP row (authors): {EXTRA_UMAP}")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.text(0.005, 0.995, "Label preservation against depth association, 33 pipelines per condition\nUMAP alternative "
             f"(best combined rank on the primary label): Ctrl {chosen['Ctrl']}; ZeroK {chosen['ZeroK']}",
             ha="left", va="top", fontsize=7)
    figs.save(fig, "prep_rank_scatter", "Label preservation vs depth association, Ctrl and ZeroK")


def fig_feature_detection(feat_sets: pd.DataFrame, figs: lib.Figures) -> None:
    groups = [("all", "all proteins")] + [(f"{n}__{f}", f"{n} {f}") for n in ("none", "median", "total")
                                          for f in ("hvg500", "hvg1000", "hvg2000", "highcov50")]
    fig = lib.new_figure(lib.FULL_MM, 112)
    ax = fig.add_axes([0.075, 0.4, 0.915, 0.47])
    w = 0.36
    for i, (key, _) in enumerate(groups):
        for j, cond in enumerate(lib.CONDITIONS):
            d = feat_sets[feat_sets["condition"] == cond]
            det = d["detection_rate"].to_numpy(float)
            mask = np.ones(len(d), bool) if key == "all" else d[key].astype(str).str.lower().eq("true").to_numpy()
            bp = ax.boxplot(det[mask], positions=[i + (j - 0.5) * w], widths=w * 0.85, showfliers=False,
                            patch_artist=True)
            bp["boxes"][0].set_facecolor(lib.COND_COLORS[cond])
            bp["boxes"][0].set_alpha(0.35)
            for part in ("boxes", "whiskers", "caps"):
                for line in bp[part]:
                    line.set_color(lib.COND_COLORS[cond])
                    line.set_linewidth(0.6)
            bp["medians"][0].set_color(lib.COND_COLORS[cond])
            bp["medians"][0].set_linewidth(1.2)
    counts = {cond: feat_sets[feat_sets["condition"] == cond] for cond in lib.CONDITIONS}

    def n_sel(key, cond):
        d = counts[cond]
        return len(d) if key == "all" else int(d[key].astype(str).str.lower().eq("true").sum())

    ax.set_xticks(range(len(groups)), [f"{name} ({n_sel(k, 'Ctrl')} / {n_sel(k, 'ZeroK')})" for k, name in groups],
                  rotation=90)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("fraction of cells\ndetecting the protein")
    ax.set_ylim(0, 1.02)
    lib.grid_y(ax)
    fig.legend(handles=[Patch(facecolor=lib.COND_COLORS[c], alpha=0.5, edgecolor=lib.COND_COLORS[c], label=c)
                        for c in lib.CONDITIONS], loc="upper right", bbox_to_anchor=(0.99, 0.965), ncol=2,
               frameon=False)
    fig.text(0.005, 0.995, "Detection rate of the selected proteins per feature set, per condition\n(tick: feature set "
             f"and proteins selected, Ctrl / ZeroK; HVG flavor '{t4.HVG_FLAVOR}' on each normalized log matrix)",
             ha="left", va="top", fontsize=7)
    figs.save(fig, "prep_feature_detection", "Detection rate of HVG / high-coverage feature sets, Ctrl and ZeroK")


def fig_umaps(coords: pd.DataFrame, labels: dict, chosen: dict, figs: lib.Figures) -> None:
    rows = [("baseline", {c: t4.BASELINE_ID for c in lib.CONDITIONS}), ("alternative", chosen),
            ("author-selected", {c: EXTRA_UMAP for c in lib.CONDITIONS})]
    variants = {v.variant_id: v for v in t4.variant_grid()}
    all_ids = np.concatenate([np.log(labels[c]["n_detected_proteins"].to_numpy(float)) for c in lib.CONDITIONS])
    lo, hi = np.percentile(all_ids, [1, 99])
    fig = lib.new_figure(lib.FULL_MM, 150)
    gs = fig.add_gridspec(3, 6, left=0.03, right=0.99, top=0.86, bottom=0.16, hspace=0.42, wspace=0.08)
    order = lib.union_order(*(labels[c][lib.SETTINGS[c].COLUMN_OF[r]] for c in lib.CONDITIONS for r in lib.ROLES))
    sca = None
    for r, (kind, vids) in enumerate(rows):
        for ci, cond in enumerate(lib.CONDITIONS):
            lab = labels[cond]
            xy = coords[coords["condition"] == cond].set_index("cell_id").loc[lab.index,
                                                                              [f"{vids[cond]}__umap1",
                                                                               f"{vids[cond]}__umap2"]].to_numpy(float)
            axes = []
            for k, panel in enumerate((*lib.ROLES, "ids")):
                ax = fig.add_subplot(gs[r, 3 * ci + k])
                if panel == "ids":
                    sca = ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(np.log(lab["n_detected_proteins"].to_numpy(float)),
                                                                     lo, hi), cmap="viridis", s=0.8, linewidths=0,
                                     rasterized=True)
                    ax.set_title("ln(protein IDs)", fontsize=7)
                else:
                    values = lab[lib.SETTINGS[cond].COLUMN_OF[panel]].astype(str).to_numpy()
                    for name in order:
                        m = values == name
                        ax.scatter(xy[m, 0], xy[m, 1], s=0.8, color=lib.LABEL_PALETTE.get(name, "0.5"), linewidths=0,
                                   rasterized=True)
                    ax.set_title(f"{panel} label", fontsize=7)
                t2.clean_axes(ax)
                axes.append(ax)
            group_header(fig, axes, f"{cond} {kind}: {short_label(variants[vids[cond]]).removesuffix(' (baseline)')}",
                         axes[0].get_position().y1 + 0.03)
    handles = [Line2D([], [], marker="o", linestyle="", color=lib.LABEL_PALETTE.get(n, "0.5"), markersize=4, label=n)
               for n in order]
    fig.legend(handles=handles, loc="lower left", ncol=len(order), frameon=False, bbox_to_anchor=(0.02, 0.0),
               handletextpad=0.2, columnspacing=1.0)
    cax = fig.add_axes([0.6, 0.085, 0.33, 0.016])
    cb = fig.colorbar(sca, cax=cax, orientation="horizontal")
    cb.set_label("ln(protein IDs), one scale for both conditions", fontsize=7)
    cb.ax.tick_params(labelsize=7, length=2)
    fig.text(0.005, 0.995, "UMAP of the baseline, each condition's top-ranked alternative and a author-selected pipeline "
             "(none/z10/hvg2000)\n(all UMAPs recomputed in step 12a with the same settings, seed 42; colours = dct_labels "
             "labels; ln(IDs) colour range: 1st-99th percentile of both conditions)", ha="left", va="top", fontsize=7)
    figs.save(fig, "prep_umap_baseline_vs_alternative", "UMAP of baseline, alternative and none/z10/hvg2000")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "preprocessing_figures", args)
    lib.ported_from(run, "ctrl_reviewer_core_preprocessing_figures.py", "preprocessing_summary_figures.py")
    tabs = {"clus": read(run, "prep_clustering_metrics"), "purity": read(run, "prep_knn_purity"),
            "sil": read(run, "prep_silhouette"), "depth": read(run, "prep_pc_depth"),
            "feat": read(run, "prep_feature_set_summary")}
    feat_sets = read(run, "prep_feature_sets")
    coords = read(run, "prep_umap_coords", dtype={"cell_id": str})
    canon = lib.read_canonical(run)
    labels = {c: canon[canon["condition"] == c].set_index("cell_id").assign(
        n_detected_proteins=lambda f: f["n_detected"].astype(int)) for c in lib.CONDITIONS}
    for cond in lib.CONDITIONS:
        ids = list(coords.loc[coords["condition"] == cond, "cell_id"])
        if ids != list(labels[cond].index):
            raise ValueError(f"{cond}: UMAP coordinate order differs from the dct_labels canonical labels.")

    summaries, ranks, chosen = {}, {}, {}
    for cond in lib.CONDITIONS:
        summaries[cond], ranks[cond], chosen[cond] = summary_and_rank(cond, tabs)
        run.note(f"{cond}: UMAP alternative = {chosen[cond]} (scope {lib.SETTINGS[cond].PRIMARY}).")
    run.table(t4.grid_table(), "prep_variant_grid")
    summary = lib.stack(summaries)
    rank = lib.stack(ranks)
    run.table(summary, "prep_variant_summary")
    run.table(rank, "prep_ranking")
    run.note("Macro averages use DCT1, DCT2 and ProLIF with >= 10 cells: Ctrl primary DCT1 + DCT2; ZeroK primary DCT1 "
             "only; second columns DCT1 + DCT2 + ProLIF in both conditions.")

    figs = lib.Figures(run, "preprocessing_figures")
    tabs_c = tabs["clus"]
    for role in lib.ROLES:
        fig_clustering(tabs_c, chosen, role, figs)
    fig_label_metric(tabs["purity"], "observed_same_label_fraction", "knn_purity",
                     "kNN label purity (k = 10, same-label fraction of neighbours) per dct_labels label", chosen, figs)
    fig_label_metric(tabs["sil"], "mean_silhouette", "silhouette", "Mean silhouette per dct_labels label (pipeline PC space)",
                     chosen, figs, diverging=True)
    fig_pc_depth(tabs["depth"], chosen, figs)
    fig_rank_scatter(rank, chosen, figs)
    fig_feature_detection(feat_sets, figs)
    fig_umaps(coords, labels, chosen, figs)
    figs.finish()
    run.finish({"label_preservation_metrics": list(p19f.LP_METRICS),
                "selection_scope": {c: lib.SETTINGS[c].PRIMARY for c in lib.CONDITIONS},
                "umap_alternative": chosen, "extra_umap_row_user": EXTRA_UMAP,
                "min_label_n_for_macro": t4.MIN_LABEL_N, "macro_labels": list(lib.CORE_LABELS),
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_preprocessing_figures.csv.")


if __name__ == "__main__":
    main()
