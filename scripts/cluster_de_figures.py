#!/usr/bin/env python
"""cluster_de step 5: figures for the DEG-based annotation.



Per dataset:
  <ds>_umap_deg_vs_submitted        submitted labels next to the DEG labels of every clustering (stored X_umap);
                                    confidence encoded by fill (hollow = shared markers only), cluster IDs at centroids
  <ds>_umap_clusters_<clustering>   one highlight panel per cluster (cluster ID, size, DEG label, confidence)
  <ds>_dotplot_top_de_<clustering>  top DE proteins per cluster (5, or 3 when >= 20 clusters):
                                    size = detected fraction in the cluster, colour = mean z (canonical .X)
  <ds>_confusion_deg_<column>       DEG label x submitted label (fill = fraction of the submitted label)
  <ds>_confusion_clusters_<column>  cluster x submitted label (fill = fraction of the cluster)
Across datasets: summary_ari_nmi (clusters and DEG labels vs submitted labels; seed range for DEG labels).
DCT: dct_prolif_clusters (ProLIF fraction, protein IDs, plate, reviewer_core cell-cycle score per cluster).

Colours: submitted label colours as stored in the kidney_object objects (Figs 2-3); new encodings use the
reference palette (blue sequential ramp, blue<->red diverging with a grey midpoint, slots 1-2).

Inputs (read-only): cluster_de step tables; kidney_object canonical objects (X_umap, .X, intensity).
Outputs: results/figures/cluster_de/cluster_de_*.pdf/.png (600 dpi)

Command (project root):
  python scripts/cluster_de_figures.py
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
import cluster_de_lib as t5

t3.apply_style()
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "cluster_de_figures.py"
INK = t3.INK
ACCENT = t3.SERIES[0]          # slot 1 blue
ACCENT_2 = t3.SERIES[1]        # slot 2 orange
REST_BAR = "#c3c2b7"           # baseline/axis ink for non-highlighted bars
DIVERGING = LinearSegmentedColormap.from_list(
    "t05_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3b0ad", "#e34948", "#9c2423"])
SEQUENTIAL = LinearSegmentedColormap.from_list("t05_seq", ["#fcfcfb", *t3.BLUE_RAMP])
Z_LIMIT = 2.0
UNKNOWN_DEG = INK["background_cells"]   # Unknown in the DEG panels (recedes behind the labelled cells)
CONF_SHORT = {"high": "high", "medium": "medium", "low_shared_only": "low (shared only)",
              "unknown_no_marker_in_top_de": "no marker in top 25"}
PLOT_COLUMNS = {"kidney": ("kidney_reannot_label", "kidney_pre_label"),
                "dct": ("v1_cell_label", "v1_cluster_label", "plate")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cluster_de step 5: figures.")
    cc.add_common_args(parser)
    return parser.parse_args()


def label_palette(adata, ds: str) -> dict[str, str]:
    colors: dict[str, str] = {}
    for col in t5.DATASETS[ds]["label_columns"]:
        colors.update(t3.label_colors(adata, col))
    colors[t5.UNKNOWN] = cc.UNKNOWN_COLOR
    return colors


def clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def short_name(name: str, ds: str) -> str:
    res = t5.DATASETS[ds]["submitted_resolution"]
    return f"stored {res:.1f}" if name.startswith("stored_") else name[1:]


# ---------------------------------------------------------------------------
# UMAPs
# ---------------------------------------------------------------------------
def umap_labels(ax, xy, groups, conf, order, colors, title: str, clusters=None) -> None:
    groups = np.asarray(groups).astype(str)
    conf = None if conf is None else np.asarray(conf).astype(str)
    for label in order:
        m = groups == label
        if not m.any():
            continue
        color = colors.get(label, INK["muted"])
        if conf is None:
            ax.scatter(xy[m, 0], xy[m, 1], s=3.5, color=color, linewidths=0, rasterized=True)
            continue
        if label == t5.UNKNOWN:   # recede: Unknown cells as small background dots (PST's submitted colour is also grey)
            ax.scatter(xy[m, 0], xy[m, 1], s=1.6, color=UNKNOWN_DEG, linewidths=0, rasterized=True, zorder=0)
            continue
        hollow = m & (conf == "low_shared_only")
        filled = m & ~hollow
        ax.scatter(xy[filled, 0], xy[filled, 1], s=3.5, color=color, linewidths=0, rasterized=True)
        ax.scatter(xy[hollow, 0], xy[hollow, 1], s=4.5, facecolors="none", edgecolors=color, linewidths=0.45,
                   rasterized=True)
    if clusters is not None:
        cl = np.asarray(clusters).astype(str)
        for c in t5.sort_clusters(cl):
            centre = np.median(xy[cl == c], axis=0)
            ax.text(centre[0], centre[1], c, fontsize=4.6, ha="center", va="center", color=INK["primary"],
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
    clean(ax)
    ax.set_title(title, fontsize=6.3)
    counts = pd.Series(groups).value_counts()
    shown = {**colors, t5.UNKNOWN: UNKNOWN_DEG} if conf is not None else colors
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor=shown.get(l, INK["muted"]),
                      markeredgewidth=0, label=f"{l} ({counts.get(l, 0)})") for l in order if counts.get(l, 0)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=4.9,
              handletextpad=0.2, columnspacing=0.6, borderaxespad=0.2)


def fig_umap_deg_vs_submitted(run, ds, adata, per_cell, ann, top_n: int = t5.TOP_N, script: str = SCRIPT) -> None:
    """Submitted labels next to the DEG labels; ``top_n`` != TOP_N only for the top-N sensitivity figure."""
    xy = np.asarray(adata.obsm["X_umap"])
    colors = label_palette(adata, ds)
    order = list(t5.DATASETS[ds]["order"])
    names = t5.clustering_names(ds)
    n_panels = len(PLOT_COLUMNS[ds]) + len(names)
    ncol = 4
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.6, 3.35 * nrow + 0.3))
    axes = axes.ravel()
    i = 0
    for col in PLOT_COLUMNS[ds]:
        if col == "plate":
            pal, lab_order = dict(cc.PLATE_PALETTE), ["P1", "P2"]
            title = "Plate (pooled sort plate; technical)"
        else:
            pal, lab_order = colors, order
            title = f"Submitted: {t5.LABEL_DISPLAY[col]}"
        umap_labels(axes[i], xy, per_cell[col], None, lab_order, pal, title)
        i += 1
    for name in names:
        n_cl = per_cell[f"{name}__cluster"].nunique()
        umap_labels(axes[i], xy, per_cell[f"{name}__deg_label"], per_cell[f"{name}__deg_confidence"], order, colors,
                    f"DEG labels: {t5.clustering_display(name, ds)} ({n_cl} clusters)", per_cell[f"{name}__cluster"])
        i += 1
    key = [Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor=INK["secondary"],
                  markeredgewidth=0, label="high / medium confidence"),
           Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor="none",
                  markeredgecolor=INK["secondary"], markeredgewidth=0.6, label="low: shared markers only"),
           Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor=UNKNOWN_DEG,
                  markeredgewidth=0, label=f"Unknown: no curated marker in top {top_n}")]
    for ax in axes[i:]:
        ax.axis("off")
    fig.legend(handles=key, loc="lower center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.5, 0.012),
               title="DEG panels: fill = confidence; numbers = Leiden cluster IDs", title_fontsize=5.8)
    note = "" if top_n == t5.TOP_N else f", top-{top_n} sensitivity"
    fig.suptitle(f"{t5.DATASETS[ds]['display']}: DEG-based labels vs submitted labels (stored UMAP{note})", x=0.01,
                 ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    t3.footer(fig, script)
    suffix ="" if top_n == t5.TOP_N else f"_top{top_n}"
    run.figure(fig, f"{ds}_umap_deg_vs_submitted{suffix}", f"DEG labels (top {top_n}) per clustering next to the submitted labels")


def fig_umap_clusters(run, ds, adata, per_cell, ann, name) -> None:
    xy = np.asarray(adata.obsm["X_umap"])
    cl = per_cell[f"{name}__cluster"].astype(str).to_numpy()
    clusters = t5.sort_clusters(cl)
    a = ann[ann["clustering"] == name].set_index("cluster")
    ncol = 6
    nrow = int(np.ceil(len(clusters) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.0, 1.42 * nrow + 0.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, clusters):
        m = cl == c
        ax.scatter(xy[~m, 0], xy[~m, 1], s=0.8, color=INK["background_cells"], linewidths=0, rasterized=True)
        ax.scatter(xy[m, 0], xy[m, 1], s=1.6, color=ACCENT, linewidths=0, rasterized=True)
        clean(ax)
        label, conf = a.at[c, "deg_label"], CONF_SHORT[a.at[c, "confidence"]]
        ax.set_title(f"cluster {c} · {int(m.sum())} cells\n{label} · {conf}", fontsize=5.2)
    for ax in axes[len(clusters):]:
        ax.axis("off")
    fig.suptitle(f"{t5.DATASETS[ds]['display']}: {t5.clustering_display(name, ds)}, one panel per cluster "
                 f"(DEG label · confidence)", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    t3.footer(fig, SCRIPT)
    run.figure(fig, f"{ds}_umap_clusters_{name}", f"cluster highlight panels, {name}")


# ---------------------------------------------------------------------------
# Dotplot
# ---------------------------------------------------------------------------
def fig_dotplot(run, ds, adata, per_cell, ann, top, markers: set[str], name) -> None:
    cl = per_cell[f"{name}__cluster"].astype(str).to_numpy()
    clusters = t5.sort_clusters(cl)
    k = 5 if len(clusters) < 20 else 3
    t = top[top["clustering"] == name].sort_values(["cluster", "rank_up"])
    proteins: list[str] = []
    for c in clusters:
        for p in t.loc[t["cluster"] == c, "protein"].head(k):
            if p not in proteins:
                proteins.append(p)
    a = ann[ann["clustering"] == name].set_index("cluster")
    x = pd.DataFrame(cc.dense(adata[:, proteins].X), columns=proteins)
    det = pd.DataFrame(cc.dense(adata[:, proteins].layers["intensity"]) > 0, columns=proteins)
    mean_z = x.groupby(cl).mean().reindex(clusters)
    frac = det.groupby(cl).mean().reindex(clusters)
    width = max(4.5, 0.105 * len(proteins) + 2.2)
    height = 0.17 * len(clusters) + 1.6
    fig, ax = plt.subplots(figsize=(width, height))
    gx, gy = np.meshgrid(np.arange(len(proteins)), np.arange(len(clusters)))
    sca = ax.scatter(gx.ravel(), gy.ravel(), s=(frac.to_numpy().ravel() * 28) + 0.3,
                     c=np.clip(mean_z.to_numpy().ravel(), -Z_LIMIT, Z_LIMIT), cmap=DIVERGING, vmin=-Z_LIMIT,
                     vmax=Z_LIMIT, linewidths=0)
    ax.set_xticks(range(len(proteins)), [f"{p}*" if p in markers else p for p in proteins], rotation=90, fontsize=4.8)
    ax.set_yticks(range(len(clusters)),
                  [f"{c} · {a.at[c, 'deg_label']} ({int((cl == c).sum())})" for c in clusters], fontsize=5.2)
    ax.set_xlim(-0.7, len(proteins) - 0.3)
    ax.set_ylim(len(clusters) - 0.5, -0.5)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{t5.DATASETS[ds]['display']}: top {k} DE proteins per cluster, {t5.clustering_display(name, ds)}"
                 f"  (* = curated marker in the workbook; rows: cluster · DEG label (cells))", fontsize=6.2)
    cbar = fig.colorbar(sca, ax=ax, fraction=0.025, pad=0.01, shrink=0.5, anchor=(0.0, 1.0))
    cbar.set_label("mean z in cluster\n(clipped ±2)", fontsize=5.2)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_visible(False)
    size_handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=INK["muted"], markeredgewidth=0,
                           markersize=np.sqrt(f * 28 + 0.3), label=f"{int(f * 100)}%") for f in (0.25, 0.5, 1.0)]
    ax.legend(handles=size_handles, title="detected in cluster", title_fontsize=5, fontsize=5, loc="lower left",
              bbox_to_anchor=(1.005, 0.0), handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    t3.footer(fig, SCRIPT)
    run.figure(fig, f"{ds}_dotplot_top_de_{name}", f"top DE proteins per cluster, {name}")


# ---------------------------------------------------------------------------
# Confusion heatmaps
# ---------------------------------------------------------------------------
def heat(ax, counts: pd.DataFrame, frac: pd.DataFrame, title: str) -> object:
    im = ax.imshow(frac.to_numpy(float), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    for (r, c), v in np.ndenumerate(counts.to_numpy()):
        if v:
            dark = frac.to_numpy(float)[r, c] > 0.6
            ax.text(c, r, str(int(v)), ha="center", va="center", fontsize=4.3,
                    color="white" if dark else INK["primary"])
    ax.set_xticks(range(counts.shape[1]), counts.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(counts.shape[0]), counts.index, fontsize=5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=6)
    return im


def fig_confusion(run, ds, per_cell, ann, col, kind: str) -> None:
    names = t5.clustering_names(ds)
    truth = per_cell[col].astype(str).to_numpy()
    cols = cc.ordered_labels(set(truth), t5.DATASETS[ds]["order"])
    tables = []
    for name in names:
        if kind == "deg":
            rows_v = per_cell[f"{name}__deg_label"].astype(str).to_numpy()
            idx = cc.ordered_labels(set(rows_v), t5.DATASETS[ds]["order"])
            counts = t5.crosstab(rows_v, truth, idx, cols)
            frac = counts / counts.sum(axis=0).replace(0, np.nan)
        else:
            rows_v = per_cell[f"{name}__cluster"].astype(str).to_numpy()
            idx = t5.sort_clusters(rows_v)
            counts = t5.crosstab(rows_v, truth, idx, cols)
            lab = ann[ann["clustering"] == name].set_index("cluster")["deg_label"]
            counts.index = [f"{c} · {lab[c]}" for c in idx]
            frac = counts.div(counts.sum(axis=1), axis=0)
        tables.append((name, counts, frac.fillna(0)))
    if kind == "deg":
        ncol = 3 if ds == "kidney" else 5
        nrow = int(np.ceil(len(names) / ncol))
        fig, axes = plt.subplots(nrow, ncol, layout="constrained",
                                 figsize=(8.6 if ds == "kidney" else 8.0, (2.6 if ds == "kidney" else 1.9) * nrow))
        fill_label = "fraction of the submitted label's cells"
    else:
        ncol = 5
        max_rows = max(len(t[1]) for t in tables)
        fig, axes = plt.subplots(1, ncol, layout="constrained",
                                 figsize=(10.5 if ds == "kidney" else 8.6, 0.13 * max_rows + 1.6))
        fill_label = "fraction of the cluster's cells"
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, (name, counts, frac) in zip(axes, tables):
        im = heat(ax, counts, frac, t5.clustering_display(name, ds))
    for ax in axes[len(tables):]:
        ax.axis("off")
    cbar = fig.colorbar(im, ax=list(axes[:len(tables)]), fraction=0.02, pad=0.01)
    cbar.set_label(fill_label, fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_visible(False)
    top = 1 - 0.32 / fig.get_figheight()   # room for the suptitle (a custom rect drops its automatic space)
    fig.get_layout_engine().set(rect=(0, 0.03, 1, top - 0.03))   # and for the footer
    what = "DEG label (rows)" if kind == "deg" else "cluster · DEG label (rows)"
    fig.suptitle(f"{t5.DATASETS[ds]['display']}: {what} vs submitted {t5.LABEL_DISPLAY[col]} (columns); "
                 f"numbers = cells", x=0.01, ha="left", fontsize=7.2)
    t3.footer(fig, SCRIPT)
    run.figure(fig, f"{ds}_confusion_{kind if kind == 'deg' else 'clusters'}_{col}",
               f"confusion {kind} vs {col}")


# ---------------------------------------------------------------------------
# Summary and ProLIF
# ---------------------------------------------------------------------------
def fig_summary(run, agree: dict, stab: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    for r, ds in enumerate(("kidney", "dct")):
        names = t5.clustering_names(ds)
        a = agree[ds]
        for c, metric in enumerate(("ari", "nmi")):
            ax = axes[r, c]
            for j, col in enumerate(t5.DATASETS[ds]["label_columns"]):
                color = t3.SERIES[j]
                sub = a[a["label_column"] == col].set_index("clustering").reindex(names)
                for what, marker, face in (("clusters", "o", color), ("deg_labels", "s", "white")):
                    y = sub[f"{metric}_{what}"].to_numpy(float)
                    xs = np.arange(len(names))
                    ax.plot(xs[:-1], y[:-1], color=color, linewidth=1.0 if what == "clusters" else 0.8)
                    ax.scatter(xs, y, marker=marker, s=16, facecolors=face, edgecolors=color, linewidths=0.9, zorder=3)
                if metric == "ari":
                    s = stab[ds]
                    for x, name in enumerate(names[:-1]):
                        v = s.loc[s["clustering"] == name, f"ari_deg_labels_vs_{col}"]
                        ax.vlines(x + (0.08 if j else -0.08), v.min(), v.max(), color=color, linewidth=0.7, alpha=0.6)
                lab_y = sub[f"{metric}_clusters"].iloc[-1]
                ax.annotate(t5.LABEL_DISPLAY[col].split(" (")[0], (len(names) - 1, lab_y), xytext=(6, 0),
                            textcoords="offset points", fontsize=5.2, va="center", color=INK["secondary"])
            ax.set_xticks(range(len(names)), [short_name(n, ds) for n in names], fontsize=5.6)
            ax.set_xlim(-0.4, len(names) + 0.9)
            ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
            ax.axvline(len(names) - 1.5, color=INK["grid"], linewidth=0.6)
            ax.set_ylabel(metric.upper(), fontsize=6)
            t3.ygrid(ax)
            ax.set_title(f"{t5.DATASETS[ds]['display']}: {metric.upper()} vs submitted labels "
                         f"(x = Leiden resolution)", fontsize=6.4)
    handles = [Line2D([0], [0], color=INK["secondary"], marker="o", markerfacecolor=INK["secondary"], linewidth=1.0,
                      markersize=4, label="Leiden clusters"),
               Line2D([0], [0], color=INK["secondary"], marker="s", markerfacecolor="white", linewidth=0.8,
                      markersize=4, label="DEG labels"),
               Line2D([0], [0], color=INK["secondary"], linewidth=0.7, label="DEG labels: range over seeds 0-9, 42 (ARI)"),
               Line2D([0], [0], color=t3.SERIES[0], linewidth=2, label="primary / V1 per-cell label column"),
               Line2D([0], [0], color=t3.SERIES[1], linewidth=2, label="pre-annotation / V1 cluster label column")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.4, bbox_to_anchor=(0.5, 0.025))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    t3.footer(fig, SCRIPT, "stored = submitted clusters (not connected to the resolution series)")
    run.figure(fig, "summary_ari_nmi", "ARI/NMI of clusters and DEG labels vs submitted labels")


def fig_prolif(run, tests: pd.DataFrame, best: pd.DataFrame, per_cell: pd.DataFrame, scores: pd.DataFrame) -> None:
    names = t5.clustering_names("dct")
    rows = [("v1_cell_label_prolif_fraction_in_cluster", "ProLIF fraction\n(V1 per-cell label)",
             float((per_cell["v1_cell_label"] == "ProLIF").mean())),
            ("v1_cluster_label_prolif_fraction_in_cluster", "ProLIF fraction\n(V1 cluster label)",
             float((per_cell["v1_cluster_label"] == "ProLIF").mean())),
            ("median_protein_ids", "median protein IDs", float(per_cell["n_detected_proteins"].median())),
            ("plate_P1_fraction", "plate P1 fraction", float((per_cell["plate"] == "P1").mean())),
            ("cell_cycle_score_median", "reviewer_core cell-cycle score\n(median)", float(scores["cell_cycle_score"].median()))]
    widths = [tests["clustering"].eq(n).sum() + 1.5 for n in names]
    fig, axes = plt.subplots(len(rows), len(names), figsize=(10.5, 6.4), sharey="row",
                             gridspec_kw={"width_ratios": widths})
    for j, name in enumerate(names):
        t = tests[tests["clustering"] == name].copy()
        t["order"] = t["cluster"].map(lambda c: t5.sort_clusters(t["cluster"]).index(c))
        t = t.sort_values("order")
        b_cell = best.loc[(best["clustering"] == name) & (best["label_column"] == "v1_cell_label"), "best_cluster"].iloc[0]
        b_clu = best.loc[(best["clustering"] == name) & (best["label_column"] == "v1_cluster_label"), "best_cluster"].iloc[0]
        colors = [ACCENT if c == str(b_cell) else (ACCENT_2 if c == str(b_clu) else REST_BAR) for c in t["cluster"]]
        for i, (col, ylabel, overall) in enumerate(rows):
            ax = axes[i, j]
            ax.bar(range(len(t)), t[col].to_numpy(float), width=0.78, color=colors, linewidth=0)
            ax.axhline(overall, color=INK["secondary"], linewidth=0.6)
            ax.set_xticks(range(len(t)), t["cluster"], fontsize=4.4 if len(t) > 15 else 5.2)
            ax.tick_params(axis="x", length=0)
            t3.ygrid(ax)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=5.6)
            if i == 0:
                ax.set_title(t5.clustering_display(name, "dct"), fontsize=6.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACCENT, label="best ProLIF match (Jaccard), V1 per-cell label"),
               plt.Rectangle((0, 0), 1, 1, color=ACCENT_2, label="best ProLIF match, V1 cluster label (if different)"),
               plt.Rectangle((0, 0), 1, 1, color=REST_BAR, label="other clusters"),
               Line2D([0], [0], color=INK["secondary"], linewidth=0.6, label="all 750 cells")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("DCT: per-cluster ProLIF fraction, depth, plate and cell-cycle score (x = Leiden cluster ID)",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    t3.footer(fig, SCRIPT, "current DCT data (to be replaced by the new control dataset)")
    run.figure(fig, "dct_prolif_clusters", "ProLIF correspondence of DCT clusters")


def main() -> None:
    args = parse_args()
    run = cc.Run(t5.TASK, "figures", args)
    agree, stab = {}, {}
    for ds in ("kidney", "dct"):
        adata, labels = t5.load(run, ds)
        per_cell = t5.read_table(run, f"{ds}_clusters_per_cell", dtype={"cell_id": str}).set_index("cell_id")
        per_cell = per_cell.astype({c: str for c in per_cell.columns if c.endswith("__cluster")})
        if list(per_cell.index) != list(adata.obs_names.astype(str)):
            raise SystemExit(f"{ds}: per-cell table order differs from the canonical object")
        ann = t5.read_table(run, f"{ds}_cluster_annotation", dtype={"cluster": str})
        top = t5.read_table(run, f"{ds}_de_top25", dtype={"cluster": str})
        cov = t5.read_table(run, f"{ds}_marker_coverage")
        markers = set(cov["marker"].astype(str))
        agree[ds] = t5.read_table(run, f"{ds}_agreement")
        stab[ds] = t5.read_table(run, f"{ds}_seed_stability")
        fig_umap_deg_vs_submitted(run, ds, adata, per_cell, ann)
        for name in t5.clustering_names(ds):
            fig_umap_clusters(run, ds, adata, per_cell, ann, name)
            fig_dotplot(run, ds, adata, per_cell, ann, top, markers, name)
        for col in t5.DATASETS[ds]["label_columns"]:
            fig_confusion(run, ds, per_cell, ann, col, "deg")
            fig_confusion(run, ds, per_cell, ann, col, "clusters")
        if ds == "dct":
            tests = t5.read_table(run, "dct_prolif_cluster_tests", dtype={"cluster": str})
            best = t5.read_table(run, "dct_prolif_best_cluster", dtype={"best_cluster": str})
            scores = pd.read_csv(run.use_input(cc.OUT_ROOTS["tables"] / "reviewer_core" /
                                               "reviewer_core_task4_cell_cycle_scores_per_cell.csv",
                                               "reviewer_core per-cell cell-cycle scores"), dtype={"cell_id": str})
            fig_prolif(run, tests, best, per_cell, scores)
    fig_summary(run, agree, stab)
    run.finish({"z_limit_dotplot": Z_LIMIT, "dotplot_top_k": "5 (< 20 clusters) or 3",
                "colours": "submitted label colours from the kidney_object objects; reference palette for new encodings"})


if __name__ == "__main__":
    main()
