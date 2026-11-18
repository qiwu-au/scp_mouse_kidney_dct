#!/usr/bin/env python
"""kidney_annotation step 2, plan step 5: summary figures of the kidney resolution scan.

Plan:
the analysis plan.

Figures (results/figures/kidney_annotation/, PDF + 600 dpi PNG):
  kidney_annotation_scan_summary        - against resolution, one line per graph: clusters, types resolved, Unknown
                                   fraction, seed stability, depth eta^2, ARI of the kidney_annotation labels with both
                                   submitted label columns;
  kidney_annotation_scan_types_heatmap  - types x resolution, one panel per graph (fill = seeds resolving the type,
                                   text = seed-42 confidence and number of clusters);
  kidney_annotation_graph_pc_depth      - Spearman rho of PC1-PC5 with ln(protein IDs) and ln(total intensity) per graph.
Colours: reference categorical slots 1-3 for the graphs (qc_mito_lib.SERIES), with a distinct marker per graph;
blue sequential ramp for the heatmap.

Inputs (read-only): kidney_annotation_scan_summary.csv, kidney_annotation_scan_types_by_resolution.csv,
  kidney_annotation_graph_pc_depth.csv, kidney_annotation_stored_leiden_depth_summary.csv, kidney_object canonical labels (counts).
Command (project root):  python scripts/kidney_annotation_scan_figures.py
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
import kidney_annotation_scan_lib as s

MM = 1 / 25.4
SCRIPT = "kidney_annotation_scan_figures.py"
GRAPH_COLORS = dict(zip(s.GRAPH_NAMES, t3.SERIES[:3]))
GRAPH_MARKERS = dict(zip(s.GRAPH_NAMES, ("o", "s", "^")))
CONF_LETTER = {"high": "H", "medium": "M", "low_shared_only": "L", s.RESCUED: "R"}
HEATMAP_NOTE = ("Text = seed-42 confidence of the type's best cluster: H high (type-specific marker, q < 0.05), "
                "M medium (type-specific marker), L shared markers only; a number = clusters with that label at seed "
                "42. Row label: cells in Fig 2B / Fig 2A.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation step 2 / plan step 5: scan summary figures.")
    cc.add_common_args(parser)
    return parser.parse_args()


def series(ax, frame: pd.DataFrame, y: str, lo: str | None = None, hi: str | None = None, linestyle: str = "-",
           markers: bool = True) -> None:
    """One line per graph against resolution; optional vertical range bars (offset per graph)."""
    for i, g in enumerate(s.GRAPH_NAMES):
        sub = frame[frame["graph"] == g].sort_values("resolution")
        x = sub["resolution"].to_numpy(float)
        dx = (i - 1) * 0.07
        if lo and hi:
            ax.vlines(x + dx, sub[lo], sub[hi], color=GRAPH_COLORS[g], linewidth=0.6, alpha=0.55)
        ax.plot(x + dx, sub[y], color=GRAPH_COLORS[g], linewidth=1.2, linestyle=linestyle,
                marker=GRAPH_MARKERS[g] if markers else None, markersize=2.8, markeredgewidth=0,
                markerfacecolor=GRAPH_COLORS[g] if linestyle == "-" else "white",
                markeredgecolor=GRAPH_COLORS[g])
    ax.set_xticks(s.RESOLUTIONS[1::2])
    ax.set_xlim(0.2, 6.3)
    ax.set_xlabel("Leiden resolution")
    t3.ygrid(ax)


def reference_line(ax, y: float, text: str, va: str = "bottom", left: bool = False,
                   text_y: float | None = None) -> None:
    """Dashed horizontal reference; its label sits on the line, or at ``text_y`` when lines run along it."""
    ax.axhline(y, color=t3.INK["muted"], linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
    ax.text(0.3 if left else 6.25, y if text_y is None else text_y, text, ha="left" if left else "right", va=va,
            fontsize=5.5, color=t3.INK["secondary"])


def summary_figure(run: cc.Run, summary: pd.DataFrame, eta2_stored: float, n_reannot: int, n_pre: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 112 * MM))
    ax = axes[0, 0]
    series(ax, summary, "n_clusters", "n_clusters_all_seeds_min", "n_clusters_all_seeds_max")
    reference_line(ax, 24, "stored Leiden 2.0: 24")
    ax.set_title("A  Clusters (seed 42; bars: seeds 0-9, 42)")
    ax.set_ylabel("clusters")

    ax = axes[0, 1]
    series(ax, summary, "n_types_resolved", "n_types_resolved_all_seeds_min", "n_types_resolved_all_seeds_max")
    series(ax, summary, "n_types_resolved_ge_medium", linestyle=(0, (3, 1.5)))
    reference_line(ax, n_reannot, f"Fig 2B: {n_reannot} types", va="bottom", left=True)
    reference_line(ax, n_pre, f"Fig 2A: {n_pre} types + Unknown", va="bottom", left=True)
    ax.set_ylim(0, n_reannot + 1.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("B  Types resolved (cluster labels)")
    ax.set_ylabel("distinct cell types")

    ax = axes[0, 2]
    series(ax, summary, "fraction_cells_unknown", "fraction_cells_unknown_all_seeds_min",
           "fraction_cells_unknown_all_seeds_max")
    series(ax, summary, "fraction_cells_below_min", linestyle=(0, (1, 1.2)), markers=False)
    ax.set_ylim(0, 1)
    ax.set_title("C  Fraction of cells Unknown")
    ax.set_ylabel("fraction of 712 cells")

    ax = axes[1, 0]
    series(ax, summary, "ari_clusters_vs_seed42_median", "ari_clusters_vs_seed42_min", "ari_clusters_vs_seed42_max")
    ax.set_ylim(0, 1.02)
    ax.set_title("D  Seed stability (seeds 0-9 vs 42)")
    ax.set_ylabel("ARI of clusters (median; bars: range)")

    ax = axes[1, 1]
    series(ax, summary, "eta2_protein_ids", "eta2_protein_ids_all_seeds_min", "eta2_protein_ids_all_seeds_max")
    reference_line(ax, eta2_stored, f"dashed line: stored Leiden 2.0 = {eta2_stored:.3f}", text_y=0.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("E  Depth: eta² of protein IDs")
    ax.set_ylabel("eta² (cluster membership)")

    ax = axes[1, 2]
    series(ax, summary, "ari_t15_labels_vs_pre")
    series(ax, summary, "ari_t15_labels_vs_reannot", linestyle=(0, (3, 1.5)))
    ax.set_ylim(0, max(0.5, float(summary[["ari_t15_labels_vs_pre", "ari_t15_labels_vs_reannot"]].max().max()) + 0.05))
    ax.set_title("F  ARI with submitted labels")
    ax.set_ylabel("ARI")

    handles = [Line2D([0], [0], color=GRAPH_COLORS[g], marker=GRAPH_MARKERS[g], markersize=3.5, linewidth=1.2,
                      markeredgewidth=0, label=s.graph_display(g)) for g in s.GRAPH_NAMES]
    style = [Line2D([0], [0], color=t3.INK["secondary"], linewidth=1.2, label="B: any confidence; C: Unknown; "
                    "F: vs kidney_pre_label (Fig 2A)"),
             Line2D([0], [0], color=t3.INK["secondary"], linewidth=1.2, linestyle=(0, (3, 1.5)),
                    label="B: >= medium (type-specific marker); F: vs kidney_reannot_label (Fig 2B)"),
             Line2D([0], [0], color=t3.INK["secondary"], linewidth=1.2, linestyle=(0, (1, 1.2)),
                    label=f"C: cells in clusters < {s.MIN_CLUSTER_SIZE} cells (not annotated)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, fontsize=6.2)
    fig.legend(handles=style, loc="upper left", bbox_to_anchor=(0.01, 0.965), ncol=2, fontsize=5.8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91), h_pad=1.2, w_pad=1.0)
    t3.footer(fig, SCRIPT, f"kidney, 712 cells; cluster_de annotation rule; min cluster size {s.MIN_CLUSTER_SIZE}; "
                           "panels B-F at seed 42 unless stated")
    run.figure(fig, "scan_summary", "kidney_annotation kidney resolution scan: clusters, types, Unknown, seed stability, depth, ARI")


def heatmap_figure(run: cc.Run, types: pd.DataFrame, counts: dict[str, dict[str, int]],
                   name: str = "scan_types_heatmap", note: str = HEATMAP_NOTE,
                   description: str = "kidney_annotation kidney resolution scan: cell types resolved per resolution and graph",
                   script: str = SCRIPT) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    seen = set(types.loc[types["n_seeds_resolved"] > 0, "cell_type"])
    rows = [t for t in s.ORDER if t != s.UNKNOWN and (t in seen or counts["reannot"].get(t, 0) > 0)]
    rows += sorted(seen.difference(s.ORDER))   # types outside the V3 workbook (extended marker sets)
    n_seeds = int(types["n_seeds"].max())
    colors = ["#ffffff", *t3.BLUE_RAMP]
    bounds = [-0.5, 0.5, 2.5, 4.5, 6.5, 8.5, 9.5, 10.5, 11.5]
    cmap, norm = ListedColormap(colors), BoundaryNorm(bounds, len(colors))
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, (28 + 4.6 * len(rows)) * MM), sharey=True)
    for ax, g in zip(axes, s.GRAPH_NAMES):
        sub = types[types["graph"] == g]
        grid = sub.pivot(index="cell_type", columns="resolution", values="n_seeds_resolved").reindex(rows)
        conf = sub.pivot(index="cell_type", columns="resolution", values="best_confidence_seed42").reindex(rows)
        ncl = sub.pivot(index="cell_type", columns="resolution", values="n_clusters_seed42").reindex(rows)
        im = ax.imshow(grid.to_numpy(float), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(rows)):
            for j in range(grid.shape[1]):
                c = conf.iat[i, j]
                if isinstance(c, str) and c:
                    n = int(ncl.iat[i, j])
                    txt = CONF_LETTER.get(c, "?") + (str(n) if n > 1 else "")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=5,
                            color="white" if grid.iat[i, j] >= 7 else t3.INK["primary"])
        ax.set_xticks(range(grid.shape[1]), [f"{r:g}" for r in grid.columns], fontsize=5.5, rotation=90)
        ax.set_yticks(range(len(rows)), [f"{t} ({counts['reannot'].get(t, 0)}/{counts['pre'].get(t, 0)})"
                                         for t in rows], fontsize=6)
        ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color=t3.INK["grid"], linewidth=0.4)
        ax.tick_params(which="minor", length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(s.graph_display(g), fontsize=6.8)
        ax.set_xlabel("Leiden resolution")
    fig.subplots_adjust(left=0.1, right=0.875, top=0.93, bottom=0.25, wspace=0.08)
    pos = axes[-1].get_position()
    cax = fig.add_axes((0.89, pos.y0, 0.012, pos.height))
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 1.5, 3.5, 5.5, 7.5, 9, 10, 11])
    cbar.ax.set_yticklabels(["0", "1-2", "3-4", "5-6", "7-8", "9", "10", "11"], fontsize=5.5)
    cbar.set_label(f"seeds (of {n_seeds}) in which the type labels >= 1 cluster", fontsize=5.8)
    cbar.outline.set_visible(False)
    import textwrap

    fig.text(0.005, 0.03, textwrap.fill(note, 175), fontsize=5.5, color=t3.INK["secondary"], va="bottom")
    t3.footer(fig, script, f"kidney, 712 cells; cluster_de rule; min cluster size {s.MIN_CLUSTER_SIZE}")
    run.figure(fig, name, description)


def depth_figure(run: cc.Run, depth: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 50 * MM), sharey=True)
    for ax, (col, title) in zip(axes, (("spearman_rho_ln_n_detected", "ln(protein IDs)"),
                                       ("spearman_rho_ln_total_intensity", "ln(total intensity)"))):
        for i, g in enumerate(s.GRAPH_NAMES):
            sub = depth[depth["graph"] == g].sort_values("pc")
            ax.plot(sub["pc"] + (i - 1) * 0.12, sub[col], linestyle="", marker=GRAPH_MARKERS[g], markersize=4,
                    color=GRAPH_COLORS[g], markeredgewidth=0)
        ax.axhline(0, color=t3.INK["muted"], linewidth=0.6)
        ax.set_xticks(range(1, s.N_DEPTH_PCS + 1), [f"PC{p}" for p in range(1, s.N_DEPTH_PCS + 1)])
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Spearman rho with {title}")
        t3.ygrid(ax)
    axes[0].set_ylabel("rho (PC sign arbitrary)")
    handles = [Line2D([0], [0], color=GRAPH_COLORS[g], marker=GRAPH_MARKERS[g], linestyle="", markersize=4,
                      markeredgewidth=0, label=s.graph_display(g)) for g in s.GRAPH_NAMES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, fontsize=6)
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    t3.footer(fig, SCRIPT, "kidney, 712 cells")
    run.figure(fig, "graph_pc_depth", "kidney_annotation graphs: PC1-PC5 vs depth")


def main() -> None:
    args = parse_args()
    run = cc.Run(s.TASK, "scan_figures", args)
    t3.apply_style()
    summary = s.read_table(run, "scan_summary")
    types = s.read_table(run, "scan_types_by_resolution", keep_default_na=False)
    types["n_seeds_resolved"] = pd.to_numeric(types["n_seeds_resolved"])
    types["n_clusters_seed42"] = pd.to_numeric(types["n_clusters_seed42"])
    types["n_seeds"] = pd.to_numeric(types["n_seeds"])
    depth = s.read_table(run, "graph_pc_depth")
    stored = s.read_table(run, "stored_leiden_depth_summary").set_index("quantity")["value"]
    eta2_stored = float(stored["eta2_protein_ids_on_cluster"])
    labels = pd.read_csv(run.use_input(cc.CANONICAL["kidney"]["labels"], "kidney_object canonical kidney labels"))
    counts = {"reannot": labels["kidney_reannot_label"].value_counts().to_dict(),
              "pre": labels["kidney_pre_label"].value_counts().to_dict()}
    n_reannot = len(counts["reannot"])
    n_pre = len([k for k in counts["pre"] if k != s.UNKNOWN])
    summary_figure(run, summary, eta2_stored, n_reannot, n_pre)
    heatmap_figure(run, types, counts)
    depth_figure(run, depth)
    run.finish({"graph_colors": GRAPH_COLORS, "graph_markers": GRAPH_MARKERS,
                "palette_note": "reference categorical slots 1-3 (qc_mito_lib.SERIES); palette validator not run "
                                "(Node.js not installed); identity also by marker shape and legend"})


if __name__ == "__main__":
    main()
