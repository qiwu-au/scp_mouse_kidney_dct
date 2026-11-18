#!/usr/bin/env python
"""reviewer_core A: reviewer summary panel (docx 'Figure X', panels A-D) under each V1 label column.

Port of
scripts/DCT_v1/08_reviewer_panel_and_readme.py (panel part only), which
follows the layout of scripts/08_task3_task4_reviewer_figures.py
(dct_task3_task4_reviewer_summary_panel_v3); its colours are copied here.

Inputs (read-only): reviewer_core_task3_marker_scores_per_cell.csv (A.1) and
reviewer_core_task4_cluster_composition_long.csv (A.2).
Outputs:
  results/tables/reviewer_core/reviewer_core_panel_<data>_<label_column>.csv (chart data)
  results/figures/reviewer_core/reviewer_core_reviewer_summary_panel_<label_column>.pdf/.png

Panels show DCT1, DCT2 and ProLIF; the 4 V1-cluster 'Unknown' cells are in the
tables but not drawn (as in the original panel script).

Command (project root):  python scripts/reviewer_core_reviewer_panel.py
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

t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

SCRIPT = "reviewer_core_reviewer_panel.py"
# scripts/08_task3_task4_reviewer_figures.py (lines 65-120).
TOKENS = {"ink": "#1F2430", "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7"}
NEUTRAL = {"light": "#E2E5EA", "dark": "#464C55"}
COLORS = {
    "blue": {"base": "#A3BEFA", "light": "#CEDFFE", "dark": "#2E4780"},
    "gold": {"base": "#FFE15B", "dark": "#736422"},
    "olive": {"base": "#A3D576", "light": "#BEEB96", "dark": "#386411"},
    "orange": {"base": "#F0986E", "dark": "#804126"},
    "pink": {"base": "#F390CA", "dark": "#8A3A6F"},
}
ORDER = list(t2.CORE_LABELS)
POSITIVE_ORDER = list(t2.POSITIVE_PANELS)
POSITIVE_DISPLAY = {"DCT_core": "DCT core", "DCT1_associated": "DCT1-associated", "DCT2_associated": "DCT2-associated"}
POSITIVE_PALETTE = {"DCT_core": COLORS["blue"]["base"], "DCT1_associated": COLORS["gold"]["base"],
                    "DCT2_associated": COLORS["olive"]["base"]}
POSITIVE_EDGES = {"DCT_core": COLORS["blue"]["dark"], "DCT1_associated": COLORS["gold"]["dark"],
                  "DCT2_associated": COLORS["olive"]["dark"]}
LEGACY_PALETTE = {"DCT1": COLORS["blue"]["light"], "DCT2": COLORS["olive"]["light"], "ProLIF": COLORS["pink"]["base"]}
BRANCHES = {"cc_removed": "Cell-cycle proteins removed", "cc_regressed": "Cell-cycle score regressed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reviewer_core: reviewer summary panel under each V1 label column.")
    cc.add_common_args(parser)
    return parser.parse_args()


def marker_composition(task3: pd.DataFrame, col: str) -> pd.DataFrame:
    counts = task3.groupby([col, "task3_dct_positive_label"]).size().rename("n_cells")
    index = pd.MultiIndex.from_product([ORDER, POSITIVE_ORDER], names=[col, "task3_dct_positive_label"])
    counts = counts.reindex(index, fill_value=0).reset_index().rename(columns={col: "label"})
    totals = counts.groupby("label")["n_cells"].transform("sum")
    counts["percent_cells"] = np.where(totals > 0, counts["n_cells"] / totals * 100, 0.0)
    return counts


def ambiguity(task3: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for label in ORDER:
        sub = task3[task3[col] == label]
        flagged, total = int(sub["ambiguous_tal_cnt_interstitial"].astype(bool).sum()), len(sub)
        rows += [{"label": label, "status": "Flagged", "n_cells": flagged, "percent_cells": 100 * flagged / total},
                 {"label": label, "status": "Not flagged", "n_cells": total - flagged,
                  "percent_cells": 100 * (total - flagged) / total}]
    return pd.DataFrame(rows)


def cell_cycle_summary(task3: pd.DataFrame, col: str) -> pd.DataFrame:
    out = (task3.groupby(col)["cell_cycle_score"]
           .agg(n_cells="count", median="median", mean="mean", q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75)).reindex(cc.ordered_labels(task3[col])).reset_index()
           .rename(columns={col: "label"}))
    return out


def concentration(composition: pd.DataFrame, col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, top_rows = [], []
    for branch, display in BRANCHES.items():
        sub = composition[(composition["object"] == branch) & (composition["resolution"] == "3.00")
                          & (composition["label_column"] == col)]
        n_cells = sub.groupby("cluster")["n_cells"].sum()
        n_prolif = sub[sub["label"] == "ProLIF"].groupby("cluster")["n_cells"].sum().reindex(n_cells.index, fill_value=0)
        per_cluster = (pd.DataFrame({"cluster": n_cells.index, "n_cells": n_cells.to_numpy(int),
                                     "n_prolif": n_prolif.to_numpy(int)})
                       .sort_values("n_prolif", ascending=False, kind="stable"))
        total = int(per_cluster["n_prolif"].sum())
        for metric, k in (("Largest corrected cluster", 1), ("Top three corrected clusters", 3)):
            n = int(per_cluster.head(k)["n_prolif"].sum())
            rows.append({"branch": branch, "branch_display": display, "metric": metric, "n_prolif": n,
                         "n_prolif_total": total, "percent_prolif": 100 * n / total if total else 0.0})
        top = per_cluster.head(8).assign(branch=branch, percent_of_all_prolif=lambda d: 100 * d["n_prolif"] / total)
        top_rows.append(top)
    return pd.DataFrame(rows), pd.concat(top_rows, ignore_index=True)


def style(ax, xgrid: bool, ygrid: bool) -> None:
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=7, length=0)
    for axis, on in ((ax.xaxis, xgrid), (ax.yaxis, ygrid)):
        if on:
            axis.grid(True, color=TOKENS["grid"], linewidth=0.6)
        else:
            axis.grid(False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def plot_panel(comp, amb, task3, conc, col: str, run: cc.Run) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    ax1, ax2, ax3, ax4 = axes.flat
    y = np.arange(len(ORDER))
    left = np.zeros(len(ORDER))
    for positive in POSITIVE_ORDER:
        part = comp[comp["task3_dct_positive_label"] == positive].set_index("label").loc[ORDER]
        values = part["percent_cells"].to_numpy()
        ax1.barh(y, values, left=left, color=POSITIVE_PALETTE[positive], edgecolor=POSITIVE_EDGES[positive],
                 linewidth=0.7, label=POSITIVE_DISPLAY[positive])
        for i, (v, n) in enumerate(zip(values, part["n_cells"])):
            if v >= 12:
                ax1.text(left[i] + v / 2, i, f"{n}\n{v:.0f}%", ha="center", va="center", fontsize=5.8, color=TOKENS["ink"])
        left += values
    counts = task3[col].value_counts()
    ax1.set_yticks(y, [f"{l} (n={counts.get(l, 0)})" for l in ORDER])
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100)
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax1.set_xlabel("Cells within label")
    ax1.set_title("A. Highest DCT marker score", fontsize=8.5, fontweight="bold")
    ax1.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=3, fontsize=6)
    style(ax1, True, False)

    data = [task3.loc[task3[col] == l, "cell_cycle_score"].to_numpy(float) for l in ORDER]
    bp = ax2.boxplot(data, positions=range(len(ORDER)), widths=0.55, showfliers=False, patch_artist=True)
    for patch, label in zip(bp["boxes"], ORDER):
        patch.set_facecolor(LEGACY_PALETTE[label])
        patch.set_edgecolor(NEUTRAL["dark"])
    for part in ("medians", "whiskers", "caps"):
        for line in bp[part]:
            line.set_color(NEUTRAL["dark"])
    rng = np.random.default_rng(0)  # jitter only
    for i, values in enumerate(data):
        ax2.scatter(i + rng.uniform(-0.2, 0.2, len(values)), values, s=1.5, color=NEUTRAL["dark"], alpha=0.18,
                    linewidths=0, rasterized=True)
        ax2.text(i, np.nanmax(values) + 0.1, f"median {np.median(values):.2f}",
                 ha="center", va="bottom", fontsize=5.8, color=TOKENS["ink"])
    ax2.axhline(0, color=TOKENS["ink"], linewidth=0.7, linestyle=":")
    ax2.set_xticks(range(len(ORDER)), ORDER)
    ax2.set_ylabel("Cell-cycle score")
    ax2.set_title("B. Cell-cycle signal", fontsize=8.5, fontweight="bold")
    style(ax2, False, True)

    flagged = amb[amb["status"] == "Flagged"].set_index("label").loc[ORDER]
    not_flagged = amb[amb["status"] == "Not flagged"].set_index("label").loc[ORDER]
    ax3.barh(y, not_flagged["percent_cells"], color=NEUTRAL["light"], edgecolor=NEUTRAL["dark"], linewidth=0.7,
             label="Not flagged")
    ax3.barh(y, flagged["percent_cells"], left=not_flagged["percent_cells"], color=COLORS["orange"]["base"],
             edgecolor=COLORS["orange"]["dark"], linewidth=0.7, label="Flagged")
    for i, label in enumerate(ORDER):
        ax3.text(101, i, f"{int(flagged.loc[label, 'n_cells'])} ({flagged.loc[label, 'percent_cells']:.1f}%)",
                 va="center", fontsize=6.5, color=TOKENS["ink"])
    ax3.set_yticks(y, ORDER)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 122)
    ax3.set_xticks(range(0, 101, 20))
    ax3.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    ax3.set_xlabel("Cells within label")
    ax3.set_title("C. TAL/CNT/interstitial-like flags", fontsize=8.5, fontweight="bold")
    ax3.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2, fontsize=6)
    style(ax3, True, False)

    yb = np.arange(len(BRANCHES))
    for offset, (metric, color, edge) in zip((-0.18, 0.18), (("Largest corrected cluster", COLORS["orange"]["base"],
                                                              COLORS["orange"]["dark"]),
                                                             ("Top three corrected clusters", COLORS["blue"]["base"],
                                                              COLORS["blue"]["dark"]))):
        part = conc[conc["metric"] == metric].set_index("branch").loc[list(BRANCHES)]
        ax4.barh(yb + offset, part["percent_prolif"], height=0.34, color=color, edgecolor=edge, linewidth=0.7,
                 label=metric)
        for yy, (v, n) in enumerate(zip(part["percent_prolif"], part["n_prolif"])):
            ax4.text(v + 1, yy + offset, f"{n} ({v:.1f}%)", va="center", fontsize=6.2, color=TOKENS["ink"])
    ax4.set_yticks(yb, list(BRANCHES.values()))
    ax4.invert_yaxis()
    ax4.set_xlim(0, max(32.0, 5 * np.ceil(conc["percent_prolif"].max() / 5) + 12))
    ax4.xaxis.set_major_formatter(mticker.PercentFormatter(100))
    n_prolif = int(counts.get("ProLIF", 0))
    ax4.set_xlabel(f"Share of all {n_prolif} ProLIF-labelled cells (Leiden 3.0)")
    ax4.set_title("D. ProLIF after cell-cycle correction", fontsize=8.5, fontweight="bold")
    ax4.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2, fontsize=6)
    style(ax4, True, False)

    fig.suptitle("Reviewer summary: DCT marker support and ProLIF cell-cycle context", x=0.01, ha="left",
                 fontsize=10, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.01, 0.945, f"Labels: {t2.LABEL_DISPLAY[col]} ({cc.DCT_LABEL_SOURCES[col]}); n = {len(task3)} cells. "
             "Descriptive panels; no new subtypes are assigned.", ha="left", va="top", fontsize=6.2, color=TOKENS["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.93), h_pad=2.5, w_pad=2.0)
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"reviewer_summary_panel_{col}", "Reviewer summary panel (docx Figure X layout) under one V1 label column")


def main() -> None:
    args = parse_args()
    run = cc.Run(t2.TASK, "reviewer_panel", args)
    t2.record_reference_code(run, ["scripts/08_task3_task4_reviewer_figures.py",
                                   "scripts/DCT_v1/08_reviewer_panel_and_readme.py"])
    task3 = t2.read_table(run, "task3_marker_scores_per_cell")
    composition = t2.read_table(run, "task4_cluster_composition_long", dtype={"resolution": str, "cluster": str})
    for col in t2.LABEL_COLUMNS:
        comp, amb, cyc = marker_composition(task3, col), ambiguity(task3, col), cell_cycle_summary(task3, col)
        conc, top = concentration(composition, col)
        for frame, stem in ((comp, "positive_marker_composition"), (amb, "ambiguity_flags"),
                            (cyc, "cell_cycle_score_summary"), (conc, "prolif_concentration"), (top, "prolif_top_clusters")):
            run.table(frame.assign(label_column=col), f"panel_{stem}_{col}")
        plot_panel(comp, amb, task3, conc, col, run)
    run.note("Panels show DCT1, DCT2 and ProLIF only; the 4 'Unknown' V1-cluster cells are in the tables but not drawn.")
    run.finish()


if __name__ == "__main__":
    main()
