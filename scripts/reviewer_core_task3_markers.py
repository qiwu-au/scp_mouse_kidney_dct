#!/usr/bin/env python
"""reviewer_core A.1: Task 3 positive and competing marker panels on the V1 labels.

Port of
scripts/DCT_v1/02_task3_markers.py (V2 labels and the V2 reproducibility check
removed).

Inputs (read-only): kidney_object canonical DCT object and labels; data/DCT/DCT_markers_V1.xlsx
(ProLIF row, for the legacy ProLIF context score).
Outputs:
  results/tables/reviewer_core/reviewer_core_task3_*.csv
  results/figures/reviewer_core/reviewer_core_task3_score_violins_by_<label_column>.pdf/.png
  results/h5ad/reviewer_core_DCT_task3_scored.h5ad (kidney_object object + Task 3 scores and flags)

Panels (Umod stays in the TAL panel), score_genes on the scaled .X
(random_state=42, use_raw=False), ambiguity flags = score > 90th percentile of
all 750 cells for the TAL, CNT/CD and interstitial panels. Detection =
intensity > 0; abundance = ln(1 + intensity) in detected cells.

Command (project root):  python scripts/reviewer_core_task3_markers.py
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
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2

ensure_scanpy_compat()
t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "reviewer_core_task3_markers.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reviewer_core A.1: Task 3 marker panels on the V1 labels.")
    parser.add_argument("--ambiguity-quantile", type=float, default=0.90)
    cc.add_common_args(parser)
    return parser.parse_args()


def score_task3(adata, legacy_prolif: list[str], quantile: float):
    """Score panels and set flags as scripts/03_dct_annotation_negative_markers.py main() does."""
    coverage, score_cols, positive_cols, negative_cols = [], [], [], []
    for panel, meta in t2.TASK3_MARKER_PANELS.items():
        score_col = f"{panel}_score"
        present, missing = t2.score(adata, meta["genes"], score_col)
        coverage.append(t2.coverage_row(meta["role"], panel, score_col, meta["genes"], present, missing))
        score_cols.append(score_col)
        (positive_cols if meta["role"] == "positive" else negative_cols).append(score_col)
    present, missing = t2.score(adata, legacy_prolif, "ProLIF_legacy_score")
    coverage.append(t2.coverage_row("context_legacy", "ProLIF_legacy (DCT_markers_V1)", "ProLIF_legacy_score",
                                    legacy_prolif, present, missing))
    score_cols.append("ProLIF_legacy_score")
    present, missing = t2.score(adata, t2.CELL_CYCLE_MARKERS, "cell_cycle_score")
    coverage.append(t2.coverage_row("context_cell_cycle", "Cell_cycle", "cell_cycle_score", t2.CELL_CYCLE_MARKERS,
                                    present, missing))
    score_cols.append("cell_cycle_score")

    positive = adata.obs[positive_cols]
    sorted_scores = np.sort(positive.to_numpy(dtype=float), axis=1)
    adata.obs["task3_dct_positive_label"] = pd.Categorical(positive.idxmax(axis=1).str.replace("_score", "", regex=False))
    adata.obs["task3_dct_positive_score"] = positive.max(axis=1).to_numpy()
    adata.obs["task3_dct_positive_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]

    thresholds, flag_cols = [], []
    for panel in t2.NEGATIVE_PANELS:
        score_col = f"{panel}_score"
        flag_col = f"{panel.lower()}_like_flag"
        thr = t2.threshold(adata.obs[score_col], quantile)
        informative = bool(np.isfinite(thr))
        values = pd.to_numeric(adata.obs[score_col], errors="coerce")
        adata.obs[flag_col] = (values > thr) if informative else False
        flag_cols.append(flag_col)
        thresholds.append({"panel": panel, "score_column": score_col, "flag_column": flag_col, "quantile": quantile,
                           "threshold": thr, "comparison": "score > threshold", "informative": informative,
                           "n_flagged": int(adata.obs[flag_col].astype(bool).sum())})
    assert tuple(flag_cols) == t2.FLAG_COLUMNS, flag_cols
    negative = adata.obs[negative_cols]
    adata.obs["max_tal_cnt_interstitial_score"] = negative.max(axis=1).to_numpy()
    adata.obs["dominant_contamination_panel"] = pd.Categorical(negative.idxmax(axis=1).str.replace("_score", "", regex=False))
    adata.obs["ambiguous_tal_cnt_interstitial"] = adata.obs[flag_cols].astype(bool).any(axis=1)
    adata.uns["task3_marker_panels"] = {panel: meta["genes"] for panel, meta in t2.TASK3_MARKER_PANELS.items()}
    adata.uns["task3_ambiguity_quantile"] = float(quantile)
    return coverage, thresholds, score_cols, positive_cols, negative_cols, flag_cols


def marker_summary(log_int: pd.DataFrame, X: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-label detection fraction and abundance for the individual markers (table only)."""
    coverage, rows = [], []
    for order, (marker, segment) in enumerate(t2.INDIVIDUAL_MARKERS, start=1):
        present = marker in log_int.columns
        detected_all = (log_int[marker] > 0) if present else None
        coverage.append({"order": order, "marker": marker, "expected_segment": segment, "present_in_matrix": present,
                         "n_cells_detected": int(detected_all.sum()) if present else 0,
                         "fraction_cells_detected": float(detected_all.mean()) if present else 0.0})
        if not present:
            continue
        for col in t2.LABEL_COLUMNS:
            for label in cc.ordered_labels(labels[col]):
                ids = labels.index[labels[col] == label]
                values = log_int.loc[ids, marker]
                detected = values[values > 0]
                rows.append({"label_column": col, "label": label, "order": order, "marker": marker,
                             "expected_segment": segment, "n_cells": len(values), "n_detected": len(detected),
                             "detection_fraction": len(detected) / len(values),
                             "median_log1p_detected": detected.median() if len(detected) else np.nan,
                             "q25_log1p_detected": detected.quantile(0.25) if len(detected) else np.nan,
                             "q75_log1p_detected": detected.quantile(0.75) if len(detected) else np.nan,
                             "mean_log1p_all_cells": values.mean(),
                             "median_scaled_X": X.loc[ids, marker].median()})
    return pd.DataFrame(coverage), pd.DataFrame(rows)


def plot_score_violins(per_cell: pd.DataFrame, groups: list[tuple[str, list[str]]], label_col: str,
                       thresholds: dict[str, float], run: cc.Run) -> None:
    order = cc.ordered_labels(per_cell[label_col])
    colors = t2.label_colors(order)
    n_cols = max(len(cols) for _, cols in groups)
    fig, axes = plt.subplots(len(groups), n_cols, figsize=(1.75 * n_cols + 0.6, 1.9 * len(groups) + 0.5),
                             squeeze=False, sharey=False)
    for r, (title, cols) in enumerate(groups):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(cols):
                ax.axis("off")
                continue
            col = cols[c]
            data = [per_cell.loc[per_cell[label_col] == lab, col].to_numpy(float) for lab in order]
            parts = ax.violinplot(data, positions=range(len(order)), widths=0.75, showextrema=False)
            for body, lab in zip(parts["bodies"], order):
                body.set_facecolor(colors[lab])
                body.set_edgecolor("white")
                body.set_alpha(0.85)
            for i, values in enumerate(data):
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                ax.vlines(i, q1, q3, color=t2.INK["primary"], linewidth=1.2)
                ax.scatter([i], [med], color="white", edgecolor=t2.INK["primary"], s=9, zorder=3, linewidths=0.6)
            if col in thresholds and np.isfinite(thresholds[col]):
                ax.axhline(thresholds[col], color=t2.INK["secondary"], linestyle="--", linewidth=0.6)
                ax.text(len(order) - 0.5, thresholds[col], "90th pct", fontsize=5, color=t2.INK["secondary"],
                        ha="right", va="bottom")
            ax.set_xticks(range(len(order)), [f"{lab}\n{len(v)}" for lab, v in zip(order, data)], fontsize=5.8)
            ax.set_title(col.replace("_score", ""), fontsize=6.8)
            ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(f"{title}\nscore_genes score (scaled X)", fontsize=6.5)
    fig.suptitle(f"Task 3 panel scores by {t2.LABEL_DISPLAY[label_col]} (dot = median, bar = IQR; numbers = cells)",
                 x=0.01, ha="left", fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    t2.footer(fig, SCRIPT)
    run.figure(fig, f"task3_score_violins_by_{label_col}", "Task 3 panel score distributions by V1 label")


def main() -> None:
    args = parse_args()
    run = cc.Run(t2.TASK, "task3_markers", args)
    t2.record_reference_code(run, ["scripts/03_dct_annotation_negative_markers.py", "scripts/00_config.py",
                                   "scripts/DCT_v1/02_task3_markers.py"])
    adata, labels = t2.load_dct(run)
    log_int = t2.ln1p_frame(adata)
    legacy = cc.read_marker_workbook(run.use_input(cc.INPUTS["dct"]["markers"], "V1 marker workbook")).get("ProLIF", [])

    coverage, thresholds, score_cols, positive_cols, negative_cols, flag_cols = score_task3(
        adata, legacy, args.ambiguity_quantile)
    run.table(pd.DataFrame(coverage), "task3_marker_coverage")
    run.table(pd.DataFrame(thresholds), "task3_ambiguity_thresholds")

    keep = ["task3_dct_positive_label", "task3_dct_positive_score", "task3_dct_positive_margin", *score_cols,
            "max_tal_cnt_interstitial_score", "dominant_contamination_panel", *flag_cols, "ambiguous_tal_cnt_interstitial"]
    per_cell = labels[["plate", *t2.LABEL_COLUMNS, "n_detected_proteins"]].join(
        adata.obs[keep].set_axis(labels.index, axis=0))
    run.table(per_cell.reset_index(), "task3_marker_scores_per_cell")

    summary_scores = score_cols + ["task3_dct_positive_score", "task3_dct_positive_margin", "max_tal_cnt_interstitial_score"]
    summary_flags = flag_cols + ["ambiguous_tal_cnt_interstitial"]
    frames = []
    for col in (*t2.LABEL_COLUMNS, "task3_dct_positive_label"):
        summary = t2.summary_by(per_cell, col, summary_scores, summary_flags).rename(columns={col: "label"})
        summary.insert(0, "label_column", col)
        frames.append(summary)
    run.table(pd.concat(frames, ignore_index=True), "task3_marker_summary_by_label")

    rows = []
    for col in t2.LABEL_COLUMNS:
        table = pd.crosstab(per_cell[col], per_cell["task3_dct_positive_label"].astype(str))
        for label in cc.ordered_labels(table.index):
            for positive in t2.POSITIVE_PANELS:
                n = int(table.loc[label, positive]) if positive in table.columns else 0
                rows.append({"label_column": col, "label": label, "task3_dct_positive_label": positive,
                             "n_cells": n, "percent_of_label": 100 * n / table.loc[label].sum()})
    run.table(pd.DataFrame(rows), "task3_positive_label_composition")

    X = pd.DataFrame(cc.dense(adata.X), index=labels.index, columns=adata.var_names)
    marker_cov, marker_table = marker_summary(log_int, X, labels)
    run.table(marker_cov, "task3_individual_marker_coverage")
    run.table(marker_table, "task3_individual_markers_by_label")
    run.note("Detection = intensity > 0 (kidney_object intensity layer); abundance = ln(1 + intensity) in detected cells.")

    thr = {t["score_column"]: t["threshold"] for t in thresholds}
    groups = [("Positive DCT panels", positive_cols), ("Competing panels", negative_cols),
              ("Context", ["cell_cycle_score", "ProLIF_legacy_score"])]
    for col in t2.LABEL_COLUMNS:
        plot_score_violins(per_cell, groups, col, thr, run)

    for col in ("task3_dct_positive_label", "dominant_contamination_panel"):
        adata.obs[col] = adata.obs[col].astype(str).astype("category")
    run.h5ad(adata, "DCT_task3_scored")
    run.finish({"legacy_prolif_markers": legacy, "individual_markers": [m for m, _ in t2.INDIVIDUAL_MARKERS],
                "n_flagged_any": int(per_cell["ambiguous_tal_cnt_interstitial"].sum())})


if __name__ == "__main__":
    main()
