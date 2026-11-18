#!/usr/bin/env python
"""ctrl_reviewer_core step 4: reviewer_core A.4 identity x cycling re-annotation on the control data (labels_core labels).

Port of
reviewer_core_reannotation.py (unchanged method; inputs and label columns replaced).

Inputs (read-only): results/h5ad/ctrl_reviewer_core_DCT_ctrl_task3_scored.h5ad (step 1); labels_core canonical labels.
Outputs:
  results/tables/ctrl_reviewer_core/ctrl_reviewer_core_reannot_*.csv
  results/figures/ctrl_reviewer_core/ctrl_reviewer_core_reannot_cycling_by_label.pdf/.png

Identity = Task 3 highest positive panel (scripts/10_reviewer_dct_reannotation.py);
'Cycling' if cell_cycle_score > the overall 90th percentile (score rule).
Alternative rule: Mki67 or Top2a detected (intensity > 0).

Command (project root):  python scripts/ctrl_reviewer_core_reannotation.py
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
import ctrl_reviewer_core_lib as lib

t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "ctrl_reviewer_core_reannotation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 4: identity x cycling re-annotation on the control data.")
    parser.add_argument("--quantile", type=float, default=0.90)
    cc.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "reannotation", args)
    lib.ported_from(run, "reviewer_core_reannotation.py")
    adata = lib.read_h5ad(run, "DCT_ctrl_task3_scored")
    _, labels = lib.load_ctrl(run)
    log_int = t2.ln1p_frame(adata)

    data = labels[["plate", *lib.LABEL_COLUMNS]].copy()
    data["task3_dct_positive_label"] = adata.obs["task3_dct_positive_label"].astype(str).to_numpy()
    data["cell_cycle_score"] = adata.obs["cell_cycle_score"].to_numpy(float)
    data["ambiguous_tal_cnt_interstitial"] = adata.obs["ambiguous_tal_cnt_interstitial"].astype(bool).to_numpy()
    threshold = float(data["cell_cycle_score"].quantile(args.quantile))
    data["reviewer_dct_identity"] = data["task3_dct_positive_label"].map(t2.IDENTITY_MAP)
    data["reviewer_cycling_status"] = np.where(data["cell_cycle_score"] > threshold, "Cycling", "Non-cycling")
    data["reviewer_dct_annotation"] = data["reviewer_cycling_status"] + " " + data["reviewer_dct_identity"]
    data["reviewer_cell_cycle_threshold"] = threshold
    for marker in ("Mki67", "Top2a"):
        data[f"{marker}_detected"] = (log_int[marker] > 0).to_numpy() if marker in log_int else False
    data["mki67_or_top2a_detected"] = data["Mki67_detected"] | data["Top2a_detected"]
    run.table(data.reset_index(), "reannot_per_cell")
    run.table(pd.DataFrame([
        {"rule": "score", "definition": f"Cycling if cell_cycle_score > overall {args.quantile:.0%} quantile",
         "threshold": threshold, "n_cycling": int((data["reviewer_cycling_status"] == "Cycling").sum())},
        {"rule": "detection", "definition": "Mki67 or Top2a detected (intensity > 0)", "threshold": np.nan,
         "n_cycling": int(data["mki67_or_top2a_detected"].sum())}]), "reannot_definitions")

    rows, cross = [], []
    for col in lib.LABEL_COLUMNS:
        for label in cc.ordered_labels(data[col]):
            sub = data[data[col] == label]
            score_cyc = sub["reviewer_cycling_status"] == "Cycling"
            rows.append({"label_column": col, "label": label, "n_cells": len(sub),
                         "n_cycling_score_rule": int(score_cyc.sum()), "fraction_cycling_score_rule": score_cyc.mean(),
                         "n_mki67_detected": int(sub["Mki67_detected"].sum()),
                         "n_top2a_detected": int(sub["Top2a_detected"].sum()),
                         "n_mki67_or_top2a_detected": int(sub["mki67_or_top2a_detected"].sum()),
                         "fraction_mki67_or_top2a_detected": sub["mki67_or_top2a_detected"].mean(),
                         "n_both_rules": int((score_cyc & sub["mki67_or_top2a_detected"]).sum()),
                         "n_contamination_flagged": int(sub["ambiguous_tal_cnt_interstitial"].sum())})
        table = (data.groupby([col, "reviewer_dct_identity", "reviewer_cycling_status"], observed=True).size()
                 .rename("n_cells").reset_index().rename(columns={col: "label"}))
        table.insert(0, "label_column", col)
        cross.append(table)
    summary = pd.DataFrame(rows)
    run.table(summary, "reannot_summary_by_label")
    run.table(pd.concat(cross, ignore_index=True), "reannot_identity_cycling_by_label")
    run.table(pd.crosstab(data["reviewer_cycling_status"],
                          data["mki67_or_top2a_detected"].map({True: "Mki67_or_Top2a_detected", False: "neither_detected"}),
                          margins=True, margins_name="Total"), "reannot_rule_agreement", index=True)

    fig, axes = plt.subplots(1, len(lib.LABEL_COLUMNS), figsize=(3.3 * len(lib.LABEL_COLUMNS), 2.5), sharey=True)
    for ax, col in zip(axes, lib.LABEL_COLUMNS):
        sub = summary[summary["label_column"] == col].set_index("label")
        order = cc.ordered_labels(sub.index)
        x = np.arange(len(order))
        a = 100 * sub.loc[order, "fraction_cycling_score_rule"].to_numpy(float)
        b = 100 * sub.loc[order, "fraction_mki67_or_top2a_detected"].to_numpy(float)
        ax.bar(x - 0.2, a, width=0.38, color=t2.SERIES[0], label="cell_cycle_score > 90th pct")
        ax.bar(x + 0.2, b, width=0.38, color=t2.SERIES[1], label="Mki67 or Top2a detected")
        for xi, (va, vb) in enumerate(zip(a, b)):
            ax.text(xi - 0.2, va + 1, f"{va:.0f}", ha="center", fontsize=5.2, color=t2.INK["secondary"])
            ax.text(xi + 0.2, vb + 1, f"{vb:.0f}", ha="center", fontsize=5.2, color=t2.INK["secondary"])
        ax.set_xticks(x, [f"{l}\n(n={int(sub.loc[l, 'n_cells'])})" for l in order])
        ax.set_title(lib.LABEL_DISPLAY[col], fontsize=7)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("% of cells in label")
    axes[0].legend(loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "reannot_cycling_by_label", "Cycling by score rule and by Mki67/Top2a detection per labels_core label (control data)")
    run.finish({"cycling_threshold": threshold})


if __name__ == "__main__":
    main()
