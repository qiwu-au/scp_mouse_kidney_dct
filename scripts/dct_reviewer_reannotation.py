#!/usr/bin/env python
"""dct_reviewer step 4: reviewer_core A.4 identity x cycling re-annotation per DCT condition (combined search, dct_labels labels).

Port of ctrl_reviewer_core_reannotation.py
(unchanged method), run once per condition.

Inputs (read-only): results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_task3_scored.h5ad (step 1); dct_labels labels.
Outputs:
  results/tables/dct_reviewer/dct_reviewer_reannot_*.csv (both conditions; ``condition`` column)
  results/figures/dct_reviewer/dct_reviewer_reannot_cycling_by_label.pdf/.png/.tif

Identity = Task 3 highest positive panel (scripts/10_reviewer_dct_reannotation.py); 'Cycling' if
cell_cycle_score > the 90th percentile of that condition's cells (score rule). Alternative rule:
Mki67 or Top2a detected (intensity > 0).

Command (project root):  python scripts/dct_reviewer_reannotation.py
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
import ctrl_reviewer_core_reannotation  # noqa: F401  (reference code; recorded as an input)
import dct_reviewer_lib as lib

lib.style()

SCRIPT = "dct_reviewer_reannotation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 4: identity x cycling re-annotation per DCT condition.")
    parser.add_argument("--quantile", type=float, default=0.90)
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str, quantile: float) -> dict:
    s = lib.SETTINGS[cond]
    adata = lib.read_h5ad(run, f"DCT_{s.KEY}_task3_scored")
    _, labels = lib.load(run, cond)
    if list(labels.index) != list(adata.obs_names.astype(str)):
        raise ValueError(f"{cond}: label and object cell order differ.")
    log_int = t2.ln1p_frame(adata)

    data = labels[["cell_key", "plate", *s.LABEL_COLUMNS]].copy()
    data["task3_dct_positive_label"] = adata.obs["task3_dct_positive_label"].astype(str).to_numpy()
    data["cell_cycle_score"] = adata.obs["cell_cycle_score"].to_numpy(float)
    data["ambiguous_tal_cnt_interstitial"] = adata.obs["ambiguous_tal_cnt_interstitial"].astype(bool).to_numpy()
    threshold = float(data["cell_cycle_score"].quantile(quantile))
    data["reviewer_dct_identity"] = data["task3_dct_positive_label"].map(t2.IDENTITY_MAP)
    data["reviewer_cycling_status"] = np.where(data["cell_cycle_score"] > threshold, "Cycling", "Non-cycling")
    data["reviewer_dct_annotation"] = data["reviewer_cycling_status"] + " " + data["reviewer_dct_identity"]
    data["reviewer_cell_cycle_threshold"] = threshold
    missing = []
    for marker in ("Mki67", "Top2a"):
        if marker not in log_int:
            missing.append(marker)
        data[f"{marker}_detected"] = (log_int[marker] > 0).to_numpy() if marker in log_int else False
    data["mki67_or_top2a_detected"] = data["Mki67_detected"] | data["Top2a_detected"]
    definitions = pd.DataFrame([
        {"rule": "score", "definition": f"Cycling if cell_cycle_score > {cond} {quantile:.0%} quantile",
         "threshold": threshold, "n_cycling": int((data["reviewer_cycling_status"] == "Cycling").sum())},
        {"rule": "detection", "definition": "Mki67 or Top2a detected (intensity > 0)"
         + (f"; not quantified: {', '.join(missing)}" if missing else ""), "threshold": np.nan,
         "n_cycling": int(data["mki67_or_top2a_detected"].sum())}])

    rows, cross = [], []
    for col in s.LABEL_COLUMNS:
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
    agreement = pd.crosstab(data["reviewer_cycling_status"],
                            data["mki67_or_top2a_detected"].map({True: "Mki67_or_Top2a_detected",
                                                                 False: "neither_detected"}),
                            margins=True, margins_name="Total").reset_index()
    agreement.columns.name = None
    return {"per_cell": lib.to_generic(data, cond).reset_index(), "definitions": definitions,
            "summary": pd.DataFrame(rows), "cross": pd.concat(cross, ignore_index=True), "agreement": agreement,
            "threshold": threshold, "missing": missing}


def plot(results: dict, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 120)
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.99, top=0.88, bottom=0.11, hspace=0.75, wspace=0.08)
    first = None
    for r, role in enumerate(lib.ROLES):
        cols = {c: lib.SETTINGS[c].COLUMN_OF[role] for c in lib.CONDITIONS}
        summaries = {c: results[c]["summary"] for c in lib.CONDITIONS}
        order = lib.union_order(*(summaries[c].loc[summaries[c]["label_column"] == cols[c], "label"]
                                  for c in lib.CONDITIONS))
        for k, cond in enumerate(lib.CONDITIONS):
            ax = fig.add_subplot(gs[r, k], sharey=first)
            first = first or ax
            sub = results[cond]["summary"]
            sub = sub[sub["label_column"] == cols[cond]].set_index("label").reindex(order)
            x = np.arange(len(order))
            a = 100 * sub["fraction_cycling_score_rule"].to_numpy(float)
            b = 100 * sub["fraction_mki67_or_top2a_detected"].to_numpy(float)
            ax.bar(x - 0.2, np.nan_to_num(a), width=0.38, color=t2.SERIES[0], label="cell_cycle_score > 90th pct")
            ax.bar(x + 0.2, np.nan_to_num(b), width=0.38, color=t2.SERIES[1], label="Mki67 or Top2a detected")
            for xi, (va, vb) in enumerate(zip(a, b)):
                if np.isfinite(va):
                    ax.text(xi - 0.2, va + 2, f"{va:.0f}", ha="center", fontsize=7, color=t2.INK["secondary"])
                    ax.text(xi + 0.2, vb + 2, f"{vb:.0f}", ha="center", fontsize=7, color=t2.INK["secondary"])
            n = sub["n_cells"].fillna(0).astype(int)
            ax.set_xticks(x, [f"{l} ({n[l]})" for l in order])
            ax.set_ylim(0, 112)
            ax.set_title(f"{cond}: {cols[cond]}", fontsize=7)
            ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            if k == 0:
                ax.set_ylabel("% of cells in label")
            else:
                ax.tick_params(labelleft=False)
    handles, labs = first.get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper left", bbox_to_anchor=(0.07, 0.965), ncol=2, frameon=False)
    thr = "; ".join(f"{c} {results[c]['threshold']:.3f}" for c in lib.CONDITIONS)
    fig.text(0.005, 0.995, f"Cycling by score rule and by Mki67/Top2a detection per label (rows: primary, second label; "
             f"label (n cells)); score thresholds {thr}", ha="left", va="top", fontsize=7)
    figs.save(fig, "reannot_cycling_by_label", "Cycling by score rule and by Mki67/Top2a detection per label, Ctrl and ZeroK")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "reannotation", args)
    lib.ported_from(run, "ctrl_reviewer_core_reannotation.py")
    results = {c: one_condition(run, c, args.quantile) for c in lib.CONDITIONS}
    for key, name in (("per_cell", "reannot_per_cell"), ("definitions", "reannot_definitions"),
                      ("summary", "reannot_summary_by_label"), ("cross", "reannot_identity_cycling_by_label"),
                      ("agreement", "reannot_rule_agreement")):
        run.table(lib.stack({c: results[c][key] for c in lib.CONDITIONS}), name)
    run.note("Score threshold = 90th percentile of cell_cycle_score within each condition: "
             + "; ".join(f"{c} {results[c]['threshold']:.4f}" for c in lib.CONDITIONS) + ".")
    figs = lib.Figures(run, "reannotation")
    plot(results, figs)
    figs.finish()
    run.finish({"cycling_threshold": {c: results[c]["threshold"] for c in lib.CONDITIONS},
                "markers_not_quantified": {c: results[c]["missing"] for c in lib.CONDITIONS},
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_reannotation.csv.")


if __name__ == "__main__":
    main()
