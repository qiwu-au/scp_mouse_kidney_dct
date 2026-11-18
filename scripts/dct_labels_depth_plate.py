#!/usr/bin/env python
"""dct_labels step 5: protein depth and sort plate by label, and ProLIF across the label columns, per condition.

Analysis plan, step 5
(repeats labels_core step 4 and the labels_core ProLIF addendum, 4b). Descriptive only; no test and no interpretation. Plates are
sort plates of one pooled sample per condition (technical label only). Conditions are not compared here (zerok_vs_control).

Depth and plate: per label of every label column (cell, cluster, primary DE-rule label and its confidence; for ZeroK
also the transfer label): cells, protein IDs per cell (median, IQR, range), log10 total intensity, P1 / P2 cells.
ProLIF breakdown (labels_core_prolif_breakdown logic): z of missing vs detected entries of the scored markers;
Top2a / Mki67 detection by label; Spearman rho of each marker score with protein IDs; ProLIF share per protein-ID
quartile.

Inputs (read-only): results/h5ad/dct_labels_DCT_{ctrl,zerok}_step1_submitted.h5ad; dct_labels_submitted_per_cell.csv;
dct_labels_deg_labels_per_cell.csv; dct_labels_transfer_labels_per_cell.csv; data/DCT/DCT_markers_V1.xlsx.
Outputs: results/tables/dct_labels/dct_labels_depth_plate_by_label.csv, dct_labels_prolif_*.csv;
figures dct_labels_depth_plate_by_label_{ctrl,zerok}.

Command (project root):  python scripts/dct_labels_depth_plate.py
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
import cluster_de_lib as t5
import labels_core_lib as t18
import dct_labels_lib as lib
from qc_mito_lib import box_strip

SCRIPT = "dct_labels_depth_plate.py"
CYCLING = ("Top2a", "Mki67")
MM = 1 / 25.4


def columns(cond: str) -> tuple[list[str], str]:
    """Label columns of a condition (cell, cluster, primary[, transfer]) and the primary confidence column."""
    return lib.label_columns(cond), lib.cs(cond, "deg_confidence")


def condition_frame(run: cc.Run, cond: str, step1_all, deg_all, transfer) -> pd.DataFrame:
    cells = step1_all.loc[step1_all["condition"] == cond, "cell_id"].astype(str).to_numpy()
    step1 = lib.per_condition(step1_all, cond, cells)
    deg = lib.per_condition(deg_all, cond, cells)
    frame = step1[["cell_key", "plate", "n_detected", "total_intensity", "DCT1_score", "DCT2_score", "ProLIF_score"]].copy()
    frame[lib.cs(cond, "cell_label")] = step1["cell_label"].astype(str)
    frame[lib.cs(cond, "cluster_label")] = step1["cluster_label"].astype(str)
    frame[lib.cs(cond, "deg_label")] = deg["deg_label"].astype(str)
    frame[lib.cs(cond, "deg_confidence")] = deg["deg_confidence"].astype(str)
    if cond == "ZeroK":
        tr = transfer.set_index("cell_id")
        if list(tr.index) != list(cells):
            raise ValueError("ZeroK transfer table cells differ from the step-1 table")
        frame["cs_zerok_transfer_label"] = tr["cs_zerok_transfer_label"].astype(str)
    return frame


def summary_rows(frame: pd.DataFrame, col: str, conf_col: str) -> list[dict]:
    rows = []
    order = (cc.ordered_labels(frame[col], t18.LABEL_ORDER) if col != conf_col
             else cc.ordered_labels(frame[col], t18.CONFIDENCE_ORDER))
    p1_all = float((frame["plate"] == "P1").mean())
    for label in [*order, "all"]:
        sub = frame if label == "all" else frame[frame[col] == label]
        ids = sub["n_detected"].to_numpy(float)
        lt = np.log10(sub["total_intensity"].to_numpy(float))
        rows.append({"label_column": col, "label": label, "n_cells": len(sub),
                     "protein_ids_median": float(np.median(ids)), "protein_ids_q25": float(np.quantile(ids, 0.25)),
                     "protein_ids_q75": float(np.quantile(ids, 0.75)), "protein_ids_min": float(ids.min()),
                     "protein_ids_max": float(ids.max()), "log10_total_intensity_median": float(np.median(lt)),
                     "log10_total_intensity_q25": float(np.quantile(lt, 0.25)),
                     "log10_total_intensity_q75": float(np.quantile(lt, 0.75)),
                     "n_P1": int((sub["plate"] == "P1").sum()), "n_P2": int((sub["plate"] == "P2").sum()),
                     "fraction_P1": float((sub["plate"] == "P1").mean()), "fraction_P1_all_cells": p1_all})
    return rows


def prolif_breakdown(run: cc.Run, cond: str, frame: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    """labels_core_prolif_breakdown.main on one condition's step-1 object (same .X and intensity layer)."""
    adata = lib.read_step1(run, cond)
    if list(adata.obs_names.astype(str)) != list(frame.index):
        raise ValueError(f"{cond}: step-1 object and per-cell table differ")
    names = pd.Index(adata.var_names.astype(str))
    X = cc.dense(adata.X).astype(np.float64)
    det = cc.dense(adata.layers["intensity"]) > 0
    ids = frame["n_detected"].to_numpy(float)
    lists = t18.present_lists(t18.read_workbook(run), names)

    rows = []
    for g in dict.fromkeys(m for ms in lists.values() for m in ms):
        j = names.get_loc(g)
        d = det[:, j]
        rows.append({"marker": g, "scored_for": ";".join(t for t, ms in lists.items() if g in ms),
                     "n_cells_detected": int(d.sum()), "fraction_cells_detected": float(d.mean()),
                     "mean_z_missing_entry": float(X[~d, j].mean()) if (~d).any() else np.nan,
                     "median_z_detected_entry": float(np.median(X[d, j])) if d.any() else np.nan})
    missing_z = pd.DataFrame(rows).sort_values("fraction_cells_detected", ascending=False)

    cyc = {g: det[:, names.get_loc(g)] for g in CYCLING}
    any_cyc = cyc["Top2a"] | cyc["Mki67"]
    rows = []
    for col in cols:
        lab = frame[col].astype(str).to_numpy()
        for label in [*cc.ordered_labels(lab, t18.LABEL_ORDER), "all"]:
            m = np.ones(len(lab), bool) if label == "all" else lab == label
            rows.append({"label_column": col, "label": label, "n_cells": int(m.sum()),
                         "n_top2a_or_mki67_detected": int((any_cyc & m).sum()),
                         "n_top2a_detected": int((cyc["Top2a"] & m).sum()), "n_mki67_detected": int((cyc["Mki67"] & m).sum()),
                         "fraction_top2a_or_mki67_detected": float(any_cyc[m].mean()),
                         "median_protein_ids": float(np.median(ids[m]))})
    by_label = pd.DataFrame(rows)

    rows = [{"quantity": f"Spearman rho, {t}_score vs protein IDs per cell", "label_column": "", "value":
             float(stats.spearmanr(frame[f"{t}_score"], ids).statistic)} for t in lists]
    rows += [{"quantity": "cells with Top2a or Mki67 detected", "label_column": "", "value": float(any_cyc.sum())},
             {"quantity": "median protein IDs, cells with Top2a or Mki67 detected", "label_column": "",
              "value": float(np.median(ids[any_cyc])) if any_cyc.any() else np.nan},
             {"quantity": "median protein IDs, other cells", "label_column": "", "value": float(np.median(ids[~any_cyc]))}]
    quart = pd.qcut(ids, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    edges = np.quantile(ids, [0, 0.25, 0.5, 0.75, 1])
    for col in cols:
        lab = frame[col].astype(str).to_numpy()
        for q in quart.categories:
            m = np.asarray(quart == q)
            rows.append({"quantity": f"ProLIF share in protein-ID quartile {q}", "label_column": col,
                         "value": float((lab[m] == "ProLIF").mean())})
    rows += [{"quantity": "protein-ID quartile edges (min, Q1, median, Q3, max)", "label_column": "",
              "value": ";".join(f"{e:g}" for e in edges)}]
    return {"prolif_marker_missing_z": missing_z, "prolif_cycling_by_label": by_label,
            "prolif_depth": pd.DataFrame(rows)}


def figure(run: cc.Run, cond: str, frame: pd.DataFrame, table: pd.DataFrame, cols: list[str]) -> None:
    import matplotlib.pyplot as plt

    t18.apply_style()
    fig, axes = plt.subplots(2, len(cols), figsize=(180 * MM, 110 * MM), gridspec_kw={"height_ratios": [2.2, 1]})
    for j, col in enumerate(cols):
        order = cc.ordered_labels(frame[col], t18.LABEL_ORDER)
        ax = axes[0, j]
        box_strip(ax, frame, col, "n_detected", order, t18.LABEL_COLORS)
        ax.set_title(col, fontsize=7)
        ax.set_ylabel("Protein IDs per cell" if j == 0 else "")
        ax.tick_params(axis="x", labelbottom=False)         # labels with n are on the plate row below
        ax = axes[1, j]
        sub = table[(table["label_column"] == col) & (table["label"] != "all")].set_index("label").reindex(order)
        x = np.arange(len(order))
        ax.bar(x, sub["fraction_P1"], 0.6, color=cc.PLATE_PALETTE["P1"], label="P1")
        ax.bar(x, 1 - sub["fraction_P1"], 0.6, bottom=sub["fraction_P1"], color=cc.PLATE_PALETTE["P2"], label="P2")
        ax.axhline(float(sub["fraction_P1_all_cells"].iloc[0]), color=t18.INK["primary"], lw=0.6, ls="--")
        ax.set_xticks(x, [f"{o} ({int(n)})" for o, n in zip(order, sub["n_cells"])], fontsize=7, rotation=35,
                      ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction of cells (P1 / P2)" if j == 0 else "")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cc.PLATE_PALETTE[p]) for p in ("P1", "P2")]
    fig.legend(handles, ["P1", "P2"], title="Sort plate", loc="upper right", bbox_to_anchor=(0.995, 0.36), fontsize=7,
               title_fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.94, bottom=0.21, wspace=0.32, hspace=0.12)
    fig.text(0.07, 0.03, f"Dashed line: P1 share of all {len(frame)} {cond} cells. Plates are sort plates of one "
                         "pooled sample.", fontsize=7, color=t18.INK["secondary"])
    t18.footer(fig, SCRIPT, f"{cond} cells, combined search; descriptive only")
    run.figure(fig, f"depth_plate_by_label_{lib.KEY[cond]}",
               f"dct_labels protein IDs per cell and sort-plate composition per label ({cond})")


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_labels step 5: depth, plate and ProLIF by label per condition.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "depth_plate", args)
    checks = t5.Checks("dct_combined")
    step1 = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str})
    deg = lib.read_table(run, "deg_labels_per_cell", dtype={"cell_id": str})
    transfer = lib.read_table(run, "transfer_labels_per_cell", dtype={"cell_id": str})
    tables: dict[str, list[pd.DataFrame]] = {}
    for cond in lib.CONDITIONS:
        frame = condition_frame(run, cond, step1, deg, transfer)
        label_cols, conf_col = columns(cond)
        table = pd.DataFrame([r for col in [*label_cols, conf_col] for r in summary_rows(frame, col, conf_col)])
        for col in label_cols:
            sub = table[(table["label_column"] == col) & (table["label"] != "all")]
            checks.add(f"{cond}: {col}: median protein IDs per label; P1 share",
                       "; ".join(f"{r.label} {r.protein_ids_median:g}, {r.fraction_P1:.2f}" for r in sub.itertuples()),
                       None)
        pro = prolif_breakdown(run, cond, frame, label_cols)
        by_label = pro["prolif_cycling_by_label"]
        for r in by_label[by_label["label"] == "ProLIF"].itertuples():
            checks.add(f"{cond}: {r.label_column}: ProLIF cells; with Top2a or Mki67 detected; median IDs",
                       f"{r.n_cells}; {r.n_top2a_or_mki67_detected}; {r.median_protein_ids:g}", None)
        for col in label_cols:
            if not (frame[col] == "ProLIF").any():
                checks.add(f"{cond}: {col}: ProLIF cells", "0 (no ProLIF label under this column)", None)
        n_all = by_label[by_label["label"] == "all"]["n_top2a_or_mki67_detected"].unique()
        checks.add(f"{cond}: cells with Top2a or Mki67 detected: same total in every label column", f"{n_all.tolist()}",
                   len(n_all) == 1)
        for name, tab in {"depth_plate_by_label": table, **pro}.items():
            tab = tab.copy()
            tab.insert(0, "condition", cond)
            tables.setdefault(name, []).append(tab)
        figure(run, cond, frame, table, label_cols)
    run.table(checks.frame(), "depth_plate_checks")
    for name, parts in tables.items():
        run.table(pd.concat(parts, ignore_index=True), name)
    run.finish({"columns": {c: [*columns(c)[0], columns(c)[1]] for c in lib.CONDITIONS},
                "depth": "n_detected and total_intensity of each condition's dct_intake object (combined search)",
                "cycling_markers": list(CYCLING), "z": "scaled .X of the step-1 object (submitted pipeline)",
                "quartiles": "pandas qcut of protein IDs per cell within condition", "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_labels_depth_plate_checks.csv.")


if __name__ == "__main__":
    main()
