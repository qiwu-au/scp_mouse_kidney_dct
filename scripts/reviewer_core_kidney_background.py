#!/usr/bin/env python
"""reviewer_core B.6: background Mki67/Top2a detection in the total-kidney dataset, beside DCT, by depth bin.



Inputs (read-only): kidney_object canonical KIDNEY object and labels (intensity layer,
kidney_reannot_label, kidney_pre_label, n_detected_proteins) and kidney_object
canonical DCT object and labels.
Outputs:
  results/tables/reviewer_core/reviewer_core_kidney_bg_*.csv
  results/figures/reviewer_core/reviewer_core_kidney_bg_mki67_top2a.pdf/.png

Detection = intensity > 0. Fractions carry Wilson 95% intervals. Depth bins
are fixed protein-ID ranges shared by both datasets. Protein IDs are counted
over each dataset's own retained proteins (KIDNEY 4,054; DCT 2,561), so equal
bins do not mean equal proteome coverage across datasets.

Command (project root):  python scripts/reviewer_core_kidney_background.py
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

SCRIPT = "reviewer_core_kidney_background.py"
BIN_EDGES = [100, 300, 500, 700, 900, 1100, 1300, np.inf]
KIDNEY_COLUMNS = ("kidney_reannot_label", "kidney_pre_label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reviewer_core B.6: kidney background Mki67/Top2a detection.")
    cc.add_common_args(parser)
    return parser.parse_args()


def bin_labels() -> list[str]:
    out = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        out.append(f"{int(lo)}+" if not np.isfinite(hi) else f"{int(lo)}-{int(hi) - 1}")
    return out


def detection_frame(adata, labels: pd.DataFrame) -> pd.DataFrame:
    intensity = cc.dense(adata.layers["intensity"])
    names = list(adata.var_names)
    out = pd.DataFrame(index=labels.index)
    for marker in ("Mki67", "Top2a"):
        out[f"{marker}_detected"] = intensity[:, names.index(marker)] > 0 if marker in names else False
    out["mki67_or_top2a_detected"] = out["Mki67_detected"] | out["Top2a_detected"]
    out["n_detected_proteins"] = labels["n_detected_proteins"].to_numpy()
    out["depth_bin"] = pd.cut(out["n_detected_proteins"], BIN_EDGES, right=False, labels=bin_labels())
    return out


def summarize(det: pd.DataFrame, group: pd.Series, dataset: str, label_column: str, by_bin: bool) -> list[dict]:
    rows = []
    keys = [group, det["depth_bin"]] if by_bin else [group]
    for key, sub in det.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        n = len(sub)
        row = {"dataset": dataset, "label_column": label_column, "label": key[0],
               "depth_bin": key[1] if by_bin else "all", "n_cells": n,
               "median_ids": float(sub["n_detected_proteins"].median())}
        for col in ("Mki67_detected", "Top2a_detected", "mki67_or_top2a_detected"):
            k = int(sub[col].sum())
            lo, hi = t2.wilson_ci(k, n)
            row.update({f"n_{col}": k, f"fraction_{col}": k / n, f"ci_low_{col}": lo, f"ci_high_{col}": hi})
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    run = cc.Run(t2.TASK, "kidney_background", args)
    for path, label in ((cc.CANONICAL["kidney"]["labels"], "kidney_object canonical KIDNEY labels"),
                        (cc.CANONICAL["kidney"]["h5ad"], "kidney_object canonical KIDNEY object")):
        run.use_input(path, label)
    kidney, klabels = cc.load_canonical("kidney")
    dct, dlabels = t2.load_dct(run)
    kdet, ddet = detection_frame(kidney, klabels), detection_frame(dct, dlabels)
    run.note(f"KIDNEY: Mki67 present {'Mki67' in kidney.var_names}, Top2a present {'Top2a' in kidney.var_names}; "
             f"{kidney.n_obs} cells x {kidney.n_vars} proteins. DCT: {dct.n_obs} cells x {dct.n_vars} proteins.")

    rows = []
    all_k = pd.Series("all", index=klabels.index)
    all_d = pd.Series("all", index=dlabels.index)
    for by_bin in (False, True):
        rows += summarize(kdet, all_k, "KIDNEY", "all_cells", by_bin)
        for col in KIDNEY_COLUMNS:
            rows += summarize(kdet, klabels[col], "KIDNEY", col, by_bin)
        rows += summarize(ddet, all_d, "DCT", "all_cells", by_bin)
        for col in t2.LABEL_COLUMNS:
            rows += summarize(ddet, dlabels[col], "DCT", col, by_bin)
    table = pd.DataFrame(rows)
    run.table(table[table["depth_bin"] == "all"], "kidney_bg_by_label")
    run.table(table[table["depth_bin"] != "all"], "kidney_bg_by_label_and_depth_bin")
    bins = pd.DataFrame([{"depth_bin": b, "lower_inclusive": lo, "upper_exclusive": hi,
                          "n_kidney_cells": int((kdet["depth_bin"] == b).sum()),
                          "n_dct_cells": int((ddet["depth_bin"] == b).sum())}
                         for b, lo, hi in zip(bin_labels(), BIN_EDGES[:-1], BIN_EDGES[1:])])
    run.table(bins, "kidney_bg_depth_bins")

    # Figure: (A) fraction by depth bin, KIDNEY all and DCT by label; (B, C) KIDNEY by cell type.
    metric = "mki67_or_top2a_detected"
    fig = plt.figure(figsize=(8.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.65, wspace=0.28)
    order = bin_labels()
    for c, dcol in enumerate(t2.LABEL_COLUMNS):
        ax = fig.add_subplot(gs[0, c])
        series = [("KIDNEY", "all_cells", "all", "KIDNEY, all cells", t2.INK["primary"], "s"),
                  ("DCT", "all_cells", "all", "DCT, all cells", t2.INK["muted"], "D")]
        series += [("DCT", dcol, lab, f"DCT {lab}", t2.LABEL_PALETTE[lab], "o") for lab in t2.CORE_LABELS]
        for k, (ds, col, lab, text, color, marker) in enumerate(series):
            sub = table[(table["dataset"] == ds) & (table["label_column"] == col) & (table["label"] == lab)
                        & (table["depth_bin"] != "all") & (table["n_cells"] >= 5)].copy()
            x = np.array([order.index(b) for b in sub["depth_bin"]]) + (k - 2) * 0.1
            y = sub[f"fraction_{metric}"].to_numpy(float)
            ax.errorbar(x, y, yerr=[y - sub[f"ci_low_{metric}"], sub[f"ci_high_{metric}"] - y], fmt=marker + "-",
                        ms=3, lw=0.8, elinewidth=0.5, color=color, label=text, mec="white", mew=0.3)
        ax.set_xticks(range(len(order)), order, rotation=30)
        ax.set_xlabel("Protein IDs per cell (own dataset's proteins)")
        ax.set_ylabel("Fraction with Mki67 or Top2a detected")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"By depth bin; DCT labels = {t2.LABEL_DISPLAY[dcol]} (bins with >= 5 cells)", fontsize=6.6)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=5.2, loc="upper left")
    for c, kcol in enumerate(KIDNEY_COLUMNS):
        ax = fig.add_subplot(gs[1, c])
        sub = table[(table["dataset"] == "KIDNEY") & (table["label_column"] == kcol) & (table["depth_bin"] == "all")]
        sub = sub.set_index("label").reindex([l for l in cc.KIDNEY_LABEL_ORDER if l in set(sub["label"])])
        y = sub[f"fraction_{metric}"].to_numpy(float)
        x = np.arange(len(sub))
        ax.bar(x, y, color=t2.SERIES[0], width=0.7)
        ax.errorbar(x, y, yerr=[y - sub[f"ci_low_{metric}"], sub[f"ci_high_{metric}"] - y], fmt="none",
                    ecolor=t2.INK["secondary"], elinewidth=0.6)
        for xi, (k, n) in enumerate(zip(sub[f"n_{metric}"], sub["n_cells"])):
            ax.text(xi, 1.0, f"{int(k)}/{int(n)}", ha="center", va="bottom", fontsize=4.8, rotation=90,
                    color=t2.INK["secondary"])
        ax.set_xticks(x, sub.index, rotation=45, ha="right")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Fraction with Mki67 or Top2a detected")
        ax.set_title(f"KIDNEY by {kcol} (Wilson 95% CI; detected/total)", fontsize=6.6)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Mki67 or Top2a detected: total-kidney dataset versus DCT dataset", x=0.01, ha="left",
                 fontsize=7.5, fontweight="bold")
    t2.footer(fig, SCRIPT, "IDs are counted over 4,054 (KIDNEY) and 2,561 (DCT) retained proteins")
    run.figure(fig, "kidney_bg_mki67_top2a", "Background Mki67/Top2a detection in kidney versus DCT by depth bin")
    run.finish({"bin_edges": [str(e) for e in BIN_EDGES]})


if __name__ == "__main__":
    main()
