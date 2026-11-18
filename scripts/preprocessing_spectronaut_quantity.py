#!/usr/bin/env python
"""preprocessing step 1: which Spectronaut quantity the raw CSVs contain, and what they show about normalization.



Reads the column headers and values of both raw Spectronaut exports and records:
  - header structure (metadata columns, one '<run>.htrms.PG.Quantity' column per cell);
  - how values are encoded (missing / zero / text / positive; integer or not; range);
  - per-cell level descriptors on the analysis matrix (retained cells x retained proteins):
    protein IDs, total intensity, median ln(intensity) over detected proteins, and median
    ln(intensity) over a core set of proteins detected in >= 90% of retained cells,
    with their spread and their Spearman rho with ln(protein IDs);
  - the Spectronaut settings stated by the authors (passed on the command line, stored verbatim).
Objective description only; no inference about the normalization that was applied.

Inputs (read-only): data/DCT/DCT_758cells.csv, data/KIDNEY/KIDNEY_768cells.csv.
Outputs: results/tables/preprocessing/preprocessing_spectronaut_*.csv;
         results/figures/preprocessing/preprocessing_spectronaut_per_cell_levels.pdf/.png

Command (project root):
  python scripts/preprocessing_spectronaut_quantity.py --cross-run-normalization "automatic" --imputation "off"
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
import re
import textwrap

import numpy as np
import pandas as pd

import scp_common as cc
import qc_mito_lib as t3
import preprocessing_lib as t4

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "preprocessing_spectronaut_quantity.py"
CORE_MIN_DETECTION = 0.90
META_COLUMNS = ("PG.ProteinGroups", "PG.Genes", "PG.ProteinDescriptions")
SAMPLE_RE = re.compile(r"^\[(\d+)\]\s+(.+?)\.htrms\.(PG\.[A-Za-z]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="preprocessing step 1: Spectronaut quantity and normalization descriptors.")
    parser.add_argument("--cross-run-normalization", required=True,
                        help="Spectronaut cross-run normalization setting as stated by the authors (stored verbatim).")
    parser.add_argument("--imputation", required=True, help="Spectronaut imputation setting as stated by the authors.")
    parser.add_argument("--answer-date", default="2026-10-13")
    cc.add_common_args(parser)
    return parser.parse_args()


def header_rows(dataset: str, columns: list[str]) -> dict:
    meta = [c for c in columns if c in META_COLUMNS]
    other = [c for c in columns if c not in META_COLUMNS]
    parsed = [SAMPLE_RE.match(c) for c in other]
    unparsed = [c for c, m in zip(other, parsed) if m is None]
    quantities = sorted({m.group(3) for m in parsed if m})
    numbers = [int(m.group(1)) for m in parsed if m]
    return {"dataset": dataset, "n_columns": len(columns), "metadata_columns": ";".join(meta),
            "n_metadata_columns": len(meta), "n_sample_columns": len(other),
            "n_sample_columns_matching_pattern": len(other) - len(unparsed),
            "sample_column_pattern": "[n] <run name>.htrms.<quantity>",
            "quantities_found": ";".join(quantities),
            "run_numbers_consecutive_from_1": numbers == list(range(1, len(numbers) + 1)),
            "first_sample_column": other[0], "last_sample_column": other[-1],
            "unparsed_columns": ";".join(unparsed)}


def value_rows(dataset: str, raw: pd.DataFrame) -> dict:
    values = raw.to_numpy(dtype=object).ravel()
    as_str = pd.Series(values).astype(str)
    is_nan = pd.isna(pd.Series(values)).to_numpy()
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    is_text = (~is_nan) & ~np.isfinite(numeric)
    pos = numeric[np.isfinite(numeric) & (numeric > 0)]
    integers = pd.Series(pos[pos == np.round(pos)]).value_counts()
    return {"dataset": dataset, "n_values": len(values), "n_missing_nan": int(is_nan.sum()),
            "n_zero": int((numeric == 0).sum()), "n_negative": int((numeric < 0).sum()),
            "n_text": int(is_text.sum()),
            "text_values": ";".join(sorted(set(as_str[is_text]))[:10]),
            "n_positive": int(len(pos)), "fraction_positive": len(pos) / len(values),
            "fraction_positive_non_integer": float(np.mean(pos != np.round(pos))) if len(pos) else np.nan,
            "min_positive": float(pos.min()), "q01_positive": float(np.quantile(pos, 0.01)),
            "median_positive": float(np.median(pos)), "q99_positive": float(np.quantile(pos, 0.99)),
            "max_positive": float(pos.max()),
            "n_positive_below_1": int((pos < 1).sum()),
            "n_positive_equal_1": int((pos == 1.0).sum()), "n_positive_below_10": int((pos < 10).sum()),
            "n_positive_integer": int(integers.sum()),
            "n_integer_values_occurring_more_than_once": int((integers > 1).sum()),
            "integer_values_occurring_more_than_once": ";".join(f"{v:g} ({n})" for v, n in integers[integers > 1].items())}


def per_cell(dataset: str, rb: cc.Rebuild) -> tuple[pd.DataFrame, int]:
    x = rb.intensity.to_numpy(dtype=np.float64)
    det = x > 0
    ln = np.where(det, np.log(np.where(det, x, 1.0)), np.nan)
    core = det.mean(axis=0) >= CORE_MIN_DETECTION
    frame = pd.DataFrame({
        "dataset": dataset,
        "cell_id": rb.intensity.index.astype(str),
        "n_detected": det.sum(axis=1),
        "total_intensity": x.sum(axis=1),
        "median_ln_intensity_detected": np.nanmedian(ln, axis=1),
        "n_core_proteins_detected": det[:, core].sum(axis=1),
        "median_ln_intensity_core": np.nanmedian(np.where(det[:, core], ln[:, core], np.nan), axis=1),
    })
    frame["ln_n_detected"] = np.log(frame["n_detected"])
    frame["ln_total_intensity"] = np.log(frame["total_intensity"])
    return frame, int(core.sum())


DESCRIPTORS = {
    "ln_total_intensity": "ln total intensity",
    "median_ln_intensity_detected": "median ln intensity, detected proteins",
    "median_ln_intensity_core": "median ln intensity, core proteins (>= 90% of cells)",
}


def summary_rows(frame: pd.DataFrame, n_core: int) -> list[dict]:
    rows = []
    for col, display in DESCRIPTORS.items():
        v = frame[col].to_numpy(float)
        rho = t3.spearman_ci(v, frame["ln_n_detected"].to_numpy(float))
        q1, q3 = np.quantile(v, [0.25, 0.75])
        rows.append({"dataset": frame["dataset"].iloc[0], "descriptor": col, "display": display,
                     "n_cells": len(v), "n_core_proteins": n_core if "core" in col else np.nan,
                     "median": float(np.median(v)), "q25": float(q1), "q75": float(q3), "iqr": float(q3 - q1),
                     "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()),
                     "range_fold_linear": float(np.exp(v.max() - v.min())),
                     "spearman_rho_vs_ln_n_detected": rho["rho"], "rho_ci_low": rho["ci_low"],
                     "rho_ci_high": rho["ci_high"], "p_value": rho["p_value"]})
    return rows


def figure(run: cc.Run, frames: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for r, dataset in enumerate(("kidney", "dct")):
        frame = frames[dataset]
        for c, (col, display) in enumerate(DESCRIPTORS.items()):
            ax = axes[r, c]
            ax.scatter(frame["n_detected"], frame[col], s=3.5, color=t3.SERIES[0], alpha=0.6, linewidths=0,
                       rasterized=True)
            ax.set_xscale("log")
            row = summary[(summary["dataset"] == dataset) & (summary["descriptor"] == col)].iloc[0]
            ax.text(0.97, 0.03, f"rho = {row['spearman_rho_vs_ln_n_detected']:.2f}\nIQR = {row['iqr']:.2f} ln units",
                    transform=ax.transAxes, va="bottom", ha="right", fontsize=5.6, color=t3.INK["secondary"],
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
            ax.set_xlabel("protein IDs per cell (log scale)")
            ax.set_ylabel(f"{t4.DATASETS[dataset]['display']}\nln units" if c == 0 else "ln units", fontsize=6)
            ax.set_title(textwrap.fill(display, 34), fontsize=6.2)
            t3.ygrid(ax)
    fig.suptitle("Per-cell intensity levels in the Spectronaut PG.Quantity export against protein IDs",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t3.footer(fig, SCRIPT, "retained cells x retained proteins; core = detected in >= 90% of retained cells")
    run.figure(fig, "spectronaut_per_cell_levels", "Per-cell intensity descriptors against protein IDs")


def main() -> None:
    args = parse_args()
    run = cc.Run(t4.TASK, "spectronaut_quantity", args)
    headers, values, frames, summaries = [], [], {}, []
    for dataset in ("kidney", "dct"):
        path = run.use_input(cc.INPUTS[dataset]["raw_csv"], f"raw Spectronaut export ({dataset})")
        data = pd.read_csv(path, low_memory=False)
        headers.append(header_rows(dataset, list(data.columns)))
        values.append(value_rows(dataset, data.drop(columns=list(META_COLUMNS))))
        rb = cc.rebuild_ln_intensity(dataset, run)
        frame, n_core = per_cell(dataset, rb)
        frames[dataset] = frame
        summaries.extend(summary_rows(frame, n_core))
    summary = pd.DataFrame(summaries)
    settings = pd.DataFrame([
        {"item": "exported quantity", "value": ";".join(sorted({h["quantities_found"] for h in headers})),
         "source": "CSV column headers ('<run>.htrms.PG.Quantity')"},
        {"item": "Spectronaut cross-run normalization", "value": args.cross_run_normalization,
         "source": f"authors in chat, {args.answer_date} (stored verbatim)"},
        {"item": "Spectronaut imputation", "value": args.imputation,
         "source": f"authors in chat, {args.answer_date} (stored verbatim)"},
        {"item": "Spectronaut report / setup file in the project", "value": "none found under DCT/ or KIDNEY/",
         "source": "file search (preprocessing planning, 2026-10-13)"},
    ])
    run.table(pd.DataFrame(headers), "spectronaut_columns")
    run.table(pd.DataFrame(values), "spectronaut_value_encoding")
    run.table(pd.concat(frames.values(), ignore_index=True), "spectronaut_per_cell")
    run.table(summary, "spectronaut_per_cell_summary")
    run.table(settings, "spectronaut_settings")
    figure(run, frames, summary)
    run.finish({"core_min_detection": CORE_MIN_DETECTION})


if __name__ == "__main__":
    main()
