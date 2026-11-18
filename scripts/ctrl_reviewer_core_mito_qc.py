#!/usr/bin/env python
"""ctrl_reviewer_core step 9: qc_mito steps 2-4 and 6 on the control DCT data (labels_core labels): mito QC metrics, plots, outliers.

Port of
qc_mito_mito_qc.py, DCT only (the kidney results stand). ctrl_reviewer_core changes: the
MitoCarta annotation of the 6,196 control proteins is built here with
qc_mito_lib.annotate_proteins (qc_mito's parsed MitoCarta3.0 list; no new
download) from the intake_core protein metadata; labels are ctrl_deg_label and
ctrl_cluster_label; step 6 has no the earlier analysis Task 1 table for the new data, so the
mt-prefix rule is reported beside the MitoCarta sets without a the earlier analysis column.

Inputs (read-only):
  labels_core labelled control object and canonical labels (layers['intensity'] = raw PG.Quantity, missing = 0)
  results/h5ad/intake_core_DCT_ctrl.h5ad (protein-group metadata only)
  results/tables/qc_mito/qc_mito_mitocarta_mouse_list.csv (qc_mito parse of data/reference/Mouse.MitoCarta3.0.xls)
  results/tables/intake_core/intake_core_mitocarta_protein_flags.csv (cross-check)
Outputs (results/tables/ctrl_reviewer_core/, results/figures/ctrl_reviewer_core/):
  ctrl_reviewer_core_protein_annotation.csv              MitoCarta flags per control protein group
  ctrl_reviewer_core_dct_qc_per_cell.csv                 metrics, labels and outlier flags per cell
  ctrl_reviewer_core_qc_summary_by_label.csv             per label column x label x metric
  ctrl_reviewer_core_outlier_thresholds.csv / _outlier_counts.csv / _outlier_counts_by_plate.csv
  ctrl_reviewer_core_outlier_prolif_enrichment.csv       ProLIF vs DCT1+DCT2 among flagged cells (Fisher, per plate, CMH)
  ctrl_reviewer_core_dct_mito_prolif_contrasts.csv       ProLIF vs DCT1/DCT2 for the mito metrics (unadjusted and depth-adjusted)
  ctrl_reviewer_core_task1_comparison.csv
  figures: dct_mito_vs_ids_*, dct_mito_by_*, dct_mito_umap

Per-cell metrics (retained proteins only: 6,196): protein IDs
(intensity > 0), total intensity, and for each protein set the number of detected
set proteins, the set fraction of protein IDs and the set fraction of summed
intensity. Outliers: |x - median| > 3 x MAD (MAD scaled to the normal SD) within
each label with >= 10 cells, per label column; no cell is excluded.

Command (project root):  python scripts/ctrl_reviewer_core_mito_qc.py
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
import qc_mito_lib as t3
import ctrl_reviewer_core_lib as lib

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "ctrl_reviewer_core_mito_qc.py"
INTAKE_CORE_FLAGS = cc.out_path("intake_core", "tables", "mitocarta_protein_flags", ".csv")
BOX_METRICS = ("mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
               "mt_encoded_intensity_fraction")
SCATTER_METRICS = ("mitocarta_intensity_fraction", "oxphos_intensity_fraction", "mitocarta_id_fraction")
SUMMARY_METRICS = ("n_detected_proteins", "total_intensity", "log10_n_detected", "log10_total_intensity",
                   *[f"n_detected_{s}" for s in t3.SETS], *[f"{s}_id_fraction" for s in t3.SETS],
                   *[f"{s}_intensity_fraction" for s in t3.SETS])
CONTRAST_METRICS = ("mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
                    "mt_encoded_intensity_fraction", "n_detected_mitocarta")
FLAG_KINDS = ("mito_high", "mito_low", "mito_any", "depth_low", "depth_high", "any")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 9: mitochondrial QC metrics and outliers (control DCT).")
    cc.add_common_args(parser)
    return parser.parse_args()


def short(col: str) -> str:
    return lib.SHORT_COLUMN[col]


def build(run: cc.Run, dataset: str):
    adata, labels = lib.load_ctrl(run)
    flags = lib.read_mito_annotation(run, adata.var_names)
    metrics = t3.mito_metrics(cc.dense(adata.layers["intensity"]), flags, adata.obs_names)
    if not np.array_equal(metrics["n_detected_proteins"].to_numpy(), labels["n_detected_proteins"].to_numpy()):
        raise ValueError(f"{dataset}: protein IDs differ from the labels_core canonical table.")
    rel_diff = np.abs(metrics["total_intensity"].to_numpy() / labels["total_intensity"].to_numpy() - 1)
    if rel_diff.max() > 1e-5:
        raise ValueError(f"{dataset}: total intensity differs from the labels_core canonical table ({rel_diff.max():.2e}).")
    run.note(f"{dataset}: protein IDs equal the labels_core table for all {len(metrics)} cells; total intensity max relative "
             f"difference {rel_diff.max():.1e}.")
    keep = ["plate", *lib.QC_MITO_DATASETS[dataset]["label_columns"]]
    frame = labels[keep].join(metrics)
    return adata, frame


def add_outlier_flags(frame: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = []
    for col in lib.QC_MITO_DATASETS[dataset]["label_columns"]:
        s = short(col)
        flags, thr = t3.mad_flags(frame, col)
        thr.insert(0, "dataset", dataset)
        thresholds.append(thr)
        for metric in t3.OUTLIER_METRICS:
            frame[f"outlier__{s}__{metric}"] = flags[metric]
        mito, depth = flags[list(t3.MITO_METRICS)], flags[list(t3.DEPTH_METRICS)]
        counts = frame[col].astype(str).map(frame[col].astype(str).value_counts())
        frame[f"evaluated__{s}"] = (counts >= t3.MIN_GROUP).to_numpy()
        frame[f"flag__{s}__mito_high"] = (mito == "high").any(axis=1)
        frame[f"flag__{s}__mito_low"] = (mito == "low").any(axis=1)
        frame[f"flag__{s}__mito_any"] = mito.isin(["high", "low"]).any(axis=1)
        frame[f"flag__{s}__depth_low"] = (depth == "low").any(axis=1)
        frame[f"flag__{s}__depth_high"] = (depth == "high").any(axis=1)
        frame[f"flag__{s}__any"] = flags.isin(["high", "low"]).any(axis=1)
    return frame, pd.concat(thresholds, ignore_index=True)


def outlier_counts(frame: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, plate_rows = [], []
    for col in lib.QC_MITO_DATASETS[dataset]["label_columns"]:
        s = short(col)
        order = t3.label_order(frame[col], dataset)
        for label in ["all_cells", *order]:
            sub = frame if label == "all_cells" else frame[frame[col].astype(str) == label]
            row = {"dataset": dataset, "label_column": col, "label": label, "n_cells": len(sub),
                   "evaluated": bool(sub[f"evaluated__{s}"].all()) if label != "all_cells" else "",
                   "n_cells_evaluated": int(sub[f"evaluated__{s}"].sum())}
            for kind in FLAG_KINDS:
                row[f"n_{kind}"] = int(sub[f"flag__{s}__{kind}"].sum())
            row["n_mito_high_and_depth_low"] = int((sub[f"flag__{s}__mito_high"] & sub[f"flag__{s}__depth_low"]).sum())
            for metric in t3.OUTLIER_METRICS:
                row[f"n_{metric}_high"] = int((sub[f"outlier__{s}__{metric}"] == "high").sum())
                row[f"n_{metric}_low"] = int((sub[f"outlier__{s}__{metric}"] == "low").sum())
            row["percent_any_of_evaluated"] = (100 * row["n_any"] / row["n_cells_evaluated"]
                                               if row["n_cells_evaluated"] else np.nan)
            rows.append(row)
        if dataset == "dct":
            for plate in ("P1", "P2"):
                for label in ["all_cells", *order]:
                    mask = frame["plate"].astype(str) == plate
                    if label != "all_cells":
                        mask &= frame[col].astype(str) == label
                    sub = frame[mask]
                    row = {"dataset": dataset, "label_column": col, "plate": plate, "label": label,
                           "n_cells": len(sub), "n_cells_evaluated": int(sub[f"evaluated__{s}"].sum())}
                    for kind in FLAG_KINDS:
                        row[f"n_{kind}"] = int(sub[f"flag__{s}__{kind}"].sum())
                    plate_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(plate_rows)


def prolif_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.contingency_tables import StratifiedTable

    rows = []
    for col in lib.QC_MITO_DATASETS["dct"]["label_columns"]:
        s = short(col)
        sub = frame[frame[col].astype(str).isin(["ProLIF", "DCT1", "DCT2"])]
        is_p = (sub[col].astype(str) == "ProLIF").to_numpy()
        plates = sub["plate"].astype(str).to_numpy()
        defs = {kind: sub[f"flag__{s}__{kind}"].to_numpy(bool) for kind in FLAG_KINDS}
        for metric in t3.OUTLIER_METRICS:
            for direction in ("high", "low"):
                defs[f"{metric}_{direction}"] = (sub[f"outlier__{s}__{metric}"] == direction).to_numpy()
        for name, flag in defs.items():
            def table(mask):
                return np.array([[int((is_p & flag & mask).sum()), int((is_p & ~flag & mask).sum())],
                                 [int((~is_p & flag & mask).sum()), int((~is_p & ~flag & mask).sum())]])
            full = table(np.ones(len(sub), bool))
            odds, p = stats.fisher_exact(full)
            row = {"label_column": col, "flag": name, "comparison": "ProLIF_vs_DCT1+DCT2",
                   "n_prolif": int(is_p.sum()), "n_prolif_flagged": int(full[0, 0]),
                   "percent_prolif_flagged": 100 * full[0, 0] / is_p.sum(),
                   "n_reference": int((~is_p).sum()), "n_reference_flagged": int(full[1, 0]),
                   "percent_reference_flagged": 100 * full[1, 0] / (~is_p).sum(),
                   "prolif_share_of_cells": is_p.mean(),
                   "prolif_share_of_flagged": full[0, 0] / flag.sum() if flag.sum() else np.nan,
                   "fisher_odds_ratio": odds, "fisher_p": p}
            strata = []
            for plate in ("P1", "P2"):
                tab = table(plates == plate)
                strata.append(tab)
                o, pp = stats.fisher_exact(tab)
                row.update({f"{plate}_n_prolif_flagged": int(tab[0, 0]), f"{plate}_n_prolif": int(tab[0].sum()),
                            f"{plate}_n_reference_flagged": int(tab[1, 0]), f"{plate}_n_reference": int(tab[1].sum()),
                            f"{plate}_fisher_odds_ratio": o, f"{plate}_fisher_p": pp})
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    st = StratifiedTable([t.astype(float) for t in strata])
                    row.update(cmh_pooled_odds_ratio=float(st.oddsratio_pooled),
                               cmh_p=float(st.test_null_odds(correction=True).pvalue))
            except (ValueError, ZeroDivisionError, FloatingPointError):
                row.update(cmh_pooled_odds_ratio=np.nan, cmh_p=np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    out = cc.add_bh(out, ["label_column"], p_col="fisher_p", out_col="fisher_p_fdr_bh")
    return cc.add_bh(out, ["label_column"], p_col="cmh_p", out_col="cmh_p_fdr_bh")


def set_summary(frame: pd.DataFrame, dataset: str, name: str, source: str, definition: str, n_proteins,
                n_det: pd.Series, id_frac: pd.Series, int_frac: pd.Series) -> dict:
    return {"dataset": dataset, "source": source, "protein_set": name, "definition": definition,
            "n_proteins_in_matrix": n_proteins, "n_cells": len(frame),
            "n_cells_ge1_detected": int((n_det >= 1).sum()), "percent_cells_ge1_detected": 100 * (n_det >= 1).mean(),
            "detected_per_cell_median": n_det.median(), "detected_per_cell_max": n_det.max(),
            "id_fraction_mean": id_frac.mean(), "id_fraction_median": id_frac.median(), "id_fraction_max": id_frac.max(),
            "intensity_fraction_mean": int_frac.mean(), "intensity_fraction_median": int_frac.median(),
            "intensity_fraction_q95": int_frac.quantile(0.95), "intensity_fraction_max": int_frac.max()}


def task1_tables(frames: dict[str, pd.DataFrame], annotations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """qc_mito step 6 without the the earlier analysis columns: no the earlier analysis Task 1 table exists for the new control data."""
    rows = [{"dataset": "dct", "source": "the earlier analysis Task 1", "protein_set": "earlier_task1_mt_prefix",
             "definition": "no the earlier analysis Task 1 table exists for the new control data",
             "n_proteins_in_matrix": np.nan, "n_cells": np.nan}]
    for dataset, frame in frames.items():
        for name in ("mt_prefix_task1", "mt_encoded", "oxphos_subunit", "oxphos", "mitocarta_symbol_only", "mitocarta"):
            rows.append(set_summary(frame, dataset, name, "ctrl_reviewer_core (labels_core control matrix)", t3.SETS[name],
                                    int(annotations[dataset][name].sum()), frame[f"n_detected_{name}"],
                                    frame[f"{name}_id_fraction"], frame[f"{name}_intensity_fraction"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_scatter(run, frame, adata, dataset, col):
    order = t3.label_order(frame[col], dataset)
    colors = lib.label_colors(col)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, metric in zip(axes, SCATTER_METRICS):
        t3.scatter_by_label(ax, frame["n_detected_proteins"], frame[metric], frame[col], order, colors)
        rho = t3.spearman_ci(frame["n_detected_proteins"], frame[metric])
        ax.set_xlabel("Protein IDs per cell")
        ax.set_ylabel(t3.METRIC_DISPLAY[metric], fontsize=6.2)
        ax.set_title(f"Spearman rho = {rho['rho']:.2f} (all cells)", fontsize=6.2, color=t3.INK["secondary"])
    t3.legend_with_counts(axes[-1], frame[col], order, colors, title=lib.QC_MITO_LABEL_DISPLAY[col])
    fig.suptitle(f"{lib.QC_MITO_DATASETS[dataset]['display']}: mitochondrial fractions vs protein IDs, by "
                 f"{lib.QC_MITO_LABEL_DISPLAY[col]}", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    t3.footer(fig, SCRIPT, "intensity = raw PG.Quantity; MitoCarta3.0 symbol or UniProt match")
    run.figure(fig, f"{dataset}_mito_vs_ids_{short(col)}", f"{dataset}: mito fractions vs IDs by {col}")


def fig_box(run, frame, adata, dataset, col):
    order = t3.label_order(frame[col], dataset)
    colors = lib.label_colors(col)
    ring_col = col if col != "plate" else lib.QC_MITO_DATASETS[dataset]["label_columns"][0]
    highlight = frame[f"flag__{short(ring_col)}__mito_high"].to_numpy(bool)
    width = 7.4 if len(order) > 6 else 5.6
    fig, axes = plt.subplots(2, 2, figsize=(width, 4.6))
    for ax, metric in zip(axes.flat, BOX_METRICS):
        t3.box_strip(ax, frame, col, metric, order, colors, highlight=highlight)
        ax.set_title(t3.METRIC_DISPLAY[metric], fontsize=6.6)
        ax.set_ylabel("fraction")
    fig.suptitle(f"{lib.QC_MITO_DATASETS[dataset]['display']}: mitochondrial metrics by {lib.QC_MITO_LABEL_DISPLAY[col]} "
                 f"(n per label under each tick)", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    note = (f"ringed = high outlier on any mito metric (3 x MAD within {ring_col} labels with >= {t3.MIN_GROUP} cells)")
    t3.footer(fig, SCRIPT, note)
    run.figure(fig, f"{dataset}_mito_by_{short(col)}", f"{dataset}: mito metrics by {col}")


def fig_umap(run, frame, adata, dataset):
    xy = np.asarray(adata.obsm["X_umap"])
    col = lib.QC_MITO_DATASETS[dataset]["label_columns"][0]
    order = t3.label_order(frame[col], dataset)
    colors = lib.label_colors(col)
    panels = [("mitocarta_intensity_fraction", t3.METRIC_DISPLAY["mitocarta_intensity_fraction"]),
              ("oxphos_intensity_fraction", t3.METRIC_DISPLAY["oxphos_intensity_fraction"]),
              ("mitocarta_id_fraction", t3.METRIC_DISPLAY["mitocarta_id_fraction"]),
              ("mt_encoded_intensity_fraction", t3.METRIC_DISPLAY["mt_encoded_intensity_fraction"]),
              ("log10_n_detected", "log10 protein IDs")]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    t3.umap_categorical(axes[0, 0], xy, frame[col], order, colors, lib.QC_MITO_LABEL_DISPLAY[col])
    t3.legend_with_counts(axes[0, 0], frame[col], order, colors, loc="upper left", bbox_to_anchor=(-0.02, -0.02),
                          ncol=4 if len(order) > 6 else len(order), columnspacing=0.6, handletextpad=0.2)
    for ax, (metric, title) in zip(list(axes.flat)[1:], panels):
        t3.umap_continuous(fig, ax, xy, frame[metric], title)
    fig.suptitle(f"{lib.QC_MITO_DATASETS[dataset]['display']}: mitochondrial metrics on the canonical UMAP "
                 "(colour range 1st-99th percentile)", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    t3.footer(fig, SCRIPT)
    run.figure(fig, f"{dataset}_mito_umap", f"{dataset}: mito metrics on UMAP")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "mito_qc", args)
    lib.ported_from(run, "qc_mito_mito_qc.py")
    adata0, _ = lib.load_ctrl(run)
    annotation = t3.annotate_proteins(lib.ctrl_protein_meta(run, adata0.var_names), t3.read_mitocarta(run))
    run.table(annotation.reset_index(), lib.ANNOTATION_NAME)
    t17 = pd.read_csv(run.use_input(INTAKE_CORE_FLAGS, "intake_core MitoCarta flags (cross-check)"), keep_default_na=False)
    t17 = t17[(t17["search"] == lib.CONDITION) & (t17["retained"].astype(str) == "True")].set_index("protein")
    t17_same = {c: bool(np.array_equal(t17.reindex(annotation.index)[c].astype(str).eq("True").to_numpy(),
                                       annotation[c].to_numpy(bool))) for c in t3.SETS}
    ours_n = {c: int(annotation[c].sum()) for c in t3.SETS}
    t17_n = {c: int(t17[c].astype(str).eq("True").sum()) for c in t3.SETS}
    run.note(f"MitoCarta annotation of the {len(annotation)} control proteins: {ours_n}; equal to the intake_core Ctrl "
             f"retained-group flags per protein: {t17_same}.")
    if not all(t17_same.values()):
        raise ValueError("ctrl_reviewer_core MitoCarta annotation differs from the intake_core flags; see the run note.")
    del adata0
    frames, annotations, adatas, thresholds, counts, plate_counts, summaries = {}, {}, {}, [], [], [], []
    for dataset in lib.QC_MITO_RUN:
        adata, frame = build(run, dataset)
        annotations[dataset] = lib.read_mito_annotation(run, adata.var_names)
        frame, thr = add_outlier_flags(frame, dataset)
        thresholds.append(thr)
        cnt, pcnt = outlier_counts(frame, dataset)
        counts.append(cnt)
        plate_counts.append(pcnt)
        for col in lib.QC_MITO_DATASETS[dataset]["plot_columns"]:
            summaries.append(t3.summarise(frame, col, SUMMARY_METRICS, dataset))
        frames[dataset], adatas[dataset] = frame, adata
        run.table(frame.reset_index(), f"{dataset}_qc_per_cell")

    run.table(pd.concat(summaries, ignore_index=True), "qc_summary_by_label")
    run.table(pd.concat(thresholds, ignore_index=True), "outlier_thresholds")
    run.table(pd.concat(counts, ignore_index=True), "outlier_counts")
    run.table(pd.concat([p for p in plate_counts if len(p)], ignore_index=True), "outlier_counts_by_plate")
    run.table(prolif_enrichment(frames["dct"]), "outlier_prolif_enrichment")
    contrasts = pd.concat([t3.prolif_contrasts(frames["dct"], col, CONTRAST_METRICS)
                           for col in lib.QC_MITO_DATASETS["dct"]["label_columns"]], ignore_index=True)
    run.table(contrasts, "dct_mito_prolif_contrasts")

    run.table(task1_tables(frames, annotations), "task1_comparison")

    for dataset, frame in frames.items():
        for col in lib.QC_MITO_DATASETS[dataset]["plot_columns"]:
            fig_scatter(run, frame, adatas[dataset], dataset, col)
            fig_box(run, frame, adatas[dataset], dataset, col)
        fig_umap(run, frame, adatas[dataset], dataset)

    run.finish({"min_group": t3.MIN_GROUP, "n_mad": t3.N_MAD, "mad_scale": "normal (1.4826)",
                "outlier_metrics": list(t3.OUTLIER_METRICS), "mito_flag_metrics": list(t3.MITO_METRICS),
                "depth_flag_metrics": list(t3.DEPTH_METRICS), "protein_sets": t3.SETS,
                "annotation_counts": ours_n, "t17_flag_counts": t17_n})


if __name__ == "__main__":
    main()
