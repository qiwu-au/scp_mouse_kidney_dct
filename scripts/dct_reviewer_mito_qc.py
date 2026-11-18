#!/usr/bin/env python
"""dct_reviewer step 9a: qc_mito steps 2-4 and 6 per DCT condition (combined search): mito QC metrics and MAD flags.

Port of ctrl_reviewer_core_mito_qc.py, run once per condition.
The ctrl_reviewer_core functions add_outlier_flags, outlier_counts, prolif_enrichment and set_summary are called unchanged under
dct_reviewer_lib.bound (settings from dct_reviewer_lib.part2_settings). Replaced because they hard-code the intake_core/labels_core
inputs: the MitoCarta annotation is built from the dct_intake protein table (dct_reviewer_lib.protein_meta; the dct_labels objects
keep gene names only) and cross-checked against the dct_intake MitoCarta flags instead of the intake_core flags; task1_tables is
copied with the source text changed (ctrl_reviewer_core: "ctrl_reviewer_core (labels_core control matrix)").

Inputs (read-only):
  dct_labels labelled objects (layers['intensity'] = raw PG.Quantity, missing = 0) and canonical labels
  results/tables/dct_intake/dct_intake_proteins.csv (protein-group strings)
  results/tables/qc_mito/qc_mito_mitocarta_mouse_list.csv (qc_mito parse of MitoCarta3.0; no download)
  results/tables/dct_intake/dct_intake_mitocarta_protein_flags.csv (cross-check)
Outputs (results/tables/dct_reviewer/, results/figures/dct_reviewer/; every table has `condition`):
  dct_reviewer_protein_annotation.csv            MitoCarta flags per protein and condition
  dct_reviewer_mito_qc_per_cell.csv              metrics, labels (generic names) and outlier flags per cell
  dct_reviewer_mito_qc_summary_by_label.csv      per label column x label x metric (and plate)
  dct_reviewer_mito_outlier_thresholds.csv / _mito_outlier_counts.csv / _mito_outlier_counts_by_plate.csv
  dct_reviewer_mito_outlier_prolif_enrichment.csv  ProLIF vs DCT1+DCT2 among flagged cells (columns with ProLIF)
  dct_reviewer_mito_prolif_contrasts.csv         ProLIF vs DCT1 / DCT2 / DCT1+DCT2 on the mito metrics
  dct_reviewer_mito_task1_comparison.csv         qc_mito step 6: Task 1 mt-prefix rule beside the MitoCarta sets
  figures: dct_reviewer_mito_vs_ids_{primary,second}, dct_reviewer_mito_by_{primary,second,plate}, dct_reviewer_mito_umap

Outliers: |x - median| > 3 x MAD (MAD scaled to the normal SD) within each label with >= 10 cells, per label column;
no cell is excluded. Columns without ProLIF: enrichment and contrasts 'do not apply' (rows with status).

Command (project root):  python scripts/dct_reviewer_mito_qc.py
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
import textwrap

import numpy as np
import pandas as pd

import scp_common as cc
import reviewer_core_lib as t2
import qc_mito_lib as t3
import ctrl_reviewer_core_mito_qc as m19
import dct_reviewer_lib as lib

lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "dct_reviewer_mito_qc.py"
DCT_INTAKE_FLAGS = cc.out_path("dct_intake", "tables", "mitocarta_protein_flags", ".csv")
BOX_METRICS = m19.BOX_METRICS
SCATTER_METRICS = m19.SCATTER_METRICS
SUMMARY_METRICS = m19.SUMMARY_METRICS
CONTRAST_METRICS = m19.CONTRAST_METRICS
SHORT_METRIC = {"mitocarta_intensity_fraction": "MitoCarta, intensity", "mitocarta_id_fraction": "MitoCarta, IDs",
                "oxphos_intensity_fraction": "OXPHOS, intensity", "mt_encoded_intensity_fraction": "mt-encoded, intensity",
                "log10_n_detected": "log10 protein IDs"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 9a: mitochondrial QC metrics and outliers per condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def task1_table(frame: pd.DataFrame, annotation: pd.DataFrame, cond: str) -> pd.DataFrame:
    """Copy of ctrl_reviewer_core_mito_qc.task1_tables for one condition (source text changed)."""
    rows = [{"dataset": "dct", "source": "the earlier analysis Task 1", "protein_set": "earlier_task1_mt_prefix",
             "definition": "no the earlier analysis Task 1 table exists for the combined-search data",
             "n_proteins_in_matrix": np.nan, "n_cells": np.nan}]
    for name in ("mt_prefix_task1", "mt_encoded", "oxphos_subunit", "oxphos", "mitocarta_symbol_only", "mitocarta"):
        rows.append(m19.set_summary(frame, "dct", name, f"dct_reviewer (dct_labels {cond} matrix, combined search)",
                                    t3.SETS[name], int(annotation[name].sum()), frame[f"n_detected_{name}"],
                                    frame[f"{name}_id_fraction"], frame[f"{name}_intensity_fraction"]))
    return pd.DataFrame(rows)


def one_condition(run: cc.Run, cond: str, mitocarta: pd.DataFrame, t26flags: pd.DataFrame) -> dict:
    s = lib.part2_settings(cond)
    adata, labels = lib.load(run, cond)
    meta = lib.protein_meta(run, cond, adata.var_names)
    annotation = t3.annotate_proteins(meta, mitocarta)
    ref = t26flags[(t26flags["condition"] == cond) & t26flags["retained"].astype(str).eq("True")].set_index("protein")
    same = {c: bool(np.array_equal(ref.reindex(annotation.index)[c].astype(str).eq("True").to_numpy(),
                                   annotation[c].to_numpy(bool))) for c in t3.SETS}
    counts = {c: int(annotation[c].sum()) for c in t3.SETS}
    checks = [{"condition": cond, "check": f"MitoCarta set '{c}' equals the dct_intake flags for every retained protein",
               "value": f"{counts[c]} proteins", "pass": same[c]} for c in t3.SETS]

    metrics = t3.mito_metrics(cc.dense(adata.layers["intensity"]), annotation, adata.obs_names)
    ids_equal = bool(np.array_equal(metrics["n_detected_proteins"].to_numpy(), labels["n_detected_proteins"].to_numpy()))
    rel = np.abs(metrics["total_intensity"].to_numpy() / labels["total_intensity"].to_numpy(float) - 1).max()
    checks += [{"condition": cond, "check": "protein IDs per cell equal the dct_labels canonical table",
                "value": f"{len(metrics)} cells", "pass": ids_equal},
               {"condition": cond, "check": "total intensity per cell equals the dct_labels canonical table (rel. diff < 1e-5)",
                "value": f"max rel diff {rel:.1e}", "pass": bool(rel < 1e-5)}]

    frame = labels[["plate", *s.LABEL_COLUMNS]].join(metrics)
    with lib.bound(s, m19):
        frame, thr = m19.add_outlier_flags(frame, "dct")
        cnt, pcnt = m19.outlier_counts(frame, "dct")
    summaries = pd.concat([t3.summarise(frame, col, SUMMARY_METRICS, "dct") for col in (*s.LABEL_COLUMNS, "plate")],
                          ignore_index=True)
    with_prolif = lib.columns_with(labels, s, "ProLIF")
    without = [c for c in s.LABEL_COLUMNS if c not in with_prolif]
    enrich, contrasts = [], []
    if with_prolif:
        with lib.bound(lib.part2_settings(cond, with_prolif), m19):
            enrich.append(m19.prolif_enrichment(frame).assign(status="tested", reason=""))
        for col in with_prolif:
            contrasts.append(t3.prolif_contrasts(frame, col, CONTRAST_METRICS).assign(status="tested", reason=""))
    if without:
        reason = "no ProLIF cell under this column (dct_labels)"
        enrich.append(lib.na_rows(cond, without, "ProLIF vs DCT1+DCT2 among flagged cells", reason)
                      .drop(columns=["condition", "label_role", "analysis"]))
        contrasts.append(lib.na_rows(cond, without, "ProLIF contrasts on mito metrics", reason)
                         .drop(columns=["condition", "label_role", "analysis"]))
    return {"annotation": annotation, "frame": frame, "thresholds": thr, "counts": cnt, "plate_counts": pcnt,
            "summary": summaries, "enrich": pd.concat(enrich, ignore_index=True),
            "contrasts": pd.concat(contrasts, ignore_index=True), "task1": task1_table(frame, annotation, cond),
            "checks": checks, "counts_sets": counts, "umap": np.asarray(adata.obsm["X_umap"]), "labels": labels}


# ---------------------------------------------------------------------------
# Figures (dotplot format; Ctrl and ZeroK side by side)
# ---------------------------------------------------------------------------
def label_handles(order, extra=()) -> list:
    colors = lib.label_colors(order)
    return [Line2D([], [], marker="o", linestyle="", color=colors[l], markersize=4, label=l) for l in order] + list(extra)


def column_for(cond: str, key: str) -> str:
    return "plate" if key == "plate" else lib.SETTINGS[cond].COLUMN_OF[key]


def fig_scatter(res: dict, role: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 150)
    gs = fig.add_gridspec(len(SCATTER_METRICS), 2, left=0.09, right=0.98, top=0.86, bottom=0.08, hspace=0.45,
                          wspace=0.12)
    order = lib.union_order(*(res[c]["frame"][column_for(c, role)] for c in lib.CONDITIONS))
    colors = lib.label_colors(order)
    for i, metric in enumerate(SCATTER_METRICS):
        first = None
        for j, cond in enumerate(lib.CONDITIONS):
            frame, col = res[cond]["frame"], column_for(cond, role)
            ax = fig.add_subplot(gs[i, j], sharey=first, sharex=first)
            first = first or ax
            for lab in order:
                m = (frame[col].astype(str) == lab).to_numpy()
                ax.scatter(frame.loc[m, "n_detected_proteins"], frame.loc[m, metric], s=2.5, color=colors[lab],
                           linewidths=0, rasterized=True, alpha=0.8)
            rho = t3.spearman_ci(frame["n_detected_proteins"], frame[metric])
            ax.set_title(f"{cond}: Spearman rho {rho['rho']:.2f} (all cells)", fontsize=7)
            lib.grid_y(ax)
            if j == 0:
                ax.set_ylabel(SHORT_METRIC[metric])
            else:
                ax.tick_params(labelleft=False)
            if i == len(SCATTER_METRICS) - 1:
                ax.set_xlabel("Protein IDs per cell")
    fig.legend(handles=label_handles(order), loc="upper right", bbox_to_anchor=(0.99, 0.945), ncol=len(order),
               frameon=False, handletextpad=0.2, columnspacing=1.0)
    fig.text(0.005, 0.995, f"Mitochondrial fractions against protein IDs per cell, by {lib.ROLE_DISPLAY[role]}\n"
             "(intensity = raw PG.Quantity; MitoCarta3.0 symbol or UniProt match; each condition on its own)",
             ha="left", va="top", fontsize=7)
    figs.save(fig, f"mito_vs_ids_{role}", f"Mito fractions vs protein IDs by {role} label, Ctrl and ZeroK",
              label_role=role)


def fig_box(res: dict, key: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 128)
    gs = fig.add_gridspec(2, 4, left=0.075, right=0.99, top=0.84, bottom=0.12, hspace=0.75, wspace=0.12)
    order = lib.union_order(*(res[c]["frame"][column_for(c, key)] for c in lib.CONDITIONS),
                            order=("P1", "P2") if key == "plate" else lib.LABEL_ORDER)
    colors = (dict(cc.PLATE_PALETTE) if key == "plate" else lib.label_colors(order))
    ring = "primary" if key == "plate" else key
    for k, metric in enumerate(BOX_METRICS):
        r, p = divmod(k, 2)
        first = None
        for j, cond in enumerate(lib.CONDITIONS):
            frame, col = res[cond]["frame"], column_for(cond, key)
            ax = fig.add_subplot(gs[r, 2 * p + j], sharey=first)
            first = first or ax
            values = frame[col].astype(str).to_numpy()
            groups = {lab: frame.loc[values == lab, metric].to_numpy(float) for lab in order}
            for i, lab in enumerate(order):
                v = groups[lab][np.isfinite(groups[lab])]
                if v.size:
                    bp = ax.boxplot(v, positions=[i], widths=0.6, showfliers=False, patch_artist=True)
                    bp["boxes"][0].set_facecolor(colors.get(lab, "0.6"))
                    bp["boxes"][0].set_alpha(0.75)
                    for part in ("whiskers", "caps"):
                        for line in bp[part]:
                            line.set_color(t2.INK["secondary"])
                    bp["medians"][0].set_color(t2.INK["primary"])
            high = frame[f"flag__{lib.SHORT_ROLE.get(ring, ring)}__mito_high"].to_numpy(bool)
            rng = np.random.default_rng(lib.SEED)
            for i, lab in enumerate(order):
                m = (values == lab) & high
                if m.any():
                    ax.scatter(i + rng.uniform(-0.18, 0.18, m.sum()), frame.loc[m, metric], s=7, facecolors="none",
                               edgecolors=t2.INK["primary"], linewidths=0.5, zorder=3)
            n = pd.Series(values).value_counts()
            ax.set_xticks(range(len(order)), [f"{l if l != 'Unknown' else 'Unk.'}\n{int(n.get(l, 0))}" for l in order])
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.set_title(f"{SHORT_METRIC[metric]}: {cond}", fontsize=7)
            lib.grid_y(ax)
            if j == 1:
                ax.tick_params(labelleft=False)
            elif p == 0:
                ax.set_ylabel("fraction")
    what = "plate (pooled sort plate)" if key == "plate" else lib.ROLE_DISPLAY[key]
    fig.text(0.005, 0.995, f"Mitochondrial metrics by {what} (box: median and IQR, whiskers 1.5 IQR; tick: label and "
             "cells)\nringed = high outlier on any mito metric (3 x MAD within the "
             f"{'primary' if key == 'plate' else key} label, labels with >= {t3.MIN_GROUP} cells; no cell excluded); "
             "Unk. = Unknown", ha="left", va="top", fontsize=7)
    figs.save(fig, f"mito_by_{key}", f"Mito metrics by {key}, Ctrl and ZeroK", label_role=key)


def fig_umap(res: dict, figs: lib.Figures) -> None:
    panels = [("primary", None), ("mitocarta_intensity_fraction", SHORT_METRIC["mitocarta_intensity_fraction"]),
              ("oxphos_intensity_fraction", SHORT_METRIC["oxphos_intensity_fraction"]),
              ("log10_n_detected", "log10 protein IDs")]
    fig = lib.new_figure(lib.FULL_MM, 96)
    gs = fig.add_gridspec(2, 4, left=0.04, right=0.955, top=0.87, bottom=0.1, hspace=0.35, wspace=0.4)
    for r, cond in enumerate(lib.CONDITIONS):
        frame, xy = res[cond]["frame"], res[cond]["umap"]
        for c, (key, title) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            if title is None:
                col = column_for(cond, key)
                values = frame[col].astype(str).to_numpy()
                for lab in cc.ordered_labels(values):
                    m = values == lab
                    ax.scatter(xy[m, 0], xy[m, 1], s=1.2, color=t2.LABEL_PALETTE.get(lab, "0.5"), linewidths=0,
                               rasterized=True)
                title = "primary label"
            else:
                v = frame[key].to_numpy(float)
                lo, hi = np.nanpercentile(v, [1, 99])
                sca = ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(v, lo, hi), cmap="viridis", s=1.2, linewidths=0,
                                 rasterized=True)
                cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.02)
                cb.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                cb.ax.tick_params(labelsize=7, length=2)
            t2.clean_axes(ax)
            ax.set_title(title, fontsize=7)
            if c == 0:
                ax.set_ylabel(cond, fontsize=7)
    order = lib.union_order(*(res[c]["frame"][column_for(c, "primary")] for c in lib.CONDITIONS))
    fig.legend(handles=label_handles(order), loc="lower center", ncol=len(order), frameon=False,
               bbox_to_anchor=(0.5, 0.005), handletextpad=0.2, columnspacing=1.2)
    fig.text(0.005, 0.995, "Mitochondrial metrics on each condition's dct_labels UMAP (colour range 1st-99th percentile "
             "within condition)", ha="left", va="top", fontsize=7)
    figs.save(fig, "mito_umap", "Mito metrics on the dct_labels UMAP, Ctrl and ZeroK")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "mito_qc", args)
    lib.ported_from(run, "ctrl_reviewer_core_mito_qc.py")
    mitocarta = t3.read_mitocarta(run)
    t26flags = pd.read_csv(run.use_input(DCT_INTAKE_FLAGS, "dct_intake MitoCarta flags (cross-check)"), keep_default_na=False)
    res = {c: one_condition(run, c, mitocarta, t26flags) for c in lib.CONDITIONS}
    checks = lib.check_rows_frame([r for c in lib.CONDITIONS for r in res[c]["checks"]])
    lib.stop_if_failed(run, checks, "mito_qc_checks")
    for cond in lib.CONDITIONS:
        run.note(f"{cond}: MitoCarta annotation of {len(res[cond]['annotation'])} proteins: {res[cond]['counts_sets']}; "
                 "equal to the dct_intake flags per protein.")

    def stacked(key: str, generic: bool = False) -> pd.DataFrame:
        frames = {}
        for c in lib.CONDITIONS:
            f = res[c][key]
            if generic:
                f = lib.to_generic(f.reset_index(), c)
            frames[c] = f
        return lib.stack(frames)

    run.table(lib.stack({c: res[c]["annotation"].reset_index() for c in lib.CONDITIONS}), "protein_annotation")
    run.table(stacked("frame", generic=True), "mito_qc_per_cell")
    run.table(stacked("summary"), "mito_qc_summary_by_label")
    run.table(stacked("thresholds"), "mito_outlier_thresholds")
    run.table(stacked("counts"), "mito_outlier_counts")
    run.table(stacked("plate_counts"), "mito_outlier_counts_by_plate")
    run.table(stacked("enrich"), "mito_outlier_prolif_enrichment")
    run.table(stacked("contrasts"), "mito_prolif_contrasts")
    run.table(stacked("task1"), "mito_task1_comparison")
    run.note("Per-cell table: generic columns primary_label / second_label; flag columns flag__{primary,second}__<kind> "
             "and outlier__{primary,second}__<metric>; evaluated__<role> = label has >= 10 cells.")
    run.note("ProLIF enrichment among flagged cells and ProLIF contrasts: tested on the second columns only (no ProLIF "
             "cell under either primary column; rows with status 'does not apply'). BH within condition x label column.")

    figs = lib.Figures(run, "mito_qc")
    for role in lib.ROLES:
        fig_scatter(res, role, figs)
    for key in (*lib.ROLES, "plate"):
        fig_box(res, key, figs)
    fig_umap(res, figs)
    figs.finish()
    run.finish({"min_group": t3.MIN_GROUP, "n_mad": t3.N_MAD, "mad_scale": "normal (1.4826)",
                "outlier_metrics": list(t3.OUTLIER_METRICS), "mito_flag_metrics": list(t3.MITO_METRICS),
                "depth_flag_metrics": list(t3.DEPTH_METRICS), "protein_sets": t3.SETS,
                "annotation_counts": {c: res[c]["counts_sets"] for c in lib.CONDITIONS},
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_mito_qc.csv.")


if __name__ == "__main__":
    main()
