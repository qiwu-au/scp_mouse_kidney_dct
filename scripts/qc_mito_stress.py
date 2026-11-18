#!/usr/bin/env python
"""qc_mito step 5: dissociation-stress panel coverage and per-cell scores, by label and against depth.



Inputs (read-only): kidney_object canonical objects and label tables.
Outputs (results/tables/qc_mito/, results/figures/qc_mito/):
  qc_mito_stress_coverage.csv               18-protein panel: present, protein group, detection
  qc_mito_stress_detection_by_label.csv     detection rate of each present panel protein per label
  qc_mito_{kidney,dct}_stress_per_cell.csv  scores per cell with labels, plate and depth
  qc_mito_stress_summary_by_label.csv
  qc_mito_stress_vs_depth.csv               Spearman with log10 IDs / log10 total intensity
  qc_mito_stress_type_tests.csv, _stress_type_coefficients.csv   across-label tests (with/without depth)
  qc_mito_stress_dct_prolif_contrasts.csv   ProLIF vs DCT1 / DCT2 / DCT1+DCT2 (MWU; OLS +/- depth, plate)
  qc_mito_stress_leave_one_out.csv          contrasts / type tests with each panel protein left out
  figures: qc_mito_stress_coverage, *_stress_by_*, *_stress_vs_depth_*, *_stress_umap,
           qc_mito_dct_stress_prolif_contrasts

Scores (panel proteins present in the matrix; a protein group shared by two
panel genes, e.g. Hspa1b;Hspa1a, counts once):
  stress_mean_z      mean of .X (per-protein z-scored ln(1 + intensity), clipped at 10) over the present proteins
  stress_score_genes scanpy score_genes on ln(1 + intensity) (ctrl_size 50, n_bins 25, random_state 42);
                     control proteins are binned on mean ln intensity (bins on the z-scored .X would be uninformative)
  stress_n_detected  number of present panel proteins with intensity > 0

Command (project root):  python scripts/qc_mito_stress.py
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

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "qc_mito_stress.py"
DEPTH_VARS = ("log10_n_detected", "log10_total_intensity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qc_mito step 5: dissociation-stress score.")
    cc.add_common_args(parser)
    return parser.parse_args()


def coverage(adata, dataset: str) -> tuple[pd.DataFrame, list[str]]:
    var = adata.var
    symbols = pd.Series([t3.split_symbols(v) for v in var["pg_genes"].astype(str)], index=var.index)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    rows, present = [], []
    for gene in t3.STRESS_PANEL:
        hits = [name for name, syms in zip(var.index.astype(str), symbols) if gene.lower() in syms]
        row = {"dataset": dataset, "gene": gene, "present": bool(hits), "matrix_protein": ";".join(hits)}
        if hits:
            name = hits[0]
            j = list(var.index.astype(str)).index(name)
            det = intensity[:, j] > 0
            shared = [g for g in t3.STRESS_PANEL if g != gene and g.lower() in symbols.iloc[j]]
            row.update(pg_genes=var["pg_genes"].iloc[j], protein_group=var["protein_group"].iloc[j],
                       shared_group_with=";".join(shared), n_cells=adata.n_obs, n_cells_detected=int(det.sum()),
                       percent_cells_detected=100 * det.mean(),
                       median_intensity_detected=float(np.median(intensity[det, j])) if det.any() else np.nan,
                       n_matches=len(hits))
            for h in hits:
                if h not in present:
                    present.append(h)
        rows.append(row)
    return pd.DataFrame(rows), present


def scores(adata, present: list[str]) -> pd.DataFrame:
    import anndata as ad
    import scanpy as sc

    names = list(adata.var_names.astype(str))
    idx = [names.index(p) for p in present]
    x = cc.dense(adata.X).astype(np.float64)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    out = pd.DataFrame(index=pd.Index(adata.obs_names.astype(str), name="cell_id"))
    out["stress_mean_z"] = x[:, idx].mean(axis=1)
    light = ad.AnnData(X=np.log1p(intensity), obs=pd.DataFrame(index=out.index), var=pd.DataFrame(index=names))
    sc.tl.score_genes(light, present, ctrl_size=50, n_bins=25, random_state=t3.SEED, use_raw=False,
                      score_name="stress_score_genes")
    out["stress_score_genes"] = light.obs["stress_score_genes"].to_numpy()
    out["stress_n_detected"] = (intensity[:, idx] > 0).sum(axis=1)
    out["stress_fraction_detected"] = out["stress_n_detected"] / len(idx)
    return out


def loo_scores(adata, present: list[str]) -> dict[str, np.ndarray]:
    names = list(adata.var_names.astype(str))
    x = cc.dense(adata.X).astype(np.float64)
    out = {}
    for drop in present:
        keep = [names.index(p) for p in present if p != drop]
        if keep:
            out[drop] = x[:, keep].mean(axis=1)
    return out


def depth_rows(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for col in t3.DATASETS[dataset]["label_columns"]:
        counts = frame[col].astype(str).value_counts()
        for label in ["all_cells", *t3.label_order(frame[col], dataset)]:
            mask = np.ones(len(frame), bool) if label == "all_cells" else (frame[col].astype(str) == label).to_numpy()
            for score in t3.STRESS_SCORES:
                for depth in DEPTH_VARS:
                    row = {"dataset": dataset, "label_column": col, "label": label, "score": score,
                           "depth_variable": depth, "n_cells": int(mask.sum())}
                    if label != "all_cells" and counts[label] < t3.MIN_GROUP:
                        row.update(rho=np.nan, note=f"not evaluated (n < {t3.MIN_GROUP})")
                    else:
                        row.update(t3.spearman_ci(frame.loc[mask, depth], frame.loc[mask, score]), note="")
                    rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_coverage(run, cov: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    genes = list(t3.STRESS_PANEL)
    y = np.arange(len(genes))
    for ax, dataset in zip(axes, ("kidney", "dct")):
        d = cov[cov["dataset"] == dataset].set_index("gene").loc[genes]
        vals = d["percent_cells_detected"].fillna(0).to_numpy(float)
        ax.barh(y, vals, height=0.65, color=t3.SERIES[0])
        for yi, (gene, row) in zip(y, d.iterrows()):
            if not row["present"]:
                text = "not in matrix"
            else:
                text = f"{row['percent_cells_detected']:.1f}%"
                if row.get("shared_group_with"):
                    text += f" (group {row['pg_genes']})"
            ax.text((vals[yi] if row["present"] else 0) + 1, yi, text, va="center", fontsize=5.2,
                    color=t3.INK["secondary"] if row["present"] else t3.INK["muted"])
        n_present = int(d["present"].sum())
        n_groups = len({p for p in d["matrix_protein"] if p})
        ax.set_title(f"{t3.DATASETS[dataset]['display']}: {n_present}/18 genes present ({n_groups} protein groups)",
                     fontsize=6.8)
        ax.set_xlim(0, 115)
        ax.set_xlabel("% of cells with intensity > 0")
        ax.xaxis.grid(True, color=t3.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, genes, fontsize=6)
    axes[0].invert_yaxis()
    fig.suptitle("Dissociation-stress panel: coverage in the protein matrices", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t3.footer(fig, SCRIPT, "match on any PG.Genes symbol (case-insensitive)")
    run.figure(fig, "stress_coverage", "Stress panel coverage")


def fig_box(run, frame, adata, dataset, col, n_present):
    order = t3.label_order(frame[col], dataset)
    colors = t3.label_colors(adata, col)
    width = 7.4 if len(order) > 6 else 5.6
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.6))
    for ax, score in zip(axes, t3.STRESS_SCORES):
        t3.box_strip(ax, frame, col, score, order, colors)
        ax.set_title(t3.STRESS_DISPLAY[score], fontsize=6.4)
    fig.suptitle(f"{t3.DATASETS[dataset]['display']}: stress scores by {t3.LABEL_DISPLAY[col]} "
                 f"({n_present} panel proteins present; n per label under each tick)", x=0.01, ha="left", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    t3.footer(fig, SCRIPT)
    run.figure(fig, f"{dataset}_stress_by_{t3.SHORT_COLUMN[col]}", f"{dataset}: stress scores by {col}")


def fig_depth(run, frame, adata, dataset, col):
    order = t3.label_order(frame[col], dataset)
    colors = t3.label_colors(adata, col)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    rng = np.random.default_rng(t3.SEED)
    for ax, score in zip(axes, t3.STRESS_SCORES):
        yv = frame[score].to_numpy(float)
        if score == "stress_n_detected":
            yv = yv + rng.uniform(-0.18, 0.18, len(yv))
        t3.scatter_by_label(ax, frame["n_detected_proteins"], yv, frame[col], order, colors)
        rho = t3.spearman_ci(frame["log10_n_detected"], frame[score])
        ax.set_xscale("log")
        ax.set_xlabel("Protein IDs per cell (log scale)")
        ax.set_ylabel(t3.STRESS_DISPLAY[score] + (" (jittered)" if score == "stress_n_detected" else ""), fontsize=6)
        ax.set_title(f"Spearman rho = {rho['rho']:.2f} (all cells)", fontsize=6.2, color=t3.INK["secondary"])
    t3.legend_with_counts(axes[-1], frame[col], order, colors, title=t3.LABEL_DISPLAY[col])
    fig.suptitle(f"{t3.DATASETS[dataset]['display']}: stress scores against depth, by {t3.LABEL_DISPLAY[col]}",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    t3.footer(fig, SCRIPT)
    run.figure(fig, f"{dataset}_stress_vs_depth_{t3.SHORT_COLUMN[col]}", f"{dataset}: stress vs depth by {col}")


def fig_umap(run, frame, adata, dataset):
    xy = np.asarray(adata.obsm["X_umap"])
    col = t3.DATASETS[dataset]["label_columns"][0]
    order = t3.label_order(frame[col], dataset)
    colors = t3.label_colors(adata, col)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2))
    t3.umap_categorical(axes[0, 0], xy, frame[col], order, colors, t3.LABEL_DISPLAY[col])
    t3.legend_with_counts(axes[0, 0], frame[col], order, colors, loc="upper left", bbox_to_anchor=(-0.02, -0.02),
                          ncol=4 if len(order) > 6 else len(order), columnspacing=0.6, handletextpad=0.2)
    for ax, score in zip(list(axes.flat)[1:], t3.STRESS_SCORES):
        t3.umap_continuous(fig, ax, xy, frame[score], t3.STRESS_DISPLAY[score])
    fig.suptitle(f"{t3.DATASETS[dataset]['display']}: stress scores on the canonical UMAP (1st-99th percentile)",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t3.footer(fig, SCRIPT)
    run.figure(fig, f"{dataset}_stress_umap", f"{dataset}: stress scores on UMAP")


def fig_contrasts(run, con: pd.DataFrame) -> None:
    models = list(t3.CONTRAST_MODELS)
    names = {"label_only": "label only", "label_plus_log10_ids": "+ log10 IDs",
             "label_plus_log10_ids_plus_plate": "+ log10 IDs + plate"}
    fig, axes = plt.subplots(1, len(t3.STRESS_SCORES), figsize=(7.4, 2.8), sharey=True)
    rows = con[["label_column", "comparison"]].drop_duplicates().reset_index(drop=True)
    y = np.arange(len(rows))
    for ax, score in zip(axes, t3.STRESS_SCORES):
        d = con[con["metric"] == score].set_index(["label_column", "comparison"])
        for k, model in enumerate(models):
            sub = d.loc[list(map(tuple, rows.to_numpy()))]
            yy = y + (k - 1) * 0.22
            ax.errorbar(sub[f"{model}_coef"], yy, xerr=[sub[f"{model}_coef"] - sub[f"{model}_ci_low"],
                                                        sub[f"{model}_ci_high"] - sub[f"{model}_coef"]],
                        fmt="o", markersize=2.8, color=t3.SERIES[k], elinewidth=0.7, capsize=0, label=names[model])
        ax.axvline(0, color=t3.INK["muted"], linewidth=0.6)
        ax.set_title(t3.STRESS_DISPLAY[score], fontsize=6.4)
        ax.set_xlabel("ProLIF minus reference (OLS, 95% CI)", fontsize=6)
        ax.xaxis.grid(True, color=t3.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [f"{c.replace('ProLIF_vs_', 'vs ')}\n{t3.LABEL_DISPLAY[l]}" for l, c in rows.to_numpy()],
                       fontsize=5.4)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=5.6, bbox_to_anchor=(0.995, 0.945))
    fig.suptitle("DCT: ProLIF vs DCT1 / DCT2 / DCT1+DCT2 on the stress scores, unadjusted and depth-adjusted",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    t3.footer(fig, SCRIPT, "HC3 standard errors; plate = pooled sort plate (technical)")
    run.figure(fig, "dct_stress_prolif_contrasts", "DCT ProLIF contrasts on stress scores")


def main() -> None:
    args = parse_args()
    run = cc.Run(t3.TASK, "stress", args)
    covs, frames, adatas, presents = [], {}, {}, {}
    det_rows, summaries, depth, tests, coefs, loo = [], [], [], [], [], []
    for dataset in ("kidney", "dct"):
        adata, labels = t3.load(run, dataset)
        cov, present = coverage(adata, dataset)
        covs.append(cov)
        if not present:
            raise ValueError(f"{dataset}: no stress panel protein in the matrix.")
        run.note(f"{dataset}: stress panel genes present {int(cov['present'].sum())}/18 in {len(present)} protein "
                 f"groups ({', '.join(present)}).")
        frame = labels[["plate", *t3.DATASETS[dataset]["label_columns"], "n_detected_proteins", "total_intensity"]].copy()
        frame["log10_n_detected"] = np.log10(frame["n_detected_proteins"])
        frame["log10_total_intensity"] = np.log10(frame["total_intensity"])
        frame = frame.join(scores(adata, present))
        intensity = cc.dense(adata.layers["intensity"])
        names = list(adata.var_names.astype(str))
        for col in t3.DATASETS[dataset]["plot_columns"]:
            summaries.append(t3.summarise(frame, col, t3.STRESS_SCORES, dataset))
            for label in t3.label_order(frame[col], dataset):
                mask = (frame[col].astype(str) == label).to_numpy()
                for p in present:
                    det = intensity[mask, names.index(p)] > 0
                    det_rows.append({"dataset": dataset, "label_column": col, "label": label, "matrix_protein": p,
                                     "n_cells": int(mask.sum()), "n_detected": int(det.sum()),
                                     "percent_detected": 100 * det.mean()})
        depth.append(depth_rows(frame, dataset))
        for col in t3.DATASETS[dataset]["label_columns"]:
            for score in t3.STRESS_SCORES:
                summary, coef = t3.type_tests(frame, col, score, t3.DATASETS[dataset]["reference_label"])
                tests.append({"dataset": dataset, "variant": "all present proteins", **summary})
                coef.insert(0, "dataset", dataset)
                coefs.append(coef)
        for drop, values in loo_scores(adata, present).items():
            tmp = frame.assign(stress_mean_z_loo=values)
            for col in t3.DATASETS[dataset]["label_columns"]:
                summary, _ = t3.type_tests(tmp, col, "stress_mean_z_loo", t3.DATASETS[dataset]["reference_label"])
                tests.append({"dataset": dataset, "variant": f"stress_mean_z without {drop}", **summary})
                if dataset == "dct":
                    c = t3.prolif_contrasts(tmp, col, ["stress_mean_z_loo"], contrasts=(("ProLIF", ("DCT1", "DCT2")),))
                    c.insert(0, "left_out_protein", drop)
                    loo.append(c)
        frames[dataset], adatas[dataset], presents[dataset] = frame, adata, present
        run.table(frame.reset_index(), f"{dataset}_stress_per_cell")

    cov = pd.concat(covs, ignore_index=True)
    run.table(cov, "stress_coverage")
    run.table(pd.DataFrame(det_rows), "stress_detection_by_label")
    run.table(pd.concat(summaries, ignore_index=True), "stress_summary_by_label")
    run.table(pd.concat(depth, ignore_index=True), "stress_vs_depth")
    tests = pd.DataFrame(tests)
    run.table(tests, "stress_type_tests")
    run.table(pd.concat(coefs, ignore_index=True), "stress_type_coefficients")
    con = pd.concat([t3.prolif_contrasts(frames["dct"], col, t3.STRESS_SCORES)
                     for col in t3.DATASETS["dct"]["label_columns"]], ignore_index=True)
    run.table(con, "stress_dct_prolif_contrasts")
    run.table(pd.concat(loo, ignore_index=True), "stress_leave_one_out")

    fig_coverage(run, cov)
    for dataset, frame in frames.items():
        for col in t3.DATASETS[dataset]["plot_columns"]:
            fig_box(run, frame, adatas[dataset], dataset, col, len(presents[dataset]))
            fig_depth(run, frame, adatas[dataset], dataset, col)
        fig_umap(run, frame, adatas[dataset], dataset)
    fig_contrasts(run, con)
    run.finish({"panel": t3.STRESS_PANEL, "present_proteins": presents,
                "score_genes": {"matrix": "ln(1 + intensity)", "ctrl_size": 50, "n_bins": 25, "random_state": t3.SEED},
                "mean_z_matrix": ".X of the kidney_object canonical object (z-scored ln(1 + intensity), clipped at 10)",
                "min_group": t3.MIN_GROUP})


if __name__ == "__main__":
    main()
