#!/usr/bin/env python
"""reviewer_core B.4-B.5: depth- and plate-regressed embeddings, and a high-coverage protein subset.



Inputs (read-only): results/h5ad/reviewer_core_DCT_task3_scored.h5ad (A.1; scaled
V1 .X, cell_cycle_score, stored V1 graph and X_pca),
results/h5ad/reviewer_core_DCT_cc_regressed.h5ad (A.2, for side-by-side rows),
kidney_object canonical labels.
Outputs:
  results/tables/reviewer_core/reviewer_core_embed_*.csv
  results/figures/reviewer_core/reviewer_core_embed_*.pdf/.png
  results/h5ad/reviewer_core_DCT_{depth_regressed,plate_regressed,depth_cc_regressed,highcov50}.h5ad

Objects (each: regress_out on the scaled .X where stated, then rescale (clip
10), arpack PCA, neighbours (10, 20 PCs), Leiden 3.0, UMAP; seed 42):
  depth_regressed     ln(protein IDs per cell) regressed out
  plate_regressed     plate (P1/P2, categorical) regressed out
  depth_cc_regressed  ln(IDs) and cell_cycle_score regressed out together
  highcov50           only proteins detected in >= 50% of the 750 cells; the
                      18-protein cell-cycle score is recomputed on this subset
Leiden at 0.3, 0.5, 0.8, 1.0 and 3.0; ARI/NMI against both V1 label columns;
ProLIF concentration; kNN purity (k = 10, 20 PCs, 1,000 permutations). The
uncorrected (stored V1) and cc_regressed objects are included for comparison.
Diagnostic: Spearman rho of PC1-PC5 with ln(IDs) and with plate.

Command (project root):  python scripts/reviewer_core_depth_embeddings.py
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
import scanpy as sc
from scipy import stats
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2

ensure_scanpy_compat()
t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "reviewer_core_depth_embeddings.py"
NEW_OBJECTS = {
    "depth_regressed": "ln(protein IDs) regressed out",
    "plate_regressed": "plate regressed out",
    "depth_cc_regressed": "ln(protein IDs) + cell_cycle_score regressed out",
    "highcov50": "proteins detected in >= 50% of cells only",
}
ALL_OBJECTS = {"uncorrected": "stored V1 graph", "cc_regressed": "cell_cycle_score regressed out (A.2)", **NEW_OBJECTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reviewer_core B.4-B.5: depth/plate-regressed and high-coverage embeddings.")
    parser.add_argument("--min-detection", type=float, default=0.50)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=1000)
    cc.add_common_args(parser)
    return parser.parse_args()


def regressed(adata, keys, name: str):
    out = adata.copy()
    sc.pp.regress_out(out, keys=keys, n_jobs=1)
    return t2.recluster(out, f"leiden_{name}_res_3.00", t2.ORIGINAL_RESOLUTION)


def pc_diagnostics(objects: dict, labels: pd.DataFrame) -> pd.DataFrame:
    lx = np.log(labels["n_detected_proteins"].to_numpy(float))
    plate = (labels["plate"] == "P2").astype(float).to_numpy()
    rows = []
    for name, obj in objects.items():
        pcs = obj.obsm["X_pca"]
        for i in range(5):
            rho_d, p_d = stats.spearmanr(pcs[:, i], lx)
            rho_p, p_p = stats.spearmanr(pcs[:, i], plate)
            rows.append({"object": name, "pc": i + 1, "spearman_rho_ln_ids": rho_d, "p_ln_ids": p_d,
                         "spearman_rho_plate_P2": rho_p, "p_plate": p_p})
    return pd.DataFrame(rows)


def plot_umaps(objects: dict, labels: pd.DataFrame, run: cc.Run) -> None:
    names = list(NEW_OBJECTS)
    lx = np.log(labels["n_detected_proteins"].to_numpy(float))
    cols = [("v1_cell_label", "cat"), ("v1_cluster_label", "cat"), ("plate", "plate"), ("ln_ids", "num")]
    fig, axes = plt.subplots(len(names), len(cols), figsize=(2.0 * len(cols) + 0.6, 2.0 * len(names)))
    for r, name in enumerate(names):
        umap = objects[name].obsm["X_umap"]
        for c, (col, kind) in enumerate(cols):
            ax = axes[r, c]
            if kind == "num":
                sca = ax.scatter(umap[:, 0], umap[:, 1], c=lx, cmap="viridis", s=1.8, linewidths=0, rasterized=True)
                cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=5)
                cb.set_label("ln(IDs)", fontsize=5.5)
            else:
                values = labels[col].astype(str).to_numpy()
                palette = cc.PLATE_PALETTE if kind == "plate" else t2.LABEL_PALETTE
                for cat in (["P1", "P2"] if kind == "plate" else cc.ordered_labels(values)):
                    m = values == cat
                    ax.scatter(umap[m, 0], umap[m, 1], s=1.8, color=palette.get(cat, "0.5"), linewidths=0,
                               label=f"{cat} ({m.sum()})", rasterized=True)
                if r == 0:
                    ax.legend(fontsize=4.8, markerscale=3, loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2,
                              handletextpad=0.1, columnspacing=0.5)
            t2.clean_axes(ax)
            if c == 0:
                ax.set_ylabel(f"{name}\n{objects[name].n_vars} proteins", fontsize=6)
            if r == 0:
                ax.set_title({"v1_cell_label": "V1 per-cell label", "v1_cluster_label": "V1 cluster label",
                              "plate": "plate", "ln_ids": "ln(protein IDs)"}[col], fontsize=6.8, pad=22)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "embed_umaps", "UMAPs of the depth-, plate- and depth+cell-cycle-regressed and high-coverage objects")


def plot_purity(table: pd.DataFrame, run: cc.Run) -> None:
    objects = list(ALL_OBJECTS)
    fig, axes = plt.subplots(1, len(t2.LABEL_COLUMNS), figsize=(3.6 * len(t2.LABEL_COLUMNS), 2.9), sharey=True)
    for ax, col in zip(axes, t2.LABEL_COLUMNS):
        sub = table[(table["label_column"] == col) & table["label"].isin(t2.CORE_LABELS)]
        for offset, label in zip((-0.22, 0, 0.22), t2.CORE_LABELS):
            part = sub[sub["label"] == label].set_index("object").reindex(objects)
            x = np.arange(len(objects)) + offset
            ax.vlines(x, part["null_q025"], part["null_q975"], color=t2.INK["background_cells"], linewidth=3.2)
            ax.scatter(x, part["observed_same_label_fraction"], color=t2.LABEL_PALETTE[label], s=14, zorder=3,
                       label=label, edgecolors="white", linewidths=0.4)
        ax.set_xticks(range(len(objects)), objects, rotation=35, ha="right")
        ax.set_title(f"kNN label purity, {t2.LABEL_DISPLAY[col]}", fontsize=7)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean same-label fraction of 10 neighbours")
    axes[0].legend(title="dot = observed; grey bar = permutation 95%", title_fontsize=5.5, loc="upper left")
    fig.suptitle("kNN label purity after regressing depth, plate, depth + cell cycle, or keeping high-coverage proteins",
                 x=0.01, ha="left", fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "embed_knn_label_purity", "kNN label purity on depth/plate-corrected embeddings")


def main() -> None:
    args = parse_args()
    run = cc.Run(t2.TASK, "depth_embeddings", args)
    adata = t2.read_h5ad(run, "DCT_task3_scored")
    cc_reg = t2.read_h5ad(run, "DCT_cc_regressed")
    _, labels = t2.load_dct(run)
    if list(adata.obs_names) != list(labels.index) or list(cc_reg.obs_names) != list(labels.index):
        raise ValueError("Cell order differs between the reviewer_core objects and the canonical labels.")
    adata.obs["ln_n_detected"] = np.log(labels["n_detected_proteins"].to_numpy(float))
    adata.obs["plate"] = pd.Categorical(labels["plate"].to_numpy(), categories=["P1", "P2"])

    objects = {"uncorrected": adata, "cc_regressed": cc_reg}
    objects["depth_regressed"] = regressed(adata, ["ln_n_detected"], "depth_regressed")
    objects["plate_regressed"] = regressed(adata, "plate", "plate_regressed")
    objects["depth_cc_regressed"] = regressed(adata, ["ln_n_detected", "cell_cycle_score"], "depth_cc_regressed")

    # B.5 high-coverage subset.
    detected_frac = (cc.dense(adata.layers["intensity"]) > 0).mean(axis=0)
    keep = adata.var_names[detected_frac >= args.min_detection]
    high = adata[:, keep].copy()
    present, missing = t2.score(high, t2.CELL_CYCLE_MARKERS, "cell_cycle_score_highcov50")
    if not present:  # no panel protein in the subset: the score is undefined, not zero
        high.obs["cell_cycle_score_highcov50"] = np.nan
    run.note(f"High-coverage subset: {len(keep)} of {adata.n_vars} proteins detected in >= {args.min_detection:.0%} of cells; "
             f"18-protein cell-cycle panel present in the subset: {';'.join(present) or 'none'} (score set to NaN if none).")
    # Alternatives within the subset: Tirosh S/G2M (Seurat-style) and the circular Mki67/Top2a-correlated set.
    _, s_hc = t2.match_genes(t2.TIROSH_S, high.var_names, "S")
    _, g_hc = t2.match_genes(t2.TIROSH_G2M, high.var_names, "G2M")
    alt_sets = {"cell_cycle_score_highcov50": present}
    if s_hc and g_hc:
        sc.tl.score_genes_cell_cycle(high, s_genes=s_hc, g2m_genes=g_hc, random_state=t2.SEED, use_raw=False)
        high.obs = high.obs.rename(columns={"S_score": "S_score_highcov50", "G2M_score": "G2M_score_highcov50",
                                            "phase": "phase_highcov50"})
        alt_sets.update(S_score_highcov50=s_hc, G2M_score_highcov50=g_hc)
    corr = t2.read_table(run, "task4_mki67_top2a_correlated_proteins")
    corr_hc = [p for p in high.var_names if p in set(corr.loc[corr["selected"], "protein"])]
    if corr_hc:
        t2.score(high, corr_hc, "cc_corr_score_highcov50")
        alt_sets["cc_corr_score_highcov50"] = corr_hc
    run.note(f"Within the subset: Tirosh S present {s_hc}, G2M present {g_hc}; "
             f"{len(corr_hc)} proteins of the circular Mki67/Top2a-correlated set.")
    objects["highcov50"] = t2.recluster(high, "leiden_highcov50_res_3.00", t2.ORIGINAL_RESOLUTION)
    cov = pd.DataFrame({"protein": adata.var_names, "fraction_detected": detected_frac,
                        "in_highcov50": detected_frac >= args.min_detection,
                        "in_original_18_cell_cycle_panel": adata.var_names.isin(t2.CELL_CYCLE_MARKERS)})
    run.table(cov[cov["in_highcov50"] | cov["in_original_18_cell_cycle_panel"]], "embed_highcov50_protein_coverage")

    defs = pd.DataFrame([{"object": name, "description": ALL_OBJECTS[name], "n_proteins": obj.n_vars,
                          "regressed_keys": {"depth_regressed": "ln_n_detected", "plate_regressed": "plate",
                                             "depth_cc_regressed": "ln_n_detected;cell_cycle_score",
                                             "cc_regressed": "cell_cycle_score"}.get(name, "")}
                         for name, obj in objects.items()])
    run.table(defs, "embed_object_definitions")

    clusters = pd.DataFrame(index=labels.index)
    for name, obj in objects.items():
        for res in t2.RESOLUTIONS:
            key = f"leiden_r{res:.2f}"
            r3 = f"leiden_{name}_res_3.00"
            if name in NEW_OBJECTS and np.isclose(res, t2.ORIGINAL_RESOLUTION):
                obj.obs[key] = obj.obs[r3].astype(str).astype("category")
            elif key not in obj.obs or name == "uncorrected":
                t2.leiden(obj, res, key)
            clusters[f"{name}__r{res:.2f}"] = obj.obs[key].astype(str).to_numpy()
    run.table(clusters.reset_index(), "embed_clusters_per_cell")
    run.table(t2.ari_nmi_rows(clusters, labels, t2.LABEL_COLUMNS), "embed_ari_nmi_clusters_vs_labels")
    run.table(t2.concentration_rows(clusters, labels, t2.LABEL_COLUMNS), "embed_prolif_concentration")

    embeddings = {name: (obj.obsm["X_pca"][:, :t2.N_PCS]) for name, obj in objects.items()}
    purity = t2.knn_purity(embeddings, labels, t2.LABEL_COLUMNS, args.knn_k, args.n_permutations)
    run.table(purity, "embed_knn_label_purity")
    run.table(pc_diagnostics(objects, labels), "embed_pc_vs_depth_and_plate")

    rows = []
    reference_score = "cell_cycle_score (2,561 proteins; A.1)"
    cc_frame = pd.DataFrame({reference_score: adata.obs["cell_cycle_score"].to_numpy(float)}, index=labels.index)
    panels = {reference_score: [g for g in t2.CELL_CYCLE_MARKERS if g in adata.var_names]}
    for name, genes in alt_sets.items():
        cc_frame[name] = high.obs[name].to_numpy(float)
        panels[name] = genes
    for score_col in cc_frame.columns:
        circular = "circular Mki67/Top2a-correlated set" if score_col.startswith("cc_corr") else ""
        for col in t2.LABEL_COLUMNS:
            for label in cc.ordered_labels(labels[col]):
                v = cc_frame.loc[labels[col] == label, score_col]
                rows.append({"score": score_col, "label_column": col, "label": label, "n_cells": len(v),
                             "median": v.median(), "q25": v.quantile(0.25), "q75": v.quantile(0.75), "mean": v.mean(),
                             "n_panel_proteins": len(panels[score_col]),
                             "panel_present": ";".join(panels[score_col]) if len(panels[score_col]) <= 20
                             else f"{len(panels[score_col])} proteins", "note": circular or
                             ("no panel protein in the subset; score undefined" if not panels[score_col] else "")})
            if not panels[score_col]:
                continue
            for group, ref in (("ProLIF", "DCT1"), ("ProLIF", "DCT2")):
                res = cc.compare_groups(cc_frame.loc[labels[col] == group, score_col], cc_frame.loc[labels[col] == ref, score_col])
                rows.append({"score": score_col, "label_column": col, "label": f"{group}_vs_{ref}",
                             "median_difference": res["median_difference"], "rank_biserial": res["rank_biserial"],
                             "p_value": res["p_value"], "n_panel_proteins": len(panels[score_col]), "note": circular})
    run.table(pd.DataFrame(rows), "embed_highcov50_cell_cycle_score_by_label")

    plot_umaps(objects, labels, run)
    plot_purity(purity, run)
    for name in NEW_OBJECTS:
        obj = objects[name]
        for col in ("plate", *t2.LABEL_COLUMNS):
            obj.obs[col] = pd.Categorical(labels[col].to_numpy())
        t2.write_h5ad_light(run, obj, f"DCT_{name}")
    run.finish({"highcov_min_detection": args.min_detection, "n_highcov_proteins": int(len(keep)),
                "highcov_cell_cycle_present": present, "highcov_cell_cycle_missing": missing})


if __name__ == "__main__":
    main()
