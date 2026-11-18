#!/usr/bin/env python
"""dct_reviewer step 6b: reviewer_core B.4-B.5 depth- and plate-regressed embeddings and a high-coverage subset per condition.

Port of ctrl_reviewer_core_depth_embeddings.py (unchanged
method), run once per condition; ``regressed`` and ``pc_diagnostics`` are the ctrl_reviewer_core functions (unchanged).
The kNN purity figure uses ``plot_purity`` of dct_reviewer_cell_cycle.py.

Inputs (read-only): results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_task3_scored.h5ad (step 1; scaled .X,
cell_cycle_score, stored dct_labels graph and X_pca), results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_cc_regressed.h5ad
(step 2), dct_reviewer_task4_mki67_top2a_correlated_proteins.csv (step 2), dct_labels labels.
Outputs:
  results/tables/dct_reviewer/dct_reviewer_embed_*.csv (both conditions; ``condition`` column)
  results/figures/dct_reviewer/dct_reviewer_embed_umap_<object>.pdf/.png/.tif
  results/figures/dct_reviewer/dct_reviewer_embed_knn_label_purity.pdf/.png/.tif
  results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_{depth_regressed,plate_regressed,depth_cc_regressed,highcov50}.h5ad

Objects (each: regress_out on the scaled .X where stated, rescale (clip 10), arpack PCA, neighbours
(10, 20 PCs), Leiden 3.0, UMAP; seed 42): depth_regressed (ln IDs), plate_regressed (plate),
depth_cc_regressed (ln IDs + cell_cycle_score), highcov50 (proteins detected in >= 50% of the
condition's cells; cell-cycle scores recomputed on the subset). Leiden 0.3-3.0; ARI/NMI against both
label columns; ProLIF concentration ('applicable' False without ProLIF); kNN purity (k = 10, 20 PCs,
1,000 permutations). Uncorrected (stored dct_labels graph) and cc_regressed objects included for comparison.

Command (project root):  python scripts/dct_reviewer_depth_embeddings.py
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
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import ctrl_reviewer_core_depth_embeddings as m19
import dct_reviewer_cell_cycle as c28
import dct_reviewer_lib as lib

ensure_scanpy_compat()
lib.style()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SCRIPT = "dct_reviewer_depth_embeddings.py"
NEW_OBJECTS = m19.NEW_OBJECTS
ALL_OBJECTS = {"uncorrected": "stored dct_labels graph", "cc_regressed": "cell_cycle_score regressed out (A.2)", **NEW_OBJECTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 6b: depth/plate-regressed and high-coverage embeddings.")
    parser.add_argument("--min-detection", type=float, default=0.50)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=1000)
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str, args) -> dict:
    s = lib.SETTINGS[cond]
    adata = lib.read_h5ad(run, f"DCT_{s.KEY}_task3_scored")
    cc_reg = lib.read_h5ad(run, f"DCT_{s.KEY}_cc_regressed")
    _, labels = lib.load(run, cond)
    if list(adata.obs_names) != list(labels.index) or list(cc_reg.obs_names) != list(labels.index):
        raise ValueError(f"{cond}: cell order differs between the dct_reviewer objects and the canonical labels.")
    if cc_reg.X is None:
        raise ValueError(f"{cond}: cc_regressed object has no .X.")
    adata.obs["ln_n_detected"] = np.log(labels["n_detected_proteins"].to_numpy(float))
    adata.obs["plate"] = pd.Categorical(labels["plate"].to_numpy(), categories=["P1", "P2"])
    notes, out = [], {"labels": labels}

    objects = {"uncorrected": adata, "cc_regressed": cc_reg}
    objects["depth_regressed"] = m19.regressed(adata, ["ln_n_detected"], "depth_regressed")
    objects["plate_regressed"] = m19.regressed(adata, "plate", "plate_regressed")
    objects["depth_cc_regressed"] = m19.regressed(adata, ["ln_n_detected", "cell_cycle_score"], "depth_cc_regressed")

    detected_frac = (cc.dense(adata.layers["intensity"]) > 0).mean(axis=0)
    keep = adata.var_names[detected_frac >= args.min_detection]
    high = adata[:, keep].copy()
    present, missing = t2.score(high, t2.CELL_CYCLE_MARKERS, "cell_cycle_score_highcov50")
    if not present:
        high.obs["cell_cycle_score_highcov50"] = np.nan
    notes.append(f"{cond}: high-coverage subset {len(keep)} of {adata.n_vars} proteins detected in >= "
                 f"{args.min_detection:.0%} of cells; 18-protein panel present in the subset: {';'.join(present) or 'none'}.")
    _, s_hc = t2.match_genes(t2.TIROSH_S, high.var_names, "S")
    _, g_hc = t2.match_genes(t2.TIROSH_G2M, high.var_names, "G2M")
    alt_sets = {"cell_cycle_score_highcov50": present}
    if s_hc and g_hc:
        sc.tl.score_genes_cell_cycle(high, s_genes=s_hc, g2m_genes=g_hc, random_state=t2.SEED, use_raw=False)
        high.obs = high.obs.rename(columns={"S_score": "S_score_highcov50", "G2M_score": "G2M_score_highcov50",
                                            "phase": "phase_highcov50"})
        alt_sets.update(S_score_highcov50=s_hc, G2M_score_highcov50=g_hc)
    corr = lib.read_table(run, "task4_mki67_top2a_correlated_proteins")
    corr = corr[corr["condition"] == cond]
    corr_hc = [p for p in high.var_names if p in set(corr.loc[corr["selected"], "protein"])]
    if corr_hc:
        t2.score(high, corr_hc, "cc_corr_score_highcov50")
        alt_sets["cc_corr_score_highcov50"] = corr_hc
    notes.append(f"{cond}: within the subset Tirosh S present {len(s_hc)}, G2M present {len(g_hc)}; "
                 f"{len(corr_hc)} proteins of the circular Mki67/Top2a-correlated set.")
    objects["highcov50"] = t2.recluster(high, "leiden_highcov50_res_3.00", t2.ORIGINAL_RESOLUTION)
    cov = pd.DataFrame({"protein": adata.var_names, "fraction_detected": detected_frac,
                        "in_highcov50": detected_frac >= args.min_detection,
                        "in_original_18_cell_cycle_panel": adata.var_names.isin(t2.CELL_CYCLE_MARKERS)})
    out["coverage"] = cov[cov["in_highcov50"] | cov["in_original_18_cell_cycle_panel"]]
    out["n_highcov"], out["present"], out["missing_cc"] = int(len(keep)), present, missing

    out["defs"] = pd.DataFrame([{"object": name, "description": ALL_OBJECTS[name], "n_proteins": obj.n_vars,
                                 "regressed_keys": {"depth_regressed": "ln_n_detected", "plate_regressed": "plate",
                                                    "depth_cc_regressed": "ln_n_detected;cell_cycle_score",
                                                    "cc_regressed": "cell_cycle_score"}.get(name, "")}
                                for name, obj in objects.items()])

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
    out["clusters"] = clusters.reset_index()
    out["ari"] = t2.ari_nmi_rows(clusters, labels, s.LABEL_COLUMNS)
    conc = t2.concentration_rows(clusters, labels, s.LABEL_COLUMNS)
    conc["applicable"] = conc["n_prolif"] > 0
    out["conc"] = conc
    embeddings = {name: obj.obsm["X_pca"][:, :t2.N_PCS] for name, obj in objects.items()}
    out["purity"] = t2.knn_purity(embeddings, labels, s.LABEL_COLUMNS, args.knn_k, args.n_permutations)
    out["pcs"] = m19.pc_diagnostics(objects, labels)

    rows = []
    reference_score = f"cell_cycle_score ({adata.n_vars:,} proteins; step 1)"
    cc_frame = pd.DataFrame({reference_score: adata.obs["cell_cycle_score"].to_numpy(float)}, index=labels.index)
    panels = {reference_score: [g for g in t2.CELL_CYCLE_MARKERS if g in adata.var_names]}
    for name, genes in alt_sets.items():
        cc_frame[name] = high.obs[name].to_numpy(float)
        panels[name] = genes
    for score_col in cc_frame.columns:
        circular = "circular Mki67/Top2a-correlated set" if score_col.startswith("cc_corr") else ""
        for col in s.LABEL_COLUMNS:
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
                a = cc_frame.loc[labels[col] == group, score_col]
                b = cc_frame.loc[labels[col] == ref, score_col]
                res = cc.compare_groups(a, b)
                absent = [l for l, v in ((group, a), (ref, b)) if v.empty]
                rows.append({"score": score_col, "label_column": col, "label": f"{group}_vs_{ref}",
                             "median_difference": res["median_difference"], "rank_biserial": res["rank_biserial"],
                             "p_value": res["p_value"], "n_panel_proteins": len(panels[score_col]),
                             "note": (f"{lib.NOT_APPLICABLE}: no {' or '.join(absent)} cell under {col}" if absent
                                      else circular)})
    out["cc_by_label"] = pd.DataFrame(rows)

    for name in NEW_OBJECTS:
        obj = objects[name]
        for col in ("plate", *s.LABEL_COLUMNS):
            obj.obs[col] = pd.Categorical(labels[col].to_numpy())
        lib.write_h5ad_light(run, obj, f"DCT_{s.KEY}_{name}")
    out["umaps"] = {name: objects[name].obsm["X_umap"] for name in NEW_OBJECTS}
    out["n_vars"] = {name: objects[name].n_vars for name in NEW_OBJECTS}
    out["notes"] = notes
    return out


def plot_umaps(res: dict, name: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 92)
    gs = fig.add_gridspec(2, 4, left=0.04, right=0.955, top=0.9, bottom=0.1, hspace=0.35, wspace=0.35)
    for r, cond in enumerate(lib.CONDITIONS):
        s, d = lib.SETTINGS[cond], res[cond]
        labels, umap = d["labels"], d["umaps"][name]
        lx = np.log(labels["n_detected_proteins"].to_numpy(float))
        for c, (col, kind) in enumerate((("primary", "cat"), ("second", "cat"), ("plate", "plate"), ("ln_ids", "num"))):
            ax = fig.add_subplot(gs[r, c])
            if kind == "num":
                sca = ax.scatter(umap[:, 0], umap[:, 1], c=lx, cmap="viridis", s=1.2, linewidths=0, rasterized=True)
                cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.02)
                cb.ax.yaxis.set_major_locator(MaxNLocator(3))
                cb.ax.tick_params(labelsize=7, length=2)
                title = "ln(protein IDs)"
            else:
                values = labels[s.COLUMN_OF[col] if kind == "cat" else "plate"].astype(str).to_numpy()
                palette = cc.PLATE_PALETTE if kind == "plate" else t2.LABEL_PALETTE
                for cat in (["P1", "P2"] if kind == "plate" else cc.ordered_labels(values)):
                    m = values == cat
                    ax.scatter(umap[m, 0], umap[m, 1], s=1.2, color=palette.get(cat, "0.5"), linewidths=0,
                               rasterized=True)
                title = f"{col} label" if kind == "cat" else "plate"
            t2.clean_axes(ax)
            ax.set_title(title, fontsize=7)
            if c == 0:
                ax.set_ylabel(cond, fontsize=7)
    fig.text(0.005, 0.995, f"{name}: {NEW_OBJECTS[name]} (UMAP of each condition's own object; "
             f"proteins Ctrl {res['Ctrl']['n_vars'][name]}, ZeroK {res['ZeroK']['n_vars'][name]})", ha="left", va="top",
             fontsize=7)
    handles = [Line2D([], [], marker="o", linestyle="", color=t2.LABEL_PALETTE.get(l, "0.5"), markersize=4, label=l)
               for l in lib.LABEL_ORDER]
    handles += [Line2D([], [], marker="o", linestyle="", color=cc.PLATE_PALETTE[p], markersize=4, label=f"plate {p}")
                for p in ("P1", "P2")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.005),
               handletextpad=0.2, columnspacing=1.2)
    figs.save(fig, f"embed_umap_{name}", f"UMAP of the {name} object per condition")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "depth_embeddings", args)
    lib.ported_from(run, "ctrl_reviewer_core_depth_embeddings.py")
    res = {c: one_condition(run, c, args) for c in lib.CONDITIONS}
    for cond in lib.CONDITIONS:
        for n in res[cond]["notes"]:
            run.note(n)
    for key, name in (("coverage", "embed_highcov50_protein_coverage"), ("defs", "embed_object_definitions"),
                      ("clusters", "embed_clusters_per_cell"), ("ari", "embed_ari_nmi_clusters_vs_labels"),
                      ("conc", "embed_prolif_concentration"), ("purity", "embed_knn_label_purity"),
                      ("pcs", "embed_pc_vs_depth_and_plate"), ("cc_by_label", "embed_highcov50_cell_cycle_score_by_label")):
        run.table(lib.stack({c: res[c][key] for c in lib.CONDITIONS}), name)
    figs = lib.Figures(run, "depth_embeddings")
    for name in NEW_OBJECTS:
        plot_umaps(res, name, figs)
    c28.plot_purity({c: res[c]["purity"] for c in lib.CONDITIONS}, "embed_knn_label_purity", list(ALL_OBJECTS),
                    "kNN label purity, depth/plate-regressed and high-coverage objects",
                    figs, height_mm=150, bottom=0.25)
    figs.finish()
    run.finish({"highcov_min_detection": args.min_detection,
                "n_highcov_proteins": {c: res[c]["n_highcov"] for c in lib.CONDITIONS},
                "highcov_cell_cycle_present": {c: res[c]["present"] for c in lib.CONDITIONS},
                "highcov_cell_cycle_missing": {c: res[c]["missing_cc"] for c in lib.CONDITIONS},
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_depth_embeddings.csv.")


if __name__ == "__main__":
    main()
