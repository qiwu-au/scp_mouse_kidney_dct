#!/usr/bin/env python
"""kidney_annotation step 2, plan step 1: neighbour graphs for the kidney resolution scan, and their depth association.

Plan:
the analysis plan (settings agreed with the authors, 14 Oct 2026).

Graphs (k = 10, 20 PCs, PCA arpack, seed 42):
  (a) stored     - the stored graph of the kidney_object canonical kidney object (submitted pipeline);
  (b) regressed  - ln(protein IDs) regressed out of ln(1 + intensity) per protein (OLS with intercept,
                   missing = 0), z-scored and clipped at 10, PCA, kNN;
  (c) t04_total_unscaled_hvg500 - the least depth-associated preprocessing kidney pipeline, rebuilt with preprocessing_lib.
For each graph: Spearman rho of PC1-PC5 with ln(protein IDs) and ln(total intensity).
Also recomputes the stored Leiden 2.0 depth facts quoted in the plan (eta^2 of protein IDs on cluster,
Kruskal-Wallis p, median IDs per cluster); the script stops if they are not reproduced.

Inputs (read-only): kidney_object canonical kidney object and labels; preprocessing kidney PC-depth table and per-cell clusters
  (reproduction checks).
Outputs:
  results/h5ad/kidney_annotation_kidney_scan_graphs.h5ad (PCs, connectivities and distances of the three graphs)
  results/tables/kidney_annotation/kidney_annotation_graph_*.csv, kidney_annotation_stored_leiden_depth*.csv,
  kidney_annotation_graphs_checks.csv

Command (project root):  python scripts/kidney_annotation_graphs.py
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
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import preprocessing_lib as t4
import cluster_de_lib as t5
import kidney_annotation_scan_lib as s

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
TOL = {"matrix": 1e-4, "rho": 1e-3, "resid_r": 1e-8, "regress_out": 1e-3, "eta2": 5e-4}
# Plan, 'Facts already established' (read-only check on the kidney_object canonical labels).
BRIEF_STORED = {
    "eta2": 0.884, "kruskal_p_1sig": "3e-120", "n_clusters": 24, "size_min": 8, "size_max": 58,
    "clusters_per_pre_label": {"PCT": 6, "TAL": 5, "PST": 3, "DVR": 3, "CNT": 2, "JG": 1, "MD": 1, "DCT2": 1,
                               "IC-B": 1, "Unknown": 1},
    "median_ids_range": {"PCT": (224, 1175), "TAL": (138, 1104)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation step 2 / plan step 1: graphs and depth association.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "graphs_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_annotation_graphs_checks.csv (nothing else written).")


def stored_leiden_facts(checks: t5.Checks, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = labels["n_detected_proteins"].to_numpy(float)
    cl = labels["leiden_res_2.00"].astype(str).to_numpy()
    rows = []
    for c in t5.sort_clusters(cl):
        m = cl == c
        row = {"cluster": c, "n_cells": int(m.sum()), "median_protein_ids": float(np.median(ids[m])),
               "min_protein_ids": float(ids[m].min()), "max_protein_ids": float(ids[m].max()),
               "q25_protein_ids": float(np.quantile(ids[m], 0.25)), "q75_protein_ids": float(np.quantile(ids[m], 0.75))}
        for col in s.LABEL_COLUMNS:
            counts = labels.loc[m, col].astype(str).value_counts()
            row[f"majority_{col}"] = counts.index[0]
            row[f"majority_{col}_fraction"] = float(counts.iloc[0] / m.sum())
        rows.append(row)
    per_cluster = pd.DataFrame(rows)
    eta2 = s.eta_squared(ids, cl)
    eta2_ln = s.eta_squared(np.log(ids), cl)
    p_kw = s.kruskal_p(ids, cl)
    n_per_pre = per_cluster["majority_kidney_pre_label"].value_counts().to_dict()
    summary = [{"quantity": "n_clusters", "value": len(per_cluster)},
               {"quantity": "cluster_size_min", "value": int(per_cluster["n_cells"].min())},
               {"quantity": "cluster_size_max", "value": int(per_cluster["n_cells"].max())},
               {"quantity": "eta2_protein_ids_on_cluster", "value": eta2},
               {"quantity": "eta2_ln_protein_ids_on_cluster", "value": eta2_ln},
               {"quantity": "kruskal_wallis_p_protein_ids", "value": p_kw}]
    for lab in cc.ordered_labels(per_cluster["majority_kidney_pre_label"], s.ORDER):
        sub = per_cluster[per_cluster["majority_kidney_pre_label"] == lab]
        summary.append({"quantity": f"{lab}: clusters; median IDs range",
                        "value": f"{len(sub)}; {sub['median_protein_ids'].min():g}-{sub['median_protein_ids'].max():g}"})
    summary = pd.DataFrame(summary)

    checks.add("stored Leiden 2.0: n clusters; size range", f"{len(per_cluster)}; {per_cluster['n_cells'].min()}-"
               f"{per_cluster['n_cells'].max()}",
               len(per_cluster) == BRIEF_STORED["n_clusters"] and per_cluster["n_cells"].min() == BRIEF_STORED["size_min"]
               and per_cluster["n_cells"].max() == BRIEF_STORED["size_max"], "plan: 24; 8-58")
    checks.add("stored Leiden 2.0: majority kidney_pre_label of each cluster = all its cells (pre label is the cluster "
               "name)", f"min fraction {per_cluster['majority_kidney_pre_label_fraction'].min():.3f}",
               bool((per_cluster["majority_kidney_pre_label_fraction"] == 1).all()), "= 1")
    checks.add("stored Leiden 2.0: clusters per pre label equal the plan",
               "; ".join(f"{k} {v}" for k, v in n_per_pre.items()), n_per_pre == BRIEF_STORED["clusters_per_pre_label"])
    checks.add("stored Leiden 2.0: eta^2 of protein IDs on cluster equals the plan", f"{eta2:.4f}",
               abs(eta2 - BRIEF_STORED["eta2"]) < TOL["eta2"], f"plan {BRIEF_STORED['eta2']} (+/- {TOL['eta2']})")
    checks.add("stored Leiden 2.0: Kruskal-Wallis p of protein IDs equals the plan (1 significant digit)",
               f"{p_kw:.3g}", f"{p_kw:.0e}" == BRIEF_STORED["kruskal_p_1sig"], f"plan {BRIEF_STORED['kruskal_p_1sig']}")
    for lab, (lo, hi) in BRIEF_STORED["median_ids_range"].items():
        sub = per_cluster[per_cluster["majority_kidney_pre_label"] == lab]["median_protein_ids"]
        checks.add(f"stored Leiden 2.0: {lab} cluster median IDs range equals the plan", f"{sub.min():g}-{sub.max():g}",
                   round(sub.min()) == lo and round(sub.max()) == hi, f"plan {lo}-{hi}")
    return per_cluster, summary


def depth_rows(graph: str, pcs: np.ndarray, variance_ratio, ln_ids, ln_total) -> pd.DataFrame:
    frame = s.pc_depth(pcs, ln_ids, ln_total, variance_ratio)
    frame.insert(0, "graph", graph)
    frame["abs_rho_ln_n_detected"] = frame["spearman_rho_ln_n_detected"].abs()
    frame["abs_rho_ln_total_intensity"] = frame["spearman_rho_ln_total_intensity"].abs()
    return frame


def t04_depth_check(checks: t5.Checks, run: cc.Run, graph: str, variant: str, frame: pd.DataFrame) -> None:
    t04 = pd.read_csv(run.use_input(s.PREPROCESSING_PC_DEPTH, "preprocessing kidney PC-depth table (reproduction check)"))
    ref = t04[(t04["variant_id"] == variant) & (t04["pc"] <= s.N_DEPTH_PCS)].set_index("pc")
    mine = frame.set_index("pc")
    d = max(float(np.max(np.abs(np.abs(mine[c]) - np.abs(ref[c].reindex(mine.index)))))
            for c in ("spearman_rho_ln_n_detected", "spearman_rho_ln_total_intensity"))
    checks.add(f"graph {graph}: |rho| of PC1-5 with ln(IDs) and ln(total) equals preprocessing {variant}", f"max abs diff {d:.3g}",
               d < TOL["rho"], f"< {TOL['rho']}")


def main() -> None:
    args = parse_args()
    run = cc.Run(s.TASK, "graphs", args)
    t_start = time.time()
    adata, labels = t5.load(run, s.DATASET)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    ln1p = np.log1p(intensity)
    ln_ids, ln_total = s.depth_covariates(labels)
    checks = t5.Checks(s.DATASET)

    print("[kidney] input checks", flush=True)
    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        expected = t5.EXPECTED_COUNTS[col]
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in expected),
                   observed == expected)
    d = float(np.abs(ln1p - cc.dense(adata.layers["ln1p_intensity"]).astype(np.float64)).max())
    checks.add("ln(1 + layers['intensity']) equals layers['ln1p_intensity']", f"max abs diff {d:.3g}",
               d < TOL["matrix"], f"< {TOL['matrix']}")
    n_det = (intensity > 0).sum(axis=1)
    checks.add("proteins > 0 per cell equal n_detected_proteins",
               f"{int((n_det == labels['n_detected_proteins'].to_numpy()).sum())}/{len(n_det)} cells",
               bool(np.array_equal(n_det, labels["n_detected_proteins"].to_numpy(int))))
    tot = intensity.sum(axis=1)
    d = float(np.max(np.abs(tot / labels["total_intensity"].to_numpy(float) - 1)))
    checks.add("sum of intensities per cell equals total_intensity (relative)", f"max rel diff {d:.3g}", d < 1e-5, "< 1e-5")
    per_cluster, stored_summary = stored_leiden_facts(checks, labels)
    if not checks.ok:
        stop(run, checks)

    pcs, var_ratio, conn, dist, defs, depth = {}, {}, {}, {}, [], []
    # (a) stored --------------------------------------------------------------------------------
    pcs["stored"] = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    var_ratio["stored"] = np.asarray(adata.uns["pca"]["variance_ratio"], dtype=np.float64)
    conn["stored"], dist["stored"] = adata.obsp["connectivities"], adata.obsp["distances"]
    params = adata.uns["neighbors"]["params"]
    checks.add("graph stored: neighbour parameters", f"n_neighbors {params['n_neighbors']}; n_pcs {params['n_pcs']}",
               int(params["n_neighbors"]) == s.K_NEIGHBORS and int(params["n_pcs"]) == s.N_PCS, "10; 20")
    depth.append(depth_rows("stored", pcs["stored"], var_ratio["stored"], ln_ids, ln_total))
    t04_depth_check(checks, run, "stored", t4.BASELINE_ID, depth[-1])
    defs.append({"graph": "stored", "n_features": adata.n_vars})

    # (b) depth-regressed ----------------------------------------------------------------------
    print("[kidney] graph regressed", flush=True)
    resid = s.regress_out(ln1p, ln_ids)
    r = np.array([np.corrcoef(resid[:, j], ln_ids)[0, 1] for j in range(resid.shape[1])])
    checks.add("graph regressed: max |Pearson r| of residuals with ln(IDs) over proteins", f"{np.nanmax(np.abs(r)):.3g}",
               np.nanmax(np.abs(r)) < TOL["resid_r"], f"< {TOL['resid_r']}")
    ref = ad.AnnData(X=ln1p.copy(), obs=pd.DataFrame({"ln_ids": ln_ids}, index=cell_ids))
    sc.pp.regress_out(ref, keys=["ln_ids"])
    d = float(np.abs(cc.dense(ref.X).astype(np.float64) - resid).max())
    checks.add("graph regressed: residuals equal sc.pp.regress_out(ln_ids)", f"max abs diff {d:.3g}",
               d < TOL["regress_out"], f"< {TOL['regress_out']}")
    design = np.column_stack([np.ones(len(ln_ids)), ln_ids])
    beta = np.linalg.lstsq(design, ln1p, rcond=None)[0]
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((ln1p - ln1p.mean(axis=0)) ** 2).sum(axis=0)
    coef = pd.DataFrame({"protein": var_names, "intercept": beta[0], "slope_per_ln_id": beta[1],
                         "r2_ln1p_on_ln_ids": 1 - ss_res / ss_tot,
                         "detection_rate": (intensity > 0).mean(axis=0)})
    g_b = t4.build_embedding(resid, np.ones(len(var_names), bool), "z10", cell_ids, var_names)
    n_clipped = int((np.abs(cc.scale_like_scanpy(resid)) >= 10).sum())
    checks.add("graph regressed: values at the +/-10 clip after z-scoring", f"{n_clipped}", None)
    g_b = t4.build_graph(g_b, s.N_PCS, s.K_NEIGHBORS)
    pcs["regressed"] = np.asarray(g_b.obsm["X_pca"], dtype=np.float64)
    var_ratio["regressed"] = np.asarray(g_b.uns["pca"]["variance_ratio"], dtype=np.float64)
    conn["regressed"], dist["regressed"] = g_b.obsp["connectivities"], g_b.obsp["distances"]
    depth.append(depth_rows("regressed", pcs["regressed"], var_ratio["regressed"], ln_ids, ln_total))
    defs.append({"graph": "regressed", "n_features": len(var_names)})

    # (c) preprocessing total . unscaled . hvg500 --------------------------------------------------------
    print("[kidney] graph t04_total_unscaled_hvg500", flush=True)
    norm_log, info = t4.normalize(intensity, "total")
    mask, hvg_var = t4.hvg_mask(norm_log, t4.HVG_SIZES["hvg500"], var_names)
    checks.add("graph t04_total_unscaled_hvg500: HVG set size", f"{int(mask.sum())}", int(mask.sum()) == 500, "= 500")
    g_c = t4.build_embedding(norm_log, mask, "unscaled", cell_ids, var_names)
    g_c = t4.build_graph(g_c, s.N_PCS, s.K_NEIGHBORS)
    pcs[s.GRAPH_NAMES[2]] = np.asarray(g_c.obsm["X_pca"], dtype=np.float64)
    var_ratio[s.GRAPH_NAMES[2]] = np.asarray(g_c.uns["pca"]["variance_ratio"], dtype=np.float64)
    conn[s.GRAPH_NAMES[2]], dist[s.GRAPH_NAMES[2]] = g_c.obsp["connectivities"], g_c.obsp["distances"]
    depth.append(depth_rows(s.GRAPH_NAMES[2], pcs[s.GRAPH_NAMES[2]], var_ratio[s.GRAPH_NAMES[2]], ln_ids, ln_total))
    t04_depth_check(checks, run, s.GRAPH_NAMES[2], s.PREPROCESSING_VARIANT, depth[-1])
    t04_cl = pd.read_csv(run.use_input(s.PREPROCESSING_CLUSTERS, "preprocessing kidney per-cell clusters (reproduction check)"),
                         dtype=str).set_index("cell_id").reindex(cell_ids)
    for res in t4.RESOLUTIONS:
        mine = t5.leiden(g_c, res, f"chk_{res}").to_numpy()
        a = float(adjusted_rand_score(t04_cl[f"{s.PREPROCESSING_VARIANT}__r{res:.1f}"].to_numpy(), mine))
        checks.add(f"graph t04_total_unscaled_hvg500: Leiden {res} (seed 42) equals preprocessing (ARI)", f"{a:.6f}", a == 1.0, "= 1")
    hvg = hvg_var.loc[mask].copy()
    hvg.insert(0, "protein", hvg.index.astype(str))
    hvg["detection_rate"] = (intensity[:, mask] > 0).mean(axis=0)
    defs.append({"graph": s.GRAPH_NAMES[2], "n_features": int(mask.sum())})
    if not checks.ok:
        stop(run, checks)

    # Tables and the graph object -------------------------------------------------------------
    depth = pd.concat(depth, ignore_index=True)
    rows = []
    for d_row in defs:
        g = d_row["graph"]
        sub = depth[depth["graph"] == g].set_index("pc")
        c = conn[g]
        rows.append({"graph": g, "label": s.GRAPHS[g]["short"], "display": s.GRAPHS[g]["display"],
                     "definition": s.GRAPHS[g]["definition"], "n_features": d_row["n_features"], "n_pcs": s.N_PCS,
                     "k_neighbors": s.K_NEIGHBORS,
                     "max_abs_rho_pc1_pc2_ln_n_detected": float(sub.loc[[1, 2], "abs_rho_ln_n_detected"].max()),
                     "max_abs_rho_pc1_pc5_ln_n_detected": float(sub["abs_rho_ln_n_detected"].max()),
                     "pc_of_max_abs_rho_ln_n_detected": int(sub["abs_rho_ln_n_detected"].idxmax()),
                     "max_abs_rho_pc1_pc5_ln_total_intensity": float(sub["abs_rho_ln_total_intensity"].max()),
                     "variance_ratio_pc1_pc20": float(var_ratio[g][:s.N_PCS].sum()),
                     "connectivities_nnz": int(c.nnz), "connectivities_symmetric": bool(abs(c - c.T).max() < 1e-6)})
    graph_defs = pd.DataFrame(rows)
    for row in graph_defs.itertuples(index=False):
        checks.add(f"graph {row.graph}: max |rho| PC1/PC2 with ln(IDs); PC1-5 (PC)",
                   f"{row.max_abs_rho_pc1_pc2_ln_n_detected:.3f}; {row.max_abs_rho_pc1_pc5_ln_n_detected:.3f} "
                   f"(PC{row.pc_of_max_abs_rho_ln_n_detected})", None)

    obs = labels[["n_detected_proteins", "total_intensity"]].copy()
    obs.index.name = None
    store = ad.AnnData(obs=obs)
    for g in s.GRAPH_NAMES:
        store.obsm[f"X_pca__{g}"] = pcs[g].astype(np.float32)
        store.obsp[f"connectivities__{g}"] = conn[g].tocsr()
        store.obsp[f"distances__{g}"] = dist[g].tocsr()
        store.uns[f"variance_ratio__{g}"] = var_ratio[g]
    store.uns["hvg500__t04_total_unscaled_hvg500"] = np.asarray(hvg["protein"], dtype=str)
    store.uns["kidney_annotation"] = {"generated_by": cc.GENERATED_BY, "script": "scripts/kidney_annotation_graphs.py",
                               "created_utc": cc.now_utc(), "seed": s.SEED, "n_pcs": s.N_PCS, "k": s.K_NEIGHBORS,
                               **{f"definition__{g}": s.GRAPHS[g]["definition"] for g in s.GRAPH_NAMES}}

    run.table(checks.frame(), "graphs_checks")
    run.table(graph_defs, "graph_definitions")
    run.table(depth, "graph_pc_depth")
    run.table(per_cluster, "stored_leiden_depth")
    run.table(stored_summary, "stored_leiden_depth_summary")
    run.table(coef, "graph_regressed_coefficients")
    run.table(hvg, "graph_t04_hvg500_proteins")
    run.h5ad(store, "kidney_scan_graphs")
    run.finish({"graphs": {g: s.GRAPHS[g]["definition"] for g in s.GRAPH_NAMES}, "n_pcs": s.N_PCS,
                "k_neighbors": s.K_NEIGHBORS, "pca": "arpack, 50 components, random_state 42 (preprocessing_lib)",
                "depth": "Spearman rho of PC1-PC5 with ln(n_detected_proteins) and ln(total_intensity)",
                "total_normalization": info, "tolerances": TOL, "scanpy": sc.__version__,
                "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "runtime_s": round(time.time() - t_start, 1),
                "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
