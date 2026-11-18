#!/usr/bin/env python
"""rna_protein_integration step 2b: joint embeddings of SCP cells and the primary RNA draw; UMAPs; joint AnnData.

Analysis plan,
step 2 (design approved by the authors, 15 Oct 2026). Primary configuration only (all atlas cells, draw seed 42,
3,569 shared features, all 35 atlas Celltypes); the sensitivities and the other draws are run by
rna_protein_integration_diagnostics.py.

Embeddings (rna_protein_integration_lib.build_joint)
---------------------------------------
- A, joint PCA: SCP (712) and RNA (6,112) rows, each z-scored within its modality (ddof 1, clip 10); 30 loadings
  from the average of the two within-modality covariance matrices (equal weight); scores = rows x loadings.
- B, Harmony: harmonypy 0.0.10 run_harmony on the 30 PCs, batch = modality, seed 42, package defaults (theta 1,
  sigma 0.1, lambda 1, clusters = min(N / 30, 100), at most 10 iterations).
- C, atlas scVI: RNA cells at their stored X_scVI (the atlas authors' model; weights not in the file); each SCP cell
  at the mean X_scVI of its 15 nearest RNA cells by cosine distance on the z-scored shared features. The SCP
  positions are a function of the shared-feature kNN by construction (no mixing metric is computed for C).
UMAP per embedding: scanpy neighbours (k = 15, use_rep), umap (seed 42, min_dist 0.5).

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation labels; kidney_rna shared features and approved mapping;
  results/h5ad/rna_protein_integration_rna_subsamples.h5ad and rna_protein_integration_rna_subsample.csv (rna_protein_integration_prepare.py).
Outputs: results/h5ad/rna_protein_integration_joint_embedding.h5ad; results/tables/rna_protein_integration/
         rna_protein_integration_pc_modality_variance.csv, rna_protein_integration_scvi_projection_per_cell.csv,
         rna_protein_integration_integrate_checks.csv
Command (project root):  python scripts/rna_protein_integration_integrate.py
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

import numpy as np
import pandas as pd

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import rna_protein_integration_lib as T

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rna_protein_integration step 2b: joint embeddings (primary configuration).")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "integrate_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see rna_protein_integration_integrate_checks.csv.")


def umap(emb: np.ndarray, seed: int = T.SEED) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=np.zeros((emb.shape[0], 1), dtype=np.float32))
    a.obsm["rep"] = np.ascontiguousarray(emb, dtype=np.float32)
    sc.pp.neighbors(a, n_neighbors=T.K_GRAPH, use_rep="rep", random_state=seed)
    sc.tl.umap(a, min_dist=T.UMAP_MIN_DIST, random_state=seed)
    return np.asarray(a.obsm["X_umap"], dtype=np.float32)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    run = cc.Run(T.TASK, "integrate", args)
    checks = t5.Checks(T.DATASET)
    checks.add("scanpy igraph RNG compatibility patch applied (_scanpy_compat.py)", f"{SCANPY_COMPAT_PATCHED}", None)
    shared = T.read_shared(run, checks)
    scp = T.load_scp(run, checks, shared)
    mapping = T.read_mapping(run, checks)
    pool = T.load_pool(run)
    if not checks.ok:
        stop(run, checks)

    j = T.build_joint(scp, pool, "primary", T.PRIMARY_DRAW, mapping)
    checks.add("primary joint matrix: SCP cells; RNA cells; features", f"{j.info['n_scp']}; {j.info['n_rna']}; "
               f"{j.info['n_features']}", j.info["n_scp"] == 712 and j.info["n_rna"] == 6112 and
               j.info["n_features"] == 3569, "712; 6112; 3569")
    for k, e in j.emb.items():
        checks.add(f"embedding {T.EMBEDDINGS[k]}: shape; all values finite", f"{e.shape}; {np.isfinite(e).all()}",
                   bool(np.isfinite(e).all()))
    checks.add("Harmony: iterations; converged; clusters; theta; sigma; lambda",
               f"{j.info['harmony_iterations']}; {j.info['harmony_converged']}; {j.info['harmony_clusters']}; "
               f"{j.info['harmony_theta']:g}; {j.info['harmony_sigma']:g}; {j.info['harmony_lambda']:g}", None)
    pcv = T.pc_modality_variance(j)
    checks.add("joint PCA: per-PC share of within-modality score variance from SCP (PC1-PC5; median over 30 PCs)",
               "; ".join(f"{v:.2f}" for v in pcv["share_scp"].head(5)) + f"; median {pcv['share_scp'].median():.2f}",
               None)

    # common labels (one label space per label set) ----------------------------------------------------------
    obs = j.obs.copy()
    for col in T.LABEL_COLUMNS:
        sets = T.expected_sets(mapping, col, "celltype")
        inv = T.inverse_map(sets)
        common = T.common_labels(obs.loc[j.is_scp, col], obs.loc[~j.is_scp, T.LEVEL], inv)
        clash = set(common[~j.is_scp]) & (set(obs.loc[j.is_scp, col]) - set(inv.values()))
        checks.add(f"{col}: common label space (RNA classes renamed to their SCP label; others 'atlas: ...'); "
                   "no RNA name equals an SCP label without counterpart", f"clash {sorted(clash)}", not clash)
        obs[f"common_{T.LABEL_SHORT[col]}"] = common

    # UMAPs --------------------------------------------------------------------------------------------------
    umaps = {}
    for k, e in j.emb.items():
        print(f"[umap] {k}", flush=True)
        umaps[k] = umap(e)

    # projection detail (C) -------------------------------------------------------------------------------------
    rna_ct = obs.loc[~j.is_scp, T.LEVEL].to_numpy()
    nb_ct = rna_ct[j.project_idx]
    top, top_frac = [], []
    for row in nb_ct:
        vals, cnt = np.unique(row, return_counts=True)
        top.append(vals[np.argmax(cnt)])
        top_frac.append(cnt.max() / len(row))
    _, dist = T.cosine_knn(j.z[j.is_scp], j.z[~j.is_scp], T.K_PROJECT)
    proj = pd.DataFrame({"cell_id": obs.loc[j.is_scp, "cell"].to_numpy(),
                         "kidney_t15_label": obs.loc[j.is_scp, "kidney_t15_label"].to_numpy(),
                         "kidney_pre_label": obs.loc[j.is_scp, "kidney_pre_label"].to_numpy(),
                         "n_detected_proteins": obs.loc[j.is_scp, "n_detected_proteins"].to_numpy(),
                         "projection_top_celltype": top, "projection_top_fraction": top_frac,
                         "mean_cosine_distance": dist.mean(axis=1), "nearest_cosine_distance": dist[:, 0]})

    # joint AnnData ------------------------------------------------------------------------------------------------
    import anndata as ad

    obs.index = pd.Index([f"{m}:{c}" for m, c in zip(obs["modality"], obs["cell"])], name="joint_id")
    var = pd.DataFrame({"protein": np.asarray(scp.proteins), "scp_detection_rate": scp.detection},
                       index=pd.Index(j.features, name="atlas_gene"))
    out = ad.AnnData(X=j.z.astype(np.float32), obs=obs, var=var)
    out.obsm["X_pca_joint"] = j.emb["pca"].astype(np.float32)
    out.obsm["X_harmony"] = j.emb["harmony"].astype(np.float32)
    out.obsm["X_scvi_projected"] = j.emb["scvi"].astype(np.float32)
    for k, u in umaps.items():
        out.obsm[f"X_umap_{k}"] = u
    out.varm["PCs_joint"] = j.loadings.astype(np.float32)
    out.uns["rna_protein_integration"] = {
        "generated_by": cc.GENERATED_BY, "script": "scripts/rna_protein_integration_integrate.py", "created_utc": cc.now_utc(),
        "configuration": T.CONFIG_TEXT["primary"], "draw_seed": T.PRIMARY_DRAW,
        "X": "rows z-scored within modality (SCP: ln(1 + intensity); RNA: log-normalized), ddof 1, clip 10",
        "X_pca_joint": "equal-weight covariance PCA (30 PCs)",
        "X_harmony": "harmonypy 0.0.10 on X_pca_joint, batch = modality, seed 42, package defaults",
        "X_scvi_projected": "RNA: stored atlas X_scVI; SCP: mean X_scVI of 15 nearest RNA cells (cosine, shared features)",
        "X_umap_*": "scanpy neighbours (k 15) + umap (seed 42, min_dist 0.5) on each embedding",
        "harmony": {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in j.info.items()},
        "eigenvalues": j.eigenvalues.tolist()}
    if not checks.ok:
        stop(run, checks)
    path = run.out("h5ad", "joint_embedding", ".h5ad")
    out.write_h5ad(path, compression="gzip")
    print(f"Saved AnnData: {cc.rel(path)} ({out.n_obs} x {out.n_vars})", flush=True)

    run.table(pcv, "pc_modality_variance")
    run.table(proj, "scvi_projection_per_cell")
    run.table(checks.frame(), "integrate_checks")
    run.finish({"configuration": T.CONFIGS["primary"], "draw_seed": T.PRIMARY_DRAW, "n_pcs": T.N_PCS,
                "k_graph": T.K_GRAPH, "k_project": T.K_PROJECT, "umap_min_dist": T.UMAP_MIN_DIST, **j.info,
                "runtime_s": round(time.time() - t0, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
