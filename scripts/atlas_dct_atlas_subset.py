#!/usr/bin/env python
"""atlas_dct step 1a: DCT-region subset of the RNA atlas and its summary.

Analysis plan,
step 1. The only atlas_dct script that loads the full atlas (mka.h5ad, read-only; about 4.6 GB in memory).

What it does
------------
1. Lists every atlas label related to DCT at each annotation level (Celltype, Celltype_finest_lowres,
   Celltype_finest): labels containing 'DCT'.
2. Takes the atlas Celltype classes DCT, DCT-CNT, CNT, CTAL, MTAL and LOH (the DCT-related classes plus the CNT and
   TAL reference groups; LOH is the mixed loop-of-Henle class, kept for the summary only) with raw counts for all
   16,711 .raw genes, the obs fields and obsm['X_scVI'] / ['X_umap'], and writes
   results/h5ad/atlas_dct_atlas_dct_region.h5ad (later steps read this file, not the atlas).
3. Summarizes the subset by Source, Technology, Origin, Age and sex (Gender), and Celltype x Celltype_finest.
4. Checks: class counts equal the kidney_rna level counts; every stored count is a positive integer; row sums equal obs
   total_counts; per atlas group (Source x Technology x Celltype x Celltype_finest_lowres x Celltype_finest) the
   sums of counts, of log-normalized expression and the detection counts equal the kidney_rna atlas pseudobulk.

Inputs (read-only): mka.h5ad; results/tables/kidney_rna/kidney_rna_atlas_level_counts.csv;
                    results/h5ad/kidney_rna_atlas_pseudobulk.h5ad.
Outputs: results/h5ad/atlas_dct_atlas_dct_region.h5ad;
         results/tables/atlas_dct/atlas_dct_atlas_dct_labels.csv, atlas_dct_region_by_class.csv,
         atlas_dct_region_study_design.csv, atlas_dct_region_finest_crosstab.csv, atlas_dct_atlas_subset_checks.csv

Command (project root):  python scripts/atlas_dct_atlas_subset.py
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
import gc
import time

import numpy as np
import pandas as pd
from scipy import sparse

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as L7
import atlas_dct_lib as T

PB_TOL = 1e-4          # relative tolerance, float32 storage of the kidney_rna pseudobulk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atlas_dct step 1a: DCT-region atlas subset and summary.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "atlas_subset_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see atlas_dct_atlas_subset_checks.csv.")


def dct_labels(obs: pd.DataFrame) -> pd.DataFrame:
    """Every label containing 'DCT' at each level, with counts per Source, Technology and Origin."""
    rows = []
    for level in L7.ATLAS_LEVELS:
        lab = obs[level].astype(str)
        for label in sorted(l for l in lab.unique() if "DCT" in l.upper()):
            sub = obs.loc[(lab == label).to_numpy()]
            rec = {"level": level, "label": label, "n_cells": len(sub)}
            rec.update({f"n_source_{k}": int(v) for k, v in sub["Source"].astype(str).value_counts().items()})
            rec.update({f"n_technology_{k}": int(v) for k, v in sub["Technology"].astype(str).value_counts().items()})
            rec["origins"] = "; ".join(f"{k} {v}" for k, v in sub["Origin"].astype(str).value_counts().items())
            if level != "Celltype":
                rec["celltype_of_these_cells"] = "; ".join(f"{k} {v}" for k, v in
                                                           sub["Celltype"].astype(str).value_counts().items())
            rows.append(rec)
    return pd.DataFrame(rows).fillna(0)


def by_class(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in T.REGION_CLASSES:
        sub = obs[obs["Celltype"].astype(str) == cls]
        rec = {"celltype": cls, "role": ("DCT-related" if "DCT" in cls else
                                         ("mixed class (summary only)" if cls == "LOH" else "reference")),
               "n_cells": len(sub), "median_total_counts": float(sub["total_counts"].median()),
               "median_n_genes_by_counts": float(sub["n_genes_by_counts"].median())}
        for f in ("Source", "Technology", "Origin", "Age", "Gender"):
            for k, v in sub[f].astype(str).value_counts().items():
                rec[f"{f}: {k}"] = int(v)
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.fillna(0)


def study_design(obs: pd.DataFrame) -> pd.DataFrame:
    keys = ["Celltype", "Origin", "Age", "Gender", "Source", "Technology", "Tissue_res"]
    g = obs.groupby(keys, observed=True)
    out = g.agg(n_cells=("total_counts", "size"), median_total_counts=("total_counts", "median"),
                median_n_genes_by_counts=("n_genes_by_counts", "median")).reset_index()
    out["Celltype"] = pd.Categorical(out["Celltype"], T.REGION_CLASSES, ordered=True)
    return out.sort_values(["Celltype", "Origin", "Source"]).reset_index(drop=True)


def main() -> None:
    import anndata as ad

    args = parse_args()
    run = cc.Run(T.TASK, "atlas_subset", args)
    t0 = time.time()
    checks = t5.Checks("atlas")
    lc = pd.read_csv(run.use_input(T.KIDNEY_RNA_LEVEL_COUNTS, "kidney_rna atlas level counts"))
    pb = L7.read_pseudobulk(run)

    print("[atlas] reading mka.h5ad (about 4.6 GB in memory)", flush=True)
    a = ad.read_h5ad(run.use_input(T.ATLAS, "RNA atlas mka.h5ad (read-only)"))
    checks.add("atlas shape (cells x .X genes); .raw genes", f"{a.shape[0]} x {a.shape[1]}; {a.raw.n_vars}",
               a.shape == T.ATLAS_EXPECTED_SHAPE and a.raw.n_vars == T.ATLAS_EXPECTED_RAW_GENES)
    checks.add("atlas obs_names unique", f"{a.obs_names.is_unique}", a.obs_names.is_unique)
    checks.add("obsm keys", "; ".join(f"{k} {a.obsm[k].shape}" for k in a.obsm.keys()
                                      if hasattr(a.obsm[k], "shape")), "X_scVI" in a.obsm)
    obs_all = a.obs
    labels_tab = dct_labels(obs_all)
    for r in labels_tab.itertuples():
        checks.add(f"atlas label containing 'DCT': {r.level} = {r.label}", f"{r.n_cells} cells", None)
    checks.add("labels named DCT1 or DCT2 at any level",
               f"{int(labels_tab['label'].isin(['DCT1', 'DCT2']).sum())}", None)

    mask = obs_all["Celltype"].astype(str).isin(T.REGION_CLASSES).to_numpy()
    exp = lc[lc["level"] == "Celltype"].set_index("label")["n_cells"]
    for cls in T.REGION_CLASSES:
        n = int((obs_all["Celltype"].astype(str) == cls).sum())
        checks.add(f"Celltype {cls}: cells equal the kidney_rna level counts", f"{n}", n == int(exp[cls]), f"{int(exp[cls])}")
    raw = a.raw.X
    if not sparse.isspmatrix_csr(raw):
        raw = sparse.csr_matrix(raw)
    sub = raw[np.flatnonzero(mask)].astype(np.float32)
    sub.sort_indices()
    genes = pd.Index(a.raw.var_names.astype(str))
    var = pd.DataFrame(index=genes)
    var["gene_id"] = (a.raw.var["gene_id-1"].astype(str).to_numpy() if "gene_id-1" in a.raw.var.columns else "")
    obs = obs_all.loc[mask, list(T.OBS_KEEP)].copy()
    for c in obs.columns:
        if isinstance(obs[c].dtype, pd.CategoricalDtype):
            obs[c] = obs[c].astype(str)
    obsm = {k: np.asarray(a.obsm[k])[mask] for k in ("X_scVI", "X_umap")}
    del a, raw, obs_all
    gc.collect()

    d = sub.data
    n_nonint = int(np.count_nonzero(d != np.round(d)))
    checks.add("subset: every stored count is a positive integer", f"{d.size} values; non-integers {n_nonint}; "
               f"min {float(d.min()):g}; max {float(d.max()):g}", n_nonint == 0 and float(d.min()) > 0)
    tot = np.asarray(sub.sum(axis=1), dtype=np.float64).ravel()
    diff = np.abs(tot - obs["total_counts"].to_numpy(np.float64))
    # info, as in kidney_rna (all atlas cells: max abs difference 493): obs total_counts is not the .raw.X row sum
    checks.add("subset: .raw.X row sums vs obs total_counts (max abs difference; cells differing by > 0.5); "
               "normalization uses the .raw.X row sums, as kidney_rna", f"{float(diff.max()):g}; {int((diff > 0.5).sum())}",
               None)
    checks.add("subset cells", f"{sub.shape[0]} x {sub.shape[1]} genes", sub.shape[0] == int(exp[list(T.REGION_CLASSES)].sum()))

    # pseudobulk cross-check -------------------------------------------------------------------------
    keys = list(L7.GROUP_KEYS)
    gobs = obs[keys].astype(str)
    gid, uniq = pd.factorize(pd.MultiIndex.from_frame(gobs), sort=True)
    ind = sparse.csr_matrix((np.ones(len(gid)), (gid, np.arange(len(gid)))), shape=(len(uniq), len(gid)))
    ln = T.lognorm(sub)
    s_counts = (ind @ sub.astype(np.float64)).toarray()
    s_ln = (ind @ ln.astype(np.float64)).toarray()
    s_det = (ind @ (sub > 0).astype(np.float64)).toarray()
    pbo = pb.obs[keys].astype(str)
    pkey = pd.MultiIndex.from_frame(pbo)
    pos = pkey.get_indexer(uniq)
    checks.add("subset atlas groups found in the kidney_rna pseudobulk", f"{int((pos >= 0).sum())} of {len(uniq)}",
               bool((pos >= 0).all()))
    if (pos >= 0).all():
        n_ok = np.array_equal(pb.obs["n_cells"].to_numpy()[pos], np.bincount(gid, minlength=len(uniq)))
        checks.add("subset group sizes equal the kidney_rna pseudobulk n_cells", f"{n_ok}", n_ok)
        if not pb.var_names.equals(genes):
            checks.add("pseudobulk genes equal the atlas .raw genes", "False", False)
        else:
            for name, ours, theirs in (("sum of counts", s_counts, pb.layers["sum_counts"]),
                                       ("sum of log-normalized expression", s_ln, pb.X),
                                       ("cells with counts > 0", s_det, pb.layers["n_detected"])):
                t_ = np.asarray(theirs[pos], dtype=np.float64)
                rel_d = float(np.max(np.abs(ours - t_) / np.maximum(np.abs(t_), 1.0)))
                checks.add(f"subset per-group {name} equals the kidney_rna pseudobulk (max relative difference)",
                           f"{rel_d:.2e}", rel_d < PB_TOL, f"< {PB_TOL:g}")
    del ln, s_ln, s_counts, s_det
    if not checks.ok:
        stop(run, checks)

    # summaries --------------------------------------------------------------------------------------------
    for f in ("Age", "Gender", "Genetic_background"):
        per_origin = obs.groupby("Origin")[f].nunique()
        checks.add(f"subset: {f} levels; levels per Origin (max)",
                   f"{'; '.join(f'{k} {v}' for k, v in obs[f].value_counts().items())}; {int(per_origin.max())}", None)
    ct = pd.crosstab(obs["Celltype"], obs["Celltype_finest"]).reindex(index=list(T.REGION_CLASSES))
    ct = ct.loc[:, ct.sum(axis=0) > 0]
    ct.index.name = "Celltype"

    out = ad.AnnData(X=sub, obs=obs, var=var)
    for k, v in obsm.items():
        out.obsm[k] = v
    out.uns["atlas_dct"] = {
        "generated_by": cc.GENERATED_BY, "script": "scripts/atlas_dct_atlas_subset.py",
        "created_utc": cc.now_utc(), "source": "mka.h5ad", "source_sha256": run.inputs[cc.rel(T.ATLAS)]["sha256"],
        "cells": f"atlas Celltype in {', '.join(T.REGION_CLASSES)}", "X": ".raw.X integer counts (16,711 genes)",
        "obsm": "X_scVI and X_umap as stored in the atlas"}
    path = run.out("h5ad", "atlas_dct_region", ".h5ad")
    out.write_h5ad(path, compression="gzip")
    print(f"Saved AnnData: {cc.rel(path)}")
    checks.add("region object written (cells x genes; file size)",
               f"{out.n_obs} x {out.n_vars}; {path.stat().st_size / 1e6:.1f} MB", None)

    run.table(labels_tab, "atlas_dct_labels")
    run.table(by_class(obs), "region_by_class")
    run.table(study_design(obs), "region_study_design")
    run.table(ct.reset_index(), "region_finest_crosstab")
    run.table(checks.frame(), "atlas_subset_checks")
    run.finish({"region_classes": list(T.REGION_CLASSES), "level": T.LEVEL, "pseudobulk_tolerance": PB_TOL,
                "runtime_s": round(time.time() - t0, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see atlas_dct_atlas_subset_checks.csv.")


if __name__ == "__main__":
    main()
