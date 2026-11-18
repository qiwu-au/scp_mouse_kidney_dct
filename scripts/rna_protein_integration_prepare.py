#!/usr/bin/env python
"""rna_protein_integration step 2a: shared features, SCP matrix check and balanced RNA subsamples of the atlas.

Analysis plan,
step 2 (design approved by the authors, 15 Oct 2026). The only rna_protein_integration script that loads the full atlas (mka.h5ad,
read-only; about 4.6 GB in memory).

What it does
------------
1. Shared features: the kidney_rna table (3,569 atlas genes matching a quantified protein) is re-derived from the atlas
   genes and the canonical protein names and checked; SCP detection per feature; flag for the >= 10% SCP detection
   sensitivity; transporter / channel prefix rule (rna_protein_integration_lib; listed for review).
2. SCP matrix: canonical layers['ln1p_intensity'] on the shared proteins, z-scored per protein (ddof 1, clip 10);
   checked against the canonical .X; kidney_annotation labels matched on cell_id and counted against the project notes.
3. Balanced RNA subsamples: per design (all atlas cells; nuclei only) and draw (seeds 42-46), 200 cells per atlas
   Celltype (35 classes, Unknown excluded) drawn without replacement (numpy default_rng(seed), classes in kidney_rna
   display order, candidates in atlas order); classes with <= 200 eligible cells are taken whole.
4. Pool of all drawn cells: .raw.X counts normalized to 10,000 per cell over all 16,711 genes and log1p (as kidney_rna),
   kept on the shared genes; stored atlas X_scVI; obs fields. Checked: counts are integers; for every class taken
   whole the pool mean equals the kidney_rna pseudobulk mean (all shared genes).

Inputs (read-only): mka.h5ad; kidney_object canonical kidney object and labels; kidney_annotation labels; kidney_rna shared features, level
counts and atlas pseudobulk.
Outputs: results/tables/rna_protein_integration/rna_protein_integration_features.csv, rna_protein_integration_transporter_channel_rule.csv,
         rna_protein_integration_rna_subsample.csv, rna_protein_integration_rna_subsample_summary.csv, rna_protein_integration_prepare_checks.csv;
         results/h5ad/rna_protein_integration_rna_subsamples.h5ad
Command (project root):  python scripts/rna_protein_integration_prepare.py
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
import rna_protein_integration_lib as T

POOL_CHECK_TOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rna_protein_integration step 2a: shared features, SCP check, RNA subsamples.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "prepare_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see rna_protein_integration_prepare_checks.csv.")


def draw(obs: pd.DataFrame, design: str, seed: int, classes: list[str]) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    ct = obs[T.LEVEL].astype(str).to_numpy()
    src = obs["Source"].astype(str).to_numpy()
    eligible_src = np.ones(len(obs), dtype=bool) if T.DESIGNS[design] is None else (src == T.DESIGNS[design])
    picked, rows = [], []
    for c in classes:
        cand = np.flatnonzero((ct == c) & eligible_src)
        if len(cand) == 0:
            rows.append({"design": design, "seed": seed, "Celltype": c, "n_eligible": 0, "n_drawn": 0,
                         "taken_whole": True})
            continue
        if len(cand) <= T.N_PER_TYPE:
            sel = cand
        else:
            sel = np.sort(rng.choice(cand, size=T.N_PER_TYPE, replace=False))
        picked.append(sel)
        rows.append({"design": design, "seed": seed, "Celltype": c, "n_eligible": int(len(cand)),
                     "n_drawn": int(len(sel)), "taken_whole": bool(len(cand) <= T.N_PER_TYPE)})
    return np.concatenate(picked), rows


def main() -> None:
    args = parse_args()
    t0 = time.time()
    run = cc.Run(T.TASK, "prepare", args)
    checks = t5.Checks(T.DATASET)

    shared = T.read_shared(run, checks)
    scp = T.load_scp(run, checks, shared)
    det_diff = float(np.max(np.abs(scp.detection - shared["scp_detection_rate"].to_numpy(float))))
    checks.add("SCP detection per shared feature equals the kidney_rna table (max abs difference)", f"{det_diff:.1e}",
               det_diff < 1e-9)
    level_counts = pd.read_csv(run.use_input(T.KIDNEY_RNA_LEVEL_COUNTS, "kidney_rna atlas level counts"))
    pb = L7.read_pseudobulk(run)
    if not checks.ok:
        stop(run, checks)

    import anndata as ad

    print("[atlas] reading mka.h5ad (about 4.6 GB in memory)", flush=True)
    a = ad.read_h5ad(run.use_input(T.ATLAS, "RNA atlas mka.h5ad (read-only)"))
    checks.add("atlas shape (cells x .X genes); .raw genes", f"{a.shape[0]} x {a.shape[1]}; {a.raw.n_vars}",
               a.shape == T.ATLAS_EXPECTED_SHAPE and a.raw.n_vars == T.ATLAS_EXPECTED_RAW_GENES)
    checks.add("atlas obs_names unique", f"{a.obs_names.is_unique}", a.obs_names.is_unique)
    obs_all = a.obs
    raw_genes = a.raw.var_names.astype(str)
    match = L7.match_proteins(scp.var_names, raw_genes)
    same = (len(match) == len(shared) and
            set(zip(match["protein"], match["atlas_gene"])) == set(zip(shared["protein"], shared["atlas_gene"])))
    checks.add("shared features re-derived from the atlas genes equal the kidney_rna table (protein <-> gene pairs)",
               f"{len(match)} pairs; same {same}", same)
    lc = level_counts[level_counts["level"] == T.LEVEL].set_index("label")["n_cells"]
    obs_ct = obs_all[T.LEVEL].astype(str).value_counts()
    same_counts = {k: int(v) for k, v in obs_ct.items()} == {k: int(v) for k, v in lc.items()}
    checks.add("atlas Celltype counts equal the kidney_rna level counts", f"{len(obs_ct)} classes; same {same_counts}",
               same_counts)
    classes = [c for c in L7.atlas_order(obs_ct.index) if c != T.ATLAS_UNKNOWN]
    checks.add("atlas Celltype classes used (Unknown excluded)", f"{len(classes)}", len(classes) == 35, "35")
    if not checks.ok:
        stop(run, checks)

    # draws ---------------------------------------------------------------------------------------------
    draw_rows, summary_rows, pool_pos = [], [], []
    for design in T.DESIGNS:
        for seed in T.SUBSAMPLE_SEEDS:
            pos, rows = draw(obs_all, design, seed, classes)
            summary_rows.extend(rows)
            pool_pos.append(pos)
            o = obs_all.iloc[pos]
            draw_rows.append(pd.DataFrame({"design": design, "seed": seed, "atlas_cell": o.index.astype(str),
                                           "Celltype": o[T.LEVEL].astype(str).to_numpy(),
                                           "Celltype_finest": o[T.FINEST].astype(str).to_numpy(),
                                           "Source": o["Source"].astype(str).to_numpy(),
                                           "Origin": o["Origin"].astype(str).to_numpy()}))
            n_whole = sum(r["taken_whole"] and r["n_drawn"] > 0 for r in rows)
            checks.add(f"draw {design} / seed {seed}: cells; classes with cells; classes taken whole",
                       f"{len(pos)}; {sum(r['n_drawn'] > 0 for r in rows)}; {n_whole}",
                       len(np.unique(pos)) == len(pos) and all(r["n_drawn"] == min(r["n_eligible"], T.N_PER_TYPE)
                                                              for r in rows))
    draws = pd.concat(draw_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    fin = draws.assign(finest_known=draws["Celltype_finest"] != T.ATLAS_UNKNOWN,
                       nuclei=draws["Source"] == "Nuclei")
    agg = fin.groupby(["design", "seed", "Celltype"]).agg(n_nuclei=("nuclei", "sum"),
                                                          n_finest_known=("finest_known", "sum")).reset_index()
    summary = summary.merge(agg, on=["design", "seed", "Celltype"], how="left").fillna({"n_nuclei": 0,
                                                                                       "n_finest_known": 0})
    primary_n = int((draws["design"] == "all").groupby(draws["seed"]).sum().iloc[0])
    checks.add("primary draw size (all cells, seed 42)", f"{primary_n}", primary_n == 6112, "6112 (plan)")

    # pool ----------------------------------------------------------------------------------------------
    pos_all = np.unique(np.concatenate(pool_pos))
    raw = a.raw.X
    if not sparse.isspmatrix_csr(raw):
        raw = sparse.csr_matrix(raw)
    counts = raw[pos_all].astype(np.float32)
    d = counts.data
    n_nonint = int(np.count_nonzero(np.mod(d, 1)))
    checks.add("pool .raw.X values are positive integers (all non-zero values)",
               f"{d.size} values; non-integers {n_nonint}; min {float(d.min()):g}", n_nonint == 0 and float(d.min()) > 0)
    ln_all = T.lognorm_rows(counts)
    gidx = pd.Index(raw_genes).get_indexer(shared["atlas_gene"])
    ln = ln_all[:, gidx].tocsr()
    obs = obs_all.iloc[pos_all][[c for c in T.OBS_KEEP if c in obs_all.columns]].copy()
    for c in obs.columns:
        if str(obs[c].dtype) == "category":
            obs[c] = obs[c].astype(str)
    scvi = np.asarray(a.obsm["X_scVI"])[pos_all].astype(np.float32)
    del a, raw, counts, ln_all
    gc.collect()

    # check: classes taken whole -> pool mean equals the kidney_rna pseudobulk mean
    ct = obs[T.LEVEL].astype(str).to_numpy()
    whole = summary[(summary["design"] == "all") & summary["taken_whole"] & (summary["n_drawn"] > 0)]
    worst = 0.0
    for c in sorted(whole["Celltype"].unique()):
        m = ct == c
        mean_pool = np.asarray(ln[m].mean(axis=0), dtype=np.float64).ravel()
        ref = L7.aggregate(pb, L7.group_mask(pb, **{T.LEVEL: c}), shared["atlas_gene"].tolist())
        n_ok = int(m.sum()) == int(ref["n_cells"].iloc[0])
        diff = float(np.max(np.abs(mean_pool - ref["mean_lognorm"].to_numpy())))
        worst = max(worst, diff if n_ok else np.inf)
    checks.add("classes taken whole: pool mean log-normalized expression equals the kidney_rna pseudobulk "
               f"({len(whole['Celltype'].unique())} classes, 3,569 genes; max abs difference)", f"{worst:.1e}",
               worst < POOL_CHECK_TOL, f"< {POOL_CHECK_TOL:g}")

    var = pd.DataFrame({"protein": shared["protein"].to_numpy()}, index=pd.Index(shared["atlas_gene"].astype(str),
                                                                                  name="atlas_gene"))
    out = ad.AnnData(X=ln.astype(np.float32), obs=obs, var=var)
    out.obsm["X_scVI"] = scvi
    out.uns["rna_protein_integration"] = {
        "generated_by": cc.GENERATED_BY, "script": "scripts/rna_protein_integration_prepare.py", "created_utc": cc.now_utc(),
        "source": "mka.h5ad", "source_sha256": run.inputs[cc.rel(T.ATLAS)]["sha256"],
        "cells": "union of the rna_protein_integration balanced draws (rna_protein_integration_rna_subsample.csv)",
        "X": ".raw.X counts / cell total over 16,711 genes x 10,000, natural log1p; shared genes only (float32)",
        "obsm": "X_scVI as stored in the atlas (the atlas authors' model)"}
    if not checks.ok:
        stop(run, checks)
    path = run.out("h5ad", "rna_subsamples", ".h5ad")
    out.write_h5ad(path, compression="gzip")
    print(f"Saved AnnData: {cc.rel(path)} ({out.n_obs} x {out.n_vars})", flush=True)
    checks.add("pool cells (union of all draws) x shared genes", f"{out.n_obs} x {out.n_vars}", None)

    # features ------------------------------------------------------------------------------------------
    feat = shared.copy()
    feat["scp_detection_rate_recomputed"] = scp.detection
    feat["in_features_scp10"] = feat["scp_detection_rate"] >= T.FEATURE_SENS_DETECTION
    feat["transporter_channel_prefix"] = [T.transporter_channel_rule(g) for g in feat["atlas_gene"]]
    feat["transporter_channel"] = feat["transporter_channel_prefix"] != ""
    checks.add("features detected in >= 10% of SCP cells (sensitivity set)", f"{int(feat['in_features_scp10'].sum())}",
               None)
    checks.add("shared features flagged by the transporter / channel prefix rule",
               f"{int(feat['transporter_channel'].sum())}", None)
    rule = pd.DataFrame(
        [{"prefix": p, "kind": "included prefix",
          "n_shared_features": int((feat["transporter_channel_prefix"] == p).sum()),
          "shared_features": ", ".join(feat.loc[feat["transporter_channel_prefix"] == p, "atlas_gene"])}
         for p in T.TRANSPORTER_PREFIXES] +
        [{"prefix": p, "kind": "excluded prefix (not a transporter or channel)",
          "n_shared_features": int(feat["atlas_gene"].astype(str).str.startswith(p).sum()),
          "shared_features": ", ".join(feat.loc[feat["atlas_gene"].astype(str).str.startswith(p), "atlas_gene"])}
         for p in T.TRANSPORTER_EXCLUDE_PREFIXES])

    run.table(feat, "features")
    run.table(rule, "transporter_channel_rule")
    run.table(draws, "rna_subsample")
    run.table(summary, "rna_subsample_summary")
    run.table(checks.frame(), "prepare_checks")
    run.finish({"n_per_type": T.N_PER_TYPE, "seeds": list(T.SUBSAMPLE_SEEDS), "designs": T.DESIGNS,
                "classes": classes, "pool_cells": int(out.n_obs), "runtime_s": round(time.time() - t0, 1),
                "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
