#!/usr/bin/env python
"""kidney_rna step 1: RNA atlas summary, count checks and the atlas pseudobulk object.

Analysis plan,
step 1. The only kidney_rna script that loads the full atlas (mka.h5ad, read-only; about 4.6 GB in memory).

What it does
------------
1. Summarizes the atlas obs: every column (dtype, levels); the study-design fields (Origin, Source, Technology,
   Tissue_res, Age, Genetic_background, Gender, batch); cell counts per Source x Technology; the Celltype /
   Celltype_finest_lowres / Celltype_finest levels, their nesting and their counts per Source, Technology and
   Origin.
2. Checks the count matrices: every non-zero value of .raw.X is a positive integer (all values, not a sample);
   layers['counts'] equals .raw.X on the 3,000 genes of .X; .X described; obs total_counts / n_genes_by_counts
   compared with .raw.X row sums / non-zeros.
3. Normalizes .raw.X to 10,000 counts per cell and log1p (as the earlier integrate_scp_with_mka.py and kidney_annotation step 3a)
   and writes results/h5ad/kidney_rna_atlas_pseudobulk.h5ad: one row per atlas group (Source x Technology x
   Celltype x Celltype_finest_lowres x Celltype_finest), all 16,711 .raw genes; .X = sum of log-normalized
   expression; layers sum_sq_lognorm, n_detected (cells with counts > 0), sum_counts; obs n_cells. Checked against
   the full matrix (column non-zeros and sums) and against an independent scanpy normalization of three groups.
4. Matches atlas genes to the 4,054 kidney protein names (case-insensitive, as kidney_annotation step 3a) and writes the shared
   feature table and the coverage of the V3 + endothelial-row markers.

Inputs (read-only): mka.h5ad; kidney_object canonical kidney object; data/KIDNEY/KIDNEY_markers_V3.xlsx.
Outputs: results/tables/kidney_rna/kidney_rna_atlas_*.csv, kidney_rna_shared_features.csv,
         kidney_rna_marker_coverage.csv, kidney_rna_atlas_summary_checks.csv;
         results/h5ad/kidney_rna_atlas_pseudobulk.h5ad

Command (project root):  python scripts/kidney_rna_atlas_summary.py
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
import kidney_rna_lib as L

CHUNK_ROWS = 10000
CHUNK_VALUES = 50_000_000
SPOT_CHECK_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_rna step 1: atlas summary, count checks, atlas pseudobulk.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "atlas_summary_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_rna_atlas_summary_checks.csv.")


# ---------------------------------------------------------------------------
# obs summaries
# ---------------------------------------------------------------------------
def obs_columns(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in obs.columns:
        s = obs[c]
        nun = int(s.nunique(dropna=True))
        levels = ""
        if nun <= 50 and not pd.api.types.is_float_dtype(s):
            levels = "; ".join(f"{k} ({v})" for k, v in s.astype(str).value_counts().items())
        desc = {}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            desc = {"min": float(s.min()), "median": float(s.median()), "max": float(s.max())}
        rows.append({"column": c, "dtype": str(s.dtype), "n_unique": nun, "n_missing": int(s.isna().sum()),
                     "levels_with_counts": levels, **desc})
    return pd.DataFrame(rows)


def obs_levels(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in L.ATLAS_FIELDS:
        vc = obs[f].astype(str).value_counts()
        for level, n in vc.items():
            rows.append({"field": f, "level": level, "n_cells": int(n), "fraction": n / len(obs)})
    return pd.DataFrame(rows)


def study_design(obs: pd.DataFrame) -> pd.DataFrame:
    keys = ["Origin", "Source", "Technology", "Tissue_res", "Age", "Genetic_background", "Gender", "batch"]
    o = obs.copy()
    o["finest_annotated"] = o["Celltype_finest"].astype(str) != L.ATLAS_UNKNOWN
    o["celltype_annotated"] = o["Celltype"].astype(str) != L.ATLAS_UNKNOWN
    g = o.groupby(keys, observed=True)
    out = g.agg(n_cells=("Celltype", "size"),
                median_n_genes_by_counts=("n_genes_by_counts", "median"),
                median_total_counts=("total_counts", "median"),
                median_pct_counts_mt=("pct_counts_mt", "median"),
                frac_celltype_not_unknown=("celltype_annotated", "mean"),
                frac_finest_not_unknown=("finest_annotated", "mean")).reset_index()
    return out.sort_values(["Origin", "Source", "Technology", "Tissue_res"]).reset_index(drop=True)


def source_technology(obs: pd.DataFrame) -> pd.DataFrame:
    tab = pd.crosstab(obs["Source"].astype(str), obs["Technology"].astype(str), margins=True, margins_name="total")
    tab.index.name = "Source"
    return tab.reset_index()


def level_counts(obs: pd.DataFrame) -> pd.DataFrame:
    """Per annotation level and label: counts overall, per Source, Technology and Origin."""
    rows = []
    sources = sorted(obs["Source"].astype(str).unique())
    techs = sorted(obs["Technology"].astype(str).unique())
    origins = sorted(obs["Origin"].astype(str).unique())
    for level in L.ATLAS_LEVELS:
        lab = obs[level].astype(str)
        for label in sorted(lab.unique()):
            m = (lab == label).to_numpy()
            sub = obs.loc[m]
            rec = {"level": level, "label": label, "n_cells": int(m.sum())}
            for s_ in sources:
                rec[f"n_source_{s_}"] = int((sub["Source"].astype(str) == s_).sum())
            for t_ in techs:
                rec[f"n_technology_{t_}"] = int((sub["Technology"].astype(str) == t_).sum())
            per_origin = {o_: int((sub["Origin"].astype(str) == o_).sum()) for o_ in origins}
            rec["n_origins_with_cells"] = sum(v > 0 for v in per_origin.values())
            rec.update({f"n_origin_{o_}": v for o_, v in per_origin.items()})
            rows.append(rec)
    return pd.DataFrame(rows)


def level_crosstab(obs: pd.DataFrame) -> pd.DataFrame:
    """Observed (Celltype, Celltype_finest_lowres, Celltype_finest) combinations with counts per Source."""
    g = obs.groupby(list(L.ATLAS_LEVELS), observed=True)
    out = g.size().rename("n_cells").reset_index()
    by_src = obs.groupby([*L.ATLAS_LEVELS, "Source"], observed=True).size().unstack("Source", fill_value=0)
    by_src.columns = [f"n_source_{c}" for c in by_src.columns]
    out = out.merge(by_src.reset_index(), on=list(L.ATLAS_LEVELS), how="left")
    out["celltype_equals_finest"] = out["Celltype"].astype(str) == out["Celltype_finest"].astype(str)
    return out.sort_values(["Celltype", "n_cells"], ascending=[True, False]).reset_index(drop=True)


def functional(obs: pd.DataFrame, a: str, b: str) -> tuple[bool, str]:
    """Is b a function of a (each level of a occurs with one level of b)?"""
    n = obs.groupby(a, observed=True)[b].nunique()
    multi = n[n > 1]
    return multi.empty, ("none" if multi.empty else "; ".join(f"{k} ({v})" for k, v in multi.items()))


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------
def integer_check(data: np.ndarray) -> tuple[bool, float, float, int]:
    ok, lo, hi, n_nonint = True, np.inf, -np.inf, 0
    for i in range(0, data.size, CHUNK_VALUES):
        d = data[i:i + CHUNK_VALUES]
        nonint = int(np.count_nonzero(d != np.round(d)))
        n_nonint += nonint
        lo, hi = min(lo, float(d.min())), max(hi, float(d.max()))
    ok = n_nonint == 0 and lo > 0
    return ok, lo, hi, n_nonint


def column_nnz(m: sparse.csr_matrix) -> np.ndarray:
    out = np.zeros(m.shape[1], dtype=np.int64)
    for i in range(0, m.indices.size, CHUNK_VALUES):
        out += np.bincount(m.indices[i:i + CHUNK_VALUES], minlength=m.shape[1])
    return out


def rowwise_inplace(m: sparse.csr_matrix, factors: np.ndarray | None, func) -> None:
    """Chunked in-place update of m.data: multiply by a per-row factor (optional), then apply func."""
    counts = np.diff(m.indptr)
    for r0 in range(0, m.shape[0], CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, m.shape[0])
        s0, s1 = m.indptr[r0], m.indptr[r1]
        d = m.data[s0:s1]
        if factors is not None:
            d *= np.repeat(factors[r0:r1], counts[r0:r1]).astype(d.dtype)
        if func is not None:
            func(d)


def compare_chunked(a_m, raw: sparse.csr_matrix, cols: np.ndarray) -> tuple[float, int]:
    """Max abs difference and number of differing entries between a_m and raw[:, cols], over row chunks."""
    a_m = a_m if sparse.issparse(a_m) else sparse.csr_matrix(np.asarray(a_m))
    a_m = sparse.csr_matrix(a_m)
    max_d, n_d = 0.0, 0
    for r0 in range(0, raw.shape[0], CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, raw.shape[0])
        d = a_m[r0:r1].astype(np.float64) - raw[r0:r1][:, cols].astype(np.float64)
        d.eliminate_zeros()
        if d.nnz:
            max_d = max(max_d, float(np.abs(d.data).max()))
            n_d += int(d.nnz)
    return max_d, n_d


def group_sum(gid: np.ndarray, n_groups: int, m: sparse.csr_matrix) -> np.ndarray:
    """Per-group column sums, accumulated in float64 over row chunks."""
    out = np.zeros((n_groups, m.shape[1]), dtype=np.float64)
    step = 2 * CHUNK_ROWS
    for r0 in range(0, m.shape[0], step):
        r1 = min(r0 + step, m.shape[0])
        ind = sparse.csr_matrix((np.ones(r1 - r0), (gid[r0:r1], np.arange(r1 - r0))), shape=(n_groups, r1 - r0))
        out += (ind @ m[r0:r1].astype(np.float64)).toarray()
    return out


def main() -> None:
    import anndata as ad
    import scanpy as sc

    args = parse_args()
    run = cc.Run(L.TASK, "atlas_summary", args)
    t_start = time.time()
    checks = t5.Checks("atlas")

    kid, _ = t5.load(run, L.DATASET)
    prot = kid.var_names.astype(str)
    prot_det = pd.Series((cc.dense(kid.layers["intensity"]) > 0).mean(axis=0), index=prot)
    prot_ndet = pd.Series((cc.dense(kid.layers["intensity"]) > 0).sum(axis=0), index=prot)
    marker_rows = L.kidney_marker_rows(run)
    by_gene = L.markers_by_gene(marker_rows)

    print("[atlas] reading mka.h5ad (about 4.6 GB in memory)", flush=True)
    a = ad.read_h5ad(run.use_input(L.ATLAS, "RNA atlas mka.h5ad (read-only)"))
    checks.add("atlas shape (cells x .X genes)", f"{a.shape[0]} x {a.shape[1]}", a.shape == L.ATLAS_EXPECTED_SHAPE,
               "141,401 x 3,000 (the project notes)")
    checks.add(".raw present; genes", f"{a.raw is not None}; {a.raw.n_vars if a.raw is not None else 0}",
               a.raw is not None and a.raw.n_vars == L.ATLAS_EXPECTED_RAW_GENES, "16,711 (the project notes)")
    needed = [*L.ATLAS_FIELDS, *L.ATLAS_LEVELS, "n_genes_by_counts", "total_counts", "pct_counts_mt"]
    missing = [c for c in needed if c not in a.obs.columns]
    checks.add("obs fields present", ", ".join(missing) or "all", not missing)
    if not checks.ok:
        stop(run, checks)
    checks.add("layers / obsm / uns keys", f"layers {list(a.layers.keys())}; obsm {list(a.obsm.keys())}; "
               f"uns {list(a.uns.keys())}", None)

    # 1. obs summaries --------------------------------------------------------------------------------
    obs = a.obs.copy()
    for f in L.ATLAS_FIELDS:
        checks.add(f"obs '{f}' levels", "; ".join(f"{k} {v}" for k, v in obs[f].astype(str).value_counts().items()),
                   None)
    for lvl in L.ATLAS_LEVELS:
        n_unk = int((obs[lvl].astype(str) == L.ATLAS_UNKNOWN).sum())
        checks.add(f"obs '{lvl}': levels; cells labelled Unknown", f"{obs[lvl].nunique()}; {n_unk} "
                   f"({n_unk / len(obs):.1%})", None)
    for a_, b_ in (("Celltype_finest", "Celltype_finest_lowres"), ("Celltype_finest", "Celltype"),
                   ("Celltype", "Celltype_finest"), ("Celltype_finest_lowres", "Celltype")):
        ok, detail = functional(obs, a_, b_)
        checks.add(f"each {a_} level occurs with one {b_} level", f"{ok}; levels with more: {detail}", None)
    same = obs["Celltype"].astype(str) == obs["Celltype_finest"].astype(str)
    annotated = obs["Celltype_finest"].astype(str) != L.ATLAS_UNKNOWN
    checks.add("cells whose Celltype equals Celltype_finest (of cells with Celltype_finest not Unknown)",
               f"{int((same & annotated).sum())} of {int(annotated.sum())}", None)

    # 2. matrix checks -----------------------------------------------------------------------------------
    raw = a.raw.X
    checks.add(".raw.X storage", f"{type(raw).__name__}; dtype {raw.dtype}; nnz "
               f"{raw.nnz if sparse.issparse(raw) else 'dense'}", None)
    if not sparse.isspmatrix_csr(raw):
        raw = sparse.csr_matrix(raw)
    raw.sort_indices()
    if raw.dtype != np.float32:
        raw = raw.astype(np.float32)
    ok_int, lo, hi, n_nonint = integer_check(raw.data)
    checks.add("every stored .raw.X value is a positive integer (all values)",
               f"{raw.nnz} values; non-integers {n_nonint}; min {lo:g}; max {hi:g}", ok_int)
    genes = pd.Index(a.raw.var_names.astype(str))
    checks.add(".raw var_names unique", f"{genes.is_unique}", genes.is_unique)
    if "gene_name-1" in a.raw.var.columns:
        gn = a.raw.var["gene_name-1"].astype(str).to_numpy()
        checks.add(".raw var_names equal raw.var['gene_name-1']", f"{int((gn == genes.to_numpy()).sum())} of "
                   f"{len(genes)} equal", None)
    hvg = pd.Index(a.var_names.astype(str))
    pos = genes.get_indexer(hvg)
    checks.add(".X genes (3,000) all in .raw", f"{int((pos >= 0).sum())} of {len(hvg)}", bool((pos >= 0).all()))
    max_diff, n_diff = compare_chunked(a.layers["counts"], raw, pos)
    checks.add("layers['counts'] equals .raw.X on the 3,000 .X genes", f"max abs difference {max_diff:g}; "
               f"differing entries {n_diff}", max_diff == 0)
    xs = a.X
    xd = xs.data if sparse.issparse(xs) else np.asarray(xs).ravel()
    xd = xd[:: max(1, xd.size // 2_000_000)]
    checks.add(".X (3,000 genes) values, sample of 2 M stored values",
               f"dtype {xs.dtype}; min {float(xd.min()):g}; max {float(xd.max()):g}; integer fraction "
               f"{float(np.mean(xd == np.round(xd))):.3f}", None)
    dx_max, dx_n = compare_chunked(xs, raw, pos)
    checks.add(".X compared with .raw.X on the 3,000 genes", f"max abs difference {dx_max:g}; differing entries "
               f"{dx_n}", None)
    row_tot = np.asarray(raw.sum(axis=1), dtype=np.float64).ravel()
    row_nnz = np.diff(raw.indptr)
    d_tot = float(np.max(np.abs(row_tot - obs["total_counts"].to_numpy(np.float64))))
    d_nnz = int(np.max(np.abs(row_nnz - obs["n_genes_by_counts"].to_numpy(np.int64))))
    checks.add("obs total_counts vs .raw.X row sums; obs n_genes_by_counts vs row non-zeros (max abs difference)",
               f"{d_tot:g}; {d_nnz}", None)
    checks.add("cells with zero .raw.X total", f"{int((row_tot == 0).sum())}", bool((row_tot > 0).all()))
    raw_var = a.raw.var.copy()
    raw_var.index = genes
    in_hvg = genes.isin(hvg)
    del a, xs, xd
    gc.collect()

    # 3. pseudobulk ----------------------------------------------------------------------------------------
    keys = list(L.GROUP_KEYS)
    gobs = obs[keys].astype(str)
    gid, uniq = pd.factorize(pd.MultiIndex.from_frame(gobs), sort=True)
    groups = uniq.to_frame(index=False)
    groups.columns = keys
    n_groups = len(groups)
    groups["n_cells"] = np.bincount(gid, minlength=n_groups)
    checks.add("pseudobulk groups (Source x Technology x Celltype x Celltype_finest_lowres x Celltype_finest)",
               f"{n_groups} groups; {int(groups['n_cells'].sum())} cells", int(groups["n_cells"].sum()) == len(obs))

    order = groups.sort_values("n_cells", ascending=False, kind="mergesort").index.to_numpy()
    spot = [order[0], order[len(order) // 2], [g for g in order if groups.loc[g, "n_cells"] >= 5][-1]]
    spot_raw = {g: raw[np.flatnonzero(gid == g)].copy() for g in spot}

    col_nnz = column_nnz(raw)
    print("[atlas] group sums of counts", flush=True)
    sum_counts = group_sum(gid, n_groups, raw)
    col_sum = group_sum(np.zeros(len(gid), dtype=np.int64), 1, raw).ravel()
    print("[atlas] normalizing (10,000 per cell, log1p) in place", flush=True)
    rowwise_inplace(raw, (L.TARGET_SUM / row_tot).astype(np.float32), lambda d: np.log1p(d, out=d))
    sum_lognorm = group_sum(gid, n_groups, raw)
    rowwise_inplace(raw, None, lambda d: np.square(d, out=d))
    sum_sq = group_sum(gid, n_groups, raw)
    rowwise_inplace(raw, None, lambda d: d.fill(1.0))
    n_det = group_sum(gid, n_groups, raw)
    del raw
    gc.collect()

    checks.add("pseudobulk n_detected summed over groups equals the .raw.X column non-zeros",
               f"max abs difference {float(np.max(np.abs(n_det.sum(axis=0) - col_nnz))):g}",
               np.array_equal(n_det.sum(axis=0).astype(np.int64), col_nnz))
    rel_cs = float(np.max(np.abs(sum_counts.sum(axis=0) - col_sum) / np.maximum(col_sum, 1)))
    checks.add("pseudobulk sum_counts summed over groups equals the .raw.X column sums (max relative difference)",
               f"{rel_cs:.2e}", rel_cs < 1e-6)
    for g in spot:
        sub = ad.AnnData(spot_raw[g].astype(np.float32))
        sc.pp.normalize_total(sub, target_sum=L.TARGET_SUM)
        sc.pp.log1p(sub)
        xm = (sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)).astype(np.float64)
        n = groups.loc[g, "n_cells"]
        d_mean = float(np.max(np.abs(xm.mean(axis=0) - sum_lognorm[g] / n)))
        d_sq = float(np.max(np.abs((xm ** 2).mean(axis=0) - sum_sq[g] / n)))
        d_det = float(np.max(np.abs((spot_raw[g] > 0).toarray().mean(axis=0) - n_det[g] / n)))
        label = " / ".join(str(groups.loc[g, k]) for k in keys)
        checks.add(f"independent scanpy normalization reproduces the pseudobulk: {label} ({n} cells); max abs "
                   "difference of mean, mean square, detection", f"{d_mean:.1e}; {d_sq:.1e}; {d_det:.1e}",
                   max(d_mean, d_sq, d_det) < SPOT_CHECK_TOL, f"< {SPOT_CHECK_TOL:g}")
    del spot_raw

    # 4. shared features and marker coverage -----------------------------------------------------------------
    match = L.match_proteins(prot, genes)
    checks.add("atlas genes matching a quantified protein (case-insensitive; protein names without '_' / 'Nan')",
               f"{len(match)} of {len(genes)} genes; {len(prot)} proteins", len(match) == L.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES,
               f"= {L.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES} (kidney_annotation step 3a)")
    tot_n = len(obs)
    prot_set = set(prot)
    det_by_source = {}
    for s_ in sorted(groups["Source"].unique()):
        m = groups["Source"].to_numpy() == s_
        det_by_source[s_] = n_det[m].sum(axis=0) / groups.loc[m, "n_cells"].sum()
    gpos = match["atlas_position"].to_numpy()
    shared = pd.DataFrame({
        "protein": match["protein"], "atlas_gene": match["atlas_gene"],
        "scp_n_cells_detected": prot_ndet.reindex(match["protein"]).to_numpy(),
        "scp_detection_rate": prot_det.reindex(match["protein"]).to_numpy(),
        "atlas_detection_rate_all": n_det.sum(axis=0)[gpos] / tot_n,
        **{f"atlas_detection_rate_{s_}": det_by_source[s_][gpos] for s_ in det_by_source},
        "atlas_mean_lognorm_all": sum_lognorm.sum(axis=0)[gpos] / tot_n,
        "in_atlas_X_3000_genes": in_hvg[gpos],
        "marker_listed_for": [";".join(by_gene.get(p, [])) for p in match["protein"]],
    })
    glook = L.atlas_gene_lookup(genes)
    plook = L.protein_lookup(prot)
    cov = []
    for short, full, mk in marker_rows:
        for pos_in_row, g in enumerate(mk, start=1):
            ag = glook.get(g.lower(), "")
            gi = genes.get_loc(ag) if ag else None
            cov.append({"type": short, "type_full_name": full, "marker": g, "workbook_position": pos_in_row,
                        "in_protein_matrix": g in prot_set, "protein_name": plook.get(g.lower(), ""),
                        "in_atlas_raw": bool(ag), "atlas_gene": ag,
                        "in_atlas_X_3000_genes": bool(ag) and bool(in_hvg[gi]),
                        "atlas_detection_rate_all": (n_det[:, gi].sum() / tot_n) if ag else np.nan,
                        **{f"atlas_detection_rate_{s_}": (det_by_source[s_][gi] if ag else np.nan)
                           for s_ in det_by_source},
                        "listed_for": ";".join(by_gene[g])})
    cov = pd.DataFrame(cov)
    n_unique = cov.drop_duplicates("marker")
    checks.add("V3 + endothelial-row markers (unique): in protein matrix; in atlas .raw; in both",
               f"{len(n_unique)}; {int(n_unique['in_protein_matrix'].sum())}; {int(n_unique['in_atlas_raw'].sum())}; "
               f"{int((n_unique['in_protein_matrix'] & n_unique['in_atlas_raw']).sum())}", None)

    # pseudobulk object ---------------------------------------------------------------------------------------
    var = pd.DataFrame(index=genes)
    var["gene_id"] = raw_var["gene_id-1"].astype(str).to_numpy() if "gene_id-1" in raw_var.columns else ""
    var["kidney_protein"] = ""
    var.loc[match["atlas_gene"].to_numpy(), "kidney_protein"] = match["protein"].to_numpy()
    var["in_atlas_X_3000_genes"] = in_hvg
    gobs_out = groups.copy()
    gobs_out.index = [f"g{i:03d}" for i in range(n_groups)]
    pb = ad.AnnData(X=sum_lognorm.astype(np.float32), obs=gobs_out, var=var)
    pb.layers["sum_sq_lognorm"] = sum_sq.astype(np.float32)
    pb.layers["n_detected"] = n_det.astype(np.float32)
    pb.layers["sum_counts"] = sum_counts.astype(np.float32)
    pb.uns["kidney_rna"] = {
        "generated_by": cc.GENERATED_BY, "script": "scripts/kidney_rna_atlas_summary.py",
        "created_utc": cc.now_utc(), "source": "mka.h5ad", "source_sha256": run.inputs[cc.rel(L.ATLAS)]["sha256"],
        "matrix": ".raw.X (integer counts, 16,711 genes)",
        "normalization": "per cell: counts / total counts x 10,000, then natural log1p (all 16,711 genes)",
        "group_keys": keys, "X": "sum over the group's cells of log-normalized expression",
        "layers": "sum_sq_lognorm: sum of squares; n_detected: cells with counts > 0; sum_counts: sum of counts",
        "precision": "sums computed in float64, stored as float32",
    }
    path = run.out("h5ad", "atlas_pseudobulk", ".h5ad")
    pb.write_h5ad(path, compression="gzip")
    print(f"Saved AnnData: {cc.rel(path)}")
    checks.add("pseudobulk object written (groups x genes; file size)", f"{pb.n_obs} x {pb.n_vars}; "
               f"{path.stat().st_size / 1e6:.1f} MB", None)

    run.table(obs_columns(obs), "atlas_obs_columns")
    run.table(obs_levels(obs), "atlas_obs_levels")
    run.table(study_design(obs), "atlas_study_design")
    run.table(source_technology(obs), "atlas_source_technology")
    run.table(level_counts(obs), "atlas_level_counts")
    run.table(level_crosstab(obs), "atlas_level_combinations")
    run.table(gobs_out.rename_axis("group_id").reset_index(), "atlas_pseudobulk_groups")
    run.table(shared, "shared_features")
    run.table(cov, "marker_coverage")
    run.table(checks.frame(), "atlas_summary_checks")
    run.finish({"target_sum": L.TARGET_SUM, "group_keys": keys, "spot_check_tolerance": SPOT_CHECK_TOL,
                "symbol_matching": "case-insensitive; protein names containing '_' and 'Nan' excluded (kidney_annotation step 3a)",
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see kidney_rna_atlas_summary_checks.csv.")


if __name__ == "__main__":
    main()
