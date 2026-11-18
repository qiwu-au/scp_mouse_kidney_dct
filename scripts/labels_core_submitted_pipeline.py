#!/usr/bin/env python
"""labels_core step 1: the submitted DCT pipeline, unchanged, on the new control data.

Analysis plan, step 1.
Settings: labels_core_lib.SUBMITTED_SETTINGS (each with its line in scp_dct_upload.py); seed 42 for Leiden and UMAP.

  float32 intensities -> log1p -> scale(max_value=10) -> neighbours (k = 10, 20 PCs; Scanpy PCA fallback)
  -> Leiden 3.0 (igraph) -> sc.tl.score_genes per DCT V1 workbook type on the markers present (on the scaled .X,
  as the submitted object, whose .raw aliased .X) -> per-cell argmax (ctrl_cell_label) -> cluster mode with the original Unknown rule (ctrl_cluster_label,
  ctrl_leiden3_cluster) -> arpack PCA -> UMAP -> DPT from the first DCT1 cell and from 20 random DCT1 roots
  -> PAGA on ctrl_cell_label.

Inputs (read-only): results/h5ad/intake_core_DCT_ctrl.h5ad; data/DCT/DCT_markers_V1.xlsx; scp_dct_upload.py (reference,
hashed only).
Outputs: results/tables/labels_core/labels_core_submitted_*.csv;
  results/h5ad/labels_core_DCT_ctrl_step1_submitted.h5ad (working object for steps 2-6: scaled .X,
  layers['intensity'], graph,
  neighbour PCA, arpack PCA, UMAP, diffusion map, PAGA, step-1 labels and scores).

Command (project root):  python scripts/labels_core_submitted_pipeline.py
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
import scanpy as sc
from scipy import stats

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import labels_core_lib as lib

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "labels_core_submitted_pipeline.py"
REFERENCE_SCRIPT = cc.PROJECT_ROOT / "first_submission" / "scp_dct_upload.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="labels_core step 1: submitted DCT pipeline on the control data.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "submitted_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see labels_core_submitted_checks.csv.")


def marker_coverage(workbook: dict[str, list[str]], lists: dict[str, list[str]], ctrl, markers: t5.MarkerSets) -> pd.DataFrame:
    detected = cc.dense(ctrl.layers["intensity"]) > 0
    var = {g: i for i, g in enumerate(ctrl.var_names.astype(str))}
    rows = []
    for t, genes in workbook.items():
        for g in genes:
            n = int(detected[:, var[g]].sum()) if g in var else 0
            rows.append({"cell_type": t, "marker": g, "in_matrix": g in var, "scored": g in lists.get(t, []),
                         "n_cells_detected": n, "fraction_cells_detected": n / ctrl.n_obs,
                         "types_listing_in_workbook": ";".join(sorted(t2 for t2, l2 in workbook.items() if g in l2)),
                         "shared_in_workbook": sum(g in l2 for l2 in workbook.values()) >= 2,
                         "shared_for_de_rule": markers.is_shared(g),
                         "note": lib.TREATED_AS_SHARED.get(g, "")})
    return pd.DataFrame(rows)


def dpt_roots(adata, cell_label: pd.Series, run: cc.Run) -> tuple[pd.DataFrame, np.ndarray]:
    """DPT from the original root (first DCT1 cell) and from N_RANDOM_ROOTS random DCT1 cells (seed 42)."""
    dct1 = np.where(cell_label.to_numpy() == "DCT1")[0]
    if len(dct1) == 0:
        raise SystemExit("No DCT1 cell: the original DPT root rule cannot be applied.")
    original = int(dct1[0])
    rng = np.random.default_rng(lib.SEED)
    others = rng.choice(dct1[dct1 != original], size=min(lib.N_RANDOM_ROOTS, len(dct1) - 1), replace=False)
    rows, ref = [], None
    for k, root in enumerate([original, *map(int, others)]):
        adata.uns["iroot"] = int(root)
        sc.tl.dpt(adata)
        pt = adata.obs["dpt_pseudotime"].to_numpy(float).copy()
        if k == 0:
            ref = pt
        finite = np.isfinite(pt) & np.isfinite(ref)
        row = {"root": "original (first DCT1 cell)" if k == 0 else f"random_{k:02d}", "root_index": int(root),
               "root_cell_id": str(adata.obs_names[root]), "root_plate": str(adata.obs["plate"].iloc[root]),
               "n_finite": int(np.isfinite(pt).sum()),
               "spearman_rho_vs_original": float(stats.spearmanr(pt[finite], ref[finite]).statistic)}
        for label in lib.CORE_TYPES:
            m = (cell_label.to_numpy() == label) & np.isfinite(pt)
            row[f"median_dpt_{label}"] = float(np.median(pt[m])) if m.any() else np.nan
        rows.append(row)
    adata.uns["iroot"] = original
    sc.tl.dpt(adata)
    run.note(f"DPT root (original rule): cell index {original} ({adata.obs_names[original]}); "
             f"{len(others)} random DCT1 roots drawn with numpy default_rng({lib.SEED}).")
    return pd.DataFrame(rows), adata.obs["dpt_pseudotime"].to_numpy(float)


def paga_edges(adata, key: str) -> pd.DataFrame:
    groups = adata.obs[key].cat.categories.tolist()
    conn = cc.dense(adata.uns["paga"]["connectivities"])
    src, tgt = np.triu_indices(len(groups), k=1)
    edges = pd.DataFrame({"source": [groups[i] for i in src], "target": [groups[j] for j in tgt],
                          "connectivity": [float(conn[i, j]) for i, j in zip(src, tgt)]})
    return edges[edges["connectivity"] > 0].sort_values("connectivity", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "submitted_pipeline", args)
    t_start = time.time()
    checks = t5.Checks("dct_ctrl")
    run.use_input(REFERENCE_SCRIPT, "submitted DCT pipeline (reference code; not run)")
    ctrl = lib.load_ctrl(run)
    intensity = np.array(cc.dense(ctrl.layers["intensity"]), dtype=np.float32, copy=True)   # untouched reference
    x_t17 = np.array(cc.dense(ctrl.X), dtype=np.float64, copy=True)
    workbook = lib.read_workbook(run)

    print("[ctrl] input checks", flush=True)
    checks.add("intake_core control object: cells x proteins", f"{ctrl.n_obs} x {ctrl.n_vars}",
               (ctrl.n_obs, ctrl.n_vars) == (727, 6196), "727 x 6196 (intake_core README)")
    checks.add("all cells are control cells", lib.counts_text(ctrl.obs["condition"], ("Ctrl",)),
               set(ctrl.obs["condition"].astype(str)) == {"Ctrl"})
    # Marker matching under the script's own naming (first PG.Genes symbol, lower().capitalize()).
    script_names = set(ctrl.var["gene_name"].astype(str).str.lower().str.capitalize())
    t17_names = set(ctrl.var_names.astype(str))
    diff = [g for genes in workbook.values() for g in genes if (g in script_names) != (g in t17_names)]
    checks.add("workbook markers found identically under the script's naming and the intake_core var_names",
               f"{len(diff)} differing{': ' + ';'.join(diff) if diff else ''}", not diff)

    # Script lines 75-112
    adata = lib.submitted_object(ctrl)
    lay_eq = np.array_equal(cc.dense(adata.layers["intensity"]), intensity)
    checks.add("layers['intensity'] equals the intake_core intensity layer (copy taken before the pipeline)", f"{lay_eq}", lay_eq)
    shared = bool(np.shares_memory(cc.dense(adata.layers["intensity"]), cc.dense(adata.X))
                  or np.shares_memory(cc.dense(ctrl.layers["intensity"]), cc.dense(adata.X)))
    checks.add(".X shares no memory with the intensity layers", f"shares memory: {shared}", not shared)
    run.note("scp_dct_upload.py sets adata.raw = adata without a copy (line 106), so its in-place log1p and scale also "
             "reach .raw: in the submitted DCT object .raw.X equals the scaled .X (kidney_object), and kidney_object/reviewer_core reproduced the "
             "stored scores exactly by scoring the scaled .X. labels_core scores the scaled .X and keeps the intensities in "
             "layers['intensity']. (scp_kidney_upload.py copies .raw before log1p; the DCT script does not.)")
    run.note("Two earlier runs of this step were discarded and replaced with --overwrite: the first aliased the intake_core "
             "intensity layer in memory (wrong detection fractions in the marker coverage table), the second scored "
             "the pre-log intensities (kidney-script behaviour, not the submitted DCT behaviour).")
    d_x = float(np.abs(cc.dense(adata.X).astype(np.float64) - x_t17).max())
    checks.add("scaled .X equals the intake_core .X (z-scored ln1p, ddof 1, clip 10)", f"max abs diff {d_x:.3g}", d_x < 1e-4,
               "< 1e-4 (float32)")
    n_clip = int((np.abs(cc.dense(adata.X)) >= 10 - 1e-6).sum())
    checks.add("scaled .X entries at the +/-10 clip", f"{n_clip}", None)
    if not checks.ok:
        stop(run, checks)

    # Script line 115: neighbours without a prior PCA (Scanpy computes the PCA itself)
    had_pca = "X_pca" in adata.obsm
    sc.pp.neighbors(adata, n_neighbors=lib.N_NEIGHBORS, n_pcs=lib.N_PCS)
    pca_neighbors = np.asarray(adata.obsm["X_pca"]).copy() if "X_pca" in adata.obsm else None
    pca_params = {k: str(v) for k, v in adata.uns.get("pca", {}).get("params", {}).items()}
    checks.add("no PCA before sc.pp.neighbors (as the script)", f"X_pca before: {had_pca}", not had_pca)
    checks.add("PCA computed by the neighbours fallback (components; params)",
               f"{None if pca_neighbors is None else pca_neighbors.shape[1]}; {pca_params}", None)
    run.note(f"sc.pp.neighbors fallback PCA: {None if pca_neighbors is None else pca_neighbors.shape} {pca_params}")

    # Script lines 118-120: Leiden 3.0 (seed 42)
    clusters = t5.leiden(adata, lib.SUBMITTED_RESOLUTION, lib.LEIDEN_KEY, seed=lib.SEED)

    # Script lines 123-143: scores and per-cell labels
    lists = lib.present_lists(workbook, adata.var_names)
    missing = {t: [g for g in genes if g not in lists.get(t, [])] for t, genes in workbook.items()}
    run.note("Workbook markers absent from the control matrix: "
             + "; ".join(f"{t}: {', '.join(g) or 'none'}" for t, g in missing.items()))
    scores = lib.score_types(adata, lists)
    scores_bare = lib.score_matrix(cc.dense(adata.X), adata.var_names, adata.obs_names, lists)
    d_s = float(np.abs(scores.to_numpy() - scores_bare.to_numpy()).max())
    checks.add("score_genes on the object's .X equals score_genes on a bare copy of .X (noise test A basis)",
               f"max abs diff {d_s:.3g}", d_s == 0, "= 0")
    # Information only: the same scoring on the pre-log intensities (what a copied .raw would have given).
    alt_label, _ = lib.cell_labels(lib.score_matrix(intensity, adata.var_names, adata.obs_names, lists))
    checks.add("no missing scores", f"{int(scores.isna().sum().sum())}", not scores.isna().any().any())
    cell_label, max_score = lib.cell_labels(scores)
    n_ties = int((scores.eq(max_score, axis=0).sum(axis=1) > 1).sum())
    checks.add("cells with tied maximum scores (argmax takes the first type)", f"{n_ties}", None)
    adata.obs["predicted_cell_type"] = cell_label.to_numpy()
    adata.obs["max_score"] = max_score.to_numpy()

    # Script lines 145-171: cluster labels
    named, naming = lib.cluster_labels(cell_label, max_score, clusters)
    adata.obs[lib.NAMED_KEY] = pd.Categorical(named.to_numpy())
    unknown = naming[naming["cluster_label"] == lib.UNKNOWN]
    checks.add("Unknown clusters (original rule): cluster (cells; mean max score; threshold)",
               "; ".join(f"{r.cluster} ({r.n_cells}; {r.mean_max_score:.4g}; < {r.score_threshold:.4g})"
                         for r in unknown.itertuples()) or "none", None)
    checks.add("Leiden 3.0 clusters with a tied mode", f"{int(naming['mode_tied'].sum())} of {len(naming)}", None)

    # Script lines 178-179: arpack PCA (overwrites X_pca) and UMAP on the stored graph
    sc.tl.pca(adata, svd_solver="arpack")
    if pca_neighbors is not None:
        adata.obsm["X_pca_neighbors"] = pca_neighbors
    sc.tl.umap(adata, random_state=lib.SEED)

    # Script lines 272-281: DPT; then random DCT1 roots
    roots, dpt = dpt_roots(adata, cell_label, run)
    checks.add("DPT pseudotime finite (original root)", f"{int(np.isfinite(dpt).sum())} of {adata.n_obs}", None)

    # Script line 325: PAGA on the per-cell label (categorical, alphabetical as Scanpy's sanitize step)
    adata.obs["predicted_cell_type"] = pd.Categorical(cell_label.to_numpy(),
                                                      categories=sorted(set(cell_label.astype(str))))
    sc.tl.paga(adata, groups="predicted_cell_type")
    edges = paga_edges(adata, "predicted_cell_type")

    # labels_core columns
    adata.obs["ctrl_cell_label"] = adata.obs["predicted_cell_type"].astype(str).to_numpy()
    adata.obs["ctrl_leiden3_cluster"] = adata.obs[lib.LEIDEN_KEY].astype(str).to_numpy()
    adata.obs["ctrl_cluster_label"] = adata.obs[lib.NAMED_KEY].astype(str).to_numpy()
    for col in ("ctrl_cell_label", "ctrl_cluster_label"):
        n_na = int(adata.obs[col].isna().sum())
        checks.add(f"{col}: every cell labelled; counts", f"{n_na} missing; {lib.counts_text(adata.obs[col])}",
                   n_na == 0)
    agree = float((adata.obs["ctrl_cell_label"] == adata.obs["ctrl_cluster_label"]).mean())
    checks.add("ctrl_cell_label == ctrl_cluster_label (fraction of cells)", f"{agree:.3f}", None)
    untouched = np.array_equal(cc.dense(ctrl.layers["intensity"]), intensity) and np.array_equal(
        cc.dense(adata.layers["intensity"]), intensity)
    checks.add("intake_core intensity layer and layers['intensity'] unchanged after the whole pipeline", f"{untouched}",
               untouched)
    alt_same = float((alt_label.to_numpy() == adata.obs["ctrl_cell_label"].to_numpy()).mean())
    checks.add("information only: per-cell argmax if scored on the pre-log intensities (counts; fraction equal to "
               "ctrl_cell_label)", f"{lib.counts_text(alt_label)}; {alt_same:.3f}", None)
    if not checks.ok:
        stop(run, checks)

    per_cell = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str), "plate": adata.obs["plate"].astype(str).to_numpy(),
        "well": adata.obs["well"].astype(str).to_numpy(), "n_detected": adata.obs["n_detected"].to_numpy(),
        "total_intensity": adata.obs["total_intensity"].to_numpy(),
        **{f"{t}_score": scores[t].to_numpy() for t in lists}, "max_score": max_score.to_numpy(),
        "max_score_tied": (scores.eq(max_score, axis=0).sum(axis=1) > 1).to_numpy(),
        "ctrl_cell_label": adata.obs["ctrl_cell_label"].to_numpy(),
        "info_cell_label_scored_on_intensity": alt_label.to_numpy(),
        "ctrl_leiden3_cluster": adata.obs["ctrl_leiden3_cluster"].to_numpy(),
        "ctrl_cluster_label": adata.obs["ctrl_cluster_label"].to_numpy(),
        "dpt_pseudotime": dpt, "umap1": adata.obsm["X_umap"][:, 0], "umap2": adata.obsm["X_umap"][:, 1]})
    counts = []
    for col in ("ctrl_cell_label", "ctrl_cluster_label"):
        vc = per_cell[col].value_counts()
        for label in cc.ordered_labels(vc.index, lib.LABEL_ORDER):
            counts.append({"label_column": col, "label": label, "n_cells": int(vc[label]),
                           "fraction": float(vc[label] / len(per_cell))})
    crosstab = pd.crosstab(per_cell["ctrl_cell_label"], per_cell["ctrl_cluster_label"], margins=True,
                           margins_name="total")
    settings = pd.DataFrame(lib.SUBMITTED_SETTINGS, columns=["setting", "value_used", "scp_dct_upload_py_lines", "note"])
    settings.loc[len(settings)] = ["neighbour PCA (observed)", f"{None if pca_neighbors is None else pca_neighbors.shape[1]}"
                                   f" components; {pca_params}", "115", "Scanpy fallback"]
    settings.loc[len(settings)] = ["markers scored", "; ".join(f"{t}: {', '.join(g)}" for t, g in lists.items()),
                                   "123-134", "present in the control matrix"]
    settings.loc[len(settings)] = ["workbook markers absent", "; ".join(f"{t}: {', '.join(g) or 'none'}"
                                                                        for t, g in missing.items()), "129", ""]

    adata.uns["labels_core_step1"] = {"script": f"scripts/{SCRIPT}", "seed": lib.SEED,
                                     "leiden_resolution": lib.SUBMITTED_RESOLUTION,
                                     "markers_scored": "; ".join(f"{t}: {','.join(g)}" for t, g in lists.items())}
    for col in ("predicted_cell_type", lib.NAMED_KEY, "plate"):
        adata.obs[col] = adata.obs[col].astype("category")

    run.table(checks.frame(), "submitted_checks")
    run.table(settings, "submitted_settings")
    run.table(marker_coverage(workbook, lists, ctrl, lib.dct_markers(run, ctrl.var_names)), "submitted_marker_coverage")
    run.table(per_cell, "submitted_per_cell")
    run.table(pd.DataFrame(counts), "submitted_label_counts")
    run.table(crosstab, "submitted_cell_vs_cluster_crosstab", index=True)
    run.table(naming, "submitted_cluster_naming")
    run.table(edges, "submitted_paga_connectivities")
    run.table(roots, "submitted_dpt_roots")
    run.h5ad(adata, "DCT_ctrl_step1_submitted")
    run.finish({"settings": [dict(zip(settings.columns, r)) for r in settings.itertuples(index=False)],
                "n_random_dpt_roots": lib.N_RANDOM_ROOTS, "scanpy": sc.__version__,
                "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "runtime_s": round(time.time() - t_start, 1),
                "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
