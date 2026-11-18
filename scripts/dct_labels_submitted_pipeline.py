#!/usr/bin/env python
"""dct_labels step 1: the submitted DCT pipeline, unchanged, on each condition of the combined search.

Analysis plan, step 1
(repeats labels_core step 1 per condition; settings labels_core_lib.SUBMITTED_SETTINGS).

  per condition (dct_intake object, own scaling): float32 intensities -> log1p -> scale(max_value=10) -> neighbours
  (k = 10, 20 PCs; Scanpy PCA fallback) -> Leiden 3.0 (igraph, seed 42) -> sc.tl.score_genes per DCT V1 workbook
  type on the markers present (scaled .X) -> per-cell argmax (cs_<cond>_cell_label) -> cluster mode with the
  original Unknown rule (cs_<cond>_cluster_label, cs_<cond>_leiden3_cluster) -> arpack PCA -> UMAP -> DPT from
  the first DCT1 cell (object order: P1 then P2, ascending well) and from 20 random DCT1 roots -> PAGA.

Inputs (read-only): results/h5ad/dct_intake_DCT_ctrl.h5ad, dct_intake_DCT_zerok.h5ad; data/DCT/DCT_markers_V1.xlsx;
results/tables/labels_core/labels_core_submitted_marker_coverage.csv (marker-set comparison only).
Outputs: results/tables/dct_labels/dct_labels_submitted_*.csv;
  results/h5ad/dct_labels_DCT_{ctrl,zerok}_step1_submitted.h5ad (working objects for steps 2-8).

Command (project root):  python scripts/dct_labels_submitted_pipeline.py
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

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import labels_core_lib as t18
import dct_labels_lib as lib
from labels_core_submitted_pipeline import marker_coverage

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "dct_labels_submitted_pipeline.py"
EXPECTED_CELLS = {"Ctrl": 729, "ZeroK": 702}     # dct_intake README


def one_condition(run: cc.Run, checks: t5.Checks, cond: str, workbook: dict[str, list[str]]):
    src = lib.read_t26(run, cond)
    key = lib.KEY[cond]
    intensity = np.array(cc.dense(src.layers["intensity"]), dtype=np.float32, copy=True)
    x_t26 = np.array(cc.dense(src.X), dtype=np.float64, copy=True)
    print(f"[{key}] input checks", flush=True)
    checks.add(f"{cond}: dct_intake object cells x proteins", f"{src.n_obs} x {src.n_vars}",
               src.n_obs == EXPECTED_CELLS[cond], f"{EXPECTED_CELLS[cond]} cells (dct_intake README)")
    checks.add(f"{cond}: all cells are {cond} cells", lib.counts_text(src.obs["condition"], (cond,)),
               set(src.obs["condition"].astype(str)) == {cond})
    order = src.obs[["plate", "well"]].astype(str).sort_values(["plate", "well"]).index
    in_order = list(order) == list(src.obs_names)
    checks.add(f"{cond}: object order is P1 then P2, ascending well (DPT root rule)", f"{in_order}", in_order)
    script_names = set(src.var["gene_name"].astype(str).str.lower().str.capitalize())
    names = set(src.var_names.astype(str))
    diff = [g for genes in workbook.values() for g in genes if (g in script_names) != (g in names)]
    checks.add(f"{cond}: workbook markers found identically under the script's naming and the dct_intake var_names",
               f"{len(diff)} differing{': ' + ';'.join(diff) if diff else ''}", not diff)
    if not checks.ok:
        lib.stop(run, checks, "submitted_checks")

    res = lib.submitted_pipeline(src, workbook, run)
    adata = res["adata"]
    lay_eq = np.array_equal(cc.dense(adata.layers["intensity"]), intensity)
    checks.add(f"{cond}: layers['intensity'] equals the dct_intake intensity layer", f"{lay_eq}", lay_eq)
    d_x = float(np.abs(cc.dense(adata.X).astype(np.float64) - x_t26).max())
    checks.add(f"{cond}: scaled .X equals the dct_intake .X (z-scored ln1p within the condition, ddof 1, clip 10)",
               f"max abs diff {d_x:.3g}", d_x < 1e-4, "< 1e-4 (float32)")
    checks.add(f"{cond}: scaled .X entries at the +/-10 clip", f"{int((np.abs(cc.dense(adata.X)) >= 10 - 1e-6).sum())}",
               None)
    checks.add(f"{cond}: PCA computed by the neighbours fallback (components; params)",
               f"{res['pca_neighbors_shape'][1]}; {res['pca_params']}", None)
    d_s = float(np.abs(res["scores"].to_numpy() - res["scores_bare"].to_numpy()).max())
    checks.add(f"{cond}: score_genes on the object's .X equals score_genes on a bare copy of .X",
               f"max abs diff {d_s:.3g}", d_s == 0, "= 0")
    checks.add(f"{cond}: no missing scores", f"{int(res['scores'].isna().sum().sum())}",
               not res["scores"].isna().any().any())
    scores, max_score = res["scores"], res["max_score"]
    n_ties = int((scores.eq(max_score, axis=0).sum(axis=1) > 1).sum())
    checks.add(f"{cond}: cells with tied maximum scores (argmax takes the first type)", f"{n_ties}", None)
    naming = res["naming"]
    unknown = naming[naming["cluster_label"] == lib.UNKNOWN]
    checks.add(f"{cond}: Unknown clusters (original rule): cluster (cells; mean max score; threshold)",
               "; ".join(f"{r.cluster} ({r.n_cells}; {r.mean_max_score:.4g}; < {r.score_threshold:.4g})"
                         for r in unknown.itertuples()) or "none", None)
    checks.add(f"{cond}: Leiden 3.0 clusters with a tied mode", f"{int(naming['mode_tied'].sum())} of {len(naming)}",
               None)
    dpt = res["dpt"]
    checks.add(f"{cond}: DPT pseudotime finite (original root)", f"{int(np.isfinite(dpt).sum())} of {adata.n_obs}", None)
    cell_label = res["cell_label"].astype(str).to_numpy()
    cluster_label = res["cluster_label"].astype(str).to_numpy()
    for col, values in (("cell_label", cell_label), ("cluster_label", cluster_label)):
        n_na = int(pd.isna(values).sum())
        checks.add(f"{cond}: {lib.cs(cond, col)} every cell labelled; counts", f"{n_na} missing; "
                   f"{lib.counts_text(values)}", n_na == 0)
    untouched = np.array_equal(cc.dense(src.layers["intensity"]), intensity)
    checks.add(f"{cond}: dct_intake intensity layer unchanged after the whole pipeline", f"{untouched}", untouched)
    checks.add(f"{cond}: information only: per-cell argmax if scored on the pre-log intensities (counts; fraction "
               f"equal to {lib.cs(cond, 'cell_label')})",
               f"{lib.counts_text(res['alt_label'])}; {float((res['alt_label'].to_numpy() == cell_label).mean()):.3f}",
               None)

    # object columns (canonical names; the object holds one condition)
    adata.obs["condition"] = cond
    adata.obs["cell_key"] = src.obs["cell_key"].astype(str).to_numpy()
    adata.obs[lib.cs(cond, "cell_label")] = cell_label
    adata.obs[lib.cs(cond, "leiden3_cluster")] = res["clusters"].astype(str).to_numpy()
    adata.obs[lib.cs(cond, "cluster_label")] = cluster_label
    adata.uns["dct_labels_step1"] = {"script": f"scripts/{SCRIPT}", "seed": lib.SEED, "condition": cond,
                                     "leiden_resolution": t18.SUBMITTED_RESOLUTION,
                                     "markers_scored": "; ".join(f"{t}: {','.join(g)}" for t, g in res["lists"].items())}
    for col in ("predicted_cell_type", t18.NAMED_KEY, "plate", "condition"):
        adata.obs[col] = adata.obs[col].astype("category")

    lists = res["lists"]
    per_cell = pd.DataFrame({
        "condition": cond, "cell_id": adata.obs_names.astype(str), "cell_key": adata.obs["cell_key"].to_numpy(),
        "plate": adata.obs["plate"].astype(str).to_numpy(), "well": adata.obs["well"].astype(str).to_numpy(),
        "n_detected": adata.obs["n_detected"].to_numpy(), "total_intensity": adata.obs["total_intensity"].to_numpy(),
        **{f"{t}_score": scores[t].to_numpy() for t in lists}, "max_score": max_score.to_numpy(),
        "max_score_tied": (scores.eq(max_score, axis=0).sum(axis=1) > 1).to_numpy(),
        "cell_label": cell_label, "info_cell_label_scored_on_intensity": res["alt_label"].to_numpy(),
        "leiden3_cluster": res["clusters"].astype(str).to_numpy(), "cluster_label": cluster_label,
        "dpt_pseudotime": dpt, "umap1": adata.obsm["X_umap"][:, 0], "umap2": adata.obsm["X_umap"][:, 1]})
    counts = []
    for col in ("cell_label", "cluster_label"):
        vc = per_cell[col].value_counts()
        for label in cc.ordered_labels(vc.index, lib.LABEL_ORDER):
            counts.append({"condition": cond, "label_column": lib.cs(cond, col), "label": label,
                           "n_cells": int(vc[label]), "fraction": float(vc[label] / len(per_cell))})
    cross = [{"condition": cond, **r} for r in lib.crosstab_long(cell_label, cluster_label, "cell_label",
                                                                    "cluster_label")]
    coverage = marker_coverage(workbook, lists, src, t18.dct_markers(run, src.var_names))
    coverage.insert(0, "condition", cond)
    missing = res["missing"]
    settings = pd.DataFrame(t18.SUBMITTED_SETTINGS, columns=["setting", "value_used", "scp_dct_upload_py_lines", "note"])
    settings.loc[settings["setting"] == "cell filter", "note"] = "applied in dct_intake"
    settings.loc[settings["setting"] == "protein filter", "note"] = "applied in dct_intake (per condition)"
    settings.loc[len(settings)] = ["neighbour PCA (observed)", f"{res['pca_neighbors_shape'][1]} components; "
                                   f"{res['pca_params']}", "115", "Scanpy fallback"]
    settings.loc[len(settings)] = ["markers scored", "; ".join(f"{t}: {', '.join(g)}" for t, g in lists.items()),
                                   "123-134", f"present in the {cond} matrix"]
    settings.loc[len(settings)] = ["workbook markers absent", "; ".join(f"{t}: {', '.join(g) or 'none'}"
                                                                        for t, g in missing.items()), "129", ""]
    settings.insert(0, "condition", cond)
    naming = naming.copy()
    naming.insert(0, "condition", cond)
    edges = res["edges"].copy()
    edges.insert(0, "condition", cond)
    roots = res["roots"].copy()
    roots.insert(0, "condition", cond)
    run.h5ad(adata, f"DCT_{key}_step1_submitted")
    return {"per_cell": per_cell, "counts": counts, "cross": cross, "coverage": coverage, "settings": settings,
            "naming": naming, "edges": edges, "roots": roots, "lists": lists}


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_labels step 1: submitted DCT pipeline per condition.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "submitted_pipeline", args)
    t_start = time.time()
    checks = t5.Checks("dct_combined")
    workbook = t18.read_workbook(run)
    out = {cond: one_condition(run, checks, cond, workbook) for cond in lib.CONDITIONS}

    # Marker sets per condition, against each other and against labels_core (separate Ctrl search)
    sets = {cond: {t: set(g) for t, g in out[cond]["lists"].items()} for cond in lib.CONDITIONS}
    same = sets["Ctrl"] == sets["ZeroK"]
    checks.add("scored workbook markers are the same in both conditions (C3 variant then drops nothing)",
               f"{same}", None)
    t18cov_path = cc.out_path("labels_core", "tables", "submitted_marker_coverage", ".csv")
    if t18cov_path.exists():
        t18cov = pd.read_csv(run.use_input(t18cov_path, "labels_core marker coverage (marker-set comparison only)"))
        t18_scored = set(t18cov.loc[t18cov["scored"].astype(str) == "True", "marker"])
    else:   # the table of the earlier search of the control runs is not a public input
        t18_scored = None
        run.note("Marker-set comparison with the earlier search skipped: its table is not available.")
    for cond in (lib.CONDITIONS if t18_scored is not None else ()):
        ours = {g for gs in out[cond]["lists"].values() for g in gs}
        checks.add(f"{cond}: scored markers vs labels_core (separate Ctrl search): added; removed",
                   f"{';'.join(sorted(ours - t18_scored)) or 'none'}; {';'.join(sorted(t18_scored - ours)) or 'none'}",
                   None)
    if not checks.ok:
        lib.stop(run, checks, "submitted_checks")

    per_cell = pd.concat([out[c]["per_cell"] for c in lib.CONDITIONS], ignore_index=True)
    run.table(checks.frame(), "submitted_checks")
    run.table(pd.concat([out[c]["settings"] for c in lib.CONDITIONS], ignore_index=True), "submitted_settings")
    run.table(pd.concat([out[c]["coverage"] for c in lib.CONDITIONS], ignore_index=True), "submitted_marker_coverage")
    run.table(per_cell, "submitted_per_cell")
    run.table(pd.DataFrame([r for c in lib.CONDITIONS for r in out[c]["counts"]]), "submitted_label_counts")
    run.table(pd.DataFrame([r for c in lib.CONDITIONS for r in out[c]["cross"]]), "submitted_cell_vs_cluster_crosstab")
    run.table(pd.concat([out[c]["naming"] for c in lib.CONDITIONS], ignore_index=True), "submitted_cluster_naming")
    run.table(pd.concat([out[c]["edges"] for c in lib.CONDITIONS], ignore_index=True), "submitted_paga_connectivities")
    run.table(pd.concat([out[c]["roots"] for c in lib.CONDITIONS], ignore_index=True), "submitted_dpt_roots")
    run.finish({"settings": [dict(zip(("setting", "value_used", "lines", "note"), r)) for r in t18.SUBMITTED_SETTINGS],
                "markers_scored": {c: out[c]["lists"] for c in lib.CONDITIONS},
                "n_random_dpt_roots": t18.N_RANDOM_ROOTS, "scanpy": sc.__version__,
                "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "runtime_s": round(time.time() - t_start, 1),
                "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
