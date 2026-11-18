#!/usr/bin/env python
"""ctrl_reviewer_core step 12a: preprocessing steps 2-3 preprocessing sensitivity on the control DCT data (labels_core labels).

Port of
preprocessing_sensitivity.py (the same 33-pipeline grid of preprocessing_lib.variant_grid(),
unchanged; DCT only). ctrl_reviewer_core changes: labels_core control object and label columns
(ctrl_deg_label, ctrl_cluster_label); stored Leiden = ctrl_leiden3_cluster; the
matrix rebuild check uses the intake_core reader; the baseline cross-check is against
ctrl_reviewer_core step 6b (uncorrected) instead of reviewer_core. Baseline-graph check (authors, 28 Oct 2026): the stored
labels_core graph was built on Scanpy's PCA fallback, the preprocessing baseline on an arpack PCA, so the preprocessing
check 'connectivities within 1e-4' is replaced by: identical neighbour sets for all cells,
max connectivity difference < 1e-3, and Leiden 3.0 / 1.0 reproducing ctrl_leiden3_cluster /
ctrl_deg_cluster (ARI 1).

Runs the 33 pipelines of preprocessing_lib.variant_grid() on the labels_core control matrix
(``layers['intensity']``, retained cells x retained proteins) and evaluates each against
the submitted labels (both label columns of the dataset):
  per clustering (pipeline x Leiden 0.5 / 1.0 / 2.0 / 3.0):
    n clusters; ARI, NMI and cluster-majority accuracy vs each label column; ARI vs the stored
    submitted Leiden clusters (KIDNEY leiden_res_2.00, DCT Leiden 3.0)
  per pipeline (no clustering):
    kNN label purity (k = 10, the pipeline's PC space, 1,000 label permutations);
    silhouette by label (the pipeline's PC space);
    Spearman rho of PC1-PC5 with ln(protein IDs) and ln(total intensity)
  UMAP (seed 42) of every pipeline's graph, for the figure step.

Sanity checks run first; if one fails the script writes the check table and stops.

Inputs (read-only): labels_core labelled control object and canonical labels; data/DCT/DCT_Ctrl_768cells.csv (rebuild check);
  ctrl_reviewer_core step 6b tables (cross-check of the baseline purity and PC-depth rho).
Outputs: results/tables/ctrl_reviewer_core/ctrl_reviewer_core_dct_*.csv

Command (project root):
  python scripts/ctrl_reviewer_core_preprocessing.py --dataset dct
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
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import preprocessing_lib as t4
import ctrl_reviewer_core_lib as lib
from types import SimpleNamespace

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "ctrl_reviewer_core_preprocessing.py"
REVIEWER_CORE_PURITY = lib.table_path("embed_knn_label_purity")         # ctrl_reviewer_core step 6b (the reviewer_core B.4-B.5 port)
REVIEWER_CORE_PC_DEPTH = lib.table_path("embed_pc_vs_depth_and_plate")
TOL = {"matrix": 1e-4, "pca_abs_r": 0.999, "connectivities": 1e-3,   # 1e-3: authors, 28 Oct 2026 (preprocessing: 1e-4)
       "t02_purity": 1e-3, "t02_rho": 0.01}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 12a: preprocessing sensitivity (control DCT).")
    parser.add_argument("--dataset", default="dct", choices=sorted(lib.PREPROCESSING_DATASETS))
    parser.add_argument("--n-permutations", type=int, default=t4.N_PERMUTATIONS)
    cc.add_common_args(parser)
    return parser.parse_args()


class Checks:
    def __init__(self, dataset: str):
        self.dataset, self.rows = dataset, []

    def add(self, check: str, observed, passed: bool | None, threshold: str = "") -> None:
        kind = "info" if passed is None else "check"
        self.rows.append({"dataset": self.dataset, "kind": kind, "check": check, "observed": observed,
                          "threshold": threshold, "pass": True if passed is None else bool(passed)})
        flag = "info" if passed is None else ("ok" if passed else "FAILED")
        print(f"  [{flag}] {check}: {observed}", flush=True)

    @property
    def ok(self) -> bool:
        return all(r["pass"] for r in self.rows)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def stop(run: cc.Run, checks: Checks, ds: str) -> None:
    run.table(checks.frame(), f"{ds}_sanity_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit(f"{ds}: sanity check failed; see ctrl_reviewer_core_{ds}_sanity_checks.csv (nothing else written).")


def pre_checks(checks: Checks, ds: str, adata, labels, rb: cc.Rebuild) -> None:
    for col in lib.PREPROCESSING_DATASETS[ds]["label_columns"]:
        observed = labels[col].astype(str).value_counts().to_dict()
        expected = lib.EXPECTED_COUNTS[col]
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in expected),
                   observed == expected)
    same_cells = list(rb.intensity.index) == list(adata.obs_names.astype(str))
    same_vars = list(rb.intensity.columns) == list(adata.var_names.astype(str))
    checks.add("rebuilt matrix has the canonical cell and protein order", f"cells {same_cells}; proteins {same_vars}",
               same_cells and same_vars)
    if not (same_cells and same_vars):
        return
    stored = cc.dense(adata.layers["intensity"])
    checks.add("rebuilt intensity equals layers['intensity'] (exact)",
               f"max abs diff {np.abs(rb.intensity.to_numpy() - stored).max():.3g}",
               bool(np.array_equal(rb.intensity.to_numpy(), stored)))
    d = np.abs(np.log1p(stored.astype(np.float64)) - cc.dense(adata.layers["ln1p_intensity"])).max()
    checks.add("ln(1 + intensity) equals layers['ln1p_intensity']", f"max abs diff {d:.3g}", d < TOL["matrix"],
               f"< {TOL['matrix']}")


def baseline_checks(checks: Checks, ds: str, adata, labels, g_base, graph_base, purity_base, depth_base) -> None:
    d = np.abs(cc.dense(g_base.X) - cc.dense(adata.X)).max()
    checks.add("baseline scaled matrix reproduces the canonical .X", f"max abs diff {d:.3g}", d < TOL["matrix"],
               f"< {TOL['matrix']}")
    stored_pca = np.asarray(adata.obsm["X_pca"])[:, :t4.BASE_N_PCS]
    new_pca = g_base.obsm["X_pca"][:, :t4.BASE_N_PCS]
    r = np.array([abs(np.corrcoef(stored_pca[:, i], new_pca[:, i])[0, 1]) for i in range(t4.BASE_N_PCS)])
    checks.add("baseline PC1-20 vs stored X_pca[:, :20]: min |Pearson r|", f"{r.min():.6f}",
               r.min() >= TOL["pca_abs_r"], f">= {TOL['pca_abs_r']}")
    new_c = cc.dense(graph_base.obsp["connectivities"])
    old_c = cc.dense(adata.obsp["connectivities"])
    same = int(((new_c > 0) == (old_c > 0)).all(axis=1).sum())
    checks.add("baseline kNN graph has the same neighbour sets as the stored labels_core graph (cells)", f"{same}/{adata.n_obs}",
               same == adata.n_obs, "all cells")
    d = np.abs(new_c - old_c).max()
    checks.add("baseline kNN connectivities (k = 10, 20 PCs) vs stored graph (tolerance set by the authors, 28 Oct 2026)",
               f"max abs diff {d:.3g}; {int((np.abs(new_c - old_c) > 1e-4).sum())} weights differ by > 1e-4",
               d < TOL["connectivities"], f"< {TOL['connectivities']}")
    for res, col in ((3.0, "ctrl_leiden3_cluster"), (1.0, "ctrl_deg_cluster")):
        cl = t4.leiden(graph_base, res, f"check_leiden_{res}").astype(str).to_numpy()
        ari = float(adjusted_rand_score(labels[col].astype(str).to_numpy(), cl))
        checks.add(f"baseline Leiden {res} (seed 42) reproduces {col} (ARI)", f"{ari:.4f}", ari == 1.0, "= 1")
    if ds == "dct":
        t02 = pd.read_csv(REVIEWER_CORE_PURITY)
        t02 = t02[t02["object"] == "uncorrected"]
        diffs = []
        for _, row in t02.iterrows():
            mine = purity_base[(purity_base["label_column"] == row["label_column"]) &
                               (purity_base["label"] == row["label"])]["observed_same_label_fraction"]
            diffs.append(abs(float(mine.iloc[0]) - float(row["observed_same_label_fraction"])))
        checks.add("DCT baseline kNN purity equals ctrl_reviewer_core step 6b 'uncorrected' (all labels, both columns)",
                   f"max abs diff {max(diffs):.3g} over {len(diffs)} rows", max(diffs) < TOL["t02_purity"],
                   f"< {TOL['t02_purity']}")
        t02 = pd.read_csv(REVIEWER_CORE_PC_DEPTH)
        t02 = t02[(t02["object"] == "uncorrected") & t02["pc"].isin([1, 2])].set_index("pc")
        diffs = [abs(abs(float(depth_base.loc[depth_base["pc"] == pc, "spearman_rho_ln_n_detected"].iloc[0]))
                     - abs(float(t02.at[pc, "spearman_rho_ln_ids"]))) for pc in (1, 2)]
        checks.add("DCT baseline |rho| of PC1/PC2 with ln(IDs) equals ctrl_reviewer_core step 6b 'uncorrected'",
                   f"max abs diff {max(diffs):.3g}", max(diffs) < TOL["t02_rho"], f"< {TOL['t02_rho']}")


def feature_tables(ds: str, var_names, det_rate, mean_ln1p, feat_cache) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_protein = pd.DataFrame({"protein": list(var_names), "detection_rate": det_rate, "mean_ln1p": mean_ln1p})
    rows = []
    for (norm, feat), (mask, var) in sorted(feat_cache.items()):
        col = f"{norm}__{feat}"
        per_protein[col] = mask
        if var is not None:
            per_protein[f"{col}__dispersions_norm"] = var["dispersions_norm"].to_numpy()
        sel = det_rate[mask]
        ref = feat_cache.get(("none", feat))
        jacc = (float((mask & ref[0]).sum() / (mask | ref[0]).sum()) if ref is not None else np.nan)
        rows.append({"dataset": ds, "normalization": norm, "features": feat, "n_proteins": int(mask.sum()),
                     "median_detection_rate_selected": float(np.median(sel)),
                     "median_detection_rate_all": float(np.median(det_rate)),
                     "fraction_selected_detected_lt50pct": float((sel < 0.5).mean()),
                     "fraction_all_detected_lt50pct": float((det_rate < 0.5).mean()),
                     "fraction_selected_detected_ge90pct": float((sel >= 0.9).mean()),
                     "jaccard_vs_none_same_features": jacc})
    return per_protein, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    ds = args.dataset
    spec = lib.PREPROCESSING_DATASETS[ds]
    run = cc.Run(lib.TASK, f"preprocessing_{ds}", args)
    lib.ported_from(run, "preprocessing_sensitivity.py")
    t_start = time.time()
    adata, labels = lib.load_ctrl(run)
    rb = SimpleNamespace(intensity=lib.rebuild_ctrl_intensity(run))
    checks = Checks(ds)
    print(f"[{ds}] sanity checks", flush=True)
    pre_checks(checks, ds, adata, labels, rb)
    if not checks.ok:
        stop(run, checks, ds)

    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    det_rate = t4.detection_rate(intensity)
    ln_ids = np.log(labels["n_detected_proteins"].to_numpy(float))
    ln_total = np.log(labels["total_intensity"].to_numpy(float))
    stored_leiden = adata.obs[spec["stored_leiden"]].astype(str).to_numpy()
    label_cols = spec["label_columns"]
    codes = {col: t4.label_codes(labels[col], ds) for col in label_cols}
    if ds == "dct":
        run.use_input(REVIEWER_CORE_PURITY, "ctrl_reviewer_core step 6b kNN purity (baseline cross-check)")
        run.use_input(REVIEWER_CORE_PC_DEPTH, "ctrl_reviewer_core step 6b PC vs depth (baseline cross-check)")

    norm_cache: dict[str, tuple[np.ndarray, dict]] = {}
    feat_cache: dict = {}
    emb_cache: dict = {}
    hvg_checked: set = set()
    cluster_rows, purity_rows, sil_rows, depth_rows, norm_rows = [], [], [], [], []
    per_cell = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    umap_coords = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))

    for i, v in enumerate(t4.variant_grid()):
        t0 = time.time()
        if v.normalization not in norm_cache:
            norm_log, info = t4.normalize(intensity, v.normalization)
            norm_cache[v.normalization] = (norm_log, info)
            norm_rows.append({"dataset": ds, **info})
            if info.get("n_detected_nonpositive_after", 0):
                checks.add("median-centring keeps every detected value > 0",
                           f"{info['n_detected_nonpositive_after']} values <= 0", False)
                stop(run, checks, ds)
            elif v.normalization == "median":
                checks.add("median-centring: detected values <= 0 before the uniform shift",
                           f"{info['n_detected_nonpositive_centred']} values in "
                           f"{info['n_cells_with_nonpositive_centred']} cells (min {info['min_detected_centred']:.3f})",
                           None)
                checks.add("median-centring keeps every detected value > 0 (after the uniform shift)",
                           f"shift +{info['uniform_shift']:.4f}; min detected value {info['min_detected_after']:.3f}",
                           True)
        norm_log = norm_cache[v.normalization][0]
        mask = t4.feature_mask(v, norm_log, det_rate, var_names, feat_cache)
        if v.features in t4.HVG_SIZES and (v.normalization, v.features) not in hvg_checked:
            hvg_checked.add((v.normalization, v.features))
            n_sel = int(mask.sum())
            checks.add(f"HVG set {v.normalization}/{v.features} has the requested size", f"{n_sel}",
                       n_sel == t4.HVG_SIZES[v.features], f"= {t4.HVG_SIZES[v.features]}")
            if n_sel != t4.HVG_SIZES[v.features]:
                stop(run, checks, ds)
        key = (v.normalization, v.features, v.scaling)
        if key not in emb_cache:
            emb_cache[key] = t4.build_embedding(norm_log, mask, v.scaling, cell_ids, var_names)
        g = emb_cache[key]
        pcs = g.obsm["X_pca"][:, :v.n_pcs]
        graph = t4.build_graph(g, v.n_pcs, v.k)

        # clustering-independent metrics
        idx = t4.neighbor_index(pcs, t4.PURITY_K)
        purity_v, sil_v = [], []
        for col in label_cols:
            c, cats = codes[col]
            p = t4.knn_purity(idx, c, cats, args.n_permutations)
            p.insert(0, "label_column", col)
            purity_v.append(p)
            s = t4.silhouette(pcs, c, cats)
            s.insert(0, "label_column", col)
            sil_v.append(s)
        purity_v = pd.concat(purity_v, ignore_index=True)
        sil_v = pd.concat(sil_v, ignore_index=True)
        depth_v = t4.pc_depth(g.obsm["X_pca"], ln_ids, ln_total, g.uns["pca"]["variance_ratio"])
        for frame, rows in ((purity_v, purity_rows), (sil_v, sil_rows), (depth_v, depth_rows)):
            frame.insert(0, "variant_id", v.variant_id)
            frame.insert(0, "dataset", ds)
            rows.append(frame)

        if v.is_baseline:
            baseline_checks(checks, ds, adata, labels, g, graph, purity_v, depth_v)
            if not checks.ok:
                stop(run, checks, ds)

        # clusterings
        for res in t4.RESOLUTIONS:
            clusters = t4.leiden(graph, res, f"leiden_{res}")
            name = f"{v.variant_id}__r{res:.1f}"
            per_cell[name] = clusters.to_numpy()
            ari_stored = float(adjusted_rand_score(stored_leiden, clusters.to_numpy()))
            for col in label_cols:
                agree = t4.cluster_agreement(clusters.to_numpy(), labels[col].to_numpy())
                cluster_rows.append({"dataset": ds, "variant_id": v.variant_id, "resolution": res,
                                     "label_column": col, "n_clusters": int(clusters.nunique()), **agree,
                                     "ari_vs_stored_leiden": ari_stored})
            if v.is_baseline and res == spec["submitted_resolution"]:
                checks.add(f"baseline Leiden {res} rerun vs stored {spec['stored_leiden']} (ARI)",
                           f"{ari_stored:.4f} ({clusters.nunique()} vs {len(set(stored_leiden))} clusters)", None)
        xy = t4.umap(graph)
        umap_coords[f"{v.variant_id}__umap1"] = xy[:, 0]
        umap_coords[f"{v.variant_id}__umap2"] = xy[:, 1]
        print(f"[{ds}] {i + 1:2d}/33 {v.variant_id}: {g.n_vars} proteins, {time.time() - t0:.1f} s", flush=True)

    per_protein, feat_summary = feature_tables(ds, var_names, det_rate, norm_cache["none"][0].mean(axis=0),
                                               feat_cache)
    run.table(checks.frame(), f"{ds}_sanity_checks")
    run.table(pd.DataFrame(norm_rows), f"{ds}_normalization_diagnostics")
    run.table(per_protein, f"{ds}_feature_sets")
    run.table(feat_summary, f"{ds}_feature_set_summary")
    run.table(pd.DataFrame(cluster_rows), f"{ds}_clustering_metrics")
    run.table(pd.concat(purity_rows, ignore_index=True), f"{ds}_knn_purity")
    run.table(pd.concat(sil_rows, ignore_index=True), f"{ds}_silhouette")
    run.table(pd.concat(depth_rows, ignore_index=True), f"{ds}_pc_depth")
    run.table(per_cell, f"{ds}_per_cell_clusters", index=True)
    run.table(umap_coords, f"{ds}_umap_coords", index=True)
    run.finish({"variants": [v.as_row() for v in t4.variant_grid()], "resolutions": list(t4.RESOLUTIONS),
                "hvg_flavor": t4.HVG_FLAVOR, "pca": f"arpack, {t4.N_COMPS} components, random_state {t4.SEED}",
                "neighbors": "sc.pp.neighbors (umap connectivities, euclidean), random_state 42",
                "leiden": "sc.tl.leiden flavor='igraph', random_state 42 (as reviewer_core/qc_mito)",
                "umap": "sc.tl.umap random_state 42", "purity_k": t4.PURITY_K,
                "n_permutations": args.n_permutations, "min_label_n_for_macro": t4.MIN_LABEL_N,
                "tolerances": TOL, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "scanpy": sc.__version__,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
