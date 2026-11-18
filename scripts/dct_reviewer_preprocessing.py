#!/usr/bin/env python
"""dct_reviewer step 12a: preprocessing steps 2-3 preprocessing sensitivity per DCT condition (combined search).

Copy of ctrl_reviewer_core_preprocessing.py, run once per
condition (the ctrl_reviewer_core functions read the labels_core object, the labels_core column names and ctrl_reviewer_core_lib.EXPECTED_COUNTS
directly). The grid is unchanged: the 33 pipelines of preprocessing_lib.variant_grid() x Leiden 0.5 / 1.0 / 2.0 / 3.0,
evaluated against both label columns of each condition (primary cs_ctrl_deg_label / cs_zerok_pipeline_label,
second cs_ctrl_cluster_label / cs_zerok_cluster_label). dct_reviewer changes: dct_labels objects and labels; stored Leiden 3.0 =
cs_*_leiden3_cluster and stored Leiden 1.0 = cs_*_{deg,pipeline}_cluster; the matrix rebuild check uses the
combined export (dct_reviewer_lib.rebuild_raw); the baseline cross-check is against dct_reviewer step 6b ('uncorrected',
filtered by condition).

Baseline-graph check (as ctrl_reviewer_core; authors, 28 Oct 2026): the stored graph was built on Scanpy's PCA fallback, the preprocessing
baseline on an arpack PCA, so the check is: identical neighbour sets for all cells, max connectivity difference
< 1e-3, and Leiden 3.0 / 1.0 (seed 42) reproducing the stored clusters (ARI 1).

Per clustering (pipeline x resolution): n clusters; ARI, NMI and cluster-majority accuracy vs each label column;
ARI vs the stored Leiden 3.0 clusters. Per pipeline: kNN label purity (k = 10, the pipeline's PC space, 1,000
label permutations); silhouette by label; Spearman rho of PC1-PC5 with ln(protein IDs) and ln(total intensity).
UMAP (seed 42) of every pipeline's graph, for the figure step. Sanity checks run first; if one fails the script
writes the check table and stops.

Inputs (read-only): dct_labels labelled objects and canonical labels; data/DCT/DCT_Ctrl_ZeroK_combine_1535cells.csv (rebuild
check); dct_reviewer_embed_knn_label_purity.csv and dct_reviewer_embed_pc_vs_depth_and_plate.csv (step 6b).
Outputs (results/tables/dct_reviewer/; every table has `condition`):
  dct_reviewer_prep_{sanity_checks,normalization_diagnostics,feature_sets,feature_set_summary,clustering_metrics,
  knn_purity,silhouette,pc_depth,per_cell_clusters,umap_coords}.csv

Command (project root):  python scripts/dct_reviewer_preprocessing.py
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
import ctrl_reviewer_core_preprocessing as p19
import dct_reviewer_lib as lib

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "dct_reviewer_preprocessing.py"
STEP6B_PURITY = lib.table_path("embed_knn_label_purity")
STEP6B_PC_DEPTH = lib.table_path("embed_pc_vs_depth_and_plate")
TOL = p19.TOL                         # connectivities 1e-3: authors, 28 Oct 2026 (preprocessing: 1e-4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 12a: preprocessing sensitivity per DCT condition.")
    parser.add_argument("--n-permutations", type=int, default=t4.N_PERMUTATIONS)
    cc.add_common_args(parser)
    return parser.parse_args()


class Checks(p19.Checks):
    """ctrl_reviewer_core_preprocessing.Checks with the condition in every row."""

    def __init__(self, cond: str):
        super().__init__("dct")
        self.cond = cond

    def add(self, check: str, observed, passed: bool | None, threshold: str = "") -> None:
        super().add(f"{self.cond}: {check}", observed, passed, threshold)
        self.rows[-1] = {"condition": self.cond, **self.rows[-1], "check": check}


def pre_checks(checks: Checks, s, adata, labels, rebuilt: pd.DataFrame) -> None:
    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        expected = s.EXPECTED_COUNTS[col]
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in expected),
                   observed == expected)
    same_cells = list(rebuilt.index) == list(adata.obs_names.astype(str))
    same_vars = list(rebuilt.columns) == list(adata.var_names.astype(str))
    checks.add("rebuilt matrix (combined export, intake_core filters) has the dct_labels cell and protein order",
               f"cells {same_cells}; proteins {same_vars}", same_cells and same_vars)
    if not (same_cells and same_vars):
        return
    stored = cc.dense(adata.layers["intensity"])
    checks.add("rebuilt intensity equals layers['intensity'] (exact)",
               f"max abs diff {np.abs(rebuilt.to_numpy() - stored).max():.3g}",
               bool(np.array_equal(rebuilt.to_numpy(), stored)))
    d = np.abs(np.log1p(stored.astype(np.float64)) - cc.dense(adata.layers["ln1p_intensity"])).max()
    checks.add("ln(1 + intensity) equals layers['ln1p_intensity']", f"max abs diff {d:.3g}", d < TOL["matrix"],
               f"< {TOL['matrix']}")


def baseline_checks(checks: Checks, s, adata, labels, g_base, graph_base, purity_base, depth_base,
                    step6b_purity: pd.DataFrame, step6b_depth: pd.DataFrame) -> None:
    d = np.abs(cc.dense(g_base.X) - cc.dense(adata.X)).max()
    checks.add("baseline scaled matrix reproduces the dct_labels .X", f"max abs diff {d:.3g}", d < TOL["matrix"],
               f"< {TOL['matrix']}")
    stored_pca = np.asarray(adata.obsm["X_pca"])[:, :t4.BASE_N_PCS]
    new_pca = g_base.obsm["X_pca"][:, :t4.BASE_N_PCS]
    r = np.array([abs(np.corrcoef(stored_pca[:, i], new_pca[:, i])[0, 1]) for i in range(t4.BASE_N_PCS)])
    checks.add("baseline PC1-20 vs stored X_pca[:, :20]: min |Pearson r|", f"{r.min():.6f}",
               r.min() >= TOL["pca_abs_r"], f">= {TOL['pca_abs_r']}")
    new_c = cc.dense(graph_base.obsp["connectivities"])
    old_c = cc.dense(adata.obsp["connectivities"])
    same = int(((new_c > 0) == (old_c > 0)).all(axis=1).sum())
    checks.add("baseline kNN graph has the same neighbour sets as the stored dct_labels graph (cells)",
               f"{same}/{adata.n_obs}", same == adata.n_obs, "all cells")
    d = np.abs(new_c - old_c).max()
    checks.add("baseline kNN connectivities (k = 10, 20 PCs) vs stored graph (tolerance set by the authors, 28 Oct 2026)",
               f"max abs diff {d:.3g}; {int((np.abs(new_c - old_c) > 1e-4).sum())} weights differ by > 1e-4",
               d < TOL["connectivities"], f"< {TOL['connectivities']}")
    for res, col in ((3.0, s.LEIDEN3_CLUSTER), (1.0, s.PRIMARY_CLUSTER)):
        cl = t4.leiden(graph_base, res, f"check_leiden_{res}").astype(str).to_numpy()
        ari = float(adjusted_rand_score(labels[col].astype(str).to_numpy(), cl))
        checks.add(f"baseline Leiden {res} (seed 42) reproduces {col} (ARI)", f"{ari:.4f}", ari == 1.0, "= 1")
    ref = step6b_purity[(step6b_purity["condition"] == s.CONDITION) & (step6b_purity["object"] == "uncorrected")]
    diffs = []
    for _, row in ref.iterrows():
        mine = purity_base[(purity_base["label_column"] == row["label_column"]) &
                           (purity_base["label"] == row["label"])]["observed_same_label_fraction"]
        diffs.append(abs(float(mine.iloc[0]) - float(row["observed_same_label_fraction"])))
    checks.add("baseline kNN purity equals dct_reviewer step 6b 'uncorrected' (all labels, both columns)",
               f"max abs diff {max(diffs):.3g} over {len(diffs)} rows", bool(diffs) and max(diffs) < TOL["t02_purity"],
               f"< {TOL['t02_purity']}")
    ref = step6b_depth[(step6b_depth["condition"] == s.CONDITION) & (step6b_depth["object"] == "uncorrected")
                       & step6b_depth["pc"].isin([1, 2])].set_index("pc")
    diffs = [abs(abs(float(depth_base.loc[depth_base["pc"] == pc, "spearman_rho_ln_n_detected"].iloc[0]))
                 - abs(float(ref.at[pc, "spearman_rho_ln_ids"]))) for pc in (1, 2)]
    checks.add("baseline |rho| of PC1/PC2 with ln(IDs) equals dct_reviewer step 6b 'uncorrected'",
               f"max abs diff {max(diffs):.3g}", max(diffs) < TOL["t02_rho"], f"< {TOL['t02_rho']}")


def one_condition(run: cc.Run, cond: str, rebuilt: pd.DataFrame, step6b: tuple, n_perm: int) -> dict:
    s = lib.part2_settings(cond)
    spec = s.PREPROCESSING_DATASETS["dct"]
    t_start = time.time()
    adata, labels = lib.load(run, cond)
    checks = Checks(cond)
    print(f"[{cond}] sanity checks", flush=True)
    pre_checks(checks, s, adata, labels, rebuilt)
    if not checks.ok:
        return {"checks": checks}

    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    det_rate = t4.detection_rate(intensity)
    ln_ids = np.log(labels["n_detected_proteins"].to_numpy(float))
    ln_total = np.log(labels["total_intensity"].to_numpy(float))
    stored_leiden = labels[spec["stored_leiden"]].astype(str).to_numpy()
    label_cols = spec["label_columns"]
    codes = {col: t4.label_codes(labels[col], "dct") for col in label_cols}

    norm_cache: dict = {}
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
            norm_rows.append({"dataset": "dct", **info})
            if info.get("n_detected_nonpositive_after", 0):
                checks.add("median-centring keeps every detected value > 0",
                           f"{info['n_detected_nonpositive_after']} values <= 0", False)
                return {"checks": checks}
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
                return {"checks": checks}
        key = (v.normalization, v.features, v.scaling)
        if key not in emb_cache:
            emb_cache[key] = t4.build_embedding(norm_log, mask, v.scaling, cell_ids, var_names)
        g = emb_cache[key]
        pcs = g.obsm["X_pca"][:, :v.n_pcs]
        graph = t4.build_graph(g, v.n_pcs, v.k)

        idx = t4.neighbor_index(pcs, t4.PURITY_K)
        purity_v, sil_v = [], []
        for col in label_cols:
            c, cats = codes[col]
            p = t4.knn_purity(idx, c, cats, n_perm)
            p.insert(0, "label_column", col)
            purity_v.append(p)
            sv = t4.silhouette(pcs, c, cats)
            sv.insert(0, "label_column", col)
            sil_v.append(sv)
        purity_v = pd.concat(purity_v, ignore_index=True)
        sil_v = pd.concat(sil_v, ignore_index=True)
        depth_v = t4.pc_depth(g.obsm["X_pca"], ln_ids, ln_total, g.uns["pca"]["variance_ratio"])
        for frame, rows in ((purity_v, purity_rows), (sil_v, sil_rows), (depth_v, depth_rows)):
            frame.insert(0, "variant_id", v.variant_id)
            frame.insert(0, "dataset", "dct")
            rows.append(frame)

        if v.is_baseline:
            baseline_checks(checks, s, adata, labels, g, graph, purity_v, depth_v, *step6b)
            if not checks.ok:
                return {"checks": checks}

        for res in t4.RESOLUTIONS:
            clusters = t4.leiden(graph, res, f"leiden_{res}")
            name = f"{v.variant_id}__r{res:.1f}"
            per_cell[name] = clusters.to_numpy()
            ari_stored = float(adjusted_rand_score(stored_leiden, clusters.to_numpy()))
            for col in label_cols:
                agree = t4.cluster_agreement(clusters.to_numpy(), labels[col].to_numpy())
                cluster_rows.append({"dataset": "dct", "variant_id": v.variant_id, "resolution": res,
                                     "label_column": col, "n_clusters": int(clusters.nunique()), **agree,
                                     "ari_vs_stored_leiden": ari_stored})
            if v.is_baseline and res == spec["submitted_resolution"]:
                checks.add(f"baseline Leiden {res} rerun vs stored {spec['stored_leiden']} (ARI)",
                           f"{ari_stored:.4f} ({clusters.nunique()} vs {len(set(stored_leiden))} clusters)", None)
        xy = t4.umap(graph)
        umap_coords[f"{v.variant_id}__umap1"] = xy[:, 0]
        umap_coords[f"{v.variant_id}__umap2"] = xy[:, 1]
        print(f"[{cond}] {i + 1:2d}/33 {v.variant_id}: {g.n_vars} proteins, {time.time() - t0:.1f} s", flush=True)

    per_protein, feat_summary = p19.feature_tables("dct", var_names, det_rate, norm_cache["none"][0].mean(axis=0),
                                                   feat_cache)
    return {"checks": checks, "norm": pd.DataFrame(norm_rows), "feature_sets": per_protein,
            "feature_summary": feat_summary, "clustering": pd.DataFrame(cluster_rows),
            "purity": pd.concat(purity_rows, ignore_index=True), "silhouette": pd.concat(sil_rows, ignore_index=True),
            "pc_depth": pd.concat(depth_rows, ignore_index=True), "per_cell": per_cell.reset_index(),
            "umap": umap_coords.reset_index(), "runtime_s": round(time.time() - t_start, 1)}


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "preprocessing", args)
    lib.ported_from(run, "ctrl_reviewer_core_preprocessing.py", "preprocessing_sensitivity.py")
    step6b = (pd.read_csv(run.use_input(STEP6B_PURITY, "dct_reviewer step 6b kNN purity (baseline cross-check)")),
              pd.read_csv(run.use_input(STEP6B_PC_DEPTH, "dct_reviewer step 6b PC vs depth (baseline cross-check)")))
    rebuilt = lib.rebuild_raw(run)
    res = {}
    for cond in lib.CONDITIONS:
        res[cond] = one_condition(run, cond, rebuilt[cond].retained, step6b, args.n_permutations)
        if not res[cond]["checks"].ok:
            break
    checks = pd.concat([r["checks"].frame() for r in res.values()], ignore_index=True)
    run.table(checks, "prep_sanity_checks")
    if not checks["pass"].all():
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("A sanity check failed; see dct_reviewer_prep_sanity_checks.csv (nothing else written).")

    def stacked(key: str) -> pd.DataFrame:
        return lib.stack({c: res[c][key] for c in lib.CONDITIONS})

    run.table(stacked("norm"), "prep_normalization_diagnostics")
    run.table(stacked("feature_sets"), "prep_feature_sets")
    run.table(stacked("feature_summary"), "prep_feature_set_summary")
    run.table(stacked("clustering"), "prep_clustering_metrics")
    run.table(stacked("purity"), "prep_knn_purity")
    run.table(stacked("silhouette"), "prep_silhouette")
    run.table(stacked("pc_depth"), "prep_pc_depth")
    run.table(stacked("per_cell"), "prep_per_cell_clusters")
    run.table(stacked("umap"), "prep_umap_coords")
    run.note("Per-cell cluster and UMAP tables hold both conditions (condition column); each condition's grid ran on its "
             "own dct_labels matrix (Ctrl 729 x 6,078, ZeroK 702 x 5,953).")
    run.finish({"variants": [v.as_row() for v in t4.variant_grid()], "resolutions": list(t4.RESOLUTIONS),
                "hvg_flavor": t4.HVG_FLAVOR, "pca": f"arpack, {t4.N_COMPS} components, random_state {t4.SEED}",
                "neighbors": "sc.pp.neighbors (umap connectivities, euclidean), random_state 42",
                "leiden": "sc.tl.leiden flavor='igraph', random_state 42 (as reviewer_core/qc_mito)",
                "umap": "sc.tl.umap random_state 42", "purity_k": t4.PURITY_K,
                "n_permutations": args.n_permutations, "min_label_n_for_macro": t4.MIN_LABEL_N,
                "tolerances": TOL, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "scanpy": sc.__version__,
                "runtime_s": {c: res[c]["runtime_s"] for c in lib.CONDITIONS}, "checks_pass": True})


if __name__ == "__main__":
    main()
