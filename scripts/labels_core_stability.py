#!/usr/bin/env python
"""labels_core step 3: stability of ctrl_cell_label, ctrl_cluster_label and ctrl_deg_label (control DCT cells).

Analysis plan, step 3
(noise-test definition agreed with the authors, 28 Oct 2026).

1. Leiden seeds 42 and 0-9 on the step-1 graph: Leiden 3.0 + the original cluster rule (ctrl_cluster_label; the
   per-cell scores are fixed) and Leiden 1.0 + the rank-free DE rule (ctrl_deg_label; seeds from the step-2a scan).
   ctrl_cell_label does not use Leiden (stability 1 by construction). ARI against seed 42 and per-cell stability.
2. Noise, N(0, 1e-6) added to the scaled .X (kept float32, as the pipeline), 20 repeats (numpy default_rng(42)):
   (A) the noisy .X is scored (sc.tl.score_genes) -> per-cell argmax; the cluster rule on the fixed Leiden 3.0
       clusters with the noisy scores is reported alongside;
   (B) the noisy .X goes through neighbours (k = 10, 20 PCs; Scanpy PCA fallback) -> Leiden 3.0 and 1.0 (seed 42)
       -> ctrl_cluster_label (original scores = graph effect only; noisy scores = whole pipeline) and ctrl_deg_label
       (rank-free DE rule; DE on the unchanged ln(1 + intensity)). A no-noise control run must reproduce step 1/2b.
3. Marker lists: leave-one-marker-out over the 12 scored markers (each removed from every list) and the variants
   noSlc12a3 / noSlc12a3_noUmod (reviewer_core section D), for all three label columns. Clusterings stay fixed (Leiden 3.0
   for ctrl_cluster_label, Leiden 1.0 for ctrl_deg_label; the cluster DE is unchanged, only the marker evidence).
4. Crosstabs of ctrl_cell_label, ctrl_cluster_label and ctrl_deg_label (pairwise and three-way).
Also: per-protein means of the scored .X, which define sc.tl.score_genes' control-protein bins
(labels_core_stability_score_bins.csv).

Inputs (read-only): results/h5ad/labels_core_DCT_ctrl_step1_submitted.h5ad; labels_core_submitted_per_cell.csv;
  labels_core_deg_labels_per_cell.csv; labels_core_scan_seed_clusters_per_cell.csv; data/DCT/DCT_markers_V1.xlsx.
Outputs: results/tables/labels_core/labels_core_stability_*.csv, labels_core_crosstab_*.csv

Command (project root):  python scripts/labels_core_stability.py
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
import cluster_de_lib as t5
import labels_core_lib as lib
from kidney_annotation_rankfree_sensitivity import rank_free

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "labels_core_stability.py"
LABEL_COLUMNS = ("ctrl_cell_label", "ctrl_cluster_label", "ctrl_deg_label")
DEG_RESOLUTION = 1.0     # authors's choice (step 2b)
DEG_RULE = "rankfree"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="labels_core step 3: stability of the three control label columns.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "stability_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see labels_core_stability_checks.csv.")


class Labeller:
    """The three labelling rules on fixed inputs; each call returns per-cell labels as numpy arrays."""

    def __init__(self, adata, lists, markers, cache, x_z, det_frame, var_names):
        self.cell_ids = adata.obs_names.astype(str)
        self.var_names = var_names
        self.lists, self.markers, self.cache = lists, markers, cache
        self.x_z, self.det_frame = x_z, det_frame

    def cell(self, X: np.ndarray, lists=None) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
        scores = lib.score_matrix(X, self.var_names, self.cell_ids, lists or self.lists)
        label, max_score = lib.cell_labels(scores)
        return label, max_score, scores

    def cluster(self, label: pd.Series, max_score: pd.Series, clusters: np.ndarray) -> np.ndarray:
        named, _ = lib.cluster_labels(label, max_score, pd.Series(clusters, index=label.index))
        return named.astype(str).to_numpy()

    def deg(self, clusters: np.ndarray, markers=None, x_z=None, de=None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        markers = markers or self.markers
        de, _, asg = lib.annotate_clustering(self.cache, clusters, self.var_names, "t18_stability", markers,
                                             self.x_z if x_z is None else x_z, self.det_frame, de=de)
        rf = rank_free(asg, de, lib.specific_present(markers))
        lab = lib.per_cell(clusters, rf.rename(columns={"rankfree_label": "deg_label"}), "deg_label").astype(str)
        conf = lib.per_cell(clusters, rf.rename(columns={"rankfree_confidence": "confidence"}), "confidence").astype(str)
        return lab, conf, de


def rebuild_graph(X: np.ndarray, obs_names):
    """Neighbours as the submitted pipeline (no prior PCA; Scanpy computes 20 PCs) on a bare matrix."""
    import anndata as ad

    g = ad.AnnData(X=np.array(X, dtype=np.float32, copy=True), obs=pd.DataFrame(index=pd.Index(obs_names)))
    sc.pp.neighbors(g, n_neighbors=lib.N_NEIGHBORS, n_pcs=lib.N_PCS)
    return g


def change_row(test: str, repeat: int, column: str, ref: np.ndarray, new: np.ndarray, **extra) -> dict:
    changed = ref != new
    row = {"test": test, "repeat": repeat, "label_column": column, "n_changed": int(changed.sum()),
           "fraction_changed": float(changed.mean()), "label_counts": lib.counts_text(new)}
    for label in cc.ordered_labels(ref, lib.LABEL_ORDER):
        m = ref == label
        row[f"n_changed_from_{label}"] = int((changed & m).sum())
    row.update(extra)
    return row


def crosstab_long(rows: np.ndarray, cols: np.ndarray, row_name: str, col_name: str, **keys) -> list[dict]:
    ct = pd.crosstab(pd.Series(rows, name=row_name), pd.Series(cols, name=col_name))
    return [{**keys, row_name: r, col_name: c, "n_cells": int(ct.loc[r, c])}
            for r in cc.ordered_labels(ct.index, lib.LABEL_ORDER) for c in cc.ordered_labels(ct.columns, lib.LABEL_ORDER)]


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "stability", args)
    t_start = time.time()
    checks = t5.Checks("dct_ctrl")
    adata = lib.load_step1(run)
    step1 = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    deg = lib.read_table(run, "deg_labels_per_cell", dtype=str).set_index("cell_id")
    seeds_deg = lib.read_table(run, "scan_seed_clusters_per_cell", dtype=str).set_index("cell_id")
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    for name, frame in (("step-1 table", step1), ("step-2b table", deg), ("scan seed table", seeds_deg)):
        same = list(frame.index) == list(cell_ids)
        checks.add(f"{name} holds the object's cells in the same order", f"{same}", same)
    if not checks.ok:
        stop(run, checks)

    X = np.array(cc.dense(adata.X), dtype=np.float32, copy=True)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    x_z = pd.DataFrame(X.astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    workbook = lib.read_workbook(run)
    lists = lib.present_lists(workbook, var_names)
    markers = lib.dct_markers(run, var_names)
    cache = t5.rank_cache(np.log1p(intensity), detected)
    L = Labeller(adata, lists, markers, cache, x_z, det_frame, var_names)

    ref = {"ctrl_cell_label": step1["ctrl_cell_label"].astype(str).to_numpy(),
           "ctrl_cluster_label": step1["ctrl_cluster_label"].astype(str).to_numpy(),
           "ctrl_deg_label": deg["ctrl_deg_label"].astype(str).to_numpy()}
    cl3 = step1["ctrl_leiden3_cluster"].astype(str).to_numpy()
    cl1 = deg["ctrl_deg_cluster"].astype(str).to_numpy()

    # Reproduction on fixed inputs
    label0, max0, scores0 = L.cell(X)
    n_cell = int((label0.to_numpy() == ref["ctrl_cell_label"]).sum())
    d_sc = float(np.abs(scores0.to_numpy() - step1[[f"{t}_score" for t in lists]].to_numpy()).max())
    checks.add("rescoring the stored .X reproduces step-1 scores and ctrl_cell_label",
               f"max abs diff {d_sc:.3g}; {n_cell}/727", d_sc < 1e-6 and n_cell == 727)
    n_clu = int((L.cluster(label0, max0, cl3) == ref["ctrl_cluster_label"]).sum())
    checks.add("cluster rule on ctrl_leiden3_cluster reproduces ctrl_cluster_label", f"{n_clu}/727", n_clu == 727)
    lab_d, conf_d, de1 = L.deg(cl1)
    n_deg = int((lab_d == ref["ctrl_deg_label"]).sum())
    checks.add("rank-free rule on ctrl_deg_cluster reproduces ctrl_deg_label", f"{n_deg}/727", n_deg == 727)
    if not checks.ok:
        stop(run, checks)

    # 1. Leiden seeds ---------------------------------------------------------------------------------------
    print("[ctrl] Leiden seeds", flush=True)
    g = lib.graph_adata(adata)
    seed_rows, seed_labels = [], {"ctrl_cluster_label": {}, "ctrl_deg_label": {}}
    for seed in lib.SEEDS:
        c3 = lib.leiden_on(g, lib.SUBMITTED_RESOLUTION, seed)
        lab3 = L.cluster(label0, max0, c3)
        if seed == lib.SEED:
            same = bool((c3 == cl3).all())
            checks.add("Leiden 3.0 seed 42 on the stored graph equals ctrl_leiden3_cluster", f"{same}", same)
            c1, lab1 = cl1, ref["ctrl_deg_label"]
        else:
            c1 = seeds_deg[f"{lib.clustering_name(DEG_RESOLUTION, seed)}__cluster"].to_numpy()
            lab1 = seeds_deg[f"{lib.clustering_name(DEG_RESOLUTION, seed)}__{DEG_RULE}_label"].to_numpy()
        seed_labels["ctrl_cluster_label"][seed] = lab3
        seed_labels["ctrl_deg_label"][seed] = lab1
        for col, cl, ref_cl, lab in (("ctrl_cluster_label", c3, cl3, lab3), ("ctrl_deg_label", c1, cl1, lab1)):
            seed_rows.append({"label_column": col, "seed": seed, "n_clusters": int(len(set(cl))),
                              "ari_clusters_vs_seed42": float(adjusted_rand_score(ref_cl, cl)),
                              "ari_labels_vs_seed42": float(adjusted_rand_score(ref[col], lab)),
                              "fraction_same_label_as_seed42": float((ref[col] == lab).mean()),
                              "label_counts": lib.counts_text(lab)})
        seed_rows.append({"label_column": "ctrl_cell_label", "seed": seed, "n_clusters": np.nan,
                          "ari_clusters_vs_seed42": np.nan, "ari_labels_vs_seed42": 1.0,
                          "fraction_same_label_as_seed42": 1.0, "label_counts": lib.counts_text(ref["ctrl_cell_label"]),
                          "note": "per-cell argmax does not use Leiden"})
    if not checks.ok:
        stop(run, checks)
    seeds = pd.DataFrame(seed_rows)
    per_cell = pd.DataFrame({"cell_id": cell_ids, "plate": step1["plate"].to_numpy(),
                             "n_detected": step1["n_detected"].to_numpy(), **{c: ref[c] for c in LABEL_COLUMNS}})
    per_cell["seed_stability_ctrl_cell_label"] = 1.0
    for col in ("ctrl_cluster_label", "ctrl_deg_label"):
        others = np.column_stack([seed_labels[col][s] for s in lib.SEEDS if s != lib.SEED])
        per_cell[f"seed_stability_{col}"] = (others == ref[col][:, None]).mean(axis=1)

    # 2. Noise tests ----------------------------------------------------------------------------------------
    print("[ctrl] noise tests (control run, then 20 repeats)", flush=True)
    rng = np.random.default_rng(lib.SEED)
    noise_rows = []
    n_changed = {k: np.zeros(len(cell_ids), int) for k in ("A_cell", "A_cluster", "B_cluster_graph", "B_cluster_full",
                                                          "B_deg")}
    for rep in range(lib.NOISE_REPEATS + 1):
        t0 = time.time()
        if rep == 0:
            Xn = X.copy()                                    # control: no noise
        else:
            Xn = (X.astype(np.float64) + rng.normal(0.0, lib.NOISE_SD, X.shape)).astype(np.float32)
        n_entries = int((Xn != X).sum())
        lab_a, max_a, scores_a = L.cell(Xn)
        lab_a = lab_a.to_numpy()
        d_scores = float(np.abs(scores_a.to_numpy() - scores0.to_numpy()).max())
        lab_a_cluster = L.cluster(pd.Series(lab_a, index=cell_ids), max_a, cl3)
        gn = rebuild_graph(Xn, cell_ids)
        c3n = lib.leiden_on(gn, lib.SUBMITTED_RESOLUTION, lib.SEED)
        c1n = lib.leiden_on(gn, DEG_RESOLUTION, lib.SEED)
        lab_b_graph = L.cluster(label0, max0, c3n)
        lab_b_full = L.cluster(pd.Series(lab_a, index=cell_ids), max_a, c3n)
        lab_b_deg, _, _ = L.deg(c1n, x_z=pd.DataFrame(Xn.astype(np.float64), index=cell_ids, columns=var_names))
        extra = {"n_x_entries_changed_after_float32": n_entries}
        results = [
            ("A: noisy .X scored", "ctrl_cell_label", lab_a, {"max_abs_score_change": d_scores}),
            ("A: noisy scores, fixed Leiden 3.0 clusters", "ctrl_cluster_label", lab_a_cluster, {}),
            ("B: noisy graph, original scores", "ctrl_cluster_label", lab_b_graph,
             {"ari_clusters_vs_original": float(adjusted_rand_score(cl3, c3n)), "n_clusters": len(set(c3n))}),
            ("B: noisy .X, whole pipeline", "ctrl_cluster_label", lab_b_full,
             {"ari_clusters_vs_original": float(adjusted_rand_score(cl3, c3n)), "n_clusters": len(set(c3n))}),
            ("B: noisy graph, rank-free DE rule at Leiden 1.0", "ctrl_deg_label", lab_b_deg,
             {"ari_clusters_vs_original": float(adjusted_rand_score(cl1, c1n)), "n_clusters": len(set(c1n))}),
        ]
        keys = ("A_cell", "A_cluster", "B_cluster_graph", "B_cluster_full", "B_deg")
        for key, (test, col, lab, ex) in zip(keys, results):
            noise_rows.append(change_row(test, rep, col, ref[col], lab, **extra, **ex))
            if rep > 0:
                n_changed[key] += (lab != ref[col]).astype(int)
        if rep == 0:
            for test, col, lab, ex in results:
                same = int((lab == ref[col]).sum())
                ok = same == 727 and ex.get("ari_clusters_vs_original", 1.0) == 1.0
                checks.add(f"no-noise control reproduces {col} ({test})", f"{same}/727", ok)
            if not checks.ok:
                stop(run, checks)
        print(f"  repeat {rep}: changed A cell {noise_rows[-5]['n_changed']}, B cluster {noise_rows[-2]['n_changed']}, "
              f"B deg {noise_rows[-1]['n_changed']}; {time.time() - t0:.1f} s", flush=True)
    noise = pd.DataFrame(noise_rows)
    for key, col in (("A_cell", "noiseA_n_repeats_changed_ctrl_cell_label"),
                     ("A_cluster", "noiseA_n_repeats_changed_ctrl_cluster_label"),
                     ("B_cluster_graph", "noiseB_graph_n_repeats_changed_ctrl_cluster_label"),
                     ("B_cluster_full", "noiseB_full_n_repeats_changed_ctrl_cluster_label"),
                     ("B_deg", "noiseB_n_repeats_changed_ctrl_deg_label")):
        per_cell[col] = n_changed[key]
    reps = noise[noise["repeat"] > 0]
    noise_summary = (reps.groupby(["test", "label_column"], sort=False)
                     .agg(n_repeats=("repeat", "size"), n_changed_median=("n_changed", "median"),
                          n_changed_min=("n_changed", "min"), n_changed_max=("n_changed", "max"),
                          fraction_changed_median=("fraction_changed", "median"),
                          fraction_changed_max=("fraction_changed", "max"),
                          n_repeats_any_change=("n_changed", lambda s: int((s > 0).sum())))
                     .reset_index())
    ari = reps.groupby("test", sort=False)["ari_clusters_vs_original"].agg(["median", "min"]).rename(
        columns={"median": "ari_clusters_median", "min": "ari_clusters_min"})
    noise_summary = noise_summary.merge(ari, left_on="test", right_index=True, how="left")
    # Rows keep the order of the tests above (groupby sort=False), which is the order of n_changed's keys.
    noise_summary["n_cells_changed_in_any_repeat"] = [int((v > 0).sum()) for v in n_changed.values()]

    # 3. Marker lists ---------------------------------------------------------------------------------------
    print("[ctrl] marker-list variants", flush=True)
    scored = list(dict.fromkeys(g for gs in lists.values() for g in gs))
    variants = {"original": ("original lists", ())}
    variants.update(lib.MARKER_VARIANTS)
    variants.update({f"loo_{g}": (f"{g} removed from every list", (g,)) for g in scored})
    var_rows, var_cross, loo_keep = [], [], {c: [] for c in LABEL_COLUMNS}
    defs = []
    for vname, (desc, genes) in variants.items():
        lists_v = lib.drop_from_lists(lists, genes)
        markers_v = lib.drop_markers(markers, genes)   # Umod is listed only for DCT1, so variant b matches reviewer_core
        lab_c, max_c, _ = L.cell(X, lists_v)
        lab_c = lab_c.to_numpy()
        lab_k = L.cluster(pd.Series(lab_c, index=cell_ids), max_c, cl3)
        lab_d, _, _ = L.deg(cl1, markers=markers_v, de=de1)
        defs.append({"variant": vname, "description": desc, "removed": ";".join(genes),
                     **{f"{t}_scored": ";".join(lists_v.get(t, [])) for t in lists},
                     **{f"{t}_specific_for_de_rule": ";".join(lib.specific_present(markers_v).get(t, [])) for t in lists}})
        for col, lab in (("ctrl_cell_label", lab_c), ("ctrl_cluster_label", lab_k), ("ctrl_deg_label", lab_d)):
            var_rows.append(change_row(vname, 0, col, ref[col], lab, removed=";".join(genes)))
            if vname.startswith("loo_"):
                loo_keep[col].append(lab == ref[col])
            elif vname != "original":
                var_cross.extend(crosstab_long(ref[col], lab, "original_label", "variant_label", variant=vname,
                                               label_column=col))
    variants_tab = pd.DataFrame(var_rows).drop(columns=["repeat"]).rename(columns={"test": "variant"})
    orig = variants_tab[variants_tab["variant"] == "original"]
    bad = orig[orig["n_changed"] != 0]
    checks.add("variant 'original' reproduces all three label columns", f"{len(bad)} columns differ", bad.empty)
    loo_rows = []
    for col in LABEL_COLUMNS:
        keep = np.column_stack(loo_keep[col]).mean(axis=1)
        per_cell[f"loo_stability_{col}"] = keep
        for label in [*cc.ordered_labels(ref[col], lib.LABEL_ORDER), "all"]:
            m = np.ones(len(keep), bool) if label == "all" else ref[col] == label
            loo_rows.append({"label_column": col, "original_label": label, "n_cells": int(m.sum()),
                             "n_loo_runs": len(loo_keep[col]), "mean_stability": float(keep[m].mean()),
                             "median_stability": float(np.median(keep[m])), "n_stability_1": int((keep[m] == 1).sum()),
                             "n_stability_below_0.5": int((keep[m] < 0.5).sum()), "n_stability_0": int((keep[m] == 0).sum())})

    # Per-label seed stability summary
    seed_label_rows = []
    for col in LABEL_COLUMNS:
        st = per_cell[f"seed_stability_{col}"].to_numpy()
        for label in [*cc.ordered_labels(ref[col], lib.LABEL_ORDER), "all"]:
            m = np.ones(len(st), bool) if label == "all" else ref[col] == label
            seed_label_rows.append({"label_column": col, "label_seed42": label, "n_cells": int(m.sum()),
                                    "mean_stability": float(st[m].mean()), "n_stable_all_10_seeds": int((st[m] == 1).sum()),
                                    "fraction_stable_all_10_seeds": float((st[m] == 1).mean()),
                                    "n_stability_below_0.5": int((st[m] < 0.5).sum())})

    # 4. Crosstabs ------------------------------------------------------------------------------------------
    pair_rows = []
    for a, b in (("ctrl_cell_label", "ctrl_cluster_label"), ("ctrl_cell_label", "ctrl_deg_label"),
                 ("ctrl_cluster_label", "ctrl_deg_label")):
        pair_rows.extend(crosstab_long(ref[a], ref[b], "row_label", "column_label", row_column=a, column_column=b))
    three = (per_cell.groupby(list(LABEL_COLUMNS)).size().rename("n_cells").reset_index()
             .sort_values("n_cells", ascending=False))
    agree_rows = [{"pair": f"{a} vs {b}", "fraction_equal": float((ref[a] == ref[b]).mean()),
                   "ari": float(adjusted_rand_score(ref[a], ref[b]))}
                  for a, b in (("ctrl_cell_label", "ctrl_cluster_label"), ("ctrl_cell_label", "ctrl_deg_label"),
                               ("ctrl_cluster_label", "ctrl_deg_label"))]
    all_three = float(((ref["ctrl_cell_label"] == ref["ctrl_cluster_label"])
                       & (ref["ctrl_cluster_label"] == ref["ctrl_deg_label"])).mean())
    agree_rows.append({"pair": "all three equal", "fraction_equal": all_three, "ari": np.nan})

    # Inputs of score_genes' control-protein bins: per-protein means of the scored matrix (the scaled .X), computed
    # as score_genes does (np.nanmean without dtype, i.e. float32 on the float32 .X); float64 means for reference.
    means = np.nanmean(X, axis=0)
    means64 = X.astype(np.float64).mean(axis=0)
    clipped = (np.abs(X) >= 10 - 1e-6).any(axis=0)
    a = reps[reps["test"] == "A: noisy .X scored"]
    bins = pd.DataFrame([
        {"quantity": "proteins in the scored matrix", "value": float(X.shape[1])},
        {"quantity": "proteins with an entry at the +/-10 clip", "value": float(clipped.sum())},
        {"quantity": "|per-protein mean of .X|, median (all proteins)", "value": float(np.median(np.abs(means)))},
        {"quantity": "|per-protein mean of .X|, median (all proteins; float64 reference)",
         "value": float(np.median(np.abs(means64)))},
        {"quantity": "|per-protein mean of .X|, max (proteins without clipped entries)",
         "value": float(np.abs(means[~clipped]).max())},
        {"quantity": "|per-protein mean of .X|, max (proteins with clipped entries)", "value": float(np.abs(means[clipped]).max())},
        {"quantity": "noise SD added in tests A and B", "value": lib.NOISE_SD},
        {"quantity": "noise A: max |score change| over repeats", "value": float(a["max_abs_score_change"].max())},
        *[{"quantity": f"{t}_score SD (step 1)", "value": float(scores0[t].std(ddof=1))} for t in lists],
    ])

    for r in noise_summary.itertuples(index=False):
        checks.add(f"noise {r.test} [{r.label_column}]: cells changed per repeat median (max); repeats with any change",
                   f"{r.n_changed_median:g} ({r.n_changed_max}); {r.n_repeats_any_change}/{r.n_repeats}", None)
    if not checks.ok:
        stop(run, checks)

    run.table(checks.frame(), "stability_checks")
    run.table(seeds, "stability_seeds")
    run.table(pd.DataFrame(seed_label_rows), "stability_seed_per_label")
    run.table(noise, "stability_noise")
    run.table(noise_summary, "stability_noise_summary")
    run.table(bins, "stability_score_bins")
    run.table(pd.DataFrame(defs), "stability_marker_variant_definitions")
    run.table(variants_tab, "stability_marker_variants")
    run.table(pd.DataFrame(var_cross), "stability_marker_variant_crosstabs")
    run.table(pd.DataFrame(loo_rows), "stability_loo_per_label")
    run.table(per_cell, "stability_per_cell")
    run.table(pd.DataFrame(pair_rows), "crosstab_label_pairs")
    run.table(three, "crosstab_three_way")
    run.table(pd.DataFrame(agree_rows), "crosstab_agreement")
    run.finish({"seeds": list(lib.SEEDS), "noise_sd": lib.NOISE_SD, "noise_repeats": lib.NOISE_REPEATS,
                "noise_rng": f"numpy default_rng({lib.SEED}); N(0, sd) added in float64, cast back to float32",
                "deg_choice": {"resolution": DEG_RESOLUTION, "rule": DEG_RULE},
                "loo_markers": scored, "variants": {k: v[0] for k, v in lib.MARKER_VARIANTS.items()},
                "scanpy": sc.__version__, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
