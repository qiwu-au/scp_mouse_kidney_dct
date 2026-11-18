#!/usr/bin/env python
"""cluster_de steps 1-3: low-resolution Leiden, cluster DE and marker-based annotation for one dataset.



Clusterings (stored neighbour graph of the kidney_object canonical object; Leiden igraph flavor, seed 42):
  KIDNEY: 0.5, 0.8, 1.0 and 2.0 (reference), plus the stored submitted Leiden 2.0 (= kidney_pre_label clusters).
  DCT:    0.3, 0.5, 0.8 and 3.0 (reference), plus the stored V1 Leiden 3.0.
Per clustering: Wilcoxon one-vs-rest DE on ln(1 + intensity) (proteins detected in >= 20% of the
cluster; BH within cluster), the top 25 up proteins (q < 0.05, ranked by AUC), hypergeometric
marker enrichment and mean marker z per cell type, and one label per cluster (rule: cluster_de_lib).
Seed stability (information only): Leiden seeds 0-9 at every non-stored resolution, annotated with the
same rule, compared with seed 42.

Sanity checks run first; if one fails the script writes the check table and stops.

Inputs (read-only): kidney_object canonical object and labels; the dataset's marker workbook (kidney V3 /
  DCT V1); DCT: reviewer_core per-cell clusters (reproduction check); KIDNEY: preprocessing per-cell clusters (information).
Outputs: results/tables/cluster_de/cluster_de_<dataset>_*.csv

Command (project root):
  python scripts/cluster_de_deg_annotation.py --dataset kidney
  python scripts/cluster_de_deg_annotation.py --dataset dct
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
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "cluster_de_deg_annotation.py"
REVIEWER_CORE_CLUSTERS = cc.OUT_ROOTS["tables"] / "reviewer_core" / "reviewer_core_task4_clusters_per_cell.csv"
PREPROCESSING_CLUSTERS = cc.OUT_ROOTS["tables"] / "preprocessing" / "preprocessing_kidney_per_cell_clusters.csv"
# kidney_object README: Leiden 2.0 rerun on the stored kidney graph (igraph, random_state=42) vs stored.
KIDNEY_OBJECT_KIDNEY_LEIDEN2 = {"ari": 0.8823, "n_clusters": 23}
TOL = {"matrix": 1e-4, "ari_t01": 5e-4, "log10p_scipy": 1e-6, "z_scanpy": 1e-3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cluster_de steps 1-3: low-resolution Leiden and DEG-based annotation.")
    parser.add_argument("--dataset", required=True, choices=sorted(t5.DATASETS))
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks, ds: str) -> None:
    run.table(checks.frame(), f"{ds}_sanity_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit(f"{ds}: sanity check failed; see cluster_de_{ds}_sanity_checks.csv (nothing else written).")


def input_checks(checks: t5.Checks, ds: str, adata, labels, intensity: np.ndarray) -> None:
    for col in t5.DATASETS[ds]["label_columns"]:
        observed = labels[col].astype(str).value_counts().to_dict()
        expected = t5.EXPECTED_COUNTS[col]
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in expected),
                   observed == expected)
    d = np.abs(np.log1p(intensity) - cc.dense(adata.layers["ln1p_intensity"]).astype(np.float64)).max()
    checks.add("ln(1 + layers['intensity']) equals layers['ln1p_intensity']", f"max abs diff {d:.3g}",
               d < TOL["matrix"], f"< {TOL['matrix']}")
    n_det = (intensity > 0).sum(axis=1)
    same = bool(np.array_equal(n_det, labels["n_detected_proteins"].to_numpy(int)))
    checks.add("proteins > 0 per cell equal n_detected_proteins", f"{int((n_det == labels['n_detected_proteins']).sum())}"
               f"/{len(n_det)} cells", same)
    if ds == "kidney":
        same = bool((labels["leiden_res_2.00"].astype(str) == labels["stored_leiden_res_2.00"]).all())
        checks.add("stored obs leiden_res_2.00 equals the canonical CSV leiden_res_2.00", f"{same}", same)


def cluster_checks(checks: t5.Checks, run: cc.Run, ds: str, clusterings: dict[str, np.ndarray],
                   cell_ids) -> None:
    spec = t5.DATASETS[ds]
    stored = clusterings[f"stored_r{spec['submitted_resolution']:.1f}"]
    ref = clusterings[f"r{spec['submitted_resolution']:.1f}"]
    ari = float(adjusted_rand_score(stored, ref))
    n_ref, n_stored = len(set(ref)), len(set(stored))
    if ds == "kidney":
        ok = abs(ari - KIDNEY_OBJECT_KIDNEY_LEIDEN2["ari"]) < TOL["ari_t01"] and n_ref == KIDNEY_OBJECT_KIDNEY_LEIDEN2["n_clusters"]
        checks.add("Leiden 2.0 rerun (seed 42) vs stored leiden_res_2.00 reproduces kidney_object (ARI, n clusters)",
                   f"ARI {ari:.4f}; {n_ref} vs {n_stored} clusters (kidney_object: {KIDNEY_OBJECT_KIDNEY_LEIDEN2['ari']}, "
                   f"{KIDNEY_OBJECT_KIDNEY_LEIDEN2['n_clusters']})", ok, f"|dARI| < {TOL['ari_t01']}; same n")
        t04 = pd.read_csv(run.use_input(PREPROCESSING_CLUSTERS, "preprocessing per-cell clusters (information)"),
                          dtype=str).set_index("cell_id").reindex(cell_ids)
        for res in (0.5, 1.0, 2.0):
            col = f"none__z10__all__r{res:.1f}"
            if col in t04 and f"r{res:.1f}" in clusterings:
                a = adjusted_rand_score(t04[col].to_numpy(), clusterings[f"r{res:.1f}"])
                checks.add(f"Leiden {res} on the stored graph vs preprocessing baseline (rebuilt graph) Leiden {res} (ARI)",
                           f"{a:.4f}", None)
    else:
        checks.add(f"Leiden {spec['submitted_resolution']} rerun (seed 42) vs stored V1 Leiden 3.0 (ARI)",
                   f"{ari:.4f}; {n_ref} vs {n_stored} clusters", None)
        t02 = pd.read_csv(run.use_input(REVIEWER_CORE_CLUSTERS, "reviewer_core per-cell clusters (reproduction check)"),
                          dtype=str).set_index("cell_id").reindex(cell_ids)
        for res in spec["resolutions"]:
            col = f"uncorrected__r{res:.2f}"
            a = float(adjusted_rand_score(t02[col].to_numpy(), clusterings[f"r{res:.1f}"]))
            checks.add(f"Leiden {res} equals reviewer_core 'uncorrected' Leiden {res} (same graph and seed; ARI)", f"{a:.6f}",
                       a == 1.0, "= 1")
        a = float(adjusted_rand_score(t02["uncorrected__stored_v1_leiden_res_3.00"].to_numpy(), stored))
        checks.add("stored V1 Leiden 3.0 equals reviewer_core's copy (ARI)", f"{a:.6f}", a == 1.0, "= 1")


def wilcoxon_checks(checks: t5.Checks, cache: t5.RankCache, ln1p: np.ndarray, clusters: np.ndarray, var_names,
                    clustering: str) -> None:
    """Rank-sum implementation vs scipy (one cluster) and vs scanpy rank_genes_groups (all clusters)."""
    first = t5.sort_clusters(clusters)[0]
    mask = clusters == first
    mine = t5.wilcoxon(cache, mask)
    ref = stats.mannwhitneyu(ln1p[mask], ln1p[~mask], axis=0, alternative="two-sided", method="asymptotic",
                             use_continuity=False)
    ok = (mine["p"] > 1e-300) & (ref.pvalue > 1e-300)
    d_p = float(np.max(np.abs(np.log10(mine["p"][ok]) - np.log10(ref.pvalue[ok]))))
    d_u = float(np.max(np.abs(mine["u"] - ref.statistic)))
    checks.add(f"Wilcoxon vs scipy mannwhitneyu ({clustering}, cluster {first}, all proteins): max |dlog10 p|; "
               f"max |dU|", f"{d_p:.3g}; {d_u:.3g}", d_p < TOL["log10p_scipy"] and d_u < 1e-6,
               f"< {TOL['log10p_scipy']}; < 1e-6")
    g = sc.AnnData(X=ln1p.astype(np.float64),
                   obs=pd.DataFrame({"cl": pd.Categorical(clusters)}, index=[str(i) for i in range(len(clusters))]),
                   var=pd.DataFrame(index=pd.Index(list(var_names))))
    sc.tl.rank_genes_groups(g, groupby="cl", method="wilcoxon", tie_correct=True, use_raw=False,
                            n_genes=g.n_vars, corr_method="benjamini-hochberg")
    worst = 0.0
    for c in t5.sort_clusters(clusters):
        df = sc.get.rank_genes_groups_df(g, group=c).set_index("names").reindex(list(var_names))
        z = t5.wilcoxon(cache, clusters == c)["z"]
        worst = max(worst, float(np.nanmax(np.abs(df["scores"].to_numpy(float) - z))))
    checks.add(f"Wilcoxon z vs sc.tl.rank_genes_groups(wilcoxon, tie_correct=True) ({clustering}, all clusters, "
               f"all proteins): max |dz|", f"{worst:.3g}", worst < TOL["z_scanpy"], f"< {TOL['z_scanpy']}")


def evidence_checks(checks: t5.Checks, ev: pd.DataFrame, de: pd.DataFrame) -> None:
    bad = ev[(ev["overlap_k"] > ev["markers_tested_K"]) | (ev["markers_tested_K"] > ev["universe_n_tested"])
             | (ev["overlap_k"] > ev["top_n"])]
    checks.add("hypergeometric inputs: overlap <= markers tested <= tested proteins; overlap <= top n",
               f"{len(bad)} violating rows of {len(ev)}", bad.empty)
    top = de[de["in_top"]]
    worst = int(top.groupby(["clustering", "cluster"]).size().max()) if len(top) else 0
    bad_top = int((~((top["q_bh"] < t5.Q_DE) & top["up"] & (top["pct_detected_cluster"] >= t5.MIN_DETECTION))).sum())
    checks.add("top DE: <= 25 per cluster; all with q < 0.05, up, detected >= 20% in the cluster",
               f"max {worst} per cluster; {bad_top} violating rows", worst <= t5.TOP_N and bad_top == 0)


def cluster_table(asg: pd.DataFrame, de: pd.DataFrame, clusters: np.ndarray, labels: pd.DataFrame, ds: str,
                  clustering: str) -> pd.DataFrame:
    spec = t5.DATASETS[ds]
    ids = labels["n_detected_proteins"].to_numpy(float)
    rows = []
    for c in t5.sort_clusters(clusters):
        mask = clusters == c
        sub = de[de["cluster"] == c]
        top = sub[sub["in_top"]].sort_values("rank_up")["protein"].tolist()
        depth = cc.compare_groups(ids[mask], ids[~mask])
        row = {"clustering": clustering, "cluster": c, "n_cells": int(mask.sum()),
               "median_protein_ids": depth["median_group"], "median_protein_ids_rest": depth["median_reference"],
               "protein_ids_rank_biserial_vs_rest": depth["rank_biserial"], "protein_ids_p_vs_rest": depth["p_value"],
               "n_tested": int(len(sub)), "n_sig_up": int((sub["up"] & (sub["q_bh"] < t5.Q_DE)).sum()),
               "n_top": len(top), "top10_de": ";".join(top[:10])}
        for col in spec["label_columns"]:
            counts = labels.loc[mask, col].astype(str).value_counts()
            row[f"majority_{col}"] = counts.index[0]
            row[f"majority_{col}_fraction"] = float(counts.iloc[0] / mask.sum())
        if ds == "dct":
            row["plate_P1_fraction"] = float((labels.loc[mask, "plate"] == "P1").mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = cc.add_bh(frame, ["clustering"], p_col="protein_ids_p_vs_rest", out_col="protein_ids_q_vs_rest")
    return asg.merge(frame, on=["clustering", "cluster"], how="left")


def main() -> None:
    args = parse_args()
    ds = args.dataset
    spec = t5.DATASETS[ds]
    run = cc.Run(t5.TASK, f"deg_annotation_{ds}", args)
    t_start = time.time()
    adata, labels = t5.load(run, ds)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    checks = t5.Checks(ds)
    print(f"[{ds}] sanity checks", flush=True)
    input_checks(checks, ds, adata, labels, intensity)
    if not checks.ok:
        stop(run, checks, ds)

    ln1p = np.log1p(intensity)
    detected = intensity > 0
    x_z = pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    markers = t5.read_markers(run, ds, var_names)
    coverage = t5.marker_coverage(markers, det_frame.mean(axis=0), spec["order"])
    cache = t5.rank_cache(ln1p, detected)

    # Step 1: clusterings ----------------------------------------------------------------------
    work = adata.copy()
    clusterings: dict[str, np.ndarray] = {}
    for res in spec["resolutions"]:
        clusterings[f"r{res:.1f}"] = t5.leiden(work, res, f"t05_r{res:.1f}").to_numpy()
    clusterings[f"stored_r{spec['submitted_resolution']:.1f}"] = labels[f"stored_{spec['stored_key']}"].to_numpy()
    cluster_checks(checks, run, ds, clusterings, cell_ids)
    for name, cl in clusterings.items():
        checks.add(f"{name}: clusters", f"{len(set(cl))}", None)
    first_low = f"r{spec['low_resolutions'][0]:.1f}"
    wilcoxon_checks(checks, cache, ln1p, clusterings[first_low], var_names, first_low)
    if not checks.ok:
        stop(run, checks, ds)

    # Steps 2-3: DE, evidence, annotation ------------------------------------------------------
    de_all, ev_all, ann_all = [], [], []
    per_cell = labels[["plate", *spec["label_columns"], "n_detected_proteins", f"stored_{spec['stored_key']}"]].copy()
    for name, cl in clusterings.items():
        t0 = time.time()
        de, ev, asg = t5.annotate(cache, cl, var_names, name, markers, x_z, det_frame, ds)
        ann = cluster_table(asg, de, cl, labels, ds, name)
        label_map = ann.set_index("cluster")["deg_label"]
        conf_map = ann.set_index("cluster")["confidence"]
        per_cell[f"{name}__cluster"] = cl
        per_cell[f"{name}__deg_label"] = pd.Series(cl, index=per_cell.index).map(label_map).to_numpy()
        per_cell[f"{name}__deg_confidence"] = pd.Series(cl, index=per_cell.index).map(conf_map).to_numpy()
        de_all.append(de)
        ev_all.append(ev)
        ann_all.append(ann)
        print(f"[{ds}] {name}: {len(set(cl))} clusters, labels "
              f"{dict(ann['deg_label'].value_counts())}, {time.time() - t0:.1f} s", flush=True)
    de_all = pd.concat(de_all, ignore_index=True)
    ev_all = pd.concat(ev_all, ignore_index=True)
    ann_all = pd.concat(ann_all, ignore_index=True)
    evidence_checks(checks, ev_all, de_all)
    n_missing = int(per_cell.filter(like="__deg_label").isna().sum().sum())
    checks.add("every cell has a DEG label in every clustering", f"{n_missing} missing", n_missing == 0)
    if not checks.ok:
        stop(run, checks, ds)

    # Seed stability (information only) --------------------------------------------------------
    stab_rows = []
    for res in spec["resolutions"]:
        name = f"r{res:.1f}"
        ref_cl, ref_lab = clusterings[name], per_cell[f"{name}__deg_label"].to_numpy()
        for seed in (t5.SEED, *t5.STABILITY_SEEDS):
            cl = ref_cl if seed == t5.SEED else t5.leiden(work, res, f"t05_seed_{seed}", seed=seed).to_numpy()
            if seed == t5.SEED:
                lab = ref_lab
            else:
                _, _, asg = t5.annotate(cache, cl, var_names, f"{name}_seed{seed}", markers, x_z, det_frame, ds)
                lab = pd.Series(cl).map(asg.set_index("cluster")["deg_label"]).to_numpy()
            row = {"clustering": name, "resolution": res, "seed": seed, "n_clusters": len(set(cl)),
                   "ari_clusters_vs_seed42": float(adjusted_rand_score(ref_cl, cl)),
                   "ari_deg_labels_vs_seed42": float(adjusted_rand_score(ref_lab, lab)),
                   "fraction_same_deg_label_as_seed42": float((ref_lab == lab).mean()),
                   "deg_labels_present": ";".join(cc.ordered_labels(lab, spec["order"]))}
            for col in spec["label_columns"]:
                truth = labels[col].to_numpy().astype(str)
                row[f"ari_deg_labels_vs_{col}"] = t5.agreement(lab, truth)["ari"]
                row[f"accuracy_vs_{col}"] = float((lab == truth).mean())
            stab_rows.append(row)
        print(f"[{ds}] seed stability {name} done", flush=True)

    run.table(checks.frame(), f"{ds}_sanity_checks")
    run.table(coverage, f"{ds}_marker_coverage")
    run.table(per_cell, f"{ds}_clusters_per_cell", index=True)
    run.table(de_all[de_all["in_top"]].sort_values(["clustering", "cluster", "rank_up"]), f"{ds}_de_top25")
    run.table(de_all[de_all["q_bh"] < t5.Q_DE], f"{ds}_de_significant")
    run.table(ev_all, f"{ds}_marker_evidence")
    run.table(ann_all, f"{ds}_cluster_annotation")
    run.table(pd.DataFrame(stab_rows), f"{ds}_seed_stability")
    run.finish({"clusterings": list(clusterings), "resolutions": list(spec["resolutions"]),
                "stored_clusters": spec["stored_key"], "graph": "stored neighbour graph of the kidney_object canonical object",
                "leiden": "sc.tl.leiden flavor='igraph', random_state 42 (as reviewer_core-preprocessing)",
                "stability_seeds": list(t5.STABILITY_SEEDS),
                "de": ("Wilcoxon rank-sum one cluster vs all other cells, ln(1 + intensity), missing = 0, normal "
                       "approximation with tie correction, no continuity correction; BH within cluster over tested "
                       "proteins"),
                "min_detection_in_cluster": t5.MIN_DETECTION, "top_n": t5.TOP_N, "q_de": t5.Q_DE,
                "top_ranking": "AUC desc, then q asc, then mean ln1p difference desc; up = AUC > 0.5 and mean diff > 0",
                "hypergeometric": ("P(X >= k), background = proteins tested in the cluster, K = type markers among "
                                   "them, n = top DE size; BH over all cluster x type tests of one clustering"),
                "mean_marker_z": "cluster mean of each cell's mean canonical .X (z-scored ln1p, clip 10) over the "
                                 "type's markers present in the matrix",
                "q_hyper_high": t5.Q_HYPER, "shared_marker_rule": ">= 2 cell types list the marker in the dataset's "
                                                                  "own workbook (authors, 13 Oct 2026)",
                "tolerances": TOL, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "scanpy": sc.__version__,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
