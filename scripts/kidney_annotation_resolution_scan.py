#!/usr/bin/env python
"""kidney_annotation step 2, plan steps 2-3: Leiden resolution scan and cluster-level annotation (kidney).

Plan:
the analysis plan (settings agreed with the authors, 14 Oct 2026).

For each graph of kidney_annotation_graphs.py ((a) stored, (b) depth-regressed, (c) preprocessing total . unscaled . hvg500):
  Leiden (igraph flavour) at 0.5-6.0 in steps of 0.5, seeds 42 and 0-9. Every clustering is annotated with the
  cluster_de rule (cluster_de_lib: top-25 DE, hypergeometric marker enrichment, shared-marker confidence, Unknown without
  evidence); clusters with < 5 cells are reported, not annotated ('below_min_size').
Per graph x resolution: number of clusters, min / median size, seed stability (ARI of seeds 0-9 vs seed 42; types
resolved and labels per seed), eta^2 of protein IDs on cluster, median IDs per cluster, types resolved (with
confidence), clusters per type, Unknown fraction, and agreement with kidney_reannot_label and kidney_pre_label
(ARI, NMI, per-label recall and precision).

Inputs (read-only): kidney_object canonical kidney object and labels; data/KIDNEY/KIDNEY_markers_V3.xlsx;
  results/h5ad/kidney_annotation_kidney_scan_graphs.h5ad; cluster_de and preprocessing per-cell clusters (reproduction checks).
Outputs: results/tables/kidney_annotation/kidney_annotation_scan_*.csv

Command (project root):  python scripts/kidney_annotation_resolution_scan.py
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
import kidney_annotation_scan_lib as s
from cluster_de_deg_annotation import cluster_table, evidence_checks

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation step 2 / plan steps 2-3: resolution scan and annotation.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "scan_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_annotation_scan_checks.csv (nothing else written).")


def types_text(types: dict[str, str]) -> str:
    return ";".join(f"{t}({c})" for t, c in types.items())


def clustering_row(graph: str, res: float, seed: int, cl: np.ndarray, lab: np.ndarray, asg: pd.DataFrame,
                   ids: np.ndarray, ref_cl: np.ndarray, ref_lab: np.ndarray) -> dict:
    sizes = pd.Series(cl).value_counts()
    types = s.types_resolved(asg)
    medians = pd.Series(ids).groupby(cl).median()
    return {"graph": graph, "resolution": res, "seed": seed, "n_clusters": int(len(sizes)),
            "cluster_size_min": int(sizes.min()), "cluster_size_median": float(sizes.median()),
            "cluster_size_max": int(sizes.max()),
            "n_clusters_below_min": int((sizes < s.MIN_CLUSTER_SIZE).sum()),
            "n_cells_below_min": int(sizes[sizes < s.MIN_CLUSTER_SIZE].sum()),
            "n_clusters_unknown": int((asg["deg_label"] == s.UNKNOWN).sum()),
            "fraction_cells_unknown": float((lab == s.UNKNOWN).mean()),
            "fraction_cells_below_min": float((lab == s.BELOW_MIN).mean()),
            "n_types_resolved": len(types),
            "n_types_resolved_ge_medium": sum(c in ("high", "medium") for c in types.values()),
            "n_types_high": sum(c == "high" for c in types.values()),
            "types_resolved": types_text(types),
            "eta2_protein_ids": s.eta_squared(ids, cl), "eta2_ln_protein_ids": s.eta_squared(np.log(ids), cl),
            "kruskal_p_protein_ids": s.kruskal_p(ids, cl),
            "cluster_median_ids_min": float(medians.min()), "cluster_median_ids_max": float(medians.max()),
            "ari_clusters_vs_seed42": float(adjusted_rand_score(ref_cl, cl)),
            "ari_labels_vs_seed42": float(adjusted_rand_score(ref_lab, lab)),
            "fraction_same_label_as_seed42": float((ref_lab == lab).mean())}


def type_rows(graph: str, res: float, asg: pd.DataFrame, cl: np.ndarray, per_seed_types: list[dict],
              types: list[str] | None = None) -> list[dict]:
    rows = []
    sizes = pd.Series(cl).value_counts()
    workbook_types = types if types is not None else [t for t in s.ORDER if t != s.UNKNOWN]
    for t in workbook_types:
        sub = asg[asg["deg_label"] == t]
        best = s.types_resolved(sub).get(t, "") if len(sub) else ""
        rows.append({"graph": graph, "resolution": res, "cell_type": t, "n_clusters_seed42": int(len(sub)),
                     "n_cells_seed42": int(sizes.reindex(sub["cluster"]).sum()) if len(sub) else 0,
                     "best_confidence_seed42": best,
                     "clusters_seed42": ";".join(f"{c}({sizes[c]}; {conf})" for c, conf in
                                                 zip(sub["cluster"], sub["confidence"])),
                     "n_seeds_resolved": sum(t in d for d in per_seed_types),
                     "n_seeds_resolved_ge_medium": sum(d.get(t) in ("high", "medium") for d in per_seed_types),
                     "n_seeds": len(per_seed_types)})
    return rows


def main() -> None:
    args = parse_args()
    run = cc.Run(s.TASK, "resolution_scan", args)
    t_start = time.time()
    adata, labels = t5.load(run, s.DATASET)
    store = s.load_graphs(run)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    ln1p = np.log1p(intensity)
    detected = intensity > 0
    ids = labels["n_detected_proteins"].to_numpy(float)
    x_z = pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    markers = t5.read_markers(run, s.DATASET, var_names)
    cache = t5.rank_cache(ln1p, detected)
    checks = t5.Checks(s.DATASET)

    print("[kidney] input checks", flush=True)
    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        expected = t5.EXPECTED_COUNTS[col]
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in expected),
                   observed == expected)
    same = list(store.obs_names.astype(str)) == list(cell_ids)
    checks.add("graph object has the canonical cells in the canonical order", f"{same}", same)
    d = float(abs(store.obsp["connectivities__stored"] - adata.obsp["connectivities"]).max())
    checks.add("graph object: stored connectivities equal the canonical object's", f"max abs diff {d:.3g}", d == 0, "= 0")
    if not checks.ok:
        stop(run, checks)

    t05 = pd.read_csv(run.use_input(s.CLUSTER_DE_CLUSTERS, "cluster_de kidney per-cell clusters (reproduction check)"),
                      dtype=str).set_index("cell_id").reindex(cell_ids)
    t04 = pd.read_csv(run.use_input(s.PREPROCESSING_CLUSTERS, "preprocessing kidney per-cell clusters (reproduction check)"),
                      dtype=str).set_index("cell_id").reindex(cell_ids)

    per_cell = labels[[*s.LABEL_COLUMNS, "n_detected_proteins", "stored_leiden_res_2.00"]].copy()
    seed_cols: dict[str, np.ndarray] = {}
    seed_rows, ann_all, ev_all, de_top_all, type_all, agree_rows, recov_rows = [], [], [], [], [], [], []
    for graph in s.GRAPH_NAMES:
        g = s.graph_adata(store, graph)
        for res in s.RESOLUTIONS:
            t0 = time.time()
            ref_cl = ref_lab = None
            per_seed_types = []
            for seed in s.SEEDS:
                name = s.clustering_name(graph, res, seed)
                cl = t5.leiden(g, res, "t15_tmp", seed=seed).to_numpy().astype(str)
                de, ev, asg = s.annotate_clustering(cache, cl, var_names, name, markers, x_z, det_frame)
                lab = s.per_cell_labels(cl, asg).astype(str)
                if seed == s.SEED:
                    ref_cl, ref_lab = cl, lab
                    conf = s.per_cell_labels(cl, asg, "confidence")
                    per_cell[f"{name}__cluster"] = cl
                    per_cell[f"{name}__label"] = lab
                    per_cell[f"{name}__confidence"] = conf
                    ann = cluster_table(asg, de, cl, labels, s.DATASET, name)
                    ann.insert(1, "graph", graph)
                    ann.insert(2, "resolution", res)
                    ann["below_min_size"] = ann["n_cells"] < s.MIN_CLUSTER_SIZE
                    ann_all.append(ann)
                    ev.insert(1, "graph", graph)
                    ev.insert(2, "resolution", res)
                    ev_all.append(ev)
                    de_top_all.append(de[de["in_top"]].sort_values(["cluster", "rank_up"]))
                    for col in s.LABEL_COLUMNS:
                        truth = labels[col].astype(str).to_numpy()
                        a_cl, a_lab = t5.agreement(cl, truth), t5.agreement(lab, truth)
                        annotated = ~np.isin(lab, [s.UNKNOWN, s.BELOW_MIN])
                        agree_rows.append({
                            "graph": graph, "resolution": res, "label_column": col, "n_clusters": len(set(cl)),
                            "ari_clusters": a_cl["ari"], "nmi_clusters": a_cl["nmi"],
                            "cluster_majority_accuracy": t5.majority_accuracy(cl, truth),
                            "ari_t15_labels": a_lab["ari"], "nmi_t15_labels": a_lab["nmi"],
                            "accuracy_t15_labels": float((lab == truth).mean()),
                            "n_cells_annotated_type": int(annotated.sum()),
                            "accuracy_annotated_type_cells": float((lab[annotated] == truth[annotated]).mean())
                            if annotated.any() else np.nan})
                        rec = t5.recovery(lab, truth, s.KIDNEY_ANNOTATION_ORDER)
                        rec.insert(0, "label_column", col)
                        rec.insert(0, "resolution", res)
                        rec.insert(0, "graph", graph)
                        recov_rows.append(rec)
                else:
                    seed_cols[f"{name}__cluster"] = cl
                    seed_cols[f"{name}__label"] = lab
                per_seed_types.append(s.types_resolved(asg))
                seed_rows.append(clustering_row(graph, res, seed, cl, lab, asg, ids, ref_cl, ref_lab))
            type_all.extend(type_rows(graph, res, ann_all[-1][["cluster", "deg_label", "confidence"]],
                                      per_cell[f"{s.clustering_name(graph, res)}__cluster"].to_numpy(), per_seed_types))
            print(f"[kidney] {graph} r{res:.1f}: {seed_rows[-len(s.SEEDS)]['n_clusters']} clusters (seed 42), types "
                  f"{seed_rows[-len(s.SEEDS)]['types_resolved'] or '-'}; {time.time() - t0:.1f} s", flush=True)

        # Reproduction checks on the graph read from the h5ad
        if graph == "stored":
            for res in (0.5, 1.0, 2.0):
                name = s.clustering_name(graph, res)
                ref = t05[f"r{res:.1f}__cluster"].to_numpy()
                a = float(adjusted_rand_score(ref, per_cell[f"{name}__cluster"]))
                checks.add(f"graph stored: Leiden {res} (seed 42) equals cluster_de r{res:.1f} (ARI)", f"{a:.6f}", a == 1.0, "= 1")
                t05_min = int(pd.Series(ref).value_counts().min())
                same_lab = int((t05[f"r{res:.1f}__deg_label"].to_numpy() == per_cell[f"{name}__label"].to_numpy()).sum())
                same_conf = int((t05[f"r{res:.1f}__deg_confidence"].to_numpy()
                                 == per_cell[f"{name}__confidence"].to_numpy()).sum())
                checks.add(f"graph stored: Leiden {res} labels and confidences equal cluster_de (smallest cluster_de cluster "
                           f"{t05_min} cells, so no cluster is below the minimum size)",
                           f"labels {same_lab}/712; confidences {same_conf}/712",
                           (same_lab == 712 and same_conf == 712) if t05_min >= s.MIN_CLUSTER_SIZE else None)
        if graph == s.GRAPH_NAMES[2]:
            for res in (0.5, 1.0, 2.0, 3.0):
                name = s.clustering_name(graph, res)
                a = float(adjusted_rand_score(t04[f"{s.PREPROCESSING_VARIANT}__r{res:.1f}"].to_numpy(), per_cell[f"{name}__cluster"]))
                checks.add(f"graph {graph}: Leiden {res} (seed 42) equals preprocessing (ARI)", f"{a:.6f}", a == 1.0, "= 1")
        if not checks.ok:
            stop(run, checks)

    ann_all = pd.concat(ann_all, ignore_index=True)
    ev_all = pd.concat(ev_all, ignore_index=True)
    de_top_all = pd.concat(de_top_all, ignore_index=True)
    evidence_checks(checks, ev_all, de_top_all)
    n_missing = int(per_cell.filter(like="__label").isna().sum().sum()) + sum(int(pd.isna(v).sum())
                                                                            for k, v in seed_cols.items())
    checks.add("every cell has a label in every clustering (all seeds)", f"{n_missing} missing", n_missing == 0)
    small_annot = ann_all[ann_all["below_min_size"] & (ann_all["deg_label"] != s.BELOW_MIN)]
    big_unannot = ann_all[~ann_all["below_min_size"] & (ann_all["deg_label"] == s.BELOW_MIN)]
    checks.add("clusters < 5 cells are labelled below_min_size and only those", f"{len(small_annot)} + "
               f"{len(big_unannot)} violating clusters", small_annot.empty and big_unannot.empty)
    if not checks.ok:
        stop(run, checks)

    # Summary per graph x resolution ----------------------------------------------------------------
    seeds = pd.DataFrame(seed_rows)
    rows = []
    for (graph, res), grp in seeds.groupby(["graph", "resolution"], sort=False):
        ref = grp[grp["seed"] == s.SEED].iloc[0].to_dict()
        other = grp[grp["seed"] != s.SEED]
        ann = ann_all[(ann_all["graph"] == graph) & (ann_all["resolution"] == res) & ~ann_all["below_min_size"]]
        rho = stats.spearmanr(ann["median_protein_ids"], ann["n_sig_up"]).statistic if len(ann) > 2 else np.nan
        row = {k: ref[k] for k in ref if k not in ("seed", "ari_clusters_vs_seed42", "ari_labels_vs_seed42",
                                                   "fraction_same_label_as_seed42")}
        row["display"] = s.graph_display(graph)
        row["rho_cluster_median_ids_vs_n_sig_up"] = float(rho)
        for key in ("n_clusters", "n_types_resolved", "n_types_resolved_ge_medium", "fraction_cells_unknown",
                    "eta2_protein_ids"):
            row[f"{key}_all_seeds_min"] = float(grp[key].min())
            row[f"{key}_all_seeds_max"] = float(grp[key].max())
        for key in ("ari_clusters_vs_seed42", "ari_labels_vs_seed42", "fraction_same_label_as_seed42"):
            row[f"{key}_median"] = float(other[key].median())
            row[f"{key}_min"] = float(other[key].min())
            row[f"{key}_max"] = float(other[key].max())
        row["type_sets_seen_all_seeds"] = " | ".join(sorted(set(
            ";".join(t.split("(")[0] for t in v.split(";") if t) for v in grp["types_resolved"])))
        for a in [r for r in agree_rows if r["graph"] == graph and r["resolution"] == res]:
            short = "reannot" if a["label_column"] == "kidney_reannot_label" else "pre"
            for key in ("ari_clusters", "nmi_clusters", "cluster_majority_accuracy", "ari_t15_labels",
                        "nmi_t15_labels", "accuracy_t15_labels"):
                row[f"{key}_vs_{short}"] = a[key]
        rows.append(row)
    summary = pd.DataFrame(rows)
    front = ["graph", "display", "resolution"]
    summary = summary[front + [c for c in summary.columns if c not in front]]
    for row in summary[summary["resolution"].isin([0.5, 1.0, 2.0, 4.0, 6.0])].itertuples(index=False):
        checks.add(f"{row.graph} r{row.resolution}: clusters; types (any / >= medium); Unknown; eta2",
                   f"{row.n_clusters}; {row.n_types_resolved} / {row.n_types_resolved_ge_medium}; "
                   f"{row.fraction_cells_unknown:.2f}; {row.eta2_protein_ids:.3f}", None)

    run.table(checks.frame(), "scan_checks")
    run.table(summary, "scan_summary")
    run.table(seeds, "scan_seed_stability")
    run.table(ann_all, "scan_cluster_annotation")
    run.table(ev_all, "scan_marker_evidence")
    run.table(de_top_all, "scan_de_top25")
    run.table(pd.DataFrame(type_all), "scan_types_by_resolution")
    run.table(pd.DataFrame(agree_rows), "scan_agreement")
    run.table(pd.concat(recov_rows, ignore_index=True), "scan_recovery")
    run.table(per_cell, "scan_clusters_per_cell", index=True)
    run.table(pd.DataFrame(seed_cols, index=per_cell.index), "scan_seed_clusters_per_cell", index=True)
    run.finish({"graphs": list(s.GRAPH_NAMES), "resolutions": list(s.RESOLUTIONS), "seeds": list(s.SEEDS),
                "min_cluster_size": s.MIN_CLUSTER_SIZE, "below_min_label": s.BELOW_MIN,
                "leiden": "sc.tl.leiden flavor='igraph' (cluster_de_lib.leiden)",
                "annotation": ("cluster_de_lib.de_one_vs_rest / marker_evidence / assign (cluster_de rule, top 25), on the "
                               "canonical ln(1 + intensity) for every graph; clusters < min size dropped before "
                               "marker_evidence (BH over the annotated clusters' cluster x type tests)"),
                "eta2": "one-way eta^2 of n_detected_proteins on cluster (also ln)",
                "types_resolved": "distinct non-Unknown labels over annotated clusters; best confidence per type",
                "scanpy": sc.__version__, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
