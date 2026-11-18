#!/usr/bin/env python
"""dct_labels step 4: stability of the three label columns per condition (combined search; labels_core step 3).

Analysis plan, step 4.
The logic is labels_core_stability.main, run per condition with the cell count and column names as parameters
(cs_<cond>_cell_label, cs_<cond>_cluster_label and the primary cs_ctrl_deg_label / cs_zerok_pipeline_label):

1. Leiden seeds 42 and 0-9 on the step-1 graph: Leiden 3.0 + the original cluster rule (scores fixed) and Leiden 1.0
   + the rank-free DE rule (seed clusters from the step-2 scan). The per-cell label does not use Leiden.
2. Noise N(0, 1e-6) added to the scaled .X (float32), 20 repeats (numpy default_rng(42), re-seeded per condition):
   (A) noisy .X scored -> per-cell argmax (and the cluster rule on the fixed Leiden 3.0 clusters);
   (B) noisy .X -> neighbours (k = 10, 20 PCs) -> Leiden 3.0 and 1.0 (seed 42) -> cluster rule and DE rule.
   A no-noise control run must reproduce steps 1 and 2.
3. Leave-one-marker-out over the scored markers; noSlc12a3; noSlc12a3_noUmod (clusterings fixed).
4. Crosstabs of the three columns (pairwise and three-way).
The ZeroK transfer label is not re-derived here; its checks are in step 3.

Inputs (read-only): results/h5ad/dct_labels_DCT_{ctrl,zerok}_step1_submitted.h5ad; dct_labels_submitted_per_cell.csv;
dct_labels_deg_labels_per_cell.csv; dct_labels_scan_seed_clusters_per_cell.csv; data/DCT/DCT_markers_V1.xlsx.
Outputs: results/tables/dct_labels/dct_labels_stability_*.csv, dct_labels_crosstab_*.csv (long, with condition).

Command (project root):  python scripts/dct_labels_stability.py
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
import labels_core_lib as t18
import dct_labels_lib as lib
from labels_core_stability import Labeller, change_row, crosstab_long, rebuild_graph

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "dct_labels_stability.py"
NOISE_KEYS = ("A_cell", "A_cluster", "B_cluster_graph", "B_cluster_full", "B_deg")


def one_condition(run: cc.Run, cond: str, checks: t5.Checks, step1_all, deg_all, seeds_all) -> dict[str, pd.DataFrame]:
    adata = lib.read_step1(run, cond)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    n = adata.n_obs
    step1 = lib.per_condition(step1_all, cond, cell_ids)
    deg = lib.per_condition(deg_all, cond, cell_ids)
    seeds_deg = lib.per_condition(seeds_all, cond, cell_ids)
    c_cell, c_cluster, c_deg = lib.cs(cond, "cell_label"), lib.cs(cond, "cluster_label"), lib.cs(cond, "deg_label")
    columns = (c_cell, c_cluster, c_deg)

    X = np.array(cc.dense(adata.X), dtype=np.float32, copy=True)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    x_z = pd.DataFrame(X.astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    workbook = t18.read_workbook(run)
    lists = t18.present_lists(workbook, var_names)
    markers = t18.dct_markers(run, var_names)
    cache = t5.rank_cache(np.log1p(intensity), detected)
    L = Labeller(adata, lists, markers, cache, x_z, det_frame, var_names)

    ref = {c_cell: step1["cell_label"].astype(str).to_numpy(),
           c_cluster: step1["cluster_label"].astype(str).to_numpy(),
           c_deg: deg["deg_label"].astype(str).to_numpy()}
    cl3 = step1["leiden3_cluster"].astype(str).to_numpy()
    cl1 = deg["deg_cluster"].astype(str).to_numpy()

    # Reproduction on fixed inputs
    label0, max0, scores0 = L.cell(X)
    n_cell = int((label0.to_numpy() == ref[c_cell]).sum())
    d_sc = float(np.abs(scores0.to_numpy() - step1[[f"{t}_score" for t in lists]].to_numpy()).max())
    checks.add(f"{cond}: rescoring the stored .X reproduces step-1 scores and {c_cell}",
               f"max abs diff {d_sc:.3g}; {n_cell}/{n}", d_sc < 1e-6 and n_cell == n)
    n_clu = int((L.cluster(label0, max0, cl3) == ref[c_cluster]).sum())
    checks.add(f"{cond}: cluster rule on the Leiden 3.0 clusters reproduces {c_cluster}", f"{n_clu}/{n}", n_clu == n)
    lab_d, _, de1 = L.deg(cl1)
    n_deg = int((lab_d == ref[c_deg]).sum())
    checks.add(f"{cond}: rank-free rule on the Leiden 1.0 clusters reproduces {c_deg}", f"{n_deg}/{n}", n_deg == n)
    if not checks.ok:
        lib.stop(run, checks, "stability_checks")

    # 1. Leiden seeds
    print(f"[{cond}] Leiden seeds", flush=True)
    g = t18.graph_adata(adata)
    seed_rows, seed_labels = [], {c_cluster: {}, c_deg: {}}
    for seed in t18.SEEDS:
        c3 = t18.leiden_on(g, t18.SUBMITTED_RESOLUTION, seed)
        lab3 = L.cluster(label0, max0, c3)
        if seed == lib.SEED:
            same = bool((c3 == cl3).all())
            checks.add(f"{cond}: Leiden 3.0 seed 42 on the stored graph equals {lib.cs(cond, 'leiden3_cluster')}",
                       f"{same}", same)
            c1, lab1 = cl1, ref[c_deg]
        else:
            c1 = seeds_deg[f"{t18.clustering_name(lib.RESOLUTION, seed)}__cluster"].astype(str).to_numpy()
            lab1 = seeds_deg[f"{t18.clustering_name(lib.RESOLUTION, seed)}__{lib.RULE}_label"].astype(str).to_numpy()
        seed_labels[c_cluster][seed] = lab3
        seed_labels[c_deg][seed] = lab1
        for col, cl, ref_cl, lab in ((c_cluster, c3, cl3, lab3), (c_deg, c1, cl1, lab1)):
            seed_rows.append({"label_column": col, "seed": seed, "n_clusters": int(len(set(cl))),
                              "ari_clusters_vs_seed42": float(adjusted_rand_score(ref_cl, cl)),
                              "ari_labels_vs_seed42": float(adjusted_rand_score(ref[col], lab)),
                              "fraction_same_label_as_seed42": float((ref[col] == lab).mean()),
                              "label_counts": lib.counts_text(lab)})
        seed_rows.append({"label_column": c_cell, "seed": seed, "n_clusters": np.nan,
                          "ari_clusters_vs_seed42": np.nan, "ari_labels_vs_seed42": 1.0,
                          "fraction_same_label_as_seed42": 1.0, "label_counts": lib.counts_text(ref[c_cell]),
                          "note": "per-cell argmax does not use Leiden"})
    if not checks.ok:
        lib.stop(run, checks, "stability_checks")
    seeds = pd.DataFrame(seed_rows)
    per_cell = pd.DataFrame({"cell_id": cell_ids, "cell_key": step1["cell_key"].to_numpy(),
                             "plate": step1["plate"].to_numpy(), "n_detected": step1["n_detected"].to_numpy(),
                             "cell_label": ref[c_cell], "cluster_label": ref[c_cluster], "deg_label": ref[c_deg]})
    per_cell["seed_stability_cell_label"] = 1.0
    for col, generic in ((c_cluster, "cluster_label"), (c_deg, "deg_label")):
        others = np.column_stack([seed_labels[col][s] for s in t18.SEEDS if s != lib.SEED])
        per_cell[f"seed_stability_{generic}"] = (others == ref[col][:, None]).mean(axis=1)

    # 2. Noise tests
    print(f"[{cond}] noise tests (control run, then {t18.NOISE_REPEATS} repeats)", flush=True)
    rng = np.random.default_rng(lib.SEED)
    noise_rows = []
    n_changed = {k: np.zeros(n, int) for k in NOISE_KEYS}
    for rep in range(t18.NOISE_REPEATS + 1):
        t0 = time.time()
        if rep == 0:
            Xn = X.copy()                                    # control: no noise
        else:
            Xn = (X.astype(np.float64) + rng.normal(0.0, t18.NOISE_SD, X.shape)).astype(np.float32)
        n_entries = int((Xn != X).sum())
        lab_a, max_a, scores_a = L.cell(Xn)
        lab_a = lab_a.to_numpy()
        d_scores = float(np.abs(scores_a.to_numpy() - scores0.to_numpy()).max())
        lab_a_cluster = L.cluster(pd.Series(lab_a, index=cell_ids), max_a, cl3)
        gn = rebuild_graph(Xn, cell_ids)
        c3n = t18.leiden_on(gn, t18.SUBMITTED_RESOLUTION, lib.SEED)
        c1n = t18.leiden_on(gn, lib.RESOLUTION, lib.SEED)
        lab_b_graph = L.cluster(label0, max0, c3n)
        lab_b_full = L.cluster(pd.Series(lab_a, index=cell_ids), max_a, c3n)
        lab_b_deg, _, _ = L.deg(c1n, x_z=pd.DataFrame(Xn.astype(np.float64), index=cell_ids, columns=var_names))
        extra = {"n_x_entries_changed_after_float32": n_entries}
        results = [
            ("A: noisy .X scored", c_cell, lab_a, {"max_abs_score_change": d_scores}),
            ("A: noisy scores, fixed Leiden 3.0 clusters", c_cluster, lab_a_cluster, {}),
            ("B: noisy graph, original scores", c_cluster, lab_b_graph,
             {"ari_clusters_vs_original": float(adjusted_rand_score(cl3, c3n)), "n_clusters": len(set(c3n))}),
            ("B: noisy .X, whole pipeline", c_cluster, lab_b_full,
             {"ari_clusters_vs_original": float(adjusted_rand_score(cl3, c3n)), "n_clusters": len(set(c3n))}),
            ("B: noisy graph, rank-free DE rule at Leiden 1.0", c_deg, lab_b_deg,
             {"ari_clusters_vs_original": float(adjusted_rand_score(cl1, c1n)), "n_clusters": len(set(c1n))}),
        ]
        for key, (test, col, lab, ex) in zip(NOISE_KEYS, results):
            noise_rows.append(change_row(test, rep, col, ref[col], lab, **extra, **ex))
            if rep > 0:
                n_changed[key] += (lab != ref[col]).astype(int)
        if rep == 0:
            for test, col, lab, ex in results:
                same = int((lab == ref[col]).sum())
                ok = same == n and ex.get("ari_clusters_vs_original", 1.0) == 1.0
                checks.add(f"{cond}: no-noise control reproduces {col} ({test})", f"{same}/{n}", ok)
            if not checks.ok:
                lib.stop(run, checks, "stability_checks")
        print(f"  repeat {rep}: changed A cell {noise_rows[-5]['n_changed']}, B cluster {noise_rows[-2]['n_changed']}, "
              f"B deg {noise_rows[-1]['n_changed']}; {time.time() - t0:.1f} s", flush=True)
    noise = pd.DataFrame(noise_rows)
    for key, col in (("A_cell", "noiseA_n_repeats_changed_cell_label"),
                     ("A_cluster", "noiseA_n_repeats_changed_cluster_label"),
                     ("B_cluster_graph", "noiseB_graph_n_repeats_changed_cluster_label"),
                     ("B_cluster_full", "noiseB_full_n_repeats_changed_cluster_label"),
                     ("B_deg", "noiseB_n_repeats_changed_deg_label")):
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
    noise_summary["n_cells_changed_in_any_repeat"] = [int((v > 0).sum()) for v in n_changed.values()]

    # 3. Marker lists
    print(f"[{cond}] marker-list variants", flush=True)
    scored = list(dict.fromkeys(g for gs in lists.values() for g in gs))
    variants = {"original": ("original lists", ())}
    variants.update(t18.MARKER_VARIANTS)
    variants.update({f"loo_{g}": (f"{g} removed from every list", (g,)) for g in scored})
    var_rows, var_cross, loo_keep, defs = [], [], {c: [] for c in columns}, []
    for vname, (desc, genes) in variants.items():
        lists_v = t18.drop_from_lists(lists, genes)
        markers_v = t18.drop_markers(markers, genes)
        lab_c, max_c, _ = L.cell(X, lists_v)
        lab_c = lab_c.to_numpy()
        lab_k = L.cluster(pd.Series(lab_c, index=cell_ids), max_c, cl3)
        lab_dv, _, _ = L.deg(cl1, markers=markers_v, de=de1)
        defs.append({"variant": vname, "description": desc, "removed": ";".join(genes),
                     **{f"{t}_scored": ";".join(lists_v.get(t, [])) for t in lists},
                     **{f"{t}_specific_for_de_rule": ";".join(t18.specific_present(markers_v).get(t, [])) for t in lists}})
        for col, lab in ((c_cell, lab_c), (c_cluster, lab_k), (c_deg, lab_dv)):
            var_rows.append(change_row(vname, 0, col, ref[col], lab, removed=";".join(genes)))
            if vname.startswith("loo_"):
                loo_keep[col].append(lab == ref[col])
            elif vname != "original":
                var_cross.extend(crosstab_long(ref[col], lab, "original_label", "variant_label", variant=vname,
                                               label_column=col))
    variants_tab = pd.DataFrame(var_rows).drop(columns=["repeat"]).rename(columns={"test": "variant"})
    orig = variants_tab[variants_tab["variant"] == "original"]
    bad = orig[orig["n_changed"] != 0]
    checks.add(f"{cond}: variant 'original' reproduces all three label columns", f"{len(bad)} columns differ", bad.empty)
    loo_rows = []
    for col, generic in zip(columns, ("cell_label", "cluster_label", "deg_label")):
        keep = np.column_stack(loo_keep[col]).mean(axis=1)
        per_cell[f"loo_stability_{generic}"] = keep
        for label in [*cc.ordered_labels(ref[col], t18.LABEL_ORDER), "all"]:
            m = np.ones(len(keep), bool) if label == "all" else ref[col] == label
            loo_rows.append({"label_column": col, "original_label": label, "n_cells": int(m.sum()),
                             "n_loo_runs": len(loo_keep[col]), "mean_stability": float(keep[m].mean()),
                             "median_stability": float(np.median(keep[m])), "n_stability_1": int((keep[m] == 1).sum()),
                             "n_stability_below_0.5": int((keep[m] < 0.5).sum()), "n_stability_0": int((keep[m] == 0).sum())})

    seed_label_rows = []
    for col, generic in zip(columns, ("cell_label", "cluster_label", "deg_label")):
        st = per_cell[f"seed_stability_{generic}"].to_numpy()
        for label in [*cc.ordered_labels(ref[col], t18.LABEL_ORDER), "all"]:
            m = np.ones(len(st), bool) if label == "all" else ref[col] == label
            seed_label_rows.append({"label_column": col, "label_seed42": label, "n_cells": int(m.sum()),
                                    "mean_stability": float(st[m].mean()), "n_stable_all_10_seeds": int((st[m] == 1).sum()),
                                    "fraction_stable_all_10_seeds": float((st[m] == 1).mean()),
                                    "n_stability_below_0.5": int((st[m] < 0.5).sum())})

    # 4. Crosstabs
    pairs = ((c_cell, c_cluster), (c_cell, c_deg), (c_cluster, c_deg))
    pair_rows = []
    for a, b in pairs:
        pair_rows.extend(crosstab_long(ref[a], ref[b], "row_label", "column_label", row_column=a, column_column=b))
    three = (pd.DataFrame({c: ref[c] for c in columns}).groupby(list(columns)).size().rename("n_cells").reset_index()
             .sort_values("n_cells", ascending=False)
             .rename(columns={c_cell: "cell_label", c_cluster: "cluster_label", c_deg: "deg_label"}))
    agree_rows = [{"pair": f"{a} vs {b}", "fraction_equal": float((ref[a] == ref[b]).mean()),
                   "ari": float(adjusted_rand_score(ref[a], ref[b]))} for a, b in pairs]
    all_three = float(((ref[c_cell] == ref[c_cluster]) & (ref[c_cluster] == ref[c_deg])).mean())
    agree_rows.append({"pair": "all three equal", "fraction_equal": all_three, "ari": np.nan})

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
        {"quantity": "noise SD added in tests A and B", "value": t18.NOISE_SD},
        {"quantity": "noise A: max |score change| over repeats", "value": float(a["max_abs_score_change"].max())},
        *[{"quantity": f"{t}_score SD (step 1)", "value": float(scores0[t].std(ddof=1))} for t in lists],
    ])

    for r in noise_summary.itertuples(index=False):
        checks.add(f"{cond}: noise {r.test} [{r.label_column}]: cells changed per repeat median (max); repeats with any "
                   f"change", f"{r.n_changed_median:g} ({r.n_changed_max}); {r.n_repeats_any_change}/{r.n_repeats}", None)
    for r in pd.DataFrame(seed_label_rows).query("label_seed42 == 'all'").itertuples(index=False):
        checks.add(f"{cond}: {r.label_column}: cells keeping the seed-42 label in all 10 other seeds",
                   f"{r.n_stable_all_10_seeds}/{r.n_cells}", None)

    def tag(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame.insert(0, "condition", cond)
        return frame

    return {name: tag(frame) for name, frame in (
        ("stability_seeds", seeds), ("stability_seed_per_label", pd.DataFrame(seed_label_rows)),
        ("stability_noise", noise), ("stability_noise_summary", noise_summary), ("stability_score_bins", bins),
        ("stability_marker_variant_definitions", pd.DataFrame(defs)), ("stability_marker_variants", variants_tab),
        ("stability_marker_variant_crosstabs", pd.DataFrame(var_cross)),
        ("stability_loo_per_label", pd.DataFrame(loo_rows)), ("stability_per_cell", per_cell),
        ("crosstab_label_pairs", pd.DataFrame(pair_rows)), ("crosstab_three_way", three),
        ("crosstab_agreement", pd.DataFrame(agree_rows)))} | {"_loo_markers": pd.DataFrame({"marker": scored})}


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_labels step 4: stability of the label columns per condition.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "stability", args)
    t_start = time.time()
    checks = t5.Checks("dct_combined")
    step1 = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str, "leiden3_cluster": str})
    deg = lib.read_table(run, "deg_labels_per_cell", dtype={"cell_id": str, "deg_cluster": str})
    seeds = lib.read_table(run, "scan_seed_clusters_per_cell", dtype=str)
    out = {cond: one_condition(run, cond, checks, step1, deg, seeds) for cond in lib.CONDITIONS}
    if not checks.ok:
        lib.stop(run, checks, "stability_checks")
    run.table(checks.frame(), "stability_checks")
    for name in out["Ctrl"]:
        if not name.startswith("_"):
            run.table(pd.concat([out[c][name] for c in lib.CONDITIONS], ignore_index=True), name)
    run.finish({"seeds": list(t18.SEEDS), "noise_sd": t18.NOISE_SD, "noise_repeats": t18.NOISE_REPEATS,
                "noise_rng": f"numpy default_rng({lib.SEED}) per condition; N(0, sd) added in float64, cast to float32",
                "deg_choice": {"resolution": lib.RESOLUTION, "rule": lib.RULE},
                "loo_markers": {c: out[c]["_loo_markers"]["marker"].tolist() for c in lib.CONDITIONS},
                "variants": {k: v[0] for k, v in t18.MARKER_VARIANTS.items()},
                "scanpy": sc.__version__, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
