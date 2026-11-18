#!/usr/bin/env python
"""dct_prolif step 5: route B, depth-free clusters annotated with the frozen dct_labels DE rule (check), per condition and
run; route A vs route B; ProLIF kNN purity in the route-B graph.

Analysis plan
(frozen), "Route B" and the step-4 reports that need route B. Runs: 'frozen' (floor 1,000) and 'nofloor' (approved
change, authors 3 Nov 2026). Cells above the floor of each run; each condition alone.

Route B (plan): features = proteins detected in >= 50% of the cells above the floor plus every Tirosh protein in the
matrix (Mki67 and Top2a included); ln(1 + intensity) with missing = 0 (PCA needs complete values); ln(n_detected)
regressed per protein (OLS residuals); scaled (sc.pp.scale formula, max_value 10); 20 PCs (scanpy arpack); plate
corrected with harmonypy 0.0.10 (seed 42, package defaults); neighbours k = 10 on the Harmony coordinates; Leiden 1.0
(igraph, seed 42). Annotation: the frozen dct_labels DE rule (cluster_de rule with the kidney_annotation rank-free rescue; Slc12a3 and Umod
shared; minimum cluster size 5), on ln(1 + intensity) of all retained proteins of the cells above the floor (as dct_labels;
depth is not regressed out of the DE). The ProLIF row is Mki67, Top2a and P (plan default 5: it replaces the workbook
ProLIF row; Slc12a3 stays shared through DCT1 and DCT2). Resolutions 0.5 and 3.0 and seeds 0-9 are information only.

Inputs (read-only): results/h5ad/dct_intake_DCT_{ctrl,zerok}.h5ad; data/DCT/DCT_markers_V1.xlsx; dct_prolif_route_a_per_cell.csv.
Outputs (results/tables/dct_prolif/): dct_prolif_route_b_{per_cell,clusters,evidence,rankfree,de_top25,
features,info,embedding,embedding_checks,checks}.csv, dct_prolif_route_a_vs_b.csv, dct_prolif_route_a_vs_b_crosstabs.csv,
dct_prolif_knn_purity_route_b_graph.csv.

Command (project root):  python scripts/dct_prolif_route_b.py
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
import logging
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import labels_core_lib as t18
import dct_prolif_lib as lib
from kidney_annotation_rankfree_sensitivity import rank_free
from labels_core_resolution_scan import rule_tables

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "dct_prolif_route_b.py"


def embedding(c: lib.Condition, above: np.ndarray, features: np.ndarray) -> tuple[ad.AnnData, dict]:
    import harmonypy

    ln = np.log1p(c.intensity[above][:, features])
    lnids = c.ln_ids[above]
    design = np.column_stack([np.ones(len(lnids)), lnids])
    beta, *_ = np.linalg.lstsq(design, ln, rcond=None)
    resid = ln - design @ beta
    Z = cc.scale_like_scanpy(resid, lib.B_MAX_VALUE)
    g = ad.AnnData(X=Z.astype(np.float64), obs=pd.DataFrame(index=c.cell_ids[above]))
    sc.tl.pca(g, n_comps=lib.B_N_PCS, svd_solver="arpack")
    pcs = np.asarray(g.obsm["X_pca"])[:, :lib.B_N_PCS]
    logging.getLogger("harmonypy").setLevel(logging.WARNING)
    meta = pd.DataFrame({"plate": pd.Categorical(c.plate[above])})
    ho = harmonypy.run_harmony(np.ascontiguousarray(pcs, dtype=np.float64), meta, ["plate"], random_state=lib.SEED,
                               verbose=False)
    obj = list(ho.objective_harmony)
    n_iter = len(obj) - 1
    converged = n_iter < ho.max_iter_harmony or (len(obj) >= 2 and (obj[-2] - obj[-1]) / abs(obj[-2]) < ho.epsilon_harmony)
    g.obsm["X_harmony"] = np.asarray(ho.Z_corr, dtype=np.float64).T
    sc.pp.neighbors(g, n_neighbors=lib.B_K, use_rep="X_harmony")
    rel_last = (obj[-2] - obj[-1]) / abs(obj[-2]) if len(obj) >= 2 else np.nan
    info = {"harmony_iterations": n_iter, "harmony_converged": bool(converged), "harmony_clusters": int(ho.K),
            "harmony_max_iter": int(ho.max_iter_harmony), "harmony_epsilon": float(ho.epsilon_harmony),
            "harmony_rel_change_last": float(rel_last),
            "harmony_theta": float(np.ravel(ho.theta)[0]), "pca_params": str(g.uns.get("pca", {}).get("params", {}))}
    return g, info


def annotate(g: ad.AnnData, res: float, seed: int, inp: dict, markers: t5.MarkerSets) -> dict:
    cl = t5.leiden(g, res, "t31b_tmp", seed=seed).to_numpy().astype(str)
    name = t18.clustering_name(res, seed)
    de, ev, asg = t18.annotate_clustering(inp["cache"], cl, inp["var_names"], name, markers, inp["x_z"], inp["det"])
    rf = rank_free(asg, de, t18.specific_present(markers))
    tabs = rule_tables(asg, rf)
    return {"clusters": cl, "de": de, "evidence": ev, "assignment": asg, "rankfree": rf,
            "label": t18.per_cell(cl, tabs["rankfree"], "deg_label").astype(str),
            "confidence": t18.per_cell(cl, tabs["rankfree"], "confidence").astype(str)}


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif step 5: route B depth-free clusters with the frozen DE rule.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "route_b", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "route_b_checks")
    sc.settings.verbosity = 0

    conds = {c: lib.load_condition(run, c) for c in lib.CONDITIONS}
    P, _ = lib.prolif_set(conds["Ctrl"], conds["ZeroK"])
    per_a = lib.read_table(run, "route_a_per_cell", dtype={"cell_id": str})

    cells_rows, cl_rows, ev_rows, rf_rows, top_rows, feat_rows, info_rows, emb_rows, emb_checks = ([] for _ in range(9))
    avb_rows, avb_ct, purity = [], [], []
    for run_key, spec in lib.RUNS.items():
        for cond, c in conds.items():
            t0 = time.time()
            tag = f"{run_key} / {cond}"
            fa = per_a[(per_a["run"] == run_key) & (per_a["condition"] == cond)].reset_index(drop=True)
            above = fa["above_floor"].astype(str).str.lower().eq("true").to_numpy()
            if list(fa["cell_id"]) != list(c.cell_ids) or not np.array_equal(above, c.ids >= spec.floor):
                raise ValueError(f"{tag}: route-A table differs from the object")
            det_a = c.detected[above]
            tir = set(lib.tirosh_present(c.var_names))
            frac = det_a.mean(axis=0)
            features = (frac >= lib.B_MIN_DETECTION) | c.var_names.isin(list(tir))
            feat_rows.append(pd.DataFrame({"run": run_key, "condition": cond, "protein": c.var_names[features],
                                           "fraction_detected_above_floor": frac[features],
                                           "reason": np.where(frac[features] >= lib.B_MIN_DETECTION,
                                                              np.where(c.var_names[features].isin(list(tir)),
                                                                       "highcov50 and Tirosh", "highcov50"), "Tirosh")}))
            checks.add(f"{tag}: route-B features (>= 50% detected + Tirosh)",
                       f"{int(features.sum())} ({int((frac >= lib.B_MIN_DETECTION).sum())} highcov50; "
                       f"{len(tir)} Tirosh in matrix)", None)
            g, hinfo = embedding(c, above, features)
            # Package defaults are fixed by the plan; non-convergence within the default limit is reported, not changed.
            checks.add(f"{tag}: Harmony convergence (package defaults: max {hinfo['harmony_max_iter']} iterations, "
                       f"epsilon {hinfo['harmony_epsilon']:g})",
                       f"{'converged' if hinfo['harmony_converged'] else 'stopped before convergence'} after "
                       f"{hinfo['harmony_iterations']} iterations; relative objective change at the last step "
                       f"{hinfo['harmony_rel_change_last']:.2e}", None)

            # depth and plate in the embedding (information)
            lnids = c.ln_ids[above]
            for space, mat in (("PCA (before Harmony)", np.asarray(g.obsm["X_pca"])[:, :5]),
                               ("Harmony", g.obsm["X_harmony"][:, :5])):
                for k in range(5):
                    rho = stats.spearmanr(mat[:, k], lnids).statistic
                    emb_checks.append({"run": run_key, "condition": cond, "space": space, "component": k + 1,
                                       "spearman_rho_ln_ids": float(rho)})
            plate_frame = pd.DataFrame({"plate": c.plate[above]})
            pm = t2.knn_purity({"route-B Harmony": g.obsm["X_harmony"], "route-B PCA (before Harmony)":
                                np.asarray(g.obsm["X_pca"])[:, :lib.B_N_PCS]}, plate_frame, ["plate"], k=lib.B_K,
                               n_perm=1000)
            for r in pm[pm["label"] == "all_cells"].itertuples():
                emb_checks.append({"run": run_key, "condition": cond, "space": r.object, "component": "same-plate kNN",
                                   "same_plate_fraction": r.observed_same_label_fraction, "null_mean": r.null_mean,
                                   "p_empirical_ge": r.p_empirical_ge})
                checks.add(f"{tag}: same-plate neighbour fraction, {r.object}",
                           f"{r.observed_same_label_fraction:.3f} (null {r.null_mean:.3f})", None)
            emb = pd.DataFrame(g.obsm["X_harmony"], columns=[f"harmony_{k + 1}" for k in range(lib.B_N_PCS)])
            emb.insert(0, "cell_id", c.cell_ids[above])
            emb.insert(0, "condition", cond)
            emb.insert(0, "run", run_key)
            emb_rows.append(emb)

            # DE rule
            int_a = c.intensity[above]
            ln1p = np.log1p(int_a)
            inp = {"cache": t5.rank_cache(ln1p, det_a), "var_names": c.var_names,
                   "x_z": pd.DataFrame(cc.scale_like_scanpy(ln1p), index=c.cell_ids[above], columns=c.var_names),
                   "det": pd.DataFrame(det_a, index=c.cell_ids[above], columns=c.var_names)}
            _, wb = lib.identity_candidates(run, c.var_names)
            markers = lib.route_b_markers(wb, P, c.var_names)
            checks.add(f"{tag}: route-B ProLIF row (Mki67, Top2a, P) present in the matrix",
                       f"{len(markers.present[lib.PROLIF])} of {len(markers.lists[lib.PROLIF])}; Slc12a3 shared "
                       f"{markers.is_shared('Slc12a3')}; Umod shared {markers.is_shared('Umod')}",
                       len(markers.present[lib.PROLIF]) == 38 and markers.is_shared("Slc12a3") and markers.is_shared("Umod"))
            main_res = annotate(g, lib.B_RESOLUTION, lib.SEED, inp, markers)
            lab, conf, cl = main_res["label"], main_res["confidence"], main_res["clusters"]
            checks.add(f"{tag}: route-B labels (Leiden 1.0, seed 42)", lib.counts_text(lab), None)

            n = len(c.cell_ids)
            full = lambda v, fill="": np.where(above, pd.Series(index=np.flatnonzero(above), data=v).reindex(range(n)).to_numpy(), fill)
            cells_rows.append(pd.DataFrame({"run": run_key, "condition": cond, "cell_id": c.cell_ids,
                                            "cell_key": fa["cell_key"].to_numpy(), "above_floor": above,
                                            "cluster": full(cl, lib.BELOW), "label": full(lab, lib.BELOW),
                                            "confidence": full(conf, lib.BELOW)}))
            # cluster table
            la = fa.loc[above, "label"].to_numpy()
            held = c.detected[above][:, c.idx(lib.HELD_OUT)].any(axis=1)
            asg = main_res["assignment"].set_index("cluster")
            rfi = main_res["rankfree"].set_index("cluster")
            de = main_res["de"]
            for k in t5.sort_clusters(cl):
                m = cl == k
                sub = de[de["cluster"] == k]
                cl_rows.append({"run": run_key, "condition": cond, "cluster": k, "n_cells": int(m.sum()),
                                "median_protein_ids": float(np.median(c.ids[above][m])),
                                "fraction_P1": float((c.plate[above][m] == "P1").mean()),
                                "label": lab[m][0], "confidence": conf[m][0],
                                "top25_label": asg.loc[k, "deg_label"] if k in asg.index else "",
                                "rescue_markers": rfi.loc[k, "rescue_markers"] if k in rfi.index else "",
                                "n_sig_up": int((sub["up"] & (sub["q_bh"] < t5.Q_DE)).sum()) if len(sub) else 0,
                                "top10_de": ";".join(sub[sub["in_top"]].sort_values("rank_up")["protein"].head(10)),
                                "n_route_a_prolif": int((la[m] == lib.PROLIF).sum()),
                                "route_a_label_counts": lib.counts_text(la[m]),
                                "fraction_mki67_or_top2a_detected": float(held[m].mean())})
            for name, frame, rows in (("evidence", main_res["evidence"], ev_rows), ("rankfree", main_res["rankfree"], rf_rows),
                                      ("de_top25", de[de["in_top"]], top_rows)):
                frame = frame.copy()
                frame.insert(0, "condition", cond)
                frame.insert(0, "run", run_key)
                rows.append(frame)

            # information: resolutions and seeds
            for res, seeds in ((lib.B_RESOLUTION, lib.B_INFO_SEEDS), *((r, (lib.SEED,)) for r in lib.B_INFO_RESOLUTIONS)):
                for s in seeds:
                    out = annotate(g, res, s, inp, markers)
                    info_rows.append({"run": run_key, "condition": cond, "resolution": res, "seed": s,
                                      "n_clusters": int(len(set(out["clusters"]))), "label_counts": lib.counts_text(out["label"]),
                                      "prolif_present": bool((out["label"] == lib.PROLIF).any()),
                                      "n_prolif": int((out["label"] == lib.PROLIF).sum()),
                                      "ari_labels_vs_seed42_r1": float(adjusted_rand_score(lab, out["label"])),
                                      "fraction_equal_vs_seed42_r1": float((lab == out["label"]).mean()),
                                      "prolif_jaccard_vs_route_a": lib.jaccard(out["label"] == lib.PROLIF, la == lib.PROLIF)})

            # route A vs route B (cells above the floor)
            ag = lib.agreement(la, lab)
            pro_b = lab == lib.PROLIF
            avb_rows.append({"run": run_key, "condition": cond, "n_cells_above_floor": int(above.sum()),
                             "route_a_counts": lib.counts_text(la), "route_b_counts": lib.counts_text(lab),
                             "fraction_equal": ag["fraction_equal"], "ari": ag["ari"],
                             "n_prolif_route_a": int((la == lib.PROLIF).sum()), "n_prolif_route_b": int(pro_b.sum()),
                             "n_prolif_both": int(((la == lib.PROLIF) & pro_b).sum()),
                             "prolif_jaccard": lib.jaccard(la == lib.PROLIF, pro_b),
                             "route_b_has_prolif_cluster": bool(pro_b.any()),
                             "route_b_prolif_clusters": ";".join(sorted(set(cl[pro_b]), key=lambda x: int(x)
                                                                        if x.isdigit() else x)) if pro_b.any() else "",
                             "route_a_prolif_in_route_b_labels": lib.counts_text(lab[la == lib.PROLIF], lib.LABEL_ORDER
                                                                                 + tuple(cc.DCT_LABEL_ORDER))})
            avb_ct.extend(lib.crosstab_long(la, lab, "route_a_label", "route_b_label", run=run_key, condition=cond))
            checks.add(f"{tag}: route A vs B: ARI; ProLIF Jaccard; route-B ProLIF cluster",
                       f"{ag['ari']:.3f}; {avb_rows[-1]['prolif_jaccard']:.3f}; {avb_rows[-1]['route_b_has_prolif_cluster']}",
                       None)

            kp = t2.knn_purity({"route-B graph (Harmony, 20 dims; cells above the floor)": g.obsm["X_harmony"]},
                               pd.DataFrame({"label": la}), ["label"], k=lib.B_K, n_perm=1000)
            kp.insert(0, "condition", cond)
            kp.insert(0, "run", run_key)
            purity.append(kp)
            r = kp[kp["label"] == lib.PROLIF]
            if len(r):
                r = r.iloc[0]
                checks.add(f"{tag}: route-A ProLIF kNN purity in the route-B graph",
                           f"{r.observed_same_label_fraction:.3f} (null {r.null_mean:.3f}; p {r.p_empirical_ge:.4f})", None)
            emb_checks.append({"run": run_key, "condition": cond, "space": "Harmony", "component": "convergence",
                               **{k: v for k, v in hinfo.items() if k != "pca_params"}})
            print(f"[{tag}] route B done in {time.time() - t0:.1f} s", flush=True)

    run.table(pd.concat(cells_rows, ignore_index=True), "route_b_per_cell")
    run.table(pd.DataFrame(cl_rows), "route_b_clusters")
    run.table(pd.concat(ev_rows, ignore_index=True), "route_b_evidence")
    run.table(pd.concat(rf_rows, ignore_index=True), "route_b_rankfree")
    run.table(pd.concat(top_rows, ignore_index=True), "route_b_de_top25")
    run.table(pd.concat(feat_rows, ignore_index=True), "route_b_features")
    run.table(pd.DataFrame(info_rows), "route_b_info")
    run.table(pd.concat(emb_rows, ignore_index=True), "route_b_embedding")
    run.table(pd.DataFrame(emb_checks), "route_b_embedding_checks")
    run.table(pd.DataFrame(avb_rows), "route_a_vs_b")
    run.table(pd.DataFrame(avb_ct), "route_a_vs_b_crosstabs")
    run.table(pd.concat(purity, ignore_index=True), "knn_purity_route_b_graph")
    run.table(checks.frame(), "route_b_checks")
    run.finish({"runs": {k: vars(v) for k, v in lib.RUNS.items()}, "approved_change": lib.APPROVED_CHANGE,
                "features": "detected in >= 50% of cells above the floor, plus every Tirosh protein in the matrix",
                "regression": "OLS residuals of ln(1 + intensity) on ln(n_detected), per protein",
                "scale_max_value": lib.B_MAX_VALUE, "n_pcs": lib.B_N_PCS, "pca": "scanpy arpack",
                "harmony": "harmonypy 0.0.10 run_harmony, batch = plate, random_state 42, package defaults",
                "k": lib.B_K, "resolution": lib.B_RESOLUTION, "info_resolutions": list(lib.B_INFO_RESOLUTIONS),
                "info_seeds": list(lib.B_INFO_SEEDS), "prolif_row": "Mki67, Top2a and P (replaces the workbook row)",
                "de": "dct_labels rule on ln(1 + intensity) of all retained proteins of the cells above the floor",
                "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_route_b_checks.csv.")


if __name__ == "__main__":
    main()
