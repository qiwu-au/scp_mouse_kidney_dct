#!/usr/bin/env python
"""rna_protein_integration step 3: integration diagnostics over draws and sensitivity configurations.

Analysis plan,
step 3 (design approved by the authors, 15 Oct 2026). Every configuration (rna_protein_integration_lib.CONFIGS) is built for the
5 RNA draws (seeds 42-46); the 'counterpart_reference' configuration once per label set.

Metrics (per configuration, draw, embedding; labels: kidney_t15_label primary, kidney_pre_label)
-----------------------------------------------------------------------------------------------
- Global modality mixing: iLISI (harmonypy compute_lisi, perplexity 30) on all cells, reported raw and scib-style
  ((median - 1) / (2 - 1)) next to the value for random placement given the modality shares; RNA-neighbour fraction
  of the SCP cells (k 15) divided by its value under random placement. Embeddings A and B only (C places SCP cells
  among RNA cells by construction).
- Modality mixing within each cell type: compartment = SCP cells of a label + RNA cells of its expected atlas
  classes (approved kidney_rna mapping); kNN (k 15) among the compartment's cells only: RNA-neighbour fraction of the SCP
  cells divided by its value under random mixing in the compartment (primary); iLISI (perplexity 5) scib-style next
  to the compartment's perfect-mixing value. Compartments with < 10 RNA cells are skipped.
- Cell-type separation: common label space (RNA classes renamed to their SCP label, others 'atlas: <class>'); SCP
  labels without counterpart (JG, Unknown) left out. cLISI (perplexity 30; scaled (L - median) / (L - 1), 1 = pure)
  for A and B; silhouette (euclidean) jointly, RNA only and SCP only for A, B and C.
- kNN label transfer RNA -> SCP: majority of the k nearest RNA cells (k 15; ties -> smallest mean distance) in A,
  B, C and directly in shared-feature space (cosine); atlas Celltype and Celltype_finest (reference cells with a
  known finest label). Accuracy strict (prediction among the label's expected classes) and incl. accepted mixed
  classes; macro-F1 over labels with a counterpart (predictions mapped back to SCP labels, others 'other');
  chance accuracy from the reference class frequencies. k 5 and 30 for the primary configuration.
Primary draw (seed 42): per-cell predictions, confusion counts, per-cell LISI; the primary embeddings are rebuilt
and compared with results/h5ad/rna_protein_integration_joint_embedding.h5ad (reproducibility check).

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation labels; kidney_rna shared features, approved mapping and
  mapping summary; rna_protein_integration_rna_subsamples.h5ad, rna_protein_integration_rna_subsample.csv, rna_protein_integration_features.csv,
  rna_protein_integration_joint_embedding.h5ad.
Outputs: results/tables/rna_protein_integration/rna_protein_integration_runs.csv, rna_protein_integration_mixing_global.csv,
  rna_protein_integration_mixing_per_type.csv, rna_protein_integration_separation.csv, rna_protein_integration_transfer_summary.csv,
  rna_protein_integration_transfer_per_label.csv, rna_protein_integration_transfer_per_cell.csv, rna_protein_integration_transfer_confusion.csv,
  rna_protein_integration_lisi_per_cell.csv, rna_protein_integration_metrics_summary.csv, rna_protein_integration_diagnostics_checks.csv
Command (project root):  python scripts/rna_protein_integration_diagnostics.py
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

import scp_common as cc
import cluster_de_lib as t5
import rna_protein_integration_lib as T

REPRO_TOL = 1e-4
MIX_EMBEDDINGS = ("pca", "harmony")
TRANSFER_SPACES = ("pca", "harmony", "scvi", T.SHARED_KNN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rna_protein_integration step 3: integration diagnostics.")
    parser.add_argument("--configs", nargs="*", default=list(T.CONFIGS), choices=list(T.CONFIGS))
    parser.add_argument("--draws", nargs="*", type=int, default=list(T.SUBSAMPLE_SEEDS), choices=list(T.SUBSAMPLE_SEEDS))
    parser.add_argument("--no-write", action="store_true", help="smoke test: compute and print, write nothing")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "diagnostics_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see rna_protein_integration_diagnostics_checks.csv.")


def rna_neighbour_fraction(e: np.ndarray, is_scp: np.ndarray, k: int = T.K_MIX) -> dict:
    """Fraction of RNA cells among the k nearest neighbours of each SCP cell (all cells), and its value under
    random placement (RNA cells / (all cells - 1))."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(e)
    _, nb = nn.kneighbors(e[is_scp])
    frac = (~is_scp[nb[:, 1:]]).mean(axis=1)
    expected = (~is_scp).sum() / (len(is_scp) - 1)
    return {"rna_neighbour_fraction_scp": float(frac.mean()), "rna_neighbour_fraction_expected": float(expected),
            "scaled_rna_neighbour_fraction_scp": float(frac.mean() / expected)}


def transfer(j: T.Joint, space: str, level: str, k: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Predicted atlas class and vote fraction per SCP cell; reference classes (for the chance level)."""
    col = T.LEVELS[level]
    ref_lab = j.obs.loc[~j.is_scp, col].astype(str).to_numpy()
    ref_ok = ref_lab != T.ATLAS_UNKNOWN
    if space == T.SHARED_KNN:
        ref, qry, metric = j.z[~j.is_scp][ref_ok], j.z[j.is_scp], "cosine"
    else:
        e = j.emb[space]
        ref, qry, metric = e[~j.is_scp][ref_ok], e[j.is_scp], "euclidean"
    pred, frac = T.knn_vote(ref, ref_lab[ref_ok], qry, k, metric)
    idx = j.obs.loc[j.is_scp, "cell"].to_numpy()
    return pd.Series(pred, index=idx), pd.Series(frac, index=idx), pd.Series(ref_lab[ref_ok])


def main() -> None:
    args = parse_args()
    t0 = time.time()
    run = cc.Run(T.TASK, "diagnostics", args)
    checks = t5.Checks(T.DATASET)
    shared = T.read_shared(run, checks)
    scp = T.load_scp(run, checks, shared)
    mapping = T.read_mapping(run, checks)
    pool = T.load_pool(run)
    feat = pd.read_csv(run.use_input(cc.out_path(T.TASK, "tables", "features", ".csv"), "rna_protein_integration features table"))
    scp10 = feat["in_features_scp10"].astype(bool).to_numpy()
    t07sum = pd.read_csv(run.use_input(T.KIDNEY_RNA_MAPPING_SUMMARY, "kidney_rna centroid mapping summary (step 4)"))
    if not checks.ok:
        stop(run, checks)
    sets = {(col, lv): T.expected_sets(mapping, col, lv) for col in T.LABEL_COLUMNS for lv in T.LEVELS}
    for (col, lv), s in sets.items():
        inv = T.inverse_map(s)          # raises if an atlas class is expected for two labels
        checks.add(f"{col} / {lv}: labels with a counterpart; atlas classes mapped back (unique)",
                   f"{len(T.counterpart_labels(s))}; {len(inv)}", True)
    labels_scp = scp.labels.copy()
    labels_scp.index = labels_scp.index.astype(str)

    runs, mix_g, mix_t, sep, tr_sum, tr_lab = [], [], [], [], [], []
    per_cell_pred, confusion, lisi_cells = None, [], None
    for config in args.configs:
        cfg = T.CONFIGS[config]
        refs = ["all"] if cfg["reference"] == "all" else list(T.LABEL_COLUMNS)
        fmask = scp10 if cfg["features"] == "scp10" else None
        for draw in args.draws:
            for ref in refs:
                t1 = time.time()
                j = T.build_joint(scp, pool, config, draw, mapping, reference=ref, feature_mask=fmask)
                key = {"config": config, "draw": draw, "reference": ref}
                label_sets = list(T.LABEL_COLUMNS) if ref == "all" else [ref]
                primary = config == "primary" and draw == T.PRIMARY_DRAW
                if primary:
                    import anndata as ad

                    stored = ad.read_h5ad(run.use_input(T.JOINT_H5AD, "rna_protein_integration joint embedding (rna_protein_integration_integrate.py)"))
                    for k_emb, k_st in (("pca", "X_pca_joint"), ("harmony", "X_harmony"), ("scvi", "X_scvi_projected")):
                        d = float(np.max(np.abs(j.emb[k_emb] - np.asarray(stored.obsm[k_st], dtype=np.float64))))
                        checks.add(f"primary draw 42 rebuilt: {T.EMBEDDINGS[k_emb]} equals the stored joint "
                                   "embedding (max abs difference; float32 storage)", f"{d:.1e}", d < REPRO_TOL,
                                   f"< {REPRO_TOL:g}")
                    if not checks.ok:
                        stop(run, checks)
                common = {col: T.common_labels(j.obs.loc[j.is_scp, col], j.obs.loc[~j.is_scp, T.LEVEL],
                                               T.inverse_map(sets[(col, "celltype")])) for col in label_sets}
                keep = {col: ~j.is_scp | j.obs[col].isin(T.counterpart_labels(sets[(col, "celltype")])).to_numpy()
                        for col in label_sets}
                lisi_keep = {}
                # global mixing and separation ---------------------------------------------------------------------
                for emb in MIX_EMBEDDINGS:
                    e = j.emb[emb]
                    meta = pd.DataFrame({"modality": j.obs["modality"].to_numpy()})
                    il = T.lisi(e, meta, ["modality"], T.LISI_PERPLEXITY)["modality"].to_numpy()
                    p = j.is_scp.mean()
                    imax = T.max_ilisi(p)
                    mix_g.append({**key, "embedding": emb, "n_scp": int(j.is_scp.sum()), "n_rna": int((~j.is_scp).sum()),
                                  "ilisi_perfect_mixing": imax, "median_ilisi_scp": float(np.median(il[j.is_scp])),
                                  "median_ilisi_rna": float(np.median(il[~j.is_scp])),
                                  "median_ilisi_all": float(np.median(il)),
                                  "ilisi_scib_scp": float(np.median(il[j.is_scp]) - 1),
                                  "ilisi_scib_all": float(np.median(il) - 1), "ilisi_perfect_scib": float(imax - 1),
                                  **rna_neighbour_fraction(e, j.is_scp)})
                    if primary:
                        lisi_keep[f"ilisi_{emb}"] = il
                    for col in label_sets:
                        m = keep[col]
                        cl = T.lisi(e[m], pd.DataFrame({"lab": common[col][m]}), ["lab"], T.LISI_PERPLEXITY)["lab"].to_numpy()
                        n_lab = len(set(common[col][m]))
                        s = T.separation(e, common[col], j.is_scp, m)
                        sep.append({**key, "embedding": emb, "label_column": col, "n_cells": int(m.sum()),
                                    "n_labels": n_lab, "median_clisi_all": float(np.median(cl)),
                                    "median_clisi_scp": float(np.median(cl[j.is_scp[m]])),
                                    "median_clisi_rna": float(np.median(cl[~j.is_scp[m]])),
                                    "scaled_clisi_all": float((n_lab - np.median(cl)) / (n_lab - 1)),
                                    "scaled_clisi_scp": float((n_lab - np.median(cl[j.is_scp[m]])) / (n_lab - 1)), **s})
                        if primary:
                            full = np.full(len(j.obs), np.nan)
                            full[m] = cl
                            lisi_keep[f"clisi_{emb}_{T.LABEL_SHORT[col]}"] = full
                        # per compartment
                        for lab, (exp, _) in sets[(col, "celltype")].items():
                            if not exp:
                                continue
                            cm = ((j.is_scp & (j.obs[col] == lab).to_numpy()) |
                                  (~j.is_scp & j.obs[T.LEVEL].isin(sorted(exp)).to_numpy()))
                            res = T.compartment_mixing(e, j.is_scp, cm)
                            rec = {**key, "embedding": emb, "label_column": col, "compartment": lab,
                                   "atlas_classes": ";".join(sorted(exp))}
                            if res is None:
                                rec.update({"n_scp": int((j.is_scp & cm).sum()), "n_rna": int((~j.is_scp & cm).sum()),
                                            "skipped": "fewer than 10 RNA cells or 2 SCP cells"})
                            else:
                                res.pop("_per_cell")
                                rec.update(res)
                                rec["skipped"] = ""
                            mix_t.append(rec)
                if "scvi" in j.emb:
                    for col in label_sets:
                        sep.append({**key, "embedding": "scvi", "label_column": col, "n_cells": int(keep[col].sum()),
                                    "n_labels": len(set(common[col][keep[col]])),
                                    **T.separation(j.emb["scvi"], common[col], j.is_scp, keep[col])})
                # transfer -----------------------------------------------------------------------------------------
                ks = (T.K_TRANSFER, *T.K_TRANSFER_SENS) if config == "primary" else (T.K_TRANSFER,)
                for space in TRANSFER_SPACES:
                    if space == "scvi" and "scvi" not in j.emb:
                        continue
                    for lv in T.LEVELS:
                        for k in ks:
                            pred, frac, refc = transfer(j, space, lv, k)
                            for col in label_sets:
                                lab = labels_scp.loc[pred.index, col]
                                summ, per = T.transfer_metrics(pred, lab, sets[(col, lv)], refc)
                                tr_sum.append({**key, "embedding": space, "label_column": col, "level": T.LEVELS[lv],
                                               "k": k, **summ})
                                per.insert(0, "k", k)
                                per.insert(0, "level", T.LEVELS[lv])
                                per.insert(0, "label_column", col)
                                per.insert(0, "embedding", space)
                                for kk, vv in reversed(list(key.items())):
                                    per.insert(0, kk, vv)
                                tr_lab.append(per)
                                if primary and k == T.K_TRANSFER:
                                    vc = pd.crosstab(lab, pred)
                                    for sl in vc.index:
                                        exp, mix = sets[(col, lv)].get(sl, (set(), set()))
                                        for ac in vc.columns:
                                            n = int(vc.loc[sl, ac])
                                            if n:
                                                confusion.append({"embedding": space, "level": T.LEVELS[lv],
                                                                  "label_column": col, "scp_label": sl,
                                                                  "atlas_class": ac, "n": n,
                                                                  "fraction_of_label": n / int(vc.loc[sl].sum()),
                                                                  "is_expected": ac in exp,
                                                                  "is_mixed_accepted": ac in (mix - exp)})
                            if primary and k == T.K_TRANSFER:
                                if per_cell_pred is None:
                                    per_cell_pred = labels_scp[["kidney_t15_label", "kidney_pre_label",
                                                                "n_detected_proteins"]].copy()
                                per_cell_pred[f"pred_{space}_{lv}"] = pred.reindex(per_cell_pred.index).to_numpy()
                                per_cell_pred[f"vote_{space}_{lv}"] = frac.reindex(per_cell_pred.index).to_numpy()
                if primary:
                    lisi_cells = j.obs[["modality", "cell", "kidney_t15_label", "kidney_pre_label", T.LEVEL]].copy()
                    for kname, vals in lisi_keep.items():
                        lisi_cells[kname] = vals
                runs.append({**key, **{k: v for k, v in j.info.items()}, "runtime_s": round(time.time() - t1, 1)})
                print(f"[{config} / draw {draw} / ref {ref}] {time.time() - t1:.0f} s "
                      f"(harmony {j.info['harmony_iterations']} it, converged {j.info['harmony_converged']})", flush=True)

    runs = pd.DataFrame(runs)
    mix_g, mix_t, sep = pd.DataFrame(mix_g), pd.DataFrame(mix_t), pd.DataFrame(sep)
    tr_sum, tr_lab = pd.DataFrame(tr_sum), pd.concat(tr_lab, ignore_index=True)
    confusion = pd.DataFrame(confusion)
    checks.add("Harmony runs meeting harmonypy's convergence criterion within 10 iterations",
               f"{int(runs['harmony_converged'].sum())} of {len(runs)}", None)

    # kidney_rna centroid comparison (top label among expected classes) ---------------------------------------------------
    t7 = t07sum[t07sum["threshold"] == "0.20/0.00"][["level", "label_column", "scp_label", "expected_fraction_top",
                                                      "expected_fraction_top_incl_mixed"]]
    t7 = t7.rename(columns={"expected_fraction_top": "t07_centroid_expected_top",
                            "expected_fraction_top_incl_mixed": "t07_centroid_expected_top_incl_mixed"})
    tr_lab = tr_lab.merge(t7, on=["level", "label_column", "scp_label"], how="left")

    # summaries over draws ----------------------------------------------------------------------------------------
    def over_draws(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
        g = frame.groupby(keys, dropna=False)
        out = g[metrics].agg(["median", "min", "max"])
        out.columns = [f"{m}_{s}" for m, s in out.columns]
        out["n_draws"] = g.size()
        return out.reset_index()

    summ_tr = over_draws(tr_sum, ["config", "reference", "embedding", "label_column", "level", "k"],
                         ["accuracy_strict", "accuracy_incl_mixed", "macro_f1", "chance_accuracy_strict"])
    summ_tr.insert(0, "metric_group", "transfer")
    summ_mg = over_draws(mix_g, ["config", "reference", "embedding"], ["median_ilisi_scp", "ilisi_scib_scp",
                                                                       "scaled_rna_neighbour_fraction_scp"])
    summ_mg.insert(0, "metric_group", "mixing_global")
    summ_sep = over_draws(sep, ["config", "reference", "embedding", "label_column"],
                          ["scaled_clisi_all", "silhouette_joint", "silhouette_rna", "silhouette_scp"])
    summ_sep.insert(0, "metric_group", "separation")
    ok_t = mix_t[mix_t["skipped"] == ""]
    summ_mt = over_draws(ok_t, ["config", "reference", "embedding", "label_column", "compartment"],
                         ["scaled_other_fraction_scp", "ilisi_scib_scp", "ilisi_perfect_scib"])
    summ_mt.insert(0, "metric_group", "mixing_per_type")
    metrics_summary = pd.concat([summ_tr, summ_mg, summ_sep, summ_mt], ignore_index=True)

    # headline checks (information) -----------------------------------------------------------------------------------
    for col in T.LABEL_COLUMNS:
        for emb in TRANSFER_SPACES:
            r = summ_tr[(summ_tr["config"] == "primary") & (summ_tr["embedding"] == emb) & (summ_tr["label_column"] == col)
                        & (summ_tr["level"] == "Celltype") & (summ_tr["k"] == T.K_TRANSFER)]
            if len(r):
                r = r.iloc[0]
                checks.add(f"primary, {col}, {emb}, Celltype, k 15: accuracy strict; macro-F1; chance (median over draws)",
                           f"{r.accuracy_strict_median:.3f}; {r.macro_f1_median:.3f}; {r.chance_accuracy_strict_median:.3f}",
                           None)
    mg_perfect = float(mix_g.loc[mix_g["config"] == "primary", "ilisi_perfect_mixing"].iloc[0])
    for emb in MIX_EMBEDDINGS:
        r = summ_mg[(summ_mg["config"] == "primary") & (summ_mg["embedding"] == emb)].iloc[0]
        checks.add(f"primary, {emb}: SCP cells, median iLISI (random placement {mg_perfect:.3f}); scaled RNA-neighbour "
                   "fraction (median over draws; range)",
                   f"{r.median_ilisi_scp_median:.3f} ({r.median_ilisi_scp_min:.3f}-{r.median_ilisi_scp_max:.3f}); "
                   f"{r.scaled_rna_neighbour_fraction_scp_median:.3f} ({r.scaled_rna_neighbour_fraction_scp_min:.3f}-"
                   f"{r.scaled_rna_neighbour_fraction_scp_max:.3f})", None)

    per_cell_pred = per_cell_pred.rename_axis("cell_id").reset_index()
    if args.no_write:
        print(metrics_summary.head(40).to_string(), flush=True)
        print(f"smoke test finished in {time.time() - t0:.0f} s; nothing written", flush=True)
        return
    run.table(runs, "runs")
    run.table(mix_g, "mixing_global")
    run.table(mix_t, "mixing_per_type")
    run.table(sep, "separation")
    run.table(tr_sum, "transfer_summary")
    run.table(tr_lab, "transfer_per_label")
    run.table(per_cell_pred, "transfer_per_cell")
    run.table(confusion, "transfer_confusion")
    run.table(lisi_cells, "lisi_per_cell")
    run.table(metrics_summary, "metrics_summary")
    run.table(checks.frame(), "diagnostics_checks")
    run.finish({"configs": {c: T.CONFIGS[c] for c in args.configs}, "draws": list(args.draws),
                "lisi_perplexity_global": T.LISI_PERPLEXITY, "mix_perplexity": T.MIX_PERPLEXITY, "k_mix": T.K_MIX,
                "k_transfer": T.K_TRANSFER, "k_transfer_sensitivity": list(T.K_TRANSFER_SENS),
                "min_compartment_rna": T.MIN_COMPARTMENT_RNA, "reproduction_tolerance": REPRO_TOL,
                "runtime_s": round(time.time() - t0, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
