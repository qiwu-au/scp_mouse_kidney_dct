#!/usr/bin/env python
"""The response to the reviewers: ALDO scores of the ProLIF cells against random protein sets, before and after regression
of the cell-cycle score.

. Approved by the authors
on 12 Nov 2026 ("Run the control").

Why: in the per-cell ProLIF cells (dct_prolif route A, no-floor run; control 37, zero-K 63) no ALDO score differs from DCT1
or DCT2 on the main matrix, but ALDO activity and machinery are higher on the matrix from which the cell-cycle score
was regressed out (report_12_percell_prolif_aldo_contrasts.csv). The cell-cycle score falls with the number of
proteins per cell (dct_reviewer: Spearman rho -0.64 control, -0.46 zero-K), so the regression may raise most proteins in
cells with a high score. This script tests that reading: the same contrast is computed for random protein sets of
the same size and detection rate as each ALDO list, on the same two stored matrices.

Method (each condition alone; nothing compares the conditions):
- matrices: the stored dct_reviewer objects, scaled X ("scaled_X", the main matrix) and the cell-cycle-regressed matrix
  ("cc_regressed_3a"); nothing is refitted;
- score of a protein set = mean protein z-score (ddof 0) over the set, as reviewer_core_lib.aldo_scores; ALDO_signed =
  score of the UP set minus score of the DOWN set;
- random sets: every protein of a list is replaced by a protein drawn from the same detection-rate bin (20 quantile
  bins of the fraction of cells with intensity > 0 in the condition), without replacement within a set, from the
  proteins that are in no ALDO list; 1,000 sets per list, seed 42; the same sets are used for both matrices;
- contrasts: ProLIF minus reference (difference of medians, Cohen's d), for the per-cell call against the cluster
  annotation (DCT1, DCT2, DCT1 + DCT2, all other cells) and for the marker-score label of the clusters against its
  own DCT1 and DCT2 (as dct_reviewer);
- reported per contrast: the observed value, the median and central 95% of the random sets, and the fractions of
  random sets at or above and at or below the observed value (empirical p, floor 1/1,001);
- also reported, without random sets: the same contrasts for the mean z-score over all proteins of the matrix.

Tables (results/tables/report/): report_aldo_random_sets.csv (contrasts),
  report_12_aldo_random_sets_proteins.csv (list proteins, detection rate, bin, pool size),
  report_12_aldo_random_sets_checks.csv; manifest report_step_12_aldo_random_sets_manifest.json

Inputs (read-only): results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_{task3_scored,cc_regressed}.h5ad;
  results/h5ad/dct_labels_DCT_{ctrl,zerok}_labelled.h5ad (layer intensity, for the detection rate);
  results/tables/dct_reviewer/dct_reviewer_aldo_scores_per_cell.csv;
  results/tables/dct_prolif/dct_prolif_dct_labels.csv;
  results/tables/report/report_12_percell_prolif_aldo_contrasts.csv

Command (project root):  python scripts/report_aldo_random_sets.py --overwrite
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

import anndata as ad
import numpy as np
import pandas as pd

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import dct_reviewer_lib as lib

TASK = "report"
STEP = "12_aldo_random_sets"
DCT_REVIEWER, DCT_PROLIF = "dct_reviewer", "dct_prolif"
N_SETS, N_BINS = 1000, 20
SOURCES = {"scaled_X": "task3_scored", "cc_regressed_3a": "cc_regressed"}
PROLIF = "ProLIF"
# (score, variant) -> (positive list, negative list); Slc12a3 is part of the UP list of the base variant.
UP_BASE = [g for g in t2.ALDO_UP_WITH_SLC12A3]
UP_NO_NCC = [g for g in UP_BASE if g != "Slc12a3"]
SCORE_DEFS = {
    ("ALDO_activity", "base"): (UP_BASE, []),
    ("ALDO_activity", "no_Slc12a3"): (UP_NO_NCC, []),
    ("ALDO_signed", "base"): (UP_BASE, list(t2.ALDO_SIGNATURE_DOWN)),
    ("ALDO_signed", "no_Slc12a3"): (UP_NO_NCC, list(t2.ALDO_SIGNATURE_DOWN)),
    ("ALDO_machinery", "base"): (list(t2.FIG3_ALDO_MACHINERY), []),
}
EXPECTED_PRESENT = {"UP_base": 25, "UP_no_Slc12a3": 24, "DOWN": 3, "machinery": 12}       # dct_reviewer README section 3
COMPARISONS = (("ProLIF_vs_DCT1", ("DCT1",)), ("ProLIF_vs_DCT2", ("DCT2",)),
               ("ProLIF_vs_DCT1_DCT2", ("DCT1", "DCT2")), ("ProLIF_vs_all_other_cells", None))


def z_matrix(adata) -> np.ndarray:
    """Per-protein z-score (ddof 0) of the stored matrix; a protein without variance scores 0 (as aldo_scores)."""
    x = np.asarray(cc.dense(adata.X), dtype=np.float64)
    sd = x.std(axis=0, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x - x.mean(axis=0)) / np.where(sd > 0, sd, np.nan)
    return np.nan_to_num(z, nan=0.0)


def contrast(scores: np.ndarray, group: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Difference of medians and Cohen's d (group - reference) for every row of scores (sets x cells)."""
    g, r = scores[:, group], scores[:, ref]
    diff = np.median(g, axis=1) - np.median(r, axis=1)
    pooled = np.sqrt(((g.shape[1] - 1) * g.var(axis=1, ddof=1) + (r.shape[1] - 1) * r.var(axis=1, ddof=1))
                     / (g.shape[1] + r.shape[1] - 2))
    return diff, (g.mean(axis=1) - r.mean(axis=1)) / pooled


def null_summary(observed: float, null: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_observed": float(observed), f"{prefix}_random_median": float(np.median(null)),
            f"{prefix}_random_q025": float(np.quantile(null, 0.025)),
            f"{prefix}_random_q975": float(np.quantile(null, 0.975)),
            f"{prefix}_inside_random_95": bool(np.quantile(null, 0.025) <= observed <= np.quantile(null, 0.975)),
            f"{prefix}_p_random_at_or_above": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
            f"{prefix}_p_random_at_or_below": float((1 + np.sum(null <= observed)) / (len(null) + 1))}


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: ALDO scores against random protein sets.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    rng = np.random.default_rng(cc.RANDOM_SEED)

    def table(task: str, name: str) -> pd.DataFrame:
        path = cc.out_path(task, "tables", name, ".csv")
        return pd.read_csv(run.use_input(path, f"{task}: {name}"), keep_default_na=False, na_values=[""])

    t31 = table(DCT_PROLIF, "dct_labels").set_index("cell_id")
    stored_scores = table(DCT_REVIEWER, "aldo_scores_per_cell").set_index("cell_id")
    tabulated = table(TASK, "12_percell_prolif_aldo_contrasts")

    rows, protein_rows = [], []
    worst_score, worst_contrast = 0.0, 0.0
    for cond in lib.CONDITIONS:
        key = lib.KEY[cond]
        objects = {source: lib.read_h5ad(run, f"DCT_{key}_{name}") for source, name in SOURCES.items()}
        labelled = ad.read_h5ad(run.use_input(lib.DCT_LABELS_H5AD[cond], "dct_labels labelled object (layer intensity)"))
        base = objects["scaled_X"]
        cells = base.obs_names.astype(str)
        proteins = base.var_names.astype(str)
        same = all(list(o.obs_names.astype(str)) == list(cells) and list(o.var_names.astype(str)) == list(proteins)
                   for o in (*objects.values(), labelled))
        checks.add(f"{cond}: the three stored objects hold the same cells and proteins in the same order",
                   f"{len(cells)} cells, {len(proteins)} proteins", same)
        if not same:
            raise SystemExit(f"{cond}: stored objects differ in cells or proteins.")
        detected = np.asarray(cc.dense(labelled.layers["intensity"])) > 0
        rate = pd.Series(detected.mean(axis=0), index=proteins)
        index_of = {p: i for i, p in enumerate(proteins)}

        # --- lists, detection-rate bins and pools ---------------------------------------------------------------
        lists = {"UP_base": [g for g in UP_BASE if g in index_of],
                 "UP_no_Slc12a3": [g for g in UP_NO_NCC if g in index_of],
                 "DOWN": [g for g in t2.ALDO_SIGNATURE_DOWN if g in index_of],
                 "machinery": [g for g in t2.FIG3_ALDO_MACHINERY if g in index_of]}
        checks.add(f"{cond}: list proteins present = the dct_reviewer coverage table",
                   "; ".join(f"{k} {len(v)}" for k, v in lists.items()),
                   {k: len(v) for k, v in lists.items()} == EXPECTED_PRESENT, str(EXPECTED_PRESENT))
        in_any_list = set(UP_BASE) | set(t2.ALDO_SIGNATURE_DOWN) | set(t2.FIG3_ALDO_MACHINERY)
        bins = pd.qcut(rate.rank(method="first"), N_BINS, labels=False)
        pool = {b: np.array([index_of[p] for p in rate.index[(bins == b).to_numpy()] if p not in in_any_list])
                for b in range(N_BINS)}
        for name, genes in lists.items():
            for g in genes:
                protein_rows.append({"condition": cond, "list": name, "protein": g,
                                     "fraction_cells_detected": float(rate[g]), "detection_bin": int(bins[g]) + 1,
                                     "bin_fraction_min": float(rate[(bins == bins[g]).to_numpy()].min()),
                                     "bin_fraction_max": float(rate[(bins == bins[g]).to_numpy()].max()),
                                     "proteins_in_bin_available": int(len(pool[int(bins[g])]))})

        def random_sets(genes: list[str]) -> np.ndarray:
            """N_SETS x len(genes) protein indices: one protein of the same detection-rate bin per list protein."""
            need = pd.Series([int(bins[g]) for g in genes]).value_counts()
            out = np.empty((N_SETS, len(genes)), dtype=np.int64)
            for s in range(N_SETS):
                out[s] = np.concatenate([rng.choice(pool[b], size=int(n), replace=False) for b, n in need.items()])
            return out

        sets = {name: random_sets(genes) for name, genes in lists.items()}
        enough = all(len(pool[int(bins[g])]) >= 25 for genes in lists.values() for g in genes)
        checks.add(f"{cond}: every detection-rate bin of a list protein holds at least 25 other proteins",
                   f"smallest pool {min(len(pool[int(bins[g])]) for genes in lists.values() for g in genes)}", enough)

        # --- groups ---------------------------------------------------------------------------------------------
        lab = t31.loc[cells]
        second = stored_scores.loc[cells, "second_label"].astype(str).to_numpy()
        primary = lab["dct_label"].astype(str).to_numpy()
        per_cell = lab["t31_label"].astype(str).eq(PROLIF).to_numpy()
        definitions = {"per_cell_call": (per_cell, primary), "marker_score_clusters": (second == PROLIF, second)}

        for source, adata in objects.items():
            z = z_matrix(adata)
            all_proteins = z.mean(axis=1)[None, :]

            def set_score(idx: np.ndarray) -> np.ndarray:
                """Scores (sets x cells) of protein index sets (sets x size)."""
                return z[:, idx].mean(axis=2).T

            observed, random = {}, {}
            for (score, variant), (pos, neg) in SCORE_DEFS.items():
                up = "UP_base" if pos is UP_BASE else "UP_no_Slc12a3" if pos is UP_NO_NCC else "machinery"
                obs = set_score(np.array([[index_of[g] for g in lists[up]]]))
                rnd = set_score(sets[up])
                if neg:
                    obs = obs - set_score(np.array([[index_of[g] for g in lists["DOWN"]]]))
                    rnd = rnd - set_score(sets["DOWN"])
                observed[(score, variant)], random[(score, variant)] = obs, rnd
                stored = stored_scores.loc[cells, f"{source}__{score}__{variant}"].to_numpy(float)
                worst_score = max(worst_score, float(np.abs(obs[0] - stored).max()))
            for definition, (is_p, labels) in definitions.items():
                for comp, wanted in COMPARISONS:
                    ref = ~is_p if wanted is None else (~is_p) & np.isin(labels, wanted)
                    if not ref.any():
                        continue
                    common = {"condition": cond, "source": source, "prolif_definition": definition,
                              "comparison": comp, "n_prolif": int(is_p.sum()), "n_reference": int(ref.sum())}
                    for (score, variant), obs in observed.items():
                        d_obs, c_obs = contrast(obs, is_p, ref)
                        d_null, c_null = contrast(random[(score, variant)], is_p, ref)
                        rows.append({**common, "score": score, "variant": variant,
                                     "n_list_proteins": int(len(SCORE_DEFS[(score, variant)][0]
                                                                and [g for g in SCORE_DEFS[(score, variant)][0]
                                                                     if g in index_of])),
                                     "n_random_sets": N_SETS,
                                     **null_summary(d_obs[0], d_null, "median_difference"),
                                     **null_summary(c_obs[0], c_null, "cohens_d")})
                        if definition == "per_cell_call":
                            t = tabulated[(tabulated["condition"] == cond) & (tabulated["source"] == source)
                                          & (tabulated["reference_annotation"] == "cluster_annotation")
                                          & (tabulated["comparison"] == comp) & (tabulated["score"] == score)
                                          & (tabulated["variant"] == variant)].iloc[0]
                            worst_contrast = max(worst_contrast,
                                                 abs(d_obs[0] - float(t["median_difference"])),
                                                 abs(c_obs[0] - float(t["cohens_d"])))
                    d_all, c_all = contrast(all_proteins, is_p, ref)
                    rows.append({**common, "score": "all_proteins", "variant": "", "n_list_proteins": len(proteins),
                                 "n_random_sets": 0, "median_difference_observed": float(d_all[0]),
                                 "cohens_d_observed": float(c_all[0])})

    checks.add("ALDO scores recomputed from the stored matrices = the stored dct_reviewer per-cell scores (largest absolute "
               "difference)", f"{worst_score:.2e}", worst_score < 1e-4, "< 1e-4")
    checks.add("observed contrasts of the per-cell call = the tabulated contrasts (largest absolute difference of "
               "median difference and Cohen's d)", f"{worst_contrast:.2e}", worst_contrast < 1e-4, "< 1e-4")
    run.table(pd.DataFrame(rows), STEP)
    run.table(pd.DataFrame(protein_rows), f"{STEP}_proteins")
    run.table(checks.frame(), f"{STEP}_checks")
    run.finish({"n_random_sets": N_SETS, "detection_rate_bins": N_BINS, "seed": cc.RANDOM_SEED,
                "sources": SOURCES, "scores": [f"{s} ({v})" for s, v in SCORE_DEFS],
                "random_set_rule": "one protein of the same detection-rate bin per list protein, without replacement "
                                   "within a set, from the proteins in no ALDO list; same sets for both matrices",
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
