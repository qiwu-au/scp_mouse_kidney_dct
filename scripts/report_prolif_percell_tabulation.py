#!/usr/bin/env python
"""The response to the reviewers: the per-cell ProLIF cells of Figure 3A in the stored cell-cycle, pseudotime and ALDO results.

. Approved by the authors
on 12 Nov 2026 ("Yes, go ahead": plan choice 1, tabulation for the per-cell ProLIF cells).

Why: dct_reviewer (cell-cycle correction, pseudotime, ALDO scores) could report ProLIF only for the marker-score label of the
clusters (cs_<cond>_cluster_label; control 119, zero-K 98 cells), because the cluster annotation (dct_labels) has no ProLIF
cluster. Figure 3A shows another group: the cells called ProLIF per cell by the dct_prolif proliferation test (route A,
no-floor run; control 37, zero-K 63). This script joins the stored per-cell tables of dct_reviewer with that call and reports,
for those cells, the same quantities. No pipeline is rerun: no clustering, no embedding, no pseudotime and no score is
recomputed; every per-cell value is read from a stored table. New here are only group summaries and group tests.

Groups (the groups of Figure 3A): ProLIF = t31_label == 'ProLIF'; every other cell keeps its cluster annotation
(dct_label: DCT1, DCT2, Unknown = "DCT, subtype not resolved"). Each condition alone; nothing compares the conditions.

Tables (results/tables/report/report_12_percell_prolif_<name>.csv):
  groups               cells, protein IDs, sort plate, dct_reviewer cell-cycle score and cycling rule, Mki67 / Top2a, per group
  pseudotime_by_group  stored diffusion pseudotime (dct_labels root) per group, for the 7 objects of the dct_reviewer cell-cycle step
  pseudotime_summary   per object: is the ProLIF median between the DCT1 and DCT2 medians; ProLIF against DCT1, DCT2,
                       DCT1 + DCT2 and all other cells (Mann-Whitney U, effect sizes); two reference annotations: the
                       cluster annotation and the marker-score label of the clusters (ProLIF cells taken out of both)
  pseudotime_same_cluster_reference
                       the ProLIF median against the medians of 1,000 draws of as many cells that are not ProLIF,
                       taken from the same stored Leiden clusters (authors, 12 Nov 2026: "Add the reference"); seed 42
  aldo_by_group        stored ALDO scores per group (3 scores x 4 variants x 2 sources)
  aldo_contrasts       ProLIF against DCT1, DCT2, DCT1 + DCT2 and all other cells: Mann-Whitney U, effect sizes
                       (scp_common.compare_groups, as dct_reviewer), BH within condition x reference annotation x source;
                       and the ProLIF coefficient of a linear model score ~ ProLIF + z ln(protein IDs) + sort plate
  cluster_spread       ProLIF cells over the stored Leiden clusters of the 7 cell-cycle objects and of the 4 other
                       dct_reviewer objects (depth-, plate-, depth-and-cell-cycle-regressed; proteins detected in >= 50%)
  checks               reproduction checks (group counts; dct_reviewer numbers recomputed from the same per-cell tables)

Limits: pseudotime from the dct_labels root only (per-cell values for the 20 random roots were not stored). The linear model
is ordinary least squares on the stored scores; it is a description, not a causal adjustment.

Inputs (read-only): results/tables/dct_prolif/dct_prolif_dct_labels.csv;
  results/tables/dct_labels/dct_labels_dct_canonical_labels.csv;
  results/tables/dct_reviewer/dct_reviewer_{task4_dpt_per_cell_original_root,task4_dpt_summary,
  task4_clusters_per_cell,embed_clusters_per_cell,task4_cell_cycle_scores_per_cell,reannot_per_cell,
  aldo_scores_per_cell,aldo_contrasts}.csv
Outputs: the tables above; results/tables/report/report_step_12_percell_prolif_manifest.json

Command (project root):  python scripts/report_prolif_percell_tabulation.py --overwrite
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

import numpy as np
import pandas as pd
import statsmodels.api as sm

import scp_common as cc
import cluster_de_lib as t5

TASK = "report"
STEP = "12_percell_prolif"
DCT_LABELS, DCT_REVIEWER, DCT_PROLIF = "dct_labels", "dct_reviewer", "dct_prolif"
CONDITIONS = ("Ctrl", "ZeroK")
PROLIF = "ProLIF"
GROUPS = ("DCT1", "DCT2", PROLIF, "Unknown")
CC_OBJECTS = ("uncorrected", "cc_removed", "cc_regressed", "tirosh_removed", "tirosh_regressed", "corr_removed",
              "corr_regressed")
EMBED_OBJECTS = ("depth_regressed", "plate_regressed", "depth_cc_regressed", "highcov50")
RESOLUTIONS = ("0.30", "0.50", "0.80", "1.00", "3.00")
SOURCES = ("scaled_X", "cc_regressed_3a")
SCORES = ("ALDO_activity", "ALDO_signed", "ALDO_machinery")
VARIANTS = ("base", "no_Slc12a3", "no_cell_cycle", "no_Slc12a3_no_cell_cycle")
REFERENCES = {"cluster_annotation": "dct_label", "marker_score_clusters": "second_label"}
COMPARISONS = (("ProLIF_vs_DCT1", ("DCT1",)), ("ProLIF_vs_DCT2", ("DCT2",)),
               ("ProLIF_vs_DCT1_DCT2", ("DCT1", "DCT2")), ("ProLIF_vs_all_other_cells", None))
# Figure 3A groups (dct_prolif README section 16, no-floor run; list of items, Figure 3A and 3B rows).
EXPECTED = {"Ctrl": {"DCT1": 362, "DCT2": 79, PROLIF: 37, "Unknown": 251},
            "ZeroK": {"DCT1": 402, PROLIF: 63, "Unknown": 237}}
EXPECTED_BY_LABEL = {"Ctrl": {"DCT1": 20, "DCT2": 6, "Unknown": 11}, "ZeroK": {"DCT1": 47, "Unknown": 16}}
# dct_prolif README section 16: stored clusters holding ProLIF cells, and the largest ProLIF fraction of a cluster.
EXPECTED_SPREAD = {"Ctrl": (8, 13, 0.18), "ZeroK": (9, 13, 0.20)}
N_DRAWS = 1000                             # same-cluster random reference for the pseudotime median


def read(run: cc.Run, task: str, name: str) -> pd.DataFrame:
    path = cc.out_path(task, "tables", name, ".csv")
    return pd.read_csv(run.use_input(path, f"{task}: {name}"), keep_default_na=False, na_values=[""])


def quartiles(values) -> dict[str, float]:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not v.size:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q25": float(q1), "q75": float(q3)}


def reference_cells(sub: pd.DataFrame, column: str, labels) -> pd.Series:
    """Cells of the reference group: not ProLIF (per-cell call), with one of `labels` (None = every other cell)."""
    mask = ~sub["is_prolif"]
    return mask if labels is None else mask & sub[column].isin(labels)


def adjusted(sub: pd.DataFrame, value: str, ref: pd.Series) -> dict[str, float]:
    """OLS value ~ ProLIF + z ln(protein IDs) + sort plate, over the ProLIF and the reference cells."""
    use = sub[sub["is_prolif"] | ref].dropna(subset=[value])
    out = {"adj_n": len(use), "adj_coef": np.nan, "adj_ci_low": np.nan, "adj_ci_high": np.nan, "adj_p_value": np.nan,
           "adj_coef_z_ln_ids": np.nan, "adj_coef_plate_P2": np.nan}
    if use["is_prolif"].sum() < 3 or (~use["is_prolif"]).sum() < 3:
        return out
    ln_ids = np.log(use["n_detected"].to_numpy(float))
    x = pd.DataFrame({"ProLIF": use["is_prolif"].astype(float).to_numpy(),
                      "z_ln_ids": (ln_ids - ln_ids.mean()) / ln_ids.std(ddof=0),
                      "plate_P2": (use["plate"] == "P2").astype(float).to_numpy()})
    fit = sm.OLS(use[value].to_numpy(float), sm.add_constant(x)).fit()
    ci = fit.conf_int()
    out.update(adj_coef=float(fit.params["ProLIF"]), adj_ci_low=float(ci.loc["ProLIF", 0]),
               adj_ci_high=float(ci.loc["ProLIF", 1]), adj_p_value=float(fit.pvalues["ProLIF"]),
               adj_coef_z_ln_ids=float(fit.params["z_ln_ids"]), adj_coef_plate_P2=float(fit.params["plate_P2"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: per-cell ProLIF cells in the stored dct_reviewer results.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")

    t31 = read(run, DCT_PROLIF, "dct_labels")
    t27 = read(run, DCT_LABELS, "dct_canonical_labels")
    dpt = read(run, DCT_REVIEWER, "task4_dpt_per_cell_original_root")
    dpt_summary = read(run, DCT_REVIEWER, "task4_dpt_summary")
    cl_cc = read(run, DCT_REVIEWER, "task4_clusters_per_cell")
    cl_embed = read(run, DCT_REVIEWER, "embed_clusters_per_cell")
    ccs = read(run, DCT_REVIEWER, "task4_cell_cycle_scores_per_cell")
    reannot = read(run, DCT_REVIEWER, "reannot_per_cell")
    aldo = read(run, DCT_REVIEWER, "aldo_scores_per_cell")
    aldo_stored = read(run, DCT_REVIEWER, "aldo_contrasts")

    # --- one row per cell ---------------------------------------------------------------------------------------
    cells = t31[["cell_id", "cell_key", "condition", "plate", "n_detected", "t31_label", "t31_label_source",
                 "cs_t31nf_identity", "mki67_or_top2a_detected", "slc12a3_detected", "dct_label"]].copy()
    checks.add("dct_prolif table: one row per cell, 729 control and 702 zero-K cells",
               "; ".join(f"{c} {int((cells['condition'] == c).sum())}" for c in CONDITIONS),
               cells["cell_id"].is_unique and (cells["condition"].value_counts().to_dict() == {"Ctrl": 729, "ZeroK": 702}))
    checks.add("dct_prolif primary run is the no-floor run (t31_label = cs_t31nf_label)",
               "; ".join(sorted(set(cells["t31_label_source"].astype(str)))),
               bool((t31["t31_label"] == t31["cs_t31nf_label"]).all()))
    both = cells.merge(t27[["cell_id", "dct_label"]], on="cell_id", how="left", suffixes=("", "_t27"),
                       validate="one_to_one")
    checks.add("cluster annotation in the dct_prolif table = the dct_labels canonical table (dct_label, all cells)",
               f"{int((both['dct_label'] == both['dct_label_t27']).sum())} of {len(both)} equal",
               bool((both["dct_label"] == both["dct_label_t27"]).all()))
    cells["is_prolif"] = cells["t31_label"].eq(PROLIF)
    cells["group"] = np.where(cells["is_prolif"], PROLIF, cells["dct_label"])
    stored = {"cell cycle": ccs, "cycling rule": reannot, "pseudotime": dpt, "ALDO": aldo, "clusters": cl_cc,
              "clusters (other objects)": cl_embed}
    checks.add("every stored dct_reviewer per-cell table holds the same 1,431 cells",
               "; ".join(f"{k} {len(v)}" for k, v in stored.items()),
               all(v["cell_id"].is_unique and set(v["cell_id"]) == set(cells["cell_id"]) for v in stored.values()))
    cells = cells.merge(ccs[["cell_id", "second_label", "cell_cycle_score", "S_score", "G2M_score"]], on="cell_id",
                        validate="one_to_one")
    cells = cells.merge(reannot[["cell_id", "reviewer_cycling_status"]], on="cell_id", validate="one_to_one")
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        got = sub["group"].value_counts().to_dict()
        by_label = sub.loc[sub["is_prolif"], "dct_label"].value_counts().to_dict()
        checks.add(f"{cond}: groups = the Figure 3A groups; ProLIF cells by cluster annotation",
                   "; ".join(f"{g} {got[g]}" for g in GROUPS if g in got) + " | ProLIF by cluster annotation: "
                   + "; ".join(f"{k} {by_label[k]}" for k in GROUPS if k in by_label),
                   got == EXPECTED[cond] and by_label == EXPECTED_BY_LABEL[cond],
                   f"{EXPECTED[cond]} | {EXPECTED_BY_LABEL[cond]}")

    # --- groups -------------------------------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        for g in GROUPS:
            s = sub[sub["group"] == g]
            if s.empty:
                continue
            ids, score = quartiles(s["n_detected"]), quartiles(s["cell_cycle_score"])
            second = s["second_label"].value_counts().to_dict()
            rows.append({"condition": cond, "group": g, "n_cells": len(s), "n_P1": int((s["plate"] == "P1").sum()),
                         "n_P2": int((s["plate"] == "P2").sum()), "protein_ids_median": ids["median"],
                         "protein_ids_q25": ids["q25"], "protein_ids_q75": ids["q75"],
                         "cell_cycle_score_median": score["median"], "cell_cycle_score_q25": score["q25"],
                         "cell_cycle_score_q75": score["q75"],
                         "n_cycling_score_rule": int((s["reviewer_cycling_status"] == "Cycling").sum()),
                         "n_mki67_or_top2a_detected": int(s["mki67_or_top2a_detected"].astype(str).str.lower()
                                                           .eq("true").sum()),
                         "n_slc12a3_detected": int(s["slc12a3_detected"].astype(str).str.lower().eq("true").sum()),
                         **{f"n_marker_score_{k}": int(second.get(k, 0)) for k in GROUPS}})
    groups = pd.DataFrame(rows)
    run.table(groups, f"{STEP}_groups")

    # --- pseudotime ---------------------------------------------------------------------------------------------
    cells_dpt = cells.merge(dpt.drop(columns="condition"), on="cell_id", validate="one_to_one")
    by_group, summary = [], []
    repro = 0.0
    for cond in CONDITIONS:
        sub = cells_dpt[cells_dpt["condition"] == cond]
        for obj in CC_OBJECTS:
            col = f"{obj}__dpt_original_root"
            # reproduction of the stored dct_reviewer medians (marker-score label of the clusters)
            st = dpt_summary[(dpt_summary["condition"] == cond) & (dpt_summary["object"] == obj)
                             & (dpt_summary["label_role"] == "second")].iloc[0]
            for lab in ("DCT1", "DCT2", PROLIF):
                repro = max(repro, abs(float(sub.loc[sub["second_label"] == lab, col].median())
                                       - float(st[f"original_median_{lab}"])))
            for g in GROUPS:
                s = sub[sub["group"] == g]
                if not s.empty:
                    by_group.append({"condition": cond, "object": obj, "group": g, **quartiles(s[col])})
            for ref_name, ref_col in REFERENCES.items():
                med = {lab: float(sub.loc[reference_cells(sub, ref_col, (lab,)), col].median())
                       if reference_cells(sub, ref_col, (lab,)).any() else np.nan for lab in ("DCT1", "DCT2")}
                med_p = float(sub.loc[sub["is_prolif"], col].median())
                applicable = bool(np.isfinite(med["DCT1"]) and np.isfinite(med["DCT2"]))
                between = (min(med.values()) < med_p < max(med.values())) if applicable else None
                for comp, labels in COMPARISONS:
                    ref = reference_cells(sub, ref_col, labels)
                    row = {"condition": cond, "object": obj, "reference_annotation": ref_name, "comparison": comp,
                           "median_ProLIF": med_p, "median_DCT1": med["DCT1"], "median_DCT2": med["DCT2"],
                           "prolif_between_dct1_and_dct2": between if applicable else "does not apply",
                           "status": "tested" if ref.any() else "does not apply",
                           "reason": "" if ref.any() else "no reference cell under this annotation"}
                    row.update(cc.compare_groups(sub.loc[sub["is_prolif"], col], sub.loc[ref, col]) if ref.any()
                               else {})
                    summary.append(row)
    checks.add("pseudotime: medians of the marker-score label (clusters) recomputed from the per-cell table = the "
               "stored dct_reviewer summary (largest absolute difference)", f"{repro:.2e}", repro < 1e-6, "< 1e-6")
    pt_by_group = pd.DataFrame(by_group)
    pt_summary = cc.add_bh(pd.DataFrame(summary), ["condition", "reference_annotation"])
    run.table(pt_by_group, f"{STEP}_pseudotime_by_group")
    run.table(pt_summary, f"{STEP}_pseudotime_summary")

    # --- pseudotime: reference of random cells from the same clusters (authors, 12 Nov 2026: "Add the reference") -----
    # For every ProLIF cell one cell that is not ProLIF is drawn from the same stored Leiden cluster of the cluster
    # annotation (without replacement within a draw); the median pseudotime of each draw is the reference.
    rng = np.random.default_rng(cc.RANDOM_SEED)
    with_cluster = cells_dpt.merge(cl_cc[["cell_id", "uncorrected__stored_primary_cluster"]], on="cell_id",
                                   validate="one_to_one")
    rows, pools_ok = [], True
    for cond in CONDITIONS:
        sub = with_cluster[with_cluster["condition"] == cond].reset_index(drop=True)
        cluster = sub["uncorrected__stored_primary_cluster"].astype(str).to_numpy()
        is_p = sub["is_prolif"].to_numpy(bool)
        need = pd.Series(cluster[is_p]).value_counts().sort_index()
        pools = {c: np.flatnonzero((cluster == c) & ~is_p) for c in need.index}
        pools_ok &= all(len(pools[c]) >= n for c, n in need.items())
        draws = np.array([np.concatenate([rng.choice(pools[c], size=int(n), replace=False) for c, n in need.items()])
                          for _ in range(N_DRAWS)])
        for obj in CC_OBJECTS:
            v = sub[f"{obj}__dpt_original_root"].to_numpy(float)
            obs = float(np.median(v[is_p]))
            null = np.median(v[draws], axis=1)
            centre = float(null.mean())
            row = {"condition": cond, "object": obj, "n_prolif": int(is_p.sum()),
                   "n_clusters_with_prolif": int(len(need)), "n_draws": N_DRAWS, "median_ProLIF": obs,
                   "reference_median": float(np.median(null)), "reference_q025": float(np.quantile(null, 0.025)),
                   "reference_q975": float(np.quantile(null, 0.975)),
                   "prolif_inside_reference_95": bool(np.quantile(null, 0.025) <= obs <= np.quantile(null, 0.975)),
                   "p_empirical_two_sided": float((1 + np.sum(np.abs(null - centre) >= abs(obs - centre)))
                                                  / (N_DRAWS + 1))}
            for ref_name in REFERENCES:
                s = pt_summary[(pt_summary["condition"] == cond) & (pt_summary["object"] == obj)
                               & (pt_summary["reference_annotation"] == ref_name)].iloc[0]
                lo, hi = sorted((float(s["median_DCT1"]), float(s["median_DCT2"])))
                ok = bool(np.isfinite(lo) and np.isfinite(hi))
                row[f"prolif_between__{ref_name}"] = s["prolif_between_dct1_and_dct2"]
                row[f"reference_fraction_between__{ref_name}"] = (float(np.mean((null > lo) & (null < hi))) if ok
                                                                  else np.nan)
            rows.append(row)
    checks.add("same-cluster reference: every cluster holds enough cells that are not ProLIF for a draw",
               f"{N_DRAWS} draws per condition, seed {cc.RANDOM_SEED}", bool(pools_ok))
    run.table(pd.DataFrame(rows), f"{STEP}_pseudotime_same_cluster_reference")

    # --- ALDO scores --------------------------------------------------------------------------------------------
    score_cols = [f"{s}__{sc}__{v}" for s in SOURCES for sc in SCORES for v in VARIANTS]
    cells_aldo = cells.merge(aldo[["cell_id", *score_cols]], on="cell_id", validate="one_to_one")
    by_group, contrasts = [], []
    for cond in CONDITIONS:
        sub = cells_aldo[cells_aldo["condition"] == cond]
        for source in SOURCES:
            for score in SCORES:
                for variant in VARIANTS:
                    col = f"{source}__{score}__{variant}"
                    for g in GROUPS:
                        s = sub[sub["group"] == g]
                        if not s.empty:
                            by_group.append({"condition": cond, "source": source, "score": score, "variant": variant,
                                             "group": g, **quartiles(s[col])})
                    for ref_name, ref_col in REFERENCES.items():
                        for comp, labels in COMPARISONS:
                            ref = reference_cells(sub, ref_col, labels)
                            row = {"condition": cond, "reference_annotation": ref_name, "comparison": comp,
                                   "source": source, "score": score, "variant": variant,
                                   "status": "tested" if ref.any() else "does not apply",
                                   "reason": "" if ref.any() else "no reference cell under this annotation"}
                            if ref.any():
                                row.update(cc.compare_groups(sub.loc[sub["is_prolif"], col], sub.loc[ref, col]))
                                row.update(adjusted(sub, col, ref))
                            contrasts.append(row)
    aldo_by_group = pd.DataFrame(by_group)
    family = ["condition", "reference_annotation", "source"]
    aldo_contrasts = cc.add_bh(pd.DataFrame(contrasts), family)
    aldo_contrasts = cc.add_bh(aldo_contrasts, family, p_col="adj_p_value", out_col="adj_p_fdr_bh")
    run.table(aldo_by_group, f"{STEP}_aldo_by_group")
    run.table(aldo_contrasts, f"{STEP}_aldo_contrasts")
    # reproduction of the stored dct_reviewer contrasts (marker-score ProLIF against marker-score DCT1, all cells)
    worst = 0.0
    for cond in CONDITIONS:
        sub = cells_aldo[cells_aldo["condition"] == cond]
        for col in score_cols:
            st = aldo_stored[(aldo_stored["condition"] == cond) & (aldo_stored["label_role"] == "second")
                             & (aldo_stored["cohort"] == "all_cells") & (aldo_stored["comparison"] == "ProLIF_vs_DCT1")
                             & (aldo_stored["score"] == col)]
            if st.empty:
                continue
            got = cc.compare_groups(sub.loc[sub["second_label"] == PROLIF, col], sub.loc[sub["second_label"] == "DCT1", col])
            worst = max(worst, abs(got["median_difference"] - float(st["median_difference"].iloc[0])),
                        abs(got["cohens_d"] - float(st["cohens_d"].iloc[0])))
    checks.add("ALDO: the stored dct_reviewer contrasts (marker-score ProLIF against DCT1, all cells) recomputed from the "
               "per-cell table (largest absolute difference of median difference and Cohen's d)", f"{worst:.2e}",
               worst < 1e-6, "< 1e-6")

    # --- ProLIF cells over the stored clusters ------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        for table, objects in ((cl_cc, CC_OBJECTS), (cl_embed, EMBED_OBJECTS)):
            t = sub[["cell_id", "is_prolif"]].merge(table, on="cell_id", validate="one_to_one")
            for obj in objects:
                for res in RESOLUTIONS:
                    cluster = t[f"{obj}__r{res}"].astype(str)
                    size = cluster.value_counts()
                    n_p = cluster[t["is_prolif"]].value_counts().reindex(size.index, fill_value=0)
                    frac = (n_p / size).sort_values(ascending=False)
                    top = n_p.sort_values(ascending=False)
                    total = int(t["is_prolif"].sum())
                    rows.append({"condition": cond, "object": obj, "resolution": res, "n_prolif": total,
                                 "n_clusters": int(len(size)), "n_clusters_with_prolif": int((n_p > 0).sum()),
                                 "top1_n": int(top.iloc[0]), "top1_percent": 100 * float(top.iloc[0]) / total,
                                 "top3_n": int(top.iloc[:3].sum()), "top3_percent": 100 * float(top.iloc[:3].sum()) / total,
                                 "max_prolif_fraction_of_a_cluster": float(frac.iloc[0]),
                                 "cluster_of_max_fraction_n_cells": int(size[frac.index[0]]),
                                 "cluster_of_max_fraction_n_prolif": int(n_p[frac.index[0]])})
    spread = pd.DataFrame(rows)
    run.table(spread, f"{STEP}_cluster_spread")
    for cond in CONDITIONS:
        r = spread[(spread["condition"] == cond) & (spread["object"] == "uncorrected")
                   & (spread["resolution"] == "1.00")].iloc[0]
        n_with, n_clusters, max_frac = EXPECTED_SPREAD[cond]
        sub = cells[cells["condition"] == cond].merge(cl_cc, on="cell_id", validate="one_to_one")
        same = pd.crosstab(sub["uncorrected__r1.00"], sub["uncorrected__stored_primary_cluster"])
        one_to_one = bool(((same > 0).sum(axis=0) == 1).all() and ((same > 0).sum(axis=1) == 1).all())
        checks.add(f"{cond}: ProLIF cells over the stored clusters of the cluster annotation (resolution 1) = the dct_prolif "
                   "overlay table",
                   f"{int(r['n_clusters_with_prolif'])} of {int(r['n_clusters'])} clusters; largest ProLIF fraction "
                   f"{r['max_prolif_fraction_of_a_cluster']:.2f}; rerun clusters = stored clusters: {one_to_one}",
                   int(r["n_clusters_with_prolif"]) == n_with and int(r["n_clusters"]) == n_clusters
                   and abs(round(float(r["max_prolif_fraction_of_a_cluster"]), 2) - max_frac) < 1e-9 and one_to_one,
                   f"{n_with} of {n_clusters}; {max_frac}")

    # --- clusters that the annotation rule labels ProLIF in the stored resolution scan (dct_labels), against the call ------
    scan = read(run, DCT_LABELS, "scan_clusters_per_cell").merge(cells[["cell_id", "is_prolif", "second_label"]],
                                                         on="cell_id", validate="one_to_one")
    rows = []
    for cond in CONDITIONS:
        sub = scan[scan["condition"] == cond]
        for col in [c for c in scan.columns if c.startswith("r") and c.endswith("_label")]:
            res, rule = col.split("__")[0][1:], col.split("__")[1].replace("_label", "")
            hit = sub[sub[col] == PROLIF]
            rows.append({"condition": cond, "resolution": float(res), "rule": rule, "n_cells_rule_prolif": len(hit),
                         "n_clusters_rule_prolif": int(hit[f"r{res}__cluster"].nunique()),
                         "n_per_cell_test_prolif_among_them": int(hit["is_prolif"].sum()),
                         "n_marker_score_prolif_among_them": int((hit["second_label"] == PROLIF).sum()),
                         "median_protein_ids": float(hit["n_detected"].median()) if len(hit) else np.nan})
    rule_prolif = pd.DataFrame(rows)
    run.table(rule_prolif, f"{STEP}_rule_prolif_clusters")
    r3 = rule_prolif[(rule_prolif["resolution"] == 3.0) & (rule_prolif["rule"] == "top25")].set_index("condition")
    checks.add("resolution 3, top-25 rule: the one cluster labelled ProLIF holds only ProLIF cells of the per-cell "
               "call (control 10, zero-K 12)",
               "; ".join(f"{c}: {int(r3.loc[c, 'n_per_cell_test_prolif_among_them'])} of "
                         f"{int(r3.loc[c, 'n_cells_rule_prolif'])}" for c in CONDITIONS),
               all(int(r3.loc[c, "n_cells_rule_prolif"]) == n == int(r3.loc[c, "n_per_cell_test_prolif_among_them"])
                   for c, n in (("Ctrl", 10), ("ZeroK", 12))))

    # --- per cell (for the supplementary figures and table; stored values only) ----------------------------------
    nf = t31.set_index("cell_id")
    per_cell = cells[["cell_id", "condition", "plate", "n_detected", "dct_label", "second_label", "group",
                      "is_prolif", "slc12a3_detected", "mki67_or_top2a_detected", "cell_cycle_score", "S_score",
                      "G2M_score", "reviewer_cycling_status"]].copy()
    for col in ("n_P_detected", "prolif_D", "prolif_null_mean", "prolif_null_sd", "prolif_p", "prolif_q", "axis"):
        per_cell[col] = nf.loc[per_cell["cell_id"], f"cs_t31nf_{col}"].to_numpy()
    per_cell["prolif_z"] = (per_cell["prolif_D"] - per_cell["prolif_null_mean"]) / per_cell["prolif_null_sd"]
    per_cell = per_cell.merge(dpt.drop(columns="condition"), on="cell_id", validate="one_to_one")
    keep = [f"{s}__{sc}__{v}" for s in SOURCES for sc, v in (("ALDO_activity", "base"), ("ALDO_activity", "no_Slc12a3"),
                                                            ("ALDO_signed", "base"), ("ALDO_signed", "no_Slc12a3"),
                                                            ("ALDO_machinery", "base"))]
    per_cell = per_cell.merge(aldo[["cell_id", *keep]], on="cell_id", validate="one_to_one")
    cycling = nf.loc[per_cell["cell_id"], "cs_t31nf_cycling"].astype(str).to_numpy()
    q_ok = (per_cell["prolif_q"].to_numpy(float) < 0.05) == (cycling == "Cycling")
    checks.add("per-cell table: the proliferation call (q < 0.05) = the stored cycling call; ProLIF = cycling with "
               "Slc12a3 detected",
               f"{int(q_ok.sum())} of {len(per_cell)} cells agree; cycling {int((cycling == 'Cycling').sum())}, "
               f"ProLIF {int(per_cell['is_prolif'].sum())}",
               bool(q_ok.all()) and bool((per_cell["is_prolif"].to_numpy()
                                          == ((cycling == "Cycling")
                                              & per_cell["slc12a3_detected"].astype(str).str.lower().eq("true")
                                              .to_numpy())).all()))
    run.table(per_cell, f"{STEP}_per_cell")

    run.table(checks.frame(), f"{STEP}_checks")
    run.finish({"groups": "ProLIF = dct_prolif route A, no-floor run (t31_label); other cells by the dct_labels cluster annotation",
                "reference_annotations": REFERENCES, "cell_cycle_objects": list(CC_OBJECTS),
                "other_objects": list(EMBED_OBJECTS), "resolutions": list(RESOLUTIONS),
                "aldo": {"sources": list(SOURCES), "scores": list(SCORES), "variants": list(VARIANTS)},
                "bh_family": "condition x reference annotation (x source for ALDO)",
                "same_cluster_reference": {"n_draws": N_DRAWS, "seed": cc.RANDOM_SEED,
                                           "clusters": "stored Leiden clusters of the cluster annotation"},
                "adjusted_model": "OLS score ~ ProLIF + z ln(protein IDs) + sort plate (ProLIF and reference cells)",
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
