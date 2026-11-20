#!/usr/bin/env python
"""The response to the reviewers: the per-cell ProLIF cells of Figure 4e in the stored dissociation-stress and mito results.

. Approved by the authors on 12 Nov 2026 ("Yes, go ahead":
plan default 1, one tabulation of stored per-cell values).

Why: dct_reviewer (dissociation-stress score, mitochondrial metrics and flags) could report ProLIF only for the marker-score
label of the clusters (cs_<cond>_cluster_label; control 119, zero-K 98 cells), because it was run before the per-cell
proliferation test existed. Figures 4e and 4f show another group: the cells called ProLIF per cell by the dct_prolif test
(route A, no-floor run; control 37, zero-K 63). This script joins the stored per-cell tables with that call and
reports the same quantities for those cells. No pipeline is rerun: no score, clustering or embedding is recomputed.
New here are group summaries, group tests and a prevalence table; the only values read from a data object are the
detection (intensity > 0) of the stress-panel proteins, Mki67 and Top2a in the dct_intake objects.

Fixed in the plan before any of these numbers was seen (12 Nov 2026):
  groups       ProLIF = per-cell test (dct_prolif route A, no-floor run); every other cell keeps its cluster annotation
               (dct_label: DCT1, DCT2, Unknown = "DCT, subtype not resolved"). Each condition alone; nothing compares
               the conditions.
  comparisons  ProLIF against all other cells, and against the other cells of clusters annotated DCT1 or DCT2
               (zero-K: DCT1 only, no cluster is annotated DCT2). These two form the planned family.
  measures     the three stored stress scores, the MitoCarta fraction of summed intensity, log10 total intensity (a
               check for doublets), and the stored mitochondrial flags.
  tests        Mann-Whitney U, and the dct_reviewer models (qc_mito_lib.prolif_contrasts: OLS with HC3 errors; label only,
               + log10 protein IDs, + log10 protein IDs + sort plate); flags by Fisher's exact test;
               Benjamini-Hochberg within condition over the planned family.
  stop rule    if a stress score is higher in ProLIF after adjustment, report to the authors before drafting.
Rows that compare ProLIF with DCT1 or with DCT2 alone are written as a second, descriptive family ("by_subtype").

Tables (results/tables/report/report_14_prolif_stress_<name>.csv):
  groups           cells, sort plate, protein IDs, total intensity, stress scores, mito fractions and flags, for the
                   groups of Figure 4e, the marker-score label of the clusters, and all cells
  panel_detection  cells with each quantified stress-panel protein detected, per group
  contrasts        ProLIF against the reference groups, per measure (Mann-Whitney U, effect sizes, three OLS models)
  flags            ProLIF against the reference groups, per stored flag (Fisher's exact test)
  score_vs_depth   Spearman rho of each stress score with log10 protein IDs, all cells and within the Figure 4e groups
  prevalence       cells per definition of a proliferating cell (per-cell test, held-out proteins, marker-score
                   labels, cell-cycle score rule), by sort plate, protein-ID quartile and cluster annotation, with
                   Wilson 95% intervals; and the Mouse Kidney Atlas DCT rows (transcripts; stored dct_rna / atlas_dct values)
  per_cell         one row per cell with every value used
  checks           reproduction checks (group counts; stored dct_reviewer numbers recomputed from the same per-cell tables)

Limits: the stress panel is 18 genes, of which 7 protein groups are quantified in control and 6 in zero-K (heat-shock
proteins; Fos in 7 control cells); the scores follow the number of proteins per cell. The OLS models are descriptions,
not causal adjustments. The prevalence rows are counts of sorted cells of one pool of 3 mice per condition.

Inputs (read-only): results/tables/report/report_12_percell_prolif_per_cell.csv;
  results/tables/dct_reviewer/dct_reviewer_{stress_per_cell,stress_coverage,stress_detection_by_label,
  stress_prolif_contrasts,mito_qc_per_cell}.csv; results/tables/dct_prolif/dct_prolif_dct_labels.csv;
  results/tables/dct_labels/dct_labels_dct_canonical_labels.csv;
  results/tables/dct_rna/dct_rna_cycling_side_by_side.csv; results/h5ad/dct_intake_DCT_{ctrl,zerok}.h5ad
Outputs: the tables above; results/tables/report/report_step_14_prolif_stress_manifest.json

Command (project root):  python scripts/report_prolif_stress_tabulation.py --overwrite
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
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

import scp_common as cc
import qc_mito_lib as t3
import cluster_de_lib as t5

TASK = "report"
STEP = "14_prolif_stress"
DCT_INTAKE, DCT_LABELS, DCT_REVIEWER, DCT_RNA, DCT_PROLIF = ("dct_intake", "dct_labels", "dct_reviewer",
                           "dct_rna", "dct_prolif")
CONDITIONS = ("Ctrl", "ZeroK")
H5AD = {"Ctrl": "DCT_ctrl", "ZeroK": "DCT_zerok"}
PROLIF = "ProLIF"
GROUPS = ("DCT1", "DCT2", PROLIF, "Unknown")
STRESS = tuple(t3.STRESS_SCORES)                       # stress_mean_z, stress_score_genes, stress_n_detected
MEASURES = (*STRESS, "mitocarta_intensity_fraction", "log10_total_intensity")
SUMMARY = ("n_detected", "total_intensity", *STRESS, "mitocarta_intensity_fraction", "mitocarta_id_fraction",
           "oxphos_intensity_fraction")
FLAGS = ("mito_high", "mito_low", "mito_any", "depth_low", "depth_high", "any")
MODELS = tuple(t3.CONTRAST_MODELS)
P_COLS = ("mannwhitney_p", *[f"{m}_p" for m in MODELS])
# Figure 4e groups (dct_prolif README section 16, no-floor run; the response to the reviewers tabulation).
EXPECTED = {"Ctrl": {"DCT1": 362, "DCT2": 79, PROLIF: 37, "Unknown": 251},
            "ZeroK": {"DCT1": 402, PROLIF: 63, "Unknown": 237}}
# dct_labels README (headline and section 5), dct_reviewer README section 4, dct_prolif README section 2.
EXPECTED_COUNTS = {"Ctrl": {"marker_score_clusters": 119, "marker_score_per_cell": 172, "cycling": 38,
                            "score_rule": 73, "Mki67": 15, "Top2a": 16, "either": 21},
                   "ZeroK": {"marker_score_clusters": 98, "marker_score_per_cell": 169, "cycling": 72,
                             "score_rule": 71, "Mki67": 52, "Top2a": 35, "either": 56}}
ATLAS_ROWS = (("DCT", "atlas DCT class, cells and nuclei"), ("DCT, Cell", "atlas DCT class, single cells"),
              ("DCT, Nuclei", "atlas DCT class, nuclei"))


def read(run: cc.Run, task: str, name: str) -> pd.DataFrame:
    path = cc.out_path(task, "tables", name, ".csv")
    return pd.read_csv(run.use_input(path, f"{task}: {name}"), keep_default_na=False, na_values=[""])


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def quartiles(values, prefix: str) -> dict[str, float]:
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not v.size:
        return {f"{prefix}_median": np.nan, f"{prefix}_q25": np.nan, f"{prefix}_q75": np.nan}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {f"{prefix}_median": float(med), f"{prefix}_q25": float(q1), f"{prefix}_q75": float(q3)}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def prevalence_row(condition: str, definition: str, subset: str, k: int, n: int, source: str) -> dict:
    lo, hi = wilson(int(k), int(n))
    return {"condition": condition, "definition": definition, "subset": subset, "k": int(k), "n_cells": int(n),
            "percent": 100 * k / n if n else np.nan, "ci_low_percent": 100 * lo, "ci_high_percent": 100 * hi,
            "source": source}


def detection(run: cc.Run, cond: str, proteins: list[str], cell_ids: pd.Series) -> pd.DataFrame:
    """Detection (intensity > 0) of the named proteins in the dct_intake object of the condition, in the given cell order."""
    path = run.use_input(cc.out_path(DCT_INTAKE, "h5ad", H5AD[cond], ".h5ad"), f"{DCT_INTAKE}: object {cond}")
    adata = ad.read_h5ad(path)
    missing = [p for p in proteins if p not in adata.var_names]
    if missing:
        raise KeyError(f"{cond}: proteins not in the dct_intake object: {missing}")
    sub = adata[cell_ids.to_numpy(), proteins]
    values = cc.dense(sub.layers["intensity"])
    return pd.DataFrame(np.nan_to_num(values, nan=0.0) > 0, columns=[f"det_{p}" for p in proteins],
                        index=cell_ids.to_numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: per-cell ProLIF cells in the stored stress results.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")

    base = read(run, TASK, "12_percell_prolif_per_cell")
    stress = read(run, DCT_REVIEWER, "stress_per_cell")
    coverage = read(run, DCT_REVIEWER, "stress_coverage")
    stored_detection = read(run, DCT_REVIEWER, "stress_detection_by_label")
    stored_contrasts = read(run, DCT_REVIEWER, "stress_prolif_contrasts")
    mito = read(run, DCT_REVIEWER, "mito_qc_per_cell")
    t31 = read(run, DCT_PROLIF, "dct_labels")
    t27 = read(run, DCT_LABELS, "dct_canonical_labels")
    rna = read(run, DCT_RNA, "cycling_side_by_side")

    # --- one row per cell ---------------------------------------------------------------------------------------
    cells = base[["cell_id", "condition", "plate", "n_detected", "dct_label", "second_label", "group", "is_prolif",
                  "slc12a3_detected", "mki67_or_top2a_detected", "cell_cycle_score", "reviewer_cycling_status",
                  "prolif_z", "prolif_q"]].copy()
    for col in ("is_prolif", "slc12a3_detected", "mki67_or_top2a_detected"):
        cells[col] = as_bool(cells[col])
    tables = {"the response to the reviewers per-cell table": base, "dct_reviewer stress": stress, "dct_reviewer mito": mito, "dct_prolif labels": t31,
              "dct_labels labels": t27}
    checks.add("every per-cell table holds the same 1,431 cells, one row each (729 control, 702 zero-K)",
               "; ".join(f"{k} {len(v)}" for k, v in tables.items()),
               all(v["cell_id"].is_unique and set(v["cell_id"]) == set(cells["cell_id"]) for v in tables.values())
               and cells["condition"].value_counts().to_dict() == {"Ctrl": 729, "ZeroK": 702})
    cells = cells.merge(stress[["cell_id", "n_detected_proteins", "total_intensity", "log10_n_detected",
                                "log10_total_intensity", *STRESS]], on="cell_id", validate="one_to_one")
    checks.add("protein IDs per cell: the response to the reviewers table = dct_reviewer stress table",
               f"{int((cells['n_detected'] == cells['n_detected_proteins']).sum())} of {len(cells)} equal",
               bool((cells["n_detected"] == cells["n_detected_proteins"]).all()))
    mito_cols = ["mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
                 *[f"flag__primary__{f}" for f in FLAGS], *[f"flag__second__{f}" for f in FLAGS]]
    cells = cells.merge(mito[["cell_id", *mito_cols]], on="cell_id", validate="one_to_one")
    for f in FLAGS:
        for role in ("primary", "second"):
            cells[f"flag__{role}__{f}"] = as_bool(cells[f"flag__{role}__{f}"])
    cells["mito_flagged_either_label_set"] = cells["flag__primary__mito_any"] | cells["flag__second__mito_any"]
    cells = cells.merge(t31[["cell_id", "cs_t31nf_cycling"]], on="cell_id", validate="one_to_one")
    cells["cycling_per_cell_test"] = cells["cs_t31nf_cycling"].astype(str).eq("Cycling")
    cell_label = t27.set_index("cell_id")
    cells["marker_score_cell_label"] = [
        cell_label.loc[i, "cs_ctrl_cell_label" if c == "Ctrl" else "cs_zerok_cell_label"]
        for i, c in zip(cells["cell_id"], cells["condition"])]
    cells["group_all"] = np.where(cells["is_prolif"], PROLIF, "other")
    for cond in CONDITIONS:
        got = cells.loc[cells["condition"] == cond, "group"].value_counts().to_dict()
        checks.add(f"{cond}: groups = the Figure 3A groups", "; ".join(f"{g} {got[g]}" for g in GROUPS if g in got),
                   got == EXPECTED[cond], str(EXPECTED[cond]))
    checks.add("mito-flagged cells under either label set = the dct_reviewer exclusion set (control 128, zero-K 101)",
               "; ".join(f"{c} {int(cells.loc[cells['condition'] == c, 'mito_flagged_either_label_set'].sum())}"
                         for c in CONDITIONS),
               [int(cells.loc[cells["condition"] == c, "mito_flagged_either_label_set"].sum())
                for c in CONDITIONS] == [128, 101])

    # --- detection of the panel proteins, Mki67 and Top2a (dct_intake objects) -------------------------------------------
    panel = {}
    det_frames = []
    for cond in CONDITIONS:
        cov = coverage[(coverage["condition"] == cond) & as_bool(coverage["present"])]
        panel[cond] = list(dict.fromkeys(cov["matrix_protein"].astype(str)))
        ids = cells.loc[cells["condition"] == cond, "cell_id"]
        det = detection(run, cond, [*panel[cond], "Mki67", "Top2a"], ids)
        det_frames.append(det)
        n_det = det[[f"det_{p}" for p in panel[cond]]].sum(axis=1).to_numpy()
        stored = cells.loc[cells["condition"] == cond, "stress_n_detected"].to_numpy(float)
        either = (det["det_Mki67"] | det["det_Top2a"]).to_numpy()
        exp = EXPECTED_COUNTS[cond]
        checks.add(f"{cond}: stress-panel proteins detected per cell (dct_intake object) = the stored dct_reviewer value",
                   f"{len(panel[cond])} protein groups ({', '.join(panel[cond])}); "
                   f"{int((n_det == stored).sum())} of {len(stored)} cells equal", bool((n_det == stored).all()))
        checks.add(f"{cond}: Mki67, Top2a and either detected = dct_reviewer / dct_rna; either = the the response to the reviewers column",
                   f"{int(det['det_Mki67'].sum())}; {int(det['det_Top2a'].sum())}; {int(either.sum())}",
                   (int(det["det_Mki67"].sum()), int(det["det_Top2a"].sum()), int(either.sum()))
                   == (exp["Mki67"], exp["Top2a"], exp["either"])
                   and bool((either == cells.loc[cells["condition"] == cond, "mki67_or_top2a_detected"]
                             .to_numpy()).all()), f"{exp['Mki67']}; {exp['Top2a']}; {exp['either']}")
    det_all = pd.concat(det_frames).fillna(False).astype(bool)
    cells = cells.merge(det_all, left_on="cell_id", right_index=True, validate="one_to_one")

    partitions = (("groups_of_figure_3a", "group"), ("marker_score_label_clusters", "second_label"))

    # --- groups -------------------------------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        sets = [("all_cells", "all", sub)]
        sets += [(part, g, sub[sub[col] == g]) for part, col in partitions for g in GROUPS if (sub[col] == g).any()]
        for part, g, s in sets:
            row = {"condition": cond, "partition": part, "group": g, "n_cells": len(s),
                   "n_P1": int((s["plate"] == "P1").sum()), "n_P2": int((s["plate"] == "P2").sum())}
            for m in SUMMARY:
                row.update(quartiles(s[m], m))
            for f in FLAGS:
                row[f"n_flag_{f}"] = int(s[f"flag__primary__{f}"].sum())
            row["n_mito_flagged_either_label_set"] = int(s["mito_flagged_either_label_set"].sum())
            row["n_per_cell_prolif"] = int(s["is_prolif"].sum())
            row["n_mki67_or_top2a_detected"] = int(s["mki67_or_top2a_detected"].sum())
            rows.append(row)
    groups = pd.DataFrame(rows)
    run.table(groups, f"{STEP}_groups")

    # --- panel detection ----------------------------------------------------------------------------------------
    rows, worst = [], 0.0
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        sets = [("all_cells", "all", sub)]
        sets += [(part, g, sub[sub[col] == g]) for part, col in partitions for g in GROUPS if (sub[col] == g).any()]
        for part, g, s in sets:
            for p in panel[cond]:
                k = int(s[f"det_{p}"].sum())
                rows.append({"condition": cond, "partition": part, "group": g, "protein": p, "n_cells": len(s),
                             "n_detected": k, "percent_detected": 100 * k / len(s)})
                if part == "marker_score_label_clusters":
                    st = stored_detection[(stored_detection["condition"] == cond)
                                          & (stored_detection["label_role"] == "second")
                                          & (stored_detection["label"] == g)
                                          & (stored_detection["matrix_protein"] == p)]
                    worst = max(worst, abs(float(st["n_detected"].iloc[0]) - k) if len(st) else np.inf)
    panel_detection = pd.DataFrame(rows)
    run.table(panel_detection, f"{STEP}_panel_detection")
    checks.add("panel detection by marker-score label (clusters) = the stored dct_reviewer table (largest difference in cells)",
               f"{worst:g}", worst == 0)

    # --- contrasts ----------------------------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond].reset_index(drop=True)
        has_dct2 = bool((sub["group"] == "DCT2").any())
        planned = [t3.prolif_contrasts(sub, "group_all", MEASURES, contrasts=((PROLIF, ("other",)),)),
                   t3.prolif_contrasts(sub, "group", MEASURES, contrasts=((PROLIF, ("DCT1", "DCT2")),))]
        subtype = [t3.prolif_contrasts(sub, "group", MEASURES, contrasts=((PROLIF, ("DCT1",)),))]
        if has_dct2:
            subtype.append(t3.prolif_contrasts(sub, "group", MEASURES, contrasts=((PROLIF, ("DCT2",)),)))
        for family, frames in (("planned", planned), ("by_subtype", subtype)):
            out = pd.concat(frames, ignore_index=True)
            out = out.drop(columns=[c for c in out.columns if c.endswith("_fdr_bh")] + ["label_column"])
            out.insert(0, "family", family)
            out.insert(0, "condition", cond)
            out["status"], out["reason"] = "tested", ""
            rows.append(out)
        if not has_dct2:
            rows.append(pd.DataFrame([{"condition": cond, "family": "by_subtype", "comparison": "ProLIF_vs_DCT2",
                                       "metric": m, "status": "does not apply",
                                       "reason": "no cluster is annotated DCT2 in this condition"}
                                      for m in MEASURES]))
    contrasts = pd.concat(rows, ignore_index=True)
    contrasts["comparison"] = contrasts["comparison"].replace(
        {"ProLIF_vs_other": "ProLIF_vs_all_other_cells", "ProLIF_vs_DCT1+DCT2": "ProLIF_vs_DCT1_DCT2"})
    for p_col in P_COLS:
        contrasts = cc.add_bh(contrasts, ["condition", "family"], p_col=p_col,
                              out_col=p_col.removesuffix("_p") + "_p_fdr_bh")
    run.table(contrasts, f"{STEP}_contrasts")
    # reproduction of the stored dct_reviewer contrasts (marker-score ProLIF against DCT1 + DCT2)
    worst = 0.0
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond].reset_index(drop=True)
        got = t3.prolif_contrasts(sub, "second_label", STRESS, contrasts=((PROLIF, ("DCT1", "DCT2")),))
        for _, r in got.iterrows():
            st = stored_contrasts[(stored_contrasts["condition"] == cond) & (stored_contrasts["label_role"] == "second")
                                  & (stored_contrasts["comparison"] == "ProLIF_vs_DCT1+DCT2")
                                  & (stored_contrasts["metric"] == r["metric"])].iloc[0]
            worst = max(worst, *[abs(float(r[f"{m}_coef"]) - float(st[f"{m}_coef"])) for m in MODELS],
                        abs(float(r["mannwhitney_p"]) - float(st["mannwhitney_p"])))
    checks.add("stress contrasts of the marker-score label (ProLIF against DCT1 + DCT2) recomputed from the per-cell "
               "table = the stored dct_reviewer table (largest absolute difference of the three coefficients and the "
               "Mann-Whitney p)", f"{worst:.2e}", worst < 1e-9, "< 1e-9")

    # --- flags --------------------------------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        p = sub["is_prolif"]
        refs = (("ProLIF_vs_all_other_cells", ~p), ("ProLIF_vs_DCT1_DCT2", ~p & sub["group"].isin(["DCT1", "DCT2"])))
        for comp, ref in refs:
            for f in (*FLAGS, "mito_flagged_either_label_set"):
                col = f if f == "mito_flagged_either_label_set" else f"flag__primary__{f}"
                a, b = int(sub.loc[p, col].sum()), int(p.sum())
                c, d = int(sub.loc[ref, col].sum()), int(ref.sum())
                odds, pval = stats.fisher_exact([[a, b - a], [c, d - c]])
                rows.append({"condition": cond, "comparison": comp, "flag": f, "n_prolif": b, "n_prolif_flagged": a,
                             "percent_prolif_flagged": 100 * a / b, "n_reference": d, "n_reference_flagged": c,
                             "percent_reference_flagged": 100 * c / d, "fisher_odds_ratio": float(odds),
                             "p_value": float(pval)})
    flags = cc.add_bh(pd.DataFrame(rows), ["condition"])
    run.table(flags, f"{STEP}_flags")

    # --- stress scores against protein IDs ------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        for g, s in [("all", sub), *[(g, sub[sub["group"] == g]) for g in GROUPS if (sub["group"] == g).any()]]:
            for score in STRESS:
                rows.append({"condition": cond, "group": g, "score": score,
                             **t3.spearman_ci(s["log10_n_detected"], s[score])})
    run.table(pd.DataFrame(rows), f"{STEP}_score_vs_depth")

    # --- prevalence ---------------------------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond].copy()
        sub["ids_quartile"] = pd.qcut(sub["n_detected"].rank(method="first"), 4, labels=["Q1 (fewest)", "Q2", "Q3",
                                                                                         "Q4 (most)"])
        n = len(sub)
        exp = EXPECTED_COUNTS[cond]
        definitions = (
            ("per-cell test: ProLIF (cycling, NCC detected)", sub["is_prolif"], "dct_prolif route A, no-floor run"),
            ("per-cell test: cycling (with or without NCC detected)", sub["cycling_per_cell_test"],
             "dct_prolif route A, no-floor run"),
            ("Mki67 or Top2a detected", sub["mki67_or_top2a_detected"], "dct_intake object; dct_reviewer README section 4"),
            ("Mki67 detected", sub["det_Mki67"], "dct_intake object"),
            ("Top2a detected", sub["det_Top2a"], "dct_intake object"),
            ("marker-score label (clusters): ProLIF", sub["second_label"].eq(PROLIF), "dct_labels canonical labels"),
            ("marker-score label (per cell): ProLIF", sub["marker_score_cell_label"].eq(PROLIF),
             "dct_labels canonical labels"),
            ("cell-cycle score above its 90th percentile (10% by construction)",
             sub["reviewer_cycling_status"].astype(str).eq("Cycling"), "dct_reviewer README section 4"),
        )
        for name, mask, source in definitions:
            rows.append(prevalence_row(cond, name, "all cells", mask.sum(), n, source))
        for name, mask, source in definitions[:3]:
            for plate in ("P1", "P2"):
                s = sub["plate"] == plate
                rows.append(prevalence_row(cond, name, f"sort plate {plate}", (mask & s).sum(), s.sum(), source))
            for q in sub["ids_quartile"].cat.categories:
                s = sub["ids_quartile"] == q
                lo_ids, hi_ids = int(sub.loc[s, "n_detected"].min()), int(sub.loc[s, "n_detected"].max())
                rows.append(prevalence_row(cond, name, f"protein IDs {q}: {lo_ids}-{hi_ids}", (mask & s).sum(),
                                           s.sum(), source))
            for lab in ("DCT1", "DCT2", "Unknown"):
                s = sub["dct_label"] == lab
                if s.any():
                    rows.append(prevalence_row(cond, name, f"cluster annotation {lab}", (mask & s).sum(), s.sum(),
                                               source))
        got = {"marker_score_clusters": int(sub["second_label"].eq(PROLIF).sum()),
               "marker_score_per_cell": int(sub["marker_score_cell_label"].eq(PROLIF).sum()),
               "cycling": int(sub["cycling_per_cell_test"].sum()),
               "score_rule": int(sub["reviewer_cycling_status"].astype(str).eq("Cycling").sum())}
        checks.add(f"{cond}: marker-score ProLIF (clusters; per cell), cycling cells of the per-cell test and cells "
                   "of the score rule = dct_labels, dct_prolif and dct_reviewer",
                   "; ".join(f"{k} {v}" for k, v in got.items()), all(got[k] == exp[k] for k in got),
                   "; ".join(f"{k} {exp[k]}" for k in got))
    for population, name in ATLAS_ROWS:
        for measure in ("Mki67_or_Top2a", "Mki67", "Top2a"):
            r = rna[(rna["condition"] == "both") & (rna["population"] == population) & (rna["measure"] == measure)]
            r = r.iloc[0]
            rows.append(prevalence_row("Mouse Kidney Atlas (RNA)", f"{measure.replace('_', ' ')} transcripts detected",
                                       name, int(r["k"]), int(r["n_cells"]),
                                       "dct_rna cycling_side_by_side (atlas_dct values reused)"))
    prevalence = pd.DataFrame(rows)
    run.table(prevalence, f"{STEP}_prevalence")
    atlas = prevalence[(prevalence["condition"] == "Mouse Kidney Atlas (RNA)")
                       & (prevalence["definition"] == "Mki67 or Top2a transcripts detected")].set_index("subset")
    checks.add("atlas DCT cells with Mki67 or Top2a transcripts = the response to the reviewers (217 of 16,381; nuclei 27 of 3,098; "
               "single cells 190 of 13,283)",
               "; ".join(f"{i}: {int(r['k'])} of {int(r['n_cells'])}" for i, r in atlas.iterrows()),
               [(int(r["k"]), int(r["n_cells"])) for _, r in atlas.iterrows()]
               == [(217, 16381), (190, 13283), (27, 3098)])

    # --- per cell -----------------------------------------------------------------------------------------------
    keep = ["cell_id", "condition", "plate", "n_detected", "total_intensity", "dct_label", "second_label",
            "marker_score_cell_label", "group", "is_prolif", "cycling_per_cell_test", "prolif_z", "prolif_q",
            "slc12a3_detected", "mki67_or_top2a_detected", "det_Mki67", "det_Top2a", *STRESS,
            *[c for c in cells.columns if c.startswith("det_") and c not in ("det_Mki67", "det_Top2a")],
            "mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
            *[f"flag__primary__{f}" for f in FLAGS], "mito_flagged_either_label_set"]
    run.table(cells[keep], f"{STEP}_per_cell")

    run.table(checks.frame(), f"{STEP}_checks")
    run.finish({"groups": "ProLIF = dct_prolif route A, no-floor run; other cells by the dct_labels cluster annotation",
                "planned_family": ["ProLIF_vs_all_other_cells", "ProLIF_vs_DCT1_DCT2"],
                "measures": list(MEASURES), "flags": [*FLAGS, "mito_flagged_either_label_set"],
                "models": {k: list(v) for k, v in t3.CONTRAST_MODELS.items()},
                "bh_family": "condition x family (contrasts); condition (flags)",
                "stress_panel": panel, "wilson_alpha": 0.05, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
