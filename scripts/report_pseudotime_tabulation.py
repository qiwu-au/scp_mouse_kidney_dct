#!/usr/bin/env python
"""The response to the reviewers: what the stored diffusion pseudotime of the DCT data follows (root cell, depth, other roots).

. Approved by the authors on 12 Nov 2026 ("Yes, go ahead": plan default 4, one tabulation
of stored values).

Why: the response says what the pseudotime of the DCT data can and cannot show. The stored results (dct_labels, dct_reviewer, thread
12) give the pseudotime of every cell from the root cell of the main analysis, in seven matrices, and group medians for
20 other root cells. They do not say which cell the root is, how the pseudotime relates to the number of proteins per
cell, or how much the ordering changes with the root. This script tabulates that from the stored tables. No pipeline is
rerun: no clustering, embedding, diffusion map or pseudotime is recomputed.

Fixed in the plan before any of these numbers was seen (12 Nov 2026):
  measures   (1) Spearman rho of pseudotime against the number of proteins identified per cell, per condition and
                 for each of the seven matrices;
             (2) the two root cells of the main analysis: proteins, sort plate, labels;
             (3) the 20 other root cells of each condition: rho of their pseudotime with the pseudotime of the main
                 root (stored by dct_labels), and how often the DCT1 median lies above the DCT2 median;
             (4) the stored PAGA connectivities of the marker-score groups (dct_labels), copied.
  stop rule  if |rho| of (1) is below 0.5 in either condition on the main matrix, report to the authors before drafting.
Each condition alone; nothing compares the conditions.
Added in this analysis inside these measures (descriptions of the same relation, not in the plan's list): rho against the sort
plate next to (1); rho of (1) within the groups of Figure 4e; the median pseudotime by quartile of proteins per cell;
proteins and labels of each of the 20 other root cells.
First run (12 Nov 2026): the stop rule was triggered for zero-K (rho -0.497; control -0.993), where the root cell lies
in the middle of the depth range and the pseudotime rises on both sides of it (quartile table). Reported to the authors,
who then approved one more measure ("Yes, add it"):
  measure 5  Spearman rho of pseudotime against the distance in protein number from the root cell,
             |ln(proteins of the cell) - ln(proteins of the root cell)|, per condition and matrix (columns
             spearman_rho_depth_distance_from_root, distance_ci_low, distance_ci_high, distance_p_value).

Tables (results/tables/report/report_15_pseudotime_<name>.csv):
  roots                the root cell of the main analysis per condition: sort plate, well, proteins identified, its
                       rank among the cells of the condition, and its label under the three annotations
  depth                Spearman rho of pseudotime (main root) with proteins per cell, with the distance in protein
                       number from the root cell, and with sort plate: all cells for each of the seven matrices; the
                       groups of Figure 4e for the main matrix
  by_quartile          median pseudotime (main root, main matrix) by quartile of proteins per cell
  other_roots          one row per other root cell (20 per condition): proteins, labels, rho with the main-root
                       pseudotime, group medians under the three annotations, order of the DCT1 and DCT2 medians
  other_roots_summary  per condition: range and sign of rho; roots with the DCT1 median above the DCT2 median; roots
                       with the ProLIF median between the DCT1 and DCT2 medians
  paga                 PAGA connectivities between the groups of the marker-score label (per cell), copied from dct_labels
  checks               reproduction checks

Limits: per-cell pseudotime is stored for the main root only, so rho between roots is the value dct_labels stored and the
groups of Figure 4e (ProLIF cells of the per-cell test) have no medians for the other roots. The 20 other roots are
cells with the per-cell marker-score label DCT1 (the rule of the main analysis), not cells of clusters annotated DCT1.
Spearman rho describes a monotone relation; a pseudotime that rises on both sides of the root is shown by the quartile
table. One pool of 3 mice per condition.

Inputs (read-only): results/tables/report/report_12_percell_prolif_per_cell.csv;
  results/tables/dct_labels/dct_labels_{dct_canonical_labels,submitted_dpt_roots,
  submitted_paga_connectivities}.csv; results/tables/dct_reviewer/dct_reviewer_task4_{dpt_summary,
  dpt_medians_per_root}.csv
Outputs: the tables above; results/tables/report/report_step_15_pseudotime_manifest.json

Command (project root):  python scripts/report_pseudotime_tabulation.py --overwrite
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

import scp_common as cc
import qc_mito_lib as t3
import cluster_de_lib as t5

TASK = "report"
STEP = "15_pseudotime"
DCT_LABELS, DCT_REVIEWER = "dct_labels", "dct_reviewer"
CONDITIONS = ("Ctrl", "ZeroK")
N_CELLS = {"Ctrl": 729, "ZeroK": 702}
MATRICES = ("uncorrected", "cc_removed", "cc_regressed", "tirosh_removed", "tirosh_regressed", "corr_removed",
            "corr_regressed")
MAIN = "uncorrected"
GROUPS = ("DCT1", "DCT2", "ProLIF", "Unknown")
CELL_LABEL = {"Ctrl": "cs_ctrl_cell_label", "ZeroK": "cs_zerok_cell_label"}
STOP_RHO = 0.5
ROOT_RULE = "first cell, in the order sort plate P1 then P2 and ascending well, with the marker-score label DCT1 (per cell)"
QUARTILES = ("Q1 (fewest)", "Q2", "Q3", "Q4 (most)")


def read(run: cc.Run, task: str, name: str) -> pd.DataFrame:
    path = cc.out_path(task, "tables", name, ".csv")
    return pd.read_csv(run.use_input(path, f"{task}: {name}"), keep_default_na=False, na_values=[""],
                       dtype={"well": str})


def dpt(matrix: str) -> str:
    return f"{matrix}__dpt_original_root"


def medians(frame: pd.DataFrame, label: str, column: str) -> dict[str, float]:
    return {g: float(frame.loc[frame[label] == g, column].median()) for g in ("DCT1", "DCT2", "ProLIF")
            if (frame[label] == g).any()}


def order_text(dct1: float, dct2: float) -> str:
    if not (np.isfinite(dct1) and np.isfinite(dct2)):
        return "does not apply"
    return "DCT1 above DCT2" if dct1 > dct2 else "DCT2 above DCT1"


def between(dct1: float, dct2: float, prolif: float) -> bool:
    return bool(min(dct1, dct2) < prolif < max(dct1, dct2))


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: stored pseudotime against depth, root and other roots.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")

    base = read(run, TASK, "12_percell_prolif_per_cell")
    labels = read(run, DCT_LABELS, "dct_canonical_labels")
    t27_roots = read(run, DCT_LABELS, "submitted_dpt_roots")
    paga = read(run, DCT_LABELS, "submitted_paga_connectivities")
    summary = read(run, DCT_REVIEWER, "task4_dpt_summary")
    per_root = read(run, DCT_REVIEWER, "task4_dpt_medians_per_root")

    # --- one row per cell -------------------------------------------------------------------------------------------
    cells = base[["cell_id", "condition", "plate", "n_detected", "dct_label", "second_label", "group", "is_prolif",
                  *[dpt(m) for m in MATRICES]]].copy()
    checks.add("the per-cell tables hold the same 1,431 cells, one row each (729 control, 702 zero-K)",
               f"The response to the reviewers table {len(cells)}; dct_labels labels {len(labels)}",
               cells["cell_id"].is_unique and labels["cell_id"].is_unique
               and set(cells["cell_id"]) == set(labels["cell_id"])
               and cells["condition"].value_counts().to_dict() == N_CELLS)
    lab = labels.set_index("cell_id")
    cells["well"] = cells["cell_id"].map(lambda i: cc.parse_dct_cell_id(i)["well"])
    cells["marker_score_cell_label"] = [lab.loc[i, CELL_LABEL[c]] for i, c in zip(cells["cell_id"], cells["condition"])]
    same = (cells["n_detected"].to_numpy() == lab.loc[cells["cell_id"], "n_detected"].to_numpy()).all()
    same &= (cells["dct_label"].to_numpy() == lab.loc[cells["cell_id"], "dct_label"].to_numpy()).all()
    checks.add("proteins per cell and cluster annotation: the response to the reviewers table = dct_labels canonical labels",
               f"{len(cells)} cells compared", bool(same))
    finite = np.isfinite(cells[[dpt(m) for m in MATRICES]].to_numpy(float)).all()
    checks.add("pseudotime (main root) is finite for every cell in all seven matrices", str(bool(finite)), bool(finite))
    cells["plate_P2"] = (cells["plate"] == "P2").astype(float)
    cells["ln_n_detected"] = np.log(cells["n_detected"].astype(float))

    # --- root cells of the main analysis -------------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond].sort_values(["plate", "well"]).reset_index(drop=True)
        zero = sub[(sub[[dpt(m) for m in MATRICES]] == 0).all(axis=1)]
        stored_root = summary.loc[(summary["condition"] == cond) & (summary["object"] == MAIN),
                                  "original_root_cell_id"].unique()
        t27_root = t27_roots.loc[(t27_roots["condition"] == cond) & t27_roots["root"].str.startswith("original"),
                                 "root_cell_id"].unique()
        first_dct1 = sub.loc[sub["marker_score_cell_label"] == "DCT1", "cell_id"].iloc[0]
        ok = (len(zero) == 1 and len(stored_root) == 1 and len(t27_root) == 1
              and zero["cell_id"].iloc[0] == stored_root[0] == t27_root[0] == first_dct1)
        checks.add(f"{cond}: one cell has pseudotime 0 in all seven matrices; it is the root named by dct_labels and dct_reviewer and "
                   f"the {ROOT_RULE}",
                   f"pseudotime 0: {list(zero['cell_id'])}; first such cell: {first_dct1}", bool(ok))
        root = zero.iloc[0]
        n = sub["n_detected"].to_numpy()
        rows.append({
            "condition": cond, "root_cell_id": root["cell_id"], "sort_plate": root["plate"], "well": root["well"],
            "root_rule": ROOT_RULE, "proteins_identified": int(root["n_detected"]),
            "n_cells": len(sub), "n_cells_with_more_proteins": int((n > root["n_detected"]).sum()),
            "n_cells_with_fewer_proteins": int((n < root["n_detected"]).sum()),
            "median_proteins_all_cells": float(np.median(n)), "min_proteins_all_cells": int(n.min()),
            "max_proteins_all_cells": int(n.max()),
            "marker_score_label_per_cell": root["marker_score_cell_label"],
            "marker_score_label_clusters": root["second_label"], "cluster_annotation": root["dct_label"],
            "figure_3a_group": root["group"], "prolif_per_cell_test": bool(str(root["is_prolif"]).lower() == "true"),
        })
    roots = pd.DataFrame(rows)
    run.table(roots, f"{STEP}_roots")
    root_id = dict(zip(roots["condition"], roots["root_cell_id"]))

    # --- stored medians reproduced from the per-cell pseudotime ---------------------------------------------------------
    worst = 0.0
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        for m in MATRICES:
            got = medians(sub, "second_label", dpt(m))
            st = summary[(summary["condition"] == cond) & (summary["object"] == m)
                         & (summary["label_role"] == "second")].iloc[0]
            worst = max(worst, *[abs(got[g] - float(st[f"original_median_{g}"])) for g in got])
    checks.add("median pseudotime by marker-score label (clusters), seven matrices, recomputed from the per-cell "
               "table = the stored dct_reviewer summary (largest absolute difference)", f"{worst:.2e}", worst < 1e-9, "< 1e-9")
    worst = 0.0
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        got = medians(sub, "marker_score_cell_label", dpt(MAIN))
        st = t27_roots[(t27_roots["condition"] == cond) & t27_roots["root"].str.startswith("original")].iloc[0]
        worst = max(worst, *[abs(got[g] - float(st[f"median_dpt_{g}"])) for g in got])
    checks.add("median pseudotime by marker-score label (per cell), main matrix, recomputed from the per-cell table = "
               "the stored dct_labels root table, so dct_labels and dct_reviewer hold the same pseudotime (largest absolute difference)",
               f"{worst:.2e}", worst < 1e-6, "< 1e-6")

    # --- pseudotime against proteins per cell, distance in protein number from the root, and sort plate -----------------
    root_ln = {c: float(np.log(roots.loc[roots["condition"] == c, "proteins_identified"].iloc[0])) for c in CONDITIONS}
    cells["depth_distance_from_root"] = [abs(v - root_ln[c]) for v, c in zip(cells["ln_n_detected"], cells["condition"])]
    at_root = cells.loc[cells["cell_id"].isin(root_id.values()), "depth_distance_from_root"]
    checks.add("distance in protein number from the root cell, |ln(proteins) - ln(proteins of the root)|: 0 for the "
               "two root cells and finite for every cell",
               f"root cells {list(at_root)}; finite {int(np.isfinite(cells['depth_distance_from_root']).sum())} of "
               f"{len(cells)}", bool((at_root == 0).all() and np.isfinite(cells["depth_distance_from_root"]).all()))
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond]
        sets = [("all cells", m, sub) for m in MATRICES]
        sets += [(g, MAIN, sub[sub["group"] == g]) for g in GROUPS if (sub["group"] == g).any()]
        for name, m, s in sets:
            depth = t3.spearman_ci(s["n_detected"], s[dpt(m)])
            dist = t3.spearman_ci(s["depth_distance_from_root"], s[dpt(m)])
            plate = t3.spearman_ci(s["plate_P2"], s[dpt(m)])
            rows.append({"condition": cond, "cells": name, "matrix": m, "n_cells": depth["n"],
                         "spearman_rho_proteins_per_cell": depth["rho"], "rho_ci_low": depth["ci_low"],
                         "rho_ci_high": depth["ci_high"], "p_value": depth["p_value"],
                         "spearman_rho_depth_distance_from_root": dist["rho"], "distance_ci_low": dist["ci_low"],
                         "distance_ci_high": dist["ci_high"], "distance_p_value": dist["p_value"],
                         "spearman_rho_plate_P2": plate["rho"], "p_plate": plate["p_value"]})
    depth = pd.DataFrame(rows)
    run.table(depth, f"{STEP}_depth")
    main_rho = {c: float(depth.loc[(depth["condition"] == c) & (depth["cells"] == "all cells")
                                   & (depth["matrix"] == MAIN), "spearman_rho_proteins_per_cell"].iloc[0])
                for c in CONDITIONS}
    stop = [c for c in CONDITIONS if abs(main_rho[c]) < STOP_RHO]
    checks.add(f"information, stop rule of the plan: |rho| of pseudotime against proteins per cell on the main matrix "
               f"is at least {STOP_RHO} in both conditions (if not, report to the authors before drafting)",
               "; ".join(f"{c} {main_rho[c]:.3f}" for c in CONDITIONS)
               + ("; STOP RULE TRIGGERED for " + ", ".join(stop) if stop else "; not triggered"), None)
    dist_rho = {c: float(depth.loc[(depth["condition"] == c) & (depth["cells"] == "all cells")
                                   & (depth["matrix"] == MAIN), "spearman_rho_depth_distance_from_root"].iloc[0])
                for c in CONDITIONS}
    checks.add("information, measure 5 (added after the stop rule was reported; authors: 'Yes, add it'): rho of "
               "pseudotime against the distance in protein number from the root cell, main matrix",
               "; ".join(f"{c} {dist_rho[c]:.3f}" for c in CONDITIONS), None)

    # --- median pseudotime by quartile of proteins per cell -------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        sub = cells[cells["condition"] == cond].copy()
        sub["quartile"] = pd.qcut(sub["n_detected"].rank(method="first"), 4, labels=list(QUARTILES))
        root_q = sub.loc[sub["cell_id"] == root_id[cond], "quartile"].iloc[0]
        for q in QUARTILES:
            s = sub[sub["quartile"] == q]
            q25, med, q75 = np.percentile(s[dpt(MAIN)].to_numpy(float), [25, 50, 75])
            rows.append({"condition": cond, "quartile_of_proteins_per_cell": q, "n_cells": len(s),
                         "proteins_min": int(s["n_detected"].min()), "proteins_max": int(s["n_detected"].max()),
                         "pseudotime_median": float(med), "pseudotime_q25": float(q25), "pseudotime_q75": float(q75),
                         "holds_root_cell": bool(q == root_q)})
    by_quartile = pd.DataFrame(rows)
    run.table(by_quartile, f"{STEP}_by_quartile")

    # --- the 20 other root cells ------------------------------------------------------------------------------------------
    info = cells.set_index("cell_id")
    rows = []
    for cond in CONDITIONS:
        t27 = t27_roots[(t27_roots["condition"] == cond) & t27_roots["root"].str.startswith("random")]
        t28 = per_root[(per_root["condition"] == cond) & (per_root["object"] == MAIN)]
        same_roots = all(
            set(t28.loc[t28["root"] == r["root"], "root_cell_id"]) == {r["root_cell_id"]} for _, r in t27.iterrows())
        labelled_dct1 = all(info.loc[i, "marker_score_cell_label"] == "DCT1" for i in t27["root_cell_id"])
        checks.add(f"{cond}: 20 other roots; dct_labels and dct_reviewer name the same cell for each; every root has the marker-score "
                   "label DCT1 (per cell); no pseudotime is infinite",
                   f"{len(t27)} roots; same cells {same_roots}; all DCT1 {labelled_dct1}; infinite "
                   f"{int(t28['n_infinite'].sum())}",
                   len(t27) == 20 and same_roots and labelled_dct1 and int(t28["n_infinite"].sum()) == 0)
        for _, r in t27.iterrows():
            cell = info.loc[r["root_cell_id"]]
            prim = t28[(t28["root"] == r["root"]) & (t28["label_role"] == "primary")].iloc[0]
            sec = t28[(t28["root"] == r["root"]) & (t28["label_role"] == "second")].iloc[0]
            a1, a2 = float(prim["median_DCT1"]), float(prim["median_DCT2"])
            s1, s2, sp = float(sec["median_DCT1"]), float(sec["median_DCT2"]), float(sec["median_ProLIF"])
            c1, c2, cp = float(r["median_dpt_DCT1"]), float(r["median_dpt_DCT2"]), float(r["median_dpt_ProLIF"])
            rows.append({
                "condition": cond, "root": r["root"].replace("random_", "other root "),
                "root_cell_id": r["root_cell_id"], "sort_plate": cell["plate"],
                "proteins_identified": int(cell["n_detected"]), "cluster_annotation": cell["dct_label"],
                "marker_score_label_clusters": cell["second_label"],
                "spearman_rho_with_main_root_pseudotime": float(r["spearman_rho_vs_original"]),
                "annotation_median_DCT1": a1, "annotation_median_DCT2": a2,
                "annotation_order": order_text(a1, a2),
                "clusters_median_DCT1": s1, "clusters_median_DCT2": s2, "clusters_median_ProLIF": sp,
                "clusters_order": order_text(s1, s2), "clusters_prolif_between": between(s1, s2, sp),
                "per_cell_median_DCT1": c1, "per_cell_median_DCT2": c2, "per_cell_median_ProLIF": cp,
                "per_cell_order": order_text(c1, c2), "per_cell_prolif_between": between(c1, c2, cp),
            })
            if bool(str(sec["prolif_between_dct1_dct2"]).lower() == "true") != between(s1, s2, sp):
                raise SystemExit(f"{cond} {r['root']}: 'ProLIF between' differs from the stored dct_reviewer flag.")
    other = pd.DataFrame(rows)
    run.table(other, f"{STEP}_other_roots")

    rows = []
    for cond in CONDITIONS:
        o = other[other["condition"] == cond]
        rho = o["spearman_rho_with_main_root_pseudotime"].to_numpy(float)
        main = {"annotation": summary[(summary["condition"] == cond) & (summary["object"] == MAIN)
                                      & (summary["label_role"] == "primary")].iloc[0],
                "clusters": summary[(summary["condition"] == cond) & (summary["object"] == MAIN)
                                    & (summary["label_role"] == "second")].iloc[0]}
        t27_main = t27_roots[(t27_roots["condition"] == cond) & t27_roots["root"].str.startswith("original")].iloc[0]
        has_dct2 = bool((o["annotation_order"] != "does not apply").all())
        row = {"condition": cond, "n_other_roots": len(o), "rho_min": float(rho.min()), "rho_median":
               float(np.median(rho)), "rho_max": float(rho.max()), "n_roots_rho_negative": int((rho < 0).sum()),
               "n_roots_rho_at_least_0.9": int((rho >= 0.9).sum()), "n_roots_rho_at_most_minus_0.9":
               int((rho <= -0.9).sum()),
               "main_root_annotation_order": order_text(float(main["annotation"]["original_median_DCT1"]),
                                                        float(main["annotation"]["original_median_DCT2"])),
               "n_roots_annotation_DCT1_above_DCT2": int((o["annotation_order"] == "DCT1 above DCT2").sum())
               if has_dct2 else np.nan,
               "main_root_clusters_order": order_text(float(main["clusters"]["original_median_DCT1"]),
                                                      float(main["clusters"]["original_median_DCT2"])),
               "n_roots_clusters_DCT1_above_DCT2": int((o["clusters_order"] == "DCT1 above DCT2").sum()),
               "n_roots_clusters_prolif_between": int(o["clusters_prolif_between"].sum()),
               "main_root_per_cell_order": order_text(float(t27_main["median_dpt_DCT1"]),
                                                      float(t27_main["median_dpt_DCT2"])),
               "n_roots_per_cell_DCT1_above_DCT2": int((o["per_cell_order"] == "DCT1 above DCT2").sum()),
               "n_roots_per_cell_prolif_between": int(o["per_cell_prolif_between"].sum())}
        rows.append(row)
        stored_between = int(main["clusters"]["n_random_prolif_between"])
        checks.add(f"{cond}: other roots with the ProLIF median between the DCT1 and DCT2 medians, marker-score label "
                   "(clusters), main matrix = the stored dct_reviewer summary",
                   f"{row['n_roots_clusters_prolif_between']} of {len(o)}",
                   row["n_roots_clusters_prolif_between"] == stored_between, f"{stored_between} of 20")
    other_summary = pd.DataFrame(rows)
    run.table(other_summary, f"{STEP}_other_roots_summary")

    # --- PAGA connectivities (dct_labels), copied --------------------------------------------------------------------------------
    out = paga.rename(columns={"source": "group_1", "target": "group_2"}).copy()
    out.insert(1, "groups", "marker-score label (per cell)")
    sizes = {c: cells[cells["condition"] == c]["marker_score_cell_label"].value_counts().to_dict() for c in CONDITIONS}
    out["n_cells_group_1"] = [sizes[c][g] for c, g in zip(out["condition"], out["group_1"])]
    out["n_cells_group_2"] = [sizes[c][g] for c, g in zip(out["condition"], out["group_2"])]
    run.table(out, f"{STEP}_paga")
    checks.add("PAGA table: three pairs of groups per condition; group sizes = the marker-score labels per cell "
               "(control DCT1 350, DCT2 207, ProLIF 172; zero-K 321, 212, 169)",
               "; ".join(f"{c}: {sizes[c]}" for c in CONDITIONS),
               out.groupby("condition").size().to_dict() == {"Ctrl": 3, "ZeroK": 3}
               and sizes["Ctrl"] == {"DCT1": 350, "DCT2": 207, "ProLIF": 172}
               and sizes["ZeroK"] == {"DCT1": 321, "DCT2": 212, "ProLIF": 169})

    run.table(checks.frame(), f"{STEP}_checks")
    run.finish({"pseudotime": "stored values (dct_labels / dct_reviewer / the response to the reviewers tables); nothing recomputed",
                "matrices": list(MATRICES), "main_matrix": MAIN, "root_rule": ROOT_RULE,
                "spearman_interval": "Fisher z, SE = 1.06 / sqrt(n - 3) (qc_mito_lib.spearman_ci)",
                "stop_rule": f"|rho| of pseudotime against proteins per cell on the main matrix < {STOP_RHO} in "
                             "either condition: report to the authors before drafting",
                "stop_rule_main_matrix_rho": main_rho, "stop_rule_triggered_for": stop,
                "measure_5": "rho of pseudotime against |ln(proteins) - ln(proteins of the root cell)|; added after "
                             "the stop rule was reported to the authors (12 Nov 2026, 'Yes, add it')",
                "measure_5_main_matrix_rho": dist_rho, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
