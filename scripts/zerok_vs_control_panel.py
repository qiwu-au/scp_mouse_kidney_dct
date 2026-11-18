#!/usr/bin/env python
"""zerok_vs_control step 3: the pre-specified panel, zero-K vs control, combined search (zerok_response_core step 2 rerun).

Analysis plan,
step 3. The frozen zerok_core panel (SHA256 checked first) is tested under the zerok_response_core scoring rule D1, with the zerok_response_core functions
(zerok_response_core_lib, zerok_response_core_panel helpers) imported unchanged:

- per protein: detection fraction per condition (difference ZeroK - Ctrl with Newcombe 95% CI, Fisher p);
  ln-intensity among detected cells (medians, mean-based log2 fold change, Mann-Whitney p); the two-part model
  (Firth logistic on detection + OLS on ln-intensity among detected, both with log n_detected of the joint object;
  LR tests summed: 2 df, or 1 df without the intensity part) = primary test;
- per score (ALDO with and without Slc12a3, ALDO machinery, cell-cycle score): OLS score ~ condition + log depth
  (primary) and Mann-Whitney;
- pseudobulk log2 fold change (D6) per sort plate and pooled;
- BH within the panel per label column x group x variant; observed direction beside the frozen expected direction;
  match / mismatch / not testable (D1).
Groups: DCT1, DCT2, ProLIF, Unknown and all_DCT (DCT1 + DCT2 + ProLIF), under dct_label and dct_label_alternative;
every variant (none = baseline, cell_median, condition_align). A group with no cells in a condition (the label is
absent under the frozen dct_labels rule) gets 'does not apply' rows. All p-values are descriptive (one pool per condition;
cells are not animal-level replicates).

Per-cell scores (D3) for every variant are written here and reused by steps 5-7.

Inputs (read-only): results/h5ad/dct_labels_DCT_joint_labelled.h5ad; dct_labels canonical labels; zerok_core panel and hash;
dct_intake panel coverage and normalization variants; data/DCT/DCT_markers_V1.xlsx (labels_core scoring lists).
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_per_cell_scores.csv, zerok_vs_control_panel_results.csv,
zerok_vs_control_panel_pseudobulk.csv, zerok_vs_control_panel_expected_vs_observed.csv, zerok_vs_control_panel_match_summary.csv,
zerok_vs_control_score_lists.csv, zerok_vs_control_panel_step_checks.csv.

Command (project root):  python scripts/zerok_vs_control_panel.py
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
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import labels_core_lib as t18
import zerok_response_core_panel as t22p
import dct_intake_lib as t26
import zerok_vs_control_lib as lib

ensure_scanpy_compat()
SCRIPT = "zerok_vs_control_panel.py"
t22 = lib.t22
KEY_COLUMNS = t22p.KEY_COLUMNS


def absent_panel_rows(joint: lib.Joint, spec: t22.PanelSpec, col: str, group: str, why: str) -> list[dict]:
    """'does not apply' rows for every panel item and variant of a group with no cells in a condition."""
    rows = []
    for item in spec.item_order:
        for v in lib.VARIANTS:
            row = lib.absent_rows(joint, col, group, why, variant=v, item=item)
            row.update(item_type="score" if item in t22.SCORE_ITEMS else "protein", testable=False,
                       expected_direction=spec.expected[item], observed_direction=lib.DOES_NOT_APPLY,
                       match_status=lib.DOES_NOT_APPLY, match_reason=why)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 3: pre-specified panel.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "panel", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    joint = lib.load_joint(run, checks)
    panel = t22.read_panel(run)
    spec = t22.panel_spec(joint, panel)

    resolved = [(item, name) for item, name, _ in spec.proteins]
    checks.add("panel proteins found in the joint object", f"{sum(1 for _, n in resolved if n)} of {len(resolved)}",
               all(n for _, n in resolved), "22 of 22")
    renamed = [f"{i}->{n}" for i, n in resolved if n and n != i]
    checks.add("panel proteins whose joint var name differs from the gene", ", ".join(renamed) or "none", None)
    members = t22.score_members_in_joint(joint)
    cov = t26.read_table(run, "panel_coverage").query("item_type == 'score'").set_index("item")["n_in_joint"]
    recorded = cov.astype(float).astype(int)
    same = all(members[i] == recorded[i] for i in recorded.index)
    checks.add("score members in the joint object equal dct_intake panel coverage (n_in_joint)",
               "; ".join(f"{i} {members[i]}" for i in recorded.index), same)
    t21_rec = panel.loc[panel["item_type"] == "score"].set_index("item")["n_in_joint"].astype(float).astype(int)
    checks.add("score members: zerok_core (separate searches) n_in_joint, for comparison",
               "; ".join(f"{i} {t21_rec[i]}" for i in t21_rec.index), None)
    checks.add("panel rows tested (proteins + scores)", f"{len(spec.item_order)}", len(spec.item_order) == 28, "28")
    t22p.model_checks(joint, checks)

    # ---- per-cell scores (D3)
    workbook = t18.read_workbook(run)
    type_lists = t18.present_lists(workbook, joint.var_names)
    list_rows = [{"list": f"{t} (DCT V1 workbook)", "n_in_workbook": len(workbook[t]), "n_in_joint": len(g),
                  "present": ";".join(g)} for t, g in type_lists.items()]
    var = set(joint.var_names)
    for item, (pos, neg) in t22.t21.SCORE_LISTS.items():
        list_rows.append({"list": item, "n_in_workbook": len(dict.fromkeys(pos + neg)), "n_in_joint": members[item],
                          "present": ";".join(g for g in dict.fromkeys(pos + neg) if g in var)})
    run.table(pd.DataFrame(list_rows), "score_lists")
    checks.add("DCT V1 workbook types scored", "; ".join(f"{t} {len(g)}" for t, g in type_lists.items()),
               {"DCT1", "DCT2"} <= set(type_lists), "DCT1 and DCT2 present")
    scores = {}
    for v in lib.VARIANTS:
        print(f"  scoring variant {v}", flush=True)
        scores[v] = t22.score_cells(joint, v, type_lists)
    base_cols = ["cell_key", "condition", "plate", "dct_label", "dct_label_alternative", lib.DEPTH_COL]
    per_cell = joint.obs[base_cols].copy()
    for v in lib.VARIANTS:
        per_cell = per_cell.join(scores[v].add_prefix(f"{v}__"))
    finite = bool(np.isfinite(per_cell.filter(like="__").to_numpy(float)).all())
    checks.add("per-cell scores finite in every variant", f"{finite}", finite)
    run.table(per_cell.reset_index(), "per_cell_scores")

    # ---- panel statistics
    rows, pb_rows, na_rows = [], [], []
    bts = {v: t22.back_transform(joint, v) for v in lib.VARIANTS}
    for col in lib.LABEL_COLUMNS:
        masks, absent = lib.split_groups(joint, col)
        for group, m in masks.items():
            print(f"  panel: {col} / {group}", flush=True)
            rows.extend(t22.panel_rows(joint, spec, np.flatnonzero(m), scores,
                                       keys={"label_column": col, "group": group}))
        for group, why in absent.items():
            na_rows.extend(absent_panel_rows(joint, spec, col, group, why))
        pb_rows.extend(t22p.pseudobulk_rows(joint, spec, bts, masks, col))
    results = t22.finalize_panel(pd.DataFrame(rows), spec, ["label_column", "group", "variant"])
    results["status"] = "tested"
    pb = pd.DataFrame(pb_rows)
    pb["p_value_note"] = "effect size only (no test)"
    wide = pb.pivot_table(index=["label_column", "group", "variant", "item"], columns="plate_scope",
                          values="pseudobulk_log2fc").add_prefix("pseudobulk_log2fc_").reset_index()
    results = results.merge(wide, on=["label_column", "group", "variant", "item"], how="left")
    results = pd.concat([results, pd.DataFrame(na_rows)], ignore_index=True)
    results = lib.set_note(lib.ordered(results, KEY_COLUMNS))

    # checks on the results
    tested = results.loc[results["status"] == "tested"]
    det_same = tested.loc[tested["item_type"] == "protein"].groupby(["label_column", "group", "item"])[
        "det_lr"].nunique(dropna=True).max()
    checks.add("detection part identical across variants (variant-independent)", f"max distinct {det_same}",
               det_same <= 1)
    n_rows = results.groupby(["label_column", "group", "variant"]).size()
    checks.add("28 panel rows per label column x group x variant", f"{n_rows.min()}-{n_rows.max()}",
               n_rows.min() == n_rows.max() == 28)
    q_ok = bool((tested["q_primary"].dropna() >= tested.loc[tested["q_primary"].notna(), "p_primary"] - 1e-12).all())
    checks.add("BH q >= p for every tested row", f"{q_ok}", q_ok)
    ca = tested.query("item_type == 'protein' and testable == True")
    shift = ca.pivot_table(index=["label_column", "group", "item"], columns="variant", values="int_coef_ln")
    delta = float(t26.read_table(run, "normalization_variants").set_index("variant").loc["condition_align", "delta_ln"])
    dev = float(np.nanmax(np.abs(shift["condition_align"] - shift["none"] - delta)))
    checks.add("condition_align intensity coefficient = none + delta (dct_intake delta, ZeroK detected values shifted)",
               f"max |deviation| {dev:.2g} (delta {delta:.4f})", dev < 1e-6)
    sep = tested.query("variant == 'none' and item_type == 'protein' and detection_separation == True")
    checks.add("panel rows with detection separation (Firth estimates)",
               f"{len(sep)}: " + ", ".join(f"{lib.LABEL_SHORT[r.label_column]}/{r.group}/{r.item}"
                                           for r in sep.itertuples()), None)
    nt = results.query("variant == 'none' and match_status == 'not testable'")
    checks.add("not testable panel rows (variant none)",
               f"{len(nt)}: " + ", ".join(f"{lib.LABEL_SHORT[r.label_column]}/{r.group}/{r.item}"
                                          for r in nt.itertuples()), None)
    na = results.query("variant == 'none' and match_status == @lib.DOES_NOT_APPLY")
    checks.add("'does not apply' groups (label absent in a condition)", "; ".join(sorted(
        {f"{lib.LABEL_SHORT[r.label_column]}/{r.group}" for r in na.itertuples()})), None)

    run.table(results, "panel_results")
    run.table(lib.set_note(pb), "panel_pseudobulk")
    exp_cols = ["label_column", "group", "variant", "item", "item_type", "expected_direction", "observed_direction",
                "match_status", "match_reason", "q_primary", "n_ctrl", "n_zerok",
                "detection_difference_zerok_minus_ctrl", "det_logodds", "int_log2fc_adjusted", "score_coef",
                "pseudobulk_log2fc_pooled", "p_value_note"]
    run.table(results[[c for c in exp_cols if c in results]], "panel_expected_vs_observed")
    summ = (results.assign(category=[t22p.category(s, r) for s, r in zip(results["match_status"],
                                                                          results["match_reason"].fillna(""))])
            .groupby(["label_column", "group", "variant", "category"]).size().unstack(fill_value=0).reset_index())
    run.table(summ, "panel_match_summary")
    run.table(checks.frame(), "panel_step_checks")
    run.finish(parameters={"groups": list(lib.LABELS) + [lib.ALL_DCT], "variants": list(lib.VARIANTS),
                           "label_columns": list(lib.LABEL_COLUMNS), "depth_covariate": f"log {lib.DEPTH_COL}",
                           "min_detected_per_condition_intensity_part": t22.MIN_DET_PER_CONDITION,
                           "min_detected_total": t22.MIN_DET_TOTAL, "alpha_bh": t22.ALPHA,
                           "bh_block": "label_column x group x variant (28 panel rows)",
                           "logistic": "Firth-penalized, penalized LR test", "intensity": "OLS, Gaussian LR test",
                           "score_genes_random_state": {"cell_cycle_score": t2.SEED, "DCT type scores": 0},
                           "absent_labels": "does not apply rows", "ported_from": "zerok_response_core_panel.py",
                           "p_value_note": lib.P_NOTE, "runtime_s": round(time.time() - t_start, 1),
                           "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_panel_step_checks.csv.")


if __name__ == "__main__":
    main()
