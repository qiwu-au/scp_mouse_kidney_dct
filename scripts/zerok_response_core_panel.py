#!/usr/bin/env python
"""zerok_response_core step 2: the pre-specified panel, zero-K vs control, per label and for all DCT cells.

Analysis plan, step 2.
The frozen zerok_core panel (SHA256 checked first) is tested under the scoring rule D1 agreed in the zerok_response_core plan
(28 Oct 2026) before any comparison between conditions was run:

- per protein: detection fraction per condition (difference ZeroK - Ctrl with Newcombe 95% CI, Fisher p);
  ln-intensity among detected cells (medians, mean-based log2 fold change, Mann-Whitney p); the two-part
  model (Firth logistic on detection + OLS on ln-intensity among detected, both with log n_detected_matched;
  LR tests summed: 2 df, or 1 df without the intensity part) = primary test;
- per score (ALDO with and without Slc12a3, ALDO machinery, cell-cycle score): OLS score ~ condition +
  log n_detected_matched (primary) and Mann-Whitney;
- pseudobulk log2 fold change (D6) per sort plate and pooled;
- BH within the panel per label column x group x variant; observed direction beside the frozen expected
  direction; match / mismatch / not testable (D1).
Groups: DCT1, DCT2, ProLIF, Unknown and all_DCT (DCT1 + DCT2 + ProLIF), under dct_label and
dct_label_alternative; every variant (none = baseline, cell_median, condition_align). All p-values are
descriptive (one pool per condition; cells are not animal-level replicates).

Per-cell scores (D3) for every variant are written here and reused by steps 4-6.

Inputs (read-only): results/h5ad/zerok_core_DCT_joint_labelled.h5ad; zerok_core panel, panel hash and joint labels;
data/DCT/DCT_markers_V1.xlsx (labels_core scoring lists).
Outputs (results/tables/zerok_response_core/): zerok_response_core_per_cell_scores.csv, zerok_response_core_panel_results.csv,
zerok_response_core_panel_pseudobulk.csv, zerok_response_core_panel_expected_vs_observed.csv, zerok_response_core_panel_match_summary.csv,
zerok_response_core_score_lists.csv, zerok_response_core_panel_step_checks.csv.

Command (project root):  python scripts/zerok_response_core_panel.py
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
import statsmodels.api as sm
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import labels_core_lib as t18
import zerok_response_core_lib as lib

ensure_scanpy_compat()
SCRIPT = "zerok_response_core_panel.py"
PLATE_SCOPES = ("P1", "P2", "pooled")
KEY_COLUMNS = ["label_column", "group", "variant", "item", "item_type", "expected_direction", "observed_direction",
               "match_status", "match_reason", "p_primary", "q_primary", "n_ctrl", "n_zerok",
               "detection_fraction_ctrl", "detection_fraction_zerok", "detection_difference_zerok_minus_ctrl",
               "detection_difference_ci_low", "detection_difference_ci_high", "detection_fisher_p",
               "det_logodds", "det_p", "log2fc_detected", "int_log2fc_adjusted", "int_p", "wilcoxon_p",
               "score_coef", "median_difference", "pseudobulk_log2fc_pooled", "pseudobulk_log2fc_P1",
               "pseudobulk_log2fc_P2"]


def model_checks(joint: lib.Joint, checks: t5.Checks) -> None:
    """Firth and Gaussian-LR implementations against statsmodels on a protein without separation."""
    col = joint.var_index(lib.resolve_protein(joint, "Stk39"))
    z = joint.is_zerok
    ld = joint.log_depth
    det = joint.detected[:, col]
    X = lib.design(z, ld)
    ml = sm.Logit(det.astype(float), X).fit(disp=0)
    ml_red = sm.Logit(det.astype(float), X[:, [0, 2]]).fit(disp=0)
    b, _, _, conv = lib.firth_logit(X, det.astype(float))
    d = abs(b[1] - ml.params[1])
    checks.add("Firth logistic vs statsmodels ML, condition coefficient (Stk39, all cells, no separation)",
               f"Firth {b[1]:.4f}, ML {ml.params[1]:.4f}, |diff| {d:.4f}, converged {conv}", d < 0.02 and conv,
               "|diff| < 0.02")
    part = lib.detection_part(det, z, ld)
    lr_ml = 2 * (ml.llf - ml_red.llf)
    checks.add("Firth penalized LR vs ML LR (same protein)", f"{part['det_lr']:.3f} vs {lr_ml:.3f}",
               abs(part["det_lr"] - lr_ml) < max(0.5, 0.05 * lr_ml), "within 5% or 0.5")
    y = joint.mats["none"][det, col]
    full = sm.OLS(y, X[det]).fit()
    red = sm.OLS(y, X[det][:, [0, 2]]).fit()
    lr_sm = full.compare_lr_test(red)[0]
    ip = lib.intensity_part(y, z[det], ld[det])
    checks.add("Gaussian LR and OLS coefficient vs statsmodels (same protein, variant none)",
               f"LR {ip['int_lr']:.4f} vs {lr_sm:.4f}; coef {ip['int_coef_ln']:.5f} vs {full.params[1]:.5f}",
               abs(ip["int_lr"] - lr_sm) < 1e-6 and abs(ip["int_coef_ln"] - full.params[1]) < 1e-8)


def category(status: str, reason: str) -> str:
    """Summary category: match status, with the mismatch reason in short."""
    if status != "mismatch":
        return status
    if reason.startswith("no BH-significant"):
        return "mismatch: no significant difference"
    if reason.startswith("mixed"):
        return "mismatch: mixed parts"
    if reason.startswith("BH-significant in the opposite"):
        return "mismatch: opposite direction"
    return "mismatch: significant difference (no change expected)"


def pseudobulk_rows(joint: lib.Joint, spec: lib.PanelSpec, bts: dict, masks: dict, label_column: str) -> list[dict]:
    plate = joint.obs["plate"].to_numpy()
    zk = joint.is_zerok
    rows = []
    for group, m in masks.items():
        for scope in PLATE_SCOPES:
            s = m & (np.ones(len(m), bool) if scope == "pooled" else plate == scope)
            ic, iz = np.flatnonzero(s & ~zk), np.flatnonzero(s & zk)
            for v in lib.VARIANTS:
                for item, name, col in spec.proteins:
                    if col < 0 or len(ic) == 0 or len(iz) == 0:
                        continue
                    mc, mz = bts[v][ic, col].mean(), bts[v][iz, col].mean()
                    rows.append({"label_column": label_column, "group": group, "variant": v, "item": item,
                                 "plate_scope": scope, "n_ctrl": len(ic), "n_zerok": len(iz),
                                 "det_ctrl": int(joint.detected[ic, col].sum()),
                                 "det_zerok": int(joint.detected[iz, col].sum()),
                                 "mean_intensity_ctrl": mc, "mean_intensity_zerok": mz,
                                 "pseudobulk_log2fc": float(np.log2(mz / mc)) if mc > 0 and mz > 0 else np.nan})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_response_core step 2: pre-specified panel.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "panel", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_panel_hash(run, checks)
    joint = lib.load_joint(run, checks)
    panel = lib.read_panel(run)
    spec = lib.panel_spec(joint, panel)

    resolved = [(item, name) for item, name, _ in spec.proteins]
    checks.add("panel proteins found in the joint object", f"{sum(1 for _, n in resolved if n)} of {len(resolved)}",
               all(n for _, n in resolved), "22 of 22")
    renamed = [f"{i}->{n}" for i, n in resolved if n and n != i]
    checks.add("panel proteins whose joint var name differs from the gene", ", ".join(renamed) or "none", None)
    members = lib.score_members_in_joint(joint)
    recorded = panel.loc[panel["item_type"] == "score"].set_index("item")["n_in_joint"].astype(float).astype(int)
    same = all(members[i] == recorded[i] for i in recorded.index)
    checks.add("score members in the joint object equal the zerok_core panel (n_in_joint)",
               "; ".join(f"{i} {members[i]}" for i in recorded.index), same)
    checks.add("panel rows tested (proteins + scores)", f"{len(spec.item_order)}", len(spec.item_order) == 28, "28")
    model_checks(joint, checks)

    # ---- per-cell scores (D3)
    workbook = t18.read_workbook(run)
    type_lists = t18.present_lists(workbook, joint.var_names)
    list_rows = [{"list": f"{t} (DCT V1 workbook)", "n_in_workbook": len(workbook[t]), "n_in_joint": len(g),
                  "present": ";".join(g)} for t, g in type_lists.items()]
    for item, (pos, neg) in lib.t21.SCORE_LISTS.items():
        var = set(joint.var_names)
        list_rows.append({"list": item, "n_in_workbook": len(dict.fromkeys(pos + neg)),
                          "n_in_joint": members[item],
                          "present": ";".join(g for g in dict.fromkeys(pos + neg) if g in var)})
    run.table(pd.DataFrame(list_rows), "score_lists")
    checks.add("DCT V1 workbook types scored", "; ".join(f"{t} {len(g)}" for t, g in type_lists.items()),
               {"DCT1", "DCT2"} <= set(type_lists), "DCT1 and DCT2 present")
    scores = {}
    for v in lib.VARIANTS:
        print(f"  scoring variant {v}", flush=True)
        scores[v] = lib.score_cells(joint, v, type_lists)
    base_cols = ["condition", "plate", "dct_label", "dct_label_alternative", lib.DEPTH_COL]
    per_cell = joint.obs[base_cols].copy()
    for v in lib.VARIANTS:
        per_cell = per_cell.join(scores[v].add_prefix(f"{v}__"))
    finite = bool(np.isfinite(per_cell.filter(like="__").to_numpy(float)).all())
    checks.add("per-cell scores finite in every variant", f"{finite}", finite)
    run.table(per_cell.reset_index(), "per_cell_scores")

    # ---- panel statistics
    rows, pb_rows = [], []
    bts = {v: lib.back_transform(joint, v) for v in lib.VARIANTS}
    for col in lib.LABEL_COLUMNS:
        masks = lib.group_masks(joint.obs, col)
        for group, m in masks.items():
            print(f"  panel: {col} / {group}", flush=True)
            rows.extend(lib.panel_rows(joint, spec, np.flatnonzero(m), scores,
                                       keys={"label_column": col, "group": group}))
        pb_rows.extend(pseudobulk_rows(joint, spec, bts, masks, col))
    results = lib.finalize_panel(pd.DataFrame(rows), spec, ["label_column", "group", "variant"])
    pb = pd.DataFrame(pb_rows)
    pb["p_value_note"] = "effect size only (no test)"
    wide = pb.pivot_table(index=["label_column", "group", "variant", "item"], columns="plate_scope",
                          values="pseudobulk_log2fc").add_prefix("pseudobulk_log2fc_").reset_index()
    results = results.merge(wide, on=["label_column", "group", "variant", "item"], how="left")
    results = lib.ordered(results, KEY_COLUMNS)

    # checks on the results
    det_same = results.loc[results["item_type"] == "protein"].groupby(["label_column", "group", "item"])[
        "det_lr"].nunique(dropna=True).max()
    checks.add("detection part identical across variants (variant-independent)", f"max distinct {det_same}",
               det_same <= 1)
    n_rows = results.groupby(["label_column", "group", "variant"]).size()
    checks.add("28 panel rows per label column x group x variant", f"{n_rows.min()}-{n_rows.max()}",
               n_rows.min() == n_rows.max() == 28)
    q_ok = bool((results["q_primary"].dropna() >= results.loc[results["q_primary"].notna(), "p_primary"] - 1e-12).all())
    checks.add("BH q >= p for every tested row", f"{q_ok}", q_ok)
    ca = results.query("item_type == 'protein' and testable == True")
    shift = ca.pivot_table(index=["label_column", "group", "item"], columns="variant", values="int_coef_ln")
    delta = float(lib.t21.read_table(run, "normalization_variants").set_index("variant").loc[
        "condition_align", "delta_ln"])
    dev = float(np.nanmax(np.abs(shift["condition_align"] - shift["none"] - delta)))
    checks.add("condition_align intensity coefficient = none + delta (zerok_core delta, ZeroK detected values shifted)",
               f"max |deviation| {dev:.2g} (delta {delta:.4f})", dev < 1e-6)
    sep = results.query("variant == 'none' and item_type == 'protein' and detection_separation == True")
    checks.add("panel rows with detection separation (Firth estimates)",
               f"{len(sep)}: " + ", ".join(f"{lib.LABEL_SHORT[r.label_column]}/{r.group}/{r.item}"
                                           for r in sep.itertuples()), None)
    nt = results.query("variant == 'none' and match_status == 'not testable'")
    checks.add("not testable panel rows (variant none)",
               f"{len(nt)}: " + ", ".join(f"{lib.LABEL_SHORT[r.label_column]}/{r.group}/{r.item}"
                                          for r in nt.itertuples()), None)

    run.table(results, "panel_results")
    run.table(pb, "panel_pseudobulk")
    exp_cols = ["label_column", "group", "variant", "item", "item_type", "expected_direction", "observed_direction",
                "match_status", "match_reason", "q_primary", "n_ctrl", "n_zerok",
                "detection_difference_zerok_minus_ctrl", "det_logodds", "int_log2fc_adjusted", "score_coef",
                "pseudobulk_log2fc_pooled", "p_value_note"]
    run.table(results[[c for c in exp_cols if c in results]], "panel_expected_vs_observed")
    summ = (results.assign(category=[category(s, r) for s, r in zip(results["match_status"],
                                                                     results["match_reason"])])
            .groupby(["label_column", "group", "variant", "category"]).size().unstack(fill_value=0).reset_index())
    run.table(summ, "panel_match_summary")
    run.table(checks.frame(), "panel_step_checks")
    run.finish(parameters={"groups": list(lib.LABELS) + [lib.ALL_DCT], "variants": list(lib.VARIANTS),
                           "label_columns": list(lib.LABEL_COLUMNS), "depth_covariate": f"log {lib.DEPTH_COL}",
                           "min_detected_per_condition_intensity_part": lib.MIN_DET_PER_CONDITION,
                           "min_detected_total": lib.MIN_DET_TOTAL, "alpha_bh": lib.ALPHA,
                           "bh_block": "label_column x group x variant (28 panel rows)",
                           "logistic": "Firth-penalized, penalized LR test", "intensity": "OLS, Gaussian LR test",
                           "score_genes_random_state": {"cell_cycle_score": t2.SEED, "DCT type scores": 0},
                           "p_value_note": lib.P_NOTE, "runtime_s": round(time.time() - t_start, 1),
                           "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_response_core_panel_step_checks.csv.")


if __name__ == "__main__":
    main()
