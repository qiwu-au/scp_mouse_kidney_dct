#!/usr/bin/env python
"""zerok_vs_control step 6: robustness of the zero-K vs control results on the combined search (zerok_response_core step 5 rerun).

Analysis plan,
step 6 (first part). Same procedure as zerok_response_core_robustness.py (its helpers and zerok_response_core_lib imported unchanged):

(a) Depth-matched subsampling (D4): the step 3 panel statistics on depth-matched cells (greedy 1:1 nearest neighbour
    on ln n_detected within label group and sort plate, caliper 0.05; 100 repeats; the same matched sets as step 5).
(b) Technical split: ZeroK vs Ctrl effects from P1 cells alone and from P2 cells alone (sort plates of the pooled
    cells; technical check, never replication): panel and proteome-wide (effect sizes only, D8).
(c) Cell downsampling: 50% and 25% of the cells of each condition within each group, 100 repeats each.
(d) Label-free version: all cells, and Slc12a3-detected cells only (panel); proteome-wide for all cells.
(e) Leave-Slc12a3-out scores: ALDO scores and the DCT1 - DCT2 score difference with and without Slc12a3.
(f) Proteome-wide effect sizes (D8) under depth matching and downsampling: Spearman with the full-data effects.
Groups: DCT1, DCT2, ProLIF, Unknown, all_DCT under dct_label and dct_label_alternative where the label has cells in
both conditions, plus label-free all cells; every variant. All p-values are descriptive.

Inputs (read-only): results/h5ad/dct_labels_DCT_joint_labelled.h5ad; dct_labels canonical labels; zerok_core panel and hash; zerok_vs_control
step 3 (per-cell scores, panel results, panel pseudobulk), step 4 (proteome tables) and step 5 (score shift).
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_robustness_panel_repeats.csv,
zerok_vs_control_robustness_plate_split_panel.csv, zerok_vs_control_robustness_plate_split_summary.csv,
zerok_vs_control_robustness_label_free_panel.csv, zerok_vs_control_robustness_label_free_proteome.csv,
zerok_vs_control_robustness_leave_slc12a3_out.csv, zerok_vs_control_robustness_proteome_effects.csv,
zerok_vs_control_robustness_summary.csv, zerok_vs_control_robustness_checks.csv.

Command (project root):  python scripts/zerok_vs_control_robustness.py
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
import zerok_response_core_robustness as t22r
import zerok_vs_control_lib as lib

SCRIPT = "zerok_vs_control_robustness.py"
t22 = lib.t22
KEY = t22r.KEY
EFFECTS = t22r.EFFECTS
PANEL_EFFECTS = t22r.PANEL_EFFECTS


def downsample_sets(joint, col, group, m, n_repeats):
    return {f"downsample_{int(frac * 100)}pct": [
        t22r.downsample(joint, m, frac, t22.repeat_rng(5, fi, t22.LABEL_COLUMN_INDEX[col],
                                                       t22.GROUP_ORDER.index(group), r)) for r in range(n_repeats)]
        for fi, frac in enumerate(t22.DOWNSAMPLE_FRACTIONS)}


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 6: robustness.")
    parser.add_argument("--repeats", type=int, default=t22.N_REPEATS, help="Repeats for (a), (c) and (f).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "robustness", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    joint = lib.load_joint(run, checks)
    obs = joint.obs
    zk = joint.is_zerok
    plate = obs["plate"].to_numpy()
    spec = t22.panel_spec(joint, t22.read_panel(run))
    panel_names = {name for _, name, _ in spec.proteins if name}
    per_cell = lib.read_table(run, "per_cell_scores", dtype={"cell_id": str}).set_index("cell_id")
    scores = {v: per_cell.filter(like=f"{v}__").rename(columns=lambda c: c.split("__", 1)[1]) for v in lib.VARIANTS}
    full_panel = lib.read_table(run, "panel_results").query("status == 'tested'")
    bts = {v: t22.back_transform(joint, v) for v in lib.VARIANTS}
    groups, absent = lib.all_groups(joint)
    checks.add("groups (label column / group) with cells in both conditions", "; ".join(
        f"{lib.LABEL_SHORT[c]}/{g}" for c, g, _ in groups), None)
    checks.add("'does not apply' groups (label absent in a condition): not repeated", "; ".join(
        f"{lib.LABEL_SHORT[c]}/{g}" for c, g, _ in absent), None)

    # ---- (d) label-free panel and proteome
    lf_rows = []
    for group, m in t22.label_free_masks(joint).items():
        lf_rows.extend(t22.panel_rows(joint, spec, np.flatnonzero(m), scores,
                                      keys={"label_column": "label_free", "group": group}))
    lf_panel = lib.set_note(t22.finalize_panel(pd.DataFrame(lf_rows), spec, ["label_column", "group", "variant"]))
    run.table(lf_panel, "robustness_label_free_panel")
    n_slc = int(t22.label_free_masks(joint)[lib.SLC12A3_DETECTED].sum())
    checks.add("label-free groups (all cells; Slc12a3-detected cells)", f"{len(obs)}; {n_slc}", None)
    checks.add("label-free panel rows (2 groups x 3 variants x 28 items)", f"{len(lf_panel)}", len(lf_panel) == 168,
               "168")
    print("  label-free proteome (all cells)", flush=True)
    lf_prot = lib.set_note(t22.proteome_group(joint, np.arange(len(obs)), bts,
                                              {"label_column": "label_free", "group": lib.ALL_CELLS}, panel_names))
    run.table(lf_prot, "robustness_label_free_proteome")
    full_ref = pd.concat([full_panel, lf_panel], ignore_index=True)

    # ---- (a) depth-matched and (c) downsampled panel repeats
    reps = []
    for col, group, m in groups:
        t0 = time.time()
        sets = t22.matched_sets(joint, col, group, m, args.repeats)
        reps.append(t22r.repeat_panel(joint, spec, scores, col, group, sets, "depth_matched"))
        for proc, ds in downsample_sets(joint, col, group, m, args.repeats).items():
            reps.append(t22r.repeat_panel(joint, spec, scores, col, group, ds, proc))
        print(f"  panel repeats {col} / {group}: {time.time() - t0:.0f} s", flush=True)
    rep = pd.concat(reps, ignore_index=True)
    rep_summary = lib.set_note(t22r.summarize_panel(rep, full_ref))
    checks.add("panel repeats per procedure x group x variant x item", f"{rep_summary['n_repeats'].min()}-"
               f"{rep_summary['n_repeats'].max()}", rep_summary["n_repeats"].eq(args.repeats).all(),
               f"{args.repeats}")
    n_cells_ds = rep.query("procedure != 'depth_matched'").groupby(["procedure", "label_column", "group"])[
        ["n_ctrl", "n_zerok"]].median()
    checks.add("downsampled groups: median cells per condition (information)", "; ".join(
        f"{p}/{lib.LABEL_SHORT[c]}/{g} {int(r.n_ctrl)}/{int(r.n_zerok)}" for (p, c, g), r in n_cells_ds.iterrows()), None)
    run.table(rep_summary, "robustness_panel_repeats")

    # ---- (b) plate split: panel
    split_rows = []
    for col, group, m in groups:
        per_plate = {}
        for p in lib.PLATES:
            rows = t22.panel_rows(joint, spec, np.flatnonzero(m & (plate == p)), scores,
                                  keys={"label_column": col, "group": group, "plate": p})
            per_plate[p] = t22.finalize_panel(pd.DataFrame(rows), spec,
                                              ["label_column", "group", "plate", "variant"]).set_index(KEY)
        a, b = per_plate["P1"], per_plate["P2"]
        for key in a.index:
            row = dict(zip(KEY, key))
            row["item_type"] = a.at[key, "item_type"]
            row["n_ctrl_P1"], row["n_zerok_P1"] = a.at[key, "n_ctrl"], a.at[key, "n_zerok"]
            row["n_ctrl_P2"], row["n_zerok_P2"] = b.at[key, "n_ctrl"], b.at[key, "n_zerok"]
            for e in PANEL_EFFECTS:
                if e in a:
                    row[f"{e}_P1"], row[f"{e}_P2"] = a.at[key, e], b.at[key, e]
            row["dominant_sign_P1"], row["dominant_sign_P2"] = a.at[key, "dominant_sign"], b.at[key, "dominant_sign"]
            row["same_dominant_sign"] = bool(row["dominant_sign_P1"] == row["dominant_sign_P2"]
                                             and np.isfinite(row["dominant_sign_P1"]))
            row["observed_direction_P1"] = a.at[key, "observed_direction"]
            row["observed_direction_P2"] = b.at[key, "observed_direction"]
            split_rows.append(row)
    pb = lib.read_table(run, "panel_pseudobulk")
    pbw = pb.pivot_table(index=["label_column", "group", "variant", "item"], columns="plate_scope",
                         values="pseudobulk_log2fc").reset_index().rename(
        columns={"P1": "pseudobulk_log2fc_P1", "P2": "pseudobulk_log2fc_P2"}).drop(columns="pooled")
    split = pd.DataFrame(split_rows).merge(pbw, on=KEY, how="left")
    split["note"] = "P1/P2 are sort plates of one pool per condition: a technical check, not replication"
    run.table(split, "robustness_plate_split_panel")

    # ---- (b) plate split summaries (panel and proteome) and (f) proteome effect sizes
    sum_rows = []
    for (col, group, v), g in split.groupby(["label_column", "group", "variant"], sort=False):
        prot = g.query("item_type == 'protein'")
        for e in ("det_logodds", "int_log2fc_adjusted", "log2fc_detected", "detection_difference_zerok_minus_ctrl",
                  "pseudobulk_log2fc"):
            rho, n, agree = t22r.spearman(prot[f"{e}_P1"], prot[f"{e}_P2"])
            sum_rows.append({"scope": "panel proteins", "label_column": col, "group": group, "variant": v,
                             "effect": e, "n": n, "spearman_P1_vs_P2": rho, "sign_agreement": agree})
    proteome = {}
    for col in lib.LABEL_COLUMNS:
        for v in lib.VARIANTS:
            proteome[(col, v)] = lib.read_table(run, f"proteome_{lib.LABEL_SHORT[col]}_{v}")
    for v in lib.VARIANTS:
        proteome[("label_free", v)] = lf_prot.query("variant == @v")
    eff_rows = []
    for (col, v), tab in proteome.items():
        for group, g in tab.groupby("group", sort=False):
            m = (np.ones(len(obs), bool) if col == "label_free" else t22.group_masks(obs, col)[group])
            cols_idx = np.array([joint.var_index(p) for p in g["protein"]])
            e1 = t22.effect_sizes(joint, v, np.flatnonzero(m & ~zk & (plate == "P1")),
                                  np.flatnonzero(m & zk & (plate == "P1")), cols_idx, bts[v])
            e2 = t22.effect_sizes(joint, v, np.flatnonzero(m & ~zk & (plate == "P2")),
                                  np.flatnonzero(m & zk & (plate == "P2")), cols_idx, bts[v])
            for e in EFFECTS:
                rho, n, agree = t22r.spearman(e1[e], e2[e])
                sum_rows.append({"scope": "proteome-wide", "label_column": col, "group": group, "variant": v,
                                 "effect": e, "n": n, "spearman_P1_vs_P2": rho, "sign_agreement": agree})
            ef = t22.effect_sizes(joint, v, np.flatnonzero(m & ~zk), np.flatnonzero(m & zk), cols_idx, bts[v])
            procs = {"depth_matched": t22.matched_sets(joint, col, group, m, args.repeats),
                     **downsample_sets(joint, col, group, m, args.repeats)}
            for proc, sets in procs.items():
                res = {e: [] for e in EFFECTS}
                for ic, iz in sets:
                    er = t22.effect_sizes(joint, v, ic, iz, cols_idx, bts[v])
                    for e in EFFECTS:
                        res[e].append(t22r.spearman(ef[e], er[e]))
                for e in EFFECTS:
                    arr = np.array(res[e], float)
                    eff_rows.append({"procedure": proc, "label_column": col, "group": group, "variant": v,
                                     "effect": e, "n_proteins": len(cols_idx), "n_repeats": len(sets),
                                     "median_n_proteins_with_effect": float(np.nanmedian(arr[:, 1])),
                                     "spearman_vs_full_median": float(np.nanmedian(arr[:, 0])),
                                     "spearman_vs_full_q025": float(np.nanquantile(arr[:, 0], 0.025)),
                                     "sign_agreement_vs_full_median": float(np.nanmedian(arr[:, 2])),
                                     "sign_agreement_vs_full_q025": float(np.nanquantile(arr[:, 2], 0.025))})
        print(f"  proteome effect sizes {col} / {v} done", flush=True)
    split_summary = pd.DataFrame(sum_rows)
    split_summary["note"] = "P1/P2 are sort plates of one pool per condition: a technical check, not replication"
    run.table(split_summary, "robustness_plate_split_summary")
    effects = pd.DataFrame(eff_rows)
    effects["note"] = "effect sizes only (D8); no per-repeat test"
    checks.add("proteome effect-size repeats per procedure", f"{effects['n_repeats'].min()}-{effects['n_repeats'].max()}",
               effects["n_repeats"].eq(args.repeats).all(), f"{args.repeats}")
    run.table(effects, "robustness_proteome_effects")

    # ---- (e) leave-Slc12a3-out scores
    lso_rows = []
    pairs = [("ALDO_activity", "ALDO_activity (base)", "ALDO_activity (no_Slc12a3)"),
             ("ALDO_signed", "ALDO_signed (base)", "ALDO_signed (no_Slc12a3)")]
    ref = full_ref.set_index(KEY)
    for (col, group, v), _ in full_ref.groupby(["label_column", "group", "variant"], sort=False):
        for fam, base, noslc in pairs:
            a, b = ref.loc[(col, group, v, base)], ref.loc[(col, group, v, noslc)]
            lso_rows.append({"label_column": col, "group": group, "variant": v, "score": fam,
                             "coef_base": a["score_coef"], "q_base": a["q_primary"],
                             "observed_base": a["observed_direction"], "coef_no_Slc12a3": b["score_coef"],
                             "q_no_Slc12a3": b["q_primary"], "observed_no_Slc12a3": b["observed_direction"],
                             "same_sign": bool(np.sign(a["score_coef"]) == np.sign(b["score_coef"])),
                             "same_observed_direction": a["observed_direction"] == b["observed_direction"],
                             "source": "panel BH (step 3 / label-free here)"})
    shift = lib.read_table(run, "state_score_shift").query("status == 'compared'")
    sref = shift.set_index(["label_column", "group", "variant", "measure"])
    for (col, group, v), _ in shift.groupby(["label_column", "group", "variant"], sort=False):
        for meas in ("DCT1_minus_DCT2_score", "DCT1_minus_DCT2_meanz"):
            a, b = sref.loc[(col, group, v, meas)], sref.loc[(col, group, v, f"{meas}__no_Slc12a3")]
            lso_rows.append({"label_column": col, "group": group, "variant": v, "score": meas,
                             "coef_base": a["score_coef"], "q_base": a["q_lm"],
                             "observed_base": "", "coef_no_Slc12a3": b["score_coef"], "q_no_Slc12a3": b["q_lm"],
                             "observed_no_Slc12a3": "",
                             "same_sign": bool(np.sign(a["score_coef"]) == np.sign(b["score_coef"])),
                             "same_observed_direction": np.nan, "source": "step 5 (BH across groups)"})
    lso = pd.DataFrame(lso_rows)
    lso["p_value_note"] = lib.P_NOTE
    run.table(lso, "robustness_leave_slc12a3_out")

    # ---- summary
    srows = []
    for (proc, col, group, v), g in rep_summary.groupby(["procedure", "label_column", "group", "variant"], sort=False):
        sig = g.loc[g["full_q_primary"] < lib.ALPHA]
        scored = g.loc[g["expected_direction"].isin([t22.UP, t22.DOWN, t22.NO_CHANGE])]
        srows.append({"check": f"panel {proc}", "label_column": col, "group": group, "variant": v,
                      "n_items_significant_full": len(sig),
                      "n_kept_sign_and_q_in_95pct": int((sig["fraction_same_sign_and_q_below_0.05"] >= 0.95).sum()),
                      "n_kept_sign_and_q_in_50pct": int((sig["fraction_same_sign_and_q_below_0.05"] >= 0.50).sum()),
                      "n_scored_items": len(scored),
                      "n_scored_match_status_kept_in_80pct": int((scored["fraction_same_match_status"] >= 0.8).sum()),
                      "median_n_ctrl": float(g["median_n_ctrl"].median()),
                      "median_n_zerok": float(g["median_n_zerok"].median())})
    for r in split_summary.itertuples():
        srows.append({"check": f"plate split ({r.scope})", "label_column": r.label_column, "group": r.group,
                      "variant": r.variant, "effect": r.effect, "n": r.n, "spearman": r.spearman_P1_vs_P2,
                      "sign_agreement": r.sign_agreement})
    for r in effects.itertuples():
        srows.append({"check": f"proteome effect sizes {r.procedure}", "label_column": r.label_column,
                      "group": r.group, "variant": r.variant, "effect": r.effect, "n": r.n_proteins,
                      "spearman": r.spearman_vs_full_median, "sign_agreement": r.sign_agreement_vs_full_median})
    for c, g, why in absent:
        srows.append({"check": "all", "label_column": c, "group": g, "status": why})
    run.table(pd.DataFrame(srows), "robustness_summary")

    run.table(checks.frame(), "robustness_checks")
    run.finish(parameters={"repeats": args.repeats, "downsample_fractions": list(t22.DOWNSAMPLE_FRACTIONS),
                           "caliper_ln": t22.CALIPER, "seeds": "numpy default_rng([42, procedure, label column, "
                           "group, repeat]) (zerok_response_core seeds)", "depth": f"ln {lib.DEPTH_COL} (joint proteins)",
                           "ported_from": "zerok_response_core_robustness.py", "p_value_note": lib.P_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_robustness_checks.csv.")


if __name__ == "__main__":
    main()
