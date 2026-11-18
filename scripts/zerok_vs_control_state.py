#!/usr/bin/env python
"""zerok_vs_control step 5: cell-state shifts between conditions on the combined search (zerok_response_core step 4 rerun).

Analysis plan,
step 5. Same procedure as zerok_response_core_state.py (zerok_response_core_lib and the zerok_response_core projection helper imported unchanged):

- DCT1 - DCT2 score difference (labels_core workbook score_genes and the mean-z check; with and without Slc12a3), per group:
  medians, OLS score ~ condition + log depth (BH across groups per label column x variant x measure), Mann-Whitney;
- cycling: Mki67 / Top2a detected, either, and the cycling-score rule (D3: above the 90th percentile of the Ctrl
  cells, per variant); ProLIF label share where the label exists;
- projected pseudotime (D5): mean dct_labels Ctrl DPT (`dpt_pseudotime` of the Ctrl cells, submitted pipeline, original
  root) of the 10 nearest Ctrl cells in the joint 20-PC embedding of each variant; Ctrl cells leave themselves out;
  the ZeroK cells' own DPT is not used;
- depth-matched versions (D4): greedy 1:1 nearest neighbour on ln n_detected within group and sort plate, caliper
  0.05, 100 repeats (the zerok_response_core seeds per label column, group and repeat; reused by step 6).
Groups: DCT1, DCT2, ProLIF, Unknown, all_DCT under both label columns, plus label-free all cells; groups with no
cells in a condition are 'does not apply' rows. All p-values are descriptive.

Inputs (read-only): results/h5ad/dct_labels_DCT_joint_labelled.h5ad; dct_labels canonical labels (dpt_pseudotime); zerok_core
panel and hash; zerok_vs_control step 3 per-cell scores.
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_state_per_cell.csv, zerok_vs_control_state_score_shift.csv,
zerok_vs_control_state_cycling.csv, zerok_vs_control_state_cycling_thresholds.csv, zerok_vs_control_state_pseudotime.csv,
zerok_vs_control_state_depth_matched.csv, zerok_vs_control_state_matching_balance.csv, zerok_vs_control_state_checks.csv.

Command (project root):  python scripts/zerok_vs_control_state.py
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
from scipy import stats

import scp_common as cc
import cluster_de_lib as t5
import zerok_response_core_state as t22s
import zerok_vs_control_lib as lib

SCRIPT = "zerok_vs_control_state.py"
t22 = lib.t22
SHIFT_MEASURES = t22s.SHIFT_MEASURES
FLAGS = t22s.FLAGS


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 5: cell-state shifts.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "state", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    joint = lib.load_joint(run, checks)
    obs = joint.obs
    zk = joint.is_zerok
    ld = joint.log_depth
    scores = lib.read_table(run, "per_cell_scores", dtype={"cell_id": str}).set_index("cell_id")
    same = scores.index.tolist() == obs.index.tolist()
    checks.add("step 3 per-cell scores in object cell order", f"{same}", same)

    # ---- per-cell state table
    per = obs[["cell_key", "condition", "plate", "dct_label", "dct_label_alternative", lib.DEPTH_COL]].copy()
    for gene in ("Mki67", "Top2a"):
        per[f"{gene}_detected"] = joint.detected[:, joint.var_index(t22.resolve_protein(joint, gene))]
    per["Mki67_or_Top2a_detected"] = per["Mki67_detected"] | per["Top2a_detected"]
    thr_rows = []
    for v in lib.VARIANTS:
        cs = scores[f"{v}__cell_cycle_score"].to_numpy(float)
        thr = float(np.quantile(cs[~zk], t22.CYCLING_QUANTILE))
        per[f"cycling_score_rule__{v}"] = cs > thr
        thr_rows.append({"variant": v, "quantile_of_control_cells": t22.CYCLING_QUANTILE, "threshold": thr,
                         "n_ctrl_above": int((cs[~zk] > thr).sum()), "n_zerok_above": int((cs[zk] > thr).sum())})
    thresholds = pd.DataFrame(thr_rows)
    frac_ctrl = (thresholds["n_ctrl_above"] / int((~zk).sum())).round(3).tolist()
    checks.add("cycling-score rule flags 10% of control cells by construction (per variant)", f"{frac_ctrl}",
               all(abs(f - 0.10) < 0.005 for f in frac_ctrl))
    labels = pd.read_csv(run.use_input(lib.LABELS_CSV, "dct_labels canonical labels (dpt_pseudotime)"),
                         dtype={"cell_id": str}).set_index("cell_id")
    keys_ok = bool((labels["cell_key"].reindex(obs.index).astype(str) == obs["cell_key"]).all())
    checks.add("dct_labels canonical rows matched on cell_id carry the same cell_key", f"{keys_ok}", keys_ok)
    dpt_all = labels["dpt_pseudotime"].reindex(obs.index).to_numpy(float)
    dpt = np.where(zk, np.nan, dpt_all)        # D5: Ctrl DPT only; the ZeroK cells' own DPT is not used
    n_fin = int(np.isfinite(dpt[~zk]).sum())
    checks.add("Ctrl cells with a finite dct_labels DPT", f"{n_fin} of {int((~zk).sum())}", n_fin > 0.95 * (~zk).sum(),
               "> 95%")
    checks.add("ZeroK cells: own dct_labels DPT present but not used (set to NaN for D5)",
               f"{int(np.isfinite(dpt_all[zk]).sum())} finite, used 0", not np.isfinite(dpt[zk]).any())
    per["ctrl_dpt_t27"] = dpt
    for v in lib.VARIANTS:
        proj, mdist = t22s.project_dpt(joint.pca[v], zk, dpt, t22.K_PSEUDOTIME)
        per[f"projected_dpt__{v}"] = proj
        per[f"projected_dpt_mean_distance__{v}"] = mdist
        ok = ~zk & np.isfinite(dpt)
        rho = stats.spearmanr(proj[ok], dpt[ok]).statistic
        ratio = float(np.median(mdist[zk]) / np.median(mdist[~zk]))
        checks.add(f"{v}: control leave-self-out projected DPT vs own DPT (Spearman)", f"{rho:.3f}", None)
        checks.add(f"{v}: median neighbour distance ZeroK / control", f"{ratio:.2f}", None)
    for m in SHIFT_MEASURES:
        for v in lib.VARIANTS:
            per[f"{m}__{v}"] = scores[f"{v}__{m}"].to_numpy(float)
    run.table(per.reset_index(), "state_per_cell")

    # ---- full-data comparisons
    shift_rows, pt_rows, cyc_rows, na_rows = [], [], [], []
    groups, absent = lib.all_groups(joint)
    for col, group, why in absent:
        na_rows.append(lib.absent_rows(joint, col, group, why))
    for col, group, m in groups:
        idx = np.flatnonzero(m)
        z, l = zk[idx], ld[idx]
        keys = {"label_column": col, "group": group}
        for v in lib.VARIANTS:
            for meas in SHIFT_MEASURES:
                shift_rows.append({**keys, "variant": v, "measure": meas,
                                   **t22s.compare_values(per[f"{meas}__{v}"].to_numpy()[idx], z, l)})
            pt_rows.append({**keys, "variant": v, "measure": "projected_dpt",
                            **t22s.compare_values(per[f"projected_dpt__{v}"].to_numpy()[idx], z, l),
                            "median_ctrl_own_dpt": float(np.nanmedian(dpt[idx][~z]))})
        flag_cols = [*FLAGS, *[f"cycling_score_rule__{v}" for v in lib.VARIANTS]]
        for f in flag_cols:
            x = per[f].to_numpy(bool)[idx]
            cyc_rows.append({**keys, "measure": f, **t22.proportion_compare(int(x[~z].sum()), int((~z).sum()),
                                                                           int(x[z].sum()), int(z.sum())),
                             "status": "compared"})
    for col in lib.LABEL_COLUMNS:
        x = obs[col].to_numpy() == "ProLIF"
        for group, m in ((lib.ALL_DCT, obs[col].isin(lib.CORE).to_numpy()), (lib.ALL_CELLS, np.ones(len(obs), bool))):
            kc, kz = int((x & m & ~zk).sum()), int((x & m & zk).sum())
            if kc == kz == 0:
                cyc_rows.append({**lib.absent_rows(joint, col, group, f"{lib.DOES_NOT_APPLY}: ProLIF is absent in both "
                                                   "conditions under the frozen dct_labels rule", mask=m),
                                 "measure": "ProLIF_label_share"})
                continue
            cyc_rows.append({"label_column": col, "group": group, "measure": "ProLIF_label_share",
                             **t22.proportion_compare(kc, int((m & ~zk).sum()), kz, int((m & zk).sum())),
                             "status": "compared"})
    shift = cc.add_bh(pd.DataFrame(shift_rows), ["label_column", "variant", "measure"], "p_score_lm", "q_lm")
    shift["status"] = "compared"
    shift = pd.concat([shift, pd.DataFrame([{**r, "measure": "all"} for r in na_rows])], ignore_index=True)
    shift["p_value_note"] = lib.P_NOTE
    pt = cc.add_bh(pd.DataFrame(pt_rows), ["label_column", "variant"], "p_score_lm", "q_lm")
    pt["status"] = "compared"
    pt = pd.concat([pt, pd.DataFrame([{**r, "measure": "projected_dpt"} for r in na_rows])], ignore_index=True)
    pt["p_value_note"] = lib.P_NOTE
    cyc = pd.DataFrame(cyc_rows + [{**r, "measure": "all"} for r in na_rows])
    cyc["p_value_note"] = lib.P_NOTE

    # ---- depth-matched versions (D4)
    dm_rows, bal_rows = [], []
    for col, group, m in groups:
        sets = t22.matched_sets(joint, col, group, m)
        idx_all = np.flatnonzero(m)
        smd_before = t22.smd(ld[idx_all][zk[idx_all]], ld[idx_all][~zk[idx_all]])
        smd_after = [t22.smd(ld[iz], ld[ic]) for ic, iz in sets]
        mean_diff = [float(ld[iz].mean() - ld[ic].mean()) if len(ic) else np.nan for ic, iz in sets]
        npairs = [len(ic) for ic, _ in sets]
        bal_rows.append({"label_column": col, "group": group, "n_ctrl": int((m & ~zk).sum()),
                         "n_zerok": int((m & zk).sum()), "median_pairs": float(np.median(npairs)),
                         "min_pairs": int(np.min(npairs)), "smd_ln_depth_before": smd_before,
                         "smd_ln_depth_after_median": float(np.nanmedian(smd_after)),
                         "smd_ln_depth_after_max_abs": float(np.nanmax(np.abs(smd_after))),
                         "mean_ln_depth_difference_after_max_abs": float(np.nanmax(np.abs(mean_diff))),
                         "n_repeats_with_min_pairs": int(sum(n >= t22.MIN_DET_PER_CONDITION for n in npairs)),
                         "status": "matched"})
        # repeats with fewer than 5 pairs are not compared (as the intensity part of D1)
        sets = [(ic, iz) for ic, iz in sets if len(ic) >= t22.MIN_DET_PER_CONDITION]
        if not sets:
            continue
        measures = ([(f"{meas}__{v}", "score_shift", v, meas) for v in lib.VARIANTS for meas in SHIFT_MEASURES]
                    + [(f"projected_dpt__{v}", "pseudotime", v, "projected_dpt") for v in lib.VARIANTS])
        for column, analysis, v, meas in measures:
            vals = per[column].to_numpy(float)
            full = float(np.median(vals[m & zk]) - np.median(vals[m & ~zk]))
            eff = [np.median(vals[iz]) - np.median(vals[ic]) for ic, iz in sets]
            pv = [stats.mannwhitneyu(vals[iz], vals[ic]).pvalue for ic, iz in sets]
            dm_rows.append({"analysis": analysis, "label_column": col, "group": group, "variant": v, "measure": meas,
                            "effect": "median difference ZeroK - Ctrl", **t22.summarize_repeats(eff, full, pv)})
        flag_cols = [*FLAGS, *[f"cycling_score_rule__{v}" for v in lib.VARIANTS]]
        for f in flag_cols:
            x = per[f].to_numpy(bool)
            full = float(x[m & zk].mean() - x[m & ~zk].mean())
            eff = [x[iz].mean() - x[ic].mean() for ic, iz in sets]
            pv = [stats.fisher_exact([[x[iz].sum(), len(iz) - x[iz].sum()], [x[ic].sum(), len(ic) - x[ic].sum()]])[1]
                  for ic, iz in sets]
            v = f.split("__")[1] if "__" in f else ""
            dm_rows.append({"analysis": "cycling", "label_column": col, "group": group, "variant": v,
                            "measure": f.split("__")[0], "effect": "fraction difference ZeroK - Ctrl",
                            **t22.summarize_repeats(eff, full, pv)})
    dm = pd.DataFrame(dm_rows)
    dm["status"] = "compared"
    dm = pd.concat([dm, pd.DataFrame([{**r, "analysis": "all"} for r in na_rows])], ignore_index=True)
    dm["p_value_note"] = lib.P_NOTE
    bal = pd.DataFrame(bal_rows)
    worst_mean = float(bal["mean_ln_depth_difference_after_max_abs"].max())
    checks.add("depth matching: max |mean ln-depth difference| after matching (all repeats and groups)",
               f"{worst_mean:.4f}", worst_mean <= t22.CALIPER, f"<= caliper {t22.CALIPER}")
    big = bal.loc[bal["median_pairs"] >= 20]
    worst = float(big["smd_ln_depth_after_max_abs"].max())
    checks.add("depth matching: max |SMD of ln depth| after matching, groups with >= 20 pairs", f"{worst:.3f}",
               worst < 0.1, "< 0.1")
    small = bal.loc[bal["median_pairs"] < 20]
    checks.add("depth matching: groups with < 20 pairs (SMD unstable; reported, not checked)", "; ".join(
        f"{lib.LABEL_SHORT[r.label_column]}/{r.group} {r.median_pairs:.0f} pairs, SMD before {r.smd_ln_depth_before:.2f}"
        for r in small.itertuples()) or "none", None)
    checks.add("depth matching: groups and median pairs", "; ".join(
        f"{lib.LABEL_SHORT[r.label_column]}/{r.group} {r.median_pairs:.0f}" for r in bal.itertuples()), None)
    bal = pd.concat([bal, pd.DataFrame(na_rows).drop(columns="p_value_note")], ignore_index=True)
    checks.add("'does not apply' groups (label absent in a condition)", "; ".join(
        f"{lib.LABEL_SHORT[c]}/{g}" for c, g, _ in absent), None)

    run.table(shift, "state_score_shift")
    run.table(cyc, "state_cycling")
    run.table(thresholds, "state_cycling_thresholds")
    run.table(pt, "state_pseudotime")
    run.table(dm, "state_depth_matched")
    run.table(bal, "state_matching_balance")
    run.table(checks.frame(), "state_checks")
    run.finish(parameters={"k_pseudotime": t22.K_PSEUDOTIME, "n_pcs": t22.N_PCS,
                           "dpt_source": "dct_labels canonical dpt_pseudotime of the Ctrl cells (submitted pipeline, "
                                         "original root); ZeroK DPT not used",
                           "cycling_quantile_of_control": t22.CYCLING_QUANTILE, "caliper_ln": t22.CALIPER,
                           "n_repeats": t22.N_REPEATS, "matching": "greedy 1:1 nearest neighbour within plate",
                           "depth": f"ln {lib.DEPTH_COL} (joint proteins)", "ported_from": "zerok_response_core_state.py",
                           "p_value_note": lib.P_NOTE, "runtime_s": round(time.time() - t_start, 1),
                           "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_state_checks.csv.")


if __name__ == "__main__":
    main()
