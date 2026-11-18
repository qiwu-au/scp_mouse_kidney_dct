#!/usr/bin/env python
"""zerok_response_core step 4: cell-state shifts between zero-K and control, with depth-matched versions.

Analysis plan, step 4.

(a) Per-cell DCT1 - DCT2 score difference (labels_core workbook scores, labels_core_lib.score_types on the joint matrix
    per variant; deterministic mean-z version as a check; both also without Slc12a3) by condition and label.
(b) Cycling: Mki67 detected, Top2a detected, either; cycling-score rule (cell_cycle_score above the 90th
    percentile of the CONTROL cells, per variant; D3); ProLIF label share where the label exists.
(c) Pseudotime (D5): each ZeroK cell gets the mean labels_core control DPT (original root) of its 10 nearest control
    cells in the joint 20-PC embedding of the variant; control cells get the same average over their 10
    nearest other control cells (leave-self-out). Distributions by condition and label.
Every comparison: full data (Mann-Whitney and OLS with log n_detected_matched; proportions with Newcombe CI and
Fisher p) and depth-matched (D4: greedy 1:1 nearest neighbour on ln n_detected_matched within label group and
sort plate, caliper 0.05, 100 repeats; the same matched sets as step 5). All p-values are descriptive (one pool
per condition; cells are not animal-level replicates).

Inputs (read-only): results/h5ad/zerok_core_DCT_joint_labelled.h5ad; zerok_core panel, hash and joint labels;
results/tables/zerok_response_core/zerok_response_core_per_cell_scores.csv (step 2); labels_core canonical control labels (DPT).
Outputs (results/tables/zerok_response_core/): zerok_response_core_state_per_cell.csv, zerok_response_core_state_score_shift.csv,
zerok_response_core_state_cycling.csv, zerok_response_core_state_cycling_thresholds.csv, zerok_response_core_state_pseudotime.csv,
zerok_response_core_state_depth_matched.csv, zerok_response_core_state_matching_balance.csv, zerok_response_core_state_checks.csv.

Command (project root):  python scripts/zerok_response_core_state.py
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
from sklearn.neighbors import NearestNeighbors

import scp_common as cc
import cluster_de_lib as t5
import zerok_response_core_lib as lib

SCRIPT = "zerok_response_core_state.py"
SHIFT_MEASURES = ("DCT1_minus_DCT2_score", "DCT1_minus_DCT2_meanz", "DCT1_minus_DCT2_score__no_Slc12a3",
                  "DCT1_minus_DCT2_meanz__no_Slc12a3")
FLAGS = ("Mki67_detected", "Top2a_detected", "Mki67_or_Top2a_detected")


def all_groups(obs: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    out = [(col, g, m) for col in lib.LABEL_COLUMNS for g, m in lib.group_masks(obs, col).items()]
    return out + [("label_free", lib.ALL_CELLS, np.ones(len(obs), bool))]


def project_dpt(pca: np.ndarray, is_zerok: np.ndarray, dpt: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean control DPT of the k nearest control cells (control cells leave themselves out)."""
    ref = np.flatnonzero(~is_zerok & np.isfinite(dpt))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pca[ref])
    dist, ind = nn.kneighbors(pca)
    out, mdist = np.full(len(pca), np.nan), np.full(len(pca), np.nan)
    for i in range(len(pca)):
        keep = ref[ind[i]] != i
        sel = ind[i][keep][:k]
        out[i] = dpt[ref[sel]].mean()
        mdist[i] = dist[i][keep][:k].mean()
    return out, mdist


def compare_values(values, z, ld) -> dict:
    return lib.score_test(np.asarray(values, float), z, ld)


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_response_core step 4: cell-state shifts.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "state", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_panel_hash(run, checks)
    joint = lib.load_joint(run, checks)
    obs = joint.obs
    zk = joint.is_zerok
    ld = joint.log_depth
    scores = lib.read_table(run, "per_cell_scores", dtype={"cell_id": str}).set_index("cell_id")
    same = scores.index.tolist() == obs.index.tolist()
    checks.add("step 2 per-cell scores in object cell order", f"{same}", same)

    # ---- per-cell state table
    per = obs[["condition", "plate", "dct_label", "dct_label_alternative", lib.DEPTH_COL]].copy()
    for gene in ("Mki67", "Top2a"):
        per[f"{gene}_detected"] = joint.detected[:, joint.var_index(lib.resolve_protein(joint, gene))]
    per["Mki67_or_Top2a_detected"] = per["Mki67_detected"] | per["Top2a_detected"]
    thr_rows = []
    for v in lib.VARIANTS:
        cs = scores[f"{v}__cell_cycle_score"].to_numpy(float)
        thr = float(np.quantile(cs[~zk], lib.CYCLING_QUANTILE))
        per[f"cycling_score_rule__{v}"] = cs > thr
        thr_rows.append({"variant": v, "quantile_of_control_cells": lib.CYCLING_QUANTILE, "threshold": thr,
                         "n_ctrl_above": int((cs[~zk] > thr).sum()), "n_zerok_above": int((cs[zk] > thr).sum())})
    thresholds = pd.DataFrame(thr_rows)
    frac_ctrl = (thresholds["n_ctrl_above"] / int((~zk).sum())).round(3).tolist()
    checks.add("cycling-score rule flags 10% of control cells by construction (per variant)", f"{frac_ctrl}",
               all(abs(f - 0.10) < 0.005 for f in frac_ctrl))
    t18 = pd.read_csv(run.use_input(lib.LABELS_CORE_LABELS, "labels_core canonical control labels (dpt_pseudotime)"),
                      dtype={"cell_id": str}).set_index("cell_id")
    dpt = t18["dpt_pseudotime"].reindex(obs.index).to_numpy(float)
    n_fin = int(np.isfinite(dpt[~zk]).sum())
    checks.add("control cells with a finite labels_core DPT", f"{n_fin} of {int((~zk).sum())}", n_fin > 0.95 * (~zk).sum(),
               "> 95%")
    checks.add("ZeroK cells have no labels_core DPT", f"{int(np.isfinite(dpt[zk]).sum())}", not np.isfinite(dpt[zk]).any())
    per["ctrl_dpt_t18"] = dpt
    for v in lib.VARIANTS:
        proj, mdist = project_dpt(joint.pca[v], zk, dpt, lib.K_PSEUDOTIME)
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
    shift_rows, pt_rows, cyc_rows = [], [], []
    groups = all_groups(obs)
    for col, group, m in groups:
        idx = np.flatnonzero(m)
        z, l = zk[idx], ld[idx]
        keys = {"label_column": col, "group": group}
        for v in lib.VARIANTS:
            for meas in SHIFT_MEASURES:
                shift_rows.append({**keys, "variant": v, "measure": meas,
                                   **compare_values(per[f"{meas}__{v}"].to_numpy()[idx], z, l)})
            pt_rows.append({**keys, "variant": v, "measure": "projected_dpt",
                            **compare_values(per[f"projected_dpt__{v}"].to_numpy()[idx], z, l),
                            "median_ctrl_own_dpt": float(np.nanmedian(dpt[idx][~z]))})
        flag_cols = [*FLAGS, *[f"cycling_score_rule__{v}" for v in lib.VARIANTS]]
        for f in flag_cols:
            x = per[f].to_numpy(bool)[idx]
            cyc_rows.append({**keys, "measure": f, **lib.proportion_compare(int(x[~z].sum()), int((~z).sum()),
                                                                           int(x[z].sum()), int(z.sum()))})
        if col != "label_free" and group in (lib.ALL_DCT,):
            x = (obs[col].to_numpy() == "ProLIF")[idx]
            cyc_rows.append({**keys, "measure": "ProLIF_label_share", **lib.proportion_compare(
                int(x[~z].sum()), int((~z).sum()), int(x[z].sum()), int(z.sum()))})
    for col in lib.LABEL_COLUMNS:
        x = obs[col].to_numpy() == "ProLIF"
        cyc_rows.append({"label_column": col, "group": lib.ALL_CELLS, "measure": "ProLIF_label_share",
                         **lib.proportion_compare(int(x[~zk].sum()), int((~zk).sum()), int(x[zk].sum()),
                                                  int(zk.sum()))})
    shift = cc.add_bh(pd.DataFrame(shift_rows), ["label_column", "variant", "measure"], "p_score_lm", "q_lm")
    shift["p_value_note"] = lib.P_NOTE
    pt = cc.add_bh(pd.DataFrame(pt_rows), ["label_column", "variant"], "p_score_lm", "q_lm")
    pt["p_value_note"] = lib.P_NOTE
    cyc = pd.DataFrame(cyc_rows)
    cyc["p_value_note"] = lib.P_NOTE

    # ---- depth-matched versions (D4)
    dm_rows, bal_rows = [], []
    for col, group, m in groups:
        sets = lib.matched_sets(joint, col, group, m)
        idx_all = np.flatnonzero(m)
        smd_before = lib.smd(ld[idx_all][zk[idx_all]], ld[idx_all][~zk[idx_all]])
        smd_after = [lib.smd(ld[iz], ld[ic]) for ic, iz in sets]
        mean_diff = [float(ld[iz].mean() - ld[ic].mean()) if len(ic) else np.nan for ic, iz in sets]
        npairs = [len(ic) for ic, _ in sets]
        bal_rows.append({"label_column": col, "group": group, "n_ctrl": int((m & ~zk).sum()),
                         "n_zerok": int((m & zk).sum()), "median_pairs": float(np.median(npairs)),
                         "min_pairs": int(np.min(npairs)), "smd_ln_depth_before": smd_before,
                         "smd_ln_depth_after_median": float(np.nanmedian(smd_after)),
                         "smd_ln_depth_after_max_abs": float(np.nanmax(np.abs(smd_after))),
                         "mean_ln_depth_difference_after_max_abs": float(np.nanmax(np.abs(mean_diff))),
                         "n_repeats_with_min_pairs": int(sum(n >= lib.MIN_DET_PER_CONDITION for n in npairs))})
        # repeats with fewer than 5 pairs are not compared (as the intensity part of D1)
        sets = [(ic, iz) for ic, iz in sets if len(ic) >= lib.MIN_DET_PER_CONDITION]
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
                            "effect": "median difference ZeroK - Ctrl", **lib.summarize_repeats(eff, full, pv)})
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
                            **lib.summarize_repeats(eff, full, pv)})
    dm = pd.DataFrame(dm_rows)
    dm["p_value_note"] = lib.P_NOTE
    bal = pd.DataFrame(bal_rows)
    worst_mean = float(bal["mean_ln_depth_difference_after_max_abs"].max())
    checks.add("depth matching: max |mean ln-depth difference| after matching (all repeats and groups)",
               f"{worst_mean:.4f}", worst_mean <= lib.CALIPER, f"<= caliper {lib.CALIPER}")
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

    run.table(shift, "state_score_shift")
    run.table(cyc, "state_cycling")
    run.table(thresholds, "state_cycling_thresholds")
    run.table(pt, "state_pseudotime")
    run.table(dm, "state_depth_matched")
    run.table(bal, "state_matching_balance")
    run.table(checks.frame(), "state_checks")
    run.finish(parameters={"k_pseudotime": lib.K_PSEUDOTIME, "n_pcs": lib.N_PCS,
                           "dpt_source": "labels_core canonical dpt_pseudotime (original root)",
                           "cycling_quantile_of_control": lib.CYCLING_QUANTILE, "caliper_ln": lib.CALIPER,
                           "n_repeats": lib.N_REPEATS, "matching": "greedy 1:1 nearest neighbour within plate",
                           "p_value_note": lib.P_NOTE, "runtime_s": round(time.time() - t_start, 1),
                           "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_response_core_state_checks.csv.")


if __name__ == "__main__":
    main()
