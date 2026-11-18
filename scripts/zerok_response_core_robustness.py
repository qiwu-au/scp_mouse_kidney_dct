#!/usr/bin/env python
"""zerok_response_core step 5: robustness of the zero-K vs control results.

Analysis plan, step 5.

(a) Depth-matched subsampling (D4): the step 2 panel statistics on depth-matched cells (greedy 1:1 nearest
    neighbour on ln n_detected_matched within label group and sort plate, caliper 0.05; 100 repeats; the same
    matched sets as step 4). Per item: how often the result keeps the full-data sign (sign of the part with the
    larger chi2), reaches BH q < 0.05 within the panel, and keeps its observed direction and match status.
(b) Technical split: ZeroK vs Ctrl effects from P1 cells alone and from P2 cells alone (sort plates of the
    pooled cells; technical check, never replication): panel (two-part coefficients, detection difference,
    log2 FC among detected, pseudobulk) and proteome-wide (step 3 proteins; effect sizes only, D8); Spearman
    and sign agreement between the plates.
(c) Cell downsampling: 50% and 25% of the cells of each condition within each group (without replacement),
    100 repeats each; panel summaries as in (a).
(d) Label-free version: all cells, and Slc12a3-detected cells only (panel); proteome-wide for all cells.
(e) Leave-Slc12a3-out scores: ALDO scores and the DCT1 - DCT2 score difference with and without Slc12a3.
(f) Proteome-wide effect sizes (D8) under depth matching and downsampling: Spearman with the full-data
    effects and sign agreement, per repeat, summarised.
Groups: DCT1, DCT2, ProLIF, Unknown, all_DCT under dct_label and dct_label_alternative, plus label-free
all_cells; every variant. All p-values are descriptive (one pool per condition; cells are not animal-level
replicates).

Inputs (read-only): results/h5ad/zerok_core_DCT_joint_labelled.h5ad; zerok_core panel, hash and joint labels;
zerok_response_core step 2 (per-cell scores, panel results), step 3 (proteome tables) and step 4 (score shift) outputs.
Outputs (results/tables/zerok_response_core/): zerok_response_core_robustness_panel_repeats.csv,
zerok_response_core_robustness_plate_split_panel.csv, zerok_response_core_robustness_plate_split_summary.csv,
zerok_response_core_robustness_label_free_panel.csv, zerok_response_core_robustness_label_free_proteome.csv,
zerok_response_core_robustness_leave_slc12a3_out.csv, zerok_response_core_robustness_proteome_effects.csv,
zerok_response_core_robustness_summary.csv, zerok_response_core_robustness_checks.csv.

Command (project root):  python scripts/zerok_response_core_robustness.py
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
import zerok_response_core_lib as lib

SCRIPT = "zerok_response_core_robustness.py"
KEY = ["label_column", "group", "variant", "item"]
KEEP = ["procedure", "repeat", *KEY, "item_type", "n_ctrl", "n_zerok", "dominant_sign", "q_primary",
        "observed_direction", "match_status", "det_logodds", "int_log2fc_adjusted", "score_coef"]
EFFECTS = ("detection_difference", "log2fc_detected", "pseudobulk_log2fc")
PANEL_EFFECTS = ("detection_difference_zerok_minus_ctrl", "det_logodds", "log2fc_detected", "int_log2fc_adjusted",
                 "score_coef", "median_difference")


def panel_groups(obs: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    out = [(col, g, m) for col in lib.LABEL_COLUMNS for g, m in lib.group_masks(obs, col).items()]
    return out + [("label_free", lib.ALL_CELLS, np.ones(len(obs), bool))]


def downsample(joint: lib.Joint, mask: np.ndarray, frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    zk = joint.is_zerok
    out = []
    for cond in (~zk, zk):
        pool = np.flatnonzero(mask & cond)
        out.append(np.sort(rng.choice(pool, size=max(1, int(round(frac * len(pool)))), replace=False)))
    return out[0], out[1]


def repeat_panel(joint, spec, scores, col, group, sets, procedure) -> pd.DataFrame:
    rows = []
    for r, (ic, iz) in enumerate(sets):
        rows.extend(lib.panel_rows(joint, spec, np.concatenate([ic, iz]), scores,
                                   keys={"procedure": procedure, "repeat": r, "label_column": col, "group": group}))
    frame = lib.finalize_panel(pd.DataFrame(rows), spec, ["procedure", "repeat", "label_column", "group", "variant"])
    return frame[[c for c in KEEP if c in frame]]


def summarize_panel(rep: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    ref = full.set_index(KEY)
    rows = []
    for (proc, *key), grp in rep.groupby(["procedure", *KEY], sort=False):
        f = ref.loc[tuple(key)]
        same_sign = (grp["dominant_sign"] == f["dominant_sign"]) & np.isfinite(grp["dominant_sign"])
        sig = grp["q_primary"] < lib.ALPHA
        rows.append({"procedure": proc, **dict(zip(KEY, key)), "item_type": f["item_type"],
                     "expected_direction": f["expected_direction"], "full_observed_direction": f["observed_direction"],
                     "full_match_status": f["match_status"], "full_q_primary": f["q_primary"],
                     "full_dominant_sign": f["dominant_sign"], "n_repeats": len(grp),
                     "median_n_ctrl": float(grp["n_ctrl"].median()), "median_n_zerok": float(grp["n_zerok"].median()),
                     "fraction_testable": float(grp["q_primary"].notna().mean()),
                     "fraction_same_sign": float(same_sign.mean()),
                     "fraction_q_below_0.05": float(sig.mean()),
                     "fraction_same_sign_and_q_below_0.05": float((same_sign & sig).mean()),
                     "fraction_same_observed_direction": float((grp["observed_direction"]
                                                                == f["observed_direction"]).mean()),
                     "fraction_same_match_status": float((grp["match_status"] == f["match_status"]).mean()),
                     "median_det_logodds": float(grp["det_logodds"].median()),
                     "median_int_log2fc_adjusted": float(grp["int_log2fc_adjusted"].median()),
                     "median_score_coef": float(grp["score_coef"].median())})
    out = pd.DataFrame(rows)
    out["p_value_note"] = lib.P_NOTE
    return out


def spearman(a, b) -> tuple[float, int, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan, int(ok.sum()), np.nan
    nz = ok & (a != 0) & (b != 0)
    return (float(stats.spearmanr(a[ok], b[ok]).statistic), int(ok.sum()),
            float(np.mean(np.sign(a[nz]) == np.sign(b[nz]))) if nz.any() else np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_response_core step 5: robustness.")
    parser.add_argument("--repeats", type=int, default=lib.N_REPEATS, help="Repeats for (a), (c) and (f).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "robustness", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_panel_hash(run, checks)
    joint = lib.load_joint(run, checks)
    obs = joint.obs
    zk = joint.is_zerok
    plate = obs["plate"].to_numpy()
    spec = lib.panel_spec(joint, lib.read_panel(run))
    panel_names = {name for _, name, _ in spec.proteins if name}
    per_cell = lib.read_table(run, "per_cell_scores", dtype={"cell_id": str}).set_index("cell_id")
    scores = {v: per_cell.filter(like=f"{v}__").rename(columns=lambda c: c.split("__", 1)[1]) for v in lib.VARIANTS}
    full_panel = lib.read_table(run, "panel_results")
    bts = {v: lib.back_transform(joint, v) for v in lib.VARIANTS}

    # ---- (d) label-free panel and proteome
    lf_rows = []
    for group, m in lib.label_free_masks(joint).items():
        lf_rows.extend(lib.panel_rows(joint, spec, np.flatnonzero(m), scores,
                                      keys={"label_column": "label_free", "group": group}))
    lf_panel = lib.finalize_panel(pd.DataFrame(lf_rows), spec, ["label_column", "group", "variant"])
    run.table(lf_panel, "robustness_label_free_panel")
    n_slc = int(lib.label_free_masks(joint)[lib.SLC12A3_DETECTED].sum())
    checks.add("label-free groups (all cells; Slc12a3-detected cells)", f"{len(obs)}; {n_slc}", None)
    print("  label-free proteome (all cells)", flush=True)
    lf_prot = lib.proteome_group(joint, np.arange(len(obs)), bts, {"label_column": "label_free",
                                                                    "group": lib.ALL_CELLS}, panel_names)
    run.table(lf_prot, "robustness_label_free_proteome")
    full_ref = pd.concat([full_panel, lf_panel], ignore_index=True)

    # ---- (a) depth-matched and (c) downsampled panel repeats
    reps = []
    for col, group, m in panel_groups(obs):
        t0 = time.time()
        sets = lib.matched_sets(joint, col, group, m, args.repeats)
        reps.append(repeat_panel(joint, spec, scores, col, group, sets, "depth_matched"))
        for fi, frac in enumerate(lib.DOWNSAMPLE_FRACTIONS):
            ds = [downsample(joint, m, frac, lib.repeat_rng(5, fi, lib.LABEL_COLUMN_INDEX[col],
                                                             lib.GROUP_ORDER.index(group), r))
                  for r in range(args.repeats)]
            reps.append(repeat_panel(joint, spec, scores, col, group, ds, f"downsample_{int(frac * 100)}pct"))
        print(f"  panel repeats {col} / {group}: {time.time() - t0:.0f} s", flush=True)
    rep = pd.concat(reps, ignore_index=True)
    rep_summary = summarize_panel(rep, full_ref)
    checks.add("panel repeats per procedure x group x variant x item", f"{rep_summary['n_repeats'].min()}-"
               f"{rep_summary['n_repeats'].max()}", rep_summary["n_repeats"].eq(args.repeats).all(),
               f"{args.repeats}")
    run.table(rep_summary, "robustness_panel_repeats")

    # ---- (b) plate split: panel
    split_rows = []
    for col, group, m in panel_groups(obs):
        per_plate = {}
        for p in lib.PLATES:
            rows = lib.panel_rows(joint, spec, np.flatnonzero(m & (plate == p)), scores,
                                  keys={"label_column": col, "group": group, "plate": p})
            per_plate[p] = lib.finalize_panel(pd.DataFrame(rows), spec,
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
            rho, n, agree = spearman(prot[f"{e}_P1"], prot[f"{e}_P2"])
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
            m = (np.ones(len(obs), bool) if col == "label_free" else lib.group_masks(obs, col)[group])
            cols_idx = np.array([joint.var_index(p) for p in g["protein"]])
            e1 = lib.effect_sizes(joint, v, np.flatnonzero(m & ~zk & (plate == "P1")),
                                  np.flatnonzero(m & zk & (plate == "P1")), cols_idx, bts[v])
            e2 = lib.effect_sizes(joint, v, np.flatnonzero(m & ~zk & (plate == "P2")),
                                  np.flatnonzero(m & zk & (plate == "P2")), cols_idx, bts[v])
            for e in EFFECTS:
                rho, n, agree = spearman(e1[e], e2[e])
                sum_rows.append({"scope": "proteome-wide", "label_column": col, "group": group, "variant": v,
                                 "effect": e, "n": n, "spearman_P1_vs_P2": rho, "sign_agreement": agree})
            ef = lib.effect_sizes(joint, v, np.flatnonzero(m & ~zk), np.flatnonzero(m & zk), cols_idx, bts[v])
            procs = {"depth_matched": lib.matched_sets(joint, col, group, m, args.repeats)}
            for fi, frac in enumerate(lib.DOWNSAMPLE_FRACTIONS):
                procs[f"downsample_{int(frac * 100)}pct"] = [
                    downsample(joint, m, frac, lib.repeat_rng(5, fi, lib.LABEL_COLUMN_INDEX[col],
                                                              lib.GROUP_ORDER.index(group), r))
                    for r in range(args.repeats)]
            for proc, sets in procs.items():
                res = {e: [] for e in EFFECTS}
                for ic, iz in sets:
                    er = lib.effect_sizes(joint, v, ic, iz, cols_idx, bts[v])
                    for e in EFFECTS:
                        res[e].append(spearman(ef[e], er[e]))
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
                             "source": "panel BH (step 2 / label-free here)"})
    shift = lib.read_table(run, "state_score_shift")
    sref = shift.set_index(["label_column", "group", "variant", "measure"])
    for (col, group, v), _ in shift.groupby(["label_column", "group", "variant"], sort=False):
        for meas in ("DCT1_minus_DCT2_score", "DCT1_minus_DCT2_meanz"):
            a, b = sref.loc[(col, group, v, meas)], sref.loc[(col, group, v, f"{meas}__no_Slc12a3")]
            lso_rows.append({"label_column": col, "group": group, "variant": v, "score": meas,
                             "coef_base": a["score_coef"], "q_base": a["q_lm"],
                             "observed_base": "", "coef_no_Slc12a3": b["score_coef"], "q_no_Slc12a3": b["q_lm"],
                             "observed_no_Slc12a3": "",
                             "same_sign": bool(np.sign(a["score_coef"]) == np.sign(b["score_coef"])),
                             "same_observed_direction": np.nan, "source": "step 4 (BH across groups)"})
    lso = pd.DataFrame(lso_rows)
    lso["p_value_note"] = lib.P_NOTE
    run.table(lso, "robustness_leave_slc12a3_out")

    # ---- summary
    srows = []
    for (proc, col, group, v), g in rep_summary.groupby(["procedure", "label_column", "group", "variant"], sort=False):
        sig = g.loc[g["full_q_primary"] < lib.ALPHA]
        scored = g.loc[g["expected_direction"].isin([lib.UP, lib.DOWN, lib.NO_CHANGE])]
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
    run.table(pd.DataFrame(srows), "robustness_summary")

    run.table(checks.frame(), "robustness_checks")
    run.finish(parameters={"repeats": args.repeats, "downsample_fractions": list(lib.DOWNSAMPLE_FRACTIONS),
                           "caliper_ln": lib.CALIPER, "seeds": "numpy default_rng([42, procedure, label column, "
                           "group, repeat])", "p_value_note": lib.P_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_response_core_robustness_checks.csv.")


if __name__ == "__main__":
    main()
