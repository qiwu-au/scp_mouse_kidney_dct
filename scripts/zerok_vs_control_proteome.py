#!/usr/bin/env python
"""zerok_vs_control step 4: proteome-wide zero-K vs control differences on the combined search (zerok_response_core step 3 rerun).

Analysis plan,
step 4. Same procedure as zerok_response_core_proteome.py (zerok_response_core_lib.proteome_group and the zerok_response_core group and top-list
helpers imported unchanged):

- groups: labels with >= 20 cells in each condition, plus all_DCT (D2), under dct_label and dct_label_alternative;
  labels absent in a condition are listed with the reason ('does not apply'), labels with 1-19 cells in a condition
  are listed as below the D2 threshold;
- proteins: the joint proteins (retained in both conditions) detected in >= 10% of the group's cells in at least one
  condition; two-part model (D1) per protein and variant; BH within label column x group x variant; observed
  direction (D1 rule);
- top 25 up and down per group and variant (sorted by q, then |effect|); no enrichment analysis (D7);
- protein groups retained in one condition only (dct_intake): listed, not tested.
All p-values are descriptive (one pool per condition; cells are not animal-level replicates).

Inputs (read-only): results/h5ad/dct_labels_DCT_joint_labelled.h5ad; dct_labels canonical labels; zerok_core panel and hash;
zerok_vs_control step 3 panel results; dct_intake one-condition-only table.
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_proteome_<primary|alternative>_<variant>.csv,
zerok_vs_control_proteome_top_lists.csv, zerok_vs_control_proteome_summary.csv, zerok_vs_control_proteome_groups.csv,
zerok_vs_control_one_condition_only_proteins.csv, zerok_vs_control_proteome_checks.csv.

Command (project root):  python scripts/zerok_vs_control_proteome.py
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
import zerok_response_core_proteome as t22pr
import dct_intake_lib as t26
import zerok_vs_control_lib as lib

SCRIPT = "zerok_vs_control_proteome.py"
t22 = lib.t22
TOP_N = t22pr.TOP_N


def group_table(joint: lib.Joint) -> pd.DataFrame:
    """Every label group with its cell counts and whether it is tested proteome-wide (D2)."""
    rows = []
    for col in lib.LABEL_COLUMNS:
        tested = t22pr.proteome_groups(joint.obs, col)
        for group, m in t22.group_masks(joint.obs, col).items():
            n_c, n_z = lib.group_counts(joint, m)
            why = lib.absent_reason(n_c, n_z)
            status = ("tested" if group in tested and not why else why if why else
                      f"not tested: fewer than {t22.MIN_GROUP_PROTEOME} cells in a condition (D2)")
            rows.append({"label_column": col, "group": group, "n_ctrl": n_c, "n_zerok": n_z, "status": status})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 4: proteome-wide differences.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "proteome", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    joint = lib.load_joint(run, checks)
    spec = t22.panel_spec(joint, t22.read_panel(run))
    panel_names = {name for _, name, _ in spec.proteins if name}
    bts = {v: t22.back_transform(joint, v) for v in lib.VARIANTS}

    groups = group_table(joint)
    frames = []
    group_rows = []
    for r in groups.loc[groups["status"] == "tested"].itertuples():
        t0 = time.time()
        m = t22.group_masks(joint.obs, r.label_column)[r.group]
        idx = np.flatnonzero(m)
        f = t22.proteome_group(joint, idx, bts, {"label_column": r.label_column, "group": r.group}, panel_names)
        frames.append(f)
        n_prot = f["protein"].nunique()
        group_rows.append({"label_column": r.label_column, "group": r.group, "n_ctrl": r.n_ctrl,
                           "n_zerok": r.n_zerok, "n_proteins_tested": n_prot})
        print(f"  {r.label_column} / {r.group}: {len(idx)} cells, {n_prot} proteins, {time.time() - t0:.0f} s",
              flush=True)
    not_tested = groups.loc[groups["status"] != "tested"]
    checks.add("label groups not tested proteome-wide (reason)", "; ".join(
        f"{lib.LABEL_SHORT[r.label_column]}/{r.group}: {r.status}" for r in not_tested.itertuples()) or "none", None)
    allf = lib.set_note(pd.concat(frames, ignore_index=True))

    # checks
    det_ok = allf.groupby(["label_column", "group", "protein"])["det_lr"].nunique(dropna=True).max() <= 1
    checks.add("detection part identical across variants", f"{det_ok}", det_ok)
    fr = allf.query("variant == 'none'")
    min_frac = float(np.maximum(fr["detection_fraction_ctrl"], fr["detection_fraction_zerok"]).min())
    checks.add("every tested protein detected in >= 10% of cells in at least one condition",
               f"min {min_frac:.3f}", min_frac >= t22.MIN_DETECTED_FRACTION - 1e-12)
    in_joint = bool(allf["protein"].isin(set(joint.var_names)).all())
    lv = sorted(set(allf["match_level"].astype(str)))
    checks.add("tested proteins: joint proteins of the combined search only", f"{in_joint}; {lv}",
               in_joint and lv == ["combined_search"])
    q_ok = bool((allf["q_two_part"].dropna() >= allf.loc[allf["q_two_part"].notna(), "p_two_part"] - 1e-12).all())
    checks.add("BH q >= p for every tested row", f"{q_ok}", q_ok)
    panel_res = lib.read_table(run, "panel_results")
    merged = panel_res.query("item_type == 'protein' and status == 'tested'").merge(
        allf[["label_column", "group", "variant", "protein", "p_two_part"]],
        left_on=["label_column", "group", "variant", "joint_var_name"],
        right_on=["label_column", "group", "variant", "protein"], suffixes=("_panel", "_proteome"))
    dev = float(np.nanmax(np.abs(merged["p_two_part_panel"] - merged["p_two_part_proteome"])))
    checks.add("panel proteins: two-part p equals step 3 (same groups, before BH)",
               f"{len(merged)} rows; max |diff| {dev:.2g}", dev < 1e-10)

    for col in lib.LABEL_COLUMNS:
        for v in lib.VARIANTS:
            run.table(allf.query("label_column == @col and variant == @v"),
                      f"proteome_{lib.LABEL_SHORT[col]}_{v}")
    run.table(t22pr.top_lists(allf), "proteome_top_lists")
    summ = (allf.groupby(["label_column", "group", "variant", "observed_direction"]).size().unstack(fill_value=0)
            .reset_index())
    summ = summ.merge(pd.DataFrame(group_rows), on=["label_column", "group"])
    nq = (allf.assign(sig=allf["q_two_part"] < lib.ALPHA).groupby(["label_column", "group", "variant"])["sig"]
          .sum().rename("n_q_below_0.05").reset_index())
    summ = summ.merge(nq, on=["label_column", "group", "variant"])
    summ["share_q_below_0.05"] = summ["n_q_below_0.05"] / summ["n_proteins_tested"]
    summ["p_value_note"] = lib.P_NOTE
    run.table(summ, "proteome_summary")
    run.table(groups, "proteome_groups")

    # retained in one condition only (dct_intake): listed, not tested
    one = t26.read_table(run, "one_condition_only", keep_default_na=False)
    one["n_cells_own_condition"] = np.where(one["retention"] == "Ctrl only", one["n_cells_ctrl"],
                                            one["n_cells_zerok"]).astype(int)
    one["n_cells_other_condition"] = np.where(one["retention"] == "Ctrl only", one["n_cells_zerok"],
                                              one["n_cells_ctrl"]).astype(int)
    own = np.where(one["retention"] == "Ctrl only", "Ctrl", "ZeroK")
    one["fraction_retained_cells_own_condition"] = one["n_cells_own_condition"] / pd.Series(own).map(lib.N_CELLS).values
    one["note"] = ("retained (>= 2 detected cells) in one condition only in the combined search; not in the joint "
                   "set and not tested; presence or absence is not read as a condition effect")
    one = one.sort_values(["retention", "n_cells_own_condition"], ascending=[True, False])
    counts = one["retention"].value_counts().to_dict()
    checks.add("protein groups retained in one condition only (dct_intake: 181 Ctrl, 56 ZeroK)",
               f"Ctrl {counts.get('Ctrl only', 0)}, ZeroK {counts.get('ZeroK only', 0)}",
               counts.get("Ctrl only") == 181 and counts.get("ZeroK only") == 56, "181 / 56")
    run.table(one, "one_condition_only_proteins")

    run.table(checks.frame(), "proteome_checks")
    run.finish(parameters={"min_cells_per_condition": t22.MIN_GROUP_PROTEOME,
                           "min_detected_fraction": t22.MIN_DETECTED_FRACTION,
                           "bh_block": "label_column x group x variant", "top_n": TOP_N,
                           "top_list_rule": "observed direction up/down (D1), sorted by q then |effect|",
                           "enrichment": "not run (D7)", "ported_from": "zerok_response_core_proteome.py",
                           "p_value_note": lib.P_NOTE, "runtime_s": round(time.time() - t_start, 1),
                           "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_proteome_checks.csv.")


if __name__ == "__main__":
    main()
