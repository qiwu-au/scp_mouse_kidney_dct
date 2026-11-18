#!/usr/bin/env python
"""zerok_response_core step 3: proteome-wide zero-K vs control differences per DCT label.

Analysis plan, step 3.

- Groups: every label with >= 20 cells in each condition (D2), and all_DCT (DCT1 + DCT2 + ProLIF), under
  dct_label (primary) and dct_label_alternative.
- Proteins: the 5,130 matched proteins (intake_core levels 1 and 2) detected in >= 10% of the group's cells in at
  least one condition.
- Statistics per protein (zerok_response_core_lib.proteome_group, the step 2 functions): detection difference (Newcombe
  CI, Fisher p), ln-intensity among detected cells (medians, mean-based log2 FC, Mann-Whitney p), the two-part
  model (Firth logistic + OLS, log n_detected_matched as covariate; summed LR test), pseudobulk log2 FC per
  sort plate and pooled; BH within label group x variant; observed direction by the D1 rule.
- Proteins found in one search only are listed separately (intake_core mapping); their presence or absence is not a
  condition effect because the searches were separate, and they are not tested.
- No enrichment (D7: a gene-set resource would have to be downloaded).
All p-values are descriptive (one pool per condition; cells are not animal-level replicates).

Inputs (read-only): results/h5ad/zerok_core_DCT_joint_labelled.h5ad; zerok_core panel, hash and joint labels;
intake_core protein mapping and per-cell table.
Outputs (results/tables/zerok_response_core/): zerok_response_core_proteome_<primary|alternative>_<variant>.csv (6 files),
zerok_response_core_proteome_top_lists.csv, zerok_response_core_proteome_summary.csv, zerok_response_core_one_search_only_proteins.csv,
zerok_response_core_proteome_checks.csv.

Command (project root):  python scripts/zerok_response_core_proteome.py
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
import zerok_core_lib as t21
import zerok_response_core_lib as lib

SCRIPT = "zerok_response_core_proteome.py"
TOP_N = 25


def proteome_groups(obs: pd.DataFrame, col: str) -> dict[str, np.ndarray]:
    zk = obs["condition"].to_numpy() == "ZeroK"
    masks = lib.group_masks(obs, col)
    return {g: m for g, m in masks.items()
            if g == lib.ALL_DCT or (int((m & ~zk).sum()) >= lib.MIN_GROUP_PROTEOME
                                    and int((m & zk).sum()) >= lib.MIN_GROUP_PROTEOME)}


def top_lists(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, grp in frame.groupby(["label_column", "group", "variant"], sort=False):
        for direction in (lib.UP, lib.DOWN):
            sel = grp.loc[grp["observed_direction"] == direction].copy()
            sel["abs_effect"] = sel["int_log2fc_adjusted"].abs().fillna(0) + sel["det_logodds"].abs().fillna(0)
            sel = sel.sort_values(["q_two_part", "abs_effect"], ascending=[True, False]).head(TOP_N)
            sel.insert(3, "rank", np.arange(1, len(sel) + 1))
            sel.insert(3, "list", direction)
            out.append(sel.drop(columns="abs_effect"))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_response_core step 3: proteome-wide differences.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "proteome", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_panel_hash(run, checks)
    joint = lib.load_joint(run, checks)
    spec = lib.panel_spec(joint, lib.read_panel(run))
    panel_names = {name for _, name, _ in spec.proteins if name}
    bts = {v: lib.back_transform(joint, v) for v in lib.VARIANTS}

    frames = []
    group_rows = []
    for col in lib.LABEL_COLUMNS:
        for group, m in proteome_groups(joint.obs, col).items():
            t0 = time.time()
            idx = np.flatnonzero(m)
            f = lib.proteome_group(joint, idx, bts, {"label_column": col, "group": group}, panel_names)
            frames.append(f)
            n_prot = f["protein"].nunique()
            group_rows.append({"label_column": col, "group": group, "n_ctrl": int((~joint.is_zerok[idx]).sum()),
                               "n_zerok": int(joint.is_zerok[idx].sum()), "n_proteins_tested": n_prot})
            print(f"  {col} / {group}: {len(idx)} cells, {n_prot} proteins, {time.time() - t0:.0f} s", flush=True)
    excluded = [f"{lib.LABEL_SHORT[c]}/{g}" for c in lib.LABEL_COLUMNS for g in lib.LABELS
                if g not in proteome_groups(joint.obs, c)]
    checks.add("labels without >= 20 cells in each condition (no proteome-wide test)", ", ".join(excluded) or "none",
               None)
    allf = pd.concat(frames, ignore_index=True)

    # checks
    det_ok = allf.groupby(["label_column", "group", "protein"])["det_lr"].nunique(dropna=True).max() <= 1
    checks.add("detection part identical across variants", f"{det_ok}", det_ok)
    fr = allf.query("variant == 'none'")
    min_frac = float(np.maximum(fr["detection_fraction_ctrl"], fr["detection_fraction_zerok"]).min())
    checks.add("every tested protein detected in >= 10% of cells in at least one condition",
               f"min {min_frac:.3f}", min_frac >= lib.MIN_DETECTED_FRACTION - 1e-12)
    lv = sorted(set(allf["match_level"].astype(str)))
    checks.add("tested proteins: matched levels only", f"{lv}", set(lv) <= {"1", "2"})
    q_ok = bool((allf["q_two_part"].dropna() >= allf.loc[allf["q_two_part"].notna(), "p_two_part"] - 1e-12).all())
    checks.add("BH q >= p for every tested row", f"{q_ok}", q_ok)
    panel_res = lib.read_table(run, "panel_results")
    merged = panel_res.query("item_type == 'protein'").merge(
        allf[["label_column", "group", "variant", "protein", "p_two_part"]],
        left_on=["label_column", "group", "variant", "joint_var_name"],
        right_on=["label_column", "group", "variant", "protein"], suffixes=("_panel", "_proteome"))
    dev = float(np.nanmax(np.abs(merged["p_two_part_panel"] - merged["p_two_part_proteome"])))
    checks.add("panel proteins: two-part p equals step 2 (same groups, before BH)",
               f"{len(merged)} rows; max |diff| {dev:.2g}", dev < 1e-10)

    for col in lib.LABEL_COLUMNS:
        for v in lib.VARIANTS:
            run.table(allf.query("label_column == @col and variant == @v"),
                      f"proteome_{lib.LABEL_SHORT[col]}_{v}")
    run.table(top_lists(allf), "proteome_top_lists")
    summ = (allf.groupby(["label_column", "group", "variant", "observed_direction"]).size().unstack(fill_value=0)
            .reset_index())
    summ = summ.merge(pd.DataFrame(group_rows), on=["label_column", "group"])
    nq = (allf.assign(sig=allf["q_two_part"] < lib.ALPHA).groupby(["label_column", "group", "variant"])["sig"]
          .sum().rename("n_q_below_0.05").reset_index())
    summ = summ.merge(nq, on=["label_column", "group", "variant"])
    summ["p_value_note"] = lib.P_NOTE
    run.table(summ, "proteome_summary")

    # one-search-only list (not tested)
    mapping = t21.read_t17_mapping(run)
    n_cells = t21.retained_cells(run)
    one = mapping.loc[mapping["match_status"] == "one_search_only"].copy()
    one["n_cells_own_search"] = np.where(one["search"] == "Ctrl", one["n_cells_ctrl"], one["n_cells_zerok"])
    one["fraction_retained_cells_own_condition"] = one["n_cells_own_search"] / one["search"].map(n_cells)
    one["retained_in_own_condition"] = np.where(one["search"] == "Ctrl", one["retained_ctrl"], one["retained_zerok"])
    one["note"] = ("found in one Spectronaut search only; presence or absence is not a condition effect because the "
                   "searches were separate; not tested")
    keep_cols = ["search", "protein_group", "pg_genes", "gene_name", "var_name", "n_cells_own_search",
                 "fraction_retained_cells_own_condition", "retained_in_own_condition", "match_reason", "note"]
    one = one[[c for c in keep_cols if c in one]].sort_values(["search", "n_cells_own_search"], ascending=[True, False])
    counts = one["search"].value_counts().to_dict()
    checks.add("one-search-only protein groups (intake_core: 1,073 Ctrl and 401 ZeroK)",
               f"Ctrl {counts.get('Ctrl', 0)}, ZeroK {counts.get('ZeroK', 0)}",
               counts.get("Ctrl") == 1073 and counts.get("ZeroK") == 401, "1073 / 401")
    run.table(one, "one_search_only_proteins")

    run.table(checks.frame(), "proteome_checks")
    run.finish(parameters={"min_cells_per_condition": lib.MIN_GROUP_PROTEOME,
                           "min_detected_fraction": lib.MIN_DETECTED_FRACTION,
                           "bh_block": "label_column x group x variant", "top_n": TOP_N,
                           "top_list_rule": "observed direction up/down (D1), sorted by q then |effect|",
                           "enrichment": "not run (D7)", "p_value_note": lib.P_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_response_core_proteome_checks.csv.")


if __name__ == "__main__":
    main()
