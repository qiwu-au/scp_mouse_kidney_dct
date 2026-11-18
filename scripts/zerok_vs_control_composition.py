#!/usr/bin/env python
"""zerok_vs_control step 2: composition by condition on the combined search (zerok_response_core step 1 rerun).

Analysis plan,
step 2. Same procedure as zerok_response_core_composition.py (functions reused by import), on the dct_labels labels:

- Label fractions per condition and per condition x sort plate, under dct_label (primary) and
  dct_label_alternative (ZeroK transfer route); Wilson 95% intervals.
- Difference in fractions ZeroK - Ctrl with the Newcombe hybrid score 95% interval (cells as units; descriptive),
  pooled and within each sort plate; Fisher exact p beside it (descriptive). A label absent in both conditions
  gives 0 vs 0 (reported as observed).
- Contamination markers (intake_core list plus the Task 3 negative panels): detection fraction per condition within each
  label group, per gene and 'any gene of the group'. Coverage from the dct_intake protein table through
  zerok_core_lib.gene_coverage (unchanged); genes outside the joint set go to the coverage table only. Groups with
  no cells in a condition are 'does not apply' rows.

Inputs (read-only): results/h5ad/dct_labels_DCT_joint_labelled.h5ad; dct_labels canonical labels; zerok_core panel and hash;
dct_intake protein and per-cell tables (coverage only).
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_label_fractions.csv, zerok_vs_control_label_fraction_differences.csv,
zerok_vs_control_contamination_detection.csv, zerok_vs_control_contamination_coverage.csv, zerok_vs_control_composition_checks.csv.

Command (project root):  python scripts/zerok_vs_control_composition.py
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
import zerok_response_core_composition as t22c
import dct_intake_lib as t26
import zerok_vs_control_lib as lib

SCRIPT = "zerok_vs_control_composition.py"
t22 = lib.t22


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 2: composition by condition.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "composition", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    joint = lib.load_joint(run, checks)
    obs = joint.obs
    zk = joint.is_zerok
    plate = obs["plate"].to_numpy()

    # ---- label fractions
    frac_rows, diff_rows = [], []
    for col in lib.LABEL_COLUMNS:
        lab = obs[col].to_numpy()
        for scope in ("all", *lib.PLATES):
            in_scope = np.ones(len(obs), bool) if scope == "all" else plate == scope
            for cond in lib.CONDITIONS:
                m = in_scope & (obs["condition"].to_numpy() == cond)
                n = int(m.sum())
                for label in lib.LABELS:
                    k = int((m & (lab == label)).sum())
                    lo, hi = t22.wilson(k, n)
                    frac_rows.append({"label_column": col, "plate": scope, "condition": cond, "label": label,
                                      "n_label": k, "n_cells": n, "fraction": k / n, "ci_low": lo, "ci_high": hi})
            mc, mz = in_scope & ~zk, in_scope & zk
            for label in lib.LABELS:
                kc, kz = int((mc & (lab == label)).sum()), int((mz & (lab == label)).sum())
                n_all = {c: int(((obs["condition"] == c).to_numpy() & (lab == label)).sum()) for c in lib.CONDITIONS}
                absent = [c for c in lib.CONDITIONS if n_all[c] == 0]
                note = (f"label absent in {' and '.join(absent)} under the frozen dct_labels rule" if absent else "")
                diff_rows.append({"label_column": col, "plate": scope, "label": label,
                                  **t22.proportion_compare(kc, int(mc.sum()), kz, int(mz.sum())),
                                  "note": note, "p_value_note": lib.P_NOTE})
    fractions = pd.DataFrame(frac_rows)
    diffs = pd.DataFrame(diff_rows)
    for col in lib.LABEL_COLUMNS:
        sums = fractions.loc[fractions["label_column"] == col].groupby(["plate", "condition"])["n_label"].sum()
        totals = fractions.loc[fractions["label_column"] == col].groupby(["plate", "condition"])["n_cells"].first()
        checks.add(f"{col}: label counts sum to the cells of every condition x plate", f"{bool((sums == totals).all())}",
                   bool((sums == totals).all()))
        for cond in lib.CONDITIONS:
            got = {r.label: r.n_label for r in fractions.query(
                "label_column == @col and plate == 'all' and condition == @cond").itertuples()}
            exp = lib.EXPECTED_COUNTS[(col, cond)]
            checks.add(f"{col} {cond} counts equal dct_labels", "; ".join(f"{k} {v}" for k, v in got.items()), got == exp,
                       "; ".join(f"{k} {v}" for k, v in exp.items()))
    ci = diffs.dropna(subset=["difference_zerok_minus_ctrl"])
    ci_ok = bool(((ci["difference_ci_low"] <= ci["difference_zerok_minus_ctrl"] + 1e-12)
                  & (ci["difference_zerok_minus_ctrl"] <= ci["difference_ci_high"] + 1e-12)).all())
    checks.add("every difference lies inside its 95% interval", f"{ci_ok}", ci_ok)
    alldct = {(c, cond): int(((obs["condition"] == cond) & obs[c].isin(lib.CORE)).sum())
              for c in lib.LABEL_COLUMNS for cond in lib.CONDITIONS}
    checks.add("all_DCT cells per condition (DCT1 + DCT2 + ProLIF)",
               "; ".join(f"{lib.LABEL_SHORT[c]} {cond} {n}" for (c, cond), n in alldct.items()), None)
    run.table(fractions, "label_fractions")
    run.table(lib.ordered(diffs, ["label_column", "plate", "label"]), "label_fraction_differences")

    # ---- contamination markers
    genes = t22c.contamination_genes()
    mapping = t26.mapping_frame(t26.read_proteins(run))
    cells = t26.read_cells(run)
    n_cells = {c: int((cells["retained"] & (cells["condition"] == c)).sum()) for c in lib.CONDITIONS}
    cov_rows = []
    for source, group, gene in genes:
        c = t21.gene_coverage(mapping, gene, n_cells)
        cov_rows.append({"source": source, "panel_group": group, "gene": gene,
                         "joint_var_name": t22.resolve_protein(joint, gene),
                         **{k: c.get(k, "") for k in ("ctrl_in_export", "zerok_in_export", "ctrl_retained",
                                                      "zerok_retained", "in_joint", "ctrl_fraction_cells",
                                                      "zerok_fraction_cells")},
                         "note": ("also a DCT2 marker and a panel protein (expected down)" if gene == "Calb1" else
                                  "TAL-associated (the project notes framing)" if gene == "Umod" else "")})
    coverage = pd.DataFrame(cov_rows)
    coverage["compared_between_conditions"] = coverage["joint_var_name"].astype(str).ne("")
    in_export = coverage["ctrl_in_export"].astype(bool) | coverage["zerok_in_export"].astype(bool)
    coverage["why_not_compared"] = np.where(
        coverage["compared_between_conditions"], "",
        np.where(in_export, "in the export but not retained in both conditions (not in the joint set): presence is "
                 "not a condition effect", "absent from the combined export"))
    joint_ok = bool((coverage["compared_between_conditions"] == coverage["in_joint"].astype(bool)).all())
    checks.add("contamination genes: resolved in the joint object = in_joint of the dct_intake protein table", f"{joint_ok}",
               joint_ok)
    run.table(coverage, "contamination_coverage")

    in_joint = [(s, g, gene, t22.resolve_protein(joint, gene)) for s, g, gene in genes]
    in_joint = [(s, g, gene, name) for s, g, gene, name in in_joint if name]
    group_sets = {}
    for s, g, gene, name in in_joint:
        group_sets.setdefault((s, g), []).append(name)
    det_rows = []
    for col in lib.LABEL_COLUMNS:
        masks, absent = lib.split_groups(joint, col)
        masks[lib.ALL_CELLS] = np.ones(len(obs), bool)
        for group, why in absent.items():
            det_rows.append(lib.absent_rows(joint, col, group, why, gene="all contamination genes"))
        for group, m in masks.items():
            mc, mz = m & ~zk, m & zk
            for s, g, gene, name in in_joint:
                d = joint.detected[:, joint.var_index(name)]
                det_rows.append({"label_column": col, "group": group, "source": s, "panel_group": g, "gene": gene,
                                 "joint_var_name": name, **t22.proportion_compare(
                                     int(d[mc].sum()), int(mc.sum()), int(d[mz].sum()), int(mz.sum())),
                                 "status": "compared", "p_value_note": lib.P_NOTE})
            for (s, g), names in group_sets.items():
                d = joint.detected[:, [joint.var_index(n) for n in names]].any(axis=1)
                det_rows.append({"label_column": col, "group": group, "source": s, "panel_group": g,
                                 "gene": f"any of {'/'.join(names)}", "joint_var_name": ";".join(names),
                                 **t22.proportion_compare(int(d[mc].sum()), int(mc.sum()), int(d[mz].sum()),
                                                          int(mz.sum())), "status": "compared",
                                 "p_value_note": lib.P_NOTE})
    detection = pd.DataFrame(det_rows)
    run.table(detection, "contamination_detection")
    checks.add("contamination genes in the joint set", f"{len(in_joint)} of {len(genes)}: "
               + ", ".join(gene for _, _, gene, _ in in_joint), None)
    missing = coverage.loc[~coverage["compared_between_conditions"], "gene"].tolist()
    checks.add("contamination genes not compared (coverage table only)", ", ".join(missing) or "none", None)
    na = detection.loc[detection["status"].astype(str).str.startswith(lib.DOES_NOT_APPLY)]
    checks.add("'does not apply' groups (label absent in a condition)", "; ".join(
        f"{lib.LABEL_SHORT[r.label_column]}/{r.group} (Ctrl {r.n_ctrl}, ZeroK {r.n_zerok})" for r in na.itertuples()),
        None)

    run.table(checks.frame(), "composition_checks")
    run.finish(parameters={"label_columns": list(lib.LABEL_COLUMNS), "ci": "Wilson (fractions); Newcombe hybrid "
                           "score (differences)", "p_value_note": lib.P_NOTE, "ported_from": "zerok_response_core_composition.py",
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_composition_checks.csv.")


if __name__ == "__main__":
    main()
