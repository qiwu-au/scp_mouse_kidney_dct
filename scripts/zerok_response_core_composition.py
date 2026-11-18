#!/usr/bin/env python
"""zerok_response_core step 1: composition by condition (label fractions, contamination markers).

Analysis plan, step 1.

- Label fractions per condition and per condition x sort plate, under dct_label (primary) and
  dct_label_alternative (ZeroK transfer route); Wilson 95% intervals.
- Difference in fractions ZeroK - Ctrl with the Newcombe hybrid score 95% interval (cells as units;
  descriptive), pooled and within each sort plate; Fisher exact p beside it (descriptive).
- Contamination markers (intake_core list: TAL, CNT/PC, IC, PT, endothelial, immune; plus the Task 3 negative
  panels of reviewer_core_lib): detection fraction per condition within each label group, per gene and
  'any gene of the group'; genes outside the matched set go to a coverage table only (separate searches).

Inputs (read-only): results/h5ad/zerok_core_DCT_joint_labelled.h5ad; zerok_core dct_joint_labels, panel and hash;
intake_core protein mapping and per-cell table (coverage only).
Outputs (results/tables/zerok_response_core/): zerok_response_core_label_fractions.csv,
zerok_response_core_label_fraction_differences.csv, zerok_response_core_contamination_detection.csv,
zerok_response_core_contamination_coverage.csv, zerok_response_core_composition_checks.csv.

Command (project root):  python scripts/zerok_response_core_composition.py
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
import reviewer_core_lib as t2
import cluster_de_lib as t5
import intake_core_lib as t17
import zerok_core_lib as t21
import zerok_response_core_lib as lib

SCRIPT = "zerok_response_core_composition.py"
EXTRA_TASK3 = {f"Task 3 {p}": t2.TASK3_MARKER_PANELS[p]["genes"] for p in t2.NEGATIVE_PANELS}


def contamination_genes() -> list[tuple[str, str, str]]:
    """(source, panel group, gene); intake_core list first, then Task 3 genes not already listed."""
    rows = [("intake_core contamination panel", g, gene) for g, genes in t17.CONTAMINATION_PANEL.items() for gene in genes]
    seen = {r[2] for r in rows}
    for panel, genes in EXTRA_TASK3.items():
        rows.extend(("Task 3 negative panel (reviewer_core_lib)", panel, g) for g in genes if g not in seen)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_response_core step 1: composition by condition.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "composition", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_panel_hash(run, checks)
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
                    lo, hi = lib.wilson(k, n)
                    frac_rows.append({"label_column": col, "plate": scope, "condition": cond, "label": label,
                                      "n_label": k, "n_cells": n, "fraction": k / n, "ci_low": lo, "ci_high": hi})
            mc, mz = in_scope & ~zk, in_scope & zk
            for label in lib.LABELS:
                diff_rows.append({"label_column": col, "plate": scope, "label": label,
                                  **lib.proportion_compare(int((mc & (lab == label)).sum()), int(mc.sum()),
                                                           int((mz & (lab == label)).sum()), int(mz.sum())),
                                  "p_value_note": lib.P_NOTE})
    fractions = pd.DataFrame(frac_rows)
    diffs = pd.DataFrame(diff_rows)
    for col in lib.LABEL_COLUMNS:
        sums = fractions.loc[fractions["label_column"] == col].groupby(["plate", "condition"])["n_label"].sum()
        totals = fractions.loc[fractions["label_column"] == col].groupby(["plate", "condition"])["n_cells"].first()
        checks.add(f"{col}: label counts sum to the cells of every condition x plate", f"{bool((sums == totals).all())}",
                   bool((sums == totals).all()))
    ctrl_counts = "; ".join(f"{r.label} {r.n_label}" for r in fractions.query(
        "label_column == 'dct_label' and plate == 'all' and condition == 'Ctrl'").itertuples())
    checks.add("control counts equal labels_core (DCT1 477; DCT2 144; ProLIF 34; Unknown 72)", ctrl_counts,
               ctrl_counts == "DCT1 477; DCT2 144; ProLIF 34; Unknown 72")
    zk_counts = "; ".join(f"{r.label} {r.n_label}" for r in fractions.query(
        "label_column == 'dct_label' and plate == 'all' and condition == 'ZeroK'").itertuples())
    checks.add("ZeroK dct_label counts equal zerok_core zerok_pipeline_label (DCT1 519; DCT2 35; ProLIF 18; Unknown 121)",
               zk_counts, zk_counts == "DCT1 519; DCT2 35; ProLIF 18; Unknown 121")
    zk_alt = "; ".join(f"{r.label} {r.n_label}" for r in fractions.query(
        "label_column == 'dct_label_alternative' and plate == 'all' and condition == 'ZeroK'").itertuples())
    checks.add("ZeroK alternative counts equal zerok_core zerok_transfer_label (DCT1 518; DCT2 108; ProLIF 17; Unknown 50)",
               zk_alt, zk_alt == "DCT1 518; DCT2 108; ProLIF 17; Unknown 50")
    ci_ok = bool(((diffs["difference_ci_low"] <= diffs["difference_zerok_minus_ctrl"] + 1e-12)
                  & (diffs["difference_zerok_minus_ctrl"] <= diffs["difference_ci_high"] + 1e-12)).all())
    checks.add("every difference lies inside its 95% interval", f"{ci_ok}", ci_ok)
    run.table(fractions, "label_fractions")
    run.table(lib.ordered(diffs, ["label_column", "plate", "label"]), "label_fraction_differences")

    # ---- contamination markers
    genes = contamination_genes()
    mapping = t21.read_t17_mapping(run)
    n_cells = t21.retained_cells(run)
    cov_rows = []
    for source, group, gene in genes:
        c = t21.gene_coverage(mapping, gene, n_cells)
        cov_rows.append({"source": source, "panel_group": group, "gene": gene, "joint_var_name": lib.resolve_protein(
            joint, gene), **{k: c.get(k, "") for k in ("ctrl_in_export", "zerok_in_export", "ctrl_match_status",
                                                         "zerok_match_status", "match_level", "in_joint",
                                                         "ctrl_fraction_cells", "zerok_fraction_cells")},
                         "note": ("also a DCT2 marker and a panel protein (expected down)" if gene == "Calb1" else
                                  "TAL-associated (the project notes framing)" if gene == "Umod" else "")})
    coverage = pd.DataFrame(cov_rows)
    coverage["compared_between_conditions"] = coverage["joint_var_name"].astype(str).ne("")
    coverage["why_not_compared"] = np.where(
        coverage["compared_between_conditions"], "",
        np.where(coverage["ctrl_in_export"].astype(bool) | coverage["zerok_in_export"].astype(bool),
                 "not in the matched set (one search only or not retained): presence is not a condition effect",
                 "absent from both exports"))
    run.table(coverage, "contamination_coverage")

    in_joint = [(s, g, gene, lib.resolve_protein(joint, gene)) for s, g, gene in genes]
    in_joint = [(s, g, gene, name) for s, g, gene, name in in_joint if name]
    group_sets = {}
    for s, g, gene, name in in_joint:
        group_sets.setdefault((s, g), []).append(name)
    det_rows = []
    for col in lib.LABEL_COLUMNS:
        masks = {**lib.group_masks(obs, col), lib.ALL_CELLS: np.ones(len(obs), bool)}
        for group, m in masks.items():
            mc, mz = m & ~zk, m & zk
            for s, g, gene, name in in_joint:
                d = joint.detected[:, joint.var_index(name)]
                det_rows.append({"label_column": col, "group": group, "source": s, "panel_group": g, "gene": gene,
                                 "joint_var_name": name, **lib.proportion_compare(
                                     int(d[mc].sum()), int(mc.sum()), int(d[mz].sum()), int(mz.sum())),
                                 "p_value_note": lib.P_NOTE})
            for (s, g), names in group_sets.items():
                d = joint.detected[:, [joint.var_index(n) for n in names]].any(axis=1)
                det_rows.append({"label_column": col, "group": group, "source": s, "panel_group": g,
                                 "gene": f"any of {'/'.join(names)}", "joint_var_name": ";".join(names),
                                 **lib.proportion_compare(int(d[mc].sum()), int(mc.sum()), int(d[mz].sum()),
                                                          int(mz.sum())), "p_value_note": lib.P_NOTE})
    detection = pd.DataFrame(det_rows)
    run.table(detection, "contamination_detection")
    checks.add("contamination genes in the matched set", f"{len(in_joint)} of {len(genes)}: "
               + ", ".join(gene for _, _, gene, _ in in_joint), None)
    missing = coverage.loc[~coverage["compared_between_conditions"], "gene"].tolist()
    checks.add("contamination genes not compared (coverage table only)", ", ".join(missing) or "none", None)

    run.table(checks.frame(), "composition_checks")
    run.finish(parameters={"label_columns": list(lib.LABEL_COLUMNS), "ci": "Wilson (fractions); Newcombe hybrid "
                           "score (differences)", "p_value_note": lib.P_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_response_core_composition_checks.csv.")


if __name__ == "__main__":
    main()
