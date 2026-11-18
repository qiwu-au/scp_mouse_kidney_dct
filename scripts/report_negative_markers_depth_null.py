#!/usr/bin/env python
"""The response to the reviewers: depth-matched random removals for the Slc12a1 / Aqp2 exclusion check.

Companion of report_negative_markers_exclusion.py.
The cells with Slc12a1 or Aqp2 detected have far more identified proteins than the other cells, so random removals
drawn within the primary label only (the dct_reviewer addendum E control used in the main script) are not matched on protein
depth. Here the 100 random removals are drawn within primary label x quartile of protein IDs per cell (quartiles
within condition), so each random set has the same number of cells per label and per depth quartile as the
marker-detected set. Everything else is the main script's: the same three exclusion sets, the same labelling
(dct_reviewer_exclusion_sensitivity.evaluate), Leiden seed 0 for the comparison, numpy default_rng(42) with one stream
per condition. Within condition only; nothing compares the conditions.

Inputs (read-only): dct_labels labelled objects and canonical labels; data/DCT/DCT_markers_V1.xlsx;
  results/tables/report/report_09_negative_marker_exclusion_runs.csv (check of the observed rows).
Outputs (results/tables/report/): report_09_negative_marker_depth_null_{runs,vs_random,checks}.csv;
  report_step_09_negative_marker_depth_null_manifest.json

Command (project root, about 25 min):  python scripts/report_negative_markers_depth_null.py
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
import qc_mito_sensitivity as t3s
import cluster_de_lib as t5
import labels_core_lib as lib18
import dct_reviewer_lib as lib
import dct_reviewer_exclusion_sensitivity as ex
import report_negative_markers_exclusion as nm

TASK = "report"
STEP = "09_negative_marker_depth_null"
P = "09_negative_marker_depth_null"
S = nm.S
MAIN_RUNS = cc.out_path(TASK, "tables", "09_negative_marker_exclusion_runs", ".csv")


def one_condition(run: cc.Run, checks: t5.Checks, cond: str, n_draws: int, t0: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = lib.SETTINGS[cond]
    rng = np.random.default_rng(cc.RANDOM_SEED)
    adata, labels = lib.load(run, cond)
    det = nm.detection(cond, adata, labels)
    lists = lib18.present_lists(lib18.read_workbook(run), adata.var_names)
    markers = lib18.dct_markers(run, adata.var_names)
    ctx = ex.Context(s, adata, labels, lists, markers)
    label = labels[s.PRIMARY].astype(str).to_numpy()
    quartile = det["protein_id_quartile"].to_numpy()
    strata = np.array([f"{a}|{b}" for a, b in zip(label, quartile)])
    ids = det["n_detected_proteins"].to_numpy(float)

    base, _ = ex.evaluate(ctx, np.zeros(len(label), bool), (S,))
    rows = [{"condition": cond, "run_type": "baseline", "exclusion_set": "none", "draw": "", **base}]
    main = pd.read_csv(run.use_input(MAIN_RUNS, "the response to the reviewers main exclusion runs"))
    main = main[(main["condition"] == cond) & (main["run_type"] == "observed")].set_index("exclusion_set")
    for exclusion, key in nm.EXCLUSIONS.items():
        mask = det[f"det_{key}"].to_numpy()
        if not mask.any():
            continue
        res, _ = ex.evaluate(ctx, mask, (S,))
        rows.append({"condition": cond, "run_type": "observed", "exclusion_set": exclusion, "draw": "",
                     "median_protein_ids_removed": float(np.median(ids[mask])), **res})
        diff = max(abs(float(res[o]) - float(main.loc[exclusion, o])) for o in nm.OUTCOMES)
        checks.add(f"{cond} {exclusion}: observed outcomes equal the main script's table (max abs diff)",
                   f"{diff:.1e}", diff < 1e-12, "< 1e-12")
        by_q = pd.Series(quartile[mask]).value_counts().reindex(["Q1", "Q2", "Q3", "Q4"]).fillna(0).astype(int)
        checks.add(f"{cond} {exclusion}: removed cells by protein-ID quartile (Q1-Q4)",
                   "; ".join(f"{q} {v}" for q, v in by_q.items()), None)
        medians = []
        for draw in range(n_draws):
            removed = t3s.stratified_draw(rng, strata, mask)
            res, _ = ex.evaluate(ctx, removed, (S,))
            medians.append(float(np.median(ids[removed])))
            rows.append({"condition": cond, "run_type": "random", "exclusion_set": exclusion, "draw": draw,
                         "n_set_cells_among_removed": int((removed & mask).sum()),
                         "median_protein_ids_removed": medians[-1], **res})
            if (draw + 1) % 20 == 0:
                print(f"  {cond} {exclusion}: {draw + 1} depth-matched draws ({time.time() - t0:.0f} s)", flush=True)
        checks.add(f"{cond} {exclusion}: median protein IDs of the removed cells, marker set vs depth-matched draws "
                   "(median of the draw medians)", f"{np.median(ids[mask]):.0f} vs {np.median(medians):.0f}", None)
        run.note(f"{cond} {exclusion}: {n_draws} depth-matched random removals done ({time.time() - t0:.0f} s).")
    runs = pd.DataFrame(rows)
    summary = nm.vs_random(cond, runs)
    summary.insert(2, "null_type", "random removals within primary label x protein-ID quartile")
    return runs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: depth-matched random removals.")
    cc.add_common_args(parser)
    parser.add_argument("--n-draws", type=int, default=nm.N_DRAWS)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    t0 = time.time()
    out = [one_condition(run, checks, cond, args.n_draws, t0) for cond in lib.CONDITIONS]
    run.table(pd.concat([o[0] for o in out], ignore_index=True), f"{P}_runs")
    run.table(pd.concat([o[1] for o in out], ignore_index=True), f"{P}_vs_random")
    run.table(checks.frame(), f"{P}_checks")
    run.finish({"request": "the response to the reviewers exclusion check (authors, 11 Nov 2026: 'Run the check'); depth-matched control",
                "strata": "primary label x quartile of protein IDs per cell (within condition)",
                "exclusion_sets": nm.EXCLUSIONS, "random_leiden_seed": S, "n_random_draws": args.n_draws,
                "design": lib.DESIGN_NOTE, "checks_pass": checks.ok, "runtime_s": round(time.time() - t0)},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{P}_checks.csv.")


if __name__ == "__main__":
    main()
