#!/usr/bin/env python
"""dct_prolif steps 0-1: freeze check, input SHA256, package versions; depth floors, depth bins and the coverage
tables for the proliferation set P and the identity markers, per condition and run.

Analysis plan
(frozen; SHA256 in results/tables/dct_prolif_frozen_brief_sha256.csv, commit 9f6adb9). Runs: 'frozen' (floor 1,000
protein IDs, as the plan) and 'nofloor' (approved change, authors 3 Nov 2026: only the >= 100-ID QC filter). Floors 500
and 1,500 are information only. No label is made here.

Inputs (read-only): results/h5ad/dct_intake_DCT_{ctrl,zerok}.h5ad; data/DCT/DCT_markers_V1.xlsx; the dct_labels canonical table,
dct_labels labelled objects and dct_reviewer task-3 objects (hashed here for the record; read in later steps).
Outputs: results/tables/dct_prolif/dct_prolif_{freeze_check,inputs,packages,depth_floor,depth_bins,
prolif_set,identity_markers,null_pool,intake_checks}.csv.

Command (project root):  python scripts/dct_prolif_intake.py
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
import subprocess
from importlib import metadata

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dct_intake_lib as t26
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_intake.py"


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=cc.PROJECT_ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif steps 0-1: freeze check, inputs, floors, bins, coverage.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "intake", args)
    checks = t5.Checks("dct_combined")

    # 0. Freeze check -------------------------------------------------------------------------------------------
    row = lib.brief_hash_row()
    freeze = pd.DataFrame([{**row}])
    if not checks.ok:
        lib.stop(run, checks, "intake_checks")

    # Inputs of every dct_prolif step (hashed here; each step records its own reads in its manifest)
    inputs = [(lib.OBJECTS["Ctrl"], "dct_intake object (Ctrl)"), (lib.OBJECTS["ZeroK"], "dct_intake object (ZeroK)"),
              (t26.MARKER_WORKBOOK, "DCT marker workbook (Supp Table 1 DCT tab)"),
              (lib.DCT_LABELS_LABELS, "dct_labels canonical labels (comparison and bridge only)"),
              *[(lib.DCT_LABELS_H5AD[c], f"dct_labels labelled object ({c}; UMAP and 20-PC graph embedding only)") for c in lib.CONDITIONS],
              *[(lib.DCT_REVIEWER_TASK3_H5AD[c], f"dct_reviewer task-3 object ({c}; contamination flags and cell-cycle score only)")
                for c in lib.CONDITIONS]]
    in_rows = []
    for path, role in inputs:
        run.use_input(path, role)
        in_rows.append({"path": cc.rel(path), "role": role, "bytes": path.stat().st_size, "sha256": cc.sha256(path)})
    for r in in_rows:
        checks.add(f"input exists: {r['path']}", f"{r['bytes']} bytes", True)

    pk = cc.package_versions()
    for name in ("harmonypy", "scikit-learn", "statsmodels"):
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = "not installed"
    packages = pd.DataFrame([{"package": k, "version": v} for k, v in pk.items()])
    checks.add("harmonypy version (installed for rna_protein_integration, author-approved)", pk["harmonypy"], pk["harmonypy"] == "0.0.10")

    # 1. Objects, floors, bins ----------------------------------------------------------------------------------
    conds = {c: lib.load_condition(run, c) for c in lib.CONDITIONS}
    workbook_candidates = {}
    floor_rows, bin_rows, marker_rows, pool_rows = [], [], [], []
    for cond, c in conds.items():
        n = len(c.cell_ids)
        checks.add(f"{cond}: cells in the dct_intake object", n, n == {"Ctrl": 729, "ZeroK": 702}[cond])
        same = np.array_equal(c.detected.sum(axis=1), c.ids.astype(int))
        checks.add(f"{cond}: obs n_detected equals detected retained proteins per cell", f"{same}", same)
        checks.add(f"{cond}: plates", ";".join(sorted(set(c.plate))), set(c.plate) == {"P1", "P2"})
        checks.add(f"{cond}: minimum protein IDs per cell (no-floor run labels every cell when >= 100)",
                   int(c.ids.min()), c.ids.min() >= lib.RUNS["nofloor"].floor)
        for floor in sorted({*[r.floor for r in lib.RUNS.values()], *lib.INFO_FLOORS}):
            for plate in ("all", "P1", "P2"):
                m = np.ones(n, bool) if plate == "all" else c.plate == plate
                below = m & (c.ids < floor)
                floor_rows.append({"condition": cond, "floor": floor, "plate": plate, "n_cells": int(m.sum()),
                                   "n_below": int(below.sum()), "n_above": int((m & ~below).sum()),
                                   "fraction_below": float(below.sum() / m.sum()),
                                   "median_ids_below": float(np.median(c.ids[below])) if below.any() else np.nan,
                                   "median_ids_above": float(np.median(c.ids[m & ~below])),
                                   "role": ("frozen primary" if floor == 1000 else "approved no-floor run (QC filter)"
                                            if floor == 100 else "information only")})
            if floor in lib.EXPECTED_BELOW[cond]:
                nb = int((c.ids < floor).sum())
                exp = lib.EXPECTED_BELOW[cond][floor]
                checks.add(f"{cond}: cells below {floor} protein IDs equal the plan", f"{nb} (plan {exp})", nb == exp)
        candidates, _ = lib.identity_candidates(run, c.var_names)
        workbook_candidates[cond] = candidates
    if not checks.ok:
        lib.stop(run, checks, "intake_checks")

    P, pm = lib.prolif_set(conds["Ctrl"], conds["ZeroK"])
    checks.add("proliferation set P: Tirosh S + G2M retained in both conditions, minus Mki67 and Top2a",
               f"{len(P)} (plan: expected 36)", len(P) == lib.EXPECTED_P)
    held = pm.loc[pm["held_out"], "matched_name_ctrl"].tolist()
    checks.add("held-out validation proteins retained in both conditions", ";".join(held),
               sorted(held) == sorted(lib.HELD_OUT) and pm.loc[pm["held_out"], "retained_both"].all())
    if not checks.ok:
        lib.stop(run, checks, "intake_checks")

    for run_key, spec in lib.RUNS.items():
        for cond, c in conds.items():
            pr = lib.prolif_setup(c, spec.floor, P)
            a = pr.above
            ids_a = c.ids[a]
            for b in range(lib.N_BINS):
                m = pr.bins == b
                bin_rows.append({"run": run_key, "condition": cond, "floor": spec.floor, "bin": b, "n_cells": int(m.sum()),
                                 "ids_min": int(ids_a[m].min()), "ids_median": float(np.median(ids_a[m])),
                                 "ids_max": int(ids_a[m].max()),
                                 "mean_P_detected": float(c.detected[a][m][:, pr.P_idx].sum(axis=1).mean()),
                                 "mean_P_detection_rate": float(pr.rates[b, pr.P_idx].mean())})
            sizes = pd.Series(pr.bins).value_counts()
            checks.add(f"{run_key} / {cond}: 20 depth bins of equal count (cells above floor {a.sum()})",
                       f"sizes {sizes.min()}-{sizes.max()}", sizes.size == lib.N_BINS and sizes.max() - sizes.min() <= 1)
            overlap = set(pr.P_idx) & set(pr.pool_idx)
            checks.add(f"{run_key} / {cond}: no P protein and no Tirosh/18-panel protein in the null pool",
                       f"pool {len(pr.pool_idx)} proteins; overlap with P {len(overlap)}", not overlap)
            for d in range(10):
                in_d = pr.decile == d
                pool_rows.append({"run": run_key, "condition": cond, "floor": spec.floor, "decile": d,
                                  "n_proteins": int(in_d.sum()), "n_pool": int(in_d[pr.pool_idx].sum()),
                                  "n_P": int(pr.need.get(d, 0)), "freq_min": float(pr.freq[in_d].min()),
                                  "freq_max": float(pr.freq[in_d].max())})
                if pr.need.get(d, 0):
                    ok = in_d[pr.pool_idx].sum() >= pr.need[d]
                    if not ok:
                        checks.add(f"{run_key} / {cond}: decile {d} pool large enough", f"{in_d[pr.pool_idx].sum()} < {pr.need[d]}", False)
            checks.add(f"{run_key} / {cond}: P proteins per detection decile",
                       "; ".join(f"d{d} {k}" for d, k in sorted(pr.need.items())), None)
            pm[f"n_detected_{lib.KEY[cond]}_all"] = pm["matched_name_ctrl"].map(
                lambda g: int(c.detected[:, c.var_names.get_loc(g)].sum()) if g in c.var_names else np.nan)
            pm[f"n_detected_{lib.KEY[cond]}_{run_key}"] = pm["matched_name_ctrl"].map(
                lambda g: int(c.detected[a, c.var_names.get_loc(g)].sum()) if g in c.var_names else np.nan)
            pm[f"decile_{lib.KEY[cond]}_{run_key}"] = pm["matched_name_ctrl"].map(
                lambda g: int(pr.decile[c.var_names.get_loc(g)]) if g in c.var_names else np.nan)
            used, mtab = lib.identity_markers(c, a, workbook_candidates[cond])
            mtab.insert(0, "condition", cond)
            mtab.insert(0, "run", run_key)
            mtab["n_cells_detected_all"] = mtab["marker"].map(lambda g: int(c.detected[:, c.var_names.get_loc(g)].sum()))
            marker_rows.append(mtab)
            exp = {t: sorted(v) for t, v in lib.EXPECTED_MARKERS.items()}
            got = {t: sorted(v) for t, v in used.items()}
            checks.add(f"{run_key} / {cond}: identity markers used (DCT1 | DCT2)",
                       f"{';'.join(used['DCT1'])} | {';'.join(used['DCT2'])}"
                       + ("" if got == exp else " (differs from the plan's expected list)"), None)
            for t in (lib.DCT1, lib.DCT2):
                checks.add(f"{run_key} / {cond}: at least one {t} marker used", len(used[t]), len(used[t]) >= 1)
            trpv5 = mtab.loc[mtab["marker"] == "Trpv5", "n_cells_detected_above_floor"]
            if len(trpv5):
                checks.add(f"{run_key} / {cond}: Trpv5 cells detected above floor (plan: expected < 20)",
                           int(trpv5.iloc[0]), None)

    run.table(freeze, "freeze_check")
    run.table(pd.DataFrame(in_rows), "inputs")
    run.table(packages, "packages")
    run.table(pd.DataFrame(floor_rows), "depth_floor")
    run.table(pd.DataFrame(bin_rows), "depth_bins")
    pm = pm.rename(columns={"set": "phase", "gene": "tirosh_gene", "aliases": "tirosh_aliases"})
    run.table(pm, "prolif_set")
    run.table(pd.concat(marker_rows, ignore_index=True), "identity_markers")
    run.table(pd.DataFrame(pool_rows), "null_pool")
    run.table(checks.frame(), "intake_checks")
    run.finish({"runs": {k: vars(v) for k, v in lib.RUNS.items()}, "info_floors": list(lib.INFO_FLOORS),
                "approved_change": lib.APPROVED_CHANGE, "P": P, "n_bins": lib.N_BINS,
                "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_intake_checks.csv.")


if __name__ == "__main__":
    main()
