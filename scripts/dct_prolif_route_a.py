#!/usr/bin/env python
"""dct_prolif steps 2-3: route A per condition and run. Proliferation (detection part D against a per-cell
matched null, BH q, cycling, ProLIF; the intensity part reported only) and the missing-aware DCT1/DCT2 axis with a
two-component Gaussian mixture. Floors 500 and 1,500 are run as information only and compared with both runs.

Analysis plan
(frozen), "Fixed settings" and steps 2-3. Runs: 'frozen' (floor 1,000) and 'nofloor' (approved change, authors
3 Nov 2026: only the >= 100-ID QC filter). Each condition alone; nothing compares the conditions.

Inputs (read-only): results/h5ad/dct_intake_DCT_{ctrl,zerok}.h5ad; data/DCT/DCT_markers_V1.xlsx; dct_prolif_prolif_set.csv.
Outputs (results/tables/dct_prolif/): dct_prolif_route_a_per_cell.csv, dct_prolif_route_a_info_floors_per_cell.csv,
dct_prolif_route_a_summary.csv, dct_prolif_mixture.csv, dct_prolif_prolif_intensity_fits.csv,
dct_prolif_null_sets_seed42.csv, dct_prolif_floor_comparison.csv, dct_prolif_floor_crosstabs.csv,
dct_prolif_route_a_checks.csv.

Command (project root):  python scripts/dct_prolif_route_a.py
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
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_route_a.py"


def summary_row(run_key: str, floor: int, cond: str, out: dict) -> dict:
    f = out["frame"]
    a = f["above_floor"].to_numpy(bool)
    mix = out["mixture"]
    p_min = 1 / (lib.N_NULL + 1)
    return {"run": run_key, "condition": cond, "floor": floor, "n_cells": len(f), "n_above_floor": int(a.sum()),
            "n_below_floor": int((~a).sum()), "label_counts": lib.counts_text(f["label"]),
            "identity_counts": lib.counts_text(f["identity"], lib.IDENTITY_ORDER),
            "cycling_counts": lib.counts_text(f["cycling"], lib.CYCLING_ORDER),
            "n_cycling": int((f["cycling"] == lib.CYCLING).sum()), "n_prolif": int((f["label"] == lib.PROLIF).sum()),
            "n_cycling_ncc_not_detected": int((f["label"] == lib.CYCLING_NO_NCC).sum()),
            "n_p_at_floor": int(np.isclose(f.loc[a, "prolif_p"], p_min).sum()),
            "min_p": float(f.loc[a, "prolif_p"].min()), "min_q": float(f.loc[a, "prolif_q"].min()),
            "median_D": float(f.loc[a, "prolif_D"].median()),
            "n_intensity_part_defined": int(f.loc[a, "prolif_intensity"].notna().sum()),
            "n_axis_defined": int(f.loc[a, "axis"].notna().sum()),
            "bic_one_component": mix["bic_1"], "bic_two_components": mix["bic_2"],
            "one_component_lower_bic": mix["bic_1"] < mix["bic_2"],
            "markers_DCT1": ";".join(out["markers_used"][lib.DCT1]), "markers_DCT2": ";".join(out["markers_used"][lib.DCT2])}


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif steps 2-3: route A proliferation and identity.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "route_a", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "route_a_checks")

    conds = {c: lib.load_condition(run, c) for c in lib.CONDITIONS}
    P, _ = lib.prolif_set(conds["Ctrl"], conds["ZeroK"])
    pset = lib.read_table(run, "prolif_set")
    listed = pset.loc[pset["in_P"].astype(str).str.lower() == "true", "matched_name_ctrl"].tolist()
    checks.add("P equals the intake list (dct_prolif_prolif_set.csv)", f"{len(P)} proteins", listed == P)
    cands = {c: lib.identity_candidates(run, conds[c].var_names)[0] for c in lib.CONDITIONS}

    per_cell, info_cells, summary, mix_rows, fit_rows, null_rows = [], [], [], [], [], []
    results: dict[tuple[int, str], pd.DataFrame] = {}
    floors = {**{spec.floor: k for k, spec in lib.RUNS.items()}, **{f: f"info_{f}" for f in lib.INFO_FLOORS}}
    for floor, run_key in sorted(floors.items()):
        for cond, c in conds.items():
            t0 = time.time()
            out = lib.route_a(c, floor, P, cands[cond])
            f = out["frame"]
            f.insert(0, "condition", cond)
            f.insert(0, "floor", floor)
            f.insert(0, "run", run_key)
            results[(floor, cond)] = f
            (per_cell if run_key in lib.RUNS else info_cells).append(f)
            summary.append(summary_row(run_key, floor, cond, out))
            mix_rows.append({"run": run_key, "condition": cond, "floor": floor,
                             **{k: v for k, v in out["mixture"].items() if not isinstance(v, np.ndarray)}})
            a = f["above_floor"].to_numpy(bool)
            print(f"[{run_key} / {cond}] floor {floor}: {lib.counts_text(f['label'])}; {time.time() - t0:.1f} s", flush=True)

            if run_key not in lib.RUNS:
                continue
            fits = out["fits"].copy()
            fits.insert(0, "condition", cond)
            fits.insert(0, "run", run_key)
            fit_rows.append(fits)
            names = c.var_names.to_numpy()
            null_rows.append(pd.DataFrame({"run": run_key, "condition": cond, "set": np.arange(lib.N_NULL),
                                           "proteins": [";".join(names[s]) for s in out["cycling"]["null_sets"]]}))
            # Checks -------------------------------------------------------------------------------------------
            tag = f"{run_key} / {cond}"
            checks.add(f"{tag}: 'Below depth floor' exactly for cells with n_detected < {floor}",
                       int((f["label"] == lib.BELOW).sum()),
                       np.array_equal(f["label"].eq(lib.BELOW).to_numpy(), f["n_detected"].to_numpy() < floor))
            checks.add(f"{tag}: D, p and q finite for every cell above the floor", int(a.sum()),
                       bool(np.isfinite(f.loc[a, ["prolif_D", "prolif_p", "prolif_q"]].to_numpy()).all()))
            pmin = f.loc[a, "prolif_p"].min()
            checks.add(f"{tag}: empirical p within [1/1001, 1]", f"min {pmin:.5f}",
                       bool(pmin >= 1 / 1001 - 1e-15 and f.loc[a, "prolif_p"].max() <= 1))
            again = lib.cycling_call(out["prolif"], out["prolif"].P_idx, np.random.default_rng(lib.SEED))
            checks.add(f"{tag}: null with seed 42 reproduces the p values exactly", "rerun",
                       np.array_equal(again["p"], f.loc[a, "prolif_p"].to_numpy()))
            pro = f["label"] == lib.PROLIF
            ok = ((f.loc[pro, "cycling"] == lib.CYCLING) & f.loc[pro, "slc12a3_detected"]).all()
            checks.add(f"{tag}: every ProLIF cell is cycling with Slc12a3 detected", int(pro.sum()), bool(ok))
            cn = f["label"] == lib.CYCLING_NO_NCC
            ok = ((f.loc[cn, "cycling"] == lib.CYCLING) & ~f.loc[cn, "slc12a3_detected"]).all()
            checks.add(f"{tag}: every 'Cycling, NCC not detected' cell is cycling without Slc12a3", int(cn.sum()), bool(ok))
            non = (f["cycling"] == lib.NON_CYCLING)
            checks.add(f"{tag}: non-cycling cells keep their identity call as label", int(non.sum()),
                       bool((f.loc[non, "label"] == f.loc[non, "identity"]).all()))
            undefined = a & f["axis"].isna().to_numpy()
            checks.add(f"{tag}: 'Unknown (markers not detected)' exactly when axis A is undefined above the floor",
                       int(undefined.sum()), np.array_equal(undefined, (f["identity"] == lib.UNKNOWN_MARKERS).to_numpy()))
            post = f.loc[a & ~undefined, "posterior"]
            checks.add(f"{tag}: posterior within [0.5, 1] where the axis is defined", f"min {post.min():.3f}",
                       bool(post.between(0.5 - 1e-12, 1 + 1e-12).all()))
            mix = out["mixture"]
            checks.add(f"{tag}: mixture converged; DCT1 component has the lower mean",
                       f"means DCT1 {mix['mean_dct1']:.3f}, DCT2 {mix['mean_dct2']:.3f}",
                       mix["converged"] and mix["mean_dct1"] < mix["mean_dct2"])
            checks.add(f"{tag}: BIC one vs two components",
                       f"{mix['bic_1']:.1f} vs {mix['bic_2']:.1f}"
                       + ("; one component has the lower BIC (calls kept, plan)" if mix["bic_1"] < mix["bic_2"] else ""),
                       None)
            checks.add(f"{tag}: label counts", lib.counts_text(f["label"]), None)
            checks.add(f"{tag}: cells at the p floor 1/1001", summary[-1]["n_p_at_floor"], None)

    per_cell = pd.concat(per_cell, ignore_index=True)
    info_cells = pd.concat(info_cells, ignore_index=True)

    # Floor comparison (information only): every floor against the frozen 1,000 run and the no-floor run --------
    comp_rows, ct_rows = [], []
    for cond in lib.CONDITIONS:
        for ref_floor in (lib.RUNS["frozen"].floor, lib.RUNS["nofloor"].floor):
            ref = results[(ref_floor, cond)]
            for floor in sorted(floors):
                if floor == ref_floor:
                    continue
                other = results[(floor, cond)]
                both = ref["above_floor"].to_numpy(bool) & other["above_floor"].to_numpy(bool)
                a_lab, b_lab = ref.loc[both, "label"].to_numpy(), other.loc[both, "label"].to_numpy()
                ag = lib.agreement(a_lab, b_lab)
                comp_rows.append({"condition": cond, "reference_floor": ref_floor, "reference_run": floors[ref_floor],
                                  "floor": floor, "run": floors[floor], "n_cells_above_both": int(both.sum()),
                                  "label_counts_reference": lib.counts_text(ref["label"]),
                                  "label_counts_floor": lib.counts_text(other["label"]),
                                  "n_prolif_reference": int((ref["label"] == lib.PROLIF).sum()),
                                  "n_prolif_floor": int((other["label"] == lib.PROLIF).sum()),
                                  "prolif_jaccard_cells_above_both": lib.jaccard(a_lab == lib.PROLIF, b_lab == lib.PROLIF),
                                  "n_prolif_both": int(((a_lab == lib.PROLIF) & (b_lab == lib.PROLIF)).sum()),
                                  "cycling_jaccard_cells_above_both": lib.jaccard(
                                      ref.loc[both, "cycling"].eq(lib.CYCLING).to_numpy(),
                                      other.loc[both, "cycling"].eq(lib.CYCLING).to_numpy()),
                                  "label_fraction_equal": ag["fraction_equal"], "label_ari": ag["ari"]})
                ct_rows += lib.crosstab_long(a_lab, b_lab, "label_reference", "label_floor", condition=cond,
                                             reference_floor=ref_floor, floor=floor)
    comp = pd.DataFrame(comp_rows)
    for r in comp.itertuples():
        checks.add(f"{r.condition}: floor {r.floor} vs {r.reference_run} ({r.reference_floor}): ProLIF n; Jaccard",
                   f"{r.n_prolif_floor} vs {r.n_prolif_reference}; {r.prolif_jaccard_cells_above_both:.3f}", None)

    run.table(per_cell, "route_a_per_cell")
    run.table(info_cells, "route_a_info_floors_per_cell")
    run.table(pd.DataFrame(summary), "route_a_summary")
    run.table(pd.DataFrame(mix_rows), "mixture")
    run.table(pd.concat(fit_rows, ignore_index=True), "prolif_intensity_fits")
    run.table(pd.concat(null_rows, ignore_index=True), "null_sets_seed42")
    run.table(comp, "floor_comparison")
    run.table(pd.DataFrame(ct_rows), "floor_crosstabs")
    run.table(checks.frame(), "route_a_checks")
    run.finish({"runs": {k: vars(v) for k, v in lib.RUNS.items()}, "info_floors": list(lib.INFO_FLOORS),
                "approved_change": lib.APPROVED_CHANGE, "P": P, "n_null": lib.N_NULL, "null_seed": lib.SEED,
                "q_cycling": lib.Q_CYCLING, "posterior_min": lib.POSTERIOR_MIN, "gmm_n_init": lib.GMM_N_INIT,
                "min_cells_intensity": lib.MIN_CELLS_INTENSITY, "min_cells_marker": lib.MIN_CELLS_MARKER,
                "tie_tolerance": lib.TIE_TOL, "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_route_a_checks.csv.")


if __name__ == "__main__":
    main()
