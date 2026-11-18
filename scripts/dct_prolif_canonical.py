#!/usr/bin/env python
"""dct_prolif step 8: canonical dct_prolif label table and labelled objects (both runs, both conditions).

Analysis plan
(frozen), step 8. Runs: 'frozen' (floor 1,000; prefixes cs_t31 / cs_t31b; kept for comparison) and 'nofloor'
(approved change, authors 3 Nov 2026; prefixes cs_t31nf / cs_t31nfb; dct_prolif primary run since 3 Nov 2026, authors: 'Primary
everywhere'). t31_label = the route-A label of the primary run (cs_t31nf_label), with t31_label_source. dct_labels labels
stay the DCT primary until the authors adopt dct_prolif.

dct_prolif_dct_labels.csv: one row per cell of both conditions (1,431), matched to the dct_labels canonical table on
condition + cell_key (cell_id checked): cell_id, cell_key, condition, plate, n_detected; slc12a3_detected and
mki67_or_top2a_detected (equal in both runs); per run, route A <prefix>_{label, identity, cycling, above_floor,
depth_bin, n_P_detected, prolif_D, prolif_null_mean, prolif_null_sd, prolif_p, prolif_q, prolif_intensity,
DCT1_score, DCT2_score, axis, n_DCT1_markers_detected, n_DCT2_markers_detected, posterior, posterior_dct2,
identity_noise_fraction_same} and route B <prefix b>_{label, cluster, confidence}; t31_label and t31_label_source
(dct_prolif primary run); dct_labels dct_label and dct_label_alternative for reference.
results/h5ad/dct_prolif_DCT_{ctrl,zerok}_labelled.h5ad: the dct_labels labelled object of the condition (X, layers, UMAP and
dct_labels labels unchanged) with the dct_prolif columns added to obs and uns['t31'] (runs, approved change, P, identity markers).

Inputs (read-only): dct_prolif_{route_a_per_cell,route_b_per_cell,noise_identity_per_cell,route_a_summary,
prolif_set,identity_markers}.csv; dct_labels canonical labels and labelled objects.

Command (project root):  python scripts/dct_prolif_canonical.py
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
import json

import anndata as ad
import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dct_reviewer_lib as t28
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_canonical.py"
ROUTE_A_COLUMNS = ("label", "identity", "cycling", "above_floor", "depth_bin", "n_P_detected", "prolif_D",
                   "prolif_null_mean", "prolif_null_sd", "prolif_p", "prolif_q", "prolif_intensity", "DCT1_score",
                   "DCT2_score", "axis", "n_DCT1_markers_detected", "n_DCT2_markers_detected", "posterior",
                   "posterior_dct2")
ROUTE_B_COLUMNS = ("label", "cluster", "confidence")
SHARED = ("slc12a3_detected", "mki67_or_top2a_detected")
BASE = ("cell_id", "cell_key", "condition", "plate", "n_detected")
EXPECTED_DCT_LABEL = {"Ctrl": {"DCT1": 382, "DCT2": 85, "Unknown": 262}, "ZeroK": {"DCT1": 449, "Unknown": 253}}


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def same_values(a: pd.Series, b: pd.Series) -> bool:
    if pd.api.types.is_float_dtype(a) or pd.api.types.is_float_dtype(b):
        return bool(np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=0, atol=1e-12, equal_nan=True))
    return bool(np.array_equal(a.astype(str).to_numpy(), b.astype(str).to_numpy()))


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif step 8: canonical labels and labelled objects.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "canonical", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "canonical_checks")

    per_a = lib.read_table(run, "route_a_per_cell", dtype={"cell_id": str, "cell_key": str})
    per_b = lib.read_table(run, "route_b_per_cell", dtype={"cell_id": str, "cell_key": str, "cluster": str})
    noise = lib.read_table(run, "noise_identity_per_cell", dtype={"cell_id": str})
    summary = lib.read_table(run, "route_a_summary")
    pset = lib.read_table(run, "prolif_set")
    mtab = lib.read_table(run, "identity_markers")
    canon = t28.read_canonical(run)
    for frame in (per_a, per_b):
        frame["above_floor"] = as_bool(frame["above_floor"])
    for c in SHARED:
        per_a[c] = as_bool(per_a[c])

    parts = []
    for cond in lib.CONDITIONS:
        ref = canon[canon["condition"] == cond].reset_index(drop=True)
        out = ref[list(BASE)].copy()
        keys = out["cell_key"]
        for run_key in lib.RUN_KEYS:
            a = per_a[(per_a["run"] == run_key) & (per_a["condition"] == cond)].set_index("cell_key")
            b = per_b[(per_b["run"] == run_key) & (per_b["condition"] == cond)].set_index("cell_key")
            n = noise[(noise["run"] == run_key) & (noise["condition"] == cond)].set_index("cell_id")
            ok = (a.index.is_unique and b.index.is_unique and set(a.index) == set(keys) and set(b.index) == set(keys)
                  and (a.loc[keys, "cell_id"].to_numpy() == out["cell_id"].to_numpy()).all()
                  and (b.loc[keys, "cell_id"].to_numpy() == out["cell_id"].to_numpy()).all()
                  and (a.loc[keys, "n_detected"].to_numpy() == out["n_detected"].to_numpy()).all())
            checks.add(f"{run_key} / {cond}: route A and B tables match the dct_labels cells on cell_key (cell_id, n_detected)",
                       f"{len(a)} / {len(b)} / {len(out)}", bool(ok))
            if not ok:
                lib.stop(run, checks, "canonical_checks")
            a = a.loc[keys]
            for c in ROUTE_A_COLUMNS:
                out[lib.col(run_key, c)] = a[c].to_numpy()
            above = a["above_floor"].to_numpy(bool)
            frac = n["fraction_repeats_same"].reindex(out["cell_id"]).to_numpy(float)
            checks.add(f"{run_key} / {cond}: identity noise result present for every cell above the floor",
                       f"{int(np.isfinite(frac[above]).sum())} of {int(above.sum())}",
                       bool(np.isfinite(frac[above]).all() and not np.isfinite(frac[~above]).any()))
            out[lib.col(run_key, "identity_noise_fraction_same")] = frac
            for c in ROUTE_B_COLUMNS:
                out[lib.col(run_key, c, "b")] = b.loc[keys, c].astype(str).to_numpy()
            if run_key == lib.RUN_KEYS[0]:
                for c in SHARED:
                    out[c] = a[c].to_numpy(bool)
            else:
                same = all((a[c].to_numpy(bool) == out[c].to_numpy(bool)).all() for c in SHARED)
                checks.add(f"{cond}: Slc12a3 and Mki67/Top2a detection equal in both runs", str(same), bool(same))
            srow = summary[(summary["run"] == run_key) & (summary["condition"] == cond)].iloc[0]
            obs_counts = lib.counts_text(out[lib.col(run_key, "label")])
            exp_counts = srow["label_counts"]
            checks.add(f"{run_key} / {cond}: {lib.col(run_key, 'label')} counts equal the route-A summary",
                       obs_counts, obs_counts == exp_counts, exp_counts)
            checks.add(f"{run_key} / {cond}: {lib.col(run_key, 'label', 'b')} counts", lib.counts_text(
                out[lib.col(run_key, "label", "b")], lib.unique_order(lib.LABEL_ORDER, cc.DCT_LABEL_ORDER)), None)
        # dct_prolif primary label (authors, 3 Nov 2026: no-floor run primary)
        pcol = lib.col(lib.PRIMARY_RUN, "label")
        out["t31_label"] = out[pcol].astype(str).to_numpy()
        out["t31_label_source"] = f"{pcol} (dct_prolif primary run '{lib.PRIMARY_RUN}', {lib.RUNS[lib.PRIMARY_RUN].display})"
        srow = summary[(summary["run"] == lib.PRIMARY_RUN) & (summary["condition"] == cond)].iloc[0]
        checks.add(f"{cond}: t31_label equals {pcol} cell by cell; counts equal the route-A summary of the primary run",
                   lib.counts_text(out["t31_label"]), bool(np.array_equal(out["t31_label"].to_numpy(),
                                                                          out[pcol].astype(str).to_numpy()))
                   and lib.counts_text(out["t31_label"]) == srow["label_counts"], srow["label_counts"])
        out["dct_label"] = ref["dct_label"].astype(str).to_numpy()
        out["dct_label_alternative"] = ref["dct_label_alternative"].astype(str).to_numpy()
        obs = out["dct_label"].value_counts().to_dict()
        checks.add(f"{cond}: dct_labels dct_label counts equal the project notes", lib.counts_text(out["dct_label"], cc.DCT_LABEL_ORDER),
                   obs == EXPECTED_DCT_LABEL[cond])
        parts.append(out)
    table = pd.concat(parts, ignore_index=True)
    n_by = table["condition"].value_counts().to_dict()
    checks.add("canonical table: one row per cell (Ctrl 729, ZeroK 702); cell_key unique",
               f"{len(table)} rows; {n_by}", len(table) == 1431 and n_by == {"Ctrl": 729, "ZeroK": 702}
               and table["cell_key"].is_unique)
    path = run.table(table, "dct_labels")

    # labelled objects: the dct_labels object plus the dct_prolif columns
    P = pset.loc[pset["in_P"].astype(bool), "matched_name_ctrl"].astype(str).tolist()
    used = mtab[mtab["used"].astype(bool)] if "used" in mtab else mtab
    uns = {"plan": "the analysis plan", "freeze_commit": lib.FREEZE_COMMIT,
           "runs": json.dumps({k: vars(v) for k, v in lib.RUNS.items()}),
           "approved_change": json.dumps(lib.APPROVED_CHANGE), "primary_run": lib.PRIMARY_RUN,
           "primary_switch": json.dumps(lib.PRIMARY_SWITCH), "P": P, "status": "dct_labels labels stay the DCT primary until "
           "the authors adopt dct_prolif", "generated_by": f"{cc.GENERATED_TAG} ({SCRIPT})"}
    back = pd.read_csv(path, dtype={"cell_id": str, "cell_key": str, "cs_t31b_cluster": str, "cs_t31nfb_cluster": str})
    t31_cols = [c for c in table.columns if c not in BASE and c not in ("dct_label", "dct_label_alternative")]
    for cond in lib.CONDITIONS:
        adata = ad.read_h5ad(run.use_input(t28.DCT_LABELS_H5AD[cond], f"dct_labels labelled object ({cond})"))
        sub = table[table["condition"] == cond].set_index("cell_id")
        oid = pd.Index(adata.obs_names.astype(str))
        same = set(oid) == set(sub.index) and len(oid) == len(sub)
        checks.add(f"{cond}: dct_labels object holds the same cells as the table", f"{len(oid)} / {len(sub)}", bool(same))
        if not same:
            lib.stop(run, checks, "canonical_checks")
        sub = sub.loc[oid]
        clash = [c for c in t31_cols if c in adata.obs.columns]
        checks.add(f"{cond}: dct_prolif columns do not overwrite dct_labels obs columns", ";".join(clash) or "none", not clash)
        for c in t31_cols:
            v = sub[c]
            adata.obs[c] = v.astype(str).to_numpy() if v.dtype == object else v.to_numpy()
        adata.uns["t31"] = {**uns, "identity_markers": used.to_json(orient="records")}
        p = run.h5ad(adata, f"DCT_{lib.KEY[cond]}_labelled")
        rd = ad.read_h5ad(p)
        b = back[back["condition"] == cond].set_index("cell_id").loc[oid]
        diff = [c for c in t31_cols if not same_values(rd.obs[c], b[c])]
        checks.add(f"{cond}: h5ad obs dct_prolif columns equal the CSV (read back; numbers within 1e-12, NaN = NaN; "
                   "others as text)", ";".join(diff) or "all equal", not diff)
    run.table(checks.frame(), "canonical_checks")
    run.finish({"runs": {k: vars(v) for k, v in lib.RUNS.items()}, "approved_change": lib.APPROVED_CHANGE,
                "primary_run": lib.PRIMARY_RUN, "primary_switch": lib.PRIMARY_SWITCH,
                "columns": list(table.columns), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_canonical_checks.csv.")


if __name__ == "__main__":
    main()
