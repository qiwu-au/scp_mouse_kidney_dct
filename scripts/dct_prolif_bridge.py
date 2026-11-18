#!/usr/bin/env python
"""dct_prolif step 6: bridge (report only) from the dct_prolif labels to the dct_labels labels and the dct_reviewer cycling rule.

Analysis plan
(frozen), step 6. Runs: 'frozen' (floor 1,000) and 'nofloor' (approved change, authors 3 Nov 2026). Matched on
condition + cell_key; cell_id is checked to agree.

For each run, condition and dct_prolif route (A = cs_t31_label, B = cs_t31b_label; nofloor: cs_t31nf_label, cs_t31nfb_label),
against every dct_labels column of that condition (Ctrl: cs_ctrl_deg_label, cs_ctrl_cluster_label, cs_ctrl_cell_label;
ZeroK: cs_zerok_pipeline_label, cs_zerok_transfer_label, cs_zerok_cluster_label, cs_zerok_cell_label) and the dct_reviewer
cycling rule (cell_cycle_score > 90th percentile within the condition):
  - crosstab with the cell count and the median n_detected of each cell pair;
  - fraction equal (literal label strings; dct_prolif 'DCT, subtype not resolved', 'Unknown (markers not detected)' and
    'Below depth floor' never equal a dct_labels label) and ARI, over all cells and over the cells above the floor;
  - depth (n_detected) of the cells whose label changes and of those whose label is kept;
  - ProLIF membership against the Leiden 3.0 ProLIF (cs_*_cluster_label) and, for information, against the per-cell
    ProLIF (cs_*_cell_label) and the dct_reviewer cycling rule; the dct_prolif cycling call (ProLIF + 'Cycling, NCC not detected')
    against the dct_reviewer cycling rule.

Inputs (read-only): dct_prolif_route_{a,b}_per_cell.csv; results/tables/dct_labels/
dct_labels_dct_canonical_labels.csv; results/tables/dct_reviewer/dct_reviewer_reannot_per_cell.csv.
Outputs (results/tables/dct_prolif/): dct_prolif_bridge_{agreement,crosstabs,depth_changed,
prolif_membership,checks}.csv.

Command (project root):  python scripts/dct_prolif_bridge.py
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

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dct_reviewer_lib as t28
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_bridge.py"
DCT_REVIEWER_CYCLING = cc.out_path("dct_reviewer", "tables", "reannot_per_cell", ".csv")
DCT_REVIEWER_COLUMN = "t28_cycling_rule"
CYCLING_SET = (lib.PROLIF, lib.CYCLING_NO_NCC)

# dct_labels columns per condition, with their role (the project notes, "DCT labels, combined search").
REFERENCES = {
    "Ctrl": {"cs_ctrl_deg_label": "dct_labels primary (dct_label)", "cs_ctrl_cluster_label": "submitted pipeline, Leiden 3.0 mode",
             "cs_ctrl_cell_label": "submitted pipeline, per-cell argmax"},
    "ZeroK": {"cs_zerok_pipeline_label": "dct_labels primary (dct_label)",
              "cs_zerok_transfer_label": "dct_labels alternative (dct_label_alternative)",
              "cs_zerok_cluster_label": "submitted pipeline, Leiden 3.0 mode",
              "cs_zerok_cell_label": "submitted pipeline, per-cell argmax"},
}
LEIDEN3 = {"Ctrl": "cs_ctrl_cluster_label", "ZeroK": "cs_zerok_cluster_label"}
PER_CELL = {"Ctrl": "cs_ctrl_cell_label", "ZeroK": "cs_zerok_cell_label"}
EXPECTED = {   # the project notes counts
    "cs_ctrl_deg_label": {"DCT1": 382, "DCT2": 85, "Unknown": 262},
    "cs_ctrl_cluster_label": {"DCT1": 460, "DCT2": 126, "ProLIF": 119, "Unknown": 24},
    "cs_ctrl_cell_label": {"DCT1": 350, "DCT2": 207, "ProLIF": 172},
    "cs_zerok_pipeline_label": {"DCT1": 449, "Unknown": 253},
    "cs_zerok_transfer_label": {"DCT1": 398, "DCT2": 68, "Unknown": 236},
    "cs_zerok_cluster_label": {"DCT1": 469, "DCT2": 114, "ProLIF": 98, "Unknown": 21},
    "cs_zerok_cell_label": {"DCT1": 321, "DCT2": 212, "ProLIF": 169},
}
EXPECTED_DCT_REVIEWER = {"Ctrl": 73, "ZeroK": 71}


def as_bool(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.lower().eq("true").to_numpy()


def depth_summary(ids: np.ndarray) -> dict:
    if len(ids) == 0:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan}
    q = np.percentile(ids, [25, 50, 75])
    return {"n": int(len(ids)), "median": float(q[1]), "q25": float(q[0]), "q75": float(q[2])}


def membership(run_key, cond, route, t31_set, ref_col, ref_set, a: np.ndarray, b: np.ndarray, ids: np.ndarray,
               scope: str) -> dict:
    row = {"run": run_key, "condition": cond, "route": route, "scope": scope, "t31_set": t31_set,
           "reference_column": ref_col, "reference_set": ref_set, "n_cells": int(len(a)),
           "n_t31": int(a.sum()), "n_reference": int(b.sum()), "n_both": int((a & b).sum()),
           "n_t31_only": int((a & ~b).sum()), "n_reference_only": int((~a & b).sum()), "jaccard": lib.jaccard(a, b)}
    for name, m in (("both", a & b), ("t31_only", a & ~b), ("reference_only", ~a & b)):
        d = depth_summary(ids[m])
        row[f"median_n_detected_{name}"] = d["median"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif step 6: bridge to the dct_labels labels and the dct_reviewer cycling rule.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "bridge", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "bridge_checks")

    per_a = lib.read_table(run, "route_a_per_cell", dtype={"cell_id": str, "cell_key": str})
    per_b = lib.read_table(run, "route_b_per_cell", dtype={"cell_id": str, "cell_key": str, "cluster": str})
    canon = t28.read_canonical(run)
    t28c = pd.read_csv(run.use_input(DCT_REVIEWER_CYCLING, "dct_reviewer cycling rule per cell (combined search)"),
                       dtype={"cell_id": str, "cell_key": str})

    for cond in lib.CONDITIONS:
        sub = canon[canon["condition"] == cond]
        for c, exp in EXPECTED.items():
            if c in REFERENCES[cond]:
                obs = sub[c].astype(str).value_counts().to_dict()
                checks.add(f"dct_labels {c} counts equal the project notes", lib.counts_text(sub[c], cc.DCT_LABEL_ORDER),
                           obs == exp, "; ".join(f"{k} {v}" for k, v in exp.items()))
        n_cyc = int((t28c.loc[t28c["condition"] == cond, "reviewer_cycling_status"] == "Cycling").sum())
        checks.add(f"dct_reviewer cycling rule ({cond}): Cycling cells", n_cyc, n_cyc == EXPECTED_DCT_REVIEWER[cond], EXPECTED_DCT_REVIEWER[cond])

    agree_rows, ct_rows, depth_rows, mem_rows = [], [], [], []
    for run_key in lib.RUN_KEYS:
        for cond in lib.CONDITIONS:
            tag = f"{run_key} / {cond}"
            fa = per_a[(per_a["run"] == run_key) & (per_a["condition"] == cond)].reset_index(drop=True)
            fb = per_b[(per_b["run"] == run_key) & (per_b["condition"] == cond)].reset_index(drop=True)
            ok_b = list(fb["cell_key"]) == list(fa["cell_key"]) and list(fb["cell_id"]) == list(fa["cell_id"])
            ref = canon[canon["condition"] == cond].set_index("cell_key")
            cyc = t28c[t28c["condition"] == cond].set_index("cell_key")
            keys = fa["cell_key"].astype(str)
            found = keys.isin(ref.index).all() and keys.isin(cyc.index).all()
            unique = ref.index.is_unique and cyc.index.is_unique and keys.is_unique
            same_n = len(fa) == len(ref) == len(cyc)
            ids_match = found and (ref.loc[keys, "cell_id"].astype(str).to_numpy() == fa["cell_id"].to_numpy()).all() \
                and (cyc.loc[keys, "cell_id"].astype(str).to_numpy() == fa["cell_id"].to_numpy()).all()
            checks.add(f"{tag}: every dct_prolif cell matched once in dct_labels and dct_reviewer on cell_key, with equal cell_id; "
                       "route A and B tables in the same order",
                       f"{len(fa)} dct_prolif, {len(ref)} dct_labels, {len(cyc)} dct_reviewer cells",
                       bool(found and unique and same_n and ids_match and ok_b))
            if not (found and unique and same_n and ids_match and ok_b):
                continue
            above = as_bool(fa["above_floor"])
            ids = fa["n_detected"].to_numpy(float)
            t31 = {"A": fa["label"].astype(str).to_numpy(), "B": fb["label"].astype(str).to_numpy()}
            refs = {c: ref.loc[keys, c].astype(str).to_numpy() for c in REFERENCES[cond]}
            refs[DCT_REVIEWER_COLUMN] = cyc.loc[keys, "reviewer_cycling_status"].astype(str).to_numpy()
            roles = {**REFERENCES[cond], DCT_REVIEWER_COLUMN: "dct_reviewer cycling rule: cell_cycle_score > 90th percentile of the condition"}
            for route, lab in t31.items():
                t31_col = lib.col(run_key, "label", route.lower())
                for rcol, rlab in refs.items():
                    keys_ = {"run": run_key, "condition": cond, "route": route, "t31_column": t31_col,
                             "reference_column": rcol}
                    if rcol == DCT_REVIEWER_COLUMN:
                        # binary: dct_prolif cycling call (ProLIF + Cycling, NCC not detected) vs dct_reviewer Cycling
                        x = np.where(np.isin(lab, CYCLING_SET), lib.CYCLING, lib.NON_CYCLING)
                        x = np.where(above, x, lib.BELOW)
                        compare = x
                    else:
                        compare = lab
                    for scope, m in (("all cells", np.ones(len(lab), bool)), ("cells above the floor", above)):
                        if scope == "cells above the floor" and m.all():
                            continue
                        ag = lib.agreement(compare[m], rlab[m])
                        changed = compare[m] != rlab[m]
                        dc, du = depth_summary(ids[m][changed]), depth_summary(ids[m][~changed])
                        agree_rows.append({**keys_, "reference_role": roles[rcol], "scope": scope,
                                           "comparison": "cycling call vs dct_reviewer rule" if rcol == DCT_REVIEWER_COLUMN else "labels",
                                           **ag, "n_changed": int(changed.sum()),
                                           "t31_counts": lib.counts_text(compare[m]),
                                           "reference_counts": lib.counts_text(rlab[m], (*cc.DCT_LABEL_ORDER,
                                                                                          lib.CYCLING, lib.NON_CYCLING)),
                                           "median_n_detected_changed": dc["median"],
                                           "q25_n_detected_changed": dc["q25"], "q75_n_detected_changed": dc["q75"],
                                           "median_n_detected_kept": du["median"],
                                           "q25_n_detected_kept": du["q25"], "q75_n_detected_kept": du["q75"]})
                    # crosstab with the dct_prolif label (not the binary call), all cells, depth per cell pair
                    frame = pd.DataFrame({"t31_label": lab, "reference_label": rlab, "n_detected": ids})
                    g = frame.groupby(["t31_label", "reference_label"])["n_detected"]
                    stats_ = pd.DataFrame({"n_cells": g.size(), "median_n_detected": g.median()})
                    for row in lib.crosstab_long(lab, rlab, "t31_label", "reference_label", **keys_):
                        k = (row["t31_label"], row["reference_label"])
                        row["median_n_detected"] = float(stats_.loc[k, "median_n_detected"]) if k in stats_.index else np.nan
                        ct_rows.append(row)
                    # depth of changed vs kept cells, per dct_labels label (labels only)
                    if rcol != DCT_REVIEWER_COLUMN:
                        for rl in cc.ordered_labels(pd.unique(rlab), cc.DCT_LABEL_ORDER):
                            m = rlab == rl
                            for status, mm in (("kept", m & (lab == rlab)), ("changed", m & (lab != rlab))):
                                d = depth_summary(ids[mm])
                                depth_rows.append({**keys_, "reference_label": rl, "status": status,
                                                   "n_cells": d["n"], "median_n_detected": d["median"],
                                                   "q25_n_detected": d["q25"], "q75_n_detected": d["q75"],
                                                   "t31_labels": lib.counts_text(lab[mm])})

                # ProLIF membership: Leiden 3.0 (plan), per-cell and dct_reviewer cycling (information)
                pro = lab == lib.PROLIF
                cyc_call = np.isin(lab, CYCLING_SET)
                for scope, m in (("all cells", np.ones(len(lab), bool)), ("cells above the floor", above)):
                    if scope == "cells above the floor" and m.all():
                        continue
                    for t31_set, a in (("ProLIF", pro), ("cycling call (ProLIF + Cycling, NCC not detected)", cyc_call)):
                        if route == "B" and t31_set != "ProLIF":
                            continue
                        for rcol, rset, b in ((LEIDEN3[cond], "ProLIF (Leiden 3.0 mode; plan)", refs[LEIDEN3[cond]] == "ProLIF"),
                                              (PER_CELL[cond], "ProLIF (per-cell argmax; information)", refs[PER_CELL[cond]] == "ProLIF"),
                                              (DCT_REVIEWER_COLUMN, "Cycling (dct_reviewer rule; information)", refs[DCT_REVIEWER_COLUMN] == "Cycling")):
                            mem_rows.append(membership(run_key, cond, route, t31_set, rcol, rset, a[m], b[m], ids[m], scope))
            r = [x for x in mem_rows if x["run"] == run_key and x["condition"] == cond and x["route"] == "A"
                 and x["t31_set"] == "ProLIF" and x["reference_column"] == LEIDEN3[cond] and x["scope"] == "all cells"][0]
            p = [x for x in agree_rows if x["run"] == run_key and x["condition"] == cond and x["route"] == "A"
                 and x["reference_column"] == list(REFERENCES[cond])[0] and x["scope"] == "all cells"][0]
            checks.add(f"{tag}: route A vs dct_labels primary (all cells): fraction equal; ARI",
                       f"{p['fraction_equal']:.3f}; {p['ari']:.3f}", None)
            checks.add(f"{tag}: route-A ProLIF vs Leiden 3.0 ProLIF: both; dct_prolif only; Leiden 3.0 only; Jaccard",
                       f"{r['n_both']}; {r['n_t31_only']}; {r['n_reference_only']}; {r['jaccard']:.3f}", None)

    agree = pd.DataFrame(agree_rows)
    ct = pd.DataFrame(ct_rows)
    # consistency: crosstab totals equal the number of cells of each comparison
    tot = ct.groupby(["run", "condition", "route", "reference_column"])["n_cells"].sum()
    n_all = agree[agree["scope"] == "all cells"].set_index(["run", "condition", "route", "reference_column"])["n_cells"]
    checks.add("crosstab totals equal the cells compared (every run, condition, route, reference)",
               f"{int((tot.sort_index() == n_all.sort_index()).sum())} of {len(tot)}",
               bool(tot.sort_index().equals(n_all.sort_index())))
    run.table(agree, "bridge_agreement")
    run.table(ct, "bridge_crosstabs")
    run.table(pd.DataFrame(depth_rows), "bridge_depth_changed")
    run.table(pd.DataFrame(mem_rows), "bridge_prolif_membership")
    run.table(checks.frame(), "bridge_checks")
    run.finish({"runs": {k: vars(v) for k, v in lib.RUNS.items()}, "approved_change": lib.APPROVED_CHANGE,
                "match": "condition + cell_key; cell_id checked", "references": REFERENCES,
                "t28_rule": "cell_cycle_score > 90th percentile within condition (dct_reviewer reannot_per_cell)",
                "fraction_equal": "literal label strings", "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_bridge_checks.csv.")


if __name__ == "__main__":
    main()
