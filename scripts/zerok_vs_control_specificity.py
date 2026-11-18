#!/usr/bin/env python
"""zerok_vs_control step 6, addendum 5b: specificity calibration on the combined search (zerok_response_core step 5b rerun).

Analysis plan,
step 6 (addendum 5b). Same procedure and settings as zerok_response_core_specificity.py (its helpers and zerok_response_core_lib
imported unchanged; settings frozen in the zerok_vs_control plan):

(a) Panel ranks: for each panel protein tested proteome-wide (step 4 tables; label-free all cells from step 6), the
    percentile of its effects among all tested proteins of the group, the directional percentile and the effect
    minus the proteome median.
(b) Matched random protein sets: the 7 directional panel proteins (up: Slc12a3, Wnk1, Wnk4; down: Klhl3, Pvalb,
    Calb1, Slc8a1) are each replaced by a random non-panel protein from its 50 nearest proteins in (detection
    fraction, ln-intensity) percentile space; 10,000 draws (zerok_response_core seeds). S1 = slots with the expected observed
    direction (proteome-wide BH of the group), S2 = mean expected sign x pseudobulk log2 FC, S3 = the same for the
    intensity log2 FC; one-sided empirical p.
(c) Within-condition plate null: the step 4 two-part pipeline on P2 vs P1 within Ctrl and within ZeroK, beside ZeroK
    vs Ctrl within P1 and within P2; groups label-free all cells, DCT1 and all_DCT under both label columns.
All p-values are descriptive (one pool per condition; cells are not animal-level replicates; P1/P2 are sort plates
of one pool, so the plate contrast is a technical null).

Inputs (read-only): results/h5ad/dct_labels_DCT_joint_labelled.h5ad; dct_labels canonical labels; zerok_core panel and hash; zerok_vs_control
step 4 proteome tables and step 6 label-free proteome table.
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_specificity_panel_ranks.csv,
zerok_vs_control_specificity_matched_null.csv, zerok_vs_control_specificity_matched_slots.csv,
zerok_vs_control_specificity_null_draws_summary.csv, zerok_vs_control_specificity_plate_null.csv,
zerok_vs_control_specificity_plate_null_panel.csv, zerok_vs_control_specificity_checks.csv.

Command (project root):  python scripts/zerok_vs_control_specificity.py
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
import warnings

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import zerok_response_core_specificity as t22sp
import zerok_vs_control_lib as lib

SCRIPT = "zerok_vs_control_specificity.py"
t22 = lib.t22
N_DRAWS = t22sp.N_DRAWS
N_NEIGHBOURS = t22sp.N_NEIGHBOURS
EFFECTS = t22sp.EFFECTS
CONTRASTS = t22sp.CONTRASTS
NULL_KIND = t22sp.NULL_KIND
CHECK_GROUP = ("dct_label_alternative", "DCT2")   # zerok_response_core used primary DCT2, which has no ZeroK cells here


def proteome_tables(run) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for col in lib.LABEL_COLUMNS:
        for v in lib.VARIANTS:
            out[(col, v)] = lib.read_table(run, f"proteome_{lib.LABEL_SHORT[col]}_{v}")
    lf = lib.read_table(run, "robustness_label_free_proteome")
    for v in lib.VARIANTS:
        out[("label_free", v)] = lf.query("variant == @v")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 6b: specificity calibration.")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "specificity", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    panel = t22.read_panel(run)
    directional = panel.loc[panel["expected_direction"].isin([t22.UP, t22.DOWN])]
    expected = dict(zip(directional["item"], directional["expected_direction"]))
    panel_proteins = set(panel.loc[panel["item_type"] == "protein", "item"])
    checks.add("directional panel proteins (expected up / down)", "; ".join(f"{k} {v}" for k, v in expected.items()),
               len(expected) == 7, "7")
    tabs = proteome_tables(run)

    # ---- (a) panel ranks and (b) matched random sets
    rank_rows, null_rows, slot_rows, draw_rows = [], [], [], []
    n_panel_neighbours = 0
    for (col, v), tab in tabs.items():
        for group, g in tab.groupby("group", sort=False):
            g = g.reset_index(drop=True)
            in_panel = g["protein"].isin(panel_proteins).to_numpy()
            ranks = {k: t22sp.pct_rank(g[c].to_numpy(float)) for k, c in EFFECTS.items()}
            medians = {k: float(np.nanmedian(g[c].to_numpy(float))) for k, c in EFFECTS.items()}
            for i in np.flatnonzero(in_panel):
                p = g.at[i, "protein"]
                exp = panel.loc[panel["item"] == p, "expected_direction"].iloc[0]
                row = {"label_column": col, "group": group, "variant": v, "protein": p, "expected_direction": exp,
                       "observed_direction_proteome_bh": g.at[i, "observed_direction"],
                       "q_two_part_proteome": g.at[i, "q_two_part"], "n_proteins_tested": len(g)}
                for k, c in EFFECTS.items():
                    row[f"{k}_effect"] = g.at[i, c]
                    row[f"{k}_percentile"] = ranks[k][i]
                    if k != "evidence_chi2":
                        row[f"{k}_minus_proteome_median"] = g.at[i, c] - medians[k]
                        sgn = {t22.UP: 1, t22.DOWN: -1}.get(exp, np.nan)
                        row[f"{k}_directional_percentile"] = (ranks[k][i] if sgn == 1 else 1 - ranks[k][i]
                                                              if sgn == -1 else np.nan)
                rank_rows.append(row)

            # matched null
            slots = [p for p in expected if p in set(g["protein"])]
            if not slots:
                continue
            space = t22sp.matching_space(g)
            cand_pool = np.flatnonzero(~in_panel)
            sign = np.array([1 if expected[p] == t22.UP else -1 for p in slots])
            obs_dir = g["observed_direction"].to_numpy()
            pb = g[EFFECTS["pseudobulk"]].to_numpy(float)
            it = g[EFFECTS["intensity"]].to_numpy(float)
            neigh = []
            for p, s in zip(slots, sign):
                i = int(np.flatnonzero(g["protein"].to_numpy() == p)[0])
                d = np.linalg.norm(space[cand_pool] - space[i], axis=1)
                nb = cand_pool[np.argsort(d)[:N_NEIGHBOURS]]
                n_panel_neighbours += int(g["protein"].iloc[nb].isin(panel_proteins).sum())
                neigh.append(nb)
                hit = obs_dir[nb] == expected[p]
                slot_rows.append({"label_column": col, "group": group, "variant": v, "protein": p,
                                  "expected_direction": expected[p], "observed_direction": obs_dir[i],
                                  "neighbour_share_expected_direction": float(hit.mean()),
                                  "pseudobulk_directional_percentile_among_neighbours":
                                      t22sp.directional_share(pb, nb, i, s),
                                  "intensity_directional_percentile_among_neighbours":
                                      t22sp.directional_share(it, nb, i, s),
                                  "max_neighbour_distance": float(np.sort(d)[N_NEIGHBOURS - 1]),
                                  "n_neighbours": len(nb)})
            idx_obs = np.array([int(np.flatnonzero(g["protein"].to_numpy() == p)[0]) for p in slots])
            exp_arr = np.array([expected[p] for p in slots])
            obs_s1 = int((obs_dir[idx_obs] == exp_arr).sum())
            obs_s2 = float(np.nanmean(sign * pb[idx_obs]))
            obs_s3 = float(np.nanmean(sign * it[idx_obs]))
            rng = t22.repeat_rng(52, t22.LABEL_COLUMN_INDEX[col], t22.GROUP_ORDER.index(group),
                                 t22.VARIANTS.index(v))
            draws = np.column_stack([rng.choice(nb, size=args.draws) for nb in neigh])   # draws x slots
            s1 = (obs_dir[draws] == exp_arr[None, :]).sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                s2 = np.nanmean(sign[None, :] * pb[draws], axis=1)
                s3 = np.nanmean(sign[None, :] * it[draws], axis=1)
            for name, obs, null in (("S1 matches", obs_s1, s1), ("S2 signed pseudobulk log2FC", obs_s2, s2),
                                    ("S3 signed intensity log2FC", obs_s3, s3)):
                nl = null[np.isfinite(null)]
                null_rows.append({"label_column": col, "group": group, "variant": v, "statistic": name,
                                  "n_slots": len(slots), "slots": ";".join(slots), "observed": obs,
                                  "null_mean": float(nl.mean()), "null_q025": float(np.quantile(nl, 0.025)),
                                  "null_q975": float(np.quantile(nl, 0.975)), "n_draws": len(nl),
                                  "empirical_p_greater_or_equal": float((1 + (nl >= obs).sum()) / (1 + len(nl))),
                                  "observed_percentile_in_null": float((nl < obs).mean())})
            if col == "dct_label" and v == lib.BASELINE and group == lib.ALL_DCT:
                for k in range(len(slots) + 1):
                    draw_rows.append({"label_column": col, "group": group, "variant": v, "statistic": "S1 matches",
                                      "value": k, "n_draws": int((s1 == k).sum())})
    ranks = pd.DataFrame(rank_rows)
    ranks["p_value_note"] = lib.P_NOTE
    nulls = pd.DataFrame(null_rows)
    nulls["p_value_note"] = ("empirical null from matched random proteins of the same comparison; inherits the "
                             "design limitation: " + lib.P_NOTE)
    slots_df = pd.DataFrame(slot_rows)
    checks.add("matched-null blocks (label column x group x variant)",
               f"{nulls[['label_column', 'group', 'variant']].drop_duplicates().shape[0]}", None)
    checks.add("draws per block", f"{int(nulls['n_draws'].min())}-{int(nulls['n_draws'].max())}",
               int(nulls.loc[nulls['statistic'] == 'S1 matches', 'n_draws'].min()) == args.draws, f"{args.draws}")
    checks.add("panel proteins among the matched neighbours (all slots, all blocks)", f"{n_panel_neighbours}",
               n_panel_neighbours == 0, "0")
    n_short = int((slots_df["n_neighbours"] < N_NEIGHBOURS).sum())
    checks.add(f"slots with fewer than {N_NEIGHBOURS} neighbours", f"{n_short}", n_short == 0, "0")
    checks.add("max neighbour distance in percentile space (worst slot)",
               f"{slots_df['max_neighbour_distance'].max():.3f}", None)
    run.table(ranks, "specificity_panel_ranks")
    run.table(nulls, "specificity_matched_null")
    run.table(slots_df, "specificity_matched_slots")
    run.table(pd.DataFrame(draw_rows), "specificity_null_draws_summary")

    # ---- (c) within-condition plate null
    joint = lib.load_joint(run, checks)
    spec = t22.panel_spec(joint, panel)
    panel_names = {name for _, name, _ in spec.proteins if name}
    bts = {v: t22.back_transform(joint, v) for v in lib.VARIANTS}
    cond = joint.obs["condition"].to_numpy()
    plate = joint.obs["plate"].to_numpy()
    absent = [(c, g, why) for c in lib.LABEL_COLUMNS for g, why in lib.split_groups(joint, c)[1].items()]
    checks.add("'does not apply' groups (label absent in a condition): no matched null", "; ".join(
        f"{lib.LABEL_SHORT[c]}/{g}" for c, g, _ in absent), None)

    # implementation check: the explicit-z path reproduces step 4 for one group
    col0, grp0 = CHECK_GROUP
    m = t22.group_masks(joint.obs, col0)[grp0]
    idx = np.flatnonzero(m)
    re = t22.proteome_group(joint, idx, bts, {"label_column": col0, "group": grp0}, panel_names,
                            variants=(lib.BASELINE,), z=joint.is_zerok[idx])
    st3 = tabs[(col0, lib.BASELINE)].query("group == @grp0").set_index("protein")["p_two_part"]
    dev = float(np.nanmax(np.abs(re.set_index("protein")["p_two_part"] - st3.reindex(re["protein"]).to_numpy())))
    checks.add("explicit comparison-side path reproduces step 4 (alternative DCT2, variant none)",
               f"max |diff p| {dev:.2g}", dev < 1e-12)

    groups = [("label_free", lib.ALL_CELLS, np.ones(len(cond), bool))]
    for col in lib.LABEL_COLUMNS:
        gm = t22.group_masks(joint.obs, col)
        groups += [(col, g, gm[g]) for g in ("DCT1", lib.ALL_DCT)]
    plate_rows, plate_panel = [], []
    for col, group, gmask in groups:
        for cname, ((c0, p0), (c1, p1)) in CONTRASTS.items():
            ref = gmask & (cond == c0) & (plate == p0)
            cmp_ = gmask & (cond == c1) & (plate == p1)
            n0, n1 = int(ref.sum()), int(cmp_.sum())
            if min(n0, n1) < t22.MIN_GROUP_PROTEOME:
                plate_rows.append({"label_column": col, "group": group, "contrast": cname, "kind": NULL_KIND[cname],
                                   "n_reference": n0, "n_comparison": n1, "note": "< 20 cells on a side; not run"})
                continue
            idx = np.flatnonzero(ref | cmp_)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f = t22.proteome_group(joint, idx, bts, {"label_column": col, "group": group, "contrast": cname},
                                       panel_names, z=cmp_[idx])
            for v, fv in f.groupby("variant", sort=False):
                sig = fv["q_two_part"] < lib.ALPHA
                plate_rows.append({"label_column": col, "group": group, "contrast": cname, "kind": NULL_KIND[cname],
                                   "variant": v, "n_reference": n0, "n_comparison": n1,
                                   "n_proteins_tested": len(fv), "n_q_below_0.05": int(sig.sum()),
                                   "share_q_below_0.05": float(sig.mean()),
                                   "n_up": int((fv["observed_direction"] == t22.UP).sum()),
                                   "n_down": int((fv["observed_direction"] == t22.DOWN).sum()),
                                   "n_mixed": int((fv["observed_direction"] == t22.OBS_MIXED).sum())})
                pp = fv.loc[fv["in_panel"]]
                for r in pp.itertuples():
                    plate_panel.append({"label_column": col, "group": group, "contrast": cname, "variant": v,
                                        "protein": r.protein, "observed_direction": r.observed_direction,
                                        "q_two_part": r.q_two_part, "det_logodds": r.det_logodds,
                                        "int_log2fc_adjusted": r.int_log2fc_adjusted,
                                        "pseudobulk_log2fc_pooled": r.pseudobulk_log2fc_pooled})
            print(f"  plate null {col} / {group} / {cname}: {n0} vs {n1} cells, {time.time() - t0:.0f} s", flush=True)
    plate_df = pd.DataFrame(plate_rows)
    plate_df["p_value_note"] = ("descriptive; P1/P2 are sort plates of one pool per condition (technical null); "
                                + lib.P_NOTE)
    plate_pan = pd.DataFrame(plate_panel)
    plate_pan["p_value_note"] = lib.P_NOTE
    run.table(plate_df, "specificity_plate_null")
    run.table(plate_pan, "specificity_plate_null_panel")
    ran = plate_df.dropna(subset=["n_proteins_tested"])
    checks.add("plate-null contrasts run (group x contrast x variant)", f"{len(ran)}", len(ran) > 0)

    run.table(checks.frame(), "specificity_checks")
    run.finish(parameters={"n_draws": args.draws, "n_neighbours": N_NEIGHBOURS,
                           "matching_space": "percentile ranks of detection fraction (both conditions) and mean of "
                                             "the two condition medians of ln-intensity among detected cells",
                           "directional_items": expected, "statistics": ["S1 matches", "S2 signed pseudobulk",
                                                                         "S3 signed intensity"],
                           "contrasts": {k: [list(a), list(b)] for k, (a, b) in CONTRASTS.items()},
                           "settings": "frozen in the zerok_vs_control plan (zerok_response_core step 5b settings)",
                           "implementation_check_group": "alternative DCT2 (zerok_response_core: primary DCT2, absent in ZeroK here)",
                           "ported_from": "zerok_response_core_specificity.py", "p_value_note": lib.P_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_specificity_checks.csv.")


if __name__ == "__main__":
    main()
