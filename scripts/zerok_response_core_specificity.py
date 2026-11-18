#!/usr/bin/env python
"""zerok_response_core step 5b: specificity calibration of the zero-K vs control panel results (addendum).

Added at the authors' request on 28 Oct 2026, after steps
1-6 had run: about 80% of all tested proteins reach q < 0.05 between conditions, so the panel results are set
against the rest of the proteome.

(a) Panel ranks: for each panel protein tested proteome-wide (step 3 tables; label-free all cells from step 5),
    the percentile of its effects among all tested proteins of the group (pseudobulk log2 FC, depth-adjusted
    intensity log2 FC, detection log odds ratio, two-part chi2), the 'directional percentile' (high = far in
    the expected direction) and the effect minus the proteome median.
(b) Matched random protein sets: the 7 panel proteins with an expected direction (up: Slc12a3, Wnk1, Wnk4; down:
    Klhl3, Pvalb, Calb1, Slc8a1; Cul3 'no change' is not directional) are each replaced by a random non-panel
    protein drawn from its 50 nearest proteins in (detection fraction over both conditions, mean of the two
    condition medians of ln-intensity among detected cells), both as percentile ranks; 10,000 draws, seed 42.
    Statistics: S1 = number of slots whose observed direction (D1 rule, BH across the proteome of the group)
    equals the expected direction; S2 = mean of expected sign x pseudobulk log2 FC; S3 = mean of expected sign x
    intensity log2 FC. Empirical p = (1 + #draws >= observed) / (1 + draws). Per slot: the share of its 50
    neighbours with the expected observed direction, and the slot's directional percentile among them.
(c) Within-condition plate null: the step 3 two-part pipeline (zerok_response_core_lib.proteome_group) on P2 vs P1 within
    Ctrl and within ZeroK cells, beside ZeroK vs Ctrl within P1 and within P2 (similar cell numbers). Count and
    share of proteins with q < 0.05; panel proteins in each contrast. Groups with >= 20 cells on each side.
All p-values are descriptive (one pool per condition; cells are not animal-level replicates; P1/P2 are sort
plates of one pool, so the plate contrast is a technical null, not a biological one).

Inputs (read-only): results/h5ad/zerok_core_DCT_joint_labelled.h5ad; zerok_core panel, hash and joint labels; zerok_response_core step 3
proteome tables and step 5 label-free proteome table.
Outputs (results/tables/zerok_response_core/): zerok_response_core_specificity_panel_ranks.csv,
zerok_response_core_specificity_matched_null.csv, zerok_response_core_specificity_matched_slots.csv,
zerok_response_core_specificity_null_draws_summary.csv, zerok_response_core_specificity_plate_null.csv,
zerok_response_core_specificity_plate_null_panel.csv, zerok_response_core_specificity_checks.csv.

Command (project root):  python scripts/zerok_response_core_specificity.py
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
from scipy import stats

import scp_common as cc
import cluster_de_lib as t5
import zerok_response_core_lib as lib

SCRIPT = "zerok_response_core_specificity.py"
N_DRAWS = 10_000
N_NEIGHBOURS = 50
EFFECTS = {"pseudobulk": "pseudobulk_log2fc_pooled", "intensity": "int_log2fc_adjusted",
           "detection": "det_logodds", "evidence_chi2": "two_part_chi2"}
CONTRASTS = {  # name: (reference mask, comparison mask) as (condition, plate) filters
    "Ctrl P2 vs P1": (("Ctrl", "P1"), ("Ctrl", "P2")),
    "ZeroK P2 vs P1": (("ZeroK", "P1"), ("ZeroK", "P2")),
    "ZeroK vs Ctrl, P1 only": (("Ctrl", "P1"), ("ZeroK", "P1")),
    "ZeroK vs Ctrl, P2 only": (("Ctrl", "P2"), ("ZeroK", "P2")),
}
NULL_KIND = {"Ctrl P2 vs P1": "within-condition plate (technical null)",
             "ZeroK P2 vs P1": "within-condition plate (technical null)",
             "ZeroK vs Ctrl, P1 only": "between conditions, one plate",
             "ZeroK vs Ctrl, P2 only": "between conditions, one plate"}


def proteome_tables(run) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    for col in lib.LABEL_COLUMNS:
        for v in lib.VARIANTS:
            out[(col, v)] = lib.read_table(run, f"proteome_{lib.LABEL_SHORT[col]}_{v}")
    lf = lib.read_table(run, "robustness_label_free_proteome")
    for v in lib.VARIANTS:
        out[("label_free", v)] = lf.query("variant == @v")
    return out


def pct_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] (average ranks; NaN kept)."""
    v = np.asarray(values, float)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum() > 1:
        out[ok] = (stats.rankdata(v[ok]) - 1) / (ok.sum() - 1)
    return out


def directional_share(values: np.ndarray, neighbours: np.ndarray, i: int, sign: int) -> float:
    """Share of neighbours (finite effects only) lying less far in the expected direction than protein i."""
    if not np.isfinite(values[i]):
        return np.nan
    nb = sign * values[neighbours]
    nb = nb[np.isfinite(nb)]
    return float(np.mean(nb < sign * values[i])) if len(nb) else np.nan


def matching_space(tab: pd.DataFrame) -> np.ndarray:
    det = (tab["det_ctrl"] + tab["det_zerok"]) / (tab["n_ctrl"] + tab["n_zerok"])
    ln = tab[["median_ln_ctrl", "median_ln_zerok"]].mean(axis=1, skipna=True)
    return np.column_stack([pct_rank(det.to_numpy(float)), pct_rank(ln.to_numpy(float))])


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_response_core step 5b: specificity calibration.")
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "specificity", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_panel_hash(run, checks)
    panel = lib.read_panel(run)
    directional = panel.loc[panel["expected_direction"].isin([lib.UP, lib.DOWN])]
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
            ranks = {k: pct_rank(g[c].to_numpy(float)) for k, c in EFFECTS.items()}
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
                        sgn = {lib.UP: 1, lib.DOWN: -1}.get(exp, np.nan)
                        row[f"{k}_directional_percentile"] = (ranks[k][i] if sgn == 1 else 1 - ranks[k][i]
                                                              if sgn == -1 else np.nan)
                rank_rows.append(row)

            # matched null
            slots = [p for p in expected if p in set(g["protein"])]
            if not slots:
                continue
            space = matching_space(g)
            cand_pool = np.flatnonzero(~in_panel)
            sign = np.array([1 if expected[p] == lib.UP else -1 for p in slots])
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
                                      directional_share(pb, nb, i, s),
                                  "intensity_directional_percentile_among_neighbours":
                                      directional_share(it, nb, i, s),
                                  "max_neighbour_distance": float(np.sort(d)[N_NEIGHBOURS - 1]),
                                  "n_neighbours": len(nb)})
            idx_obs = np.array([int(np.flatnonzero(g["protein"].to_numpy() == p)[0]) for p in slots])
            exp_arr = np.array([expected[p] for p in slots])
            obs_s1 = int((obs_dir[idx_obs] == exp_arr).sum())
            obs_s2 = float(np.nanmean(sign * pb[idx_obs]))
            obs_s3 = float(np.nanmean(sign * it[idx_obs]))
            rng = lib.repeat_rng(52, lib.LABEL_COLUMN_INDEX[col], lib.GROUP_ORDER.index(group),
                                 lib.VARIANTS.index(v))
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
    spec = lib.panel_spec(joint, panel)
    panel_names = {name for _, name, _ in spec.proteins if name}
    bts = {v: lib.back_transform(joint, v) for v in lib.VARIANTS}
    cond = joint.obs["condition"].to_numpy()
    plate = joint.obs["plate"].to_numpy()

    # implementation check: the explicit-z path reproduces step 3 for one group
    m = lib.group_masks(joint.obs, "dct_label")["DCT2"]
    idx = np.flatnonzero(m)
    re = lib.proteome_group(joint, idx, bts, {"label_column": "dct_label", "group": "DCT2"}, panel_names,
                            variants=(lib.BASELINE,), z=joint.is_zerok[idx])
    st3 = tabs[("dct_label", lib.BASELINE)].query("group == 'DCT2'").set_index("protein")["p_two_part"]
    dev = float(np.nanmax(np.abs(re.set_index("protein")["p_two_part"] - st3.reindex(re["protein"]).to_numpy())))
    checks.add("explicit comparison-side path reproduces step 3 (primary DCT2, variant none)",
               f"max |diff p| {dev:.2g}", dev < 1e-12)

    groups = [("label_free", lib.ALL_CELLS, np.ones(len(cond), bool))]
    for col in lib.LABEL_COLUMNS:
        gm = lib.group_masks(joint.obs, col)
        groups += [(col, g, gm[g]) for g in ("DCT1", lib.ALL_DCT)]
    plate_rows, plate_panel = [], []
    for col, group, gmask in groups:
        for cname, ((c0, p0), (c1, p1)) in CONTRASTS.items():
            ref = gmask & (cond == c0) & (plate == p0)
            cmp_ = gmask & (cond == c1) & (plate == p1)
            n0, n1 = int(ref.sum()), int(cmp_.sum())
            if min(n0, n1) < lib.MIN_GROUP_PROTEOME:
                plate_rows.append({"label_column": col, "group": group, "contrast": cname, "kind": NULL_KIND[cname],
                                   "n_reference": n0, "n_comparison": n1, "note": "< 20 cells on a side; not run"})
                continue
            idx = np.flatnonzero(ref | cmp_)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                f = lib.proteome_group(joint, idx, bts, {"label_column": col, "group": group, "contrast": cname},
                                       panel_names, z=cmp_[idx])
            for v, fv in f.groupby("variant", sort=False):
                sig = fv["q_two_part"] < lib.ALPHA
                plate_rows.append({"label_column": col, "group": group, "contrast": cname, "kind": NULL_KIND[cname],
                                   "variant": v, "n_reference": n0, "n_comparison": n1,
                                   "n_proteins_tested": len(fv), "n_q_below_0.05": int(sig.sum()),
                                   "share_q_below_0.05": float(sig.mean()),
                                   "n_up": int((fv["observed_direction"] == lib.UP).sum()),
                                   "n_down": int((fv["observed_direction"] == lib.DOWN).sum()),
                                   "n_mixed": int((fv["observed_direction"] == lib.OBS_MIXED).sum())})
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
                           "requested_by": "authors, 28 Oct 2026 (after steps 1-6)", "p_value_note": lib.P_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_response_core_specificity_checks.csv.")


if __name__ == "__main__":
    main()
