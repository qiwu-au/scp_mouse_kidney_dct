#!/usr/bin/env python
"""dct_prolif step 4: the five fixed acceptance tests for route-A ProLIF per condition and run, the DCT1/DCT2
reports (no pass/fail) and the other reported checks.

Analysis plan
(frozen), "Acceptance tests". Runs: 'frozen' (floor 1,000) and 'nofloor' (approved change, authors 3 Nov 2026). Cells
above the floor of each run; each condition alone.

Tests (ProLIF passes in a condition when all five pass):
1. Not shallower: logistic ProLIF ~ z(ln n_detected) + plate among DCT1, DCT2, "DCT, subtype not resolved" and ProLIF cells; pass if
   the 95% CI of the depth odds ratio includes or exceeds 1 (upper bound >= 1).
2. Not plate-driven: same model; pass if the 95% CI of the plate odds ratio includes 1.
3. Held-out markers: Mki67 or Top2a detected more often in ProLIF than in DCT1 + DCT2 + "DCT, subtype not resolved" cells
   (Fisher, one-sided p < 0.05).
4. Specific to the proliferation program: 100 random sets matched to P on detection deciles (drawn from the null pool
   with seed sequence [42, 4]), each with its own 1,000-set null (seed sequence [42, 4, i]; pool without the set's own
   proteins); pass if fewer than 5 of 100 give at least as many cycling cells as P.
5. Stable: null seeds 0-9; pass if the median Jaccard between the seed-42 ProLIF set and each seed's set is >= 0.8.
A test that cannot be computed (no ProLIF cell; failed fit) fails, with the reason.

Reported (no pass/fail): DCT2 vs DCT1 odds ratios (identity column; same model form); BIC and the posterior
distribution; identity calls kept over 20 noise repeats (Gaussian noise SD 1e-6 x intensity on detected values, seed
42; detection, hence D and the cycling call, does not change); share of Mki67/Top2a-positive cells called cycling;
ProLIF per plate; share of cycling cells with Slc12a3 detected; dct_reviewer contamination flags by label; ProLIF kNN purity in
the dct_labels graph (all cells of the condition, 20 PCs, k = 10, 1,000 label permutations); Spearman correlation between the
intensity part and D.

Inputs (read-only): results/h5ad/dct_intake_DCT_{ctrl,zerok}.h5ad; dct_prolif_route_a_per_cell.csv;
results/h5ad/dct_labels_DCT_{ctrl,zerok}_labelled.h5ad (X_pca_neighbors); results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_task3_scored.h5ad
(contamination flags).
Outputs (results/tables/dct_prolif/): dct_prolif_acceptance_{tests,summary,checks}.csv,
dct_prolif_test4_random_sets.csv, dct_prolif_test5_null_seeds.csv, dct_prolif_identity_odds.csv,
dct_prolif_posterior_distribution.csv, dct_prolif_noise_identity{,_per_cell}.csv, dct_prolif_reported_checks.csv,
dct_prolif_contamination_by_label.csv, dct_prolif_knn_purity_t27_graph.csv.

Command (project root):  python scripts/dct_prolif_acceptance.py
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

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_acceptance.py"
TEST4_SEED = (lib.SEED, 4)
POSTERIOR_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0 + 1e-12]


def test_row(run_key, cond, test, name, value, threshold, passed, computable=True, note="") -> dict:
    return {"run": run_key, "condition": cond, "test": test, "name": name, "value": value, "threshold": threshold,
            "computable": computable, "pass": bool(passed) if computable else False, "note": note}


def tests_1_3(run_key: str, cond: str, f: pd.DataFrame) -> tuple[list[dict], dict]:
    a = f["above_floor"].to_numpy(bool)
    sub = f[a & f["label"].isin(lib.TEST_SET).to_numpy()]
    y = (sub["label"] == lib.PROLIF).to_numpy()
    fit = lib.logit_or(y, np.log(sub["n_detected"].to_numpy(float)), sub["plate"].to_numpy())
    rows = []
    if fit["computable"]:
        rows.append(test_row(run_key, cond, 1, "not shallower: depth OR (95% CI) per SD of ln(n_detected)",
                             f"{fit['depth_or']:.3g} ({fit['depth_ci_low']:.3g}-{fit['depth_ci_high']:.3g})",
                             "upper 95% CI >= 1", fit["depth_ci_high"] >= 1,
                             note=f"{fit['n_positive']} ProLIF of {fit['n_cells']} cells; p {fit['depth_p']:.3g}"))
        rows.append(test_row(run_key, cond, 2, "not plate-driven: plate OR P2 vs P1 (95% CI)",
                             f"{fit['plate_or']:.3g} ({fit['plate_ci_low']:.3g}-{fit['plate_ci_high']:.3g})",
                             "95% CI includes 1", fit["plate_ci_low"] <= 1 <= fit["plate_ci_high"],
                             note=f"p {fit['plate_p']:.3g}"))
    else:
        for t, name in ((1, "not shallower"), (2, "not plate-driven")):
            rows.append(test_row(run_key, cond, t, name, "", "", False, computable=False, note=fit["reason"]))
    pro = y
    held = sub["mki67_or_top2a_detected"].to_numpy(bool)
    if pro.sum() == 0:
        rows.append(test_row(run_key, cond, 3, "held-out Mki67 or Top2a", "", "Fisher one-sided p < 0.05", False,
                             computable=False, note="no ProLIF cell"))
        fisher = {}
    else:
        table = [[int((pro & held).sum()), int((pro & ~held).sum())], [int((~pro & held).sum()), int((~pro & ~held).sum())]]
        res = stats.fisher_exact(table, alternative="greater")
        fisher = {"table": table, "odds_ratio": float(res.statistic), "p": float(res.pvalue)}
        rows.append(test_row(run_key, cond, 3, "held-out Mki67 or Top2a: ProLIF vs DCT1 + DCT2 + DCT, subtype not resolved",
                             f"{table[0][0]}/{pro.sum()} vs {table[1][0]}/{(~pro).sum()}; OR {res.statistic:.3g}; "
                             f"p {res.pvalue:.3g}", "Fisher one-sided p < 0.05", res.pvalue < 0.05))
    return rows, {"logit": fit, "fisher": fisher}


def test_4(run_key: str, cond: str, pr: lib.Prolif, n_cycling_P: int, var_names) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(list(TEST4_SEED))
    Q = lib.draw_sets(rng, pr.pool_idx, pr.decile, pr.need, lib.N_RANDOM_SETS)
    rows = []
    names = np.asarray(var_names)
    for i, q in enumerate(Q):
        pool = np.setdiff1d(pr.pool_idx, q)
        res = lib.cycling_call(pr, q, np.random.default_rng([*TEST4_SEED, i]), pool=pool)
        rows.append({"run": run_key, "condition": cond, "set": i, "n_cycling": int(res["cycling"].sum()),
                     "min_q": float(res["q"].min()), "proteins": ";".join(names[q])})
    frame = pd.DataFrame(rows)
    n_ge = int((frame["n_cycling"] >= n_cycling_P).sum())
    row = test_row(run_key, cond, 4, "specific to the proliferation program: random matched sets with >= as many "
                   "cycling cells as P", f"{n_ge} of {lib.N_RANDOM_SETS} (P: {n_cycling_P} cycling; random sets median "
                   f"{frame['n_cycling'].median():g}, max {frame['n_cycling'].max()})",
                   f"< {lib.TEST4_MAX_GE} of {lib.N_RANDOM_SETS}", n_ge < lib.TEST4_MAX_GE)
    return row, frame


def test_5(run_key: str, cond: str, pr: lib.Prolif, f: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    a = f["above_floor"].to_numpy(bool)
    ncc = f.loc[a, "slc12a3_detected"].to_numpy(bool)
    ref = (f.loc[a, "label"] == lib.PROLIF).to_numpy()
    rows = []
    for s in lib.NULL_SEEDS:
        res = lib.cycling_call(pr, pr.P_idx, np.random.default_rng(s))
        pro = res["cycling"] & ncc
        rows.append({"run": run_key, "condition": cond, "null_seed": s, "n_cycling": int(res["cycling"].sum()),
                     "n_prolif": int(pro.sum()), "n_prolif_seed42": int(ref.sum()),
                     "n_both": int((pro & ref).sum()), "jaccard_vs_seed42": lib.jaccard(pro, ref)})
    frame = pd.DataFrame(rows)
    med = frame["jaccard_vs_seed42"].median()
    if ref.sum() == 0:
        return test_row(run_key, cond, 5, "stable over null seeds 0-9", "", f"median Jaccard >= {lib.JACCARD_MIN}",
                        False, computable=False, note="no ProLIF cell at seed 42"), frame
    return test_row(run_key, cond, 5, "stable over null seeds 0-9: median Jaccard vs seed 42",
                    f"{med:.3f} (range {frame['jaccard_vs_seed42'].min():.3f}-{frame['jaccard_vs_seed42'].max():.3f})",
                    f"median >= {lib.JACCARD_MIN}", med >= lib.JACCARD_MIN), frame


def noise_identity(run_key: str, cond: str, c: lib.Condition, f: pd.DataFrame, used: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = f["above_floor"].to_numpy(bool)
    ref = f.loc[a, "identity"].to_numpy(object)
    base = c.intensity[a]
    det = c.detected[a]
    rng = np.random.default_rng(lib.SEED)
    kept = np.zeros(a.sum(), int)
    rows = []
    for rep in range(lib.NOISE_REPEATS + 1):
        if rep == 0:
            noisy = base
        else:
            noisy = base + np.where(det, rng.normal(0.0, 1.0, base.shape) * lib.NOISE_REL_SD * base, 0.0)
        ident = lib.identity_scores(noisy, det, c.ln_ids[a], c.var_names, used)
        call = lib.fit_mixture(ident["axis"].to_numpy(float))["call"]
        same = call == ref
        if rep == 0 and not same.all():
            raise ValueError(f"{run_key} / {cond}: no-noise control does not reproduce the identity calls")
        if rep:
            kept += same
        rows.append({"run": run_key, "condition": cond, "repeat": rep, "n_cells": int(a.sum()),
                     "n_changed": int((~same).sum()), "counts": lib.counts_text(call, lib.IDENTITY_ORDER)})
    per_cell = pd.DataFrame({"run": run_key, "condition": cond, "cell_id": f.loc[a, "cell_id"].to_numpy(),
                             "identity": ref, "n_repeats_same": kept,
                             "fraction_repeats_same": kept / lib.NOISE_REPEATS})
    return pd.DataFrame(rows), per_cell


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif step 4: acceptance tests and reported checks.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "acceptance", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "acceptance_checks")

    conds = {c: lib.load_condition(run, c) for c in lib.CONDITIONS}
    P, _ = lib.prolif_set(conds["Ctrl"], conds["ZeroK"])
    cands = {c: lib.identity_candidates(run, conds[c].var_names)[0] for c in lib.CONDITIONS}
    per_cell = lib.read_table(run, "route_a_per_cell", dtype={"cell_id": str})
    t27_emb, flags = {}, {}
    for cond in lib.CONDITIONS:
        o = ad.read_h5ad(run.use_input(lib.DCT_LABELS_H5AD[cond], f"dct_labels labelled object ({cond})"))
        if list(o.obs_names.astype(str)) != list(conds[cond].cell_ids):
            raise ValueError(f"{cond}: dct_labels object cell order differs from dct_intake")
        t27_emb[cond] = np.asarray(o.obsm["X_pca_neighbors"])[:, :20]
        o3 = ad.read_h5ad(run.use_input(lib.DCT_REVIEWER_TASK3_H5AD[cond], f"dct_reviewer task-3 object ({cond})"))
        if list(o3.obs_names.astype(str)) != list(conds[cond].cell_ids):
            raise ValueError(f"{cond}: dct_reviewer object cell order differs from dct_intake")
        flags[cond] = o3.obs[list(t2.FLAG_COLUMNS)].astype(bool).reset_index(drop=True)

    tests, summ, t4_all, t5_all, odds, post_rows, noise_rows, noise_cells = [], [], [], [], [], [], [], []
    reported, contam, purity = [], [], []
    for run_key, spec in lib.RUNS.items():
        for cond in lib.CONDITIONS:
            t0 = time.time()
            c = conds[cond]
            f = per_cell[(per_cell["run"] == run_key) & (per_cell["condition"] == cond)].reset_index(drop=True)
            if list(f["cell_id"]) != list(c.cell_ids):
                raise ValueError(f"{run_key} / {cond}: route-A table cells differ from the object")
            for b in ("above_floor", "slc12a3_detected", "mki67_or_top2a_detected"):
                f[b] = f[b].astype(str).str.lower().eq("true")
            pr = lib.prolif_setup(c, spec.floor, P)
            a = f["above_floor"].to_numpy(bool)
            if not np.array_equal(pr.above, a):
                raise ValueError("floor mismatch")
            check = lib.cycling_call(pr, pr.P_idx, np.random.default_rng(lib.SEED))
            diff = float(np.abs(check["p"] - f.loc[a, "prolif_p"].to_numpy()).max())
            checks.add(f"{run_key} / {cond}: recomputed seed-42 p values equal the route-A table (CSV round trip)",
                       f"max abs diff {diff:.2g}", diff <= 1e-12, "<= 1e-12")
            n_cyc = int((f["cycling"] == lib.CYCLING).sum())

            rows13, extra = tests_1_3(run_key, cond, f)
            row4, t4 = test_4(run_key, cond, pr, n_cyc, c.var_names)
            row5, t5_ = test_5(run_key, cond, pr, f)
            rows = [*rows13, row4, row5]
            tests.extend(rows)
            t4_all.append(t4)
            t5_all.append(t5_)
            ok = all(r["pass"] for r in rows)
            n_pro = int((f["label"] == lib.PROLIF).sum())
            summ.append({"run": run_key, "condition": cond, "floor": spec.floor, "n_prolif": n_pro,
                         "n_cycling": n_cyc, **{f"test{r['test']}_pass": r["pass"] for r in rows},
                         "failed_tests": ";".join(str(r["test"]) for r in rows if not r["pass"]),
                         "prolif_passes": ok,
                         "outcome": ("pass: offered as a candidate primary label for this condition" if ok else
                                     "fail: dct_labels labels stay (no retuning)")})
            for r in rows:
                checks.add(f"{run_key} / {cond}: test {r['test']} ({r['name'].split(':')[0]})",
                           f"{r['value']} [{r['threshold']}] {r['note']}".strip(), None)
            checks.add(f"{run_key} / {cond}: ProLIF passes all five tests", f"{ok} (ProLIF {n_pro})", None)

            # DCT1 / DCT2 reports -----------------------------------------------------------------------------
            for column in ("identity", "label"):
                m = a & f[column].isin([lib.DCT1, lib.DCT2]).to_numpy()
                fit = lib.logit_or((f.loc[m, column] == lib.DCT2).to_numpy(), np.log(f.loc[m, "n_detected"].to_numpy(float)),
                                   f.loc[m, "plate"].to_numpy())
                odds.append({"run": run_key, "condition": cond, "column": column, "model": "DCT2 ~ z(ln n_detected) + plate",
                             "n_DCT1": int((f.loc[m, column] == lib.DCT1).sum()),
                             "n_DCT2": int((f.loc[m, column] == lib.DCT2).sum()), **fit})
            post = f.loc[a & f["axis"].notna().to_numpy(), "posterior"].to_numpy(float)
            hist = np.histogram(post, bins=POSTERIOR_BINS)[0]
            for lo, hi, n in zip(POSTERIOR_BINS[:-1], POSTERIOR_BINS[1:], hist):
                post_rows.append({"run": run_key, "condition": cond, "posterior_from": lo, "posterior_to": min(hi, 1.0),
                                  "n_cells": int(n), "fraction": float(n / len(post))})
            post_rows.append({"run": run_key, "condition": cond, "posterior_from": "median", "posterior_to": "",
                              "n_cells": len(post), "fraction": float(np.median(post))})
            used = {t: [g for g in cands[cond][t] if c.detected[a, c.var_names.get_loc(g)].sum() >= lib.MIN_CELLS_MARKER]
                    for t in (lib.DCT1, lib.DCT2)}
            nr, nc = noise_identity(run_key, cond, c, f, used)
            noise_rows.append(nr)
            noise_cells.append(nc)
            checks.add(f"{run_key} / {cond}: no-noise control reproduces the identity calls", "True", True)
            all_kept = int((nc["n_repeats_same"] == lib.NOISE_REPEATS).sum())
            checks.add(f"{run_key} / {cond}: identity calls kept in all 20 noise repeats",
                       f"{all_kept}/{len(nc)} ({all_kept / len(nc):.3f}); cells changed per repeat max "
                       f"{nr.loc[nr['repeat'] > 0, 'n_changed'].max()}", None)

            # Other reported checks ---------------------------------------------------------------------------
            held = f["mki67_or_top2a_detected"].to_numpy(bool) & a
            cyc = (f["cycling"] == lib.CYCLING).to_numpy()
            ncc = f["slc12a3_detected"].to_numpy(bool)
            pro = (f["label"] == lib.PROLIF).to_numpy()
            rep = {"run": run_key, "condition": cond, "floor": spec.floor,
                   "n_mki67_or_top2a_detected_above": int(held.sum()),
                   "n_mki67_or_top2a_called_cycling": int((held & cyc).sum()),
                   "share_mki67_or_top2a_called_cycling": float((held & cyc).sum() / held.sum()) if held.any() else np.nan,
                   "n_cycling": int(cyc.sum()), "n_cycling_slc12a3_detected": int((cyc & ncc).sum()),
                   "share_cycling_with_slc12a3": float((cyc & ncc).sum() / cyc.sum()) if cyc.any() else np.nan}
            for plate in ("P1", "P2"):
                pm = f["plate"].to_numpy() == plate
                rep[f"n_prolif_{plate}"] = int((pro & pm).sum())
                rep[f"n_above_{plate}"] = int((a & pm).sum())
                rep[f"fraction_prolif_{plate}"] = float((pro & pm).sum() / (a & pm).sum())
            ok_i = a & f["prolif_intensity"].notna().to_numpy()
            rho = stats.spearmanr(f.loc[ok_i, "prolif_intensity"], f.loc[ok_i, "prolif_D"])
            rep.update(n_intensity_part_defined=int(ok_i.sum()), spearman_intensity_vs_D=float(rho.statistic),
                       spearman_p=float(rho.pvalue))
            for name, m in (("cycling", ok_i & cyc), ("non_cycling", ok_i & ~cyc)):
                if m.sum() >= 3:
                    r2 = stats.spearmanr(f.loc[m, "prolif_intensity"], f.loc[m, "prolif_D"])
                    rep[f"spearman_intensity_vs_D_{name}"] = float(r2.statistic)
                rep[f"median_intensity_part_{name}"] = float(f.loc[m, "prolif_intensity"].median()) if m.any() else np.nan
            ids = f["n_detected"].to_numpy(float)
            rep.update(median_ids_prolif=float(np.median(ids[pro])) if pro.any() else np.nan,
                       median_ids_other_test_set=float(np.median(ids[a & f["label"].isin(lib.TEST_SET).to_numpy() & ~pro])))
            reported.append(rep)

            fl = flags[cond]
            for label in [*cc.ordered_labels(f["label"], lib.LABEL_ORDER), "all"]:
                m = np.ones(len(f), bool) if label == "all" else (f["label"] == label).to_numpy()
                row = {"run": run_key, "condition": cond, "label": label, "n_cells": int(m.sum())}
                for col_ in t2.FLAG_COLUMNS:
                    row[f"n_{col_}"] = int(fl.loc[m, col_].sum())
                    row[f"fraction_{col_}"] = float(fl.loc[m, col_].mean())
                anyf = fl[list(t2.FLAG_COLUMNS)].any(axis=1).to_numpy()
                row["n_any_flag"], row["fraction_any_flag"] = int(anyf[m].sum()), float(anyf[m].mean())
                contam.append(row)

            lab_frame = pd.DataFrame({"label": f["label"].astype(str).to_numpy()})
            kp = t2.knn_purity({"dct_labels graph (X_pca_neighbors, 20 PCs; all cells)": t27_emb[cond]}, lab_frame, ["label"],
                               k=10, n_perm=1000)
            kp.insert(0, "condition", cond)
            kp.insert(0, "run", run_key)
            purity.append(kp)
            pr_row = kp[kp["label"] == lib.PROLIF]
            if len(pr_row):
                r = pr_row.iloc[0]
                checks.add(f"{run_key} / {cond}: ProLIF kNN purity, dct_labels graph",
                           f"{r.observed_same_label_fraction:.3f} (null {r.null_mean:.3f}; p {r.p_empirical_ge:.4f})", None)
            print(f"[{run_key} / {cond}] done in {time.time() - t0:.1f} s", flush=True)

    run.table(pd.DataFrame(tests), "acceptance_tests")
    run.table(pd.DataFrame(summ), "acceptance_summary")
    run.table(pd.concat(t4_all, ignore_index=True), "test4_random_sets")
    run.table(pd.concat(t5_all, ignore_index=True), "test5_null_seeds")
    run.table(pd.DataFrame(odds), "identity_odds")
    run.table(pd.DataFrame(post_rows), "posterior_distribution")
    run.table(pd.concat(noise_rows, ignore_index=True), "noise_identity")
    run.table(pd.concat(noise_cells, ignore_index=True), "noise_identity_per_cell")
    run.table(pd.DataFrame(reported), "reported_checks")
    run.table(pd.DataFrame(contam), "contamination_by_label")
    run.table(pd.concat(purity, ignore_index=True), "knn_purity_t27_graph")
    run.table(checks.frame(), "acceptance_checks")
    run.finish({"runs": {k: vars(v) for k, v in lib.RUNS.items()}, "approved_change": lib.APPROVED_CHANGE,
                "test4_seed_sequence": list(TEST4_SEED), "test4_null_seed": "[42, 4, set index]",
                "test5_null_seeds": list(lib.NULL_SEEDS), "noise": {"repeats": lib.NOISE_REPEATS,
                "relative_sd": lib.NOISE_REL_SD, "seed": lib.SEED}, "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_acceptance_checks.csv.")


if __name__ == "__main__":
    main()
