#!/usr/bin/env python
"""The response to the reviewers: Slc12a1 and Aqp2 as negative markers in the DCT data - detection per label, cluster and
protein depth, and the annotation after excluding the cells in which they are detected.

Run at the authors' request (11 Nov 2026, the response to the reviewers: "Run the
check") for reviewer 2, major comment 2 ("Slc12a1 to exclude TAL and Aqp2 to exclude CNT"). The stored dct_reviewer tables
report negative-panel scores and flags, but no analysis removed the cells with Slc12a1 or Aqp2 detected, and dct_reviewer has
no per-label Aqp2 table. Everything is within condition (Ctrl and ZeroK side by side); nothing compares the
conditions (zerok_vs_control). One pool of 3 mice per condition; P1/P2 are sort plates of the pooled cells.

Part A, detection (layers['intensity'] > 0 in the dct_labels labelled objects of the combined search):
  per cell; per label (primary: cs_ctrl_deg_label / cs_zerok_pipeline_label; second: cs_<cond>_cluster_label; all
  cells) with Wilson 95% intervals and the protein IDs of detected against not-detected cells (Mann-Whitney U, AUC =
  U / (n1 * n2)); per primary Leiden 1.0 cluster; per quartile of protein IDs within condition; and detection of
  Slc12a3, Pvalb, Umod, Egf, Klk1 and Calb1 in the cells with and without each negative marker.
Part B, exclusion (the dct_reviewer step 9c / qc_mito addendum E code, imported unchanged from
  dct_reviewer_exclusion_sensitivity: Context, evaluate, reproduction_checks): three exclusion sets (Slc12a1 detected;
  Aqp2 detected; either). After removing a set, the remaining cells are re-scaled and the labels re-derived: the
  primary label by the frozen dct_labels rule (cluster_de DE rule + kidney_annotation rank-free rescue; Slc12a3 and Umod shared; minimum cluster
  size 5) on the stored Leiden 1.0 clusters and on a Leiden 1.0 rerun of the rebuilt graph, the second label and the
  per-cell label as in dct_reviewer. Control: 100 random removals of the same number of cells, stratified by the primary
  label (numpy default_rng(42), one stream per condition). Baseline and observed runs use Leiden seeds 0, 1, 2, 3,
  42; random removals use seed 0, and the comparison with the random distribution uses seed 0 (as qc_mito, ctrl_reviewer_core, dct_reviewer).
  Resolution, rule and markers are not changed; the dct_labels labels stay the labels of record.

Inputs (read-only): dct_labels labelled objects and canonical labels; data/DCT/DCT_markers_V1.xlsx; for checks
  dct_reviewer_task3_individual_marker_coverage.csv, dct_reviewer_umod_slc12a1_codetection.csv,
  dct_labels_deg_cluster_annotation.csv.
Outputs (results/tables/report/): report_09_negative_marker_{per_cell,by_label,by_cluster,depth,
  codetection,exclusion_sets,exclusion_runs,exclusion_vs_random,exclusion_label_counts,exclusion_clusters,
  exclusion_checks}.csv; report_step_09_negative_marker_exclusion_manifest.json

Command (project root, about 25 min):  python scripts/report_negative_markers_exclusion.py
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
from scipy import stats

import scp_common as cc
import qc_mito_sensitivity as t3s
import cluster_de_lib as t5
import labels_core_lib as lib18
import dct_reviewer_lib as lib
import dct_reviewer_exclusion_sensitivity as ex   # dct_reviewer step 9c: Context, evaluate, reproduction_checks (unchanged)

TASK = "report"
STEP = "09_negative_marker_exclusion"
P = "09_negative_marker"
NEGATIVE = {"Slc12a1": "TAL", "Aqp2": "CNT / collecting duct"}          # the two proteins named by the reviewer
CO_MARKERS = ["Slc12a3", "Pvalb", "Umod", "Egf", "Klk1", "Calb1"]
MARKER_SETS = {"Slc12a1": ("Slc12a1",), "Aqp2": ("Aqp2",), "Slc12a1_or_Aqp2": ("Slc12a1", "Aqp2")}
EXCLUSIONS = {"slc12a1_detected": "Slc12a1", "aqp2_detected": "Aqp2", "slc12a1_or_aqp2_detected": "Slc12a1_or_Aqp2"}
OBS_SEEDS = ex.OBS_SEEDS          # 0, 1, 2, 3, 42
S = ex.NULL_SEED                  # 0
N_DRAWS = ex.N_DRAWS              # 100
OUTCOMES = {
    "deg_stored_agree": "primary label agreement (DE rule on the stored Leiden 1.0 clusters, remaining cells)",
    f"deg_rerun_agree_s{S}": f"primary label agreement (Leiden 1.0 rerun on the rebuilt graph, seed {S})",
    f"ari1_s{S}": f"ARI, Leiden 1.0 rerun (seed {S}) vs stored clusters",
    "cluster_stored_agree": "second label agreement (stored Leiden 3.0 clusters re-named)",
    f"cluster_rerun_agree_s{S}": f"second label agreement (Leiden 3.0 rerun, seed {S})",
    f"ari3_s{S}": f"ARI, Leiden 3.0 rerun (seed {S}) vs stored clusters",
    "cell_agree": "per-cell label agreement (re-scored)",
}
DCT_REVIEWER_COVERAGE = lib.table_path("task3_individual_marker_coverage")
DCT_REVIEWER_CODETECTION = lib.table_path("umod_slc12a1_codetection")
DCT_LABELS_CLUSTERS = cc.out_path("dct_labels", "tables", "deg_cluster_annotation", ".csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Slc12a1 / Aqp2 detection and exclusion check.")
    cc.add_common_args(parser)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    return parser.parse_args()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Part A: detection
# ---------------------------------------------------------------------------
def detection(cond: str, adata, labels: pd.DataFrame) -> pd.DataFrame:
    s = lib.SETTINGS[cond]
    names = pd.Index(adata.var_names.astype(str))
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    out = pd.DataFrame({
        "condition": cond, "cell_id": labels.index.to_numpy(), "cell_key": labels["cell_key"].astype(str).to_numpy(),
        "plate": labels["plate"].astype(str).to_numpy(),
        "primary_label": labels[s.PRIMARY].astype(str).to_numpy(),
        "primary_cluster": labels[s.PRIMARY_CLUSTER].astype(str).to_numpy(),
        "primary_confidence": labels[s.CONFIDENCE].astype(str).to_numpy(),
        "second_label": labels[s.SECOND].astype(str).to_numpy(),
        "n_detected_proteins": labels["n_detected_proteins"].to_numpy(int)})
    for gene in (*NEGATIVE, *CO_MARKERS):
        if gene not in names or int((names == gene).sum()) != 1:
            raise SystemExit(f"{cond}: {gene} is not a unique protein name in the dct_labels object.")
        v = intensity[:, names.get_loc(gene)]
        out[f"det_{gene}"] = v > 0
        out[f"ln1p_{gene}"] = np.log1p(v)
    out["det_Slc12a1_or_Aqp2"] = out["det_Slc12a1"] | out["det_Aqp2"]
    out["det_Slc12a1_and_Aqp2"] = out["det_Slc12a1"] & out["det_Aqp2"]
    out["protein_id_quartile"] = pd.qcut(out["n_detected_proteins"], 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    return out


def by_label(cond: str, det: pd.DataFrame) -> pd.DataFrame:
    s = lib.SETTINGS[cond]
    groups = [("all_cells", "all_cells", "all_cells", np.ones(len(det), bool))]
    for role, col in (("primary", "primary_label"), ("second", "second_label")):
        for lab in cc.ordered_labels(set(det[col])):
            groups.append((role, s.COLUMN_OF[role], lab, (det[col] == lab).to_numpy()))
    ids = det["n_detected_proteins"].to_numpy(float)
    rows = []
    for role, column, lab, m in groups:
        n = int(m.sum())
        for key in ("Slc12a1", "Aqp2", "Slc12a1_or_Aqp2", "Slc12a1_and_Aqp2"):
            d = det[f"det_{key}"].to_numpy() & m
            other = m & ~det[f"det_{key}"].to_numpy()
            k = int(d.sum())
            lo, hi = wilson(k, n)
            row = {"condition": cond, "label_role": role, "label_column": column, "label": lab, "marker_set": key,
                   "n_cells": n, "n_detected": k, "fraction_detected": k / n if n else np.nan,
                   "ci_low": lo, "ci_high": hi,
                   "median_protein_ids_detected": float(np.median(ids[d])) if k else np.nan,
                   "median_protein_ids_not_detected": float(np.median(ids[other])) if other.any() else np.nan,
                   "auc_protein_ids_detected_vs_not": np.nan, "p_mannwhitney_protein_ids": np.nan}
            if k and other.any():
                u, p = stats.mannwhitneyu(ids[d], ids[other], alternative="two-sided")
                row.update(auc_protein_ids_detected_vs_not=float(u / (k * other.sum())), p_mannwhitney_protein_ids=float(p))
            rows.append(row)
    return pd.DataFrame(rows)


def by_cluster(cond: str, det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster in sorted(det["primary_cluster"].unique(), key=int):
        g = det[det["primary_cluster"] == cluster]
        row = {"condition": cond, "primary_cluster": cluster, "n_cells": len(g),
               "primary_label": ";".join(sorted(g["primary_label"].unique())),
               "primary_confidence": ";".join(sorted(g["primary_confidence"].unique())),
               "median_protein_ids": float(g["n_detected_proteins"].median())}
        for key in ("Slc12a1", "Aqp2", "Slc12a1_or_Aqp2"):
            k = int(g[f"det_{key}"].sum())
            row.update({f"n_{key}": k, f"fraction_{key}": k / len(g)})
        rows.append(row)
    return pd.DataFrame(rows)


def by_depth(cond: str, det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in ("Q1", "Q2", "Q3", "Q4"):
        g = det[det["protein_id_quartile"] == q]
        for key in ("Slc12a1", "Aqp2", "Slc12a1_or_Aqp2"):
            k = int(g[f"det_{key}"].sum())
            lo, hi = wilson(k, len(g))
            rows.append({"condition": cond, "protein_id_quartile": q,
                         "protein_ids_min": int(g["n_detected_proteins"].min()),
                         "protein_ids_max": int(g["n_detected_proteins"].max()), "marker_set": key,
                         "n_cells": len(g), "n_detected": k, "fraction_detected": k / len(g), "ci_low": lo,
                         "ci_high": hi})
    return pd.DataFrame(rows)


def codetection(cond: str, det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q4 = (det["protein_id_quartile"] == "Q4").to_numpy()
    for gene in NEGATIVE:
        d = det[f"det_{gene}"].to_numpy()
        for group, m in ((f"{gene} detected", d), (f"{gene} not detected", ~d),
                         (f"{gene} not detected, protein-ID quartile Q4", ~d & q4)):
            g = det[m]
            row = {"condition": cond, "negative_marker": gene, "group": group, "n_cells": len(g),
                   "median_protein_ids": float(g["n_detected_proteins"].median()) if len(g) else np.nan}
            for co in (*CO_MARKERS, *[x for x in NEGATIVE if x != gene]):
                k = int(g[f"det_{co}"].sum())
                row.update({f"n_{co}_detected": k, f"fraction_{co}_detected": k / len(g) if len(g) else np.nan})
            ncc = g.loc[g["det_Slc12a3"], "ln1p_Slc12a3"]
            row["median_ln1p_Slc12a3_in_detected_cells"] = float(ncc.median()) if len(ncc) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def stored_table_checks(run: cc.Run, checks: t5.Checks, cond: str, det: pd.DataFrame) -> None:
    cov = pd.read_csv(run.use_input(DCT_REVIEWER_COVERAGE, "dct_reviewer individual marker coverage"))
    cov = cov[(cov["condition"] == cond) & (cov["marker"] == "Slc12a1")]
    k = int(det["det_Slc12a1"].sum())
    checks.add(f"{cond}: cells with Slc12a1 detected equal dct_reviewer_task3_individual_marker_coverage.csv",
               f"{k} vs {int(cov['n_cells_detected'].iloc[0])}", k == int(cov["n_cells_detected"].iloc[0]))
    co = pd.read_csv(run.use_input(DCT_REVIEWER_CODETECTION, "dct_reviewer Umod / Slc12a1 co-detection"))
    co = co[(co["condition"] == cond) & (co["label_role"] == "primary")].set_index("label")
    mine = det.groupby("primary_label")["det_Slc12a1"].sum().astype(int).to_dict()
    theirs = co["n_slc12a1_detected"].astype(int).to_dict()
    checks.add(f"{cond}: Slc12a1 by primary label equals dct_reviewer_umod_slc12a1_codetection.csv",
               "; ".join(f"{k_} {v}" for k_, v in mine.items()), mine == theirs)
    no_umod = int((det["det_Slc12a1"] & ~det["det_Umod"]).sum())
    checks.add(f"{cond}: Slc12a1-detected cells without Umod (dct_reviewer: 0)", no_umod, no_umod == 0)
    cl = pd.read_csv(run.use_input(DCT_LABELS_CLUSTERS, "dct_labels primary-rule cluster annotation"), dtype={"cluster": str})
    cl = cl[cl["condition"] == cond].set_index("cluster")
    ok = True
    for gene, col in (("Slc12a1", "detected_TAL_Slc12a1"), ("Aqp2", "detected_CNT_PC_Aqp2")):
        mine_c = det.groupby("primary_cluster")[f"det_{gene}"].mean()
        diff = float(np.max(np.abs(mine_c.reindex(cl.index).to_numpy() - cl[col].to_numpy(float))))
        ok = ok and diff < 1e-9
    checks.add(f"{cond}: Slc12a1 and Aqp2 detected fractions per primary cluster equal "
               "dct_labels_deg_cluster_annotation.csv", f"max abs diff < 1e-9: {ok}", ok)
    for key in ("Slc12a1", "Aqp2", "Slc12a1_or_Aqp2", "Slc12a1_and_Aqp2"):
        checks.add(f"{cond}: cells with {key} detected", f"{int(det[f'det_{key}'].sum())} of {len(det)}", None)


# ---------------------------------------------------------------------------
# Part B: exclusion against random removals
# ---------------------------------------------------------------------------
def label_count_rows(cond: str, run_name: str, exclusion: str, n_removed: int, frame: pd.DataFrame) -> list[dict]:
    rows = []
    for col in frame.columns:
        counts = frame[col].astype(str).value_counts()
        for lab in cc.ordered_labels(set(counts.index)):
            rows.append({"condition": cond, "run": run_name, "exclusion_set": exclusion, "n_removed": n_removed,
                         "n_remaining": len(frame), "label_source": col, "label": lab,
                         "n_cells": int(counts[lab]), "fraction": float(counts[lab] / len(frame))})
    return rows


def cluster_rows(cond: str, exclusion: str, s, labels: pd.DataFrame, mask: np.ndarray, cells: pd.DataFrame) -> list[dict]:
    after = pd.Series(cells["primary_label_stored_clusters"].astype(str).to_numpy(), index=cells["cell_id"].to_numpy())
    cl = labels[s.PRIMARY_CLUSTER].astype(str)
    before = labels[s.PRIMARY].astype(str)
    removed = pd.Series(mask, index=labels.index)
    rows = []
    for cluster in sorted(cl.unique(), key=int):
        idx = cl.index[(cl == cluster).to_numpy()]
        rem = idx[~removed[idx].to_numpy()]
        lab_before = ";".join(sorted(before[idx].unique()))
        lab_after = ";".join(sorted(after.reindex(rem).unique())) if len(rem) else "no cells left"
        rows.append({"condition": cond, "exclusion_set": exclusion, "primary_cluster": cluster, "n_cells": len(idx),
                     "n_removed": len(idx) - len(rem), "n_remaining": len(rem), "label_before": lab_before,
                     "label_after_stored_clusters": lab_after, "label_changed": lab_before != lab_after})
    return rows


def vs_random(cond: str, runs: pd.DataFrame) -> pd.DataFrame:
    base = runs[runs["run_type"] == "baseline"].iloc[0]
    rows = []
    for exclusion in runs.loc[runs["run_type"] == "observed", "exclusion_set"]:
        obs = runs[(runs["run_type"] == "observed") & (runs["exclusion_set"] == exclusion)].iloc[0]
        null = runs[(runs["run_type"] == "random") & (runs["exclusion_set"] == exclusion)]
        for outcome, display in OUTCOMES.items():
            x = null[outcome].to_numpy(float)
            o = float(obs[outcome])
            row = {"condition": cond, "exclusion_set": exclusion, "outcome": outcome, "display": display,
                   "n_removed": int(obs["n_removed"]), "n_remaining": int(obs["n_remaining"]),
                   "baseline_all_cells": float(base[outcome]), "observed": o, "random_n": len(x)}
            if len(x):
                p_le = (1 + (x <= o).sum()) / (len(x) + 1)
                p_ge = (1 + (x >= o).sum()) / (len(x) + 1)
                row.update(random_median=float(np.median(x)), random_q025=float(np.quantile(x, 0.025)),
                           random_q975=float(np.quantile(x, 0.975)), random_min=float(x.min()),
                           random_max=float(x.max()), observed_percentile_in_random=100 * float((x < o).mean()),
                           p_empirical_two_sided=min(1.0, 2 * min(p_le, p_ge)),
                           inside_central_95=bool(np.quantile(x, 0.025) <= o <= np.quantile(x, 0.975)))
            rows.append(row)
    return pd.DataFrame(rows)


def one_condition(run: cc.Run, checks: t5.Checks, cond: str, n_draws: int, t0: float) -> dict:
    s = lib.SETTINGS[cond]
    rng = np.random.default_rng(cc.RANDOM_SEED)
    adata, labels = lib.load(run, cond)
    det = detection(cond, adata, labels)
    stored_table_checks(run, checks, cond, det)
    lists = lib18.present_lists(lib18.read_workbook(run), adata.var_names)
    markers = lib18.dct_markers(run, adata.var_names)
    ctx = ex.Context(s, adata, labels, lists, markers)
    strata = labels[s.PRIMARY].astype(str).to_numpy()
    second = labels[s.SECOND].astype(str).to_numpy()

    base, cells = ex.evaluate(ctx, np.zeros(len(strata), bool), OBS_SEEDS)
    for r in ex.reproduction_checks(cond, s, adata, labels, base):
        checks.add(f"{cond}: {r['check']}", r["value"], bool(r["pass"]))
    print(f"{cond}: baseline done ({time.time() - t0:.0f} s)", flush=True)
    if not checks.ok:
        return {"failed": True}
    rows = [{"condition": cond, "run_type": "baseline", "exclusion_set": "none", "draw": "", **base}]
    counts = label_count_rows(cond, "baseline", "none", 0, cells.drop(columns="cell_id"))
    sets, clusters = [], []
    for exclusion, key in EXCLUSIONS.items():
        mask = det[f"det_{key}"].to_numpy()
        if not mask.any():
            run.note(f"{cond} {exclusion}: no cell in the set; skipped.")
            continue
        res, cells = ex.evaluate(ctx, mask, OBS_SEEDS)
        rows.append({"condition": cond, "run_type": "observed", "exclusion_set": exclusion, "draw": "", **res})
        keep = ~mask
        t27_remaining = pd.DataFrame({"t27_primary_label_remaining_cells": strata[keep],
                                      "t27_second_label_remaining_cells": second[keep]})
        counts += label_count_rows(cond, f"without_{exclusion}", exclusion, int(mask.sum()),
                                   pd.concat([t27_remaining, cells.drop(columns="cell_id")], axis=1))
        crows = cluster_rows(cond, exclusion, s, labels, mask, cells)
        clusters += crows
        changed = int((cells["primary_label_stored_clusters"].astype(str).to_numpy() != strata[keep]).sum())
        rerun = np.array([res[f"deg_rerun_agree_s{seed}"] for seed in OBS_SEEDS], float)
        sets.append({"condition": cond, "exclusion_set": exclusion, "marker_set": key, "n_excluded": int(mask.sum()),
                     "n_remaining": int(keep.sum()),
                     "excluded_by_primary_label": "; ".join(
                         f"{k}: {v}" for k, v in pd.Series(strata[mask]).value_counts().items()),
                     "excluded_by_second_label": "; ".join(
                         f"{k}: {v}" for k, v in pd.Series(second[mask]).value_counts().items()),
                     "n_primary_label_changed_stored_clusters": changed,
                     "n_clusters_label_changed": int(sum(r["label_changed"] for r in crows)),
                     "primary_rerun_agreement_seeds": ";".join(map(str, OBS_SEEDS)),
                     "primary_rerun_agreement_min": float(rerun.min()),
                     "primary_rerun_agreement_median": float(np.median(rerun)),
                     "primary_rerun_agreement_max": float(rerun.max())})
        for draw in range(n_draws):
            removed = t3s.stratified_draw(rng, strata, mask)
            res, _ = ex.evaluate(ctx, removed, (S,))
            rows.append({"condition": cond, "run_type": "random", "exclusion_set": exclusion, "draw": draw,
                         "n_set_cells_among_removed": int((removed & mask).sum()), **res})
            if (draw + 1) % 20 == 0:
                print(f"  {cond} {exclusion}: {draw + 1} draws ({time.time() - t0:.0f} s)", flush=True)
        run.note(f"{cond} {exclusion}: {int(mask.sum())} cells removed; {n_draws} random removals done "
                 f"({time.time() - t0:.0f} s since start).")
    runs = pd.DataFrame(rows)
    base_rerun = np.array([base[f"deg_rerun_agree_s{seed}"] for seed in OBS_SEEDS], float)
    checks.add(f"{cond}: all cells, primary label agreement of the Leiden 1.0 rerun per seed "
               f"({';'.join(map(str, OBS_SEEDS))}; seed-to-seed reference for the rerun outcomes)",
               "; ".join(f"{v:.4f}" for v in base_rerun), None)
    return {"failed": False, "det": det, "by_label": by_label(cond, det), "by_cluster": by_cluster(cond, det),
            "depth": by_depth(cond, det), "codetection": codetection(cond, det), "runs": runs,
            "summary": vs_random(cond, runs), "sets": pd.DataFrame(sets), "counts": pd.DataFrame(counts),
            "clusters": pd.DataFrame(clusters)}


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    t0 = time.time()
    res = {}
    for cond in lib.CONDITIONS:
        res[cond] = one_condition(run, checks, cond, args.n_draws, t0)
        if res[cond]["failed"]:
            run.table(checks.frame(), f"{P}_exclusion_checks")
            run.finish({"checks_pass": False}, status="stopped_failed_checks")
            raise SystemExit(f"{cond}: a reproduction check failed; see report_{P}_exclusion_checks.csv.")
    stack = lambda key: pd.concat([res[c][key] for c in lib.CONDITIONS], ignore_index=True)  # noqa: E731
    run.table(stack("det"), f"{P}_per_cell")
    run.table(stack("by_label"), f"{P}_by_label")
    run.table(stack("by_cluster"), f"{P}_by_cluster")
    run.table(stack("depth"), f"{P}_depth")
    run.table(stack("codetection"), f"{P}_codetection")
    run.table(stack("sets"), f"{P}_exclusion_sets")
    run.table(stack("runs"), f"{P}_exclusion_runs")
    run.table(stack("summary"), f"{P}_exclusion_vs_random")
    run.table(stack("counts"), f"{P}_exclusion_label_counts")
    run.table(stack("clusters"), f"{P}_exclusion_clusters")
    run.table(checks.frame(), f"{P}_exclusion_checks")
    run.finish({"request": "authors, 11 Nov 2026: 'Run the check'",
                "negative_markers": NEGATIVE, "co_markers": CO_MARKERS, "exclusion_sets": EXCLUSIONS,
                "detection": "layers['intensity'] > 0", "observed_leiden_seeds": list(OBS_SEEDS),
                "random_leiden_seed": S, "n_random_draws": args.n_draws,
                "random_removals": "stratified by the primary label; numpy default_rng(42), one stream per condition",
                "labelling": "dct_reviewer_exclusion_sensitivity.evaluate (dct_labels rule; Leiden 1.0 / 3.0, igraph)",
                "design": lib.DESIGN_NOTE, "checks_pass": checks.ok, "runtime_s": round(time.time() - t0)},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{P}_exclusion_checks.csv.")


if __name__ == "__main__":
    main()
