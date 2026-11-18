#!/usr/bin/env python
"""dct_reviewer step 9c: qc_mito addendum E per DCT condition (combined search) - labels and PAGA after excluding
QC-flagged cells, against random removals of the same number of cells.

Copy of ctrl_reviewer_core_exclusion_sensitivity.py, run once per
condition. The ctrl_reviewer_core functions read the labels_core column names directly (ctrl_cell_label, ctrl_cluster_label,
ctrl_deg_label, ctrl_leiden3_cluster, ctrl_deg_cluster), so Context, evaluate, exclusion_masks, composition,
null_summary and seed_summary are copied with the columns taken from each condition's settings
(dct_reviewer_lib.SETTINGS: CELL_COLUMN, SECOND, PRIMARY, LEIDEN3_CLUSTER, PRIMARY_CLUSTER). Generic pieces are
imported unchanged from qc_mito_sensitivity (stratified_draw, leiden, paga, frac, share, the igraph message
counter) and labels_core_lib (graph_adata, score_types, cell_labels, cluster_labels, annotate_clustering, per_cell,
present_lists, dct_markers). The labelling is the dct_labels one, so the all-cells run reproduces the dct_labels labels:
  - matrix: float32 intensities of the remaining cells -> sc.pp.log1p -> sc.pp.scale(max_value=10), fixed protein set;
  - graph: sc.pp.neighbors(n_neighbors=10, n_pcs=20) without a prior PCA (Scanpy fallback);
  - per-cell label: score_genes on the re-scaled matrix -> per-cell argmax;
  - second label: cluster mode with the original 'Unknown' rule, on the stored Leiden 3.0 clusters (restricted to the
    remaining cells) and on a Leiden 3.0 rerun;
  - primary label: cluster_de DE rule + kidney_annotation rank-free rescue (minimum cluster size 5; Slc12a3 and Umod shared) on the stored
    Leiden 1.0 clusters (restricted) and on a Leiden 1.0 rerun.
Exclusion sets (union over the primary and second label columns, from step 9a): mito_flagged (high or low on any
mitochondrial metric) and any_flagged (any qc_mito metric). Control: 100 random removals of the same number of cells,
stratified by the primary label (numpy default_rng(42), one stream per condition). Baseline and observed runs use
Leiden seeds 0, 1, 2, 3, 42; random removals use seed 0, and the comparison with the random distribution uses seed
0 throughout (as qc_mito and ctrl_reviewer_core).

dct_reviewer addition: outcomes that need a label absent at baseline under their column (ProLIF under either primary column;
DCT2 under the ZeroK primary) are computed but marked applicable = False ("does not apply (label absent at
baseline)"), with a comparator outcome beside them: the second label for ProLIF shares, the per-cell label for PAGA
pairs (ctrl_reviewer_core reports PAGA on the per-cell and the primary label only).

Inputs (read-only): dct_labels labelled objects and canonical labels; dct_reviewer_mito_qc_per_cell.csv (step 9a);
data/DCT/DCT_markers_V1.xlsx.
Outputs (results/tables/dct_reviewer/, results/figures/dct_reviewer/; every table has `condition`):
  dct_reviewer_exclusion_{sets,composition,runs,vs_random,leiden_seeds,per_cell_labels,checks}.csv
  dct_reviewer_exclusion_vs_random.pdf/.png/.tif

Command (project root, about 15 min):  python scripts/dct_reviewer_exclusion_sensitivity.py
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
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import qc_mito_lib as t3
import qc_mito_sensitivity as t3s
import cluster_de_lib as t5
import labels_core_lib as lib18
import dct_reviewer_lib as lib
from kidney_annotation_rankfree_sensitivity import rank_free

ensure_scanpy_compat()
lib.style()
import matplotlib.pyplot as plt  # noqa: E402,F401

SCRIPT = "dct_reviewer_exclusion_sensitivity.py"
OBS_SEEDS = t3s.OBS_SEEDS            # 0, 1, 2, 3, 42
NULL_SEED = t3s.NULL_LEIDEN_SEED     # 0
N_DRAWS = t3s.N_DRAWS                # 100
EXCLUSIONS = t3s.EXCLUSIONS
PAGA_PAIRS = t3s.PAGA_PAIRS
S = NULL_SEED
OUTCOMES = ["deg_stored_agree", "deg_stored_prolif_share", f"deg_rerun_agree_s{S}", f"deg_rerun_prolif_share_s{S}",
            f"ari1_s{S}", "cell_agree", "cell_prolif_share", "cluster_stored_agree", "cluster_stored_prolif_share",
            f"cluster_rerun_agree_s{S}", f"cluster_rerun_prolif_share_s{S}", f"ari3_s{S}",
            *[f"paga_stored_{a}_{b}" for a, b in PAGA_PAIRS], *[f"paga_rebuilt_{a}_{b}" for a, b in PAGA_PAIRS],
            *[f"paga_deg_stored_{a}_{b}" for a, b in PAGA_PAIRS], *[f"paga_deg_rebuilt_{a}_{b}" for a, b in PAGA_PAIRS]]
OUTCOME_DISPLAY = {
    "deg_stored_agree": "primary label agreement (DE rule on stored Leiden 1.0 clusters)",
    "deg_stored_prolif_share": "ProLIF share, primary label (stored clusters)",
    f"deg_rerun_agree_s{S}": "primary label agreement (Leiden 1.0 rerun)",
    f"deg_rerun_prolif_share_s{S}": "ProLIF share, primary label (Leiden 1.0 rerun)",
    f"ari1_s{S}": "ARI Leiden 1.0 rerun vs stored",
    "cell_agree": "per-cell label agreement (re-scored)",
    "cell_prolif_share": "ProLIF share, re-scored per-cell label",
    "cluster_stored_agree": "second label agreement (stored Leiden 3.0 clusters re-named)",
    "cluster_stored_prolif_share": "ProLIF share, second label (stored clusters)",
    f"cluster_rerun_agree_s{S}": "second label agreement (Leiden 3.0 rerun)",
    f"cluster_rerun_prolif_share_s{S}": "ProLIF share, second label (Leiden 3.0 rerun)",
    f"ari3_s{S}": "ARI Leiden 3.0 rerun vs stored",
    **{f"paga_stored_{a}_{b}": f"PAGA {a}-{b}, per-cell label, stored graph" for a, b in PAGA_PAIRS},
    **{f"paga_rebuilt_{a}_{b}": f"PAGA {a}-{b}, re-scored per-cell label, rebuilt graph" for a, b in PAGA_PAIRS},
    **{f"paga_deg_stored_{a}_{b}": f"PAGA {a}-{b}, primary label, stored graph" for a, b in PAGA_PAIRS},
    **{f"paga_deg_rebuilt_{a}_{b}": f"PAGA {a}-{b}, re-derived primary label (rerun), rebuilt graph"
       for a, b in PAGA_PAIRS},
}
# Outcome -> (settings attribute of the label column, labels that must be present at baseline).
NEEDS = {
    "deg_stored_prolif_share": ("PRIMARY", ("ProLIF",)), f"deg_rerun_prolif_share_s{S}": ("PRIMARY", ("ProLIF",)),
    "cell_prolif_share": ("CELL_COLUMN", ("ProLIF",)), "cluster_stored_prolif_share": ("SECOND", ("ProLIF",)),
    f"cluster_rerun_prolif_share_s{S}": ("SECOND", ("ProLIF",)),
    **{f"paga_stored_{a}_{b}": ("CELL_COLUMN", (a, b)) for a, b in PAGA_PAIRS},
    **{f"paga_rebuilt_{a}_{b}": ("CELL_COLUMN", (a, b)) for a, b in PAGA_PAIRS},
    **{f"paga_deg_stored_{a}_{b}": ("PRIMARY", (a, b)) for a, b in PAGA_PAIRS},
    **{f"paga_deg_rebuilt_{a}_{b}": ("PRIMARY", (a, b)) for a, b in PAGA_PAIRS},
}
COMPARATOR = {
    "deg_stored_prolif_share": "cluster_stored_prolif_share",
    f"deg_rerun_prolif_share_s{S}": f"cluster_rerun_prolif_share_s{S}",
    **{f"paga_deg_stored_{a}_{b}": f"paga_stored_{a}_{b}" for a, b in PAGA_PAIRS},
    **{f"paga_deg_rebuilt_{a}_{b}": f"paga_rebuilt_{a}_{b}" for a, b in PAGA_PAIRS},
}
FIG_PANELS = [("deg_stored_agree", "primary label agreement\n(DE rule, stored Leiden 1.0)"),
              (f"deg_rerun_agree_s{S}", f"primary label agreement\n(Leiden 1.0 rerun, seed {S})"),
              ("cluster_stored_agree", "second label agreement\n(stored Leiden 3.0)"),
              ("cell_agree", "per-cell label agreement\n(re-scored)"),
              ("deg_stored_prolif_share", "ProLIF share, primary\nlabel (stored clusters)"),
              ("cluster_stored_prolif_share", "ProLIF share, second\nlabel (stored clusters)"),
              ("cell_prolif_share", "ProLIF share, per-cell\nlabel (re-scored)"),
              ("paga_stored_DCT1_ProLIF", "PAGA DCT1-ProLIF, per-cell\nlabel, stored graph")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 9c: label sensitivity to excluding QC-flagged cells.")
    cc.add_common_args(parser)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    return parser.parse_args()


def exclusion_masks(qc: pd.DataFrame) -> dict[str, np.ndarray]:
    """Union of the step 9a flags over the primary and second label columns (generic flag names)."""
    mito = np.zeros(len(qc), bool)
    anyf = np.zeros(len(qc), bool)
    for role in lib.ROLES:
        mito |= qc[f"flag__{role}__mito_any"].astype(str).str.lower().eq("true").to_numpy()
        anyf |= qc[f"flag__{role}__any"].astype(str).str.lower().eq("true").to_numpy()
    return {"mito_flagged": mito, "any_flagged": anyf}


class Context:
    def __init__(self, s, adata, labels, lists, markers):
        self.s, self.labels, self.lists, self.markers = s, labels, lists, markers
        self.intensity = np.array(cc.dense(adata.layers["intensity"]), dtype=np.float32)
        self.ids = np.asarray(adata.obs_names.astype(str))
        self.var_names = adata.var_names.astype(str)
        self.graph = lib18.graph_adata(adata)          # stored dct_labels graph only
        self.cl3 = labels[s.LEIDEN3_CLUSTER].astype(str).to_numpy()
        self.cl1 = labels[s.PRIMARY_CLUSTER].astype(str).to_numpy()


def submitted_matrix(intensity: np.ndarray) -> np.ndarray:
    a = ad.AnnData(X=np.array(intensity, dtype=np.float32, copy=True))
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    return np.asarray(a.X, dtype=np.float32)


def deg_labels(cache, clusters: np.ndarray, var_names, markers, x_z, det) -> np.ndarray:
    """labels_core_stability.Labeller.deg on a bare matrix."""
    de, _, asg = lib18.annotate_clustering(cache, clusters, var_names, "t28_exclusion", markers, x_z, det)
    rf = rank_free(asg, de, lib18.specific_present(markers))
    return lib18.per_cell(clusters, rf.rename(columns={"rankfree_label": "deg_label"}), "deg_label").astype(str)


def evaluate(ctx: Context, removed: np.ndarray, seeds) -> tuple[dict, pd.DataFrame]:
    s = ctx.s
    keep = ~removed
    ids = ctx.ids[keep]
    sub = ctx.labels.loc[ids]
    X = submitted_matrix(ctx.intensity[keep])
    g = ad.AnnData(X=X.copy(), obs=pd.DataFrame(index=pd.Index(ids, name="cell_id")))
    sc.pp.neighbors(g, n_neighbors=lib18.N_NEIGHBORS, n_pcs=lib18.N_PCS)
    intensity = ctx.intensity[keep].astype(np.float64)
    detected = intensity > 0
    cache = t5.rank_cache(np.log1p(intensity), detected)
    x_z = pd.DataFrame(X.astype(np.float64), index=ids, columns=ctx.var_names)
    det = pd.DataFrame(detected, index=ids, columns=ctx.var_names)
    out = {"n_removed": int(removed.sum()), "n_remaining": int(keep.sum())}
    cells = pd.DataFrame({"cell_id": ids})

    scores = lib18.score_types(ad.AnnData(X=X.copy(), obs=pd.DataFrame(index=pd.Index(ids)),
                                          var=pd.DataFrame(index=ctx.var_names)), ctx.lists)
    cell, max_score = lib18.cell_labels(scores)
    named, _ = lib18.cluster_labels(cell, max_score, pd.Series(ctx.cl3[keep], index=cell.index))
    deg_stored = deg_labels(cache, ctx.cl1[keep], ctx.var_names, ctx.markers, x_z, det)
    out.update(cell_agree=t3s.frac(cell, sub[s.CELL_COLUMN]), cell_prolif_share=t3s.share(cell),
               cluster_stored_agree=t3s.frac(named, sub[s.SECOND]), cluster_stored_prolif_share=t3s.share(named),
               deg_stored_agree=t3s.frac(deg_stored, sub[s.PRIMARY]), deg_stored_prolif_share=t3s.share(deg_stored),
               submitted_deg_prolif_share=t3s.share(sub[s.PRIMARY]),
               submitted_cluster_prolif_share=t3s.share(sub[s.SECOND]),
               submitted_cell_prolif_share=t3s.share(sub[s.CELL_COLUMN]))
    cells["cell_label_rederived"] = cell.to_numpy()
    cells["second_label_renamed_stored_clusters"] = named.to_numpy()
    cells["primary_label_stored_clusters"] = deg_stored
    deg_rerun_last = None
    for seed in seeds:
        c3 = t3s.leiden(g, lib18.SUBMITTED_RESOLUTION, seed, f"leiden3_{seed}").to_numpy()
        named_r, _ = lib18.cluster_labels(cell, max_score, pd.Series(c3, index=cell.index))
        c1 = t3s.leiden(g, 1.0, seed, f"leiden1_{seed}").to_numpy()
        deg_r = deg_labels(cache, c1, ctx.var_names, ctx.markers, x_z, det)
        out.update({f"cluster_rerun_agree_s{seed}": t3s.frac(named_r, sub[s.SECOND]),
                    f"cluster_rerun_prolif_share_s{seed}": t3s.share(named_r),
                    f"ari3_s{seed}": float(adjusted_rand_score(ctx.cl3[keep], c3)), f"n_clusters3_s{seed}": len(set(c3)),
                    f"deg_rerun_agree_s{seed}": t3s.frac(deg_r, sub[s.PRIMARY]),
                    f"deg_rerun_prolif_share_s{seed}": t3s.share(deg_r),
                    f"ari1_s{seed}": float(adjusted_rand_score(ctx.cl1[keep], c1)), f"n_clusters1_s{seed}": len(set(c1))})
        cells[f"second_label_rerun_seed{seed}"] = named_r.to_numpy()
        cells[f"primary_label_rerun_seed{seed}"] = deg_r
        if seed == NULL_SEED:
            deg_rerun_last = deg_r
    stored_g = ctx.graph[keep].copy()
    for prefix, graph, values in (("paga_stored", stored_g, sub[s.CELL_COLUMN]), ("paga_rebuilt", g, cell),
                                  ("paga_deg_stored", stored_g, sub[s.PRIMARY]),
                                  ("paga_deg_rebuilt", g, deg_rerun_last if deg_rerun_last is not None else deg_stored)):
        for (a, b), v in t3s.paga(graph, values).items():
            out[f"{prefix}_{a}_{b}"] = v
    return out, cells


def composition(s, labels: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for col in (*s.LABEL_COLUMNS, "plate"):
        values = labels[col].astype(str).to_numpy()
        for exclusion, mask in masks.items():
            keep = ~mask
            for label in t3.label_order(values, "dct"):
                n_before = int((values == label).sum())
                n_after = int((values[keep] == label).sum())
                rows.append({"dataset": "dct", "exclusion_set": exclusion, "label_column": col, "label": label,
                             "n_before": n_before, "n_after": n_after, "n_excluded": n_before - n_after,
                             "percent_excluded_of_label": 100 * (n_before - n_after) / n_before if n_before else np.nan,
                             "share_before": n_before / len(values), "share_after": n_after / keep.sum()})
    return pd.DataFrame(rows)


def applicability(s, labels: pd.DataFrame, outcome: str) -> tuple[bool, str]:
    if outcome not in NEEDS:
        return True, ""
    attr, needed = NEEDS[outcome]
    col = getattr(s, attr)
    absent = [lab for lab in needed if not lib.present(labels, col, lab)]
    if absent:
        return False, f"{lib.NOT_APPLICABLE} (label absent at baseline: {', '.join(absent)} under {col})"
    return True, ""


def null_summary(s, labels: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """ctrl_reviewer_core_exclusion_sensitivity.null_summary plus applicability and the comparator outcome."""
    rows = []
    base = runs[runs["run_type"] == "baseline"].iloc[0]
    for exclusion in EXCLUSIONS:
        obs = runs[(runs["run_type"] == "observed") & (runs["exclusion_set"] == exclusion)].iloc[0]
        null = runs[(runs["run_type"] == "random") & (runs["exclusion_set"] == exclusion)]
        for outcome in OUTCOMES:
            ok, reason = applicability(s, labels, outcome)
            x = null[outcome].to_numpy(float)
            x = x[np.isfinite(x)]
            o = float(obs[outcome])
            n = len(x)
            row = {"dataset": "dct", "exclusion_set": exclusion, "outcome": outcome,
                   "display": OUTCOME_DISPLAY[outcome], "applicable": ok, "reason": reason,
                   "n_removed": int(obs["n_removed"]), "baseline": float(base[outcome]), "observed": o}
            if n == 0:   # e.g. a PAGA pair whose label is absent in every random run
                row.update(random_n=0, note="undefined in every random run (label absent)")
            else:
                p_le = (1 + (x <= o).sum()) / (n + 1)
                p_ge = (1 + (x >= o).sum()) / (n + 1)
                row.update(observed_minus_baseline=o - float(base[outcome]), random_n=n,
                           random_median=float(np.median(x)), random_q025=float(np.quantile(x, 0.025)),
                           random_q975=float(np.quantile(x, 0.975)), random_min=float(x.min()),
                           random_max=float(x.max()), observed_percentile_in_random=100 * float((x < o).mean()),
                           p_empirical_lower=p_le, p_empirical_upper=p_ge,
                           p_empirical_two_sided=min(1.0, 2 * min(p_le, p_ge)),
                           inside_central_95=bool(np.quantile(x, 0.025) <= o <= np.quantile(x, 0.975)))
            rows.append(row)
    out = pd.DataFrame(rows)
    # Comparator beside each primary-label outcome whose label is absent (second / per-cell label result).
    idx = out.set_index(["exclusion_set", "outcome"])
    for col in ("comparator_outcome", "comparator_applicable", "comparator_baseline", "comparator_observed",
                "comparator_random_median", "comparator_inside_central_95"):
        out[col] = None
    for i, r in out.iterrows():
        comp = COMPARATOR.get(r["outcome"])
        if comp is None or r["applicable"]:
            continue
        c = idx.loc[(r["exclusion_set"], comp)]
        out.loc[i, ["comparator_outcome", "comparator_applicable", "comparator_baseline", "comparator_observed",
                    "comparator_random_median", "comparator_inside_central_95"]] = [
            comp, c["applicable"], c["baseline"], c["observed"], c.get("random_median"), c.get("inside_central_95")]
    return out


def seed_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in runs[runs["run_type"].isin(["baseline", "observed"])].iterrows():
        row = {"run_type": r["run_type"], "exclusion_set": r["exclusion_set"], "n_removed": int(r["n_removed"]),
               "seeds": ";".join(map(str, OBS_SEEDS))}
        for key in ("deg_rerun_agree", "cluster_rerun_agree", "ari1", "ari3"):
            vals = np.array([r[f"{key}_s{s}"] for s in OBS_SEEDS], float)
            row.update({f"{key}_min": vals.min(), f"{key}_median": np.median(vals), f"{key}_max": vals.max()})
        rows.append(row)
    return pd.DataFrame(rows)


def reproduction_checks(cond: str, s, adata, labels: pd.DataFrame, base: dict) -> list[dict]:
    """The all-cells run with Leiden seed 42 must give the dct_labels labels; PAGA on the stored graph must equal the
    PAGA stored in the dct_labels object (groups = predicted_cell_type = the per-cell label, sorted categories)."""
    rows = [{"check": f"all cells: re-scored per-cell label equals {s.CELL_COLUMN}", "value": f"{base['cell_agree']:.4f}",
             "pass": base["cell_agree"] == 1.0},
            {"check": f"all cells: stored Leiden 3.0 clusters re-named equal {s.SECOND}",
             "value": f"{base['cluster_stored_agree']:.4f}", "pass": base["cluster_stored_agree"] == 1.0},
            {"check": f"all cells: DE rule on stored {s.PRIMARY_CLUSTER} equals {s.PRIMARY}",
             "value": f"{base['deg_stored_agree']:.4f}", "pass": base["deg_stored_agree"] == 1.0},
            {"check": f"all cells, seed 42: Leiden 3.0 rerun equals {s.LEIDEN3_CLUSTER} and its names {s.SECOND}",
             "value": f"ARI {base['ari3_s42']:.4f}; agreement {base['cluster_rerun_agree_s42']:.4f}",
             "pass": base["ari3_s42"] == 1.0 and base["cluster_rerun_agree_s42"] == 1.0},
            {"check": f"all cells, seed 42: Leiden 1.0 rerun equals {s.PRIMARY_CLUSTER} and its DE labels {s.PRIMARY}",
             "value": f"ARI {base['ari1_s42']:.4f}; agreement {base['deg_rerun_agree_s42']:.4f}",
             "pass": base["ari1_s42"] == 1.0 and base["deg_rerun_agree_s42"] == 1.0}]
    paga = adata.uns.get("paga", {})
    cats = sorted(set(labels[s.CELL_COLUMN].astype(str)))
    groups_ok = paga.get("groups") == "predicted_cell_type" and np.array_equal(
        adata.obs["predicted_cell_type"].astype(str).to_numpy(), labels[s.CELL_COLUMN].astype(str).to_numpy())
    stored_conn = cc.dense(paga["connectivities"]) if "connectivities" in paga else np.full((len(cats),) * 2, np.nan)
    diffs = [abs(base[f"paga_stored_{a}_{b}"] - stored_conn[cats.index(a), cats.index(b)]) for a, b in PAGA_PAIRS]
    rows.append({"check": f"PAGA ({s.CELL_COLUMN}) on the stored graph equals the PAGA stored in the dct_labels object",
                 "value": "; ".join(f"{a}-{b} {base[f'paga_stored_{a}_{b}']:.4f}" for a, b in PAGA_PAIRS),
                 "pass": bool(groups_ok and max(diffs) < 1e-6)})
    return [{"condition": cond, **r} for r in rows]


def one_condition(run: cc.Run, cond: str, qc_all: pd.DataFrame, n_draws: int, t0: float) -> dict:
    s = lib.SETTINGS[cond]
    rng = np.random.default_rng(t3.SEED)
    adata, labels = lib.load(run, cond)
    qc = qc_all[qc_all["condition"] == cond].set_index("cell_id")
    if list(qc.index) != list(labels.index):
        raise ValueError(f"{cond}: step 9a per-cell order differs from the dct_labels canonical labels.")
    masks = exclusion_masks(qc)
    lists = lib18.present_lists(lib18.read_workbook(run), adata.var_names)
    markers = lib18.dct_markers(run, adata.var_names)
    ctx = Context(s, adata, labels, lists, markers)
    strata = labels[s.PRIMARY].astype(str).to_numpy()

    base, cells = evaluate(ctx, np.zeros(len(strata), bool), OBS_SEEDS)
    checks = reproduction_checks(cond, s, adata, labels, base)
    print(f"{cond}: baseline done ({time.time() - t0:.0f} s)", flush=True)
    if not all(r["pass"] for r in checks):
        return {"checks": checks, "failed": True}
    rows = [{"dataset": "dct", "run_type": "baseline", "exclusion_set": "none", "draw": "", **base}]
    cells.insert(0, "run", "baseline")
    cells_out, sets = [cells], []
    for exclusion, mask in masks.items():
        sets.append({"dataset": "dct", "exclusion_set": exclusion, "n_excluded": int(mask.sum()),
                     "n_remaining": int((~mask).sum()),
                     "excluded_by_primary_label": "; ".join(f"{k}: {v}" for k, v in
                                                           pd.Series(strata[mask]).value_counts().items())})
        res, cells = evaluate(ctx, mask, OBS_SEEDS)
        rows.append({"dataset": "dct", "run_type": "observed", "exclusion_set": exclusion, "draw": "", **res})
        cells.insert(0, "run", f"without_{exclusion}")
        cells_out.append(cells)
        for draw in range(n_draws):
            removed = t3s.stratified_draw(rng, strata, mask)
            res, _ = evaluate(ctx, removed, (NULL_SEED,))
            rows.append({"dataset": "dct", "run_type": "random", "exclusion_set": exclusion, "draw": draw,
                         "n_flagged_among_removed": int((removed & mask).sum()), **res})
            if (draw + 1) % 20 == 0:
                print(f"  {cond} {exclusion}: {draw + 1} draws ({time.time() - t0:.0f} s)", flush=True)
        run.note(f"{cond} {exclusion}: {int(mask.sum())} flagged cells; {n_draws} random removals done "
                 f"({time.time() - t0:.0f} s since start).")
    runs = pd.DataFrame(rows)
    return {"checks": checks, "failed": False, "runs": runs, "summary": null_summary(s, labels, runs),
            "sets": pd.DataFrame(sets), "composition": composition(s, labels, masks), "seeds": seed_summary(runs),
            "cells": pd.concat(cells_out, ignore_index=True)}


# ---------------------------------------------------------------------------
# Figure (dotplot format; rows = condition)
# ---------------------------------------------------------------------------
def figure(res: dict, figs: lib.Figures) -> None:
    colors = {"mito_flagged": t3.SERIES[0], "any_flagged": t3.SERIES[1]}
    fig = lib.new_figure(lib.FULL_MM, 196)
    gs = fig.add_gridspec(4, 4, left=0.115, right=0.985, top=0.885, bottom=0.04, hspace=0.95, wspace=0.18)
    rng = np.random.default_rng(lib.SEED)
    for ci, cond in enumerate(lib.CONDITIONS):
        summ, runs = res[cond]["summary"], res[cond]["runs"]
        for k, (outcome, title) in enumerate(FIG_PANELS):
            r, c = divmod(k, 4)
            ax = fig.add_subplot(gs[2 * ci + r, c])
            ax.set_title(title, fontsize=7)
            rows = summ[summ["outcome"] == outcome]
            if not bool(rows["applicable"].iloc[0]):
                lib.na_axis(ax, f"{lib.NOT_APPLICABLE}:\nlabel absent at\nbaseline (second\nlabel beside)")
                continue
            for e, exclusion in enumerate(EXCLUSIONS):
                null = runs[(runs["run_type"] == "random") & (runs["exclusion_set"] == exclusion)][outcome].to_numpy(float)
                null = null[np.isfinite(null)]
                ax.scatter(null, e + rng.uniform(-0.18, 0.18, len(null)), s=2.5, color=t2.INK["muted"], alpha=0.6,
                           linewidths=0, rasterized=True)
                if null.size:
                    q = np.quantile(null, [0.025, 0.5, 0.975])
                    ax.plot([q[0], q[2]], [e, e], color=t2.INK["secondary"], linewidth=0.8)
                    ax.scatter([q[1]], [e], marker="|", s=40, color=t2.INK["primary"], linewidths=1.0)
                row = rows[rows["exclusion_set"] == exclusion].iloc[0]
                ax.scatter([row["observed"]], [e], s=20, color=colors[exclusion], zorder=4, linewidths=0.4,
                           edgecolors="white")
            ax.axvline(float(rows["baseline"].iloc[0]), color=t2.INK["primary"], linewidth=0.6, linestyle=(0, (3, 2)))
            ax.set_ylim(1.6, -0.6)
            if c == 0:
                ax.set_yticks([0, 1], ["mito-flagged", "any-flagged"])
            else:
                ax.set_yticks([0, 1], ["", ""])
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
        fig.text(0.008, 0.885 - (0.885 - 0.04) * (ci + 0.5) / 2, cond, rotation=90, ha="left", va="center",
                 fontsize=7, fontweight="bold")
    fig.text(0.005, 0.995, "Excluding QC-flagged cells vs removing the same number of random cells (stratified by the "
             "primary label), per condition\ncoloured = flagged cells removed (blue: mito-flagged; orange: any-flagged); "
             "grey = random removals (bar: median, central 95%)\ndashed = all cells; reruns use Leiden seed "
             f"{S}; agreement = share of remaining cells keeping their dct_labels label", ha="left", va="top", fontsize=7)
    figs.save(fig, "exclusion_vs_random", "Flagged-cell exclusion against random removal, Ctrl and ZeroK")


def main() -> None:
    args = parse_args()
    sys.unraisablehook = t3s._unraisable
    run = cc.Run(lib.TASK, "exclusion_sensitivity", args)
    lib.ported_from(run, "ctrl_reviewer_core_exclusion_sensitivity.py", "qc_mito_sensitivity.py")
    qc_all = pd.read_csv(run.use_input(lib.table_path("mito_qc_per_cell"), "dct_reviewer step 9a per-cell mito QC"),
                         dtype={"cell_id": str}, keep_default_na=False, na_values=[""])
    t0 = time.time()
    res = {}
    for cond in lib.CONDITIONS:
        res[cond] = one_condition(run, cond, qc_all, args.n_draws, t0)
        if res[cond]["failed"]:
            break
    checks = lib.check_rows_frame([r for c in res for r in res[c]["checks"]])
    lib.stop_if_failed(run, checks, "exclusion_checks")

    def stacked(key: str) -> pd.DataFrame:
        return lib.stack({c: res[c][key] for c in lib.CONDITIONS})

    run.table(stacked("sets"), "exclusion_sets")
    run.table(stacked("composition"), "exclusion_composition")
    run.table(stacked("runs"), "exclusion_runs")
    run.table(stacked("summary"), "exclusion_vs_random")
    run.table(stacked("seeds"), "exclusion_leiden_seeds")
    run.table(stacked("cells"), "exclusion_per_cell_labels")
    run.note("Outcome names keep the ctrl_reviewer_core stems: deg_ = primary label (DE rule; cs_ctrl_deg_label / "
             "cs_zerok_pipeline_label), cluster_ = second label (cs_*_cluster_label), cell_ = per-cell label "
             "(cs_*_cell_label). Outcomes with applicable = False need a label absent at baseline under their column; "
             "the comparator columns give the second-label (ProLIF shares) or per-cell-label (PAGA) result.")
    run.note(f"Ignored igraph RNG messages ('high is out of bounds for int32'): {t3s._IGNORED['igraph_int32_randint']}.")
    figs = lib.Figures(run, "exclusion_sensitivity")
    figure(res, figs)
    figs.finish()
    run.finish({"n_draws": args.n_draws, "obs_seeds": list(OBS_SEEDS), "null_leiden_seed": NULL_SEED,
                "resolutions": {"second_label": lib18.SUBMITTED_RESOLUTION, "primary_label": 1.0},
                "strata": {c: lib.SETTINGS[c].PRIMARY for c in lib.CONDITIONS},
                "rng": "numpy default_rng(42), one stream per condition over all draws",
                "ignored_igraph_rng_messages": t3s._IGNORED["igraph_int32_randint"],
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_exclusion_sensitivity.csv.")


if __name__ == "__main__":
    main()
