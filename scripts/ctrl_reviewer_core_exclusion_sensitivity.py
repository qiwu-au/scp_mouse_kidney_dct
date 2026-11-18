#!/usr/bin/env python
"""ctrl_reviewer_core step 9c: qc_mito addendum E on the control DCT data - labels and PAGA after excluding QC-flagged cells.

Port of
qc_mito_sensitivity.py (DCT only; included in ctrl_reviewer_core with the authors, 28 Oct 2026,
because the point 7 / 15 drafts quote its numbers). Generic pieces are imported
from qc_mito_sensitivity (stratified_draw, leiden, paga, frac, share, the
igraph message counter); the labelling follows labels_core so that the all-cells run
reproduces the labels_core labels exactly:
  - matrix: float32 intensities of the remaining cells -> sc.pp.log1p ->
    sc.pp.scale(max_value=10) (the submitted pipeline, as labels_core), fixed protein set;
  - graph: sc.pp.neighbors(n_neighbors=10, n_pcs=20) without a prior PCA (Scanpy
    fallback, as labels_core; qc_mito used an explicit arpack PCA);
  - ctrl_cell_label: score_genes on the re-scaled matrix -> per-cell argmax;
  - ctrl_cluster_label: cluster mode with the original 'Unknown' rule, on the
    stored ctrl_leiden3_cluster (restricted to the remaining cells) and on a
    Leiden 3.0 rerun;
  - ctrl_deg_label: cluster_de DE rule + kidney_annotation rank-free rescue (minimum cluster size 5)
    on the stored ctrl_deg_cluster (restricted) and on a Leiden 1.0 rerun.
Exclusion sets (union over ctrl_deg_label and ctrl_cluster_label, from step 9):
mito_flagged (high or low on any mitochondrial metric) and any_flagged (any qc_mito
metric). Control: 100 random removals of the same number of cells, stratified
by ctrl_deg_label (numpy default_rng(42), one stream). Baseline and observed runs
use Leiden seeds 0, 1, 2, 3, 42; random removals use seed 0, and the comparison
with the random distribution uses seed 0 throughout (as qc_mito).

Inputs (read-only): labels_core labelled control object and canonical labels;
ctrl_reviewer_core_dct_qc_per_cell.csv (step 9); data/DCT/DCT_markers_V1.xlsx.
Outputs (results/tables/ctrl_reviewer_core/, results/figures/ctrl_reviewer_core/):
  ctrl_reviewer_core_sensitivity_{exclusion_sets,composition,runs,vs_random,leiden_seeds,per_cell_labels,checks}.csv
  ctrl_reviewer_core_sensitivity_vs_random.pdf/.png

Command (project root):  python scripts/ctrl_reviewer_core_exclusion_sensitivity.py
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
import textwrap
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import qc_mito_lib as t3
import qc_mito_sensitivity as t3s
import cluster_de_lib as t5
import labels_core_lib as lib18
import ctrl_reviewer_core_lib as lib
from kidney_annotation_rankfree_sensitivity import rank_free

ensure_scanpy_compat()
t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "ctrl_reviewer_core_exclusion_sensitivity.py"
OBS_SEEDS = t3s.OBS_SEEDS            # 0, 1, 2, 3, 42
NULL_SEED = t3s.NULL_LEIDEN_SEED     # 0
N_DRAWS = t3s.N_DRAWS                # 100
EXCLUSIONS = t3s.EXCLUSIONS
PAGA_PAIRS = t3s.PAGA_PAIRS
STRATA = lib.PRIMARY
S = NULL_SEED
OUTCOMES = ["deg_stored_agree", "deg_stored_prolif_share", f"deg_rerun_agree_s{S}", f"deg_rerun_prolif_share_s{S}",
            f"ari1_s{S}", "cell_agree", "cell_prolif_share", "cluster_stored_agree", "cluster_stored_prolif_share",
            f"cluster_rerun_agree_s{S}", f"cluster_rerun_prolif_share_s{S}", f"ari3_s{S}",
            *[f"paga_stored_{a}_{b}" for a, b in PAGA_PAIRS], *[f"paga_rebuilt_{a}_{b}" for a, b in PAGA_PAIRS],
            *[f"paga_deg_stored_{a}_{b}" for a, b in PAGA_PAIRS], *[f"paga_deg_rebuilt_{a}_{b}" for a, b in PAGA_PAIRS]]
OUTCOME_DISPLAY = {
    "deg_stored_agree": "ctrl_deg_label agreement (DE rule on stored Leiden 1.0 clusters)",
    "deg_stored_prolif_share": "ProLIF share, ctrl_deg_label (stored clusters)",
    f"deg_rerun_agree_s{S}": "ctrl_deg_label agreement (Leiden 1.0 rerun)",
    f"deg_rerun_prolif_share_s{S}": "ProLIF share, ctrl_deg_label (Leiden 1.0 rerun)",
    f"ari1_s{S}": "ARI Leiden 1.0 rerun vs stored",
    "cell_agree": "ctrl_cell_label agreement (re-scored)",
    "cell_prolif_share": "ProLIF share, re-scored ctrl_cell_label",
    "cluster_stored_agree": "ctrl_cluster_label agreement (stored Leiden 3.0 clusters re-named)",
    "cluster_stored_prolif_share": "ProLIF share, ctrl_cluster_label (stored clusters)",
    f"cluster_rerun_agree_s{S}": "ctrl_cluster_label agreement (Leiden 3.0 rerun)",
    f"cluster_rerun_prolif_share_s{S}": "ProLIF share, ctrl_cluster_label (Leiden 3.0 rerun)",
    f"ari3_s{S}": "ARI Leiden 3.0 rerun vs stored",
    **{f"paga_stored_{a}_{b}": f"PAGA {a}-{b}, ctrl_cell_label, stored graph" for a, b in PAGA_PAIRS},
    **{f"paga_rebuilt_{a}_{b}": f"PAGA {a}-{b}, re-scored ctrl_cell_label, rebuilt graph" for a, b in PAGA_PAIRS},
    **{f"paga_deg_stored_{a}_{b}": f"PAGA {a}-{b}, ctrl_deg_label, stored graph" for a, b in PAGA_PAIRS},
    **{f"paga_deg_rebuilt_{a}_{b}": f"PAGA {a}-{b}, re-derived ctrl_deg_label (rerun), rebuilt graph"
       for a, b in PAGA_PAIRS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_reviewer_core step 9c: label sensitivity to excluding QC-flagged cells.")
    cc.add_common_args(parser)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    return parser.parse_args()


def exclusion_masks(qc: pd.DataFrame) -> dict[str, np.ndarray]:
    """Union of the step-9 flags over both label columns (qc_mito_sensitivity.exclusion_masks with ctrl_reviewer_core columns)."""
    mito = np.zeros(len(qc), bool)
    anyf = np.zeros(len(qc), bool)
    for col in lib.LABEL_COLUMNS:
        s = lib.SHORT_COLUMN[col]
        mito |= qc[f"flag__{s}__mito_any"].astype(str).str.lower().eq("true").to_numpy()
        anyf |= qc[f"flag__{s}__any"].astype(str).str.lower().eq("true").to_numpy()
    return {"mito_flagged": mito, "any_flagged": anyf}


class Context:
    def __init__(self, adata, labels, lists, markers):
        self.labels, self.lists, self.markers = labels, lists, markers
        self.intensity = np.array(cc.dense(adata.layers["intensity"]), dtype=np.float32)
        self.ids = np.asarray(adata.obs_names.astype(str))
        self.var_names = adata.var_names.astype(str)
        self.graph = lib18.graph_adata(adata)          # stored labels_core graph only
        self.cl3 = labels["ctrl_leiden3_cluster"].astype(str).to_numpy()
        self.cl1 = labels["ctrl_deg_cluster"].astype(str).to_numpy()


def submitted_matrix(intensity: np.ndarray) -> np.ndarray:
    a = ad.AnnData(X=np.array(intensity, dtype=np.float32, copy=True))
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    return np.asarray(a.X, dtype=np.float32)


def deg_labels(cache, clusters: np.ndarray, var_names, markers, x_z, det) -> np.ndarray:
    """labels_core_stability.Labeller.deg on a bare matrix."""
    de, _, asg = lib18.annotate_clustering(cache, clusters, var_names, "t19_exclusion", markers, x_z, det)
    rf = rank_free(asg, de, lib18.specific_present(markers))
    return lib18.per_cell(clusters, rf.rename(columns={"rankfree_label": "deg_label"}), "deg_label").astype(str)


def evaluate(ctx: Context, removed: np.ndarray, seeds) -> tuple[dict, pd.DataFrame]:
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
    out.update(cell_agree=t3s.frac(cell, sub["ctrl_cell_label"]), cell_prolif_share=t3s.share(cell),
               cluster_stored_agree=t3s.frac(named, sub["ctrl_cluster_label"]), cluster_stored_prolif_share=t3s.share(named),
               deg_stored_agree=t3s.frac(deg_stored, sub["ctrl_deg_label"]), deg_stored_prolif_share=t3s.share(deg_stored),
               submitted_deg_prolif_share=t3s.share(sub["ctrl_deg_label"]),
               submitted_cluster_prolif_share=t3s.share(sub["ctrl_cluster_label"]),
               submitted_cell_prolif_share=t3s.share(sub["ctrl_cell_label"]))
    cells["ctrl_cell_label_rederived"] = cell.to_numpy()
    cells["ctrl_cluster_label_renamed_stored_clusters"] = named.to_numpy()
    cells["ctrl_deg_label_stored_clusters"] = deg_stored
    deg_rerun_last = None
    for seed in seeds:
        c3 = t3s.leiden(g, lib18.SUBMITTED_RESOLUTION, seed, f"leiden3_{seed}").to_numpy()
        named_r, _ = lib18.cluster_labels(cell, max_score, pd.Series(c3, index=cell.index))
        c1 = t3s.leiden(g, 1.0, seed, f"leiden1_{seed}").to_numpy()
        deg_r = deg_labels(cache, c1, ctx.var_names, ctx.markers, x_z, det)
        out.update({f"cluster_rerun_agree_s{seed}": t3s.frac(named_r, sub["ctrl_cluster_label"]),
                    f"cluster_rerun_prolif_share_s{seed}": t3s.share(named_r),
                    f"ari3_s{seed}": float(adjusted_rand_score(ctx.cl3[keep], c3)), f"n_clusters3_s{seed}": len(set(c3)),
                    f"deg_rerun_agree_s{seed}": t3s.frac(deg_r, sub["ctrl_deg_label"]),
                    f"deg_rerun_prolif_share_s{seed}": t3s.share(deg_r),
                    f"ari1_s{seed}": float(adjusted_rand_score(ctx.cl1[keep], c1)), f"n_clusters1_s{seed}": len(set(c1))})
        cells[f"ctrl_cluster_label_rerun_seed{seed}"] = named_r.to_numpy()
        cells[f"ctrl_deg_label_rerun_seed{seed}"] = deg_r
        if seed == NULL_SEED:
            deg_rerun_last = deg_r
    stored_g = ctx.graph[keep].copy()
    for prefix, graph, values in (("paga_stored", stored_g, sub["ctrl_cell_label"]), ("paga_rebuilt", g, cell),
                                  ("paga_deg_stored", stored_g, sub["ctrl_deg_label"]),
                                  ("paga_deg_rebuilt", g, deg_rerun_last if deg_rerun_last is not None else deg_stored)):
        for (a, b), v in t3s.paga(graph, values).items():
            out[f"{prefix}_{a}_{b}"] = v
    return out, cells


def composition(labels: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for col in (*lib.LABEL_COLUMNS, "plate"):
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


def null_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """qc_mito_sensitivity.null_summary for the ctrl_reviewer_core outcomes (DCT only)."""
    rows = []
    base = runs[runs["run_type"] == "baseline"].iloc[0]
    for exclusion in EXCLUSIONS:
        obs = runs[(runs["run_type"] == "observed") & (runs["exclusion_set"] == exclusion)].iloc[0]
        null = runs[(runs["run_type"] == "random") & (runs["exclusion_set"] == exclusion)]
        for outcome in OUTCOMES:
            x = null[outcome].to_numpy(float)
            x = x[np.isfinite(x)]
            o = float(obs[outcome])
            n = len(x)
            if n == 0:   # e.g. a PAGA pair whose label is absent in every random run
                rows.append({"dataset": "dct", "exclusion_set": exclusion, "outcome": outcome,
                             "display": OUTCOME_DISPLAY[outcome], "n_removed": int(obs["n_removed"]),
                             "baseline": float(base[outcome]), "observed": o, "random_n": 0,
                             "note": "undefined in every random run (label absent)"})
                continue
            p_le = (1 + (x <= o).sum()) / (n + 1)
            p_ge = (1 + (x >= o).sum()) / (n + 1)
            rows.append({"dataset": "dct", "exclusion_set": exclusion, "outcome": outcome,
                         "display": OUTCOME_DISPLAY[outcome], "n_removed": int(obs["n_removed"]),
                         "baseline": float(base[outcome]), "observed": o, "observed_minus_baseline": o - float(base[outcome]),
                         "random_n": n, "random_median": float(np.median(x)), "random_q025": float(np.quantile(x, 0.025)),
                         "random_q975": float(np.quantile(x, 0.975)), "random_min": float(x.min()),
                         "random_max": float(x.max()), "observed_percentile_in_random": 100 * float((x < o).mean()),
                         "p_empirical_lower": p_le, "p_empirical_upper": p_ge,
                         "p_empirical_two_sided": min(1.0, 2 * min(p_le, p_ge)),
                         "inside_central_95": bool(np.quantile(x, 0.025) <= o <= np.quantile(x, 0.975))})
    return pd.DataFrame(rows)


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


def figure(run, summ: pd.DataFrame, runs: pd.DataFrame) -> None:
    panels = ["deg_stored_agree", f"deg_rerun_agree_s{S}", "deg_stored_prolif_share", "paga_deg_stored_DCT1_ProLIF",
              "cell_agree", "cluster_stored_agree", "cell_prolif_share", "paga_stored_DCT1_ProLIF"]
    colors = {"mito_flagged": t3.SERIES[0], "any_flagged": t3.SERIES[1]}
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.3))
    rng = np.random.default_rng(t3.SEED)
    for ax, outcome in zip(axes.flat, panels):
        for k, exclusion in enumerate(EXCLUSIONS):
            null = runs[(runs["run_type"] == "random") & (runs["exclusion_set"] == exclusion)][outcome].to_numpy(float)
            null = null[np.isfinite(null)]
            ax.scatter(null, k + rng.uniform(-0.18, 0.18, len(null)), s=3, color=t3.INK["background_cells"],
                       linewidths=0, rasterized=True)
            q = np.quantile(null, [0.025, 0.5, 0.975])
            ax.plot([q[0], q[2]], [k, k], color=t3.INK["muted"], linewidth=0.8)
            ax.scatter([q[1]], [k], marker="|", s=40, color=t3.INK["secondary"], linewidths=1.0)
            r = summ[(summ["exclusion_set"] == exclusion) & (summ["outcome"] == outcome)].iloc[0]
            ax.scatter([r["observed"]], [k], s=22, color=colors[exclusion], zorder=4, linewidths=0.4, edgecolors="white")
        ax.axvline(summ[summ["outcome"] == outcome]["baseline"].iloc[0], color=t3.INK["primary"], linewidth=0.6,
                   linestyle=(0, (3, 2)))
        ax.set_yticks([0, 1], ["mito-flagged", "any-flagged"], fontsize=5.6)
        ax.set_ylim(1.6, -0.6)
        ax.set_title(textwrap.fill(OUTCOME_DISPLAY[outcome], 34), fontsize=5.6)
        ax.tick_params(axis="x", labelsize=5.4)
        ax.xaxis.grid(True, color=t3.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Control DCT: excluding QC-flagged cells vs removing the same number of random cells "
                 "(stratified by ctrl_deg_label)", x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    t3.footer(fig, SCRIPT, f"coloured = flagged cells removed; grey = random removals (bar: median, 95% range); "
                           f"dashed = all cells (rerun, Leiden seed {S})")
    run.figure(fig, "sensitivity_vs_random", "Flagged-cell exclusion against random removal (control DCT)")


def main() -> None:
    args = parse_args()
    sys.unraisablehook = t3s._unraisable
    run = cc.Run(lib.TASK, "exclusion_sensitivity", args)
    lib.ported_from(run, "qc_mito_sensitivity.py")
    rng = np.random.default_rng(t3.SEED)
    adata, labels = lib.load_ctrl(run)
    qc = pd.read_csv(run.use_input(lib.table_path("dct_qc_per_cell"), "ctrl_reviewer_core per-cell QC (step 9)"),
                     dtype={"cell_id": str}).set_index("cell_id")
    if list(qc.index) != list(labels.index):
        raise ValueError("Per-cell QC order differs from the canonical labels.")
    masks = exclusion_masks(qc)
    lists = lib18.present_lists(lib18.read_workbook(run), adata.var_names)
    markers = lib18.dct_markers(run, adata.var_names)
    ctx = Context(adata, labels, lists, markers)
    strata = labels[STRATA].astype(str).to_numpy()

    # Reproduction: the all-cells run with Leiden seed 42 must give the labels_core labels; PAGA on the stored graph
    # must equal the PAGA stored in the labels_core object (ctrl_cell_label = predicted_cell_type).
    t0 = time.time()
    base, cells = evaluate(ctx, np.zeros(len(strata), bool), OBS_SEEDS)
    checks = [{"check": "all cells: re-scored ctrl_cell_label equals labels_core", "observed": f"{base['cell_agree']:.4f}",
               "pass": base["cell_agree"] == 1.0},
              {"check": "all cells: stored Leiden 3.0 clusters re-named equal ctrl_cluster_label",
               "observed": f"{base['cluster_stored_agree']:.4f}", "pass": base["cluster_stored_agree"] == 1.0},
              {"check": "all cells: DE rule on stored ctrl_deg_cluster equals ctrl_deg_label",
               "observed": f"{base['deg_stored_agree']:.4f}", "pass": base["deg_stored_agree"] == 1.0},
              {"check": "all cells, seed 42: Leiden 3.0 rerun equals ctrl_leiden3_cluster and its names ctrl_cluster_label",
               "observed": f"ARI {base['ari3_s42']:.4f}; agreement {base['cluster_rerun_agree_s42']:.4f}",
               "pass": base["ari3_s42"] == 1.0 and base["cluster_rerun_agree_s42"] == 1.0},
              {"check": "all cells, seed 42: Leiden 1.0 rerun equals ctrl_deg_cluster and its DE labels ctrl_deg_label",
               "observed": f"ARI {base['ari1_s42']:.4f}; agreement {base['deg_rerun_agree_s42']:.4f}",
               "pass": base["ari1_s42"] == 1.0 and base["deg_rerun_agree_s42"] == 1.0}]
    cats = list(adata.obs["predicted_cell_type"].astype("category").cat.categories)
    stored_conn = cc.dense(adata.uns["paga"]["connectivities"])
    diffs = [abs(base[f"paga_stored_{a}_{b}"] - stored_conn[cats.index(a), cats.index(b)]) for a, b in PAGA_PAIRS]
    checks.append({"check": "PAGA (ctrl_cell_label) on the stored graph equals the PAGA stored in the labels_core object",
                   "observed": "; ".join(f"{a}-{b} {base[f'paga_stored_{a}_{b}']:.4f}" for a, b in PAGA_PAIRS),
                   "pass": max(diffs) < 1e-6})
    checks = pd.DataFrame(checks)
    run.table(checks, "sensitivity_checks")
    if not checks["pass"].all():
        run.finish(status="stopped_failed_checks")
        raise SystemExit("Reproduction check failed; see ctrl_reviewer_core_sensitivity_checks.csv.")
    rows = [{"dataset": "dct", "run_type": "baseline", "exclusion_set": "none", "draw": "", **base}]
    cells.insert(0, "run", "baseline")
    cells_out, sets = [cells], []
    print(f"baseline done ({time.time() - t0:.0f} s)", flush=True)
    for exclusion, mask in masks.items():
        sets.append({"dataset": "dct", "exclusion_set": exclusion, "n_excluded": int(mask.sum()),
                     "n_remaining": int((~mask).sum()),
                     "excluded_by_primary_label": "; ".join(f"{k}: {v}" for k, v in
                                                           pd.Series(strata[mask]).value_counts().items())})
        res, cells = evaluate(ctx, mask, OBS_SEEDS)
        rows.append({"dataset": "dct", "run_type": "observed", "exclusion_set": exclusion, "draw": "", **res})
        cells.insert(0, "run", f"without_{exclusion}")
        cells_out.append(cells)
        for draw in range(args.n_draws):
            removed = t3s.stratified_draw(rng, strata, mask)
            res, _ = evaluate(ctx, removed, (NULL_SEED,))
            rows.append({"dataset": "dct", "run_type": "random", "exclusion_set": exclusion, "draw": draw,
                         "n_flagged_among_removed": int((removed & mask).sum()), **res})
            if (draw + 1) % 20 == 0:
                print(f"  {exclusion}: {draw + 1} draws ({time.time() - t0:.0f} s)", flush=True)
        run.note(f"{exclusion}: {int(mask.sum())} flagged cells; {args.n_draws} random removals done "
                 f"({time.time() - t0:.0f} s since start).")

    runs = pd.DataFrame(rows)
    summ = null_summary(runs)
    run.table(pd.DataFrame(sets), "sensitivity_exclusion_sets")
    run.table(composition(labels, masks), "sensitivity_composition")
    run.table(runs, "sensitivity_runs")
    run.table(summ, "sensitivity_vs_random")
    run.table(seed_summary(runs), "sensitivity_leiden_seeds")
    run.table(pd.concat(cells_out, ignore_index=True), "sensitivity_per_cell_labels")
    figure(run, summ, runs)
    run.note(f"Ignored igraph RNG messages ('high is out of bounds for int32'): {t3s._IGNORED['igraph_int32_randint']}.")
    run.finish({"n_draws": args.n_draws, "obs_seeds": list(OBS_SEEDS), "null_leiden_seed": NULL_SEED,
                "resolutions": {"ctrl_cluster_label": lib18.SUBMITTED_RESOLUTION, "ctrl_deg_label": 1.0},
                "strata": STRATA, "rng": "numpy default_rng(42), one stream over all draws",
                "ignored_igraph_rng_messages": t3s._IGNORED["igraph_int32_randint"]})


if __name__ == "__main__":
    main()
