#!/usr/bin/env python
"""qc_mito addendum: do labels and Fig 3 structure change when QC-flagged cells are excluded?

(requested by the authors, 13 Oct 2026; random-removal
control added by the authors' decision after the first run, see the qc_mito README).

Exclusion sets (from the step mito_qc per-cell tables; union over both label columns of each dataset):
  mito_flagged  cells flagged high or low on any mitochondrial metric (MitoCarta intensity / ID fraction,
                OXPHOS intensity fraction), 3 x MAD within label
  any_flagged   cells flagged on any qc_mito metric (mitochondrial or depth)

Runs (the same pipeline every time; the protein set is kept fixed):
  baseline  all cells, rebuilt in this environment (the reference for every comparison)
  observed  without the flagged cells
  random    100 draws per exclusion set removing the same number of cells at random, stratified by the
            primary label (kidney_reannot_label / v1_cell_label) so the removed label mix equals the flagged
            set's (numpy default_rng(42))

Pipeline: ln(1 + intensity) -> per-protein z-score (clip 10) -> PCA (arpack, 20) -> neighbours (10, 20 PCs)
-> Leiden (DCT 3.0, KIDNEY 2.0; igraph) -> naming with the submission rules.
  DCT: V1 marker scores on the re-scaled matrix (score_genes defaults) -> per-cell argmax (v1_cell_label)
       and cluster majority with the 'Unknown' rule (v1_cluster_label)          [reviewer_core_lib functions]
  KIDNEY: V3 marker scores on raw intensities (as scp_kidney_upload.py) -> cluster majority with the
       'Unknown' rule (kidney_pre_label)                                        [copied from kidney_object_kidney_canonical.py]
Naming is applied to the stored Leiden clusters (restricted to the remaining cells) and to rerun Leiden
clusters (baseline and observed: seeds 0, 1, 2, 3, 42; random draws: seed 0). kidney_reannot_label cannot
be re-derived (kidney_object) and is reported for composition only.

Outcomes per run: agreement (fraction of remaining cells whose re-derived label equals the submitted label),
ProLIF share (DCT), ARI of rerun clusters vs stored Leiden, and DCT PAGA connectivities for the per-cell
labels (stored graph restricted to the remaining cells with submitted labels; rebuilt graph with re-derived
labels). Observed values are placed in the random-removal distribution (empirical two-sided p).

Inputs (read-only): kidney_object canonical objects/labels; qc_mito per-cell QC tables; data/DCT/DCT_markers_V1.xlsx;
data/KIDNEY/KIDNEY_markers_V3.xlsx.
Outputs: results/tables/qc_mito/qc_mito_sensitivity_*.csv; results/figures/qc_mito/qc_mito_sensitivity_*.

Command (project root):  python scripts/qc_mito_sensitivity.py
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

import numpy as np
import pandas as pd

import scp_common as cc
import reviewer_core_lib as t2
import qc_mito_lib as t3

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "qc_mito_sensitivity.py"
OBS_SEEDS = (0, 1, 2, 3, 42)
NULL_LEIDEN_SEED = 0
N_DRAWS = 100
RESOLUTION = {"dct": 3.0, "kidney": 2.0}
STORED_LEIDEN = {"dct": "v1_leiden_res3_cluster", "kidney": "leiden_res_2.00"}
PRIMARY = {"dct": "v1_cell_label", "kidney": "kidney_reannot_label"}
EXCLUSIONS = ("mito_flagged", "any_flagged")
PAGA_PAIRS = (("DCT1", "DCT2"), ("DCT1", "ProLIF"), ("DCT2", "ProLIF"))
FIG3G = {("DCT1", "DCT2"): 0.9725, ("DCT1", "ProLIF"): 0.8041, ("DCT2", "ProLIF"): 0.7671}
OUTCOMES = {
    "dct": ["cell_agree", "cell_prolif_share", "cluster_stored_agree", "cluster_stored_prolif_share",
            f"cluster_rerun_agree_s{NULL_LEIDEN_SEED}", f"cluster_rerun_prolif_share_s{NULL_LEIDEN_SEED}",
            f"ari_s{NULL_LEIDEN_SEED}", *[f"paga_stored_{a}_{b}" for a, b in PAGA_PAIRS],
            *[f"paga_rebuilt_{a}_{b}" for a, b in PAGA_PAIRS]],
    "kidney": ["pre_stored_agree", f"pre_rerun_agree_s{NULL_LEIDEN_SEED}", f"ari_s{NULL_LEIDEN_SEED}"],
}
OUTCOME_DISPLAY = {
    "cell_agree": "v1_cell_label agreement (re-scored)",
    "cell_prolif_share": "ProLIF share, re-scored v1_cell_label",
    "cluster_stored_agree": "v1_cluster_label agreement (stored clusters re-named)",
    "cluster_stored_prolif_share": "ProLIF share, stored clusters re-named",
    f"cluster_rerun_agree_s{NULL_LEIDEN_SEED}": "v1_cluster_label agreement (Leiden rerun)",
    f"cluster_rerun_prolif_share_s{NULL_LEIDEN_SEED}": "ProLIF share, Leiden rerun",
    f"ari_s{NULL_LEIDEN_SEED}": "ARI rerun vs stored Leiden",
    "pre_stored_agree": "kidney_pre_label agreement (stored clusters re-named)",
    f"pre_rerun_agree_s{NULL_LEIDEN_SEED}": "kidney_pre_label agreement (Leiden rerun)",
    **{f"paga_stored_{a}_{b}": f"PAGA {a}-{b}, stored graph" for a, b in PAGA_PAIRS},
    **{f"paga_rebuilt_{a}_{b}": f"PAGA {a}-{b}, rebuilt graph" for a, b in PAGA_PAIRS},
}

# igraph's random-number bridge prints an ignored 'high is out of bounds for int32' ValueError for many
# draws on this Windows setup; they are counted here instead of printed (runs are not affected otherwise).
_IGNORED = {"igraph_int32_randint": 0}


def _unraisable(hook_args) -> None:
    if isinstance(hook_args.exc_value, ValueError) and "out of bounds for int32" in str(hook_args.exc_value):
        _IGNORED["igraph_int32_randint"] += 1
    else:
        sys.__unraisablehook__(hook_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qc_mito addendum: sensitivity to excluding QC-flagged cells.")
    cc.add_common_args(parser)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    return parser.parse_args()


def exclusion_masks(qc: pd.DataFrame, dataset: str) -> dict[str, np.ndarray]:
    shorts = [t3.SHORT_COLUMN[c] for c in t3.DATASETS[dataset]["label_columns"]]
    mito = np.zeros(len(qc), bool)
    anyf = np.zeros(len(qc), bool)
    for s in shorts:
        mito |= qc[f"flag__{s}__mito_any"].astype(str).str.lower().eq("true").to_numpy()
        anyf |= qc[f"flag__{s}__any"].astype(str).str.lower().eq("true").to_numpy()
    return {"mito_flagged": mito, "any_flagged": anyf}


def stratified_draw(rng, strata: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    """Remove, within each primary label, as many random cells as the flagged set removes there."""
    mask = np.zeros(len(strata), bool)
    for label in np.unique(strata):
        idx = np.flatnonzero(strata == label)
        k = int((flagged & (strata == label)).sum())
        if k:
            mask[rng.choice(idx, size=k, replace=False)] = True
    return mask


# ---------------------------------------------------------------------------
# Pipeline pieces
# ---------------------------------------------------------------------------
def rebuild_graph(ln1p: np.ndarray, cell_ids, var_names):
    """Scale (clip 10), arpack PCA (20 comps), neighbours (10, 20 PCs), as the submitted pipelines."""
    import anndata as ad
    import scanpy as sc

    x = cc.scale_like_scanpy(ln1p).astype(np.float32)
    g = ad.AnnData(X=x, obs=pd.DataFrame(index=pd.Index(list(cell_ids), name="cell_id")),
                   var=pd.DataFrame(index=pd.Index(list(var_names))))
    sc.pp.pca(g, n_comps=20, svd_solver="arpack")
    sc.pp.neighbors(g, n_neighbors=10, n_pcs=20)
    return g


def leiden(g, resolution: float, seed: int, key: str) -> pd.Series:
    import scanpy as sc

    try:
        sc.tl.leiden(g, key_added=key, resolution=resolution, flavor="igraph", random_state=seed)
    except TypeError:
        sc.tl.leiden(g, key_added=key, resolution=resolution, random_state=seed)
    return g.obs[key].astype(str)


def kidney_marker_genes(path) -> dict[str, list[str]]:
    """Marker parsing exactly as scp_kidney_upload.py (strip, lower, capitalize); copied from kidney_object."""
    marker_df = pd.read_excel(path)
    return {str(row[0]): [g.strip().lower().capitalize() for g in str(row[1]).split(",")]
            for row in marker_df.itertuples(index=False)}


def kidney_scores(x_scaled: np.ndarray, intensity: np.ndarray, var_names, cell_ids, marker_genes) -> pd.DataFrame:
    """V3 marker scores as scp_kidney_upload.py: score_genes defaults on adata.raw (pre-log intensities).

    Copied from kidney_object_kidney_canonical.rederive_pre_labels (which reproduces the stored scores exactly).
    """
    import anndata as ad
    import scanpy as sc

    obs = pd.DataFrame(index=pd.Index(list(cell_ids), name="cell_id"))
    var = pd.DataFrame(index=pd.Index(list(var_names)))
    tmp = ad.AnnData(X=np.asarray(x_scaled, dtype=np.float32).copy(), obs=obs, var=var)
    tmp.raw = ad.AnnData(X=np.asarray(intensity, dtype=np.float32).copy(), obs=obs.copy(), var=var.copy())
    matching = {ct: [g for g in gl if g in tmp.var_names] for ct, gl in marker_genes.items()}
    matching = {ct: gl for ct, gl in matching.items() if gl}
    for ct, gl in matching.items():
        sc.tl.score_genes(tmp, gene_list=gl, score_name=f"{ct}_score")
    return pd.DataFrame({ct: tmp.obs[f"{ct}_score"].to_numpy() for ct in matching}, index=tmp.obs_names)


def kidney_names(scores: pd.DataFrame, clusters: pd.Series) -> pd.Series:
    """Cluster naming rules of scp_kidney_upload.py (kidney_object): per-cell argmax -> cluster mode; clusters whose mean
    max score is below the 1% quantile of cluster means (or whose diversity exceeds the maximum) -> 'Unknown'."""
    predicted = scores.idxmax(axis=1).astype(object)
    frame = pd.DataFrame({"cluster": clusters.astype(str).to_numpy(), "predicted": predicted.to_numpy(),
                          "max_score": scores.max(axis=1).to_numpy()}, index=scores.index)
    clust_mean = frame.groupby("cluster")["max_score"].mean()
    clust_div = frame.groupby("cluster")["predicted"].nunique()
    unknown = clust_mean[clust_mean < clust_mean.quantile(0.01)].index.union(
        clust_div[clust_div > clust_div.quantile(1)].index)
    mapping = frame.groupby("cluster")["predicted"].agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    named = frame["cluster"].map(mapping)
    named[frame["cluster"].isin(unknown)] = "Unknown"
    return named.map(lambda v: cc.KIDNEY_ABBREVIATIONS.get(v, v)).astype(str)


def paga(g, labels) -> dict[tuple[str, str], float]:
    import scanpy as sc

    tmp = g.copy()
    tmp.obs["grp"] = pd.Categorical(np.asarray(labels).astype(str))
    sc.tl.paga(tmp, groups="grp")
    cats = list(tmp.obs["grp"].cat.categories)
    conn = cc.dense(tmp.uns["paga"]["connectivities"])
    return {(a, b): float(conn[cats.index(a), cats.index(b)]) if a in cats and b in cats else np.nan
            for a, b in PAGA_PAIRS}


def frac(a, b) -> float:
    return float((np.asarray(a).astype(str) == np.asarray(b).astype(str)).mean())


def share(values, label: str = "ProLIF") -> float:
    return float((np.asarray(values).astype(str) == label).mean())


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
class Context:
    def __init__(self, dataset, adata, labels, markers):
        self.dataset, self.adata, self.labels, self.markers = dataset, adata, labels, markers
        self.intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
        self.ln1p = np.log1p(self.intensity)
        self.ids = np.asarray(adata.obs_names.astype(str))
        self.var_names = adata.var_names.astype(str)
        self.stored_leiden = adata.obs[STORED_LEIDEN[dataset]].astype(str).to_numpy()


def evaluate(ctx: Context, removed: np.ndarray, seeds) -> tuple[dict, pd.DataFrame | None]:
    from sklearn.metrics import adjusted_rand_score

    keep = ~removed
    ids = ctx.ids[keep]
    sub = ctx.labels.loc[ids]
    g = rebuild_graph(ctx.ln1p[keep], ids, ctx.var_names)
    stored = pd.Series(ctx.stored_leiden[keep], index=ids)
    out = {"n_removed": int(removed.sum()), "n_remaining": int(keep.sum())}
    cells = pd.DataFrame({"cell_id": ids})
    if ctx.dataset == "dct":
        scores, _ = t2.score_v1_lists(g, ctx.markers)
        cell, max_score, _, _ = t2.cell_labels(scores)
        named, _, _ = t2.cluster_labels(cell, max_score, stored)
        out.update(cell_agree=frac(cell, sub["v1_cell_label"]), cell_prolif_share=share(cell),
                   cluster_stored_agree=frac(named, sub["v1_cluster_label"]), cluster_stored_prolif_share=share(named),
                   submitted_cell_prolif_share=share(sub["v1_cell_label"]),
                   submitted_cluster_prolif_share=share(sub["v1_cluster_label"]))
        cells["v1_cell_label_submitted"] = sub["v1_cell_label"].to_numpy()
        cells["v1_cell_label_rederived"] = cell.to_numpy()
        cells["v1_cluster_label_submitted"] = sub["v1_cluster_label"].to_numpy()
        cells["v1_cluster_label_renamed_stored_clusters"] = named.to_numpy()
        for seed in seeds:
            clusters = leiden(g, RESOLUTION["dct"], seed, f"leiden_{seed}")
            named_r, _, _ = t2.cluster_labels(cell, max_score, clusters)
            out.update({f"cluster_rerun_agree_s{seed}": frac(named_r, sub["v1_cluster_label"]),
                        f"cluster_rerun_prolif_share_s{seed}": share(named_r),
                        f"ari_s{seed}": float(adjusted_rand_score(stored.to_numpy(), clusters.to_numpy())),
                        f"n_clusters_s{seed}": int(clusters.nunique())})
            cells[f"v1_cluster_label_rerun_seed{seed}"] = named_r.to_numpy()
        for (a, b), v in paga(ctx.adata[keep].copy(), sub["v1_cell_label"]).items():
            out[f"paga_stored_{a}_{b}"] = v
        for (a, b), v in paga(g, cell).items():
            out[f"paga_rebuilt_{a}_{b}"] = v
    else:
        scores = kidney_scores(cc.dense(g.X), ctx.intensity[keep], ctx.var_names, ids, ctx.markers)
        named = kidney_names(scores, stored)
        out.update(pre_stored_agree=frac(named, sub["kidney_pre_label"]))
        cells["kidney_pre_label_submitted"] = sub["kidney_pre_label"].to_numpy()
        cells["kidney_pre_label_renamed_stored_clusters"] = named.to_numpy()
        for seed in seeds:
            clusters = leiden(g, RESOLUTION["kidney"], seed, f"leiden_{seed}")
            named_r = kidney_names(scores, pd.Series(clusters.to_numpy(), index=ids))
            out.update({f"pre_rerun_agree_s{seed}": frac(named_r, sub["kidney_pre_label"]),
                        f"ari_s{seed}": float(adjusted_rand_score(stored.to_numpy(), clusters.to_numpy())),
                        f"n_clusters_s{seed}": int(clusters.nunique())})
            cells[f"kidney_pre_label_rerun_seed{seed}"] = named_r.to_numpy()
    return out, cells


# ---------------------------------------------------------------------------
# Summaries and figure
# ---------------------------------------------------------------------------
def composition(labels: pd.DataFrame, dataset: str, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for col in t3.DATASETS[dataset]["plot_columns"]:
        values = labels[col].astype(str).to_numpy()
        for exclusion, mask in masks.items():
            keep = ~mask
            for label in t3.label_order(values, dataset):
                n_before = int((values == label).sum())
                n_after = int((values[keep] == label).sum())
                rows.append({"dataset": dataset, "exclusion_set": exclusion, "label_column": col, "label": label,
                             "n_before": n_before, "n_after": n_after, "n_excluded": n_before - n_after,
                             "percent_excluded_of_label": 100 * (n_before - n_after) / n_before if n_before else np.nan,
                             "share_before": n_before / len(values), "share_after": n_after / keep.sum()})
    return pd.DataFrame(rows)


def null_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("kidney", "dct"):
        base = runs[(runs["dataset"] == dataset) & (runs["run_type"] == "baseline")].iloc[0]
        for exclusion in EXCLUSIONS:
            obs = runs[(runs["dataset"] == dataset) & (runs["run_type"] == "observed") &
                       (runs["exclusion_set"] == exclusion)].iloc[0]
            null = runs[(runs["dataset"] == dataset) & (runs["run_type"] == "random") &
                        (runs["exclusion_set"] == exclusion)]
            for outcome in OUTCOMES[dataset]:
                x = null[outcome].to_numpy(float)
                x = x[np.isfinite(x)]
                o = float(obs[outcome])
                n = len(x)
                p_le = (1 + (x <= o).sum()) / (n + 1)
                p_ge = (1 + (x >= o).sum()) / (n + 1)
                rows.append({"dataset": dataset, "exclusion_set": exclusion, "outcome": outcome,
                             "display": OUTCOME_DISPLAY[outcome], "n_removed": int(obs["n_removed"]),
                             "baseline": float(base[outcome]), "observed": o,
                             "observed_minus_baseline": o - float(base[outcome]),
                             "random_n": n, "random_median": float(np.median(x)), "random_q025": float(np.quantile(x, 0.025)),
                             "random_q975": float(np.quantile(x, 0.975)), "random_min": float(x.min()),
                             "random_max": float(x.max()), "observed_percentile_in_random": 100 * float((x < o).mean()),
                             "p_empirical_lower": p_le, "p_empirical_upper": p_ge,
                             "p_empirical_two_sided": min(1.0, 2 * min(p_le, p_ge))})
    return pd.DataFrame(rows)


def obs_seed_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Rerun-Leiden agreement over the five seeds for the baseline and observed runs."""
    rows = []
    for _, r in runs[runs["run_type"].isin(["baseline", "observed"])].iterrows():
        key = "cluster_rerun_agree_s" if r["dataset"] == "dct" else "pre_rerun_agree_s"
        vals = np.array([r[f"{key}{s}"] for s in OBS_SEEDS], float)
        aris = np.array([r[f"ari_s{s}"] for s in OBS_SEEDS], float)
        rows.append({"dataset": r["dataset"], "run_type": r["run_type"], "exclusion_set": r["exclusion_set"],
                     "n_removed": int(r["n_removed"]), "seeds": ";".join(map(str, OBS_SEEDS)),
                     "rerun_agree_min": vals.min(), "rerun_agree_median": np.median(vals), "rerun_agree_max": vals.max(),
                     "ari_min": aris.min(), "ari_median": np.median(aris), "ari_max": aris.max()})
    return pd.DataFrame(rows)


def figure(run, summ: pd.DataFrame, runs: pd.DataFrame) -> None:
    panels = [("dct", "cell_agree"), ("dct", "cluster_stored_agree"), ("dct", f"cluster_rerun_agree_s{NULL_LEIDEN_SEED}"),
              ("dct", "cell_prolif_share"), ("dct", "paga_stored_DCT1_ProLIF"), ("dct", "paga_stored_DCT2_ProLIF"),
              ("kidney", "pre_stored_agree"), ("kidney", f"pre_rerun_agree_s{NULL_LEIDEN_SEED}")]
    colors = {"mito_flagged": t3.SERIES[0], "any_flagged": t3.SERIES[1]}
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.3))
    rng = np.random.default_rng(t3.SEED)
    for ax, (dataset, outcome) in zip(axes.flat, panels):
        for k, exclusion in enumerate(EXCLUSIONS):
            null = runs[(runs["dataset"] == dataset) & (runs["run_type"] == "random") &
                        (runs["exclusion_set"] == exclusion)][outcome].to_numpy(float)
            ax.scatter(null, k + rng.uniform(-0.18, 0.18, len(null)), s=3, color=t3.INK["background_cells"],
                       linewidths=0, rasterized=True)
            q = np.quantile(null, [0.025, 0.5, 0.975])
            ax.plot([q[0], q[2]], [k, k], color=t3.INK["muted"], linewidth=0.8)
            ax.scatter([q[1]], [k], marker="|", s=40, color=t3.INK["secondary"], linewidths=1.0)
            r = summ[(summ["dataset"] == dataset) & (summ["exclusion_set"] == exclusion) & (summ["outcome"] == outcome)].iloc[0]
            ax.scatter([r["observed"]], [k], s=22, color=colors[exclusion], zorder=4, linewidths=0.4, edgecolors="white")
        base = summ[(summ["dataset"] == dataset) & (summ["outcome"] == outcome)]["baseline"].iloc[0]
        ax.axvline(base, color=t3.INK["primary"], linewidth=0.6, linestyle=(0, (3, 2)))
        if outcome.startswith("paga_stored"):
            pair = tuple(outcome.removeprefix("paga_stored_").split("_"))
            ax.axvline(FIG3G[pair], color=t3.INK["muted"], linewidth=0.6)
        ax.set_yticks([0, 1], ["mito-flagged", "any-flagged"], fontsize=5.6)
        ax.set_ylim(1.6, -0.6)
        title = textwrap.fill(f"{t3.DATASETS[dataset]['display']}: {OUTCOME_DISPLAY[outcome]}", 34)
        ax.set_title(title, fontsize=5.6)
        ax.tick_params(axis="x", labelsize=5.4)
        ax.xaxis.grid(True, color=t3.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Excluding QC-flagged cells vs removing the same number of random cells (label-stratified)",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    t3.footer(fig, SCRIPT, "coloured = flagged cells removed; grey = 100 random removals (bar: median, 95% range); "
                           "dashed = all cells (rerun); solid grey = Fig 3G")
    run.figure(fig, "sensitivity_vs_random", "Flagged-cell exclusion against random removal")


def main() -> None:
    args = parse_args()
    sys.unraisablehook = _unraisable
    run = cc.Run(t3.TASK, "sensitivity", args)
    rng = np.random.default_rng(t3.SEED)
    rows, cells_out, comp, sets, checks = [], [], [], [], []
    for dataset in ("kidney", "dct"):
        adata, labels = t3.load(run, dataset)
        qc_path = cc.out_path(t3.TASK, "tables", f"{dataset}_qc_per_cell", ".csv")
        qc = pd.read_csv(run.use_input(qc_path, f"qc_mito per-cell QC ({dataset})"), dtype={"cell_id": str}).set_index("cell_id")
        if list(qc.index) != list(labels.index):
            raise ValueError(f"{dataset}: per-cell QC order differs from the canonical labels.")
        masks = exclusion_masks(qc, dataset)
        comp.append(composition(labels, dataset, masks))
        marker_path = run.use_input(cc.INPUTS[dataset]["markers"], f"{dataset} marker workbook (Supp Table 1)")
        markers = cc.read_marker_workbook(marker_path) if dataset == "dct" else kidney_marker_genes(marker_path)
        ctx = Context(dataset, adata, labels, markers)
        strata = labels[PRIMARY[dataset]].astype(str).to_numpy()

        # Reproduction checks on the submitted objects (not on the rebuilt matrix).
        if dataset == "dct":
            scores, _ = t2.score_v1_lists(adata, markers)
            cell, _, _, _ = t2.cell_labels(scores)
            n_ok = int((cell.to_numpy() == labels["v1_cell_label"].to_numpy()).sum())
            checks.append({"dataset": dataset, "check": "V1 scoring on the stored .X reproduces v1_cell_label",
                           "observed": f"{n_ok}/{len(cell)}", "pass": n_ok == len(cell)})
            conn = paga(adata, labels["v1_cell_label"])
            ok = all(abs(conn[p] - FIG3G[p]) < 5e-5 for p in PAGA_PAIRS)
            checks.append({"dataset": dataset, "check": "PAGA on the full stored graph reproduces Fig 3G",
                           "observed": "; ".join(f"{a}-{b} {conn[(a, b)]:.4f}" for a, b in PAGA_PAIRS), "pass": ok})
        else:
            scores = kidney_scores(cc.dense(adata.X), ctx.intensity, ctx.var_names, ctx.ids, markers)
            named = kidney_names(scores, pd.Series(ctx.stored_leiden, index=ctx.ids))
            n_ok = int((named.to_numpy() == labels["kidney_pre_label"].to_numpy()).sum())
            checks.append({"dataset": dataset, "check": "naming of the stored Leiden 2.0 clusters reproduces kidney_pre_label",
                           "observed": f"{n_ok}/{len(named)}", "pass": n_ok == len(named)})

        t0 = time.time()
        res, cells = evaluate(ctx, np.zeros(len(strata), bool), OBS_SEEDS)
        rows.append({"dataset": dataset, "run_type": "baseline", "exclusion_set": "none", "draw": "", **res})
        cells.insert(0, "run", "baseline")
        cells.insert(0, "dataset", dataset)
        cells_out.append(cells)
        for exclusion, mask in masks.items():
            sets.append({"dataset": dataset, "exclusion_set": exclusion, "n_excluded": int(mask.sum()),
                         "n_remaining": int((~mask).sum()),
                         "excluded_by_primary_label": "; ".join(f"{k}: {v}" for k, v in
                                                               pd.Series(strata[mask]).value_counts().items())})
            res, cells = evaluate(ctx, mask, OBS_SEEDS)
            rows.append({"dataset": dataset, "run_type": "observed", "exclusion_set": exclusion, "draw": "", **res})
            cells.insert(0, "run", f"without_{exclusion}")
            cells.insert(0, "dataset", dataset)
            cells_out.append(cells)
            for draw in range(args.n_draws):
                removed = stratified_draw(rng, strata, mask)
                res, _ = evaluate(ctx, removed, (NULL_LEIDEN_SEED,))
                rows.append({"dataset": dataset, "run_type": "random", "exclusion_set": exclusion, "draw": draw,
                             "n_flagged_among_removed": int((removed & mask).sum()), **res})
            run.note(f"{dataset}/{exclusion}: {int(mask.sum())} flagged cells; {args.n_draws} random removals done "
                     f"({time.time() - t0:.0f} s since the dataset started).")
        print(f"[{dataset}] done in {time.time() - t0:.0f} s", flush=True)

    runs = pd.DataFrame(rows)
    summ = null_summary(runs)
    checks = pd.DataFrame(checks)
    run.table(pd.DataFrame(sets), "sensitivity_exclusion_sets")
    run.table(pd.concat(comp, ignore_index=True), "sensitivity_composition")
    run.table(runs, "sensitivity_runs")
    run.table(summ, "sensitivity_vs_random")
    run.table(obs_seed_summary(runs), "sensitivity_leiden_seeds")
    run.table(pd.concat(cells_out, ignore_index=True), "sensitivity_per_cell_labels")
    run.table(checks, "sensitivity_checks")
    figure(run, summ, runs)
    run.note(f"Ignored igraph RNG messages ('high is out of bounds for int32'): {_IGNORED['igraph_int32_randint']}.")
    ok = bool(checks["pass"].all())
    run.finish({"n_draws": args.n_draws, "obs_seeds": list(OBS_SEEDS), "null_leiden_seed": NULL_LEIDEN_SEED,
                "resolution": RESOLUTION, "strata": PRIMARY, "rng": "numpy default_rng(42), one stream over all draws",
                "ignored_igraph_rng_messages": _IGNORED["igraph_int32_randint"], "checks_pass": ok},
               status="completed" if ok else "completed_with_failed_checks")
    if not ok:
        raise SystemExit("Reproduction check failed; see qc_mito_sensitivity_checks.csv.")


if __name__ == "__main__":
    main()
