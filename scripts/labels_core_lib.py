"""labels_core shared definitions and functions. Runs no analysis by itself.

Imported by the
``labels_core_*.py`` step scripts. Plan: ``the analysis plan``.

Settings agreed with the authors (28 Oct 2026)
-------------------------------------------
- Data: the intake_core control object (727 cells x 6,196 proteins; new DCT data). The
  retired 758-cell V1 data and their labels are not read.
- Submitted pipeline (``scp_dct_upload.py``), unchanged except the seed:
  float32 intensities; ``sc.pp.log1p``; ``sc.pp.scale(max_value=10)``;
  ``sc.pp.neighbors(n_neighbors=10, n_pcs=20)`` without a prior PCA (Scanpy's
  fallback computes 20 PCs); Leiden 3.0 (igraph); ``sc.tl.score_genes`` per
  workbook type on the markers present, scored on the scaled ``.X``; per-cell
  argmax; cluster mode with the original 'Unknown' rule;
  ``sc.tl.pca(svd_solver='arpack')``; UMAP; DPT from the first DCT1 cell; PAGA
  on the per-cell label.
- Scored matrix: the script sets ``adata.raw = adata`` without a copy (line 106),
  so the in-place log1p and scale also reach ``.raw``. In the submitted DCT
  object ``.raw.X`` equals the scaled ``.X`` (kidney_object), and kidney_object/reviewer_core reproduced the
  stored scores exactly by scoring the scaled ``.X``. labels_core scores ``.X`` directly
  and keeps the intensities in ``layers['intensity']`` (no ``.raw``). The kidney
  script differs (``adata.raw = adata.copy()``: pre-log intensities).
- Seed 42 for Leiden and UMAP (plan; the script used Scanpy's default 0, which
  is one of the 11 stability seeds). PCA, neighbours and score_genes keep the
  script's Scanpy defaults.
- DE-rule labels: the cluster_de rule (``cluster_de_lib``) on the same graph at
  Leiden 0.3, 0.5, 0.8, 1.0 and 3.0, seeds 42 and 0-9, minimum cluster size 5
  (kidney_annotation). Calls resting only on Slc12a3 and/or Umod get confidence
  'low_shared_only' (plan: Umod is treated as shared, TAL-associated), so Umod
  is not type-specific for the confidence or the kidney_annotation rank-free rescue.
- Noise tests (authors, 28 Oct 2026; the plan's 'reviewer_core noise test' does not exist):
  N(0, 1e-6) noise, 20 repeats; (A) on the scored matrix (the scaled ``.X``)
  -> scores -> per-cell label; (B) on the scaled ``.X`` -> PCA, graph, Leiden
  -> cluster labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import intake_core_lib as t17
from reviewer_core_lib import LABEL_PALETTE
from qc_mito_lib import INK, SERIES, apply_style, footer

__all__ = ["INK", "SERIES", "LABEL_PALETTE", "apply_style", "footer"]

TASK = "labels_core"
SEED = cc.RANDOM_SEED
CONDITION = "Ctrl"
DATASET = "dct"                       # cluster_de_lib dataset key (DCT label order, DCT1 flag)

# Submitted pipeline (scp_dct_upload.py; line numbers of the root copy)
N_NEIGHBORS = 10
N_PCS = 20
SUBMITTED_RESOLUTION = 3.0
LEIDEN_KEY = f"leiden_res_{SUBMITTED_RESOLUTION:.2f}"
NAMED_KEY = f"{LEIDEN_KEY}_named"
UNKNOWN_SCORE_QUANTILE = 0.01         # clusters with mean max score below this quantile -> Unknown (line 150)
UNKNOWN_DIVERSITY_QUANTILE = 1.0      # clusters with more distinct per-cell labels than this quantile (line 151)
N_RANDOM_ROOTS = 20
SUBMITTED_SETTINGS = (
    # (setting, value used here, scp_dct_upload.py line, note)
    ("cell filter", "cells with >= 100 proteins > 0 (all rows)", "103", "applied in intake_core"),
    ("protein filter", "proteins > 0 in >= 2 retained cells", "104", "applied in intake_core"),
    ("matrix dtype", "float32, missing = 0", "75", ""),
    (".raw", "not set; scores use the scaled .X; intensities kept in layers['intensity']", "106",
     "script: adata.raw = adata without a copy, so .raw.X = scaled .X in the submitted object (kidney_object)"),
    ("log transform", "sc.pp.log1p (natural log)", "109", ""),
    ("scaling", "sc.pp.scale(max_value=10)", "112", ""),
    ("neighbours", f"sc.pp.neighbors(n_neighbors={N_NEIGHBORS}, n_pcs={N_PCS}); no PCA before it", "115",
     "Scanpy computes the PCA itself (fallback, Scanpy defaults)"),
    ("Leiden", f"resolution {SUBMITTED_RESOLUTION}, flavor='igraph', random_state={SEED}", "118-120",
     "script: Scanpy default random_state=0"),
    ("marker scores", "sc.tl.score_genes per workbook type on the markers present, on the scaled .X (defaults: "
     "ctrl_size=50, n_bins=25, random_state=0)", "123-134", "as reproduced exactly for the submitted object (kidney_object, reviewer_core D.1)"),
    ("per-cell label", "argmax of the scores (first type in workbook order on ties)", "140", "ctrl_cell_label"),
    ("cluster label", "mode of the per-cell label per Leiden cluster (alphabetical on ties); clusters with mean max "
     "score < 1% quantile of cluster means, or more distinct labels than the maximum, -> Unknown", "145-168",
     "ctrl_cluster_label"),
    ("PCA after annotation", "sc.tl.pca(svd_solver='arpack') (50 PCs; overwrites X_pca)", "178",
     "the graph keeps the neighbour PCA (stored as X_pca_neighbors)"),
    ("UMAP", f"sc.tl.umap on the stored graph, random_state={SEED}", "179", "script: Scanpy default 0"),
    ("DPT root", "first cell (object order) with per-cell label DCT1", "272-278", ""),
    ("DPT", "sc.tl.dpt (diffusion map computed by its fallback, Scanpy defaults)", "281", ""),
    ("PAGA", "sc.tl.paga(groups=per-cell label); edges = upper triangle > 0", "325-340", ""),
)

# DE rule (cluster_de / kidney_annotation)
SCAN_RESOLUTIONS = (0.3, 0.5, 0.8, 1.0, 3.0)
LOW_RESOLUTIONS = (0.3, 0.5, 0.8, 1.0)
SEEDS = (SEED, *t5.STABILITY_SEEDS)   # 42, then 0-9
MIN_CLUSTER_SIZE = 5
UNKNOWN = t5.UNKNOWN
BELOW_MIN = "below_min_size"          # same string as kidney_annotation_scan_lib.BELOW_MIN (rank_free relies on it)
RULES = ("top25", "rankfree")
RULE_DISPLAY = {"top25": "top-25 rule (cluster_de)", "rankfree": "rank-free rescue (kidney_annotation)"}
RESCUED = "specific_any_rank"
CONFIDENCE_ORDER = ("high", "medium", RESCUED, "low_shared_only", "unknown_no_marker_in_top_de", BELOW_MIN)
CONFIDENCE_RANK = {"high": 3, "medium": 2, RESCUED: 1.5, "low_shared_only": 1}
# Plan: low confidence for calls resting only on shared markers; Umod is TAL-associated (the project notes framing).
TREATED_AS_SHARED = {"Umod": "TAL-associated (the project notes framing; labels_core plan)"}

# Noise tests (authors, 28 Oct 2026)
NOISE_SD = 1e-6
NOISE_REPEATS = 20

CORE_TYPES = ("DCT1", "DCT2", "ProLIF")
LABEL_ORDER = (*CORE_TYPES, UNKNOWN, BELOW_MIN)
LABEL_COLORS = {**LABEL_PALETTE, BELOW_MIN: INK["background_cells"]}
MARKER_WORKBOOK = cc.INPUTS["dct"]["markers"]     # data/DCT/DCT_markers_V1.xlsx (the workbook only)
CONTAMINATION_PANEL = t17.CONTAMINATION_PANEL     # information only (not used for any label)

STEP1_H5AD = cc.out_path(TASK, "h5ad", "DCT_ctrl_step1_submitted", ".h5ad")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_ctrl(run: cc.Run):
    """intake_core control object (input recorded)."""
    return t17.read_t17_object(run, CONDITION)


def load_step1(run: cc.Run):
    import anndata as ad

    if not STEP1_H5AD.exists():
        raise FileNotFoundError(f"{cc.rel(STEP1_H5AD)} not found; run labels_core_submitted_pipeline.py first.")
    return ad.read_h5ad(run.use_input(STEP1_H5AD, "labels_core step 1 object (submitted pipeline)"))


def read_workbook(run: cc.Run) -> dict[str, list[str]]:
    """DCT V1 workbook lists in workbook order (names cleaned as the script: strip, lower, capitalize)."""
    return cc.read_marker_workbook(run.use_input(MARKER_WORKBOOK, "DCT marker workbook (Supp Table 1 DCT tab)"))


def present_lists(workbook: dict[str, list[str]], var_names) -> dict[str, list[str]]:
    """Script lines 123-129: markers present in the matrix; types without any are dropped."""
    var = set(map(str, var_names))
    lists = {t: [g for g in genes if g in var] for t, genes in workbook.items()}
    return {t: genes for t, genes in lists.items() if genes}


def dct_markers(run: cc.Run, var_names) -> t5.MarkerSets:
    """cluster_de marker sets for the DCT workbook with Umod treated as shared (``TREATED_AS_SHARED``)."""
    markers = t5.read_markers(run, DATASET, var_names)
    for gene, reason in TREATED_AS_SHARED.items():
        if gene in markers.types_by_gene:
            markers.types_by_gene[gene] = set(markers.types_by_gene[gene]) | {reason}
    return markers


def specific_present(markers: t5.MarkerSets) -> dict[str, list[str]]:
    return {t: markers.specific_present(t) for t in markers.lists}


def drop_markers(markers: t5.MarkerSets, genes) -> t5.MarkerSets:
    """Marker sets with ``genes`` removed from every list (sharedness of the other markers unchanged)."""
    drop = set(genes)
    return t5.MarkerSets({t: [g for g in gs if g not in drop] for t, gs in markers.lists.items()},
                         {t: [g for g in gs if g not in drop] for t, gs in markers.present.items()},
                         dict(markers.full_names),
                         {g: set(ts) for g, ts in markers.types_by_gene.items() if g not in drop})


def drop_from_lists(lists: dict[str, list[str]], genes) -> dict[str, list[str]]:
    """Scoring lists with ``genes`` removed; types left without markers are dropped (script lines 128-129)."""
    out = {t: [g for g in gs if g not in set(genes)] for t, gs in lists.items()}
    return {t: gs for t, gs in out.items() if gs}


# Marker-list variants (reviewer_core section D)
MARKER_VARIANTS = {
    "noSlc12a3": ("Slc12a3 removed from every list", ("Slc12a3",)),
    "noSlc12a3_noUmod": ("Slc12a3 removed from every list; Umod removed from DCT1", ("Slc12a3", "Umod")),
}


# ---------------------------------------------------------------------------
# Submitted pipeline
# ---------------------------------------------------------------------------
def submitted_object(ctrl):
    """Script lines 75-112 on the intake_core matrix: float32 intensities, log1p, scale (intensities kept as a layer)."""
    import anndata as ad
    import scanpy as sc

    obs = ctrl.obs[["plate", "well", "n_detected", "total_intensity"]].copy()
    intensity = np.array(cc.dense(ctrl.layers["intensity"]), dtype=np.float32, copy=True)
    adata = ad.AnnData(X=intensity.copy(), obs=obs, var=pd.DataFrame(index=ctrl.var_names.astype(str).copy()))
    adata.var["gene_name"] = ctrl.var["gene_name"].astype(str).to_numpy()
    adata.layers["intensity"] = intensity
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    return adata


def score_types(adata, lists: dict[str, list[str]]) -> pd.DataFrame:
    """Script lines 133-137: ``sc.tl.score_genes`` per type on ``.X`` (writes ``<type>_score`` to obs)."""
    import scanpy as sc

    for cell_type, genes in lists.items():
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"{cell_type}_score", use_raw=False)
    return pd.DataFrame({t: adata.obs[f"{t}_score"].to_numpy() for t in lists}, index=adata.obs_names)


def score_matrix(matrix: np.ndarray, var_names, obs_names, lists: dict[str, list[str]]) -> pd.DataFrame:
    """The same scores computed on a bare matrix (noise test A; information-only scoring inputs)."""
    import anndata as ad

    tmp = ad.AnnData(X=np.array(matrix, dtype=np.float32, copy=True), obs=pd.DataFrame(index=pd.Index(obs_names)),
                     var=pd.DataFrame(index=pd.Index(var_names).astype(str)))
    return score_types(tmp, lists)


def cell_labels(scores: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Script lines 140-143: per-cell argmax (first column on ties) and max score."""
    return scores.idxmax(axis=1).astype(str), scores.max(axis=1)


def cluster_labels(cell_label: pd.Series, max_score: pd.Series, clusters: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Script lines 145-171: cluster mode of the per-cell label with the original 'Unknown' rule.

    Returns the per-cell cluster label and a per-cluster table (mean max score, distinct labels, mode, ties).
    """
    obs = pd.DataFrame({"cluster": pd.Categorical(clusters.astype(str),
                                                  categories=t5.sort_clusters(clusters.astype(str))),
                        "label": cell_label.astype(str).to_numpy(), "max_score": max_score.to_numpy()},
                       index=cell_label.index)
    cluster_scores = obs.groupby("cluster", observed=True)["max_score"].mean()
    cluster_diversity = obs.groupby("cluster", observed=True)["label"].nunique()
    score_threshold = cluster_scores.quantile(UNKNOWN_SCORE_QUANTILE)
    diversity_threshold = cluster_diversity.quantile(UNKNOWN_DIVERSITY_QUANTILE)
    unknown = cluster_scores[cluster_scores < score_threshold].index.union(
        cluster_diversity[cluster_diversity > diversity_threshold].index)
    mode = obs.groupby("cluster", observed=True)["label"].agg(lambda x: x.mode()[0] if not x.mode().empty else UNKNOWN)
    named = obs["cluster"].astype(str).map(mode.to_dict()).astype(str)
    named[obs["cluster"].astype(str).isin(set(map(str, unknown)))] = UNKNOWN
    rows = []
    for c, grp in obs.groupby("cluster", observed=True):
        counts = grp["label"].value_counts()
        n_modes = int((counts == counts.max()).sum())
        rows.append({"cluster": str(c), "n_cells": int(len(grp)), "mean_max_score": float(cluster_scores[c]),
                     "n_distinct_cell_labels": int(cluster_diversity[c]), "mode_cell_label": str(mode[c]),
                     "mode_fraction": float(counts.max() / len(grp)), "mode_tied": n_modes > 1,
                     "cell_label_counts": ";".join(f"{k} {v}" for k, v in counts.items()),
                     "unknown_by_score": bool(cluster_scores[c] < score_threshold),
                     "unknown_by_diversity": bool(cluster_diversity[c] > diversity_threshold),
                     "cluster_label": UNKNOWN if str(c) in set(map(str, unknown)) else str(mode[c]),
                     "score_threshold": float(score_threshold), "diversity_threshold": float(diversity_threshold)})
    return named, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DE rule (cluster_de with minimum cluster size, as kidney_annotation_scan_lib.annotate_clustering, DCT dataset)
# ---------------------------------------------------------------------------
def leiden_on(graph_obj, resolution: float, seed: int) -> np.ndarray:
    return t5.leiden(graph_obj, resolution, "t18_tmp", seed=seed).to_numpy().astype(str)


def graph_adata(adata):
    """Small AnnData carrying only the step-1 graph (for repeated Leiden calls)."""
    import anndata as ad
    from scipy import sparse

    g = ad.AnnData(X=sparse.csr_matrix((adata.n_obs, 1), dtype=np.float32),
                   obs=pd.DataFrame(index=adata.obs_names.copy()))
    g.obsp["connectivities"] = adata.obsp["connectivities"]
    g.obsp["distances"] = adata.obsp["distances"]
    g.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": dict(adata.uns["neighbors"]["params"])}
    return g


def annotate_clustering(cache: t5.RankCache, clusters: np.ndarray, var_names, name: str, markers: t5.MarkerSets,
                        x_z: pd.DataFrame, detected: pd.DataFrame, min_size: int = MIN_CLUSTER_SIZE,
                        de: pd.DataFrame | None = None):
    """cluster_de DE, evidence and assignment for clusters with >= ``min_size`` cells (kidney_annotation rule; DCT dataset).

    Smaller clusters get ``below_min_size`` and stay in the 'rest' of the other clusters' tests.
    Returns (de, evidence, assignment); ``de`` holds the annotated clusters only.
    """
    clusters = np.asarray(clusters).astype(str)
    sizes = pd.Series(clusters).value_counts()
    big = {str(c) for c, n in sizes.items() if n >= min_size}
    if de is None:
        de = t5.de_one_vs_rest(cache, clusters, var_names, name)
    de = de[de["cluster"].isin(big)].reset_index(drop=True)
    ev = t5.marker_evidence(de, markers, x_z, detected, clusters, t5.DATASETS[DATASET]["order"])
    ev.insert(0, "clustering", name)
    asg = t5.assign(ev, markers, DATASET)
    asg.insert(0, "clustering", name)
    small = [c for c in t5.sort_clusters(clusters) if c not in big]
    if small:
        asg = pd.concat([asg, pd.DataFrame({"clustering": name, "cluster": small, "deg_label": BELOW_MIN,
                                            "confidence": BELOW_MIN})], ignore_index=True)
    order = {c: i for i, c in enumerate(t5.sort_clusters(clusters))}
    asg = asg.sort_values("cluster", key=lambda s: s.map(order)).reset_index(drop=True)
    return de, ev, asg


def per_cell(clusters: np.ndarray, table: pd.DataFrame, column: str) -> np.ndarray:
    return pd.Series(np.asarray(clusters).astype(str)).map(table.set_index("cluster")[column]).to_numpy()


def types_resolved(asg: pd.DataFrame, label_col: str = "deg_label", conf_col: str = "confidence") -> dict[str, str]:
    """Distinct types assigned to >= 1 annotated cluster -> best confidence over its clusters."""
    out: dict[str, str] = {}
    for label, conf in zip(asg[label_col], asg[conf_col]):
        if label in (UNKNOWN, BELOW_MIN):
            continue
        best = out.get(label)
        if best is None or CONFIDENCE_RANK.get(conf, 0) > CONFIDENCE_RANK.get(best, 0):
            out[label] = conf
    return {t: out[t] for t in cc.ordered_labels(out, LABEL_ORDER)}


def clustering_name(resolution: float, seed: int | None = None) -> str:
    return f"r{resolution:.1f}" if seed is None or seed == SEED else f"r{resolution:.1f}__s{seed}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def counts_text(values, order=LABEL_ORDER) -> str:
    counts = pd.Series(np.asarray(values).astype(str)).value_counts()
    return "; ".join(f"{k} {int(counts[k])}" for k in cc.ordered_labels(counts.index, order))


def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the labels_core step that writes it first.")
    return pd.read_csv(run.use_input(path, "labels_core output"), **kwargs)
