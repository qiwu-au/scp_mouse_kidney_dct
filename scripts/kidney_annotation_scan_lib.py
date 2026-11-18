"""kidney_annotation step 2 shared definitions and functions. Runs no analysis by itself.

Imported by the kidney_annotation step-2 scripts
(``kidney_annotation_graphs.py``, ``kidney_annotation_resolution_scan.py``, ``kidney_annotation_rare_candidates.py``,
``kidney_annotation_scan_figures.py``, ``kidney_annotation_chosen_labels.py``). Plan:
``the analysis plan``.

Settings agreed with the authors (14 Oct 2026)
------------------------------------------
- Graphs (k = 10, 20 PCs, PCA arpack, seed 42):
    (a) ``stored``: the stored neighbour graph of the kidney_object canonical kidney object (as cluster_de);
    (b) ``regressed``: ln(protein IDs) regressed out of ln(1 + intensity) per protein (OLS with intercept,
        missing = 0), then z-score clipped at 10, PCA, kNN;
    (c) ``t04_total_unscaled_hvg500``: the preprocessing kidney pipeline with the lowest max |rho| of PC1/PC2 with
        ln(protein IDs) (0.654), rebuilt with the preprocessing code.
- Leiden (igraph flavour, as reviewer_core-cluster_de) at 0.5-6.0 in steps of 0.5, seeds 42 and 0-9.
- Every clustering (all seeds) is annotated with the cluster_de rule (``cluster_de_lib``: top-25 DE,
  hypergeometric marker enrichment, shared-marker confidence, Unknown without evidence).
- Clusters with fewer than 5 cells are reported but not annotated (label ``below_min_size``); their
  cells stay in the 'rest' of the other clusters' one-vs-rest DE.

The DE matrix is the canonical ln(1 + intensity) for every graph (cluster_de rule); the graph changes only
the cluster membership.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

import scp_common as cc
import cluster_de_lib as t5

TASK = "kidney_annotation"
SEED = cc.RANDOM_SEED
DATASET = "kidney"

RESOLUTIONS = tuple(round(0.5 * i, 1) for i in range(1, 13))       # 0.5 ... 6.0
SEEDS = (SEED, *t5.STABILITY_SEEDS)                                  # 42, then 0-9
MIN_CLUSTER_SIZE = 5                                                 # authors, 14 Oct 2026
BELOW_MIN = "below_min_size"
UNKNOWN = t5.UNKNOWN
N_PCS = 20
K_NEIGHBORS = 10
N_DEPTH_PCS = 5
PREPROCESSING_VARIANT = "total__unscaled__hvg500"
LABEL_COLUMNS = ("kidney_reannot_label", "kidney_pre_label")
PRIMARY = "kidney_reannot_label"
LABEL_DISPLAY = t5.LABEL_DISPLAY
ORDER = cc.KIDNEY_LABEL_ORDER
KIDNEY_ANNOTATION_ORDER = (*[t for t in cc.KIDNEY_LABEL_ORDER if t != UNKNOWN], UNKNOWN, BELOW_MIN)
# Rank-free sensitivity (authors, 14 Oct 2026): a type-specific marker significantly up at any rank.
RESCUED = "specific_any_rank"
CONFIDENCE_RANK = {"high": 3, "medium": 2, RESCUED: 1.5, "low_shared_only": 1}
SPECIFIC_CONFIDENCES = ("high", "medium", RESCUED)
# V3 + endothelial row (authors, 14 Oct 2026): atlas Endo / AVR markers proposed after the step-3c detection check.
ENDO_TYPE = "Endothelial"
ENDOTHELIAL_MARKERS = ("Pecam1", "Emcn", "Eng", "Esam", "Icam2", "Fabp4", "Ehd4")

GRAPHS = {
    "stored": {
        "short": "a", "display": "(a) stored graph",
        "definition": ("stored neighbour graph of the kidney_object canonical object (k = 10, 20 PCs of the per-protein "
                       "z-scored ln(1 + intensity), clip 10; submitted pipeline)"),
    },
    "regressed": {
        "short": "b", "display": "(b) depth-regressed",
        "definition": ("per protein, ln(1 + intensity) (missing = 0) regressed on ln(protein IDs) with an intercept "
                       "(OLS); residuals z-scored per protein and clipped at +/-10; PCA arpack (50 components, seed "
                       "42); kNN on the first 20 PCs, k = 10"),
    },
    "t04_total_unscaled_hvg500": {
        "short": "c", "display": "(c) preprocessing total · unscaled · hvg500",
        "definition": ("preprocessing pipeline total__unscaled__hvg500: intensity x (median total / cell total), ln(1 + x); "
                       "500 HVGs (scanpy flavor 'seurat'); no scaling; PCA arpack (50 components, seed 42); kNN on "
                       "the first 20 PCs, k = 10 (rebuilt with preprocessing_lib)"),
    },
}
GRAPH_NAMES = tuple(GRAPHS)

# Plan, 'Facts already established': quantified markers listed for only one type (rare-type flags).
BRIEF_TYPE_SPECIFIC = {
    "JG": ("Akr1b7",), "IC-A": ("Slc4a1", "Adgrf5", "Mme"), "IC-B": ("Slc26a4",), "ATL": ("Clcnka",),
    "DCT1": ("Pvalb",), "PC": ("Rhbg", "Aqp3", "Kcnj10"), "Podo": ("Synpo",),
}
BRIEF_NO_SPECIFIC = ("DVR", "DTL", "MD", "DCT2", "CNT")
RARE_TYPES = ("JG", "ATL", "DCT1", "PC", "IC-A", "IC-B", "Podo")   # flagged types (Fig 2B sizes <= 18 or PC)
ALSO_FLAGGED = ("PCT", "PST", "TAL")                                # plan: 'plus PCT, PST and TAL markers'

GRAPHS_H5AD = cc.out_path(TASK, "h5ad", "kidney_scan_graphs", ".h5ad")
PREPROCESSING_PC_DEPTH = cc.OUT_ROOTS["tables"] / "preprocessing" / "preprocessing_kidney_pc_depth.csv"
PREPROCESSING_CLUSTERS = cc.OUT_ROOTS["tables"] / "preprocessing" / "preprocessing_kidney_per_cell_clusters.csv"
CLUSTER_DE_CLUSTERS = cc.OUT_ROOTS["tables"] / "cluster_de" / "cluster_de_kidney_clusters_per_cell.csv"


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def clustering_name(graph: str, resolution: float, seed: int | None = None) -> str:
    base = f"{graph}__r{resolution:.1f}"
    return base if seed is None or seed == SEED else f"{base}__s{seed}"


def graph_display(graph: str) -> str:
    return GRAPHS[graph]["display"]


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------
def depth_covariates(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """ln(protein IDs) and ln(total intensity) per cell (canonical columns)."""
    return (np.log(labels["n_detected_proteins"].to_numpy(float)),
            np.log(labels["total_intensity"].to_numpy(float)))


def regress_out(y: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Residuals of each column of ``y`` after OLS on ``covariate`` with an intercept."""
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([np.ones(len(covariate)), np.asarray(covariate, dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def eta_squared(values, groups) -> float:
    """Between-group sum of squares / total sum of squares (one-way ANOVA eta^2)."""
    v = np.asarray(values, dtype=np.float64)
    g = pd.Series(np.asarray(groups).astype(str))
    means = pd.Series(v).groupby(g).transform("mean").to_numpy()
    ss_total = float(np.sum((v - v.mean()) ** 2))
    return float(np.sum((means - v.mean()) ** 2) / ss_total) if ss_total > 0 else np.nan


def kruskal_p(values, groups) -> float:
    v = np.asarray(values, dtype=np.float64)
    g = np.asarray(groups).astype(str)
    samples = [v[g == c] for c in t5.sort_clusters(g)]
    return float(stats.kruskal(*samples).pvalue) if len(samples) > 1 else np.nan


def pc_depth(pcs: np.ndarray, ln_ids: np.ndarray, ln_total: np.ndarray, variance_ratio) -> pd.DataFrame:
    """Spearman rho of PC1-PC5 with ln(protein IDs) and ln(total intensity) (preprocessing definition)."""
    import preprocessing_lib as t4

    return t4.pc_depth(pcs[:, :N_DEPTH_PCS], ln_ids, ln_total, np.asarray(variance_ratio)[:N_DEPTH_PCS])


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def graph_adata(store, graph: str):
    """Small AnnData with one stored graph as ``obsp['connectivities']`` (for ``cluster_de_lib.leiden``)."""
    import anndata as ad

    g = ad.AnnData(X=sparse.csr_matrix((store.n_obs, 1), dtype=np.float32),
                   obs=pd.DataFrame(index=store.obs_names.copy()))
    g.obsp["connectivities"] = store.obsp[f"connectivities__{graph}"]
    g.obsp["distances"] = store.obsp[f"distances__{graph}"]
    g.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                          "params": {"n_neighbors": K_NEIGHBORS, "n_pcs": N_PCS, "method": "umap",
                                     "metric": "euclidean"}}
    return g


def load_graphs(run: cc.Run):
    import anndata as ad

    path = run.use_input(GRAPHS_H5AD, "kidney_annotation kidney scan graphs (kidney_annotation_graphs.py)")
    return ad.read_h5ad(path)


# ---------------------------------------------------------------------------
# Annotation (cluster_de rule with a minimum cluster size)
# ---------------------------------------------------------------------------
def annotate_clustering(cache: t5.RankCache, clusters: np.ndarray, var_names, name: str, markers: t5.MarkerSets,
                        x_z: pd.DataFrame, detected: pd.DataFrame, min_size: int = MIN_CLUSTER_SIZE,
                        de: pd.DataFrame | None = None):
    """cluster_de DE, evidence and assignment for clusters with >= ``min_size`` cells.

    Smaller clusters get the label and confidence ``below_min_size``; their cells remain in the 'rest' of every
    other cluster's one-vs-rest test. BH over cluster x type tests covers the annotated clusters only.
    ``de``: a DE table from an earlier call on the same clustering (reused, e.g. for another marker set).
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


def per_cell_labels(clusters: np.ndarray, asg: pd.DataFrame, column: str = "deg_label") -> np.ndarray:
    return pd.Series(np.asarray(clusters).astype(str)).map(asg.set_index("cluster")[column]).to_numpy()


def types_resolved(asg: pd.DataFrame) -> dict[str, str]:
    """Distinct types assigned to >= 1 annotated cluster -> best confidence over its clusters."""
    out: dict[str, str] = {}
    for row in asg.itertuples(index=False):
        if row.deg_label in (UNKNOWN, BELOW_MIN):
            continue
        best = out.get(row.deg_label)
        if best is None or CONFIDENCE_RANK.get(row.confidence, 0) > CONFIDENCE_RANK.get(best, 0):
            out[row.deg_label] = row.confidence
    return {t: out[t] for t in cc.ordered_labels(out, ORDER)}


def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the kidney_annotation step that writes it first.")
    return pd.read_csv(run.use_input(path, "kidney_annotation step-2 output"), **kwargs)
