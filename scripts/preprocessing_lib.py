"""preprocessing shared definitions and functions. Runs no analysis by itself.

Imported by the
``preprocessing_*.py`` step scripts.

Contents
--------
- The variant grid (33 pipelines per dataset; agreed with the authors on 13 Oct 2026):
  normalization {none, median, total} x scaling {z10, unscaled} x features
  {all, HVG 500 / 1,000 / 2,000} (24), the baseline with 10 / 30 / 50 PCs (3),
  the baseline with k = 5 / 15 / 30 neighbours (3), and proteins detected in
  >= 50% of cells for each normalization, z-scored (3; the reviewer_core ``highcov50`` rule).
  Every pipeline is clustered with Leiden at 0.5, 1.0, 2.0 and 3.0.
- Preprocessing: normalization -> HVG / feature selection -> scaling -> PCA ->
  kNN graph -> Leiden (-> UMAP), with the submitted pipelines' settings as the baseline.
- Metrics: ARI / NMI / cluster-majority accuracy against the submitted labels,
  kNN label purity with a permutation null (logic copied from reviewer_core_lib.knn_purity),
  silhouette by label on the PCA, Spearman rho of PCs with depth.
- Figure style re-used from qc_mito_lib (same palette and helpers as reviewer_core/qc_mito).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc

TASK = "preprocessing"
SEED = cc.RANDOM_SEED

# ---------------------------------------------------------------------------
# Datasets and labels
# ---------------------------------------------------------------------------
DATASETS = {
    "kidney": {"display": "KIDNEY", "label_columns": ("kidney_reannot_label", "kidney_pre_label"),
               "primary": ("kidney_reannot_label",), "order": cc.KIDNEY_LABEL_ORDER,
               "stored_leiden": "leiden_res_2.00", "submitted_resolution": 2.0},
    "dct": {"display": "DCT", "label_columns": ("v1_cell_label", "v1_cluster_label"),
            "primary": ("v1_cell_label", "v1_cluster_label"), "order": cc.DCT_LABEL_ORDER,
            "stored_leiden": "v1_leiden_res3_cluster", "submitted_resolution": 3.0},
}
LABEL_DISPLAY = {
    "kidney_reannot_label": "re-annotated label (Fig 2B)",
    "kidney_pre_label": "pre-annotation label (Fig 2A)",
    "v1_cell_label": "V1 per-cell label",
    "v1_cluster_label": "V1 cluster label",
}
# the project notes counts (checked before any analysis).
EXPECTED_COUNTS = {
    "kidney_reannot_label": {"PCT": 219, "PC": 187, "PST": 72, "DTL": 56, "CNT": 53, "TAL": 45, "DCT2": 27,
                             "DCT1": 18, "DVR": 16, "ATL": 7, "JG": 6, "IC-B": 3, "IC-A": 2, "MD": 1},
    "kidney_pre_label": {"PCT": 168, "TAL": 154, "PST": 132, "DVR": 88, "CNT": 73, "DCT2": 24, "JG": 22,
                         "MD": 19, "IC-B": 8, "Unknown": 24},
    "v1_cell_label": {"DCT1": 262, "DCT2": 235, "ProLIF": 253},
    "v1_cluster_label": {"DCT1": 367, "DCT2": 139, "ProLIF": 240, "Unknown": 4},
}
MIN_LABEL_N = 10   # labels with fewer cells are reported but left out of macro averages

# ---------------------------------------------------------------------------
# Variant grid
# ---------------------------------------------------------------------------
NORMALIZATIONS = {
    "none": "ln(1 + intensity); missing = 0 (submitted pipelines)",
    "median": ("per-cell median-centring of ln(intensity) over detected values: ln(x) - cell median + median of "
               "the cell medians, then one constant per dataset added to every detected value so that the smallest "
               "is 0.01 (authors decision, 13 Oct 2026); missing stays 0"),
    "total": "intensity x (median total / cell total), then ln(1 + x); totals over the retained proteins",
}
SCALINGS = {
    "z10": "per-protein z-score, clipped at +/-10 (sc.pp.scale(max_value=10); submitted pipelines)",
    "unscaled": "no scaling (PCA still centres each protein)",
}
FEATURES = {
    "all": "all retained proteins (submitted pipelines)",
    "hvg500": "500 highly variable proteins (scanpy flavor 'seurat')",
    "hvg1000": "1,000 highly variable proteins (scanpy flavor 'seurat')",
    "hvg2000": "2,000 highly variable proteins (scanpy flavor 'seurat')",
    "highcov50": "proteins detected in >= 50% of cells (reviewer_core highcov50 rule)",
}
HVG_SIZES = {"hvg500": 500, "hvg1000": 1000, "hvg2000": 2000}
HVG_FLAVOR = "seurat"
HIGHCOV_MIN_DETECTION = 0.50
MEDIAN_MIN_DETECTED = 0.01   # smallest detected value after median-centring (uniform shift; authors, 13 Oct 2026)
RESOLUTIONS = (0.5, 1.0, 2.0, 3.0)
BASE_N_PCS = 20
BASE_K = 10
N_COMPS = 50
PURITY_K = 10
N_PERMUTATIONS = 1000
N_DEPTH_PCS = 5


@dataclass(frozen=True)
class Variant:
    variant_id: str
    block: str
    normalization: str
    scaling: str
    features: str
    n_pcs: int = BASE_N_PCS
    k: int = BASE_K

    @property
    def is_baseline(self) -> bool:
        return self.variant_id == BASELINE_ID

    @property
    def embedding_key(self) -> str:
        """Pipelines sharing this key have the same PCA (only n_pcs or k differ)."""
        return f"{self.normalization}__{self.scaling}__{self.features}"

    def as_row(self) -> dict:
        return {"variant_id": self.variant_id, "block": self.block, "normalization": self.normalization,
                "scaling": self.scaling, "features": self.features, "n_pcs": self.n_pcs, "k_neighbors": self.k,
                "is_baseline": self.is_baseline, "embedding_key": self.embedding_key}


BASELINE_ID = "none__z10__all"


def variant_grid() -> list[Variant]:
    grid: list[Variant] = []
    for norm in ("none", "median", "total"):
        for scaling in ("z10", "unscaled"):
            for feat in ("all", "hvg500", "hvg1000", "hvg2000"):
                grid.append(Variant(f"{norm}__{scaling}__{feat}", "core_factorial", norm, scaling, feat))
    for n_pcs in (10, 30, 50):
        grid.append(Variant(f"{BASELINE_ID}__pcs{n_pcs}", "pca_test", "none", "z10", "all", n_pcs=n_pcs))
    for k in (5, 15, 30):
        grid.append(Variant(f"{BASELINE_ID}__k{k}", "knn_test", "none", "z10", "all", k=k))
    for norm in ("none", "median", "total"):
        grid.append(Variant(f"{norm}__z10__highcov50", "highcov_features", norm, "z10", "highcov50"))
    ids = [v.variant_id for v in grid]
    assert len(ids) == len(set(ids)) == 33 and ids[0] == BASELINE_ID
    return grid


def grid_table() -> pd.DataFrame:
    rows = []
    for v in variant_grid():
        row = v.as_row()
        row.update(normalization_definition=NORMALIZATIONS[v.normalization],
                   scaling_definition=SCALINGS[v.scaling], features_definition=FEATURES[v.features],
                   leiden_resolutions=";".join(map(str, RESOLUTIONS)))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def normalize(intensity: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    """Normalized log matrix (cells x proteins, float64) and per-method diagnostics."""
    x = np.asarray(intensity, dtype=np.float64)
    detected = x > 0
    info: dict = {"method": method}
    if method == "none":
        return np.log1p(x), info
    if method == "median":
        ln = np.full(x.shape, np.nan)
        ln[detected] = np.log(x[detected])
        cell_median = np.nanmedian(ln, axis=1)
        target = float(np.median(cell_median))
        centred = ln - cell_median[:, None] + target
        min_centred = float(np.nanmin(centred))
        n_nonpositive = int((centred[detected] <= 0).sum())
        # Uniform shift (authors decision, 13 Oct 2026): a few small detected values fall to <= 0 after centring and
        # would merge with missing = 0; one constant keeps them positive without changing any rank among them.
        shift = max(0.0, MEDIAN_MIN_DETECTED - min_centred)
        out = np.where(detected, centred + shift, 0.0)
        info.update(target_ln=target, cell_median_min=float(cell_median.min()),
                    cell_median_max=float(cell_median.max()), min_detected_centred=min_centred,
                    n_detected_nonpositive_centred=n_nonpositive,
                    n_cells_with_nonpositive_centred=int((detected & (np.nan_to_num(centred, nan=1.0) <= 0)).any(axis=1).sum()),
                    uniform_shift=shift, min_detected_after=float(out[detected].min()),
                    n_detected_nonpositive_after=int((out[detected] <= 0).sum()))
        return out, info
    if method == "total":
        total = x.sum(axis=1)
        factor = np.median(total) / total
        info.update(median_total=float(np.median(total)), factor_min=float(factor.min()),
                    factor_max=float(factor.max()))
        return np.log1p(x * factor[:, None]), info
    raise ValueError(f"unknown normalization {method!r}")


def detection_rate(intensity: np.ndarray) -> np.ndarray:
    return (np.asarray(intensity) > 0).mean(axis=0)


def hvg_mask(norm_log: np.ndarray, n_top: int, var_names) -> tuple[np.ndarray, pd.DataFrame]:
    """scanpy highly_variable_genes(flavor='seurat', n_top_genes) on the normalized log matrix."""
    import anndata as ad
    import scanpy as sc

    g = ad.AnnData(X=np.asarray(norm_log, dtype=np.float32), var=pd.DataFrame(index=pd.Index(list(var_names))))
    sc.pp.highly_variable_genes(g, flavor=HVG_FLAVOR, n_top_genes=n_top, inplace=True)
    var = g.var[["means", "dispersions", "dispersions_norm", "highly_variable"]].copy()
    return var["highly_variable"].to_numpy(bool), var


def feature_mask(variant: Variant, norm_log: np.ndarray, det_rate: np.ndarray, var_names,
                 cache: dict) -> np.ndarray:
    key = (variant.normalization, variant.features)
    if key not in cache:
        if variant.features == "all":
            cache[key] = (np.ones(len(var_names), bool), None)
        elif variant.features == "highcov50":
            cache[key] = (det_rate >= HIGHCOV_MIN_DETECTION, None)
        else:
            cache[key] = hvg_mask(norm_log, HVG_SIZES[variant.features], var_names)
    return cache[key][0]


def build_embedding(norm_log: np.ndarray, mask: np.ndarray, scaling: str, cell_ids, var_names):
    """AnnData with the selected proteins, scaled or not, and an arpack PCA (50 components, seed 42)."""
    import anndata as ad
    import scanpy as sc

    x = norm_log[:, mask]
    if scaling == "z10":
        x = cc.scale_like_scanpy(x, max_value=10.0)
    elif scaling != "unscaled":
        raise ValueError(f"unknown scaling {scaling!r}")
    g = ad.AnnData(X=np.asarray(x, dtype=np.float32), obs=pd.DataFrame(index=pd.Index(list(cell_ids), name="cell_id")),
                   var=pd.DataFrame(index=pd.Index(list(np.asarray(var_names)[mask]))))
    n_comps = int(min(N_COMPS, g.n_obs - 1, g.n_vars - 1))
    sc.pp.pca(g, n_comps=n_comps, svd_solver="arpack", random_state=SEED)
    return g


def build_graph(g, n_pcs: int, k: int):
    import scanpy as sc

    h = g.copy()
    sc.pp.neighbors(h, n_neighbors=k, n_pcs=n_pcs, random_state=SEED)
    return h


def leiden(g, resolution: float, key: str) -> pd.Series:
    """Leiden as in reviewer_core/qc_mito (igraph flavor, seed 42)."""
    import scanpy as sc

    try:
        sc.tl.leiden(g, key_added=key, resolution=resolution, flavor="igraph", random_state=SEED)
    except TypeError:
        sc.tl.leiden(g, key_added=key, resolution=resolution, random_state=SEED)
    return g.obs[key].astype(str)


def umap(g) -> np.ndarray:
    import scanpy as sc

    h = g.copy()
    sc.tl.umap(h, random_state=SEED)
    return np.asarray(h.obsm["X_umap"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def label_codes(values, dataset: str) -> tuple[np.ndarray, list[str]]:
    cats = cc.ordered_labels(values, DATASETS[dataset]["order"])
    codes = pd.Categorical(np.asarray(values).astype(str), categories=cats).codes.astype(int)
    return codes, cats


def cluster_agreement(clusters, labels) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    clusters = np.asarray(clusters).astype(str)
    labels = np.asarray(labels).astype(str)
    frame = pd.DataFrame({"c": clusters, "l": labels})
    majority = frame.groupby("c")["l"].agg(lambda s: s.value_counts().index[0])
    return {"ari": float(adjusted_rand_score(labels, clusters)),
            "nmi": float(normalized_mutual_info_score(labels, clusters)),
            "majority_accuracy": float((frame["c"].map(majority) == frame["l"]).mean())}


def neighbor_index(embedding: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbours (Euclidean, self excluded); copied from reviewer_core_lib.neighbor_index."""
    from sklearn.neighbors import NearestNeighbors

    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(embedding)
    return np.vstack([row[row != i][:k] for i, row in enumerate(idx)])


def _purity(idx: np.ndarray, codes: np.ndarray, n_labels: int) -> tuple[np.ndarray, float]:
    per_cell = (codes[idx] == codes[:, None]).mean(axis=1)
    return (np.bincount(codes, weights=per_cell, minlength=n_labels) / np.bincount(codes, minlength=n_labels),
            float(per_cell.mean()))


def knn_purity(idx: np.ndarray, codes: np.ndarray, cats: list[str], n_perm: int = N_PERMUTATIONS) -> pd.DataFrame:
    """Mean same-label fraction of the k nearest neighbours against label permutations.

    Same logic as reviewer_core_lib.knn_purity: the RNG is re-seeded (42) for every
    embedding x label column. ``expected`` = (n_label - 1) / (n - 1), the exact mean
    of the permutation null for a label.
    """
    observed, observed_all = _purity(idx, codes, len(cats))
    rng = np.random.default_rng(SEED)
    null = np.empty((n_perm, len(cats)))
    null_all = np.empty(n_perm)
    for b in range(n_perm):
        null[b], null_all[b] = _purity(idx, rng.permutation(codes), len(cats))
    n = len(codes)
    counts = np.bincount(codes, minlength=len(cats))
    rows = []
    for j, label in enumerate([*cats, "all_cells"]):
        is_all = label == "all_cells"
        obs_v = observed_all if is_all else observed[j]
        nul = null_all if is_all else null[:, j]
        n_lab = n if is_all else int(counts[j])
        expected = float(np.sum(counts * (counts - 1)) / (n * (n - 1))) if is_all else (n_lab - 1) / (n - 1)
        rows.append({"label": label, "n_cells": n_lab, "observed_same_label_fraction": float(obs_v),
                     "expected_random": expected,
                     "enrichment": float(obs_v / expected) if expected > 0 else np.nan,
                     "null_mean": float(nul.mean()), "null_q975": float(np.quantile(nul, 0.975)),
                     "p_empirical_ge": float((1 + (nul >= obs_v).sum()) / (n_perm + 1))})
    return pd.DataFrame(rows)


def silhouette(embedding: np.ndarray, codes: np.ndarray, cats: list[str]) -> pd.DataFrame:
    """Silhouette (Euclidean) per cell, averaged per label and over all cells (sklearn: singletons get 0)."""
    from sklearn.metrics import silhouette_samples

    s = silhouette_samples(embedding, codes, metric="euclidean")
    rows = [{"label": label, "n_cells": int((codes == j).sum()),
             "mean_silhouette": float(s[codes == j].mean()) if (codes == j).any() else np.nan}
            for j, label in enumerate(cats)]
    rows.append({"label": "all_cells", "n_cells": len(codes), "mean_silhouette": float(s.mean())})
    return pd.DataFrame(rows)


def pc_depth(pcs: np.ndarray, ln_ids: np.ndarray, ln_total: np.ndarray, variance_ratio: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(min(N_DEPTH_PCS, pcs.shape[1])):
        r_ids, p_ids = stats.spearmanr(pcs[:, i], ln_ids)
        r_tot, p_tot = stats.spearmanr(pcs[:, i], ln_total)
        rows.append({"pc": i + 1, "variance_ratio": float(variance_ratio[i]),
                     "spearman_rho_ln_n_detected": float(r_ids), "p_ln_n_detected": float(p_ids),
                     "spearman_rho_ln_total_intensity": float(r_tot), "p_ln_total_intensity": float(p_tot)})
    return pd.DataFrame(rows)


def macro(frame: pd.DataFrame, value: str) -> float:
    """Mean of ``value`` over labels with >= MIN_LABEL_N cells (the 'all_cells' row excluded)."""
    sub = frame[(frame["label"] != "all_cells") & (frame["n_cells"] >= MIN_LABEL_N)]
    return float(sub[value].mean()) if len(sub) else np.nan


# ---------------------------------------------------------------------------
# Helpers for the step scripts
# ---------------------------------------------------------------------------
def load(run: cc.Run, dataset: str):
    """kidney_object canonical object and labels (inputs recorded)."""
    for path, label in ((cc.CANONICAL[dataset]["labels"], f"kidney_object canonical {dataset} labels"),
                        (cc.CANONICAL[dataset]["h5ad"], f"kidney_object canonical {dataset} object")):
        run.use_input(path, label)
    return cc.load_canonical(dataset)


def label_colors(adata, column: str) -> dict[str, str]:
    obs = adata.obs[column]
    return dict(zip(map(str, obs.cat.categories), adata.uns[f"{column}_colors"]))


def tbl(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(cc.out_path(TASK, "tables", name, ".csv"), **kwargs)
