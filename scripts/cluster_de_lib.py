"""cluster_de shared definitions and functions. Runs no analysis by itself.

Imported by the
``cluster_de_*.py`` step scripts.

Contents
--------
- Clusterings: Leiden (igraph flavor, seed 42, as reviewer_core-preprocessing) on the stored
  neighbour graph of the kidney_object canonical object at low resolutions, the submitted
  resolution, and the stored submitted clusters.
- Differential expression: one cluster versus all other cells, two-sided
  Wilcoxon rank-sum (normal approximation, tie-corrected, no continuity
  correction; the same statistic as ``sc.tl.rank_genes_groups(method='wilcoxon',
  tie_correct=True)``) on ln(1 + intensity), missing = 0. Only proteins detected
  (intensity > 0) in >= 20% of the cluster's cells are tested; BH within cluster.
- Annotation rule (agreed with the authors on 13 Oct 2026), per cluster:
    1. top DE = up to 25 proteins with q < 0.05, AUC > 0.5 and a positive mean
       difference, ranked by AUC (then q, then mean difference);
    2. per cell type: hypergeometric test of its markers among the top DE, with
       the cluster's tested proteins as the background (BH over all cluster x type
       tests of one clustering), and the cluster mean of each cell's mean z
       (canonical ``.X``) over the type's markers present in the matrix;
    3. candidates = types with >= 1 marker in the top DE; none -> 'Unknown';
    4. best = lowest hypergeometric p (ties: higher mean z); candidate types whose
       overlap set equals the best type's cannot be told apart by the DE evidence,
       and the one with the highest mean z among them is assigned;
    5. confidence: 'high' = >= 1 type-specific marker in the overlap and q < 0.05;
       'medium' = >= 1 type-specific marker, q >= 0.05; 'low_shared_only' = only
       shared markers. A marker is shared if it is listed for >= 2 cell types in
       the dataset's own workbook (kidney V3, DCT V1).
  Information only (not used for the label or the confidence):
  ``discriminating_vs_runner_up`` (overlap markers absent from the runner-up
  type's list), agreement with the highest mean z, for DCT a flag for DCT1
  calls resting only on Umod and/or Slc12a3, and every marker's rank among the
  cluster's significant up proteins (beyond the top 25).
- Metrics: ARI, NMI (arithmetic), accuracy, cluster-majority accuracy,
  per-label precision / recall / F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc

TASK = "cluster_de"
SEED = cc.RANDOM_SEED

MIN_DETECTION = 0.20      # tested proteins: detected in >= 20% of the cluster's cells
TOP_N = 25                # top DE proteins per cluster
Q_DE = 0.05               # BH q threshold for a DE protein
Q_HYPER = 0.05            # BH q threshold for marker enrichment ('high' confidence)
STABILITY_SEEDS = tuple(range(10))
UNKNOWN = "Unknown"
CONFIDENCE_ORDER = ("high", "medium", "low_shared_only", "unknown_no_marker_in_top_de")

DATASETS = {
    "kidney": {"display": "KIDNEY", "label_columns": ("kidney_reannot_label", "kidney_pre_label"),
               "primary": "kidney_reannot_label", "order": cc.KIDNEY_LABEL_ORDER,
               "resolutions": (0.5, 0.8, 1.0, 2.0), "low_resolutions": (0.5, 0.8, 1.0),
               "submitted_resolution": 2.0, "stored_key": "leiden_res_2.00",
               "markers": cc.INPUTS["kidney"]["markers"], "marker_label": "kidney V3 marker workbook (Supp Table 1)"},
    "dct": {"display": "DCT", "label_columns": ("v1_cell_label", "v1_cluster_label"),
            "primary": "v1_cell_label", "order": cc.DCT_LABEL_ORDER,
            "resolutions": (0.3, 0.5, 0.8, 3.0), "low_resolutions": (0.3, 0.5, 0.8),
            "submitted_resolution": 3.0, "stored_key": "v1_leiden_res3_cluster",
            "markers": cc.INPUTS["dct"]["markers"], "marker_label": "DCT V1 marker workbook (Supp Table 1)"},
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
# DCT1 calls resting only on these markers are flagged (Umod is TAL-associated; Slc12a3 is shared).
DCT1_WEAK_MARKERS = {"Umod", "Slc12a3"}


def clustering_names(dataset: str) -> list[str]:
    spec = DATASETS[dataset]
    return [f"r{r:.1f}" for r in spec["resolutions"]] + [f"stored_r{spec['submitted_resolution']:.1f}"]


def clustering_display(name: str, dataset: str) -> str:
    if name.startswith("stored_"):
        return f"stored Leiden {DATASETS[dataset]['submitted_resolution']:.1f} (submitted)"
    return f"Leiden {name[1:]}"


def sort_clusters(values) -> list[str]:
    def key(v: str):
        return (0, int(v), "") if str(v).isdigit() else (1, 0, str(v))
    return sorted(set(map(str, values)), key=key)


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
@dataclass
class MarkerSets:
    """Workbook marker lists (short names), presence in the matrix and sharedness."""

    lists: dict[str, list[str]]           # type -> markers (workbook order, de-duplicated)
    present: dict[str, list[str]]         # type -> markers in the matrix
    full_names: dict[str, str]            # short -> workbook name
    types_by_gene: dict[str, set[str]]    # gene -> types listing it (workbook level)

    def is_shared(self, gene: str) -> bool:
        return len(self.types_by_gene.get(gene, set())) >= 2

    def specific_present(self, cell_type: str) -> list[str]:
        return [g for g in self.present[cell_type] if not self.is_shared(g)]


def read_markers(run: cc.Run, dataset: str, var_names) -> MarkerSets:
    path = run.use_input(DATASETS[dataset]["markers"], DATASETS[dataset]["marker_label"])
    raw = cc.read_marker_workbook(path)
    var = set(map(str, var_names))
    lists, present, full = {}, {}, {}
    for name, genes in raw.items():
        short = cc.KIDNEY_ABBREVIATIONS.get(name, name) if dataset == "kidney" else name
        genes = list(dict.fromkeys(genes))
        lists[short], full[short] = genes, name
        present[short] = [g for g in genes if g in var]
    types_by_gene: dict[str, set[str]] = {}
    for t, genes in lists.items():
        for g in genes:
            types_by_gene.setdefault(g, set()).add(t)
    return MarkerSets(lists, present, full, types_by_gene)


def marker_coverage(markers: MarkerSets, detection_rate: pd.Series, order) -> pd.DataFrame:
    rows = []
    for t in cc.ordered_labels(markers.lists, order):
        for g in markers.lists[t]:
            in_matrix = g in markers.present[t]
            rows.append({"cell_type": t, "cell_type_workbook": markers.full_names[t], "marker": g,
                         "in_matrix": in_matrix, "n_types_listing": len(markers.types_by_gene[g]),
                         "shared": markers.is_shared(g), "types_listing": ";".join(sorted(markers.types_by_gene[g])),
                         "detection_rate_all_cells": float(detection_rate[g]) if in_matrix else np.nan})
    frame = pd.DataFrame(rows)
    frame["specific_in_matrix"] = frame["in_matrix"] & ~frame["shared"]
    per_type = frame.groupby("cell_type", sort=False)
    frame["type_n_markers"] = per_type["marker"].transform("size")
    frame["type_n_present"] = per_type["in_matrix"].transform("sum")
    frame["type_n_specific_present"] = per_type["specific_in_matrix"].transform("sum")
    return frame


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def leiden(adata, resolution: float, key: str, seed: int = SEED) -> pd.Series:
    """Leiden on the object's stored graph, as in reviewer_core/qc_mito/preprocessing (igraph flavor)."""
    import scanpy as sc

    try:
        sc.tl.leiden(adata, key_added=key, resolution=resolution, flavor="igraph", random_state=seed)
    except TypeError:
        sc.tl.leiden(adata, key_added=key, resolution=resolution, random_state=seed)
    return adata.obs[key].astype(str)


# ---------------------------------------------------------------------------
# Differential expression
# ---------------------------------------------------------------------------
@dataclass
class RankCache:
    """Per-protein average ranks over all cells and the tie term (computed once per dataset)."""

    ln1p: np.ndarray
    detected: np.ndarray
    ranks: np.ndarray
    tie_term: np.ndarray   # sum over tie groups of t^3 - t, per protein

    @property
    def n(self) -> int:
        return self.ln1p.shape[0]


def rank_cache(ln1p: np.ndarray, detected: np.ndarray) -> RankCache:
    ln1p = np.asarray(ln1p, dtype=np.float64)
    ranks = stats.rankdata(ln1p, axis=0)
    tie = np.empty(ln1p.shape[1])
    for j in range(ln1p.shape[1]):
        _, counts = np.unique(ln1p[:, j], return_counts=True)
        counts = counts.astype(np.float64)
        tie[j] = float(np.sum(counts ** 3 - counts))
    return RankCache(ln1p, np.asarray(detected, bool), ranks, tie)


def wilcoxon(cache: RankCache, mask: np.ndarray, tie_correct: bool = True) -> dict[str, np.ndarray]:
    """Rank-sum statistic of the cells in ``mask`` against all others, for every protein."""
    n = cache.n
    n1 = int(mask.sum())
    n2 = n - n1
    u1 = cache.ranks[mask].sum(axis=0) - n1 * (n1 + 1) / 2.0
    tie = cache.tie_term / (n * (n - 1)) if tie_correct else 0.0
    var = n1 * n2 / 12.0 * ((n + 1) - tie)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, (u1 - n1 * n2 / 2.0) / np.sqrt(np.maximum(var, 0)), 0.0)
    return {"u": u1, "z": z, "p": 2.0 * stats.norm.sf(np.abs(z)), "auc": u1 / (n1 * n2)}


def de_one_vs_rest(cache: RankCache, clusters: np.ndarray, var_names, clustering: str,
                   top_n: int = TOP_N) -> pd.DataFrame:
    """Tested proteins (>= MIN_DETECTION in the cluster) for every cluster, with effect sizes and BH q.

    ``in_top`` marks the first ``top_n`` significant up proteins (TOP_N = 25 for the primary rule;
    other values only for the top-N sensitivity).
    """
    var_names = np.asarray(var_names).astype(str)
    frames = []
    for c in sort_clusters(clusters):
        mask = clusters == c
        pct_in = cache.detected[mask].mean(axis=0)
        pct_out = cache.detected[~mask].mean(axis=0)
        tested = pct_in >= MIN_DETECTION
        w = wilcoxon(cache, mask)
        mean_in = cache.ln1p[mask].mean(axis=0)
        mean_out = cache.ln1p[~mask].mean(axis=0)
        frame = pd.DataFrame({
            "clustering": clustering, "cluster": c, "n_cells_cluster": int(mask.sum()),
            "n_cells_rest": int((~mask).sum()), "protein": var_names,
            "pct_detected_cluster": pct_in, "pct_detected_rest": pct_out,
            "mean_ln1p_cluster": mean_in, "mean_ln1p_rest": mean_out, "mean_diff_ln1p": mean_in - mean_out,
            # scanpy rank_genes_groups 'logfoldchanges' on ln(1 + x) data.
            "log2fc_scanpy_style": np.log2((np.expm1(mean_in) + 1e-9) / (np.expm1(mean_out) + 1e-9)),
            "auc": w["auc"], "rank_biserial": 2 * w["auc"] - 1, "z_score": w["z"], "p_value": w["p"],
        })[tested]
        frame = frame.copy()
        frame["q_bh"] = stats.false_discovery_control(frame["p_value"].to_numpy(), method="bh") if len(frame) else []
        frame["up"] = (frame["auc"] > 0.5) & (frame["mean_diff_ln1p"] > 0)
        sig_up = frame[frame["up"] & (frame["q_bh"] < Q_DE)].sort_values(
            ["auc", "q_bh", "mean_diff_ln1p"], ascending=[False, True, False])
        frame["rank_up"] = np.nan
        frame.loc[sig_up.index, "rank_up"] = np.arange(1, len(sig_up) + 1)
        frame["in_top"] = frame["rank_up"] <= top_n
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
def marker_evidence(de: pd.DataFrame, markers: MarkerSets, x_z: pd.DataFrame, detected: pd.DataFrame,
                    clusters: np.ndarray, order) -> pd.DataFrame:
    """Cluster x cell-type evidence: hypergeometric enrichment in the top DE and mean marker z."""
    types = cc.ordered_labels(markers.lists, order)
    rows = []
    for c, group in de.groupby("cluster", sort=False):
        tested = set(group["protein"])
        top = list(group.loc[group["in_top"]].sort_values("rank_up")["protein"])   # overlap keeps the DE rank order
        rank_up = group.dropna(subset=["rank_up"]).set_index("protein")["rank_up"]
        mask = clusters == c
        for t in types:
            present = markers.present[t]
            k_markers = [g for g in present if g in tested]
            overlap = [g for g in top if g in present]
            specific = [g for g in overlap if not markers.is_shared(g)]
            n_universe, n_k, n_top, k = len(tested), len(k_markers), len(top), len(overlap)
            p = float(stats.hypergeom.sf(k - 1, n_universe, n_k, n_top)) if n_k and n_top else np.nan
            # Information only: every marker that is a significant up protein, with its rank (any rank, not only top N).
            ranked = sorted((int(rank_up[g]), g) for g in present if g in rank_up.index)
            rows.append({
                "cluster": c, "cell_type": t, "n_markers_workbook": len(markers.lists[t]),
                "n_markers_present": len(present), "n_markers_specific_present": len(markers.specific_present(t)),
                "universe_n_tested": n_universe, "markers_tested_K": n_k, "top_n": n_top, "overlap_k": k,
                "overlap": ";".join(overlap), "overlap_specific": ";".join(specific),
                "overlap_shared": ";".join(g for g in overlap if markers.is_shared(g)),
                "p_hypergeom": p if k else (1.0 if n_k and n_top else np.nan),
                "mean_marker_z": float(x_z.loc[mask, present].mean(axis=1).mean()) if present else np.nan,
                "marker_detected_fraction": float(detected.loc[mask, present].to_numpy().mean()) if present else np.nan,
                "is_candidate": k >= 1,
                "markers_sig_up_with_rank": ";".join(f"{g}({r})" for r, g in ranked),
                "best_marker_rank_up": ranked[0][0] if ranked else np.nan,
                "n_sig_up_cluster": int(len(rank_up)),
            })
    ev = pd.DataFrame(rows)
    ev["q_hypergeom"] = np.nan
    ok = np.isfinite(ev["p_hypergeom"].to_numpy(float))
    if ok.any():
        ev.loc[ok, "q_hypergeom"] = stats.false_discovery_control(ev.loc[ok, "p_hypergeom"].to_numpy(float), method="bh")
    return ev


def assign(ev: pd.DataFrame, markers: MarkerSets, dataset: str) -> pd.DataFrame:
    """One label per cluster from the evidence table (rule in the module docstring)."""
    rows = []
    for c, group in ev.groupby("cluster", sort=False):
        z_best = group.dropna(subset=["mean_marker_z"]).sort_values("mean_marker_z", ascending=False)
        z_best_type = str(z_best["cell_type"].iloc[0]) if len(z_best) else ""
        cands = group[group["is_candidate"]].sort_values(["p_hypergeom", "mean_marker_z"], ascending=[True, False])
        row = {"cluster": c, "mean_z_best_type": z_best_type,
               "mean_z_best_value": float(z_best["mean_marker_z"].iloc[0]) if len(z_best) else np.nan,
               "n_candidate_types": int(len(cands))}
        if cands.empty:
            row.update(deg_label=UNKNOWN, confidence="unknown_no_marker_in_top_de", overlap="", overlap_specific="",
                       p_hypergeom=np.nan, q_hypergeom=np.nan, mean_marker_z=np.nan, tied_types="", runner_up="",
                       discriminating_vs_runner_up="", agrees_with_mean_z_best=np.nan, dct1_umod_slc12a3_only=False)
            rows.append(row)
            continue
        first_overlap = set(cands["overlap"].iloc[0].split(";"))
        tie = cands[cands["overlap"].map(lambda s: set(s.split(";")) == first_overlap)]
        chosen = tie.sort_values(["mean_marker_z", "p_hypergeom"], ascending=[False, True]).iloc[0]
        label = str(chosen["cell_type"])
        others = cands[cands["cell_type"] != label]
        runner_up = str(others["cell_type"].iloc[0]) if len(others) else ""
        overlap = [g for g in chosen["overlap"].split(";") if g]
        specific = [g for g in chosen["overlap_specific"].split(";") if g]
        q = float(chosen["q_hypergeom"])
        if specific and q < Q_HYPER:
            confidence = "high"
        elif specific:
            confidence = "medium"
        else:
            confidence = "low_shared_only"
        runner_list = set(markers.lists.get(runner_up, []))
        row.update(deg_label=label, confidence=confidence, overlap=";".join(overlap), overlap_specific=";".join(specific),
                   p_hypergeom=float(chosen["p_hypergeom"]), q_hypergeom=q,
                   mean_marker_z=float(chosen["mean_marker_z"]),
                   tied_types=";".join(t for t in tie["cell_type"] if t != label), runner_up=runner_up,
                   discriminating_vs_runner_up=";".join(g for g in overlap if g not in runner_list) if runner_up else "",
                   agrees_with_mean_z_best=label == z_best_type,
                   dct1_umod_slc12a3_only=bool(dataset == "dct" and label == "DCT1" and set(overlap) <= DCT1_WEAK_MARKERS))
        rows.append(row)
    return pd.DataFrame(rows)


def annotate(cache: RankCache, clusters: np.ndarray, var_names, clustering: str, markers: MarkerSets,
             x_z: pd.DataFrame, detected: pd.DataFrame, dataset: str, top_n: int = TOP_N):
    """DE, evidence and assignment for one clustering. Returns (de, evidence, assignment)."""
    de = de_one_vs_rest(cache, clusters, var_names, clustering, top_n)
    ev = marker_evidence(de, markers, x_z, detected, clusters, DATASETS[dataset]["order"])
    ev.insert(0, "clustering", clustering)
    asg = assign(ev, markers, dataset)
    asg.insert(0, "clustering", clustering)
    return de, ev, asg


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def agreement(pred, truth) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pred = np.asarray(pred).astype(str)
    truth = np.asarray(truth).astype(str)
    return {"ari": float(adjusted_rand_score(truth, pred)), "nmi": float(normalized_mutual_info_score(truth, pred))}


def majority_accuracy(clusters, truth) -> float:
    frame = pd.DataFrame({"c": np.asarray(clusters).astype(str), "l": np.asarray(truth).astype(str)})
    majority = frame.groupby("c")["l"].agg(lambda s: s.value_counts().index[0])
    return float((frame["c"].map(majority) == frame["l"]).mean())


def recovery(pred, truth, order) -> pd.DataFrame:
    """Per-label precision, recall and F1 of the DEG labels (names matched) against one submitted column."""
    pred = np.asarray(pred).astype(str)
    truth = np.asarray(truth).astype(str)
    rows = []
    for label in cc.ordered_labels(set(pred) | set(truth), order):
        n_pred, n_true = int((pred == label).sum()), int((truth == label).sum())
        n_both = int(((pred == label) & (truth == label)).sum())
        precision = n_both / n_pred if n_pred else np.nan
        recall = n_both / n_true if n_true else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if n_pred and n_true and (precision + recall) > 0 else (0.0 if n_pred and n_true else np.nan))
        rows.append({"label": label, "n_submitted": n_true, "n_deg_label": n_pred, "n_both": n_both,
                     "precision": precision, "recall": recall, "f1": f1, "assigned_by_deg": n_pred > 0,
                     "in_submitted": n_true > 0, "is_unknown": label == UNKNOWN})
    return pd.DataFrame(rows)


def crosstab(rows, cols, row_order, col_order) -> pd.DataFrame:
    table = pd.crosstab(pd.Series(np.asarray(rows).astype(str), name="row"),
                        pd.Series(np.asarray(cols).astype(str), name="col"))
    return table.reindex(index=row_order, columns=col_order, fill_value=0)


# ---------------------------------------------------------------------------
# Helpers for the step scripts
# ---------------------------------------------------------------------------
def load(run: cc.Run, dataset: str):
    """kidney_object canonical object and labels (inputs recorded); adds the stored submitted clusters to labels."""
    for path, label in ((cc.CANONICAL[dataset]["labels"], f"kidney_object canonical {dataset} labels"),
                        (cc.CANONICAL[dataset]["h5ad"], f"kidney_object canonical {dataset} object")):
        run.use_input(path, label)
    adata, labels = cc.load_canonical(dataset)
    labels = labels.copy()
    key = DATASETS[dataset]["stored_key"]
    labels[f"stored_{key}"] = adata.obs[key].astype(str).to_numpy()
    return adata, labels


def table_path(name: str, subdir: str = "") -> Path:
    return cc.out_path(TASK, "tables", name, ".csv", subdir)


def read_table(run: cc.Run, name: str, subdir: str = "", **kwargs) -> pd.DataFrame:
    path = table_path(name, subdir)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the cluster_de step that writes it first.")
    return pd.read_csv(run.use_input(path, "cluster_de output"), **kwargs)


class Checks:
    def __init__(self, dataset: str):
        self.dataset, self.rows = dataset, []

    def add(self, check: str, observed, passed: bool | None, threshold: str = "") -> None:
        kind = "info" if passed is None else "check"
        self.rows.append({"dataset": self.dataset, "kind": kind, "check": check, "observed": observed,
                          "threshold": threshold, "pass": True if passed is None else bool(passed)})
        flag = "info" if passed is None else ("ok" if passed else "FAILED")
        print(f"  [{flag}] {check}: {observed}", flush=True)

    @property
    def ok(self) -> bool:
        return all(r["pass"] for r in self.rows)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
