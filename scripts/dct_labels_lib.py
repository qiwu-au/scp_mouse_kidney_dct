"""dct_labels shared definitions and functions. Runs no analysis by itself.

Imported by the
``dct_labels_*.py`` step scripts. Plan: ``the analysis plan``
(frozen; SHA256 in ``results/tables/dct_intake_zerok_vs_control_frozen_briefs_sha256.csv``), with the
rules in ``the analysis plan``.

Frozen settings (labels_core, zerok_core; plan)
---------------------------------
- Data: the dct_intake objects of the combined search, each condition filtered and scaled on
  its own (Ctrl = the combined search without the ZeroK runs; authors, 2 Nov 2026).
- Submitted pipeline per condition, unchanged (labels_core step 1): float32 intensities,
  ``sc.pp.log1p``, ``sc.pp.scale(max_value=10)``, ``sc.pp.neighbors(n_neighbors=10,
  n_pcs=20)`` with Scanpy's PCA fallback; Leiden 3.0 (igraph, seed 42);
  ``sc.tl.score_genes`` per DCT V1 workbook type on the markers present, on the scaled
  ``.X``; per-cell argmax; cluster mode with the original Unknown rule; arpack PCA;
  UMAP (seed 42); DPT from the first DCT1 cell in object order (P1 then P2, ascending
  well) and from 20 random DCT1 roots; PAGA on the per-cell label.
- Primary rule per condition (labels_core step 2b = zerok_core route b): cluster_de DE rule with the kidney_annotation
  rank-free rescue on Leiden 1.0 (igraph, seed 42) of the same graph; Slc12a3 and Umod
  shared; minimum cluster size 5. Ctrl -> ``cs_ctrl_deg_*``; ZeroK -> ``cs_zerok_pipeline_*``.
- ZeroK alternative (zerok_core route a): majority ``cs_ctrl_deg_label`` of the 10 nearest Ctrl
  cells in the dct_intake joint 20-PC embedding, per normalization variant; ties to the label
  of the nearest tied neighbour (C2). ``cs_zerok_transfer_label`` = variant ``none``.
- Markers: DCT V1 workbook markers present in each condition's matrix.

The labelling code is the labels_core/zerok_core code, imported. Functions copied into this module
take the cell count or the label-column names as parameters where the labels_core originals
hard-code the 727 control cells or the ``ctrl_`` column names; their logic is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import labels_core_lib as t18
import dct_intake_lib as t26
from kidney_annotation_rankfree_sensitivity import rank_free
from labels_core_resolution_scan import rule_tables

TASK = "dct_labels"
SEED = cc.RANDOM_SEED
CONDITIONS = ("Ctrl", "ZeroK")
KEY = {"Ctrl": "ctrl", "ZeroK": "zerok"}
PREFIX = {"Ctrl": "cs_ctrl", "ZeroK": "cs_zerok"}
PRIMARY_STEM = {"Ctrl": "deg", "ZeroK": "pipeline"}     # cs_ctrl_deg_*, cs_zerok_pipeline_* (plan)
RESOLUTION = 1.0
RULE = "rankfree"
TRANSFER_K = 10
VARIANTS = ("none", "cell_median", "condition_align")   # dct_intake joint embeddings (zerok_core definitions)
BASELINE_VARIANT = "none"
LABEL_ORDER = t18.LABEL_ORDER
CORE_TYPES = t18.CORE_TYPES
UNKNOWN = t18.UNKNOWN
LABEL_COLORS = t18.LABEL_COLORS
counts_text = t18.counts_text

PLANS = ()   # no planning document is part of the public code
FREEZE_TABLE = t26.FREEZE_TABLE
OBJECTS = t26.OBJECTS

# Bridge only (plan step 7)
LABELS_CORE_CANONICAL = cc.out_path("labels_core", "tables", "dct_ctrl_canonical_labels", ".csv")
LABELS_CORE_MARKER_ORDER = cc.out_path("labels_core", "tables", "marker_order_dct", ".csv")
ZEROK_CORE_LABELS = cc.out_path("zerok_core", "tables", "zerok_labels", ".csv")

STEP1_H5AD = {c: cc.out_path(TASK, "h5ad", f"DCT_{KEY[c]}_step1_submitted", ".h5ad") for c in CONDITIONS}
LABELLED_H5AD = {**{c: cc.out_path(TASK, "h5ad", f"DCT_{KEY[c]}_labelled", ".h5ad") for c in CONDITIONS},
                 "joint": cc.out_path(TASK, "h5ad", "DCT_joint_labelled", ".h5ad")}

# Generic per-cell column names in the step tables (one row per cell, both conditions, with `condition`)
# and their canonical names (plan step 8).
GENERIC_TO_CS = {
    "cell_label": "cell_label", "cluster_label": "cluster_label", "leiden3_cluster": "leiden3_cluster",
    "deg_label": "{stem}_label", "deg_cluster": "{stem}_cluster", "deg_confidence": "{stem}_confidence",
}


def cs(cond: str, generic: str) -> str:
    """Canonical column name of a generic step-table column for one condition (e.g. cs_zerok_pipeline_label)."""
    return f"{PREFIX[cond]}_{GENERIC_TO_CS[generic].format(stem=PRIMARY_STEM[cond])}"


def label_columns(cond: str) -> list[str]:
    """Canonical label columns of a condition, primary last for Ctrl; ZeroK adds the transfer label."""
    cols = [cs(cond, "cell_label"), cs(cond, "cluster_label"), cs(cond, "deg_label")]
    return cols + (["cs_zerok_transfer_label"] if cond == "ZeroK" else [])


def primary_column(cond: str) -> str:
    return cs(cond, "deg_label")


LABEL_DISPLAY = {
    "cell_label": "Per-cell label (submitted pipeline)",
    "cluster_label": "Cluster label (submitted pipeline, Leiden 3.0)",
    "deg_label": "DE-rule label (rank-free, Leiden 1.0)",
    "transfer_label": "Transfer label (10 nearest Ctrl cells)",
}


def display(column: str) -> str:
    """Human label of a canonical column (cs_ctrl_deg_label -> DE-rule label ...)."""
    for generic, text in (("transfer_label", LABEL_DISPLAY["transfer_label"]),
                          ("cell_label", LABEL_DISPLAY["cell_label"]),
                          ("cluster_label", LABEL_DISPLAY["cluster_label"])):
        if column.endswith(generic):
            return text
    return LABEL_DISPLAY["deg_label"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def read_t26(run: cc.Run, cond: str):
    return t26.read_object(run, cond)


def read_step1(run: cc.Run, cond: str):
    import anndata as ad

    path = STEP1_H5AD[cond]
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run dct_labels_submitted_pipeline.py first.")
    return ad.read_h5ad(run.use_input(path, f"dct_labels step 1 object ({cond})"))


def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the dct_labels step that writes it first.")
    return pd.read_csv(run.use_input(path, "dct_labels output"), **kwargs)


def per_condition(frame: pd.DataFrame, cond: str, cell_ids) -> pd.DataFrame:
    """Rows of one condition of a long step table, indexed by cell_id, in the given cell order."""
    sub = frame[frame["condition"] == cond].set_index("cell_id")
    if list(sub.index) != list(map(str, cell_ids)):
        raise ValueError(f"{cond}: step table cells differ from the object (order or membership)")
    return sub


def brief_hash_rows() -> list[dict]:
    """SHA256 of the two dct_labels plans against the committed freeze record."""
    frozen = pd.read_csv(FREEZE_TABLE).set_index("path")
    rows = []
    for path in PLANS:
        observed = cc.sha256(cc.PROJECT_ROOT / path)
        rows.append({"path": path, "sha256_observed": observed, "sha256_frozen": frozen.loc[path, "sha256"],
                     "pass": observed == frozen.loc[path, "sha256"]})
    return rows


def cell_key_from_id(cell_id: str) -> str:
    p = cc.parse_dct_cell_id(cell_id)
    return t26.cell_key(p["condition"], p["plate"], p["well"])


# ---------------------------------------------------------------------------
# Submitted pipeline (labels_core step 1 sequence, unchanged)
# ---------------------------------------------------------------------------
def submitted_pipeline(src, workbook: dict[str, list[str]], run: cc.Run) -> dict:
    """labels_core step 1 on one condition's object (layers['intensity'], obs plate/well/n_detected/total_intensity).

    The call sequence is the labels_core one, including the information-only scoring of the pre-log intensities
    (score_genes reseeds NumPy's global generator, so the sequence is kept as labels_core ran it).
    """
    import scanpy as sc

    from labels_core_submitted_pipeline import dpt_roots, paga_edges

    intensity = np.array(cc.dense(src.layers["intensity"]), dtype=np.float32, copy=True)
    adata = t18.submitted_object(src)
    sc.pp.neighbors(adata, n_neighbors=t18.N_NEIGHBORS, n_pcs=t18.N_PCS)
    pca_neighbors = np.asarray(adata.obsm["X_pca"]).copy()
    pca_params = {k: str(v) for k, v in adata.uns.get("pca", {}).get("params", {}).items()}
    clusters = t5.leiden(adata, t18.SUBMITTED_RESOLUTION, t18.LEIDEN_KEY, seed=SEED)
    lists = t18.present_lists(workbook, adata.var_names)
    missing = {t: [g for g in genes if g not in lists.get(t, [])] for t, genes in workbook.items()}
    scores = t18.score_types(adata, lists)
    scores_bare = t18.score_matrix(cc.dense(adata.X), adata.var_names, adata.obs_names, lists)
    alt_label, _ = t18.cell_labels(t18.score_matrix(intensity, adata.var_names, adata.obs_names, lists))
    cell_label, max_score = t18.cell_labels(scores)
    adata.obs["predicted_cell_type"] = cell_label.to_numpy()
    adata.obs["max_score"] = max_score.to_numpy()
    named, naming = t18.cluster_labels(cell_label, max_score, clusters)
    adata.obs[t18.NAMED_KEY] = pd.Categorical(named.to_numpy())
    sc.tl.pca(adata, svd_solver="arpack")
    adata.obsm["X_pca_neighbors"] = pca_neighbors
    sc.tl.umap(adata, random_state=SEED)
    roots, dpt = dpt_roots(adata, cell_label, run)
    adata.obs["predicted_cell_type"] = pd.Categorical(cell_label.to_numpy(),
                                                      categories=sorted(set(cell_label.astype(str))))
    sc.tl.paga(adata, groups="predicted_cell_type")
    edges = paga_edges(adata, "predicted_cell_type")
    return {"adata": adata, "intensity": intensity, "scores": scores, "scores_bare": scores_bare,
            "alt_label": alt_label, "cell_label": cell_label, "max_score": max_score, "clusters": clusters,
            "cluster_label": named, "naming": naming, "roots": roots, "dpt": dpt, "edges": edges, "lists": lists,
            "missing": missing, "pca_neighbors_shape": pca_neighbors.shape, "pca_params": pca_params}


# ---------------------------------------------------------------------------
# Primary rule (labels_core step 2b / zerok_core route b)
# ---------------------------------------------------------------------------
@dataclass
class DEInputs:
    cell_ids: pd.Index
    var_names: pd.Index
    intensity: np.ndarray
    detected: np.ndarray
    cache: t5.RankCache
    x_z: pd.DataFrame
    det_frame: pd.DataFrame
    graph: object


def de_inputs(adata) -> DEInputs:
    """DE inputs of a step-1 object: ln(1 + intensity) ranks, scaled .X, detection, the stored graph."""
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    return DEInputs(cell_ids, var_names, intensity, detected, t5.rank_cache(np.log1p(intensity), detected),
                    pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=cell_ids, columns=var_names),
                    pd.DataFrame(detected, index=cell_ids, columns=var_names), t18.graph_adata(adata))


def label_clustering(inp: DEInputs, markers: t5.MarkerSets, resolution: float, seed: int) -> dict:
    """Leiden on the stored graph, then the cluster_de DE rule and the kidney_annotation rank-free rescue (both rules returned)."""
    cl = t18.leiden_on(inp.graph, resolution, seed)
    name = t18.clustering_name(resolution, seed)
    de, ev, asg = t18.annotate_clustering(inp.cache, cl, inp.var_names, name, markers, inp.x_z, inp.det_frame)
    rf = rank_free(asg, de, t18.specific_present(markers))
    tabs = rule_tables(asg, rf)
    out = {"name": name, "clusters": cl, "de": de, "evidence": ev, "assignment": asg, "rankfree": rf, "tables": tabs}
    for rule, tab in tabs.items():
        out[f"{rule}_label"] = t18.per_cell(cl, tab, "deg_label").astype(str)
        out[f"{rule}_confidence"] = t18.per_cell(cl, tab, "confidence").astype(str)
    out["label"], out["confidence"] = out[f"{RULE}_label"], out[f"{RULE}_confidence"]
    return out


def cluster_rows(res: float, cl: np.ndarray, asg: pd.DataFrame, rf: pd.DataFrame, de: pd.DataFrame,
                 step1: pd.DataFrame, intensity: np.ndarray, var_index: dict[str, int],
                 step1_labels=("cell_label", "cluster_label")) -> pd.DataFrame:
    """labels_core_resolution_scan.cluster_rows with the step-1 label columns as a parameter (logic unchanged)."""
    ids = step1["n_detected"].to_numpy(float)
    rows = []
    for c in t5.sort_clusters(cl):
        mask = cl == c
        sub = de[de["cluster"] == c]
        top = sub[sub["in_top"]].sort_values("rank_up")["protein"].tolist()
        depth = cc.compare_groups(ids[mask], ids[~mask])
        row = {"resolution": res, "cluster": c, "n_cells": int(mask.sum()),
               "median_protein_ids": depth["median_group"], "median_protein_ids_rest": depth["median_reference"],
               "protein_ids_rank_biserial_vs_rest": depth["rank_biserial"], "protein_ids_p_vs_rest": depth["p_value"],
               "plate_P1_fraction": float((step1.loc[mask, "plate"] == "P1").mean()),
               "n_tested": int(len(sub)), "n_sig_up": int((sub["up"] & (sub["q_bh"] < t5.Q_DE)).sum()),
               "top10_de": ";".join(top[:10])}
        for col in step1_labels:
            counts = step1.loc[mask, col].astype(str).value_counts()
            row[f"majority_{col}"] = counts.index[0]
            row[f"majority_{col}_fraction"] = float(counts.iloc[0] / mask.sum())
            row[f"{col}_counts"] = counts_text(step1.loc[mask, col])
        for group, genes in t18.CONTAMINATION_PANEL.items():
            for g in genes:
                if g in var_index:
                    row[f"detected_{group}_{g}".replace("/", "_")] = float((intensity[mask, var_index[g]] > 0).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    keep = ["cluster", "deg_label", "confidence", "overlap", "overlap_specific", "p_hypergeom", "q_hypergeom",
            "mean_marker_z", "mean_z_best_type", "runner_up", "tied_types", "dct1_umod_slc12a3_only"]
    top25 = asg[[k for k in keep if k in asg.columns]].rename(
        columns={"deg_label": "top25_label", "confidence": "top25_confidence"})
    rescue = rf[["cluster", "rankfree_label", "rankfree_confidence", "rescued", "rescue_candidates", "rescue_markers",
                 "rescue_best_rank"]]
    return frame.merge(top25, on="cluster", how="left").merge(rescue, on="cluster", how="left")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def agreement(a, b) -> dict:
    from sklearn.metrics import adjusted_rand_score

    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    return {"n_cells": len(a), "n_equal": int((a == b).sum()), "fraction_equal": float((a == b).mean()),
            "ari": float(adjusted_rand_score(a, b))}


def crosstab_long(rows, cols, row_name: str, col_name: str, **keys) -> list[dict]:
    ct = pd.crosstab(pd.Series(np.asarray(rows).astype(str), name=row_name),
                     pd.Series(np.asarray(cols).astype(str), name=col_name))
    return [{**keys, row_name: r, col_name: c, "n_cells": int(ct.loc[r, c])}
            for r in cc.ordered_labels(ct.index, LABEL_ORDER) for c in cc.ordered_labels(ct.columns, LABEL_ORDER)]


def crosstab_wide(rows, cols, row_name: str, col_name: str) -> pd.DataFrame:
    ct = pd.crosstab(pd.Series(np.asarray(rows).astype(str), name=row_name),
                     pd.Series(np.asarray(cols).astype(str), name=col_name))
    ct = ct.reindex(index=cc.ordered_labels(ct.index, LABEL_ORDER), columns=cc.ordered_labels(ct.columns, LABEL_ORDER))
    return ct.fillna(0).astype(int)


def stop(run: cc.Run, checks: t5.Checks, name: str) -> None:
    run.table(checks.frame(), name)
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit(f"A check failed; see dct_labels_{name}.csv.")
