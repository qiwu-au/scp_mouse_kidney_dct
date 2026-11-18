"""kidney_rna shared definitions and functions. Runs no analysis by itself.

Imported by the ``kidney_rna_*.py`` step scripts.
Plan: ``the analysis plan``.

Contents
--------
- Paths and settings: the RNA atlas (``mka.h5ad``, read-only), the atlas pseudobulk object written by
  ``kidney_rna_atlas_summary.py``, the kidney_annotation inputs and the three kidney label columns.
- ``load_kidney``: kidney_object canonical kidney object and labels with ``kidney_t15_label`` added from the kidney_annotation file
  (matched on ``cell_id``); label counts are checked against the project notes.
- ``match_proteins``: case-insensitive protein name -> atlas gene symbol matching (as kidney_annotation step 3a).
- ``kidney_marker_rows``: V3 workbook rows plus the kidney_annotation endothelial row.
- Pseudobulk helpers: the atlas pseudobulk holds, per atlas group (Source x Technology x Celltype x
  Celltype_finest_lowres x Celltype_finest), the number of cells and, per gene, the sum and sum of squares of
  log-normalized expression, the number of cells with counts > 0 and the sum of counts. Any union of groups is
  summarized exactly (to float32 precision) from these sums, so later steps do not reload the full atlas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dotplot_lib as t6

TASK = "kidney_rna"
SEED = cc.RANDOM_SEED
DATASET = "kidney"

# Atlas ----------------------------------------------------------------------------------------------
ATLAS = cc.PROJECT_ROOT / "data" / "reference" / "mka.h5ad"
ATLAS_EXPECTED_SHAPE = (141401, 3000)
ATLAS_EXPECTED_RAW_GENES = 16711
TARGET_SUM = 1e4
ATLAS_LEVELS = ("Celltype", "Celltype_finest_lowres", "Celltype_finest")
ATLAS_FIELDS = ("Origin", "Source", "Technology", "Tissue_res", "Age", "Genetic_background", "Gender", "batch")
GROUP_KEYS = ("Source", "Technology", *ATLAS_LEVELS)
ATLAS_UNKNOWN = "Unknown"
SOURCE_DISPLAY = {"Cell": "scRNA-seq (cells)", "Nuclei": "snRNA-seq (nuclei)"}
PSEUDOBULK_H5AD = cc.out_path(TASK, "h5ad", "atlas_pseudobulk", ".h5ad")
PSEUDOBULK_LAYERS = ("sum_sq_lognorm", "n_detected", "sum_counts")   # .X = sum_lognorm

# Display order of atlas classes (nephron, then vascular, stromal, immune); others follow alphabetically.
ATLAS_ORDER = ("PTS1", "PTS2", "PTS1-2", "PCT", "PT", "PTS3", "PTS3T2", "PST", "DTL", "DTL-ATL", "ATL", "LOH", "MTAL",
               "CTAL", "MD", "DCT", "DCT-CNT", "CNT", "PC", "CD-Trans", "IC", "ICA", "ICB", "Podo", "PEC", "Endo",
               "Glom-Endo", "Vas-Afferens", "Vas-Efferens", "Desc-Vasa-Recta", "Asc-Vasa-Recta", "Fib", "Stroma",
               "Peri_Stromal", "Per", "MC", "Macro", "Mp/DC", "DC", "Neutro", "Lymphocytes", "B lymph", "T lymph", "NK",
               "Unknown")


def atlas_order(labels) -> list[str]:
    present = set(map(str, labels))
    return [a for a in ATLAS_ORDER if a in present] + sorted(present - set(ATLAS_ORDER))


# kidney_annotation inputs ------------------------------------------------------------------------------------------
KIDNEY_ANNOTATION_DIR = cc.OUT_ROOTS["tables"] / "kidney_annotation"
KIDNEY_ANNOTATION_LABELS = KIDNEY_ANNOTATION_DIR / "kidney_annotation_cluster_labels_per_cell_a_endo.csv"
KIDNEY_ANNOTATION_ATLAS_MAPPING = KIDNEY_ANNOTATION_DIR / "kidney_annotation_atlas_celltype_mapping.csv"
KIDNEY_ANNOTATION_MARKER_ORDER = KIDNEY_ANNOTATION_DIR / "kidney_annotation_t15_dotplot_marker_order_a_endo.csv"
KIDNEY_ANNOTATION_DOTPLOT_VALUES = KIDNEY_ANNOTATION_DIR / "kidney_annotation_t15_dotplot_values_a_endo.csv"
KIDNEY_ANNOTATION_ATLAS_SHARED_GENES = 3569        # kidney_annotation step 3a check: atlas genes matching a quantified protein
ENDO_TYPE = "Endothelial"
ENDOTHELIAL_MARKERS = ("Pecam1", "Emcn", "Eng", "Esam", "Icam2", "Fabp4", "Ehd4")   # = kidney_annotation_scan_lib

# Kidney labels ----------------------------------------------------------------------------------------
PRIMARY = "kidney_t15_label"
LABEL_COLUMNS = ("kidney_t15_label", "kidney_pre_label", "kidney_reannot_label")
LABEL_ROLE = {
    "kidney_t15_label": "primary (kidney_annotation (a); adopted 14 Oct 2026)",
    "kidney_pre_label": "Fig 2A (pre-annotation; comparison)",
    "kidney_reannot_label": "Fig 2B (submitted re-annotation; step-4 comparison only)",
}
EXPECTED_COUNTS = {   # the project notes
    "kidney_t15_label": {"PCT": 160, "PST": 158, "TAL": 138, "DCT1": 58, "PC": 38, "JG": 24, "Endothelial": 19,
                         "IC-B": 8, "Unknown": 109},
    "kidney_pre_label": {"PCT": 168, "TAL": 154, "PST": 132, "DVR": 88, "CNT": 73, "DCT2": 24, "JG": 22, "MD": 19,
                         "IC-B": 8, "Unknown": 24},
    "kidney_reannot_label": {"PCT": 219, "PC": 187, "PST": 72, "DTL": 56, "CNT": 53, "TAL": 45, "DCT2": 27,
                             "DCT1": 18, "DVR": 16, "ATL": 7, "JG": 6, "IC-B": 3, "IC-A": 2, "MD": 1},
}
# Row order used in every kidney_rna table and figure: the dotplot kidney order with the endothelial row first.
ROW_ORDER = (ENDO_TYPE, *t6.KIDNEY_ROW_ORDER, "Unknown")


def row_order(labels) -> list[str]:
    present = set(map(str, labels))
    extra = sorted(present - set(ROW_ORDER))
    if extra:
        raise ValueError(f"labels without a kidney_rna row position: {extra}")
    return [r for r in ROW_ORDER if r in present]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_kidney(run: cc.Run, checks: t5.Checks | None = None):
    """kidney_object canonical kidney object and labels, with ``kidney_t15_label`` (and cluster, confidence) from kidney_annotation."""
    adata, labels = t5.load(run, DATASET)
    t15 = pd.read_csv(run.use_input(KIDNEY_ANNOTATION_LABELS, "kidney_annotation adopted kidney labels (a)"), dtype={"cell_id": str})
    t15 = t15.set_index("cell_id")
    same_cells = set(t15.index) == set(labels.index) and len(t15) == len(labels)
    if not same_cells:
        raise ValueError("kidney_annotation label file and kidney_object canonical labels hold different cells")
    t15 = t15.reindex(labels.index)
    for col in ("kidney_reannot_label", "kidney_pre_label"):
        if not np.array_equal(t15[col].astype(str).to_numpy(), labels[col].astype(str).to_numpy()):
            raise ValueError(f"kidney_annotation file column {col} differs from the kidney_object canonical labels")
    labels = labels.copy()
    labels["kidney_t15_label"] = t15["kidney_t15_label"].astype(str)
    labels["t15_cluster"] = t15["t15_cluster"].astype(str)
    labels["kidney_t15_confidence"] = t15["kidney_t15_confidence"].astype(str)
    if checks is not None:
        checks.add("kidney_annotation labels matched on cell_id (same 712 cells; reannot / pre columns identical to kidney_object)",
                   f"{len(labels)} cells", same_cells)
        for col in LABEL_COLUMNS:
            observed = labels[col].value_counts().to_dict()
            ok = observed == EXPECTED_COUNTS[col]
            checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {v}" for k, v in
                                                                  sorted(observed.items(), key=lambda kv: -kv[1])), ok)
    return adata, labels


def kidney_marker_rows(run: cc.Run) -> list[tuple[str, str, list[str]]]:
    """V3 workbook rows (short, full name, markers) plus the kidney_annotation endothelial row."""
    workbook, _ = t6.read_kidney_workbook(run.use_input(cc.INPUTS["kidney"]["markers"],
                                                        "kidney V3 marker workbook (Supp Table 1)"))
    return [*workbook, (ENDO_TYPE, "Endothelial (kidney_annotation extension; atlas Endo / AVR markers)",
                        list(ENDOTHELIAL_MARKERS))]


def markers_by_gene(rows: list[tuple[str, str, list[str]]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for short, _, genes in rows:
        for g in genes:
            out.setdefault(g, [])
            if short not in out[g]:
                out[g].append(short)
    return out


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------
def protein_lookup(protein_names) -> dict[str, str]:
    """Lower-case symbol -> protein name, for protein names usable as gene symbols (as kidney_annotation step 3a)."""
    return {p.lower(): p for p in map(str, protein_names) if "_" not in p and p != "Nan"}


def match_proteins(protein_names, atlas_genes) -> pd.DataFrame:
    """One row per atlas gene matching a protein name case-insensitively: protein, atlas_gene, atlas position."""
    base = protein_lookup(protein_names)
    rows = [{"protein": base[g.lower()], "atlas_gene": g, "atlas_position": i}
            for i, g in enumerate(map(str, atlas_genes)) if g.lower() in base]
    out = pd.DataFrame(rows, columns=["protein", "atlas_gene", "atlas_position"])
    if out["protein"].duplicated().any() or out["atlas_gene"].duplicated().any():
        raise ValueError("protein <-> atlas gene matching is not one-to-one")
    return out


def atlas_gene_lookup(atlas_genes) -> dict[str, str]:
    return {g.lower(): g for g in map(str, atlas_genes)}


# ---------------------------------------------------------------------------
# Pseudobulk
# ---------------------------------------------------------------------------
def read_pseudobulk(run: cc.Run):
    import anndata as ad

    if not PSEUDOBULK_H5AD.exists():
        raise FileNotFoundError(f"{cc.rel(PSEUDOBULK_H5AD)} not found; run kidney_rna_atlas_summary.py first.")
    return ad.read_h5ad(run.use_input(PSEUDOBULK_H5AD, "kidney_rna atlas pseudobulk (kidney_rna_atlas_summary.py)"))


def group_mask(pb, **criteria) -> np.ndarray:
    """Boolean mask over pseudobulk groups; each criterion is a value or a collection of values of an obs key."""
    mask = np.ones(pb.n_obs, dtype=bool)
    for key, value in criteria.items():
        values = [value] if isinstance(value, str) else list(value)
        mask &= pb.obs[key].astype(str).isin(values).to_numpy()
    return mask


def aggregate(pb, mask: np.ndarray, genes=None) -> pd.DataFrame:
    """Per gene over the union of the masked groups: n cells, mean and SD of log-normalized expression,
    fraction of cells with counts > 0, mean counts."""
    cols = slice(None) if genes is None else pb.var_names.get_indexer(list(genes))
    if genes is not None and (np.asarray(cols) < 0).any():
        raise KeyError("genes not in the pseudobulk")
    n = float(pb.obs["n_cells"].to_numpy()[mask].sum())
    names = pb.var_names if genes is None else pd.Index(list(genes))
    if n == 0:
        return pd.DataFrame({"n_cells": 0, "mean_lognorm": np.nan, "sd_lognorm": np.nan, "frac_detected": np.nan,
                             "mean_counts": np.nan}, index=names)
    s1 = np.asarray(pb.X[mask][:, cols], dtype=np.float64).sum(axis=0)
    s2 = np.asarray(pb.layers["sum_sq_lognorm"][mask][:, cols], dtype=np.float64).sum(axis=0)
    nd = np.asarray(pb.layers["n_detected"][mask][:, cols], dtype=np.float64).sum(axis=0)
    sc_ = np.asarray(pb.layers["sum_counts"][mask][:, cols], dtype=np.float64).sum(axis=0)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    return pd.DataFrame({"n_cells": int(n), "mean_lognorm": mean, "sd_lognorm": np.sqrt(var),
                         "frac_detected": nd / n, "mean_counts": sc_ / n}, index=names)
