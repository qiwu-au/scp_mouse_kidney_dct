"""ctrl_reviewer_core shared definitions and functions. Runs no analysis by itself.

Imported by the
``ctrl_reviewer_core_*.py`` step scripts. Plan: ``the analysis plan``.

ctrl_reviewer_core repeats the DCT reviewer analyses of reviewer_core, qc_mito and preprocessing on the new control
data (intake_core object, labels_core labels; 727 cells x 6,196 proteins):
- primary label column ``ctrl_deg_label`` (authors, 28 Oct 2026, labels_core step 7);
- second label column ``ctrl_cluster_label`` (authors, 28 Oct 2026, ctrl_reviewer_core plan).
Plates P1/P2 are sort plates of one pooled sample: a technical label, never a
replicate. No number is compared with the retired 758-cell data.

The step scripts import the functions of ``reviewer_core_lib``, ``qc_mito_lib``
and ``preprocessing_lib``. Definitions there that hard-code the kidney_object paths or the V1
label columns are replaced here, with the same names where the step scripts use
them:
- ``reviewer_core_lib.load_dct`` -> ``load_ctrl`` (labels_core object and canonical table);
- ``reviewer_core_lib.TASK``, ``LABEL_COLUMNS``, ``LABEL_DISPLAY`` -> ``TASK``,
  ``LABEL_COLUMNS``, ``LABEL_DISPLAY``;
- ``reviewer_core_lib.h5ad_path``, ``table_path``, ``read_table``, ``read_h5ad``
  -> the same functions for the ctrl_reviewer_core folders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
from reviewer_core_lib import LABEL_PALETTE

TASK = "ctrl_reviewer_core"
SEED = cc.RANDOM_SEED
CONDITION = "Ctrl"

PRIMARY = "ctrl_deg_label"
SECOND = "ctrl_cluster_label"
LABEL_COLUMNS = (PRIMARY, SECOND)
LABEL_DISPLAY = {"ctrl_deg_label": "DE-rule label (primary)",
                 "ctrl_cluster_label": "submitted cluster label",
                 "ctrl_cell_label": "submitted per-cell label"}
# Replaces scp_common.DCT_LABEL_SOURCES (V1 columns) for figure captions.
LABEL_SOURCES = {
    "ctrl_deg_label": "labels_core: cluster_de DE rule + kidney_annotation rank-free rescue on Leiden 1.0 clusters of the submitted graph",
    "ctrl_cluster_label": "labels_core: submitted pipeline, Leiden 3.0 cluster mode with the original Unknown rule",
    "ctrl_cell_label": "labels_core: submitted pipeline, per-cell argmax of the DCT V1 workbook scores",
}
# Clusters each label column was named on (labels_core): column -> (cluster column, Leiden resolution).
STORED_CLUSTERS = {"ctrl_deg_label": ("ctrl_deg_cluster", 1.0), "ctrl_cluster_label": ("ctrl_leiden3_cluster", 3.0)}
LABEL_ORDER = cc.DCT_LABEL_ORDER
# labels_core README / the project notes (28 Oct 2026).
EXPECTED_COUNTS = {
    "ctrl_deg_label": {"DCT1": 477, "DCT2": 144, "ProLIF": 34, "Unknown": 72},
    "ctrl_cluster_label": {"DCT1": 386, "DCT2": 169, "ProLIF": 146, "Unknown": 26},
    "ctrl_cell_label": {"DCT1": 347, "DCT2": 217, "ProLIF": 163},
}
LABELS_CORE_TASK = "labels_core"
LABELS_CORE_H5AD = cc.out_path(LABELS_CORE_TASK, "h5ad", "DCT_ctrl_labelled", ".h5ad")
LABELS_CORE_LABELS = cc.out_path(LABELS_CORE_TASK, "tables", "dct_ctrl_canonical_labels", ".csv")
COMPARED_OBS = ("plate", "ctrl_cell_label", "ctrl_cluster_label", "ctrl_leiden3_cluster", "ctrl_deg_label",
                "ctrl_deg_cluster", "ctrl_deg_confidence", "n_detected")
DESIGN_NOTE = ("One pool of 3 mice per condition; plates P1/P2 are sort plates of the pooled cells (technical label, "
               "never a replicate).")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_ctrl(run: cc.Run):
    """labels_core labelled control object and canonical labels, matched on ``cell_id`` (replaces ``reviewer_core_lib.load_dct``).

    Checks that the CSV and the h5ad hold the same cells in the same order and the
    same label, cluster, plate and depth columns, and that the label counts equal
    the project notes. ``labels`` is indexed by ``cell_id`` in object order and gains
    ``n_detected_proteins`` (= ``n_detected``, the reviewer_core column name).
    """
    import anndata as ad

    for path, label in ((LABELS_CORE_LABELS, "labels_core canonical control labels"), (LABELS_CORE_H5AD, "labels_core labelled control object")):
        run.use_input(path, label)
    labels = pd.read_csv(LABELS_CORE_LABELS, dtype={"cell_id": str, "ctrl_leiden3_cluster": str, "ctrl_deg_cluster": str})
    labels = labels.set_index("cell_id")
    adata = ad.read_h5ad(LABELS_CORE_H5AD)
    if list(adata.obs_names.astype(str)) != list(labels.index):
        raise ValueError("labels_core canonical CSV and h5ad cell order differ.")
    for col in COMPARED_OBS:
        a = adata.obs[col].astype(str).to_numpy()
        b = labels[col].astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"labels_core column {col} differs between the canonical CSV and the h5ad.")
    for col, expected in EXPECTED_COUNTS.items():
        counts = labels[col].astype(str).value_counts().to_dict()
        if counts != expected:
            raise ValueError(f"{col} counts {counts} differ from the project notes {expected}.")
    labels = labels.copy()
    labels["n_detected_proteins"] = labels["n_detected"].astype(int)
    return adata, labels


# ---------------------------------------------------------------------------
# Paths (reviewer_core_lib helpers with the ctrl_reviewer_core task)
# ---------------------------------------------------------------------------
def h5ad_path(name: str) -> Path:
    return cc.out_path(TASK, "h5ad", name, ".h5ad")


def table_path(name: str, subdir: str = "") -> Path:
    return cc.out_path(TASK, "tables", name, ".csv", subdir)


def read_table(run: cc.Run, name: str, subdir: str = "", **kwargs) -> pd.DataFrame:
    path = table_path(name, subdir)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the ctrl_reviewer_core step that writes it first.")
    return pd.read_csv(run.use_input(path, "ctrl_reviewer_core output"), **kwargs)


def read_h5ad(run: cc.Run, name: str):
    import anndata as ad

    path = h5ad_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the ctrl_reviewer_core step that writes it first.")
    return ad.read_h5ad(run.use_input(path, "ctrl_reviewer_core AnnData"))


def ported_from(run: cc.Run, *scripts: str) -> None:
    """Record the reviewer_core-preprocessing scripts a step ports (reference code; not run)."""
    for name in scripts:
        run.use_input(cc.SCRIPTS_DIR / name, "reference code: ported revision script (not run)")


# ---------------------------------------------------------------------------
# Part 2: replacements for qc_mito_lib definitions that name the kidney_object objects or the V1 columns
# (DATASETS, SHORT_COLUMN, LABEL_DISPLAY, load, read_annotation, canonical_meta, label_colors)
# ---------------------------------------------------------------------------
QC_MITO_RUN = ("dct",)   # ctrl_reviewer_core runs the DCT part only (the kidney results of qc_mito and preprocessing stand)
QC_MITO_DATASETS = {"dct": {"display": "DCT control", "label_columns": LABEL_COLUMNS,
                        "plot_columns": (*LABEL_COLUMNS, "plate"), "order": cc.DCT_LABEL_ORDER,
                        "reference_label": "DCT1"}}
SHORT_COLUMN = {"ctrl_deg_label": "deg", "ctrl_cluster_label": "cluster", "ctrl_cell_label": "cell", "plate": "plate"}
QC_MITO_LABEL_DISPLAY = {**LABEL_DISPLAY, "plate": "plate (pooled sort plate)"}
ANNOTATION_NAME = "protein_annotation"


def label_colors(column: str) -> dict[str, str]:
    """Replaces ``qc_mito_lib.label_colors`` (colours stored in the kidney_object objects): labels_core label colours or plate palette."""
    return dict(cc.PLATE_PALETTE) if column == "plate" else dict(LABEL_PALETTE)


def mito_meta(var: pd.DataFrame, names) -> pd.DataFrame:
    """``qc_mito_lib.canonical_meta`` for the intake_core metadata columns (gene_name, protein_group, pg_genes, description)."""
    return pd.DataFrame({"pg_genes": var["pg_genes"].astype(str).to_numpy(),
                         "protein_group": var["protein_group"].astype(str).to_numpy(),
                         "first_symbol": var["gene_name"].astype(str).to_numpy(),
                         "protein_description": var["protein_description"].astype(str).to_numpy()},
                        index=pd.Index(list(map(str, names)), name="protein"))


def ctrl_protein_meta(run: cc.Run, var_names) -> pd.DataFrame:
    """Protein-group metadata of the control matrix from the intake_core object (the labels_core object keeps gene names only)."""
    import intake_core_lib as t17

    obj = t17.read_t17_object(run, CONDITION)
    if list(obj.var_names.astype(str)) != list(map(str, var_names)):
        raise ValueError("intake_core and labels_core control objects hold different proteins or protein order.")
    return mito_meta(obj.var, obj.var_names)


def read_mito_annotation(run: cc.Run, var_names) -> pd.DataFrame:
    """Replaces ``qc_mito_lib.read_annotation``: the ctrl_reviewer_core MitoCarta annotation written by ctrl_reviewer_core_mito_qc.py."""
    from qc_mito_lib import SETS

    path = table_path(ANNOTATION_NAME)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run ctrl_reviewer_core_mito_qc.py first.")
    frame = pd.read_csv(run.use_input(path, "ctrl_reviewer_core protein annotation (MitoCarta flags)"), keep_default_na=False)
    frame = frame.set_index("protein")
    if list(frame.index.astype(str)) != list(map(str, var_names)):
        raise ValueError("ctrl_reviewer_core protein annotation order differs from the control object.")
    for col in SETS:
        frame[col] = frame[col].astype(str).str.lower().eq("true")
    return frame


def rebuild_ctrl_intensity(run: cc.Run) -> pd.DataFrame:
    """Retained control cells x retained proteins rebuilt from the raw export with the intake_core reader and filters.

    Replaces ``scp_common.rebuild_ln_intensity`` (it reads the retired exports) for the preprocessing rebuild check.
    """
    import intake_core_lib as t17

    raw = t17.read_raw(CONDITION, run)
    filt = t17.apply_filters(raw)
    names = raw.meta["var_name"].astype(str).to_numpy()[filt.keep_proteins]
    cells = raw.cells.index.astype(str)[filt.keep_cells]
    return pd.DataFrame(t17.intensity_matrix(raw, filt), index=pd.Index(cells, name="cell_id"), columns=names)


# ---------------------------------------------------------------------------
# Part 2: replacements for preprocessing_lib definitions that name the kidney_object objects or the V1 columns
# (DATASETS, EXPECTED_COUNTS, load, macro)
# ---------------------------------------------------------------------------
CORE_LABELS = ("DCT1", "DCT2", "ProLIF")
PREPROCESSING_DATASETS = {"dct": {"display": "DCT control", "label_columns": LABEL_COLUMNS, "primary": (PRIMARY,),
                        "order": cc.DCT_LABEL_ORDER, "stored_leiden": "ctrl_leiden3_cluster",
                        "submitted_resolution": 3.0}}


def macro_core(frame: pd.DataFrame, value: str, min_n: int = 10) -> float:
    """``preprocessing_lib.macro`` over DCT1, DCT2 and ProLIF only (labels with >= min_n cells).

    Unknown is reported per label but left out of the macro averages (ctrl_reviewer_core plan, authors 28 Oct 2026).
    """
    sub = frame[frame["label"].isin(CORE_LABELS) & (frame["n_cells"] >= min_n)]
    return float(sub[value].mean()) if len(sub) else np.nan


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def counts_text(values, order=LABEL_ORDER) -> str:
    counts = pd.Series(np.asarray(values).astype(str)).value_counts()
    return "; ".join(f"{k} {int(counts[k])}" for k in cc.ordered_labels(counts.index, order))


__all__ = ["LABEL_PALETTE"]
