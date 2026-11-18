"""ctrl_rna_core shared definitions and functions. Runs no analysis by itself.

Imported by the ``ctrl_rna_core_*.py`` step scripts.
Plan: ``the analysis plan`` (Editors 1; R1 ProLIF prevalence).

ctrl_rna_core repeats the SCP side of atlas_dct (RNA comparator for the DCT-enriched SCP-MS data) on the new control data
(intake_core object, labels_core labels; 727 cells x 6,196 proteins). The atlas side of atlas_dct is reused unchanged: the DCT-region
subset ``results/h5ad/atlas_dct_atlas_dct_region.h5ad``, the atlas_dct-derived DCT1 / DCT2 split and the atlas_dct atlas
cycling fractions. No number is compared with the retired 758-cell DCT data.

Decisions by the authors (28 Oct 2026)
-----------------------------------
- Label columns: ``ctrl_deg_label`` (primary; labels_core step 7) and ``ctrl_cluster_label`` (second column, as ctrl_reviewer_core).
- Unknown is its own SCP row in steps 2-4 (72 cells under ``ctrl_deg_label``, 26 under ``ctrl_cluster_label``);
  step 5 compares DCT1 and DCT2 only.

Reuse
-----
The step scripts import ``atlas_dct_lib`` (atlas region, log-normalization, group statistics, matching rule,
Wilson intervals, figure sheet and checks) and the helper functions of the atlas_dct step scripts unchanged. The atlas_dct
``main()`` and figure functions hard-code the kidney_object object, the V1 label columns and the reviewer_core / dotplot tables; they are
ported into the ctrl_rna_core scripts. Definitions replaced here:
- ``atlas_dct_lib.load_scp`` -> ``load_scp`` (labels_core object and canonical labels via ``ctrl_reviewer_core_lib.load_ctrl``);
- ``atlas_dct_lib.TASK``, ``LABEL_COLUMNS``, ``LABEL_DISPLAY``, ``LABEL_SHORT``, ``SCP_GROUPS``,
  ``EXPECTED_COUNTS``, ``read_table`` -> the same names here;
- ``atlas_dct_lib.reviewer_core_DIR`` / ``dotplot_DIR`` -> ``ctrl_reviewer_core_DIR`` / ``labels_core_DIR``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import atlas_dct_lib as T
import ctrl_reviewer_core_lib as L19

TASK = "ctrl_rna_core"
SEED = cc.RANDOM_SEED
DATASET = "dct_ctrl"

# SCP labels -----------------------------------------------------------------------------------------------
PRIMARY = L19.PRIMARY                                # ctrl_deg_label (authors, 28 Oct 2026, labels_core step 7)
SECOND = L19.SECOND                                  # ctrl_cluster_label (authors, 28 Oct 2026, ctrl_rna_core plan)
LABEL_COLUMNS = (PRIMARY, SECOND)
LABEL_DISPLAY = {"ctrl_deg_label": "DE-rule labels (primary)", "ctrl_cluster_label": "submitted cluster labels"}
LABEL_SHORT = {"ctrl_deg_label": "deg", "ctrl_cluster_label": "cluster"}
CORE_GROUPS = ("DCT1", "DCT2", "ProLIF")
SCP_GROUPS = (*CORE_GROUPS, "Unknown")              # Unknown its own row (authors, 28 Oct 2026)
EXPECTED_COUNTS = {k: L19.EXPECTED_COUNTS[k] for k in LABEL_COLUMNS}
SCP_PALETTE = T.SCP_PALETTE                          # Fig 3 colours (DCT1, DCT2, ProLIF, Unknown)
N_CELLS, N_PROTEINS = 727, 6196

# Inputs of other tasks ---------------------------------------------------------------------------------------
ATLAS_DCT_TASK = T.TASK
LABELS_CORE_TASK = L19.LABELS_CORE_TASK
CTRL_REVIEWER_CORE_TASK = L19.TASK
LABELS_CORE_DIR = cc.OUT_ROOTS["tables"] / LABELS_CORE_TASK
CTRL_REVIEWER_CORE_DIR = cc.OUT_ROOTS["tables"] / CTRL_REVIEWER_CORE_TASK
KIDNEY_RNA_PROVENANCE = cc.OUT_ROOTS["tables"] / "kidney_rna" / "kidney_rna_atlas_provenance.md"

DESIGN_NOTE = L19.DESIGN_NOTE
ATLAS_DCT_SPLIT_CAVEAT = ("DCT1 (atlas_dct) / DCT2 (atlas_dct): atlas_dct-derived split of the atlas DCT cells, not atlas labels; the DCT1 (atlas_dct) "
                    "call is not Pvalb-driven and DCT2 (atlas_dct) holds 39 nuclei.")
UNPAIRED_NOTE = "RNA and protein come from different animals and studies (unpaired comparison)."


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_scp(run: cc.Run, checks: t5.Checks | None = None):
    """labels_core labelled control object and canonical labels (``ctrl_reviewer_core_lib.load_ctrl``: cells, order, label, cluster,
    plate and depth columns, the project notes counts); ln(1 + intensity) (missing = 0) and detection (intensity > 0).

    Replaces ``atlas_dct_lib.load_scp``. ``labels`` is indexed by ``cell_id`` in object order and holds
    ``plate``, the label columns and ``n_detected_proteins``.
    """
    adata, labels = L19.load_ctrl(run)
    if checks is not None:
        checks.add("labels_core object: cells x proteins", f"{adata.n_obs} x {adata.n_vars}",
                   adata.shape == (N_CELLS, N_PROTEINS))
        for col in LABEL_COLUMNS:
            observed = labels[col].astype(str).value_counts().to_dict()
            checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in SCP_GROUPS),
                       observed == EXPECTED_COUNTS[col])
    names = pd.Index(adata.var_names.astype(str))
    inten = cc.dense(adata.layers["intensity"]).astype(np.float64)
    ln1p = pd.DataFrame(np.log1p(inten), index=labels.index, columns=names)
    det = pd.DataFrame(inten > 0, index=labels.index, columns=names)
    return adata, labels, ln1p, det


def read_table(run: cc.Run, name: str, task: str = TASK, **kwargs) -> pd.DataFrame:
    """A table written by a ctrl_rna_core step (or, with ``task``, by another revision task)."""
    path = cc.out_path(task, "tables", name, ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the step that writes it first.")
    return pd.read_csv(run.use_input(path, f"{task} output"), **kwargs)


def per_cell_table(run: cc.Run, task: str, name: str, labels: pd.DataFrame, checks: t5.Checks) -> pd.DataFrame:
    """A per-cell table of labels_core / ctrl_reviewer_core reindexed to the label order; checks cells and the label columns."""
    tab = read_table(run, name, task, dtype={"cell_id": str}).set_index("cell_id")
    same_cells = set(tab.index) == set(labels.index) and len(tab) == len(labels)
    tab = tab.reindex(labels.index)
    cols = [c for c in (*LABEL_COLUMNS, "plate") if c in tab.columns]
    same = same_cells and tab[cols].astype(str).equals(labels[cols].astype(str))
    checks.add(f"{task} {name}: same {len(labels)} cells, labels and plate as the labels_core canonical table", f"{same}", same)
    return tab


def ported_from(run: cc.Run, *scripts: str) -> None:
    """Record the atlas_dct scripts a step ports or imports functions from (reference code; not run)."""
    for name in scripts:
        run.use_input(cc.SCRIPTS_DIR / name, "reference code: atlas_dct script (functions imported / ported; not run)")


def counts_text(values, order=SCP_GROUPS) -> str:
    counts = pd.Series(np.asarray(values).astype(str)).value_counts()
    return "; ".join(f"{k} {int(counts[k])}" for k in order if k in counts.index)


def label_rows(col: str) -> list[tuple[str, str]]:
    """(label column, SCP group) pairs in display order."""
    return [(col, g) for g in SCP_GROUPS]
