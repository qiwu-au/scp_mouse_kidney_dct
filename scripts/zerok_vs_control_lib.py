#!/usr/bin/env python
"""zerok_vs_control shared settings: zero-K vs control on the combined Spectronaut search (zerok_response_core rerun).

Imported by the ``zerok_vs_control_*.py`` step scripts; runs no
analysis by itself. Plan: ``the analysis plan`` (frozen 2 Nov 2026; rules in
``00_combined_search_overview.md``).

The statistics are the zerok_response_core functions, imported unchanged from ``zerok_response_core_lib`` (two-part model, D1 direction and
match status, depth matching, scores, panel and proteome functions). Settings D1-D8 and the step 5 / 5b / 5c
settings are the zerok_response_core settings (manifests win). What differs from zerok_response_core is the input and its checks (plan approved
by the authors, 2 Nov 2026):

- Input: the dct_labels labelled joint object (1,431 cells x 5,897 protein groups retained in both conditions of one
  combined search); labels from the dct_labels canonical table, matched on ``cell_id`` (checked) and ``cell_key``.
- Depth covariate: log ``n_detected`` of the joint object (joint proteins per cell); zerok_response_core used
  ``n_detected_matched``. ``Joint.log_depth`` is overridden here; nothing in zerok_response_core_lib is rebound.
- Labels absent in a condition (primary ProLIF and ZeroK DCT2, alternative ProLIF) give 'does not apply' rows
  instead of a test; the other label column's result is reported beside them.
- One combined search: the p-value note names condition and pool only. The Ctrl intensities still carry
  Spectronaut's cross-run normalization over all 1,535 runs.
- The joint var gains ``match_level`` = 'combined_search' and ``protein_group_ctrl`` = ``protein_group_zerok`` =
  the combined protein group, so that the zerok_response_core proteome and bulk-matching functions apply unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import zerok_response_core_lib as t22
import dct_intake_lib as t26

TASK = "zerok_vs_control"
SEED = cc.RANDOM_SEED
CONDITIONS = t22.CONDITIONS
PLATES = t22.PLATES

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
FREEZE_TABLE = t26.FREEZE_TABLE
PANEL_SHA256 = t22.PANEL_SHA256
PANEL_CSV = t22.PANEL_CSV
PANEL_HASH_CSV = t22.PANEL_HASH_CSV
ZEROK_CORE_DESIGN_MD = t22.DESIGN_MD
DCT_LABELS_TASK = "dct_labels"
JOINT_H5AD = cc.out_path(DCT_LABELS_TASK, "h5ad", "DCT_joint_labelled", ".h5ad")
LABELS_CSV = cc.out_path(DCT_LABELS_TASK, "tables", "dct_canonical_labels", ".csv")
ZEROK_RESPONSE_CORE_TASK = t22.TASK

DEPTH_COL = "n_detected"
N_CELLS = {"Ctrl": 729, "ZeroK": 702}
N_PROTEINS = 5897
EXPECTED_COUNTS = {  # dct_labels README (canonical table)
    ("dct_label", "Ctrl"): {"DCT1": 382, "DCT2": 85, "ProLIF": 0, "Unknown": 262},
    ("dct_label", "ZeroK"): {"DCT1": 449, "DCT2": 0, "ProLIF": 0, "Unknown": 253},
    ("dct_label_alternative", "Ctrl"): {"DCT1": 382, "DCT2": 85, "ProLIF": 0, "Unknown": 262},
    ("dct_label_alternative", "ZeroK"): {"DCT1": 398, "DCT2": 68, "ProLIF": 0, "Unknown": 236},
}
LABEL_SOURCES = {("dct_label", "Ctrl"): "cs_ctrl_deg_label", ("dct_label", "ZeroK"): "cs_zerok_pipeline_label",
                 ("dct_label_alternative", "Ctrl"): "cs_ctrl_deg_label",
                 ("dct_label_alternative", "ZeroK"): "cs_zerok_transfer_label"}

# ---------------------------------------------------------------------------
# zerok_response_core settings and names (unchanged)
# ---------------------------------------------------------------------------
VARIANTS = t22.VARIANTS
BASELINE = t22.BASELINE
LABEL_COLUMNS = t22.LABEL_COLUMNS
LABEL_SHORT = t22.LABEL_SHORT
LABEL_DISPLAY = {"dct_label": "dct_label (primary: Ctrl cs_ctrl_deg_label + ZeroK cs_zerok_pipeline_label)",
                 "dct_label_alternative": "alternative (Ctrl cs_ctrl_deg_label + ZeroK cs_zerok_transfer_label)",
                 "label_free": "label-free"}
CORE, LABELS = t22.CORE, t22.LABELS
ALL_DCT, ALL_CELLS, SLC12A3_DETECTED = t22.ALL_DCT, t22.ALL_CELLS, t22.SLC12A3_DETECTED
ALPHA = t22.ALPHA

P_NOTE = ("descriptive: one pool of 3 mice per condition (condition and pool cannot be separated); "
          "cells are not animal-level replicates")
DESIGN_NOTE = ("One pool of 3 mice per condition: condition and pool cannot be separated; cells are not "
               "animal-level replicates; P1/P2 are sort plates of the pooled cells (technical only). Both conditions "
               "come from one combined Spectronaut search; the Ctrl intensities carry its cross-run normalization "
               "over all 1,535 runs.")
DOES_NOT_APPLY = "does not apply"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the zerok_vs_control step that writes it first.")
    return pd.read_csv(run.use_input(path, "zerok_vs_control output"), **kwargs)


def t22_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    """A zerok_response_core (separate-search) table: bridge step only."""
    return pd.read_csv(run.use_input(t22.table_path(name), "zerok_response_core output (bridge only)"), **kwargs)


def check_inputs(run: cc.Run, checks: t5.Checks) -> None:
    """Plan and overview SHA256 against the freeze record; the zerok_core panel hash (stops on a difference)."""
    freeze = pd.read_csv(run.use_input(FREEZE_TABLE, "freeze record of the dct_intake-zerok_vs_control plans"))
    recorded = dict(zip(freeze["path"], freeze["sha256"]))
    for path in ():   # no planning document is part of the public code
        key = cc.rel(path).replace("\\", "/")
        observed = cc.sha256(run.use_input(path, "frozen plan"))
        ok = observed == recorded.get(key)
        checks.add(f"SHA256 of {key} equals the freeze record", observed, ok, str(recorded.get(key)))
        if not ok:
            raise SystemExit(f"{key} differs from the freeze record; stopping (zerok_vs_control plan).")
    t22.check_panel_hash(run, checks)


# ---------------------------------------------------------------------------
# Joint object
# ---------------------------------------------------------------------------
@dataclass
class Joint(t22.Joint):
    """The dct_labels labelled joint object as plain arrays; depth = joint proteins detected per cell."""

    @property
    def log_depth(self) -> np.ndarray:
        return np.log(self.obs[DEPTH_COL].to_numpy(float))


def load_joint(run: cc.Run, checks: t5.Checks) -> Joint:
    """dct_labels labelled joint object + dct_labels canonical labels, with the input checks every step repeats."""
    import anndata as ad

    adata = ad.read_h5ad(run.use_input(JOINT_H5AD, "dct_labels labelled joint object (dct_intake joint, 5,897 protein groups)"))
    labels = pd.read_csv(run.use_input(LABELS_CSV, "dct_labels canonical DCT labels (combined search)"),
                         dtype={"cell_id": str, "well": str})
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    for col in ("condition", "plate", "cell_key", "dct_label", "dct_label_alternative"):
        obs[col] = obs[col].astype(str)
    n_by = obs["condition"].value_counts().to_dict()
    checks.add("joint object cells (Ctrl / ZeroK)", f"{len(obs)} ({n_by.get('Ctrl', 0)} / {n_by.get('ZeroK', 0)})",
               len(obs) == sum(N_CELLS.values()) and all(n_by.get(c) == n for c, n in N_CELLS.items()),
               "1431 (729 / 702)")
    both = bool((adata.var["n_cells_ctrl"].to_numpy() >= 2).all() and (adata.var["n_cells_zerok"].to_numpy() >= 2).all())
    checks.add("joint proteins: retained in both conditions (>= 2 cells each)", f"{adata.n_vars} proteins; {both}",
               adata.n_vars == N_PROTEINS and both, "5897; True")
    same_order = labels["cell_id"].tolist() == obs.index.tolist()
    checks.add("dct_labels canonical label CSV in object cell order", f"{same_order}", same_order)
    for col in ("cell_key", "condition", "plate", "dct_label", "dct_label_alternative"):
        eq = int((labels[col].astype(str).to_numpy() == obs[col].to_numpy()).sum())
        checks.add(f"object obs {col} equals the dct_labels canonical table", f"{eq}/{len(obs)}", eq == len(obs))
    for (col, cond), source in LABEL_SOURCES.items():
        m = (obs["condition"] == cond).to_numpy()
        eq = int((obs.loc[m, col].to_numpy() == labels.loc[m, source].astype(str).to_numpy()).sum())
        checks.add(f"{cond} {col} equals dct_labels {source}", f"{eq}/{int(m.sum())}", eq == int(m.sum()))
        counts = obs.loc[m, col].value_counts().to_dict()
        observed = {lab: int(counts.get(lab, 0)) for lab in LABELS}
        checks.add(f"{cond} {col} counts equal dct_labels", "; ".join(f"{k} {v}" for k, v in observed.items()),
                   observed == EXPECTED_COUNTS[(col, cond)] and set(counts) <= set(LABELS),
                   "; ".join(f"{k} {v}" for k, v in EXPECTED_COUNTS[(col, cond)].items()))

    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    mats = {}
    for variant in VARIANTS:
        m = cc.dense(adata.layers[t22.VARIANT_LAYER[variant]]).astype(np.float64)
        ok = bool((m[detected] > 0).all() and (m[~detected] == 0).all())
        checks.add(f"variant {variant}: detected > 0 and missing = 0", f"{ok}", ok)
        mats[variant] = m
    d_none = float(np.abs(mats["none"] - np.log1p(intensity)).max())
    checks.add("variant none equals ln(1 + intensity)", f"max abs diff {d_none:.2g}", d_none < 1e-4, "< 1e-4")
    n_det = detected.sum(axis=1)
    d_depth = int(np.abs(n_det - obs[DEPTH_COL].to_numpy(int)).max())
    checks.add(f"obs {DEPTH_COL} equals detected joint proteins per cell", f"max abs diff {d_depth}", d_depth == 0)
    pca = {v: np.asarray(adata.obsm[f"X_pca__{v}"], dtype=np.float64)[:, :t22.N_PCS] for v in VARIANTS}
    var = adata.var.copy()
    var.index = var.index.astype(str)
    var["match_level"] = "combined_search"
    var["protein_group_ctrl"] = var["protein_group"].astype(str)
    var["protein_group_zerok"] = var["protein_group"].astype(str)
    return Joint(obs=obs, var=var, detected=detected, mats=mats, pca=pca, var_names=np.asarray(var.index.astype(str)))


# ---------------------------------------------------------------------------
# Groups and labels absent in a condition
# ---------------------------------------------------------------------------
def group_counts(joint: Joint, mask: np.ndarray) -> tuple[int, int]:
    return int((mask & ~joint.is_zerok).sum()), int((mask & joint.is_zerok).sum())


def absent_reason(n_ctrl: int, n_zerok: int) -> str:
    """'' when both conditions have cells; otherwise why the comparison does not apply."""
    missing = [c for c, n in (("Ctrl", n_ctrl), ("ZeroK", n_zerok)) if n == 0]
    if not missing:
        return ""
    where = " and ".join(missing)
    return (f"{DOES_NOT_APPLY}: the label is absent in {where} under the frozen dct_labels rule "
            f"(Ctrl {n_ctrl}, ZeroK {n_zerok} cells)")


def split_groups(joint: Joint, label_column: str) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Label groups of one column (zerok_response_core D2): applicable masks and, for absent labels, the reason."""
    ok, absent = {}, {}
    for group, mask in t22.group_masks(joint.obs, label_column).items():
        why = absent_reason(*group_counts(joint, mask))
        if why:
            absent[group] = why
        else:
            ok[group] = mask
    return ok, absent


def all_groups(joint: Joint) -> tuple[list[tuple[str, str, np.ndarray]], list[tuple[str, str, str]]]:
    """(label column, group, mask) for both columns plus label-free all cells; and the absent groups."""
    ok, absent = [], []
    for col in LABEL_COLUMNS:
        o, a = split_groups(joint, col)
        ok.extend((col, g, m) for g, m in o.items())
        absent.extend((col, g, why) for g, why in a.items())
    ok.append(("label_free", ALL_CELLS, np.ones(len(joint.obs), bool)))
    return ok, absent


def absent_rows(joint: Joint, label_column: str, group: str, why: str, mask: np.ndarray | None = None,
                **extra) -> dict:
    if mask is None:
        mask = t22.group_masks(joint.obs, label_column)[group]
    n_c, n_z = group_counts(joint, mask)
    return {"label_column": label_column, "group": group, **extra, "n_ctrl": n_c, "n_zerok": n_z,
            "status": why, "p_value_note": P_NOTE}


def set_note(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the zerok_response_core p-value note (which names separate searches) by the zerok_vs_control note."""
    if "p_value_note" in frame:
        frame = frame.copy()
        frame["p_value_note"] = P_NOTE
    return frame


def ordered(frame: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    return t22.ordered(frame, first)
