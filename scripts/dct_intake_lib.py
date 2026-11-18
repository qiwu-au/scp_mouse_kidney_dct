"""dct_intake shared definitions and functions. Runs no analysis by itself.

Imported by the
``dct_intake_*.py`` step scripts and, later, by dct_labels-zerok_vs_control (object paths).

Contents
--------
- Design facts of the DCT data, with the combined Spectronaut search (authors,
  28 Oct and 2 Nov 2026), and the search settings the authors gave on 2 Nov 2026.
- Paths of the combined export, the freeze record and the dct_intake objects (kept
  here so that ``scp_common.py`` is not edited by this task).
- Strict reader for the combined export: the intake_core reader rules (missing values
  must be the literal ``NaN``; any other non-numeric token, zero, negative or
  infinite value is reported, and the step scripts stop on it), with the cells
  split by condition into ``intake_core_lib.Raw`` objects so that the intake_core
  filters and object builder apply unchanged.
- ``cell_key`` = ``<condition>_<plate>_<well>`` (cell matching across searches).
- A intake_core-style mapping frame (one row per protein group and condition) so that
  the intake_core and zerok_core coverage functions apply unchanged.

Frozen settings (dct_intake plan; intake_core and zerok_core manifests)
--------------------------------------------------
- Filters per condition: cells with >= 100 protein groups > 0 over all export
  rows, then protein groups > 0 in >= 2 retained cells of that condition.
- Objects: the intake_core layout (``intake_core_lib.build_object``).
- Joint object: protein groups retained in both conditions. One search, so a
  row of the export is the same protein group in both conditions; no matching
  between conditions is needed.
- Normalization variants, joint embedding and mixing: ``zerok_core_lib``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import intake_core_lib as t17

TASK = "dct_intake"
SEED = cc.RANDOM_SEED
CONDITIONS = t17.CONDITIONS                 # ("Ctrl", "ZeroK")
FILE_KEY = t17.FILE_KEY                     # {"Ctrl": "ctrl", "ZeroK": "zerok"}
PLATES = t17.PLATES
META_COLUMNS = t17.META_COLUMNS
QUANTITY_SUFFIX = t17.QUANTITY_SUFFIX
MISSING_TOKEN = t17.MISSING_TOKEN
WELLS_PER_PLATE = t17.WELLS_PER_PLATE
CONDITION_PALETTE = t17.CONDITION_PALETTE

COMBINED_INPUT = cc.DCT_DIR / "DCT_Ctrl_ZeroK_combine_1535cells.csv"
FREEZE_TABLE = cc.DATA_DIR / "input_checksums.csv"
PANEL_PATH = cc.out_path("zerok_core", "tables", "prespecified_panel", ".csv")
PANEL_SHA256 = "c2fd78fc106d8e1334211b6a2b883aedd6bb606d34e3d10b2ee7533a5505e88a"
MARKER_WORKBOOK = cc.DCT_DIR / "DCT_markers_V1.xlsx"
OBJECTS = {
    "Ctrl": cc.OUT_ROOTS["h5ad"] / "dct_intake_DCT_ctrl.h5ad",
    "ZeroK": cc.OUT_ROOTS["h5ad"] / "dct_intake_DCT_zerok.h5ad",
    "joint": cc.OUT_ROOTS["h5ad"] / "dct_intake_DCT_joint.h5ad",
}

# Plan (header read before the freeze, 2 Nov 2026) and the intake_core cell set.
EXPECTED = {
    "n_cells": 1535,
    "n_protein_groups": 6208,
    "conditions": {cond: {"n_cells": t17.EXPECTED[cond]["n_cells"], "plates": dict(t17.EXPECTED[cond]["plates"]),
                          "missing_wells": list(t17.EXPECTED[cond]["missing_wells"])} for cond in CONDITIONS},
}

DESIGN_FACTS = [
    *t17.DESIGN_FACTS[:3],
    "Both conditions were searched together in one Spectronaut search (combined export, 2 Nov 2026). Protein "
    "inference, FDR filtering and cross-run normalization ran over all 1,535 runs, so a row of the export is the "
    "same protein group in both conditions.",
    "The Ctrl and ZeroK objects are the cells of each condition taken from this one search and filtered and scaled "
    "on their own. Their intensities still carry Spectronaut's cross-run normalization over all 1,535 runs "
    "(authors, 2 Nov 2026: control-only analyses use these Ctrl cells, never data/DCT/DCT_Ctrl_768cells.csv).",
]
ONE_SEARCH_NOTE = "one combined Spectronaut search; cross-run normalization over all 1,535 runs"

# Search settings (authors, 2 Nov 2026). The authors stated that the separate searches used the same settings.
SETTINGS_SOURCE = "authors, 2 Nov 2026"
SEARCH_SETTINGS = [
    ("Spectronaut version", "21.1.260825.94842"),
    ("FASTA", "Uniprot_mouse_reviewed_20261013.fasta"),
    ("library-free or library", "library-free (directDIA)"),
    ("precursor FDR", "1%"),
    ("protein FDR", "1%"),
    ("protein inference", "automatic, IDPicker"),
    ("cross-run normalization", "automatic"),
    ("imputation", "no"),
]
SEPARATE_SAME_NOTE = "same as the combined search (authors, 2 Nov 2026: 'Same as in separate searches')"

OBS_COLUMNS = ["condition", "plate", "well", "file_column_index", "cell_key"]


def cell_key(condition: str, plate: str, well: str) -> str:
    """``<condition>_<plate>_<well>``: matches cells across searches ([n] changes between exports)."""
    return f"{condition}_{plate}_{str(well).zfill(4)}"


def add_cell_key(frame: pd.DataFrame) -> pd.Series:
    return pd.Series([cell_key(c, p, w) for c, p, w in zip(frame["condition"], frame["plate"], frame["well"])],
                     index=frame.index, name="cell_key")


# ---------------------------------------------------------------------------
# Reading (intake_core reader rules on one file holding both conditions)
# ---------------------------------------------------------------------------
class Combined:
    """The combined export as read, before any filter, with per-condition views."""

    def __init__(self, path: Path, header: list[str], cells: pd.DataFrame, meta: pd.DataFrame,
                 values: np.ndarray, nonnumeric_per_cell: np.ndarray, nonnumeric_examples: str):
        self.path, self.header, self.cells, self.meta, self.values = path, header, cells, meta, values
        self.nonnumeric_per_cell = nonnumeric_per_cell
        self.nonnumeric_examples = nonnumeric_examples

    def value_report(self, mask: np.ndarray | None = None) -> dict:
        """The intake_core value report over all cells, or over the cells in ``mask``."""
        values = self.values if mask is None else self.values[mask]
        n_nonnum = int(self.nonnumeric_per_cell.sum() if mask is None else self.nonnumeric_per_cell[mask].sum())
        finite = np.isfinite(values)
        positive = finite & (values > 0)
        return {
            "n_values": int(values.size),
            "n_missing_NaN": int(np.isnan(values).sum() - n_nonnum),
            "n_nonnumeric": n_nonnum,
            "nonnumeric_examples": self.nonnumeric_examples if n_nonnum else "",
            "n_zero": int((values == 0).sum()),
            "n_negative": int((values < 0).sum()),
            "n_infinite": int(np.isinf(values).sum()),
            "min_positive": float(values[positive].min()) if positive.any() else np.nan,
            "max": float(values[finite].max()) if finite.any() else np.nan,
        }

    def raw(self, condition: str) -> t17.Raw:
        """One condition as a ``intake_core_lib.Raw`` (cells in file order; all protein rows)."""
        mask = (self.cells["condition"] == condition).to_numpy()
        return t17.Raw(condition, self.path, self.header, self.cells.loc[mask].copy(), self.meta,
                       self.values[mask], self.value_report(mask))


def read_combined(run: cc.Run | None = None) -> Combined:
    """Read the combined export; cell IDs must parse, missing values must be the literal 'NaN'."""
    path = COMBINED_INPUT
    if run is not None:
        run.use_input(path, "raw Spectronaut PG.Quantity, combined search (Ctrl + ZeroK)")
    header = list(pd.read_csv(path, nrows=0).columns.astype(str))
    if tuple(header[:3]) != META_COLUMNS:
        raise ValueError(f"first three columns are {header[:3]}, expected {list(META_COLUMNS)}")
    cell_cols = header[3:]
    parsed = [cc.parse_dct_cell_id(c) for c in cell_cols]   # raises on an unparseable header
    cells = pd.DataFrame(parsed, index=pd.Index(cell_cols, name="cell_id"))
    cells["cell_key"] = add_cell_key(cells)

    data = pd.read_csv(path, keep_default_na=False, na_values={c: [MISSING_TOKEN] for c in cell_cols},
                       dtype={c: str for c in META_COLUMNS}, low_memory=False)
    nonnum = np.zeros(len(cell_cols), dtype=np.int64)
    examples: list[str] = []
    numeric = {}
    for i, col in enumerate(cell_cols):
        series = data[col]
        if series.dtype.kind != "f":
            coerced = pd.to_numeric(series, errors="coerce")
            bad = coerced.isna() & series.notna()
            nonnum[i] = int(bad.sum())
            examples.extend(series[bad].astype(str).unique()[:3].tolist())
            series = coerced
        numeric[col] = series.to_numpy(dtype=np.float64)
    values = np.column_stack([numeric[c] for c in cell_cols]).T   # cells x proteins
    pg = data["PG.ProteinGroups"].astype(str).tolist()
    genes = data["PG.Genes"].astype(str).tolist()
    gene_names = [cc.clean_gene_symbol(g) for g in genes]
    meta = pd.DataFrame({
        "file_row": np.arange(len(data)),
        "protein_group": pg,
        "pg_genes": genes,
        "protein_description": data["PG.ProteinDescriptions"].astype(str).tolist(),
        "gene_name": gene_names,
        "var_name": t17.protein_names(gene_names, pg),
    })
    if meta["protein_group"].duplicated().any():
        raise ValueError("duplicated PG.ProteinGroups strings in the combined export")
    if run is not None:
        run.note(f"Read {cc.rel(path)}: {len(cells)} cells x {len(meta)} protein groups.")
    return Combined(path, header, cells, meta, values, nonnum, ";".join(dict.fromkeys(examples))[:200])


# ---------------------------------------------------------------------------
# Protein table and the intake_core-style mapping frame
# ---------------------------------------------------------------------------
def protein_table(meta: pd.DataFrame, filts: dict[str, t17.Filtered]) -> pd.DataFrame:
    """One row per protein group of the export: detection and retention in each condition."""
    frame = meta[["file_row", "protein_group", "pg_genes", "gene_name", "var_name", "protein_description"]].copy()
    for cond in CONDITIONS:
        k = FILE_KEY[cond]
        frame[f"n_cells_all_{k}"] = filts[cond].n_cells_all
        frame[f"n_cells_{k}"] = filts[cond].n_cells_retained
        frame[f"retained_{k}"] = filts[cond].keep_proteins
    frame["in_joint"] = frame["retained_ctrl"] & frame["retained_zerok"]
    frame["retention"] = np.select(
        [frame["in_joint"], frame["retained_ctrl"], frame["retained_zerok"]],
        ["both conditions", "Ctrl only", "ZeroK only"], default="neither")
    return frame


def mapping_frame(proteins: pd.DataFrame) -> pd.DataFrame:
    """intake_core ``protein_mapping`` layout (one row per group and condition) for the intake_core/zerok_core coverage functions.

    In one search both rows of a group are the same protein group: match status 'combined_search'.
    """
    frames = []
    for cond in CONDITIONS:
        frame = proteins[["file_row", "protein_group", "pg_genes", "gene_name", "var_name",
                          "n_cells_ctrl", "n_cells_zerok", "retained_ctrl", "retained_zerok", "in_joint"]].copy()
        frame.insert(0, "search", cond)
        frame["match_status"] = "combined_search"
        frame["match_level"] = "one_search"
        frame["joint_var_name"] = np.where(frame["in_joint"], frame["var_name"], "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the dct_intake step that writes it first.")
    return pd.read_csv(run.use_input(path, f"dct_intake table {name}"), **kwargs)


def read_object(run: cc.Run, key: str):
    import anndata as ad

    path = OBJECTS[key]
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run dct_intake_objects.py first.")
    return ad.read_h5ad(run.use_input(path, f"dct_intake object ({key})"))


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def read_proteins(run: cc.Run) -> pd.DataFrame:
    frame = read_table(run, "proteins", keep_default_na=False)
    for col in ("retained_ctrl", "retained_zerok", "in_joint"):
        frame[col] = as_bool(frame[col])
    return frame


def read_cells(run: cc.Run) -> pd.DataFrame:
    frame = read_table(run, "dct_cells", dtype={"well": str})
    frame["retained"] = as_bool(frame["retained"])
    return frame
