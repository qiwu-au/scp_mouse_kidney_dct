"""intake_core shared definitions and functions. Runs no analysis by itself.

Imported by the
``intake_core_*.py`` step scripts.

Contents
--------
- Design facts and expected shapes of the new DCT data (authors, 28 Oct 2026).
- Strict reader for the two Spectronaut exports: missing values must be the
  literal ``NaN``; any other non-numeric token, zero, negative or infinite
  value is reported (the step scripts stop on it).
- Protein names, the submitted filters (``scp_common``) and the two-level
  protein-group matching between the separate searches.
- Marker panels for coverage (contamination, zero-K candidates, cell cycle).
- Object builder with the kidney_object layout.
- Figure style and helpers reused from qc_mito_lib.py.

Matching rule (authors decision, 28 Oct 2026)
------------------------------------------
- Level 1: identical ``PG.ProteinGroups`` string; always kept.
- Level 2: same cleaned first gene symbol, accepted only when that gene is the
  first gene of exactly one still-unmatched group in each search and has no
  level-1 match.
- Any other gene with unmatched groups in both searches is ambiguous; those
  groups are listed and left out of the joint object.
- Joint object: matched pairs retained (>= 2 retained cells) in both conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
from reviewer_core_lib import CELL_CYCLE_MARKERS, TIROSH_ALIASES, TIROSH_G2M, TIROSH_S
from qc_mito_lib import (INK, SERIES, annotate_proteins, apply_style, box_strip, footer, read_mitocarta,
                            sequential_cmap, split_symbols, ygrid)

__all__ = ["INK", "SERIES", "annotate_proteins", "apply_style", "box_strip", "footer", "read_mitocarta",
           "sequential_cmap", "split_symbols", "ygrid", "CELL_CYCLE_MARKERS", "TIROSH_S", "TIROSH_G2M",
           "TIROSH_ALIASES"]

TASK = "intake_core"
SEED = cc.RANDOM_SEED
CONDITIONS = ("Ctrl", "ZeroK")
FILE_KEY = {"Ctrl": "ctrl", "ZeroK": "zerok"}
PLATES = ("P1", "P2")
META_COLUMNS = ("PG.ProteinGroups", "PG.Genes", "PG.ProteinDescriptions")
QUANTITY_SUFFIX = ".PG.Quantity"
MISSING_TOKEN = "NaN"
WELLS_PER_PLATE = 384
PLATE_ROWS, PLATE_COLS = 16, 24

# Plan and file names (authors, 28 Oct 2026).
EXPECTED = {
    "Ctrl": {"n_cells": 768, "n_protein_groups": 6257, "plates": {"P1": 384, "P2": 384}, "missing_wells": []},
    "ZeroK": {"n_cells": 767, "n_protein_groups": 5583, "plates": {"P1": 383, "P2": 384},
              "missing_wells": ["P1_0368"]},
}
DESIGN_FACTS = [
    "Control diet vs zero-K diet for 4 days. One pool of 3 mice per condition, so condition and pool cannot "
    "be separated and cells are not animal-level replicates.",
    "Cells were sorted onto two plates per condition (P1, P2). Plates are sort plates of the pooled cells: a "
    "technical label, never a biological replicate.",
    "Control and zero-K cells were sorted and acquired in the same sessions.",
    "The two conditions were searched separately in Spectronaut. Any cross-run normalization was applied "
    "within each condition only, protein grouping can differ between the files, and intensity scales are not "
    "aligned between conditions.",
]
# Categorical slots 3-4 (slots 1-2 are the plate colours, cc.PLATE_PALETTE).
CONDITION_PALETTE = {"Ctrl": SERIES[2], "ZeroK": SERIES[3]}
CROSS_CONDITION_NOTE = "separate Spectronaut searches: values are not directly comparable between conditions"

# ---------------------------------------------------------------------------
# Marker panels (coverage only; plan step 6)
# ---------------------------------------------------------------------------
CONTAMINATION_PANEL = {
    "TAL": ["Umod", "Slc12a1"],
    "CNT/PC": ["Aqp2", "Calb1", "Scnn1a", "Scnn1b", "Scnn1g"],
    "IC": ["Atp6v1b1", "Slc4a1", "Slc26a4"],
    "PT": ["Lrp2", "Slc34a1"],
    "Endothelial": ["Pecam1", "Emcn"],
    "Immune": ["Ptprc"],
}
# Group names are protein names only; the pre-specified response is defined in zerok_core.
ZEROK_PANEL = {
    "Slc12a3": ["Slc12a3"],
    "Wnk1/Wnk4/Stk39/Oxsr1/Klhl3/Cul3": ["Wnk1", "Wnk4", "Stk39", "Oxsr1", "Klhl3", "Cul3"],
    "Kcnj10/Kcnj16": ["Kcnj10", "Kcnj16"],
    "Atp1a1/Atp1b1/Fxyd2": ["Atp1a1", "Atp1b1", "Fxyd2"],
    "Pvalb/Calb1/Trpm6/Egf": ["Pvalb", "Calb1", "Trpm6", "Egf"],
    "Slc8a1/Atp2b1": ["Slc8a1", "Atp2b1"],
    "Mki67/Top2a/Pcna/Mcm2": ["Mki67", "Top2a", "Pcna", "Mcm2"],
}
CELL_CYCLE_PANELS = {
    "original_18_panel (scripts/00_config.py)": list(CELL_CYCLE_MARKERS),
    "Tirosh_S": list(TIROSH_S),
    "Tirosh_G2M": list(TIROSH_G2M),
}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
@dataclass
class Raw:
    """One Spectronaut export as read, before any filter."""

    condition: str
    path: Path
    header: list[str]          # all CSV column names in file order
    cells: pd.DataFrame        # index cell_id (column name unchanged): file_column_index, condition, plate, well
    meta: pd.DataFrame         # one row per protein group in file order (index = file_row, 0-based)
    values: np.ndarray         # cells x protein groups, float64; NaN = missing as exported
    value_report: dict


def first_accession(protein_group: object) -> str:
    return str(protein_group).split(";")[0].strip()


def protein_names(gene_names: list[str], protein_groups: list[str]) -> list[str]:
    """Unique protein names: cleaned first gene symbol (first accession when PG.Genes is empty);
    a repeated name gets a running count (``Name_1``, ``Name_2``) in file order."""
    base = [g if g != "Nan" else first_accession(pg) for g, pg in zip(gene_names, protein_groups)]
    seen: dict[str, int] = {}
    names = []
    for name in base:
        count = seen.get(name, 0)
        names.append(name if count == 0 else f"{name}_{count}")
        seen[name] = count + 1
    if len(set(names)) != len(names):
        raise ValueError("protein names are not unique after suffixing")
    return names


def read_raw(condition: str, run: cc.Run | None = None) -> Raw:
    """Read one export; cell IDs must parse, missing values must be the literal 'NaN'."""
    path = cc.DCT_NEW_INPUTS[condition]
    if run is not None:
        run.use_input(path, f"raw Spectronaut PG.Quantity, new DCT data ({condition})")
    header = list(pd.read_csv(path, nrows=0).columns.astype(str))
    if tuple(header[:3]) != META_COLUMNS:
        raise ValueError(f"{condition}: first three columns are {header[:3]}, expected {list(META_COLUMNS)}")
    cell_cols = header[3:]
    parsed = [cc.parse_dct_cell_id(c) for c in cell_cols]   # raises on an unparseable header
    bad_cond = [c for c, p in zip(cell_cols, parsed) if p["condition"] != condition]
    if bad_cond:
        raise ValueError(f"{condition}: {len(bad_cond)} cell columns carry another condition, e.g. {bad_cond[0]}")
    cells = pd.DataFrame(parsed, index=pd.Index(cell_cols, name="cell_id"))

    data = pd.read_csv(path, keep_default_na=False, na_values={c: [MISSING_TOKEN] for c in cell_cols},
                       dtype={c: str for c in META_COLUMNS}, low_memory=False)
    report = {"n_values": 0, "n_missing_NaN": 0, "n_nonnumeric": 0, "nonnumeric_examples": "",
              "n_zero": 0, "n_negative": 0, "n_infinite": 0}
    examples: list[str] = []
    numeric = {}
    for col in cell_cols:
        series = data[col]
        if series.dtype.kind != "f":
            coerced = pd.to_numeric(series, errors="coerce")
            bad = coerced.isna() & series.notna()
            report["n_nonnumeric"] += int(bad.sum())
            examples.extend(series[bad].astype(str).unique()[:3].tolist())
            series = coerced
        numeric[col] = series.to_numpy(dtype=np.float64)
    values = np.column_stack([numeric[c] for c in cell_cols]).T   # cells x proteins
    finite = np.isfinite(values)
    report.update(
        n_values=int(values.size),
        n_missing_NaN=int(np.isnan(values).sum() - report["n_nonnumeric"]),
        nonnumeric_examples=";".join(dict.fromkeys(examples))[:200],
        n_zero=int((values == 0).sum()),
        n_negative=int((values < 0).sum()),
        n_infinite=int(np.isinf(values).sum()),
        min_positive=float(values[finite & (values > 0)].min()) if (finite & (values > 0)).any() else np.nan,
        max=float(values[finite].max()) if finite.any() else np.nan,
    )
    pg = data["PG.ProteinGroups"].astype(str).tolist()
    genes = data["PG.Genes"].astype(str).tolist()
    gene_names = [cc.clean_gene_symbol(g) for g in genes]
    meta = pd.DataFrame({
        "file_row": np.arange(len(data)),
        "protein_group": pg,
        "pg_genes": genes,
        "protein_description": data["PG.ProteinDescriptions"].astype(str).tolist(),
        "gene_name": gene_names,
        "var_name": protein_names(gene_names, pg),
    })
    if meta["protein_group"].duplicated().any():
        raise ValueError(f"{condition}: duplicated PG.ProteinGroups strings")
    if run is not None:
        run.note(f"Read {cc.rel(path)}: {len(cells)} cells x {len(meta)} protein groups.")
    return Raw(condition, path, header, cells, meta, values, report)


# ---------------------------------------------------------------------------
# Filters (submitted thresholds, scp_common)
# ---------------------------------------------------------------------------
@dataclass
class Filtered:
    keep_cells: np.ndarray           # bool over cells: >= MIN_CELL_PROTEINS groups > 0 (all rows)
    keep_proteins: np.ndarray        # bool over groups: > 0 in >= MIN_PROTEIN_CELLS retained cells
    n_detected_all_rows: np.ndarray  # per cell, groups > 0 over all rows
    n_cells_retained: np.ndarray     # per group, retained cells with intensity > 0
    n_cells_all: np.ndarray          # per group, all cells with intensity > 0


def apply_filters(raw: Raw) -> Filtered:
    detected = np.nan_to_num(raw.values, nan=0.0) > 0
    n_all = detected.sum(axis=1)
    keep_cells = n_all >= cc.MIN_CELL_PROTEINS
    n_cells = detected[keep_cells].sum(axis=0)
    keep_proteins = n_cells >= cc.MIN_PROTEIN_CELLS
    return Filtered(keep_cells, keep_proteins, n_all, n_cells, detected.sum(axis=0))


def intensity_matrix(raw: Raw, filt: Filtered) -> np.ndarray:
    """Retained cells x retained groups, float32, missing = 0 (as the kidney_object objects)."""
    sub = raw.values[np.ix_(filt.keep_cells, filt.keep_proteins)]
    return np.nan_to_num(sub, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Matching between the two searches
# ---------------------------------------------------------------------------
def match_protein_groups(meta: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Two-level matching (see module docstring).

    Returns ``pairs`` (one row per matched pair, Ctrl file order), ``status``
    (per condition: one row per group with match_status, match_level,
    partner_file_row, match_reason) and ``genes`` (every gene that is the first
    gene of more than one group in either search, or that was ambiguous).
    """
    c, z = meta["Ctrl"], meta["ZeroK"]
    status = {cond: pd.DataFrame({"match_status": "", "match_level": 0, "partner_file_row": -1,
                                  "match_reason": ""}, index=meta[cond].index) for cond in CONDITIONS}
    z_by_pg = pd.Series(z.index, index=z["protein_group"].to_numpy())
    partner = c["protein_group"].map(z_by_pg)
    ci = c.index[partner.notna()].to_numpy()
    zj = partner[partner.notna()].astype(int).to_numpy()
    for cond, own, other in (("Ctrl", ci, zj), ("ZeroK", zj, ci)):
        status[cond].loc[own, "match_status"] = "level1"
        status[cond].loc[own, "match_level"] = 1
        status[cond].loc[own, "partner_file_row"] = other
        status[cond].loc[own, "match_reason"] = "identical PG.ProteinGroups"
    pairs = [(1, int(i), int(j)) for i, j in zip(ci, zj)]

    groups = {cond: meta[cond].loc[meta[cond]["gene_name"] != "Nan"].groupby("gene_name").groups
              for cond in CONDITIONS}
    gene_rows = []
    for gene in sorted(set(groups["Ctrl"]) | set(groups["ZeroK"])):
        rows = {cond: list(groups[cond].get(gene, [])) for cond in CONDITIONS}
        level1 = [i for i in rows["Ctrl"] if status["Ctrl"].at[i, "match_status"] == "level1"]
        unmatched = {cond: [i for i in rows[cond] if status[cond].at[i, "match_status"] == ""] for cond in CONDITIONS}
        handling = ""
        if len(unmatched["Ctrl"]) == 1 and len(unmatched["ZeroK"]) == 1 and not level1:
            i, j = unmatched["Ctrl"][0], unmatched["ZeroK"][0]
            pairs.append((2, i, j))
            reason = "same gene; one unmatched group in each search"
            status["Ctrl"].loc[i, ["match_status", "match_level", "partner_file_row", "match_reason"]] = \
                ["level2", 2, j, reason]
            status["ZeroK"].loc[j, ["match_status", "match_level", "partner_file_row", "match_reason"]] = \
                ["level2", 2, i, reason]
            handling = "level2_match"
        elif unmatched["Ctrl"] and unmatched["ZeroK"]:
            reason = (f"gene {gene}: {len(unmatched['Ctrl'])} unmatched Ctrl group(s), "
                      f"{len(unmatched['ZeroK'])} unmatched ZeroK group(s), {len(level1)} level-1 match(es)")
            for cond in CONDITIONS:
                for i in unmatched[cond]:
                    status[cond].loc[i, ["match_status", "match_reason"]] = ["ambiguous_gene", reason]
            handling = "ambiguous_excluded"
        multi = len(rows["Ctrl"]) > 1 or len(rows["ZeroK"]) > 1
        if multi or handling == "ambiguous_excluded":
            gene_rows.append({
                "gene_name": gene,
                "n_groups_ctrl": len(rows["Ctrl"]), "n_groups_zerok": len(rows["ZeroK"]),
                "n_level1_pairs": len(level1),
                "n_unmatched_ctrl_before_level2": len(unmatched["Ctrl"]),
                "n_unmatched_zerok_before_level2": len(unmatched["ZeroK"]),
                "handling": handling or ("level1_only" if level1 and not (unmatched["Ctrl"] or unmatched["ZeroK"])
                                         else "unmatched_groups_in_one_search_only"),
                "groups_ctrl": " | ".join(c.loc[rows["Ctrl"], "protein_group"]),
                "groups_zerok": " | ".join(z.loc[rows["ZeroK"], "protein_group"]),
            })

    matched_genes = {cond: set(meta[cond].loc[status[cond]["match_status"].isin(["level1", "level2"]), "gene_name"])
                     for cond in CONDITIONS}
    for cond, other in (("Ctrl", "ZeroK"), ("ZeroK", "Ctrl")):
        left = status[cond]["match_status"] == ""
        genes = meta[cond]["gene_name"]
        reason = np.where(genes == "Nan", "no PG.Genes symbol; no identical group string in the other search",
                          np.where(genes.isin(meta[other]["gene_name"]),
                                   np.where(genes.isin(matched_genes[cond]),
                                            "other group(s) of this gene matched; no unmatched partner",
                                            "gene in the other search only as matched group(s)"),
                                   "gene not in the other search"))
        status[cond].loc[left, "match_status"] = "one_search_only"
        status[cond].loc[left, "match_reason"] = reason[left.to_numpy()]

    pair_frame = pd.DataFrame(pairs, columns=["match_level", "file_row_ctrl", "file_row_zerok"])
    pair_frame = pair_frame.sort_values("file_row_ctrl").reset_index(drop=True)
    for cond, col in (("Ctrl", "file_row_ctrl"), ("ZeroK", "file_row_zerok")):
        suffix = FILE_KEY[cond]
        sub = meta[cond].loc[pair_frame[col]]
        for field in ("protein_group", "pg_genes", "gene_name", "var_name", "protein_description"):
            pair_frame[f"{field}_{suffix}"] = sub[field].to_numpy()
    multi_genes = {g["gene_name"] for g in gene_rows if g["n_groups_ctrl"] > 1 or g["n_groups_zerok"] > 1}
    pair_frame["gene_multi_group"] = (pair_frame["gene_name_ctrl"].isin(multi_genes)
                                      | pair_frame["gene_name_zerok"].isin(multi_genes))
    gene_frame = pd.DataFrame(gene_rows)
    if not gene_frame.empty:
        gene_frame["gene_multi_group"] = gene_frame["gene_name"].isin(multi_genes)
    return pair_frame, status, gene_frame


def joint_names(pairs: pd.DataFrame) -> list[str]:
    """Joint protein names: Ctrl gene symbol (first accession if none), running count for repeats."""
    return protein_names(pairs["gene_name_ctrl"].tolist(), pairs["protein_group_ctrl"].tolist())


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------
def build_object(intensity: np.ndarray, obs: pd.DataFrame, var: pd.DataFrame, provenance: dict):
    """AnnData with the kidney_object layout: layers intensity (missing = 0) and ln1p_intensity; .X scaled."""
    import anndata as ad

    intensity = np.asarray(intensity, dtype=np.float32)
    ln1p = np.log1p(intensity)
    adata = ad.AnnData(X=cc.scale_like_scanpy(ln1p).astype(np.float32), obs=obs, var=var)
    adata.layers["intensity"] = intensity
    adata.layers["ln1p_intensity"] = ln1p
    adata.obs["condition"] = pd.Categorical(adata.obs["condition"].astype(str),
                                            categories=[c for c in CONDITIONS if c in set(adata.obs["condition"])])
    adata.obs["plate"] = pd.Categorical(adata.obs["plate"].astype(str), categories=list(PLATES))
    adata.uns["condition_colors"] = [CONDITION_PALETTE[c] for c in adata.obs["condition"].cat.categories]
    adata.uns["plate_colors"] = [cc.PLATE_PALETTE[p] for p in adata.obs["plate"].cat.categories]
    adata.uns["intake_core"] = {k: (v if isinstance(v, (str, int, float)) else str(v)) for k, v in provenance.items()}
    return adata


def well_position(well: str) -> tuple[int, int]:
    """Plate row (0 = A) and column (0 = 1) of well 0001-0384; wells run row by row, A1-A24 then B1 (authors, 28 Oct 2026)."""
    idx = int(well) - 1
    return idx // PLATE_COLS, idx % PLATE_COLS


def read_t17_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = cc.out_path(TASK, "tables", name, ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the earlier intake_core steps first.")
    return pd.read_csv(run.use_input(path, f"intake_core table {name}"), **kwargs)


def read_t17_object(run: cc.Run, key: str):
    import anndata as ad

    path = cc.DCT_NEW_OBJECTS[key]
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run intake_core_matching.py first.")
    return ad.read_h5ad(run.use_input(path, f"intake_core object ({key})"))
