#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-01-a (cell and protein filtering per dataset), one workbook.

The authors asked on 12 Nov 2026 to assemble Supplementary
Tables S-01-a, S-02-d and S-03-b ("Yes, assemble them"). The response to the reviewers, section 5, fixes the content of S-01-a: cells and
protein groups before and after filtering, the protein range of the removed cells and the proteins per retained
cell, for the total-kidney data and for the DCT data per diet condition and sort plate; the per-cell lists of the
removed cells are the optional part of that plan (sheet 2). No new statistics: every value is copied from the
stored tables.

What the assembly changes (all recorded per table in the sources table):
- rows: the kidney rows of the two kidney_object tables (their DCT rows are the retired 758-cell data); the dct_intake tables
  (combined control + zero-K search) are taken whole;
- columns: `dataset` is dropped after a check that it is constant; `cell_key` and `file_column_index` of the DCT
  list are dropped (they repeat the condition, plate, well and the number in the run name; checked);
- names: Ctrl / ZeroK -> Control / Zero-K, reader-facing names for the filtering stages, `protein_ids_*` ->
  `proteins_per_cell_*`, `plate` -> `sort_plate`.

Workbook: results/tables/report/report_01_S01a_tables.xlsx (sheet Contents, then sheets 1-2), in the
layout of Supplementary Table S-07-d (shared code: report_table_lib.py).

Inputs (read-only): the five kidney_object / dct_intake tables named in SPECS.
Outputs: the workbook; results/tables/report/report_01_s01a_table_{sources,checks}.csv;
         results/tables/report/report_step_01_s01a_table_manifest.json

Command (project root):  python scripts/report_filtering_table.py --overwrite
"""

from __future__ import annotations
# Select the project environment before importing analysis libraries.
if __name__ == "__main__":
    import sys as _runtime_sys
    from pathlib import Path as _RuntimePath
    _runtime_root = next(
        parent for parent in _RuntimePath(__file__).resolve().parents
        if (parent / "_scanpy_runtime.py").is_file()
    )
    _runtime_sys.path.insert(0, str(_runtime_root))
    from _scanpy_runtime import ensure_scanpy_environment as _select_project_python
    _select_project_python(__file__)


import argparse

import scp_common as cc
import report_table_lib as lib
from report_table_lib import CONDITION, Spec, WorkbookDef, _pick

STEP = "01_s01a_table"
KIDNEY_OBJECT = "results/tables/kidney_object/kidney_object_"
DCT_INTAKE = "results/tables/dct_intake/dct_intake_"
ALL, REMOVED, RETAINED_ALL, RETAINED = ("all cells", "removed cells (fewer than 100 proteins)",
                                        "retained cells, all protein groups",
                                        "retained cells, retained protein groups (final matrix)")
STAGE = {"raw_all_cells": ALL, "removed_cells": REMOVED, "retained_cells_all_proteins": RETAINED_ALL,
         "retained_cells_retained_proteins": RETAINED}
PER_CELL = ((r"^protein_ids_", "proteins_per_cell_"),)

SHEETS = {
    1: ("1 Filtering summary", "Cells and protein groups before and after filtering, and proteins per cell", ()),
    2: ("2 Removed cells", "Cells removed by the filter (fewer than 100 detected proteins)", ()),
}

SPECS = [
    Spec("1a", KIDNEY_OBJECT + "filtering_summary.csv", "Total kidney: cells and protein groups at each filtering stage",
         ("One row per stage; the last row is the final matrix (712 cells, 4,054 protein groups).",
          "proteins_per_cell_ columns: detected proteins per cell over the protein groups named in the column "
          "definition."),
         rows=(("dataset", ("KIDNEY",)),), drop_const=("dataset",), values={"stage": STAGE}, strict=("stage",),
         rename_re=PER_CELL,
         contents_note="Quartiles of the 712 retained kidney cells: Supplementary Table S-02-d, table 2b (all "
                       "cells, n_detected_proteins).",
         assembly_note="The plan asks for min, quartiles, median and max; the kidney source table holds no "
                       "quartiles, so table 1a has none (q25 357 and q75 1,098 are in S-02-d, table 2b)."),
    Spec("1b", DCT_INTAKE + "filtering_summary.csv",
         "DCT, Control and Zero-K: cells and protein groups at each filtering stage, all cells and per sort plate",
         ("Each condition was filtered on its own. sort_plate all: both sort plates of the condition.",
          "Zero-K, plate P1: 383 cells, because one well is absent from the export."),
         values={"condition": CONDITION, "stage": STAGE}, strict=("condition", "stage"),
         rename={"plate": "sort_plate"}, rename_re=PER_CELL),
    Spec("1c", DCT_INTAKE + "proteins_per_cell_range.csv",
         "DCT, Control and Zero-K: proteins per retained cell, per condition and sort plate",
         ("Detected proteins per retained cell over the retained protein groups of the condition "
          "(n_retained_proteins_condition).",),
         values={"condition": CONDITION}, strict=("condition",), rename={"plate": "sort_plate"}, rename_re=PER_CELL),
    Spec("2a", KIDNEY_OBJECT + "filtering_removed_cells.csv", "Total kidney: removed cells",
         ("proteins_detected: detected proteins over all protein groups of the export.",),
         rows=(("dataset", ("KIDNEY",)),), drop_const=("dataset",), rename={"protein_ids": "proteins_detected"},
         assembly_note="Optional part of the plan (per-cell list)."),
    Spec("2b", DCT_INTAKE + "filtering_removed_cells.csv", "DCT, Control and Zero-K: removed cells",
         ("proteins_detected: detected proteins over all protein groups of the export. well: position on the sort "
          "plate (1 to 384).",),
         values={"condition": CONDITION}, strict=("condition",), drop=("cell_key", "file_column_index"),
         drop_reason="cell_key repeats condition, plate and well; file_column_index is the number in brackets in "
                     "cell_id (both checked)",
         rename={"plate": "sort_plate", "n_detected_all_rows": "proteins_detected"},
         assembly_note="Optional part of the plan (per-cell list)."),
]

TERMS = [
    ("Datasets", "Total kidney: 768 randomly sorted kidney cells. DCT: sorted DCT cells of mice fed a control or a "
     "zero-potassium (Zero-K) diet for 4 days; one pool of 3 mice per condition; the control and zero-K runs were "
     "searched together in one Spectronaut search (6,208 protein groups)."),
    ("protein group; detected protein", "A protein group of the Spectronaut export. A protein is detected in a cell "
     "if its intensity is above zero."),
    ("Filters", "Applied in this order: cells with fewer than 100 detected proteins were removed; then protein "
     "groups detected in fewer than two of the retained cells were removed."),
    ("stage", "all cells: all cells of the export. removed cells: cells with fewer than 100 detected proteins. "
     "retained cells, all protein groups: after the cell filter. retained cells, retained protein groups: after "
     "both filters (the matrix used for all analyses)."),
    ("n_protein_groups", "Protein groups in the matrix at this stage. n_protein_groups_detected_in_any_cell: of "
     "these, the protein groups detected in at least one cell of the row."),
    ("proteins_per_cell_", "Number of detected proteins per cell: minimum (min), quartiles (q25, q75), median, "
     "maximum (max) and mean. For the removed cells this is the range of the removed cells."),
    ("condition; sort_plate", "Control, Zero-K: diet condition. P1, P2: the two 384-well sort plates of the pooled "
     "cells of a condition; they are technical plates, not animals or replicates."),
    ("cell_id", "Name of the run in the Spectronaut export."),
]


def on(condition: str, plate: str = "all") -> dict:
    return {"condition": condition, "sort_plate": plate}


LETTER = [
    ("kidney: cells in the export", "1a", lambda f: _pick(f, "n_cells", stage=ALL), 768, None),
    ("kidney: retained cells", "1a", lambda f: _pick(f, "n_cells", stage=RETAINED), 712, None),
    ("kidney: protein groups in the export", "1a", lambda f: _pick(f, "n_protein_groups", stage=ALL), 4144, None),
    ("kidney: retained protein groups", "1a", lambda f: _pick(f, "n_protein_groups", stage=RETAINED), 4054, None),
    ("kidney: fewest proteins per retained cell", "1a", lambda f: _pick(f, "proteins_per_cell_min", stage=RETAINED),
     101, None),
    ("kidney: most proteins per retained cell", "1a", lambda f: _pick(f, "proteins_per_cell_max", stage=RETAINED),
     2188, None),
    ("kidney: median proteins per retained cell", "1a",
     lambda f: _pick(f, "proteins_per_cell_median", stage=RETAINED), 622, 1),
    ("kidney: removed cells, with their protein range", "1a",
     lambda f: ", ".join(str(int(_pick(f, c, stage=REMOVED)))
                         for c in ("n_cells", "proteins_per_cell_min", "proteins_per_cell_max")), "56, 0, 97", None),
    ("control: cells sorted, cells retained", "1b",
     lambda f: f"{_pick(f, 'n_cells', stage=ALL, **on('Control'))}, {_pick(f, 'n_cells', stage=RETAINED, **on('Control'))}",
     "768, 729", None),
    ("control: protein groups in the search, protein groups retained", "1b",
     lambda f: f"{_pick(f, 'n_protein_groups', stage=ALL, **on('Control'))}, "
               f"{_pick(f, 'n_protein_groups', stage=RETAINED, **on('Control'))}", "6208, 6078", None),
    ("control: proteins per retained cell, min and max", "1b",
     lambda f: f"{_pick(f, 'proteins_per_cell_min', stage=RETAINED, **on('Control'))}, "
               f"{_pick(f, 'proteins_per_cell_max', stage=RETAINED, **on('Control'))}", "100, 4715", None),
    ("control: median proteins per retained cell", "1b",
     lambda f: _pick(f, "proteins_per_cell_median", stage=RETAINED, **on("Control")), 1767, 1),
    ("control: removed cells, with their protein range", "1b",
     lambda f: ", ".join(str(int(_pick(f, c, stage=REMOVED, **on("Control"))))
                         for c in ("n_cells", "proteins_per_cell_min", "proteins_per_cell_max")), "39, 2, 92", None),
    ("zero-K: cells sorted, cells retained", "1b",
     lambda f: f"{_pick(f, 'n_cells', stage=ALL, **on('Zero-K'))}, {_pick(f, 'n_cells', stage=RETAINED, **on('Zero-K'))}",
     "767, 702", None),
    ("zero-K: protein groups retained", "1b",
     lambda f: _pick(f, "n_protein_groups", stage=RETAINED, **on("Zero-K")), 5953, None),
    ("zero-K: proteins per retained cell, min and max", "1b",
     lambda f: f"{_pick(f, 'proteins_per_cell_min', stage=RETAINED, **on('Zero-K'))}, "
               f"{_pick(f, 'proteins_per_cell_max', stage=RETAINED, **on('Zero-K'))}", "105, 4466", None),
    ("zero-K: median proteins per retained cell", "1b",
     lambda f: _pick(f, "proteins_per_cell_median", stage=RETAINED, **on("Zero-K")), 1780.5, 1),
    ("zero-K: removed cells, with their protein range", "1b",
     lambda f: ", ".join(str(int(_pick(f, c, stage=REMOVED, **on("Zero-K"))))
                         for c in ("n_cells", "proteins_per_cell_min", "proteins_per_cell_max")), "65, 0, 76", None),
    ("removed cells per plate: control P1, P2; zero-K P1, P2", "1b",
     lambda f: ", ".join(str(int(_pick(f, "n_cells", stage=REMOVED, **on(c, p))))
                         for c in ("Control", "Zero-K") for p in ("P1", "P2")), "23, 16, 25, 40", None),
    ("retained cells and median proteins per cell, control and zero-K (range table)", "1c",
     lambda f: ", ".join(f"{_pick(f, 'n_retained_cells', **on(c))}: {float(_pick(f, 'proteins_per_cell_median', **on(c))):.1f}"
                         for c in ("Control", "Zero-K")), "729: 1767.0, 702: 1780.5", None),
    ("kidney: cells in the list of removed cells, all below 100 proteins", "2a",
     lambda f: f"{len(f)}, {int(f['proteins_detected'].max())}", "56, 97", None),
    ("DCT: cells in the list of removed cells, control and zero-K", "2b",
     lambda f: ", ".join(str(int(f["condition"].eq(c).sum())) for c in ("Control", "Zero-K")), "39, 65", None),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    """The two dropped columns of the DCT list repeat kept columns; the lists hold the removed cells of table 1."""
    src = lib.read_source(cc.PROJECT_ROOT / (DCT_INTAKE + "filtering_removed_cells.csv"))
    key = src["condition"] + "_" + src["plate"] + "_" + src["well"].map("{:04d}".format)
    prefix = "[" + src["file_column_index"].astype(str) + "] "
    same = int((key == src["cell_key"]).sum())
    starts = int(sum(cell.startswith(p) for cell, p in zip(src["cell_id"], prefix)))
    lib.check(checks, "table 2b: the dropped columns repeat kept columns (cell_key = condition_plate_well; "
              "file_column_index = number in brackets in cell_id)", f"{same} and {starts} of {len(src)} rows",
              f"{len(src)} and {len(src)}", same == len(src) and starts == len(src))
    b, c = stored["1b"], stored["2b"]
    counts = {(cond, plate): int((c["condition"].eq(cond) & c["sort_plate"].eq(plate)).sum())
              for cond in ("Control", "Zero-K") for plate in ("P1", "P2")}
    summary = {k: int(_pick(b, "n_cells", stage=REMOVED, condition=k[0], sort_plate=k[1])) for k in counts}
    lib.check(checks, "tables 1b and 2b agree: removed cells per condition and sort plate", counts, summary,
              counts == summary)
    lib.check(checks, "tables 1a and 2a agree: removed kidney cells", len(stored["2a"]),
              int(_pick(stored["1a"], "n_cells", stage=REMOVED)),
              len(stored["2a"]) == int(_pick(stored["1a"], "n_cells", stage=REMOVED)))


WORKBOOK = WorkbookDef(
    label="S-01-a", step=STEP, workbook="report_01_S01a_tables.xlsx",
    title="Supplementary Table S-01-a. Cell and protein filtering: cells and protein groups before and after "
          "filtering, and proteins per cell, in the total-kidney and the DCT data",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names "
             "are explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"kidney": "kidney_object tables, KIDNEY rows only", "dct": "combined control + zero-K search (dct_intake tables)"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-01-a (filtering per dataset).")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
