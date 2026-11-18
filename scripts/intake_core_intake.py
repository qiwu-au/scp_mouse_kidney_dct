#!/usr/bin/env python
"""intake_core step 1: inputs, cell IDs, value checks and filtering of the new DCT data.



Inputs (read-only): data/DCT/DCT_Ctrl_768cells.csv, data/DCT/DCT_ZeroK_767cells.csv.
Outputs (results/tables/intake_core/):
  intake_core_input_files.csv              SHA256, size, header and shape of each export
  intake_core_header_columns.csv           every column header with its parsed condition, plate and well
  intake_core_wells.csv                    wells present, missing and duplicated per condition and plate
  intake_core_value_checks.csv             quantity values: NaN, non-numeric, zero, negative, infinite
  intake_core_filtering_summary.csv        raw / removed / retained cells and proteins by condition and plate
  intake_core_filtering_removed_cells.csv  cells with < 100 protein IDs
  intake_core_protein_filter.csv           protein groups removed by the >= 2 retained cells filter
  intake_core_proteins_per_cell_range.csv  protein IDs per retained cell (R2 minor 3)

Filters (submitted thresholds, scp_common): cells with >= 100 protein groups
> 0 over all rows, then protein groups > 0 in >= 2 retained cells; applied per
condition. Stops (exit 1) when a header does not parse, a count differs from the
plan, an ID or well is duplicated, a missing well is unexpected, or a quantity
is non-numeric, zero, negative or infinite.

Command (project root):  python scripts/intake_core_intake.py
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
import sys

import numpy as np
import pandas as pd

import scp_common as cc
import intake_core_lib as lib


def describe(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"protein_ids_min": np.nan, "protein_ids_q25": np.nan, "protein_ids_median": np.nan,
                "protein_ids_q75": np.nan, "protein_ids_max": np.nan}
    q = np.percentile(values, [25, 50, 75])
    return {"protein_ids_min": int(values.min()), "protein_ids_q25": float(q[0]), "protein_ids_median": float(q[1]),
            "protein_ids_q75": float(q[2]), "protein_ids_max": int(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description="intake_core: inputs, IDs, value checks and filtering.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "intake", args)

    raws = {cond: lib.read_raw(cond, run) for cond in lib.CONDITIONS}
    problems: list[str] = []

    # 1. Inputs and IDs ------------------------------------------------------
    file_rows, header_rows, well_rows, value_rows = [], [], [], []
    for cond, raw in raws.items():
        exp = lib.EXPECTED[cond]
        key = cc.rel(raw.path)
        cell_cols = list(raw.cells.index)
        quantity_ok = all(c.endswith(lib.QUANTITY_SUFFIX) for c in cell_cols)
        file_rows.append({
            "condition": cond, "path": key, "sha256": run.inputs[key]["sha256"], "bytes": run.inputs[key]["bytes"],
            "n_columns": len(raw.header), "metadata_columns": ";".join(raw.header[:3]),
            "n_cell_columns": len(cell_cols), "expected_n_cells": exp["n_cells"],
            "n_protein_groups": len(raw.meta), "expected_n_protein_groups": exp["n_protein_groups"],
            "quantity_column": "PG.Quantity" if quantity_ok else "not all columns end in .PG.Quantity",
            "missing_value_token": lib.MISSING_TOKEN,
            "first_cell_column": cell_cols[0], "last_cell_column": cell_cols[-1],
        })
        if not quantity_ok:
            problems.append(f"{cond}: not every cell column ends in {lib.QUANTITY_SUFFIX}")
        if len(cell_cols) != exp["n_cells"]:
            problems.append(f"{cond}: {len(cell_cols)} cell columns, expected {exp['n_cells']}")
        if len(raw.meta) != exp["n_protein_groups"]:
            problems.append(f"{cond}: {len(raw.meta)} protein groups, expected {exp['n_protein_groups']}")
        for pos, col in enumerate(raw.header, start=1):
            row = {"condition": cond, "column_position": pos, "column_name": col,
                   "kind": "metadata" if pos <= 3 else "cell"}
            if pos > 3:
                row.update(raw.cells.loc[col].to_dict())
            header_rows.append(row)

        n_dup_ids = int(raw.cells.index.duplicated().sum())
        if n_dup_ids:
            problems.append(f"{cond}: {n_dup_ids} duplicated cell IDs")
        index_seq = raw.cells["file_column_index"].to_numpy()
        if not np.array_equal(index_seq, np.arange(1, len(index_seq) + 1)):
            problems.append(f"{cond}: [n] column indices are not 1..{len(index_seq)} in file order")
        other_plates = sorted(set(raw.cells["plate"]) - set(lib.PLATES))
        if other_plates:
            problems.append(f"{cond}: unexpected plates {other_plates}")
        missing_all = []
        for plate in lib.PLATES:
            sub = raw.cells.loc[raw.cells["plate"] == plate]
            wells = sub["well"].astype(int)
            missing = [f"{plate}_{w:04d}" for w in range(1, lib.WELLS_PER_PLATE + 1) if w not in set(wells)]
            dup = sorted(sub.loc[sub["well"].duplicated(), "well"].unique())
            out_of_range = sorted(wells[(wells < 1) | (wells > lib.WELLS_PER_PLATE)].astype(str))
            missing_all.extend(missing)
            well_rows.append({
                "condition": cond, "plate": plate, "n_cells": len(sub), "expected_n_cells": exp["plates"][plate],
                "n_missing_wells": len(missing), "missing_wells": ";".join(missing),
                "n_duplicated_wells": len(dup), "duplicated_wells": ";".join(dup),
                "wells_out_of_range": ";".join(out_of_range), "n_duplicated_cell_ids_in_file": n_dup_ids,
            })
            if len(sub) != exp["plates"][plate]:
                problems.append(f"{cond} {plate}: {len(sub)} cells, expected {exp['plates'][plate]}")
            if dup or out_of_range:
                problems.append(f"{cond} {plate}: duplicated or out-of-range wells")
        if missing_all != exp["missing_wells"]:
            problems.append(f"{cond}: missing wells {missing_all}, expected {exp['missing_wells']}")

        rep = raw.value_report
        value_rows.append({"condition": cond, **rep, "fraction_missing": rep["n_missing_NaN"] / rep["n_values"]})
        for field in ("n_nonnumeric", "n_zero", "n_negative", "n_infinite"):
            if rep[field]:
                problems.append(f"{cond}: {rep[field]} quantity values counted in {field}")

    run.table(pd.DataFrame(file_rows), "input_files")
    run.table(pd.DataFrame(header_rows), "header_columns")
    run.table(pd.DataFrame(well_rows), "wells")
    run.table(pd.DataFrame(value_rows), "value_checks")
    if problems:
        for text in problems:
            run.note(f"STOP: {text}")
        run.finish(status="stopped: input check failed")
        sys.exit(1)
    run.note("Headers parse, counts match the plan, the only missing well is ZeroK P1_0368, "
             "and every quantity is NaN or a positive finite number.")

    # 2. Filtering -----------------------------------------------------------
    summary_rows, removed_rows, protein_rows, range_rows = [], [], [], []
    for cond, raw in raws.items():
        filt = lib.apply_filters(raw)
        detected = np.nan_to_num(raw.values, nan=0.0) > 0
        all_proteins = np.ones(detected.shape[1], dtype=bool)
        plates = raw.cells["plate"].to_numpy()
        for plate in ("all", *lib.PLATES):
            in_plate = np.ones(len(plates), dtype=bool) if plate == "all" else plates == plate
            stages = (
                ("raw_all_cells", in_plate, all_proteins),
                ("removed_cells", in_plate & ~filt.keep_cells, all_proteins),
                ("retained_cells_all_proteins", in_plate & filt.keep_cells, all_proteins),
                ("retained_cells_retained_proteins", in_plate & filt.keep_cells, filt.keep_proteins),
            )
            for stage, cells, proteins in stages:
                sub = detected[np.ix_(cells, proteins)]
                summary_rows.append({"condition": cond, "plate": plate, "stage": stage, "n_cells": int(cells.sum()),
                                     "n_protein_groups": int(proteins.sum()),
                                     "n_protein_groups_detected_in_any_cell": int(sub.any(axis=0).sum()),
                                     **describe(sub.sum(axis=1))})
            ids = detected[np.ix_(in_plate & filt.keep_cells, filt.keep_proteins)].sum(axis=1)
            range_rows.append({"condition": cond, "plate": plate, "n_retained_cells": int(ids.size),
                               "n_retained_proteins_condition": int(filt.keep_proteins.sum()), **describe(ids),
                               "protein_ids_mean": float(ids.mean())})
        removed = raw.cells.loc[~filt.keep_cells].copy()
        removed["n_detected_all_rows"] = filt.n_detected_all_rows[~filt.keep_cells]
        removed_rows.append(removed.reset_index())
        protein_rows.append({
            "condition": cond, "n_protein_groups_in_file": int(detected.shape[1]),
            "n_detected_in_any_cell": int((filt.n_cells_all > 0).sum()),
            "n_never_detected_in_any_cell": int((filt.n_cells_all == 0).sum()),
            "n_detected_in_no_retained_cell": int((filt.n_cells_retained == 0).sum()),
            "n_detected_in_exactly_one_retained_cell": int((filt.n_cells_retained == 1).sum()),
            "n_retained": int(filt.keep_proteins.sum()), "n_removed": int((~filt.keep_proteins).sum()),
            "n_cells_retained": int(filt.keep_cells.sum()), "n_cells_removed": int((~filt.keep_cells).sum()),
        })
        run.note(f"{cond}: {len(raw.cells)} -> {int(filt.keep_cells.sum())} cells; {detected.shape[1]} -> "
                 f"{int(filt.keep_proteins.sum())} protein groups.")

    run.table(pd.DataFrame(summary_rows), "filtering_summary")
    removed = pd.concat(removed_rows, ignore_index=True)
    run.table(removed[["condition", "plate", "well", "cell_id", "file_column_index", "n_detected_all_rows"]],
              "filtering_removed_cells")
    run.table(pd.DataFrame(protein_rows), "protein_filter")
    run.table(pd.DataFrame(range_rows), "proteins_per_cell_range")
    run.finish(parameters={"MIN_CELL_PROTEINS": cc.MIN_CELL_PROTEINS, "MIN_PROTEIN_CELLS": cc.MIN_PROTEIN_CELLS,
                           "missing_value_token": lib.MISSING_TOKEN})


if __name__ == "__main__":
    main()
