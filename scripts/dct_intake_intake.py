#!/usr/bin/env python
"""dct_intake steps 0-2: freeze check, inputs, cell IDs, value checks and filtering of the combined search.



Inputs (read-only): data/DCT/DCT_Ctrl_ZeroK_combine_1535cells.csv;
results/tables/dct_intake_zerok_vs_control_frozen_briefs_sha256.csv and the six plans it lists;
results/tables/intake_core/intake_core_dct_cells.csv (cell set by cell_key only).
Outputs (results/tables/dct_intake/):
  dct_intake_freeze_check.csv             SHA256 and size of the plans and the export against the freeze record;
                                          git commit of the record
  dct_intake_input_files.csv              SHA256, size, line count, header and shape of the export
  dct_intake_header_columns.csv           every column header with its parsed condition, plate, well, cell_key
  dct_intake_column_blocks.csv            runs of consecutive columns with the same condition and plate
  dct_intake_wells.csv                    wells present, missing and duplicated per condition and plate
  dct_intake_cell_set_vs_t17.csv          cell_key sets per condition and plate, combined export vs intake_core
  dct_intake_value_checks.csv             quantity values: NaN, non-numeric, zero, negative, infinite
  dct_intake_search_settings.csv          Spectronaut settings (authors, 2 Nov 2026) beside the separate searches
  dct_intake_filtering_summary.csv        raw / removed / retained cells and proteins by condition and plate
  dct_intake_filtering_removed_cells.csv  cells with < 100 protein IDs
  dct_intake_protein_filter.csv           protein groups removed by the >= 2 retained cells filter
  dct_intake_proteins_per_cell_range.csv  protein IDs per retained cell (R2 minor 3)

Filters (submitted thresholds, scp_common; intake_core step 2): cells with >= 100
protein groups > 0 over all export rows, then protein groups > 0 in >= 2
retained cells of that condition; applied per condition. Stops (exit 1) when a
plan or the export differs from the freeze record, a header does not parse, a
cell_key repeats, the cell set differs from intake_core's, a count differs from the
plan, or a quantity is non-numeric, zero, negative or infinite.

Command (project root):  python scripts/dct_intake_intake.py
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
import subprocess
import sys

import numpy as np
import pandas as pd

import scp_common as cc
import intake_core_intake as t17_intake
import intake_core_lib as t17
import dct_intake_lib as lib


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=cc.PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    return out.stdout.strip()


def freeze_check(run: cc.Run) -> tuple[pd.DataFrame, list[str]]:
    """Step 0: the export against the recorded checksum (data/input_checksums.csv)."""
    record = pd.read_csv(run.use_input(lib.FREEZE_TABLE, "record of the input checksums"))
    rows, problems = [], []
    for item in record.itertuples(index=False):
        path = cc.PROJECT_ROOT / item.path
        exists = path.exists()
        digest = cc.sha256(path) if exists else ""
        size = path.stat().st_size if exists else -1
        ok = exists and digest == item.sha256 and size == int(item.bytes)
        rows.append({"path": item.path, "role": item.role, "expected_bytes": int(item.bytes), "observed_bytes": size,
                     "expected_sha256": item.sha256, "observed_sha256": digest, "pass": ok})
        if not ok:
            problems.append(f"{item.path} differs from the freeze record")
    return pd.DataFrame(rows), problems


def column_blocks(cells: pd.DataFrame) -> pd.DataFrame:
    key = (cells["condition"] + " " + cells["plate"]).to_numpy()
    rows, start = [], 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[start]:
            rows.append({"block": len(rows) + 1, "condition_plate": key[start],
                         "first_column_index": int(cells["file_column_index"].iloc[start]),
                         "last_column_index": int(cells["file_column_index"].iloc[i - 1]), "n_columns": i - start,
                         "first_well": cells["well"].iloc[start], "last_well": cells["well"].iloc[i - 1]})
            start = i
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_intake: freeze check, inputs, IDs, value checks, filtering.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "intake", args)

    # 0. Freeze check ---------------------------------------------------------
    freeze, problems = freeze_check(run)
    run.table(freeze, "freeze_check")
    if problems:
        for text in problems:
            run.note(f"STOP: {text}")
        run.finish(status="stopped: freeze check failed")
        sys.exit(1)
    run.note("Input check: the export matches the recorded checksum.")

    # 1. Inputs and IDs ------------------------------------------------------
    comb = lib.read_combined(run)
    key = cc.rel(comb.path)
    with open(comb.path, "rb") as handle:
        n_lines = sum(1 for _ in handle)
    cell_cols = list(comb.cells.index)
    quantity_ok = all(c.endswith(lib.QUANTITY_SUFFIX) for c in cell_cols)
    run.table(pd.DataFrame([{
        "path": key, "sha256": run.inputs[key]["sha256"], "bytes": run.inputs[key]["bytes"], "n_lines": n_lines,
        "n_columns": len(comb.header), "metadata_columns": ";".join(comb.header[:3]),
        "n_cell_columns": len(cell_cols), "expected_n_cells": lib.EXPECTED["n_cells"],
        "n_protein_groups": len(comb.meta), "expected_n_protein_groups": lib.EXPECTED["n_protein_groups"],
        "quantity_column": "PG.Quantity" if quantity_ok else "not all columns end in .PG.Quantity",
        "missing_value_token": lib.MISSING_TOKEN,
        "first_cell_column": cell_cols[0], "last_cell_column": cell_cols[-1],
    }]), "input_files")
    if not quantity_ok:
        problems.append(f"not every cell column ends in {lib.QUANTITY_SUFFIX}")
    if len(cell_cols) != lib.EXPECTED["n_cells"]:
        problems.append(f"{len(cell_cols)} cell columns, expected {lib.EXPECTED['n_cells']}")
    if len(comb.meta) != lib.EXPECTED["n_protein_groups"]:
        problems.append(f"{len(comb.meta)} protein groups, expected {lib.EXPECTED['n_protein_groups']}")
    if n_lines != len(comb.meta) + 1:
        problems.append(f"{n_lines} lines, expected {len(comb.meta) + 1} (header + one per protein group)")

    header_rows = []
    for pos, col in enumerate(comb.header, start=1):
        row = {"column_position": pos, "column_name": col, "kind": "metadata" if pos <= 3 else "cell"}
        if pos > 3:
            row.update(comb.cells.loc[col].to_dict())
        header_rows.append(row)
    run.table(pd.DataFrame(header_rows), "header_columns")
    blocks = column_blocks(comb.cells)
    run.table(blocks, "column_blocks")
    run.note("Column blocks: " + "; ".join(f"{b.condition_plate} [{b.first_column_index}]-[{b.last_column_index}]"
                                           for b in blocks.itertuples(index=False)))

    n_dup_ids = int(comb.cells.index.duplicated().sum())
    n_dup_keys = int(comb.cells["cell_key"].duplicated().sum())
    if n_dup_ids:
        problems.append(f"{n_dup_ids} duplicated cell IDs")
    if n_dup_keys:
        problems.append(f"{n_dup_keys} duplicated cell_key values")
    index_seq = comb.cells["file_column_index"].to_numpy()
    if not np.array_equal(index_seq, np.arange(1, len(index_seq) + 1)):
        problems.append(f"[n] column indices are not 1..{len(index_seq)} in file order")
    other_conditions = sorted(set(comb.cells["condition"]) - set(lib.CONDITIONS))
    other_plates = sorted(set(comb.cells["plate"]) - set(lib.PLATES))
    if other_conditions or other_plates:
        problems.append(f"unexpected conditions {other_conditions} or plates {other_plates}")

    well_rows = []
    for cond in lib.CONDITIONS:
        exp = lib.EXPECTED["conditions"][cond]
        own = comb.cells.loc[comb.cells["condition"] == cond]
        if len(own) != exp["n_cells"]:
            problems.append(f"{cond}: {len(own)} cells, expected {exp['n_cells']}")
        missing_all = []
        for plate in lib.PLATES:
            sub = own.loc[own["plate"] == plate]
            wells = sub["well"].astype(int)
            missing = [f"{plate}_{w:04d}" for w in range(1, lib.WELLS_PER_PLATE + 1) if w not in set(wells)]
            dup = sorted(sub.loc[sub["well"].duplicated(), "well"].unique())
            out_of_range = sorted(wells[(wells < 1) | (wells > lib.WELLS_PER_PLATE)].astype(str))
            missing_all.extend(missing)
            well_rows.append({
                "condition": cond, "plate": plate, "n_cells": len(sub), "expected_n_cells": exp["plates"][plate],
                "n_missing_wells": len(missing), "missing_wells": ";".join(missing),
                "n_duplicated_wells": len(dup), "duplicated_wells": ";".join(dup),
                "wells_out_of_range": ";".join(out_of_range),
            })
            if len(sub) != exp["plates"][plate]:
                problems.append(f"{cond} {plate}: {len(sub)} cells, expected {exp['plates'][plate]}")
            if dup or out_of_range:
                problems.append(f"{cond} {plate}: duplicated or out-of-range wells")
        if missing_all != exp["missing_wells"]:
            problems.append(f"{cond}: missing wells {missing_all}, expected {exp['missing_wells']}")
    run.table(pd.DataFrame(well_rows), "wells")

    # Cell set against intake_core (cell_key only; no intake_core value is read here).
    t17_path = cc.out_path(t17.TASK, "tables", "dct_cells", ".csv")
    set_rows = []
    if t17_path.exists():
        t17_cells = pd.read_csv(run.use_input(t17_path, "intake_core per-cell table (cell set by cell_key only)"),
                                dtype={"well": str}, usecols=["cell_id", "condition", "plate", "well"])
        t17_cells["cell_key"] = lib.add_cell_key(t17_cells)
    else:   # the per-cell table of the earlier search of the same runs is not a public input
        t17_cells = None
        run.note("Cell-set comparison skipped: the per-cell table of the earlier search is not available.")
    for cond in (lib.CONDITIONS if t17_cells is not None else ()):
        for plate in lib.PLATES:
            mine = set(comb.cells.loc[(comb.cells["condition"] == cond) & (comb.cells["plate"] == plate), "cell_key"])
            theirs = set(t17_cells.loc[(t17_cells["condition"] == cond) & (t17_cells["plate"] == plate), "cell_key"])
            only_c, only_t = sorted(mine - theirs), sorted(theirs - mine)
            set_rows.append({"condition": cond, "plate": plate, "n_combined": len(mine), "n_t17": len(theirs),
                             "n_common": len(mine & theirs), "only_combined": ";".join(only_c),
                             "only_t17": ";".join(only_t), "equal": not only_c and not only_t})
            if only_c or only_t:
                problems.append(f"{cond} {plate}: cell set differs from intake_core ({len(only_c)} combined only, "
                                f"{len(only_t)} intake_core only)")
    run.table(pd.DataFrame(set_rows), "cell_set_vs_t17")

    value_rows = []
    for cond in (*lib.CONDITIONS, "all"):
        mask = None if cond == "all" else (comb.cells["condition"] == cond).to_numpy()
        rep = comb.value_report(mask)
        value_rows.append({"condition": cond, **rep, "fraction_missing": rep["n_missing_NaN"] / rep["n_values"]})
        for field in ("n_nonnumeric", "n_zero", "n_negative", "n_infinite"):
            if rep[field] and cond == "all":
                problems.append(f"{rep[field]} quantity values counted in {field}")
    run.table(pd.DataFrame(value_rows), "value_checks")

    run.table(pd.DataFrame([{"setting": name, "combined_search": value, "separate_search_ctrl": value,
                             "separate_search_zerok": value, "source": lib.SETTINGS_SOURCE,
                             "note": lib.SEPARATE_SAME_NOTE} for name, value in lib.SEARCH_SETTINGS]),
              "search_settings")

    if problems:
        for text in problems:
            run.note(f"STOP: {text}")
        run.finish(status="stopped: input check failed")
        sys.exit(1)
    run.note("Headers parse, cell_key is unique, counts match the plan, the cell set per condition and plate equals "
             "the earlier table's (where it is available), the only missing well is ZeroK P1_0368, and every quantity is NaN or a positive finite number.")

    # 2. Filtering (intake_core step 2, per condition) --------------------------------
    describe = t17_intake.describe
    summary_rows, removed_rows, protein_rows, range_rows = [], [], [], []
    for cond in lib.CONDITIONS:
        raw = comb.raw(cond)
        filt = t17.apply_filters(raw)
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
    run.table(removed[["condition", "plate", "well", "cell_key", "cell_id", "file_column_index",
                       "n_detected_all_rows"]], "filtering_removed_cells")
    run.table(pd.DataFrame(protein_rows), "protein_filter")
    run.table(pd.DataFrame(range_rows), "proteins_per_cell_range")
    run.finish(parameters={"MIN_CELL_PROTEINS": cc.MIN_CELL_PROTEINS, "MIN_PROTEIN_CELLS": cc.MIN_PROTEIN_CELLS,
                           "missing_value_token": lib.MISSING_TOKEN, "n_lines": n_lines,
                           "search_settings": dict(lib.SEARCH_SETTINGS),
                           "search_settings_source": lib.SETTINGS_SOURCE})


if __name__ == "__main__":
    main()
