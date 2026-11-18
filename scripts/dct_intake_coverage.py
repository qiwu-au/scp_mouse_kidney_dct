#!/usr/bin/env python
"""dct_intake step 5: marker, zerok_core panel and MitoCarta coverage per condition (coverage only).



Inputs (read-only): data/DCT/DCT_markers_V1.xlsx; results/tables/dct_intake/dct_intake_proteins.csv and
dct_intake_dct_cells.csv; results/tables/zerok_core/zerok_core_prespecified_panel.csv (frozen; SHA256
checked, items and directions only); results/tables/qc_mito/qc_mito_mitocarta_mouse_list.csv (qc_mito parse of
data/reference/Mouse.MitoCarta3.0.xls; no download).
Outputs (results/tables/dct_intake/):
  dct_intake_marker_coverage.csv          one row per list x group x marker (workbook, contamination, intake_core
                                          zero-K candidate names, cell cycle); per condition: in export,
                                          retained, cells detected and fraction of retained cells; in joint
  dct_intake_marker_coverage_summary.csv  per list x group: markers in export / retained / in joint
  dct_intake_panel_coverage.csv           every row of the frozen zerok_core panel with its coverage in the combined
                                          search; low_coverage = protein detected in < 10% of retained cells
                                          in either condition
  dct_intake_mitocarta_coverage.csv       MitoCarta3.0 protein groups per condition and protein set
  dct_intake_mitocarta_protein_flags.csv  MitoCarta flags of every protein group of the export

Coverage functions are intake_core's (intake_core_coverage: coverage, coverage_summary,
mitocarta_tables) and zerok_core's (zerok_core_lib: gene_coverage, list_coverage),
applied to a intake_core-style mapping frame of the one combined search
(dct_intake_lib.mapping_frame). Marker rule: cleaned first PG.Genes symbol;
secondary-symbol hits listed separately. Fractions use the retained cells of
each condition.

Command (project root):  python scripts/dct_intake_coverage.py
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

import pandas as pd

import scp_common as cc
import intake_core_coverage as t17cov
import zerok_core_lib as t21
import dct_intake_lib as lib

PANEL_COLUMNS = ["item", "item_type", "panel_group", "expected_direction", "measurable_here",
                 "not_measurable_reason", "low_coverage",
                 "ctrl_in_export", "ctrl_retained", "ctrl_n_cells_detected", "ctrl_fraction_cells",
                 "zerok_in_export", "zerok_retained", "zerok_n_cells_detected", "zerok_fraction_cells",
                 "in_joint", "ctrl_var_name", "zerok_var_name", "ctrl_protein_group", "zerok_protein_group",
                 "ctrl_n_groups", "zerok_n_groups",
                 "n_proteins_in_list", "ctrl_n_present", "zerok_n_present", "ctrl_n_retained", "zerok_n_retained",
                 "n_in_joint", "absent_both", "present_in_joint", "genes_checked"]


def panel_coverage(run: cc.Run, mapping: pd.DataFrame, n_cells: dict[str, int]) -> pd.DataFrame:
    """Coverage of every frozen zerok_core panel row in the combined search (the zerok_core build rule, combined mapping)."""
    path = run.use_input(lib.PANEL_PATH, "zerok_core frozen pre-specified panel (items and directions only)")
    digest = cc.sha256(path)
    if digest != lib.PANEL_SHA256:
        run.note(f"STOP: zerok_core panel SHA256 {digest} differs from the frozen {lib.PANEL_SHA256}")
        run.finish(status="stopped: panel hash differs")
        sys.exit(1)
    panel = pd.read_csv(path, keep_default_na=False)
    rows = []
    for entry in panel.itertuples(index=False):
        row = {"item": entry.item, "item_type": entry.item_type, "panel_group": entry.panel_group,
               "expected_direction": entry.expected_direction}
        if entry.item_type == "protein":
            row.update(t21.gene_coverage(mapping, entry.item, n_cells))
            row["genes_checked"] = entry.item
            row["measurable_here"] = bool(row["in_joint"])
            row["not_measurable_reason"] = "" if row["in_joint"] else (
                "not retained in both conditions of the combined search (or absent)")
        elif entry.item_type == "score":
            genes = [*t21.SCORE_LISTS[entry.item][0], *t21.SCORE_LISTS[entry.item][1]]
            row.update(t21.list_coverage(mapping, genes, n_cells))
            row["genes_checked"] = ";".join(dict.fromkeys(genes))
            row["measurable_here"] = row["n_in_joint"] > 0
            row["not_measurable_reason"] = "" if row["n_in_joint"] else "no list member in the joint object"
        else:  # limit
            genes = t21.PANEL_GENES[entry.item]
            row["genes_checked"] = ";".join(genes)
            row["measurable_here"] = False
            if genes:
                cov = t21.list_coverage(mapping, genes, n_cells)
                row.update(cov)
                row["not_measurable_reason"] = (
                    f"absent from the combined export ({cov['absent_both']})" if not cov["ctrl_n_present"]
                    and not cov["zerok_n_present"] else
                    f"present in the combined export (Ctrl {cov['ctrl_n_present']}, ZeroK {cov['zerok_n_present']} "
                    f"of {cov['n_proteins_in_list']})")
            else:
                row["not_measurable_reason"] = ("phosphosite: no phospho-enrichment in this workflow, so "
                                                "phosphorylation is not quantified")
        rows.append(row)
    frame = pd.DataFrame(rows)
    fractions = frame[["ctrl_fraction_cells", "zerok_fraction_cells"]].apply(pd.to_numeric, errors="coerce")
    frame["low_coverage"] = (frame["item_type"] == "protein") & (fractions.min(axis=1) < t21.LOW_COVERAGE_FRACTION)
    for col in PANEL_COLUMNS:
        if col not in frame:
            frame[col] = ""
    run.note(f"zerok_core panel SHA256 {digest} (frozen value; matches).")
    return frame[PANEL_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_intake: marker, panel and MitoCarta coverage.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "coverage", args)

    proteins = lib.read_proteins(run)
    mapping = lib.mapping_frame(proteins)
    cells = lib.read_cells(run)
    n_cells = {cond: int((cells["retained"] & (cells["condition"] == cond)).sum()) for cond in lib.CONDITIONS}

    cov = t17cov.coverage(t17cov.marker_lists(run), mapping, n_cells)
    run.table(cov, "marker_coverage")
    summary = t17cov.coverage_summary(cov)
    run.table(summary, "marker_coverage_summary")

    panel = panel_coverage(run, mapping, n_cells)
    run.table(panel, "panel_coverage")
    low = panel.loc[panel["low_coverage"].astype(bool), "item"].tolist()
    run.note(f"zerok_core panel, low coverage (< 10% of retained cells in either condition): {';'.join(low) or '(none)'}")

    mito, flags = t17cov.mitocarta_tables(run, mapping)
    mito["protein_set"] = mito["protein_set"].replace(
        {"matched (joint) groups": "joint groups (retained in both conditions)"})
    flags = flags.rename(columns={"search": "condition"})
    run.table(mito, "mitocarta_coverage")
    run.table(flags, "mitocarta_protein_flags")

    for row in summary.itertuples(index=False):
        run.note(f"{row.list} / {row.group}: Ctrl {row.ctrl_in_export}/{row.n_markers}, ZeroK "
                 f"{row.zerok_in_export}/{row.n_markers} in export; {row.in_joint}/{row.n_markers} in joint.")
    run.finish(parameters={"marker_rule": "cleaned first PG.Genes symbol (Tirosh aliases checked for cell-cycle "
                                          "lists); secondary-symbol hits listed separately",
                           "retained_cells": n_cells, "low_coverage_fraction": t21.LOW_COVERAGE_FRACTION,
                           "panel_sha256": lib.PANEL_SHA256})


if __name__ == "__main__":
    main()
