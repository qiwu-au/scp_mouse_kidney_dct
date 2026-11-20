#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S16 (diffusion pseudotime in the DCT data), one workbook.

. The workbook holds the evidence of the response on what the pseudotime of the DCT data
follows: the root cells, pseudotime against the number of proteins per cell and against the distance in protein number
from the root cell, the same ordering computed from 20 other root cells, and the graph-abstraction (PAGA)
connectivities of the marker-score groups. No new statistics here: every value is copied from the tables written by
report_pseudotime_tabulation.py (stored dct_labels, dct_reviewer and the response to the reviewers values).

Not copied (cited from Supplementary Table S15, sheet 3): pseudotime by group and matrix, the group tests and the
comparison with random cells of the same clusters. The principal components against protein number are in
Supplementary Table S21, sheet 6.

Workbook: results/tables/report/report_15_S15a_tables.xlsx (sheet Contents, then sheets 1-4), in the layout of
Supplementary Table S7 (shared code: report_table_lib.py).

Inputs (read-only): results/tables/report/report_15_pseudotime_{roots,depth,by_quartile,other_roots,
  other_roots_summary,paga}.csv
Outputs: the workbook; results/tables/report/report_15_s15a_table_{sources,checks}.csv;
         results/tables/report/report_step_15_s15a_table_manifest.json

Command (project root):  python scripts/report_pseudotime_tables.py --overwrite
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
from report_table_lib import CONDITION, MARKER_SCORE, UNKNOWN_NOTE, Spec, WorkbookDef, _pick

STEP = "15_s15a_table"
SRC = "results/tables/report/report_15_pseudotime_"
CA, SC, SP = "cluster annotation", "marker-score label (clusters)", "marker-score label (per cell)"
MAIN = "no correction"
MATRIX = {
    "uncorrected": MAIN, "cc_removed": "cell-cycle panel removed",
    "cc_regressed": "cell-cycle panel score regressed out", "tirosh_removed": "S and G2/M lists removed",
    "tirosh_regressed": "S and G2/M scores regressed out", "corr_removed": "correlated proteins removed",
    "corr_regressed": "score of the correlated proteins regressed out",
}
UNKNOWN_LINE = UNKNOWN_NOTE + "."
ANNOTATION_COLUMNS = ((r"^annotation_", "cluster_annotation_"), (r"^clusters_", "marker_score_clusters_"),
                      (r"^per_cell_", "marker_score_per_cell_"),
                      (r"^main_root_annotation_", "main_root_cluster_annotation_"),
                      (r"^main_root_clusters_", "main_root_marker_score_clusters_"),
                      (r"^main_root_per_cell_", "main_root_marker_score_per_cell_"),
                      (r"^n_roots_annotation_", "n_roots_cluster_annotation_"),
                      (r"^n_roots_clusters_", "n_roots_marker_score_clusters_"),
                      (r"^n_roots_per_cell_", "n_roots_marker_score_per_cell_"))
DEPTH_COLUMNS = {"rho_ci_low": "proteins_rho_ci_low", "rho_ci_high": "proteins_rho_ci_high",
                 "p_value": "proteins_p_value", "distance_ci_low": "distance_rho_ci_low",
                 "distance_ci_high": "distance_rho_ci_high", "p_plate": "plate_p_value"}

SHEETS = {
    1: ("1 Root cells", "The root cell of the diffusion pseudotime in each diet condition", (UNKNOWN_LINE,)),
    2: ("2 Pseudotime and depth", "Diffusion pseudotime against the number of proteins identified per cell",
        (UNKNOWN_LINE,
         "Pseudotime from the root cell of sheet 1. Each diet condition was analysed on its own.")),
    3: ("3 Other root cells", "Diffusion pseudotime computed from 20 other root cells per diet condition",
        (UNKNOWN_LINE,
         "Other root cells: 20 cells drawn at random among the cells with the marker-score label DCT1 (per cell), "
         "the rule that chose the root cell of sheet 1.")),
    4: ("4 Graph abstraction (PAGA)", "Partition-based graph abstraction (PAGA) of the marker-score groups", ()),
}

SPECS = [
    Spec("1a", SRC + "roots.csv", "Root cell of the pseudotime: sort plate, proteins identified and labels",
         ("One row per diet condition. The pseudotime of the root cell is 0.",
          "n_cells_with_more_proteins, n_cells_with_fewer_proteins: the other cells of the condition against the "
          "root cell.",
          f"Labels of the root cell under the three annotations ({CA}; {SC}; {SP}) and its group in Figure 3A."),
         values={"condition": CONDITION}, strict=("condition",), dct_unknown=True),
    Spec("2a", SRC + "depth.csv",
         "Spearman correlation of pseudotime with proteins per cell, with the distance in protein number from the "
         "root cell, and with sort plate: all cells, seven matrices",
         ("spearman_rho_depth_distance_from_root: pseudotime against |ln(proteins of the cell) - ln(proteins of the "
          "root cell)|.",
          "ci columns: 95% interval (Fisher z). spearman_rho_plate_P2: correlation with the sort plate coded P1 = 0, "
          "P2 = 1."),
         rows=(("cells", ("all cells",)),), drop_const=("cells",),
         values={"condition": CONDITION, "matrix": MATRIX}, strict=("condition", "matrix"), rename=DEPTH_COLUMNS),
    Spec("2b", SRC + "depth.csv", "The same correlations within the cell groups of Figure 3A (no correction)",
         ("Groups: ProLIF, the cells of the per-cell test; DCT1, DCT2 and Unknown, the other cells by the annotation "
          "of their cluster. No Zero-K cluster is annotated DCT2.",),
         rows=(("cells", ("DCT1", "DCT2", "ProLIF", "Unknown")),), drop_const=("matrix",),
         values={"condition": CONDITION}, strict=("condition",),
         rename={"cells": "figure_3a_group", **DEPTH_COLUMNS}, dct_unknown=True),
    Spec("2c", SRC + "by_quartile.csv", "Pseudotime by quartile of proteins per cell (no correction)",
         ("Quartiles of the number of proteins identified per cell, within each condition. holds_root_cell: the "
          "quartile that contains the root cell.",),
         values={"condition": CONDITION}, strict=("condition",)),
    Spec("3a", SRC + "other_roots.csv", "Pseudotime from each of the 20 other root cells (no correction)",
         ("proteins_identified and the two labels describe the root cell. spearman_rho_with_main_root_pseudotime: "
          "correlation, over all cells of the condition, of the pseudotime from this root with the pseudotime from "
          "the root cell of sheet 1.",
          f"Median pseudotime of the DCT1, DCT2 and ProLIF cells under the {CA} (cluster_annotation_ columns), the "
          f"{SC} (marker_score_clusters_) and the {SP} (marker_score_per_cell_).",
          "_order: which of the DCT1 and DCT2 medians is higher. _prolif_between: the ProLIF median lies between "
          "the DCT1 and DCT2 medians. Empty cells: no Zero-K cluster is annotated DCT2."),
         values={"condition": CONDITION}, strict=("condition",), rename_re=ANNOTATION_COLUMNS, dct_unknown=True),
    Spec("3b", SRC + "other_roots_summary.csv", "The 20 other root cells: summary per diet condition",
         ("rho_: correlation of the pseudotime from another root with the pseudotime from the root cell of sheet 1 "
          "(minimum, median and maximum over the 20 roots; number of roots with a negative, a high positive or a "
          "high negative correlation).",
          "main_root_ columns: order of the DCT1 and DCT2 medians with the root cell of sheet 1. n_roots_ columns: "
          "number of the 20 other roots with the DCT1 median above the DCT2 median, or with the ProLIF median "
          "between the two. Empty cell: no Zero-K cluster is annotated DCT2."),
         values={"condition": CONDITION}, strict=("condition",), rename_re=ANNOTATION_COLUMNS),
    Spec("4a", SRC + "paga.csv", f"PAGA connectivity between the DCT1, DCT2 and ProLIF groups of the {SP}",
         ("Computed on the neighbour graph of the main analysis of each condition. connectivity: the number of "
          "connections between two groups on that graph, divided by the number expected if the groups were connected "
          "at random, and capped at 1 (Scanpy paga, default model).",
          "These groups are not those of Figure 3A: the cluster annotation has no ProLIF cluster, and PAGA was not "
          "computed for the ProLIF cells of the per-cell test."),
         drop_const=("groups",), values={"condition": CONDITION}, strict=("condition",)),
]

TERMS = [
    ("Data", "Sorted DCT cells of mice fed a control or a zero-potassium (Zero-K) diet for 4 days; one pool of 3 mice "
     "per condition; 729 Control and 702 Zero-K cells retained; one search of all runs. The two conditions are shown "
     "side by side and were not tested against each other. Sort plates P1 and P2 are plates of the pooled cells."),
    ("pseudotime", "Diffusion pseudotime: a distance of each cell from a root cell on the nearest-neighbour graph "
     "(10 neighbours, 20 principal components), scaled from 0 (the root cell) to 1."),
    ("root cell", "The cell from which pseudotime is measured. Main analysis: the first cell, in the order sort plate "
     "P1 then P2 and ascending well, with the marker-score label DCT1 (per cell)."),
    ("proteins identified; proteins per cell", "Number of proteins with an intensity above zero in a cell."),
    ("distance in protein number from the root cell", "Absolute difference between the natural logarithm of the "
     "number of proteins identified in a cell and that of the root cell."),
    ("matrix", "The data after one correction for cell-cycle proteins: no correction (the main analysis); a protein "
     "set removed; or a per-cell score regressed out of every protein (Supplementary Table S-12-d, sheet 1). Each "
     "matrix was scaled, reduced by PCA and given a pseudotime from the same root cell."),
    ("cluster annotation", "Cell types assigned to Leiden clusters (resolution 1) by differential expression of marker "
     "proteins, for each condition on its own (DCT1, DCT2, Unknown). " + UNKNOWN_NOTE + ". No cluster is annotated "
     "ProLIF, and no Zero-K cluster is annotated DCT2."),
    ("marker-score label", MARKER_SCORE),
    ("groups of Figure 3A; per-cell test", "ProLIF: cells called proliferating by the per-cell test (more S- and "
     "G2/M-phase proteins detected than expected, and Slc12a3 detected; Supplementary Table S-12-d, sheet 5). The "
     "other cells keep the annotation of their cluster."),
    ("spearman_rho", "Spearman rank correlation."),
    ("connectivity", "PAGA connectivity between two groups of cells: observed connections on the neighbour graph over "
     "those expected for randomly connected groups, capped at 1."),
]


def both(getter) -> str:
    return ", ".join(str(getter(c)) for c in ("Control", "Zero-K"))


def rho(f, column: str, condition: str, digits: int = 3) -> str:
    return f"{float(_pick(f, column, condition=condition, matrix=MAIN)):.{digits}f}"


LETTER = [
    ("root cell: proteins identified, control and zero-K", "1a",
     lambda f: both(lambda c: _pick(f, "proteins_identified", condition=c)), "3577, 1918", None),
    ("root cell: cells with more proteins, of all cells, control and zero-K", "1a",
     lambda f: both(lambda c: f"{_pick(f, 'n_cells_with_more_proteins', condition=c)} of "
                              f"{_pick(f, 'n_cells', condition=c)}"), "56 of 729, 295 of 702", None),
    ("median proteins per cell, control and zero-K", "1a",
     lambda f: both(lambda c: f"{float(_pick(f, 'median_proteins_all_cells', condition=c)):.1f}"),
     "1767.0, 1780.5", None),
    ("control root cell: cluster annotation; marker-score label (clusters); marker-score label (per cell)", "1a",
     lambda f: "; ".join(str(_pick(f, col, condition="Control")) for col in
                         ("cluster_annotation", "marker_score_label_clusters", "marker_score_label_per_cell")),
     "Unknown; DCT2; DCT1", None),
    ("zero-K root cell: cluster annotation; marker-score label (clusters); marker-score label (per cell)", "1a",
     lambda f: "; ".join(str(_pick(f, col, condition="Zero-K")) for col in
                         ("cluster_annotation", "marker_score_label_clusters", "marker_score_label_per_cell")),
     "DCT1; DCT1; DCT1", None),
    ("pseudotime against proteins per cell, no correction, control and zero-K", "2a",
     lambda f: both(lambda c: rho(f, "spearman_rho_proteins_per_cell", c)), "-0.993, -0.497", None),
    ("pseudotime against proteins per cell, seven matrices, control: lowest and highest", "2a",
     lambda f: (lambda v: f"{v.min():.2f}, {v.max():.2f}")(
         f.loc[f["condition"].eq("Control"), "spearman_rho_proteins_per_cell"].astype(float)), "-0.99, -0.95", None),
    ("pseudotime against proteins per cell, seven matrices, zero-K: lowest and highest", "2a",
     lambda f: (lambda v: f"{v.min():.2f}, {v.max():.2f}")(
         f.loc[f["condition"].eq("Zero-K"), "spearman_rho_proteins_per_cell"].astype(float)), "-0.54, -0.44", None),
    ("pseudotime against the distance in protein number from the root, no correction, control and zero-K", "2a",
     lambda f: both(lambda c: rho(f, "spearman_rho_depth_distance_from_root", c)), "0.990, 0.974", None),
    ("the same, seven matrices, control: lowest and highest", "2a",
     lambda f: (lambda v: f"{v.min():.2f}, {v.max():.2f}")(
         f.loc[f["condition"].eq("Control"), "spearman_rho_depth_distance_from_root"].astype(float)),
     "0.95, 0.99", None),
    ("the same, seven matrices, zero-K: lowest and highest", "2a",
     lambda f: (lambda v: f"{v.min():.2f}, {v.max():.2f}")(
         f.loc[f["condition"].eq("Zero-K"), "spearman_rho_depth_distance_from_root"].astype(float)),
     "0.93, 0.98", None),
    ("pseudotime against sort plate, no correction, control and zero-K", "2a",
     lambda f: both(lambda c: rho(f, "spearman_rho_plate_P2", c, 2)), "0.14, 0.03", None),
    ("control, within the groups of Figure 3A: rho against proteins per cell, lowest and highest", "2b",
     lambda f: (lambda v: f"{v.min():.3f}, {v.max():.3f}")(
         f.loc[f["condition"].eq("Control"), "spearman_rho_proteins_per_cell"].astype(float)),
     "-0.995, -0.916", None),
    ("zero-K: median pseudotime by quartile of proteins per cell (fewest to most)", "2c",
     lambda f: ", ".join(f"{float(v):.2f}" for v in f.loc[f["condition"].eq("Zero-K"), "pseudotime_median"]),
     "0.66, 0.23, 0.08, 0.26", None),
    ("control: median pseudotime by quartile of proteins per cell (fewest to most)", "2c",
     lambda f: ", ".join(f"{float(v):.2f}" for v in f.loc[f["condition"].eq("Control"), "pseudotime_median"]),
     "0.77, 0.50, 0.36, 0.14", None),
    ("quartile that holds the root cell, control and zero-K", "2c",
     lambda f: both(lambda c: _pick(f, "quartile_of_proteins_per_cell", condition=c, holds_root_cell=True)),
     "Q4 (most), Q3", None),
    ("other roots, control: rho with the main-root pseudotime, lowest and highest; roots with a negative rho", "3b",
     lambda f: f"{float(_pick(f, 'rho_min', condition='Control')):.2f}, "
               f"{float(_pick(f, 'rho_max', condition='Control')):.2f}; "
               f"{_pick(f, 'n_roots_rho_negative', condition='Control')}", "-0.99, 1.00; 9", None),
    ("other roots, zero-K: rho with the main-root pseudotime, lowest and highest; roots with a negative rho", "3b",
     lambda f: f"{float(_pick(f, 'rho_min', condition='Zero-K')):.2f}, "
               f"{float(_pick(f, 'rho_max', condition='Zero-K')):.2f}; "
               f"{_pick(f, 'n_roots_rho_negative', condition='Zero-K')}", "-0.50, 0.99; 6", None),
    ("control: other roots with the DCT1 median above the DCT2 median: cluster annotation; marker-score label "
     "(clusters); marker-score label (per cell)", "3b",
     lambda f: "; ".join(str(int(float(_pick(f, col, condition="Control")))) for col in
                         ("n_roots_cluster_annotation_DCT1_above_DCT2", "n_roots_marker_score_clusters_DCT1_above_DCT2",
                          "n_roots_marker_score_per_cell_DCT1_above_DCT2")), "14; 3; 3", None),
    ("other roots with the ProLIF median between DCT1 and DCT2, marker-score label (per cell), control and zero-K",
     "3b", lambda f: both(lambda c: _pick(f, "n_roots_marker_score_per_cell_prolif_between", condition=c)),
     "0, 0", None),
    ("control: the 20 other roots by cluster annotation (DCT1, DCT2, Unknown)", "3a",
     lambda f: ", ".join(str(int((f.loc[f["condition"].eq("Control"), "cluster_annotation"] == g).sum()))
                         for g in ("DCT1", "DCT2", "Unknown")), "10, 2, 8", None),
    ("PAGA, control: DCT1-DCT2, DCT1-ProLIF, DCT2-ProLIF", "4a",
     lambda f: ", ".join(f"{float(_pick(f, 'connectivity', condition='Control', group_1=a, group_2=b)):.2f}"
                         for a, b in (("DCT1", "DCT2"), ("DCT1", "ProLIF"), ("DCT2", "ProLIF"))),
     "0.76, 0.65, 0.78", None),
    ("PAGA, zero-K: DCT1-DCT2, DCT1-ProLIF, DCT2-ProLIF", "4a",
     lambda f: ", ".join(f"{float(_pick(f, 'connectivity', condition='Zero-K', group_1=a, group_2=b)):.2f}"
                         for a, b in (("DCT1", "DCT2"), ("DCT1", "ProLIF"), ("DCT2", "ProLIF"))),
     "0.92, 0.55, 0.68", None),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    """The tables agree with each other, and no row compares the two diet conditions."""
    roots, depth, per_root, summary = stored["1a"], stored["2a"], stored["3a"], stored["3b"]
    cells = {c: int(_pick(roots, "n_cells", condition=c)) for c in ("Control", "Zero-K")}
    in_2a = {c: int(_pick(depth, "n_cells", condition=c, matrix=MAIN)) for c in cells}
    lib.check(checks, "tables 1a and 2a agree: cells per condition (729 Control, 702 Zero-K)", in_2a, cells,
              in_2a == cells == {"Control": 729, "Zero-K": 702})
    quart = {c: int(stored["2c"].loc[stored["2c"]["condition"].eq(c), "n_cells"].astype(int).sum()) for c in cells}
    lib.check(checks, "table 2c: the four quartiles add up to the cells of the condition", quart, cells, quart == cells)
    counted = {c: int((per_root.loc[per_root["condition"].eq(c),
                                    "spearman_rho_with_main_root_pseudotime"].astype(float) < 0).sum())
               for c in cells}
    listed = {c: int(_pick(summary, "n_roots_rho_negative", condition=c)) for c in cells}
    lib.check(checks, "tables 3a and 3b agree: roots with a negative correlation with the main-root pseudotime",
              counted, listed, counted == listed)
    main_roots = set(roots["root_cell_id"])
    lib.check(checks, "table 3a: 20 other roots per condition, none of them the root cell of table 1a",
              per_root.groupby("condition").size().to_dict(), {"Control": 20, "Zero-K": 20},
              per_root.groupby("condition").size().to_dict() == {"Control": 20, "Zero-K": 20}
              and not (set(per_root["root_cell_id"]) & main_roots))
    between = [key for key, frame in stored.items() for value in frame.to_numpy().ravel()
               if isinstance(value, str) and ("between conditions" in value or "Zero-K against Control" in value)]
    lib.check(checks, "within-condition rows only: no table row compares the two diet conditions",
              sorted(set(between)) or "none", "none", not between)


WORKBOOK = WorkbookDef(
    label="S-15-a", step=STEP, workbook="report_15_S15a_tables.xlsx",
    title="Supplementary Table S-15-a. Diffusion pseudotime in the DCT data: root cells, the number of proteins per "
          "cell, other root cells, and graph abstraction",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names are "
             "explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"dct": "combined control + zero-K search; stored pseudotime of the dct_labels / dct_reviewer analyses",
                "scope": "within condition only; pseudotime by group and matrix is in Supplementary Table S-12-d, "
                         "sheet 3, and is not copied"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-15-a (pseudotime in the DCT data).")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
