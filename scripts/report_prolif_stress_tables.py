#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-14-b (ProLIF cells: prevalence, dissociation-stress proteins and
quality flags in the DCT data), one workbook.

. One workbook with a Contents sheet and five data sheets, built from nine stored
tables: 7 of the response to the reviewers (report_prolif_stress_tabulation.py) and 2 of dct_reviewer (stress-panel coverage; stress
contrasts of the marker-score label). DCT data of the combined control + zero-K search only. No new statistics:
every value is copied from a stored table. Layout, style and checks are those of the other supplementary tables
(report_table_lib.py).

Not copied here, because other supplementary tables already hold them: the five criteria of the per-cell test and
its protein set (S-12-d, tables 4d and 5a); depth and sort plate of the ProLIF cells of the marker-score label
(S-12-d, tables 4a to 4c); the mitochondrial metrics, the flags of the marker-score ProLIF cells and the exclusion of
flagged cells against random removals (S-02-d, tables 2d, 6h, 7e and 7f); cycling markers by atlas class and by
SCP-MS group (S-11-c, sheet 5).

What the assembly changes (recorded per table in the sources table):
- conditions: Ctrl -> Control, ZeroK -> Zero-K; the conditions are listed one after the other and never compared;
- labels: the cluster annotation keeps its value Unknown, and every sheet that shows it carries the note
  "Unknown = DCT, subtype not resolved (cells of clusters without marker support)"; cs_<cond>_cluster_label is the
  "marker-score label (clusters)"; the dct_prolif route-A call of the no-floor run is the "per-cell test";
- model columns: label_only_ -> unadjusted_; label_plus_log10_ids_ -> adjusted_ids_;
  label_plus_log10_ids_plus_plate_ -> adjusted_ids_plate_;
- the provenance column of the prevalence table (internal file and task names) is dropped; it is in the stored table.

Workbook: results/tables/report/report_14_S14b_tables.xlsx (sheet Contents, then sheets 1-5; Arial 10;
values only, no formulas).

Inputs (read-only): the nine tables named in SPECS.
Outputs: the workbook; results/tables/report/report_14_s14b_table_{sources,checks}.csv;
         results/tables/report/report_step_14_s14b_table_manifest.json

Command (project root):  python scripts/report_prolif_stress_tables.py --overwrite
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

STEP = "14_s14b_table"
REPORT = "results/tables/report/report_14_prolif_stress_"
DCT_REVIEWER = "results/tables/dct_reviewer/dct_reviewer_"
CA, SP, TEST = "cluster annotation", "marker-score label (clusters)", "per-cell test"
UNKNOWN_LINE = (UNKNOWN_NOTE + ". The Unknown of the marker-score label is a label of its own.",)

PARTITION = {"all_cells": "all cells", "groups_of_figure_3a": "groups of Figure 3A",
             "marker_score_label_clusters": SP}
GROUP = {"all": "all cells"}
COMPARISON = {"ProLIF_vs_all_other_cells": "ProLIF minus all other cells",
              "ProLIF_vs_DCT1_DCT2": "ProLIF minus DCT1 and DCT2", "ProLIF_vs_DCT1+DCT2": "ProLIF minus DCT1 and DCT2",
              "ProLIF_vs_DCT1": "ProLIF minus DCT1", "ProLIF_vs_DCT2": "ProLIF minus DCT2"}
METRIC = {"stress_mean_z": "stress score (mean z)", "stress_score_genes": "stress score (score_genes)",
          "stress_n_detected": "stress-panel proteins detected (n)",
          "mitocarta_intensity_fraction": "MitoCarta fraction of summed intensity",
          "log10_total_intensity": "log10 total intensity"}
FAMILY = {"planned": "planned", "by_subtype": "by subtype"}
FLAG = {"mito_high": "high on a mitochondrial metric", "mito_low": "low on a mitochondrial metric",
        "mito_any": "high or low on a mitochondrial metric", "depth_low": "low on protein IDs or total intensity",
        "depth_high": "high on protein IDs or total intensity", "any": "any of these flags",
        "mito_flagged_either_label_set": "mitochondrial flag under either annotation"}
MODEL_COLUMNS = ((r"^label_only_", "unadjusted_"), (r"^label_plus_log10_ids_plus_plate_", "adjusted_ids_plate_"),
                 (r"^label_plus_log10_ids_", "adjusted_ids_"))
MODEL_NOTE = ("Mann-Whitney U test with effect sizes (group minus reference), and the coefficient of ProLIF in three "
              "linear models (robust HC3 errors): unadjusted_ (ProLIF only), adjusted_ids_ (+ log10 protein IDs) and "
              "adjusted_ids_plate_ (+ log10 protein IDs + sort plate).")

SHEETS = {
    1: ("1 Prevalence", "Cells counted as proliferating, by definition", UNKNOWN_LINE),
    2: ("2 Stress panel", "Dissociation-stress panel: proteins quantified, detection by group, scores against protein "
                          "depth", UNKNOWN_LINE),
    3: ("3 Stress by group", "Stress scores, mitochondrial fraction and total intensity of ProLIF cells against other "
                             "cells", UNKNOWN_LINE),
    4: ("4 Quality flags", "Mitochondrial and depth flags among the ProLIF cells of the per-cell test", ()),
    5: ("5 Per cell", "Per-cell values", UNKNOWN_LINE),
}

SPECS = [
    Spec("1a", REPORT + "prevalence.csv", "Cells per definition of a proliferating cell, with Wilson 95% intervals",
         ("k of n_cells: cells that meet the definition among the cells of the subset. Subsets: all cells of the "
          "condition; the two sort plates; quartiles of the number of proteins identified per cell (with their "
          "range); the cells of clusters annotated DCT1, DCT2 or Unknown. Mouse Kidney Atlas rows: cells or nuclei "
          "of the atlas DCT class with at least one transcript count; the same rows apply to both conditions.",),
         drop=("source",), drop_reason="provenance of each row (internal table names); kept in the stored table",
         values={"condition": CONDITION}, dct_unknown=True,
         contents_note="The conditions are listed one after the other and were not tested against each other."),
    Spec("2a", DCT_REVIEWER + "stress_coverage.csv", "Stress panel: the 18 genes and the protein groups quantified",
         ("present: the gene is part of a quantified protein group (protein_in_matrix; Hspa1a and Hspa1b share one "
          "group). n_cells_detected: cells with intensity above zero.",),
         drop_const=("dataset",), values={"condition": CONDITION}, strict=("condition",),
         rename={"matrix_protein": "protein_in_matrix", "pg_genes": "genes_of_protein_group"}),
    Spec("2b", REPORT + "panel_detection.csv", "Cells with each quantified stress-panel protein detected, by group",
         ("Groups of Figure 3A: ProLIF = cells of the per-cell test; the other cells by the annotation of their "
          f"cluster. {SP}: its four labels.",),
         values={"condition": CONDITION, "partition": PARTITION, "group": GROUP}, strict=("condition", "partition"),
         rename={"partition": "grouping"}, dct_unknown=True),
    Spec("2c", REPORT + "score_vs_depth.csv", "Stress scores against the number of proteins identified per cell",
         ("Spearman correlation with log10 protein IDs, over all cells of a condition and within the groups of "
          "Figure 3A; ci_: Fisher-z 95% interval.",),
         values={"condition": CONDITION, "group": GROUP, "score": METRIC}, strict=("condition", "score"),
         dct_unknown=True),
    Spec("3a", REPORT + "groups.csv", "Groups: cells, sort plate, protein depth, stress scores, mitochondrial fractions "
         "and flags",
         ("One row per group. _median, _q25, _q75: median and quartiles over the cells of the group. n_flag_ columns: "
          "cells flagged as outliers within the cluster annotation of their cell (more than 3 scaled median absolute "
          "deviations from the median of that label). n_mito_flagged_either_annotation: cells with a mitochondrial "
          "flag under the cluster annotation or the marker-score label.",),
         values={"condition": CONDITION, "partition": PARTITION, "group": GROUP}, strict=("condition", "partition"),
         rename={"partition": "grouping", "n_mito_flagged_either_label_set": "n_mito_flagged_either_annotation",
                 "n_per_cell_prolif": "n_prolif_per_cell_test"},
         rename_re=((r"^n_detected_", "protein_ids_"),), dct_unknown=True),
    Spec("3b", REPORT + "contrasts.csv", f"ProLIF cells of the {TEST} against other cells",
         ("family planned: the two comparisons fixed before the tabulation (all other cells; the other cells of "
          "clusters annotated DCT1 or DCT2; in Zero-K no cluster is annotated DCT2). family by subtype: DCT1 and "
          "DCT2 on their own. " + MODEL_NOTE + " _p_fdr_bh: Benjamini-Hochberg within condition and family.",),
         values={"condition": CONDITION, "family": FAMILY, "comparison": COMPARISON, "metric": METRIC},
         strict=("condition", "family", "comparison", "metric"), rename_re=MODEL_COLUMNS),
    Spec("3c", DCT_REVIEWER + "stress_prolif_contrasts.csv", f"ProLIF cells of the {SP} against DCT1 and DCT2 cells",
         (MODEL_NOTE + " _p_fdr_bh: Benjamini-Hochberg within condition.",),
         rows=(("label_role", ("second",)),), drop_const=("label_role", "status"),
         drop_dependent=(("label_column", "condition"),),
         values={"condition": CONDITION, "comparison": COMPARISON, "metric": METRIC},
         strict=("condition", "comparison", "metric"), rename_re=MODEL_COLUMNS),
    Spec("4a", REPORT + "flags.csv", f"Flagged cells among the ProLIF cells of the {TEST} and among the reference cells",
         ("Flags: outliers within the cluster annotation of the cell (more than 3 scaled median absolute deviations "
          "from the median of that label) on a mitochondrial metric (MitoCarta or OXPHOS fraction of summed "
          "intensity, MitoCarta fraction of identified proteins) or on depth (protein IDs, total intensity). "
          "Fisher's exact test; p_fdr_bh: Benjamini-Hochberg within condition.",),
         values={"condition": CONDITION, "comparison": COMPARISON, "flag": FLAG},
         strict=("condition", "comparison", "flag")),
    Spec("5a", REPORT + "per_cell.csv", "Per-cell values",
         ("One row per retained cell (729 Control, 702 Zero-K). detected_ columns: the protein has intensity above "
          "zero in the cell. prolif_z, prolif_q: score and Benjamini-Hochberg q of the per-cell test. flag_ columns: "
          "see sheet 4.",),
         values={"condition": CONDITION}, strict=("condition",),
         rename={"n_detected": "protein_ids", "dct_label": "cluster_annotation",
                 "second_label": "marker_score_label_clusters", "marker_score_cell_label": "marker_score_label_per_cell",
                 "group": "figure_3a_group", "is_prolif": "prolif_per_cell_test",
                 "mito_flagged_either_label_set": "mito_flag_either_annotation"},
         rename_re=((r"^flag__primary__", "flag_"), (r"^det_", "detected_")), dct_unknown=True),
]

TERMS = [
    ("Data", "Sorted DCT cells of mice fed a control or a zero-potassium (Zero-K) diet for 4 days; one pool of 3 mice "
     "per condition; 729 Control and 702 Zero-K cells retained; one search of all runs. The two conditions are shown "
     "side by side and were not tested against each other. Sort plates P1 and P2 are plates of the pooled cells."),
    ("cluster annotation", "Cell types assigned to Leiden clusters (resolution 1) by differential expression of marker "
     "proteins, for each condition on its own (DCT1, DCT2, Unknown). " + UNKNOWN_NOTE + ". No cluster is annotated "
     "ProLIF."),
    ("marker-score label", MARKER_SCORE + " It is the only cluster-level label that contains ProLIF."),
    ("per-cell test (ProLIF)", "A cell is called cycling when the fraction of the 36 proliferation-set proteins "
     "detected in it exceeds the fraction expected from 1,000 random protein sets of the same detection rate "
     "(Benjamini-Hochberg q < 0.05); a cycling cell in which Slc12a3 is detected is labelled ProLIF. Mki67 and Top2a "
     "are not used. These are the ProLIF cells of Figure 3A (Supplementary Table S-12-d, sheets 4 and 5)."),
    ("groups of Figure 3A", "ProLIF: the cells of the per-cell test; DCT1, DCT2 and Unknown: every other cell by the "
     "annotation of its cluster."),
    ("stress panel; stress scores", "Eighteen genes reported to be induced by tissue dissociation in single-cell RNA "
     "sequencing (heat-shock and immediate-early genes). Stress score (mean z): mean of the z-scored ln(1 + "
     "intensity) of the quantified panel proteins; stress score (score_genes): panel mean minus the mean of "
     "expression-matched control proteins (Scanpy score_genes); stress-panel proteins detected: number of panel "
     "protein groups with intensity above zero in the cell."),
    ("MitoCarta fraction of summed intensity", "Summed intensity of the MitoCarta3.0 proteins of a cell divided by "
     "its total intensity."),
    ("protein IDs", "Number of proteins identified (intensity above zero) in a cell."),
    ("Wilson 95% interval", "Confidence interval of a proportion (ci_low_percent, ci_high_percent)."),
    ("cohens_d; rank_biserial", "Effect sizes of a two-group comparison (group minus reference)."),
]

_PREV = "per-cell test: ProLIF (cycling, NCC detected)"
_EITHER = "Mki67 or Top2a detected"
LETTER = [
    ("control: ProLIF cells of the per-cell test, all cells", "1a",
     lambda f: f"{_pick(f, 'k', condition='Control', definition=_PREV, subset='all cells')} of "
               f"{_pick(f, 'n_cells', condition='Control', definition=_PREV, subset='all cells')}", "37 of 729", None),
    ("zero-K: ProLIF cells of the per-cell test, all cells", "1a",
     lambda f: f"{_pick(f, 'k', condition='Zero-K', definition=_PREV, subset='all cells')} of "
               f"{_pick(f, 'n_cells', condition='Zero-K', definition=_PREV, subset='all cells')}", "63 of 702", None),
    ("control: ProLIF cells of the per-cell test, percent", "1a",
     lambda f: _pick(f, "percent", condition="Control", definition=_PREV, subset="all cells"), 5.1, 1),
    ("zero-K: ProLIF cells of the per-cell test, percent", "1a",
     lambda f: _pick(f, "percent", condition="Zero-K", definition=_PREV, subset="all cells"), 9.0, 1),
    ("control and zero-K: cells with Mki67 or Top2a detected", "1a",
     lambda f: f"{_pick(f, 'k', condition='Control', definition=_EITHER, subset='all cells')}; "
               f"{_pick(f, 'k', condition='Zero-K', definition=_EITHER, subset='all cells')}", "21; 56", None),
    ("control and zero-K: ProLIF cells of the marker-score label (clusters)", "1a",
     lambda f: f"{_pick(f, 'k', condition='Control', definition='marker-score label (clusters): ProLIF', subset='all cells')}; "
               f"{_pick(f, 'k', condition='Zero-K', definition='marker-score label (clusters): ProLIF', subset='all cells')}",
     "119; 98", None),
    ("atlas DCT class: cells and nuclei with Mki67 or Top2a transcripts, percent", "1a",
     lambda f: _pick(f, "percent", definition="Mki67 or Top2a transcripts detected",
                     subset="atlas DCT class, cells and nuclei"), 1.3, 1),
    ("control: stress-panel protein groups quantified", "2a",
     lambda f: f[(f["condition"] == "Control") & (f["present"].astype(str).str.lower() == "true")]
     ["protein_in_matrix"].nunique(), 7, None),
    ("zero-K: stress-panel protein groups quantified", "2a",
     lambda f: f[(f["condition"] == "Zero-K") & (f["present"].astype(str).str.lower() == "true")]
     ["protein_in_matrix"].nunique(), 6, None),
    ("control: stress score against log10 protein IDs, all cells, Spearman rho", "2c",
     lambda f: _pick(f, "rho", condition="Control", group="all cells", score=METRIC["stress_mean_z"]), 0.81, 2),
    ("zero-K: stress score against log10 protein IDs, all cells, Spearman rho", "2c",
     lambda f: _pick(f, "rho", condition="Zero-K", group="all cells", score=METRIC["stress_mean_z"]), 0.79, 2),
    ("control: per-cell ProLIF minus all other cells, stress score, adjusted for protein IDs and sort plate", "3b",
     lambda f: _pick(f, "adjusted_ids_plate_coef", condition="Control", family="planned",
                     comparison=COMPARISON["ProLIF_vs_all_other_cells"], metric=METRIC["stress_mean_z"]), 0.006, 3),
    ("zero-K: per-cell ProLIF minus all other cells, stress score, adjusted for protein IDs and sort plate", "3b",
     lambda f: _pick(f, "adjusted_ids_plate_coef", condition="Zero-K", family="planned",
                     comparison=COMPARISON["ProLIF_vs_all_other_cells"], metric=METRIC["stress_mean_z"]), -0.011, 3),
    ("zero-K: per-cell ProLIF minus all other cells, stress score, Mann-Whitney p", "3b",
     lambda f: _pick(f, "mannwhitney_p", condition="Zero-K", family="planned",
                     comparison=COMPARISON["ProLIF_vs_all_other_cells"], metric=METRIC["stress_mean_z"]), 0.014, 3),
    ("control: per-cell ProLIF minus DCT1 and DCT2, MitoCarta fraction, adjusted", "3b",
     lambda f: _pick(f, "adjusted_ids_plate_coef", condition="Control", family="planned",
                     comparison=COMPARISON["ProLIF_vs_DCT1_DCT2"], metric=METRIC["mitocarta_intensity_fraction"]),
     -0.057, 3),
    ("control: marker-score ProLIF minus DCT1 and DCT2, stress score, adjusted", "3c",
     lambda f: _pick(f, "adjusted_ids_plate_coef", condition="Control", comparison="ProLIF minus DCT1 and DCT2",
                     metric=METRIC["stress_mean_z"]), -0.069, 3),
    ("zero-K: marker-score ProLIF minus DCT1 and DCT2, stress score, adjusted", "3c",
     lambda f: _pick(f, "adjusted_ids_plate_coef", condition="Zero-K", comparison="ProLIF minus DCT1 and DCT2",
                     metric=METRIC["stress_mean_z"]), 0.065, 3),
    ("control: per-cell ProLIF cells with a mitochondrial flag; other cells with one", "4a",
     lambda f: f"{_pick(f, 'n_prolif_flagged', condition='Control', comparison=COMPARISON['ProLIF_vs_all_other_cells'], flag=FLAG['mito_any'])} of 37; "
               f"{_pick(f, 'n_reference_flagged', condition='Control', comparison=COMPARISON['ProLIF_vs_all_other_cells'], flag=FLAG['mito_any'])} of 692",
     "0 of 37; 105 of 692", None),
    ("zero-K: per-cell ProLIF cells with a mitochondrial flag; other cells with one", "4a",
     lambda f: f"{_pick(f, 'n_prolif_flagged', condition='Zero-K', comparison=COMPARISON['ProLIF_vs_all_other_cells'], flag=FLAG['mito_any'])} of 63; "
               f"{_pick(f, 'n_reference_flagged', condition='Zero-K', comparison=COMPARISON['ProLIF_vs_all_other_cells'], flag=FLAG['mito_any'])} of 639",
     "0 of 63; 97 of 639", None),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    cells = stored["5a"]
    got = {c: cells.loc[cells["condition"].eq(c), "figure_3a_group"].value_counts().to_dict()
           for c in ("Control", "Zero-K")}
    expected = {"Control": {"DCT1": 362, "DCT2": 79, "ProLIF": 37, "Unknown": 251},
                "Zero-K": {"DCT1": 402, "ProLIF": 63, "Unknown": 237}}
    lib.check(checks, "table 5a: cells per group of Figure 3A (729 Control, 702 Zero-K)", got, expected, got == expected)


WORKBOOK = WorkbookDef(
    label="S-14-b", step=STEP, workbook="report_14_S14b_tables.xlsx",
    title="Supplementary Table S-14-b. ProLIF cells in the DCT data: prevalence by definition, dissociation-stress "
          "proteins, mitochondrial fraction and quality flags",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names are "
             "explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"dct": "combined control + zero-K search (dct_labels labels, dct_reviewer tables, dct_prolif per-cell test, the response to the reviewers tables)",
                "unknown_note": UNKNOWN_NOTE,
                "vocabulary": "cs_<cond>_cluster_label = marker-score label (clusters); dct_prolif route A, no-floor run = "
                              "per-cell test",
                "not_copied": "S-12-d tables 4a-4d and 5a; S-02-d tables 2d, 6h, 7e, 7f; S-11-c sheet 5"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-14-b.")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
