#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-02-d (mitochondrial metrics), one workbook with seven sheets.

The authors asked on 12 Nov 2026 to assemble Supplementary
Tables S-01-a, S-02-d and S-03-b ("Yes, assemble them"). The response to the reviewers, section 5, fixes the content of S-02-d: (1)
per-cell metrics and flags, kidney and DCT; (2) summary by label; (3) variance explained; (4) global cutoffs and
tests; (5) within-label correlations; (6) removed against retained wells and flags by metric; (7) exclusion sets
and outcomes against random removals, including the ProLIF rows. It is built from 29 stored tables (qc_mito, kidney_annotation
addendum and sensitivity run, dct_reviewer, and the flag counts of the response to the reviewers). No new statistics: every value is copied.

What the assembly changes (all recorded per table in the sources table):
- kidney: only the rows of the revised annotation (label_column kidney_t15_label; one-annotation rule, authors 30 Oct
  2026); the per-cell table takes kidney_t15_label from the kidney_annotation label table (matched on cell_id) and loses the
  labels and the flag columns of the submitted annotation; display names DCT1 -> DCT, PC -> CNT/PC;
- DCT: the dct_reviewer tables (combined control + zero-K search) are taken whole; Ctrl / ZeroK -> Control / Zero-K; the
  label columns are named cluster annotation (cs_ctrl_deg_label, cs_zerok_pipeline_label) and marker-score label
  (clusters) (cs_ctrl_cluster_label, cs_zerok_cluster_label; the per-cell label behind it is the marker-score
  label (per cell)); the label value Unknown stays in the cells and every sheet that shows it carries the note
  "Unknown = DCT, subtype not resolved (cells of clusters without marker support)";
- final products only (authors, 12 Nov 2026: "The readers will not know the revision process, so we only show the
  final products"): no cell, column name or sheet name refers to the earlier manuscript; a check enforces it;
- rows and columns of the rule `mt_prefix_task1` (a rule of an earlier draft, not used) are dropped; the qc_mito
  tables lose their rows of the retired 758-cell DCT data;
- columns that only name the data set or the label column are dropped after a check that they are constant or
  fixed by a kept column; columns without any value are dropped.

Placement not fixed by the plan (noted in the sources table): the kidney protein-set table (2a), the ProLIF
contrasts (2d) and the ProLIF flag table (6h). One infinite odds ratio (table 6h, three cells) is written as the
text inf, because a workbook cannot hold an infinite number.

Workbook: results/tables/report/report_02_S02d_tables.xlsx (sheet Contents, then sheets 1-7), in the
layout of Supplementary Table S-07-d (shared code: report_table_lib.py).

Inputs (read-only): the 29 tables named in SPECS and the kidney_annotation label table.
Outputs: the workbook; results/tables/report/report_02_s02d_table_{sources,checks}.csv;
         results/tables/report/report_step_02_s02d_table_manifest.json

Command (project root):  python scripts/report_mito_tables.py --overwrite
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
import re

import pandas as pd

import scp_common as cc
import report_table_lib as lib
from report_table_lib import (ALL_CELLS, CONDITION, DATASET, KIDNEY_LABEL, KIDNEY_TEXT, LABEL_SET,
                                       LABEL_TEXT, MARKER_SCORE, UNKNOWN_NOTE, Join, Spec, WorkbookDef, _pick)

STEP = "02_s02d_table"
QC_MITO = "results/tables/qc_mito/qc_mito_"
T15A = "results/tables/kidney_annotation/kidney_annotation_addendum_t03_"
T15S = "results/tables/kidney_annotation/kidney_annotation_sensitivity_"
KIDNEY_ANNOTATION_LABELS = "results/tables/kidney_annotation/kidney_annotation_cluster_labels_per_cell_a_endo.csv"
T28M = "results/tables/dct_reviewer/dct_reviewer_mito_"
T28E = "results/tables/dct_reviewer/dct_reviewer_exclusion_"
FLAGS = "results/tables/report/report_02_mito_flags_by_metric.csv"

CA, SP, PLATE = "cluster annotation", "marker-score label (clusters)", "sort plate"
SP_COLUMN = "marker_score_label_clusters"                     # the per-cell table column that holds that label
KIDNEY_TYPES = {"PCT": 160, "PST": 158, "TAL": 138, "DCT": 58, "CNT/PC": 38, "JG": 24, "Endothelial": 19,
                "IC-B": 8, "Unknown": 109}                                     # the project notes, with the display names
DCT_COUNTS = {("Control", "cluster_annotation"): {"DCT1": 382, "DCT2": 85, "Unknown": 262},
              ("Zero-K", "cluster_annotation"): {"DCT1": 449, "Unknown": 253},
              ("Control", SP_COLUMN): {"DCT1": 460, "DCT2": 126, "ProLIF": 119,
                                                                 "Unknown": 24},
              ("Zero-K", SP_COLUMN): {"DCT1": 469, "DCT2": 114, "ProLIF": 98,
                                                                "Unknown": 21}}   # the project notes (dct_labels)
NOT_USED = "mt_prefix_task1: a rule of an earlier draft, not used in the revision"
PLACED = "The plan names this file without a sheet; it was placed here at assembly."
PROLIF_REASON = ((r"no ProLIF cell under this column \(dct_labels\)", "no ProLIF cell in the cluster annotation"),)
ABSENT_REASON = ((r" under cs_(ctrl_deg|zerok_pipeline)_label\)", " in the cluster annotation)"),)
UNKNOWN_LINE = (f"DCT tables, cluster annotation: {UNKNOWN_NOTE}. In the marker-score label (clusters), Unknown "
                "is a different label. Kidney tables: Unknown = cells without a marker-based call.",)

# Pieces shared by the kidney tables of the revised annotation and by the DCT tables with a label column.
KIDNEY = dict(rows=(("label_column", ("kidney_t15_label",)),), kidney=True)
KIDNEY_BY_TYPE = dict(**KIDNEY, drop_const=("dataset", "label_column"), values={"label": KIDNEY_LABEL},
                      rename={"label": "cell_type"})
DCT_SETS = dict(strict=("condition", "label_column"), drop_dependent=(("label_role", "label_column"),),
                rename={"label_column": "label_set"})
DCT_VALUES = {"condition": CONDITION, "label_column": LABEL_SET}

SHEETS = {
    1: ("1 Per-cell metrics", "Mitochondrial metrics per cell, and flags", UNKNOWN_LINE),
    2: ("2 Summary by label", "Mitochondrial protein sets, metrics by label, and ProLIF contrasts", UNKNOWN_LINE),
    3: ("3 Variance explained", "Variance of the mitochondrial fractions explained by label and depth", ()),
    4: ("4 Global cutoffs", "Simulated global cutoffs on the mitochondrial fraction, and their tests", UNKNOWN_LINE),
    5: ("5 Within-label correlation", "Correlation of the mitochondrial fractions with depth within each label",
        UNKNOWN_LINE),
    6: ("6 Removed wells and flags", "Removed against retained wells, and outlier flags", UNKNOWN_LINE),
    7: ("7 Exclusion vs random", "Exclusion of the flagged cells against random removals", UNKNOWN_LINE),
}

SPECS = [
    Spec("1a", QC_MITO + "kidney_qc_per_cell.csv", "Total kidney: mitochondrial metrics per cell",
         ("One row per retained cell (712). cell_type: cell type of Figure 2A. Columns: see Contents (Protein sets; "
          "_intensity_fraction, _id_fraction).",),
         join=Join(KIDNEY_ANNOTATION_LABELS, "cell_id", "kidney_t15_label", after="cell_id"), drop_const=("plate",),
         drop=("kidney_reannot_label", "kidney_pre_label"),
         drop_re=r"^(outlier|evaluated|flag)__(reannot|pre)(__|$)|mt_prefix_task1",
         drop_reason="labels and flag columns of the submitted annotation; " + NOT_USED,
         values={"kidney_t15_label": KIDNEY_LABEL}, rename={"kidney_t15_label": "cell_type"}, kidney=True,
         contents_note="Flags per cell are not given for the kidney cells; their counts per cell type are in "
                       "tables 6c to 6e.",
         assembly_note="The plan asks for per-cell metrics and flags. The stored per-cell flags of the kidney cells "
                       "belong to the submitted annotation and are dropped; under kidney_t15_label flags are "
                       "stored as counts only (tables 6c to 6e), so table 1a holds no flag column."),
    Spec("1b", T28M + "qc_per_cell.csv", "DCT, Control and Zero-K: mitochondrial metrics and flags per cell",
         ("One row per retained cell (Control 729, Zero-K 702), with its label in both label sets.",
          "outlier__ columns: high, low or none against the cells of the same label; flag__ columns: see Contents "
          "(Flags). The middle part of the name gives the label set."),
         values={"condition": CONDITION}, strict=("condition",), drop_re=r"mt_prefix_task1", drop_reason=NOT_USED,
         rename={"plate": "sort_plate", "primary_label": "cluster_annotation",
                 "second_label": SP_COLUMN},
         rename_re=((r"__primary(__|$)", r"__cluster_annotation\1"),
                    (r"__second(__|$)", r"__marker_score_clusters\1")),
         dct_unknown=True),
    Spec("2a", QC_MITO + "mito_coverage.csv", "Total kidney: mitochondrial protein sets in the protein matrix",
         ("n_protein_groups_in_matrix: proteins of the set among the 4,054 retained proteins. Rows without a "
          "definition are matching diagnostics; proteins: the proteins of small sets.",),
         rows=(("dataset", ("kidney",)),), drop_rows=(("protein_set", "mt_prefix_task1"),), drop_const=("dataset",),
         kidney=True, assembly_note=PLACED + " The plan lists no such table for the DCT data (876 of 6,078 and 876 "
                                    "of 5,953 MitoCarta proteins are not in this workbook)."),
    Spec("2b", T15A + "qc_summary_by_label.csv", "Total kidney: metrics by cell type",
         ("One row per cell type and metric; all cells: the 712 retained cells. Columns: see Contents (Summary "
          "columns).",),
         drop_rows=(("metric", "mt_prefix_task1"),), **KIDNEY_BY_TYPE),
    Spec("2c", T28M + "qc_summary_by_label.csv", "DCT, Control and Zero-K: metrics by label",
         ("One row per condition, label set, label and metric; all cells: the retained cells of the condition.",),
         drop_rows=(("metric", "mt_prefix_task1"),), drop_const=("dataset",),
         values={**DCT_VALUES, "label": ALL_CELLS}, dct_unknown=True, **DCT_SETS),
    Spec("2d", T28M + "prolif_contrasts.csv", "DCT, Control and Zero-K: ProLIF cells against DCT1 and DCT2 cells",
         ("Marker-score label (clusters) only: the cluster annotation has no ProLIF cells (status does not "
          "apply). Columns: see Contents (ProLIF tables).",),
         values=DCT_VALUES, value_re={"reason": PROLIF_REASON}, assembly_note=PLACED, **DCT_SETS),
    Spec("3a", T15A + "variance_explained.csv", "Total kidney: variance explained by cell type and depth",
         ("One row per metric, scale and depth variable. Columns: see Contents (Variance explained).",),
         drop_const=("dataset", "label_column"), **KIDNEY),
    Spec("3b", T28M + "variance_explained.csv", "DCT, Control and Zero-K: variance explained by label, depth and "
         "sort plate",
         ("One row per condition, label set, metric, scale and depth variable.",),
         drop_const=("dataset",), values=DCT_VALUES, **DCT_SETS),
    Spec("4a", T15A + "global_cutoff_simulation.csv", "Total kidney: cells above each simulated cutoff, by cell type",
         ("No cell was excluded. Columns: see Contents (cutoff; Cutoff columns).",), **KIDNEY_BY_TYPE),
    Spec("4b", T15A + "global_cutoff_tests.csv", "Total kidney: dependence of each cutoff on cell type",
         ("Chi-square statistic of cell type against removed, with its permutation p value.",),
         drop_const=("dataset", "label_column"), **KIDNEY),
    Spec("4c", T28M + "global_cutoff_simulation.csv", "DCT, Control and Zero-K: cells above each simulated cutoff, "
         "by label",
         ("No cell was excluded. One block per condition and label set (and per sort plate).",),
         drop_const=("dataset",), values={**DCT_VALUES, "label": ALL_CELLS}, dct_unknown=True, **DCT_SETS),
    Spec("4d", T28M + "global_cutoff_tests.csv", "DCT, Control and Zero-K: dependence of each cutoff on label",
         ("Chi-square statistic of label (or sort plate) against removed, with its permutation p value.",),
         drop_const=("dataset",), values=DCT_VALUES, **DCT_SETS),
    Spec("5a", T15A + "within_type_depth_correlation.csv", "Total kidney: correlation with depth within each cell "
         "type",
         ("Columns: see Contents (Correlation columns). IC-B was not evaluated (fewer than 10 cells).",),
         **KIDNEY_BY_TYPE),
    Spec("5b", T28M + "within_type_depth_correlation.csv", "DCT, Control and Zero-K: correlation with depth within "
         "each label",
         ("Columns: see Contents (Correlation columns).",),
         drop_const=("dataset",), values={**DCT_VALUES, "label": ALL_CELLS}, dct_unknown=True, **DCT_SETS),
    Spec("6a", QC_MITO + "removed_vs_retained.csv", "Total kidney: wells removed by the filter against retained cells",
         ("Columns: see Contents (Removed against retained).",),
         rows=(("dataset", ("kidney",)),), drop_const=("dataset",), kidney=True),
    Spec("6b", T28M + "removed_vs_retained.csv", "DCT, Control and Zero-K: wells removed by the filter against "
         "retained cells",
         ("Columns: see Contents (Removed against retained).",),
         drop_const=("dataset",), values={"condition": CONDITION}, strict=("condition",)),
    Spec("6c", FLAGS, "Flagged cells by metric: total kidney, DCT Control and DCT Zero-K",
         ("Cells flagged within their label, per metric. Columns: see Contents (Flags; Flags by metric).",),
         values={"dataset": DATASET, "grouping": {"kidney_t15_label": "cell types", "dct_labels primary label column": CA,
                                                  "dct_labels second label column": SP}},
         strict=("dataset", "grouping"), rename={"grouping": "label_set"}),
    Spec("6d", T15A + "outlier_counts.csv", "Total kidney: flagged cells by cell type",
         ("n_<metric>_high, n_<metric>_low: cells flagged on one metric. Columns: see Contents (Flags).",),
         **KIDNEY_BY_TYPE),
    Spec("6e", T15A + "outlier_thresholds.csv", "Total kidney: flag thresholds by cell type and metric",
         ("Columns: see Contents (Threshold columns).",), **KIDNEY_BY_TYPE),
    Spec("6f", T28M + "outlier_counts.csv", "DCT, Control and Zero-K: flagged cells by label",
         ("n_<metric>_high, n_<metric>_low: cells flagged on one metric. Columns: see Contents (Flags).",),
         drop_const=("dataset",), values={**DCT_VALUES, "label": ALL_CELLS}, dct_unknown=True, **DCT_SETS),
    Spec("6g", T28M + "outlier_thresholds.csv", "DCT, Control and Zero-K: flag thresholds by label and metric",
         ("Columns: see Contents (Threshold columns).",),
         drop_const=("dataset",), values=DCT_VALUES, dct_unknown=True, **DCT_SETS),
    Spec("6h", T28M + "outlier_prolif_enrichment.csv", "DCT, Control and Zero-K: flagged ProLIF cells against "
         "flagged DCT1 and DCT2 cells",
         ("Marker-score label (clusters) only: the cluster annotation has no ProLIF cells (status does not "
          "apply). Columns: see Contents (ProLIF tables). inf: an infinite odds ratio.",),
         values=DCT_VALUES, value_re={"reason": PROLIF_REASON}, assembly_note=PLACED, **DCT_SETS),
    Spec("7a", T15S + "exclusion_sets.csv", "Total kidney: exclusion sets",
         ("Cells flagged within their cell type, with the number per cell type.",),
         drop_const=("label_column",), value_re={"excluded_by_label": KIDNEY_TEXT},
         rename={"excluded_by_label": "excluded_by_cell_type"}, **KIDNEY),
    Spec("7b", T15S + "vs_random.csv", "Total kidney: outcomes after exclusion against 100 random removals",
         ("Columns: see Contents (Exclusion against random). position: observed inside, below or above the central "
          "95% of the random removals.",),
         values={"outcome": {"recall_DCT1": "recall_DCT", "recall_PC": "recall_CNT/PC"}}, kidney=True),
    Spec("7c", T15S + "runs.csv", "Total kidney: outcomes of every run",
         ("One row per run: baseline (all cells; Leiden seeds 0 to 3 and 42), observed (after exclusion; the same "
          "seeds) and random (100 draws per exclusion set, numbered in draw; Leiden seed 42).",
          "types: cell types found. recall_ and n_ columns: per cell type."),
         value_re={"types": KIDNEY_TEXT},
         rename={"recall_DCT1": "recall_DCT", "n_DCT1": "n_DCT", "recall_PC": "recall_CNT/PC", "n_PC": "n_CNT/PC"},
         kidney=True),
    Spec("7d", T28E + "sets.csv", "DCT, Control and Zero-K: exclusion sets",
         ("Cells flagged under the cluster annotation or under the marker-score label (clusters), with the "
          "number per label of the cluster annotation.",),
         drop_const=("dataset",), values={"condition": CONDITION}, strict=("condition",),
         rename={"excluded_by_primary_label": "excluded_by_cluster_annotation"}, dct_unknown=True),
    Spec("7e", T28E + "composition.csv", "DCT, Control and Zero-K: composition of the exclusion sets",
         ("n_before, n_after: cells of the label before and after exclusion; share_before, share_after: their "
          "share of all cells. Includes the ProLIF rows of the marker-score label (clusters).",),
         drop_const=("dataset",), values=DCT_VALUES, dct_unknown=True, **DCT_SETS),
    Spec("7f", T28E + "vs_random.csv", "DCT, Control and Zero-K: outcomes after exclusion against 100 random removals",
         ("Columns: see Contents (Exclusion against random). applicable = FALSE: the outcome needs a label that is "
          "absent in the cluster annotation; the comparator_ columns then give the matching outcome under a "
          "marker-score label.",
          "p_empirical_ columns: empirical p values against the random removals (lower tail, upper tail, and "
          "two-sided = twice the smaller, at most 1)."),
         drop_const=("dataset",), drop=("outcome",),
         drop_reason="outcome: internal code of the outcome; its text (display) is kept under the name outcome",
         values={"condition": CONDITION}, strict=("condition",),
         value_re={"display": LABEL_TEXT, "reason": ABSENT_REASON},
         map_by=(("comparator_outcome", "outcome", "display"),), rename={"display": "outcome"}),
    Spec("7g", T28E + "leiden_seeds.csv", "DCT, Control and Zero-K: outcomes over five Leiden seeds",
         ("Smallest, median and largest value over the Leiden seeds, with all cells (baseline) and after exclusion "
          "(observed).",
          "_rerun_agree_: cells that keep their label after reclustering (cluster annotation: Leiden 1.0; "
          "marker-score label (clusters): Leiden 3.0). ari_leiden1_, ari_leiden3_: adjusted Rand index of "
          "the rerun clusters with the stored clusters."),
         values={"condition": CONDITION}, strict=("condition",),
         rename_re=((r"^deg_rerun_agree_", "cluster_annotation_rerun_agree_"),
                    (r"^cluster_rerun_agree_", "marker_score_clusters_rerun_agree_"), (r"^ari1_", "ari_leiden1_"),
                    (r"^ari3_", "ari_leiden3_"))),
]

TERMS = [
    ("Datasets", "Total kidney: 712 retained cells, 4,054 proteins. DCT: sorted DCT cells of mice fed a control or "
     "a zero-potassium (Zero-K) diet for 4 days; one pool of 3 mice per condition; 729 Control and 702 Zero-K cells "
     "retained. The two conditions are shown side by side and were not tested against each other."),
    ("cell_type; label; label_set", "Total kidney: cell types of Figure 2A (label_set cell types). DCT, label_set "
     "cluster annotation: cell types assigned to Leiden clusters by differential expression (DE) of marker "
     "proteins, for each condition on its own (DCT1, DCT2, Unknown). DCT, label_set marker-score label "
     "(clusters): see Marker-score label. label_set sort plate: the "
     "two 384-well sort plates (P1, P2) of the pooled cells of a condition; technical plates, not animals."),
    ("Marker-score label", f"{MARKER_SCORE} Its labels are DCT1, DCT2 and ProLIF, and for clusters also Unknown."),
    ("Unknown", f"DCT, cluster annotation: {UNKNOWN_NOTE}. DCT, marker-score label (clusters): a different "
     "label. Total kidney: cells without a marker-based call."),
    ("n_detected_proteins; IDs", "Identified proteins of a cell (protein groups with intensity above zero); "
     "log10_n_detected: its log10. total_intensity: summed intensity of the cell; log10_total_intensity: its log10."),
    ("Protein sets", "mitocarta: MitoCarta3.0 mouse inventory, matched by gene symbol or UniProt accession. "
     "mitocarta_symbol_only: matched by gene symbol only (sensitivity). oxphos: MitoPathway 'OXPHOS' (subunits and "
     "assembly factors). oxphos_subunit: MitoPathway 'OXPHOS subunits'. mt_encoded: mitochondrially encoded "
     "proteins."),
    ("_intensity_fraction; _id_fraction", "Per cell, over detected proteins only. _intensity_fraction: summed "
     "intensity of the proteins of a set divided by total_intensity. _id_fraction: identified proteins of the set "
     "divided by n_detected_proteins. n_detected_<set>: identified proteins of the set."),
    ("Summary columns", "n_cells: cells of the label; n_finite: cells with a value; mean, median, quartiles (q25, "
     "q75), min and max of the metric over these cells."),
    ("Variance explained", "Ordinary least squares on the fraction (scale raw) or on its rank (scale rank). "
     "r2_label, r2_depth, r2_label_plus_depth: R2 with the label, the depth variable, or both as predictors; adj_: "
     "adjusted R2. unique_r2_label = r2_label_plus_depth - r2_depth; unique_r2_depth = r2_label_plus_depth - "
     "r2_label; shared_r2 = r2_label + r2_depth - r2_label_plus_depth; partial_r2_label_after_depth = "
     "unique_r2_label / (1 - r2_depth). DCT: r2_plate and _plus_plate, with the sort plate as predictor."),
    ("cutoff", "Simulated cutoffs; no cell was excluded. global_top5pct, global_top10pct: the 5% or 10% of cells "
     "with the highest fraction. global_median_plus_3MAD: above the median + 3 scaled MAD of all cells. "
     "within_label_3MAD_high: above the median + 3 scaled MAD of the cell's own label. threshold: the cutoff value."),
    ("Cutoff columns", "n_removed: cells above the cutoff. percent_of_label_removed = n_removed / n_cells x 100. "
     "share_of_all_cells: cells of the label / all cells. share_of_removed: cells of the label among the removed "
     "cells. ratio_share_removed_to_share_cells = share_of_removed / share_of_all_cells. chi2, p_asymptotic: "
     "chi-square test of label against removed; p_permutation: the same statistic against n_permutations "
     "permutations."),
    ("Correlation columns", "rho: Spearman correlation within the label between the metric and the depth variable; "
     "n: cells used; ci_low, ci_high: 95% confidence interval (Fisher z); p_fdr_bh_within_labels: "
     "Benjamini-Hochberg (BH) correction across the labels. Labels with fewer than 10 cells were not evaluated."),
    ("Removed against retained", "Wells removed by the filter (fewer than 100 identified proteins) against retained "
     "cells: all of them, or those in the lowest decile of identified proteins (limit in the comparison). n_group, "
     "median_group: removed wells with at least one identified protein; n_reference, median_reference: retained "
     "cells; median_difference = median_group - median_reference. rank_biserial, cohens_d: effect sizes. "
     "mannwhitney_u, p_value: Mann-Whitney test."),
    ("Flags", "A cell is an outlier on a metric if it lies more than 3 scaled median absolute deviations (MAD x "
     "1.4826) above (high) or below (low) the median of its label; labels with fewer than 10 cells are not "
     "evaluated. mito_high, mito_low, mito_any: outlier on mitocarta_intensity_fraction, mitocarta_id_fraction or "
     "oxphos_intensity_fraction. depth_low, depth_high: outlier on log10_n_detected or log10_total_intensity. any: "
     "outlier on any of the five metrics."),
    ("Threshold columns", "median, mad_scaled: median and scaled MAD of the label; lower, upper = median - / + 3 x "
     "mad_scaled; n_high, n_low: cells above upper or below lower."),
    ("Flags by metric", "flag_source: one metric, or any_mito_metric (any of the three fractions), "
     "id_fraction_only (flagged on mitocarta_id_fraction and on neither intensity fraction), any_intensity_metric "
     "(mitocarta_intensity_fraction or oxphos_intensity_fraction). n_high_and_depth_low, n_low_and_depth_low: of "
     "the cells flagged high or low, those that also carry a low-depth flag."),
    ("ProLIF tables", "ProLIF is a marker-score label. Flag table: flagged cells among ProLIF cells "
     "and among DCT1 + DCT2 cells (reference), with Fisher's exact test over all cells and within each sort plate "
     "(P1_, P2_) and the Cochran-Mantel-Haenszel test across the plates (cmh_). Contrast table: Mann-Whitney test "
     "(group ProLIF, reference DCT1, DCT2 or both) and linear regressions of the metric on the label alone "
     "(label_only_), with log10_n_detected (label_plus_log10_ids_) and with the sort plate in addition "
     "(_plus_plate_): coefficient of the label with 95% CI, p and R2. _fdr_bh: BH correction."),
    ("exclusion_set", "mito_flagged: cells with a mitochondrial flag (mito_any). any_flagged: cells with any flag. "
     "DCT: cells flagged under the cluster annotation or under the marker-score label (clusters)."),
    ("Exclusion against random", "The cells of an exclusion set were removed and the analysis was rebuilt from the "
     "protein matrix (scaling, PCA, neighbor graph, Leiden clustering, annotation). baseline: all cells. observed: "
     "after exclusion. random_: 100 random removals of the same number of cells, stratified by label (median, 2.5% "
     "and 97.5% quantiles; DCT also min and max; random_n: removals with a value). inside_central_95: observed "
     "within the 2.5% to 97.5% range. observed_percentile_in_random: random removals below the observed value "
     "(kidney: fraction; DCT: percent). "
     "Kidney outcomes: agreement (remaining cells that keep their cell type), n_types (cell types found), "
     "unknown_fraction, ari_vs_baseline (adjusted Rand index, ARI, of the clusters with those of all cells), "
     "recall_<cell type> (cells of the cell type that keep it). DCT outcomes are named in the table; PAGA: "
     "connectivity between two labels."),
]


def rows(frame: pd.DataFrame, **sel) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for key, value in sel.items():
        mask &= frame[key].astype(str).eq(str(value))
    return frame[mask]


FRAC = "mitocarta_intensity_fraction"
TOP10 = dict(metric=FRAC, cutoff="global_top10pct")
BY_IDS = dict(metric=FRAC, depth_variable="log10_n_detected")


def inside(f: pd.DataFrame, **sel) -> str:
    """Outcomes with an observed value (and applicable, in the DCT table) inside the central 95% of the random runs."""
    d = rows(f, exclusion_set="mito_flagged", **sel)
    if "applicable" in d.columns:
        d = d[d["applicable"].astype(str).eq("True") & d["observed"].notna()]
    return f"{int(d['inside_central_95'].astype(str).eq('True').sum())} of {len(d)}"


LETTER = [
    ("kidney: MitoCarta proteins, of the matrix proteins", "2a",
     lambda f: f"{_pick(f, 'n_protein_groups_in_matrix', protein_set='mitocarta')} of "
               f"{_pick(f, 'n_matrix_proteins', protein_set='mitocarta')}", "691 of 4054", None),
    ("kidney: mitochondrially encoded proteins quantified", "2a",
     lambda f: _pick(f, "n_protein_groups_in_matrix", protein_set="mt_encoded"), 8, None),
    ("kidney: median MitoCarta fraction of summed intensity, all cells", "2b",
     lambda f: _pick(f, "median", cell_type="all cells", metric=FRAC), 0.344, 3),
    ("kidney: median fraction in DCT cells", "2b", lambda f: _pick(f, "median", cell_type="DCT", metric=FRAC),
     0.458, 3),
    ("kidney: median fraction in Endothelial cells", "2b",
     lambda f: _pick(f, "median", cell_type="Endothelial", metric=FRAC), 0.010, 3),
    ("kidney: median fraction in CNT/PC cells", "2b", lambda f: _pick(f, "median", cell_type="CNT/PC", metric=FRAC),
     0.237, 3),
    ("kidney: quartiles of the identified proteins per cell, all cells (cited by Supplementary Table S-01-a)", "2b",
     lambda f: f"{_pick(f, 'q25', cell_type='all cells', metric='n_detected_proteins'):g}, "
               f"{_pick(f, 'q75', cell_type='all cells', metric='n_detected_proteins'):g}", "357, 1098", None),
    ("control: median fraction, all cells", "2c",
     lambda f: _pick(f, "median", condition="Control", label_set=CA, label="all cells", metric=FRAC), 0.490, 3),
    ("zero-K: median fraction, all cells", "2c",
     lambda f: _pick(f, "median", condition="Zero-K", label_set=CA, label="all cells", metric=FRAC), 0.500, 3),
    ("kidney: variance explained by cell type", "3a",
     lambda f: _pick(f, "r2_label", scale="raw", **BY_IDS), 0.274, 3),
    ("kidney: variance explained by log10 total intensity; by cell type + total intensity", "3a",
     lambda f: (lambda d: f"{d['r2_depth'].iloc[0]:.3f}, {d['r2_label_plus_depth'].iloc[0]:.3f}")(
         rows(f, metric=FRAC, scale="raw", depth_variable="log10_total_intensity")), "0.284, 0.540", None),
    ("control and zero-K: variance explained by the cluster annotation", "3b",
     lambda f: ", ".join(f"{_pick(f, 'r2_label', condition=c, label_set=CA, scale='raw', **BY_IDS):.3f}"
                         for c in ("Control", "Zero-K")), "0.123, 0.087", None),
    ("kidney, top-10% cutoff: threshold", "4b", lambda f: _pick(f, "threshold", **TOP10), 0.601, 3),
    ("kidney, top-10% cutoff: cells removed, permutation p", "4b",
     lambda f: f"{_pick(f, 'n_removed', **TOP10)}, {_pick(f, 'p_permutation', **TOP10):.4f}", "72, 0.0001", None),
    ("kidney, top-10% cutoff: DCT, TAL, PCT, PST, Unknown cells removed, of the cell type", "4a",
     lambda f: ", ".join(f"{_pick(f, 'n_removed', cell_type=t, **TOP10)}/{_pick(f, 'n_cells', cell_type=t, **TOP10)}"
                         for t in ("DCT", "TAL", "PCT", "PST", "Unknown")),
     "11/58, 20/138, 16/160, 4/158, 21/109", None),
    ("kidney, top-10% cutoff: CNT/PC, IC-B, JG, Endothelial cells removed", "4a",
     lambda f: ", ".join(str(_pick(f, "n_removed", cell_type=t, **TOP10))
                         for t in ("CNT/PC", "IC-B", "JG", "Endothelial")), "0, 0, 0, 0", None),
    ("control, top-10% cutoff: % of DCT1, DCT2 and Unknown removed (cluster annotation)", "4c",
     lambda f: ", ".join(f"{_pick(f, 'percent_of_label_removed', condition='Control', label_set=CA, label=t, **TOP10):.1f}"
                         for t in ("DCT1", "DCT2", "Unknown")), "12.3, 1.2, 9.5", None),
    ("zero-K, top-10% cutoff: % of DCT1 and Unknown removed (cluster annotation)", "4c",
     lambda f: ", ".join(f"{_pick(f, 'percent_of_label_removed', condition='Zero-K', label_set=CA, label=t, **TOP10):.1f}"
                         for t in ("DCT1", "Unknown")), "9.6, 11.1", None),
    ("top-10% cutoff: % of ProLIF removed, control and zero-K (marker-score label, clusters)", "4c",
     lambda f: ", ".join(f"{_pick(f, 'percent_of_label_removed', condition=c, label_set=SP, label='ProLIF', **TOP10):.1f}"
                         for c in ("Control", "Zero-K")), "27.7, 16.3", None),
    ("control, top-10% cutoff: permutation p (cluster annotation)", "4d",
     lambda f: _pick(f, "p_permutation", condition="Control", label_set=CA, **TOP10), 0.0084, 4),
    ("zero-K, top-10% cutoff: permutation p (cluster annotation)", "4d",
     lambda f: _pick(f, "p_permutation", condition="Zero-K", label_set=CA, **TOP10), 0.59, 2),
    ("kidney: rho of the intensity fraction with identified proteins in PCT, PST, TAL, DCT, Unknown, Endothelial",
     "5a", lambda f: ", ".join(f"{_pick(f, 'rho', cell_type=t, **BY_IDS):.2f}"
                               for t in ("PCT", "PST", "TAL", "DCT", "Unknown", "Endothelial")),
     "-0.15, -0.12, 0.02, -0.30, -0.30, 0.62", None),
    ("control: rho in DCT1, DCT2, Unknown (cluster annotation)", "5b",
     lambda f: ", ".join(f"{_pick(f, 'rho', condition='Control', label_set=CA, label=t, **BY_IDS):.2f}"
                         for t in ("DCT1", "DCT2", "Unknown")), "0.02, 0.12, -0.04", None),
    ("zero-K: rho in DCT1 with its q value (cluster annotation)", "5b",
     lambda f: (lambda d: f"{d['rho'].iloc[0]:.2f}, {d['p_fdr_bh_within_labels'].iloc[0]:.4f}")(
         rows(f, condition="Zero-K", label_set=CA, label="DCT1", **BY_IDS)), "-0.13, 0.0095", None),
    ("kidney: removed against retained wells, medians and p", "6a",
     lambda f: (lambda d: f"{d['median_group'].iloc[0]:.3f}, {d['median_reference'].iloc[0]:.3f}, "
                          f"{d['p_value'].iloc[0]:.5f}, {d['n_group'].iloc[0]} of {d['n_removed_cells'].iloc[0]}")(
         rows(f, metric=FRAC, comparison="removed_vs_retained_all")), "0.211, 0.344, 0.00066, 55 of 56", None),
    ("control and zero-K: removed against retained wells, medians", "6b",
     lambda f: "; ".join((lambda d: f"{d['median_group'].iloc[0]:.3f}, {d['median_reference'].iloc[0]:.3f}, "
                                    f"{d['n_group'].iloc[0]} of {d['n_removed_cells'].iloc[0]}")(
         rows(f, condition=c, metric=FRAC, comparison="removed_vs_retained_all")) for c in ("Control", "Zero-K")),
     "0.021, 0.490, 39 of 39; 0.000, 0.500, 63 of 65", None),
    ("cells high on the MitoCarta intensity fraction, and of these low-depth: kidney, control, zero-K", "6c",
     lambda f: "; ".join(f"{_pick(f, 'n_high', dataset=d, label_set=s, flag_source=FRAC)}, "
                         f"{_pick(f, 'n_high_and_depth_low', dataset=d, label_set=s, flag_source=FRAC)}"
                         for d, s in (("Total kidney", "cell types"), ("DCT Control", CA), ("DCT Zero-K", CA))),
     "6, 1; 8, 7; 11, 8", None),
    ("kidney: cells with a mitochondrial flag, of the evaluated cells", "6d",
     lambda f: f"{_pick(f, 'n_mito_any', cell_type='all cells')} of {_pick(f, 'n_cells_evaluated', cell_type='all cells')}",
     "20 of 704", None),
    ("control and zero-K: cells with a mitochondrial flag; high flags; high on the ID fraction; high and low-depth "
     "(cluster annotation)", "6f",
     lambda f: "; ".join(", ".join(str(_pick(f, col, condition=c, label_set=CA, label="all cells"))
                                   for col in ("n_mito_any", "n_mito_high", "n_mitocarta_id_fraction_high",
                                               "n_mito_high_and_depth_low")) for c in ("Control", "Zero-K")),
     "105, 81, 80, 72; 97, 71, 70, 48", None),
    ("ProLIF cells flagged within the marker-score label (clusters), control and zero-K", "6f",
     lambda f: ", ".join(f"{_pick(f, 'n_mito_any', condition=c, label_set=SP, label='ProLIF')} of "
                         f"{_pick(f, 'n_cells', condition=c, label_set=SP, label='ProLIF')}"
                         for c in ("Control", "Zero-K")), "1 of 119, 0 of 98", None),
    ("kidney: cells in the exclusion set", "7a", lambda f: _pick(f, "n_excluded", exclusion_set="mito_flagged"),
     20, None),
    ("kidney: outcomes inside the central 95% of the random removals", "7b", inside, "11 of 12", None),
    ("kidney: fraction of Unknown cells after exclusion, and the random range", "7b",
     lambda f: (lambda d: f"{d['observed'].iloc[0]:.3f}, {d['random_q025'].iloc[0]:.3f}, {d['random_q975'].iloc[0]:.3f}")(
         rows(f, exclusion_set="mito_flagged", outcome="unknown_fraction")), "0.048, 0.074, 0.155", None),
    ("kidney: random removals that recover the JG cluster", "7c",
     lambda f: (lambda d: f"{int(d['types'].map(lambda t: 'JG' in str(t).split(';')).sum())} of {len(d)}")(
         rows(f, exclusion_set="mito_flagged", run="random")), "31 of 100", None),
    ("control and zero-K: cells in the exclusion set", "7d",
     lambda f: ", ".join(str(_pick(f, "n_excluded", condition=c, exclusion_set="mito_flagged"))
                         for c in ("Control", "Zero-K")), "128, 101", None),
    ("ProLIF cells in the exclusion set, control and zero-K", "7e",
     lambda f: ", ".join(f"{_pick(f, 'n_excluded', condition=c, exclusion_set='mito_flagged', label_set=SP, label='ProLIF')}"
                         f" of {_pick(f, 'n_before', condition=c, exclusion_set='mito_flagged', label_set=SP, label='ProLIF')}"
                         for c in ("Control", "Zero-K")), "71 of 119, 17 of 98", None),
    ("control: outcomes inside the central 95% of the random removals", "7f",
     lambda f: inside(f, condition="Control"), "14 of 18", None),
    ("zero-K: outcomes inside the central 95% of the random removals", "7f",
     lambda f: inside(f, condition="Zero-K"), "15 of 16", None),
    ("control: agreement of the cluster annotation after reclustering, and the random range", "7f",
     lambda f: (lambda d: f"{d['observed'].iloc[0]:.3f}, {d['random_q025'].iloc[0]:.3f}, {d['random_q975'].iloc[0]:.3f}")(
         rows(f, condition="Control", exclusion_set="mito_flagged",
              outcome="cluster annotation agreement (Leiden 1.0 rerun)")), "0.661, 0.691, 0.964", None),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    a, b = stored["1a"], stored["1b"]
    counts = a["cell_type"].value_counts().to_dict()
    lib.check(checks, "table 1a: cells per cell type equal the counts of the revised annotation (712 cells)", counts,
              KIDNEY_TYPES, counts == KIDNEY_TYPES)
    got = {key: b.loc[b["condition"].eq(key[0]), key[1]].value_counts().to_dict() for key in DCT_COUNTS}
    lib.check(checks, "table 1b: cells per label equal the counts of the combined search, both label sets and "
              "conditions (729 Control, 702 Zero-K)", got, DCT_COUNTS, got == DCT_COUNTS)
    other = sorted({f"{key}: {c}" for key, f in stored.items() for c in map(str, f.columns)
                    if re.search(r"reannot|(^|_)pre(_|$)", c)})
    lib.check(checks, "no column of the submitted kidney annotation left", other or "none", "none", not other)
    types = sorted({str(v) for f in stored.values() if "cell_type" in f.columns for v in f["cell_type"].dropna()}
                   - set(KIDNEY_TYPES) - {"all cells"})
    lib.check(checks, "kidney tables: only the cell types of the revised annotation in the cell_type columns",
              types or "none", "none", not types)
    sets = sorted({str(v) for f in stored.values() if "label_set" in f.columns for v in f["label_set"].dropna()})
    conditions = sorted({str(v) for f in stored.values() if "condition" in f.columns for v in f["condition"].dropna()})
    lib.check(checks, "DCT tables: conditions Control and Zero-K; label sets by their reader-facing names",
              f"{conditions}; {sets}", f"['Control', 'Zero-K']; {sorted(['cell types', CA, SP, PLATE])}",
              conditions == ["Control", "Zero-K"] and sets == sorted(["cell types", CA, SP, PLATE]))
    union = {c: int((b.loc[b["condition"].eq(c), "flag__cluster_annotation__mito_any"].astype(bool)
                     | b.loc[b["condition"].eq(c), "flag__marker_score_clusters__mito_any"].astype(bool)).sum())
             for c in ("Control", "Zero-K")}
    sets_7d = {c: int(_pick(stored["7d"], "n_excluded", condition=c, exclusion_set="mito_flagged"))
               for c in ("Control", "Zero-K")}
    lib.check(checks, "tables 1b and 7d agree: cells with a mitochondrial flag in either label set = exclusion set",
              union, sets_7d, union == sets_7d)
    for key in ("2c", "4c", "5b", "6f"):
        f = stored[key]
        n = {(c, s): int(rows(f, condition=c, label_set=s, label="Unknown")["n_cells"].iloc[0])
             for c in ("Control", "Zero-K") for s in (CA, SP)}
        want = {(c, s): DCT_COUNTS[(c, "cluster_annotation" if s == CA else SP_COLUMN)]["Unknown"]
                for c, s in n}
        lib.check(checks, f"table {key}: the Unknown rows of the two label sets are kept apart (cells per row)", n, want,
                  n == want)


WORKBOOK = WorkbookDef(
    label="S-02-d", step=STEP, workbook="report_02_S02d_tables.xlsx",
    title="Supplementary Table S-02-d. Mitochondrial protein fraction as a cell-quality metric: per-cell metrics, "
          "flags and tests in the total-kidney and the DCT data",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names "
             "are explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"kidney": "revised annotation only (kidney_t15_label; display names DCT, CNT/PC)",
                "dct": "combined control + zero-K search (dct_reviewer tables); cluster annotation and marker-score label "
                       "(clusters)", "unknown_note": UNKNOWN_NOTE,
                "vocabulary": "final products only (authors, 12 Nov 2026): cs_<cond>_cluster_label = marker-score label "
                              "(clusters); cs_<cond>_cell_label = marker-score label (per cell)"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-02-d (mitochondrial metrics).")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
