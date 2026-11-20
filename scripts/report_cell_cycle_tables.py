#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S15 (ProLIF and the cell cycle in the DCT data), one workbook.

. One workbook with a Contents sheet and seven data sheets, built from 22 stored tables: 10 of
dct_reviewer (cell-cycle correction, pseudotime, depth and plate, ALDO scores), 3 of dct_prolif (per-cell proliferation test) and 9
of the response to the reviewers (the per-cell ProLIF cells in those results; ALDO scores against random protein sets). DCT data of the
combined control + zero-K search only. No new statistics: every value is copied from a stored table. Layout, style
and checks are those of the other supplementary tables (report_table_lib.py).

What the assembly changes (recorded per table in the sources table):
- conditions: Ctrl -> Control, ZeroK -> Zero-K; the conditions are listed one after the other and never compared;
- labels: the cluster annotation (cs_ctrl_deg_label / cs_zerok_pipeline_label) keeps its value Unknown, and every
  sheet that shows it carries the note "Unknown = DCT, subtype not resolved (cells of clusters without marker
  support)"; cs_<cond>_cluster_label is the "marker-score label (clusters)"; the dct_prolif route-A call of the no-floor
  run is the "per-cell test";
- matrices: uncorrected -> no correction; cc_ -> cell-cycle panel; tirosh_ -> S and G2/M lists; corr_ -> correlated
  proteins; scaled_X -> main matrix; cc_regressed_3a -> cell-cycle score regressed out;
- dct_reviewer tables that hold rows for both label columns: the rows of the marker-score label are kept where the result
  needs a ProLIF label (the cluster annotation has none), and both are kept in the neighbour table;
- columns that hold an internal column name are dropped after a check that the condition fixes them.

Workbook: results/tables/report/report_12_S12d_tables.xlsx (sheet Contents, then sheets 1-7; Arial 10;
values only, no formulas).

Inputs (read-only): the 22 tables named in SPECS.
Outputs: the workbook; results/tables/report/report_12_s12d_table_{sources,checks}.csv;
         results/tables/report/report_step_12_s12d_table_manifest.json

Command (project root):  python scripts/report_cell_cycle_tables.py --overwrite
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
import report_item_names as pn
import report_table_lib as lib
from report_table_lib import CONDITION, MARKER_SCORE, UNKNOWN_NOTE, Spec, WorkbookDef, _pick

STEP = "12_s12d_table"
REPORT = "results/tables/report/report_12_"
DCT_REVIEWER = "results/tables/dct_reviewer/dct_reviewer_"
DCT_PROLIF = "results/tables/dct_prolif/dct_prolif_"
CA, SP, TEST = "cluster annotation", "marker-score label (clusters)", "per-cell test"
UNKNOWN_LINE = (UNKNOWN_NOTE + ". The Unknown of the marker-score label is a label of its own.",)
NO_COLUMN = "internal column name; the condition (and the annotation column) gives it"

MATRIX = {
    "uncorrected": "no correction", "cc_removed": "cell-cycle panel removed",
    "cc_regressed": "cell-cycle panel score regressed out", "tirosh_removed": "S and G2/M lists removed",
    "tirosh_regressed": "S and G2/M scores regressed out", "corr_removed": "correlated proteins removed",
    "corr_regressed": "score of the correlated proteins regressed out",
    "depth_regressed": "ln(protein IDs) regressed out", "plate_regressed": "sort plate regressed out",
    "depth_cc_regressed": "ln(protein IDs) and cell-cycle panel score regressed out",
    "highcov50": "proteins detected in at least 50% of the cells only",
}
ROLE = {"primary": CA, "second": SP}
REFERENCE = {"cluster_annotation": CA, "marker_score_clusters": SP}
DEFINITION = {"per_cell_call": TEST, "marker_score_clusters": SP}
COMPARISON = {"ProLIF_vs_DCT1": "ProLIF minus DCT1", "ProLIF_vs_DCT2": "ProLIF minus DCT2",
              "ProLIF_vs_DCT1_DCT2": "ProLIF minus DCT1 and DCT2",
              "ProLIF_vs_all_other_cells": "ProLIF minus all other cells", "DCT2_vs_DCT1": "DCT2 minus DCT1"}
SOURCE = {"scaled_X": "main matrix", "cc_regressed_3a": "cell-cycle panel score regressed out"}
SCORE_NAME = {"cell_cycle_score": "cell-cycle panel score", "S_score": "S score", "G2M_score": "G2/M score",
              "cc_corr_score": "score of the correlated proteins"}
METRIC = {
    "scaled_X__ALDO_activity__base": "ALDO activity", "scaled_X__ALDO_activity__no_Slc12a3": "ALDO activity, no Slc12a3",
    "scaled_X__ALDO_signed__base": "ALDO signed", "scaled_X__ALDO_signed__no_Slc12a3": "ALDO signed, no Slc12a3",
    "scaled_X__ALDO_machinery__base": "ALDO machinery", "Slc12a3_ln1p_all_cells": "Slc12a3, ln(1 + intensity), all cells",
    "cell_cycle_score": "cell-cycle panel score",
}
METHOD = {"unmatched_all_cells": "all cells, not matched",
          "nn_matched_within_plate": "pairs matched on protein IDs within sort plate",
          "depth_decile_strata_within_plate": "strata of protein IDs (deciles) within sort plate"}
PSEUDOTIME_COLUMNS = ((r"^uncorrected__dpt_original_root$", "pseudotime__no_correction"),
                      (r"^cc_(removed|regressed)__dpt_original_root$", r"pseudotime__panel_\1"),
                      (r"^tirosh_(removed|regressed)__dpt_original_root$", r"pseudotime__s_g2m_\1"),
                      (r"^corr_(removed|regressed)__dpt_original_root$", r"pseudotime__correlated_\1"),
                      (r"^scaled_X__(.+)$", r"\1__main_matrix"), (r"^cc_regressed_3a__(.+)$", r"\1__score_regressed"))

SHEETS = {
    1: ("1 Corrections", "Cell-cycle corrections and the cell-cycle scores against protein depth", ()),
    2: ("2 Clusters after correction", "ProLIF cells over the Leiden clusters, and neighbours with the same label, "
                                       "before and after correction", UNKNOWN_LINE),
    3: ("3 Pseudotime", "Diffusion pseudotime of ProLIF, DCT1 and DCT2 cells before and after correction", UNKNOWN_LINE),
    4: ("4 Depth and sort plate", "ProLIF against the number of proteins per cell and the sort plate", ()),
    5: ("5 Per-cell test", "The per-cell proliferation test: protein set, groups and clusters", UNKNOWN_LINE),
    6: ("6 ALDO scores", "ALDO scores: coverage, contrasts, and contrasts against random protein sets", ()),
    7: ("7 Per cell", "Per-cell values", UNKNOWN_LINE),
}

SPECS = [
    Spec("1a", DCT_REVIEWER + "task4_correction_definitions.csv", "Matrices: what was removed or regressed out",
         ("One row per matrix and condition. proteins_removed: proteins taken out before scaling, PCA and the "
          "neighbour graph; regressed_scores: per-cell scores regressed out of every protein before scaling.",),
         values={"condition": CONDITION, "object": MATRIX,
                 "gene_set": {"none": "none", "original_18_panel": "cell-cycle panel",
                              "tirosh_S_G2M_plus_original": "S and G2/M lists and the cell-cycle panel",
                              "tirosh_S_G2M": "S and G2/M lists",
                              "Mki67_Top2a_correlated (circular)": "proteins correlated with Mki67 or Top2a"},
                 "correction": {"stored dct_labels graph": "none (graph of the main analysis)",
                                "proteins removed": "proteins removed",
                                "cell_cycle_score regressed": "cell-cycle panel score regressed out",
                                "S_score + G2M_score regressed": "S and G2/M scores regressed out",
                                "cc_corr_score regressed": "score of the correlated proteins regressed out"}},
         strict=("condition", "object", "gene_set", "correction"),
         rename={"object": "matrix", "gene_set": "protein_set", "regressed_keys": "regressed_scores"}),
    Spec("1b", DCT_REVIEWER + "task4_confounding_scores_vs_ids.csv", "Cell-cycle scores against the number of proteins per cell",
         ("Spearman correlation over the cells of a condition.",),
         values={"condition": CONDITION, "score": SCORE_NAME, "vs": {"n_detected_proteins": "protein IDs per cell"}},
         strict=("condition", "score", "vs"), rename={"vs": "against"}),
    Spec("2a", REPORT + "percell_prolif_cluster_spread.csv", f"ProLIF cells of the {TEST} over the Leiden clusters",
         ("One row per matrix and Leiden resolution. top1_, top3_: ProLIF cells in the one and the three clusters "
          "with most of them. max_prolif_fraction_of_a_cluster: the largest share of one cluster made up of ProLIF "
          "cells, with the size of that cluster.",),
         values={"condition": CONDITION, "object": MATRIX}, strict=("condition", "object"), rename={"object": "matrix"}),
    Spec("2b", DCT_REVIEWER + "task4_prolif_concentration.csv", f"ProLIF cells of the {SP} over the Leiden clusters",
         ("As table 2a, for the ProLIF cells of the marker-score label, cell-cycle matrices.",),
         rows=(("label_role", ("second",)),), drop_const=("label_role", "applicable"),
         drop_dependent=(("label_column", "condition"), ("resolution_numeric", "resolution")),
         values={"condition": CONDITION, "object": MATRIX}, strict=("condition", "object"), rename={"object": "matrix"}),
    Spec("2c", DCT_REVIEWER + "task4_knn_label_purity.csv", "Neighbours with the same label (k = 10, 20 principal components)",
         ("observed_same_label_fraction: mean fraction of a cell's 10 nearest neighbours that carry its label; null_ "
          "columns: 1,000 permutations of the labels (p floor 1/1,001).",),
         drop=("label_column", "embedding"), drop_reason=NO_COLUMN + "; embedding is fixed text",
         values={"condition": CONDITION, "object": MATRIX, "label_role": ROLE, "label": {"all_cells": "all cells"}},
         strict=("condition", "object", "label_role"), rename={"object": "matrix", "label_role": "annotation"},
         dct_unknown=True),
    Spec("3a", DCT_REVIEWER + "task4_dpt_summary.csv", f"Median pseudotime by the {SP}",
         ("Diffusion pseudotime from the root cell of the main analysis (root_cell_id) and from 20 random DCT1 root "
          "cells. prolif_between_dct1_and_dct2: the ProLIF median lies between the DCT1 and DCT2 medians.",),
         rows=(("label_role", ("second",)),), drop_const=("label_role", "prolif_between_applicable"),
         drop_dependent=(("label_column", "condition"),),
         values={"condition": CONDITION, "object": MATRIX}, strict=("condition", "object"),
         rename={"object": "matrix", "original_root_cell_id": "root_cell_id", "original_median_DCT1": "median_DCT1",
                 "original_median_DCT2": "median_DCT2", "original_median_ProLIF": "median_ProLIF",
                 "original_prolif_between": "prolif_between_dct1_and_dct2",
                 "n_random_prolif_between": "n_random_roots_prolif_between",
                 "fraction_random_prolif_between": "fraction_random_roots_prolif_between"}),
    Spec("3b", REPORT + "percell_prolif_pseudotime_by_group.csv", f"Pseudotime by group: ProLIF of the {TEST}, other "
         f"cells by the {CA}",
         ("Groups of Figure 3A. Pseudotime from the root cell of the main analysis.",),
         values={"condition": CONDITION, "object": MATRIX}, strict=("condition", "object"), rename={"object": "matrix"},
         dct_unknown=True),
    Spec("3c", REPORT + "percell_prolif_pseudotime_summary.csv", f"Pseudotime of the ProLIF cells of the {TEST} against "
         "DCT1 and DCT2",
         ("reference_annotation: the annotation that defines DCT1 and DCT2 (ProLIF cells taken out). Mann-Whitney U "
          "test; p_fdr_bh: Benjamini-Hochberg within condition and reference annotation.",),
         values={"condition": CONDITION, "object": MATRIX, "reference_annotation": REFERENCE, "comparison": COMPARISON},
         strict=("condition", "object", "reference_annotation", "comparison"), rename={"object": "matrix"}),
    Spec("3d", REPORT + "percell_prolif_pseudotime_same_cluster_reference.csv", f"Median pseudotime of the ProLIF cells "
         f"of the {TEST} against random cells of the same clusters",
         ("reference_ columns: medians of 1,000 draws of as many cells that are not ProLIF, taken from the same "
          "Leiden clusters (resolution 1). reference_fraction_between: draws whose median lies between the DCT1 "
          "and DCT2 medians of the named annotation.",),
         values={"condition": CONDITION, "object": MATRIX}, strict=("condition", "object"), rename={"object": "matrix"}),
    Spec("4a", DCT_REVIEWER + "task4_confounding_prolif_vs_ids_and_plate.csv", f"ProLIF of the {SP}: proteins per cell and "
         "sort plate",
         ("ids_ columns: protein IDs per cell, ProLIF against all other cells (Mann-Whitney U). P1_, P2_: sort "
          "plates; Fisher exact test of ProLIF by plate.",),
         rows=(("label_role", ("second",)),), drop_const=("label_role", "status"),
         drop_dependent=(("label_column", "condition"),), values={"condition": CONDITION}, strict=("condition",)),
    Spec("4b", DCT_REVIEWER + "depth_prolif_logistic_models.csv", f"Logistic models of the ProLIF label of the {SP}",
         ("Odds ratios per standard deviation of the z-scored terms; or_per_unscaled_unit: per unit of the original "
          "variable.",),
         rows=(("label_role", ("second",)),), drop_const=("label_role", "status"),
         drop_dependent=(("label_column", "condition"),),
         values={"condition": CONDITION,
                 "model": {"ProLIF ~ ln_ids + plate": "ProLIF ~ ln(protein IDs) + sort plate",
                           "ProLIF ~ ln_ids + plate + cell_cycle_score":
                               "ProLIF ~ ln(protein IDs) + sort plate + cell-cycle panel score"},
                 "term": {"z_ln_ids": "ln(protein IDs)", "plate_P2": "sort plate P2",
                          "z_cell_cycle_score": "cell-cycle panel score"}},
         strict=("condition", "model", "term")),
    Spec("4c", DCT_REVIEWER + "depth_matched_comparisons.csv", f"ProLIF of the {SP} against DCT1 and DCT2 cells of the same "
         "depth",
         ("method: all cells; pairs matched on ln(protein IDs) within sort plate (caliper 0.05); strata of protein "
          "IDs within sort plate. difference: ProLIF minus comparator. p_fdr_bh: Benjamini-Hochberg within "
          "condition and method.",),
         rows=(("label_role", ("second",)),), drop_const=("label_role", "status"),
         drop_dependent=(("label_column", "condition"),),
         values={"condition": CONDITION, "metric": METRIC, "method": METHOD}, strict=("condition", "method")),
    Spec("4d", DCT_PROLIF + "acceptance_tests.csv", f"ProLIF of the {TEST}: depth, sort plate, held-out proteins, "
         "specificity and stability",
         ("Five tests with criteria fixed before the test was run. Test 1 and 2: logistic models of ProLIF on "
          "ln(protein IDs) and sort plate. Test 3: Mki67 or Top2a, which the test does not use. Test 4: 100 random "
          "protein sets in place of the proliferation set. Test 5: 10 other random seeds.",),
         rows=(("run", ("nofloor",)),), drop_const=("run", "computable"), values={"condition": CONDITION},
         strict=("condition",), rename={"threshold": "criterion", "pass": "criterion_met"}),
    Spec("5a", DCT_PROLIF + "prolif_set.csv", "S and G2/M proteins and the proliferation set of the per-cell test",
         ("in_proliferation_set: the 36 proteins of the test (quantified in both conditions); held_out: Mki67 and "
          "Top2a, not used by the test. n_detected_: cells with the protein detected; decile_: decile of the "
          "detection rate among all proteins.",),
         drop_re=r"_frozen$|_all$", drop_reason="values of a run that is not shown; the _all columns equal the columns kept",
         values={"phase": {"S": "S", "G2M": "G2/M"}}, strict=("phase",),
         rename_re=((r"^tirosh_", ""), (r"_ctrl", "_control"), (r"_zerok", "_zero_k"), (r"_nofloor$", ""),
                    (r"^in_P$", "in_proliferation_set"))),
    Spec("5b", REPORT + "percell_prolif_groups.csv", "Groups of Figure 3A: cells, depth, sort plate and proliferation",
         ("n_cycling_score_rule: cells above the 90th percentile of the cell-cycle panel score within the condition. "
          "n_marker_score_ columns: the cells of the group by their marker-score label (clusters).",),
         values={"condition": CONDITION}, strict=("condition",), dct_unknown=True),
    Spec("5c", DCT_PROLIF + "prolif_overlay_by_t27_cluster.csv", f"ProLIF cells of the {TEST} per Leiden cluster of the {CA}",
         ("One row per cluster (resolution 1). n_cycling_ncc_not_detected: cells called cycling in which Slc12a3 "
          "was not detected (not labelled ProLIF).",),
         rows=(("run", ("nofloor",)),), drop_const=("run", "floor", "n_below_floor"),
         drop_dependent=(("t27_cluster_column", "condition"),), values={"condition": CONDITION}, strict=("condition",),
         rename={"t27_cluster": "cluster", "t27_label": "cluster_annotation", "median_n_detected": "median_protein_ids"},
         dct_unknown=True),
    Spec("6a", DCT_REVIEWER + "aldo_marker_coverage.csv", "ALDO lists: proteins listed and quantified",
         ("variant no_Slc12a3: the list without Slc12a3. direction negative: the down list of the signed score.",),
         rows=(("variant", ("base", "no_Slc12a3")),), values={"condition": CONDITION}, strict=("condition",)),
    Spec("6b", DCT_REVIEWER + "aldo_contrasts.csv", f"ALDO scores by label: {CA} and {SP}",
         ("All cells. Rows exist only where both groups exist (the cluster annotation has no ProLIF cluster and, in "
          "Zero-K, no DCT2 cluster). Mann-Whitney U; p_fdr_bh: Benjamini-Hochberg within condition, annotation and "
          "matrix.",),
         rows=(("cohort", ("all_cells",)), ("status", ("tested",))), drop_const=("cohort", "status"),
         drop=("label_column", "score"), drop_reason=NO_COLUMN + "; score = matrix, family and variant",
         values={"condition": CONDITION, "label_role": ROLE, "comparison": COMPARISON, "source": SOURCE},
         strict=("condition", "label_role", "comparison", "source"),
         rename={"label_role": "annotation", "source": "matrix", "family": "score"}),
    Spec("6c", REPORT + "percell_prolif_aldo_contrasts.csv", f"ALDO scores of the ProLIF cells of the {TEST}",
         ("Mann-Whitney U and effect sizes; adj_ columns: coefficient of ProLIF in a linear model score ~ ProLIF + "
          "ln(protein IDs) + sort plate over the ProLIF and the reference cells. p_fdr_bh, adj_p_fdr_bh: "
          "Benjamini-Hochberg within condition, reference annotation and matrix.",),
         values={"condition": CONDITION, "reference_annotation": REFERENCE, "comparison": COMPARISON, "source": SOURCE},
         strict=("condition", "reference_annotation", "comparison", "source"), rename={"source": "matrix"}),
    Spec("6d", REPORT + "aldo_random_sets.csv", "ALDO contrasts against 1,000 random protein sets",
         ("_observed: the ALDO list; _random_ columns: random sets of the same size and detection rate (median and "
          "central 95%); _p_random_at_or_above / _at_or_below: fraction of random sets at or beyond the observed "
          "value (floor 1/1,001). score 'all proteins of the matrix': the mean z-score over all proteins, no random "
          "sets.",),
         values={"condition": CONDITION, "source": SOURCE, "prolif_definition": DEFINITION, "comparison": COMPARISON,
                 "score": {"all_proteins": "all proteins of the matrix"}},
         strict=("condition", "source", "prolif_definition", "comparison"), rename={"source": "matrix"}),
    Spec("6e", REPORT + "aldo_random_sets_proteins.csv", "ALDO list proteins: detection rate and bin for the random sets",
         ("detection_bin: one of 20 bins of the fraction of cells with the protein detected; "
          "proteins_in_bin_available: proteins of that bin that are in no ALDO list.",),
         values={"condition": CONDITION,
                 "list": {"UP_base": "up list (with Slc12a3)", "UP_no_Slc12a3": "up list without Slc12a3",
                          "DOWN": "down list", "machinery": "machinery list"}},
         strict=("condition", "list")),
    Spec("7a", REPORT + "percell_prolif_per_cell.csv", "Per-cell values",
         ("One row per retained cell (729 Control, 702 Zero-K). proliferation_ and prolif_ columns: the per-cell "
          "test; pseudotime__ columns: by matrix; ALDO columns: by matrix (main matrix; cell-cycle panel score "
          "regressed out).",),
         values={"condition": CONDITION}, strict=("condition",),
         rename={"n_detected": "protein_ids", "dct_label": "cluster_annotation",
                 "second_label": "marker_score_label_clusters", "group": "figure_3a_group",
                 "is_prolif": "prolif_per_cell_test", "reviewer_cycling_status": "cycling_by_score_rule",
                 "n_P_detected": "n_proliferation_set_proteins_detected", "prolif_D": "proliferation_D",
                 "axis": "dct2_minus_dct1_marker_axis"},
         rename_re=PSEUDOTIME_COLUMNS, dct_unknown=True),
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
     "are not used. These are the ProLIF cells of Figure 3A."),
    ("matrix", "The data after one correction: no correction (the main analysis); a protein set removed; or a per-cell "
     "score regressed out of every protein. Each matrix was scaled, reduced by PCA (20 components), clustered "
     "(Leiden) and given a diffusion pseudotime as in the main analysis."),
    ("cell-cycle panel; S and G2/M lists; correlated proteins", "Three protein sets: an 18-protein cell-cycle panel "
     "(12 quantified); published S-phase and G2/M-phase lists; the proteins whose detection correlates with Mki67 or "
     "Top2a (Spearman rho >= 0.3, q < 0.05). The scores are computed from the scaled matrix."),
    ("pseudotime", "Diffusion pseudotime with a DCT1 cell as root."),
    ("ALDO scores; variant", "ALDO activity: mean z-score of the aldosterone-induced list; ALDO signed: induced list "
     "minus repressed list; ALDO machinery: mean z-score of the signalling-component list. variant base: Slc12a3 is "
     "part of the induced list; no_Slc12a3: without it; no_cell_cycle: without cell-cycle panel proteins."),
    ("protein IDs", "Number of proteins identified (intensity above zero) in a cell."),
    ("cohens_d; rank_biserial", "Effect sizes of a two-group comparison (group minus reference)."),
]

LETTER = [
    ("control: cell-cycle panel score against protein IDs, Spearman rho", "1b",
     lambda f: _pick(f, "spearman_rho", condition="Control", score="cell-cycle panel score"), -0.64, 2),
    ("zero-K: cell-cycle panel score against protein IDs, Spearman rho", "1b",
     lambda f: _pick(f, "spearman_rho", condition="Zero-K", score="cell-cycle panel score"), -0.46, 2),
    ("control: per-cell ProLIF, largest share of a cluster, no correction, resolution 1", "2a",
     lambda f: _pick(f, "max_prolif_fraction_of_a_cluster", condition="Control", matrix="no correction", resolution=1),
     0.18, 2),
    ("control: per-cell ProLIF, clusters with ProLIF cells of all clusters, no correction, resolution 1", "2a",
     lambda f: f"{_pick(f, 'n_clusters_with_prolif', condition='Control', matrix='no correction', resolution=1)} of "
               f"{_pick(f, 'n_clusters', condition='Control', matrix='no correction', resolution=1)}", "8 of 13", None),
    ("control: per-cell ProLIF cells in the one pure cluster at resolution 3, no correction", "2a",
     lambda f: _pick(f, "cluster_of_max_fraction_n_prolif", condition="Control", matrix="no correction", resolution=3),
     10, None),
    ("control: marker-score ProLIF, same-label neighbour fraction, no correction", "2c",
     lambda f: _pick(f, "observed_same_label_fraction", condition="Control", matrix="no correction", annotation=SP,
                     label="ProLIF"), 0.88, 2),
    ("control: marker-score ProLIF, median pseudotime, no correction", "3a",
     lambda f: _pick(f, "median_ProLIF", condition="Control", matrix="no correction"), 0.82, 2),
    ("zero-K: marker-score ProLIF, median pseudotime, no correction", "3a",
     lambda f: _pick(f, "median_ProLIF", condition="Zero-K", matrix="no correction"), 0.67, 2),
    ("control: per-cell ProLIF, median pseudotime, no correction", "3d",
     lambda f: _pick(f, "median_ProLIF", condition="Control", matrix="no correction"), 0.39, 2),
    ("control: random cells of the same clusters with a median between DCT1 and DCT2, no correction", "3d",
     lambda f: _pick(f, "reference_fraction_between__cluster_annotation", condition="Control", matrix="no correction"),
     0.77, 2),
    ("control: marker-score ProLIF, median protein IDs", "4a",
     lambda f: _pick(f, "ids_median_group", condition="Control"), 516, 0),
    ("control: other cells, median protein IDs", "4a",
     lambda f: _pick(f, "ids_median_reference", condition="Control"), 1916, 0),
    ("control: per-cell ProLIF cells; with Mki67 or Top2a detected", "5b",
     lambda f: f"{_pick(f, 'n_cells', condition='Control', group='ProLIF')}; "
               f"{_pick(f, 'n_mki67_or_top2a_detected', condition='Control', group='ProLIF')}", "37; 14", None),
    ("zero-K: per-cell ProLIF cells; with Mki67 or Top2a detected", "5b",
     lambda f: f"{_pick(f, 'n_cells', condition='Zero-K', group='ProLIF')}; "
               f"{_pick(f, 'n_mki67_or_top2a_detected', condition='Zero-K', group='ProLIF')}", "63; 37", None),
    ("control: per-cell ProLIF minus DCT1, ALDO activity after regression, Cohen's d", "6d",
     lambda f: _pick(f, "cohens_d_observed", condition="Control", matrix=SOURCE["cc_regressed_3a"],
                     prolif_definition=TEST, comparison="ProLIF minus DCT1", score="ALDO_activity", variant="base"),
     1.18, 2),
    ("control: the same contrast for all proteins of the matrix, Cohen's d", "6d",
     lambda f: _pick(f, "cohens_d_observed", condition="Control", matrix=SOURCE["cc_regressed_3a"],
                     prolif_definition=TEST, comparison="ProLIF minus DCT1", score="all proteins of the matrix"),
     1.83, 2),
    ("control: per-cell ProLIF minus DCT1, ALDO activity, main matrix, Cohen's d", "6d",
     lambda f: _pick(f, "cohens_d_observed", condition="Control", matrix=SOURCE["scaled_X"],
                     prolif_definition=TEST, comparison="ProLIF minus DCT1", score="ALDO_activity", variant="base"),
     0.08, 2),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    cells = stored["7a"]
    group = pn.paper_names("figure_3a_group")     # the column as the workbook names it (panel of the paper)
    got = {c: cells.loc[cells["condition"].eq(c), group].value_counts().to_dict()
           for c in ("Control", "Zero-K")}
    expected = {"Control": {"DCT1": 362, "DCT2": 79, "ProLIF": 37, "Unknown": 251},
                "Zero-K": {"DCT1": 402, "ProLIF": 63, "Unknown": 237}}
    lib.check(checks, "table 7a: cells per group of Figure 3A (729 Control, 702 Zero-K)", got, expected, got == expected)


WORKBOOK = WorkbookDef(
    label="S-12-d", step=STEP, workbook="report_12_S12d_tables.xlsx",
    title="Supplementary Table S-12-d. ProLIF and the cell cycle in the DCT data: cell-cycle corrections, clusters, "
          "pseudotime, depth, the per-cell proliferation test and ALDO scores",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names are "
             "explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"dct": "combined control + zero-K search (dct_labels labels, dct_reviewer and dct_prolif tables, the response to the reviewers tables)",
                "unknown_note": UNKNOWN_NOTE,
                "vocabulary": "cs_<cond>_cluster_label = marker-score label (clusters); dct_prolif route A, no-floor run = "
                              "per-cell test; scaled_X = main matrix"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-12-d.")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
