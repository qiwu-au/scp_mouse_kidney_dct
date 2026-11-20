#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S18 (zero-K against control in DCT cells), one workbook.

(plan approved by the authors on 12 Nov 2026: "Approve as
planned"). One workbook with a Contents sheet and eight data sheets (31 tables), built from 29 stored tables of
zerok_vs_control (zero-K against control on the combined search), one of dct_intake (panel coverage in the combined search) and the
frozen zerok_core panel (expected directions and sources). No new statistics: every value is copied from a stored table. The
assembly and its checks are those of report_table_lib.py (layout of Supplementary Table S7).

What the assembly changes (recorded per table in the sources table):
- conditions: Ctrl -> Control, ZeroK -> Zero-K, in values and column names;
- labels: dct_label -> "cluster annotation"; dct_label_alternative -> "transferred zero-K label"; label_free ->
  "no labels"; all_DCT -> "DCT1 + DCT2"; the label value Unknown stays, with the note "Unknown = DCT, subtype not
  resolved" on every sheet;
- normalization variants: none -> "ln(1 + intensity)"; cell_median -> "per-cell median centred"; condition_align
  -> "zero-K median aligned to control";
- rows of label groups without cells in a condition ("does not apply") are left out, and the sheet note says so;
- columns with internal names or working notes are dropped (listed per table);
- the frozen panel keeps its items, expected directions and sources; its coverage columns come from the combined
  search (dct_intake), not from the panel file.
Left out of the workbook (stored, named in the response file): the per-protein results of the other groups, label
routes and normalization variants; the projected pseudotime; the DCT-enriched gene list of the RNA atlas.

Workbook: results/tables/report/report_19_S19e_tables.xlsx
Outputs: the workbook; results/tables/report/report_19_s19e_table_{sources,checks}.csv;
         results/tables/report/report_step_19_s19e_table_manifest.json

Command (project root):  python scripts/report_zerok_tables.py --overwrite
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
from report_table_lib import CONDITION, UNKNOWN_NOTE, Spec, WorkbookDef, _pick

STEP = "19_s19e_table"
ZEROK_VS_CONTROL = "results/tables/zerok_vs_control/zerok_vs_control_"
DCT_INTAKE = "results/tables/dct_intake/dct_intake_"
ZEROK_CORE = "results/tables/zerok_core/zerok_core_"

CLUSTER, TRANSFER, NOLABEL = "cluster annotation", "transferred zero-K label", "no labels"
LC = {"dct_label": CLUSTER, "dct_label_alternative": TRANSFER, "label_free": NOLABEL}
BOTH, ALLCELLS, NCC = "DCT1 + DCT2", "all cells", "cells with NCC (Slc12a3) detected"
GR = {"all_DCT": BOTH, "all_cells": ALLCELLS, "Slc12a3_detected": NCC}
LN, CELLMED, ALIGNED = "ln(1 + intensity)", "per-cell median centred", "zero-K median aligned to control"
VAR = {"none": LN, "cell_median": CELLMED, "condition_align": ALIGNED}
DEPTH, HALF, QUARTER = "depth-matched", "50% of cells", "25% of cells"
PROC = {"depth_matched": DEPTH, "downsample_50pct": HALF, "downsample_25pct": QUARTER}
C_P, Z_P, ZC1, ZC2, ZCB = ("Control, P2 vs P1", "Zero-K, P2 vs P1", "Zero-K vs control, P1 only",
                           "Zero-K vs control, P2 only", "Zero-K vs control, both plates")
CONTRAST = {"Ctrl P2 vs P1": C_P, "ZeroK P2 vs P1": Z_P, "ZeroK vs Ctrl, P1 only": ZC1, "ZeroK vs Ctrl, P2 only": ZC2,
            "ZeroK vs Ctrl (pooled)": ZCB, "Ctrl P2 vs P1 (technical)": C_P, "ZeroK P2 vs P1 (technical)": Z_P}
MEASURE = {"Mki67_detected": "Mki67 detected", "Top2a_detected": "Top2a detected",
           "Mki67_or_Top2a_detected": "Mki67 or Top2a detected",
           "cycling_score_rule": "cell-cycle score above the 90th percentile of control cells",
           "cycling_score_rule__none": f"cell-cycle score above the 90th percentile of control cells; {LN}",
           "cycling_score_rule__cell_median": f"cell-cycle score above the 90th percentile of control cells; {CELLMED}",
           "cycling_score_rule__condition_align": f"cell-cycle score above the 90th percentile of control cells; {ALIGNED}"}
RN = ((r"^label_column$", "labels"), (r"zerok", "zero_k"), (r"ctrl", "control"))
V3 = {"label_column": LC, "group": GR, "variant": VAR}
S3 = ("label_column", "variant")
DESIGN = ("One pool of 3 mice per condition: condition and pool cannot be separated, cells are not animal-level "
          "replicates, and every p and q value is descriptive. P1 and P2 are sort plates of the pooled cells "
          "(technical).")
LABELS = (f"labels: {CLUSTER} = cluster labels of each condition (zero-K has DCT1 and Unknown only); {TRANSFER} = "
          "each zero-K cell takes the most frequent cluster annotation of its 10 nearest control cells; "
          f"{NOLABEL} = no label is used. {BOTH}: cells labelled DCT1 or DCT2.")
ABSENT = "Groups without cells in one of the conditions are not listed."
NOTES = (DESIGN, LABELS, UNKNOWN_NOTE, ABSENT)

SHEETS = {
    1: ("1 Design and panel", "Cells per condition and sort plate, and the pre-specified protein panel", (DESIGN,)),
    2: ("2 Panel results", "Pre-specified panel: zero-K against control in every cell group", NOTES),
    3: ("3 Label fractions", "Label fractions per condition, and proteins of other segments and cell types", NOTES),
    4: ("4 Proteome-wide", "All tested proteins: summary and per-protein results", NOTES),
    5: ("5 Cell state", "DCT1 - DCT2 marker score and cycling proteins", NOTES),
    6: ("6 Robustness", "Depth-matched and downsampled repeats, sort plates, scores without Slc12a3", NOTES),
    7: ("7 Specificity", "Matched random protein sets and the sort-plate null", NOTES),
    8: ("8 Bulk kidney proteome", "Comparison with a bulk whole-kidney proteome on a 0% or 1% potassium diet", NOTES),
}

SPECS = [
    Spec("1a", ZEROK_VS_CONTROL + "design_cells.csv", "Cells sorted, retained and removed per condition and sort plate",
         ("Cells with fewer than 100 detected proteins were removed. sort_plate all: both plates of the condition.",),
         values={"condition": CONDITION}, strict=("condition",), rename={"plate": "sort_plate"}),
    Spec("1b", ZEROK_CORE + "prespecified_panel.csv",
         "Pre-specified panel: items, expected direction under the zero-K diet, and sources",
         ("The panel and its expected directions were fixed before any comparison between the conditions was "
          "tested. Expected directions come from published low-potassium studies of whole kidney or cortex.",
          "item_type limit: named in advance as not measurable (phosphorylation: no phospho-enrichment; Kcnj1 and "
          "Scnn1a, Scnn1b, Scnn1g: not detected)."),
         drop_re=r"^(direction_set_by|needs_user_confirmation|measurable_here|not_measurable_reason|low_coverage|"
                 r"ctrl_.*|zerok_.*|in_joint|match_level|n_proteins_in_list|n_in_joint|absent_both|present_in_joint|"
                 r"note|genes_checked)$",
         drop_reason="coverage columns of the panel file refer to an earlier export (table 1c gives the coverage in "
                     "the search used here); working notes",
         value_re={"sources_measured": ((r"This project's own group \(source accepted by the authors, 28 Oct 2026\)",
                                         "Work of the authors' group"),
                                        (r"This project's own group", "Work of the authors' group"))}),
    Spec("1c", DCT_INTAKE + "panel_coverage.csv", "Pre-specified panel: coverage in the combined control and zero-K search",
         ("n_cells_detected and fraction_cells: retained cells of the condition with the protein detected.",
          "Scores: n_members = proteins of the score list; members retained in both conditions enter the score."),
         drop=("panel_group", "expected_direction", "ctrl_var_name", "zerok_var_name", "ctrl_n_groups",
               "zerok_n_groups", "genes_checked"),
         drop_dependent=(("zerok_protein_group", "ctrl_protein_group"),),
         drop_reason="panel_group and expected_direction are in table 1b; the other columns repeat the item name",
         rename={"ctrl_protein_group": "protein_group", "absent_both": "members_not_quantified",
                 "present_in_joint": "members_retained_in_both_conditions", "in_joint": "retained_in_both_conditions",
                 "n_in_joint": "n_members_retained_in_both_conditions", "n_proteins_in_list": "n_members"},
         rename_re=RN),
    Spec("2a", ZEROK_VS_CONTROL + "panel_results.csv", "Panel results per label route, cell group and normalization variant",
         ("Two-part test per protein (detection: Firth logistic regression; intensity among detected cells: least "
          "squares), both with log proteins per cell as covariate; scores: least squares with the same covariate.",
          "q_primary: Benjamini-Hochberg within the panel, per label route, group and variant. An observed "
          "direction is assigned when q_primary < 0.05."),
         rows=(("status", ("tested",)),), drop_const=("status",), drop=("joint_var_name", "score_column", "p_value_note"),
         drop_reason="internal names; the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("2b", ZEROK_VS_CONTROL + "robustness_label_free_panel.csv", "Panel results without labels",
         (f"{ALLCELLS}: all retained cells (729 control, 702 zero-K). {NCC}: cells in which NCC was detected.",),
         drop=("joint_var_name", "score_column", "p_value_note"),
         drop_reason="internal names; the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN),
    Spec("2c", ZEROK_VS_CONTROL + "panel_match_summary.csv", "Scored items per group: observed against expected direction",
         ("Eight items have an expected direction (up, down or no change) and are scored; 20 items have none.",
          "does not apply: items of a group without cells in one of the conditions."),
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("2d", ZEROK_VS_CONTROL + "panel_pseudobulk.csv", "Panel proteins: mean intensity per cell and its log2 ratio, per sort plate",
         ("mean_intensity: mean back-transformed intensity per cell, non-detection counted as 0. plate_scope pooled: "
          "both sort plates.",),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("3a", ZEROK_VS_CONTROL + "label_fractions.csv", "Cells per label, per condition and sort plate",
         ("Wilson 95% intervals.",), values={"label_column": LC, "condition": CONDITION},
         strict=("label_column", "condition"), rename={"plate": "sort_plate"}, rename_re=RN, dct_unknown=True),
    Spec("3b", ZEROK_VS_CONTROL + "label_fraction_differences.csv", "Difference in label fraction, zero-K minus control",
         ("Newcombe 95% intervals. The zero-K DCT2 fraction depends on the label route (0 of 702 cells under the "
          f"{CLUSTER}, 68 under the {TRANSFER}).",),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values={"label_column": LC}, strict=("label_column",),
         value_re={"note": ((r" under the frozen dct_labels rule", ""), (r"ZeroK", "zero-K"), (r"Ctrl", "control"))},
         rename={"plate": "sort_plate"}, rename_re=RN, dct_unknown=True),
    Spec("3c", ZEROK_VS_CONTROL + "contamination_detection.csv",
         "Proteins of other tubule segments and cell types: cells with the protein detected, per condition",
         ("Detection only (intensity above zero). Not compared because not retained in both conditions: Scnn1a, "
          "Scnn1b, Scnn1g, Emcn, Ptprc, Aqp3, Col5a2, Dcn.",),
         rows=(("status", ("compared",)),), drop_const=("status",), drop=("source", "joint_var_name", "p_value_note"),
         drop_reason="internal names; the note repeats the design statement of the sheet",
         values={"label_column": LC, "group": GR,
                 "panel_group": {"Task 3 CNT_CD": "CNT and collecting duct (second panel)",
                                 "Task 3 Interstitial": "Interstitial"}},
         strict=("label_column",), rename_re=RN, dct_unknown=True),
    Spec("4a", ZEROK_VS_CONTROL + "proteome_summary.csv", "Tested proteins and proteins with q < 0.05 per group and variant",
         ("Proteins detected in at least 10% of the cells of a condition were tested (two-part test as in table 2a; "
          "Benjamini-Hochberg within the group).",
          "The same test between the two sort plates of one condition gives a similar share (table 7d), so these "
          "counts are not read as the size of a diet response."),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("4b", ZEROK_VS_CONTROL + "proteome_primary_none.csv",
         f"Per-protein results, {BOTH} of the {CLUSTER} (467 control, 449 zero-K cells), {LN}",
         ("One row per tested protein. q_two_part: Benjamini-Hochberg over the tested proteins of the group.",),
         rows=(("group", ("all_DCT",)),),
         drop_const=("label_column", "group", "variant", "match_level", "p_value_note", "n_ctrl", "n_zerok"),
         drop=("det_lr", "det_converged", "int_lr", "int_coef_ln", "median_difference_ln", "testable"),
         drop_reason="intermediate statistics of the two-part test",
         rename_re=RN,
         assembly_note="The DCT1 and Unknown groups, the transferred zero-K label and the two other normalization "
                       "variants are stored and not in the workbook."),
    Spec("4c", ZEROK_VS_CONTROL + "one_condition_only_proteins.csv", "Protein groups retained in one condition only (not tested)",
         ("Retained = detected in at least two retained cells of the condition. Presence in one condition only is "
          "not read as a diet effect.",),
         values={"retention": {"Ctrl only": "Control only", "ZeroK only": "Zero-K only"}}, strict=("retention",),
         drop=("var_name", "file_row"), drop_reason="internal name and row number of the export", rename_re=RN),
    Spec("5a", ZEROK_VS_CONTROL + "state_score_shift.csv", "DCT1 - DCT2 marker score, zero-K against control",
         ("Score difference per cell (DCT1 marker score minus DCT2 marker score; meanz: mean z-score version; "
          "no_Slc12a3: without Slc12a3). score_coef: least squares with log proteins per cell as covariate.",),
         rows=(("status", ("compared",)),), drop_const=("status",), drop=("p_value_note",),
         drop_reason="the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("5b", ZEROK_VS_CONTROL + "state_cycling.csv", "Cells with cycling proteins, zero-K against control",
         ("Wilson 95% intervals per condition; Newcombe 95% interval of the difference.",),
         rows=(("status", ("compared",)),), drop_const=("status",), drop=("p_value_note",),
         drop_reason="the note repeats the design statement of the sheet",
         values={"label_column": LC, "group": GR, "measure": MEASURE}, strict=("label_column", "measure"),
         rename_re=RN, dct_unknown=True),
    Spec("5c", ZEROK_VS_CONTROL + "state_cycling_thresholds.csv", "Cell-cycle score threshold (90th percentile of control cells)",
         ("18-protein cell-cycle list, 12 proteins quantified.",), values={"variant": VAR}, strict=("variant",),
         rename_re=RN),
    Spec("5d", ZEROK_VS_CONTROL + "state_depth_matched.csv", "Cell-state measures after matching cells on protein depth",
         ("100 repeats of 1:1 matching on ln proteins per cell within sort plate (caliper 0.05).",),
         rows=(("status", ("compared",)), ("analysis", ("score_shift", "cycling"))), drop_const=("status",),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values={"label_column": LC, "group": GR, "variant": VAR, "measure": MEASURE,
                 "analysis": {"score_shift": "marker score", "cycling": "cycling proteins"},
                 "effect": {"median difference ZeroK - Ctrl": "median difference, zero-K minus control",
                            "fraction difference ZeroK - Ctrl": "difference in fraction, zero-K minus control"}},
         strict=("label_column", "analysis", "effect"), rename_re=RN, dct_unknown=True),
    Spec("6a", ZEROK_VS_CONTROL + "robustness_panel_repeats.csv", "Panel items in 100 depth-matched or downsampled repeats",
         ("fraction_same_match_status: share of the repeats with the same result (match or mismatch) as all cells "
          "of the group. Downsampling keeps 50% or 25% of the cells of each condition.",),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values={**V3, "procedure": PROC}, strict=(*S3, "procedure"), rename_re=RN, dct_unknown=True),
    Spec("6b", ZEROK_VS_CONTROL + "robustness_plate_split_panel.csv", "Panel items estimated from each sort plate separately",
         ("P1 and P2 are sort plates of one pool per condition: a technical check, not replication.",),
         drop_const=("note",), values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("6c", ZEROK_VS_CONTROL + "robustness_plate_split_summary.csv", "Agreement of the effects of the two sort plates",
         ("Spearman correlation and sign agreement of the P1-only and P2-only effects.",),
         drop_const=("note",),
         values={**V3, "effect": {"detection_difference_zerok_minus_ctrl": "detection_difference_zero_k_minus_control"}},
         strict=S3, rename_re=RN, dct_unknown=True),
    Spec("6d", ZEROK_VS_CONTROL + "robustness_proteome_effects.csv", "Proteome-wide effects in the repeats against all cells",
         ("Effect sizes only; no test per repeat.",), drop_const=("note",),
         values={**V3, "procedure": PROC}, strict=(*S3, "procedure"), rename_re=RN, dct_unknown=True),
    Spec("6e", ZEROK_VS_CONTROL + "robustness_leave_slc12a3_out.csv", "Score differences with and without Slc12a3",
         ("ALDO: aldosterone-related scores.",), drop=("source", "p_value_note"),
         drop_reason="internal step names; the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("7a", ZEROK_VS_CONTROL + "specificity_matched_null.csv", "Directional panel proteins against 10,000 matched random sets",
         ("Each of the 7 proteins with an expected increase or decrease is replaced by a random protein outside the "
          "panel, drawn from its 50 nearest proteins in detection fraction and intensity.",
          "S1: proteins in the expected direction. S2, S3: mean of expected sign x log2 ratio of mean intensity per "
          "cell (S2) or x log2 fold change among detected cells (S3). Empirical one-sided p."),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("7b", ZEROK_VS_CONTROL + "specificity_matched_slots.csv", "Each directional panel protein among its 50 matched proteins",
         (), values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("7c", ZEROK_VS_CONTROL + "specificity_panel_ranks.csv", "Panel proteins among all tested proteins of the group",
         ("Directional percentile: 1 = furthest in the expected direction.",),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values=V3, strict=S3, rename_re=RN, dct_unknown=True),
    Spec("7d", ZEROK_VS_CONTROL + "specificity_plate_null.csv",
         "Sort-plate null: tested proteins with q < 0.05 between plates and between conditions",
         ("The proteome-wide test of table 4a applied to the two sort plates of one condition (technical null) and "
          "to the two conditions within one plate.",),
         drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values={**V3, "contrast": CONTRAST}, strict=(*S3, "contrast"), rename_re=RN),
    Spec("7e", ZEROK_VS_CONTROL + "specificity_plate_null_panel.csv", "Panel proteins in the sort-plate and within-plate contrasts",
         (), drop=("p_value_note",), drop_reason="the note repeats the design statement of the sheet",
         values={**V3, "contrast": CONTRAST}, strict=(*S3, "contrast"), rename_re=RN),
    Spec("8a", ZEROK_VS_CONTROL + "bulk_concordance.csv", "Correlation of the DCT-cell effects with the bulk whole-kidney effects",
         ("Bulk: whole kidney, 0% against 1% potassium diet for 2 weeks, 5 and 7 mice. Proteins matched by UniProt "
          "accession, else by unique gene symbol. Spearman correlation with Fisher-z 95% interval.",
          "DCT-enriched: proteins whose gene is enriched in DCT cells of a mouse kidney RNA atlas."),
         drop=("note",), drop_reason="the note repeats the differences between the two studies given on this sheet",
         values=V3, strict=S3, rename_re=RN),
    Spec("8b", ZEROK_VS_CONTROL + "bulk_plate_null.csv", "The same correlation for the sort-plate and within-plate contrasts",
         (), drop=("note",), drop_reason="the note repeats the differences between the two studies given on this sheet",
         values={**V3, "contrast": CONTRAST}, strict=(*S3, "contrast"), rename_re=RN),
    Spec("8c", ZEROK_VS_CONTROL + "bulk_sign_tables.csv", "Proteins significant in both data sets: direction",
         ("DCT cells: two-part q < 0.05 with direction up or down; bulk: q < 0.05.",),
         drop=("note",), drop_reason="the note repeats the differences between the two studies given on this sheet",
         values=V3, strict=S3, rename_re=RN),
    Spec("8d", ZEROK_VS_CONTROL + "bulk_panel.csv", "Panel proteins in the bulk whole-kidney proteome and in DCT cells",
         (f"DCT cells: {BOTH} of the {CLUSTER}, {LN}.",),
         drop=("note",), drop_const=("scp_scope",),
         drop_reason="the note repeats the differences between the two studies given on this sheet"),
]

TERMS = [
    ("Design", "DCT cells of mice fed a control or a zero-potassium (Zero-K) diet for 4 days; one pool of 3 mice per "
     "condition; two sort plates (P1, P2) per pool; both conditions sorted and acquired in the same sessions and "
     "searched together. 729 control and 702 zero-K cells were retained; 5,897 protein groups were retained in both "
     "conditions and are compared."),
    ("detected", "A protein is detected in a cell if its intensity is above zero. Non-detection is kept as "
     "non-detection (no imputation)."),
    ("labels; group", f"{CLUSTER}, {TRANSFER}, {NOLABEL}: see the note on each sheet. group: the cells compared "
     f"(DCT1, DCT2, Unknown, {BOTH}, {ALLCELLS}, {NCC})."),
    ("variant", f"{LN}: baseline. {CELLMED}: the median ln-intensity of each cell is subtracted. {ALIGNED}: the "
     "zero-K values are shifted so that the median ln-intensity equals that of control."),
    ("expected_direction", "Direction under the zero-K diet fixed in advance: up, down, no change (a cited report "
     "of no difference) or no expectation."),
    ("observed_direction", "up or down when q < 0.05 and the detection and intensity parts do not point in opposite "
     "directions; mixed when they do; otherwise no BH-significant difference."),
    ("match_status", "match or mismatch against the expected direction; not scored for items without expectation."),
    ("det_logodds; det_p", "Detection part: coefficient (zero-K against control) and p of the Firth logistic "
     "regression with log proteins per cell as covariate."),
    ("int_log2fc_adjusted; int_p", "Intensity part: log2 fold change among detected cells, adjusted for log "
     "proteins per cell, and its p."),
    ("pseudobulk_log2fc", "log2 ratio (zero-K / control) of the mean intensity per cell, non-detection counted as "
     "0; P1, P2: cells of one sort plate; pooled: both plates."),
    ("q values", "Benjamini-Hochberg; within the panel (q_primary) or over the tested proteins of a group "
     "(q_two_part). Descriptive, as every p value here."),
    ("score items", "ALDO_activity, ALDO_signed, ALDO_machinery: aldosterone-related protein scores (base: with "
     "Slc12a3; no_Slc12a3: without). cell_cycle_score: 18-protein cell-cycle list."),
    ("contrast", f"{ZCB}, {ZC1}, {ZC2}: zero-K against control. {C_P}, {Z_P}: the two sort plates of one condition "
     "(technical)."),
]


def lab(group: str = BOTH, labels: str = CLUSTER, variant: str = LN) -> dict:
    return {"labels": labels, "group": group, "variant": variant}


def pb(protein: str):
    return lambda f: _pick(f, "pseudobulk_log2fc_pooled", item=protein, **lab())


LETTER = [
    ("retained cells, control and zero-K", "1a",
     lambda f: f"{_pick(f, 'n_cells_retained', condition='Control', sort_plate='all')}, "
               f"{_pick(f, 'n_cells_retained', condition='Zero-K', sort_plate='all')}", "729, 702", None),
    ("panel rows: proteins, scores, limits", "1b",
     lambda f: ", ".join(str(int(f["item_type"].eq(k).sum())) for k in ("protein", "score", "limit")), "22, 6, 5", None),
    ("expected up / down / no change", "1b",
     lambda f: ", ".join(str(int(f["expected_direction"].eq(k).sum())) for k in ("up", "down", "no change")),
     "3, 4, 1", None),
    (f"{BOTH}: cells, control and zero-K", "2a",
     lambda f: f"{_pick(f, 'n_control', item='Wnk1', **lab())}, {_pick(f, 'n_zero_k', item='Wnk1', **lab())}",
     "467, 449", None),
    (f"{BOTH}: scored items matching; Cul3", "2c",
     lambda f: f"{_pick(f, 'match', **lab())}, "
               f"{_pick(f, 'mismatch: significant difference (no change expected)', **lab())}", "7, 1", None),
    (f"{BOTH}, {ALIGNED}: scored items matching", "2c", lambda f: _pick(f, "match", **lab(variant=ALIGNED)), 8, None),
    ("DCT1, cluster annotation: scored items matching", "2c", lambda f: _pick(f, "match", **lab("DCT1")), 7, None),
    (f"{BOTH}, {TRANSFER}: scored items matching", "2c", lambda f: _pick(f, "match", **lab(labels=TRANSFER)), 7, None),
    ("all cells: scored items matching (match rows)", "2b",
     lambda f: int((f["labels"].eq(NOLABEL) & f["group"].eq(ALLCELLS) & f["variant"].eq(LN)
                    & f["match_status"].eq("match")).sum()), 7, None),
    ("NCC detected, control", "2a", lambda f: _pick(f, "detection_fraction_control", item="Slc12a3", **lab()), 0.912, 3),
    ("NCC detected, zero-K", "2a", lambda f: _pick(f, "detection_fraction_zero_k", item="Slc12a3", **lab()), 0.935, 3),
    ("NCC intensity log2 fold change", "2a", lambda f: _pick(f, "int_log2fc_adjusted", item="Slc12a3", **lab()), 0.60, 2),
    ("Wnk1 detected, control", "2a", lambda f: _pick(f, "detection_fraction_control", item="Wnk1", **lab()), 0.069, 3),
    ("Wnk1 detected, zero-K", "2a", lambda f: _pick(f, "detection_fraction_zero_k", item="Wnk1", **lab()), 0.437, 3),
    ("Wnk4 detected, control", "2a", lambda f: _pick(f, "detection_fraction_control", item="Wnk4", **lab()), 0.034, 3),
    ("Wnk4 detected, zero-K", "2a", lambda f: _pick(f, "detection_fraction_zero_k", item="Wnk4", **lab()), 0.290, 3),
    ("Klhl3 detected, control", "2a", lambda f: _pick(f, "detection_fraction_control", item="Klhl3", **lab()), 0.379, 3),
    ("Klhl3 detected, zero-K", "2a", lambda f: _pick(f, "detection_fraction_zero_k", item="Klhl3", **lab()), 0.274, 3),
    ("Pvalb intensity log2 fold change", "2a", lambda f: _pick(f, "int_log2fc_adjusted", item="Pvalb", **lab()), -0.37, 2),
    ("Calb1 intensity log2 fold change", "2a", lambda f: _pick(f, "int_log2fc_adjusted", item="Calb1", **lab()), -0.43, 2),
    ("Slc8a1 intensity log2 fold change", "2a", lambda f: _pick(f, "int_log2fc_adjusted", item="Slc8a1", **lab()), -0.49, 2),
    ("Cul3 intensity log2 fold change", "2a", lambda f: _pick(f, "int_log2fc_adjusted", item="Cul3", **lab()), 0.17, 2),
    ("Cul3 q", "2a", lambda f: _pick(f, "q_primary", item="Cul3", **lab()), 0.011, 3),
    ("log2 ratio of mean intensity, NCC", "2a", pb("Slc12a3"), 0.62, 2),
    ("log2 ratio of mean intensity, Wnk1", "2a", pb("Wnk1"), 4.01, 2),
    ("log2 ratio of mean intensity, Wnk4", "2a", pb("Wnk4"), 3.74, 2),
    ("zero-K DCT2 cells: cluster annotation, transferred label", "3a",
     lambda f: f"{_pick(f, 'n_label', labels=CLUSTER, condition='Zero-K', sort_plate='all', label='DCT2')}, "
               f"{_pick(f, 'n_label', labels=TRANSFER, condition='Zero-K', sort_plate='all', label='DCT2')}",
     "0, 68", None),
    ("zero-K DCT1 and Unknown cells, cluster annotation", "3a",
     lambda f: f"{_pick(f, 'n_label', labels=CLUSTER, condition='Zero-K', sort_plate='all', label='DCT1')}, "
               f"{_pick(f, 'n_label', labels=CLUSTER, condition='Zero-K', sort_plate='all', label='Unknown')}",
     "449, 253", None),
    (f"{BOTH}: share of tested proteins with q < 0.05", "4a", lambda f: _pick(f, "share_q_below_0.05", **lab()), 0.572, 3),
    (f"{BOTH}: tested proteins", "4a", lambda f: _pick(f, "n_proteins_tested", **lab()), 3684, None),
    ("per-protein table: rows", "4b", lambda f: len(f), 3684, None),
    ("protein groups retained in one condition only: control, zero-K", "4c",
     lambda f: f"{int(f['retention'].eq('Control only').sum())}, {int(f['retention'].eq('Zero-K only').sum())}",
     "181, 56", None),
    ("cells with Mki67 or Top2a detected, all cells: control, zero-K", "5b",
     lambda f: f"{int(_pick(f, 'k_control', labels=NOLABEL, group=ALLCELLS, measure='Mki67 or Top2a detected'))}, "
               f"{int(_pick(f, 'k_zero_k', labels=NOLABEL, group=ALLCELLS, measure='Mki67 or Top2a detected'))}",
     "21, 56", None),
    ("depth-matched repeats with the same result: Klhl3", "6a",
     lambda f: _pick(f, "fraction_same_match_status", procedure=DEPTH, item="Klhl3", **lab()), 1.0, 2),
    ("depth-matched repeats with the same result: Slc8a1", "6a",
     lambda f: _pick(f, "fraction_same_match_status", procedure=DEPTH, item="Slc8a1", **lab()), 0.90, 2),
    ("depth-matched repeats with the same result: Calb1", "6a",
     lambda f: _pick(f, "fraction_same_match_status", procedure=DEPTH, item="Calb1", **lab()), 0.31, 2),
    ("depth-matched repeats with the same result: Pvalb", "6a",
     lambda f: _pick(f, "fraction_same_match_status", procedure=DEPTH, item="Pvalb", **lab()), 0.16, 2),
    ("depth-matched repeats with the same result: NCC, all cells", "6a",
     lambda f: _pick(f, "fraction_same_match_status", procedure=DEPTH, item="Slc12a3", **lab(ALLCELLS, NOLABEL)), 0.09, 2),
    ("matched random sets: observed proteins in the expected direction", "7a",
     lambda f: _pick(f, "observed", statistic="S1 matches", **lab()), 7, 0),
    ("matched random sets: mean of the random sets", "7a",
     lambda f: _pick(f, "null_mean", statistic="S1 matches", **lab()), 1.63, 2),
    ("matched random sets: empirical p", "7a",
     lambda f: _pick(f, "empirical_p_greater_or_equal", statistic="S1 matches", **lab()), 0.0001, 4),
    ("sort-plate null, all cells: control P2 vs P1", "7d",
     lambda f: _pick(f, "share_q_below_0.05", contrast=C_P, **lab(ALLCELLS, NOLABEL)), 0.417, 3),
    ("sort-plate null, all cells: zero-K P2 vs P1", "7d",
     lambda f: _pick(f, "share_q_below_0.05", contrast=Z_P, **lab(ALLCELLS, NOLABEL)), 0.542, 3),
    ("NCC, Wnk1, Wnk4 between the sort plates of one condition: rows with a difference", "7e",
     lambda f: int((f["labels"].eq(NOLABEL) & f["group"].eq(ALLCELLS) & f["variant"].eq(LN)
                    & f["contrast"].isin([C_P, Z_P]) & f["protein"].isin(["Slc12a3", "Wnk1", "Wnk4"])
                    & f["observed_direction"].ne("no BH-significant difference")).sum()), 0, None),
    ("NCC, Wnk1, Wnk4 within each sort plate: rows observed up", "7e",
     lambda f: int((f["labels"].eq(NOLABEL) & f["group"].eq(ALLCELLS) & f["variant"].eq(LN)
                    & f["contrast"].isin([ZC1, ZC2]) & f["protein"].isin(["Slc12a3", "Wnk1", "Wnk4"])
                    & f["observed_direction"].eq("up")).sum()), 6, None),
    ("bulk: Spearman, all shared proteins", "8a",
     lambda f: _pick(f, "spearman", scp_effect="pseudobulk", protein_set="all shared", **lab()), 0.13, 2),
    ("bulk: Spearman, proteins with bulk q < 0.05", "8a",
     lambda f: _pick(f, "spearman", scp_effect="pseudobulk", protein_set="bulk q < 0.05", **lab()), 0.35, 2),
    ("bulk: Spearman of the control sort-plate contrast, all shared", "8b",
     lambda f: _pick(f, "spearman", contrast=C_P, protein_set="all shared", **lab()), -0.03, 2),
    ("bulk: Spearman of the zero-K sort-plate contrast, all shared", "8b",
     lambda f: _pick(f, "spearman", contrast=Z_P, protein_set="all shared", **lab()), -0.11, 2),
    ("bulk: proteins significant in both", "8c",
     lambda f: _pick(f, "n_both_significant", protein_set="all shared", **lab()), 411, None),
    ("bulk: same direction", "8c", lambda f: _pick(f, "agreement", protein_set="all shared", **lab()), 0.64, 2),
    ("bulk log2 fold change: Pvalb, Calb1, Slc8a1", "8d",
     lambda f: ", ".join(f"{float(_pick(f, 'bulk_log2fc', protein=p)):.2f}" for p in ("Pvalb", "Calb1", "Slc8a1")),
     "-0.87, -1.02, -1.08", None),
]

WORKBOOK = WorkbookDef(
    label="S-19-e", step=STEP, workbook="report_19_S19e_tables.xlsx",
    title="Supplementary Table S-19-e. DCT cells of mice on a zero-potassium diet compared with control: "
          "pre-specified protein panel, label fractions, proteome-wide results, robustness, specificity and "
          "comparison with a bulk kidney proteome",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values. One pool of 3 mice per condition: "
             "condition and pool cannot be separated, and cells are not animal-level replicates. The terms used in "
             "the column names are explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER,
    parameters={"dct": "combined control + zero-K search (zerok_vs_control tables; dct_intake panel coverage; frozen zerok_core panel)",
                "left_out": "per-protein results of the other groups, label routes and variants; projected "
                            "pseudotime; DCT-enriched gene list"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-19-e (zero-K against control).")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
