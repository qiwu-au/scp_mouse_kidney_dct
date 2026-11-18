#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-09-c (Slc12a1 and Aqp2 as negative markers, DCT data), one workbook.

The authors asked on 12 Nov 2026 to assemble Supplementary
Tables S-08-d and S-09-c ("Yes, assemble them"). The response to the reviewers, section 5, fixes the content of S-09-c: one workbook
with six sheets, built from 18 stored tables (10 tables of the response to the reviewers, 8 tables of dct_reviewer). DCT data of the combined
control + zero-K search only. No new statistics: every value is copied from the stored tables, and no row is
dropped. The layout follows Supplementary Table S-07-d; the assembly code is imported from
scripts/report_rna_protein_tables.py (build, Spec, Rules, Layout).

What the assembly changes (all recorded per table in the sources table):
- conditions: Ctrl -> Control, ZeroK -> Zero-K. The two conditions were analysed separately; they are listed one
  after the other in each table and nothing compares them;
- labels: the primary label column (cs_ctrl_deg_label / cs_zerok_pipeline_label) is the "cluster annotation"; its
  value Unknown stays in the cells and every sheet carries the note "Unknown = DCT, subtype not resolved (cells of
  clusters without marker support)". The second label column (cs_<cond>_cluster_label) is the "marker-score label
  (clusters)"; its Unknown is not annotated. The per-cell label behind it (cs_<cond>_cell_label; rows of three
  tables of sheet 2) is the "marker-score label (per cell)". Readers see only the final products (authors, 12 Nov
  2026), so no cell, header or sheet name refers to the revision process; a check enforces it;
- columns: the columns that hold an internal column name or code are dropped after a check that another column
  gives them (label_column, label_version, run, outcome);
- names: internal names are replaced in column names and cell values (primary / second, confidence codes,
  exclusion sets, label sources, outcomes, panel names, task3_, legacy, det_ / ln1p_);
- the two sets of random removals are named "random removals, same labels" and "random removals, same labels and
  protein depth".

Workbook: results/tables/report/report_09_S09c_tables.xlsx (sheet Contents, then sheets 1-6; Arial 10;
values only, no formulas).

Inputs (read-only): the 18 tables named in SPECS.
Outputs: the workbook; results/tables/report/report_09_s09c_table_{sources,checks}.csv;
         results/tables/report/report_step_09_s09c_table_manifest.json

Command (project root):  python scripts/report_negative_markers_tables.py --overwrite
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


import re

import pandas as pd

from report_pt_tal_tables import _pick, check
from report_rna_protein_tables import Layout, Rules, Spec, build, new_name, rows_where

REPORT, DCT_REVIEWER, SENS = "report", "dct_reviewer", "marker_sensitivity"
NOTE = "Unknown = DCT, subtype not resolved (cells of clusters without marker support)"
CLUSTER, SCORE_CLUSTERS, SCORE_CELL = ("cluster annotation", "marker-score label (clusters)",
                                       "marker-score label (per cell)")
MARKER_SCORE = ("Marker-score label: each cell takes the cell type whose marker proteins give the highest score (per "
                "cell); a cluster at Leiden resolution 3 takes the most frequent label of its cells (clusters).")
SAME_LABELS, SAME_LABELS_DEPTH = "random removals, same labels", "random removals, same labels and protein depth"
STORED = f"{CLUSTER}: remaining cells keeping their label (rule re-applied to the stored clusters)"
EITHER = "Slc12a1 or Aqp2 detected"
RULE_STORED = f"{CLUSTER}, rule re-applied to the stored clusters"

LABEL_SOURCE = {
    "t27_primary_label_remaining_cells": f"{CLUSTER}, labels of the remaining cells before removal",
    "t27_second_label_remaining_cells": f"{SCORE_CLUSTERS}, labels of the remaining cells before removal",
    "primary_label_stored_clusters": RULE_STORED,
    "second_label_renamed_stored_clusters": f"{SCORE_CLUSTERS}, stored clusters named again",
    "cell_label_rederived": f"{SCORE_CELL}, scored again",
    **{f"primary_label_rerun_seed{s}": f"{CLUSTER}, clusters recomputed (seed {s})" for s in (0, 1, 2, 3, 42)},
    **{f"second_label_rerun_seed{s}": f"{SCORE_CLUSTERS}, clusters recomputed (seed {s})" for s in (0, 1, 2, 3, 42)},
}
OUTCOME = {
    "primary label agreement (DE rule on the stored Leiden 1.0 clusters, remaining cells)": STORED,
    "primary label agreement (Leiden 1.0 rerun on the rebuilt graph, seed 0)":
        f"{CLUSTER}: remaining cells keeping their label (clusters recomputed on the rebuilt graph, resolution 1, "
        "seed 0)",
    "ARI, Leiden 1.0 rerun (seed 0) vs stored clusters":
        "adjusted Rand index of the recomputed clusters (resolution 1, seed 0) with the stored clusters",
    "second label agreement (stored Leiden 3.0 clusters re-named)":
        f"{SCORE_CLUSTERS}: remaining cells keeping their label (stored clusters named again)",
    "second label agreement (Leiden 3.0 rerun, seed 0)":
        f"{SCORE_CLUSTERS}: remaining cells keeping their label (clusters recomputed, resolution 3, seed 0)",
    "ARI, Leiden 3.0 rerun (seed 0) vs stored clusters":
        "adjusted Rand index of the recomputed clusters (resolution 3, seed 0) with the stored clusters",
    "per-cell label agreement (re-scored)": f"{SCORE_CELL}: remaining cells keeping their label (scored again)",
}

RULES = Rules(
    label_display={}, label_allowed=frozenset(),
    column_values={
        "condition": {"Ctrl": "Control", "ZeroK": "Zero-K"},
        "label_role": {"all_cells": "all cells", "primary": CLUSTER, "second": SCORE_CLUSTERS,
                       "task3_dct_positive_label": "highest DCT panel score"},
        "label": {"all_cells": "all cells", "DCT_core": "DCT core", "DCT1_associated": "DCT1-associated",
                  "DCT2_associated": "DCT2-associated"},
        "primary_confidence": {"high": "type-specific marker among the 25 most increased proteins",
                               "specific_any_rank": "type-specific marker, any rank",
                               "low_shared_only": "shared markers only",
                               "unknown_no_marker_in_top_de": "no marker support (subtype not resolved)"},
        "marker_set": {"Slc12a1": "Slc12a1", "Aqp2": "Aqp2", "Slc12a1_or_Aqp2": "Slc12a1 or Aqp2",
                       "Slc12a1_and_Aqp2": "Slc12a1 and Aqp2"},
        "exclusion_set": {"none": "none (all cells)", "slc12a1_detected": "Slc12a1 detected",
                          "aqp2_detected": "Aqp2 detected", "slc12a1_or_aqp2_detected": EITHER},
        "label_source": LABEL_SOURCE,
        "display": OUTCOME,
        "null_type": {"random removals within primary label x protein-ID quartile": SAME_LABELS_DEPTH},
        "panel": {"DCT_core": "DCT core", "DCT1_associated": "DCT1-associated", "DCT2_associated": "DCT2-associated",
                  "TAL": "TAL", "CNT_CD": "CNT and collecting duct", "Interstitial": "Interstitial",
                  "ProLIF_legacy (DCT_markers_V1)": "ProLIF (DCT marker list)", "Cell_cycle": "Cell cycle"},
        "role": {"positive": "DCT panel", "negative_contamination": "negative panel",
                 "context_legacy": "context (ProLIF markers)", "context_cell_cycle": "context (cell cycle)"},
        "score_column": {"ProLIF_legacy_score": "ProLIF_score"},
        "description": {"original DCT V1 workbook lists": "original DCT marker lists"},
    },
    strict=frozenset({"condition", "label_role", "primary_confidence", "marker_set", "exclusion_set", "label_source",
                      "display", "null_type", "panel", "role"}),
    rename_exact={
        "label_role": "annotation", "primary_cluster": "cluster", "primary_label": "cluster_annotation",
        "primary_confidence": "marker_support", "second_label": "marker_score_label_clusters",
        "n_detected_proteins": "n_protein_ids",
        "excluded_by_primary_label": "excluded_by_cluster_annotation",
        "excluded_by_second_label": "excluded_by_marker_score_label",
        "n_primary_label_changed_stored_clusters": "n_cells_label_changed_stored_clusters",
        "primary_rerun_agreement_seeds": "recomputed_clusters_seeds",
        "primary_rerun_agreement_min": "recomputed_clusters_agreement_min",
        "primary_rerun_agreement_median": "recomputed_clusters_agreement_median",
        "primary_rerun_agreement_max": "recomputed_clusters_agreement_max",
        "label_source": "labels", "display": "outcome", "null_type": "random_removals",
        "n_ambiguous_tal_cnt_interstitial": "n_any_flag",
        "fraction_ambiguous_tal_cnt_interstitial": "fraction_any_flag",
        "n_prolif": "n_ProLIF", "prolif_share": "fraction_ProLIF",
    },
    rename_re=((re.compile(r"^ProLIF_legacy_score"), "ProLIF_score"),
               (re.compile(r"^task3_dct_positive_"), "highest_DCT_panel_"),
               (re.compile(r"^max_tal_cnt_interstitial_score"), "max_negative_panel_score"),
               (re.compile(r"_specific_for_de_rule$"), "_type_specific_markers"),
               (re.compile(r"^det_(.+)$"), r"\1_detected"), (re.compile(r"^ln1p_(.+)$"), r"\1_ln1p_intensity")),
    forbidden=re.compile(r"(?<![A-Za-z0-9])t\d\d(?![0-9A-Za-z])|cs_(ctrl|zerok)|label_column|\bprimary\b|\bsecond\b"
                         r"|\bdeg\b|task3|legacy|\bCtrl\b|\bZeroK\b|(?<![A-Za-z0-9])v[123](?![A-Za-z0-9])"
                         r"|\bDE rule|de_rule|unknown_no_marker|specific_any_rank|low_shared_only|rederived|rerun"
                         r"|negative_contamination|\bbrief\b|\.csv|\.h5ad|\buser\b", re.I),
    forbidden_header=re.compile(r"primary|second_|task3|legacy|de_rule|label_column|label_role|label_source"
                                r"|label_version|null_type|ambiguous|^det_|^ln1p_|prolif_share|n_prolif"),
    other_rows=lambda src: pd.Series(False, index=src.index),
)

LABEL_COLUMN = {"label_column": ("condition", "label_role")}
SENS_COLUMNS = {"label_version": ("label_role",), "label_column": ("condition", "label_role")}
SHEETS = {
    1: ("1 Slc12a1 and Aqp2 detection", "Detection of Slc12a1 and Aqp2 by label, cluster and protein depth, and "
                                        "together with NCC and parvalbumin"),
    2: ("2 Exclusion, random removals", "Annotation without the cells with Slc12a1 or Aqp2 detected, against random "
                                        "removals"),
    3: ("3 Panel scores and flags", "Panel coverage, flag thresholds, and scores and flags by label"),
    4: ("4 List markers by label", "The list markers, Umod and Slc12a1 by label"),
    5: ("5 Marker-list variants", "Marker-list variants and leave-one-out"),
    6: ("6 Per cell", "Per-cell values"),
}

SPECS = [
    Spec("1a", (REPORT, "09_negative_marker_by_label"), "Slc12a1 and Aqp2 detection by label",
         ("Cells with the protein detected (n_detected of n_cells; fraction_detected with its Wilson 95% interval), "
          f"for all cells and per label of the {CLUSTER} and of the {SCORE_CLUSTERS}.",
          "median_protein_ids_, auc_ and p_ columns: protein IDs of the cells with the marker detected against the "
          "cells without; see Contents (protein IDs)."),
         determined=LABEL_COLUMN),
    Spec("1b", (REPORT, "09_negative_marker_by_cluster"), "Slc12a1 and Aqp2 detection by cluster",
         (f"Leiden clusters of the {CLUSTER}, with their label and marker support (see Contents). n_ and fraction_ "
          "columns: cells of the cluster with the protein detected.",)),
    Spec("1c", (REPORT, "09_negative_marker_depth"), "Slc12a1 and Aqp2 detection by quartile of protein IDs",
         ("Quartiles of the number of proteins identified per cell within the condition (Q1 lowest); "
          "protein_ids_min, protein_ids_max: range of the quartile.",)),
    Spec("1d", (REPORT, "09_negative_marker_codetection"), "NCC, parvalbumin and other list markers in the cells with "
         "and without Slc12a1 or Aqp2",
         ("Groups: cells with the negative marker detected; cells without it; cells without it in the top quartile "
          "of protein IDs (Q4). n_ and fraction_ columns: cells of the group with the protein detected (Slc12a3, "
          "NCC; Pvalb, parvalbumin).",
          "median_ln1p_Slc12a3_in_detected_cells: median ln(1 + intensity) of NCC in the cells with NCC detected.")),
    Spec("2a", (REPORT, "09_negative_marker_exclusion_sets"), "Cells removed (exclusion sets) and the effect on the "
         "cluster annotation",
         ("n_excluded, n_remaining: cells removed and left; excluded_by_ columns: removed cells per label. "
          "n_cells_label_changed_stored_clusters, n_clusters_label_changed: remaining cells and clusters whose "
          "label changes when the rule is re-applied to the stored clusters.",
          "recomputed_clusters_agreement_ columns: fraction of the remaining cells that keep their label with "
          "recomputed clusters, over the seeds in recomputed_clusters_seeds.")),
    Spec("2b", (REPORT, "09_negative_marker_exclusion_label_counts"), "Label counts before and after removal",
         ("exclusion_set none: no cell removed. labels: the labels counted and how they were obtained; see Contents "
          "(Removal of the marker-detected cells). n_cells, fraction: cells per label, and their fraction of the "
          "remaining cells.",),
         determined={"run": ("exclusion_set",)}),
    Spec("2c", (REPORT, "09_negative_marker_exclusion_clusters"), "Clusters before and after removal",
         (f"Per cluster of the {CLUSTER}: cells, cells removed and left, label before, and label after the rule is "
          "re-applied to the stored clusters.",)),
    Spec("2d", (REPORT, "09_negative_marker_exclusion_vs_random"), f"Removal of the marker-detected cells against "
         f"{SAME_LABELS}",
         (f"{SAME_LABELS}: 100 removals of the same number of cells, drawn within each label of the {CLUSTER}. "
          "Columns: see Contents (Random-removal columns).",),
         determined={"outcome": ("display",)}),
    Spec("2e", (REPORT, "09_negative_marker_depth_null_vs_random"), f"Removal of the marker-detected cells against "
         f"{SAME_LABELS_DEPTH}",
         (f"{SAME_LABELS_DEPTH}: 100 removals of the same number of cells, drawn within each label of the {CLUSTER} "
          "and quartile of protein IDs. Columns as table 2d.",),
         determined={"outcome": ("display",)}),
    Spec("3a", (DCT_REVIEWER, "task3_marker_coverage"), "Panels and their coverage",
         ("n_input, n_present, n_missing: proteins listed for the panel, quantified and not quantified in the "
          "condition (present, missing). score_column: name of the panel score in table 3c.",)),
    Spec("3b", (DCT_REVIEWER, "task3_ambiguity_thresholds"), "Flag thresholds of the negative panels",
         ("A cell is flagged on a panel if its score is above the 90th percentile of the condition (threshold). "
          "n_flagged: flagged cells. informative: the threshold is defined.",)),
    Spec("3c", (DCT_REVIEWER, "task3_marker_summary_by_label"), "Panel scores and flags by label",
         (f"Per label of the {CLUSTER} and of the {SCORE_CLUSTERS}, and per group of cells by their highest DCT panel "
          "score. n_ and fraction_ flag columns: flagged cells; any_flag: at least one flag.",
          "_score_median, _score_mean: median and mean of each panel score. See Contents (Panel scores)."),
         determined=LABEL_COLUMN),
    Spec("4a", (DCT_REVIEWER, "umod_markers_by_label"), "Detection and intensity of Umod, Slc12a1 and the DCT list markers "
         "by label",
         ("expected_segment: segment the marker is expected in. detection_fraction with its Wilson 95% interval. "
          "median, q25 and q75 _ln1p_detected: ln(1 + intensity) in the cells with the protein detected; "
          "_ln1p_all_cells: over all cells of the label (0 if not detected).",),
         determined=LABEL_COLUMN),
    Spec("4b", (DCT_REVIEWER, "umod_slc12a1_codetection"), "Umod and Slc12a1 detected together, by label",
         ("Cells with Umod, with Slc12a1, with both, with one and not the other, and with neither.",),
         determined=LABEL_COLUMN),
    Spec("5a", (DCT_REVIEWER, "marker_sensitivity_variant_definitions", SENS), "Marker-list variants",
         ("The variants are the same in both conditions. _markers_scored: markers of the list that are quantified; "
          "_type_specific_markers: markers that count as type-specific in the cluster annotation (Slc12a3 and Umod "
          "are treated as shared).",)),
    Spec("5b", (DCT_REVIEWER, "marker_sensitivity_label_counts", SENS), "Label counts per marker-list variant",
         ("n_cells, fraction: cells per label under the variant; n_cells_original: under the original lists.",),
         determined=SENS_COLUMNS),
    Spec("5c", (DCT_REVIEWER, "marker_sensitivity_loo_marker_impact", SENS), "Leave-one-out: cells that change label when "
         "one marker is removed",
         ("n_changed, fraction_changed: cells whose label differs from the label under the original lists; "
          "n_changed_from_ columns: these cells by their original label. n_ProLIF, fraction_ProLIF: ProLIF cells "
          "under the variant.",
          "n_ties_max_score: cells with tied highest scores. lists_present: the lists used."),
         determined=SENS_COLUMNS),
    Spec("6a", (REPORT, "09_negative_marker_per_cell"), "Per cell: labels, protein IDs and marker detection",
         ("One row per cell (729 control and 702 zero-K cells). _detected: intensity above zero; _ln1p_intensity: "
          "ln(1 + intensity), 0 if not detected. cluster, marker_support: Leiden cluster of the cluster annotation "
          "and its marker support.",
          "plate, cell_key, cell_id, n_protein_ids, protein_id_quartile: see Contents.")),
]

TERMS = [
    ("Data", "DCT cells of mice on a control or a zero-potassium (zero-K) diet for 4 days, searched together: 729 "
     "control and 702 zero-K cells; one pool of 3 mice per condition. Control and Zero-K were analysed separately. "
     "They are listed one after the other in each table, and no value compares the two conditions."),
    (CLUSTER, "Label of the Leiden cluster of the cell (resolution 1) under the annotation rule: DCT1, DCT2 or "
     f"Unknown. {NOTE}. Control: DCT1 382, DCT2 85, Unknown 262 cells; zero-K: DCT1 449, Unknown 253 cells."),
    ("marker-score label", f"{MARKER_SCORE} Its labels are DCT1, DCT2 and ProLIF, and for clusters also Unknown."),
    ("marker_support", f"Marker evidence behind the {CLUSTER} of a cluster: type-specific marker among the 25 most "
     "increased proteins; type-specific marker, any rank (significantly increased at any rank); shared markers "
     "only (Slc12a3 or Umod); no marker support (subtype not resolved)."),
    ("detected", "Intensity above zero in the cell. fraction_detected, detection_fraction: fraction of the cells; "
     "ci_low, ci_high: Wilson 95% interval."),
    ("protein IDs", "Number of proteins identified in a cell (n_protein_ids). protein_id_quartile: quartile within "
     "the condition (Q1 lowest). auc_protein_ids_detected_vs_not: probability that a cell with the marker detected "
     "has more protein IDs than a cell without it; p_mannwhitney_protein_ids: Mann-Whitney test."),
    ("marker_set; exclusion_set", "Slc12a1; Aqp2; Slc12a1 or Aqp2 (at least one detected); Slc12a1 and Aqp2 (both "
     "detected). exclusion_set: the cells removed."),
    ("ln1p", "ln(1 + intensity); 0 if the protein is not detected."),
    ("Removal of the marker-detected cells", "The cells of an exclusion set were removed, the remaining cells were "
     "log-transformed and scaled again, and the annotation rule was re-applied to the stored Leiden clusters "
     "(stored clusters) and to clusters recomputed on a rebuilt neighbour graph (recomputed clusters; resolution 1; "
     f"seeds 0, 1, 2, 3 and 42). The {SCORE_CLUSTERS} and the {SCORE_CELL} were derived again in the same way "
     "(resolution 3)."),
    (SAME_LABELS, f"100 random removals of the same number of cells as the exclusion set, drawn within each label "
     f"of the {CLUSTER}."),
    (SAME_LABELS_DEPTH, f"100 random removals of the same number of cells, drawn within each label of the {CLUSTER} "
     "and quartile of protein IDs."),
    ("Random-removal columns", "outcome: fraction of the remaining cells that keep their label, or adjusted Rand "
     "index of the recomputed with the stored clusters. baseline_all_cells: the outcome with no cell removed. "
     "observed: marker-detected cells removed. random_median, random_q025, random_q975, random_min, random_max: over "
     "the random_n random removals (q025 to q975: central 95%). observed_percentile_in_random: percentage of the "
     "random removals below the observed value. p_empirical_two_sided: empirical two-sided p of the observed value "
     "among the random removals. inside_central_95: observed value within the central 95%."),
    ("Panel scores", "Scanpy score_genes on the scaled matrix; panels and their quantified proteins in table 3a. "
     "Negative panels: TAL; CNT and collecting duct; Interstitial. A cell is flagged on a negative panel if its "
     "score is above the 90th percentile of the condition (table 3b; tal_like_flag, cnt_cd_like_flag, "
     "interstitial_like_flag; any_flag: at least one). The flags mark the highest tenth of the cells by construction "
     "and do not estimate contamination. highest DCT panel score: cells grouped by the DCT panel (DCT core, "
     "DCT1-associated, DCT2-associated) that scores highest; highest_DCT_panel_score, highest_DCT_panel_margin: that "
     "score and its difference to the next DCT panel; max_negative_panel_score: highest of the three negative-panel "
     "scores."),
    ("Marker-list variants", "The annotation was repeated with changed marker lists (table 5a). original: the DCT "
     "marker lists; noSlc12a3: Slc12a3 removed from all lists; noSlc12a3_noUmod: Slc12a3 removed and Umod removed "
     "from DCT1; loo_ followed by a marker: this marker removed from every list (leave-one-out)."),
    ("plate; cell_key; cell_id", "plate: sort plate of the pooled cells (P1, P2); technical only, not an animal or "
     "replicate. cell_key: condition, plate and well. cell_id: name of the run in the raw data."),
]


def _n(frame: pd.DataFrame, col: str, **sel) -> int:
    return int(_pick(frame, col, **sel))


def _triple(frame: pd.DataFrame, cols: tuple[str, ...], **sel) -> str:
    return ", ".join(str(_pick(frame, c, **sel)) for c in cols)


def _labels(frame: pd.DataFrame, col: str = "n_cells", **sel) -> str:
    d = rows_where(frame, **sel)
    return ", ".join(f"{a} {int(b)}" for a, b in zip(d["label"], d[col]))


C, Z = "Control", "Zero-K"
ALL = dict(annotation="all cells", label="all cells")
CLUSTER_COLS = ("n_cells", "n_Slc12a1", "n_Aqp2", "n_Slc12a1_or_Aqp2")
NCC = ("n_Slc12a3_detected", "n_Pvalb_detected", "n_cells")
Q4 = "Slc12a1 not detected, protein-ID quartile Q4"
RANDOM = ("observed", "random_median", "random_q025", "random_q975")


def _rounded(frame: pd.DataFrame, cols: tuple[str, ...], **sel) -> str:
    return ", ".join(f"{float(_pick(frame, c, **sel)):.3f}" for c in cols)


LETTER = [
    *[(f"{cond}: cells with {ms} detected", "1a",
       lambda f, cond=cond, ms=ms: _n(f, "n_detected", condition=cond, marker_set=ms, **ALL), value, None)
      for cond, ms, value in ((C, "Slc12a1", 75), (C, "Aqp2", 17), (C, "Slc12a1 or Aqp2", 83),
                              (C, "Slc12a1 and Aqp2", 9), (Z, "Slc12a1", 39), (Z, "Aqp2", 5),
                              (Z, "Slc12a1 or Aqp2", 43), (Z, "Slc12a1 and Aqp2", 1))],
    *[(f"{cond}: fraction of cells with {ms} detected", "1a",
       lambda f, cond=cond, ms=ms: _pick(f, "fraction_detected", condition=cond, marker_set=ms, **ALL), value, 3)
      for cond, ms, value in ((C, "Slc12a1", 0.103), (Z, "Slc12a1", 0.056), (C, "Aqp2", 0.023), (Z, "Aqp2", 0.007))],
    *[(f"{cond}, {label}: cells with {ms} detected, of cells", "1a",
       lambda f, cond=cond, label=label, ms=ms: _triple(f, ("n_detected", "n_cells"), condition=cond,
                                                        annotation=CLUSTER, label=label, marker_set=ms), value, None)
      for cond, label, ms, value in ((C, "DCT1", "Slc12a1", "16, 382"), (C, "DCT2", "Slc12a1", "22, 85"),
                                     (C, "Unknown", "Slc12a1", "37, 262"), (Z, "DCT1", "Slc12a1", "17, 449"),
                                     (Z, "Unknown", "Slc12a1", "22, 253"), (C, "DCT1", "Aqp2", "2, 382"),
                                     (C, "DCT2", "Aqp2", "2, 85"), (C, "Unknown", "Aqp2", "13, 262"),
                                     (Z, "DCT1", "Aqp2", "5, 449"), (Z, "Unknown", "Aqp2", "0, 253"))],
    *[(f"{cond}, top quartile of protein IDs: cells with {ms} detected, of cells", "1c",
       lambda f, cond=cond, ms=ms: _triple(f, ("n_detected", "n_cells"), condition=cond, protein_id_quartile="Q4",
                                           marker_set=ms), value, None)
      for cond, ms, value in ((C, "Slc12a1", "45, 182"), (C, "Aqp2", "15, 182"), (Z, "Slc12a1", "26, 176"),
                              (Z, "Aqp2", "2, 176"))],
    *[(f"{cond}, cluster {cluster}: cells, Slc12a1, Aqp2, either", "1b",
       lambda f, cond=cond, cluster=cluster: _triple(f, CLUSTER_COLS, condition=cond, cluster=cluster), value, None)
      for cond, cluster, value in ((C, 0, "64, 26, 12, 30"), (C, 2, "75, 22, 2, 23"), (Z, 5, "66, 19, 0, 19"))],
    ("Control, cluster 0: median protein IDs", "1b",
     lambda f: _pick(f, "median_protein_ids", condition=C, cluster=0), 3910, 0),
    ("Zero-K, cluster 5: median protein IDs", "1b",
     lambda f: _pick(f, "median_protein_ids", condition=Z, cluster=5), 3007, 0),
    *[(f"{cond}, {group}: cells with NCC, with parvalbumin, of cells", "1d",
       lambda f, cond=cond, group=group: _triple(f, NCC, condition=cond, group=group), value, None)
      for cond, group, value in ((C, "Slc12a1 detected", "59, 52, 75"), (Z, "Slc12a1 detected", "33, 28, 39"),
                                 (C, Q4, "125, 110, 137"), (Z, Q4, "141, 131, 150"))],
    *[(f"{cond}: cells removed (Slc12a1, Aqp2, either)", "2a",
       lambda f, cond=cond: ", ".join(str(v) for v in rows_where(f, condition=cond)["n_excluded"]), value, None)
      for cond, value in ((C, "75, 17, 83"), (Z, "39, 5, 43"))],
    *[(f"{cond}: labels after removal of the cells with either marker, {text}", "2b",
       lambda f, cond=cond, labels=labels: _labels(f, condition=cond, exclusion_set=EITHER, labels=labels),
       value, None)
      for cond, text, labels, value in (
          (C, "rule re-applied to the stored clusters", RULE_STORED, "DCT1 364, DCT2 138, Unknown 144"),
          (Z, "rule re-applied to the stored clusters", RULE_STORED, "DCT1 297, DCT2 131, Unknown 231"),
          (C, "labels before removal", LABEL_SOURCE["t27_primary_label_remaining_cells"],
           "DCT1 364, DCT2 62, Unknown 220"),
          (Z, "labels before removal", LABEL_SOURCE["t27_primary_label_remaining_cells"],
           "DCT1 428, Unknown 231"))],
    *[(f"{cond}, cluster {cluster}, either marker removed: cells, removed, label before, label after", "2c",
       lambda f, cond=cond, cluster=cluster: _triple(
           f, ("n_cells", "n_removed", "label_before", "label_after_stored_clusters"), condition=cond,
           exclusion_set=EITHER, cluster=cluster), value, None)
      for cond, cluster, value in ((C, 4, "81, 5, Unknown, DCT2"), (Z, 1, "85, 4, DCT1, DCT2"),
                                   (Z, 12, "52, 2, DCT1, DCT2"), (C, 0, "64, 30, Unknown, Unknown"),
                                   (C, 2, "75, 23, DCT2, DCT2"), (Z, 5, "66, 19, Unknown, Unknown"))],
    *[(f"{cond}, {name}: observed, median, 2.5% and 97.5% of the random removals", key,
       lambda f, cond=cond: _rounded(f, RANDOM, condition=cond, exclusion_set=EITHER, outcome=STORED), value, None)
      for cond, name, key, value in ((C, SAME_LABELS, "2d", "0.882, 0.951, 0.843, 1.000"),
                                     (Z, SAME_LABELS, "2d", "0.801, 0.903, 0.807, 1.000"),
                                     (C, SAME_LABELS_DEPTH, "2e", "0.882, 1.000, 0.943, 1.000"),
                                     (Z, SAME_LABELS_DEPTH, "2e", "0.801, 0.879, 0.800, 1.000"))],
    *[(f"{cond}: cells flagged per negative panel", "3b",
       lambda f, cond=cond: ", ".join(str(v) for v in rows_where(f, condition=cond)["n_flagged"]), value, None)
      for cond, value in ((C, "73, 73, 73"), (Z, "71, 71, 71"))],
    *[(f"{cond}: cells with at least one flag, by label of the cluster annotation", "3c",
       lambda f, cond=cond: _labels(f, "n_any_flag", condition=cond, annotation=CLUSTER), value, None)
      for cond, value in ((C, "DCT1 82, DCT2 22, Unknown 95"), (Z, "DCT1 106, Unknown 85"))],
    *[(f"{cond}: cells with Umod detected; cells with Slc12a1 and no Umod", "4b",
       lambda f, cond=cond: ", ".join(str(int(sum(int(v) for v in rows_where(f, condition=cond,
                                                                              annotation=CLUSTER)[col])))
                                      for col in ("n_umod_detected", "n_slc12a1_without_umod")), value, None)
      for cond, value in ((C, "712, 0"), (Z, "690, 0"))],
    *[(f"{cond}: labels without Slc12a3 and Umod in the lists", "5b",
       lambda f, cond=cond: _labels(f, condition=cond, variant="noSlc12a3_noUmod", annotation=CLUSTER), value, None)
      for cond, value in ((C, "DCT1 287, DCT2 85, Unknown 357"), (Z, "DCT1 354, Unknown 348"))],
    ("Zero-K: labels without Egf in the lists", "5b",
     lambda f: _labels(f, condition=Z, variant="loo_Egf", annotation=CLUSTER), "DCT1 258, DCT2 137, Unknown 307",
     None),
    *[(f"{cond}: cells that change label without {marker}", "5c",
       lambda f, cond=cond, marker=marker: _n(f, "n_changed", condition=cond, dropped_marker=marker,
                                              annotation=CLUSTER), value, None)
      for cond, marker, value in ((C, "Klk1", 0), (Z, "Klk1", 0), (C, "Egf", 0), (Z, "Egf", 191))],
    ("cells: control, zero-K", "6a",
     lambda f: f"{len(rows_where(f, condition=C))}, {len(rows_where(f, condition=Z))}", "729, 702", None),
    ("per cell: control cells with Slc12a1, with Aqp2, with either", "6a",
     lambda f: ", ".join(str(int(sum(bool(v) for v in rows_where(f, condition=C)[col])))
                         for col in ("Slc12a1_detected", "Aqp2_detected", "Slc12a1_or_Aqp2_detected")),
     "75, 17, 83", None),
    ("per cell: cluster annotation Unknown, control and zero-K", "6a",
     lambda f: ", ".join(str(len(rows_where(f, condition=cond, cluster_annotation="Unknown"))) for cond in (C, Z)),
     "262, 253", None),
]

LAYOUT = Layout(
    number="S-09-c", step="09_s09c_table", workbook="report_09_S09c_tables.xlsx",
    title="Supplementary Table S-09-c. Slc12a1 and Aqp2 as negative markers in the DCT data: detection, annotation "
          "without the cells in which they were detected, panel scores and marker-list variants",
    intro=("SCP-MS of DCT cells from mice on a control or a zero-potassium (zero-K) diet: 729 control and 702 zero-K "
           "cells. All values are observed values; the terms used in the column names are explained below the list "
           "of tables.",
           "Control and Zero-K were analysed separately and are listed one after the other in each table; no value "
           "compares the two conditions.",
           f"Cluster annotation: {NOTE}."),
    sheets=SHEETS, terms=TERMS,
    sheet_lines={n: (f"Cluster annotation: {NOTE}.",) for n in SHEETS},
)


def extra_checks(frames: dict, state: dict, checks: list[dict]) -> None:
    bad = {k: sorted(set(f["condition"].astype(str)) if "condition" in f.columns else ["no condition column"])
           for k, f in frames.items()
           if "condition" not in f.columns or set(f["condition"].astype(str)) != {"Control", "Zero-K"}}
    check(checks, "every table has a condition column with the values Control and Zero-K in every row (each row "
          "belongs to one condition; no row compares them)", bad or f"{len(frames)} tables", "all tables", not bad)
    with_label = sorted({spec.sheet for spec in SPECS if spec.key in frames and (
        {"cluster_annotation", "annotation", "labels", "label_before", "excluded_by_cluster_annotation",
         "outcome"} & set(frames[spec.key].columns))})
    noted = sorted(n for n, lines in LAYOUT.sheet_lines.items() if any(NOTE in line for line in lines))
    check(checks, "sheets that show the cluster annotation carry the note on Unknown", f"sheets with the cluster "
          f"annotation: {with_label}; sheets with the note: {noted}", "every sheet with the cluster annotation",
          set(with_label) <= set(noted) and len(with_label) > 0)
    values = {}
    for spec in SPECS:
        f = frames.get(spec.key)
        if f is None:
            continue
        if "annotation" in f.columns and "label" in f.columns:
            values.setdefault(spec.key, set()).update(f.loc[f["annotation"].eq(CLUSTER), "label"].astype(str))
        if "cluster_annotation" in f.columns:
            values.setdefault(spec.key, set()).update(f["cluster_annotation"].astype(str))
    odd = {k: sorted(v - {"DCT1", "DCT2", "Unknown", "ProLIF"}) for k, v in values.items()
           if v - {"DCT1", "DCT2", "Unknown", "ProLIF"}}
    has_unknown = [k for k, v in values.items() if "Unknown" in v]
    check(checks, "cluster annotation: the label values are unchanged (Unknown stays Unknown in the cells)",
          odd or f"{len(values)} tables with the cluster annotation; Unknown present in {len(has_unknown)}",
          "DCT1, DCT2, Unknown (ProLIF only in marker-list variants)", not odd and len(has_unknown) == len(values))
    used = {v for f in frames.values() for c in ("annotation", "labels", "outcome") if c in f.columns
            for v in f[c].dropna().astype(str)}
    in_cells = {name: sum(name in v for v in used) for name in (SCORE_CLUSTERS, SCORE_CELL)}
    defined = [term for term, meaning in TERMS if MARKER_SCORE in meaning]
    check(checks, "shared vocabulary: the names 'marker-score label (clusters)' and 'marker-score label (per cell)' "
          "are used in the tables, and Contents gives the agreed definition word for word",
          f"distinct values with the name: {in_cells}; defined under: {defined}",
          "both names used; one definition", all(in_cells.values()) and len(defined) == 1)
    names = [new_name(c, spec, RULES) for spec in SPECS for c in ("label_role", "primary_label", "second_label")]
    check(checks, "names of the two label columns", sorted(set(names)), "annotation, cluster_annotation, "
          "marker_score_label_clusters", set(names) == {"annotation", "cluster_annotation",
                                                         "marker_score_label_clusters"})


if __name__ == "__main__":
    build(LAYOUT, SPECS, RULES, LETTER, "the response to the reviewers: Supplementary Table S-09-c, combined control and zero-K "
          "search.", extra_checks)
