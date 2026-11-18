#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-07-d (PT / TAL heterogeneity), one workbook, revised annotation only.

The authors asked on 11 Nov 2026 to "assemble Table S-07-d".
The response to the reviewers, section 5, fixes its content: one workbook with ten sheets, built from 16 stored pt_tal tables (sheets 1-7,
Mouse Kidney Atlas) and 16 stored rat_segments tables (sheets 8-10, rat segment proteomes), keeping only the rows and columns
of the revised annotation (kidney_t15_label; one-annotation rule, authors 30 Oct 2026). No new statistics: every value
is copied from the stored tables.

What the assembly changes (all recorded per table in the sources table):
- rows: only the revised-annotation rows (grouping t15 / label_column kidney_t15_label / panels_from t15);
- columns: the columns that hold another annotation are dropped (scp_pre_*, scp_consensus_*, scp_fig2a_*,
  md_vs_fig2a_*, pre_*, *_pre, scp_det_md_fig2a); the columns that only name the annotation are dropped after a
  check that they are constant;
- names: internal task codes are replaced in column names and cell values (t15, t09, limbutara, v3, plan);
- two free-text tables: rat_segments coverage_summary loses the four rows of the other annotation; pt_tal power_unresolved loses
  two rows that compare the annotations, and four rows lose the clauses on the other annotation (each edit is a
  pattern that must match exactly once).

Workbook: results/tables/report/report_07_S07d_tables.xlsx (sheet Contents, then sheets 1-10; Arial 10;
values only, no formulas; numbers stored with 16 significant digits and shown with a display format). The shared output guard knows no .xlsx
extension, so the workbook path is checked with cc._check_writable and added to the run outputs by hand.

Inputs (read-only): the 32 pt_tal / rat_segments tables named in SPECS.
Outputs: the workbook; results/tables/report/report_07_s07d_table_{sources,letter_numbers,checks}.csv;
         results/tables/report/report_step_07_s07d_table_manifest.json

Command (project root):  python scripts/report_pt_tal_tables.py --overwrite
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
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import scp_common as cc

TASK = "report"
STEP = "07_s07d_table"
WORKBOOK = "report_07_S07d_tables.xlsx"
COL = "kidney_t15_label"
PT_TAL, RAT_SEGMENTS = "pt_tal", "rat_segments"
TABLE_TITLE = ("Supplementary Table S-07-d. Proximal tubule and TAL heterogeneity: comparison of the SCP-MS cells with "
               "the Mouse Kidney Atlas and with the proteomes of microdissected rat tubule segments")

ID_COLS = ("label_column", "grouping", "grouping_text", "panels_from")       # they only name the annotation
OTHER_ANNOTATION = re.compile(r"^(scp_pre_|scp_consensus_|scp_fig2a_|md_vs_fig2a_|pre_)|_pre$|^scp_det_md_fig2a$")
RENAME_EXACT = {
    "in_panels_t15": "in_atlas_panels", "ari_t15": "ari_cell_types", "in_panels_fc2_t15": "in_rat_panels_fc2",
    "scp_10pct_t15": "scp_10pct", "atlas_direction_t09": "atlas_direction", "atlas_contrast_t09": "atlas_contrast",
    "brief_group": "expected_group", "atlas_direction_vs_brief_group": "atlas_direction_vs_expected_group",
    "v3_marker_types": "annotation_marker_types", "n_v3": "n_annotation_markers",
    "n_v3_markers_in_panel": "n_annotation_markers_in_panel",
}
RENAME_RE = ((re.compile(r"^scp_t15_"), "scp_"), (re.compile(r"^t15_"), ""), (re.compile(r"limbutara"), "rat"),
             (re.compile(r"t09"), "atlas"))
VALUE_EXACT = {
    "no_v3": "no_annotation_markers", "pt_tal atlas: S1/S2 - S3": "atlas S1/S2 - S3",
    "Limbutara panel -> atlas": "rat panel -> atlas", "pt_tal panel -> Limbutara": "atlas panel -> rat",
    "limbutara_variable": "rat_variable",
}
VALUE_RE = ((re.compile(r"the plan's group"), "the expected group"),
            (re.compile(r"^plan gives no "), "no expected "))
# Internal names that must not reach the workbook (gene symbols such as Krt15 or Ndufv3 do not match).
FORBIDDEN = re.compile(r"(?<![A-Za-z0-9])t(15|09|24)(?![0-9])|fig ?2a|kidney_pre|consensus|limbutara_|\bbrief"
                       r"|(?<![A-Za-z0-9])v3(?![A-Za-z0-9])|\bunion\b|label sets?|both sets"
                       r"|(?<![A-Za-z])pre(?![A-Za-z])", re.I)
P_LIKE = re.compile(r"(^|_)(p|q)(_|$)")
DISPLAY_LABEL_COLS = ("label", "group", "scp_group")

SHEETS = {
    1: ("1 Atlas subtypes", "Mouse Kidney Atlas subtypes and SCP-MS cell types: cells, nuclei and studies"),
    2: ("2 Atlas panels", "Atlas subtype panels and their coverage in the SCP-MS data"),
    3: ("3 Literature markers", "Literature markers: direction in the atlas and in the rat data, protein detection, "
                                "PCT versus PST"),
    4: ("4 Atlas scores", "Atlas subtype scores: PCT versus PST cells, and TAL cells"),
    5: ("5 Atlas mapping", "Nearest atlas subtype of the SCP-MS cells"),
    6: ("6 Subsets subclusters", "Proximal tubule and TAL subsets: embeddings and subclusters"),
    7: ("7 Power unresolved", "Detectable differences, and subtypes and markers not resolved"),
    8: ("8 Rat reference", "Rat tubule segment proteomes: matching, panel rule and panels"),
    9: ("9 Rat scores", "Rat segment scores, nearest rat segment and pseudobulk comparison"),
    10: ("10 Atlas vs rat", "Agreement between the atlas and the rat reference"),
}


@dataclass
class Spec:
    key: str                                   # table id in the workbook, e.g. "4a" (sheet 4, first table)
    task: str
    name: str                                  # source table, without the stage prefix
    title: str
    notes: tuple[str, ...]
    rows: tuple[str, tuple[str, ...]] | None = None        # (column, kept values)
    values: dict[str, dict[str, str]] = field(default_factory=dict)   # source column -> {old value: new value}
    keep_id: tuple[str, ...] = ()              # ID_COLS kept because they separate data sets, not annotations
    special: str = ""                          # "scp_groups" | "coverage" | "unresolved"

    @property
    def sheet(self) -> int:
        return int(self.key[:-1])


SPECS = [
    Spec("1a", PT_TAL, "power_groups", "Group sizes: SCP-MS cell types and atlas subtypes",
         ("SCP-MS rows: cells, and proteins per cell (median and interquartile range).",
          "Atlas rows (annotation level Celltype): cells and nuclei together (n_cells), nuclei, studies, and "
          "study x cells/nuclei batches with >= 20 cells."),
         rows=("grouping", ("kidney_annotation labels", "atlas Celltype")), keep_id=("grouping",),
         values={"grouping": {"kidney_annotation labels": "SCP-MS cell type", "atlas Celltype": "Mouse Kidney Atlas subtype"}}),
    Spec("1b", PT_TAL, "atlas_subtypes", "Atlas labels of the proximal tubule and the TAL at the three annotation levels",
         ("n_source_Cell, n_source_Nuclei: single cells and single nuclei. origins: cells per study.",
          "celltype_of_these_cells: Celltype-level labels of the cells that carry a finer label.")),
    Spec("2a", PT_TAL, "panel_coverage", "Atlas subtype panels: genes at each filter step",
         ("One row per atlas contrast. The filters are listed in Contents (Atlas panel filters).",
          "The PTS2 contrast gave no panel."), rows=("label_column", (COL,))),
    Spec("2b", PT_TAL, "panels", "Atlas subtype panels: genes",
         ("Panel genes in rank order (Wilcoxon score), with the atlas statistics and the detected fraction in the "
          "SCP-MS cells of the segment (scp_detection).",
          "frac_group, frac_reference: detected fraction in the subtype and in the reference. "
          "annotation_marker_types: cell types that list the protein in the annotation marker list."),
         rows=("label_column", (COL,))),
    Spec("3a", PT_TAL, "literature", "Literature markers in the atlas and in the SCP-MS data",
         ("expected_group: the subtype that the marker is reported for. atlas_ columns: the contrast checked, its "
          "statistics and its direction.",
          "n_cells_detected_all_712: cells with the protein detected, of all 712 cells. scp_ columns: PCT versus PST "
          "cells (cells, detected fractions, AUC, Mann-Whitney p).")),
    Spec("3b", PT_TAL, "literature_scp_groups", "Literature markers: detection per SCP-MS cell type and subset",
         ("Quantified markers only. mean_ln1p: mean ln(1 + intensity).",), special="scp_groups"),
    Spec("3c", RAT_SEGMENTS, "compare_literature", "Literature markers in the rat segment proteomes",
         ("log2_ columns: segment means (log2, centred per sample). n_quant_ columns: samples with the protein "
          "quantified.",
          "rat_contrast: the contrast checked. samples_separated: every quantified group sample above every "
          "quantified reference sample.")),
    Spec("4a", PT_TAL, "scores_pct_vs_pst", "PCT versus PST cells on the atlas subtype scores",
         ("Group 1 = PCT cells, group 2 = PST cells. One row per score, gene set and method; primary rows: gene_set "
          "all, method mean_z.",
          "Last row: ln(protein IDs) itself. Columns: see Contents (PCT versus PST columns; coef_ columns)."),
         rows=("grouping", ("t15",))),
    Spec("4b", PT_TAL, "scores_tal", "Atlas TAL scores in the TAL cells",
         ("Median and quartiles of each score in the TAL cells, and its Spearman correlation with ln(protein IDs) "
          "(rho_ln_ids_tal).",
          "rho_with_other_tal_score: Spearman correlation between the CTAL and the MTAL score."),
         rows=("label_column", (COL,))),
    Spec("5a", PT_TAL, "map_summary", "Mapping to the atlas subtypes: summary per variant",
         ("Cells of each subset mapped to the atlas subtype centroids by cosine similarity. variant and the "
          "columns: see Contents (variant; Mapping columns).",
          "frac_top_ columns: fraction of the cells nearest to each subtype.")),
    Spec("5b", PT_TAL, "map_pt_agreement", "Mapping to the atlas subtypes: PCT and PST cells nearest to the expected "
         "subtypes",
         ("Primary variant. Expected: PCT cells nearest PTS1 or PTS2, PST cells nearest PTS3 or PTS3T2.",),
         rows=("grouping", ("t15",))),
    Spec("5c", PT_TAL, "map_confusion", "Mapping to the atlas subtypes: cells per SCP-MS cell type and subtype",
         ("n_top: cells of the cell type (label; n_label cells in the subset) nearest to the subtype (class). "
          "n_top_confident: of these, assigned with confidence. frac_top: n_top / n_label.",),
         rows=("label_column", (COL,))),
    Spec("6a", PT_TAL, "subset_pcs", "Principal components of the subsets",
         ("variance_pc1_20: variance explained by PC1 to PC20. rho_pc_ columns: Spearman correlation of a PC with "
          "ln(protein IDs).",)),
    Spec("6b", PT_TAL, "sub_scan", "Subclusters of the subsets per graph and resolution",
         ("cluster_sizes: cells per subcluster. Columns: see Contents (Subcluster columns). kruskal_p_ids: "
          "Kruskal-Wallis test of the proteins per cell across the subclusters.",)),
    Spec("6c", PT_TAL, "sub_de_summary", "Subclusters at resolution 0.2: increased proteins and atlas panel proteins",
         ("n_sig_up: proteins significantly increased in the subcluster against the other cells of the subset "
          "(the differential-expression rule of the annotation); top25: the first 25.",
          "_panel_sig_up columns: proteins of the atlas panel among the increased proteins.")),
    Spec("7a", PT_TAL, "power_detectable", "Smallest detectable PCT versus PST difference",
         ("Smallest AUC and Hedges' g detectable with 80% power at a two-sided alpha of 0.05 (Noether's "
          "approximation), next to the observed values of the atlas S1/S2 - S3 score.",),
         rows=("grouping", ("kidney_annotation labels",))),
    Spec("7b", PT_TAL, "power_unresolved", "Subtypes and markers not resolved",
         ("One row per item, with the observed facts.",), special="unresolved"),
    Spec("8a", RAT_SEGMENTS, "coverage_summary", "Matching of the rat proteins to the kidney SCP-MS proteins",
         ("Rat protein groups of the Kidney Tubule Expression Atlas (KTEA), matched by identical gene symbol.",),
         special="coverage"),
    Spec("8b", RAT_SEGMENTS, "rule_diagnostics", "Rat panel candidates under each selection rule",
         ("Per contrast, value scale, rule and threshold: entries quantified in every segment of the contrast "
          "(tested), passing the rule (rat_pass), of these matched to a kidney protein (matched), and of these "
          "detected in >= 10% of the SCP-MS cells of the segment (scp_10pct).",
          "Adopted: values centred per sample, all samples separated. The planned rule (Welch test, BH q < 0.05, "
          "copies per cell) selected no protein.")),
    Spec("8c", RAT_SEGMENTS, "panel_funnel", "Rat panels: proteins at each filter step",
         ("Adopted rule; see Contents (Rat panel rule). n_in_atlas_panel: panel proteins also in the matching atlas "
          "panel.",
          "group_only_abundant: entries quantified in the group segments only (not tested) that pass the abundance "
          "filter; _matched: of these, matched to a kidney protein."),
         rows=("label_column", (COL,))),
    Spec("8d", RAT_SEGMENTS, "panels", "Rat panels: proteins",
         ("Panel proteins in rank order, with the rat values (log2, centred per sample) and the detected fraction "
          "in the SCP-MS cells of the segment (scp_detection).",
          "p_welch_max, q_bh: largest Welch p against a reference segment and its BH correction; for information, "
          "not used by the adopted rule."),
         rows=("label_column", (COL,))),
    Spec("9a", RAT_SEGMENTS, "scores_pct_vs_pst", "PCT versus PST cells on the rat segment scores",
         ("As table 4a, for the rat panels at both thresholds; primary rows: threshold fc2, gene_set all, method "
          "mean_z.",
          "Reference rows (threshold '-'): the atlas S1/S2 - S3 score of table 4a and ln(protein IDs)."),
         rows=("grouping", ("t15",))),
    Spec("9b", RAT_SEGMENTS, "scores_tal", "Rat TAL scores in the TAL cells",
         ("As table 4b. rho_with_other_tal_score: Spearman correlation between the cTAL and the mTAL score.",),
         rows=("label_column", (COL,))),
    Spec("9c", RAT_SEGMENTS, "map_pt_agreement", "Mapping to the rat segments: PCT and PST cells nearest to the expected "
         "segments",
         ("As table 5b. Expected: PCT cells nearest S1 or S2, PST cells nearest S3.",),
         rows=("grouping", ("t15",))),
    Spec("9d", RAT_SEGMENTS, "map_confusion", "Mapping to the rat segments: cells per SCP-MS cell type and segment",
         ("As table 5c. variant: threshold | panels | values.",), rows=("label_column", (COL,))),
    Spec("9e", RAT_SEGMENTS, "pseudobulk_pt_ratio", "Pseudobulk: PCT - PST difference against the rat segment differences",
         ("Spearman correlation (rho) over n_proteins proteins, with the permutation reference (null_ columns). "
          "Columns: see Contents (Pseudobulk columns).",
          "frac_null_ge_observed: share of the permutations with a correlation at or above the observed one."),
         rows=("label_column", (COL,))),
    Spec("9f", RAT_SEGMENTS, "pseudobulk_matrix", "Pseudobulk: SCP-MS cell types against the rat segments",
         ("Spearman correlation of the mean protein profile of each SCP-MS cell type with each rat segment. values: "
          "raw, or z (each protein z-scored across the groups of its data set).",),
         rows=("label_column", (COL,))),
    Spec("10a", RAT_SEGMENTS, "compare_panel_overlap", "Proteins shared by the rat panels and the matching atlas panels",
         ("n_rat, n_atlas: panel sizes. jaccard: shared proteins / proteins in either panel.",),
         rows=("label_column", (COL,))),
    Spec("10b", RAT_SEGMENTS, "compare_direction_summary", "Direction of the panel proteins in the other reference",
         ("rat panel -> atlas: rat panel proteins with a higher atlas mean in the matching atlas contrast. atlas "
          "panel -> rat: atlas panel proteins with a higher rat segment mean in the matching rat contrast.",
          "n_panel_in_other: panel proteins tested in the other reference. frac_agree: n_agree / n_panel_in_other."),
         rows=("label_column", (COL,))),
    Spec("10c", RAT_SEGMENTS, "compare_reference_rho", "Rat differences against atlas differences over all shared genes",
         ("Spearman correlation of the rat difference of segment means with the atlas log2 fold change and with "
          "the atlas difference of means, over the genes tested in both references.",)),
    Spec("10d", RAT_SEGMENTS, "compare_scores", "Per-cell agreement of the rat and the atlas S1/S2 - S3 scores",
         ("Spearman correlation between the two scores (rat fc2 panels; method mean_z; all panel proteins).",),
         rows=("panels_from", (COL,)),
         values={"cells": {"PT union cells": "PT subset", "PCT + PST cells (kidney_annotation labels)": "PCT + PST cells"}}),
    Spec("10e", RAT_SEGMENTS, "compare_s2", "Rat S2 panel proteins in the atlas PTS2 contrast",
         ("atlas_pts2_ columns: PTS2 against the other proximal tubule subtypes in the atlas. "
          "atlas_pts2_passes_rna_filters: passes the atlas panel filters on the RNA side.",),
         rows=("label_column", (COL,))),
]

SCP_GROUPS = {"t15: PCT": "PCT", "t15: PST": "PST", "t15: TAL": "TAL", "union PT subset": "PT subset",
              "union TAL subset": "TAL subset"}
UNRESOLVED_DROP = ("PCT / PST label sets", "Cells with the same PT label in both sets")
UNRESOLVED_EDITS = {      # item -> column -> [(pattern, replacement)]; every pattern must match exactly once
    "MTAL panel": {"status": [(r"^one protein \(kidney_annotation\) / none \(Fig 2A\)$", "one protein")],
                   "observed": [(r"1 under kidney_annotation \(Cldn10\), 0 under Fig 2A", "1 (Cldn10)")]},
    "CTAL vs MTAL in SCP": {"observed": [
        (r"TAL is one label in both sets \(kidney_annotation (\d+), Fig 2A \d+ cells\)", r"TAL is one cell type (\1 cells)"),
        (r"\(primary graph, r ([\d.]+)\)", r"(primary graph, resolution \1)")]},
    "MD": {"observed": [
        (r"Nos1 detected in \d+ of \d+ Fig 2A MD cells and (\d+ of \d+ cells) overall", r"Nos1 detected in \1"),
        (r"; the \d+ Fig 2A MD cells are kidney_annotation Endothelial", "")]},
    "PT subclusters": {"observed": [
        (r"^primary graph r ([\d.]+):", r"primary graph, resolution \1:"),
        (r"ARI with kidney_annotation / Fig 2A labels ([\d.]+) / [\d.]+", r"ARI with the cell types \1")]},
}

TERMS = [
    ("SCP-MS cell types", "Cell types of the 712 kidney SCP-MS cells as in Figure 2A (among them PCT 160, PST 158 and "
     "TAL 138 cells). Unknown: cells without an assigned cell type."),
    ("PT subset; TAL subset", "PT subset (356 cells): the PCT and PST cells and 38 unassigned cells that had a "
     "proximal tubule label in the original clustering. TAL subset (154 cells): the TAL cells and 16 cells with a "
     "TAL label in the original clustering."),
    ("Atlas", "Mouse Kidney Atlas (single-cell and single-nucleus RNA; annotation level Celltype). Proximal tubule: "
     "PTS1, PTS2, PTS3, PTS3T2. TAL: CTAL, MTAL. MD: macula densa (46 nuclei; descriptive only). S1/S2 = PTS1 + "
     "PTS2; S3 = PTS3 + PTS3T2."),
    ("Rat; KTEA", "Protein copies per cell of microdissected rat tubule segments (S1, S2, S3, cTAL, mTAL; Limbutara "
     "et al. 2020), from the Kidney Tubule Expression Atlas (KTEA). Values are log2 and centred per sample."),
    ("ln_ids; ln(protein IDs)", "Natural logarithm of the number of proteins detected in a cell."),
    ("detection; detected", "Fraction (or number) of cells with an intensity (protein) or a count (RNA) above zero."),
    ("rho", "Spearman correlation."),
    ("kind", "Panel contrast. one_vs_rest: a subtype or segment against the others of the same tubule segment. "
     "grouped: S1/S2 against S3, or the reverse. descriptive: MD against CTAL + MTAL."),
    ("Atlas panel filters", "Wilcoxon rank-sum test with Benjamini-Hochberg (BH) correction: q < 0.05; log2 fold "
     "change >= 1; detection in >= 25% of the subtype's cells; same direction in >= 2/3 of the study x cells/nuclei "
     "batches with >= 20 cells on both sides (frac_batches_same_sign, n_batches_eligible); protein detected in >= "
     "10% of the SCP-MS cells of the segment (proximal tubule: PCT + PST cells; TAL: TAL cells). Top 30 genes by "
     "Wilcoxon score."),
    ("threshold", "Rat panel threshold. fc2 (primary): the smallest group segment mean is >= 1 log2 unit above the "
     "largest reference segment mean. fc1.5 (sensitivity): >= 0.585 log2 units. '-': does not apply."),
    ("Rat panel rule", "A protein enters a panel if every segment of the contrast is quantified, that is detected "
     "in >= 2/3 of its samples (tested); each group segment mean is at least the 25th percentile of its segment "
     "(plus_abundance); the threshold is met (plus_fc); every quantified group sample is above every quantified "
     "reference sample (plus_separated); the protein is matched to a kidney SCP-MS protein (plus_matched) and "
     "detected in >= 10% of the SCP-MS cells of the segment (plus_scp_10pct). Top 30 proteins by difference."),
    ("log2fc_min_max; log2fc_mean", "Rat data. log2fc_min_max: smallest group segment mean minus largest reference "
     "segment mean. log2fc_mean: group mean minus reference mean."),
    ("method", "Score of a cell on a panel. mean_z (primary): mean of the scaled ln(1 + intensity) of the panel "
     "proteins. score_genes: Scanpy score_genes. detected_fraction: fraction of the panel proteins detected in the "
     "cell."),
    ("gene_set", "all: all panel proteins. no_annotation_markers: without the proteins of the annotation marker list "
     "(Supplementary Table 1). no_ribo_mito: without ribosomal and MitoCarta3.0 proteins."),
    ("score", "A name such as PTS1 or S3 is the score on that panel. 'S1/S2 - S3', 'PTS1 - PTS3', 'S1 - S3', 'CTAL - "
     "MTAL' and 'cTAL - mTAL' are differences of two scores."),
    ("PCT versus PST columns", "auc: probability that a PCT cell has a higher value than a PST cell; auc_ci_low and "
     "auc_ci_high: bootstrap 95% CI (1,000 resamples). cliffs_delta = 2 x auc - 1. hedges_g: Hedges' g. "
     "p_mannwhitney: Mann-Whitney test; q_bh_mannwhitney: BH correction over the scores of one gene set and method."),
    ("coef_ columns", "Linear regression of the score on the group (PCT against PST), without (unadjusted) and with "
     "ln_ids as covariate (adjusted_ln_ids). coef_sd_, ci_sd_low_, ci_sd_high_: coefficient and its 95% CI in SD of "
     "the score. coef_ln_ids, p_ln_ids: the covariate. rho_score_ln_ids: correlation of the score with ln_ids."),
    ("variant", "Mapping variant. Atlas: classes | genes | values. classes: primary = the subtypes of the segment; "
     "with_md = CTAL, MTAL and MD. genes: panels = the panel genes; all_rna_passing = all genes that pass the RNA "
     "filters. values: subset_z (primary) = SCP-MS values z-scored within the subset; canonical = values scaled "
     "over all 712 cells. Rat: threshold | panels | values."),
    ("Mapping columns", "A cell is nearest to the class with the highest cosine similarity. Confident: similarity "
     ">= 0.20 and >= 0.05 above the second class; the other cells are low-confidence (frac_low_confidence). "
     "n_informative_genes: genes used. median_margin: median difference between the two highest similarities."),
    ("graph", "primary: neighbour graph of the subset. regressed: after regressing out ln_ids."),
    ("Subcluster columns", "Leiden algorithm; resolutions 0.2, 0.3 and 0.5; seed 42; minimum cluster size 5. "
     "seed_ari_ columns: adjusted Rand index (ARI) of seeds 0 to 9 against seed 42. eta2_ln_ids: share of the "
     "variance of ln_ids explained by the subclusters. ari_cell_types: ARI of the subclusters with the SCP-MS cell "
     "types within the subset."),
    ("Pseudobulk columns", "scp_ratio: raw = PCT - PST difference of the mean ln(1 + intensity); depth_adjusted = PCT "
     "coefficient of a regression per protein with ln_ids as covariate. protein_set: all_shared = proteins "
     "quantified in rat S1, S2 and S3 and detected in >= 10% of the PCT or the PST cells; rat_variable = of these, "
     "proteins whose largest difference among S1, S2 and S3 is >= 1 log2 unit. null_ columns: 1,000 permutations of "
     "the PCT and PST labels within deciles of ln_ids. p_perm_two_sided: twice the smaller tail of the permutation "
     "distribution."),
]

# Numbers that the response to the reviewers, section 2, takes from these tables, at the precision written there.
# (text, table, value from the re-read workbook table, expected, decimals or None for an exact match)
def _pick(frame: pd.DataFrame, col: str, **sel):
    mask = pd.Series(True, index=frame.index)
    for key, value in sel.items():
        mask &= frame[key].astype(str).eq(str(value))
    hit = frame.loc[mask, col]
    if len(hit) != 1:
        raise LookupError(f"{len(hit)} rows for {sel}")
    return hit.iloc[0]


def _ratio(frame: pd.DataFrame, group: str) -> float:
    return float(_pick(frame, "n_nuclei", group=group)) / float(_pick(frame, "n_cells", group=group))


def _ntop(frame: pd.DataFrame, variant: str, segment: str, label: str, classes: tuple[str, ...]) -> int:
    return int(sum(_pick(frame, "n_top", variant=variant, segment=segment, label=label, **{"class": c})
                   for c in classes))


def _rho_range(frame: pd.DataFrame, scores: tuple[str, ...], **sel) -> str:
    values = [float(_pick(frame, "rho_score_ln_ids", score=s, **sel)) for s in scores]
    return f"{min(values):.2f}, {max(values):.2f}"


PRIMARY_MAP = "primary|panels|subset_z"
A_ALL = dict(gene_set="all", method="mean_z")
R_ALL = dict(threshold="fc2", gene_set="all", method="mean_z")
LETTER: list[tuple[str, str, Callable[[pd.DataFrame], object], object, int | None]] = [
    ("atlas S1/S2 - S3: AUC", "4a", lambda f: _pick(f, "auc", score="S1/S2 - S3", **A_ALL), 0.620, 3),
    ("atlas S1/S2 - S3: AUC, lower 95% bound", "4a", lambda f: _pick(f, "auc_ci_low", score="S1/S2 - S3", **A_ALL),
     0.556, 3),
    ("atlas S1/S2 - S3: AUC, upper 95% bound", "4a", lambda f: _pick(f, "auc_ci_high", score="S1/S2 - S3", **A_ALL),
     0.679, 3),
    ("atlas S1/S2 - S3: p of the adjusted difference", "4a",
     lambda f: _pick(f, "p_adjusted_ln_ids", score="S1/S2 - S3", **A_ALL), 0.0014, 4),
    ("ln(protein IDs): p PCT vs PST", "4a", lambda f: _pick(f, "p_mannwhitney", score="ln(protein IDs)"), 0.50, 2),
    ("atlas single-panel scores (PTS1, PTS3, PTS3T2, S1/S2, S3): smallest and largest rho with ln(protein IDs)", "4a",
     lambda f: _rho_range(f, ("PTS1", "PTS3", "PTS3T2", "S1/S2", "S3"), **A_ALL), "0.54, 0.82", None),
    ("rat S1/S2 - S3: AUC, 3 decimals", "9a", lambda f: _pick(f, "auc", score="S1/S2 - S3", **R_ALL), 0.641, 3),
    ("rat S1/S2 - S3: AUC, lower 95% bound", "9a", lambda f: _pick(f, "auc_ci_low", score="S1/S2 - S3", **R_ALL),
     0.578, 3),
    ("rat S1/S2 - S3: AUC, upper 95% bound", "9a", lambda f: _pick(f, "auc_ci_high", score="S1/S2 - S3", **R_ALL),
     0.698, 3),
    ("rat S1/S2 - S3: p of the adjusted difference", "9a",
     lambda f: _pick(f, "p_adjusted_ln_ids", score="S1/S2 - S3", **R_ALL), 7.2e-07, 8),
    ("rat S1/S2 - S3 without annotation markers: AUC", "9a",
     lambda f: _pick(f, "auc", threshold="fc2", score="S1/S2 - S3", gene_set="no_annotation_markers",
                     method="mean_z"), 0.638, 3),
    ("rat S1/S2 - S3 without ribosomal and MitoCarta proteins: AUC", "9a",
     lambda f: _pick(f, "auc", threshold="fc2", score="S1/S2 - S3", gene_set="no_ribo_mito", method="mean_z"),
     0.631, 3),
    ("rat single-panel scores (S1, S2, S3, S1/S2): smallest and largest rho with ln(protein IDs)", "9a",
     lambda f: _rho_range(f, ("S1", "S2", "S3", "S1/S2"), **R_ALL), "0.86, 0.94", None),
    ("Slc5a12: detected in PCT", "3a", lambda f: _pick(f, "scp_detection_pct", gene="Slc5a12"), 0.23, 2),
    ("Slc5a12: detected in PST", "3a", lambda f: _pick(f, "scp_detection_pst", gene="Slc5a12"), 0.11, 2),
    ("Slc22a8: detected in PCT", "3a", lambda f: _pick(f, "scp_detection_pct", gene="Slc22a8"), 0.59, 2),
    ("Slc22a8: detected in PST", "3a", lambda f: _pick(f, "scp_detection_pst", gene="Slc22a8"), 0.40, 2),
    ("Slc22a6: detected in PCT", "3a", lambda f: _pick(f, "scp_detection_pct", gene="Slc22a6"), 0.45, 2),
    ("Slc22a6: detected in PST", "3a", lambda f: _pick(f, "scp_detection_pst", gene="Slc22a6"), 0.44, 2),
    ("Atp11a: detected in PCT", "3a", lambda f: _pick(f, "scp_detection_pct", gene="Atp11a"), 0.11, 2),
    ("Atp11a: detected in PST", "3a", lambda f: _pick(f, "scp_detection_pst", gene="Atp11a"), 0.15, 2),
    ("Slc22a7, Cldn10, Cldn19: cells detected, of 712", "3a",
     lambda f: ", ".join(str(int(_pick(f, "n_cells_detected_all_712", gene=g)))
                         for g in ("Slc22a7", "Cldn10", "Cldn19")), "14, 16, 22", None),
    ("atlas S1/S2 - S3: AUC", "4a", lambda f: _pick(f, "auc", score="S1/S2 - S3", **A_ALL), 0.62, 2),
    ("atlas S1/S2 - S3: adjusted difference (SD)", "4a",
     lambda f: _pick(f, "coef_sd_adjusted_ln_ids", score="S1/S2 - S3", **A_ALL), 0.36, 2),
    ("atlas S1/S2 - S3: rho with ln(protein IDs)", "4a",
     lambda f: _pick(f, "rho_score_ln_ids", score="S1/S2 - S3", **A_ALL), 0.12, 2),
    ("ln(protein IDs): AUC PCT vs PST", "4a", lambda f: _pick(f, "auc", score="ln(protein IDs)"), 0.52, 2),
    ("atlas S1/S2 - S3 without annotation markers: AUC", "4a",
     lambda f: _pick(f, "auc", score="S1/S2 - S3", gene_set="no_annotation_markers", method="mean_z"), 0.586, 3),
    ("atlas S1/S2 - S3 without annotation markers: adjusted difference (SD)", "4a",
     lambda f: _pick(f, "coef_sd_adjusted_ln_ids", score="S1/S2 - S3", gene_set="no_annotation_markers",
                     method="mean_z"), 0.22, 2),
    ("atlas S1/S2 - S3 without ribosomal and MitoCarta proteins: AUC", "4a",
     lambda f: _pick(f, "auc", score="S1/S2 - S3", gene_set="no_ribo_mito", method="mean_z"), 0.655, 3),
    ("atlas S1/S2 - S3 without ribosomal and MitoCarta proteins: adjusted difference (SD)", "4a",
     lambda f: _pick(f, "coef_sd_adjusted_ln_ids", score="S1/S2 - S3", gene_set="no_ribo_mito", method="mean_z"),
     0.51, 2),
    ("atlas PTS1 - PTS3: AUC", "4a", lambda f: _pick(f, "auc", score="PTS1 - PTS3", **A_ALL), 0.659, 3),
    ("atlas PTS1 - PTS3: adjusted difference (SD)", "4a",
     lambda f: _pick(f, "coef_sd_adjusted_ln_ids", score="PTS1 - PTS3", **A_ALL), 0.55, 2),
    ("rat S1/S2 - S3: AUC", "9a", lambda f: _pick(f, "auc", score="S1/S2 - S3", **R_ALL), 0.64, 2),
    ("rat S1/S2 - S3: adjusted difference (SD)", "9a",
     lambda f: _pick(f, "coef_sd_adjusted_ln_ids", score="S1/S2 - S3", **R_ALL), 0.54, 2),
    ("rat S1/S2 - S3: rho with ln(protein IDs)", "9a",
     lambda f: _pick(f, "rho_score_ln_ids", score="S1/S2 - S3", **R_ALL), -0.15, 2),
    ("rat S2: AUC", "9a", lambda f: _pick(f, "auc", score="S2", **R_ALL), 0.558, 3),
    ("rat S2: rho with ln(protein IDs)", "9a", lambda f: _pick(f, "rho_score_ln_ids", score="S2", **R_ALL), 0.867, 3),
    ("atlas reference row in the rat table: AUC", "9a", lambda f: _pick(f, "auc", score="atlas S1/S2 - S3"), 0.62, 2),
    ("mapping: PCT nearest PTS1 or PTS2 (fraction)", "5b", lambda f: _pick(f, "frac_top_expected", label="PCT"),
     0.61, 2),
    ("mapping: PST nearest PTS3 or PTS3T2 (fraction)", "5b", lambda f: _pick(f, "frac_top_expected", label="PST"),
     0.54, 2),
    ("mapping: PCT confident in the expected class", "5b",
     lambda f: _pick(f, "frac_top_expected_confident", label="PCT"), 0.49, 2),
    ("mapping: PST confident in the expected class", "5b",
     lambda f: _pick(f, "frac_top_expected_confident", label="PST"), 0.42, 2),
    ("mapping: PCT low-confidence", "5b", lambda f: _pick(f, "frac_low_confidence", label="PCT"), 0.19, 2),
    ("mapping: PST low-confidence", "5b", lambda f: _pick(f, "frac_low_confidence", label="PST"), 0.20, 2),
    ("mapping: PCT cells nearest PTS1 or PTS2", "5c",
     lambda f: _ntop(f, PRIMARY_MAP, "PT", "PCT", ("PTS1", "PTS2")), 97, None),
    ("mapping: PST cells nearest PTS3 or PTS3T2", "5c",
     lambda f: _ntop(f, PRIMARY_MAP, "PT", "PST", ("PTS3", "PTS3T2")), 86, None),
    ("mapping: Unknown PT cells nearest PTS2", "5c", lambda f: _ntop(f, PRIMARY_MAP, "PT", "Unknown", ("PTS2",)),
     37, None),
    ("mapping: TAL cells nearest CTAL", "5c", lambda f: _ntop(f, PRIMARY_MAP, "TAL", "TAL", ("CTAL",)), 60, None),
    ("mapping: TAL cells nearest MTAL", "5c", lambda f: _ntop(f, PRIMARY_MAP, "TAL", "TAL", ("MTAL",)), 78, None),
    ("mapping: informative genes, PT", "5a",
     lambda f: _pick(f, "n_informative_genes", segment="PT", variant=PRIMARY_MAP), 65, None),
    ("mapping: informative genes, TAL", "5a",
     lambda f: _pick(f, "n_informative_genes", segment="TAL", variant=PRIMARY_MAP), 6, None),
    ("pseudobulk raw, rat S1/S2 - S3: rho", "9e",
     lambda f: _pick(f, "rho", protein_set="all_shared", scp_ratio="raw", rat_ratio="S1/S2 - S3"), 0.226, 3),
    ("pseudobulk raw: proteins", "9e",
     lambda f: _pick(f, "n_proteins", protein_set="all_shared", scp_ratio="raw", rat_ratio="S1/S2 - S3"), 1417, None),
    ("pseudobulk raw: permutation 2.5%", "9e",
     lambda f: _pick(f, "null_q025", protein_set="all_shared", scp_ratio="raw", rat_ratio="S1/S2 - S3"), 0.028, 3),
    ("pseudobulk raw: permutation 97.5%", "9e",
     lambda f: _pick(f, "null_q975", protein_set="all_shared", scp_ratio="raw", rat_ratio="S1/S2 - S3"), 0.204, 3),
    ("pseudobulk raw: permutation p", "9e",
     lambda f: _pick(f, "p_perm_two_sided", protein_set="all_shared", scp_ratio="raw", rat_ratio="S1/S2 - S3"),
     0.008, 3),
    ("pseudobulk depth-adjusted: rho", "9e",
     lambda f: _pick(f, "rho", protein_set="all_shared", scp_ratio="depth_adjusted", rat_ratio="S1/S2 - S3"),
     0.225, 3),
    ("pseudobulk depth-adjusted: permutation 2.5%", "9e",
     lambda f: _pick(f, "null_q025", protein_set="all_shared", scp_ratio="depth_adjusted", rat_ratio="S1/S2 - S3"),
     0.058, 3),
    ("pseudobulk depth-adjusted: permutation 97.5%", "9e",
     lambda f: _pick(f, "null_q975", protein_set="all_shared", scp_ratio="depth_adjusted", rat_ratio="S1/S2 - S3"),
     0.212, 3),
    ("pseudobulk depth-adjusted: permutation p", "9e",
     lambda f: _pick(f, "p_perm_two_sided", protein_set="all_shared", scp_ratio="depth_adjusted",
                     rat_ratio="S1/S2 - S3"), 0.014, 3),
    ("pseudobulk matrix: PCT with rat S2", "9f",
     lambda f: _pick(f, "rho", values="z", scp_group="PCT", rat_segment="S2"), 0.23, 2),
    ("pseudobulk matrix: PCT with rat S1", "9f",
     lambda f: _pick(f, "rho", values="z", scp_group="PCT", rat_segment="S1"), 0.16, 2),
    ("pseudobulk matrix: PST with rat S3", "9f",
     lambda f: _pick(f, "rho", values="z", scp_group="PST", rat_segment="S3"), 0.25, 2),
    ("pseudobulk matrix: TAL with rat mTAL", "9f",
     lambda f: _pick(f, "rho", values="z", scp_group="TAL", rat_segment="mTAL"), 0.48, 2),
    ("pseudobulk matrix: TAL with rat cTAL", "9f",
     lambda f: _pick(f, "rho", values="z", scp_group="TAL", rat_segment="cTAL"), 0.42, 2),
    ("pseudobulk matrix: proteins", "9f",
     lambda f: _pick(f, "n_proteins", values="z", scp_group="PCT", rat_segment="S2"), 1351, None),
    ("Slc5a2: AUC PCT vs PST", "3a", lambda f: _pick(f, "scp_auc_pct_vs_pst", gene="Slc5a2"), 0.66, 2),
    ("Slc5a2: detected in PCT", "3a", lambda f: _pick(f, "scp_detection_pct", gene="Slc5a2"), 0.74, 2),
    ("Slc5a2: detected in PST", "3a", lambda f: _pick(f, "scp_detection_pst", gene="Slc5a2"), 0.50, 2),
    ("Slc5a12: AUC PCT vs PST", "3a", lambda f: _pick(f, "scp_auc_pct_vs_pst", gene="Slc5a12"), 0.56, 2),
    ("Slc22a8: AUC PCT vs PST", "3a", lambda f: _pick(f, "scp_auc_pct_vs_pst", gene="Slc22a8"), 0.60, 2),
    ("Slc22a6: AUC PCT vs PST", "3a", lambda f: _pick(f, "scp_auc_pct_vs_pst", gene="Slc22a6"), 0.50, 2),
    ("Atp11a: AUC PCT vs PST", "3a", lambda f: _pick(f, "scp_auc_pct_vs_pst", gene="Atp11a"), 0.48, 2),
    ("Slc22a7: AUC PCT vs PST", "3a", lambda f: _pick(f, "scp_auc_pct_vs_pst", gene="Slc22a7"), 0.50, 2),
    ("Slc7a13 quantified as protein", "3a", lambda f: bool(_pick(f, "quantified_protein", gene="Slc7a13")),
     False, None),
    ("Cldn16 quantified as protein", "3a", lambda f: bool(_pick(f, "quantified_protein", gene="Cldn16")),
     False, None),
    ("Ptgs2 quantified as protein", "3a", lambda f: bool(_pick(f, "quantified_protein", gene="Ptgs2")), False, None),
    ("Nos1: cells detected, of 712", "3a", lambda f: _pick(f, "n_cells_detected_all_712", gene="Nos1"), 2, None),
    ("Cldn10: TAL cells detected, of 138", "3b", lambda f: _pick(f, "n_detected", gene="Cldn10", scp_group="TAL"),
     15, None),
    ("Cldn19: TAL cells detected, of 138", "3b", lambda f: _pick(f, "n_detected", gene="Cldn19", scp_group="TAL"),
     20, None),
    ("rat Slc5a2: log2 difference", "3c", lambda f: _pick(f, "log2fc_mean", gene="Slc5a2"), 8.04, 2),
    ("rat Atp11a: log2 difference", "3c", lambda f: _pick(f, "log2fc_mean", gene="Atp11a"), 2.78, 2),
    ("rat Slc22a6: log2 difference", "3c", lambda f: _pick(f, "log2fc_mean", gene="Slc22a6"), 3.41, 2),
    ("rat Cldn10: log2 difference", "3c", lambda f: _pick(f, "log2fc_mean", gene="Cldn10"), -2.98, 2),
    ("rat Cldn16: log2 difference", "3c", lambda f: _pick(f, "log2fc_mean", gene="Cldn16"), 1.22, 2),
    ("rat Cldn19: log2 difference", "3c", lambda f: _pick(f, "log2fc_mean", gene="Cldn19"), 2.25, 2),
    ("PT subclusters: eta2 of ln(protein IDs)", "6b",
     lambda f: _pick(f, "eta2_ln_ids", segment="PT", graph="primary", resolution=0.2), 0.80, 2),
    ("PT subclusters: ARI with the cell types", "6b",
     lambda f: _pick(f, "ari_cell_types", segment="PT", graph="primary", resolution=0.2), 0.152, 3),
    ("PT subclusters, depth regressed: ARI", "6b",
     lambda f: _pick(f, "ari_cell_types", segment="PT", graph="regressed", resolution=0.2), 0.065, 3),
    ("TAL subclusters: eta2 of ln(protein IDs)", "6b",
     lambda f: _pick(f, "eta2_ln_ids", segment="TAL", graph="primary", resolution=0.2), 0.78, 2),
    ("TAL subclusters: ARI with the cell types", "6b",
     lambda f: _pick(f, "ari_cell_types", segment="TAL", graph="primary", resolution=0.2), 0.02, 2),
    ("subcluster PT-s3: cells", "6c", lambda f: _pick(f, "n_cells", graph="primary", subcluster="PT-s3"), 45, None),
    ("subcluster PT-s3: S3-panel proteins increased", "6c",
     lambda f: _pick(f, "S3_panel_sig_up", graph="primary", subcluster="PT-s3"), "11 of 30", None),
    ("subcluster PT-s3: PTS1-panel proteins increased", "6c",
     lambda f: _pick(f, "PTS1_panel_sig_up", graph="primary", subcluster="PT-s3"), "0 of 16", None),
    ("atlas CTAL score: rho with ln(protein IDs)", "4b", lambda f: _pick(f, "rho_ln_ids_tal", score="CTAL", **A_ALL),
     0.833, 3),
    ("atlas MD score in TAL cells: rho with ln(protein IDs)", "4b",
     lambda f: _pick(f, "rho_ln_ids_tal", score="MD", **A_ALL), 0.95, 2),
    ("rat cTAL score: rho with ln(protein IDs)", "9b", lambda f: _pick(f, "rho_ln_ids_tal", score="cTAL", **R_ALL),
     0.941, 3),
    ("rat mTAL score: rho with ln(protein IDs)", "9b", lambda f: _pick(f, "rho_ln_ids_tal", score="mTAL", **R_ALL),
     0.925, 3),
    ("rat cTAL vs mTAL scores: rho", "9b",
     lambda f: _pick(f, "rho_with_other_tal_score", score="cTAL", **R_ALL), 0.86, 2),
    ("smallest detectable AUC, 160 vs 158 cells", "7a", lambda f: f["min_detectable_auc_80pct_power"].iloc[0],
     0.591, 3),
    ("cells compared: PCT", "7a", lambda f: f["n_1"].iloc[0], 160, None),
    ("cells compared: PST", "7a", lambda f: f["n_2"].iloc[0], 158, None),
    ("atlas S2 marker: unresolved row", "7b",
     lambda f: "1 also with log2 FC >= 1 and detection >= 25%; 0 after the batch-sign rule"
     in str(_pick(f, "observed", item="S2 (atlas PTS2) panel")), True, None),
    ("atlas MD: nuclei", "1a", lambda f: _pick(f, "n_nuclei", group="MD"), 46, None),
    ("atlas MD: studies", "1a", lambda f: _pick(f, "n_studies", group="MD"), 2, None),
    ("atlas PTS3T2: fraction nuclei", "1a", lambda f: _ratio(f, "PTS3T2"), 0.93, 2),
    ("atlas CTAL: fraction nuclei", "1a", lambda f: _ratio(f, "CTAL"), 0.76, 2),
    ("atlas MTAL: fraction nuclei", "1a", lambda f: _ratio(f, "MTAL"), 0.74, 2),
    ("atlas MD: fraction nuclei", "1a", lambda f: _ratio(f, "MD"), 1.00, 2),
    ("atlas panel sizes PTS1, PTS2, PTS3, PTS3T2, S1/S2, S3, CTAL, MTAL, MD", "2a",
     lambda f: ", ".join(str(int(_pick(f, "n_in_panel", panel=p)))
                         for p in ("PTS1", "PTS2", "PTS3", "PTS3T2", "S1/S2", "S3", "CTAL", "MTAL", "MD")),
     "16, 0, 22, 30, 12, 30, 5, 1, 23", None),
    ("atlas CTAL panel proteins", "2b",
     lambda f: ", ".join(sorted(f.loc[f["panel"].eq("CTAL"), "protein"].astype(str))),
     "Atp6v0a4, Egf, Fkbp5, Gls, Slit2", None),
    ("atlas MTAL panel protein", "2b",
     lambda f: ", ".join(sorted(f.loc[f["panel"].eq("MTAL"), "protein"].astype(str))), "Cldn10", None),
    ("rat matching: kidney proteins matched", "8a", lambda f: _pick(f, "n", item="kidney proteins with a KTEA match"),
     3192, None),
    ("rat matching: kidney proteins", "8a", lambda f: _pick(f, "n", item="kidney proteins (matrix)"), 4054, None),
    ("rat matching: quantified in S1, S2 and S3", "8a",
     lambda f: _pick(f, "n", item="kidney proteins with a KTEA match quantified in S1, S2 and S3"), 1790, None),
    ("rat matching: quantified in cTAL and mTAL", "8a",
     lambda f: _pick(f, "n", item="kidney proteins with a KTEA match quantified in cTAL and mTAL"), 2274, None),
    ("rat panel sizes (fc2) S1, S2, S3, S1/S2, cTAL, mTAL", "8c",
     lambda f: ", ".join(str(int(_pick(f, "panel_size", threshold="fc2", panel=p)))
                         for p in ("S1", "S2", "S3", "S1/S2", "cTAL", "mTAL")), "30, 16, 30, 30, 30, 30", None),
    ("nearest rat segment: PCT nearest S1 or S2", "9c",
     lambda f: _pick(f, "frac_top_expected", threshold="fc2", label="PCT"), 0.78, 2),
    ("nearest rat segment: PST nearest S3", "9c",
     lambda f: _pick(f, "frac_top_expected", threshold="fc2", label="PST"), 0.43, 2),
    ("nearest rat segment: PCT low-confidence", "9c",
     lambda f: _pick(f, "frac_low_confidence", threshold="fc2", label="PCT"), 0.61, 2),
    ("nearest rat segment: PST low-confidence", "9c",
     lambda f: _pick(f, "frac_low_confidence", threshold="fc2", label="PST"), 0.67, 2),
    ("panel overlap rat S1 with atlas PTS1: shared proteins", "10a",
     lambda f: _pick(f, "n_shared", threshold="fc2", rat_panel="S1", atlas_panel="PTS1"), 6, None),
    ("panel overlap rat S1 with atlas PTS1: Jaccard", "10a",
     lambda f: _pick(f, "jaccard", threshold="fc2", rat_panel="S1", atlas_panel="PTS1"), 0.15, 2),
    ("largest Jaccard index of the fc2 panel pairs", "10a",
     lambda f: float(f.loc[f["threshold"].eq("fc2"), "jaccard"].max()), 0.15, 2),
    ("rat panel -> atlas (fc2) S1, S1/S2, S3, S2, cTAL, mTAL", "10b",
     lambda f: ", ".join(f"{float(_pick(f, 'frac_agree', direction='rat panel -> atlas', threshold='fc2', panel=p)):.2f}"
                         for p in ("S1", "S1/S2", "S3", "S2", "cTAL", "mTAL")),
     "0.73, 0.79, 0.59, 0.27, 0.70, 0.72", None),
    ("atlas panel -> rat PTS1, PTS3, S3, S1/S2", "10b",
     lambda f: ", ".join(f"{float(_pick(f, 'frac_agree', direction='atlas panel -> rat', panel=p)):.2f}"
                         for p in ("PTS1", "PTS3", "S3", "S1/S2")), "0.90, 0.71, 0.88, 0.43", None),
    ("atlas panel -> rat CTAL: agree of tested", "10b",
     lambda f: f"{int(_pick(f, 'n_agree', direction='atlas panel -> rat', panel='CTAL'))}/"
               f"{int(_pick(f, 'n_panel_in_other', direction='atlas panel -> rat', panel='CTAL'))}", "3/3", None),
    ("per-cell rho of the rat and atlas S1/S2 - S3 scores, PCT + PST cells", "10d",
     lambda f: _pick(f, "rho", cells="PCT + PST cells"), 0.65, 2),
    ("rat S2 panel (fc2): proteins, in the atlas table, higher in atlas PTS2, passing the atlas filters", "10e",
     lambda f: (lambda d: f"{len(d)}, {int(d['in_atlas'].astype(bool).sum())}, "
                          f"{int((pd.to_numeric(d['atlas_pts2_mean_difference']) > 0).sum())}, "
                          f"{int(d['atlas_pts2_passes_rna_filters'].fillna(False).astype(bool).sum())}")(
         f[f["threshold"].eq("fc2")]), "16, 15, 4, 0", None),
]

F = Font(name="Arial", size=10)
FB = Font(name="Arial", size=10, bold=True)
FT = Font(name="Arial", size=11, bold=True)
FN = Font(name="Arial", size=10, color="595959")
HEAD_FILL = PatternFill("solid", fgColor="EDEDED")
HEAD_BORDER = Border(bottom=Side(style="thin", color="7F7F7F"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-07-d, revised annotation only.")
    cc.add_common_args(parser)
    return parser.parse_args()


def check(checks: list[dict], name: str, observed, expected, ok: bool) -> None:
    checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})


# ---------------------------------------------------------------------------
# Assembly of one table
# ---------------------------------------------------------------------------
def rename_column(name: str) -> str:
    if name in RENAME_EXACT:
        return RENAME_EXACT[name]
    for pattern, new in RENAME_RE:
        name = pattern.sub(new, name)
    return name


def replace_value(value):
    if not isinstance(value, str):
        return value
    value = VALUE_EXACT.get(value, value)
    for pattern, new in VALUE_RE:
        value = pattern.sub(new, value)
    return value


def assemble_table(spec: Spec, src: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rows and columns of the revised annotation, with reader-facing names; the record says what was done."""
    df, problems, row_filter, edits = src.copy(), [], "all rows", []
    if spec.special == "scp_groups":
        df = df[df["scp_group"].isin(SCP_GROUPS)].copy()
        df["scp_group"] = df["scp_group"].map(SCP_GROUPS)
        row_filter = "scp_group in " + " / ".join(SCP_GROUPS)
        edits.append("scp_group: " + "; ".join(f"{k} -> {v}" for k, v in SCP_GROUPS.items()))
    elif spec.special == "coverage":
        keep, parent = [], ""
        for item in df["item"]:
            parent = parent if item.startswith(" ") else item
            keep.append("Fig 2A" not in parent)
        dropped = int(len(df) - sum(keep))
        if dropped != 4:
            problems.append(f"coverage: {dropped} rows of the other annotation instead of 4")
        df = df[keep].copy()
        df["item"] = df["item"].str.replace(" (kidney_annotation labels)", "", regex=False)
        row_filter = "items of the other annotation (Fig 2A labels) and their sub-items dropped"
        edits.append("item: ' (kidney_annotation labels)' removed")
    elif spec.special == "unresolved":
        missing = [i for i in (*UNRESOLVED_DROP, *UNRESOLVED_EDITS) if i not in set(df["item"])]
        if missing:
            problems.append(f"unresolved: items not found: {missing}")
        df = df[~df["item"].isin(UNRESOLVED_DROP)].copy()
        for item, by_col in UNRESOLVED_EDITS.items():
            for col, subs in by_col.items():
                for pattern, new in subs:
                    idx = df.index[df["item"].eq(item)]
                    old = str(df.loc[idx[0], col]) if len(idx) == 1 else ""
                    text, n = re.subn(pattern, new, old)
                    if n != 1:
                        problems.append(f"unresolved: pattern {pattern!r} matched {n} times in {item} / {col}")
                        continue
                    df.loc[idx[0], col] = text
                    edits.append(f"{item} / {col}: {pattern} -> {new}")
        row_filter = "items dropped: " + "; ".join(UNRESOLVED_DROP)
    elif spec.rows is not None:
        col, kept = spec.rows
        df = df[df[col].isin(kept)].copy()
        row_filter = f"{col} in {' / '.join(kept)}"
    id_dropped = []
    for col in ID_COLS:
        if col in df.columns and col not in spec.keep_id:
            if df[col].nunique(dropna=False) != 1:
                problems.append(f"{col} is not constant after the row filter")
                continue
            id_dropped.append(f"{col} = {df[col].iloc[0]}")
            df = df.drop(columns=col)
    other = [c for c in df.columns if OTHER_ANNOTATION.search(c)]
    df = df.drop(columns=other)
    for col, mapping in spec.values.items():
        unknown = sorted(set(df[col].astype(str)) - set(mapping))
        if unknown:
            problems.append(f"{col}: values without a display name: {unknown}")
        df[col] = df[col].map(lambda v: mapping.get(v, v))
        edits.append(f"{col}: " + "; ".join(f"{k} -> {v}" for k, v in mapping.items()))
    replaced = set()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            new = df[col].map(replace_value)
            changed = df[col].notna() & new.ne(df[col])
            replaced.update(zip(df.loc[changed, col], new[changed]))
            df[col] = new
    edits.extend(f"{a} -> {b}" for a, b in sorted(replaced))
    renamed = {c: rename_column(c) for c in df.columns if rename_column(c) != c}
    df = df.rename(columns=renamed)
    if df.columns.duplicated().any():
        problems.append(f"duplicate column names after renaming: {list(df.columns[df.columns.duplicated()])}")
    if df.empty:
        problems.append("no rows left")
    for col in df.columns:                      # whole-number columns that pandas read as float (empty cells)
        s = df[col]
        if pd.api.types.is_float_dtype(s) and s.notna().any():
            v = s.dropna()
            if bool((v == v.round()).all()) and (col.startswith("n_") or float(v.abs().max()) >= 2):
                df[col] = s.astype("Int64")
    record = {"sheet": spec.sheet, "table": spec.key, "title": spec.title,
              "source": f"results/tables/{spec.task}/{spec.task}_{spec.name}.csv",
              "rows_source": len(src), "rows_kept": len(df), "columns_source": src.shape[1],
              "columns_kept": df.shape[1], "row_filter": row_filter,
              "annotation_columns_dropped": "; ".join(id_dropped),
              "other_annotation_columns_dropped": "; ".join(other),
              "columns_renamed": "; ".join(f"{a} -> {b}" for a, b in renamed.items()),
              "values_changed": " | ".join(edits), "problems": " | ".join(problems)}
    return df.reset_index(drop=True), record


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------
def native(value):
    """A value as openpyxl writes it: None, bool, int, float or str."""
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def number_format(name: str, series: pd.Series) -> str | None:
    """Display format of a column; the stored values keep full precision."""
    if not pd.api.types.is_float_dtype(series) or not series.notna().any():
        return None
    v = series.dropna()
    if P_LIKE.search(name):
        return "0.00E+00"
    if bool((v == v.round(2)).all()):
        return None                              # e.g. resolution 0.2, median 674.5
    return "0.0" if float(v.abs().max()) >= 100 else "0.000"


def shown(value, fmt: str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and fmt:
        return f"{value:.2E}" if fmt == "0.00E+00" else f"{value:.1f}" if fmt == "0.0" else f"{value:.3f}"
    return str(value)


class Book:
    def __init__(self):
        self.wb = Workbook()
        self.wb.remove(self.wb.active)
        self.widths: dict[tuple[str, int], int] = {}
        self.positions: dict[str, dict] = {}

    def cell(self, ws, row: int, col: int, value, font=F, fmt: str | None = None, width: bool = True):
        c = ws.cell(row=row, column=col, value=value)
        if isinstance(value, str):
            c.data_type = "s"                    # never a formula, also for text that starts with "="
        c.font = font
        if fmt:
            c.number_format = fmt
        if width:
            key = (ws.title, col)
            self.widths[key] = max(self.widths.get(key, 0), len(shown(value, fmt)))
        return c

    def table(self, ws, row: int, spec: Spec, frame: pd.DataFrame) -> int:
        self.cell(ws, row, 1, f"Table {spec.key}. {spec.title}", FB, width=False)
        row += 1
        for line in spec.notes:
            self.cell(ws, row, 1, line, FN, width=False)
            row += 1
        for j, name in enumerate(frame.columns, 1):
            c = self.cell(ws, row, j, str(name), FB)
            c.fill, c.border = HEAD_FILL, HEAD_BORDER
        self.positions[spec.key] = {"sheet": ws.title, "header_row": row, "n_rows": len(frame),
                                    "n_cols": frame.shape[1]}
        formats = [number_format(str(name), frame[name]) for name in frame.columns]
        for i, values in enumerate(frame.itertuples(index=False, name=None), row + 1):
            for j, value in enumerate(values, 1):
                self.cell(ws, i, j, native(value), F, formats[j - 1])
        return row + len(frame) + 3              # two empty rows before the next table

    def set_widths(self, cap: int = 48) -> None:
        for (title, col), n in self.widths.items():
            width = n + 2 if title == "Contents" and col == 3 else min(max(n + 2, 9), cap)
            self.wb[title].column_dimensions[get_column_letter(col)].width = width


def write_workbook(path, frames: dict[str, pd.DataFrame]) -> Book:
    book = Book()
    ws = book.wb.create_sheet("Contents")
    book.cell(ws, 1, 1, TABLE_TITLE, FT, width=False)
    book.cell(ws, 2, 1, "SCP-MS: single-cell proteomics of 712 mouse kidney cells. All values are observed values; "
              "the terms used in the column names are explained below the list of tables.", FN, width=False)
    row = 4
    for j, name in enumerate(("sheet", "table", "content", "rows", "columns"), 1):
        c = book.cell(ws, row, j, name, FB, width=j != 3)
        c.fill, c.border = HEAD_FILL, HEAD_BORDER
    for spec in SPECS:
        row += 1
        for j, value in enumerate((SHEETS[spec.sheet][0], spec.key, spec.title, len(frames[spec.key]),
                                   frames[spec.key].shape[1]), 1):
            book.cell(ws, row, j, value, F, width=j != 3)
    row += 3
    for j, name in ((1, "term or column"), (3, "meaning")):
        c = book.cell(ws, row, j, name, FB, width=False)
        c.fill, c.border = HEAD_FILL, HEAD_BORDER
    ws.cell(row=row, column=2).fill = HEAD_FILL
    ws.cell(row=row, column=2).border = HEAD_BORDER
    for term, meaning in TERMS:                  # the term overflows into the empty column B; the meaning wraps
        row += 1
        book.cell(ws, row, 1, term, F, width=False).alignment = Alignment(vertical="top")
        book.cell(ws, row, 3, meaning, F, width=False).alignment = Alignment(wrap_text=True, vertical="top")
    book.widths[("Contents", 1)] = max(len(SHEETS[s][0]) for s in SHEETS)
    book.widths[("Contents", 3)] = 118
    for number, (name, title) in SHEETS.items():
        ws = book.wb.create_sheet(name)
        book.cell(ws, 1, 1, f"Supplementary Table S-07-d, sheet {number}. {title}", FT, width=False)
        book.cell(ws, 2, 1, "Terms and column names: see the sheet Contents.", FN, width=False)
        row = 4
        for spec in SPECS:
            if spec.sheet == number:
                row = book.table(ws, row, spec, frames[spec.key])
    book.set_widths()
    book.wb.properties.title = "Supplementary Table S-07-d"
    book.wb.properties.creator = cc.GENERATED_BY
    book.wb.properties.description = f"{cc.GENERATED_TAG}; scripts/report_pt_tal_tables.py"
    book.wb.save(path)
    return book


def read_back(path, positions: dict[str, dict]) -> tuple[dict[str, pd.DataFrame], dict]:
    """The tables as stored in the saved workbook, and counts over every cell with a value."""
    wb = load_workbook(path)
    frames = {}
    for key, pos in positions.items():
        ws = wb[pos["sheet"]]
        header = [ws.cell(row=pos["header_row"], column=j).value for j in range(1, pos["n_cols"] + 1)]
        rows = [[ws.cell(row=pos["header_row"] + i, column=j).value for j in range(1, pos["n_cols"] + 1)]
                for i in range(1, pos["n_rows"] + 1)]
        frames[key] = pd.DataFrame(rows, columns=header, dtype=object)       # cells as stored: None stays None
    stats = {"cells": 0, "not_arial": 0, "formulas": 0, "forbidden": []}
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if c.value is None:
                    continue
                stats["cells"] += 1
                stats["not_arial"] += c.font.name != "Arial"
                stats["formulas"] += c.data_type == "f"
                if isinstance(c.value, str) and FORBIDDEN.search(c.value):
                    stats["forbidden"].append(f"{ws.title}!{c.coordinate}: {c.value[:60]}")
    stats["sheets"] = wb.sheetnames
    return frames, stats


REL_TOL = 1e-15            # openpyxl stores numbers with 16 significant digits ("%.16g")


def rel_diff(a, b) -> float:
    """0 for equal cells, the relative difference for two numbers, infinity for any other mismatch."""
    if a is None or b is None:
        return 0.0 if a is None and b is None else float("inf")
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if isinstance(a, bool) and isinstance(b, bool) and a == b else float("inf")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return 0.0 if float(a) == float(b) else abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)))
    return 0.0 if a == b else float("inf")


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)
    checks: list[dict] = []

    frames, records = {}, []
    for spec in SPECS:
        path = run.use_input(cc.out_path(spec.task, "tables", spec.name, ".csv"), f"{spec.task} {spec.name}")
        frames[spec.key], record = assemble_table(spec, pd.read_csv(path))
        record["source_sha256"] = run.inputs[cc.rel(path)]["sha256"]
        records.append(record)
    sources = pd.DataFrame(records)

    is_t24 = sources["source"].str.contains("rat_segments")
    check(checks, "source tables read: 16 pt_tal tables (sheets 1-7) and 16 rat_segments tables (sheets 8-10, and table 3c)",
          f"{int((~is_t24).sum())} pt_tal, {int(is_t24.sum())} rat_segments; sheets {sources['sheet'].nunique()}",
          "16 pt_tal, 16 rat_segments; sheets 10",
          int((~is_t24).sum()) == 16 and int(is_t24.sum()) == 16 and sources["sheet"].nunique() == 10
          and sources["source"].is_unique
          and bool((is_t24 == (sources["sheet"].ge(8) | sources["table"].eq("3c"))).all()))
    problems = sources.loc[sources["problems"].ne(""), ["table", "problems"]]
    check(checks, "assembly: row filters, constant annotation columns, display names, free-text edits, unique "
          "column names, no empty table", "; ".join(f"{r.table}: {r.problems}" for r in problems.itertuples()) or
          "no problem in 32 tables", "no problem", problems.empty)
    left = {k: [c for c in f.columns if OTHER_ANNOTATION.search(str(c)) or c in ID_COLS and c != "grouping"]
            for k, f in frames.items()}
    check(checks, "no column of another annotation and no annotation-name column left",
          {k: v for k, v in left.items() if v} or "none", "none", not any(left.values()))
    bad_labels = sorted({str(v) for f in frames.values() for c in DISPLAY_LABEL_COLS if c in f.columns
                         for v in f[c].dropna() if str(v) in ("DCT1", "PC")})
    check(checks, "cell-type names in the label columns need no display name (no DCT1 or PC row)",
          bad_labels or "none", "none", not bad_labels)

    out = cc._check_writable(cc.OUT_ROOTS["tables"] / TASK / WORKBOOK, run.overwrite)
    book = write_workbook(out, frames)
    run.outputs.append(cc.rel(out))
    print(f"Saved workbook: {cc.rel(out)}")

    stored, stats = read_back(out, book.positions)
    check(checks, "workbook sheets", ", ".join(stats["sheets"]), "Contents, then sheets 1 to 10",
          stats["sheets"] == ["Contents", *[SHEETS[s][0] for s in sorted(SHEETS)]])
    differ, largest = [], 0.0
    for key, frame in frames.items():
        got = stored[key]
        if list(got.columns) != [str(c) for c in frame.columns] or got.shape != frame.shape:
            differ.append(f"{key}: shape or header")
            continue
        diffs = [rel_diff(native(a), b) for row_a, row_b in zip(frame.itertuples(index=False, name=None),
                                                                 got.itertuples(index=False, name=None))
                 for a, b in zip(row_a, row_b)]
        n = sum(d > REL_TOL for d in diffs)
        largest = max([largest, *[d for d in diffs if d <= REL_TOL]])
        if n:
            differ.append(f"{key}: {n} cells")
    check(checks, "round trip: every table read back from the saved workbook equals the assembled table (header, "
          "shape, text and whole numbers exactly, other numbers to the 16 significant digits a workbook stores)",
          "; ".join(differ) or f"{len(frames)} tables, {int(sum(f.size for f in frames.values()))} values equal; "
          f"largest relative difference {largest:.1e}", f"32 tables equal; relative difference <= {REL_TOL:.0e}",
          not differ and len(frames) == 32)
    check(checks, "workbook cells: Arial, no formula", f"{stats['cells']} cells with a value; not Arial "
          f"{stats['not_arial']}; formulas {stats['formulas']}", "not Arial 0; formulas 0",
          stats["not_arial"] == 0 and stats["formulas"] == 0)
    check(checks, "no internal name in any cell of the workbook (task codes, other annotation, marker-list "
          "version, plan)", "; ".join(stats["forbidden"][:12]) or "none", "none", not stats["forbidden"])

    numbers = []
    for text, key, getter, expected, decimals in LETTER:
        try:
            value = getter(stored[key])
            value = value.item() if isinstance(value, np.generic) else value
            ok = (value == expected if decimals is None
                  else abs(float(value) - float(expected)) <= 0.5 * 10 ** -decimals + 1e-9)
        except Exception as error:                # a missing row or column is a failed check, not a crash
            value, ok = f"error: {error}", False
        numbers.append({"number": text, "table": key, "workbook_value": value, "thread_07_value": expected,
                        "decimals": "" if decimals is None else decimals, "match": bool(ok)})
    numbers = pd.DataFrame(numbers)
    wrong = numbers.loc[~numbers["match"], "number"].tolist()
    check(checks, "numbers of the response to the reviewers, section 2, found in the saved workbook at the written precision",
          f"{int(numbers['match'].sum())} of {len(numbers)}" + (f"; not matched: {wrong}" if wrong else ""),
          f"{len(numbers)} of {len(numbers)}", not wrong)

    for c in checks:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['observed']}")
    run.table(sources, f"{STEP}_sources")
    run.table(numbers, f"{STEP}_letter_numbers")
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    all_ok = all(c["pass"] for c in checks)
    run.finish({"label_column": COL, "workbook": cc.rel(out), "tables": len(frames),
                "rows": int(sum(len(f) for f in frames.values())),
                "style": "Contents sheet and ten data sheets; Arial 10; values only; 16 significant digits with a display "
                         "format", "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit("A check failed; see report_07_s07d_table_checks.csv.")


if __name__ == "__main__":
    main()
