#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S14 (RNA comparator for the DCT data), one workbook.

Editors' comment 1 (no scRNA-seq comparator for the DCT
cells); plan approved by the authors on 12 Nov 2026 ("OK as proposed"). One workbook with seven sheets, built from 11
stored tables of dct_rna (RNA comparator, combined control + zero-K search), one of atlas_dct (the split of the atlas DCT
cells) and the value tables of the three the response to the reviewers figures. No new statistics: every value is copied from the
stored tables. Layout and assembly code as Supplementary Tables S7, S8 and S11
(scripts/report_rna_protein_tables.py: build, Spec, Rules, Layout).

What the assembly changes (all recorded per table in the sources table):
- conditions: Ctrl -> Control, ZeroK -> Zero-K, both -> atlas (rows that hold atlas values only). Control and zero-K
  were analysed separately; they are listed one after the other and nothing compares them;
- labels: the primary label column (cs_ctrl_deg_label / cs_zerok_pipeline_label) is the "cluster annotation"; its
  value Unknown stays in the cells and the sheets carry the note "Unknown = DCT, subtype not resolved (cells of
  clusters without marker support)". The second label column (cs_<cond>_cluster_label) is the "marker-score label
  (clusters)"; its Unknown is not renamed (authors, 12 Nov 2026);
- atlas groups: the split of the atlas DCT cells made for this comparison is named "(derived)" (stored: "(atlas_dct)"),
  the atlas's own classes "atlas classes" (stored: "published");
- rows: the centroid-mapping variants with the alternative centroid weighting (stored name ..._t07_style) are left
  out (the main variant and the equal-weight variants with 25 and 100 genes are kept). They are counted in the
  sources table under rows_other_annotation, the column the shared assembly code uses for rows left out;
- columns: label_column (given by condition and annotation), the free-text reason column and the source-file
  columns of the figure value tables are dropped.

Workbook: results/tables/report/report_11_S11c_tables.xlsx (sheet Contents, then sheets 1-7; Arial 10;
values only, no formulas).

Inputs (read-only): the tables named in SPECS.
Outputs: the workbook; results/tables/report/report_11_s11c_table_{sources,checks}.csv;
         results/tables/report/report_step_11_s11c_table_manifest.json

Command (project root):  python scripts/report_dct_rna_tables.py
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
from report_rna_protein_tables import Layout, Rules, Spec, build, rows_where, source_path

REPORT, DCT_RNA, ATLAS_DCT = "report", "dct_rna", "atlas_dct"
NOTE = "Unknown = DCT, subtype not resolved (cells of clusters without marker support)"
CLUSTER, SCORE_CLUSTERS = "cluster annotation", "marker-score label (clusters)"
MARKER_SCORE = ("Marker-score label: each cell takes the cell type whose marker proteins give the highest score (per "
                "cell); a cluster at Leiden resolution 3 takes the most frequent label of its cells (clusters).")
ATLAS_LEVEL, SPLIT_LEVEL = "atlas classes", "derived DCT split and atlas classes"
MAIN_VARIANT = "top 50 genes per class"


def shown(value: str) -> str:
    """Reader-facing name of a stored group, population or modality value."""
    value = value.replace("(atlas_dct)", "(derived)")
    value = value.replace("(primary)", f"({CLUSTER})").replace("(second)", "(marker-score label, clusters)")
    value = value.replace("RNA (Mouse Kidney Atlas; derived)", "RNA (Mouse Kidney Atlas)")
    value = re.sub(r"^SCP-MS Ctrl DCT \(combined search; dct_reviewer\)$", "SCP-MS, control DCT cells", value)
    value = re.sub(r"^SCP-MS ZeroK DCT \(combined search; dct_reviewer\)$", "SCP-MS, zero-K DCT cells", value)
    return value.replace("RNA (Mouse Kidney Atlas; atlas_dct)", "RNA (Mouse Kidney Atlas)")


def value_map(sources: tuple[tuple[str, ...], ...], columns: tuple[str, ...]) -> dict[str, str]:
    """{stored value: reader-facing name} over the listed columns of the listed source tables."""
    out: dict[str, str] = {}
    for src in sources:
        frame = pd.read_csv(source_path(src))
        for col in columns:
            if col in frame.columns:
                out.update({v: shown(v) for v in frame[col].dropna().astype(str).unique()})
    return out


S_SHARED_SUM, S_SHARED = (DCT_RNA, "shared_features_summary"), (DCT_RNA, "shared_features")
S_RHO = (DCT_RNA, "pseudobulk_rho")
S_MAP, S_DEPTH, S_VARIANTS, S_GENES, S_CELL = ((DCT_RNA, "mapping_summary"), (DCT_RNA, "mapping_plate_depth"),
                                               (DCT_RNA, "mapping_variants"), (DCT_RNA, "mapping_informative_genes"),
                                               (DCT_RNA, "mapping_per_cell"))
S_SPLIT, S_FC, S_TC = (ATLAS_DCT, "dct_split_subclusters"), (DCT_RNA, "fc_concordance"), (DCT_RNA, "fc_transporters_channels")
S_CYC, S_DOT = (DCT_RNA, "cycling_side_by_side"), (DCT_RNA, "dotplot_rna_values")
S_F3B, S_S11A, S_S11B = (REPORT, "11_fig3b_protein_rna_values"), (REPORT, "11_s11a_figure_values"), (REPORT, "11_s11b_figure_values")
GROUPS = value_map((S_RHO, S_GENES, S_DOT, S_CYC, S_SPLIT, S_CELL),
                   ("rna_group", "group", "population", "modality", "name_thr_0.10", "name_thr_0.25", "name_thr_0.50",
                    "published_top_class", "published_second_class", "derived_top_class", "derived_second_class"))
SAME = {v: v for v in ("Control", "Zero-K", "atlas")}
VARIANTS = {"top25_equal_weight_z": "top 25 genes per class", "top50_equal_weight_z": MAIN_VARIANT,
            "top100_equal_weight_z": "top 100 genes per class"}
KEEP_VARIANTS = ("variant", "in", tuple(VARIANTS))


def left_out_rows(src: pd.DataFrame) -> pd.Series:
    if "variant" in src.columns:
        return ~src["variant"].isin(tuple(VARIANTS))
    return pd.Series(False, index=src.index)


RULES = Rules(
    label_display={}, label_allowed=frozenset(),
    column_values={
        "condition": {"Ctrl": "Control", "ZeroK": "Zero-K", "both": "atlas", **SAME},
        "level": {"published": ATLAS_LEVEL, "derived": SPLIT_LEVEL},
        "rna_source": {"all": "all cells and nuclei", "nuclei": "nuclei", "cells": "cells"},
        "label_role": {"primary": CLUSTER, "second": SCORE_CLUSTERS},
        "scp_set": {"with_unknown": "all groups", "core_only": "without Unknown"},
        "value_type": {"raw": "mean level", "z": "z-score across groups"},
        "variant": VARIANTS,
        "by": {"plate": "sort plate", "depth_bin": "protein-ID quartile",
               "spearman_top_score_vs_protein_ids": "Spearman, top score against protein IDs",
               "spearman_margin_vs_protein_ids": "Spearman, margin against protein IDs"},
        "gene_set": {"without DCT V1 workbook markers": "without the DCT list markers"},
        **{col: GROUPS for col in ("rna_group", "group", "population", "modality", "name_thr_0.10", "name_thr_0.25",
                                   "name_thr_0.50", "published_top_class", "published_second_class",
                                   "derived_top_class", "derived_second_class")},
    },
    strict=frozenset({"condition", "level", "rna_source", "label_role", "scp_set", "value_type", "variant", "by"}),
    rename_exact={
        "label_role": "annotation", "primary_label": "cluster_annotation",
        "second_label": "marker_score_label_clusters", "n_detected_proteins": "n_protein_ids",
        "ambiguous_tal_cnt_interstitial": "any_flag", "primary_variant": "main_variant",
        "primary_rna": "main_rna_variant", "workbook_marker": "dct_list_marker", "depth_bin": "protein_id_quartile",
        "scp_set": "scp_groups", "by": "split_by",
    },
    rename_re=((re.compile(r"\(atlas_dct\)"), "(derived)"), (re.compile(r"_atlas_dct$"), "_derived"),
               (re.compile(r"^published_"), "atlas_classes_"), (re.compile(r"^derived_"), "with_split_"),
               (re.compile(r"_second_(class|score)$"), r"_runner_up_\1"),
               (re.compile(r"_primary_"), "_cluster_annotation_"), (re.compile(r"_second_"), "_marker_score_")),
    forbidden=re.compile(r"(?<![A-Za-z0-9])t\d\d(?![0-9A-Za-z])|cs_(ctrl|zerok)|label_column|\bprimary\b|\bsecond\b"
                         r"|\bCtrl\b|\bZeroK\b|(?<![A-Za-z0-9])v[123](?![A-Za-z0-9])|\bbrief\b|\.csv"
                         r"|\.h5ad|\buser\b|\bpublished\b|mka", re.I),
    forbidden_header=re.compile(r"primary|second_|label_column|label_role|t08|published|^derived_", re.I),
    other_rows=left_out_rows,
)

LABEL_COLUMN = {"label_column": ("condition", "label_role")}
FILES = "names a source file"
SHEETS = {
    1: ("1 Shared genes", "Proteins matched to atlas genes, and their detection in the SCP-MS groups and atlas classes"),
    2: ("2 Group correlations", "Spearman correlation between atlas class means and SCP-MS group means"),
    3: ("3 Nearest atlas class", "Nearest atlas class of each SCP-MS cell: by label, protein depth, gene number, "
                                 "and the genes used"),
    4: ("4 DCT2 against DCT1", "The split of the atlas DCT cells, and DCT2-against-DCT1 differences in RNA and protein"),
    5: ("5 Cycling markers", "Cells with Mki67 or Top2a in the atlas and in the SCP-MS data"),
    6: ("6 Figure values", "Values plotted in Figure 3B and in Supplementary Figures S-11-a and S-11-b, and the RNA "
                           "marker values for all atlas subsets"),
    7: ("7 Per cell", "Per-cell labels, scores and nearest atlas class"),
}

SPECS = [
    Spec("1a", S_SHARED_SUM, "Proteins by matching status",
         ("Each protein group was matched to the atlas genes by its first gene symbol (case-insensitive, "
          "one-to-one). Shared genes = matched, same case + matched, case differs.",)),
    Spec("1b", S_SHARED, "Per protein: matched atlas gene and detection",
         ("One row per protein and condition. shared: matched to an atlas gene. scp_detection_ columns: fraction of "
          "the cells with intensity above zero, over all cells and per group of the cluster annotation and of the "
          "marker-score label.",
          "atlas_detection_ columns: fraction of the atlas cells of the class with at least one count (the same "
          "values in both conditions); _derived: the two groups of the derived DCT split.")),
    Spec("2a", S_RHO, "Spearman correlation across genes, atlas groups against SCP-MS groups",
         ("One row per condition, atlas grouping (level), atlas subset (rna_source), annotation, set of SCP-MS "
          "groups, detection threshold, value type, atlas group and SCP-MS group. n_genes: genes detected in at "
          "least the threshold fraction of at least one group in each data set.",
          "status: 'does not apply' where the label has no cell or only one SCP-MS group exists (every z-score is "
          "0). Main rows: level atlas classes, all cells and nuclei, cluster annotation, all groups, threshold 0.1."),
         determined=LABEL_COLUMN),
    Spec("3a", S_MAP, "Nearest atlas class per SCP-MS label",
         ("frac_top_ columns: fraction of the cells of the label whose highest cosine similarity is with that atlas "
          "class; TAL = CTAL + MTAL. frac_low_confidence: cells with a top score below 0.20 or a margin to the "
          "runner-up below 0.05. _confident: without the low-confidence cells.",
          f"main_variant: {MAIN_VARIANT}, used in the text and in Supplementary Figure S-11-b. status 'does not "
          "apply': the label has no cell."),
         rows=(KEEP_VARIANTS,), determined=LABEL_COLUMN),
    Spec("3b", S_DEPTH, "Nearest atlas class by sort plate and by quartile of protein IDs",
         ("Main variant. value: plate, or quartile of the number of proteins identified per cell within the "
          "condition. The two Spearman rows give the correlation (rho, p_value) of the top score and of the margin "
          "with the number of protein IDs per cell.",)),
    Spec("3c", S_VARIANTS, "Number of informative genes: agreement with the main variant",
         ("fraction_same_top_class_as_primary: fraction of the cells with the same nearest class as under the main "
          "variant.",), rows=(KEEP_VARIANTS,),
         rename={"fraction_same_top_class_as_primary": "fraction_same_top_class_as_main_variant"}),
    Spec("3d", S_GENES, "Informative genes of the main variant and the class centroids",
         ("Genes higher in one atlas group than in the others (Wilcoxon, q < 0.05, log2 fold change >= 1, detected "
          "in >= 25% of the group) whose protein is detected in >= 10% of the cells of the condition; up to 50 per "
          "group. group: the group the gene is listed for. centroid_z_ columns: mean RNA level of the gene in each "
          "group, z-scored across the groups.",)),
    Spec("4a", S_SPLIT, "Subclusters of the atlas DCT cells and the derived DCT1 / DCT2 names",
         ("The atlas has no DCT1 or DCT2 class. The 16,381 atlas DCT cells were clustered on the atlas's stored "
          "latent space (Leiden; resolution and seed per row). z_ columns: subcluster mean of each naming gene "
          "minus the mean over all DCT cells, in SD units; dct1_score, dct2_score: mean z of the DCT1 and of the "
          "DCT2 naming genes; difference = dct1_score - dct2_score.",
          "name_thr_ columns: name of the subcluster at a difference of at least the threshold (0.25 is used; "
          "resolution 0.2, seed 42). n_nuclei, n_cells_sc: nuclei and single cells; det_ columns: fraction of the "
          "subcluster's cells with at least one count.")),
    Spec("4b", S_FC, "Agreement of DCT2-against-DCT1 differences between RNA and protein",
         ("RNA: derived DCT2 minus derived DCT1, mean log-normalized expression. Protein: DCT2 minus DCT1, mean "
          "ln(1 + intensity) (raw), or the DCT2 coefficient of a linear model with ln(protein IDs) and plate "
          "(adjusted). spearman_rho across the genes of the gene set; fraction_same_sign with its binomial p.",
          "rna_variant: atlas subset and naming threshold of the split (all_thr0.25 is the main one); threshold: "
          "detection threshold for the genes compared. status 'does not apply': the annotation has no DCT2 cells."),
         determined=LABEL_COLUMN, drop={"reason": "free-text working note"}),
    Spec("4c", S_TC, "Transporters and channels: DCT2-against-DCT1 differences in RNA and protein",
         ("Genes selected by gene-symbol prefix (class). direction_: concordant if the RNA and protein differences "
          "have the same sign; small_difference: absolute difference below 0.1 in either data set. naming_gene: "
          "used to name the derived split; dct_list_marker: listed in the DCT marker list.",),
         determined=LABEL_COLUMN),
    Spec("5a", S_CYC, "Cells with cycling markers, atlas and SCP-MS",
         ("k of n_cells with the measure; fraction with its Wilson 95% interval. Atlas rows: at least one count; "
          "SCP-MS rows: intensity above zero. S_or_G2M: cell-cycle phase from S and G2M scores (G1 if both below "
          "0); cycling_score_rule: the highest tenth of a protein cycling score within the condition; ProLIF_share: "
          "cells labelled ProLIF.",
          "Atlas rows are the same for both conditions (condition 'atlas'). depth-matched: ProLIF cells of the "
          "marker-score label and DCT1 + DCT2 cells matched on plate and protein IDs.")),
    Spec("6a", S_F3B, "Values of Figure 3B",
         ("half: protein (the four groups of Figure 3A, control cells) or RNA (atlas classes, all cells and "
          "nuclei). frac_detected: fraction of cells with intensity above zero or at least one count; "
          "colour_scaled_0_1: group mean scaled 0-1 per marker across the rows of the half. grey: gene not among "
          "the atlas genes.",),
         drop={"row": "stored label value; row_display is the name shown in the figure"}),
    Spec("6b", S_S11A, "Values of Supplementary Figure S-11-a, panel A (marker dot plot)",
         ("block: protein by cluster annotation (control, zero-K) or RNA by atlas class. marker: column of the "
          "dot plot. Columns as table 6a.",),
         rows=(("panel", "in", ("A",)),), constant=("panel",), drop_empty=True,
         drop={"row": "stored label value; row_display is the name shown in the figure", "source": FILES},
         rename={"column": "marker"}),
    Spec("6c", S_S11A, "Values of Supplementary Figure S-11-a, panel B (correlations)",
         ("block: value type and condition. row_display: atlas class; column_display: SCP-MS group of the cluster "
          "annotation. rho: Spearman correlation across n_genes genes.",),
         rows=(("panel", "in", ("B",)),), constant=("panel",), drop_empty=True,
         drop={"row": "repeats row_display", "column": "stored label value; column_display is the name shown",
               "source": FILES}),
    Spec("6d", S_S11B, "Values of Supplementary Figure S-11-b",
         ("Panels A and B: fraction of the cells of the row nearest to each atlas class, and the low-confidence "
          "fraction. Panel C: k of n_cells with the marker, fraction and Wilson 95% interval.",),
         drop={"row": "stored label value; row_display is the name shown in the figure", "source": FILES}),
    Spec("6e", S_DOT, "RNA marker values for all atlas groupings and subsets",
         ("One row per atlas grouping (level), subset (rna_source), atlas group and marker of the DCT list. "
          "frac_detected: fraction of cells with at least one count; mean_lognorm: mean log-normalized expression; "
          "mean_scaled_0_1: scaled 0-1 per gene across the groups of the grouping. gene_in_atlas false: not an "
          "atlas gene.",)),
    Spec("7a", S_CELL, "Per cell: labels, protein IDs, flags and nearest atlas class",
         ("One row per cell (729 control and 702 zero-K cells). atlas_classes_ columns: nearest class, score, "
          "runner-up, margin and low-confidence flag among the atlas classes; with_split_ columns: the same with "
          "the derived DCT split. _score_ columns: cosine similarity with each class centroid.",
          "tal_like_flag, cnt_cd_like_flag, interstitial_like_flag, any_flag: panel-score flags (Supplementary "
          "Table S-09-c). plate: sort plate of the pooled cells, technical only.")),
]

TERMS = [
    ("Data", "DCT cells of mice on a control or a zero-potassium (zero-K) diet for 4 days, searched together: 729 "
     "control and 702 zero-K cells; one pool of 3 mice per condition. Control and Zero-K were analysed separately. "
     "They are listed one after the other in each table, and no value compares the two conditions."),
    ("RNA data", "Mouse Kidney Atlas (Novella-Rausell et al., iScience 2023): single-cell and single-nucleus RNA-seq "
     "of healthy kidneys of male C57BL/6 mice from eight studies. RNA and protein come from different animals and "
     "studies; groups are compared, not cells. The atlas mice were on a normal diet."),
    (ATLAS_LEVEL, "Classes of the atlas at its annotation level Celltype: DCT (16,381 cells and nuclei), DCT-CNT "
     "(810), CNT (3,903), CTAL (3,600; cortical thick ascending limb) and MTAL (3,840; medullary thick ascending "
     "limb)."),
    ("derived", "The atlas has no DCT1 or DCT2 class. '(derived)' marks the split of the atlas DCT cells made for "
     "this comparison (table 4a): DCT1 (derived) 8,864 cells, DCT2 (derived) 2,752 cells (39 nuclei), 4,765 cells "
     "unassigned. These names are not atlas labels."),
    ("rna_source", "all cells and nuclei; nuclei (single-nucleus RNA-seq); cells (single-cell RNA-seq)."),
    (CLUSTER, "Label of the Leiden cluster of the cell (resolution 1) under the annotation rule: DCT1, DCT2 or "
     f"Unknown. {NOTE}. Control: DCT1 382, DCT2 85, Unknown 262 cells; zero-K: DCT1 449, Unknown 253 cells."),
    ("marker-score label", f"{MARKER_SCORE} Its labels are DCT1, DCT2, ProLIF and Unknown."),
    ("detection", "Protein: intensity above zero in the cell. RNA: at least one count in the cell. Detection "
     "fractions depend on the sampling depth of each method and are not abundances."),
    ("mean level; z-score across groups", "RNA: mean log-normalized expression of the group (counts scaled to "
     "10,000 per cell, ln(1 + x)). Protein: mean ln(1 + intensity), 0 if not detected. z-score: each gene "
     "standardized across the groups within its data set; with two groups the z-scores are -1, 0 or 1."),
    ("scp_groups", "all groups: every label of the annotation that has cells; without Unknown: DCT1, DCT2 and "
     "ProLIF only."),
    ("protein IDs", "Number of proteins identified in a cell (n_protein_ids); protein_id_quartile: quartile within "
     "the condition."),
    ("plate; cell_key; cell_id", "plate: sort plate of the pooled cells (P1, P2); technical only, not an animal or "
     "replicate. cell_key: condition, plate and well. cell_id: name of the run in the raw data."),
]

C, Z, A = "Control", "Zero-K", "atlas"
MAIN_RHO = dict(level=ATLAS_LEVEL, rna_source="all cells and nuclei", annotation=CLUSTER, scp_groups="all groups",
                threshold=0.1)
MAIN_MAP = dict(level=ATLAS_LEVEL, variant=MAIN_VARIANT, annotation=CLUSTER)
MAIN_FC = dict(rna_variant="all_thr0.25", annotation=CLUSTER, threshold=0.1, gene_set="all compared genes")
RNA, SCP_C, SCP_Z = "RNA (Mouse Kidney Atlas)", "SCP-MS, control DCT cells", "SCP-MS, zero-K DCT cells"

LETTER = [
    *[(f"{cond}: proteins matched to an atlas gene, same case", "1a",
       lambda f, cond=cond: int(_pick(f, "n_proteins", condition=cond, status="matched, same case")), value, None)
      for cond, value in ((C, 5520), (Z, 5415))],
    *[(f"{cond}: proteins matched, case differs", "1a",
       lambda f, cond=cond: int(_pick(f, "n_proteins", condition=cond, status="matched, case differs")), 8, None)
      for cond in (C, Z)],
    *[(f"{cond}: shared genes in the per-protein table", "1b",
       lambda f, cond=cond: int(sum(bool(v) for v in rows_where(f, condition=cond)["shared"])), value, None)
      for cond, value in ((C, 5528), (Z, 5423))],
    *[(f"{cond}: rho, atlas {rg} against {sg} ({vt})", "2a",
       lambda f, cond=cond, rg=rg, sg=sg, vt=vt: _pick(f, "rho", condition=cond, value_type=vt, rna_group=rg,
                                                       scp_group=sg, **MAIN_RHO), value, 2)
      for cond, vt, rg, sg, value in (
          (C, "mean level", "DCT", "DCT1", 0.50), (C, "mean level", "DCT", "DCT2", 0.47),
          (C, "mean level", "DCT", "Unknown", 0.46), (Z, "mean level", "DCT", "DCT1", 0.50),
          (Z, "mean level", "DCT", "Unknown", 0.46), (C, "mean level", "CNT", "DCT2", 0.17),
          (Z, "mean level", "CNT", "Unknown", 0.18), (C, "mean level", "DCT-CNT", "Unknown", -0.04),
          (C, "z-score across groups", "DCT", "DCT1", 0.37), (C, "z-score across groups", "DCT", "Unknown", -0.35),
          (Z, "z-score across groups", "DCT", "DCT1", 0.18), (Z, "z-score across groups", "DCT", "Unknown", -0.16))],
    *[(f"{cond}: genes compared in the main rows", "2a",
       lambda f, cond=cond: int(_pick(f, "n_genes", condition=cond, value_type="mean level", rna_group="DCT",
                                      scp_group="DCT1", **MAIN_RHO)), value, None)
      for cond, value in ((C, 2022), (Z, 1791))],
    *[(f"{cond}, {label}: fraction nearest to atlas {cls}", "3a",
       lambda f, cond=cond, label=label, cls=cls: _pick(f, f"frac_top_{cls}", condition=cond, scp_label=label,
                                                        **MAIN_MAP), value, 2)
      for cond, label, cls, value in ((C, "all", "DCT", 0.40), (C, "all", "DCT-CNT", 0.22), (C, "all", "CNT", 0.24),
                                      (C, "all", "TAL", 0.14), (Z, "all", "DCT", 0.32), (Z, "all", "DCT-CNT", 0.29),
                                      (Z, "all", "CNT", 0.28), (Z, "all", "TAL", 0.11), (C, "DCT1", "DCT", 0.47),
                                      (C, "DCT2", "DCT", 0.18), (Z, "DCT1", "DCT", 0.36))],
    *[(f"{cond}: low-confidence fraction, all cells", "3a",
       lambda f, cond=cond: _pick(f, "frac_low_confidence", condition=cond, scp_label="all", **MAIN_MAP), value, 2)
      for cond, value in ((C, 0.25), (Z, 0.30))],
    *[(f"{cond}, quartile {q}: fraction nearest to atlas {cls}", "3b",
       lambda f, cond=cond, q=q, cls=cls: _pick(f, f"frac_top_{cls}", condition=cond, level=ATLAS_LEVEL,
                                                split_by="protein-ID quartile", value=q), value, 2)
      for cond, q, cls, value in ((C, "Q1 (lowest)", "CNT", 0.70), (C, "Q4 (highest)", "DCT-CNT", 0.57),
                                  (C, "Q4 (highest)", "TAL", 0.43), (Z, "Q1 (lowest)", "CNT", 0.82),
                                  (Z, "Q4 (highest)", "DCT-CNT", 0.74), (Z, "Q4 (highest)", "TAL", 0.26))],
    ("derived split: cells of DCT1 (derived), resolution 0.2, seed 42", "4a",
     lambda f: int(_pick(f, "n_cells", resolution=0.2, seed=42, **{"name_thr_0.25": "DCT1 (derived)"})), 8864, None),
    ("derived split: cells and nuclei of DCT2 (derived)", "4a",
     lambda f: ", ".join(str(int(_pick(f, c, resolution=0.2, seed=42, **{"name_thr_0.25": "DCT2 (derived)"})))
                         for c in ("n_cells", "n_nuclei")), "2752, 39", None),
    *[(f"Control: DCT2-against-DCT1 rho, {pd_} ({what})", "4b",
       lambda f, pd_=pd_, col=col: _pick(f, col, condition=C, protein_difference=pd_, **MAIN_FC), value, dec)
      for pd_, col, what, value, dec in (("raw", "spearman_rho", "rho", -0.050, 3),
                                         ("adjusted for depth and plate", "spearman_rho", "rho", -0.034, 3),
                                         ("raw", "n_genes", "genes", 1812, 0),
                                         ("adjusted for depth and plate", "fraction_same_sign", "same sign", 0.54, 2))],
    *[(f"{mod}, {pop}: cells with {measure} (k, n)", "5a",
       lambda f, mod=mod, pop=pop, measure=measure: ", ".join(
           str(int(_pick(f, c, modality=mod, population=pop, measure=measure))) for c in ("k", "n_cells")), value,
       None)
      for mod, pop, measure, value in ((RNA, "DCT", "Mki67_or_Top2a", "217, 16381"), (RNA, "DCT", "Mki67", "118, 16381"),
                                       (RNA, "DCT", "Top2a", "113, 16381"),
                                       (SCP_C, "all cells", "Mki67_or_Top2a", "21, 729"),
                                       (SCP_C, "all cells", "Mki67", "15, 729"), (SCP_C, "all cells", "Top2a", "16, 729"),
                                       (SCP_Z, "all cells", "Mki67_or_Top2a", "56, 702"),
                                       (SCP_Z, "all cells", "Mki67", "52, 702"), (SCP_Z, "all cells", "Top2a", "35, 702"))],
    *[(f"atlas {grp}: fraction of cells with {marker} counts", "6e",
       lambda f, grp=grp, marker=marker: _pick(f, "frac_detected", level=ATLAS_LEVEL,
                                               rna_source="all cells and nuclei", group=grp, marker=marker), value, 2)
      for grp, marker, value in (("DCT", "Slc12a3", 0.84), ("DCT", "Pvalb", 0.45), ("DCT", "Umod", 0.48),
                                 ("CTAL", "Umod", 0.78), ("CTAL", "Slc12a1", 0.98), ("DCT", "Slc12a1", 0.10))],
    ("cells: control, zero-K", "7a",
     lambda f: f"{len(rows_where(f, condition=C))}, {len(rows_where(f, condition=Z))}", "729, 702", None),
]

LAYOUT = Layout(
    number="S-11-c", step="11_s11c_table", workbook="report_11_S11c_tables.xlsx",
    title="Supplementary Table S-11-c. RNA comparator for the DCT data: shared genes, group correlations, nearest "
          "atlas class, DCT2-against-DCT1 differences and cycling markers",
    intro=("SCP-MS of DCT cells from mice on a control or a zero-potassium (zero-K) diet (729 control and 702 "
           "zero-K cells) compared with the Mouse Kidney Atlas. All values are observed values; the terms used in "
           "the column names are explained below the list of tables.",
           "RNA and protein come from different animals and studies. Control and Zero-K were analysed separately "
           "and are listed one after the other in each table; no value compares the two conditions.",
           f"Cluster annotation: {NOTE}."),
    sheets=SHEETS, terms=TERMS,
    sheet_lines={n: (f"Cluster annotation: {NOTE}.",) for n in SHEETS},
)


def extra_checks(frames: dict, state: dict, checks: list[dict]) -> None:
    bad = {k: sorted(set(f["condition"].astype(str))) for k, f in frames.items()
           if "condition" in f.columns and not set(f["condition"].astype(str)) <= {C, Z, A}}
    check(checks, "condition columns hold Control, Zero-K or atlas only (each row belongs to one condition or to "
          "the atlas; no row compares the conditions)", bad or f"{sum('condition' in f.columns for f in frames.values())} "
          "tables with a condition column", "all tables", not bad)
    left = {k: sorted({v for col in ("rna_group", "group", "population") if col in f.columns
                       for v in f[col].dropna().astype(str) if "atlas_dct" in v or "(primary)" in v or "(second)" in v})
            for k, f in frames.items()}
    left = {k: v for k, v in left.items() if v}
    check(checks, "group names: no stored suffix left ((atlas_dct), (primary), (second))", left or "none", "none", not left)
    used = {v for f in frames.values() if "annotation" in f.columns for v in f["annotation"].dropna().astype(str)}
    check(checks, "annotation names used in the tables", sorted(used), f"{CLUSTER}; {SCORE_CLUSTERS}",
          used == {CLUSTER, SCORE_CLUSTERS})
    defined = [term for term, meaning in TERMS if MARKER_SCORE in meaning]
    check(checks, "Contents gives the agreed definition of the marker-score label word for word", defined,
          "one definition", len(defined) == 1)


if __name__ == "__main__":
    build(LAYOUT, SPECS, RULES, LETTER, "the response to the reviewers: Supplementary Table S-11-c, RNA comparator for the DCT data.",
          extra_checks)
