#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S21 (technical variation within the DCT experiment), one workbook.

. The workbook holds the
evidence of the response: the design and the cells per sort plate, technical QC by diet condition and sort plate,
the comparison of the two sort plates of one pooled sample (proteins per cell, plate mean profiles, label
fractions), the share of nearest neighbours from the same plate, the proteins that differ between the two plates,
and the principal components against protein number and plate. No new statistics: every value is copied from the
stored dct_intake, dct_labels, dct_reviewer and zerok_vs_control tables (combined control + zero-K search).

Only within-condition (technical) rows are taken. The rows that compare the two diet conditions belong to the
response to the Editors' comment 2 and are left out here (recorded per table in the sources table).

What the assembly changes (all recorded per table in the sources table):
- rows: the cluster annotation and all cells (the marker-score label rows are left out); the within-condition plate
  contrasts of the zerok_vs_control table; the three embeddings named in sheet 6;
- columns: internal label-column names and constant columns are dropped (checked);
- names: Ctrl / ZeroK -> Control / Zero-K; reader-facing names for label sets, embeddings and columns.

Workbook: results/tables/report/report_20_S20a_tables.xlsx (sheet Contents, then sheets 1-6), in the
layout of Supplementary Table S7 (shared code: report_table_lib.py).

Inputs (read-only): the ten dct_intake / dct_labels / dct_reviewer / zerok_vs_control tables named in SPECS.
Outputs: the workbook; results/tables/report/report_20_s20a_table_{sources,checks}.csv;
         results/tables/report/report_step_20_s20a_table_manifest.json

Command (project root):  python scripts/report_plate_tables.py --overwrite
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

STEP = "20_s20a_table"
DCT_INTAKE = "results/tables/dct_intake/dct_intake_"
DCT_LABELS = "results/tables/dct_labels/dct_labels_"
DCT_REVIEWER = "results/tables/dct_reviewer/dct_reviewer_"
ZEROK_VS_CONTROL = "results/tables/zerok_vs_control/zerok_vs_control_"

ALL_CELLS = "all cells"
ANNOTATION = "cluster annotation"
CELL_SET = {"all_cells": ALL_CELLS, "primary": ANNOTATION}
KEPT_SETS = (("label_role", ("all_cells", "primary")),)
ANNOTATION_ONLY = (("label_role", ("primary",)),)
INTERNAL = "label_column holds the internal name of the label set; the row filter and cell_set name it"
PLATE_COLS = ((r"^n_(P[12])$", r"n_cells_\1"), (r"^(median|q25|q75)_(P[12])$", r"proteins_per_cell_\1_\2"))

METRIC = {"protein IDs per cell": "detected proteins per cell",
          "fraction missing per cell": "fraction of the retained proteins not detected, per cell",
          "log10 total intensity": "log10 total intensity per cell",
          "median ln(1+intensity), detected": "median ln(1 + intensity) of the detected proteins, per cell"}
RETENTION = {"both conditions": "both conditions", "Ctrl only": "Control only", "ZeroK only": "Zero-K only",
             "neither": "neither condition"}
OWN = "principal components of the main analysis of the condition (20 components)"
EMBEDDING = {
    "dct_intake joint X_pca__none (cells of the condition)":
        "principal components of both conditions together (20 components), no per-cell normalization",
    "dct_intake joint X_pca__cell_median (cells of the condition)":
        "principal components of both conditions together (20 components), per-cell median normalization",
    "dct_intake joint X_pca__condition_align (cells of the condition)":
        "principal components of both conditions together (20 components), Zero-K intensities shifted to Control",
    "own step-1 graph PCA (X_pca_neighbors, 20 PCs)": OWN,
}
CONTRAST = {"Ctrl P2 vs P1": "Control, plate P2 against plate P1", "ZeroK P2 vs P1": "Zero-K, plate P2 against plate P1"}
CONTRAST_SET = {"label_free": ALL_CELLS, "dct_label": ANNOTATION}
NORMALIZATION = {"none": "none (main pipeline)", "cell_median": "per-cell median"}
MAIN, NO_DEPTH, NO_PLATE = "main analysis", "protein number regressed out", "sort plate regressed out"
OBJECT = {"uncorrected": MAIN, "depth_regressed": NO_DEPTH, "plate_regressed": NO_PLATE}

SHEETS = {
    1: ("1 Design and cells", "Cells per diet condition and sort plate, and protein groups retained per condition",
        ("One pool of 3 mice per diet condition; each pool was sorted onto two 384-well plates (P1, P2).",)),
    2: ("2 QC by condition and plate", "Technical quality metrics per retained cell, by diet condition and sort plate",
        ("Each condition was filtered on its own; the metrics are counted over the protein groups retained in the "
         "condition.",)),
    3: ("3 Sort plates compared", "The two sort plates of one pooled sample compared, within each diet condition",
        (UNKNOWN_NOTE + ".",
         "The two plates of a condition hold cells of the same pooled suspension, so a difference between them is "
         "technical.")),
    4: ("4 Neighbours by plate", "Share of the 10 nearest neighbours of a cell that come from the same sort plate",
        ("Within each diet condition; a share of about 0.50 is expected if the plates are fully mixed.",)),
    5: ("5 Proteins by plate", "Proteins that differ between the two sort plates of one diet condition",
        ("Technical contrast: both plates hold cells of the same pooled suspension. All p- and q-values are "
         "descriptive: cells are not animal-level replicates.",)),
    6: ("6 Components, depth, plate", "Principal components against protein number per cell and sort plate",
        ("Within each diet condition.",)),
}

SPECS = [
    Spec("1a", ZEROK_VS_CONTROL + "design_cells.csv", "Cells sorted, retained and removed, per diet condition and sort plate",
         ("Removed: cells with fewer than 100 detected proteins. sort_plate all: both plates of the condition.",
          "Zero-K, plate P1: 383 cells, because one well is absent from the export."),
         values={"condition": CONDITION}, strict=("condition",), rename={"plate": "sort_plate"},
         contents_note="The filtering stages and the proteins per cell are in Supplementary Table S-01-a."),
    Spec("1b", DCT_INTAKE + "retention_summary.csv", "Protein groups of the search (6,208) by the condition they were retained in",
         ("Retained: detected in at least two retained cells of the condition.",),
         values={"retention": RETENTION}, strict=("retention",), rename={"retention": "retained_in"}),
    Spec("2a", DCT_INTAKE + "qc_by_condition_plate.csv",
         "Detected proteins, missing fraction and intensity per retained cell: distribution by condition and sort plate",
         ("One row per condition, sort plate and metric: minimum, quartiles, median and maximum over the cells.",),
         rows=(("protein_set", ("own retained proteins",)),), drop_const=("protein_set",), drop=("metric",),
         drop_reason="metric holds the internal code of the metric that metric_display names",
         values={"condition": CONDITION, "metric_display": METRIC}, strict=("condition", "metric_display"),
         rename={"plate": "sort_plate", "metric_display": "metric"},
         assembly_note="The rows counted over the protein groups retained in both conditions are left out; their "
                       "medians differ from the rows shown by at most 1 protein per cell."),
    Spec("3a", DCT_REVIEWER + "plate_detected_proteins.csv", "Detected proteins per cell on plate P1 and plate P2",
         ("Mann-Whitney test and rank-biserial correlation of plate P2 against plate P1 (negative: fewer proteins "
          "per cell on plate P2).",),
         rows=KEPT_SETS, drop=("label_column",), drop_reason=INTERNAL,
         values={"condition": CONDITION, "label_role": CELL_SET, "label": {"all": ALL_CELLS}},
         strict=("condition", "label_role"), rename={"label_role": "cell_set", "label": "cell_group"},
         rename_re=PLATE_COLS, dct_unknown=True),
    Spec("3b", DCT_REVIEWER + "plate_pseudobulk_spearman.csv",
         "Mean protein profile of plate P1 against plate P2: Spearman correlation across proteins",
         ("Profile of a plate: ln(1 + mean intensity over the cells of the plate), a protein not detected in a cell "
          "counting as 0.",
          "n_proteins: proteins detected on both plates (used for the correlation); n_proteins_only_: proteins "
          "detected on one plate only (left out)."),
         rows=KEPT_SETS, drop=("label_column",), drop_reason=INTERNAL, drop_const=("p_value",),
         values={"condition": CONDITION, "label_role": CELL_SET, "label": {"all": ALL_CELLS}},
         strict=("condition", "label_role"), rename={"label_role": "cell_set", "label": "cell_group"},
         dct_unknown=True,
         assembly_note="p_value is 0 in every row (below the smallest number the software represents) and is "
                       "dropped as a constant column."),
    Spec("3c", DCT_REVIEWER + "plate_label_fractions.csv", "Cluster annotation: cells per label on each sort plate",
         ("fraction_of_plate: share of the retained cells of the plate.",),
         rows=ANNOTATION_ONLY, drop_const=("label_role",), drop=("label_column",), drop_reason=INTERNAL,
         values={"condition": CONDITION}, strict=("condition",), rename={"plate": "sort_plate"}, dct_unknown=True),
    Spec("3d", DCT_REVIEWER + "plate_label_tests.csv", "Cluster annotation: chi-square test of sort plate against label",
         ("One test per condition, over all labels of the condition.",),
         rows=ANNOTATION_ONLY, drop_const=("label_role",), drop=("label_column",), drop_reason=INTERNAL,
         values={"condition": CONDITION}, strict=("condition",), rename={"dof": "degrees_of_freedom"},
         dct_unknown=True),
    Spec("4a", DCT_LABELS + "plate_mixing.csv",
         "Share of the 10 nearest neighbours from the same sort plate, against 1,000 permutations of the plate labels",
         ("cells_of: the cells whose neighbours are counted (plate P1, plate P2, or all cells of the condition).",
          "The first three embeddings use the principal components computed on both conditions together; the "
          "neighbours are searched among the cells of the condition only."),
         values={"condition": CONDITION, "embedding": EMBEDDING, "label": {"all_cells": ALL_CELLS}},
         strict=("condition", "embedding"),
         rename={"label": "cells_of", "observed_same_label_fraction": "same_plate_neighbour_fraction",
                 "expected_random": "expected_if_mixed", "enrichment": "observed_over_expected",
                 "null_mean": "permutation_mean", "null_q975": "permutation_q975",
                 "p_empirical_ge": "p_permutation"}),
    Spec("5a", ZEROK_VS_CONTROL + "specificity_plate_null.csv",
         "Proteins that differ between plate P2 and plate P1 of one condition (q < 0.05)",
         ("Per protein: two-part model (detection: penalized logistic regression; intensity among the detecting "
          "cells: linear regression), each with ln(detected proteins per cell) as covariate; Benjamini-Hochberg q "
          "over the tested proteins.",
          "Tested: proteins detected in at least 10% of the cells of one of the two plates.",
          "n_mixed_direction: the detection part and the intensity part point in opposite directions."),
         rows=(("kind", ("within-condition plate (technical null)",)), ("label_column", ("label_free", "dct_label")),
               ("group", ("all_cells", "DCT1")), ("variant", ("none", "cell_median"))),
         drop_const=("kind", "p_value_note"),
         values={"label_column": CONTRAST_SET, "group": {"all_cells": ALL_CELLS}, "contrast": CONTRAST,
                 "variant": NORMALIZATION},
         strict=("label_column", "contrast", "variant"),
         rename={"label_column": "cell_set", "group": "cell_group", "variant": "per_cell_normalization",
                 "n_reference": "n_cells_P1", "n_comparison": "n_cells_P2",
                 "n_q_below_0.05": "n_proteins_q_below_0.05", "n_up": "n_higher_on_P2", "n_down": "n_lower_on_P2",
                 "n_mixed": "n_mixed_direction"},
         assembly_note="Left out: the contrasts between the two diet conditions within one plate (response to the "
                       "Editors' comment 2), the second label set, the group of all labelled DCT cells (in Zero-K "
                       "it equals DCT1) and the normalization that shifts the Zero-K intensities (within a "
                       "condition it equals no normalization)."),
    Spec("6a", DCT_REVIEWER + "embed_pc_vs_depth_and_plate.csv",
         "Spearman correlation of principal components 1 to 5 with protein number per cell and with sort plate",
         ("Embeddings: the main analysis; the same after ln(detected proteins per cell) was regressed out of every "
          "protein; the same after the sort plate was regressed out.",
          "spearman_rho_plate_P2: correlation with the plate coded P1 = 0, P2 = 1."),
         rows=(("object", tuple(OBJECT)),), values={"condition": CONDITION, "object": OBJECT},
         strict=("condition", "object"),
         rename={"object": "embedding", "pc": "principal_component",
                 "spearman_rho_ln_ids": "spearman_rho_ln_proteins_per_cell", "p_ln_ids": "p_ln_proteins_per_cell"},
         assembly_note="Left out: the embeddings with the cell-cycle score regressed out and the embedding on "
                       "proteins detected in at least half of the cells (response to Reviewer 2, major comment 1)."),
]

TERMS = [
    ("Design", "DCT cells of mice fed a control or a zero-potassium (Zero-K) diet for 4 days. One pool of 3 mice "
     "per condition: condition and pool cannot be separated, and cells are not animal-level replicates. Both "
     "conditions were sorted and acquired in the same sessions and searched together in one Spectronaut search "
     "(6,208 protein groups)."),
    ("condition; sort_plate", "Control, Zero-K: diet condition. P1, P2: the two 384-well sort plates of the pooled "
     "cells of a condition; they are technical plates, not animals or replicates."),
    ("detected protein; retained", "A protein is detected in a cell if its intensity is above zero. Retained cells: "
     "at least 100 detected proteins. Retained protein groups: detected in at least two retained cells of the "
     "condition."),
    ("cell_set; cell_group", "all cells: all retained cells of the condition. cluster annotation: the labels of the "
     "cell clusters (DCT1, DCT2, Unknown), with cell_group naming the label."),
    ("Unknown", "DCT, subtype not resolved: cells of clusters without marker support for DCT1 or DCT2."),
    ("proteins_per_cell_", "Number of detected proteins per cell: median and quartiles (q25, q75) on the plate."),
    ("spearman_rho", "Spearman rank correlation."),
    ("same_plate_neighbour_fraction", "Mean share of the 10 nearest neighbours of a cell that were sorted onto the "
     "same plate. expected_if_mixed: the share expected from the plate sizes. permutation_mean, permutation_q975 "
     "and p_permutation: mean, 97.5th percentile and one-sided empirical p-value from 1,000 permutations of the "
     "plate labels."),
    ("share_q_below_0.05", "Share of the tested proteins with q < 0.05 between the two plates. n_higher_on_P2, "
     "n_lower_on_P2: direction of the difference."),
    ("embedding", "The space in which neighbours or principal components were computed (per-protein scaled "
     "ln(1 + intensity), 20 principal components)."),
]


def sel(condition: str, **more) -> dict:
    return {"condition": condition, **more}


def both(getter) -> str:
    return ", ".join(str(getter(c)) for c in ("Control", "Zero-K"))


EVERY = {"cell_set": ALL_CELLS, "cell_group": ALL_CELLS}
PLATE5 = {"cell_set": ALL_CELLS, "per_cell_normalization": NORMALIZATION["none"]}

LETTER = [
    ("cells sorted and retained, control and zero-K", "1a",
     lambda f: both(lambda c: f"{_pick(f, 'n_cells_retained', sort_plate='all', **sel(c))} of "
                              f"{_pick(f, 'n_cells_sorted', sort_plate='all', **sel(c))}"),
     "729 of 768, 702 of 767", None),
    ("retained cells per plate: control P1, P2; zero-K P1, P2", "1a",
     lambda f: ", ".join(str(_pick(f, "n_cells_retained", sort_plate=p, **sel(c)))
                         for c in ("Control", "Zero-K") for p in ("P1", "P2")), "361, 368, 358, 344", None),
    ("protein groups retained in both conditions", "1b",
     lambda f: _pick(f, "n_protein_groups", retained_in="both conditions"), 5897, None),
    ("protein groups retained in one condition only: control, zero-K", "1b",
     lambda f: f"{_pick(f, 'n_protein_groups', retained_in='Control only')}, "
               f"{_pick(f, 'n_protein_groups', retained_in='Zero-K only')}", "181, 56", None),
    ("median detected proteins per cell, control", "2a",
     lambda f: _pick(f, "median", sort_plate="all", metric=METRIC["protein IDs per cell"], **sel("Control")), 1767, 1),
    ("median detected proteins per cell, zero-K", "2a",
     lambda f: _pick(f, "median", sort_plate="all", metric=METRIC["protein IDs per cell"], **sel("Zero-K")),
     1780.5, 1),
    ("median log10 total intensity per cell, control", "2a",
     lambda f: _pick(f, "median", sort_plate="all", metric=METRIC["log10 total intensity"], **sel("Control")),
     6.47, 2),
    ("median log10 total intensity per cell, zero-K", "2a",
     lambda f: _pick(f, "median", sort_plate="all", metric=METRIC["log10 total intensity"], **sel("Zero-K")),
     6.46, 2),
    ("median missing fraction per cell, control", "2a",
     lambda f: _pick(f, "median", sort_plate="all", metric=METRIC["fraction missing per cell"], **sel("Control")),
     0.71, 2),
    ("median missing fraction per cell, zero-K", "2a",
     lambda f: _pick(f, "median", sort_plate="all", metric=METRIC["fraction missing per cell"], **sel("Zero-K")),
     0.70, 2),
    ("control: median proteins per cell on P1 and P2", "3a",
     lambda f: f"{float(_pick(f, 'proteins_per_cell_median_P1', **EVERY, **sel('Control'))):.1f}, "
               f"{float(_pick(f, 'proteins_per_cell_median_P2', **EVERY, **sel('Control'))):.1f}",
     "1940.0, 1583.5", None),
    ("zero-K: median proteins per cell on P1 and P2", "3a",
     lambda f: f"{float(_pick(f, 'proteins_per_cell_median_P1', **EVERY, **sel('Zero-K'))):.1f}, "
               f"{float(_pick(f, 'proteins_per_cell_median_P2', **EVERY, **sel('Zero-K'))):.1f}",
     "1893.5, 1599.5", None),
    ("control: Mann-Whitney p of proteins per cell, P2 against P1", "3a",
     lambda f: f"{float(_pick(f, 'mannwhitney_p_P2_vs_P1', **EVERY, **sel('Control'))):.1e}", "1.1e-05", None),
    ("zero-K: Mann-Whitney p of proteins per cell, P2 against P1", "3a",
     lambda f: f"{float(_pick(f, 'mannwhitney_p_P2_vs_P1', **EVERY, **sel('Zero-K'))):.4f}", "0.0039", None),
    ("control: Spearman rho of the plate mean profiles, all cells", "3b",
     lambda f: _pick(f, "spearman_rho", **EVERY, **sel("Control")), 0.971, 3),
    ("zero-K: Spearman rho of the plate mean profiles, all cells", "3b",
     lambda f: _pick(f, "spearman_rho", **EVERY, **sel("Zero-K")), 0.965, 3),
    ("proteins in the plate profiles, control and zero-K", "3b",
     lambda f: both(lambda c: _pick(f, "n_proteins", **EVERY, **sel(c))), "5980, 5774", None),
    ("rho within the labels of the cluster annotation: lowest and highest", "3b",
     lambda f: (lambda v: f"{v.min():.3f}, {v.max():.3f}")(
         f.loc[f["cell_set"].eq(ANNOTATION), "spearman_rho"].astype(float)), "0.928, 0.968", None),
    ("chi-square p of plate against label, control", "3d",
     lambda f: _pick(f, "p_value", **sel("Control")), 0.0096, 4),
    ("chi-square p of plate against label, zero-K", "3d",
     lambda f: _pick(f, "p_value", **sel("Zero-K")), 0.82, 2),
    ("control: unresolved cells on P1 and P2 (share of the plate)", "3c",
     lambda f: ", ".join(f"{float(_pick(f, 'fraction_of_plate', sort_plate=p, label='Unknown', **sel('Control'))):.3f}"
                         for p in ("P1", "P2")), "0.305, 0.413", None),
    ("control: same-plate neighbour share, main analysis", "4a",
     lambda f: _pick(f, "same_plate_neighbour_fraction", embedding=OWN, cells_of=ALL_CELLS, **sel("Control")),
     0.687, 3),
    ("zero-K: same-plate neighbour share, main analysis", "4a",
     lambda f: _pick(f, "same_plate_neighbour_fraction", embedding=OWN, cells_of=ALL_CELLS, **sel("Zero-K")),
     0.691, 3),
    ("expected same-plate share if mixed, control and zero-K", "4a",
     lambda f: both(lambda c: f"{float(_pick(f, 'expected_if_mixed', embedding=OWN, cells_of=ALL_CELLS, **sel(c))):.3f}"),
     "0.499, 0.499", None),
    ("same-plate share over all four embeddings (all cells): lowest and highest", "4a",
     lambda f: (lambda v: f"{v.min():.3f}, {v.max():.3f}")(
         f.loc[f["cells_of"].eq(ALL_CELLS), "same_plate_neighbour_fraction"].astype(float)), "0.655, 0.691", None),
    ("control: proteins tested and with q < 0.05 between the plates", "5a",
     lambda f: f"{_pick(f, 'n_proteins_tested', contrast=CONTRAST['Ctrl P2 vs P1'], **PLATE5)}, "
               f"{_pick(f, 'n_proteins_q_below_0.05', contrast=CONTRAST['Ctrl P2 vs P1'], **PLATE5)}",
     "4059, 1691", None),
    ("zero-K: proteins tested and with q < 0.05 between the plates", "5a",
     lambda f: f"{_pick(f, 'n_proteins_tested', contrast=CONTRAST['ZeroK P2 vs P1'], **PLATE5)}, "
               f"{_pick(f, 'n_proteins_q_below_0.05', contrast=CONTRAST['ZeroK P2 vs P1'], **PLATE5)}",
     "3680, 1993", None),
    ("control: share of tested proteins with q < 0.05 between the plates", "5a",
     lambda f: _pick(f, "share_q_below_0.05", contrast=CONTRAST["Ctrl P2 vs P1"], **PLATE5), 0.417, 3),
    ("zero-K: share of tested proteins with q < 0.05 between the plates", "5a",
     lambda f: _pick(f, "share_q_below_0.05", contrast=CONTRAST["ZeroK P2 vs P1"], **PLATE5), 0.542, 3),
    ("PC1 against ln(proteins per cell), main analysis, control and zero-K", "6a",
     lambda f: both(lambda c: f"{float(_pick(f, 'spearman_rho_ln_proteins_per_cell', embedding=MAIN, principal_component=1, **sel(c))):.3f}"),
     "0.999, 0.999", None),
    ("PC1 against sort plate, main analysis, control and zero-K", "6a",
     lambda f: both(lambda c: f"{float(_pick(f, 'spearman_rho_plate_P2', embedding=MAIN, principal_component=1, **sel(c))):.2f}"),
     "-0.16, -0.11", None),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    """The tables agree with each other, and no row compares the two diet conditions."""
    a, d, p = stored["1a"], stored["3a"], stored["5a"]
    cells = {(c, pl): int(_pick(a, "n_cells_retained", condition=c, sort_plate=pl))
             for c in ("Control", "Zero-K") for pl in ("P1", "P2")}
    in_3a = {(c, pl): int(_pick(d, f"n_cells_{pl}", condition=c, **EVERY)) for c, pl in cells}
    lib.check(checks, "tables 1a and 3a agree: retained cells per condition and sort plate", in_3a, cells,
              in_3a == cells)
    in_5a = {(c, pl): int(_pick(p, f"n_cells_{pl}", contrast=CONTRAST[k], **PLATE5))
             for (c, k) in (("Control", "Ctrl P2 vs P1"), ("Zero-K", "ZeroK P2 vs P1")) for pl in ("P1", "P2")}
    lib.check(checks, "tables 1a and 5a agree: plate P1 is the reference and plate P2 the comparison of every "
              "contrast (cells per plate)", in_5a, cells, in_5a == cells)
    b = stored["1b"]
    total = int(b["n_protein_groups"].astype(int).sum())
    lib.check(checks, "table 1b: the four classes add up to the protein groups of the search", total, 6208,
              total == 6208)
    src = lib.read_source(cc.PROJECT_ROOT / (ZEROK_VS_CONTROL + "specificity_plate_null.csv"))
    same = src[src["kind"].eq("within-condition plate (technical null)")]
    none, align = (same[same["variant"].eq(v)].drop(columns="variant").reset_index(drop=True)
                   for v in ("none", "condition_align"))
    lib.check(checks, "table 5a, row left out: within a condition, shifting the Zero-K intensities gives the same "
              "counts as no normalization", f"{int((none == align).all(axis=1).sum())} of {len(none)} rows equal",
              f"{len(none)} of {len(none)}", bool(none.equals(align)))
    between = [key for key, frame in stored.items()
               for value in frame.to_numpy().ravel()
               if isinstance(value, str) and ("between conditions" in value or " vs Ctrl" in value
                                              or "Zero-K against Control" in value)]
    lib.check(checks, "within-condition rows only: no table row compares the two diet conditions",
              sorted(set(between)) or "none", "none", not between)
    q = lib.read_source(cc.PROJECT_ROOT / (DCT_INTAKE + "qc_by_condition_plate.csv"))
    own = q[q["protein_set"].eq("own retained proteins") & q["metric"].eq("n_detected")].reset_index(drop=True)
    joint = q[q["metric"].eq("n_detected_joint")].reset_index(drop=True)
    gap = float((own["median"] - joint["median"]).abs().max())
    lib.check(checks, "table 2a, rows left out: median proteins per cell over the protein groups retained in both "
              "conditions differ from the rows shown by at most 1", gap, "<= 1", gap <= 1.0)


WORKBOOK = WorkbookDef(
    label="S-20-a", step=STEP, workbook="report_20_S20a_tables.xlsx",
    title="Supplementary Table S-20-a. Technical variation within the DCT experiment: the two diet conditions and "
          "the two sort plates of each pooled sample",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names "
             "are explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"dct": "combined control + zero-K search (dct_intake, dct_labels, dct_reviewer and zerok_vs_control tables)",
                "scope": "within-condition (technical) rows only; rows comparing the diet conditions are left out"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-20-a (technical variation).")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
