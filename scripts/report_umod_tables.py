#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-10-b (uromodulin in DCT1 and DCT2 cells of the DCT data), one
workbook.

(authors, 12 Nov 2026: "Yes, go ahead" on the the response to the reviewers plan;
Editors' comment 3). One workbook with a Contents sheet and five data sheets, in the layout of Supplementary
Table S-07-d (shared code: report_table_lib.py). No new statistics: every value is copied from a stored
table. Detection and intensity of Umod and of the DCT list markers by label are in Supplementary Table S-09-c
(sheet 4) and are not repeated here.

Sheets:
  1  Umod, DCT1 against DCT2: detection and intensity tests per condition and label set (dct_reviewer).
  2  DCT2 - DCT1 difference of the DCT list markers, Umod and Slc12a1, unadjusted and adjusted for proteins per
     cell and sort plate (dct_rna values, collected by report_umod_evidence.py), and the same markers in the
     Mouse Kidney Atlas with the derived DCT split.
  3  Umod in the cluster test of the annotation, per cluster (dct_labels values, collected by the evidence script).
  4  Umod, Slc12a1 and Pvalb by cell type in the total-kidney data (kidney_annotation values).
  5  Umod, Pvalb, Slc12a1 and Slc12a3 transcripts by atlas group (dct_rna values).

What the assembly changes (all recorded per table in the sources table): Ctrl / ZeroK -> Control / Zero-K; the
two label sets are named "cluster annotation" and "marker-score label (clusters)"; columns that hold an internal
column name are dropped; marker-support codes, kidney labels (DCT1 -> DCT, PC -> CNT/PC) and the atlas groups of
the derived split (-> DCT1-like, DCT2-like) get their reader-facing names. The cluster label value Unknown stays
in the cells, with the note "Unknown = DCT, subtype not resolved". Control and zero-K were analysed separately and
are listed one after the other; no value compares them.

Workbook: results/tables/report/report_10_S10b_tables.xlsx
Inputs (read-only): the seven tables named in SPECS.
Outputs: the workbook; results/tables/report/report_10_s10b_table_{sources,checks}.csv;
         results/tables/report/report_step_10_s10b_table_manifest.json

Command (project root), after report_umod_evidence.py:
  python scripts/report_umod_tables.py --overwrite
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
from report_table_lib import CONDITION, KIDNEY_LABEL, MARKER_SCORE, UNKNOWN_NOTE, Spec, WorkbookDef, _pick

STEP = "10_s10b_table"
KIDNEY_ANNOTATION = "results/tables/kidney_annotation/kidney_annotation_"
REPORT = "results/tables/report/report_"
DCT_REVIEWER = "results/tables/dct_reviewer/dct_reviewer_"
DCT_RNA = "results/tables/dct_rna/dct_rna_"
CLUSTER, SCORE = "cluster annotation", "marker-score label (clusters)"
ANNOTATION = {"primary": CLUSTER, "second": SCORE}
NO_DCT2 = ((r"no DCT2 cell under cs_zerok_pipeline_label", "no cluster was annotated DCT2"),)
INTERNAL = "internal column name; the label set is given by the columns condition and annotation"
SUPPORT = {"high": "type-specific marker among the 25 most increased proteins",
           "specific_any_rank": "type-specific marker, any rank",
           "low_shared_only": "shared markers only",
           "unknown_no_marker_in_top_de": "no marker support (subtype not resolved)"}
ATLAS_LEVEL = {"published": "atlas cell types", "derived": "atlas cell types, DCT cells split (DCT1-like, DCT2-like)"}
ATLAS_SOURCE = {"all": "cells and nuclei", "nuclei": "nuclei only", "cells": "cells only"}
ATLAS_GROUP = {"DCT1 (atlas_dct)": "DCT1-like", "DCT2 (atlas_dct)": "DCT2-like"}
SPLIT = ("DCT1-like, DCT2-like: the atlas does not annotate DCT1 and DCT2. Its DCT cells were split into subclusters, "
         "and a subcluster was named DCT1-like or DCT2-like by the mean expression of the DCT1 and DCT2 marker genes "
         "of the DCT marker list, without Umod and Slc12a3. These are derived groups, not atlas labels.")

SHEETS = {
    1: ("1 Umod DCT1 vs DCT2", "Umod in DCT1 and DCT2 cells: detection and intensity tests", ()),
    2: ("2 Markers DCT2 minus DCT1", "DCT2 - DCT1 difference of the DCT markers, Umod and Slc12a1, protein and RNA",
        ()),
    3: ("3 Umod by cluster", "Umod in the cluster test of the annotation, per cluster",
        (f"Cluster annotation: {UNKNOWN_NOTE}.",)),
    4: ("4 Umod in total kidney", "Umod, Slc12a1 and Pvalb by cell type in the total-kidney data", ()),
    5: ("5 Atlas transcripts", "Umod, Pvalb, Slc12a1 and Slc12a3 transcripts in the Mouse Kidney Atlas", ()),
}

SPECS = [
    Spec("1a", DCT_REVIEWER + "umod_dct1_vs_dct2_tests.csv", "Umod, DCT1 against DCT2: detection and intensity",
         ("scope: detected cells only (cells with Umod detected); all cells, non-detected = 0; detection (cells with "
          "Umod detected against cells without). median_DCT1, median_DCT2: median ln(1 + intensity), or the detected "
          "fractions in the detection rows.",
          "status does not apply: the label set has no DCT2 cells in this condition. p_fdr_bh: Benjamini-Hochberg "
          "within condition and label set."),
         drop=("label_column",), drop_reason=INTERNAL,
         values={"condition": CONDITION, "label_role": ANNOTATION, "comparison": {"DCT1_vs_DCT2": "DCT1 vs DCT2"}},
         strict=("condition", "label_role", "comparison"), value_re={"reason": NO_DCT2},
         rename={"label_role": "annotation"}),
    Spec("2a", REPORT + "10_marker_dct2_vs_dct1.csv", "DCT2 - DCT1 difference per marker, protein",
         ("difference_mean_ln1p: mean ln(1 + intensity) of the DCT2 cells minus that of the DCT1 cells, non-detected "
          "= 0; q_mannwhitney_bh: Mann-Whitney U test, Benjamini-Hochberg over all compared proteins.",
          "difference_adjusted, p_adjusted: coefficient of DCT2 in ln(1 + intensity) ~ DCT2 + ln(proteins per cell) "
          "+ sort plate, fitted to the DCT1 and DCT2 cells. listed_for_in_dct_list: cell types of the DCT marker "
          "list that list the marker.",
          "The DCT1 and DCT2 markers were used to annotate the clusters, so their differences between DCT1 and DCT2 "
          "cells are expected. Umod and Slc12a3 were treated as shared markers."),
         drop=("label_column",), drop_reason=INTERNAL,
         values={"condition": CONDITION, "label_role": ANNOTATION}, strict=("condition", "label_role"),
         value_re={"reason": NO_DCT2}, rename={"label_role": "annotation"}),
    Spec("2b", REPORT + "10_marker_rna_dct_split.csv", "The same markers in the Mouse Kidney Atlas, DCT1-like and "
         "DCT2-like cells",
         ("rna_fraction_detected: fraction of the atlas cells with at least one count (cells and nuclei). "
          "rna_difference: mean log-normalized expression, DCT2-like minus DCT1-like; _cells_only, _nuclei_only: "
          "from the atlas cells or nuclei alone.",
          "The atlas cells and the SCP-MS cells come from different animals and studies.")),
    Spec("3a", REPORT + "10_umod_by_cluster.csv", "Umod in each cluster against all other cells of the condition",
         ("One row per Leiden cluster (resolution 1). The cluster test is the test of the annotation: Wilcoxon "
          "rank-sum test on ln(1 + intensity), non-detected = 0, each cluster against all other cells; auc, p_value, "
          "q_bh from that test.",
          "umod_vs_other_cells: increased (q < 0.05, auc > 0.5, mean difference > 0), decreased (q < 0.05, auc < "
          "0.5, mean difference < 0), not different (q >= 0.05), mixed (q < 0.05, auc and mean difference of "
          "opposite sign).",
          "umod_rank_among_increased: rank of Umod among the significantly increased proteins of the cluster; "
          "marker_support: see Contents."),
         values={"condition": CONDITION, "marker_support": SUPPORT}, strict=("condition", "marker_support"),
         dct_unknown=True),
    Spec("4a", KIDNEY_ANNOTATION + "t15_dotplot_values_a_endo.csv", "Umod, Slc12a1 and Pvalb by cell type, total kidney",
         ("Cell types of Figure 2A (712 cells). n_detected, frac_detected: cells with the protein detected. mean_z: "
          "group mean of the scaled matrix; mean_ln1p_intensity: mean ln(1 + intensity), non-detected = 0; "
          "mean_z_scaled_0_1: mean_z scaled from 0 to 1 over the cell types.",),
         rows=(("marker", ("Umod", "Slc12a1", "Pvalb")),), drop_const=("label_column",),
         values={"group": KIDNEY_LABEL}, rename={"group": "cell_type", "group_position": "cell_type_position"},
         kidney=True),
    Spec("5a", DCT_RNA + "dotplot_rna_values.csv", "Umod, Pvalb, Slc12a1 and Slc12a3 transcripts by atlas group",
         ("frac_detected: fraction of the atlas cells of the group with at least one count; mean_lognorm: mean "
          "log-normalized expression; mean_scaled_0_1: mean_lognorm scaled from 0 to 1 over the groups of the row "
          "set. CTAL, MTAL: cortical and medullary thick ascending limb; DCT-CNT: atlas cell type between DCT and "
          "connecting tubule.",
          SPLIT),
         rows=(("marker", ("Umod", "Pvalb", "Slc12a1", "Slc12a3")),), drop_const=("condition", "gene_in_atlas"),
         drop_dependent=(("atlas_gene", "marker"),),
         values={"level": ATLAS_LEVEL, "rna_source": ATLAS_SOURCE, "group": ATLAS_GROUP},
         strict=("level", "rna_source"),
         rename={"level": "atlas_groups", "rna_source": "atlas_cells", "group_position": "group_position_in_figure"}),
]

TERMS = [
    ("Data", "DCT cells of mice on a control or a zero-potassium (zero-K) diet for 4 days, searched together: 729 "
     "control and 702 zero-K cells; one pool of 3 mice per condition. Control and Zero-K were analysed separately. "
     "They are listed one after the other in each table, and no value compares the two conditions. Sheet 4: total "
     "kidney cells (712). Sheet 5 and table 2b: Mouse Kidney Atlas (RNA)."),
    (CLUSTER, "Label of the Leiden cluster of the cell (resolution 1) under the annotation rule: DCT1, DCT2 or "
     f"Unknown. {UNKNOWN_NOTE}. Control: DCT1 382, DCT2 85, Unknown 262 cells; zero-K: DCT1 449, Unknown 253 cells "
     "(no cluster was annotated DCT2)."),
    ("marker-score label", f"{MARKER_SCORE} Its labels are DCT1, DCT2 and ProLIF, and for clusters also Unknown."),
    ("marker_support", "Marker evidence behind the cluster annotation of a cluster: type-specific marker among the "
     "25 most increased proteins; type-specific marker, any rank (significantly increased at any rank); shared "
     "markers only (Slc12a3 or Umod); no marker support (subtype not resolved)."),
    ("detected", "Intensity above zero in the cell. fraction_detected, frac_detected: fraction of the cells."),
    ("ln1p", "ln(1 + intensity); 0 if the protein is not detected."),
    ("protein IDs; proteins per cell", "Number of proteins identified in a cell. median_protein_ids: median over "
     "the cells of the group."),
    ("marker_group", "core DCT: Slc12a3 (listed for all DCT cell types). DCT1, DCT2: markers listed for that cell "
     "type only. TAL: Umod (listed for DCT1 in the DCT marker list; treated as a shared, TAL-associated marker) and "
     "Slc12a1 (not in the DCT marker list; negative marker)."),
    ("DCT1-like; DCT2-like", SPLIT),
]


def _count(frame, **sel) -> int:
    mask = None
    for key, value in sel.items():
        hit = frame[key].astype(str).eq(str(value))
        mask = hit if mask is None else mask & hit
    return int(mask.sum())


C, Z = "Control", "Zero-K"
DERIVED, BOTH = ATLAS_LEVEL["derived"], ATLAS_SOURCE["all"]
LETTER = [
    ("control, cluster annotation: Umod median ln(1 + intensity) in detected cells, DCT1", "1a",
     lambda f: _pick(f, "median_DCT1", condition=C, annotation=CLUSTER, scope="detected cells only"), 9.11, 2),
    ("control, cluster annotation: Umod median ln(1 + intensity) in detected cells, DCT2", "1a",
     lambda f: _pick(f, "median_DCT2", condition=C, annotation=CLUSTER, scope="detected cells only"), 8.63, 2),
    ("control, cluster annotation: Umod, Mann-Whitney P (detected cells), as written with two digits", "1a",
     lambda f: f"{float(_pick(f, 'p_value', condition=C, annotation=CLUSTER, scope='detected cells only')):.1e}",
     "3.3e-05", None),
    ("control, cluster annotation: Umod detection, Fisher P", "1a",
     lambda f: _pick(f, "p_value", condition=C, annotation=CLUSTER, scope="detection"), 0.36, 2),
    ("zero-K, cluster annotation: status of the three tests", "1a",
     lambda f: "; ".join(sorted(set(f.loc[f["condition"].eq(Z) & f["annotation"].eq(CLUSTER), "status"]))),
     "does not apply", None),
    ("zero-K, marker-score label (clusters): Umod, Mann-Whitney P (detected cells)", "1a",
     lambda f: _pick(f, "p_value", condition=Z, annotation=SCORE, scope="detected cells only"), 0.086, 3),
    ("control, cluster annotation: Umod DCT2 - DCT1, unadjusted", "2a",
     lambda f: _pick(f, "difference_mean_ln1p_DCT2_minus_DCT1", condition=C, annotation=CLUSTER, marker="Umod"),
     -0.35, 2),
    ("control, cluster annotation: Umod DCT2 - DCT1, adjusted", "2a",
     lambda f: _pick(f, "difference_adjusted_DCT2_minus_DCT1", condition=C, annotation=CLUSTER, marker="Umod"),
     -0.04, 2),
    ("control, cluster annotation: Umod, P after adjustment", "2a",
     lambda f: _pick(f, "p_adjusted", condition=C, annotation=CLUSTER, marker="Umod"), 0.80, 2),
    ("control, marker-score label (clusters): Umod, P after adjustment, as written with two digits", "2a",
     lambda f: f"{float(_pick(f, 'p_adjusted', condition=C, annotation=SCORE, marker='Umod')):.1e}", "4.9e-04", None),
    ("zero-K, marker-score label (clusters): Umod, P after adjustment", "2a",
     lambda f: _pick(f, "p_adjusted", condition=Z, annotation=SCORE, marker="Umod"), 0.77, 2),
    ("control, cluster annotation: Pvalb, P after adjustment, as written with two digits", "2a",
     lambda f: f"{float(_pick(f, 'p_adjusted', condition=C, annotation=CLUSTER, marker='Pvalb')):.1e}", "1.6e-12",
     None),
    ("control, cluster annotation: median proteins per cell, DCT1; DCT2", "2a",
     lambda f: f"{float(_pick(f, 'median_protein_ids_DCT1', condition=C, annotation=CLUSTER, marker='Umod')):g}; "
               f"{float(_pick(f, 'median_protein_ids_DCT2', condition=C, annotation=CLUSTER, marker='Umod')):g}",
     "1851; 2294", None),
    ("zero-K, cluster annotation: markers with status does not apply", "2a",
     lambda f: _count(f, condition=Z, annotation=CLUSTER, status="does not apply"), 11, None),
    ("atlas: Umod, fraction of DCT1-like cells with transcripts", "2b",
     lambda f: _pick(f, "rna_fraction_detected_DCT1_like", marker="Umod"), 0.56, 2),
    ("atlas: Umod, fraction of DCT2-like cells with transcripts", "2b",
     lambda f: _pick(f, "rna_fraction_detected_DCT2_like", marker="Umod"), 0.16, 2),
    *[(f"{cond}: clusters with Umod {cls}", "3a",
       lambda f, cond=cond, cls=cls: _count(f, condition=cond, umod_vs_other_cells=cls), n, None)
      for cond, cls, n in ((C, "increased", 6), (C, "decreased", 4), (C, "not different", 2), (C, "mixed", 1),
                           (Z, "increased", 4), (Z, "decreased", 3), (Z, "not different", 5), (Z, "mixed", 1))],
    ("control: clusters labelled DCT1 through shared markers only, with Umod among the 25 most increased proteins",
     "3a", lambda f: _count(f, condition=C, marker_support=SUPPORT["low_shared_only"],
                            umod_among_25_most_increased=True), 2, None),
    *[(f"total kidney, {cell_type}: cells with {marker} detected; cells", "4a",
       lambda f, cell_type=cell_type, marker=marker: f"{_pick(f, 'n_detected', cell_type=cell_type, marker=marker)}; "
                                                     f"{_pick(f, 'n_cells', cell_type=cell_type, marker=marker)}",
       value, None)
      for cell_type, marker, value in (("TAL", "Umod", "83; 138"), ("DCT", "Umod", "28; 58"),
                                       ("TAL", "Slc12a1", "85; 138"), ("DCT", "Slc12a1", "6; 58"))],
    *[(f"atlas, {group}: fraction of cells with Umod transcripts", "5a",
       lambda f, level=level, group=group: _pick(f, "frac_detected", atlas_groups=level, atlas_cells=BOTH,
                                                 group=group, marker="Umod"), value, 2)
      for level, group, value in ((ATLAS_LEVEL["published"], "CTAL", 0.78), (ATLAS_LEVEL["published"], "MTAL", 0.66),
                                  (ATLAS_LEVEL["published"], "DCT", 0.48), (DERIVED, "DCT1-like", 0.56),
                                  (DERIVED, "DCT2-like", 0.16))],
    ("atlas, nuclei only: nuclei in the DCT2-like group", "5a",
     lambda f: _pick(f, "n_atlas_cells", atlas_groups=DERIVED, atlas_cells=ATLAS_SOURCE["nuclei"], group="DCT2-like",
                     marker="Umod"), 39, None),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    bad = {k: sorted(set(stored[k]["condition"].astype(str))) for k in ("1a", "2a", "3a")
           if set(stored[k]["condition"].astype(str)) != {"Control", "Zero-K"}}
    lib.check(checks, "DCT tables 1a, 2a and 3a: every row belongs to one condition (Control or Zero-K); no row "
              "compares them", bad or "3 tables", "3 tables", not bad)
    labels = set(stored["3a"]["cluster_label"].astype(str))
    lib.check(checks, "table 3a: the cluster label values are unchanged (Unknown stays Unknown in the cells)",
              sorted(labels), "DCT1, DCT2, Unknown", labels == {"DCT1", "DCT2", "Unknown"})
    cells = {c: int(sum(int(v) for v in stored["3a"].loc[stored["3a"]["condition"].eq(c), "n_cells"]))
             for c in ("Control", "Zero-K")}
    lib.check(checks, "table 3a: cells over the clusters = 729 control and 702 zero-K cells", cells,
              {"Control": 729, "Zero-K": 702}, cells == {"Control": 729, "Zero-K": 702})
    kid = stored["4a"]
    n_kid = int(sum(int(v) for v in kid.loc[kid["marker"].eq("Umod"), "n_cells"]))
    lib.check(checks, "table 4a: kidney cells over the cell types = 712; 9 cell types x 3 proteins",
              f"{n_kid} cells; {len(kid)} rows", "712 cells; 27 rows", n_kid == 712 and len(kid) == 27)
    names = sorted(set(stored["5a"]["group"].astype(str)))
    lib.check(checks, "table 5a: atlas groups carry their reader-facing names", names,
              "CNT, CTAL, DCT, DCT-CNT, DCT1-like, DCT2-like, MTAL",
              names == ["CNT", "CTAL", "DCT", "DCT-CNT", "DCT1-like", "DCT2-like", "MTAL"])


WORKBOOK = WorkbookDef(
    label="S-10-b", step=STEP, workbook="report_10_S10b_tables.xlsx",
    title="Supplementary Table S-10-b. Uromodulin (Umod) in DCT1 and DCT2 cells: tests, marker differences between "
          "DCT2 and DCT1, Umod per cluster, and Umod in the total-kidney data and in the Mouse Kidney Atlas",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names "
             "are explained below the list of tables. Detection and intensity of Umod and of the DCT markers by "
             "label are in Supplementary Table S-09-c, sheet 4.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"dct": "combined control + zero-K search (dct_labels, dct_reviewer, dct_rna tables)", "kidney": "kidney_annotation values",
                "atlas": "dct_rna values (Mouse Kidney Atlas; derived DCT split of atlas_dct)"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-10-b (Umod in DCT1 and DCT2).")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
