#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-06-b (marker placement for Figures 2B and 3B), one workbook.

The authors asked on 12 Nov 2026 to assemble the table
("Yes, assemble them"). The response to the reviewers, section 5, fixes its content: one workbook with four sheets, built from six stored
tables (kidney_annotation marker order and detection, kidney_figure2 Figure 2B values, the Figure 3B values of the response to the reviewers, dct_labels DCT marker
order). No new statistics: every value is copied from the stored tables. Layout, assembly and checks are those of
Supplementary Table S-05-b (report_annotation_tables.py, imported; it follows Supplementary Table S-07-d).

What the assembly changes (all recorded per table in the sources table):
- rows: table 1a keeps the rows of the 45-marker layout (version 'full'); the rows of the 19-marker selection, a rule
  that was dropped, are left out; every other table keeps all rows;
- columns: left out are the columns of the dropped 19-marker rule (eligible, rank_in_group, selected, selection),
  constant, duplicate and internal columns (figure file names, internal notes);
- names: internal names are replaced in column names and cell values; kidney cell types carry the display names of
  the revised annotation (DCT1 -> DCT, PC -> CNT/PC; the kidney Unknown keeps its name); the Figure 3B row of the
  DCT cells without a resolved subtype is shown with the display name of the redrawn figure (row_display of the
  values file: "DCT, subtype not resolved"); DCT conditions are Control and Zero-K, side by side;
- final products only (authors, 12 Nov 2026: "The readers will not know the revision process, so we only show the final
  products."): the endothelial category reads "Endothelial" and the Contents term "the kidney list with its
  endothelial row" (both said "added in the revision" before); the shared check finds no "submitted", "submission",
  "re-annotation" or "in the revision" in any cell; the two wording changes are rows of the sources table.

Inputs (read-only): the six tables named in SPECS.
Outputs: results/tables/report/report_06_S06b_tables.xlsx;
         results/tables/report/report_06_s06b_table_{sources,checks}.csv;
         results/tables/report/report_step_06_s06b_table_manifest.json

Command (project root):  python scripts/report_dotplot_tables.py --overwrite
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

import report_annotation_tables as base     # Spec, Config, build, pick and the kidney display names: reused unchanged
import report_pt_tal_tables as s7

Spec, pick, kidney_labels = base.Spec, base.pick, base.kidney_labels
KIDNEY_ANNOTATION, KIDNEY_FIGURE2, DCT_LABELS, REPORT = "kidney_annotation", "kidney_figure2", "dct_labels", "report"
STEP = "06_s06b_table"
TABLE_TITLE = ("Supplementary Table S-06-b. Marker dotplots of Figures 2B and 3B: marker placement, detection and "
               "plotted values")
INTRO = ("Kidney: 712 cells, cell types of the annotation (Figure 2A). DCT: control and zero-potassium "
         "(Zero-K) cells, searched together. All values are observed values; the terms used in the column names are "
         "explained below the list of tables.")
UNRESOLVED = "DCT, subtype not resolved"

GROUPS = {t: t for t in ("Endothelial", "JG", "PCT", "PST", "TAL", "IC-B")} | base.KIDNEY_DISPLAY
CATEGORY = {c: c for c in ("Juxtaglomerular cells", "Proximal convoluted tubule", "Proximal straight tubule",
                           "Thick ascending limb of LOH", "DCT1", "CD-principal cells",
                           "CD-intercalated cells (type B)")} | {
    "Endothelial (kidney_annotation extension; atlas Endo / AVR markers)": "Endothelial"}     # final products only (12 Nov 2026)
PLACEMENT = {"only listing type": "only listing type", "highest mean z among listing types":
             "highest mean z among listing types", "absent from matrix: first listing type":
             "not quantified: first listing type"}
PANELS = {("kidney_figure2_fig2b_marker_dotplot_full", "protein", "all"): "protein (Figure 2B)",
          ("kidney_figure2_fig2b_marker_dotplot_full", "RNA", "all"): "RNA, all cells and nuclei (Figure 2B)",
          ("kidney_figure2_fig2b_marker_dotplot_full_nuclei", "RNA", "nuclei"):
              "RNA, nuclei only (Supplementary Figure S-05-e)"}
LAYOUT = {"panel": "Figure 3B", "workbook": "Supplementary Figure S-06-a"}
OLD_RULE = "column of the 19-marker selection rule, which was dropped (Figure 2B shows all 45 quantified markers)"
OLD_RULE_COLUMNS = ("eligible", "rank_in_group", "selected", "selection")


def fig2b_panels(df: pd.DataFrame, problems: list, edits: list) -> pd.DataFrame:
    """One panel name from the figure file, the panel and the RNA source (three blocks of rows)."""
    keys = list(zip(df["figure"], df["panel"], df["rna_source"]))
    if set(keys) != set(PANELS):
        problems.append(f"unexpected figure / panel / rna_source combinations: {sorted(set(keys) ^ set(PANELS))}")
    df = df.copy()
    df["panel"] = [PANELS.get(k, " / ".join(k)) for k in keys]
    edits.append("panel (from figure / panel / rna_source): "
                 + "; ".join(f"{' / '.join(k)} -> {v}" for k, v in PANELS.items()))
    return df


def fig3b_rows(df: pd.DataFrame, problems: list, edits: list) -> pd.DataFrame:
    """row_display must be the group name, except for the group Unknown (display name of the redrawn figure)."""
    expected = df["group"].map(lambda g: UNRESOLVED if g == "Unknown" else g)
    if not (expected == df["row_display"]).all() or "Unknown" not in set(df["group"]):
        problems.append("row_display is not the group name with Unknown shown as the display name")
    edits.append(f"row (row_display of the values file): Unknown -> {UNRESOLVED}")
    return df


SHEETS = {
    1: ("1 Kidney marker placement", "Kidney: listed markers of the eight cell types and their placement in "
                                     "Figure 2B"),
    2: ("2 Kidney marker detection", "Kidney: detection and specificity of the markers shown in Figure 2B"),
    3: ("3 Figure 2B and 3B values", "Plotted values of Figure 2B (kidney) and Figure 3B (DCT)"),
    4: ("4 DCT marker order", "DCT: marker order of Figure 3B and of Supplementary Figure S-06-a, with detection "
                              "per condition"),
}

SPECS = [
    Spec("1a", KIDNEY_ANNOTATION, "t15_dotplot_marker_order_a_endo", "Kidney marker list of the eight cell types: quantification "
         "and placement in Figure 2B",
         ("One row per listed marker of the eight cell types (75 markers; 45 quantified and shown, 30 not "
          "quantified). column_group: cell type under which the marker is placed; marker_list_category: its "
          "category in Supplementary Table 1; position_in_figure_2b: column, 1 to 45.",
          "listed_for: cell types, among the eight, that list the marker (also_listed_for: those other than the "
          "column group); shared: listed for two or more of them. mean_z_listing_types: group mean in each listing "
          "cell type. top_row: cell type with the highest group mean; column_group_is_top_row."),
         rows=("version", ("full",)), constant=("version", "repeat"), empty=("selection",),
         values={"group": GROUPS, "group_full_name": CATEGORY, "listed_for": kidney_labels,
                 "also_listed_for": kidney_labels, "placement": PLACEMENT, "mean_z_listing_types": kidney_labels,
                 "top_row_all_types": GROUPS},
         rename={"group": "column_group", "group_full_name": "marker_list_category",
                 "group_position": "column_group_position",
                 "workbook_position_in_group": "position_in_marker_list_category", "in_protein_matrix": "quantified",
                 "plotted": "shown_in_figure_2b", "top_row_all_types": "top_row",
                 "assigned_is_top_of_all_rows": "column_group_is_top_row", "position": "position_in_figure_2b"}),
    Spec("2a", KIDNEY_ANNOTATION, "t15_dotplot_main_selection_a_endo", "Kidney: detection and specificity of the 45 quantified "
         "markers in their own cell type",
         ("One row per marker shown in Figure 2B. frac_detected_own: fraction of the cells of its cell type "
          "(column_group; n_cells_own_type cells) with the protein detected; mean_z_own: their mean z.",
          "max_mean_z_other, other_type_with_highest_mean_z: highest mean z among the other seven cell types, and "
          "that cell type. specificity = mean_z_own - max_mean_z_other. own_rank_by_mean_z: rank of the own cell "
          "type among the eight (1 = highest)."),
         drop={c: OLD_RULE for c in OLD_RULE_COLUMNS},
         values={"group": GROUPS, "max_other_group": GROUPS},
         rename={"group": "column_group", "n_cells": "n_cells_own_type",
                 "max_other_group": "other_type_with_highest_mean_z"}),
    Spec("3a", KIDNEY_FIGURE2, "fig2b_full_dotplot_values", "Figure 2B: plotted values, protein and RNA",
         ("One row per panel, row and marker (3 panels x 9 rows x 45 markers). The nuclei-only RNA panel is shown in "
          "Supplementary Figure S-05-e with the same protein panel. n_cells_row: SCP-MS cells, or atlas cells and "
          "nuclei, of the row.",
          "frac_detected: dot size. colour_scaled_0_1: dot colour. shown_grey: no RNA value (no atlas counterpart: "
          "JG, Unknown; gene not in the atlas: Slc22a6, Slc26a4). See Contents (Dot size and colour; RNA panels)."),
         pre=fig2b_panels,
         drop={"figure": "figure file name; the panel name gives the figure",
               "rna_source": "the RNA source is part of the panel name"},
         rename={"bracket": "column_group", "grey": "shown_grey"}),
    Spec("3b", KIDNEY_FIGURE2, "fig2b_full_own_row", "Figure 2B: row with the highest colour for each marker",
         ("One row per marker (position: column in Figure 2B; column_group: its cell type). protein_top_row: row "
          "with the highest protein colour among the nine rows (Unknown included); protein_rows_tied_at_top: rows "
          "with that value; protein_own_row_top: the top row is the marker's own cell type.",
          "rna_ columns: the same for the two RNA panels. 'grey column or own row': no RNA value for the marker or "
          "for its own cell type."),
         same_as={"t15_assigned_is_top_of_all_rows": "protein_own_row_top",
                  "t15_top_row_all_types": "protein_top_row"},
         rename={"own_type": "column_group", "rna_all_top_row": "rna_all_cells_nuclei_top_row",
                 "rna_nuclei_top_row": "rna_nuclei_only_top_row",
                 "rna_all_own_row_top": "rna_all_cells_nuclei_own_row_top",
                 "rna_nuclei_own_row_top": "rna_nuclei_only_own_row_top"}),
    Spec("3c", REPORT, "06_fig3b_dotplot_values", "Figure 3B: plotted values, control DCT cells",
         ("One row per cell group of Figure 3A (row; n_cells) and marker (4 rows x 13 markers; 729 control cells). "
          f"{UNRESOLVED}: cells of clusters without marker support. ProLIF: the cells of the per-cell "
          "proliferation test.",
          "n_detected, frac_detected (dot size): cells with the protein detected. mean_z; colour_scaled_0_1 (dot "
          "colour): mean_z scaled 0 to 1 per marker across the four rows. mean_ln1p_intensity: mean ln(1 + "
          "intensity). column_group: marker group, not a cell label."),
         pre=fig3b_rows, constant=("label_column",),
         drop={"group": "row name of the data file; the row is shown with the display name of the redrawn figure"},
         rename={"row_display": "row", "group_position": "row_position", "marker_group": "column_group",
                 "mean_z_scaled_0_1": "colour_scaled_0_1"}),
    Spec("4a", DCT_LABELS, "marker_order_dct", "DCT marker order of Figure 3B and of Supplementary Figure S-06-a, with "
         "detection per condition",
         ("layout Figure 3B: the 13 markers of Figure 3B in fixed marker groups; Col5a2 was not quantified. layout "
          "Supplementary Figure S-06-a: every marker of the DCT list, under the first cell type that lists it; "
          "Col5a2 and Cenpf were not quantified. position_in_figure: column; shown: quantified in both conditions.",
          "types_listing_in_marker_list: cell types of the DCT list (Slc12a1 is a TAL marker of the kidney list). "
          "_control and _zero_k columns: quantified, and cells with the protein detected, among the 729 control and "
          "the 702 Zero-K cells."),
         constant=("repeat",), same_as={"in_matrix": "plotted"},
         drop={"note": "internal notes on where each marker came from"},
         values={"version": LAYOUT},
         rename={"version": "layout", "plotted_position": "position_in_figure", "group": "column_group",
                 "plotted": "shown", "types_listing_in_workbook": "types_listing_in_marker_list",
                 "in_matrix_ctrl": "quantified_control", "n_cells_detected_ctrl": "n_cells_detected_control",
                 "fraction_cells_detected_ctrl": "fraction_cells_detected_control",
                 "in_matrix_zerok": "quantified_zero_k", "n_cells_detected_zerok": "n_cells_detected_zero_k",
                 "fraction_cells_detected_zerok": "fraction_cells_detected_zero_k"}),
]

TERMS = [
    ("Kidney cell types", "The eight cell types of the annotation (Figure 2A), in nephron order: "
     "Endothelial, JG, PCT, PST, TAL, DCT, CNT/PC, IC-B. Unknown: cells without an assigned cell type (last row of "
     "Figure 2B). DCT and CNT/PC are the names used for the marker-list categories DCT1 and CD-principal cells."),
    ("Marker list; quantified", "Supplementary Table 1: the kidney list with its endothelial row, and the DCT list. "
     "quantified: the protein is in the data."),
    ("Placement in Figure 2B", "Each quantified marker is shown once, under the cell type with the highest group "
     "mean among the cell types that list it; the column groups follow the row order. A marker that was not "
     "quantified is listed under the first cell type that lists it and is not shown."),
    ("mean z", "Group mean of the scaled ln(1 + intensity)."),
    ("Dot size and colour", "frac_detected: fraction of the row's cells with an intensity above zero (RNA: with a "
     "count above zero). colour_scaled_0_1: row mean of the scaled values (protein: per-protein z-scored ln(1 + "
     "intensity), capped at 10), rescaled from 0 to 1 per marker across the rows shown."),
    ("RNA panels", "Mouse Kidney Atlas (single-cell and single-nucleus RNA-seq of healthy mouse kidneys): all cells "
     "and nuclei (Figure 2B), or nuclei only (Supplementary Figure S-05-e). Each SCP-MS cell type is matched to "
     "fixed atlas cell types (Supplementary Table S-08-d). RNA and protein come from different animals and studies."),
    ("Figure 3B rows", "The cell groups of Figure 3A (control DCT cells): DCT1, DCT2 and 'DCT, subtype not "
     "resolved' (cells of clusters without marker support) from the cluster annotation, and ProLIF, the cells of "
     "the per-cell proliferation test, as their own row."),
    ("Figure 3B column groups", "Fixed marker groups: Core (Slc12a3), DCT1, DCT2, TAL (Umod, Slc12a1) and Cycling "
     "(Mki67, Top2a). They are marker groups, not cell labels; Umod, listed for DCT1, is grouped with TAL."),
    ("Control; Zero-K", "DCT cells of mice on the control diet and on the zero-potassium diet, shown side by side."),
]

FORBIDDEN = re.compile(
    r"(?<![A-Za-z0-9])T\d{2}(?![0-9A-Za-z])|t15_|workbook|retired|\bCtrl\b|_ctrl|zerok|figure_3a_group"
    r"|in_matrix|in_protein_matrix|absent from matrix|(?<![A-Za-z0-9])v3(?![A-Za-z0-9])|kidney_|eligible"
    r"|rank_in_group|selected|extension|\bFig\.? ?[23]|cs_|rank-?free|hvg|\bbrief\b|\.X\b", re.I)

TOKEN = r"(?<![A-Za-z0-9/-]){}(?![A-Za-z0-9])"
ROWS_2B = ["Endothelial", "JG", "PCT", "PST", "TAL", "DCT", "CNT/PC", "IC-B", "Unknown"]


def _count(frame: pd.DataFrame, col: str, by: str, order: list[str]) -> str:
    return ", ".join(f"{g} {int(frame.loc[frame[by].eq(g), col].astype(bool).sum())}" for g in order)


NUMBERS = [
    ("listed markers of the eight cell types; quantified; not quantified", "1a",
     lambda f: f"{len(f)}, {int(f['quantified'].astype(bool).sum())}, {int((~f['quantified'].astype(bool)).sum())}",
     "75, 45, 30", None),
    ("markers shown in Figure 2B, each once, in positions 1 to 45", "1a",
     lambda f: sorted(int(v) for v in f.loc[f["shown_in_figure_2b"].astype(bool), "position_in_figure_2b"])
     == list(range(1, 46)) and f["marker"].is_unique, True, None),
    ("markers per column group of Figure 2B", "1a",
     lambda f: _count(f, "shown_in_figure_2b", "column_group", ROWS_2B[:8]),
     "Endothelial 7, JG 1, PCT 12, PST 6, TAL 6, DCT 7, CNT/PC 5, IC-B 1", None),
    ("markers with the highest mean in their own cell type, of the 45 shown", "3b",
     lambda f: f"{int(f['protein_own_row_top'].astype(bool).sum())} of {len(f)}", "39 of 45", None),
    ("the six other markers, with the row that is highest", "3b",
     lambda f: "; ".join(f"{r.marker} ({r.column_group}) {r.protein_top_row}" for r in
                         f[~f["protein_own_row_top"].astype(bool)].itertuples()),
     "Slc9a3 (PCT) TAL; Slc13a3 (PST) PCT; Slc27a2 (PST) PCT; Slc22a6 (PST) PCT; Calb1 (DCT) CNT/PC; "
     "Slc8a1 (DCT) CNT/PC", None),
    ("the same six in the placement table", "1a",
     lambda f: ", ".join(f.loc[f["column_group_is_top_row"].eq(False), "marker"]),
     "Slc9a3, Slc13a3, Slc27a2, Slc22a6, Calb1, Slc8a1", None),
    ("rows of Figure 2B, in order", "3a",
     lambda f: list(pd.unique(f.loc[f["panel"].eq("protein (Figure 2B)"), "row"])), ROWS_2B, None),
    ("plotted values per panel of Figure 2B (9 rows x 45 markers)", "3a",
     lambda f: sorted(f["panel"].value_counts().tolist()), [405, 405, 405], None),
    ("markers detected in fewer than 25% of the cells of their own cell type", "2a",
     lambda f: int((pd.to_numeric(f["frac_detected_own"]) < 0.25).sum()), 11, None),
    ("rows of Figure 3B with their cells", "3c",
     lambda f: ", ".join(f"{r} {n}" for r, n in f[["row", "n_cells"]].drop_duplicates().itertuples(index=False)),
     f"DCT1 362, DCT2 79, ProLIF 37, {UNRESOLVED} 251", None),
    ("markers of Figure 3B per marker group", "3c",
     lambda f: ", ".join(f"{g} {n}" for g, n in f.drop_duplicates("marker")["column_group"].value_counts(
         sort=False).items()), "Core 1, DCT1 3, DCT2 5, TAL 2, Cycling 2", None),
    ("Figure 3B, Mki67 and Top2a detected in the ProLIF row (cells of 37)", "3c",
     lambda f: f"{pick(f, 'n_detected', row='ProLIF', marker='Mki67')}, "
               f"{pick(f, 'n_detected', row='ProLIF', marker='Top2a')}", "8, 13", None),
    ("DCT list markers quantified and shown in Supplementary Figure S-06-a; not quantified", "4a",
     lambda f: (lambda d: f"{int(d['shown'].astype(bool).sum())}; "
                          f"{', '.join(d.loc[~d['shown'].astype(bool), 'marker'])}")(
         f[f["layout"].eq("Supplementary Figure S-06-a")]), "13; Col5a2, Cenpf", None),
]


def extra_checks(frames: dict, stored: dict, cells: dict, checks: list[dict]) -> None:
    """Display names of the kidney cell types and of the Figure 3B rows; the dropped rule's columns are gone."""
    label_cols = {"1a": ("column_group", "listed_for", "also_listed_for", "mean_z_listing_types", "top_row"),
                  "2a": ("column_group", "other_type_with_highest_mean_z"), "3a": ("row", "column_group"),
                  "3b": ("column_group", "protein_top_row", "rna_all_cells_nuclei_top_row",
                         "rna_nuclei_only_top_row")}
    text = [str(v) for key, cols in label_cols.items() for c in cols for v in stored[key][c].dropna()]
    internal = sorted({m for t in text for m in re.findall(TOKEN.format("(?:DCT1|PC)"), t)})
    shown = {name: any(re.search(TOKEN.format(re.escape(name)), t) for t in text)
             for name in ("DCT", "CNT/PC", "Unknown")}
    s7.check(checks, "kidney cell types (tables 1a, 2a, 3a, 3b): display names DCT and CNT/PC, no DCT1 or PC; the "
             "kidney Unknown keeps its name", f"internal names: {internal or 'none'}; found: {shown}",
             "internal names: none; DCT, CNT/PC and Unknown found", not internal and all(shown.values()))
    left = sorted({c for f in stored.values() for c in f.columns if c in OLD_RULE_COLUMNS})
    s7.check(checks, "columns of the dropped 19-marker rule (eligible, rank_in_group, selected, selection) are in "
             "no table; table 1a holds the 45-marker layout only", f"columns left: {left or 'none'}; rows in 1a: "
             f"{len(stored['1a'])}", "none; 75 rows", not left and len(stored["1a"]) == 75)
    rows = list(pd.unique(stored["3c"]["row"]))
    s7.check(checks, "Figure 3B rows carry the display names of the redrawn figure; the label value Unknown is not "
             "shown for the DCT cells", f"rows: {rows}",
             f"DCT1, DCT2, ProLIF, {UNRESOLVED}", rows == ["DCT1", "DCT2", "ProLIF", UNRESOLVED]
             and not any("Unknown" in str(v) for v in stored["3c"].to_numpy().ravel()))
    pairs = [c for c in stored["4a"].columns if c.endswith(("_control", "_zero_k"))]
    s7.check(checks, "DCT marker order: Control and Zero-K columns side by side, no comparison column",
             ", ".join(pairs), "three columns per condition",
             len(pairs) == 6 and sum(c.endswith("_control") for c in pairs) == 3
             and not any(re.search(r"diff|ratio|fold|vs", c) for c in stored["4a"].columns))


# Wording changed at the authors' request, 12 Nov 2026 ("The readers will not know the revision process, so we only show
# the final products."; "Yes, remove them"): rows of the sources record.
ON_REQUEST = "wording changed at the authors' request, 12 Nov 2026: no reference to the revision process"
WORDING = [
    {"status": ON_REQUEST, "title": "table 1a, marker_list_category (7 cells): 'Endothelial (row added in the "
     "revision)' -> 'Endothelial'"},
    {"status": ON_REQUEST, "title": "Contents, term 'Marker list; quantified': 'the kidney list with the endothelial "
     "row added in the revision' -> 'the kidney list with its endothelial row'"},
]

CONFIG = base.Config(step=STEP, workbook="report_06_S06b_tables.xlsx", table_id="S-06-b", title=TABLE_TITLE,
                     intro=INTRO, sheets=SHEETS, specs=SPECS, terms=TERMS, forbidden=FORBIDDEN,
                     script="report_dotplot_tables.py", numbers=NUMBERS, extra_checks=extra_checks,
                     wording=WORDING)


if __name__ == "__main__":
    base.build(CONFIG, base.parse_args("the response to the reviewers: Supplementary Table S-06-b, marker placement for Figures 2B "
                                       "and 3B."))
