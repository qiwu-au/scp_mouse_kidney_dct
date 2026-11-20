#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S1 (preprocessing sensitivity), one workbook with six sheets.

The authors asked on 12 Nov 2026 to assemble Supplementary Tables S9, S3 and S1 ("Yes, assemble them"). The response to the reviewers, section 5, fixes the content of S1: (1) all
pipelines in the three datasets; (2) ranges by group; (3) detection of the selected proteins; (4) normalization
factors and per-cell intensity descriptors; (5) kidney per-label purity, silhouette and per-resolution clustering
scores under the revised annotation; (6) depth-regressed embeddings. It is built from the 11 stored tables named
there. No new statistics: every value is copied.

What the assembly changes (all recorded per table in the sources table):
- rows: none dropped (the kidney rows of these tables are scored against kidney_t15_label only; the DCT rows come
  from the combined control + zero-K search, dct_reviewer);
- names: kidney / Ctrl / ZeroK -> Total kidney / DCT Control / DCT Zero-K; the label columns are named cell types
  (kidney_t15_label), cluster annotation (cs_ctrl_deg_label, cs_zerok_pipeline_label) and marker-score label
  (clusters) (cs_ctrl_cluster_label, cs_zerok_cluster_label); `variant_id` -> `pipeline`; the group codes are
  replaced by their stored text; "the primary label" -> "the cluster annotation" in the measures of sheet 6;
  kidney display names DCT1 -> DCT, PC -> CNT/PC;
- final products only (authors, 12 Nov 2026: "The readers will not know the revision process, so we only show the
  final products"): the pipeline that the paper uses is the "main pipeline" (stored names: baseline, submitted),
  its graph the "graph of the main analysis"; `is_baseline` -> `is_main_pipeline`, `submitted` -> `main_pipeline`,
  `n_above_submitted` -> `n_above_main`, `n_equal_submitted` -> `n_equal_main`; no cell, column name or sheet
  name refers to the earlier manuscript, and a check enforces it;
- columns: `label_role`, and `dataset` / `label_column` where constant, are dropped after a check; the `source`
  column of the depth table (internal file names) and the group code of the group table are dropped.

The stored measure text "revised rule and markers" (table 6a, two rows) reads "annotation rule and markers" (authors, 12 Nov 2026).

Workbook: results/tables/report/report_03_S03b_tables.xlsx (sheet Contents, then sheets 1-6), in the
layout of Supplementary Table S7 (shared code: report_table_lib.py).

Inputs (read-only): the 11 tables named in SPECS.
Outputs: the workbook; results/tables/report/report_03_s03b_table_{sources,checks}.csv;
         results/tables/report/report_step_03_s03b_table_manifest.json

Command (project root):  python scripts/report_preprocessing_tables.py --overwrite
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
from report_table_lib import (CONDITION, DATASET, KIDNEY_LABEL, LABEL_SET, MARKER_SCORE, UNKNOWN_NOTE, Spec,
                                       WorkbookDef, _pick)

STEP = "03_s03b_table"
REPORT = "results/tables/report/report_03_"
DCT_REVIEWER_SETS = "results/tables/dct_reviewer/dct_reviewer_prep_feature_set_summary.csv"
PREPROCESSING_SETS = "results/tables/preprocessing/preprocessing_kidney_feature_set_summary.csv"

KIDNEY, CONTROL, ZEROK = "Total kidney", "DCT Control", "DCT Zero-K"
CT, CA, MS = "cell types", "cluster annotation", "marker-score label (clusters)"
MAIN, TOTAL, MEDIAN = "none__z10__all", "total__z10__all", "median__z10__all"        # pipeline ids
HVG = "highly variable proteins (500, 1,000, 2,000)"
MAIN_GROUP = "main pipeline"
STORED_NAME = ((r"^submitted pipeline", MAIN_GROUP),)            # stored text of a group -> name in the workbook
GROUP = {"submitted": MAIN_GROUP, "cellwise_scaled_all": "cell-wise normalization, scaled, all proteins",
         "unscaled_all": "no per-protein scaling, all proteins", "hvg": HVG,
         "highcov50": "proteins detected in >= 50% of cells", "pcs_or_k": "main pipeline with other PCs or k"}
GRAPH = ((r"the primary label", "the cluster annotation"), (r"submitted graph", "graph of the main analysis"),
         (r"revised rule and markers", "annotation rule and markers"))   # authors, 12 Nov 2026: "Yes, change it"
MAIN_GRAPH = "graph of the main analysis"
BY_LABEL_SET = dict(strict=("dataset", "label_column"), drop_dependent=(("label_role", "label_column"),))
LEFT_OUT = ("Labels with fewer than 10 cells and the label Unknown are left out of the means over labels. Total "
            f"kidney: Unknown = cells without a marker-based call. DCT, cluster annotation: {UNKNOWN_NOTE}. DCT, "
            "marker-score label (clusters): its own label Unknown.",)

SHEETS = {
    1: ("1 Pipelines", "The main pipeline and 32 alternatives: definition and scores", LEFT_OUT),
    2: ("2 Ranges by group", "Ranges of the scores by group of pipelines", LEFT_OUT),
    3: ("3 Detection", "Detection of the proteins selected by each pipeline", ()),
    4: ("4 Normalization and intensity", "Normalization factors and per-cell intensity descriptors", ()),
    5: ("5 Kidney per-label scores", "Total kidney: scores per cell type and per clustering resolution",
        ("Unknown: cells without a marker-based call. all cells: every cell, whatever its cell type.",)),
    6: ("6 Depth-regressed", "Embeddings after regressing out the number of identified proteins", ()),
}

SPECS = [
    Spec("1a", REPORT + "pipeline_summary.csv", "All pipelines: definition and scores in the three datasets",
         ("33 pipelines per dataset and label set: the main pipeline (is_main_pipeline) and 32 alternatives.",
          "Columns: see Contents (main pipeline; pipeline; group; Clustering scores; macro_ columns; Depth; Ranks)."),
         values={"dataset": DATASET, "label_column": LABEL_SET, "group": GROUP},
         strict=("dataset", "label_column", "group"), drop_dependent=(("label_role", "label_column"),),
         rename={"label_column": "label_set", "variant_id": "pipeline", "is_baseline": "is_main_pipeline"}),
    Spec("2a", REPORT + "group_summary.csv", "Ranges of the scores by group of pipelines, against the value of the main "
         "pipeline",
         ("One row per dataset, label set, group and score (metric). main_pipeline: value of the main pipeline; "
          "min, max: range over the n_pipelines pipelines of the group.",
          "n_above_main, n_equal_main: pipelines above, or equal to, the main pipeline's value; pipeline_at_max: the "
          "pipeline with the largest value."),
         values={"dataset": DATASET, "label_column": LABEL_SET}, value_re={"group_display": STORED_NAME},
         drop=("group",),
         drop_reason="group: internal code of the group; its text (group_display) is kept under the name group",
         rename={"label_column": "label_set", "group_display": "group", "submitted": "main_pipeline",
                 "n_above_submitted": "n_above_main", "n_equal_submitted": "n_equal_main"}, **BY_LABEL_SET),
    Spec("3a", REPORT + "detection_summary.csv", "Detection of the selected proteins: the three datasets",
         ("One row per dataset, normalization and protein set (features). Columns: see Contents (Detection "
          "columns).",),
         values={"dataset": DATASET}, strict=("dataset",)),
    Spec("3b", PREPROCESSING_SETS, "Total kidney: protein sets of the pipelines",
         ("As table 3a, with the values over all retained proteins (_all) and the overlap with the set selected "
          "without per-cell normalization (jaccard_vs_none_same_features).",),
         drop_const=("dataset",)),
    Spec("3c", DCT_REVIEWER_SETS, "DCT, Control and Zero-K: protein sets of the pipelines",
         ("As table 3b, for each condition.",),
         drop_const=("dataset",), values={"condition": CONDITION}, strict=("condition",)),
    Spec("4a", REPORT + "normalization_factors.csv", "Total-intensity normalization factors",
         ("Factor of a cell: median total intensity of the cells divided by the cell's own total intensity. "
          "fold_range = factor_max / factor_min.",),
         values={"dataset": DATASET}, strict=("dataset",)),
    Spec("4b", REPORT + "intensity_descriptors.csv", "Per-cell intensity descriptors and their correlation with the "
         "number of identified proteins",
         ("descriptor: ln_total_intensity (ln of the summed intensity of a cell); median_ln_intensity_detected "
          "(median ln intensity of the proteins detected in the cell); median_ln_intensity_core (median ln "
          "intensity of the core proteins, detected in at least 90% of cells; n_core_proteins).",
          "median, min, max: over the cells; range_fold_linear = exp(max - min). spearman_rho_vs_ln_n_detected: "
          "Spearman correlation with ln(identified proteins per cell), with its 95% CI and p value."),
         values={"dataset": DATASET}, strict=("dataset",)),
    Spec("5a", REPORT + "kidney_t15_knn_purity.csv", "Total kidney: kNN purity per cell type and pipeline",
         ("observed_same_label_fraction: mean fraction of the 10 nearest neighbors (principal components of the "
          "pipeline) with the same cell type; expected_random: expected for random neighbors; enrichment = "
          "observed / expected.",
          "null_mean, null_q975, p_empirical_ge: permutations of the cell types."),
         drop_const=("dataset", "label_column"), values={"label": KIDNEY_LABEL},
         rename={"label": "cell_type", "variant_id": "pipeline"}, kidney=True),
    Spec("5b", REPORT + "kidney_t15_silhouette.csv", "Total kidney: silhouette width per cell type and pipeline",
         ("mean_silhouette: mean silhouette width of the cells of the cell type, in the principal components of "
          "the pipeline.",),
         drop_const=("dataset", "label_column"), values={"label": KIDNEY_LABEL},
         rename={"label": "cell_type", "variant_id": "pipeline"}, kidney=True),
    Spec("5c", REPORT + "kidney_t15_clustering_metrics.csv", "Total kidney: Leiden clusters against the cell types, per "
         "pipeline and resolution",
         ("Columns: see Contents (Clustering scores).",),
         drop_const=("dataset", "label_column"), rename={"variant_id": "pipeline"}, kidney=True),
    Spec("6a", REPORT + "depth_regressed.csv", "Graph of the main analysis against the depth-regressed graph",
         ("graph of the main analysis: neighbor graph of the main pipeline; depth-regressed graph: after regressing "
          "out the number of identified proteins per cell. ln(IDs): ln of the identified proteins per cell.",
          "Total kidney: the annotation rule was applied to both graphs. DCT: the stored labels were scored on "
          "both embeddings and were not derived again."),
         values={"dataset": DATASET}, strict=("dataset",), value_re={"measure": GRAPH}, drop=("source",),
         drop_reason="source: names of internal files"),
]

TERMS = [
    ("Datasets", "Total kidney: 712 cells, 4,054 proteins. DCT Control: 729 cells, 6,078 proteins. DCT Zero-K: 702 "
     "cells, 5,953 proteins (control or zero-potassium diet for 4 days; one pool of 3 mice per condition). Each "
     "dataset was analysed on its own; the two conditions are shown side by side and were not tested against each "
     "other."),
    ("main pipeline", "The preprocessing used for the analyses of the paper (pipeline none__z10__all): no per-cell "
     "normalization, ln(1 + x), each protein scaled to zero mean and unit variance (values capped at 10), all "
     "retained proteins, 20 principal components, 10 neighbors. graph of the main analysis: its neighbor graph."),
    ("label_set", "The labels that the clusters and embeddings are scored against. cell types: total kidney, the "
     "cell types of Figure 2A. cluster annotation: DCT, cell types assigned to Leiden clusters by differential "
     "expression of marker proteins, for each condition on its own. marker-score label (clusters): DCT, see "
     "Marker-score label. All labels were derived from clusters on the graph of the main analysis, so the "
     "agreement scores favor the main pipeline."),
    ("Marker-score label", f"{MARKER_SCORE} Its labels are DCT1, DCT2 and ProLIF, and for clusters also Unknown."),
    ("pipeline", "normalization__scaling__features. normalization: none (main pipeline); total (each cell "
     "multiplied by the median total intensity divided by its own total, before ln(1 + x)); median (ln intensity "
     "centered on the cell's median over detected proteins). scaling: z10 (each protein scaled to zero mean and "
     "unit variance, values capped at 10; main pipeline) or unscaled. features: all (all retained proteins; main "
     "pipeline); hvg500, hvg1000, hvg2000 (highly variable proteins, Scanpy flavor 'seurat'); highcov50 (proteins "
     "detected in at least 50% of cells). __pcsN, __kN: the matrix of the main analysis with N principal "
     "components (n_pcs; main pipeline 20) or N neighbors (k_neighbors; main pipeline 10). n_proteins: proteins "
     "used."),
    ("group; block", "group: the main pipeline and five groups of alternatives: cell-wise (per-cell) "
     "normalization, scaled, all proteins (2 pipelines); no per-protein scaling, all proteins (3); highly variable "
     "proteins (18); proteins detected in >= 50% of cells (3); main pipeline with other PCs or k (6; other numbers "
     "of principal components or neighbors). block: part of the design (core_factorial: normalization x scaling x "
     "features; pca_test; knn_test; highcov_features)."),
    ("Clustering scores", "Leiden clusters at resolutions 0.5, 1, 2 and 3 (seed 42) against the labels. ari: "
     "adjusted Rand index; nmi: normalized mutual information; majority_accuracy: cells whose label is the most "
     "frequent label of their cluster. best_: the best of the four resolutions (best_ari_resolution, "
     "best_nmi_resolution)."),
    ("macro_ columns", "kNN label purity (k = 10) and silhouette width by label, in the principal components of "
     "the pipeline, averaged over the labels with at least 10 cells, Unknown left out. "
     "macro_knn_purity_enrichment: purity divided by the purity expected for random neighbors. _incl_unknown "
     "(total kidney): with Unknown included."),
    ("Depth", "rho_pc1_ln_n_detected, rho_pc2_ln_n_detected: Spearman correlation of PC1 and PC2 with ln(identified "
     "proteins per cell); max_abs_rho_pc12: the larger absolute value."),
    ("Ranks", "Among the 33 pipelines of a dataset and label set; 1 is first. label_preservation_rank: rank of the "
     "mean rank on best_ari, best_nmi, macro_knn_purity_enrichment and macro_silhouette. depth_independence_rank: "
     "rank of max_abs_rho_pc12, smallest first. combined_rank: rank of the mean of these two ranks."),
    ("Detection columns", "Detection of a protein: fraction of cells with an intensity above zero. "
     "median_detection_rate_selected: median detection of the selected proteins. "
     "fraction_selected_detected_lt50pct, _ge90pct: selected proteins detected in fewer than 50%, or in at least "
     "90%, of cells."),
    ("identified proteins; IDs; n_detected", "Protein groups with an intensity above zero in a cell."),
]


def scores(f: pd.DataFrame, dataset: str, label_set: str, pipeline: str) -> str:
    sel = dict(dataset=dataset, label_set=label_set, pipeline=pipeline)
    return " / ".join(f"{_pick(f, c, **sel):.3f}" for c in ("best_nmi", "macro_knn_purity", "max_abs_rho_pc12"))


def ranges(f: pd.DataFrame, dataset: str, label_set: str) -> str:
    sel = dict(dataset=dataset, label_set=label_set, group=HVG)
    return " / ".join(f"{_pick(f, 'min', metric=m, **sel):.3f}-{_pick(f, 'max', metric=m, **sel):.3f}"
                      for m in ("best_nmi", "macro_knn_purity", "max_abs_rho_pc12"))


def measure(f: pd.DataFrame, dataset: str, text: str) -> float:
    return float(_pick(f, "value", dataset=dataset, measure=text))


PRIMARY = ((KIDNEY, CT), (CONTROL, CA), (ZEROK, CA))
RHO = "largest |rho| of PC1 or PC2 with ln(IDs), depth-regressed graph"
BEST = "best NMI of Leiden 0.3, 0.5, 0.8, 1.0, 3.0 with the cluster annotation, "
LETTER = [
    ("main pipeline: NMI / kNN purity / largest |rho|, kidney; control; zero-K", "1a",
     lambda f: "; ".join(scores(f, d, s, MAIN) for d, s in PRIMARY),
     "0.764 / 0.875 / 0.994; 0.561 / 0.857 / 0.999; 0.416 / 0.968 / 0.999", None),
    ("total-intensity normalization: NMI / kNN purity / largest |rho|", "1a",
     lambda f: "; ".join(scores(f, d, s, TOTAL) for d, s in PRIMARY),
     "0.645 / 0.850 / 0.986; 0.579 / 0.857 / 0.997; 0.292 / 0.961 / 0.997", None),
    ("median normalization: NMI / kNN purity / largest |rho|", "1a",
     lambda f: "; ".join(scores(f, d, s, MEDIAN) for d, s in PRIMARY),
     "0.660 / 0.865 / 0.996; 0.494 / 0.852 / 1.000; 0.303 / 0.965 / 1.000", None),
    ("control: ARI and silhouette of total-intensity normalization against the main pipeline", "1a",
     lambda f: ", ".join(f"{_pick(f, c, dataset=CONTROL, label_set=CA, pipeline=p):.3f}"
                         for c in ("best_ari", "macro_silhouette") for p in (TOTAL, MAIN)),
     "0.317, 0.311, 0.120, 0.113", None),
    ("main pipeline: label-preservation, depth-independence and combined rank, kidney; control; zero-K", "1a",
     lambda f: "; ".join(", ".join(f"{_pick(f, c, dataset=d, label_set=s, pipeline=MAIN):g}"
                                   for c in ("label_preservation_rank", "depth_independence_rank", "combined_rank"))
                         for d, s in PRIMARY), "1, 26, 3; 2, 24, 5; 2, 26, 8", None),
    ("highly variable proteins (18 pipelines): ranges of NMI / kNN purity / largest |rho|", "2a",
     lambda f: "; ".join(ranges(f, d, s) for d, s in PRIMARY),
     "0.410-0.603 / 0.487-0.804 / 0.654-0.981; 0.211-0.334 / 0.644-0.816 / 0.893-0.966; "
     "0.165-0.240 / 0.832-0.927 / 0.900-0.979", None),
    ("highly variable proteins: pipelines above the main pipeline's NMI, kNN purity and silhouette", "2a",
     lambda f: "; ".join(", ".join(str(_pick(f, "n_above_main", dataset=d, label_set=s, group=HVG, metric=m))
                                   for m in ("best_nmi", "macro_knn_purity", "macro_silhouette"))
                         for d, s in PRIMARY), "0, 0, 5; 0, 0, 15; 0, 0, 5", None),
    ("main pipeline's value in the group table = its value in the pipeline table (NMI), kidney; control; zero-K",
     "2a", lambda f: "; ".join(f"{_pick(f, 'main_pipeline', dataset=d, label_set=s, group=HVG, metric='best_nmi'):.3f}"
                               for d, s in PRIMARY), "0.764; 0.561; 0.416", None),
    ("median detection of all proteins, % of cells: kidney, control, zero-K", "3a",
     lambda f: ", ".join(f"{100 * _pick(f, 'median_detection_rate_selected', dataset=d, normalization='none', features='all'):.1f}"
                         for d in (KIDNEY, CONTROL, ZEROK)), "8.6, 18.7, 16.4", None),
    ("median detection of the highly variable sets, % of cells: smallest and largest per dataset", "3a",
     lambda f: "; ".join((lambda v: f"{100 * v.min():.1f}-{100 * v.max():.1f}")(
         f.loc[f["dataset"].eq(d) & f["features"].astype(str).str.startswith("hvg"),
               "median_detection_rate_selected"].astype(float)) for d in (KIDNEY, CONTROL, ZEROK)),
     "2.0-3.9; 4.8-8.5; 3.3-6.0", None),
    ("fold range of the total-intensity factors: kidney, control, zero-K", "4a",
     lambda f: ", ".join(f"{_pick(f, 'fold_range', dataset=d):.1f}" for d in (KIDNEY, CONTROL, ZEROK)),
     "542.6, 141.4, 392.9", None),
    ("rho with identified proteins: total intensity; core proteins; detected proteins", "4b",
     lambda f: "; ".join(", ".join(f"{_pick(f, 'spearman_rho_vs_ln_n_detected', dataset=d, descriptor=x):.3f}"
                                   for d in (KIDNEY, CONTROL, ZEROK))
                         for x in ("ln_total_intensity", "median_ln_intensity_core", "median_ln_intensity_detected")),
     "0.639, 0.640, 0.675; 0.494, 0.646, 0.723; -0.876, -0.996, -0.992", None),
    ("kidney, main pipeline: kNN purity of PCT cells", "5a",
     lambda f: f"{_pick(f, 'observed_same_label_fraction', pipeline=MAIN, cell_type='PCT'):.3f}", "0.911", None),
    ("kidney, main pipeline: NMI and majority accuracy at resolution 2", "5c",
     lambda f: f"{_pick(f, 'nmi', pipeline=MAIN, resolution=2):.3f}, "
               f"{_pick(f, 'majority_accuracy', pipeline=MAIN, resolution=2):.3f}", "0.764, 1.000", None),
    ("largest |rho| after regressing out protein number: kidney, control, zero-K", "6a",
     lambda f: ", ".join(f"{measure(f, d, RHO):.3f}" for d in (KIDNEY, CONTROL, ZEROK)), "0.016, 0.214, 0.336", None),
    ("kidney: cell types on the two graphs, and cells with the same label", "6a",
     lambda f: f"{measure(f, KIDNEY, f'cell types, {MAIN_GRAPH}, Leiden 2.0, annotation rule and markers'):g}, "
               f"{measure(f, KIDNEY, 'cell types, depth-regressed graph, Leiden 2.0, annotation rule and markers'):g}, "
               f"{measure(f, KIDNEY, 'fraction of cells with the same label on the two graphs'):.3f}",
     "8, 8, 0.802", None),
    ("best NMI, depth-regressed graph against graph of the main analysis: control; zero-K", "6a",
     lambda f: "; ".join(f"{measure(f, d, BEST + 'depth-regressed graph'):.3f}, "
                         f"{measure(f, d, BEST + MAIN_GRAPH):.3f}" for d in (CONTROL, ZEROK)),
     "0.448, 0.561; 0.269, 0.416", None),
]


def extra_checks(frames: dict, stored: dict, checks: list) -> None:
    a, g = stored["1a"], stored["2a"]
    blocks = a.groupby(["dataset", "label_set"]).agg(n=("pipeline", "nunique"), main=("is_main_pipeline", "sum"))
    want = {(KIDNEY, CT), (CONTROL, CA), (CONTROL, MS), (ZEROK, CA), (ZEROK, MS)}
    is_main = a["is_main_pipeline"].astype(bool)
    lib.check(checks, "table 1a: 33 pipelines per dataset and label set, one of them the main pipeline "
              f"({MAIN}); kidney under the cell types of the revised annotation only; DCT under the two label sets",
              "; ".join(f"{d}, {s}: {r['n']} ({int(r['main'])} main)" for (d, s), r in blocks.iterrows()),
              "5 blocks of 33 (1 main)",
              set(blocks.index) == want and bool((blocks["n"] == 33).all()) and bool((blocks["main"] == 1).all())
              and set(a.loc[is_main, "pipeline"]) == {MAIN} and set(a.loc[is_main, "group"]) == {MAIN_GROUP})
    src = lib.read_source(cc.PROJECT_ROOT / (REPORT + "group_summary.csv"))
    pairs = {k: re.sub(*STORED_NAME[0], v)
             for k, v in src[["group", "group_display"]].drop_duplicates().itertuples(index=False, name=None)}
    mine = {k: v for k, v in GROUP.items() if k != "submitted"}
    lib.check(checks, "tables 1a and 2a: the group names of table 1a are the stored texts of the group table (with "
              "'main pipeline' for the pipeline of the paper)", pairs, mine,
              pairs == mine and set(g["group"]) | {MAIN_GROUP} == set(a["group"]))
    keys = ["normalization", "features", "median_detection_rate_selected", "fraction_selected_detected_lt50pct",
            "fraction_selected_detected_ge90pct"]
    parts = [stored["3b"].assign(dataset=KIDNEY)] + [stored["3c"][stored["3c"]["condition"].eq(c)].assign(dataset=d)
                                                     for c, d in (("Control", CONTROL), ("Zero-K", ZEROK))]
    both = stored["3a"].merge(pd.concat(parts)[["dataset", *keys]], on=["dataset", "normalization", "features"],
                              suffixes=("", "_sets"), validate="one_to_one")
    largest = max(float((both[c].astype(float) - both[f"{c}_sets"].astype(float)).abs().max()) for c in keys[2:])
    lib.check(checks, "tables 3a, 3b and 3c agree: detection of the selected proteins (the stored tables differ only "
              "by the rounding of their number text)", f"{len(both)} rows; largest difference {largest:.1e}",
              "45 rows; difference <= 1e-15", len(both) == 45 and largest <= 1e-15)
    types = sorted({str(v) for k in ("5a", "5b") for v in stored[k]["cell_type"]})
    expected = sorted(["PCT", "PST", "TAL", "DCT", "CNT/PC", "IC-B", "JG", "Unknown", "Endothelial", "all cells"])
    lib.check(checks, "tables 5a and 5b: the cell types of the revised annotation, with the display names", types,
              expected, types == expected)
    names = sorted({str(v) for f in stored.values() if "label_set" in f.columns for v in f["label_set"]})
    lib.check(checks, "shared vocabulary: label sets 'cell types', 'cluster annotation' and 'marker-score label "
              "(clusters)'; the definition of the marker-score label is on the Contents sheet", names,
              sorted([CT, CA, MS]), names == sorted([CT, CA, MS]) and any(t == "Marker-score label" for t, _ in TERMS))


WORKBOOK = WorkbookDef(
    label="S-03-b", step=STEP, workbook="report_03_S03b_tables.xlsx",
    title="Supplementary Table S-03-b. Preprocessing sensitivity analyses: the main pipeline and 32 alternatives "
          "in the total-kidney and the DCT data",
    subtitle="SCP-MS: single-cell proteomics. All values are observed values; the terms used in the column names "
             "are explained below the list of tables.",
    sheets=SHEETS, specs=SPECS, terms=TERMS, letter=LETTER, extra_checks=extra_checks,
    parameters={"kidney": "revised annotation only (kidney_t15_label; display names DCT, CNT/PC)",
                "dct": "combined control + zero-K search (dct_reviewer grid); cluster annotation and marker-score label "
                       "(clusters)", "unknown_note": UNKNOWN_NOTE,
                "vocabulary": "final products only (authors, 12 Nov 2026): baseline / submitted pipeline = main "
                              "pipeline; submitted graph = graph of the main analysis; cs_<cond>_cluster_label = "
                              "marker-score label (clusters)",
                "not_renamed": "none (table 6a: 'revised rule and markers' now reads 'annotation rule and markers')"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Table S-03-b (preprocessing).")
    cc.add_common_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    lib.build(WORKBOOK, parse_args())
