#!/usr/bin/env python
"""The response to the reviewers: evidence tables for the Editors' comment 3 (is uromodulin restricted to DCT1 and absent
from DCT2 in the Pvalb-GFP-sorted DCT data?).

(authors, 12 Nov 2026: "Yes, go ahead" on the the response to the reviewers plan).
No new statistics: every value is read from a stored table of kidney_annotation, dct_labels, dct_reviewer, dct_rna or the response to the reviewers and is merged,
reshaped or counted here. The script writes the tables that Supplementary Figure S17 and Supplementary Table S12 are built from, and it checks every number of the the response to the reviewers letter draft against its source.

Tables written (results/tables/report/):
- report_10_umod_by_cluster.csv: Umod in the cluster test of the annotation (each Leiden cluster against all
  other cells; dct_labels), with the cluster label, its marker support and the median number of proteins per cell.
  umod_vs_other_cells: increased (q < 0.05, AUC > 0.5, mean difference > 0; the rule's own definition), decreased
  (q < 0.05, AUC < 0.5, mean difference < 0), not different (q >= 0.05), mixed (q < 0.05, AUC and mean difference
  of opposite sign).
- report_10_marker_dct2_vs_dct1.csv: DCT2 - DCT1 difference of the DCT list markers, Umod and Slc12a1, per
  condition and label set (dct_rna step 5 values, long format): detection, difference of mean ln(1 + intensity) with
  non-detected = 0 (Mann-Whitney, BH q over all compared proteins), and the difference adjusted for ln(proteins
  per cell) and sort plate (OLS). Rows without a DCT2 group say 'does not apply'.
- report_10_marker_rna_dct_split.csv: the same markers in the Mouse Kidney Atlas, DCT cells split into the
  derived DCT1-like and DCT2-like groups (atlas_dct rule; not atlas labels).
- report_10_letter_numbers.csv: every number of the letter draft (response file, section 2) with its source.
- report_10_umod_evidence_checks.csv; report_step_10_umod_evidence_manifest.json.

Control and zero-K are analysed within condition and listed side by side; nothing here compares them.

Inputs (read-only): see SOURCES. Command (project root):
  python scripts/report_umod_evidence.py --overwrite
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
import math

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5

TASK = "report"
STEP = "10_umod_evidence"
CONDITIONS = ("Ctrl", "ZeroK")
PRIMARY = {"Ctrl": "cs_ctrl_deg_label", "ZeroK": "cs_zerok_pipeline_label"}
SECOND = {"Ctrl": "cs_ctrl_cluster_label", "ZeroK": "cs_zerok_cluster_label"}
Q_DE = t5.Q_DE
# DCT list markers of the comparison (dct_labels marker order: panel groups; Trpm7 is in the workbook version only),
# with Umod and Slc12a1 as the TAL group. Mki67 and Top2a (cycling) belong to the response to the reviewers and are left out.
MARKERS = (("Slc12a3", "core DCT"), ("Pvalb", "DCT1"), ("Egf", "DCT1"), ("Ptgfr", "DCT1"), ("Trpm7", "DCT1"),
           ("Calb1", "DCT2"), ("Slc8a1", "DCT2"), ("S100g", "DCT2"), ("Klk1", "DCT2"), ("Umod", "TAL"),
           ("Slc12a1", "TAL"))
KIDNEY_ANNOTATION = "results/tables/kidney_annotation/kidney_annotation_"
DCT_LABELS = "results/tables/dct_labels/dct_labels_"
DCT_REVIEWER = "results/tables/dct_reviewer/dct_reviewer_"
DCT_RNA = "results/tables/dct_rna/dct_rna_"
REPORT = "results/tables/report/report_"
SOURCES = {
    "canonical": DCT_LABELS + "dct_canonical_labels.csv",
    "de": DCT_LABELS + "deg_de_tested.csv",
    "clusters": DCT_LABELS + "deg_cluster_annotation.csv",
    "order": DCT_LABELS + "marker_order_dct.csv",
    "by_label": DCT_REVIEWER + "umod_markers_by_label.csv",
    "codet": DCT_REVIEWER + "umod_slc12a1_codetection.csv",
    "tests": DCT_REVIEWER + "umod_dct1_vs_dct2_tests.csv",
    "coverage": DCT_REVIEWER + "task3_individual_marker_coverage.csv",
    "fc": DCT_RNA + "fc_genes.csv",
    "rna": DCT_RNA + "dotplot_rna_values.csv",
    "kidney": KIDNEY_ANNOTATION + "t15_dotplot_values_a_endo.csv",
    "per_cell": REPORT + "09_negative_marker_per_cell.csv",
}


def sig2(x: float) -> str:
    """A number with two significant digits in e-notation (P and q values of the letter)."""
    return f"{float(x):.1e}"


class Letter:
    """Numbers of the letter draft: observed from the source, expected as written in the letter."""

    def __init__(self, checks: t5.Checks):
        self.rows, self.checks = [], checks

    def add(self, statement: str, observed, expected, source: str, decimals: int | None = None) -> None:
        if decimals is None:
            ok = observed == expected
            shown = observed
        else:
            ok = abs(float(observed) - float(expected)) <= 0.5 * 10 ** -decimals + 1e-12
            shown = round(float(observed), decimals + 2)
        self.rows.append({"statement": statement, "value_in_letter": expected, "value_from_source": shown,
                          "source": source, "equal": bool(ok)})
        self.checks.add(f"letter: {statement}", shown, bool(ok), str(expected))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: evidence tables for Umod in DCT1 and DCT2.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")
    src = {k: pd.read_csv(run.use_input(cc.PROJECT_ROOT / p, k), float_precision="round_trip", low_memory=False)
           for k, p in SOURCES.items()}
    letter = Letter(checks)

    # ------------------------------------------------------------------------------------------------------------
    # Per-cell table of the response to the reviewers (labels, proteins per cell, Umod and Pvalb) against the canonical labels and dct_reviewer
    # ------------------------------------------------------------------------------------------------------------
    pc, can = src["per_cell"], src["canonical"].set_index("cell_key")
    same_primary = all(
        (can.loc[pc.loc[pc["condition"] == c, "cell_key"], PRIMARY[c]].astype(str).to_numpy()
         == pc.loc[pc["condition"] == c, "primary_label"].astype(str).to_numpy()).all() for c in CONDITIONS)
    same_second = all(
        (can.loc[pc.loc[pc["condition"] == c, "cell_key"], SECOND[c]].astype(str).to_numpy()
         == pc.loc[pc["condition"] == c, "second_label"].astype(str).to_numpy()).all() for c in CONDITIONS)
    same_ids = bool((can.loc[pc["cell_key"], "n_detected"].to_numpy() == pc["n_detected_proteins"].to_numpy()).all())
    checks.add("per-cell table: cluster annotation, marker-score label (clusters) and proteins per cell equal the "
               "dct_labels canonical table, matched on cell_key", f"{len(pc)} cells; {same_primary}; {same_second}; {same_ids}",
               same_primary and same_second and same_ids and len(pc) == 1431)
    bl = src["by_label"]
    worst, n_cmp = 0.0, 0
    for role, col in (("primary", "primary_label"), ("second", "second_label")):
        for (cond, lab), d in pc.groupby(["condition", col]):
            for marker in ("Umod", "Pvalb"):
                row = bl[(bl["condition"] == cond) & (bl["label_role"] == role) & (bl["label"] == lab)
                         & (bl["marker"] == marker)]
                det = d[f"det_{marker}"].astype(bool)
                med = float(d.loc[det, f"ln1p_{marker}"].median())
                ok = (len(row) == 1 and int(row["n_cells"].iloc[0]) == len(d)
                      and int(row["n_detected"].iloc[0]) == int(det.sum()))
                worst = max(worst, abs(med - float(row["median_ln1p_detected"].iloc[0])) if ok else np.inf)
                n_cmp += 1
    checks.add("per-cell table: cells, detected cells and median ln(1 + intensity) of Umod and Pvalb per label equal "
               "the dct_reviewer table (largest absolute difference of the medians)", f"{n_cmp} label x marker rows; {worst:.2e}",
               worst < 1e-9, "< 1e-9")

    # ------------------------------------------------------------------------------------------------------------
    # Table: Umod in the cluster test, by cluster
    # ------------------------------------------------------------------------------------------------------------
    de = src["de"]
    de = de[(de["protein"] == "Umod") & (de["clustering"] == "r1.0")]
    an = src["clusters"]
    an = an[an["resolution"] == 1.0][["condition", "cluster", "deg_label", "deg_confidence", "n_cells",
                                     "median_protein_ids", "top10_de"]]
    bc = de.merge(an, on=["condition", "cluster"], how="outer", validate="one_to_one", indicator=True)
    checks.add("by cluster: Umod was tested in every cluster of both conditions (13 + 13), and the cluster sizes of "
               "the two dct_labels tables agree", f"{len(bc)} rows; {dict(bc['_merge'].value_counts())}",
               len(bc) == 26 and (bc["_merge"] == "both").all() and (bc["n_cells_cluster"] == bc["n_cells"]).all())
    sig = bc["q_bh"] < Q_DE
    up = sig & (bc["auc"] > 0.5) & (bc["mean_diff_ln1p"] > 0)
    down = sig & (bc["auc"] < 0.5) & (bc["mean_diff_ln1p"] < 0)
    bc["umod_vs_other_cells"] = np.select([up, down, ~sig], ["increased", "decreased", "not different"], "mixed")
    checks.add("by cluster: 'increased' = the rule's own flag (ranked among the significantly increased proteins)",
               f"{int(up.sum())} clusters increased; {int(bc['rank_up'].notna().sum())} with a rank",
               bool((bc["rank_up"].notna() == up).all()))
    bc["umod_in_10_most_increased"] = [("Umod" in str(t).split(";")) for t in bc["top10_de"]]
    by_cluster = bc.rename(columns={
        "deg_label": "cluster_label", "deg_confidence": "marker_support", "pct_detected_cluster":
        "umod_fraction_detected_cluster", "pct_detected_rest": "umod_fraction_detected_other_cells",
        "mean_ln1p_cluster": "umod_mean_ln1p_cluster", "mean_ln1p_rest": "umod_mean_ln1p_other_cells",
        "mean_diff_ln1p": "umod_mean_difference_ln1p", "rank_up": "umod_rank_among_increased",
        "in_top": "umod_among_25_most_increased"})[
        ["condition", "cluster", "cluster_label", "marker_support", "n_cells", "median_protein_ids",
         "umod_fraction_detected_cluster", "umod_fraction_detected_other_cells", "umod_mean_ln1p_cluster",
         "umod_mean_ln1p_other_cells", "umod_mean_difference_ln1p", "auc", "p_value", "q_bh", "umod_vs_other_cells",
         "umod_rank_among_increased", "umod_among_25_most_increased", "umod_in_10_most_increased"]]
    by_cluster = by_cluster.sort_values(["condition", "cluster"]).reset_index(drop=True)
    run.table(by_cluster, "10_umod_by_cluster")
    for cond in CONDITIONS:
        d = by_cluster[by_cluster["condition"] == cond]
        got = {lab: int(n) for lab, n in d.groupby("cluster_label")["n_cells"].sum().items()}
        checks.add(f"by cluster, {cond}: cells per cluster label = canonical counts", got,
                   got == dict(pc[pc["condition"] == cond]["primary_label"].value_counts()))

    def cluster_text(cond: str, cls: str) -> str:
        d = by_cluster[(by_cluster["condition"] == cond) & (by_cluster["umod_vs_other_cells"] == cls)]
        if d.empty:
            return "0 clusters"
        labs = "; ".join(f"{lab} {n}" for lab, n in d["cluster_label"].value_counts().sort_index().items())
        return (f"{len(d)} clusters ({labs}); median proteins per cell {d['median_protein_ids'].min():g} to "
                f"{d['median_protein_ids'].max():g}")

    src_bc = f"{REPORT}10_umod_by_cluster.csv (from {DCT_LABELS}deg_de_tested.csv and {DCT_LABELS}deg_cluster_annotation.csv)"
    letter.add("control: clusters with Umod increased against the other cells", cluster_text("Ctrl", "increased"),
               "6 clusters (DCT1 3; Unknown 3); median proteins per cell 490 to 1632", src_bc)
    letter.add("control: clusters with Umod decreased", cluster_text("Ctrl", "decreased"),
               "4 clusters (DCT1 2; DCT2 1; Unknown 1); median proteins per cell 2241 to 3910", src_bc)
    letter.add("control: clusters with Umod not different", cluster_text("Ctrl", "not different"),
               "2 clusters (DCT1 1; DCT2 1); median proteins per cell 1879 to 2164.5", src_bc)
    letter.add("control: clusters with a mixed result", cluster_text("Ctrl", "mixed"),
               "1 clusters (Unknown 1); median proteins per cell 318 to 318", src_bc)
    letter.add("zero-K: clusters with Umod increased", cluster_text("ZeroK", "increased"),
               "4 clusters (DCT1 2; Unknown 2); median proteins per cell 524 to 1220", src_bc)
    letter.add("zero-K: clusters with Umod decreased", cluster_text("ZeroK", "decreased"),
               "3 clusters (DCT1 2; Unknown 1); median proteins per cell 2405 to 3007", src_bc)
    letter.add("zero-K: clusters with Umod not different", cluster_text("ZeroK", "not different"),
               "5 clusters (DCT1 3; Unknown 2); median proteins per cell 1214 to 1971", src_bc)
    letter.add("zero-K: clusters with a mixed result", cluster_text("ZeroK", "mixed"),
               "1 clusters (Unknown 1); median proteins per cell 221 to 221", src_bc)
    shared = by_cluster[by_cluster["marker_support"] == "low_shared_only"]
    letter.add("clusters labelled DCT1 through shared markers only: clusters, cells, Umod among the 25 most "
               "increased proteins (control; zero-K)",
               "; ".join(f"{c}: {len(d)} clusters, {int(d['n_cells'].sum())} cells, {int(d['umod_among_25_most_increased'].sum())} with Umod"
                         for c, d in shared.groupby("condition")),
               "Ctrl: 2 clusters, 95 cells, 2 with Umod; ZeroK: 2 clusters, 95 cells, 2 with Umod", src_bc)

    # ------------------------------------------------------------------------------------------------------------
    # Table: DCT2 - DCT1 difference per marker (dct_rna step 5), long format; RNA of the derived atlas split
    # ------------------------------------------------------------------------------------------------------------
    fc = src["fc"]
    order = src["order"]
    listing = (order.drop_duplicates("marker").set_index("marker")["types_listing_in_workbook"].fillna("")
               .astype(str).to_dict())
    rows, rna_rows = [], []
    for cond in CONDITIONS:
        sub_pc = pc[pc["condition"] == cond]
        for role, col_name, col in (("primary", PRIMARY[cond], "primary_label"), ("second", SECOND[cond],
                                                                               "second_label")):
            n1, n2 = int((sub_pc[col] == "DCT1").sum()), int((sub_pc[col] == "DCT2").sum())
            ids1 = float(sub_pc.loc[sub_pc[col] == "DCT1", "n_detected_proteins"].median()) if n1 else np.nan
            ids2 = float(sub_pc.loc[sub_pc[col] == "DCT2", "n_detected_proteins"].median()) if n2 else np.nan
            for marker, group in MARKERS:
                r = fc[(fc["condition"] == cond) & (fc["protein"] == marker)]
                if len(r) != 1:
                    raise SystemExit(f"{cond} {marker}: {len(r)} rows in the dct_rna fold-change table")
                r = r.iloc[0]
                applies = n2 > 0
                rows.append({
                    "condition": cond, "label_role": role, "label_column": col_name, "marker": marker,
                    "marker_group": group, "listed_for_in_dct_list": listing.get(marker, ""),
                    "status": "compared" if applies else "does not apply",
                    "reason": "" if applies else f"no DCT2 cell under {col_name}",
                    "n_DCT1": n1, "n_DCT2": n2, "median_protein_ids_DCT1": ids1, "median_protein_ids_DCT2": ids2,
                    "fraction_detected_DCT1": r[f"prot_det_DCT1_{role}"], "fraction_detected_DCT2":
                    r[f"prot_det_DCT2_{role}"], "difference_mean_ln1p_DCT2_minus_DCT1": r[f"prot_diff_{role}"],
                    "q_mannwhitney_bh": r[f"prot_mw_q_{role}"], "difference_adjusted_DCT2_minus_DCT1":
                    r[f"prot_diff_adjusted_{role}"], "p_adjusted": r[f"prot_adjusted_p_{role}"]})
                if cond == "Ctrl" and role == "primary":
                    rna_rows.append({"marker": marker, "marker_group": group, "atlas_gene": r["atlas_gene"],
                                     "rna_fraction_detected_DCT1_like": r["rna_det_DCT1_atlas_dct"],
                                     "rna_fraction_detected_DCT2_like": r["rna_det_DCT2_atlas_dct"],
                                     "rna_difference_DCT2_like_minus_DCT1_like": r["rna_diff_DCT2_minus_DCT1"],
                                     "rna_difference_cells_only": r["rna_diff_cells_thr0.25"],
                                     "rna_difference_nuclei_only": r["rna_diff_nuclei_thr0.25"]})
    md = pd.DataFrame(rows)
    rna_split = pd.DataFrame(rna_rows)
    run.table(md, "10_marker_dct2_vs_dct1")
    run.table(rna_split, "10_marker_rna_dct_split")
    na_ok = md.loc[md["status"] == "does not apply", ["fraction_detected_DCT2", "difference_mean_ln1p_DCT2_minus_DCT1",
                                                     "p_adjusted"]].isna().all().all()
    checks.add("marker differences: rows = 11 markers x 2 conditions x 2 label sets; 'does not apply' only for the "
               "zero-K cluster annotation (no DCT2 cluster), with empty values",
               f"{len(md)} rows; does not apply: "
               f"{dict(md[md['status'] == 'does not apply'].groupby(['condition', 'label_role']).size())}",
               len(md) == 44 and bool(na_ok)
               and set(map(tuple, md.loc[md["status"] == "does not apply", ["condition", "label_role"]].to_numpy()))
               == {("ZeroK", "primary")})
    same_rna = all(np.allclose(fc[(fc["condition"] == "Ctrl") & (fc["protein"] == m)][c].to_numpy(),
                               fc[(fc["condition"] == "ZeroK") & (fc["protein"] == m)][c].to_numpy())
                   for m, _g in MARKERS for c in ("rna_det_DCT1_atlas_dct", "rna_det_DCT2_atlas_dct", "rna_diff_DCT2_minus_DCT1"))
    checks.add("marker differences: the atlas values of a marker are the same in the control and zero-K rows of the "
               "dct_rna table", same_rna, bool(same_rna))
    groups_ok = all(
        (g == "TAL" and m in ("Umod", "Slc12a1")) or (g == "core DCT" and listing[m] == "DCT1;DCT2;ProLIF")
        or listing[m] == g for m, g in MARKERS)
    checks.add("marker groups: DCT1 and DCT2 markers are listed for that cell type only; Slc12a3 for all three; Umod "
               "(listed for DCT1) and Slc12a1 (not in the DCT list) form the TAL group",
               "; ".join(f"{m}: {listing[m] or 'not listed'}" for m, _g in MARKERS), bool(groups_ok))

    def fcv(cond: str, role: str, marker: str, col: str) -> float:
        d = md[(md["condition"] == cond) & (md["label_role"] == role) & (md["marker"] == marker)]
        return float(d[col].iloc[0])

    src_md = f"{REPORT}10_marker_dct2_vs_dct1.csv (from {DCT_RNA}fc_genes.csv)"
    letter.add("control, cluster annotation: median proteins per cell, DCT1; DCT2",
               f"{fcv('Ctrl', 'primary', 'Umod', 'median_protein_ids_DCT1'):g}; "
               f"{fcv('Ctrl', 'primary', 'Umod', 'median_protein_ids_DCT2'):g}", "1851; 2294",
               f"{REPORT}09_negative_marker_per_cell.csv; dct_rna README section 5")
    letter.add("control, cluster annotation: Umod DCT2 - DCT1, difference of mean ln(1 + intensity)",
               fcv("Ctrl", "primary", "Umod", "difference_mean_ln1p_DCT2_minus_DCT1"), -0.35, src_md, 2)
    letter.add("control, cluster annotation: Umod DCT2 - DCT1 after adjustment",
               fcv("Ctrl", "primary", "Umod", "difference_adjusted_DCT2_minus_DCT1"), -0.04, src_md, 2)
    letter.add("control, cluster annotation: Umod, P after adjustment",
               fcv("Ctrl", "primary", "Umod", "p_adjusted"), 0.80, src_md, 2)
    letter.add("control, marker-score labels of the clusters: Umod DCT2 - DCT1 after adjustment; P",
               f"{fcv('Ctrl', 'second', 'Umod', 'difference_adjusted_DCT2_minus_DCT1'):.2f}; "
               f"{sig2(fcv('Ctrl', 'second', 'Umod', 'p_adjusted'))}", "-0.45; 4.9e-04", src_md)
    letter.add("zero-K, marker-score labels of the clusters: Umod DCT2 - DCT1 after adjustment; P",
               f"{fcv('ZeroK', 'second', 'Umod', 'difference_adjusted_DCT2_minus_DCT1'):.2f}; "
               f"{fcv('ZeroK', 'second', 'Umod', 'p_adjusted'):.2f}", "-0.03; 0.77", src_md)
    letter.add("control, cluster annotation: Pvalb DCT2 - DCT1, unadjusted; q",
               f"{fcv('Ctrl', 'primary', 'Pvalb', 'difference_mean_ln1p_DCT2_minus_DCT1'):.2f}; "
               f"{sig2(fcv('Ctrl', 'primary', 'Pvalb', 'q_mannwhitney_bh'))}", "-1.80; 2.3e-11", src_md)
    letter.add("control, cluster annotation: Pvalb DCT2 - DCT1 after adjustment; P",
               f"{fcv('Ctrl', 'primary', 'Pvalb', 'difference_adjusted_DCT2_minus_DCT1'):.2f}; "
               f"{sig2(fcv('Ctrl', 'primary', 'Pvalb', 'p_adjusted'))}", "-2.44; 1.6e-12", src_md)
    adjusted = {m: (fcv("Ctrl", "primary", m, "difference_adjusted_DCT2_minus_DCT1"),
                    fcv("Ctrl", "primary", m, "p_adjusted")) for m, _g in MARKERS}
    lower = sorted(m for m, (d, p) in adjusted.items() if p < 0.05 and d < 0)
    higher = sorted(m for m, (d, p) in adjusted.items() if p < 0.05 and d > 0)
    neither = sorted(m for m, (d, p) in adjusted.items() if p >= 0.05)
    letter.add("control, cluster annotation, after adjustment: markers lower in DCT2 (P < 0.05)", ", ".join(lower),
               "Egf, Ptgfr, Pvalb, Slc12a3, Trpm7", src_md)
    letter.add("control, cluster annotation, after adjustment: markers higher in DCT2 (P < 0.05)", ", ".join(higher),
               "Klk1, S100g, Slc12a1", src_md)
    letter.add("control, cluster annotation, after adjustment: markers not different (P >= 0.05)", ", ".join(neither),
               "Calb1, Slc8a1, Umod", src_md)
    letter.add("control, after adjustment: largest P among Pvalb, Egf, Ptgfr and Trpm7",
               sig2(max(adjusted[m][1] for m in ("Pvalb", "Egf", "Ptgfr", "Trpm7"))), "3.1e-07", src_md)
    letter.add("control, after adjustment: P of S100g; Klk1; Calb1; Slc8a1; Slc12a3",
               "; ".join((f"{adjusted[m][1]:.4f}" if adjusted[m][1] >= 0.001 else sig2(adjusted[m][1]))
                         for m in ("S100g", "Klk1", "Calb1", "Slc8a1", "Slc12a3")),
               "0.0031; 0.0023; 0.3226; 0.6534; 7.9e-10", src_md)

    # ------------------------------------------------------------------------------------------------------------
    # Detection by label (dct_reviewer), tests (dct_reviewer), totals
    # ------------------------------------------------------------------------------------------------------------
    def det(cond: str, role: str, label: str, marker: str) -> str:
        r = bl[(bl["condition"] == cond) & (bl["label_role"] == role) & (bl["label"] == label)
               & (bl["marker"] == marker)].iloc[0]
        return f"{int(r['n_detected'])} of {int(r['n_cells'])}"

    src_bl = f"{DCT_REVIEWER}umod_markers_by_label.csv"
    for cond, role, label, marker, expected in (
            ("Ctrl", "primary", "DCT1", "Umod", "375 of 382"), ("Ctrl", "primary", "DCT2", "Umod", "85 of 85"),
            ("Ctrl", "primary", "Unknown", "Umod", "252 of 262"), ("ZeroK", "primary", "DCT1", "Umod", "447 of 449"),
            ("ZeroK", "primary", "Unknown", "Umod", "243 of 253"), ("Ctrl", "second", "DCT1", "Umod", "456 of 460"),
            ("Ctrl", "second", "DCT2", "Umod", "124 of 126"), ("ZeroK", "second", "DCT1", "Umod", "467 of 469"),
            ("ZeroK", "second", "DCT2", "Umod", "113 of 114"), ("Ctrl", "primary", "DCT1", "Pvalb", "329 of 382"),
            ("Ctrl", "primary", "DCT2", "Pvalb", "61 of 85"), ("Ctrl", "primary", "Unknown", "Pvalb", "182 of 262"),
            ("ZeroK", "primary", "DCT1", "Pvalb", "396 of 449"), ("ZeroK", "primary", "Unknown", "Pvalb", "144 of 253"),
            ("Ctrl", "primary", "DCT1", "Slc12a3", "348 of 382"), ("Ctrl", "primary", "DCT2", "Slc12a3", "78 of 85")):
        letter.add(f"{cond}, {role} label {label}: cells with {marker} detected", det(cond, role, label, marker),
                   expected, src_bl)
    cov = src["coverage"].set_index(["condition", "marker"])
    for cond, marker, expected in (("Ctrl", "Umod", "712 of 729"), ("ZeroK", "Umod", "690 of 702"),
                                   ("Ctrl", "Pvalb", "572 of 729"), ("ZeroK", "Pvalb", "540 of 702")):
        n = int(cov.loc[(cond, marker), "n_cells_detected"])
        total = int((pc["condition"] == cond).sum())
        letter.add(f"{cond}: cells with {marker} detected, all cells", f"{n} of {total}", expected,
                   f"{DCT_REVIEWER}task3_individual_marker_coverage.csv")
    codet = src["codet"]
    for cond, expected in (("Ctrl", "75; 0; 637"), ("ZeroK", "39; 0; 651")):
        d = codet[(codet["condition"] == cond) & (codet["label_role"] == "primary")]
        letter.add(f"{cond}: cells with Slc12a1; with Slc12a1 and no Umod; with Umod and no Slc12a1",
                   f"{int(d['n_slc12a1_detected'].sum())}; {int(d['n_slc12a1_without_umod'].sum())}; "
                   f"{int(d['n_umod_without_slc12a1'].sum())}", expected, f"{DCT_REVIEWER}umod_slc12a1_codetection.csv")
    tests = src["tests"]

    def test(cond: str, role: str, scope: str):
        return tests[(tests["condition"] == cond) & (tests["label_role"] == role) & (tests["scope"] == scope)].iloc[0]

    src_t = f"{DCT_REVIEWER}umod_dct1_vs_dct2_tests.csv"
    t = test("Ctrl", "primary", "detected cells only")
    letter.add("control, cluster annotation: Umod median ln(1 + intensity) in detected cells, DCT1; DCT2",
               f"{t['median_DCT1']:.2f}; {t['median_DCT2']:.2f}", "9.11; 8.63", src_t)
    letter.add("control, cluster annotation: Umod, ratio of the medians DCT1 / DCT2 (exp of the difference)",
               math.exp(float(t["median_difference"])), 1.6, src_t, 1)
    letter.add("control, cluster annotation: Umod, Mann-Whitney P (detected cells); q",
               f"{sig2(t['p_value'])}; {sig2(t['p_fdr_bh'])}", "3.3e-05; 9.8e-05", src_t)
    letter.add("control, cluster annotation: Umod detection DCT1 vs DCT2, Fisher P",
               float(test("Ctrl", "primary", "detection")["p_value"]), 0.36, src_t, 2)
    t = test("Ctrl", "second", "detected cells only")
    letter.add("control, marker-score labels of the clusters: Umod medians DCT1; DCT2; ratio; Mann-Whitney P",
               f"{t['median_DCT1']:.2f}; {t['median_DCT2']:.2f}; {math.exp(float(t['median_difference'])):.1f}; "
               f"{sig2(t['p_value'])}", "9.15; 8.31; 2.3; 1.3e-10", src_t)
    t = test("ZeroK", "second", "detected cells only")
    letter.add("zero-K, marker-score labels of the clusters: Umod medians DCT1; DCT2; Mann-Whitney P",
               f"{t['median_DCT1']:.2f}; {t['median_DCT2']:.2f}; {float(t['p_value']):.3f}", "9.23; 9.29; 0.086", src_t)
    zk = tests[(tests["condition"] == "ZeroK") & (tests["label_role"] == "primary")]
    letter.add("zero-K, cluster annotation: status of the DCT1 vs DCT2 tests", "; ".join(sorted(set(zk["status"]))),
               "does not apply", src_t)
    med = bl[(bl["condition"] == "Ctrl") & (bl["label_role"] == "primary") & (bl["marker"] == "Pvalb")
             ].set_index("label")["median_ln1p_detected"]
    letter.add("control, cluster annotation: Pvalb, ratio of the medians DCT1 / DCT2 in detected cells",
               math.exp(float(med["DCT1"] - med["DCT2"])), 2.9, src_bl, 1)
    n = pc[pc["condition"] == "Ctrl"]["primary_label"].value_counts()
    letter.add("control: DCT2 cells among the cells with a subtype; percentage; percentage of all cells",
               f"{int(n['DCT2'])} of {int(n['DCT1'] + n['DCT2'])}; {100 * n['DCT2'] / (n['DCT1'] + n['DCT2']):.0f}; "
               f"{100 * n['DCT2'] / n.sum():.0f}", "85 of 467; 18; 12", f"{DCT_LABELS}dct_canonical_labels.csv")
    for cond, expected in (("Ctrl", "98; 100; 96; 86; 72"), ("ZeroK", "100; 96; 88; 57")):
        d = bl[(bl["condition"] == cond) & (bl["label_role"] == "primary")].set_index(["marker", "label"])
        keys = ([("Umod", "DCT1"), ("Umod", "DCT2"), ("Umod", "Unknown"), ("Pvalb", "DCT1"), ("Pvalb", "DCT2")]
                if cond == "Ctrl" else [("Umod", "DCT1"), ("Umod", "Unknown"), ("Pvalb", "DCT1"), ("Pvalb", "Unknown")])
        letter.add(f"{cond}: percentages detected ({'; '.join(f'{m} {lab}' for m, lab in keys)})",
                   "; ".join(f"{100 * float(d.loc[k, 'detection_fraction']):.0f}" for k in keys), expected, src_bl)

    # ------------------------------------------------------------------------------------------------------------
    # Total kidney (kidney_annotation values) and Mouse Kidney Atlas (dct_rna values)
    # ------------------------------------------------------------------------------------------------------------
    kid = src["kidney"]
    checks.add("kidney: label column of the values table", "; ".join(sorted(set(kid["label_column"]))),
               set(kid["label_column"]) == {"kidney_t15_label"})
    ku = kid[kid["marker"] == "Umod"].set_index("group")
    ks = kid[kid["marker"] == "Slc12a1"].set_index("group")
    src_k = f"{KIDNEY_ANNOTATION}t15_dotplot_values_a_endo.csv"
    letter.add("total kidney: Umod detected, TAL; DCT (label DCT1 of the table)",
               f"{int(ku.loc['TAL', 'n_detected'])} of {int(ku.loc['TAL', 'n_cells'])}; "
               f"{int(ku.loc['DCT1', 'n_detected'])} of {int(ku.loc['DCT1', 'n_cells'])}", "83 of 138; 28 of 58", src_k)
    other = ku.drop(index=["TAL", "DCT1"])
    letter.add("total kidney: Umod detected in the other cells; all cells",
               f"{int(other['n_detected'].sum())} of {int(other['n_cells'].sum())}; "
               f"{int(ku['n_detected'].sum())} of {int(ku['n_cells'].sum())}", "15 of 516; 126 of 712", src_k)
    letter.add("total kidney: Slc12a1 detected, TAL; DCT",
               f"{int(ks.loc['TAL', 'n_detected'])} of {int(ks.loc['TAL', 'n_cells'])}; "
               f"{int(ks.loc['DCT1', 'n_detected'])} of {int(ks.loc['DCT1', 'n_cells'])}", "85 of 138; 6 of 58", src_k)
    rna = src["rna"]

    def frac(level: str, source: str, group: str, marker: str) -> float:
        return float(rna[(rna["level"] == level) & (rna["rna_source"] == source) & (rna["group"] == group)
                         & (rna["marker"] == marker)]["frac_detected"].iloc[0])

    src_r = f"{DCT_RNA}dotplot_rna_values.csv"
    letter.add("atlas: percentage of cells with Umod transcripts, cortical TAL; medullary TAL; DCT",
               "; ".join(f"{100 * frac('published', 'all', g, 'Umod'):.0f}" for g in ("CTAL", "MTAL", "DCT")),
               "78; 66; 48", src_r)
    letter.add("atlas, derived split, all cells: Umod, DCT1-like; DCT2-like (percent)",
               "; ".join(f"{100 * frac('derived', 'all', g, 'Umod'):.0f}" for g in ("DCT1 (atlas_dct)", "DCT2 (atlas_dct)")),
               "56; 16", src_r)
    letter.add("atlas, derived split, nuclei only: Umod, DCT1-like; DCT2-like (percent)",
               "; ".join(f"{100 * frac('derived', 'nuclei', g, 'Umod'):.0f}" for g in ("DCT1 (atlas_dct)", "DCT2 (atlas_dct)")),
               "38; 38", src_r)
    letter.add("atlas, derived split, all cells: Pvalb, DCT1-like; DCT2-like (percent)",
               "; ".join(f"{100 * frac('derived', 'all', g, 'Pvalb'):.0f}" for g in ("DCT1 (atlas_dct)", "DCT2 (atlas_dct)")),
               "45; 20", src_r)
    n_at = rna[(rna["level"] == "derived") & (rna["marker"] == "Umod")].set_index(["rna_source", "group"])["n_atlas_cells"]
    letter.add("atlas, derived split: cells in DCT1-like; DCT2-like; nuclei in DCT2-like",
               f"{int(n_at[('all', 'DCT1 (atlas_dct)')])}; {int(n_at[('all', 'DCT2 (atlas_dct)')])}; "
               f"{int(n_at[('nuclei', 'DCT2 (atlas_dct)')])}", "8864; 2752; 39", src_r)
    same = all(abs(float(rna_split.set_index("marker").loc[m, f"rna_fraction_detected_{a}_like"])
                   - frac("derived", "all", f"{a} (atlas_dct)", m)) < 1e-12 for m in ("Umod", "Pvalb") for a in ("DCT1", "DCT2"))
    checks.add("atlas: the detected fractions of the derived split in the dct_rna fold-change table equal those of the "
               "dct_rna dotplot table (Umod, Pvalb)", same, bool(same))

    run.table(letter.frame(), "10_letter_numbers")
    run.table(checks.frame(), f"{STEP}_checks")
    n_fail = int((~checks.frame()["pass"]).sum())
    print(f"Checks: {len(checks.rows) - n_fail} passed, {n_fail} failed; letter numbers: {len(letter.rows)}")
    run.finish({"markers": [m for m, _g in MARKERS], "q_threshold_cluster_test": Q_DE,
                "label_columns": {"primary": PRIMARY, "second": SECOND}, "letter_numbers": len(letter.rows),
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
