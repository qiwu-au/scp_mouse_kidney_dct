#!/usr/bin/env python
"""The response to the reviewers: kidney mitochondrial figure under the revised annotation, and flag counts by metric.

(reviewer 1, comment 1; the earlier analysis point 7). Reruns no analysis: it
reads the per-cell and test tables of qc_mito, kidney_annotation (addendum) and dct_reviewer, joins the kidney per-cell metrics to
`kidney_t15_label` on cell_id, and counts existing flags. New numbers are counts of existing flags by metric.

Why it exists:
1. The qc_mito kidney figures use the submitted labels; no figure shows these metrics under `kidney_t15_label`, the one
   kidney annotation of the revision (authors, 30 Oct 2026). Supplementary Figure S3 is drawn here.
2. qc_mito, kidney_annotation and dct_reviewer count mito flags per label, not per metric. The letter needs the split: which metric raised the
   flags, and how many of those cells are also low on depth.

Supplementary Figure S3 (one page, 180 mm; no in-figure title or footnote, the legend text is in the the response to the reviewers file; rows
shared by all panels; display names DCT and CNT/PC):
A. MitoCarta fraction of summed intensity per cell type (cells, median and IQR) and the global top-10% cutoff.
B. Percentage of each cell type above that cutoff (kidney_annotation addendum simulation; no cell excluded).
C. Spearman rho of the intensity fraction with log10 protein IDs within each cell type.
D. Spearman rho of the intensity fraction with log10 total intensity (the fraction's denominator).
E. Spearman rho of the ID fraction with log10 protein IDs.
C-E: 95% CI from the kidney_annotation table; filled = BH q < 0.05 within the label column; IC-B not evaluated (n < 10).

Colour: one hue for the data marks (dataviz reference blue); filled or hollow carries the q < 0.05 state, so nothing
depends on colour alone. Neutral grey for single cells.

Inputs (read-only): results/tables/qc_mito/qc_mito_{kidney_qc_per_cell,all_raw_cells_mito,mito_coverage,
                    removed_vs_retained}.csv;
                    results/tables/kidney_annotation/kidney_annotation_cluster_labels_per_cell_a_endo.csv,
                    kidney_annotation_addendum_t03_{outlier_thresholds,outlier_counts,qc_summary_by_label,
                    global_cutoff_simulation,global_cutoff_tests,within_type_depth_correlation,
                    variance_explained}.csv, kidney_annotation_sensitivity_vs_random.csv;
                    results/tables/dct_reviewer/dct_reviewer_{mito_qc_per_cell,mito_all_raw_cells,
                    mito_outlier_counts,exclusion_sets,exclusion_vs_random}.csv
Outputs: results/figures/report/report_02_kidney_mito.{pdf,png};
         results/tables/report/report_02_mito_flags_by_metric.csv,
         report_02_mito_panels_{values,checks,figure_checks}.csv;
         results/tables/report/report_step_02_mito_panels_manifest.json

Command (project root):  python scripts/report_mito_kidney_figure.py
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

import numpy as np
import pandas as pd

import scp_common as cc
import atlas_dct_lib as T8

plt = T8.setup_matplotlib()

TASK = "report"
STEP = "02_mito_panels"
FIGURE = "02_kidney_mito"
QC_MITO = cc.OUT_ROOTS["tables"] / "qc_mito"
KIDNEY_ANNOTATION = cc.OUT_ROOTS["tables"] / "kidney_annotation"
DCT_REVIEWER = cc.OUT_ROOTS["tables"] / "dct_reviewer"
LABEL = "kidney_t15_label"
DISPLAY = {"DCT1": "DCT", "PC": "CNT/PC"}                 # authors, 30 Oct 2026
EXPECTED_KIDNEY_ANNOTATION = {"PCT": 160, "PST": 158, "TAL": 138, "DCT1": 58, "PC": 38, "JG": 24, "Endothelial": 19, "IC-B": 8,
                "Unknown": 109}                           # the project notes
EXPECTED_DCT = {"Ctrl": 729, "ZeroK": 702}                # the project notes
INTENSITY = ("mitocarta_intensity_fraction", "oxphos_intensity_fraction")
ID_METRIC = "mitocarta_id_fraction"
MITO = (INTENSITY[0], ID_METRIC, INTENSITY[1])
DEPTH = ("log10_total_intensity", "log10_n_detected")
M, FONT, INK = T8.M, T8.FONT, T8.INK
BLUE = "#2a78d6"
ROW_H = 5.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: kidney mito figure and flag counts by metric.")
    cc.add_common_args(parser)
    return parser.parse_args()


def kidney_flags(cells: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    """Per-cell outlier state under kidney_t15_label, from the stored kidney_annotation addendum thresholds (qc_mito rule)."""
    flags = pd.DataFrame(index=cells.index)
    for metric in MITO + DEPTH:
        t = thr[thr["metric"] == metric].set_index("label")
        lower, upper = cells[LABEL].map(t["lower"]), cells[LABEL].map(t["upper"])
        x = cells[metric]
        flags[metric] = np.where(lower.isna(), "not_evaluated", np.where(x > upper, "high",
                                                                         np.where(x < lower, "low", "none")))
    return flags


def flag_rows(dataset: str, grouping: str, flags: pd.DataFrame) -> list[dict]:
    """Counts per metric, for any mito metric, and the split of flagged cells into ID-only and intensity-based."""
    depth_low = (flags[list(DEPTH)] == "low").any(axis=1)
    rows = []

    def row(source: str, high: pd.Series, low: pd.Series) -> dict:
        return {"dataset": dataset, "grouping": grouping, "flag_source": source, "n_cells": len(flags),
                "n_high": int(high.sum()), "n_low": int(low.sum()), "n_flagged": int((high | low).sum()),
                "n_high_and_depth_low": int((high & depth_low).sum()),
                "n_low_and_depth_low": int((low & depth_low).sum())}

    for metric in MITO:
        rows.append(row(metric, flags[metric] == "high", flags[metric] == "low"))
    rows.append(row("any_mito_metric", (flags[list(MITO)] == "high").any(axis=1),
                    (flags[list(MITO)] == "low").any(axis=1)))
    id_flag = flags[ID_METRIC].isin(["high", "low"])
    int_flag = flags[list(INTENSITY)].isin(["high", "low"]).any(axis=1)
    rows.append(row("id_fraction_only", (flags[ID_METRIC] == "high") & ~int_flag, (flags[ID_METRIC] == "low") & ~int_flag))
    rows.append(row("any_intensity_metric", (flags[list(INTENSITY)] == "high").any(axis=1),
                    (flags[list(INTENSITY)] == "low").any(axis=1)))
    assert int((id_flag & ~int_flag).sum()) == rows[-2]["n_flagged"]
    return rows


def clean(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK["muted"])
    ax.tick_params(axis="x", length=1.5, width=0.4, labelsize=FONT, colors=INK["secondary"], pad=1.5)
    ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"], pad=2)


def rows_axis(ax, n: int, labels: list[str] | None = None) -> None:
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels if labels else [""] * n)
    for i in range(n):
        ax.axhline(i, color=INK["grid"], lw=0.3, zorder=0)
    clean(ax)


def forest(ax, sub: pd.DataFrame, order: list[str]) -> list[dict]:
    n = len(order)
    rows_axis(ax, n)
    ax.axvline(0, color=INK["muted"], lw=0.4, zorder=1)
    ax.set_xlim(-1.08, 1.08)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−1", "", "0", "", "1"])
    out = []
    t = sub.set_index("label")
    for i, lab in enumerate(order):
        r = t.loc[lab]
        if not np.isfinite(r["rho"]):
            ax.text(0, i, "n < 10", ha="center", va="center", fontsize=FONT, color=INK["secondary"],
                    bbox={"facecolor": INK["surface"], "edgecolor": "none", "pad": 0.6})
            out.append({"label": lab, "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "q": np.nan})
            continue
        sig = bool(r["p_fdr_bh_within_labels"] < 0.05)
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=BLUE, lw=0.7, zorder=2, solid_capstyle="butt")
        ax.plot([r["rho"]], [i], "o", ms=3.4, mec=BLUE, mew=0.7, mfc=BLUE if sig else "white", zorder=3)
        out.append({"label": lab, "rho": r["rho"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "q": r["p_fdr_bh_within_labels"]})
    return out


def legend(sh, y: float) -> None:
    """One row of keys under the panels, drawn in mm so that spacing is exact."""
    ax = sh.ax(0, y, sh.W, 4.0)
    ax.set_xlim(0, sh.W)
    ax.set_ylim(4, 0)
    ax.axis("off")
    x = M + 1.0
    yc = 2.0

    def key(draw, text: str, w: float = 4.0) -> None:
        nonlocal x
        draw(x, w)
        x += w + 1.2
        sh.text(x, y + yc, text, va="center", color=INK["secondary"])
        x += T8.text_mm(text)[0] + 4.0

    key(lambda a, w: ax.plot([a + w / 2], [yc], "o", ms=1.8, mfc=INK["muted"], mec="none"), "one cell", 1.5)
    key(lambda a, w: (ax.add_patch(plt.Rectangle((a, yc - 1.1), w, 2.2, fill=False, ec=INK["primary"], lw=0.6)),
                      ax.plot([a + w / 2] * 2, [yc - 1.1, yc + 1.1], color=INK["primary"], lw=1.1)),
        "median and IQR")
    key(lambda a, w: ax.plot([a, a + w], [yc, yc], color=INK["secondary"], lw=0.7, ls=(0, (3, 2))),
        "global top-10% cutoff", 5.0)
    key(lambda a, w: ax.plot([a + w / 2], [yc], "o", ms=3.4, mec=BLUE, mew=0.7, mfc=BLUE), "BH q < 0.05", 1.8)
    key(lambda a, w: ax.plot([a + w / 2], [yc], "o", ms=3.4, mec=BLUE, mew=0.7, mfc="white"), "q ≥ 0.05", 1.8)
    key(lambda a, w: ax.plot([a, a + w], [yc, yc], color=BLUE, lw=0.7), "95% CI")


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)

    def rd(folder, name: str, what: str) -> pd.DataFrame:
        return pd.read_csv(run.use_input(folder / name, what))

    per = rd(QC_MITO, "qc_mito_kidney_qc_per_cell.csv", "qc_mito kidney per-cell metrics (qc_mito_mito_qc.py)")
    lab = rd(KIDNEY_ANNOTATION, "kidney_annotation_cluster_labels_per_cell_a_endo.csv", "kidney_t15_label per cell (kidney_annotation)")
    thr = rd(KIDNEY_ANNOTATION, "kidney_annotation_addendum_t03_outlier_thresholds.csv", "kidney_annotation addendum thresholds")
    cnt = rd(KIDNEY_ANNOTATION, "kidney_annotation_addendum_t03_outlier_counts.csv", "kidney_annotation addendum outlier counts")
    qcs = rd(KIDNEY_ANNOTATION, "kidney_annotation_addendum_t03_qc_summary_by_label.csv", "kidney_annotation addendum summary by label")
    sim = rd(KIDNEY_ANNOTATION, "kidney_annotation_addendum_t03_global_cutoff_simulation.csv", "kidney_annotation addendum global cutoffs")
    tst = rd(KIDNEY_ANNOTATION, "kidney_annotation_addendum_t03_global_cutoff_tests.csv", "kidney_annotation addendum global cutoff tests")
    cor = rd(KIDNEY_ANNOTATION, "kidney_annotation_addendum_t03_within_type_depth_correlation.csv", "kidney_annotation addendum within-type rho")
    var = rd(KIDNEY_ANNOTATION, "kidney_annotation_addendum_t03_variance_explained.csv", "kidney_annotation addendum variance explained")
    sen = rd(KIDNEY_ANNOTATION, "kidney_annotation_sensitivity_vs_random.csv", "kidney_annotation kidney exclusion vs random")
    cov = rd(QC_MITO, "qc_mito_mito_coverage.csv", "qc_mito MitoCarta coverage")
    raw = rd(QC_MITO, "qc_mito_all_raw_cells_mito.csv", "qc_mito all raw cells")
    rvr = rd(QC_MITO, "qc_mito_removed_vs_retained.csv", "qc_mito removed vs retained")
    dct = rd(DCT_REVIEWER, "dct_reviewer_mito_qc_per_cell.csv", "dct_reviewer per-cell metrics and flags")
    draw = rd(DCT_REVIEWER, "dct_reviewer_mito_all_raw_cells.csv", "dct_reviewer all raw cells")
    doc = rd(DCT_REVIEWER, "dct_reviewer_mito_outlier_counts.csv", "dct_reviewer outlier counts")
    dex = rd(DCT_REVIEWER, "dct_reviewer_exclusion_sets.csv", "dct_reviewer exclusion sets")
    dvr = rd(DCT_REVIEWER, "dct_reviewer_exclusion_vs_random.csv", "dct_reviewer exclusion vs random")
    for frame in (thr, cnt, qcs, sim, tst, cor, var):
        keep = frame["label_column"] == LABEL
        frame.drop(frame.index[~keep], inplace=True)

    checks: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {observed}")

    cells = per.merge(lab[["cell_id", LABEL]], on="cell_id", how="inner", validate="one_to_one")
    obs = cells[LABEL].value_counts().to_dict()
    check("kidney cells joined to kidney_t15_label on cell_id", len(cells), 712, len(cells) == 712 == len(per))
    check("kidney label counts equal the project notes", obs, EXPECTED_KIDNEY_ANNOTATION, obs == EXPECTED_KIDNEY_ANNOTATION)
    med = cells.groupby(LABEL)[INTENSITY[0]].median()
    ref = qcs[qcs["metric"] == INTENSITY[0]].set_index("label")["median"].drop("all_cells")
    check("joined medians reproduce the kidney_annotation addendum summary", float((med - ref.reindex(med.index)).abs().max()), 0.0,
          bool(np.allclose(med, ref.reindex(med.index))))

    # flag counts by metric
    kf = kidney_flags(cells, thr)
    all_rows = flag_rows("kidney", LABEL, kf)
    ref = cnt[cnt["label"] == "all_cells"].iloc[0]
    k = {r["flag_source"]: r for r in all_rows}
    for metric in MITO:
        pair = (k[metric]["n_high"], k[metric]["n_low"])
        want = (int(ref[f"n_{metric}_high"]), int(ref[f"n_{metric}_low"]))
        check(f"kidney high/low flags reproduce the kidney_annotation addendum: {metric}", pair, want, pair == want)
    pair = (k["any_mito_metric"]["n_flagged"], k["any_mito_metric"]["n_high_and_depth_low"])
    want = (int(ref["n_mito_any"]), int(ref["n_mito_high_and_depth_low"]))
    check("kidney mito_any and mito_high & depth_low reproduce the kidney_annotation addendum", pair, want, pair == want)

    union_rows = []
    for cond, g in dct.groupby("condition"):
        check(f"{cond}: retained cells", len(g), EXPECTED_DCT[cond], len(g) == EXPECTED_DCT[cond])
        roles = {}
        for role in ("primary", "second"):
            fl = g[[f"outlier__{role}__{m}" for m in MITO + DEPTH]].copy()
            fl.columns = list(MITO + DEPTH)
            roles[role] = fl
            rows = flag_rows(cond, f"dct_labels {role} label column", fl)
            all_rows += rows
            d = {r["flag_source"]: r for r in rows}
            ref = doc[(doc["condition"] == cond) & (doc["label_role"] == role) & (doc["label"] == "all_cells")].iloc[0]
            pair = (d["any_mito_metric"]["n_high"], d["any_mito_metric"]["n_flagged"],
                    d["any_mito_metric"]["n_high_and_depth_low"])
            want = (int(ref["n_mito_high"]), int(ref["n_mito_any"]), int(ref["n_mito_high_and_depth_low"]))
            check(f"{cond} {role}: high, any and high & depth_low reproduce dct_reviewer", pair, want, pair == want)
        is_id = roles["primary"][ID_METRIC].isin(["high", "low"]) | roles["second"][ID_METRIC].isin(["high", "low"])
        is_int = (roles["primary"][list(INTENSITY)].isin(["high", "low"]).any(axis=1)
                  | roles["second"][list(INTENSITY)].isin(["high", "low"]).any(axis=1))
        union = g["flag__primary__mito_any"].astype(bool) | g["flag__second__mito_any"].astype(bool)
        n_set = int(dex[(dex["condition"] == cond) & (dex["exclusion_set"] == "mito_flagged")]["n_excluded"].iloc[0])
        check(f"{cond}: union of mito flags equals the dct_reviewer exclusion set", int(union.sum()), n_set, int(union.sum()) == n_set)
        union_rows.append({"dataset": cond, "n_union": int(union.sum()), "n_id_fraction_only": int((is_id & ~is_int).sum()),
                           "n_any_intensity_metric": int(is_int.sum()),
                           "median_ids_flagged": float(g.loc[union, "n_detected_proteins"].median()),
                           "median_ids_not_flagged": float(g.loc[~union, "n_detected_proteins"].median())})
    flags_out = pd.DataFrame(all_rows)
    run.table(flags_out, "02_mito_flags_by_metric")

    # statement-level values used in the the response to the reviewers letter
    vals: list[dict] = []

    def val(section: str, statement: str, dataset: str, value, source: str) -> None:
        vals.append({"section": section, "statement": statement, "dataset": dataset, "value": value, "source": source})

    c = cov[(cov["dataset"] == "kidney") & (cov["protein_set"] == "mitocarta")].iloc[0]
    val("coverage", "MitoCarta protein groups of matrix proteins", "kidney",
        f"{int(c['n_protein_groups_in_matrix'])} of {int(c['n_matrix_proteins'])}", "qc_mito_mito_coverage.csv")
    r0 = raw[raw["dataset"] == "kidney"]
    for ds, frame in (("kidney", r0), ("Ctrl", draw[draw["condition"] == "Ctrl"]), ("ZeroK", draw[draw["condition"] == "ZeroK"])):
        rem = frame[~frame["retained"].astype(bool)]
        val("removed wells", "wells removed by the 100-protein filter", ds, len(rem), "raw-cell table")
        val("removed wells", "median proteins per removed well", ds, float(rem["n_detected_proteins"].median()), "raw-cell table")
        val("removed wells", "median MitoCarta proteins per removed well", ds, float(rem["n_detected_mitocarta"].median()), "raw-cell table")
    check("removed wells: kidney, Ctrl, ZeroK", [v["value"] for v in vals if v["statement"].startswith("wells removed")],
          [56, 39, 65], [v["value"] for v in vals if v["statement"].startswith("wells removed")] == [56, 39, 65])
    for _, r in rvr[(rvr["dataset"] == "kidney") & (rvr["metric"] == INTENSITY[0])].iterrows():
        val("removed wells", f"removed vs retained, {r['comparison']}: medians and p", "kidney",
            f"{r['median_group']:.3f} vs {r['median_reference']:.3f}; p {r['p_value']:.2g}", "qc_mito_removed_vs_retained.csv")
    for depth in ("log10_n_detected", "log10_total_intensity"):
        r = var[(var["metric"] == INTENSITY[0]) & (var["scale"] == "raw") & (var["depth_variable"] == depth)].iloc[0]
        val("variance", f"R2 label; R2 {depth}; R2 both", "kidney",
            f"{r['r2_label']:.3f}; {r['r2_depth']:.3f}; {r['r2_label_plus_depth']:.3f}",
            "kidney_annotation_addendum_t03_variance_explained.csv (raw scale)")
    tt = tst[(tst["metric"] == INTENSITY[0]) & (tst["cutoff"] == "global_top10pct")].iloc[0]
    val("cutoff", "global top-10% cutoff: threshold, removed, permutation p", "kidney",
        f"{tt['threshold']:.3f}; {int(tt['n_removed'])} of {int(tt['n_cells'])}; {tt['p_permutation']:.4f}",
        "kidney_annotation_addendum_t03_global_cutoff_tests.csv")
    for row_ in union_rows:
        val("flags", "mito exclusion set: size; ID fraction only; any intensity metric; median IDs flagged / not", row_["dataset"],
            f"{row_['n_union']}; {row_['n_id_fraction_only']}; {row_['n_any_intensity_metric']}; "
            f"{row_['median_ids_flagged']:.0f} / {row_['median_ids_not_flagged']:.0f}",
            "dct_reviewer_mito_qc_per_cell.csv (this script)")
    k_set = sen[sen["exclusion_set"] == "mito_flagged"]
    n_in = int(k_set["inside_central_95"].astype(bool).sum())
    val("exclusion", "outcomes inside the central 95% of random removals, mito-flagged set", "kidney",
        f"{n_in} of {len(k_set)}", "kidney_annotation_sensitivity_vs_random.csv")
    check("kidney exclusion: inside of 12", n_in, 11, n_in == 11 and len(k_set) == 12)
    app = dvr[dvr["applicable"].astype(str).str.lower().isin(["true", "yes", "1"]) & dvr["observed"].notna()]
    want = {("Ctrl", "mito_flagged"): (14, 18), ("ZeroK", "mito_flagged"): (15, 16),
            ("Ctrl", "any_flagged"): (14, 18), ("ZeroK", "any_flagged"): (13, 16)}
    for (cond, eset), g in app.groupby(["condition", "exclusion_set"]):
        got = (int((g["inside_central_95"].astype(str) == "True").sum()), len(g))
        outside = "; ".join(g.loc[g["inside_central_95"].astype(str) != "True", "outcome"])
        val("exclusion", f"outcomes inside the central 95% of random removals, {eset}", cond, f"{got[0]} of {got[1]}",
            "dct_reviewer_exclusion_vs_random.csv (applicable, observed defined)")
        val("exclusion", f"outcomes outside, {eset}", cond, outside, "dct_reviewer_exclusion_vs_random.csv")
        check(f"{cond} {eset}: inside of applicable (dct_reviewer README totals 28/36, 28/32)", got, want[(cond, eset)],
              got == want[(cond, eset)])

    # figure
    medians = cells.groupby(LABEL)[INTENSITY[0]].median().drop("Unknown").sort_values(ascending=False)
    order = list(medians.index) + ["Unknown"]
    n = len(order)
    disp = [f"{DISPLAY.get(x, x)} ({EXPECTED_KIDNEY_ANNOTATION[x]})" for x in order]
    cut = float(tt["threshold"])
    top10 = sim[(sim["metric"] == INTENSITY[0]) & (sim["cutoff"] == "global_top10pct")].set_index("label")

    x0 = M + max(T8.text_mm(s)[0] for s in disp) + 1.5
    gap = 4.0
    w_b, w_f = 21.0, 23.0
    w_a = T8.W - M - x0 - w_b - 3 * w_f - 4 * gap
    let_h, head_h = 3.6, 6.9
    y_ax = M + let_h + head_h + 0.6
    h_ax = n * ROW_H
    y_leg = y_ax + h_ax + 5.2
    H = y_leg + 4.0 + M
    sh = T8.Sheet(H)
    xs = [x0, x0 + w_a + gap]
    xs += [xs[1] + w_b + gap, xs[1] + w_b + 2 * gap + w_f, xs[1] + w_b + 3 * gap + 2 * w_f]
    heads = ["MitoCarta fraction of\nsummed intensity", "Cells above the\ncutoff (%)", "Intensity fraction vs\nprotein IDs (ρ)",
             "Intensity fraction vs\ntotal intensity (ρ)", "ID fraction vs\nprotein IDs (ρ)"]
    for letter, x, head in zip("ABCDE", xs, heads):
        sh.text(x, M, letter, fontsize=8.0, fontweight="bold", va="top")
        sh.text(x, M + let_h, head, va="top", linespacing=1.1, color=INK["primary"])

    ax = sh.ax(xs[0], y_ax, w_a, h_ax)
    rows_axis(ax, n, disp)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    rng = np.random.default_rng(cc.RANDOM_SEED if hasattr(cc, "RANDOM_SEED") else 42)
    plotted = []
    for i, lab_ in enumerate(order):
        v = cells.loc[cells[LABEL] == lab_, INTENSITY[0]].to_numpy()
        ax.scatter(v, i + rng.uniform(-0.3, 0.3, v.size), s=1.6, c=INK["muted"], alpha=0.55, linewidths=0, zorder=2)
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        ax.add_patch(plt.Rectangle((q1, i - 0.33), q3 - q1, 0.66, fill=False, ec=INK["primary"], lw=0.6, zorder=3))
        ax.plot([q2, q2], [i - 0.33, i + 0.33], color=INK["primary"], lw=1.1, zorder=4, solid_capstyle="butt")
        plotted.append({"panel": "A", "label": lab_, "n": v.size, "q25": q1, "median": q2, "q75": q3})
    ax.axvline(cut, color=INK["secondary"], lw=0.7, ls=(0, (3, 2)), zorder=1)

    ax = sh.ax(xs[1], y_ax, w_b, h_ax)
    rows_axis(ax, n)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 10, 20, 30])
    for i, lab_ in enumerate(order):
        p = float(top10.loc[lab_, "percent_of_label_removed"])
        ax.barh(i, p, height=0.56, color=BLUE, linewidth=0, zorder=2)
        ax.text(p + 1.0, i, f"{p:.0f}", va="center", ha="left", fontsize=FONT, color=INK["primary"])
        plotted.append({"panel": "B", "label": lab_, "n": int(top10.loc[lab_, "n_removed"]), "median": p})

    for panel, x, metric, depth in (("C", xs[2], INTENSITY[0], "log10_n_detected"),
                                    ("D", xs[3], INTENSITY[0], "log10_total_intensity"),
                                    ("E", xs[4], ID_METRIC, "log10_n_detected")):
        sub = cor[(cor["metric"] == metric) & (cor["depth_variable"] == depth)]
        ax = sh.ax(x, y_ax, w_f, h_ax)
        for r in forest(ax, sub, order):
            plotted.append({"panel": panel, **r})
    legend(sh, y_leg)

    fig_rows: list[dict] = []
    ok = T8.save_checked(run, sh, FIGURE, "Supplementary Figure S-02-a: kidney mitochondrial fraction by cell type, global "
                         "top-10% cutoff and within-type Spearman rho; revised annotation; from qc_mito and kidney_annotation tables", fig_rows)
    check("figure passes the format checks (>= 7 pt, Arial, no text overlaps, text inside, 180 mm)", fig_rows[0]["pass"], True, ok)

    values = pd.concat([pd.DataFrame(vals), pd.DataFrame(plotted).assign(section="figure S-02-a")], ignore_index=True)
    run.table(values, f"{STEP}_values")
    run.table(pd.DataFrame(fig_rows), f"{STEP}_figure_checks")
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    all_ok = all(c_["pass"] for c_ in checks)
    run.finish({"label_column": LABEL, "display_names": DISPLAY, "row_order": order, "cutoff": cut, "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit("A check failed; see report_02_mito_panels_checks.csv.")


if __name__ == "__main__":
    main()
