#!/usr/bin/env python
"""The response to the reviewers: the preprocessing kidney grid scored against `kidney_t15_label`, DCT intensity descriptors, figure S2.

(reviewer 1, comment 2; the earlier analysis points 8-9). Run with the authors'
OK (5 Nov 2026). No Leiden clustering is rerun.

Why it exists:
1. preprocessing scored its 33 kidney pipelines against the submitted labels only (`kidney_pre_label`, `kidney_reannot_label`).
   Kidney results are reported under `kidney_t15_label` only (authors, 30 Oct 2026), so the grid is scored here against it:
   - ARI, NMI and cluster-majority accuracy from the 132 clusterings saved by preprocessing;
   - kNN label purity and silhouette on each pipeline's PCs, rebuilt with preprocessing_lib (arpack PCA, seed 42).
   The rebuild is checked against preprocessing: the same scores against `kidney_pre_label` and the PC-depth values must match.
2. dct_reviewer repeated preprocessing steps 2-5 for the DCT conditions but not step 1, so the per-cell intensity descriptors of the
   combined search do not exist. They are computed here with the preprocessing definitions from the dct_labels objects, and the kidney
   row is checked against the preprocessing table. The retired 758-cell values are not used.
3. Collects every number of the the response to the reviewers letter in three small tables, and draws Supplementary Figure S2. preprocessing's
   kidney figures show the submitted labels and cannot be cited.

Macro averages: labels with >= 10 cells, Unknown left out (as dct_reviewer). The preprocessing rule, which keeps Unknown, is also reported
(`*_incl_unknown`) and is the one used for the checks against the preprocessing tables.

Supplementary Figure S2 (180 mm; no in-figure title; rows A-E, columns total kidney, DCT control, DCT zero-K, one x scale per
row): one dot per pipeline. Colour: one hue for the data marks; filled or hollow carries per-protein scaling, so
nothing depends on colour alone. The vertical offset inside a strip only separates dots.

Inputs (read-only): results/h5ad/kidney_object_KIDNEY_submitted.h5ad and the kidney_object kidney labels;
                    results/h5ad/dct_labels_DCT_{ctrl,zerok}_labelled.h5ad;
                    results/tables/kidney_annotation/kidney_annotation_cluster_labels_per_cell_{a,b}_endo.csv,
                    kidney_annotation_graph_definitions.csv, kidney_annotation_v3endo_summary.csv;
                    results/tables/preprocessing/preprocessing_{kidney_per_cell_clusters,kidney_clustering_metrics,
                    kidney_knn_purity,kidney_silhouette,kidney_pc_depth,kidney_variant_summary,
                    kidney_feature_set_summary,kidney_normalization_diagnostics,spectronaut_per_cell_summary}.csv;
                    results/tables/dct_reviewer/dct_reviewer_prep_{variant_summary,ranking,feature_set_summary,
                    normalization_diagnostics}.csv, dct_reviewer_embed_{pc_vs_depth_and_plate,knn_label_purity,
                    ari_nmi_clusters_vs_labels}.csv
Outputs: results/tables/report/report_03_{kidney_t15_clustering_metrics,kidney_t15_knn_purity,
         kidney_t15_silhouette,pipeline_summary,group_summary,detection_summary,normalization_factors,
         intensity_descriptors,depth_regressed,preprocessing_scores_checks,preprocessing_scores_figure_checks}.csv;
         results/tables/report/report_step_03_preprocessing_scores_manifest.json;
         results/figures/report/report_03_preprocessing_summary.{pdf,png}

Command (project root):  python scripts/report_preprocessing_scores.py
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
import time

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import qc_mito_lib as t3
import preprocessing_lib as t4
import atlas_dct_lib as T8

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
plt = T8.setup_matplotlib()

TASK = "report"
STEP = "03_preprocessing_scores"
FIGURE = "03_preprocessing_summary"
PREPROCESSING = cc.OUT_ROOTS["tables"] / "preprocessing"
KIDNEY_ANNOTATION = cc.OUT_ROOTS["tables"] / "kidney_annotation"
DCT_REVIEWER = cc.OUT_ROOTS["tables"] / "dct_reviewer"
LABEL = "kidney_t15_label"
CHECK_LABEL = "kidney_pre_label"            # used only to prove that the rebuild reproduces preprocessing
EXPECTED_KIDNEY_ANNOTATION = {"PCT": 160, "PST": 158, "TAL": 138, "DCT1": 58, "PC": 38, "JG": 24, "Endothelial": 19, "IC-B": 8,
                "Unknown": 109}             # the project notes
DCT = {"Ctrl": {"key": "ctrl", "primary": "cs_ctrl_deg_label", "second": "cs_ctrl_cluster_label", "n": 729, "p": 6078},
       "ZeroK": {"key": "zerok", "primary": "cs_zerok_pipeline_label", "second": "cs_zerok_cluster_label", "n": 702,
                 "p": 5953}}
DATASETS = ("kidney", "Ctrl", "ZeroK")
COLUMN = {"kidney": "Total kidney (712 cells)", "Ctrl": "DCT control (729 cells)", "ZeroK": "DCT zero-K (702 cells)"}
CORE_MIN_DETECTION = 0.90                   # preprocessing definition of core proteins
LP_METRICS = ("best_ari", "best_nmi", "macro_knn_purity_enrichment", "macro_silhouette")
SCORES = ("best_ari", "best_nmi", "macro_knn_purity", "macro_silhouette", "max_abs_rho_pc12")
GROUPS = {"cellwise_scaled_all": "cell-wise normalization, scaled, all proteins",
          "unscaled_all": "no per-protein scaling, all proteins",
          "hvg": "highly variable proteins (500, 1,000, 2,000)",
          "highcov50": "proteins detected in >= 50% of cells",
          "pcs_or_k": "submitted pipeline with other PCs or k"}
STRIPS = (("all_proteins", "All proteins (5)"), ("hvg", "HVG sets (18)"), ("highcov50", "High coverage (3)"),
          ("pcs_or_k", "PCs or k changed (6)"))
M, FONT, INK = T8.M, T8.FONT, T8.INK
BLUE = "#2a78d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: kidney grid under kidney_t15_label; DCT descriptors.")
    cc.add_common_args(parser)
    parser.add_argument("--n-permutations", type=int, default=t4.N_PERMUTATIONS)
    return parser.parse_args()


def group_of(row) -> str:
    if row["block"] in ("pca_test", "knn_test"):
        return "pcs_or_k"
    if row["features"] == "highcov50":
        return "highcov50"
    if str(row["features"]).startswith("hvg"):
        return "hvg"
    if row["is_baseline"]:
        return "submitted"
    return "unscaled_all" if row["scaling"] == "unscaled" else "cellwise_scaled_all"


def macro_core(frame: pd.DataFrame, value: str) -> float:
    """Labels with >= 10 cells, Unknown and the 'all_cells' row left out (the dct_reviewer rule)."""
    sub = frame[~frame["label"].isin(["all_cells", "Unknown"]) & (frame["n_cells"] >= t4.MIN_LABEL_N)]
    return float(sub[value].mean())


def rank_pipelines(frame: pd.DataFrame) -> pd.DataFrame:
    """As preprocessing_summary_figures.rank_scope."""
    out = frame.copy().reset_index(drop=True)
    for m in LP_METRICS:
        out[f"rank_{m}"] = out[m].rank(ascending=False, method="average")
    out["label_preservation_mean_rank"] = out[[f"rank_{m}" for m in LP_METRICS]].mean(axis=1)
    out["label_preservation_rank"] = out["label_preservation_mean_rank"].rank(method="min")
    out["depth_independence_rank"] = out["max_abs_rho_pc12"].rank(method="min")
    out["combined_score"] = (out["label_preservation_rank"] + out["depth_independence_rank"]) / 2
    out["combined_rank"] = out["combined_score"].rank(method="min")
    return out


def descriptors(dataset: str, x: np.ndarray, cell_ids) -> pd.DataFrame:
    """Per-cell intensity summaries over the retained cells and proteins (preprocessing step 1, summary_rows)."""
    det = x > 0
    ln = np.where(det, np.log(np.where(det, x, 1.0)), np.nan)
    core = det.mean(axis=0) >= CORE_MIN_DETECTION
    ln_ids = np.log(det.sum(axis=1))
    columns = {"ln_total_intensity": np.log(x.sum(axis=1)),
               "median_ln_intensity_detected": np.nanmedian(ln, axis=1),
               "median_ln_intensity_core": np.nanmedian(np.where(det[:, core], ln[:, core], np.nan), axis=1)}
    rows = []
    for name, v in columns.items():
        rho = t3.spearman_ci(v, ln_ids)
        rows.append({"dataset": dataset, "descriptor": name, "n_cells": len(cell_ids), "n_proteins": x.shape[1],
                     "n_core_proteins": int(core.sum()) if "core" in name else np.nan,
                     "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
                     "range_fold_linear": float(np.exp(v.max() - v.min())),
                     "spearman_rho_vs_ln_n_detected": rho["rho"], "rho_ci_low": rho["ci_low"],
                     "rho_ci_high": rho["ci_high"], "p_value": rho["p_value"],
                     "median_n_detected": float(np.median(det.sum(axis=1)))})
    return pd.DataFrame(rows)


def clean(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK["muted"])
    ax.tick_params(axis="x", length=1.5, width=0.4, labelsize=FONT, colors=INK["secondary"], pad=1.5)
    ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"], pad=2)


def strip_panel(ax, strips: list[tuple[str, pd.DataFrame]], value: str, base: float, xlim, ticks, fmt: str,
                labels: bool, plotted: list[dict], tag: str) -> None:
    """One strip per pipeline group; dots spread evenly in y inside the strip (the offset carries no meaning)."""
    n = len(strips)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[0] for s in strips] if labels else [""] * n)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt.format(t) for t in ticks])
    for i in range(n):
        ax.axhline(i, color=INK["grid"], lw=0.3, zorder=0)
    clean(ax)
    ax.axvline(base, color=INK["secondary"], lw=0.7, ls=(0, (3, 2)), zorder=1)
    for i, (_, sub) in enumerate(strips):
        sub = sub.sort_values(value).reset_index(drop=True)
        offs = np.zeros(len(sub)) if len(sub) == 1 else np.tile([-0.24, 0.0, 0.24], len(sub))[:len(sub)]
        for off, (_, r) in zip(offs, sub.iterrows()):
            scaled = r["scaling"] == "z10"
            ax.plot([r[value]], [i + off], "o", ms=2.5, mec=BLUE, mew=0.5, mfc=BLUE if scaled else "white", zorder=3,
                    clip_on=False)
            plotted.append({"panel": tag, "strip": i, "variant_id": r["variant_id"], "value": float(r[value])})
    frac = (base - xlim[0]) / (xlim[1] - xlim[0])
    ax.text(base, -0.5, f"{base:.1f}" if fmt == "{:.0f}" else f"{base:.3f}", ha="right" if frac > 0.85 else "center",
            va="bottom", fontsize=FONT, color=INK["primary"])


def draw_figure(run: cc.Run, summary: pd.DataFrame, detect: pd.DataFrame, plotted: list[dict], fig_rows: list[dict]) -> bool:
    rows = (("A", "best_nmi", "Agreement of Leiden clusters with the cell labels (NMI, best of four resolutions)",
             (0.0, 0.8), [0, 0.2, 0.4, 0.6, 0.8], "{:.1f}"),
            ("B", "macro_knn_purity", "kNN label purity (mean over labels with at least 10 cells, Unknown left out)",
             (0.4, 1.0), [0.4, 0.6, 0.8, 1.0], "{:.1f}"),
            ("C", "macro_silhouette", "Silhouette width by label (mean over the same labels)", None, None, "{:.1f}"),
            ("D", "max_abs_rho_pc12", "Largest |ρ| of PC1 or PC2 with protein IDs per cell",
             (0.6, 1.0), [0.6, 0.7, 0.8, 0.9, 1.0], "{:.1f}"))
    lo = float(summary["macro_silhouette"].min())
    hi = float(summary["macro_silhouette"].max())
    sil_lim = (np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10)
    sil_ticks = [t for t in (-0.2, 0.0, 0.2, 0.4, 0.6) if sil_lim[0] <= t <= sil_lim[1]]

    labels_w = max(T8.text_mm(s[1])[0] for s in STRIPS)
    x0 = M + labels_w + 1.5
    gap = 5.0
    pw = (T8.W - M - x0 - 2 * gap) / 3
    strip_h = 4.2
    head_h, value_h, axis_h, row_gap = 3.4, 3.2, 4.0, 2.2
    col_h = 3.4
    y = M + col_h + 0.6
    layout = []
    for r in rows:
        layout.append((r, y))
        y += head_h + value_h + 4 * strip_h + axis_h + row_gap
    y_e = y
    y_leg = y_e + head_h + value_h + 3 * strip_h + axis_h + 2.0
    H = y_leg + 4.0 + M
    sh = T8.Sheet(H)
    xs = [x0 + i * (pw + gap) for i in range(3)]
    for ds, x in zip(DATASETS, xs):
        sh.text(x, M, COLUMN[ds], fontweight="bold", va="top")

    for (letter, value, head, xlim, ticks, fmt), y_row in layout:
        sh.text(M, y_row, letter, fontsize=8.0, fontweight="bold", va="top")
        sh.text(M + 4.0, y_row + 0.2, head, va="top")
        if value == "macro_silhouette":
            xlim, ticks = sil_lim, sil_ticks
        for j, (ds, x) in enumerate(zip(DATASETS, xs)):
            sub = summary[summary["dataset"] == ds]
            base = float(sub.loc[sub["group"] == "submitted", value].iloc[0])
            strips = []
            for key, text in STRIPS:
                part = sub[sub["group"].isin(["cellwise_scaled_all", "unscaled_all"])] if key == "all_proteins" \
                    else sub[sub["group"] == key]
                strips.append((text, part))
            ax = sh.ax(x, y_row + head_h + value_h, pw, 4 * strip_h)
            strip_panel(ax, strips, value, base, xlim, ticks, fmt, j == 0, plotted, f"{letter} {ds}")

    sh.text(M, y_e, "E", fontsize=8.0, fontweight="bold", va="top")
    sh.text(M + 4.0, y_e + 0.2, "Median detection of the selected proteins (% of cells)", va="top")
    for j, (ds, x) in enumerate(zip(DATASETS, xs)):
        d = detect[detect["dataset"] == ds].copy()
        d["scaling"] = "z10"
        d["variant_id"] = d["normalization"] + "__" + d["features"]
        d["pct"] = 100 * d["median_detection_rate_selected"]
        base = float(d.loc[(d["normalization"] == "none") & (d["features"] == "all"), "pct"].iloc[0])
        strips = [("All proteins", d[(d["normalization"] == "none") & (d["features"] == "all")]),
                  ("HVG sets (9)", d[d["features"].str.startswith("hvg")]),
                  ("High coverage", d[(d["normalization"] == "none") & (d["features"] == "highcov50")])]
        ax = sh.ax(x, y_e + head_h + value_h, pw, 3 * strip_h)
        strip_panel(ax, strips, "pct", base, (0, 80), [0, 20, 40, 60, 80], "{:.0f}", j == 0, plotted, f"E {ds}")

    ax = sh.ax(0, y_leg, sh.W, 4.0)
    ax.set_xlim(0, sh.W)
    ax.set_ylim(4, 0)
    ax.axis("off")
    x = x0
    yc = 2.0

    def key(draw, text: str, w: float = 2.0) -> None:
        nonlocal x
        draw(x, w)
        x += w + 1.2
        sh.text(x, y_leg + yc, text, va="center", color=INK["secondary"])
        x += T8.text_mm(text)[0] + 5.0

    key(lambda a, w: ax.plot([a + w / 2], [yc], "o", ms=2.5, mec=BLUE, mew=0.5, mfc=BLUE), "with per-protein scaling")
    key(lambda a, w: ax.plot([a + w / 2], [yc], "o", ms=2.5, mec=BLUE, mew=0.5, mfc="white"), "without per-protein scaling")
    key(lambda a, w: ax.plot([a, a + w], [yc, yc], color=INK["secondary"], lw=0.7, ls=(0, (3, 2))),
        "submitted pipeline, with its value above the panel", 5.0)
    return T8.save_checked(run, sh, FIGURE, "Supplementary Figure S-03-a: 33 preprocessing pipelines per dataset, label "
                           "agreement, silhouette, depth and detection; kidney from this script, DCT from dct_reviewer", fig_rows)


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)
    t_start = time.time()
    checks: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {observed}", flush=True)

    def rd(folder, name: str, what: str, **kw) -> pd.DataFrame:
        return pd.read_csv(run.use_input(folder / name, what), **kw)

    # ------------------------------------------------------------------ kidney inputs
    adata, labels = t4.load(run, "kidney")
    cells = adata.obs_names.astype(str)
    t15 = rd(KIDNEY_ANNOTATION, "kidney_annotation_cluster_labels_per_cell_a_endo.csv", "kidney_t15_label per cell (kidney_annotation)").set_index("cell_id")
    saved = rd(PREPROCESSING, "preprocessing_kidney_per_cell_clusters.csv", "preprocessing saved kidney clusterings",
               dtype=str).set_index("cell_id")
    ref_clus = rd(PREPROCESSING, "preprocessing_kidney_clustering_metrics.csv", "preprocessing kidney clustering metrics")
    ref_pur = rd(PREPROCESSING, "preprocessing_kidney_knn_purity.csv", "preprocessing kidney kNN purity")
    ref_sil = rd(PREPROCESSING, "preprocessing_kidney_silhouette.csv", "preprocessing kidney silhouette")
    ref_dep = rd(PREPROCESSING, "preprocessing_kidney_pc_depth.csv", "preprocessing kidney PC-depth (label-free)")
    ref_var = rd(PREPROCESSING, "preprocessing_kidney_variant_summary.csv", "preprocessing kidney variant summary")
    ref_feat = rd(PREPROCESSING, "preprocessing_kidney_feature_set_summary.csv", "preprocessing kidney feature sets (label-free)")
    ref_norm = rd(PREPROCESSING, "preprocessing_kidney_normalization_diagnostics.csv", "preprocessing kidney normalization diagnostics")
    ref_desc = rd(PREPROCESSING, "preprocessing_spectronaut_per_cell_summary.csv", "preprocessing per-cell descriptors (kidney rows only)")
    grid = t4.variant_grid()
    t15 = t15.reindex(cells)
    saved = saved.reindex(cells)
    lab = t15[LABEL].astype(str).to_numpy()
    pre = labels[CHECK_LABEL].astype(str).to_numpy()
    obs = pd.Series(lab).value_counts().to_dict()
    check("kidney_t15_label counts equal the project notes", obs, EXPECTED_KIDNEY_ANNOTATION, obs == EXPECTED_KIDNEY_ANNOTATION)
    check("saved clusterings cover the 712 cells", int(saved.notna().all(axis=1).sum()), 712,
          int(saved.notna().all(axis=1).sum()) == 712)
    want = [f"{v.variant_id}__r{r:.1f}" for v in grid for r in t4.RESOLUTIONS]
    check("saved clusterings: 33 pipelines x 4 resolutions", saved.shape[1], 132, sorted(want) == sorted(saved.columns))
    ari_t15 = float(adjusted_rand_score(saved["none__z10__all__r2.0"], t15["t15_cluster"].astype(str)))
    check("preprocessing baseline Leiden 2.0 equals t15_cluster (ARI)", round(ari_t15, 6), 1.0, ari_t15 > 0.999999)

    # ------------------------------------------------------------------ ARI, NMI, majority from the saved clusterings
    clus_rows, diffs = [], []
    ref = ref_clus[ref_clus["label_column"] == CHECK_LABEL].set_index(["variant_id", "resolution"])
    for v in grid:
        for res in t4.RESOLUTIONS:
            c = saved[f"{v.variant_id}__r{res:.1f}"].to_numpy()
            clus_rows.append({"dataset": "kidney", "variant_id": v.variant_id, "resolution": res, "label_column": LABEL,
                              "n_clusters": int(pd.Series(c).nunique()), **t4.cluster_agreement(c, lab)})
            a = t4.cluster_agreement(c, pre)
            r = ref.loc[(v.variant_id, res)]
            diffs.append(max(abs(a[m] - float(r[m])) for m in ("ari", "nmi", "majority_accuracy")))
    clus = pd.DataFrame(clus_rows)
    check("scores from the saved clusterings reproduce preprocessing against kidney_pre_label (max abs diff, 132 clusterings)",
          f"{max(diffs):.2e}", "< 1e-9", max(diffs) < 1e-9)

    # ------------------------------------------------------------------ kNN purity, silhouette on rebuilt PCs
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    var_names = adata.var_names.astype(str)
    det_rate = t4.detection_rate(intensity)
    ln_ids = np.log(labels["n_detected_proteins"].to_numpy(float))
    ln_total = np.log(labels["total_intensity"].to_numpy(float))
    c_t15, cats_t15 = pd.factorize(pd.Categorical(lab, categories=cc.ordered_labels(lab, cc.KIDNEY_LABEL_ORDER)),
                                   sort=True)
    cats_t15 = list(cats_t15.astype(str))
    c_pre, cats_pre = t4.label_codes(pre, "kidney")
    norm_cache, feat_cache, emb_cache = {}, {}, {}
    pur_rows, sil_rows, dep_rows = [], [], []
    d_pur, d_sil, d_dep = [], [], []
    for i, v in enumerate(grid):
        if v.normalization not in norm_cache:
            norm_cache[v.normalization] = t4.normalize(intensity, v.normalization)
        norm_log = norm_cache[v.normalization][0]
        mask = t4.feature_mask(v, norm_log, det_rate, var_names, feat_cache)
        key = (v.normalization, v.features, v.scaling)
        if key not in emb_cache:
            emb_cache[key] = t4.build_embedding(norm_log, mask, v.scaling, cells, var_names)
        g = emb_cache[key]
        pcs = g.obsm["X_pca"][:, :v.n_pcs]
        idx = t4.neighbor_index(pcs, t4.PURITY_K)
        p = t4.knn_purity(idx, c_t15, cats_t15, args.n_permutations)
        s = t4.silhouette(pcs, c_t15, cats_t15)
        dep = t4.pc_depth(g.obsm["X_pca"], ln_ids, ln_total, g.uns["pca"]["variance_ratio"])
        for frame, rows in ((p, pur_rows), (s, sil_rows), (dep, dep_rows)):
            f = frame.copy()
            f.insert(0, "variant_id", v.variant_id)
            f.insert(0, "dataset", "kidney")
            rows.append(f)
        # the same PCs against kidney_pre_label must return the preprocessing values
        p_ref = t4.knn_purity(idx, c_pre, cats_pre, 1).set_index("label")["observed_same_label_fraction"]
        s_ref = t4.silhouette(pcs, c_pre, cats_pre).set_index("label")["mean_silhouette"]
        rp = ref_pur[(ref_pur["variant_id"] == v.variant_id) & (ref_pur["label_column"] == CHECK_LABEL)]
        rs = ref_sil[(ref_sil["variant_id"] == v.variant_id) & (ref_sil["label_column"] == CHECK_LABEL)]
        d_pur.append(float((p_ref - rp.set_index("label")["observed_same_label_fraction"]).abs().max()))
        d_sil.append(float((s_ref - rs.set_index("label")["mean_silhouette"]).abs().max()))
        rd_ = ref_dep[ref_dep["variant_id"] == v.variant_id].set_index("pc")["spearman_rho_ln_n_detected"]
        d_dep.append(float((dep.set_index("pc")["spearman_rho_ln_n_detected"].abs() - rd_.abs()).abs().max()))
        print(f"[kidney] {i + 1:2d}/33 {v.variant_id}: {g.n_vars} proteins", flush=True)
    purity = pd.concat(pur_rows, ignore_index=True)
    sil = pd.concat(sil_rows, ignore_index=True)
    depth = pd.concat(dep_rows, ignore_index=True)
    purity.insert(2, "label_column", LABEL)
    sil.insert(2, "label_column", LABEL)
    for name, d in (("kNN purity", d_pur), ("silhouette", d_sil), ("|rho| of PC1-5 with ln(IDs)", d_dep)):
        check(f"rebuilt PCs reproduce preprocessing: {name} against kidney_pre_label or depth (max abs diff, 33 pipelines)",
              f"{max(d):.2e}", "< 1e-4", max(d) < 1e-4)

    # ------------------------------------------------------------------ one row per pipeline
    n_prot = ref_feat.set_index(["normalization", "features"])["n_proteins"]
    rows = []
    for v in grid:
        c = clus[clus["variant_id"] == v.variant_id].set_index("resolution")
        p = purity[purity["variant_id"] == v.variant_id]
        s = sil[sil["variant_id"] == v.variant_id]
        d = depth[depth["variant_id"] == v.variant_id].set_index("pc")
        r1, r2 = float(d.at[1, "spearman_rho_ln_n_detected"]), float(d.at[2, "spearman_rho_ln_n_detected"])
        rows.append({"dataset": "kidney", "label_role": "primary", "label_column": LABEL, **v.as_row(),
                     "n_proteins": 4054 if v.features == "all" else int(n_prot.loc[(v.normalization, v.features)]),
                     "best_ari": float(c["ari"].max()), "best_ari_resolution": float(c["ari"].idxmax()),
                     "best_nmi": float(c["nmi"].max()), "best_nmi_resolution": float(c["nmi"].idxmax()),
                     "best_majority_accuracy": float(c["majority_accuracy"].max()),
                     "macro_knn_purity": macro_core(p, "observed_same_label_fraction"),
                     "macro_knn_purity_enrichment": macro_core(p, "enrichment"),
                     "macro_silhouette": macro_core(s, "mean_silhouette"),
                     "macro_knn_purity_incl_unknown": t4.macro(p, "observed_same_label_fraction"),
                     "macro_silhouette_incl_unknown": t4.macro(s, "mean_silhouette"),
                     "rho_pc1_ln_n_detected": r1, "rho_pc2_ln_n_detected": r2, "max_abs_rho_pc12": max(abs(r1), abs(r2))})
    kid = rank_pipelines(pd.DataFrame(rows))

    t28 = rd(DCT_REVIEWER, "dct_reviewer_prep_variant_summary.csv", "dct_reviewer pipelines, per condition and label column")
    t28_rank = rd(DCT_REVIEWER, "dct_reviewer_prep_ranking.csv", "dct_reviewer ranking")
    t28_feat = rd(DCT_REVIEWER, "dct_reviewer_prep_feature_set_summary.csv", "dct_reviewer feature sets (label-free)")
    t28_norm = rd(DCT_REVIEWER, "dct_reviewer_prep_normalization_diagnostics.csv", "dct_reviewer normalization diagnostics")
    keep_rank = ["label_preservation_rank", "depth_independence_rank", "combined_rank"]
    frames = [kid]
    for cond, spec in DCT.items():
        for role in ("primary", "second"):
            part = t28[(t28["condition"] == cond) & (t28["label_column"] == spec[role])].copy()
            part["dataset"] = cond
            part["label_role"] = role
            rk = t28_rank[(t28_rank["condition"] == cond) & (t28_rank["scope_role"] == role)]
            part = part.merge(rk[["variant_id", *keep_rank]], on="variant_id", how="left", validate="one_to_one")
            frames.append(part)
    summary = pd.concat(frames, ignore_index=True)
    summary["group"] = summary.apply(group_of, axis=1)
    cols = ["dataset", "label_role", "label_column", "variant_id", "group", "block", "normalization", "scaling", "features",
            "n_pcs", "k_neighbors", "is_baseline", "n_proteins", "best_ari", "best_ari_resolution", "best_nmi",
            "best_nmi_resolution", "best_majority_accuracy", "macro_knn_purity", "macro_knn_purity_enrichment",
            "macro_silhouette", "macro_knn_purity_incl_unknown", "macro_silhouette_incl_unknown",
            "rho_pc1_ln_n_detected", "rho_pc2_ln_n_detected", "max_abs_rho_pc12", *keep_rank]
    summary = summary[cols]
    run.table(summary, "03_pipeline_summary")
    sizes = summary[(summary["label_role"] == "primary")].groupby(["dataset", "group"]).size().unstack()
    want_sizes = {"cellwise_scaled_all": 2, "unscaled_all": 3, "hvg": 18, "highcov50": 3, "pcs_or_k": 6, "submitted": 1}
    check("pipelines per group, every dataset", sizes.iloc[0].to_dict(), want_sizes,
          all(sizes.loc[d].to_dict() == want_sizes for d in DATASETS))
    for cond, spec in DCT.items():
        b = summary[(summary["dataset"] == cond) & (summary["label_role"] == "primary") & summary["is_baseline"]].iloc[0]
        r = t28_rank[(t28_rank["condition"] == cond) & (t28_rank["scope_role"] == "primary")
                     & (t28_rank["variant_id"] == t4.BASELINE_ID)].iloc[0]
        check(f"{cond}: baseline ranks carried over from dct_reviewer", (b["label_preservation_rank"], b["depth_independence_rank"]),
              (r["label_preservation_rank"], r["depth_independence_rank"]),
              b["depth_independence_rank"] == r["depth_independence_rank"])

    # ------------------------------------------------------------------ groups: range and number above the submitted value
    g_rows = []
    for (ds, role), sub in summary.groupby(["dataset", "label_role"], sort=False):
        base = sub[sub["group"] == "submitted"].iloc[0]
        for group, text in GROUPS.items():
            part = sub[sub["group"] == group]
            for m in SCORES:
                g_rows.append({"dataset": ds, "label_role": role, "label_column": base["label_column"], "group": group,
                               "group_display": text, "metric": m, "n_pipelines": len(part),
                               "submitted": float(base[m]), "min": float(part[m].min()), "max": float(part[m].max()),
                               "n_above_submitted": int((part[m] > base[m]).sum()),
                               "n_equal_submitted": int(np.isclose(part[m], base[m], rtol=0, atol=1e-12).sum()),
                               "pipeline_at_max": part.loc[part[m].idxmax(), "variant_id"]})
    groups = pd.DataFrame(g_rows)
    run.table(groups, "03_group_summary")

    # ------------------------------------------------------------------ detection, factors, descriptors
    feat = pd.concat([ref_feat.assign(dataset="kidney"), t28_feat.assign(dataset=t28_feat["condition"])], ignore_index=True)
    k_all = {"normalization": ["none"], "features": ["all"], "n_proteins": [4054], "dataset": ["kidney"],
             "median_detection_rate_selected": [float(np.median(det_rate))],
             "fraction_selected_detected_lt50pct": [float((det_rate < 0.5).mean())],
             "fraction_selected_detected_ge90pct": [float((det_rate >= 0.9).mean())]}
    feat = pd.concat([feat, pd.DataFrame(k_all)], ignore_index=True)
    feat = feat.drop_duplicates(["dataset", "normalization", "features"], keep="first")
    keep = ["dataset", "normalization", "features", "n_proteins", "median_detection_rate_selected",
            "fraction_selected_detected_lt50pct", "fraction_selected_detected_ge90pct"]
    run.table(feat[keep], "03_detection_summary")
    check("kidney all-protein median detection equals preprocessing (0.086)", round(float(np.median(det_rate)), 3), 0.086,
          round(float(np.median(det_rate)), 3) == 0.086)

    fac = [{"dataset": "kidney", "factor_min": float(ref_norm.loc[ref_norm["method"] == "total", "factor_min"].iloc[0]),
            "factor_max": float(ref_norm.loc[ref_norm["method"] == "total", "factor_max"].iloc[0])}]
    desc = [descriptors("kidney", intensity, cells)]
    k_ref = ref_desc[ref_desc["dataset"] == "kidney"].set_index("descriptor")
    d0 = desc[0].set_index("descriptor")
    diff = float((d0["spearman_rho_vs_ln_n_detected"] - k_ref["spearman_rho_vs_ln_n_detected"]).abs().max())
    check("kidney descriptors reproduce preprocessing step 1 (max abs diff in rho)", f"{diff:.2e}", "< 1e-9", diff < 1e-9)
    for cond, spec in DCT.items():
        path = cc.out_path("dct_labels", "h5ad", f"DCT_{spec['key']}_labelled", ".h5ad")
        obj = ad.read_h5ad(run.use_input(path, f"dct_labels labelled object ({cond})"))
        x = cc.dense(obj.layers["intensity"]).astype(np.float64)
        check(f"{cond}: retained cells and proteins", x.shape, (spec["n"], spec["p"]), x.shape == (spec["n"], spec["p"]))
        desc.append(descriptors(cond, x, obj.obs_names))
        info = t4.normalize(x, "total")[1]
        r = t28_norm[(t28_norm["condition"] == cond) & (t28_norm["method"] == "total")].iloc[0]
        diff = max(abs(info["factor_min"] - r["factor_min"]), abs(info["factor_max"] - r["factor_max"]))
        check(f"{cond}: total-intensity factors reproduce dct_reviewer (max abs diff)", f"{diff:.2e}", "< 1e-9", diff < 1e-9)
        fac.append({"dataset": cond, "factor_min": info["factor_min"], "factor_max": info["factor_max"]})
    desc = pd.concat(desc, ignore_index=True)
    run.table(desc, "03_intensity_descriptors")
    fac = pd.DataFrame(fac)
    fac["fold_range"] = fac["factor_max"] / fac["factor_min"]
    run.table(fac, "03_normalization_factors")
    run.table(clus, "03_kidney_t15_clustering_metrics")
    run.table(purity, "03_kidney_t15_knn_purity")
    run.table(sil, "03_kidney_t15_silhouette")

    # ------------------------------------------------------------------ depth-regressed embeddings (finished kidney_annotation, dct_reviewer results)
    reg: list[dict] = []
    gd = rd(KIDNEY_ANNOTATION, "kidney_annotation_graph_definitions.csv", "kidney_annotation graphs").set_index("graph")
    tb = rd(KIDNEY_ANNOTATION, "kidney_annotation_cluster_labels_per_cell_b_endo.csv", "kidney_annotation labels on the depth-regressed graph").set_index("cell_id")
    vs = rd(KIDNEY_ANNOTATION, "kidney_annotation_v3endo_summary.csv", "kidney_annotation types with the endothelial row")
    tb = tb.reindex(cells)
    same = float((tb[LABEL].astype(str).to_numpy() == lab).mean())
    ari_ab = float(adjusted_rand_score(lab, tb[LABEL].astype(str).to_numpy()))
    n_types = vs[(vs["resolution"] == 2.0) & (vs["marker_set"] == "v3_endo") & (vs["rule"] == "rankfree")].set_index("graph")
    reg += [{"dataset": "kidney", "measure": "largest |rho| of PC1 or PC2 with ln(IDs), submitted graph",
             "value": gd.at["stored", "max_abs_rho_pc1_pc2_ln_n_detected"], "source": "kidney_annotation_graph_definitions.csv"},
            {"dataset": "kidney", "measure": "largest |rho| of PC1 or PC2 with ln(IDs), depth-regressed graph",
             "value": gd.at["regressed", "max_abs_rho_pc1_pc2_ln_n_detected"], "source": "kidney_annotation_graph_definitions.csv"},
            {"dataset": "kidney", "measure": "cell types, submitted graph, Leiden 2.0, revised rule and markers",
             "value": n_types.at["stored", "n_types"], "source": "kidney_annotation_v3endo_summary.csv"},
            {"dataset": "kidney", "measure": "cell types, depth-regressed graph, Leiden 2.0, revised rule and markers",
             "value": n_types.at["regressed", "n_types"], "source": "kidney_annotation_v3endo_summary.csv"},
            {"dataset": "kidney", "measure": "fraction of cells with the same label on the two graphs", "value": same,
             "source": "kidney_annotation_cluster_labels_per_cell_{a,b}_endo.csv (this script)"},
            {"dataset": "kidney", "measure": "ARI of the labels on the two graphs", "value": ari_ab,
             "source": "kidney_annotation_cluster_labels_per_cell_{a,b}_endo.csv (this script)"}]
    pcd = rd(DCT_REVIEWER, "dct_reviewer_embed_pc_vs_depth_and_plate.csv", "dct_reviewer PCs vs depth, embeddings")
    pur = rd(DCT_REVIEWER, "dct_reviewer_embed_knn_label_purity.csv", "dct_reviewer kNN purity, embeddings")
    nmi = rd(DCT_REVIEWER, "dct_reviewer_embed_ari_nmi_clusters_vs_labels.csv", "dct_reviewer ARI and NMI, embeddings")
    for cond in DCT:
        for obj, name in (("uncorrected", "submitted graph"), ("depth_regressed", "depth-regressed graph")):
            p12 = pcd[(pcd["condition"] == cond) & (pcd["object"] == obj) & (pcd["pc"].isin([1, 2]))]
            reg.append({"dataset": cond, "measure": f"largest |rho| of PC1 or PC2 with ln(IDs), {name}",
                        "value": float(p12["spearman_rho_ln_ids"].abs().max()),
                        "source": "dct_reviewer_embed_pc_vs_depth_and_plate.csv"})
            sub = pur[(pur["condition"] == cond) & (pur["object"] == obj) & (pur["label_role"] == "primary")]
            for _, r in sub[sub["label"].isin(["DCT1", "DCT2"])].iterrows():
                reg.append({"dataset": cond, "measure": f"kNN purity of {r['label']}, {name}",
                            "value": r["observed_same_label_fraction"], "source": "dct_reviewer_embed_knn_label_purity.csv"})
            n1 = nmi[(nmi["condition"] == cond) & (nmi["object"] == obj) & (nmi["label_role"] == "primary")]
            reg.append({"dataset": cond, "measure": f"NMI of Leiden 1.0 clusters with the primary label, {name}",
                        "value": float(n1.loc[np.isclose(n1["resolution"].astype(float), 1.0), "nmi"].iloc[0]),
                        "source": "dct_reviewer_embed_ari_nmi_clusters_vs_labels.csv"})
            reg.append({"dataset": cond, "measure": f"best NMI of Leiden 0.3, 0.5, 0.8, 1.0, 3.0 with the primary label, {name}",
                        "value": float(n1["nmi"].max()), "source": "dct_reviewer_embed_ari_nmi_clusters_vs_labels.csv"})
    run.table(pd.DataFrame(reg), "03_depth_regressed")

    # ------------------------------------------------------------------ figure
    fig_rows: list[dict] = []
    plotted: list[dict] = []
    prim = summary[summary["label_role"] == "primary"]
    ok = draw_figure(run, prim, feat, plotted, fig_rows)
    check("figure passes the format checks (>= 7 pt, Arial, no text overlaps, text inside, 180 mm)", fig_rows[0]["pass"],
          True, ok)
    run.table(pd.DataFrame(fig_rows), f"{STEP}_figure_checks")
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    all_ok = all(c["pass"] for c in checks)
    run.finish({"label_column": LABEL, "n_permutations": args.n_permutations, "seed": t4.SEED, "purity_k": t4.PURITY_K,
                "min_label_n_for_macro": t4.MIN_LABEL_N, "macro_rule": "labels with >= 10 cells, Unknown left out (dct_reviewer)",
                "macro_labels": [c for c in cats_t15 if c != "Unknown" and EXPECTED_KIDNEY_ANNOTATION[c] >= t4.MIN_LABEL_N],
                "leiden_rerun": False, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit("A check failed; see report_03_preprocessing_scores_checks.csv.")


if __name__ == "__main__":
    main()
