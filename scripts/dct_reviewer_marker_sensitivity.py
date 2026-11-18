#!/usr/bin/env python
"""dct_reviewer step 8: reviewer_core D marker-list sensitivity of the dct_labels labels per DCT condition (combined search).

Port of ctrl_reviewer_core_marker_sensitivity.py, run once
per condition. Label versions: 'deg' = the primary column (cs_ctrl_deg_label / cs_zerok_pipeline_label;
cluster_de DE rule + kidney_annotation rank-free rescue on the primary Leiden 1.0 clusters, cluster DE fixed, only the marker
evidence changing) and 'cluster' = the second column (cs_<cond>_cluster_label; submitted scoring ->
per-cell argmax -> cluster mode on cs_<cond>_leiden3_cluster). Rules: labels_core_stability.Labeller
(imported, as ctrl_reviewer_core and dct_labels). ``label_variant`` and ``comparison_wide`` are the ctrl_reviewer_core functions (unchanged);
``variant_summary`` and ``downstream`` are copied with the label-version map passed in and the ALDO source
'scaled_X' (ctrl_reviewer_core 'ctrl_X'); ALDO contrasts of absent labels are not in the downstream comparison.

Inputs (read-only): dct_labels labelled objects and canonical labels (scaled .X, intensities, stored clusters,
X_pca, scores); data/DCT/DCT_markers_V1.xlsx; dct_labels_stability_marker_variants.csv (check); dct_reviewer outputs
dct_reviewer_task3_marker_scores_per_cell.csv, dct_reviewer_aldo_scores_per_cell.csv,
dct_reviewer_aldo_contrasts.csv, dct_reviewer_task4_knn_label_purity.csv and
results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_cc_regressed.h5ad (X_pca).
Outputs (subfolder marker_sensitivity/ of the dct_reviewer table and figure folders):
  dct_reviewer_marker_sensitivity_*.csv (both conditions; ``condition`` column) and figures.

D.1 Reproduce the dct_labels scores (max abs diff <= 1e-6), cs_<cond>_cell_label, the second column (incl. its
    Unknown clusters), the primary column and its confidence, and the per-variant changed-cell counts of
    dct_labels_stability_marker_variants.csv. Stops (exit 1) if any is not reproduced.
D.2 Variants noSlc12a3, noSlc12a3_noUmod and leave-one-out (each scored marker removed from every list).
D.3 Per variant: label counts, ProLIF share, crosstab, changed fraction, ties and all-zero scores,
    ProLIF fraction per plate, ProLIF vs non-ProLIF protein IDs.
D.4 Variants (a) and (b) vs original: ambiguity flags, cell_cycle_score by label, ProLIF kNN purity
    (uncorrected and cc-regressed, 1,000 permutations), ALDO ProLIF contrasts (BH within label set x source).
D.5 Leave-one-out stability by original label and marker impact.

Command (project root):  python scripts/dct_reviewer_marker_sensitivity.py
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
import sys

import numpy as np
import pandas as pd
from scipy import stats
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import labels_core_lib as lib18
import ctrl_reviewer_core_marker_sensitivity as m19
import dct_reviewer_lib as lib
from labels_core_stability import Labeller

ensure_scanpy_compat()
lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCRIPT = "dct_reviewer_marker_sensitivity.py"
SUB = "marker_sensitivity"
MAIN_VARIANTS = m19.MAIN_VARIANTS
VARIANT_SHORT = {"original": "original", "noSlc12a3": "(a) no Slc12a3", "noSlc12a3_noUmod": "(b) no Slc12a3, Umod"}
DCT_LABELS_VARIANTS = cc.out_path(lib.DCT_LABELS_TASK, "tables", "stability_marker_variants", ".csv")
GENERIC_VERSION = {"deg": "primary_label", "cluster": "second_label", "cell": "cell_label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 8: marker-list sensitivity of the dct_labels labels per condition.")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=1000)
    cc.add_common_args(parser)
    return parser.parse_args()


def versions_of(s: lib.Settings) -> dict[str, str]:
    return {"deg": s.PRIMARY, "cluster": s.SECOND}


# ---------------------------------------------------------------------------
# D.3 / D.4 (copies of the ctrl_reviewer_core functions; label-version map and ALDO prefix passed in)
# ---------------------------------------------------------------------------
def variant_summary(name: str, res: dict, labels: pd.DataFrame, versions: dict[str, str]):
    count_rows, cross_rows, change_rows = [], [], []
    ids = labels["n_detected_proteins"].to_numpy(float)
    plate = labels["plate"].to_numpy()
    for version, orig_col in versions.items():
        new = res[version].reindex(labels.index).to_numpy()
        orig = labels[orig_col].to_numpy()
        for label in cc.ordered_labels(np.concatenate([new, orig])):
            n = int((new == label).sum())
            count_rows.append({"variant": name, "label_version": version, "label_column": orig_col, "label": label,
                               "n_cells": n, "fraction": n / len(new), "n_cells_original": int((orig == label).sum())})
        table = pd.crosstab(pd.Series(orig, name="original_label"), pd.Series(new, name="variant_label"))
        for o, row in table.iterrows():
            for v, n in row.items():
                if n:
                    cross_rows.append({"variant": name, "label_version": version, "label_column": orig_col,
                                       "original_label": o, "variant_label": v, "n_cells": int(n)})
        is_p = new == "ProLIF"
        p1, p2 = is_p & (plate == "P1"), is_p & (plate == "P2")
        fisher = stats.fisher_exact([[int(p1.sum()), int((plate == "P1").sum() - p1.sum())],
                                     [int(p2.sum()), int((plate == "P2").sum() - p2.sum())]])
        ids_cmp = cc.compare_groups(ids[is_p], ids[~is_p])
        row = {"variant": name, "label_version": version, "label_column": orig_col, "n_cells": len(new),
               "n_changed": int((new != orig).sum()), "fraction_changed": float((new != orig).mean()),
               "n_prolif": int(is_p.sum()), "prolif_share": float(is_p.mean()),
               "n_ties_max_score": int(res["ties"].sum()), "n_all_zero_scores": int(res["all_zero"].sum()),
               "P1_prolif_fraction": p1.sum() / (plate == "P1").sum(), "P2_prolif_fraction": p2.sum() / (plate == "P2").sum(),
               "fisher_odds_ratio_P1_vs_P2": fisher[0] if is_p.any() else np.nan,
               "fisher_p_plate": fisher[1] if is_p.any() else np.nan,
               "median_ids_prolif": ids_cmp["median_group"], "median_ids_non_prolif": ids_cmp["median_reference"],
               "mannwhitney_p_ids": ids_cmp["p_value"], "rank_biserial_ids": ids_cmp["rank_biserial"],
               "lists_present": " | ".join(f"{ct}: {';'.join(g)}" for ct, g in res["present"].items())}
        if version == "cluster":
            row.update(unknown_clusters=";".join(res["unknown_clusters"]), n_unknown_cells=res["cluster_info"]["n_unknown_cells"],
                       unknown_score_threshold=res["cluster_info"]["score_threshold"])
        change_rows.append(row)
    return count_rows, cross_rows, change_rows


def downstream(label_sets: dict[str, pd.DataFrame], task3: pd.DataFrame, aldo: pd.DataFrame,
               embeddings: dict[str, np.ndarray], args, versions: dict[str, str]):
    rows, purity_frames, aldo_frames = [], [], []
    score_cols = [c for c in aldo.columns if c.startswith(("scaled_X__", "cc_regressed_3a__"))]
    for variant, frame in label_sets.items():
        for version in versions:
            lab = frame[version]
            for label in cc.ordered_labels(lab):
                m = (lab == label).to_numpy()
                flagged = task3["ambiguous_tal_cnt_interstitial"].to_numpy(bool)[m]
                ccs = task3["cell_cycle_score"].to_numpy(float)[m]
                rows += [
                    {"variant": variant, "label_version": version, "metric_group": "ambiguity_flags", "label": label,
                     "quantity": "fraction flagged (TAL/CNT/interstitial)", "value": flagged.mean(),
                     "n": int(flagged.sum()), "n_total": int(m.sum())},
                    {"variant": variant, "label_version": version, "metric_group": "cell_cycle_score", "label": label,
                     "quantity": "median cell_cycle_score", "value": float(np.median(ccs)), "n_total": int(m.sum()),
                     "q25": float(np.quantile(ccs, 0.25)), "q75": float(np.quantile(ccs, 0.75))}]
        frame_labels = pd.DataFrame({f"{variant}__{v}": frame[v] for v in versions})
        pur = t2.knn_purity(embeddings, frame_labels, list(frame_labels.columns), args.knn_k, args.n_permutations)
        pur.insert(0, "variant", variant)
        pur["label_version"] = pur["label_column"].str.split("__").str[1]
        purity_frames.append(pur)
        for version in versions:
            per_cell = aldo[["exclude_tal_cnt_interstitial", *score_cols]].copy()
            per_cell["variant_label"] = frame[version].reindex(per_cell.index).to_numpy()
            contrast = t2.aldo_contrast_table(per_cell, "variant_label", score_cols)
            contrast.insert(0, "label_variant", variant)
            contrast.insert(1, "label_version", version)
            contrast["status"] = np.where((contrast["n_group"] > 0) & (contrast["n_reference"] > 0), "tested",
                                          lib.NOT_APPLICABLE)
            aldo_frames.append(contrast.drop(columns="label_column"))
    purity = pd.concat(purity_frames, ignore_index=True)
    aldo_contrasts = pd.concat(aldo_frames, ignore_index=True)
    for _, r in purity[purity["label"] == "ProLIF"].iterrows():
        rows.append({"variant": r["variant"], "label_version": r["label_version"], "metric_group": "knn_purity_prolif",
                     "label": "ProLIF", "quantity": f"observed same-label fraction, {r['object']}",
                     "value": r["observed_same_label_fraction"], "n_total": r["n_cells"], "null_mean": r["null_mean"],
                     "null_q975": r["null_q975"], "p_empirical_ge": r["p_empirical_ge"]})
    sel = aldo_contrasts[aldo_contrasts["comparison"].str.startswith("ProLIF")
                         & aldo_contrasts["label_variant"].isin(["original", *MAIN_VARIANTS])
                         & aldo_contrasts["source"].isin(["scaled_X", "cc_regressed_3a"])
                         & aldo_contrasts["variant"].isin(["base", "no_Slc12a3"])
                         & (aldo_contrasts["status"] == "tested")]
    for _, r in sel.iterrows():
        rows.append({"variant": r["label_variant"], "label_version": r["label_version"], "metric_group": "aldo_prolif_contrast",
                     "label": r["comparison"], "quantity": f"{r['source']} {r['family']} {r['variant']} [{r['cohort']}]: median difference",
                     "value": r["median_difference"], "n": r["n_group"], "n_total": r["n_reference"],
                     "p_value": r["p_value"], "p_fdr_bh": r["p_fdr_bh"]})
    return pd.DataFrame(rows), purity, aldo_contrasts


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------
def one_condition(run: cc.Run, cond: str, args) -> dict:
    s = lib.SETTINGS[cond]
    versions = versions_of(s)
    adata, labels = lib.load(run, cond)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    X = np.array(cc.dense(adata.X), dtype=np.float32, copy=True)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected_m = intensity > 0
    x_z = pd.DataFrame(X.astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected_m, index=cell_ids, columns=var_names)
    lists = lib18.present_lists(lib18.read_workbook(run), var_names)
    markers = lib18.dct_markers(run, var_names)
    L = Labeller(adata, lists, markers, t5.rank_cache(np.log1p(intensity), detected_m), x_z, det_frame, var_names)
    cl3 = labels[s.LEIDEN3_CLUSTER].astype(str).to_numpy()
    cl1 = labels[s.PRIMARY_CLUSTER].astype(str).to_numpy()
    _, _, de1 = L.deg(cl1)   # cluster DE on the primary clusters; fixed for every variant (as labels_core/dct_labels)

    # D.1 reproduction of the dct_labels scores and labels (stop if it fails).
    orig = m19.label_variant(L, X, lists, markers, cl3, cl1, de1)
    checks = []
    for ct in lists:
        col = f"{ct}_score"
        diff = float(np.abs(orig["scores"][ct].to_numpy() - labels[col].to_numpy(float)).max())
        checks.append({"condition": cond, "check": f"{col} reproduced (max abs diff vs the dct_labels canonical table)",
                       "value": diff, "pass": diff < 1e-6})
    for key, col in (("cell", s.CELL_COLUMN), ("cluster", s.SECOND), ("deg", s.PRIMARY), ("deg_confidence", s.CONFIDENCE)):
        n = int((orig[key].to_numpy() == labels[col].astype(str).to_numpy()).sum())
        checks.append({"condition": cond, "check": f"re-derived {col} equals dct_labels (cells)", "value": f"{n}/{len(labels)}",
                       "pass": n == len(labels)})
    checks.append({"condition": cond, "check": f"'Unknown' cluster(s) of {s.SECOND} re-derived",
                   "value": ";".join(orig["unknown_clusters"]),
                   "pass": orig["cluster_info"]["n_unknown_cells"] == int((labels[s.SECOND] == "Unknown").sum())})
    checks.append({"condition": cond, "check": "ties in max score (original lists)", "value": int(orig["ties"].sum()),
                   "pass": True})

    scored = list(dict.fromkeys(g for genes in lists.values() for g in genes))
    variant_genes = {"original": (), **{k: v[1] for k, v in lib18.MARKER_VARIANTS.items()},
                     **{f"loo_{g}": (g,) for g in scored}}
    variants = {}
    for name, genes in variant_genes.items():
        variants[name] = orig if name == "original" else m19.label_variant(
            L, X, lib18.drop_from_lists(lists, genes), lib18.drop_markers(markers, genes), cl3, cl1, de1)
    t27 = pd.read_csv(run.use_input(DCT_LABELS_VARIANTS, "dct_labels marker-variant table (check)"))
    t27 = t27[t27["condition"] == cond]
    key_of = {s.CELL_COLUMN: "cell", s.SECOND: "cluster", s.PRIMARY: "deg"}
    n_bad, missing_variants = 0, []
    for _, r in t27.iterrows():
        res = variants.get(r["variant"])
        if res is None:
            missing_variants.append(r["variant"])
        ours = (-1 if res is None else
                int((res[key_of[r["label_column"]]].to_numpy() != labels[r["label_column"]].astype(str).to_numpy()).sum()))
        n_bad += int(ours != int(r["n_changed"]))
    checks.append({"condition": cond, "check": "cells changed per variant and label column equal "
                   "dct_labels_stability_marker_variants.csv", "value": f"{len(t27) - n_bad}/{len(t27)} rows equal"
                   + (f"; variants not run here: {sorted(set(missing_variants))}" if missing_variants else ""),
                   "pass": n_bad == 0 and len(t27) > 0})
    out = {"checks": checks, "labels": labels, "versions": versions, "scored": scored}
    if not all(c["pass"] for c in checks):
        return out

    out["definitions"] = pd.DataFrame([{"variant": name, "description": m19.VARIANT_TEXT.get(
        name, f"leave-one-out: {name[4:]} removed from every list"), "removed": ";".join(variant_genes[name]),
        **{f"{ct}_markers_scored": ";".join(res["present"].get(ct, [])) for ct in lists},
        **{f"{ct}_specific_for_de_rule": ";".join(lib18.specific_present(res["markers"]).get(ct, [])) for ct in lists}}
        for name, res in variants.items()])

    per_cell = labels[["cell_key", "plate", "n_detected_proteins", *s.LABEL_COLUMNS, s.CELL_COLUMN]].copy()
    for name in MAIN_VARIANTS:
        for version in ("deg", "cluster", "cell"):
            per_cell[f"{GENERIC_VERSION[version]}_{name}"] = variants[name][version].reindex(labels.index).to_numpy()
    loo_names = [n for n in variants if n.startswith("loo_")]
    for version, col in versions.items():
        keep = np.column_stack([variants[n][version].reindex(labels.index).to_numpy() == labels[col].to_numpy()
                                for n in loo_names])
        per_cell[f"label_stability_loo_{version}"] = keep.mean(axis=1)
    for name in ("original", *MAIN_VARIANTS):
        for ct in lists:
            per_cell[f"score_{ct}_{name}"] = variants[name]["scores"][ct].reindex(labels.index).to_numpy()
    out["per_cell"] = lib.to_generic(per_cell, cond).reset_index()
    out["_per_cell"] = per_cell
    loo_wide = pd.DataFrame(index=labels.index)
    for n in loo_names:
        for version in ("deg", "cluster", "cell"):
            loo_wide[f"{n}__{version}"] = variants[n][version].reindex(labels.index).to_numpy()
    out["loo_wide"] = loo_wide.reset_index()

    counts, cross, changes = [], [], []
    for name, res in variants.items():
        a, b, c = variant_summary(name, res, labels, versions)
        counts += a
        cross += b
        changes += c
    out["counts"], out["cross"], out["changes"] = pd.DataFrame(counts), pd.DataFrame(cross), pd.DataFrame(changes)

    task3 = lib.read_table(run, "task3_marker_scores_per_cell", dtype={"cell_id": str})
    task3 = task3[task3["condition"] == cond].set_index("cell_id").reindex(labels.index)
    aldo = lib.read_table(run, "aldo_scores_per_cell", dtype={"cell_id": str})
    aldo = aldo[aldo["condition"] == cond].set_index("cell_id").reindex(labels.index)
    cc_reg = lib.read_h5ad(run, f"DCT_{s.KEY}_cc_regressed")
    embeddings = {"uncorrected": adata.obsm["X_pca"][:, :t2.N_PCS], "cc_regressed": cc_reg.obsm["X_pca"][:, :t2.N_PCS]}
    label_sets = {name: pd.DataFrame({v: variants[name][v].reindex(labels.index) for v in versions})
                  for name in ("original", *MAIN_VARIANTS)}
    out["label_sets"] = label_sets
    long, purity, aldo_contrasts = downstream(label_sets, task3, aldo, embeddings, args, versions)
    out["long"], out["purity"], out["aldo_contrasts"] = long, purity, aldo_contrasts
    out["wide"] = m19.comparison_wide(long)

    # Consistency: 'original' rows must equal the step 2 purity and step 3 ALDO tables.
    a2 = lib.read_table(run, "task4_knn_label_purity")
    a2 = a2[a2["condition"] == cond]
    a3 = lib.read_table(run, "aldo_contrasts")
    a3 = a3[(a3["condition"] == cond) & (a3["status"] == "tested")]
    p_orig = purity[purity["variant"] == "original"].assign(label_column=lambda d: d["label_version"].map(versions))
    m = p_orig.merge(a2, on=["object", "label_column", "label"], suffixes=("", "_a2"))
    c_orig = aldo_contrasts[(aldo_contrasts["label_variant"] == "original") & (aldo_contrasts["status"] == "tested")].assign(
        label_column=lambda d: d["label_version"].map(versions))
    m2 = c_orig.merge(a3, on=["label_column", "cohort", "comparison", "score"], suffixes=("", "_a3"))
    out["consistency"] = {
        "knn_purity_rows": len(m),
        "knn_purity_max_abs_diff": float(np.abs(m["observed_same_label_fraction"] - m["observed_same_label_fraction_a2"]).max()),
        "knn_null_mean_max_abs_diff": float(np.abs(m["null_mean"] - m["null_mean_a2"]).max()),
        "aldo_rows": len(m2), "aldo_rows_step3": len(a3),
        "aldo_median_difference_max_abs_diff": float(np.abs(m2["median_difference"] - m2["median_difference_a3"]).max()),
        "aldo_q_max_abs_diff": float(np.abs(m2["p_fdr_bh"] - m2["p_fdr_bh_a3"]).max())}

    stab_rows = []
    for version, col in versions.items():
        stab = per_cell[f"label_stability_loo_{version}"]
        for label in cc.ordered_labels(labels[col]):
            v = stab[labels[col] == label]
            stab_rows.append({"label_version": version, "label_column": col, "original_label": label, "n_cells": len(v),
                              "n_loo_runs": len(loo_names), "mean_stability": v.mean(), "median_stability": v.median(),
                              "n_stability_1": int((v == 1).sum()), "n_stability_below_0.5": int((v < 0.5).sum()),
                              "n_stability_0": int((v == 0).sum())})
    out["stability"] = pd.DataFrame(stab_rows)
    changes = out["changes"]
    impact = changes[changes["variant"].str.startswith("loo_")].copy()
    impact.insert(0, "dropped_marker", impact["variant"].str.removeprefix("loo_"))
    cross = out["cross"]
    by_label = (cross[cross["variant"].str.startswith("loo_") & (cross["original_label"] != cross["variant_label"])]
                .groupby(["variant", "label_version", "original_label"])["n_cells"].sum().unstack(fill_value=0)
                .add_prefix("n_changed_from_").reset_index())
    impact = impact.merge(by_label, on=["variant", "label_version"], how="left")
    impact = impact.fillna({c: 0 for c in impact.columns if c.startswith("n_changed_from_")})
    impact = impact.sort_values(["label_version", "n_changed"], ascending=[True, False])
    out["impact"] = impact[["dropped_marker", "label_version", "label_column", "n_changed", "fraction_changed",
                            "n_prolif", "prolif_share", *[c for c in impact.columns if c.startswith("n_changed_from_")],
                            "n_ties_max_score", "lists_present"]]
    out["loo_names"] = loo_names
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def alluvial(ax, orig: np.ndarray, new: np.ndarray, title: str) -> None:
    """ctrl_reviewer_core alluvial at 7 pt; stack labels pushed apart so they do not overlap."""
    order = cc.ordered_labels(np.concatenate([orig, new]))
    total = len(orig)
    gap = 0.015 * total

    def stacks(values):
        pos, y = {}, 0.0
        for lab in order:
            n = int((values == lab).sum())
            if n:
                pos[lab] = [y, y + n]
                y += n + gap
        return pos

    left, right = stacks(orig), stacks(new)
    lcur = {k: v[0] for k, v in left.items()}
    rcur = {k: v[0] for k, v in right.items()}
    xs = np.linspace(0.12, 0.88, 60)
    sig = 1 / (1 + np.exp(-12 * (xs - 0.5)))
    for o in order:
        for n_lab in order:
            n = int(((orig == o) & (new == n_lab)).sum())
            if not n:
                continue
            y0l, y0r = lcur[o], rcur[n_lab]
            lcur[o] += n
            rcur[n_lab] += n
            bottom = y0l + (y0r - y0l) * sig
            ax.fill_between(xs, bottom, bottom + n, color=t2.LABEL_PALETTE.get(o, "0.5"), alpha=0.35 if o != n_lab else 0.6,
                            linewidth=0)
    min_sep = 0.11 * total
    for side, pos, x0 in (("orig", left, 0.0), ("new", right, 0.88)):
        last = -np.inf
        for lab, (a, b) in pos.items():
            ax.add_patch(plt.Rectangle((x0, a), 0.12, b - a, color=t2.LABEL_PALETTE.get(lab, "0.5"), linewidth=0))
            y = max((a + b) / 2, last + min_sep)
            last = y
            ax.text(x0 - 0.02 if side == "orig" else x0 + 0.14, y, f"{lab} {int(round(b - a))}",
                    ha="right" if side == "orig" else "left", va="center", fontsize=7)
    changed = int((orig != new).sum())
    ax.set_title(f"{title}\nchanged {changed}/{total} ({100 * changed / total:.1f}%)", fontsize=7)
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(total * 1.12, -total * 0.02)
    t2.clean_axes(ax)


def plot_flows(res: dict, figs: lib.Figures) -> None:
    rows = [(cond, version) for version in ("deg", "cluster") for cond in lib.CONDITIONS]
    fig = lib.new_figure(lib.FULL_MM, 196)
    gs = fig.add_gridspec(len(rows), 2, left=0.01, right=0.99, top=0.92, bottom=0.02, hspace=0.5, wspace=0.05)
    for r, (cond, version) in enumerate(rows):
        d = res[cond]
        col = d["versions"][version]
        for c, variant in enumerate(MAIN_VARIANTS):
            ax = fig.add_subplot(gs[r, c])
            alluvial(ax, d["labels"][col].to_numpy(), d["label_sets"][variant][version].reindex(d["labels"].index).to_numpy(),
                     f"{cond}, {col}: {VARIANT_SHORT[variant]}")
    fig.text(0.005, 0.995, "Label changes when Slc12a3 (a) and also Umod (b) are removed from the DCT V1 workbook lists "
             "(left stack = original label, right stack = variant label;\nribbon colour = original label; rows: primary "
             "columns, then second columns)", ha="left", va="top", fontsize=7)
    figs.save(fig, "marker_sensitivity_label_flows", "Original versus variant labels per condition", subdir=SUB)


def plot_counts(res: dict, figs: lib.Figures) -> None:
    variants = ["original", *MAIN_VARIANTS]
    fig = lib.new_figure(lib.FULL_MM, 150)
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.99, top=0.9, bottom=0.14, hspace=0.6, wspace=0.35,
                          width_ratios=[1.4, 1, 1])
    for r, cond in enumerate(lib.CONDITIONS):
        d = res[cond]
        counts, changes = d["counts"], d["changes"]
        ax = fig.add_subplot(gs[r, 0])
        xpos, ticks, groups = 0, [], []
        for version in ("deg", "cluster"):
            groups.append((xpos + (len(variants) - 1) / 2, "primary" if version == "deg" else "second"))
            for variant in variants:
                sub = counts[(counts["variant"] == variant) & (counts["label_version"] == version)].set_index("label")
                bottom = 0
                for lab in cc.ordered_labels(sub.index):
                    v = 100 * sub.loc[lab, "fraction"]
                    ax.bar(xpos, v, bottom=bottom, color=t2.LABEL_PALETTE.get(lab, "0.5"), width=0.75, edgecolor="white",
                           linewidth=0.6)
                    if v >= 9:
                        ax.text(xpos, bottom + v / 2, f"{int(sub.loc[lab, 'n_cells'])}", ha="center", va="center",
                                fontsize=7, color="white", rotation=90)
                    bottom += v
                ticks.append((xpos, {"original": "orig.", "noSlc12a3": "(a)", "noSlc12a3_noUmod": "(b)"}[variant]))
                xpos += 1
            xpos += 0.6
        ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks])
        for x, text in groups:
            ax.text(x, -0.17, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
        ax.set_ylabel(f"% of {len(d['labels'])} cells")
        ax.set_title(f"{cond}: label composition (numbers = cells)", fontsize=7)
        for k, version in enumerate(("deg", "cluster")):
            ax = fig.add_subplot(gs[r, 1 + k])
            sub = changes[(changes["label_version"] == version) & changes["variant"].isin(variants)].set_index(
                "variant").reindex(variants)
            col = d["versions"][version]
            if int(sub["n_prolif"].sum()) == 0:
                lib.na_axis(ax, f"{cond}, {col}:\nProLIF by plate\n{lib.NOT_APPLICABLE}\n(no ProLIF cell in any variant)")
                continue
            x = np.arange(len(variants))
            ax.bar(x - 0.18, 100 * sub["P1_prolif_fraction"], width=0.34, color=cc.PLATE_PALETTE["P1"], label="P1")
            ax.bar(x + 0.18, 100 * sub["P2_prolif_fraction"], width=0.34, color=cc.PLATE_PALETTE["P2"], label="P2")
            ax.set_xticks(x, ["original", "(a)", "(b)"])
            ax.set_ylim(0, 40)
            ax.set_ylabel("% of plate labelled ProLIF")
            ax.set_title(f"{cond}, {col}:\nProLIF by plate", fontsize=7)
            lib.grid_y(ax)
            ax.legend(loc="upper right", frameon=False)
    fig.text(0.005, 0.995, "Label counts and ProLIF fraction per plate by marker-list variant ((a) Slc12a3 removed; "
             "(b) Slc12a3 and Umod removed; orig. = original lists);\nFisher p for plate x ProLIF and median IDs ProLIF / "
             "other are in dct_reviewer_marker_sensitivity_change_summary.csv", ha="left", va="top", fontsize=7)
    handles = [Patch(color=t2.LABEL_PALETTE[l], label=l) for l in t2.LABEL_PALETTE]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=len(handles), frameon=False)
    figs.save(fig, "marker_sensitivity_counts_and_plate", "Label counts and ProLIF fraction per plate by variant",
              subdir=SUB)


def plot_loo(res: dict, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 150)
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.96, top=0.9, bottom=0.1, hspace=0.6, wspace=0.42,
                          width_ratios=[1, 1, 1.1])
    rng = np.random.default_rng(0)  # jitter only
    xmax = max(float(res[c]["impact"]["n_changed"].max()) for c in lib.CONDITIONS)
    for r, cond in enumerate(lib.CONDITIONS):
        d = res[cond]
        per_cell, labels = d["_per_cell"], d["labels"]
        for k, (version, col) in enumerate(d["versions"].items()):
            ax = fig.add_subplot(gs[r, k])
            stab = f"label_stability_loo_{version}"
            order = cc.ordered_labels(labels[col])
            for i, lab in enumerate(order):
                v = per_cell.loc[labels[col] == lab, stab].to_numpy(float)
                ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v + rng.uniform(-0.012, 0.012, len(v)), s=2.5,
                           color=t2.LABEL_PALETTE.get(lab, "0.5"), alpha=0.5, linewidths=0, rasterized=True)
                ax.scatter([i], [v.mean()], marker="D", s=12, color=t2.INK["primary"], zorder=3)
            ax.set_xticks(range(len(order)), [f"{lab}\n{int((per_cell.loc[labels[col] == lab, stab] == 1).sum())}/"
                                              f"{int((labels[col] == lab).sum())}" for lab in order])
            ax.set_ylim(-0.05, 1.05)
            if k == 0:
                ax.set_ylabel(f"fraction of {len(d['loo_names'])} runs keeping the label")
            ax.set_title(f"{cond}: {col}", fontsize=7)
            lib.grid_y(ax)
        ax = fig.add_subplot(gs[r, 2])
        impact = d["impact"]
        markers = [m for m in d["scored"] if m in set(impact["dropped_marker"])]
        y = np.arange(len(markers))
        for k, (version, color) in enumerate((("deg", t2.SERIES[0]), ("cluster", t2.SERIES[1]))):
            sub = impact[impact["label_version"] == version].set_index("dropped_marker").reindex(markers)
            ax.barh(y + (k - 0.5) * 0.38, sub["n_changed"], height=0.36, color=color,
                    label="primary" if version == "deg" else "second")
        ax.set_yticks(y, markers)
        ax.set_ylim(len(markers) - 0.5, -0.5)
        ax.set_xlim(0, xmax * 1.05)
        ax.set_xlabel("cells whose label changes")
        ax.set_title(f"{cond}: leave-one-marker-out impact", fontsize=7)
        ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        if r == 0:
            ax.legend(loc="upper right", frameon=False)
    fig.text(0.005, 0.995, "Leave-one-marker-out label stability (dot = cell; diamond = mean; tick: label and cells "
             "keeping it in all runs / cells) and marker impact", ha="left", va="top", fontsize=7)
    figs.save(fig, "marker_sensitivity_loo", "Leave-one-marker-out label stability and marker impact", subdir=SUB)


def plot_downstream(res: dict, figs: lib.Figures) -> None:
    variants = ["original", *MAIN_VARIANTS]
    rows = [(cond, version) for version in ("deg", "cluster") for cond in lib.CONDITIONS]
    fig = lib.new_figure(lib.FULL_MM, 200)
    gs = fig.add_gridspec(len(rows), 3, left=0.075, right=0.98, top=0.92, bottom=0.08, hspace=0.75, wspace=0.35)
    for r, (cond, version) in enumerate(rows):
        d = res[cond]
        col = d["versions"][version]
        sub = d["long"][d["long"]["label_version"] == version]
        for c, group in enumerate(("ambiguity_flags", "cell_cycle_score")):
            ax = fig.add_subplot(gs[r, c])
            part = sub[sub["metric_group"] == group]
            labs = [l for l in cc.ordered_labels(part["label"]) if l in t2.CORE_LABELS]
            for k, variant in enumerate(variants):
                vals = part[part["variant"] == variant].set_index("label").reindex(labs)
                x = np.arange(len(labs)) + (k - 1) * 0.26
                yv = vals["value"].to_numpy(float) * (100 if group == "ambiguity_flags" else 1)
                ax.bar(x, np.nan_to_num(yv), width=0.24, color=t2.SERIES[k], label=VARIANT_SHORT[variant])
            ax.set_xticks(range(len(labs)), labs)
            ax.set_ylabel("% flagged" if group == "ambiguity_flags" else "median cc score")
            ax.set_title(f"{cond}, {col}:\n{'ambiguity flags' if group == 'ambiguity_flags' else 'cell-cycle score'}",
                         fontsize=7)
            ax.axhline(0, color=t2.INK["muted"], linewidth=0.5)
            lib.grid_y(ax)
        ax = fig.add_subplot(gs[r, 2])
        part = sub[sub["metric_group"] == "knn_purity_prolif"]
        if part.empty:
            lib.na_axis(ax, f"{cond}, {col}:\nProLIF kNN purity\n{lib.NOT_APPLICABLE} (no ProLIF cell)")
            continue
        objects = list(dict.fromkeys(part["quantity"]))
        for k, variant in enumerate(variants):
            vals = part[part["variant"] == variant].set_index("quantity").reindex(objects)
            x = np.arange(len(objects)) + (k - 1) * 0.26
            ax.vlines(x, vals["null_mean"], vals["null_q975"], color=t2.INK["background_cells"], linewidth=3)
            ax.scatter(x, vals["value"], color=t2.SERIES[k], s=12, zorder=3)
        ax.set_xticks(range(len(objects)), [o.split(", ")[-1] for o in objects])
        ax.set_ylim(0, 1)
        ax.set_ylabel("ProLIF same-label fraction")
        ax.set_title(f"{cond}, {col}:\nProLIF kNN purity", fontsize=7)
        lib.grid_y(ax)
    handles = [Patch(color=t2.SERIES[k], label=VARIANT_SHORT[v]) for k, v in enumerate(variants)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.0), ncol=3, frameon=False)
    fig.text(0.005, 0.995, "Downstream summaries under the original and variant labels (rows: primary columns, then "
             "second columns; kNN purity: grey = null mean to 97.5%;\nALDO contrasts are in "
             "dct_reviewer_marker_sensitivity_downstream_comparison.csv)", ha="left", va="top", fontsize=7)
    figs.save(fig, "marker_sensitivity_downstream", "Downstream summaries under the original and variant labels",
              subdir=SUB)


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "marker_sensitivity", args)
    lib.ported_from(run, "ctrl_reviewer_core_marker_sensitivity.py", "labels_core_stability.py")
    res = {c: one_condition(run, c, args) for c in lib.CONDITIONS}
    sanity = pd.DataFrame([row for c in lib.CONDITIONS for row in res[c]["checks"]])
    run.table(sanity, "marker_sensitivity_reproduction", subdir=SUB)
    if not sanity["pass"].all():
        run.note("Reproduction FAILED; section D stopped. See dct_reviewer_marker_sensitivity_reproduction.csv.")
        run.finish(status="stopped: reproduction failed")
        sys.exit(1)
    run.note("Reproduction passed in both conditions: dct_labels scores (max abs diff <= 1e-6), cell, second and primary labels "
             "and primary confidence for every cell; dct_labels variant table rows equal.")
    for key, name in (("definitions", "marker_sensitivity_variant_definitions"),
                      ("per_cell", "marker_sensitivity_labels_per_cell"),
                      ("loo_wide", "marker_sensitivity_loo_labels_per_cell"),
                      ("counts", "marker_sensitivity_label_counts"), ("cross", "marker_sensitivity_crosstabs"),
                      ("changes", "marker_sensitivity_change_summary"), ("long", "marker_sensitivity_downstream_long"),
                      ("wide", "marker_sensitivity_downstream_comparison"), ("purity", "marker_sensitivity_knn_purity"),
                      ("aldo_contrasts", "marker_sensitivity_aldo_contrasts"),
                      ("stability", "marker_sensitivity_loo_stability_by_label"),
                      ("impact", "marker_sensitivity_loo_marker_impact")):
        run.table(lib.stack({c: res[c][key] for c in lib.CONDITIONS}), name, subdir=SUB)
    consistency = pd.DataFrame([{"condition": c, **res[c]["consistency"]} for c in lib.CONDITIONS])
    run.table(consistency, "marker_sensitivity_consistency_vs_steps_2_3", subdir=SUB)
    run.note("Per-cell variant columns use generic names: primary_label_<variant>, second_label_<variant>, "
             "cell_label_<variant>.")
    figs = lib.Figures(run, "marker_sensitivity")
    plot_flows(res, figs)
    plot_counts(res, figs)
    plot_loo(res, figs)
    plot_downstream(res, figs)
    figs.finish()
    run.finish({"scored_markers": {c: res[c]["scored"] for c in lib.CONDITIONS},
                "consistency_vs_steps_2_3": {c: res[c]["consistency"] for c in lib.CONDITIONS},
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_marker_sensitivity.csv.")


if __name__ == "__main__":
    main()
