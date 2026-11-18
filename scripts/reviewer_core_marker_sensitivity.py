#!/usr/bin/env python
"""reviewer_core D: marker-list sensitivity of the V1 labels (Slc12a3, Umod, leave-one-marker-out).



Inputs (read-only):
  kidney_object canonical DCT object and labels (scaled .X, stored Leiden 3.0 IDs, X_pca);
  data/DCT/DCT_markers_V1.xlsx; data/DCT/V1/DCT_758cells_processed.h5ad (obs only: stored
  DCT1/DCT2/ProLIF scores and max_score, for the reproduction check);
  reviewer_core outputs reviewer_core_task3_marker_scores_per_cell.csv (flags,
  cell_cycle_score), reviewer_core_aldo_scores_per_cell.csv,
  reviewer_core_aldo_contrasts.csv and reviewer_core_task4_knn_label_purity.csv
  (consistency checks of the 'original' rows), and
  results/h5ad/reviewer_core_DCT_cc_regressed.h5ad (X_pca).
Outputs (results/tables/reviewer_core/marker_sensitivity/ and results/figures/reviewer_core/marker_sensitivity/):
  reviewer_core_marker_sensitivity_*.csv / .pdf / .png

D.1 Reproduce the stored V1 scores and predicted_cell_type with the submission's
    scoring (score_genes defaults on the scaled .X; per-cell argmax), then the
    cluster labels on the stored Leiden 3.0 clusters (cluster mode; the cluster
    with mean max_score below the 1% quantile of cluster means = 'Unknown').
    The script stops (exit 1) if either is not reproduced exactly.
D.2 Variants (same scoring and labelling; only the lists change):
    noSlc12a3 (Slc12a3 removed from all three lists), noSlc12a3_noUmod (also Umod
    removed from DCT1), and leave-one-out: each detected marker removed from
    every list that contains it. label_stability_loo = fraction of the
    leave-one-out runs that keep the cell's original label (per-cell label);
    label_stability_loo_cluster = the same for the cluster label.
D.3 Per variant: label counts, ProLIF share, crosstab and changed fraction,
    ties and all-zero scores, per-plate ProLIF fraction, ProLIF vs non-ProLIF
    median protein IDs.
D.4 Variants noSlc12a3 and noSlc12a3_noUmod: ambiguity-flag fractions,
    cell_cycle_score by label, ProLIF kNN purity (uncorrected and
    cell-cycle-regressed, 1,000 permutations), ALDO ProLIF contrasts (BH within
    label set x expression source, as in A.3), beside the original V1 values.
D.5 Leave-one-out stability by original label and marker impact.

Command (project root):  python scripts/reviewer_core_marker_sensitivity.py
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

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2

ensure_scanpy_compat()
t2.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCRIPT = "reviewer_core_marker_sensitivity.py"
SUB = "marker_sensitivity"
VERSIONS = {"cell": "v1_cell_label", "cluster": "v1_cluster_label"}
MAIN_VARIANTS = ("noSlc12a3", "noSlc12a3_noUmod")
VARIANT_TEXT = {"original": "original V1 lists", "noSlc12a3": "(a) Slc12a3 removed from all lists",
                "noSlc12a3_noUmod": "(b) Slc12a3 removed; Umod removed from DCT1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reviewer_core D: marker-list sensitivity of the V1 labels.")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=1000)
    cc.add_common_args(parser)
    return parser.parse_args()


def label_variant(adata, lists: dict[str, list[str]], clusters: pd.Series) -> dict:
    scores, present = t2.score_v1_lists(adata, lists)
    cell, max_score, ties, all_zero = t2.cell_labels(scores)
    cluster, unknown, info = t2.cluster_labels(cell, max_score, clusters)
    return {"scores": scores, "present": present, "cell": cell, "max_score": max_score, "ties": ties,
            "all_zero": all_zero, "cluster": cluster, "unknown_clusters": unknown, "cluster_info": info}


def drop(lists: dict[str, list[str]], remove: dict[str, set[str]] | set[str]) -> dict[str, list[str]]:
    """Remove genes from lists; ``remove`` is a set (all lists) or {cell_type: set}."""
    if isinstance(remove, set):
        return {ct: [g for g in genes if g not in remove] for ct, genes in lists.items()}
    return {ct: [g for g in genes if g not in remove.get(ct, set())] for ct, genes in lists.items()}


# ---------------------------------------------------------------------------
# D.3 summaries
# ---------------------------------------------------------------------------
def variant_summary(name: str, res: dict, labels: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    count_rows, cross_rows, change_rows = [], [], []
    ids = labels["n_detected_proteins"].to_numpy(float)
    plate = labels["plate"].to_numpy()
    for version, orig_col in VERSIONS.items():
        new = res[version].reindex(labels.index).to_numpy()
        orig = labels[orig_col].to_numpy()
        for label in cc.ordered_labels(np.concatenate([new, orig])):
            n = int((new == label).sum())
            count_rows.append({"variant": name, "label_version": version, "label": label, "n_cells": n,
                               "fraction": n / len(new), "n_cells_original": int((orig == label).sum())})
        table = pd.crosstab(pd.Series(orig, name="original_label"), pd.Series(new, name="variant_label"))
        for o, row in table.iterrows():
            for v, n in row.items():
                if n:
                    cross_rows.append({"variant": name, "label_version": version, "original_label": o,
                                       "variant_label": v, "n_cells": int(n)})
        is_p = new == "ProLIF"
        p1, p2 = is_p & (plate == "P1"), is_p & (plate == "P2")
        fisher = stats.fisher_exact([[int(p1.sum()), int((plate == "P1").sum() - p1.sum())],
                                     [int(p2.sum()), int((plate == "P2").sum() - p2.sum())]])
        ids_cmp = cc.compare_groups(ids[is_p], ids[~is_p])
        row = {"variant": name, "label_version": version, "n_cells": len(new),
               "n_changed": int((new != orig).sum()), "fraction_changed": float((new != orig).mean()),
               "n_prolif": int(is_p.sum()), "prolif_share": float(is_p.mean()),
               "n_ties_max_score": int(res["ties"].sum()), "n_all_zero_scores": int(res["all_zero"].sum()),
               "P1_prolif_fraction": p1.sum() / (plate == "P1").sum(), "P2_prolif_fraction": p2.sum() / (plate == "P2").sum(),
               "fisher_odds_ratio_P1_vs_P2": fisher[0], "fisher_p_plate": fisher[1],
               "median_ids_prolif": ids_cmp["median_group"], "median_ids_non_prolif": ids_cmp["median_reference"],
               "mannwhitney_p_ids": ids_cmp["p_value"], "rank_biserial_ids": ids_cmp["rank_biserial"],
               "lists_present": " | ".join(f"{ct}: {';'.join(g)}" for ct, g in res["present"].items())}
        if version == "cluster":
            row.update(unknown_clusters=";".join(res["unknown_clusters"]), n_unknown_cells=res["cluster_info"]["n_unknown_cells"],
                       unknown_score_threshold=res["cluster_info"]["score_threshold"])
        change_rows.append(row)
    return count_rows, cross_rows, change_rows


# ---------------------------------------------------------------------------
# D.4 downstream
# ---------------------------------------------------------------------------
def downstream(label_sets: dict[str, pd.DataFrame], task3: pd.DataFrame, aldo: pd.DataFrame,
               embeddings: dict[str, np.ndarray], args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """label_sets: variant -> frame with columns 'cell' and 'cluster' (index = cell_id)."""
    rows, purity_frames, aldo_frames = [], [], []
    score_cols = [c for c in aldo.columns if c.startswith(("V1_X__", "cc_regressed_3a__", "submission__"))]
    for variant, frame in label_sets.items():
        for version in VERSIONS:
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
        frame_labels = pd.DataFrame({f"{variant}__{v}": frame[v] for v in VERSIONS})
        pur = t2.knn_purity(embeddings, frame_labels, list(frame_labels.columns), args.knn_k, args.n_permutations)
        pur.insert(0, "variant", variant)
        pur["label_version"] = pur["label_column"].str.split("__").str[1]
        purity_frames.append(pur)
        for version in VERSIONS:
            per_cell = aldo[["exclude_tal_cnt_interstitial", *score_cols]].copy()
            per_cell["variant_label"] = frame[version].reindex(per_cell.index).to_numpy()
            contrast = t2.aldo_contrast_table(per_cell, "variant_label", score_cols)
            contrast.insert(0, "label_variant", variant)  # 'variant' already holds the ALDO score variant
            contrast.insert(1, "label_version", version)
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
                         & aldo_contrasts["source"].isin(["V1_X", "cc_regressed_3a"])
                         & aldo_contrasts["variant"].isin(["base", "no_Slc12a3"])]
    for _, r in sel.iterrows():
        rows.append({"variant": r["label_variant"], "label_version": r["label_version"], "metric_group": "aldo_prolif_contrast",
                     "label": r["comparison"], "quantity": f"{r['source']} {r['family']} {r['variant']} [{r['cohort']}]: median difference",
                     "value": r["median_difference"], "n": r["n_group"], "n_total": r["n_reference"],
                     "p_value": r["p_value"], "p_fdr_bh": r["p_fdr_bh"]})
    return pd.DataFrame(rows), purity, aldo_contrasts


def comparison_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per metric; original / noSlc12a3 / noSlc12a3_noUmod side by side."""
    key = ["label_version", "metric_group", "label", "quantity"]
    long = long.copy()

    def fmt(r):
        if r["metric_group"] == "ambiguity_flags":
            return f"{int(r['n'])}/{int(r['n_total'])} ({100 * r['value']:.1f}%)"
        if r["metric_group"] == "cell_cycle_score":
            return f"{r['value']:.3f} (n={int(r['n_total'])})"
        if r["metric_group"] == "knn_purity_prolif":
            return f"{r['value']:.3f} (null {r['null_mean']:.3f}; p={r['p_empirical_ge']:.3g}; n={int(r['n_total'])})"
        return f"{r['value']:+.3f} (q={r['p_fdr_bh']:.2g}; n={int(r['n'])} vs {int(r['n_total'])})"

    long["display"] = long.apply(fmt, axis=1)
    wide = long.pivot_table(index=key, columns="variant", values="display", aggfunc="first").reset_index()
    numeric = long.pivot_table(index=key, columns="variant", values="value", aggfunc="first").reset_index()
    numeric.columns = [c if c in key else f"value_{c}" for c in numeric.columns]
    wide = wide.merge(numeric, on=key)
    cols = key + [c for c in ("original", *MAIN_VARIANTS) if c in wide] + [f"value_{c}" for c in ("original", *MAIN_VARIANTS)
                                                                          if f"value_{c}" in wide]
    order = {"ambiguity_flags": 0, "cell_cycle_score": 1, "knn_purity_prolif": 2, "aldo_prolif_contrast": 3}
    return (wide[cols].assign(_o=wide["metric_group"].map(order))
            .sort_values(["label_version", "_o", "label", "quantity"]).drop(columns="_o"))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def alluvial(ax, orig: np.ndarray, new: np.ndarray, title: str) -> None:
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
    s = 1 / (1 + np.exp(-12 * (xs - 0.5)))
    for o in order:
        for n_lab in order:
            n = int(((orig == o) & (new == n_lab)).sum())
            if not n:
                continue
            y0l, y0r = lcur[o], rcur[n_lab]
            lcur[o] += n
            rcur[n_lab] += n
            bottom = y0l + (y0r - y0l) * s
            ax.fill_between(xs, bottom, bottom + n, color=t2.LABEL_PALETTE.get(o, "0.5"), alpha=0.35 if o != n_lab else 0.6,
                            linewidth=0)
    for side, pos, x0 in (("orig", left, 0.0), ("new", right, 0.88)):
        for lab, (a, b) in pos.items():
            ax.add_patch(plt.Rectangle((x0, a), 0.12, b - a, color=t2.LABEL_PALETTE.get(lab, "0.5"), linewidth=0))
            ax.text(x0 - 0.02 if side == "orig" else x0 + 0.14, (a + b) / 2, f"{lab} {int(b - a)}",
                    ha="right" if side == "orig" else "left", va="center", fontsize=5.6)
    changed = int((orig != new).sum())
    ax.set_title(f"{title}\nchanged: {changed}/{total} ({100 * changed / total:.1f}%)", fontsize=6.4)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(total * 1.08, -total * 0.02)
    ax.text(0.06, -total * 0.015, "original", ha="center", va="bottom", fontsize=5.6, color=t2.INK["secondary"])
    ax.text(0.94, -total * 0.015, "variant", ha="center", va="bottom", fontsize=5.6, color=t2.INK["secondary"])
    t2.clean_axes(ax)


def plot_flows(label_sets: dict[str, pd.DataFrame], labels: pd.DataFrame, run: cc.Run) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0))
    for r, variant in enumerate(MAIN_VARIANTS):
        for c, (version, col) in enumerate(VERSIONS.items()):
            alluvial(axes[r, c], labels[col].to_numpy(), label_sets[variant][version].reindex(labels.index).to_numpy(),
                     f"{VARIANT_TEXT[variant]}\n{t2.LABEL_DISPLAY[col]}")
    fig.suptitle("Label changes when Slc12a3 (and Umod) are removed from the V1 marker lists", x=0.01, ha="left",
                 fontsize=7.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    t2.footer(fig, SCRIPT, "ribbon colour = original label")
    run.figure(fig, "marker_sensitivity_label_flows", "Original versus variant labels (Sankey-style)", subdir=SUB)


def plot_summary(changes: pd.DataFrame, counts: pd.DataFrame, run: cc.Run) -> None:
    variants = ["original", *MAIN_VARIANTS]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    xpos, ticks = 0, []
    for version in VERSIONS:
        for variant in variants:
            sub = counts[(counts["variant"] == variant) & (counts["label_version"] == version)].set_index("label")
            bottom = 0
            for lab in cc.ordered_labels(sub.index):
                v = 100 * sub.loc[lab, "fraction"]
                ax.bar(xpos, v, bottom=bottom, color=t2.LABEL_PALETTE.get(lab, "0.5"), width=0.75, edgecolor="white",
                       linewidth=0.6)
                if v >= 7:
                    ax.text(xpos, bottom + v / 2, f"{int(sub.loc[lab, 'n_cells'])}", ha="center", va="center",
                            fontsize=5, color="white")
                bottom += v
            ticks.append((xpos, f"{version}\n{variant.replace('noSlc12a3_noUmod', '(b)').replace('noSlc12a3', '(a)')}"))
            xpos += 1
        xpos += 0.6
    ax.set_xticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=5.4)
    ax.set_ylabel("% of 750 cells")
    ax.set_title("Label composition (numbers = cells)", fontsize=6.8)
    ax.legend(handles=[Patch(color=t2.LABEL_PALETTE[l], label=l) for l in t2.LABEL_PALETTE], fontsize=5.2,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax, (version, col) in zip(axes[1:], VERSIONS.items()):
        sub = changes[(changes["label_version"] == version) & changes["variant"].isin(variants)].set_index("variant").reindex(variants)
        x = np.arange(len(variants))
        ax.bar(x - 0.18, 100 * sub["P1_prolif_fraction"], width=0.34, color=cc.PLATE_PALETTE["P1"], label="P1")
        ax.bar(x + 0.18, 100 * sub["P2_prolif_fraction"], width=0.34, color=cc.PLATE_PALETTE["P2"], label="P2")
        for xi, (a, b, p) in enumerate(zip(sub["P1_prolif_fraction"], sub["P2_prolif_fraction"], sub["fisher_p_plate"])):
            ax.text(xi, 100 * max(a, b) + 2, f"p={p:.1g}", ha="center", fontsize=5, color=t2.INK["secondary"])
        ax.set_xticks(x, ["original", "(a)", "(b)"])
        ax.set_ylim(0, 60)
        ax.set_ylabel("% of plate labelled ProLIF")
        ax.set_title(f"ProLIF by plate, {t2.LABEL_DISPLAY[col]}\nmedian IDs ProLIF / other: "
                     + ", ".join(f"{int(a)}/{int(b)}" for a, b in zip(sub["median_ids_prolif"], sub["median_ids_non_prolif"])),
                     fontsize=6.2)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    axes[1].legend(loc="upper right", fontsize=5.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    t2.footer(fig, SCRIPT, "(a) Slc12a3 removed; (b) Slc12a3 and Umod removed; Fisher p for plate x ProLIF")
    run.figure(fig, "marker_sensitivity_counts_and_plate", "Label counts and ProLIF fraction per plate by variant", subdir=SUB)


def plot_loo(per_cell: pd.DataFrame, impact: pd.DataFrame, labels: pd.DataFrame, run: cc.Run) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    rng = np.random.default_rng(0)  # jitter only
    for ax, (version, col), stab in zip(axes[:2], VERSIONS.items(), ("label_stability_loo", "label_stability_loo_cluster")):
        order = cc.ordered_labels(labels[col])
        for i, lab in enumerate(order):
            v = per_cell.loc[labels[col] == lab, stab].to_numpy(float)
            ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v + rng.uniform(-0.012, 0.012, len(v)), s=2.5,
                       color=t2.LABEL_PALETTE.get(lab, "0.5"), alpha=0.5, linewidths=0, rasterized=True)
            ax.scatter([i], [v.mean()], marker="D", s=14, color=t2.INK["primary"], zorder=3)
            ax.text(i, 1.06, f"{(v == 1).sum()}/{len(v)}\nall kept", ha="center", fontsize=4.8, color=t2.INK["secondary"])
        ax.set_xticks(range(len(order)), order)
        ax.set_ylim(-0.05, 1.16)
        ax.set_ylabel("Fraction of 11 leave-one-out runs\nkeeping the original label")
        ax.set_title(f"Label stability, {t2.LABEL_DISPLAY[col]} (diamond = mean)", fontsize=6.4)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    ax = axes[2]
    markers = list(dict.fromkeys(impact["dropped_marker"]))
    y = np.arange(len(markers))
    for k, (version, color) in enumerate((("cell", t2.SERIES[0]), ("cluster", t2.SERIES[1]))):
        sub = impact[impact["label_version"] == version].set_index("dropped_marker").reindex(markers)
        ax.barh(y + (k - 0.5) * 0.38, sub["n_changed"], height=0.36, color=color, label=f"{version} label")
        for yy, n in zip(y + (k - 0.5) * 0.38, sub["n_changed"]):
            ax.text(n + 3, yy, str(int(n)), va="center", fontsize=5, color=t2.INK["secondary"])
    ax.set_yticks(y, markers)
    ax.invert_yaxis()
    ax.set_xlabel("Cells whose label changes when the marker is removed")
    ax.set_title("Leave-one-marker-out impact", fontsize=6.4)
    ax.set_xlim(0, impact["n_changed"].max() * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.5)
    ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    t2.footer(fig, SCRIPT)
    run.figure(fig, "marker_sensitivity_loo", "Leave-one-marker-out label stability and marker impact", subdir=SUB)


def plot_downstream(long: pd.DataFrame, run: cc.Run) -> None:
    variants = ["original", *MAIN_VARIANTS]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.8))
    for r, version in enumerate(VERSIONS):
        sub = long[long["label_version"] == version]
        for c, group in enumerate(("ambiguity_flags", "cell_cycle_score")):
            ax = axes[r, c]
            part = sub[sub["metric_group"] == group]
            labs = [l for l in cc.ordered_labels(part["label"]) if l in t2.CORE_LABELS]
            for k, variant in enumerate(variants):
                vals = part[part["variant"] == variant].set_index("label").reindex(labs)
                x = np.arange(len(labs)) + (k - 1) * 0.26
                yv = vals["value"].to_numpy(float) * (100 if group == "ambiguity_flags" else 1)
                ax.bar(x, yv, width=0.24, color=t2.SERIES[k], label=variant)
            ax.set_xticks(range(len(labs)), labs)
            ax.set_ylabel("% flagged" if group == "ambiguity_flags" else "median cell_cycle_score")
            ax.set_title(f"{'Ambiguity flags' if group == 'ambiguity_flags' else 'Cell-cycle score'} by {version} label", fontsize=6.4)
            ax.axhline(0, color=t2.INK["muted"], linewidth=0.5)
            ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
        ax = axes[r, 2]
        part = sub[sub["metric_group"] == "knn_purity_prolif"]
        objects = list(dict.fromkeys(part["quantity"]))
        for k, variant in enumerate(variants):
            vals = part[part["variant"] == variant].set_index("quantity").reindex(objects)
            x = np.arange(len(objects)) + (k - 1) * 0.26
            ax.vlines(x, vals["null_mean"], vals["null_q975"], color=t2.INK["background_cells"], linewidth=3)
            ax.scatter(x, vals["value"], color=t2.SERIES[k], s=14, zorder=3, label=variant)
        ax.set_xticks(range(len(objects)), [o.split(", ")[-1] for o in objects])
        ax.set_ylim(0, 1)
        ax.set_ylabel("ProLIF same-label fraction (k = 10)")
        ax.set_title(f"ProLIF kNN purity, {version} label (grey: null mean to 97.5%)", fontsize=6.4)
        ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    handles, texts = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [f"variant: {t}" for t in texts], loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=3,
               fontsize=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    t2.footer(fig, SCRIPT, "(a) noSlc12a3; (b) noSlc12a3_noUmod; ALDO contrasts are in the comparison table")
    run.figure(fig, "marker_sensitivity_downstream", "Downstream summaries under the original and variant labels", subdir=SUB)


def main() -> None:
    args = parse_args()
    run = cc.Run(t2.TASK, "marker_sensitivity", args)
    t2.record_reference_code(run, ["scp_dct_upload.py"])
    adata, labels = t2.load_dct(run)
    lists = cc.read_marker_workbook(run.use_input(cc.INPUTS["dct"]["markers"], "V1 marker workbook (Supp Table 1)"))
    stored = ad.read_h5ad(run.use_input(cc.INPUTS["dct"]["v1_h5ad"], "V1 processed AnnData (obs only: stored scores)"),
                          backed="r").obs.copy()
    stored.index = stored.index.astype(str)
    stored = stored.reindex(labels.index)
    clusters = labels["v1_leiden_res3_cluster"]

    # D.1 reproduction (stop if it fails).
    orig = label_variant(adata, lists, clusters)
    checks = []
    for ct in lists:
        col = f"{ct}_score"
        diff = float(np.abs(orig["scores"][ct].to_numpy() - stored[col].to_numpy(float)).max())
        checks.append({"check": f"{col} reproduced (max abs diff vs stored V1 obs)", "value": diff, "pass": diff < 1e-6})
    diff = float(np.abs(orig["max_score"].to_numpy() - stored["max_score"].to_numpy(float)).max())
    checks.append({"check": "max_score reproduced (max abs diff vs stored V1 obs)", "value": diff, "pass": diff < 1e-6})
    n_cell = int((orig["cell"].to_numpy() == labels["v1_cell_label"].to_numpy()).sum())
    checks.append({"check": "per-cell argmax == v1_cell_label (cells)", "value": n_cell, "pass": n_cell == len(labels)})
    n_clu = int((orig["cluster"].reindex(labels.index).to_numpy() == labels["v1_cluster_label"].to_numpy()).sum())
    checks.append({"check": "cluster labels on stored leiden_res_3.00 == v1_cluster_label (cells)", "value": n_clu,
                   "pass": n_clu == len(labels)})
    checks.append({"check": "'Unknown' cluster(s) re-derived", "value": ";".join(orig["unknown_clusters"]),
                   "pass": orig["cluster_info"]["n_unknown_cells"] == int((labels["v1_cluster_label"] == "Unknown").sum())})
    checks.append({"check": "ties in max score (original lists)", "value": int(orig["ties"].sum()), "pass": True})
    sanity = pd.DataFrame(checks)
    run.table(sanity, "marker_sensitivity_reproduction", subdir=SUB)
    if not sanity["pass"].all():
        run.note("Reproduction FAILED; section D stopped. See reviewer_core_marker_sensitivity_reproduction.csv.")
        run.finish(status="stopped: reproduction failed")
        sys.exit(1)
    run.note(f"Reproduction passed: scores max abs diff 0-level; {n_cell}/750 per-cell and {n_clu}/750 cluster labels; "
             f"Unknown cluster(s) {orig['unknown_clusters']}.")

    # D.2 variants.
    detected = list(dict.fromkeys(g for genes in orig["present"].values() for g in genes))
    variants = {"original": orig,
                "noSlc12a3": label_variant(adata, drop(lists, {"Slc12a3"}), clusters),
                "noSlc12a3_noUmod": label_variant(adata, drop(drop(lists, {"Slc12a3"}), {"DCT1": {"Umod"}}), clusters)}
    for marker in detected:
        variants[f"loo_{marker}"] = label_variant(adata, drop(lists, {marker}), clusters)
    definitions = pd.DataFrame([{"variant": name, "description": VARIANT_TEXT.get(name, f"leave-one-out: {name[4:]} removed "
                                                                              "from every list"),
                                 **{f"{ct}_markers_scored": ";".join(res["present"].get(ct, [])) for ct in lists}}
                                for name, res in variants.items()])
    run.table(definitions, "marker_sensitivity_variant_definitions", subdir=SUB)

    per_cell = labels[["plate", "n_detected_proteins", *t2.LABEL_COLUMNS]].copy()
    for name in MAIN_VARIANTS:
        per_cell[f"v1_cell_label_{name}"] = variants[name]["cell"].reindex(labels.index).to_numpy()
        per_cell[f"v1_cluster_label_{name}"] = variants[name]["cluster"].reindex(labels.index).to_numpy()
    loo_names = [n for n in variants if n.startswith("loo_")]
    keep_cell = np.column_stack([variants[n]["cell"].reindex(labels.index).to_numpy() == labels["v1_cell_label"].to_numpy()
                                 for n in loo_names])
    keep_clu = np.column_stack([variants[n]["cluster"].reindex(labels.index).to_numpy() == labels["v1_cluster_label"].to_numpy()
                                for n in loo_names])
    per_cell["label_stability_loo"] = keep_cell.mean(axis=1)
    per_cell["label_stability_loo_cluster"] = keep_clu.mean(axis=1)
    for name in ("original", *MAIN_VARIANTS):
        for ct in lists:
            per_cell[f"score_{ct}_{name}"] = variants[name]["scores"][ct].reindex(labels.index).to_numpy()
    run.table(per_cell.reset_index(), "marker_sensitivity_labels_per_cell", subdir=SUB)
    loo_wide = pd.DataFrame(index=labels.index)
    for n in loo_names:
        loo_wide[f"{n}__cell"] = variants[n]["cell"].reindex(labels.index).to_numpy()
        loo_wide[f"{n}__cluster"] = variants[n]["cluster"].reindex(labels.index).to_numpy()
    run.table(loo_wide.reset_index(), "marker_sensitivity_loo_labels_per_cell", subdir=SUB)

    # D.3 summaries for every variant.
    counts, cross, changes = [], [], []
    for name, res in variants.items():
        a, b, c = variant_summary(name, res, labels)
        counts += a
        cross += b
        changes += c
    counts, cross, changes = pd.DataFrame(counts), pd.DataFrame(cross), pd.DataFrame(changes)
    run.table(counts, "marker_sensitivity_label_counts", subdir=SUB)
    run.table(cross, "marker_sensitivity_crosstabs", subdir=SUB)
    run.table(changes, "marker_sensitivity_change_summary", subdir=SUB)

    # D.4 downstream (original, a, b).
    task3 = t2.read_table(run, "task3_marker_scores_per_cell").set_index("cell_id").reindex(labels.index)
    aldo = t2.read_table(run, "aldo_scores_per_cell").set_index("cell_id").reindex(labels.index)
    cc_reg = t2.read_h5ad(run, "DCT_cc_regressed")
    embeddings = {"uncorrected": adata.obsm["X_pca"][:, :t2.N_PCS], "cc_regressed": cc_reg.obsm["X_pca"][:, :t2.N_PCS]}
    label_sets = {name: pd.DataFrame({"cell": variants[name]["cell"].reindex(labels.index),
                                      "cluster": variants[name]["cluster"].reindex(labels.index)})
                  for name in ("original", *MAIN_VARIANTS)}
    long, purity, aldo_contrasts = downstream(label_sets, task3, aldo, embeddings, args)
    run.table(long, "marker_sensitivity_downstream_long", subdir=SUB)
    run.table(comparison_wide(long), "marker_sensitivity_downstream_comparison", subdir=SUB)
    run.table(purity, "marker_sensitivity_knn_purity", subdir=SUB)
    run.table(aldo_contrasts, "marker_sensitivity_aldo_contrasts", subdir=SUB)

    # Consistency: 'original' rows must equal the A.2 purity and A.3 ALDO tables.
    a2 = t2.read_table(run, "task4_knn_label_purity")
    a3 = t2.read_table(run, "aldo_contrasts")
    p_orig = purity[purity["variant"] == "original"].assign(
        label_column=lambda d: d["label_version"].map(VERSIONS))
    m = p_orig.merge(a2, on=["object", "label_column", "label"], suffixes=("", "_a2"))
    d_pur = float(np.abs(m["observed_same_label_fraction"] - m["observed_same_label_fraction_a2"]).max())
    d_null = float(np.abs(m["null_mean"] - m["null_mean_a2"]).max())
    c_orig = aldo_contrasts[aldo_contrasts["label_variant"] == "original"].assign(
        label_column=lambda d: d["label_version"].map(VERSIONS))
    m2 = c_orig.merge(a3, on=["label_column", "cohort", "comparison", "score"], suffixes=("", "_a3"))
    d_q = float(np.abs(m2["p_fdr_bh"] - m2["p_fdr_bh_a3"]).max())
    d_md = float(np.abs(m2["median_difference"] - m2["median_difference_a3"]).max())
    run.note(f"Original rows versus A.2/A.3: kNN purity max abs diff {d_pur:.3g} (null mean {d_null:.3g}; {len(m)} rows); "
             f"ALDO median difference {d_md:.3g}, BH q {d_q:.3g} ({len(m2)} rows).")

    # D.5 leave-one-out stability.
    stab_rows = []
    for version, col in VERSIONS.items():
        stab = per_cell["label_stability_loo" if version == "cell" else "label_stability_loo_cluster"]
        for label in cc.ordered_labels(labels[col]):
            v = stab[labels[col] == label]
            stab_rows.append({"label_version": version, "original_label": label, "n_cells": len(v),
                              "n_loo_runs": len(loo_names), "mean_stability": v.mean(), "median_stability": v.median(),
                              "n_stability_1": int((v == 1).sum()), "n_stability_below_0.5": int((v < 0.5).sum()),
                              "n_stability_0": int((v == 0).sum())})
    run.table(pd.DataFrame(stab_rows), "marker_sensitivity_loo_stability_by_label", subdir=SUB)
    impact = changes[changes["variant"].str.startswith("loo_")].copy()
    impact.insert(0, "dropped_marker", impact["variant"].str.removeprefix("loo_"))
    by_label = (cross[cross["variant"].str.startswith("loo_") & (cross["original_label"] != cross["variant_label"])]
                .groupby(["variant", "label_version", "original_label"])["n_cells"].sum().unstack(fill_value=0)
                .add_prefix("n_changed_from_").reset_index())
    impact = impact.merge(by_label, on=["variant", "label_version"], how="left").fillna(
        {c: 0 for c in by_label.columns if c.startswith("n_changed_from_")})
    impact = impact.sort_values(["label_version", "n_changed"], ascending=[True, False])
    run.table(impact[["dropped_marker", "label_version", "n_changed", "fraction_changed", "n_prolif", "prolif_share",
                      *[c for c in impact.columns if c.startswith("n_changed_from_")], "n_ties_max_score",
                      "lists_present"]], "marker_sensitivity_loo_marker_impact", subdir=SUB)

    plot_flows(label_sets, labels, run)
    plot_summary(changes, counts, run)
    plot_loo(per_cell, impact.sort_values("dropped_marker", key=lambda s: s.map({m: i for i, m in enumerate(detected)})),
             labels, run)
    plot_downstream(long, run)
    run.finish({"detected_markers": detected, "loo_runs": loo_names,
                "consistency_vs_A2_A3": {"knn_purity_max_abs_diff": d_pur, "aldo_median_difference_max_abs_diff": d_md,
                                         "aldo_q_max_abs_diff": d_q}})


if __name__ == "__main__":
    main()
