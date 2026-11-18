#!/usr/bin/env python
"""labels_core step 2a: DE-rule labels at lower Leiden resolution on the control DCT graph (resolution scan).

Analysis plan, step 2
(cluster_de method, agreed 13 Oct 2026; minimum cluster size 5 and seed 42, authors 28 Oct 2026).

On the step-1 graph (k = 10, 20 PCs; submitted pipeline): Leiden (igraph) at 0.3, 0.5, 0.8, 1.0 and 3.0 (reference),
seeds 42 and 0-9. Every clustering is annotated with
  - the cluster_de rule (cluster_de_lib: Wilcoxon cluster-vs-rest on ln(1 + intensity), tested = detected in >= 20% of the
    cluster, top 25 significant up proteins, hypergeometric marker enrichment plus mean marker z; Unknown without a
    marker in the top 25), with Slc12a3 and Umod treated as shared (calls resting only on them: low_shared_only);
  - the kidney_annotation rank-free rescue (kidney_annotation_rankfree_sensitivity.rank_free: a cluster left Unknown or shared-only takes
    the type whose type-specific marker is significantly up at any rank);
  clusters with < 5 cells are reported, not annotated ('below_min_size').
Reported per resolution x rule: clusters, labels and cells per label, confidence, types resolved, seed stability (ARI
and per-cell label stability against seed 42), agreement with ctrl_cell_label / ctrl_cluster_label, protein-ID eta^2.
Information only (not used for any label): per-cluster detection of the intake_core contamination panel, plate share, the
step-1 label composition. No label column is named here: the authors choose resolution and rule (step 2b).

Inputs (read-only): results/h5ad/labels_core_DCT_ctrl_step1_submitted.h5ad; labels_core_submitted_per_cell.csv;
  data/DCT/DCT_markers_V1.xlsx.
Outputs: results/tables/labels_core/labels_core_scan_*.csv;
  results/figures/labels_core/labels_core_scan_summary.pdf/.png, labels_core_scan_umaps.pdf/.png

Command (project root):  python scripts/labels_core_resolution_scan.py
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

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import labels_core_lib as lib
from cluster_de_deg_annotation import evidence_checks
from kidney_annotation_rankfree_sensitivity import rank_free
from kidney_annotation_scan_lib import eta_squared

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "labels_core_resolution_scan.py"
STEP1_LABELS = ("ctrl_cell_label", "ctrl_cluster_label")
MM = 1 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="labels_core step 2a: DE-rule resolution scan on the control DCT graph.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "scan_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see labels_core_scan_checks.csv.")


def rule_tables(asg: pd.DataFrame, rf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-cluster label and confidence for each rule."""
    return {"top25": asg[["cluster", "deg_label", "confidence"]],
            "rankfree": rf.rename(columns={"rankfree_label": "deg_label", "rankfree_confidence": "confidence"})[
                ["cluster", "deg_label", "confidence"]]}


def seed_row(res: float, seed: int, rule: str, cl: np.ndarray, lab: np.ndarray, tab: pd.DataFrame, ids: np.ndarray,
             ref_cl: np.ndarray, ref_lab: np.ndarray) -> dict:
    sizes = pd.Series(cl).value_counts()
    types = lib.types_resolved(tab)
    return {"resolution": res, "rule": rule, "seed": seed, "n_clusters": int(len(sizes)),
            "cluster_size_min": int(sizes.min()), "cluster_size_median": float(sizes.median()),
            "cluster_size_max": int(sizes.max()), "n_clusters_below_min": int((sizes < lib.MIN_CLUSTER_SIZE).sum()),
            "n_cells_below_min": int(sizes[sizes < lib.MIN_CLUSTER_SIZE].sum()),
            "n_clusters_unknown": int((tab["deg_label"] == lib.UNKNOWN).sum()),
            "fraction_cells_unknown": float((lab == lib.UNKNOWN).mean()),
            "n_types_resolved": len(types), "types_resolved": ";".join(f"{t}({c})" for t, c in types.items()),
            "label_counts": lib.counts_text(lab), "eta2_protein_ids": eta_squared(ids, cl),
            "ari_clusters_vs_seed42": float(adjusted_rand_score(ref_cl, cl)),
            "ari_labels_vs_seed42": float(adjusted_rand_score(ref_lab, lab)),
            "fraction_same_label_as_seed42": float((ref_lab == lab).mean())}


def cluster_rows(res: float, cl: np.ndarray, asg: pd.DataFrame, rf: pd.DataFrame, de: pd.DataFrame,
                 step1: pd.DataFrame, intensity: np.ndarray, var_index: dict[str, int]) -> pd.DataFrame:
    """Seed-42 cluster table: both rules, DE summary, depth, plate and information-only marker detection."""
    ids = step1["n_detected"].to_numpy(float)
    rows = []
    for c in t5.sort_clusters(cl):
        mask = cl == c
        sub = de[de["cluster"] == c]
        top = sub[sub["in_top"]].sort_values("rank_up")["protein"].tolist()
        depth = cc.compare_groups(ids[mask], ids[~mask])
        row = {"resolution": res, "cluster": c, "n_cells": int(mask.sum()),
               "median_protein_ids": depth["median_group"], "median_protein_ids_rest": depth["median_reference"],
               "protein_ids_rank_biserial_vs_rest": depth["rank_biserial"], "protein_ids_p_vs_rest": depth["p_value"],
               "plate_P1_fraction": float((step1.loc[mask, "plate"] == "P1").mean()),
               "n_tested": int(len(sub)), "n_sig_up": int((sub["up"] & (sub["q_bh"] < t5.Q_DE)).sum()),
               "top10_de": ";".join(top[:10])}
        for col in STEP1_LABELS:
            counts = step1.loc[mask, col].astype(str).value_counts()
            row[f"majority_{col}"] = counts.index[0]
            row[f"majority_{col}_fraction"] = float(counts.iloc[0] / mask.sum())
            row[f"{col}_counts"] = lib.counts_text(step1.loc[mask, col])
        for group, genes in lib.CONTAMINATION_PANEL.items():
            for g in genes:
                if g in var_index:
                    row[f"detected_{group}_{g}".replace("/", "_")] = float((intensity[mask, var_index[g]] > 0).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    keep = ["cluster", "deg_label", "confidence", "overlap", "overlap_specific", "p_hypergeom", "q_hypergeom",
            "mean_marker_z", "mean_z_best_type", "runner_up", "tied_types", "dct1_umod_slc12a3_only"]
    top25 = asg[[k for k in keep if k in asg.columns]].rename(
        columns={"deg_label": "top25_label", "confidence": "top25_confidence"})
    rescue = rf[["cluster", "rankfree_label", "rankfree_confidence", "rescued", "rescue_candidates", "rescue_markers",
                 "rescue_best_rank"]]
    return frame.merge(top25, on="cluster", how="left").merge(rescue, on="cluster", how="left")


def wilcoxon_check(checks: t5.Checks, cache: t5.RankCache, cl: np.ndarray, de: pd.DataFrame, name: str,
                   var_names) -> None:
    c = t5.sort_clusters(cl)[0]
    sub = de[de["cluster"] == c].set_index("protein")
    mask = cl == c
    cols = [i for i, p in enumerate(var_names) if p in sub.index][:400]
    p = np.array([stats.mannwhitneyu(cache.ln1p[mask, j], cache.ln1p[~mask, j], alternative="two-sided",
                                     use_continuity=False, method="asymptotic").pvalue for j in cols])
    ours = sub.loc[[var_names[j] for j in cols], "p_value"].to_numpy(float)
    d = float(np.nanmax(np.abs(np.log10(np.maximum(p, 1e-300)) - np.log10(np.maximum(ours, 1e-300)))))
    checks.add(f"Wilcoxon vs scipy mannwhitneyu ({name}, cluster {c}, first {len(cols)} tested proteins): "
               "max |dlog10 p|", f"{d:.3g}", d < 1e-6, "< 1e-6")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def summary_figure(run: cc.Run, summary: pd.DataFrame, seeds: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    lib.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 120 * MM))
    res_pos = {r: i for i, r in enumerate(lib.SCAN_RESOLUTIONS)}
    rule_style = {"top25": (lib.SERIES[0], -0.17), "rankfree": (lib.SERIES[1], 0.17)}

    ax = axes[0, 0]
    top = seeds[seeds["rule"] == "top25"]
    for r, grp in top.groupby("resolution"):
        x = res_pos[r]
        ax.vlines(x, grp["n_clusters"].min(), grp["n_clusters"].max(), color=lib.INK["muted"], lw=1)
        ax.plot(x, int(grp.loc[grp["seed"] == lib.SEED, "n_clusters"].iloc[0]), "o", color=lib.INK["primary"], ms=4)
    ax.set_xticks(range(len(res_pos)), [f"{r:g}" for r in res_pos])
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Clusters")
    ax.set_title("a  Clusters (dot: seed 42; line: 11-seed range)")

    ax = axes[0, 1]
    width = 0.32
    for rule, (color, shift) in rule_style.items():
        bottom = np.zeros(len(res_pos))
        for label in lib.LABEL_ORDER:
            col = f"n_{label}"
            vals = np.array([summary.loc[(summary["rule"] == rule) & (summary["resolution"] == r), col].iloc[0]
                             for r in res_pos], float) / 727
            ax.bar(np.arange(len(res_pos)) + shift, vals, width, bottom=bottom, color=lib.LABEL_COLORS[label],
                   edgecolor="white", lw=0.3, label=label if rule == "top25" else None)
            bottom += vals
    ax.set_xticks(range(len(res_pos)), [f"{r:g}\nT  R" for r in res_pos])
    ax.set_xlabel("Leiden resolution (T = top-25 rule, R = rank-free rescue)")
    ax.set_ylabel("Fraction of cells (seed 42)")
    ax.set_title("b  Cells per DE-rule label")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6)

    ax = axes[1, 0]
    for rule, (color, shift) in rule_style.items():
        sub = summary[summary["rule"] == rule]
        x = np.array([res_pos[r] for r in sub["resolution"]]) + shift / 2
        ax.vlines(x, sub["ari_labels_vs_seed42_min"], sub["ari_labels_vs_seed42_max"], color=color, lw=1)
        ax.plot(x, sub["ari_labels_vs_seed42_median"], "o", color=color, ms=4, label=lib.RULE_DISPLAY[rule])
    ax.set_xticks(range(len(res_pos)), [f"{r:g}" for r in res_pos])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("ARI of labels vs seed 42\n(seeds 0-9; dot: median, line: range)")
    ax.set_title("c  Label stability over Leiden seeds")
    ax.legend(loc="lower right", fontsize=6)

    ax = axes[1, 1]
    conf_colors = {"high": lib.SERIES[2], "medium": lib.SERIES[3], lib.RESCUED: lib.SERIES[4],
                   "low_shared_only": lib.SERIES[7], "unknown_no_marker_in_top_de": lib.LABEL_COLORS["Unknown"],
                   lib.BELOW_MIN: lib.LABEL_COLORS[lib.BELOW_MIN]}
    for rule, (color, shift) in rule_style.items():
        bottom = np.zeros(len(res_pos))
        for conf, ccol in conf_colors.items():
            vals = np.array([summary.loc[(summary["rule"] == rule) & (summary["resolution"] == r),
                                         f"n_cells_conf_{conf}"].iloc[0] for r in res_pos], float) / 727
            ax.bar(np.arange(len(res_pos)) + shift, vals, width, bottom=bottom, color=ccol, edgecolor="white",
                   lw=0.3, label=conf if rule == "top25" else None)
            bottom += vals
    ax.set_xticks(range(len(res_pos)), [f"{r:g}\nT  R" for r in res_pos])
    ax.set_xlabel("Leiden resolution (T = top-25 rule, R = rank-free rescue)")
    ax.set_ylabel("Fraction of cells (seed 42)")
    ax.set_title("d  Confidence of the cluster call")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 1), w_pad=3.0)
    lib.footer(fig, SCRIPT, "control DCT cells (n = 727); graph of the submitted pipeline (k = 10, 20 PCs)")
    run.figure(fig, "scan_summary", "labels_core resolution scan: clusters, DE-rule labels, seed stability, confidence")


def wrapped_counts(lab: np.ndarray) -> str:
    """Label counts on two lines for a narrow panel title."""
    parts = lib.counts_text(lab).replace("Unknown", "Unk").split("; ")
    half = (len(parts) + 1) // 2
    return "; ".join(parts[:half]) + "\n" + "; ".join(parts[half:])


def umap_figure(run: cc.Run, umap: np.ndarray, per_cell: pd.DataFrame, step1: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    lib.apply_style()
    n = len(lib.SCAN_RESOLUTIONS)
    fig, axes = plt.subplots(3, n + 1, figsize=(180 * MM, 125 * MM))
    cmap = plt.get_cmap("tab20")
    for j, res in enumerate(lib.SCAN_RESOLUTIONS):
        name = lib.clustering_name(res)
        cl = per_cell[f"{name}__cluster"].to_numpy()
        ax = axes[0, j]
        order = t5.sort_clusters(cl)
        idx = {c: i for i, c in enumerate(order)}
        ax.scatter(umap[:, 0], umap[:, 1], s=1.2, c=[cmap(idx[c] % 20) for c in cl], lw=0, rasterized=True)
        for c in order:
            m = cl == c
            ax.text(np.median(umap[m, 0]), np.median(umap[m, 1]), c, fontsize=5, ha="center", va="center",
                    color=lib.INK["primary"])
        ax.set_title(f"Leiden {res:g}: {len(order)} clusters", fontsize=6.5)
        for i, rule in enumerate(lib.RULES, start=1):
            lab = per_cell[f"{name}__{rule}_label"].astype(str).to_numpy()
            ax = axes[i, j]
            for label in lib.LABEL_ORDER:
                m = lab == label
                if m.any():
                    ax.scatter(umap[m, 0], umap[m, 1], s=1.2, color=lib.LABEL_COLORS[label], lw=0, rasterized=True)
            ax.set_title(f"{'T' if rule == 'top25' else 'R'} {res:g}\n{wrapped_counts(lab)}", fontsize=5)
    for i, col in enumerate(STEP1_LABELS):
        ax = axes[i, n]
        lab = step1[col].astype(str).to_numpy()
        for label in lib.LABEL_ORDER:
            m = lab == label
            if m.any():
                ax.scatter(umap[m, 0], umap[m, 1], s=1.2, color=lib.LABEL_COLORS[label], lw=0, rasterized=True)
        ax.set_title(f"{col} (step 1)\n{wrapped_counts(lab)}", fontsize=5)
    ax = axes[2, n]
    ids = step1["n_detected"].to_numpy(float)
    order = np.argsort(ids)
    pts = ax.scatter(umap[order, 0], umap[order, 1], s=1.2, c=ids[order], cmap="viridis", lw=0, rasterized=True)
    cbar = fig.colorbar(pts, ax=ax, fraction=0.05, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    ax.set_title("Protein IDs per cell", fontsize=5.5)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=lib.LABEL_COLORS[label], label=label)
               for label in lib.LABEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=6, bbox_to_anchor=(0.5, 0.015),
               frameon=False)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("Leiden clusters (seed 42)")
    axes[1, 0].set_ylabel("Top-25 rule (cluster_de)")
    axes[2, 0].set_ylabel("Rank-free rescue (kidney_annotation)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    lib.footer(fig, SCRIPT, "UMAP of the submitted pipeline (step 1); labels at seed 42")
    run.figure(fig, "scan_umaps", "labels_core resolution scan on the step-1 UMAP: clusters and DE-rule labels per resolution")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "resolution_scan", args)
    t_start = time.time()
    checks = t5.Checks("dct_ctrl")
    adata = lib.load_step1(run)
    step1 = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    same = list(step1.index) == list(cell_ids)
    checks.add("step-1 table and object hold the same cells in the same order", f"{same}", same)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    checks.add("intensity layer: no negative or missing values", f"min {intensity.min():.3g}",
               bool(np.isfinite(intensity).all() and intensity.min() >= 0))
    ln1p = np.log1p(intensity)
    detected = intensity > 0
    ids = step1["n_detected"].to_numpy(float)
    x_z = pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    markers = lib.dct_markers(run, var_names)
    specific = lib.specific_present(markers)
    checks.add("type-specific markers present (Slc12a3 shared by the workbook; Umod treated as shared)",
               "; ".join(f"{t}: {','.join(g)}" for t, g in specific.items()), None)
    checks.add("shared markers present", ";".join(sorted({g for t in markers.present for g in markers.present[t]
                                                          if markers.is_shared(g)})), None)
    cache = t5.rank_cache(ln1p, detected)
    g = lib.graph_adata(adata)
    var_index = {v: i for i, v in enumerate(var_names)}
    if not checks.ok:
        stop(run, checks)

    per_cell = step1[["plate", "n_detected", *STEP1_LABELS, "ctrl_leiden3_cluster"]].copy()
    seed_cols: dict[str, np.ndarray] = {}
    seed_rows, clus_all, ev_all, de_top_all, type_rows = [], [], [], [], []
    for res in lib.SCAN_RESOLUTIONS:
        t0 = time.time()
        ref: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        per_seed_types: dict[str, list[dict]] = {r: [] for r in lib.RULES}
        for seed in lib.SEEDS:
            name = lib.clustering_name(res, seed)
            cl = lib.leiden_on(g, res, seed)
            de, ev, asg = lib.annotate_clustering(cache, cl, var_names, name, markers, x_z, det_frame)
            rf = rank_free(asg, de, specific)
            tabs = rule_tables(asg, rf)
            if seed == lib.SEED:
                if res == lib.SCAN_RESOLUTIONS[0]:
                    wilcoxon_check(checks, cache, cl, de, name, list(var_names))
                for rule, tab in tabs.items():
                    lab = lib.per_cell(cl, tab, "deg_label").astype(str)
                    ref[rule] = (cl, lab)
                    per_cell[f"{name}__{rule}_label"] = lab
                    per_cell[f"{name}__{rule}_confidence"] = lib.per_cell(cl, tab, "confidence").astype(str)
                per_cell[f"{name}__cluster"] = cl
                clus_all.append(cluster_rows(res, cl, asg, rf, de, step1, intensity, var_index))
                ev.insert(1, "resolution", res)
                ev_all.append(ev)
                de_top_all.append(de[de["in_top"]].sort_values(["cluster", "rank_up"]))
                kept = rf[rf["top25_confidence"].isin(["high", "medium"])]
                bad = kept[kept["rankfree_label"] != kept["top25_label"]]
                checks.add(f"r{res:g}: rank-free keeps every high/medium top-25 label", f"{len(bad)} changed",
                           bad.empty)
            else:
                seed_cols[f"{name}__cluster"] = cl
            for rule, tab in tabs.items():
                lab = lib.per_cell(cl, tab, "deg_label").astype(str)
                if seed != lib.SEED:
                    seed_cols[f"{name}__{rule}_label"] = lab
                per_seed_types[rule].append(lib.types_resolved(tab))
                seed_rows.append(seed_row(res, seed, rule, cl, lab, tab, ids, *ref[rule]))
        for rule in lib.RULES:
            cl42 = per_cell[f"{lib.clustering_name(res)}__cluster"].to_numpy()
            lab42 = per_cell[f"{lib.clustering_name(res)}__{rule}_label"].to_numpy()
            conf42 = per_cell[f"{lib.clustering_name(res)}__{rule}_confidence"].to_numpy()
            sizes = pd.Series(cl42).value_counts()
            for t in lib.CORE_TYPES:
                m = lab42 == t
                cls = sorted(set(cl42[m]), key=lambda c: t5.sort_clusters(cl42).index(c))
                best = lib.types_resolved(pd.DataFrame({"deg_label": lab42[m], "confidence": conf42[m]})).get(t, "")
                type_rows.append({"resolution": res, "rule": rule, "cell_type": t, "n_clusters_seed42": len(cls),
                                  "n_cells_seed42": int(m.sum()), "best_confidence_seed42": best,
                                  "clusters_seed42": ";".join(f"{c}({sizes[c]})" for c in cls),
                                  "n_seeds_resolved": sum(t in d for d in per_seed_types[rule]),
                                  "n_seeds_resolved_ge_medium": sum(d.get(t) in ("high", "medium")
                                                                    for d in per_seed_types[rule]),
                                  "n_seeds": len(per_seed_types[rule])})
        r42 = [r for r in seed_rows if r["resolution"] == res and r["seed"] == lib.SEED]
        print(f"[ctrl] r{res:g}: {r42[0]['n_clusters']} clusters; top25 {r42[0]['label_counts']}; rankfree "
              f"{r42[1]['label_counts']}; {time.time() - t0:.1f} s", flush=True)

    # Reproduction: Leiden 3.0 seed 42 on the stored graph = step-1 clusters
    a = float(adjusted_rand_score(step1["ctrl_leiden3_cluster"].astype(str), per_cell["r3.0__cluster"]))
    checks.add("Leiden 3.0 (seed 42) equals step-1 ctrl_leiden3_cluster (ARI)", f"{a:.6f}", a == 1.0, "= 1")
    clus_all = pd.concat(clus_all, ignore_index=True)
    clus_all = cc.add_bh(clus_all, ["resolution"], p_col="protein_ids_p_vs_rest", out_col="protein_ids_q_vs_rest")
    ev_all = pd.concat(ev_all, ignore_index=True)
    de_top_all = pd.concat(de_top_all, ignore_index=True)
    evidence_checks(checks, ev_all, de_top_all)
    n_missing = int(per_cell.filter(like="_label").isna().sum().sum()) + sum(int(pd.isna(v).sum())
                                                                           for v in seed_cols.values())
    checks.add("every cell has a label in every clustering (all seeds, both rules)", f"{n_missing} missing",
               n_missing == 0)
    small = clus_all[(clus_all["n_cells"] < lib.MIN_CLUSTER_SIZE) != (clus_all["top25_label"] == lib.BELOW_MIN)]
    checks.add("clusters < 5 cells are labelled below_min_size and only those (seed 42)",
               f"{len(small)} violating clusters", small.empty)
    if not checks.ok:
        stop(run, checks)

    # Per-cell stability over seeds 0-9 against seed 42
    seeds = pd.DataFrame(seed_rows)
    stab_rows, stab_cells = [], pd.DataFrame(index=per_cell.index)
    for res in lib.SCAN_RESOLUTIONS:
        for rule in lib.RULES:
            name = lib.clustering_name(res)
            lab42 = per_cell[f"{name}__{rule}_label"].to_numpy()
            others = np.column_stack([seed_cols[f"{lib.clustering_name(res, s)}__{rule}_label"]
                                      for s in lib.SEEDS if s != lib.SEED])
            frac = (others == lab42[:, None]).mean(axis=1)
            stab_cells[f"{name}__{rule}_stability"] = frac
            for label in [*cc.ordered_labels(lab42, lib.LABEL_ORDER), "all"]:
                m = np.ones(len(lab42), bool) if label == "all" else lab42 == label
                stab_rows.append({"resolution": res, "rule": rule, "label_seed42": label, "n_cells": int(m.sum()),
                                  "mean_stability": float(frac[m].mean()), "median_stability": float(np.median(frac[m])),
                                  "n_stable_all_10_seeds": int((frac[m] == 1).sum()),
                                  "fraction_stable_all_10_seeds": float((frac[m] == 1).mean()),
                                  "n_stability_below_0.5": int((frac[m] < 0.5).sum())})
    stability = pd.DataFrame(stab_rows)

    # Summary per resolution x rule
    rows = []
    for (res, rule), grp in seeds.groupby(["resolution", "rule"], sort=False):
        ref = grp[grp["seed"] == lib.SEED].iloc[0].to_dict()
        other = grp[grp["seed"] != lib.SEED]
        name = lib.clustering_name(res)
        lab = per_cell[f"{name}__{rule}_label"].astype(str).to_numpy()
        conf = per_cell[f"{name}__{rule}_confidence"].astype(str).to_numpy()
        row = {k: ref[k] for k in ("resolution", "rule", "n_clusters", "cluster_size_min", "cluster_size_median",
                                   "cluster_size_max", "n_clusters_below_min", "n_cells_below_min",
                                   "n_clusters_unknown", "fraction_cells_unknown", "n_types_resolved",
                                   "types_resolved", "label_counts", "eta2_protein_ids")}
        row["rule_display"] = lib.RULE_DISPLAY[rule]
        for label in lib.LABEL_ORDER:
            row[f"n_{label}"] = int((lab == label).sum())
        for c in lib.CONFIDENCE_ORDER:
            row[f"n_cells_conf_{c}"] = int((conf == c).sum())
        # Information: cells in clusters whose top-25 DCT1 call rests only on Umod and/or Slc12a3 (cluster_de flag).
        flag = clus_all[clus_all["resolution"] == res].set_index("cluster")["dct1_umod_slc12a3_only"]
        row["n_cells_top25_dct1_umod_slc12a3_only"] = int(per_cell[f"{name}__cluster"].map(flag).eq(True).sum())
        for key in ("n_clusters", "n_types_resolved", "fraction_cells_unknown"):
            row[f"{key}_all_seeds_min"] = float(grp[key].min())
            row[f"{key}_all_seeds_max"] = float(grp[key].max())
        for key in ("ari_clusters_vs_seed42", "ari_labels_vs_seed42", "fraction_same_label_as_seed42"):
            row[f"{key}_median"] = float(other[key].median())
            row[f"{key}_min"] = float(other[key].min())
            row[f"{key}_max"] = float(other[key].max())
        st = stability[(stability["resolution"] == res) & (stability["rule"] == rule) & (stability["label_seed42"] == "all")]
        row["per_cell_stability_mean"] = float(st["mean_stability"].iloc[0])
        row["fraction_cells_stable_all_10_seeds"] = float(st["fraction_stable_all_10_seeds"].iloc[0])
        row["type_sets_seen_all_seeds"] = " | ".join(sorted(set(
            ";".join(t.split("(")[0] for t in v.split(";") if t) for v in grp["types_resolved"])))
        for col in STEP1_LABELS:
            truth = per_cell[col].astype(str).to_numpy()
            short = col.replace("ctrl_", "").replace("_label", "")
            row[f"ari_vs_{col}"] = float(adjusted_rand_score(truth, lab))
            row[f"fraction_equal_{col}"] = float((truth == lab).mean())
            core = np.isin(lab, lib.CORE_TYPES)
            row[f"fraction_equal_{col}_core_labelled_cells"] = float((truth[core] == lab[core]).mean()) if core.any() else np.nan
            row[f"ari_clusters_vs_{short}"] = float(adjusted_rand_score(truth, per_cell[f"{name}__cluster"]))
        p1 = clus_all.loc[clus_all["resolution"] == res, "plate_P1_fraction"]
        row["cluster_P1_fraction_min"] = float(p1.min())
        row["cluster_P1_fraction_max"] = float(p1.max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    front = ["resolution", "rule", "rule_display"]
    summary = summary[front + [c for c in summary.columns if c not in front]]
    for r in summary.itertuples(index=False):
        checks.add(f"r{r.resolution:g} {r.rule}: clusters; labels; ARI labels vs seed 42 (median, min)",
                   f"{r.n_clusters}; {r.label_counts}; {r.ari_labels_vs_seed42_median:.3f}, "
                   f"{r.ari_labels_vs_seed42_min:.3f}", None)

    crosstabs = []
    for res in lib.SCAN_RESOLUTIONS:
        for rule in lib.RULES:
            lab = per_cell[f"{lib.clustering_name(res)}__{rule}_label"].astype(str)
            for col in STEP1_LABELS:
                ct = pd.crosstab(lab, per_cell[col].astype(str))
                for row_label, values in ct.iterrows():
                    crosstabs.append({"resolution": res, "rule": rule, "versus": col, "deg_label": row_label,
                                      **values.to_dict()})

    run.table(checks.frame(), "scan_checks")
    run.table(summary, "scan_summary")
    run.table(seeds, "scan_seed_stability")
    run.table(stability, "scan_per_cell_stability_summary")
    run.table(clus_all, "scan_cluster_annotation")
    run.table(ev_all, "scan_marker_evidence")
    run.table(de_top_all, "scan_de_top25")
    run.table(pd.DataFrame(type_rows), "scan_types_by_resolution")
    run.table(pd.DataFrame(crosstabs).fillna(0), "scan_crosstab_vs_step1")
    run.table(per_cell.join(stab_cells), "scan_clusters_per_cell", index=True)
    run.table(pd.DataFrame(seed_cols, index=per_cell.index), "scan_seed_clusters_per_cell", index=True)
    summary_figure(run, summary, seeds)
    umap_figure(run, np.asarray(adata.obsm["X_umap"]), per_cell, step1)
    run.finish({"resolutions": list(lib.SCAN_RESOLUTIONS), "seeds": list(lib.SEEDS),
                "min_cluster_size": lib.MIN_CLUSTER_SIZE, "below_min_label": lib.BELOW_MIN,
                "treated_as_shared": lib.TREATED_AS_SHARED, "type_specific_markers": specific,
                "leiden": "sc.tl.leiden flavor='igraph' on the step-1 graph (cluster_de_lib.leiden)",
                "annotation": ("cluster_de_lib.de_one_vs_rest / marker_evidence / assign (top 25) on ln(1 + intensity) "
                               "(float64); x_z = step-1 scaled .X; clusters < 5 cells dropped before marker_evidence"),
                "rankfree": "kidney_annotation_rankfree_sensitivity.rank_free (type-specific markers, Umod excluded)",
                "scanpy": sc.__version__, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
