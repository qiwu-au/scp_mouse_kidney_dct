#!/usr/bin/env python
"""dct_labels step 2: the frozen primary rule per condition, with the resolution scan as information.

Analysis plan, step 2
(repeats labels_core steps 2a-2b and zerok_core step 4 route b per condition).

  Primary rule, fixed in advance: cluster_de DE rule with the kidney_annotation rank-free rescue on Leiden 1.0 (igraph, seed 42) of the
  step-1 graph; Slc12a3 and Umod shared; minimum cluster size 5 -> cs_ctrl_deg_* (Ctrl) and cs_zerok_pipeline_*
  (ZeroK). The resolution scan (0.3, 0.5, 0.8, 1.0, 3.0; top-25 and rank-free; seeds 42 and 0-9; eta^2 of protein
  IDs on cluster) is reported as information only and chooses nothing. Cluster tables as labels_core step 2 / zerok_core step 4,
  with the intake_core contamination panel per cluster (information only, never a label).

Inputs (read-only): results/h5ad/dct_labels_DCT_{ctrl,zerok}_step1_submitted.h5ad;
results/tables/dct_labels/dct_labels_submitted_per_cell.csv; data/DCT/DCT_markers_V1.xlsx.
Outputs: results/tables/dct_labels/dct_labels_{scan,deg}_*.csv and dct_labels_primary_checks.csv;
results/figures/dct_labels/dct_labels_scan_{summary,umaps}_{ctrl,zerok}.

Command (project root):  python scripts/dct_labels_primary_labels.py
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
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import labels_core_lib as t18
import dct_labels_lib as lib
from cluster_de_deg_annotation import evidence_checks
from labels_core_resolution_scan import seed_row, wilcoxon_check

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "dct_labels_primary_labels.py"
STEP1_LABELS = ("cell_label", "cluster_label")
MM = 1 / 25.4


# ---------------------------------------------------------------------------
# Figures (labels_core_resolution_scan figures with the cell count and condition as parameters)
# ---------------------------------------------------------------------------
def summary_figure(run: cc.Run, summary: pd.DataFrame, seeds: pd.DataFrame, cond: str, n_cells: int) -> None:
    import matplotlib.pyplot as plt

    t18.apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 120 * MM))
    res_pos = {r: i for i, r in enumerate(t18.SCAN_RESOLUTIONS)}
    rule_style = {"top25": (t18.SERIES[0], -0.17), "rankfree": (t18.SERIES[1], 0.17)}

    ax = axes[0, 0]
    top = seeds[seeds["rule"] == "top25"]
    for r, grp in top.groupby("resolution"):
        x = res_pos[r]
        ax.vlines(x, grp["n_clusters"].min(), grp["n_clusters"].max(), color=t18.INK["muted"], lw=1)
        ax.plot(x, int(grp.loc[grp["seed"] == lib.SEED, "n_clusters"].iloc[0]), "o", color=t18.INK["primary"], ms=4)
    ax.set_xticks(range(len(res_pos)), [f"{r:g}" for r in res_pos])
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Clusters")
    ax.set_title("a  Clusters (dot: seed 42; line: 11-seed range)")

    ax = axes[0, 1]
    width = 0.32
    for rule, (color, shift) in rule_style.items():
        bottom = np.zeros(len(res_pos))
        for label in t18.LABEL_ORDER:
            vals = np.array([summary.loc[(summary["rule"] == rule) & (summary["resolution"] == r), f"n_{label}"].iloc[0]
                             for r in res_pos], float) / n_cells
            ax.bar(np.arange(len(res_pos)) + shift, vals, width, bottom=bottom, color=t18.LABEL_COLORS[label],
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
        ax.plot(x, sub["ari_labels_vs_seed42_median"], "o", color=color, ms=4, label=t18.RULE_DISPLAY[rule])
    ax.set_xticks(range(len(res_pos)), [f"{r:g}" for r in res_pos])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("ARI of labels vs seed 42\n(seeds 0-9; dot: median, line: range)")
    ax.set_title("c  Label stability over Leiden seeds")
    ax.legend(loc="lower right", fontsize=6)

    ax = axes[1, 1]
    conf_colors = {"high": t18.SERIES[2], "medium": t18.SERIES[3], t18.RESCUED: t18.SERIES[4],
                   "low_shared_only": t18.SERIES[7], "unknown_no_marker_in_top_de": t18.LABEL_COLORS["Unknown"],
                   t18.BELOW_MIN: t18.LABEL_COLORS[t18.BELOW_MIN]}
    for rule, (color, shift) in rule_style.items():
        bottom = np.zeros(len(res_pos))
        for conf, ccol in conf_colors.items():
            vals = np.array([summary.loc[(summary["rule"] == rule) & (summary["resolution"] == r),
                                         f"n_cells_conf_{conf}"].iloc[0] for r in res_pos], float) / n_cells
            ax.bar(np.arange(len(res_pos)) + shift, vals, width, bottom=bottom, color=ccol, edgecolor="white",
                   lw=0.3, label=conf if rule == "top25" else None)
            bottom += vals
    ax.set_xticks(range(len(res_pos)), [f"{r:g}\nT  R" for r in res_pos])
    ax.set_xlabel("Leiden resolution (T = top-25 rule, R = rank-free rescue)")
    ax.set_ylabel("Fraction of cells (seed 42)")
    ax.set_title("d  Confidence of the cluster call")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 1), w_pad=3.0)
    t18.footer(fig, SCRIPT, f"{cond} DCT cells, combined search (n = {n_cells}); graph of the submitted pipeline "
                            "(k = 10, 20 PCs); information only (the primary rule is fixed: rank-free, Leiden 1.0)")
    run.figure(fig, f"scan_summary_{lib.KEY[cond]}",
               f"dct_labels resolution scan ({cond}): clusters, DE-rule labels, seed stability, confidence")


def wrapped_counts(lab: np.ndarray) -> str:
    parts = lib.counts_text(lab).replace("Unknown", "Unk").split("; ")
    half = (len(parts) + 1) // 2
    return "; ".join(parts[:half]) + "\n" + "; ".join(parts[half:])


def umap_figure(run: cc.Run, umap: np.ndarray, per_cell: pd.DataFrame, step1: pd.DataFrame, cond: str) -> None:
    import matplotlib.pyplot as plt

    t18.apply_style()
    n = len(t18.SCAN_RESOLUTIONS)
    fig, axes = plt.subplots(3, n + 1, figsize=(180 * MM, 125 * MM))
    cmap = plt.get_cmap("tab20")
    for j, res in enumerate(t18.SCAN_RESOLUTIONS):
        name = t18.clustering_name(res)
        cl = per_cell[f"{name}__cluster"].to_numpy()
        ax = axes[0, j]
        order = t5.sort_clusters(cl)
        idx = {c: i for i, c in enumerate(order)}
        ax.scatter(umap[:, 0], umap[:, 1], s=1.2, c=[cmap(idx[c] % 20) for c in cl], lw=0, rasterized=True)
        for c in order:
            m = cl == c
            ax.text(np.median(umap[m, 0]), np.median(umap[m, 1]), c, fontsize=5, ha="center", va="center",
                    color=t18.INK["primary"])
        ax.set_title(f"Leiden {res:g}: {len(order)} clusters", fontsize=6.5)
        for i, rule in enumerate(t18.RULES, start=1):
            lab = per_cell[f"{name}__{rule}_label"].astype(str).to_numpy()
            ax = axes[i, j]
            for label in t18.LABEL_ORDER:
                m = lab == label
                if m.any():
                    ax.scatter(umap[m, 0], umap[m, 1], s=1.2, color=t18.LABEL_COLORS[label], lw=0, rasterized=True)
            ax.set_title(f"{'T' if rule == 'top25' else 'R'} {res:g}\n{wrapped_counts(lab)}", fontsize=5)
    for i, col in enumerate(STEP1_LABELS):
        ax = axes[i, n]
        lab = step1[col].astype(str).to_numpy()
        for label in t18.LABEL_ORDER:
            m = lab == label
            if m.any():
                ax.scatter(umap[m, 0], umap[m, 1], s=1.2, color=t18.LABEL_COLORS[label], lw=0, rasterized=True)
        ax.set_title(f"{lib.cs(cond, col)} (step 1)\n{wrapped_counts(lab)}", fontsize=5)
    ax = axes[2, n]
    ids = step1["n_detected"].to_numpy(float)
    order = np.argsort(ids)
    pts = ax.scatter(umap[order, 0], umap[order, 1], s=1.2, c=ids[order], cmap="viridis", lw=0, rasterized=True)
    cbar = fig.colorbar(pts, ax=ax, fraction=0.05, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    ax.set_title("Protein IDs per cell", fontsize=5.5)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3.5, color=t18.LABEL_COLORS[label], label=label)
               for label in t18.LABEL_ORDER]
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
    t18.footer(fig, SCRIPT, f"{cond}, combined search; UMAP of the submitted pipeline (step 1); labels at seed 42")
    run.figure(fig, f"scan_umaps_{lib.KEY[cond]}",
               f"dct_labels resolution scan ({cond}) on the step-1 UMAP: clusters and DE-rule labels per resolution")


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------
def one_condition(run: cc.Run, checks: t5.Checks, cond: str, step1_all: pd.DataFrame) -> dict:
    key = lib.KEY[cond]
    adata = lib.read_step1(run, cond)
    cell_ids = adata.obs_names.astype(str)
    step1 = lib.per_condition(step1_all, cond, cell_ids)
    inp = lib.de_inputs(adata)
    checks.add(f"{cond}: intensity layer: no negative or missing values", f"min {inp.intensity.min():.3g}",
               bool(np.isfinite(inp.intensity).all() and inp.intensity.min() >= 0))
    ids = step1["n_detected"].to_numpy(float)
    markers = t18.dct_markers(run, inp.var_names)
    specific = t18.specific_present(markers)
    checks.add(f"{cond}: type-specific markers present (Slc12a3 shared by the workbook; Umod treated as shared)",
               "; ".join(f"{t}: {','.join(g)}" for t, g in specific.items()), None)
    checks.add(f"{cond}: shared markers present",
               ";".join(sorted({g for t in markers.present for g in markers.present[t] if markers.is_shared(g)})), None)
    var_index = {v: i for i, v in enumerate(inp.var_names)}
    if not checks.ok:
        lib.stop(run, checks, "primary_checks")

    per_cell = step1[["plate", "n_detected", *STEP1_LABELS, "leiden3_cluster"]].copy()
    seed_cols: dict[str, np.ndarray] = {}
    seed_rows, clus_all, ev_all, de_top_all, type_rows = [], [], [], [], []
    chosen = None
    for res in t18.SCAN_RESOLUTIONS:
        t0 = time.time()
        ref: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        per_seed_types: dict[str, list[dict]] = {r: [] for r in t18.RULES}
        for seed in t18.SEEDS:
            out = lib.label_clustering(inp, markers, res, seed)
            name, cl, tabs = out["name"], out["clusters"], out["tables"]
            if seed == lib.SEED:
                if res == t18.SCAN_RESOLUTIONS[0]:
                    wilcoxon_check(checks, inp.cache, cl, out["de"], f"{cond} {name}", list(inp.var_names))
                if res == lib.RESOLUTION:
                    chosen = out
                for rule in t18.RULES:
                    lab = out[f"{rule}_label"]
                    ref[rule] = (cl, lab)
                    per_cell[f"{name}__{rule}_label"] = lab
                    per_cell[f"{name}__{rule}_confidence"] = out[f"{rule}_confidence"]
                per_cell[f"{name}__cluster"] = cl
                clus_all.append(lib.cluster_rows(res, cl, out["assignment"], out["rankfree"], out["de"], step1,
                                                 inp.intensity, var_index, STEP1_LABELS))
                ev = out["evidence"].copy()
                ev.insert(1, "resolution", res)
                ev_all.append(ev)
                de_top_all.append(out["de"][out["de"]["in_top"]].sort_values(["cluster", "rank_up"]))
                rf = out["rankfree"]
                kept = rf[rf["top25_confidence"].isin(["high", "medium"])]
                bad = kept[kept["rankfree_label"] != kept["top25_label"]]
                checks.add(f"{cond} r{res:g}: rank-free keeps every high/medium top-25 label", f"{len(bad)} changed",
                           bad.empty)
            else:
                seed_cols[f"{name}__cluster"] = cl
            for rule, tab in tabs.items():
                lab = out[f"{rule}_label"]
                if seed != lib.SEED:
                    seed_cols[f"{name}__{rule}_label"] = lab
                per_seed_types[rule].append(t18.types_resolved(tab))
                seed_rows.append(seed_row(res, seed, rule, cl, lab, tab, ids, *ref[rule]))
        for rule in t18.RULES:
            cl42 = per_cell[f"{t18.clustering_name(res)}__cluster"].to_numpy()
            lab42 = per_cell[f"{t18.clustering_name(res)}__{rule}_label"].to_numpy()
            conf42 = per_cell[f"{t18.clustering_name(res)}__{rule}_confidence"].to_numpy()
            sizes = pd.Series(cl42).value_counts()
            for t in t18.CORE_TYPES:
                m = lab42 == t
                cls = sorted(set(cl42[m]), key=lambda c: t5.sort_clusters(cl42).index(c))
                best = t18.types_resolved(pd.DataFrame({"deg_label": lab42[m], "confidence": conf42[m]})).get(t, "")
                type_rows.append({"condition": cond, "resolution": res, "rule": rule, "cell_type": t,
                                  "n_clusters_seed42": len(cls), "n_cells_seed42": int(m.sum()),
                                  "best_confidence_seed42": best,
                                  "clusters_seed42": ";".join(f"{c}({sizes[c]})" for c in cls),
                                  "n_seeds_resolved": sum(t in d for d in per_seed_types[rule]),
                                  "n_seeds_resolved_ge_medium": sum(d.get(t) in ("high", "medium")
                                                                    for d in per_seed_types[rule]),
                                  "n_seeds": len(per_seed_types[rule])})
        r42 = [r for r in seed_rows if r["resolution"] == res and r["seed"] == lib.SEED]
        print(f"[{key}] r{res:g}: {r42[0]['n_clusters']} clusters; top25 {r42[0]['label_counts']}; rankfree "
              f"{r42[1]['label_counts']}; {time.time() - t0:.1f} s", flush=True)

    a = float(adjusted_rand_score(step1["leiden3_cluster"].astype(str), per_cell["r3.0__cluster"]))
    checks.add(f"{cond}: Leiden 3.0 (seed 42) equals step-1 leiden3_cluster (ARI)", f"{a:.6f}", a == 1.0, "= 1")
    clus_all = pd.concat(clus_all, ignore_index=True)
    clus_all = cc.add_bh(clus_all, ["resolution"], p_col="protein_ids_p_vs_rest", out_col="protein_ids_q_vs_rest")
    ev_all = pd.concat(ev_all, ignore_index=True)
    de_top_all = pd.concat(de_top_all, ignore_index=True)
    evidence_checks(checks, ev_all, de_top_all)
    n_missing = int(per_cell.filter(like="_label").isna().sum().sum()) + sum(int(pd.isna(v).sum())
                                                                           for v in seed_cols.values())
    checks.add(f"{cond}: every cell has a label in every clustering (all seeds, both rules)", f"{n_missing} missing",
               n_missing == 0)
    small = clus_all[(clus_all["n_cells"] < t18.MIN_CLUSTER_SIZE) != (clus_all["top25_label"] == t18.BELOW_MIN)]
    checks.add(f"{cond}: clusters < 5 cells are labelled below_min_size and only those (seed 42)",
               f"{len(small)} violating clusters", small.empty)
    if not checks.ok:
        lib.stop(run, checks, "primary_checks")

    # Per-cell stability over seeds 0-9 against seed 42
    seeds = pd.DataFrame(seed_rows)
    stab_rows, stab_cells = [], pd.DataFrame(index=per_cell.index)
    for res in t18.SCAN_RESOLUTIONS:
        for rule in t18.RULES:
            name = t18.clustering_name(res)
            lab42 = per_cell[f"{name}__{rule}_label"].to_numpy()
            others = np.column_stack([seed_cols[f"{t18.clustering_name(res, s)}__{rule}_label"]
                                      for s in t18.SEEDS if s != lib.SEED])
            frac = (others == lab42[:, None]).mean(axis=1)
            stab_cells[f"{name}__{rule}_stability"] = frac
            for label in [*cc.ordered_labels(lab42, t18.LABEL_ORDER), "all"]:
                m = np.ones(len(lab42), bool) if label == "all" else lab42 == label
                stab_rows.append({"condition": cond, "resolution": res, "rule": rule, "label_seed42": label,
                                  "n_cells": int(m.sum()), "mean_stability": float(frac[m].mean()),
                                  "median_stability": float(np.median(frac[m])),
                                  "n_stable_all_10_seeds": int((frac[m] == 1).sum()),
                                  "fraction_stable_all_10_seeds": float((frac[m] == 1).mean()),
                                  "n_stability_below_0.5": int((frac[m] < 0.5).sum())})
    stability = pd.DataFrame(stab_rows)

    # Summary per resolution x rule
    rows = []
    for (res, rule), grp in seeds.groupby(["resolution", "rule"], sort=False):
        ref = grp[grp["seed"] == lib.SEED].iloc[0].to_dict()
        other = grp[grp["seed"] != lib.SEED]
        name = t18.clustering_name(res)
        lab = per_cell[f"{name}__{rule}_label"].astype(str).to_numpy()
        conf = per_cell[f"{name}__{rule}_confidence"].astype(str).to_numpy()
        row = {"condition": cond}
        row.update({k: ref[k] for k in ("resolution", "rule", "n_clusters", "cluster_size_min", "cluster_size_median",
                                        "cluster_size_max", "n_clusters_below_min", "n_cells_below_min",
                                        "n_clusters_unknown", "fraction_cells_unknown", "n_types_resolved",
                                        "types_resolved", "label_counts", "eta2_protein_ids")})
        row["rule_display"] = t18.RULE_DISPLAY[rule]
        row["frozen_primary"] = bool(res == lib.RESOLUTION and rule == lib.RULE)
        for label in t18.LABEL_ORDER:
            row[f"n_{label}"] = int((lab == label).sum())
        for c in t18.CONFIDENCE_ORDER:
            row[f"n_cells_conf_{c}"] = int((conf == c).sum())
        flag = clus_all[clus_all["resolution"] == res].set_index("cluster")["dct1_umod_slc12a3_only"]
        row["n_cells_top25_dct1_umod_slc12a3_only"] = int(per_cell[f"{name}__cluster"].map(flag).eq(True).sum())
        for k in ("n_clusters", "n_types_resolved", "fraction_cells_unknown"):
            row[f"{k}_all_seeds_min"] = float(grp[k].min())
            row[f"{k}_all_seeds_max"] = float(grp[k].max())
        for k in ("ari_clusters_vs_seed42", "ari_labels_vs_seed42", "fraction_same_label_as_seed42"):
            row[f"{k}_median"] = float(other[k].median())
            row[f"{k}_min"] = float(other[k].min())
            row[f"{k}_max"] = float(other[k].max())
        st = stability[(stability["resolution"] == res) & (stability["rule"] == rule)
                       & (stability["label_seed42"] == "all")]
        row["per_cell_stability_mean"] = float(st["mean_stability"].iloc[0])
        row["fraction_cells_stable_all_10_seeds"] = float(st["fraction_stable_all_10_seeds"].iloc[0])
        row["type_sets_seen_all_seeds"] = " | ".join(sorted(set(
            ";".join(t.split("(")[0] for t in v.split(";") if t) for v in grp["types_resolved"])))
        for col in STEP1_LABELS:
            truth = per_cell[col].astype(str).to_numpy()
            row[f"ari_vs_{col}"] = float(adjusted_rand_score(truth, lab))
            row[f"fraction_equal_{col}"] = float((truth == lab).mean())
            core = np.isin(lab, t18.CORE_TYPES)
            row[f"fraction_equal_{col}_core_labelled_cells"] = (float((truth[core] == lab[core]).mean())
                                                                if core.any() else np.nan)
            row[f"ari_clusters_vs_{col}"] = float(adjusted_rand_score(truth, per_cell[f"{name}__cluster"]))
        p1 = clus_all.loc[clus_all["resolution"] == res, "plate_P1_fraction"]
        row["cluster_P1_fraction_min"] = float(p1.min())
        row["cluster_P1_fraction_max"] = float(p1.max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    front = ["condition", "resolution", "rule", "rule_display", "frozen_primary"]
    summary = summary[front + [c for c in summary.columns if c not in front]]
    for r in summary.itertuples(index=False):
        checks.add(f"{cond} r{r.resolution:g} {r.rule}: clusters; labels; ARI labels vs seed 42 (median, min)",
                   f"{r.n_clusters}; {r.label_counts}; {r.ari_labels_vs_seed42_median:.3f}, "
                   f"{r.ari_labels_vs_seed42_min:.3f}", None)

    crosstabs = []
    for res in t18.SCAN_RESOLUTIONS:
        for rule in t18.RULES:
            lab = per_cell[f"{t18.clustering_name(res)}__{rule}_label"].astype(str)
            for col in STEP1_LABELS:
                ct = pd.crosstab(lab, per_cell[col].astype(str))
                for row_label, values in ct.iterrows():
                    crosstabs.append({"condition": cond, "resolution": res, "rule": rule, "versus": lib.cs(cond, col),
                                      "deg_label": row_label, **values.to_dict()})

    # ---- the frozen primary rule (Leiden 1.0, rank-free, seed 42)
    cl, lab, conf = chosen["clusters"], chosen["label"], chosen["confidence"]
    name = chosen["name"]
    same_cl = bool((per_cell[f"{name}__cluster"].to_numpy() == cl).all())
    same_lab = bool((per_cell[f"{name}__{lib.RULE}_label"].to_numpy() == lab).all())
    checks.add(f"{cond}: primary labels equal the scan at Leiden 1.0 rank-free seed 42 (clusters; labels)",
               f"{same_cl}; {same_lab}", same_cl and same_lab)
    checks.add(f"{cond}: {lib.cs(cond, 'deg_label')} counts", lib.counts_text(lab), None)
    checks.add(f"{cond}: {lib.cs(cond, 'deg_confidence')} counts (cells)",
               "; ".join(f"{k} {v}" for k, v in pd.Series(conf).value_counts().items()), None)
    missing_types = [t for t in t18.CORE_TYPES if t not in set(lab)]
    checks.add(f"{cond}: workbook types with no cell under the primary rule (reported as observed; nothing changed)",
               ";".join(missing_types) or "none", None)
    deg = pd.DataFrame({"condition": cond, "cell_id": cell_ids, "cell_key": step1["cell_key"].to_numpy(),
                        "deg_cluster": cl, "deg_label": lab, "deg_confidence": conf,
                        "top25_label": chosen["top25_label"], "top25_confidence": chosen["top25_confidence"],
                        "canonical_label_column": lib.cs(cond, "deg_label")})
    counts = []
    for col in ("deg_label", "deg_confidence"):
        vc = deg[col].value_counts()
        for k in vc.index:
            counts.append({"condition": cond, "column": lib.cs(cond, col), "value": k, "n_cells": int(vc[k]),
                           "fraction": float(vc[k] / len(deg))})
    clusters = lib.cluster_rows(lib.RESOLUTION, cl, chosen["assignment"], chosen["rankfree"], chosen["de"], step1,
                                inp.intensity, var_index, STEP1_LABELS)
    clusters.insert(1, "deg_label", clusters[f"{lib.RULE}_label"])
    clusters.insert(2, "deg_confidence", clusters[f"{lib.RULE}_confidence"])
    clusters = cc.add_bh(clusters, ["resolution"], p_col="protein_ids_p_vs_rest", out_col="protein_ids_q_vs_rest")
    clusters.insert(0, "condition", cond)
    ev = chosen["evidence"].copy()
    ev.insert(0, "condition", cond)
    de = chosen["de"].copy()
    de.insert(0, "condition", cond)

    for frame in (seeds, clus_all, ev_all, de_top_all):
        frame.insert(0, "condition", cond)
    per_cell = per_cell.join(stab_cells)
    per_cell.insert(0, "condition", cond)
    seed_frame = pd.DataFrame(seed_cols, index=per_cell.index)
    seed_frame.insert(0, "condition", cond)
    summary_figure(run, summary, seeds, cond, len(cell_ids))
    umap_figure(run, np.asarray(adata.obsm["X_umap"]), per_cell, step1, cond)
    return {"summary": summary, "seeds": seeds, "stability": stability, "clus_all": clus_all, "ev_all": ev_all,
            "de_top_all": de_top_all, "types": pd.DataFrame(type_rows), "crosstabs": pd.DataFrame(crosstabs),
            "per_cell": per_cell.reset_index(), "seed_cols": seed_frame.reset_index(), "deg": deg,
            "counts": pd.DataFrame(counts), "clusters": clusters, "ev": ev, "de": de,
            "specific": specific}


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_labels step 2: primary rule per condition; resolution scan (info).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "primary_labels", args)
    t_start = time.time()
    checks = t5.Checks("dct_combined")
    step1_all = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str, "leiden3_cluster": str, "well": str})
    out = {cond: one_condition(run, checks, cond, step1_all) for cond in lib.CONDITIONS}

    def cat(key: str) -> pd.DataFrame:
        return pd.concat([out[c][key] for c in lib.CONDITIONS], ignore_index=True)

    run.table(checks.frame(), "primary_checks")
    run.table(cat("summary"), "scan_summary")
    run.table(cat("seeds"), "scan_seed_stability")
    run.table(cat("stability"), "scan_per_cell_stability_summary")
    run.table(cat("clus_all"), "scan_cluster_annotation")
    run.table(cat("ev_all"), "scan_marker_evidence")
    run.table(cat("de_top_all"), "scan_de_top25")
    run.table(cat("types"), "scan_types_by_resolution")
    run.table(cat("crosstabs").fillna(0), "scan_crosstab_vs_step1")
    run.table(cat("per_cell"), "scan_clusters_per_cell")
    run.table(cat("seed_cols"), "scan_seed_clusters_per_cell")
    run.table(cat("deg"), "deg_labels_per_cell")
    run.table(cat("counts"), "deg_label_counts")
    run.table(cat("clusters"), "deg_cluster_annotation")
    run.table(cat("ev"), "deg_marker_evidence")
    run.table(cat("de"), "deg_de_tested")
    run.finish({"primary_rule": {"resolution": lib.RESOLUTION, "rule": lib.RULE, "seed": lib.SEED,
                                 "fixed_by": "dct_labels plan (frozen 2 Nov 2026); labels_core authors choice 28 Oct 2026"},
                "scan": {"resolutions": list(t18.SCAN_RESOLUTIONS), "seeds": list(t18.SEEDS), "status": "information"},
                "min_cluster_size": t18.MIN_CLUSTER_SIZE, "treated_as_shared": t18.TREATED_AS_SHARED,
                "type_specific_markers": {c: out[c]["specific"] for c in lib.CONDITIONS},
                "canonical_columns": {c: lib.cs(c, "deg_label") for c in lib.CONDITIONS},
                "scanpy": sc.__version__, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
