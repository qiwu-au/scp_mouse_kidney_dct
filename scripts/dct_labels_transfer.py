#!/usr/bin/env python
"""dct_labels step 3: ZeroK transfer route per normalization variant, with the zerok_core C4 checks.

Analysis plan, step 3
(repeats zerok_core step 4 route a and its checks).

  cs_zerok_transfer_label__<variant>: majority cs_ctrl_deg_label of the 10 nearest Ctrl cells in the dct_intake joint
      20-PC embedding (X_pca__<variant>); vote fraction = confidence; ties -> label of the nearest tied neighbour
      (C2). cs_zerok_transfer_label = variant 'none'. Functions imported unchanged from zerok_core_labels.
  Checks (C4): Ctrl leave-one-out self-transfer; neighbour-distance ratio (ZeroK to Ctrl over Ctrl to Ctrl);
      ZeroK pipeline seed stability (from the step-2 scan, Leiden 1.0 rank-free, seeds 0-9 vs 42); plate mixing
      within each condition (joint embedding per variant, as dct_intake; and each condition's own step-1 graph PCA).
  Shared-marker variant (C3): the primary rule with only the workbook markers present in BOTH conditions, for
      each condition (the marker sets are equal in the combined search, so this must reproduce the primary).
  Agreement of pipeline vs transfer (crosstab, fraction equal, ARI) and the cells with conflicting calls.

Inputs (read-only): results/h5ad/dct_intake_DCT_joint.h5ad; results/h5ad/dct_labels_DCT_{ctrl,zerok}_step1_submitted.h5ad;
results/tables/dct_labels/dct_labels_{deg_labels_per_cell,scan_seed_stability,scan_clusters_per_cell}.csv;
data/DCT/DCT_markers_V1.xlsx.
Outputs: results/tables/dct_labels/dct_labels_transfer_*.csv, dct_labels_zerok_pipeline_seed_stability.csv,
dct_labels_plate_mixing.csv, dct_labels_shared_marker_variant.csv; figures dct_labels_transfer_{zerok_labels,agreement}.

Command (project root):  python scripts/dct_labels_transfer.py
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

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import qc_mito_lib as t3
import cluster_de_lib as t5
import labels_core_lib as t18
import zerok_core_lib as t21
import dct_labels_lib as lib
from zerok_core_labels import self_transfer, transfer_labels, within_distance

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "dct_labels_transfer.py"
LABEL_ORDER = list(cc.DCT_LABEL_ORDER)
PLATES = ("P1", "P2")


def fig_labels(run, frame: pd.DataFrame, umap: np.ndarray) -> None:
    """zerok_core_labels.fig_labels with dct_labels column names (ZeroK cells on the baseline joint UMAP)."""
    colors = {**{k: v for k, v in zip(LABEL_ORDER, t3.SERIES)}, "Unknown": cc.UNKNOWN_COLOR}
    columns = [(f"transfer__{v}", f"transfer ({v})") for v in lib.VARIANTS]
    columns.append(("cs_zerok_pipeline_label", "pipeline (ZeroK alone)"))
    fig, axes = plt.subplots(1, len(columns), figsize=(7.4, 2.3))
    for ax, (col, title) in zip(axes, columns):
        t3.umap_categorical(ax, umap, frame[col], LABEL_ORDER, colors, title)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", ms=4, color=colors[l], label=l) for l in LABEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.015))
    t3.footer(fig, SCRIPT, "ZeroK cells on the combined-search joint UMAP ('none'); label names are those of "
                           "cs_ctrl_deg_label")
    run.figure(fig, "transfer_zerok_labels", "ZeroK labels by transfer (per variant) and by the primary pipeline")


def fig_agreement(run, agree: pd.DataFrame, counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    sub = agree[agree["comparison"].str.startswith("transfer")]
    axes[0].barh(np.arange(len(sub)), sub["fraction_equal"], color=t3.SERIES[0])
    axes[0].set_yticks(np.arange(len(sub)), sub["comparison"], fontsize=6)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("fraction of ZeroK cells with the same label")
    axes[0].invert_yaxis()
    t3.ygrid(axes[0])
    short = {"cs_zerok_pipeline_label": "pipeline", "cs_zerok_pipeline_label__shared_markers": "pipeline (C3)",
             **{f"transfer__{v}": f"transfer {v}" for v in lib.VARIANTS}}
    pivot = (counts.pivot(index="label", columns="column", values="n_cells").reindex(LABEL_ORDER)
             .reindex(columns=list(short)).fillna(0).rename(columns=short))
    x = np.arange(len(pivot.columns))
    bottom = np.zeros(len(pivot.columns))
    label_colors = {**{k: v for k, v in zip(LABEL_ORDER, t3.SERIES)}, "Unknown": cc.UNKNOWN_COLOR}
    for label in pivot.index:
        axes[1].bar(x, pivot.loc[label], 0.6, bottom=bottom, color=label_colors[label], label=label)
        bottom += pivot.loc[label].to_numpy()
    axes[1].set_xticks(x, pivot.columns, fontsize=5.5, rotation=30, ha="right")
    axes[1].set_ylabel("ZeroK cells")
    axes[1].set_ylim(0, bottom.max() * 1.18)
    axes[1].legend(fontsize=5.5, frameon=False, ncol=4, loc="upper center")
    t3.ygrid(axes[1])
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.27, top=0.95, wspace=0.35)
    t3.footer(fig, SCRIPT, "label counts are within ZeroK only; no comparison with Ctrl is made here (zerok_vs_control)")
    run.figure(fig, "transfer_agreement", "Agreement between the ZeroK label routes and label counts")


def main() -> None:
    import anndata as ad

    parser = argparse.ArgumentParser(description="dct_labels step 3: ZeroK transfer route and C3/C4 checks.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "transfer", args)
    t_start = time.time()
    checks = t5.Checks("dct_combined")

    joint = lib.t26.read_object(run, "joint")
    condition = joint.obs["condition"].astype(str).to_numpy()
    is_zerok = condition == "ZeroK"
    joint_cells = joint.obs_names.astype(str).to_numpy()
    deg = lib.read_table(run, "deg_labels_per_cell", dtype={"cell_id": str, "deg_cluster": str})
    objs = {c: lib.read_step1(run, c) for c in lib.CONDITIONS}
    for cond, mask in (("Ctrl", ~is_zerok), ("ZeroK", is_zerok)):
        same = list(objs[cond].obs_names.astype(str)) == list(joint_cells[mask])
        checks.add(f"joint {cond} cells equal the {cond} object cells in order", f"{same}", same)
    if not checks.ok:
        lib.stop(run, checks, "transfer_checks")
    ctrl_tab = lib.per_condition(deg, "Ctrl", joint_cells[~is_zerok])
    zerok_tab = lib.per_condition(deg, "ZeroK", joint_cells[is_zerok])
    ctrl_deg = ctrl_tab["deg_label"].astype(str).to_numpy()
    checks.add("cs_ctrl_deg_label (step 2) counts used as the transfer reference", lib.counts_text(ctrl_deg), None)

    # ---- transfer per variant
    frame = pd.DataFrame({"cell_id": joint_cells[is_zerok], "cell_key": joint.obs["cell_key"].astype(str).to_numpy()[is_zerok],
                          "plate": joint.obs["plate"].astype(str).to_numpy()[is_zerok],
                          "n_detected": objs["ZeroK"].obs["n_detected"].to_numpy()})
    self_rows = []
    for variant in lib.VARIANTS:
        pca = np.asarray(joint.obsm[f"X_pca__{variant}"])
        checks.add(f"X_pca__{variant}: components", f"{pca.shape[1]}", pca.shape[1] == t21.N_PCS, f"{t21.N_PCS}")
        labels, fractions, tied, idx, dist = transfer_labels(pca, is_zerok, ctrl_deg)
        frame[f"transfer__{variant}"] = labels
        frame[f"transfer_confidence__{variant}"] = fractions
        frame[f"transfer_tied__{variant}"] = tied
        loo = self_transfer(pca, is_zerok, ctrl_deg)
        d_cross = dist.mean(axis=1)
        d_ctrl = within_distance(pca[~is_zerok])
        d_zerok = within_distance(pca[is_zerok])
        frame[f"transfer_mean_distance__{variant}"] = d_cross
        self_rows.append({"variant": variant, "n_control_cells": len(ctrl_deg),
                          "leave_one_out_accuracy": float((loo == ctrl_deg).mean()),
                          **{f"accuracy_{l}": float((loo[ctrl_deg == l] == l).mean())
                             for l in LABEL_ORDER if (ctrl_deg == l).any()},
                          "n_tied_votes_zerok": int(tied.sum()),
                          "median_distance_zerok_to_ctrl": float(np.median(d_cross)),
                          "median_distance_ctrl_to_ctrl": float(np.median(d_ctrl)),
                          "median_distance_zerok_to_zerok": float(np.median(d_zerok)),
                          "distance_ratio_cross_over_within_ctrl": float(np.median(d_cross) / np.median(d_ctrl))})
        run.note(f"{variant}: Ctrl leave-one-out accuracy {self_rows[-1]['leave_one_out_accuracy']:.3f}; ZeroK transfer "
                 f"{lib.counts_text(labels)}; {int(tied.sum())} tied votes; distance ratio "
                 f"{self_rows[-1]['distance_ratio_cross_over_within_ctrl']:.2f}")
    frame["cs_zerok_transfer_label"] = frame[f"transfer__{lib.BASELINE_VARIANT}"]
    frame["cs_zerok_transfer_confidence"] = frame[f"transfer_confidence__{lib.BASELINE_VARIANT}"]
    frame["cs_zerok_pipeline_cluster"] = zerok_tab["deg_cluster"].astype(str).to_numpy()
    frame["cs_zerok_pipeline_label"] = zerok_tab["deg_label"].astype(str).to_numpy()
    frame["cs_zerok_pipeline_confidence"] = zerok_tab["deg_confidence"].astype(str).to_numpy()

    # ---- ZeroK pipeline seed stability (step-2 scan, Leiden 1.0 rank-free)
    seeds = lib.read_table(run, "scan_seed_stability")
    stab = seeds[(seeds["condition"] == "ZeroK") & (seeds["resolution"] == lib.RESOLUTION) & (seeds["rule"] == lib.RULE)
                 & (seeds["seed"] != lib.SEED)][["seed", "n_clusters", "ari_clusters_vs_seed42", "ari_labels_vs_seed42",
                                                  "fraction_same_label_as_seed42", "label_counts"]].reset_index(drop=True)
    scan_cells = lib.read_table(run, "scan_clusters_per_cell", dtype={"cell_id": str})
    frame["cs_zerok_pipeline_seed_stability"] = lib.per_condition(scan_cells, "ZeroK", frame["cell_id"])[
        f"{t18.clustering_name(lib.RESOLUTION)}__{lib.RULE}_stability"].to_numpy()
    checks.add("ZeroK pipeline seed stability: fraction of cells keeping the seed-42 label, median (min)",
               f"{stab['fraction_same_label_as_seed42'].median():.3f} ({stab['fraction_same_label_as_seed42'].min():.3f})",
               None)

    # ---- shared-marker variant (C3), both conditions
    workbook = t18.read_workbook(run)
    present = {c: {g for gs in t18.present_lists(workbook, objs[c].var_names).values() for g in gs}
               for c in lib.CONDITIONS}
    shared_rows = []
    for cond in lib.CONDITIONS:
        other = "ZeroK" if cond == "Ctrl" else "Ctrl"
        only = sorted(present[cond] - present[other])
        checks.add(f"{cond}: workbook markers present in {cond} but not in {other} (dropped in C3)",
                   ";".join(only) or "(none)", None)
        inp = lib.de_inputs(objs[cond])
        markers = t18.drop_markers(t18.dct_markers(run, inp.var_names), only)
        out = lib.label_clustering(inp, markers, lib.RESOLUTION, lib.SEED)
        prim = lib.per_condition(deg, cond, objs[cond].obs_names)["deg_label"].astype(str).to_numpy()
        same = int((out["label"] == prim).sum())
        checks.add(f"{cond}: shared-marker variant (C3) reproduces {lib.primary_column(cond)}",
                   f"{same}/{len(prim)}; {lib.counts_text(out['label'])}", None)
        shared_rows.append({"condition": cond, "markers_dropped": ";".join(only) or "(none)",
                            "primary_column": lib.primary_column(cond), "n_cells": len(prim), "n_equal": same,
                            "fraction_equal": same / len(prim), "label_counts_c3": lib.counts_text(out["label"]),
                            "label_counts_primary": lib.counts_text(prim)})
        if cond == "ZeroK":
            frame["cs_zerok_pipeline_label__shared_markers"] = out["label"]

    # ---- plate mixing within condition
    mix_rows = []
    plates_joint = joint.obs["plate"].astype(str).to_numpy()
    for variant in lib.VARIANTS:
        pca = np.asarray(joint.obsm[f"X_pca__{variant}"], dtype=np.float64)
        for cond, own in (("Ctrl", ~is_zerok), ("ZeroK", is_zerok)):
            pm = t21.mixing(pca[own][:, :t21.N_PCS], plates_joint[own], list(PLATES))
            pm.insert(0, "embedding", f"dct_intake joint X_pca__{variant} (cells of the condition)")
            pm.insert(1, "condition", cond)
            mix_rows.append(pm)
    for cond in lib.CONDITIONS:
        pca = np.asarray(objs[cond].obsm["X_pca_neighbors"], dtype=np.float64)
        pm = t21.mixing(pca[:, :t21.N_PCS], objs[cond].obs["plate"].astype(str).to_numpy(), list(PLATES))
        pm.insert(0, "embedding", "own step-1 graph PCA (X_pca_neighbors, 20 PCs)")
        pm.insert(1, "condition", cond)
        mix_rows.append(pm)
    mixing = pd.concat(mix_rows, ignore_index=True)
    for row in mixing[mixing["label"] == "all_cells"].itertuples(index=False):
        checks.add(f"plate mixing, {row.condition}, {row.embedding}: same-plate fraction vs expected (p)",
                   f"{row.observed_same_label_fraction:.3f} vs {row.expected_random:.3f} (p {row.p_empirical_ge:.3g})",
                   None)

    # ---- agreement
    agree_rows, crosstabs = [], []
    base = frame[f"transfer__{lib.BASELINE_VARIANT}"].to_numpy()
    for variant in lib.VARIANTS:
        if variant != lib.BASELINE_VARIANT:
            agree_rows.append({"comparison": f"transfer {lib.BASELINE_VARIANT} vs transfer {variant}",
                               **lib.agreement(base, frame[f"transfer__{variant}"])})
    for variant in lib.VARIANTS:
        agree_rows.append({"comparison": f"transfer {variant} vs pipeline",
                           **lib.agreement(frame[f"transfer__{variant}"], frame["cs_zerok_pipeline_label"])})
        crosstabs.extend(lib.crosstab_long(frame[f"transfer__{variant}"], frame["cs_zerok_pipeline_label"],
                                           "transfer_label", "pipeline_label", variant=variant))
    agree_rows.append({"comparison": "pipeline vs pipeline (shared markers only)",
                       **lib.agreement(frame["cs_zerok_pipeline_label"], frame["cs_zerok_pipeline_label__shared_markers"])})
    agree = pd.DataFrame(agree_rows)
    for row in agree.itertuples(index=False):
        checks.add(f"{row.comparison}: fraction equal (ARI)", f"{row.fraction_equal:.3f} ({row.ari:.3f})", None)
    count_cols = [*[f"transfer__{v}" for v in lib.VARIANTS], "cs_zerok_pipeline_label",
                  "cs_zerok_pipeline_label__shared_markers"]
    counts = pd.concat([pd.DataFrame({"column": col, "label": frame[col].value_counts().index,
                                      "n_cells": frame[col].value_counts().to_numpy()}) for col in count_cols],
                       ignore_index=True)
    conflicts = frame.loc[frame["cs_zerok_transfer_label"] != frame["cs_zerok_pipeline_label"],
                          ["cell_id", "cell_key", "plate", "n_detected", *[f"transfer__{v}" for v in lib.VARIANTS],
                           "cs_zerok_transfer_confidence", "cs_zerok_pipeline_cluster", "cs_zerok_pipeline_label",
                           "cs_zerok_pipeline_confidence", "cs_zerok_pipeline_seed_stability"]]
    checks.add("ZeroK cells with conflicting calls (transfer 'none' vs pipeline)", f"{len(conflicts)}", None)
    for label in lib.CORE_TYPES:
        both = int(((frame["cs_zerok_transfer_label"] == label) & (frame["cs_zerok_pipeline_label"] == label)).sum())
        checks.add(f"ZeroK cells labelled {label} by both routes", f"{both}", None)

    run.table(frame, "transfer_labels_per_cell")
    run.table(pd.DataFrame(self_rows), "transfer_self_check")
    run.table(agree, "transfer_agreement")
    run.table(pd.DataFrame(crosstabs), "transfer_crosstabs")
    run.table(counts, "transfer_label_counts")
    run.table(conflicts, "transfer_conflicts")
    run.table(stab, "zerok_pipeline_seed_stability")
    run.table(mixing, "plate_mixing")
    run.table(pd.DataFrame(shared_rows), "shared_marker_variant")
    run.table(checks.frame(), "transfer_checks")
    fig_labels(run, frame, np.asarray(joint.obsm[f"X_umap__{lib.BASELINE_VARIANT}"])[is_zerok])
    fig_agreement(run, agree, counts)
    run.finish({"transfer_k": lib.TRANSFER_K, "transfer_tie_rule": "nearest tied neighbour (C2)",
                "embedding": "dct_intake joint X_pca__<variant> (z-score over all cells, arpack PCA, 20 PCs)",
                "reference": "cs_ctrl_deg_label (step 2)", "variants": list(lib.VARIANTS),
                "plate_mixing": "zerok_core_lib.mixing (k = 10, 1,000 label permutations)",
                "seed_stability": "step-2 scan, Leiden 1.0 rank-free, seeds 0-9 vs 42",
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_labels_transfer_checks.csv.")


if __name__ == "__main__":
    main()
