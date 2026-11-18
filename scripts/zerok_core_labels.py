#!/usr/bin/env python
"""zerok_core step 4: zero-K cell labels by two independent routes.



Label names are the labels_core primary column's (DCT1, DCT2, ProLIF, Unknown). No protein-level
or label-fraction comparison between conditions is made here; that is zerok_response_core.

  (a) transfer   each ZeroK cell takes the majority ctrl_deg_label of its 10 nearest
                 CONTROL cells in the joint 20-PC embedding, once per normalization
                 variant; the vote fraction is the confidence. Ties go to the label of
                 the nearest tied neighbour (authors default C2, 28 Oct 2026).
  (b) pipeline   the labels_core primary pipeline on the ZeroK cells alone, with their own
                 scaling: submitted graph settings (k = 10, 20 PCs, Scanpy PCA
                 fallback), Leiden 1.0 (igraph, seed 42), the cluster_de DE rule with the kidney_annotation
                 rank-free rescue, DCT V1 markers present in the ZeroK matrix, Slc12a3
                 and Umod treated as shared, minimum cluster size 5.

The same pipeline code is run on the CONTROL cells as a check: it must return all 727
labels_core ctrl_deg_cluster and ctrl_deg_label calls unchanged.

Also reported (authors default C4): control leave-one-out self-transfer accuracy, ZeroK
pipeline stability over Leiden seeds 0-9, and a marker-sensitivity variant that keeps
only the markers present in BOTH conditions (default C3).

Inputs (read-only): results/h5ad/zerok_core_DCT_joint_norm.h5ad (step 3);
results/h5ad/intake_core_DCT_zerok.h5ad and intake_core_DCT_ctrl.h5ad;
results/tables/labels_core/labels_core_dct_ctrl_canonical_labels.csv;
data/DCT/DCT_markers_V1.xlsx.
Outputs: tables/ and results/figures/zerok_core/ and
results/h5ad/zerok_core_DCT_zerok_pipeline.h5ad.

The primary ZeroK column is the authors' choice after this step; step 5 writes dct_label.

Command (project root):  python scripts/zerok_core_labels.py
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
from sklearn.metrics import adjusted_rand_score

import scp_common as cc
import qc_mito_lib as t3
import cluster_de_lib as t5
import intake_core_lib as t17
import labels_core_lib as t18
import zerok_core_lib as lib
from kidney_annotation_rankfree_sensitivity import rank_free
from labels_core_resolution_scan import rule_tables

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "zerok_core_labels.py"
RESOLUTION = 1.0
RULE = "rankfree"
LABEL_ORDER = list(cc.DCT_LABEL_ORDER)        # DCT1, DCT2, ProLIF, Unknown
TRANSFER_K = 10
STABILITY_SEEDS = tuple(range(10))            # against seed 42, as labels_core


# ---------------------------------------------------------------------------
# (a) transfer from the control cells
# ---------------------------------------------------------------------------
def transfer_labels(pca: np.ndarray, is_zerok: np.ndarray, ctrl_labels: np.ndarray, k: int = TRANSFER_K):
    """Majority ctrl label among the k nearest CONTROL cells; ties -> nearest tied neighbour."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(pca[~is_zerok])
    dist, idx = nn.kneighbors(pca[is_zerok])
    neighbour_labels = np.asarray(ctrl_labels)[idx]
    labels, fractions, tied = [], [], []
    for row, d in zip(neighbour_labels, dist):
        counts = pd.Series(row).value_counts()
        top = counts.max()
        winners = [lab for lab, n in counts.items() if n == top]
        if len(winners) == 1:
            labels.append(winners[0])
        else:
            nearest = min(winners, key=lambda lab: d[np.argmax(row == lab)])
            labels.append(nearest)
        fractions.append(top / k)
        tied.append(len(winners) > 1)
    return np.array(labels, dtype=object), np.array(fractions), np.array(tied), idx, dist


def within_distance(points: np.ndarray, k: int = TRANSFER_K) -> np.ndarray:
    """Mean distance to the k nearest OTHER points of the same set (self excluded)."""
    from sklearn.neighbors import NearestNeighbors

    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(points).kneighbors(points)
    return np.vstack([d[row != i][:k] for i, (row, d) in enumerate(zip(idx, dist))]).mean(axis=1)


def self_transfer(pca: np.ndarray, is_zerok: np.ndarray, ctrl_labels: np.ndarray, k: int = TRANSFER_K):
    """Leave-one-out: every control cell labelled from its k nearest OTHER control cells."""
    from sklearn.neighbors import NearestNeighbors

    ctrl = pca[~is_zerok]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(ctrl)
    dist, idx = nn.kneighbors(ctrl)
    idx = np.vstack([row[row != i][:k] for i, row in enumerate(idx)])
    dist = np.vstack([d[row != i][:k] for i, (row, d) in enumerate(zip(nn.kneighbors(ctrl)[1], dist))])
    neighbour_labels = np.asarray(ctrl_labels)[idx]
    out = []
    for row, d in zip(neighbour_labels, dist):
        counts = pd.Series(row).value_counts()
        top = counts.max()
        winners = [lab for lab, n in counts.items() if n == top]
        out.append(winners[0] if len(winners) == 1 else min(winners, key=lambda lab: d[np.argmax(row == lab)]))
    return np.array(out, dtype=object)


# ---------------------------------------------------------------------------
# (b) the labels_core primary pipeline on one condition alone
# ---------------------------------------------------------------------------
def pipeline(run: cc.Run, adata17, markers, seed: int = lib.SEED, resolution: float = RESOLUTION,
             cache=None, obj=None):
    """labels_core primary pipeline (submitted graph settings; DE rule with the rank-free rescue)."""
    import scanpy as sc

    if obj is None:
        obj = t18.submitted_object(adata17)
        sc.pp.neighbors(obj, n_neighbors=t18.N_NEIGHBORS, n_pcs=t18.N_PCS)
    var_names = obj.var_names.astype(str)
    intensity = cc.dense(adata17.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    if cache is None:
        cache = t5.rank_cache(np.log1p(intensity), detected)
    x_z = pd.DataFrame(cc.dense(obj.X).astype(np.float64), index=obj.obs_names.astype(str), columns=var_names)
    det_frame = pd.DataFrame(detected, index=obj.obs_names.astype(str), columns=var_names)
    cl = t18.leiden_on(t18.graph_adata(obj), resolution, seed)
    name = t18.clustering_name(resolution, seed)
    de, ev, asg = t18.annotate_clustering(cache, cl, var_names, name, markers, x_z, det_frame)
    rf = rank_free(asg, de, t18.specific_present(markers))
    tab = rule_tables(asg, rf)[RULE]
    label = t18.per_cell(cl, tab, "deg_label").astype(str)
    conf = t18.per_cell(cl, tab, "confidence").astype(str)
    return {"obj": obj, "cache": cache, "clusters": cl, "label": label, "confidence": conf,
            "de": de, "evidence": ev, "assignment": asg, "table": tab}


def cluster_table(res: dict, obs: pd.DataFrame, intensity: np.ndarray, var_names) -> pd.DataFrame:
    """Per-cluster summary: size, depth, plate share, DE counts and the rescue marker."""
    cl, de = res["clusters"], res["de"]
    ids = obs["n_detected"].to_numpy(float)
    rows = []
    for c in t5.sort_clusters(cl):
        mask = cl == c
        sub = de[de["cluster"] == c]
        depth = cc.compare_groups(ids[mask], ids[~mask])
        row = {"cluster": c, "n_cells": int(mask.sum()),
               "median_protein_ids": depth["median_group"], "median_protein_ids_rest": depth["median_reference"],
               "protein_ids_p_vs_rest": depth["p_value"],
               "plate_P1_fraction": float((obs.loc[mask, "plate"] == "P1").mean()),
               "n_tested": int(len(sub)), "n_sig_up": int((sub["up"] & (sub["q_bh"] < t5.Q_DE)).sum())}
        tab = res["table"].set_index("cluster")
        row["zerok_pipeline_label"] = str(tab.loc[c, "deg_label"])
        row["zerok_pipeline_confidence"] = str(tab.loc[c, "confidence"])
        top = sub[sub["in_top"]].sort_values("rank_up")["protein"].tolist()
        row["top10_de"] = ";".join(top[:10])
        rows.append(row)
    return pd.DataFrame(rows)


def agreement(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    return {"n_cells": len(a), "fraction_equal": float((a == b).mean()),
            "n_equal": int((a == b).sum()), "ari": float(adjusted_rand_score(a, b))}


def crosstab(a, b, name_a: str, name_b: str) -> pd.DataFrame:
    tab = pd.crosstab(pd.Series(np.asarray(a).astype(str), name=name_a),
                      pd.Series(np.asarray(b).astype(str), name=name_b))
    return tab.reindex(index=[l for l in LABEL_ORDER if l in tab.index],
                       columns=[l for l in LABEL_ORDER if l in tab.columns]).fillna(0).astype(int).reset_index()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_labels(run, frame: pd.DataFrame, umap: dict[str, np.ndarray], is_zerok: np.ndarray) -> None:
    colors = {**{k: v for k, v in zip(LABEL_ORDER, t3.SERIES)}, "Unknown": cc.UNKNOWN_COLOR}
    columns = [(f"transfer__{v}", f"transfer ({v})") for v in lib.VARIANTS]
    columns.append(("zerok_pipeline_label", "pipeline (ZeroK alone)"))
    fig, axes = plt.subplots(1, len(columns), figsize=(7.4, 2.3))
    xy = umap[f"X_umap__{lib.BASELINE_VARIANT}"][is_zerok]
    for ax, (col, title) in zip(axes, columns):
        t3.umap_categorical(ax, xy, frame[col], LABEL_ORDER, colors, title)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", ms=4, color=colors[l], label=l) for l in LABEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 0.015))
    t3.footer(fig, SCRIPT, "ZeroK cells on the baseline joint UMAP ('none'); label names are the labels_core primary "
                           "column's")
    run.figure(fig, "zerok_labels", "ZeroK labels by transfer (per variant) and by the labels_core pipeline")


def fig_agreement(run, agree: pd.DataFrame, counts: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    sub = agree[agree["comparison"].str.startswith("transfer")]
    axes[0].barh(np.arange(len(sub)), sub["fraction_equal"], color=t3.SERIES[0])
    axes[0].set_yticks(np.arange(len(sub)), sub["comparison"], fontsize=6)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("fraction of ZeroK cells with the same label")
    axes[0].invert_yaxis()
    t3.ygrid(axes[0])

    pivot = counts.pivot(index="label", columns="column", values="n_cells").reindex(LABEL_ORDER).fillna(0)
    x = np.arange(len(pivot.columns))
    bottom = np.zeros(len(pivot.columns))
    label_colors = {**{k: v for k, v in zip(LABEL_ORDER, t3.SERIES)}, "Unknown": cc.UNKNOWN_COLOR}
    for label in pivot.index:
        axes[1].bar(x, pivot.loc[label], 0.6, bottom=bottom, color=label_colors[label], label=label)
        bottom += pivot.loc[label].to_numpy()
    axes[1].set_xticks(x, pivot.columns, fontsize=5.5, rotation=20, ha="right")
    axes[1].set_ylabel("ZeroK cells")
    axes[1].legend(fontsize=5.5, frameon=False, ncol=2)
    t3.ygrid(axes[1])
    t3.footer(fig, SCRIPT, "label counts are within ZeroK only; no comparison with the control is made here")
    run.figure(fig, "zerok_label_agreement", "Agreement between the ZeroK label routes and label counts")


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_core step 4: ZeroK labels.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "labels", args)
    t_start = time.time()
    checks = t5.Checks("dct_zerok")
    import anndata as ad

    joint = ad.read_h5ad(run.use_input(cc.out_path(lib.TASK, "h5ad", "DCT_joint_norm", ".h5ad"),
                                       "zerok_core step 3 joint object (normalized layers and embeddings)"))
    condition = joint.obs["condition"].astype(str).to_numpy()
    is_zerok = condition == "ZeroK"
    joint_cells = joint.obs_names.astype(str).to_numpy()

    ctrl_labels_table = pd.read_csv(run.use_input(
        cc.out_path(t18.TASK, "tables", "dct_ctrl_canonical_labels", ".csv"), "labels_core canonical control labels"),
        dtype={"cell_id": str})
    ctrl_cells = joint_cells[~is_zerok]
    table = ctrl_labels_table.set_index("cell_id").reindex(ctrl_cells)
    checks.add("labels_core labels found for every control cell of the joint object",
               f"{int(table['ctrl_deg_label'].notna().sum())}/{len(ctrl_cells)}",
               bool(table["ctrl_deg_label"].notna().all()))
    ctrl_deg = table["ctrl_deg_label"].astype(str).to_numpy()
    checks.add("control label counts equal labels_core", t18.counts_text(ctrl_deg, LABEL_ORDER),
               t18.counts_text(ctrl_deg, LABEL_ORDER) == "DCT1 477; DCT2 144; ProLIF 34; Unknown 72",
               "DCT1 477; DCT2 144; ProLIF 34; Unknown 72")

    # ---- (a) transfer, per normalization variant
    zerok_cells = joint_cells[is_zerok]
    frame = pd.DataFrame({"cell_id": zerok_cells,
                          "plate": joint.obs["plate"].astype(str).to_numpy()[is_zerok],
                          "n_detected": joint.obs["n_detected"].to_numpy()[is_zerok]})
    self_rows = []
    for variant in lib.VARIANTS:
        pca = np.asarray(joint.obsm[f"X_pca__{variant}"])
        labels, fractions, tied, idx, dist = transfer_labels(pca, is_zerok, ctrl_deg)
        frame[f"transfer__{variant}"] = labels
        frame[f"transfer_confidence__{variant}"] = fractions
        frame[f"transfer_tied__{variant}"] = tied
        loo = self_transfer(pca, is_zerok, ctrl_deg)
        acc = float((loo == ctrl_deg).mean())
        # How far away are the control cells a ZeroK cell is labelled from? Compared with the within-condition
        # neighbour distance, this says whether transfer interpolates among neighbours or extrapolates.
        d_cross = dist.mean(axis=1)
        d_ctrl = within_distance(pca[~is_zerok])
        d_zerok = within_distance(pca[is_zerok])
        frame[f"transfer_mean_distance__{variant}"] = d_cross
        self_rows.append({"variant": variant, "n_control_cells": len(ctrl_deg),
                          "leave_one_out_accuracy": acc,
                          **{f"accuracy_{l}": float((loo[ctrl_deg == l] == l).mean())
                             for l in LABEL_ORDER if (ctrl_deg == l).any()},
                          "median_distance_zerok_to_ctrl": float(np.median(d_cross)),
                          "median_distance_ctrl_to_ctrl": float(np.median(d_ctrl)),
                          "median_distance_zerok_to_zerok": float(np.median(d_zerok)),
                          "distance_ratio_cross_over_within_ctrl":
                              float(np.median(d_cross) / np.median(d_ctrl))})
        run.note(f"{variant}: control leave-one-out self-transfer accuracy {acc:.3f}; "
                 f"ZeroK transfer counts {t18.counts_text(labels, LABEL_ORDER)}; "
                 f"{int(tied.sum())} cells had a tied vote; "
                 f"median distance to the 10 control cells {np.median(d_cross):.2f} vs "
                 f"{np.median(d_ctrl):.2f} within control "
                 f"(ratio {np.median(d_cross) / np.median(d_ctrl):.2f})")
    frame["zerok_transfer_label"] = frame[f"transfer__{lib.BASELINE_VARIANT}"]
    frame["zerok_transfer_confidence"] = frame[f"transfer_confidence__{lib.BASELINE_VARIANT}"]

    # ---- (b) the labels_core pipeline on ZeroK alone, and the same code on the control cells as a check
    zerok17 = ad.read_h5ad(run.use_input(cc.DCT_NEW_OBJECTS["ZeroK"], "intake_core object (ZeroK)"))
    ctrl17 = ad.read_h5ad(run.use_input(cc.DCT_NEW_OBJECTS["Ctrl"], "intake_core object (Ctrl)"))
    checks.add("ZeroK object cells equal the joint ZeroK cells in order",
               f"{list(zerok17.obs_names.astype(str)) == list(zerok_cells)}",
               list(zerok17.obs_names.astype(str)) == list(zerok_cells))

    ctrl_markers = t18.dct_markers(run, ctrl17.var_names.astype(str))
    ctrl_res = pipeline(run, ctrl17, ctrl_markers)
    same_cluster = int((ctrl_res["clusters"] == table["ctrl_deg_cluster"].astype(str).to_numpy()).sum())
    same_label = int((ctrl_res["label"] == ctrl_deg).sum())
    checks.add("the same pipeline code reproduces labels_core ctrl_deg_cluster", f"{same_cluster}/{len(ctrl_deg)}",
               same_cluster == len(ctrl_deg), "727/727")
    checks.add("the same pipeline code reproduces labels_core ctrl_deg_label", f"{same_label}/{len(ctrl_deg)}",
               same_label == len(ctrl_deg), "727/727")
    same_conf = int((ctrl_res["confidence"] == table["ctrl_deg_confidence"].astype(str).to_numpy()).sum())
    checks.add("the same pipeline code reproduces labels_core ctrl_deg_confidence", f"{same_conf}/{len(ctrl_deg)}",
               same_conf == len(ctrl_deg), "727/727")

    zerok_var = zerok17.var_names.astype(str)
    markers = t18.dct_markers(run, zerok_var)
    present_ctrl = {g for genes in ctrl_markers.present.values() for g in genes}
    present_zerok = {g for genes in markers.present.values() for g in genes}
    only_zerok = sorted(present_zerok - present_ctrl)
    checks.add("workbook markers present in ZeroK but not in Ctrl (dropped in the shared-marker variant)",
               ";".join(only_zerok) or "(none)", None)
    checks.add("workbook markers present in Ctrl but not in ZeroK", ";".join(sorted(present_ctrl - present_zerok))
               or "(none)", None)

    res = pipeline(run, zerok17, markers)
    frame["zerok_pipeline_cluster"] = res["clusters"]
    frame["zerok_pipeline_label"] = res["label"]
    frame["zerok_pipeline_confidence"] = res["confidence"]
    run.note(f"ZeroK pipeline: {len(set(res['clusters']))} Leiden {RESOLUTION:g} clusters; "
             f"{t18.counts_text(res['label'], LABEL_ORDER)}")

    # marker sensitivity (C3): keep only markers present in both conditions
    shared_markers = t18.drop_markers(markers, only_zerok)
    res_shared = pipeline(run, zerok17, shared_markers, obj=res["obj"], cache=res["cache"])
    frame["zerok_pipeline_label__shared_markers"] = res_shared["label"]
    shared_same = int((res_shared["label"] == res["label"]).sum())
    checks.add("shared-marker variant agrees with the ZeroK marker set",
               f"{shared_same}/{len(frame)} cells; {t18.counts_text(res_shared['label'], LABEL_ORDER)}", None)

    # seed stability (C4)
    stability = []
    per_seed = {}
    for seed in STABILITY_SEEDS:
        r = pipeline(run, zerok17, markers, seed=seed, obj=res["obj"], cache=res["cache"])
        per_seed[seed] = r["label"]
        stability.append({"seed": seed, "n_clusters": int(len(set(r["clusters"]))),
                          "ari_clusters_vs_seed42": float(adjusted_rand_score(res["clusters"], r["clusters"])),
                          "ari_labels_vs_seed42": float(adjusted_rand_score(res["label"], r["label"])),
                          "fraction_same_label_as_seed42": float((r["label"] == res["label"]).mean()),
                          "label_counts": t18.counts_text(r["label"], LABEL_ORDER)})
    stability = pd.DataFrame(stability)
    seed_matrix = np.vstack([per_seed[s] for s in STABILITY_SEEDS])
    frame["zerok_pipeline_seed_stability"] = (seed_matrix == res["label"]).mean(axis=0)
    run.note(f"ZeroK pipeline seed stability: median fraction of cells keeping the seed-42 label "
             f"{stability['fraction_same_label_as_seed42'].median():.3f} "
             f"(min {stability['fraction_same_label_as_seed42'].min():.3f})")

    # ---- agreement
    agree_rows, crosstabs = [], []
    base = frame[f"transfer__{lib.BASELINE_VARIANT}"].to_numpy()
    for variant in lib.VARIANTS:
        if variant == lib.BASELINE_VARIANT:
            continue
        agree_rows.append({"comparison": f"transfer {lib.BASELINE_VARIANT} vs transfer {variant}",
                           **agreement(base, frame[f"transfer__{variant}"])})
    for variant in lib.VARIANTS:
        agree_rows.append({"comparison": f"transfer {variant} vs pipeline",
                           **agreement(frame[f"transfer__{variant}"], frame["zerok_pipeline_label"])})
        crosstabs.append(crosstab(frame[f"transfer__{variant}"], frame["zerok_pipeline_label"],
                                  f"transfer__{variant}", "zerok_pipeline_label").assign(
            comparison=f"transfer {variant} vs pipeline"))
    agree_rows.append({"comparison": "pipeline vs pipeline (shared markers only)",
                       **agreement(frame["zerok_pipeline_label"],
                                   frame["zerok_pipeline_label__shared_markers"])})
    agree = pd.DataFrame(agree_rows)
    for row in agree.itertuples(index=False):
        run.note(f"{row.comparison}: {row.fraction_equal:.3f} equal (ARI {row.ari:.3f})")

    counts = pd.concat([
        pd.DataFrame({"column": col, "label": frame[col].value_counts().index,
                      "n_cells": frame[col].value_counts().to_numpy()})
        for col in [*[f"transfer__{v}" for v in lib.VARIANTS], "zerok_pipeline_label",
                    "zerok_pipeline_label__shared_markers"]], ignore_index=True)

    conflicts = frame.loc[frame[f"transfer__{lib.BASELINE_VARIANT}"] != frame["zerok_pipeline_label"],
                          ["cell_id", "plate", "n_detected", *[f"transfer__{v}" for v in lib.VARIANTS],
                           "zerok_transfer_confidence", "zerok_pipeline_cluster", "zerok_pipeline_label",
                           "zerok_pipeline_confidence", "zerok_pipeline_seed_stability"]]

    run.table(frame, "zerok_labels_per_cell")
    run.table(agree, "label_agreement")
    run.table(pd.concat(crosstabs, ignore_index=True), "label_crosstabs")
    run.table(counts, "zerok_label_counts")
    run.table(conflicts, "label_conflicts")
    run.table(cluster_table(res, zerok17.obs.assign(plate=zerok17.obs["plate"].astype(str)),
                            cc.dense(zerok17.layers["intensity"]), zerok_var), "zerok_pipeline_clusters")
    run.table(res["evidence"], "zerok_pipeline_marker_evidence")
    run.table(res["de"], "zerok_pipeline_de_tested")
    run.table(pd.DataFrame(self_rows), "transfer_self_check")
    run.table(stability, "zerok_pipeline_seed_stability")
    run.table(checks.frame(), "labels_checks")

    fig_labels(run, frame, {k: np.asarray(v) for k, v in joint.obsm.items()}, is_zerok)
    fig_agreement(run, agree, counts)

    out = res["obj"].copy()
    out.obs["zerok_pipeline_cluster"] = pd.Categorical(res["clusters"])
    out.obs["zerok_pipeline_label"] = pd.Categorical(res["label"], categories=LABEL_ORDER)
    out.obs["zerok_pipeline_confidence"] = res["confidence"]
    for variant in lib.VARIANTS:
        out.obs[f"transfer__{variant}"] = pd.Categorical(frame[f"transfer__{variant}"].to_numpy(),
                                                         categories=LABEL_ORDER)
    out.uns["zerok_core"] = {"resolution": RESOLUTION, "rule": RULE, "seed": lib.SEED,
                             "min_cluster_size": t18.MIN_CLUSTER_SIZE,
                             "treated_as_shared": t18.TREATED_AS_SHARED,
                             "note": "labels_core primary pipeline on the ZeroK cells alone; the same code reproduces "
                                     "the 727 control calls (checked)."}
    lib.write_h5ad(run, out, "DCT_zerok_pipeline")

    run.finish(parameters={"transfer_k": TRANSFER_K, "transfer_tie_rule": "nearest tied neighbour (authors C2)",
                           "resolution": RESOLUTION, "rule": RULE, "stability_seeds": list(STABILITY_SEEDS),
                           "markers": "DCT V1 present in the ZeroK matrix; shared-marker variant drops "
                                      + (";".join(only_zerok) or "(nothing)"),
                           "primary_column": "not chosen yet; the authors decide before step 5 writes dct_label",
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_core_labels_checks.csv.")


if __name__ == "__main__":
    main()
