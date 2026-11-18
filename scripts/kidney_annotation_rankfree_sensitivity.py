#!/usr/bin/env python
"""kidney_annotation step 2, sensitivity: rank-free rescue of clusters with a type-specific marker beyond the top 25 (kidney).

Requested and agreed by the authors on 14 Oct 2026, after the
scan showed that the top-25 rule leaves clusters Unknown when their type-specific marker is significantly up but
outranked (number of significant up proteins per cluster 4 to > 2,000, following cluster depth). The top-25 rule
stays the primary rule.

Rule (fixed before any rank-free result was seen):
  1. Same DE as the scan (cluster_de_lib; significant up = q < 0.05, AUC > 0.5, positive mean difference; tested =
     detected in >= 20% of the cluster's cells).
  2. Clusters with a high or medium top-25 label keep it; clusters below the minimum size stay unannotated.
  3. A cluster labelled Unknown or from shared markers only by the top-25 rule takes type T if a type-specific
     quantified V3 marker of T is significantly up at any rank (confidence 'specific_any_rank'). Several types:
     highest AUC of the type's best type-specific marker, then more significant type-specific markers, then lower q.
  4. Otherwise the top-25 result stands.
Every clustering of the scan is rebuilt (3 graphs x 12 resolutions x seeds 42, 0-9); the top-25 labels must equal
the scan's per-cell tables (the script stops otherwise).

Inputs (read-only): kidney_object canonical kidney object and labels; data/KIDNEY/KIDNEY_markers_V3.xlsx;
  results/h5ad/kidney_annotation_kidney_scan_graphs.h5ad; kidney_annotation_scan_clusters_per_cell.csv,
  kidney_annotation_scan_seed_clusters_per_cell.csv, kidney_annotation_scan_summary.csv (reproduction check and comparison).
Outputs: results/tables/kidney_annotation/kidney_annotation_rankfree_*.csv;
  results/figures/kidney_annotation/kidney_annotation_rankfree_{summary,types_heatmap}.pdf/.png

Command (project root):  python scripts/kidney_annotation_rankfree_sensitivity.py
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

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import qc_mito_lib as t3
import cluster_de_lib as t5
import kidney_annotation_scan_lib as s
import kidney_annotation_scan_figures as sf
from kidney_annotation_resolution_scan import type_rows

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "kidney_annotation_rankfree_sensitivity.py"
RULES = ("top25", "rankfree")
RULE_DISPLAY = {"top25": "top-25 rule (primary)", "rankfree": "rank-free rescue (sensitivity)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation sensitivity: rank-free rescue by type-specific markers.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "rankfree_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_annotation_rankfree_checks.csv.")


def rank_free(asg: pd.DataFrame, de: pd.DataFrame, specific: dict[str, list[str]]) -> pd.DataFrame:
    """Rank-free labels per cluster from the top-25 assignment and the cluster DE (rule in the module docstring)."""
    sig = de[de["up"] & (de["q_bh"] < t5.Q_DE)]
    rows = []
    for r in asg.itertuples(index=False):
        row = {"cluster": r.cluster, "top25_label": r.deg_label, "top25_confidence": r.confidence,
               "rankfree_label": r.deg_label, "rankfree_confidence": r.confidence, "rescued": False,
               "rescue_candidates": "", "rescue_markers": "", "rescue_best_rank": np.nan, "rescue_best_auc": np.nan,
               "rescue_n_specific_markers_sig": 0}
        if r.confidence in ("high", "medium") or r.deg_label == s.BELOW_MIN:
            rows.append(row)
            continue
        sc = sig[sig["cluster"] == r.cluster]
        cands = []
        for t, genes in specific.items():
            hit = sc[sc["protein"].isin(genes)]
            if len(hit):
                cands.append((t, float(hit["auc"].max()), len(hit), float(hit["q_bh"].min()), hit.sort_values("rank_up")))
        if cands:
            cands.sort(key=lambda c: (-c[1], -c[2], c[3]))
            t, auc, n, _, hit = cands[0]
            row.update(rankfree_label=t, rankfree_confidence=s.RESCUED, rescued=True,
                       rescue_candidates=";".join(f"{c[0]} ({c[1]:.3f})" for c in cands),
                       rescue_markers="; ".join(
                           f"{h.protein} (rank {int(h.rank_up)}; AUC {h.auc:.3f}; q {h.q_bh:.2g}; detected "
                           f"{h.pct_detected_cluster:.2f} vs {h.pct_detected_rest:.2f})" for h in hit.itertuples()),
                       rescue_best_rank=float(hit["rank_up"].min()), rescue_best_auc=auc,
                       rescue_n_specific_markers_sig=n)
        rows.append(row)
    return pd.DataFrame(rows)


def as_assignment(rf: pd.DataFrame, rule: str) -> pd.DataFrame:
    return rf.rename(columns={f"{rule}_label": "deg_label", f"{rule}_confidence": "confidence"})[
        ["cluster", "deg_label", "confidence"]]


def seed_row(graph, res, seed, cl, labs: dict, confs: dict, rf, ref: dict) -> dict:
    row = {"graph": graph, "resolution": res, "seed": seed, "n_clusters": len(set(cl)),
           "n_clusters_rescued": int(rf["rescued"].sum()),
           "n_cells_rescued": int(pd.Series(cl).isin(rf.loc[rf["rescued"], "cluster"]).sum())}
    for rule in RULES:
        types = s.types_resolved(as_assignment(rf, rule))
        row[f"{rule}_n_types"] = len(types)
        row[f"{rule}_n_types_specific"] = sum(c in s.SPECIFIC_CONFIDENCES for c in types.values())
        row[f"{rule}_types"] = ";".join(f"{t}({c})" for t, c in types.items())
        row[f"{rule}_fraction_unknown"] = float((labs[rule] == s.UNKNOWN).mean())
        row[f"{rule}_fraction_same_label_as_seed42"] = float((ref[rule] == labs[rule]).mean())
    return row


def cluster_rows(graph, res, cl, rf, labels, ids) -> pd.DataFrame:
    rows = []
    for r in rf.itertuples(index=False):
        m = cl == r.cluster
        row = {"graph": graph, "resolution": res, "clustering": s.clustering_name(graph, res), **r._asdict(),
               "n_cells": int(m.sum()), "median_protein_ids": float(np.median(ids[m]))}
        for col in s.LABEL_COLUMNS:
            vc = labels.loc[m, col].astype(str).value_counts()
            row[f"majority_{col}"] = vc.index[0]
            row[f"majority_{col}_fraction"] = float(vc.iloc[0] / m.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summary_figure(run: cc.Run, summ: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(180 * mm, 68 * mm))
    panels = (("n_types", "A  Types resolved (any confidence)", "distinct cell types"),
              ("n_types_specific", "B  Types with a type-specific marker", "distinct cell types"),
              ("fraction_unknown", "C  Fraction of cells Unknown", "fraction of 712 cells"))
    for ax, (key, title, ylab) in zip(axes, panels):
        ax.axvspan(0.2, 2.25, color=t3.INK["grid"], alpha=0.45, linewidth=0, zorder=0)
        sf.series(ax, summ, f"top25_{key}", linestyle=(0, (3, 1.5)))
        lo = f"rankfree_{key}_all_seeds_min" if key != "fraction_unknown" else None
        hi = f"rankfree_{key}_all_seeds_max" if key != "fraction_unknown" else None
        sf.series(ax, summ, f"rankfree_{key}", lo, hi)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if key == "fraction_unknown":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 14.5)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            sf.reference_line(ax, 14, "Fig 2B: 14 types", left=True, text_y=14.1)
        ax.text(1.22, 0.93 if key == "fraction_unknown" else 0.3, "res <= 2.0", ha="center", va="bottom", fontsize=5.5,
                color=t3.INK["secondary"])
    handles = [Line2D([0], [0], color=sf.GRAPH_COLORS[g], marker=sf.GRAPH_MARKERS[g], markersize=3.5, linewidth=1.2,
                      markeredgewidth=0, label=s.graph_display(g)) for g in s.GRAPH_NAMES]
    style = [Line2D([0], [0], color=t3.INK["secondary"], linewidth=1.2, label="rank-free rescue (sensitivity); bars: "
                    "seeds 0-9, 42"),
             Line2D([0], [0], color=t3.INK["secondary"], linewidth=1.2, linestyle=(0, (3, 1.5)),
                    label="top-25 rule (primary)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, fontsize=6.2)
    fig.legend(handles=style, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=2, fontsize=5.8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.84), w_pad=1.0)
    t3.footer(fig, SCRIPT, "kidney, 712 cells; seed 42 unless stated; shaded: resolution <= 2.0 (submitted and lower)")
    run.figure(fig, "rankfree_summary", "kidney_annotation rank-free sensitivity vs top-25 rule against resolution")


def main() -> None:
    args = parse_args()
    run = cc.Run(s.TASK, "rankfree_sensitivity", args)
    t3.apply_style()
    t_start = time.time()
    adata, labels = t5.load(run, s.DATASET)
    store = s.load_graphs(run)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    ids = labels["n_detected_proteins"].to_numpy(float)
    x_z = pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    markers = t5.read_markers(run, s.DATASET, var_names)
    specific = {t: markers.specific_present(t) for t in cc.ordered_labels(markers.lists, s.ORDER)
                if markers.specific_present(t)}
    cache = t5.rank_cache(np.log1p(intensity), detected)
    scan42 = s.read_table(run, "scan_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    scan_seeds = s.read_table(run, "scan_seed_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    scan_summary = s.read_table(run, "scan_summary")
    checks = t5.Checks(s.DATASET)
    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in
                                                               t5.EXPECTED_COUNTS[col]),
                   observed == t5.EXPECTED_COUNTS[col])
    checks.add("types with a type-specific quantified marker (rescue candidates)",
               "; ".join(f"{t}: {','.join(g)}" for t, g in specific.items()), None)
    if not checks.ok:
        stop(run, checks)

    per_cell = labels[[*s.LABEL_COLUMNS, "n_detected_proteins"]].copy()
    seed_rows, clus_all, type_all, agree_rows, recov_rows = [], [], [], [], []
    mism = {"clusters": 0, "top25_labels": 0, "top25_confidence": 0}
    for graph in s.GRAPH_NAMES:
        g = s.graph_adata(store, graph)
        for res in s.RESOLUTIONS:
            t0 = time.time()
            ref: dict = {}
            per_seed_types = {rule: [] for rule in RULES}
            for seed in s.SEEDS:
                name = s.clustering_name(graph, res, seed)
                cl = t5.leiden(g, res, "t15_rf_tmp", seed=seed).to_numpy().astype(str)
                de, _, asg = s.annotate_clustering(cache, cl, var_names, name, markers, x_z, det_frame)
                rf = rank_free(asg, de, specific)
                labs = {rule: s.per_cell_labels(cl, as_assignment(rf, rule)).astype(str) for rule in RULES}
                confs = {rule: s.per_cell_labels(cl, as_assignment(rf, rule), "confidence").astype(str)
                         for rule in RULES}
                src = scan42 if seed == s.SEED else scan_seeds
                mism["clusters"] += int((src[f"{name}__cluster"].to_numpy() != cl).sum())
                mism["top25_labels"] += int((src[f"{name}__label"].to_numpy() != labs["top25"]).sum())
                if seed == s.SEED:
                    mism["top25_confidence"] += int((src[f"{name}__confidence"].to_numpy() != confs["top25"]).sum())
                    ref = labs
                    per_cell[f"{name}__cluster"] = cl
                    for rule in RULES:
                        per_cell[f"{name}__{rule}_label"] = labs[rule]
                        per_cell[f"{name}__{rule}_confidence"] = confs[rule]
                    clus_all.append(cluster_rows(graph, res, cl, rf, labels, ids))
                    rf42 = rf
                    for col in s.LABEL_COLUMNS:
                        truth = labels[col].astype(str).to_numpy()
                        for rule in RULES:
                            a = t5.agreement(labs[rule], truth)
                            agree_rows.append({"graph": graph, "resolution": res, "label_column": col, "rule": rule,
                                               "ari": a["ari"], "nmi": a["nmi"],
                                               "accuracy": float((labs[rule] == truth).mean())})
                            rec = t5.recovery(labs[rule], truth, s.KIDNEY_ANNOTATION_ORDER)
                            rec.insert(0, "rule", rule)
                            rec.insert(0, "label_column", col)
                            rec.insert(0, "resolution", res)
                            rec.insert(0, "graph", graph)
                            recov_rows.append(rec)
                for rule in RULES:
                    per_seed_types[rule].append(s.types_resolved(as_assignment(rf, rule)))
                seed_rows.append(seed_row(graph, res, seed, cl, labs, confs, rf, ref))
            cl42 = per_cell[f"{s.clustering_name(graph, res)}__cluster"].to_numpy()
            for rule in RULES:
                for r in type_rows(graph, res, as_assignment(rf42, rule), cl42, per_seed_types[rule]):
                    type_all.append({"rule": rule, **r})
            r42 = seed_rows[-len(s.SEEDS)]
            print(f"[kidney] {graph} r{res:.1f}: top25 {r42['top25_n_types']} types, rank-free {r42['rankfree_n_types']} "
                  f"({r42['n_clusters_rescued']} clusters rescued); {time.time() - t0:.1f} s", flush=True)
    for k, v in mism.items():
        checks.add(f"rebuilt clusterings reproduce the scan per-cell tables: {k} (all graphs, resolutions"
                   f"{', seed 42' if k == 'top25_confidence' else ', seeds'})", f"{v} mismatching cell entries", v == 0, "= 0")
    if not checks.ok:
        stop(run, checks)

    # Summary per graph x resolution -----------------------------------------------------------------
    seeds = pd.DataFrame(seed_rows)
    rows = []
    agree = pd.DataFrame(agree_rows)
    for (graph, res), grp in seeds.groupby(["graph", "resolution"], sort=False):
        r42 = grp[grp["seed"] == s.SEED].iloc[0]
        other = grp[grp["seed"] != s.SEED]
        row = {"graph": graph, "display": s.graph_display(graph), "resolution": res, "n_clusters": int(r42["n_clusters"]),
               "n_clusters_rescued": int(r42["n_clusters_rescued"]), "n_cells_rescued": int(r42["n_cells_rescued"])}
        for rule in RULES:
            for key in ("n_types", "n_types_specific", "types", "fraction_unknown"):
                row[f"{rule}_{key}"] = r42[f"{rule}_{key}"]
            for key in ("n_types", "n_types_specific"):
                row[f"{rule}_{key}_all_seeds_min"] = int(grp[f"{rule}_{key}"].min())
                row[f"{rule}_{key}_all_seeds_max"] = int(grp[f"{rule}_{key}"].max())
            row[f"{rule}_fraction_same_label_as_seed42_median"] = float(other[f"{rule}_fraction_same_label_as_seed42"].median())
            for col, short in (("kidney_reannot_label", "reannot"), ("kidney_pre_label", "pre")):
                a = agree[(agree["graph"] == graph) & (agree["resolution"] == res) & (agree["label_column"] == col)
                          & (agree["rule"] == rule)].iloc[0]
                row[f"{rule}_ari_vs_{short}"] = a["ari"]
        sm = scan_summary[(scan_summary["graph"] == graph) & (scan_summary["resolution"] == res)].iloc[0]
        for key in ("eta2_protein_ids", "ari_clusters_vs_seed42_median"):
            row[key] = sm[key]
        rows.append(row)
    summ = pd.DataFrame(rows)
    both = summ.merge(scan_summary[["graph", "resolution", "n_types_resolved", "n_types_resolved_ge_medium"]],
                      on=["graph", "resolution"], how="outer", validate="one_to_one")
    d = int(max((both["top25_n_types"] - both["n_types_resolved"]).abs().max(),
                (both["top25_n_types_specific"] - both["n_types_resolved_ge_medium"]).abs().max()))
    checks.add("top-25 types resolved (any; >= medium) per graph x resolution equal the scan summary",
               f"max abs diff {d} over {len(both)} rows", d == 0 and len(both) == len(summ), "= 0")
    for row in summ[summ["resolution"] <= 2.0].itertuples(index=False):
        checks.add(f"{row.graph} r{row.resolution}: types top-25 -> rank-free (with type-specific marker); Unknown",
                   f"{row.top25_n_types} ({row.top25_n_types_specific}) -> {row.rankfree_n_types} "
                   f"({row.rankfree_n_types_specific}); {row.top25_fraction_unknown:.2f} -> {row.rankfree_fraction_unknown:.2f}",
                   None)
    if not checks.ok:
        stop(run, checks)

    clus_all = pd.concat(clus_all, ignore_index=True)
    types_df = pd.DataFrame(type_all)
    run.table(checks.frame(), "rankfree_checks")
    run.table(summ, "rankfree_summary")
    run.table(seeds, "rankfree_seed_stability")
    run.table(clus_all, "rankfree_cluster_annotation")
    run.table(types_df, "rankfree_types_by_resolution")
    run.table(agree, "rankfree_agreement")
    run.table(pd.concat(recov_rows, ignore_index=True), "rankfree_recovery")
    run.table(per_cell, "rankfree_clusters_per_cell", index=True)

    # Figures -----------------------------------------------------------------------------------
    summary_figure(run, summ)
    counts = {"reannot": labels["kidney_reannot_label"].value_counts().to_dict(),
              "pre": labels["kidney_pre_label"].value_counts().to_dict()}
    rft = types_df[types_df["rule"] == "rankfree"].drop(columns="rule")
    sf.heatmap_figure(run, rft, counts, name="rankfree_types_heatmap",
                      note=("Rank-free sensitivity. Text = seed-42 confidence of the type's best cluster: H high, M medium "
                            "(type-specific marker in the top 25), R rescued (type-specific marker significant beyond the "
                            "top 25), L shared markers only; a number = clusters with that label. Row label: cells in "
                            "Fig 2B / Fig 2A."),
                      description="kidney_annotation rank-free sensitivity: cell types resolved per resolution and graph",
                      script=SCRIPT)
    run.finish({"rule": __doc__.split("Rule (fixed before any rank-free result was seen):")[1].split("Every clustering")[0]
                .strip(), "specific_markers": specific, "graphs": list(s.GRAPH_NAMES),
                "resolutions": list(s.RESOLUTIONS), "seeds": list(s.SEEDS), "min_cluster_size": s.MIN_CLUSTER_SIZE,
                "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "runtime_s": round(time.time() - t_start, 1),
                "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
