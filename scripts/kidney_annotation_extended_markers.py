#!/usr/bin/env python
"""kidney_annotation step 3b: cluster annotation with atlas-extended marker sets (kidney; option 2, authors 14 Oct 2026).

Declared deviation from the 'kidney V3 markers only' rule;
the V3-only results stay the reference.

Marker sets (markers from kidney_annotation_atlas_reference.py; strict shared rule within each set, as cluster_de):
  v3        - V3 workbook only (re-run here as a check: it must reproduce the scan and rank-free tables);
  atlas     - atlas-derived markers only (types = mapped atlas types; DCT is one type; no JG);
  v3_atlas  - V3 plus atlas markers per type. DCT1 and DCT2 are merged into one type 'DCT' (V3 DCT1 + V3 DCT2 + atlas
              DCT), because the atlas does not split them; atlas-only types (endothelial, stromal, immune, PEC) are
              added.
Clusterings: the scan's graphs (a), (b), (c) at resolution 0.5, 1.0, 1.5, 2.0 (the authors' range), seeds 42 and 0-9.
Rules: the top-25 rule (cluster_de_lib, primary) and the rank-free rescue (kidney_annotation_rankfree_sensitivity.rank_free),
each with the marker set's own type-specific markers. Minimum cluster size 5, as the scan.

Inputs (read-only): kidney_object canonical kidney object and labels; data/KIDNEY/KIDNEY_markers_V3.xlsx; kidney_annotation graphs h5ad;
  kidney_annotation_atlas_markers.csv; kidney_annotation_scan_clusters_per_cell.csv, kidney_annotation_scan_seed_clusters_per_cell.csv,
  kidney_annotation_rankfree_clusters_per_cell.csv (reproduction checks).
Outputs: results/tables/kidney_annotation/kidney_annotation_extmarkers_*.csv;
  results/figures/kidney_annotation/kidney_annotation_extmarkers_*.pdf/.png

Command (project root):  python scripts/kidney_annotation_extended_markers.py
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
import kidney_annotation_scan_lib as s
import kidney_annotation_scan_figures as sf
from kidney_annotation_rankfree_sensitivity import as_assignment, cluster_rows, rank_free
from kidney_annotation_resolution_scan import type_rows

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "kidney_annotation_extended_markers.py"
RESOLUTIONS = (0.5, 1.0, 1.5, 2.0)
SETS = ("v3", "atlas", "v3_atlas")
SET_DISPLAY = {"v3": "V3 only", "atlas": "atlas only", "v3_atlas": "V3 + atlas"}
RULES = ("top25", "rankfree")
EXT_ORDER = ("PCT", "PST", "DTL", "ATL", "TAL", "MD", "DCT1", "DCT2", "DCT", "CNT", "PC", "IC-A", "IC-B", "DVR", "AVR",
             "JG", "Podo", "PEC", "Endo", "MC", "Fib", "Per", "T lymph", "B lymph", "NK", "Macro", "DC", "Neutro")
SET_COLORS = {"v3": t3.SERIES[3], "atlas": t3.SERIES[4], "v3_atlas": t3.SERIES[6]}
SET_MARKERS = {"v3": "o", "atlas": "s", "v3_atlas": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation step 3b: annotation with atlas-extended marker sets.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "extmarkers_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_annotation_extmarkers_checks.csv.")


def marker_set(lists: dict[str, list[str]], var_names) -> t5.MarkerSets:
    var = set(map(str, var_names))
    lists = {t: list(dict.fromkeys(g)) for t, g in lists.items() if g}
    present = {t: [g for g in genes if g in var] for t, genes in lists.items()}
    by_gene: dict[str, set[str]] = {}
    for t, genes in lists.items():
        for g in genes:
            by_gene.setdefault(g, set()).add(t)
    return t5.MarkerSets(lists, present, {t: t for t in lists}, by_gene)


def build_sets(v3: t5.MarkerSets, atlas: pd.DataFrame, var_names) -> dict[str, t5.MarkerSets]:
    a_lists = {t: list(g["protein"]) for t, g in atlas.groupby("scp_type", sort=False)}
    union: dict[str, list[str]] = {}
    for t, genes in v3.lists.items():
        key = "DCT" if t in ("DCT1", "DCT2") else t
        union.setdefault(key, []).extend(genes)
    for t, genes in a_lists.items():
        union.setdefault(t, []).extend(genes)
    return {"v3": v3, "atlas": marker_set(a_lists, var_names), "v3_atlas": marker_set(union, var_names)}


def specific_of(ms: t5.MarkerSets) -> dict[str, list[str]]:
    return {t: ms.specific_present(t) for t in cc.ordered_labels(ms.lists, EXT_ORDER) if ms.specific_present(t)}


def set_table(sets: dict[str, t5.MarkerSets]) -> pd.DataFrame:
    rows = []
    for name, ms in sets.items():
        for t in cc.ordered_labels(ms.lists, EXT_ORDER):
            for g in ms.lists[t]:
                rows.append({"marker_set": name, "cell_type": t, "marker": g, "quantified": g in ms.present[t],
                             "n_types_listing": len(ms.types_by_gene[g]), "shared": ms.is_shared(g),
                             "types_listing": ";".join(sorted(ms.types_by_gene[g]))})
    return pd.DataFrame(rows)


def summary_figure(run: cc.Run, summ: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 3, figsize=(180 * mm, 66 * mm), sharey=True)
    for ax, g in zip(axes, s.GRAPH_NAMES):
        for k, ms in enumerate(SETS):
            for rule, ls in (("rankfree", "-"), ("top25", (0, (3, 1.5)))):
                sub = summ[(summ["graph"] == g) & (summ["marker_set"] == ms) & (summ["rule"] == rule)].sort_values("resolution")
                x = sub["resolution"].to_numpy(float) + (k - 1) * 0.04
                if rule == "rankfree":
                    ax.vlines(x, sub["n_types_all_seeds_min"], sub["n_types_all_seeds_max"], color=SET_COLORS[ms],
                              linewidth=0.6, alpha=0.55)
                ax.plot(x, sub["n_types"], color=SET_COLORS[ms], linestyle=ls, linewidth=1.2, marker=SET_MARKERS[ms],
                        markersize=3, markeredgewidth=0, markerfacecolor=SET_COLORS[ms] if ls == "-" else "white",
                        markeredgecolor=SET_COLORS[ms])
        ax.axhline(9, color=t3.INK["muted"], linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
        ax.text(0.42, 9.1, "Fig 2A: 9 types", fontsize=5.5, color=t3.INK["secondary"], va="bottom")
        ax.set_xticks(RESOLUTIONS)
        ax.set_xlim(0.35, 2.15)
        ax.set_ylim(0, 12)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(s.graph_display(g), fontsize=6.8)
        ax.set_xlabel("Leiden resolution")
        t3.ygrid(ax)
    axes[0].set_ylabel("distinct cell types (seed 42)")
    handles = [Line2D([0], [0], color=SET_COLORS[m], marker=SET_MARKERS[m], markersize=3.5, linewidth=1.2,
                      markeredgewidth=0, label=SET_DISPLAY[m]) for m in SETS]
    style = [Line2D([0], [0], color=t3.INK["secondary"], linewidth=1.2, label="rank-free rescue; bars: seeds 0-9, 42"),
             Line2D([0], [0], color=t3.INK["secondary"], linewidth=1.2, linestyle=(0, (3, 1.5)), label="top-25 rule")]
    fig.legend(handles=handles + style, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=5, fontsize=5.8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9), w_pad=0.8)
    t3.footer(fig, SCRIPT, "kidney, 712 cells; marker sets: V3 workbook, atlas-derived (mka.h5ad), both; DCT1/DCT2 "
                           "merged in V3 + atlas")
    run.figure(fig, "extmarkers_summary", "kidney_annotation types resolved with V3, atlas and V3 + atlas marker sets")


def main() -> None:
    args = parse_args()
    run = cc.Run(s.TASK, "extended_markers", args)
    t3.apply_style()
    t_start = time.time()
    adata, labels = t5.load(run, s.DATASET)
    store = s.load_graphs(run)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    x_z = pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=cell_ids, columns=var_names)
    det_frame = pd.DataFrame(detected, index=cell_ids, columns=var_names)
    cache = t5.rank_cache(np.log1p(intensity), detected)
    v3 = t5.read_markers(run, s.DATASET, var_names)
    atlas = s.read_table(run, "atlas_markers", keep_default_na=False)
    sets = build_sets(v3, atlas, var_names)
    specific = {k: specific_of(ms) for k, ms in sets.items()}
    scan42 = s.read_table(run, "scan_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    seeds_tab = s.read_table(run, "scan_seed_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    rf42 = s.read_table(run, "rankfree_clusters_per_cell", dtype=str).set_index("cell_id").reindex(cell_ids)
    checks = t5.Checks(s.DATASET)
    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {observed.get(k, 0)}" for k in
                                                               t5.EXPECTED_COUNTS[col]),
                   observed == t5.EXPECTED_COUNTS[col])
    missing = sorted(set(atlas["protein"]) - set(var_names))
    checks.add("every atlas marker is a quantified protein", ", ".join(missing) or "none", not missing)
    for k, sp in specific.items():
        checks.add(f"marker set {k}: types with >= 1 type-specific quantified marker",
                   "; ".join(f"{t} {len(g)}" for t, g in sp.items()), None)
    if not checks.ok:
        stop(run, checks)

    per_cell = labels[[*s.LABEL_COLUMNS, "n_detected_proteins"]].copy()
    ids = labels["n_detected_proteins"].to_numpy(float)
    seed_rows, clus_all, type_all, agree_rows, recov_rows = [], [], [], [], []
    mism = {"clusters": 0, "v3_top25_seed42": 0, "v3_rankfree_seed42": 0, "v3_top25_seeds": 0}
    for graph in s.GRAPH_NAMES:
        g = s.graph_adata(store, graph)
        for res in RESOLUTIONS:
            t0 = time.time()
            per_seed_types = {(m, r): [] for m in SETS for r in RULES}
            ref = {}
            rf_42 = {}
            for seed in s.SEEDS:
                name = s.clustering_name(graph, res, seed)
                cl = t5.leiden(g, res, "t15_ext_tmp", seed=seed).to_numpy().astype(str)
                src = scan42 if seed == s.SEED else seeds_tab
                mism["clusters"] += int((src[f"{name}__cluster"].to_numpy() != cl).sum())
                de_all = t5.de_one_vs_rest(cache, cl, var_names, name)
                row = {"graph": graph, "resolution": res, "seed": seed, "n_clusters": len(set(cl))}
                for m in SETS:
                    de, _, asg = s.annotate_clustering(cache, cl, var_names, name, sets[m], x_z, det_frame, de=de_all)
                    rf = rank_free(asg, de, specific[m])
                    rf["top25_overlap"] = rf["cluster"].map(asg.set_index("cluster")["overlap"]).fillna("")
                    labs = {r: s.per_cell_labels(cl, as_assignment(rf, r)).astype(str) for r in RULES}
                    if m == "v3":
                        if seed == s.SEED:
                            mism["v3_top25_seed42"] += int((rf42[f"{name}__top25_label"].to_numpy() != labs["top25"]).sum())
                            mism["v3_rankfree_seed42"] += int((rf42[f"{name}__rankfree_label"].to_numpy()
                                                               != labs["rankfree"]).sum())
                        else:
                            mism["v3_top25_seeds"] += int((src[f"{name}__label"].to_numpy() != labs["top25"]).sum())
                    if seed == s.SEED:
                        ref[m] = labs
                        rf_42[m] = rf
                        per_cell[f"{name}__cluster"] = cl
                        for r in RULES:
                            per_cell[f"{name}__{m}__{r}_label"] = labs[r]
                        cr = cluster_rows(graph, res, cl, rf, labels, ids)
                        cr.insert(3, "marker_set", m)
                        clus_all.append(cr)
                        for col in s.LABEL_COLUMNS:
                            truth = labels[col].astype(str).to_numpy()
                            for r in RULES:
                                a = t5.agreement(labs[r], truth)
                                agree_rows.append({"graph": graph, "resolution": res, "marker_set": m, "rule": r,
                                                   "label_column": col, "ari": a["ari"], "nmi": a["nmi"],
                                                   "accuracy": float((labs[r] == truth).mean())})
                                rec = t5.recovery(labs[r], truth, (*EXT_ORDER, s.UNKNOWN, s.BELOW_MIN))
                                rec.insert(0, "rule", r)
                                rec.insert(0, "marker_set", m)
                                rec.insert(0, "label_column", col)
                                rec.insert(0, "resolution", res)
                                rec.insert(0, "graph", graph)
                                recov_rows.append(rec)
                    for r in RULES:
                        types = s.types_resolved(as_assignment(rf, r))
                        per_seed_types[(m, r)].append(types)
                        row[f"{m}__{r}__n_types"] = len(types)
                        row[f"{m}__{r}__n_types_specific"] = sum(c in s.SPECIFIC_CONFIDENCES for c in types.values())
                        row[f"{m}__{r}__types"] = ";".join(f"{t}({c})" for t, c in types.items())
                        row[f"{m}__{r}__fraction_unknown"] = float((labs[r] == s.UNKNOWN).mean())
                        row[f"{m}__{r}__same_label_as_seed42"] = float((ref[m][r] == labs[r]).mean())
                seed_rows.append(row)
            cl42 = per_cell[f"{s.clustering_name(graph, res)}__cluster"].to_numpy()
            for m in SETS:
                for r in RULES:
                    for tr in type_rows(graph, res, as_assignment(rf_42[m], r), cl42, per_seed_types[(m, r)],
                                        types=cc.ordered_labels(sets[m].lists, EXT_ORDER)):
                        type_all.append({"marker_set": m, "rule": r, **tr})
            r42 = seed_rows[-len(s.SEEDS)]
            print(f"[kidney] {graph} r{res:.1f}: types (top-25 / rank-free): "
                  + "; ".join(f"{m} {r42[f'{m}__top25__n_types']}/{r42[f'{m}__rankfree__n_types']}" for m in SETS)
                  + f"; {time.time() - t0:.1f} s", flush=True)
    for k, v in mism.items():
        checks.add(f"reproduction of the scan / rank-free tables: {k}", f"{v} mismatching cell entries", v == 0, "= 0")
    if not checks.ok:
        stop(run, checks)

    # Summary ---------------------------------------------------------------------------------
    seeds = pd.DataFrame(seed_rows)
    agree = pd.DataFrame(agree_rows)
    rows = []
    for (graph, res), grp in seeds.groupby(["graph", "resolution"], sort=False):
        r42 = grp[grp["seed"] == s.SEED].iloc[0]
        other = grp[grp["seed"] != s.SEED]
        for m in SETS:
            for r in RULES:
                p = f"{m}__{r}__"
                row = {"graph": graph, "display": s.graph_display(graph), "resolution": res, "marker_set": m, "rule": r,
                       "n_clusters": int(r42["n_clusters"]), "n_types": int(r42[p + "n_types"]),
                       "n_types_specific": int(r42[p + "n_types_specific"]), "types": r42[p + "types"],
                       "n_types_all_seeds_min": int(grp[p + "n_types"].min()),
                       "n_types_all_seeds_max": int(grp[p + "n_types"].max()),
                       "fraction_unknown": float(r42[p + "fraction_unknown"]),
                       "same_label_as_seed42_median": float(other[p + "same_label_as_seed42"].median())}
                for col, short in (("kidney_reannot_label", "reannot"), ("kidney_pre_label", "pre")):
                    a = agree[(agree["graph"] == graph) & (agree["resolution"] == res) & (agree["marker_set"] == m)
                              & (agree["rule"] == r) & (agree["label_column"] == col)].iloc[0]
                    row[f"ari_vs_{short}"] = a["ari"]
                rows.append(row)
    summ = pd.DataFrame(rows)
    best = summ.sort_values(["n_types", "n_types_specific"], ascending=False).iloc[0]
    checks.add("largest number of types (any marker set, rule, graph, resolution <= 2.0; seed 42)",
               f"{best['n_types']} ({best['marker_set']}, {best['rule']}, {best['graph']} {best['resolution']}): "
               f"{best['types']}", None)

    run.table(checks.frame(), "extmarkers_checks")
    run.table(set_table(sets), "extmarkers_marker_sets")
    run.table(summ, "extmarkers_summary")
    run.table(seeds, "extmarkers_seed_stability")
    run.table(pd.concat(clus_all, ignore_index=True), "extmarkers_cluster_annotation")
    types_df = pd.DataFrame(type_all)
    run.table(types_df, "extmarkers_types_by_resolution")
    run.table(agree, "extmarkers_agreement")
    run.table(pd.concat(recov_rows, ignore_index=True), "extmarkers_recovery")
    run.table(per_cell, "extmarkers_clusters_per_cell", index=True)

    summary_figure(run, summ)
    counts = {"reannot": labels["kidney_reannot_label"].value_counts().to_dict(),
              "pre": labels["kidney_pre_label"].value_counts().to_dict()}
    for r, letter in (("rankfree", "rank-free rescue"), ("top25", "top-25 rule")):
        sub = types_df[(types_df["marker_set"] == "v3_atlas") & (types_df["rule"] == r)].drop(columns=["marker_set", "rule"])
        sf.heatmap_figure(run, sub, counts, name=f"extmarkers_v3_atlas_{r}_heatmap", script=SCRIPT,
                          note=(f"V3 + atlas markers, {letter}. Text = seed-42 confidence of the type's best cluster: "
                                "H high, M medium (type-specific marker in the top 25), R rescued (type-specific marker "
                                "significant beyond the top 25), L shared markers only; a number = clusters with that "
                                "label. DCT1 and DCT2 are merged into DCT. Row label: cells in Fig 2B / Fig 2A."),
                          description=f"kidney_annotation V3 + atlas markers ({letter}): cell types per resolution and graph")
    run.finish({"marker_sets": {k: SET_DISPLAY[k] for k in SETS}, "dct_merge": "v3_atlas: DCT = V3 DCT1 + V3 DCT2 + "
                "atlas DCT", "resolutions": list(RESOLUTIONS), "seeds": list(s.SEEDS), "rules": list(RULES),
                "min_cluster_size": s.MIN_CLUSTER_SIZE, "scanpy_compat_patched": SCANPY_COMPAT_PATCHED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
