#!/usr/bin/env python
"""kidney_annotation step 2, plan step 6: cluster labels at the graph and resolution chosen by the authors (kidney).

Plan:
the analysis plan. Runs only after the authors' choice at the step-5
checkpoint; the canonical kidney_object labels are not touched (the new labels are an extra column for the authors).

Takes the seed-42 clustering of the resolution scan at --graph / --resolution and writes:
  - kidney_annotation_cluster_labels_per_cell.csv: kidney_t15_label (+ cluster, confidence) next to both submitted columns;
  - crosstabs of the new labels against kidney_reannot_label and kidney_pre_label;
  - kidney_annotation_t15_umap_vs_fig2: stored X_umap, Fig 2A / Fig 2B labels (Fig 2 colours) next to the new labels;
  - kidney_annotation_t15_movement: label-movement heatmaps against both submitted columns;
  - kidney_annotation_t15_dotplot_{main,full}: dotplot-style marker dotplot (SISKA Fig 1c layout; dotplot_lib /
    dotplot_dotplots: same encoding, placement on the new labels, dagger repeats, format checks).

Inputs (read-only): kidney_object canonical kidney object and labels; data/KIDNEY/KIDNEY_markers_V3.xlsx;
  kidney_annotation_scan_clusters_per_cell.csv, kidney_annotation_scan_cluster_annotation.csv.
Command (project root):
  python scripts/kidney_annotation_chosen_labels.py --graph <stored|regressed|t04_total_unscaled_hvg500> --resolution <r>
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
import qc_mito_lib as t3
import cluster_de_lib as t5
import dotplot_lib as t6
import kidney_annotation_scan_lib as s
import cluster_de_figures as t5fig          # umap_labels, heat, label_palette (sets the qc_mito style)
import dotplot_dotplots as t6dot         # Panel, build_figure, checks (sets Arial, 7 pt)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "kidney_annotation_chosen_labels.py"
LABEL_COL = "kidney_t15_label"
INK = t3.INK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation step 2 / plan step 6: labels at the chosen graph/resolution.")
    parser.add_argument("--graph", required=True, choices=s.GRAPH_NAMES)
    parser.add_argument("--resolution", required=True, type=float, choices=s.RESOLUTIONS)
    parser.add_argument("--rule", default="top25", choices=("top25", "rankfree"),
                        help="top25 = primary cluster_de rule; rankfree = rank-free rescue sensitivity "
                             "(kidney_annotation_rankfree_sensitivity.py)")
    parser.add_argument("--markers", default="v3", choices=("v3", "v3_endo"),
                        help="v3 = V3 workbook; v3_endo = V3 plus the endothelial row (kidney_annotation_v3_endothelial.py; "
                             "graphs stored / regressed at 2.0 only)")
    parser.add_argument("--tag", default="", help="suffix for the output names and the step name (keeps runs apart)")
    cc.add_common_args(parser)
    return parser.parse_args()


SUFFIX = {"value": ""}


def nm(base: str) -> str:
    """Output name with the run's tag."""
    return f"{base}_{SUFFIX['value']}" if SUFFIX["value"] else base


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), nm("chosen_checks"))
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit(f"A check failed; see kidney_annotation_{nm('chosen_checks')}.csv.")


def t15_rows(present) -> tuple[list[str], list[str]]:
    """(cell types: endothelial first, then dotplot row order, then extras), and the plotted rows (Unknown and
    below_min_size last)."""
    present = set(map(str, present))
    types = [s.ENDO_TYPE] if s.ENDO_TYPE in present else []
    types += [t for t in t6.KIDNEY_ROW_ORDER if t in present]
    types += [t for t in s.ORDER if t in present and t not in types and t != s.UNKNOWN]
    types += sorted(present - set(types) - {s.UNKNOWN, s.BELOW_MIN})
    return types, types + [x for x in (s.UNKNOWN, s.BELOW_MIN) if x in present]


def palette(adata, rows: list[str]) -> dict[str, str]:
    colors = t5fig.label_palette(adata, s.DATASET)
    spare = [c for c in t3.SERIES if c not in colors.values()]
    for t in rows:
        if t not in colors and t != s.BELOW_MIN:
            colors[t] = spare.pop(0)
    colors[s.BELOW_MIN] = INK["muted"]
    return colors


def umap_figure(run: cc.Run, adata, per_cell: pd.DataFrame, colors: dict[str, str], title_t15: str) -> None:
    xy = np.asarray(adata.obsm["X_umap"])
    order = [s.ENDO_TYPE, *[t for t in s.ORDER if t != s.UNKNOWN], s.UNKNOWN, s.BELOW_MIN]
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 3.2))
    t5fig.umap_labels(axes[0], xy, per_cell["kidney_pre_label"], None, order, colors,
                      "Fig 2A: kidney_pre_label (pre-annotation)")
    t5fig.umap_labels(axes[1], xy, per_cell["kidney_reannot_label"], None, order, colors,
                      "Fig 2B: kidney_reannot_label (re-annotated)")
    t5fig.umap_labels(axes[2], xy, per_cell[LABEL_COL], per_cell["kidney_t15_confidence"], order, colors, title_t15,
                      per_cell["t15_cluster"])
    key = [Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor=INK["secondary"],
                  markeredgewidth=0, label="high / medium confidence"),
           Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor="none",
                  markeredgecolor=INK["secondary"], markeredgewidth=0.6, label="low: shared markers only"),
           Line2D([0], [0], marker="o", linestyle="", markersize=3.6, markerfacecolor=t5fig.UNKNOWN_DEG,
                  markeredgewidth=0, label="Unknown: no V3 marker in the top 25")]
    fig.legend(handles=key, loc="lower center", ncol=3, fontsize=5.6, bbox_to_anchor=(0.5, 0.01),
               title="Right panel: fill = confidence; numbers = Leiden cluster IDs", title_fontsize=5.8)
    fig.suptitle("KIDNEY: stored UMAP (Fig 2) coloured by the submitted labels and by the kidney_annotation cluster labels",
                 x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    t3.footer(fig, SCRIPT, "the kidney_annotation labels are an extra column; canonical labels unchanged")
    run.figure(fig, nm("t15_umap_vs_fig2"), "kidney_annotation cluster labels next to Fig 2A and Fig 2B on the stored UMAP")


def movement_figure(run: cc.Run, per_cell: pd.DataFrame, rows_t15: list[str]) -> dict[str, pd.DataFrame]:
    tables = {}
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 3.6), layout="constrained")
    im = None
    for ax, col, disp in zip(axes, ("kidney_reannot_label", "kidney_pre_label"), ("Fig 2B", "Fig 2A")):
        truth = per_cell[col].astype(str).to_numpy()
        idx = cc.ordered_labels(set(truth), s.ORDER)
        counts = t5.crosstab(truth, per_cell[LABEL_COL].astype(str).to_numpy(), idx, rows_t15)
        frac = counts.div(counts.sum(axis=1), axis=0).fillna(0)
        im = t5fig.heat(ax, counts, frac, f"{col} ({disp}; rows) x {LABEL_COL} (columns)")
        tables[col] = counts
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.01)
    cbar.set_label("fraction of the submitted label's cells", fontsize=5.5)
    cbar.ax.tick_params(labelsize=5)
    cbar.outline.set_visible(False)
    top = 1 - 0.3 / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, 0.04, 1, top - 0.03))
    fig.suptitle("KIDNEY: where cells move from the submitted labels to the kidney_annotation cluster labels; numbers = cells",
                 x=0.01, ha="left", fontsize=7.2)
    t3.footer(fig, SCRIPT)
    run.figure(fig, nm("t15_movement"), "label movement, submitted labels vs kidney_annotation cluster labels")
    return tables


def dotplots(run: cc.Run, adata, labels_t15: pd.Series, types: list[str], rows: list[str], colors: dict[str, str],
             title: str, extra_rows: list[tuple[str, str, list[str]]] = ()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """dotplot rules on the kidney_annotation labels: placement among listing types, dagger repeats, main version (<= 3 per type).

    ``extra_rows``: (short, full name, markers) appended to the V3 workbook rows (e.g. the endothelial row).
    """
    workbook, _ = t6.read_kidney_workbook(run.use_input(cc.INPUTS["kidney"]["markers"],
                                                        "kidney V3 marker workbook (Supp Table 1)"))
    workbook = [*workbook, *extra_rows]
    var = set(map(str, adata.var_names))
    listed = t6.listed_markers(workbook, types, adata.var_names)
    present = [g for g in listed if g in var]
    values = t6.group_values(adata, labels_t15, rows, present)
    vt = values[values["group"].isin(types)]
    mean_z = vt.pivot(index="group", columns="marker", values="mean_z").reindex(types)
    frac = vt.pivot(index="group", columns="marker", values="frac_detected").reindex(types)
    order_full = t6.place_markers(workbook, types, adata.var_names, mean_z)
    order_full = t6.fill_empty_types(order_full, workbook, types, mean_z, frac)
    selection = t6.select_main(vt, order_full, types)
    order_main = t6.main_order(order_full, selection)
    order_full["version"], order_main["version"] = "full", "main"
    order = pd.concat([order_full, order_main], ignore_index=True)
    values.insert(0, "label_column", LABEL_COL)
    values["mean_z_scaled_0_1"] = t6.scaled_01(values)
    counts = labels_t15.value_counts()
    label_order = pd.DataFrame({"label_column": LABEL_COL, "label": rows, "position": range(1, len(rows) + 1),
                                "plotted": True, "n_cells": [int(counts[r]) for r in rows],
                                "colour": [colors.get(r, INK["muted"]) for r in rows]})
    t6dot.REPEAT_NOTE = (f"{t6dot.DAGGER} Shown so that each cell type of the kidney_annotation cluster labels has a column: a marker "
                         "listed for this cell type whose group mean is higher in another cell type that lists it.")
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig_rows = []
    for version, vtxt in (("main", "V3 markers, up to 3 per cell type"),
                          ("full", "all quantified V3 markers")):
        o = t6dot.plotted_order(order, version)
        panel = t6dot.make_panel(f"Kidney, kidney_annotation labels ({title}): {vtxt}", values, label_order, LABEL_COL, o)
        cv, pitch = t6dot.build_figure([panel], 180.0, "right", 4.5)
        tc = t6dot.text_checks(cv)
        paths = run.figure(cv.fig, nm(f"t15_dotplot_{version}"), f"kidney marker dotplot, {version} version, {LABEL_COL}",
                           formats=(".pdf", ".png", ".tif"), tight=False)
        fc = t6dot.file_checks(paths, cv.W, cv.H)
        arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
        ok = bool(tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
                  and fc["width_ok"] and fc.get("tif_mode") == "RGB" and fc.get("pdf_truetype_embedded")
                  and fc.get("pdf_arial"))
        fig_rows.append({"figure": nm(f"t15_dotplot_{version}"), "width_mm": round(cv.W, 2), "height_mm": round(cv.H, 2),
                         "pitch_mm": round(pitch, 2), "n_rows": len(panel.rows), "n_columns": len(panel.markers),
                         **tc, **fc, "pass": ok})
    return order, values, selection, fig_rows


def main() -> None:
    args = parse_args()
    SUFFIX["value"] = args.tag
    run = cc.Run(s.TASK, nm("chosen_labels"), args)
    adata, labels = t5.load(run, s.DATASET)
    cell_ids = adata.obs_names.astype(str)
    name = s.clustering_name(args.graph, args.resolution)
    ann = s.read_table(run, "scan_cluster_annotation", dtype={"cluster": str})
    extra_rows = []
    if args.markers == "v3_endo":
        scan = s.read_table(run, "v3endo_clusters_per_cell", dtype=str).set_index("cell_id")
        label_key, conf_key = f"{name}__{args.rule}_label", f"{name}__{args.rule}_confidence"
        rf = s.read_table(run, "v3endo_cluster_annotation", dtype={"cluster": str}, keep_default_na=False)
        rf = rf[(rf["clustering"] == name) & (rf["marker_set"] == "v3_endo")].set_index("cluster")
        rule_ann = rf[f"{args.rule}_label"]
        rescue = rf["rescue_markers"] if args.rule == "rankfree" else pd.Series(dtype=str)
        extra_rows = [(s.ENDO_TYPE, "Endothelial (kidney_annotation extension; atlas Endo / AVR markers)", list(s.ENDOTHELIAL_MARKERS))]
    elif args.rule == "top25":
        scan = s.read_table(run, "scan_clusters_per_cell", dtype=str).set_index("cell_id")
        label_key, conf_key = f"{name}__label", f"{name}__confidence"
        rule_ann = ann[ann["clustering"] == name].set_index("cluster")["deg_label"]
        rescue = pd.Series(dtype=str)
    else:
        scan = s.read_table(run, "rankfree_clusters_per_cell", dtype=str).set_index("cell_id")
        label_key, conf_key = f"{name}__rankfree_label", f"{name}__rankfree_confidence"
        rf = s.read_table(run, "rankfree_cluster_annotation", dtype={"cluster": str}, keep_default_na=False)
        rf = rf[rf["clustering"] == name].set_index("cluster")
        rule_ann, rescue = rf["rankfree_label"], rf["rescue_markers"]
    checks = t5.Checks(s.DATASET)
    for col in s.LABEL_COLUMNS:
        observed = labels[col].astype(str).value_counts().to_dict()
        checks.add(f"{col} counts equal the project notes (canonical labels unchanged)",
                   "; ".join(f"{k} {observed.get(k, 0)}" for k in t5.EXPECTED_COUNTS[col]),
                   observed == t5.EXPECTED_COUNTS[col])
    checks.add("scan per-cell table has the canonical cells in order", f"{list(scan.index) == list(cell_ids)}",
               list(scan.index) == list(cell_ids))
    checks.add(f"chosen clustering {name} ({args.rule}) is in the per-cell table", f"{label_key in scan.columns}",
               label_key in scan.columns)
    if not checks.ok:
        stop(run, checks)

    a = ann[ann["clustering"] == name].set_index("cluster")
    per_cell = labels[[*s.LABEL_COLUMNS, "n_detected_proteins", "total_intensity"]].copy()
    per_cell["t15_graph"], per_cell["t15_resolution"], per_cell["t15_seed"] = args.graph, args.resolution, s.SEED
    per_cell["t15_rule"] = args.rule
    per_cell["t15_markers"] = args.markers
    per_cell["t15_cluster"] = scan[f"{name}__cluster"].to_numpy()
    per_cell[LABEL_COL] = scan[label_key].to_numpy()
    per_cell["kidney_t15_confidence"] = scan[conf_key].to_numpy()
    overlap = rf["top25_overlap"] if args.markers == "v3_endo" else a["overlap"]
    per_cell["kidney_t15_top25_overlap"] = per_cell["t15_cluster"].map(overlap).fillna("").to_numpy()
    per_cell["kidney_t15_rescue_markers"] = per_cell["t15_cluster"].map(rescue).fillna("").to_numpy()
    same = bool((per_cell["t15_cluster"].map(rule_ann) == per_cell[LABEL_COL]).all())
    checks.add(f"per-cell labels equal the {args.rule} cluster annotation", f"{same}", same)
    if not checks.ok:
        stop(run, checks)

    types, rows = t15_rows(per_cell[LABEL_COL])
    colors = palette(adata, rows)
    rule_txt = "top-25 rule" if args.rule == "top25" else "rank-free rescue"
    mk_txt = "V3 + endothelial row" if args.markers == "v3_endo" else "V3"
    disp = f"{s.graph_display(args.graph)}, Leiden {args.resolution:g}, seed 42, {rule_txt}, {mk_txt}"
    counts = per_cell[LABEL_COL].value_counts()
    checks.add("kidney_t15_label counts", "; ".join(f"{r} {int(counts[r])}" for r in rows), None)

    umap_figure(run, adata, per_cell, colors, f"kidney_annotation cluster labels: {disp}")
    tables = movement_figure(run, per_cell, rows)
    short = {"stored": "(a) stored", "regressed": "(b) regressed", "t04_total_unscaled_hvg500": "(c) preprocessing hvg500"}
    dot_title = (f"{short[args.graph]}, Leiden {args.resolution:g}, {'rank-free' if args.rule == 'rankfree' else 'top-25'}"
                 + (", + endothelial row" if args.markers == "v3_endo" else ""))
    order, values, selection, fig_rows = dotplots(run, adata, per_cell[LABEL_COL], types, rows, colors, dot_title,
                                                  extra_rows)
    for r in fig_rows:
        checks.add(f"{r['figure']}: format checks (>= 7 pt, Arial, no overlaps, 180 mm, TIFF RGB, fonts embedded)",
                   f"min font {r['min_font_pt']}; overlaps {r['n_text_overlaps']}; outside {r['n_text_outside_figure']}",
                   r["pass"])

    if args.markers == "v3_endo":
        clusters = rf.reset_index()
    else:
        clusters = ann[ann["clustering"] == name].copy()
        if args.rule == "rankfree":
            clusters = clusters.merge(rf[["rankfree_label", "rankfree_confidence", "rescued", "rescue_candidates",
                                          "rescue_markers"]].reset_index(), on="cluster", how="left")
    run.table(checks.frame(), nm("chosen_checks"))
    run.table(per_cell, nm("cluster_labels_per_cell"), index=True)
    run.table(clusters, nm("chosen_clusters"))
    for col, tab in tables.items():
        out = tab.copy()
        out["total"] = out.sum(axis=1)
        out.loc["total"] = out.sum(axis=0)
        out.index.name = col
        run.table(out, nm(f"t15_vs_{'reannot' if col == 'kidney_reannot_label' else 'pre'}_crosstab"), index=True)
    run.table(order, nm("t15_dotplot_marker_order"))
    run.table(values, nm("t15_dotplot_values"))
    run.table(selection, nm("t15_dotplot_main_selection"))
    run.table(pd.DataFrame(fig_rows), nm("t15_figure_checks"))
    run.finish({"graph": args.graph, "resolution": args.resolution, "rule": args.rule, "markers": args.markers,
                "tag": args.tag, "seed": s.SEED, "label_column": LABEL_COL,
                "rows": rows, "dotplot_rules": "dotplot_lib placement / dagger / main selection on kidney_t15_label",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")


if __name__ == "__main__":
    main()
