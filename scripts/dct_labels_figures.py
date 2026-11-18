#!/usr/bin/env python
"""dct_labels step 6: Fig 3A (UMAP) and Fig 3B (marker dotplot) replacements per condition (combined search).

Analysis plan, step 6.
Same figures as labels_core_figures.py, per condition, with the cell count and the label columns as parameters
(Ctrl: cs_ctrl_cell_label, cs_ctrl_cluster_label, cs_ctrl_deg_label; ZeroK: cs_zerok_cell_label,
cs_zerok_cluster_label, cs_zerok_pipeline_label, cs_zerok_transfer_label). Conditions are not compared here (zerok_vs_control).

Fig 3A replacement: UMAP of the condition's submitted pipeline (step 1) coloured by each label column, by protein IDs
  per cell and by ln(1 + intensity) of Slc12a3, Pvalb, Calb1 and Mki67 (undetected cells grey); also one 85 mm UMAP
  per label column. Ctrl: 2 x 4 as labels_core; ZeroK: four label panels on top, protein IDs and the four proteins below.
Fig 3B replacement: SISKA-style dotplots in the dotplot format (dot area = fraction of cells with intensity > 0; colour =
  group mean of the scaled .X rescaled 0-1 per protein across the rows shown; Arial >= 7 pt; exact 180 mm and 85 mm
  widths; PDF + 600-dpi PNG + LZW TIFF), per label column and marker set:
    panel    = dotplot step 5 groups plus the DCT V1 workbook markers absent from the retired data (reviewer_core README: Col5a2,
               Ptgfr under DCT1; Trpv5 under DCT2; Cenpf under ProLIF, which is not a panel group), plotted when
               quantified (labels_core rule; plan default A, authors 2 Nov 2026);
    workbook = every DCT V1 workbook marker present, under the first workbook type that lists it.
  dct_labels_marker_order_dct.csv (both marker sets) and its differences from labels_core_marker_order_dct.csv.
All UMAP figures are also written as TIFF and checked with the dotplot text and file checks (labels_core wrote PDF + PNG only).

Inputs (read-only): results/h5ad/dct_labels_DCT_{ctrl,zerok}_step1_submitted.h5ad; dct_labels_submitted_per_cell.csv;
dct_labels_deg_labels_per_cell.csv; dct_labels_transfer_labels_per_cell.csv; data/DCT/DCT_markers_V1.xlsx;
bridge only: results/tables/labels_core/labels_core_marker_order_dct.csv.
Outputs: results/figures/dct_labels/dct_labels_{umap,dotplot}_*; tables dct_labels_marker_order_dct.csv,
dct_labels_marker_order_vs_t18.csv, dct_labels_dotplot_values.csv, dct_labels_figure_checks.csv,
dct_labels_figure_captions.md.

Command (project root):  python scripts/dct_labels_figures.py
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
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import dotplot_lib as t6
import labels_core_lib as t18
import dct_labels_lib as lib
import dotplot_dotplots as t6d   # sets the dotplot figure style (Arial) on import
from labels_core_figures import UMAP_GENES, UNDETECTED, label_panel, panel_for, rows_for, value_panel

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

SCRIPT = "dct_labels_figures.py"
# DCT V1 workbook markers absent from the retired 758-cell data (reviewer_core README, "V1 marker proteins absent from the DCT
# matrix"), with the first workbook type listing them. labels_core added the one detected in its data (Col5a2).
RETIRED_ABSENT = {"Col5a2": "DCT1", "Ptgfr": "DCT1", "Trpv5": "DCT2", "Cenpf": "ProLIF"}
MM = 1 / 25.4
ORDER_FIELDS = ["position", "plotted_position", "group", "in_matrix", "plotted", "note"]


def arial() -> None:
    t18.apply_style()
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})


# ---------------------------------------------------------------------------
# Marker order (labels_core rule)
# ---------------------------------------------------------------------------
def marker_orders(workbook: dict[str, list[str]], var_names: dict[str, set[str]],
                  detected: dict[str, pd.Series]) -> pd.DataFrame:
    """labels_core_figures.marker_orders with both conditions: plotted = quantified in both condition matrices."""
    listing = {g: [t for t, gs in workbook.items() if g in gs] for gs in workbook.values() for g in gs}
    panel = [(group, list(genes)) for group, genes in t6.DCT_PANEL]
    for group, genes in panel:
        genes.extend(g for g, t in RETIRED_ABSENT.items() if t == group and g not in genes)
    rows = []
    for group, genes in panel:
        for g in genes:
            rows.append({"version": "panel", "group": group, "marker": g,
                         "note": "added: workbook marker absent from the retired data" if g in RETIRED_ABSENT
                         else "dotplot step 5 panel"})
    seen = set()
    for t, genes in workbook.items():
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            rows.append({"version": "workbook", "group": t, "marker": g, "note": "first workbook type listing it"})
    order = pd.DataFrame(rows)
    for cond in lib.CONDITIONS:
        k = lib.KEY[cond]
        order[f"in_matrix_{k}"] = order["marker"].isin(var_names[cond])
        order[f"n_cells_detected_{k}"] = order["marker"].map(lambda g: int(detected[cond].get(g, 0)))
        order[f"fraction_cells_detected_{k}"] = order[f"n_cells_detected_{k}"] / len(detected["n_cells"][cond])
    order["in_matrix"] = order[[f"in_matrix_{lib.KEY[c]}" for c in lib.CONDITIONS]].all(axis=1)
    order["plotted"] = order["in_matrix"]
    order["types_listing_in_workbook"] = order["marker"].map(lambda g: ";".join(listing.get(g, [])))
    order["first_listing_type"] = order["marker"].map(lambda g: (listing.get(g) or [""])[0])
    order["position"] = order.groupby("version").cumcount() + 1
    order["plotted_position"] = np.nan
    for v in order["version"].unique():
        m = (order["version"] == v) & order["plotted"]
        order.loc[m, "plotted_position"] = np.arange(1, m.sum() + 1)
    order["repeat"] = False
    det_cols = [f"{p}_{lib.KEY[c]}" for c in lib.CONDITIONS for p in ("in_matrix", "n_cells_detected",
                                                                        "fraction_cells_detected")]
    return order[["version", "position", "plotted_position", "group", "marker", "in_matrix", "plotted",
                  "types_listing_in_workbook", "first_listing_type", *det_cols, "repeat", "note"]]


def order_differences(order: pd.DataFrame, t18_order: pd.DataFrame) -> pd.DataFrame:
    a = order.set_index(["version", "marker"])
    b = t18_order.set_index(["version", "marker"])
    rows = []
    for key in list(dict.fromkeys([*b.index, *a.index])):
        if key not in a.index:
            rows.append({"version": key[0], "marker": key[1], "status": "only in labels_core"})
            continue
        if key not in b.index:
            rows.append({"version": key[0], "marker": key[1], "status": "only in dct_labels",
                         **{f"t27_{f}": a.loc[key, f] for f in ORDER_FIELDS}})
            continue
        changed = [f for f in ORDER_FIELDS if not (pd.isna(a.loc[key, f]) and pd.isna(b.loc[key, f]))
                   and str(a.loc[key, f]) != str(b.loc[key, f])]
        rows.append({"version": key[0], "marker": key[1], "status": "changed: " + ";".join(changed) if changed else "same",
                     **{f"t18_{f}": b.loc[key, f] for f in ORDER_FIELDS},
                     **{f"t27_{f}": a.loc[key, f] for f in ORDER_FIELDS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure checks (dotplot) for figures not built on a t6d canvas
# ---------------------------------------------------------------------------
def drawn_texts(fig) -> list[Text]:
    """Non-empty texts that matplotlib draws: tick labels only for ticks inside the axis view interval."""
    fig.canvas.draw()
    ticks, drawn = set(), []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * max(1.0, abs(hi - lo))
            locs = axis.get_majorticklocs()
            for loc, tick in zip(locs, axis.get_major_ticks(len(locs))):
                for lab in (tick.label1, tick.label2):
                    ticks.add(id(lab))
                    if lo - eps <= loc <= hi + eps and lab.get_visible():
                        drawn.append(lab)
    drawn += [t for t in fig.findobj(Text) if id(t) not in ticks]
    return [t for t in drawn if t.get_text().strip()]


def save_checked(run: cc.Run, fig, name: str, description: str, width_mm: float, cond: str, kind: str,
                 col: str = "") -> dict:
    """dotplot text checks on the drawn figure (before run.figure closes it), then save and run the file checks."""
    shim = SimpleNamespace(fig=fig, texts=drawn_texts(fig))
    tc = t6d.text_checks(shim)
    w, h = fig.get_size_inches() / MM
    paths = run.figure(fig, name, description, formats=FORMATS, tight=False)
    fc = t6d.file_checks(paths, w, h)
    ok = (tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
          and "arial" in tc["fonts"] and fc["width_ok"] and fc.get("pdf_truetype_embedded", False)
          and abs(w - width_mm) < 0.05)
    return {"figure": name, "condition": cond, "kind": kind, "label_column": col, "width_mm": round(w, 2),
            "height_mm": round(h, 2), **tc, **fc, "pass": ok}


FORMATS = (".pdf", ".png", ".tif")


# ---------------------------------------------------------------------------
# UMAPs
# ---------------------------------------------------------------------------
def umap_figures(run: cc.Run, cond: str, adata, labels: pd.DataFrame, cols: list[str]) -> list[dict]:
    arial()
    umap = np.asarray(adata.obsm["X_umap"])
    names = pd.Index(adata.var_names.astype(str))
    intensity = cc.dense(adata.layers["intensity"])
    note = (f"{cond} DCT cells, combined search (n = {adata.n_obs}); UMAP of the submitted pipeline (k = 10, 20 PCs). "
            f"Label columns: {lib.PREFIX[cond]}_*. Grey: protein not detected.")
    labels_row = [("label", c) for c in cols]
    genes = [("gene", g) for g in UMAP_GENES]
    if cond == "Ctrl":      # labels_core layout: 2 x 4
        layout, height = [labels_row + [("ids", None)], genes], 112
    else:                   # four label columns: 4 / 3 / 3, the note in the free cell
        layout, height = [labels_row, [("ids", None), *genes[:2]], [*genes[2:], ("note", None)]], 165
    span = int(np.lcm.reduce([len(r) for r in layout]))
    fig = plt.figure(figsize=(180 * MM, height * MM))
    gs = fig.add_gridspec(len(layout), span)
    for r, items in enumerate(layout):
        w = span // len(items)
        for j, (kind, item) in enumerate(items):
            ax = fig.add_subplot(gs[r, j * w:(j + 1) * w])
            if kind == "label":
                label_panel(ax, umap, labels[item].to_numpy(), item.replace(f"{lib.PREFIX[cond]}_", ""))
            elif kind == "ids":
                value_panel(fig, ax, umap, labels["n_detected"].to_numpy(float), "Protein IDs per cell", "Protein IDs",
                            None)
            elif kind == "gene":
                x = intensity[:, names.get_loc(item)].astype(np.float64)
                det = x > 0
                value_panel(fig, ax, umap, np.log1p(x), f"{item} ({int(det.sum())} of {len(x)} detected)",
                            "ln(1 + intensity)", ~det)
            else:
                ax.axis("off")
                ax.text(0.04, 0.5, "\n".join(textwrap.wrap(note, 42)), fontsize=7, color=t18.INK["secondary"],
                        va="center", ha="left", transform=ax.transAxes)
    if cond == "Ctrl":
        fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=2.5)
        fig.text(0.005, 0.012, note, fontsize=7, color=t18.INK["secondary"])
    else:
        fig.tight_layout(h_pad=2.5)
    checks = [save_checked(run, fig, f"umap_labels_markers_{lib.KEY[cond]}_180mm",
                           f"Fig 3A replacement ({cond}): UMAP by the label columns, markers and protein IDs", 180.0,
                           cond, "umap")]
    for col in cols:
        arial()
        fig, ax = plt.subplots(figsize=(85 * MM, 80 * MM))
        label_panel(ax, umap, labels[col].to_numpy(), f"{cond}: {lib.display(col)}")
        fig.tight_layout()
        checks.append(save_checked(run, fig, f"umap_{col}_85mm", f"Fig 3A replacement ({cond}): UMAP coloured by {col}",
                                   85.0, cond, "umap", col))
    return checks


# ---------------------------------------------------------------------------
# Dotplots
# ---------------------------------------------------------------------------
def dotplots(run: cc.Run, cond: str, values: pd.DataFrame, rows: dict[str, list[str]], order: pd.DataFrame,
             cols: list[str]) -> list[dict]:
    checks = []
    for version in ("panel", "workbook"):
        o = order[(order["version"] == version) & order["plotted"]].sort_values("plotted_position").reset_index(drop=True)
        for col in cols:
            title = (f"{cond}: {lib.display(col)}: "
                     f"{'marker panel' if version == 'panel' else 'DCT V1 workbook markers'}")
            for width, legend in ((180.0, "right"), (85.0, "below")):
                arial()
                panel = panel_for(values, col, rows[col], o, title if width == 180.0 else f"{cond}: {lib.display(col)}")
                cv, pitch = t6d.build_figure([panel], width, legend, 4.5)
                tc = t6d.text_checks(cv)
                name = f"dotplot_{version}_{col}_{int(width)}mm"
                paths = run.figure(cv.fig, name, f"Fig 3B replacement ({version} markers), {col}, {int(width)} mm",
                                   formats=FORMATS, tight=False)
                fc = t6d.file_checks(paths, cv.W, cv.H)
                ok = (tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
                      and "arial" in tc["fonts"] and fc["width_ok"] and fc.get("pdf_truetype_embedded", False))
                checks.append({"figure": name, "condition": cond, "kind": "dotplot", "version": version,
                               "label_column": col, "width_mm": width, "height_mm": round(cv.H, 2),
                               "pitch_mm": round(pitch, 3), **tc, **fc, "pass": ok})
    return checks


def captions(order: pd.DataFrame, rows: dict[str, dict[str, list[str]]], labels: dict[str, pd.DataFrame]) -> str:
    panel = order[(order["version"] == "panel") & order["plotted"]]
    listed_not = order[(order["version"] == "panel") & ~order["plotted"]]
    wb = order[order["version"] == "workbook"]
    absent = ", ".join(wb.loc[~wb["in_matrix"], "marker"]) or "none"
    lines = ["# dct_labels figure captions (draft; observed facts only)", "", f"{cc.GENERATED_TAG} ({SCRIPT}).", ""]
    for cond in lib.CONDITIONS:
        cols = lib.label_columns(cond)
        n = len(labels[cond])
        lines += [f"## Fig 3A replacement, {cond} (`dct_labels_umap_labels_markers_{lib.KEY[cond]}_180mm`)", "",
                  f"UMAP of {n} {cond} DCT cells of the combined Spectronaut search (submitted pipeline on the {cond} "
                  "cells alone: per-protein z-scored ln(1 + intensity), 10 neighbours on 20 principal components). "
                  "Cells coloured by the per-cell label (argmax of DCT V1 workbook marker scores), the Leiden 3.0 "
                  "cluster label of the submitted pipeline, the DE-rule label (Leiden 1.0 clusters annotated by "
                  "cluster-versus-rest Wilcoxon tests and the DCT V1 workbook, rank-free rescue)"
                  + (", the transfer label (majority DE-rule label of the 10 nearest Ctrl cells in the joint "
                     "embedding)" if cond == "ZeroK" else "")
                  + ", and by protein IDs per cell; ln(1 + intensity) of Slc12a3, Pvalb, Calb1 and Mki67 (grey: "
                  "no intensity for that protein).", ""]
        for col in cols:
            lines.append(f"- {col}: {lib.counts_text(labels[cond][col])}.")
        lines.append("")
    lines += ["## Fig 3B replacement (`dct_labels_dotplot_{panel,workbook}_<label column>_{180,85}mm`)", "",
              "Dot area: fraction of cells in the row with intensity > 0. Colour: mean of the per-protein z-scored "
              "ln(1 + intensity) in the row, rescaled 0–1 per protein across the rows shown (within one condition). "
              "Rows: " + "; ".join(f"{c}: {', '.join(rows[cond][c])}" for cond in lib.CONDITIONS
                                    for c in lib.label_columns(cond)) + ".", "",
              "- Panel markers (groups as dotplot step 5, plus DCT V1 workbook markers absent from the retired data and "
              "quantified in the combined search): "
              + " | ".join(f"{g}: {', '.join(panel.loc[panel['group'] == g, 'marker'])}"
                           for g in dict.fromkeys(panel["group"]))
              + f". Listed but not quantified: {', '.join(listed_not['marker']) or 'none'}. Group names are marker "
                "groups, not cell labels.",
              "- Workbook markers: every DCT V1 workbook marker quantified in both conditions, under the first workbook "
              f"type that lists it (Slc12a3 is listed for DCT1, DCT2 and ProLIF). Not quantified: {absent}.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_labels step 6: Fig 3A / 3B replacements per condition.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "figures", args)
    checks = t5.Checks("dct_combined")
    step1 = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str})
    deg = lib.read_table(run, "deg_labels_per_cell", dtype={"cell_id": str})
    transfer = lib.read_table(run, "transfer_labels_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    workbook = t18.read_workbook(run)

    objs, labels, var_names, detected = {}, {}, {}, {"n_cells": {}}
    for cond in lib.CONDITIONS:
        adata = lib.read_step1(run, cond)
        cells = adata.obs_names.astype(str)
        s1, dg = lib.per_condition(step1, cond, cells), lib.per_condition(deg, cond, cells)
        frame = pd.DataFrame({"n_detected": s1["n_detected"].to_numpy(),
                              lib.cs(cond, "cell_label"): s1["cell_label"].astype(str).to_numpy(),
                              lib.cs(cond, "cluster_label"): s1["cluster_label"].astype(str).to_numpy(),
                              lib.cs(cond, "deg_label"): dg["deg_label"].astype(str).to_numpy()}, index=cells)
        if cond == "ZeroK":
            same = list(transfer.index) == list(cells)
            checks.add("ZeroK object and transfer table: same cells, same order", f"{same}", same)
            if not same:
                lib.stop(run, checks, "figure_checks")
            frame["cs_zerok_transfer_label"] = transfer["cs_zerok_transfer_label"].astype(str).to_numpy()
        checks.add(f"{cond}: object and step tables hold the same cells in the same order", "True", True)
        adata.layers["ln1p_intensity"] = np.log1p(cc.dense(adata.layers["intensity"]).astype(np.float64))
        objs[cond], labels[cond] = adata, frame
        var_names[cond] = set(adata.var_names.astype(str))
        detected[cond] = pd.Series((cc.dense(adata.layers["intensity"]) > 0).sum(axis=0),
                                   index=adata.var_names.astype(str))
        detected["n_cells"][cond] = cells

    order = marker_orders(workbook, var_names, detected)
    if lib.LABELS_CORE_MARKER_ORDER.exists():
        t18_order = pd.read_csv(run.use_input(lib.LABELS_CORE_MARKER_ORDER, "labels_core marker order (bridge: list differences only)"))
        diffs = order_differences(order, t18_order)
    else:   # the marker order of the earlier search of the control runs is not a public input
        diffs = pd.DataFrame(columns=["version", "marker", "status"])
        run.note("Marker-order comparison with the earlier search skipped: its table is not available.")
    for r in diffs[diffs["status"] != "same"].itertuples(index=False):
        checks.add(f"marker order vs labels_core: {r.version} {r.marker}", r.status, None)
    markers = list(dict.fromkeys(order.loc[order["plotted"], "marker"]))

    values, rows, fig_checks = [], {}, []
    for cond in lib.CONDITIONS:
        cols = lib.label_columns(cond)
        rows[cond] = {col: rows_for(labels[cond][col]) for col in cols}
        vals = []
        for col in cols:
            v = t6.group_values(objs[cond], labels[cond][col], rows[cond][col], markers)
            v.insert(0, "label_column", col)
            vals.append(v)
        vals = pd.concat(vals, ignore_index=True)
        vals["mean_z_scaled_0_1"] = t6.scaled_01(vals)
        vals.insert(0, "condition", cond)
        values.append(vals)
        fig_checks.extend(umap_figures(run, cond, objs[cond], labels[cond], cols))
        fig_checks.extend(dotplots(run, cond, vals, rows[cond], order, cols))
    values = pd.concat(values, ignore_index=True)
    fig_checks = pd.DataFrame(fig_checks)
    for kind in ("dotplot", "umap"):
        sub = fig_checks[fig_checks["kind"] == kind]
        n_fail = int((~sub["pass"]).sum())
        checks.add(f"{kind} figures: Arial, >= 7 pt, no text overlap or text outside, exact width, TrueType in PDF",
                   f"{len(sub) - n_fail}/{len(sub)} pass" + (f"; failed: {';'.join(sub.loc[~sub['pass'], 'figure'])}"
                                                            if n_fail else ""), n_fail == 0)
    missing = order.loc[~order["in_matrix"], "marker"].drop_duplicates().tolist()
    checks.add("markers not quantified in both conditions (not plotted)", ";".join(missing) or "none", None)

    run.table(order, "marker_order_dct")
    run.table(diffs, "marker_order_vs_t18")
    run.table(values, "dotplot_values")
    run.table(pd.concat([checks.frame(), fig_checks], ignore_index=True), "figure_checks")
    run.text(captions(order, rows, labels), "figure_captions")
    run.finish({"umap_genes": list(UMAP_GENES), "retired_absent_markers_added_to_panel": RETIRED_ABSENT,
                "undetected_colour": UNDETECTED,
                "dotplot_encoding": "dotplot (dotplot_dotplots): area = fraction detected; colour = group mean of the "
                                    "scaled .X rescaled 0-1 per protein across the rows shown (within condition)",
                "formats": list(FORMATS), "rows": rows, "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A figure check failed; see dct_labels_figure_checks.csv.")


if __name__ == "__main__":
    main()
