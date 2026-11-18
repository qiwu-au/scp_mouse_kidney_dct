#!/usr/bin/env python
"""labels_core step 5: Fig 3A (UMAP) and Fig 3B (marker dotplot) replacements for the control DCT cells.

Analysis plan, step 5.

Fig 3A replacement: UMAP of the submitted pipeline (step 1) coloured by ctrl_cell_label, ctrl_cluster_label and
  ctrl_deg_label, by ln(1 + intensity) of Slc12a3, Pvalb, Calb1 and Mki67 (undetected cells grey) and by protein IDs
  per cell; also one 85 mm UMAP per label column.
Fig 3B replacement: SISKA-style dotplots in the dotplot format (dotplot_dotplots: dot area = fraction of cells with
  intensity > 0; colour = group mean of the scaled .X rescaled 0-1 per protein across the rows shown; Arial >= 7 pt;
  exact 180 mm and 85 mm widths; PDF + 600-dpi PNG + LZW TIFF), one per label column and marker set:
    panel    = dotplot step 5 groups (Core Slc12a3 | DCT1 Pvalb, Egf | DCT2 Calb1, Slc8a1, S100g, Klk1 | TAL Umod,
               Slc12a1 | Cycling Mki67, Top2a) plus the workbook marker detected in the new data but not in the
               retired data (Col5a2, DCT1 group);
    workbook = every DCT V1 workbook marker present, under the first workbook type that lists it.
  Rows: DCT1, DCT2, ProLIF, then Unknown when the column has it (n cells in the row label).
  labels_core_marker_order_dct.csv (both marker sets) is written for ctrl_rna_core and zerok_response_core.

Inputs (read-only): results/h5ad/labels_core_DCT_ctrl_step1_submitted.h5ad; labels_core_submitted_per_cell.csv;
  labels_core_deg_labels_per_cell.csv; data/DCT/DCT_markers_V1.xlsx.
Outputs: results/figures/labels_core/labels_core_umap_*.pdf/.png, labels_core_dotplot_*.pdf/.png/.tif;
  results/tables/labels_core/labels_core_marker_order_dct.csv, labels_core_dotplot_values.csv,
  labels_core_figure_checks.csv, labels_core_figure_captions.md

Command (project root):  python scripts/labels_core_figures.py
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
import cluster_de_lib as t5
import dotplot_lib as t6
import labels_core_lib as lib
import dotplot_dotplots as t6d   # sets the dotplot figure style (Arial) on import

import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "labels_core_figures.py"
LABEL_COLUMNS = ("ctrl_cell_label", "ctrl_cluster_label", "ctrl_deg_label")
LABEL_DISPLAY = {"ctrl_cell_label": "Per-cell label (submitted pipeline)",
                 "ctrl_cluster_label": "Cluster label (submitted pipeline, Leiden 3.0)",
                 "ctrl_deg_label": "DE-rule label (rank-free, Leiden 1.0)"}
UMAP_GENES = ("Slc12a3", "Pvalb", "Calb1", "Mki67")
NEW_WORKBOOK_MARKERS = {"Col5a2": "DCT1"}   # detected in the new control data; absent from the retired data (reviewer_core)
MM = 1 / 25.4
UNDETECTED = "#d6d5cf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="labels_core step 5: Fig 3A / 3B replacements for the control DCT cells.")
    cc.add_common_args(parser)
    return parser.parse_args()


def arial() -> None:
    lib.apply_style()
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})


# ---------------------------------------------------------------------------
# Marker orders
# ---------------------------------------------------------------------------
def marker_orders(workbook: dict[str, list[str]], var_names, detected: pd.Series) -> pd.DataFrame:
    var = set(map(str, var_names))
    listing = {g: [t for t, gs in workbook.items() if g in gs] for gs in workbook.values() for g in gs}
    rows = []
    panel = [(group, list(genes)) for group, genes in t6.DCT_PANEL]
    for group, genes in panel:
        genes.extend(g for g, t in NEW_WORKBOOK_MARKERS.items() if t == group and g not in genes)
    for group, genes in panel:
        for g in genes:
            rows.append({"version": "panel", "group": group, "marker": g,
                         "note": "added: workbook marker detected in the new data" if g in NEW_WORKBOOK_MARKERS
                         else "dotplot step 5 panel"})
    seen = set()
    for t, genes in workbook.items():
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            rows.append({"version": "workbook", "group": t, "marker": g, "note": "first workbook type listing it"})
    order = pd.DataFrame(rows)
    order["in_matrix"] = order["marker"].isin(var)
    order["plotted"] = order["in_matrix"]
    order["types_listing_in_workbook"] = order["marker"].map(lambda g: ";".join(listing.get(g, [])))
    order["first_listing_type"] = order["marker"].map(lambda g: (listing.get(g) or [""])[0])
    order["n_cells_detected"] = order["marker"].map(lambda g: int(detected[g]) if g in detected.index else 0)
    order["fraction_cells_detected"] = order["n_cells_detected"] / 727
    order["position"] = order.groupby("version").cumcount() + 1
    order["plotted_position"] = np.nan
    for v in order["version"].unique():
        m = (order["version"] == v) & order["plotted"]
        order.loc[m, "plotted_position"] = np.arange(1, m.sum() + 1)
    order["repeat"] = False
    return order[["version", "position", "plotted_position", "group", "marker", "in_matrix", "plotted",
                  "types_listing_in_workbook", "first_listing_type", "n_cells_detected", "fraction_cells_detected",
                  "repeat", "note"]]


def rows_for(labels: pd.Series) -> list[str]:
    return cc.ordered_labels(labels.astype(str), lib.LABEL_ORDER)


# ---------------------------------------------------------------------------
# UMAPs
# ---------------------------------------------------------------------------
def clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def label_panel(ax, umap, lab: np.ndarray, title: str, legend: bool = True) -> None:
    for label in lib.LABEL_ORDER:
        m = lab == label
        if m.any():
            ax.scatter(umap[m, 0], umap[m, 1], s=2.0, color=lib.LABEL_COLORS[label], lw=0, rasterized=True,
                       label=f"{label} ({int(m.sum())})")
    ax.set_title(title, loc="left")
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2, frameon=False, handletextpad=0.1,
                  columnspacing=0.6, markerscale=2.5, borderaxespad=0)
    clean(ax)


def value_panel(fig, ax, umap, values: np.ndarray, title: str, cbar_label: str, undetected: np.ndarray | None) -> None:
    if undetected is not None and undetected.any():
        ax.scatter(umap[undetected, 0], umap[undetected, 1], s=2.0, color=UNDETECTED, lw=0, rasterized=True)
    keep = ~undetected if undetected is not None else np.ones(len(values), bool)
    order = np.argsort(values[keep])
    pts = ax.scatter(umap[keep][order, 0], umap[keep][order, 1], s=2.0, c=values[keep][order], cmap=t6d.SEQUENTIAL,
                     lw=0, rasterized=True)
    cbar = fig.colorbar(pts, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.4)
    ax.set_title(title, loc="left")
    clean(ax)


def umap_figures(run: cc.Run, adata, labels: pd.DataFrame) -> list[str]:
    arial()
    umap = np.asarray(adata.obsm["X_umap"])
    names = pd.Index(adata.var_names.astype(str))
    intensity = cc.dense(adata.layers["intensity"])
    fig, axes = plt.subplots(2, 4, figsize=(180 * MM, 112 * MM))
    for j, col in enumerate(LABEL_COLUMNS):
        label_panel(axes[0, j], umap, labels[col].to_numpy(), col)
    value_panel(fig, axes[0, 3], umap, labels["n_detected"].to_numpy(float), "Protein IDs per cell", "Protein IDs", None)
    for j, g in enumerate(UMAP_GENES):
        x = intensity[:, names.get_loc(g)].astype(np.float64)
        det = x > 0
        value_panel(fig, axes[1, j], umap, np.log1p(x), f"{g} ({int(det.sum())} of {len(x)} detected)",
                    "ln(1 + intensity)", ~det)
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=2.5)
    fig.text(0.005, 0.012, "Control DCT cells (n = 727); UMAP of the submitted pipeline (k = 10, 20 PCs). Grey: "
             "protein not detected in the cell.", fontsize=7, color=lib.INK["secondary"])
    names_out = ["umap_labels_markers_180mm"]
    run.figure(fig, names_out[0], "Fig 3A replacement: UMAP by the three label columns, markers and protein IDs")
    for col in LABEL_COLUMNS:
        arial()
        fig, ax = plt.subplots(figsize=(85 * MM, 80 * MM))
        label_panel(ax, umap, labels[col].to_numpy(), LABEL_DISPLAY[col])
        fig.tight_layout()
        run.figure(fig, f"umap_{col}_85mm", f"Fig 3A replacement: UMAP coloured by {col}")
        names_out.append(f"umap_{col}_85mm")
    return names_out


# ---------------------------------------------------------------------------
# Dotplots
# ---------------------------------------------------------------------------
def panel_for(values: pd.DataFrame, col: str, rows: list[str], order: pd.DataFrame, title: str) -> "t6d.Panel":
    v = values[values["label_column"] == col]
    markers = order["marker"].tolist()
    frac = v.pivot(index="group", columns="marker", values="frac_detected").reindex(index=rows, columns=markers)
    colr = v.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=rows, columns=markers)
    if frac.isna().any().any() or colr.isna().any().any():
        raise ValueError(f"missing dotplot values for {col}")
    n = v.drop_duplicates("group").set_index("group")["n_cells"].reindex(rows).astype(int).tolist()
    return t6d.Panel(title, rows, n, [lib.LABEL_COLORS[r] for r in rows], markers, t6d.groups_of(order),
                     frac.to_numpy(float), colr.to_numpy(float), [False] * len(markers))


def dotplots(run: cc.Run, values: pd.DataFrame, rows: dict[str, list[str]], order: pd.DataFrame) -> pd.DataFrame:
    checks = []
    for version in ("panel", "workbook"):
        o = order[(order["version"] == version) & order["plotted"]].sort_values("plotted_position").reset_index(drop=True)
        for col in LABEL_COLUMNS:
            title = f"{LABEL_DISPLAY[col]}: {'marker panel' if version == 'panel' else 'DCT V1 workbook markers'}"
            for width, legend in ((180.0, "right"), (85.0, "below")):
                arial()
                panel = panel_for(values, col, rows[col], o, title if width == 180.0 else LABEL_DISPLAY[col])
                cv, pitch = t6d.build_figure([panel], width, legend, 4.5)
                tc = t6d.text_checks(cv)
                name = f"dotplot_{version}_{col}_{int(width)}mm"
                paths = run.figure(cv.fig, name, f"Fig 3B replacement ({version} markers), {col}, {int(width)} mm",
                                   formats=(".pdf", ".png", ".tif"), tight=False)
                fc = t6d.file_checks(paths, cv.W, cv.H)
                ok = (tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
                      and "arial" in tc["fonts"] and fc["width_ok"] and fc.get("pdf_truetype_embedded", False))
                checks.append({"figure": name, "version": version, "label_column": col, "width_mm": width,
                               "height_mm": round(cv.H, 2), "pitch_mm": round(pitch, 3), **tc, **fc, "pass": ok})
    return pd.DataFrame(checks)


def captions(order: pd.DataFrame, rows: dict[str, list[str]], labels: pd.DataFrame) -> str:
    panel = order[(order["version"] == "panel") & order["plotted"]]
    wb = order[order["version"] == "workbook"]
    absent = ", ".join(wb.loc[~wb["in_matrix"], "marker"])
    lines = [f"# labels_core figure captions (draft; observed facts only)", "", f"{cc.GENERATED_TAG} ({SCRIPT}).", ""]
    lines += ["## Fig 3A replacement (`labels_core_umap_labels_markers_180mm`)", "",
              "UMAP of 727 control DCT cells (submitted pipeline: per-protein z-scored ln(1 + intensity), 10 neighbours "
              "on 20 principal components). Top: cells coloured by the per-cell label (argmax of DCT V1 workbook marker "
              "scores), the Leiden 3.0 cluster label of the submitted pipeline, and the DE-rule label (Leiden 1.0 "
              "clusters annotated by cluster-versus-rest Wilcoxon tests and the DCT V1 workbook, rank-free rescue), and by "
              "protein IDs per cell. Bottom: ln(1 + intensity) of Slc12a3, Pvalb, Calb1 and Mki67; grey cells have no "
              "intensity for that protein.", ""]
    for col in LABEL_COLUMNS:
        lines.append(f"- {col}: {lib.counts_text(labels[col])}.")
    lines += ["", "## Fig 3B replacement (`labels_core_dotplot_{panel,workbook}_<label column>_{180,85}mm`)", "",
              "Dot area: fraction of cells in the row with intensity > 0. Colour: mean of the per-protein z-scored "
              "ln(1 + intensity) in the row, rescaled 0–1 per protein across the rows shown. Rows: "
              + "; ".join(f"{c}: {', '.join(rows[c])}" for c in LABEL_COLUMNS) + ".",
              "",
              "- Panel markers (groups as dotplot step 5, plus Col5a2, a DCT1 workbook marker detected in the new data): "
              + " | ".join(f"{g}: {', '.join(panel.loc[panel['group'] == g, 'marker'])}"
                           for g in dict.fromkeys(panel["group"])) + ". Group names are marker groups, not cell labels.",
              "- Workbook markers: every DCT V1 workbook marker quantified in the control data, under the first workbook "
              f"type that lists it (Slc12a3 is listed for DCT1, DCT2 and ProLIF). Not quantified: {absent}.", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "figures", args)
    checks = t5.Checks("dct_ctrl")
    adata = lib.load_step1(run)
    step1 = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    deg = lib.read_table(run, "deg_labels_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    same = list(step1.index) == list(adata.obs_names.astype(str)) == list(deg.index)
    checks.add("object, step-1 and step-2b tables hold the same cells in the same order", f"{same}", same)
    if not checks.ok:
        run.table(checks.frame(), "figure_checks")
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("A check failed; see labels_core_figure_checks.csv.")
    labels = step1[["n_detected", "ctrl_cell_label", "ctrl_cluster_label"]].join(deg[["ctrl_deg_label"]])
    adata.layers["ln1p_intensity"] = np.log1p(cc.dense(adata.layers["intensity"]).astype(np.float64))

    workbook = lib.read_workbook(run)
    detected = pd.Series((cc.dense(adata.layers["intensity"]) > 0).sum(axis=0), index=adata.var_names.astype(str))
    order = marker_orders(workbook, adata.var_names, detected)
    rows = {col: rows_for(labels[col]) for col in LABEL_COLUMNS}
    markers = list(dict.fromkeys(order.loc[order["plotted"], "marker"]))
    values = []
    for col in LABEL_COLUMNS:
        v = t6.group_values(adata, labels[col], rows[col], markers)
        v.insert(0, "label_column", col)
        values.append(v)
    values = pd.concat(values, ignore_index=True)
    values["mean_z_scaled_0_1"] = t6.scaled_01(values)

    umap_names = umap_figures(run, adata, labels)
    fig_checks = dotplots(run, values, rows, order)
    n_fail = int((~fig_checks["pass"]).sum())
    checks.add("dotplots: Arial, >= 7 pt, no text overlap or text outside, exact width, TrueType in PDF",
               f"{len(fig_checks) - n_fail}/{len(fig_checks)} pass", n_fail == 0)
    checks.add("UMAP figures written", ";".join(umap_names), None)
    missing = order.loc[~order["in_matrix"], ["version", "marker"]].drop_duplicates("marker")["marker"].tolist()
    checks.add("markers not quantified in the control data (not plotted)", ";".join(missing) or "none", None)

    run.table(order, "marker_order_dct")
    run.table(values, "dotplot_values")
    run.table(pd.concat([checks.frame(), fig_checks.assign(kind="figure")], ignore_index=True), "figure_checks")
    run.text(captions(order, rows, labels), "figure_captions")
    run.finish({"umap_genes": list(UMAP_GENES), "new_workbook_markers_added_to_panel": NEW_WORKBOOK_MARKERS,
                "dotplot_encoding": "dotplot (dotplot_dotplots): area = fraction detected; colour = group mean of the "
                                    "scaled .X rescaled 0-1 per protein across the rows shown",
                "rows": rows, "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A figure check failed; see labels_core_figure_checks.csv.")


if __name__ == "__main__":
    main()
