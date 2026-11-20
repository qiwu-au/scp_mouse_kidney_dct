#!/usr/bin/env python
"""The response to the reviewers: revised Figure 4e with the key label "DCT, subtype not resolved".



Request:
  - authors, 12 Nov 2026, via the response to the reviewers: "Yes, rename there". In reader-facing text and in Figures 4e and 4f, the DCT
    cells of clusters without marker support (dct_labels label value 'Unknown') are called "DCT, subtype not resolved".
  - authors, 12 Nov 2026, the response to the reviewers: "Yes, draw it" (a report copy of Figure 4e; the committed dct_prolif and dct_labels files stay).

Source figure:
  results/figures/dct_prolif/dct_prolif_prolif_overlay_nofloor_ctrl_85mm, drawn by
  scripts/dct_prolif_prolif_overlay.py. This script reuses that script's colours, point sizes, draw order,
  legend order, panel size and legend layout.

Changes (display only):
  - The key reads "DCT, subtype not resolved (251)" instead of "Unknown (251)". The text is dct_prolif_lib.UNRESOLVED,
    the same as the Figure 4f row.
  - There is no title. The source title held the internal run name "no floor". One plot per file, as in the kidney_figure2
    Figure 2 files and Figure 4f.
Unchanged:
  - cells, groups, counts and colours (the not-resolved group keeps the grey of 'Unknown');
  - coordinates: X_umap of the dct_labels labelled object;
  - panel size 85 x 80 mm.
The groups and the checks below keep the dct_labels value 'Unknown'.

Inputs (read-only):
  - results/tables/dct_prolif/dct_prolif_prolif_overlay_per_cell.csv (run 'nofloor', condition 'Ctrl');
  - results/tables/dct_prolif/dct_prolif_dct_labels.csv (ProLIF check);
  - results/h5ad/dct_labels_DCT_ctrl_labelled.h5ad (coordinate and label check).
Outputs:
  - results/figures/report/report_05_fig3a_umap_85mm.{pdf,png,tif};
  - results/tables/report/report_05_fig3a_umap_checks.csv;
  - results/tables/report/report_step_05_fig3a_umap_manifest.json.

Command (project root):  python scripts/report_dct_umap.py
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

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import scp_common as cc
import cluster_de_lib as t5
import labels_core_figures as t18f
import dct_reviewer_lib as t28
import dct_prolif_lib as t31
import dct_prolif_prolif_overlay as ov31

TASK = "report"
STEP = "05_fig3a_umap"
DCT_PROLIF_DIR = cc.OUT_ROOTS["tables"] / "dct_prolif"
DCT_PROLIF_OVERLAY = DCT_PROLIF_DIR / "dct_prolif_prolif_overlay_per_cell.csv"
DCT_PROLIF_LABELS = DCT_PROLIF_DIR / "dct_prolif_dct_labels.csv"
SOURCE_FIGURE = "results/figures/dct_prolif/dct_prolif_prolif_overlay_nofloor_ctrl_85mm"
DCT_LABELS_H5AD = t28.DCT_LABELS_H5AD["Ctrl"]

PRIMARY = "cs_ctrl_deg_label"            # dct_labels Ctrl primary (the project notes, DCT labels, combined search)
DCT_PROLIF_COL = "cs_t31nf_label"               # dct_prolif no-floor run; route-A ProLIF only is used
UNKNOWN = ov31.UNKNOWN                   # dct_labels label value 'Unknown'
EXPECTED = {t31.DCT1: 362, t31.DCT2: 79, t31.PROLIF: 37, UNKNOWN: 251}   # source figure (dct_prolif README section 16)
PROLIF_BY_DCT_LABELS = {t31.DCT1: 20, t31.DCT2: 6, UNKNOWN: 11}                  # Figure 4e legend draft
DISPLAY = {UNKNOWN: t31.UNRESOLVED}      # display only (authors, 12 Nov 2026); data keep 'Unknown'


def panel(umap: np.ndarray, group: np.ndarray):
    """The dct_prolif overlay panel (dct_prolif_prolif_overlay.overlay_panel) with display names and no title."""
    fig, ax = plt.subplots(figsize=(ov31.W85 * t28.MM, ov31.H80 * t28.MM))
    counts = {g: int((group == g).sum()) for g in ov31.LEGEND_ORDER}
    for g in ov31.DRAW_ORDER:
        m = group == g
        if m.any():
            ax.scatter(umap[m, 0], umap[m, 1], s=ov31.POINT_SIZE[g], color=ov31.GROUP_COLORS[g], lw=0,
                       rasterized=True, zorder=3 if g == t31.PROLIF else 2)
    labels = [f"{DISPLAY.get(g, g)} ({counts[g]})" for g in ov31.LEGEND_ORDER if counts[g]]
    handles = [Line2D([], [], marker="o", ls="", ms=3.0, color=ov31.GROUP_COLORS[g], label=lab)
               for g, lab in zip([g for g in ov31.LEGEND_ORDER if counts[g]], labels)]
    ncol = 2
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=ncol, frameon=False,
                    handletextpad=0.1, columnspacing=0.6, borderaxespad=0)
    fig.canvas.draw()
    if leg.get_window_extent().width > 0.97 * fig.bbox.width:        # two columns do not fit in 85 mm: one column
        leg.remove()
        ncol = 1
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=ncol, frameon=False,
                  handletextpad=0.1, borderaxespad=0)
    t18f.clean(ax)
    fig.tight_layout()
    return fig, counts, ncol, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Figure 3A with the 'DCT, subtype not resolved' key.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_ctrl_combined")

    # --- Figure 4e groups (dct_prolif overlay table, the source of the dct_prolif figure) --------------------------------------
    ov = pd.read_csv(run.use_input(DCT_PROLIF_OVERLAY, "dct_prolif overlay per cell (Figure 3A groups and coordinates)"),
                     dtype={"cell_id": str})
    ov = ov[(ov["run"] == "nofloor") & (ov["condition"] == "Ctrl")].set_index("cell_id")
    counts = ov["overlay_group"].value_counts().to_dict()
    checks.add("groups (overlay table, no-floor run, Ctrl) equal the source figure",
               "; ".join(f"{g} {counts.get(g, 0)}" for g in EXPECTED) + f"; total {len(ov)}",
               counts == EXPECTED and len(ov) == 729, "DCT1 362; DCT2 79; ProLIF 37; Unknown 251; total 729")

    # --- dct_labels labelled object: same cells, coordinates and primary label --------------------------------------------
    adata = ad.read_h5ad(run.use_input(DCT_LABELS_H5AD, "dct_labels labelled object (Ctrl)"))
    ids = pd.Index(adata.obs_names.astype(str))
    same = set(ids) == set(ov.index) and len(ids) == len(ov)
    checks.add("dct_labels Ctrl object and the overlay table hold the same cells (matched on cell_id)",
               f"{len(ids)} object cells; {len(ov)} overlay cells; same set {same}", same)
    if not same:
        raise SystemExit("Cell sets differ; see the checks above.")
    ov = ov.reindex(ids)
    u27 = np.asarray(adata.obsm["X_umap"], float)
    uov = ov[["umap_1", "umap_2"]].to_numpy(float)
    diff = float(np.max(np.abs(u27 - uov)))
    checks.add("coordinates: overlay umap_1/umap_2 equal X_umap of the dct_labels object", f"max abs difference {diff:.1e}",
               diff < 1e-6, "< 1e-6")
    obs_primary = adata.obs[PRIMARY].astype(str).to_numpy()
    eq_lab = bool((ov["t27_label"].astype(str).to_numpy() == obs_primary).all())
    checks.add(f"overlay t27_label equals {PRIMARY} of the dct_labels object", str(eq_lab), eq_lab)

    # --- ProLIF group = dct_prolif route-A ProLIF (no-floor run); every other cell keeps its dct_labels label -------------------
    lab31 = pd.read_csv(run.use_input(DCT_PROLIF_LABELS, "dct_prolif canonical labels"), dtype={"cell_id": str})
    lab31 = lab31[lab31["condition"] == "Ctrl"].set_index("cell_id").reindex(ids)
    group = ov["overlay_group"].astype(str).to_numpy()
    pro31 = lab31[DCT_PROLIF_COL].astype(str).to_numpy() == t31.PROLIF
    pro_ov = group == t31.PROLIF
    ok_pro = bool((pro31 == pro_ov).all()) and int(pro_ov.sum()) == 37
    checks.add(f"ProLIF group = cells with {DCT_PROLIF_COL} ProLIF (dct_prolif_dct_labels.csv)",
               f"{int(pro_ov.sum())} overlay; {int(pro31.sum())} dct_prolif; equal {bool((pro31 == pro_ov).all())}", ok_pro)
    ok_rest = bool((group[~pro_ov] == obs_primary[~pro_ov]).all())
    checks.add(f"every other cell's group = its {PRIMARY}", f"{int((~pro_ov).sum())} cells; equal {ok_rest}", ok_rest)
    by27 = pd.Series(obs_primary[pro_ov]).value_counts().to_dict()
    checks.add(f"ProLIF cells by {PRIMARY} equal the legend draft",
               "; ".join(f"{k} {by27.get(k, 0)}" for k in PROLIF_BY_DCT_LABELS), by27 == PROLIF_BY_DCT_LABELS,
               "DCT1 20; DCT2 6; Unknown 11")

    # --- figure ---------------------------------------------------------------------------------------------------
    t28.style()
    figs = t28.Figures(run, STEP)
    fig, shown, ncol, labels = panel(u27, group)
    exp_labels = ["DCT1 (362)", "DCT2 (79)", "ProLIF (37)", f"{t31.UNRESOLVED} (251)"]
    checks.add("key labels shown (display only; the data keep the dct_labels value 'Unknown')", "; ".join(labels),
               labels == exp_labels, "; ".join(exp_labels))
    checks.add("legend columns (two unless two do not fit in 85 mm)", ncol, None)
    row = figs.save(fig, f"{STEP}_85mm", "Revised Figure 3A: control DCT cells on the dct_labels UMAP, dct_labels cluster labels "
                    "with the dct_prolif route-A ProLIF cells on top; key 'DCT, subtype not resolved' for the dct_labels value "
                    "'Unknown'; no title", ov31.W85)
    plt.close(fig)
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 85 mm, TrueType in PDF)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; "
               f"overlaps {row['n_text_overlaps']}; outside {row['n_text_outside_figure']}", bool(row["pass"]))

    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"request": "authors, 12 Nov 2026: 'Yes, rename there' (via the response to the reviewers); 'Yes, draw it'",
                "source_figure": SOURCE_FIGURE, "source_script": "scripts/dct_prolif_prolif_overlay.py",
                "display_names": DISPLAY, "title": "none (source title held the internal run name 'no floor')",
                "expected_groups": EXPECTED, "primary_label": PRIMARY, "prolif_source": DCT_PROLIF_COL,
                "colors": ov31.GROUP_COLORS, "point_size": ov31.POINT_SIZE, "legend_columns": ncol,
                "size_mm": [ov31.W85, ov31.H80], "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit("A check failed; see report_05_fig3a_umap_checks.csv.")


if __name__ == "__main__":
    main()
