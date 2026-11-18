#!/usr/bin/env python
"""dct_prolif information only: dct_labels cluster labels with the route-A ProLIF cells on top, on the dct_labels UMAP.

Analysis plan
(frozen). The authors asked on 3 Nov 2026 which UMAP can be shown in parallel to the total-kidney figure (kidney Fig 2A:
cluster labels from the cluster_de cluster DE rule + kidney_annotation rank-free rescue on the stored graph). The DCT equivalent is the
dct_labels primary UMAP (`dct_labels_umap_cs_ctrl_deg_label_85mm`, `..._cs_zerok_pipeline_label_85mm`; same rule). dct_labels has
no ProLIF. A first version ('Plan the overlay', 'Run it') drew ProLIF on grey cells; the authors then asked for DCT1,
DCT2 and ProLIF in one figure ('I want DCT1, DCT2, and ProLIF on the same figure'; 'Yes, build it'), so this version
colours every cell by its dct_labels primary label and recolours the dct_prolif route-A ProLIF cells on top.

  coordinates: X_umap of the dct_labels labelled objects (submitted pipeline, k = 10, 20 PCs), checked equal to the dct_prolif objects;
  colour groups: dct_labels primary label (cs_ctrl_deg_label / cs_zerok_pipeline_label: DCT1, DCT2, Unknown; ZeroK has no
    DCT2) for every cell that is not route-A ProLIF; ProLIF (cs_t31_label / cs_t31nf_label; drawn on top, larger
    points); 'Below depth floor' cells light grey ('Below depth floor, not tested'; frozen run only, the no-floor run
    has none). 'Cycling, NCC not detected' cells keep their dct_labels colour (counted in the caption);
  panel: dct_labels 85-mm panel style (85 x 80 mm, no axes, title at the left, legend below, tight layout).

Each condition alone (no condition comparison; that is zerok_vs_control). Nothing is refitted and no label changes: the dct_labels labels
stay primary and the cs_t31_* / cs_t31nf_* labels, the canonical table and the labelled objects are only read. The
figure mixes two label sources (dct_labels cluster rule; dct_prolif per-cell cycling call); the captions say so.

Outputs (results/tables/dct_prolif/):
  dct_prolif_prolif_overlay_per_cell.csv: per cell (both runs): UMAP coordinates, dct_labels label and cluster, dct_prolif label,
    colour group;
  dct_prolif_prolif_overlay_by_t27_cluster.csv: per run x condition x dct_labels cluster: its dct_labels label, cells, ProLIF,
    'Cycling, NCC not detected' and below-floor cells, ProLIF fraction, median protein IDs;
  dct_prolif_prolif_overlay_checks.csv (checks, information rows, one format row per figure);
  dct_prolif_prolif_overlay_figure_captions.md.
Figures (results/figures/dct_prolif/; dotplot format, 85 mm): dct_prolif_prolif_overlay_<run>_<cond>_85mm.

Command (project root):  python scripts/dct_prolif_prolif_overlay.py
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
import dct_prolif_lib as lib
import dct_prolif_unresolved as un

SCRIPT = "dct_prolif_prolif_overlay.py"
REQUEST = {
    "date": "2026-11-03",
    "request": "authors, dct_prolif session: a DCT UMAP in parallel to the total-kidney figure; the dct_labels primary UMAP uses the same "
               "cluster rule as kidney Fig 2A but has no ProLIF ('Plan the overlay', 'Run it'); after the first "
               "version (ProLIF on grey cells) the authors asked for DCT1, DCT2 and ProLIF in one figure ('I want DCT1, "
               "DCT2, and ProLIF on the same figure'; 'Yes, build it'): dct_labels primary labels with the dct_prolif route-A ProLIF "
               "cells recoloured on top",
    "status": "information only; cs_t31_* and cs_t31nf_* labels unchanged; dct_labels labels stay primary",
}
# dct_labels primary label and cluster columns (the project notes, DCT labels, combined search; dct_label is checked equal)
PRIMARY = {"Ctrl": ("cs_ctrl_deg_label", "cs_ctrl_deg_cluster"),
           "ZeroK": ("cs_zerok_pipeline_label", "cs_zerok_pipeline_cluster")}
UNKNOWN = "Unknown"
BELOW_GROUP = "Below depth floor, not tested"
GROUP_COLORS = {lib.DCT1: lib.LABEL_COLORS[lib.DCT1], lib.DCT2: lib.LABEL_COLORS[lib.DCT2],
                UNKNOWN: lib.LABEL_COLORS[UNKNOWN], lib.PROLIF: lib.LABEL_COLORS[lib.PROLIF],
                BELOW_GROUP: lib.LABEL_COLORS[lib.t18.BELOW_MIN]}            # dct_labels panel colours + labels_core below-minimum grey
DRAW_ORDER = (BELOW_GROUP, lib.DCT1, lib.DCT2, UNKNOWN, lib.PROLIF)      # dct_labels order (Unknown over DCT1/2); ProLIF last
LEGEND_ORDER = (lib.DCT1, lib.DCT2, lib.PROLIF, UNKNOWN, BELOW_GROUP)
POINT_SIZE = {BELOW_GROUP: 2.0, lib.DCT1: 2.0, lib.DCT2: 2.0, UNKNOWN: 2.0, lib.PROLIF: 5.0}   # dct_labels panels: 2.0
W85, H80 = t28.HALF_MM, 80.0


def group_of(t31_label: np.ndarray, t27_label: np.ndarray) -> np.ndarray:
    """Colour group: route-A ProLIF, else 'Below depth floor, not tested', else the dct_labels primary label."""
    return np.where(t31_label == lib.PROLIF, lib.PROLIF,
                    np.where(t31_label == lib.BELOW, BELOW_GROUP, t27_label)).astype(object)


def cluster_order(values) -> list[str]:
    return sorted(pd.unique(values), key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))


# ---------------------------------------------------------------------------
# Figure (dct_labels 85-mm panel: dct_labels_figures.umap_figures + labels_core_figures.label_panel)
# ---------------------------------------------------------------------------
def overlay_panel(umap: np.ndarray, group: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(W85 * t28.MM, H80 * t28.MM))
    counts = {g: int((group == g).sum()) for g in LEGEND_ORDER}
    for g in DRAW_ORDER:
        m = group == g
        if m.any():
            ax.scatter(umap[m, 0], umap[m, 1], s=POINT_SIZE[g], color=GROUP_COLORS[g], lw=0, rasterized=True,
                       zorder=3 if g == lib.PROLIF else 2)
    ax.set_title(title, loc="left")
    handles = [Line2D([], [], marker="o", ls="", ms=3.0, color=GROUP_COLORS[g], label=f"{g} ({counts[g]})")
               for g in LEGEND_ORDER if counts[g]]
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
    return fig, counts, ncol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif information only: dct_labels labels with route-A ProLIF on the dct_labels UMAP.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "prolif_overlay", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "prolif_overlay_checks")

    summ_a = lib.read_table(run, "route_a_summary")
    bridge = lib.read_table(run, "bridge_crosstabs")
    canon = lib.read_table(run, "dct_labels", dtype={"cell_id": str, "cell_key": str, "cs_t31b_cluster": str,
                                                     "cs_t31nfb_cluster": str})

    cells, cluster_rows, legend_cols = [], [], {}
    figs = t28.Figures(run, "prolif_overlay")
    captions = ["# dct_prolif cluster labels with ProLIF on the dct_labels UMAP: figure captions (draft; observed facts only)", "",
                f"{cc.GENERATED_TAG} ({SCRIPT}). Information only (authors, 3 Nov 2026: 'Plan the overlay', 'Run it'; "
                "'I want DCT1, DCT2, and ProLIF on the same figure', 'Yes, build it'); the dct_prolif labels are unchanged "
                "and the dct_labels labels stay primary. Each figure mixes two label sources: DCT1, DCT2 and Unknown are dct_labels "
                "cluster labels (cluster DE rule + rank-free rescue, as kidney Fig 2A); ProLIF is the dct_prolif route-A "
                "per-cell call (proliferation null, BH q < 0.05, Slc12a3 detected).", ""]
    for cond in lib.CONDITIONS:
        lab_col, clu_col = PRIMARY[cond]
        a31 = ad.read_h5ad(run.use_input(cc.out_path(lib.TASK, "h5ad", f"DCT_{lib.KEY[cond]}_labelled", ".h5ad"),
                                         f"dct_prolif labelled object ({cond})"))
        a27 = ad.read_h5ad(run.use_input(t28.DCT_LABELS_H5AD[cond], f"dct_labels labelled object ({cond})"))
        ids = pd.Index(a27.obs_names.astype(str))
        same = set(ids) == set(a31.obs_names.astype(str)) and a27.n_obs == a31.n_obs
        checks.add(f"{cond}: dct_prolif and dct_labels objects hold the same cells", f"{a31.n_obs} / {a27.n_obs}", bool(same))
        if not same:
            lib.stop(run, checks, "prolif_overlay_checks")
        obs31 = a31.obs.copy()
        obs31.index = obs31.index.astype(str)
        pos = pd.Index(a31.obs_names.astype(str)).get_indexer(ids)
        u27 = np.asarray(a27.obsm["X_umap"], float)
        u31 = np.asarray(a31.obsm["X_umap"], float)[pos]
        checks.add(f"{cond}: X_umap of the dct_prolif object equals the dct_labels object (cells matched on cell_id)",
                   f"max abs difference {np.max(np.abs(u31 - u27)):.1e}", bool(np.array_equal(u31, u27)), "0")
        obs27 = a27.obs.copy()
        obs27.index = ids
        obs31 = obs31.loc[ids]
        t27_label = obs27[lab_col].astype(str).to_numpy()
        t27_cluster = obs27[clu_col].astype(str).to_numpy()
        checks.add(f"{cond}: dct_labels {lab_col} equals dct_label (dct_labels object) and dct_label of the dct_prolif object",
                   int((t27_label != obs27["dct_label"].astype(str).to_numpy()).sum()
                       + (t27_label != obs31["dct_label"].astype(str).to_numpy()).sum()),
                   bool(np.array_equal(t27_label, obs27["dct_label"].astype(str).to_numpy())
                        and np.array_equal(t27_label, obs31["dct_label"].astype(str).to_numpy())), "0 cells differ")
        multi = pd.Series(t27_label).groupby(t27_cluster).nunique()
        checks.add(f"{cond}: each dct_labels cluster ({clu_col}) carries one dct_labels label", f"{int((multi > 1).sum())} of "
                   f"{len(multi)} clusters with more than one", bool((multi == 1).all()), "0 clusters")
        bad_lab = sorted(set(t27_label) - {lib.DCT1, lib.DCT2, UNKNOWN})
        checks.add(f"{cond}: dct_labels labels are DCT1, DCT2 or Unknown only", lib.counts_text(t27_label, cc.DCT_LABEL_ORDER),
                   not bad_lab)
        can = canon[canon["condition"] == cond].set_index("cell_id").reindex(ids)

        for run_key in lib.RUN_KEYS:
            tag = f"{run_key} / {cond}"
            col = f"{lib.RUNS[run_key].prefix_a}_label"
            label = obs31[col].astype(str).to_numpy()
            checks.add(f"{tag}: {col} of the dct_prolif object equals dct_prolif_dct_labels.csv (matched on cell_id)",
                       int((label != can[col].astype(str).to_numpy()).sum()),
                       bool(np.array_equal(label, can[col].astype(str).to_numpy())), "0 cells differ")
            srow = summ_a[(summ_a["run"] == run_key) & (summ_a["condition"] == cond)].iloc[0]
            checks.add(f"{tag}: label counts equal dct_prolif_route_a_summary.csv", lib.counts_text(label),
                       lib.counts_text(label) == srow["label_counts"])
            n_pro, n_cnn = int((label == lib.PROLIF).sum()), int((label == lib.CYCLING_NO_NCC).sum())
            checks.add(f"{tag}: ProLIF and 'Cycling, NCC not detected' counts equal the route-A summary",
                       f"{n_pro}; {n_cnn}", n_pro == int(srow["n_prolif"])
                       and n_cnn == int(srow["n_cycling_ncc_not_detected"]))

            # ProLIF x dct_labels label equals the bridge crosstab -------------------------------------------------------------
            obs_x = pd.Series(t27_label[label == lib.PROLIF]).value_counts().to_dict()
            br = bridge[(bridge["run"] == run_key) & (bridge["condition"] == cond) & (bridge["route"] == "A")
                        & (bridge["t31_column"] == col) & (bridge["reference_column"] == lab_col)
                        & (bridge["t31_label"] == lib.PROLIF)]
            ref_x = dict(zip(br["reference_label"].astype(str), br["n_cells"].astype(int)))
            ref_x = {k: v for k, v in ref_x.items() if v}
            txt = "; ".join(f"{k} {obs_x[k]}" for k in cc.ordered_labels(obs_x, cc.DCT_LABEL_ORDER))
            checks.add(f"{tag}: ProLIF cells by dct_labels label equal dct_prolif_bridge_crosstabs.csv", txt,
                       obs_x == ref_x and len(br) > 0)

            group = group_of(label, t27_label)
            ok_below = not (run_key == "nofloor" and (group == BELOW_GROUP).any())
            checks.add(f"{tag}: below-floor cells only in the frozen run", int((group == BELOW_GROUP).sum()), ok_below)
            keep = (label != lib.PROLIF) & (label != lib.BELOW)
            checks.add(f"{tag}: every cell that is neither ProLIF nor below the floor is coloured by its dct_labels label",
                       int((group[keep] != t27_label[keep]).sum()), bool(np.array_equal(group[keep], t27_label[keep])),
                       "0 cells differ")
            n_t27 = pd.Series(t27_label).value_counts()
            moved = {lab: int(((t27_label == lab) & ~keep).sum()) for lab in n_t27.index}
            exp = {lab: int(n_t27[lab]) - moved[lab] for lab in n_t27.index}
            got = {lab: int((group == lab).sum()) for lab in n_t27.index}
            checks.add(f"{tag}: dct_labels groups in the figure = dct_labels label counts minus ProLIF and below-floor cells",
                       "; ".join(f"{lab} {got[lab]}" for lab in cc.ordered_labels(got, cc.DCT_LABEL_ORDER)), got == exp)
            below_by = pd.Series(t27_label[label == lib.BELOW]).value_counts()
            cnn_by = pd.Series(t27_label[label == lib.CYCLING_NO_NCC]).value_counts()
            checks.add(f"{tag}: figure groups", lib.counts_text(group, LEGEND_ORDER), None)
            if len(below_by):
                checks.add(f"{tag}: below-floor cells by dct_labels label", "; ".join(
                    f"{k} {int(below_by[k])}" for k in cc.ordered_labels(below_by.index, cc.DCT_LABEL_ORDER)), None)

            # Per dct_labels cluster --------------------------------------------------------------------------------------------
            nd = obs27["n_detected"].to_numpy(float)
            for c in cluster_order(t27_cluster):
                m = t27_cluster == c
                cluster_rows.append({
                    "run": run_key, "condition": cond, "floor": lib.RUNS[run_key].floor, "t27_cluster_column": clu_col,
                    "t27_cluster": c, "t27_label": t27_label[m][0], "n_cells": int(m.sum()),
                    "n_prolif": int((m & (label == lib.PROLIF)).sum()),
                    "fraction_prolif": float((m & (label == lib.PROLIF)).sum() / m.sum()),
                    "n_cycling_ncc_not_detected": int((m & (label == lib.CYCLING_NO_NCC)).sum()),
                    "n_below_floor": int((m & (label == lib.BELOW)).sum()),
                    "median_n_detected": float(np.median(nd[m]))})
            cr = pd.DataFrame([r for r in cluster_rows if r["run"] == run_key and r["condition"] == cond])
            hold = cr[cr["n_prolif"] > 0]
            top_n = cr.sort_values(["n_prolif", "fraction_prolif"], ascending=False).iloc[0]
            top_f = cr.sort_values(["fraction_prolif", "n_prolif"], ascending=False).iloc[0]
            checks.add(f"{tag}: ProLIF cells summed over dct_labels clusters equal the ProLIF count", int(cr["n_prolif"].sum()),
                       int(cr["n_prolif"].sum()) == n_pro)
            checks.add(f"{tag}: dct_labels clusters holding at least one ProLIF cell",
                       f"{len(hold)} of {len(cr)} (" + "; ".join(f"{lab} {int((hold['t27_label'] == lab).sum())}"
                                                             for lab in cc.ordered_labels(hold["t27_label"].unique(),
                                                                                          cc.DCT_LABEL_ORDER)) + ")",
                       None)
            checks.add(f"{tag}: dct_labels cluster holding the most ProLIF cells",
                       f"cluster {top_n['t27_cluster']} ({top_n['t27_label']}): {int(top_n['n_prolif'])} of {n_pro} "
                       f"ProLIF cells; {int(top_n['n_prolif'])} of {int(top_n['n_cells'])} cells of the cluster", None)
            checks.add(f"{tag}: largest ProLIF fraction of a dct_labels cluster",
                       f"cluster {top_f['t27_cluster']} ({top_f['t27_label']}): {int(top_f['n_prolif'])} of "
                       f"{int(top_f['n_cells'])} ({top_f['fraction_prolif']:.2f})", None)

            cells.append(pd.DataFrame({
                "run": run_key, "condition": cond, "floor": lib.RUNS[run_key].floor, "cell_id": ids,
                "cell_key": obs27["cell_key"].astype(str).to_numpy(), "plate": obs27["plate"].astype(str).to_numpy(),
                "n_detected": obs27["n_detected"].to_numpy(), "umap_1": u27[:, 0], "umap_2": u27[:, 1],
                "t27_label_column": lab_col, "t27_label": t27_label, "t27_cluster": t27_cluster, "t31_label_column": col,
                "t31_label": label, "overlay_group": group}))

            # Figure ---------------------------------------------------------------------------------------------------
            t28.style()
            fig, counts, ncol = overlay_panel(u27, group, f"{cond}: cluster labels and ProLIF ({un.RUN_TITLE[run_key]})")
            stem = f"prolif_overlay_{run_key}_{lib.KEY[cond]}"
            legend_cols[stem] = ncol
            figs.save(fig, f"{stem}_85mm", f"dct_labels labels with route-A ProLIF on the dct_labels UMAP, {cond}, {run_key}", W85)
            plt.close(fig)
            captions.append(
                f"- `dct_prolif_{stem}_85mm`: {cond}, {lib.RUNS[run_key].display}. UMAP of the dct_labels labelled object "
                f"(submitted pipeline, k = 10, 20 PCs; the coordinates of `dct_labels_umap_{lab_col}_85mm`). Colour: "
                f"dct_labels primary label ({lab_col}; dct_labels counts "
                + "; ".join(f"{k} {int(n_t27[k])}" for k in cc.ordered_labels(n_t27.index, cc.DCT_LABEL_ORDER))
                + f"), with the dct_prolif route-A ProLIF cells ({col}; {counts[lib.PROLIF]} cells; by dct_labels label {txt}) "
                "recoloured and drawn on top with larger points"
                + (f"; cells below the depth floor of {lib.RUNS[run_key].floor:,} protein IDs, not tested for "
                   f"proliferation, light grey ({counts[BELOW_GROUP]}; by dct_labels label "
                   + "; ".join(f"{k} {int(below_by[k])}" for k in cc.ordered_labels(below_by.index, cc.DCT_LABEL_ORDER))
                   + ")" if counts[BELOW_GROUP] else "")
                + ". Shown: " + "; ".join(f"{g} {counts[g]}" for g in LEGEND_ORDER if counts[g]) + "."
                + (f" The {n_cnn} 'Cycling, NCC not detected' cells keep their dct_labels colour (by dct_labels label "
                   + "; ".join(f"{k} {int(cnn_by[k])}" for k in cc.ordered_labels(cnn_by.index, cc.DCT_LABEL_ORDER))
                   + ")." if n_cnn else "")
                + (" The dct_labels primary label has no DCT2 in ZeroK." if lib.DCT2 not in n_t27.index else ""))
            print(f"[{tag}] done", flush=True)

    frame = pd.DataFrame(figs.rows)
    n_fail = int((~frame["pass"]).sum())
    checks.add("figures: Arial, >= 7 pt, no text overlap or text outside, exact width, TrueType in PDF",
               f"{len(frame) - n_fail}/{len(frame)} pass" + (f"; failed: {';'.join(frame.loc[~frame['pass'], 'figure'])}"
                                                             if n_fail else ""), n_fail == 0)
    checks.add("figures: legend columns (two unless two do not fit in 85 mm)",
               "; ".join(f"{k} {v}" for k, v in legend_cols.items()), None)
    run.table(pd.concat(cells, ignore_index=True), "prolif_overlay_per_cell")
    run.table(pd.DataFrame(cluster_rows), "prolif_overlay_by_t27_cluster")
    run.table(pd.concat([checks.frame(), frame], ignore_index=True), "prolif_overlay_checks")
    run.text("\n".join(captions) + "\n", "prolif_overlay_figure_captions")
    run.finish({"request": REQUEST, "coordinates": "X_umap of the dct_labels labelled objects",
                "colour_groups": "dct_labels primary label; route-A ProLIF on top; below-floor cells light grey (frozen run)",
                "primary_columns": PRIMARY, "point_size": POINT_SIZE, "colors": GROUP_COLORS,
                "legend_columns": legend_cols, "runs": {k: vars(v) for k, v in lib.RUNS.items()}, "seed": lib.SEED,
                "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_prolif_overlay_checks.csv.")


if __name__ == "__main__":
    main()
