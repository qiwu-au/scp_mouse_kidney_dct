#!/usr/bin/env python
"""dct_prolif step 7: figures for the route-A labels, per run and condition (dotplot format).

Analysis plan
(frozen), step 7. Runs: 'frozen' (floor 1,000) and 'nofloor' (approved change, authors 3 Nov 2026).

Figures (results/figures/dct_prolif/; one plot per file, no panel letters; Arial >= 7 pt; PDF + 600 dpi PNG +
TIFF; 85 or 180 mm wide; text-overlap, font and file checks as dct_labels/dct_reviewer):
  - dct_prolif_umap_label_<run>_<cond>_85mm: dct_labels UMAP (submitted pipeline, own condition) by cs_t31_label;
  - dct_prolif_umap_D_<run>_<cond>_85mm: the same UMAP by D (grey: below the depth floor, no D);
  - dct_prolif_umap_ids_<cond>_85mm: the same UMAP by protein IDs per cell (does not depend on the run);
  - dct_prolif_D_vs_depth_<run>_<cond>_85mm: D against ln(n_detected), coloured by the cycling call;
  - dct_prolif_axis_mixture_<run>_<cond>_85mm: histogram of axis A (DCT2 - DCT1 score) with the two-component fit;
  - dct_prolif_dotplot_<run>_<cond>_180mm: SISKA-style dotplot by cs_t31_label (180 mm only: at 85 mm the 25
    columns and the long row labels give a pitch below the 7-pt text height, so the row labels overlap): DCT V1 workbook markers
    (dct_labels plotted order; Mki67 and Top2a are in its ProLIF row) and the 10 proteins of P detected in the most cells
    of both conditions together (chosen by detection, not by label). Dot area: fraction of cells with intensity > 0;
    colour: mean of the dct_labels object's .X (submitted pipeline: per-protein z-scored ln(1 + intensity)), rescaled 0-1
    per protein across the rows shown (dct_labels format);
  - dct_prolif_acceptance_summary_180mm: the five acceptance tests per run and condition.

Inputs (read-only): dct_prolif_{route_a_per_cell,mixture,acceptance_tests,prolif_set}.csv; results/h5ad/
dct_labels_DCT_{ctrl,zerok}_labelled.h5ad; results/tables/dct_labels/dct_labels_marker_order_dct.csv.
Outputs: the figures above; tables dct_prolif_dotplot_markers.csv, dct_prolif_dotplot_values.csv,
dct_prolif_figure_checks.csv; dct_prolif_figure_captions.md.

Command (project root):  python scripts/dct_prolif_figures.py
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import scp_common as cc
import cluster_de_lib as t5
import dotplot_dotplots as t6d
import dotplot_lib as t6
import dct_reviewer_lib as t28
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_figures.py"
DCT_LABELS_ORDER = cc.out_path("dct_labels", "tables", "marker_order_dct", ".csv")
N_TOP_P = 10
P_GROUP = "P: 10 most detected"
UNDETECTED = "#d6d5cf"
NOT_CYCLING = "Not cycling"
CALL_COLORS = {lib.PROLIF: lib.LABEL_COLORS[lib.PROLIF], lib.CYCLING_NO_NCC: lib.LABEL_COLORS[lib.CYCLING_NO_NCC],
               NOT_CYCLING: "#b5b4ad"}
RUN_SHORT = {"frozen": "floor 1,000 IDs", "nofloor": "no floor; QC ≥ 100 IDs"}
UMAP_NOTE = "UMAP of the dct_labels object (submitted pipeline on\nthis condition; k = 10, 20 PCs)."
PASS_FILL, FAIL_FILL = "#dcebd6", "#f4d3cd"
W85, W180 = t28.HALF_MM, t28.FULL_MM


def axes_mm(fig, W: float, H: float, left: float, bottom: float, width: float, height: float):
    return fig.add_axes([left / W, bottom / H, width / W, height / H])


def clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(0.4)


def note(fig, H: float, text: str, y_mm: float = 1.5) -> None:
    fig.text(1.5 / W85, y_mm / H, text, fontsize=7, color=t6d.INK["secondary"], ha="left", va="bottom")


# ---------------------------------------------------------------------------
# UMAPs, D and the mixture
# ---------------------------------------------------------------------------
def umap_label(umap: np.ndarray, lab: np.ndarray, title: str):
    W, H = W85, 96.0
    fig = t28.new_figure(W, H)
    ax = axes_mm(fig, W, H, 14.5, 32.0, 56.0, 56.0)
    handles = []
    for label in lib.LABEL_ORDER:
        m = lab == label
        if m.any():
            ax.scatter(umap[m, 0], umap[m, 1], s=2.0, color=lib.LABEL_COLORS[label], lw=0, rasterized=True)
            handles.append(Line2D([], [], marker="o", ls="", ms=3.5, color=lib.LABEL_COLORS[label],
                                  label=f"{label} ({int(m.sum())})"))
    ax.set_title(title, loc="left", fontsize=7)
    clean(ax)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(14.5 / W, 30.5 / H), ncol=1, frameon=False,
               handletextpad=0.1, borderaxespad=0, fontsize=7, labelspacing=0.25)
    note(fig, H, "UMAP of the dct_labels object (submitted pipeline on this condition).")
    return fig


def umap_value(umap: np.ndarray, values: np.ndarray, missing: np.ndarray, title: str, cbar_label: str,
               diverging: bool, missing_note: str):
    W, H = W85, 78.0
    fig = t28.new_figure(W, H)
    ax = axes_mm(fig, W, H, 6.0, 11.0, 56.0, 56.0)
    if missing.any():
        ax.scatter(umap[missing, 0], umap[missing, 1], s=2.0, color=UNDETECTED, lw=0, rasterized=True)
    keep = ~missing
    order = np.argsort(np.abs(values[keep]) if diverging else values[keep])
    if diverging:
        lim = float(np.nanpercentile(np.abs(values[keep]), 99))
        kw = {"cmap": "RdBu_r", "norm": TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)}
    else:
        kw = {"cmap": t6d.SEQUENTIAL}
    pts = ax.scatter(umap[keep][order, 0], umap[keep][order, 1], s=2.0, c=values[keep][order], lw=0,
                     rasterized=True, **kw)
    ax.set_title(title, loc="left", fontsize=7)
    clean(ax)
    cax = axes_mm(fig, W, H, 64.0, 19.0, 2.2, 40.0)
    cbar = fig.colorbar(pts, cax=cax, extend="both" if diverging else "neither")
    cbar.set_label(cbar_label, fontsize=7)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=7, width=0.4, length=2)
    note(fig, H, missing_note)
    return fig


def d_vs_depth(f: pd.DataFrame, title: str):
    W, H = W85, 82.0
    fig = t28.new_figure(W, H)
    ax = axes_mm(fig, W, H, 13.0, 25.0, 68.0, 50.0)
    a = f["above_floor"].to_numpy(bool)
    x = np.log(f.loc[a, "n_detected"].to_numpy(float))
    y = f.loc[a, "prolif_D"].to_numpy(float)
    lab = f.loc[a, "label"].to_numpy(str)
    call = np.where(np.isin(lab, (lib.PROLIF, lib.CYCLING_NO_NCC)), lab, NOT_CYCLING)
    handles = []
    for k in (NOT_CYCLING, lib.PROLIF, lib.CYCLING_NO_NCC):
        m = call == k
        if m.any():
            ax.scatter(x[m], y[m], s=3.0 if k != NOT_CYCLING else 2.0, color=CALL_COLORS[k], lw=0, rasterized=True)
            handles.append(Line2D([], [], marker="o", ls="", ms=3.5, color=CALL_COLORS[k], label=f"{k} ({int(m.sum())})"))
    ax.axhline(0, color=t6d.INK["muted"], lw=0.4, zorder=0)
    ax.set_xlabel("ln(protein IDs per cell)")
    ax.set_ylabel("D (mean of P minus depth-bin mean)")
    ax.set_title(title, loc="left", fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(13.0 / W, 14.5 / H), ncol=2, frameon=False,
               handletextpad=0.1, columnspacing=0.8, borderaxespad=0, fontsize=7)
    return fig


def axis_mixture(f: pd.DataFrame, mix: pd.Series, title: str):
    W, H = W85, 77.0
    fig = t28.new_figure(W, H)
    ax = axes_mm(fig, W, H, 13.0, 20.0, 68.0, 50.0)
    a = f["above_floor"].to_numpy(bool)
    axis = f.loc[a, "axis"].to_numpy(float)
    axis = axis[np.isfinite(axis)]
    if len(axis) != int(mix["n_fit"]):
        raise ValueError(f"{title}: {len(axis)} axis values, mixture table n_fit {mix['n_fit']}")
    ax.hist(axis, bins=40, density=True, color=UNDETECTED, edgecolor="white", lw=0.3)
    grid = np.linspace(axis.min(), axis.max(), 400)

    def comp(k):
        m, s, w = mix[f"mean_{k}"], mix[f"sd_{k}"], mix[f"weight_{k}"]
        return w * np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    d1, d2 = comp("dct1"), comp("dct2")
    ax.plot(grid, d1, color=lib.LABEL_COLORS[lib.DCT1], lw=0.9, label="DCT1 component")
    ax.plot(grid, d2, color=lib.LABEL_COLORS[lib.DCT2], lw=0.9, label="DCT2 component")
    ax.plot(grid, d1 + d2, color=t6d.INK["primary"], lw=0.7, ls="--", label="Sum")
    ax.set_xlabel("Axis A (DCT2 score minus DCT1 score)")
    ax.set_ylabel("Density")
    ax.set_title(title, loc="left", fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.2)
    note(fig, H, f"{len(axis)} cells with both marker types detected.\nBIC: 1 component {mix['bic_1']:.1f}; "
                 f"2 components {mix['bic_2']:.1f}.")
    return fig


# ---------------------------------------------------------------------------
# Dotplot
# ---------------------------------------------------------------------------
def dotplot_markers(order: pd.DataFrame, P: list[str], objs: dict[str, ad.AnnData]) -> pd.DataFrame:
    wb = order[(order["version"] == "workbook") & order["plotted"].astype(bool)].sort_values("plotted_position")
    rows = [{"group": g, "marker": m, "source": "DCT V1 workbook (dct_labels plotted order)"} for g, m in zip(wb["group"], wb["marker"])]
    det = {}
    for p in P:
        n = sum(int((cc.dense(o.layers["intensity"][:, o.var_names.get_loc(p)]) > 0).sum()) for o in objs.values())
        det[p] = n
    n_cells = sum(o.n_obs for o in objs.values())
    ranked = sorted(P, key=lambda p: (-det[p], p))
    out = pd.DataFrame(rows)
    out["n_cells_detected_both_conditions"] = np.nan
    top = [{"group": P_GROUP, "marker": p, "source": f"P, rank {i + 1} by cells detected (both conditions, all cells)",
            "n_cells_detected_both_conditions": det[p]} for i, p in enumerate(ranked[:N_TOP_P])]
    out = pd.concat([out, pd.DataFrame(top)], ignore_index=True)
    out["fraction_cells_detected_both_conditions"] = out["n_cells_detected_both_conditions"] / n_cells
    out.insert(0, "position", np.arange(1, len(out) + 1))
    return out


def dotplot_panel(values: pd.DataFrame, rows: list[str], markers: pd.DataFrame, title: str) -> t6d.Panel:
    names = markers["marker"].tolist()
    frac = values.pivot(index="group", columns="marker", values="frac_detected").reindex(index=rows, columns=names)
    colr = values.pivot(index="group", columns="marker", values="mean_z_scaled_0_1").reindex(index=rows, columns=names)
    if frac.isna().any().any() or colr.isna().any().any():
        raise ValueError(f"missing dotplot values for {title}")
    n = values.drop_duplicates("group").set_index("group")["n_cells"].reindex(rows).astype(int).tolist()
    return t6d.Panel(title, rows, n, [lib.LABEL_COLORS[r] for r in rows], names, t6d.groups_of(markers),
                     frac.to_numpy(float), colr.to_numpy(float), [False] * len(names))


# ---------------------------------------------------------------------------
# Acceptance summary
# ---------------------------------------------------------------------------
TEST_ROWS = {1: "1  Depth: OR per SD of ln(IDs),\n    upper 95% CI >= 1",
             2: "2  Plate: OR P2 vs P1,\n    95% CI includes 1",
             3: "3  Held-out Mki67/Top2a:\n    Fisher one-sided p < 0.05",
             4: "4  Random matched sets:\n    < 5 of 100 call >= P's cycling",
             5: "5  Null seeds 0-9:\n    median ProLIF Jaccard >= 0.8"}


def cell_text(test: int, value: str) -> str:
    v = str(value).replace("-", "–")
    if test in (1, 2):
        return f"OR {v}"
    if test == 3:
        parts = str(value).split("; ")
        return f"{parts[0]}\n{parts[-1]}"
    if test == 4:
        n, rest = str(value).split(" (", 1)
        return f"{n} sets\n({rest.split(';')[0]})"
    med, rng = str(value).split(" (", 1)
    return f"median {med}\n({rng.replace('range ', '').replace('-', '–')}"


def acceptance_figure(tests: pd.DataFrame):
    W, H = W180, 78.0
    fig = t28.new_figure(W, H)
    left, top, row_h, head_h = 50.0, H - 7.0, 11.0, 9.0
    col_w = (W - left - 1.5) / 4
    ax = axes_mm(fig, W, H, 0, 0, W, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cols = [(r, c) for r in lib.RUN_KEYS for c in lib.CONDITIONS]
    ax.text(1.5, H - 1.5, "dct_prolif acceptance tests (cells above the floor of each run; route A)", fontsize=7.5,
            fontweight="bold", va="top", ha="left")
    for j, (r, c) in enumerate(cols):
        x = left + j * col_w
        ax.text(x + col_w / 2, top - head_h / 2, f"{c}\n{RUN_SHORT[r]}", ha="center", va="center", fontsize=7)
    for i, t in enumerate(sorted(TEST_ROWS)):
        y = top - head_h - (i + 1) * row_h
        ax.text(1.5, y + row_h / 2, TEST_ROWS[t], ha="left", va="center", fontsize=7, linespacing=1.15)
        for j, (r, c) in enumerate(cols):
            row = tests[(tests["run"] == r) & (tests["condition"] == c) & (tests["test"] == t)].iloc[0]
            ok = bool(row["pass"])
            x = left + j * col_w
            ax.add_patch(Rectangle((x + 0.4, y + 0.4), col_w - 0.8, row_h - 0.8, facecolor=PASS_FILL if ok else FAIL_FILL,
                                   edgecolor="none"))
            ax.text(x + col_w / 2, y + row_h / 2 + 1.6, cell_text(t, row["value"]), ha="center", va="center",
                    fontsize=7, linespacing=1.1)
            ax.text(x + col_w / 2, y + 1.3, "Pass" if ok else "Fail", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")
    ax.text(1.5, 1.5, "Green: pass; red: fail. Thresholds fixed in the frozen plan (dct_prolif); P = 36 Tirosh S/G2M "
            "proteins, Mki67 and Top2a held out.", ha="left", va="bottom", fontsize=7, color=t6d.INK["secondary"])
    return fig


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif step 7: figures.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "figures", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "figure_checks")

    per_a = lib.read_table(run, "route_a_per_cell", dtype={"cell_id": str, "cell_key": str})
    mixture = lib.read_table(run, "mixture")
    tests = lib.read_table(run, "acceptance_tests")
    pset = lib.read_table(run, "prolif_set")
    P = pset.loc[pset["in_P"].astype(bool), "matched_name_ctrl"].astype(str).tolist()
    checks.add("P read from the prolif_set table", len(P), len(P) == lib.EXPECTED_P, lib.EXPECTED_P)
    order = pd.read_csv(run.use_input(DCT_LABELS_ORDER, "dct_labels marker order (DCT V1 workbook, plotted)"))
    objs = {c: ad.read_h5ad(run.use_input(t28.DCT_LABELS_H5AD[c], f"dct_labels labelled object ({c})")) for c in lib.CONDITIONS}
    markers = dotplot_markers(order, P, objs)
    for p in markers["marker"]:
        for c, o in objs.items():
            if p not in o.var_names:
                raise KeyError(f"{p} not in the {c} matrix")

    figs = t28.Figures(run, "figures")
    values_all, captions = [], ["# dct_prolif figure captions (draft; observed facts only)", "",
                                f"{cc.GENERATED_TAG} ({SCRIPT}).", ""]
    t28.style()
    for cond in lib.CONDITIONS:
        o = objs[cond]
        umap = np.asarray(o.obsm["X_umap"], float)
        oid = pd.Index(o.obs_names.astype(str))
        ids = o.obs["n_detected"].to_numpy(float)
        key = lib.KEY[cond]
        t28.style()
        fig = umap_value(umap, ids, np.zeros(len(ids), bool), f"{cond}: protein IDs per cell", "Protein IDs",
                         False, UMAP_NOTE)
        figs.save(fig, f"umap_ids_{key}_85mm", f"dct_labels UMAP by protein IDs, {cond}", W85)
        plt.close(fig)
        for run_key in lib.RUN_KEYS:
            tag = f"{run_key} / {cond}"
            f = per_a[(per_a["run"] == run_key) & (per_a["condition"] == cond)].set_index("cell_id")
            same = set(f.index) == set(oid) and len(f) == len(oid)
            checks.add(f"{tag}: route-A table and dct_labels object hold the same cells", f"{len(f)} / {len(oid)}", same)
            if not same:
                lib.stop(run, checks, "figure_checks")
            f = f.loc[oid].reset_index()
            f["above_floor"] = f["above_floor"].astype(str).str.lower().eq("true")
            lab = f["label"].astype(str).to_numpy()
            a = f["above_floor"].to_numpy(bool)
            short = RUN_SHORT[run_key]
            stem = f"{run_key}_{key}"

            t28.style()
            fig = umap_label(umap, lab, f"{cond}: {lib.col(run_key, 'label')} ({short})")
            figs.save(fig, f"umap_label_{stem}_85mm", f"dct_labels UMAP by {lib.col(run_key, 'label')}, {cond}", W85)
            plt.close(fig)

            t28.style()
            D = f["prolif_D"].to_numpy(float)
            fig = umap_value(umap, np.where(a, D, 0.0), ~a, f"{cond}: D ({short})", "D", True,
                             "Grey: below the depth floor (no D)." if (~a).any() else UMAP_NOTE)
            figs.save(fig, f"umap_D_{stem}_85mm", f"dct_labels UMAP by D, {cond}, {run_key}", W85)
            plt.close(fig)

            t28.style()
            fig = d_vs_depth(f, f"{cond}: D against depth ({short})")
            figs.save(fig, f"D_vs_depth_{stem}_85mm", f"D against ln(n_detected), {cond}, {run_key}", W85)
            plt.close(fig)

            t28.style()
            mix = mixture[(mixture["run"] == run_key) & (mixture["condition"] == cond)].iloc[0]
            fig = axis_mixture(f, mix, f"{cond}: axis A and two-component fit ({short})")
            figs.save(fig, f"axis_mixture_{stem}_85mm", f"Axis A histogram with mixture, {cond}, {run_key}", W85)
            plt.close(fig)

            rows = cc.ordered_labels(lab, lib.LABEL_ORDER)
            v = t6.group_values(o, pd.Series(lab, index=oid), rows, markers["marker"].tolist())
            v.insert(0, "label_column", lib.col(run_key, "label"))
            v["mean_z_scaled_0_1"] = t6.scaled_01(v)
            v.insert(0, "condition", cond)
            v.insert(0, "run", run_key)
            values_all.append(v)
            # 180 mm only: at 85 mm the 25 columns and the long row labels leave a pitch below the 7-pt text height.
            t28.style()
            cv, pitch = t6d.build_figure([dotplot_panel(v, rows, markers, f"{cond}: {lib.col(run_key, 'label')} "
                                                                         f"({short})")], W180, "right", 4.5)
            figs.save(cv.fig, f"dotplot_{stem}_180mm", f"Dotplot by {lib.col(run_key, 'label')}, {cond}", W180)
            plt.close(cv.fig)
            captions += [f"## {cond}, {run_key} ({lib.RUNS[run_key].display})", "",
                         f"- Labels ({lib.col(run_key, 'label')}): {lib.counts_text(lab)}.",
                         f"- `dct_prolif_umap_label_{stem}_85mm`: {UMAP_NOTE} Cells coloured by "
                         f"{lib.col(run_key, 'label')}.",
                         f"- `dct_prolif_umap_D_{stem}_85mm`: same UMAP coloured by D (mean over P of detected minus "
                         "depth-bin detection rate; colour range ±99th percentile of |D|)"
                         + ("; grey: below the depth floor." if (~a).any() else "."),
                         f"- `dct_prolif_D_vs_depth_{stem}_85mm`: D against ln(protein IDs) for the {int(a.sum())} "
                         "cells above the floor, coloured by the cycling call (BH q < 0.05 against 1,000 "
                         "detection-decile-matched null sets).",
                         f"- `dct_prolif_axis_mixture_{stem}_85mm`: axis A (DCT2 minus DCT1 residual-z score) of the "
                         f"{int(mix['n_fit'])} cells with both marker types detected; two-component Gaussian fit "
                         f"(means {mix['mean_dct1']:.2f} and {mix['mean_dct2']:.2f}; weights {mix['weight_dct1']:.2f} "
                         f"and {mix['weight_dct2']:.2f}); BIC {mix['bic_1']:.1f} (1 component) vs {mix['bic_2']:.1f} "
                         "(2 components).",
                         f"- `dct_prolif_dotplot_{stem}_180mm`: rows {', '.join(rows)}.", ""]
        captions.append("")

    t28.style()
    fig = acceptance_figure(tests)
    figs.save(fig, "acceptance_summary_180mm", "Acceptance tests 1-5 per run and condition", W180)
    plt.close(fig)

    frame = pd.DataFrame(figs.rows)
    n_fail = int((~frame["pass"]).sum())
    checks.add("figures: Arial, >= 7 pt, no text overlap or text outside, exact width, TrueType in PDF",
               f"{len(frame) - n_fail}/{len(frame)} pass" + (f"; failed: {';'.join(frame.loc[~frame['pass'], 'figure'])}"
                                                             if n_fail else ""), n_fail == 0)
    captions += ["## Dotplots (all)", "",
                 "Dot area: fraction of cells in the row with intensity > 0. Colour: mean of the dct_labels object's .X "
                 "(submitted pipeline: per-protein z-scored ln(1 + intensity), missing = 0 before scaling) in the "
                 "row, rescaled 0–1 per protein across the rows shown. Columns: DCT V1 workbook markers in the dct_labels "
                 "plotted order (Mki67 and Top2a are in its ProLIF row; Col5a2 and Cenpf are not quantified) and "
                 f"the {N_TOP_P} proteins of P detected in the most cells of both conditions together: "
                 + ", ".join(markers.loc[markers["group"] == P_GROUP, "marker"]) + ".", "",
                 "## Acceptance summary (`dct_prolif_acceptance_summary_180mm`)", "",
                 "Five pre-specified tests per run and condition (cells above the floor; route A). Values as in "
                 "dct_prolif_acceptance_tests.csv.", ""]
    run.table(markers, "dotplot_markers")
    run.table(pd.concat(values_all, ignore_index=True), "dotplot_values")
    run.table(pd.concat([checks.frame(), frame], ignore_index=True), "figure_checks")
    run.text("\n".join(captions), "figure_captions")
    run.finish({"runs": {k: vars(v) for k, v in lib.RUNS.items()}, "approved_change": lib.APPROVED_CHANGE,
                "formats": [".pdf", ".png (600 dpi)", ".tif"], "widths_mm": [W85, W180], "n_top_P": N_TOP_P,
                "dotplot_colour": "mean of dct_labels .X, rescaled 0-1 per protein across rows", "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_figure_checks.csv.")


if __name__ == "__main__":
    main()
