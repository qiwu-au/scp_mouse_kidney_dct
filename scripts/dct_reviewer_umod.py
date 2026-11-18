#!/usr/bin/env python
"""dct_reviewer step 7: reviewer_core C Umod and neighbouring-segment markers per label and DCT condition (Editors' point 3).

Port of ctrl_reviewer_core_umod.py (unchanged method),
run once per condition.

Inputs (read-only): dct_labels labelled objects (intensity layer) and canonical labels.
Outputs:
  results/tables/dct_reviewer/dct_reviewer_umod_markers_by_label.csv
  results/tables/dct_reviewer/dct_reviewer_umod_slc12a1_codetection.csv
  results/tables/dct_reviewer/dct_reviewer_umod_dct1_vs_dct2_tests.csv
  results/figures/dct_reviewer/dct_reviewer_umod_dotplot_<role>.pdf/.png/.tif
  results/figures/dct_reviewer/dct_reviewer_umod_violins_<role>.pdf/.png/.tif

Per label (both label columns): detection fraction (intensity > 0, Wilson 95% CI) and median
ln(1 + intensity) in detected cells for Umod, Slc12a1, Pvalb, Egf, Slc12a3, Calb1, Slc8a1, S100g and
Klk1; Umod/Slc12a1 co-detection; Umod DCT1 vs DCT2 (Mann-Whitney U on detected cells and on all
cells, Fisher's exact on detection), BH-FDR within condition x label column. Without DCT2 cells
under a column the DCT1 vs DCT2 tests 'do not apply' (status column; not in the BH family).

Command (project root):  python scripts/dct_reviewer_umod.py
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
from scipy import stats

import scp_common as cc
import reviewer_core_lib as t2
import ctrl_reviewer_core_umod as m19
import dct_reviewer_lib as lib

lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "dct_reviewer_umod.py"
MARKERS = m19.MARKERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 7: Umod and neighbouring markers per label and condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str) -> dict:
    s = lib.SETTINGS[cond]
    adata, labels = lib.load(run, cond)
    ln1p = t2.ln1p_frame(adata)
    missing = [m for m, _ in MARKERS if m not in ln1p.columns]
    rows, codet, tests = [], [], []
    for col in s.LABEL_COLUMNS:
        for label in cc.ordered_labels(labels[col]):
            mask = (labels[col] == label).to_numpy()
            for order, (marker, segment) in enumerate(MARKERS, start=1):
                if marker not in ln1p.columns:
                    continue
                v = ln1p.loc[mask, marker]
                d = v[v > 0]
                lo, hi = t2.wilson_ci(len(d), len(v))
                rows.append({"label_column": col, "label": label, "order": order, "marker": marker,
                             "expected_segment": segment, "n_cells": len(v), "n_detected": len(d),
                             "detection_fraction": len(d) / len(v), "detection_ci_low": lo, "detection_ci_high": hi,
                             "median_ln1p_detected": d.median() if len(d) else np.nan,
                             "q25_ln1p_detected": d.quantile(0.25) if len(d) else np.nan,
                             "q75_ln1p_detected": d.quantile(0.75) if len(d) else np.nan,
                             "median_ln1p_all_cells": v.median(), "mean_ln1p_all_cells": v.mean()})
            u, sl = ln1p.loc[mask, "Umod"] > 0, ln1p.loc[mask, "Slc12a1"] > 0
            codet.append({"label_column": col, "label": label, "n_cells": int(mask.sum()), "n_umod_detected": int(u.sum()),
                          "n_slc12a1_detected": int(sl.sum()), "n_umod_and_slc12a1": int((u & sl).sum()),
                          "n_umod_without_slc12a1": int((u & ~sl).sum()), "n_slc12a1_without_umod": int((~u & sl).sum()),
                          "n_neither": int((~u & ~sl).sum())})
        a = ln1p.loc[(labels[col] == "DCT1").to_numpy(), "Umod"]
        b = ln1p.loc[(labels[col] == "DCT2").to_numpy(), "Umod"]
        absent = [l for l, v in (("DCT1", a), ("DCT2", b)) if v.empty]
        status = lib.NOT_APPLICABLE if absent else "tested"
        reason = f"no {' or '.join(absent)} cell under {col}" if absent else ""
        for scope, x, y in (("detected cells only", a[a > 0], b[b > 0]), ("all cells, non-detected = 0", a, b)):
            res = cc.compare_groups(x, y)
            tests.append({"label_column": col, "marker": "Umod", "comparison": "DCT1_vs_DCT2", "scope": scope,
                          "status": status, "reason": reason,
                          "test": "Mann-Whitney U (two-sided)", "n_DCT1": res["n_group"], "n_DCT2": res["n_reference"],
                          "median_DCT1": res["median_group"], "median_DCT2": res["median_reference"],
                          "median_difference": res["median_difference"], "rank_biserial": res["rank_biserial"],
                          "statistic": res["mannwhitney_u"], "p_value": res["p_value"]})
        row = {"label_column": col, "marker": "Umod", "comparison": "DCT1_vs_DCT2", "scope": "detection",
               "status": status, "reason": reason, "test": "Fisher exact (detected vs not)", "n_DCT1": len(a),
               "n_DCT2": len(b), "note": "median_* columns hold detection fractions; statistic = odds ratio"}
        if not absent:
            table = [[int((a > 0).sum()), int((a == 0).sum())], [int((b > 0).sum()), int((b == 0).sum())]]
            odds, p = stats.fisher_exact(table)
            row.update(median_DCT1=float((a > 0).mean()), median_DCT2=float((b > 0).mean()),
                       median_difference=float((a > 0).mean() - (b > 0).mean()), statistic=odds, p_value=p)
        tests.append(row)
    return {"table": pd.DataFrame(rows), "codet": pd.DataFrame(codet),
            "tests": cc.add_bh(pd.DataFrame(tests), ["label_column"]), "ln1p": ln1p[[m for m, _ in MARKERS
                                                                                   if m in ln1p.columns]],
            "labels": labels, "missing": missing}


def plot_dotplot(res: dict, role: str, figs: lib.Figures) -> None:
    markers = [m for m, _ in MARKERS]
    seg = dict(MARKERS)
    fig = lib.new_figure(lib.FULL_MM, 100)
    gs = fig.add_gridspec(1, 2, left=0.2, right=0.98, top=0.86, bottom=0.3, wspace=0.08)
    subs = {c: res[c]["table"][res[c]["table"]["label_column"] == lib.SETTINGS[c].COLUMN_OF[role]]
            for c in lib.CONDITIONS}
    allmed = np.concatenate([subs[c]["median_ln1p_detected"].to_numpy(float) for c in lib.CONDITIONS])
    vmin, vmax = np.nanmin(allmed), np.nanmax(allmed)
    cmap = LinearSegmentedColormap.from_list("blue", t2.BLUE_RAMP)
    order = lib.union_order(*(subs[c]["label"] for c in lib.CONDITIONS))
    for k, cond in enumerate(lib.CONDITIONS):
        ax = fig.add_subplot(gs[0, k])
        sub = subs[cond]
        for i, marker in enumerate(markers):
            rows = sub[sub["marker"] == marker].set_index("label").reindex(order)
            med = rows["median_ln1p_detected"].to_numpy(float)
            frac = rows["detection_fraction"].to_numpy(float)
            ok = np.isfinite(frac)
            sizes = np.maximum(np.nan_to_num(frac) * 160, 1.0)
            colors = [cmap((m - vmin) / (vmax - vmin)) if np.isfinite(m) else t2.INK["grid"] for m in med]
            x = np.arange(len(order))
            ax.scatter(x[ok], np.full(ok.sum(), i), s=sizes[ok], c=[c for c, o in zip(colors, ok) if o],
                       edgecolors=t2.INK["secondary"], linewidths=0.3)
        n = sub.drop_duplicates("label").set_index("label")["n_cells"].reindex(order).fillna(0).astype(int)
        ax.set_xticks(range(len(order)), [f"{l} ({n[l]})" for l in order], rotation=90)
        ax.set_yticks(range(len(markers)), [f"{m} ({seg[m]})" for m in markers] if k == 0 else [""] * len(markers))
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(len(markers) - 0.5, -0.5)
        ax.set_title(f"{cond}: {lib.SETTINGS[cond].COLUMN_OF[role]}", fontsize=7)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=np.sqrt(f * 160), markerfacecolor=t2.INK["grid"],
                      markeredgecolor=t2.INK["secondary"], label=f"{int(f * 100)}%") for f in (0.1, 0.5, 1.0)]
    fig.legend(handles=handles, title="detected", loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=3, frameon=False,
               title_fontsize=7, columnspacing=1.0)
    cax = fig.add_axes([0.62, 0.05, 0.32, 0.025])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label("median ln(1+intensity), detected cells", fontsize=7)
    fig.text(0.005, 0.995, f"Umod and neighbouring markers by {lib.ROLE_DISPLAY[role]}: detection (dot size) and median "
             "ln(1+intensity) in detected cells\n(colour; one scale across both conditions); label (n cells)",
             ha="left", va="top", fontsize=7)
    figs.save(fig, f"umod_dotplot_{role}", f"Umod and neighbouring markers (dotplot), {role} label, Ctrl and ZeroK",
              label_role=role)


def plot_violins(res: dict, role: str, figs: lib.Figures) -> None:
    markers = [m for m, _ in MARKERS]
    n_pairs = 2
    n_rows = int(np.ceil(len(markers) / n_pairs))
    fig = lib.new_figure(lib.FULL_MM, 44 * n_rows + 14)
    gs = fig.add_gridspec(n_rows, 2 * n_pairs, left=0.075, right=0.99, top=0.915, bottom=0.05, hspace=1.0, wspace=0.12)
    order = lib.union_order(*(res[c]["labels"][lib.SETTINGS[c].COLUMN_OF[role]] for c in lib.CONDITIONS))
    colors = t2.label_colors(order)
    for i, marker in enumerate(markers):
        r, p = divmod(i, n_pairs)
        first = None
        for j, cond in enumerate(lib.CONDITIONS):
            ax = fig.add_subplot(gs[r, 2 * p + j], sharey=first)
            first = first or ax
            d, col = res[cond], lib.SETTINGS[cond].COLUMN_OF[role]
            data = [d["ln1p"].loc[(d["labels"][col] == l).to_numpy(), marker] for l in order]
            det = [v[v > 0].to_numpy() for v in data]
            for k, (v, lab) in enumerate(zip(det, order)):
                if len(v) >= 3:
                    parts = ax.violinplot([v], positions=[k], widths=0.8, showextrema=False)
                    for body in parts["bodies"]:
                        body.set_facecolor(colors[lab])
                        body.set_edgecolor("white")
                        body.set_alpha(0.85)
                elif len(v):
                    ax.scatter(np.full(len(v), k), v, color=colors[lab], s=6, zorder=2, linewidths=0)
                if len(v):
                    ax.scatter([k], [np.median(v)], color="white", edgecolor=t2.INK["primary"], s=7, zorder=3,
                               linewidths=0.5)
            ax.set_xticks(range(len(order)), [f"{l if l != 'Unknown' else 'Unk.'}\n{len(v)}"
                                              for l, v in zip(order, det)])
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.set_title(f"{marker}: {cond}", fontsize=7)
            lib.grid_y(ax)
            if j == 0 and p == 0:
                ax.set_ylabel("ln(1+intensity)")
            elif j == 1:
                plt.setp(ax.get_yticklabels(), visible=False)
    totals = "; ".join(f"{c} " + ", ".join(f"{l} {int((res[c]['labels'][lib.SETTINGS[c].COLUMN_OF[role]] == l).sum())}"
                                             for l in order) for c in lib.CONDITIONS)
    fig.text(0.005, 0.995, f"Umod and neighbouring markers by {lib.ROLE_DISPLAY[role]}: ln(1+intensity) in detected "
             "cells\n(dot = median; fewer than 3 detected cells drawn as points); tick labels: label and detected cells; "
             f"Unk. = Unknown\ncells per label: {totals}", ha="left", va="top", fontsize=7)
    figs.save(fig, f"umod_violins_{role}", f"Umod and neighbouring markers (violins), {role} label, Ctrl and ZeroK",
              label_role=role)


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "umod", args)
    lib.ported_from(run, "ctrl_reviewer_core_umod.py")
    res = {c: one_condition(run, c) for c in lib.CONDITIONS}
    for cond in lib.CONDITIONS:
        if res[cond]["missing"]:
            run.note(f"{cond}: markers absent from the protein matrix: {res[cond]['missing']}")
    run.table(lib.stack({c: res[c]["table"] for c in lib.CONDITIONS}), "umod_markers_by_label")
    run.table(lib.stack({c: res[c]["codet"] for c in lib.CONDITIONS}), "umod_slc12a1_codetection")
    run.table(lib.stack({c: res[c]["tests"] for c in lib.CONDITIONS}), "umod_dct1_vs_dct2_tests")
    run.note("Umod DCT1 vs DCT2 tests: BH within condition x label column over tested rows; rows without DCT1 or DCT2 "
             "cells have status 'does not apply'.")
    figs = lib.Figures(run, "umod")
    for role in lib.ROLES:
        plot_dotplot(res, role, figs)
        plot_violins(res, role, figs)
    figs.finish()
    run.finish({"markers": [m for m, _ in MARKERS], "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_umod.csv.")


if __name__ == "__main__":
    main()
