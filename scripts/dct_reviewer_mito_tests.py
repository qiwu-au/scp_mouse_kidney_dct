#!/usr/bin/env python
"""dct_reviewer step 9b: qc_mito tests A-D per DCT condition (combined search).

Port of ctrl_reviewer_core_mito_tests.py, run once per condition.
The ctrl_reviewer_core functions variance_explained (A), global_cutoffs (B) and within_type (C) are called unchanged under
dct_reviewer_lib.bound (settings from dct_reviewer_lib.part2_settings). removed_cells (D) is copied because it reads
the separate control export through the intake_core reader: here each condition is rebuilt from the combined export with
the intake_core filters (dct_reviewer_lib.rebuild_raw), and the retained-row MitoCarta sets are compared with the step 9a
annotation (dct_reviewer_protein_annotation.csv) instead of the ctrl_reviewer_core one.

Inputs (read-only):
  results/tables/dct_reviewer/dct_reviewer_mito_qc_per_cell.csv and dct_reviewer_protein_annotation.csv (step 9a)
  results/tables/dct_labels/dct_labels_dct_canonical_labels.csv (label cross-check)
  results/tables/qc_mito/qc_mito_mitocarta_mouse_list.csv
  data/DCT/DCT_Ctrl_ZeroK_combine_1535cells.csv (combined export, for test D only)
Outputs (results/tables/dct_reviewer/, results/figures/dct_reviewer/; every table has `condition`):
  A  dct_reviewer_mito_variance_explained.csv          R2 of label, depth, plate and their sums (OLS; raw and rank scale)
  B  dct_reviewer_mito_global_cutoff_simulation.csv     share of each label removed by global cutoffs
     dct_reviewer_mito_global_cutoff_tests.csv          label x removed: chi-square with permutation p
  C  dct_reviewer_mito_within_type_depth_correlation.csv  Spearman of mito metrics with depth inside each label
  D  dct_reviewer_mito_all_raw_cells.csv, dct_reviewer_mito_removed_vs_retained.csv
  dct_reviewer_mito_tests_checks.csv
  figures: dct_reviewer_mito_variance_explained, dct_reviewer_mito_global_cutoff,
           dct_reviewer_mito_within_type_depth_correlation, dct_reviewer_mito_removed_vs_retained

Definitions (as ctrl_reviewer_core): depth = log10(protein IDs) and log10(total intensity). B cutoffs (high side): global median
+ 3 x MAD (scaled), global top 5%, global top 10%, and the within-label 3 x MAD high flags of step 9a. B
permutation p: removal flags permuted 10,000 times (numpy default_rng(42), one stream per condition); statistic =
Pearson chi-square. D: metrics over all protein rows of the combined export (rows not detected in a condition add
zero) for all raw cells of the condition, so cells removed by the >= 100-ID filter and retained cells are measured
on the same protein set.

Command (project root):  python scripts/dct_reviewer_mito_tests.py
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
import reviewer_core_lib as t2
import qc_mito_lib as t3
import ctrl_reviewer_core_lib as l19
import ctrl_reviewer_core_mito_tests as m19t
import dct_reviewer_lib as lib

lib.style()
import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCRIPT = "dct_reviewer_mito_tests.py"
DEPTH_VARS = m19t.DEPTH_VARS
DEPTH_DISPLAY = m19t.DEPTH_DISPLAY
CUTOFFS = m19t.CUTOFFS
CUTOFF_DISPLAY = m19t.CUTOFF_DISPLAY
D_METRICS = m19t.D_METRICS
N_PERM = m19t.N_PERM
SHORT_METRIC = {"mitocarta_intensity_fraction": "MitoCarta, intensity", "mitocarta_id_fraction": "MitoCarta, IDs",
                "oxphos_intensity_fraction": "OXPHOS, intensity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 9b: qc_mito tests A-D per DCT condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def read_frames(run: cc.Run) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Step 9a per-cell table per condition with the condition's own label column names (cell_id index)."""
    path = lib.table_path("mito_qc_per_cell")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run dct_reviewer_mito_qc.py first.")
    per_cell = pd.read_csv(run.use_input(path, "dct_reviewer step 9a per-cell mito QC"), dtype={"cell_id": str},
                           keep_default_na=False, na_values=[""])
    canon = lib.read_canonical(run)
    frames, checks = {}, []
    for cond in lib.CONDITIONS:
        s = lib.SETTINGS[cond]
        frame = lib.from_generic(per_cell[per_cell["condition"] == cond].drop(columns="condition"), cond)
        frame = frame.set_index("cell_id")
        ref = canon[canon["condition"] == cond].set_index("cell_id")
        same = list(frame.index) == list(ref.index) and all(
            np.array_equal(frame[c].astype(str).to_numpy(), ref[c].astype(str).to_numpy()) for c in s.LABEL_COLUMNS)
        checks.append({"condition": cond, "check": "step 9a cells, order and both label columns equal the dct_labels table",
                       "value": f"{len(frame)} cells", "pass": bool(same and len(frame) == lib.N_CELLS[cond])})
        frames[cond] = frame
    return frames, checks


# ---------------------------------------------------------------------------
# D. Cells removed by the >= 100-ID filter (copy of ctrl_reviewer_core_mito_tests.removed_cells, combined export)
# ---------------------------------------------------------------------------
def removed_cells(run: cc.Run, mitocarta: pd.DataFrame, frames: dict[str, pd.DataFrame]):
    annotation = pd.read_csv(run.use_input(lib.table_path("protein_annotation"), "dct_reviewer step 9a protein annotation"),
                             keep_default_na=False)
    rebuilt = lib.rebuild_raw(run)
    per_cell, rows, checks = [], [], []
    for cond in lib.CONDITIONS:
        rb = rebuilt[cond]
        flags_all = t3.annotate_proteins(l19.mito_meta(rb.raw.meta, rb.names), mitocarta)
        kept = flags_all.loc[rb.filt.keep_proteins]
        ann = annotation[annotation["condition"] == cond].set_index("protein")
        checks.append({"condition": cond, "check": "retained protein rows of the rebuild equal the step 9a annotation rows",
                       "value": f"{int(rb.filt.keep_proteins.sum())} of {len(rb.names)} rows",
                       "pass": list(kept.index) == list(ann.index.astype(str))})
        for name in t3.SETS:
            same = (list(kept.index) == list(ann.index.astype(str))
                    and np.array_equal(kept[name].to_numpy(bool), ann[name].astype(str).str.lower().eq("true").to_numpy()))
            checks.append({"condition": cond, "check": f"set '{name}' on retained rows equals the step 9a annotation",
                           "value": f"{int(kept[name].sum())} proteins", "pass": bool(same)})
        raw_int = pd.DataFrame(np.nan_to_num(rb.raw.values, nan=0.0), index=rb.raw.cells.index.astype(str),
                               columns=rb.names)
        retained_ids = list(raw_int.index[rb.filt.keep_cells])
        checks.append({"condition": cond,
                       "check": f"cells passing the >= 100-ID filter equal the {lib.N_CELLS[cond]} dct_labels cells (order)",
                       "value": f"{len(retained_ids)} of {len(raw_int)} raw cells",
                       "pass": retained_ids == list(frames[cond].index)})
        metrics = t3.mito_metrics(raw_int.to_numpy(), flags_all, raw_int.index)
        metrics.index.name = "cell_id"
        metrics.insert(0, "retained", rb.filt.keep_cells)
        metrics.insert(0, "dataset", "dct")
        per_cell.append(lib.tag(metrics.reset_index(), cond))
        ids = metrics["n_detected_proteins"]
        retained = metrics["retained"].to_numpy(bool)
        low_cut = float(np.quantile(ids[retained], 0.10))
        groups = {"retained_all": retained,
                  f"retained_lowest_decile_ids (<= {low_cut:.0f} IDs)": retained & (ids <= low_cut).to_numpy()}
        removed = ~retained
        for metric in D_METRICS:
            x = metrics[metric]
            for ref_name, ref in groups.items():
                row = {"condition": cond, "dataset": "dct", "metric": metric, "comparison": f"removed_vs_{ref_name}",
                       "n_removed_cells": int(removed.sum()),
                       "n_removed_with_0_ids": int((removed & (ids == 0).to_numpy()).sum()),
                       "removed_ids_min": int(ids[removed].min()), "removed_ids_max": int(ids[removed].max()),
                       "reference_ids_min": int(ids[ref].min()), "reference_ids_max": int(ids[ref].max())}
                row.update(cc.compare_groups(x[removed], x[ref]))
                rows.append(row)
        run.note(f"{cond}: test D over {raw_int.shape[1]} protein rows of the combined export; {len(raw_int)} raw cells, "
                 f"{int(removed.sum())} removed ({int((removed & (ids == 0).to_numpy()).sum())} with 0 IDs); "
                 f"MitoCarta rows {int(flags_all['mitocarta'].sum())}.")
    return pd.concat(per_cell, ignore_index=True), pd.DataFrame(rows), checks


# ---------------------------------------------------------------------------
# Figures (dotplot format; Ctrl and ZeroK side by side)
# ---------------------------------------------------------------------------
def role_of(cond: str, col: str) -> str:
    return lib.SETTINGS[cond].role(col)


def fig_variance(ve: pd.DataFrame, figs: lib.Figures) -> None:
    sub = ve[(ve["scale"] == "raw") & (ve["depth_variable"] == "log10_n_detected")]
    bars = [("r2_depth", "depth only (log10 IDs)", t3.SERIES[0]), ("r2_label", "label only", t3.SERIES[1]),
            ("r2_label_plus_depth", "label + depth", t3.SERIES[2])]
    fig = lib.new_figure(lib.FULL_MM, 118)
    gs = fig.add_gridspec(1, 2, left=0.17, right=0.98, top=0.84, bottom=0.1, wspace=0.08)
    first = None
    xmax = float(sub[[b[0] for b in bars]].max().max()) * 1.3
    for j, cond in enumerate(lib.CONDITIONS):
        d = sub[sub["condition"] == cond].reset_index(drop=True)
        ax = fig.add_subplot(gs[0, j], sharey=first)
        first = first or ax
        y = np.arange(len(d))
        h = 0.27
        for k, (key, name, color) in enumerate(bars):
            ax.barh(y + (k - 1) * h, d[key], height=h * 0.9, color=color, label=name)
            for yi, v in zip(y + (k - 1) * h, d[key]):
                ax.text(v + xmax * 0.01, yi, f"{v:.2f}", va="center", fontsize=7, color=t2.INK["secondary"])
        if j == 0:
            ax.set_yticks(y, [f"{SHORT_METRIC[m]}\n{role_of(cond, c)} label" for m, c in zip(d["metric"], d["label_column"])])
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(len(d) - 0.5, -0.5)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("R² (OLS, all cells)")
        ax.set_title(f"{cond} ({int(d['n_cells'].iloc[0])} cells)", fontsize=7)
        ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
    handles = [Patch(color=c, label=n) for _, n, c in bars]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.935), ncol=3, frameon=False)
    fig.text(0.005, 0.995, "Variance of mitochondrial fractions explained by cell label and by depth, per condition\n"
             "(rank scale, log10 total intensity and plate terms in dct_reviewer_mito_variance_explained.csv)",
             ha="left", va="top", fontsize=7)
    figs.save(fig, "mito_variance_explained", "R2 of label and depth for mito fractions, Ctrl and ZeroK")


def fig_cutoffs(sim: pd.DataFrame, figs: lib.Figures) -> None:
    metric = "mitocarta_intensity_fraction"
    fig = lib.new_figure(lib.FULL_MM, 120)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.99, top=0.83, bottom=0.1, hspace=0.55, wspace=0.1)
    w = 0.2
    for r, role in enumerate(lib.ROLES):
        first = None
        for j, cond in enumerate(lib.CONDITIONS):
            col = lib.SETTINGS[cond].COLUMN_OF[role]
            d = sim[(sim["condition"] == cond) & (sim["label_column"] == col) & (sim["metric"] == metric)]
            order = t3.label_order(d.loc[d["label"] != "all_cells", "label"], "dct")
            ax = fig.add_subplot(gs[r, j], sharey=first)
            first = first or ax
            x = np.arange(len(order))
            for k, cutoff in enumerate(CUTOFFS):
                dd = d[d["cutoff"] == cutoff].set_index("label")
                if dd.empty:
                    continue
                ax.bar(x + (k - 1.5) * w, dd.loc[order, "percent_of_label_removed"].to_numpy(float), width=w * 0.92,
                       color=t3.SERIES[k])
                ax.axhline(float(dd.loc["all_cells", "percent_of_label_removed"]), color=t3.SERIES[k], linewidth=0.6,
                           linestyle=(0, (3, 2)))
            n = d[d["cutoff"] == CUTOFFS[0]].set_index("label")["n_cells"]
            ax.set_xticks(x, [f"{lab if lab != 'Unknown' else 'Unk.'}\n{int(n[lab])}" for lab in order])
            ax.set_xlim(-0.6, max(len(order), 3) - 0.4)
            ax.set_title(f"{cond}, {lib.ROLE_DISPLAY[role]}", fontsize=7)
            lib.grid_y(ax)
            if j == 0:
                ax.set_ylabel("% of label removed")
            else:
                ax.tick_params(labelleft=False)
    handles = [Patch(color=t3.SERIES[k], label=CUTOFF_DISPLAY[c]) for k, c in enumerate(CUTOFFS)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.925), ncol=4, frameon=False,
               handletextpad=0.3, columnspacing=1.0)
    fig.text(0.005, 0.995, "Cells removed by cutoffs on the MitoCarta fraction of summed intensity (high side), per "
             "condition; simulation only, no cell excluded\ndashed: % of all cells removed; within-label rule evaluates "
             f"labels with >= {t3.MIN_GROUP} cells; tick: label and cells; Unk. = Unknown", ha="left", va="top",
             fontsize=7)
    figs.save(fig, "mito_global_cutoff", "Global vs within-label mito cutoffs, Ctrl and ZeroK")


def fig_within(wt: pd.DataFrame, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 170)
    panels = [(dv, role) for dv in DEPTH_VARS for role in lib.ROLES]
    gs = fig.add_gridspec(len(panels), 2, left=0.13, right=0.97, top=0.9, bottom=0.07, hspace=0.6, wspace=0.32)
    for i, (depth_var, role) in enumerate(panels):
        for j, cond in enumerate(lib.CONDITIONS):
            col = lib.SETTINGS[cond].COLUMN_OF[role]
            ax = fig.add_subplot(gs[i, j])
            d = wt[(wt["condition"] == cond) & (wt["label_column"] == col) & (wt["depth_variable"] == depth_var)]
            labels = [lab for lab in ["all_cells", *t3.label_order(d.loc[d["label"] != "all_cells", "label"], "dct")]
                      if d.loc[d["label"] == lab, "rho"].notna().any()]
            y = np.arange(len(labels))
            for k, metric in enumerate(t3.MITO_METRICS):
                dd = d[d["metric"] == metric].set_index("label").loc[labels]
                ax.errorbar(dd["rho"], y + (k - 1) * 0.22, xerr=[dd["rho"] - dd["ci_low"], dd["ci_high"] - dd["rho"]],
                            fmt="o", markersize=2.6, color=t3.SERIES[k], elinewidth=0.7, capsize=0)
            ax.axvline(0, color=t2.INK["muted"], linewidth=0.6)
            n = d[d["metric"] == t3.MITO_METRICS[0]].set_index("label")["n_cells"]
            ax.set_yticks(y, [f"{'all cells' if lab == 'all_cells' else lab} ({int(n[lab])})" for lab in labels])
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.set_xlim(-1, 1)
            ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_title(f"{cond}, {role} label vs {DEPTH_DISPLAY[depth_var]}", fontsize=7)
            if i == len(panels) - 1:
                ax.set_xlabel("Spearman rho (95% CI)")
            else:
                ax.tick_params(labelbottom=False)
    handles = [Line2D([], [], marker="o", linestyle="", color=t3.SERIES[k], markersize=4, label=SHORT_METRIC[m])
               for k, m in enumerate(t3.MITO_METRICS)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.955), ncol=3, frameon=False)
    fig.text(0.005, 0.995, f"Mitochondrial fractions vs depth within each label (labels with >= {t3.MIN_GROUP} cells), "
             "per condition\n(Fisher-z interval, SE = 1.06/sqrt(n-3); tick: label and cells)", ha="left", va="top",
             fontsize=7)
    figs.save(fig, "mito_within_type_depth_correlation", "Within-label Spearman of mito fractions with depth")


def fig_removed(per_cell: pd.DataFrame, figs: lib.Figures) -> None:
    metric = "mitocarta_intensity_fraction"
    fig = lib.new_figure(lib.FULL_MM, 140)
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.99, top=0.9, bottom=0.08, hspace=0.75, wspace=0.25,
                          width_ratios=[1.6, 1])
    for row, cond in enumerate(lib.CONDITIONS):
        d = per_cell[per_cell["condition"] == cond]
        ids = d["n_detected_proteins"].to_numpy(float)
        x = d[metric].to_numpy(float)
        ret = d["retained"].astype(str).str.lower().eq("true").to_numpy()
        ok = ids > 0
        ax = fig.add_subplot(gs[row, 0])
        ax.scatter(ids[ok & ret], x[ok & ret], s=2.5, color=t2.INK["muted"], linewidths=0, rasterized=True,
                   label=f"retained ({int(ret.sum())})")
        ax.scatter(ids[ok & ~ret], x[ok & ~ret], s=9, color=t3.SERIES[1], linewidths=0.3, edgecolors="white",
                   label=f"removed, < 100 IDs ({int((~ret).sum())}; {int((~ret & ~ok).sum())} with 0 IDs not shown)")
        ax.axvline(100, color=t2.INK["muted"], linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlabel("Protein IDs per cell (all protein rows; log scale)")
        ax.set_ylabel("MitoCarta, intensity")
        ax.set_title(f"{cond}: all {len(d)} raw cells", fontsize=7)
        ax.set_ylim(-0.03, np.nanmax(x[ok]) * 1.45)
        ax.legend(loc="upper left", frameon=False, markerscale=1.6, handletextpad=0.2)
        lib.grid_y(ax)
        ax = fig.add_subplot(gs[row, 1])
        low_cut = np.quantile(ids[ret], 0.10)
        groups = [("removed", ~ret & ok, t3.SERIES[1]), ("retained,\nlowest 10%", ret & (ids <= low_cut), t3.SERIES[0]),
                  ("retained,\nall", ret, t2.INK["secondary"])]
        rng = np.random.default_rng(lib.SEED)
        for i, (name, mask, color) in enumerate(groups):
            vals = x[mask & np.isfinite(x)]
            bp = ax.boxplot(vals, positions=[i], widths=0.55, showfliers=False, patch_artist=True)
            bp["boxes"][0].set_facecolor("none")
            bp["boxes"][0].set_edgecolor(t2.INK["secondary"])
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set_color(t2.INK["secondary"])
            bp["medians"][0].set_color(t2.INK["primary"])
            ax.scatter(i + rng.uniform(-0.2, 0.2, vals.size), vals, s=2.5, color=color, alpha=0.6, linewidths=0,
                       rasterized=True)
        ax.set_xticks(range(len(groups)), [f"{g[0]}\n{int((g[1] & np.isfinite(x)).sum())}" for g in groups])
        ax.set_title(f"{cond}: lowest 10% = <= {low_cut:.0f} IDs", fontsize=7)
        lib.grid_y(ax)
    fig.text(0.005, 0.995, "Cells removed by the >= 100-ID filter: MitoCarta fraction of summed intensity, per condition"
             "\n(metrics over all protein rows of the combined export; removed cells have few IDs, so fractions are "
             "noisy; tick: group and cells)", ha="left", va="top", fontsize=7)
    figs.save(fig, "mito_removed_vs_retained", "Removed vs retained cells, MitoCarta fraction, Ctrl and ZeroK")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "mito_tests", args)
    lib.ported_from(run, "ctrl_reviewer_core_mito_tests.py")
    frames, checks = read_frames(run)
    mitocarta = t3.read_mitocarta(run)
    per_cell, removed, d_checks = removed_cells(run, mitocarta, frames)
    checks = lib.check_rows_frame(checks + d_checks)
    lib.stop_if_failed(run, checks, "mito_tests_checks")

    ve, sim, tests, wt = {}, {}, {}, {}
    for cond in lib.CONDITIONS:
        with lib.bound(lib.part2_settings(cond), m19t):
            data = {"dct": frames[cond]}
            ve[cond] = m19t.variance_explained(data)
            sim[cond], tests[cond] = m19t.global_cutoffs(data)
            wt[cond] = m19t.within_type(data)
    ve, sim, tests, wt = (lib.stack(x) for x in (ve, sim, tests, wt))
    run.table(ve, "mito_variance_explained")
    run.table(sim, "mito_global_cutoff_simulation")
    run.table(tests, "mito_global_cutoff_tests")
    run.table(wt, "mito_within_type_depth_correlation")
    run.table(per_cell, "mito_all_raw_cells")
    run.table(removed, "mito_removed_vs_retained")
    run.note("Tests A-C use the step 9a per-cell metrics of the 729 Ctrl and 702 ZeroK cells; each condition on its own. "
             "B permutation: a new default_rng(42) per condition (ctrl_reviewer_core call per dataset). BH in C within condition x "
             "label column x metric x depth variable, over labels.")

    figs = lib.Figures(run, "mito_tests")
    fig_variance(ve, figs)
    fig_cutoffs(sim, figs)
    fig_within(wt, figs)
    fig_removed(per_cell, figs)
    figs.finish()
    run.finish({"depth_variables": list(DEPTH_VARS), "cutoffs": list(CUTOFFS), "n_permutations": N_PERM,
                "min_group": t3.MIN_GROUP, "n_mad": t3.N_MAD, "permutation_rng": "numpy default_rng(42) per condition",
                "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_mito_tests.csv.")


if __name__ == "__main__":
    main()
