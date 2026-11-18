#!/usr/bin/env python
"""dct_reviewer step 3: reviewer_core A.3 ALDO score sensitivity per DCT condition (combined search, dct_labels labels).

Port of ctrl_reviewer_core_aldo.py (unchanged method:
``reviewer_core_lib.aldo_scores`` and ``aldo_contrast_table``, ctrl_reviewer_core ``summarize``), run once per condition.
The expression source 'ctrl_X' of ctrl_reviewer_core is named 'scaled_X' (each condition's own scaled .X).

Inputs (read-only): results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_task3_scored.h5ad (step 1; scaled .X and
Task 3 flags), results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_cc_regressed.h5ad (step 2), dct_labels labels.
Outputs:
  results/tables/dct_reviewer/dct_reviewer_aldo_*.csv (both conditions; ``condition`` column)
  results/figures/dct_reviewer/dct_reviewer_aldo_contrasts_forest_<role>.pdf/.png/.tif
  results/figures/dct_reviewer/dct_reviewer_aldo_scores_<source>_<role>.pdf/.png/.tif

Score families (ALDO_activity, ALDO_signed, ALDO_machinery) x variants (base, no_Slc12a3,
no_cell_cycle, no_Slc12a3_no_cell_cycle) = mean protein z-score (ddof=0, within the condition) of the
UP panel minus the DOWN panel, on two sources: scaled_X and cc_regressed_3a. Contrasts DCT2 vs DCT1,
ProLIF vs DCT1 and ProLIF vs DCT2 (Mann-Whitney) in all cells and after excluding
TAL/CNT/interstitial-flagged cells. A contrast whose group or reference has no cell under a column
'does not apply' (status column); BH-FDR within condition x label column x source covers the tested
contrasts only (``cc.add_bh`` skips NaN p-values), so the family is smaller than in ctrl_reviewer_core where labels
are absent.

Command (project root):  python scripts/dct_reviewer_aldo.py
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
import ctrl_reviewer_core_aldo as m19
import dct_reviewer_lib as lib

lib.style()
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "dct_reviewer_aldo.py"
SOURCES = ("scaled_X", "cc_regressed_3a")
SOURCE_DISPLAY = {"scaled_X": "scaled .X", "cc_regressed_3a": "cc-regressed"}
COMPARISON_COLORS = m19.COMPARISON_COLORS
COHORTS = ("all_cells", "exclude_TAL_CNT_interstitial_flagged")
COHORT_DISPLAY = {"all_cells": "all cells", "exclude_TAL_CNT_interstitial_flagged": "flagged cells excluded"}
FAMILY_DISPLAY = {"ALDO_activity": "activity", "ALDO_signed": "signed", "ALDO_machinery": "machinery"}
VARIANT_DISPLAY = {"base": "base", "no_Slc12a3": "no Slc12a3", "no_cell_cycle": "no cell cycle",
                   "no_Slc12a3_no_cell_cycle": "no Slc12a3 or cc"}
ORDER = list(t2.CORE_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 3: ALDO score sensitivity per DCT condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str) -> dict:
    s = lib.SETTINGS[cond]
    adata = lib.read_h5ad(run, f"DCT_{s.KEY}_task3_scored")
    regressed = lib.read_h5ad(run, f"DCT_{s.KEY}_cc_regressed")
    if list(regressed.obs_names) != list(adata.obs_names) or list(regressed.var_names) != list(adata.var_names):
        raise ValueError(f"{cond}: regressed object does not share cells/proteins with the A.1 object.")
    _, labels = lib.load(run, cond)
    if list(labels.index) != list(adata.obs_names.astype(str)):
        raise ValueError(f"{cond}: label and object cell order differ.")

    per_cell = labels[["cell_key", "plate", *s.LABEL_COLUMNS]].copy()
    for col in t2.FLAG_COLUMNS:
        per_cell[col] = adata.obs[col].astype(bool).to_numpy()
    per_cell["exclude_tal_cnt_interstitial"] = per_cell[list(t2.FLAG_COLUMNS)].any(axis=1)
    frames = [t2.aldo_scores(adata, "scaled_X"), t2.aldo_scores(regressed, "cc_regressed_3a")]
    per_cell = per_cell.join(pd.concat(frames, axis=1))
    score_cols = [c for f in frames for c in f.columns]
    stored = [name for name in t2.ALDO_SUBMISSION_SCORES if name in adata.obs]
    for name in stored:
        per_cell[f"submission__{name}"] = adata.obs[name].to_numpy(float)
        score_cols.append(f"submission__{name}")

    summary = pd.concat([m19.summarize(per_cell, col, score_cols) for col in s.LABEL_COLUMNS], ignore_index=True)
    contrast = pd.concat([t2.aldo_contrast_table(per_cell, col, score_cols) for col in s.LABEL_COLUMNS],
                         ignore_index=True)
    tested = (contrast["n_group"] > 0) & (contrast["n_reference"] > 0)
    contrast["status"] = np.where(tested, "tested", lib.NOT_APPLICABLE)
    contrast["reason"] = ""
    for idx in contrast.index[~tested]:
        row = contrast.loc[idx]
        group, reference = row["comparison"].split("_vs_")
        missing = [l for l, n in ((group, row["n_group"]), (reference, row["n_reference"])) if n == 0]
        contrast.loc[idx, "reason"] = f"no {' or '.join(missing)} cell under {row['label_column']} ({row['cohort']})"
    return {"coverage": t2.aldo_coverage(adata.var_names), "per_cell": lib.to_generic(per_cell, cond).reset_index(),
            "summary": summary, "contrast": contrast, "stored": stored, "_per_cell": per_cell}


def key_order() -> list[tuple[str, str]]:
    return [(f, v) for f in t2.ALDO_FAMILIES for v in t2.ALDO_VARIANTS]


def plot_forest(results: dict, role: str, figs: lib.Figures) -> None:
    keys = key_order()
    ylabels = [f"{FAMILY_DISPLAY[f]}, {VARIANT_DISPLAY[v]}" for f, v in keys]
    y = np.arange(len(keys))
    tested = pd.concat([results[c]["contrast"].query("status == 'tested'").assign(condition=c) for c in lib.CONDITIONS])
    tested = tested[tested["label_column"].isin([lib.SETTINGS[c].COLUMN_OF[role] for c in lib.CONDITIONS])
                    & tested["source"].isin(SOURCES)]
    lo, hi = tested["median_difference"].min(), tested["median_difference"].max()
    pad = 0.06 * (hi - lo) if np.isfinite(hi - lo) and hi > lo else 0.1
    xlim = (min(lo, 0) - pad, max(hi, 0) + pad)

    fig = lib.new_figure(lib.FULL_MM, 132)
    gs = fig.add_gridspec(2, 4, left=0.205, right=0.99, top=0.8, bottom=0.17, hspace=0.42, wspace=0.12)
    for r, cohort in enumerate(COHORTS):
        for k, (source, cond) in enumerate([(src, c) for src in SOURCES for c in lib.CONDITIONS]):
            ax = fig.add_subplot(gs[r, k])
            col = lib.SETTINGS[cond].COLUMN_OF[role]
            con = results[cond]["contrast"]
            part = con[(con["label_column"] == col) & (con["cohort"] == cohort) & (con["source"] == source)]
            title = f"{cond}, {SOURCE_DISPLAY[source]}\n{COHORT_DISPLAY[cohort]}"
            if not (part["status"] == "tested").any():
                lib.na_axis(ax, f"{title}\n\n{lib.NOT_APPLICABLE}:\nno label pair with cells\nunder {col}")
                continue
            for offset, comparison in zip((-0.25, 0, 0.25), COMPARISON_COLORS):
                p = (part[(part["comparison"] == comparison) & (part["status"] == "tested")]
                     .set_index(["family", "variant"]).reindex(keys))
                yy = y + offset
                sig = (p["p_fdr_bh"] < 0.05).to_numpy()
                color = COMPARISON_COLORS[comparison]
                x = p["median_difference"].to_numpy(float)
                ax.scatter(x[sig], yy[sig], color=color, s=10, linewidths=0)
                ax.scatter(x[~sig], yy[~sig], facecolors="white", edgecolors=color, s=10, linewidths=0.8)
            ax.axvline(0, color=t2.INK["muted"], linewidth=0.6)
            ax.set_yticks(y, ylabels if k == 0 else [""] * len(y))
            ax.set_ylim(len(keys) - 0.5, -0.5)
            ax.set_xlim(*xlim)
            ax.set_title(title, fontsize=7)
            ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            if r == 1:
                ax.set_xlabel("median difference")
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=4, label=k.replace("_", " "))
               for k, c in COMPARISON_COLORS.items()]
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#52514e",
                          markersize=4, label="open: BH q >= 0.05"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.2, 0.0), ncol=4, frameon=False)
    lines = [f"ALDO contrasts by {lib.ROLE_DISPLAY[role]} (median difference, group minus reference)",
             "filled: BH q < 0.05 within condition x label column x source; cells per label:"]
    for cond in lib.CONDITIONS:
        pc = results[cond]["_per_cell"]
        col = lib.SETTINGS[cond].COLUMN_OF[role]
        keep = ~pc["exclude_tal_cnt_interstitial"].to_numpy(bool)
        lines.append(f"{cond} ({col}): " + ", ".join(f"{l} {int((pc[col] == l).sum())}" for l in ORDER)
                     + "; flagged excluded: " + ", ".join(f"{l} {int(((pc[col] == l) & keep).sum())}" for l in ORDER))
    fig.text(0.005, 0.995, "\n".join(lines), ha="left", va="top", fontsize=7)
    figs.save(fig, f"aldo_contrasts_forest_{role}", f"ALDO contrasts, {role} label, Ctrl and ZeroK", label_role=role)


def plot_boxes(results: dict, role: str, source: str, figs: lib.Figures) -> None:
    variants = ["base", "no_Slc12a3"]
    fig = lib.new_figure(lib.FULL_MM, 160)
    gs = fig.add_gridspec(len(t2.ALDO_FAMILIES), 4, left=0.085, right=0.99, top=0.9, bottom=0.075, hspace=0.75,
                          wspace=0.12)
    panels = [(c, cohort) for c in lib.CONDITIONS for cohort in COHORTS]
    for r, family in enumerate(t2.ALDO_FAMILIES):
        first = None
        for k, (cond, cohort) in enumerate(panels):
            ax = fig.add_subplot(gs[r, k], sharey=first)
            first = first or ax
            pc = results[cond]["_per_cell"]
            col = lib.SETTINGS[cond].COLUMN_OF[role]
            keep = (np.ones(len(pc), bool) if cohort == "all_cells"
                    else ~pc["exclude_tal_cnt_interstitial"].to_numpy(bool))
            counts = []
            for i, label in enumerate(ORDER):
                sel = keep & (pc[col] == label).to_numpy()
                counts.append(int(sel.sum()))
                for j, variant in enumerate(variants):
                    vals = pc.loc[sel, f"{source}__{family}__{variant}"].dropna().to_numpy(float)
                    if not vals.size:
                        continue
                    bp = ax.boxplot(vals, positions=[i + (j - 0.5) * 0.36], widths=0.3, showfliers=False,
                                    patch_artist=True)
                    for patch in bp["boxes"]:
                        patch.set_facecolor(t2.LABEL_PALETTE[label] if j == 0 else "white")
                        patch.set_edgecolor(t2.LABEL_PALETTE[label])
                    for part in ("medians", "whiskers", "caps"):
                        for line in bp[part]:
                            line.set_color(t2.INK["primary"] if part == "medians" else t2.LABEL_PALETTE[label])
                            line.set_linewidth(0.8)
            ax.set_xticks(range(len(ORDER)), [f"{l}\n{n}" for l, n in zip(ORDER, counts)])
            ax.set_xlim(-0.6, len(ORDER) - 0.4)
            ax.set_title(f"{FAMILY_DISPLAY[family]}: {cond}\n{COHORT_DISPLAY[cohort]}", fontsize=7)
            ax.yaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
            ax.set_axisbelow(True)
            if k == 0:
                ax.set_ylabel("UP minus DOWN mean z" if family == "ALDO_signed" else "mean protein z")
            else:
                ax.tick_params(labelleft=False)
    fig.text(0.005, 0.995, f"ALDO scores ({SOURCE_DISPLAY[source]}) by {lib.ROLE_DISPLAY[role]}; filled box = base, "
             "open box = no_Slc12a3;\nnumbers = cells (Unknown not drawn); z within each condition", ha="left",
             va="top", fontsize=7)
    figs.save(fig, f"aldo_scores_{source}_{role}", f"ALDO scores ({source}), {role} label, Ctrl and ZeroK",
              label_role=role)


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "aldo", args)
    lib.ported_from(run, "ctrl_reviewer_core_aldo.py")
    results = {c: one_condition(run, c) for c in lib.CONDITIONS}
    for cond in lib.CONDITIONS:
        stored = results[cond]["stored"]
        run.note(f"{cond}: stored submission ALDO columns in the object: {stored or 'none'}; the reviewer_core 'submission' rows "
                 + ("are included." if stored else "do not apply."))
    run.table(lib.stack({c: results[c]["coverage"] for c in lib.CONDITIONS}), "aldo_marker_coverage")
    run.table(lib.stack({c: results[c]["per_cell"] for c in lib.CONDITIONS}), "aldo_scores_per_cell")
    run.table(lib.stack({c: results[c]["summary"] for c in lib.CONDITIONS}), "aldo_summary_by_label")
    contrast = lib.stack({c: results[c]["contrast"] for c in lib.CONDITIONS})
    run.table(contrast, "aldo_contrasts")
    headline = contrast[contrast["variant"].isin(["base", "no_Slc12a3", "submission"])]
    run.table(headline[["condition", "label_role", "label_column", "source", "family", "variant", "cohort", "comparison",
                        "status", "n_group", "n_reference", "median_group", "median_reference", "median_difference",
                        "cohens_d", "p_value", "p_fdr_bh"]], "aldo_contrasts_headline")
    run.note("BH-FDR within condition x label column x expression source, across cohorts, comparisons and scores; "
             "contrasts that do not apply (absent label) have NaN p and are not in the BH family.")
    families = (contrast[contrast["status"] == "tested"].groupby(["condition", "label_column", "source"]).size()
                .rename("n_tests_in_bh_family").reset_index())
    run.table(families, "aldo_bh_family_sizes")

    figs = lib.Figures(run, "aldo")
    for role in lib.ROLES:
        for source in SOURCES:
            plot_boxes(results, role, source, figs)
        plot_forest(results, role, figs)
    figs.finish()
    run.finish({"pairs": t2.ALDO_PAIRS, "expression_sources": list(SOURCES), "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_aldo.csv.")


if __name__ == "__main__":
    main()
