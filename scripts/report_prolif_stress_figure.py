#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S20 (dissociation-stress proteins and prevalence of ProLIF cells).

. One reader-facing figure with panel letters, drawn from stored tables only (no new
statistics here): the the response to the reviewers tables written by report_prolif_stress_tabulation.py and the stored dct_reviewer
contrasts of the marker-score label. Control and zero-K are shown side by side, each analysed on its own; nothing
compares the conditions.

  S20  (A) cells with each quantified stress-panel protein detected, by group; (B) stress score by group, for the
          groups of Figure 4e and for the ProLIF cells of the marker-score label; (C) stress score against proteins
          identified per cell; (D) ProLIF minus reference cells, difference in stress score, unadjusted and adjusted
          for proteins per cell and sort plate (95% CI); (E) cells counted as proliferating under three definitions,
          all cells and by quartile of proteins per cell, with the Mouse Kidney Atlas DCT class (Wilson 95% CI).

Panels D and E were added inside the approved figure (the plan named A to C); recorded in the list of items.

Reader-facing names (list of items, decisions of 12 Nov 2026): "Control" / "Zero-K"; "not resolved" for the label value
Unknown of the cluster annotation; "marker-score label" for cs_<cond>_cluster_label; no task code, file or column
name, and no word for the revision process in the figure (checked).
Names of the paper (step 33, 20 Nov 2026): the groups of panel B are those of a main-figure panel. That panel is
written here by its label (Figure 4e); the name it has in the paper is looked up in the numbering map through
report_item_names, for the words under panel B and for the series names of the values table. A check stops
the script if a drawn text or a series name still holds a label.
Format (dotplot / dct_labels): Arial >= 7 pt, exact 180 mm width, PDF + 600-dpi PNG + TIFF, no in-figure title or footer.
Helpers (axes in mm, violins, tick lines, legend, checks) are imported from report_cell_cycle_figures.py.

Inputs (read-only): results/tables/report/report_14_prolif_stress_{per_cell,panel_detection,contrasts,
  prevalence}.csv; results/tables/dct_reviewer/dct_reviewer_stress_prolif_contrasts.csv
Outputs: results/figures/report/report_14_S14a_stress_prevalence (.pdf, .png, .tif);
  results/tables/report/report_14_s14a_figure_{values,checks}.csv;
  results/tables/report/report_step_14_s14a_figure_manifest.json

Command (project root):  python scripts/report_prolif_stress_figure.py --overwrite
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
import report_cell_cycle_figures as s12
import report_item_names as pn
import dct_reviewer_lib as lib

lib.style()
import matplotlib.pyplot as plt  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

TASK = "report"
STEP = "14_s14a_figure"
DCT_REVIEWER = "dct_reviewer"
W = s12.W
CONDS = s12.CONDS
COND_DISPLAY = s12.COND_DISPLAY
COL = s12.COL
INK, DARK = s12.INK, s12.DARK
GROUPS = s12.GROUPS
GROUP_DISPLAY = s12.GROUP_DISPLAY
PANEL_ORDER = ("Hspa8", "Hsp90aa1", "Hspa1b", "Hsph1", "Dnajb1", "Hspb1", "Fos")
PANEL_DISPLAY = {"Hspa1b": "Hspa1a/b"}
MODEL_ADJ = "label_plus_log10_ids_plus_plate"
# Numbers of the the response to the reviewers letter (section 2) that the figure shows.
LETTER = {
    "Ctrl": {"n_call": 37, "n_score": 119, "n_cells": 729, "n_panel": 7, "adj_call_all": 0.006, "adj_call_dct": 0.027,
             "adj_score": -0.069, "either": 21, "q1": (2, 183)},
    "ZeroK": {"n_call": 63, "n_score": 98, "n_cells": 702, "n_panel": 6, "adj_call_all": -0.011,
              "adj_call_dct": -0.014, "adj_score": 0.065, "either": 56, "q1": (5, 176)},
}
ATLAS = (217, 16381)
# Panel B: the groups of a main-figure panel. Written with the label of that panel; drawn with its name in the paper.
GROUPS_OF_LABEL = "groups of Figure 3A"
GROUPS_OF = pn.paper_names(GROUPS_OF_LABEL)


def figure(run, checks, figs, values, per_cell, detection, contrasts, stored, prevalence) -> None:
    h_mm = 222.0
    col_x, ax_w = {"Ctrl": 16.0, "ZeroK": 104.0}, 70.0
    rows = {"A": (188.0, 22.0), "B": (144.0, 24.0), "C": (98.0, 25.0), "D": (61.0, 20.0), "E": (22.0, 22.0)}
    fig = lib.new_figure(W, h_mm)
    for cond in CONDS:
        fig.text((col_x[cond] + ax_w / 2) / W, 218.5 / h_mm, COND_DISPLAY[cond], fontsize=8, fontweight="bold",
                 ha="center", va="baseline")
        pc = per_cell[per_cell["condition"] == cond]
        exp = LETTER[cond]
        call = pc["is_prolif"].astype(str).str.lower().eq("true")
        score = pc["second_label"].eq("ProLIF")

        # A: cells with each stress-panel protein detected
        y0, ah = rows["A"]
        ax = s12.mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        det = detection[detection["condition"] == cond]
        series = [("groups_of_figure_3a", g, COL[g], "o", GROUP_DISPLAY[g]) for g in GROUPS]
        series.append(("marker_score_label_clusters", "ProLIF", COL["score"], "D", "ProLIF, marker-score label"))
        offsets = np.linspace(-0.3, 0.3, len(series))
        plotted = set()
        for (part, grp, colour, marker, name), dx in zip(series, offsets):
            d = det[(det["partition"] == part) & (det["group"] == grp)].set_index("protein")
            for i, protein in enumerate(PANEL_ORDER):
                if protein not in d.index:
                    continue
                plotted.add(protein)
                ax.plot(i + dx, d.loc[protein, "percent_detected"], linestyle="none", marker=marker,
                        markersize=3.2 if marker == "D" else 3.6, color=colour, zorder=3)
                values.append({"figure": "S-14-a", "panel": "A", "condition": cond, "series": name,
                               "x": PANEL_DISPLAY.get(protein, protein), "n": int(d.loc[protein, "n_cells"]),
                               "value": float(d.loc[protein, "percent_detected"]),
                               "value_name": "% of cells with the protein detected (n = cells of the group)",
                               "source": "report_14_prolif_stress_panel_detection.csv"})
        ax.set_xlim(-0.6, 6.6)
        ax.set_ylim(-4, 104)
        ax.set_yticks([0, 50, 100])
        ax.set_xticks(range(7), [PANEL_DISPLAY.get(p, p) for p in PANEL_ORDER], rotation=90)
        if cond == "Ctrl":
            ax.set_ylabel("% of cells")
        checks.add(f"S-14-a A, {cond}: quantified stress-panel protein groups plotted",
                   f"{len(plotted)}: {', '.join(p for p in PANEL_ORDER if p in plotted)}",
                   len(plotted) == exp["n_panel"], str(exp["n_panel"]))

        # B: stress score by group
        y0, ah = rows["B"]
        ax = s12.mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        sets = [(GROUP_DISPLAY[g], pc.loc[pc["group"] == g, "stress_mean_z"], COL[g]) for g in GROUPS]
        sets += [("ProLIF", pc.loc[score, "stress_mean_z"], COL["score"]),
                 ("other cells", pc.loc[~score, "stress_mean_z"], COL["other"])]
        positions = (0, 1, 2, 3.4, 5.3, 6.7)
        for pos, (name, v, colour) in zip(positions, sets):
            s12.violin(ax, pos, v, colour)
            part = GROUPS_OF if pos < 4.5 else "marker-score label"
            values.append({"figure": "S-14-a", "panel": "B", "condition": cond, "series": f"{name}, {part}",
                           "x": pos, "n": int(len(v)), "value": float(np.median(v)) if len(v) else np.nan,
                           "value_name": "median stress score", "source": "report_14_prolif_stress_per_cell.csv"})
        ax.set_xlim(-0.6, 7.4)
        ax.set_ylim(-2.6, 2.9)
        ax.set_yticks([-2, 0, 2])
        s12.tick_lines(ax, ah, positions, [(n, f"({len(v)})") for n, v, _ in sets])
        trans = ax.get_xaxis_transform()
        for x0, x1, text in ((-0.4, 4.2, GROUPS_OF), (4.9, 7.3, "marker-score label")):
            ax.plot([x0, x1], [-8.2 / ah] * 2, transform=trans, color=DARK, linewidth=0.6, clip_on=False)
            ax.text((x0 + x1) / 2, -9.2 / ah, text, transform=trans, ha="center", va="top", fontsize=7)
        if cond == "Ctrl":
            ax.set_ylabel("stress score")
        checks.add(f"S-14-a B, {cond}: ProLIF cells of the per-cell test and of the marker-score label = the letter",
                   f"{int(call.sum())}; {int(score.sum())}",
                   (int(call.sum()), int(score.sum())) == (exp["n_call"], exp["n_score"]))

        # C: stress score against proteins per cell
        y0, ah = rows["C"]
        ax = s12.mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        for grp in ("Unknown", "DCT1", "DCT2", "ProLIF"):
            s = pc[pc["group"] == grp]
            ax.scatter(s["n_detected"], s["stress_mean_z"], s=10 if grp == "ProLIF" else 3, color=COL[grp],
                       linewidths=0, zorder=4 if grp == "ProLIF" else 2, rasterized=True)
            values.append({"figure": "S-14-a", "panel": "C", "condition": cond, "series": GROUP_DISPLAY[grp],
                           "n": int(len(s)), "value": float(s["n_detected"].median()) if len(s) else np.nan,
                           "value_name": "cells plotted (n); median proteins per cell",
                           "source": "report_14_prolif_stress_per_cell.csv"})
        ax.set_xscale("log")
        ax.set_xlim(90, 5500)
        ax.xaxis.set_major_locator(FixedLocator([100, 300, 1000, 3000]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["100", "300", "1,000", "3,000"])
        ax.xaxis.grid(False)
        ax.set_ylim(-2.6, 2.9)
        ax.set_yticks([-2, 0, 2])
        ax.set_xlabel("proteins identified per cell")
        if cond == "Ctrl":
            ax.set_ylabel("stress score")
        checks.add(f"S-14-a C, {cond}: every cell plotted", f"{len(pc)}", len(pc) == exp["n_cells"])

        # D: ProLIF minus reference cells, difference in stress score
        y0, ah = rows["D"]
        ax = s12.mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        con = contrasts[(contrasts["condition"] == cond) & (contrasts["family"] == "planned")
                        & (contrasts["metric"] == "stress_mean_z")].set_index("comparison")
        st = stored[(stored["condition"] == cond) & (stored["label_role"] == "second")
                    & (stored["comparison"] == "ProLIF_vs_DCT1+DCT2") & (stored["metric"] == "stress_mean_z")].iloc[0]
        items = ((("per-cell test", "all other cells"), con.loc["ProLIF_vs_all_other_cells"], COL["ProLIF"],
                  "report_14_prolif_stress_contrasts.csv"),
                 (("per-cell test", "DCT1 and DCT2" if cond == "Ctrl" else "DCT1"), con.loc["ProLIF_vs_DCT1_DCT2"],
                  COL["ProLIF"], "report_14_prolif_stress_contrasts.csv"),
                 (("marker-score label", "DCT1 and DCT2"), st, COL["score"], "dct_reviewer_stress_prolif_contrasts.csv"))
        got = {}
        for i, (lines, r, colour, source) in enumerate(items):
            for model, dx, filled in (("label_only", -0.13, False), (MODEL_ADJ, 0.13, True)):
                coef, lo, hi = (float(r[f"{model}_{k}"]) for k in ("coef", "ci_low", "ci_high"))
                ax.vlines(i + dx, lo, hi, color=colour, linewidth=0.9, zorder=2)
                ax.plot(i + dx, coef, linestyle="none", marker="o", markersize=4, markeredgewidth=0.8,
                        markeredgecolor=colour, markerfacecolor=colour if filled else "white", zorder=3)
                values.append({"figure": "S-14-a", "panel": "D", "condition": cond,
                               "series": f"ProLIF ({lines[0]}) minus {lines[1]}",
                               "x": "adjusted for proteins per cell and sort plate" if filled else "unadjusted",
                               "n": int(r["n_group"]), "value": coef, "value_name": "difference in stress score",
                               "reference": f"{lo:.4f} to {hi:.4f} (95% CI); reference cells {int(r['n_reference'])}",
                               "source": source})
                if filled:
                    got[i] = coef
        ax.axhline(0, color=DARK, linewidth=0.6, zorder=1)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1.3, 0.5)
        ax.set_yticks([-1.0, -0.5, 0, 0.5])
        s12.tick_lines(ax, ah, range(3), [(lines[0], "vs " + lines[1]) for lines, *_ in items])
        if cond == "Ctrl":
            ax.set_ylabel("difference")
        checks.add(f"S-14-a D, {cond}: adjusted difference in stress score, ProLIF of the per-cell test against all "
                   "other cells and against DCT1 (and DCT2) cells; ProLIF of the marker-score label = the letter",
                   "; ".join(f"{got[i]:.3f}" for i in range(3)),
                   [round(got[i], 3) for i in range(3)] == [exp["adj_call_all"], exp["adj_call_dct"], exp["adj_score"]],
                   f"{exp['adj_call_all']}; {exp['adj_call_dct']}; {exp['adj_score']}")

        # E: cells counted as proliferating, by definition
        y0, ah = rows["E"]
        ax = s12.mm_axes(fig, h_mm, col_x[cond], y0, ax_w, ah)
        pv = prevalence[prevalence["condition"] == cond]
        quart = sorted(s for s in pv["subset"].unique() if s.startswith("protein IDs Q"))
        blocks = (("per-cell test: ProLIF (cycling, NCC detected)", COL["ProLIF"], 0.0, True),
                  ("Mki67 or Top2a detected", INK, 5.6, True),
                  ("marker-score label (clusters): ProLIF", COL["score"], 11.2, False))
        ticks, labels = [], []
        for definition, colour, x0, with_quartiles in blocks:
            subsets = ["all cells", *quart] if with_quartiles else ["all cells"]
            for k, subset in enumerate(subsets):
                r = pv[(pv["definition"] == definition) & (pv["subset"] == subset)].iloc[0]
                x = x0 + k
                ax.vlines(x, r["ci_low_percent"], r["ci_high_percent"], color=colour, linewidth=0.9, zorder=2)
                ax.plot(x, r["percent"], linestyle="none", marker="o", markersize=3.8, color=colour, zorder=3)
                ticks.append(x)
                labels.append("all" if subset == "all cells" else subset.split()[2].rstrip(":"))
                values.append({"figure": "S-14-a", "panel": "E", "condition": cond, "series": definition, "x": subset,
                               "n": int(r["n_cells"]), "value": int(r["k"]),
                               "value_name": "cells (n = cells of the subset)",
                               "reference": f"{r['percent']:.2f}% ({r['ci_low_percent']:.2f} to "
                                            f"{r['ci_high_percent']:.2f}, Wilson 95% CI)",
                               "source": "report_14_prolif_stress_prevalence.csv"})
        a = prevalence[(prevalence["condition"] == "Mouse Kidney Atlas (RNA)")
                       & (prevalence["definition"] == "Mki67 or Top2a transcripts detected")
                       & (prevalence["subset"] == "atlas DCT class, cells and nuclei")].iloc[0]
        x_atlas = 13.0
        ax.vlines(x_atlas, a["ci_low_percent"], a["ci_high_percent"], color=DARK, linewidth=0.9, zorder=2)
        ax.plot(x_atlas, a["percent"], linestyle="none", marker="s", markersize=3.6, color=DARK, zorder=3)
        values.append({"figure": "S-14-a", "panel": "E", "condition": cond, "series": "Mouse Kidney Atlas, DCT class",
                       "x": "Mki67 or Top2a transcripts", "n": int(a["n_cells"]), "value": int(a["k"]),
                       "value_name": "cells and nuclei (n = cells and nuclei of the class)",
                       "reference": f"{a['percent']:.2f}% ({a['ci_low_percent']:.2f} to {a['ci_high_percent']:.2f}, "
                                    "Wilson 95% CI)", "source": "report_14_prolif_stress_prevalence.csv"})
        ax.set_xlim(-0.8, 13.8)
        ax.set_ylim(0, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_xticks(ticks, labels)
        ax.set_xticks([11.2, x_atlas], ["", ""], minor=True)
        trans = ax.get_xaxis_transform()
        for x0, x1, lines in ((-0.4, 4.4, ("per-cell test",)), (5.2, 10.0, ("Mki67 or Top2a",)),
                              (10.6, 11.8, ("marker", "score")), (12.4, 13.6, ("atlas", "RNA"))):
            ax.plot([x0, x1], [-5.2 / ah] * 2, transform=trans, color=DARK, linewidth=0.6, clip_on=False)
            for k, text in enumerate(lines):
                ax.text((x0 + x1) / 2, -(6.2 + 3.0 * k) / ah, text, transform=trans, ha="center", va="top", fontsize=7)
        if cond == "Ctrl":
            ax.set_ylabel("% of cells")
        r_call = pv[(pv["definition"] == blocks[0][0]) & (pv["subset"] == "all cells")].iloc[0]
        r_either = pv[(pv["definition"] == blocks[1][0]) & (pv["subset"] == "all cells")].iloc[0]
        r_score = pv[(pv["definition"] == blocks[2][0]) & (pv["subset"] == "all cells")].iloc[0]
        r_q1 = pv[(pv["definition"] == blocks[0][0]) & (pv["subset"] == quart[0])].iloc[0]
        checks.add(f"S-14-a E, {cond}: cells per definition (per-cell test; Mki67 or Top2a; marker-score label), the "
                   "per-cell test in the quartile with the fewest proteins, and the atlas DCT class = the letter",
                   f"{int(r_call['k'])}; {int(r_either['k'])}; {int(r_score['k'])} of {int(r_call['n_cells'])}; "
                   f"{int(r_q1['k'])} of {int(r_q1['n_cells'])}; atlas {int(a['k'])} of {int(a['n_cells'])}",
                   (int(r_call["k"]), int(r_either["k"]), int(r_score["k"]), int(r_call["n_cells"]))
                   == (exp["n_call"], exp["either"], exp["n_score"], exp["n_cells"])
                   and (int(r_q1["k"]), int(r_q1["n_cells"])) == exp["q1"]
                   and (int(a["k"]), int(a["n_cells"])) == ATLAS)

    x0 = col_x["Ctrl"] - 11.0
    s12.letter_title(fig, h_mm, x0, 213.4, "A", "Stress-panel proteins: cells with the protein detected")
    s12.letter_title(fig, h_mm, x0, 171.4, "B", "Stress score by group (number of cells)")
    s12.letter_title(fig, h_mm, x0, 126.4, "C", "Stress score against the number of proteins identified per cell")
    s12.letter_title(fig, h_mm, x0, 84.4, "D", "ProLIF cells minus reference cells: difference in stress score (95% CI)")
    s12.letter_title(fig, h_mm, x0, 47.4, "E", "Cells counted as proliferating, all cells and by quartile of proteins "
                                              "per cell (Q1, fewest; 95% CI)")
    handles = [s12.dot(COL[g], GROUP_DISPLAY[g], size=3.6) for g in ("DCT1", "DCT2", "Unknown")]
    handles.append(s12.dot(COL["ProLIF"], "ProLIF (per-cell test)", size=4.2))
    handles.append(s12.dot(COL["score"], "ProLIF (marker-score label)", marker="D", size=3.4))
    s12.legend_row(fig, h_mm, col_x["Ctrl"], 5.6, handles)
    handles = [s12.dot(DARK, "unadjusted (D)", filled=False), s12.dot(DARK, "adjusted for proteins per cell and "
                                                                           "sort plate (D)"),
               Line2D([], [], marker="s", linestyle="none", markersize=3.6, color=DARK,
                      label="Mouse Kidney Atlas, DCT class (E)")]
    s12.legend_row(fig, h_mm, col_x["Ctrl"], 1.4, handles)
    drawn = [t.get_text() for t in s12.t27f.drawn_texts(fig)]
    series = [str(v["series"]) for v in values]
    left = pn.labels_left(drawn + series)
    checks.add("S-14-a: names of the paper: no working label left in the drawn texts or in the series names of the "
               "values table", f"{len(drawn)} drawn texts, {len(series)} series names; under panel B: "
               f"{GROUPS_OF!r} ({drawn.count(GROUPS_OF)} places); labels left: {', '.join(left) or 'none'}",
               not left and drawn.count(GROUPS_OF) == len(CONDS), "none left; one place per condition")
    s12.save(checks, figs, fig, "14_S14a_stress_prevalence",
             "S-14-a: dissociation-stress proteins and prevalence of ProLIF cells in the DCT data, control and zero-K: "
             "stress-panel protein detection (A), stress score by group (B) and against proteins per cell (C), "
             "ProLIF minus reference with and without adjustment (D), cells counted as proliferating by definition (E)")


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-14-a.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_combined")

    def table(task: str, name: str) -> pd.DataFrame:
        path = cc.out_path(task, "tables", name, ".csv")
        return pd.read_csv(run.use_input(path, f"{task}: {name}"), keep_default_na=False, na_values=[""])

    per_cell = table(TASK, "14_prolif_stress_per_cell")
    detection = table(TASK, "14_prolif_stress_panel_detection")
    contrasts = table(TASK, "14_prolif_stress_contrasts")
    prevalence = table(TASK, "14_prolif_stress_prevalence")
    stored = table(DCT_REVIEWER, "stress_prolif_contrasts")

    figs = lib.Figures(run, STEP)
    values: list[dict] = []
    figure(run, checks, figs, values, per_cell, detection, contrasts, stored, prevalence)
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"figures": ["14_S14a_stress_prevalence"], "width_mm": W, "colours": COL,
                "label_display": GROUP_DISPLAY, "panel_order": list(PANEL_ORDER), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
