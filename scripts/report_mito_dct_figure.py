#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S-02-b - mitochondrial protein fractions in the DCT data, control and
zero-K side by side, as one assembled reader-facing figure.

(reviewer 1, comment 1). The authors asked on 12 Nov 2026 for the
un-assembled dct_reviewer panels (dct_reviewer_mito_by_primary, _mito_global_cutoff, _mito_removed_vs_retained,
_mito_within_type_depth_correlation) to be redrawn as one figure ("Yes, redraw them"), following the re-render
checklist of results/tables/report/report_02_mito_qc.md, section 5. No analysis is rerun and no test is
recomputed: every plotted number is read from the stored dct_reviewer tables; the per-cell panels read the stored per-cell and
per-well tables. Each condition is shown on its own; nothing compares the conditions.

Labels: the cluster annotation (cs_ctrl_deg_label, cs_zerok_pipeline_label; the dct_reviewer "primary" column). Its value
'Unknown' is shown as "not resolved" (D4 of the response to the reviewers Nov 2026: DCT cells whose subtype was not resolved, clusters
without marker support). The marker-score label (clusters) (cs_<cond>_cluster_label, the dct_reviewer "second" column) is not
drawn: its rows are in Supplementary Table S-02-d, as the letter states. No drawn text names the revision process
(authors, 12 Nov 2026: "we only show the final products"); a check stops the run on "submitted", "submission" or
"re-annotation".

Figure (one file, 180 mm; no in-figure title, note or footer; the legend carries that text). Rows of the label panels:
Control (DCT1, DCT2, not resolved) and Zero-K (DCT1, DCT2 with 0 cells kept as an empty slot, not resolved).
  A  MitoCarta fraction of summed intensity per cell, median and IQR (stored summary), and the global top-10% cutoff
     of each condition (dashed).
  B  Percentage of each label above three global cutoffs (top 10%, top 5%, median + 3 MAD), drawn as nested bars
     from zero; the number is the top-10% value. Simulation only: no cell was excluded.
  C  MitoCarta fraction of identified proteins; D  OXPHOS fraction of summed intensity; E  fraction of summed
     intensity from the eight mitochondrially encoded proteins (cells, median and IQR).
  F  All wells: MitoCarta fraction of summed intensity against protein IDs per well (log scale); wells removed by the
     100-protein filter (dotted line) against retained cells. y axis 0-1, key outside the axes (checklist).
  G  The same fraction for removed wells, the retained cells of the lowest decile of protein IDs, and all retained
     cells (wells, median and IQR). The medians and group sizes are the stored ones; the box quartiles are
     percentiles of the stored per-well values (no stored table holds them). F and G use the per-well table over all
     protein rows of the search output (as dct_reviewer test D), A-E and H-J the retained proteins.
  H-J  Spearman rho within each label with 95% CI (stored): intensity fraction vs protein IDs (H) and vs total
     intensity (I), ID fraction vs protein IDs (J). Filled = BH q < 0.05 within the label column.
Colour: cells take the label colours of the DCT figures (dct_reviewer_lib.LABEL_PALETTE); removed wells red; all other
marks neutral, so that colour carries the label only.

Inputs (read-only): results/tables/dct_reviewer/dct_reviewer_mito_{qc_per_cell,all_raw_cells,
                    qc_summary_by_label,global_cutoff_simulation,global_cutoff_tests,removed_vs_retained,
                    within_type_depth_correlation}.csv;
                    results/tables/dct_labels/dct_labels_dct_canonical_labels.csv;
                    results/tables/report/report_02_mito_panels_values.csv
Outputs: results/figures/report/report_02_S02b_dct_mito.{pdf,png};
         results/tables/report/report_02_s02b_figure_{values,checks}.csv;
         results/tables/report/report_step_02_s02b_figure_manifest.json

Command (project root):  python scripts/report_mito_dct_figure.py --overwrite
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
import re

import numpy as np
import pandas as pd

import scp_common as cc
import atlas_dct_lib as T8
import dct_reviewer_lib as lib28

plt = T8.setup_matplotlib()
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

TASK = "report"
STEP = "02_s02b_figure"
P = "02_s02b_figure"
FIGURE = "02_S02b_dct_mito"
DCT_REVIEWER = cc.OUT_ROOTS["tables"] / lib28.TASK
REPORT = cc.OUT_ROOTS["tables"] / TASK
CONDITIONS = lib28.CONDITIONS                              # Ctrl, ZeroK
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
LABEL_DISPLAY = {"Unknown": "not resolved"}                # The response to the reviewers, D4 (12 Nov 2026)
ORDER = ("DCT1", "DCT2", "Unknown")                        # labels of the cluster annotation in either condition
ROLE = "primary"                                           # dct_reviewer name of the cluster-annotation column
INT, IDF, OXP, MTE = ("mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
                      "mt_encoded_intensity_fraction")
CUTOFFS = (("global_top10pct", "top 10%"), ("global_top5pct", "top 5%"), ("global_median_plus_3MAD", "median + 3 MAD"))
M, FONT, INK = T8.M, T8.FONT, T8.INK
SHADE = ("#c9c7c0", "#898781", "#3b3a37")                  # cutoff bars: top 10%, top 5%, median + 3 MAD
BAR_H = (0.62, 0.40, 0.20)
RED = T8.SERIES[7]                                         # removed wells
ROW_H = 4.8
PAD = 0.02                                                 # strip panels: axis padding as a fraction of the range
DOTTED = (0, (1, 1.6))                                     # the 100-protein filter (dashed = the top-10% cutoff)
LETTER_PT = 8.0
MINUS = "−"
# Numbers the the response to the reviewers letter quotes for this figure (section 2 of report_02_mito_qc.md).
LETTER = {
    "top10_percent": {"Ctrl": {"DCT1": 12, "DCT2": 1, "Unknown": 10}, "ZeroK": {"DCT1": 10, "Unknown": 11}},
    "top10_p": {"Ctrl": "0.008", "ZeroK": "0.59"},
    "removed_vs_retained_percent": {"Ctrl": (2, 49), "ZeroK": (0, 50)},
    "removed_median_ids": {"Ctrl": 10, "ZeroK": 6},
    "all_cells_median_percent": {"Ctrl": 49, "ZeroK": 50},
    "rho_ids": {"Ctrl": {"DCT1": "0.02", "DCT2": "0.12", "Unknown": "-0.04"}, "ZeroK": {"DCT1": "-0.13", "Unknown": "0.03"}},
    "rho_ids_significant": {("ZeroK", "DCT1"): "0.010"},
    "rho_total": {"Ctrl": {"DCT1": "-0.02", "Unknown": "-0.31"}, "ZeroK": {"DCT1": "0.04", "Unknown": "-0.34"}},
    "rho_idfraction": {"Ctrl": {"DCT1": "-0.96", "DCT2": "-0.70", "Unknown": "-0.72"},
                       "ZeroK": {"DCT1": "-0.93", "Unknown": "-0.76"}},
}
NOT_READER_FACING = re.compile(r"Unknown|Ctrl|ZeroK|\bT\d{2}\b|primary|second|per-cell|_|Generated|\bP[12]\b")
REVISION_PROCESS = re.compile(r"submitted|submission|re-?annotation", re.IGNORECASE)   # authors, 12 Nov 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-02-b (DCT, control and zero-K).")
    cc.add_common_args(parser)
    return parser.parse_args()


def r2(x: float) -> str:
    """Two decimals as the letter prints them (no negative zero)."""
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s


def clean(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK["muted"])
    ax.tick_params(axis="x", length=1.5, width=0.4, labelsize=FONT, colors=INK["secondary"], pad=1.5)
    ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"], pad=2)


def rows_axis(ax, slots: list[dict], labels: bool) -> None:
    """Horizontal row slots: a header slot per condition (no line), then one slot per label or group."""
    n = len(slots)
    data = [i for i, s in enumerate(slots) if s["kind"] == "row"]
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(data)
    ax.set_yticklabels([slots[i]["text"] for i in data] if labels else [""] * len(data))
    for i in data:
        ax.axhline(i, color=INK["grid"], lw=0.3, zorder=0)
    clean(ax)


def headers(sh, slots: list[dict], x: float, y_ax: float, row_h: float) -> None:
    for i, s in enumerate(slots):
        if s["kind"] == "head":
            sh.text(x, y_ax + (i + 0.5) * row_h, s["text"], fontweight="bold", va="center")


def strip(ax, i: int, v: np.ndarray, colour: str, rng, size: float = 1.3, alpha: float = 0.7) -> None:
    ax.scatter(v, i + rng.uniform(-0.3, 0.3, v.size), s=size, c=colour, alpha=alpha, linewidths=0, zorder=2)


def box(ax, i: int, q1: float, q2: float, q3: float) -> None:
    ax.add_patch(plt.Rectangle((q1, i - 0.33), q3 - q1, 0.66, fill=False, ec=INK["primary"], lw=0.6, zorder=3))
    ax.plot([q2, q2], [i - 0.33, i + 0.33], color=INK["primary"], lw=1.1, zorder=4, solid_capstyle="butt")


def no_cells(ax, i: int, x: float) -> None:
    ax.text(x, i, "no cells", ha="center", va="center", fontsize=FONT, color=INK["secondary"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})


def key_row(sh, y: float, entries: list[tuple]) -> None:
    """One row of keys, drawn in mm so that the spacing is exact. entries: (draw(ax, x, w, yc), text, marker width)."""
    ax = sh.ax(0, y, sh.W, 4.0)
    ax.set_xlim(0, sh.W)
    ax.set_ylim(4, 0)
    ax.axis("off")
    x, yc = M + 1.0, 2.0
    for draw, text, w in entries:
        if draw is not None:
            draw(ax, x, w, yc)
            x += w + 1.2
        sh.text(x, y + yc, text, va="center", color=INK["secondary"])
        x += T8.text_mm(text)[0] + (4.0 if draw is not None else 1.8)
    if x - 4.0 > sh.W - M:
        raise ValueError(f"key row is {x - 4.0:.1f} mm wide, more than the figure")


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)

    def rd(path, what: str, **kw) -> pd.DataFrame:
        return pd.read_csv(run.use_input(path, what), **kw)

    per = rd(DCT_REVIEWER / "dct_reviewer_mito_qc_per_cell.csv", "dct_reviewer per-cell mitochondrial metrics (retained cells)",
             dtype={"cell_id": str})
    raw = rd(DCT_REVIEWER / "dct_reviewer_mito_all_raw_cells.csv", "dct_reviewer per-well metrics, all wells, before protein filtering",
             dtype={"cell_id": str})
    summ = rd(DCT_REVIEWER / "dct_reviewer_mito_qc_summary_by_label.csv", "dct_reviewer summary by label")
    sim = rd(DCT_REVIEWER / "dct_reviewer_mito_global_cutoff_simulation.csv", "dct_reviewer global cutoff simulation")
    tst = rd(DCT_REVIEWER / "dct_reviewer_mito_global_cutoff_tests.csv", "dct_reviewer global cutoff tests")
    rvr = rd(DCT_REVIEWER / "dct_reviewer_mito_removed_vs_retained.csv", "dct_reviewer removed wells against retained cells")
    cor = rd(DCT_REVIEWER / "dct_reviewer_mito_within_type_depth_correlation.csv", "dct_reviewer within-label Spearman rho")
    canon = rd(lib28.DCT_LABELS_LABELS, "dct_labels canonical DCT labels (combined search)", dtype={"cell_id": str},
               keep_default_na=False, na_values=[""])
    t23 = rd(REPORT / "report_02_mito_panels_values.csv", "the response to the reviewers statement values (proteins per removed well)")
    summ, sim, tst, cor = (f[f["label_role"] == ROLE] for f in (summ, sim, tst, cor))

    checks: list[dict] = []
    values: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {observed}", flush=True)

    # ------------------------------------------------------------------ cells and labels
    counts = {}
    for cond in CONDITIONS:
        g = per[per["condition"] == cond]
        col = lib28.SETTINGS[cond].PRIMARY
        counts[cond] = g["primary_label"].astype(str).value_counts().to_dict()
        check(f"{cond}: retained cells in the per-cell table", len(g), lib28.N_CELLS[cond], len(g) == lib28.N_CELLS[cond])
        check(f"{cond}: cells per label equal the canonical counts (the project notes, {col})", counts[cond],
              lib28.EXPECTED_COUNTS[cond][col], counts[cond] == lib28.EXPECTED_COUNTS[cond][col])
        mine = g.set_index("cell_id")["primary_label"].astype(str)
        ref = canon[canon["condition"] == cond].set_index("cell_id")[col].astype(str).reindex(mine.index)
        check(f"{cond}: labels equal the dct_labels canonical table, matched on cell_id", f"{int((mine == ref).sum())} of {len(mine)}",
              f"{lib28.N_CELLS[cond]} of {lib28.N_CELLS[cond]}", bool((mine == ref).all()) and len(mine) == lib28.N_CELLS[cond])

    slots = []
    for cond in CONDITIONS:
        slots.append({"kind": "head", "cond": cond, "text": COND_DISPLAY[cond]})
        for lab in ORDER:
            n = counts[cond].get(lab, 0)
            slots.append({"kind": "row", "cond": cond, "label": lab, "n": n, "text": f"{LABEL_DISPLAY.get(lab, lab)} ({n})"})
    data_slots = [(i, s) for i, s in enumerate(slots) if s["kind"] == "row"]
    shown = {cond: [s["text"] for _, s in data_slots if s["cond"] == cond] for cond in CONDITIONS}
    check("row labels (display names and cells)", shown,
          "DCT1 (382), DCT2 (85), not resolved (262); DCT1 (449), DCT2 (0), not resolved (253)",
          shown == {"Ctrl": ["DCT1 (382)", "DCT2 (85)", "not resolved (262)"],
                    "ZeroK": ["DCT1 (449)", "DCT2 (0)", "not resolved (253)"]})

    # ------------------------------------------------------------------ geometry (mm; y downwards)
    rng = np.random.default_rng(cc.RANDOM_SEED)
    colours = lib28.label_colors(ORDER)
    lab_w = max(T8.text_mm(s["text"])[0] for _, s in data_slots)
    x0 = M + lab_w + 2.0
    let_h, head_h = 3.6, 6.9
    h_rows = len(slots) * ROW_H
    tier_gap = 4.6
    # tier 1: A-E
    gap = 4.0
    w_b, w_s = 20.0, 25.0
    w_a = T8.W - M - x0 - w_b - 3 * w_s - 4 * gap
    y1 = M
    y1_ax = y1 + let_h + head_h + 0.6
    xs1 = [x0, x0 + w_a + gap]
    xs1 += [xs1[1] + w_b + gap + k * (w_s + gap) for k in range(3)]
    # tier 2: F (two axes), G
    y2 = y1_ax + h_rows + 4.2 + tier_gap
    y2_ax = y2 + let_h + head_h + 0.6
    h2 = 33.0
    g_rows = 8
    g_row_h = h2 / g_rows
    xf, w_f, f_gap = M + 14.5, 37.0, 5.5
    # tier 3: H-J
    y3 = y2_ax + h2 + 8.6 + tier_gap
    y3_ax = y3 + let_h + head_h + 0.6
    gap3 = 7.0
    w_h = (T8.W - M - x0 - 2 * gap3) / 3
    xs3 = [x0 + k * (w_h + gap3) for k in range(3)]
    y_key = y3_ax + h_rows + 6.2
    H = y_key + 2 * 4.4 + M
    sh = T8.Sheet(H)

    def letter(x: float, y: float, char: str, head: str) -> None:
        sh.text(x, y, char, fontsize=LETTER_PT, fontweight="bold", va="top")
        sh.text(x, y + let_h, head, va="top", linespacing=1.1, color=INK["primary"])

    # ------------------------------------------------------------------ tier 1: fractions by label, global cutoffs
    heads1 = ["MitoCarta fraction of\nsummed intensity", "Cells above global\ncutoffs (%)",
              "MitoCarta fraction of\nidentified proteins", "OXPHOS fraction of\nsummed intensity",
              "mt-encoded fraction of\nsummed intensity"]
    for char, x, head in zip("ABCDE", xs1, heads1):
        letter(x, y1, char, head)
    headers(sh, slots, M, y1_ax, ROW_H)

    cut = {}
    for cond in CONDITIONS:
        t = tst[(tst["condition"] == cond) & (tst["metric"] == INT)].set_index("cutoff")
        cut[cond] = {c: float(t.loc[c, "threshold"]) for c, _ in CUTOFFS}
        order_ok = cut[cond]["global_median_plus_3MAD"] > cut[cond]["global_top5pct"] > cut[cond]["global_top10pct"]
        check(f"{cond}: thresholds ordered median + 3 MAD > top 5% > top 10% (nested bars are valid)",
              ", ".join(f"{cut[cond][c]:.3f}" for c, _ in reversed(CUTOFFS)), "decreasing", order_ok)

    def summary(cond: str, lab: str, metric: str) -> pd.Series:
        s = summ[(summ["condition"] == cond) & (summ["label"] == lab) & (summ["metric"] == metric)]
        if len(s) != 1:
            raise ValueError(f"{cond} {lab} {metric}: {len(s)} summary rows")
        return s.iloc[0]

    panels1 = (("A", xs1[0], w_a, INT, (0, 1), [0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"]),
               ("C", xs1[2], w_s, IDF, (0, 0.7), [0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"]),
               ("D", xs1[3], w_s, OXP, (0, 0.7), [0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"]),
               ("E", xs1[4], w_s, MTE, (0, 0.025), [0, 0.01, 0.02], ["0", "0.01", "0.02"]))
    worst = 0.0
    for char, x, w, metric, xlim, ticks, ticklabels in panels1:
        ax = sh.ax(x, y1_ax, w, h_rows)
        rows_axis(ax, slots, labels=char == "A")
        ax.set_xlim(xlim[0] - PAD * (xlim[1] - xlim[0]), xlim[1] + PAD * (xlim[1] - xlim[0]))   # cells at 0 stay whole
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels)
        for i, s in data_slots:
            v = per.loc[(per["condition"] == s["cond"]) & (per["primary_label"] == s["label"]), metric].to_numpy(float)
            if v.size == 0:
                if char == "A":
                    no_cells(ax, i, 0.5)
                continue
            if v.min() < xlim[0] or v.max() > xlim[1]:
                raise ValueError(f"panel {char}: {metric} outside the axis {xlim}")
            st = summary(s["cond"], s["label"], metric)
            worst = max(worst, float(np.abs(np.percentile(v, [25, 50, 75]) - st[["q25", "median", "q75"]].to_numpy(float)).max()),
                        abs(v.size - int(st["n_cells"])))
            strip(ax, i, v, colours[s["label"]], rng)
            box(ax, i, float(st["q25"]), float(st["median"]), float(st["q75"]))
            values.append({"panel": char, "condition": COND_DISPLAY[s["cond"]], "row": s["text"].split(" (")[0],
                           "source_label": s["label"], "quantity": metric, "n": v.size, "q25": st["q25"],
                           "median": st["median"], "q75": st["q75"], "source": "dct_reviewer_mito_qc_summary_by_label.csv; "
                           "cells from dct_reviewer_mito_qc_per_cell.csv"})
        if char == "A":
            for cond in CONDITIONS:
                idx = [i for i, s in data_slots if s["cond"] == cond]
                c10 = cut[cond]["global_top10pct"]
                ax.plot([c10, c10], [idx[0] - 0.5, idx[-1] + 0.5], color=INK["secondary"], lw=0.7, ls=(0, (3, 2)), zorder=5)
                values.append({"panel": "A", "condition": COND_DISPLAY[cond], "row": "dashed line",
                               "quantity": "global top-10% cutoff of the MitoCarta fraction of summed intensity",
                               "value": c10, "source": "dct_reviewer_mito_global_cutoff_tests.csv"})
    check("A, C-E: boxes (stored summary by label) equal the quartiles and cell numbers of the per-cell table",
          f"largest difference {worst:.2g}", "0", worst < 1e-9)

    ax = sh.ax(xs1[1], y1_ax, w_b, h_rows)
    rows_axis(ax, slots, labels=False)
    ax.set_xlim(0, 15)
    ax.set_xticks([0, 5, 10, 15])
    got10, n_above_ok = {c: {} for c in CONDITIONS}, True
    for i, s in data_slots:
        if s["n"] == 0:
            continue
        cond, lab = s["cond"], s["label"]
        d = sim[(sim["condition"] == cond) & (sim["metric"] == INT) & (sim["label"] == lab)].set_index("cutoff")
        pct = [float(d.loc[c, "percent_of_label_removed"]) for c, _ in CUTOFFS]
        if not pct[0] >= pct[1] >= pct[2]:
            raise ValueError(f"{cond} {lab}: percentages above the cutoffs are not nested: {pct}")
        v = per.loc[(per["condition"] == cond) & (per["primary_label"] == lab), INT].to_numpy(float)
        for (c, name), p, hgt, shade in zip(CUTOFFS, pct, BAR_H, SHADE):
            ax.barh(i, p, height=hgt, color=shade, linewidth=0, zorder=2)
            n_above_ok &= int((v > cut[cond][c]).sum()) == int(d.loc[c, "n_removed"]) and int(d.loc[c, "n_cells"]) == s["n"]
            values.append({"panel": "B", "condition": COND_DISPLAY[cond], "row": s["text"].split(" (")[0],
                           "source_label": lab, "quantity": f"percent of label above the global cutoff: {name}",
                           "n": int(d.loc[c, "n_cells"]), "n_above": int(d.loc[c, "n_removed"]), "value": p,
                           "threshold": cut[cond][c], "source": "dct_reviewer_mito_global_cutoff_simulation.csv"})
        ax.text(pct[0] + 0.5, i, f"{pct[0]:.0f}", va="center", ha="left", fontsize=FONT, color=INK["primary"])
        got10[cond][lab] = int(f"{pct[0]:.0f}")
    check("B: cells above each cutoff in the per-cell table equal the stored simulation (3 cutoffs, every label)",
          n_above_ok, True, n_above_ok)
    check("B: top-10% percentages as printed equal the letter (12, 1, 10; 10, 11)", got10, LETTER["top10_percent"],
          got10 == LETTER["top10_percent"])
    p10 = {c: float(tst[(tst["condition"] == c) & (tst["metric"] == INT) & (tst["cutoff"] == "global_top10pct")]
                    ["p_permutation"].iloc[0]) for c in CONDITIONS}
    got_p = {"Ctrl": f"{p10['Ctrl']:.3f}", "ZeroK": f"{p10['ZeroK']:.2f}"}
    check("B: permutation p of the top-10% cutoff (stored; not drawn) equals the letter (0.008; 0.59)", got_p,
          LETTER["top10_p"], got_p == LETTER["top10_p"])
    for cond in CONDITIONS:
        values.append({"panel": "B", "condition": COND_DISPLAY[cond], "row": "not drawn",
                       "quantity": "permutation p, label x above the top-10% cutoff (10,000 permutations)",
                       "value": p10[cond], "source": "dct_reviewer_mito_global_cutoff_tests.csv"})

    # ------------------------------------------------------------------ tier 2: removed wells against retained cells
    letter(xf, y2, "F", "Removed wells and retained cells")
    wells, med_ids = {}, {}
    for k, cond in enumerate(CONDITIONS):
        x = xf + k * (w_f + f_gap)
        sh.text(x, y2 + let_h + 3.3, COND_DISPLAY[cond], va="top", fontweight="bold")
        r = raw[raw["condition"] == cond]
        keep = r["retained"].astype(bool)
        ret, rem = r[keep], r[~keep]
        drawn = rem[rem["n_detected_proteins"] > 0]
        wells[cond] = (ret, rem, drawn)
        med_ids[cond] = float(rem["n_detected_proteins"].median())
        ax = sh.ax(x, y2_ax, w_f, h2)
        ax.scatter(ret["n_detected_proteins"], ret[INT], s=1.5, c=INK["muted"], alpha=0.6, linewidths=0, zorder=2)
        ax.scatter(drawn["n_detected_proteins"], drawn[INT], s=5.5, c=RED, linewidths=0.25, edgecolors="white", zorder=3)
        ax.axvline(100, color=INK["secondary"], lw=0.8, ls=DOTTED, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 7000)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1,000"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"] if k == 0 else [""] * 6)
        for yy in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
            ax.axhline(yy, color=INK["grid"], lw=0.3, zorder=0)
        T8.style_axes(ax)
        ax.tick_params(which="minor", length=1.0, width=0.3)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.4)
            ax.spines[sp].set_color(INK["muted"])
        sh.text(x + w_f / 2, y2_ax + h2 + 4.6, "Protein IDs per well (log scale)", ha="center", va="top",
                color=INK["secondary"])
        values.append({"panel": "F", "condition": COND_DISPLAY[cond], "row": "retained cells", "n": len(ret),
                       "quantity": "wells drawn (x: protein IDs per well; y: MitoCarta fraction of summed intensity)",
                       "source": "dct_reviewer_mito_all_raw_cells.csv"})
        values.append({"panel": "F", "condition": COND_DISPLAY[cond], "row": "removed wells (fewer than 100 protein IDs)",
                       "n": len(drawn), "n_not_drawn_no_protein": len(rem) - len(drawn),
                       "quantity": "wells drawn; median protein IDs per removed well in 'value'", "value": med_ids[cond],
                       "source": "dct_reviewer_mito_all_raw_cells.csv"})
    sh.text(M + 1.2, y2_ax + h2 / 2, "MitoCarta fraction of\nsummed intensity", rotation=90, ha="left", va="center",
            linespacing=1.1, color=INK["secondary"], multialignment="center")
    n_wells = {c: (len(wells[c][0]) + len(wells[c][1]), len(wells[c][0]), len(wells[c][1]), len(wells[c][2])) for c in CONDITIONS}
    check("F: wells per condition (all, retained, removed, removed with at least one protein = drawn)", n_wells,
          {"Ctrl": (768, 729, 39, 39), "ZeroK": (767, 702, 65, 63)},
          n_wells == {"Ctrl": (768, 729, 39, 39), "ZeroK": (767, 702, 65, 63)})
    ids_ok = all(int(wells[c][1]["n_detected_proteins"].max()) < 100 <= int(wells[c][0]["n_detected_proteins"].min())
                 for c in CONDITIONS)
    check("F: every removed well has fewer than 100 protein IDs and every retained cell at least 100", ids_ok, True, ids_ok)
    ref_ids = {c: float(t23[(t23["statement"] == "median proteins per removed well") & (t23["dataset"] == c)]["value"].iloc[0])
               for c in CONDITIONS}
    check("F: median protein IDs per removed well equal the the response to the reviewers values file and the letter (10; 6)", med_ids,
          LETTER["removed_median_ids"], med_ids == ref_ids and {c: int(v) for c, v in med_ids.items()} == LETTER["removed_median_ids"])

    g_slots, g_vals = [], {}
    for cond in CONDITIONS:
        ret, rem, _ = wells[cond]
        t = rvr[(rvr["condition"] == cond) & (rvr["metric"] == INT)]
        all_row = t[t["comparison"] == "removed_vs_retained_all"].iloc[0]
        low_row = t[t["comparison"].str.startswith("removed_vs_retained_lowest_decile_ids")].iloc[0]
        thr = int(re.search(r"<= (\d+) IDs", low_row["comparison"]).group(1))
        low = ret[ret["n_detected_proteins"] <= thr]
        groups = (("Removed", rem[INT].dropna().to_numpy(float), float(all_row["median_group"]), int(all_row["n_group"]), RED),
                  ("Retained, lowest 10%", low[INT].to_numpy(float), float(low_row["median_reference"]),
                   int(low_row["n_reference"]), INK["muted"]),
                  ("Retained, all", ret[INT].to_numpy(float), float(all_row["median_reference"]),
                   int(all_row["n_reference"]), INK["muted"]))
        g_slots.append({"kind": "head", "cond": cond, "text": COND_DISPLAY[cond]})
        for name, v, med, n, colour in groups:
            g_slots.append({"kind": "row", "cond": cond, "label": name, "n": v.size, "text": f"{name} ({v.size})"})
            g_vals[(cond, name)] = (v, med, n, colour, thr)
    g_lab_w = max(T8.text_mm(s["text"])[0] for s in g_slots if s["kind"] == "row")
    xg_lab = xf + 2 * w_f + f_gap + 7.0
    xg = xg_lab + g_lab_w + 2.0
    w_g = T8.W - M - xg - 1.0
    letter(xg, y2, "G", "MitoCarta fraction of\nsummed intensity")
    headers(sh, g_slots, xg_lab, y2_ax, g_row_h)
    ax = sh.ax(xg, y2_ax, w_g, h2)
    rows_axis(ax, g_slots, labels=True)
    ax.set_xlim(-PAD, 1 + PAD)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    g_ok, got_rr, got_all = True, {}, {}
    for i, s in enumerate(g_slots):
        if s["kind"] != "row":
            continue
        v, med, n, colour, thr = g_vals[(s["cond"], s["label"])]
        q1, q2, q3 = np.percentile(v, [25, 50, 75])
        g_ok &= bool(np.isclose(q2, med, atol=1e-12)) and v.size == n
        strip(ax, i, v, colour, rng, size=1.5 if colour == RED else 1.2, alpha=0.75 if colour == RED else 0.55)
        box(ax, i, q1, med, q3)
        values.append({"panel": "G", "condition": COND_DISPLAY[s["cond"]], "row": s["label"], "n": v.size,
                       "quantity": INT + " (all protein rows of the search output)", "q25": q1, "median": med, "q75": q3,
                       "threshold": thr if s["label"].startswith("Retained, lowest") else np.nan,
                       "source": "median and n: dct_reviewer_mito_removed_vs_retained.csv; wells and box quartiles: "
                                 "dct_reviewer_mito_all_raw_cells.csv"})
        if s["label"] == "Removed":
            got_rr[s["cond"]] = [int(f"{100 * med:.0f}")]
        elif s["label"] == "Retained, all":
            got_rr[s["cond"]].append(int(f"{100 * med:.0f}"))
            got_all[s["cond"]] = int(f"{100 * float(summary(s['cond'], 'all_cells', INT)['median']):.0f}")
    got_rr = {c: tuple(v) for c, v in got_rr.items()}
    check("G: medians and group sizes drawn from the per-well table equal the stored removed-vs-retained table", g_ok, True, g_ok)
    check("G: removed vs retained medians in % equal the letter (2 vs 49; 0 vs 50)", got_rr,
          LETTER["removed_vs_retained_percent"], got_rr == LETTER["removed_vs_retained_percent"])
    check("median MitoCarta fraction of all retained cells in % (stored summary; = 'Retained, all' in G) equals the "
          "letter (49; 50)", got_all, LETTER["all_cells_median_percent"],
          got_all == LETTER["all_cells_median_percent"] and all(got_rr[c][1] == got_all[c] for c in CONDITIONS))

    # ------------------------------------------------------------------ tier 3: Spearman rho within labels
    heads3 = [f"Intensity fraction vs\nprotein IDs (ρ)", "Intensity fraction vs\ntotal intensity (ρ)",
              "ID fraction vs\nprotein IDs (ρ)"]
    specs3 = (("H", INT, "log10_n_detected"), ("I", INT, "log10_total_intensity"), ("J", IDF, "log10_n_detected"))
    headers(sh, slots, M, y3_ax, ROW_H)
    got_rho = {char: {c: {} for c in CONDITIONS} for char, _, _ in specs3}
    got_q = {}
    for (char, metric, depth), x, head in zip(specs3, xs3, heads3):
        letter(x, y3, char, head)
        ax = sh.ax(x, y3_ax, w_h, h_rows)
        rows_axis(ax, slots, labels=char == "H")
        ax.axvline(0, color=INK["muted"], lw=0.4, zorder=1)
        ax.set_xlim(-1.08, 1.08)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels([f"{MINUS}1", f"{MINUS}0.5", "0", "0.5", "1"])
        for i, s in data_slots:
            if s["n"] == 0:
                if char == "H":
                    no_cells(ax, i, 0.0)
                continue
            r = cor[(cor["condition"] == s["cond"]) & (cor["metric"] == metric) & (cor["depth_variable"] == depth)
                    & (cor["label"] == s["label"])]
            if len(r) != 1 or int(r.iloc[0]["n"]) != s["n"]:
                raise ValueError(f"{char} {s['cond']} {s['label']}: correlation row missing or n differs")
            r = r.iloc[0]
            q = float(r["p_fdr_bh_within_labels"])
            sig = q < 0.05
            ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=INK["primary"], lw=0.7, zorder=2, solid_capstyle="butt")
            ax.plot([r["rho"]], [i], "o", ms=3.4, mec=INK["primary"], mew=0.7, mfc=INK["primary"] if sig else "white", zorder=3)
            got_rho[char][s["cond"]][s["label"]] = r2(float(r["rho"]))
            if char == "H":
                got_q[(s["cond"], s["label"])] = q
            values.append({"panel": char, "condition": COND_DISPLAY[s["cond"]], "row": s["text"].split(" (")[0],
                           "source_label": s["label"], "quantity": f"Spearman rho, {metric} vs {depth}", "n": int(r["n"]),
                           "rho": r["rho"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "q_bh_within_labels": q,
                           "filled": sig, "source": "dct_reviewer_mito_within_type_depth_correlation.csv"})
    check("H: rho of the intensity fraction with protein IDs equals the letter (0.02, 0.12, -0.04; -0.13, 0.03)",
          got_rho["H"], LETTER["rho_ids"], got_rho["H"] == LETTER["rho_ids"])
    sig_h = {k: f"{v:.3f}" for k, v in got_q.items() if v < 0.05}
    check("H: filled dots (q < 0.05) equal the letter (zero-K DCT1 only, q 0.010; control none)", sig_h,
          LETTER["rho_ids_significant"], sig_h == LETTER["rho_ids_significant"])
    sub_i = {c: {k: v for k, v in got_rho["I"][c].items() if k in LETTER["rho_total"][c]} for c in CONDITIONS}
    check("I: rho with total intensity equals the letter (DCT1 -0.02, 0.04; not resolved -0.31, -0.34)", sub_i,
          LETTER["rho_total"], sub_i == LETTER["rho_total"])
    check("J: rho of the ID fraction with protein IDs equals the letter (-0.70 to -0.96)", got_rho["J"],
          LETTER["rho_idfraction"], got_rho["J"] == LETTER["rho_idfraction"])

    # ------------------------------------------------------------------ keys
    def k_cells(ax, x, w, yc):
        for j, lab in enumerate(ORDER):
            ax.plot([x + 0.8 + j * 1.7], [yc], "o", ms=2.3, mfc=colours[lab], mec="none")

    def k_box(ax, x, w, yc):
        ax.add_patch(plt.Rectangle((x, yc - 1.1), w, 2.2, fill=False, ec=INK["primary"], lw=0.6))
        ax.plot([x + w / 2] * 2, [yc - 1.1, yc + 1.1], color=INK["primary"], lw=1.1)

    def k_dash(ax, x, w, yc):
        ax.plot([x, x + w], [yc, yc], color=INK["secondary"], lw=0.7, ls=(0, (3, 2)))

    def k_dots(ax, x, w, yc):
        ax.plot([x, x + w], [yc, yc], color=INK["secondary"], lw=0.8, ls=DOTTED)

    def k_shade(j):
        return lambda ax, x, w, yc: ax.add_patch(plt.Rectangle((x, yc - 1.1 * BAR_H[j] / BAR_H[0]), w,
                                                               2.2 * BAR_H[j] / BAR_H[0], fc=SHADE[j], ec="none"))

    def k_dot(colour, ms, fill=True):
        return lambda ax, x, w, yc: ax.plot([x + w / 2], [yc], "o", ms=ms, mec=colour, mew=0.7 if not fill or ms > 3 else 0,
                                            mfc=colour if fill else "white")

    def k_line(ax, x, w, yc):
        ax.plot([x, x + w], [yc, yc], color=INK["primary"], lw=0.7)

    key_row(sh, y_key, [(k_cells, "one cell", 5.0), (k_box, "median and IQR", 4.0),
                        (k_dash, "global top-10% cutoff", 5.0), (None, "cells above:", 0), (k_shade(0), "top 10%", 3.0),
                        (k_shade(1), "top 5%", 3.0), (k_shade(2), "median + 3 MAD", 3.0)])
    key_row(sh, y_key + 4.4, [(k_dot(RED, 2.6), "removed well", 1.8), (k_dot(INK["muted"], 2.0), "retained cell", 1.8),
                              (k_dots, "100 protein IDs", 5.0), (k_dot(INK["primary"], 3.4), "BH q < 0.05", 1.8),
                              (k_dot(INK["primary"], 3.4, fill=False), "q ≥ 0.05", 1.8), (k_line, "95% CI", 4.0)])

    # ------------------------------------------------------------------ reader-facing text, format, save
    texts = [t.get_text() for t in sh.fig.findobj(Text) if t.get_visible() and t.get_text().strip()]   # drawn texts
    bad = sorted({t for t in texts if NOT_READER_FACING.search(t)})
    check("figure text is reader-facing (no 'Unknown', Ctrl, ZeroK, task codes, column or file names)", bad or "none",
          "none", not bad)
    process = sorted({t for t in texts if REVISION_PROCESS.search(t)})
    check("no drawn text names the revision process ('submitted', 'submission', 're-annotation')", process or "none",
          "none", not process)
    letters = [t for t in texts if re.fullmatch(r"[A-J]", t)]
    check("panel letters", "".join(letters), "ABCDEFGHIJ", "".join(sorted(letters)) == "ABCDEFGHIJ")
    fig_rows: list[dict] = []
    ok = T8.save_checked(run, sh, FIGURE, "Supplementary Figure S-02-b: mitochondrial protein fractions in the DCT data, "
                         "control and zero-K: by label with global cutoffs (A-E), removed wells against retained cells "
                         "(F, G), Spearman rho within labels (H-J); from the stored dct_reviewer tables", fig_rows)
    row = fig_rows[0]
    check("figure format (Arial, every text >= 7 pt, no text overlap, no text outside, 180 mm, PDF TrueType + 600-dpi PNG)",
          f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps {row['n_text_overlaps']}; "
          f"outside {row['n_text_outside_figure']}; fonts {row['fonts']}; PNG {row.get('png_px')}", "pass", ok)
    check("figure height (one page with the legend)", f"{H:.1f} mm", "<= 230 mm", H <= 230)

    run.table(pd.DataFrame(values), f"{P}_values")
    run.table(pd.concat([pd.DataFrame(checks), pd.DataFrame(fig_rows)], ignore_index=True), f"{P}_checks")
    all_ok = all(c["pass"] for c in checks)
    run.finish({"figure": "Supplementary Figure S-02-b", "label_columns": {c: lib28.SETTINGS[c].PRIMARY for c in CONDITIONS},
                "label_display": LABEL_DISPLAY, "condition_display": COND_DISPLAY, "width_mm": T8.W, "height_mm": round(H, 1),
                "panels": "A-E by label; F, G removed wells; H-J Spearman rho", "jitter_seed": cc.RANDOM_SEED,
                "checks_pass": all_ok}, status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit(f"A check failed; see report_{P}_checks.csv.")


if __name__ == "__main__":
    main()
