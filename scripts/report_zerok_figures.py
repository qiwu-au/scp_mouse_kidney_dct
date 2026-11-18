#!/usr/bin/env python
"""The response to the reviewers: figures for the response to the Editors' comment 2 (zero-K against control, DCT cells).

(plan approved by the authors on 12 Nov 2026: "Approve as
planned"). Reader-facing redraws of the zerok_vs_control analysis figures (zerok_vs_control_figures.py). No new statistics: every
plotted value is read from a stored zerok_vs_control or dct_labels table; confidence limits of the intensity effect are the stored
coefficient +/- 1.96 x the stored standard error, as in the zerok_vs_control figures. Nothing in zerok_vs_control or dct_labels is changed.

Design shown in every legend (not in the figures): one pool of 3 mice per condition, so condition and pool cannot
be separated, cells are not animal-level replicates and every p value is descriptive; P1 and P2 are sort plates of
the pooled cells (technical).

Main figure F-19 (placeholder), single plots without panel letters (report addendum of 12 Nov 2026):
  report_19_fig_panel_effects   180 mm  the 8 panel proteins with an expected direction, DCT1 + DCT2 cells of
                                            the cluster annotation (467 control, 449 zero-K): detection, intensity
                                            among detected cells, mean-intensity ratio per sort plate, observed
                                            direction and match
  report_19_fig_matched_sets     85 mm  the 7 directional panel proteins against 10,000 matched random sets
  report_19_fig_F19_compiled    180 mm  the two plots as one file with the letters A and B (authors, 12 Nov 2026:
                                            "Compile F-19"); drawn by the same code at the same size, A above B
Supplementary figures, one assembled file each, with panel letters:
  report_19_S19a_zerok_annotation   A, B zero-K UMAP by cluster annotation and by the label transferred from
                                        control cells; C label fractions; D cells with cycling proteins
  report_19_S19b_panel_overview     A observed direction of all 28 panel items in every group and normalization
                                        variant; B share of 100 repeats keeping the result (depth-matched, 50%, 25%)
  report_19_S19c_specificity        A, B matched random sets; C directional percentiles; D sort-plate null
  report_19_S19d_bulk_benchmark     A-C comparison with a bulk low-K whole-kidney proteome
Display names: Ctrl -> Control, ZeroK -> Zero-K; label value Unknown -> "DCT, subtype not resolved" / "not
resolved"; dct_label -> cluster annotation; dct_label_alternative -> zero-K label transferred from control cells,
marked "(t)"; all_DCT -> "DCT1 + DCT2"; label_free -> all cells. No internal name is drawn (checked).

Inputs (read-only): results/tables/zerok_vs_control/zerok_vs_control_*.csv (panel_results, robustness_label_free_panel,
  robustness_panel_repeats, label_fractions, state_cycling, specificity_*, proteome_summary, bulk_*);
  results/tables/dct_labels/dct_labels_dct_canonical_labels.csv; the frozen zerok_core panel (hash checked).
Outputs: results/figures/report/report_19_*.{pdf,png,tif};
  results/tables/report/report_19_zerok_figures_{values,checks}.csv;
  results/tables/report/report_step_19_zerok_figures_manifest.json

Command (project root):  python scripts/report_zerok_figures.py --overwrite
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
import atlas_dct_lib as t8
import zerok_vs_control_lib as lib30

plt = t8.setup_matplotlib()
import matplotlib  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

TASK, STEP = "report", "19_zerok_figures"
t22 = lib30.t22
INK, FONT = t8.INK, t8.FONT
FORMATS = (".pdf", ".png", ".tif")
COND = dict(t22.CONDITION_PALETTE)
COND_NAME = {"Ctrl": "Control", "ZeroK": "Zero-K"}
PLATE = cc.PLATE_PALETTE
LABEL_COLOR = {"DCT1": "#1f77b4", "DCT2": "#ff7f0e", "Unknown": "#B3B3B3"}     # Figure 3 colours
LABEL_NAME = {"DCT1": "DCT1", "DCT2": "DCT2", "Unknown": "DCT, subtype not resolved"}
DIR_COLOR = {t22.UP: "#eb6834", t22.DOWN: "#2a78d6", t22.OBS_MIXED: "#4a3aa7", t22.OBS_NS: "#d6d5cf"}
DIR_NAME = {t22.UP: "higher in zero-K", t22.DOWN: "lower in zero-K", t22.OBS_MIXED: "mixed (detection and intensity "
            "in opposite directions)", t22.OBS_NS: "no difference (q ≥ 0.05)"}
EXP = {t22.UP: "higher", t22.DOWN: "lower", t22.NO_CHANGE: "unchanged", t22.NO_EXPECTATION: "–"}
EXP_SHORT = {t22.UP: "↑", t22.DOWN: "↓", t22.NO_CHANGE: "=", t22.NO_EXPECTATION: "–"}
OBS = {t22.UP: "higher", t22.DOWN: "lower", t22.OBS_MIXED: "mixed", t22.OBS_NS: "no difference"}
MATCH = {"match": "yes", "mismatch": "no"}
PROT = {"Slc12a3": "NCC (Slc12a3)", "Wnk1": "WNK1 (Wnk1)", "Wnk4": "WNK4 (Wnk4)", "Klhl3": "KLHL3 (Klhl3)",
        "Cul3": "CUL3 (Cul3)", "Pvalb": "Parvalbumin (Pvalb)", "Calb1": "Calbindin-D28k (Calb1)",
        "Slc8a1": "NCX1 (Slc8a1)"}
SCORE_NAME = {"ALDO_activity (base)": "Aldosterone activity score",
              "ALDO_activity (no_Slc12a3)": "Aldosterone activity score, no Slc12a3",
              "ALDO_signed (base)": "Aldosterone signed score",
              "ALDO_signed (no_Slc12a3)": "Aldosterone signed score, no Slc12a3",
              "ALDO_machinery": "Aldosterone machinery score",
              "cell_cycle_score (18-protein panel)": "Cell-cycle score"}
PRI, ALT, FREE = "dct_label", "dct_label_alternative", "label_free"
GROUP = {(PRI, "all_DCT"): "DCT1 + DCT2", (PRI, "DCT1"): "DCT1", (PRI, "Unknown"): "not resolved",
         (ALT, "all_DCT"): "DCT1 + DCT2 (t)", (ALT, "DCT1"): "DCT1 (t)", (ALT, "DCT2"): "DCT2 (t)",
         (ALT, "Unknown"): "not resolved (t)", (FREE, "all_cells"): "all cells",
         (FREE, "Slc12a3_detected"): "NCC detected"}
ROWS8 = [(PRI, "all_DCT"), (PRI, "DCT1"), (PRI, "Unknown"), (ALT, "all_DCT"), (ALT, "DCT1"), (ALT, "DCT2"),
         (ALT, "Unknown"), (FREE, "all_cells")]
ROWS5 = [(PRI, "all_DCT"), (PRI, "DCT1"), (ALT, "all_DCT"), (ALT, "DCT1"), (FREE, "all_cells")]
CONTRAST = {"Ctrl P2 vs P1": "Control, P2 vs P1", "ZeroK P2 vs P1": "Zero-K, P2 vs P1",
            "ZeroK vs Ctrl, P1 only": "Zero-K vs control, P1", "ZeroK vs Ctrl, P2 only": "Zero-K vs control, P2",
            "ZeroK vs Ctrl (pooled)": "Zero-K vs control, both plates",
            "Ctrl P2 vs P1 (technical)": "Control, P2 vs P1", "ZeroK P2 vs P1 (technical)": "Zero-K, P2 vs P1"}
# Internal names that must not be drawn (reader-facing figures).
BANNED = ("ctrl", "zerok", "unknown", "generated", "pipeline", "primary", "alternative", "submitted",
          "label-free", "label_free", "_", ".csv", "t27", "t30", "frozen")
VALUES: list[dict] = []
RECORD = True                       # False while the compiled Figure F-19 repeats the two plots (values recorded once)
TEXTS: dict[str, list[str]] = {}    # drawn texts of every saved figure (filled by save)
PANEL_PITCH, PANEL_TOP = 4.4, 12.0
PANEL_H = PANEL_TOP + len(PROT) * PANEL_PITCH + 21.0        # 68.2 mm: height of the panel-effects plot
MATCHED_W, MATCHED_H = 85.0, 50.0                           # size of the matched-sets plot


class Sub:
    """A region of a sheet: the same text and ax calls, shifted by (dx, dy) mm."""

    def __init__(self, sheet, dx: float = 0.0, dy: float = 0.0):
        self.sheet, self.dx, self.dy = sheet, dx, dy
        self.fig, self.W, self.H = sheet.fig, sheet.W, sheet.H

    def text(self, x, y, s, **kw):
        return self.sheet.text(x + self.dx, y + self.dy, s, **kw)

    def ax(self, x, y, w, h):
        return self.sheet.ax(x + self.dx, y + self.dy, w, h)


def val(figure: str, panel: str, row, series, value, low=np.nan, high=np.nan, source: str = "") -> None:
    if not RECORD:
        return
    VALUES.append({"figure": figure, "panel": panel, "row": row, "series": series, "value": value, "low": low,
                   "high": high, "source": source})


def style(ax) -> None:
    t8.style_axes(ax)
    ax.tick_params(labelsize=FONT, colors=INK["secondary"])


def letter(sheet, x, y, s) -> None:
    sheet.text(x, y, s, fontsize=8.5, fontweight="bold", va="top")


def row_axis(ax, n: int, guides: bool = True) -> None:
    """Axis whose rows are given by text outside it: no y ticks, rows from top to bottom."""
    if guides:
        for i in range(n):
            ax.axhline(i, color=INK["grid"], lw=0.4, zorder=0)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    style(ax)
    ax.spines["left"].set_visible(False)


def clean_ticks(fig) -> None:
    """Drop tick locations outside the view limits (their hidden labels would count in the text checks)."""
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(lim)
            tol = 1e-9 * (hi - lo)
            ticks = list(axis.get_ticklocs())
            keep = [t for t in ticks if lo - tol <= t <= hi + tol]
            if len(keep) != len(ticks):
                axis.set_ticks(keep)


def save(run, sheet, name: str, desc: str, rows: list, checks: t5.Checks) -> bool:
    clean_ticks(sheet.fig)
    sheet.fig.canvas.draw()
    texts = [t.get_text() for t in sheet.fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    TEXTS[name] = texts
    bad = sorted({t for t in texts if any(b in t.lower() for b in BANNED)})
    checks.add(f"{name}: reader-facing text only (no internal label, column, task or file name)",
               f"{len(texts)} texts; {len(bad)} with an internal name" + (f": {bad[:6]}" if bad else ""), not bad)
    ok = t8.save_checked(run, sheet, name, desc, rows, formats=FORMATS)
    checks.add(f"{name}: format checks (exact width, Arial >= 7 pt, no text overlap, no text outside, TrueType PDF)",
               f"{rows[-1]['width_mm']} x {rows[-1]['height_mm']} mm; min font {rows[-1]['min_font_pt']} pt; overlaps "
               f"{rows[-1]['n_text_overlaps']}; outside {rows[-1]['n_text_outside_figure']}", ok)
    return ok


def item_name(item: str, low: tuple[str, ...] = ()) -> str:
    name = SCORE_NAME.get(item, item)
    return f"{name}*" if item in low else name


def pick(frame: pd.DataFrame, **where) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for col, value in where.items():
        mask &= frame[col].eq(value)
    return frame[mask]


# ---------------------------------------------------------------------------
# Main figure F-19
# ---------------------------------------------------------------------------
def draw_panel(sheet, T, name: str, checks) -> None:
    """The panel-effects plot on a sheet (or a region of one); 180 x PANEL_H mm."""
    res = pick(T["panel"], label_column=PRI, group="all_DCT", variant="none").set_index("item").loc[list(PROT)]
    n, pitch, top = len(res), PANEL_PITCH, PANEL_TOP
    h = n * pitch
    cx ={"name": 3.0, "exp": 36.0, "det": 52.0, "int": 90.0, "plate": 124.0, "obs": 154.0, "match": 168.0}
    wd = {"det": 32.0, "int": 28.0, "plate": 26.0}
    head = dict(color=INK["secondary"], fontweight="bold", va="top")
    for key, s in (("name", "Protein"), ("exp", "Expected"), ("obs", "Observed"), ("match", "Match")):
        sheet.text(cx[key], 4.0, s, **head)
    for key, s in (("det", "Detection"), ("int", "Intensity"), ("plate", "Per sort plate")):
        sheet.text(cx[key] + wd[key] / 2, 4.0, s, ha="center", **head)
    for i, (item, r) in enumerate(res.iterrows()):
        yy = top + (i + 0.5) * pitch
        sheet.text(cx["name"], yy, PROT[item], va="center")
        sheet.text(cx["exp"], yy, EXP[r["expected_direction"]], va="center")
        sheet.text(cx["obs"], yy, OBS[r["observed_direction"]], va="center")
        sheet.text(cx["match"], yy, MATCH[r["match_status"]], va="center")
        val(name, "text", item, "expected / observed / match",
            f"{r['expected_direction']} / {r['observed_direction']} / {r['match_status']}",
            source="zerok_vs_control_panel_results.csv")
        val(name, "text", item, "q (BH within the panel)", float(r["q_primary"]), source="zerok_vs_control_panel_results.csv")
    y = np.arange(n)
    src = "zerok_vs_control_panel_results.csv"
    axd = sheet.ax(cx["det"], top, wd["det"], h)
    for cond, off in (("Ctrl", -0.17), ("ZeroK", 0.17)):
        k = cond.lower()
        f = 100 * res[f"detection_fraction_{k}"].to_numpy(float)
        lo = 100 * res[f"detection_ci_low_{k}"].to_numpy(float)
        hi = 100 * res[f"detection_ci_high_{k}"].to_numpy(float)
        axd.hlines(y + off, lo, hi, color=COND[cond], lw=0.7)
        axd.plot(f, y + off, "o", ms=3.2, color=COND[cond], ls="", label=COND_NAME[cond])
        for item, a, b, c in zip(res.index, f, lo, hi):
            val(name, "Detection", item, f"{COND_NAME[cond]}: cells with the protein detected (%), Wilson 95% interval",
                a, b, c, src)
    axd.set_xlim(-3, 103)
    axd.set_xticks([0, 50, 100])
    axd.set_xlabel("cells with the protein\ndetected (%)", labelpad=1.5)
    row_axis(axd, n)
    axd.legend(loc="upper center", bbox_to_anchor=(0.5, -12.5 / h), ncol=2, frameon=False, handletextpad=0.2,
               columnspacing=1.0, borderaxespad=0)
    axi = sheet.ax(cx["int"], top, wd["int"], h)
    est = res["int_log2fc_adjusted"].to_numpy(float)
    se = res["int_coef_se"].to_numpy(float) / np.log(2)
    axi.hlines(y, est - 1.96 * se, est + 1.96 * se, color=INK["secondary"], lw=0.7)
    axi.plot(est, y, "o", ms=3.2, color=INK["primary"], ls="")
    axi.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    lim = float(np.max(np.abs(np.r_[est - 1.96 * se, est + 1.96 * se]))) * 1.08
    axi.set_xlim(-lim, lim)
    axi.set_xticks([-1, 0, 1])
    axi.set_xlabel("intensity in detected cells\n(log2 fold change)", labelpad=1.5)
    row_axis(axi, n)
    for item, a, b in zip(res.index, est, se):
        val(name, "Intensity", item, "log2 fold change among detected cells, adjusted for proteins per cell; 95% "
            "interval = estimate +/- 1.96 x standard error", a, a - 1.96 * b, a + 1.96 * b, src)
    axp = sheet.ax(cx["plate"], top, wd["plate"], h)
    for plate, mk, off in (("P1", "^", -0.17), ("P2", "s", 0.17)):
        v = res[f"pseudobulk_log2fc_{plate}"].to_numpy(float)
        axp.plot(v, y + off, mk, ms=3.2, color=PLATE[plate], ls="", label=plate)
        for item, a in zip(res.index, v):
            val(name, "Per sort plate", item, f"log2 ratio of mean intensity per cell, zero-K / control, {plate} cells",
                a, source=src)
    for item, a in zip(res.index, res["pseudobulk_log2fc_pooled"].to_numpy(float)):
        val(name, "Per sort plate", item, "log2 ratio of mean intensity per cell, both plates (not drawn)", a,
            source=src)
    axp.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    axp.set_xlim(-1.6, 4.8)
    axp.set_xticks([0, 2, 4])
    axp.set_xlabel("mean intensity per cell\n(log2 ratio)", labelpad=1.5)
    row_axis(axp, n)
    axp.legend(loc="upper center", bbox_to_anchor=(0.5, -12.5 / h), ncol=2, frameon=False, handletextpad=0.2,
               columnspacing=1.0, borderaxespad=0)
    n_c, n_z = int(res["n_ctrl"].iloc[0]), int(res["n_zerok"].iloc[0])
    match = res["match_status"].value_counts().to_dict()
    checks.add("panel plot: DCT1 + DCT2 cells of the cluster annotation, control / zero-K", f"{n_c} / {n_z}",
               (n_c, n_z) == (467, 449), "467 / 449")
    checks.add("panel plot: scored proteins in the expected direction", f"{match.get('match', 0)} of {n}; not as "
               f"expected: {', '.join(res.index[res['match_status'].ne('match')])}",
               match.get("match") == 7 and list(res.index[res["match_status"].ne("match")]) == ["Cul3"], "7 of 8; Cul3")


def fig_panel(run, T, rows, checks) -> bool:
    name = "19_fig_panel_effects"
    sheet = t8.Sheet(PANEL_H)
    draw_panel(sheet, T, name, checks)
    return save(run, sheet, name, "F-19: pre-specified panel proteins with an expected direction, zero-K against "
                "control, DCT1 + DCT2 cells of the cluster annotation: detection, intensity, per sort plate", rows, checks)


def draw_matched(sheet, T, name: str, checks) -> None:
    """The matched-sets plot on a sheet (or a region of one); MATCHED_W x MATCHED_H mm."""
    d = pick(T["spec_draws"], label_column=PRI, group="all_DCT", variant="none", statistic="S1 matches").sort_values("value")
    m = pick(T["spec_null"], label_column=PRI, group="all_DCT", variant="none", statistic="S1 matches").iloc[0]
    ax = sheet.ax(14.0, 4.0, 67.0, 34.0)
    share = 100 * d["n_draws"].to_numpy(float) / float(d["n_draws"].sum())
    ax.bar(d["value"].to_numpy(int), share, width=0.8, color=INK["muted"], linewidth=0)
    obs, slots = int(m["observed"]), int(m["n_slots"])
    ymax = float(share.max()) * 1.12
    ax.annotate(f"observed: {obs} of {slots}\nempirical p = {float(m['empirical_p_greater_or_equal']):.4f}",
                xy=(obs, 0.4), xytext=(obs + 0.35, 0.62 * ymax), ha="right", va="bottom", fontsize=FONT,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK["primary"], shrinkA=1, shrinkB=0,
                                relpos=(0.93, 0.0)))
    ax.set_xlim(-0.6, slots + 0.6)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(slots + 1))
    ax.set_xlabel("proteins changed in the expected direction", labelpad=1.5)
    ax.set_ylabel("matched random sets (%)", labelpad=1.5)
    style(ax)
    for v, s_, k in zip(d["value"], share, d["n_draws"]):
        val(name, "bars", int(v), "share of the random sets (%)", float(s_), source="zerok_vs_control_specificity_null_draws_summary.csv")
        val(name, "bars", int(v), "number of random sets", int(k), source="zerok_vs_control_specificity_null_draws_summary.csv")
    val(name, "arrow", "observed", "panel proteins in the expected direction", obs, float(m["null_q025"]),
        float(m["null_q975"]), "zerok_vs_control_specificity_matched_null.csv (low, high = 2.5-97.5% of the random sets)")
    checks.add("matched random sets (DCT1 + DCT2, cluster annotation): draws; observed; null mean [2.5-97.5%]; p",
               f"{int(d['n_draws'].sum())}; {obs} of {slots}; {float(m['null_mean']):.2f} [{float(m['null_q025']):.0f}-"
               f"{float(m['null_q975']):.0f}]; {float(m['empirical_p_greater_or_equal']):.4f}",
               int(d["n_draws"].sum()) == 10000 == int(m["n_draws"]) and (obs, slots) == (7, 7)
               and round(float(m["null_mean"]), 2) == 1.63
               and round(float(m["empirical_p_greater_or_equal"]), 4) == 0.0001,
               "10000; 7 of 7; 1.63 [0-4]; 0.0001")


def fig_matched(run, T, rows, checks) -> bool:
    name = "19_fig_matched_sets"
    sheet = t8.Sheet(MATCHED_H, width=MATCHED_W)
    draw_matched(sheet, T, name, checks)
    return save(run, sheet, name, "F-19: the 7 directional panel proteins against 10,000 matched random protein sets "
                "(DCT1 + DCT2 cells of the cluster annotation)", rows, checks)


def fig_compiled(run, T, rows, checks) -> bool:
    """Figure F-19 as one file (authors, 12 Nov 2026: "Compile F-19"): the two plots above, drawn by the same code at
    the same size on one 180 mm page, A above B, with panel letters. Values are those of the two single plots."""
    global RECORD
    name = "19_fig_F19_compiled"
    pad, gap = 3.0, 3.0                          # room for the letter above each plot; gap between the plots (mm)
    y_b = pad + PANEL_H + gap
    sheet = t8.Sheet(y_b + pad + MATCHED_H)
    scratch = t5.Checks("repeat of the single plots")
    RECORD = False
    try:
        letter(sheet, 3.0, 1.0, "A")
        draw_panel(Sub(sheet, 0.0, pad), T, name, scratch)
        letter(sheet, 3.0, y_b + 1.0, "B")
        draw_matched(Sub(sheet, 0.0, y_b + pad), T, name, scratch)
    finally:
        RECORD = True
    val(name, "A", "all rows", "same values as the plot 19_fig_panel_effects", "see that figure",
        source="zerok_vs_control_panel_results.csv")
    val(name, "B", "all rows", "same values as the plot 19_fig_matched_sets", "see that figure",
        source="zerok_vs_control_specificity_null_draws_summary.csv; zerok_vs_control_specificity_matched_null.csv")
    ok = save(run, sheet, name, "F-19 compiled: (A) pre-specified panel proteins with an expected direction, zero-K "
              "against control; (B) the 7 directional proteins against 10,000 matched random protein sets", rows, checks)
    from collections import Counter
    want = Counter(TEXTS["19_fig_panel_effects"]) + Counter(TEXTS["19_fig_matched_sets"]) + Counter(["A", "B"])
    got = Counter(TEXTS[name])
    checks.add("compiled Figure F-19: drawn texts = the texts of the two single plots plus the letters A and B",
               f"{sum(got.values())} texts; {sum((got - want).values())} extra; {sum((want - got).values())} missing",
               got == want)
    checks.add("compiled Figure F-19: size (mm) = 180 x (3 + panel plot + 3 + 3 + matched-sets plot)",
               f"{rows[-1]['width_mm']} x {rows[-1]['height_mm']}; plots {PANEL_H:.1f} and {MATCHED_H:.1f} mm high",
               abs(rows[-1]["height_mm"] - (PANEL_H + MATCHED_H + 9.0)) < 0.06)
    return ok


# ---------------------------------------------------------------------------
# S-19-a: zero-K annotation, label fractions, cycling proteins
# ---------------------------------------------------------------------------
def fig_s19a(run, T, rows, checks) -> bool:
    name = "19_S19a_zerok_annotation"
    can = T["labels"]
    z = can[can["condition"] == "ZeroK"]
    sheet = t8.Sheet(122.0)
    for k, (col, lab, x0) in enumerate((("cs_zerok_pipeline_label", "A", 9.0), ("cs_zerok_transfer_label", "B", 78.0))):
        letter(sheet, x0 - 6.0, 3.0, lab)
        ax = sheet.ax(x0, 9.0, 54.0, 54.0)
        counts = z[col].astype(str).value_counts().to_dict()
        for value in ("Unknown", "DCT1", "DCT2"):
            sub = z[z[col].astype(str) == value]
            if len(sub):
                ax.scatter(sub["umap1_own"], sub["umap2_own"], s=3.0, c=LABEL_COLOR[value], linewidths=0,
                           rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1", labelpad=2)
        ax.set_ylabel("UMAP 2", labelpad=2)
        style(ax)
        handles = [Line2D([], [], marker="o", ls="", ms=3.4, color=LABEL_COLOR[v], label=f"{LABEL_NAME[v]} ({counts[v]})")
                   for v in ("DCT1", "DCT2", "Unknown") if v in counts]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -6.5 / 54.0), ncol=1, frameon=False,
                  handletextpad=0.2, borderaxespad=0, labelspacing=0.3)
        for v, c in counts.items():
            val(name, lab, LABEL_NAME[v], "zero-K cells", int(c), source="dct_labels_dct_canonical_labels.csv "
                f"({'cluster annotation' if k == 0 else 'label transferred from control cells'}; own UMAP of the zero-K cells)")
        expected = {"DCT1": 449, "Unknown": 253} if k == 0 else {"DCT1": 398, "DCT2": 68, "Unknown": 236}
        checks.add(f"S-19-a {lab}: zero-K cells per label", "; ".join(f"{a} {b}" for a, b in sorted(counts.items())),
                   counts == expected and len(z) == 702, str(expected))
    # C label fractions
    letter(sheet, 141.0, 3.0, "C")
    fr = T["fractions"]
    bars = [("Control", PRI, "Ctrl"), ("Zero-K", PRI, "ZeroK"), ("Zero-K (t)", ALT, "ZeroK")]
    ax = sheet.ax(160.0, 9.0, 16.0, 54.0)
    for i, (lab, col, cond) in enumerate(bars):
        sel = pick(fr, label_column=col, condition=cond, plate="all").set_index("label")
        bottom = 0.0
        for value in ("DCT1", "DCT2", "Unknown"):
            f = 100 * float(sel.loc[value, "fraction"])
            if f > 0:
                ax.bar(i, f, bottom=bottom, width=0.7, color=LABEL_COLOR[value], edgecolor="white", linewidth=0.6)
            val(name, "C", lab, f"{LABEL_NAME[value]} (% of cells; cells in the column value_low)", f,
                int(sel.loc[value, "n_label"]), int(sel.loc[value, "n_cells"]), "zerok_vs_control_label_fractions.csv")
            bottom += f
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], rotation=90)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("cells (%)", labelpad=1.5)
    style(ax)
    # D cycling proteins
    letter(sheet, 3.0, 86.0, "D")
    cyc = pick(T["cycling"], label_column=FREE, group="all_cells").set_index("measure")
    measures = [("Mki67_detected", "Ki-67 (Mki67) detected"), ("Top2a_detected", "Top2a detected"),
                ("Mki67_or_Top2a_detected", "Ki-67 or Top2a detected"),
                ("cycling_score_rule__none", "High cell-cycle score")]
    n = len(measures)
    for i, (_, lab) in enumerate(measures):
        sheet.text(9.0, 93.0 + (i + 0.5) * 5.0, lab, va="center")
    ax = sheet.ax(46.0, 93.0, 60.0, n * 5.0)
    y = np.arange(n)
    for cond, off in (("Ctrl", -0.17), ("ZeroK", 0.17)):
        k = cond.lower()
        f = np.array([100 * float(cyc.loc[m, f"fraction_{k}"]) for m, _ in measures])
        lo = np.array([100 * float(cyc.loc[m, f"ci_low_{k}"]) for m, _ in measures])
        hi = np.array([100 * float(cyc.loc[m, f"ci_high_{k}"]) for m, _ in measures])
        ax.hlines(y + off, lo, hi, color=COND[cond], lw=0.7)
        ax.plot(f, y + off, "o", ms=3.2, color=COND[cond], ls="", label=f"{COND_NAME[cond]} (n = {int(cyc[f'n_{k}'].iloc[0])})")
        for (m, lab), a, b, c in zip(measures, f, lo, hi):
            val(name, "D", lab, f"{COND_NAME[cond]}: cells (%), Wilson 95% interval; {int(cyc.loc[m, f'k_{k}'])} of "
                f"{int(cyc.loc[m, f'n_{k}'])} cells", a, b, c, "zerok_vs_control_state_cycling.csv (all retained cells)")
    ax.set_xlim(0, 25)
    ax.set_xlabel("cells (%)", labelpad=1.5)
    row_axis(ax, n)
    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), ncol=1, frameon=False, handletextpad=0.2, borderaxespad=0)
    k_c, k_z = int(cyc.loc["Mki67_or_Top2a_detected", "k_ctrl"]), int(cyc.loc["Mki67_or_Top2a_detected", "k_zerok"])
    checks.add("S-19-a D: cells with Ki-67 or Top2a detected, all retained cells, control / zero-K",
               f"{k_c} of {int(cyc['n_ctrl'].iloc[0])} / {k_z} of {int(cyc['n_zerok'].iloc[0])}",
               (k_c, k_z, int(cyc["n_ctrl"].iloc[0]), int(cyc["n_zerok"].iloc[0])) == (21, 56, 729, 702),
               "21 of 729 / 56 of 702")
    return save(run, sheet, name, "S-19-a: zero-K DCT cells: cluster annotation (A) and label transferred from control "
                "cells (B) on the UMAP of the zero-K cells; label fractions (C); cells with cycling proteins (D)", rows,
                checks)


# ---------------------------------------------------------------------------
# S-19-b: every panel item in every group and normalization variant; robustness
# ---------------------------------------------------------------------------
def fig_s19b(run, T, rows, checks, items: list[str], low: tuple[str, ...]) -> bool:
    name = "19_S19b_panel_overview"
    res = pd.concat([T["panel"], T["lf_panel"]], ignore_index=True)
    idx = res.set_index(["label_column", "group", "variant", "item"])
    blocks = [("Cluster annotation", [(PRI, "DCT1"), (PRI, "all_DCT"), (PRI, "Unknown")]),
              ("Transferred zero-K label", [(ALT, "DCT1"), (ALT, "DCT2"), (ALT, "all_DCT"), (ALT, "Unknown")]),
              ("No labels", [(FREE, "all_cells"), (FREE, "Slc12a3_detected")])]
    cw, ch, top = 3.6, 3.3, 36.0
    name_w = max(t8.text_mm(item_name(i, low))[0] for i in items) + 2.0
    x0 = 3.0 + name_w + 8.0
    n = len(items)
    bot = top + n * ch
    hb, cwb, chb = bot + 32.0, 9.0, 4.2
    sheet = t8.Sheet(hb + 8.0 * chb + 4.0)
    letter(sheet, 3.0, 3.0, "A")
    exp = pick(T["panel"], label_column=PRI, group="all_DCT", variant="none").set_index("item")["expected_direction"]
    sheet.text(3.0 + name_w, top - 1.0, "Exp.", va="bottom", color=INK["secondary"], fontweight="bold")
    for i, item in enumerate(items):
        yy = top + (i + 0.5) * ch
        sheet.text(3.0, yy, item_name(item, low), va="center")
        sheet.text(3.0 + name_w + 1.5, yy, EXP_SHORT[exp[item]], va="center")
    x = x0
    counts = {}
    for title, groups in blocks:
        sheet.text(x, 7.0, title, va="top", fontweight="bold", color=INK["secondary"])
        for col, g in groups:
            for v in lib30.VARIANTS:
                for i, item in enumerate(items):
                    r = idx.loc[(col, g, v, item)]
                    sheet.rect(x, top + i * ch, cw - 0.3, ch - 0.3, facecolor=DIR_COLOR[r["observed_direction"]],
                               edgecolor=INK["grid"], linewidth=0.3)
                    cxm, cym = (x + (cw - 0.3) / 2) / sheet.W, 1 - (top + i * ch + (ch - 0.3) / 2) / sheet.H
                    if r["match_status"] == "match":
                        sheet.fig.add_artist(Line2D([cxm], [cym], marker="o", ms=1.8, color=INK["primary"],
                                                    transform=sheet.fig.transFigure))
                    elif r["match_status"] == "mismatch":
                        sheet.fig.add_artist(Line2D([cxm], [cym], marker="x", ms=2.6, mew=0.7, color=INK["primary"],
                                                    transform=sheet.fig.transFigure))
                    val(name, "A", item_name(item), f"{GROUP[(col, g)]}; variant {'abc'[lib30.VARIANTS.index(v)]}",
                        f"{r['observed_direction']} / {r['match_status']}",
                        source="zerok_vs_control_panel_results.csv" if col != FREE else "zerok_vs_control_robustness_label_free_panel.csv")
                sheet.text(x + (cw - 0.3) / 2, top - 1.0, "abc"[lib30.VARIANTS.index(v)], ha="center", va="bottom")
                if v == "cell_median":
                    sheet.text(x + (cw - 0.3) / 2, top - 5.0, GROUP[(col, g)].replace(" (t)", ""), ha="center",
                               va="bottom", rotation=90)
                if v == "none":
                    sub = pick(res, label_column=col, group=g, variant=v)
                    sub = sub[sub["expected_direction"].isin([t22.UP, t22.DOWN, t22.NO_CHANGE])]
                    counts[GROUP[(col, g)]] = int(sub["match_status"].eq("match").sum())
                x += cw
            x += 0.8
        x += 2.5
    ly, lx = bot + 4.0, 3.0
    drawn = set(res.loc[res["variant"].isin(lib30.VARIANTS), "observed_direction"])
    checks.add("S-19-b A: observed directions drawn (the key lists these only)", "; ".join(sorted(drawn)), None)
    for key in (t22.UP, t22.DOWN, t22.OBS_MIXED, t22.OBS_NS):
        if key not in drawn:
            continue
        sheet.rect(lx, ly, 3.0, 2.6, facecolor=DIR_COLOR[key], edgecolor=INK["grid"], linewidth=0.3)
        sheet.text(lx + 4.0, ly + 1.3, DIR_NAME[key], va="center")
        lx += 4.0 + t8.text_mm(DIR_NAME[key])[0] + 4.0
    ly += 5.0
    for lx, mk, kw, text in ((3.0, "o", dict(ms=1.8), "as expected"), (26.0, "x", dict(ms=2.6, mew=0.7), "not as expected")):
        sheet.fig.add_artist(Line2D([(lx + 1.5) / sheet.W], [1 - (ly + 1.3) / sheet.H], marker=mk, color=INK["primary"],
                                    transform=sheet.fig.transFigure, **kw))
        sheet.text(lx + 4.0, ly + 1.3, text, va="center")
    checks.add("S-19-b A: scored items in the expected direction (of 8), ln(1 + intensity), per group",
               "; ".join(f"{k} {v}" for k, v in counts.items()),
               counts == {"DCT1": 7, "DCT1 + DCT2": 7, "not resolved": 3, "DCT1 (t)": 7, "DCT2 (t)": 2,
                          "DCT1 + DCT2 (t)": 7, "not resolved (t)": 4, "all cells": 7, "NCC detected": 7},
               "7; 7; 3; 7; 2; 7; 4; 7; 7")
    # B robustness of the scored items
    letter(sheet, 3.0, hb - 13.0, "B")
    rep = pick(T["panel_repeats"], variant="none").set_index(["procedure", "label_column", "group", "item"])
    cmap = t8.sequential_cmap()
    procs = [("depth_matched", "depth"), ("downsample_50pct", "50%"), ("downsample_25pct", "25%")]
    xb = 3.0 + max(t8.text_mm(PROT[i])[0] for i in PROT) + 3.0
    for i, item in enumerate(PROT):
        sheet.text(3.0, hb + (i + 0.5) * chb, PROT[item], va="center")
    shown = {}
    for j, (col, g) in enumerate(((PRI, "all_DCT"), (PRI, "DCT1"), (FREE, "all_cells"))):
        gx = xb + j * (3 * cwb + 3.0)
        sheet.text(gx + 1.5 * cwb, hb - 6.0, GROUP[(col, g)], ha="center", va="bottom", fontweight="bold",
                   color=INK["secondary"])
        for k, (p, plab) in enumerate(procs):
            sheet.text(gx + (k + 0.5) * cwb, hb - 1.0, plab, ha="center", va="bottom")
            for i, item in enumerate(PROT):
                f = float(rep.loc[(p, col, g, item), "fraction_same_match_status"])
                sheet.rect(gx + k * cwb, hb + i * chb, cwb - 0.3, chb - 0.3, facecolor=cmap(f), edgecolor=INK["grid"],
                           linewidth=0.3)
                sheet.text(gx + (k + 0.5) * cwb - 0.15, hb + (i + 0.5) * chb - 0.15, f"{100 * f:.0f}", ha="center",
                           va="center", color="white" if f > 0.55 else INK["primary"])
                val(name, "B", PROT[item], f"{GROUP[(col, g)]}; {plab}: share of 100 repeats with the same result as "
                    "all cells (%)", 100 * f, source="zerok_vs_control_robustness_panel_repeats.csv")
                shown[(g if col == PRI else "all_cells", p, item)] = f
    got = [round(shown[("all_DCT", "depth_matched", i)], 2) for i in PROT]
    checks.add("S-19-b B: DCT1 + DCT2, depth-matched: share of repeats with the same result (panel order)",
               "; ".join(f"{i} {v:.2f}" for i, v in zip(PROT, got)),
               got == [1.0, 1.0, 1.0, 1.0, 1.0, 0.16, 0.31, 0.9], "1; 1; 1; 1; 1; 0.16; 0.31; 0.9")
    return save(run, sheet, name, "S-19-b: observed direction of all pre-specified panel items in every cell group and "
                "normalization variant (A); share of 100 repeats keeping the result of the scored proteins (B)", rows,
                checks)


# ---------------------------------------------------------------------------
# S-19-c: specificity and the sort-plate null
# ---------------------------------------------------------------------------
def fig_s19c(run, T, rows, checks) -> bool:
    name = "19_S19c_specificity"
    nulls, ranks, slots, plate = T["spec_null"], T["spec_ranks"], T["spec_slots"], T["spec_plate"]
    sheet = t8.Sheet(112.0)
    lab8 = [GROUP[k] for k in ROWS8]
    y = np.arange(len(ROWS8))
    for k, (stat, lt, xlabel) in enumerate((("S1 matches", "A", "proteins in the\nexpected direction"),
                                            ("S2 signed pseudobulk log2FC", "B", "mean signed log2 ratio"))):
        x0 = 32.0 + k * 46.0
        letter(sheet, 3.0 if k == 0 else x0 - 5.0, 3.0, lt)
        ax = sheet.ax(x0, 13.0, 30.0, 42.0)
        s = pick(nulls, variant="none", statistic=stat).set_index(["label_column", "group"]).reindex(ROWS8)
        ax.hlines(y, s["null_q025"], s["null_q975"], color=INK["muted"], lw=1.6)
        ax.plot(s["null_mean"], y, "|", ms=5, color=INK["muted"], ls="")
        ax.plot(s["observed"], y, "o", ms=3.4, color=INK["primary"], ls="")
        for i, p in enumerate(s["empirical_p_greater_or_equal"]):
            ax.text(1.03, i, f"{p:.2g}", transform=ax.get_yaxis_transform(), va="center", fontsize=FONT)
        ax.text(1.03, -1.0, "p", transform=ax.get_yaxis_transform(), va="center", fontsize=FONT, color=INK["secondary"])
        ax.set_yticks(y)
        ax.set_yticklabels(lab8 if k == 0 else [])
        ax.set_ylim(len(ROWS8) - 0.5, -0.5)
        ax.set_xlabel(xlabel, labelpad=1.5)
        if k == 0:
            ax.set_xlim(-0.4, 7.4)
            ax.set_xticks([0, 2, 4, 6])
        style(ax)
        for key, r in s.iterrows():
            val(name, lt, GROUP[key], "observed (low, high = 2.5-97.5% of 10,000 matched random sets)",
                float(r["observed"]), float(r["null_q025"]), float(r["null_q975"]), "zerok_vs_control_specificity_matched_null.csv")
            val(name, lt, GROUP[key], "mean of the random sets", float(r["null_mean"]), source="zerok_vs_control_specificity_matched_null.csv")
            val(name, lt, GROUP[key], "empirical one-sided p", float(r["empirical_p_greater_or_equal"]),
                source="zerok_vs_control_specificity_matched_null.csv")
        if k == 0:
            got = [int(v) for v in s["observed"]]
            checks.add("S-19-c A: proteins in the expected direction per group (row order of the figure)",
                       "; ".join(f"{a} {b}" for a, b in zip(lab8, got)), got == [7, 7, 2, 7, 7, 1, 2, 7],
                       "7; 7; 2; 7; 7; 1; 2; 7")
    handles = [Line2D([], [], marker="o", ls="", ms=3.4, color=INK["primary"], label="observed"),
               Line2D([], [], marker="|", ls="-", lw=1.6, ms=5, color=INK["muted"], label="random sets: mean, 2.5–97.5%")]
    sheet.fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(32.0 / sheet.W, 1 - 66.0 / sheet.H), ncol=2,
                     frameon=False, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
    # C directional percentiles
    letter(sheet, 119.0, 3.0, "C")
    r = pick(ranks, label_column=PRI, group="all_DCT", variant="none")
    r = r[r["expected_direction"].isin([t22.UP, t22.DOWN])].set_index("protein").loc[[p for p in PROT if p != "Cul3"]]
    sl = pick(slots, label_column=PRI, group="all_DCT", variant="none").set_index("protein").reindex(r.index)
    ax = sheet.ax(146.0, 13.0, 30.0, 30.0)
    yy = np.arange(len(r))
    ax.plot(r["pseudobulk_directional_percentile"], yy, "o", ms=3.4, color=INK["primary"], ls="", label="all proteins")
    ax.plot(sl["pseudobulk_directional_percentile_among_neighbours"], yy, "o", ms=3.4, mfc="white", mec=INK["primary"],
            mew=0.6, ls="", label="50 matched")
    ax.axvline(0.5, color=INK["muted"], lw=0.5, zorder=0)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{p}, {EXP[e]}" for p, e in zip(r.index, r["expected_direction"])])
    ax.set_ylim(len(r) - 0.5, -0.5)
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("directional percentile", labelpad=1.5)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.05), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0)
    style(ax)
    for p in r.index:
        val(name, "C", p, "directional percentile among all tested proteins", float(r.loc[p, "pseudobulk_directional_percentile"]),
            source="zerok_vs_control_specificity_panel_ranks.csv")
        val(name, "C", p, "directional percentile among the 50 matched proteins",
            float(sl.loc[p, "pseudobulk_directional_percentile_among_neighbours"]), source="zerok_vs_control_specificity_matched_slots.csv")
    # D sort-plate null
    letter(sheet, 3.0, 70.0, "D")
    pn = pick(plate, variant="none")
    ax = sheet.ax(32.0, 81.0, 62.0, 22.0)
    marks = {"Ctrl P2 vs P1": ("o", COND["Ctrl"]), "ZeroK P2 vs P1": ("o", COND["ZeroK"]),
             "ZeroK vs Ctrl, P1 only": ("^", PLATE["P1"]), "ZeroK vs Ctrl, P2 only": ("s", PLATE["P2"])}
    for cname, (mk, colr) in marks.items():
        s = pick(pn, contrast=cname).set_index(["label_column", "group"]).reindex(ROWS5)
        ax.plot(100 * s["share_q_below_0.05"], np.arange(len(ROWS5)), mk, ms=3.4, color=colr, ls="", label=CONTRAST[cname])
        for key, v, nt in zip(ROWS5, s["share_q_below_0.05"], s["n_proteins_tested"]):
            val(name, "D", GROUP[key], f"{CONTRAST[cname]}: tested proteins with q < 0.05 (%); proteins tested in low",
                100 * float(v), int(nt), source="zerok_vs_control_specificity_plate_null.csv")
    pooled = T["proteome_summary"].query("variant == 'none'").set_index(["label_column", "group"])
    pv = [100 * float(pooled.loc[k, "share_q_below_0.05"]) if k in pooled.index else np.nan for k in ROWS5]
    ax.plot(pv, np.arange(len(ROWS5)), "D", ms=3.2, mfc="white", mec=INK["primary"], mew=0.6, ls="",
            label=CONTRAST["ZeroK vs Ctrl (pooled)"])
    for key, v in zip(ROWS5, pv):
        val(name, "D", GROUP[key], f"{CONTRAST['ZeroK vs Ctrl (pooled)']}: tested proteins with q < 0.05 (%)", v,
            source="zerok_vs_control_proteome_summary.csv (not computed for all cells)")
    ax.set_yticks(range(len(ROWS5)))
    ax.set_yticklabels([GROUP[k] for k in ROWS5])
    ax.set_ylim(len(ROWS5) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("tested proteins with q < 0.05 (%)", labelpad=1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), ncol=1, frameon=False, handletextpad=0.2, borderaxespad=0)
    style(ax)
    free = pick(pn, label_column=FREE, group="all_cells").set_index("contrast")["share_q_below_0.05"]
    got = [round(100 * float(free[c]), 1) for c in marks]
    checks.add("S-19-c D: all cells, tested proteins with q < 0.05 (%): control P2 vs P1; zero-K P2 vs P1; zero-K vs "
               "control in P1; in P2", "; ".join(map(str, got)), got == [41.7, 54.2, 54.7, 37.8], "41.7; 54.2; 54.7; 37.8")
    got = [round(pv[0], 1), round(pv[1], 1)]
    checks.add("S-19-c D: zero-K vs control, both plates, tested proteins with q < 0.05 (%): DCT1 + DCT2; DCT1",
               "; ".join(map(str, got)), got == [57.2, 56.8], "57.2; 56.8")
    return save(run, sheet, name, "S-19-c: the directional panel proteins against matched random protein sets (A, B), "
                "their directional percentiles (C), and the sort-plate null (D)", rows, checks)


# ---------------------------------------------------------------------------
# S-19-d: bulk low-K whole-kidney proteome
# ---------------------------------------------------------------------------
def side_labels(ax, points: list[tuple[str, float, float]], gap_mm: float = 3.6) -> None:
    """Names in a column right of the axes, sorted by y and spread apart, joined to their points by lines (zerok_vs_control)."""
    if not points:
        return
    ylo, yhi = ax.get_ylim()
    points = sorted(points, key=lambda t: t[2])
    gap = gap_mm / (ax.get_position().height * ax.figure.get_figheight() * 25.4)
    yf = [min(max((y - ylo) / (yhi - ylo), 0.03), 0.97) for _, _, y in points]
    for i in range(1, len(yf)):
        yf[i] = max(yf[i], yf[i - 1] + gap)
    over = yf[-1] - 0.98
    if over > 0:
        yf = [v - over for v in yf]
        for i in range(len(yf) - 2, -1, -1):
            yf[i] = min(yf[i], yf[i + 1] - gap)
    for (g, x, y), v in zip(points, yf):
        ax.annotate("", xy=(x, y), xycoords="data", xytext=(1.035, v), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=INK["muted"], shrinkA=0, shrinkB=2))
        ax.text(1.04, v, g, transform=ax.transAxes, va="center", ha="left", fontsize=FONT)


def fig_s19d(run, T, rows, checks) -> bool:
    name = "19_S19d_bulk_benchmark"
    conc, nulls, signs, matched = T["bulk_conc"], T["bulk_null"], T["bulk_signs"], T["bulk_matched"]
    sheet = t8.Sheet(122.0)
    letter(sheet, 3.0, 3.0, "A")
    d = pick(matched, label_column=PRI, group="all_DCT", variant="none")
    ax = sheet.ax(16.0, 10.0, 56.0, 56.0)
    x, y = d["bulk_log2fc"].to_numpy(float), d["pseudobulk_log2fc_pooled"].to_numpy(float)
    sig = (d["bulk_q"] < 0.05).to_numpy()
    enr = d["dct_enriched"].astype(str).str.lower().eq("true").to_numpy()
    ax.scatter(x[~sig], y[~sig], s=1.2, c=INK["grid"], linewidths=0, rasterized=True)
    ax.scatter(x[sig], y[sig], s=1.6, c=INK["muted"], linewidths=0, rasterized=True)
    ax.scatter(x[enr], y[enr], s=5, c=t8.SERIES[0], linewidths=0, zorder=3)
    lx = float(np.nanquantile(np.abs(x), 0.999)) * 1.05
    ly = float(np.nanquantile(np.abs(y[np.isfinite(y)]), 0.995)) * 1.05
    ax.set_xlim(-lx, lx)
    ax.set_ylim(-ly, ly)
    ax.axhline(0, color=INK["muted"], lw=0.5, zorder=0)
    ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    pts = []
    for g in PROT:
        r = d.loc[d["protein"] == g]
        if len(r) and np.isfinite(r["pseudobulk_log2fc_pooled"].iloc[0]):
            px = float(np.clip(r["bulk_log2fc"].iloc[0], -lx, lx))
            py = float(np.clip(r["pseudobulk_log2fc_pooled"].iloc[0], -ly, ly))
            ax.scatter([px], [py], s=16, facecolors="none", edgecolors=INK["primary"], linewidths=0.6, zorder=4,
                       clip_on=False)
            pts.append((g, px, py))
            val(name, "A", g, "ring: bulk log2 fold change (value), SCP-MS log2 ratio (low); drawn at the axis edge "
                "when beyond the range", float(r["bulk_log2fc"].iloc[0]), float(r["pseudobulk_log2fc_pooled"].iloc[0]),
                source="zerok_vs_control_bulk_matched.csv")
    side_labels(ax, [p for p in pts if p[1] >= 0])
    left = sorted([p for p in pts if p[1] < 0], key=lambda t: -t[1])
    step = 3.6 / 56.0 * (2 * ly)
    for i, (g, px, py) in enumerate(left):
        ty = -0.35 * ly - i * step
        ax.annotate("", xy=(px, py), xycoords="data", xytext=(-0.64 * lx, ty), textcoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=INK["muted"], shrinkA=0, shrinkB=2))
        ax.text(-0.62 * lx, ty, g, va="center", ha="left", fontsize=FONT)
    c = pick(conc, label_column=PRI, group="all_DCT", variant="none", scp_effect="pseudobulk", protein_set="all shared").iloc[0]
    ax.text(0.03, 0.97, f"ρ = {c['spearman']:.2f}\nn = {int(c['n'])}", transform=ax.transAxes, va="top", fontsize=FONT)
    ax.set_xlabel("whole kidney, 0% vs 1% K$^{+}$ diet (log2 fold change)", labelpad=1.5)
    ax.set_ylabel("DCT cells, zero-K vs control (log2 ratio)", labelpad=1.5)
    style(ax)
    val(name, "A", "all shared proteins", "points drawn (bulk q < 0.05 dark; DCT-enriched blue)", int(len(d)),
        int(sig.sum()), int(enr.sum()), "zerok_vs_control_bulk_matched.csv (low = bulk q < 0.05, high = DCT-enriched)")
    # B correlations
    lab5 = [GROUP[k] for k in ROWS5]
    marks = {"ZeroK vs Ctrl (pooled)": ("o", INK["primary"], INK["primary"]),
             "ZeroK vs Ctrl, P1 only": ("^", PLATE["P1"], PLATE["P1"]),
             "ZeroK vs Ctrl, P2 only": ("s", PLATE["P2"], PLATE["P2"]),
             "Ctrl P2 vs P1 (technical)": ("o", COND["Ctrl"], "white"),
             "ZeroK P2 vs P1 (technical)": ("o", COND["ZeroK"], "white")}
    letter(sheet, 104.0, 3.0, "B")
    got = {}
    for k, (pset, title) in enumerate((("all shared", "all shared proteins"),
                                       ("bulk q < 0.05", "proteins with q < 0.05 in whole kidney"))):
        ax = sheet.ax(128.0, 24.0 + k * 34.0, 48.0, 25.0)
        s0 = pick(nulls, variant="none", protein_set=pset)
        offs = np.linspace(-0.28, 0.28, len(marks))
        for (cname, (mk, ec, fc)), off in zip(marks.items(), offs):
            s = pick(s0, contrast=cname).set_index(["label_column", "group"]).reindex(ROWS5)
            yy = np.arange(len(ROWS5)) + off
            ax.hlines(yy, s["ci_low"], s["ci_high"], color=ec, lw=0.6)
            ax.plot(s["spearman"], yy, mk, ms=3, mec=ec, mfc=fc, mew=0.6, ls="", label=CONTRAST[cname])
            for key, r in s.iterrows():
                val(name, "B", GROUP[key], f"{title}; {CONTRAST[cname]}: Spearman rho (Fisher-z 95% interval); n = {int(r['n'])}",
                    float(r["spearman"]), float(r["ci_low"]), float(r["ci_high"]), "zerok_vs_control_bulk_plate_null.csv")
                got[(pset, cname, key)] = (float(r["spearman"]), int(r["n"]))
        ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
        ax.set_yticks(range(len(ROWS5)))
        ax.set_yticklabels(lab5)
        ax.set_ylim(len(ROWS5) - 0.5, -0.5)
        ax.set_title(title, fontsize=FONT, pad=2)
        if k == 1:
            ax.set_xlabel("Spearman ρ with whole kidney", labelpad=1.5)
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(-0.52, 1.14), ncol=2, frameon=False, handletextpad=0.2,
                      columnspacing=0.6, borderaxespad=0)
        style(ax)
    key = (PRI, "all_DCT")
    a = got[("all shared", "ZeroK vs Ctrl (pooled)", key)]
    b = got[("bulk q < 0.05", "ZeroK vs Ctrl (pooled)", key)]
    t1 = got[("all shared", "Ctrl P2 vs P1 (technical)", key)][0]
    t2 = got[("all shared", "ZeroK P2 vs P1 (technical)", key)][0]
    checks.add("S-19-d B: DCT1 + DCT2: rho all shared (n); rho bulk-significant (n); plate contrasts, all shared",
               f"{a[0]:.2f} ({a[1]}); {b[0]:.2f} ({b[1]}); {t1:.2f}, {t2:.2f}",
               (round(a[0], 2), a[1], round(b[0], 2), b[1], round(t1, 2), round(t2, 2)) == (0.13, 3445, 0.35, 715, -0.03, -0.11),
               "0.13 (3445); 0.35 (715); -0.03, -0.11")
    # C sign agreement
    letter(sheet, 3.0, 80.0, "C")
    ax = sheet.ax(30.0, 90.0, 42.0, 22.0)
    for k, (pset, colr, lab) in enumerate((("all shared", INK["primary"], "all shared proteins"),
                                           ("DCT-enriched", t8.SERIES[0], "DCT-enriched proteins"))):
        s = pick(signs, variant="none", protein_set=pset).set_index(["label_column", "group"]).reindex(ROWS5)
        yy = np.arange(len(ROWS5)) + (k - 0.5) * 0.3
        ax.hlines(yy, 100 * s["agreement_ci_low"], 100 * s["agreement_ci_high"], color=colr, lw=0.6)
        ax.plot(100 * s["agreement"], yy, "o", ms=3, color=colr, ls="", label=lab)
        for key2, r in s.iterrows():
            val(name, "C", GROUP[key2], f"{lab}: same direction (%), Wilson 95% interval; proteins significant in both: "
                f"{int(r['n_both_significant'])}", 100 * float(r["agreement"]), 100 * float(r["agreement_ci_low"]),
                100 * float(r["agreement_ci_high"]), "zerok_vs_control_bulk_sign_tables.csv")
    ax.axvline(50, color=INK["muted"], lw=0.5, zorder=0)
    ax.set_yticks(range(len(ROWS5)))
    ax.set_yticklabels(lab5)
    ax.set_ylim(len(ROWS5) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("same direction (%)", labelpad=1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), frameon=False, handletextpad=0.2, borderaxespad=0)
    style(ax)
    r = pick(signs, variant="none", protein_set="all shared", label_column=PRI, group="all_DCT").iloc[0]
    checks.add("S-19-d C: DCT1 + DCT2, all shared: proteins significant in both; same direction; odds ratio",
               f"{int(r['n_both_significant'])}; {float(r['agreement']):.2f}; {float(r['odds_ratio']):.2f}",
               (int(r["n_both_significant"]), round(float(r["agreement"]), 2), round(float(r["odds_ratio"]), 2)) == (411, 0.64, 3.59),
               "411; 0.64; 3.59")
    return save(run, sheet, name, "S-19-d: zero-K against control in DCT cells compared with a bulk whole-kidney "
                "proteome of mice on a 0% or 1% potassium diet: per-protein effects (A), correlations (B), direction (C)",
                rows, checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: figures for the zero-K response (F-19, S-19-a to d).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("dct_joint")
    lib30.check_inputs(run, checks)          # plan hashes against the freeze record; frozen zerok_core panel hash
    rd = lambda n, **kw: lib30.read_table(run, n, **kw)  # noqa: E731
    panel = rd("panel_results")
    T = {"panel": panel[panel["status"] == "tested"], "lf_panel": rd("robustness_label_free_panel"),
         "panel_repeats": rd("robustness_panel_repeats"), "fractions": rd("label_fractions"),
         "cycling": rd("state_cycling"), "spec_null": rd("specificity_matched_null"),
         "spec_draws": rd("specificity_null_draws_summary"), "spec_ranks": rd("specificity_panel_ranks"),
         "spec_slots": rd("specificity_matched_slots"), "spec_plate": rd("specificity_plate_null"),
         "proteome_summary": rd("proteome_summary"), "bulk_conc": rd("bulk_concordance"),
         "bulk_null": rd("bulk_plate_null"), "bulk_signs": rd("bulk_sign_tables"), "bulk_matched": rd("bulk_matched"),
         "labels": pd.read_csv(run.use_input(lib30.LABELS_CSV, "dct_labels canonical DCT labels (combined search)"),
                               dtype={"cell_id": str, "well": str}, keep_default_na=False, na_values=[""])}
    items = list(t22.panel_spec_items(t22.read_panel(run)))
    base = pick(T["panel"], label_column=PRI, group="all_DCT", variant="none", item_type="protein")
    low = tuple(base.loc[np.minimum(base["detection_fraction_ctrl"], base["detection_fraction_zerok"])
                         < t22.MIN_DETECTED_FRACTION, "item"])
    checks.add("panel items (22 proteins, 6 scores); proteins detected in < 10% of cells in a condition (marked *)",
               f"{len(items)} items; {', '.join(low)}", len(items) == 28 and low == ("Wnk1", "Wnk4", "Trpm6", "Mki67", "Top2a"),
               "28; Wnk1, Wnk4, Trpm6, Mki67, Top2a")
    rows: list[dict] = []
    ok = [fig_panel(run, T, rows, checks), fig_matched(run, T, rows, checks), fig_compiled(run, T, rows, checks),
          fig_s19a(run, T, rows, checks),
          fig_s19b(run, T, rows, checks, items, low), fig_s19c(run, T, rows, checks), fig_s19d(run, T, rows, checks)]
    run.table(pd.DataFrame(VALUES), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"figures": [r["figure"] for r in rows], "formats": list(FORMATS), "font": "Arial", "min_font_pt": FONT,
                "dpi": 600, "low_coverage_marked": list(low), "new_statistics": "none (stored zerok_vs_control and dct_labels values)",
                "figures_pass": int(sum(ok)), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        failed = [r["check"] for r in checks.rows if not r["pass"]]
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv: {failed}")


if __name__ == "__main__":
    main()
