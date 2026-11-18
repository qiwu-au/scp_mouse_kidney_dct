#!/usr/bin/env python
"""zerok_vs_control step 7: figures for the zero-K vs control comparison on the combined search (dotplot format).

Analysis plan,
step 7. Same figures and layout as zerok_response_core_figures.py (panel helpers ported; text checks and file checks from
atlas_dct_lib.save_checked), with these changes for the combined search:
- PDF + 600-dpi PNG + TIFF (LZW);
- design text: condition and pool cannot be separated (both conditions are in one search);
- labels from dct_labels (cs_ctrl_deg_label; ZeroK cs_zerok_pipeline_label primary, cs_zerok_transfer_label alternative);
- groups whose label has no cells in a condition (primary DCT2, ProLIF under both label columns) are not drawn and
  are named 'does not apply' in the panel or the note;
- '*' marks panel proteins detected in < 10% of cells in a condition (primary labels, all_DCT, variant none),
  computed from the step 3 table.
Reads the zerok_vs_control tables only (no new statistics).

Main-text candidate (zerok_vs_control_main_candidate): (A) label composition by condition; (B) pre-specified panel,
two-part effects beside the frozen expected direction (all_DCT, dct_label, variant none); (C) volcano (same
group); (D) DCT1 - DCT2 score by label and condition.
Supplementary: composition by plate and fraction differences; contamination detection; panel overview; volcanoes
per label; volcanoes per variant and label-free; state shifts; robustness; specificity (5b); bulk benchmark (5c).

Inputs (read-only): results/tables/zerok_vs_control/zerok_vs_control_*.csv (steps 2-6).
Outputs: results/figures/zerok_vs_control/zerok_vs_control_*.pdf/.png/.tif;
results/tables/zerok_vs_control/zerok_vs_control_figure_checks.csv.

Command (project root):  python scripts/zerok_vs_control_figures.py
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
import time

import numpy as np
import pandas as pd

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import atlas_dct_lib as t8
import zerok_vs_control_lib as lib

plt = t8.setup_matplotlib()
SCRIPT = "zerok_vs_control_figures.py"
t22 = lib.t22
INK = t8.INK
FONT = t8.FONT
COND = t22.CONDITION_PALETTE
PLATE = cc.PLATE_PALETTE
LABEL_COLORS = dict(t2.LABEL_PALETTE)
DIR_COLORS = {t22.UP: "#eb6834", t22.DOWN: "#2a78d6", t22.OBS_MIXED: "#4a3aa7", t22.OBS_NS: "#d6d5cf",
              t22.NOT_TESTABLE: "#ffffff"}
DIR_SHORT = {t22.UP: "up", t22.DOWN: "down", t22.OBS_MIXED: "mixed", t22.OBS_NS: "n.s.", t22.NOT_TESTABLE: "n.t."}
EXP_SHORT = {t22.UP: "up", t22.DOWN: "down", t22.NO_CHANGE: "n.c.", t22.NO_EXPECTATION: "–"}
MATCH_SHORT = {"match": "match", "mismatch": "mismatch", t22.NOT_TESTABLE: "n.t.", t22.NOT_SCORED: ""}
SCORE_SHORT = {"ALDO_activity (base)": "ALDO activity", "ALDO_activity (no_Slc12a3)": "ALDO activity, no Slc12a3",
               "ALDO_signed (base)": "ALDO signed", "ALDO_signed (no_Slc12a3)": "ALDO signed, no Slc12a3",
               "ALDO_machinery": "ALDO machinery", "cell_cycle_score (18-protein panel)": "Cell-cycle score"}
VARIANT_LETTER = {"none": "a", "cell_median": "b", "condition_align": "c"}
LOW_COVERAGE: tuple[str, ...] = ()      # set in main() from the step 3 table
EXPECTED_PROTEINS = ("Slc12a3", "Wnk1", "Wnk4", "Klhl3", "Cul3", "Pvalb", "Calb1", "Slc8a1")
DESIGN_TEXT = ("One pool of 3 mice per condition: condition and pool cannot be separated; cells are not "
               "animal-level replicates, so every p-value is descriptive. Ctrl and ZeroK are in one combined "
               "Spectronaut search.")
ABSENT_TEXT = ("Under the frozen dct_labels rule ProLIF has no cells under either label column and the primary labels "
               "have no ZeroK DCT2 cells; those groups do not apply.")
LINE = 0.5


def letter(sheet, x, y, s):
    sheet.text(x, y, s, fontsize=8, fontweight="bold", va="top")


def style(ax):
    t8.style_axes(ax)
    ax.tick_params(labelsize=FONT, colors=INK["secondary"])


def short_item(item: str) -> str:
    name = SCORE_SHORT.get(item, item)
    return f"{name}*" if item in LOW_COVERAGE else name


def text_width(s: str) -> float:
    return t8.text_mm(s)[0]


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


def save(run, sheet, name, desc, rows) -> bool:
    clean_ticks(sheet.fig)
    return t8.save_checked(run, sheet, name, desc, rows, formats=(".pdf", ".png", ".tif"))


def present_labels(fr: pd.DataFrame, col: str) -> list[str]:
    """Labels with at least one cell in either condition under a label column (plate 'all')."""
    sel = fr.query("label_column == @col and plate == 'all'")
    have = set(sel.loc[sel["n_label"] > 0, "label"])
    return [l for l in t22.LABELS if l in have]


def tested(frame: pd.DataFrame, status: str = "tested") -> pd.DataFrame:
    return frame.loc[frame["status"] == status] if "status" in frame else frame


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def composition_bars(ax, rows: list[tuple[str, pd.Series, int]], legend_labels: list[str] | None = None):
    """Stacked horizontal label fractions; rows = (label, fractions by label, n cells)."""
    for i, (_, fr, _) in enumerate(rows):
        left = 0.0
        for lab in t22.LABELS:
            w = float(fr.get(lab, 0.0))
            if w <= 0:
                continue
            ax.barh(i, w, left=left, height=0.62, color=LABEL_COLORS[lab], edgecolor="white", linewidth=0.8,
                    hatch="////" if lab == "ProLIF" else None)
            left += w
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r[0]} (n = {r[2]})" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if legend_labels:
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=LABEL_COLORS[l], edgecolor="white", hatch="////" if l == "ProLIF" else None,
                         label=l) for l in legend_labels]
        ax.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
                  handlelength=1.2, handletextpad=0.4, columnspacing=0.9, borderaxespad=0)


def fractions_of(fr: pd.DataFrame, col: str, cond: str, plate: str = "all") -> tuple[pd.Series, int]:
    sel = fr.query("label_column == @col and condition == @cond and plate == @plate").set_index("label")
    return sel["fraction"], int(sel["n_cells"].iloc[0])


def effect_axis(ax, est, lo, hi, filled, n_rows, xlabel):
    y = np.arange(n_rows)
    for i in range(n_rows):
        if not np.isfinite(est[i]):
            continue
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=INK["secondary"], lw=0.6, solid_capstyle="butt")
        ax.plot(est[i], y[i], "o", ms=3.2, mfc=INK["primary"] if filled[i] else "white", mec=INK["primary"],
                mew=0.6)
    ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(xlabel, labelpad=1.5)


def symmetric_limits(ax, values, pad=1.1):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    m = float(np.max(np.abs(v))) * pad if len(v) else 1.0
    ax.set_xlim(-m, m)


def panel_effects(sheet, x0, y0, res: pd.DataFrame, items: list[str]):
    """Panel B: text columns + detection and intensity effect axes; scores below. Returns bottom y (mm)."""
    pitch = 3.4
    prot = res.loc[res["item_type"] == "protein"].set_index("item").reindex(
        [i for i in items if i in set(res.loc[res["item_type"] == "protein", "item"])])
    scores = res.loc[res["item_type"] == "score"].set_index("item").reindex(
        [i for i in items if i in set(res.loc[res["item_type"] == "score", "item"])])
    name_w = max(text_width(short_item(i)) for i in prot.index) + 1.5
    cx = {"name": x0, "exp": x0 + name_w, "det": x0 + name_w + 9, "int": x0 + name_w + 9 + 27}
    cx["obs"] = cx["int"] + 27 + 1.5
    cx["match"] = cx["obs"] + 9
    head_y = y0
    for key, label in (("name", "Protein"), ("exp", "Exp."), ("obs", "Obs."), ("match", "Match")):
        sheet.text(cx[key], head_y, label, va="top", color=INK["secondary"], fontweight="bold")
    sheet.text(cx["det"] + 12.5, head_y, "Detection", va="top", ha="center", color=INK["secondary"],
               fontweight="bold")
    sheet.text(cx["int"] + 12.5, head_y, "Intensity", va="top", ha="center", color=INK["secondary"],
               fontweight="bold")
    top = head_y + 4.5
    n = len(prot)
    h = n * pitch
    for i, (item, r) in enumerate(prot.iterrows()):
        yy = top + (i + 0.5) * pitch
        sheet.text(cx["name"], yy, short_item(item), va="center")
        sheet.text(cx["exp"], yy, EXP_SHORT[r["expected_direction"]], va="center")
        sheet.text(cx["obs"], yy, DIR_SHORT[r["observed_direction"]], va="center")
        sheet.text(cx["match"], yy, MATCH_SHORT[r["match_status"]], va="center")
    axd = sheet.ax(cx["det"], top, 25, h)
    axi = sheet.ax(cx["int"], top, 25, h)
    det = prot["det_logodds"].to_numpy(float)
    dse = prot["det_logodds_se"].to_numpy(float)
    effect_axis(axd, det, det - 1.96 * dse, det + 1.96 * dse, prot["det_p"].to_numpy(float) < lib.ALPHA, n,
                "log odds ratio")
    li = prot["int_log2fc_adjusted"].to_numpy(float)
    lse = prot["int_coef_se"].to_numpy(float) / np.log(2)
    effect_axis(axi, li, li - 1.96 * lse, li + 1.96 * lse, prot["int_p"].to_numpy(float) < lib.ALPHA, n,
                "log2 fold change")
    symmetric_limits(axd, np.r_[det - 1.96 * dse, det + 1.96 * dse])
    symmetric_limits(axi, np.r_[li - 1.96 * lse, li + 1.96 * lse])
    for ax in (axd, axi):
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    # scores
    sy = top + h + 12
    sheet.text(cx["name"], sy, "Score", va="top", color=INK["secondary"], fontweight="bold")
    sheet.text(cx["det"] + 25 + 1, sy, "Score difference (OLS, depth-adjusted)", va="top", ha="center",
               color=INK["secondary"], fontweight="bold")
    stop = sy + 4.5
    ns = len(scores)
    for i, (item, r) in enumerate(scores.iterrows()):
        yy = stop + (i + 0.5) * pitch
        sheet.text(cx["name"], yy, short_item(item), va="center")
        sheet.text(cx["obs"], yy, DIR_SHORT[r["observed_direction"]], va="center")
        sheet.text(cx["match"], yy, MATCH_SHORT[r["match_status"]], va="center")
    sw = max(text_width(short_item(i)) for i in scores.index) + 1.5
    sx0 = max(cx["det"], x0 + sw)
    axs = sheet.ax(sx0, stop, cx["int"] + 25 - sx0, ns * pitch)
    sc_ = scores["score_coef"].to_numpy(float)
    sse = scores["score_coef_se"].to_numpy(float)
    effect_axis(axs, sc_, sc_ - 1.96 * sse, sc_ + 1.96 * sse, scores["p_score_lm"].to_numpy(float) < lib.ALPHA,
                ns, "difference, ZeroK − Ctrl (score units)")
    symmetric_limits(axs, np.r_[sc_ - 1.96 * sse, sc_ + 1.96 * sse])
    style(axs)
    axs.spines["left"].set_visible(False)
    axs.tick_params(axis="y", length=0)
    return stop + ns * pitch + 8


def volcano(ax, frame: pd.DataFrame, label: tuple[str, ...] = (), cap: float = 60.0, marker_panel: bool = True,
            label_side: str = "right"):
    f = frame.copy()
    f = f.loc[np.isfinite(f["int_log2fc_adjusted"])]
    y = -np.log10(np.clip(f["p_two_part"].to_numpy(float), 1e-300, 1))
    capped = y > cap
    y = np.minimum(y, cap)
    x = f["int_log2fc_adjusted"].to_numpy(float)
    d = f["observed_direction"].to_numpy()
    for key in (t22.OBS_NS, t22.OBS_MIXED, t22.DOWN, t22.UP):
        m = d == key
        ax.scatter(x[m & ~capped], y[m & ~capped], s=2.5 if key == t22.OBS_NS else 3.5, c=DIR_COLORS[key],
                   linewidths=0, rasterized=True, zorder=1 if key == t22.OBS_NS else 2)
        ax.scatter(x[m & capped], y[m & capped], s=5, c=DIR_COLORS[key], marker="^", linewidths=0,
                   rasterized=True, zorder=2)
    lim = float(np.nanquantile(np.abs(x), 0.998)) * 1.1 if len(x) else 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, cap * 1.04)
    ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    if marker_panel:
        pm = f["in_panel"].astype(str).str.lower().eq("true").to_numpy()
        ax.scatter(np.clip(x[pm], -lim, lim), y[pm], s=14, facecolors="none", edgecolors=INK["primary"],
                   linewidths=0.6, zorder=3)
    if label:
        sel = [(g, xx, yy) for g, xx, yy in zip(f["protein"], x, y) if g in label]
        sel.sort(key=lambda t: t[2])
        gap = 3.6 / (ax.get_position().height * ax.figure.get_figheight() * 25.4)
        yf = [min(max(yy / (cap * 1.04), 0.03), 0.97) for _, _, yy in sel]
        for i in range(1, len(yf)):
            yf[i] = max(yf[i], yf[i - 1] + gap)
        over = yf[-1] - 0.98 if yf else 0
        if over > 0:
            yf = [v - over for v in yf]
            for i in range(len(yf) - 2, -1, -1):
                yf[i] = min(yf[i], yf[i + 1] - gap)
        tx = 1.04 if label_side == "right" else -0.04
        for (g, xx, yy), v in zip(sel, yf):
            # leader line as an empty annotation (an Annotation's extent includes its arrow), name as plain text
            ax.annotate("", xy=(np.clip(xx, -lim, lim), yy), xycoords="data", xytext=(tx - 0.005, v),
                        textcoords="axes fraction", arrowprops=dict(arrowstyle="-", lw=0.4, color=INK["muted"],
                                                                    shrinkA=0, shrinkB=2))
            ax.text(tx, v, g, transform=ax.transAxes, va="center",
                    ha="left" if label_side == "right" else "right", fontsize=FONT)
    style(ax)
    return int((~np.isfinite(frame["int_log2fc_adjusted"])).sum()), int(capped.sum())


def direction_legend(ax, loc="upper left", anchor=(0, 1.0), ncol=4, panel=True):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=3.5, mfc=DIR_COLORS[k], mec="none", label=DIR_SHORT[k])
         for k in (t22.UP, t22.DOWN, t22.OBS_MIXED, t22.OBS_NS)]
    if panel:
        h.append(Line2D([], [], ls="", marker="o", ms=4.5, mfc="none", mec=INK["primary"], mew=0.6, label="panel"))
    ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=ncol, frameon=False, handletextpad=0.2,
              columnspacing=0.7, borderaxespad=0)


def boxes(ax, values: dict[tuple[str, str], np.ndarray], labels: list[str], ylabel: str, rng):
    for i, lab in enumerate(labels):
        for j, cond in enumerate(t22.CONDITIONS):
            v = values.get((lab, cond), np.array([]))
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            pos = i + (-0.19 if j == 0 else 0.19)
            bp = ax.boxplot([v], positions=[pos], widths=0.3, showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK["primary"], lw=0.8),
                            whiskerprops=dict(color=INK["secondary"], lw=0.5),
                            capprops=dict(color=INK["secondary"], lw=0.5),
                            boxprops=dict(edgecolor=INK["secondary"], lw=0.5))
            bp["boxes"][0].set_facecolor(COND[cond])
            bp["boxes"][0].set_alpha(0.55)
            jit = pos + rng.uniform(-0.1, 0.1, len(v))
            ax.scatter(jit, v, s=0.8, c=COND[cond], linewidths=0, alpha=0.6, rasterized=True, zorder=0)
    ax.set_xticks(range(len(labels)))
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel)
    style(ax)


def condition_legend(ax, loc="lower left", anchor=(0, 1.02), extra=()):
    from matplotlib.patches import Patch
    h = [Patch(facecolor=COND[c], alpha=0.75, label=c) for c in t22.CONDITIONS] + list(extra)
    ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=len(h), frameon=False, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.9, borderaxespad=0)


def does_not_apply(ax, title: str, n_ctrl: int, n_zerok: int) -> None:
    """Empty panel slot for a group whose label has no cells in a condition."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=FONT, pad=2)
    ax.text(0.5, 0.5, f"does not apply\nCtrl {n_ctrl}, ZeroK {n_zerok} cells", transform=ax.transAxes,
            ha="center", va="center", fontsize=FONT, color=INK["secondary"])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_main(run, T, rows, items):
    fr, res, prot, per = T["fractions"], tested(T["panel"]), T["proteome_primary_none"], T["state_per_cell"]
    note_text = (f"{DESIGN_TEXT} (A) Labels (dct_labels): Ctrl cs_ctrl_deg_label; ZeroK pipeline = cs_zerok_pipeline_label "
                 "(primary), transfer = cs_zerok_transfer_label (alternative). (B-C) all_DCT = all DCT labels under "
                 "the primary labels (Ctrl DCT1 + DCT2, ZeroK DCT1 only: the primary labels have no ZeroK DCT2 and "
                 "no ProLIF), normalization variant none (baseline). (B) Two-part model: Firth logistic on "
                 "detection and OLS on ln-intensity among detected cells, both with log proteins per cell; dots "
                 "with 95% Wald intervals, filled if the part has p < 0.05. Exp. = frozen zerok_core direction (n.c. = no "
                 "change, – = no expectation); Obs. = observed direction (BH q < 0.05 within the panel; n.s. = no "
                 "BH-significant difference); * = detected in < 10% of cells in a condition. (C) x = intensity "
                 "part, y = two-part LR test; colour = observed direction; rings = panel proteins; triangles = "
                 "capped. (D) DCT1 − DCT2 score (DCT V1 workbook, score_genes as labels_core), variant none; n = Ctrl / "
                 "ZeroK. Other variants, the alternative labels and robustness: Supplementary.")
    note, nh = t8.note_block(note_text)
    H = 166 + nh
    sheet = t8.Sheet(H)
    # A: composition (left, top)
    letter(sheet, 3, 3, "A")
    comp = [("Ctrl",) + fractions_of(fr, "dct_label", "Ctrl"),
            ("ZeroK, pipeline",) + fractions_of(fr, "dct_label", "ZeroK"),
            ("ZeroK, transfer",) + fractions_of(fr, "dct_label_alternative", "ZeroK")]
    lab_w = max(text_width(f"{r[0]} (n = {r[2]})") for r in comp) + 2
    axa = sheet.ax(6 + lab_w, 13, 98 - 6 - lab_w, 14)
    composition_bars(axa, comp, legend_labels=present_labels(fr, "dct_label_alternative"))
    style(axa)
    axa.spines["left"].set_visible(False)
    axa.tick_params(axis="y", length=0)
    # B: panel (left, below A)
    letter(sheet, 3, 36, "B")
    sel = res.query("label_column == 'dct_label' and group == 'all_DCT' and variant == 'none'")
    panel_effects(sheet, 6, 37, sel, items)
    # C: volcano (right, top)
    xr = 110
    letter(sheet, xr, 3, "C")
    axc = sheet.ax(xr + 12, 15, 40, 62)
    nx, ncap = volcano(axc, prot.query("group == 'all_DCT'"), label=EXPECTED_PROTEINS)
    axc.set_xlabel("log2 fold change, intensity part", labelpad=1.5)
    axc.set_ylabel("−log10 p, two-part", labelpad=1.5)
    direction_legend(axc, loc="lower left", anchor=(-0.25, 1.03), ncol=5)
    # D: state (right, bottom)
    letter(sheet, xr, 92, "D")
    axd = sheet.ax(xr + 12, 104, 177 - xr - 12, 49)
    labs = present_labels(fr, "dct_label")
    vals = {(l, c): per.loc[(per["dct_label"] == l) & (per["condition"] == c),
                            "DCT1_minus_DCT2_score__none"].to_numpy(float) for l in labs for c in t22.CONDITIONS}
    boxes(axd, vals, labs, "DCT1 − DCT2 score", t22.repeat_rng(60))
    counts = [f"{l}\n{len(vals[(l, 'Ctrl')])} / {len(vals[(l, 'ZeroK')])}" for l in labs]
    axd.set_xticklabels(counts)
    condition_legend(axd)
    sheet.note(note, nh)
    rows.append({"figure": "main_candidate", "proteins_without_intensity_part_not_shown": nx,
                 "points_capped": ncap})
    return save(run, sheet, "main_candidate",
                "zerok_vs_control main-text candidate: composition, panel, volcano, DCT1-DCT2 score", T["checks"])


def fig_composition(run, T):
    fr, diff = T["fractions"], T["fraction_diff"]
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} Top: label fractions per condition and sort plate (P1/P2 are technical sort plates of one "
        "pool). Bottom: difference in label fraction, ZeroK − Ctrl, with Newcombe 95% intervals (cells as units; "
        "descriptive), pooled and within each plate. Left: primary labels (ZeroK cs_zerok_pipeline_label); right: "
        "alternative (ZeroK cs_zerok_transfer_label); control labels identical (cs_ctrl_deg_label, dct_labels). ProLIF "
        "has no cells under either label column and is not drawn.")
    H = 100 + nh
    sheet = t8.Sheet(H)
    for k, col in enumerate(lib.LABEL_COLUMNS):
        labs = present_labels(fr, col)
        x0 = 3 + k * 90
        letter(sheet, x0, 3, "AB"[k])
        sheet.text(x0 + 5, 3.4, lib.LABEL_SHORT[col] + " labels", va="top", fontweight="bold")
        comp = [(f"{c} {p}",) + fractions_of(fr, col, c, p) for c in t22.CONDITIONS for p in ("P1", "P2", "all")]
        ax = sheet.ax(x0 + 26, 15, 55, 30)
        composition_bars(ax, comp, legend_labels=labs if k == 0 else None)
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        axb = sheet.ax(x0 + 26, 58, 55, 32)
        d = diff.query("label_column == @col")
        for j, (scope, mk, colr) in enumerate((("all", "o", INK["primary"]), ("P1", "^", PLATE["P1"]),
                                                ("P2", "s", PLATE["P2"]))):
            s = d.query("plate == @scope").set_index("label").reindex(labs)
            yy = np.arange(len(labs)) + (j - 1) * 0.22
            axb.errorbar(s["difference_zerok_minus_ctrl"], yy,
                         xerr=[s["difference_zerok_minus_ctrl"] - s["difference_ci_low"],
                               s["difference_ci_high"] - s["difference_zerok_minus_ctrl"]],
                         fmt=mk, ms=3, color=colr, mfc=colr, mec="none", elinewidth=0.6, capsize=0,
                         label="pooled" if scope == "all" else scope)
        axb.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
        axb.set_yticks(range(len(labs)))
        axb.set_yticklabels(labs)
        axb.set_ylim(len(labs) - 0.5, -0.5)
        axb.set_xlabel("fraction difference, ZeroK − Ctrl", labelpad=1.5)
        style(axb)
        if k == 0:
            axb.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, handletextpad=0.2,
                       columnspacing=0.8, borderaxespad=0)
    sheet.note(note, nh)
    return save(run, sheet, "supp_composition", "zerok_vs_control label composition by condition and plate", T["checks"])


def fig_contamination(run, T):
    det, cov = T["contamination"], T["contamination_coverage"]
    groups = [lib.ALL_CELLS, lib.ALL_DCT, "DCT1", "Unknown"]
    d = det.query("label_column == 'dct_label' and status == 'compared'")
    genes = [g for g in dict.fromkeys(d["gene"]) if not (g.startswith("any of") and "/" not in g)]
    names = {g: g if not g.startswith("any of") else f"any {d.loc[d['gene'] == g, 'panel_group'].iloc[0]}"
             for g in genes}
    groups_of = d.drop_duplicates("gene").set_index("gene")["panel_group"].to_dict()
    names = {g: (f"{n} ({groups_of[g]})" if not g.startswith("any of") else n) for g, n in names.items()}
    not_cmp = [g for g in dict.fromkeys(cov.loc[~cov["compared_between_conditions"].astype(str).str.lower()
                                                 .eq("true"), "gene"])]
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} Detection fraction (protein detected in the cell) per condition within each group, primary "
        "labels (all_DCT: Ctrl DCT1 + DCT2, ZeroK DCT1). Genes: intake_core contamination list (TAL, CNT/PC, IC, PT, "
        "endothelial, immune) and the Task 3 negative panels; 'any' rows = any gene of a multi-gene group. Not "
        "in the joint protein set of the combined search and therefore not compared: " + ", ".join(not_cmp)
        + " (zerok_vs_control_contamination_coverage.csv). Primary DCT2 (no ZeroK cells) does not apply. Calb1 is also "
        "a DCT2 marker and a panel protein; Umod is TAL-associated.")
    pitch = 3.4
    H = 14 + len(genes) * pitch + 12 + nh
    sheet = t8.Sheet(H)
    name_w = max(text_width(n) for n in names.values()) + 2
    w = (180 - 3 - 3 - name_w - 3 * 3) / 4
    for k, grp in enumerate(groups):
        x0 = 3 + name_w + k * (w + 3)
        ax = sheet.ax(x0, 12, w, len(genes) * pitch)
        s = d.query("group == @grp").set_index("gene").reindex(genes)
        y = np.arange(len(genes))
        ax.hlines(y, s["fraction_ctrl"], s["fraction_zerok"], color=INK["muted"], lw=0.6)
        ax.plot(s["fraction_ctrl"], y, "o", ms=3, color=COND["Ctrl"], label="Ctrl")
        ax.plot(s["fraction_zerok"], y, "o", ms=3, color=COND["ZeroK"], label="ZeroK")
        ax.set_ylim(len(genes) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks(y)
        ax.set_yticklabels([names[g] for g in genes] if k == 0 else [])
        n_c, n_z = int(s["n_ctrl"].dropna().iloc[0]), int(s["n_zerok"].dropna().iloc[0])
        ax.set_title(f"{grp} ({n_c} / {n_z})", fontsize=FONT, pad=2)
        ax.set_xlabel("detection fraction", labelpad=1.5)
        style(ax)
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.07), ncol=2, frameon=False, handletextpad=0.2,
                      borderaxespad=0)
    sheet.note(note, nh)
    return save(run, sheet, "supp_contamination", "zerok_vs_control contamination marker detection by condition", T["checks"])


def fig_panel_overview(run, T, items):
    res = tested(T["panel"])
    cols = []
    for c in lib.LABEL_COLUMNS:
        have = set(res.loc[res["label_column"] == c, "group"])
        cols.append((c, [g for g in [*t22.LABELS, lib.ALL_DCT] if g in have]))
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} Observed direction of every pre-specified item (colour) under the D1 rule: BH q < 0.05 "
        "within the panel per label column × group × variant (two-part model for proteins, depth-adjusted OLS for "
        "scores). Dot = match, cross = mismatch with the frozen expected direction (Exp.; n.c. = no change, – = no "
        "expectation, not scored); white = not testable (< 5 detected cells). Variants: a = none (baseline), b = "
        "cell_median, c = condition_align. * = detected in < 10% of cells in a condition. " + ABSENT_TEXT
        + " They are not drawn.")
    cw, ch = 3.9, 3.3
    name_w = max(text_width(short_item(i)) for i in items) + 2
    x0 = 3 + name_w + 9
    H = 24 + len(items) * ch + 14 + nh
    sheet = t8.Sheet(H)
    top = 22
    for i, item in enumerate(items):
        yy = top + (i + 0.5) * ch
        sheet.text(3, yy, short_item(item), va="center")
        sheet.text(3 + name_w, yy, EXP_SHORT[res.loc[res["item"] == item, "expected_direction"].iloc[0]],
                   va="center")
    sheet.text(3 + name_w, top - 1, "Exp.", va="bottom", color=INK["secondary"], fontweight="bold")
    idx = res.set_index(["label_column", "group", "variant", "item"])
    x = x0
    for c, groups in cols:
        sheet.text(x, top - 10, f"{lib.LABEL_SHORT[c]} labels", va="bottom", fontweight="bold")
        for g in groups:
            for v in lib.VARIANTS:
                for i, item in enumerate(items):
                    r = idx.loc[(c, g, v, item)]
                    sheet.rect(x, top + i * ch, cw - 0.3, ch - 0.3, facecolor=DIR_COLORS[r["observed_direction"]],
                               edgecolor=INK["grid"], linewidth=0.3)
                    cxm = (x + (cw - 0.3) / 2) / 180
                    cym = 1 - (top + i * ch + (ch - 0.3) / 2) / H
                    if r["match_status"] == "match":
                        sheet.fig.add_artist(plt.Line2D([cxm], [cym], marker="o", ms=1.8, color=INK["primary"],
                                                        transform=sheet.fig.transFigure))
                    elif r["match_status"] == "mismatch":
                        sheet.fig.add_artist(plt.Line2D([cxm], [cym], marker="x", ms=2.6, mew=0.7,
                                                        color=INK["primary"], transform=sheet.fig.transFigure))
                sheet.text(x + (cw - 0.3) / 2, top - 1, VARIANT_LETTER[v], ha="center", va="bottom")
                if v == "cell_median":
                    sheet.text(x + (cw - 0.3) / 2, top - 5, g if g != lib.ALL_DCT else "all", ha="center",
                               va="bottom")
                x += cw
            x += 0.8
        x += 3
    # legend
    ly = top + len(items) * ch + 4
    lx = 3
    for key in (t22.UP, t22.DOWN, t22.OBS_MIXED, t22.OBS_NS, t22.NOT_TESTABLE):
        sheet.rect(lx, ly, 3, 2.6, facecolor=DIR_COLORS[key], edgecolor=INK["grid"], linewidth=0.3)
        sheet.text(lx + 4, ly + 1.3, {t22.OBS_NS: "no BH-significant difference"}.get(key, key), va="center")
        lx += 4 + text_width({t22.OBS_NS: "no BH-significant difference"}.get(key, key)) + 4
    sheet.note(note, nh)
    return save(run, sheet, "supp_panel_overview", "zerok_vs_control panel: observed direction for every group, variant and "
                "label column", T["checks"])


def fig_volcanoes(run, T):
    groups = ["DCT1", "DCT2", "Unknown", lib.ALL_DCT]
    pg = T["proteome_groups"].set_index(["label_column", "group"])
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} Volcano plots, normalization variant none (baseline). x = depth-adjusted intensity part "
        "(log2 fold change among detected cells); y = two-part LR test (detection + intensity); colour = observed "
        "direction (D1 rule; BH within group); rings = panel proteins; triangles = capped at 60. Proteins without "
        "an intensity part (< 5 detected cells in a condition) are not drawn. Rows: primary and alternative ZeroK "
        "labels; n = Ctrl / ZeroK. " + ABSENT_TEXT)
    H = 112 + nh
    sheet = t8.Sheet(H)
    w = 34
    for r, col in enumerate(lib.LABEL_COLUMNS):
        tab = T[f"proteome_{lib.LABEL_SHORT[col]}_none"]
        for k, g in enumerate(groups):
            ax = sheet.ax(14 + k * (w + 8), 12 + r * 50, w, 36)
            f = tab.query("group == @g")
            title_pre = f"{lib.LABEL_SHORT[col][:3]}.: {g}"
            if f.empty:
                n_c, n_z = int(pg.loc[(col, g), "n_ctrl"]), int(pg.loc[(col, g), "n_zerok"])
                does_not_apply(ax, title_pre, n_c, n_z)
                continue
            volcano(ax, f)
            nc, nz = int(f["n_ctrl"].iloc[0]), int(f["n_zerok"].iloc[0])
            ax.set_title(f"{title_pre} ({nc}/{nz})", fontsize=FONT, pad=2)
            if k == 0:
                ax.set_ylabel("−log10 p", labelpad=1.5)
            if r == 1:
                ax.set_xlabel("log2 FC, intensity part", labelpad=1.5)
            if r == 0 and k == 0:
                direction_legend(ax, loc="lower left", anchor=(0, 1.13), ncol=5)
    sheet.note(note, nh)
    return save(run, sheet, "supp_volcanoes", "zerok_vs_control volcano plots per label (variant none)", T["checks"])


def fig_volcano_variants(run, T):
    panels = [("primary", "cell_median", lib.ALL_DCT), ("primary", "condition_align", lib.ALL_DCT),
              ("alternative", "none", lib.ALL_DCT), ("label_free", "none", lib.ALL_CELLS)]
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} As the main-text volcano, for the other two normalization variants (primary labels, "
        "all_DCT), the alternative ZeroK labels (variant none) and the label-free version (all cells, variant "
        "none). Rings = panel proteins; triangles = capped at 60.")
    H = 66 + nh
    sheet = t8.Sheet(H)
    w = 34
    for k, (col, v, g) in enumerate(panels):
        tab = T["label_free_proteome"].query("variant == @v") if col == "label_free" else T[f"proteome_{col}_{v}"]
        ax = sheet.ax(14 + k * (w + 8), 18, w, 36)
        f = tab.query("group == @g")
        volcano(ax, f)
        ax.set_title(f"{col[:3] + '.' if col != 'label_free' else 'label-free'}, {v}\n{g}", fontsize=FONT, pad=2)
        if k == 0:
            ax.set_ylabel("−log10 p", labelpad=1.5)
            direction_legend(ax, loc="lower left", anchor=(0, 1.13), ncol=5)
        ax.set_xlabel("log2 FC, intensity part", labelpad=1.5)
    sheet.note(note, nh)
    return save(run, sheet, "supp_volcano_variants", "zerok_vs_control volcano plots: variants, alternative labels, label-free",
                T["checks"])


def fig_state(run, T):
    per, cyc, dm, fr = T["state_per_cell"], T["cycling"], T["depth_matched"], T["fractions"]
    ratio = T["state_checks"].set_index("check").loc["none: median neighbour distance ZeroK / control", "observed"]
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} (A) DCT1 − DCT2 score per cell: DCT V1 workbook score_genes as labels_core (left) and the "
        "deterministic mean-z check (right), primary labels, variant none; n = Ctrl / ZeroK. (B) Cycling fractions "
        "with Wilson 95% intervals: Mki67 or Top2a detected, and the cycling-score rule (cell-cycle score above "
        "the 90th percentile of control cells, variant none). ProLIF label share: does not apply (no ProLIF cells "
        "under either label column). (C) Projected pseudotime: mean dct_labels control DPT of the 10 nearest control "
        "cells in the joint 20-PC embedding (control cells leave themselves out), variant none; ZeroK cells lie "
        f"{ratio} times as far from their control neighbours as control cells do (median). Primary DCT2 has no "
        "ZeroK cells. (D) Median difference ZeroK − Ctrl: full data (open) and depth-matched (dot: median of 100 "
        "repeats; line: 2.5–97.5%), primary labels, variant none.")
    H = 140 + nh
    sheet = t8.Sheet(H)
    rng = t22.repeat_rng(61)
    labs = present_labels(fr, "dct_label")
    for k, meas in enumerate(("DCT1_minus_DCT2_score", "DCT1_minus_DCT2_meanz")):
        ax = sheet.ax(14 + k * 88, 12, 76, 36)
        vals = {(l, c): per.loc[(per["dct_label"] == l) & (per["condition"] == c), f"{meas}__none"].to_numpy(float)
                for l in labs for c in t22.CONDITIONS}
        boxes(ax, vals, labs, "DCT1 − DCT2 " + ("score" if k == 0 else "mean z"), rng)
        ax.set_xticklabels([f"{l}\n{len(vals[(l, 'Ctrl')])} / {len(vals[(l, 'ZeroK')])}" for l in labs])
        if k == 0:
            letter(sheet, 3, 3, "A")
            condition_legend(ax)
    # B: cycling
    letter(sheet, 3, 60, "B")
    meas = [("Mki67_or_Top2a_detected", "Mki67 or Top2a detected"), ("cycling_score_rule__none", "cycling-score rule")]
    cyc = tested(cyc, "compared")
    for k, (col, grp) in enumerate((("dct_label", lib.ALL_CELLS), ("dct_label", lib.ALL_DCT),
                                     ("dct_label_alternative", lib.ALL_DCT))):
        ax = sheet.ax(40 + k * 47, 70, 38, 13)
        rows = []
        for m, name in meas:
            sel = cyc.query("label_column == @col and group == @grp and measure == @m")
            if sel.empty and col == "dct_label" and grp == lib.ALL_CELLS:
                sel = cyc.query("label_column == 'label_free' and group == @grp and measure == @m")
            rows.append((name, sel.iloc[0] if len(sel) else None))
        y = np.arange(len(rows))
        for i, (name, r) in enumerate(rows):
            if r is None:
                continue
            for cond, off in (("Ctrl", -0.15), ("ZeroK", 0.15)):
                k_ = cond.lower()
                ax.errorbar(r[f"fraction_{k_}"], i + off, xerr=[[r[f"fraction_{k_}"] - r[f"ci_low_{k_}"]],
                                                                 [r[f"ci_high_{k_}"] - r[f"fraction_{k_}"]]],
                            fmt="o", ms=3, color=COND[cond], elinewidth=0.6, capsize=0)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in rows] if k == 0 else [])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(0, None)
        ax.set_title(f"{lib.LABEL_SHORT[col] if grp != lib.ALL_CELLS else 'all cells'}"
                     f"{': ' + grp if grp != lib.ALL_CELLS else ''}", fontsize=FONT, pad=2)
        ax.set_xlabel("fraction of cells", labelpad=1.5)
        style(ax)
    # C: pseudotime ECDF
    letter(sheet, 3, 96, "C")
    for k, g in enumerate(["DCT1", "DCT2", "Unknown", lib.ALL_DCT]):
        ax = sheet.ax(14 + k * 26, 104, 20, 22)
        m = per["dct_label"].isin(t22.CORE) if g == lib.ALL_DCT else per["dct_label"] == g
        n_z = 0
        for cond in t22.CONDITIONS:
            v = np.sort(per.loc[m & (per["condition"] == cond), "projected_dpt__none"].to_numpy(float))
            if not len(v):
                continue
            n_z += len(v) if cond == "ZeroK" else 0
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COND[cond], lw=0.9)
        title = g if g != lib.ALL_DCT else "all_DCT"
        ax.set_title(title if n_z else f"{title} (Ctrl only)", fontsize=FONT, pad=2)
        ax.set_xlabel("projected DPT", labelpad=1.5)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        if k == 0:
            ax.set_ylabel("cumulative fraction", labelpad=1.5)
        else:
            ax.set_yticklabels([])
        style(ax)
    # D: depth-matched
    letter(sheet, 118, 96, "D")
    ax = sheet.ax(140, 104, 37, 26)
    sel = tested(dm, "compared").query("label_column == 'dct_label' and variant == 'none' and "
                                       "measure in ['DCT1_minus_DCT2_score', 'projected_dpt']")
    rows = [(r.measure, r.group) for r in sel.itertuples()]
    y = np.arange(len(rows))
    for i, r in enumerate(sel.itertuples()):
        ax.plot([r.effect_q025, r.effect_q975], [i, i], color=INK["secondary"], lw=0.6)
        ax.plot(r.effect_median, i, "o", ms=3, color=INK["primary"])
        ax.plot(r.effect_full, i, "o", ms=3.4, mfc="white", mec=INK["primary"], mew=0.6)
    ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'score' if m.startswith('DCT1') else 'DPT'}: {g}" for m, g in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("median difference", labelpad=1.5)
    style(ax)
    sheet.note(note, nh)
    return save(run, sheet, "supp_state", "zerok_vs_control cell-state shifts", T["checks"])


def fig_robustness(run, T, items):
    rep, eff, lso = T["panel_repeats"], T["proteome_effects"], T["leave_slc"]
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} (A) Share of 100 repeats in which a panel item keeps its full-data sign (part with the "
        "larger chi2) and BH q < 0.05: depth-matched (1:1 on ln proteins per cell within plate, caliper 0.05), "
        "50% and 25% downsampling; primary labels, variant none; F = q < 0.05 in all cells of the group (black). "
        "Primary DCT2 (no ZeroK cells) does not apply; Unknown shown instead. (B) Pseudobulk log2 fold change "
        "from P1 cells only vs P2 cells only (sort plates of one pool: a technical check), all_DCT, primary, "
        "variant none. (C) Spearman correlation of proteome-wide effects (log2 FC among detected cells) between "
        "each repeat and the full data, median of 100 repeats, variant none. (D) ALDO and DCT1 − DCT2 score "
        "differences with and without Slc12a3 (all groups and variants).")
    H = 150 + nh
    sheet = t8.Sheet(H)
    # A heatmap
    letter(sheet, 3, 3, "A")
    groups = [lib.ALL_DCT, "DCT1", "Unknown"]
    procs = ["depth_matched", "downsample_50pct", "downsample_25pct"]
    name_w = max(text_width(short_item(i)) for i in items) + 2
    cw, ch = 4.2, 3.2
    top = 20
    cmap = t8.sequential_cmap()
    r0 = rep.query("label_column == 'dct_label' and variant == 'none'").set_index(["procedure", "group", "item"])
    full = tested(T["panel"]).query("label_column == 'dct_label' and variant == 'none'").set_index(["group", "item"])
    for i, item in enumerate(items):
        sheet.text(3, top + (i + 0.5) * ch, short_item(item), va="center")
    x = 3 + name_w
    for g in groups:
        sheet.text(x, top - 7.5, g if g != lib.ALL_DCT else "all_DCT", va="bottom")
        for j, p in enumerate(["full", *procs]):
            for i, item in enumerate(items):
                if p == "full":
                    q = full.loc[(g, item), "q_primary"]
                    col = INK["primary"] if np.isfinite(q) and q < lib.ALPHA else "#ffffff"
                else:
                    v = r0.loc[(p, g, item), "fraction_same_sign_and_q_below_0.05"]
                    col = cmap(float(v))
                sheet.rect(x + j * cw, top + i * ch, cw - 0.3, ch - 0.3, facecolor=col, edgecolor=INK["grid"],
                           linewidth=0.3)
            sheet.text(x + j * cw + (cw - 0.3) / 2, top - 1, ["F", "D", "50", "25"][j], ha="center", va="bottom",
                       fontsize=FONT)
        x += 4 * cw + 3
    # colour bar
    cb = sheet.ax(x + 2, top, 2.5, 30)
    grad = np.linspace(0, 1, 100)[:, None]
    cb.imshow(grad[::-1], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    cb.set_xticks([])
    cb.yaxis.tick_right()
    cb.set_yticks([0, 0.5, 1])
    style(cb)
    for s in cb.spines.values():
        s.set_visible(False)
    sheet.text(x + 2, top - 1, "share", va="bottom")
    # B plate split scatter
    xb = 118
    letter(sheet, xb - 6, 3, "B")
    ax = sheet.ax(xb + 8, 12, 30, 30)
    tab = T["proteome_primary_none"].query("group == 'all_DCT'")
    a, b = tab["pseudobulk_log2fc_P1"].to_numpy(float), tab["pseudobulk_log2fc_P2"].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    ax.scatter(a[ok], b[ok], s=1.2, c=INK["muted"], linewidths=0, rasterized=True)
    lim = float(np.nanquantile(np.abs(np.r_[a[ok], b[ok]]), 0.995))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], color=INK["grid"], lw=0.6, zorder=0)
    from scipy import stats as st
    rho = st.spearmanr(a[ok], b[ok]).statistic
    ax.text(0.03, 0.97, f"ρ = {rho:.2f}\nn = {int(ok.sum())}", transform=ax.transAxes, va="top", fontsize=FONT)
    ax.set_xlabel("log2 FC, P1 cells", labelpad=1.5)
    ax.set_ylabel("log2 FC, P2 cells", labelpad=1.5)
    style(ax)
    # C proteome effect spearman
    letter(sheet, xb - 6, 52, "C")
    ax = sheet.ax(xb + 18, 60, 40, 30)
    e = eff.query("variant == 'none' and effect == 'log2fc_detected'")
    rows = [(c, g) for c, g in e[["label_column", "group"]].drop_duplicates().itertuples(index=False)]
    marks = {"depth_matched": ("o", INK["primary"]), "downsample_50pct": ("^", t2.SERIES[0]),
             "downsample_25pct": ("s", t2.SERIES[1])}
    for p, (mk, colr) in marks.items():
        s = e.query("procedure == @p").set_index(["label_column", "group"]).reindex(rows)
        ax.plot(s["spearman_vs_full_median"], np.arange(len(rows)), mk, ms=3, color=colr, ls="",
                label={"depth_matched": "depth-matched", "downsample_50pct": "50%", "downsample_25pct": "25%"}[p])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{lib.LABEL_SHORT[c][:3] if c != 'label_free' else 'free'}: {g}" for c, g in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spearman ρ with full data", labelpad=1.5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.45, 1.02), ncol=3, frameon=False, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0)
    style(ax)
    # D leave-Slc12a3-out
    letter(sheet, xb - 6, 100, "D")
    ax = sheet.ax(xb + 4, 108, 26, 26)
    for fam, mk, colr in (("ALDO_activity", "o", t2.SERIES[0]), ("ALDO_signed", "^", t2.SERIES[1]),
                          ("DCT1_minus_DCT2_score", "s", t2.SERIES[4]), ("DCT1_minus_DCT2_meanz", "D", t2.SERIES[6])):
        s = lso.query("score == @fam")
        ax.plot(s["coef_base"], s["coef_no_Slc12a3"], mk, ms=2.4, color=colr, ls="", alpha=0.8,
                label={"ALDO_activity": "ALDO activity", "ALDO_signed": "ALDO signed",
                       "DCT1_minus_DCT2_score": "DCT1−DCT2 score", "DCT1_minus_DCT2_meanz": "DCT1−DCT2 mean z"}[fam])
    v = np.r_[lso["coef_base"], lso["coef_no_Slc12a3"]]
    lim = float(np.nanmax(np.abs(v))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=INK["grid"], lw=0.6, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("difference, with Slc12a3", labelpad=1.5)
    ax.set_ylabel("without Slc12a3", labelpad=1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0), ncol=1, frameon=False, handletextpad=0.2,
              borderaxespad=0)
    style(ax)
    sheet.note(note, nh)
    return save(run, sheet, "supp_robustness", "zerok_vs_control robustness", T["checks"])


def fig_specificity(run, T):
    nulls, ranks, slots, plate = T["spec_null"], T["spec_ranks"], T["spec_slots"], T["spec_plate"]
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} Addendum 5b. (A) The 7 directional panel proteins (up: Slc12a3, Wnk1, Wnk4; down: Klhl3, "
        "Pvalb, Calb1, Slc8a1) against 10,000 sets of random non-panel proteins, each drawn from the 50 proteins "
        "nearest to the panel protein in detection fraction and ln-intensity; variant none. Left: number of "
        "proteins whose observed direction (BH across the group's proteome) equals the expected one; right: mean "
        "of expected sign × pseudobulk log2 FC. Line = 2.5–97.5% of the null; dot = observed; p = empirical "
        "one-sided. (B) Directional percentile of each panel protein (1 = furthest in the expected direction) "
        "among all tested proteins (filled) and among its 50 matched proteins (open); all_DCT, primary, variant "
        "none. (C) Share of proteins with q < 0.05 (two-part model): P2 vs P1 within one condition (technical "
        "null) and ZeroK vs Ctrl within one plate; variant none. " + ABSENT_TEXT)
    H = 118 + nh
    sheet = t8.Sheet(H)
    rows = [(c, g) for c, g in nulls.query("variant == 'none'")[["label_column", "group"]]
            .drop_duplicates().itertuples(index=False)]
    lab = [f"{lib.LABEL_SHORT[c][:3] if c != 'label_free' else 'free'}: {g}" for c, g in rows]
    letter(sheet, 3, 3, "A")
    for k, stat in enumerate(("S1 matches", "S2 signed pseudobulk log2FC")):
        ax = sheet.ax(26 + k * 47, 13, 34, 46)
        s = nulls.query("variant == 'none' and statistic == @stat").set_index(["label_column", "group"]).reindex(rows)
        y = np.arange(len(rows))
        ax.hlines(y, s["null_q025"], s["null_q975"], color=INK["muted"], lw=1.6)
        ax.plot(s["null_mean"], y, "|", ms=5, color=INK["muted"])
        ax.plot(s["observed"], y, "o", ms=3.4, color=INK["primary"])
        for i, p in enumerate(s["empirical_p_greater_or_equal"]):
            ax.text(1.02, i, f"{p:.2g}", transform=ax.get_yaxis_transform(), va="center", fontsize=FONT)
        ax.set_yticks(y)
        ax.set_yticklabels(lab if k == 0 else [])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlabel("matching proteins" if k == 0 else "mean signed log2 FC", labelpad=1.5)
        ax.text(1.02, -1.0, "p", transform=ax.get_yaxis_transform(), va="center", fontsize=FONT,
                color=INK["secondary"])
        style(ax)
    # B ranks
    letter(sheet, 118, 3, "B")
    r = ranks.query("label_column == 'dct_label' and group == 'all_DCT' and variant == 'none' and "
                    "expected_direction in ['up', 'down']")
    sl = slots.query("label_column == 'dct_label' and group == 'all_DCT' and variant == 'none'").set_index("protein")
    ax = sheet.ax(142, 13, 34, 30)
    y = np.arange(len(r))
    ax.plot(r["pseudobulk_directional_percentile"], y, "o", ms=3.4, color=INK["primary"], ls="", label="all proteins")
    ax.plot(sl.reindex(r["protein"])["pseudobulk_directional_percentile_among_neighbours"], y, "o", ms=3.4,
            mfc="white", mec=INK["primary"], mew=0.6, ls="", label="matched 50")
    ax.axvline(0.5, color=INK["muted"], lw=0.5, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p} ({e})" for p, e in zip(r["protein"], r["expected_direction"])])
    ax.set_ylim(len(r) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("directional percentile", labelpad=1.5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.5, 1.04), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0)
    style(ax)
    # C plate null
    letter(sheet, 3, 72, "C")
    p = plate.query("variant == 'none'")
    prow = [(c, g) for c, g in p[["label_column", "group"]].drop_duplicates().itertuples(index=False)]
    ax = sheet.ax(26, 84, 70, 24)
    marks = {"Ctrl P2 vs P1": ("o", COND["Ctrl"]), "ZeroK P2 vs P1": ("o", COND["ZeroK"]),
             "ZeroK vs Ctrl, P1 only": ("^", PLATE["P1"]), "ZeroK vs Ctrl, P2 only": ("s", PLATE["P2"])}
    for cname, (mk, colr) in marks.items():
        s = p.query("contrast == @cname").set_index(["label_column", "group"]).reindex(prow)
        ax.plot(s["share_q_below_0.05"], np.arange(len(prow)), mk, ms=3.4, color=colr, ls="", label=cname)
    ax.set_yticks(range(len(prow)))
    ax.set_yticklabels([f"{lib.LABEL_SHORT[c][:3] if c != 'label_free' else 'free'}: {g}" for c, g in prow])
    ax.set_ylim(len(prow) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of tested proteins with q < 0.05", labelpad=1.5)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.3, 1.04), ncol=2, frameon=False, handletextpad=0.2,
              columnspacing=0.8, borderaxespad=0)
    style(ax)
    sheet.note(note, nh)
    return save(run, sheet, "supp_specificity", "zerok_vs_control specificity calibration (addendum 5b)", T["checks"])


def side_labels(ax, points: list[tuple[str, float, float]], gap_mm: float = 3.6) -> None:
    """Names in a column right of the axes, sorted by y and spread apart, joined to their points by lines."""
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


def fig_bulk(run, T):
    conc, nulls, signs, matched = T["bulk_conc"], T["bulk_null"], T["bulk_signs"], T["bulk_matched"]
    note, nh = t8.note_block(
        f"{DESIGN_TEXT} Addendum 5c: bulk whole-kidney proteome, 0% vs 1% K+ for 2 weeks (Murali et al., JCI "
        "Insight 2026; 5 vs 7 mice; this group's data), against the SCP zero-K vs control effects (4 days; one "
        "pool per condition). Proteins matched by UniProt accession of the combined-search protein group (else "
        "unique gene symbol); variant none. (A) SCP pseudobulk log2 FC (primary labels, all_DCT) vs bulk log2 FC; "
        "dark = bulk q < 0.05; blue = DCT-enriched in the RNA atlas (kidney_rna); rings = panel proteins with an "
        "expected direction (rings beyond the axis range are drawn at its edge). (B) Spearman ρ (Fisher-z 95% CI) between SCP pseudobulk effects and bulk log2 FC: "
        "ZeroK vs Ctrl pooled and within one plate, and P2 vs P1 within one condition (technical). (C) Share of "
        "proteins significant in both data sets (SCP two-part q < 0.05 with an up/down direction; bulk q < 0.05) "
        "that change in the same direction (Wilson 95% CI). Differences in diet duration, control diet, tissue, "
        "platform and search apply throughout.")
    H = 128 + nh
    sheet = t8.Sheet(H)
    # A scatter
    letter(sheet, 3, 3, "A")
    d = matched.query("label_column == 'dct_label' and group == 'all_DCT' and variant == 'none'")
    ax = sheet.ax(16, 12, 56, 56)
    x, y = d["bulk_log2fc"].to_numpy(float), d["pseudobulk_log2fc_pooled"].to_numpy(float)
    sig = (d["bulk_q"] < 0.05).to_numpy()
    enr = d["dct_enriched"].astype(str).str.lower().eq("true").to_numpy()
    ax.scatter(x[~sig], y[~sig], s=1.2, c=INK["grid"], linewidths=0, rasterized=True)
    ax.scatter(x[sig], y[sig], s=1.6, c=INK["muted"], linewidths=0, rasterized=True)
    ax.scatter(x[enr], y[enr], s=5, c=t2.SERIES[0], linewidths=0, zorder=3)
    lx = float(np.nanquantile(np.abs(x), 0.999)) * 1.05
    ly = float(np.nanquantile(np.abs(y[np.isfinite(y)]), 0.995)) * 1.05
    ax.set_xlim(-lx, lx)
    ax.set_ylim(-ly, ly)
    ax.axhline(0, color=INK["muted"], lw=0.5, zorder=0)
    ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
    pts = []
    for g in EXPECTED_PROTEINS:
        r = d.loc[d["protein"] == g]
        if len(r) and np.isfinite(r["pseudobulk_log2fc_pooled"].iloc[0]):
            px = float(np.clip(r["bulk_log2fc"].iloc[0], -lx, lx))
            py = float(np.clip(r["pseudobulk_log2fc_pooled"].iloc[0], -ly, ly))
            ax.scatter([px], [py], s=16, facecolors="none", edgecolors=INK["primary"], linewidths=0.6, zorder=4,
                       clip_on=False)
            pts.append((g, px, py))
    side_labels(ax, [p for p in pts if p[1] >= 0])
    # proteins left of zero: names stacked in the empty lower-left corner (no line across the plot)
    left = sorted([p for p in pts if p[1] < 0], key=lambda t: -t[1])   # rightmost point -> top label (no crossing)
    step = 3.6 / 56 * (2 * ly)                       # 3.6 mm in data units (axes 56 mm tall)
    for i, (g, px, py) in enumerate(left):
        ty = -0.35 * ly - i * step
        ax.annotate("", xy=(px, py), xycoords="data", xytext=(-0.64 * lx, ty), textcoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=INK["muted"], shrinkA=0, shrinkB=2))
        ax.text(-0.62 * lx, ty, g, va="center", ha="left", fontsize=FONT)
    c = conc.query("label_column == 'dct_label' and group == 'all_DCT' and variant == 'none' and "
                   "scp_effect == 'pseudobulk' and protein_set == 'all shared'").iloc[0]
    ax.text(0.03, 0.97, f"ρ = {c['spearman']:.2f}\nn = {int(c['n'])}", transform=ax.transAxes, va="top",
            fontsize=FONT)
    ax.set_xlabel("bulk log2 FC, 0% vs 1% K+ (whole kidney)", labelpad=1.5)
    ax.set_ylabel("SCP pseudobulk log2 FC, ZeroK vs Ctrl", labelpad=1.5)
    style(ax)
    # B correlations
    rows = [(c_, g_) for c_, g_ in nulls[["label_column", "group"]].drop_duplicates().itertuples(index=False)]
    lab = [f"{lib.LABEL_SHORT[c_][:3] if c_ != 'label_free' else 'free'}: {g_}" for c_, g_ in rows]
    marks = {"ZeroK vs Ctrl (pooled)": ("o", INK["primary"], INK["primary"]),
             "ZeroK vs Ctrl, P1 only": ("^", PLATE["P1"], PLATE["P1"]),
             "ZeroK vs Ctrl, P2 only": ("s", PLATE["P2"], PLATE["P2"]),
             "Ctrl P2 vs P1 (technical)": ("o", COND["Ctrl"], "white"),
             "ZeroK P2 vs P1 (technical)": ("o", COND["ZeroK"], "white")}
    letter(sheet, 104, 3, "B")
    for k, pset in enumerate(("all shared", "bulk q < 0.05")):
        ax = sheet.ax(126, 22 + k * 34, 50, 25)
        s0 = nulls.query("variant == 'none' and protein_set == @pset")
        offs = np.linspace(-0.28, 0.28, len(marks))
        for (cname, (mk, ec, fc)), off in zip(marks.items(), offs):
            s = s0.query("contrast == @cname").set_index(["label_column", "group"]).reindex(rows)
            yy = np.arange(len(rows)) + off
            ax.hlines(yy, s["ci_low"], s["ci_high"], color=ec, lw=0.6)
            ax.plot(s["spearman"], yy, mk, ms=3, mec=ec, mfc=fc, mew=0.6, ls="", label=cname)
        ax.axvline(0, color=INK["muted"], lw=0.5, zorder=0)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(lab)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(f"{pset}", fontsize=FONT, pad=2)
        if k == 1:
            ax.set_xlabel("Spearman ρ with bulk log2 FC", labelpad=1.5)
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(-0.55, 1.13), ncol=2, frameon=False, handletextpad=0.2,
                      columnspacing=0.6, borderaxespad=0)
        style(ax)
    # C sign agreement
    letter(sheet, 3, 82, "C")
    ax = sheet.ax(30, 92, 42, 22)
    for k, (pset, colr) in enumerate((("all shared", INK["primary"]), ("DCT-enriched", t2.SERIES[0]))):
        s = signs.query("variant == 'none' and protein_set == @pset").set_index(["label_column", "group"]).reindex(rows)
        yy = np.arange(len(rows)) + (k - 0.5) * 0.3
        ax.hlines(yy, s["agreement_ci_low"], s["agreement_ci_high"], color=colr, lw=0.6)
        ax.plot(s["agreement"], yy, "o", ms=3, color=colr, ls="", label=pset)
    ax.axvline(0.5, color=INK["muted"], lw=0.5, zorder=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(lab)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("same direction (both significant)", labelpad=1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False, handletextpad=0.2, borderaxespad=0)
    style(ax)
    sheet.note(note, nh)
    return save(run, sheet, "supp_bulk_benchmark", "zerok_vs_control benchmark against the bulk low-K kidney proteome "
                "(addendum 5c)", T["checks"])


def main() -> None:
    global LOW_COVERAGE
    parser = argparse.ArgumentParser(description="zerok_vs_control step 7: figures.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "figures", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    rd = lambda name, **kw: lib.read_table(run, name, **kw)  # noqa: E731
    T = {"fractions": rd("label_fractions"), "fraction_diff": rd("label_fraction_differences"),
         "contamination": rd("contamination_detection"),
         "contamination_coverage": rd("contamination_coverage", keep_default_na=False),
         "panel": rd("panel_results"), "proteome_groups": rd("proteome_groups"),
         "state_per_cell": rd("state_per_cell", dtype={"cell_id": str}), "cycling": rd("state_cycling"),
         "depth_matched": rd("state_depth_matched"), "state_checks": rd("state_checks"),
         "panel_repeats": rd("robustness_panel_repeats"), "proteome_effects": rd("robustness_proteome_effects"),
         "leave_slc": rd("robustness_leave_slc12a3_out"), "label_free_proteome": rd("robustness_label_free_proteome"),
         "spec_null": rd("specificity_matched_null"), "spec_ranks": rd("specificity_panel_ranks"),
         "spec_slots": rd("specificity_matched_slots"), "spec_plate": rd("specificity_plate_null"),
         "bulk_conc": rd("bulk_concordance"), "bulk_null": rd("bulk_plate_null"), "bulk_signs": rd("bulk_sign_tables"),
         "bulk_matched": rd("bulk_matched"), "checks": []}
    for col in lib.LABEL_COLUMNS:
        for v in lib.VARIANTS:
            T[f"proteome_{lib.LABEL_SHORT[col]}_{v}"] = rd(f"proteome_{lib.LABEL_SHORT[col]}_{v}")
    items = list(t22.panel_spec_items(t22.read_panel(run)))
    base = tested(T["panel"]).query("label_column == 'dct_label' and group == 'all_DCT' and variant == 'none' "
                                    "and item_type == 'protein'")
    low = base.loc[np.minimum(base["detection_fraction_ctrl"], base["detection_fraction_zerok"])
                   < t22.MIN_DETECTED_FRACTION, "item"]
    LOW_COVERAGE = tuple(low)
    checks.add("panel proteins marked * (detected in < 10% of cells in a condition; primary all_DCT, variant none)",
               "; ".join(LOW_COVERAGE) or "none", None)
    extra = []
    ok = [fig_main(run, T, extra, items), fig_composition(run, T), fig_contamination(run, T),
          fig_panel_overview(run, T, items), fig_volcanoes(run, T), fig_volcano_variants(run, T), fig_state(run, T),
          fig_robustness(run, T, items), fig_specificity(run, T), fig_bulk(run, T)]
    fc = pd.DataFrame(T["checks"])
    run.table(fc, "figure_checks")
    checks.add("figures passing the format checks (180 mm, Arial >= 7 pt, no text overlap, inside the figure, "
               "PDF + 600-dpi PNG + TIFF)", f"{sum(ok)} of {len(ok)}", all(ok))
    for r in extra:
        checks.add(f"{r['figure']}: proteins without an intensity part not drawn; capped points",
                   f"{r['proteins_without_intensity_part_not_shown']}; {r['points_capped']}", None)
    run.table(checks.frame(), "figures_step_checks")
    run.finish(parameters={"width_mm": t8.W, "font": "Arial", "min_font_pt": FONT, "dpi": 600,
                           "formats": [".pdf", ".png", ".tif"], "low_coverage_marked": list(LOW_COVERAGE),
                           "ported_from": "zerok_response_core_figures.py",
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A figure check failed; see zerok_vs_control_figure_checks.csv.")


if __name__ == "__main__":
    main()
