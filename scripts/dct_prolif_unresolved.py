#!/usr/bin/env python
"""dct_prolif information only: cells whose DCT1/DCT2 subtype route A does not resolve, described.

Analysis plan
(frozen). The authors asked on 3 Nov 2026 whether the group without a subtype could be made smaller, because reviewers will
ask why many DCT cells carry no subtype. After the equal-variance and posterior-0.5 variants (information only; the
plan fit was kept), the authors chose 'Reframe (1)': no new call; these cells are reported as DCT with the subtype not
resolved, and the per-cell axis is reported. The authors then asked for the label 'DCT, subtype not resolved' everywhere
(dct_prolif_lib.UNRESOLVED). This step describes these cells from the stored dct_prolif tables only.

Nothing is refitted or recomputed from the dct_intake objects: every value comes from dct_prolif_route_a_per_cell.csv (both
runs, both conditions), the plan fit's call zones in dct_prolif_mixvar_summary.csv and the dct_reviewer contamination flags
already counted in dct_prolif_contamination_by_label.csv. Groups are the route-A labels (`label`: ProLIF overlays
cycling cells), so 'DCT, subtype not resolved' here excludes cycling cells. The cs_t31_* / cs_t31nf_* labels, the
canonical table and the labelled objects are not changed by this step; dct_labels labels stay primary.

Outputs (results/tables/dct_prolif/):
  dct_prolif_unresolved_summary.csv: per run x condition x label: cells, Slc12a3 detected, protein IDs (median, IQR),
    identity markers detected (of the 8 used), axis A (median, IQR), plate split, dct_reviewer contamination flags;
  dct_prolif_unresolved_by_markers.csv: cells with a DCT1, DCT2 or 'DCT, subtype not resolved' label by identity
    markers detected, and the fraction with the subtype not resolved;
  dct_prolif_unresolved_axis_zones.csv: the same labels by position on axis A against the observed calls of the plan
    fit (left of the DCT1 calls, within them, between the DCT1 and DCT2 calls, within the right-hand DCT2 calls);
  dct_prolif_unresolved_odds.csv: subtype not resolved vs DCT1 + DCT2 labels: ~ z(ln n_detected) + plate
    (lib.logit_or) and ~ identity markers detected (per marker);
  dct_prolif_unresolved_axis_per_cell.csv: per cell (both runs): plate, protein IDs, markers detected, Slc12a3, scores,
    axis, axis zone, posteriors, identity and label;
  dct_prolif_unresolved_response_text.md: observed numbers and a draft paragraph for the response letter (numbers filled
    in from the tables above);
  dct_prolif_unresolved_checks.csv (checks plus one format row per figure); dct_prolif_unresolved_figure_captions.md.
Figures (results/figures/dct_prolif/; dotplot format):
  dct_prolif_unresolved_axis_vs_depth_<run>_<cond>_85mm: axis A against protein IDs, coloured by label, with the
    boundaries of the plan fit's call zones;
  dct_prolif_unresolved_by_markers_180mm: fraction with the subtype not resolved by identity markers detected, both
    runs and conditions.

Command (project root):  python scripts/dct_prolif_unresolved.py
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

import scp_common as cc
import cluster_de_lib as t5
import dotplot_dotplots as t6d
import dct_reviewer_lib as t28
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_unresolved.py"
REQUEST = {
    "date": "2026-11-03",
    "request": "authors, dct_prolif session: reviewers will ask why many DCT cells cannot be labelled; after the mixture "
               "variants the authors chose 'Reframe (1)' (no new call; report these cells as DCT with the "
               "subtype not resolved, with the per-cell axis) and approved this step ('Run it')",
    "status": "information only; cs_t31_* and cs_t31nf_* labels unchanged; dct_labels labels stay primary",
}
AXIS_LABELS = (lib.DCT1, lib.DCT2, lib.UNRESOLVED)
# Positions on axis A against the observed calls of the plan fit (its calls are intervals: DCT2 | not resolved |
# DCT1 | not resolved | DCT2, section 10 of the README).
ZONE_LEFT = "left of the DCT1 calls"
ZONE_DCT1 = "within the DCT1 calls"
ZONE_BETWEEN = "between the DCT1 and DCT2 calls"
ZONE_DCT2 = "within the right-hand DCT2 calls"
ZONES = (ZONE_LEFT, ZONE_DCT1, ZONE_BETWEEN, ZONE_DCT2)
ALLOWED = {lib.DCT1: {ZONE_DCT1}, lib.DCT2: {ZONE_LEFT, ZONE_DCT2}, lib.UNRESOLVED: {ZONE_LEFT, ZONE_BETWEEN}}
HIGH_MARKERS = (7, 8)
RUN_TITLE = {"frozen": "floor 1,000", "nofloor": "no floor"}
W85, W180 = t28.HALF_MM, t28.FULL_MM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def q(values: np.ndarray, p: float) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else float("nan")


def call_bounds(axis: np.ndarray, identity: np.ndarray) -> dict:
    """Observed range of the DCT1 calls and the lowest right-hand DCT2 call (axis above the DCT1 calls)."""
    a1 = axis[identity == lib.DCT1]
    lo1, hi1 = float(a1.min()), float(a1.max())
    a2 = axis[(identity == lib.DCT2) & (axis > hi1)]
    return {"dct1_calls_min": lo1, "dct1_calls_max": hi1,
            "dct2_right_calls_min": float(a2.min()) if len(a2) else float("inf")}


def zone_of(axis: np.ndarray, b: dict) -> np.ndarray:
    return np.where(axis < b["dct1_calls_min"], ZONE_LEFT,
                    np.where(axis <= b["dct1_calls_max"], ZONE_DCT1,
                             np.where(axis < b["dct2_right_calls_min"], ZONE_BETWEEN, ZONE_DCT2)))


def zone_edges(zones_text: str) -> list[float]:
    """Boundaries between the plan's call zones, from dct_prolif_mixvar_summary.csv (text, 2 decimals)."""
    import re

    z = re.findall(r"(DCT1|DCT2|DCT, subtype not resolved) (-?[\d.]+) to (-?[\d.]+)", zones_text)
    return [float(b) for _, _, b in z[:-1]]


def logit_markers(y: np.ndarray, n_markers: np.ndarray) -> dict:
    """Logistic y ~ identity markers detected (count): odds ratio per marker with 95% CI (statsmodels)."""
    import warnings

    import statsmodels.api as sm

    y = np.asarray(y, float)
    out = {"markers_computable": False, "markers_reason": ""}
    if y.sum() == 0 or y.sum() == len(y):
        out["markers_reason"] = "no positive cells" if y.sum() == 0 else "all cells positive"
        return out
    X = sm.add_constant(np.asarray(n_markers, float))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:          # perfect separation, non-convergence
        out["markers_reason"] = f"fit failed: {type(exc).__name__}: {exc}"[:300]
        return out
    ci = np.exp(fit.conf_int(0.05))
    out.update(markers_computable=True, markers_or=float(np.exp(fit.params[1])), markers_ci_low=float(ci[1, 0]),
               markers_ci_high=float(ci[1, 1]), markers_p=float(fit.pvalues[1]))
    return out


def pct(n: int, d: int) -> str:
    return f"{n} of {d} ({100 * n / d:.0f}%)" if d else f"{n} of 0"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def scatter_figure(f: pd.DataFrame, edges: list[float], title: str, note_text: str):
    W, H = W85, 93.0
    fig = t28.new_figure(W, H)
    ax = fig.add_axes([15.0 / W, 37.0 / H, 66.0 / W, 47.0 / H])
    counts = {}
    for lab in (lib.DCT1, lib.UNRESOLVED, lib.DCT2):
        m = f["label"].eq(lab)
        counts[lab] = int(m.sum())
        ax.scatter(f.loc[m, "axis"], f.loc[m, "n_detected"], s=3.5, lw=0, alpha=0.85,
                   color=lib.LABEL_COLORS[lab], zorder=2 if lab != lib.DCT2 else 3, rasterized=False)
    for e in edges:
        ax.axvline(e, color=t6d.INK["secondary"], lw=0.5, ls=(0, (1.5, 1.5)), zorder=1)
    ax.set_yscale("log")
    lo, hi = float(f["n_detected"].min()), float(f["n_detected"].max())
    ticks = [t for t in (100, 250, 500, 1000, 2000, 4000, 8000) if lo * 0.8 <= t <= hi * 1.25]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.set_xlabel("Axis A (DCT2 score minus DCT1 score)")
    ax.set_ylabel("Protein IDs per cell")
    ax.set_title(title, loc="left", fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=lib.LABEL_COLORS[lib.DCT1], lw=0, label=f"DCT1 ({counts[lib.DCT1]})"),
               Patch(facecolor=lib.LABEL_COLORS[lib.DCT2], lw=0, label=f"DCT2 ({counts[lib.DCT2]})"),
               Patch(facecolor=lib.LABEL_COLORS[lib.UNRESOLVED], lw=0,
                     label=f"DCT, subtype not resolved ({counts[lib.UNRESOLVED]})"),
               Line2D([], [], color=t6d.INK["secondary"], lw=0.5, ls=(0, (1.5, 1.5)), label="Call-zone boundary")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(15.0 / W, 24.5 / H), ncol=1, frameon=False,
               handlelength=1.4, handletextpad=0.4, borderaxespad=0, fontsize=7, labelspacing=0.3)
    fig.text(1.5 / W, 1.5 / H, note_text, fontsize=7, color=t6d.INK["secondary"], ha="left", va="bottom")
    return fig, counts


def markers_figure(bym: pd.DataFrame, note_text: str):
    W, H = W180, 72.0
    fig = t28.new_figure(W, H)
    ax = fig.add_axes([16.0 / W, 20.0 / H, 118.0 / W, 46.0 / H])
    ks = sorted(bym["n_markers_detected"].unique())
    series = [(r, c) for c in lib.CONDITIONS for r in lib.RUN_KEYS]
    width = 0.8 / len(series)
    handles = []
    for i, (r, c) in enumerate(series):
        s = bym[(bym["run"] == r) & (bym["condition"] == c)].set_index("n_markers_detected")
        x = np.arange(len(ks)) + (i - (len(series) - 1) / 2) * width
        y = [float(s.loc[k, "fraction_unresolved"]) if k in s.index else np.nan for k in ks]
        alpha = 1.0 if r == "frozen" else 0.45
        ax.bar(x, y, width=width * 0.92, color=t28.COND_COLORS[c], alpha=alpha, lw=0)
        handles.append(Patch(facecolor=t28.COND_COLORS[c], alpha=alpha, lw=0, label=f"{c}, {RUN_TITLE[r]}"))
    ax.set_xticks(np.arange(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlim(-0.6, len(ks) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Identity markers detected per cell (of 8 used)")
    ax.set_ylabel("Fraction subtype not resolved")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(140.0 / W, 66.0 / H), ncol=1, frameon=False,
               handlelength=1.4, handletextpad=0.4, borderaxespad=0, fontsize=7, labelspacing=0.4)
    fig.text(140.0 / W, 30.0 / H, note_text, fontsize=7, color=t6d.INK["secondary"], ha="left", va="top")
    return fig


# ---------------------------------------------------------------------------
# Response text
# ---------------------------------------------------------------------------
def response_text(summ: pd.DataFrame, bym: pd.DataFrame, zones: pd.DataFrame, odds: pd.DataFrame) -> str:
    def s(r, c, lab):
        return summ[(summ["run"] == r) & (summ["condition"] == c) & (summ["label"] == lab)].iloc[0]

    def both(fn) -> str:
        """'<frozen> (no floor: <nofloor>)'."""
        return f"{fn('frozen')} (no floor: {fn('nofloor')})"

    def axis_n(r, c):
        return int(sum(s(r, c, lab)["n_cells"] for lab in AXIS_LABELS))

    def high(r, c):
        b = bym[(bym["run"] == r) & (bym["condition"] == c) & bym["n_markers_detected"].isin(HIGH_MARKERS)]
        return pct(int(b["n_unresolved"].sum()), int(b["n_cells"].sum()))

    def zone_n(r, c, z):
        m = zones[(zones["run"] == r) & (zones["condition"] == c) & (zones["label"] == lib.UNRESOLVED)
                  & (zones["zone"] == z)]
        return int(m["n_cells"].sum())

    def o(r, c, key):
        row = odds[(odds["run"] == r) & (odds["condition"] == c)].iloc[0]
        if key == "depth":
            return (f"{row['depth_or']:.2f} ({row['depth_ci_low']:.2f}-{row['depth_ci_high']:.2f})"
                    if row["computable"] else f"not computable ({row['reason']})")
        return (f"{row['markers_or']:.2f} ({row['markers_ci_low']:.2f}-{row['markers_ci_high']:.2f})"
                if row["markers_computable"] else f"not computable ({row['markers_reason']})")

    L = ["# dct_prolif: cells with the DCT subtype not resolved, numbers for the response letter (draft; observed facts only)", "",
         f"{cc.GENERATED_TAG} ({SCRIPT}, {cc.now_utc()}). Information only (authors, 3 Nov 2026: 'Reframe (1)'): "
         "no new call; the dct_prolif labels are unchanged and the dct_labels labels stay primary. Values: frozen run (depth floor "
         "1,000 protein IDs), with the approved no-floor run in brackets. 'DCT, subtype not resolved' is the route-A label for "
         "cells with at least one DCT1 and one DCT2 identity marker detected whose larger mixture posterior is "
         "below 0.8 (cycling cells are ProLIF or 'Cycling, NCC not detected' and are not counted here). Identity "
         "markers used: DCT1 " + ", ".join(lib.EXPECTED_MARKERS["DCT1"]) + "; DCT2 "
         + ", ".join(lib.EXPECTED_MARKERS["DCT2"]) + ".", ""]
    L += ["## Observed", ""]
    for c in lib.CONDITIONS:
        L += [f"**{c}**", ""]
        L.append("- DCT, subtype not resolved: " + both(lambda r: pct(int(s(r, c, lib.UNRESOLVED)["n_cells"]), axis_n(r, c)))
                 + " of the cells on the DCT1/DCT2 axis (labels DCT1, DCT2 or DCT, subtype not resolved).")
        L.append("- Slc12a3 (NCC) detected: subtype not resolved "
                 + both(lambda r: f"{100 * s(r, c, lib.UNRESOLVED)['fraction_slc12a3_detected']:.0f}%")
                 + "; DCT1 " + both(lambda r: f"{100 * s(r, c, lib.DCT1)['fraction_slc12a3_detected']:.0f}%")
                 + "; DCT2 " + both(lambda r: f"{100 * s(r, c, lib.DCT2)['fraction_slc12a3_detected']:.0f}%") + ".")
        L.append("- Median protein IDs: subtype not resolved "
                 + both(lambda r: f"{s(r, c, lib.UNRESOLVED)['median_ids']:,.0f}")
                 + "; DCT1 " + both(lambda r: f"{s(r, c, lib.DCT1)['median_ids']:,.0f}")
                 + "; DCT2 " + both(lambda r: f"{s(r, c, lib.DCT2)['median_ids']:,.0f}") + ".")
        L.append("- Median identity markers detected (of 8): subtype not resolved "
                 + both(lambda r: f"{s(r, c, lib.UNRESOLVED)['median_markers_detected']:g}")
                 + "; DCT1 " + both(lambda r: f"{s(r, c, lib.DCT1)['median_markers_detected']:g}")
                 + "; DCT2 " + both(lambda r: f"{s(r, c, lib.DCT2)['median_markers_detected']:g}") + ".")
        L.append(f"- Cells with {HIGH_MARKERS[0]} or {HIGH_MARKERS[1]} identity markers detected that are "
                 "subtype not resolved: " + both(lambda r: high(r, c)) + ".")
        L.append("- Subtype not resolved vs DCT1 + DCT2, odds ratio (95% CI) per SD of ln(protein IDs), adjusted for plate: "
                 + both(lambda r: o(r, c, "depth")) + "; per identity marker detected: "
                 + both(lambda r: o(r, c, "markers")) + ".")
        L.append("- Position of the subtype-unresolved cells on axis A (observed calls of the plan fit): " + ZONE_BETWEEN
                 + " " + both(lambda r: str(zone_n(r, c, ZONE_BETWEEN))) + "; " + ZONE_LEFT + " (the plan fit's "
                 "left-end band, README section 10) " + both(lambda r: str(zone_n(r, c, ZONE_LEFT))) + ".")
        L.append("")

    def para(c):
        r = "frozen"
        ind = s(r, c, lib.UNRESOLVED)
        return (f"{c}: {int(ind['n_cells'])} of {axis_n(r, c)} cells; Slc12a3 detected in "
                f"{100 * ind['fraction_slc12a3_detected']:.0f}%; median {ind['median_ids']:,.0f} protein IDs "
                f"(DCT1 {s(r, c, lib.DCT1)['median_ids']:,.0f}); median {ind['median_markers_detected']:g} of 8 "
                f"identity markers detected (DCT1 {s(r, c, lib.DCT1)['median_markers_detected']:g}, DCT2 "
                f"{s(r, c, lib.DCT2)['median_markers_detected']:g})")

    L += ["## Draft paragraph (for the authors to edit; frozen run)", "",
          "> Cells whose DCT1 and DCT2 marker scores did not separate them at a posterior probability of 0.8 were "
          "not forced into a subtype and are reported as DCT with the subtype not resolved ("
          + "; ".join(para(c) for c in lib.CONDITIONS) + "). Of these, "
          + " and ".join(f"{zone_n('frozen', c, ZONE_BETWEEN)} ({c})" for c in lib.CONDITIONS)
          + " lie between the DCT1 and DCT2 calls on the marker axis. The odds ratio of an unresolved call per SD "
          "of ln(protein IDs), adjusted for sort plate, was "
          + " and ".join(f"{o('frozen', c, 'depth')} in {c}" for c in lib.CONDITIONS)
          + ", and " + " and ".join(f"{high('frozen', c)} ({c})" for c in lib.CONDITIONS)
          + " of the cells with 7 or 8 of the 8 identity markers detected remained unresolved. Per-cell axis "
          "values, posteriors and markers detected are given in Supplementary Table X "
          "(`dct_prolif_unresolved_axis_per_cell.csv`).", "",
          "Coverage limits: one pool of 3 mice per condition (cells are not animal-level replicates); 8 identity "
          "markers detected in at least 20 cells per condition (Trpv5 in fewer than 20; Slc12a3 and Umod shared and "
          "not used); the frozen run leaves cells below 1,000 protein IDs without a subtype ('Below depth floor')."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif information only: cells with the DCT subtype not resolved, described.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "unresolved", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "unresolved_checks")

    per_a = lib.read_table(run, "route_a_per_cell", dtype={"cell_id": str, "cell_key": str})
    summ_a = lib.read_table(run, "route_a_summary")
    contam = lib.read_table(run, "contamination_by_label")
    mixvar = lib.read_table(run, "mixvar_summary")
    for b in ("above_floor", "slc12a3_detected"):
        per_a[b] = per_a[b].astype(str).str.lower().eq("true")

    summary, bym_rows, zone_rows, odds_rows, cells = [], [], [], [], []
    figs = t28.Figures(run, "unresolved")
    captions = ["# dct_prolif cells with the DCT subtype not resolved: figure captions (draft; observed facts only)", "",
                f"{cc.GENERATED_TAG} ({SCRIPT}). Information only (authors, 3 Nov 2026: 'Reframe (1)'); the dct_prolif labels "
                "are unchanged.", ""]
    for run_key in lib.RUN_KEYS:
        for cond in lib.CONDITIONS:
            tag = f"{run_key} / {cond}"
            f = per_a[(per_a["run"] == run_key) & (per_a["condition"] == cond)].reset_index(drop=True)
            srow = summ_a[(summ_a["run"] == run_key) & (summ_a["condition"] == cond)].iloc[0]
            label = f["label"].astype(str).to_numpy()
            identity = f["identity"].astype(str).to_numpy()
            checks.add(f"{tag}: label counts equal dct_prolif_route_a_summary.csv", lib.counts_text(label),
                       lib.counts_text(label) == srow["label_counts"])
            checks.add(f"{tag}: identity counts equal dct_prolif_route_a_summary.csv",
                       lib.counts_text(identity, lib.IDENTITY_ORDER),
                       lib.counts_text(identity, lib.IDENTITY_ORDER) == srow["identity_counts"])

            axis = f["axis"].to_numpy(float)
            defined = np.isfinite(axis)
            n_mark = (f["n_DCT1_markers_detected"].fillna(0) + f["n_DCT2_markers_detected"].fillna(0)).to_numpy(int)
            has_both = (f["n_DCT1_markers_detected"].fillna(0) >= 1) & (f["n_DCT2_markers_detected"].fillna(0) >= 1)
            on_axis = np.isin(identity, AXIS_LABELS)
            checks.add(f"{tag}: axis A defined exactly for cells with >= 1 DCT1 and >= 1 DCT2 marker above the floor",
                       int((defined != (has_both.to_numpy() & f["above_floor"].to_numpy())).sum()),
                       np.array_equal(defined, has_both.to_numpy() & f["above_floor"].to_numpy()), "0 cells differ")
            checks.add(f"{tag}: DCT1, DCT2 and subtype-not-resolved identities are exactly the cells with axis A defined",
                       int((on_axis != defined).sum()), np.array_equal(on_axis, defined), "0 cells differ")
            post = f["posterior"].to_numpy(float)
            bad = int(((identity == lib.UNRESOLVED) & (post >= lib.POSTERIOR_MIN)).sum()
                      + (np.isin(identity, (lib.DCT1, lib.DCT2)) & (post < lib.POSTERIOR_MIN)).sum())
            checks.add(f"{tag}: subtype not resolved iff larger posterior < 0.8 (cells on the axis)", bad, bad == 0,
                       "0 cells differ")

            # Zones against the observed calls of the plan fit; cross-check with the mixture-variant table ------
            bounds = call_bounds(axis, identity)
            zone = np.full(len(f), "", dtype=object)
            zone[defined] = zone_of(axis[defined], bounds)
            bad = sum(int((on_axis & (identity == lab) & ~np.isin(zone, list(ok))).sum()) for lab, ok in ALLOWED.items())
            checks.add(f"{tag}: calls are intervals on axis A (DCT1 only within the DCT1 range; subtype not resolved only "
                       "left of it or between the DCT1 and DCT2 calls; DCT2 only at the two ends)", bad, bad == 0,
                       "0 cells outside")
            mv = mixvar[(mixvar["run"] == run_key) & (mixvar["condition"] == cond) & (mixvar["variant"] == "plan")].iloc[0]
            n_left = int(((identity == lib.UNRESOLVED) & (zone == ZONE_LEFT)).sum())
            checks.add(f"{tag}: subtype-not-resolved identities left of the DCT1 calls equal those below the DCT1 component "
                       "mean in dct_prolif_mixvar_summary.csv", f"{n_left} vs {int(mv['n_below_dct1_mean_unresolved'])}",
                       n_left == int(mv["n_below_dct1_mean_unresolved"]))
            edges = zone_edges(str(mv["call_zones_observed_range"]))

            # Summary per label ------------------------------------------------------------------------------------
            for lab in cc.ordered_labels(pd.unique(label), lib.LABEL_ORDER):
                m = label == lab
                ids = f.loc[m, "n_detected"].to_numpy(float)
                cr = contam[(contam["run"] == run_key) & (contam["condition"] == cond) & (contam["label"] == lab)]
                if len(cr) != 1 or int(cr.iloc[0]["n_cells"]) != int(m.sum()):
                    checks.add(f"{tag} / {lab}: one contamination row with the same cell count",
                               f"{len(cr)} rows", False)
                    continue
                cr = cr.iloc[0]
                summary.append({
                    "run": run_key, "condition": cond, "floor": lib.RUNS[run_key].floor, "label": lab,
                    "n_cells": int(m.sum()),
                    "n_slc12a3_detected": int(f.loc[m, "slc12a3_detected"].sum()),
                    "fraction_slc12a3_detected": float(f.loc[m, "slc12a3_detected"].mean()),
                    "median_ids": q(ids, 50), "q1_ids": q(ids, 25), "q3_ids": q(ids, 75),
                    "n_axis_defined": int((m & defined).sum()),
                    "median_markers_detected": q(n_mark[m & defined], 50) if (m & defined).any() else np.nan,
                    "median_DCT1_markers_detected": q(f.loc[m & defined, "n_DCT1_markers_detected"], 50),
                    "median_DCT2_markers_detected": q(f.loc[m & defined, "n_DCT2_markers_detected"], 50),
                    "median_axis": q(axis[m], 50), "q1_axis": q(axis[m], 25), "q3_axis": q(axis[m], 75),
                    "n_P1": int((f.loc[m, "plate"] == "P1").sum()), "n_P2": int((f.loc[m, "plate"] == "P2").sum()),
                    "n_tal_like_flag": int(cr["n_tal_like_flag"]), "n_cnt_cd_like_flag": int(cr["n_cnt_cd_like_flag"]),
                    "n_interstitial_like_flag": int(cr["n_interstitial_like_flag"]),
                    "n_any_flag": int(cr["n_any_flag"]), "fraction_any_flag": float(cr["fraction_any_flag"])})

            # Cells with a DCT1, DCT2 or 'DCT, subtype not resolved' label (non-cycling, axis defined) ---------------------------
            ax_lab = np.isin(label, AXIS_LABELS)
            for k in sorted(np.unique(n_mark[ax_lab])):
                m = ax_lab & (n_mark == k)
                n_ind = int((m & (label == lib.UNRESOLVED)).sum())
                bym_rows.append({"run": run_key, "condition": cond, "n_markers_detected": int(k),
                                 "n_cells": int(m.sum()), "n_DCT1": int((m & (label == lib.DCT1)).sum()),
                                 "n_DCT2": int((m & (label == lib.DCT2)).sum()), "n_unresolved": n_ind,
                                 "fraction_unresolved": n_ind / int(m.sum())})
            for lab in AXIS_LABELS:
                for z in ZONES:
                    zone_rows.append({"run": run_key, "condition": cond, "label": lab, "zone": z, **bounds,
                                      "n_cells": int(((label == lab) & (zone == z)).sum())})
            y = label[ax_lab] == lib.UNRESOLVED
            res = lib.logit_or(y, np.log(f.loc[ax_lab, "n_detected"].to_numpy(float)), f.loc[ax_lab, "plate"].to_numpy())
            resm = logit_markers(y, n_mark[ax_lab])
            odds_rows.append({"run": run_key, "condition": cond, "comparison": "Subtype not resolved vs DCT1 + DCT2 (labels)",
                              "model_depth": "Subtype not resolved ~ z(ln n_detected) + plate",
                              "model_markers": "Subtype not resolved ~ identity markers detected",
                              "n_unresolved": int(y.sum()), "n_called": int((~y).sum()), **res, **resm})
            checks.add(f"{tag}: Subtype not resolved vs DCT1 + DCT2, depth OR (95% CI); per-marker OR (95% CI)",
                       (f"{res['depth_or']:.2f} ({res['depth_ci_low']:.2f}-{res['depth_ci_high']:.2f})"
                        if res["computable"] else res["reason"]) + "; "
                       + (f"{resm['markers_or']:.2f} ({resm['markers_ci_low']:.2f}-{resm['markers_ci_high']:.2f})"
                          if resm["markers_computable"] else resm["markers_reason"]), None)

            cells.append(pd.DataFrame({
                "run": run_key, "condition": cond, "floor": lib.RUNS[run_key].floor, "cell_id": f["cell_id"],
                "cell_key": f["cell_key"], "plate": f["plate"], "n_detected": f["n_detected"],
                "above_floor": f["above_floor"], "n_DCT1_markers_detected": f["n_DCT1_markers_detected"],
                "n_DCT2_markers_detected": f["n_DCT2_markers_detected"], "slc12a3_detected": f["slc12a3_detected"],
                "cycling": f["cycling"], "DCT1_score": f["DCT1_score"], "DCT2_score": f["DCT2_score"], "axis": axis,
                "axis_zone": np.where(defined, zone, ""), "posterior": f["posterior"],
                "posterior_dct2": f["posterior_dct2"], "identity": identity, "label": label}))

            # Figure: axis A against depth -------------------------------------------------------------------------
            t28.style()
            fa = f[np.isin(label, AXIS_LABELS)]
            note_text = (f"{len(fa)} non-cycling cells with both marker types\ndetected; dotted lines: call zones "
                         "of the plan fit.")
            fig, counts = scatter_figure(fa, edges, f"{cond}, {RUN_TITLE[run_key]}", note_text)
            stem = f"unresolved_axis_vs_depth_{run_key}_{lib.KEY[cond]}"
            figs.save(fig, f"{stem}_85mm", f"Axis A against protein IDs by label, {cond}, {run_key}", W85)
            plt.close(fig)
            captions.append(f"- `dct_prolif_{stem}_85mm`: {cond}, {lib.RUNS[run_key].display}. Axis A (DCT2 minus "
                            f"DCT1 residual-z score) against protein IDs per cell (log scale) for the {len(fa)} "
                            "non-cycling cells with at least one DCT1 and one DCT2 identity marker detected, coloured "
                            f"by route-A label: DCT1 {counts[lib.DCT1]}; DCT2 {counts[lib.DCT2]}; DCT, subtype not "
                            f"resolved {counts[lib.UNRESOLVED]}. Dotted lines: boundaries of the plan fit's call zones "
                            f"({mv['call_zones_observed_range']}).")
            print(f"[{tag}] done", flush=True)

    bym = pd.DataFrame(bym_rows)
    t28.style()
    note_text = ("Non-cycling cells with at\nleast one DCT1 and one DCT2\nidentity marker detected\n"
                 "(labels DCT1, DCT2 or\nDCT, subtype not resolved).")
    fig = markers_figure(bym, note_text)
    figs.save(fig, "unresolved_by_markers_180mm", "Fraction with the subtype not resolved by identity markers detected",
              W180)
    plt.close(fig)
    captions.append("- `dct_prolif_unresolved_by_markers_180mm`: fraction of cells labelled 'DCT, subtype not resolved' "
                    "by the number of identity markers detected per cell (of the 8 used: DCT1 " + ", ".join(lib.EXPECTED_MARKERS["DCT1"])
                    + "; DCT2 " + ", ".join(lib.EXPECTED_MARKERS["DCT2"]) + "), among non-cycling cells with at least "
                    "one DCT1 and one DCT2 marker detected; Ctrl and ZeroK, frozen run (floor 1,000; solid) and "
                    "no-floor run (light). Cells per bar: `dct_prolif_unresolved_by_markers.csv`.")

    frame = pd.DataFrame(figs.rows)
    n_fail = int((~frame["pass"]).sum())
    checks.add("figures: Arial, >= 7 pt, no text overlap or text outside, exact width, TrueType in PDF",
               f"{len(frame) - n_fail}/{len(frame)} pass" + (f"; failed: {';'.join(frame.loc[~frame['pass'], 'figure'])}"
                                                             if n_fail else ""), n_fail == 0)
    summ, zones, odds = pd.DataFrame(summary), pd.DataFrame(zone_rows), pd.DataFrame(odds_rows)
    run.table(summ, "unresolved_summary")
    run.table(bym, "unresolved_by_markers")
    run.table(zones, "unresolved_axis_zones")
    run.table(odds, "unresolved_odds")
    run.table(pd.concat(cells, ignore_index=True), "unresolved_axis_per_cell")
    run.text(response_text(summ, bym, zones, odds), "unresolved_response_text")
    run.table(pd.concat([checks.frame(), frame], ignore_index=True), "unresolved_checks")
    run.text("\n".join(captions), "unresolved_figure_captions")
    run.finish({"request": REQUEST, "label_not_resolved": lib.UNRESOLVED, "zones": list(ZONES),
                "high_markers": list(HIGH_MARKERS), "runs": {k: vars(v) for k, v in lib.RUNS.items()},
                "seed": lib.SEED, "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_unresolved_checks.csv.")


if __name__ == "__main__":
    main()
