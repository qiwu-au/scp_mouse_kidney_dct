#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S-05-c, resolution scans of the kidney and the DCT data.

Drawn at the authors' request (12 Nov 2026: "Yes, redraw
them"; the response to the reviewers, section 4 re-render checklist and section 5). It assembles, as one reader-facing file, the three
analysis figures results/figures/kidney_annotation/kidney_annotation_rankfree_summary and
results/figures/dct_labels/dct_labels_scan_summary_{ctrl,zerok} (internal names, lower-case panel letters,
footers); those files are not edited. No new statistics: every plotted number is read from the stored kidney_annotation and dct_labels
scan tables.

Changed on 12 Nov 2026 at the authors' request, relayed by the coordinator ("in Supplementary Figure S-05-c, delete
submitted graph and submitted re-annotation. The readers will not know the revision process, so we only show the
final products."; asked about the three lines of panels A-C, the final choice was "Keep all three, rename blue"):
panels A-C keep the three graphs with their colours; the blue series is named "graph of the main analysis" (was
"submitted graph"); the dashed reference line "submitted re-annotation: 14 types" is not drawn; the shading is named
"2 and lower"; no drawn text names the submission or the revision process. The number of the removed line stays in
the checks file as an information row marked "not drawn".

One figure, 180 mm wide, seven panels in three rows:
  Total kidney (712 cells; marker list without the endothelial row; three graphs; Leiden 0.5-6.0)
    A  cell types resolved; B  cell types with a type-specific marker; C  fraction of cells labelled Unknown.
       Solid line, filled symbols: with the added step (seed 42; bars = range over the 11 Leiden seeds);
       dashed line, open symbols: top-25 rule only. Shaded: resolution 2 and lower.
  DCT, Control (729 cells) and DCT, Zero-K (702 cells), one row each, each on its own graph (Leiden 0.3-3.0);
  nothing compares the conditions
    D  clusters (seed 42; bar = range over the 11 seeds); E  cells per label; F  adjusted Rand index of the labels
       of seeds 0-9 against seed 42 (point = median, bar = range); G  cells per confidence class of their cluster.
       E, G: left bar (T) = top-25 rule only, right bar (T+) = with the added step; F: open / filled point.
  A triangle marks the resolution used (kidney 2, DCT 1).
Names: rules "top-25 rule only" / "with the added step";
  graphs "graph of the main analysis" (Supplementary Table S-05-b: "submitted graph"), "depth-regressed graph",
  "unscaled, 500 most variable proteins"; DCT label value
  'Unknown' shown as "not resolved"; confidence classes "type-specific marker, any rank", "shared markers only",
  "no marker support (subtype not resolved)"; the kidney 'Unknown' keeps its name. The values file keeps the stored
  names next to the display names. No figure title, no footer.
Format: Arial, every text >= 7 pt, exact 180 mm width, PDF (TrueType) + 600-dpi PNG + LZW TIFF, checked with
  dct_reviewer_lib.Figures (dotplot / dct_labels text and file checks).

Inputs (read-only): results/tables/kidney_annotation/kidney_annotation_rankfree_summary.csv,
  kidney_annotation_rankfree_seed_stability.csv, kidney_annotation_v3endo_summary.csv, kidney_annotation_cluster_labels_per_cell_a_endo.csv
  (number of cells only); results/tables/dct_labels/dct_labels_scan_summary.csv,
  dct_labels_scan_seed_stability.csv.
Outputs: results/figures/report/report_05_S05c_resolution_scans.{pdf,png,tif};
  results/tables/report/report_05_s05c_figure_{values,checks}.csv;
  results/tables/report/report_step_05_s05c_figure_manifest.json

Command (project root):  python scripts/report_resolution_scan_figure.py
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
import qc_mito_lib as t3
import cluster_de_lib as t5
import dotplot_dotplots as t6d            # text_mm (Arial); sets the dotplot style on import
import kidney_annotation_scan_lib as s15            # graph names, resolutions and seeds of the kidney scan
import kidney_annotation_scan_figures as sf         # graph colours and markers of the kidney_annotation figures
import labels_core_lib as t18                 # DCT scan resolutions, label and series colours of the dct_labels figures
import dct_labels_figures as t27f            # drawn_texts
import dct_reviewer_lib as lib

lib.style()
from matplotlib.patches import Rectangle  # noqa: E402

TASK = "report"
STEP = "05_s05c_figure"
FIGURE = "05_S05c_resolution_scans"
KIDNEY_ANNOTATION_DIR = cc.OUT_ROOTS["tables"] / "kidney_annotation"
DCT_LABELS_DIR = cc.OUT_ROOTS["tables"] / "dct_labels"
K_SUMMARY = KIDNEY_ANNOTATION_DIR / "kidney_annotation_rankfree_summary.csv"
K_SEEDS = KIDNEY_ANNOTATION_DIR / "kidney_annotation_rankfree_seed_stability.csv"
K_ENDO = KIDNEY_ANNOTATION_DIR / "kidney_annotation_v3endo_summary.csv"
K_PER_CELL = KIDNEY_ANNOTATION_DIR / "kidney_annotation_cluster_labels_per_cell_a_endo.csv"
D_SUMMARY = DCT_LABELS_DIR / "dct_labels_scan_summary.csv"
D_SEEDS = DCT_LABELS_DIR / "dct_labels_scan_seed_stability.csv"

# --- display names --------------------------------
RULES = ("top25", "rankfree")
RULE_DISPLAY = {"top25": "top-25 rule only", "rankfree": "with the added step"}
RULE_CODE = {"top25": "T", "rankfree": "T+"}
MAIN_GRAPH = "stored"                                             # the graph the annotation of Figure 2A is made on
GRAPH_DISPLAY = {"stored": "graph of the main analysis", "regressed": "depth-regressed graph",
                 "t04_total_unscaled_hvg500": "unscaled, 500 most variable proteins"}
NOT_DRAWN = "not drawn (removed at the authors' request, 12 Nov 2026)"
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
LABELS = ("DCT1", "DCT2", "ProLIF", "Unknown")                    # stacking order of the dct_labels figure
LABEL_DISPLAY = {"Unknown": "not resolved"}                       # DCT only (D9); the kidney Unknown keeps its name
CONFIDENCES = ("high", "medium", "specific_any_rank", "low_shared_only", "unknown_no_marker_in_top_de")
CONF_DISPLAY = {"high": "high", "medium": "medium", "specific_any_rank": "type-specific marker, any rank",
                "low_shared_only": "shared markers only",
                "unknown_no_marker_in_top_de": "no marker support (subtype not resolved)"}
CONF_COLORS = {"high": t18.SERIES[2], "medium": t18.SERIES[3], "specific_any_rank": t18.SERIES[4],
               "low_shared_only": t18.SERIES[7], "unknown_no_marker_in_top_de": t18.LABEL_COLORS["Unknown"]}
SHADE_TEXT = "2 and lower"
N_REANNOT_TYPES = 14                                              # the project notes: kidney_reannot_label; line not drawn

# --- expected values (the project notes; the response to the reviewers, section 2 letter, its sources and details tables, section 7) ---------
N_KIDNEY = 712
N_DCT = lib.N_CELLS                                               # Ctrl 729, ZeroK 702
ADOPTED = {"kidney": 2.0, "dct": 1.0}
EXPECTED_ADOPTED_LABELS = {"Ctrl": {"DCT1": 382, "DCT2": 85, "ProLIF": 0, "Unknown": 262},
                           "ZeroK": {"DCT1": 449, "DCT2": 0, "ProLIF": 0, "Unknown": 253}}
EXPECTED_ADOPTED_CONF = {"Ctrl": {"high": 10, "medium": 0, "specific_any_rank": 362, "low_shared_only": 95,
                                  "unknown_no_marker_in_top_de": 262},
                         "ZeroK": {"high": 0, "medium": 0, "specific_any_rank": 354, "low_shared_only": 95,
                                   "unknown_no_marker_in_top_de": 253}}

# --- geometry (mm; y downwards) ------------------------------------------------------------------------------------
W = lib.FULL_MM
M = 2.0
FONT, HEAD_PT, LETTER_PT, ROW_PT = 7.0, 7.5, 8.5, 7.5
PT = t6d.PT                                                       # points per mm
ROW_X = 3.4                                                       # centre of the rotated row names
HEAD = 5.5                                                        # letter and heading line above a panel row
K_X, K_W, K_H = (16.0, 73.0, 131.0), 46.0, 38.0
D_X, D_W, D_H = (16.0, 51.0, 102.0, 137.0), (23.0, 39.0, 23.0, 39.0), 33.0
KEY_ROW = 3.7
K_TOP = M + HEAD
K_KEY = K_TOP + K_H + 9.6                                         # centre of the first kidney key row
D_TOP = {"Ctrl": K_KEY + 2 * KEY_ROW + 1.9 + 3.0 + HEAD}
D_TOP["ZeroK"] = D_TOP["Ctrl"] + D_H + 10.0 + 3.0
D_KEY = D_TOP["ZeroK"] + D_H + 12.4
H = D_KEY + 2 * KEY_ROW + 1.9 + M
BAR_W, SHIFT = 0.40, 0.22                                         # bars of E and G (in resolution steps)
DOT_SHIFT = 0.16                                                  # points of F
INK = t3.INK
SHADE = dict(color=INK["grid"], alpha=0.45, linewidth=0)


def new_axes(fig, x, y, w, h):
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])


def tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2, pad=1.5, labelsize=FONT)
    lib.grid_y(ax)


def used_mark(ax, x: float) -> None:
    """Small triangle at the top of the axes: the resolution used."""
    ax.plot([x], [1 - 1.6 / (ax.get_position().height * H)], marker="v", markersize=3.6, color=INK["primary"],
            markeredgewidth=0, linestyle="none", transform=ax.get_xaxis_transform(), clip_on=False, zorder=6)


class Overlay:
    """Figure-wide axes in mm (y downwards) for letters, headings, row names and the keys."""

    def __init__(self, fig):
        self.ax = fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ax.set_axis_off()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.set_autoscale_on(False)

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FONT)
        kw.setdefault("color", INK["primary"])
        kw.setdefault("ha", "left")
        kw.setdefault("va", "center_baseline")
        return self.ax.text(x, y, s, clip_on=False, **kw)

    def handle(self, x, y, kind, spec) -> float:
        """Draw one key handle with its left edge at x; return its width in mm."""
        if kind == "patch":
            self.ax.add_patch(Rectangle((x, y - 1.1), 3.0, 2.2, clip_on=False, **spec))
            return 3.0
        if kind == "line":
            self.ax.plot([x, x + 7.0], [y, y], clip_on=False, **{k: v for k, v in spec.items() if k != "marker"})
            if spec.get("marker"):
                self.ax.plot([x + 3.5], [y], clip_on=False, linestyle="none", **{k: v for k, v in spec.items()
                                                                                 if k not in ("linestyle", "linewidth")})
            return 7.0
        if kind == "marker":
            self.ax.plot([x + 1.5], [y], clip_on=False, linestyle="none", **spec)
            return 3.0
        if kind == "range":
            self.ax.plot([x + 1.0, x + 1.0], [y - 1.3, y + 1.3], clip_on=False, **spec)
            return 2.0
        raise ValueError(kind)

    def key_row(self, y, x, parts, name_w: float, gap: float = 4.0) -> float:
        """One key row: ('name', text) starts a group; (kind, spec, text) is an entry. Returns the right edge."""
        for part in parts:
            if part[0] == "name":
                self.text(x, y, part[1], color=INK["secondary"])
                x += name_w + 2.0
                continue
            kind, spec, label = part
            x += self.handle(x, y, kind, spec) + 1.0
            self.text(x, y, label)
            x += t6d.text_mm(label)[0] + gap
        return x - gap


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-05-c.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks = t5.Checks("kidney_and_dct_combined")

    # =================================================================================================================
    # Stored tables and the numbers the letter quotes
    # =================================================================================================================
    k = pd.read_csv(run.use_input(K_SUMMARY, "kidney_annotation kidney scan: top-25 rule and with the added step"))
    kseed = pd.read_csv(run.use_input(K_SEEDS, "kidney_annotation kidney scan, every Leiden seed"))
    kendo = pd.read_csv(run.use_input(K_ENDO, "kidney_annotation kidney, resolution 2 with the endothelial row"))
    n_kidney = len(pd.read_csv(run.use_input(K_PER_CELL, "kidney_annotation per-cell labels (number of kidney cells)"),
                               usecols=["cell_id"]))
    d = pd.read_csv(run.use_input(D_SUMMARY, "dct_labels DCT scan summary (combined search)"))
    dseed = pd.read_csv(run.use_input(D_SEEDS, "dct_labels DCT scan, every Leiden seed"))
    graphs, k_res, d_res = list(s15.GRAPH_NAMES), list(s15.RESOLUTIONS), list(t18.SCAN_RESOLUTIONS)
    conds = list(lib.CONDITIONS)

    ok = (len(k) == len(graphs) * len(k_res) and set(k["graph"]) == set(graphs) and n_kidney == N_KIDNEY
          and all(sorted(k.loc[k["graph"] == g, "resolution"]) == k_res for g in graphs))
    checks.add("kidney scan table: 3 graphs x 12 resolutions (0.5-6.0); 712 cells",
               f"{len(k)} rows; graphs {', '.join(graphs)}; resolutions {k_res[0]}-{k_res[-1]}; {n_kidney} cells", ok)
    checks.add("kidney, A-C: all three graphs are drawn; the blue series is named 'graph of the main analysis' (authors, "
               "12 Nov 2026)", "; ".join(f"{g} -> {GRAPH_DISPLAY[g]}" for g in graphs),
               list(GRAPH_DISPLAY) == graphs and graphs[0] == MAIN_GRAPH
               and GRAPH_DISPLAY[MAIN_GRAPH] == "graph of the main analysis")
    ok = (len(d) == len(conds) * len(d_res) * len(RULES)
          and all(sorted(d.loc[(d["condition"] == c) & (d["rule"] == r), "resolution"]) == d_res
                  for c in conds for r in RULES))
    checks.add("DCT scan table: 2 conditions x 5 resolutions (0.3, 0.5, 0.8, 1, 3) x 2 rules",
               f"{len(d)} rows; conditions {', '.join(conds)}; resolutions {', '.join(f'{r:g}' for r in d_res)}", ok)
    same = sf.GRAPH_COLORS == dict(zip(graphs, t3.SERIES[:3])) and sf.GRAPH_MARKERS == dict(zip(graphs, "os^"))
    checks.add("colours: graphs as the kidney_annotation figures; DCT labels and confidence classes as the dct_labels figures",
               f"graphs {[sf.GRAPH_COLORS[g] for g in graphs]}; labels {[t18.LABEL_COLORS[x] for x in LABELS]}; confidence "
               f"{[CONF_COLORS[x] for x in CONFIDENCES]}; equal {same}", same)

    # kidney: seed ranges of the summary against the per-seed table
    seeds_k = sorted(kseed["seed"].unique())
    agg = kseed.groupby(["graph", "resolution"]).agg(
        a_min=("rankfree_n_types", "min"), a_max=("rankfree_n_types", "max"),
        b_min=("rankfree_n_types_specific", "min"), b_max=("rankfree_n_types_specific", "max"), n=("seed", "size"))
    kk = k.set_index(["graph", "resolution"])
    j = kk.join(agg)
    s42 = kseed[kseed["seed"] == 42].set_index(["graph", "resolution"]).reindex(kk.index)
    ok = (bool((j["n"] == 11).all()) and seeds_k == sorted(s15.SEEDS)
          and bool((j["a_min"] == j["rankfree_n_types_all_seeds_min"]).all())
          and bool((j["a_max"] == j["rankfree_n_types_all_seeds_max"]).all())
          and bool((j["b_min"] == j["rankfree_n_types_specific_all_seeds_min"]).all())
          and bool((j["b_max"] == j["rankfree_n_types_specific_all_seeds_max"]).all())
          and all(bool((s42[c] == kk[c]).all()) for c in ("rankfree_n_types", "rankfree_n_types_specific",
                                                          "top25_n_types", "top25_n_types_specific"))
          and all(float(np.max(np.abs(s42[c] - kk[c]))) < 1e-12 for c in ("rankfree_fraction_unknown",
                                                                         "top25_fraction_unknown")))
    checks.add("kidney: plotted lines = seed 42; bars = minimum and maximum over the 11 Leiden seeds (42, 0-9) of the "
               "per-seed table", f"{len(j)} graph x resolution rows; 11 seeds each {bool((j['n'] == 11).all())}; "
               f"equal {ok}", ok and len(j) == len(graphs) * len(k_res))

    st = k[k["graph"] == MAIN_GRAPH].set_index("resolution")
    lo, hi = 100 * st["top25_fraction_unknown"].min(), 100 * st["top25_fraction_unknown"].max()
    checks.add("letter: top-25 rule only, kidney, graph of the main analysis: cells Unknown at 0.5-6.0",
               f"{lo:.1f}-{hi:.1f}%", (round(lo), round(hi)) == (38, 54), "38-54%")
    got = [int(st.loc[r, "rankfree_n_types"]) for r in (0.5, 1.0, 1.5, 2.0)]
    checks.add("letter: with the added step, marker list without the endothelial row, graph of the main analysis: "
               "cell types at 0.5, 1.0, 1.5, 2.0", ", ".join(map(str, got)), got == [4, 6, 6, 7], "4, 6, 6, 7")
    got = [round(100 * float(st.loc[r, "rankfree_fraction_unknown"])) for r in (0.5, 1.0, 1.5, 2.0)]
    checks.add("the response to the reviewers details: kidney Unknown with the added step at 0.5, 1.0, 1.5, 2.0 (graph of the main "
               "analysis)", ", ".join(f"{v}%" for v in got), got == [7, 8, 12, 18], "7%, 8%, 12%, 18%")
    # The dashed reference line of A and B is no longer drawn: its number is kept as an information row.
    checks.add(f"{NOT_DRAWN}: reference line of A and B, number of cell types of the Figure 2B labels of the first "
               "version (kidney_reannot_label)", f"{N_REANNOT_TYPES} (the project notes)", None)
    got = {g: (int(k.loc[k["graph"] == g, "top25_n_types"].min()), int(k.loc[k["graph"] == g, "top25_n_types"].max()))
           for g in graphs[1:]}
    checks.add("the response to the reviewers details: cell types with the top-25 rule on the depth-regressed and the unscaled graph",
               "; ".join(f"{GRAPH_DISPLAY[g]} {a}-{b}" for g, (a, b) in got.items()),
               got == {"regressed": (3, 7), "t04_total_unscaled_hvg500": (2, 8)}, "3-7; 2-8")
    e = kendo[(kendo["graph"] == MAIN_GRAPH) & (kendo["resolution"] == 2.0) & (kendo["rule"] == "rankfree")]
    e = e.set_index("marker_set")
    ok = (int(e.loc["v3", "n_types"]) == int(st.loc[2.0, "rankfree_n_types"]) == 7
          and abs(float(e.loc["v3", "fraction_unknown"]) - float(st.loc[2.0, "rankfree_fraction_unknown"])) < 1e-12
          and (int(e.loc["v3_endo", "n_types"]), int(e.loc["v3_endo", "n_types_all_seeds_min"]),
               int(e.loc["v3_endo", "n_types_all_seeds_max"])) == (8, 8, 8)
          and round(float(e.loc["v3_endo", "fraction_unknown"]) * N_KIDNEY) == 109)
    checks.add("legend statement (value not drawn): the scan uses the marker list without the endothelial row; at resolution "
               "2 on the graph of the main analysis (added step) the list with this row gives 8 cell types in all 11 "
               "seeds and 109 of 712 cells Unknown (= Figure 2A)",
               f"without the row {int(e.loc['v3', 'n_types'])} types (= scan table); with the endothelial row "
               f"{int(e.loc['v3_endo', 'n_types'])} types (seeds {int(e.loc['v3_endo', 'n_types_all_seeds_min'])}-"
               f"{int(e.loc['v3_endo', 'n_types_all_seeds_max'])}), Unknown "
               f"{100 * float(e.loc['v3_endo', 'fraction_unknown']):.1f}%", ok)

    # DCT
    dd = d.set_index(["condition", "rule", "resolution"]).sort_index()
    n_of = {c: int(dd.loc[(c, "rankfree", 1.0), [f"n_{x}" for x in LABELS]].sum()) for c in conds}
    lab_sum = d[[f"n_{x}" for x in LABELS]].sum(axis=1)
    conf_sum = d[[f"n_cells_conf_{x}" for x in CONFIDENCES]].sum(axis=1)
    n_row = d["condition"].map(N_DCT)
    none_small = bool((d[["n_below_min_size", "n_cells_conf_below_min_size", "n_clusters_below_min"]] == 0).all().all())
    checks.add("DCT: cells per condition; every bar sums to all cells (labels and confidence classes); no cluster "
               "below 5 cells in any clustering (class not drawn)",
               f"Ctrl {n_of['Ctrl']}; ZeroK {n_of['ZeroK']}; label sums equal {bool((lab_sum == n_row).all())}; "
               f"confidence sums equal {bool((conf_sum == n_row).all())}; no cluster below 5 cells {none_small}",
               n_of == N_DCT and bool((lab_sum == n_row).all()) and bool((conf_sum == n_row).all()) and none_small)
    for c in conds:
        got = {x: int(dd.loc[(c, "rankfree", 1.0), f"n_{x}"]) for x in LABELS}
        conf = {x: int(dd.loc[(c, "rankfree", 1.0), f"n_cells_conf_{x}"]) for x in CONFIDENCES}
        used = d[d["frozen_primary"] & (d["condition"] == c)]
        ok = (got == EXPECTED_ADOPTED_LABELS[c] and conf == EXPECTED_ADOPTED_CONF[c] and len(used) == 1
              and used["rule"].iloc[0] == "rankfree" and float(used["resolution"].iloc[0]) == 1.0
              and int(used["n_clusters"].iloc[0]) == 13)
        checks.add(f"{COND_DISPLAY[c]}: the setting used (resolution 1, with the added step) gives the canonical counts "
                   "(the project notes), 13 clusters",
                   "; ".join(f"{LABEL_DISPLAY.get(x, x)} {v}" for x, v in got.items() if v) + " | "
                   + "; ".join(f"{CONF_DISPLAY[x]} {v}" for x, v in conf.items() if v)
                   + f" | clusters {int(used['n_clusters'].iloc[0])}", ok)
    unk = dd.loc[("Ctrl", "top25"), "n_Unknown"]
    checks.add("letter: top-25 rule only, control: cells with subtype not resolved at 0.3-3.0",
               f"{int(unk.min())}-{int(unk.max())} of {N_DCT['Ctrl']}", (int(unk.min()), int(unk.max())) == (457, 624),
               "457-624 of 729")
    pro = {f"{r:g}": int(dd.loc[("Ctrl", "rankfree", r), "n_ProLIF"]) for r in d_res}
    types = {f"{r:g}": [x for x in LABELS[:3] if dd.loc[("Ctrl", "rankfree", r), f"n_{x}"] > 0] for r in d_res}
    checks.add("letter: with the added step, control: ProLIF assigned only at resolution 3 (10 cells); DCT1 only at "
               "0.3; DCT1 and DCT2 at 0.5-1",
               "; ".join(f"{r}: {'+'.join(types[r])}" + (f" (ProLIF {pro[r]})" if pro[r] else "") for r in pro),
               pro == {"0.3": 0, "0.5": 0, "0.8": 0, "1": 0, "3": 10} and types["0.3"] == ["DCT1"]
               and all(types[r] == ["DCT1", "DCT2"] for r in ("0.5", "0.8", "1")))
    got = {c: (round(float(dd.loc[(c, "rankfree", 1.0), "ari_labels_vs_seed42_median"]), 3),
               round(float(dd.loc[(c, "rankfree", 1.0), "ari_labels_vs_seed42_min"]), 3),
               round(float(dd.loc[(c, "rankfree", 3.0), "ari_labels_vs_seed42_median"]), 3)) for c in conds}
    checks.add("the response to the reviewers details and section 7: label ARI over seeds, with the added step: control median 0.922 "
               "(min 0.638) at 1 and 0.778 at 3; zero-K 0.778 at 1 and 0.611 at 3",
               "; ".join(f"{COND_DISPLAY[c]} {v[0]} (min {v[1]}) at 1, {v[2]} at 3" for c, v in got.items()),
               got == {"Ctrl": (0.922, 0.638, 0.778), "ZeroK": (0.778, 0.446, 0.611)})
    got = (int(dd.loc[("ZeroK", "rankfree", 3.0), "n_ProLIF"]),
           [int(dd.loc[("Ctrl", "top25", r), "n_ProLIF"]) for r in (0.5, 0.8, 3.0)],
           [int(dd.loc[("ZeroK", "rankfree", r), "n_DCT2"]) for r in d_res])
    checks.add("the project notes / the response to the reviewers section 7: zero-K ProLIF at 3 with the added step; control ProLIF with the "
               "top-25 rule at 0.5, 0.8, 3; zero-K DCT2 with the added step by resolution",
               f"{got[0]}; {got[1]}; {got[2]}", got == (46, [145, 138, 10], [155, 133, 133, 0, 125]))
    # DCT: seed columns of the summary against the per-seed table
    g_all = dseed.groupby(["condition", "rule", "resolution"])
    g_09 = dseed[dseed["seed"] != 42].groupby(["condition", "rule", "resolution"])
    s42d = dseed[dseed["seed"] == 42].set_index(["condition", "rule", "resolution"]).reindex(dd.index)
    ok = (bool((g_all.size().reindex(dd.index) == 11).all()) and sorted(dseed["seed"].unique()) == sorted(t18.SEEDS)
          and bool((g_all["n_clusters"].min().reindex(dd.index) == dd["n_clusters_all_seeds_min"]).all())
          and bool((g_all["n_clusters"].max().reindex(dd.index) == dd["n_clusters_all_seeds_max"]).all())
          and bool((s42d["n_clusters"] == dd["n_clusters"]).all())
          and all(float(np.max(np.abs(getattr(g_09["ari_labels_vs_seed42"], f)().reindex(dd.index)
                                      - dd[f"ari_labels_vs_seed42_{f}"]))) < 1e-12 for f in ("median", "min", "max")))
    checks.add("DCT: clusters = seed 42 with the range over the 11 Leiden seeds (42, 0-9); label ARI = median and "
               "range of seeds 0-9 against seed 42 (per-seed table)", f"{len(dd)} rows; equal {ok}", ok)

    # =================================================================================================================
    # Figure
    # =================================================================================================================
    fig = lib.new_figure(W, H)
    art, values = {}, []
    gc, gm = sf.GRAPH_COLORS, sf.GRAPH_MARKERS

    # --- kidney: A, B, C ---------------------------------------------------------------------------------------------
    k_panels = (("A", "n_types", "Cell types resolved", "number of cell types"),
                ("B", "n_types_specific", "Cell types with a type-specific marker", "number of cell types"),
                ("C", "fraction_unknown", "Cells labelled Unknown", "fraction of cells"))
    heads = {}
    for (letter, key, head, ylab), x in zip(k_panels, K_X):
        ax = new_axes(fig, x, K_TOP, K_W, K_H)
        ax.axvspan(0.2, ADOPTED["kidney"] + 0.25, zorder=0, **SHADE)
        for i, g in enumerate(graphs):
            sub = k[k["graph"] == g].sort_values("resolution")
            xr = sub["resolution"].to_numpy(float) + (i - 1) * 0.07
            for rule in RULES:
                col = f"{rule}_{key}"
                solid = rule == "rankfree"
                lo_c, hi_c = f"{col}_all_seeds_min", f"{col}_all_seeds_max"
                ranged = solid and key != "fraction_unknown"
                if ranged:
                    art[(letter, g, "range")] = (ax.vlines(xr, sub[lo_c], sub[hi_c], color=gc[g], linewidth=0.7,
                                                           alpha=0.6, zorder=2), sub[[lo_c, hi_c]].to_numpy(float))
                (line,) = ax.plot(xr, sub[col], color=gc[g], linewidth=1.2 if solid else 1.0,
                                  linestyle="-" if solid else (0, (3, 1.5)), marker=gm[g], markersize=2.8,
                                  markerfacecolor=gc[g] if solid else "white", markeredgecolor=gc[g],
                                  markeredgewidth=0.6, zorder=4 if solid else 3)
                art[(letter, g, rule)] = (line, sub[col].to_numpy(float))
                for r in sub.itertuples(index=False):
                    values.append({"panel": letter, "data": "Total kidney", "condition": "", "n_cells": N_KIDNEY,
                                   "graph": GRAPH_DISPLAY[g], "graph_stored": g, "rule": RULE_DISPLAY[rule],
                                   "rule_stored": rule, "resolution": r.resolution, "quantity": head,
                                   "category": "", "category_stored": "", "value": getattr(r, col),
                                   "range_low": getattr(r, lo_c) if ranged else np.nan,
                                   "range_high": getattr(r, hi_c) if ranged else np.nan,
                                   "value_is": "seed 42" + ("; range over the 11 Leiden seeds" if ranged else ""),
                                   "source_table": K_SUMMARY.name, "source_column": col})
        ax.set_xlim(0.2, 6.3)
        ax.set_xticks(k_res[1::2], [f"{r:g}" for r in k_res[1::2]])
        ax.set_xlabel("Leiden resolution", labelpad=2)
        ax.set_ylabel(ylab, labelpad=2)
        if key == "fraction_unknown":
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 6))
        else:
            ax.set_ylim(0, 11)
            ax.set_yticks(range(0, 11, 2))
        tidy(ax)
        used_mark(ax, ADOPTED["kidney"])
        heads[letter] = (x, K_TOP, head)

    # --- DCT: D, E, F, G per condition ---------------------------------------------------------------------------------
    pos = np.arange(len(d_res))
    i_used = d_res.index(ADOPTED["dct"])
    d_heads = (("D", "Clusters"), ("E", "Cells per label"), ("F", "Label stability"), ("G", "Confidence of the label"))
    rule_shift = {"top25": -1.0, "rankfree": 1.0}
    for cond in conds:
        n = N_DCT[cond]
        base = {"data": "DCT", "condition": COND_DISPLAY[cond], "n_cells": n, "graph": "", "graph_stored": "",
                "source_table": D_SUMMARY.name}
        axes = [new_axes(fig, x, D_TOP[cond], w, D_H) for x, w in zip(D_X, D_W)]
        # D: clusters
        ax = axes[0]
        sub = dd.loc[(cond, "top25")].reindex(d_res)
        same = bool((sub["n_clusters"] == dd.loc[(cond, "rankfree")].reindex(d_res)["n_clusters"]).all())
        checks.add(f"{COND_DISPLAY[cond]}: the clusters are the same for both rules (one series in D)", f"{same}", same)
        rng = sub[["n_clusters_all_seeds_min", "n_clusters_all_seeds_max"]].to_numpy(float)
        art[("D", cond, "range")] = (ax.vlines(pos, rng[:, 0], rng[:, 1], color=INK["muted"], linewidth=1.0), rng)
        (line,) = ax.plot(pos, sub["n_clusters"], "o", color=INK["primary"], markersize=3.2, markeredgewidth=0)
        art[("D", cond, "seed42")] = (line, sub["n_clusters"].to_numpy(float))
        ax.set_ylim(0, 35)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_ylabel("number of clusters", labelpad=2)
        for r, v, (a, b) in zip(d_res, sub["n_clusters"], rng):
            values.append({**base, "panel": "D", "rule": "", "rule_stored": "both", "resolution": r,
                           "quantity": "Clusters", "category": "", "category_stored": "", "value": v, "range_low": a,
                           "range_high": b, "value_is": "seed 42; range over the 11 Leiden seeds",
                           "source_column": "n_clusters"})
        # E and G: stacked bars, two rules per resolution
        for ax, letter, cats, disp, colors, stem, quantity in (
                (axes[1], "E", LABELS, LABEL_DISPLAY, t18.LABEL_COLORS, "n_", "Cells per label"),
                (axes[3], "G", CONFIDENCES, CONF_DISPLAY, CONF_COLORS, "n_cells_conf_", "Confidence of the label")):
            for rule in RULES:
                sub = dd.loc[(cond, rule)].reindex(d_res)
                bottom = np.zeros(len(d_res))
                for cat in cats:
                    cells = sub[f"{stem}{cat}"].to_numpy(float)
                    bars = ax.bar(pos + rule_shift[rule] * SHIFT, cells / n, BAR_W, bottom=bottom,
                                  color=colors[cat], edgecolor="white", linewidth=0.3)
                    art[(letter, cond, rule, cat)] = (bars, cells, bottom.copy() * n)
                    bottom += cells / n
                    for r, v in zip(d_res, cells):
                        values.append({**base, "panel": letter, "rule": RULE_DISPLAY[rule], "rule_stored": rule,
                                       "resolution": r, "quantity": quantity, "category": disp.get(cat, cat),
                                       "category_stored": cat, "value": v / n, "n_cells_in_category": int(v),
                                       "range_low": np.nan, "range_high": np.nan,
                                       "value_is": "fraction of cells, seed 42", "source_column": f"{stem}{cat}"})
                for xi in pos:
                    ax.text(xi + rule_shift[rule] * SHIFT, -1.0 / D_H, RULE_CODE[rule], ha="center", va="top",
                            fontsize=FONT, color=INK["secondary"], transform=ax.get_xaxis_transform(), clip_on=False)
            ax.set_ylim(0, 1.13)
            ax.set_yticks([0, 0.5, 1])
            ax.set_ylabel("fraction of cells", labelpad=2)
        # F: label stability over Leiden seeds
        ax = axes[2]
        for rule in RULES:
            sub = dd.loc[(cond, rule)].reindex(d_res)
            solid = rule == "rankfree"
            color = INK["primary"] if solid else INK["muted"]
            xs = pos + rule_shift[rule] * DOT_SHIFT
            rng = sub[["ari_labels_vs_seed42_min", "ari_labels_vs_seed42_max"]].to_numpy(float)
            art[("F", cond, rule, "range")] = (ax.vlines(xs, rng[:, 0], rng[:, 1], color=color, linewidth=1.0), rng)
            (line,) = ax.plot(xs, sub["ari_labels_vs_seed42_median"], "o", color=color, markersize=3.2,
                              markerfacecolor=color if solid else "white", markeredgecolor=color, markeredgewidth=0.7,
                              zorder=3)
            art[("F", cond, rule)] = (line, sub["ari_labels_vs_seed42_median"].to_numpy(float))
            for r, v, (a, b) in zip(d_res, sub["ari_labels_vs_seed42_median"], rng):
                values.append({**base, "panel": "F", "rule": RULE_DISPLAY[rule], "rule_stored": rule, "resolution": r,
                               "quantity": "Label stability", "category": "", "category_stored": "", "value": v,
                               "range_low": a, "range_high": b,
                               "value_is": "adjusted Rand index of the labels, seeds 0-9 against seed 42: median; "
                                           "range", "source_column": "ari_labels_vs_seed42_median"})
        ax.set_ylim(-0.2, 1.15)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel("ARI, seeds 0–9 vs seed 42", labelpad=2)
        for ax, (letter, head), x in zip(axes, d_heads, D_X):
            ax.set_xlim(-0.6, len(d_res) - 0.4)
            ax.set_xticks(pos, [f"{r:g}" for r in d_res])
            ax.set_xlabel("Leiden resolution", labelpad=2)
            tidy(ax)
            if letter in "EG":
                ax.tick_params(axis="x", length=0, pad=(1.0 + t6d.LINE_H + 0.7) * PT)
            used_mark(ax, i_used)
            if cond == conds[0]:
                heads[letter] = (x, D_TOP[cond], head)

    # --- letters, headings, row names, keys --------------------------------------------------------------------------
    ov = Overlay(fig)
    for letter, (x, top, head) in heads.items():
        ov.text(x - 5.0, top - 1.7, letter, fontsize=LETTER_PT, fontweight="bold", va="baseline")
        ov.text(x, top - 1.7, head, fontsize=HEAD_PT, va="baseline")
    row_names = {"kidney": (K_TOP + K_H / 2, f"Total kidney ({N_KIDNEY} cells)")}
    row_names.update({c: (D_TOP[c] + D_H / 2, f"DCT, {COND_DISPLAY[c]} ({N_DCT[c]} cells)") for c in conds})
    for y, name in row_names.values():
        ov.text(ROW_X, y, name, rotation=90, ha="center", va="center", fontsize=ROW_PT, fontweight="bold")
    name_w = max(t6d.text_mm(s)[0] for s in ("Graph", "Rule", "Resolution", "Label", "Confidence"))
    grey = INK["secondary"]
    tri = dict(marker="v", markersize=3.6, color=INK["primary"], markeredgewidth=0)
    right = [
        ov.key_row(K_KEY, M, [("name", "Graph")] + [
            ("line", dict(color=gc[g], linewidth=1.2, marker=gm[g], markersize=2.8, markeredgewidth=0),
             GRAPH_DISPLAY[g]) for g in graphs], name_w),
        ov.key_row(K_KEY + KEY_ROW, M, [
            ("name", "Rule"),
            ("line", dict(color=grey, linewidth=1.2, linestyle="-", marker="o", markersize=2.8, markerfacecolor=grey,
                          markeredgecolor=grey, markeredgewidth=0.6), f"{RULE_DISPLAY['rankfree']} (adopted rule)"),
            ("line", dict(color=grey, linewidth=1.0, linestyle=(0, (3, 1.5)), marker="o", markersize=2.8,
                          markerfacecolor="white", markeredgecolor=grey, markeredgewidth=0.6), RULE_DISPLAY["top25"]),
            ("range", dict(color=grey, linewidth=0.9), "range over 11 Leiden seeds")], name_w),
        ov.key_row(K_KEY + 2 * KEY_ROW, M, [
            ("name", "Resolution"), ("patch", dict(facecolor=INK["grid"], alpha=0.45, edgecolor="none"),
                                     SHADE_TEXT), ("marker", tri, "used")], name_w),
        ov.key_row(D_KEY, M, [("name", "Label")] + [
            ("patch", dict(facecolor=t18.LABEL_COLORS[x], edgecolor="none"), LABEL_DISPLAY.get(x, x)) for x in LABELS],
            name_w),
        ov.key_row(D_KEY + KEY_ROW, M, [("name", "Confidence")] + [
            ("patch", dict(facecolor=CONF_COLORS[x], edgecolor="none"), CONF_DISPLAY[x]) for x in CONFIDENCES],
            name_w, gap=3.2),
    ]
    x_end = ov.key_row(D_KEY + 2 * KEY_ROW, M, [
        ("name", "Rule"),
        ("marker", dict(marker="o", markersize=3.2, markerfacecolor="white", markeredgecolor=INK["muted"],
                        markeredgewidth=0.7), f"{RULE_CODE['top25']}, {RULE_DISPLAY['top25']}"),
        ("marker", dict(marker="o", markersize=3.2, markerfacecolor=INK["primary"], markeredgecolor=INK["primary"],
                        markeredgewidth=0.7), f"{RULE_CODE['rankfree']}, {RULE_DISPLAY['rankfree']} (adopted rule)")],
        name_w)
    right.append(ov.key_row(D_KEY + 2 * KEY_ROW, x_end + 8.0, [("name", "Resolution"), ("marker", tri, "used")],
                            name_w))

    # =================================================================================================================
    # What is drawn against the stored tables
    # =================================================================================================================
    n_cmp, max_diff = 0, 0.0
    for key, item in art.items():
        obj, stored = item[0], np.asarray(item[1], float)
        if key[-1] == "range":
            drawn = np.array([[seg[0][1], seg[1][1]] for seg in obj.get_segments()], float)
        elif key[0] in ("E", "G"):
            n = N_DCT[key[1]]
            drawn = np.array([b.get_height() for b in obj], float) * n
            bottoms = np.array([b.get_y() for b in obj], float) * n
            max_diff = max(max_diff, float(np.max(np.abs(bottoms - item[2]))))
            n_cmp += bottoms.size
        else:
            drawn = np.asarray(obj.get_ydata(), float)
        if drawn.shape != stored.shape:
            raise SystemExit(f"{key}: drawn {drawn.shape} and stored {stored.shape} values differ in shape.")
        n_cmp += drawn.size
        max_diff = max(max_diff, float(np.max(np.abs(drawn - stored))))
    n_values = len(values) + sum(1 for v in values if np.isfinite(v["range_low"])) * 2
    checks.add("plotted values read back from the drawn lines, bars and range bars equal the stored tables",
               f"{len(art)} drawn series; {n_cmp} numbers compared (values file: {len(values)} rows); max abs diff "
               f"{max_diff:.1e} (bars compared in cells)", max_diff < 1e-9 and n_cmp >= n_values)
    texts = [t.get_text() for t in t27f.drawn_texts(fig)]
    letters = sorted(s for s in texts if len(s) == 1 and s.isalpha() and s != "T")
    shown = set(texts)
    need = ([RULE_DISPLAY["top25"], f"{RULE_DISPLAY['rankfree']} (adopted rule)", "not resolved", SHADE_TEXT]
            + list(GRAPH_DISPLAY.values()) + [CONF_DISPLAY[x] for x in CONFIDENCES]
            + [v[1] for v in row_names.values()])
    missing = [s for s in need if s not in shown]
    checks.add("names shown: rules, the three graphs, 'not resolved', confidence classes, "
               "conditions Control and Zero-K", f"{len(need)} names; missing: {', '.join(missing) or 'none'}",
               not missing)
    process = ("submitted", "submission", "re-annotation", "reannotation", "revis", "resubmi", "first version")
    found = sorted({w for w in process for s in texts if w in s.lower()})
    n_hlines = sum(1 for ax in fig.axes for ln in ax.get_lines()
                   if len(ln.get_ydata()) == 2 and ln.get_ydata()[0] == ln.get_ydata()[1] == N_REANNOT_TYPES)
    checks.add("no drawn text contains 'submitted' or 're-annotation' (nor another word for the revision process); "
               "the reference line at 14 cell types is not drawn",
               f"{len(texts)} texts; found: {', '.join(found) or 'none'}; horizontal lines at {N_REANNOT_TYPES}: "
               f"{n_hlines}", not found and n_hlines == 0)
    banned = ("rank-free", "rescue", "stored", "hvg", "Fig 2", "_", "Generated", "primary", "sensitivity", "DE-rule", "Ctrl", "ZeroK")
    hits = sorted({b for b in banned for s in texts if b in s})
    n_unknown = sum("Unknown" in s for s in texts)
    checks.add("no figure title, footer, task code, file name or internal name among the drawn texts; 'Unknown' only "
               "in the kidney heading (C); panel letters A-G, upper case",
               f"{len(texts)} texts; found: {', '.join(hits) or 'none'}; texts with 'Unknown': {n_unknown}; letters "
               f"{''.join(letters)}", not hits and n_unknown == 1 and letters == list("ABCDEFG"))
    checks.add("keys end inside the figure (right edge of the widest key row, mm)", f"{max(right):.1f} of {W:.0f}",
               max(right) <= W - M)

    figs = lib.Figures(run, STEP)
    row = figs.save(fig, FIGURE,
                    "Supplementary Figure S-05-c: resolution scans. Total kidney: cell types resolved (A), with a "
                    "type-specific marker (B), fraction Unknown (C), three graphs, both rules. DCT control and zero-K: "
                    "clusters (D), cells per label (E), label stability over Leiden seeds (F), confidence (G)",
                    lib.FULL_MM)
    ok_fonts = set(row["fonts"].split(";")) <= {"arial.ttf", "arialbd.ttf"}
    checks.add("figure format checks (Arial >= 7 pt, no text overlap or text outside, 180 mm, TrueType in PDF, 600 "
               "dpi; height <= 230 mm)",
               f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; fonts {row['fonts']}; "
               f"overlaps {row['n_text_overlaps']}; outside {row['n_text_outside_figure']}; png {row.get('png_dpi')}",
               bool(row["pass"]) and ok_fonts and row["height_mm"] <= 230)
    run.table(pd.DataFrame(values), f"{STEP}_values")
    run.table(pd.concat([checks.frame(), pd.DataFrame(figs.rows)], ignore_index=True), f"{STEP}_checks")
    run.finish({"request": "authors, 12 Nov 2026: 'Yes, redraw them'",
                "replaces_for_readers": ["results/figures/kidney_annotation/kidney_annotation_rankfree_summary",
                                         "results/figures/dct_labels/dct_labels_scan_summary_ctrl",
                                         "results/figures/dct_labels/dct_labels_scan_summary_zerok"],
                "panels": {**{p[0]: f"Total kidney: {p[2]}" for p in k_panels},
                           **{a: f"DCT, per condition: {b}" for a, b in d_heads}},
                "rule_display": RULE_DISPLAY, "rule_code": RULE_CODE, "graph_display": GRAPH_DISPLAY,
                "condition_display": COND_DISPLAY, "label_display": LABEL_DISPLAY, "confidence_display": CONF_DISPLAY,
                "change_12_nov_2026": "authors via the coordinator: no reference to the submission in the figure; all "
                                      "three graphs kept, blue series named 'graph of the main analysis'; reference "
                                      "line 'submitted re-annotation: 14 types' removed; shading named '2 and lower'",
                "resolution_used": ADOPTED, "graph_colours": gc, "label_colours": {x: t18.LABEL_COLORS[x] for x in LABELS},
                "confidence_colours": CONF_COLORS, "width_mm": W, "height_mm": round(H, 2), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit("A check failed; see report_05_s05c_figure_checks.csv.")


if __name__ == "__main__":
    main()
