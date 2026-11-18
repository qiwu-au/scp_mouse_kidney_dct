#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S-03-a (preprocessing sensitivity), redrawn with reader-facing names.

(authors, 12 Nov 2026: "The readers will not know the revision
process, so we only show the final products"; "Yes, remove them"). The figure of the response to the reviewers
(report_03_preprocessing_summary, drawn by report_preprocessing_scores.py, which is not edited) names
"the submitted pipeline" and says "Unknown left out" above the DCT columns. This script redraws the same five
panels from the stored value tables of the response to the reviewers. No pipeline is recomputed and no statistic is new: every plotted
value is read from
- results/tables/report/report_03_pipeline_summary.csv (panels A-D; rows of the primary labels:
  kidney_t15_label, cs_ctrl_deg_label, cs_zerok_pipeline_label), and
- results/tables/report/report_03_detection_summary.csv (panel E; shown as % of cells),
and the cell numbers of the column heads from report_03_intensity_descriptors.csv.

What differs from the earlier figure (layout, limits, ticks, marks and colours are the same):
- "submitted pipeline" -> "main pipeline" (key below the panels);
- column heads "DCT Control", "DCT Zero-K";
- panel B heading: the kidney label Unknown keeps its name; the DCT cells of clusters without marker support are
  "not resolved" ("left out: Unknown in kidney, not resolved in DCT").
No in-figure title or footer; Arial >= 7 pt; 180 mm; PDF and 600-dpi PNG.

Checks (written to the checks table): the values held by the drawn marks, lines and value labels equal the stored
tables; the numbers that the the response to the reviewers letter takes from this figure; the format checks; the drawn texts are the
expected ones and none contains "submitted", "submission", "re-annotation" or an internal name.

Inputs (read-only): the three stored tables above.
Outputs: results/figures/report/report_03_S03a_preprocessing_summary.{pdf,png};
         results/tables/report/report_03_s03a_figure_{values,checks}.csv;
         results/tables/report/report_step_03_s03a_figure_manifest.json

Command (project root):  python scripts/report_preprocessing_figure.py --overwrite
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
from report_table_lib import FORBIDDEN, PROCESS_WORDS

plt = T8.setup_matplotlib()

TASK = "report"
STEP = "03_s03a_figure"
FIGURE = "03_S03a_preprocessing_summary"
TABLES = cc.OUT_ROOTS["tables"] / TASK
SUMMARY, DETECT, CELLS = (TABLES / f"report_03_{name}.csv" for name in
                          ("pipeline_summary", "detection_summary", "intensity_descriptors"))

DATASETS = ("kidney", "Ctrl", "ZeroK")                       # values of `dataset` in the stored tables
NAME = {"kidney": "Total kidney", "Ctrl": "DCT Control", "ZeroK": "DCT Zero-K"}
MAIN_CODE = "submitted"                                      # stored group code of the pipeline the paper uses
MAIN_ID = "none__z10__all"
STRIPS = (("All proteins (5)", ("cellwise_scaled_all", "unscaled_all")), ("HVG sets (18)", ("hvg",)),
          ("High coverage (3)", ("highcov50",)), ("PCs or k changed (6)", ("pcs_or_k",)))
STRIPS_E = ("All proteins", "HVG sets (9)", "High coverage")
# letter, stored column, heading, x limits, ticks, tick format
PANELS = (("A", "best_nmi", "Agreement of Leiden clusters with the cell labels (NMI, best of four resolutions)",
           (0.0, 0.8), [0, 0.2, 0.4, 0.6, 0.8], "{:.1f}"),
          ("B", "macro_knn_purity", "kNN label purity (mean over labels with at least 10 cells; left out: Unknown in "
           "kidney, not resolved in DCT)", (0.4, 1.0), [0.4, 0.6, 0.8, 1.0], "{:.1f}"),
          ("C", "macro_silhouette", "Silhouette width by label (mean over the same labels)", None, None, "{:.1f}"),
          ("D", "max_abs_rho_pc12", "Largest |ρ| of PC1 or PC2 with protein IDs per cell",
           (0.6, 1.0), [0.6, 0.7, 0.8, 0.9, 1.0], "{:.1f}"))
HEAD_E = "Median detection of the selected proteins (% of cells)"
KEYS = ("with per-protein scaling", "without per-protein scaling", "main pipeline, with its value above the panel")
M, FONT, INK = T8.M, T8.FONT, T8.INK
BLUE = "#2a78d6"
DASH = (0, (3, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-03-a from the stored tables.")
    cc.add_common_args(parser)
    return parser.parse_args()


def read(path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def value_text(base: float, fmt: str) -> str:
    return f"{base:.1f}" if fmt == "{:.0f}" else f"{base:.3f}"


# ---------------------------------------------------------------------------
# What each panel shows, from the stored tables
# ---------------------------------------------------------------------------
def panel_strips(summary: pd.DataFrame, dataset: str, column: str) -> tuple[float, list[tuple[str, pd.DataFrame]]]:
    """Panels A-D: value of the main pipeline, and per strip the pipelines with their value and scaling."""
    sub = summary[(summary["dataset"] == dataset) & (summary["label_role"] == "primary")]
    base = float(sub.loc[sub["group"] == MAIN_CODE, column].iloc[0])
    strips = []
    for text, groups in STRIPS:
        part = sub[sub["group"].isin(groups)]
        strips.append((text, pd.DataFrame({"pipeline": part["variant_id"].to_numpy(),
                                           "value": part[column].to_numpy(float),
                                           "scaled": (part["scaling"] == "z10").to_numpy()})))
    return base, strips


def detection_strips(detect: pd.DataFrame, dataset: str) -> tuple[float, list[tuple[str, pd.DataFrame]]]:
    """Panel E: median detection (% of cells) of the protein sets; detection does not depend on scaling."""
    d = detect[detect["dataset"] == dataset]
    frame = pd.DataFrame({"pipeline": (d["normalization"] + "__" + d["features"]).to_numpy(),
                          "value": (100 * d["median_detection_rate_selected"]).to_numpy(float), "scaled": True,
                          "normalization": d["normalization"].to_numpy(), "features": d["features"].to_numpy()})
    none = frame["normalization"] == "none"
    parts = (frame[none & (frame["features"] == "all")], frame[frame["features"].str.startswith("hvg")],
             frame[none & (frame["features"] == "highcov50")])
    base = float(parts[0]["value"].iloc[0])
    return base, [(text, part[["pipeline", "value", "scaled"]]) for text, part in zip(STRIPS_E, parts)]


# ---------------------------------------------------------------------------
# Drawing (as in the earlier figure)
# ---------------------------------------------------------------------------
def clean(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK["muted"])
    ax.tick_params(axis="x", length=1.5, width=0.4, labelsize=FONT, colors=INK["secondary"], pad=1.5)
    ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"], pad=2)


def strip_panel(ax, strips, base: float, xlim, ticks, fmt: str, labels: bool, plotted: list[dict], panel: str,
                dataset: str) -> None:
    """One strip per group of pipelines; dots spread evenly in y inside the strip (the offset carries no meaning)."""
    n = len(strips)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s[0] for s in strips] if labels else [""] * n)
    ax.set_xlim(*xlim)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt.format(t) for t in ticks])
    for i in range(n):
        ax.axhline(i, color=INK["grid"], lw=0.3, zorder=0)
    clean(ax)
    ax.axvline(base, color=INK["secondary"], lw=0.7, ls=DASH, zorder=1, gid="main")
    for i, (text, sub) in enumerate(strips):
        sub = sub.sort_values("value").reset_index(drop=True)
        offs = np.zeros(len(sub)) if len(sub) == 1 else np.tile([-0.24, 0.0, 0.24], len(sub))[:len(sub)]
        for off, r in zip(offs, sub.itertuples()):
            ax.plot([r.value], [i + off], "o", ms=2.5, mec=BLUE, mew=0.5, mfc=BLUE if r.scaled else "white", zorder=3,
                    clip_on=False, gid="point")
            plotted.append({"panel": panel, "dataset": NAME[dataset], "strip": text, "pipeline": r.pipeline,
                            "per_protein_scaling": bool(r.scaled) if panel != "E" else "", "value": float(r.value),
                            "main_pipeline_value": base})
    frac = (base - xlim[0]) / (xlim[1] - xlim[0])
    ax.text(base, -0.5, value_text(base, fmt), ha="right" if frac > 0.85 else "center", va="bottom", fontsize=FONT,
            color=INK["primary"], gid="value")


def draw(summary: pd.DataFrame, detect: pd.DataFrame, heads: dict[str, str], plotted: list[dict]):
    sil = summary.loc[summary["label_role"] == "primary", "macro_silhouette"]
    sil_lim = (np.floor(float(sil.min()) * 10) / 10, np.ceil(float(sil.max()) * 10) / 10)
    sil_ticks = [t for t in (-0.2, 0.0, 0.2, 0.4, 0.6) if sil_lim[0] <= t <= sil_lim[1]]

    labels_w = max(T8.text_mm(s[0])[0] for s in STRIPS)
    x0 = M + labels_w + 1.5
    gap = 5.0
    pw = (T8.W - M - x0 - 2 * gap) / 3
    strip_h = 4.2
    head_h, value_h, axis_h, row_gap = 3.4, 3.2, 4.0, 2.2
    col_h = 3.4
    y = M + col_h + 0.6
    layout = []
    for p in PANELS:
        layout.append((p, y))
        y += head_h + value_h + 4 * strip_h + axis_h + row_gap
    y_e = y
    y_leg = y_e + head_h + value_h + 3 * strip_h + axis_h + 2.0
    sh = T8.Sheet(y_leg + 4.0 + M)
    xs = [x0 + i * (pw + gap) for i in range(3)]
    for ds, x in zip(DATASETS, xs):
        sh.text(x, M, heads[ds], fontweight="bold", va="top")

    axes = {}
    for (letter, column, head, xlim, ticks, fmt), y_row in layout:
        sh.text(M, y_row, letter, fontsize=8.0, fontweight="bold", va="top")
        sh.text(M + 4.0, y_row + 0.2, head, va="top")
        if column == "macro_silhouette":
            xlim, ticks = sil_lim, sil_ticks
        for j, (ds, x) in enumerate(zip(DATASETS, xs)):
            base, strips = panel_strips(summary, ds, column)
            ax = axes[(letter, ds)] = sh.ax(x, y_row + head_h + value_h, pw, 4 * strip_h)
            strip_panel(ax, strips, base, xlim, ticks, fmt, j == 0, plotted, letter, ds)

    sh.text(M, y_e, "E", fontsize=8.0, fontweight="bold", va="top")
    sh.text(M + 4.0, y_e + 0.2, HEAD_E, va="top")
    for j, (ds, x) in enumerate(zip(DATASETS, xs)):
        base, strips = detection_strips(detect, ds)
        ax = axes[("E", ds)] = sh.ax(x, y_e + head_h + value_h, pw, 3 * strip_h)
        strip_panel(ax, strips, base, (0, 80), [0, 20, 40, 60, 80], "{:.0f}", j == 0, plotted, "E", ds)

    ax = sh.ax(0, y_leg, sh.W, 4.0)
    ax.set_xlim(0, sh.W)
    ax.set_ylim(4, 0)
    ax.axis("off")
    x = x0
    yc = 2.0

    def key(mark, text: str, w: float = 2.0) -> None:
        nonlocal x
        mark(x, w)
        x += w + 1.2
        sh.text(x, y_leg + yc, text, va="center", color=INK["secondary"])
        x += T8.text_mm(text)[0] + 5.0

    key(lambda a, w: ax.plot([a + w / 2], [yc], "o", ms=2.5, mec=BLUE, mew=0.5, mfc=BLUE), KEYS[0])
    key(lambda a, w: ax.plot([a + w / 2], [yc], "o", ms=2.5, mec=BLUE, mew=0.5, mfc="white"), KEYS[1])
    key(lambda a, w: ax.plot([a, a + w], [yc, yc], color=INK["secondary"], lw=0.7, ls=DASH), KEYS[2], 5.0)
    return sh, axes


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def drawn(ax) -> dict:
    """What an axes holds: the points per strip, the dashed line and the value label."""
    points = [(int(round(float(line.get_ydata()[0]))), float(line.get_xdata()[0])) for line in ax.lines
              if line.get_gid() == "point"]
    line = [float(l.get_xdata()[0]) for l in ax.lines if l.get_gid() == "main"]
    label = [t.get_text() for t in ax.texts if t.get_gid() == "value"]
    hollow = sum(1 for l in ax.lines if l.get_gid() == "point" and l.get_markerfacecolor() == "white")
    return {"points": points, "line": line, "label": label, "hollow": hollow}


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)
    checks: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})

    summary = read(run.use_input(SUMMARY, "pipelines: scores per dataset and label set"))
    detect = read(run.use_input(DETECT, "detection of the selected proteins"))
    cells = read(run.use_input(CELLS, "cells per dataset"))
    n_cells = {ds: sorted({int(n) for n in cells.loc[cells["dataset"] == ds, "n_cells"]}) for ds in DATASETS}
    check("cells per dataset in the stored table (column heads)", n_cells, "712, 729, 702",
          n_cells == {"kidney": [712], "Ctrl": [729], "ZeroK": [702]})
    heads = {ds: f"{NAME[ds]} ({n_cells[ds][0]} cells)" for ds in DATASETS}

    prim = summary[summary["label_role"] == "primary"]
    sizes = {ds: prim[prim["dataset"] == ds].groupby("group").size().to_dict() for ds in DATASETS}
    want = {"submitted": 1, "cellwise_scaled_all": 2, "unscaled_all": 3, "hvg": 18, "highcov50": 3, "pcs_or_k": 6}
    labels = sorted(set(prim["label_column"]))
    check("stored table: 33 pipelines per dataset under the primary labels (the main pipeline and 2 + 3, 18, 3 and 6 "
          "alternatives, as the strip names say); the main pipeline is none__z10__all",
          f"{sizes}; labels {labels}; main {sorted(set(prim.loc[prim['group'] == MAIN_CODE, 'variant_id']))}",
          f"{want} in each dataset", all(sizes[ds] == want for ds in DATASETS)
          and labels == ["cs_ctrl_deg_label", "cs_zerok_pipeline_label", "kidney_t15_label"]
          and set(prim.loc[prim["group"] == MAIN_CODE, "variant_id"]) == {MAIN_ID})

    plotted: list[dict] = []
    sh, axes = draw(summary, detect, heads, plotted)
    values = pd.DataFrame(plotted)

    # 1. What the artists hold against the stored tables, read again from disk.
    fresh_s = read(SUMMARY)
    fresh_s = fresh_s[fresh_s["label_role"] == "primary"]
    fresh_d = read(DETECT)
    n_points, worst, bad_lines, bad_labels, hollow = 0, 0.0, [], [], {}
    for (letter, ds), ax in axes.items():
        got = drawn(ax)
        if letter == "E":
            d = fresh_d[fresh_d["dataset"] == ds]
            none = d["normalization"] == "none"
            stored = [100 * d.loc[none & (d["features"] == "all"), "median_detection_rate_selected"],
                      100 * d.loc[d["features"].str.startswith("hvg"), "median_detection_rate_selected"],
                      100 * d.loc[none & (d["features"] == "highcov50"), "median_detection_rate_selected"]]
            fmt = "{:.0f}"
        else:
            column = next(p[1] for p in PANELS if p[0] == letter)
            f = fresh_s[fresh_s["dataset"] == ds]
            stored = [f.loc[f["group"].isin(groups), column] for _, groups in STRIPS]
            fmt = "{:.1f}"
            hollow[(letter, ds)] = (got["hollow"], int((f.loc[f["group"] != MAIN_CODE, "scaling"] != "z10").sum()))
        base = float(stored[0].iloc[0]) if letter == "E" else float(f.loc[f["group"] == MAIN_CODE, column].iloc[0])
        for i, series in enumerate(stored):
            a = np.sort(np.array([x for strip, x in got["points"] if strip == i]))
            b = np.sort(series.to_numpy(float))
            n_points += len(a)
            worst = max(worst, float(np.abs(a - b).max()) if len(a) == len(b) else np.inf)
        if got["line"] != [base]:
            bad_lines.append(f"{letter} {ds}")
        if got["label"] != [value_text(base, fmt)]:
            bad_labels.append(f"{letter} {ds}: {got['label']}")
    check("plotted values equal the stored tables: x position of every point, per panel, dataset and strip",
          f"{n_points} points in {len(axes)} panels; largest difference {worst:g}", "417 points; difference 0",
          n_points == 417 and worst == 0.0)
    check("dashed lines sit at the stored value of the main pipeline, and the label above each panel is that value",
          f"lines off: {bad_lines or 'none'}; labels off: {bad_labels or 'none'}", "none; none",
          not bad_lines and not bad_labels)
    check("hollow points = pipelines without per-protein scaling (panels A-D)",
          sorted(set(hollow.values())), "[(12, 12)]", set(hollow.values()) == {(12, 12)})
    check("values table = points drawn", f"{len(values)} rows", "417 rows",
          len(values) == 417 and not values.duplicated(["panel", "dataset", "pipeline"]).any())

    # 2. Numbers that the the response to the reviewers letter takes from this figure (from the values of the drawn points).
    def v(panel: str, ds: str, strip: str | None = None) -> pd.DataFrame:
        sub = values[(values["panel"] == panel) & (values["dataset"] == NAME[ds])]
        return sub if strip is None else sub[sub["strip"] == strip]

    def main_values(panel: str, digits: int = 3) -> str:
        return ", ".join(f"{v(panel, ds)['main_pipeline_value'].iloc[0]:.{digits}f}" for ds in DATASETS)

    def span(panel: str, strip: str, digits: int = 3) -> str:
        return "; ".join(f"{v(panel, ds, strip)['value'].min():.{digits}f}-{v(panel, ds, strip)['value'].max():.{digits}f}"
                         for ds in DATASETS)

    def above(panel: str, strip: str) -> str:
        return ", ".join(str(int((v(panel, ds, strip)["value"] > v(panel, ds)["main_pipeline_value"].iloc[0]).sum()))
                         for ds in DATASETS)

    def one(panel: str, ds: str, pipeline: str) -> float:
        return float(v(panel, ds).loc[v(panel, ds)["pipeline"] == pipeline, "value"].iloc[0])

    hvg, allp = "HVG sets (18)", "All proteins (5)"
    letter = [
        ("main pipeline, NMI (A): kidney, control, zero-K", main_values("A"), "0.764, 0.561, 0.416"),
        ("main pipeline, kNN purity (B)", main_values("B"), "0.875, 0.857, 0.968"),
        ("main pipeline, silhouette width (C)", main_values("C"), "0.022, 0.113, 0.264"),
        ("main pipeline, largest |rho| (D)", main_values("D"), "0.994, 0.999, 0.999"),
        ("all proteins, median detection in % of cells (E)", main_values("E", 1), "8.6, 18.7, 16.4"),
        ("HVG pipelines, NMI range (A)", span("A", hvg), "0.410-0.603; 0.211-0.334; 0.165-0.240"),
        ("HVG pipelines, kNN purity range (B)", span("B", hvg), "0.487-0.804; 0.644-0.816; 0.832-0.927"),
        ("HVG pipelines, largest |rho| range (D)", span("D", hvg), "0.654-0.981; 0.893-0.966; 0.900-0.979"),
        ("HVG pipelines above the main pipeline: NMI (A)", above("A", hvg), "0, 0, 0"),
        ("HVG pipelines above the main pipeline: kNN purity (B)", above("B", hvg), "0, 0, 0"),
        ("HVG pipelines above the main pipeline: silhouette width (C)", above("C", hvg), "5, 15, 5"),
        ("all proteins with per-cell normalization or without scaling, largest |rho| range (D)", span("D", allp),
         "0.981-0.996; 0.997-1.000; 0.994-1.000"),
        ("total-intensity normalization: NMI / kNN purity / largest |rho|",
         "; ".join(" / ".join(f"{one(p, ds, 'total__z10__all'):.3f}" for p in "ABD") for ds in DATASETS),
         "0.645 / 0.850 / 0.986; 0.579 / 0.857 / 0.997; 0.292 / 0.961 / 0.997"),
        ("median normalization: NMI / kNN purity / largest |rho|",
         "; ".join(" / ".join(f"{one(p, ds, 'median__z10__all'):.3f}" for p in "ABD") for ds in DATASETS),
         "0.660 / 0.865 / 0.996; 0.494 / 0.852 / 1.000; 0.303 / 0.965 / 1.000"),
        ("control, total-intensity normalization: silhouette width against the main pipeline (C)",
         f"{one('C', 'Ctrl', 'total__z10__all'):.3f} vs {v('C', 'Ctrl')['main_pipeline_value'].iloc[0]:.3f}",
         "0.120 vs 0.113"),
        ("HVG sets, median detection in % of cells (E)", span("E", "HVG sets (9)", 1), "2.0-3.9; 4.8-8.5; 3.3-6.0"),
    ]
    for text, observed, expected in letter:
        check(f"number of the the response to the reviewers letter in the figure: {text}", observed, expected, observed == expected)

    # 3. Drawn texts.
    import matplotlib

    sh.fig.canvas.draw()
    texts = [t.get_text() for t in sh.fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    words = sorted({t for t in texts if PROCESS_WORDS.search(t)})
    check("final products only: no drawn text contains 'submitted', 'submission' or 're-annotation'",
          words or "none", "none", not words)
    internal = sorted({t for t in texts if FORBIDDEN.search(t)})
    check("no drawn text holds a task code or an internal name (Ctrl, ZeroK, column names, file names)",
          internal or "none", "none", not internal)
    expected = {*heads.values(), *"ABCDE", *(p[2] for p in PANELS), HEAD_E, *(s[0] for s in STRIPS), *STRIPS_E, *KEYS}
    number = re.compile(r"^[-−]?\d+(\.\d+)?$")
    other = sorted({t for t in texts if t not in expected and not number.match(t)})
    missing = sorted(expected - set(texts))
    check("drawn texts: column heads, panel letters, five headings, strip names, key and numbers only (no title, no "
          "footer)", f"{len(texts)} texts; other: {other or 'none'}; missing: {missing or 'none'}",
          "other: none; missing: none", not other and not missing)
    unknown = [t for t in texts if "Unknown" in t]
    check("names: column heads DCT Control and DCT Zero-K; Unknown is named for the kidney only, the DCT cells are "
          "'not resolved' (panel B heading)", f"heads {list(heads.values())}; texts with Unknown: {unknown}",
          "Unknown in one text, with 'in kidney' and 'not resolved in DCT'",
          list(heads.values()) == ["Total kidney (712 cells)", "DCT Control (729 cells)", "DCT Zero-K (702 cells)"]
          and len(unknown) == 1 and "Unknown in kidney, not resolved in DCT" in unknown[0])

    # 4. Format (>= 7 pt, Arial, no overlaps, text inside, 180 mm) and files.
    rows: list[dict] = []
    ok = T8.save_checked(run, sh, FIGURE, "Supplementary Figure S-03-a: preprocessing sensitivity analyses", rows)
    f = rows[0]
    check("format: Arial, smallest font >= 7 pt, no text overlaps, no text outside the figure, width 180 mm, PDF "
          "with embedded TrueType Arial, PNG at 600 dpi",
          f"{f['width_mm']:.0f} x {f['height_mm']} mm; min font {f['min_font_pt']} pt; fonts {f['fonts']}; overlaps "
          f"{f['n_text_overlaps']}; outside {f['n_text_outside_figure']}; PNG {f['png_px']} px "
          f"({f['png_width_mm']} mm); PDF width {f['pdf_width_mm']} mm", "pass", ok)

    for c in checks:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['observed']}")
    run.table(values, f"{STEP}_values")
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    n_fail = sum(not c["pass"] for c in checks)
    print(f"Checks: {len(checks) - n_fail} passed, {n_fail} failed")
    run.finish({"figure": "Supplementary Figure S-03-a", "size_mm": f"{f['width_mm']:.0f} x {f['height_mm']}",
                "points": len(values), "source": "stored value tables of the response to the reviewers; nothing recomputed",
                "vocabulary": "main pipeline; DCT Control / DCT Zero-K; kidney Unknown, DCT not resolved",
                "checks": len(checks), "checks_failed": n_fail, "checks_pass": n_fail == 0},
               status="completed" if n_fail == 0 else "completed_with_failed_checks")
    if n_fail:
        raise SystemExit("A check failed; see report_03_s03a_figure_checks.csv.")


if __name__ == "__main__":
    main()
