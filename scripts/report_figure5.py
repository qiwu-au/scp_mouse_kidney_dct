#!/usr/bin/env python
"""The response to the reviewers: Figure 5 (zero-K against control, pre-specified panel proteins) in the journal's style.

(authors, 15 Nov 2026: "Yes, redraw"; the journal's "Checklist
for Style"). The compiled figure of the response to the reviewers (report_19_fig_F19_compiled) is drawn again by its own functions
(``draw_panel``, ``draw_matched``, ``letter``, ``Sub`` of report_zerok_figures.py, imported), from the same
stored zerok_vs_control tables, at the same size (180 x 127.2 mm). That drawing is compared with the current PNG pixel for pixel;
then these strings are changed in the drawn figure and nothing else:
  panel letters   A, B -> a, b
  axis labels     sentence case: "Cells with the protein detected (%)", "Intensity in detected cells (log2 fold
                  change)", "Mean intensity per cell (log2 ratio)", "Proteins changed in the expected direction",
                  "Matched random sets (%)" (line breaks kept)
  P value         "empirical p = 0.0001" -> "empirical P = 0.0001", the P capital and italic (Arial Italic)
The italic P: matplotlib sets one font per text, so the annotation's second line is written as three runs
("empirical ", an italic "P", " = 0.0001") on the baseline of that line, ending at its right edge; the first line
("observed: 7 of 7") is written at its own place. The annotation itself keeps its two-line text for its text box only
and draws its arrow, so the arrow is the one of the current figure at every resolution. No mathtext is used.
No new statistics; no value, colour, position or size is changed. The the response to the reviewers files are not written.

Inputs (read-only): results/tables/zerok_vs_control/zerok_vs_control_panel_results.csv,
  zerok_vs_control_specificity_matched_null.csv, zerok_vs_control_specificity_null_draws_summary.csv; the frozen zerok_core panel and
  the plan hashes (checked as in the response to the reviewers); results/tables/report/report_19_zerok_figures_values.csv
  (values check); results/figures/report/report_19_fig_F19_compiled.{pdf,png} (comparison).
Outputs: results/figures/report/report_27_figF19_ki_style.pdf (vector), .png (600 dpi), .tif (1000 dpi, drawn
  by matplotlib at dpi 1000; RGB, LZW);
  results/tables/report/report_27_redraw_figF19_checks.csv;
  results/tables/report/report_step_27_redraw_figF19_manifest.json

Command (project root):  python scripts/report_figure5.py --overwrite
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
import types
from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import report_zerok_figures as z19        # drawing functions of the response to the reviewers (sets the atlas_dct figure style)
import report_figure_style_lib as R

from matplotlib import font_manager
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Affine2D, IdentityTransform, offset_copy

t8, lib30 = z19.t8, z19.lib30
TASK = R.TASK
STEP = "27_redraw_figF19"
NAME = "27_figF19_ki_style"
FIG_DIR = cc.OUT_ROOTS["figures"] / TASK
OLD_PDF = FIG_DIR / "report_19_fig_F19_compiled.pdf"
OLD_PNG = FIG_DIR / "report_19_fig_F19_compiled.png"
OLD_VALUES = cc.OUT_ROOTS["tables"] / TASK / "report_19_zerok_figures_values.csv"
SINGLE = ("19_fig_panel_effects", "19_fig_matched_sets")      # names under which the response to the reviewers stored the plotted values

# whole strings of the current figure -> journal style
STRINGS = {
    "A": "a",
    "B": "b",
    "cells with the protein\ndetected (%)": "Cells with the protein\ndetected (%)",
    "intensity in detected cells\n(log2 fold change)": "Intensity in detected cells\n(log2 fold change)",
    "mean intensity per cell\n(log2 ratio)": "Mean intensity per cell\n(log2 ratio)",
    "proteins changed in the expected direction": "Proteins changed in the expected direction",
    "matched random sets (%)": "Matched random sets (%)",
}
P_OLD = ("observed: 7 of 7", "empirical p = 0.0001")          # the two lines of the annotation in the current figure
P_RUNS = ("empirical ", "P", " = 0.0001")                     # second line in the journal's style; the P is italic
# lines of the PDF text (pypdf) that change: old -> new
EXPECTED_LINES = [
    ("cells with the protein", "Cells with the protein"),
    ("intensity in detected cells", "Intensity in detected cells"),
    ("mean intensity per cell", "Mean intensity per cell"),
    ("proteins changed in the expected direction", "Proteins changed in the expected direction"),
    ("30matched random sets (%)", "30Matched random sets (%)"),
    ("empirical p = 0.0001", "empirical P = 0.0001"),
    ("A", "a"),
    ("B", "b"),
]


class RunText(Text):
    """One run of a line of text: drawn so that it ends where the runs after it begin.

    The text is right-aligned at its anchor and moved left by the width that the following runs have in the renderer
    in use (so the runs join at 600 dpi, at 1000 dpi and in the PDF, where text widths differ slightly).
    """

    def __init__(self, x, y, text, following=(), **kwargs):
        super().__init__(x, y, text, **kwargs)
        self._following = tuple(following)          # (string, FontProperties) of the runs to the right

    def _moved(self, renderer):
        width = sum(renderer.get_text_width_height_descent(s, fp, False)[0] for s, fp in self._following)
        return Affine2D().translate(-width, 0.0)

    def draw(self, renderer):
        base = self.get_transform()
        self.set_transform(base + self._moved(renderer))
        try:
            super().draw(renderer)
        finally:
            self.set_transform(base)

    def get_window_extent(self, renderer=None, dpi=None):
        if renderer is None:
            renderer = self.get_figure(root=True)._get_renderer()
        base = self.get_transform()
        self.set_transform(base + self._moved(renderer))
        try:
            return super().get_window_extent(renderer, dpi)
        finally:
            self.set_transform(base)


def _draw_arrow_only(self, renderer):
    """Annotation.draw without the text: the arrow keeps the geometry given by the annotation's own text box."""
    if not self.get_visible():
        return
    self.update_positions(renderer)
    if self.arrow_patch is not None:
        self.arrow_patch.draw(renderer)


def italic_p(fig) -> dict:
    """Rewrite the annotation "observed ... / empirical p = ..." with a capital italic P (see the module docstring).

    Returns the texts written, the annotation (now drawing its arrow only) and the pixel boxes of the old and the new
    text in the 600-dpi image.
    """
    anns = fig.findobj(Annotation)
    if len(anns) != 1 or tuple(anns[0].get_text().split("\n")) != P_OLD:
        raise SystemExit(f"expected one annotation with the lines {P_OLD}; found {[a.get_text() for a in anns]}")
    ann = anns[0]
    if (ann.get_ha(), ann.get_va(), ann.anncoords, ann.get_rotation()) != ("right", "bottom", "data", 0.0):
        raise SystemExit("the annotation is not right- and bottom-aligned at a data position; layout rule not valid")
    ax = ann.axes
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    scale = R.DPI_PNG / fig.dpi
    height_px = int(fig.bbox.height * scale)
    box_old = R._box(Text.get_window_extent(ann, r), height_px, scale)
    regular = ann.get_fontproperties().copy()
    italic = regular.copy()
    italic.set_style("italic")
    # Baseline of the last line above the bottom of a two-line text, and the line pitch (matplotlib's own layout,
    # read from a two-line probe in the annotation's font; both in points, which do not depend on the resolution).
    probe = Text(0.0, 0.0, "x\nx", fontproperties=regular, va="baseline", ha="left", transform=IdentityTransform())
    probe.set_figure(fig)
    pb = probe.get_window_extent(r)
    to_pt = 72.0 / fig.dpi
    last_pt, pitch_pt = (0.0 - pb.y0) * to_pt, pb.height / 2 * to_pt
    two_lines = Text.get_window_extent(ann, r).height * to_pt
    if abs(two_lines - 2 * pitch_pt) > 1e-6:
        raise SystemExit("the annotation's lines are not at the regular line pitch; layout rule not valid")
    x, y = ann.xyann
    common = dict(ha="right", va="baseline", color=ann.get_color(), clip_on=False, zorder=ann.get_zorder())
    line1 = Text(x, y, P_OLD[0], fontproperties=regular,
                 transform=offset_copy(ax.transData, fig=fig, x=0.0, y=last_pt + pitch_pt, units="points"), **common)
    fonts = (regular, italic, regular)
    runs = []
    for i, (s, fp) in enumerate(zip(P_RUNS, fonts)):
        runs.append(RunText(x, y, s, following=list(zip(P_RUNS[i + 1:], fonts[i + 1:])), fontproperties=fp,
                            transform=offset_copy(ax.transData, fig=fig, x=0.0, y=last_pt, units="points"), **common))
    for t in (line1, *runs):
        fig.add_artist(t)
    ann.draw = types.MethodType(_draw_arrow_only, ann)      # the annotation keeps its text for its box; not drawn
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes_new = [R._box(t.get_window_extent(r), height_px, scale) for t in (line1, *runs)]
    return {"annotation": ann, "line1": line1, "runs": runs, "box_old": box_old, "boxes_new": boxes_new,
            "italic_file": Path(font_manager.findfont(italic)).name.lower(),
            "baseline_pt": last_pt, "pitch_pt": pitch_pt}


def draw(T: dict):
    """The compiled figure of the response to the reviewers, as report_zerok_figures.fig_compiled draws it (letters A and B).
    The plotted values are recorded under the names of the two single plots."""
    pad, gap = 3.0, 3.0
    y_b = pad + z19.PANEL_H + gap
    sheet = t8.Sheet(y_b + pad + z19.MATCHED_H)
    scratch = t5.Checks("dct_joint")
    z19.VALUES.clear()
    z19.RECORD = True
    z19.letter(sheet, 3.0, 1.0, "A")
    z19.draw_panel(z19.Sub(sheet, 0.0, pad), T, SINGLE[0], scratch)
    z19.letter(sheet, 3.0, y_b + 1.0, "B")
    z19.draw_matched(z19.Sub(sheet, 0.0, y_b + pad), T, SINGLE[1], scratch)
    z19.clean_ticks(sheet.fig)
    return sheet, scratch, pd.DataFrame(z19.VALUES)


def same_values(drawn: pd.DataFrame, stored: pd.DataFrame) -> tuple[int, int]:
    """Rows of the drawn values and how many of them equal the stored the response to the reviewers rows (text as text, numbers as numbers)."""
    key = ["figure", "panel", "row", "series"]
    a = drawn.assign(row=drawn["row"].astype(str))
    b = stored[stored["figure"].isin(SINGLE)].assign(row=lambda d: d["row"].astype(str))
    m = a.merge(b, on=key, how="outer", suffixes=("_drawn", "_stored"), indicator=True)
    ok = m["_merge"].eq("both")
    for col in ("value", "low", "high"):
        x = pd.to_numeric(m[f"{col}_drawn"], errors="coerce").to_numpy(float)
        s = pd.to_numeric(m[f"{col}_stored"], errors="coerce").to_numpy(float)
        number = np.isclose(x, s, rtol=1e-12, atol=1e-12) | (np.isnan(x) & np.isnan(s))
        text = m[f"{col}_drawn"].astype(str).to_numpy() == m[f"{col}_stored"].astype(str).to_numpy()
        ok &= np.where(np.isnan(x) & np.isnan(s), text | (m[f"{col}_drawn"].isna() & m[f"{col}_stored"].isna()), number)
    return len(m), int(ok.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Figure F-19 in the journal's style.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    R.stop_if_outputs_exist(run, {NAME: (".pdf", ".png", ".tif")}, (f"{STEP}_checks",))
    checks = t5.Checks("dct_joint")
    lib30.check_inputs(run, checks)          # plan hashes against the freeze record; frozen zerok_core panel hash
    rd = lambda n, **kw: lib30.read_table(run, n, **kw)  # noqa: E731
    panel = rd("panel_results")
    T = {"panel": panel[panel["status"] == "tested"], "spec_null": rd("specificity_matched_null"),
         "spec_draws": rd("specificity_null_draws_summary")}

    # --- the current figure, drawn again -------------------------------------------------------------------------------
    sheet, scratch, drawn = draw(T)
    fig = sheet.fig
    checks.add("the response to the reviewers checks of the two plots (cells per condition; proteins in the expected direction; matched "
               "random sets)", f"{sum(r['pass'] for r in scratch.rows)} of {len(scratch.rows)} pass", scratch.ok)
    old_rgb, old_png = R.read_image(run.use_input(OLD_PNG, "current Figure F-19 (PNG, 600 dpi)"))
    ref = R.opaque_rgb(R.render(fig, R.DPI_PNG), "reference drawing")
    same = ref.shape == old_rgb.shape
    n = int((ref != old_rgb).any(axis=2).sum()) if same else -1
    checks.add("drawing with the current strings equals the current PNG pixel for pixel (600 dpi)",
               f"{ref.shape[1]} x {ref.shape[0]} px; {n} pixels differ", same and n == 0, "0")
    stored = pd.read_csv(run.use_input(OLD_VALUES, "the response to the reviewers values table"))
    n_rows, n_same = same_values(drawn, stored)
    checks.add("plotted values equal the values the response to the reviewers stored for the two plots (rows; equal)",
               f"{n_rows}; {n_same}", n_rows == n_same == len(drawn) and n_rows > 0)

    # --- the strings ------------------------------------------------------------------------------------------------------
    letters = [t for t in fig.texts if t.get_text() in ("A", "B")]
    changes = R.edit_texts(fig, R.replace_exact(STRINGS))
    got = sorted((c["old"], c["new"]) for c in changes)
    checks.add("texts changed in the figure (old -> new)", R.changes_text(changes), got == sorted(STRINGS.items()),
               f"{len(STRINGS)} texts")
    p = italic_p(fig)
    checks.add("P value: lines written; font file of the P; baseline of the last line above the text bottom and line "
               "pitch (pt)", f"{p['line1'].get_text()!r}; {' + '.join(repr(t.get_text()) for t in p['runs'])}; "
               f"{p['italic_file']}; {p['baseline_pt']:.3f}; {p['pitch_pt']:.3f}",
               p["italic_file"] == "ariali.ttf" and "".join(t.get_text() for t in p["runs"]) == "empirical P = 0.0001")
    order = [t.get_text() for t in sorted(letters, key=lambda t: -t.get_position()[1])]
    checks.add("panel letters are lower case and in order from the top", " ".join(order), order == ["a", "b"], "a b")
    texts = [t for t in fig.findobj(Text) if t is not p["annotation"]]
    tc = R.text_layout(fig, texts)
    arial = all(f in ("arial.ttf", "arialbd.ttf", "ariali.ttf") for f in tc["fonts"].split(";"))
    joins = [(a, b) for a, b in zip(p["runs"][:-1], p["runs"][1:])]
    r = fig.canvas.get_renderer()
    gaps = [b.get_window_extent(r).x0 - a.get_window_extent(r).x1 for a, b in joins]
    lone = [o for o in tc["overlaps"].split("; ") if o and "empirical " not in o and "'P'" not in o]
    checks.add("format after the change (texts >= 7 pt, Arial, no text overlap other than the runs of the P-value "
               "line touching, no text outside the figure)",
               f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; {tc['fonts']}; overlaps {tc['n_text_overlaps']} "
               f"{tc['overlaps'][:120]}; outside {tc['n_text_outside_figure']}; gaps between the runs "
               f"{', '.join(f'{g:.2f}' for g in gaps)} px",
               tc["min_font_pt"] >= 7 and arial and not lone and tc["n_text_outside_figure"] == 0
               and all(abs(g) < 1.0 for g in gaps))

    # --- draw and write -------------------------------------------------------------------------------------------------
    new600 = R.opaque_rgb(R.render(fig, R.DPI_PNG), "600-dpi drawing")
    new1000 = R.opaque_rgb(R.render(fig, R.DPI_TIF), "1000-dpi drawing")
    desc = ("Figure F-19 in the journal's style: (a) pre-specified panel proteins with an expected direction, zero-K "
            "against control; (b) the 7 directional proteins against 10,000 matched random protein sets; values and "
            "layout of report_19_fig_F19_compiled")
    tif = run.out("figures", NAME, ".tif")
    R.save_tif(tif, new1000, desc)
    pdf, png = run.figure(fig, NAME, desc, formats=(".pdf", ".png"), tight=False)

    # --- checks on the written files -------------------------------------------------------------------------------------
    boxes = R.change_boxes(changes) + [p["box_old"], *p["boxes_new"]]
    d = R.diff_outside(old_rgb, new600, boxes)
    checks.add("new PNG against the current PNG: pixels that differ; of these outside the boxes of the changed texts "
               "(the arrow is outside these boxes)", f"{d['n_diff']} differ; {d['n_outside']} outside "
               f"{d['outside_box']}", d["same_size"] and d["n_outside"] == 0 and d["n_diff"] > 0, "0 outside")
    x0, y0, x1, y1 = p["boxes_new"][0]                       # "observed: 7 of 7": not changed, so its pixels stay
    n1 = int((old_rgb[y0 - 3:y1 + 3, x0 - 3:x1 + 3] != new600[y0 - 3:y1 + 3, x0 - 3:x1 + 3]).any(axis=2).sum())
    checks.add("line 'observed: 7 of 7' (not changed): pixels that differ from the current PNG in its box", n1,
               n1 == 0, "0")
    written, _ = R.read_image(png)
    checks.add("written PNG equals the 600-dpi drawing", f"{int((written != new600).any(axis=2).sum())} pixels differ",
               written.shape == new600.shape and bool((written == new600).all()), "0")
    old = R.pdf_facts(run.use_input(OLD_PDF, "current Figure F-19 (PDF)"))
    R.add_file_checks(checks, "F-19", pdf=pdf, png=png, tif=tif, size_mm=(old["width_mm"], old["height_mm"]),
                      png_px=(old_png["width_px"], old_png["height_px"]),
                      tif_px=(int(sheet.W / 25.4 * R.DPI_TIF), int(sheet.H / 25.4 * R.DPI_TIF)),
                      size_from="the size of the current figure")
    tif_rgb, _ = R.read_image(tif)
    checks.add("written TIFF equals the drawing made by matplotlib at dpi 1000",
               f"{int((tif_rgb != new1000).any(axis=2).sum())} pixels differ",
               tif_rgb.shape == new1000.shape and bool((tif_rgb == new1000).all()), "0")
    R.add_tif_png_check(checks, "F-19", tif_rgb, new600)
    new = R.pdf_facts(pdf)
    R.add_text_checks(checks, "F-19", old["lines"], new["lines"], EXPECTED_LINES, "the current PDF")
    names = sorted({n for n, _ in new["fonts"]})
    checks.add("PDF: Arial Italic is embedded (the P)", ", ".join(names),
               any(n == "Arial-ItalicMT" and e for n, e in new["fonts"]), "Arial-ItalicMT embedded")
    checks.add("PDF text: the panel letters as lines of their own; no line 'A' or 'B'",
               ", ".join(x for x in new["lines"] if x in ("a", "b", "A", "B")),
               [x for x in new["lines"] if x in ("a", "b", "A", "B")] == ["a", "b"], "a, b")

    run.table(checks.frame(), f"{STEP}_checks")
    n_fail = int((~checks.frame()["pass"]).sum())
    print(f"Checks: {len(checks.rows) - n_fail} passed or information, {n_fail} failed")
    run.finish({"figure": "F-19", "source_script": "scripts/report_zerok_figures.py (draw_panel, "
                "draw_matched, letter, Sub, clean_ticks imported)",
                "strings_changed": [f"{c['old']} -> {c['new']}" for c in changes] + ["empirical p = 0.0001 -> "
                                                                                    "empirical P = 0.0001 (italic P)"],
                "italic_p": "three runs on the baseline of the annotation's second line (Arial, Arial Italic, Arial); "
                            "no mathtext; the annotation draws its arrow only",
                "size_mm": [sheet.W, round(sheet.H, 3)], "png_dpi": R.DPI_PNG, "tif_dpi": R.DPI_TIF,
                "tif": "drawn by matplotlib at dpi 1000 (not enlarged from the 600-dpi image)",
                "new_statistics": "none", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
