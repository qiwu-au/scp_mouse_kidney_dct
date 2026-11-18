#!/usr/bin/env python
"""The response to the reviewers: Figure F-08 (RNA vs protein detection and concordance) in the journal's style.

(authors, 15 Nov 2026: "Yes, redraw"; the journal's "Checklist
for Style"). The figure of the response to the reviewers (report_rna_protein_concordance_figure) is drawn again by its own functions
(``panel_a``, ``legend``, ``panel_b`` of report_rna_protein_concordance_figure.py, imported), from the same kidney_rna tables, at
the same size. That drawing is compared with the current PNG pixel for pixel; then these strings are changed in the
drawn figure and nothing else:
  panel letters           A, B -> a, b
  axis label              "genes (n = 3,569)" -> "Genes (n = 3569)"          (sentence case; four digits closed up)
  axis values             "1,000", "2,000", "3,000" -> "1000", "2000", "3000"
  counts inside the bars  every count of four digits loses the comma (1,358; 1,082; 1,303)
  key                     "B:" -> "b:" and "A, rows:" -> "a, rows:"          (both name a panel)
"A, rows:" was not on the list of changes given for this script; it names panel a in the same way as "B:" names
panel b, so it follows the letters. To keep it in upper case, remove its entry from ``STRINGS`` and run again.
No new statistics; no value, colour, position or size is changed. The the response to the reviewers files are not written.

Inputs (read-only): results/tables/kidney_rna/kidney_rna_concordance_class_counts.csv, kidney_rna_paired_marker_rho.csv;
  results/tables/report/report_08_marker_concordance_values.csv (values check);
  results/figures/report/report_rna_protein_concordance_figure.{pdf,png} (comparison).
Outputs: results/figures/report/report_27_figF08_ki_style.pdf (vector), .png (600 dpi), .tif (1000 dpi, drawn
  by matplotlib at dpi 1000; RGB, LZW);
  results/tables/report/report_27_redraw_figF08_checks.csv;
  results/tables/report/report_step_27_redraw_figF08_manifest.json

Command (project root):  python scripts/report_figure3.py --overwrite
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
import math

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import report_rna_protein_concordance_figure as m08     # drawing functions of the response to the reviewers (sets the atlas_dct figure style)
import report_figure_style_lib as R

from matplotlib.text import Text

T8, plt = m08.T8, m08.plt
TASK = R.TASK
STEP = "27_redraw_figF08"
NAME = "27_figF08_ki_style"
FIG_DIR = cc.OUT_ROOTS["figures"] / TASK
OLD_PDF = FIG_DIR / "report_rna_protein_concordance_figure.pdf"
OLD_PNG = FIG_DIR / "report_rna_protein_concordance_figure.png"
OLD_VALUES = cc.OUT_ROOTS["tables"] / TASK / "report_08_marker_concordance_values.csv"

# whole strings of the current figure -> journal style (numbers of four digits are closed up before this lookup)
STRINGS = {
    "A": "a",
    "B": "b",
    "genes (n = 3569)": "Genes (n = 3569)",
    "B:": "b:",
    "A, rows:": "a, rows:",          # not on the list given for this script; see the docstring
}
# lines of the PDF text (pypdf) that change: old -> new
EXPECTED_LINES = [
    ("0 1,000 2,000 3,000", "0 1000 2000 3000"),
    ("genes (n = 3,569)", "Genes (n = 3569)"),
    ("1,358 584 856 176 453 142", "1358 584 856 176 453 142"),
    ("1,082 438 550 240 867 392", "1082 438 550 240 867 392"),
    ("616 207 219 233 1,303 991", "616 207 219 233 1303 991"),
    ("A", "a"),
    ("A, rows: detection threshold: a gene is detected in a modality if found in at least this fraction of the cells "
     "of at least one cell type",
     "a, rows: detection threshold: a gene is detected in a modality if found in at least this fraction of the cells "
     "of at least one cell type"),
    ("B: dot colour = class at ≥ 10%; no ρ: protein not quantified gene not in the atlas in neither data set",
     "b: dot colour = class at ≥ 10%; no ρ: protein not quantified gene not in the atlas in neither data set"),
    ("B", "b"),
]
EXPECTED_TEXTS = {("A", "a"): 1, ("B", "b"): 1, ("genes (n = 3,569)", "Genes (n = 3569)"): 1, ("B:", "b:"): 1,
                  ("A, rows:", "a, rows:"): 1, ("1,000", "1000"): 1, ("2,000", "2000"): 1, ("3,000", "3000"): 1,
                  ("1,358", "1358"): 1, ("1,082", "1082"): 1, ("1,303", "1303"): 1}


def journal_style(s: str) -> str:
    s = R.close_up(s)
    return STRINGS.get(s, s)


def draw(counts: pd.DataFrame, mk: pd.DataFrame, lo: float, hi: float):
    """The the response to the reviewers figure, laid out and drawn as report_rna_protein_concordance_figure.main does (letters A and B)."""
    M = m08.M
    let_h = T8.text_mm("B", 8.0, "bold")[1] + 0.8
    probe = T8.Sheet(400.0)
    ha, rows_a = m08.panel_a(probe, M + let_h, counts)
    hl = m08.legend(probe, 0.0, lo, hi)
    hb = math.ceil(len(mk) / 2) * 2.9 + 2 * m08.LH + 2.5
    plt.close(probe.fig)
    y_a = M + let_h
    y_leg = y_a + ha + 2.5
    y_b_letter = y_leg + hl + 1.5
    y_b = y_b_letter + let_h
    sh = T8.Sheet(y_b + hb + M)
    letter_a = sh.text(M, M, "A", fontsize=8.0, fontweight="bold", va="top")
    m08.panel_a(sh, y_a, counts)
    m08.legend(sh, y_leg, lo, hi)
    letter_b = sh.text(M, y_b_letter, "B", fontsize=8.0, fontweight="bold", va="top")
    m08.panel_b(sh, y_b, mk, lo, hi)
    return sh, rows_a, (letter_a, letter_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Figure F-08 in the journal's style.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    R.stop_if_outputs_exist(run, {NAME: (".pdf", ".png", ".tif")}, (f"{STEP}_checks",))
    checks = t5.Checks("kidney")

    # --- inputs: as report_rna_protein_concordance_figure.main ---------------------------------------------------------
    counts = pd.read_csv(run.use_input(m08.CLASS_COUNTS, "kidney_rna concordance class counts"))
    counts = counts[(counts["rna_source"] == "all") & counts["label_column"].isin(m08.LABELS)]
    rho = pd.read_csv(run.use_input(m08.MARKER_RHO, "kidney_rna per-marker rho"))
    mk = rho[rho["label_column"] == "kidney_t15_label"].sort_values(["type_pos", "wb_pos"]).reset_index(drop=True)
    mk = mk.rename(columns={"class": "cls"})
    mk["row_label"] = mk["listed_for"] + " | " + mk["marker"]
    lo, hi = float(mk["chance_q025"].iloc[0]), float(mk["chance_q975"].iloc[0])

    # --- the current figure, drawn again -------------------------------------------------------------------------------
    sh, rows_a, letters = draw(counts, mk, lo, hi)
    fig = sh.fig
    old_rgb, old_png = R.read_image(run.use_input(OLD_PNG, "current Figure F-08 (PNG, 600 dpi)"))
    ref = R.opaque_rgb(R.render(fig, R.DPI_PNG), "reference drawing")
    same = ref.shape == old_rgb.shape
    n = int((ref != old_rgb).any(axis=2).sum()) if same else -1
    checks.add("drawing with the current strings equals the current PNG pixel for pixel (600 dpi)",
               f"{ref.shape[1]} x {ref.shape[0]} px; {n} pixels differ", same and n == 0, "0")

    # values drawn = values the the response to the reviewers script stored
    stored = pd.read_csv(run.use_input(OLD_VALUES, "the response to the reviewers values table"))
    sa = stored[stored["panel"] == "A"].reset_index(drop=True)
    ra = pd.DataFrame(rows_a)
    ok_a = (len(sa) == len(ra) == 18 and (sa["class"] == ra["class"]).all()
            and np.allclose(sa["threshold"], ra["threshold"]) and (sa["n_genes"].astype(int) == ra["n_genes"]).all()
            and (sa["labelled"].astype(bool) == ra["labelled"]).all())
    checks.add("panel a: classes, thresholds, gene counts and which bars carry a count equal the stored the response to the reviewers "
               "values", f"{len(ra)} bars; {int(ra['labelled'].sum())} with a count; equal {bool(ok_a)}", bool(ok_a))
    sb = stored[stored["panel"] == "B"].reset_index(drop=True)
    x_new = np.where(mk["coverage"] == "protein and atlas", mk["spearman_rho"], m08.NO_RHO_X).astype(float)
    ok_b = (len(sb) == len(mk) == 98 and (sb["marker"] == mk["marker"]).all()
            and np.allclose(sb["plotted_x"].to_numpy(float), x_new, rtol=0, atol=1e-12)
            and (sb["class"].fillna("") == mk["cls"].fillna("")).all())
    checks.add("panel b: markers, plotted x and classes equal the stored the response to the reviewers values",
               f"{len(mk)} markers; equal {bool(ok_b)}", bool(ok_b))

    # --- the strings ------------------------------------------------------------------------------------------------------
    changes = R.edit_texts(fig, journal_style)
    got = {}
    for c in changes:
        got[(c["old"], c["new"])] = got.get((c["old"], c["new"]), 0) + 1
    checks.add("texts changed in the figure (old -> new)", R.changes_text(changes), got == EXPECTED_TEXTS,
               f"{sum(EXPECTED_TEXTS.values())} texts")
    order = [t.get_text() for t in sorted(letters, key=lambda t: -t.get_position()[1])]
    checks.add("panel letters are lower case and in order from the top", " ".join(order), order == ["a", "b"], "a b")
    texts = [t for t in fig.findobj(Text)]
    tc = R.text_layout(fig, texts)
    arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
    checks.add("format after the change (texts >= 7 pt, Arial, no text overlap, no text outside the figure)",
               f"{tc['n_texts']} texts; min {tc['min_font_pt']} pt; {tc['fonts']}; overlaps {tc['n_text_overlaps']} "
               f"{tc['overlaps'][:100]}; outside {tc['n_text_outside_figure']} {tc['outside'][:60]}",
               tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0)

    # --- draw and write -------------------------------------------------------------------------------------------------
    new600 = R.opaque_rgb(R.render(fig, R.DPI_PNG), "600-dpi drawing")
    new1000 = R.opaque_rgb(R.render(fig, R.DPI_TIF), "1000-dpi drawing")
    desc = ("Figure F-08 in the journal's style: RNA vs protein detection and concordance classes (a) and per-marker "
            "rho with coverage (b); values and layout of report_rna_protein_concordance_figure")
    tif = run.out("figures", NAME, ".tif")
    R.save_tif(tif, new1000, desc)
    pdf, png = run.figure(fig, NAME, desc, formats=(".pdf", ".png"), tight=False)

    # --- checks on the written files -------------------------------------------------------------------------------------
    d = R.diff_outside(old_rgb, new600, R.change_boxes(changes))
    checks.add("new PNG against the current PNG: pixels that differ; of these outside the boxes of the changed texts",
               f"{d['n_diff']} differ; {d['n_outside']} outside {d['outside_box']}", d["same_size"]
               and d["n_outside"] == 0 and d["n_diff"] > 0, "0 outside")
    written, _ = R.read_image(png)
    checks.add("written PNG equals the 600-dpi drawing", f"{int((written != new600).any(axis=2).sum())} pixels differ",
               written.shape == new600.shape and bool((written == new600).all()), "0")
    old = R.pdf_facts(run.use_input(OLD_PDF, "current Figure F-08 (PDF)"))
    R.add_file_checks(checks, "F-08", pdf=pdf, png=png, tif=tif, size_mm=(old["width_mm"], old["height_mm"]),
                      png_px=(old_png["width_px"], old_png["height_px"]),
                      tif_px=(int(sh.W / 25.4 * R.DPI_TIF), int(sh.H / 25.4 * R.DPI_TIF)),
                      size_from="the size of the current figure")
    tif_rgb, _ = R.read_image(tif)
    checks.add("written TIFF equals the drawing made by matplotlib at dpi 1000",
               f"{int((tif_rgb != new1000).any(axis=2).sum())} pixels differ",
               tif_rgb.shape == new1000.shape and bool((tif_rgb == new1000).all()), "0")
    R.add_tif_png_check(checks, "F-08", tif_rgb, new600)
    new = R.pdf_facts(pdf)
    R.add_text_checks(checks, "F-08", old["lines"], new["lines"], EXPECTED_LINES, "the current PDF")
    checks.add("PDF text: lines made of one panel letter", ", ".join(x for x in new["lines"] if len(x) == 1),
               [x for x in new["lines"] if len(x) == 1 and x.isalpha()] == ["a", "b"], "a, b")

    run.table(checks.frame(), f"{STEP}_checks")
    n_fail = int((~checks.frame()["pass"]).sum())
    print(f"Checks: {len(checks.rows) - n_fail} passed or information, {n_fail} failed")
    run.finish({"figure": "F-08", "source_script": "scripts/report_rna_protein_concordance_figure.py (panel_a, "
                "legend, panel_b imported)", "strings_changed": [f"{c['old']} -> {c['new']}" for c in changes],
                "beyond_the_list": "'A, rows:' -> 'a, rows:' (names panel a, as 'B:' names panel b)",
                "size_mm": [sh.W, round(sh.H, 3)], "png_dpi": R.DPI_PNG, "tif_dpi": R.DPI_TIF,
                "tif": "drawn by matplotlib at dpi 1000 (not enlarged from the 600-dpi image)",
                "new_statistics": "none", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed_with_failed_checks")
    if not checks.ok:
        raise SystemExit(f"A check failed; see report_{STEP}_checks.csv.")


if __name__ == "__main__":
    main()
