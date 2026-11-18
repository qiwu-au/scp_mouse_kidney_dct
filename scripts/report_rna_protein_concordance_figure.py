#!/usr/bin/env python
"""The response to the reviewers: main Figure F-08 (formerly Supplementary Figure S-08-d), RNA vs protein detection and
concordance.

Built at the authors' request (30 Oct 2026) to replace
rna_protein_integration_strengths_t15 in S-08-d, whose colours were reused for different categories; made a main figure by the
authors later the same day (placeholder F-08; the response to the reviewers numbers it). Panel letters are upper case, as in the submitted
Figures 1-4. Drawn only from existing kidney_rna tables; no new statistics.

Panels (one page, 180 mm wide; no in-figure title or footnote, the legend text is in the the response to the reviewers file):
A. Detection and concordance classes of the 3,569 genes quantified in both the SCP-MS data and the Mouse Kidney
   Atlas, at detection thresholds of 5%, 10% and 20%, for the revised annotation (kidney_t15_label; 7 cell types;
   the only kidney annotation, authors 30 Oct 2026); all atlas cells.
B. Every kidney V3 + endothelial-row marker (98; each once, under its first listing type in Supplementary Table 1):
   Spearman rho of mean RNA vs mean protein across the 7 cell types of the revised annotation (x), coloured by its
   class at 10% (same colours as in A). Markers that are not in both data sets have no rho and are drawn as hollow
   shapes in the left column (protein not quantified; gene not in the atlas; neither). Grey band: central 95% of
   rho for unrelated rankings of 7 cell types (from the kidney_rna table).

Colours: one colour per class, used for nothing else. The three 'detected in both' classes are one blue ordinal
ramp (dataviz reference steps 650 / 450 / 250; validate_palette.js --ordinal: pass); RNA only and protein only are
categorical slots 2 and 3 (slots 1-3 pass all-pairs); detected in neither is the neutral baseline grey.

Inputs (read-only): results/tables/kidney_rna/kidney_rna_concordance_class_counts.csv,
                    results/tables/kidney_rna/kidney_rna_paired_marker_rho.csv
Outputs: results/figures/report/report_rna_protein_concordance_figure.{pdf,png};
         results/tables/report/report_08_marker_concordance_{values,checks,figure_checks}.csv;
         results/tables/report/report_step_08_marker_concordance_manifest.json

Command (project root):  python scripts/report_rna_protein_concordance_figure.py
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
import atlas_dct_lib as T8

plt = T8.setup_matplotlib()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

TASK = "report"
STEP = "08_marker_concordance"
KIDNEY_RNA_DIR = cc.OUT_ROOTS["tables"] / "kidney_rna"
CLASS_COUNTS = KIDNEY_RNA_DIR / "kidney_rna_concordance_class_counts.csv"
MARKER_RHO = KIDNEY_RNA_DIR / "kidney_rna_paired_marker_rho.csv"
THR = 0.10
THRESHOLDS = (0.05, 0.10, 0.20)
N_SHARED = 3569
# one kidney annotation (authors, 30 Oct 2026); 8 types, of which JG has no atlas counterpart and is not a cell type here
LABELS = {"kidney_t15_label": "The 7 cell types with an atlas counterpart (all except JG)"}   # authors, 30 Oct 2026
W, M, FONT = T8.W, T8.M, T8.FONT
INK = T8.INK
SURFACE = INK["surface"]
BAND = "#f0efec"                     # dataviz neutral midpoint grey (chance band)
# kidney_rna class -> (legend text, fill, ink of a count printed inside the fill)
CLASSES = {
    "Detected in both; concordant": ("concordant (ρ ≥ 0.5)", "#104281", "#ffffff"),
    "Detected in both; intermediate": ("intermediate (0.2 ≤ ρ < 0.5)", "#2a78d6", "#ffffff"),
    "Detected in both; discordant": ("discordant (ρ < 0.2)", "#86b6ef", INK["primary"]),
    "RNA-detected only": ("RNA only", "#eb6834", "#ffffff"),
    "Protein-detected only": ("protein only", "#1baf7a", INK["primary"]),
    "Detected in neither": ("neither", "#c3c2b7", INK["primary"]),
}
# kidney_rna coverage of markers without a rho -> (legend text, matplotlib marker)
NOT_BOTH = {
    "atlas only (protein not quantified)": ("protein not quantified", "v"),
    "protein only (gene not in atlas)": ("gene not in the atlas", "s"),
    "neither": ("in neither data set", "x"),
}
# counts stated in the kidney_rna README (step 5, marker table; step 1 '98; 53; 87; 51') and in the the response to the reviewers letter
EXPECTED_MARKER_CLASSES = {"Detected in both; concordant": 32, "Detected in both; intermediate": 5,
                           "Detected in both; discordant": 3, "RNA-detected only": 5, "Protein-detected only": 3,
                           "Detected in neither": 3}
EXPECTED_NOT_BOTH = {"atlas only (protein not quantified)": 36, "protein only (gene not in atlas)": 2, "neither": 9}
EXPECTED_KIDNEY_ANNOTATION_10 = {"Detected in both; concordant": 1082, "Detected in both; intermediate": 438,
                   "Detected in both; discordant": 550, "RNA-detected only": 240, "Protein-detected only": 867,
                   "Detected in neither": 392}
NO_RHO_X, SEP_X, XMIN, XMAX = -1.32, -1.16, -1.48, 1.08
LH = T8.text_mm("Hg", FONT)[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: main Figure F-08, marker concordance.")
    cc.add_common_args(parser)
    return parser.parse_args()


def tw(s: str, size: float = FONT, weight: str = "normal") -> float:
    return T8.text_mm(s, size, weight)[0]


def thr_text(t: float) -> str:
    return f"≥ {round(t * 100)}%"


def clean(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK["muted"])
    ax.tick_params(axis="x", length=1.5, width=0.4, labelsize=FONT, colors=INK["secondary"])
    ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"])


def panel_a(sh, y0: float, counts: pd.DataFrame) -> tuple[float, list[dict]]:
    """Stacked class bars per threshold, one block per label set; returns the height used and plotted values."""
    rh, bh = 4.6, 3.4
    lab_w = max(tw(thr_text(t)) for t in THRESHOLDS) + 1.5
    gap = 8.0
    pw = (W - 2 * M - len(LABELS) * lab_w - (len(LABELS) - 1) * gap) / len(LABELS)
    sub_h = LH + 1.3
    body = rh * len(THRESHOLDS)
    rows = []
    for k, col in enumerate(LABELS):
        x0 = M + lab_w + k * (pw + gap + lab_w)
        sh.text(x0, y0, LABELS[col], va="top")
        ax = sh.ax(x0, y0 + sub_h, pw, body)
        for i, t in enumerate(THRESHOLDS):
            sub = counts[(counts["label_column"] == col) & np.isclose(counts["threshold"], t)]
            n_by = sub.set_index("class")["n_features"]
            left = 0
            for cls, (_, fill, ink) in CLASSES.items():
                n = int(n_by.get(cls, 0))
                ax.barh(i, n, left=left, height=bh / rh, color=fill, edgecolor=SURFACE, linewidth=0.6, zorder=2)
                seg_mm = n / N_SHARED * pw
                if seg_mm >= tw(f"{n:,}") + 1.0:
                    ax.text(left + n / 2, i, f"{n:,}", ha="center", va="center", color=ink, fontsize=FONT, zorder=3)
                rows.append({"panel": "A", "label_column": col, "threshold": t, "class": cls, "n_genes": n,
                             "labelled": bool(seg_mm >= tw(f"{n:,}") + 1.0)})
                left += n
        ax.set_xlim(0, N_SHARED)
        ax.set_ylim(len(THRESHOLDS) - 0.5, -0.5)
        ax.set_yticks(range(len(THRESHOLDS)), [thr_text(t) for t in THRESHOLDS])
        ax.set_xticks([0, 1000, 2000, 3000], ["0", "1,000", "2,000", "3,000"])
        ax.set_xlabel(f"genes (n = {N_SHARED:,})", fontsize=FONT, labelpad=1.5, color=INK["secondary"])
        clean(ax)
    return sub_h + body + 2 * LH + 2.5, rows


def legend(sh, y0: float, lo: float, hi: float) -> float:
    """Legend lines (panel A thresholds; classes; panel B symbols), wrapped to the page width; returns the height
    used. The threshold line was added at the authors' request (30 Oct 2026)."""
    items0 = [("prefix", "A, rows:"), ("text", "detection threshold: a gene is detected in a modality if found in at "
                                                "least this fraction of the cells of at least one cell type")]
    items1 = [("prefix", "Detected in both:")] + [("class", c) for c in list(CLASSES)[:3]] + \
             [("prefix", "Detected in:")] + [("class", c) for c in list(CLASSES)[3:]]
    band = f"ρ range of unrelated rankings (central 95%, {lo:.2f} to {hi:.2f})".replace("-", "−")
    items2 = [("prefix", "B:"), ("text", f"dot colour = class at {thr_text(THR)};"), ("prefix", "no ρ:")] + \
             [("shape", c) for c in NOT_BOTH] + [("band", band)]
    y = y0
    for items in (items0, items1, items2):
        x = M
        for kind, key in items:
            if kind in ("prefix", "text"):
                txt, w_sym = key, 0.0
            elif kind == "class":
                txt, w_sym = CLASSES[key][0], 2.6
            elif kind == "shape":
                txt, w_sym = NOT_BOTH[key][0], 2.6
            else:
                txt, w_sym = key, 3.6
            w_item = w_sym + tw(txt)
            if x + w_item > W - M:
                x, y = M, y + LH + 1.6
            yc = y + LH / 2 + 0.3
            if kind == "class":
                sh.fig.add_artist(Line2D([(x + 1.0) / W], [1 - yc / sh.H], marker="s", ms=4.2, lw=0,
                                         mfc=CLASSES[key][1], mec=INK["secondary"], mew=0.3))
            elif kind == "shape":
                mk = NOT_BOTH[key][1]
                sh.fig.add_artist(Line2D([(x + 1.0) / W], [1 - yc / sh.H], marker=mk, ms=3.6 if mk != "x" else 3.2,
                                         lw=0, mfc="white", mec=INK["secondary"], mew=0.6))
            elif kind == "band":
                sh.rect(x, yc - 1.1, 2.8, 2.2, facecolor=BAND, edgecolor=INK["grid"], linewidth=0.4)
            sh.text(x + w_sym, yc, txt, va="center_baseline",
                    color=INK["secondary"] if kind == "prefix" else INK["primary"])
            x += w_item + (2.0 if kind == "prefix" else 4.0)
        y += LH + 1.6
    return y - y0


def panel_b(sh, y0: float, mk: pd.DataFrame, lo: float, hi: float) -> float:
    """Marker rho plot in two columns; returns the height used."""
    half = math.ceil(len(mk) / 2)
    parts = [mk.iloc[:half], mk.iloc[half:]]
    rh = 2.9
    lab_w = max(tw(s) for s in mk["row_label"]) + 1.0
    gap = 5.0
    pw = (W - 2 * M - 2 * lab_w - gap) / 2
    body = half * rh
    for k, part in enumerate(parts):
        x0 = M + lab_w + k * (pw + gap + lab_w)
        ax = sh.ax(x0, y0, pw, body)
        ax.set_ylim(half - 0.5, -0.5)
        ax.set_xlim(XMIN, XMAX)
        ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
        for v in (0.2, 0.5):
            ax.axvline(v, color=INK["muted"], lw=0.5, ls=":", zorder=1)
        ax.axvline(0, color=INK["muted"], lw=0.4, zorder=1)
        ax.axvline(SEP_X, color=INK["grid"], lw=0.6, zorder=1)
        for i, r in enumerate(part.itertuples(index=False)):
            ax.axhline(i, color=INK["grid"], lw=0.3, zorder=0.5)
            if r.coverage == "protein and atlas":
                ax.plot(r.spearman_rho, i, "o", ms=3.6, mfc=CLASSES[r.cls][1], mec=INK["secondary"], mew=0.3,
                        zorder=3)
            else:
                shape = NOT_BOTH[r.coverage][1]
                ax.plot(NO_RHO_X, i, shape, ms=3.4 if shape != "x" else 3.0, mfc="white", mec=INK["secondary"],
                        mew=0.6, zorder=3)
        ax.set_yticks(range(len(part)), part["row_label"])
        ax.set_xticks([NO_RHO_X, -0.5, 0, 0.5, 1], ["no ρ", "−0.5", "0", "0.5", "1"])
        ax.set_xlabel("Spearman ρ, RNA vs protein across the 7 cell types", fontsize=FONT, labelpad=1.5,
                      color=INK["secondary"])
        clean(ax)
    return body + 2 * LH + 2.5


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)
    counts = pd.read_csv(run.use_input(CLASS_COUNTS, "kidney_rna concordance class counts (kidney_rna_concordance.py)"))
    counts = counts[(counts["rna_source"] == "all") & counts["label_column"].isin(LABELS)]
    rho = pd.read_csv(run.use_input(MARKER_RHO, "kidney_rna per-marker rho (kidney_rna_paired_figures.py)"))
    mk = rho[rho["label_column"] == "kidney_t15_label"].sort_values(["type_pos", "wb_pos"]).reset_index(drop=True)
    mk = mk.rename(columns={"class": "cls"})
    mk["row_label"] = mk["listed_for"] + " | " + mk["marker"]
    lo, hi = float(mk["chance_q025"].iloc[0]), float(mk["chance_q975"].iloc[0])

    checks: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {observed}")

    check("markers plotted (unique, revised annotation)", f"{len(mk)} rows, {mk['marker'].nunique()} unique",
          "98 rows, 98 unique", len(mk) == 98 and mk["marker"].nunique() == 98)
    both = mk[mk["coverage"] == "protein and atlas"]
    obs = both["cls"].value_counts().to_dict()
    check("markers in both data sets by class (10%)", obs, EXPECTED_MARKER_CLASSES, obs == EXPECTED_MARKER_CLASSES)
    obs = mk.loc[mk["coverage"] != "protein and atlas", "coverage"].value_counts().to_dict()
    check("markers not in both data sets by coverage", obs, EXPECTED_NOT_BOTH, obs == EXPECTED_NOT_BOTH)
    check("every marker in both data sets has a rho in [-1, 1]",
          f"{int(both['spearman_rho'].between(-1, 1).sum())} of {len(both)}", len(both),
          bool(both["spearman_rho"].between(-1, 1).all()))
    check("no rho for markers not in both data sets",
          f"{int(mk.loc[mk['coverage'] != 'protein and atlas', 'spearman_rho'].notna().sum())} with a rho", 0,
          bool(mk.loc[mk["coverage"] != "protein and atlas", "spearman_rho"].isna().all()))
    check("all marker classes and coverages have a colour or shape",
          sorted(set(both["cls"]) - set(CLASSES)) + sorted(set(mk["coverage"]) - set(NOT_BOTH) - {"protein and atlas"}),
          "[]", set(both["cls"]) <= set(CLASSES) and set(mk["coverage"]) <= set(NOT_BOTH) | {"protein and atlas"})
    sums = counts.groupby(["label_column", "threshold"])["n_features"].sum()
    check("panel a: classes sum to the shared genes per label set and threshold", sorted(set(sums.astype(int))),
          [N_SHARED], bool((sums == N_SHARED).all()) and len(sums) == len(THRESHOLDS) * len(LABELS))
    t15 = counts[(counts["label_column"] == "kidney_t15_label") & np.isclose(counts["threshold"], THR)]
    obs = t15.set_index("class")["n_features"].astype(int).to_dict()
    check("panel a: revised annotation at 10% (letter numbers)", obs, EXPECTED_KIDNEY_ANNOTATION_10, obs == EXPECTED_KIDNEY_ANNOTATION_10)
    check("chance band (central 95% of rho, 7 cell types)", f"{lo:.2f} to {hi:.2f}", "-0.75 to 0.75",
          bool(np.isclose(lo, -0.75) and np.isclose(hi, 0.75)))

    # layout (mm, y downwards); heights are measured first, the sheet is then drawn at that size
    let_h = T8.text_mm("B", 8.0, "bold")[1] + 0.8
    probe = T8.Sheet(400.0)
    ha, rows_a = panel_a(probe, M + let_h, counts)
    hl = legend(probe, 0.0, lo, hi)
    hb = math.ceil(len(mk) / 2) * 2.9 + 2 * LH + 2.5
    plt.close(probe.fig)
    y_a = M + let_h
    y_leg = y_a + ha + 2.5
    y_b_letter = y_leg + hl + 1.5
    y_b = y_b_letter + let_h
    H = y_b + hb + M
    sh = T8.Sheet(H)
    sh.text(M, M, "A", fontsize=8.0, fontweight="bold", va="top")
    panel_a(sh, y_a, counts)
    legend(sh, y_leg, lo, hi)
    sh.text(M, y_b_letter, "B", fontsize=8.0, fontweight="bold", va="top")
    panel_b(sh, y_b, mk, lo, hi)

    fig_rows: list[dict] = []
    ok = T8.save_checked(run, sh, STEP, "Figure F-08 (main): RNA vs protein detection and concordance classes (A) and "
                         "per-marker rho with coverage (B); revised annotation; from kidney_rna tables", fig_rows)
    check("figure passes the format checks (>= 7 pt, Arial, no text overlaps, text inside, 180 mm)",
          fig_rows[0]["pass"], True, ok)

    vals_b = mk[["marker", "listed_for", "row_label", "coverage", "cls", "spearman_rho", "max_rna_detection",
                 "max_protein_detection"]].rename(columns={"cls": "class"})
    vals_b.insert(0, "panel", "B")
    vals_b["plotted_x"] = np.where(vals_b["coverage"] == "protein and atlas", vals_b["spearman_rho"], NO_RHO_X)
    vals_b["colour_or_shape"] = [CLASSES[c][1] if cov == "protein and atlas" else NOT_BOTH[cov][1]
                                 for c, cov in zip(vals_b["class"], vals_b["coverage"])]
    run.table(pd.concat([pd.DataFrame(rows_a), vals_b], ignore_index=True), f"{STEP}_values")
    run.table(pd.DataFrame(fig_rows), f"{STEP}_figure_checks")
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    all_ok = all(c["pass"] for c in checks)
    run.finish({"threshold": THR, "thresholds_panel_a": list(THRESHOLDS), "rna_source": "all",
                "markers_label_column": "kidney_t15_label", "chance_band": [lo, hi],
                "colours": {k: v[1] for k, v in CLASSES.items()}, "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit("A check failed; see report_08_marker_concordance_checks.csv.")


if __name__ == "__main__":
    main()
