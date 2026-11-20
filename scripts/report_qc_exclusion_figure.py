#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figure S4 - exclusion of the cells flagged on a mitochondrial metric
against 100 random removals, total kidney and DCT (control and zero-K side by side), as one assembled figure.

(reviewer 1, comment 1). The authors asked on 12 Nov 2026 for the
un-assembled panels (kidney_annotation_sensitivity_vs_random, dct_reviewer_exclusion_vs_random) to be redrawn as one
reader-facing figure ("Yes, redraw them"), following the re-render checklist of
results/tables/report/report_02_mito_qc.md, section 5. No analysis is rerun: the observed values, the
all-cell values and the median and central 95% of the random removals are read from the stored kidney_annotation and dct_reviewer summary
tables; the dots of the single random removals are read from the stored run tables.

What is drawn: the exclusion set the letter reports (cells flagged on any mitochondrial metric; kidney 20 of 712
under kidney_t15_label; DCT 128 of 729 control and 101 of 702 zero-K, union over the cluster annotation and the
marker-score label (clusters)), with every outcome the letter counts: 12 in kidney, 18 in control, 16 in zero-K
("11 of 12", "14 of 18", "15 of 16" inside the central 95% of the random removals). One row per outcome; all
outcomes except the number of cell types lie between 0 and 1.
  A  Total kidney, revised annotation (display names DCT, CNT/PC; 'Unknown' keeps its name).
  B  DCT, Control.    C  DCT, Zero-K (same rows; each condition on its own, nothing compares them).
     Rows are grouped by the label they use: cluster annotation (cs_ctrl_deg_label, cs_zerok_pipeline_label),
     marker-score label (clusters) (cs_<cond>_cluster_label), marker-score label (per cell) (cs_<cond>_cell_label).
     Outcomes that need a label absent from the cluster annotation are not rows (ProLIF) or are marked
     "does not apply" (DCT2 in zero-K), as stored (`applicable`).
Reader names (authors, 12 Nov 2026: "The readers will not know the revision process, so we only show the final
products"): no drawn text names the submission; the two marker-score labels carry the shared names above (each cell
takes the cell type whose marker proteins give the highest score; a cluster at Leiden resolution 3 takes the most
frequent label of its cells). A check stops the run if a drawn text holds "submitted", "submission" or
"re-annotation".
Marks: grey dots = single random removals; grey line and black tick = their central 95% and median; black circle =
flagged cells excluded, inside that range; orange diamond = outside (shape and colour); blue dashed tick = all cells.
Kidney cell types carry the colours of the revised Figure 2A (kidney_figure2 colour key).

Inputs (read-only): results/tables/kidney_annotation/kidney_annotation_sensitivity_{vs_random,runs,exclusion_sets}.csv,
                    kidney_annotation_cluster_labels_per_cell_a_endo.csv;
                    results/tables/dct_reviewer/dct_reviewer_exclusion_{vs_random,runs,sets}.csv;
                    results/tables/dct_labels/dct_labels_dct_canonical_labels.csv;
                    results/tables/kidney_figure2/kidney_figure2_fig2a_umap_key.csv
Outputs: results/figures/report/report_02_S02c_exclusion.{pdf,png};
         results/tables/report/report_02_s02c_figure_{values,checks}.csv;
         results/tables/report/report_step_02_s02c_figure_manifest.json

Command (project root):  python scripts/report_qc_exclusion_figure.py --overwrite
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

TASK = "report"
STEP = "02_s02c_figure"
P = "02_s02c_figure"
FIGURE = "02_S02c_exclusion"
KIDNEY_ANNOTATION = cc.OUT_ROOTS["tables"] / "kidney_annotation"
DCT_REVIEWER = cc.OUT_ROOTS["tables"] / lib28.TASK
KIDNEY_FIGURE2_KEY = cc.out_path("kidney_figure2", "tables", "fig2a_umap_key", ".csv")
SET = "mito_flagged"                                       # the exclusion set of the letter
KIDNEY_COL = "kidney_t15_label"
KIDNEY_DISPLAY = {"DCT1": "DCT", "PC": "CNT/PC"}           # authors, 30 Oct 2026
KIDNEY_COUNTS = {"PCT": 160, "PST": 158, "TAL": 138, "DCT1": 58, "PC": 38, "JG": 24, "Endothelial": 19, "IC-B": 8,
                 "Unknown": 109}                           # the project notes
KIDNEY_TYPES = ("Endothelial", "JG", "PCT", "PST", "TAL", "DCT1", "PC", "IC-B")   # order of the revised Figure 2A key
CONDITIONS = lib28.CONDITIONS
COND_DISPLAY = {"Ctrl": "Control", "ZeroK": "Zero-K"}
M, FONT, INK = T8.M, T8.FONT, T8.INK
BLUE, ORANGE = T8.SERIES[0], T8.SERIES[1]
ROW_H = 4.4
LETTER_PT = 8.0
DASH = "–"
# Rows: (outcome column or None for a group header, text). The texts are the reader names of the stored outcomes.
KIDNEY_ROWS = [
    ("agreement", "Cells keeping their label"),
    ("unknown_fraction", "Cells labelled Unknown"),
    ("ari_vs_baseline", "Clusters vs all-cell clusters (ARI)"),
    (None, "Cells keeping their label, by cell type"),
    *[(f"recall_{t}", t) for t in KIDNEY_TYPES],
]
KIDNEY_COUNT_ROW = ("n_types", "Cell types resolved (number)")
DCT_ROWS = [
    (None, "Cluster annotation"),
    ("deg_stored_agree", "Cells keeping their label, stored clusters"),
    ("deg_rerun_agree_s0", "Cells keeping their label, new clusters"),
    ("ari1_s0", "New vs stored clusters (ARI)"),
    ("paga_deg_stored_DCT1_DCT2", f"PAGA DCT1{DASH}DCT2, stored graph"),
    ("paga_deg_rebuilt_DCT1_DCT2", f"PAGA DCT1{DASH}DCT2, rebuilt graph"),
    (None, "Marker-score label (clusters)"),
    ("cluster_stored_agree", "Cells keeping their label, stored clusters"),
    ("cluster_rerun_agree_s0", "Cells keeping their label, new clusters"),
    ("ari3_s0", "New vs stored clusters (ARI)"),
    ("cluster_stored_prolif_share", "ProLIF cells (fraction), stored clusters"),
    ("cluster_rerun_prolif_share_s0", "ProLIF cells (fraction), new clusters"),
    (None, "Marker-score label (per cell)"),
    ("cell_agree", "Cells keeping their label"),
    ("cell_prolif_share", "ProLIF cells (fraction)"),
    *[(f"paga_stored_{a}_{b}", f"PAGA {a}{DASH}{b}, stored graph") for a, b in (("DCT1", "DCT2"), ("DCT1", "ProLIF"),
                                                                              ("DCT2", "ProLIF"))],
    *[(f"paga_rebuilt_{a}_{b}", f"PAGA {a}{DASH}{b}, rebuilt graph") for a, b in (("DCT1", "DCT2"), ("DCT1", "ProLIF"),
                                                                                ("DCT2", "ProLIF"))],
]
NOT_APPLICABLE_TEXT = "does not apply (no DCT2 cluster)"
# Numbers the the response to the reviewers letter quotes for this figure (section 2 of report_02_mito_qc.md).
LETTER = {
    "excluded": {"kidney": (20, 712), "Ctrl": (128, 729), "ZeroK": (101, 702)},
    "inside": {"kidney": (11, 12), "Ctrl": (14, 18), "ZeroK": (15, 16)},
    "outside": {
        "kidney": {"unknown_fraction": ("0.048", "0.074", "0.155")},
        "Ctrl": {"deg_rerun_agree_s0": ("0.661", "0.691", "0.964"), "paga_stored_DCT1_ProLIF": ("0.523", "0.608", "0.699"),
                 "paga_stored_DCT2_ProLIF": ("0.855", "0.718", "0.840"),
                 "paga_deg_stored_DCT1_DCT2": ("0.253", "0.182", "0.248")},
        "ZeroK": {"paga_stored_DCT1_ProLIF": ("0.483", "0.524", "0.599")},
    },
    "jg_recovered_in_random": 31,
    "kidney_agreement_percent": (81, 75, 96),
}
NOT_READER_FACING = re.compile(r"Ctrl|ZeroK|\bT\d{2}\b|primary|second label|_|Generated|Leiden|flagged set")
REVISION_PROCESS = re.compile(r"submitted|submission|re-?annotation", re.IGNORECASE)   # authors, 12 Nov 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: Supplementary Figure S-02-c (exclusion against random "
                                                 "removals, kidney and DCT).")
    cc.add_common_args(parser)
    return parser.parse_args()


def truthy(x) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def clean(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color(INK["muted"])
    ax.tick_params(axis="x", length=1.5, width=0.4, labelsize=FONT, colors=INK["secondary"], pad=1.5)
    ax.tick_params(axis="y", length=0, labelsize=FONT, labelcolor=INK["primary"], pad=2)


def rows_axis(ax, slots: list[dict], labels: bool) -> None:
    n = len(slots)
    data = [i for i, s in enumerate(slots) if s["outcome"]]
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(data)
    ax.set_yticklabels([slots[i]["text"] for i in data] if labels else [""] * len(data))
    for i in data:
        ax.axhline(i, color=INK["grid"], lw=0.3, zorder=0)
    clean(ax)


def unit_axis(ax) -> None:
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])


def draw_row(ax, i: int, r: dict, rng, x_jitter: float = 0.0) -> None:
    """One outcome: single random removals, their central 95% and median, the all-cell value and the observed value."""
    d = r["draws"]
    x = d + rng.uniform(-x_jitter, x_jitter, d.size) if x_jitter else d
    ax.scatter(x, i + rng.uniform(-0.28, 0.28, d.size), s=2.0, c=INK["muted"], alpha=0.55, linewidths=0, zorder=2)
    ax.plot([r["random_q025"], r["random_q975"]], [i, i], color=INK["secondary"], lw=1.0, zorder=3, solid_capstyle="butt")
    ax.plot([r["random_median"]] * 2, [i - 0.3, i + 0.3], color=INK["primary"], lw=1.1, zorder=4, solid_capstyle="butt")
    if np.isfinite(r["baseline"]):
        ax.plot([r["baseline"]] * 2, [i - 0.42, i + 0.42], color=BLUE, lw=0.8, ls=(0, (2, 1.2)), zorder=4)
    if r["inside"]:
        ax.plot([r["observed"]], [i], "o", ms=3.6, mfc=INK["primary"], mec="white", mew=0.4, zorder=5)
    else:
        ax.plot([r["observed"]], [i], "D", ms=3.8, mfc=ORANGE, mec="white", mew=0.4, zorder=5)


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)

    def rd(path, what: str, **kw) -> pd.DataFrame:
        return pd.read_csv(run.use_input(path, what), **kw)

    k_vs = rd(KIDNEY_ANNOTATION / "kidney_annotation_sensitivity_vs_random.csv", "kidney_annotation kidney exclusion against random removals (summary)")
    k_runs = rd(KIDNEY_ANNOTATION / "kidney_annotation_sensitivity_runs.csv", "kidney_annotation kidney exclusion: single runs")
    k_sets = rd(KIDNEY_ANNOTATION / "kidney_annotation_sensitivity_exclusion_sets.csv", "kidney_annotation kidney exclusion sets")
    k_lab = rd(KIDNEY_ANNOTATION / "kidney_annotation_cluster_labels_per_cell_a_endo.csv", "kidney_t15_label per cell (kidney_annotation)", dtype={"cell_id": str})
    d_vs = rd(DCT_REVIEWER / "dct_reviewer_exclusion_vs_random.csv", "dct_reviewer DCT exclusion against random removals (summary)")
    d_runs = rd(DCT_REVIEWER / "dct_reviewer_exclusion_runs.csv", "dct_reviewer DCT exclusion: single runs")
    d_sets = rd(DCT_REVIEWER / "dct_reviewer_exclusion_sets.csv", "dct_reviewer DCT exclusion sets")
    canon = rd(lib28.DCT_LABELS_LABELS, "dct_labels canonical DCT labels (combined search)", dtype={"cell_id": str},
               keep_default_na=False, na_values=[""])
    key = rd(KIDNEY_FIGURE2_KEY, "kidney_figure2 colour key of the revised Figure 2A")
    kidney_colours = dict(zip(key["label"], key["colour"]))

    checks: list[dict] = []
    values: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {observed}", flush=True)

    def f3(x: float) -> str:
        return f"{x:.3f}"

    # ------------------------------------------------------------------ kidney
    got = k_lab[KIDNEY_COL].astype(str).value_counts().to_dict()
    check("kidney: cells per label of the revised annotation equal the canonical counts (the project notes)", got, KIDNEY_COUNTS,
          got == KIDNEY_COUNTS)
    n_kidney = len(k_lab)
    k_base = k_runs[(k_runs["run"] == "baseline") & (k_runs["seed"] == 42)].iloc[0]
    k_obs = k_runs[(k_runs["exclusion_set"] == SET) & (k_runs["run"] == "observed") & (k_runs["seed"] == 42)].iloc[0]
    k_rnd = k_runs[(k_runs["exclusion_set"] == SET) & (k_runs["run"] == "random")]
    k_sum = k_vs[k_vs["exclusion_set"] == SET].set_index("outcome")
    n_type = {t: int(k_base[f"n_{t}"]) for t in KIDNEY_TYPES}
    check("kidney: cells per type in the run table (all cells) equal the canonical counts", n_type,
          {t: KIDNEY_COUNTS[t] for t in KIDNEY_TYPES}, n_type == {t: KIDNEY_COUNTS[t] for t in KIDNEY_TYPES})
    n_excl = int(k_sets[(k_sets["exclusion_set"] == SET) & (k_sets["label_column"] == KIDNEY_COL)]["n_excluded"].iloc[0])
    got = (n_excl, int(k_obs["n_removed"]), len(k_rnd), n_kidney)
    check("kidney: cells excluded (set table, observed run), random removals, cells", got, (20, 20, 100, 712),
          got == (20, 20, 100, 712) and (n_excl, n_kidney) == LETTER["excluded"]["kidney"])
    check("kidney: colours of the 8 cell types are in the revised Figure 2A key",
          {t: kidney_colours.get(t) for t in KIDNEY_TYPES}, "8 colours", all(t in kidney_colours for t in KIDNEY_TYPES))

    def kidney_row(outcome: str) -> dict:
        s = k_sum.loc[outcome]
        d = k_rnd[outcome].to_numpy(float)
        d = d[np.isfinite(d)]
        same = bool(np.allclose(np.quantile(d, [0.025, 0.5, 0.975]), [s["random_q025"], s["random_median"], s["random_q975"]])
                    and np.isclose(float(k_obs[outcome]), s["observed"]) and np.isclose(float(k_base[outcome]), s["baseline"])
                    and d.size == int(s["random_n"]))
        return {"outcome": outcome, "draws": d, "observed": float(s["observed"]), "baseline": float(s["baseline"]),
                "random_median": float(s["random_median"]), "random_q025": float(s["random_q025"]),
                "random_q975": float(s["random_q975"]), "inside": truthy(s["inside_central_95"]), "runs_match": same,
                "n_removed": int(s["n_removed"]), "applicable": True}

    k_slots = []
    for outcome, text in KIDNEY_ROWS:
        if outcome is None:
            k_slots.append({"outcome": None, "text": text})
            continue
        t = outcome.replace("recall_", "") if outcome.startswith("recall_") else None
        k_slots.append({"outcome": outcome, "type": t, "text": f"{KIDNEY_DISPLAY.get(t, t)} ({n_type[t]})" if t else text,
                        **kidney_row(outcome)})
    k_count = {"outcome": KIDNEY_COUNT_ROW[0], "type": None, "text": KIDNEY_COUNT_ROW[1], **kidney_row(KIDNEY_COUNT_ROW[0])}
    k_all = [s for s in k_slots if s["outcome"]] + [k_count]
    shown_types = [s["text"] for s in k_slots if s.get("type")]
    check("kidney: cell-type rows (display names DCT and CNT/PC; cells before exclusion)", shown_types,
          "Endothelial (19), JG (24), PCT (160), PST (158), TAL (138), DCT (58), CNT/PC (38), IC-B (8)",
          shown_types == ["Endothelial (19)", "JG (24)", "PCT (160)", "PST (158)", "TAL (138)", "DCT (58)", "CNT/PC (38)",
                          "IC-B (8)"])
    check("kidney: outcomes drawn = every outcome of the stored summary", f"{len(k_all)} of {len(k_sum)}", "12 of 12",
          len(k_all) == len(k_sum) == 12 and {s["outcome"] for s in k_all} == set(k_sum.index))
    check("kidney: single runs reproduce the stored observed, all-cell, median and central-95% values (12 outcomes)",
          sum(s["runs_match"] for s in k_all), 12, all(s["runs_match"] for s in k_all))
    inside = (sum(s["inside"] for s in k_all), len(k_all))
    out = {s["outcome"]: (f3(s["observed"]), f3(s["random_q025"]), f3(s["random_q975"])) for s in k_all if not s["inside"]}
    check("kidney: outcomes inside the central 95% equal the letter (11 of 12)", inside, LETTER["inside"]["kidney"],
          inside == LETTER["inside"]["kidney"])
    check("kidney: the outcome outside and its values equal the letter (Unknown 4.8% vs 7.4-15.5%)", out,
          LETTER["outside"]["kidney"], out == LETTER["outside"]["kidney"])
    jg = k_rnd["types"].astype(str).str.split(";").map(lambda t: "JG" in t)
    got = (float(k_obs["recall_JG"]), int(jg.sum()), int((~jg).sum()), int((k_rnd["recall_JG"] > 0).sum()))
    check("kidney: JG not recovered after exclusion; recovered in 31 and not in 69 of 100 random removals (letter)", got,
          (0.0, 31, 69, 31), got == (0.0, LETTER["jg_recovered_in_random"], 69, 31))
    a = next(s for s in k_all if s["outcome"] == "agreement")
    got = (int(f"{100 * a['observed']:.0f}"), int(f"{100 * a['random_q025']:.0f}"), int(f"{100 * a['random_q975']:.0f}"))
    check("kidney: cells keeping their label, %", got, LETTER["kidney_agreement_percent"],
          got == LETTER["kidney_agreement_percent"])

    # ------------------------------------------------------------------ DCT
    dct = {}
    for cond in CONDITIONS:
        col = lib28.SETTINGS[cond].PRIMARY
        got = canon[canon["condition"] == cond][col].astype(str).value_counts().to_dict()
        check(f"{cond}: cells per label of the cluster annotation equal the canonical counts (the project notes, {col})", got,
              lib28.EXPECTED_COUNTS[cond][col], got == lib28.EXPECTED_COUNTS[cond][col])
        st = d_sets[(d_sets["condition"] == cond) & (d_sets["exclusion_set"] == SET)].iloc[0]
        runs = d_runs[d_runs["condition"] == cond]
        base = runs[runs["run_type"] == "baseline"].iloc[0]
        obs = runs[(runs["run_type"] == "observed") & (runs["exclusion_set"] == SET)].iloc[0]
        rnd = runs[(runs["run_type"] == "random") & (runs["exclusion_set"] == SET)]
        got = (int(st["n_excluded"]), int(obs["n_removed"]), int(st["n_remaining"]), len(rnd), int(base["n_remaining"]))
        want = (LETTER["excluded"][cond][0], LETTER["excluded"][cond][0], 601, 100, LETTER["excluded"][cond][1])
        check(f"{cond}: cells excluded (set table, observed run), remaining, random removals, cells", got, want, got == want)
        summ = d_vs[(d_vs["condition"] == cond) & (d_vs["exclusion_set"] == SET)].set_index("outcome")
        rows = {}
        for outcome, _ in DCT_ROWS:
            if outcome is None:
                continue
            s = summ.loc[outcome]
            if not truthy(s["applicable"]):
                rows[outcome] = {"outcome": outcome, "applicable": False, "reason": str(s["reason"])}
                continue
            d = rnd[outcome].to_numpy(float)
            d = d[np.isfinite(d)]
            same = bool(np.allclose(np.quantile(d, [0.025, 0.5, 0.975]), [s["random_q025"], s["random_median"], s["random_q975"]])
                        and np.isclose(float(obs[outcome]), s["observed"]) and np.isclose(float(base[outcome]), s["baseline"]))
            rows[outcome] = {"outcome": outcome, "applicable": True, "draws": d, "observed": float(s["observed"]),
                             "baseline": float(s["baseline"]), "random_median": float(s["random_median"]),
                             "random_q025": float(s["random_q025"]), "random_q975": float(s["random_q975"]),
                             "inside": truthy(s["inside_central_95"]), "runs_match": same, "n_removed": int(s["n_removed"])}
        dct[cond] = rows
        app = [r for r in rows.values() if r["applicable"]]
        stored_app = summ[summ["applicable"].map(truthy) & summ["observed"].notna()]
        check(f"{cond}: outcomes drawn = every applicable outcome of the stored summary", f"{len(app)} of {len(stored_app)}",
              f"{LETTER['inside'][cond][1]} of {LETTER['inside'][cond][1]}",
              len(app) == len(stored_app) == LETTER["inside"][cond][1] and {r["outcome"] for r in app} == set(stored_app.index))
        check(f"{cond}: single runs reproduce the stored observed, all-cell, median and central-95% values",
              f"{sum(r['runs_match'] for r in app)} of {len(app)}", "all", all(r["runs_match"] for r in app))
        inside = (sum(r["inside"] for r in app), len(app))
        check(f"{cond}: outcomes inside the central 95% equal the letter", inside, LETTER["inside"][cond],
              inside == LETTER["inside"][cond])
        out = {r["outcome"]: (f3(r["observed"]), f3(r["random_q025"]), f3(r["random_q975"])) for r in app if not r["inside"]}
        check(f"{cond}: the outcomes outside and their values equal the letter", out, LETTER["outside"][cond],
              out == LETTER["outside"][cond])
        na = sorted(o for o, r in rows.items() if not r["applicable"])
        want_na = [] if cond == "Ctrl" else ["paga_deg_rebuilt_DCT1_DCT2", "paga_deg_stored_DCT1_DCT2"]
        check(f"{cond}: rows marked 'does not apply' (stored: label absent from the cluster annotation)", na or "none",
              want_na or "none", na == want_na and all("DCT2" in rows[o]["reason"] for o in na))
    d_slots = [{"outcome": o, "text": t} for o, t in DCT_ROWS]

    # ------------------------------------------------------------------ geometry (mm; y downwards)
    rng = np.random.default_rng(cc.RANDOM_SEED)
    lab_w = max(T8.text_mm(s["text"])[0] for s in [*k_slots, k_count, *d_slots] if s["outcome"])
    head_w = max(T8.text_mm(s["text"], weight="bold")[0] for s in [*k_slots, *d_slots] if not s["outcome"])
    x0 = M + max(lab_w, head_w - 6.0) + 2.0
    gap = 7.0
    w_ax = (T8.W - M - x0 - gap) / 2 - 0.5
    let_h, head_h = 3.6, 3.6
    ya = M
    ya_ax = ya + let_h + head_h + 0.8
    ha = len(k_slots) * ROW_H
    ya_n = ya_ax + ha + 6.4
    hn = ROW_H * 1.25
    yb = ya_n + hn + 4.4 + 4.6
    yb_ax = yb + let_h + head_h + 0.8
    hb = len(d_slots) * ROW_H
    H = yb_ax + hb + 4.4 + M
    sh = T8.Sheet(H)

    def letter(x: float, y: float, char: str, head: str) -> None:
        sh.text(x, y, char, fontsize=LETTER_PT, fontweight="bold", va="top")
        sh.text(x, y + let_h, head, va="top", color=INK["primary"])

    def group_heads(slots: list[dict], y_ax: float) -> None:
        for i, s in enumerate(slots):
            if not s["outcome"]:
                sh.text(M, y_ax + (i + 0.5) * ROW_H, s["text"], fontweight="bold", va="center")

    def record(dataset: str, panel: str, s: dict, text: str) -> None:
        base = {"dataset": dataset, "panel": panel, "row": text, "outcome": s["outcome"], "applicable": s["applicable"]}
        if not s["applicable"]:
            values.append({**base, "marker": NOT_APPLICABLE_TEXT, "note": s["reason"]})
            return
        values.append({**base, "n_removed": s["n_removed"], "all_cells": s["baseline"], "observed": s["observed"],
                       "random_median": s["random_median"], "random_q025": s["random_q025"],
                       "random_q975": s["random_q975"], "n_random_drawn": s["draws"].size,
                       "inside_central_95": s["inside"],
                       "marker": "black circle (inside)" if s["inside"] else "orange diamond (outside)"})

    # ------------------------------------------------------------------ A: kidney
    n_k, n_all = LETTER["excluded"]["kidney"]
    letter(x0, ya, "A", f"Total kidney: {n_k} of {n_all} cells excluded")
    group_heads(k_slots, ya_ax)
    ax = sh.ax(x0, ya_ax, w_ax, ha)
    rows_axis(ax, k_slots, labels=True)
    unit_axis(ax)
    for i, s in enumerate(k_slots):
        if not s["outcome"]:
            continue
        draw_row(ax, i, s, rng)
        record("total kidney", "A", s, s["text"])
        if s.get("type"):
            w = T8.text_mm(s["text"])[0]
            sh.rect(x0 - w - 4.2, ya_ax + (i + 0.5) * ROW_H - 1.1, 2.2, 2.2, fc=kidney_colours[s["type"]], ec="none")
    ax = sh.ax(x0, ya_n, w_ax, hn)
    ax.set_ylim(0.5, -0.5)
    ax.set_yticks([0])
    ax.set_yticklabels([k_count["text"]])
    ax.axhline(0, color=INK["grid"], lw=0.3, zorder=0)
    clean(ax)
    ax.set_xlim(6.4, 9.6)
    ax.set_xticks([7, 8, 9])
    draw_row(ax, 0, k_count, rng, x_jitter=0.2)
    record("total kidney", "A", k_count, k_count["text"])
    n_types = pd.Series(k_count["draws"]).astype(int).value_counts().sort_index().to_dict()
    check("kidney: number of cell types after exclusion 7 (all cells 8); random removals within the axis 7-9",
          f"observed {k_count['observed']:.0f}; all cells {k_count['baseline']:.0f}; random {n_types}",
          "7; 8; values 7, 8, 9", k_count["observed"] == 7 and k_count["baseline"] == 8 and set(n_types) <= {7, 8, 9})

    # key, to the right of A
    xk, yk = x0 + w_ax + gap + 1.0, ya_ax
    axk = sh.ax(xk, yk, 8.0, 5 * ROW_H)
    axk.set_xlim(0, 8)
    axk.set_ylim(5 - 0.5, -0.5)
    axk.axis("off")
    krng = np.random.default_rng(1)
    axk.plot([4], [0], "o", ms=3.6, mfc=INK["primary"], mec="white", mew=0.4)
    axk.plot([4], [1], "D", ms=3.8, mfc=ORANGE, mec="white", mew=0.4)
    axk.scatter(krng.uniform(1.5, 6.5, 9), 2 + krng.uniform(-0.25, 0.25, 9), s=2.0, c=INK["muted"], alpha=0.55, linewidths=0)
    axk.plot([1, 7], [3, 3], color=INK["secondary"], lw=1.0, solid_capstyle="butt")
    axk.plot([4, 4], [3 - 0.3, 3 + 0.3], color=INK["primary"], lw=1.1, solid_capstyle="butt")
    axk.plot([4, 4], [4 - 0.42, 4 + 0.42], color=BLUE, lw=0.8, ls=(0, (2, 1.2)))
    for j, text in enumerate(("flagged cells excluded: inside the range", "flagged cells excluded: outside the range",
                              "one of 100 random removals", "random removals: central 95%, median", "all cells")):
        sh.text(xk + 9.2, yk + (j + 0.5) * ROW_H, text, va="center", color=INK["secondary"])

    # ------------------------------------------------------------------ B, C: DCT
    group_heads(d_slots, yb_ax)
    for k, (char, cond) in enumerate(zip("BC", CONDITIONS)):
        x = x0 + k * (w_ax + gap)
        n_c, n_all = LETTER["excluded"][cond]
        letter(x, yb, char, f"DCT, {COND_DISPLAY[cond]}: {n_c} of {n_all} cells excluded")
        ax = sh.ax(x, yb_ax, w_ax, hb)
        rows_axis(ax, d_slots, labels=k == 0)
        unit_axis(ax)
        for i, s in enumerate(d_slots):
            if not s["outcome"]:
                continue
            r = dct[cond][s["outcome"]]
            group = next(d_slots[j]["text"] for j in range(i, -1, -1) if not d_slots[j]["outcome"])
            record(f"DCT, {COND_DISPLAY[cond]}", char, r, f"{group}: {s['text']}")
            if r["applicable"]:
                draw_row(ax, i, r, rng)
            else:
                ax.text(0.5, i, NOT_APPLICABLE_TEXT, ha="center", va="center", fontsize=FONT, color=INK["secondary"],
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})

    # ------------------------------------------------------------------ reader-facing text, format, save
    texts = [t.get_text() for t in sh.fig.findobj(Text) if t.get_visible() and t.get_text().strip()]   # drawn texts
    bad = sorted({t for t in texts if NOT_READER_FACING.search(t)})
    check("figure text is reader-facing (no Ctrl, ZeroK, task codes, column or file names)", bad or "none", "none", not bad)
    process = sorted({t for t in texts if REVISION_PROCESS.search(t)})
    check("no drawn text names the revision process ('submitted', 'submission', 're-annotation')", process or "none",
          "none", not process)
    heads = [s["text"] for s in d_slots if not s["outcome"]]
    check("row-group headings of B and C (shared names)", heads,
          ["Cluster annotation", "Marker-score label (clusters)", "Marker-score label (per cell)"],
          heads == ["Cluster annotation", "Marker-score label (clusters)", "Marker-score label (per cell)"]
          and all(h in texts for h in heads))
    unknown = [t for t in texts if "Unknown" in t]
    check("'Unknown' appears only in the kidney row (the DCT panels have no label of that name)", unknown,
          ["Cells labelled Unknown"], unknown == ["Cells labelled Unknown"])
    letters = sorted(t for t in texts if re.fullmatch(r"[A-C]", t))
    check("panel letters", "".join(letters), "ABC", letters == ["A", "B", "C"])
    fig_rows: list[dict] = []
    ok = T8.save_checked(run, sh, FIGURE, "Supplementary Figure S-02-c: exclusion of the cells flagged on a mitochondrial "
                         "metric against 100 random removals: total kidney (A), DCT control (B) and zero-K (C); from "
                         "the stored kidney_annotation and dct_reviewer tables", fig_rows)
    row = fig_rows[0]
    check("figure format (Arial, every text >= 7 pt, no text overlap, no text outside, 180 mm, PDF TrueType + 600-dpi PNG)",
          f"{row['width_mm']} x {row['height_mm']} mm; min font {row['min_font_pt']} pt; overlaps {row['n_text_overlaps']}; "
          f"outside {row['n_text_outside_figure']}; fonts {row['fonts']}; PNG {row.get('png_px')}", "pass", ok)
    check("figure height (one page with the legend)", f"{H:.1f} mm", "<= 230 mm", H <= 230)

    run.table(pd.DataFrame(values), f"{P}_values")
    run.table(pd.concat([pd.DataFrame(checks), pd.DataFrame(fig_rows)], ignore_index=True), f"{P}_checks")
    all_ok = all(c["pass"] for c in checks)
    run.finish({"figure": "Supplementary Figure S-02-c", "exclusion_set": SET, "kidney_label_column": KIDNEY_COL,
                "kidney_display": KIDNEY_DISPLAY, "dct_label_columns": {c: lib28.SETTINGS[c].PRIMARY for c in CONDITIONS},
                "condition_display": COND_DISPLAY, "width_mm": T8.W, "height_mm": round(H, 1),
                "kidney_rows": [o for o, _ in KIDNEY_ROWS if o] + [KIDNEY_COUNT_ROW[0]],
                "dct_rows": [o for o, _ in DCT_ROWS if o], "jitter_seed": cc.RANDOM_SEED, "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit(f"A check failed; see report_{P}_checks.csv.")


if __name__ == "__main__":
    main()
