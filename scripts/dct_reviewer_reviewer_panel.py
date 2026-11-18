#!/usr/bin/env python
"""dct_reviewer step 1b: reviewer_core reviewer summary panel (panels A-D) per DCT condition and label role.

Calls the ctrl_reviewer_core step-1b functions
(``ctrl_reviewer_core_reviewer_panel``: marker_composition, cell_cycle_summary, concentration; unchanged)
per condition; ``ambiguity`` is the copy in ``dct_reviewer_lib`` (the ctrl_reviewer_core function divides by the label
size and fails for an absent core label).

Inputs (read-only): dct_reviewer_task3_marker_scores_per_cell.csv (step 1) and
dct_reviewer_task4_cluster_composition_long.csv (step 2).
Outputs:
  results/tables/dct_reviewer/dct_reviewer_panel_<data>.csv (chart data; both conditions and roles)
  results/figures/dct_reviewer/dct_reviewer_reviewer_summary_panel_<role>.pdf/.png/.tif

Panels show DCT1, DCT2 and ProLIF; 'Unknown' cells are in the tables but not drawn (as ctrl_reviewer_core). A core
label absent under a column has 0 cells; panel D (ProLIF concentration) does not apply under a
column without ProLIF.

Command (project root):  python scripts/dct_reviewer_reviewer_panel.py
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
import reviewer_core_lib as t2
import ctrl_reviewer_core_reviewer_panel as m19
import dct_reviewer_lib as lib

lib.style()

SCRIPT = "dct_reviewer_reviewer_panel.py"
ORDER = list(t2.CORE_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 1b: reviewer summary panel per DCT condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def plot_panel(data: dict, role: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 185)
    gs = fig.add_gridspec(4, 2, left=0.14, right=0.98, top=0.915, bottom=0.1, hspace=0.75, wspace=0.3)
    shared, legends = {}, {}
    for c, cond in enumerate(lib.CONDITIONS):
        d = data[(cond, role)]
        n = d["n"]
        ylabels = [f"{l} ({n.get(l, 0)})" for l in ORDER]
        y = np.arange(len(ORDER))
        # A: positive-marker composition
        ax = fig.add_subplot(gs[0, c], sharex=shared.get(0))
        shared.setdefault(0, ax)
        left = np.zeros(len(ORDER))
        for p in m19.POSITIVE_ORDER:
            vals = d["comp"].set_index(["label", "task3_dct_positive_label"])["percent_cells"].reindex(
                [(l, p) for l in ORDER]).fillna(0).to_numpy(float)
            ax.barh(y, vals, left=left, color=m19.POSITIVE_PALETTE[p], edgecolor="white", linewidth=0.5,
                    label=m19.POSITIVE_DISPLAY[p])
            left += vals
        ax.set_title(f"A  {cond}: highest positive DCT panel", fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of cells")
        # B: TAL / CNT-CD / interstitial ambiguity
        axb = fig.add_subplot(gs[1, c], sharex=shared.get(0))
        flagged = d["amb"][d["amb"]["status"] == "Flagged"].set_index("label")["percent_cells"].reindex(ORDER)
        axb.barh(y, flagged.fillna(0).to_numpy(float), color=m19.COLORS["orange"]["base"], height=0.6)
        axb.set_title(f"B  {cond}: TAL / CNT-CD / interstitial flag (> 90th pct)", fontsize=7)
        axb.set_xlabel("% of cells flagged")
        # C: cell-cycle score
        axc = fig.add_subplot(gs[2, c], sharex=shared.get(2))
        shared.setdefault(2, axc)
        task3, col = d["task3"], d["col"]
        for i, lab in enumerate(ORDER):
            v = task3.loc[task3[col] == lab, "cell_cycle_score"].to_numpy(float)
            if v.size:
                bp = axc.boxplot(v, positions=[i], vert=False, widths=0.55, showfliers=False, patch_artist=True)
                bp["boxes"][0].set_facecolor(m19.LEGACY_PALETTE[lab])
                bp["boxes"][0].set_edgecolor("#464C55")
                bp["medians"][0].set_color("#1F2430")
        axc.set_title(f"C  {cond}: cell-cycle score (18-protein panel)", fontsize=7)
        axc.set_xlabel("score_genes score (scaled X)")
        # D: ProLIF concentration after cell-cycle correction
        axd = fig.add_subplot(gs[3, c], sharex=shared.get(3))
        conc = d["conc"]
        if int(conc["n_prolif_total"].max()) == 0:
            lib.na_axis(axd, f"D  {cond}: {lib.NOT_APPLICABLE} (no ProLIF cell under {col})")
        else:
            shared.setdefault(3, axd)
            yy = np.arange(len(m19.BRANCHES))
            for k, (metric, color) in enumerate((("Largest corrected cluster", m19.COLORS["pink"]["base"]),
                                                  ("Top three corrected clusters", m19.COLORS["pink"]["dark"]))):
                vals = conc[conc["metric"] == metric].set_index("branch").reindex(list(m19.BRANCHES))["percent_prolif"]
                axd.barh(yy + (k - 0.5) * 0.36, vals.to_numpy(float), height=0.34, color=color, label=metric)
            wrapped = [v.replace(" proteins ", " proteins\n").replace(" score ", " score\n") for v in m19.BRANCHES.values()]
            axd.set_yticks(yy, wrapped if c == 0 else ["", ""])
            axd.invert_yaxis()
            axd.set_xlim(0, 100)
            axd.set_title(f"D  {cond}: ProLIF in Leiden 3.0 clusters (n = {int(conc['n_prolif_total'].max())})",
                          fontsize=7)
            axd.set_xlabel("% of ProLIF cells")
            legends.setdefault("D", axd.get_legend_handles_labels())
        for a in (ax, axb, axc):
            a.set_yticks(y, ylabels)
            a.invert_yaxis()
            a.xaxis.grid(True, color="#e1e0d9", linewidth=0.5)
            a.set_axisbelow(True)
        legends.setdefault("A", ax.get_legend_handles_labels())
    fig.text(0.005, 0.995, f"Reviewer summary panel by {lib.ROLE_DISPLAY[role]} (label (n cells); Unknown not drawn)",
             ha="left", va="top", fontsize=7)
    fig.legend(*legends["A"], loc="upper left", bbox_to_anchor=(0.135, 0.98), ncol=3, frameon=False)
    if "D" in legends:
        fig.legend(*legends["D"], loc="lower left", bbox_to_anchor=(0.135, 0.0), ncol=2, frameon=False)
    figs.save(fig, f"reviewer_summary_panel_{role}", f"Reviewer summary panel, {role} label, Ctrl and ZeroK",
              label_role=role)


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "reviewer_panel", args)
    lib.ported_from(run, "ctrl_reviewer_core_reviewer_panel.py")
    task3_all = lib.read_table(run, "task3_marker_scores_per_cell")
    comp_all = lib.read_table(run, "task4_cluster_composition_long", dtype={"resolution": str, "cluster": str})
    tables = {k: {} for k in ("positive_marker_composition", "ambiguity_flags", "cell_cycle_score_summary",
                              "prolif_concentration", "prolif_top_clusters")}
    data, notes = {}, []
    for cond in lib.CONDITIONS:
        s = lib.SETTINGS[cond]
        task3 = lib.from_generic(task3_all[task3_all["condition"] == cond], cond)
        composition = comp_all[comp_all["condition"] == cond]
        frames = {k: [] for k in tables}
        for role in lib.ROLES:
            col = s.COLUMN_OF[role]
            comp, amb, cyc = m19.marker_composition(task3, col), lib.ambiguity(task3, col), m19.cell_cycle_summary(task3, col)
            conc, top = m19.concentration(composition, col)
            conc["applicable"] = conc["n_prolif_total"] > 0
            for frame, stem in ((comp, "positive_marker_composition"), (amb, "ambiguity_flags"),
                                (cyc, "cell_cycle_score_summary"), (conc, "prolif_concentration"), (top, "prolif_top_clusters")):
                frames[stem].append(frame.assign(label_column=col))
            data[(cond, role)] = {"comp": comp, "amb": amb, "conc": conc, "task3": task3, "col": col,
                                  "n": task3[col].value_counts().to_dict()}
        for stem in tables:
            tables[stem][cond] = pd.concat(frames[stem], ignore_index=True)
        notes.append(f"{cond}: Unknown cells not drawn (" + "; ".join(
            f"{c} {int((task3[c] == 'Unknown').sum())}" for c in s.LABEL_COLUMNS) + ").")
    for stem, by_cond in tables.items():
        run.table(lib.stack(by_cond), f"panel_{stem}")
    for n in notes:
        run.note(n)
    run.note("Core labels absent under a column have 0 cells (percent NaN in panel_ambiguity_flags, 0 in "
             "panel_positive_marker_composition); prolif_concentration 'applicable' is False without ProLIF.")
    figs = lib.Figures(run, "reviewer_panel")
    for role in lib.ROLES:
        plot_panel(data, role, figs)
    figs.finish()
    run.finish({"figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_reviewer_panel.csv.")


if __name__ == "__main__":
    main()
