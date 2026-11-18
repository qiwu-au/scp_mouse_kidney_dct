#!/usr/bin/env python
"""dct_reviewer step 10: qc_mito step 5 dissociation-stress score per DCT condition (combined search).

Port of ctrl_reviewer_core_stress.py, run once per condition.
The ctrl_reviewer_core functions coverage, scores, loo_scores and depth_rows are called unchanged (depth_rows under
dct_reviewer_lib.bound with dct_reviewer_lib.part2_settings); the protein-group strings come from the dct_intake protein
table (dct_reviewer_lib.protein_meta) because the dct_labels objects keep gene names only. ProLIF contrasts and the
leave-one-out ProLIF contrasts run on the label columns that have ProLIF (the second columns); the primary columns
get 'does not apply' rows.

Inputs (read-only): dct_labels labelled objects and canonical labels; results/tables/dct_intake/dct_intake_proteins.csv
and dct_intake_stress_panel_coverage.csv (cross-check; dct_intake counts the joint proteins, so a panel protein
retained in one condition only is present in that condition's matrix and absent from the dct_intake table).
Outputs (results/tables/dct_reviewer/, results/figures/dct_reviewer/; every table has `condition`):
  dct_reviewer_stress_coverage.csv               18-protein panel: present, protein group, detection
  dct_reviewer_stress_detection_by_label.csv     detection rate of each present panel protein per label
  dct_reviewer_stress_per_cell.csv               scores per cell with labels (generic names), plate and depth
  dct_reviewer_stress_summary_by_label.csv
  dct_reviewer_stress_vs_depth.csv               Spearman with log10 IDs / log10 total intensity
  dct_reviewer_stress_type_tests.csv, _stress_type_coefficients.csv   across-label tests (with/without depth)
  dct_reviewer_stress_prolif_contrasts.csv       ProLIF vs DCT1 / DCT2 / DCT1+DCT2 (MWU; OLS +/- depth, plate)
  dct_reviewer_stress_leave_one_out.csv          type tests / ProLIF contrasts with each panel protein left out
  dct_reviewer_stress_checks.csv
  figures: dct_reviewer_stress_coverage, dct_reviewer_stress_by_{primary,second,plate},
           dct_reviewer_stress_vs_depth_{primary,second}, dct_reviewer_stress_umap, dct_reviewer_stress_prolif_contrasts

Scores (as ctrl_reviewer_core; panel proteins present in the matrix; a protein group shared by two panel genes counts once):
  stress_mean_z      mean of .X (per-protein z-scored ln(1 + intensity), clipped at 10) over the present proteins
  stress_score_genes scanpy score_genes on ln(1 + intensity) (ctrl_size 50, n_bins 25, random_state 42)
  stress_n_detected  number of present panel proteins with intensity > 0

Command (project root):  python scripts/dct_reviewer_stress.py
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
import qc_mito_lib as t3
import ctrl_reviewer_core_stress as s19
import dct_intake_lib as t26
import dct_reviewer_lib as lib

lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "dct_reviewer_stress.py"
DEPTH_VARS = s19.DEPTH_VARS
DCT_INTAKE_COVERAGE = cc.out_path("dct_intake", "tables", "stress_panel_coverage", ".csv")
SHORT_SCORE = {"stress_mean_z": "mean z (ln intensity)", "stress_score_genes": "score_genes",
               "stress_n_detected": "proteins detected (n)"}
MODEL_NAMES = {"label_only": "label only", "label_plus_log10_ids": "+ log10 IDs",
               "label_plus_log10_ids_plus_plate": "+ log10 IDs + plate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 10: dissociation-stress score per DCT condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str, t26cov: pd.DataFrame, proteins: pd.DataFrame) -> dict:
    s = lib.part2_settings(cond)
    adata, labels = lib.load(run, cond)
    meta = lib.protein_meta(run, cond, adata.var_names)
    for c in ("pg_genes", "protein_group"):
        adata.var[c] = meta[c].to_numpy()
    cov, present = s19.coverage(adata, "dct")
    # dct_intake coverage is over the joint proteins (retained in both conditions); each condition's own matrix may add
    # panel proteins retained in that condition only.
    ref = t26cov.set_index("gene")
    joint = [g for g in ref.index if str(ref.loc[g, "present"]) == "True"]
    own = cov.set_index("gene")
    joint_ok = all(bool(own.loc[g, "present"]) and own.loc[g, "matrix_protein"] == ref.loc[g, "matrix_protein"]
                   for g in joint)
    extra = [g for g in own.index if bool(own.loc[g, "present"]) and g not in joint]
    retention = proteins.set_index("var_name")["retention"].astype(str)
    extra_ok = all(retention.get(own.loc[g, "matrix_protein"]) == f"{cond} only" for g in extra)
    checks = [{"condition": cond, "check": "every stress panel gene present among the dct_intake joint proteins is present "
               "here with the same matrix protein", "value": f"{len(joint)} joint genes", "pass": joint_ok},
              {"condition": cond, "check": "panel genes present here beyond the joint proteins are retained in this "
               "condition only (dct_intake protein table)", "value": ", ".join(extra) or "none", "pass": extra_ok},
              {"condition": cond, "check": "at least one stress panel protein in the matrix",
               "value": f"{int(cov['present'].sum())}/18 genes in {len(present)} protein groups: {', '.join(present)}",
               "pass": bool(present)}]
    if not present:
        return {"checks": checks}
    frame = labels[["plate", *s.LABEL_COLUMNS, "n_detected_proteins", "total_intensity"]].copy()
    frame["log10_n_detected"] = np.log10(frame["n_detected_proteins"])
    frame["log10_total_intensity"] = np.log10(frame["total_intensity"])
    frame = frame.join(s19.scores(adata, present))
    intensity = cc.dense(adata.layers["intensity"])
    names = list(adata.var_names.astype(str))
    det_rows, summaries, tests, coefs, loo = [], [], [], [], []
    for col in (*s.LABEL_COLUMNS, "plate"):
        summaries.append(t3.summarise(frame, col, t3.STRESS_SCORES, "dct"))
        for label in t3.label_order(frame[col], "dct"):
            mask = (frame[col].astype(str) == label).to_numpy()
            for p in present:
                det = intensity[mask, names.index(p)] > 0
                det_rows.append({"dataset": "dct", "label_column": col, "label": label, "matrix_protein": p,
                                 "n_cells": int(mask.sum()), "n_detected": int(det.sum()),
                                 "percent_detected": 100 * det.mean()})
    with lib.bound(s, s19):
        depth = s19.depth_rows(frame, "dct")
    for col in s.LABEL_COLUMNS:
        for score in t3.STRESS_SCORES:
            summary, coef = t3.type_tests(frame, col, score, "DCT1")
            tests.append({"dataset": "dct", "variant": "all present proteins", **summary})
            coef.insert(0, "dataset", "dct")
            coefs.append(coef)
    with_prolif = lib.columns_with(labels, s, "ProLIF")
    without = [c for c in s.LABEL_COLUMNS if c not in with_prolif]
    for drop, values in s19.loo_scores(adata, present).items():
        tmp = frame.assign(stress_mean_z_loo=values)
        for col in s.LABEL_COLUMNS:
            summary, _ = t3.type_tests(tmp, col, "stress_mean_z_loo", "DCT1")
            tests.append({"dataset": "dct", "variant": f"stress_mean_z without {drop}", **summary})
        for col in with_prolif:
            c = t3.prolif_contrasts(tmp, col, ["stress_mean_z_loo"], contrasts=(("ProLIF", ("DCT1", "DCT2")),))
            c.insert(0, "left_out_protein", drop)
            loo.append(c.assign(status="tested", reason=""))
    reason = "no ProLIF cell under this column (dct_labels)"
    con = [t3.prolif_contrasts(frame, col, t3.STRESS_SCORES).assign(status="tested", reason="") for col in with_prolif]
    if without:
        na = lib.na_rows(cond, without, "ProLIF contrasts on stress scores", reason).drop(
            columns=["condition", "label_role", "analysis"])
        con.append(na)
        loo.append(na.assign(left_out_protein=""))
    return {"checks": checks, "coverage": cov, "present": present, "frame": frame,
            "detection": pd.DataFrame(det_rows), "summary": pd.concat(summaries, ignore_index=True), "depth": depth,
            "tests": pd.DataFrame(tests), "coefs": pd.concat(coefs, ignore_index=True),
            "contrasts": pd.concat(con, ignore_index=True), "loo": pd.concat(loo, ignore_index=True),
            "umap": np.asarray(adata.obsm["X_umap"])}


# ---------------------------------------------------------------------------
# Figures (dotplot format; Ctrl and ZeroK side by side)
# ---------------------------------------------------------------------------
def column_for(cond: str, key: str) -> str:
    return "plate" if key == "plate" else lib.SETTINGS[cond].COLUMN_OF[key]


def label_handles(order, colors) -> list:
    return [Line2D([], [], marker="o", linestyle="", color=colors[l], markersize=4, label=l) for l in order]


def fig_coverage(res: dict, figs: lib.Figures) -> None:
    genes = list(t3.STRESS_PANEL)
    y = np.arange(len(genes))
    fig = lib.new_figure(lib.FULL_MM, 110)
    gs = fig.add_gridspec(1, 2, left=0.1, right=0.98, top=0.88, bottom=0.1, wspace=0.08)
    first = None
    for j, cond in enumerate(lib.CONDITIONS):
        d = res[cond]["coverage"].set_index("gene").loc[genes]
        ax = fig.add_subplot(gs[0, j], sharey=first)
        first = first or ax
        vals = d["percent_cells_detected"].fillna(0).to_numpy(float)
        ax.barh(y, vals, height=0.65, color=t3.SERIES[0])
        for yi, (gene, row) in zip(y, d.iterrows()):
            if not row["present"]:
                text = "not in matrix"
            else:
                text = f"{row['percent_cells_detected']:.1f}%"
                if row.get("shared_group_with"):
                    text += f" ({row['pg_genes']})"
            ax.text((vals[yi] if row["present"] else 0) + 1.5, yi, text, va="center",
                    color=t2.INK["secondary"] if row["present"] else t2.INK["muted"])
        n_present = int(d["present"].sum())
        n_groups = len({p for p in d["matrix_protein"].fillna("") if p})
        ax.set_title(f"{cond}: {n_present}/18 genes present ({n_groups} protein groups; "
                     f"{lib.N_CELLS[cond]} cells)", fontsize=7)
        ax.set_xlim(0, 175)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("% of cells with intensity > 0")
        ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_yticks(y, genes)
            ax.set_ylim(len(genes) - 0.5, -0.5)
        else:
            ax.tick_params(labelleft=False)
    fig.text(0.005, 0.995, "Dissociation-stress panel: coverage in each condition's protein matrix\n(match on any "
             "PG.Genes symbol, case-insensitive; a group shared by two panel genes counts once)", ha="left", va="top",
             fontsize=7)
    figs.save(fig, "stress_coverage", "Stress panel coverage, Ctrl and ZeroK")


def fig_box(res: dict, key: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 118)
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.99, top=0.86, bottom=0.11, hspace=0.75, wspace=0.3)
    order = lib.union_order(*(res[c]["frame"][column_for(c, key)] for c in lib.CONDITIONS),
                            order=("P1", "P2") if key == "plate" else lib.LABEL_ORDER)
    colors = dict(cc.PLATE_PALETTE) if key == "plate" else lib.label_colors(order)
    rng = np.random.default_rng(lib.SEED)
    for r, cond in enumerate(lib.CONDITIONS):
        frame, col = res[cond]["frame"], column_for(cond, key)
        values = frame[col].astype(str).to_numpy()
        n = pd.Series(values).value_counts()
        for k, score in enumerate(t3.STRESS_SCORES):
            ax = fig.add_subplot(gs[r, k])
            for i, lab in enumerate(order):
                v = frame.loc[values == lab, score].to_numpy(float)
                v = v[np.isfinite(v)]
                if not v.size:
                    continue
                ax.scatter(i + rng.uniform(-0.22, 0.22, v.size), v, s=1.5, color=colors.get(lab, "0.6"), alpha=0.35,
                           linewidths=0, rasterized=True)
                bp = ax.boxplot(v, positions=[i], widths=0.6, showfliers=False, patch_artist=True)
                bp["boxes"][0].set_facecolor("none")
                bp["boxes"][0].set_edgecolor(t2.INK["primary"])
                for part in ("whiskers", "caps"):
                    for line in bp[part]:
                        line.set_color(t2.INK["secondary"])
                bp["medians"][0].set_color(t2.INK["primary"])
            ax.set_xticks(range(len(order)), [f"{l if l != 'Unknown' else 'Unk.'}\n{int(n.get(l, 0))}" for l in order])
            ax.set_xlim(-0.6, len(order) - 0.4)
            ax.set_title(f"{cond}: {SHORT_SCORE[score]}", fontsize=7)
            lib.grid_y(ax)
    what = "plate (pooled sort plate)" if key == "plate" else lib.ROLE_DISPLAY[key]
    n_present = {c: len(res[c]["present"]) for c in lib.CONDITIONS}
    fig.text(0.005, 0.995, f"Stress scores by {what} (box: median and IQR, whiskers 1.5 IQR; points: cells; tick: label "
             f"and cells)\npanel protein groups present: Ctrl {n_present['Ctrl']}, ZeroK {n_present['ZeroK']}; "
             "Unk. = Unknown", ha="left", va="top", fontsize=7)
    figs.save(fig, f"stress_by_{key}", f"Stress scores by {key}, Ctrl and ZeroK", label_role="" if key == "plate" else key)


def fig_depth(res: dict, role: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 118)
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.99, top=0.84, bottom=0.1, hspace=0.6, wspace=0.32)
    order = lib.union_order(*(res[c]["frame"][column_for(c, role)] for c in lib.CONDITIONS))
    colors = lib.label_colors(order)
    rng = np.random.default_rng(lib.SEED)
    for r, cond in enumerate(lib.CONDITIONS):
        frame, col = res[cond]["frame"], column_for(cond, role)
        values = frame[col].astype(str).to_numpy()
        for k, score in enumerate(t3.STRESS_SCORES):
            ax = fig.add_subplot(gs[r, k])
            yv = frame[score].to_numpy(float)
            if score == "stress_n_detected":
                yv = yv + rng.uniform(-0.18, 0.18, len(yv))
            for lab in order:
                m = values == lab
                ax.scatter(frame.loc[m, "n_detected_proteins"], yv[m], s=2, color=colors[lab], linewidths=0,
                           alpha=0.8, rasterized=True)
            rho = t3.spearman_ci(frame["log10_n_detected"], frame[score])
            ax.set_xscale("log")
            ax.set_title(f"{cond}: {SHORT_SCORE[score]}, rho {rho['rho']:.2f}", fontsize=7)
            if r == 1:
                ax.set_xlabel("Protein IDs per cell (log)")
            lib.grid_y(ax)
    fig.legend(handles=label_handles(order, colors), loc="upper right", bbox_to_anchor=(0.99, 0.935),
               ncol=len(order), frameon=False, handletextpad=0.2, columnspacing=1.0)
    fig.text(0.005, 0.995, f"Stress scores against depth, by {lib.ROLE_DISPLAY[role]}\n(Spearman rho with log10 "
             "protein IDs, all cells of the condition; n detected jittered)", ha="left", va="top", fontsize=7)
    figs.save(fig, f"stress_vs_depth_{role}", f"Stress scores vs depth by {role} label, Ctrl and ZeroK", label_role=role)


def fig_umap(res: dict, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 96)
    gs = fig.add_gridspec(2, 4, left=0.04, right=0.955, top=0.87, bottom=0.1, hspace=0.35, wspace=0.4)
    order = lib.union_order(*(res[c]["frame"][column_for(c, "primary")] for c in lib.CONDITIONS))
    for r, cond in enumerate(lib.CONDITIONS):
        frame, xy = res[cond]["frame"], res[cond]["umap"]
        values = frame[column_for(cond, "primary")].astype(str).to_numpy()
        for c in range(4):
            ax = fig.add_subplot(gs[r, c])
            if c == 0:
                for lab in order:
                    m = values == lab
                    ax.scatter(xy[m, 0], xy[m, 1], s=1.2, color=lib.LABEL_PALETTE.get(lab, "0.5"), linewidths=0,
                               rasterized=True)
                ax.set_title("primary label", fontsize=7)
            else:
                score = t3.STRESS_SCORES[c - 1]
                v = frame[score].to_numpy(float)
                lo, hi = np.nanpercentile(v, [1, 99])
                sca = ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(v, lo, hi), cmap="viridis", s=1.2, linewidths=0,
                                 rasterized=True)
                cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.02)
                cb.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                cb.ax.tick_params(labelsize=7, length=2)
                ax.set_title(SHORT_SCORE[score], fontsize=7)
            t2.clean_axes(ax)
            if c == 0:
                ax.set_ylabel(cond, fontsize=7)
    fig.legend(handles=label_handles(order, lib.label_colors(order)), loc="lower center", ncol=len(order),
               frameon=False, bbox_to_anchor=(0.5, 0.005), handletextpad=0.2, columnspacing=1.2)
    fig.text(0.005, 0.995, "Stress scores on each condition's dct_labels UMAP (colour range 1st-99th percentile within "
             "condition)", ha="left", va="top", fontsize=7)
    figs.save(fig, "stress_umap", "Stress scores on the dct_labels UMAP, Ctrl and ZeroK")


def fig_contrasts(con: pd.DataFrame, figs: lib.Figures) -> None:
    tested = con[con["status"] == "tested"]
    rows = tested[["condition", "label_column", "comparison"]].drop_duplicates().reset_index(drop=True)
    y = np.arange(len(rows))
    fig = lib.new_figure(lib.FULL_MM, 100)
    gs = fig.add_gridspec(1, 3, left=0.2, right=0.985, top=0.8, bottom=0.12, wspace=0.12)
    first = None
    for k, score in enumerate(t3.STRESS_SCORES):
        ax = fig.add_subplot(gs[0, k], sharey=first)
        first = first or ax
        d = tested[tested["metric"] == score].set_index(["condition", "label_column", "comparison"])
        sub = d.loc[list(map(tuple, rows.to_numpy()))]
        for m, model in enumerate(t3.CONTRAST_MODELS):
            ax.errorbar(sub[f"{model}_coef"], y + (m - 1) * 0.22,
                        xerr=[sub[f"{model}_coef"] - sub[f"{model}_ci_low"], sub[f"{model}_ci_high"] - sub[f"{model}_coef"]],
                        fmt="o", markersize=2.6, color=t3.SERIES[m], elinewidth=0.7, capsize=0)
        ax.axvline(0, color=t2.INK["muted"], linewidth=0.6)
        ax.set_title(SHORT_SCORE[score], fontsize=7)
        ax.set_xlabel("ProLIF minus reference")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.grid(True, color=t2.INK["grid"], linewidth=0.5)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_yticks(y, [f"{c}: {cmp.replace('ProLIF_vs_', 'vs ')}" for c, _, cmp in rows.to_numpy()])
            ax.set_ylim(len(rows) - 0.5, -0.5)
        else:
            ax.tick_params(labelleft=False)
    handles = [Line2D([], [], marker="o", linestyle="", color=t3.SERIES[m], markersize=4, label=MODEL_NAMES[model])
               for m, model in enumerate(t3.CONTRAST_MODELS)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.915), ncol=3, frameon=False)
    fig.text(0.005, 0.995, "ProLIF vs DCT1 / DCT2 / DCT1+DCT2 on the stress scores under the second label (submitted "
             "cluster label), unadjusted and depth-adjusted\n(OLS, HC3 95% CI; plate = pooled sort plate; the primary "
             "labels have no ProLIF: does not apply)", ha="left", va="top", fontsize=7)
    figs.save(fig, "stress_prolif_contrasts", "ProLIF contrasts on stress scores (second label), Ctrl and ZeroK",
              label_role="second")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "stress", args)
    lib.ported_from(run, "ctrl_reviewer_core_stress.py")
    t26cov = pd.read_csv(run.use_input(DCT_INTAKE_COVERAGE, "dct_intake stress panel coverage (cross-check)"), keep_default_na=False)
    proteins = t26.read_proteins(run)
    res = {c: one_condition(run, c, t26cov, proteins) for c in lib.CONDITIONS}
    checks = lib.check_rows_frame([r for c in lib.CONDITIONS for r in res[c]["checks"]])
    lib.stop_if_failed(run, checks, "stress_checks")
    for cond in lib.CONDITIONS:
        cov = res[cond]["coverage"]
        run.note(f"{cond}: stress panel genes present {int(cov['present'].sum())}/18 in {len(res[cond]['present'])} "
                 f"protein groups ({', '.join(res[cond]['present'])}).")

    def stacked(key: str, generic: bool = False) -> pd.DataFrame:
        frames = {}
        for c in lib.CONDITIONS:
            f = res[c][key]
            frames[c] = lib.to_generic(f.reset_index(), c) if generic else f
        return lib.stack(frames)

    run.table(stacked("coverage"), "stress_coverage")
    run.table(stacked("detection"), "stress_detection_by_label")
    run.table(stacked("frame", generic=True), "stress_per_cell")
    run.table(stacked("summary"), "stress_summary_by_label")
    run.table(stacked("depth"), "stress_vs_depth")
    run.table(stacked("tests"), "stress_type_tests")
    run.table(stacked("coefs"), "stress_type_coefficients")
    con = stacked("contrasts")
    run.table(con, "stress_prolif_contrasts")
    run.table(stacked("loo"), "stress_leave_one_out")
    run.note("ProLIF contrasts (and their leave-one-out versions) run on the second columns only; the primary columns "
             "have no ProLIF (rows with status 'does not apply'). BH within condition x label column (qc_mito prolif_contrasts).")

    figs = lib.Figures(run, "stress")
    fig_coverage(res, figs)
    for key in (*lib.ROLES, "plate"):
        fig_box(res, key, figs)
    for role in lib.ROLES:
        fig_depth(res, role, figs)
    fig_umap(res, figs)
    fig_contrasts(con, figs)
    figs.finish()
    run.finish({"panel": t3.STRESS_PANEL, "present_proteins": {c: res[c]["present"] for c in lib.CONDITIONS},
                "score_genes": {"matrix": "ln(1 + intensity)", "ctrl_size": 50, "n_bins": 25, "random_state": t3.SEED},
                "mean_z_matrix": ".X of the dct_labels object (z-scored ln(1 + intensity), clipped at 10)",
                "min_group": t3.MIN_GROUP, "figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_stress.csv.")


if __name__ == "__main__":
    main()
