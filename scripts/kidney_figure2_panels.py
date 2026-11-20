#!/usr/bin/env python
"""kidney_figure2 step 2: Figure 2C-2F (submitted 2D-2G) and Supplementary Table S6 under ``kidney_t15_label``.

Task kidney_figure2 (authors, 31 Oct 2026).

Panel letters: the submitted Figure 2B is removed, so the submitted 2C-2G are now 2B-2F (authors, 31 Oct 2026); file
and table names use the new letters. Same definitions and parameters as the submitted code (kidney_figure2_lib;
reproduced in step 1), with the one revised kidney annotation (kidney_annotation (a), matched on cell_id) in place of the dropped
re-annotation:
- 2C (submitted 2D): IDs per cell (proteins > 0 among the 4,054 retained proteins; the section-7 count over all
  export rows is kept as a column) and within-cell dynamic range (log10 P95 - log10 P5 of the detected intensities),
  per type; Unknown is its own grey group (authors, 31 Oct 2026).
- 2D (submitted 2E) / Table S3: core (>= 50% of the type's cells), exact core-set combinations, top 20 (ties ordered
  by fewer types, then label); members of the top 20 in the submitted Table S3 layout. Unique core (core, < 10% in
  every other type).
- 2E (submitted 2F): Jaccard of the core sets.
- 2F (submitted 2G): Wilcoxon (scanpy, on the pre-log intensities) per type vs the rest of the known cells; top 10
  by score over all types; stacked violin of the 0-1 scaled intensities, fill = group median (kidney_figure2 stacked-violin
  function).
Unknown is left out of 2D-2F (submitted 2E-2G; submitted rule); the sensitivity tables keep it as a group.
Display names DCT1 -> DCT, PC -> CNT/PC; row order and colours of the revised Figure 2A / 2B (step fig2ab).
User, 31 Oct 2026, after the first run: 2F keeps the submitted rule; 2D uses the UpSet layout only; Table S3 also
lists the unique-core proteins (second CSV); every plot is its own file without panel letters or titles (the authors
assemble Figure 2 in PowerPoint), so 2C is two files.
User, 10 Nov 2026: PST is drawn in tab10 brown #8c564b instead of the kidney_annotation palette grey #7f7f7f (kidney_figure2_lib.palette;
only the 2C violins use the type colours).

Inputs (read-only): kidney_object kidney object and labels; kidney_annotation labels; data/KIDNEY/KIDNEY_768cells.csv; data/KIDNEY/V3 per-cell file
                    (dynamic-range check); V3 workbook (+ the kidney_annotation endothelial row); kidney_figure2 step-1 tables.
Outputs: results/tables/kidney_figure2/kidney_figure2_t15_*.csv, kidney_figure2_step_panels_manifest.json;
         results/figures/kidney_figure2/kidney_figure2_fig2c_ids_per_cell, kidney_figure2_fig2c_dynamic_range,
         kidney_figure2_fig2d_core_combinations, kidney_figure2_fig2e_core_jaccard, kidney_figure2_fig2f_stacked_violin_top10
         (.pdf/.png each)

Command (project root):  python scripts/kidney_figure2_panels.py
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
import warnings

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as t7
import kidney_annotation_chosen_labels as t15c     # kidney_annotation row order and palette (as the revised Figure 2A / 2B)
import kidney_figure2_lib as T

STEP = "panels"
TYPE_NAMES = {   # Supplementary Table S6 abbreviation sheet (draft)
    "Endothelial": "Endothelial cells",
    "JG": "Juxtaglomerular cells",
    "PCT": "Proximal convoluted tubule",
    "PST": "Proximal straight tubule",
    "TAL": "Thick ascending limb of the loop of Henle",
    "DCT": "Distal convoluted tubule (DCT1 and DCT2 not separated)",
    "CNT/PC": "Connecting tubule and collecting duct principal cells (not separated)",
    "IC-B": "Collecting duct intercalated cells, type B",
}


def q(x, p):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy()
    return float(np.percentile(x, p)) if x.size else np.nan


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig2c(run, per_cell, rows, colors, fig_rows):
    """Figure 2C (submitted 2D) as two files (IDs per cell; dynamic range), each with its own x axis."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    T.setup_style()
    lab = per_cell["label"].astype(str).to_numpy()
    ticks = [f"{T.display(r)}\n({int((lab == r).sum())})" for r in rows]
    pal = {r: (T.UNKNOWN_COLOR if r == T.UNKNOWN else colors[r]) for r in rows}
    for name, col, ylab, desc in (
            ("fig2c_ids_per_cell", "ids_retained", "Protein IDs per cell", "Figure 2C (IDs per cell)"),
            ("fig2c_dynamic_range", "dynr_value", "Dynamic range\n(log10 P95 / P5)", "Figure 2C (dynamic range)")):
        fig, ax = plt.subplots(figsize=(T.PANEL_W_MM * T.MM, 50 * T.MM))
        d = pd.DataFrame({"g": lab, "v": per_cell[col].to_numpy(float)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.violinplot(data=d, x="g", y="v", hue="g", order=rows, hue_order=rows, palette=pal, cut=0,
                           inner="quartile", linewidth=0.5, linecolor="#52514e", width=0.8,
                           legend=False, ax=ax, inner_kws={"linewidth": 0.5})
        if col == "ids_retained":
            ax.set_ylim(0, None)
        ax.set_ylabel(ylab)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(ticks, rotation=0, fontsize=6)
        ax.set_xlabel("Cell type (n cells); Unknown = clusters without marker support")
        ax.tick_params(axis="both", length=2)
        plt.tight_layout()
        paths = run.figure(fig, name, f"{desc} under kidney_t15_label")
        fig_rows.append(T.figure_checks(fig, name, paths, ticks))


def fig2d(run, top, types, n_cells, fig_rows):
    """Bars of the top 20 exact core-set combinations with an UpSet-style membership matrix below."""
    import matplotlib.pyplot as plt

    T.setup_style()
    n = len(top)
    fig = plt.figure(figsize=(84 * T.MM, 78 * T.MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    mx = fig.add_subplot(gs[1], sharex=ax)
    ax.bar(range(n), top["count"].to_numpy(), color="#2a78d6", edgecolor="none", width=0.7)
    for i, c in enumerate(top["count"].to_numpy()):
        ax.text(i, c, f"{c}", ha="center", va="bottom", fontsize=5.2, color="#52514e")
    ax.set_ylabel("Proteins (exact\ncore-set membership)")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_ylim(0, top["count"].max() * 1.12)
    ax.tick_params(axis="y", length=2)
    labels = [f"{T.display(t)} ({n_cells[t]})" for t in types]
    for r in range(len(types)):
        if r % 2 == 0:
            mx.axhspan(r - 0.5, r + 0.5, color="#f2f1ed", zorder=0, lw=0)
    for i, combo in enumerate(top["combo"]):
        members = combo.split(";")
        ys = [types.index(m) for m in members]
        mx.scatter([i] * len(types), range(len(types)), s=7, color="#d6d5cf", zorder=1, lw=0)
        mx.scatter([i] * len(ys), ys, s=9, color="#0b0b0b", zorder=3, lw=0)
        if len(ys) > 1:
            mx.plot([i, i], [min(ys), max(ys)], color="#0b0b0b", lw=0.8, zorder=2)
    mx.set_yticks(range(len(types)))
    mx.set_yticklabels(labels)
    mx.set_ylim(len(types) - 0.5, -0.5)
    mx.set_xlim(-0.6, n - 0.4)
    mx.tick_params(axis="both", length=0)
    mx.tick_params(axis="x", labelbottom=False)
    for side in mx.spines.values():
        side.set_visible(False)
    paths = run.figure(fig, "fig2d_core_combinations", "Figure 2D under kidney_t15_label (UpSet layout)")
    fig_rows.append(T.figure_checks(fig, "fig2d_core_combinations", paths, labels))


def fig2e(run, jac, types, n_cells, fig_rows):
    import matplotlib.pyplot as plt

    T.setup_style()
    n = len(types)
    labels = [f"{T.display(t)} ({n_cells[t]})" for t in types]
    jac_ma = np.ma.array(jac, mask=np.triu(np.ones_like(jac, dtype=bool), k=1))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    fig, ax = plt.subplots(figsize=(T.PANEL_W_MM * T.MM, 74 * T.MM))
    im = ax.imshow(jac_ma, aspect="equal", cmap=cmap, vmin=0.0, vmax=1.0)
    for i in range(n):
        for j in range(i):
            ax.text(j, i, f"{jac[i, j]:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if jac[i, j] < 0.55 else "#0b0b0b")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Jaccard index (core proteomes)")
    cb.ax.tick_params(length=2)
    plt.tight_layout()
    paths = run.figure(fig, "fig2e_core_jaccard", "Figure 2E under kidney_t15_label")
    fig_rows.append(T.figure_checks(fig, "fig2e_core_jaccard", paths, labels))


def fig2f(run, scaled, groups, types, n_cells, top10, fig_rows):
    import matplotlib.pyplot as plt

    T.setup_style()
    fig = plt.figure(figsize=(71 * T.MM, 70 * T.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.07], wspace=0.06)
    labels = [f"{T.display(t)} ({n_cells[t]})" for t in types]
    _, med, norm = T.stacked_violin(fig, gs[0, 0], scaled, groups, types, labels, list(top10), linewidth=0.3)
    cax = fig.add_subplot(gs[0, 1].subgridspec(3, 1, height_ratios=[1, 1.4, 1])[1, 0])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="Blues"), cax=cax)
    cb.set_label("Median in group\n(intensity scaled 0-1)", fontsize=6)
    cb.ax.tick_params(length=2, labelsize=6)
    paths = run.figure(fig, "fig2f_stacked_violin_top10", "Figure 2F under kidney_t15_label")
    fig_rows.append(T.figure_checks(fig, "fig2f_stacked_violin_top10", paths, labels + list(top10)))
    return med


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="kidney_figure2 step 2: Figure 2C-2F and Table S3 under kidney_t15_label.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(T.TASK, STEP, args)
    checks = t5.Checks(T.DATASET)
    np.random.seed(T.SEED)

    adata, labels = T.load(run, checks)
    cell_ids = labels.index.astype(str)
    raw = T.intensity(adata)
    det = (raw > 0).astype(np.int8)
    var_names = list(adata.var_names.astype(str))
    var = adata.var
    lab = labels[T.PRIMARY].astype(str)
    types, rows = t15c.t15_rows(lab.unique())
    colors = T.palette(adata, rows, checks)      # kidney_annotation palette, PST #8c564b (authors, 10 Nov 2026)
    checks.add("cell types in the revised Figure 2A / 2B order (Unknown last)", "; ".join(map(T.display, rows)),
               rows[-1] == T.UNKNOWN and len(types) == 8)
    n_cells = {t: int((lab == t).sum()) for t in rows}
    abbr = {t: T.display(t) for t in rows}

    # --- 2D -------------------------------------------------------------------------------------------------------
    metrics = T.cell_metrics(raw, cell_ids)
    ok = bool((metrics["sens_n_proteins"].to_numpy() == labels["n_detected_proteins"].to_numpy()).all())
    checks.add("IDs per cell = kidney_object n_detected_proteins (retained proteins)", "712 of 712" if ok else "differ", ok)
    sub_pc = pd.read_csv(run.use_input(T.SUB["per_cell"], "submitted per-cell dynamic range"))
    sub_pc = sub_pc.set_index(sub_pc.columns[0]).reindex(cell_ids)
    mad = float(np.nanmax(np.abs(metrics["dynr_value"].to_numpy() - sub_pc["dynr_value"].to_numpy())))
    checks.add("dynamic range per cell equals the submitted values (label-independent; max abs diff)", f"{mad:.2e}",
               mad < 1e-5, "< 1e-5")
    ids_all = T.ids_all_export_rows(run.use_input(cc.INPUTS["kidney"]["raw_csv"], "raw kidney export"), cell_ids)
    per_cell = pd.DataFrame({"cell_id": cell_ids, "label": lab.to_numpy(), "display_label": lab.map(T.display).to_numpy(),
                             "t15_cluster": labels["t15_cluster"].to_numpy(),
                             "kidney_t15_confidence": labels["kidney_t15_confidence"].to_numpy(),
                             "ids_retained": metrics["sens_n_proteins"].to_numpy(),
                             "ids_all_export_rows": ids_all.to_numpy(),
                             **{c: metrics[c].to_numpy() for c in ("dynr_value", "dynr_low", "dynr_high")}})
    run.table(per_cell, "t15_per_cell_ids_dynamic_range")
    rows_2d = []
    for r in rows:
        d = per_cell[per_cell["label"] == r]
        rows_2d.append({"label": r, "display_label": T.display(r), "n_cells": len(d),
                        "ids_median": float(d["ids_retained"].median()), "ids_q1": q(d["ids_retained"], 25),
                        "ids_q3": q(d["ids_retained"], 75), "ids_iqr": q(d["ids_retained"], 75) - q(d["ids_retained"], 25),
                        "ids_min": int(d["ids_retained"].min()), "ids_max": int(d["ids_retained"].max()),
                        "ids_all_export_rows_median": float(d["ids_all_export_rows"].median()),
                        "dr_median": float(d["dynr_value"].median()), "dr_q1": q(d["dynr_value"], 25),
                        "dr_q3": q(d["dynr_value"], 75), "dr_iqr": q(d["dynr_value"], 75) - q(d["dynr_value"], 25),
                        "dr_min": float(d["dynr_value"].min()), "dr_max": float(d["dynr_value"].max()),
                        "dr_missing": int(d["dynr_value"].isna().sum())})
    tab_2d = pd.DataFrame(rows_2d)
    tab_2d["ids_median_rank_known"] = tab_2d["ids_median"].where(tab_2d["label"] != T.UNKNOWN).rank(ascending=False)
    tab_2d["dr_median_rank_known"] = tab_2d["dr_median"].where(tab_2d["label"] != T.UNKNOWN).rank(ascending=False)
    run.table(tab_2d, "t15_fig2c_by_type")
    agg = T.summarize_by(lab, metrics).set_index("cell_type")
    t2 = tab_2d.set_index("label").loc[types]
    d2 = max(float((agg.loc[types, "sens_median"] - t2["ids_median"]).abs().max()),
             float((agg.loc[types, "dr_median"] - t2["dr_median"]).abs().max()))
    checks.add("per-type medians equal the submitted summarize_by (known types)", f"max abs diff {d2:.1e}", d2 < 1e-9)
    checks.add("cells without a dynamic-range value (< 10 detected values)", f"{int(per_cell['dynr_value'].isna().sum())}",
               None)

    # --- core / unique core ---------------------------------------------------------------------------------------
    _, coverage = T.coverage_matrix(det, lab, types)
    core, unique = T.core_unique(coverage)
    combos, counts = T.exact_combinations(types, core, abbr, stable=True)
    exact_only = counts.set_index("combo")["count"]
    pid = T.protein_ids_by_celltype(det, lab, metrics["sens_n_proteins"], types).set_index("cell_type")
    summ = pd.DataFrame({"label": types, "display_label": [T.display(t) for t in types],
                         "n_cells": [n_cells[t] for t in types],
                         "core_min_cells": [T.ceil_half(n_cells[t]) for t in types],
                         "n_core": core.sum(axis=1), "n_unique_core": unique.sum(axis=1),
                         "n_exact_only": [int(exact_only.get(t, 0)) for t in types],
                         "union_ids": pid.loc[types, "union_ids"].to_numpy(),
                         "ids_median": pid.loc[types, "median_per_cell_IDs"].to_numpy()})
    summ["unique_core_rank"] = summ["n_unique_core"].rank(ascending=False, method="min").astype(int)
    summ["exact_only_rank"] = summ["n_exact_only"].rank(ascending=False, method="min").astype(int)
    run.table(summ, "t15_core_unique_summary")
    checks.add("unique-core proteins per type", "; ".join(f"{T.display(t)} {int(u)}" for t, u in
                                                         zip(types, unique.sum(axis=1))), None)
    n_core_any = int(core.any(axis=0).sum())
    checks.add("proteins core in at least one type; in all 8 types", f"{n_core_any}; "
               f"{int(core.all(axis=0).sum())}", None)
    cov = pd.DataFrame(coverage.T, columns=[f"coverage_{T.display(t)}" for t in types])
    unk = (lab == T.UNKNOWN).to_numpy()
    cov.insert(0, "protein", var_names)
    cov.insert(1, "gene_name", var["gene_name"].astype(str).to_numpy())
    cov.insert(2, "protein_group", var["protein_group"].astype(str).to_numpy())
    cov["coverage_Unknown"] = det[unk].mean(axis=0)
    cov["core_in"] = ["; ".join(T.display(t) for t in c.split(";")) if c != "(none)" else "" for c in combos]
    cov["unique_core_in"] = ["; ".join(T.display(types[i]) for i in np.flatnonzero(unique[:, p]))
                             for p in range(len(var_names))]
    run.table(cov, "t15_coverage_by_type")
    marker_rows = t7.kidney_marker_rows(run)
    listed: dict[str, list[str]] = {}
    for short, _, genes in marker_rows:
        for g in genes:
            listed.setdefault(g, []).append(short)
    long = []
    for i, t in enumerate(types):
        for p in np.flatnonzero(core[i]):
            others = np.delete(coverage[:, p], i)
            j = int(np.argmax(others))
            long.append({"label": t, "display_label": T.display(t), "protein": var_names[p],
                         "gene_name": var["gene_name"].iat[p], "protein_group": var["protein_group"].iat[p],
                         "protein_description": var["protein_description"].iat[p],
                         "coverage_in_type": coverage[i, p], "max_coverage_other_type": others[j],
                         "other_type_with_max": (T.display(np.delete(np.array(types), i)[j]) if others[j] > 0
                                                 else "none"),
                         "coverage_unknown": cov["coverage_Unknown"].iat[p], "unique_core": bool(unique[i, p]),
                         "exact_combination": abbr_combo(combos[p], abbr),
                         "marker_list_rows": "; ".join(T.display(x) for x in listed.get(var_names[p], []))})
    long = pd.DataFrame(long).sort_values(["label", "unique_core", "protein"], ascending=[True, False, True])
    run.table(long, "t15_core_proteins")
    uq = long[long["unique_core"]]
    run.table(uq, "t15_unique_core_proteins")
    checks.add("unique-core proteins that are markers of the V3 + endothelial list (any row)",
               f"{int((uq['marker_list_rows'] != '').sum())} of {len(uq)}: " + "; ".join(
                   f"{r.protein} ({r.display_label})" for r in uq[uq["marker_list_rows"] != ""].itertuples()), None)

    # --- exact combinations (2E) and Table S3 ----------------------------------------------------------------------
    top = T.top_combinations(counts, stable=True)
    c20, c21 = counts["count"].iloc[T.UPSET_TOPN - 1], counts["count"].iloc[T.UPSET_TOPN]
    checks.add("count at bar 20 and bar 21 (ties ordered by fewer types, then label)", f"{c20} and {c21}", None)
    counts_out = counts.copy()
    counts_out["rank"] = np.arange(1, len(counts_out) + 1)
    counts_out["in_top20"] = counts_out["rank"] <= T.UPSET_TOPN
    run.table(counts_out, "t15_exact_combinations")
    members = T.combination_members(combos, top, var_names, types, abbr)
    gene_of = dict(zip(var_names, var["gene_name"].astype(str)))
    members.insert(1, "gene_name", members["Protein"].map(gene_of))
    run.table(members, "t15_supp_table3_members_long")
    cols = top["combo_abbr"].tolist()
    lists = {c: sorted(members.loc[members["combo_abbr"] == c, "Protein"].astype(str)) for c in cols}
    depth = max(len(v) for v in lists.values())
    wide = [["Supplemental Table 3. List of core protein memberships shown in Figure 2D"] + [""] * (len(cols) - 1),
            [""] * len(cols), cols]
    wide += [[lists[c][k] if k < len(lists[c]) else "" for c in cols] for k in range(depth)]
    path = run.out("tables", "t15_supp_table3_wide", ".csv")
    pd.DataFrame(wide).to_csv(path, index=False, header=False)
    print(f"Saved table: {cc.rel(path)}")
    run.table(pd.DataFrame({"Short": [T.display(t) for t in types], "Cell type": [TYPE_NAMES[T.display(t)]
                                                                                for t in types]}),
              "t15_supp_table3_abbreviations")
    checks.add("Table S3 members = sum of the top-20 counts", f"{len(members)} vs {int(top['count'].sum())}",
               len(members) == int(top["count"].sum()))
    # second sheet: the unique-core proteins the Results sentence refers to (authors, 31 Oct 2026)
    order = {t: k for k, t in enumerate(types)}
    u = uq.assign(_o=uq["label"].map(order)).sort_values(["_o", "protein"])
    head = ["Cell type", "Protein", "Gene", "Protein group(s)", "Protein description",
            "Detected in the cell type (% of cells)", "Highest detection in another cell type (% of cells)",
            "Cell type with the highest other detection", "Marker in Supplementary Table 1"]
    sheet = [["Supplemental Table 3 (continued). Unique-core proteins per cell type: detected in at least 50% of "
              "the cells of the cell type and in less than 10% of the cells of every other cell type (Unknown "
              "cells not included)."] + [""] * (len(head) - 1), [""] * len(head), head]
    sheet += [[r.display_label, r.protein, r.gene_name, r.protein_group, r.protein_description,
               round(100 * r.coverage_in_type, 1), round(100 * r.max_coverage_other_type, 1), r.other_type_with_max,
               r.marker_list_rows if isinstance(r.marker_list_rows, str) and r.marker_list_rows else "no"]
              for r in u.itertuples()]
    path = run.out("tables", "t15_supp_table3_unique_core", ".csv")
    pd.DataFrame(sheet).to_csv(path, index=False, header=False)
    print(f"Saved table: {cc.rel(path)}")
    n_uq = int(unique.sum())
    checks.add("Table S3 unique-core sheet rows = unique-core proteins (sum over the 8 types)",
               f"{len(u)} vs {n_uq}", len(u) == n_uq)

    # --- Jaccard (2F) ----------------------------------------------------------------------------------------------
    jac, inter, union = T.jaccard(core)
    mat = pd.DataFrame(jac, index=[T.display(t) for t in types], columns=[T.display(t) for t in types])
    run.table(mat.reset_index(names="type"), "t15_core_jaccard_matrix")
    jl = pd.DataFrame([{"type_i": T.display(types[i]), "type_j": T.display(types[j]), "n_core_i": int(core[i].sum()),
                        "n_core_j": int(core[j].sum()), "intersection": int(inter[i, j]), "union": int(union[i, j]),
                        "jaccard": float(jac[i, j])} for i in range(len(types)) for j in range(i)])
    jl = jl.sort_values("jaccard", ascending=False)
    jl["rank"] = np.arange(1, len(jl) + 1)
    run.table(jl, "t15_core_jaccard_pairs")

    # --- DE top 10 (2G) ----------------------------------------------------------------------------------------------
    ad_de = T.de_adata(raw, lab, var_names, order=types)
    checks.add("DE groups (known types with >= 2 cells), cells", f"{ad_de.obs['group'].nunique()}; {ad_de.n_obs}",
               ad_de.obs["group"].nunique() == len(types))
    de_df, top10 = T.de_rank(ad_de)
    de_df = de_df.rename(columns={"group": "label"})
    de_df["display_label"] = de_df["label"].astype(str).map(T.display)
    de_df["rank_in_type"] = de_df.groupby("label", observed=True)["scores"].rank(ascending=False, method="first")
    de_df = de_df.drop(columns=["logfoldchanges"], errors="ignore")
    run.table(de_df, "t15_de_wilcoxon_all")
    first =de_df.drop_duplicates("names").set_index("names")
    lab_de = ad_de.obs["group"].astype(str).to_numpy()
    xd = np.asarray(ad_de.X) > 0
    top_rows = []
    for k, g in enumerate(top10, 1):
        j = var_names.index(g)
        grp = str(first.loc[g, "label"])
        top_rows.append({"rank": k, "protein": g, "gene_name": gene_of[g], "top_type": T.display(grp),
                         "score": float(first.loc[g, "scores"]), "pvals_adj": float(first.loc[g, "pvals_adj"]),
                         "detected_in_top_type": float(xd[lab_de == grp, j].mean()),
                         "detected_in_other_known": float(xd[lab_de != grp, j].mean()),
                         "marker_list_rows": "; ".join(T.display(x) for x in listed.get(g, [])),
                         "in_submitted_top10": g in T.SUBMITTED_TOP10})
    top_tab = pd.DataFrame(top_rows)
    run.table(top_tab, "t15_de_top10")
    # information only (the panel keeps the submitted rule): the top 3 proteins of each type by score
    per_type = []
    for t in types:
        sub = de_df[de_df["label"].astype(str) == t].sort_values("scores", ascending=False).head(3)
        for r in sub.itertuples():
            j = var_names.index(r.names)
            per_type.append({"type": T.display(t), "rank_in_type": int(r.rank_in_type), "protein": r.names,
                             "score": float(r.scores), "pvals_adj": float(r.pvals_adj),
                             "detected_in_type": float(xd[lab_de == t, j].mean()),
                             "detected_in_other_known": float(xd[lab_de != t, j].mean()),
                             "rank_over_all_types": int(de_df.index.get_loc(r.Index)) + 1,
                             "marker_list_rows": "; ".join(T.display(x) for x in listed.get(r.names, []))})
    run.table(pd.DataFrame(per_type), "t15_de_top3_per_type")
    checks.add("types holding the top-10 proteins under the submitted rule (all types' rows sorted by score)",
               "; ".join(f"{k} {v}" for k, v in top_tab["top_type"].value_counts().items()), None)
    checks.add("top-10 DE proteins (kidney_t15_label)", "; ".join(f"{r.protein} ({r.top_type})" for r in
                                                              top_tab.itertuples()), None)
    checks.add("top-10 proteins in the V3 + endothelial marker list", f"{int((top_tab['marker_list_rows'] != '').sum())}"
               f" of 10", None)

    # named proteins of the submitted Results
    named = []
    for g, role in T.NAMED_PROTEINS.items():
        if g not in var_names:
            named.append({"protein": g, "role_in_submitted_results": role, "quantified": False})
            continue
        j = var_names.index(g)
        sub = de_df[de_df["names"] == g].set_index("label")
        best = str(sub["scores"].idxmax())
        row = {"protein": g, "role_in_submitted_results": role, "quantified": True,
               "in_t15_top10": g in top10, "t15_top10_rank": (top10.index(g) + 1) if g in top10 else np.nan,
               "type_with_highest_score": T.display(best), "highest_score": float(sub.loc[best, "scores"]),
               "rank_in_that_type": int(sub.loc[best, "rank_in_type"]),
               "padj_in_that_type": float(sub.loc[best, "pvals_adj"]),
               "marker_list_rows": "; ".join(T.display(x) for x in listed.get(g, []))}
        claimed = T.SUBMITTED_TO_KIDNEY_ANNOTATION.get(role)
        if claimed:
            row.update({"named_type": T.display(claimed), "score_in_named_type": float(sub.loc[claimed, "scores"]),
                        "rank_in_named_type": int(sub.loc[claimed, "rank_in_type"]),
                        "padj_in_named_type": float(sub.loc[claimed, "pvals_adj"])})
        for t in rows:
            row[f"detected_{T.display(t)}"] = float(det[(lab == t).to_numpy(), j].mean())
        named.append(row)
    run.table(pd.DataFrame(named), "t15_named_proteins")

    # --- sensitivity: Unknown as a group -----------------------------------------------------------------------------
    types_u = [*types, T.UNKNOWN]
    _, cov_u = T.coverage_matrix(det, lab, types_u)
    core_u, unique_u = T.core_unique(cov_u)
    jac_u, _, _ = T.jaccard(core_u)
    lab_u = lab.replace({T.UNKNOWN: "Unassigned"})
    ad_u = T.de_adata(raw, lab_u, var_names, order=[*types, "Unassigned"])
    _, top10_u = T.de_rank(ad_u)
    sens = pd.DataFrame({"label": types_u, "display_label": [T.display(t) for t in types_u],
                         "n_core_primary": [*core.sum(axis=1), np.nan], "n_core_unknown_as_group": core_u.sum(axis=1),
                         "n_unique_core_primary": [*unique.sum(axis=1), np.nan],
                         "n_unique_core_unknown_as_group": unique_u.sum(axis=1),
                         "jaccard_with_unknown": jac_u[-1, :]})
    run.table(sens, "t15_sensitivity_unknown_as_group")
    run.table(pd.DataFrame({"rank": range(1, 11), "primary_top10": top10, "unknown_as_group_top10": top10_u}),
              "t15_sensitivity_unknown_top10")
    lost = int(unique.sum() - unique_u[:-1].sum())
    checks.add("unique-core proteins lost when Unknown counts as an 'other type' (sum over the 8 types)",
               f"{lost} of {int(unique.sum())}", None)
    checks.add("top 10 with Unknown as a DE group: proteins shared with the primary top 10",
               f"{len(set(top10) & set(top10_u))} of 10", None)

    # --- figures --------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")

    fig_rows: list[dict] = []
    fig2c(run, per_cell, rows, colors, fig_rows)
    fig2d(run, top, types, n_cells, fig_rows)
    fig2e(run, jac, types, n_cells, fig_rows)
    idx = [var_names.index(g) for g in top10]
    scaled = T.scale_var(np.asarray(ad_de.X)[:, idx])
    med = fig2f(run, scaled, ad_de.obs["group"].astype(str).to_numpy(), types, n_cells, top10, fig_rows)
    run.table(pd.DataFrame(med, index=pd.Index([T.display(t) for t in types], name="type"), columns=top10)
              .reset_index(), "t15_fig2f_medians")
    figs = pd.DataFrame(fig_rows)
    run.table(figs, "t15_figure_checks")
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}",
               bool(figs["pass"].all()))

    run.table(checks.frame(), "t15_checks")
    n_fail = int((~checks.frame()["pass"]).sum())
    run.note(f"{len(checks.rows)} rows; {n_fail} failed.")
    run.finish({"label": T.PRIMARY, "display": T.DISPLAY, "core_frac": T.CORE_FRAC,
                "unique_low_frac": T.UNIQUE_LOW_FRAC, "upset_topn": T.UPSET_TOPN, "de_top": T.DE_TOP,
                "unknown": "own group in 2C; left out of 2D-2F and Table S3; sensitivity tables keep it",
                "ids_per_cell": "proteins > 0 among the 4,054 retained proteins",
                "checks_pass": n_fail == 0},
               status="completed" if n_fail == 0 else "completed with failed checks")
    if n_fail:
        raise SystemExit(f"{n_fail} checks failed; see kidney_figure2_t15_checks.csv.")


def abbr_combo(combo: str, abbr: dict[str, str]) -> str:
    return " + ".join(abbr.get(t, t) for t in combo.split(";")) if combo != "(none)" else ""


if __name__ == "__main__":
    main()
