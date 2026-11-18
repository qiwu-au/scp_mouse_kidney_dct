#!/usr/bin/env python
"""pt_tal step 0: PT / TAL label changes between the two label sets, union subsets and their embeddings.

Analysis plan
(Inputs: "Report how many cells of each group change label between the two sets"; steps 3-4 subset UMAPs).

What it does
------------
1. Loads the kidney_object canonical kidney object with ``kidney_t15_label`` (kidney_annotation file, matched on cell_id) and
   ``kidney_pre_label`` (Fig 2A); checks the label counts against the project notes.
2. Label changes: for PCT, PST, TAL and MD, the cells in each set, in both, and where they go / come from; the full
   Fig 2A x kidney_annotation crosstab of the union cells; median protein IDs per group.
3. Union subsets (authors decision, 15 Oct 2026): PT = cells labelled PCT or PST in either set; TAL = cells labelled TAL
   in either set. The 19 Fig 2A MD cells are kept as a separate descriptive group.
4. Embedding of each subset (``pt_tal_lib.subset_embedding``): proteins detected in >= 2 subset cells;
   ln(1 + intensity) scaled within the subset (primary) or first regressed on ln(protein IDs) (sensitivity); PCA
   (arpack, 20 PCs), kNN (k = 10), UMAP (seed 42). Spearman rho of PC1-5 with ln(protein IDs).

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation kidney_annotation_cluster_labels_per_cell_a_endo.csv.
Outputs: results/tables/pt_tal/pt_tal_label_changes.csv, pt_tal_label_crosstab.csv,
         pt_tal_subsets_per_cell.csv, pt_tal_subset_pcs.csv, pt_tal_subsets_checks.csv;
         results/figures/pt_tal/pt_tal_label_changes, pt_tal_subsets_umap,
         pt_tal_subsets_umap_regressed (.pdf/.png)

Command (project root):  python scripts/pt_tal_subsets.py
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
from scipy import stats

import scp_common as cc
import cluster_de_lib as t5
import atlas_dct_lib as T8
import pt_tal_lib as T

FOCUS = ("PCT", "PST", "TAL", "MD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pt_tal step 0: label changes, union subsets, subset embeddings.")
    cc.add_common_args(parser)
    return parser.parse_args()


def label_changes(labels: pd.DataFrame) -> pd.DataFrame:
    pre, t15 = labels["kidney_pre_label"].astype(str), labels["kidney_t15_label"].astype(str)
    ids = labels["n_detected_proteins"].astype(float)
    rows = []
    for g in FOCUS:
        a, b = pre == g, t15 == g
        dest = t15[a & ~b].value_counts()
        orig = pre[b & ~a].value_counts()
        rows.append({
            "label": g, "n_fig2a": int(a.sum()), "n_t15": int(b.sum()), "n_in_both": int((a & b).sum()),
            "n_fig2a_only": int((a & ~b).sum()), "n_t15_only": int((b & ~a).sum()),
            "fig2a_cells_now_in_t15": "; ".join(f"{k} {v}" for k, v in dest.items()),
            "t15_cells_from_fig2a": "; ".join(f"{k} {v}" for k, v in orig.items()),
            "median_protein_ids_fig2a": float(ids[a].median()) if a.any() else np.nan,
            "median_protein_ids_t15": float(ids[b].median()) if b.any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "subsets", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, _, ln1p, _ = T.load_scp(run, checks)
    colors = T.label_colors(adata)
    checks.add("label colours (Fig 2 colours; PST replaced if grey or equal to PCT / TAL / MD / Unknown)",
               "; ".join(f"{k} {colors[k]}" for k in ("PCT", "PST", "TAL", "MD", "Unknown")), None)
    masks = T.union_masks(labels)
    for seg, exp in T.EXPECTED_UNION.items():
        n = int(masks[seg].sum())
        checks.add(f"{seg} union subset (segment label in either set): cells", f"{n}", n == exp, f"{exp} (plan)")
    n_md = int(masks["MD_pre"].sum())
    checks.add("Fig 2A MD cells", f"{n_md}", n_md == T.EXPECTED_MD_PRE, f"{T.EXPECTED_MD_PRE}")
    checks.add("PT and TAL union subsets overlap (cells)", f"{int((masks['PT'] & masks['TAL']).sum())}", None)
    checks.add("Fig 2A MD cells in a union subset", f"{int((masks['MD_pre'] & (masks['PT'] | masks['TAL'])).sum())}",
               None)

    lc = label_changes(labels)
    for r in lc.itertuples():
        checks.add(f"{r.label}: Fig 2A / kidney_annotation / both; Fig 2A cells now in kidney_annotation", f"{r.n_fig2a} / {r.n_t15} / "
                   f"{r.n_in_both}; {r.fig2a_cells_now_in_t15 or '-'}", None)
    union = masks["PT"] | masks["TAL"] | masks["MD_pre"]
    ct = pd.crosstab(labels.loc[union, "kidney_pre_label"], labels.loc[union, "kidney_t15_label"])
    ct.index.name = "kidney_pre_label"
    ct.columns.name = None
    same_pt = int((labels["kidney_pre_label"].isin(T.PT_LABELS)
                   & labels["kidney_pre_label"].eq(labels["kidney_t15_label"])).sum())
    n_pre_pt = int(labels["kidney_pre_label"].isin(T.PT_LABELS).sum())
    checks.add("Fig 2A PCT / PST cells with the same PT label under kidney_annotation", f"{same_pt} of {n_pre_pt}", None)

    # embeddings ---------------------------------------------------------------------------------------
    ids = labels["n_detected_proteins"].astype(float)
    per_cell, pc_rows, embeds = [], [], {}
    for seg in T.SEGMENTS:
        m = masks[seg].to_numpy()
        base = labels.loc[m, ["kidney_t15_label", "kidney_pre_label", "kidney_reannot_label", "t15_cluster",
                              "n_detected_proteins"]].copy()
        base.insert(0, "segment", seg)
        for graph in T.GRAPHS:
            a = T.subset_embedding(ln1p.loc[m], ids[m].to_numpy(), regressed=graph == "regressed")
            embeds[(seg, graph)] = a
            base[f"umap1_{graph}"] = a.obsm["X_umap"][:, 0]
            base[f"umap2_{graph}"] = a.obsm["X_umap"][:, 1]
            vr = a.uns["pca"]["variance_ratio"]
            rec = {"segment": seg, "graph": graph, "n_cells": int(m.sum()), "n_proteins": a.n_vars,
                   "variance_pc1_20": float(vr.sum())}
            for k in range(5):
                r = stats.spearmanr(a.obsm["X_pca"][:, k], np.log(ids[m]))
                rec[f"rho_pc{k + 1}_ln_ids"] = float(r.statistic)
            rec.update({f"variance_ratio_pc{k + 1}": float(v) for k, v in enumerate(vr)})
            pc_rows.append(rec)
            checks.add(f"{seg} / {graph}: proteins; variance PC1-20; rho PC1-5 vs ln(IDs)",
                       f"{a.n_vars}; {vr.sum():.3f}; " + ", ".join(f"{rec[f'rho_pc{k + 1}_ln_ids']:.2f}"
                                                                   for k in range(5)), None)
        per_cell.append(base)
    per_cell = pd.concat(per_cell).reset_index()
    per_cell["in_md_fig2a"] = per_cell["cell_id"].map(masks["MD_pre"]).astype(bool)

    fig_rows = []
    figures(run, labels, per_cell, ct, lc, colors, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(lc, "label_changes")
    run.table(ct.reset_index(), "label_crosstab")
    run.table(per_cell, "subsets_per_cell")
    run.table(pd.DataFrame(pc_rows), "subset_pcs")
    run.table(figs, "subsets_figure_checks")
    run.table(checks.frame(), "subsets_checks")
    run.finish({"union_rule": "segment label (PT: PCT or PST; TAL: TAL) in kidney_t15_label or kidney_pre_label",
                "protein_filter": f"detected in >= {cc.MIN_PROTEIN_CELLS} subset cells",
                "n_pcs": T.N_PCS, "n_neighbors": T.N_NEIGHBORS, "pca": "arpack", "umap_seed": T.SEED,
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see pt_tal_subsets_checks.csv.")


def figures(run, labels, per_cell, ct: pd.DataFrame, lc: pd.DataFrame, colors, fig_rows) -> None:
    plt = T.setup()
    from matplotlib import colormaps

    cmap_seq = T8.sequential_cmap()
    LH = T.LH
    # 1. label changes heatmap ---------------------------------------------------------------------------
    rows = [r for r in (*FOCUS, "DVR", "JG", "Unknown") if r in ct.index]
    rows += [r for r in ct.index if r not in rows]
    cols = [c for c in ("PCT", "PST", "TAL", "Endothelial", "JG", "Unknown") if c in ct.columns]
    cols += [c for c in ct.columns if c not in cols]
    sub = ct.reindex(index=rows, columns=cols, fill_value=0)
    counts = sub.to_numpy()
    frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    title = "Cells of the PT / TAL union subsets and the Fig 2A MD cells: Fig 2A label (rows) vs kidney_annotation label (columns)"
    note = ("Union subsets: PT = PCT or PST in either label set (356 cells); TAL = TAL in either set (154); plus the "
            "19 Fig 2A MD cells. Numbers: cells; fill: fraction of the Fig 2A row. Fig 2A = kidney_pre_label; kidney_annotation = "
            "kidney_t15_label (primary).")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    row_txt = [f"{r} ({int(sub.loc[r].sum())})" for r in rows]
    col_txt = [f"{c} ({int(sub[c].sum())})" for c in cols]
    lw = max(T8.text_mm(r)[0] for r in row_txt) + 1.0 + 1.6 + 0.8 + 6.0
    cw, rh = 12.0, 4.6
    col_h = max(T8.text_mm(c)[0] for c in col_txt)
    H = T8.M + title_h + LH + 1.5 + len(rows) * rh + 0.8 + col_h + 2.0 + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    x0 = T8.M + lw
    y0 = T8.M + title_h + LH + 1.5
    sh.text(T8.M, T8.M + title_h, "Rows: Fig 2A label (row total); columns: kidney_annotation label (column total)", va="top")
    right, _ = T.count_heatmap(sh, x0, y0, counts, frac, row_txt, col_txt, cw, rh, cmap_seq,
                               row_colors=[colors.get(r, T.INK["muted"]) for r in rows])
    T.colorbar_vertical(sh, right + 6.0, y0, min(20.0, len(rows) * rh), cmap_seq, "Fraction of row")
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "label_changes", title, fig_rows)

    # 2. subset UMAPs ----------------------------------------------------------------------------------------
    cmap_ids = colormaps["viridis"]
    for graph in T.GRAPHS:
        title = f"PT and TAL union subsets: UMAP of each subset ({'primary' if graph == 'primary' else 'depth-regressed'})"
        note = (f"Each subset embedded on its own: {T.GRAPH_TEXT[graph]}; proteins detected in >= 2 subset cells; "
                "PCA (20 PCs), kNN (k = 10), UMAP (seed 42). Left: kidney_annotation labels; middle: Fig 2A labels; right: ln(protein "
                "IDs per cell). Numbers: cells.")
        note_text, note_h = T8.note_block(note)
        title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
        pw, ph, gap = 52.0, 40.0, 7.0
        # pre-compute legend heights
        blocks = []
        for seg in T.SEGMENTS:
            d = per_cell[per_cell["segment"] == seg]
            leg_h = 0.0
            for col in T.LABELS:
                order = order_labels(d[col])
                ent = [(f"{k} ({int((d[col] == k).sum())})", "") for k in order]
                leg_h = max(leg_h, len(T.legend_rows(ent, pw)) * (LH + 0.4))
            blocks.append((seg, d, leg_h))
        row_h = [LH + 1.5 + LH + 0.8 + ph + 0.8 + max(lh, 1.8 + 0.5 + LH + 1.0) + 2.0 for _, _, lh in blocks]
        H = T8.M + title_h + sum(row_h) + note_h + T8.M
        sh = T8.Sheet(H)
        sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
        y = T8.M + title_h
        for (seg, d, _), rh_ in zip(blocks, row_h):
            sh.text(T8.M, y, f"{seg} union subset ({len(d)} cells)", fontweight="bold", va="top")
            yy = y + LH + 1.5
            xy = d[[f"umap1_{graph}", f"umap2_{graph}"]].to_numpy(float)
            for k, col in enumerate(T.LABELS):
                T.umap_categorical(sh, T8.M + k * (pw + gap), yy, pw, ph, xy, d[col], order_labels(d[col]), colors,
                                   T.LABEL_DISPLAY[col])
            T.umap_continuous(sh, T8.M + 2 * (pw + gap), yy, pw, ph, xy, np.log(d["n_detected_proteins"].astype(float)),
                              "ln(protein IDs)", cmap_ids)
            y += rh_
        sh.note(note_text, note_h)
        T8.save_checked(run, sh, "subsets_umap" if graph == "primary" else "subsets_umap_regressed", title, fig_rows)


order_labels = T.order_labels


if __name__ == "__main__":
    main()
