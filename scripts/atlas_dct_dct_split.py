#!/usr/bin/env python
"""atlas_dct step 1b: atlas_dct-derived DCT1 / DCT2 split of the atlas DCT cells (option 1a, authors 15 Oct 2026).

Analysis plan,
steps 1, 3-6. The atlas has no DCT1 / DCT2 labels (step 1a); this split is derived here and is labelled
"atlas_dct-derived" wherever it is used.

Rule (fixed before running)
---------------------------
1. Cells: atlas Celltype DCT (16,381). Graph: scanpy kNN (k = 15) on the stored scVI latent obsm['X_scVI'],
   random_state 42. Leiden (igraph) at resolutions 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, seed 42.
2. Naming genes: DCT V1 workbook DCT1 row (Pvalb, Egf, Trpm7, Col5a2, Ptgfr) and DCT2 row (Slc8a1, Calb1, S100g,
   Klk1, Trpv5), without Umod and Slc12a3 (authors, 15 Oct 2026).
3. Per subcluster and naming gene: z = (subcluster mean - mean over all DCT cells) / SD over all DCT cells of the
   log-normalized expression (10,000 per cell, log1p). DCT1 score = mean z of the DCT1 genes, DCT2 score = mean z of
   the DCT2 genes, difference = DCT1 score - DCT2 score.
4. Name: DCT1 (atlas_dct) if difference >= 0.25, DCT2 (atlas_dct) if <= -0.25, otherwise DCT unassigned (atlas_dct). Thresholds 0.10
   and 0.50 are reported alongside.
5. Primary resolution: the lowest resolution at which at least one DCT1 and one DCT2 subcluster occur.
   Stability: per-cell agreement of the split label with the primary across resolutions, and across 11 Leiden
   seeds (42, 0-9) at the primary resolution (fraction identical; ARI).

Also reported per subcluster: n cells, Source / Origin composition, median counts, detection of context genes
(not used for naming). UMAP (scanpy, random_state 42) of the DCT graph for the figure.

Inputs (read-only): results/h5ad/atlas_dct_atlas_dct_region.h5ad; data/DCT/DCT_markers_V1.xlsx.
Outputs: results/tables/atlas_dct/atlas_dct_dct_split_per_cell.csv, atlas_dct_dct_split_subclusters.csv,
         atlas_dct_dct_split_stability.csv, atlas_dct_dct_split_checks.csv;
         results/figures/atlas_dct/atlas_dct_dct_split_umap.pdf/.png, atlas_dct_dct_split_naming.pdf/.png,
         atlas_dct_dct_split_figure_checks.csv

Command (project root):  python scripts/atlas_dct_dct_split.py
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
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as L7
import atlas_dct_lib as T

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
CONTEXT_GENES = ("Slc12a3", "Pvalb", "Calb1", "Slc8a1", "Trpv5", "Umod", "Slc12a1", "Aqp2", "Mki67", "Top2a")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atlas_dct step 1b: atlas_dct-derived DCT1 / DCT2 split of the atlas DCT cells.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks, msg: str) -> None:
    run.table(checks.frame(), "dct_split_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit(msg)


def name_of(diff: float, thr: float) -> str:
    return T.DCT1_ATLAS_DCT if diff >= thr else (T.DCT2_ATLAS_DCT if diff <= -thr else T.DCT_UNASSIGNED)


def naming_table(clusters: np.ndarray, Z: np.ndarray, genes: list[str], lists: dict[str, list[str]]) -> pd.DataFrame:
    """Per subcluster: n, z per naming gene, DCT1 / DCT2 scores, difference, names at each threshold."""
    rows = []
    for c in sorted(np.unique(clusters), key=lambda s: int(s)):
        m = clusters == c
        z = Z[m].mean(axis=0)
        rec = {"subcluster": f"s{c}", "n_cells": int(m.sum())}
        rec.update({f"z_{g}": float(v) for g, v in zip(genes, z)})
        s1 = float(np.mean([z[genes.index(g)] for g in lists["DCT1"]]))
        s2 = float(np.mean([z[genes.index(g)] for g in lists["DCT2"]]))
        rec.update(dct1_score=s1, dct2_score=s2, difference=s1 - s2)
        for thr in T.NAMING_THRESHOLDS:
            rec[f"name_thr_{thr:.2f}"] = name_of(s1 - s2, thr)
        rows.append(rec)
    return pd.DataFrame(rows)


def cell_labels(clusters: np.ndarray, tab: pd.DataFrame, thr: float = T.NAMING_THRESHOLD) -> np.ndarray:
    lab = dict(zip(tab["subcluster"], tab[f"name_thr_{thr:.2f}"]))
    return np.array([lab[f"s{c}"] for c in clusters])


def main() -> None:
    import scanpy as sc

    args = parse_args()
    run = cc.Run(T.TASK, "dct_split", args)
    t0 = time.time()
    checks = t5.Checks("atlas")
    checks.add("scanpy igraph RNG compatibility patch applied (_scanpy_compat.py)", f"{SCANPY_COMPAT_PATCHED}", None)
    lists = T.naming_lists(run, checks)
    region = T.load_region(run, with_split=False)
    dct = region[region.obs["rna_group"].to_numpy() == "DCT"].copy()
    del region
    checks.add("atlas Celltype DCT cells", f"{dct.n_obs}", dct.n_obs == 16381, "16,381 (step 1a)")
    look = L7.atlas_gene_lookup(dct.var_names)
    listed = [*lists["DCT1"], *lists["DCT2"]]
    absent = [g for g in listed if g.lower() not in look]
    # Absent genes are reported, not fatal (the project notes coverage rule): the scores use the genes in the atlas.
    checks.add("naming genes in the atlas genes (absent genes are left out of the scores)",
               f"{len(listed) - len(absent)} of {len(listed)}; absent: {', '.join(absent) or 'none'}", None)
    lists = {k: [g for g in v if g.lower() in look] for k, v in lists.items()}
    checks.add("naming genes used: DCT1; DCT2 (at least 3 each)", f"{', '.join(lists['DCT1'])}; "
               f"{', '.join(lists['DCT2'])}", min(len(v) for v in lists.values()) >= 3)
    if not checks.ok:
        stop(run, checks, "Too few naming genes in the atlas; see atlas_dct_dct_split_checks.csv.")
    genes = [*lists["DCT1"], *lists["DCT2"]]
    agenes = [look[g.lower()] for g in genes]
    ln = T.lognorm(dct.X)
    idx = dct.var_names.get_indexer(agenes)
    Xn = ln[:, idx].toarray().astype(np.float64)
    mu, sd = Xn.mean(axis=0), Xn.std(axis=0, ddof=0)
    checks.add("naming genes: detection in atlas DCT cells (fraction with counts > 0)",
               "; ".join(f"{g} {f:.3f}" for g, f in
                         zip(genes, np.asarray((dct.X[:, idx] > 0).mean(axis=0)).ravel())), None)
    checks.add("naming genes with zero SD (z undefined)", f"{int((sd == 0).sum())}", bool((sd > 0).all()))
    Z = (Xn - mu) / sd

    # graph and subclusters --------------------------------------------------------------------------------
    sc.pp.neighbors(dct, use_rep="X_scVI", n_neighbors=T.SPLIT_N_NEIGHBORS, random_state=T.SEED)
    per_res, tabs = {}, []
    for res in T.SPLIT_RESOLUTIONS:
        cl = t5.leiden(dct, res, f"leiden_{res}", seed=T.SEED).to_numpy()
        tab = naming_table(cl, Z, genes, lists)
        tab.insert(0, "resolution", res)
        tab.insert(1, "seed", T.SEED)
        tabs.append(tab)
        per_res[res] = (cl, tab)
        names = tab[f"name_thr_{T.NAMING_THRESHOLD:.2f}"]
        checks.add(f"resolution {res}: subclusters; named DCT1 / DCT2 / unassigned (threshold {T.NAMING_THRESHOLD})",
                   f"{len(tab)}; {int((names == T.DCT1_ATLAS_DCT).sum())} / {int((names == T.DCT2_ATLAS_DCT).sum())} / "
                   f"{int((names == T.DCT_UNASSIGNED).sum())}", None)
    primary = next((r for r in T.SPLIT_RESOLUTIONS
                    if {T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT} <= set(per_res[r][1][f"name_thr_{T.NAMING_THRESHOLD:.2f}"])), None)
    checks.add("primary resolution (lowest with at least one DCT1 and one DCT2 subcluster)", f"{primary}",
               primary is not None)
    if primary is None:
        stop(run, checks, "No resolution gives both DCT1 and DCT2 subclusters; see atlas_dct_dct_split_checks.csv.")
    cl0, tab0 = per_res[primary]
    lab0 = cell_labels(cl0, tab0)
    vc = pd.Series(lab0).value_counts()
    checks.add("primary split: cells DCT1 / DCT2 / unassigned (atlas_dct)",
               f"{int(vc.get(T.DCT1_ATLAS_DCT, 0))} / {int(vc.get(T.DCT2_ATLAS_DCT, 0))} / {int(vc.get(T.DCT_UNASSIGNED, 0))}",
               None)

    # composition of the subclusters (all resolutions) ---------------------------------------------------
    obs = dct.obs
    ctx = [g for g in CONTEXT_GENES if g.lower() in look]
    cidx = dct.var_names.get_indexer([look[g.lower()] for g in ctx])
    cdet = (dct.X[:, cidx] > 0).toarray()
    comp_rows = []
    for res, (cl, tab) in per_res.items():
        for c in sorted(np.unique(cl), key=int):
            m = cl == c
            sub = obs.loc[m]
            org = sub["Origin"].value_counts()
            rec = {"resolution": res, "subcluster": f"s{c}", "n_cells": int(m.sum()),
                   "n_nuclei": int((sub["Source"] == "Nuclei").sum()), "n_cells_sc": int((sub["Source"] == "Cell").sum()),
                   "top_origin": org.index[0], "top_origin_share": float(org.iloc[0] / m.sum()),
                   "origins": "; ".join(f"{k} {v}" for k, v in org.items()),
                   "median_total_counts": float(np.median(np.asarray(dct.X[m].sum(axis=1)).ravel()))}
            rec.update({f"det_{g}": float(cdet[m, j].mean()) for j, g in enumerate(ctx)})
            comp_rows.append(rec)
    comp = pd.DataFrame(comp_rows)
    subclusters = pd.concat(tabs, ignore_index=True).merge(comp, on=["resolution", "subcluster", "n_cells"])
    ari_origin = adjusted_rand_score(obs["Origin"].astype(str), cl0)
    checks.add(f"primary subclusters vs Origin (ARI); vs Source (ARI)",
               f"{ari_origin:.3f}; {adjusted_rand_score(obs['Source'].astype(str), cl0):.3f}", None)

    # stability ----------------------------------------------------------------------------------------------
    stab = []
    for res, (cl, tab) in per_res.items():
        lab = cell_labels(cl, tab)
        stab.append({"comparison": "resolution", "resolution": res, "seed": T.SEED,
                     "n_subclusters": len(tab), "n_DCT1": int((lab == T.DCT1_ATLAS_DCT).sum()),
                     "n_DCT2": int((lab == T.DCT2_ATLAS_DCT).sum()), "n_unassigned": int((lab == T.DCT_UNASSIGNED).sum()),
                     "fraction_same_label_as_primary": float((lab == lab0).mean()),
                     "ari_label_vs_primary": float(adjusted_rand_score(lab0, lab)),
                     "ari_subclusters_vs_primary": float(adjusted_rand_score(cl0, cl))})
    for seed in T.SPLIT_SEEDS:
        cl = t5.leiden(dct, primary, f"leiden_seed_{seed}", seed=seed).to_numpy()
        tab = naming_table(cl, Z, genes, lists)
        lab = cell_labels(cl, tab)
        stab.append({"comparison": "seed", "resolution": primary, "seed": seed, "n_subclusters": len(tab),
                     "n_DCT1": int((lab == T.DCT1_ATLAS_DCT).sum()), "n_DCT2": int((lab == T.DCT2_ATLAS_DCT).sum()),
                     "n_unassigned": int((lab == T.DCT_UNASSIGNED).sum()),
                     "fraction_same_label_as_primary": float((lab == lab0).mean()),
                     "ari_label_vs_primary": float(adjusted_rand_score(lab0, lab)),
                     "ari_subclusters_vs_primary": float(adjusted_rand_score(cl0, cl))})
    stab = pd.DataFrame(stab)
    s_seed = stab[stab["comparison"] == "seed"]
    checks.add(f"seeds at resolution {primary}: fraction of cells with the primary split label (min; median)",
               f"{s_seed['fraction_same_label_as_primary'].min():.3f}; "
               f"{s_seed['fraction_same_label_as_primary'].median():.3f}", None)
    checks.add(f"seeds at resolution {primary}: seeds giving both DCT1 and DCT2 cells",
               f"{int(((s_seed['n_DCT1'] > 0) & (s_seed['n_DCT2'] > 0)).sum())} of {len(s_seed)}", None)

    # per-cell table and UMAP --------------------------------------------------------------------------------
    sc.tl.umap(dct, random_state=T.SEED)
    um = dct.obsm["X_umap"]
    per_cell = pd.DataFrame({"atlas_cell": dct.obs_names.astype(str), "Origin": obs["Origin"].to_numpy(),
                             "Source": obs["Source"].to_numpy(), "Technology": obs["Technology"].to_numpy(),
                             "subcluster": [f"s{c}" for c in cl0], "split_label": lab0,
                             "dct1_minus_dct2_cell": Z[:, :len(lists["DCT1"])].mean(axis=1)
                             - Z[:, len(lists["DCT1"]):].mean(axis=1),
                             "umap_1": um[:, 0], "umap_2": um[:, 1]})
    for thr in T.NAMING_THRESHOLDS:
        if thr != T.NAMING_THRESHOLD:
            per_cell[f"split_label_thr_{thr:.2f}"] = cell_labels(cl0, tab0, thr)
    for res, (cl, tab) in per_res.items():
        per_cell[f"split_label_res_{res}"] = cell_labels(cl, tab)
    run.table(per_cell, "dct_split_per_cell")
    run.table(subclusters, "dct_split_subclusters")
    run.table(stab, "dct_split_stability")

    # figures --------------------------------------------------------------------------------------------------
    fig_rows = []
    figures(run, per_cell, subclusters[subclusters["resolution"] == primary], genes, lists, primary, fig_rows)
    figs = pd.DataFrame(fig_rows)
    run.table(figs, "dct_split_figure_checks")
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(checks.frame(), "dct_split_checks")
    run.finish({"cells": "atlas Celltype DCT", "graph": f"kNN k={T.SPLIT_N_NEIGHBORS} on obsm['X_scVI']",
                "resolutions": list(T.SPLIT_RESOLUTIONS), "seeds": list(T.SPLIT_SEEDS), "primary_resolution": primary,
                "naming_genes": lists, "naming_excluded": list(T.NAMING_EXCLUDED),
                "naming_threshold": T.NAMING_THRESHOLD, "naming_thresholds_reported": list(T.NAMING_THRESHOLDS),
                "runtime_s": round(time.time() - t0, 1), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see atlas_dct_dct_split_checks.csv.")


def legend_block(sh, x, y, items, ncol, col_w):
    LH = T.text_mm("Hg")[1]
    for i, (lab, col) in enumerate(items):
        r, c = divmod(i, ncol)
        xx, yy = x + c * col_w, y + r * (LH + 0.6)
        sh.rect(xx, yy + LH / 2 - 0.8, 1.6, 1.6, facecolor=col, edgecolor="none")
        sh.text(xx + 2.4, yy + LH / 2, lab, va="center_baseline")
    return (len(items) + ncol - 1) // ncol * (LH + 0.6)


def figures(run, pc: pd.DataFrame, sub: pd.DataFrame, genes, lists, res, fig_rows) -> None:
    plt = T.setup_matplotlib()
    # 1. UMAP: subclusters, split label, study ------------------------------------------------------------
    LH = T.text_mm("Hg")[1]
    pw, gap = 54.0, 5.0
    subs = sorted(pc["subcluster"].unique(), key=lambda s: int(s[1:]))
    sub_cols = {s: (T.SERIES + T.BLUE_RAMP[2:] + ["#9c2423", "#b07800", "#5f5e5a"])[i % 15] for i, s in enumerate(subs)}
    lab_cols = {k: T.RNA_PALETTE[k] for k in (T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT, T.DCT_UNASSIGNED)}
    origins = pc["Origin"].value_counts().index.tolist()
    org_cols = {o: T.SERIES[i % len(T.SERIES)] for i, o in enumerate(origins)}
    nm = dict(zip(sub["subcluster"], sub[f"name_thr_{T.NAMING_THRESHOLD:.2f}"]))
    panels = [(f"Leiden subclusters (resolution {res})", "subcluster", sub_cols,
               [(f"{s} ({nm[s].split(' ')[0] if nm[s] != T.DCT_UNASSIGNED else 'unassigned'})", sub_cols[s]) for s in subs]),
              ("atlas_dct-derived split", "split_label", lab_cols,
               [(f"{k} {int((pc['split_label'] == k).sum()):,}", c) for k, c in lab_cols.items()]),
              ("Study (atlas Origin)", "Origin", org_cols,
               [(f"{o} {int((pc['Origin'] == o).sum()):,}", org_cols[o]) for o in origins])]
    title_h = LH + 1.2
    leg_h = max((len(p[3]) + 1) // 2 for p in panels) * (LH + 0.6)
    note = (f"Atlas Celltype DCT cells (n = {len(pc):,}); UMAP of the kNN graph on the stored scVI latent (k = 15). "
            f"Subclusters named by the DCT V1 workbook DCT1 / DCT2 genes without Umod and Slc12a3 (standardized mean "
            f"difference >= {T.NAMING_THRESHOLD}); not atlas labels.")
    note_text, note_h = T.note_block(note)
    H = T.M + title_h + pw + 2.0 + leg_h + note_h + T.M
    sh = T.Sheet(H)
    order = np.random.default_rng(T.SEED).permutation(len(pc))
    for k, (title, col, cols, items) in enumerate(panels):
        x = T.M + k * (pw + gap)
        sh.text(x, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
        ax = sh.ax(x, T.M + title_h, pw, pw)
        ax.scatter(pc["umap_1"].to_numpy()[order], pc["umap_2"].to_numpy()[order],
                   c=[cols[v] for v in pc[col].to_numpy()[order]], s=0.15, linewidths=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        legend_block(sh, x, T.M + title_h + pw + 2.0, items, 2, pw / 2)
    sh.note(note_text, note_h)
    T.save_checked(run, sh, "dct_split_umap", "atlas_dct-derived DCT1 / DCT2 split: UMAP", fig_rows)

    # 2. naming heatmap -----------------------------------------------------------------------------------------
    cmap = T.diverging_cmap()
    zcols = [f"z_{g}" for g in genes]
    mat = sub[zcols].to_numpy(float)
    rows = [f"{s} (n = {n:,})" for s, n in zip(sub["subcluster"], sub["n_cells"])]
    lw = max(T.text_mm(r)[0] for r in rows) + 2.0
    cw, rh = 7.0, 4.0
    gh = max(T.text_mm(g)[0] for g in genes) + 1.5
    bw = 30.0
    title = f"Naming genes per subcluster (resolution {res}): z vs all atlas DCT cells"
    note = ("z = (subcluster mean - mean over all atlas DCT cells) / SD over cells, log-normalized expression. "
            "Difference = mean z of the DCT1 genes - mean z of the DCT2 genes; dashed lines at "
            f"+/-{T.NAMING_THRESHOLD} (naming threshold).")
    note_text, note_h = T.note_block(note)
    grp_h = LH + 1.5
    H = T.M + title_h + grp_h + len(rows) * rh + gh + 3.0 + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    x0, y0 = T.M + lw, T.M + title_h + grp_h
    for gname, xs, xe in (("DCT1 genes", 0, len(lists["DCT1"])), ("DCT2 genes", len(lists["DCT1"]), len(genes))):
        sh.text(x0 + (xs + xe) / 2 * cw, y0 - 1.0, gname, ha="center", va="bottom")
    ax = sh.ax(x0, y0, len(genes) * cw, len(rows) * rh)
    ax.imshow(mat, cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto", interpolation="nearest",
              extent=(0, len(genes), len(rows), 0))
    ax.axvline(len(lists["DCT1"]), color="white", lw=1.5)
    ax.set_axis_off()
    for i, r in enumerate(rows):
        sh.text(x0 - 1.0, y0 + (i + 0.5) * rh, r, ha="right", va="center_baseline")
    for j, g in enumerate(genes):
        sh.text(x0 + (j + 0.5) * cw, y0 + len(rows) * rh + 0.8, g, rotation=90, ha="center", va="top")
    bx = x0 + len(genes) * cw + 6.0
    axb = sh.ax(bx, y0, bw, len(rows) * rh)
    d = sub["difference"].to_numpy(float)
    names = sub[f"name_thr_{T.NAMING_THRESHOLD:.2f}"].tolist()
    axb.barh(np.arange(len(rows)) + 0.5, d, height=0.7, color=[T.RNA_PALETTE[n] for n in names])
    lim = max(1.0, float(np.abs(d).max()) * 1.1)
    axb.set_xlim(-lim, lim)
    axb.set_ylim(len(rows), 0)
    axb.axvline(0, color=T.INK["secondary"], lw=0.5)
    for t_ in (-T.NAMING_THRESHOLD, T.NAMING_THRESHOLD):
        axb.axvline(t_, color=T.INK["muted"], lw=0.5, ls="--")
    axb.set_yticks([])
    axb.spines["left"].set_visible(False)
    T.style_axes(axb)
    axb.set_xlabel("DCT1 - DCT2 score", labelpad=1)
    sh.text(bx, y0 - 1.0, "Difference", va="bottom")
    nx = bx + bw + 3.0
    for i, n in enumerate(names):
        sh.text(nx, y0 + (i + 0.5) * rh, n.replace(" (atlas_dct)", ""), va="center_baseline", color=T.RNA_PALETTE[n]
                if n != T.DCT_UNASSIGNED else T.INK["secondary"])
    cax = sh.ax(nx + 22.0, y0, 2.4, min(len(rows) * rh, 22.0))
    cax.imshow(np.linspace(1.5, -1.5, 256)[:, None], cmap=cmap, vmin=-1.5, vmax=1.5, aspect="auto",
               extent=(0, 1, -1.5, 1.5))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([-1.5, 0, 1.5])
    cax.tick_params(length=1.5, width=0.4, labelsize=T.FONT, pad=1)
    sh.text(nx + 22.0 + 2.4 + 6.5, y0 + min(len(rows) * rh, 22.0) / 2, "z", va="center")
    sh.note(note_text, note_h)
    T.save_checked(run, sh, "dct_split_naming", "atlas_dct-derived split: naming genes per subcluster", fig_rows)


if __name__ == "__main__":
    main()
