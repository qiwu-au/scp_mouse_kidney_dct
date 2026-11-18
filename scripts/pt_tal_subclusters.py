#!/usr/bin/env python
"""pt_tal step 4: unsupervised check of the PT and TAL union subsets; mapping to atlas subtype centroids.

Analysis plan, step 4.

Method (approved plan, 15 Oct 2026)
-----------------------------------
1. Subset graphs: rebuilt with ``pt_tal_lib.subset_embedding`` exactly as step 0 (checked against the step-0 PCA
   variance ratios and UMAP coordinates): primary (scaled ln(1 + intensity)) and depth-regressed.
2. Leiden (igraph, ``cluster_de_lib.leiden``) at resolutions 0.2, 0.3, 0.5; seed 42 and seeds 0-9 (ARI vs seed 42);
   clusters < 5 cells -> 'below_min_size'. Primary resolution per graph = the lowest with >= 2 clusters of >= 5 cells
   at seed 42. eta^2 of ln(protein IDs) on the subclusters; ARI with each label set.
3. DE between subclusters (primary resolution, both graphs): one subcluster vs the other subset cells, Wilcoxon on
   ln(1 + intensity) (``cluster_de_lib.de_one_vs_rest``: proteins detected in >= 20% of the subcluster; BH within
   subcluster); top 25 significant up proteins; overlap with the step-1 panels (kidney_annotation and Fig 2A).
4. Mapping of each subset cell to atlas subtype centroids (cosine):
   - classes: PT -> PTS1, PTS2, PTS3, PTS3T2; TAL -> CTAL, MTAL (primary) and CTAL, MTAL, MD (sensitivity);
   - informative genes: union of the step-1 one-vs-rest panel genes of the segment's classes (both label sets);
     sensitivity: every gene passing the step-1 RNA filters for those contrasts with the protein detected in >= 10%
     of the subset cells (no top-30 cap);
   - centroids: atlas class mean of log-normalized expression minus the segment mean, divided by the segment SD
     (all atlas cells of the segment's classes; step-1 tables), so that two classes (TAL) are also defined;
   - SCP cells: ln(1 + intensity) z-scored across the subset cells (clip 10); sensitivity: canonical .X;
   - low confidence: top score < 0.20 or margin < 0.05 (kidney_rna / atlas_dct thresholds).
5. Confusion of subclusters and mapped classes against both label sets.

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation labels; pt_tal step-0/1/3 tables (subsets_per_cell,
                    subset_pcs, panels, atlas_de, atlas_class_stats, atlas_segment_stats, scores_per_cell).
Outputs: results/tables/pt_tal/pt_tal_sub_*.csv, pt_tal_map_*.csv, pt_tal_subclusters_checks.csv;
         results/figures/pt_tal/pt_tal_subclusters_umap, pt_tal_subclusters_crosstab,
         pt_tal_map_confusion (.pdf/.png)

Command (project root):  python scripts/pt_tal_subclusters.py
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

PCA_TOL = 1e-6
MAP_CLASSES = {("PT", "primary"): T.PT_CLASSES, ("TAL", "primary"): T.TAL_CLASSES,
               ("TAL", "with_md"): (*T.TAL_CLASSES, T.MD_CLASS)}
MAP_SEGMENT_STATS = {("PT", "primary"): "PT", ("TAL", "primary"): "TAL", ("TAL", "with_md"): "TAL+MD"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pt_tal step 4: subclusters, DE, atlas mapping.")
    cc.add_common_args(parser)
    return parser.parse_args()


def apply_min_size(cl: pd.Series, seg: str) -> pd.Series:
    sizes = cl.value_counts()
    small = set(sizes[sizes < T.MIN_CLUSTER].index)
    return cl.map(lambda c: T.BELOW_MIN if c in small else f"{seg}-s{c}")


def cosine(cell_x: np.ndarray, cent: pd.DataFrame) -> pd.DataFrame:
    c = cent.to_numpy(float)
    num = cell_x @ c.T
    den = np.linalg.norm(cell_x, axis=1)[:, None] * np.linalg.norm(c, axis=1)[None, :]
    return pd.DataFrame(num / np.where(den > 0, den, np.nan), columns=cent.index)


def summarize(scores: pd.DataFrame, index) -> pd.DataFrame:
    s = scores.to_numpy()
    o = np.argsort(-np.nan_to_num(s, nan=-9), axis=1)
    ar = np.arange(len(s))
    out = pd.DataFrame({"top_class": scores.columns.to_numpy()[o[:, 0]], "top_score": s[ar, o[:, 0]],
                        "second_class": scores.columns.to_numpy()[o[:, 1]], "second_score": s[ar, o[:, 1]]}, index=index)
    out["margin"] = out["top_score"] - out["second_score"]
    out["low_confidence"] = (out["top_score"] < T.SCORE_MIN) | (out["margin"] < T.MARGIN_MIN) | out["top_score"].isna()
    return out


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "subclusters", args)
    T.install_igraph_hook()
    checks = t5.Checks(T.DATASET)
    adata, labels, xz, ln1p, det = T.load_scp(run, checks)
    sub = T.read_table(run, "subsets_per_cell", dtype={"cell_id": str})
    pcs = T.read_table(run, "subset_pcs")
    panels = T.read_table(run, "panels")
    de_atlas = T.read_table(run, "atlas_de")
    cs = T.read_table(run, "atlas_class_stats")
    ss = T.read_table(run, "atlas_segment_stats")
    scores = T.read_table(run, "scores_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    v3 = T.v3_marker_genes(run)
    ids = labels["n_detected_proteins"].astype(float)

    clus_rows, scan_rows, de_rows, de_sum_rows, cross_rows, score_rows = [], [], [], [], [], []
    per_cell = []
    primary_res = {}
    for seg in T.SEGMENTS:
        d = sub[sub["segment"] == seg].set_index("cell_id")
        cells = d.index
        pc_seg = pd.DataFrame(index=cells)
        for graph in T.GRAPHS:
            a = T.subset_embedding(ln1p.loc[cells], ids.loc[cells].to_numpy(), regressed=graph == "regressed")
            ref = pcs[(pcs["segment"] == seg) & (pcs["graph"] == graph)].iloc[0]
            vr = a.uns["pca"]["variance_ratio"]
            dvr = float(np.max(np.abs(vr - ref[[f"variance_ratio_pc{k + 1}" for k in range(T.N_PCS)]].to_numpy(float))))
            checks.add(f"{seg} / {graph}: rebuilt subset PCA equals step 0 (max abs difference of variance ratios)",
                       f"{dvr:.1e}", dvr < PCA_TOL, f"< {PCA_TOL:g}")
            du = float(np.max(np.abs(a.obsm["X_umap"] - d[[f"umap1_{graph}", f"umap2_{graph}"]].to_numpy(float))))
            checks.add(f"{seg} / {graph}: rebuilt UMAP vs step 0 (max abs difference; information)", f"{du:.1e}", None)
            chosen = None
            for res in T.SUB_RESOLUTIONS:
                base = apply_min_size(t5.leiden(a, res, "tmp", T.SEED).reindex(cells), seg)
                aris = [t5.agreement(apply_min_size(t5.leiden(a, res, "tmp", s_).reindex(cells), seg), base)["ari"]
                        for s_ in T.SUB_SEEDS]
                ok = base != T.BELOW_MIN
                n_ok = int(base[ok].nunique())
                rec = {"segment": seg, "graph": graph, "resolution": res, "n_cells": len(cells),
                       "n_clusters_ge_min": n_ok, "n_cells_below_min": int((~ok).sum()),
                       "cluster_sizes": "; ".join(f"{k} {v}" for k, v in base.value_counts().sort_index().items()),
                       "seed_ari_median": float(np.median(aris)), "seed_ari_min": float(np.min(aris)),
                       "seed_ari_max": float(np.max(aris)),
                       "eta2_ln_ids": T.eta_squared(np.log(ids.loc[cells][ok]), base[ok]) if n_ok > 1 else np.nan,
                       "kruskal_p_ids": (float(stats.kruskal(*[ids.loc[cells][ok][base[ok] == k]
                                                              for k in base[ok].unique()]).pvalue)
                                         if n_ok > 1 else np.nan)}
                for lab_col in T.LABELS:
                    rec[f"ari_{T.LABEL_SHORT[lab_col]}"] = t5.agreement(base, labels.loc[cells, lab_col])["ari"]
                scan_rows.append(rec)
                pc_seg[f"sub_{graph}_r{res:.1f}"] = base
                if chosen is None and n_ok >= 2:
                    chosen = res
            primary_res[(seg, graph)] = chosen
            checks.add(f"{seg} / {graph}: primary resolution (lowest with >= 2 clusters of >= {T.MIN_CLUSTER} cells)",
                       f"{chosen}", chosen is not None)
            if chosen is None:
                continue
            cl = pc_seg[f"sub_{graph}_r{chosen:.1f}"]
            pc_seg[f"sub_{graph}"] = cl
            r = [x for x in scan_rows if x["segment"] == seg and x["graph"] == graph and x["resolution"] == chosen][0]
            checks.add(f"{seg} / {graph} r{chosen}: clusters; seed ARI median (min); eta2 ln(IDs); ARI kidney_annotation / Fig 2A",
                       f"{r['cluster_sizes']}; {r['seed_ari_median']:.2f} ({r['seed_ari_min']:.2f}); "
                       f"{r['eta2_ln_ids']:.3f}; {r['ari_t15']:.2f} / {r['ari_pre']:.2f}", None)
            # DE between subclusters
            cache = t5.rank_cache(ln1p.loc[cells].to_numpy(), det.loc[cells].to_numpy())
            de = t5.de_one_vs_rest(cache, cl.to_numpy(), ln1p.columns, f"{seg}_{graph}_r{chosen:.1f}")
            de = de[de["cluster"] != T.BELOW_MIN].copy()
            de.insert(0, "graph", graph)
            de.insert(0, "segment", seg)
            for lab_col in T.LABELS:
                s = T.LABEL_SHORT[lab_col]
                pm = panels[panels["label_column"] == lab_col].groupby("protein")["panel"].agg(";".join)
                de[f"in_panels_{s}"] = de["protein"].map(pm).fillna("")
            de["v3_marker_types"] = de["protein"].map(lambda p: ";".join(v3.get(p, [])))
            de_rows.append(de)
            for c, g in de.groupby("cluster"):
                sig = g[g["up"] & (g["q_bh"] < t5.Q_DE)]
                rec = {"segment": seg, "graph": graph, "resolution": chosen, "subcluster": c,
                       "n_cells": int(g["n_cells_cluster"].iloc[0]),
                       "median_protein_ids": float(ids.loc[cells][cl == c].median()),
                       "n_sig_up": int(len(sig)),
                       "top25": ", ".join(sig.sort_values("rank_up").head(t5.TOP_N)["protein"])}
                for lab_col in T.LABELS:
                    s = T.LABEL_SHORT[lab_col]
                    for p in T.CONTRASTS:
                        if T.CONTRASTS[p][0] != seg:
                            continue
                        pp = set(panels[(panels["label_column"] == lab_col) & (panels["panel"] == p)]["protein"])
                        if pp:
                            rec[f"{s}_{T.CONTRAST_SAFE[p]}_panel_sig_up"] = f"{len(pp & set(sig['protein']))} of {len(pp)}"
                de_sum_rows.append(rec)
            # crosstabs
            for lab_col in T.LABELS:
                ct = pd.crosstab(cl, labels.loc[cells, lab_col])
                for c in ct.index:
                    for l_ in ct.columns:
                        cross_rows.append({"segment": seg, "graph": graph, "rows": "subcluster", "row": c,
                                           "columns": lab_col, "column": l_, "n_cells": int(ct.loc[c, l_])})
            # scores by subcluster (kidney_annotation panels, mean_z, all genes)
            for p in (*T.PT_ONE_VS_REST, *T.PT_GROUPED, "S1/S2 - S3") if seg == "PT" else (*T.TAL_PANELS, "CTAL - MTAL", "MD"):
                key = f"score__t15__{p.replace('/', '').replace(' - ', '_minus_')}__all__mean_z"
                v = scores.loc[cells, key]
                if v.isna().all():
                    continue
                ok = cl != T.BELOW_MIN
                groups = [v[ok & (cl == k)] for k in sorted(cl[ok].unique())]
                for k in sorted(cl[ok].unique()):
                    score_rows.append({"segment": seg, "graph": graph, "score": p, "subcluster": k,
                                       "n_cells": int((cl == k).sum()), "median_score": float(v[cl == k].median()),
                                       "kruskal_p_across_subclusters": float(stats.kruskal(*groups).pvalue)
                                       if len(groups) > 1 else np.nan})
        pc_seg.insert(0, "segment", seg)
        per_cell.append(pc_seg)

    # mapping -----------------------------------------------------------------------------------------------------
    map_rows, map_conf, map_sum = [], [], []
    mapped = {}
    all_cs = cs[cs["rna_source"] == "all"]
    for (seg, version), classes in MAP_CLASSES.items():
        cells = sub[sub["segment"] == seg]["cell_id"].tolist()
        contrasts = [c for c in classes]          # one-vs-rest contrasts named after the classes
        for gene_rule in ("panels", "all_rna_passing"):
            if gene_rule == "panels":
                pr = panels[panels["panel"].isin(contrasts)]
                genes = list(dict.fromkeys(pr.sort_values(["panel", "rank"])["atlas_gene"]))
            else:
                dd = de_atlas[de_atlas["contrast"].isin(contrasts) & de_atlas["passes_rna_filters"]
                              & de_atlas["quantified_protein"]]
                sd = det.loc[cells, dd["protein"].unique()].mean(axis=0)
                keep = set(sd[sd >= T.SCP_DET_MIN].index)
                genes = list(dict.fromkeys(dd[dd["protein"].isin(keep)]["atlas_gene"]))
            prot = de_atlas.drop_duplicates("atlas_gene").set_index("atlas_gene").loc[genes, "protein"].tolist()
            seg_s = ss[ss["segment"] == MAP_SEGMENT_STATS[(seg, version)]].set_index("atlas_gene")
            mu, sd_ = seg_s.loc[genes, "mean_lognorm"].to_numpy(), seg_s.loc[genes, "sd_lognorm"].to_numpy()
            cent = pd.DataFrame([(all_cs[all_cs["class"] == c].set_index("atlas_gene").loc[genes, "mean_lognorm"]
                                  .to_numpy() - mu) / np.where(sd_ > 0, sd_, 1.0) for c in classes],
                                index=list(classes), columns=genes)
            for scp_matrix in ("subset_z", "canonical"):
                if scp_matrix == "subset_z":
                    x = cc.scale_like_scanpy(ln1p.loc[cells, prot].to_numpy())
                else:
                    x = xz.loc[cells, prot].to_numpy()
                sc_ = cosine(x, cent)
                pred = summarize(sc_, pd.Index(cells, name="cell_id"))
                variant = f"{version}|{gene_rule}|{scp_matrix}"
                primary = gene_rule == "panels" and scp_matrix == "subset_z"
                mapped[(seg, variant)] = pred
                rho = stats.spearmanr(pred["top_score"], np.log(ids.loc[cells]))
                map_sum.append({"segment": seg, "classes": ";".join(classes), "variant": variant,
                                "primary": primary and version == "primary", "n_informative_genes": len(genes),
                                "n_cells": len(cells), "frac_low_confidence": float(pred["low_confidence"].mean()),
                                "median_top_score": float(pred["top_score"].median()),
                                "median_margin": float(pred["margin"].median()),
                                "rho_top_score_ln_ids": float(rho.statistic), "p_rho": float(rho.pvalue),
                                **{f"frac_top_{c}": float((pred["top_class"] == c).mean()) for c in classes}})
                out = pred.copy()
                out.insert(0, "variant", variant)
                out.insert(0, "segment", seg)
                for c in classes:
                    out[f"score_{c}"] = sc_[c].to_numpy()
                map_rows.append(out.reset_index())
                for lab_col in T.LABELS:
                    for l_, g in labels.loc[cells, lab_col].groupby(labels.loc[cells, lab_col]):
                        p_ = pred.loc[g.index]
                        for c in classes:
                            map_conf.append({"segment": seg, "variant": variant, "label_column": lab_col, "label": l_,
                                             "n_label": len(g), "class": c,
                                             "n_top": int((p_["top_class"] == c).sum()),
                                             "n_top_confident": int(((p_["top_class"] == c) & ~p_["low_confidence"]).sum()),
                                             "frac_top": float((p_["top_class"] == c).mean())})
            if version == "primary":
                checks.add(f"{seg} mapping ({gene_rule}): informative genes; low confidence (subset z)",
                           f"{len(genes)}; {mapped[(seg, f'{version}|{gene_rule}|subset_z')]['low_confidence'].mean():.2f}",
                           None)
    map_conf = pd.DataFrame(map_conf)
    # S1/S2 vs S3 agreement summary
    agree_rows = []
    pv = "primary|panels|subset_z"
    pred = mapped[("PT", pv)]
    groupings = T.pt_groupings(labels)
    for g, lab in groupings.items():
        l_ = lab.reindex(pred.index)
        for k, cls_ in (("PCT", ("PTS1", "PTS2")), ("PST", ("PTS3", "PTS3T2"))):
            m = l_.eq(k)
            top = pred.loc[m, "top_class"]
            conf = ~pred.loc[m, "low_confidence"]
            agree_rows.append({"grouping": g, "label": k, "n_cells": int(m.sum()),
                               "expected_classes": ";".join(cls_),
                               "frac_top_expected": float(top.isin(cls_).mean()),
                               "frac_top_expected_confident": float((top.isin(cls_) & conf).mean()),
                               "frac_low_confidence": float((~conf).mean())})
    agree = pd.DataFrame(agree_rows)
    for r in agree.itertuples():
        checks.add(f"PT mapping (primary), {T.GROUPING_TEXT[r.grouping]}: {r.label} cells mapped to {r.expected_classes}"
                   " (all; confident)", f"{r.frac_top_expected:.2f}; {r.frac_top_expected_confident:.2f} of {r.n_cells}",
                   None)
    for seg in T.SEGMENTS:
        if (seg, "primary") in primary_res and primary_res[(seg, "primary")] is not None:
            pr = mapped[(seg, pv)]
            cl = [p for p in per_cell if p["segment"].iloc[0] == seg][0]["sub_primary"]
            ct = pd.crosstab(cl.reindex(pr.index), pr["top_class"])
            for c in ct.index:
                for k in ct.columns:
                    cross_rows.append({"segment": seg, "graph": "primary", "rows": "subcluster", "row": c,
                                       "columns": "mapped class (primary)", "column": k, "n_cells": int(ct.loc[c, k])})

    per_cell = pd.concat(per_cell)
    per_cell.index.name = "cell_id"
    fig_rows = []
    figures(run, adata, labels, sub, per_cell, mapped, map_conf, primary_res, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    checks.add("ignored igraph RNG messages ('high is out of bounds for int32'; counted, not printed; as qc_mito)",
               f"{T.IGNORED['igraph_int32_randint']:,}", None)
    run.note(f"Ignored igraph RNG messages ('high is out of bounds for int32'): {T.IGNORED['igraph_int32_randint']}.")
    run.table(per_cell.reset_index(), "sub_clusters_per_cell")
    run.table(pd.DataFrame(scan_rows), "sub_scan")
    run.table(pd.concat(de_rows, ignore_index=True), "sub_de")
    run.table(pd.DataFrame(de_sum_rows), "sub_de_summary")
    run.table(pd.DataFrame(cross_rows), "sub_crosstab")
    run.table(pd.DataFrame(score_rows), "sub_scores")
    run.table(pd.concat(map_rows, ignore_index=True), "map_per_cell")
    run.table(pd.DataFrame(map_sum), "map_summary")
    run.table(map_conf, "map_confusion")
    run.table(agree, "map_pt_agreement")
    run.table(figs, "subclusters_figure_checks")
    run.table(checks.frame(), "subclusters_checks")
    run.finish({"resolutions": list(T.SUB_RESOLUTIONS), "seeds": [T.SEED, *T.SUB_SEEDS], "min_cluster": T.MIN_CLUSTER,
                "primary_resolution": {f"{k[0]}_{k[1]}": v for k, v in primary_res.items()},
                "de": "cluster_de_lib.de_one_vs_rest on ln(1 + intensity) within the subset",
                "mapping": {"score_min": T.SCORE_MIN, "margin_min": T.MARGIN_MIN,
                            "centroids": "(class mean - segment mean) / segment SD, atlas log-normalized",
                            "scp": "ln(1 + intensity) z-scored across the subset (primary); canonical .X (sensitivity)"},
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see pt_tal_subclusters_checks.csv.")


def figures(run, adata, labels, sub, per_cell, mapped, map_conf, primary_res, fig_rows) -> None:
    plt = T.setup()
    LH = T.LH
    colors = T.label_colors(adata)
    cmap = T8.sequential_cmap()
    pv = "primary|panels|subset_z"
    # 1. UMAPs: subclusters (primary, regressed) and mapped class ------------------------------------------------
    title = "PT and TAL union subsets: Leiden subclusters and the nearest atlas subtype centroid"
    note = ("Left: subclusters on the primary subset graph (UMAP of step 0); middle: subclusters on the depth-regressed "
            "graph (its own UMAP); right: top atlas class by cosine similarity (primary mapping: step-1 panel genes, "
            "centroids standardized within the atlas segment, SCP ln(1 + intensity) z-scored across the subset); open "
            f"circles: low confidence (score < {T.SCORE_MIN}, margin < {T.MARGIN_MIN}). Resolution: lowest of "
            f"{', '.join(map(str, T.SUB_RESOLUTIONS))} with >= 2 clusters of >= {T.MIN_CLUSTER} cells.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    pw, ph, gap = 52.0, 38.0, 7.0
    rh = LH + 1.5 + LH + 0.8 + ph + 0.8 + 2 * (LH + 0.4) + 3.0
    H = T8.M + title_h + 2 * rh + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    for i, seg in enumerate(T.SEGMENTS):
        d = sub[sub["segment"] == seg].set_index("cell_id")
        pc = per_cell[per_cell["segment"] == seg].reindex(d.index)
        y = T8.M + title_h + i * rh
        sh.text(T8.M, y, f"{seg} union subset ({len(d)} cells)", fontweight="bold", va="top")
        y += LH + 1.5
        for k, graph in enumerate(T.GRAPHS):
            res = primary_res.get((seg, graph))
            if res is None:
                continue
            cl = pc[f"sub_{graph}"].astype(str)
            order = sorted(c for c in cl.unique() if c != T.BELOW_MIN) + ([T.BELOW_MIN] if (cl == T.BELOW_MIN).any() else [])
            cols = {c: T.SERIES[j % len(T.SERIES)] for j, c in enumerate(order)}
            cols[T.BELOW_MIN] = T.INK["muted"]
            xy = d[[f"umap1_{graph}", f"umap2_{graph}"]].to_numpy(float)
            T.umap_categorical(sh, T8.M + k * (pw + gap), y, pw, ph, xy, cl, order, cols,
                               f"Subclusters, {graph} graph (r {res})")
        pred = mapped[(seg, pv)].reindex(d.index)
        xy = d[["umap1_primary", "umap2_primary"]].to_numpy(float)
        x0 = T8.M + 2 * (pw + gap)
        sh.text(x0, y, "Nearest atlas class (primary UMAP)", va="top")
        ax = sh.ax(x0, y + LH + 0.8, pw, ph)
        classes = list(T.PT_CLASSES if seg == "PT" else T.TAL_CLASSES)
        for c in classes:
            m = (pred["top_class"] == c).to_numpy()
            conf = ~pred["low_confidence"].to_numpy()
            ax.scatter(xy[m & conf, 0], xy[m & conf, 1], s=2.2, color=T.RNA_PALETTE[c], linewidths=0, rasterized=True)
            ax.scatter(xy[m & ~conf, 0], xy[m & ~conf, 1], s=2.2, facecolors="none", edgecolors=T.RNA_PALETTE[c],
                       linewidths=0.35, rasterized=True)
        ax.set_axis_off()
        ent = [(f"{c} ({int((pred['top_class'] == c).sum())})", T.RNA_PALETTE[c]) for c in classes]
        T.draw_legend(sh, x0, y + LH + 0.8 + ph + 0.8, T.legend_rows(ent, pw))
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "subclusters_umap", title, fig_rows)

    # 2. crosstabs: subcluster x labels; label x mapped class --------------------------------------------------------
    title = "Subclusters (primary graph) and nearest atlas class against the kidney_annotation and Fig 2A labels"
    note = ("Numbers: cells; fill: fraction of the row. Subclusters: primary subset graph at the primary resolution. "
            "Mapped class: primary mapping (all cells, regardless of confidence). Rows of the mapping panels: labels "
            "within the union subset.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    cw, rh_ = 8.5, 4.4
    blocks = []
    for seg in T.SEGMENTS:
        d = sub[sub["segment"] == seg].set_index("cell_id")
        cl = per_cell[per_cell["segment"] == seg].reindex(d.index)["sub_primary"].astype(str)
        rows = sorted(c for c in cl.unique() if c != T.BELOW_MIN) + ([T.BELOW_MIN] if (cl == T.BELOW_MIN).any() else [])
        pred = mapped[(seg, pv)].reindex(d.index)
        classes = list(T.PT_CLASSES if seg == "PT" else T.TAL_CLASSES)
        panels_ = []
        for lab_col in T.LABELS:
            ct = pd.crosstab(cl, d[lab_col]).reindex(index=rows)
            ct = ct[T.order_labels(d[lab_col])]
            panels_.append((f"subcluster x {T.LABEL_DISPLAY[lab_col]}", ct))
        for lab_col in T.LABELS:
            ct = pd.crosstab(d[lab_col], pred["top_class"]).reindex(columns=classes, fill_value=0)
            ct = ct.reindex(index=T.order_labels(d[lab_col]))
            panels_.append((f"{T.LABEL_DISPLAY[lab_col]} x atlas class", ct))
        blocks.append((seg, panels_))
    # layout: panels left to right, wrapped to a new line when wider than the page
    placed, y = [], T8.M + title_h
    for seg, ps in blocks:
        placed.append(("header", seg, T8.M, y, None))
        y += LH + 1.5
        x, line_h = T8.M, 0.0
        for name, ct in ps:
            rl = [f"{r} ({int(ct.loc[r].sum())})" for r in ct.index]
            cl_ = [f"{c} ({int(ct[c].sum())})" for c in ct.columns]
            lw = max(T8.text_mm(r_)[0] for r_ in rl) + 1.5
            w = max(lw + len(cl_) * cw, T8.text_mm(name)[0])
            h = LH + 1.0 + len(rl) * rh_ + 0.8 + max(T8.text_mm(c)[0] for c in cl_) + 3.0
            if x > T8.M and x + w > 180.0 - T8.M:
                y += line_h
                x, line_h = T8.M, 0.0
            placed.append(("panel", name, x, y, (ct, rl, cl_, lw)))
            x += w + 6.0
            line_h = max(line_h, h)
        y += line_h
    H = y + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    for kind, name, x, y, payload in placed:
        if kind == "header":
            sh.text(x, y, f"{name} union subset", fontweight="bold", va="top")
            continue
        ct, rl, cl_, lw = payload
        sh.text(x, y, name, va="top")
        counts = ct.to_numpy()
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        T.count_heatmap(sh, x + lw, y + LH + 1.0, counts, frac, rl, cl_, cw, rh_, cmap)
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "subclusters_crosstab", title, fig_rows)

    # 3. mapping confusion with confidence -------------------------------------------------------------------------
    title = "Nearest atlas subtype centroid per SCP label (fraction of the label; primary mapping)"
    note = ("Fill and numbers: cells of the label whose top class is the column (in brackets: confident, score >= "
            f"{T.SCORE_MIN} and margin >= {T.MARGIN_MIN}). PT: atlas PTS1 / PTS2 / PTS3 / PTS3T2; TAL: CTAL / MTAL. "
            "Labels with >= 5 cells in the union subset.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    cw = 14.0
    rows_all = []
    for seg in T.SEGMENTS:
        for lab_col in T.LABELS:
            c = map_conf[(map_conf["segment"] == seg) & (map_conf["variant"] == pv) & (map_conf["label_column"] == lab_col)
                         & (map_conf["n_label"] >= 5)]
            for l_ in T.order_labels(c["label"]):
                rows_all.append((seg, lab_col, l_))
    rl = [f"{l_} ({T.GROUPING_SHORT[T.LABEL_SHORT[c_]]}, {s})" for s, c_, l_ in rows_all]
    lw = max(T8.text_mm(r)[0] for r in rl) + 2.0
    n_pt = sum(1 for r in rows_all if r[0] == "PT")
    H = T8.M + title_h + LH + 1.0 + max(n_pt, len(rows_all) - n_pt) * 4.6 + 4.0 + 6.0 + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    y0 = T8.M + title_h + LH + 1.0
    for seg, x_off in (("PT", 0.0), ("TAL", 4 * cw + 12.0)):
        classes = list(T.PT_CLASSES if seg == "PT" else T.TAL_CLASSES)
        idx = [i for i, r in enumerate(rows_all) if r[0] == seg]
        sub_rows = [rows_all[i] for i in idx]
        counts, confn = [], []
        for s, c_, l_ in sub_rows:
            c = map_conf[(map_conf["segment"] == s) & (map_conf["variant"] == pv) & (map_conf["label_column"] == c_)
                         & (map_conf["label"] == l_)].set_index("class")
            counts.append([int(c.loc[k, "n_top"]) for k in classes])
            confn.append([int(c.loc[k, "n_top_confident"]) for k in classes])
        counts, confn = np.array(counts), np.array(confn)
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        x0 = T8.M + lw + x_off
        yy = y0 + (0 if seg == "PT" else 0.0)
        ax = sh.ax(x0, yy, len(classes) * cw, len(sub_rows) * 4.6)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=(0, len(classes), len(sub_rows), 0))
        ax.set_axis_off()
        for a in range(len(sub_rows)):
            for b in range(len(classes)):
                sh.text(x0 + (b + 0.5) * cw, yy + (a + 0.5) * 4.6, f"{counts[a, b]} ({confn[a, b]})", ha="center",
                        va="center_baseline", color="white" if frac[a, b] > 0.55 else T.INK["primary"])
        if seg == "PT":
            for a, i in enumerate(idx):
                sh.text(x0 - 1.0, yy + (a + 0.5) * 4.6, rl[i], ha="right", va="center_baseline")
        else:
            for a, i in enumerate(idx):
                sh.text(x0 + len(classes) * cw + 1.0, yy + (a + 0.5) * 4.6, rl[i], ha="left", va="center_baseline")
        for b, k in enumerate(classes):
            sh.text(x0 + (b + 0.5) * cw, yy + len(sub_rows) * 4.6 + 0.8, k, ha="center", va="top")
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "map_confusion", title, fig_rows)


if __name__ == "__main__":
    main()
