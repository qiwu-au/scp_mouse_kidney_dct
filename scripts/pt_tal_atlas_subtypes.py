#!/usr/bin/env python
"""pt_tal step 1: PT and TAL subtypes in the RNA atlas, DE among them, candidate subtype panels.

Analysis plan, step 1.
The only pt_tal script that loads the full atlas (mka.h5ad, read-only; about 4.6 GB in memory).

Method (approved plan, 15 Oct 2026)
-----------------------------------
1. PT / TAL subtypes at all three atlas levels (labels starting with 'PT', PCT, PST, CTAL, MTAL, TAL, LOH, MD), with
   counts by Source (cells / nuclei) and Origin; Celltype x Celltype_finest crosstab; cells per Celltype x Origin x
   Source batch.
2. Reference level atlas ``Celltype`` (authors decision): PT = PTS1, PTS2, PTS3, PTS3T2; TAL = CTAL, MTAL; MD descriptive.
   LOH (mixed class, cells only) is listed but not used.
3. Genes: the atlas genes matching a quantified protein (case-insensitive; exact-name protein entries; kidney_rna: 3,569),
   plus the plan's literature candidates found among the atlas genes. Log-normalized expression = counts / total
   counts over all 16,711 genes x 10,000, log1p (as kidney_rna / atlas_dct).
4. DE (scanpy rank_genes_groups, Wilcoxon, tie-corrected; all atlas cells and nuclei):
   - PT one-vs-rest: each of PTS1, PTS2, PTS3, PTS3T2 against the other PT classes;
   - PT grouped: S1/S2 (PTS1 + PTS2) against S3 (PTS3 + PTS3T2), and S3 against S1/S2;
   - TAL: CTAL against MTAL and MTAL against CTAL;
   - MD (descriptive): MD against CTAL + MTAL.
5. RNA filters: BH q < 0.05; log2 fold change >= 1; detected (counts > 0) in >= 25% of the group's cells; batch
   consistency: the group-minus-reference difference of mean log-normalized expression has the overall sign in
   >= 2/3 of the Origin x Source batches with >= 20 cells on both sides (at least one such batch).
6. SCP filter, per label set: protein detected (intensity > 0) in >= 10% of the SCP cells of the segment (PT panels:
   PCT + PST cells; TAL and MD panels: TAL cells) under that label set.
7. Panel = genes passing 5 and 6, top 30 by Wilcoxon score. Coverage per panel and label set.

Inputs (read-only): mka.h5ad; results/h5ad/kidney_rna_atlas_pseudobulk.h5ad (cross-check); kidney_object canonical kidney object
                    and labels; kidney_annotation labels; KIDNEY_markers_V3.xlsx (marker flags only).
Outputs: results/tables/pt_tal/pt_tal_atlas_subtypes.csv, pt_tal_atlas_finest_crosstab.csv,
         pt_tal_atlas_batches.csv, pt_tal_atlas_class_stats.csv, pt_tal_atlas_segment_stats.csv,
         pt_tal_atlas_de.csv, pt_tal_atlas_batch_signs.csv, pt_tal_panels.csv,
         pt_tal_panel_coverage.csv, pt_tal_atlas_checks.csv;
         results/figures/pt_tal/pt_tal_atlas_composition, pt_tal_panel_funnel (.pdf/.png)

Command (project root):  python scripts/pt_tal_atlas_subtypes.py
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
import gc
import time

import numpy as np
import pandas as pd
from scipy import sparse

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as L7
import atlas_dct_lib as T8
import pt_tal_lib as T

PB_TOL = 1e-4
DESIGNS = {   # design -> (cells: classes, group function, tested groups, reference)
    "PT_one_vs_rest": (T.PT_CLASSES, lambda c: c, T.PT_ONE_VS_REST, "rest"),
    "PT_grouped": (T.PT_CLASSES, lambda c: "S1/S2" if c in ("PTS1", "PTS2") else "S3", T.PT_GROUPED, "rest"),
    "TAL": (T.TAL_CLASSES, lambda c: c, T.TAL_PANELS, "rest"),
    "MD": ((*T.TAL_CLASSES, T.MD_CLASS), lambda c: "MD" if c == "MD" else "CTAL+MTAL", ("MD",), "CTAL+MTAL"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pt_tal step 1: atlas PT / TAL subtypes, DE and candidate panels.")
    cc.add_common_args(parser)
    return parser.parse_args()


def family(label: str) -> str:
    if label.startswith("PT") or label in ("PCT", "PST"):
        return "PT"
    if label in ("CTAL", "MTAL", "TAL", "LOH", "MD") or "TAL" in label:
        return "TAL / LOH / MD"
    return ""


def subtype_listing(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in L7.ATLAS_LEVELS:
        lab = obs[level].astype(str)
        for label in L7.atlas_order([l for l in lab.unique() if family(l)]):
            sub = obs.loc[(lab == label).to_numpy()]
            rec = {"level": level, "label": label, "family": family(label), "n_cells": len(sub)}
            src = sub["Source"].astype(str).value_counts()
            rec.update(n_source_Cell=int(src.get("Cell", 0)), n_source_Nuclei=int(src.get("Nuclei", 0)))
            org = sub["Origin"].astype(str).value_counts()
            rec["n_origins"] = int(len(org))
            rec["origins"] = "; ".join(f"{k} {v}" for k, v in org.items())
            if level != T.LEVEL:
                rec["celltype_of_these_cells"] = "; ".join(f"{k} {v}" for k, v in
                                                           sub[T.LEVEL].astype(str).value_counts().items())
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    import anndata as ad
    import scanpy as sc

    args = parse_args()
    run = cc.Run(T.TASK, "atlas_subtypes", args)
    t0 = time.time()
    checks = t5.Checks("atlas")
    adata, labels, _, _, det = T.load_scp(run, checks)
    v3 = T.v3_marker_genes(run)
    pb = L7.read_pseudobulk(run)

    print("[atlas] reading mka.h5ad (about 4.6 GB in memory)", flush=True)
    a = ad.read_h5ad(run.use_input(T.ATLAS, "RNA atlas mka.h5ad (read-only)"))
    checks.add("atlas shape (cells x .X genes); .raw genes", f"{a.shape[0]} x {a.shape[1]}; {a.raw.n_vars}",
               a.shape == L7.ATLAS_EXPECTED_SHAPE and a.raw.n_vars == L7.ATLAS_EXPECTED_RAW_GENES)
    obs_all = a.obs
    listing = subtype_listing(obs_all)
    for r in listing[listing["level"] == T.LEVEL].itertuples():
        checks.add(f"atlas Celltype {r.label}: cells (cells / nuclei; origins)",
                   f"{r.n_cells} ({r.n_source_Cell} / {r.n_source_Nuclei}; {r.n_origins})", None)
    keep_ct = (*T.REGION_CLASSES, *T.SUMMARY_ONLY)
    ct_mask = obs_all[T.LEVEL].astype(str).isin(keep_ct).to_numpy()
    cross = pd.crosstab(obs_all.loc[ct_mask, T.LEVEL].astype(str), obs_all.loc[ct_mask, "Celltype_finest"].astype(str))
    cross = cross.reindex(index=list(keep_ct))
    cross = cross[L7.atlas_order(cross.columns)]
    cross.index.name = "Celltype"
    batches = (obs_all.loc[ct_mask].assign(Celltype=obs_all.loc[ct_mask, T.LEVEL].astype(str),
                                           Origin=obs_all.loc[ct_mask, "Origin"].astype(str),
                                           Source=obs_all.loc[ct_mask, "Source"].astype(str))
               .groupby(["Celltype", "Origin", "Source"]).size().rename("n_cells").reset_index())
    batches["Celltype"] = pd.Categorical(batches["Celltype"], list(keep_ct), ordered=True)
    batches = batches.sort_values(["Celltype", "Origin", "Source"]).reset_index(drop=True)

    # genes ------------------------------------------------------------------------------------------------
    atlas_genes = pd.Index(a.raw.var_names.astype(str))
    shared = L7.match_proteins(adata.var_names, atlas_genes)
    checks.add("atlas genes matching a quantified protein (as kidney_rna)", f"{len(shared)}",
               len(shared) == L7.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES, f"{L7.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES}")
    look = L7.atlas_gene_lookup(atlas_genes)
    lit_in_atlas = {g: look[g.lower()] for g, _, _ in T.LITERATURE if g.lower() in look}
    extra = [ag for ag in lit_in_atlas.values() if ag not in set(shared["atlas_gene"])]
    genes = pd.Index([*shared["atlas_gene"], *extra])
    checks.add("literature candidates among the atlas genes; added to the tested genes (not quantified as protein)",
               f"{len(lit_in_atlas)} of {len(T.LITERATURE)}: {', '.join(lit_in_atlas)}; added: "
               f"{', '.join(extra) or 'none'}", None)
    prot_of = dict(zip(shared["atlas_gene"], shared["protein"]))

    # region cells -----------------------------------------------------------------------------------------
    reg = obs_all[T.LEVEL].astype(str).isin(T.REGION_CLASSES).to_numpy()
    raw = a.raw.X
    if not sparse.isspmatrix_csr(raw):
        raw = sparse.csr_matrix(raw)
    counts_all = raw[np.flatnonzero(reg)]
    totals = np.asarray(counts_all.sum(axis=1), dtype=np.float64).ravel()
    counts = counts_all[:, atlas_genes.get_indexer(genes)].astype(np.float32)
    counts.sort_indices()
    obs = obs_all.loc[reg, [T.LEVEL, "Origin", "Source", "Celltype_finest"]].astype(str).copy()
    del a, raw, counts_all, obs_all
    gc.collect()
    d = counts.data
    checks.add("region counts are positive integers", f"{d.size} values; non-integers "
               f"{int(np.count_nonzero(d != np.round(d)))}", bool(np.all(d == np.round(d)) and d.min() > 0))
    ln = T.lognorm_with_totals(counts, totals)
    detected = counts.copy()
    detected.data = np.ones_like(detected.data)
    cls = obs[T.LEVEL].to_numpy()
    n_cls = {c: int((cls == c).sum()) for c in T.REGION_CLASSES}
    checks.add("region cells per class", "; ".join(f"{k} {v}" for k, v in n_cls.items()), None)

    # strata sums (class x Origin x Source) -----------------------------------------------------------------
    skey = pd.MultiIndex.from_frame(obs[[T.LEVEL, "Origin", "Source"]])
    sid, strata = pd.factorize(skey, sort=True)
    ind = sparse.csr_matrix((np.ones(len(sid)), (sid, np.arange(len(sid)))), shape=(len(strata), len(sid)))
    S = {"n": np.bincount(sid, minlength=len(strata)).astype(float),
         "sum": np.asarray((ind @ ln.astype(np.float64)).todense()),
         "sumsq": np.asarray((ind @ ln.multiply(ln).astype(np.float64)).todense()),
         "det": np.asarray((ind @ detected.astype(np.float64)).todense())}
    st = strata.to_frame(index=False)
    st.columns = ["cls", "Origin", "Source"]
    st["batch"] = st["Origin"] + " | " + st["Source"]

    def agg(mask: np.ndarray) -> dict[str, np.ndarray]:
        n = S["n"][mask].sum()
        if n == 0:
            nan = np.full(len(genes), np.nan)
            return {"n": 0, "mean": nan, "sd": nan, "frac": nan}
        s1, s2 = S["sum"][mask].sum(axis=0), S["sumsq"][mask].sum(axis=0)
        mean = s1 / n
        return {"n": int(n), "mean": mean, "sd": np.sqrt(np.maximum(s2 / n - mean ** 2, 0)),
                "frac": S["det"][mask].sum(axis=0) / n}

    class_rows, pb_diff = [], {"mean": 0.0, "frac": 0.0}
    for c in T.REGION_CLASSES:
        for src, source in T.RNA_SOURCES.items():
            m = (st["cls"] == c).to_numpy() & (np.ones(len(st), bool) if source is None
                                               else (st["Source"] == source).to_numpy())
            r = agg(m)
            class_rows.append(pd.DataFrame({"class": c, "rna_source": src, "atlas_gene": genes, "n_cells": r["n"],
                                            "mean_lognorm": r["mean"], "sd_lognorm": r["sd"],
                                            "frac_detected": r["frac"]}))
            if r["n"]:
                crit = {"Celltype": c} if source is None else {"Celltype": c, "Source": source}
                ref = L7.aggregate(pb, L7.group_mask(pb, **crit), list(genes))
                pb_diff["mean"] = max(pb_diff["mean"], float(np.max(np.abs(ref["mean_lognorm"].to_numpy() - r["mean"]))))
                pb_diff["frac"] = max(pb_diff["frac"], float(np.max(np.abs(ref["frac_detected"].to_numpy() - r["frac"]))))
    class_stats = pd.concat(class_rows, ignore_index=True)
    checks.add("class means and detection fractions equal the kidney_rna pseudobulk (max abs difference: mean; fraction)",
               f"{pb_diff['mean']:.1e}; {pb_diff['frac']:.1e}", max(pb_diff.values()) < PB_TOL, f"< {PB_TOL:g}")
    seg_rows = []
    for seg, classes in (("PT", T.PT_CLASSES), ("TAL", T.TAL_CLASSES), ("TAL+MD", (*T.TAL_CLASSES, T.MD_CLASS))):
        r = agg(st["cls"].isin(classes).to_numpy())
        seg_rows.append(pd.DataFrame({"segment": seg, "classes": ";".join(classes), "atlas_gene": genes,
                                      "n_cells": r["n"], "mean_lognorm": r["mean"], "sd_lognorm": r["sd"]}))
    seg_stats = pd.concat(seg_rows, ignore_index=True)

    # DE ---------------------------------------------------------------------------------------------------------
    de_rows = []
    for design, (classes, fn, tested, reference) in DESIGNS.items():
        m = np.isin(cls, classes)
        grp = np.array([fn(c) for c in cls[m]])
        print(f"[DE] {design}: {int(m.sum())} cells x {len(genes)} genes", flush=True)
        ad_ = ad.AnnData(X=ln[np.flatnonzero(m)], obs=pd.DataFrame({"g": pd.Categorical(grp)},
                                                                    index=np.arange(int(m.sum())).astype(str)),
                         var=pd.DataFrame(index=list(genes)))
        sc.tl.rank_genes_groups(ad_, "g", groups=list(tested), reference=reference, method="wilcoxon",
                                use_raw=False, tie_correct=True)
        for g in tested:
            dd = sc.get.rank_genes_groups_df(ad_, group=g)[["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]]
            dd = dd.rename(columns={"names": "atlas_gene", "scores": "wilcoxon_score", "logfoldchanges": "log2fc",
                                    "pvals": "p_value", "pvals_adj": "q_bh"}).set_index("atlas_gene").reindex(genes)
            seg, gcls, rcls, kind = T.CONTRASTS[g]
            rg, rr = agg(st["cls"].isin(gcls).to_numpy()), agg(st["cls"].isin(rcls).to_numpy())
            diff = rg["mean"] - rr["mean"]
            # batch consistency
            elig, same = np.zeros(len(genes)), np.zeros(len(genes))
            for b, sb in st.groupby("batch"):
                mg = st.index.isin(sb.index) & st["cls"].isin(gcls).to_numpy()
                mr = st.index.isin(sb.index) & st["cls"].isin(rcls).to_numpy()
                if S["n"][mg].sum() >= T.BATCH_MIN_CELLS and S["n"][mr].sum() >= T.BATCH_MIN_CELLS:
                    db = agg(mg)["mean"] - agg(mr)["mean"]
                    elig += 1
                    same += np.sign(db) == np.sign(diff)
            out = pd.DataFrame({
                "contrast": g, "segment": seg, "kind": kind, "group_classes": ";".join(gcls),
                "reference_classes": ";".join(rcls), "n_group": rg["n"], "n_reference": rr["n"],
                "atlas_gene": genes, "protein": [prot_of.get(x, "") for x in genes],
                "quantified_protein": [x in prot_of for x in genes],
                "wilcoxon_score": dd["wilcoxon_score"].to_numpy(), "log2fc": dd["log2fc"].to_numpy(),
                "p_value": dd["p_value"].to_numpy(), "q_bh": dd["q_bh"].to_numpy(),
                "mean_lognorm_group": rg["mean"], "mean_lognorm_reference": rr["mean"], "mean_difference": diff,
                "frac_group": rg["frac"], "frac_reference": rr["frac"],
                "n_batches_eligible": elig.astype(int), "n_batches_same_sign": same.astype(int)})
            out["frac_batches_same_sign"] = np.where(elig > 0, same / np.maximum(elig, 1), np.nan)
            out["batch_consistent"] = (elig >= 1) & (out["frac_batches_same_sign"] >= T.BATCH_AGREE - 1e-12)
            out["passes_rna_filters"] = ((out["q_bh"] < T.Q_MAX) & (out["log2fc"] >= T.LFC_MIN)
                                         & (out["frac_group"] >= T.PCT_MIN) & out["batch_consistent"])
            de_rows.append(out)
        del ad_
        gc.collect()
    de = pd.concat(de_rows, ignore_index=True)

    # SCP filter and panels ---------------------------------------------------------------------------------
    md_cells = labels["kidney_pre_label"].eq(T.MD_LABEL)
    for col in T.LABELS:
        s = T.LABEL_SHORT[col]
        for seg in T.SEGMENTS:
            cells = T.segment_cells(labels, col, seg)
            frac = det.loc[cells].mean(axis=0)
            m = de["segment"] == seg
            de.loc[m, f"scp_det_{s}"] = de.loc[m, "protein"].map(frac).to_numpy()
    de["scp_det_md_fig2a"] = de["protein"].map(det.loc[md_cells].mean(axis=0))
    de["v3_marker_types"] = de["protein"].map(lambda p: ";".join(v3.get(p, [])) if p else "")
    fl = T.protein_flags(run, adata.var_names.astype(str)).set_index("protein")
    de["ribosomal"] = de["protein"].map(fl["ribosomal"]).fillna(False).astype(bool)
    de["mitocarta"] = de["protein"].map(fl["mitocarta"]).fillna(False).astype(bool)
    panel_rows, cov_rows = [], []
    for col in T.LABELS:
        s = T.LABEL_SHORT[col]
        de[f"passes_{s}"] = de["passes_rna_filters"] & de["quantified_protein"] & (de[f"scp_det_{s}"] >= T.SCP_DET_MIN)
        de[f"panel_rank_{s}"] = np.nan
        for c in T.CONTRASTS:
            m = (de["contrast"] == c) & de[f"passes_{s}"]
            top = de[m].sort_values("wilcoxon_score", ascending=False).head(T.PANEL_TOP)
            de.loc[top.index, f"panel_rank_{s}"] = np.arange(1, len(top) + 1)
            p = top.assign(label_column=col, panel=c, rank=np.arange(1, len(top) + 1),
                           scp_detection=top[f"scp_det_{s}"])
            panel_rows.append(p[["label_column", "panel", "segment", "kind", "rank", "atlas_gene", "protein",
                                 "wilcoxon_score", "log2fc", "q_bh", "frac_group", "frac_reference",
                                 "frac_batches_same_sign", "n_batches_eligible", "scp_detection", "scp_det_md_fig2a",
                                 "v3_marker_types", "ribosomal", "mitocarta"]])
            dc = de[de["contrast"] == c]
            rna = dc["passes_rna_filters"]
            cov_rows.append({
                "label_column": col, "panel": c, "segment": T.CONTRASTS[c][0], "kind": T.CONTRASTS[c][3],
                "n_genes_tested": int(len(dc)), "n_q_lt_0.05_up": int(((dc["q_bh"] < T.Q_MAX) & (dc["log2fc"] > 0)).sum()),
                "n_pass_q_lfc_detection": int(((dc["q_bh"] < T.Q_MAX) & (dc["log2fc"] >= T.LFC_MIN)
                                               & (dc["frac_group"] >= T.PCT_MIN)).sum()),
                "n_pass_rna_filters_incl_batch": int(rna.sum()),
                "n_rna_pass_quantified_protein": int((rna & dc["quantified_protein"]).sum()),
                "n_rna_pass_scp_detected_10pct": int(dc[f"passes_{s}"].sum()),
                "n_in_panel": int(len(top)),
                "median_scp_detection_panel": float(top[f"scp_det_{s}"].median()) if len(top) else np.nan,
                "n_v3_markers_in_panel": int((top["v3_marker_types"] != "").sum()),
                "n_ribosomal_in_panel": int(top["ribosomal"].sum()),
                "n_mitocarta_in_panel": int(top["mitocarta"].sum()),
                "panel_genes": ", ".join(top["protein"])})
    panels = pd.concat(panel_rows, ignore_index=True)
    cov = pd.DataFrame(cov_rows)
    for r in cov[cov["label_column"] == T.PRIMARY].itertuples():
        checks.add(f"panel {r.panel} (kidney_annotation labels): RNA filters incl. batch; + SCP >= 10%; panel size",
                   f"{r.n_pass_rna_filters_incl_batch}; {r.n_rna_pass_scp_detected_10pct}; {r.n_in_panel}", None)
    empty = cov[(cov["n_in_panel"] == 0) & (cov["kind"] != "descriptive")]
    checks.add("non-descriptive panels left empty (label set: panel)",
               "; ".join(f"{T.LABEL_SHORT[r.label_column]}: {r.panel}" for r in empty.itertuples()) or "none", None)

    # batch signs for panel and literature genes --------------------------------------------------------------
    keep_genes = set(panels["atlas_gene"]) | set(lit_in_atlas.values())
    bs_rows = []
    for c, (seg, gcls, rcls, _) in T.CONTRASTS.items():
        gi = [i for i, gname in enumerate(genes) if gname in keep_genes]
        for b, sb in st.groupby("batch"):
            mg = st.index.isin(sb.index) & st["cls"].isin(gcls).to_numpy()
            mr = st.index.isin(sb.index) & st["cls"].isin(rcls).to_numpy()
            ng, nr = S["n"][mg].sum(), S["n"][mr].sum()
            if ng == 0 or nr == 0:
                continue
            db = agg(mg)["mean"] - agg(mr)["mean"]
            for i in gi:
                bs_rows.append({"contrast": c, "batch": b, "n_group": int(ng), "n_reference": int(nr),
                                "eligible": bool(ng >= T.BATCH_MIN_CELLS and nr >= T.BATCH_MIN_CELLS),
                                "atlas_gene": genes[i], "mean_difference": float(db[i])})
    batch_signs = pd.DataFrame(bs_rows)

    fig_rows = []
    figures(run, batches, cov, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(listing, "atlas_subtypes")
    run.table(cross.reset_index(), "atlas_finest_crosstab")
    run.table(batches, "atlas_batches")
    run.table(class_stats, "atlas_class_stats")
    run.table(seg_stats, "atlas_segment_stats")
    run.table(de, "atlas_de")
    run.table(batch_signs, "atlas_batch_signs")
    run.table(panels, "panels")
    run.table(cov, "panel_coverage")
    run.table(figs, "atlas_figure_checks")
    run.table(checks.frame(), "atlas_checks")
    run.finish({"level": T.LEVEL, "region_classes": list(T.REGION_CLASSES), "contrasts": {k: list(map(str, v)) for k, v
                                                                                           in T.CONTRASTS.items()},
                "q_max": T.Q_MAX, "log2fc_min": T.LFC_MIN, "frac_group_min": T.PCT_MIN,
                "batch_min_cells": T.BATCH_MIN_CELLS, "batch_agree": T.BATCH_AGREE, "scp_detection_min": T.SCP_DET_MIN,
                "panel_top": T.PANEL_TOP, "n_genes_tested": len(genes), "runtime_s": round(time.time() - t0, 1),
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see pt_tal_atlas_checks.csv.")


def figures(run, batches: pd.DataFrame, cov: pd.DataFrame, fig_rows) -> None:
    plt = T.setup()
    LH = T.LH
    # 1. atlas composition: cells per class by Source and by Origin --------------------------------------------
    classes = [c for c in (*T.REGION_CLASSES, *T.SUMMARY_ONLY)]
    origins = sorted(batches["Origin"].unique())
    title = "Atlas Celltype classes used in pt_tal: cells by study (Origin) and by cells / nuclei (Source)"
    note = ("Mouse Kidney Atlas (mka.h5ad; Novella-Rausell et al. 2023), level Celltype. Bars: fraction of the class's "
            "cells; numbers: cells. LOH (mixed loop-of-Henle class) is listed but not used. MD: nuclei only.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    rh = 4.4
    lw = max(T8.text_mm(f"{c} ({int(batches.loc[batches['Celltype'] == c, 'n_cells'].sum()):,})")[0] for c in classes) + 2
    ow = 100.0
    sw = 180.0 - 2 * T8.M - lw - ow - 10.0
    org_cols = dict(zip(origins, [*T.SERIES, "#898781", "#52514e"][:len(origins)]))
    src_cols = {"Cell": "#3987e5", "Nuclei": "#0d366b"}
    leg_o = T.legend_rows([(o, org_cols[o]) for o in origins], ow)
    leg_s = T.legend_rows([("cells (scRNA-seq)", src_cols["Cell"]), ("nuclei (snRNA-seq)", src_cols["Nuclei"])], sw)
    leg_h = max(len(leg_o), len(leg_s)) * (LH + 0.4)
    H = T8.M + title_h + LH + 1.0 + len(classes) * rh + 2.0 + leg_h + 2.0 + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    y0 = T8.M + title_h + LH + 1.0
    xo = T8.M + lw
    xs = xo + ow + 10.0
    sh.text(xo, T8.M + title_h, "Study (Origin)", va="top")
    sh.text(xs, T8.M + title_h, "Source", va="top")
    for i, c in enumerate(classes):
        b = batches[batches["Celltype"] == c]
        n = int(b["n_cells"].sum())
        yc = y0 + (i + 0.5) * rh
        sh.text(xo - 1.0, yc, f"{c} ({n:,})", ha="right", va="center_baseline")
        x = xo
        for o in origins:
            w = b.loc[b["Origin"] == o, "n_cells"].sum() / n * ow
            if w > 0:
                sh.rect(x, yc - rh * 0.35, w, rh * 0.7, facecolor=org_cols[o], edgecolor="white", linewidth=0.3)
            x += w
        x = xs
        for s_ in ("Cell", "Nuclei"):
            w = b.loc[b["Source"] == s_, "n_cells"].sum() / n * sw
            if w > 0:
                sh.rect(x, yc - rh * 0.35, w, rh * 0.7, facecolor=src_cols[s_], edgecolor="white", linewidth=0.3)
            x += w
    ly = y0 + len(classes) * rh + 2.0
    T.draw_legend(sh, xo, ly, leg_o)
    T.draw_legend(sh, xs, ly, leg_s)
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "atlas_composition", title, fig_rows)

    # 2. panel funnel -----------------------------------------------------------------------------------------
    stages = [("n_pass_q_lfc_detection", "q, log2 FC, detection"), ("n_pass_rna_filters_incl_batch", "+ batch sign"),
              ("n_rna_pass_scp_detected_10pct", "+ SCP detected >= 10%"), ("n_in_panel", "panel (top 30)")]
    title = "Candidate subtype panels: genes left after each filter, per RNA contrast and label set"
    note = (f"RNA (atlas Celltype; Wilcoxon within the segment): q < {T.Q_MAX}, log2 FC >= {T.LFC_MIN:g}, detected in >= "
            f"{T.PCT_MIN:.0%} of the group; batch sign = same sign in >= 2/3 of the study x cells/nuclei batches with >= "
            f"{T.BATCH_MIN_CELLS} cells on both sides. SCP: protein detected in >= {T.SCP_DET_MIN:.0%} of the segment's "
            "SCP cells under each label set (PT panels: PCT + PST cells; TAL and MD panels: TAL cells). Bars: kidney_annotation labels "
            "(dark) and Fig 2A labels (light); log scale. MD is descriptive.")
    note_text, note_h = T8.note_block(note)
    title_h = T8.text_mm(title, T8.TITLE, "bold")[1] + 1.5
    contrasts = list(T.CONTRASTS)
    pw, ph = 36.0, 50.0
    H = T8.M + title_h + LH + 1.0 + ph + 2.0 + 16.0 + note_h + T8.M
    sh = T8.Sheet(H)
    sh.text(T8.M, T8.M, title, fontsize=T8.TITLE, fontweight="bold", va="top")
    shades = {"kidney_t15_label": "#184f95", "kidney_pre_label": "#9ec5f4"}
    for k, (key, lab) in enumerate(stages):
        x = T8.M + 9.0 + k * (pw + 5.0)
        sh.text(x, T8.M + title_h, lab, va="top")
        ax = sh.ax(x, T8.M + title_h + LH + 1.0, pw, ph)
        for j, col in enumerate(T.LABELS):
            v = [max(int(cov[(cov["label_column"] == col) & (cov["panel"] == c)][key].iloc[0]), 0) for c in contrasts]
            ax.bar(np.arange(len(contrasts)) + (j - 0.5) * 0.4, np.maximum(v, 0.8), width=0.4, color=shades[col],
                   linewidth=0)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 3000)
        ax.set_yticks([1, 10, 100, 1000])
        ax.set_yticklabels(["1", "10", "100", "1000"] if k == 0 else [])
        ax.set_xticks(np.arange(len(contrasts)))
        ax.set_xticklabels(contrasts, rotation=90)
        ax.set_xlim(-0.6, len(contrasts) - 0.4)
        T8.style_axes(ax)
    sh.note(note_text, note_h)
    T8.save_checked(run, sh, "panel_funnel", title, fig_rows)


if __name__ == "__main__":
    main()
