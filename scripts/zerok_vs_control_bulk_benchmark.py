#!/usr/bin/env python
"""zerok_vs_control step 6, addendum 5c: bulk low-K proteome benchmark on the combined search (zerok_response_core step 5c rerun).

Analysis plan,
step 6 (addendum 5c). Same procedure and settings as zerok_response_core_bulk_benchmark.py (its helpers read_bulk,
match_proteins, dct_enriched, spearman_ci and sign_stats imported unchanged; settings B1-B6 frozen in the zerok_vs_control plan):

B1  Bulk reference already on disk (downloaded with the authors' approval on 28 Oct 2026 for zerok_response_core; SHA256 checked):
    Murali SK et al. JCI Insight 2026;11(5):e196339, Supplemental data set 1, sheet 'Quantifiable pros &
    subclusters' (whole kidney, 0% vs 1% K+ for 2 weeks; 5 vs 7 mice; PXD035354).
B2  SCP proteins (joint set of the combined search, 5,897 proteins) matched to bulk rows by UniProt accession (any
    accession of the combined protein group), else by unique gene symbol; match type recorded.
B3  SCP effect: pseudobulk log2 FC pooled (step 4 / step 6 tables) as primary; depth-adjusted intensity log2 FC and
    detection log OR as secondary. Groups all_DCT and DCT1 under dct_label and dct_label_alternative, label-free all
    cells; all three variants.
B4  Spearman correlation with the bulk log2 FC (Fisher-z 95% CI) over all shared proteins and over bulk q < 0.05;
    sign agreement, raw and median-centred; 2 x 2 table of proteins significant in both (Fisher exact); panel table.
B5  Null: the same correlation for SCP plate contrasts (P2 vs P1 within Ctrl and within ZeroK; technical) beside
    ZeroK vs Ctrl within P1, within P2 and pooled (pseudobulk effects, zerok_response_core_lib.effect_sizes).
B6  DCT-enriched subset from the kidney_rna RNA atlas pseudobulk (Celltype level, Cell and Nuclei pooled).
All p-values are descriptive (SCP: one pool per condition; the bulk study has individual mice, but the two studies
differ in diet duration, control diet, tissue, platform and search).

Inputs (read-only): data/reference/jci.insight.196339.sdd1.xlsx; results/h5ad/dct_labels_DCT_joint_labelled.h5ad;
dct_labels canonical labels; zerok_core panel and hash; zerok_vs_control step 4 proteome tables, step 6 label-free proteome table;
results/h5ad/kidney_rna_atlas_pseudobulk.h5ad.
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_bulk_reference.csv, zerok_vs_control_bulk_matched.csv,
zerok_vs_control_bulk_concordance.csv, zerok_vs_control_bulk_sign_tables.csv, zerok_vs_control_bulk_plate_null.csv,
zerok_vs_control_bulk_panel.csv, zerok_vs_control_bulk_dct_enriched_genes.csv, zerok_vs_control_bulk_checks.csv.

Command (project root):  python scripts/zerok_vs_control_bulk_benchmark.py
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
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc
import cluster_de_lib as t5
import zerok_response_core_bulk_benchmark as t22b
import zerok_vs_control_lib as lib

SCRIPT = "zerok_vs_control_bulk_benchmark.py"
t22 = lib.t22
SCP_EFFECTS = t22b.SCP_EFFECTS
GROUPS = [("dct_label", lib.ALL_DCT), ("dct_label", "DCT1"), ("dct_label_alternative", lib.ALL_DCT),
          ("dct_label_alternative", "DCT1"), ("label_free", lib.ALL_CELLS)]
CONTRASTS = t22b.CONTRASTS
BULK_NOTE = ("bulk: whole kidney, 0% vs 1% K+ for 2 weeks, 5 vs 7 mice; SCP: sorted DCT cells, zero-K vs control "
             "diet for 4 days, one pool of 3 mice per condition, one combined search; different platforms and "
             "searches between the two studies; descriptive")


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 6c: bulk low-K proteome benchmark.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "bulk", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)
    bulk = t22b.read_bulk(run, checks)
    joint = lib.load_joint(run, checks)
    panel = t22.read_panel(run)
    matched = t22b.match_proteins(joint, bulk, checks).rename(columns={"match_level_t17": "match_level"})
    enr = t22b.dct_enriched(run, checks)
    enriched = set(enr.loc[enr["dct_enriched"], "gene"])
    matched["dct_enriched"] = matched["gene_name"].isin(enriched)
    checks.add("matched proteins that are DCT-enriched (B6)", f"{int(matched['dct_enriched'].sum())} of {len(matched)}",
               None)
    run.table(bulk, "bulk_reference")
    run.table(enr.sort_values("log2_ratio_dct_vs_other", ascending=False), "bulk_dct_enriched_genes")

    # SCP tables
    tabs = {}
    for col in lib.LABEL_COLUMNS:
        for v in lib.VARIANTS:
            tabs[(col, v)] = lib.read_table(run, f"proteome_{lib.LABEL_SHORT[col]}_{v}")
    lf = lib.read_table(run, "robustness_label_free_proteome")
    for v in lib.VARIANTS:
        tabs[("label_free", v)] = lf.query("variant == @v")
    bts = {v: t22.back_transform(joint, v) for v in lib.VARIANTS}
    cond = joint.obs["condition"].to_numpy()
    plate = joint.obs["plate"].to_numpy()
    mcols = ["protein", "match_type", "bulk_accession", "bulk_gene", "bulk_log2fc", "bulk_q", "bulk_direction",
             "dct_enriched"]

    conc_rows, sign_rows, null_rows, long_rows = [], [], [], []
    max_dev = 0.0
    for col, group in GROUPS:
        gmask = (np.ones(len(cond), bool) if col == "label_free" else t22.group_masks(joint.obs, col)[group])
        for v in lib.VARIANTS:
            t = tabs[(col, v)].query("group == @group")
            d = t.merge(matched[mcols], on="protein", how="inner")
            d["bulk_sig"] = d["bulk_q"] < lib.ALPHA
            long_rows.append(d.assign(label_column=col, group=group, variant=v)[
                ["label_column", "group", "variant", "protein", "gene_name", "match_type", "dct_enriched",
                 "bulk_log2fc", "bulk_q", "bulk_direction", *SCP_EFFECTS.values(), "observed_direction",
                 "q_two_part"]])
            sets = {"all shared": np.ones(len(d), bool), "bulk q < 0.05": d["bulk_sig"].to_numpy(),
                    "DCT-enriched": d["dct_enriched"].to_numpy(),
                    "DCT-enriched, bulk q < 0.05": (d["dct_enriched"] & d["bulk_sig"]).to_numpy()}
            for ename, ecol in SCP_EFFECTS.items():
                s_all = d[ecol].to_numpy(float)
                b_all = d["bulk_log2fc"].to_numpy(float)
                s_med, b_med = np.nanmedian(s_all), np.nanmedian(b_all)
                for sname, smask in sets.items():
                    row = {"label_column": col, "group": group, "variant": v, "scp_effect": ename,
                           "protein_set": sname, **t22b.spearman_ci(s_all[smask], b_all[smask])}
                    row.update(t22b.sign_stats(s_all[smask], b_all[smask]))
                    cs = t22b.sign_stats(s_all[smask] - s_med, b_all[smask] - b_med)
                    row.update({"sign_agreement_centred": cs["sign_agreement"],
                                "sign_centred_ci_low": cs["sign_ci_low"], "sign_centred_ci_high": cs["sign_ci_high"],
                                "scp_median_shift": s_med, "bulk_median_shift": b_med})
                    conc_rows.append(row)
            # 2 x 2 of proteins significant in both
            for sname in ("all shared", "DCT-enriched"):
                sel = d.loc[d["bulk_sig"] & d["observed_direction"].isin([t22.UP, t22.DOWN])
                            & (sets[sname] if sname == "all shared" else d["dct_enriched"])]
                tab = pd.crosstab(sel["observed_direction"], sel["bulk_direction"]).reindex(
                    index=[t22.UP, t22.DOWN], columns=[t22.UP, t22.DOWN], fill_value=0)
                a_, b_, c_, d_ = (int(tab.at[t22.UP, t22.UP]), int(tab.at[t22.UP, t22.DOWN]),
                                  int(tab.at[t22.DOWN, t22.UP]), int(tab.at[t22.DOWN, t22.DOWN]))
                orr, p = stats.fisher_exact([[a_, b_], [c_, d_]]) if (a_ + b_) and (c_ + d_) else (np.nan, np.nan)
                n_ = a_ + b_ + c_ + d_
                lo, hi = t22.wilson(a_ + d_, n_) if n_ else (np.nan, np.nan)
                sign_rows.append({"label_column": col, "group": group, "variant": v, "protein_set": sname,
                                  "scp_up_bulk_up": a_, "scp_up_bulk_down": b_, "scp_down_bulk_up": c_,
                                  "scp_down_bulk_down": d_, "n_both_significant": n_,
                                  "agreement": (a_ + d_) / n_ if n_ else np.nan, "agreement_ci_low": lo,
                                  "agreement_ci_high": hi, "odds_ratio": orr, "fisher_p": p})
            # B5 plate null (pseudobulk effects)
            cols_idx = np.array([joint.var_index(p) for p in d["protein"]])
            bvals = d["bulk_log2fc"].to_numpy(float)
            for cname, ((c0, p0), (c1, p1)) in CONTRASTS.items():
                ref = gmask & (cond == c0) & (plate == p0 if p0 else True)
                cmp_ = gmask & (cond == c1) & (plate == p1 if p1 else True)
                if min(ref.sum(), cmp_.sum()) < t22.MIN_GROUP_PROTEOME:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    e = t22.effect_sizes(joint, v, np.flatnonzero(ref), np.flatnonzero(cmp_), cols_idx, bts[v])
                if cname == "ZeroK vs Ctrl (pooled)":
                    max_dev = max(max_dev, float(np.nanmax(np.abs(e["pseudobulk_log2fc"]
                                                                  - d["pseudobulk_log2fc_pooled"].to_numpy(float)))))
                for sname in ("all shared", "bulk q < 0.05", "DCT-enriched"):
                    sm = sets[sname]
                    null_rows.append({"label_column": col, "group": group, "variant": v, "contrast": cname,
                                      "technical": "technical" in cname, "protein_set": sname,
                                      "n_reference": int(ref.sum()), "n_comparison": int(cmp_.sum()),
                                      **t22b.spearman_ci(e["pseudobulk_log2fc"][sm], bvals[sm]),
                                      **t22b.sign_stats(e["pseudobulk_log2fc"][sm], bvals[sm])})
    checks.add("pooled pseudobulk effect recomputed equals the step 4 / step 6 tables", f"max |diff| {max_dev:.2g}",
               max_dev < 1e-9)
    conc = pd.DataFrame(conc_rows)
    conc["note"] = BULK_NOTE
    signs = pd.DataFrame(sign_rows)
    signs["note"] = BULK_NOTE
    nulls = pd.DataFrame(null_rows)
    nulls["note"] = BULK_NOTE + "; plate contrasts are technical (sort plates of one pool)"
    long = pd.concat(long_rows, ignore_index=True)

    # panel table (primary labels, all_DCT, variant none)
    pan = panel.loc[panel["item_type"] == "protein", ["item", "expected_direction"]].rename(columns={"item": "protein"})
    t = tabs[("dct_label", lib.BASELINE)].query("group == 'all_DCT'")[
        ["protein", *SCP_EFFECTS.values(), "observed_direction", "q_two_part"]]
    pan = (pan.merge(matched[["protein", "match_type", "bulk_log2fc", "bulk_q", "bulk_direction", "dct_enriched"]],
                     on="protein", how="left").merge(t, on="protein", how="left"))
    pan.insert(0, "scp_scope", "dct_label, all_DCT, variant none")
    pan["note"] = BULK_NOTE
    checks.add("panel proteins found in the bulk table", f"{int(pan['bulk_log2fc'].notna().sum())} of {len(pan)}",
               None)

    run.table(long, "bulk_matched")
    run.table(conc, "bulk_concordance")
    run.table(signs, "bulk_sign_tables")
    run.table(nulls, "bulk_plate_null")
    run.table(pan, "bulk_panel")
    run.table(checks.frame(), "bulk_checks")
    run.finish(parameters={"bulk_source": t22b.BULK_SOURCE, "bulk_sheet": t22b.BULK_SHEET,
                           "bulk_sha256": t22b.BULK_SHA256,
                           "matching": "UniProt accession of the combined-search protein group, else unique gene "
                                       "symbol",
                           "scp_effects": SCP_EFFECTS, "groups": GROUPS, "contrasts": list(CONTRASTS),
                           "dct_enriched_rule": f"log2((CPM_DCT+1)/(CPM_other+1)) >= {t22b.DCT_LOG2_RATIO} and "
                                                f"CPM_DCT >= {t22b.DCT_MIN_CPM}; atlas Celltype level, Cell + Nuclei "
                                                f"pooled",
                           "downloaded": "28 Oct 2026 for zerok_response_core, author-approved; reused from data/reference/",
                           "settings": "frozen in the zerok_vs_control plan (zerok_response_core step 5c, B1-B6)",
                           "ported_from": "zerok_response_core_bulk_benchmark.py", "note": BULK_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_bulk_checks.csv.")


if __name__ == "__main__":
    main()
