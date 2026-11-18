#!/usr/bin/env python
"""zerok_response_core step 5c: benchmark of the zero-K vs control SCP effects against a bulk low-K kidney proteome.

Added at the authors' request (28 Oct 2026; plan defaults
B1-B6 approved the same day) after steps 1-6 and 5b had run.

Bulk reference (downloaded with the authors' approval, 28 Oct 2026; copied to data/reference/):
  Murali SK et al. Dietary potassium restriction causes hypercalciuria, hypocalcemia, and bone loss in male mice.
  JCI Insight 2026;11(5):e196339 (this project's own group). Supplemental data set 1 (jci.insight.196339.sdd1.xlsx),
  sheet 'Quantifiable pros & subclusters': whole kidney, male C57BL/6J, 0% K+ vs 1% K+ for 2 weeks (study 1), 6,078
  proteins, 7 (normal K) and 5 (0% K) mice, log2 FC (0K/NK), t-test p and BH q. PRIDE PXD035354. The paper's Methods
  text names study 1 '0.1K+ or 1K+' while its Results and the data set say 0K; recorded, not resolved here.

B2  SCP proteins (intake_core joint set, 5,130 matched proteins) matched to bulk rows by UniProt accession (any accession
    of either search's protein group), else by unique gene symbol; match type recorded.
B3  SCP effect: pseudobulk log2 FC pooled (step 3 / step 5 tables; mean back-transformed intensity per cell,
    missing = 0) as primary; depth-adjusted intensity log2 FC and detection log OR as secondary. Groups all_DCT and
    DCT1 under dct_label and dct_label_alternative, label-free all cells; all three variants.
B4  Spearman correlation with the bulk log2 FC (Fisher-z 95% CI) over all shared proteins and over bulk q < 0.05;
    sign agreement among bulk q < 0.05 proteins, raw and after centring each data set on its median; 2 x 2 table of
    proteins significant in both (SCP D1 direction up/down vs bulk up/down; Fisher exact); panel table.
B5  Null: the same correlation for SCP plate contrasts (P2 vs P1 within Ctrl and within ZeroK; technical) beside
    ZeroK vs Ctrl within P1, within P2 and pooled (pseudobulk effects, zerok_response_core_lib.effect_sizes).
B6  DCT-enriched subset from the kidney_rna RNA atlas pseudobulk (Celltype level, Cell and Nuclei pooled): log2((CPM_DCT
    + 1) / (CPM_other + 1)) >= 2 and CPM_DCT >= 10, other = all other Celltypes pooled.
All p-values are descriptive (SCP: one pool per condition; the bulk study has individual mice, but the two studies
differ in diet duration, control diet, tissue, platform and search).

Inputs (read-only): data/reference/jci.insight.196339.sdd1.xlsx; results/h5ad/zerok_core_DCT_joint_labelled.h5ad;
zerok_core panel, hash and joint labels; zerok_response_core step 3 proteome tables, step 5 label-free proteome table;
results/h5ad/kidney_rna_atlas_pseudobulk.h5ad.
Outputs (results/tables/zerok_response_core/): zerok_response_core_bulk_reference.csv, zerok_response_core_bulk_matched.csv,
zerok_response_core_bulk_concordance.csv, zerok_response_core_bulk_sign_tables.csv, zerok_response_core_bulk_plate_null.csv,
zerok_response_core_bulk_panel.csv, zerok_response_core_bulk_dct_enriched_genes.csv, zerok_response_core_bulk_checks.csv.

Command (project root):  python scripts/zerok_response_core_bulk_benchmark.py
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
import zerok_response_core_lib as lib

SCRIPT = "zerok_response_core_bulk_benchmark.py"
BULK_XLSX = cc.PROJECT_ROOT / "data" / "reference" / "jci.insight.196339.sdd1.xlsx"
BULK_SHA256 = "2f2398212810203df02c4432a4a87be13c159700696f4b67a775eab1fc6d5c68"   # as downloaded, 28 Oct 2026
BULK_SHEET = "Quantifiable pros & subclusters"
BULK_SOURCE = ("Murali SK et al. JCI Insight 2026;11(5):e196339, Supplemental data set 1 (whole kidney, 0% vs 1% K+, "
               "2 weeks; PXD035354)")
ATLAS_PB = cc.out_path("kidney_rna", "h5ad", "atlas_pseudobulk", ".h5ad")
DCT_LOG2_RATIO = 2.0
DCT_MIN_CPM = 10.0
SCP_EFFECTS = {"pseudobulk": "pseudobulk_log2fc_pooled", "intensity": "int_log2fc_adjusted",
               "detection": "det_logodds"}
GROUPS = [("dct_label", lib.ALL_DCT), ("dct_label", "DCT1"), ("dct_label_alternative", lib.ALL_DCT),
          ("dct_label_alternative", "DCT1"), ("label_free", lib.ALL_CELLS)]
CONTRASTS = {  # name: ((condition, plate) reference, (condition, plate) comparison); plate None = both plates
    "ZeroK vs Ctrl (pooled)": (("Ctrl", None), ("ZeroK", None)),
    "ZeroK vs Ctrl, P1 only": (("Ctrl", "P1"), ("ZeroK", "P1")),
    "ZeroK vs Ctrl, P2 only": (("Ctrl", "P2"), ("ZeroK", "P2")),
    "Ctrl P2 vs P1 (technical)": (("Ctrl", "P1"), ("Ctrl", "P2")),
    "ZeroK P2 vs P1 (technical)": (("ZeroK", "P1"), ("ZeroK", "P2")),
}
BULK_NOTE = ("bulk: whole kidney, 0% vs 1% K+ for 2 weeks, 5 vs 7 mice; SCP: sorted DCT cells, zero-K vs control "
             "diet for 4 days, one pool of 3 mice per condition; different platforms and searches; descriptive")


def read_bulk(run: cc.Run, checks: t5.Checks) -> pd.DataFrame:
    path = run.use_input(BULK_XLSX, BULK_SOURCE)
    sha = cc.sha256(path)
    checks.add("bulk file SHA256 equals the downloaded file", sha, sha == BULK_SHA256, BULK_SHA256)
    b = pd.read_excel(path, sheet_name=BULK_SHEET)
    b = b.rename(columns={"Accession": "bulk_accession", "Description": "bulk_description", "Gene name": "bulk_gene",
                          "Fold Change (0K/NK)": "bulk_fold_change", "Log2FC (0K/NK)": "bulk_log2fc",
                          "Pvalue": "bulk_p", "Qvalue": "bulk_q"})
    b = b[["bulk_accession", "bulk_description", "bulk_gene", "bulk_fold_change", "bulk_log2fc", "bulk_p", "bulk_q"]]
    b["bulk_accession"] = b["bulk_accession"].astype(str).str.strip()
    b["bulk_gene"] = b["bulk_gene"].astype(str).str.strip().replace({"nan": ""})
    n_up = int(((b["bulk_q"] < 0.05) & (b["bulk_log2fc"] > 0)).sum())
    n_down = int(((b["bulk_q"] < 0.05) & (b["bulk_log2fc"] < 0)).sum())
    checks.add("bulk proteins; q < 0.05 up / down (paper: 6,078; 773 up, 333 down)",
               f"{len(b)}; {n_up} / {n_down}", len(b) == 6078 and n_up == 773 and n_down == 333,
               "6078; 773 / 333")
    dev = float(np.nanmax(np.abs(np.log2(b["bulk_fold_change"]) - b["bulk_log2fc"])))
    checks.add("bulk Log2FC equals log2(Fold Change)", f"max |diff| {dev:.2g}", dev < 1e-6)
    b["bulk_direction"] = np.where(b["bulk_q"] >= 0.05, "n.s.", np.where(b["bulk_log2fc"] > 0, lib.UP, lib.DOWN))
    return b


def match_proteins(joint: lib.Joint, bulk: pd.DataFrame, checks: t5.Checks) -> pd.DataFrame:
    """B2: accession first (any accession of either search's protein group), then unique gene symbol."""
    acc_index = {}
    for i, a in enumerate(bulk["bulk_accession"]):
        for x in a.split(";"):
            acc_index.setdefault(x.strip(), []).append(i)
    gene_counts = bulk["bulk_gene"].value_counts()
    gene_index = {g: int(np.flatnonzero(bulk["bulk_gene"].to_numpy() == g)[0])
                  for g, n in gene_counts.items() if g and n == 1}
    rows = []
    var = joint.var
    for name in joint.var_names:
        accs = []
        for col in ("protein_group_ctrl", "protein_group_zerok"):
            accs += [x.strip() for x in str(var.at[name, col]).split(";") if x.strip() and x.strip() != "nan"]
        hits = sorted({i for a in dict.fromkeys(accs) for i in acc_index.get(a, [])})
        if len(hits) == 1:
            rows.append((name, hits[0], "accession"))
        elif len(hits) > 1:
            rows.append((name, -1, f"accession ambiguous ({len(hits)} bulk rows)"))
        else:
            g = str(var.at[name, "gene_name"])
            rows.append((name, gene_index.get(g, -1), "gene symbol" if g in gene_index else "no match"))
    m = pd.DataFrame(rows, columns=["protein", "bulk_row", "match_type"])
    counts = m["match_type"].value_counts().to_dict()
    checks.add("SCP joint proteins matched to bulk (accession; gene symbol; ambiguous; none)",
               "; ".join(f"{k} {v}" for k, v in counts.items()), None)
    dup = m.loc[m["bulk_row"] >= 0, "bulk_row"].duplicated().sum()
    checks.add("bulk rows matched by more than one SCP protein", f"{int(dup)}", None)
    ok = m["bulk_row"] >= 0
    out = m.loc[ok].merge(bulk.reset_index().rename(columns={"index": "bulk_row"}), on="bulk_row", how="left")
    out["gene_name"] = var.loc[out["protein"], "gene_name"].astype(str).to_numpy()
    out["match_level_t17"] = var.loc[out["protein"], "match_level"].astype(str).to_numpy()
    return out


def dct_enriched(run: cc.Run, checks: t5.Checks) -> pd.DataFrame:
    """B6: DCT-enriched genes from the kidney_rna atlas pseudobulk (Celltype level)."""
    import anndata as ad

    a = ad.read_h5ad(run.use_input(ATLAS_PB, "kidney_rna atlas pseudobulk (per-group sums, all genes)"))
    counts = pd.DataFrame(cc.dense(a.layers["sum_counts"]), index=a.obs_names, columns=a.var_names.astype(str))
    by = counts.groupby(a.obs["Celltype"].astype(str).to_numpy()).sum()
    dct = by.loc["DCT"]
    other = by.drop(index="DCT").sum(axis=0)
    cpm_dct = dct / dct.sum() * 1e6
    cpm_other = other / other.sum() * 1e6
    ratio = np.log2((cpm_dct + 1) / (cpm_other + 1))
    frame = pd.DataFrame({"gene": by.columns, "cpm_dct": cpm_dct.to_numpy(), "cpm_other": cpm_other.to_numpy(),
                          "log2_ratio_dct_vs_other": ratio.to_numpy()})
    frame["dct_enriched"] = (frame["log2_ratio_dct_vs_other"] >= DCT_LOG2_RATIO) & (frame["cpm_dct"] >= DCT_MIN_CPM)
    n = int(frame["dct_enriched"].sum())
    known = [g for g in ("Slc12a3", "Pvalb", "Calb1", "Trpm6", "Egf", "Kcnj10", "Wnk4") if g in set(frame["gene"])]
    flags = frame.set_index("gene").loc[known, "dct_enriched"]
    checks.add("atlas DCT-enriched genes (log2 ratio >= 2, CPM >= 10)", f"{n}", n > 0)
    checks.add("DCT marker genes flagged DCT-enriched", "; ".join(f"{g} {bool(v)}" for g, v in flags.items()), None)
    return frame


def spearman_ci(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 5:
        return {"n": n, "spearman": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}
    res = stats.spearmanr(x[ok], y[ok])
    z, se = np.arctanh(np.clip(res.statistic, -0.999999, 0.999999)), 1 / np.sqrt(n - 3)
    return {"n": n, "spearman": float(res.statistic), "ci_low": float(np.tanh(z - 1.96 * se)),
            "ci_high": float(np.tanh(z + 1.96 * se)), "p": float(res.pvalue)}


def sign_stats(scp, bulk) -> dict:
    s, b = np.asarray(scp, float), np.asarray(bulk, float)
    ok = np.isfinite(s) & np.isfinite(b) & (s != 0) & (b != 0)
    if not ok.any():
        return {"n_sign": 0, "sign_agreement": np.nan, "sign_ci_low": np.nan, "sign_ci_high": np.nan}
    k, n = int((np.sign(s[ok]) == np.sign(b[ok])).sum()), int(ok.sum())
    lo, hi = lib.wilson(k, n)
    return {"n_sign": n, "sign_agreement": k / n, "sign_ci_low": lo, "sign_ci_high": hi}


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_response_core step 5c: bulk low-K proteome benchmark.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "bulk", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")
    lib.check_panel_hash(run, checks)
    bulk = read_bulk(run, checks)
    joint = lib.load_joint(run, checks)
    panel = lib.read_panel(run)
    matched = match_proteins(joint, bulk, checks)
    enr = dct_enriched(run, checks)
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
    bts = {v: lib.back_transform(joint, v) for v in lib.VARIANTS}
    cond = joint.obs["condition"].to_numpy()
    plate = joint.obs["plate"].to_numpy()
    mcols = ["protein", "match_type", "bulk_accession", "bulk_gene", "bulk_log2fc", "bulk_q", "bulk_direction",
             "dct_enriched"]

    conc_rows, sign_rows, null_rows, long_rows = [], [], [], []
    max_dev = 0.0
    for col, group in GROUPS:
        gmask = (np.ones(len(cond), bool) if col == "label_free" else lib.group_masks(joint.obs, col)[group])
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
                           "protein_set": sname, **spearman_ci(s_all[smask], b_all[smask])}
                    row.update({f"{k}": v_ for k, v_ in sign_stats(s_all[smask], b_all[smask]).items()})
                    cs = sign_stats(s_all[smask] - s_med, b_all[smask] - b_med)
                    row.update({"sign_agreement_centred": cs["sign_agreement"],
                                "sign_centred_ci_low": cs["sign_ci_low"], "sign_centred_ci_high": cs["sign_ci_high"],
                                "scp_median_shift": s_med, "bulk_median_shift": b_med})
                    conc_rows.append(row)
            # 2 x 2 of proteins significant in both
            for sname in ("all shared", "DCT-enriched"):
                sel = d.loc[d["bulk_sig"] & d["observed_direction"].isin([lib.UP, lib.DOWN])
                            & (sets[sname] if sname == "all shared" else d["dct_enriched"])]
                tab = pd.crosstab(sel["observed_direction"], sel["bulk_direction"]).reindex(
                    index=[lib.UP, lib.DOWN], columns=[lib.UP, lib.DOWN], fill_value=0)
                a_, b_, c_, d_ = (int(tab.at[lib.UP, lib.UP]), int(tab.at[lib.UP, lib.DOWN]),
                                  int(tab.at[lib.DOWN, lib.UP]), int(tab.at[lib.DOWN, lib.DOWN]))
                orr, p = stats.fisher_exact([[a_, b_], [c_, d_]]) if (a_ + b_) and (c_ + d_) else (np.nan, np.nan)
                n_ = a_ + b_ + c_ + d_
                lo, hi = lib.wilson(a_ + d_, n_) if n_ else (np.nan, np.nan)
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
                if min(ref.sum(), cmp_.sum()) < lib.MIN_GROUP_PROTEOME:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    e = lib.effect_sizes(joint, v, np.flatnonzero(ref), np.flatnonzero(cmp_), cols_idx, bts[v])
                if cname == "ZeroK vs Ctrl (pooled)":
                    max_dev = max(max_dev, float(np.nanmax(np.abs(e["pseudobulk_log2fc"]
                                                                  - d["pseudobulk_log2fc_pooled"].to_numpy(float)))))
                for sname in ("all shared", "bulk q < 0.05", "DCT-enriched"):
                    sm = sets[sname]
                    null_rows.append({"label_column": col, "group": group, "variant": v, "contrast": cname,
                                      "technical": "technical" in cname, "protein_set": sname,
                                      "n_reference": int(ref.sum()), "n_comparison": int(cmp_.sum()),
                                      **spearman_ci(e["pseudobulk_log2fc"][sm], bvals[sm]),
                                      **sign_stats(e["pseudobulk_log2fc"][sm], bvals[sm])})
    checks.add("pooled pseudobulk effect recomputed equals the step 3 / step 5 tables", f"max |diff| {max_dev:.2g}",
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
    run.finish(parameters={"bulk_source": BULK_SOURCE, "bulk_sheet": BULK_SHEET, "bulk_sha256": BULK_SHA256,
                           "matching": "UniProt accession of either search's protein group, else unique gene symbol",
                           "scp_effects": SCP_EFFECTS, "groups": GROUPS, "contrasts": list(CONTRASTS),
                           "dct_enriched_rule": f"log2((CPM_DCT+1)/(CPM_other+1)) >= {DCT_LOG2_RATIO} and "
                                                f"CPM_DCT >= {DCT_MIN_CPM}; atlas Celltype level, Cell + Nuclei pooled",
                           "downloaded": "28 Oct 2026, author-approved", "note": BULK_NOTE,
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_response_core_bulk_checks.csv.")


if __name__ == "__main__":
    main()
