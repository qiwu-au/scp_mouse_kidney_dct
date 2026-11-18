#!/usr/bin/env python
"""dct_rna step 1: genes shared between the RNA atlas and each condition's protein matrix (combined search).

Analysis plan,
step 1 (port of ctrl_rna_core_shared_features.py with a condition argument; atlas_dct rule and ``status_of`` unchanged).

Rule (atlas_dct / ctrl_rna_core): protein names (dct_intake ``var_name`` = cleaned first PG.Genes symbol, the intake_core naming rule) are matched
to the atlas .raw gene symbols case-insensitively, one-to-one (``atlas_dct_lib.shared_features``). Groups listing
several genes are matched by their first symbol only; repeated names made unique (``Name_1``; the first entry keeps
the bare name) are not matched; groups without a PG.Genes symbol (named by their UniProt accession) are not matched.

Per condition (Ctrl first) and protein: match status, PG.Genes (dct_intake protein table), SCP detection (all cells and per
label of the primary and second columns; absent labels left empty), atlas detection per published RNA group and in
the atlas_dct-derived DCT1 / DCT2 groups. Check per condition: the region-subset mean log-normalized expression and
detection on the shared genes equal the kidney_rna atlas pseudobulk.

Inputs (read-only): dct_labels labelled objects and canonical labels (via dct_reviewer_lib.load); dct_intake protein table;
                    dct_intake filtering summary; atlas_dct region subset and split table; kidney_rna atlas pseudobulk.
Outputs: results/tables/dct_rna/dct_rna_shared_features.csv, dct_rna_shared_features_summary.csv,
         dct_rna_shared_features_checks.csv

Command (project root):  python scripts/dct_rna_shared_features.py
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
import kidney_rna_lib as L7
import atlas_dct_lib as T
import dct_rna_lib as L
from atlas_dct_shared_features import status_of
from ctrl_rna_core_shared_features import NO_SYMBOL, PB_TOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_rna step 1: shared features (atlas genes x DCT proteins, per condition).")
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run, cond, region, counts, genes, ln_all, pb, checks):
    s = L.SETTINGS[cond]
    adata, labels, ln1p, det = L.load_scp(run, cond, checks)
    L.depth_quartiles(run, cond, labels, checks)
    prot = pd.Index(adata.var_names.astype(str))
    pgs = L.protein_groups(run, cond, prot)
    checks.add("dct_intake protein table: retained proteins and order equal the dct_labels object", f"{len(pgs)}", True, "", cond)

    match = T.shared_features(prot, genes)
    matched = dict(zip(match["protein"], match["atlas_gene"]))
    checks.add("protein <-> atlas gene matching one-to-one", f"{len(match)} pairs",
               match["protein"].is_unique and match["atlas_gene"].is_unique, "", cond)
    names, atlas_lower = set(prot), {g.lower() for g in genes}
    rows = []
    for p in prot:
        if pgs.loc[p, "pg_genes"] == "":
            status, ag = NO_SYMBOL, ""
        else:
            status, ag = status_of(p, matched, names, atlas_lower)
        rows.append({"protein": p, "atlas_gene": ag, "status": status})
    tab = pd.DataFrame(rows).join(pgs, on="protein")
    tab["shared"] = tab["atlas_gene"] != ""
    bad = int((tab["status"].eq(NO_SYMBOL) & tab["shared"]).sum())
    checks.add("no protein without a PG.Genes symbol is matched", f"{bad}", bad == 0, "", cond)
    tab["scp_detection_all"] = det.mean(axis=0).to_numpy()
    for col in s.LABEL_COLUMNS:
        role = L.role_of(cond, col)
        groups = L.present(labels, col)
        st = T.scp_group_stats(ln1p, det, labels[col], groups)
        for g in L.SCP_GROUPS:
            tab[f"scp_detection_{role}_{g}"] = st["frac"].loc[g].to_numpy() if g in groups else np.nan
        checks.add(f"{col} ({role}): labels present; absent (columns left empty)",
                   f"{', '.join(groups)}; {', '.join(L.absent(labels, col)) or 'none'}", None, "", cond)

    shared_genes = tab.loc[tab["shared"], "atlas_gene"].tolist()
    pos = genes.get_indexer(shared_genes)
    sub_counts = counts[:, pos]
    for derived in (False, True):
        grp = region.obs[T.group_column(derived)].astype(str).to_numpy()
        for g in T.rna_groups(derived):
            if derived and g not in (T.DCT1_ATLAS_DCT, T.DCT2_ATLAS_DCT):
                continue
            m = grp == g
            frac = np.asarray((sub_counts[np.flatnonzero(m)] > 0).mean(axis=0)).ravel()
            c = f"atlas_detection_{g.replace(' (atlas_dct)', '_atlas_dct')}"
            tab[c] = np.nan
            tab.loc[tab["shared"], c] = frac

    # region subset vs kidney_rna pseudobulk (published groups, this condition's shared genes) -----------------------------
    st = T.group_stats(ln_all[:, pos], sub_counts, region.obs["rna_group"], T.PUBLISHED_GROUPS, pd.Index(shared_genes))
    d_mean, d_frac, n_ok = 0.0, 0.0, True
    for g in T.PUBLISHED_GROUPS:
        agg = L7.aggregate(pb, L7.group_mask(pb, **{T.LEVEL: g}), shared_genes)
        n_ok &= int(agg["n_cells"].iloc[0]) == int(st["n"][g])
        d_mean = max(d_mean, float(np.max(np.abs(st["mean"].loc[g].to_numpy() - agg["mean_lognorm"].to_numpy())
                                          / np.maximum(np.abs(agg["mean_lognorm"].to_numpy()), 1.0))))
        d_frac = max(d_frac, float(np.max(np.abs(st["frac"].loc[g].to_numpy() - agg["frac_detected"].to_numpy()))))
    checks.add("region subset vs kidney_rna pseudobulk, published groups x shared genes: cells per group equal; max relative "
               "difference of mean log-normalized; max abs difference of detection",
               f"{n_ok}; {d_mean:.1e}; {d_frac:.1e}", n_ok and d_mean < PB_TOL and d_frac < PB_TOL, f"< {PB_TOL:g}", cond)

    summ = tab["status"].str.replace(r"duplicate entry of .*", "duplicate entry (exact-name entry used)",
                                     regex=True).value_counts().rename_axis("status").reset_index(name="n_proteins")
    for r in summ.itertuples():
        checks.add(f"proteins: {r.status}", f"{r.n_proteins}", None, "", cond)
    checks.add("shared genes / proteins", f"{int(tab['shared'].sum())} / {len(tab)}", None, "", cond)
    shared = tab[tab["shared"]]
    for t_ in T.DETECTION_SENS:
        rna_any = (shared[[c for c in shared.columns if c.startswith("atlas_detection_") and "_atlas_dct" not in c]]
                   >= t_).any(axis=1)
        prot_any = (shared[[c for c in shared.columns if c.startswith("scp_detection_primary_")
                            or c.startswith("scp_detection_second_")]] >= t_).any(axis=1)
        checks.add(f"shared genes detected in >= {t_:.0%} of some group: RNA (published groups); protein (any label "
                   "group of either column); both",
                   f"{int(rna_any.sum())}; {int(prot_any.sum())}; {int((rna_any & prot_any).sum())}", None, "", cond)
    dup = tab["status"].str.startswith("duplicate entry")
    checks.add("duplicate protein entries (suffix)", f"{int(dup.sum())}: " + ", ".join(tab.loc[dup, "protein"]), None,
               "", cond)
    other_us = tab["status"].eq("name with '_' (not matched)")
    if other_us.any():
        checks.add("names with '_' that are not suffixed repeats", ", ".join(tab.loc[other_us, "protein"]), None, "",
                   cond)
    case = tab["status"].eq("matched, case differs")
    checks.add("matched with a case difference (protein / atlas gene)",
               ", ".join(f"{r.protein} / {r.atlas_gene}" for r in tab[case].itertuples()) or "none", None, "", cond)
    multi = tab["n_genes_in_group"] > 1
    checks.add("protein groups listing > 1 gene: all; shared (matched by the first symbol)",
               f"{int(multi.sum())}; {int((multi & tab['shared']).sum())}", None, "", cond)
    matched_lower = {g.lower() for g in matched.values()}
    sec = sorted({x.strip() for v in tab.loc[multi, "pg_genes"] for x in v.split(";")[1:]
                  if x.strip().lower() in atlas_lower and x.strip().lower() not in matched_lower})
    checks.add("secondary PG.Genes symbols that are atlas genes without a matched protein (not used, as atlas_dct)",
               f"{len(sec)}: {', '.join(sec)}" if sec else "0", None, "", cond)
    return tab, summ


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "shared_features", args)
    L.ported_from(run, "ctrl_rna_core_shared_features.py", "atlas_dct_shared_features.py")
    checks = L.Checks(L.DATASET)
    region = T.load_region(run, with_split=True)
    genes = pd.Index(region.var_names.astype(str))
    checks.add("atlas genes (.raw)", f"{len(genes)}", len(genes) == T.ATLAS_EXPECTED_RAW_GENES)
    split_n = region.obs.loc[region.obs["rna_group"] == "DCT", "dct_split_label"].value_counts()
    checks.add("atlas_dct-derived split: cells DCT1 / DCT2 / unassigned (atlas_dct) equal the atlas_dct README",
               " / ".join(f"{int(split_n.get(k, 0))}" for k in L.ATLAS_DCT_SPLIT_EXPECTED),
               split_n.to_dict() == L.ATLAS_DCT_SPLIT_EXPECTED)
    counts = region.X
    ln_all = T.lognorm(counts)
    pb = L7.read_pseudobulk(run)
    tabs, summs = {}, {}
    for cond in L.CONDITIONS:
        print(f"[shared_features] {cond}", flush=True)
        tabs[cond], summs[cond] = one_condition(run, cond, region, counts, genes, ln_all, pb, checks)
    both = set(tabs["Ctrl"].loc[tabs["Ctrl"]["shared"], "atlas_gene"]) & set(tabs["ZeroK"].loc[tabs["ZeroK"]["shared"],
                                                                                               "atlas_gene"])
    union = set(tabs["Ctrl"].loc[tabs["Ctrl"]["shared"], "atlas_gene"]) | set(tabs["ZeroK"].loc[tabs["ZeroK"]["shared"],
                                                                                                "atlas_gene"])
    checks.add("shared genes in both conditions; in either (information; no comparison between conditions)",
               f"{len(both)}; {len(union)}", None)
    run.table(L.stack(tabs), "shared_features")
    run.table(L.stack(summs), "shared_features_summary")
    run.table(checks.frame(), "shared_features_checks")
    run.finish({"matching": "case-insensitive on the first PG.Genes symbol; suffixed repeats ('_<n>') and groups "
                "without a PG.Genes symbol not matched (atlas_dct rule)", "detection": "SCP intensity > 0; atlas counts > 0",
                "label_columns": {c: list(L.SETTINGS[c].LABEL_COLUMNS) for c in L.CONDITIONS},
                "scp_groups": list(L.SCP_GROUPS), "pseudobulk_tolerance": PB_TOL, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see dct_rna_shared_features_checks.csv.")


if __name__ == "__main__":
    main()
