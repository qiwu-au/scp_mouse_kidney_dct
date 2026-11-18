#!/usr/bin/env python
"""ctrl_rna_core step 1: genes shared between the RNA atlas and the control DCT protein matrix (6,196 proteins).

Analysis plan,
step 1 (port of atlas_dct_shared_features.py; atlas_dct rule unchanged).

Rule (atlas_dct / kidney_rna): protein names (intake_core ``var_names`` = cleaned first PG.Genes symbol) are matched to the atlas .raw
gene symbols case-insensitively, one-to-one (``atlas_dct_lib.shared_features``). Protein groups listing several
genes are matched by their first symbol only. Repeated names made unique by intake_core (``Name_1``, ``Name_2``; the first
entry keeps the bare name) are not matched; the exact-name entry is used. Groups without a PG.Genes symbol (named by
their first UniProt accession in intake_core) are not matched.

Per protein the table gives the match status, PG.Genes (from the intake_core object), SCP detection (all cells and per label
of both ctrl_rna_core label columns, intensity > 0) and atlas detection (fraction with counts > 0) in each published RNA group
and in the atlas_dct-derived DCT1 / DCT2 groups. Check: per published group, the region-subset mean log-normalized
expression and detection on the shared genes equal the kidney_rna atlas pseudobulk.

Inputs (read-only): labels_core labelled control object and canonical labels; results/h5ad/intake_core_DCT_ctrl.h5ad (var
                    only: PG.Genes); results/h5ad/atlas_dct_atlas_dct_region.h5ad;
                    results/tables/atlas_dct/atlas_dct_dct_split_per_cell.csv; results/h5ad/kidney_rna_atlas_pseudobulk.h5ad.
Outputs: results/tables/ctrl_rna_core/ctrl_rna_core_shared_features.csv, ctrl_rna_core_shared_features_summary.csv,
         ctrl_rna_core_shared_features_checks.csv

Command (project root):  python scripts/ctrl_rna_core_shared_features.py
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
import cluster_de_lib as t5
import kidney_rna_lib as L7
import atlas_dct_lib as T
import ctrl_rna_core_lib as L
from atlas_dct_shared_features import status_of

PB_TOL = 1e-5            # relative tolerance, float32 pseudobulk sums vs region means
NO_SYMBOL = "no gene symbol (UniProt accession name)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_rna_core step 1: shared features (atlas genes x control DCT proteins).")
    cc.add_common_args(parser)
    return parser.parse_args()


def protein_groups(run: cc.Run, var_names) -> pd.DataFrame:
    """PG.Genes of the control proteins from the intake_core object (the labels_core object keeps gene_name only)."""
    import anndata as ad

    a = ad.read_h5ad(run.use_input(cc.DCT_NEW_OBJECTS["Ctrl"], "intake_core object (Ctrl): var (PG.Genes)"), backed="r")
    var = a.var.copy()
    names = list(a.var_names.astype(str))
    a.file.close()
    if names != list(map(str, var_names)):
        raise ValueError("intake_core and labels_core control objects hold different proteins")
    pg = var["pg_genes"].astype(str).where(var["pg_genes"].notna(), "").str.strip().replace("nan", "")
    genes = pg.str.split(";").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    return pd.DataFrame({"protein_group": var["protein_group"].astype(str).to_numpy(), "pg_genes": pg.to_numpy(),
                         "n_genes_in_group": genes.apply(lambda xs: len(dict.fromkeys(xs))).to_numpy()},
                        index=pd.Index(names, name="protein"))


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "shared_features", args)
    L.ported_from(run, "atlas_dct_shared_features.py")
    checks = t5.Checks(L.DATASET)
    adata, labels, ln1p, det = L.load_scp(run, checks)
    region = T.load_region(run, with_split=True)
    prot = pd.Index(adata.var_names.astype(str))
    genes = pd.Index(region.var_names.astype(str))
    checks.add("control proteins; atlas genes", f"{len(prot)}; {len(genes)}",
               len(prot) == L.N_PROTEINS and len(genes) == T.ATLAS_EXPECTED_RAW_GENES)
    split_n = region.obs.loc[region.obs["rna_group"] == "DCT", "dct_split_label"].value_counts()
    exp_split = {T.DCT1_ATLAS_DCT: 8864, T.DCT2_ATLAS_DCT: 2752, T.DCT_UNASSIGNED: 4765}
    checks.add("atlas_dct-derived split: cells DCT1 / DCT2 / unassigned (atlas_dct) equal the atlas_dct README",
               " / ".join(f"{int(split_n.get(k, 0))}" for k in exp_split), split_n.to_dict() == exp_split)
    pgs = protein_groups(run, prot)

    match = T.shared_features(prot, genes)
    matched = dict(zip(match["protein"], match["atlas_gene"]))
    checks.add("protein <-> atlas gene matching one-to-one", f"{len(match)} pairs", True)
    names, atlas_lower = set(prot), {g.lower() for g in genes}

    rows = []
    for p in prot:
        if pgs.loc[p, "pg_genes"] == "":
            status, ag = NO_SYMBOL, ""
        else:
            status, ag = status_of(p, matched, names, atlas_lower)
        rows.append({"protein": p, "atlas_gene": ag, "status": status})
    tab = pd.DataFrame(rows)
    tab = tab.join(pgs, on="protein")
    tab["shared"] = tab["atlas_gene"] != ""
    checks.add("no protein without a PG.Genes symbol is matched", f"{int((tab['status'].eq(NO_SYMBOL) & tab['shared']).sum())}",
               not bool((tab["status"].eq(NO_SYMBOL) & tab["shared"]).any()))
    tab["scp_detection_all"] = det.mean(axis=0).to_numpy()
    for col in L.LABEL_COLUMNS:
        st = T.scp_group_stats(ln1p, det, labels[col], L.SCP_GROUPS)
        for g in L.SCP_GROUPS:
            tab[f"scp_detection_{L.LABEL_SHORT[col]}_{g}"] = st["frac"].loc[g].to_numpy()

    counts = region.X
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
            col = f"atlas_detection_{g.replace(' (atlas_dct)', '_atlas_dct')}"
            tab[col] = np.nan
            tab.loc[tab["shared"], col] = frac

    # region subset vs kidney_rna pseudobulk (published groups, shared genes) ------------------------------------
    pb = L7.read_pseudobulk(run)
    ln = T.lognorm(counts)[:, pos]
    st = T.group_stats(ln, sub_counts, region.obs["rna_group"], T.PUBLISHED_GROUPS, pd.Index(shared_genes))
    d_mean, d_frac, n_ok = 0.0, 0.0, True
    for g in T.PUBLISHED_GROUPS:
        agg = L7.aggregate(pb, L7.group_mask(pb, **{T.LEVEL: g}), shared_genes)
        n_ok &= int(agg["n_cells"].iloc[0]) == int(st["n"][g])
        d_mean = max(d_mean, float(np.max(np.abs(st["mean"].loc[g].to_numpy() - agg["mean_lognorm"].to_numpy())
                                          / np.maximum(np.abs(agg["mean_lognorm"].to_numpy()), 1.0))))
        d_frac = max(d_frac, float(np.max(np.abs(st["frac"].loc[g].to_numpy() - agg["frac_detected"].to_numpy()))))
    checks.add("region subset vs kidney_rna pseudobulk, published groups x shared genes: cells per group equal; max relative "
               "difference of mean log-normalized; max abs difference of detection",
               f"{n_ok}; {d_mean:.1e}; {d_frac:.1e}", n_ok and d_mean < PB_TOL and d_frac < PB_TOL, f"< {PB_TOL:g}")

    summ = tab["status"].str.replace(r"duplicate entry of .*", "duplicate entry (exact-name entry used)",
                                     regex=True).value_counts().rename_axis("status").reset_index(name="n_proteins")
    for r in summ.itertuples():
        checks.add(f"proteins: {r.status}", f"{r.n_proteins}", None)
    checks.add("shared genes / control proteins", f"{int(tab['shared'].sum())} / {len(tab)}", None)
    shared = tab[tab["shared"]]
    for t_ in T.DETECTION_SENS:
        rna_any = (shared[[c for c in shared.columns if c.startswith("atlas_detection_") and "_atlas_dct" not in c]]
                   >= t_).any(axis=1)
        prot_any = (shared[[c for c in shared.columns if c.startswith("scp_detection_deg_")
                            or c.startswith("scp_detection_cluster_")]] >= t_).any(axis=1)
        checks.add(f"shared genes detected in >= {t_:.0%} of some group: RNA (published groups); protein (any label "
                   "group of either column); both",
                   f"{int(rna_any.sum())}; {int(prot_any.sum())}; {int((rna_any & prot_any).sum())}", None)
    dup = tab["status"].str.startswith("duplicate entry")
    checks.add("duplicate protein entries (suffix)", f"{int(dup.sum())}: " + ", ".join(tab.loc[dup, "protein"]), None)
    other_us = tab["status"].eq("name with '_' (not matched)")
    if other_us.any():
        checks.add("names with '_' that are not suffixed repeats", ", ".join(tab.loc[other_us, "protein"]), None)
    case = tab["status"].eq("matched, case differs")
    checks.add("matched with a case difference (protein / atlas gene)",
               ", ".join(f"{r.protein} / {r.atlas_gene}" for r in tab[case].itertuples()) or "none", None)
    multi = tab["n_genes_in_group"] > 1
    checks.add("protein groups listing > 1 gene: all; shared (matched by the first symbol)",
               f"{int(multi.sum())}; {int((multi & tab['shared']).sum())}", None)
    # secondary symbols of multi-gene groups that are atlas genes not matched by any protein (information only)
    matched_lower = {g.lower() for g in matched.values()}
    sec = sorted({x.strip() for s in tab.loc[multi, "pg_genes"] for x in s.split(";")[1:]
                  if x.strip().lower() in atlas_lower and x.strip().lower() not in matched_lower})
    checks.add("secondary PG.Genes symbols that are atlas genes without a matched protein (not used, as atlas_dct)",
               f"{len(sec)}: {', '.join(sec)}" if sec else "0", None)

    run.table(tab, "shared_features")
    run.table(summ, "shared_features_summary")
    run.table(checks.frame(), "shared_features_checks")
    run.finish({"matching": "case-insensitive on the first PG.Genes symbol; suffixed repeats ('_<n>') and groups "
                "without a PG.Genes symbol not matched (atlas_dct rule)", "detection": "SCP intensity > 0; atlas counts > 0",
                "label_columns": list(L.LABEL_COLUMNS), "scp_groups": list(L.SCP_GROUPS), "pseudobulk_tolerance": PB_TOL,
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see ctrl_rna_core_shared_features_checks.csv.")


if __name__ == "__main__":
    main()
