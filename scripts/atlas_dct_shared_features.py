#!/usr/bin/env python
"""atlas_dct step 2: genes shared between the RNA atlas and the DCT protein matrix (2,561 proteins).

Analysis plan, step 2.

Rule (as kidney_rna / kidney_annotation step 3a): protein names (first PG.Genes symbol, mouse capitalization) are matched to the atlas
.raw gene symbols case-insensitively. Protein entries made unique with a suffix (``Name_<column index>``, as
scp_dct_upload.py) are duplicates of an existing symbol and are not matched; the entry with the exact name is
used. Proteins without a symbol ('Nan') are not matched. The match must be one-to-one.

Per protein the table gives the match status, SCP detection (all cells and per V1 label, intensity > 0) and atlas
detection (fraction with counts > 0) in each published RNA group and in the atlas_dct-derived DCT1 / DCT2 groups.

Inputs (read-only): kidney_object canonical DCT object and labels; results/h5ad/atlas_dct_atlas_dct_region.h5ad;
                    results/tables/atlas_dct/atlas_dct_dct_split_per_cell.csv.
Outputs: results/tables/atlas_dct/atlas_dct_shared_features.csv, atlas_dct_shared_features_summary.csv,
         atlas_dct_shared_features_checks.csv

Command (project root):  python scripts/atlas_dct_shared_features.py
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
import atlas_dct_lib as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atlas_dct step 2: shared features (atlas genes x DCT proteins).")
    cc.add_common_args(parser)
    return parser.parse_args()


def status_of(name: str, matched: dict[str, str], names: set[str], atlas_lower: set[str]) -> tuple[str, str]:
    if name in matched:
        return ("matched, same case" if matched[name] == name else "matched, case differs"), matched[name]
    if name == "Nan" or name.startswith("Nan_"):
        return "no gene symbol (Nan)", ""
    if "_" in name:
        base = name.rsplit("_", 1)[0]
        if base in names:
            return f"duplicate entry of {base} (exact-name entry used)", ""
        return "name with '_' (not matched)", ""
    return ("not in atlas genes" if name.lower() not in atlas_lower else "not matched"), ""


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "shared_features", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, ln1p, det = T.load_scp(run, checks)
    region = T.load_region(run, with_split=True)
    prot = pd.Index(adata.var_names.astype(str))
    genes = pd.Index(region.var_names.astype(str))
    checks.add("DCT proteins; atlas genes", f"{len(prot)}; {len(genes)}", len(prot) == 2561 and len(genes) == 16711)

    match = T.shared_features(prot, genes)
    matched = dict(zip(match["protein"], match["atlas_gene"]))
    checks.add("protein <-> atlas gene matching one-to-one", f"{len(match)} pairs", True)
    names, atlas_lower = set(prot), {g.lower() for g in genes}

    rows = [dict(zip(("status", "atlas_gene"), status_of(p, matched, names, atlas_lower)), protein=p) for p in prot]
    tab = pd.DataFrame(rows)[["protein", "atlas_gene", "status"]]
    tab["shared"] = tab["atlas_gene"] != ""
    tab["scp_detection_all"] = det.mean(axis=0).to_numpy()
    for col in T.LABEL_COLUMNS:
        st = T.scp_group_stats(ln1p, det, labels[col])
        for g in T.SCP_GROUPS:
            tab[f"scp_detection_{T.LABEL_SHORT[col]}_{g}"] = st["frac"].loc[g].to_numpy()

    counts = region.X
    pos = genes.get_indexer(tab.loc[tab["shared"], "atlas_gene"])
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
    tab.loc[~tab["shared"], [c for c in tab.columns if c.startswith("atlas_detection")]] = np.nan

    summ = tab["status"].str.replace(r"duplicate entry of .*", "duplicate entry (exact-name entry used)",
                                     regex=True).value_counts().rename_axis("status").reset_index(name="n_proteins")
    for r in summ.itertuples():
        checks.add(f"proteins: {r.status}", f"{r.n_proteins}", None)
    shared = tab[tab["shared"]]
    for t_ in T.DETECTION_SENS:
        rna_any = (shared[[c for c in shared.columns if c.startswith("atlas_detection_") and "_atlas_dct" not in c]]
                   >= t_).any(axis=1)
        prot_any = (shared[[c for c in shared.columns if c.startswith("scp_detection_cell_")
                            or c.startswith("scp_detection_cluster_")]] >= t_).any(axis=1)
        checks.add(f"shared genes detected in >= {t_:.0%} of some group: RNA (published groups); protein (any V1 "
                   "label group); both", f"{int(rna_any.sum())}; {int(prot_any.sum())}; {int((rna_any & prot_any).sum())}",
                   None)
    dup = tab["status"].str.startswith("duplicate entry")
    checks.add("duplicate protein entries (suffix) and their symbols",
               f"{int(dup.sum())}: " + ", ".join(tab.loc[dup, "protein"].tolist()), None)
    run.table(tab, "shared_features")
    run.table(summ, "shared_features_summary")
    run.table(checks.frame(), "shared_features_checks")
    run.finish({"matching": "case-insensitive; names with '_' (duplicate entries) and 'Nan' not matched (as kidney_rna)",
                "detection": "SCP intensity > 0; atlas counts > 0", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see atlas_dct_shared_features_checks.csv.")


if __name__ == "__main__":
    main()
