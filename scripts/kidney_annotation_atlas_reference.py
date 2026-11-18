#!/usr/bin/env python
"""kidney_annotation step 3a: atlas-derived marker candidates for the kidney cluster annotation (extended marker set).

Requested by the authors on 14 Oct 2026 (option 2: markers
beyond the V3 workbook, derived from the RNA atlas, used with the same DEG-based cluster annotation). This is a
declared deviation from the 'kidney V3 markers only' rule; the V3-only results stay the reference.

Atlas: mka.h5ad (read-only; 141,401 cells). Counts from .raw.X (checked to be integers), normalized to 10,000 per
cell and log1p (as the earlier integrate_scp_with_mka.py). Atlas Celltype -> SCP type mapping as proposed to and
approved by the authors (14 Oct 2026; table below). Mixed classes (DTL-ATL, LOH, DCT-CNT, CD-Trans) and Unknown are
left out of the marker derivation.

Marker derivation, on up to 2,000 cells per atlas Celltype (seed 42), for genes quantified in the 4,054-protein
kidney matrix (case-insensitive symbol match):
  per SCP type vs all other sampled cells: detection (counts > 0) in / out, Wilcoxon AUC (cluster_de_lib, tie
  corrected), mean log-normalized expression and scanpy-style log2 fold change;
  marker = detected in >= 25% of the type's cells, AUC >= 0.70, log2FC >= 1; the top 20 per type by AUC are kept.
  A gene selected for >= 2 types is shared (strict rule, as cluster_de).
Exclusion filter (authors, 14 Oct 2026, decided after the first run's marker table and before any annotation): genes
are not eligible if they are ribosomal proteins (Rps*, Rpl*, Rplp*, Rpsa), MitoCarta3.0 proteins (qc_mito annotation)
or detected in > 50% of the 712 SCP cells (label-independent); the top 20 are then picked from the eligible genes.

Inputs (read-only): mka.h5ad; kidney_object canonical kidney object (protein names, detection); data/KIDNEY/KIDNEY_markers_V3.xlsx.
Outputs: results/tables/kidney_annotation/kidney_annotation_atlas_*.csv

Command (project root):  python scripts/kidney_annotation_atlas_reference.py
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
import kidney_annotation_scan_lib as s

ATLAS = cc.PROJECT_ROOT / "data" / "reference" / "mka.h5ad"
LABEL_COL = "Celltype"
MAX_PER_CELLTYPE = 2000
TARGET_SUM = 1e4
MIN_DET_IN = 0.25
MIN_AUC = 0.70
MIN_LOG2FC = 1.0
TOP_PER_TYPE = 20
MAX_SCP_DETECTION = 0.50
RIBOSOMAL = r"^(Rp[ls]\d|Rplp\d|Rpsa$)"
QC_MITO_ANNOTATION = cc.OUT_ROOTS["tables"] / "qc_mito" / "qc_mito_kidney_protein_annotation.csv"
APPROVED = "proposed to and approved by the authors, 14 Oct 2026"
# atlas Celltype -> (SCP type or None, role, note)
ATLAS_MAP = {
    "PTS1": ("PCT", "target", "proximal tubule S1"), "PTS2": ("PCT", "target", "proximal tubule S2"),
    "PTS3": ("PST", "target", "proximal tubule S3"), "PTS3T2": ("PST", "target", "proximal tubule S3 (T2)"),
    "DTL": ("DTL", "target", ""), "ATL": ("ATL", "target", ""),
    "MTAL": ("TAL", "target", "medullary TAL"), "CTAL": ("TAL", "target", "cortical TAL"),
    "MD": ("MD", "target", ""),
    "DCT": ("DCT", "target", "the atlas does not split DCT1 / DCT2"),
    "CNT": ("CNT", "target", ""), "PC": ("PC", "target", ""),
    "ICA": ("IC-A", "target", ""), "ICB": ("IC-B", "target", ""),
    "Desc-Vasa-Recta": ("DVR", "target", ""), "Asc-Vasa-Recta": ("AVR", "target", ""),
    "Podo": ("Podo", "target", ""),
    "Endo": ("Endo", "target", "endothelial (other)"), "Glom-Endo": ("Endo", "target", "endothelial (other)"),
    "Vas-Afferens": ("Endo", "target", "endothelial (other)"), "Vas-Efferens": ("Endo", "target", "endothelial (other)"),
    "Fib": ("Fib", "target", "non-epithelial"), "Per": ("Per", "target", "non-epithelial"),
    "MC": ("MC", "target", "non-epithelial"), "PEC": ("PEC", "target", "parietal epithelial cells"),
    "T lymph": ("T lymph", "target", "immune"), "B lymph": ("B lymph", "target", "immune"),
    "NK": ("NK", "target", "immune"), "Macro": ("Macro", "target", "immune"), "DC": ("DC", "target", "immune"),
    "Neutro": ("Neutro", "target", "immune"),
    "DTL-ATL": (None, "mixed_excluded", "mixed class; left out of the marker derivation"),
    "LOH": (None, "mixed_excluded", "mixed class; left out of the marker derivation"),
    "DCT-CNT": (None, "mixed_excluded", "mixed class; left out of the marker derivation"),
    "CD-Trans": (None, "mixed_excluded", "mixed class; left out of the marker derivation"),
    "Unknown": (None, "excluded", "unlabelled in the atlas"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation step 3a: atlas-derived marker candidates.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "atlas_reference_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_annotation_atlas_reference_checks.csv.")


def main() -> None:
    import anndata as ad

    args = parse_args()
    run = cc.Run(s.TASK, "atlas_reference", args)
    t_start = time.time()
    checks = t5.Checks(s.DATASET)
    kid, _ = t5.load(run, s.DATASET)
    prot = kid.var_names.astype(str)
    prot_det = pd.Series((cc.dense(kid.layers["intensity"]) > 0).mean(axis=0), index=prot)
    v3 = t5.read_markers(run, s.DATASET, prot)
    base = {p.lower(): p for p in prot if "_" not in p and p != "Nan"}

    print("[atlas] reading mka.h5ad (about 4.6 GB in memory)", flush=True)
    a = ad.read_h5ad(run.use_input(ATLAS, "RNA atlas mka.h5ad (read-only)"))
    raw = a.raw.X.tocsr() if sparse.issparse(a.raw.X) else sparse.csr_matrix(a.raw.X)
    genes = np.asarray(a.raw.var_names.astype(str))
    sample_vals = raw.data[:: max(1, raw.data.size // 200000)]
    is_int = bool(np.all(np.equal(np.mod(sample_vals, 1), 0)) and sample_vals.min() > 0)
    checks.add(".raw.X holds positive integer counts (sample of non-zero values)",
               f"{sample_vals.size} values; integers {is_int}; min {sample_vals.min():g}", is_int)
    celltype = a.obs[LABEL_COL].astype(str).to_numpy()
    unmapped = sorted(set(celltype) - set(ATLAS_MAP))
    checks.add("every atlas Celltype is in the approved mapping", ", ".join(unmapped) or "none", not unmapped)
    if not checks.ok:
        stop(run, checks)

    shared_pos = np.array([i for i, g in enumerate(genes) if g.lower() in base])
    shared_prot = [base[genes[i].lower()] for i in shared_pos]
    checks.add("atlas genes matching a quantified protein (case-insensitive)", f"{len(shared_pos)} of {len(genes)} "
               f"genes; proteins {len(prot)}", len(shared_pos) > 0)
    if len(set(shared_prot)) != len(shared_prot):
        checks.add("one atlas gene per protein", "duplicates found", False)
        stop(run, checks)

    # Sample, normalize, slice -------------------------------------------------------------------
    rng = np.random.default_rng(s.SEED)
    rows, idx = [], []
    for ct in sorted(set(celltype)):
        scp, role, note = ATLAS_MAP[ct]
        cells = np.flatnonzero(celltype == ct)
        take = np.sort(rng.choice(cells, size=min(MAX_PER_CELLTYPE, len(cells)), replace=False)) if role == "target" \
            else np.array([], dtype=int)
        idx.append(take)
        rows.append({"atlas_celltype": ct, "n_cells": len(cells), "n_sampled": len(take), "scp_type": scp or "",
                     "role": role, "note": note, "approval": APPROVED})
    mapping = pd.DataFrame(rows).sort_values(["role", "scp_type", "atlas_celltype"]).reset_index(drop=True)
    idx = np.concatenate(idx)
    sub = raw[idx]
    totals = np.asarray(sub.sum(axis=1)).ravel()
    sub = sparse.diags(TARGET_SUM / totals) @ sub
    sub.data = np.log1p(sub.data)
    x = sub[:, shared_pos].toarray()
    group = np.array([ATLAS_MAP[c][0] for c in celltype[idx]])
    n_full = len(celltype)
    del a, raw, sub
    gc.collect()
    checks.add("sampled atlas cells (target classes, <= 2,000 per Celltype, seed 42)",
               f"{len(idx)} of {n_full}", len(idx) > 0)

    # Per-type statistics ------------------------------------------------------------------------
    print(f"[atlas] ranks for {x.shape[0]} cells x {x.shape[1]} genes", flush=True)
    det = x > 0
    cache = t5.rank_cache(x, det)
    types = cc.ordered_labels(set(group), (*s.ORDER, "DCT", "Endo"))
    stats_rows = []
    det_by_type = {}
    for t in types:
        m = group == t
        w = t5.wilcoxon(cache, m)
        mi, mo = x[m].mean(axis=0), x[~m].mean(axis=0)
        det_by_type[t] = det[m].mean(axis=0)
        stats_rows.append(pd.DataFrame({
            "scp_type": t, "n_atlas_cells_sampled": int(m.sum()), "gene": genes[shared_pos], "protein": shared_prot,
            "det_in": det_by_type[t], "det_out": det[~m].mean(axis=0), "mean_log_in": mi, "mean_log_out": mo,
            "log2fc": np.log2((np.expm1(mi) + 1e-9) / (np.expm1(mo) + 1e-9)), "auc": w["auc"], "p_value": w["p"]}))
    stats = pd.concat(stats_rows, ignore_index=True)
    det_tab = pd.DataFrame(det_by_type, index=shared_prot)
    stats["max_det_other_type"] = np.concatenate([det_tab.drop(columns=t).max(axis=1).to_numpy() for t in types])
    stats["passes"] = (stats["det_in"] >= MIN_DET_IN) & (stats["auc"] >= MIN_AUC) & (stats["log2fc"] >= MIN_LOG2FC)
    mito = pd.read_csv(run.use_input(QC_MITO_ANNOTATION, "qc_mito MitoCarta3.0 protein annotation (kidney)")).set_index("protein")
    stats["scp_detection_rate"] = stats["protein"].map(prot_det)
    stats["excluded_ribosomal"] = stats["protein"].str.match(RIBOSOMAL)
    stats["excluded_mitocarta"] = stats["protein"].map(mito["mitocarta"]).fillna(False).astype(bool)
    stats["excluded_scp_det_gt50"] = stats["scp_detection_rate"] > MAX_SCP_DETECTION
    stats["excluded"] = stats["excluded_ribosomal"] | stats["excluded_mitocarta"] | stats["excluded_scp_det_gt50"]
    stats["eligible"] = stats["passes"] & ~stats["excluded"]
    p = stats[stats["passes"]]
    checks.add("passing (type, gene) pairs removed by the exclusion filter: ribosomal; MitoCarta; SCP detection > 50%; "
               "any", f"{int(p['excluded_ribosomal'].sum())}; {int(p['excluded_mitocarta'].sum())}; "
               f"{int(p['excluded_scp_det_gt50'].sum())}; {int(p['excluded'].sum())} of {len(p)}", None)
    sel = (stats[stats["eligible"]].sort_values(["scp_type", "auc"], ascending=[True, False])
           .groupby("scp_type", sort=False).head(TOP_PER_TYPE).copy())
    n_types = sel.groupby("protein")["scp_type"].nunique()
    sel["n_types_selecting"] = sel["protein"].map(n_types)
    sel["shared_atlas"] = sel["n_types_selecting"] >= 2
    sel["rank_in_type"] = sel.groupby("scp_type")["auc"].rank(ascending=False, method="first").astype(int)
    sel["v3_listed_for"] = sel["protein"].map(lambda p: ";".join(sorted(v3.types_by_gene.get(p, set()))))
    sel = sel.sort_values(["scp_type", "rank_in_type"]).reset_index(drop=True)

    summ = []
    for t in types:
        st = stats[stats["scp_type"] == t]
        sl = sel[sel["scp_type"] == t]
        summ.append({"scp_type": t, "atlas_celltypes": ";".join(mapping.loc[mapping["scp_type"] == t, "atlas_celltype"]),
                     "n_atlas_cells": int(mapping.loc[mapping["scp_type"] == t, "n_cells"].sum()),
                     "n_sampled": int(st["n_atlas_cells_sampled"].iloc[0]), "n_genes_passing": int(st["passes"].sum()),
                     "n_passing_excluded": int((st["passes"] & st["excluded"]).sum()),
                     "n_genes_eligible": int(st["eligible"].sum()),
                     "n_selected": len(sl), "n_selected_type_specific": int((~sl["shared_atlas"]).sum()),
                     "selected": ";".join(sl["protein"]),
                     "selected_also_in_v3": ";".join(sl.loc[sl["v3_listed_for"] != "", "protein"])})
    summ = pd.DataFrame(summ)
    for r in summ.itertuples(index=False):
        checks.add(f"{r.scp_type}: atlas markers selected (type-specific)", f"{r.n_selected} ({r.n_selected_type_specific}); "
                   f"{r.n_atlas_cells} atlas cells", None)

    keep = stats["passes"] | (stats["auc"] >= 0.6) | stats["protein"].isin(v3.types_by_gene)
    run.table(checks.frame(), "atlas_reference_checks")
    run.table(mapping, "atlas_celltype_mapping")
    run.table(summ, "atlas_marker_summary")
    run.table(sel, "atlas_markers")
    run.table(stats[keep].reset_index(drop=True), "atlas_marker_stats")
    run.finish({"atlas_label": LABEL_COL, "max_per_celltype": MAX_PER_CELLTYPE, "target_sum": TARGET_SUM,
                "criteria": {"min_det_in": MIN_DET_IN, "min_auc": MIN_AUC, "min_log2fc": MIN_LOG2FC,
                             "top_per_type": TOP_PER_TYPE},
                "exclusion_filter": {"ribosomal_regex": RIBOSOMAL, "mitocarta": "qc_mito kidney protein annotation",
                                     "max_scp_detection_rate": MAX_SCP_DETECTION,
                                     "decided": "authors, 14 Oct 2026, after run 1's marker table, before any annotation"},
                "normalization": "normalize_total 10,000 then log1p on .raw.X (the earlier analysis integrate_scp_with_mka.py)",
                "auc": "cluster_de_lib.wilcoxon (tie-corrected rank-sum), type vs all other sampled cells",
                "stats_table_rows": "passes, or AUC >= 0.6, or V3-listed", "mapping_approval": APPROVED,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
