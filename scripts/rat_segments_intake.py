#!/usr/bin/env python
"""rat_segments steps 0 and 2: Limbutara et al. 2020 intake; rat -> mouse matching; coverage.

Task rat_segments (authors, 30 Oct 2026): Limbutara et al. 2020 rat
tubule-segment proteome as a protein-level comparator for PT and TAL heterogeneity, next to pt_tal.

What it does
------------
0. Records the two KTEA files (``data/reference/KTEA/``; source and SHA256 in SOURCE.md) and checks
   their SHA256, size (7,429 protein groups) and samples (14 segments; 3 samples each, 4 for mTAL and CCD).
   Per protein and segment: samples quantified (value > 0), quantified flag (>= 2/3 of the samples), log2 mean and
   SD of the quantified samples. Cross-check against the KTEA geometric-mean file.
2. Matches the KTEA gene symbols to the kidney protein names (identical symbols, case-insensitive; then a gene of
   the kidney protein group; no ortholog table, authors 30 Oct 2026). Several KTEA entries on one mouse protein: the
   entry with the highest mean log2 abundance over S1, S2, S3, cTAL, mTAL is kept. Every KTEA entry is listed in the
   coverage table with its match or the reason it is unmatched.

Inputs (read-only): KTEA files and SOURCE.md; kidney_object canonical kidney object and labels; kidney_annotation labels.
Outputs: results/tables/rat_segments/rat_segments_ktea_segment_stats.csv, rat_segments_ktea_mean_crosscheck.csv,
         rat_segments_rat_mouse_matches.csv, rat_segments_limbutara_mouse.csv, rat_segments_coverage.csv,
         rat_segments_coverage_summary.csv, rat_segments_intake_checks.csv, rat_segments_step_intake_manifest.json

Command (project root):  python scripts/rat_segments_intake.py
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
import pt_tal_lib as T9
import rat_segments_lib as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rat_segments steps 0 and 2: KTEA intake, rat -> mouse matching, coverage.")
    cc.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "intake", args)
    checks = t5.Checks(T.DATASET)

    # step 0: files ----------------------------------------------------------------------------------------------
    sha = T.check_sources(run)
    for name, h in sha.items():
        checks.add(f"SHA256 of {name} equals SOURCE.md", h[:16] + "...", h == T.EXPECTED_SHA256[name])
    ann, vals, seg = T.read_samples()
    means = T.read_means()
    checks.add("KTEA per-sample file: protein groups", len(ann), len(ann) == T.EXPECTED_N_PROTEINS)
    checks.add("KTEA per-sample file: unique 'Majority protein IDs'", int(ann.index.is_unique), ann.index.is_unique)
    n_by_seg = seg.value_counts().to_dict()
    checks.add("KTEA samples per segment (14 segments; 3 each, 4 for mTAL and CCD)",
               "; ".join(f"{s} {n_by_seg.get(s, 0)}" for s in T.SEGMENTS),
               n_by_seg == T.EXPECTED_SAMPLES)
    checks.add("KTEA mean file: same protein groups as the per-sample file", len(means),
               set(means.index) == set(ann.index) and len(means) == len(ann))
    checks.add("KTEA mean file: segment columns", ", ".join(c for c in means.columns if c in T.SEGMENTS),
               all(s in means.columns for s in T.SEGMENTS))
    means = means.reindex(ann.index)
    same_gene = (means[T.GENE_COL].astype(str) == ann[T.GENE_COL].astype(str))
    checks.add("KTEA mean file vs per-sample file: identical 'gene' entries (after matching on IDs)",
               f"{int(same_gene.sum())} of {len(ann)}", None)
    checks.add("KTEA per-sample values: all finite and >= 0 (read_samples raises otherwise); values > 0",
               int(vals.notna().sum().sum()), True)

    st = T.segment_stats(vals, seg)
    # cross-check of the KTEA mean file. Observed (format check before this step): KTEA mean = 2^(mean of
    # log2(x + 1) over all samples, zeros included) - 1. rat_segments itself uses the log2 mean of the quantified samples only
    # (zeros = not quantified; approved rule); both are given per segment for proteins quantified in some samples.
    cross = []
    for s in T.SEGMENTS:
        k = means[s].astype(float)
        sc_ = seg.index[seg == s]
        n = len(sc_)
        nq = st[f"n_quant_{s}"]
        ktea_formula = 2 ** np.log2(vals[sc_].fillna(0) + 1).mean(axis=1) - 1
        rel = (ktea_formula / k.where(k > 0) - 1).abs()
        gm_q = 2 ** st[f"log2_{s}"]
        full, part = nq == n, (nq > 0) & (nq < n)
        cross.append({"segment": s, "n_samples": n,
                      "max_rel_diff_ktea_formula": float(np.nanmax(rel.to_numpy())),
                      "n_ktea_0": int((k == 0).sum()), "n_none_quantified": int((nq == 0).sum()),
                      "n_all_quantified": int(full.sum()), "n_some_quantified": int(part.sum()),
                      "median_ratio_t24_gm_quantified_to_ktea_some_quantified":
                          float((gm_q[part] / k[part]).median()) if part.any() else np.nan,
                      "max_rel_diff_t24_to_ktea_all_quantified": float((gm_q[full] / k[full] - 1).abs().max())})
    cross = pd.DataFrame(cross)
    checks.add("KTEA mean file = 2^(mean log2(x + 1) over all samples) - 1 (max relative difference, values > 0)",
               f"{cross['max_rel_diff_ktea_formula'].max():.2e}", bool(cross["max_rel_diff_ktea_formula"].max() < 1e-6))
    checks.add("KTEA mean 0 exactly when no sample of the segment is quantified",
               f"{int(cross['n_ktea_0'].sum())} vs {int(cross['n_none_quantified'].sum())}",
               bool((cross["n_ktea_0"] == cross["n_none_quantified"]).all()))
    checks.add("rat_segments log2 mean of the quantified samples vs KTEA mean, all samples quantified (max relative "
               "difference; the +1 offset)", f"{cross['max_rel_diff_t24_to_ktea_all_quantified'].max():.2e}", None)
    for s in T.STUDY_SEGMENTS:
        checks.add(f"KTEA {s}: proteins quantified (>= 2/3 samples) / in >= 1 sample",
                   f"{int(st[f'quant_{s}'].sum())} / {int((st[f'n_quant_{s}'] > 0).sum())}", None)

    seg_tab = pd.concat([ann[[T.GENE_COL, T.NAME_COL]].rename(columns={T.GENE_COL: "rat_gene",
                                                                       T.NAME_COL: "rat_protein_names"}),
                         st, means[list(T.SEGMENTS)].add_prefix("ktea_mean_")], axis=1)
    seg_tab.index.name = "rat_id"

    # step 2: rat -> mouse -----------------------------------------------------------------------------------------
    adata, labels, xz, ln1p, det = T9.load_scp(run, checks)
    names = adata.var_names.astype(str)
    matches = T.match_to_mouse(ann, names, adata.var["pg_genes"].astype(str).to_numpy())
    abundance = st[[f"log2_{s}" for s in T.STUDY_SEGMENTS]].mean(axis=1, skipna=True)
    matches = T.resolve_many_to_one(matches, abundance)
    kept = matches[matches["kept"]]
    checks.add("kept matches: each mouse protein once", f"{kept['mouse_protein'].nunique()} proteins, {len(kept)} rows",
               kept["mouse_protein"].is_unique)
    checks.add("every KTEA entry listed in the match table", f"{matches['rat_id'].nunique()} of {len(ann)}",
               matches["rat_id"].nunique() == len(ann))
    checks.add("kept mouse proteins are kidney protein names", f"{int(kept['mouse_protein'].isin(names).sum())}",
               bool(kept["mouse_protein"].isin(names).all()))

    masks = T9.union_masks(labels)
    n_det = det.sum(axis=0)
    pt_det = {c: det[T9.segment_cells(labels, c, "PT").to_numpy()].mean(axis=0) for c in T9.LABELS}
    tal_det = {c: det[T9.segment_cells(labels, c, "TAL").to_numpy()].mean(axis=0) for c in T9.LABELS}

    lm = kept.set_index("mouse_protein")[["rat_id", "rat_gene", "rat_symbols", "match", "n_mouse_hits",
                                          "many_to_one"]].copy()
    lm["one_to_many"] = lm["n_mouse_hits"] > 1
    lm = lm.join(seg_tab.drop(columns=["rat_gene"]), on="rat_id")
    lm["scp_n_cells_detected_712"] = n_det.reindex(lm.index).to_numpy()
    for c in T9.LABELS:
        s = T9.LABEL_SHORT[c]
        lm[f"scp_det_pt_{s}"] = pt_det[c].reindex(lm.index).to_numpy()
        lm[f"scp_det_tal_{s}"] = tal_det[c].reindex(lm.index).to_numpy()
    lm.index.name = "protein"

    # coverage: one row per KTEA entry
    status = matches.groupby("rat_id").agg(
        mouse_proteins_kept=("mouse_protein", lambda x: ";".join(p for p, k in zip(x, matches.loc[x.index, "kept"]) if k)),
        match=("match", lambda x: ";".join(sorted(set(x)))),
        reason=("reason", lambda x: "; ".join(sorted({r for r in x if r}))))
    cov = seg_tab[["rat_gene"]].join(status)
    cov["status"] = np.where(cov["mouse_proteins_kept"].ne(""), "matched",
                             np.where(cov["match"].str.contains("unmatched"), "unmatched", "dropped"))
    cov["quant_any_pt"] = st[[f"quant_{s}" for s in T.PT_SEGMENTS]].any(axis=1)
    cov["quant_any_tal"] = st[[f"quant_{s}" for s in T.TAL_SEGMENTS]].any(axis=1)
    cov["quant_any_study_segment"] = cov["quant_any_pt"] | cov["quant_any_tal"]
    first = cov["mouse_proteins_kept"].str.split(";").str[0]
    cov["scp_n_cells_detected_712"] = first.map(n_det).where(cov["status"].eq("matched"))
    for c in T9.LABELS:
        s = T9.LABEL_SHORT[c]
        cov[f"scp_det_pt_{s}"] = first.map(pt_det[c]).where(cov["status"].eq("matched"))
        cov[f"scp_det_tal_{s}"] = first.map(tal_det[c]).where(cov["status"].eq("matched"))

    # summary
    srows = [("KTEA protein groups", len(cov)),
             ("quantified (>= 2/3 samples) in >= 1 of S1, S2, S3, cTAL, mTAL", int(cov["quant_any_study_segment"].sum())),
             ("matched to a kidney protein (kept)", int(cov["status"].eq("matched").sum())),
             ("  of which by protein name", int(kept["match"].eq("protein_name").groupby(kept["rat_id"]).any().sum())),
             ("  of which by a protein-group gene", int(kept["match"].eq("protein_group_gene").groupby(kept["rat_id"]).all().sum())),
             ("  of which on several mouse proteins (one-to-many)", int(kept.groupby("rat_id").size().gt(1).sum())),
             ("dropped (another KTEA entry kept for the same mouse protein, or name match took precedence)",
              int(cov["status"].eq("dropped").sum())),
             ("unmatched: no gene symbol in KTEA (UniProt ID only)",
              int((cov["status"].eq("unmatched") & cov["reason"].str.contains("UniProt")).sum())),
             ("unmatched: no identical mouse symbol among the kidney proteins",
              int((cov["status"].eq("unmatched") & cov["reason"].str.contains("identical")).sum())),
             ("matched and quantified in >= 1 study segment",
              int((cov["status"].eq("matched") & cov["quant_any_study_segment"]).sum())),
             ("kidney proteins (matrix)", len(names)),
             ("kidney proteins with a KTEA match", len(lm)),
             ("kidney proteins with a KTEA match quantified in S1, S2 and S3",
              int(lm[[f"quant_{s}" for s in T.PT_SEGMENTS]].all(axis=1).sum())),
             ("kidney proteins with a KTEA match quantified in cTAL and mTAL",
              int(lm[[f"quant_{s}" for s in T.TAL_SEGMENTS]].all(axis=1).sum()))]
    for c in T9.LABELS:
        s = T9.LABEL_SHORT[c]
        pt10 = pt_det[c] >= T.SCP_DET_MIN
        tal10 = tal_det[c] >= T.SCP_DET_MIN
        srows += [(f"kidney proteins detected in >= 10% of PT cells ({T9.LABEL_DISPLAY[c]})", int(pt10.sum())),
                  ("  of which with a KTEA match quantified in S1, S2 and S3",
                   int(pt10[pt10].index.isin(lm.index[lm[[f"quant_{x}" for x in T.PT_SEGMENTS]].all(axis=1)]).sum())),
                  (f"kidney proteins detected in >= 10% of TAL cells ({T9.LABEL_DISPLAY[c]})", int(tal10.sum())),
                  ("  of which with a KTEA match quantified in cTAL and mTAL",
                   int(tal10[tal10].index.isin(lm.index[lm[[f"quant_{x}" for x in T.TAL_SEGMENTS]].all(axis=1)]).sum()))]
    summary = pd.DataFrame(srows, columns=["item", "n"])
    for item, n in srows:
        checks.add(f"coverage: {item.strip()}", n, None)
    n_kept_entries = int(cov["status"].eq("matched").sum())
    checks.add("coverage table: matched + dropped + unmatched = KTEA entries",
               f"{n_kept_entries} + {int(cov['status'].eq('dropped').sum())} + {int(cov['status'].eq('unmatched').sum())}",
               int(cov["status"].isin(["matched", "dropped", "unmatched"]).sum()) == len(cov))
    checks.add("PT / TAL union subsets (pt_tal)", f"{int(masks['PT'].sum())} / {int(masks['TAL'].sum())}",
               int(masks["PT"].sum()) == T9.EXPECTED_UNION["PT"] and int(masks["TAL"].sum()) == T9.EXPECTED_UNION["TAL"])

    run.table(seg_tab.reset_index(), "ktea_segment_stats")
    run.table(cross, "ktea_mean_crosscheck")
    run.table(matches, "rat_mouse_matches")
    run.table(lm.reset_index(), "limbutara_mouse")
    run.table(cov.reset_index(), "coverage")
    run.table(summary, "coverage_summary")
    run.table(checks.frame(), "intake_checks")
    run.finish({"quant_frac": T.QUANT_FRAC, "matching": "identical symbol (case-insensitive) to the kidney protein "
                "name, then to a gene of the kidney protein group; no ortholog table (authors, 30 Oct 2026)",
                "many_to_one": "highest mean log2 over S1, S2, S3, cTAL, mTAL kept", "source": T.SOURCE_TEXT,
                "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see rat_segments_intake_checks.csv.")


if __name__ == "__main__":
    main()
