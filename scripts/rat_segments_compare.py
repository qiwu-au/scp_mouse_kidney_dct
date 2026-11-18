#!/usr/bin/env python
"""rat_segments step 5: Limbutara (rat protein) vs pt_tal (mouse RNA atlas) panels, directions, scores, S2, literature.

Task rat_segments (authors, 30 Oct 2026).

What it does
------------
1. Panel overlap: each Limbutara panel against its pt_tal counterpart(s) (S1 - PTS1; S2 - PTS2; S3 - PTS3 and S3;
   S1/S2 - S1/S2; cTAL - CTAL; mTAL - MTAL), per label set and threshold: sizes, shared proteins, Jaccard.
2. Direction agreement both ways: atlas log2 FC (pt_tal atlas DE, matching contrast) of the Limbutara panel proteins;
   Limbutara log2 FC (difference of the group and reference means, centred values; contrast tested) of the pt_tal panel
   proteins (PTS3T2 against Limbutara S3).
3. Reference vs reference: Spearman rho of the Limbutara log2 FC (difference of means) with the atlas log2 FC and
   with the atlas difference of mean log-normalized expression, per matching contrast, over the genes in the pt_tal
   atlas DE table whose protein has a tested KTEA match.
4. Per-cell agreement of the Limbutara and pt_tal S1/S2 - S3 scores (mean z, all panel proteins): Spearman rho over
   the PT union cells and over the PCT + PST cells of each label set.
5. S2: Limbutara S2 panel sizes along the rule, and the atlas PTS2 contrast for the S2 panel proteins.
6. The 12 pt_tal literature candidates (plan list): Limbutara values (rat symbol lookup, independent of the kidney
   matrix), log2 FC in the matching contrast, replicate separation, direction against the plan's group; atlas
   direction from pt_tal.

Inputs (read-only): rat_segments panels, rat_contrasts, coverage, panel_funnel, scores_per_cell; pt_tal panels, atlas_de,
                    literature, scores_per_cell.
Outputs: results/tables/rat_segments/rat_segments_compare_panel_overlap.csv, rat_segments_compare_direction.csv,
         rat_segments_compare_direction_summary.csv, rat_segments_compare_reference_rho.csv,
         rat_segments_compare_scores.csv, rat_segments_compare_s2.csv, rat_segments_compare_literature.csv,
         rat_segments_compare_checks.csv, rat_segments_step_compare_manifest.json

Command (project root):  python scripts/rat_segments_compare.py
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
import pt_tal_lib as T9
import rat_segments_lib as T

# pt_tal panel -> Limbutara contrast used for its direction (PTS3T2 has no Limbutara counterpart; S3 is the closest)
PT_TAL_TO_LIM = {"PTS1": "S1", "PTS2": "S2", "PTS3": "S3", "PTS3T2": "S3", "S1/S2": "S1/S2", "S3": "S3",
              "CTAL": "cTAL", "MTAL": "mTAL"}
# plan group of the literature candidates -> Limbutara contrast (MD: cTAL vs mTAL, descriptive; MD lies in cTAL)
LIT_CONTRAST = {"S1": "S1", "S1/S2": "S1/S2", "S3": "S3", "TAL: Cldn10": "cTAL", "TAL: Cldn16/Cldn19": "cTAL",
                "MD": "cTAL"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rat_segments step 5: comparison with pt_tal.")
    cc.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "compare", args)
    checks = t5.Checks(T.DATASET)
    panels = T.read_table(run, "panels")
    rc = T.read_table(run, "rat_contrasts", dtype={"rat_id": str, "mouse_protein": str}, keep_default_na=False,
                      na_values=[""])
    rc["mouse_protein"] = rc["mouse_protein"].fillna("")
    funnel = T.read_table(run, "panel_funnel")
    pc24 = T.read_table(run, "scores_per_cell", dtype={"cell_id": str}).set_index("cell_id")
    t09 = T9.read_table(run, "panels")
    de = T9.read_table(run, "atlas_de")
    lit = T9.read_table(run, "literature")
    pc09 = T9.read_table(run, "scores_per_cell", dtype={"cell_id": str}).set_index("cell_id").reindex(pc24.index)
    by_prot = {c: rc[rc["mouse_protein"].ne("") & rc["contrast"].eq(c)].set_index("mouse_protein")
               for c in T.CONTRASTS}

    # 1. overlap ------------------------------------------------------------------------------------------------------
    ov = []
    for c in T9.LABELS:
        for thr in T.THRESHOLDS:
            for lp, t09ps in T.PT_TAL_PANELS.items():
                a = set(panels[(panels["label_column"] == c) & (panels["threshold"] == thr) & (panels["panel"] == lp)]["protein"])
                for tp in t09ps:
                    b = set(t09[(t09["label_column"] == c) & (t09["panel"] == tp)]["protein"])
                    sh = sorted(a & b)
                    ov.append({"label_column": c, "threshold": thr, "limbutara_panel": lp, "t09_panel": tp,
                               "n_limbutara": len(a), "n_t09": len(b), "n_shared": len(sh),
                               "jaccard": len(sh) / len(a | b) if a | b else np.nan, "shared": ", ".join(sh)})
    ov = pd.DataFrame(ov)
    for r in ov[(ov["threshold"] == T.PRIMARY_THR)].itertuples():
        checks.add(f"overlap ({T9.LABEL_SHORT[r.label_column]}, fc2) Limbutara {r.limbutara_panel} vs pt_tal {r.t09_panel}",
                   f"{r.n_shared} of {r.n_limbutara} / {r.n_t09}: {r.shared}", None)

    # 2. directions ---------------------------------------------------------------------------------------------------
    drows = []
    for r in panels.itertuples():
        ac = T.PT_TAL_CONTRAST[r.panel]
        d = de[(de["contrast"] == ac) & (de["protein"] == r.protein)]
        rec = {"direction": "Limbutara panel -> atlas", "label_column": r.label_column, "threshold": r.threshold,
               "panel": r.panel, "other_contrast": ac, "protein": r.protein, "in_other": not d.empty}
        if not d.empty:
            d = d.iloc[0]
            rec.update(other_log2fc=float(d.log2fc), other_mean_difference=float(d.mean_difference), other_q=float(d.q_bh),
                       other_batches_same_sign=f"{d.n_batches_same_sign} of {d.n_batches_eligible}",
                       agree=bool(np.sign(d.mean_difference) > 0))
        drows.append(rec)
    for r in t09.itertuples():
        lc = PT_TAL_TO_LIM.get(r.panel)
        if lc is None:
            continue
        d = by_prot[lc].loc[[r.protein]] if r.protein in by_prot[lc].index else None
        rec = {"direction": "pt_tal panel -> Limbutara", "label_column": r.label_column, "threshold": "-", "panel": r.panel,
               "other_contrast": lc, "protein": r.protein, "in_other": d is not None and bool(d["quantified_all"].iloc[0])}
        if rec["in_other"]:
            d = d.iloc[0]
            rec.update(other_log2fc=float(d.log2fc_mean), other_log2fc_min_max=float(d.log2fc_min_max),
                       other_samples_separated=bool(d.samples_separated), agree=bool(d.log2fc_mean > 0))
        drows.append(rec)
    dirs = pd.DataFrame(drows)
    dsum = (dirs[dirs["in_other"]].groupby(["direction", "label_column", "threshold", "panel", "other_contrast"])
            .agg(n_panel_in_other=("protein", "size"), n_agree=("agree", lambda x: int(x.astype(float).sum())))
            .reset_index())
    tot = dirs.groupby(["direction", "label_column", "threshold", "panel", "other_contrast"]).size().rename("n_panel")
    dsum = dsum.merge(tot.reset_index(), how="right").fillna({"n_panel_in_other": 0, "n_agree": 0})
    dsum["frac_agree"] = dsum["n_agree"] / dsum["n_panel_in_other"].where(dsum["n_panel_in_other"] > 0)
    for r in dsum[dsum["threshold"].isin([T.PRIMARY_THR, "-"]) & dsum["label_column"].eq(T9.PRIMARY)].itertuples():
        checks.add(f"direction ({r.direction}, kidney_annotation, {r.panel} -> {r.other_contrast}): in other / same direction",
                   f"{int(r.n_panel_in_other)} of {int(r.n_panel)} / {int(r.n_agree)}", None)

    # 3. reference vs reference ----------------------------------------------------------------------------------------
    rr = []
    for lc, ac in T.PT_TAL_CONTRAST.items():
        a = de[(de["contrast"] == ac) & de["quantified_protein"]].drop_duplicates("protein").set_index("protein")
        b = by_prot[lc][by_prot[lc]["quantified_all"]]
        shared = a.index.intersection(b.index)
        rec = {"limbutara_contrast": lc, "atlas_contrast": ac, "n_genes": len(shared)}
        for col in ("log2fc", "mean_difference"):
            r = stats.spearmanr(b.loc[shared, "log2fc_mean"], a.loc[shared, col])
            rec[f"rho_atlas_{col}"] = float(r.statistic)
            rec[f"p_atlas_{col}"] = float(r.pvalue)
        rec["frac_same_sign_mean_difference"] = float((np.sign(b.loc[shared, "log2fc_mean"])
                                                       == np.sign(a.loc[shared, "mean_difference"])).mean())
        rr.append(rec)
    rr = pd.DataFrame(rr)
    for r in rr.itertuples():
        checks.add(f"reference vs reference: Limbutara {r.limbutara_contrast} vs atlas {r.atlas_contrast}: genes; rho "
                   "(atlas log2 FC; mean difference); same sign", f"{r.n_genes}; {r.rho_atlas_log2fc:+.2f}; "
                   f"{r.rho_atlas_mean_difference:+.2f}; {r.frac_same_sign_mean_difference:.2f}", r.n_genes > 100)

    # 4. per-cell score agreement --------------------------------------------------------------------------------------
    sc_rows = []
    for c in T9.LABELS:
        s = T9.LABEL_SHORT[c]
        a = pc24[T.score_col(s, T.PRIMARY_THR, "S1/S2 - S3", "all", "mean_z")]
        b = pc09[f"score__{s}__S1S2_minus_S3__all__mean_z"]
        for cells, m in (("PT union cells", pc24["in_pt_union"].astype(bool)),
                         (f"PCT + PST cells ({T9.LABEL_DISPLAY[c]})", pc24[c].isin(T9.PT_LABELS))):
            r = stats.spearmanr(a[m], b[m])
            sc_rows.append({"panels_from": c, "cells": cells, "n_cells": int(m.sum()), "rho": float(r.statistic),
                            "p": float(r.pvalue)})
    scs = pd.DataFrame(sc_rows)
    for r in scs.itertuples():
        checks.add(f"per-cell S1/S2 - S3, Limbutara (fc2) vs pt_tal ({T9.LABEL_SHORT[r.panels_from]} panels), {r.cells}",
                   f"rho {r.rho:+.2f} (n {r.n_cells})", None)

    # 5. S2 ---------------------------------------------------------------------------------------------------------
    s2 = panels[panels["panel"] == "S2"][["label_column", "threshold", "rank", "protein", "rat_gene", "log2fc_min_max",
                                          "log2fc_mean", "scp_detection"]].copy()
    a2 = de[de["contrast"] == "PTS2"].drop_duplicates("protein").set_index("protein")
    for col in ("log2fc", "mean_difference", "q_bh", "frac_batches_same_sign", "passes_rna_filters"):
        s2[f"atlas_pts2_{col}"] = s2["protein"].map(a2[col])
    s2["in_atlas"] = s2["protein"].isin(a2.index)
    f2 = funnel[funnel["panel"] == "S2"]
    for r in f2.itertuples():
        checks.add(f"S2 panel ({T9.LABEL_SHORT[r.label_column]}, {r.threshold}): tested / +abund / +FC / +separated / "
                   "matched / +SCP / panel", f"{r.tested} / {r.plus_abundance} / {r.plus_fc} / {r.plus_separated} / "
                   f"{r.plus_matched} / {r.plus_scp_10pct} / {r.panel_size}", None)
    for (c, thr), d in s2.groupby(["label_column", "threshold"]):
        ina = d[d["in_atlas"]]
        checks.add(f"S2 panel ({T9.LABEL_SHORT[c]}, {thr}) proteins in the atlas; atlas PTS2 mean difference > 0; "
                   "passing the pt_tal RNA filters", f"{len(ina)} of {len(d)}; {int((ina['atlas_pts2_mean_difference'] > 0).sum())}; "
                   f"{int(ina['atlas_pts2_passes_rna_filters'].astype(bool).sum())}", None)

    # 6. literature --------------------------------------------------------------------------------------------------
    stc = T.read_table(run, "ktea_segment_stats_centred", dtype={"rat_id": str}).set_index("rat_id")
    sym = rc.drop_duplicates(["rat_id"])[["rat_id", "rat_gene"]].copy()
    sym["symbols"] = sym["rat_gene"].map(lambda g: [x.lower() for x in T.rat_symbols(g)])
    lrows = []
    for r in lit.itertuples():
        hits = sym[sym["symbols"].map(lambda xs: r.gene.lower() in xs)]
        lc = LIT_CONTRAST[r.brief_group]
        rec = {"gene": r.gene, "brief_group": r.brief_group, "limbutara_contrast": lc,
               "n_ktea_entries": len(hits), "quantified_protein_scp": r.quantified_protein,
               "atlas_direction_t09": r.atlas_direction_observed, "atlas_contrast_t09": r.atlas_contrast_checked}
        if len(hits):
            rid = hits["rat_id"].iloc[0]
            d = rc[(rc["rat_id"] == rid) & (rc["contrast"] == lc)].iloc[0]
            pt = rc[(rc["rat_id"] == rid) & (rc["contrast"] == "S1")].iloc[0]
            rec.update(rat_id=rid, rat_gene=hits["rat_gene"].iloc[0],
                       **{f"log2_{x}": stc.loc[rid, f"log2_{x}"] for x in T.STUDY_SEGMENTS},
                       **{f"n_quant_{x}": f"{int(stc.loc[rid, f'n_quant_{x}'])} of {int(stc.loc[rid, f'n_samples_{x}'])}"
                          for x in T.STUDY_SEGMENTS},
                       quantified_all=bool(d.quantified_all), quantified_group=bool(d.quantified_group),
                       log2fc_mean=float(d.log2fc_mean), log2fc_min_max=float(d.log2fc_min_max),
                       samples_separated=bool(d.samples_separated))
            if r.brief_group in ("TAL: Cldn10", "TAL: Cldn16/Cldn19", "MD"):
                rec["limbutara_direction"] = ("not quantified in both cTAL and mTAL" if not d.quantified_all else
                                              "higher in cTAL" if d.log2fc_mean > 0 else "higher in mTAL")
            else:
                rec["limbutara_direction"] = ("not quantified in every contrast segment" if not d.quantified_all else
                                              "higher in the plan's group" if d.log2fc_mean > 0 else
                                              "lower in the plan's group")
            for c in T9.LABELS:
                inp = panels[(panels["label_column"] == c) & (panels["threshold"] == T.PRIMARY_THR)
                             & (panels["rat_id"].astype(str) == rid)]
                rec[f"in_panels_fc2_{T9.LABEL_SHORT[c]}"] = "; ".join(f"{p} (rank {k})" for p, k in zip(inp["panel"], inp["rank"]))
        else:
            rec["limbutara_direction"] = "not in KTEA"
        lrows.append(rec)
    lit24 = pd.DataFrame(lrows)
    for r in lit24.itertuples():
        checks.add(f"literature {r.gene} ({r.brief_group}): Limbutara {r.limbutara_contrast} log2 FC (means); separated; "
                   "direction; atlas (pt_tal)",
                   f"{getattr(r, 'log2fc_mean', np.nan):+.2f}; {getattr(r, 'samples_separated', '')}; "
                   f"{r.limbutara_direction}; {r.atlas_direction_t09}", None)
    checks.add("literature candidates looked up", len(lit24), len(lit24) == len(T9.LITERATURE))

    run.table(ov, "compare_panel_overlap")
    run.table(dirs, "compare_direction")
    run.table(dsum, "compare_direction_summary")
    run.table(rr, "compare_reference_rho")
    run.table(scs, "compare_scores")
    run.table(s2, "compare_s2")
    run.table(lit24, "compare_literature")
    run.table(checks.frame(), "compare_checks")
    run.finish({"t09_to_limbutara": PT_TAL_TO_LIM, "literature_contrast": LIT_CONTRAST, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see rat_segments_compare_checks.csv.")


if __name__ == "__main__":
    main()
