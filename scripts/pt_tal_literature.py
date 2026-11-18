#!/usr/bin/env python
"""pt_tal step 2: literature candidates (the plan's examples) in the protein matrix and in the atlas.

Analysis plan, step 2.
Candidates = exactly the examples listed in the plan (authors, 15 Oct 2026; no web lookup, Limbutara et al. 2020 not
added): Slc5a2, Slc5a12 (S1); Slc22a6, Slc22a8 (S1/S2); Slc7a13, Atp11a, Slc22a7 (S3); Cldn10 vs Cldn16 / Cldn19
(TAL); Nos1, Ptgs2 (MD). They are candidates to check, not ground truth.

Per candidate
-------------
- Protein matrix: quantified or not; cells detected (all 712); detection and mean ln(1 + intensity) per SCP group
  (kidney_annotation and Fig 2A PCT, PST, TAL; Fig 2A MD; union subsets).
- Atlas (step-1 tables, atlas Celltype, all cells and nuclei): gene present or not; mean log-normalized expression and
  detection per class; the class with the highest mean within the segment; the step-1 contrast of the plan's group
  (S1 -> PTS1 vs other PT; S1/S2 -> PTS1 + PTS2 vs PTS3 + PTS3T2; S3 -> PTS3 + PTS3T2 vs PTS1 + PTS2; MD -> MD vs
  CTAL + MTAL): log2 fold change, q, batch signs, whether it passes the panel RNA filters; observed direction.
  TAL claudins: the plan assigns no CTAL / MTAL direction; the CTAL vs MTAL contrast is reported and whether Cldn10
  and Cldn16 / Cldn19 differ in sign.
- SCP direction (PT candidates): PCT vs PST on ln(1 + intensity) (AUC, detection difference) for kidney_annotation labels, Fig 2A
  labels and cells with the same PT label in both sets; MD candidates: Fig 2A MD vs Fig 2A TAL detection.

Inputs (read-only): kidney_object canonical kidney object and labels; kidney_annotation labels; pt_tal step-1 tables (atlas_de, atlas_class_stats,
                    panels).
Outputs: results/tables/pt_tal/pt_tal_literature.csv, pt_tal_literature_scp_groups.csv,
         pt_tal_literature_atlas_classes.csv, pt_tal_literature_checks.csv

Command (project root):  python scripts/pt_tal_literature.py
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
import pt_tal_lib as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pt_tal step 2: literature candidates (plan list).")
    cc.add_common_args(parser)
    return parser.parse_args()


def direction(log2fc: float, q: float) -> str:
    if not np.isfinite(log2fc) or not np.isfinite(q):
        return "not tested"
    if q >= T.Q_MAX:
        return f"no significant difference (q >= {T.Q_MAX})"
    return "higher in group" if log2fc > 0 else "lower in group"


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "literature", args)
    checks = t5.Checks(T.DATASET)
    adata, labels, _, ln1p, det = T.load_scp(run, checks)
    de = T.read_table(run, "atlas_de")
    cs = T.read_table(run, "atlas_class_stats")
    panels = T.read_table(run, "panels")
    names = set(adata.var_names.astype(str))
    masks = T.union_masks(labels)
    groups = {}
    for col in T.LABELS:
        s = T.LABEL_SHORT[col]
        for g in (*T.PT_LABELS, T.TAL_LABEL, T.MD_LABEL):
            m = labels[col].eq(g)
            if m.any():
                groups[f"{s}: {g}"] = m
    groups["union PT subset"] = masks["PT"]
    groups["union TAL subset"] = masks["TAL"]
    groupings = T.pt_groupings(labels)

    rows, grp_rows, cls_rows = [], [], []
    all_cs = cs[cs["rna_source"] == "all"]
    for gene, brief_group, contrast in T.LITERATURE:
        quant = gene in names
        sub = all_cs[all_cs["atlas_gene"].str.lower() == gene.lower()]
        in_atlas = len(sub) > 0
        atlas_gene = sub["atlas_gene"].iloc[0] if in_atlas else ""
        seg = "PT" if brief_group.startswith("S") else "TAL"
        rec = {"gene": gene, "brief_group": brief_group, "segment": seg, "quantified_protein": quant,
               "n_cells_detected_all_712": int(det[gene].sum()) if quant else 0,
               "detection_all_712": float(det[gene].mean()) if quant else np.nan,
               "in_atlas": in_atlas, "atlas_gene": atlas_gene}
        if quant:
            for gname, m in groups.items():
                grp_rows.append({"gene": gene, "scp_group": gname, "n_cells": int(m.sum()),
                                 "n_detected": int(det.loc[m, gene].sum()), "detection": float(det.loc[m, gene].mean()),
                                 "mean_ln1p": float(ln1p.loc[m, gene].mean())})
        if in_atlas:
            seg_classes = T.PT_CLASSES if seg == "PT" else (*T.TAL_CLASSES, T.MD_CLASS)
            s2 = sub.set_index("class")
            for c in T.REGION_CLASSES:
                cls_rows.append({"gene": gene, "atlas_gene": atlas_gene, "class": c,
                                 "n_cells": int(s2.loc[c, "n_cells"]), "mean_lognorm": float(s2.loc[c, "mean_lognorm"]),
                                 "frac_detected": float(s2.loc[c, "frac_detected"])})
            seg_means = s2.loc[list(seg_classes), "mean_lognorm"]
            rec["atlas_highest_class_in_segment"] = str(seg_means.idxmax())
            rec["atlas_class_means"] = "; ".join(f"{c} {s2.loc[c, 'mean_lognorm']:.2f}" for c in seg_classes)
            rec["atlas_class_detection"] = "; ".join(f"{c} {s2.loc[c, 'frac_detected']:.2f}" for c in seg_classes)
        checked = contrast if contrast else "CTAL"
        rec["atlas_contrast_checked"] = (f"{checked}: {';'.join(T.CONTRASTS[checked][1])} vs "
                                         f"{';'.join(T.CONTRASTS[checked][2])}")
        if in_atlas:
            d = de[(de["contrast"] == checked) & (de["atlas_gene"] == atlas_gene)].iloc[0]
            rec.update(atlas_log2fc=float(d["log2fc"]), atlas_q=float(d["q_bh"]),
                       atlas_mean_difference=float(d["mean_difference"]), atlas_frac_group=float(d["frac_group"]),
                       atlas_frac_reference=float(d["frac_reference"]),
                       atlas_batches_same_sign=f"{int(d['n_batches_same_sign'])} of {int(d['n_batches_eligible'])}",
                       atlas_passes_panel_rna_filters=bool(d["passes_rna_filters"]),
                       atlas_direction_observed=direction(float(d["log2fc"]), float(d["q_bh"])))
            if contrast:
                rec["atlas_direction_vs_brief_group"] = {
                    "higher in group": "higher in the plan's group",
                    "lower in group": "lower in the plan's group"}.get(rec["atlas_direction_observed"],
                                                                        rec["atlas_direction_observed"])
            else:
                rec["atlas_direction_vs_brief_group"] = ("plan gives no CTAL / MTAL direction; observed: "
                                                         + {"higher in group": "higher in CTAL",
                                                            "lower in group": "higher in MTAL"}.get(
                                                                rec["atlas_direction_observed"],
                                                                rec["atlas_direction_observed"]))
        else:
            rec["atlas_direction_vs_brief_group"] = "not in atlas"
        for col in T.LABELS:
            s = T.LABEL_SHORT[col]
            hit = panels[(panels["label_column"] == col) & (panels["protein"] == gene)]
            rec[f"in_panels_{s}"] = "; ".join(f"{r.panel} (rank {int(r.rank)})" for r in hit.itertuples())
        # SCP direction
        if quant and seg == "PT":
            for gname, lab in groupings.items():
                a, b = lab.eq("PCT"), lab.eq("PST")
                e = T.effect_sizes(ln1p.loc[a, gene], ln1p.loc[b, gene])
                rec[f"scp_{gname}_n_pct_pst"] = f"{int(a.sum())} / {int(b.sum())}"
                rec[f"scp_{gname}_detection_pct"] = float(det.loc[a, gene].mean())
                rec[f"scp_{gname}_detection_pst"] = float(det.loc[b, gene].mean())
                rec[f"scp_{gname}_auc_pct_vs_pst"] = e["auc"]
                rec[f"scp_{gname}_p"] = e["p_mannwhitney"]
        if quant and brief_group == "MD":
            md, tal = labels["kidney_pre_label"].eq("MD"), labels["kidney_pre_label"].eq("TAL")
            rec["scp_fig2a_md_detected"] = f"{int(det.loc[md, gene].sum())} of {int(md.sum())}"
            rec["scp_fig2a_tal_detected"] = f"{int(det.loc[tal, gene].sum())} of {int(tal.sum())}"
        rows.append(rec)
        checks.add(f"{gene} ({brief_group}): quantified; cells detected; in atlas; atlas direction",
                   f"{quant}; {rec['n_cells_detected_all_712']}; {in_atlas}; {rec['atlas_direction_vs_brief_group']}",
                   None)
    lit = pd.DataFrame(rows)
    # Cldn10 vs Cldn16 / Cldn19 sign
    tal = lit[lit["brief_group"].str.startswith("TAL") & lit["in_atlas"]].set_index("gene")
    if {"Cldn10", "Cldn19"} <= set(tal.index):
        opp = np.sign(tal.loc["Cldn10", "atlas_mean_difference"]) != np.sign(tal.loc["Cldn19", "atlas_mean_difference"])
        checks.add("atlas CTAL - MTAL difference: Cldn10 and Cldn19 have opposite signs (Cldn16 not an atlas gene)",
                   f"{bool(opp)} (Cldn10 {tal.loc['Cldn10', 'atlas_mean_difference']:+.3f}; Cldn19 "
                   f"{tal.loc['Cldn19', 'atlas_mean_difference']:+.3f})", None)
    n_q = int(lit["quantified_protein"].sum())
    n_a = int(lit["in_atlas"].sum())
    checks.add("candidates quantified as protein; in the atlas; both", f"{n_q}; {n_a}; "
               f"{int((lit['quantified_protein'] & lit['in_atlas']).sum())} of {len(lit)}", None)
    md = labels["kidney_pre_label"].eq("MD")
    checks.add("Nos1 detected in the Fig 2A MD cells (the project notes: 0 of 19)", f"{int(det.loc[md, 'Nos1'].sum())} of "
               f"{int(md.sum())}", int(det.loc[md, "Nos1"].sum()) == 0, "0")
    run.table(lit, "literature")
    run.table(pd.DataFrame(grp_rows), "literature_scp_groups")
    run.table(pd.DataFrame(cls_rows), "literature_atlas_classes")
    run.table(checks.frame(), "literature_checks")
    run.finish({"candidates": [g for g, _, _ in T.LITERATURE], "source": "plan examples only (authors, 15 Oct 2026)",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see pt_tal_literature_checks.csv.")


if __name__ == "__main__":
    main()
