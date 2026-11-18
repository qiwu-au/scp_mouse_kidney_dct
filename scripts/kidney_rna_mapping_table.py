#!/usr/bin/env python
"""kidney_rna step 2: proposed mapping of every SCP kidney label to atlas Celltype / Celltype_finest classes.

Analysis plan,
step 2. The plan says to stop and show the table to the authors before any analysis uses it. Run 1 wrote the proposal;
the authors approved it on 14 Oct 2026, with every flagged row as proposed (USER_DECISIONS); run 2 records the approval.

Rule
----
Each label is mapped by its nominal identity (the V3 workbook full name; for 'Endothelial' the kidney_annotation row), starting
from kidney_annotation's approved atlas Celltype table (kidney_annotation_atlas_celltype_mapping.csv, approved 14 Oct 2026):
- expected (strict) atlas classes: the classes of that identity;
- mixed classes accepted: the atlas classes that pool two neighbouring identities (DCT-CNT, DTL-ATL, LOH, CD-Trans)
  and, at the finest level, the unsegmented study labels (PT, IC). Step 4 reports the expected-label fraction with
  and without them;
- labels without an atlas counterpart (JG, Unknown) get no classes: empty RNA dotplot row, no expected fraction,
  not in the concordance.
Observations from kidney_annotation that bear on a row (cluster composition, endothelial proteins in the Fig 2A MD cells) are
listed per row, with the alternative mapping that the authors may prefer ('decision' column).

Objective support per row ('marker support'): the atlas Celltypes (Unknown excluded) ranked by the mean, over the
label's listed markers found in the atlas (all listed markers, quantified as protein or not), of the per-gene mean
log-normalized expression rescaled 0-1 across Celltypes (from the kidney_rna pseudobulk).

Inputs (read-only): kidney_rna step-1 outputs (level counts, pseudobulk); kidney_annotation approved atlas mapping; kidney_object canonical
kidney labels and the kidney_annotation labels; data/KIDNEY/KIDNEY_markers_V3.xlsx.
Outputs: results/tables/kidney_rna/kidney_rna_celltype_mapping.csv, kidney_rna_mapping_marker_support.csv,
         kidney_rna_mapping_table_checks.csv

Command (project root):  python scripts/kidney_rna_mapping_table.py
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

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as L

STATUS = ("proposed in this analysis and approved by the authors, 14 Oct 2026 (kidney_rna step 2; the four flagged rows as proposed: "
          "kidney_annotation PC -> PC only; kidney_annotation Endothelial -> all six endothelial classes; JG -> no counterpart; Fig 2A MD -> "
          "atlas MD)")
# User's answers to the flagged rows (14 Oct 2026, kidney_rna step-2 checkpoint): every row as proposed.
USER_DECISIONS = {
    ("kidney_t15_label", "PC"): "PC only (as proposed)",
    ("kidney_t15_label", "Endothelial"): "all six endothelial classes (as proposed)",
    ("kidney_t15_label", "JG"): "no counterpart (as proposed)",
    ("kidney_pre_label", "JG"): "no counterpart (as proposed)",
    ("kidney_reannot_label", "JG"): "no counterpart (as proposed)",
    ("kidney_pre_label", "MD"): "atlas MD by name (as proposed)",
}
KIDNEY_ANNOTATION_SOURCE = "kidney_annotation approved atlas Celltype table (14 Oct 2026)"
LEVEL_COUNTS = cc.out_path(L.TASK, "tables", "atlas_level_counts", ".csv")
MIXED_CELLTYPES = ("CD-Trans", "DCT-CNT", "DTL-ATL", "LOH")
ENDO_ALL = ("Endo", "Glom-Endo", "Vas-Afferens", "Vas-Efferens", "Asc-Vasa-Recta", "Desc-Vasa-Recta")

# Per SCP identity: (Celltype strict, Celltype mixed, finest strict, finest mixed, kidney_annotation SCP type, rationale).
TYPE_MAP = {
    "PCT": (("PTS1", "PTS2"), (), ("PCT", "PTS1", "PTS2", "PTS1-2"), ("PT",), "PCT",
            "V3 'Proximal convoluted tubule'; atlas PTS1 and PTS2 are the S1 and S2 proximal tubule segments "
            "(as kidney_annotation). Finest: PCT (Miao21), PTS1, PTS2, PTS1-2 (Wu19, S1 + S2); the unsegmented PT label (Park18) "
            "is accepted as mixed."),
    "PST": (("PTS3", "PTS3T2"), (), ("PST", "PTS3", "PTS3T2"), ("PT",), "PST",
            "V3 'Proximal straight tubule'; atlas PTS3 and PTS3T2 are the S3 segment (as kidney_annotation). Finest: PST (Miao21), "
            "PTS3, PTS3T2; unsegmented PT accepted as mixed."),
    "DTL": (("DTL",), ("DTL-ATL", "LOH"), ("DTL",), ("DTL-ATL", "LOH"), "DTL",
            "V3 'Descending thin limb of LOH'; atlas DTL (as kidney_annotation). DTL-ATL and LOH pool limbs: accepted as mixed."),
    "ATL": (("ATL",), ("DTL-ATL", "LOH"), ("ATL",), ("DTL-ATL", "LOH"), "ATL",
            "V3 'Ascending thin limb of LOH'; atlas ATL (as kidney_annotation). DTL-ATL and LOH pool limbs: accepted as mixed."),
    "TAL": (("CTAL", "MTAL"), ("LOH",), ("CTAL", "MTAL"), ("LOH",), "TAL",
            "V3 'Thick ascending limb of LOH'; atlas CTAL and MTAL (as kidney_annotation). LOH (loop of Henle, limbs not split) "
            "accepted as mixed (the earlier analysis counted LOH as expected for TAL)."),
    "MD": (("MD",), (), ("MD",), (), "MD",
           "V3 'Macula densa'; atlas MD (as kidney_annotation): 46 cells, all snRNA-seq (Kirita20 39, Hinze20 7)."),
    "DCT1": (("DCT",), ("DCT-CNT",), ("DCT",), ("DCT-CNT",), "DCT",
             "V3 'DCT1'; the atlas has one DCT class, not split into DCT1 / DCT2 (as kidney_annotation). DCT-CNT accepted as "
             "mixed."),
    "DCT2": (("DCT",), ("DCT-CNT",), ("DCT",), ("DCT-CNT",), "DCT",
             "V3 'DCT2'; the atlas has one DCT class, not split into DCT1 / DCT2 (as kidney_annotation). DCT-CNT accepted as "
             "mixed."),
    "CNT": (("CNT",), ("DCT-CNT",), ("CNT",), ("DCT-CNT",), "CNT",
            "V3 'CNT'; atlas CNT (as kidney_annotation). DCT-CNT accepted as mixed."),
    "PC": (("PC",), ("CD-Trans",), ("PC",), ("CD-Trans",), "PC",
           "V3 'CD-principal cells'; atlas PC (as kidney_annotation). CD-Trans (collecting-duct transitional class) accepted as "
           "mixed."),
    "IC-A": (("ICA",), ("CD-Trans",), ("ICA",), ("IC", "CD-Trans"), "IC-A",
             "V3 'CD-intercalated cells (type A)'; atlas ICA (as kidney_annotation). CD-Trans accepted as mixed; at the finest level "
             "also the unsplit IC label (Miao21, Park18, Janosevic21)."),
    "IC-B": (("ICB",), ("CD-Trans",), ("ICB",), ("IC", "CD-Trans"), "IC-B",
             "V3 'CD-intercalated cells (type B)'; atlas ICB (as kidney_annotation). CD-Trans accepted as mixed; at the finest level "
             "also the unsplit IC label."),
    "DVR": (("Desc-Vasa-Recta",), (), ("Desc-Vasa-Recta",), (), "DVR",
            "V3 'Descending vasa recta'; atlas Desc-Vasa-Recta (as kidney_annotation): 167 cells, all scRNA-seq (Dumas20 152)."),
    "JG": ((), (), (), (), None,
           "V3 'Juxtaglomerular cells'; no JG (or renin) class at any atlas level (Celltype, Celltype_finest_lowres, "
           "Celltype_finest); no counterpart proposed. the earlier concordance script used Per + MC."),
    "Endothelial": (ENDO_ALL, (), ENDO_ALL, (), "Endo",
                    "kidney_annotation row 'Endothelial' (Pecam1, Emcn, Eng, Esam, Icam2: atlas Endo markers; Fabp4, Ehd4: atlas "
                    "AVR / DVR markers). The label names no vascular bed, so all six atlas endothelial classes are "
                    "proposed: kidney_annotation's 'Endo' group (Endo, Glom-Endo, Vas-Afferens, Vas-Efferens) plus Asc-Vasa-Recta "
                    "and Desc-Vasa-Recta."),
    "Unknown": ((), (), (), (), None,
                "No identity; no counterpart. Not in the expected-label fractions or the concordance; empty RNA "
                "dotplot row."),
}

# (label column, label): (kidney_annotation observation, decision for the authors, alternative mapping).
LABEL_NOTES = {
    ("kidney_t15_label", "DCT1"): ("kidney_annotation clusters 15 + 16: all 24 Fig 2A DCT2 cells and 34 Fig 2A CNT cells; suggested "
                                   "display name DCT (the project notes).", "", ""),
    ("kidney_t15_label", "PC"): ("kidney_annotation cluster 7: 38 cells, 100% Fig 2A CNT; rescued by Rhbg / Aqp3; suggested display "
                                 "name CNT/PC (the project notes).", "PC only, or CNT + PC?",
                                 "Celltype CNT + PC (display name CNT/PC); mixed DCT-CNT, CD-Trans"),
    ("kidney_t15_label", "Endothelial"): ("kidney_annotation cluster 20 = the 19 Fig 2A 'MD' cells.",
                                          "all six endothelial classes, or kidney_annotation's 'Endo' group only?",
                                          "Celltype Endo, Glom-Endo, Vas-Afferens, Vas-Efferens (kidney_annotation 'Endo')"),
    ("kidney_t15_label", "JG"): ("kidney_annotation cluster 9: 22 of the 22 Fig 2A JG cells, plus 1 TAL and 1 CNT.",
                                 "no counterpart, or Per + MC (the earlier analysis)?", "Celltype Per + MC"),
    ("kidney_t15_label", "TAL"): ("kidney_annotation: 138 of the 154 Fig 2A TAL cells.", "", ""),
    ("kidney_pre_label", "MD"): ("kidney_annotation: endothelial proteins detected in these 19 cells; Nos1 in 0 of 19 (dotplot / kidney_annotation); "
                                 "they form kidney_annotation's Endothelial cluster.", "MD by name, or the endothelial classes?",
                                 "Celltype Endo, Glom-Endo, Vas-Afferens, Vas-Efferens, Asc-Vasa-Recta, "
                                 "Desc-Vasa-Recta"),
    ("kidney_pre_label", "DVR"): ("kidney_annotation: 52 of 88 -> PST, 33 Unknown, 3 PCT.", "", ""),
    ("kidney_pre_label", "DCT2"): ("kidney_annotation: all 24 -> DCT1.", "", ""),
    ("kidney_pre_label", "CNT"): ("kidney_annotation: 34 -> DCT1, 38 -> PC, 1 -> JG.", "", ""),
    ("kidney_pre_label", "JG"): ("kidney_annotation: all 22 -> JG.", "no counterpart, or Per + MC (the earlier analysis)?", "Celltype Per + MC"),
    ("kidney_reannot_label", "MD"): ("1 cell.", "", ""),
    ("kidney_reannot_label", "JG"): ("", "as for the other label columns", "Celltype Per + MC"),
}
REANNOT_NOTE = ("Submitted re-annotation (Fig 2B); 40% of its labels come from the competing-marker penalty (dotplot "
                "addenda); used only in the step-4 comparison.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_rna step 2: proposed SCP label -> atlas class mapping.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "mapping_table_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_rna_mapping_table_checks.csv.")


def full_name(label: str) -> str:
    if label == L.ENDO_TYPE:
        return "Endothelial (kidney_annotation extension)"
    inv = {v: k for k, v in cc.KIDNEY_ABBREVIATIONS.items()}
    return inv.get(label, label)


def marker_support(pb, rows, celltypes: list[str]) -> pd.DataFrame:
    """Atlas Celltypes ranked per SCP type by the mean 0-1 scaled expression of the type's listed markers."""
    look = L.atlas_gene_lookup(pb.var_names)
    means = {}
    for ct in celltypes:
        agg = L.aggregate(pb, L.group_mask(pb, Celltype=ct))
        means[ct] = agg["mean_lognorm"]
    m = pd.DataFrame(means)                                   # genes x celltypes
    out = []
    for short, _, genes in rows:
        found = [look[g.lower()] for g in genes if g.lower() in look]
        if not found:
            continue
        sub = m.loc[found]
        lo, hi = sub.min(axis=1), sub.max(axis=1)
        scaled = sub.sub(lo, axis=0).div((hi - lo).where(hi > lo), axis=0).fillna(0.0)
        score = scaled.mean(axis=0).sort_values(ascending=False)
        for rank, (ct, val) in enumerate(score.items(), start=1):
            out.append({"scp_type": short, "markers_in_atlas": ";".join(found), "n_markers_in_atlas": len(found),
                        "n_markers_listed": len(genes), "atlas_celltype": ct, "score": float(val), "rank": rank})
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "mapping_table", args)
    t_start = time.time()
    checks = t5.Checks(L.DATASET)

    _, labels = L.load_kidney(run, checks)
    rows_markers = L.kidney_marker_rows(run)
    levels = pd.read_csv(run.use_input(LEVEL_COUNTS, "kidney_rna atlas level counts (kidney_rna_atlas_summary.py)"))
    t15map = pd.read_csv(run.use_input(L.KIDNEY_ANNOTATION_ATLAS_MAPPING, "kidney_annotation approved atlas Celltype mapping"))
    pb = L.read_pseudobulk(run)

    vocab = {lvl: set(levels.loc[levels["level"] == lvl, "label"]) for lvl in ("Celltype", "Celltype_finest")}
    bad = []
    for t, (cs, cm, fs, fm, _, _) in TYPE_MAP.items():
        bad += [f"{t}: Celltype {c}" for c in (*cs, *cm) if c not in vocab["Celltype"]]
        bad += [f"{t}: finest {c}" for c in (*fs, *fm) if c not in vocab["Celltype_finest"]]
    checks.add("every mapped atlas class exists at its level", "; ".join(bad) or "all", not bad)
    present = sorted(set().union(*[set(labels[c]) for c in L.LABEL_COLUMNS]))
    missing = [p for p in present if p not in TYPE_MAP]
    checks.add("every label of the three label columns has a mapping definition", ", ".join(missing) or "all",
               not missing)
    if not checks.ok:
        stop(run, checks)

    t15_sets = t15map[t15map["role"] == "target"].groupby("scp_type")["atlas_celltype"].apply(lambda s: set(s))
    counts = levels.set_index(["level", "label"])

    def n_cells(level: str, names, col: str = "n_cells") -> int:
        return int(sum(counts.loc[(level, n), col] for n in names))

    celltypes = sorted(vocab["Celltype"] - {L.ATLAS_UNKNOWN})
    support = marker_support(pb, rows_markers, celltypes)
    support["is_expected"], support["is_mixed_accepted"] = False, False
    for t, (cs, cm, _, _, _, _) in TYPE_MAP.items():
        sel = support["scp_type"] == t
        support.loc[sel, "is_expected"] = support.loc[sel, "atlas_celltype"].isin(cs)
        support.loc[sel, "is_mixed_accepted"] = support.loc[sel, "atlas_celltype"].isin(cm)
    support["is_mixed_class"] = support["atlas_celltype"].isin(MIXED_CELLTYPES)
    nm = ~support["is_mixed_class"]
    support.loc[nm, "rank_excl_mixed"] = support[nm].groupby("scp_type")["score"].rank(ascending=False,
                                                                                           method="first")

    out = []
    for col in L.LABEL_COLUMNS:
        vc = labels[col].value_counts()
        for lab in L.row_order(vc.index):
            cs, cm, fs, fm, t15type, rationale = TYPE_MAP[lab]
            obs_note, decision, alternative = LABEL_NOTES.get((col, lab), ("", "", ""))
            if col == "kidney_reannot_label":
                obs_note = (obs_note + " " + REANNOT_NOTE).strip()
            if not cs:
                source = "no counterpart (kidney_rna)"
            elif t15type in t15_sets.index and set(cs) == t15_sets[t15type]:
                source = KIDNEY_ANNOTATION_SOURCE
            elif t15type in t15_sets.index and t15_sets[t15type] <= set(cs):
                source = f"{KIDNEY_ANNOTATION_SOURCE}, extended: kidney_annotation '{t15type}' group plus " + \
                         ", ".join(sorted(set(cs) - t15_sets[t15type]))
            else:
                source = "new (kidney_rna)"
            sup = support[support["scp_type"] == lab].sort_values("rank")
            top3 = "; ".join(f"{r.atlas_celltype} {r.score:.2f}" for r in sup.head(3).itertuples())
            exp_rank = int(sup.loc[sup["is_expected"].astype(bool), "rank"].min()) if (len(sup) and cs) else np.nan
            sup_nm = sup[~sup["is_mixed_class"].astype(bool)]
            top3_nm = "; ".join(f"{r.atlas_celltype} {r.score:.2f}" for r in sup_nm.head(3).itertuples())
            exp_rank_nm = (int(sup_nm.loc[sup_nm["is_expected"].astype(bool), "rank_excl_mixed"].min())
                           if (len(sup_nm) and cs) else np.nan)
            used = ("step-4 comparison only" if col == "kidney_reannot_label" else
                    ("dotplot RNA row; mapping; concordance" if cs else "dotplot (empty RNA row)"))
            out.append({
                "label_column": col, "label_role": L.LABEL_ROLE[col], "scp_label": lab,
                "scp_label_full_name": full_name(lab), "n_scp_cells": int(vc[lab]),
                "atlas_celltype_expected": ";".join(cs), "atlas_celltype_mixed_accepted": ";".join(cm),
                "n_atlas_expected": n_cells("Celltype", cs) if cs else 0,
                "n_atlas_expected_cell": n_cells("Celltype", cs, "n_source_Cell") if cs else 0,
                "n_atlas_expected_nuclei": n_cells("Celltype", cs, "n_source_Nuclei") if cs else 0,
                "n_atlas_mixed_accepted": n_cells("Celltype", cm) if cm else 0,
                "atlas_finest_expected": ";".join(fs), "atlas_finest_mixed_accepted": ";".join(fm),
                "n_atlas_finest_expected": n_cells("Celltype_finest", fs) if fs else 0,
                "n_atlas_finest_expected_nuclei": n_cells("Celltype_finest", fs, "n_source_Nuclei") if fs else 0,
                "used_in": used, "source": source, "rationale": rationale, "t15_observation": obs_note,
                "decision_for_user": decision, "user_decision": USER_DECISIONS.get((col, lab), ""),
                "alternative_mapping": alternative,
                "marker_support_top3": top3, "marker_support_rank_of_expected": exp_rank,
                "marker_support_top3_excl_mixed": top3_nm, "marker_support_rank_of_expected_excl_mixed": exp_rank_nm,
                "n_markers_in_atlas": int(sup["n_markers_in_atlas"].iloc[0]) if len(sup) else 0,
                "status": STATUS,
            })
    table = pd.DataFrame(out)

    for col in L.LABEL_COLUMNS:
        sub = table[table["label_column"] == col]
        checks.add(f"{col}: rows; labels with expected classes; SCP cells covered",
                   f"{len(sub)}; {int((sub['atlas_celltype_expected'] != '').sum())}; "
                   f"{int(sub.loc[sub['atlas_celltype_expected'] != '', 'n_scp_cells'].sum())} of "
                   f"{int(sub['n_scp_cells'].sum())}", len(sub) == len(L.EXPECTED_COUNTS[col]))
    n_dec = int((table["decision_for_user"] != "").sum())
    checks.add("rows with a decision for the authors", f"{n_dec}: " + "; ".join(
        f"{r.label_column}/{r.scp_label}" for r in table[table["decision_for_user"] != ""].itertuples()), None)
    undecided = table[(table["decision_for_user"] != "") & (table["user_decision"] == "")]
    checks.add("every flagged row has the authors' decision", ", ".join(
        f"{r.label_column}/{r.scp_label}" for r in undecided.itertuples()) or "all", undecided.empty)
    for rank_col, top_col, what in (("marker_support_rank_of_expected", "marker_support_top3", "all Celltypes"),
                                    ("marker_support_rank_of_expected_excl_mixed", "marker_support_top3_excl_mixed",
                                     "mixed classes excluded")):
        not_top = table[(table["label_column"] != "kidney_reannot_label") & (table["atlas_celltype_expected"] != "")
                        & (table[rank_col] > 1)]
        checks.add(f"kidney_annotation / pre labels whose expected class is not the top marker-support Celltype ({what})",
                   "; ".join(f"{r.label_column}/{r.scp_label} (rank {int(getattr(r, rank_col))}; top "
                             f"{getattr(r, top_col).split(';')[0]})" for r in not_top.itertuples()) or "none", None)
    no_markers = sorted(set(table.loc[(table["atlas_celltype_expected"] != "") & (table["n_markers_in_atlas"] == 0),
                                      "scp_label"]))
    checks.add("labels with expected classes but no listed marker in the atlas", ", ".join(no_markers) or "none",
               None)

    run.table(table, "celltype_mapping")
    run.table(support, "mapping_marker_support")
    run.table(checks.frame(), "mapping_table_checks")
    run.finish({"rule": "nominal identity of the label (V3 full name; kidney_annotation endothelial row), starting from kidney_annotation's "
                        "approved Celltype table", "mixed_celltypes": list(MIXED_CELLTYPES),
                "marker_support": "per gene, Celltype mean log-normalized expression (pseudobulk; Unknown excluded) "
                                  "rescaled 0-1 across Celltypes; score = mean over the type's listed markers in "
                                  "the atlas", "status": STATUS,
                "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
