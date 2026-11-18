#!/usr/bin/env python
"""rna_protein_integration step 4: where RNA or protein detects a marker, per cell type (from the kidney_rna concordance tables).

Analysis plan,
step 4. No new RNA or protein quantities: the per-compartment detection fractions and the across-compartment
concordance classes are read from kidney_rna step 5 (kidney_rna_concordance_long.csv, _features.csv, _markers.csv).

Per compartment (SCP label with an atlas counterpart; RNA = atlas cells of its expected Celltypes) and feature:
detected in both / RNA only / protein only / neither at the kidney_rna threshold (detection fraction >= 10% in that
compartment; RNA: counts > 0, protein: intensity > 0). Across compartments: the kidney_rna class (concordant rho >= 0.5,
intermediate 0.2-0.5, discordant < 0.2, RNA-detected only, protein-detected only, detected in neither).
Markers: kidney V3 + the kidney_annotation endothelial row, quantified as protein, in kidney_annotation dotplot order (full version, no
repeats). Transporters and channels: gene-symbol prefix rule of rna_protein_integration_lib (rna_protein_integration_transporter_channel_rule.csv;
not a curated annotation). Primary: kidney_t15_label, all atlas cells, threshold 10%; kidney_pre_label and atlas
nuclei only are in the tables.

Inputs (read-only): kidney_rna concordance tables; kidney_annotation dotplot marker order (a_endo); rna_protein_integration_features.csv.
Outputs: results/tables/rna_protein_integration/rna_protein_integration_strengths_markers.csv, rna_protein_integration_strengths_per_type.csv,
         rna_protein_integration_strengths_classes.csv, rna_protein_integration_strengths_checks.csv
Command (project root):  python scripts/rna_protein_integration_strengths.py
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
import rna_protein_integration_lib as T

THR = 0.10
DETECTION_CLASSES = ("detected in both", "RNA only", "protein only", "detected in neither")
KIDNEY_RNA_CLASSES = ("Detected in both; concordant", "Detected in both; intermediate", "Detected in both; discordant",
               "RNA-detected only", "Protein-detected only", "Detected in neither")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rna_protein_integration step 4: RNA vs protein detection per cell type.")
    cc.add_common_args(parser)
    return parser.parse_args()


def detection_class(rna: float, prot: float, thr: float = THR) -> str:
    if not np.isfinite(rna):
        return "not in atlas"
    r, p = rna >= thr, prot >= thr
    return DETECTION_CLASSES[0] if r and p else DETECTION_CLASSES[1] if r else DETECTION_CLASSES[2] if p else DETECTION_CLASSES[3]


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "strengths", args)
    checks = t5.Checks(T.DATASET)

    def t07(name: str) -> pd.DataFrame:
        return pd.read_csv(run.use_input(cc.out_path(L7.TASK, "tables", name, ".csv"), f"kidney_rna step 5 {name}"))

    long_df, feat07, mk07 = t07("concordance_long"), t07("concordance_features"), t07("concordance_markers")
    feat = pd.read_csv(run.use_input(cc.out_path(T.TASK, "tables", "features", ".csv"), "rna_protein_integration features table"))
    order = pd.read_csv(run.use_input(L7.KIDNEY_ANNOTATION_MARKER_ORDER, "kidney_annotation a_endo dotplot marker order"))
    order = order[(order["version"] == "full") & order["plotted"].astype(bool) & ~order["repeat"].fillna(False).astype(bool)]
    order = order.sort_values("position").reset_index(drop=True)
    tc = dict(zip(feat["atlas_gene"].astype(str), feat["transporter_channel"].astype(bool)))
    tc_prefix = dict(zip(feat["atlas_gene"].astype(str), feat["transporter_channel_prefix"].fillna("").astype(str)))
    prot2gene = dict(zip(feat["protein"].astype(str), feat["atlas_gene"].astype(str)))
    checks.add("kidney_annotation dotplot markers (full, no repeats)", f"{len(order)}", None)

    # markers per compartment -----------------------------------------------------------------------------------
    rows = []
    for col in T.LABEL_COLUMNS:
        for src in ("all", "nuclei"):
            lg = long_df[(long_df["label_column"] == col) & (long_df["rna_source"] == src)]
            comps = L7.row_order(lg["compartment"].unique())
            f7 = feat07[(feat07["label_column"] == col) & (feat07["rna_source"] == src) & np.isclose(feat07["threshold"], THR)]
            f7 = f7.set_index("protein")
            m7 = mk07[(mk07["label_column"] == col) & (mk07["rna_source"] == src) & np.isclose(mk07["threshold"], THR)]
            m7 = m7.drop_duplicates("marker").set_index("marker")
            lgi = lg.set_index(["compartment", "protein"])
            for pos, r in enumerate(order.itertuples(), start=1):
                marker = str(r.marker)
                gene = prot2gene.get(marker)
                cls07 = str(m7.loc[marker, "class"]) if marker in m7.index else "not listed"
                rho = float(f7.loc[marker, "spearman_rho"]) if marker in f7.index else np.nan
                for comp in comps:
                    if gene is not None and (comp, marker) in lgi.index:
                        x = lgi.loc[(comp, marker)]
                        rna, prot = float(x["rna_detection"]), float(x["protein_detection"])
                        rmean, pmean = float(x["rna_mean_lognorm"]), float(x["protein_mean_ln1p"])
                    else:
                        rna, rmean = np.nan, np.nan
                        prot, pmean = np.nan, np.nan
                    rows.append({"label_column": col, "rna_source": src, "threshold": THR, "compartment": comp,
                                 "marker": marker, "marker_position": pos, "marker_group": r.group,
                                 "listed_for": r.listed_for, "atlas_gene": gene or "",
                                 "transporter_channel": bool(tc.get(gene, False)) if gene else False,
                                 "transporter_channel_prefix": tc_prefix.get(gene, "") if gene else "",
                                 "rna_detection": rna, "protein_detection": prot, "rna_mean_lognorm": rmean,
                                 "protein_mean_ln1p": pmean,
                                 "detection_class": detection_class(rna, prot if np.isfinite(prot) else 0.0),
                                 "t07_class_across_compartments": cls07, "t07_spearman_rho": rho})
    markers = pd.DataFrame(rows)
    prim = markers[(markers["label_column"] == "kidney_t15_label") & (markers["rna_source"] == "all")]
    checks.add("primary marker x compartment cells (kidney_annotation labels, all atlas cells, 10%)",
               f"{prim['marker'].nunique()} markers x {prim['compartment'].nunique()} compartments", None)
    # Consistency with kidney_rna: a marker detected in both at some compartment <=> kidney_rna class 'Detected in both; ...'
    agg = prim.groupby("marker").agg(any_rna=("rna_detection", lambda s: bool((s >= THR).any())),
                                     any_prot=("protein_detection", lambda s: bool((s >= THR).any())),
                                     cls=("t07_class_across_compartments", "first"))
    exp_cls = np.where(agg["any_rna"] & agg["any_prot"], "both", np.where(agg["any_rna"], "RNA-detected only",
                       np.where(agg["any_prot"], "Protein-detected only", "Detected in neither")))
    obs_cls = np.where(agg["cls"].str.startswith("Detected in both"), "both", agg["cls"])
    in_atlas = agg["cls"] != "not in atlas"
    ok = bool((exp_cls[in_atlas.to_numpy()] == obs_cls[in_atlas.to_numpy()]).all())
    checks.add("per-compartment detection reproduces the kidney_rna across-compartment class of every marker in the atlas",
               f"{int(in_atlas.sum())} markers; agree {ok}", ok)

    # per type: transporter / channel vs other shared features ------------------------------------------------------
    per_type = []
    for col in T.LABEL_COLUMNS:
        for src in ("all", "nuclei"):
            lg = long_df[(long_df["label_column"] == col) & (long_df["rna_source"] == src)].copy()
            lg["transporter_channel"] = lg["gene"].astype(str).map(tc).fillna(False).astype(bool)
            lg["cls"] = [detection_class(a, b) for a, b in zip(lg["rna_detection"], lg["protein_detection"])]
            for comp in L7.row_order(lg["compartment"].unique()):
                sub = lg[lg["compartment"] == comp]
                for grp, gm in (("transporter / channel", sub["transporter_channel"]), ("other shared features",
                                                                                      ~sub["transporter_channel"])):
                    s = sub[gm.to_numpy()]
                    vc = s["cls"].value_counts()
                    rec = {"label_column": col, "rna_source": src, "threshold": THR, "compartment": comp,
                           "atlas_celltypes": s["atlas_celltypes"].iloc[0], "n_scp_cells": int(s["n_scp_cells"].iloc[0]),
                           "n_atlas_cells": int(s["n_atlas_cells"].iloc[0]), "feature_group": grp, "n_features": len(s)}
                    for c in DETECTION_CLASSES:
                        rec[f"n_{c.replace(' ', '_')}"] = int(vc.get(c, 0))
                        rec[f"fraction_{c.replace(' ', '_')}"] = vc.get(c, 0) / len(s) if len(s) else np.nan
                    per_type.append(rec)
    per_type = pd.DataFrame(per_type)

    # across-compartment classes: transporter / channel vs other ------------------------------------------------------
    classes = []
    for (col, src, thr), f in feat07.groupby(["label_column", "rna_source", "threshold"]):
        f = f.copy()
        f["transporter_channel"] = f["gene"].astype(str).map(tc).fillna(False).astype(bool)
        for grp, gm in (("transporter / channel", f["transporter_channel"]), ("other shared features",
                                                                               ~f["transporter_channel"])):
            s = f[gm.to_numpy()]
            vc = s["class"].value_counts()
            rec = {"label_column": col, "rna_source": src, "threshold": thr, "feature_group": grp, "n_features": len(s),
                   "median_rho_detected_both": float(s.loc[s["class"].str.startswith("Detected in both"),
                                                           "spearman_rho"].median())}
            for c in KIDNEY_RNA_CLASSES:
                rec[f"n_{c}"] = int(vc.get(c, 0))
                rec[f"fraction_{c}"] = vc.get(c, 0) / len(s) if len(s) else np.nan
            classes.append(rec)
    classes = pd.DataFrame(classes)
    p = classes[(classes["label_column"] == "kidney_t15_label") & (classes["rna_source"] == "all")
                & np.isclose(classes["threshold"], THR)]
    for r in p.itertuples():
        checks.add(f"kidney_annotation, all, 10%: {r.feature_group}: n; concordant / intermediate / discordant / RNA only / "
                   "protein only / neither",
                   f"{r.n_features}; " + " / ".join(str(int(p.loc[r.Index, f"n_{c}"])) for c in KIDNEY_RNA_CLASSES), None)
    n_tc = int(feat["transporter_channel"].sum())
    checks.add("transporter / channel features among the 3,569 (rna_protein_integration rule)", f"{n_tc}",
               int(p["n_features"].sum()) == len(feat) and int(p.loc[p["feature_group"] == "transporter / channel",
                                                                    "n_features"].iloc[0]) == n_tc)

    run.table(markers, "strengths_markers")
    run.table(per_type, "strengths_per_type")
    run.table(classes, "strengths_classes")
    run.table(checks.frame(), "strengths_checks")
    run.finish({"threshold": THR, "primary": "kidney_t15_label, all atlas cells",
                "transporter_rule": {"prefixes": list(T.TRANSPORTER_PREFIXES),
                                     "excluded_prefixes": list(T.TRANSPORTER_EXCLUDE_PREFIXES)},
                "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see rna_protein_integration_strengths_checks.csv.")


if __name__ == "__main__":
    main()
