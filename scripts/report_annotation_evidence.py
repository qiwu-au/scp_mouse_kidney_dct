#!/usr/bin/env python
"""report: per-cluster evidence table for the revised kidney annotation (kidney_t15_label).

Requested by the authors on 30 Oct 2026 (the response to the reviewers session):
present the revised annotation with its per-cluster evidence (marker, rank, AUC, confidence), and keep the revised
annotation as the only kidney annotation in the manuscript. The table is meant for the Supplementary Material of the
comment 4 response. No new statistics: every value is read from the kidney_annotation outputs of
the adopted labels (stored graph, Leiden 2.0, seed 42, cluster_de rule with the kidney_annotation rank-free rescue, V3 + endothelial row).

Per cluster (23): label (display names DCT1 -> DCT, PC -> CNT/PC), cells, median proteins per cell, evidence route,
the label's marker evidence with rank among the cluster's significantly increased proteins, AUC, BH q and detection
in / out of the cluster, the top-25 rule's result (before the rescue) and the other candidate types of a rescue.

Inputs (read-only):
  results/tables/kidney_annotation/kidney_annotation_chosen_clusters_a_endo.csv       (labels, routes, rescue markers)
  results/tables/kidney_annotation/kidney_annotation_scan_de_top25.csv                 (top-25 DE of the same clustering)
  results/tables/kidney_annotation/kidney_annotation_cluster_labels_per_cell_a_endo.csv (per-cell labels, for checks)
Outputs: results/tables/report/report_annotation_evidence.{csv,md}, report_annotation_evidence_checks.csv,
         report_step_annotation_evidence_manifest.json

Command (project root):  python scripts/report_annotation_evidence.py
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

import pandas as pd

import scp_common as cc

TASK = "report"
STEP = "annotation_evidence"
KIDNEY_ANNOTATION = cc.OUT_ROOTS["tables"] / "kidney_annotation"
CLUSTERS = KIDNEY_ANNOTATION / "kidney_annotation_chosen_clusters_a_endo.csv"
DE_TOP25 = KIDNEY_ANNOTATION / "kidney_annotation_scan_de_top25.csv"
PER_CELL = KIDNEY_ANNOTATION / "kidney_annotation_cluster_labels_per_cell_a_endo.csv"
CLUSTERING = "stored__r2.0"
DISPLAY = {"DCT1": "DCT", "PC": "CNT/PC"}      # authors, 30 Oct 2026
ROUTE = {   # kidney_annotation final confidence -> (route, confidence shown)
    "high": ("type-specific marker among the top 25 increased proteins", "high"),
    "medium": ("type-specific marker among the top 25 increased proteins", "medium"),
    "specific_any_rank": ("rescue: type-specific marker significantly increased (any rank), replacing an Unknown or "
                          "shared-marker result of the top-25 rule", "rescued"),
    "low_shared_only": ("shared markers only among the top 25 increased proteins", "low (shared markers only)"),
    "unknown_no_marker_in_top_de": ("no marker among the top 25; no type-specific marker significantly increased",
                                    "Unknown"),
}
TOP25_CONF = {"high": "type-specific marker", "medium": "type-specific marker",
              "low_shared_only": "shared markers only", "unknown_no_marker_in_top_de": "no marker"}
EXPECTED = {"PCT": 160, "PST": 158, "TAL": 138, "Unknown": 109, "DCT1": 58, "PC": 38, "JG": 24, "Endothelial": 19,
            "IC-B": 8}                         # the project notes, kidney_t15_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="report: per-cluster evidence for the revised kidney annotation.")
    cc.add_common_args(parser)
    return parser.parse_args()


def disp(label: str) -> str:
    return DISPLAY.get(label, label)


def fmt_q(q: float) -> str:
    return f"{q:.1e}" if q < 0.001 else f"{q:.3f}"


def top25_marker_text(de: pd.DataFrame, cluster: int, markers: list[str]) -> tuple[str, list[str]]:
    """Rank, AUC, q and detection of each overlap marker in the cluster's top-25 DE list."""
    sub = de[(de["clustering"] == CLUSTERING) & (de["cluster"] == cluster)].set_index("protein")
    parts, missing = [], []
    for m in markers:
        if m not in sub.index or not bool(sub.loc[m, "in_top"]):
            missing.append(m)
            continue
        r = sub.loc[m]
        parts.append(f"{m} (rank {int(r['rank_up'])}; AUC {r['auc']:.3f}; q {fmt_q(float(r['q_bh']))}; "
                     f"detected {r['pct_detected_cluster']:.2f} vs {r['pct_detected_rest']:.2f})")
    return "; ".join(parts), missing


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)
    cl = pd.read_csv(run.use_input(CLUSTERS, "kidney_annotation adopted clusters (kidney_annotation_chosen_labels.py, a_endo)"))
    de = pd.read_csv(run.use_input(DE_TOP25, "kidney_annotation top-25 DE per cluster (kidney_annotation_resolution_scan.py)"))
    pc = pd.read_csv(run.use_input(PER_CELL, "kidney_annotation adopted labels per cell (a_endo)"))
    cl = cl.sort_values("cluster").reset_index(drop=True)
    checks: list[dict] = []

    def check(name: str, observed, expected, ok: bool) -> None:
        checks.append({"check": name, "observed": str(observed), "expected": str(expected), "pass": bool(ok)})
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {observed}")

    rows, missing_all = [], []
    for r in cl.itertuples(index=False):
        route, conf = ROUTE[r.rankfree_confidence]
        overlap = [] if pd.isna(r.top25_overlap) else [m for m in str(r.top25_overlap).split(";") if m]
        if r.rankfree_confidence == "specific_any_rank":
            evidence = str(r.rescue_markers)
            others = [c for c in str(r.rescue_candidates).split(";") if c and not c.startswith(f"{r.rankfree_label} ")]
        elif overlap:
            evidence, missing = top25_marker_text(de, int(r.cluster), overlap)
            missing_all += [f"{r.cluster}:{m}" for m in missing]
            others = []
        else:
            evidence, others = "", []
        rows.append({
            "cluster": int(r.cluster), "label": disp(r.rankfree_label), "label_internal": r.rankfree_label,
            "cells": int(r.n_cells), "median_proteins_per_cell": float(r.median_protein_ids),
            "evidence_route": route, "confidence": conf, "marker_evidence": evidence,
            "top25_rule_label": disp(r.top25_label), "top25_rule_basis": TOP25_CONF[r.top25_confidence],
            "other_candidate_types_best_marker_auc": "; ".join(disp(o) for o in others),
        })
    out = pd.DataFrame(rows)

    # checks
    check("clusters", len(out), 23, len(out) == 23)
    check("cells in clusters", int(out["cells"].sum()), 712, int(out["cells"].sum()) == 712)
    by_label = out.groupby("label_internal")["cells"].sum().astype(int).to_dict()
    check("cells per label (= the project notes kidney_t15_label counts)", by_label, EXPECTED, by_label == EXPECTED)
    n_de = de[de["clustering"] == CLUSTERING].groupby("cluster")["n_cells_cluster"].first().astype(int)
    same = all(int(n_de.get(c, -1)) == n for c, n in zip(out["cluster"], out["cells"]))
    check("cluster sizes equal in the DE table (same clustering)", same, True, same)
    pc_lab = pc.groupby("t15_cluster")["kidney_t15_label"].agg(lambda s: ";".join(sorted(set(s))))
    agree = all(pc_lab.get(c) == lab for c, lab in zip(out["cluster"], out["label_internal"]))
    check("every cell carries its cluster's label (per-cell file)", agree, True, agree)
    no_ev = out[(out["label_internal"] != "Unknown") & (out["marker_evidence"] == "")]["cluster"].tolist()
    check("every labelled cluster has marker evidence with statistics", no_ev, "[]", not no_ev)
    check("top-25 overlap markers found in the cluster's top-25 DE list", missing_all, "[]", not missing_all)
    routes = out["confidence"].value_counts().to_dict()
    cells_route = out.groupby("confidence")["cells"].sum().astype(int).to_dict()
    check("clusters / cells per route (information)", f"{routes} / {cells_route}", "-", True)

    run.table(out.drop(columns=["label_internal"]), STEP)
    summary = "; ".join(f"{k}: {routes[k]} clusters, {cells_route[k]} cells" for k in
                        ("medium", "high", "rescued", "low (shared markers only)", "Unknown") if k in routes)
    md = ["# Revised kidney annotation: evidence per cluster", "", f"Routes: {summary}.", "",
          "(`scripts/report_annotation_evidence.py`) from the kidney_annotation outputs of the "
          "adopted labels. Leiden 2.0 on the stored neighbour graph (k = 10, 20 PCs), seed 42. Each cluster was "
          "compared with all other cells (Wilcoxon rank-sum on ln(1 + intensity); proteins detected in >= 20% of the "
          "cluster's cells; increased = BH q < 0.05, AUC > 0.5). The label is the kidney V3 (+ endothelial row) cell "
          "type whose markers are enriched among the cluster's 25 most increased proteins (top-25 rule). A cluster "
          "left Unknown or labelled from shared markers only (markers listed for >= 2 types) by that rule takes a "
          "type whose type-specific marker is significantly increased at any rank (rescue); the top-25 rule's result "
          "is given for every cluster. Rank = position among the cluster's significantly increased proteins, "
          "ordered by AUC. Detected = fraction of cells with intensity > 0 in the cluster vs the other cells. Other "
          "candidate types: types that a rescue could also have assigned, with the AUC of their best type-specific "
          "marker. Display names: DCT1 -> DCT, PC -> CNT/PC.", "",
          cc.markdown_table(out.drop(columns=["label_internal"])), ""]
    run.text("\n".join(md), STEP)
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    ok = all(c["pass"] for c in checks)
    run.finish({"clustering": CLUSTERING, "display_names": DISPLAY, "checks_pass": ok},
               status="completed" if ok else "completed_with_failed_checks")
    if not ok:
        raise SystemExit("A check failed; see report_annotation_evidence_checks.csv.")


if __name__ == "__main__":
    main()
