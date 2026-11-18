#!/usr/bin/env python
"""kidney_rna step 5: RNA-protein detection and concordance across matched kidney compartments.

Analysis plan,
step 5 (reviewer R1: "where RNA and protein correlate, where only RNA can be detected"). Logic of the earlier
rna_protein_concordance_analysis.py (re-implemented), with one change the plan asks for:

- Compartment = an SCP label with an atlas counterpart in the approved mapping (kidney_rna_celltype_mapping.csv):
  protein = the SCP cells with that label; RNA = the atlas cells of its expected Celltypes (mixed classes not
  included). kidney_t15_label (primary; JG and Unknown have no counterpart) and kidney_pre_label (JG and Unknown
  excluded). RNA subsets: all atlas cells (primary) and nuclei only (snRNA-seq; supplement); a compartment needs
  >= 10 atlas cells in the subset.
- Features: the 3,569 atlas genes matching a quantified protein (case-insensitive).
- Per feature x compartment: detection fraction (protein: intensity > 0; RNA: counts > 0), protein mean
  ln(1 + intensity) (missing = 0), RNA mean log-normalized expression (10,000 per cell, log1p).
- Same detection rule for both modalities (the change): a feature is detected in a modality if its detection
  fraction is >= t in at least one compartment, t = 5%, 10%, 20%. (the earlier analysis used the compartment maximum for RNA but
  the all-cell protein coverage for protein.)
- Per feature: Spearman rho of RNA mean vs protein mean across the compartments. Classes (the earlier analysis cut-offs):
  detected in both and rho >= 0.5 concordant; 0.2 <= rho < 0.5 intermediate; rho < 0.2 (or undefined) discordant;
  RNA-only; protein-only; neither.
- Per compartment: Spearman rho across features of RNA mean vs protein mean, for the features detected in both
  modalities in that compartment (>= t), and for all shared features.
- Canonical markers: every V3 + endothelial-row marker with its class (or why it has none).
No interpretive sentence is generated.

Inputs (read-only): kidney_object canonical kidney object, kidney_annotation labels, kidney_rna mapping table and atlas pseudobulk, V3 workbook.
Outputs: results/tables/kidney_rna/kidney_rna_concordance_{long,features,class_counts,compartments,markers,checks}.csv
         results/figures/kidney_rna/kidney_rna_concordance_{class_counts,detection,correlations}.pdf/.png

Command (project root):  python scripts/kidney_rna_concordance.py
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
import qc_mito_lib as t3
import cluster_de_lib as t5
import kidney_rna_lib as L

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

MAPPING = cc.out_path(L.TASK, "tables", "celltype_mapping", ".csv")
LABELS_USED = ("kidney_t15_label", "kidney_pre_label")
SHORT = {"kidney_t15_label": "t15", "kidney_pre_label": "pre"}
DISPLAY = {"kidney_t15_label": "kidney_annotation labels", "kidney_pre_label": "Fig 2A labels"}
RNA_SOURCES = {"all": None, "nuclei": "Nuclei"}
PRIMARY_SOURCE = "all"
THRESHOLDS = (0.05, 0.10, 0.20)
CONCORDANT, WEAK = 0.50, 0.20
MIN_ATLAS_CELLS = 10
CLASSES = ("Detected in both; concordant", "Detected in both; intermediate", "Detected in both; discordant",
           "RNA-detected only", "Protein-detected only", "Detected in neither")
CLASS_COLORS = dict(zip(CLASSES, (t3.BLUE_RAMP[5], t3.BLUE_RAMP[3], t3.BLUE_RAMP[1], t3.SERIES[1], t3.SERIES[2],
                                  t3.INK["background_cells"])))
INK = t3.INK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_rna step 5: RNA-protein concordance across compartments.")
    cc.add_common_args(parser)
    return parser.parse_args()


def thr_text(t: float) -> str:
    return f"{int(round(t * 100))}%"


def classify(max_rna: float, max_prot: float, rho: float, t: float) -> str:
    r, p = max_rna >= t, max_prot >= t
    if r and p:
        if np.isfinite(rho) and rho >= CONCORDANT:
            return CLASSES[0]
        if np.isfinite(rho) and rho >= WEAK:
            return CLASSES[1]
        return CLASSES[2]
    if r:
        return CLASSES[3]
    if p:
        return CLASSES[4]
    return CLASSES[5]


def null_rho(n: int) -> np.ndarray:
    """Spearman rho of every permutation of n ranks against 1..n (exact null without ties)."""
    from itertools import permutations

    base = np.arange(n)
    d2 = np.array([np.sum((base - np.asarray(p)) ** 2) for p in permutations(base)], dtype=float)
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def null_table(ns) -> pd.DataFrame:
    rows = []
    for n in sorted(set(ns)):
        r = null_rho(n)
        rows.append({"n_compartments": n, "n_permutations": len(r),
                     "p_rho_ge_0.5_concordant": float(np.mean(r >= CONCORDANT - 1e-12)),
                     "p_rho_0.2_to_0.5_intermediate": float(np.mean((r >= WEAK - 1e-12) & (r < CONCORDANT - 1e-12))),
                     "p_rho_lt_0.2_discordant": float(np.mean(r < WEAK - 1e-12)), "median_rho": float(np.median(r))})
    return pd.DataFrame(rows)


def compartments(mapping: pd.DataFrame, col: str, labels: pd.DataFrame) -> list[tuple[str, list[str]]]:
    sub = mapping[mapping["label_column"] == col].set_index("scp_label")
    out = []
    for lab in L.row_order(labels[col].unique()):
        exp = sub.loc[lab, "atlas_celltype_expected"]
        if isinstance(exp, str) and exp:
            out.append((lab, exp.split(";")))
    return out


def figures(run, features: pd.DataFrame, counts: pd.DataFrame, comp: pd.DataFrame) -> None:
    src = PRIMARY_SOURCE
    # 1. class counts
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 62 / 25.4), sharey=True, layout="constrained")
    for ax, col in zip(axes, LABELS_USED):
        c = counts[(counts["label_column"] == col) & (counts["rna_source"] == src)]
        for i, t in enumerate(THRESHOLDS):
            left = 0
            ct = c[np.isclose(c["threshold"], t)].set_index("class")["n_features"].reindex(CLASSES).fillna(0)
            for cl in CLASSES:
                n = ct[cl]
                ax.barh(i, n, left=left, color=CLASS_COLORS[cl], edgecolor="white", linewidth=0.4, height=0.7)
                if n >= 160:
                    dark = cl in CLASSES[:2]
                    ax.text(left + n / 2, i, f"{int(n)}", ha="center", va="center", fontsize=6,
                            color="white" if dark else INK["primary"])
                left += n
        ax.set_yticks(range(len(THRESHOLDS)), [f">= {thr_text(t)}" for t in THRESHOLDS])
        ax.set_xlabel(f"features (n = {int(ct.sum())} shared genes / proteins)")
        ax.set_title(f"{DISPLAY[col]}: {comp[(comp['label_column'] == col) & (comp['rna_source'] == src)]['compartment'].nunique()} "
                     "compartments", loc="left", fontsize=7, fontweight="bold")
        ax.invert_yaxis()
    axes[0].set_ylabel("detection threshold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASSES]
    fig.legend(handles, CLASSES, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
    run.figure(fig, "concordance_class_counts", "feature classes per detection threshold (all atlas cells)")
    plt.close(fig)

    # 2. detection scatter at 10%
    t = 0.10
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 92 / 25.4), sharex=True, sharey=True, layout="constrained")
    for ax, col in zip(axes, LABELS_USED):
        f = features[(features["label_column"] == col) & (features["rna_source"] == src)
                     & np.isclose(features["threshold"], t)]
        for cl in CLASSES[::-1]:
            g = f[f["class"] == cl]
            ax.scatter(g["max_rna_detection"], g["max_protein_detection"], s=4, color=CLASS_COLORS[cl],
                       linewidths=0, label=cl, rasterized=True)
        mk = f[f["marker_listed_for"].fillna("") != ""]
        ax.scatter(mk["max_rna_detection"], mk["max_protein_detection"], s=12, facecolors="none",
                   edgecolors=INK["primary"], linewidths=0.5, label="V3 / endothelial-row marker")
        for tt in THRESHOLDS:
            ax.axvline(tt, color=INK["muted"], lw=0.4, ls=":")
            ax.axhline(tt, color=INK["muted"], lw=0.4, ls=":")
        ax.set_xlabel("RNA: highest detection fraction across compartments")
        ax.set_title(f"{DISPLAY[col]} (classes at {thr_text(t)})", loc="left", fontsize=7, fontweight="bold")
    axes[0].set_ylabel("protein: highest detection fraction across compartments")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=4, frameon=False, fontsize=7, markerscale=2)
    run.figure(fig, "concordance_detection", "highest detection fraction per feature, RNA vs protein (10%)")
    plt.close(fig)

    # 3. correlations
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 62 / 25.4), layout="constrained")
    bins = np.linspace(-1, 1, 21)
    for ax, col in zip(axes[:2], LABELS_USED):
        f = features[(features["label_column"] == col) & (features["rna_source"] == src)]
        for t, color in zip(THRESHOLDS, (t3.BLUE_RAMP[2], t3.BLUE_RAMP[4], t3.BLUE_RAMP[6])):
            r = f.loc[np.isclose(f["threshold"], t) & f["class"].str.startswith("Detected in both"), "spearman_rho"]
            ax.hist(r.dropna(), bins=bins, histtype="step", color=color, lw=1.0,
                    label=f">= {thr_text(t)} (n = {r.notna().sum()}; median {r.median():.2f})")
        n_comp = int(f["n_compartments"].iloc[0])
        r10 = f.loc[np.isclose(f["threshold"], 0.10) & f["class"].str.startswith("Detected in both"), "spearman_rho"]
        nr = null_rho(n_comp)
        ax.hist(nr, bins=bins, weights=np.full(len(nr), r10.notna().sum() / len(nr)), histtype="step",
                color=INK["muted"], lw=0.8, ls="--", label=f"chance (all rank permutations), n as >= 10%")
        ax.set_xlabel(f"per-feature Spearman rho across {n_comp} compartments")
        ax.set_ylabel("features detected in both")
        ax.set_title(DISPLAY[col], loc="left", fontsize=7, fontweight="bold")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax = axes[2]
    for k, col in enumerate(LABELS_USED):
        c = comp[(comp["label_column"] == col) & (comp["rna_source"] == src)]
        for i, t in enumerate(THRESHOLDS):
            v = c[np.isclose(c["threshold"], t)]["spearman_rho_detected_both"]
            x = i + (k - 0.5) * 0.3
            ax.scatter(np.full(len(v), x), v, s=8, color=t3.SERIES[k], linewidths=0,
                       label=DISPLAY[col] if i == 0 else None)
    ax.set_xticks(range(len(THRESHOLDS)), [f">= {thr_text(t)}" for t in THRESHOLDS])
    ax.set_ylabel("per-compartment Spearman rho across features")
    ax.set_xlabel("features detected in both (threshold)")
    ax.set_title("Per compartment", loc="left", fontsize=7, fontweight="bold")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    run.figure(fig, "concordance_correlations", "per-feature and per-compartment Spearman correlations")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "concordance", args)
    checks = t5.Checks(L.DATASET)
    adata, labels = L.load_kidney(run, checks)
    mapping = pd.read_csv(run.use_input(MAPPING, "kidney_rna approved SCP label -> atlas mapping (step 2)"))
    approved = mapping["status"].astype(str).str.contains("approved by the authors").all()
    checks.add("mapping table approved by the authors", f"{approved}", approved)
    pb = L.read_pseudobulk(run)
    rows_markers = L.kidney_marker_rows(run)
    by_gene = L.markers_by_gene(rows_markers)
    if not checks.ok:
        raise SystemExit("mapping table not approved")

    match = L.match_proteins(adata.var_names, pb.var_names)
    checks.add("shared features (protein <-> atlas gene)", f"{len(match)}", len(match) == L.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES)
    pidx = pd.Index(adata.var_names.astype(str)).get_indexer(match["protein"])
    genes = match["atlas_gene"].tolist()
    inten = cc.dense(adata.layers["intensity"])[:, pidx]
    ln1p = cc.dense(adata.layers["ln1p_intensity"])[:, pidx].astype(np.float64)
    ok_ln = bool(np.allclose(ln1p, np.log1p(inten), atol=1e-5))
    checks.add("layers['ln1p_intensity'] equals ln(1 + intensity) on the shared proteins", f"{ok_ln}", ok_ln)

    long_rows, feat_rows, comp_rows, excluded = [], [], [], []
    for col in LABELS_USED:
        comps = compartments(mapping, col, labels)
        for src, source in RNA_SOURCES.items():
            per = []
            for lab, classes in comps:
                crit = {"Celltype": classes, **({"Source": source} if source else {})}
                agg = L.aggregate(pb, L.group_mask(pb, **crit), genes)
                n_atlas = int(agg["n_cells"].iloc[0])
                m = (labels[col] == lab).to_numpy()
                if n_atlas < MIN_ATLAS_CELLS:
                    excluded.append(f"{col} / {src}: {lab} ({n_atlas} atlas cells)")
                    continue
                per.append(pd.DataFrame({
                    "label_column": col, "rna_source": src, "compartment": lab, "atlas_celltypes": ";".join(classes),
                    "n_scp_cells": int(m.sum()), "n_atlas_cells": n_atlas, "gene": genes,
                    "protein": match["protein"].to_numpy(),
                    "rna_mean_lognorm": agg["mean_lognorm"].to_numpy(), "rna_detection": agg["frac_detected"].to_numpy(),
                    "protein_mean_ln1p": ln1p[m].mean(axis=0), "protein_detection": (inten[m] > 0).mean(axis=0)}))
            lg = pd.concat(per, ignore_index=True)
            long_rows.append(lg)
            names = [c for c, _ in comps if c in set(lg["compartment"])]
            piv = {k: lg.pivot(index="gene", columns="compartment", values=k).reindex(index=genes, columns=names)
                   for k in ("rna_mean_lognorm", "rna_detection", "protein_mean_ln1p", "protein_detection")}
            rho = np.full(len(genes), np.nan)
            pval = np.full(len(genes), np.nan)
            rm, pm = piv["rna_mean_lognorm"].to_numpy(), piv["protein_mean_ln1p"].to_numpy()
            for i in range(len(genes)):
                if np.ptp(rm[i]) > 0 and np.ptp(pm[i]) > 0:
                    r = stats.spearmanr(rm[i], pm[i])
                    rho[i], pval[i] = r.statistic, r.pvalue
            max_r = piv["rna_detection"].max(axis=1).to_numpy()
            max_p = piv["protein_detection"].max(axis=1).to_numpy()
            arg_r = piv["rna_detection"].idxmax(axis=1).to_numpy()
            arg_p = piv["protein_detection"].idxmax(axis=1).to_numpy()
            for t in THRESHOLDS:
                feat_rows.append(pd.DataFrame({
                    "label_column": col, "rna_source": src, "threshold": t, "n_compartments": len(names),
                    "gene": genes, "protein": match["protein"].to_numpy(),
                    "max_rna_detection": max_r, "compartment_max_rna": arg_r,
                    "n_compartments_rna_detected": (piv["rna_detection"].to_numpy() >= t).sum(axis=1),
                    "max_protein_detection": max_p, "compartment_max_protein": arg_p,
                    "n_compartments_protein_detected": (piv["protein_detection"].to_numpy() >= t).sum(axis=1),
                    "spearman_rho": rho, "spearman_p": pval, "spearman_defined": np.isfinite(rho),
                    "class": [classify(a, b, r_, t) for a, b, r_ in zip(max_r, max_p, rho)],
                    "marker_listed_for": [";".join(by_gene.get(p, [])) for p in match["protein"]]}))
                for name in names:
                    g = lg[lg["compartment"] == name]
                    both = (g["rna_detection"] >= t) & (g["protein_detection"] >= t)
                    r_b = stats.spearmanr(g.loc[both, "rna_mean_lognorm"], g.loc[both, "protein_mean_ln1p"]) \
                        if both.sum() >= 3 else None
                    r_a = stats.spearmanr(g["rna_mean_lognorm"], g["protein_mean_ln1p"])
                    comp_rows.append({
                        "label_column": col, "rna_source": src, "threshold": t, "compartment": name,
                        "atlas_celltypes": g["atlas_celltypes"].iloc[0], "n_scp_cells": int(g["n_scp_cells"].iloc[0]),
                        "n_atlas_cells": int(g["n_atlas_cells"].iloc[0]),
                        "n_rna_detected": int((g["rna_detection"] >= t).sum()),
                        "n_protein_detected": int((g["protein_detection"] >= t).sum()),
                        "n_detected_both": int(both.sum()),
                        "spearman_rho_detected_both": r_b.statistic if r_b else np.nan,
                        "spearman_p_detected_both": r_b.pvalue if r_b else np.nan,
                        "spearman_rho_all_shared": r_a.statistic, "spearman_p_all_shared": r_a.pvalue})
    long_df = pd.concat(long_rows, ignore_index=True)
    features = pd.concat(feat_rows, ignore_index=True)
    comp = pd.DataFrame(comp_rows)
    checks.add(f"compartments excluded (< {MIN_ATLAS_CELLS} atlas cells in the subset)", "; ".join(excluded) or "none",
               None)
    counts = (features.groupby(["label_column", "rna_source", "threshold", "class"]).size()
              .rename("n_features").reset_index())
    counts["fraction"] = counts["n_features"] / len(genes)
    for col in LABELS_USED:
        for src in RNA_SOURCES:
            for t in THRESHOLDS:
                c = counts[(counts["label_column"] == col) & (counts["rna_source"] == src)
                           & np.isclose(counts["threshold"], t)].set_index("class")["n_features"]
                checks.add(f"{SHORT[col]} / RNA {src} / {thr_text(t)}: feature classes",
                           "; ".join(f"{k}: {int(c.get(k, 0))}" for k in CLASSES), int(c.sum()) == len(genes))
    undefined = features.loc[features["class"].str.startswith("Detected in both") & ~features["spearman_defined"]]
    checks.add("detected-in-both features with undefined rho (counted as discordant)", f"{len(undefined)}", None)
    null = null_table(features["n_compartments"].unique())
    for r in null.itertuples():
        checks.add(f"chance distribution of rho across {r.n_compartments} compartments (all {r.n_permutations} rank "
                   "permutations): P(concordant); P(intermediate); P(discordant)",
                   f"{r[3]:.3f}; {r[4]:.3f}; {r[5]:.3f}", None)

    # canonical markers
    mk = []
    look = L.atlas_gene_lookup(pb.var_names)
    prot_set = set(map(str, adata.var_names))
    for short, full, markers in rows_markers:
        for g in markers:
            in_prot = g in prot_set
            ag = look.get(g.lower(), "")
            for col in LABELS_USED:
                for src in RNA_SOURCES:
                    for t in THRESHOLDS:
                        f = features[(features["label_column"] == col) & (features["rna_source"] == src)
                                     & np.isclose(features["threshold"], t) & (features["protein"] == g)]
                        rec = {"marker": g, "listed_for": short, "type_full_name": full, "in_protein_matrix": in_prot,
                               "in_atlas": bool(ag), "label_column": col, "rna_source": src, "threshold": t}
                        if len(f):
                            r = f.iloc[0]
                            rec.update({"class": r["class"], "spearman_rho": r["spearman_rho"],
                                        "max_rna_detection": r["max_rna_detection"],
                                        "compartment_max_rna": r["compartment_max_rna"],
                                        "max_protein_detection": r["max_protein_detection"],
                                        "compartment_max_protein": r["compartment_max_protein"]})
                        else:
                            rec["class"] = ("not quantified as protein; not in atlas" if not in_prot and not ag else
                                            "not quantified as protein" if not in_prot else "not in atlas")
                            if ag and not in_prot:
                                sub = long_df[(long_df["label_column"] == col) & (long_df["rna_source"] == src)]
                                comps_ = [c for c, _ in compartments(mapping, col, labels)
                                          if c in set(sub["compartment"])]
                                vals = {}
                                for c_, cls in compartments(mapping, col, labels):
                                    if c_ in comps_:
                                        crit = {"Celltype": cls, **({"Source": RNA_SOURCES[src]} if RNA_SOURCES[src]
                                                                    else {})}
                                        vals[c_] = float(L.aggregate(pb, L.group_mask(pb, **crit), [ag])
                                                         ["frac_detected"].iloc[0])
                                if vals:
                                    best = max(vals, key=vals.get)
                                    rec.update({"max_rna_detection": vals[best], "compartment_max_rna": best})
                        mk.append(rec)
    markers_df = pd.DataFrame(mk)
    m10 = markers_df[(markers_df["label_column"] == "kidney_t15_label") & (markers_df["rna_source"] == PRIMARY_SOURCE)
                     & np.isclose(markers_df["threshold"], 0.10)].drop_duplicates("marker")
    checks.add("V3 + endothelial-row markers (unique), kidney_annotation / all cells / 10%: classes",
               "; ".join(f"{k}: {v}" for k, v in m10["class"].value_counts().items()), None)

    figures(run, features, counts, comp)
    run.table(long_df, "concordance_long")
    run.table(features, "concordance_features")
    run.table(counts, "concordance_class_counts")
    run.table(comp, "concordance_compartments")
    run.table(markers_df, "concordance_markers")
    run.table(null, "concordance_null")
    run.table(checks.frame(), "concordance_checks")
    run.finish({"thresholds": list(THRESHOLDS), "concordant_rho": CONCORDANT, "weak_rho": WEAK,
                "detection_rule": "same for both modalities: detection fraction >= t in at least one compartment",
                "rna_sources": {k: v or "all" for k, v in RNA_SOURCES.items()}, "primary_rna_source": PRIMARY_SOURCE,
                "min_atlas_cells": MIN_ATLAS_CELLS, "protein_mean": "ln(1 + intensity), missing = 0",
                "rna_mean": "log-normalized (10,000 per cell, log1p)", "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see kidney_rna_concordance_checks.csv.")


if __name__ == "__main__":
    main()
