#!/usr/bin/env python
"""kidney_rna step 4: kidney SCP cells mapped to Mouse Kidney Atlas centroids (cosine similarity).

Analysis plan,
step 4. Logic of the earlier integrate_scp_with_mka.py (re-implemented, not imported):
- shared features: atlas genes matching a quantified protein (case-insensitive; 3,569);
- SCP: canonical .X (per-protein z-scored ln(1 + intensity), missing = 0, clip 10) on the shared proteins, then
  z-scored per feature across the 712 cells;
- RNA centroids: per atlas class, mean log-normalized expression (10,000 per cell, log1p; kidney_rna pseudobulk),
  standardized with the atlas-wide gene mean and SD (all 141,401 cells, SD ddof = 0, SD < 1e-8 -> 1);
- cosine similarity of every cell with every centroid; top label, score, second label, margin (top - second);
  low confidence if score < min_score or margin < min_margin.
Levels: Celltype and Celltype_finest. Deviation from the earlier analysis: the atlas class 'Unknown' is not a centroid (it is
unlabelled). A run with the Unknown centroid reproduces the earlier per-cell table first (method check).
Thresholds (score / margin): 0.20 / 0.00 and 0.20 / 0.05.

The per-cell predictions do not depend on the SCP labels; the labels enter only the summaries:
kidney_t15_label (primary) and kidney_pre_label; kidney_reannot_label for the comparison only. Expected atlas
classes per label come from the approved mapping (kidney_rna_celltype_mapping.csv): strict, and strict plus the
accepted mixed classes. No interpretive sentence is generated.

Inputs (read-only): kidney_object canonical kidney object, kidney_annotation labels, kidney_rna mapping table and atlas pseudobulk; the earlier
tables/mka_mapping_results_full_best_match_minScore0p20_margin0p00_mouseGeneCase/scp_mka_prediction_table.csv
(reproduction check only).
Outputs: results/tables/kidney_rna/kidney_rna_mapping_predictions_per_cell.csv, kidney_rna_mapping_summary.csv,
         kidney_rna_mapping_confusion.csv, kidney_rna_mapping_label_effect.csv, kidney_rna_mapping_centroids.csv,
         kidney_rna_mapping_checks.csv; results/figures/kidney_rna/kidney_rna_mapping_confusion_{celltype,finest}_
         {0p20_0p00,0p20_0p05}.pdf/.png

Command (project root):  python scripts/kidney_rna_reference_mapping.py
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

MAPPING = cc.out_path(L.TASK, "tables", "celltype_mapping", ".csv")
EARLIER_TABLE = (cc.PROJECT_ROOT / "tables" / "mka_mapping_results_full_best_match_minScore0p20_margin0p00_mouseGeneCase"
               / "scp_mka_prediction_table.csv")
LEVELS = {"celltype": "Celltype", "finest": "Celltype_finest"}
MAPPING_COLS = {"celltype": ("atlas_celltype_expected", "atlas_celltype_mixed_accepted"),
                "finest": ("atlas_finest_expected", "atlas_finest_mixed_accepted")}
THRESHOLDS = {"0.20/0.00": (0.20, 0.00), "0.20/0.05": (0.20, 0.05)}
LOW = "Low confidence"
REPRO_TOL = 1e-4
SEQ = LinearSegmentedColormap.from_list("t07_seq", ["#fcfcfb", *t3.BLUE_RAMP])
INK = t3.INK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_rna step 4: kidney SCP -> atlas centroid mapping.")
    cc.add_common_args(parser)
    return parser.parse_args()


def stop(run: cc.Run, checks: t5.Checks) -> None:
    run.table(checks.frame(), "mapping_checks")
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit("A check failed; see kidney_rna_mapping_checks.csv.")


def standardize(x: np.ndarray) -> np.ndarray:
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    z = (x - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z


def centroids(pb, level: str, classes: list[str], genes: list[str]) -> tuple[np.ndarray, list[int]]:
    glob = L.aggregate(pb, np.ones(pb.n_obs, dtype=bool), genes)
    gmean, gsd = glob["mean_lognorm"].to_numpy(), glob["sd_lognorm"].to_numpy()
    gsd = np.where(~np.isfinite(gsd) | (gsd < 1e-8), 1.0, gsd)
    rows, ns = [], []
    for c in classes:
        agg = L.aggregate(pb, L.group_mask(pb, **{level: c}), genes)
        rows.append((agg["mean_lognorm"].to_numpy() - gmean) / gsd)
        ns.append(int(agg["n_cells"].iloc[0]))
    z = np.vstack(rows)
    z[~np.isfinite(z)] = 0.0
    return z, ns


def predict(scp_z: np.ndarray, cent_z: np.ndarray, classes: list[str]) -> pd.DataFrame:
    a = scp_z / np.maximum(np.linalg.norm(scp_z, axis=1, keepdims=True), 1e-12)
    b = cent_z / np.maximum(np.linalg.norm(cent_z, axis=1, keepdims=True), 1e-12)
    sims = a @ b.T
    order = np.argsort(-sims, axis=1, kind="stable")
    idx = np.arange(sims.shape[0])
    lab = np.asarray(classes, dtype=object)
    top, second = order[:, 0], order[:, 1]
    return pd.DataFrame({"top": lab[top], "score": sims[idx, top], "second": lab[second],
                         "second_score": sims[idx, second], "margin": sims[idx, top] - sims[idx, second]})


def conservative(pred: pd.DataFrame, min_score: float, min_margin: float) -> pd.Series:
    low = (pred["score"] < min_score) | (pred["margin"] < min_margin) | ~np.isfinite(pred["margin"])
    return pd.Series(np.where(low, LOW, pred["top"]), index=pred.index)


def split(v) -> set[str]:
    return set() if pd.isna(v) or str(v) == "" else set(str(v).split(";"))


def heatmap_figure(run, conf: pd.DataFrame, level: str, thr: str, labels: pd.DataFrame) -> None:
    sub = conf[(conf["level"] == LEVELS[level]) & (conf["assignment"] == f"conservative {thr}")]
    if sub.empty:
        raise ValueError(f"no confusion rows for {level} / {thr}")
    cols_present = sorted(set(sub.loc[sub["n"] > 0, "atlas_label"]))
    cols = L.atlas_order([c for c in cols_present if c != LOW]) + ([LOW] if LOW in cols_present else [])
    panels = [c for c in ("kidney_t15_label", "kidney_pre_label")]
    nrows = [len(L.row_order(labels[c].unique())) for c in panels]
    h_mm = 14 + sum(n * 4.2 for n in nrows) + 18 * len(panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(180 / 25.4, h_mm / 25.4),
                             gridspec_kw={"height_ratios": nrows}, layout="constrained")
    for ax, col in zip(axes, panels):
        rows = L.row_order(labels[col].unique())
        s = sub[sub["label_column"] == col]
        cnt = s.pivot_table(index="scp_label", columns="atlas_label", values="n", aggfunc="sum",
                            fill_value=0).reindex(index=rows, columns=cols, fill_value=0)
        frac = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        im = ax.imshow(frac.to_numpy(float), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        for (r, c), v in np.ndenumerate(cnt.to_numpy()):
            if v:
                ax.text(c, r, str(int(v)), ha="center", va="center", fontsize=6,
                        color="white" if frac.to_numpy(float)[r, c] > 0.6 else INK["primary"])
        ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(rows)), [f"{r} ({int(cnt.loc[r].sum())})" for r in rows], fontsize=7)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        disp = "kidney_annotation labels" if col == "kidney_t15_label" else "Fig 2A labels"
        ax.set_title(f"{disp} -> atlas {LEVELS[level]} (score >= {thr.split('/')[0]}, margin >= {thr.split('/')[1]})",
                     fontsize=7, loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("fraction of the SCP label's cells", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    run.figure(fig, f"mapping_confusion_{level}_{thr.replace('.', 'p').replace('/', '_')}",
               f"SCP label x atlas {LEVELS[level]} (conservative assignment, {thr})")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "reference_mapping", args)
    checks = t5.Checks(L.DATASET)
    adata, labels = L.load_kidney(run, checks)
    mapping = pd.read_csv(run.use_input(MAPPING, "kidney_rna approved SCP label -> atlas mapping (step 2)"))
    approved = mapping["status"].astype(str).str.contains("approved by the authors").all()
    checks.add("mapping table approved by the authors", f"{approved}", approved)
    pb = L.read_pseudobulk(run)
    if not checks.ok:
        stop(run, checks)

    match = L.match_proteins(adata.var_names, pb.var_names)
    checks.add("shared features (protein <-> atlas gene)", f"{len(match)}", len(match) == L.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES,
               f"= {L.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES}")
    pidx = pd.Index(adata.var_names.astype(str)).get_indexer(match["protein"])
    genes = match["atlas_gene"].tolist()
    scp_z = standardize(cc.dense(adata.X)[:, pidx].astype(np.float64))
    cells = labels.index.astype(str)

    per_cell = labels[["kidney_t15_label", "kidney_pre_label", "kidney_reannot_label", "n_detected_proteins"]].copy()
    centroid_rows, preds = [], {}
    for lv, level in LEVELS.items():
        classes_all = L.atlas_order(pb.obs[level].astype(str).unique())
        classes = [c for c in classes_all if c != L.ATLAS_UNKNOWN]
        if lv == "celltype" and EARLIER_TABLE.exists():
            # the earlier analysis resolved duplicated protein symbols by the highest .X mean; use its protein per gene here.
            cxf = pd.read_csv(run.use_input(EARLIER_TABLE.with_name("shared_features.csv"),
                                            "the earlier analysis shared features (reproduction check)"))
            cx_prot = dict(zip(cxf["shared_feature"].astype(str).str.lower(), cxf["scp_original_id"].astype(str)))
            prot_cx = [cx_prot.get(g.lower(), p) for g, p in zip(genes, match["protein"])]
            n_diff = sum(a != b for a, b in zip(prot_cx, match["protein"]))
            checks.add("shared genes for which the earlier analysis used another protein entry (duplicated symbols)",
                       f"{n_diff}: " + ", ".join(f"{g} -> {a}" for g, a, b in zip(genes, prot_cx, match["protein"])
                                                 if a != b), None)
            pidx_cx = pd.Index(adata.var_names.astype(str)).get_indexer(prot_cx)
            scp_z_cx = standardize(cc.dense(adata.X)[:, pidx_cx].astype(np.float64))
            cz_all, _ = centroids(pb, level, classes_all, genes)
            earlier_like = predict(scp_z_cx, cz_all, classes_all)
            earlier_like.index = cells
            if True:
                cx = pd.read_csv(run.use_input(EARLIER_TABLE, "the earlier analysis per-cell MKA predictions (reproduction check)"),
                                 dtype={"cell_id": str}).set_index("cell_id").reindex(cells)
                same_top = float((cx["MKA_predicted_label"].astype(str) == earlier_like["top"]).mean())
                d_score = float(np.nanmax(np.abs(cx["MKA_prediction_score"] - earlier_like["score"])))
                d_margin = float(np.nanmax(np.abs(cx["MKA_prediction_margin"] - earlier_like["margin"])))
                checks.add("with the atlas Unknown centroid and the earlier protein entries, the earlier per-cell table is "
                           "reproduced (same top label; max abs difference of score, margin)",
                           f"{same_top:.4f}; {d_score:.1e}; {d_margin:.1e}",
                           same_top == 1.0 and max(d_score, d_margin) < REPRO_TOL, f"1; < {REPRO_TOL:g}")
                per_cell["earlier_like_top_celltype_incl_unknown"] = earlier_like["top"]
            if not checks.ok:
                stop(run, checks)
        elif lv == "celltype":
            checks.add("the earlier analysis per-cell table for the reproduction check", "not found", None)
        cz, ns = centroids(pb, level, classes, genes)
        pred = predict(scp_z, cz, classes)
        pred.index = cells
        preds[lv] = pred
        for c, n in zip(classes, ns):
            centroid_rows.append({"level": level, "atlas_class": c, "n_atlas_cells": n})
        for k in ("top", "score", "second", "second_score", "margin"):
            per_cell[f"{lv}_{k}"] = pred[k]
        for thr, (ms, mm) in THRESHOLDS.items():
            per_cell[f"{lv}_conservative_{thr.replace('/', '_')}"] = conservative(pred, ms, mm)
            n_conf = int((conservative(pred, ms, mm) != LOW).sum())
            checks.add(f"{level}: cells with a confident assignment at {thr}", f"{n_conf} of {len(pred)} "
                       f"({n_conf / len(pred):.1%})", None)
        rho = stats.spearmanr(pred["score"], labels["n_detected_proteins"])
        checks.add(f"{level}: Spearman rho of top score with protein IDs per cell (all 712)",
                   f"{rho.statistic:.3f} (p {rho.pvalue:.1e})", None)
        checks.add(f"{level}: centroids (Unknown excluded); top labels used",
                   f"{len(classes)}; " + "; ".join(f"{k} {v}" for k, v in pred["top"].value_counts().items()), None)

    # summaries ---------------------------------------------------------------------------------------
    mp = mapping.set_index(["label_column", "scp_label"])
    summary, confusion = [], []
    assignments = {"top": None, **{f"conservative {t}": t for t in THRESHOLDS}}
    for lv, level in LEVELS.items():
        pred = preds[lv]
        exp_col, mix_col = MAPPING_COLS[lv]
        for col in L.LABEL_COLUMNS:
            for lab in L.row_order(labels[col].unique()):
                m = (labels[col] == lab).to_numpy()
                p = pred[m]
                exp = split(mp.loc[(col, lab), exp_col])
                mix = exp | split(mp.loc[(col, lab), mix_col])
                for thr, (ms, mm) in THRESHOLDS.items():
                    cons = conservative(p, ms, mm)
                    conf = cons != LOW
                    dom = p["top"].value_counts()
                    domc = cons.value_counts()
                    summary.append({
                        "level": level, "threshold": thr, "label_column": col, "label_role": L.LABEL_ROLE[col],
                        "scp_label": lab, "n_cells": int(m.sum()),
                        "expected_atlas": ";".join(sorted(exp)), "expected_incl_mixed": ";".join(sorted(mix)),
                        "dominant_top": dom.index[0], "dominant_top_fraction": dom.iloc[0] / m.sum(),
                        "dominant_conservative": domc.index[0], "dominant_conservative_fraction": domc.iloc[0] / m.sum(),
                        "expected_fraction_top": p["top"].isin(exp).mean() if exp else np.nan,
                        "expected_fraction_top_incl_mixed": p["top"].isin(mix).mean() if exp else np.nan,
                        "expected_fraction_confident": (conf & p["top"].isin(exp)).mean() if exp else np.nan,
                        "expected_fraction_confident_incl_mixed": (conf & p["top"].isin(mix)).mean() if exp else np.nan,
                        "low_confidence_fraction": 1 - conf.mean(), "mean_score": p["score"].mean(),
                        "median_score": p["score"].median(), "mean_margin": p["margin"].mean(),
                        "median_margin": p["margin"].median()})
                for name, thr in assignments.items():
                    a = p["top"] if thr is None else conservative(p, *THRESHOLDS[thr])
                    vc = a.value_counts()
                    for atl, n in vc.items():
                        confusion.append({"level": level, "assignment": name, "label_column": col, "scp_label": lab,
                                          "atlas_label": atl, "n": int(n), "fraction_of_label": n / m.sum(),
                                          "is_expected": atl in exp, "is_mixed_accepted": atl in (mix - exp)})
    summary = pd.DataFrame(summary)
    confusion = pd.DataFrame(confusion)

    # label effect: kidney_annotation and reannot vs pre, per expected atlas identity (labels pooled) -------------------------
    effect = []
    for lv, level in LEVELS.items():
        pred = preds[lv]
        exp_col, mix_col = MAPPING_COLS[lv]
        idents: dict[str, dict[str, list[str]]] = {}
        for col in L.LABEL_COLUMNS:
            sub = mapping[mapping["label_column"] == col]
            for r in sub.itertuples():
                key = ";".join(sorted(split(getattr(r, exp_col))))
                if key:
                    idents.setdefault(key, {}).setdefault(col, []).append(r.scp_label)
        for key, by_col in idents.items():
            exp = set(key.split(";"))
            for variant in ("strict", "incl_mixed"):
                for name, thr in assignments.items():
                    rec = {"level": level, "expected_atlas": key, "variant": variant, "assignment": name}
                    vals = {}
                    for col in L.LABEL_COLUMNS:
                        labs = by_col.get(col, [])
                        short = {"kidney_t15_label": "t15", "kidney_pre_label": "pre",
                                 "kidney_reannot_label": "reannot"}[col]
                        rec[f"labels_{short}"] = ";".join(labs)
                        m = labels[col].isin(labs).to_numpy()
                        rec[f"n_{short}"] = int(m.sum())
                        if not labs:
                            vals[short] = np.nan
                        else:
                            mix = exp | set().union(*[split(mp.loc[(col, lb), mix_col]) for lb in labs])
                            target = exp if variant == "strict" else mix
                            a = pred.loc[m, "top"] if thr is None else conservative(pred[m], *THRESHOLDS[thr])
                            vals[short] = float(a.isin(target).mean())
                        rec[f"fraction_{short}"] = vals[short]
                    for short in ("t15", "reannot"):
                        d = vals[short] - vals["pre"]
                        rec[f"delta_{short}_vs_pre_pp"] = 100 * d
                        rec[f"direction_{short}_vs_pre"] = ("not comparable" if not np.isfinite(d) else
                                                             "higher" if d > 0 else "lower" if d < 0 else "equal")
                    effect.append(rec)
    effect = pd.DataFrame(effect)
    for short in ("t15", "reannot"):
        e = effect[(effect["level"] == "Celltype") & (effect["variant"] == "strict") & (effect["assignment"] == "top")]
        checks.add(f"Celltype, strict, top label: {short} vs pre (expected identity: direction, delta pp)", "; ".join(
            f"{r.labels_pre or '-'}->{getattr(r, 'labels_' + short) or '-'}: {getattr(r, 'direction_' + short + '_vs_pre')}"
            f"{'' if not np.isfinite(getattr(r, 'delta_' + short + '_vs_pre_pp')) else ' %+.1f' % getattr(r, 'delta_' + short + '_vs_pre_pp')}"
            for r in e.itertuples()), None)

    for lv in LEVELS:
        for thr in THRESHOLDS:
            heatmap_figure(run, confusion, lv, thr, labels)

    per_cell = per_cell.rename_axis("cell_id").reset_index()
    run.table(per_cell, "mapping_predictions_per_cell")
    run.table(summary, "mapping_summary")
    run.table(confusion, "mapping_confusion")
    run.table(effect, "mapping_label_effect")
    run.table(pd.DataFrame(centroid_rows), "mapping_centroids")
    run.table(checks.frame(), "mapping_checks")
    run.finish({"levels": list(LEVELS.values()), "thresholds": {k: list(v) for k, v in THRESHOLDS.items()},
                "unknown_centroid": "excluded (deviation from the earlier analysis; reproduction run with it included)",
                "scp_matrix": "canonical .X on the shared proteins, z-scored per feature across 712 cells",
                "rna_centroids": "class mean log-normalized expression, standardized by atlas-wide gene mean / SD",
                "reproduction_tolerance": REPRO_TOL, "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
