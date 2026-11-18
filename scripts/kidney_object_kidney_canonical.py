#!/usr/bin/env python
"""kidney_object step 3: KIDNEY canonical labels (V3 markers) and object.



Inputs (read-only; kidney V3 only):
  data/KIDNEY/KIDNEY_768cells.csv               raw Spectronaut intensities (768 cells)
  data/KIDNEY/KIDNEY_markers_V3.xlsx            V3 markers (Supplemental Table 1, kidney tab)
  data/KIDNEY/V3/SENS_DYNR_per_cell_Leiden_2.0.csv
                                           submitted labels: cell_type_v2 (after re-annotation, Fig 2B)
                                           and cell_type_pre (before re-annotation, Fig 2A)
  data/KIDNEY/V3/ProteinIDs_byCellType_{AFTER,BEFORE}_Leiden_2.0.csv
  h5ad/KIDNEY_768cells_processed_v2.h5ad   earlier processed object; its PCA, neighbour graph, UMAP and
                                           Leiden 2.0 clusters are reused after the checks below
  h5ad/KIDNEY_768cells_*.h5ad (obs only)   the earlier analysis labels, for the reference agreement table only
Outputs:
  results/tables/kidney_object/kidney_object_kidney_*.csv
  results/tables/kidney_object/kidney_object_kidney_canonical_labels.csv
  results/tables/kidney_object/kidney_object_kidney_pre_vs_reannot_crosstab.csv
  results/h5ad/kidney_object_KIDNEY_submitted.h5ad

Checks (exit status 1 and no canonical outputs if any fails):
  rebuilt 712 x 4,054 matrix matches the object (order, scaled X, intensities);
  all 712 cells match the submitted label file; both label columns map to the
  the project notes short names with the the project notes counts; counts and median protein IDs
  per type equal the V3 AFTER/BEFORE tables; the object's named Leiden 2.0
  clusters, and the same naming re-derived with the scp_kidney_upload.py rules,
  reproduce kidney_pre_label for all 712 cells.

Command (project root):  python scripts/kidney_object_kidney_canonical.py
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
import copy
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from _scanpy_compat import ensure_scanpy_compat
from sklearn.metrics import adjusted_rand_score

import scp_common as cc

ensure_scanpy_compat()

TASK = "kidney_object"
INP = cc.INPUTS["kidney"]
EXPECTED_SHAPE = (712, 4054)
EXPECTED_COUNTS = {
    "kidney_reannot_label": {"PCT": 219, "PC": 187, "PST": 72, "DTL": 56, "CNT": 53, "TAL": 45, "DCT2": 27,
                             "DCT1": 18, "DVR": 16, "ATL": 7, "JG": 6, "IC-B": 3, "IC-A": 2, "MD": 1},
    "kidney_pre_label": {"PCT": 168, "TAL": 154, "PST": 132, "DVR": 88, "CNT": 73, "DCT2": 24, "JG": 22,
                         "MD": 19, "IC-B": 8, "Unknown": 24},
}
LEIDEN_KEY = "leiden_res_2.00"
# scp_kidney_upload.py palette inputs.
PREFERRED_ORDER = ["PT", "DTL", "TAL", "DCT", "CNT", "CD", "Unknown"]
EARLIER_OBJECTS = ("KIDNEY_768cells_processed_v2.h5ad", "KIDNEY_768cells_processed.h5ad",
                 "KIDNEY_768cells_deg_marker_annotated.h5ad", "KIDNEY_768cells_deg_marker_annotated_v2.h5ad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_object: KIDNEY canonical labels and object.")
    parser.add_argument("--matrix-tolerance", type=float, default=1e-4,
                        help="Maximum |scale(rebuilt ln1p) - stored X|.")
    cc.add_common_args(parser)
    return parser.parse_args()


class Checks:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, group: str, check: str, expected, observed, ok: bool | None, note: str = "") -> None:
        """ok=None records an informational row that cannot fail."""
        self.rows.append({"group": group, "check": check, "expected": expected, "observed": observed,
                          "pass": "info" if ok is None else bool(ok), "note": note})

    @property
    def failures(self) -> list[str]:
        return [f"{r['group']}: {r['check']} (expected {r['expected']}, observed {r['observed']})"
                for r in self.rows if r["pass"] is False]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def original_marker_genes(path) -> dict[str, list[str]]:
    """Marker parsing exactly as scp_kidney_upload.py (strip, lower, capitalize)."""
    marker_df = pd.read_excel(path)
    return {str(row[0]): [g.strip().lower().capitalize() for g in str(row[1]).split(",")]
            for row in marker_df.itertuples(index=False)}


def rederive_pre_labels(X, intensity, var_names, cell_ids, leiden, marker_genes) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Leiden 2.0 cluster naming with the scp_kidney_upload.py rules on the stored clusters.

    adata.raw holds the pre-log intensities there, so sc.tl.score_genes
    (use_raw=None) scores on them. Per-cell argmax -> cluster mode; clusters
    whose mean max score is below the 1% quantile of cluster means (or whose
    label diversity exceeds the maximum) become 'Unknown'.
    """
    obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
    obs[LEIDEN_KEY] = pd.Categorical(leiden)
    var = pd.DataFrame(index=pd.Index(var_names))
    tmp = ad.AnnData(X=np.asarray(X, dtype=np.float32).copy(), obs=obs, var=var)
    tmp.raw = ad.AnnData(X=np.asarray(intensity, dtype=np.float32).copy(), obs=obs.copy(), var=var.copy())
    matching = {ct: [g for g in gl if g in tmp.var_names] for ct, gl in marker_genes.items()}
    matching = {ct: gl for ct, gl in matching.items() if gl}
    for ct, gl in matching.items():
        sc.tl.score_genes(tmp, gene_list=gl, score_name=f"{ct}_score")
    scores = pd.DataFrame({ct: tmp.obs[f"{ct}_score"].to_numpy() for ct in matching}, index=tmp.obs_names)
    predicted = scores.idxmax(axis=1).astype(object)
    tmp.obs["predicted_cell_type"] = predicted.to_numpy()
    tmp.obs["max_score"] = scores.max(axis=1).to_numpy()
    clust_mean = tmp.obs.groupby(LEIDEN_KEY, observed=True)["max_score"].mean()
    clust_div = tmp.obs.groupby(LEIDEN_KEY, observed=True)["predicted_cell_type"].nunique()
    unknown = clust_mean[clust_mean < clust_mean.quantile(0.01)].index.union(
        clust_div[clust_div > clust_div.quantile(1)].index)
    mapping = (tmp.obs.groupby(LEIDEN_KEY, observed=True)["predicted_cell_type"]
               .agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown").to_dict())
    named = tmp.obs[LEIDEN_KEY].astype(str).map({str(k): v for k, v in mapping.items()})
    named[tmp.obs[LEIDEN_KEY].astype(str).isin([str(u) for u in unknown])] = "Unknown"
    return named.astype(str), scores, predicted.astype(str)


def original_palette(pre_full: pd.Series, reannot_full: pd.Series) -> dict[str, str]:
    """Colours as scp_kidney_upload.py _build_global_palette (tab20/b/c even-odd; Unknown grey)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    union: list[str] = []
    for values in (pre_full, reannot_full, pre_full):  # cell_type_pre, cell_type_v2, leiden_named
        for c in sorted(set(map(str, values))):
            if c not in union:
                union.append(c)
    is_unknown = lambda c: c.lower() == "unknown"
    pref = [c for c in PREFERRED_ORDER if c in union and not is_unknown(c)]
    others = [c for c in union if c not in pref and not is_unknown(c)]
    unk = [c for c in union if is_unknown(c)]
    order = pref + others + unk[:1]

    def even_odd(colors):
        colors = list(colors)
        return colors[0::2] + colors[1::2]

    base = (even_odd(plt.get_cmap("tab20").colors) + even_odd(plt.get_cmap("tab20b").colors)
            + even_odd(plt.get_cmap("tab20c").colors))
    pal, i = {}, 0
    for c in order:
        if is_unknown(c):
            pal[c] = "#b3b3b3"
        else:
            pal[c] = to_hex(base[i])
            i += 1
    return pal


def ordered_short(values) -> list[str]:
    return cc.ordered_labels(values, cc.KIDNEY_LABEL_ORDER)


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, "kidney_canonical", args)
    checks = Checks()

    obj = sc.read_h5ad(run.use_input(INP["processed_v2_h5ad"],
                                     "the earlier analysis processed object (matrix, PCA, graph, UMAP, Leiden 2.0 reused)"))
    ids = pd.Index(obj.obs_names.astype(str), name="cell_id")
    X = cc.dense(obj.X)
    checks.add("shape", "n_obs x n_vars", f"{EXPECTED_SHAPE[0]} x {EXPECTED_SHAPE[1]}",
               f"{obj.n_obs} x {obj.n_vars}", obj.shape == EXPECTED_SHAPE)

    # 1. Rebuild the matrix and compare with the object.
    rebuild = cc.rebuild_ln_intensity("kidney", run)
    same_cells = list(rebuild.intensity.index) == list(ids)
    same_proteins = list(rebuild.intensity.columns) == list(obj.var_names.astype(str))
    checks.add("matrix", "rebuilt cell order == object", True, same_cells, same_cells)
    checks.add("matrix", "rebuilt protein order == object", True, same_proteins, same_proteins)
    aligned = same_cells and same_proteins
    intensity = rebuild.intensity.to_numpy()
    scale_err = float(np.abs(cc.scale_like_scanpy(rebuild.ln1p_intensity.to_numpy()) - X.astype(np.float64)).max()) \
        if aligned else np.inf
    checks.add("matrix", "max |scale(rebuilt ln1p) - stored X|", f"< {args.matrix_tolerance:g}", f"{scale_err:.3g}",
               scale_err < args.matrix_tolerance)
    for name, stored in (("layers['intensity']", obj.layers.get("intensity")),
                         (".raw.X", None if obj.raw is None else obj.raw.X)):
        diff = float(np.abs(cc.dense(stored).astype(np.float64) - intensity).max()) if aligned and stored is not None else np.inf
        checks.add("matrix", f"max |rebuilt intensity - stored {name}|", 0, f"{diff:.3g}", diff == 0)
    run.note(f"max |scale(rebuilt ln1p) - stored X| = {scale_err:.3g}")
    n_detected = (intensity > 0).sum(axis=1)
    raw_rows = rebuild.raw_intensity.loc[ids].to_numpy()
    n_detected_all_rows = (raw_rows > 0).sum(axis=1)
    total_intensity = intensity.astype(np.float64).sum(axis=1)

    # 2. Submitted labels.
    sens = pd.read_csv(run.use_input(INP["sens_dynr"], "submitted per-cell labels (V3, Leiden 2.0)"), index_col=0)
    sens.index = sens.index.astype(str)
    missing_in_file = ids.difference(sens.index)
    extra_in_file = sens.index.difference(ids)
    checks.add("cells", "object cells found in SENS_DYNR file", len(ids), len(ids) - len(missing_in_file),
               len(missing_in_file) == 0)
    checks.add("cells", "SENS_DYNR cells not in object", 0, len(extra_in_file), len(extra_in_file) == 0)
    checks.add("cells", "SENS_DYNR rows (unique)", len(ids), f"{len(sens)} ({sens.index.nunique()})",
               len(sens) == len(ids) and sens.index.is_unique)
    sens = sens.reindex(ids)
    reannot_full = sens["cell_type_v2"].astype(str)
    pre_full = sens["cell_type_pre"].astype(str)

    # 3. Short names: workbook 'Short' column versus the the project notes names, then counts.
    workbook = pd.read_excel(run.use_input(INP["markers"], "kidney V3 marker workbook (Supp Table 1)"))
    short_col = workbook.columns[2] if workbook.shape[1] >= 3 else None
    for row in workbook.itertuples(index=False):
        full, short = str(row[0]).strip(), (str(row[2]).strip() if short_col is not None else "")
        ours = cc.KIDNEY_ABBREVIATIONS.get(full, "")
        checks.add("short_names", f"workbook '{full}' short name", ours, short, short == ours)
    unmapped = sorted((set(reannot_full) | set(pre_full)) - set(cc.KIDNEY_ABBREVIATIONS))
    checks.add("short_names", "label values without a short name", "none", ";".join(unmapped) or "none", not unmapped)
    reannot = reannot_full.map(cc.KIDNEY_ABBREVIATIONS).fillna("UNMAPPED")
    pre = pre_full.map(cc.KIDNEY_ABBREVIATIONS).fillna("UNMAPPED")
    for col, values in (("kidney_reannot_label", reannot), ("kidney_pre_label", pre)):
        expected = EXPECTED_COUNTS[col]
        observed = values.value_counts().to_dict()
        checks.add("label_counts", f"{col} number of labels", len(expected), len(observed), len(observed) == len(expected))
        for label in ordered_short(set(expected) | set(observed)):
            checks.add("label_counts", f"{col} {label}", expected.get(label, 0), observed.get(label, 0),
                       observed.get(label, 0) == expected.get(label, 0))

    # 4. Counts and protein IDs per type versus the V3 tables.
    sens_ids = sens["sens_n_proteins"].to_numpy()
    match_filtered = int((sens_ids == n_detected).sum())
    match_all_rows = int((sens_ids == n_detected_all_rows).sum())
    checks.add("protein_ids", "per-cell sens_n_proteins == proteins > 0 among 4,054 retained", len(ids),
               match_filtered, match_filtered == len(ids))
    checks.add("protein_ids", "per-cell sens_n_proteins == proteins > 0 among all 4,144 CSV rows", "info",
               match_all_rows, None)
    detected = intensity > 0
    v3_rows = []
    for tag, labels_full, key in (("AFTER", reannot_full, "protein_ids_after"), ("BEFORE", pre_full, "protein_ids_before")):
        table = pd.read_csv(run.use_input(INP[key], f"V3 protein IDs by cell type ({tag})")).set_index("cell_type")
        present = sorted(set(labels_full) - {"Unknown"})
        checks.add("v3_tables", f"{tag}: cell types in table == label types (Unknown excluded)", ";".join(present),
                   ";".join(sorted(table.index)), sorted(table.index) == present)
        for ct in sorted(set(table.index) | set(present)):
            mask = (labels_full == ct).to_numpy()
            n = int(mask.sum())
            n_cells_v3 = int(table.loc[ct, "n_cells"]) if ct in table.index else 0
            med = float(np.median(n_detected[mask])) if n else np.nan
            med_all = float(np.median(n_detected_all_rows[mask])) if n else np.nan
            med_v3 = float(table.loc[ct, "median_per_cell_IDs"]) if ct in table.index else np.nan
            col_sums = detected[mask].sum(axis=0)
            row = {"table": tag, "cell_type": ct, "short_name": cc.KIDNEY_ABBREVIATIONS.get(ct, ct),
                   "n_cells": n, "n_cells_v3": n_cells_v3,
                   "median_ids": med, "median_ids_v3": med_v3, "median_ids_all_csv_rows": med_all,
                   "union_ids": int((col_sums > 0).sum()),
                   "union_ids_v3": int(table.loc[ct, "union_ids"]) if ct in table.index else np.nan}
            for frac in (10, 25, 50):
                thr = int(np.ceil(frac / 100 * n))
                row[f"coverage_ge_{frac}pct"] = int((col_sums >= thr).sum())
                row[f"coverage_ge_{frac}pct_v3"] = int(table.loc[ct, f"coverage_ge_{frac}pct"]) if ct in table.index else np.nan
            v3_rows.append(row)
            checks.add("v3_tables", f"{tag} {ct}: n_cells", n_cells_v3, n, n == n_cells_v3)
            checks.add("v3_tables", f"{tag} {ct}: median protein IDs per cell", med_v3, med, med == med_v3)
        checks.add("v3_tables", f"{tag}: Unknown cells (not in table; the project notes count)",
                   EXPECTED_COUNTS["kidney_pre_label"]["Unknown"] if tag == "BEFORE" else 0,
                   int((labels_full == "Unknown").sum()),
                   int((labels_full == "Unknown").sum()) == (24 if tag == "BEFORE" else 0))
    v3_frame = pd.DataFrame(v3_rows)
    union_ok = bool((v3_frame["union_ids"] == v3_frame["union_ids_v3"]).all())
    cov_ok = all((v3_frame[f"coverage_ge_{f}pct"] == v3_frame[f"coverage_ge_{f}pct_v3"]).all() for f in (10, 25, 50))
    checks.add("v3_tables", "union and coverage protein IDs per type equal the V3 tables", "info",
               f"union {union_ok}; coverage {cov_ok}", None)

    # 5. Named Leiden 2.0 clusters versus kidney_pre_label.
    leiden = obj.obs[LEIDEN_KEY].astype(str).to_numpy()
    for col in (f"{LEIDEN_KEY}_named", "cell_type_pre"):
        n_match = int((obj.obs[col].astype(str).to_numpy() == pre_full.to_numpy()).sum())
        checks.add("leiden_naming", f"stored obj.obs['{col}'] == kidney_pre_label", len(ids), n_match,
                   n_match == len(ids) if col == f"{LEIDEN_KEY}_named" else None,
                   "earlier processed object" if col == "cell_type_pre" else "")
    purity = pd.crosstab(leiden, pre_full.to_numpy())
    n_impure = int(((purity > 0).sum(axis=1) > 1).sum())
    checks.add("leiden_naming", "Leiden 2.0 clusters containing more than one kidney_pre_label", 0, n_impure, n_impure == 0)
    marker_genes = original_marker_genes(INP["markers"])
    named, scores, predicted = rederive_pre_labels(X, intensity, obj.var_names.astype(str), ids, leiden, marker_genes)
    n_rederived = int((named.to_numpy() == pre_full.to_numpy()).sum())
    checks.add("leiden_naming", "re-derived naming (scp_kidney_upload.py rules, stored clusters) == kidney_pre_label",
               len(ids), n_rederived, n_rederived == len(ids))
    unknown_clusters = sorted(set(leiden[named.to_numpy() == "Unknown"]), key=int)
    run.note(f"Re-derived Unknown cluster(s): {unknown_clusters}; {int((named == 'Unknown').sum())} cells.")
    score_diffs = [float(np.abs(obj.obs[f"{ct}_score"].to_numpy(float) - scores[ct].to_numpy()).max())
                   for ct in scores.columns if f"{ct}_score" in obj.obs]
    checks.add("leiden_naming", "re-derived marker scores vs stored object *_score (max abs diff)", "info",
               f"{max(score_diffs):.3g} over {len(score_diffs)} types", None)
    n_pred = int((predicted.to_numpy() == obj.obs["predicted_cell_type"].astype(str).to_numpy()).sum())
    checks.add("leiden_naming", "re-derived per-cell argmax == stored predicted_cell_type", "info", f"{n_pred}/{len(ids)}", None)

    # 6. Stored graph: informational reproducibility in this environment.
    tmp = ad.AnnData(X=X.copy(), obs=pd.DataFrame(index=ids), var=pd.DataFrame(index=obj.var_names.astype(str)))
    tmp.obsm["X_pca"] = np.asarray(obj.obsm["X_pca"])
    sc.pp.neighbors(tmp, n_neighbors=10, n_pcs=20, use_rep="X_pca")
    conn_diff = float(np.abs(cc.dense(tmp.obsp["connectivities"]) - cc.dense(obj.obsp["connectivities"])).max())
    checks.add("graph_info", "neighbours rebuilt from stored X_pca[:, :20] vs stored connectivities (max abs diff)",
               "info", f"{conn_diff:.3g}", None)
    pca20 = sc.pp.pca(X.astype(np.float32), n_comps=20, svd_solver="arpack")
    corr = [abs(np.corrcoef(pca20[:, i], obj.obsm["X_pca"][:, i])[0, 1]) for i in range(20)]
    checks.add("graph_info", "PCA (arpack, 20 comps, as the original implicit PCA) vs stored X_pca[:, :20]: min |r|",
               "info", f"{min(corr):.6f}", None)
    for seed in (0, cc.RANDOM_SEED):
        g = obj.copy()
        try:
            sc.tl.leiden(g, key_added="rerun", resolution=2.0, flavor="igraph", random_state=seed)
            ari = adjusted_rand_score(leiden, g.obs["rerun"].astype(str).to_numpy())
            checks.add("graph_info", f"Leiden 2.0 rerun on stored graph (igraph, random_state={seed}): ARI vs stored",
                       "info", f"{ari:.4f} ({g.obs['rerun'].nunique()} clusters)", None)
        except Exception as exc:  # report, do not fail
            checks.add("graph_info", f"Leiden 2.0 rerun (random_state={seed})", "info", f"error: {exc}", None)

    # 7. Reference agreements (not used as labels anywhere).
    ref_rows = []
    comparisons = [("kidney_reannot_label", reannot_full, "cell_type_v2", "the earlier analysis re-annotation"),
                   ("kidney_reannot_label", reannot_full, "predicted_cell_type", "per-cell argmax (stored)"),
                   ("kidney_pre_label", pre_full, "cell_type_pre", "the earlier analysis cell_type_pre"),
                   ("kidney_pre_label", pre_full, f"{LEIDEN_KEY}_named", "the earlier analysis named Leiden 2.0")]
    for fname in EARLIER_OBJECTS:
        path = cc.PROJECT_ROOT / "h5ad" / fname
        if not path.exists():
            continue
        other = ad.read_h5ad(run.use_input(path, "earlier processed object (obs only, reference agreement)"), backed="r")
        other_obs = other.obs.copy()
        other.file.close()
        other_obs.index = other_obs.index.astype(str)
        for ours_col, ours, theirs_col, desc in comparisons:
            if theirs_col not in other_obs:
                continue
            theirs = other_obs[theirs_col].astype(str).reindex(ids)
            n_agree = int((ours.to_numpy() == theirs.to_numpy()).sum())
            ref_rows.append({"label_column": ours_col, "compared_with": f"{fname}: obs['{theirs_col}']",
                             "description": desc, "n_labels_other": int(theirs.nunique()),
                             "n_agree": n_agree, "n_cells": len(ids), "fraction_agree": n_agree / len(ids)})
    n_agree = int((reannot_full.to_numpy() == predicted.to_numpy()).sum())
    ref_rows.append({"label_column": "kidney_reannot_label", "compared_with": "re-derived per-cell argmax (this script)",
                     "description": "per-cell argmax of V3 marker scores", "n_labels_other": int(predicted.nunique()),
                     "n_agree": n_agree, "n_cells": len(ids), "fraction_agree": n_agree / len(ids)})
    n_agree = int((reannot_full.to_numpy() == pre_full.to_numpy()).sum())
    ref_rows.append({"label_column": "kidney_reannot_label", "compared_with": "kidney_pre_label",
                     "description": "same label before and after re-annotation", "n_labels_other": int(pre_full.nunique()),
                     "n_agree": n_agree, "n_cells": len(ids), "fraction_agree": n_agree / len(ids)})
    reference = pd.DataFrame(ref_rows)

    # 8. Crosstab: where cells move between pre and re-annotated labels.
    pre_order, re_order = ordered_short(pre), ordered_short(reannot)
    crosstab = pd.crosstab(pd.Categorical(pre, pre_order), pd.Categorical(reannot, re_order),
                           rownames=["kidney_pre_label"], colnames=["kidney_reannot_label"], dropna=False)
    crosstab["total"] = crosstab.sum(axis=1)
    crosstab.loc["total"] = crosstab.sum(axis=0)

    # 9. Plate search.
    raw_header = pd.read_csv(INP["raw_csv"], nrows=0).columns.astype(str)
    cell_cols = raw_header[3:]
    plate_re = re.compile(r"plate|_P\d+_|mouse|batch", re.I)
    kidney_files = [p for p in cc.KIDNEY_DIR.rglob("*") if p.is_file() and plate_re.search(p.name)]
    plate_rows = [
        {"source": cc.rel(INP["raw_csv"]) + " cell column names", "searched_for": plate_re.pattern,
         "result": f"{int(sum(bool(plate_re.search(c)) for c in cell_cols))} of {len(cell_cols)} names match; "
                   f"pattern of names: {cell_cols[0]} ... {cell_cols[-1]}"},
        {"source": cc.rel(INP["raw_csv"]) + " metadata columns", "searched_for": "any plate/mouse/batch column",
         "result": ", ".join(raw_header[:3])},
        {"source": cc.rel(INP["sens_dynr"]) + " columns", "searched_for": "any plate/mouse/batch column",
         "result": ", ".join(map(str, sens.columns))},
        {"source": cc.rel(INP["processed_v2_h5ad"]) + " obs columns", "searched_for": "plate/mouse/batch/sample",
         "result": ", ".join(c for c in obj.obs.columns if plate_re.search(c) or "sample" in c.lower()) or "none"},
        {"source": cc.rel(INP["processed_v2_h5ad"]) + " obs['plate'] values", "searched_for": "values",
         "result": str(obj.obs["plate"].astype(str).value_counts().to_dict()) if "plate" in obj.obs else "absent"},
        {"source": "KIDNEY/ file names (recursive)", "searched_for": plate_re.pattern,
         "result": "; ".join(cc.rel(p) for p in kidney_files) or "none"},
        {"source": "conclusion", "searched_for": "", "result": "plate not determinable from the data; recorded as 'unknown'"},
    ]
    plate_search = pd.DataFrame(plate_rows)

    # 10. V3 marker coverage in the 4,054-protein matrix.
    var_set = set(obj.var_names.astype(str))
    coverage = pd.DataFrame([{"cell_type": ct, "short_name": cc.KIDNEY_ABBREVIATIONS.get(ct, ""),
                              "n_markers": len(gl), "n_present": sum(g in var_set for g in gl),
                              "markers": ";".join(gl), "present": ";".join(g for g in gl if g in var_set),
                              "missing": ";".join(g for g in gl if g not in var_set)}
                             for ct, gl in marker_genes.items()])

    check_frame = checks.frame()
    run.table(check_frame, "kidney_sanity_checks")
    run.table(v3_frame, "kidney_counts_vs_v3_tables")
    run.table(reference, "kidney_label_agreement_reference")
    run.table(crosstab, "kidney_pre_vs_reannot_crosstab", index=True)
    run.table(plate_search, "kidney_plate_search")
    run.table(coverage, "kidney_v3_marker_coverage")
    failures = checks.failures
    if failures:
        run.table(pd.DataFrame({"failure": failures}), "kidney_failures")
        run.finish({"n_failures": len(failures)}, status="failed")
        print("KIDNEY SANITY CHECKS FAILED; no canonical outputs written:\n  " + "\n  ".join(failures))
        sys.exit(1)

    # Canonical label table.
    labels = pd.DataFrame(index=ids)
    labels["plate"] = "unknown"
    labels["kidney_reannot_label"] = reannot.to_numpy()
    labels["kidney_pre_label"] = pre.to_numpy()
    labels["kidney_reannot_label_full"] = reannot_full.to_numpy()
    labels["kidney_pre_label_full"] = pre_full.to_numpy()
    labels[LEIDEN_KEY] = leiden
    labels["n_detected_proteins"] = n_detected
    labels["total_intensity"] = total_intensity
    run.table(labels.reset_index(), "kidney_canonical_labels")

    # Canonical object.
    obs = labels.copy()
    obs["kidney_reannot_label"] = pd.Categorical(obs["kidney_reannot_label"], categories=re_order)
    obs["kidney_pre_label"] = pd.Categorical(obs["kidney_pre_label"], categories=pre_order)
    obs[LEIDEN_KEY] = pd.Categorical(leiden, categories=sorted(set(leiden), key=int))
    var = pd.DataFrame(index=pd.Index(obj.var_names.astype(str), name="protein"))
    pm = rebuild.protein_meta.set_index("var_name")
    var["gene_name"] = obj.var["gene_name"].astype(str).to_numpy()
    var["protein_group"] = pm.loc[var.index, "protein_group"].to_numpy()
    var["pg_genes"] = pm.loc[var.index, "gene_symbol"].to_numpy()
    var["protein_description"] = pm.loc[var.index, "protein_description"].to_numpy()
    var["n_cells_detected"] = detected.sum(axis=0)
    out = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    for key in ("X_pca", "X_umap"):
        out.obsm[key] = np.asarray(obj.obsm[key])
    for key in ("connectivities", "distances"):
        out.obsp[key] = obj.obsp[key]
    for key in ("neighbors", "umap", "pca"):
        out.uns[key] = copy.deepcopy(obj.uns[key])
    out.layers["intensity"] = intensity.astype(np.float32)
    out.layers["ln1p_intensity"] = rebuild.ln1p_intensity.to_numpy().astype(np.float32)
    palette = original_palette(pre_full, reannot_full)
    short_palette = {cc.KIDNEY_ABBREVIATIONS[k]: v for k, v in palette.items()}
    out.uns["kidney_reannot_label_colors"] = [short_palette[c] for c in re_order]
    out.uns["kidney_pre_label_colors"] = [short_palette[c] for c in pre_order]
    out.uns["kidney_object"] = {
        "generated_by": cc.GENERATED_BY,
        "script": cc.rel(__file__),
        "created_utc": cc.now_utc(),
        "source_object": cc.rel(INP["processed_v2_h5ad"]),
        "source_object_sha256": run.inputs[cc.rel(INP["processed_v2_h5ad"])]["sha256"],
        "X": "per-protein z-scored ln(1+intensity) (ddof=1), clipped at +10 (as scp_kidney_upload.py); equals the rebuilt matrix scaled",
        "layers_intensity": f"Spectronaut PG.Quantity from {cc.rel(INP['raw_csv'])}, missing -> 0",
        "layers_ln1p_intensity": "ln(1 + intensity), rebuilt; scaling reproduces X",
        "graph_pca_umap": "reused from the earlier processed object (neighbors n_neighbors=10, n_pcs=20; X_pca 50 comps; X_umap); "
                          "not the submitted run's own objects, which were not saved",
        "leiden_res_2.00": "stored Leiden 2.0 clusters of the earlier processed object; their naming reproduces kidney_pre_label",
        "kidney_reannot_label": cc.KIDNEY_LABEL_SOURCES["kidney_reannot_label"],
        "kidney_pre_label": cc.KIDNEY_LABEL_SOURCES["kidney_pre_label"],
        "label_colors": "scp_kidney_upload.py palette rule (tab20/tab20b/tab20c even-odd; Unknown grey)",
        "plate": "unknown: not encoded in the cell IDs or metadata",
        "markers": "data/KIDNEY/KIDNEY_markers_V3.xlsx only",
    }
    run.h5ad(out, "KIDNEY_submitted")
    run.finish({"n_failures": 0, "unknown_clusters_rederived": unknown_clusters})
    print("All KIDNEY checks passed.")


if __name__ == "__main__":
    main()
