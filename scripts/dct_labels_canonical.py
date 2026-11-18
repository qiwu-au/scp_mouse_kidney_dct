#!/usr/bin/env python
"""dct_labels step 8: canonical DCT label table (both conditions, combined search) and labelled objects.

Analysis plan, step 8.

- results/tables/dct_labels/dct_labels_dct_canonical_labels.csv: one row per cell of both conditions (dct_intake
  joint object order): cell_id, cell_key, condition, plate, well, n_detected, total_intensity, every cs_ column,
  the three marker scores, max_score, dpt_pseudotime, own and joint UMAP coordinates, and
    dct_label             = cs_ctrl_deg_label (Ctrl) + cs_zerok_pipeline_label (ZeroK)   (fixed in the plan)
    dct_label_alternative = cs_ctrl_deg_label (Ctrl) + cs_zerok_transfer_label (ZeroK)
  with their source column and confidence. Later tasks read combined-search labels only here (match on cell_key).
- results/h5ad/dct_labels_DCT_{ctrl,zerok}_labelled.h5ad: the step-1 object of the condition (scaled .X; layers
  'intensity' and 'ln1p_intensity'; graph; X_pca_neighbors and X_pca; X_umap; X_diffmap; PAGA) with the labels.
- results/h5ad/dct_labels_DCT_joint_labelled.h5ad: the dct_intake joint object (all layers and embeddings) with the labels
  (gzip, as zerok_core).
- dct_labels_label_column_summary.csv: one row per label column (counts, stability, depth and plate, agreement),
  read from the step 2-5 tables.

Inputs (read-only): results/h5ad/dct_intake_DCT_{ctrl,zerok,joint}.h5ad; dct_labels_DCT_{ctrl,zerok}_step1_submitted.h5ad;
the dct_labels step 1-5 tables.
Outputs: the files above and dct_labels_canonical_checks.csv.

Command (project root):  python scripts/dct_labels_canonical.py
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
import labels_core_lib as t18
import zerok_core_lib as t21
import dct_labels_lib as lib
from kidney_annotation_scan_lib import eta_squared

SCRIPT = "dct_labels_canonical.py"
DCT_LABEL = {"Ctrl": "cs_ctrl_deg_label", "ZeroK": "cs_zerok_pipeline_label"}
DCT_LABEL_ALT = {"Ctrl": "cs_ctrl_deg_label", "ZeroK": "cs_zerok_transfer_label"}
SOURCES = {
    "cell_label": "submitted pipeline: per-cell argmax of sc.tl.score_genes on the scaled .X (DCT V1 workbook)",
    "cluster_label": "submitted pipeline: Leiden 3.0 mode with the original Unknown rule",
    "leiden3_cluster": "submitted pipeline: Leiden 3.0 (igraph, seed 42) on the condition's own graph",
    "deg_label": "cluster_de cluster DE rule + kidney_annotation rank-free rescue on Leiden 1.0 (seed 42); Slc12a3 and Umod shared",
    "deg_cluster": "Leiden 1.0 (igraph, seed 42) on the condition's own submitted graph",
    "deg_confidence": "high / medium / specific_any_rank (rank-free rescue) / low_shared_only / "
                      "unknown_no_marker_in_top_de",
    "transfer_label": "majority cs_ctrl_deg_label of the 10 nearest Ctrl cells in the dct_intake joint 20-PC embedding "
                      "(variant 'none'); ties to the nearest tied neighbour",
}
TRANSFER_VARIANT_COLUMNS = {f"transfer__{v}": f"cs_zerok_transfer_label__{v}" for v in lib.VARIANTS}
TRANSFER_CONF_COLUMNS = {f"transfer_confidence__{v}": f"cs_zerok_transfer_confidence__{v}" for v in lib.VARIANTS}


def stop(run: cc.Run, checks: t5.Checks) -> None:
    lib.stop(run, checks, "canonical_checks")


def condition_table(run: cc.Run, cond: str, obj, step1_all, deg_all, stab_all, transfer) -> pd.DataFrame:
    cells = obj.obs_names.astype(str)
    s1 = lib.per_condition(step1_all, cond, cells)
    dg = lib.per_condition(deg_all, cond, cells)
    st = lib.per_condition(stab_all, cond, cells)
    f = pd.DataFrame(index=pd.Index(cells, name="cell_id"))
    f["cell_key"] = s1["cell_key"].astype(str)
    f["condition"] = cond
    for c in ("plate", "well", "n_detected", "total_intensity"):
        f[c] = s1[c]
    f["well"] = f["well"].astype(str).str.zfill(4)
    for generic in ("cell_label", "cluster_label", "leiden3_cluster"):
        f[lib.cs(cond, generic)] = s1[generic].astype(str)
    for generic in ("deg_label", "deg_cluster", "deg_confidence"):
        f[lib.cs(cond, generic)] = dg[generic].astype(str)
    stem = f"{lib.PREFIX[cond]}_{lib.PRIMARY_STEM[cond]}"
    f[f"{stem}_seed_stability"] = st["seed_stability_deg_label"].astype(float)
    f[f"{lib.PREFIX[cond]}_cluster_label_seed_stability"] = st["seed_stability_cluster_label"].astype(float)
    if cond == "ZeroK":
        tr = transfer.set_index("cell_id").loc[cells]
        f["cs_zerok_transfer_label"] = tr["cs_zerok_transfer_label"].astype(str)
        f["cs_zerok_transfer_confidence"] = tr["cs_zerok_transfer_confidence"].astype(float)
        for src, dst in {**TRANSFER_VARIANT_COLUMNS, **TRANSFER_CONF_COLUMNS}.items():
            f[dst] = tr[src]
        d = float(np.abs(tr["cs_zerok_pipeline_seed_stability"].to_numpy(float) - f[f"{stem}_seed_stability"]).max())
        run.note(f"ZeroK pipeline seed stability: step-3 (scan) vs step-4 tables max abs diff {d:.3g}")
    for c in ("DCT1_score", "DCT2_score", "ProLIF_score", "max_score", "dpt_pseudotime"):
        f[c] = s1[c].astype(float)
    f["umap1_own"], f["umap2_own"] = s1["umap1"].astype(float), s1["umap2"].astype(float)
    f["dct_label"] = f[DCT_LABEL[cond]]
    f["dct_label_source"] = DCT_LABEL[cond]
    f["dct_label_confidence"] = f[lib.cs(cond, "deg_confidence")]
    f["dct_label_alternative"] = f[DCT_LABEL_ALT[cond]]
    f["dct_label_alternative_source"] = DCT_LABEL_ALT[cond]
    f["dct_label_alternative_confidence"] = (f[lib.cs(cond, "deg_confidence")] if cond == "Ctrl" else
                                             [f"vote_fraction_{v:.1f}" for v in f["cs_zerok_transfer_confidence"]])
    return f


def summary(run: cc.Run, canon: pd.DataFrame) -> pd.DataFrame:
    seeds = lib.read_table(run, "stability_seeds")
    seed_lab = lib.read_table(run, "stability_seed_per_label")
    noise = lib.read_table(run, "stability_noise_summary")
    variants = lib.read_table(run, "stability_marker_variants")
    loo = lib.read_table(run, "stability_loo_per_label")
    depth = lib.read_table(run, "depth_plate_by_label")
    agree = lib.read_table(run, "crosstab_agreement")
    transfer_agree = lib.read_table(run, "transfer_agreement")
    rows = []
    for cond in lib.CONDITIONS:
        sub = canon[canon["condition"] == cond]
        ids = sub["n_detected"].to_numpy(float)
        noise_test = {lib.cs(cond, "cell_label"): "A: noisy .X scored",
                      lib.cs(cond, "cluster_label"): "B: noisy .X, whole pipeline",
                      lib.cs(cond, "deg_label"): "B: noisy graph, rank-free DE rule at Leiden 1.0"}
        for col in lib.label_columns(cond):
            row = {"condition": cond, "label_column": col,
                   "role": ("primary (dct_label)" if col == DCT_LABEL[cond] else
                            "alternative (dct_label_alternative)" if col == DCT_LABEL_ALT[cond] else "comparison"),
                   "source": SOURCES["transfer_label" if "transfer" in col else
                                     next(g for g in ("cell_label", "cluster_label", "deg_label")
                                          if col == lib.cs(cond, g))],
                   "label_counts": lib.counts_text(sub[col]),
                   "eta2_protein_ids_on_label": eta_squared(ids, sub[col].to_numpy())}
            dp = depth[(depth["condition"] == cond) & (depth["label_column"] == col) & (depth["label"] != "all")]
            row["median_protein_ids_by_label"] = "; ".join(f"{r.label} {r.protein_ids_median:g}" for r in dp.itertuples())
            row["fraction_P1_by_label"] = "; ".join(f"{r.label} {r.fraction_P1:.2f}" for r in dp.itertuples())
            if col in noise_test:
                c = (seeds["condition"] == cond) & (seeds["label_column"] == col) & (seeds["seed"] != lib.SEED)
                sl = seed_lab[(seed_lab["condition"] == cond) & (seed_lab["label_column"] == col)
                              & (seed_lab["label_seed42"] == "all")].iloc[0]
                nz = noise[(noise["condition"] == cond) & (noise["label_column"] == col)
                           & (noise["test"] == noise_test[col])].iloc[0]
                lo = loo[(loo["condition"] == cond) & (loo["label_column"] == col) & (loo["original_label"] == "all")].iloc[0]
                v = variants[(variants["condition"] == cond) & (variants["label_column"] == col)].set_index("variant")["n_changed"]
                row.update({"seed_ari_labels_median": float(seeds.loc[c, "ari_labels_vs_seed42"].median()),
                            "seed_ari_labels_min": float(seeds.loc[c, "ari_labels_vs_seed42"].min()),
                            "seed_fraction_cells_same_label_all_10_seeds": float(sl["fraction_stable_all_10_seeds"]),
                            "noise_test": nz["test"], "noise_cells_changed_median": float(nz["n_changed_median"]),
                            "noise_cells_changed_max": int(nz["n_changed_max"]),
                            "noise_cells_changed_in_any_repeat": int(nz["n_cells_changed_in_any_repeat"]),
                            "loo_mean_stability": float(lo["mean_stability"]),
                            "loo_n_cells_stability_below_0.5": int(lo["n_stability_below_0.5"]),
                            "noSlc12a3_cells_changed": int(v["noSlc12a3"]),
                            "noSlc12a3_noUmod_cells_changed": int(v["noSlc12a3_noUmod"])})
                a = agree[agree["condition"] == cond]
                row["agreement_with_other_columns"] = "; ".join(
                    f"{r.pair.replace(col, '').replace(' vs ', '').strip()} {r.fraction_equal:.2f}"
                    for r in a.itertuples() if col in r.pair)
            else:
                ta = transfer_agree.set_index("comparison")
                row["agreement_with_other_columns"] = (
                    f"cs_zerok_pipeline_label {ta.loc['transfer none vs pipeline', 'fraction_equal']:.2f}; "
                    f"transfer cell_median {ta.loc['transfer none vs transfer cell_median', 'fraction_equal']:.2f}; "
                    f"transfer condition_align {ta.loc['transfer none vs transfer condition_align', 'fraction_equal']:.2f}")
            rows.append(row)
    return pd.DataFrame(rows)


def as_text(values: pd.Series) -> np.ndarray:
    """Values as strings with missing entries as '' (NaN survives astype(str) for categoricals and str dtype)."""
    return values.astype(object).where(values.notna(), "").astype(str).to_numpy()


def to_obs(obj, frame: pd.DataFrame, columns: list[str]) -> None:
    """Copy columns into obs; label columns become categoricals in the labels_core label order (NaN where not defined)."""
    for col in columns:
        values = frame[col]
        if col.endswith("_label") or "_label__" in col or col in ("dct_label", "dct_label_alternative"):
            cats = cc.ordered_labels(values.dropna(), t18.LABEL_ORDER)
            obj.obs[col] = pd.Categorical(values.where(values.notna()), categories=cats)
            obj.uns[f"{col}_colors"] = [t18.LABEL_COLORS[c] for c in cats]
        elif values.dtype == object:
            obj.obs[col] = pd.Categorical(values.where(values.notna()))
        else:
            obj.obs[col] = values.to_numpy()


def main() -> None:
    import anndata as ad

    parser = argparse.ArgumentParser(description="dct_labels step 8: canonical labels and labelled objects.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "canonical", args)
    checks = t5.Checks("dct_combined")
    step1 = lib.read_table(run, "submitted_per_cell", dtype={"cell_id": str, "leiden3_cluster": str, "well": str})
    deg = lib.read_table(run, "deg_labels_per_cell", dtype={"cell_id": str, "deg_cluster": str})
    stab = lib.read_table(run, "stability_per_cell", dtype={"cell_id": str})
    transfer = lib.read_table(run, "transfer_labels_per_cell", dtype={"cell_id": str})
    joint = lib.t26.read_object(run, "joint")
    joint_ids = joint.obs_names.astype(str)

    objs, tables = {}, {}
    for cond in lib.CONDITIONS:
        obj = lib.read_step1(run, cond)
        t26obj = lib.read_t26(run, cond)
        same = list(obj.obs_names.astype(str)) == list(t26obj.obs_names.astype(str))
        checks.add(f"{cond}: step-1 object holds the dct_intake {cond} cells in the dct_intake order", f"{same}", same)
        if not same:
            stop(run, checks)
        f = condition_table(run, cond, obj, step1, deg, stab, transfer)
        for col in ("n_detected", "total_intensity"):
            d = float(np.abs(f[col].to_numpy(float) - t26obj.obs[col].to_numpy(float)).max())
            checks.add(f"{cond}: {col} equals the dct_intake obs", f"max abs diff {d:.3g}", d < 1e-6)
        ok = bool((f["cell_key"].to_numpy() == t26obj.obs["cell_key"].astype(str).to_numpy()).all())
        checks.add(f"{cond}: cell_key equals the dct_intake obs", f"{ok}", ok)
        ln1p = np.log1p(cc.dense(obj.layers["intensity"]).astype(np.float32))
        if "ln1p_intensity" in t26obj.layers:
            d = float(np.abs(ln1p - cc.dense(t26obj.layers["ln1p_intensity"])).max())
            checks.add(f"{cond}: ln(1 + intensity) of the step-1 object equals the dct_intake ln1p layer",
                       f"max abs diff {d:.3g}", d < 1e-5)
        obj.layers["ln1p_intensity"] = np.array(ln1p, dtype=np.float32, copy=True)
        ok = bool((obj.obs[lib.cs(cond, "cell_label")].astype(str).to_numpy() == f[lib.cs(cond, "cell_label")].to_numpy()).all())
        checks.add(f"{cond}: object obs {lib.cs(cond, 'cell_label')} equals the step-1 table", f"{ok}", ok)
        objs[cond], tables[cond] = obj, f
        del t26obj

    # Canonical table in the dct_intake joint order
    canon = pd.concat([tables[c] for c in lib.CONDITIONS])
    same_set = set(canon.index) == set(joint_ids) and len(canon) == len(joint_ids)
    checks.add("canonical cells equal the dct_intake joint cells", f"{len(canon)} rows; {same_set}", same_set)
    if not checks.ok:
        stop(run, checks)
    canon = canon.loc[joint_ids]
    canon["umap1_joint"] = np.asarray(joint.obsm[f"X_umap__{lib.BASELINE_VARIANT}"])[:, 0]
    canon["umap2_joint"] = np.asarray(joint.obsm[f"X_umap__{lib.BASELINE_VARIANT}"])[:, 1]
    ok = bool((canon["cell_key"].to_numpy() == joint.obs["cell_key"].astype(str).to_numpy()).all()
              and (canon["condition"].to_numpy() == joint.obs["condition"].astype(str).to_numpy()).all())
    checks.add("canonical cell_key and condition equal the dct_intake joint obs", f"{ok}", ok)
    uniq = canon["cell_key"].is_unique
    checks.add("cell_key is unique", f"{uniq}", uniq)
    lead = ["cell_key", "condition", "plate", "well", "n_detected", "total_intensity", "dct_label", "dct_label_source",
            "dct_label_confidence", "dct_label_alternative", "dct_label_alternative_source",
            "dct_label_alternative_confidence"]
    cs_cols = [c for c in canon.columns if c.startswith("cs_")]
    tail = ["DCT1_score", "DCT2_score", "ProLIF_score", "max_score", "dpt_pseudotime", "umap1_own", "umap2_own",
            "umap1_joint", "umap2_joint"]
    canon = canon[lead + cs_cols + tail].reset_index()
    for cond in lib.CONDITIONS:
        sub = canon[canon["condition"] == cond]
        own = [c for c in cs_cols if c.startswith(lib.PREFIX[cond])]
        other = [c for c in cs_cols if not c.startswith(lib.PREFIX[cond])]
        na_own = int(sub[own + lead + tail].isna().sum().sum())
        na_other = bool(sub[other].isna().all().all())
        checks.add(f"{cond}: no missing values in its own columns; other condition's cs_ columns empty",
                   f"{na_own}; {na_other}", na_own == 0 and na_other)
        eq = bool((sub["dct_label"] == sub[DCT_LABEL[cond]]).all() and
                  (sub["dct_label_alternative"] == sub[DCT_LABEL_ALT[cond]]).all())
        checks.add(f"{cond}: dct_label = {DCT_LABEL[cond]}; dct_label_alternative = {DCT_LABEL_ALT[cond]}", f"{eq}", eq)
        checks.add(f"{cond}: dct_label counts", lib.counts_text(sub["dct_label"]), None)
        checks.add(f"{cond}: dct_label_alternative counts", lib.counts_text(sub["dct_label_alternative"]), None)
        for col in lib.label_columns(cond):
            checks.add(f"{cond}: {col} counts", lib.counts_text(sub[col]), None)
    if not checks.ok:
        stop(run, checks)

    # Labelled objects
    label_cols = {c: [k for k in canon.columns if k.startswith(lib.PREFIX[c])] + ["dct_label", "dct_label_source",
                  "dct_label_confidence", "dct_label_alternative", "dct_label_alternative_source",
                  "dct_label_alternative_confidence"] for c in lib.CONDITIONS}
    provenance = {"task": lib.TASK, "generated_by": cc.GENERATED_BY, "script": f"scripts/{SCRIPT}",
                  "seed": lib.SEED, "search": "combined Spectronaut search of Ctrl and ZeroK (DCT_Ctrl_ZeroK_combine_1535cells.csv)",
                  "dct_label": "cs_ctrl_deg_label (Ctrl) + cs_zerok_pipeline_label (ZeroK); fixed in the dct_labels plan",
                  "dct_label_alternative": "cs_ctrl_deg_label (Ctrl) + cs_zerok_transfer_label (ZeroK)",
                  "design": "one pool of 3 mice per condition; plates P1/P2 are sort plates (technical only)",
                  **{f"source_{k}": v for k, v in SOURCES.items()}}
    indexed = canon.set_index("cell_id")
    for cond in lib.CONDITIONS:
        obj = objs[cond]
        drop = [c for c in obj.obs.columns if c.startswith(lib.PREFIX[cond])]
        obj.obs = obj.obs.drop(columns=drop)
        to_obs(obj, indexed.loc[obj.obs_names.astype(str)], label_cols[cond])
        obj.uns["dct_labels"] = {**provenance, "condition": cond}
        for key in ("X_umap", "X_pca", "X_pca_neighbors", "X_diffmap"):
            checks.add(f"{cond}: labelled object obsm has {key}", f"{key in obj.obsm}", key in obj.obsm)
        ok = "connectivities" in obj.obsp and "paga" in obj.uns
        checks.add(f"{cond}: labelled object has the neighbour graph and PAGA", f"{ok}", ok)
    jcols = [c for c in canon.columns if c.startswith("cs_")] + label_cols["Ctrl"][-6:]
    to_obs(joint, indexed.loc[joint_ids], jcols)
    joint.uns["dct_labels"] = provenance
    if not checks.ok:
        stop(run, checks)

    summ = summary(run, canon)
    csv_path = run.table(canon, "dct_canonical_labels")
    run.table(summ, "label_column_summary")
    paths = {c: run.h5ad(objs[c], f"DCT_{lib.KEY[c]}_labelled") for c in lib.CONDITIONS}
    paths["joint"] = t21.write_h5ad(run, joint, "DCT_joint_labelled")

    # Read-back
    back = pd.read_csv(csv_path, dtype={"cell_id": str, "cs_ctrl_leiden3_cluster": str, "cs_ctrl_deg_cluster": str,
                                        "cs_zerok_leiden3_cluster": str, "cs_zerok_pipeline_cluster": str, "well": str})
    ok_csv = back.shape == canon.shape and bool((back["cell_key"] == canon["cell_key"]).all())
    checks.add("canonical CSV read back: same shape and cell_key order", f"{back.shape}; {ok_csv}", ok_csv)
    for key, path in paths.items():
        re = ad.read_h5ad(path)
        sub = back.set_index("cell_id").loc[re.obs_names.astype(str)]
        cols = label_cols[key] if key in label_cols else jcols
        bad = [c for c in cols if not np.array_equal(as_text(re.obs[c]), as_text(sub[c]))]
        ok = len(re) == (len(back) if key == "joint" else int((back["condition"] == key).sum())) and not bad
        checks.add(f"{key}: labelled h5ad read back equals the CSV (cells, label columns)",
                   f"{re.n_obs} cells; columns differing: {';'.join(bad) or 'none'}", ok)
    run.table(checks.frame(), "canonical_checks")
    run.finish({"dct_label": DCT_LABEL, "dct_label_alternative": DCT_LABEL_ALT, "sources": SOURCES,
                "canonical_columns": list(canon.columns), "row_order": "dct_intake joint object order",
                "checks_pass": checks.ok}, status="completed" if checks.ok else "stopped_failed_checks")
    if not checks.ok:
        raise SystemExit("A read-back check failed; see dct_labels_canonical_checks.csv.")


if __name__ == "__main__":
    main()
