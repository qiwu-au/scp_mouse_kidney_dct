#!/usr/bin/env python
"""dct_intake step 3: Ctrl, ZeroK and joint objects of the combined search; protein and per-cell tables.



Inputs (read-only): data/DCT/DCT_Ctrl_ZeroK_combine_1535cells.csv.
Outputs:
  results/tables/dct_intake/
    dct_intake_proteins.csv               one row per protein group of the export: detection and retention per
                                          condition, in joint
    dct_intake_one_condition_only.csv     groups retained in one condition only, with detection counts (not
                                          tested later; the zerok_response_core rule)
    dct_intake_retention_summary.csv      groups by retention class
    dct_intake_dct_cells.csv              per-cell table (all 1,535 cells; retained flag)
    dct_intake_normalization_variants.csv the three zerok_core variants with their diagnostics
    dct_intake_objects_checks.csv         checks of the normalization layers and embeddings
  results/h5ad/dct_intake_DCT_ctrl.h5ad, dct_intake_DCT_zerok.h5ad, dct_intake_DCT_joint.h5ad

Objects (intake_core layout): layers['intensity'] (raw PG.Quantity, missing = 0),
layers['ln1p_intensity'], .X = per-protein z-scored ln(1+intensity) (ddof = 1)
clipped at +/-10 (each condition scaled on its own; the joint .X across both).
var_names = cleaned first PG.Genes symbol (first accession when empty; _1, _2
for repeats), set once over all export rows so that a group has the same name
in every object. Joint object: proteins retained in both conditions, the union
of retained cells (Ctrl then ZeroK), layers norm__cell_median and
norm__condition_align (variant 'none' = ln1p_intensity; zerok_core_lib), and
obsm X_pca__<variant> (20 PCs) and X_umap__<variant> (zerok_core joint embedding:
z-score over all cells, arpack PCA, k = 10, seed 42).

Command (project root):  python scripts/dct_intake_objects.py
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
import intake_core_lib as t17
import zerok_core_lib as t21
import dct_intake_lib as lib

VAR_NAMES_RULE = ("cleaned first PG.Genes symbol (first accession if empty); repeats get _1, _2 in file order "
                  "over all rows of the combined export (the same name in every dct_intake object)")


def provenance(**extra) -> dict:
    return {"generated_by": cc.GENERATED_BY, "task": lib.TASK, "created_utc": cc.now_utc(),
            "source_csv": cc.rel(lib.COMBINED_INPUT),
            "filters": f"cells >= {cc.MIN_CELL_PROTEINS} protein groups > 0 (all rows); proteins > 0 in >= "
                       f"{cc.MIN_PROTEIN_CELLS} retained cells (per condition)",
            "layers_intensity": "raw Spectronaut PG.Quantity, missing (NaN in the export) = 0",
            "var_names": VAR_NAMES_RULE, "search": lib.ONE_SEARCH_NOTE, "design": " ".join(lib.DESIGN_FACTS),
            **extra}


def finish_object(adata):
    """build_object stores provenance under uns['intake_core']; dct_intake objects carry it as uns['dct_intake']."""
    adata.uns["dct_intake"] = adata.uns.pop("intake_core")
    return adata


def condition_object(raw: t17.Raw, filt: t17.Filtered, proteins: pd.DataFrame):
    intensity = t17.intensity_matrix(raw, filt)
    obs = raw.cells.loc[filt.keep_cells, lib.OBS_COLUMNS].copy()
    obs["n_detected"] = (intensity > 0).sum(axis=1)
    obs["total_intensity"] = intensity.sum(axis=1, dtype=np.float64)
    obs["n_detected_all_rows"] = filt.n_detected_all_rows[filt.keep_cells]
    own = proteins.loc[filt.keep_proteins]
    var = pd.DataFrame({
        "gene_name": own["gene_name"].to_numpy(),
        "protein_group": own["protein_group"].to_numpy(),
        "pg_genes": own["pg_genes"].to_numpy(),
        "protein_description": own["protein_description"].to_numpy(),
        "n_cells_detected": (intensity > 0).sum(axis=0),
        "file_row": own["file_row"].to_numpy(),
        "in_joint": own["in_joint"].to_numpy(dtype=bool),
        "n_cells_ctrl": own["n_cells_ctrl"].to_numpy(),
        "n_cells_zerok": own["n_cells_zerok"].to_numpy(),
        "retained_ctrl": own["retained_ctrl"].to_numpy(dtype=bool),
        "retained_zerok": own["retained_zerok"].to_numpy(dtype=bool),
    }, index=pd.Index(own["var_name"].to_numpy(), name="protein"))
    obs.index.name = "cell_id"
    adata = t17.build_object(intensity, obs, var, provenance(
        condition=raw.condition, X="per-protein z-scored ln(1+intensity) within this condition (ddof=1), clipped "
                                   "at +/-10"))
    return finish_object(adata)


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_intake: objects, protein and per-cell tables.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "objects", args)
    t_start = time.time()
    checks = t5.Checks("dct_combined_joint")

    comb = lib.read_combined(run)
    raws = {cond: comb.raw(cond) for cond in lib.CONDITIONS}
    filts = {cond: t17.apply_filters(raws[cond]) for cond in lib.CONDITIONS}

    # Protein tables.
    proteins = lib.protein_table(comb.meta, filts)
    run.table(proteins, "proteins")
    one = proteins.loc[proteins["retention"].isin(["Ctrl only", "ZeroK only"]),
                       ["retention", "file_row", "protein_group", "pg_genes", "gene_name", "var_name",
                        "n_cells_ctrl", "n_cells_zerok", "n_cells_all_ctrl", "n_cells_all_zerok"]]
    run.table(one, "one_condition_only")
    retention = proteins.groupby("retention").size().reindex(
        ["both conditions", "Ctrl only", "ZeroK only", "neither"], fill_value=0).rename("n_protein_groups")
    run.table(retention.reset_index(), "retention_summary")
    run.note("Retention: " + "; ".join(f"{k} {v}" for k, v in retention.items()))

    # Condition objects.
    objects = {cond: condition_object(raws[cond], filts[cond], proteins) for cond in lib.CONDITIONS}
    for cond in lib.CONDITIONS:
        run.h5ad(objects[cond], f"DCT_{lib.FILE_KEY[cond]}")
        run.note(f"{cond} object: {objects[cond].n_obs} cells x {objects[cond].n_vars} proteins.")

    # Joint object.
    joint_rows = proteins.loc[proteins["in_joint"], "file_row"].to_numpy()
    jp = proteins.loc[proteins["in_joint"]]
    blocks, obs_blocks = [], []
    for cond in lib.CONDITIONS:
        raw, filt = raws[cond], filts[cond]
        block = np.nan_to_num(raw.values[np.ix_(filt.keep_cells, joint_rows)], nan=0.0).astype(np.float32)
        obs = raw.cells.loc[filt.keep_cells, lib.OBS_COLUMNS].copy()
        obs["n_detected"] = (block > 0).sum(axis=1)
        obs["total_intensity"] = block.sum(axis=1, dtype=np.float64)
        obs["n_detected_condition"] = objects[cond].obs["n_detected"].to_numpy()
        obs["total_intensity_condition"] = objects[cond].obs["total_intensity"].to_numpy()
        blocks.append(block)
        obs_blocks.append(obs)
    intensity = np.vstack(blocks)
    obs = pd.concat(obs_blocks)
    obs.index.name = "cell_id"
    var = pd.DataFrame({
        "gene_name": jp["gene_name"].to_numpy(),
        "protein_group": jp["protein_group"].to_numpy(),
        "pg_genes": jp["pg_genes"].to_numpy(),
        "protein_description": jp["protein_description"].to_numpy(),
        "file_row": jp["file_row"].to_numpy(),
        "n_cells_detected": (intensity > 0).sum(axis=0),
        "n_cells_ctrl": jp["n_cells_ctrl"].to_numpy(),
        "n_cells_zerok": jp["n_cells_zerok"].to_numpy(),
    }, index=pd.Index(jp["var_name"].to_numpy(), name="protein"))
    joint = finish_object(t17.build_object(intensity, obs, var, provenance(
        cells="union of the retained cells of both conditions (Ctrl first, then ZeroK, file order)",
        proteins="protein groups retained (>= 2 retained cells) in both conditions",
        X="per-protein z-scored ln(1+intensity) across both conditions (ddof=1), clipped at +/-10")))

    # Normalization variants and joint embedding (zerok_core step 3, unchanged functions).
    condition = joint.obs["condition"].astype(str).to_numpy()
    is_zerok = condition == "ZeroK"
    x64 = intensity.astype(np.float64)
    detected = x64 > 0
    norms, infos = {}, {}
    for name in t21.VARIANTS:
        norms[name], infos[name] = t21.normalize_joint(x64, condition, name)
        checks.add(f"{name}: every detected value > 0 and every missing value 0",
                   f"min detected {norms[name][detected].min():.4g}; "
                   f"max missing {np.abs(norms[name][~detected]).max():.4g}",
                   bool(norms[name][detected].min() > 0) and bool(np.abs(norms[name][~detected]).max() == 0))
    diff_ln1p = float(np.abs(norms["none"] - joint.layers["ln1p_intensity"].astype(np.float64)).max())
    checks.add("variant 'none' equals the stored ln1p layer", f"max abs diff {diff_ln1p:.2e}", diff_ln1p < 1e-6,
               "< 1e-6")
    ctrl_unchanged = float(np.abs(norms["condition_align"][~is_zerok] - norms["none"][~is_zerok]).max())
    checks.add("condition_align leaves the control cells unchanged", f"max abs diff {ctrl_unchanged:.2e}",
               ctrl_unchanged == 0.0, "0 (control is the reference)")
    run.note(f"condition_align delta = {infos['condition_align']['delta_ln']:+.4f} ln units from "
             f"{infos['condition_align']['n_proteins_used_for_delta']} proteins "
             f"(uniform guard shift {infos['condition_align']['uniform_shift_applied']:.3g}); cell_median uniform "
             f"shift {infos['cell_median']['uniform_shift']:.3f} ({infos['cell_median']['n_detected_nonpositive_centred']} "
             f"detected values in {infos['cell_median']['n_cells_with_nonpositive_centred']} cells <= 0 after centring)")
    for name in t21.VARIANTS:
        if name != t21.BASELINE_VARIANT:
            joint.layers[f"norm__{name}"] = norms[name].astype(np.float32)

    cell_ids = joint.obs_names.astype(str)
    var_names = joint.var_names.astype(str)
    for name, matrix in norms.items():
        h, umap_xy = t21.joint_embedding(matrix, cell_ids, var_names)
        pca = np.asarray(h.obsm["X_pca"])[:, :t21.N_PCS]
        joint.obsm[f"X_pca__{name}"] = pca.astype(np.float32)
        joint.obsm[f"X_umap__{name}"] = np.asarray(umap_xy, dtype=np.float32)
        checks.add(f"{name}: PCA and UMAP shapes", f"{pca.shape}; {np.asarray(umap_xy).shape}",
                   pca.shape == (joint.n_obs, t21.N_PCS) and np.asarray(umap_xy).shape == (joint.n_obs, 2))
    joint.uns["dct_intake_embedding"] = {
        "variants": dict(t21.VARIANTS), "baseline_variant": t21.BASELINE_VARIANT, "n_pcs": t21.N_PCS,
        "k_neighbors": t21.K_NEIGHBORS, "seed": lib.SEED,
        "pca": "arpack, 50 comps, seed 42 (preprocessing_lib.build_embedding); first 20 stored",
        "note": "variant 'none' is layers['ln1p_intensity']; the others are layers['norm__<variant>']"}
    t21.write_h5ad(run, joint, "DCT_joint")
    run.note(f"Joint object: {joint.n_obs} cells x {joint.n_vars} proteins retained in both conditions.")
    run.table(t21.variant_table(infos), "normalization_variants")
    run.table(checks.frame(), "objects_checks")

    # Per-cell table (all cells, both conditions).
    cell_rows = []
    for cond in lib.CONDITIONS:
        raw, filt = raws[cond], filts[cond]
        values = np.nan_to_num(raw.values, nan=0.0)
        ret = values[:, filt.keep_proteins]
        jnt = values[:, joint_rows]
        frame = raw.cells[lib.OBS_COLUMNS].copy()
        frame["retained"] = filt.keep_cells
        frame["n_detected_all_rows"] = filt.n_detected_all_rows
        frame["n_detected"] = (ret > 0).sum(axis=1)
        frame["total_intensity"] = ret.sum(axis=1)
        frame["n_detected_joint"] = (jnt > 0).sum(axis=1)
        frame["total_intensity_joint"] = jnt.sum(axis=1)
        cell_rows.append(frame.reset_index())
    cells = pd.concat(cell_rows, ignore_index=True)
    cells = cells[["cell_id", "cell_key", "condition", "plate", "well", "file_column_index", "retained",
                   "n_detected", "total_intensity", "n_detected_all_rows", "n_detected_joint",
                   "total_intensity_joint"]]
    run.table(cells, "dct_cells")

    run.finish(parameters={"MIN_CELL_PROTEINS": cc.MIN_CELL_PROTEINS, "MIN_PROTEIN_CELLS": cc.MIN_PROTEIN_CELLS,
                           "var_names": VAR_NAMES_RULE, "variants": dict(t21.VARIANTS),
                           "condition_align": infos["condition_align"],
                           "cell_median": {k: v for k, v in infos["cell_median"].items() if k != "definition"},
                           "n_pcs": t21.N_PCS, "k_neighbors": t21.K_NEIGHBORS,
                           "pca": "arpack, 50 comps, seed 42 (preprocessing_lib.build_embedding)",
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_intake_objects_checks.csv.")


if __name__ == "__main__":
    main()
