#!/usr/bin/env python
"""deposit step 1: build the Zenodo-ready processed data (no upload, no re-analysis).

Analysis plan
(scope approved by the authors in the response to the reviewers Nov 2026; plan approved "OK, run it", 13 Nov 2026).

The script only reshapes stored results under public names:

  Inputs (read-only; ``deposit_lib.SOURCES``)
  - kidney: kidney_object object (matrix, layers, var, stored X_umap / X_pca / graph), kidney_annotation labels, qc_mito MitoCarta fraction;
  - DCT: dct_intake objects (layers, var; joint object with its two normalization layers and 20 PCs), dct_labels labelled objects
    (the scaled X behind the PCA, X_umap, the graph and its 20 PCs), dct_labels canonical labels, dct_prolif per-cell
    proliferation test (cs_t31nf_*), dct_reviewer MitoCarta fraction; dct_intake search settings and normalization constants.

  Outputs
  - results/h5ad/zenodo/: four .h5ad objects, three CSV files per object, data_dictionary.csv, README.md,
    SHA256SUMS.txt (public names; no deposit_ prefix, by the plan).
  - results/tables/deposit/: deposit_build_checks.csv, deposit_public_name_mapping.csv,
    deposit_data_dictionary.csv, deposit_deposit_files.csv, deposit_step_build_manifest.json.

Nothing is written when an input check fails. The deposited files are checked by deposit_validate.py, which
reloads them in a new process.

Command (project root):  python scripts/deposit_build.py [--overwrite]
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
import sys
from importlib import metadata

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import scp_common as cc
import cluster_de_lib as t5
import deposit_lib as lib

STEP = "build"
TOL_X = 1e-4          # scaled X against the z-score recomputed from ln1p_intensity
TOL_X_SOURCES = 1e-5  # scaled X of the dct_labels object against the dct_intake object
TOL_LAYER = 1e-4      # normalization layers against their definition (float32 values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="deposit: build the Zenodo-ready processed data.")
    cc.add_common_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def categorical(values, order) -> pd.Categorical:
    """Categorical with the present values in the given order; a value outside ``order`` is an error."""
    values = pd.Series(values).astype(object)
    if values.isna().any():
        raise ValueError("unmapped label value")
    extra = set(values) - set(order)
    if extra:
        raise ValueError(f"label values outside the public list: {sorted(extra)}")
    present = set(values)
    return pd.Categorical(values.to_numpy(), categories=[v for v in order if v in present])


def numbered(values, prefix: str = "") -> pd.Categorical:
    """Cluster IDs as text, categories in numeric order."""
    ints = pd.Series(values).astype(float).astype(int)
    return pd.Categorical([f"{prefix}{i}" for i in ints], categories=[f"{prefix}{i}" for i in sorted(set(ints))])


def clean_text(series) -> np.ndarray:
    """Text column with missing values as empty text."""
    values = pd.Series(series).astype(object)
    values = values.where(values.notna(), "").astype(str)
    return values.where(values != "nan", "").to_numpy()


def f32(matrix) -> np.ndarray:
    return np.ascontiguousarray(cc.dense(matrix), dtype=np.float32)


def plain(value):
    """numpy scalars and arrays of an uns entry as plain Python values."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def counts_text(series, order) -> str:
    counts = pd.Series(series).astype(str).value_counts()
    return "; ".join(f"{label} {int(counts[label])}" for label in order if label in counts.index)


def matrix_checks(checks: t5.Checks, name: str, adata) -> None:
    """Relations between the matrices and the counts of an assembled object."""
    intensity = np.asarray(adata.layers["intensity"])
    ln1p = np.asarray(adata.layers["ln1p_intensity"])
    detected = intensity > 0
    diff = float(np.max(np.abs(np.log1p(intensity.astype(np.float64)) - ln1p)))
    checks.add(f"{name}: ln1p_intensity equals ln(1 + intensity) (max abs diff)", f"{diff:.2e}", diff < 1e-5, "< 1e-5")
    z = cc.scale_like_scanpy(ln1p)
    zdiff = float(np.max(np.abs(z - np.asarray(adata.X))))
    checks.add(f"{name}: X equals the clipped z-score of ln1p_intensity (max abs diff)", f"{zdiff:.2e}", zdiff < TOL_X,
               f"< {TOL_X:g}")
    n_prot = detected.sum(axis=1)
    bad = int((n_prot != adata.obs["n_proteins"].to_numpy()).sum())
    checks.add(f"{name}: n_proteins equals the detected protein groups of the intensity layer", bad, bad == 0, "0 mismatches")
    total = intensity.sum(axis=1, dtype=np.float64)
    rel = float(np.max(np.abs(total / adata.obs["total_intensity"].to_numpy() - 1)))
    checks.add(f"{name}: total_intensity equals the row sums of the intensity layer (max rel diff)", f"{rel:.2e}",
               rel < 1e-6, "< 1e-6")
    bad = int((detected.sum(axis=0) != adata.var["n_cells_detected"].to_numpy()).sum())
    checks.add(f"{name}: n_cells_detected equals the detected cells of the intensity layer", bad, bad == 0, "0 mismatches")
    mito = adata.obs["mitocarta_intensity_fraction"].to_numpy()
    ok = bool(np.isfinite(mito).all() and (mito >= 0).all() and (mito <= 1).all())
    checks.add(f"{name}: mitocarta_intensity_fraction is complete and within 0-1",
               f"{float(mito.min()):.4f}-{float(mito.max()):.4f}", ok, "no missing value")
    na_names = sorted(set(adata.var_names) & {"NA", "NaN", "nan", "None", "NULL", "null", "N/A", "n/a", ""})
    checks.add(f"{name}: no protein name that CSV readers take for a missing value", na_names, not na_names, "none")
    checks.add(f"{name}: protein names and protein groups are unique",
               f"{adata.var_names.is_unique} / {adata.var['protein_group'].is_unique}",
               bool(adata.var_names.is_unique and adata.var["protein_group"].is_unique), "unique")
    for column in ("cell_type", "cluster_evidence"):
        per_cluster = adata.obs.groupby("cluster", observed=True)[column].nunique()
        checks.add(f"{name}: one {column} per cluster", f"{int((per_cluster == 1).sum())}/{len(per_cluster)}",
                   bool((per_cluster == 1).all()), "every cluster")


def var_frame(source_var: pd.DataFrame, names) -> pd.DataFrame:
    var = pd.DataFrame(index=pd.Index(np.asarray(names, dtype=str), name="protein"))
    var["protein_group"] = source_var["protein_group"].astype(str).to_numpy()
    var["gene_symbol"] = clean_text(source_var["pg_genes"])
    var["description"] = clean_text(source_var["protein_description"])
    var["n_cells_detected"] = source_var["n_cells_detected"].to_numpy(dtype=np.int64)
    return var


def neighbors_uns(source) -> dict:
    params = plain(dict(source.uns["neighbors"]["params"]))
    return {"connectivities_key": "connectivities", "distances_key": "distances", "params": params}


# ---------------------------------------------------------------------------
# Kidney
# ---------------------------------------------------------------------------
def build_kidney(run: cc.Run, checks: t5.Checks, norm: dict):
    spec = lib.SPECS["kidney"]
    name = "kidney"
    source = ad.read_h5ad(lib.use(run, "kidney_h5ad"))
    ids = source.obs_names.astype(str)
    labels = pd.read_csv(lib.use(run, "kidney_labels"), dtype={"cell_id": str}).set_index("cell_id")
    qc = pd.read_csv(lib.use(run, "kidney_qc"), dtype={"cell_id": str}).set_index("cell_id")

    checks.add(f"{name}: source object cells x protein groups", f"{source.n_obs} x {source.n_vars}",
               source.shape == (spec.n_cells, spec.n_proteins), f"{spec.n_cells} x {spec.n_proteins}")
    same = list(labels.index) == list(ids)
    checks.add(f"{name}: label table holds the cells of the object in the same order", same, same, "identical")
    covered = set(ids) <= set(qc.index)
    checks.add(f"{name}: QC table holds every cell", covered, covered, "all cells")
    labels, qc = labels.loc[ids], qc.loc[ids]

    codes = labels["kidney_t15_label"].astype(str)
    display = codes.map(lib.KIDNEY_DISPLAY)
    counts = display.value_counts().to_dict()
    checks.add(f"{name}: cell_type counts", counts_text(display, lib.KIDNEY_ORDER),
               counts == lib.EXPECTED["kidney"]["cell_type"], "the project notes counts")
    evidence = labels["kidney_t15_confidence"].astype(str).map(lib.EVIDENCE_KIDNEY)
    checks.add(f"{name}: cluster_evidence counts", counts_text(evidence, lib.EVIDENCE_ORDER), None)

    obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
    obs["cell_type"] = categorical(display, lib.KIDNEY_ORDER)
    obs["cluster"] = numbered(labels["t15_cluster"])
    obs["cluster_evidence"] = categorical(evidence, lib.EVIDENCE_ORDER)
    obs["n_proteins"] = source.obs["n_detected_proteins"].to_numpy(dtype=np.int64)
    obs["total_intensity"] = source.obs["total_intensity"].to_numpy(dtype=np.float64)
    obs["mitocarta_intensity_fraction"] = qc["mitocarta_intensity_fraction"].to_numpy(dtype=np.float64)
    n_clusters = obs["cluster"].nunique()
    checks.add(f"{name}: clusters", n_clusters, n_clusters == lib.EXPECTED["kidney"]["n_clusters"], "23")
    bad = int((qc["n_detected_proteins"].to_numpy() != obs["n_proteins"].to_numpy()).sum())
    checks.add(f"{name}: QC table counts the same proteins per cell as the object", bad, bad == 0, "0 mismatches")

    pca = np.asarray(source.obsm["X_pca"])
    checks.add(f"{name}: stored X_pca components (the first 20 are kept)", pca.shape[1], pca.shape[1] >= 20, ">= 20")
    none_key = None in source.layers and source.layers[None] is source.X
    checks.add(f"{name}: the key None of the source's layers is anndata's view of X, not a stored layer", none_key, None)
    adata = ad.AnnData(
        X=f32(source.X), obs=obs, var=var_frame(source.var, source.var_names),
        layers={"intensity": f32(source.layers["intensity"]), "ln1p_intensity": f32(source.layers["ln1p_intensity"])},
        obsm={"X_pca": np.ascontiguousarray(pca[:, :20], dtype=np.float32),
              "X_umap": np.ascontiguousarray(source.obsm["X_umap"], dtype=np.float32)},
        obsp={"connectivities": sparse.csr_matrix(source.obsp["connectivities"]),
              "distances": sparse.csr_matrix(source.obsp["distances"])},
        uns={"about": lib.about_text(spec, norm), "neighbors": neighbors_uns(source)},
    )
    matrix_checks(checks, name, adata)
    return adata


# ---------------------------------------------------------------------------
# DCT, one diet
# ---------------------------------------------------------------------------
def prolif_columns(checks: t5.Checks, name: str, obs: pd.DataFrame, test: pd.DataFrame, slc12a3_detected) -> None:
    """Add the columns of the per-cell proliferation test and cell_type_prolif to ``obs``."""
    values = set(test["cs_t31nf_cycling"].astype(str))
    checks.add(f"{name}: values of the cycling column of the test", sorted(values), values <= {"Cycling", "Non-cycling"},
               "Cycling / Non-cycling")
    prolif = (test["cs_t31nf_label"].astype(str) == "ProLIF").to_numpy()
    cycling = (test["cs_t31nf_cycling"].astype(str) == "Cycling").to_numpy()
    score = test["cs_t31nf_prolif_D"].to_numpy(dtype=np.float64)
    q = test["cs_t31nf_prolif_q"].to_numpy(dtype=np.float64)
    complete = bool(np.isfinite(score).all() and np.isfinite(q).all())
    checks.add(f"{name}: proliferation score and q value for every cell", complete, complete, "no missing value")
    same = bool(((q < 0.05) == cycling).all())
    checks.add(f"{name}: cycling equals proliferation_q < 0.05", same, same, "every cell")
    same = bool((prolif == (cycling & np.asarray(slc12a3_detected))).all())
    checks.add(f"{name}: prolif equals cycling and Slc12a3 detected", same, same, "every cell")
    obs["prolif"] = prolif
    obs["cycling"] = cycling
    obs["proliferation_score"] = score
    obs["proliferation_q"] = q
    groups = np.where(prolif, lib.PROLIF, obs["cell_type"].astype(str).to_numpy())
    obs["cell_type_prolif"] = categorical(groups, lib.DCT_ORDER)


def id_columns(checks: t5.Checks, name: str, obs: pd.DataFrame, source_obs: pd.DataFrame, ids) -> None:
    """condition, sort_plate and well, checked against the run names."""
    parsed = pd.DataFrame([cc.parse_dct_cell_id(i) for i in ids], index=ids)
    condition = source_obs["condition"].astype(str).to_numpy()
    plate = source_obs["plate"].astype(str).to_numpy()
    well = source_obs["well"].astype(str).to_numpy()
    same = bool((parsed["condition"].to_numpy() == condition).all() and (parsed["plate"].to_numpy() == plate).all()
                and (parsed["well"].to_numpy() == well).all())
    checks.add(f"{name}: condition, sort plate and well equal the run names", same, same, "every cell")
    obs["condition"] = categorical(pd.Series(condition).map(lib.CONDITION), lib.CONDITION_ORDER)
    obs["sort_plate"] = categorical(plate, lib.PLATE_ORDER)
    obs["well"] = pd.Series(well).astype(int).to_numpy(dtype=np.int64)


def build_diet(run: cc.Run, checks: t5.Checks, key: str, labels: pd.DataFrame, test: pd.DataFrame, qc: pd.DataFrame,
               norm: dict):
    spec = lib.SPECS[key]
    code = spec.conditions[0]
    low = code.lower()
    prefix = "cs_ctrl_deg" if code == "Ctrl" else "cs_zerok_pipeline"
    name = key
    matrix = ad.read_h5ad(lib.use(run, f"{low}_h5ad"))          # dct_intake: layers, var
    figure = ad.read_h5ad(lib.use(run, f"{low}_fig_h5ad"))      # dct_labels: scaled X, embedding, graph
    ids = matrix.obs_names.astype(str)

    checks.add(f"{name}: source object cells x protein groups", f"{matrix.n_obs} x {matrix.n_vars}",
               matrix.shape == (spec.n_cells, spec.n_proteins), f"{spec.n_cells} x {spec.n_proteins}")
    same = list(figure.obs_names) == list(ids) and list(figure.var_names) == list(matrix.var_names)
    checks.add(f"{name}: matrix object and embedding object hold the same cells and proteins in the same order", same,
               same, "identical")
    same = all(np.array_equal(np.asarray(matrix.layers[k]), np.asarray(figure.layers[k])) for k in lib.LAYERS)
    checks.add(f"{name}: intensity layers of the two source objects are equal", same, same, "exact")
    diff = float(np.max(np.abs(f32(matrix.X) - f32(figure.X))))
    checks.add(f"{name}: scaled X of the embedding object against the matrix object (max abs diff; the first is written)",
               f"{diff:.2e}", diff < TOL_X_SOURCES, f"< {TOL_X_SOURCES:g}")
    for table, label in ((labels, "label table"), (test, "proliferation table"), (qc, "QC table")):
        covered = set(ids) <= set(table.index)
        checks.add(f"{name}: {label} holds every cell", covered, covered, "all cells")
    lab, tst, mito = labels.loc[ids], test.loc[ids], qc.loc[ids]
    same = bool((lab["condition"].astype(str) == code).all() and (tst["condition"].astype(str) == code).all())
    checks.add(f"{name}: tables give the condition of the object for every cell", same, same, code)
    for column in (f"{prefix}_label", f"{prefix}_cluster", f"{prefix}_confidence"):
        obj = figure.obs[column].astype(str).to_numpy()
        tab = lab[column]
        tab = tab.astype(float).astype(int).astype(str).to_numpy() if column.endswith("_cluster") else tab.astype(str).to_numpy()
        same = bool((obj == tab).all())
        checks.add(f"{name}: {column} of the labelled object equals the canonical table", same, same, "every cell")

    obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
    id_columns(checks, name, obs, matrix.obs, ids)
    display = lab[f"{prefix}_label"].astype(str).map(lib.DCT_DISPLAY)
    obs["cell_type"] = categorical(display, lib.DCT_ORDER)
    obs["cluster"] = numbered(lab[f"{prefix}_cluster"])
    evidence = lab[f"{prefix}_confidence"].astype(str).map(lib.EVIDENCE_DCT)
    obs["cluster_evidence"] = categorical(evidence, lib.EVIDENCE_ORDER)
    expected = lib.EXPECTED[key]
    checks.add(f"{name}: cell_type counts", counts_text(display, lib.DCT_ORDER),
               display.value_counts().to_dict() == expected["cell_type"], "the project notes counts")
    checks.add(f"{name}: cluster_evidence counts", counts_text(evidence, lib.EVIDENCE_ORDER),
               evidence.value_counts().to_dict() == expected["cluster_evidence"], "the project notes counts")
    checks.add(f"{name}: clusters", obs["cluster"].nunique(), obs["cluster"].nunique() == expected["n_clusters"], "13")

    intensity = f32(matrix.layers["intensity"])
    slc = list(matrix.var_names).index("Slc12a3")
    prolif_columns(checks, name, obs, tst, intensity[:, slc] > 0)
    checks.add(f"{name}: ProLIF cells", int(obs["prolif"].sum()), int(obs["prolif"].sum()) == expected["n_prolif"],
               str(expected["n_prolif"]))
    checks.add(f"{name}: cell_type_prolif counts", counts_text(obs["cell_type_prolif"], lib.DCT_ORDER),
               obs["cell_type_prolif"].astype(str).value_counts().to_dict() == expected["cell_type_prolif"],
               "plan: groups of Figure 3A")
    if key == "zerok":
        transferred = lab["cs_zerok_transfer_label"].astype(str).map(lib.DCT_DISPLAY)
        obs["cell_type_transferred"] = categorical(transferred, lib.DCT_ORDER)
        obs["cell_type_transferred_vote_fraction"] = lab["cs_zerok_transfer_confidence"].to_numpy(dtype=np.float64)
        checks.add(f"{name}: cell_type_transferred counts", counts_text(transferred, lib.DCT_ORDER),
                   transferred.value_counts().to_dict() == expected["cell_type_transferred"], "the project notes counts")
    obs["n_proteins"] = matrix.obs["n_detected"].to_numpy(dtype=np.int64)
    obs["total_intensity"] = matrix.obs["total_intensity"].to_numpy(dtype=np.float64)
    obs["mitocarta_intensity_fraction"] = mito["mitocarta_intensity_fraction"].to_numpy(dtype=np.float64)
    bad = int((mito["n_detected_proteins"].to_numpy() != obs["n_proteins"].to_numpy()).sum())
    checks.add(f"{name}: QC table counts the same proteins per cell as the object", bad, bad == 0, "0 mismatches")
    obs = obs[list(spec.obs)]

    pca = np.asarray(figure.obsm["X_pca_neighbors"])
    checks.add(f"{name}: components of the graph PCA", pca.shape[1], pca.shape[1] == 20, "20")
    adata = ad.AnnData(
        X=f32(figure.X), obs=obs, var=var_frame(matrix.var, matrix.var_names),
        layers={"intensity": intensity, "ln1p_intensity": f32(matrix.layers["ln1p_intensity"])},
        obsm={"X_pca": np.ascontiguousarray(pca, dtype=np.float32),
              "X_umap": np.ascontiguousarray(figure.obsm["X_umap"], dtype=np.float32)},
        obsp={"connectivities": sparse.csr_matrix(figure.obsp["connectivities"]),
              "distances": sparse.csr_matrix(figure.obsp["distances"])},
        uns={"about": lib.about_text(spec, norm), "neighbors": neighbors_uns(figure)},
    )
    matrix_checks(checks, name, adata)
    return adata


# ---------------------------------------------------------------------------
# DCT, both diets
# ---------------------------------------------------------------------------
def build_both(run: cc.Run, checks: t5.Checks, labels: pd.DataFrame, test: pd.DataFrame, qc: pd.DataFrame,
               diets: dict, norm: dict):
    spec = lib.SPECS["both"]
    name = "both"
    joint = ad.read_h5ad(lib.use(run, "joint_h5ad"))
    ids = joint.obs_names.astype(str)
    checks.add(f"{name}: source object cells x protein groups", f"{joint.n_obs} x {joint.n_vars}",
               joint.shape == (spec.n_cells, spec.n_proteins), f"{spec.n_cells} x {spec.n_proteins}")
    expected_ids = list(diets["control"].obs_names) + list(diets["zerok"].obs_names)
    same = list(ids) == expected_ids
    checks.add(f"{name}: cells are the Control cells followed by the Zero-K cells", same, same, "identical")
    lab, tst, mito = labels.loc[ids], test.loc[ids], qc.loc[ids]

    obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
    id_columns(checks, name, obs, joint.obs, ids)
    is_ctrl = (joint.obs["condition"].astype(str) == "Ctrl").to_numpy()
    code_label = np.where(is_ctrl, lab["cs_ctrl_deg_label"].astype(object), lab["cs_zerok_pipeline_label"].astype(object))
    same = bool((pd.Series(code_label).astype(str).to_numpy() == lab["dct_label"].astype(str).to_numpy()).all())
    checks.add(f"{name}: label by condition equals dct_label of the canonical table", same, same, "every cell")
    display = pd.Series(code_label).astype(str).map(lib.DCT_DISPLAY)
    obs["cell_type"] = categorical(display, lib.DCT_ORDER)
    cluster_id = np.where(is_ctrl, lab["cs_ctrl_deg_cluster"], lab["cs_zerok_pipeline_cluster"]).astype(float).astype(int)
    cluster_text = [f"{lib.CONDITION['Ctrl' if c else 'ZeroK']}_{i}" for c, i in zip(is_ctrl, cluster_id)]
    order = [f"{lib.CONDITION[code]}_{i}" for code, mask in (("Ctrl", is_ctrl), ("ZeroK", ~is_ctrl))
             for i in sorted(set(cluster_id[mask]))]
    obs["cluster"] = pd.Categorical(cluster_text, categories=order)
    confidence = np.where(is_ctrl, lab["cs_ctrl_deg_confidence"].astype(object),
                          lab["cs_zerok_pipeline_confidence"].astype(object))
    evidence = pd.Series(confidence).astype(str).map(lib.EVIDENCE_DCT)
    obs["cluster_evidence"] = categorical(evidence, lib.EVIDENCE_ORDER)
    expected = lib.EXPECTED["both"]
    checks.add(f"{name}: cell_type counts", counts_text(display, lib.DCT_ORDER),
               display.value_counts().to_dict() == expected["cell_type"], "sum of the the project notes counts")
    checks.add(f"{name}: clusters", obs["cluster"].nunique(), obs["cluster"].nunique() == expected["n_clusters"], "26")

    intensity = f32(joint.layers["intensity"])
    slc = list(joint.var_names).index("Slc12a3")
    prolif_columns(checks, name, obs, tst, intensity[:, slc] > 0)
    checks.add(f"{name}: ProLIF cells", int(obs["prolif"].sum()), int(obs["prolif"].sum()) == expected["n_prolif"], "100")
    checks.add(f"{name}: cell_type_prolif counts", counts_text(obs["cell_type_prolif"], lib.DCT_ORDER),
               obs["cell_type_prolif"].astype(str).value_counts().to_dict() == expected["cell_type_prolif"],
               "sum of the two diets")
    obs["n_proteins"] = joint.obs["n_detected"].to_numpy(dtype=np.int64)
    obs["total_intensity"] = joint.obs["total_intensity"].to_numpy(dtype=np.float64)
    obs["mitocarta_intensity_fraction"] = mito["mitocarta_intensity_fraction"].to_numpy(dtype=np.float64)
    obs = obs[list(spec.obs)]

    var = var_frame(joint.var, joint.var_names)
    var["n_cells_detected_control"] = joint.var["n_cells_ctrl"].to_numpy(dtype=np.int64)
    var["n_cells_detected_zeroK"] = joint.var["n_cells_zerok"].to_numpy(dtype=np.int64)
    detected = intensity > 0
    bad = int((detected[is_ctrl].sum(axis=0) != var["n_cells_detected_control"].to_numpy()).sum()
              + (detected[~is_ctrl].sum(axis=0) != var["n_cells_detected_zeroK"].to_numpy()).sum())
    checks.add(f"{name}: detected cells per diet equal the intensity layer", bad, bad == 0, "0 mismatches")

    # the two normalization layers against the definitions printed in the data dictionary
    ln1p = np.log1p(intensity.astype(np.float64))
    aligned = np.asarray(joint.layers["norm__condition_align"], dtype=np.float64)
    target = ln1p + np.where(detected & ~is_ctrl[:, None], norm["delta"], 0.0)
    diff = float(np.max(np.abs(aligned - target)))
    checks.add(f"{name}: condition-aligned layer equals ln1p with the Zero-K shift of the dictionary (max abs diff)",
               f"{diff:.2e}", diff < TOL_LAYER, f"< {TOL_LAYER:g}")
    with np.errstate(divide="ignore"):
        ln = np.where(detected, np.log(np.where(detected, intensity.astype(np.float64), 1.0)), np.nan)
    medians = np.nanmedian(ln, axis=1)
    centred = np.where(detected, ln - medians[:, None] + np.median(medians) + norm["shift"], 0.0)
    cell_median = np.asarray(joint.layers["norm__cell_median"], dtype=np.float64)
    diff = float(np.max(np.abs(cell_median - centred)))
    checks.add(f"{name}: median-centred layer equals the definition of the dictionary (max abs diff)", f"{diff:.2e}",
               diff < TOL_LAYER, f"< {TOL_LAYER:g}")
    smallest = float(cell_median[detected].min())
    checks.add(f"{name}: smallest detected value of the median-centred layer", f"{smallest:.4f}",
               abs(smallest - 0.01) < 1e-4, "0.01")

    pca = np.asarray(joint.obsm["X_pca__none"])
    checks.add(f"{name}: principal components of the variant without normalization", pca.shape[1], pca.shape[1] == 20, "20")
    adata = ad.AnnData(
        X=f32(joint.X), obs=obs, var=var,
        layers={"intensity": intensity, "ln1p_intensity": f32(joint.layers["ln1p_intensity"]),
                lib.LAYER_CELL_MEDIAN: f32(joint.layers["norm__cell_median"]),
                lib.LAYER_ALIGNED: f32(joint.layers["norm__condition_align"])},
        obsm={"X_pca": np.ascontiguousarray(pca, dtype=np.float32)},
        uns={"about": lib.about_text(spec, norm)},
    )
    matrix_checks(checks, name, adata)

    # the same cell carries the same values in the object of its diet
    shared = [c for c in lib.DCT_OBS if c not in ("cluster", "n_proteins", "total_intensity")]
    for key, diet in diets.items():
        sub = adata[list(diet.obs_names)]
        same = all((sub.obs[c].astype(str).to_numpy() == diet.obs[c].astype(str).to_numpy()).all() for c in shared)
        checks.add(f"{name}: obs values of the {key} cells equal the {key} object", same, same, "every shared column")
        cols = [list(diet.var_names).index(p) for p in adata.var_names]
        same = np.array_equal(np.asarray(sub.layers["intensity"]), np.asarray(diet.layers["intensity"])[:, cols])
        checks.add(f"{name}: intensities of the {key} cells equal the {key} object on the shared protein groups", same,
                   same, "exact")
    return adata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, STEP, args)
    checks = t5.Checks("build")

    names = lib.deposit_files()
    existing = sorted(p.name for p in lib.DEPOSIT_DIR.iterdir()) if lib.DEPOSIT_DIR.exists() else []
    foreign = sorted(set(existing) - set(names))
    if foreign:
        sys.exit(f"Unexpected files in {cc.rel(lib.DEPOSIT_DIR)}: {foreign}. Nothing was written.")
    if existing and not run.overwrite:
        sys.exit(f"{cc.rel(lib.DEPOSIT_DIR)} already holds {len(existing)} deposit files; rerun with --overwrite.")

    # ---------------------------------------------------------------- shared DCT tables and constants
    labels = pd.read_csv(lib.use(run, "dct_labels"), dtype={"cell_id": str}).set_index("cell_id")
    test = pd.read_csv(lib.use(run, "dct_prolif"), dtype={"cell_id": str}).set_index("cell_id")
    qc = pd.read_csv(lib.use(run, "dct_qc"), dtype={"cell_id": str}).set_index("cell_id")
    variants = pd.read_csv(lib.use(run, "dct_norm")).set_index("variant")
    norm = {"shift": float(variants.loc["cell_median", "uniform_shift"]),
            "delta": float(variants.loc["condition_align", "delta_ln"]),
            "n_delta": int(variants.loc["condition_align", "n_proteins_used_for_delta"])}
    checks.add("normalization constants (median-centring shift; Zero-K shift; protein groups behind it)",
               f"{norm['shift']:.4f}; {norm['delta']:.4f}; {norm['n_delta']}", None)
    search = pd.read_csv(lib.use(run, "dct_search")).set_index("setting")["combined_search"].astype(str).to_dict()
    for table, label in ((labels, "label table"), (test, "proliferation table"), (qc, "QC table")):
        checks.add(f"DCT {label}: rows and unique cells", f"{len(table)} / {table.index.nunique()}",
                   len(table) == 1431 and table.index.is_unique, "1431 / 1431")

    # ---------------------------------------------------------------- objects
    objects = {"kidney": build_kidney(run, checks, norm)}
    for key in ("control", "zerok"):
        objects[key] = build_diet(run, checks, key, labels, test, qc, norm)
    objects["both"] = build_both(run, checks, labels, test, qc, {k: objects[k] for k in ("control", "zerok")}, norm)

    for key, adata in objects.items():
        spec = lib.SPECS[key]
        fields = (list(adata.obs.columns) == list(spec.obs) and list(adata.var.columns) == list(spec.var)
                  and lib.layer_names(adata) == sorted(spec.layers) and sorted(adata.obsm.keys()) == sorted(spec.obsm)
                  and sorted(adata.obsp.keys()) == sorted(spec.obsp) and sorted(adata.uns.keys()) == sorted(spec.uns))
        checks.add(f"{key}: fields of the assembled object equal the public field list", fields, fields, "exact")
        checks.add(f"{key}: assembled object", f"{adata.n_obs} x {adata.n_vars}",
                   adata.shape == (spec.n_cells, spec.n_proteins), f"{spec.n_cells} x {spec.n_proteins}")
        missing = int(adata.obs.isna().sum().sum() + adata.var.isna().sum().sum())
        checks.add(f"{key}: missing values in obs and var", missing, missing == 0, "0")

    dictionary = pd.DataFrame(lib.dictionary_rows(norm), columns=list(lib.DICTIONARY_COLUMNS))
    undocumented = []
    for key, adata in objects.items():
        spec = lib.SPECS[key]
        for field in ["X", "cell_id", "protein", *spec.layers, *spec.obsm, *spec.obsp, *spec.uns, *spec.obs, *spec.var]:
            if not lib.dictionary_covers(dictionary, spec.h5ad, field):
                undocumented.append(f"{spec.h5ad}:{field}")
    checks.add("data dictionary names every field of the four objects", undocumented or "all", not undocumented, "all")

    if not checks.ok:
        run.table(checks.frame(), "build_checks")
        run.finish(status="failed: input checks")
        sys.exit("Input checks failed; no deposit file was written.")

    # ---------------------------------------------------------------- write the deposit
    written = {}
    for key, adata in objects.items():
        spec = lib.SPECS[key]
        path = lib.deposit_path(spec.h5ad, run.overwrite)
        adata.write_h5ad(path, compression="gzip", convert_strings_to_categoricals=False)
        written[spec.h5ad] = path
        path = lib.deposit_path(spec.matrix_csv, run.overwrite)
        lib.write_csv_gz(lib.matrix_table(adata), path)
        written[spec.matrix_csv] = path
        path = lib.deposit_path(spec.cell_csv, run.overwrite)
        lib.write_csv(lib.cell_table(adata, spec), path)
        written[spec.cell_csv] = path
        path = lib.deposit_path(spec.protein_csv, run.overwrite)
        lib.write_csv(lib.protein_table(adata), path)
        written[spec.protein_csv] = path
        print(f"Wrote {spec.stem}: h5ad and three CSV files", flush=True)

    path = lib.deposit_path("data_dictionary.csv", run.overwrite)
    lib.write_csv(dictionary, path)
    written["data_dictionary.csv"] = path

    raw_cells = {"Control": lib.SPECS["control"].n_raw_cells, "Zero-K": lib.SPECS["zerok"].n_raw_cells}
    count_rows = []
    for key, order in (("kidney", lib.KIDNEY_ORDER), ("control", lib.DCT_ORDER), ("zerok", lib.DCT_ORDER)):
        count_rows.append(f"| {lib.SPECS[key].title} | {counts_text(objects[key].obs['cell_type'], order)} |")
    facts = {
        "sizes": {name: p.stat().st_size for name, p in written.items()},
        "software": {"Python": sys.version.split()[0], **{p: metadata.version(p) for p in
                                                           ("anndata", "scanpy", "igraph", "numpy", "pandas", "scipy")}},
        "search": search,
        "n_raw_control": raw_cells["Control"], "n_raw_zerok": raw_cells["Zero-K"],
        "count_rows": "\n".join(count_rows),
        "n_prolif_control": int(objects["control"].obs["prolif"].sum()),
        "n_prolif_zerok": int(objects["zerok"].obs["prolif"].sum()),
        **{f"range_{key}": (int(objects[key].obs["n_proteins"].min()), int(objects[key].obs["n_proteins"].max()))
           for key in ("kidney", "control", "zerok", "both")},
    }
    path = lib.deposit_path("README.md", run.overwrite)
    lib.write_text(path, lib.deposit_readme(facts))
    written["README.md"] = path

    path = lib.deposit_path("SHA256SUMS.txt", run.overwrite)
    lib.write_text(path, "".join(f"{cc.sha256(written[name])}  {name}\n" for name in sorted(written)))
    written["SHA256SUMS.txt"] = path

    # ---------------------------------------------------------------- project records
    files = pd.DataFrame([{"file": name, "path": cc.rel(written[name]), "bytes": written[name].stat().st_size,
                           "sha256": cc.sha256(written[name])} for name in names])
    mapping = pd.DataFrame(lib.MAPPING, columns=["object", "part", "public_name", "project_source", "project_column",
                                                 "value_mapping"])
    run.table(checks.frame(), "build_checks")
    run.table(mapping, "public_name_mapping")
    run.table(dictionary, "data_dictionary")
    run.table(files, "deposit_files")
    for row in files.itertuples(index=False):
        print(f"Deposit file: {row.path} ({row.bytes / 1e6:.2f} MB)")
    run.finish({
        "deposit_dir": cc.rel(lib.DEPOSIT_DIR),
        "deposit_files": {row.path: row.sha256 for row in files.itertuples(index=False)},
        "objects": {key: {"file": lib.SPECS[key].h5ad, "n_cells": int(a.n_obs), "n_protein_groups": int(a.n_vars),
                          "obs": list(a.obs.columns), "var": list(a.var.columns), "layers": lib.layer_names(a),
                          "obsm": sorted(a.obsm.keys()), "obsp": sorted(a.obsp.keys()), "uns": sorted(a.uns.keys())}
                    for key, a in objects.items()},
        "normalization_constants": norm,
        "software_in_deposit_readme": facts["software"],
        "h5ad_write": "anndata write_h5ad(compression='gzip', convert_strings_to_categoricals=False); dense float32",
        "tolerances": {"X_vs_zscore": TOL_X, "X_between_sources": TOL_X_SOURCES, "normalization_layers": TOL_LAYER},
    })


if __name__ == "__main__":
    main()
