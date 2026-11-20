#!/usr/bin/env python
"""deposit step 2: validate the Zenodo deposit by reloading every file in a new process.

Analysis plan, step 5.
Run after deposit_build.py. It reads the 19 files of results/h5ad/zenodo/ from disk and compares them with

  - the plan's field lists and the label counts of the project notes;
  - the two raw Spectronaut exports (every value of every object, and the two filters);
  - the figure objects (kidney: kidney_object object; DCT: dct_labels labelled objects, the dct_labels canonical table and the dct_prolif
    per-cell table of the Figure 4e overlay; both diets: dct_intake joint object);
  - each other (CSV copies against the h5ad; the object of both diets against the two single-diet objects);
  - the 'not exported' list and the word list of the plan (every key, column name and text value);
  - the statements of the deposit's README and data dictionary (placeholders, graph from the stored principal
    components, clusters from the stored graph, transferred label from the object of both diets).

Outputs: results/tables/deposit/deposit_validation_checks.csv, deposit_word_scan.csv and
deposit_step_validate_manifest.json. Exit code 1 when a check fails. Nothing in the deposit is changed.

Command (project root):  python scripts/deposit_validate.py [--overwrite]
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
import fnmatch
import gzip
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import cluster_de_lib as t5
import deposit_lib as lib

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
STEP = "validate"
D = lib.DEPOSIT_DIR
TOL_X = 1e-4
TOL_CSV_UMAP = 1e-4     # UMAP coordinates stored as decimal text in project tables
LEIDEN = {"kidney": 2.0, "control": 1.0, "zerok": 1.0}
INVERSE_KIDNEY = {v: k for k, v in lib.KIDNEY_DISPLAY.items()}
INVERSE_DCT = {v: k for k, v in lib.DCT_DISPLAY.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="deposit: validate the Zenodo deposit (reload every file).")
    cc.add_common_args(parser)
    return parser.parse_args()


def dense32(matrix) -> np.ndarray:
    return np.asarray(cc.dense(matrix), dtype=np.float32)


def counts(series) -> dict:
    return {str(k): int(v) for k, v in pd.Series(series).astype(str).value_counts().items()}


def sparse_equal(a, b) -> bool:
    a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
    return a.shape == b.shape and (a != b).nnz == 0


# ---------------------------------------------------------------------------
# 1. Files and checksums
# ---------------------------------------------------------------------------
def check_files(checks: t5.Checks, run: cc.Run) -> None:
    present = sorted(p.name for p in D.iterdir())
    expected = sorted(lib.deposit_files())
    checks.add("files: the deposit holds the expected files and nothing else", f"{len(present)} files",
               present == expected, f"{len(expected)} names of the plan")
    for name in expected:
        if (D / name).exists():
            run.use_input(D / name, "deposit file")
    lines = [ln for ln in (D / "SHA256SUMS.txt").read_text(encoding="utf-8").split("\n") if ln]
    listed = dict(reversed(ln.split("  ", 1)) for ln in lines)
    others = sorted(set(expected) - {"SHA256SUMS.txt"})
    checks.add("files: SHA256SUMS.txt lists every other file once", f"{len(listed)} lines",
               sorted(listed) == others and len(lines) == len(others), f"{len(others)} files")
    wrong = [name for name, digest in listed.items() if not (D / name).exists() or cc.sha256(D / name) != digest]
    checks.add("files: every listed checksum equals the file on disk", wrong or "all equal", not wrong, "all")
    for name in expected:
        size = (D / name).stat().st_size if (D / name).exists() else 0
        checks.add(f"files: size of {name} (bytes)", size, size > 0, "> 0")


# ---------------------------------------------------------------------------
# 2. Structure, labels and matrices of one object
# ---------------------------------------------------------------------------
def check_structure(checks: t5.Checks, spec: lib.Spec, a) -> None:
    n = spec.key
    checks.add(f"{n}: cells x protein groups", f"{a.n_obs} x {a.n_vars}", a.shape == (spec.n_cells, spec.n_proteins),
               f"{spec.n_cells} x {spec.n_proteins}")
    checks.add(f"{n}: obs columns", list(a.obs.columns), list(a.obs.columns) == list(spec.obs), "public list, in order")
    checks.add(f"{n}: var columns", list(a.var.columns), list(a.var.columns) == list(spec.var), "public list, in order")
    layer_keys = lib.layer_names(a)
    checks.add(f"{n}: layers", layer_keys, layer_keys == sorted(spec.layers), "public list")
    import h5py

    with h5py.File(D / spec.h5ad, "r") as handle:
        on_disk = sorted(handle["layers"].keys())
        top = sorted(handle.keys())
    checks.add(f"{n}: layers stored in the file (no layer named None)", on_disk,
               on_disk == sorted(spec.layers) and "None" not in on_disk, "public list")
    checks.add(f"{n}: top-level groups of the file", top, None)
    checks.add(f"{n}: obsm keys", sorted(a.obsm.keys()), sorted(a.obsm.keys()) == sorted(spec.obsm), "public list")
    checks.add(f"{n}: obsp keys", sorted(a.obsp.keys()), sorted(a.obsp.keys()) == sorted(spec.obsp), "public list")
    checks.add(f"{n}: uns keys", sorted(a.uns.keys()), sorted(a.uns.keys()) == sorted(spec.uns), "public list")
    checks.add(f"{n}: varm and varp are empty", f"{len(a.varm)} / {len(a.varp)}", len(a.varm) == 0 and len(a.varp) == 0, "0 / 0")
    names_ok = a.obs.index.name == "cell_id" and a.var.index.name == "protein"
    checks.add(f"{n}: index names", f"{a.obs.index.name} / {a.var.index.name}", names_ok, "cell_id / protein")
    unique = a.obs_names.is_unique and a.var_names.is_unique and a.var["protein_group"].is_unique
    checks.add(f"{n}: cell IDs, protein names and protein groups are unique", unique, unique, "unique")
    run_names = all(lib.RUN_NAME_RE.fullmatch(i) for i in a.obs_names)
    checks.add(f"{n}: every cell ID is a Spectronaut run name", run_names, run_names, "all")
    dtypes = {k: str(np.asarray(a.layers[k]).dtype) for k in layer_keys}
    dtypes["X"] = str(np.asarray(a.X).dtype)
    dense = not sparse.issparse(a.X) and not any(sparse.issparse(a.layers[k]) for k in layer_keys)
    checks.add(f"{n}: X and layers are dense 32-bit floating point", dtypes,
               dense and set(dtypes.values()) == {"float32"}, "float32")
    finite = bool(np.isfinite(np.asarray(a.X)).all() and all(np.isfinite(np.asarray(v)).all() for v in a.layers.values())
                  and all(np.isfinite(np.asarray(v)).all() for v in a.obsm.values()))
    checks.add(f"{n}: no missing or infinite value in X, layers and obsm", finite, finite, "all finite")
    missing = int(a.obs.isna().sum().sum())
    checks.add(f"{n}: missing values in obs (labels included)", missing, missing == 0, "0")
    missing = int(a.var.isna().sum().sum())
    checks.add(f"{n}: missing values in var", missing, missing == 0, "0")
    label_columns = [c for c in spec.obs if c.startswith(("cell_type", "cluster", "condition", "sort_plate"))
                     and not c.endswith("vote_fraction")]
    categorical = all(isinstance(a.obs[c].dtype, pd.CategoricalDtype) for c in label_columns)
    empty = sum(int((a.obs[c].astype(str).str.strip() == "").sum()) for c in label_columns)
    unused = [c for c in label_columns if (a.obs[c].value_counts() == 0).any()]
    checks.add(f"{n}: label columns are categorical, without empty labels or unused categories",
               f"{categorical}; empty {empty}; unused {unused}", categorical and empty == 0 and not unused, "all")
    for column, kind in (("n_proteins", "i"), ("total_intensity", "f"), ("mitocarta_intensity_fraction", "f"),
                         ("prolif", "b"), ("cycling", "b"), ("proliferation_score", "f"), ("proliferation_q", "f"),
                         ("well", "i"), ("cell_type_transferred_vote_fraction", "f")):
        if column in a.obs:
            checks.add(f"{n}: dtype of {column}", str(a.obs[column].dtype), a.obs[column].dtype.kind == kind, kind)
    text = all(pd.api.types.is_string_dtype(a.var[c]) for c in ("protein_group", "gene_symbol", "description"))
    checks.add(f"{n}: var text columns are text", text, text, "text")
    # the data dictionary's statement on protein names
    names = pd.Series(a.var_names.astype(str))
    base = names.str.replace(r"_\d+$", "", regex=True)
    first = a.var["gene_symbol"].astype(str).str.split(";").str[0].str.strip().reset_index(drop=True)
    cleaned = first.str.lower().str.capitalize()
    accession = a.var["protein_group"].astype(str).str.split(";").str[0].str.strip().reset_index(drop=True)
    has = first != ""
    fallback = base[~has] == ("Nan" if n == "kidney" else accession[~has])
    wrong = int((base[has] != cleaned[has]).sum()) + int((~fallback).sum())
    checks.add(f"{n}: protein names follow the rule of the data dictionary (first gene symbol, or the fallback; "
               f"{int((~has).sum())} groups without a first symbol, {int((names != base).sum())} names with a suffix)",
               wrong, wrong == 0, "0 exceptions")
    about = a.uns["about"]
    same = int(about["n_cells"]) == a.n_obs and int(about["n_protein_groups"]) == a.n_vars
    checks.add(f"{n}: uns['about'] gives the numbers of the object", f"{about['n_cells']} x {about['n_protein_groups']}",
               same, "equal")


def check_labels(checks: t5.Checks, spec: lib.Spec, a) -> None:
    n = spec.key
    expected = lib.EXPECTED[n]
    obs = a.obs
    checks.add(f"{n}: cell_type counts", counts(obs["cell_type"]), counts(obs["cell_type"]) == expected["cell_type"],
               "the project notes")
    checks.add(f"{n}: clusters", obs["cluster"].nunique(), obs["cluster"].nunique() == expected["n_clusters"],
               str(expected["n_clusters"]))
    for column in ("cell_type", "cluster_evidence"):
        per = obs.groupby("cluster", observed=True)[column].nunique()
        checks.add(f"{n}: one {column} per cluster", f"{int((per == 1).sum())}/{len(per)}", bool((per == 1).all()), "all")
    if "cluster_evidence" in expected:
        checks.add(f"{n}: cluster_evidence counts", counts(obs["cluster_evidence"]),
                   counts(obs["cluster_evidence"]) == expected["cluster_evidence"], "the project notes")
    else:
        checks.add(f"{n}: cluster_evidence counts", counts(obs["cluster_evidence"]), None)
    if n == "kidney":
        unknown = bool(((obs["cell_type"] == "Unknown") == (obs["cluster_evidence"] == "no marker support")).all())
        checks.add(f"{n}: 'no marker support' exactly for the cells labelled Unknown", unknown, unknown, "every cell")
        return
    unresolved = bool(((obs["cell_type"] == lib.NOT_RESOLVED)
                       == (obs["cluster_evidence"] == "no marker support (subtype not resolved)")).all())
    checks.add(f"{n}: 'no marker support' exactly for the cells with the subtype not resolved", unresolved, unresolved,
               "every cell")
    checks.add(f"{n}: ProLIF cells", int(obs["prolif"].sum()), int(obs["prolif"].sum()) == expected["n_prolif"],
               str(expected["n_prolif"]))
    checks.add(f"{n}: cell_type_prolif counts", counts(obs["cell_type_prolif"]),
               counts(obs["cell_type_prolif"]) == expected["cell_type_prolif"], "plan")
    derived = np.where(obs["prolif"].to_numpy(), lib.PROLIF, obs["cell_type"].astype(str).to_numpy())
    same = bool((derived == obs["cell_type_prolif"].astype(str).to_numpy()).all())
    checks.add(f"{n}: cell_type_prolif equals cell_type with the ProLIF cells set to ProLIF", same, same, "every cell")
    same = bool(((obs["proliferation_q"] < 0.05) == obs["cycling"]).all())
    checks.add(f"{n}: cycling equals proliferation_q < 0.05", same, same, "every cell")
    slc = dense32(a.layers["intensity"])[:, list(a.var_names).index("Slc12a3")] > 0
    same = bool((obs["prolif"].to_numpy() == (obs["cycling"].to_numpy() & slc)).all())
    checks.add(f"{n}: prolif equals cycling and Slc12a3 detected", same, same, "every cell")
    conditions = counts(obs["condition"])
    want = {"control": {"Control": 729}, "zerok": {"Zero-K": 702}, "both": {"Control": 729, "Zero-K": 702}}[n]
    checks.add(f"{n}: condition counts", conditions, conditions == want, str(want))
    parsed = pd.DataFrame([cc.parse_dct_cell_id(i) for i in a.obs_names])
    same = bool((parsed["condition"].map(lib.CONDITION).to_numpy() == obs["condition"].astype(str).to_numpy()).all()
                and (parsed["plate"].to_numpy() == obs["sort_plate"].astype(str).to_numpy()).all()
                and (parsed["well"].astype(int).to_numpy() == obs["well"].to_numpy()).all())
    checks.add(f"{n}: condition, sort_plate and well equal the run names", same, same, "every cell")
    checks.add(f"{n}: sort_plate counts", counts(obs["sort_plate"]), None)
    if n == "zerok":
        checks.add(f"{n}: cell_type_transferred counts", counts(obs["cell_type_transferred"]),
                   counts(obs["cell_type_transferred"]) == expected["cell_type_transferred"], "the project notes")
        votes = obs["cell_type_transferred_vote_fraction"]
        checks.add(f"{n}: vote fraction range", f"{votes.min()}-{votes.max()}", bool(votes.between(0.1, 1.0).all()),
                   "0.1-1")


def check_matrices(checks: t5.Checks, spec: lib.Spec, a) -> None:
    n = spec.key
    intensity = dense32(a.layers["intensity"])
    ln1p = dense32(a.layers["ln1p_intensity"])
    detected = intensity > 0
    checks.add(f"{n}: intensities are 0 or positive", float(intensity.min()), bool((intensity >= 0).all()), ">= 0")
    diff = float(np.max(np.abs(np.log1p(intensity.astype(np.float64)) - ln1p)))
    checks.add(f"{n}: ln1p_intensity equals ln(1 + intensity) (max abs diff)", f"{diff:.2e}", diff < 1e-5, "< 1e-5")
    zdiff = float(np.max(np.abs(cc.scale_like_scanpy(ln1p) - np.asarray(a.X))))
    checks.add(f"{n}: X equals the clipped z-score of ln1p_intensity (max abs diff)", f"{zdiff:.2e}", zdiff < TOL_X,
               f"< {TOL_X:g}")
    clipped = float(np.abs(np.asarray(a.X)).max())
    checks.add(f"{n}: largest absolute value of X", f"{clipped:.4f}", clipped <= 10 + 1e-6, "<= 10")
    for key in spec.layers[2:]:
        same = np.array_equal(dense32(a.layers[key]) > 0, detected)
        checks.add(f"{n}: layer {key} is 0 exactly where intensity is 0", same, same, "same pattern")
    bad = int((detected.sum(axis=1) != a.obs["n_proteins"].to_numpy()).sum())
    checks.add(f"{n}: n_proteins equals the detected protein groups per cell", bad, bad == 0, "0 mismatches")
    rel = float(np.max(np.abs(intensity.sum(axis=1, dtype=np.float64) / a.obs["total_intensity"].to_numpy() - 1)))
    checks.add(f"{n}: total_intensity equals the row sums (max rel diff)", f"{rel:.2e}", rel < 1e-6, "< 1e-6")
    bad = int((detected.sum(axis=0) != a.var["n_cells_detected"].to_numpy()).sum())
    checks.add(f"{n}: n_cells_detected equals the detected cells per protein group", bad, bad == 0, "0 mismatches")
    # The cell filter counts all protein groups of the export (checked against the raw export below), so the count
    # over the object's own protein groups is reported, not tested: one Control cell has 100 groups in the export and
    # 99 of the 5,897 groups of the object of both diets (first validation run, 13 Nov 2026).
    rare = int((a.var["n_cells_detected"] < 2).sum())
    checks.add(f"{n}: protein groups detected in fewer than 2 cells of the object", rare, rare == 0, "0")
    low = a.obs.index[a.obs["n_proteins"] < 100]
    checks.add(f"{n}: cells with fewer than 100 of the object's own protein groups (the cell filter counts all protein "
               f"groups of the export)", f"{len(low)}" + (f": {list(low)}" if len(low) else ""), None)
    checks.add(f"{n}: detected entries (share of the matrix)", f"{int(detected.sum())} ({detected.mean():.3f})", None)
    checks.add(f"{n}: protein groups per cell, range", f"{int(a.obs['n_proteins'].min())}-{int(a.obs['n_proteins'].max())}", None)
    if n == "both":
        is_ctrl = (a.obs["condition"] == "Control").to_numpy()
        bad = int((detected[is_ctrl].sum(axis=0) != a.var["n_cells_detected_control"].to_numpy()).sum()
                  + (detected[~is_ctrl].sum(axis=0) != a.var["n_cells_detected_zeroK"].to_numpy()).sum())
        checks.add(f"{n}: detected cells per diet equal the intensity layer", bad, bad == 0, "0 mismatches")
    shapes = {k: tuple(np.asarray(v).shape) for k, v in a.obsm.items()}
    want = {"X_pca": (a.n_obs, 20), **({"X_umap": (a.n_obs, 2)} if spec.has_umap else {})}
    checks.add(f"{n}: obsm shapes", shapes, shapes == want, str(want))


# ---------------------------------------------------------------------------
# 3. Round trip to the raw Spectronaut exports
# ---------------------------------------------------------------------------
class Raw:
    """A raw PG.Quantity export: protein groups in rows, cells in columns, missing = NaN."""

    def __init__(self, path):
        data = pd.read_csv(path, low_memory=False)
        self.meta = data.iloc[:, :3]
        self.groups = self.meta["PG.ProteinGroups"].astype(str).to_numpy()
        self.genes = self.meta["PG.Genes"].astype(object).where(self.meta["PG.Genes"].notna(), "").astype(str).to_numpy()
        self.descriptions = (self.meta["PG.ProteinDescriptions"].astype(object)
                             .where(self.meta["PG.ProteinDescriptions"].notna(), "").astype(str).to_numpy())
        self.cells = data.columns[3:].astype(str).to_numpy()
        self.values = data.iloc[:, 3:].to_numpy(dtype=np.float64)       # groups x cells
        self.row = {g: i for i, g in enumerate(self.groups)}
        self.col = {c: i for i, c in enumerate(self.cells)}


def check_raw(checks: t5.Checks, spec: lib.Spec, a, raw: Raw, kept_groups: dict) -> None:
    n = spec.key
    if spec.conditions:
        in_set = np.array([cc.parse_dct_cell_id(c)["condition"] in spec.conditions for c in raw.cells])
    else:
        in_set = np.ones(len(raw.cells), dtype=bool)
    checks.add(f"{n}: runs of the data set in the raw export / protein groups in the export",
               f"{int(in_set.sum())} / {len(raw.groups)}",
               int(in_set.sum()) == spec.n_raw_cells and len(raw.groups) == spec.n_raw_groups,
               f"{spec.n_raw_cells} / {spec.n_raw_groups}")
    known = all(c in raw.col for c in a.obs_names) and all(g in raw.row for g in a.var["protein_group"])
    checks.add(f"{n}: every cell ID and protein group is in the raw export", known, known, "all")
    if not known:
        return
    cols = np.array([raw.col[c] for c in a.obs_names])
    rows = np.array([raw.row[g] for g in a.var["protein_group"]])
    block = raw.values[np.ix_(rows, cols)].T                           # cells x groups
    intensity = dense32(a.layers["intensity"])
    same = np.array_equal(np.nan_to_num(block, nan=0.0).astype(np.float32), intensity)
    checks.add(f"{n}: every intensity equals the raw export ({intensity.size:,} values; missing = 0)", same, same, "exact")
    same = np.array_equal(np.isnan(block), intensity == 0)
    checks.add(f"{n}: 0 exactly where the raw export has no value", same, same, "same pattern")
    same = bool((raw.genes[rows] == a.var["gene_symbol"].astype(str).to_numpy()).all()
                and (raw.descriptions[rows] == a.var["description"].astype(str).to_numpy()).all())
    checks.add(f"{n}: gene_symbol and description equal PG.Genes and PG.ProteinDescriptions of the export", same, same,
               "every protein group")
    # the two filters, recomputed from the export
    detected_all = ~np.isnan(raw.values) & (raw.values > 0)
    if n == "both":
        want_groups = [g for g in raw.groups if g in kept_groups["control"] and g in kept_groups["zerok"]]
        same = want_groups == list(a.var["protein_group"])
        checks.add(f"{n}: protein groups are those kept in both diets, in export order", f"{len(want_groups)}", same,
                   "identical")
        return
    keep_cells = in_set & (detected_all.sum(axis=0) >= cc.MIN_CELL_PROTEINS)
    same = list(raw.cells[keep_cells]) == list(a.obs_names)
    checks.add(f"{n}: cells are the runs with at least 100 protein groups detected, in export order",
               f"{int(keep_cells.sum())} of {int(in_set.sum())}", same, "identical")
    keep_groups = detected_all[:, keep_cells].sum(axis=1) >= cc.MIN_PROTEIN_CELLS
    same = list(raw.groups[keep_groups]) == list(a.var["protein_group"])
    checks.add(f"{n}: protein groups are those detected in at least 2 kept cells, in export order",
               f"{int(keep_groups.sum())} of {len(raw.groups)}", same, "identical")
    kept_groups[n] = set(raw.groups[keep_groups])


# ---------------------------------------------------------------------------
# 4. Figure objects and source tables
# ---------------------------------------------------------------------------
def check_kidney_reference(checks: t5.Checks, run: cc.Run, a) -> None:
    n = "kidney"
    ref = ad.read_h5ad(lib.use(run, "kidney_h5ad"))
    same = list(ref.obs_names) == list(a.obs_names) and list(ref.var_names) == list(a.var_names)
    checks.add(f"{n}: cells and proteins equal the figure object, in order", same, same, "identical")
    same = np.array_equal(np.asarray(ref.obsm["X_umap"], dtype=np.float32), np.asarray(a.obsm["X_umap"]))
    checks.add(f"{n}: X_umap equals the figure object", same, same, "exact")
    same = np.array_equal(np.asarray(ref.obsm["X_pca"], dtype=np.float32)[:, :20], np.asarray(a.obsm["X_pca"]))
    checks.add(f"{n}: X_pca equals the first 20 components of the figure object", same, same, "exact")
    same = sparse_equal(ref.obsp["distances"], a.obsp["distances"]) and sparse_equal(ref.obsp["connectivities"],
                                                                                    a.obsp["connectivities"])
    checks.add(f"{n}: neighbour graph equals the figure object", same, same, "exact")
    same = np.array_equal(dense32(ref.X), np.asarray(a.X)) and all(
        np.array_equal(dense32(ref.layers[k]), dense32(a.layers[k])) for k in lib.LAYERS)
    checks.add(f"{n}: X and layers equal the figure object", same, same, "exact")
    labels = pd.read_csv(lib.use(run, "kidney_labels"), dtype={"cell_id": str}).set_index("cell_id").loc[a.obs_names]
    same = bool((a.obs["cell_type"].astype(str).map(INVERSE_KIDNEY).to_numpy()
                 == labels["kidney_t15_label"].astype(str).to_numpy()).all()
                and (a.obs["cluster"].astype(str).to_numpy() == labels["t15_cluster"].astype(int).astype(str).to_numpy()).all()
                and (a.obs["cluster_evidence"].astype(str).to_numpy()
                     == labels["kidney_t15_confidence"].astype(str).map(lib.EVIDENCE_KIDNEY).to_numpy()).all())
    checks.add(f"{n}: cell_type, cluster and cluster_evidence equal the label table cell by cell", same, same, "712/712")
    qc = pd.read_csv(lib.use(run, "kidney_qc"), dtype={"cell_id": str}).set_index("cell_id").loc[a.obs_names]
    same = np.array_equal(qc["mitocarta_intensity_fraction"].to_numpy(), a.obs["mitocarta_intensity_fraction"].to_numpy())
    checks.add(f"{n}: mitocarta_intensity_fraction equals the QC table", same, same, "exact")


def check_diet_reference(checks: t5.Checks, run: cc.Run, key: str, a, labels, test, overlay, qc) -> None:
    n = key
    code = lib.SPECS[key].conditions[0]
    prefix = "cs_ctrl_deg" if code == "Ctrl" else "cs_zerok_pipeline"
    ref = ad.read_h5ad(lib.use(run, f"{code.lower()}_fig_h5ad"))
    same = list(ref.obs_names) == list(a.obs_names) and list(ref.var_names) == list(a.var_names)
    checks.add(f"{n}: cells and proteins equal the figure object, in order", same, same, "identical")
    same = np.array_equal(np.asarray(ref.obsm["X_umap"], dtype=np.float32), np.asarray(a.obsm["X_umap"]))
    checks.add(f"{n}: X_umap equals the figure object", same, same, "exact")
    same = np.array_equal(np.asarray(ref.obsm["X_pca_neighbors"], dtype=np.float32), np.asarray(a.obsm["X_pca"]))
    checks.add(f"{n}: X_pca equals the principal components of the figure object's graph", same, same, "exact")
    same = sparse_equal(ref.obsp["distances"], a.obsp["distances"]) and sparse_equal(ref.obsp["connectivities"],
                                                                                    a.obsp["connectivities"])
    checks.add(f"{n}: neighbour graph equals the figure object", same, same, "exact")
    same = np.array_equal(dense32(ref.X), np.asarray(a.X)) and all(
        np.array_equal(dense32(ref.layers[k]), dense32(a.layers[k])) for k in lib.LAYERS)
    checks.add(f"{n}: X and layers equal the figure object", same, same, "exact")
    lab, tst, mito = labels.loc[a.obs_names], test.loc[a.obs_names], qc.loc[a.obs_names]
    umap = np.asarray(a.obsm["X_umap"], dtype=np.float64)
    diff = float(np.max(np.abs(lab[["umap1_own", "umap2_own"]].to_numpy() - umap)))
    checks.add(f"{n}: X_umap equals the coordinates of the canonical label table (max abs diff)", f"{diff:.2e}",
               diff < TOL_CSV_UMAP, f"< {TOL_CSV_UMAP:g}")
    fig = overlay[(overlay["run"] == "nofloor") & (overlay["condition"] == code)].set_index("cell_id").loc[a.obs_names]
    diff = float(np.max(np.abs(fig[["umap_1", "umap_2"]].to_numpy() - umap)))
    checks.add(f"{n}: X_umap equals the coordinates drawn in the DCT UMAP figure (max abs diff)", f"{diff:.2e}",
               diff < TOL_CSV_UMAP, f"< {TOL_CSV_UMAP:g}")
    same = bool((fig["overlay_group"].astype(str).map(lib.DCT_DISPLAY | {lib.PROLIF: lib.PROLIF}).to_numpy()
                 == a.obs["cell_type_prolif"].astype(str).to_numpy()).all())
    checks.add(f"{n}: cell_type_prolif equals the groups drawn in the DCT UMAP figure", same, same, "every cell")
    same = bool((a.obs["cell_type"].astype(str).map(INVERSE_DCT).to_numpy() == lab[f"{prefix}_label"].astype(str).to_numpy()).all()
                and (a.obs["cluster"].astype(str).to_numpy()
                     == lab[f"{prefix}_cluster"].astype(float).astype(int).astype(str).to_numpy()).all()
                and (a.obs["cluster_evidence"].astype(str).to_numpy()
                     == lab[f"{prefix}_confidence"].astype(str).map(lib.EVIDENCE_DCT).to_numpy()).all())
    checks.add(f"{n}: cell_type, cluster and cluster_evidence equal the label table cell by cell", same, same, "every cell")
    same = bool((a.obs["prolif"].to_numpy() == (tst["cs_t31nf_label"].astype(str) == "ProLIF").to_numpy()).all()
                and (a.obs["cycling"].to_numpy() == (tst["cs_t31nf_cycling"].astype(str) == "Cycling").to_numpy()).all()
                and np.array_equal(a.obs["proliferation_score"].to_numpy(), tst["cs_t31nf_prolif_D"].to_numpy())
                and np.array_equal(a.obs["proliferation_q"].to_numpy(), tst["cs_t31nf_prolif_q"].to_numpy()))
    checks.add(f"{n}: prolif, cycling, proliferation_score and proliferation_q equal the test table", same, same,
               "every cell")
    same = np.array_equal(mito["mitocarta_intensity_fraction"].to_numpy(), a.obs["mitocarta_intensity_fraction"].to_numpy())
    checks.add(f"{n}: mitocarta_intensity_fraction equals the QC table", same, same, "exact")
    if key == "zerok":
        same = bool((a.obs["cell_type_transferred"].astype(str).map(INVERSE_DCT).to_numpy()
                     == lab["cs_zerok_transfer_label"].astype(str).to_numpy()).all()
                    and np.array_equal(a.obs["cell_type_transferred_vote_fraction"].to_numpy(),
                                       lab["cs_zerok_transfer_confidence"].to_numpy()))
        checks.add(f"{n}: cell_type_transferred and its vote fraction equal the label table", same, same, "every cell")


def check_both_reference(checks: t5.Checks, run: cc.Run, a, objects: dict, qc) -> None:
    n = "both"
    ref = ad.read_h5ad(lib.use(run, "joint_h5ad"))
    same = list(ref.obs_names) == list(a.obs_names) and list(ref.var_names) == list(a.var_names)
    checks.add(f"{n}: cells and proteins equal the matrix of the diet comparison, in order", same, same, "identical")
    pairs = (("intensity", "intensity"), ("ln1p_intensity", "ln1p_intensity"), (lib.LAYER_CELL_MEDIAN, "norm__cell_median"),
             (lib.LAYER_ALIGNED, "norm__condition_align"))
    same = np.array_equal(dense32(ref.X), np.asarray(a.X)) and all(
        np.array_equal(dense32(ref.layers[r]), dense32(a.layers[p])) for p, r in pairs)
    checks.add(f"{n}: X and the four layers equal the matrix of the diet comparison", same, same, "exact")
    same = np.array_equal(np.asarray(ref.obsm["X_pca__none"], dtype=np.float32), np.asarray(a.obsm["X_pca"]))
    checks.add(f"{n}: X_pca equals the components of the variant without normalization", same, same, "exact")
    same = list(a.obs_names) == list(objects["control"].obs_names) + list(objects["zerok"].obs_names)
    checks.add(f"{n}: cells are the Control object's cells followed by the Zero-K object's cells", same, same, "identical")
    shared = [c for c in lib.DCT_OBS if c not in ("cluster", "n_proteins", "total_intensity")]
    for key in ("control", "zerok"):
        diet = objects[key]
        sub = a[list(diet.obs_names)]
        same = all((sub.obs[c].astype(str).to_numpy() == diet.obs[c].astype(str).to_numpy()).all() for c in shared)
        checks.add(f"{n}: obs values of the {key} cells equal the {key} object", same, same, "every shared column")
        diet_name = "Control" if key == "control" else "Zero-K"
        same = bool((sub.obs["cluster"].astype(str).to_numpy()
                     == (diet_name + "_" + diet.obs["cluster"].astype(str)).to_numpy()).all())
        checks.add(f"{n}: cluster of the {key} cells is the {key} object's cluster with the diet in front", same, same,
                   "every cell")
        inside = set(a.var_names) <= set(diet.var_names)
        checks.add(f"{n}: every protein group is in the {key} object", inside, inside, "all")
        if inside:
            cols = [list(diet.var_names).index(p) for p in a.var_names]
            same = np.array_equal(dense32(sub.layers["intensity"]), dense32(diet.layers["intensity"])[:, cols])
            checks.add(f"{n}: intensities of the {key} cells equal the {key} object on the shared protein groups", same,
                       same, "exact")
            same = bool((a.var["protein_group"].to_numpy() == diet.var["protein_group"].to_numpy()[cols]).all())
            checks.add(f"{n}: protein names point to the same protein groups as in the {key} object", same, same, "all")
    common = sorted(set(objects["control"].var_names) & set(objects["zerok"].var_names))
    checks.add(f"{n}: protein groups = those in both single-diet objects", len(common), common == sorted(a.var_names),
               "5897")
    mito = qc.loc[a.obs_names, "mitocarta_intensity_fraction"].to_numpy()
    same = np.array_equal(mito, a.obs["mitocarta_intensity_fraction"].to_numpy())
    checks.add(f"{n}: mitocarta_intensity_fraction equals the QC table", same, same, "exact")


# ---------------------------------------------------------------------------
# 5. CSV copies
# ---------------------------------------------------------------------------
def check_csv(checks: t5.Checks, spec: lib.Spec, a) -> None:
    n = spec.key
    cells = pd.read_csv(D / spec.cell_csv, dtype=str, keep_default_na=False)
    want = ["cell_id", *spec.obs] + (["UMAP1", "UMAP2"] if spec.has_umap else [])
    checks.add(f"{n}: columns of the cell table", list(cells.columns), list(cells.columns) == want, "cell_id, obs, UMAP")
    if list(cells.columns) == want:
        differing = [] if list(cells["cell_id"]) == list(a.obs_names) else ["cell_id"]
        for column in spec.obs:
            series = a.obs[column]
            if series.dtype.kind == "b":
                same = (cells[column] == series.map({True: "True", False: "False"}).to_numpy()).all()
            elif series.dtype.kind == "i":
                same = (cells[column].astype(np.int64).to_numpy() == series.to_numpy()).all()
            elif series.dtype.kind == "f":
                same = np.array_equal(cells[column].astype(np.float64).to_numpy(), series.to_numpy())
            else:
                same = (cells[column].to_numpy() == series.astype(str).to_numpy()).all()
            if not same:
                differing.append(column)
        if spec.has_umap:
            umap = np.asarray(a.obsm["X_umap"], dtype=np.float32)
            for i, column in enumerate(("UMAP1", "UMAP2")):
                if not np.array_equal(cells[column].astype(np.float64).to_numpy().astype(np.float32), umap[:, i]):
                    differing.append(column)
        checks.add(f"{n}: cell table equals the obs of the object ({len(cells)} rows)", differing or "all columns equal",
                   not differing and len(cells) == a.n_obs, "exact")

    proteins = pd.read_csv(D / spec.protein_csv, dtype=str, keep_default_na=False)
    want = ["protein", *spec.var]
    checks.add(f"{n}: columns of the protein table", list(proteins.columns), list(proteins.columns) == want, "protein, var")
    if list(proteins.columns) == want:
        differing = [] if list(proteins["protein"]) == list(a.var_names) else ["protein"]
        for column in spec.var:
            series = a.var[column]
            if series.dtype.kind == "i":
                same = (proteins[column].astype(np.int64).to_numpy() == series.to_numpy()).all()
            else:
                same = (proteins[column].to_numpy() == series.astype(str).to_numpy()).all()
            if not same:
                differing.append(column)
        checks.add(f"{n}: protein table equals the var of the object ({len(proteins)} rows)",
                   differing or "all columns equal", not differing and len(proteins) == a.n_vars, "exact")

    with gzip.open(D / spec.matrix_csv, "rt", encoding="utf-8") as handle:
        matrix = pd.read_csv(handle, dtype={c: str for c in lib.MATRIX_META}, keep_default_na=False, na_values=[""])
    same = list(matrix.columns) == [*lib.MATRIX_META, *a.obs_names]
    checks.add(f"{n}: columns of the intensity matrix are the four protein columns and the cell IDs in order", same, same,
               "identical")
    if same:
        meta_same = bool((matrix["protein"].to_numpy() == a.var_names.to_numpy()).all()
                         and (matrix["protein_group"].to_numpy() == a.var["protein_group"].astype(str).to_numpy()).all()
                         and (matrix["gene_symbol"].fillna("").to_numpy() == a.var["gene_symbol"].astype(str).to_numpy()).all()
                         and (matrix["description"].fillna("").to_numpy() == a.var["description"].astype(str).to_numpy()).all())
        checks.add(f"{n}: protein columns of the intensity matrix equal the var of the object", meta_same, meta_same,
                   "every row")
        values = matrix.iloc[:, len(lib.MATRIX_META):].to_numpy(dtype=np.float64).T        # cells x groups
        intensity = dense32(a.layers["intensity"])
        same = np.array_equal(np.nan_to_num(values, nan=0.0).astype(np.float32), intensity)
        checks.add(f"{n}: intensity matrix equals layers['intensity'] ({intensity.size:,} values)", same, same, "exact")
        same = np.array_equal(np.isnan(values), intensity == 0)
        checks.add(f"{n}: intensity matrix is empty exactly where the layer is 0", same, same, "same pattern")


# ---------------------------------------------------------------------------
# 6. Not exported; words; documentation
# ---------------------------------------------------------------------------
def h5ad_names(path) -> list[str]:
    """Every group, dataset and attribute name of an h5ad file."""
    import h5py

    names: set[str] = set()

    def visit(name, obj):
        names.update(name.split("/"))
        names.update(obj.attrs.keys())
        order = obj.attrs.get("column-order")
        if order is not None:
            names.update(str(v) for v in np.atleast_1d(order))

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
    return sorted(names)


def check_not_exported(checks: t5.Checks) -> None:
    for spec in lib.SPECS.values():
        names = set(h5ad_names(D / spec.h5ad))
        for csv_name in (spec.cell_csv, spec.protein_csv):
            names.update(pd.read_csv(D / csv_name, nrows=0).columns)
        with gzip.open(D / spec.matrix_csv, "rt", encoding="utf-8") as handle:
            header = pd.read_csv(handle, nrows=0).columns
        names.update(c for c in header if not lib.RUN_NAME_RE.fullmatch(c))
        bad = lib.not_exported_hits(names)
        checks.add(f"{spec.key}: no field of the 'not exported' list in the object or its CSV files "
                   f"({len(names)} names)", bad or "none", not bad, "none")


def check_words(checks: t5.Checks) -> pd.DataFrame:
    rows = []
    for name in lib.deposit_files():
        hits, n_texts = lib.scan_deposit_file(D / name)
        own = [h for h in hits if h["kind"] == "own text"]
        source = [h for h in hits if h["kind"] == "source text"]
        checks.add(f"words: {name} ({n_texts:,} names and text values scanned)",
                   "; ".join(sorted({f"{h['pattern']} at {h['where']}" for h in own})) or "no match", not own, "no match")
        if source:
            checks.add(f"words: {name}, matches inside Spectronaut / UniProt text (kept as exported)",
                       "; ".join(sorted({f"{h['pattern']}: {h['text']}" for h in source}))[:500], None)
        rows.extend(hits)
        rows.append({"file": name, "where": "(file)", "kind": "summary", "pattern": "",
                     "text": f"{n_texts} names and text values scanned; {len(own)} own-text matches; "
                             f"{len(source)} source-text matches"})
    return pd.DataFrame(rows, columns=["file", "where", "kind", "pattern", "text"])


def check_documentation(checks: t5.Checks, objects: dict) -> None:
    readme = (D / "README.md").read_text(encoding="utf-8")
    left = sorted(set(lib.PLACEHOLDER_RE.findall(readme)))
    checks.add("README: no placeholder left", left or "none", not left, "none")
    for name, text in (("Zenodo record", lib.ZENODO_URL), ("PRIDE dataset", lib.PRIDE_ACCESSION),
                       ("manuscript title", lib.MANUSCRIPT_TITLE), ("author list", lib.AUTHORS)):
        checks.add(f"README: {name} named", readme.count(text), text in readme, ">= 1")
    accessions = sorted(set(re.findall(r"PXD\d+", readme)))
    checks.add("README: PRIDE accessions named", accessions, accessions == [lib.PRIDE_ACCESSION],
               f"only {lib.PRIDE_ACCESSION}")
    doi = sorted({d.rstrip(".,;") for d in re.findall(r"10\.\d{4,9}/[^\s)]+", readme)})
    checks.add("README: DOIs written out", doi, doi == sorted([lib.ZENODO_DOI, lib.PRIDE_DOI]),
               "the Zenodo record and the PRIDE dataset")
    checks.add("README: licence and code repository as in the plan",
               f"{lib.LICENCE in readme} / {lib.CODE_REPOSITORY in readme}",
               lib.LICENCE in readme and lib.CODE_REPOSITORY in readme, "both present")
    for spec in lib.SPECS.values():
        row = f"| `{spec.h5ad}` | {spec.n_cells:,} | {spec.n_proteins:,} |"
        checks.add(f"README: table row of {spec.h5ad} gives the numbers of the object", row in readme, row in readme,
                   "present")
    for key, order in (("kidney", lib.KIDNEY_ORDER), ("control", lib.DCT_ORDER), ("zerok", lib.DCT_ORDER)):
        have = counts(objects[key].obs["cell_type"])
        text = "; ".join(f"{label} {have[label]}" for label in order if label in have)
        checks.add(f"README: cell_type counts of {key} equal the object", text in readme, text in readme, "present")
    listed = all(f"`{name}`" in readme for name in lib.deposit_files())
    checks.add("README: file table names every file of the deposit", listed, listed, "all")

    dictionary = pd.read_csv(D / "data_dictionary.csv", dtype=str, keep_default_na=False)
    checks.add("dictionary: columns", list(dictionary.columns), list(dictionary.columns) == list(lib.DICTIONARY_COLUMNS),
               "file, location, field, definition, source")
    empty = int((dictionary.apply(lambda c: c.str.strip() == "")).sum().sum())
    checks.add("dictionary: empty cells", empty, empty == 0, "0")
    file_rows = dictionary[dictionary["location"] == "file"]
    checks.add("dictionary: one row per file of the deposit", len(file_rows),
               sorted(file_rows["field"]) == sorted(lib.deposit_files()), "19")
    missing = []
    fields_of = {}
    for spec in lib.SPECS.values():
        a = objects[spec.key]
        fields_of[spec.h5ad] = {"X", "cell_id", "protein", *lib.layer_names(a), *a.obsm.keys(), *a.obsp.keys(), *a.uns.keys(),
                                *a.obs.columns, *a.var.columns}
        fields_of[spec.cell_csv] = set(pd.read_csv(D / spec.cell_csv, nrows=0).columns)
        fields_of[spec.protein_csv] = set(pd.read_csv(D / spec.protein_csv, nrows=0).columns)
        fields_of[spec.matrix_csv] = {*lib.MATRIX_META, "one column per cell_id"}
    for file_name, fields in fields_of.items():
        missing += [f"{file_name}:{f}" for f in sorted(map(str, fields)) if not lib.dictionary_covers(dictionary, file_name, f)]
    checks.add(f"dictionary: every field of every file is defined ({sum(len(v) for v in fields_of.values())} fields)",
               missing or "all", not missing, "all")
    stale = []
    for row in dictionary[dictionary["location"] != "file"].itertuples(index=False):
        patterns = [p.strip() for p in row.file.split(";")]
        targets = [f for f in fields_of if any(fnmatch.fnmatchcase(f, p) for p in patterns)]
        if not targets or not all(row.field in fields_of[f] for f in targets):
            stale.append(f"{row.file}:{row.field}")
    checks.add("dictionary: every row names a field that exists in every file it lists", stale or "all", not stale, "all")
    definitions = dictionary.drop_duplicates(["file", "field"])
    checks.add("dictionary: rows / distinct (file, field) pairs", f"{len(dictionary)} / {len(definitions)}",
               len(dictionary) == len(definitions), "equal")


# ---------------------------------------------------------------------------
# 7. Statements of the README: graph, clusters, transferred label
# ---------------------------------------------------------------------------
def check_statements(checks: t5.Checks, objects: dict) -> None:
    checks.add("scanpy random-number adapter set to 64-bit integers for this process", SCANPY_COMPAT_PATCHED, None)
    for key, resolution in LEIDEN.items():
        a = objects[key]
        pca = np.asarray(a.obsm["X_pca"], dtype=np.float64)
        dist, idx = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(pca).kneighbors(pca)
        stored = sparse.csr_matrix(a.obsp["distances"])
        rows_same, worst = 0, 0.0
        for i in range(a.n_obs):
            row = stored.getrow(i)
            have = dict(zip(row.indices.tolist(), row.data.tolist()))
            mine = {int(j): float(d) for j, d in zip(idx[i, 1:], dist[i, 1:])}
            if set(have) == set(mine):
                rows_same += 1
                worst = max(worst, max(abs(have[j] - mine[j]) for j in have))
        checks.add(f"{key}: stored graph = 9 nearest other cells in X_pca (cells with the same neighbours; max distance "
                   f"diff)", f"{rows_same}/{a.n_obs}; {worst:.2e}", rows_same == a.n_obs and worst < 1e-4, "all; < 1e-4")
        params = dict(a.uns["neighbors"]["params"])
        ok = int(params.get("n_neighbors", -1)) == 10 and int(params.get("n_pcs", -1)) == 20
        checks.add(f"{key}: uns['neighbors'] settings", {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()},
                   ok, "n_neighbors 10, n_pcs 20")
        trial = a.copy()
        sc.tl.leiden(trial, key_added="leiden_check", resolution=resolution, flavor="igraph", random_state=42)
        new = trial.obs["leiden_check"].astype(str).to_numpy()
        same = int((new == a.obs["cluster"].astype(str).to_numpy()).sum())
        checks.add(f"{key}: Leiden (resolution {resolution:g}, igraph, seed 42) on the stored graph returns the cluster "
                   f"column", f"{same}/{a.n_obs} cells; {len(set(new))} clusters", same == a.n_obs, "every cell")

    both, zerok = objects["both"], objects["zerok"]
    pca = np.asarray(both.obsm["X_pca"], dtype=np.float64)
    is_zerok = (both.obs["condition"] == "Zero-K").to_numpy()
    control_labels = both.obs["cell_type"].astype(str).to_numpy()[~is_zerok]
    dist, idx = NearestNeighbors(n_neighbors=10).fit(pca[~is_zerok]).kneighbors(pca[is_zerok])
    labels, fractions = [], []
    for row, d in zip(control_labels[idx], dist):
        tally = pd.Series(row).value_counts()
        winners = [label for label, count in tally.items() if count == tally.max()]
        labels.append(winners[0] if len(winners) == 1 else min(winners, key=lambda label: d[np.argmax(row == label)]))
        fractions.append(tally.max() / 10)
    recomputed = pd.DataFrame({"label": labels, "fraction": fractions}, index=both.obs_names[is_zerok]).loc[zerok.obs_names]
    same = int((recomputed["label"].to_numpy() == zerok.obs["cell_type_transferred"].astype(str).to_numpy()).sum())
    close = bool(np.allclose(recomputed["fraction"].to_numpy(), zerok.obs["cell_type_transferred_vote_fraction"].to_numpy()))
    checks.add("zerok: cell_type_transferred recomputed from the object of both diets (10 nearest Control cells in X_pca)",
               f"{same}/{zerok.n_obs} labels; vote fractions equal {close}", same == zerok.n_obs and close, "every cell")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, STEP, args)
    checks = t5.Checks("deposit")
    if not D.is_dir():
        sys.exit(f"{cc.rel(D)} not found; run deposit_build.py first.")

    check_files(checks, run)
    objects = {key: ad.read_h5ad(D / spec.h5ad) for key, spec in lib.SPECS.items()}
    for key, spec in lib.SPECS.items():
        check_structure(checks, spec, objects[key])
        check_labels(checks, spec, objects[key])
        check_matrices(checks, spec, objects[key])

    raws = {"kidney_raw": Raw(lib.use(run, "kidney_raw")), "dct_raw": Raw(lib.use(run, "dct_raw"))}
    kept_groups: dict = {}
    for key, spec in lib.SPECS.items():          # the single diets come before the object of both diets
        check_raw(checks, spec, objects[key], raws[spec.raw], kept_groups)
    del raws

    labels = pd.read_csv(lib.use(run, "dct_labels"), dtype={"cell_id": str}).set_index("cell_id")
    test = pd.read_csv(lib.use(run, "dct_prolif"), dtype={"cell_id": str}).set_index("cell_id")
    overlay = pd.read_csv(lib.use(run, "dct_prolif_fig"), dtype={"cell_id": str})
    qc = pd.read_csv(lib.use(run, "dct_qc"), dtype={"cell_id": str}).set_index("cell_id")
    check_kidney_reference(checks, run, objects["kidney"])
    for key in ("control", "zerok"):
        check_diet_reference(checks, run, key, objects[key], labels, test, overlay, qc)
    check_both_reference(checks, run, objects["both"], objects, qc)

    for key, spec in lib.SPECS.items():
        check_csv(checks, spec, objects[key])
    check_not_exported(checks)
    words = check_words(checks)
    check_documentation(checks, objects)
    check_statements(checks, objects)

    frame = checks.frame()
    run.table(frame, "validation_checks")
    run.table(words, "word_scan")
    n_checks = int((frame["kind"] == "check").sum())
    n_failed = int((~frame["pass"].astype(bool)).sum())
    print(f"Validation: {n_checks} checks, {n_failed} failed, {int((frame['kind'] == 'info').sum())} information rows")
    run.finish({"n_checks": n_checks, "n_failed": n_failed, "scanpy_compat_patched": bool(SCANPY_COMPAT_PATCHED),
                "tolerances": {"X_vs_zscore": TOL_X, "umap_vs_csv_tables": TOL_CSV_UMAP},
                "leiden": {"flavor": "igraph", "random_state": 42, "resolution": LEIDEN}},
               status="completed" if n_failed == 0 else "failed: validation checks")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
