"""deposit shared definitions: the Zenodo deposit's public names, texts, guarded writer and word scan.

Imported by ``deposit_build.py``,
``deposit_validate.py`` and ``deposit_readme.py``; it runs nothing by itself.

Plan: ``the analysis plan`` (scope approved by the authors in the response to the reviewers Nov 2026;
plan approved "OK, run it", 13 Nov 2026).

Contents
--------
- Inputs (read-only) and the four objects of the deposit with their public file names and exact field lists.
- Public names and label values (kidney DCT and CNT/PC; DCT data "DCT, subtype not resolved"; cluster evidence in
  the wording of the supplementary cluster tables).
- The deposit folder ``results/h5ad/zenodo/``: the one place where the files carry no ``deposit_`` prefix
  (authors, 27 Oct and 13 Nov 2026). ``deposit_path`` writes nowhere else and refuses to replace a file without
  ``--overwrite``.
- Texts of the deposit: data dictionary, README, ``uns['about']``. They carry no task code or internal name.
- Word scan of deposited files (plan step 4). Spectronaut run names (the cell IDs) are removed before the scan,
  because they are the raw file names; text that comes from Spectronaut / UniProt (protein names, gene symbols,
  accessions, descriptions) is scanned and reported separately.
"""

from __future__ import annotations

import fnmatch
import gzip
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc

TASK = "deposit"
DEPOSIT_DIR = cc.OUT_ROOTS["h5ad"] / "zenodo"
_T = cc.OUT_ROOTS["tables"]
_H = cc.OUT_ROOTS["h5ad"]

# ---------------------------------------------------------------------------
# Inputs (read-only)
# ---------------------------------------------------------------------------
SOURCES = {
    "kidney_h5ad": (_H / "kidney_object_KIDNEY_submitted.h5ad",
                    "kidney_object kidney object (matrix, layers, var, stored X_umap, X_pca, graph)"),
    "kidney_labels": (_T / "kidney_annotation" / "kidney_annotation_cluster_labels_per_cell_a_endo.csv",
                      "kidney_annotation kidney labels per cell (kidney_t15_label, t15_cluster, kidney_t15_confidence)"),
    "kidney_qc": (_T / "qc_mito" / "qc_mito_kidney_qc_per_cell.csv", "qc_mito kidney per-cell QC (MitoCarta fraction)"),
    "kidney_raw": (cc.KIDNEY_DIR / "KIDNEY_768cells.csv", "raw Spectronaut export, total kidney (round-trip check only)"),
    "ctrl_h5ad": (_H / "dct_intake_DCT_ctrl.h5ad", "dct_intake object, Ctrl (layers, var)"),
    "zerok_h5ad": (_H / "dct_intake_DCT_zerok.h5ad", "dct_intake object, ZeroK (layers, var)"),
    "joint_h5ad": (_H / "dct_intake_DCT_joint.h5ad", "dct_intake joint object (matrix, layers, var, 20 PCs of variant none)"),
    "ctrl_fig_h5ad": (_H / "dct_labels_DCT_ctrl_labelled.h5ad",
                      "dct_labels labelled object, Ctrl (scaled X, X_umap, graph PCA, graph)"),
    "zerok_fig_h5ad": (_H / "dct_labels_DCT_zerok_labelled.h5ad",
                       "dct_labels labelled object, ZeroK (scaled X, X_umap, graph PCA, graph)"),
    "dct_labels": (_T / "dct_labels" / "dct_labels_dct_canonical_labels.csv", "dct_labels canonical DCT labels"),
    "dct_prolif": (_T / "dct_prolif" / "dct_prolif_dct_labels.csv",
                   "dct_prolif labels (cs_t31nf_*: per-cell proliferation test, no-floor run)"),
    "dct_prolif_fig": (_T / "dct_prolif" / "dct_prolif_prolif_overlay_per_cell.csv",
                       "dct_prolif per-cell table of the Figure 3A overlay (check only)"),
    "dct_qc": (_T / "dct_reviewer" / "dct_reviewer_mito_qc_per_cell.csv", "dct_reviewer DCT per-cell QC (MitoCarta fraction)"),
    "dct_raw": (cc.DCT_DIR / "DCT_Ctrl_ZeroK_combine_1535cells.csv",
                "raw Spectronaut export, DCT combined search (round-trip check only)"),
    "dct_search": (_T / "dct_intake" / "dct_intake_search_settings.csv", "dct_intake search settings (deposit README)"),
    "dct_norm": (_T / "dct_intake" / "dct_intake_normalization_variants.csv",
                 "dct_intake normalization variants (constants of the two layers; data dictionary)"),
}


def src(key: str) -> Path:
    return SOURCES[key][0]


def use(run: cc.Run, key: str) -> Path:
    """Record an input in the step manifest and return its path."""
    return run.use_input(SOURCES[key][0], SOURCES[key][1])


# ---------------------------------------------------------------------------
# Public names and values
# ---------------------------------------------------------------------------
NOT_RESOLVED = "DCT, subtype not resolved"
PROLIF = "ProLIF"
KIDNEY_DISPLAY = {"PCT": "PCT", "PST": "PST", "TAL": "TAL", "DCT1": "DCT", "PC": "CNT/PC", "JG": "JG",
                  "Endothelial": "Endothelial", "IC-B": "IC-B", "Unknown": "Unknown"}
KIDNEY_ORDER = ("Endothelial", "JG", "PCT", "PST", "TAL", "DCT", "CNT/PC", "IC-B", "Unknown")   # figure order
DCT_DISPLAY = {"DCT1": "DCT1", "DCT2": "DCT2", "Unknown": NOT_RESOLVED}
DCT_ORDER = ("DCT1", "DCT2", PROLIF, NOT_RESOLVED)
CONDITION = {"Ctrl": "Control", "ZeroK": "Zero-K"}
CONDITION_ORDER = ("Control", "Zero-K")
PLATE_ORDER = ("P1", "P2")
# Wording of the supplementary cluster tables (the response to the reviewers, S10 / S13).
EVIDENCE_KIDNEY = {"high": "high", "medium": "medium", "specific_any_rank": "type-specific marker, any rank",
                   "low_shared_only": "shared markers only", "unknown_no_marker_in_top_de": "no marker support"}
EVIDENCE_DCT = {**EVIDENCE_KIDNEY, "unknown_no_marker_in_top_de": "no marker support (subtype not resolved)"}
EVIDENCE_ORDER = ("high", "medium", "type-specific marker, any rank", "shared markers only", "no marker support",
                  "no marker support (subtype not resolved)")
LAYER_CELL_MEDIAN = "ln_intensity_cell_median_centred"
LAYER_ALIGNED = "ln1p_intensity_condition_aligned"

KIDNEY_OBS = ("cell_type", "cluster", "cluster_evidence", "n_proteins", "total_intensity", "mitocarta_intensity_fraction")
_DCT_HEAD = ("condition", "sort_plate", "well", "cell_type", "cluster", "cluster_evidence", "prolif", "cycling",
             "proliferation_score", "proliferation_q", "cell_type_prolif")
_DCT_TAIL = ("n_proteins", "total_intensity", "mitocarta_intensity_fraction")
DCT_OBS = _DCT_HEAD + _DCT_TAIL
ZEROK_OBS = _DCT_HEAD + ("cell_type_transferred", "cell_type_transferred_vote_fraction") + _DCT_TAIL
VAR = ("protein_group", "gene_symbol", "description", "n_cells_detected")
BOTH_VAR = VAR + ("n_cells_detected_control", "n_cells_detected_zeroK")
LAYERS = ("intensity", "ln1p_intensity")
BOTH_LAYERS = LAYERS + (LAYER_CELL_MEDIAN, LAYER_ALIGNED)


@dataclass(frozen=True)
class Spec:
    key: str
    stem: str
    title: str
    n_cells: int
    n_proteins: int
    obs: tuple
    var: tuple
    layers: tuple
    obsm: tuple
    obsp: tuple
    uns: tuple
    raw: str                     # SOURCES key of the raw export
    conditions: tuple            # internal condition codes of the raw run names; () for kidney
    n_raw_cells: int
    n_raw_groups: int

    @property
    def h5ad(self) -> str:
        return f"{self.stem}.h5ad"

    @property
    def matrix_csv(self) -> str:
        return f"{self.stem}_intensity_matrix.csv.gz"

    @property
    def cell_csv(self) -> str:
        return f"{self.stem}_cell_metadata.csv"

    @property
    def protein_csv(self) -> str:
        return f"{self.stem}_protein_metadata.csv"

    @property
    def files(self) -> tuple:
        return (self.h5ad, self.matrix_csv, self.cell_csv, self.protein_csv)

    @property
    def has_umap(self) -> bool:
        return "X_umap" in self.obsm


_GRAPH = dict(obsm=("X_pca", "X_umap"), obsp=("connectivities", "distances"), uns=("about", "neighbors"))
SPECS = {
    "kidney": Spec("kidney", "SCP_kidney_total_712cells", "total kidney", 712, 4054, KIDNEY_OBS, VAR, LAYERS,
                   raw="kidney_raw", conditions=(), n_raw_cells=768, n_raw_groups=4144, **_GRAPH),
    "control": Spec("control", "SCP_DCT_control_729cells", "DCT cells, control diet", 729, 6078, DCT_OBS, VAR, LAYERS,
                    raw="dct_raw", conditions=("Ctrl",), n_raw_cells=768, n_raw_groups=6208, **_GRAPH),
    "zerok": Spec("zerok", "SCP_DCT_zeroK_702cells", "DCT cells, zero-potassium diet", 702, 5953, ZEROK_OBS, VAR,
                  LAYERS, raw="dct_raw", conditions=("ZeroK",), n_raw_cells=767, n_raw_groups=6208, **_GRAPH),
    "both": Spec("both", "SCP_DCT_control_zeroK_1431cells", "DCT cells, both diets", 1431, 5897, DCT_OBS, BOTH_VAR,
                 BOTH_LAYERS, obsm=("X_pca",), obsp=(), uns=("about",), raw="dct_raw", conditions=("Ctrl", "ZeroK"),
                 n_raw_cells=1535, n_raw_groups=6208),
}
DOC_FILES = ("data_dictionary.csv", "README.md", "SHA256SUMS.txt")


def layer_names(adata) -> list[str]:
    """Stored layers of an object. anndata 0.13 also lists ``None`` in ``layers``: its view of ``X``, not a layer."""
    return sorted(k for k in adata.layers.keys() if k is not None)


def deposit_files() -> list[str]:
    """The 19 file names of the deposit."""
    return [name for spec in SPECS.values() for name in spec.files] + list(DOC_FILES)


# Counts of the project notes, in public values (plan step 5).
NR = NOT_RESOLVED
EXPECTED = {
    "kidney": {"cell_type": {"PCT": 160, "PST": 158, "TAL": 138, "DCT": 58, "CNT/PC": 38, "JG": 24, "Endothelial": 19,
                             "IC-B": 8, "Unknown": 109}},
    "control": {"cell_type": {"DCT1": 382, "DCT2": 85, NR: 262},
                "cluster_evidence": {"type-specific marker, any rank": 362, "no marker support (subtype not resolved)": 262,
                                     "shared markers only": 95, "high": 10},
                "cell_type_prolif": {"DCT1": 362, "DCT2": 79, PROLIF: 37, NR: 251}, "n_prolif": 37, "n_clusters": 13},
    "zerok": {"cell_type": {"DCT1": 449, NR: 253},
              "cluster_evidence": {"type-specific marker, any rank": 354, "no marker support (subtype not resolved)": 253,
                                   "shared markers only": 95},
              "cell_type_prolif": {"DCT1": 402, PROLIF: 63, NR: 237},
              "cell_type_transferred": {"DCT1": 398, "DCT2": 68, NR: 236}, "n_prolif": 63, "n_clusters": 13},
    "both": {"cell_type": {"DCT1": 831, "DCT2": 85, NR: 515},
             "cell_type_prolif": {"DCT1": 764, "DCT2": 79, PROLIF: 100, NR: 488}, "n_prolif": 100, "n_clusters": 26},
}
EXPECTED["kidney"]["n_clusters"] = 23

# Public name -> project column (project records only; never written into the deposit).
MAPPING = [
    # object, part, public name, project source, project column, value mapping
    ("kidney", "obs index", "cell_id", "kidney_object_KIDNEY_submitted.h5ad", "obs index (cell_id)", ""),
    ("kidney", "obs", "cell_type", "kidney_annotation_cluster_labels_per_cell_a_endo.csv", "kidney_t15_label", "DCT1 -> DCT; PC -> CNT/PC"),
    ("kidney", "obs", "cluster", "kidney_annotation_cluster_labels_per_cell_a_endo.csv", "t15_cluster", ""),
    ("kidney", "obs", "cluster_evidence", "kidney_annotation_cluster_labels_per_cell_a_endo.csv", "kidney_t15_confidence", "evidence wording"),
    ("kidney", "obs", "n_proteins", "kidney_object_KIDNEY_submitted.h5ad", "obs n_detected_proteins", ""),
    ("kidney", "obs", "total_intensity", "kidney_object_KIDNEY_submitted.h5ad", "obs total_intensity", ""),
    ("kidney", "obs", "mitocarta_intensity_fraction", "qc_mito_kidney_qc_per_cell.csv", "mitocarta_intensity_fraction", ""),
    ("kidney", "X, layers", "X; intensity; ln1p_intensity", "kidney_object_KIDNEY_submitted.h5ad", "X; layers intensity, ln1p_intensity", ""),
    ("kidney", "obsm", "X_pca", "kidney_object_KIDNEY_submitted.h5ad", "obsm X_pca, first 20 of 50 components", ""),
    ("kidney", "obsm, obsp, uns", "X_umap; distances; connectivities; neighbors", "kidney_object_KIDNEY_submitted.h5ad", "same keys", ""),
    ("control / zerok / both", "obs index", "cell_id", "dct_intake_DCT_{ctrl,zerok,joint}.h5ad", "obs index (cell_id)", ""),
    ("control / zerok / both", "obs", "condition", "dct_intake_DCT_*.h5ad", "obs condition", "Ctrl -> Control; ZeroK -> Zero-K"),
    ("control / zerok / both", "obs", "sort_plate", "dct_intake_DCT_*.h5ad", "obs plate", ""),
    ("control / zerok / both", "obs", "well", "dct_intake_DCT_*.h5ad", "obs well", "four-digit text -> integer"),
    ("control", "obs", "cell_type", "dct_labels_dct_canonical_labels.csv", "cs_ctrl_deg_label", "Unknown -> DCT, subtype not resolved"),
    ("control", "obs", "cluster", "dct_labels_dct_canonical_labels.csv", "cs_ctrl_deg_cluster", ""),
    ("control", "obs", "cluster_evidence", "dct_labels_dct_canonical_labels.csv", "cs_ctrl_deg_confidence", "evidence wording"),
    ("zerok", "obs", "cell_type", "dct_labels_dct_canonical_labels.csv", "cs_zerok_pipeline_label", "Unknown -> DCT, subtype not resolved"),
    ("zerok", "obs", "cluster", "dct_labels_dct_canonical_labels.csv", "cs_zerok_pipeline_cluster", ""),
    ("zerok", "obs", "cluster_evidence", "dct_labels_dct_canonical_labels.csv", "cs_zerok_pipeline_confidence", "evidence wording"),
    ("zerok", "obs", "cell_type_transferred", "dct_labels_dct_canonical_labels.csv", "cs_zerok_transfer_label", "Unknown -> DCT, subtype not resolved"),
    ("zerok", "obs", "cell_type_transferred_vote_fraction", "dct_labels_dct_canonical_labels.csv", "cs_zerok_transfer_confidence", ""),
    ("both", "obs", "cell_type; cluster_evidence", "dct_labels_dct_canonical_labels.csv",
     "cs_ctrl_deg_* for Ctrl cells, cs_zerok_pipeline_* for ZeroK cells (= dct_label, dct_label_confidence)", "as above"),
    ("both", "obs", "cluster", "dct_labels_dct_canonical_labels.csv", "cs_ctrl_deg_cluster / cs_zerok_pipeline_cluster",
     "'Control_<id>' / 'Zero-K_<id>'"),
    ("control / zerok / both", "obs", "prolif", "dct_prolif_dct_labels.csv", "cs_t31nf_label == 'ProLIF'", "bool"),
    ("control / zerok / both", "obs", "cycling", "dct_prolif_dct_labels.csv", "cs_t31nf_cycling == 'Cycling'", "bool"),
    ("control / zerok / both", "obs", "proliferation_score", "dct_prolif_dct_labels.csv", "cs_t31nf_prolif_D", ""),
    ("control / zerok / both", "obs", "proliferation_q", "dct_prolif_dct_labels.csv", "cs_t31nf_prolif_q", ""),
    ("control / zerok / both", "obs", "cell_type_prolif", "derived", "cell_type with prolif cells set to ProLIF "
     "(= overlay_group of dct_prolif_prolif_overlay_per_cell.csv, run nofloor)", ""),
    ("control / zerok", "obs", "n_proteins; total_intensity", "dct_intake_DCT_{ctrl,zerok}.h5ad", "obs n_detected; total_intensity", ""),
    ("both", "obs", "n_proteins; total_intensity", "dct_intake_DCT_joint.h5ad", "obs n_detected; total_intensity (joint proteins)", ""),
    ("control / zerok / both", "obs", "mitocarta_intensity_fraction", "dct_reviewer_mito_qc_per_cell.csv", "mitocarta_intensity_fraction", ""),
    ("control / zerok", "X", "X", "dct_labels_DCT_{ctrl,zerok}_labelled.h5ad", "X (the matrix behind the PCA and UMAP)", ""),
    ("control / zerok", "layers", "intensity; ln1p_intensity", "dct_intake_DCT_{ctrl,zerok}.h5ad", "layers intensity, ln1p_intensity", ""),
    ("control / zerok", "obsm", "X_pca", "dct_labels_DCT_{ctrl,zerok}_labelled.h5ad", "obsm X_pca_neighbors (20 components)", ""),
    ("control / zerok", "obsm, obsp, uns", "X_umap; distances; connectivities; neighbors", "dct_labels_DCT_{ctrl,zerok}_labelled.h5ad", "same keys", ""),
    ("both", "X, layers", "X; intensity; ln1p_intensity", "dct_intake_DCT_joint.h5ad", "X; layers intensity, ln1p_intensity", ""),
    ("both", "layers", LAYER_CELL_MEDIAN, "dct_intake_DCT_joint.h5ad", "layers norm__cell_median", ""),
    ("both", "layers", LAYER_ALIGNED, "dct_intake_DCT_joint.h5ad", "layers norm__condition_align", ""),
    ("both", "obsm", "X_pca", "dct_intake_DCT_joint.h5ad", "obsm X_pca__none", ""),
    ("all", "var index", "protein", "source object", "var index (protein)", ""),
    ("all", "var", "protein_group", "source object", "var protein_group", ""),
    ("all", "var", "gene_symbol", "source object", "var pg_genes", "missing -> empty text"),
    ("all", "var", "description", "source object", "var protein_description", ""),
    ("all", "var", "n_cells_detected", "source object", "var n_cells_detected", ""),
    ("both", "var", "n_cells_detected_control; n_cells_detected_zeroK", "dct_intake_DCT_joint.h5ad", "var n_cells_ctrl; n_cells_zerok", ""),
]

# Not exported (plan step 1): exact names and substrings that must not occur among the fields of the deposit.
NOT_EXPORTED_NAMES = (
    "kidney_pre_label", "kidney_reannot_label", "kidney_pre_label_full", "kidney_reannot_label_full", "leiden_res_2.00",
    "leiden_res_3.00", "leiden_res_3.00_named", "predicted_cell_type", "cell_type_v2", "cell_type_pre", "plate",
    "DCT1_score", "DCT2_score", "ProLIF_score", "max_score", "dpt_pseudotime", "dct_label", "dct_label_alternative",
    "t31_label", "X_diffmap", "X_pca_neighbors", "paga", "iroot", "diffmap_evals", "None", "file_column_index",
    "cell_key", "v1_cell_label", "v1_cluster_label", "ctrl_deg_label", "zerok_pipeline_label", "zerok_transfer_label",
)
NOT_EXPORTED_SUBSTRINGS = ("cs_", "t31", "t15", "leiden", "pseudotime", "paga", "aldo", "cell_cycle", "panel",
                           "seed_stability", "_colors", "__", "v1_", "reannot", "_pre_", "deg_", "pipeline_")
ALLOWED_SCORE_FIELDS = ("proliferation_score",)     # the only field with 'score' in its name


# ---------------------------------------------------------------------------
# Deposit folder
# ---------------------------------------------------------------------------
def deposit_path(name: str, overwrite: bool) -> Path:
    """Path of a deposit file; refuses other folders, unknown names and existing files without --overwrite."""
    if name not in deposit_files():
        raise PermissionError(f"{name!r} is not a file of the deposit")
    path = (DEPOSIT_DIR / name).resolve()
    if path.parent != DEPOSIT_DIR.resolve():
        raise PermissionError(f"Refusing to write outside {cc.rel(DEPOSIT_DIR)}: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {cc.rel(path)} (rerun with --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: Path, text: str) -> None:
    """UTF-8 text with LF line ends (the same bytes on every platform)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def write_csv_gz(frame: pd.DataFrame, path: Path, float_format: str = "%.9g") -> None:
    """Gzip CSV without a time stamp or file name in the gzip header (reproducible bytes)."""
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, float_format=float_format, lineterminator="\n")


def float32_text_values(values) -> list[float]:
    """float32 values as the shortest decimal that reads back to the same float32."""
    return [float(str(v)) for v in np.asarray(values, dtype=np.float32)]


def cell_table(adata, spec: Spec) -> pd.DataFrame:
    """cell_id, the obs columns and, where the object has one, the UMAP coordinates."""
    table = adata.obs.copy()
    table.insert(0, "cell_id", adata.obs_names.astype(str))
    if spec.has_umap:
        umap = np.asarray(adata.obsm["X_umap"])
        table["UMAP1"] = float32_text_values(umap[:, 0])
        table["UMAP2"] = float32_text_values(umap[:, 1])
    return table.reset_index(drop=True)


def protein_table(adata) -> pd.DataFrame:
    table = adata.var.copy()
    table.insert(0, "protein", adata.var_names.astype(str))
    return table.reset_index(drop=True)


MATRIX_META = ("protein", "gene_symbol", "protein_group", "description")


def matrix_table(adata) -> pd.DataFrame:
    """Protein groups x cells; not detected = empty."""
    intensity = cc.dense(adata.layers["intensity"])
    wide = pd.DataFrame(np.where(intensity > 0, intensity, np.nan).T, columns=adata.obs_names.astype(str))
    meta = protein_table(adata)[list(MATRIX_META)]
    return pd.concat([meta, wide], axis=1)


# ---------------------------------------------------------------------------
# Word scan (plan step 4)
# ---------------------------------------------------------------------------
RUN_NAME_RE = re.compile(r"\[\d+\] (?:Ctrl|ZeroK)_GFP_1cell_WISHDIA_P\d+_\d{4}\.htrms\.PG\.Quantity"
                         r"|\[\d+\] Kidney_1cell_\d+\.htrms\.PG\.Quantity")
FORBIDDEN = (
    ("task code kidney_object-dct_prolif", re.compile(r"(?<![A-Za-z0-9])T(?:0[1-9]|[12][0-9]|3[01])(?![0-9])", re.I)),
    ("cs_", re.compile(r"(?<![A-Za-z0-9])cs_")),
    ("a_endo", re.compile(r"a_endo", re.I)),
    ("Ctrl", re.compile(r"Ctrl")),
    ("ZeroK", re.compile(r"ZeroK")),
    ("submitted", re.compile(r"submitted", re.I)),
    ("revision", re.compile(r"revision", re.I)),
    ("revised", re.compile(r"revised", re.I)),
    ("re-annotation", re.compile(r"re-?annot", re.I)),
    # further internal names, beyond the plan's list
    ("nofloor", re.compile(r"no-?floor", re.I)),
    ("internal label column", re.compile(r"kidney_pre|deg_label|deg_cluster|pipeline_label|pipeline_cluster|leiden3|"
                                         r"leiden_res|predicted_cell_type|dct_label", re.I)),
)
SOURCE_TEXT_FIELDS = ("protein", "protein_group", "gene_symbol", "description")   # Spectronaut / UniProt text


def scan_text(text: str) -> list[str]:
    """Labels of the forbidden patterns found in a text; Spectronaut run names are removed first."""
    clean = RUN_NAME_RE.sub("", str(text))
    return [label for label, pattern in FORBIDDEN if pattern.search(clean)]


def _hit(file: str, where: str, kind: str, text: str) -> list[dict]:
    return [{"file": file, "where": where, "kind": kind, "pattern": label, "text": str(text)[:120]}
            for label in scan_text(text)]


def scan_h5ad(path: Path) -> tuple[list[dict], int]:
    """Scan every group, dataset and attribute name and every text value of an h5ad file (read with h5py)."""
    import h5py

    hits: list[dict] = []
    n_texts = 0

    def is_source(name: str) -> bool:
        parts = name.split("/")
        return len(parts) >= 2 and parts[0] == "var" and parts[1] in SOURCE_TEXT_FIELDS + ("_index",)

    def texts_of(value) -> list[str]:
        if isinstance(value, bytes):
            return [value.decode("utf-8", "replace")]
        if isinstance(value, str):
            return [value]
        if isinstance(value, np.ndarray) and value.dtype.kind in "OSU":
            flat = value.ravel()
            return [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in pd.unique(flat)]
        return []

    def visit(name: str, obj) -> None:
        nonlocal n_texts
        kind = "source text" if is_source(name) else "own text"
        n_texts += 1
        # a var index or var column name is ours; only the values below it come from Spectronaut / UniProt
        hits.extend(_hit(path.name, f"key {name}", "own text", name))
        for attr, value in obj.attrs.items():
            hits.extend(_hit(path.name, f"attribute {name}@{attr}", "own text", attr))
            for text in texts_of(value):
                n_texts += 1
                hits.extend(_hit(path.name, f"attribute {name}@{attr}", "own text", text))
        if isinstance(obj, h5py.Dataset) and (h5py.check_string_dtype(obj.dtype) is not None or obj.dtype.kind in "SU"):
            for text in texts_of(obj[()] if obj.shape == () else obj[...]):
                n_texts += 1
                hits.extend(_hit(path.name, f"values {name}", kind, text))

    with h5py.File(path, "r") as handle:
        handle.visititems(visit)
    return hits, n_texts


def scan_csv(path: Path, usecols=None) -> tuple[list[dict], int]:
    """Scan the column names and every distinct text cell of a CSV (optionally only some columns)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols)
    hits: list[dict] = []
    n_texts = 0
    for column in table.columns:
        n_texts += 1
        hits.extend(_hit(path.name, f"column name {RUN_NAME_RE.sub('<run name>', column)}", "own text", column))
        kind = "source text" if column in SOURCE_TEXT_FIELDS else "own text"
        for text in pd.unique(table[column]):
            n_texts += 1
            hits.extend(_hit(path.name, f"column {RUN_NAME_RE.sub('<run name>', column)}", kind, text))
    return hits, n_texts


def scan_matrix_header(path: Path) -> tuple[list[dict], int]:
    """Column names of a gzip matrix (the text columns are scanned with scan_csv(usecols=MATRIX_META))."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
    return _hit(path.name, "header line", "own text", header), 1


def scan_lines(path: Path) -> tuple[list[dict], int]:
    hits: list[dict] = []
    lines = path.read_text(encoding="utf-8").split("\n")
    for number, line in enumerate(lines, start=1):
        hits.extend(_hit(path.name, f"line {number}", "own text", line))
    return hits, len(lines)


def scan_deposit_file(path: Path) -> tuple[list[dict], int]:
    """Word scan of one deposited file, by its type."""
    name = path.name
    if name.endswith(".h5ad"):
        return scan_h5ad(path)
    if name.endswith("_intensity_matrix.csv.gz"):
        head, n1 = scan_matrix_header(path)
        body, n2 = scan_csv(path, usecols=list(MATRIX_META))
        return head + [h for h in body if not h["where"].startswith("column name")], n1 + n2
    if name.endswith(".csv"):
        return scan_csv(path)
    return scan_lines(path)


def not_exported_hits(names) -> list[str]:
    """Field names that belong to the 'not exported' list of the plan."""
    bad = []
    for name in map(str, names):
        low = name.lower()
        if name in NOT_EXPORTED_NAMES or any(part in low for part in NOT_EXPORTED_SUBSTRINGS):
            bad.append(name)
        elif "score" in low and name not in ALLOWED_SCORE_FIELDS:
            bad.append(name)
    return bad


# ---------------------------------------------------------------------------
# Texts of the deposit
# ---------------------------------------------------------------------------
# Record and accession as given by the authors on 20 Nov 2026. The PRIDE dataset holds all raw data of the manuscript
# and replaces the earlier accession. Title and authors as on the title page of the manuscript.
ZENODO_DOI = "10.5281/zenodo.24365599"
ZENODO_URL = f"https://doi.org/{ZENODO_DOI}"
PRIDE_ACCESSION = "PXD087695"
PRIDE_DOI = "10.6019/PXD087695"
MANUSCRIPT_TITLE = "Single-cell proteomics: a powerful new tool to study kidney cell heterogeneity"
AUTHORS = ("Qi Wu, Lei Cheng, Adrienne M. Assmus, Xiang Zheng, Cristina Esteva Font, Charlotte C. Petersen, "
           "Xiangning Ding, Søren R. Paludan and Robert A. Fenton")
PLACEHOLDERS = ()                                   # none left in the README of the deposit
PLACEHOLDER_RE = re.compile(r"\[[A-Za-z][^\]\n]*\]")  # a bracketed note such as "[author list]"
CODE_REPOSITORY = "https://github.com/qiwu-au/scp_mouse_kidney_dct"      # as named in the manuscript
LICENCE = "CC BY 4.0"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/"
EXAMPLE_RUN_DCT = "[1] Ctrl_GFP_1cell_WISHDIA_P1_0001.htrms.PG.Quantity"
EXAMPLE_RUN_KIDNEY = "[1] Kidney_1cell_001.htrms.PG.Quantity"

S_EXPORT = "Spectronaut export (PG.Quantity)"
S_MATRIX = "processed matrix of the data set"
S_RUN = "Spectronaut run name"
S_ANNOT = "cluster annotation (Leiden clusters; markers of Supplementary Table S2 of the manuscript)"
S_PROLIF = "per-cell proliferation test"
S_TRANSFER = "label transfer from the Control cells"
S_QC = "per-cell quality control (MitoCarta3.0)"
S_EMBED = "principal component analysis, neighbour graph and UMAP of the data set"
S_DOC = "documentation of the record"

F_H5AD = "*.h5ad"
F_GRAPH = "; ".join(SPECS[k].h5ad for k in ("kidney", "control", "zerok"))
F_KIDNEY = SPECS["kidney"].h5ad
F_DCT = "SCP_DCT_*.h5ad"
F_ZEROK = SPECS["zerok"].h5ad
F_BOTH = SPECS["both"].h5ad
F_CELLS = "*_cell_metadata.csv"
F_PROT = "*_protein_metadata.csv"
F_MATRIX = "*_intensity_matrix.csv.gz"


def _cells(pattern: str) -> str:
    """File pattern of a cell field: the object(s) and their cell tables."""
    tables = "; ".join(p.replace(".h5ad", "_cell_metadata.csv") for p in pattern.split("; "))
    return f"{pattern}; {tables}"


def _proteins(pattern: str, matrix: bool = False) -> str:
    out = [pattern, "; ".join(p.replace(".h5ad", "_protein_metadata.csv") for p in pattern.split("; "))]
    if matrix:
        out.append("; ".join(p.replace(".h5ad", "_intensity_matrix.csv.gz") for p in pattern.split("; ")))
    return "; ".join(out)


EVIDENCE_TEXT = (
    "Each cluster was compared with all other cells of the data set (two-sided Wilcoxon rank-sum test on "
    "ln(1 + intensity), proteins detected in at least 20% of the cluster's cells, Benjamini-Hochberg q < 0.05). "
    "'high': a marker listed for one cell type only is among the 25 most increased proteins, and the markers of "
    "that cell type are enriched among them (hypergeometric test, q < 0.05); 'medium': the same without significant "
    "enrichment; 'type-specific marker, any rank': no such marker among the 25 most increased proteins, but a marker "
    "listed for one cell type only is significantly increased at a lower rank; 'shared markers only': only markers "
    "listed for two or more cell types are among the 25 most increased proteins; ")


def dictionary_rows(norm: dict) -> list[tuple[str, str, str, str, str]]:
    """Rows (file, location, field, definition, source) of data_dictionary.csv.

    ``norm`` holds the constants of the two normalization layers (from the processed data): ``shift``,
    ``delta``, ``n_delta``.
    """
    rows: list[tuple[str, str, str, str, str]] = []
    for spec in SPECS.values():
        rows += [
            (spec.h5ad, "file", spec.h5ad, f"AnnData object (h5ad, gzip-compressed) of the data set '{spec.title}': "
             f"{spec.n_cells:,} cells x {spec.n_proteins:,} protein groups", S_MATRIX),
            (spec.matrix_csv, "file", spec.matrix_csv, "intensity matrix as text (gzip-compressed CSV): one row per protein "
             f"group, one column per cell; the values of layers['intensity'] of {spec.h5ad}; empty = not detected", S_MATRIX),
            (spec.cell_csv, "file", spec.cell_csv, f"one row per cell: the obs columns of {spec.h5ad}"
             + (", with the UMAP coordinates" if spec.has_umap else ""), S_MATRIX),
            (spec.protein_csv, "file", spec.protein_csv, f"one row per protein group: the var columns of {spec.h5ad}", S_MATRIX),
        ]
    rows += [
        ("data_dictionary.csv", "file", "data_dictionary.csv", "this table: every file and field of the record", S_DOC),
        ("README.md", "file", "README.md", "description of the record: data sets, experimental design, processing, files, "
         "raw data, code, licence and citation", S_DOC),
        ("SHA256SUMS.txt", "file", "SHA256SUMS.txt", "SHA-256 checksum of every other file of the record, in the format of "
         "the sha256sum tool", S_DOC),
        # ---- matrices
        (F_H5AD, "X", "X", "scaled matrix used for the principal component analysis: per-protein z-score of ln(1 + intensity) "
         "over the cells of the object (not detected = 0 before scaling; standard deviation with n - 1), clipped at +10 and "
         "-10; cells x protein groups, 32-bit floating point", S_MATRIX),
        (F_H5AD, "layers", "intensity", "protein group quantity as exported by Spectronaut (PG.Quantity; cross-run "
         "normalization of the search included); 0 = not detected; no value was imputed", S_EXPORT),
        (F_H5AD, "layers", "ln1p_intensity", "natural logarithm of (1 + intensity); 0 = not detected", S_MATRIX),
        (F_BOTH, "layers", LAYER_CELL_MEDIAN, "normalization variant of the comparison of the two diets: for the detected "
         "values, ln(intensity) minus the median of the cell plus the median of all cell medians; then the constant "
         f"{norm['shift']:.4f} added to every detected value, so that the smallest detected value is 0.01; 0 = not detected",
         S_MATRIX),
        (F_BOTH, "layers", LAYER_ALIGNED, "normalization variant of the comparison of the two diets: ln(1 + intensity), "
         f"with every detected value of a Zero-K cell shifted by {norm['delta']:.4f}; the shift is the median, over the "
         f"{norm['n_delta']:,} protein groups detected in at least 5 cells of each diet, of the difference between the "
         "median detected value of the Control cells and that of the Zero-K cells; Control cells unchanged; "
         "0 = not detected", S_MATRIX),
        # ---- embeddings
        (F_H5AD, "obsm", "X_pca", "20 principal components of X. Total kidney, Control and Zero-K objects: the components "
         "from which the neighbour graph and the UMAP were computed. Object of both diets: the components of all 1,431 "
         "cells, used for the transferred label", S_EMBED),
        (F_GRAPH, "obsm", "X_umap", "UMAP coordinates of the cells, computed from the neighbour graph; the UMAP shown in "
         "the manuscript (the object of both diets has none)", S_EMBED),
        (F_GRAPH, "obsp", "distances", "neighbour graph: Euclidean distance in X_pca from each cell to its 9 nearest other "
         "cells (scanpy, n_neighbors = 10)", S_EMBED),
        (F_GRAPH, "obsp", "connectivities", "neighbour graph: connectivity weights computed by scanpy from the distances "
         "(UMAP method); the graph that was clustered", S_EMBED),
        (F_GRAPH, "uns", "neighbors", "settings of the neighbour graph in scanpy's format (n_neighbors = 10, 20 principal "
         "components, Euclidean metric), so that scanpy functions can use the stored graph", S_EMBED),
        (F_H5AD, "uns", "about", "short description of the object: data set, numbers of cells and protein groups, matrices, "
         "missing values, filters, embedding, annotation, documentation and licence", S_DOC),
        # ---- cells
        (_cells(F_H5AD), "obs index; column", "cell_id", "Spectronaut run name of the cell (one mass spectrometry run per "
         "sorted cell): '[n] <raw file name>.htrms.PG.Quantity', where n is the column position in the Spectronaut export",
         S_RUN),
        (_cells(F_DCT), "obs; column", "condition", "diet of the mice the cell came from: Control or Zero-K (zero-potassium "
         "diet), 4 days. Each diet is one pool of three mice, so diet and pool cannot be separated", S_RUN),
        (_cells(F_DCT), "obs; column", "sort_plate", "384-well plate the cell was sorted into (P1 or P2). The two plates of a "
         "diet hold cells of the same pool: a technical label, not a biological replicate", S_RUN),
        (_cells(F_DCT), "obs; column", "well", "well number on the sort plate, 1 to 384, counted row by row (1 = A1, "
         "24 = A24, 25 = B1); the last four digits of the raw file name", S_RUN),
        (_cells(F_KIDNEY), "obs; column", "cell_type", "cell type assigned to the cell's cluster: PCT (proximal convoluted "
         "tubule), PST (proximal straight tubule), TAL (thick ascending limb), DCT (distal convoluted tubule), CNT/PC "
         "(connecting tubule / principal cells), JG (juxtaglomerular cells), Endothelial, IC-B (type B intercalated cells) "
         "or Unknown (cluster without marker support)", S_ANNOT),
        (_cells(F_DCT), "obs; column", "cell_type", "cell type assigned to the cell's cluster: DCT1, DCT2 or 'DCT, subtype "
         "not resolved' (cluster without marker support for DCT1 or DCT2). The cells of each diet were clustered and "
         "annotated on their own", S_ANNOT),
        (_cells(F_KIDNEY), "obs; column", "cluster", "Leiden cluster of the neighbour graph (scanpy, igraph implementation, "
         "resolution 2, random seed 42): 23 clusters, 0 to 22", S_ANNOT),
        (_cells(F_DCT), "obs; column", "cluster", "Leiden cluster of the neighbour graph of the diet (scanpy, igraph "
         "implementation, resolution 1, random seed 42): 13 clusters per diet, 0 to 12. In the object of both diets the "
         "cluster is preceded by its diet (Control_0 to Control_12, Zero-K_0 to Zero-K_12)", S_ANNOT),
        (_cells(F_KIDNEY), "obs; column", "cluster_evidence", "marker evidence for the cell type of the cell's cluster. "
         + EVIDENCE_TEXT + "'no marker support': none of these (cell type Unknown)", S_ANNOT),
        (_cells(F_DCT), "obs; column", "cluster_evidence", "marker evidence for the cell type of the cell's cluster. "
         + EVIDENCE_TEXT + "'no marker support (subtype not resolved)': none of these. In the DCT data Slc12a3 and Umod "
         "count as shared markers", S_ANNOT),
        (_cells(F_DCT), "obs; column", "prolif", "True for the cells called ProLIF by the per-cell proliferation test: "
         "cycling is True and Slc12a3 (NCC) was detected in the cell", S_PROLIF),
        (_cells(F_DCT), "obs; column", "cycling", "True if proliferation_q is below 0.05", S_PROLIF),
        (_cells(F_DCT), "obs; column", "proliferation_score", "score of the per-cell proliferation test: the mean, over 36 "
         "S-phase and G2/M-phase proteins quantified in both diets (Mki67 and Top2a not among them), of the difference "
         "between detection in the cell (1 or 0) and the detection rate of that protein among the cells of the same diet "
         "with a similar number of proteins (20 equal-sized bins)", S_PROLIF),
        (_cells(F_DCT), "obs; column", "proliferation_q", "Benjamini-Hochberg q value, over the cells of the diet, of the "
         "empirical P value of proliferation_score against 1,000 random sets of 36 proteins matched on detection "
         "frequency", S_PROLIF),
        (_cells(F_DCT), "obs; column", "cell_type_prolif", "cell_type, with the ProLIF cells set to ProLIF: the groups shown "
         "on the DCT UMAP of the manuscript", f"{S_ANNOT}; {S_PROLIF}"),
        (_cells(F_ZEROK), "obs; column", "cell_type_transferred", "second label of the Zero-K cells, transferred from the "
         "Control cells: the most frequent cell_type among the 10 nearest Control cells (Euclidean distance in X_pca of "
         f"{F_BOTH}); with a tie, the label of the nearest of the tied cells", S_TRANSFER),
        (_cells(F_ZEROK), "obs; column", "cell_type_transferred_vote_fraction", "fraction of the 10 nearest Control cells "
         "that carry the transferred label (0.5 to 1)", S_TRANSFER),
        (_cells(F_H5AD), "obs; column", "n_proteins", "number of protein groups of the object with intensity above 0 in the "
         "cell. The cell filter (at least 100) counted all protein groups of the export, so this number can be lower "
         "in the object of both diets, which holds fewer protein groups", S_MATRIX),
        (_cells(F_H5AD), "obs; column", "total_intensity", "sum of the intensities of the cell over the protein groups of "
         "the object", S_MATRIX),
        (_cells(F_H5AD), "obs; column", "mitocarta_intensity_fraction", "share of the cell's total intensity that comes from "
         "MitoCarta3.0 proteins, over the protein groups retained for the data set (in the object of both diets: over the "
         "protein groups retained for the diet of the cell, the value of the Control or Zero-K object)", S_QC),
        ("; ".join(SPECS[k].cell_csv for k in ("kidney", "control", "zerok")), "column", "UMAP1",
         "first UMAP coordinate (X_umap of the object)", S_EMBED),
        ("; ".join(SPECS[k].cell_csv for k in ("kidney", "control", "zerok")), "column", "UMAP2",
         "second UMAP coordinate (X_umap of the object)", S_EMBED),
        # ---- protein groups
        (_proteins(F_H5AD, matrix=True), "var index; column", "protein", "unique name of the protein group in the analysis: "
         "the first gene symbol of the group with the first letter in upper case and the others in lower case; a repeated "
         "name gets a suffix (_1, _2). Where the first gene symbol is missing, the name is 'Nan' (total kidney) or the "
         "first UniProt accession of the group (DCT data)", S_MATRIX),
        (_proteins(F_H5AD, matrix=True), "var; column", "protein_group", "UniProt accessions of the protein group "
         "(PG.ProteinGroups), separated by ';'", S_EXPORT),
        (_proteins(F_H5AD, matrix=True), "var; column", "gene_symbol", "gene symbols of the protein group as exported "
         "(PG.Genes), separated by ';'; empty if the group has none", S_EXPORT),
        (_proteins(F_H5AD, matrix=True), "var; column", "description", "protein descriptions of the group "
         "(PG.ProteinDescriptions)", S_EXPORT),
        (_proteins(F_H5AD), "var; column", "n_cells_detected", "number of cells of the object with intensity above 0 for "
         "the protein group", S_MATRIX),
        (_proteins(F_BOTH), "var; column", "n_cells_detected_control", "number of Control cells of the object with "
         "intensity above 0 for the protein group", S_MATRIX),
        (_proteins(F_BOTH), "var; column", "n_cells_detected_zeroK", "number of Zero-K cells of the object with intensity "
         "above 0 for the protein group", S_MATRIX),
        (F_MATRIX, "columns", "one column per cell_id", "intensity of the protein group in that cell (PG.Quantity); "
         "empty = not detected", S_EXPORT),
    ]
    return rows


DICTIONARY_COLUMNS = ("file", "location", "field", "definition", "source")


def dictionary_covers(dictionary: pd.DataFrame, file_name: str, field: str) -> bool:
    """True if a dictionary row names the field for this file (file patterns separated by ';')."""
    for row in dictionary[dictionary["field"] == field].itertuples(index=False):
        if any(fnmatch.fnmatchcase(file_name, pattern.strip()) for pattern in str(row.file).split(";")):
            return True
    return False


def about_text(spec: Spec, norm: dict) -> dict:
    """The ``uns['about']`` entry of an object (plain text and numbers)."""
    about = {
        "data_set": f"Single-cell proteomics by mass spectrometry (one run per sorted cell), mouse: {spec.title}",
        "n_cells": int(spec.n_cells),
        "n_protein_groups": int(spec.n_proteins),
        "X": "per-protein z-score of ln(1 + intensity) over the cells of this object, clipped at +10 and -10",
        "layers": "intensity = Spectronaut PG.Quantity; ln1p_intensity = ln(1 + intensity)",
        "missing_values": "a protein group not detected in a cell is 0 in the layers; no value was imputed",
        "filters": "cells with at least 100 protein groups detected, counted over all protein groups of the export; "
                   "then protein groups detected in at least 2 retained cells",
        "documentation": "README.md and data_dictionary.csv of the same record",
        "licence": LICENCE,
    }
    if spec.key == "kidney":
        about["embedding"] = "X_pca: 20 principal components of X; neighbour graph: 10 nearest neighbours on X_pca; X_umap"
        about["annotation"] = ("cell_type per Leiden cluster (resolution 2, random seed 42) from the marker proteins "
                               "increased in the cluster")
    elif spec.key in ("control", "zerok"):
        about["embedding"] = "X_pca: 20 principal components of X; neighbour graph: 10 nearest neighbours on X_pca; X_umap"
        about["annotation"] = ("cell_type per Leiden cluster (resolution 1, random seed 42) from the marker proteins "
                               "increased in the cluster; prolif = per-cell proliferation test")
        about["design"] = ("one pool of three mice per diet; sort plates P1 and P2 are technical, not biological, "
                           "replicates")
    else:
        about["protein_groups"] = "the protein groups retained in both diets"
        about["layers"] += (f"; {LAYER_CELL_MEDIAN} and {LAYER_ALIGNED} = normalization variants of the comparison "
                            "of the two diets")
        about["embedding"] = "X_pca: 20 principal components of X over the cells of both diets; no UMAP and no graph"
        about["annotation"] = ("cell_type, cluster and prolif as in the objects of the single diets (each diet "
                               "annotated on its own)")
        about["design"] = ("one pool of three mice per diet, so diet and pool cannot be separated; sort plates P1 and "
                           "P2 are technical, not biological, replicates")
    return about


def deposit_readme(facts: dict) -> str:
    """README.md of the deposit. ``facts`` holds numbers measured during the build."""
    s = SPECS

    def size_text(n_bytes) -> str:
        return "< 0.1" if n_bytes is None or n_bytes < 50_000 else f"{n_bytes / 1e6:.1f}"

    file_rows = "\n".join(f"| `{name}` | {size_text(facts['sizes'].get(name))} |" for name in deposit_files())
    n_runs = facts["n_raw_control"] + facts["n_raw_zerok"]
    software = ", ".join(f"{k} {v}" for k, v in facts["software"].items())
    search = facts["search"]
    return f"""# Single-cell proteomics of mouse kidney and distal convoluted tubule cells: processed data

{AUTHORS}

Processed data of the manuscript "{MANUSCRIPT_TITLE}". Zenodo record: {ZENODO_URL}. Licence: {LICENCE}.

## Content

Four data sets of single-cell proteomics by mass spectrometry (SCP-MS; one mass spectrometry run per sorted cell),
each as an AnnData object (`.h5ad`) and as CSV files.

| data set | object | cells | protein groups |
| --- | --- | --- | --- |
| total kidney | `{s['kidney'].h5ad}` | {s['kidney'].n_cells:,} | {s['kidney'].n_proteins:,} |
| DCT cells, control diet | `{s['control'].h5ad}` | {s['control'].n_cells:,} | {s['control'].n_proteins:,} |
| DCT cells, zero-potassium diet | `{s['zerok'].h5ad}` | {s['zerok'].n_cells:,} | {s['zerok'].n_proteins:,} |
| DCT cells, both diets | `{s['both'].h5ad}` | {s['both'].n_cells:,} | {s['both'].n_proteins:,} |

The fourth object holds the cells of the second and third object on the protein groups retained in both diets. It
is the matrix of the comparison of the two diets.

For each object there are three plain-text files, for readers who do not work in Python:

- `<object>_intensity_matrix.csv.gz`: protein groups in rows, cells in columns; an empty field means not detected;
- `<object>_cell_metadata.csv`: one row per cell (cell type, cluster, quality measures, UMAP coordinates);
- `<object>_protein_metadata.csv`: one row per protein group (accessions, gene symbols, description).

`data_dictionary.csv` defines every file and field. `SHA256SUMS.txt` lists the SHA-256 checksum of every other file.

## Experimental design

- **Total kidney.** Cells from two male C57Bl/6 mice. The cell suspensions of the two mice were pooled before
  sorting, so no cell can be assigned to a mouse. {s['kidney'].n_raw_cells} cells were sorted and measured.
- **DCT cells.** Mice received a control diet or a zero-potassium (Zero-K) diet for 4 days. The cells of the three
  mice of a diet were pooled, giving one pool per diet. Diet and pool therefore cannot be separated, and the cells
  are not animal-level replicates. Each pool was sorted into two 384-well plates (P1, P2). The plates are technical
  sort plates of the pooled cells, not biological replicates. Control and Zero-K cells were sorted and acquired in
  the same sessions. {facts['n_raw_control']} Control and {facts['n_raw_zerok']} Zero-K cells were measured.
- Mouse lines, diets, cell sorting, sample preparation and mass spectrometry are described in the Methods of the
  manuscript.

## Processing

1. **Search.** Spectronaut, library-free (directDIA).
   - Total kidney: Spectronaut v20.2.250922.92449, mouse UniProt database downloaded on 7 October 2025, cross-run
     normalization "automatic", no imputation; {s['kidney'].n_raw_groups:,} protein groups in the export.
   - DCT cells: one search of all {n_runs:,} runs of both diets, so that protein inference, FDR filtering and
     cross-run normalization ran over both diets; {s['both'].n_raw_groups:,} protein groups in the export.
     - Spectronaut v{search['Spectronaut version']}; FASTA `{search['FASTA']}`;
     - precursor FDR {search['precursor FDR']}; protein FDR {search['protein FDR']}; protein inference "{search['protein inference']}";
     - cross-run normalization "{search['cross-run normalization']}"; imputation: {search['imputation']}.
2. **Filters**, applied to each data set on its own: cells with at least 100 protein groups detected, counted over
   all protein groups of the export, were kept; then protein groups detected in at least 2 of the kept cells were
   kept.
   - Total kidney: {s['kidney'].n_raw_cells} to {s['kidney'].n_cells} cells; {s['kidney'].n_raw_groups:,} to {s['kidney'].n_proteins:,} protein groups.
   - Control: {facts['n_raw_control']} to {s['control'].n_cells} cells; {s['control'].n_raw_groups:,} to {s['control'].n_proteins:,} protein groups.
   - Zero-K: {facts['n_raw_zerok']} to {s['zerok'].n_cells} cells; {s['zerok'].n_raw_groups:,} to {s['zerok'].n_proteins:,} protein groups.
   - Both diets: the {s['both'].n_proteins:,} protein groups kept in both diets.
3. **Matrices.** `layers['intensity']` is the Spectronaut quantity (PG.Quantity). `layers['ln1p_intensity']` is
   ln(1 + intensity). `.X` is the per-protein z-score of ln(1 + intensity) over the cells of the object, clipped at
   +10 and -10. The object of both diets also holds two normalization variants of the comparison of the diets.
4. **Missing values.** A protein group that was not detected in a cell is 0 in the layers and empty in the CSV
   matrix. No value was imputed. The z-scores in `.X` were computed with these zeros.
5. **Embedding.** Principal component analysis of `.X`; neighbour graph of the 10 nearest neighbours on 20
   principal components (`X_pca`); UMAP of that graph (`X_umap`). The object of both diets holds 20 principal
   components of its own, and no graph and no UMAP.
6. **Cell types.** The neighbour graph was clustered with the Leiden algorithm (resolution 2 for total kidney,
   resolution 1 for each diet of the DCT data; random seed 42). Each cluster was compared with all other cells, and
   it received the cell type whose marker proteins were among its most increased proteins. The marker lists are in
   Supplementary Table S2 of the manuscript; the kidney list has an added endothelial row. `cluster_evidence` gives
   the level of marker evidence. Clusters without marker support are `Unknown` (total kidney) or
   `{NOT_RESOLVED}` (DCT data).
7. **Proliferating cells (ProLIF), DCT data.** ProLIF cells were called per cell, not per cluster. A cell was
   called cycling when more of 36 S-phase and G2/M-phase proteins were detected in it than in cells of similar
   depth, judged against 1,000 random protein sets of matched detection rate (Benjamini-Hochberg q < 0.05). It was
   called ProLIF when Slc12a3 (NCC) was also detected. Mki67 and Top2a were not used for the call.
   `cell_type_prolif` is `cell_type` with the ProLIF cells set to ProLIF.
8. **Transferred label, Zero-K cells.** `cell_type_transferred` is the most frequent `cell_type` among the 10
   nearest Control cells in the principal components of both diets.

## Numbers of cells

| data set | `cell_type` |
| --- | --- |
{facts['count_rows']}

ProLIF cells: {facts['n_prolif_control']} of {s['control'].n_cells} Control cells and {facts['n_prolif_zerok']} of {s['zerok'].n_cells} Zero-K cells.

## Reading the files

```python
import anndata as ad

adata = ad.read_h5ad("{s['control'].h5ad}")
adata.obs["cell_type"].value_counts()      # cells per cell type
adata.layers["intensity"]                  # cells x protein groups; 0 = not detected
adata.var.loc["Slc12a3"]                   # accessions, gene symbols and description of a protein group
```

The cell ID is the Spectronaut run name, for example `{EXAMPLE_RUN_KIDNEY}` or
`{EXAMPLE_RUN_DCT}`. The number in square brackets is the column position in the
Spectronaut export; the rest is the name of the raw file. In the DCT data the raw file name holds the diet, the sort
plate and the well.

The objects were written with {software}.

The clusters can be recomputed from the stored graph with
`scanpy.tl.leiden(adata, resolution=1, flavor="igraph", random_state=42)` (resolution 2 for total kidney). A
different software version or platform can return a different partition, so the column `cluster` is the reference.
On Windows, scanpy {facts['software'].get('scanpy', '')} needs 64-bit random integers in its igraph random-number adapter for the seed
to take effect; the clusters were computed with that setting.

## Limits

- The cells of a data set come from one pool of mice. They are not animal-level replicates.
- The number of protein groups detected per cell (`n_proteins`) ranges from {facts['range_kidney'][0]:,} to {facts['range_kidney'][1]:,} (total kidney),
  {facts['range_control'][0]:,} to {facts['range_control'][1]:,} (Control) and {facts['range_zerok'][0]:,} to {facts['range_zerok'][1]:,} (Zero-K).
- `n_proteins` and `total_intensity` count the protein groups of the object. In the object of both diets they
  therefore cover fewer protein groups than in the object of the single diet, and `n_proteins` can be below 100
  (lowest value {facts['range_both'][0]:,}).
- Intensities carry the cross-run normalization of Spectronaut. No further normalization was applied to
  `intensity`, `ln1p_intensity` or `.X`.

## Raw data and code

- Raw mass spectrometry files of all data sets: ProteomeXchange/PRIDE, {PRIDE_ACCESSION} (DOI {PRIDE_DOI}).
- Analysis code: {CODE_REPOSITORY} (not part of this record).

## Files

| file | size (MB) |
| --- | --- |
{file_rows}

## Licence and citation

The data are available under the Creative Commons Attribution 4.0 International licence ({LICENCE}, {LICENCE_URL}).

Please cite the manuscript ({AUTHORS}. {MANUSCRIPT_TITLE}.) and this record ({ZENODO_URL}).
"""
