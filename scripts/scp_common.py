"""Shared paths, output guard and helpers of the analysis scripts.

Imported by the scripts in ``scripts/``; it never runs an analysis by itself.

Layout (relative to the folder that holds ``scripts/``)
------------------------------------------------------
- ``data/``: inputs (see ``data/README.md``); read only.
- ``results/tables/<stage>/``, ``results/figures/<stage>/``, ``results/h5ad/``: outputs. Every output file name
  starts with the name of its stage, and every file goes through ``out_path()``.
- Existing files are never replaced unless the script was started with ``--overwrite``.
- ``use_input()`` refuses the marker versions and objects that the analysis does not use.

Cell IDs of the DCT data: ``parse_dct_cell_id()`` (condition, sort plate, well, file column index).
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
RANDOM_SEED = 42

GENERATED_BY = "SCP kidney and DCT analysis code"
GENERATED_TAG = "SCP analysis code"

# Original preprocessing (scp_kidney_upload.py, scp_dct_upload.py).
MIN_CELL_PROTEINS = 100
MIN_PROTEIN_CELLS = 2

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
DCT_DIR = DATA_DIR / "DCT"
KIDNEY_DIR = DATA_DIR / "KIDNEY"
INPUTS = {
    "dct": {
        "raw_csv": DCT_DIR / "DCT_758cells.csv",
        "v1_h5ad": DCT_DIR / "V1" / "DCT_758cells_processed.h5ad",
        "markers": DCT_DIR / "DCT_markers_V1.xlsx",
        "cell_metadata": DCT_DIR / "V1" / "DCT_758cells_CellMetadata.csv",
        "detailed_umap": DCT_DIR / "V1" / "Detailed_UMAP_CellType_Info.csv",
        "umap_meta": DCT_DIR / "V1" / "UMAP_coords_with_meta.csv",
        "paga": DCT_DIR / "V1" / "Leiden_3_PAGA_connectivities.csv",
        "aldo_percell": DCT_DIR / "V1" / "20260122_144608" / "PerCell_scores.csv",
    },
    "kidney": {
        "raw_csv": KIDNEY_DIR / "KIDNEY_768cells.csv",
        "markers": KIDNEY_DIR / "KIDNEY_markers_V3.xlsx",
        "sens_dynr": KIDNEY_DIR / "V3" / "SENS_DYNR_per_cell_Leiden_2.0.csv",
        "protein_ids_after": KIDNEY_DIR / "V3" / "ProteinIDs_byCellType_AFTER_Leiden_2.0.csv",
        "protein_ids_before": KIDNEY_DIR / "V3" / "ProteinIDs_byCellType_BEFORE_Leiden_2.0.csv",
        "umap_after_png": KIDNEY_DIR / "V3" / "UMAP_AFTER_Leiden_2.0.png",
        "umap_before_png": KIDNEY_DIR / "V3" / "UMAP_BEFORE_Leiden_2.0.png",
        # Processed object of the earlier preprocessing run: same 712 x 4,054 matrix; its PCA, graph, UMAP
        # and Leiden 2.0 are reused (see README.md, "Kidney").
        "processed_v2_h5ad": KIDNEY_DIR / "KIDNEY_768cells_processed.h5ad",
    },
}

# New DCT data (authors, 28 Oct 2026): control vs zero-K diet for 4 days, one pool of 3 mice per
# condition, P1/P2 = sort plates of the pooled cells, Spectronaut searched each condition separately.
DCT_NEW_INPUTS = {
    "Ctrl": DCT_DIR / "DCT_Ctrl_768cells.csv",
    "ZeroK": DCT_DIR / "DCT_ZeroK_767cells.csv",
}
DCT_CELL_ID_RE = re.compile(r"^\[(\d+)\] (Ctrl|ZeroK)_GFP_1cell_WISHDIA_P(\d+)_(\d{4})\.htrms\.PG\.Quantity$")

# Inputs the revision does not use for analysis (the project notes).
EXCLUDED_INPUTS = (
    re.compile(r"(^|/)data/DCT/DCT_markers_V2\.xlsx$", re.I),
    re.compile(r"(^|/)data/DCT/V2/", re.I),
    re.compile(r"(^|/)data/KIDNEY/KIDNEY_markers_V1\.xlsx$", re.I),
    re.compile(r"(^|/)data/KIDNEY/V1/", re.I),
    re.compile(r"(^|/)data/KIDNEY/V3_reannotation_v2", re.I),
)

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
OUT_ROOTS = {
    "tables": PROJECT_ROOT / "results" / "tables",
    "figures": PROJECT_ROOT / "results" / "figures",
    "h5ad": PROJECT_ROOT / "results" / "h5ad",
}
INDEX_PATH = OUT_ROOTS["tables"] / "INDEX.md"
# intake_core objects of the new DCT data (labels come from labels_core for control and zerok_core for zero-K).
DCT_NEW_OBJECTS = {
    "Ctrl": OUT_ROOTS["h5ad"] / "intake_core_DCT_ctrl.h5ad",
    "ZeroK": OUT_ROOTS["h5ad"] / "intake_core_DCT_zerok.h5ad",
    "joint": OUT_ROOTS["h5ad"] / "intake_core_DCT_joint.h5ad",
}
ALLOWED_EXT = {
    "tables": {".csv", ".md", ".json", ".txt"},
    "figures": {".pdf", ".png", ".tif"},
    "h5ad": {".h5ad"},
}
TASK_RE = re.compile(r"^([a-z][a-z0-9_]*)$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SUBDIR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

CANONICAL = {
    "dct": {
        "labels": OUT_ROOTS["tables"] / "kidney_object" / "kidney_object_dct_canonical_labels.csv",
        "h5ad": OUT_ROOTS["h5ad"] / "kidney_object_DCT_V1_submitted.h5ad",
        "label_columns": ("v1_cell_label", "v1_cluster_label"),
    },
    "kidney": {
        "labels": OUT_ROOTS["tables"] / "kidney_object" / "kidney_object_kidney_canonical_labels.csv",
        "h5ad": OUT_ROOTS["h5ad"] / "kidney_object_KIDNEY_submitted.h5ad",
        "label_columns": ("kidney_reannot_label", "kidney_pre_label"),
    },
}

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
DCT_LABEL_ORDER = ("DCT1", "DCT2", "ProLIF", "Unknown")
DCT_LABEL_SOURCES = {
    "v1_cell_label": "V1 h5ad obs['predicted_cell_type']: per-cell argmax of DCT_markers_V1 scores (Fig 3B, 3D, 3G, DPT root, 3H-K)",
    "v1_cluster_label": "V1 h5ad obs['leiden_res_3.00_named']: Leiden 3.0 cluster majority (Fig 3A, the '32%')",
}
KIDNEY_LABEL_SOURCES = {
    "kidney_reannot_label": "data/KIDNEY/V3/SENS_DYNR_per_cell_Leiden_2.0.csv 'cell_type_v2': after re-annotation (Fig 2B; primary)",
    "kidney_pre_label": "data/KIDNEY/V3/SENS_DYNR_per_cell_Leiden_2.0.csv 'cell_type_pre': Leiden 2.0 cluster majority before re-annotation (Fig 2A)",
}
# Full V3 names -> short names (KIDNEY_markers_V3.xlsx 'Short' column; the project notes labels).
KIDNEY_ABBREVIATIONS = {
    "Proximal convoluted tubule": "PCT",
    "Proximal straight tubule": "PST",
    "Descending thin limb of LOH": "DTL",
    "Ascending thin limb of LOH": "ATL",
    "Thick ascending limb of LOH": "TAL",
    "Macula densa": "MD",
    "DCT1": "DCT1",
    "DCT2": "DCT2",
    "CNT": "CNT",
    "CD-principal cells": "PC",
    "CD-intercalated cells (type A)": "IC-A",
    "CD-intercalated cells (type B)": "IC-B",
    "Descending vasa recta": "DVR",
    "Ascending vasa recta": "AVR",
    "Juxtaglomerular cells": "JG",
    "Adult podocytes": "Podo",
    "Unknown": "Unknown",
}
# Display order along the nephron, then vascular/JG, then Unknown.
KIDNEY_LABEL_ORDER = ("PCT", "PST", "DTL", "ATL", "TAL", "MD", "DCT1", "DCT2", "CNT", "PC",
                      "IC-A", "IC-B", "DVR", "AVR", "JG", "Podo", "Unknown")
UNKNOWN_COLOR = "#B3B3B3"
# Technical plate label (DCT): reference categorical slots 1-2.
PLATE_PALETTE = {"P1": "#2a78d6", "P2": "#eb6834"}

PACKAGES = ("scanpy", "anndata", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "igraph",
            "leidenalg", "umap-learn", "pynndescent", "numba", "matplotlib", "seaborn", "openpyxl")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    """Project-relative path for manifests and messages."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def infer_plate_from_cell_id(cell_id: object) -> str:
    """DCT sort plate (P1/P2) from the cell ID; 'unknown' when the ID carries none."""
    match = re.search(r"_P(\d+)_", str(cell_id))
    return f"P{match.group(1)}" if match else "unknown"


def parse_dct_cell_id(cell_id: object) -> dict[str, object]:
    """Condition, sort plate, well and file column index of a new-DCT cell ID (intake_core).

    ``[n]`` is the column position in the export, not the well: after a missing
    well the two differ (zero-K: ``[368]`` is well P1_0369). Raises ValueError
    when the ID does not parse.
    """
    match = DCT_CELL_ID_RE.match(str(cell_id))
    if not match:
        raise ValueError(f"cell ID does not parse: {cell_id!r}")
    return {"file_column_index": int(match.group(1)), "condition": match.group(2),
            "plate": f"P{match.group(3)}", "well": match.group(4)}


def ordered_labels(values, order=DCT_LABEL_ORDER) -> list[str]:
    present = set(map(str, values))
    return [label for label in order if label in present] + sorted(present.difference(order))


def clean_gene_symbol(value: object) -> str:
    """Mouse-style first symbol of a gene/protein-group entry (strips NBSP)."""
    symbol = str(value).split(";")[0].strip().replace(" ", "")
    if not symbol or symbol.lower() == "nan":
        return "Nan"
    return symbol.lower().capitalize()


def read_marker_workbook(path: Path) -> dict[str, list[str]]:
    """Marker workbook rows: first column cell type, second column comma-separated genes."""
    marker_df = pd.read_excel(path)
    markers: dict[str, list[str]] = {}
    for row in marker_df.itertuples(index=False):
        cell_type = str(row[0]).strip()
        genes = [clean_gene_symbol(g) for g in str(row[1]).split(",")]
        genes = [g for g in genes if g and g != "Nan"]
        if cell_type and genes:
            markers[cell_type] = genes
    return markers


def markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    """Small Markdown table writer (avoids an optional tabulate dependency)."""
    def fmt(value) -> str:
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return ""
            if float(value).is_integer() and abs(value) < 1e12:
                return str(int(value))
            return f"{value:.1f}" if abs(value) >= 1000 else floatfmt.format(value)
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def compare_groups(x, y) -> dict[str, float]:
    """Two-sided Mann-Whitney U plus effect sizes (group x relative to reference y)."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    out = {"n_group": len(x), "n_reference": len(y), "median_group": np.nan, "median_reference": np.nan,
           "median_difference": np.nan, "cohens_d": np.nan, "rank_biserial": np.nan,
           "mannwhitney_u": np.nan, "p_value": np.nan}
    if len(x) == 0 or len(y) == 0:
        return out
    out.update(median_group=float(np.median(x)), median_reference=float(np.median(y)),
               median_difference=float(np.median(x) - np.median(y)))
    test = stats.mannwhitneyu(x, y, alternative="two-sided")
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
                     / max(len(x) + len(y) - 2, 1)) if len(x) > 1 and len(y) > 1 else np.nan
    out.update(cohens_d=float((x.mean() - y.mean()) / pooled) if pooled and pooled > 0 else np.nan,
               rank_biserial=float(2 * test.statistic / (len(x) * len(y)) - 1),
               mannwhitney_u=float(test.statistic), p_value=float(test.pvalue))
    return out


def add_bh(frame: pd.DataFrame, group_cols: list[str], p_col: str = "p_value", out_col: str = "p_fdr_bh") -> pd.DataFrame:
    """Benjamini-Hochberg FDR within each block defined by group_cols (NaN p-values are skipped)."""
    frame = frame.copy()
    frame[out_col] = np.nan
    for _, index in frame.groupby(group_cols, dropna=False).groups.items():
        p = frame.loc[index, p_col].to_numpy(dtype=float)
        ok = np.isfinite(p)
        if ok.any():
            q = np.full(len(p), np.nan)
            q[ok] = stats.false_discovery_control(p[ok], method="bh")
            frame.loc[index, out_col] = q
    return frame


# ---------------------------------------------------------------------------
# Naming, figures and manifests
# ---------------------------------------------------------------------------
def out_path(task: str, kind: str, name: str, ext: str, subdir: str = "") -> Path:
    """Path of an output; the file name starts with the name of the stage.

    ``task`` is the stage (e.g. ``dct_labels``), ``kind`` one of ``tables``,
    ``figures`` or ``h5ad``, ``name`` the file name without the stage prefix,
    and ``ext`` the extension including the dot. Tables and figures go into
    ``results/<kind>/<stage>/`` (or one level deeper when ``subdir`` is given);
    AnnData files into ``results/h5ad/``. The file is not created.
    """
    match = TASK_RE.match(task)
    if not match:
        raise ValueError(f"task must look like 'kidney_object', got {task!r}")
    if kind not in OUT_ROOTS:
        raise ValueError(f"kind must be one of {sorted(OUT_ROOTS)}, got {kind!r}")
    if not NAME_RE.match(name):
        raise ValueError(f"name must be a bare file name, got {name!r}")
    if ext not in ALLOWED_EXT[kind]:
        raise ValueError(f"extension {ext!r} not allowed for {kind}; use one of {sorted(ALLOWED_EXT[kind])}")
    if subdir and (kind == "h5ad" or not SUBDIR_RE.match(subdir)):
        raise ValueError(f"subdir must be one plain folder name for tables/figures, got {subdir!r}")
    filename = f"{match.group(1)}_{name}{ext}"
    folder = OUT_ROOTS[kind] if kind == "h5ad" else OUT_ROOTS[kind] / task
    return (folder / subdir if subdir else folder) / filename


def _check_writable(path: Path, overwrite: bool) -> Path:
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in OUT_ROOTS.values()):
        raise PermissionError(f"Refusing to write outside the results folders: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {rel(path)} (rerun with --overwrite to replace earlier outputs)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, task: str, name: str, *, overwrite: bool = False, description: str = "",
                subdir: str = "", formats: tuple[str, ...] = (".pdf", ".png"), tight: bool = True) -> list[Path]:
    """Save one figure as PDF and 600 dpi PNG (optionally TIFF); each carries a provenance tag.

    ``tight=False`` keeps the figure's own size (e.g. an exact journal column width).
    TIFF is written as LZW-compressed RGB without alpha.
    """
    import matplotlib.pyplot as plt

    script = rel(Path(sys.argv[0])) if sys.argv and sys.argv[0] else "interactive"
    paths = []
    for ext in formats:
        path = _check_writable(out_path(task, "figures", name, ext, subdir), overwrite)
        kwargs = {"dpi": 600, "bbox_inches": "tight" if tight else None}
        if ext == ".pdf":
            kwargs["metadata"] = {"Creator": f"{GENERATED_BY}; {script}", "Subject": description or name,
                                  "Title": path.stem, "Keywords": GENERATED_TAG}
        elif ext == ".png":
            kwargs["metadata"] = {"Software": f"{GENERATED_BY}; {script}", "Description": description or name,
                                  "Title": path.stem}
        elif ext == ".tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw",
                                    "description": f"{GENERATED_BY}; {script}; {description or name}"}
            kwargs["facecolor"] = "white"
        fig.savefig(path, **kwargs)
        if ext == ".tif":
            from PIL import Image

            with Image.open(path) as im:
                rgb = im.convert("RGB") if im.mode != "RGB" else im.copy()
            rgb.save(path, compression="tiff_lzw", dpi=(600, 600),
                     description=f"{GENERATED_BY}; {script}; {description or name}")
        paths.append(path)
    plt.close(fig)
    print(f"Saved figure: {rel(paths[0].with_suffix(''))}{'/'.join(formats)}")
    return paths


def write_manifest(path: Path, payload: dict, *, overwrite: bool = False) -> Path:
    """Write a JSON manifest (package versions, seed, command and the given payload)."""
    path = _check_writable(path, overwrite)
    manifest = {
        "generated_by": GENERATED_BY,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "random_seed": RANDOM_SEED,
        **payload,
    }
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Saved manifest: {rel(path)}")
    return path


def add_common_args(parser) -> None:
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace this step's own earlier outputs.")


class Run:
    """Input recorder, output guard and manifest writer for one step of a task."""

    def __init__(self, task: str, step: str, args=None):
        if not TASK_RE.match(task):
            raise ValueError(f"task must look like 'kidney_object', got {task!r}")
        self.task, self.step, self.args = task, step, args
        self.overwrite = bool(getattr(args, "overwrite", False))
        self.inputs: dict[str, dict[str, object]] = {}
        self.outputs: list[str] = []
        self.notes: list[str] = []
        self.started = now_utc()
        self.manifest_path = out_path(task, "tables", f"step_{step}_manifest", ".json")
        print(f"[{task}/{step}] started {self.started}")

    # inputs ---------------------------------------------------------------
    def use_input(self, path: Path, label: str = "", *, allow_excluded: bool = False) -> Path:
        """Record an input (with SHA256) and return its path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        key = rel(path)
        if not allow_excluded and any(p.search(key) for p in EXCLUDED_INPUTS):
            raise PermissionError(f"{key} is not used by the analysis (marker and label versions).")
        if key not in self.inputs:
            self.inputs[key] = {"label": label, "sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    def note(self, text: str) -> None:
        print(f"[{self.task}/{self.step}] {text}")
        self.notes.append(text)

    # outputs --------------------------------------------------------------
    def out(self, kind: str, name: str, ext: str, subdir: str = "") -> Path:
        path = _check_writable(out_path(self.task, kind, name, ext, subdir), self.overwrite)
        self.outputs.append(rel(path))
        return path

    def table(self, frame: pd.DataFrame, name: str, *, index: bool = False, subdir: str = "") -> Path:
        path = self.out("tables", name, ".csv", subdir)
        frame.to_csv(path, index=index)
        print(f"Saved table: {rel(path)}")
        return path

    def text(self, content: str, name: str, ext: str = ".md", subdir: str = "") -> Path:
        path = self.out("tables", name, ext, subdir)
        path.write_text(content, encoding="utf-8")
        print(f"Saved text: {rel(path)}")
        return path

    def figure(self, fig, name: str, description: str = "", subdir: str = "",
               formats: tuple[str, ...] = (".pdf", ".png"), tight: bool = True) -> list[Path]:
        paths = save_figure(fig, self.task, name, overwrite=self.overwrite, description=description,
                            subdir=subdir, formats=formats, tight=tight)
        self.outputs.extend(rel(path) for path in paths)
        return paths

    def h5ad(self, adata, name: str) -> Path:
        path = self.out("h5ad", name, ".h5ad")
        adata.write_h5ad(path)
        print(f"Saved AnnData: {rel(path)}")
        return path

    def finish(self, parameters: dict[str, object] | None = None, status: str = "completed") -> Path:
        params = {} if self.args is None else {k: (str(v) if isinstance(v, Path) else v)
                                               for k, v in vars(self.args).items()}
        params.update(parameters or {})
        payload = {
            "task": self.task,
            "step": self.step,
            "status": status,
            "script": rel(Path(sys.argv[0])),
            "command": [sys.executable, *sys.argv],
            "launcher": "python",
            "started_utc": self.started,
            "finished_utc": now_utc(),
            "parameters": params,
            "inputs": self.inputs,
            "outputs": {p: sha256(PROJECT_ROOT / p) for p in self.outputs},
            "notes": self.notes,
        }
        return write_manifest(self.manifest_path, payload, overwrite=self.overwrite)


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------
@dataclass
class Rebuild:
    """Raw Spectronaut matrix and the original filtering, rebuilt from the CSV."""

    dataset: str
    raw_intensity: pd.DataFrame      # all cells x all protein rows (NaN -> 0), float32
    protein_meta_all: pd.DataFrame   # one row per protein row: var_name, protein_group, gene_symbol, description
    keep_cells: np.ndarray           # bool over raw cells (>= MIN_CELL_PROTEINS proteins > 0)
    keep_proteins: np.ndarray        # bool over protein rows (> 0 in >= MIN_PROTEIN_CELLS retained cells)
    intensity: pd.DataFrame          # retained cells x retained proteins, float32
    ln1p_intensity: pd.DataFrame     # ln(1 + intensity), float32 (as sc.pp.log1p on float32)

    @property
    def protein_meta(self) -> pd.DataFrame:
        return self.protein_meta_all.loc[self.keep_proteins].reset_index(drop=True)


def _var_names(symbols: list[str], dataset: str) -> list[str]:
    """Unique protein names as the original scripts made them.

    DCT (scp_dct_upload.py): a repeated name gets its column index (``Name_<i>``).
    KIDNEY (scp_kidney_upload.py): a repeated name gets a running count (``Name_1``, ``Name_2``).
    """
    seen: dict[str, int] = {}
    names = []
    for idx, name in enumerate(symbols):
        count = seen.get(name, 0)
        if count == 0:
            names.append(name)
        else:
            names.append(f"{name}_{idx}" if dataset == "dct" else f"{name}_{count}")
        seen[name] = count + 1
    return names


def rebuild_ln_intensity(dataset: str, run: Run | None = None) -> Rebuild:
    """Rebuild intensity and ln(1 + intensity) from the raw CSV with the original preprocessing.

    Missing -> 0 (float32); first ``PG.Genes`` symbol, lower-cased then
    capitalized (missing symbols become ``Nan`` as under the original pandas);
    duplicates made unique as in the dataset's original script; cells with
    >= 100 proteins > 0; then proteins > 0 in >= 2 retained cells.
    """
    if dataset not in INPUTS:
        raise ValueError(f"dataset must be one of {sorted(INPUTS)}")
    path = INPUTS[dataset]["raw_csv"]
    if run is not None:
        run.use_input(path, f"raw Spectronaut intensities ({dataset})")
    data = pd.read_csv(path, low_memory=False)
    meta = data.iloc[:, :3]
    symbols = (meta["PG.Genes"].fillna("nan").astype(str).str.split(";").str[0]
               .str.strip().str.lower().str.capitalize().tolist())
    names = _var_names(symbols, dataset)
    protein_meta = pd.DataFrame({
        "var_name": names,
        "protein_group": meta["PG.ProteinGroups"].astype(str).to_numpy(),
        "gene_symbol": meta["PG.Genes"].fillna("").astype(str).to_numpy(),
        "protein_description": meta["PG.ProteinDescriptions"].fillna("").astype(str).to_numpy(),
    })
    values = data.iloc[:, 3:].fillna(0).astype(np.float32)
    cells = values.columns.astype(str)
    raw = pd.DataFrame(values.to_numpy().T, index=pd.Index(cells, name="cell_id"), columns=names)
    keep_cells = (raw.to_numpy() > 0).sum(axis=1) >= MIN_CELL_PROTEINS
    kept = raw.loc[keep_cells]
    keep_proteins = (kept.to_numpy() > 0).sum(axis=0) >= MIN_PROTEIN_CELLS
    intensity = kept.loc[:, keep_proteins]
    ln1p = pd.DataFrame(np.log1p(intensity.to_numpy()), index=intensity.index, columns=intensity.columns)
    if run is not None:
        run.note(f"Rebuilt {dataset} matrix from {rel(path)}: {raw.shape[0]} cells x {raw.shape[1]} protein rows -> "
                 f"{intensity.shape[0]} cells x {intensity.shape[1]} proteins.")
    return Rebuild(dataset, raw, protein_meta, keep_cells, keep_proteins, intensity, ln1p)


def scale_like_scanpy(ln1p: np.ndarray, max_value: float = 10.0) -> np.ndarray:
    """Per-protein z-score (ddof=1; zero SD -> 1) clipped at +/- max_value, as sc.pp.scale."""
    x = np.asarray(ln1p, dtype=np.float64)
    sd = x.std(axis=0, ddof=1)
    return np.clip((x - x.mean(axis=0)) / np.where(sd == 0, 1.0, sd), -max_value, max_value)


# ---------------------------------------------------------------------------
# Canonical objects
# ---------------------------------------------------------------------------
def load_canonical(dataset: str):
    """Return ``(adata, labels)`` for 'dct' or 'kidney' from the kidney_object outputs.

    ``labels`` is the canonical CSV indexed by ``cell_id`` in object order. The
    function checks that the CSV and the h5ad hold the same cells and labels.
    """
    import anndata as ad

    if dataset not in CANONICAL:
        raise ValueError(f"dataset must be one of {sorted(CANONICAL)}")
    spec = CANONICAL[dataset]
    for path in (spec["labels"], spec["h5ad"]):
        if not path.exists():
            raise FileNotFoundError(f"{rel(path)} not found; run kidney_object first.")
    labels = pd.read_csv(spec["labels"], dtype={"cell_id": str}).set_index("cell_id")
    adata = ad.read_h5ad(spec["h5ad"])
    if list(adata.obs_names.astype(str)) != list(labels.index):
        raise ValueError(f"{dataset}: canonical CSV and h5ad cell order differ.")
    for col in spec["label_columns"]:
        if not np.array_equal(adata.obs[col].astype(str).to_numpy(), labels[col].astype(str).to_numpy()):
            raise ValueError(f"{dataset}: label column {col} differs between CSV and h5ad.")
    return adata, labels
