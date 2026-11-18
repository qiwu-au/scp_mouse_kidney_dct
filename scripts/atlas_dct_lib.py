"""atlas_dct shared definitions and functions. Runs no analysis by itself.

Imported by the ``atlas_dct_*.py`` step scripts.
Plan: ``the analysis plan`` (Editors 1; R1 ProLIF prevalence).

Decisions by the authors (15 Oct 2026)
-----------------------------------
- RNA reference level: atlas ``Celltype`` (primary; the plan named ``Celltype_finest``, which has the same DCT
  labels, DCT and DCT-CNT, on about half the cells); ``Celltype_finest`` in the step-1 summary only.
- The atlas has no DCT1 / DCT2 labels. Published classes (DCT, DCT-CNT, CNT, CTAL, MTAL) are used throughout; a
  atlas_dct-derived DCT1 / DCT2 split of the atlas DCT cells is added (option 1a): Leiden subclusters of the atlas
  ``Celltype`` DCT cells on the stored scVI latent (``obsm['X_scVI']``), each subcluster named by a fixed rule on
  the DCT V1 workbook lists without Umod and Slc12a3. Labelled "atlas_dct-derived" wherever used.
- No second reference (Su et al. 2024) in this run.

Contents
--------
- Paths and settings: RNA atlas (read-only; loaded only by ``atlas_dct_atlas_subset.py``), the DCT-region subset
  written by that step, the split table written by ``atlas_dct_dct_split.py``.
- ``load_region``: DCT-region atlas cells (raw counts, all 16,711 genes) with the RNA group columns.
- ``lognorm``: counts per cell / total x 10,000, natural log1p (as kidney_rna).
- ``group_stats``: per group and gene, n cells, mean log-normalized, fraction with counts > 0.
- ``load_scp``: kidney_object canonical DCT object and labels (both V1 label columns) with ln(1 + intensity) and detection.
- ``naming_lists``: DCT1 / DCT2 naming genes from the DCT V1 workbook (Umod and Slc12a3 removed).
- Figure helpers (``Sheet``, ``text_checks``, ``save_checked``), adapted from ``kidney_rna_paired_figures.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import scp_common as cc
import cluster_de_lib as t5
import dotplot_lib as t6
import kidney_rna_lib as L7

TASK = "atlas_dct"
SEED = cc.RANDOM_SEED
DATASET = "dct"

# Atlas ----------------------------------------------------------------------------------------------
ATLAS = L7.ATLAS
ATLAS_EXPECTED_SHAPE = L7.ATLAS_EXPECTED_SHAPE
ATLAS_EXPECTED_RAW_GENES = L7.ATLAS_EXPECTED_RAW_GENES
TARGET_SUM = L7.TARGET_SUM
LEVEL = "Celltype"                                   # primary atlas level (authors, 15 Oct 2026)
REGION_CLASSES = ("DCT", "DCT-CNT", "CNT", "CTAL", "MTAL", "LOH")
PUBLISHED_GROUPS = ("DCT", "DCT-CNT", "CNT", "CTAL", "MTAL")   # LOH: mixed class, summarized only (as kidney_rna)
TAL_CLASSES = ("CTAL", "MTAL")
OBS_KEEP = ("Origin", "Source", "Technology", "Tissue_res", "Age", "Genetic_background", "Gender", "batch",
            "Celltype", "Celltype_finest_lowres", "Celltype_finest", "total_counts", "n_genes_by_counts",
            "pct_counts_mt")
REGION_H5AD = cc.out_path(TASK, "h5ad", "atlas_dct_region", ".h5ad")
KIDNEY_RNA_LEVEL_COUNTS = cc.out_path(L7.TASK, "tables", "atlas_level_counts", ".csv")
RNA_SOURCES = {"all": None, "nuclei": "Nuclei", "cells": "Cell"}
SOURCE_TEXT = {"all": "all atlas cells and nuclei", "nuclei": "atlas nuclei (snRNA-seq)",
               "cells": "atlas cells (scRNA-seq)"}

# atlas_dct-derived split of the atlas DCT cells ------------------------------------------------------------
DCT1_ATLAS_DCT = "DCT1 (atlas_dct)"
DCT2_ATLAS_DCT = "DCT2 (atlas_dct)"
DCT_UNASSIGNED = "DCT unassigned (atlas_dct)"
DERIVED_GROUPS = (DCT1_ATLAS_DCT, DCT2_ATLAS_DCT, "DCT-CNT", "CNT", "CTAL", "MTAL")
SPLIT_RESOLUTIONS = (0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
SPLIT_SEEDS = (SEED, *range(10))                     # 11 Leiden seeds (as kidney_annotation)
SPLIT_N_NEIGHBORS = 15                               # scanpy default
NAMING_THRESHOLD = 0.25                              # |DCT1 score - DCT2 score| (standardized mean difference)
NAMING_THRESHOLDS = (0.10, 0.25, 0.50)
NAMING_EXCLUDED = ("Umod", "Slc12a3")                # authors, 15 Oct 2026 (TAL-associated; listed for both)
NAMING_EXPECTED = {"DCT1": ("Pvalb", "Egf", "Trpm7", "Col5a2", "Ptgfr"),
                   "DCT2": ("Slc8a1", "Calb1", "S100g", "Klk1", "Trpv5")}
SPLIT_CSV = cc.out_path(TASK, "tables", "dct_split_per_cell", ".csv")

# SCP ------------------------------------------------------------------------------------------------
LABEL_COLUMNS = ("v1_cell_label", "v1_cluster_label")
SCP_GROUPS = t6.DCT_ROW_ORDER                       # DCT1, DCT2, ProLIF (Unknown n = 4 not a group)
LABEL_DISPLAY = {"v1_cell_label": "V1 per-cell labels", "v1_cluster_label": "V1 cluster labels"}
LABEL_SHORT = {"v1_cell_label": "cell", "v1_cluster_label": "cluster"}
EXPECTED_COUNTS = {k: t6.EXPECTED_COUNTS[k] for k in LABEL_COLUMNS}
REVIEWER_CORE_DIR = cc.OUT_ROOTS["tables"] / "reviewer_core"
DOTPLOT_DIR = cc.OUT_ROOTS["tables"] / "dotplot"

# Thresholds ---------------------------------------------------------------------------------------------
DETECTION = 0.10                                     # as kidney_rna (primary)
DETECTION_SENS = (0.05, 0.10, 0.20)

# Colours ----------------------------------------------------------------------------------------------------
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "surface": "#fcfcfb", "background_cells": "#d6d5cf"}         # = qc_mito_lib.INK
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING_COLORS = ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3b0ad", "#e34948", "#9c2423"]  # = cluster_de
SCP_PALETTE = {"DCT1": "#1f77b4", "DCT2": "#ff7f0e", "ProLIF": "#2ca02c", "Unknown": "#B3B3B3"}  # Fig 3
RNA_PALETTE = {"DCT": "#4a3aa7", DCT1_ATLAS_DCT: "#1f77b4", DCT2_ATLAS_DCT: "#ff7f0e", DCT_UNASSIGNED: "#B3B3B3",
               "DCT-CNT": "#e87ba4", "CNT": "#1baf7a", "CTAL": "#eda100", "MTAL": "#b07800", "LOH": "#898781"}


def rna_groups(derived: bool) -> tuple[str, ...]:
    return DERIVED_GROUPS if derived else PUBLISHED_GROUPS


def group_column(derived: bool) -> str:
    return "rna_group_derived" if derived else "rna_group"


LEVEL_TEXT = {False: "atlas Celltype classes", True: "atlas_dct-derived DCT1 / DCT2 split + atlas classes"}
LEVEL_SHORT = {False: "published", True: "derived"}


# ---------------------------------------------------------------------------
# Atlas region
# ---------------------------------------------------------------------------
def load_region(run: cc.Run, with_split: bool = True):
    """DCT-region atlas cells (atlas_dct_atlas_subset.py): .X = raw counts (CSR float32, 16,711 genes).

    obs gains ``rna_group`` (atlas Celltype) and, with ``with_split``, the per-cell atlas_dct split
    (``dct_subcluster``, ``dct_split_label``) and ``rna_group_derived`` (split label for DCT cells, Celltype
    otherwise).
    """
    import anndata as ad

    if not REGION_H5AD.exists():
        raise FileNotFoundError(f"{cc.rel(REGION_H5AD)} not found; run atlas_dct_atlas_subset.py first.")
    a = ad.read_h5ad(run.use_input(REGION_H5AD, "atlas_dct DCT-region atlas subset (atlas_dct_atlas_subset.py)"))
    a.obs["rna_group"] = a.obs[LEVEL].astype(str)
    if with_split:
        if not SPLIT_CSV.exists():
            raise FileNotFoundError(f"{cc.rel(SPLIT_CSV)} not found; run atlas_dct_dct_split.py first.")
        sp = pd.read_csv(run.use_input(SPLIT_CSV, "atlas_dct-derived DCT1 / DCT2 split (atlas_dct_dct_split.py)"),
                         dtype={"atlas_cell": str}).set_index("atlas_cell")
        dct = a.obs["rna_group"] == "DCT"
        if set(sp.index) != set(a.obs_names[dct]):
            raise ValueError("split table and atlas DCT cells differ")
        a.obs["dct_subcluster"] = ""
        a.obs["dct_split_label"] = ""
        a.obs.loc[dct, "dct_subcluster"] = sp.loc[a.obs_names[dct], "subcluster"].astype(str).to_numpy()
        a.obs.loc[dct, "dct_split_label"] = sp.loc[a.obs_names[dct], "split_label"].astype(str).to_numpy()
        a.obs["rna_group_derived"] = np.where(dct, a.obs["dct_split_label"], a.obs["rna_group"])
    return a


def lognorm(counts) -> sparse.csr_matrix:
    """counts / row total x 10,000, natural log1p (float32 CSR); totals over all genes of the matrix."""
    m = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    tot = np.asarray(m.sum(axis=1), dtype=np.float64).ravel()
    m.data *= np.repeat((TARGET_SUM / np.where(tot > 0, tot, 1)).astype(np.float32), np.diff(m.indptr))
    np.log1p(m.data, out=m.data)
    return m


def source_mask(obs: pd.DataFrame, src: str) -> np.ndarray:
    source = RNA_SOURCES[src]
    return np.ones(len(obs), dtype=bool) if source is None else (obs["Source"].astype(str) == source).to_numpy()


def group_stats(ln: sparse.csr_matrix, counts: sparse.csr_matrix, groups: pd.Series, order, genes: pd.Index,
                mask: np.ndarray | None = None) -> dict[str, pd.DataFrame]:
    """Per group (rows, in ``order``) x gene: n cells, mean log-normalized, fraction with counts > 0."""
    g = groups.astype(str).to_numpy()
    keep = np.ones(len(g), dtype=bool) if mask is None else mask
    mean, frac, n = [], [], []
    for grp in order:
        idx = np.flatnonzero(keep & (g == grp))
        n.append(len(idx))
        if len(idx) == 0:
            mean.append(np.full(len(genes), np.nan))
            frac.append(np.full(len(genes), np.nan))
            continue
        mean.append(np.asarray(ln[idx].mean(axis=0), dtype=np.float64).ravel())
        frac.append(np.asarray((counts[idx] > 0).mean(axis=0), dtype=np.float64).ravel())
    return {"n": pd.Series(n, index=list(order), name="n_cells"),
            "mean": pd.DataFrame(mean, index=list(order), columns=genes),
            "frac": pd.DataFrame(frac, index=list(order), columns=genes)}


# ---------------------------------------------------------------------------
# SCP
# ---------------------------------------------------------------------------
def load_scp(run: cc.Run, checks: t5.Checks | None = None):
    """kidney_object canonical DCT object and labels; ln(1 + intensity) (missing = 0) and detection (intensity > 0)."""
    adata, labels = t5.load(run, DATASET)
    if checks is not None:
        for col in LABEL_COLUMNS:
            observed = labels[col].astype(str).value_counts().to_dict()
            checks.add(f"{col} counts equal the project notes", "; ".join(f"{k} {v}" for k, v in sorted(observed.items())),
                       observed == EXPECTED_COUNTS[col])
    names = pd.Index(adata.var_names.astype(str))
    inten = cc.dense(adata.layers["intensity"]).astype(np.float64)
    ln1p = pd.DataFrame(np.log1p(inten), index=labels.index, columns=names)
    det = pd.DataFrame(inten > 0, index=labels.index, columns=names)
    return adata, labels, ln1p, det


def scp_group_stats(ln1p: pd.DataFrame, det: pd.DataFrame, labels: pd.Series, order=SCP_GROUPS):
    lab = labels.astype(str).to_numpy()
    mean = pd.DataFrame([ln1p.to_numpy()[lab == g].mean(axis=0) for g in order], index=list(order),
                        columns=ln1p.columns)
    frac = pd.DataFrame([det.to_numpy()[lab == g].mean(axis=0) for g in order], index=list(order),
                        columns=det.columns)
    n = pd.Series([int((lab == g).sum()) for g in order], index=list(order), name="n_cells")
    return {"n": n, "mean": mean, "frac": frac}


def shared_features(protein_names, atlas_genes) -> pd.DataFrame:
    """Protein -> atlas gene (case-insensitive; protein names with '_' (duplicate entries) and 'Nan' excluded)."""
    return L7.match_proteins(protein_names, atlas_genes)


def naming_lists(run: cc.Run, checks: t5.Checks | None = None) -> dict[str, list[str]]:
    """DCT1 / DCT2 rows of the DCT V1 workbook without Umod and Slc12a3 (approved naming genes)."""
    wb = {short: genes for short, _, genes in
          t6.read_dct_workbook(run.use_input(cc.INPUTS["dct"]["markers"], "DCT V1 marker workbook (Supp Table 1)"))}
    out = {k: [g for g in wb[k] if g not in NAMING_EXCLUDED] for k in ("DCT1", "DCT2")}
    if checks is not None:
        for k, exp in NAMING_EXPECTED.items():
            checks.add(f"naming genes {k} (V1 workbook without {', '.join(NAMING_EXCLUDED)}) equal the approved list",
                       ", ".join(out[k]), set(out[k]) == set(exp), ", ".join(exp))
    return out


def workbook_markers(run: cc.Run) -> set[str]:
    wb = t6.read_dct_workbook(run.use_input(cc.INPUTS["dct"]["markers"], "DCT V1 marker workbook (Supp Table 1)"))
    return {g for _, _, genes in wb for g in genes}


def read_table(run: cc.Run, name: str, task: str = TASK, **kwargs) -> pd.DataFrame:
    path = cc.out_path(task, "tables", name, ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the step that writes it first.")
    return pd.read_csv(run.use_input(path, f"{task} output"), **kwargs)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return min(p, max(0.0, c - h)), max(p, min(1.0, c + h))


def zscore_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """z-score of each column across the rows (ddof 0; zero SD -> 0)."""
    sd = frame.std(axis=0, ddof=0)
    return frame.subtract(frame.mean(axis=0), axis=1).divide(sd.where(sd > 0), axis=1).fillna(0.0)


# ---------------------------------------------------------------------------
# Figures (adapted from kidney_rna_paired_figures.py: Sheet, text checks, save)
# ---------------------------------------------------------------------------
W = 180.0
M = 3.0
FONT = 7.0
TITLE = 7.5


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Arial", "font.size": FONT, "axes.titlesize": TITLE, "axes.labelsize": FONT,
        "xtick.labelsize": FONT, "ytick.labelsize": FONT, "legend.fontsize": FONT, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.edgecolor": INK["muted"], "axes.labelcolor": INK["secondary"],
        "text.color": INK["primary"], "xtick.color": INK["secondary"], "ytick.color": INK["secondary"],
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.titlelocation": "left", "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
    })
    return plt


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("t08_div", DIVERGING_COLORS)


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("t08_seq", ["#fcfcfb", *BLUE_RAMP])


def text_mm(s: str, size: float = FONT, weight: str = "normal") -> tuple[float, float]:
    import matplotlib.pyplot as plt

    fig = plt.figure(dpi=600)
    t = fig.text(0, 0, s, fontsize=size, fontweight=weight, linespacing=1.15)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    return bb.width / 600 * 25.4, bb.height / 600 * 25.4


def wrap_mm(text: str, width: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and text_mm(trial)[0] > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def note_block(note: str, width: float = W) -> tuple[str, float]:
    text = "\n".join(wrap_mm(note, width - 2 * M))
    return text, text_mm(text)[1] + 1.5


class Sheet:
    """Figure of exact size W x H mm with helpers that place axes and text in mm (y downwards)."""

    def __init__(self, height: float, width: float = W):
        import matplotlib.pyplot as plt

        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=600)

    def ax(self, x, y, w, h):
        return self.fig.add_axes((x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H))

    def text(self, x, y, s, **kw):
        kw.setdefault("fontsize", FONT)
        kw.setdefault("color", INK["primary"])
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def note(self, text: str, note_h: float) -> None:
        self.text(M, self.H - M - note_h + 1.5, text, va="top", linespacing=1.15, color=INK["secondary"])

    def rect(self, x, y, w, h, **kw):
        from matplotlib.patches import Rectangle

        self.fig.add_artist(Rectangle((x / self.W, 1 - (y + h) / self.H), w / self.W, h / self.H, **kw))


def text_checks(fig) -> dict:
    import matplotlib
    from matplotlib import font_manager

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    boxes, sizes, fonts, outside = [], [], set(), []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(r)
        if bb.width == 0:
            continue
        boxes.append((t.get_text(), bb))
        sizes.append(t.get_fontsize())
        fonts.add(Path(font_manager.findfont(t.get_fontproperties())).name.lower())
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            outside.append(t.get_text()[:30])
    shrink = 0.05 / 25.4 * fig.dpi
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > shrink and min(a.y1, b.y1) - max(a.y0, b.y0) > shrink:
                overlaps.append(f"{boxes[i][0]!r}/{boxes[j][0]!r}")
    return {"n_texts": len(boxes), "min_font_pt": min(sizes) if sizes else np.nan, "fonts": ";".join(sorted(fonts)),
            "n_text_overlaps": len(overlaps), "overlaps": "; ".join(overlaps[:8]),
            "n_text_outside_figure": len(outside), "outside": "; ".join(outside[:8])}


def file_checks(paths, width_mm: float) -> dict:
    import re
    from PIL import Image

    out = {}
    for p in paths:
        if p.suffix == ".png":
            with Image.open(p) as im:
                out.update(png_px=f"{im.size[0]}x{im.size[1]}", png_width_mm=round(im.size[0] / 600 * 25.4, 2))
        elif p.suffix == ".tif":
            with Image.open(p) as im:
                out.update(tif_mode=im.mode, tif_compression=im.info.get("compression", ""))
        elif p.suffix == ".pdf":
            raw = p.read_bytes()
            m = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
            out.update(pdf_width_mm=round(float(m.group(1)) / (72 / 25.4), 2) if m else np.nan,
                       pdf_truetype_embedded=b"/FontFile2" in raw, pdf_arial=b"Arial" in raw)
    png_w = int(out.get("png_px", "0x0").split("x")[0])
    out["width_ok"] = bool(abs(png_w - round(width_mm / 25.4 * 600)) <= 1
                           and abs(out.get("pdf_width_mm", 0) - width_mm) < 0.1)
    return out


def save_checked(run: cc.Run, sheet: Sheet, name: str, desc: str, rows: list[dict],
                 formats: tuple[str, ...] = (".pdf", ".png")) -> bool:
    """Save at exact size (PDF + 600-dpi PNG) after the kidney_rna text checks; appends a check row."""
    import matplotlib.pyplot as plt

    tc = text_checks(sheet.fig)
    paths = run.figure(sheet.fig, name, desc, formats=formats, tight=False)
    fc = file_checks(paths, sheet.W)
    arial = all(f in ("arial.ttf", "arialbd.ttf") for f in tc["fonts"].split(";"))
    ok = bool(tc["min_font_pt"] >= 7 and arial and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
              and fc["width_ok"] and fc.get("pdf_truetype_embedded") and fc.get("pdf_arial"))
    rows.append({"figure": name, "width_mm": sheet.W, "height_mm": round(sheet.H, 1), **tc, **fc, "pass": ok})
    print(f"  {name}: {sheet.W:.0f} x {sheet.H:.1f} mm, min font {tc['min_font_pt']}, overlaps "
          f"{tc['n_text_overlaps']} {tc['overlaps'][:120]!r}, outside {tc['n_text_outside_figure']} "
          f"{tc['outside'][:80]!r}, pass {ok}", flush=True)
    plt.close(sheet.fig)
    return ok


def style_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, width=0.5, pad=1.5, labelsize=FONT)


def trim_ticks(ax) -> None:
    """Explicit ticks inside the current limits (out-of-range tick labels count in the text checks, as in kidney_rna)."""
    lo, hi = sorted(ax.get_xlim())
    ax.set_xticks([t for t in ax.get_xticks() if lo - 1e-9 <= t <= hi + 1e-9])
    lo, hi = sorted(ax.get_ylim())
    ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 1e-9])
