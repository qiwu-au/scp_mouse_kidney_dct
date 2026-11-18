"""pt_tal shared definitions and functions. Runs no analysis by itself.

Imported by the ``pt_tal_*.py`` step scripts.
Plan: ``the analysis plan`` (Editors 4: known PT and TAL heterogeneity).

Decisions by the authors (15 Oct 2026)
-----------------------------------
- RNA reference level: atlas ``Celltype`` (primary; harmonized across the eight atlas studies). The plan named
  ``Celltype_finest``, whose PT / TAL labels are study-specific (e.g. PTS3T2 53 cells from one study; CTAL / MTAL
  only Kirita20 nuclei); it is reported in the step-1 crosstab only.
- Literature candidates: only the examples listed in the plan (no web lookup); Limbutara et al. 2020 not added.
- Unsupervised check: one subset per segment, the union of the cells carrying a segment label in either label set,
  compared against both label sets.

Contents
--------
- Paths and settings; SCP label sets (``kidney_t15_label`` primary, ``kidney_pre_label``); segments and subsets.
- Atlas classes and the RNA contrasts (one-vs-rest within PT and within TAL; S1/S2 vs S3; MD descriptive).
- ``load_scp``: kidney_object canonical kidney object with the kidney_annotation labels (via ``kidney_rna_lib.load_kidney``), canonical
  ``.X``, ln(1 + intensity), detection.
- ``subset_embedding``: PCA / kNN / UMAP of one SCP subset (optionally depth-regressed).
- ``effect_sizes``, ``ols_group``: group comparisons with and without ln(protein IDs).
- Figure helpers are taken from ``atlas_dct_lib`` (Sheet, text and file checks, save_checked).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse, stats

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as L7
import atlas_dct_lib as T8

TASK = "pt_tal"
SEED = cc.RANDOM_SEED
DATASET = "kidney"

# SCP labels -------------------------------------------------------------------------------------------
LABELS = ("kidney_t15_label", "kidney_pre_label")
PRIMARY = "kidney_t15_label"
LABEL_SHORT = {"kidney_t15_label": "t15", "kidney_pre_label": "pre"}
LABEL_DISPLAY = {"kidney_t15_label": "kidney_annotation labels", "kidney_pre_label": "Fig 2A labels"}
PT_LABELS = ("PCT", "PST")
TAL_LABEL = "TAL"
MD_LABEL = "MD"                                  # Fig 2A only (kidney_annotation: Endothelial)
SEGMENT_LABELS = {"PT": PT_LABELS, "TAL": (TAL_LABEL,)}
SEGMENTS = ("PT", "TAL")
CONSENSUS = "consensus"                          # cells with the same PT label in both sets
EXPECTED_UNION = {"PT": 356, "TAL": 154}         # union subsets (plan, 15 Oct 2026)
EXPECTED_MD_PRE = 19

# Atlas ------------------------------------------------------------------------------------------------
ATLAS = L7.ATLAS
LEVEL = "Celltype"
PT_CLASSES = ("PTS1", "PTS2", "PTS3", "PTS3T2")
TAL_CLASSES = ("CTAL", "MTAL")
MD_CLASS = "MD"
REGION_CLASSES = (*PT_CLASSES, *TAL_CLASSES, MD_CLASS)
SUMMARY_ONLY = ("LOH",)                          # mixed loop-of-Henle class, cells only (as kidney_rna / atlas_dct)
SEGMENT_CLASSES = {"PT": PT_CLASSES, "TAL": TAL_CLASSES}
BATCH_KEYS = ("Origin", "Source")
RNA_SOURCES = {"all": None, "nuclei": "Nuclei", "cells": "Cell"}
SOURCE_TEXT = {"all": "all atlas cells and nuclei", "nuclei": "atlas nuclei (snRNA-seq)",
               "cells": "atlas cells (scRNA-seq)"}

# RNA contrasts: name -> (segment, group classes, reference classes, kind)
CONTRASTS = {
    "PTS1": ("PT", ("PTS1",), ("PTS2", "PTS3", "PTS3T2"), "one_vs_rest"),
    "PTS2": ("PT", ("PTS2",), ("PTS1", "PTS3", "PTS3T2"), "one_vs_rest"),
    "PTS3": ("PT", ("PTS3",), ("PTS1", "PTS2", "PTS3T2"), "one_vs_rest"),
    "PTS3T2": ("PT", ("PTS3T2",), ("PTS1", "PTS2", "PTS3"), "one_vs_rest"),
    "S1/S2": ("PT", ("PTS1", "PTS2"), ("PTS3", "PTS3T2"), "grouped"),
    "S3": ("PT", ("PTS3", "PTS3T2"), ("PTS1", "PTS2"), "grouped"),
    "CTAL": ("TAL", ("CTAL",), ("MTAL",), "one_vs_rest"),
    "MTAL": ("TAL", ("MTAL",), ("CTAL",), "one_vs_rest"),
    "MD": ("TAL", ("MD",), ("CTAL", "MTAL"), "descriptive"),
}
CONTRAST_SAFE = {c: c.replace("/", "") for c in CONTRASTS}   # file-name safe (S1/S2 -> S1S2)
PT_ONE_VS_REST = ("PTS1", "PTS2", "PTS3", "PTS3T2")
PT_GROUPED = ("S1/S2", "S3")
TAL_PANELS = ("CTAL", "MTAL")

# Filters (approved plan; as atlas_dct step 4) ---------------------------------------------------------------
Q_MAX = 0.05
LFC_MIN = 1.0                                    # scanpy log2 fold change
PCT_MIN = 0.25                                   # detected (counts > 0) in >= 25% of the group's cells
BATCH_MIN_CELLS = 20                             # cells of both sides in an Origin x Source batch
BATCH_AGREE = 2 / 3                              # fraction of eligible batches with the overall sign
SCP_DET_MIN = 0.10                               # protein detected in >= 10% of the SCP segment cells
PANEL_TOP = 30

# Mapping (kidney_rna / atlas_dct thresholds) -----------------------------------------------------------------------
SCORE_MIN, MARGIN_MIN = 0.20, 0.05

# Subset embedding and subclustering -------------------------------------------------------------------
N_PCS = 20
N_NEIGHBORS = 10
SUB_RESOLUTIONS = (0.2, 0.3, 0.5)
SUB_SEEDS = tuple(range(10))
MIN_CLUSTER = 5
BELOW_MIN = "below_min_size"
GRAPHS = ("primary", "regressed")
GRAPH_TEXT = {"primary": "ln(1 + intensity), scaled within the subset",
              "regressed": "ln(1 + intensity) regressed on ln(protein IDs), scaled within the subset"}

# Literature candidates: exactly the examples in the plan (authors, 15 Oct 2026). (gene, plan group, contrast whose
# direction is checked in the atlas; None = the plan gives no CTAL / MTAL direction).
LITERATURE = (
    ("Slc5a2", "S1", "PTS1"), ("Slc5a12", "S1", "PTS1"),
    ("Slc22a6", "S1/S2", "S1/S2"), ("Slc22a8", "S1/S2", "S1/S2"),
    ("Slc7a13", "S3", "S3"), ("Atp11a", "S3", "S3"), ("Slc22a7", "S3", "S3"),
    ("Cldn10", "TAL: Cldn10", None), ("Cldn16", "TAL: Cldn16/Cldn19", None), ("Cldn19", "TAL: Cldn16/Cldn19", None),
    ("Nos1", "MD", "MD"), ("Ptgs2", "MD", "MD"),
)

# Colours ----------------------------------------------------------------------------------------------
INK = T8.INK
SERIES = T8.SERIES
RNA_PALETTE = {"PTS1": "#2a78d6", "PTS2": "#6da7ec", "PTS3": "#1baf7a", "PTS3T2": "#008300",
               "CTAL": "#eda100", "MTAL": "#b07800", "MD": "#e34948", "LOH": "#898781"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def table_path(name: str):
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, task: str = TASK, **kwargs) -> pd.DataFrame:
    path = cc.out_path(task, "tables", name, ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the step that writes it first.")
    return pd.read_csv(run.use_input(path, f"{task} output"), **kwargs)


# ---------------------------------------------------------------------------
# SCP
# ---------------------------------------------------------------------------
def load_scp(run: cc.Run, checks: t5.Checks | None = None):
    """Canonical kidney object and labels with the kidney_annotation labels; canonical .X, ln(1 + intensity), detection."""
    adata, labels = L7.load_kidney(run, checks)
    names = pd.Index(adata.var_names.astype(str))
    xz = pd.DataFrame(cc.dense(adata.X).astype(np.float64), index=labels.index, columns=names)
    ln1p = pd.DataFrame(cc.dense(adata.layers["ln1p_intensity"]).astype(np.float64), index=labels.index,
                        columns=names)
    det = pd.DataFrame(cc.dense(adata.layers["intensity"]) > 0, index=labels.index, columns=names)
    return adata, labels, xz, ln1p, det


def union_masks(labels: pd.DataFrame) -> dict[str, pd.Series]:
    """Union subsets (a segment label in either label set) and the Fig 2A MD cells."""
    out = {}
    for seg, labs in SEGMENT_LABELS.items():
        out[seg] = labels[LABELS[0]].isin(labs) | labels[LABELS[1]].isin(labs)
    out["MD_pre"] = labels["kidney_pre_label"].eq(MD_LABEL)
    return out


def segment_cells(labels: pd.DataFrame, col: str, seg: str) -> pd.Series:
    return labels[col].isin(SEGMENT_LABELS[seg])


def pt_groupings(labels: pd.DataFrame) -> dict[str, pd.Series]:
    """PCT / PST groupings: each label set, and the consensus (same PT label in both sets); others NaN."""
    out = {}
    for col in LABELS:
        out[LABEL_SHORT[col]] = labels[col].where(labels[col].isin(PT_LABELS))
    same = labels[LABELS[0]].eq(labels[LABELS[1]]) & labels[LABELS[0]].isin(PT_LABELS)
    out[CONSENSUS] = labels[LABELS[0]].where(same)
    return out


GROUPING_TEXT = {"t15": "kidney_annotation labels", "pre": "Fig 2A labels", CONSENSUS: "same PT label in both sets"}
GROUPING_SHORT = {"t15": "kidney_annotation", "pre": "Fig 2A", CONSENSUS: "both"}


def order_labels(values) -> list[str]:
    """Plot / legend order: PCT, PST, TAL, MD, other labels alphabetically, Unknown last."""
    present = list(pd.Series(values).astype(str).value_counts().index)
    first = [k for k in (*PT_LABELS, TAL_LABEL, MD_LABEL) if k in present]
    rest = sorted(k for k in present if k not in first and k != "Unknown")
    return first + rest + (["Unknown"] if "Unknown" in present else [])


def label_colors(adata) -> dict[str, str]:
    """Fig 2 colours (stored in the canonical object) for the labels used here; distinct fallbacks."""
    colors = {}
    for col in ("kidney_pre_label", "kidney_reannot_label"):
        cats = list(map(str, adata.obs[col].cat.categories))
        colors.update(dict(zip(cats, adata.uns[f"{col}_colors"])))
    colors["Unknown"] = cc.UNKNOWN_COLOR
    used = {c.lower() for k, c in colors.items() if k in ("PCT", "TAL", "MD", "Unknown")}
    pst = colors.get("PST", "").lower()
    grey = len(pst) == 7 and pst[1:3] == pst[3:5] == pst[5:7]
    if pst in used or grey:          # the Fig 2 PST colour is grey (#7f7f7f), close to Unknown
        colors["PST"] = "#1baf7a"
    for k in ("Endothelial", "JG", "DCT1", "PC", "IC-B", "DVR", "DCT2", "CNT"):
        colors.setdefault(k, INK["muted"])
    return colors


def v3_marker_genes(run: cc.Run) -> dict[str, list[str]]:
    """Marker gene -> V3 (+ kidney_annotation endothelial row) types listing it."""
    return L7.markers_by_gene(L7.kidney_marker_rows(run))


# Ribosomal / MitoCarta flags (rule of kidney_annotation_atlas_reference.py; flags only, used for a scoring sensitivity).
RIBOSOMAL = r"^(Rp[ls]\d|Rplp\d|Rpsa$)"
QC_MITO_ANNOTATION = cc.OUT_ROOTS["tables"] / "qc_mito" / "qc_mito_kidney_protein_annotation.csv"


def protein_flags(run: cc.Run, proteins) -> pd.DataFrame:
    """Per protein: ribosomal (symbol rule) and MitoCarta3.0 (qc_mito kidney protein annotation)."""
    ann = pd.read_csv(run.use_input(QC_MITO_ANNOTATION, "qc_mito MitoCarta3.0 protein annotation (kidney)"))
    mito = ann.set_index("protein")["mitocarta"].astype(bool)
    p = pd.Series(list(proteins), dtype=str)
    return pd.DataFrame({"protein": p, "ribosomal": p.str.match(RIBOSOMAL).to_numpy(),
                         "mitocarta": p.map(mito).fillna(False).astype(bool).to_numpy()})


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def subset_embedding(ln1p_sub: pd.DataFrame, ids: np.ndarray, regressed: bool, seed: int = SEED):
    """AnnData of one SCP subset with PCA (arpack, 20), kNN (k = 10) and UMAP (seed 42).

    Proteins detected in >= 2 subset cells (original filter); ln(1 + intensity), optionally regressed on
    ln(protein IDs) per protein (residuals, as kidney_annotation graph (b)); scaled within the subset (z, clip 10).
    """
    import anndata as ad
    import scanpy as sc

    x = ln1p_sub.to_numpy(np.float64)
    keep = (x > 0).sum(axis=0) >= cc.MIN_PROTEIN_CELLS
    x = x[:, keep]
    if regressed:
        design = np.column_stack([np.ones(len(x)), np.log(np.asarray(ids, float))])
        beta, *_ = np.linalg.lstsq(design, x, rcond=None)
        x = x - design @ beta
    xs = cc.scale_like_scanpy(x).astype(np.float32)
    a = ad.AnnData(X=xs, obs=pd.DataFrame(index=ln1p_sub.index.astype(str)),
                   var=pd.DataFrame(index=ln1p_sub.columns[keep].astype(str)))
    sc.tl.pca(a, n_comps=N_PCS, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a


# igraph's random-number bridge raises an ignored 'high is out of bounds for int32' ValueError for many draws on
# this Windows setup (qc_mito README); they are counted instead of printed, as in qc_mito_sensitivity.py.
IGNORED = {"igraph_int32_randint": 0}


def _unraisable(hook_args) -> None:
    import sys

    if isinstance(hook_args.exc_value, ValueError) and "out of bounds for int32" in str(hook_args.exc_value):
        IGNORED["igraph_int32_randint"] += 1
    else:
        sys.__unraisablehook__(hook_args)


def install_igraph_hook() -> None:
    import sys

    sys.unraisablehook = _unraisable


def eta_squared(values, groups) -> float:
    v = np.asarray(values, float)
    g = np.asarray(groups).astype(str)
    grand = v.mean()
    ss_t = float(((v - grand) ** 2).sum())
    ss_b = float(sum(((v[g == k].mean() - grand) ** 2) * (g == k).sum() for k in np.unique(g)))
    return ss_b / ss_t if ss_t > 0 else np.nan


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def effect_sizes(x, y, n_boot: int = 0, seed: int = SEED) -> dict[str, float]:
    """x vs y: AUC = P(x > y) + 0.5 P(tie), Cliff's delta, Hedges' g, Mann-Whitney p; optional bootstrap CI of AUC."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    out = {"n_1": len(x), "n_2": len(y), "median_1": np.nan, "median_2": np.nan, "mean_1": np.nan, "mean_2": np.nan,
           "auc": np.nan, "cliffs_delta": np.nan, "hedges_g": np.nan, "p_mannwhitney": np.nan,
           "auc_ci_low": np.nan, "auc_ci_high": np.nan}
    if len(x) < 2 or len(y) < 2:
        return out
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    auc = float(u.statistic / (len(x) * len(y)))
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    d = (x.mean() - y.mean()) / sp if sp > 0 else np.nan
    j = 1 - 3 / (4 * (len(x) + len(y)) - 9)
    out.update(median_1=float(np.median(x)), median_2=float(np.median(y)), mean_1=float(x.mean()),
               mean_2=float(y.mean()), auc=auc, cliffs_delta=2 * auc - 1, hedges_g=float(d * j),
               p_mannwhitney=float(u.pvalue))
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            xb, yb = rng.choice(x, len(x)), rng.choice(y, len(y))
            r = stats.rankdata(np.concatenate([xb, yb]))
            boots[b] = (r[:len(xb)].sum() - len(xb) * (len(xb) + 1) / 2) / (len(xb) * len(yb))
        out.update(auc_ci_low=float(np.quantile(boots, 0.025)), auc_ci_high=float(np.quantile(boots, 0.975)))
    return out


def ols_group(score, is_group, ln_ids) -> dict[str, float]:
    """OLS score ~ group + ln(protein IDs); group coefficient in units of the score SD (and raw), SE, p."""
    import statsmodels.api as sm

    s = np.asarray(score, float)
    g = np.asarray(is_group, float)
    z = np.asarray(ln_ids, float)
    ok = np.isfinite(s) & np.isfinite(g) & np.isfinite(z)
    s, g, z = s[ok], g[ok], z[ok]
    sd = s.std(ddof=1)
    out = {"n_ols": int(ok.sum())}
    for name, design in (("unadjusted", np.column_stack([np.ones_like(g), g])),
                         ("adjusted_ln_ids", np.column_stack([np.ones_like(g), g, z]))):
        fit = sm.OLS(s, design).fit()
        out[f"coef_{name}"] = float(fit.params[1])
        out[f"coef_sd_{name}"] = float(fit.params[1] / sd) if sd > 0 else np.nan
        ci = fit.conf_int()[1]
        out[f"ci_sd_low_{name}"] = float(ci[0] / sd) if sd > 0 else np.nan
        out[f"ci_sd_high_{name}"] = float(ci[1] / sd) if sd > 0 else np.nan
        out[f"p_{name}"] = float(fit.pvalues[1])
        if name == "adjusted_ln_ids":
            out["coef_ln_ids"] = float(fit.params[2])
            out["p_ln_ids"] = float(fit.pvalues[2])
            out["r2_adjusted_model"] = float(fit.rsquared)
    return out


def min_detectable_auc(n1: int, n2: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Noether's approximation for the Mann-Whitney test: AUC - 0.5 = (z_a/2 + z_b) / sqrt(12 c (1 - c) N)."""
    if min(n1, n2) < 2:
        return np.nan
    n = n1 + n2
    c = n1 / n
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float(min(1.0, 0.5 + (za + zb) / np.sqrt(12 * c * (1 - c) * n)))


def min_detectable_g(n1: int, n2: int, alpha: float = 0.05, power: float = 0.80) -> float:
    if min(n1, n2) < 2:
        return np.nan
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    return float((za + zb) * np.sqrt(1 / n1 + 1 / n2))


# ---------------------------------------------------------------------------
# Atlas helpers
# ---------------------------------------------------------------------------
def lognorm_with_totals(counts: sparse.csr_matrix, totals: np.ndarray) -> sparse.csr_matrix:
    """counts / total x 10,000 (totals over all 16,711 genes), natural log1p; float32 CSR (as kidney_rna / atlas_dct)."""
    m = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    scale = (L7.TARGET_SUM / np.where(totals > 0, totals, 1)).astype(np.float32)
    m.data *= np.repeat(scale, np.diff(m.indptr))
    np.log1p(m.data, out=m.data)
    return m


# ---------------------------------------------------------------------------
# Figure helpers (mm layout on atlas_dct_lib.Sheet; all text 7 pt)
# ---------------------------------------------------------------------------
LH = None          # line height (mm), set by setup()


def setup():
    """Matplotlib with the atlas_dct style (Arial 7 pt); returns pyplot."""
    global LH
    plt = T8.setup_matplotlib()
    LH = T8.text_mm("Hg")[1]
    return plt


def legend_rows(entries: list[tuple[str, str]], width: float) -> list[list[tuple[str, str]]]:
    """Greedy packing of (text, colour) legend entries into lines of at most ``width`` mm."""
    lines, cur, cur_w = [], [], 0.0
    for text, col in entries:
        w = 1.6 + 0.8 + T8.text_mm(text)[0] + 2.0
        if cur and cur_w + w > width:
            lines.append(cur)
            cur, cur_w = [], 0.0
        cur.append((text, col))
        cur_w += w
    return lines + ([cur] if cur else [])


def draw_legend(sh, x: float, y: float, lines: list[list[tuple[str, str]]]) -> float:
    """Chip + text legend lines starting at (x, y) (top); returns the bottom y."""
    for i, line in enumerate(lines):
        yc = y + (i + 0.5) * (LH + 0.4)
        xx = x
        for text, col in line:
            sh.rect(xx, yc - 0.8, 1.6, 1.6, facecolor=col, edgecolor="none")
            sh.text(xx + 1.6 + 0.8, yc, text, va="center_baseline")
            xx += 1.6 + 0.8 + T8.text_mm(text)[0] + 2.0
    return y + len(lines) * (LH + 0.4)


def umap_categorical(sh, x, y, w, h, xy, lab, order, colors, title, point=2.2) -> float:
    """Title, UMAP scatter coloured by label (first entries of ``order`` drawn last = on top), legend with n.
    Returns the bottom y of the legend."""
    lab = np.asarray(lab).astype(str)
    sh.text(x, y, title, va="top")
    ax = sh.ax(x, y + LH + 0.8, w, h)
    for k in reversed(order):
        m = lab == k
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=point, color=colors.get(k, INK["muted"]), linewidths=0,
                       rasterized=True, zorder=1 if k == "Unknown" else 2)
    ax.set_axis_off()
    entries = [(f"{k} ({int((lab == k).sum())})", colors.get(k, INK["muted"])) for k in order if (lab == k).any()]
    return draw_legend(sh, x, y + LH + 0.8 + h + 0.8, legend_rows(entries, w))


def umap_continuous(sh, x, y, w, h, xy, values, title, cmap, vmin=None, vmax=None, point=2.2, nticks=3) -> float:
    """Title, UMAP scatter coloured by a continuous value (high values on top), horizontal colour bar below."""
    v = np.asarray(values, float)
    vmin = float(np.nanquantile(v, 0.02)) if vmin is None else vmin
    vmax = float(np.nanquantile(v, 0.98)) if vmax is None else vmax
    sh.text(x, y, title, va="top")
    ax = sh.ax(x, y + LH + 0.8, w, h)
    o = np.argsort(np.where(np.isfinite(v), v, -np.inf))
    ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=point, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_axis_off()
    by = y + LH + 0.8 + h + 1.0
    bw = min(w * 0.7, 30.0)
    cax = sh.ax(x + 1.0, by, bw, 1.8)
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_xticks([])
    cax.set_yticks([])
    for s_ in cax.spines.values():
        s_.set_linewidth(0.4)
        s_.set_color(INK["muted"])
    ticks = np.linspace(vmin, vmax, nticks)
    for t in ticks:
        xt = x + 1.0 + (t - vmin) / (vmax - vmin) * bw if vmax > vmin else x + 1.0
        sh.text(xt, by + 1.8 + 0.5, f"{t:.2g}", ha="center", va="top")
    return by + 1.8 + 0.5 + LH


def count_heatmap(sh, x0, y0, counts: np.ndarray, frac: np.ndarray, row_labels, col_labels, cw, rh, cmap,
                  row_colors=None) -> tuple[float, float]:
    """Counts as text on a 0-1 fraction fill; row labels left of x0 (with optional chips), column labels below
    (rotated). Returns (right x, bottom y of the column labels)."""
    nr, nc = counts.shape
    ax = sh.ax(x0, y0, nc * cw, nr * rh)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest", extent=(0, nc, nr, 0))
    ax.set_axis_off()
    for a in range(nr):
        for b in range(nc):
            if counts[a, b] > 0:
                sh.text(x0 + (b + 0.5) * cw, y0 + (a + 0.5) * rh, f"{int(counts[a, b])}", ha="center",
                        va="center_baseline", color="white" if frac[a, b] > 0.55 else INK["primary"])
    for a, lab in enumerate(row_labels):
        yc = y0 + (a + 0.5) * rh
        xr = x0 - 1.0
        if row_colors is not None:
            sh.rect(xr - 1.6, yc - 0.8, 1.6, 1.6, facecolor=row_colors[a], edgecolor="none")
            xr -= 1.6 + 0.8
        sh.text(xr, yc, lab, ha="right", va="center_baseline")
    bottom = y0 + nr * rh + 0.8
    for b, lab in enumerate(col_labels):
        sh.text(x0 + (b + 0.5) * cw, bottom, lab, rotation=90, ha="center", va="top")
    return x0 + nc * cw, bottom + max(T8.text_mm(c)[0] for c in col_labels)


def colorbar_vertical(sh, x, y, h, cmap, label: str, ticks=(0, 0.5, 1)) -> None:
    cax = sh.ax(x, y, 2.4, h)
    cax.imshow(np.linspace(1, 0, 256)[:, None], cmap=cmap, vmin=0, vmax=1, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks(list(ticks))
    cax.tick_params(length=1.5, width=0.4, labelsize=T8.FONT, pad=1)
    sh.text(x + 2.4 + 6.5, y + h / 2, label, rotation=90, ha="left", va="center")
