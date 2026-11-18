"""reviewer_core shared definitions and functions. Runs no analysis by itself.

Imported by the
``reviewer_core_*.py`` step scripts.

Definitions were copied (not imported) so that later edits to the source
scripts cannot change reviewer_core results. Sources (read-only; SHA256 recorded by the
step scripts through ``record_reference_code``):
  - Task 3 panels, ``_score``, ``_threshold``, ``_summary_by``:
    scripts/03_dct_annotation_negative_markers.py (lines 64-89, 100-133, 190-207).
  - 18-protein cell-cycle panel, ALDO signature lists, N_NEIGHBORS / N_PCS:
    scripts/00_config.py (lines 78-82, 112-131).
  - ``_recluster`` (scale, PCA, neighbours, Leiden, UMAP):
    scripts/04_cell_cycle_regression.py (lines 130-142).
  - ALDO score families, variants and z-scoring:
    scripts/05_aldo_score_sensitivity.py (lines 42-65, 91-106).
  - Identity map for the identity x cycling re-annotation:
    scripts/10_reviewer_dct_reannotation.py (lines 27-31).
  - Tirosh S/G2M lists, kNN purity, ProLIF concentration, label contrasts:
    scripts/DCT_v1/03_task4_cell_cycle.py and scripts/DCT_v1/_v1_common.py
    (earlier revision V1 rerun).
  - V1 marker scoring and cluster naming (incl. the 'Unknown' rule):
    scp_dct_upload.py (lines 115-171).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc

TASK = "reviewer_core"
SEED = cc.RANDOM_SEED
N_NEIGHBORS = 10
N_PCS = 20
LABEL_COLUMNS = ("v1_cell_label", "v1_cluster_label")
LABEL_DISPLAY = {"v1_cell_label": "V1 per-cell label", "v1_cluster_label": "V1 cluster label"}
CORE_LABELS = ("DCT1", "DCT2", "ProLIF")
RESOLUTIONS = (0.3, 0.5, 0.8, 1.0, 3.0)
ORIGINAL_RESOLUTION = 3.0

REFERENCE_CODE = {
    "scp_dct_upload.py": "DCT pipeline of the submission (V1 marker scoring and cluster naming)",
    "scripts/00_config.py": "the earlier analysis config (cell-cycle panel, ALDO lists, N_NEIGHBORS/N_PCS)",
    "scripts/03_dct_annotation_negative_markers.py": "the earlier analysis Task 3 panels and scoring helpers",
    "scripts/04_cell_cycle_regression.py": "the earlier analysis Task 4 recluster helpers",
    "scripts/05_aldo_score_sensitivity.py": "the earlier analysis Task 5 ALDO score families",
    "scripts/08_task3_task4_reviewer_figures.py": "the earlier analysis reviewer summary panel (docx Figure X)",
    "scripts/10_reviewer_dct_reannotation.py": "the earlier analysis identity x cycling re-annotation",
    "scripts/DCT_v1/_v1_common.py": "earlier revision V1 rerun (helpers)",
    "scripts/DCT_v1/02_task3_markers.py": "earlier revision V1 rerun (Task 3)",
    "scripts/DCT_v1/03_task4_cell_cycle.py": "earlier revision V1 rerun (Task 4)",
    "scripts/DCT_v1/04_aldo_scores.py": "earlier revision V1 rerun (ALDO)",
    "scripts/DCT_v1/05_reannotation.py": "earlier revision V1 rerun (re-annotation)",
    "scripts/DCT_v1/06_plate_stratification.py": "earlier revision V1 rerun (plate)",
    "scripts/DCT_v1/08_reviewer_panel_and_readme.py": "earlier revision V1 rerun (reviewer panel)",
}

# ---------------------------------------------------------------------------
# Marker definitions
# ---------------------------------------------------------------------------
# scripts/03_dct_annotation_negative_markers.py (Umod stays in the TAL panel).
TASK3_MARKER_PANELS = {
    "DCT_core": {"role": "positive", "genes": ["Slc12a3", "Pvalb"]},
    "DCT1_associated": {"role": "positive", "genes": ["Pvalb", "Egf"]},
    "DCT2_associated": {"role": "positive", "genes": ["Slc8a1", "Calb1", "Klk1"]},
    "TAL": {"role": "negative_contamination", "genes": ["Slc12a1", "Umod"]},
    "CNT_CD": {"role": "negative_contamination", "genes": ["Aqp2", "Scnn1g", "Aqp3", "Klk1"]},
    "Interstitial": {"role": "negative_contamination", "genes": ["Col1a1", "Col1a2", "Col5a2", "Dcn"]},
}
POSITIVE_PANELS = tuple(p for p, m in TASK3_MARKER_PANELS.items() if m["role"] == "positive")
NEGATIVE_PANELS = tuple(p for p, m in TASK3_MARKER_PANELS.items() if m["role"] == "negative_contamination")
FLAG_COLUMNS = ("tal_like_flag", "cnt_cd_like_flag", "interstitial_like_flag")

# scripts/00_config.py: original 18-protein cell-cycle panel.
CELL_CYCLE_MARKERS = [
    "Mki67", "Top2a", "Cenpf", "Pcna", "Mcm2", "Mcm3", "Mcm4", "Mcm5",
    "Mcm6", "Cdk1", "Ccnb1", "Ccnb2", "Cdc20", "Ube2c", "Birc5",
    "Tyms", "Pclaf", "Hmgb2",
]

# Tirosh et al., Science 2016 (Seurat cc.genes.updated.2019), mouse capitalization.
TIROSH_S = [
    "Mcm5", "Pcna", "Tyms", "Fen1", "Mcm2", "Mcm4", "Rrm1", "Ung", "Gins2", "Mcm6", "Cdca7", "Dtl", "Prim1",
    "Uhrf1", "Cenpu", "Hells", "Rfc2", "Rpa2", "Nasp", "Rad51ap1", "Gmnn", "Wdr76", "Slbp", "Ccne2", "Ubr7",
    "Pold3", "Msh2", "Atad2", "Rad51", "Rrm2", "Cdc45", "Cdc6", "Exo1", "Tipin", "Dscc1", "Blm", "Casp8ap2",
    "Usp1", "Clspn", "Pola1", "Chaf1b", "Brip1", "E2f8",
]
TIROSH_G2M = [
    "Hmgb2", "Cdk1", "Nusap1", "Ube2c", "Birc5", "Tpx2", "Top2a", "Ndc80", "Cks2", "Nuf2", "Cks1b", "Mki67",
    "Tmpo", "Cenpf", "Tacc3", "Pimreg", "Smc4", "Ccnb2", "Ckap2l", "Ckap2", "Aurkb", "Bub1", "Kif11", "Anp32e",
    "Tubb4b", "Gtse1", "Kif20b", "Hjurp", "Cdca3", "Jpt1", "Cdc20", "Ttk", "Cdc25c", "Kif2c", "Rangap1",
    "Ncapd2", "Dlgap5", "Cdca2", "Cdca8", "Ect2", "Kif23", "Hmmr", "Aurka", "Psrc1", "Anln", "Lbr", "Ckap5",
    "Cenpe", "Ctcf", "Nek2", "G2e3", "Gas2l3", "Cbx5", "Cenpa",
]
TIROSH_ALIASES = {"Cenpu": ["Mlf1ip"], "Pimreg": ["Fam64a"], "Jpt1": ["Hn1"]}
assert len(TIROSH_S) == 43 and len(TIROSH_G2M) == 54
TIROSH_ALL = [*TIROSH_S, *TIROSH_G2M, *[a for v in TIROSH_ALIASES.values() for a in v]]

# scripts/00_config.py ALDO lists and scripts/05_aldo_score_sensitivity.py families.
ALDO_SIGNATURE_UP = [
    "Klk1", "Ppargc1a", "Klf15", "Sgk1", "Usp2", "Scnn1a", "Ndrg2",
    "Per3", "Tsc22d3", "Zbtb16", "Mlph", "Atp1a1", "Fxyd2", "Fxyd4",
    "Cox17", "Cited4", "Arnt2", "Emx2", "Acss3", "Wnk1", "Fkbp5",
    "Nr3c2", "Slc6a15", "Slc9a4", "Slc25a35", "Pitrm1", "Tmprss2",
    "Tmprss4", "Cwh43", "Def8", "Iqgap2", "Iqsec2", "Myh14", "Nxpe3",
    "Optn", "Osbp2", "Rab27a", "Sec23b", "Sybu", "Tango2", "Tbc1d24",
    "Trak1", "Vopp1", "Yipf1",
]
ALDO_SIGNATURE_DOWN = [
    "Nedd4l", "Myc", "Egr4", "Nfil3", "Atp12a", "Slc1a3", "Slc7a1",
    "Gfra1", "Gdf15", "Wnt4", "Dusp5", "Dusp8", "Rnf39", "Pim1", "Pdk4",
]
ALDO_UP_WITH_SLC12A3 = list(dict.fromkeys([*ALDO_SIGNATURE_UP, "Slc12a3"]))
FIG3_ALDO_MACHINERY = [
    "Nr3c2", "Hsd11b2", "Wnk4", "Stk39",
    "Hsp90aa1", "Hsp90ab1", "Fkbp5",
    "Ncoa1", "Ncoa2", "Ncoa3",
    "Ppp2ca", "Ppp2r1a", "Ywhae", "Ywhab", "Ywhaz",
]
ALDO_FAMILIES = {
    "ALDO_activity": (ALDO_UP_WITH_SLC12A3, []),
    "ALDO_signed": (ALDO_UP_WITH_SLC12A3, list(ALDO_SIGNATURE_DOWN)),
    "ALDO_machinery": (FIG3_ALDO_MACHINERY, []),
}
ALDO_VARIANTS = {
    "base": set(),
    "no_Slc12a3": {"Slc12a3"},
    "no_cell_cycle": set(CELL_CYCLE_MARKERS),
    "no_Slc12a3_no_cell_cycle": {"Slc12a3", *CELL_CYCLE_MARKERS},
}
ALDO_SUBMISSION_SCORES = ("ALDO_signed", "Aldo_machinery", "ALDO_activity_noMach")
ALDO_PAIRS = [("DCT2", "DCT1"), ("ProLIF", "DCT1"), ("ProLIF", "DCT2")]

# scripts/10_reviewer_dct_reannotation.py
IDENTITY_MAP = {
    "DCT1_associated": "DCT1-associated",
    "DCT2_associated": "DCT2-associated",
    "DCT_core": "DCT-core/indeterminate",
}

# Editors' point 3 markers (scripts/DCT_v1/02_task3_markers.py), in order, with expected segment.
INDIVIDUAL_MARKERS = [
    ("Umod", "TAL"), ("Slc12a1", "TAL"),
    ("Pvalb", "DCT1"), ("Egf", "DCT1"),
    ("Slc12a3", "DCT1 and DCT2"),
    ("Calb1", "DCT2/CNT"), ("Slc8a1", "DCT2/CNT"), ("S100g", "DCT2/CNT"), ("Klk1", "DCT2/CNT"), ("Trpv5", "DCT2/CNT"),
    ("Mki67", "Cell cycle"), ("Top2a", "Cell cycle"),
]

# Corrected objects of section A.2 (name -> (gene set, correction)).
CC_BRANCHES = {
    "cc_removed": ("original_18_panel", "proteins removed"),
    "cc_regressed": ("original_18_panel", "cell_cycle_score regressed"),
    "tirosh_removed": ("tirosh_S_G2M_plus_original", "proteins removed"),
    "tirosh_regressed": ("tirosh_S_G2M", "S_score + G2M_score regressed"),
    "corr_removed": ("Mki67_Top2a_correlated (circular)", "proteins removed"),
    "corr_regressed": ("Mki67_Top2a_correlated (circular)", "cc_corr_score regressed"),
}

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "surface": "#fcfcfb", "background_cells": "#d6d5cf"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Submitted Fig 3 label colours, as stored in the V1 object (kidney_object canonical uns).
LABEL_PALETTE = {"DCT1": "#1f77b4", "DCT2": "#ff7f0e", "ProLIF": "#2ca02c", "Unknown": "#B3B3B3"}


def apply_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "axes.edgecolor": INK["muted"], "axes.labelcolor": INK["secondary"], "text.color": INK["primary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
        "axes.titlelocation": "left",
    })


def footer(fig, script: str, note: str = "") -> None:
    text = f"{cc.GENERATED_TAG} · scripts/{script}" + (f" · {note}" if note else "")
    fig.text(0.005, 0.002, text, ha="left", va="bottom", fontsize=4.8, color=INK["muted"])


def label_colors(labels) -> dict[str, str]:
    return {label: LABEL_PALETTE.get(label, INK["muted"]) for label in labels}


def clean_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # min/max guard against floating-point rounding at p = 0 or p = 1.
    return min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Inputs and outputs of this task
# ---------------------------------------------------------------------------
def record_reference_code(run: cc.Run, names=None) -> None:
    for name in (names or REFERENCE_CODE):
        path = cc.PROJECT_ROOT / name
        if path.exists():
            run.use_input(path, f"reference code: {REFERENCE_CODE[name]}")


def load_dct(run: cc.Run):
    """kidney_object canonical DCT object and labels (inputs recorded); adds the stored Leiden 3.0 IDs to labels."""
    for path, label in ((cc.CANONICAL["dct"]["labels"], "kidney_object canonical DCT labels"),
                        (cc.CANONICAL["dct"]["h5ad"], "kidney_object canonical DCT object")):
        run.use_input(path, label)
    adata, labels = cc.load_canonical("dct")
    labels = labels.copy()
    labels["v1_leiden_res3_cluster"] = adata.obs["v1_leiden_res3_cluster"].astype(str).to_numpy()
    return adata, labels


def h5ad_path(name: str) -> Path:
    return cc.out_path(TASK, "h5ad", name, ".h5ad")


def table_path(name: str, subdir: str = "") -> Path:
    return cc.out_path(TASK, "tables", name, ".csv", subdir)


def read_table(run: cc.Run, name: str, subdir: str = "", **kwargs) -> pd.DataFrame:
    path = table_path(name, subdir)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the reviewer_core step that writes it first.")
    return pd.read_csv(run.use_input(path, "reviewer_core output"), **kwargs)


def read_h5ad(run: cc.Run, name: str):
    import anndata as ad

    path = h5ad_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the reviewer_core step that writes it first.")
    return ad.read_h5ad(run.use_input(path, "reviewer_core AnnData"))


def write_h5ad_light(run: cc.Run, adata, name: str) -> Path:
    """Write an AnnData without the intensity layers (X, obs, var, obsm, obsp, uns are kept)."""
    out = adata.copy()
    for key in [k for k in list(out.layers.keys()) if k is not None]:
        del out.layers[key]
    out.raw = None
    return run.h5ad(out, name)


def ln1p_frame(adata) -> pd.DataFrame:
    """ln(1 + intensity) in float64 from the kidney_object intensity layer (detection = intensity > 0)."""
    values = np.log1p(cc.dense(adata.layers["intensity"]).astype(np.float64))
    return pd.DataFrame(values, index=pd.Index(adata.obs_names.astype(str), name="cell_id"),
                        columns=adata.var_names.astype(str))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(adata, genes: list[str], name: str) -> tuple[list[str], list[str]]:
    """Task 3 / Task 4 scoring: score_genes on .X, random_state=42, use_raw=False; 0 if no gene present."""
    import scanpy as sc

    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in present]
    if present:
        sc.tl.score_genes(adata, present, score_name=name, random_state=SEED, use_raw=False)
    else:
        adata.obs[name] = 0.0
    return present, missing


def coverage_row(role: str, panel: str, score_column: str, genes, present, missing) -> dict[str, object]:
    return {"role": role, "panel": panel, "score_column": score_column, "n_input": len(genes),
            "n_present": len(present), "n_missing": len(missing), "present": ";".join(present),
            "missing": ";".join(missing)}


def threshold(values, quantile: float) -> float:
    """Finite quantile threshold, or NaN when the score is uninformative."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or np.nanstd(arr) == 0:
        return float("nan")
    return float(np.nanquantile(arr, quantile))


def summary_by(frame: pd.DataFrame, group_col: str, score_cols: list[str], flag_cols: list[str]) -> pd.DataFrame:
    rows = []
    frame = frame[[group_col] + score_cols + flag_cols].copy()
    frame[group_col] = frame[group_col].astype(str)
    for group, sub in frame.groupby(group_col, observed=True):
        row: dict[str, object] = {group_col: str(group), "n_cells": int(sub.shape[0])}
        for flag in flag_cols:
            vals = sub[flag].astype(bool)
            row[f"n_{flag}"] = int(vals.sum())
            row[f"fraction_{flag}"] = float(vals.mean()) if vals.size else np.nan
        for score_col in score_cols:
            vals = pd.to_numeric(sub[score_col], errors="coerce")
            row[f"{score_col}_median"] = float(vals.median())
            row[f"{score_col}_mean"] = float(vals.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def match_genes(genes: list[str], var_names, phase: str) -> tuple[list[dict], list[str]]:
    rows, present = [], []
    names = set(var_names)
    for gene in genes:
        hit = next((c for c in (gene, *TIROSH_ALIASES.get(gene, [])) if c in names), None)
        rows.append({"set": phase, "gene": gene, "aliases": ";".join(TIROSH_ALIASES.get(gene, [])),
                     "matched_name": hit or "", "present": hit is not None})
        if hit:
            present.append(hit)
    return rows, present


def aldo_scores(adata, source: str) -> pd.DataFrame:
    """ALDO family x variant scores: mean protein z-score (ddof=0) of UP minus DOWN, as in Task 5.

    Uses .X in its stored dtype (float32), as scripts/05_aldo_score_sensitivity.py matrix_frame() does.
    """
    values = cc.dense(adata.X)
    frame = pd.DataFrame(values, index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))
    sds = frame.std(axis=0, ddof=0).replace(0, np.nan)
    z = frame.subtract(frame.mean(axis=0), axis=1).divide(sds, axis=1).fillna(0.0)

    def panel(genes):
        present = [g for g in genes if g in z.columns]
        return z[present].mean(axis=1).to_numpy() if present else np.full(z.shape[0], np.nan)

    out = {}
    for family, (positive, negative) in ALDO_FAMILIES.items():
        for variant, excluded in ALDO_VARIANTS.items():
            pos = [g for g in positive if g not in excluded]
            neg = [g for g in negative if g not in excluded]
            out[f"{source}__{family}__{variant}"] = panel(pos) - (panel(neg) if neg else 0.0)
    return pd.DataFrame(out, index=pd.Index(adata.obs_names.astype(str), name="cell_id"))


def aldo_coverage(var_names) -> pd.DataFrame:
    rows = []
    for family, (positive, negative) in ALDO_FAMILIES.items():
        for variant, excluded in ALDO_VARIANTS.items():
            for direction, original in (("positive", positive), ("negative", negative)):
                retained = [g for g in original if g not in excluded]
                present = [g for g in retained if g in var_names]
                rows.append({"score_family": family, "variant": variant, "direction": direction,
                             "n_configured": len(original), "n_removed_by_variant": len(original) - len(retained),
                             "n_retained": len(retained), "n_present": len(present),
                             "n_missing": len(retained) - len(present),
                             "removed_by_variant": ";".join(g for g in original if g in excluded),
                             "present": ";".join(present),
                             "missing": ";".join(g for g in retained if g not in var_names)})
    return pd.DataFrame(rows)


def split_score(col: str) -> tuple[str, str, str]:
    parts = col.split("__")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else (parts[0], parts[1], "submission")


def label_contrasts(frame: pd.DataFrame, label_col: str, score_cols: list[str],
                    pairs: list[tuple[str, str]], cohorts: dict[str, np.ndarray]) -> pd.DataFrame:
    """Rank-sum contrasts between label pairs for every score and cohort."""
    labels = frame[label_col].astype(str).to_numpy()
    rows = []
    for cohort, keep in cohorts.items():
        for group, reference in pairs:
            for score_col in score_cols:
                values = frame[score_col].to_numpy()
                row = {"label_column": label_col, "cohort": cohort, "comparison": f"{group}_vs_{reference}",
                       "score": score_col}
                row.update(cc.compare_groups(values[keep & (labels == group)], values[keep & (labels == reference)]))
                rows.append(row)
    return pd.DataFrame(rows)


def aldo_contrast_table(per_cell: pd.DataFrame, label_col: str, score_cols: list[str]) -> pd.DataFrame:
    """ALDO contrasts for one label column; BH within label column x expression source (as in the V1 rerun)."""
    cohorts = {"all_cells": np.ones(len(per_cell), dtype=bool),
               "exclude_TAL_CNT_interstitial_flagged": ~per_cell["exclude_tal_cnt_interstitial"].to_numpy(bool)}
    contrast = label_contrasts(per_cell, label_col, score_cols, ALDO_PAIRS, cohorts)
    parts = contrast["score"].map(split_score)
    contrast.insert(3, "source", parts.map(lambda p: p[0]))
    contrast.insert(4, "family", parts.map(lambda p: p[1]))
    contrast.insert(5, "variant", parts.map(lambda p: p[2]))
    return cc.add_bh(contrast, ["label_column", "source"])


# ---------------------------------------------------------------------------
# Embeddings and clusters
# ---------------------------------------------------------------------------
def leiden(obj, resolution: float, key: str) -> None:
    import scanpy as sc

    try:
        sc.tl.leiden(obj, key_added=key, resolution=resolution, flavor="igraph", random_state=SEED)
    except TypeError:
        sc.tl.leiden(obj, key_added=key, resolution=resolution, random_state=SEED)


def recluster(adata, cluster_key: str, resolution: float):
    """Scale (clip 10), arpack PCA, neighbours (10, 20 PCs), Leiden, UMAP (scripts/04_cell_cycle_regression.py)."""
    import scanpy as sc

    adata.raw = None
    sc.pp.scale(adata, max_value=10)
    n_comps = int(max(2, min(50, adata.n_obs - 1, adata.n_vars - 1)))
    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=min(N_PCS, n_comps))
    leiden(adata, resolution, cluster_key)
    sc.tl.umap(adata, random_state=SEED)
    return adata


def concentration(clusters: pd.Series, is_prolif: np.ndarray) -> dict[str, float]:
    counts = (pd.Series(is_prolif.astype(int), index=clusters.index).groupby(clusters.to_numpy()).sum()
              .sort_values(ascending=False))
    total = int(counts.sum())
    top1, top3 = int(counts.iloc[:1].sum()), int(counts.iloc[:3].sum())
    return {"n_prolif": total, "n_clusters": int(len(counts)), "n_clusters_with_prolif": int((counts > 0).sum()),
            "top1_n": top1, "top1_percent": 100 * top1 / total if total else np.nan,
            "top3_n": top3, "top3_percent": 100 * top3 / total if total else np.nan}


def neighbor_index(embedding: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(embedding)
    return np.vstack([row[row != i][:k] for i, row in enumerate(idx)])


def _purity(idx: np.ndarray, codes: np.ndarray, n_labels: int) -> tuple[np.ndarray, float]:
    per_cell = (codes[idx] == codes[:, None]).mean(axis=1)
    return (np.bincount(codes, weights=per_cell, minlength=n_labels) / np.bincount(codes, minlength=n_labels),
            float(per_cell.mean()))


def knn_purity(embeddings: dict[str, np.ndarray], labels: pd.DataFrame, label_cols, k: int = 10,
               n_perm: int = 1000) -> pd.DataFrame:
    """Mean same-label fraction of k nearest neighbours (20-PC space) against label permutations.

    The permutation RNG is re-seeded (42) for every object x label column, as in the V1 rerun.
    """
    rows = []
    for name, embedding in embeddings.items():
        idx = neighbor_index(embedding, k)
        for col in label_cols:
            cats = cc.ordered_labels(labels[col])
            codes = pd.Categorical(labels[col].astype(str), categories=cats).codes.astype(int)
            observed, observed_all = _purity(idx, codes, len(cats))
            rng = np.random.default_rng(SEED)
            null = np.empty((n_perm, len(cats)))
            null_all = np.empty(n_perm)
            for b in range(n_perm):
                null[b], null_all[b] = _purity(idx, rng.permutation(codes), len(cats))
            for j, label in enumerate([*cats, "all_cells"]):
                obs_v = observed[j] if label != "all_cells" else observed_all
                nul = null[:, j] if label != "all_cells" else null_all
                sd = nul.std(ddof=1)
                rows.append({"object": name, "label_column": col, "label": label,
                             "n_cells": int((codes == j).sum()) if label != "all_cells" else len(codes),
                             "k": k, "embedding": f"X_pca[:, :{N_PCS}]", "observed_same_label_fraction": obs_v,
                             "null_mean": nul.mean(), "null_sd": sd, "null_q025": np.quantile(nul, 0.025),
                             "null_q975": np.quantile(nul, 0.975), "z": (obs_v - nul.mean()) / sd if sd > 0 else np.nan,
                             "p_empirical_ge": (1 + (nul >= obs_v).sum()) / (n_perm + 1), "n_permutations": n_perm})
    return pd.DataFrame(rows)


def ari_nmi_rows(clusters: pd.DataFrame, labels: pd.DataFrame, label_cols) -> pd.DataFrame:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    rows = []
    for column in clusters.columns:
        name, res = column.split("__")
        for col in label_cols:
            rows.append({"object": name, "resolution": res.removeprefix("r"), "n_clusters": clusters[column].nunique(),
                         "label_column": col, "ari": adjusted_rand_score(labels[col], clusters[column]),
                         "nmi": normalized_mutual_info_score(labels[col], clusters[column])})
    return pd.DataFrame(rows)


def concentration_rows(clusters: pd.DataFrame, labels: pd.DataFrame, label_cols) -> pd.DataFrame:
    rows = []
    for column in clusters.columns:
        name, res = column.split("__")
        for col in label_cols:
            row = {"object": name, "resolution": res.removeprefix("r"), "label_column": col}
            row.update(concentration(clusters[column], (labels[col] == "ProLIF").to_numpy()))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["resolution_numeric"] = pd.to_numeric(out["resolution"], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# V1 marker scoring and cluster naming (scp_dct_upload.py lines 115-171)
# ---------------------------------------------------------------------------
def score_v1_lists(adata, lists: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """score_genes with the submission's settings (all defaults: ctrl_size=50, n_bins=25, random_state=0) on .X.

    Lists keep the workbook order; genes absent from the matrix are skipped and a
    list with no present gene is dropped (as in the submission).
    """
    import anndata as ad
    import scanpy as sc

    light = ad.AnnData(X=cc.dense(adata.X).copy(), obs=pd.DataFrame(index=adata.obs_names.astype(str)),
                       var=pd.DataFrame(index=adata.var_names.astype(str)))
    present_lists = {ct: [g for g in genes if g in light.var_names] for ct, genes in lists.items()}
    present_lists = {ct: genes for ct, genes in present_lists.items() if genes}
    for ct, genes in present_lists.items():
        sc.tl.score_genes(light, gene_list=genes, score_name=f"{ct}_score")
    scores = pd.DataFrame({ct: light.obs[f"{ct}_score"].to_numpy() for ct in present_lists},
                          index=pd.Index(light.obs_names, name="cell_id"))
    return scores, present_lists


def cell_labels(scores: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Per-cell argmax (first column wins a tie, as pandas idxmax), max score, tie and all-zero flags."""
    predicted = scores.idxmax(axis=1).astype(str)
    max_score = scores.max(axis=1)
    ties = scores.eq(max_score, axis=0).sum(axis=1) > 1
    all_zero = scores.eq(0).all(axis=1)
    return predicted, max_score, ties, all_zero


def cluster_labels(predicted: pd.Series, max_score: pd.Series, clusters: pd.Series) -> tuple[pd.Series, list[str], dict]:
    """Cluster-majority names with the submission's 'Unknown' rule.

    Cluster name = most common per-cell label (pandas mode; ties go to the
    alphabetically first label). Clusters whose mean max_score is below the 1%
    quantile of the cluster means are 'Unknown'; the diversity rule
    (n distinct labels > its 100% quantile) can never fire, as in the original.
    """
    frame = pd.DataFrame({"cluster": clusters.astype(str).to_numpy(), "predicted": predicted.to_numpy(),
                          "max_score": max_score.to_numpy(float)}, index=predicted.index)
    cluster_scores = frame.groupby("cluster")["max_score"].mean()
    cluster_diversity = frame.groupby("cluster")["predicted"].nunique()
    score_threshold = cluster_scores.quantile(0.01)
    diversity_threshold = cluster_diversity.quantile(1)
    unknown = cluster_scores[cluster_scores < score_threshold].index.union(
        cluster_diversity[cluster_diversity > diversity_threshold].index)
    mapping = frame.groupby("cluster")["predicted"].agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
    named = frame["cluster"].map(mapping)
    named[frame["cluster"].isin(unknown)] = "Unknown"
    info = {"score_threshold": float(score_threshold), "unknown_clusters": sorted(map(str, unknown)),
            "n_unknown_cells": int(frame["cluster"].isin(unknown).sum())}
    return named.astype(str), sorted(map(str, unknown)), info
