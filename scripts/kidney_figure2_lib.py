"""kidney_figure2 shared definitions and functions. Runs no analysis by itself.

Imported by the ``kidney_figure2_*.py`` step scripts.
Task: recompute the kidney panels of the submitted Figure 2 that were made with the dropped per-cell re-annotation
(``kidney_reannot_label`` = ``cell_type_v2``) under the one revised kidney annotation, ``kidney_t15_label``:
Figure 2D (protein IDs and within-cell dynamic range per cell type), 2E (exact core-set membership), 2F (pairwise
Jaccard of core proteomes), 2G (stacked violin of the top-10 DE proteins) and Supplementary Table S6.

Decisions by the authors (31 Oct 2026, plan approval)
--------------------------------------------------
- Task number kidney_figure2. Plan approved as written.
- Unknown (109 cells under ``kidney_t15_label``): its own grey group in Figure 2D; left out of 2E-2G and Table S3
  (the submitted code's rule); a sensitivity table keeps Unknown as a group.
- IDs per cell in the revised 2D: proteins > 0 among the 4,054 retained proteins. The
  submitted 2D top row came from section 7 of the script (all 4,144 export rows).
- Display names: DCT1 -> DCT, PC -> CNT/PC. Row order and colours as the revised Figure 2A / 2C (kidney_annotation / kidney_rna).
- 2G keeps the submitted rule (all types' Wilcoxon scores sorted together; first 10 unique proteins), although under
  ``kidney_t15_label`` it selects ten PCT proteins (authors, 31 Oct 2026, after step 2).
- 2E: UpSet layout (bars + membership matrix). Supplementary Table S6 also lists the 49 unique-core proteins. Every plot
  is its own file with no panel letter or title (the authors assemble Figure 2 in PowerPoint) (authors, 31 Oct 2026).
- PST is drawn in tab10 brown #8c564b instead of the kidney_annotation palette grey #7f7f7f (authors, 10 Nov 2026), as in the S-07
  panels; ``palette()`` below applies it in every kidney_figure2 figure.

Submitted definitions (``first_submission/scp_kidney_upload.py``; ported, not imported: the script runs at import)
----------------------------------------------------------------------------------------------------------------
- ``compute_cell_metrics``: IDs = values != 0 in ``.raw`` (pre-log intensities of the retained proteins); dynamic
  range = log10(P95) - log10(P5) of the positive values, cells with >= 10 values ('linear' scale detected).
- section 7 (``_compute_ids_from_csv``): IDs recounted from the raw export (non-missing and != 0, all rows); this
  count drew the submitted 2D top row (``_plot_sens_per_celltype``).
- ``_compute_coverage_matrix``: fraction of a type's cells with the protein > 0; Unknown left out.
- ``write_core_unique_tables``: core = coverage >= CORE_FRAC (0.50); unique core = core and coverage <
  UNIQUE_LOW_FRAC (0.10) in every other type.
- ``plot_upset_core``: per protein the exact set of types where it is core; counts per set; top UPSET_TOPN (20).
- ``plot_core_jaccard``: |A_i & A_j| / |A_i | A_j| of the core sets.
- ``plot_stacked_violin``: Wilcoxon (``sc.tl.rank_genes_groups`` on ``.raw``) over the known groups with >= 2 cells;
  all groups' results sorted by score; the first 10 unique proteins; violins of ``.raw`` with standard_scale='var'.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import scp_common as cc

TASK = "kidney_figure2"
SEED = cc.RANDOM_SEED
DATASET = "kidney"

# Submitted parameters (scp_kidney_upload.py) --------------------------------------------------------------------
CORE_FRAC = 0.50
UNIQUE_LOW_FRAC = 0.10
UPSET_TOPN = 20
DE_TOP = 10
MIN_DR_VALUES = 10           # compute_cell_metrics: dynamic range only for cells with >= 10 detected values
MIN_DE_CELLS = 2             # plot_stacked_violin: groups with >= 2 cells
COVERAGE_FRACS = (0.10, 0.25, 0.50)
LEIDEN_RES = 2.0

# Submitted inputs ------------------------------------------------------------------------------------------------
SUBMISSION_DIR = cc.PROJECT_ROOT / "first_submission"
SUBMITTED_SCRIPT = SUBMISSION_DIR / "scp_kidney_upload.py"
SUBMITTED_FIGURE = SUBMISSION_DIR / "Figure 2.tif"
SUPP_T3 = SUBMISSION_DIR / "Supplemental Table 3.xlsx"
V3 = cc.KIDNEY_DIR / "V3"
SUB = {
    "per_cell": V3 / "SENS_DYNR_per_cell_Leiden_2.0.csv",
    "by_type": V3 / "SENS_DYNR_by_celltype_AFTER_Leiden_2.0.csv",
    "protein_ids": V3 / "ProteinIDs_byCellType_AFTER_Leiden_2.0.csv",
    "core_summary": V3 / "CoreUnique_Summary_AFTER_core50_low10_Leiden_2.0.csv",
    "upset": V3 / "UpSet_CoreCombination_AFTER_core50_Leiden_2.0.csv",
    "members": V3 / "UpSet_CoreCombination_AFTER_core50_Top20_Leiden_2.0_members.csv",
    "png_sens": V3 / "Violin_sens_n_proteins_AFTER_Leiden_2.0.png",
    "png_dynr": V3 / "Violin_dynr_value_AFTER_Leiden_2.0.png",
    "png_upset": V3 / "UpSet_CoreCombination_AFTER_core50_Top20_Leiden_2.0.png",
    "png_jaccard": V3 / "Heatmap_Core_Jaccard_AFTER_core50_Leiden_2.0.png",
    "png_violin": V3 / "StackedViolin_top10_AFTER_Leiden_2.0.png",
}
SUB_CORE_LIST = "Core_Proteins_{ct}_AFTER_core50_low10_Leiden_2.0.csv"
SUB_UNIQUE_LIST = "UniqueCore_Proteins_{ct}_AFTER_core50_low10_Leiden_2.0.csv"
# Read from the submitted Figure 2G (x axis, left to right).
SUBMITTED_TOP10 = ("S100g", "Slc27a2", "Calb1", "Ces2c", "Slc12a1", "Ckmt1", "Umod", "Gatm", "Lactb2", "Acox1")

# Submitted Results sentences on Figure 2D-2G (Mouse SCP manuscript-KI.docx) -----------------------------------
NAMED_PROTEINS = {   # protein -> cell type it was named for in the Results (short submitted label)
    "S100g": "annotation marker", "Slc27a2": "annotation marker", "Calb1": "annotation marker",
    "Slc12a1": "annotation marker", "Umod": "annotation marker",
    "Ces2c": "PST", "Ckmt1": "IC-B", "Gatm": "PCT", "Lactb2": "PST", "Acox1": "PST",
}
# The kidney_annotation types that carry the submitted names of the named-protein sentence.
SUBMITTED_TO_KIDNEY_ANNOTATION = {"PST": "PST", "PCT": "PCT", "IC-B": "IC-B"}

# Labels ----------------------------------------------------------------------------------------------------------
PRIMARY = "kidney_t15_label"
SUBMITTED_LABEL = "kidney_reannot_label"
SUBMITTED_LABEL_FULL = "kidney_reannot_label_full"
UNKNOWN = "Unknown"
DISPLAY = {"DCT1": "DCT", "PC": "CNT/PC"}
UNKNOWN_COLOR = cc.UNKNOWN_COLOR
# PST colour (authors, 10 Nov 2026: "Change PST colour"): tab10 brown instead of the kidney_annotation palette grey (#7f7f7f, next to
# Unknown #B3B3B3); the colour of the S-07 panels (report_pt_tal_figures.py), so one colour per cell type across
# Figure 2 and the supplement. The response to the reviewers CIELAB check against the other Figure 2A colours: smallest distance 40
# (normal vision) and 34 (simulated deutan / protan), against 20 for the grey. kidney_annotation outputs are not changed.
PST = "PST"
PST_COLOR = "#8c564b"
PST_COLOR_KIDNEY_ANNOTATION = "#7f7f7f"

# Figures (KI half-page panels) -------------------------------------------------------------------------------------
MM = 1 / 25.4
PANEL_W_MM = 88.0
MAX_W_MM = 90.0
MIN_FONT_PT = 5.0


def display(label: str) -> str:
    return DISPLAY.get(str(label), str(label))


def is_unknown(x) -> bool:
    return isinstance(x, str) and x.lower() == "unknown"


def palette(adata, rows: list[str], checks=None) -> dict[str, str]:
    """Row colours of the revised Figure 2: ``kidney_annotation_chosen_labels.palette`` with PST replaced by PST_COLOR
    (authors, 10 Nov 2026). With ``checks``, adds one check row (the kidney_annotation value was PST_COLOR_kidney_annotation and no other drawn row
    uses PST_COLOR) and one information row (palette labels not drawn here that carry PST_COLOR: the kidney_annotation palette
    also holds the submitted labels, where MD is #8c564b in the submitted Figure 2A)."""
    import kidney_annotation_chosen_labels as t15c
    colors = t15c.palette(adata, rows)
    before = colors.get(PST)
    same = lambda c: str(c).lower() == PST_COLOR.lower()      # noqa: E731
    others = sorted(t for t in rows if t != PST and same(colors.get(t, "")))
    not_drawn = sorted(t for t, c in colors.items() if t not in rows and same(c))
    colors[PST] = PST_COLOR
    if checks is not None:
        checks.add("PST colour: kidney_annotation palette value replaced by tab10 brown (authors, 10 Nov 2026); no other drawn row "
                   "uses it", f"{before} -> {PST_COLOR}; other drawn rows with {PST_COLOR}: "
                   f"{', '.join(others) or 'none'}", before == PST_COLOR_KIDNEY_ANNOTATION and not others,
                   f"{PST_COLOR_KIDNEY_ANNOTATION} -> {PST_COLOR}")
        checks.add(f"palette labels not drawn here that carry {PST_COLOR} (submitted labels)",
                   ", ".join(not_drawn) or "none", None)
    return colors


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load(run: cc.Run, checks=None):
    """kidney_object canonical kidney object and labels with ``kidney_t15_label`` (kidney_annotation, matched on cell_id; checked)."""
    import kidney_rna_lib as t7

    return t7.load_kidney(run, checks)


def intensity(adata) -> np.ndarray:
    """Pre-log intensities of the retained proteins (missing = 0): the submitted ``.raw.X`` (kidney_object checked equal)."""
    return cc.dense(adata.layers["intensity"]).astype(np.float32)


def submitted_abbreviations(run: cc.Run) -> dict[str, str]:
    """``abbr_map`` of the submitted script: workbook column 1 (names) -> column 3 (abbreviations)."""
    marker_df = pd.read_excel(run.use_input(cc.INPUTS["kidney"]["markers"], "kidney V3 marker workbook (Supp Table 1)"))
    names = marker_df.iloc[:, 0].astype(str).str.strip()
    abbrs = marker_df.iloc[:, 2].astype(str).str.strip()
    return dict(zip(names, abbrs))


# ---------------------------------------------------------------------------
# Ports of the submitted functions (logic unchanged; inputs passed explicitly)
# ---------------------------------------------------------------------------
def detect_scale(raw: np.ndarray) -> str:
    """``_detect_intensity_scale``."""
    vals = np.asarray(raw).ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals != 0]
    if vals.size == 0:
        return "linear"
    med = np.nanmedian(vals)
    has_neg = np.nanmin(vals) < 0
    return "log" if (has_neg or med < 50) else "linear"


def cell_metrics(raw: np.ndarray, index) -> pd.DataFrame:
    """``compute_cell_metrics``: IDs per cell and within-cell dynamic range (P5 / P95 of the detected values)."""
    scale = detect_scale(raw)
    n = raw.shape[0]
    sens = np.zeros(n, dtype=int)
    dr, p05s, p95s = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        row = np.asarray(raw[i, :]).ravel()
        vals = row[np.isfinite(row)]
        vals = vals[vals != 0]
        sens[i] = int(vals.size)
        if vals.size >= MIN_DR_VALUES:
            if scale == "linear":
                pos = vals[vals > 0]
                if pos.size >= MIN_DR_VALUES:
                    q05, q95 = np.nanpercentile(pos, 5), np.nanpercentile(pos, 95)
                    if q95 > 0 and q05 > 0:
                        p05s[i], p95s[i] = np.log10(q05), np.log10(q95)
                        dr[i] = p95s[i] - p05s[i]
            else:
                q05, q95 = np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)
                p05s[i], p95s[i] = q05, q95
                dr[i] = q95 - q05
    return pd.DataFrame({"sens_n_proteins": sens, "dynr_value": dr, "dynr_low": p05s, "dynr_high": p95s,
                         "dynr_scale": scale}, index=pd.Index(index, name="cell_id"))


def ids_all_export_rows(csv_path, cell_ids) -> pd.Series:
    """Section 7 (``_compute_ids_from_csv`` + ``_overwrite_obs_counts_from_series``): IDs over all export rows."""
    df = pd.read_csv(csv_path, low_memory=False)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols and df.shape[1] > 3:
        num_cols = list(df.columns[3:])
    mat = df[num_cols]
    det = mat.notna().to_numpy() & (mat.to_numpy() != 0)
    s = pd.Series(np.sum(det, axis=0), index=pd.Index(num_cols).astype(str), name="ids_all_export_rows")
    return s.reindex(pd.Index(cell_ids).astype(str)).fillna(0).astype(int)


def _iqr(x: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    if a.size == 0:
        return float("nan")
    return float(np.nanpercentile(a, 75) - np.nanpercentile(a, 25))


def summarize_by(labels: pd.Series, metrics: pd.DataFrame) -> pd.DataFrame:
    """``summarize_by``: n, median and IQR of IDs and dynamic range per known type (rows in order of appearance)."""
    df = pd.DataFrame({"g": labels.astype(str).to_numpy(), "sens_n_proteins": metrics["sens_n_proteins"].to_numpy(),
                       "dynr_value": metrics["dynr_value"].to_numpy()})
    df = df[df["g"].str.lower() != "unknown"].dropna(subset=["sens_n_proteins", "dynr_value"])
    df["g"] = df["g"].astype("category").cat.remove_unused_categories()
    agg = (df.groupby("g", observed=True, sort=False)
             .agg(n_cells=("sens_n_proteins", "size"), sens_median=("sens_n_proteins", "median"),
                  sens_iqr=("sens_n_proteins", _iqr), dr_median=("dynr_value", "median"),
                  dr_iqr=("dynr_value", _iqr))
             .reset_index().rename(columns={"g": "cell_type"}))
    agg["cell_type"] = agg["cell_type"].astype(str)
    return agg


def known_types(labels: pd.Series, order=None) -> list[str]:
    """Types the submitted functions loop over: categories of the known labels (sorted), or a given order."""
    present = sorted(set(labels.astype(str)) - {UNKNOWN})
    present = [t for t in present if not is_unknown(t)]
    if order is None:
        return present
    return [t for t in order if t in present] + [t for t in present if t not in order]


def protein_ids_by_celltype(det: np.ndarray, labels: pd.Series, sens: pd.Series, types=None) -> pd.DataFrame:
    """``protein_ids_by_celltype``: union IDs, proteins in >= 10/25/50% of the type's cells, median IDs per cell."""
    lab = labels.astype(str).to_numpy()
    types = types or known_types(labels)
    rows = []
    for ct in types:
        sel = lab == ct
        x = det[sel]
        n_cells = int(sel.sum())
        col_sums = x.sum(axis=0)
        row = {"cell_type": ct, "n_cells": n_cells, "union_ids": int((x.max(axis=0) > 0).sum())}
        for frac in COVERAGE_FRACS:
            thr = int(np.ceil(frac * n_cells))
            row[f"coverage_ge_{int(frac * 100)}pct"] = int((col_sums >= thr).sum())
        row["median_per_cell_IDs"] = float(np.median(np.asarray(sens)[sel]))
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_matrix(det: np.ndarray, labels: pd.Series, types=None) -> tuple[list[str], np.ndarray]:
    """``_compute_coverage_matrix``: rows = types (Unknown left out), columns = proteins, fraction of cells > 0."""
    lab = labels.astype(str).to_numpy()
    types = types or known_types(labels)
    coverage = np.zeros((len(types), det.shape[1]), dtype=float)
    for i, ct in enumerate(types):
        sel = lab == ct
        if sel.sum():
            coverage[i, :] = det[sel].sum(axis=0) / float(sel.sum())
    return types, coverage


def core_unique(coverage: np.ndarray, core_frac: float = CORE_FRAC,
                low_frac: float = UNIQUE_LOW_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """``write_core_unique_tables``: core masks and unique-core masks (types x proteins)."""
    n_types, n_prot = coverage.shape
    core = coverage >= core_frac
    unique = np.zeros_like(core, dtype=bool)
    for i in range(n_types):
        low_else = (np.all(coverage[np.arange(n_types) != i, :] < low_frac, axis=0) if n_types > 1
                    else np.ones(n_prot, bool))
        unique[i, :] = core[i, :] & low_else
    return core, unique


def exact_combinations(types: list[str], core: np.ndarray, abbr: dict[str, str],
                       stable: bool = False) -> tuple[list[str], pd.DataFrame]:
    """``plot_upset_core``: per protein the exact set of types where it is core, and the counts per set.

    ``stable=False`` keeps the submitted ``value_counts`` order (ties in any order the installed pandas gives);
    ``stable=True`` orders ties by fewer types, then by the abbreviated label (used for the revised panels).
    """
    combos = []
    for p in range(core.shape[1]):
        members = [types[i] for i in range(len(types)) if core[i, p]]
        combos.append(";".join(members) if members else "(none)")
    s = pd.Series(combos, name="combo").value_counts().reset_index()
    s.columns = ["combo", "count"]
    s = s[s["combo"] != "(none)"].copy()
    s["combo_abbr"] = s["combo"].map(lambda c: " + ".join(abbr.get(t, t) for t in c.split(";") if t))
    s["n_types"] = s["combo"].str.count(";") + 1
    if stable:
        s = s.sort_values(["count", "n_types", "combo_abbr"], ascending=[False, True, True], kind="mergesort")
    return combos, s.reset_index(drop=True)


def top_combinations(counts: pd.DataFrame, topn: int = UPSET_TOPN, stable: bool = False) -> pd.DataFrame:
    """``s.nlargest(topn, 'count')`` (submitted) or the first ``topn`` rows of the stable order."""
    return counts.head(topn) if stable else counts.nlargest(topn, "count")


def combination_members(combos: list[str], top: pd.DataFrame, protein_names, types, abbr) -> pd.DataFrame:
    """``plot_upset_core`` members file: proteins of the plotted combinations with 0/1 flags per type."""
    top_set = set(top["combo"])
    rows = []
    for p, combo in enumerate(combos):
        if combo in top_set:
            parts = combo.split(";")
            row = {"Protein": protein_names[p], "combo_full": combo,
                   "combo_abbr": " + ".join(abbr.get(t, t) for t in parts), "n_types": len(parts)}
            for t in types:
                row[abbr.get(t, t)] = 1 if t in parts else 0
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_types", "combo_abbr", "Protein"], ascending=[True, True, True])


def jaccard(core: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``plot_core_jaccard``: Jaccard matrix of the core sets, plus intersection and union counts."""
    n = core.shape[0]
    jac, inter, union = np.zeros((n, n)), np.zeros((n, n), int), np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            inter[i, j] = int(np.logical_and(core[i], core[j]).sum())
            union[i, j] = int(np.logical_or(core[i], core[j]).sum())
            jac[i, j] = inter[i, j] / union[i, j] if union[i, j] > 0 else 0.0
    return jac, inter, union


def de_adata(raw: np.ndarray, labels: pd.Series, var_names, order=None):
    """AnnData of the known groups with >= 2 cells (``plot_stacked_violin``); ``.X`` = the submitted ``.raw``."""
    import anndata as ad

    lab = labels.astype(str)
    known = lab.str.lower() != "unknown"
    counts = lab[known].value_counts()
    keep = counts[counts >= MIN_DE_CELLS].index.tolist()
    mask = (known & lab.isin(keep)).to_numpy()
    cats = [t for t in (order or sorted(keep)) if t in keep] + sorted(set(keep) - set(order or keep))
    obs = pd.DataFrame({"group": pd.Categorical(lab[mask].to_numpy(), categories=cats)},
                       index=pd.Index(lab.index[mask].astype(str)))
    return ad.AnnData(X=np.asarray(raw[mask], dtype=np.float32), obs=obs,
                      var=pd.DataFrame(index=pd.Index(list(var_names), dtype=str)))


def de_rank(adata_de) -> tuple[pd.DataFrame, list[str]]:
    """``plot_stacked_violin``: Wilcoxon per group vs the rest; all rows sorted by score; first 10 unique names.

    Fold changes of ``rank_genes_groups`` assume log data and overflow on the raw intensities (warnings silenced);
    the submitted ranking uses the scores only.
    """
    import warnings

    import scanpy as sc

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        sc.tl.rank_genes_groups(adata_de, groupby="group", method="wilcoxon", key_added="rank_genes_group")
    de_df = sc.get.rank_genes_groups_df(adata_de, group=None, key="rank_genes_group")
    de_df = de_df.sort_values(by="scores", ascending=False)
    valid = set(adata_de.var_names)
    top: list[str] = []
    for g in de_df["names"].tolist():
        if g in valid and g not in top:
            top.append(g)
        if len(top) == DE_TOP:
            break
    return de_df, top


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def setup_style() -> None:
    """qc_mito style (7 pt, no top/right spines) with Arial, as the revised Figure 2A / 2C."""
    import qc_mito_lib as t3

    t3.apply_style()
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})


def scale_var(values: np.ndarray) -> np.ndarray:
    """``standard_scale='var'`` of scanpy's violin plots: per protein (min, max) -> (0, 1) over the plotted cells."""
    x = np.asarray(values, dtype=float)
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return (x - lo) / span


def group_medians(scaled: np.ndarray, groups: np.ndarray, rows: list[str]) -> np.ndarray:
    return np.array([np.median(scaled[groups == r], axis=0) for r in rows])


def stacked_violin(fig, frame_spec, scaled: np.ndarray, groups: np.ndarray, rows: list[str], row_labels: list[str],
                   protein_labels: list[str], cmap="Blues", linewidth: float = 0.25, rotation: float = 45):
    """Stacked violin with the layout of scanpy's ``StackedViolin`` (rows = groups, columns = proteins).

    As scanpy: one row axis per group with its own y range, seaborn violins (``density_norm='width'``, ``cut=0``,
    no inner marks), fill = the group median of the 0-1 scaled values on one colour scale (min-max of the medians),
    half-column padding inside an outer frame. Drawn without hue dodging (scanpy 1.11.3 with seaborn 0.13.2 and
    pandas 3 dodges the hue and draws narrow, shifted violins). Returns (frame axis, medians, Normalize).
    """
    import warnings

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize, to_hex
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    med = group_medians(scaled, groups, rows)
    norm = Normalize(vmin=float(np.nanmin(med)), vmax=float(np.nanmax(med)))
    n_r, n_c = len(rows), len(protein_labels)
    frame = fig.add_subplot(frame_spec)
    frame.set_xlim(-0.5, n_c + 0.5)
    frame.set_ylim(n_r + 0.5, -0.5)
    frame.set_xticks(np.arange(n_c) + 0.5)
    frame.set_xticklabels(protein_labels, rotation=rotation, ha="right" if rotation else "center",
                          rotation_mode="anchor")
    frame.set_yticks(np.arange(n_r) + 0.5)
    frame.set_yticklabels(row_labels)
    for side in frame.spines.values():
        side.set_visible(True)
        side.set_linewidth(0.6)
    frame.set_zorder(0)
    gs = GridSpecFromSubplotSpec(n_r + 2, n_c + 2, subplot_spec=frame_spec, hspace=0, wspace=0,
                                 height_ratios=[0.5] + [1] * n_r + [0.5], width_ratios=[0.5] + [1] * n_c + [0.5])
    cols = [f"p{j}" for j in range(n_c)]
    for i, r in enumerate(rows):
        rax = fig.add_subplot(gs[i + 1, 1:-1])
        sub = pd.DataFrame(scaled[groups == r], columns=cols).melt(var_name="protein", value_name="value")
        pal = {c: to_hex(cm(norm(med[i, j]))) for j, c in enumerate(cols)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.violinplot(data=sub, x="protein", y="value", hue="protein", order=cols, hue_order=cols, palette=pal,
                           dodge=False, density_norm="width", cut=0, inner=None, linewidth=linewidth,
                           linecolor="#52514e", legend=False, ax=rax)
        rax.set_xlim(-0.5, n_c - 0.5)
        rax.set_axis_off()
        rax.patch.set_alpha(0)
        rax.set_zorder(1)
    return frame, med, norm


def figure_checks(fig, name: str, paths, expected_labels=None, max_w_mm: float = MAX_W_MM) -> dict:
    """Format checks on a saved figure: PDF and PNG written, PNG at 600 dpi, width <= max_w_mm, fonts >= 5 pt,
    and (optionally) every expected label drawn as text."""
    from PIL import Image

    paths = [cc.PROJECT_ROOT / p if not str(p).startswith(str(cc.PROJECT_ROOT)) else p for p in paths]
    pdf_ok = any(str(p).endswith(".pdf") and p.exists() for p in paths)
    png = [p for p in paths if str(p).endswith(".png")]
    dpi, w_mm, h_mm = np.nan, np.nan, np.nan
    if png and png[0].exists():
        with Image.open(png[0]) as im:
            dpi = float(im.info.get("dpi", (np.nan,))[0])
            w_mm, h_mm = im.size[0] / dpi * 25.4, im.size[1] / dpi * 25.4
    texts = [t for t in fig.findobj(match=lambda o: hasattr(o, "get_text") and hasattr(o, "get_fontsize"))
             if t.get_text().strip() and t.get_visible()]
    min_font = min((t.get_fontsize() for t in texts), default=np.nan)
    drawn = {t.get_text().strip() for t in texts}
    missing = [x for x in (expected_labels or []) if x not in drawn]
    ok = bool(pdf_ok and abs(dpi - 600) < 1 and w_mm <= max_w_mm + 0.5 and min_font >= MIN_FONT_PT and not missing)
    return {"figure": name, "pdf_written": pdf_ok, "png_dpi": round(dpi, 1), "width_mm": round(w_mm, 1),
            "height_mm": round(h_mm, 1), "min_font_pt": round(float(min_font), 2),
            "labels_missing": "; ".join(missing), "pass": ok}


def ceil_half(n: int) -> int:
    return int(math.ceil(CORE_FRAC * n))
