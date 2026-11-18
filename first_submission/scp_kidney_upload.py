# ----------------------------------------------------------------------------------------
# Written by Qi Wu, Department of Biomedicine, Aarhus University, with the aid of ChatGPT
# Date: 11/Feb/2026
# Contact Qi Wu (qi.wu@biomed.au.dk) for information and comments
# ----------------------------------------------------------------------------------------

from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
from matplotlib.colors import to_hex

np.random.seed(42)

# =========================
# -------- CONFIG ---------
# =========================
CSV_FILE_PATH = r"PATH\TO\KIDNEY_768cells.csv"
MARKER_XLSX  = r"PATH\TO\KIDNEY_markers.xlsx"

# Analysis params
MIN_CELL_GENES = 100
MIN_GENE_CELLS = 2
N_NEIGHBORS    = 10
N_PCS          = 20
LEIDEN_RES     = 2.0
KEEP_UNKNOWN   = True   # keep Unknown, else resolve via voting and/or map to CNT
UNKNOWN_FALLBACK = 'Connecting tubule cell'

# Preferred display order (Unknown last)
PREFERRED_ORDER = ['PT', 'DTL', 'TAL', 'DCT', 'CNT', 'CD', 'Unknown']

# Custom thresholds for "core"/"unique" analyses (overridable by env)
CORE_FRAC       = float(os.getenv('CORE_FRAC', '0.50'))
UNIQUE_LOW_FRAC = float(os.getenv('UNIQUE_LOW_FRAC', '0.10'))
UPSET_TOPN      = int(os.getenv('UPSET_TOPN', '20'))

# =========================
# -------- I/O ------------
# =========================
print("📂 Data folder detected:", os.path.dirname(CSV_FILE_PATH))

raw_df = pd.read_csv(CSV_FILE_PATH)
marker_df = pd.read_excel(MARKER_XLSX)

# marker lists per cell type; also optional abbreviations (3rd column)
marker_genes: Dict[str, List[str]] = {
    str(row[0]): [g.strip().lower().capitalize() for g in str(row[1]).split(',')]
    for row in marker_df.itertuples(index=False)
}
abbr_map: Dict[str, str] = {}
if marker_df.shape[1] >= 3:
    names = marker_df.iloc[:, 0].astype(str).str.strip()
    abbrs = marker_df.iloc[:, 2].astype(str).str.strip()
    abbr_map = dict(zip(names, abbrs))

# =========================
# ---- Build AnnData ------
# =========================
metadata = raw_df.iloc[:, :3]
X = raw_df.iloc[:, 3:].fillna(0).astype(np.float32).T
adata = sc.AnnData(X)
adata.uns['source_csv'] = CSV_FILE_PATH

# gene names
if 'PG.Genes' in metadata.columns:
    genes = metadata['PG.Genes'].astype(str).str.split(';').str[0].str.strip().fillna('Unknown')
else:
    print("⚠️ Column 'PG.Genes' not found; filling as 'Unknown'.")
    genes = pd.Series(['Unknown'] * adata.n_vars)
adata.var['gene_name'] = genes[:adata.n_vars].values
adata.var_names = adata.var['gene_name'].astype(str).str.lower().str.capitalize()

# make var names unique
if adata.var_names.duplicated().any():
    print("⚠️ Duplicate gene names found. Making them unique…")
    seen = {}
    new_names = []
    for name in adata.var_names:
        c = seen.get(name, 0)
        new_names.append(name if c == 0 else f"{name}_{c}")
        seen[name] = c + 1
    adata.var_names = pd.Index(new_names)

# QC & norm
sc.pp.filter_cells(adata, min_genes=MIN_CELL_GENES)
sc.pp.filter_genes(adata, min_cells=MIN_GENE_CELLS)
adata.raw = adata.copy()  # preserve pre-log values for presence/absence
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# neighbors / Leiden
leiden_key = f"leiden_res_{LEIDEN_RES:.2f}"
sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
sc.tl.leiden(adata, key_added=leiden_key, resolution=LEIDEN_RES, flavor='igraph')

# =========================
# ---- Utilities ----------
# =========================

def _is_unknown(x: str) -> bool:
    return isinstance(x, str) and x.lower() == 'unknown'


def _states() -> List[Tuple[str, str]]:
    """Return available (group_key, tag) pairs for plotting/aggregation."""
    out: List[Tuple[str, str]] = []
    if 'cell_type_pre' in adata.obs.columns:
        out.append(('cell_type_pre', 'BEFORE'))
    if 'cell_type_v2' in adata.obs.columns:
        out.append(('cell_type_v2', 'AFTER'))
    elif f"{leiden_key}_named" in adata.obs.columns:
        out.append((f"{leiden_key}_named", 'BEFORE'))
    return out


def _collect_union_categories() -> List[str]:
    union: List[str] = []
    for col in ['cell_type_pre', 'cell_type_v2', f"{leiden_key}_named"]:
        if col in adata.obs.columns:
            ser = adata.obs[col].astype('category')
            for c in ser.cat.categories:
                if c not in union:
                    union.append(c)
    return union


def _build_global_palette() -> Tuple[Dict[str, str], List[str]]:
    """Build a reproducible global palette across groupings; Unknown is grey."""
    union = _collect_union_categories()
    pref  = [c for c in PREFERRED_ORDER if c in union and not _is_unknown(c)]
    others = [c for c in union if c not in pref and not _is_unknown(c)]
    unk = [c for c in union if _is_unknown(c)]
    order = pref + others + (unk[:1] if unk else [])

    # base colors from Matplotlib tab schemes + HUSL if needed
    def _even_odd(colors: Iterable):
        colors = list(colors)
        return colors[0::2] + colors[1::2]

    tab20  = _even_odd(plt.get_cmap('tab20').colors)
    tab20b = _even_odd(plt.get_cmap('tab20b').colors)
    tab20c = _even_odd(plt.get_cmap('tab20c').colors)
    base   = list(tab20) + list(tab20b) + list(tab20c)
    if len(order) > len(base):
        base += sns.color_palette('husl', len(order)-len(base))

    pal: Dict[str, str] = {}
    i = 0
    for c in order:
        if _is_unknown(c):
            pal[c] = '#b3b3b3'
        else:
            pal[c] = to_hex(base[i]); i += 1
    adata.uns['celltype_global_palette'] = pal
    adata.uns['celltype_global_order']   = order
    return pal, order


def _palette_for(group_key: str) -> Tuple[Dict[str, str], List[str]]:
    pal, _ = _build_global_palette()
    cats = adata.obs[group_key].astype('category').cat.categories.tolist()
    pref = [c for c in PREFERRED_ORDER if c in cats and not _is_unknown(c)]
    others = [c for c in cats if c not in pref and not _is_unknown(c)]
    unk = [c for c in cats if _is_unknown(c)]
    order = pref + others + (unk[:1] if unk else [])
    return {c: pal.get(c, '#aaaaaa') for c in order}, order


def _ensure_umap():
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, svd_solver='arpack')
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)


# =========================
# 1) Initial annotation by marker scoring
# =========================
# keep only markers present
matching_markers = {ct: [g for g in gl if g in adata.var_names]
                    for ct, gl in marker_genes.items()}
matching_markers = {ct: gl for ct, gl in matching_markers.items() if gl}

if matching_markers:
    for ct, gl in matching_markers.items():
        sc.tl.score_genes(adata, gene_list=gl, score_name=f"{ct}_score")
    scores_df = pd.DataFrame({ct: adata.obs[f"{ct}_score"] for ct in matching_markers})
    adata.obs['predicted_cell_type'] = scores_df.idxmax(axis=1)
    adata.obs['max_score'] = scores_df.max(axis=1)

    clust_mean = adata.obs.groupby(leiden_key, observed=True)['max_score'].mean()
    clust_div  = adata.obs.groupby(leiden_key, observed=True)['predicted_cell_type'].nunique()
    score_threshold    = clust_mean.quantile(0.01)  # bottom 1%
    diversity_threshold = clust_div.quantile(1)     # most mixed
    unknown_clusters = clust_mean[clust_mean < score_threshold].index.union(
        clust_div[clust_div > diversity_threshold].index
    )

    cluster_to_celltype = (
        adata.obs.groupby(leiden_key)['predicted_cell_type']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        .to_dict()
    )
    adata.obs[f"{leiden_key}_named"] = adata.obs[leiden_key].map(cluster_to_celltype)
    adata.obs.loc[adata.obs[leiden_key].isin(unknown_clusters), f"{leiden_key}_named"] = 'Unknown'
    adata.obs[f"{leiden_key}_named"] = adata.obs[f"{leiden_key}_named"].astype('category')
    adata.obs['cell_type_pre'] = adata.obs[f"{leiden_key}_named"].astype('category')

# =========================
# 2) Re-annotation with shared marker penalization + auto threshold tuning
# =========================

# positive markers per type
marker_pos = {ct: [g for g in gl if g in adata.var_names] for ct, gl in marker_genes.items()}
marker_pos = {ct: gl for ct, gl in marker_pos.items() if gl}

# negative marker set: markers in competing types
_gene_to_cts: Dict[str, set] = defaultdict(set)
for ct, gl in marker_genes.items():
    for g in gl:
        _gene_to_cts[g].add(ct)
marker_neg: Dict[str, List[str]] = {}
for ct, gl in marker_genes.items():
    competitors = set()
    for g in gl:
        competitors |= (_gene_to_cts[g] - {ct})
    neg = set()
    for other in competitors:
        neg |= set(marker_genes.get(other, []))
    neg -= set(gl)
    marker_neg[ct] = [g for g in neg if g in adata.var_names]

alpha = 0.7  # penalty strength
scores: Dict[str, np.ndarray] = {}
for ct in marker_genes.keys():
    pos_list = marker_pos.get(ct, [])
    neg_list = marker_neg.get(ct, [])
    pos = np.zeros(adata.n_obs) if not pos_list else sc.tl.score_genes(adata, pos_list, score_name='__score__', copy=True).obs['__score__'].to_numpy()
    neg = np.zeros(adata.n_obs) if not neg_list else sc.tl.score_genes(adata, neg_list, score_name='__score__', copy=True).obs['__score__'].to_numpy()
    scores[ct] = pos - alpha * neg

score_df = pd.DataFrame(scores, index=adata.obs_names)
best_label = score_df.idxmax(axis=1)
best_val   = score_df.max(axis=1)
second_val = score_df.apply(lambda r: np.partition(r.values, -2)[-2] if r.size > 1 else 0.0, axis=1)
margin     = best_val - second_val

# neighbor agreement helper
if 'connectivities' not in adata.obsp:
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
A = adata.obsp['connectivities'].tocsr()

def _neighbor_agreement(labels: pd.Series, A_csr: sp.csr_matrix) -> float:
    lab = labels.astype(str).values
    indptr, indices = A_csr.indptr, A_csr.indices
    agree = 0; total = 0
    for i in range(A_csr.shape[0]):
        nbrs = indices[indptr[i]:indptr[i+1]]
        if nbrs.size == 0: continue
        total += nbrs.size
        agree += (lab[nbrs] == lab[i]).sum()
    return agree / max(total, 1)

base_labels = adata.obs[f"{leiden_key}_named"].astype(str).copy()

ABS_GRID    = np.round(np.arange(0.10, 0.90, 0.05), 2)
MARGIN_GRID = np.round(np.arange(0.05, 0.50, 0.05), 2)

results = []
for abs_min in ABS_GRID:
    for margin_min in MARGIN_GRID:
        mask = (best_val > abs_min) & (margin > margin_min)
        trial = base_labels.copy()
        trial.loc[mask] = best_label.loc[mask]
        nar = _neighbor_agreement(trial, A)
        frac_changed = (trial != base_labels).mean()
        conf_strength = best_val[mask].mean() if mask.any() else 0.0
        unknown_frac = (trial.str.lower() == 'unknown').mean()
        UNKNOWN_WEIGHT = 0.0 if KEEP_UNKNOWN else 0.2
        composite = (1.0 * nar) + (0.3 * conf_strength) - (0.5 * frac_changed) - (UNKNOWN_WEIGHT * unknown_frac)
        results.append((composite, abs_min, margin_min, nar, conf_strength, frac_changed, unknown_frac))

results.sort(reverse=True, key=lambda x: x[0])
ABS_MIN, MARGIN_MIN = results[0][1], results[0][2]
print(f"✅ Tuned thresholds: ABS_MIN={ABS_MIN}, MARGIN_MIN={MARGIN_MIN}")

# save grid table & heatmap
res_df = pd.DataFrame(results, columns=['composite','ABS_MIN','MARGIN_MIN','NAR','mean_conf','changed','unknown'])
res_df.to_csv(os.path.join(os.path.dirname(CSV_FILE_PATH), 'Threshold_tuning_results.csv'), index=False)
try:
    piv = res_df.pivot(index='MARGIN_MIN', columns='ABS_MIN', values='composite')
    plt.figure(figsize=(8, 5))
    im = plt.imshow(piv.values, aspect='auto')
    plt.colorbar(im, label='Composite score')
    plt.xticks(range(piv.shape[1]), [f"{v:.2f}" for v in piv.columns], rotation=45, ha='right')
    plt.yticks(range(piv.shape[0]), [f"{v:.2f}" for v in piv.index])
    plt.xlabel('ABS_MIN'); plt.ylabel('MARGIN_MIN'); plt.title('Threshold tuning grid — composite score')
    out_heat = os.path.join(os.path.dirname(CSV_FILE_PATH), 'Threshold_tuning_heatmap.png')
    plt.tight_layout(); plt.savefig(out_heat, dpi=300, bbox_inches='tight'); plt.close()
    print("✅ Saved threshold tuning heatmap:", out_heat)
except Exception as e:
    print("(heatmap skipped)", e)

# apply re‑annotation
final_labels = base_labels.copy()
confident = (best_val > ABS_MIN) & (margin > MARGIN_MIN)
final_labels.loc[confident] = best_label.loc[confident]

if (not KEEP_UNKNOWN):
    # neighbor voting for remaining Unknowns
    unk_idx = np.where(final_labels.str.lower() == 'unknown')[0]
    if len(unk_idx) > 0:
        print(f"🔄 Resolving {len(unk_idx)} 'Unknown' cells by neighbor voting…")
        for i in unk_idx:
            nn = A.getrow(i).indices
            if nn.size == 0: continue
            neigh = final_labels.iloc[nn]
            neigh = neigh[neigh.str.lower() != 'unknown']
            if neigh.size > 0:
                final_labels.iloc[i] = neigh.mode().iat[0]
    final_labels = final_labels.replace({'Unknown': UNKNOWN_FALLBACK})

adata.obs['cell_type_v2'] = pd.Categorical(final_labels)
if KEEP_UNKNOWN and 'Unknown' in adata.obs['cell_type_v2'].cat.categories:
    cats = [c for c in adata.obs['cell_type_v2'].cat.categories if c != 'Unknown'] + ['Unknown']
    adata.obs['cell_type_v2'] = adata.obs['cell_type_v2'].cat.reorder_categories(cats, ordered=False)
print("✅ Re‑annotation complete.\n", adata.obs['cell_type_v2'].value_counts())

# =========================
# 3) Plots: UMAP + Stacked violin + Dotplot (forced markers)
# =========================
_ensure_umap()

print('🧪 States to plot:', _states())


def plot_umap(group_key: str, tag: str):
    """Plot UMAP colored by the given cell-type grouping and export a high-resolution PNG."""
    pal, order = _palette_for(group_key)
    df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1','UMAP2'])
    df['Cell Type'] = adata.obs[group_key].astype(str).values
    out = os.path.join(os.path.dirname(CSV_FILE_PATH), f"UMAP_{tag}_Leiden_{LEIDEN_RES}.png")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='UMAP1', y='UMAP2', hue='Cell Type', palette=pal,
                    hue_order=order, edgecolor='black', alpha=0.8, s=50, ax=ax)
    ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = {l: h for h, l in zip(handles, labels) if l != 'Cell Type'}
    ordered = [l for l in order if l in label_to_handle]
    leg = ax.legend([label_to_handle[l] for l in ordered], ordered,
                    loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10,
                    title='Cell Type', frameon=False)
    leg.get_title().set_fontweight('bold')
    plt.subplots_adjust(right=0.75, bottom=0.15)
    plt.savefig(out, dpi=600, bbox_inches='tight'); plt.close()
    print(f"✅ UMAP ({tag}) saved to: {out}")

def plot_stacked_violin(group_key: str, tag: str):
    """Rank proteins per group (Wilcoxon) and plot a stacked violin of the top 10 markers."""
    use_raw = getattr(adata, 'raw', None) is not None

    # Exclude Unknown and groups with < 2 cells to avoid Scanpy rank_genes error
    mask_known = adata.obs[group_key].astype(str).str.lower() != 'unknown'
    counts = adata.obs.loc[mask_known, group_key].value_counts()
    keep_groups = counts[counts >= 2].index.tolist()
    if len(keep_groups) < 2:
        print(f"⚠️ Stacked violin skipped for {tag}: fewer than two groups with ≥2 cells.")
        return
    mask = mask_known & adata.obs[group_key].isin(keep_groups)
    ad_view = adata[mask].copy()

    de_key = f"rank_genes_{group_key}"
    sc.tl.rank_genes_groups(ad_view, groupby=group_key, method='wilcoxon', key_added=de_key)
    de_df = sc.get.rank_genes_groups_df(ad_view, group=None, key=de_key)

    # choose the best available sorting column
    sort_col = 'scores' if 'scores' in de_df.columns else ('score' if 'score' in de_df.columns else ('logfoldchanges' if 'logfoldchanges' in de_df.columns else None))
    if sort_col:
        de_df = de_df.sort_values(by=sort_col, ascending=False)

    valid_names = (ad_view.raw.var_names if use_raw and getattr(ad_view, 'raw', None) is not None else ad_view.var_names)
    top_genes: List[str] = []
    for g in de_df['names'].tolist():
        if g in valid_names and g not in top_genes:
            top_genes.append(g)
        if len(top_genes) == 10:
            break
    if not top_genes:
        print(f"⚠️ No valid DE genes found for stacked violin ({tag}).")
        return

    n_groups = ad_view.obs[group_key].nunique()
    fig_w = max(8, 0.8 * len(top_genes)); fig_h = max(4, 0.5 * n_groups)
    sc.pl.stacked_violin(ad_view, var_names=top_genes, groupby=group_key,
                         use_raw=use_raw, standard_scale='var', swap_axes=False,
                         figsize=(fig_w, fig_h), show=False)
    fig = plt.gcf()
    for ax in fig.axes:
        labs = ax.get_xticklabels()
        if labs and any(lbl.get_text() for lbl in labs):
            plt.setp(labs, rotation=45, ha='right')
    out = os.path.join(os.path.dirname(CSV_FILE_PATH), f"StackedViolin_top10_{tag}_Leiden_{LEIDEN_RES}.png")
    plt.tight_layout(); plt.savefig(out, dpi=600, bbox_inches='tight'); plt.close()
    print(f"✅ Stacked violin ({tag}) saved to: {out}")

def plot_dotplot(group_key: str, tag: str):
    """Plot a dotplot for the marker panel (including forced-missing markers) and export SmartDraw-safe PNGs."""
    # Keep marker order; allow abbreviations as group headers
    panel_order: List[str] = []
    for ct, gl in marker_genes.items():
        for g in gl:
            if g not in panel_order:
                panel_order.append(g)

    missing = [g for g in panel_order if g not in adata.var_names]
    # extend matrix with forced‑missing genes so they show as grey/italic labels
    X = adata.X
    if sp.issparse(X):
        zeros = sp.csr_matrix((adata.n_obs, len(missing)), dtype=X.dtype) if missing else None
        X_ext = sp.hstack([X, zeros], format='csr') if missing else X.copy()
    else:
        zeros = np.zeros((adata.n_obs, len(missing)), dtype=X.dtype) if missing else None
        X_ext = np.hstack([X, zeros]) if missing else X.copy()
    var_ext = adata.var.copy()
    if missing:
        add_var = pd.DataFrame(index=missing); add_var['forced_missing'] = True
        var_ext = pd.concat([var_ext, add_var], axis=0)
    var_ext.index = var_ext.index.astype(str)
    tmp = sc.AnnData(X_ext, obs=adata.obs.copy(), var=var_ext)

    # build label map using abbreviations
    ct_order = list(marker_genes.keys())
    abbr_order = [abbr_map.get(ct, ct) for ct in ct_order]
    seen = {}
    abbr_unique = []
    for a in abbr_order:
        seen[a] = seen.get(a, 0) + 1
        abbr_unique.append(a if seen[a] == 1 else f"{a}_{seen[a]-1}")
    plot_markers_abbr = {a: [str(g).strip() for g in marker_genes[ct]] for a, ct in zip(abbr_unique, ct_order)}

    mask = tmp.obs[group_key].astype(str).str.lower() != 'unknown'
    tmp = tmp[mask, :].copy()

    num_markers = sum(len(v) for v in plot_markers_abbr.values())
    fig_height = min(max(num_markers * 0.15, 4), 10)
    sc.pl.dotplot(tmp, plot_markers_abbr, groupby=group_key, use_raw=False,
                  figsize=(32, fig_height), color_map='Reds', dot_max=0.8, dot_min=0.0,
                  standard_scale='var', dendrogram=False, show=False)
    fig = plt.gcf()

    # ---------------------------------------------------------------------
    # Hi-res, SmartDraw-safe export strategy (prevents blur when scaling up)
    # ---------------------------------------------------------------------
    # Export at a large fixed pixel width so the image can be placed at ~100%
    # size in SmartDraw (avoid upscaling, which causes blur).
    DOTPLOT_DPI = 600
    DOTPLOT_TARGET_WIDTH_PX_SMARTDRAW = 7000   # SmartDraw-friendly (reduces blurry proxy rendering)
    DOTPLOT_TARGET_WIDTH_PX_HIRES     = 12000  # Archive hi-res (optional)

    # NOTE: We do NOT permanently scale the figure here. We scale only at save-time
    # to create a SmartDraw-friendly PNG (and optionally a separate hi-res PNG).

    # Force opaque white background (helps SmartDraw import)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    for ax in fig.axes:
        ax.set_facecolor("white")

    forced = set(missing)
    # ---- Font sizes for dotplot (marker genes) ----
    DOTPLOT_GENE_FONTSIZE = 8   # x-axis gene labels
    DOTPLOT_GROUP_FONTSIZE = 9  # y-axis group labels
    DOTPLOT_LABEL_FONTSIZE = 9  # axis labels / titles / headings
    for ax in fig.axes:
        for lab in ax.get_xticklabels():
            gname = lab.get_text()
            if gname in forced:
                lab.set_color('0.45'); lab.set_alpha(0.9); lab.set_fontstyle('italic')
        # Scale axis labels/titles/other text for readability
        ax.tick_params(axis='both', which='major', labelsize=DOTPLOT_LABEL_FONTSIZE)
        ax.xaxis.label.set_size(DOTPLOT_LABEL_FONTSIZE)
        ax.yaxis.label.set_size(DOTPLOT_LABEL_FONTSIZE)
        ax.title.set_fontsize(DOTPLOT_LABEL_FONTSIZE)
        for _t in ax.texts:
            _t.set_fontsize(DOTPLOT_LABEL_FONTSIZE)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=DOTPLOT_GENE_FONTSIZE)
        plt.setp(ax.get_yticklabels(), fontsize=DOTPLOT_GROUP_FONTSIZE)
    for _t in fig.texts:
        _t.set_fontsize(DOTPLOT_LABEL_FONTSIZE)

    # Save dotplot as PNG(s) optimized for SmartDraw and for archiving
    base_dir = os.path.dirname(CSV_FILE_PATH)
    out_smart = os.path.join(base_dir, f"Dotplot_{tag}_Leiden_{LEIDEN_RES}_SmartDraw.png")
    out_hires = os.path.join(base_dir, f"Dotplot_{tag}_Leiden_{LEIDEN_RES}_HiRes.png")

    # Helper: scale figure to a target *pixel width* at a fixed DPI, then save as opaque RGB PNG
    def _save_png_target(fig, out_path, target_width_px, dpi):
        try:
            from PIL import Image
        except Exception:
            Image = None

        w0, h0 = fig.get_size_inches()
        try:
            target_w_in = float(target_width_px) / float(dpi)
            if w0 and w0 > 0:
                s = target_w_in / w0
                fig.set_size_inches(w0 * s, h0 * s)

            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)

            # Normalize to standard RGB PNG (removes alpha/palette quirks that can break SmartDraw)
            if Image is not None:
                im = Image.open(out_path)
                if im.mode != "RGB":
                    im_rgba = im.convert("RGBA")
                    bg = Image.new("RGBA", im_rgba.size, (255, 255, 255, 255))
                    bg.alpha_composite(im_rgba)
                    im_rgb = bg.convert("RGB")
                else:
                    im_rgb = im
                # Embed DPI metadata for apps that respect it
                im_rgb.save(out_path, format="PNG", optimize=True, dpi=(dpi, dpi))
        finally:
            fig.set_size_inches(w0, h0)

        if Image is not None:
            im2 = Image.open(out_path)
            print(f"ℹ️ {os.path.basename(out_path)}: {im2.size[0]}×{im2.size[1]} px @ {dpi} dpi")

    # SmartDraw-friendly: large enough to be crisp, but not so huge that SmartDraw switches
    # to a low-res proxy renderer.
    _save_png_target(fig, out_smart, DOTPLOT_TARGET_WIDTH_PX_SMARTDRAW, DOTPLOT_DPI)

    # Optional archive hi-res PNG
    _save_png_target(fig, out_hires, DOTPLOT_TARGET_WIDTH_PX_HIRES, DOTPLOT_DPI)

    plt.close(fig)
    print(f"✅ Dotplot ({tag}) saved to: {out_smart}")
    print(f"✅ Dotplot ({tag}) saved to: {out_hires}")


for grp, tag in _states():
    plot_umap(grp, tag)
    plot_stacked_violin(grp, tag)
    plot_dotplot(grp, tag)

# =========================
# 4) Sensitivity & Dynamic Range
# =========================

def _detect_intensity_scale() -> str:
    X = adata.raw.X if getattr(adata, 'raw', None) is not None else adata.X
    vals = np.asarray(X.data if sp.issparse(X) else np.asarray(X)).ravel()
    vals = vals[np.isfinite(vals)]; vals = vals[vals != 0]
    if vals.size == 0: return 'linear'
    med = np.nanmedian(vals); has_neg = np.nanmin(vals) < 0
    return 'log' if (has_neg or med < 50) else 'linear'


def compute_cell_metrics():
    """Compute per-cell sensitivity (IDs) and dynamic-range metrics and write a per-cell CSV table."""
    scale = _detect_intensity_scale()
    adata.uns['intensity_scale_detected'] = scale
    X = adata.raw.X if getattr(adata, 'raw', None) is not None else adata.X
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)
    sens = np.zeros(adata.n_obs, dtype=int)
    dr   = np.full(adata.n_obs, np.nan)
    p05s = np.full(adata.n_obs, np.nan)
    p95s = np.full(adata.n_obs, np.nan)
    for i in range(adata.n_obs):
        row = (X.getrow(i).toarray().ravel() if sp.issparse(X) else np.asarray(X[i, :]).ravel())
        vals = row[np.isfinite(row)]; vals = vals[vals != 0]
        sens[i] = int(vals.size)
        if vals.size >= 10:
            if scale == 'linear':
                pos = vals[vals > 0]
                if pos.size >= 10:
                    q05, q95 = np.nanpercentile(pos, 5), np.nanpercentile(pos, 95)
                    if q95 > 0 and q05 > 0:
                        p05s[i], p95s[i] = np.log10(q05), np.log10(q95)
                        dr[i] = p95s[i] - p05s[i]
            else:
                q05, q95 = np.nanpercentile(vals, 5), np.nanpercentile(vals, 95)
                p05s[i], p95s[i] = q05, q95
                dr[i] = q95 - q05
    adata.obs['sens_n_proteins'] = sens
    adata.obs['dynr_value'] = dr
    adata.obs['dynr_low']   = p05s
    adata.obs['dynr_high']  = p95s
    adata.obs['dynr_scale'] = scale
    out_pc = os.path.join(os.path.dirname(CSV_FILE_PATH), f"SENS_DYNR_per_cell_Leiden_{LEIDEN_RES}.csv")
    cols = ['sens_n_proteins','dynr_value','dynr_low','dynr_high','dynr_scale']
    for col in ['cell_type_pre','cell_type_v2']:
        if col in adata.obs.columns: cols.append(col)
    pd.DataFrame(adata.obs[cols]).to_csv(out_pc, index=True)
    print(f"✅ Saved per‑cell sensitivity/dynamic range table: {out_pc}")


def summarize_by(group_key: str, tag: str) -> pd.DataFrame | None:
    df = adata.obs[[group_key, 'sens_n_proteins', 'dynr_value']].copy()
    df = df[df[group_key].astype(str).str.lower() != 'unknown']
    df = df.dropna(subset=['sens_n_proteins', 'dynr_value'])
    if df.empty:
        print(f"⚠️ No data for {tag} summary."); return None
    df[group_key] = df[group_key].astype('category').cat.remove_unused_categories()

    def _iqr(x: pd.Series) -> float:
        a = pd.to_numeric(x, errors='coerce').dropna().to_numpy()
        if a.size == 0: return float('nan')
        return np.nanpercentile(a, 75) - np.nanpercentile(a, 25)

    agg = (df.groupby(group_key, observed=True, sort=False)
             .agg(n_cells=('sens_n_proteins','size'),
                  sens_median=('sens_n_proteins','median'),
                  sens_iqr=('sens_n_proteins', _iqr),
                  dr_median=('dynr_value','median'),
                  dr_iqr=('dynr_value', _iqr))
             .reset_index().rename(columns={group_key: 'cell_type'}))
    if agg.empty:
        print(f"⚠️ No groups with valid data for {tag} summary."); return None
    out = os.path.join(os.path.dirname(CSV_FILE_PATH), f"SENS_DYNR_by_celltype_{tag}_Leiden_{LEIDEN_RES}.csv")
    agg.to_csv(out, index=False)
    print(f"✅ Saved by‑celltype ({tag}) summary: {out}")
    return agg


def plot_umap_metrics():
    """Plot UMAPs colored by sensitivity and dynamic-range metrics."""
    for metric, label in [('sens_n_proteins','Sensitivity (IDs per cell)'),
                          ('dynr_value','Dynamic range')]:
        for grp, tag in _states():
            out = os.path.join(os.path.dirname(CSV_FILE_PATH), f"UMAP_{metric}_{tag}_Leiden_{LEIDEN_RES}.png")
            fig, ax = plt.subplots(figsize=(10, 8))
            df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1','UMAP2'])
            df[metric] = adata.obs[metric].values
            im = ax.scatter(df['UMAP1'], df['UMAP2'], c=df[metric], s=28, cmap='viridis', rasterized=True)
            ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label(label)
            plt.tight_layout(); plt.savefig(out, dpi=600, bbox_inches='tight'); plt.close()
            print(f"✅ UMAP metric '{metric}' ({tag}) saved to: {out}")


def plot_group_distributions():
    """Plot violin distributions of sensitivity and dynamic range by cell type."""
    for metric, yl in [('sens_n_proteins','IDs per cell'), ('dynr_value','Dynamic range')]:
        for grp, tag in _states():
            pal, order = _palette_for(grp)
            data = adata.obs[[grp, metric]].copy()
            data = data[data[grp].astype(str).str.lower() != 'unknown']
            if data.empty: continue
            out = os.path.join(os.path.dirname(CSV_FILE_PATH), f"Violin_{metric}_{tag}_Leiden_{LEIDEN_RES}.png")
            fig, ax = plt.subplots(figsize=(max(8, 0.6*len(order)), 4.8))
            sns.violinplot(data=data, x=grp, y=metric, order=order, palette=pal,
                           inner='box', cut=0, linewidth=0.8, ax=ax)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_xlabel('Cell type'); ax.set_ylabel(yl)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout(); plt.savefig(out, dpi=600, bbox_inches='tight'); plt.close()
            print(f"✅ Violin '{metric}' ({tag}) saved to: {out}")


def plot_rank_bars():
    """Plot ranked bar charts of median sensitivity and median dynamic range by cell type."""
    for metric, yl in [('sens_n_proteins','Median IDs per cell'), ('dynr_value','Median dynamic range')]:
        for grp, tag in _states():
            agg = summarize_by(grp, tag)
            if agg is None or agg.empty: continue
            pal, order = _palette_for(grp)
            sort_col = 'sens_median' if metric == 'sens_n_proteins' else 'dr_median'
            agg = agg.dropna(subset=[sort_col])
            if agg.empty: continue
            agg_sorted = agg.sort_values(by=sort_col, ascending=False)
            out = os.path.join(os.path.dirname(CSV_FILE_PATH), f"BarRank_{metric}_{tag}_Leiden_{LEIDEN_RES}.png")
            fig, ax = plt.subplots(figsize=(max(6, 0.6*len(agg_sorted)), 4.8))
            colors = [pal.get(ct, '#aaaaaa') for ct in agg_sorted['cell_type']]
            ax.bar(agg_sorted['cell_type'], agg_sorted[sort_col].to_numpy(float),
                   edgecolor='black', linewidth=0.6, color=colors)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel(yl); ax.set_xlabel('Cell type')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            plt.tight_layout(); plt.savefig(out, dpi=600, bbox_inches='tight'); plt.close()
            print(f"✅ Ranked bar '{metric}' ({tag}) saved to: {out}")


compute_cell_metrics()
plot_umap_metrics()
plot_group_distributions()
plot_rank_bars()
for grp, tag in _states():
    summarize_by(grp, tag)

# =========================
# 5) Protein IDs per cell type: union & coverage + median per cell
# =========================

def _get_detection_matrix():
    """Return a binary (detected / not detected) matrix based on the stored intensity matrix."""
    X = adata.raw.X if getattr(adata, 'raw', None) is not None else adata.X
    if sp.issparse(X):
        return (X > 0).astype('int8')
    arr = np.asarray(X)
    return (arr > 0).astype('int8')


def protein_ids_by_celltype(group_key: str, tag: str, coverage_fracs: Tuple[float, ...] = (0.10, 0.25, 0.50)):
    """Compute union/coverage protein IDs per cell type and export summary tables and bar charts."""
    if group_key not in adata.obs.columns:
        print(f"⚠️ '{group_key}' not in adata.obs; skipping protein‑ID‑by‑celltype ({tag})."); return
    pal, order = _palette_for(group_key)
    obs = adata.obs[[group_key]].copy()
    obs = obs[obs[group_key].astype(str).str.lower() != 'unknown']
    if obs.empty: return
    obs[group_key] = obs[group_key].astype('category').cat.remove_unused_categories()

    X_bin = _get_detection_matrix()
    idx_obs = pd.Index(adata.obs_names)
    keep_ilocs = idx_obs.get_indexer(obs.index)
    X_sub = X_bin[keep_ilocs, :]

    rows = []
    for ct in obs[group_key].cat.categories:
        idx = obs.index[obs[group_key] == ct]
        ilocs_in_obs = pd.Index(obs.index).get_indexer(idx)
        X_ct = X_sub[ilocs_in_obs, :]
        if sp.issparse(X_ct):
            union_vec = (X_ct.max(axis=0)).A1
            col_sums = np.asarray(X_ct.sum(axis=0)).ravel()
        else:
            union_vec = X_ct.max(axis=0)
            col_sums = X_ct.sum(axis=0)
        union_count = int(np.asarray(union_vec).ravel().sum())
        n_cells = len(idx)
        cov_counts = {}
        for frac in coverage_fracs:
            thr = int(np.ceil(frac * n_cells))
            cov_counts[f'coverage_ge_{int(frac*100)}pct'] = int((np.asarray(col_sums) >= thr).sum())
        rows.append({'cell_type': ct, 'n_cells': n_cells, 'union_ids': union_count, **cov_counts})

    df_union = pd.DataFrame(rows)
    if 'sens_n_proteins' in adata.obs.columns:
        med = (adata.obs.loc[obs.index, [group_key, 'sens_n_proteins']]
                  .groupby(group_key, observed=True)['sens_n_proteins']
                  .median().rename('median_per_cell_IDs').reset_index())
        df_union = df_union.merge(med, left_on='cell_type', right_on=group_key, how='left').drop(columns=[group_key])
    else:
        df_union['median_per_cell_IDs'] = np.nan

    out_csv = os.path.join(os.path.dirname(CSV_FILE_PATH), f"ProteinIDs_byCellType_{tag}_Leiden_{LEIDEN_RES}.csv")
    df_union.sort_values('cell_type').to_csv(out_csv, index=False)
    print(f"✅ Saved: {out_csv}")

    # plots
    order_ct = [c for c in order if c in set(df_union['cell_type'])] or df_union['cell_type'].tolist()
    colors = [pal.get(ct, '#aaaaaa') for ct in order_ct]

    def _bar(vals, ylabel, fname):
        out = os.path.join(os.path.dirname(CSV_FILE_PATH), fname)
        fig, ax = plt.subplots(figsize=(max(6, 0.6*len(order_ct)), 4.8))
        ax.bar(order_ct, vals, edgecolor='black', linewidth=0.6, color=colors)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel(ylabel); ax.set_xlabel('Cell type')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); plt.savefig(out, dpi=600, bbox_inches='tight'); plt.close()
        print(f"✅ Saved: {out}")

    _bar([int(df_union.loc[df_union['cell_type']==ct, 'union_ids'].values[0]) for ct in order_ct],
         'Proteins identified (union across cells)',
         f"Bar_ProteinIDs_byCellType_UNION_{tag}_Leiden_{LEIDEN_RES}.png")

    cov_cols = [c for c in df_union.columns if c.startswith('coverage_ge_')]
    if cov_cols:
        target = None
        for pref in ['coverage_ge_50pct','coverage_ge_25pct','coverage_ge_10pct']:
            if pref in cov_cols: target = pref; break
        target = target or sorted(cov_cols)[-1]
        _bar([int(df_union.loc[df_union['cell_type']==ct, target].values[0]) for ct in order_ct],
             target.replace('coverage_ge_','≥').replace('pct','%') + ' of cells',
             f"Bar_ProteinIDs_byCellType_COVERAGE_{tag}_Leiden_{LEIDEN_RES}.png")

    _bar([float(df_union.loc[df_union['cell_type']==ct, 'median_per_cell_IDs'].values[0]) for ct in order_ct],
         'Median proteins identified per cell',
         f"Bar_ProteinIDs_byCellType_MEDIAN_perCell_{tag}_Leiden_{LEIDEN_RES}.png")


for grp, tag in _states():
    protein_ids_by_celltype(grp, tag)

# =========================
# 6) Core / Unique core tables + UpSet combos + Core Jaccard heatmap
# =========================

def _compute_coverage_matrix(group_key: str) -> Tuple[List[str], np.ndarray]:
    if group_key not in adata.obs.columns:
        raise KeyError(f"{group_key} not in adata.obs")
    obs = adata.obs[[group_key]].copy()
    obs = obs[obs[group_key].astype(str).str.lower() != 'unknown']
    if not len(obs):
        raise ValueError("No cells after excluding 'Unknown'")
    obs[group_key] = obs[group_key].astype('category').cat.remove_unused_categories()

    X_bin = _get_detection_matrix()
    keep_ilocs = pd.Index(adata.obs_names).get_indexer(obs.index)
    X_sub = X_bin[keep_ilocs, :]

    types = list(obs[group_key].cat.categories)
    n_types, n_prot = len(types), adata.n_vars
    coverage = np.zeros((n_types, n_prot), dtype=float)
    for i, ct in enumerate(types):
        idx_ct = obs.index[obs[group_key] == ct]
        ct_ilocs = pd.Index(obs.index).get_indexer(idx_ct)
        X_ct = X_sub[ct_ilocs, :]
        n_cells = len(idx_ct)
        if n_cells == 0: continue
        col_sums = (np.asarray(X_ct.sum(axis=0)).ravel() if sp.issparse(X_ct) else X_ct.sum(axis=0))
        coverage[i, :] = col_sums / float(n_cells)
    return types, coverage


def _symbol_column() -> str | None:
    for c in ['gene_symbol','Gene Symbol','gene','Gene','Symbol','symbol','GeneName','GENE_SYMBOL']:
        if c in adata.var.columns: return c
    return None


def write_core_unique_tables(group_key: str, tag: str, core_frac: float, low_frac: float):
    types, coverage = _compute_coverage_matrix(group_key)
    n_types, n_prot = coverage.shape
    core_masks = coverage >= core_frac
    unique_core_masks = np.zeros_like(core_masks, dtype=bool)
    for i in range(n_types):
        in_i = core_masks[i, :]
        low_else = np.all(coverage[np.arange(n_types)!=i, :] < low_frac, axis=0) if n_types>1 else np.ones(n_prot, bool)
        unique_core_masks[i, :] = in_i & low_else

    var_names = np.array(adata.var_names)
    sym_col = _symbol_column()
    symbol_vals = np.array(adata.var[sym_col].astype(str)) if sym_col else None

    def _mk(ids: np.ndarray) -> pd.DataFrame:
        if ids.size == 0:
            return pd.DataFrame(columns=['protein_id'] + ([sym_col] if sym_col else []))
        df = pd.DataFrame({'protein_id': ids})
        if sym_col:
            take = np.flatnonzero(np.isin(var_names, ids))
            df[sym_col] = symbol_vals[take]
        sort_cols = ([sym_col] if sym_col else []) + ['protein_id']
        return df.sort_values(sort_cols).reset_index(drop=True)

    summary = []
    for i, ct in enumerate(types):
        core_ids = var_names[core_masks[i, :]]
        uniq_ids = var_names[unique_core_masks[i, :]]
        suffix = f"core{int(core_frac*100)}_low{int(low_frac*100)}"
        out_core = os.path.join(os.path.dirname(CSV_FILE_PATH), f"Core_Proteins_{ct}_{tag}_{suffix}_Leiden_{LEIDEN_RES}.csv")
        out_uniq = os.path.join(os.path.dirname(CSV_FILE_PATH), f"UniqueCore_Proteins_{ct}_{tag}_{suffix}_Leiden_{LEIDEN_RES}.csv")
        _mk(core_ids).to_csv(out_core, index=False)
        _mk(uniq_ids).to_csv(out_uniq, index=False)
        print("✅ Saved:", out_core)
        print("✅ Saved:", out_uniq)
        summary.append({'cell_type': ct, 'n_core': int(core_masks[i, :].sum()), 'n_unique_core': int(unique_core_masks[i, :].sum())})

    df_sum = pd.DataFrame(summary)
    out_sum = os.path.join(os.path.dirname(CSV_FILE_PATH), f"CoreUnique_Summary_{tag}_core{int(core_frac*100)}_low{int(low_frac*100)}_Leiden_{LEIDEN_RES}.csv")
    df_sum.to_csv(out_sum, index=False)
    print("✅ Saved:", out_sum)

def plot_upset_core(group_key: str, tag: str, core_frac: float, topn: int = 20, protein_names=None):
    """
    Creates the UpSet-style bar chart (exclusive intersections) and writes:
      - <...>_Top{topn}_... .png    # the figure (unchanged)
      - <...>_Top{topn}_... .csv    # all intersections + counts (unchanged if you already save it)
      - <...>_Top{topn}_..._members.csv  # NEW: proteins contributing to the plotted Top N bars
    """
    # --- build coverage matrix: rows=cell types, cols=proteins; values in [0..1] = frac cells with ID ---
    types, coverage = _compute_coverage_matrix(group_key)   # unchanged in your code
    n_types, n_prot = coverage.shape

    # protein name fallback
    if protein_names is None:
        try:
            protein_names = adata.var_names.tolist()  # works if you built coverage from adata
        except Exception:
            protein_names = [f"Protein_{i+1}" for i in range(n_prot)]

    # core membership mask
    core_masks = (coverage >= core_frac)

    # exclusive combo key per protein (exact set of types where it's core)
    combos = []
    for p in range(n_prot):
        members = [types[i] for i in range(n_types) if core_masks[i, p]]
        combos.append(';'.join(members) if members else '(none)')

    # count intersections and pick Top N (excluding '(none)')
    s = pd.Series(combos, name='combo').value_counts().reset_index()
    s.columns = ['combo', 'count']
    s = s[s['combo'] != '(none)']
    s_top = s.nlargest(topn, 'count')

    # --- helper: abbreviate a combo label with your abbr_map (defined earlier in your script) ---
    def _abbr_combo(combo: str) -> str:
        parts = [t for t in combo.split(';') if t]
        return ' + '.join([abbr_map.get(t, t) for t in parts])  # fall back to full name if not in map

    # OPTIONAL: also save the intersection table with abbreviated labels (useful for review/QC)
    s_with_abbr = s.copy()
    s_with_abbr['combo_abbr'] = s_with_abbr['combo'].apply(_abbr_combo)
    out_csv = os.path.join(os.path.dirname(CSV_FILE_PATH),
                           f"UpSet_CoreCombination_{tag}_core{int(core_frac*100)}_Leiden_{LEIDEN_RES}.csv")
    s_with_abbr.to_csv(out_csv, index=False)
    print("✅ Saved intersections:", out_csv)

    # --- NEW: write the protein members that feed the plotted Top N bars ---
    top_set = set(s_top['combo'])
    rows = []
    for p in range(n_prot):
        combo_full = combos[p]
        if combo_full in top_set:  # only what appears in the figure
            parts = combo_full.split(';') if combo_full else []
            row = {
                "Protein": protein_names[p],
                "combo_full": combo_full,
                "combo_abbr": _abbr_combo(combo_full),
                "n_types": len(parts),
            }
            # 0/1 flags per type for easy filtering
            for t in types:
                row[abbr_map.get(t, t)] = 1 if t in parts else 0
            rows.append(row)

    members_df = pd.DataFrame(rows).sort_values(
        ["n_types", "combo_abbr", "Protein"], ascending=[True, True, True]
    )
    out_members = os.path.join(
        os.path.dirname(CSV_FILE_PATH),
        f"UpSet_CoreCombination_{tag}_core{int(core_frac*100)}_Top{topn}_Leiden_{LEIDEN_RES}_members.csv"
    )
    members_df.to_csv(out_members, index=False)
    print("✅ Saved members:", out_members)

    # --- plot (with abbreviations on the x-axis) ---
    labels_abbr = s_top['combo'].apply(_abbr_combo).values
    fig, ax = plt.subplots(figsize=(max(6, 0.45*len(s_top)), 4.8))
    ax.bar(range(len(s_top)), s_top['count'].values, edgecolor='black', linewidth=0.6)
    ax.set_xticks(range(len(s_top)))
    ax.set_xticklabels(labels_abbr, rotation=45, ha='right')
    ax.set_ylabel('Proteins (exact core-set membership)')
#   ax.set_xlabel(f'Core combinations (≥{int(core_frac*100)}% within each type)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(CSV_FILE_PATH),
                           f"UpSet_CoreCombination_{tag}_core{int(core_frac*100)}_Top{topn}_Leiden_{LEIDEN_RES}.png")
    plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close()
    print("✅ Saved figure:", out_png)


def plot_core_jaccard(group_key: str, tag: str, core_frac: float):
    """Plot pairwise Jaccard similarity between cell-type *core* protein sets.

    Definitions (as implemented in this script)
    -----------------------------------------
    1) For each cell type, a protein is considered *core* if it is detected in
       at least `core_frac` of cells within that type (presence/absence based).

    2) For two cell types i and j with core sets Aᵢ and Aⱼ, the Jaccard index is:

           J(Aᵢ, Aⱼ) = |Aᵢ ∩ Aⱼ| / |Aᵢ ∪ Aⱼ|

       Range: 0–1 (1 = identical core sets; 0 = no shared core proteins).
       The diagonal is 1 by definition. The matrix is symmetric, so the upper
       triangle duplicates the lower triangle.

    Plotting choices
    ----------------
    - Square figure (same height and width) for consistent geometry.
    - Only the bottom-left triangle (incl. diagonal) is shown; the upper-right
      triangle is masked to avoid duplicating information.
    """
    types, coverage = _compute_coverage_matrix(group_key)
    core_masks = coverage >= core_frac
    n_types = len(types)

    # Compute Jaccard matrix
    jacc = np.zeros((n_types, n_types), float)
    for i in range(n_types):
        Ai = core_masks[i, :]
        for j in range(n_types):
            Aj = core_masks[j, :]
            inter = int(np.logical_and(Ai, Aj).sum())
            union = int(np.logical_or(Ai, Aj).sum())
            jacc[i, j] = (inter / union) if union > 0 else 0.0

    # Mask upper triangle (excluding diagonal) to keep only the bottom-left half
    mask = np.triu(np.ones_like(jacc, dtype=bool), k=1)
    jacc_ma = np.ma.array(jacc, mask=mask)

    # Colormap with a clean background for masked cells
    cmap = plt.get_cmap('viridis')
    try:
        cmap = cmap.copy()
    except Exception:
        pass
    try:
        cmap.set_bad(color='white')
    except Exception:
        pass

    side = max(6.0, 0.55 * n_types)  # square figure
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(jacc_ma, aspect='equal', cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n_types)); ax.set_yticks(range(n_types))
    ax.set_xticklabels(types, rotation=45, ha='right')
    ax.set_yticklabels(types)
    
    # Remove top/right spines so the masked (empty) triangle has no outer border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)

#   ax.set_title(f'Core‑set Jaccard (≥{int(core_frac*100)}%)')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Jaccard index')

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(CSV_FILE_PATH),
                           f"Heatmap_Core_Jaccard_{tag}_core{int(core_frac*100)}_Leiden_{LEIDEN_RES}.png")
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close()
    print("✅ Saved:", out_png)


for grp, tag in _states():
    write_core_unique_tables(grp, tag, CORE_FRAC, UNIQUE_LOW_FRAC)
    plot_upset_core(grp, tag, CORE_FRAC, UPSET_TOPN)
    plot_core_jaccard(grp, tag, CORE_FRAC)

# =========================
# 7) HARD FIX: recompute per cell IDs from raw CSV (NaN-aware) and re-plot
# =========================

def _find_source_csv() -> str | None:
    # Priority: explicit uns; else largest CSV in folder
    for k in ['source_csv','raw_csv','input_csv','matrix_csv','expr_csv','proteomics_csv']:
        v = adata.uns.get(k, None) if hasattr(adata, 'uns') else None
        if isinstance(v, str) and os.path.isfile(v):
            return v
    folder = os.path.dirname(CSV_FILE_PATH)
    csvs = sorted(glob.glob(os.path.join(folder, '*.csv')), key=lambda x: os.path.getsize(x), reverse=True)
    return csvs[0] if csvs else None


def _compute_ids_from_csv(csv_path: str, treat_zero_as_missing: bool=True) -> pd.Series:
    df = pd.read_csv(csv_path, low_memory=False)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols and df.shape[1] > 3:
        num_cols = list(df.columns[3:])
    if not num_cols:
        raise ValueError('Could not identify numeric cell columns in CSV.')
    mat = df[num_cols]
    det = (mat.notna().to_numpy() & (mat.to_numpy() != 0)) if treat_zero_as_missing else mat.notna().to_numpy()
    counts = np.sum(det, axis=0)
    return pd.Series(counts, index=num_cols, name='sens_n_proteins')


def _overwrite_obs_counts_from_series(s: pd.Series):
    """Overwrite `adata.obs['sens_n_proteins']` from a Series keyed by cell column names."""
    s.index = s.index.astype(str)
    obs_names = pd.Index(adata.obs_names.astype(str))
    aligned = s.reindex(obs_names)
    if aligned.isna().all():
        lc_map = {k.lower(): v for k, v in s.items()}
        aligned = obs_names.to_series().str.lower().map(lc_map)
    adata.obs['sens_n_proteins'] = aligned.fillna(0).astype(int).values


def _plot_sens_per_celltype(group_key: str, tag: str):
    """Re-plot sensitivity distributions and UMAP gradient using the recomputed per-cell ID counts."""
    if group_key not in adata.obs.columns or 'sens_n_proteins' not in adata.obs.columns:
        return
    pal, order = _palette_for(group_key)
    obs = adata.obs[[group_key, 'sens_n_proteins']].copy()
    unique_types = [c for c in order if c in set(obs[group_key])] or list(obs[group_key].unique())

    # BarRank (median)
    med = (obs.groupby(group_key)['sens_n_proteins'].median().reindex(unique_types))
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(unique_types)), 4.8))
    colors = [pal.get(ct, '#aaaaaa') for ct in unique_types]
    ax.bar(unique_types, med.values, edgecolor='black', linewidth=0.6, color=colors)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Median IDs per cell'); ax.set_xlabel('Cell type')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    out_png = os.path.join(os.path.dirname(CSV_FILE_PATH), f"BarRank_sens_n_proteins_{tag}_Leiden_{LEIDEN_RES}.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close(); print("✅ Saved:", out_png)

    # Violin
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(unique_types)), 4.8))
    sns.violinplot(data=obs, x=group_key, y='sens_n_proteins', order=unique_types,
                   palette=[pal.get(ct, '#aaaaaa') for ct in unique_types],
                   cut=0, inner='quartile', linewidth=0.6)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('Protein IDs per cell'); ax.set_xlabel('Cell type')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    out_png = os.path.join(os.path.dirname(CSV_FILE_PATH), f"Violin_sens_n_proteins_{tag}_Leiden_{LEIDEN_RES}.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close(); print("✅ Saved:", out_png)

    # UMAP gradient
    if 'X_umap' in adata.obsm:
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        sca = ax.scatter(adata.obsm['X_umap'][:,0], adata.obsm['X_umap'][:,1],
                         c=adata.obs['sens_n_proteins'].values, s=6, alpha=0.9, linewidths=0)
        cbar = plt.colorbar(sca); cbar.set_label('IDs per cell')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('UMAP‑1'); ax.set_ylabel('UMAP‑2')
        out_png = os.path.join(os.path.dirname(CSV_FILE_PATH), f"UMAP_sens_n_proteins_{tag}_Leiden_{LEIDEN_RES}.png")
        plt.tight_layout(); plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close(); print("✅ Saved:", out_png)


def recompute_and_replot_ids_per_cell():
    """Recompute per-cell IDs from the source CSV (preferred) and regenerate sensitivity plots."""
    csv_path = _find_source_csv()
    if csv_path is None:
        print("⚠️ Could not locate source CSV automatically; falling back to AnnData‑based detection.")
        Xcand = None
        if getattr(adata, 'layers', None):
            for k in ['raw','intensity_raw','counts','intensity','data']:
                if k in adata.layers:
                    L = adata.layers[k]
                    arr = L.toarray() if sp.issparse(L) else np.asarray(L)
                    if np.isnan(arr).any() or (arr == 0).any():
                        Xcand = arr; break
        if Xcand is None:
            Xcand = adata.raw.X if getattr(adata,'raw',None) is not None else adata.X
            Xcand = Xcand.toarray() if sp.issparse(Xcand) else np.asarray(Xcand)
        det = np.isfinite(Xcand) & (Xcand != 0)
        counts = det.sum(axis=1) if Xcand.shape[0] == adata.n_obs else det.sum(axis=0)
        if counts.shape[0] != adata.n_obs and det.T.shape[0] == adata.n_obs:
            counts = det.T.sum(axis=1)
        adata.obs['sens_n_proteins'] = counts.astype(int)
    else:
        print("ℹ️ Using raw CSV for per‑cell IDs:", os.path.basename(csv_path))
        s = _compute_ids_from_csv(csv_path, treat_zero_as_missing=True)
        _overwrite_obs_counts_from_series(s)

    for grp, tag in _states():
        _plot_sens_per_celltype(grp, tag)


try:
    recompute_and_replot_ids_per_cell()
except Exception as e:
    print("⚠️ Recompute/Replot per‑cell IDs failed:", e)

# =========================

# =========================
# IDs-per-cell Histogram (100-protein bins, same style as DCT script)
# =========================
def plot_ids_histogram(bin_width: int = 100):
    """Plot a histogram of proteins detected per cell (bin width configurable) and export PNG."""
    try:
        import numpy as _np
        import matplotlib.pyplot as _plt
        import os as _os
    except Exception as _e:
        print("⚠️ Missing deps for histogram:", _e)
        return
    if 'sens_n_proteins' not in adata.obs.columns:
        print("⚠️ 'sens_n_proteins' not found; run recompute_and_replot_ids_per_cell() first.")
        return
    counts = adata.obs['sens_n_proteins'].to_numpy()
    if counts.size == 0:
        print("⚠️ No values in 'sens_n_proteins'; skipping histogram.")
        return
    maxc = int(_np.ceil(counts.max() / bin_width) * bin_width) if counts.max() > 0 else bin_width
    bins = _np.arange(0, maxc + bin_width, bin_width)
    _plt.figure(figsize=(6, 4.5), dpi=600)
    _plt.hist(counts, bins=bins, edgecolor="black")
    _plt.xlabel("Proteins detected per cell", fontsize=12)
    _plt.ylabel("Number of cells", fontsize=12)
    _plt.title("Sensitivity (IDs per cell)", fontsize=12, fontweight="bold")
    _plt.tight_layout()
    out = _os.path.join(_os.path.dirname(CSV_FILE_PATH), "Histogram_ProteinsDetected_perCell.png")
    _plt.savefig(out, dpi=600, bbox_inches='tight'); _plt.close()
    print(f"✅ Histogram saved → {out}")

# Try to render after recompute (uses CSV-aware counts)
try:
    plot_ids_histogram(100)
except Exception as _e:
    print("⚠️ Histogram plot failed:", _e)

# ---------- END ----------
# =========================