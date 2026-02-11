# ----------------------------------------------------------------------------------------
# Written by Qi Wu, Department of Biomedicine, Aarhus University, with the aid of ChatGPT
# Date: 11/Feb/2026
# Contact Qi Wu (qi.wu@biomed.au.dk) for information and comments
# ----------------------------------------------------------------------------------------

# --- Imports (consolidated) ---
from typing import Dict, List
from matplotlib.collections import PathCollection
import os, time, json, math
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import transforms as mtransforms
from pathlib import Path
# --- End imports ---

# --- compatibility aliases for merged blocks ---
_np = np
_pd = pd
_os = os
_os_core = os
_sp = sp
_pearsonr = st.pearsonr
_plt = plt
_sns = sns

# Set random seed for reproducibility
np.random.seed(42)

# Load your data
csv_file_path = r"PATH\TO\DCT_758cells.csv"
data = pd.read_csv(csv_file_path)

# 📌 **Get the folder path automatically from the CSV file**
data_folder = os.path.dirname(csv_file_path)  # Extract the directory

print(f"📂 Data folder detected: {data_folder}")
# 📌 **Load Marker Genes from Excel File**
marker_genes_file = r"PATH\TO\DCT_markers.xlsx"
marker_df = pd.read_excel(marker_genes_file)

# Convert marker genes from a comma-separated string to a list
marker_genes = {
    row[0]: row[1].split(",") for row in marker_df.itertuples(index=False)
}

# Convert gene names to uppercase
marker_genes = {cell_type: [gene.strip().lower().capitalize() for gene in genes] for cell_type, genes in marker_genes.items()}

# Separate metadata and expression data
metadata = data.iloc[:, :3]  # First three columns as metadata
expression_data = data.iloc[:, 3:]  # Remaining columns are expression values

# Ensure all values are float32 (to avoid integer overflow)
expression_data = expression_data.fillna(0).astype(np.float32)

# Transpose if necessary (e.g., proteins as rows, cells as columns)
expression_data = expression_data.T

# Convert to AnnData format
adata = sc.AnnData(expression_data)

# Ensure gene names are properly extracted from metadata['PG.Genes']
if 'PG.Genes' in metadata.columns:
    genes = metadata['PG.Genes'].astype(str).str.split(";").str[0].str.strip().fillna("Unknown")
else:
    print("⚠️ Column 'PG.Genes' not found in metadata. Check column names.")
    genes = pd.Series(["Unknown"] * adata.shape[1])  # Fallback

# Assign gene names to adata.var
adata.var['gene_name'] = genes[:adata.n_vars].values

# Ensure proper format of gene names
adata.var_names = adata.var["gene_name"].astype(str).str.lower().str.capitalize()

# Fix duplicate gene names by making them unique
if adata.var_names.duplicated().any():
    print("⚠️ Duplicate gene names found. Making them unique...")
    adata.var_names = pd.Index([f"{name}_{i}" if name in adata.var_names[:i] else name
                                for i, name in enumerate(adata.var_names)])

# Filter out lowly expressed cells/proteins
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=2)

adata.raw = adata

# Log-normalize the data
sc.pp.log1p(adata)

# Scale data
sc.pp.scale(adata, max_value=10)

# Ensure sc.pp.neighbors() is Run Before Leiden Clustering
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# Use only Leiden resolution X
leiden_res = 3
leiden_key = f"leiden_res_{leiden_res:.2f}"
sc.tl.leiden(adata, key_added=leiden_key, resolution=leiden_res, flavor="igraph")

# Find overlapping marker genes
matching_marker_genes = {
    cell_type: [gene for gene in genes if gene in adata.var_names]
    for cell_type, genes in marker_genes.items()
}

# Remove cell types with no matching genes
matching_marker_genes = {cell_type: genes for cell_type, genes in matching_marker_genes.items() if genes}

if matching_marker_genes:
    # Compute cell scores using sc.tl.score_genes()
    for cell_type, genes in matching_marker_genes.items():
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"{cell_type}_score")

    # Create a DataFrame of scores
    scores_df = pd.DataFrame({cell_type: adata.obs[f"{cell_type}_score"] for cell_type in matching_marker_genes.keys()})

    # Assign the highest scoring cell type to each cell
    adata.obs["predicted_cell_type"] = scores_df.idxmax(axis=1)

    # Compute max marker score per cell
    adata.obs["max_score"] = scores_df.max(axis=1)

    # Compute mean score per cluster
    cluster_scores = adata.obs.groupby(leiden_key, observed=True)["max_score"].mean()
    cluster_diversity = adata.obs.groupby(leiden_key, observed=True)["predicted_cell_type"].nunique()

    # Set refined threshold for "Unknown" clusters
    score_threshold = cluster_scores.quantile(0.01)  # Bottom 1%
    diversity_threshold = cluster_diversity.quantile(1)  # Most mixed clusters

    unknown_clusters = cluster_scores[cluster_scores < score_threshold].index.union(
        cluster_diversity[cluster_diversity > diversity_threshold].index
    )

    # Assign most common cell type per cluster, keeping "Unknown" for ambiguous clusters
    cluster_to_celltype = (
        adata.obs.groupby(leiden_key)["predicted_cell_type"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
        .to_dict()
    )

    # Initialize all clusters as their most common cell type
    adata.obs[f"{leiden_key}_named"] = adata.obs[leiden_key].map(cluster_to_celltype)

    # Ensure "Unknown" clusters are correctly labeled and not overwritten
    adata.obs.loc[adata.obs[leiden_key].isin(unknown_clusters), f"{leiden_key}_named"] = "Unknown"

    # Convert to categorical format
    adata.obs[f"{leiden_key}_named"] = adata.obs[f"{leiden_key}_named"].astype("category")

    # Debug: Print final cluster names
    print(f"\nFinal Cluster Names for Leiden {leiden_res:.2f}:")
    print(adata.obs[f"{leiden_key}_named"].value_counts())

# Now perform PCA and UMAP AFTER cell types are assigned
sc.tl.pca(adata, svd_solver='arpack')
sc.tl.umap(adata)

# === EARLY UMAP EXPORT (right after initial Leiden + first UMAP) ===
EARLY_OUT = r"C:\Users\au520563\OneDrive - Aarhus universitet\DATA\20251021\DCT\UMAP_coords_with_meta_INITIAL.csv"

# pick the celltype label to export
preferred_cols = ["celltype", f"{leiden_key}_named", "leiden_named", leiden_key, "leiden", "predicted_cell_type"]
label_col = next((c for c in preferred_cols if c in adata.obs.columns), None)

um = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)

if label_col is None:
    df = um.copy()
    df["celltype"] = "Unknown"
else:
    df = um.join(adata.obs[label_col].astype(str).rename("celltype"))

df["cell_id"] = df.index.astype(str)
df["dataset"] = "DCT"
df = df[["cell_id", "UMAP1", "UMAP2", "celltype", "dataset"]]

Path(EARLY_OUT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(EARLY_OUT, index=False)
print(f"✅ Early UMAP exported → {EARLY_OUT}  (rows={len(df)})")

# ---- Early Metadata export (must match the same cells/labels) ----
EARLY_META = r"C:\Users\au520563\OneDrive - Aarhus universitet\DATA\20251021\DCT\DCT_758cells_CellMetadata_INITIAL.csv"
meta_cols = ["cell_id","celltype","dataset"]
tmp = adata.obs.copy()
tmp["cell_id"] = tmp.index.astype(str)
tmp["dataset"] = "DCT"
if "celltype" not in tmp.columns and label_col:
    tmp["celltype"] = tmp[label_col].astype(str)
elif "celltype" not in tmp.columns:
    tmp["celltype"] = "Unknown"
df_meta = tmp[meta_cols]
df_meta.to_csv(EARLY_META, index=False)
print(f"✅ Early metadata exported → {EARLY_META}  (rows={len(df_meta)})")

# ---------------------------------------------------------------------------------------
# UMAP for a user list of selected genes — one figure per gene
# ---------------------------------------------------------------------------------------
genes_to_plot = ['Slc12a3', 'Mki67', 'Pvalb']   # <-- edit this list

_use_raw = True if getattr(adata, "raw", None) is not None else False
present = [g for g in genes_to_plot if g in (adata.raw.var_names if _use_raw else adata.var_names)]
missing = [g for g in genes_to_plot if g not in (adata.raw.var_names if _use_raw else adata.var_names)]
if missing:
    print(f"⚠️ Skipping missing genes: {missing}")

# Pre-fetch coordinates once
_umap = adata.obsm['X_umap']

for gene in present:
    # fetch expression vector
    if _use_raw:
        vec = adata.raw[:, gene].X
    else:
        vec = adata[:, gene].X
    vec = vec.toarray().ravel() if sp.issparse(vec) else np.asarray(vec).ravel()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)
    pts = ax.scatter(
        _umap[:, 0], _umap[:, 1],
        c=vec,
        cmap='Reds',
        s=16,
        linewidth=0,
        alpha=0.9
    )
    # axis labels & ticks @12
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    # gene title @12 and **bold**
    ax.set_title(gene, fontsize=12, fontweight='bold')

    cbar = plt.colorbar(pts, ax=ax, pad=0.01)
    cbar.set_label('Expression abundance', rotation=90, labelpad=12, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    out = os.path.join(data_folder, f"UMAP_expr_{gene}.png")
    plt.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved UMAP for {gene}: {out}")

# -----------------------------------------
# Trajectory Analysis using Diffusion Pseudotime (DPT)
# -----------------------------------------
# Set "DCT1" as the root cell type for trajectory analysis.
# Here we choose the first cell with predicted cell type "DCT1".
prolif_indices = np.where(adata.obs["predicted_cell_type"] == "DCT1")[0]
if len(prolif_indices) > 0:
    adata.uns['iroot'] = int(prolif_indices[0])
    print(f"Setting root cell as 'DCT1' with index: {adata.uns['iroot']}")
else:
    print("No cell with predicted cell type 'DCT1' found. Using default root cell index 0.")
    adata.uns['iroot'] = 0

# Compute diffusion pseudotime
sc.tl.dpt(adata)

# Plot UMAP with diffusion pseudotime coloring
# output path
dpt_fig_path = os.path.join(data_folder, "UMAP_DPT_Pseudotime.png")

# extract coords & pseudotime
coords = adata.obsm['X_umap']
pseudotime = adata.obs['dpt_pseudotime'].values

# set up figure
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

# scatter the cells, colored by pseudotime
scat = ax.scatter(
    coords[:, 0], coords[:, 1],
    c=pseudotime, cmap='viridis',
    s=30, alpha=0.8, linewidth=0, rasterized=True,
)

# axes labels + ticks @12
ax.set_xlabel('UMAP1', fontsize=12)
ax.set_ylabel('UMAP2', fontsize=12)
ax.tick_params(axis='both', labelsize=12)

# no title
ax.set_title('')

# colorbar with label & tick size @12
cbar = fig.colorbar(scat, ax=ax, pad=0.01)
cbar.set_label('Diffusion Pseudotime', rotation=90, labelpad=15, fontsize=12)
cbar.ax.tick_params(labelsize=12)

# tidy up and save
plt.tight_layout()
plt.savefig(dpt_fig_path, dpi=600, bbox_inches='tight')
plt.close(fig)

print(f"✅ Diffusion Pseudotime UMAP saved to: {dpt_fig_path}")

# ----------------------------------------------------------------------------
# PAGA / graph-based trajectory inference + edge scores
# ----------------------------------------------------------------------------
# 1) Run PAGA on cell-type groups
sc.tl.paga(adata, groups='predicted_cell_type')

# 2) Extract connectivity matrix (group × group) and group order
groups = adata.obs['predicted_cell_type'].cat.categories.tolist()
conn_mat = adata.uns['paga']['connectivities']
conn = conn_mat.toarray() if sp.issparse(conn_mat) else np.asarray(conn_mat)

# 3) Build an edge table (upper triangle, no self-edges)
src_idx, tgt_idx = np.triu_indices(len(groups), k=1)
edges = []
for i, j in zip(src_idx, tgt_idx):
    w = float(conn[i, j])
    edges.append({"source": groups[i], "target": groups[j], "connectivity": w})

edge_df = pd.DataFrame(edges)
edge_df = edge_df[edge_df["connectivity"] > 0].sort_values("connectivity", ascending=False)

# 4) Save edge scores to CSV
paga_scores_csv = os.path.join(data_folder, f"Leiden_{leiden_res}_PAGA_connectivities.csv")
edge_df.to_csv(paga_scores_csv, index=False)
print(f"✅ PAGA edge scores saved to: {paga_scores_csv}")

# 5) Report the 3 connections of interest, if present


def _lookup(a, b):
    if a in groups and b in groups:
        ia, ib = groups.index(a), groups.index(b)
        return float(conn[min(ia, ib), max(ia, ib)])
    return np.nan

for a, b in [("DCT1", "DCT2"), ("DCT1", "ProLIF"), ("DCT2", "ProLIF")]:
    val = _lookup(a, b)
    if not np.isnan(val):
        print(f"   • Connectivity {a}–{b}: {val:.3f}")

# 6) Plot the abstract PAGA graph and annotate edges with weights
paga_fig = os.path.join(data_folder, f"Leiden_{leiden_res}_PAGA.png")
plt.figure(figsize=(6, 6))
ax = sc.pl.paga(
    adata,
    show=False,
    frameon=False
)

# Enlarge nodes a bit
node_colls = [c for c in ax.collections if isinstance(c, PathCollection)]
if node_colls:
    node_coll = node_colls[0]
    node_coll.set_sizes(node_coll.get_sizes() * 3)

# Nudge labels outward and bump font
cx, cy = np.mean(ax.get_xlim()), np.mean(ax.get_ylim())
for txt in ax.texts:
    txt.set_fontsize(15)
    x, y = txt.get_position()
    dx, dy = x - cx, y - cy
    d = np.hypot(dx, dy)
    if d > 0:
        txt.set_position((x + 0.10 * dx / d, y + 0.10 * dy / d))

# If PAGA stored node positions, use them to place edge weights
pos = adata.uns.get('paga', {}).get('pos', None)
if pos is not None and len(pos) == len(groups):
    for i, j in zip(src_idx, tgt_idx):
        w = float(conn[i, j])
        if w <= 0:
            continue
        x_mid = 0.5 * (pos[i, 0] + pos[j, 0])
        y_mid = 0.5 * (pos[i, 1] + pos[j, 1])
        ax.text(
            x_mid, y_mid, f"{w:.2f}",
            ha="center", va="center",
            fontsize=18, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            zorder=5
        )

plt.tight_layout()
plt.savefig(paga_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ PAGA graph saved to: {paga_fig}")

# ----------------------------------------------------------------------------
# Pseudotime distribution by cell type (boxplot) + median overlays
# ----------------------------------------------------------------------------
cell_types   = adata.obs['predicted_cell_type']
pseudotimes  = adata.obs['dpt_pseudotime']

# Keep plotting order consistent with the categorical order
order = cell_types.cat.categories.tolist()

dpt_group_fig = os.path.join(data_folder, f"predicted_cell_type_pseudotime_boxplot.png")
plt.figure(figsize=(8, 6), dpi=600)

# draw box + individual points (use the same order for both)
sns.boxplot(x=cell_types, y=pseudotimes, palette='Set2', order=order)
sns.stripplot(x=cell_types, y=pseudotimes, color='black', size=3, alpha=0.4, jitter=True, order=order)

# labels & ticks @12
ax = plt.gca()
ax.set_xlabel('', fontsize=15)  # keep empty label, but set size consistently
ax.set_ylabel('Diffusion Pseudotime', fontsize=15)
ax.tick_params(axis='both', labelsize=15)
plt.xticks(rotation=45, ha='right', fontsize=15)

plt.tight_layout()
plt.savefig(dpt_group_fig, dpi=600, bbox_inches='tight')
plt.close()
print(f"✅ Pseudotime distribution boxplot saved to: {dpt_group_fig}")

# -----------------------------------------------------------------------------
# Gene Expression vs. Diffusion Pseudotime (robust version)
# -----------------------------------------------------------------------------
if "dpt_pseudotime" not in adata.obs:
    print("⚠️  No DPT results found; skipping expression‑vs‑pseudotime plots.")
else:
    # Make sure scanpy is still the scanpy module

    genes_pt = ['Slc12a3','Mki67','Top2a','Pcna']  # or any list, e.g. ['Slc12a3','Mki67']
    display_names = {
        "Mki67": "Ki-67",
        # add more if you have other special cases...
    }

    present = [g for g in genes_pt if g in adata.var_names]
    missing = [g for g in genes_pt if g not in adata.var_names]
    if missing:
        print(f"⚠️  Skipping {missing}; not in adata.var_names")
    dpt = adata.obs["dpt_pseudotime"].values

    for gene in present:
        disp = display_names.get(gene, gene)
        expr = adata[:, gene].X
        expr = expr.toarray().flatten() if hasattr(expr, "toarray") else expr.flatten()

        plt.figure(figsize=(5, 4), dpi=600)
        pts = plt.scatter(
            dpt, expr,
            c=dpt, cmap="viridis", s=20, edgecolors="none"
        )
        cbar = plt.colorbar(pts)
        cbar.set_label("Diffusion Pseudotime", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.xlabel("Diffusion pseudotime", fontsize=12)
        plt.ylabel(f"{disp} expression", fontsize=12)
        plt.tick_params(axis='both', labelsize=12)

        out = os.path.join(data_folder, f"{gene}_expr_vs_pseudotime.png")
        plt.tight_layout()
        plt.savefig(out, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved {disp} expression‑vs‑pseudotime: {out}")

# 📌 **Extract UMAP coordinates manually**
umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
umap_df["Cell Type"] = adata.obs[f"{leiden_key}_named"].values  # Assign predicted cell types

# This section creates a file with each cell's idenpngier, its UMAP coordinates, and its predicted cell type.
umap_celltype_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
umap_celltype_df["Cell ID"] = umap_celltype_df.index
umap_celltype_df["Predicted Cell Type"] = adata.obs[f"{leiden_key}_named"].values
umap_celltype_df = umap_celltype_df.reset_index(drop=False)  # Optionally keep the original index as a column
detailed_umap_csv_path = os.path.join(data_folder, "Detailed_UMAP_CellType_Info.csv")
umap_celltype_df.to_csv(detailed_umap_csv_path, index=False)
print(f"✅ Detailed UMAP cell type information saved to: {detailed_umap_csv_path}")

# Build a palette that is identical to seaborn's default ordering,
# except map 'Unknown' explicitly to grey.
cats = list(pd.Series(umap_df["Cell Type"]).astype(str).unique())
base_colors = sns.color_palette(None, n_colors=len(cats))
palette_map = {}
i = 0
for c in cats:
    if isinstance(c, str) and c.lower() == 'unknown':
        palette_map[c] = (0.7, 0.7, 0.7)  # light grey for Unknown
    else:
        palette_map[c] = base_colors[i % len(base_colors)]
        i += 1

# 📌 **Save UMAP Plot**
umap_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_UMAP_plot.png")

fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"], hue=umap_df["Cell Type"],
    palette=palette_map,
    edgecolor="black", alpha=0.8, s=50, ax=ax
)

# axes labels + ticks @12; keep bold axis labels
ax.set_xlabel("UMAP1", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("UMAP2", fontsize=12, fontweight="bold", labelpad=10)
ax.tick_params(axis="both", labelsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend outside, fontsize 12, title bold "Cell Type"
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles, labels,
    loc="center left", bbox_to_anchor=(1, 0.5),
    fontsize=12, title="Cell Type", frameon=False
)
leg.get_title().set_fontsize(12)
leg.get_title().set_fontweight("bold")  # ← make "Cell Type" bold

plt.subplots_adjust(right=0.75, bottom=0.15)
plt.savefig(umap_fig_path, dpi=600, bbox_inches="tight")
plt.close()

print(f"✅ UMAP figure saved to: {umap_fig_path}")

#------------Marker Dotplot Section-------------
# 📌 **Calculate Dynamic Figure Height**
num_markers = sum(len(genes) for genes in matching_marker_genes.values())
fig_height = min(max(num_markers * 0.15, 4), 10)  # Ensure proper height without too much blank space

# 1. Draw into the current figure, but don’t show it yet
sc.pl.dotplot(
    adata,
    matching_marker_genes,
    groupby="predicted_cell_type",
    figsize=(8, fig_height),
    color_map="Reds",
    dot_max=0.8, dot_min=0.05,
    standard_scale="var",
    dendrogram=False,
    var_group_rotation=0,
    show=False
)

# 2. Grab that figure
fig = plt.gcf()

# 3. Rotate the bottom x-tick labels (your genes)
for ax in fig.axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# 4. Save and close
dotplot_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_Dotplot.png")
fig.savefig(dotplot_fig_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"✅ Dotplot figure saved to: {dotplot_fig_path}")

# ===== Stacked violin: Top 10 DE genes across predicted cell types =====
# Grouping to show on the stacked violin
groupby_key_sv = 'predicted_cell_type'

# Run DE for these groups without overwriting earlier DE results
de_key_sv = 'rank_genes_predicted_celltype'
if (de_key_sv not in adata.uns) or (
    adata.uns.get(de_key_sv, {}).get('params', {}).get('groupby') != groupby_key_sv
):
    sc.tl.rank_genes_groups(adata, groupby=groupby_key_sv, method='wilcoxon', key_added=de_key_sv)

# Gather all DE results; pick overall top 10 unique genes (highest score/logFC)
de_df = sc.get.rank_genes_groups_df(adata, group=None, key=de_key_sv)
sort_col = 'scores' if 'scores' in de_df.columns else (
    'score' if 'score' in de_df.columns else (
    'logfoldchanges' if 'logfoldchanges' in de_df.columns else None))
if sort_col:
    de_df = de_df.sort_values(by=sort_col, ascending=False)

_use_raw = getattr(adata, 'raw', None) is not None
valid_names = adata.raw.var_names if _use_raw else adata.var_names

top_genes = []
for g in de_df['names'].tolist():
    if g in valid_names and g not in top_genes:
        top_genes.append(g)
    if len(top_genes) == 10:
        break

if not top_genes:
    print("⚠️ No valid DE genes found for stacked violin.")
else:
    # Size scales with number of groups/genes
    n_groups = adata.obs[groupby_key_sv].nunique()
    fig_w = max(8, 0.8 * len(top_genes))
    fig_h = max(4.5, 0.45 * n_groups)

    sc.pl.stacked_violin(
        adata,
        var_names=top_genes,
        groupby=groupby_key_sv,
        use_raw=_use_raw,
        standard_scale='var',
        swap_axes=False,           # genes on X, cell types on Y (as in your figure)
        figsize=(fig_w, fig_h),
        show=False,
    )

    # Rotate the X-axis gene labels for readability (no swap_axes)
    fig = plt.gcf()
    for ax in fig.axes:
        labs = ax.get_xticklabels()
        if labs and any(lbl.get_text() for lbl in labs):
            plt.setp(labs, rotation=45, ha="right")

    stacked_out = os.path.join(
        data_folder, f"StackedViolin_top10_{groupby_key_sv}_Leiden_{leiden_res}.png"
    )
    plt.tight_layout()
    plt.savefig(stacked_out, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Stacked violin (top 10 across {groupby_key_sv}) saved to: {stacked_out}")
# ======================================================================
# === Sensitivity & Dynamic Range BLOCK (auto-inserted) ===
# --- User-editable colors for dataset comparison (boxplots) ---
# Map dataset names to hex colors. Keys must match the labels shown on the x-axis
# (e.g., the basename of your CSV if you use comparison_csv_path).
# Examples:
# dataset_color_map = {
#     "CurrentDataset": "#1b9e77",
#     "Kidney_768cells": "#d95f02",
#     "ProLIF_758cells": "#7570b3",
# }
dataset_color_map = {}

# Computes per-cell sensitivity (#IDs) and dynamic range (log10 decades P05→P95 of non-zero intensities),
# saves tidy CSVs, and renders figures. Colors for per-type plots follow UMAP palette in
# adata.uns["predicted_cell_type_colors"] aligned to adata.obs["predicted_cell_type"].
# Protein-ID bar plot has per-bar error bars colored to match bar fill.

try:
    pass
except Exception as _e:
    print(f"⚠️ Could not import dependencies for sensitivity/dynamic-range block: {_e}")

# Ensure output directory
try:
    data_folder  # noqa: F821
except NameError:
    try:
        output_dir  # noqa: F821
        data_folder = output_dir
    except NameError:
        data_folder = _os.path.join(_os.getcwd(), "outputs")
    _os.makedirs(data_folder, exist_ok=True)

# Optional: set a comparison file path (leave empty to skip)
comparison_csv_path = r""


def _per_cell_metrics(X, cell_ids, cell_types=None, p_low=5.0, p_high=95.0):
    out = []
    n_cells = X.shape[0]
    for i in range(n_cells):
        v = X[i, :]
        v = v.toarray().ravel() if _sp.issparse(v) else _np.asarray(v).ravel()
        nz = v[v > 0]
        n_detected = int(nz.size)
        total = int(v.size)
        miss_pct = 100.0 * (1.0 - n_detected / total)
        if n_detected >= 10:
            q_lo = _np.percentile(nz, p_low)
            q_hi = _np.percentile(nz, p_high)
            lod  = _np.percentile(nz, 1.0)
            loq  = _np.percentile(nz, p_low)
            dr_decades = (_np.log10(q_hi) - _np.log10(q_lo)) if (q_lo > 0 and q_hi > 0) else _np.nan
        else:
            q_lo = q_hi = lod = loq = _np.nan
            dr_decades = _np.nan
        out.append({
            "Cell ID": cell_ids[i],
            "Predicted Cell Type": (cell_types[i] if cell_types is not None else "NA"),
            "Proteins detected": n_detected,
            "Proteins total": total,
            "Missing %": miss_pct,
            "P05_nonzero": q_lo,
            "P95_nonzero": q_hi,
            "LOD~P01_nonzero": lod,
            "LOQ~P05_nonzero": loq,
            "Dynamic range (decades)": dr_decades,
            "Total intensity (sum)": float(nz.sum()) if n_detected > 0 else 0.0,
            "Median intensity (nonzero)": float(_np.median(nz)) if n_detected > 0 else _np.nan,
        })
    return _pd.DataFrame(out)


def _load_wide_csv_as_matrix(csv_path):
    df = _pd.read_csv(csv_path)
    expr = df.iloc[:, 3:]  # first 3 columns are metadata
    expr = expr.fillna(0).astype(_np.float32).T  # cells × proteins
    return expr


def _umap_palette_and_order(df_col_name="Predicted Cell Type"):
    # Get desired order from adata.obs categorical
    _order_ids = None
    if "predicted_cell_type" in adata.obs:
        try:
            _order_ids = list(adata.obs["predicted_cell_type"].cat.categories)
        except Exception:
            _order_ids = None
    # Build palette map from UMAP colors
    _palette_map = None
    try:
        _cats = list(adata.obs["predicted_cell_type"].cat.categories) if "predicted_cell_type" in adata.obs else []
        _umap_cols = list(adata.uns.get("predicted_cell_type_colors", []))
        if _cats and _umap_cols and len(_umap_cols) >= len(_cats):
            _palette_map = {c: _umap_cols[i] for i, c in enumerate(_cats)}
    except Exception:
        _palette_map = None
    return _order_ids, _palette_map


def _compute_and_plot_sensitivity_dynamic_range():
    # Require an AnnData named 'adata' with intensities in .raw.X (preferred) or .X
    try:
        adata  # noqa: F821
    except NameError:
        print("⚠️ 'adata' not found; skipping sensitivity/dynamic-range block.")
        return

    _rawX = getattr(getattr(adata, "raw", None), "X", None)
    if _rawX is None:
        _rawX = adata.X

    _cell_ids = _np.array(adata.obs_names)
    _cell_types = _np.array(adata.obs.get("predicted_cell_type", _pd.Series("NA", index=adata.obs_names)))
    metrics_df = _per_cell_metrics(_rawX, _cell_ids, _cell_types)

    percell_csv = _os.path.join(data_folder, "Sensitivity_DynamicRange_perCell.csv")
    metrics_df.to_csv(percell_csv, index=False)

    bytype_df = (metrics_df
                 .groupby("Predicted Cell Type", observed=True)
                 .agg({"Proteins detected":"median",
                       "Dynamic range (decades)":"median",
                       "Missing %":"median"})
                 .reset_index())
    bytype_csv = _os.path.join(data_folder, "Sensitivity_DynamicRange_byCellType_median.csv")
    bytype_df.to_csv(bytype_csv, index=False)

    print(f"✅ Sensitivity/dynamic-range (per-cell) → {percell_csv}")
    print(f"✅ Sensitivity/dynamic-range (per-type median) → {bytype_csv}")

    # Use a cell-type label that includes 'Unknown' when available (e.g., '<leiden_key>_named').
    _plot_ct = None
    for _c in [f"{leiden_key}_named", "leiden_named", "celltype", "Cell Type", "CellType", "predicted_cell_type"]:
        if _c in adata.obs.columns:
            try:
                _plot_ct = _np.array(adata.obs[_c].astype(str).values)
            except Exception:
                _plot_ct = None
            break
    _has_unknown = False
    if _plot_ct is not None:
        try:
            _has_unknown = bool(_pd.Series(_plot_ct).astype(str).str.lower().eq("unknown").any())
        except Exception:
            _has_unknown = False

    # 0) Proteins detected by cell type — violin (UMAP palette + strict order)
    _pdf = metrics_df.copy()
    if not _pdf.empty:
        if _plot_ct is not None and len(_plot_ct) == len(_pdf):
            _pdf["Predicted Cell Type"] = _plot_ct
        _pdf["Predicted Cell Type"] = _pdf["Predicted Cell Type"].astype(str)
        _pdf.loc[_pdf["Predicted Cell Type"].str.lower() == "unknown", "Predicted Cell Type"] = "Unknown"

        _order_ids, _palette_map = _umap_palette_and_order()
        # Strict order limited to present categories
        if _order_ids is not None:
            _order_present = [ct for ct in _order_ids if ct in _pdf["Predicted Cell Type"].unique()]
        else:
            _order_present = sorted(_pdf["Predicted Cell Type"].dropna().unique().tolist())
        if _has_unknown and "Unknown" not in _order_present:
            _order_present = _order_present + ["Unknown"]

        _pdf["Predicted Cell Type"] = _pd.Categorical(_pdf["Predicted Cell Type"], categories=_order_present, ordered=True)
        _palette = [_palette_map.get(ct, "#BDBDBD") for ct in _order_present] if _palette_map else None

        _plt.figure(figsize=(8, 5.5), dpi=600)
        _sns.violinplot(data=_pdf, x="Predicted Cell Type", y="Proteins detected", inner="box",
                        order=_order_present, cut=0, palette=_palette)
        _plt.xticks(rotation=45, ha="right", fontsize=11)
        _plt.yticks(fontsize=11)
        _plt.xlabel("")
        _plt.ylabel("IDs per cell", fontsize=12)
#       _plt.title("Proteins detected by cell type", fontsize=12, fontweight="bold")
        _plt.tight_layout()
        ids_violin_path = _os.path.join(data_folder, "ProteinsDetected_byCellType_violin.png")
        _plt.savefig(ids_violin_path, dpi=600, bbox_inches="tight"); _plt.close()
        print(f"✅ Violin (IDs by cell type) → {ids_violin_path}")

    # 0b) Proteins detected by cell type — bar (median + IQR) with UMAP palette + color-matched error bars
    _stat = (metrics_df.groupby("Predicted Cell Type", observed=True)["Proteins detected"]
             .agg(median="median",
                  q1=lambda s: s.quantile(0.25),
                  q3=lambda s: s.quantile(0.75))
             .reset_index())

    if not _stat.empty:
        _order_ids, _palette_map = _umap_palette_and_order()
        if _order_ids is not None:
            _order_present = [ct for ct in _order_ids if ct in set(_stat["Predicted Cell Type"])]
        else:
            _order_present = sorted(_stat["Predicted Cell Type"].dropna().unique().tolist())
        _stat = _stat.set_index("Predicted Cell Type").reindex(_order_present).reset_index()
        _stat = _stat[_stat["Predicted Cell Type"].notna()]

        _err_low = (_stat["median"] - _stat["q1"]).to_numpy()
        _err_high = (_stat["q3"] - _stat["median"]).to_numpy()
        _x = _np.arange(len(_stat))
        _bar_colors = ([_palette_map.get(ct, "#BDBDBD") for ct in _stat["Predicted Cell Type"].tolist()]
                       if _palette_map else ["#BDBDBD"] * len(_stat))

        _plt.figure(figsize=(8, 5.5), dpi=600)
        _plt.bar(_x, _stat["median"].to_numpy(), color=_bar_colors)
        for i, xi in enumerate(_x):
            _plt.errorbar(xi,
                          _stat["median"].to_numpy()[i],
                          yerr=_np.array([[ _err_low[i] ], [ _err_high[i] ]]),
                          fmt="none", capsize=3, linewidth=1, ecolor=_bar_colors[i])
        _plt.xticks(_x, _stat["Predicted Cell Type"].tolist(), rotation=45, ha="right", fontsize=11)
        _plt.yticks(fontsize=11)
        _plt.ylabel("Proteins detected per cell (median ± IQR)", fontsize=12)
        _plt.title("Proteins detected by cell type (median)", fontsize=12, fontweight="bold")
        _plt.tight_layout()
        ids_bar_path = _os.path.join(data_folder, "ProteinsDetected_byCellType_bar.png")
        _plt.savefig(ids_bar_path, dpi=600, bbox_inches="tight"); _plt.close()
        print(f"✅ Bar (IDs by cell type, median±IQR) → {ids_bar_path}")

    # 1) Histogram (100-protein bins)
    _counts = metrics_df["Proteins detected"].values
    if _counts.size > 0:
        _bins_max = int(_np.ceil(_counts.max() / 100.0) * 100) if _counts.max() > 0 else 100
        _bins = _np.arange(0, _bins_max + 100, 100)
        _plt.figure(figsize=(6, 4.5), dpi=600)
        _plt.hist(_counts, bins=_bins, edgecolor="black")
        _plt.xlabel("Proteins detected per cell", fontsize=12)
        _plt.ylabel("Number of cells", fontsize=12)
        _plt.title("Sensitivity (IDs per cell)", fontsize=12, fontweight="bold")
        _plt.tight_layout()
        hist_path = _os.path.join(data_folder, "Histogram_ProteinsDetected_perCell.png")
        _plt.savefig(hist_path, dpi=600, bbox_inches="tight"); _plt.close()
        print(f"✅ Histogram saved → {hist_path}")

    # 2) Dynamic range by cell type (violin) — UMAP palette + strict order
    _vdf = metrics_df.dropna(subset=["Dynamic range (decades)"]).copy()
    if not _vdf.empty:
        if _plot_ct is not None and len(_plot_ct) == len(metrics_df):
            _vdf["Predicted Cell Type"] = _plot_ct[_vdf.index.to_numpy()]
        _vdf["Predicted Cell Type"] = _vdf["Predicted Cell Type"].astype(str)
        _vdf.loc[_vdf["Predicted Cell Type"].str.lower() == "unknown", "Predicted Cell Type"] = "Unknown"

        _order_ids, _palette_map = _umap_palette_and_order()
        if _order_ids is not None:
            _order_present = [ct for ct in _order_ids if ct in _vdf["Predicted Cell Type"].unique()]
        else:
            _order_present = sorted(_vdf["Predicted Cell Type"].dropna().unique().tolist())
        if _has_unknown and "Unknown" not in _order_present:
            _order_present = _order_present + ["Unknown"]
        _vdf["Predicted Cell Type"] = _pd.Categorical(_vdf["Predicted Cell Type"], categories=_order_present, ordered=True)
        _palette = [_palette_map.get(ct, "#BDBDBD") for ct in _order_present] if _palette_map else None

        _plt.figure(figsize=(8, 5.5), dpi=600)
        _sns.violinplot(data=_vdf, x="Predicted Cell Type", y="Dynamic range (decades)",
                        inner="box", order=_order_present, cut=0, palette=_palette)
        _plt.xticks(rotation=45, ha="right", fontsize=11)
        _plt.yticks(fontsize=11)
        _plt.xlabel("")
        _plt.ylabel("Dynamic range", fontsize=12)
#       _plt.title("Dynamic range by cell type", fontsize=12, fontweight="bold")
        _plt.tight_layout()
        violin_path = _os.path.join(data_folder, "DynamicRange_byCellType.png")
        _plt.savefig(violin_path, dpi=600, bbox_inches="tight"); _plt.close()
        print(f"✅ Violin saved → {violin_path}")

    # 3) Scatter: sensitivity vs dynamic range
    if not _vdf.empty:
        _xy = _vdf.dropna(subset=["Proteins detected", "Dynamic range (decades)"])
        if not _xy.empty:
            r, p = _pearsonr(_xy["Proteins detected"].values, _xy["Dynamic range (decades)"].values)
            _plt.figure(figsize=(5.2, 4.6), dpi=600)
            _sns.scatterplot(data=_xy, x="Proteins detected", y="Dynamic range (decades)", s=28, alpha=0.7, edgecolor=None)
            _plt.xlabel("Proteins detected per cell", fontsize=12)
            _plt.ylabel("Dynamic range (log10 decades)", fontsize=12)
            _plt.title(f"Sensitivity vs dynamic range (r = {r:.2f}, p = {p:.1e})", fontsize=12, fontweight="bold")
            _plt.tight_layout()
            scatter_path = _os.path.join(data_folder, "Sensitivity_vs_DynamicRange_scatter.png")
            _plt.savefig(scatter_path, dpi=600, bbox_inches="tight"); _plt.close()
            print(f"✅ Scatter saved → {scatter_path}")

    # Optional dataset comparison (boxplots)
    if isinstance(comparison_csv_path, str) and len(comparison_csv_path) > 0 and _os.path.exists(comparison_csv_path):
        expr2 = _load_wide_csv_as_matrix(comparison_csv_path)
        ids2 = _np.array([f"Cell_{i+1}" for i in range(expr2.shape[0])])
        metrics2 = _per_cell_metrics(expr2, ids2, cell_types=_np.array(["NA"]*expr2.shape[0]))
        metrics2["Dataset"] = _os.path.splitext(_os.path.basename(comparison_csv_path))[0]

        metrics1 = metrics_df.copy()
        try:
            dataset_name = _os.path.splitext(_os.path.basename(csv_file_path))[0]  # noqa: F821
        except Exception:
            dataset_name = "CurrentDataset"
        metrics1["Dataset"] = dataset_name

        both = _pd.concat([metrics1[["Dataset","Proteins detected","Dynamic range (decades)"]],
                           metrics2[["Dataset","Proteins detected","Dynamic range (decades)"]]],
                          ignore_index=True)


        # Build dataset order (current first) and palette from UMAP colors (fallback grey)
        _all_ds = list(both["Dataset"].unique())
        if dataset_name in _all_ds:
            _others = [d for d in _all_ds if d != dataset_name]
            ds_order = [dataset_name] + _others
        else:
            ds_order = _all_ds

        _umap_cols = list(adata.uns.get("predicted_cell_type_colors", []))
        ds_colors = {}
        for i, ds in enumerate(ds_order):
            # Prefer user-defined colors; otherwise fall back to UMAP colors, then grey
            if isinstance(globals().get("dataset_color_map", {}), dict) and ds in dataset_color_map and dataset_color_map[ds]:
                ds_colors[ds] = dataset_color_map[ds]
            else:
                ds_colors[ds] = _umap_cols[i] if i < len(_umap_cols) else "#BDBDBD"

        # Boxplot: Proteins detected (color by dataset)
        _plt.figure(figsize=(5.2, 4.6), dpi=600)
        _sns.boxplot(data=both, x="Dataset", y="Proteins detected", order=ds_order, palette=ds_colors)
        _sns.stripplot(data=both, x="Dataset", y="Proteins detected", hue="Dataset",
                       order=ds_order, palette=ds_colors, dodge=False, size=3, alpha=0.35, jitter=True, legend=False)
        _plt.ylabel("Proteins detected per cell", fontsize=12); _plt.xlabel("")
        _plt.title("Sensitivity comparison", fontsize=12, fontweight="bold"); _plt.tight_layout()
        comp_ids_path = _os.path.join(data_folder, "Compare_Sensitivity_ProteinsDetected.png")
        _plt.savefig(comp_ids_path, dpi=600, bbox_inches="tight"); _plt.close()

        # Boxplot: Dynamic range (color by dataset)
        _both_dr = both.dropna(subset=["Dynamic range (decades)"]).copy()
        _plt.figure(figsize=(5.2, 4.6), dpi=600)
        _sns.boxplot(data=_both_dr, x="Dataset", y="Dynamic range (decades)", order=ds_order, palette=ds_colors)
        _sns.stripplot(data=_both_dr, x="Dataset", y="Dynamic range (decades)", hue="Dataset",
                       order=ds_order, palette=ds_colors, dodge=False, size=3, alpha=0.35, jitter=True, legend=False)
        _plt.ylabel("Dynamic range (log10 decades)", fontsize=12); _plt.xlabel("")
        _plt.title("Dynamic range comparison", fontsize=12, fontweight="bold"); _plt.tight_layout()
        comp_dr_path = _os.path.join(data_folder, "Compare_DynamicRange.png")
        _plt.savefig(comp_dr_path, dpi=600, bbox_inches="tight"); _plt.close()


        comp_csv = _os.path.join(data_folder, f"Sensitivity_DynamicRange_perCell_{metrics2['Dataset'].iloc[0]}.csv")
        metrics2.to_csv(comp_csv, index=False)
        print(f"✅ Comparison figures → {comp_ids_path}; {comp_dr_path}")
        print(f"✅ Comparison per-cell metrics → {comp_csv}")

# Auto-run at the end of the script (safe to re-run)
try:
    _compute_and_plot_sensitivity_dynamic_range()
except Exception as _e:
    print(f"⚠️ Sensitivity/dynamic-range block failed: { _e }")
# === End of Sensitivity & Dynamic Range BLOCK ===


# =========================
# Core / Unique core tables + UpSet combos + Core Jaccard heatmap
# (ported from kidney script; uses presence/absence thresholding)
# =========================

CORE_FRAC       = float(_os_core.getenv('CORE_FRAC', '0.50'))
UNIQUE_LOW_FRAC = float(_os_core.getenv('UNIQUE_LOW_FRAC', '0.10'))
UPSET_TOPN      = int(_os_core.getenv('UPSET_TOPN', '20'))


def _get_detection_matrix():
    X = adata.raw.X if getattr(adata, 'raw', None) is not None else adata.X
    if sp.issparse(X):
        return (X > 0).astype('int8')
    arr = np.asarray(X)
    return (arr > 0).astype('int8')


def _compute_coverage_matrix(group_key: str):
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
        if n_cells == 0:
            continue
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
        out_core = _os_core.path.join(data_folder, f"Core_Proteins_{ct}_{tag}_{suffix}.csv")
        out_uniq = _os_core.path.join(data_folder, f"UniqueCore_Proteins_{ct}_{tag}_{suffix}.csv")
        _mk(core_ids).to_csv(out_core, index=False)
        _mk(uniq_ids).to_csv(out_uniq, index=False)
        print("✅ Saved:", out_core)
        print("✅ Saved:", out_uniq)
        summary.append({'cell_type': ct, 'n_core': int(core_masks[i, :].sum()), 'n_unique_core': int(unique_core_masks[i, :].sum())})

    df_sum = pd.DataFrame(summary)
    out_sum = _os_core.path.join(data_folder, f"CoreUnique_Summary_{tag}_{suffix}.csv")
    df_sum.to_csv(out_sum, index=False)
    print("✅ Saved:", out_sum)


def plot_upset_core(group_key: str, tag: str, core_frac: float, topn: int = 20):
    types, coverage = _compute_coverage_matrix(group_key)
    core_masks = coverage >= core_frac
    n_types, n_prot = core_masks.shape
    combos = []
    for p in range(n_prot):
        members = [types[i] for i in range(n_types) if core_masks[i, p]]
        combos.append(';'.join(members) if members else '(none)')
    s = pd.Series(combos, name='combo').value_counts().reset_index()
    s.columns = ['combo','count']
    s = s[s['combo'] != '(none)']
    s_top = s.nlargest(topn, 'count')
    out_csv = _os_core.path.join(data_folder, f"UpSet_CoreCombination_{tag}_core{int(core_frac*100)}.csv")
    s.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv)
    fig, ax = plt.subplots(figsize=(max(6, 0.45*len(s_top)), 4.8))
    ax.bar(range(len(s_top)), s_top['count'].values, edgecolor='black', linewidth=0.6)
    ax.set_xticks(range(len(s_top)))
    ax.set_xticklabels(s_top['combo'].values, rotation=45, ha='right')
    ax.set_ylabel('Proteins (exact core‑set membership)')
    ax.set_xlabel(f'Core combinations (≥{int(core_frac*100)}% within each type)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out_png = _os_core.path.join(data_folder, f"UpSet_CoreCombination_{tag}_core{int(core_frac*100)}_Top{topn}.png")
    plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close(); print("✅ Saved:", out_png)


def plot_core_jaccard(group_key: str, tag: str, core_frac: float):
    types, coverage = _compute_coverage_matrix(group_key)
    core_masks = coverage >= core_frac
    n_types = len(types)
    jacc = np.zeros((n_types, n_types), float)
    for i in range(n_types):
        Ai = core_masks[i, :]
        for j in range(n_types):
            Aj = core_masks[j, :]
            inter = int(np.logical_and(Ai, Aj).sum())
            union = int(np.logical_or(Ai, Aj).sum())
            jacc[i, j] = (inter / union) if union > 0 else 0.0
    fig, ax = plt.subplots(figsize=(max(5, 0.6*n_types), max(4, 0.6*n_types)))
    im = ax.imshow(jacc, aspect='auto')
    ax.set_xticks(range(n_types)); ax.set_yticks(range(n_types))
    ax.set_xticklabels(types, rotation=45, ha='right'); ax.set_yticklabels(types)
    ax.set_title(f'Core‑set Jaccard (≥{int(core_frac*100)}%)')
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('Jaccard index')
    plt.tight_layout()
    out_png = _os_core.path.join(data_folder, f"Heatmap_Core_Jaccard_{tag}_core{int(core_frac*100)}.png")
    plt.savefig(out_png, dpi=600, bbox_inches='tight'); plt.close(); print("✅ Saved:", out_png)

# Run core/unique analyses on predicted_cell_type
try:
    write_core_unique_tables('predicted_cell_type', 'DCT', CORE_FRAC, UNIQUE_LOW_FRAC)
    plot_upset_core('predicted_cell_type', 'DCT', CORE_FRAC, UPSET_TOPN)
    plot_core_jaccard('predicted_cell_type', 'DCT', CORE_FRAC)
except Exception as _e:
    print("⚠️ Core/unique analysis failed:", _e)

# =========================
# Export Cell‑metadata CSV for the browser
# =========================


def export_cell_metadata_dct():
    """Export per‑cell metadata and return (celltype_col, output_csv_path)."""
    # Prefer predicted labels; fall back to named Leiden
    preferred_cols = ['predicted_cell_type', f"{leiden_key}_named"]
    celltype_col = None
    for c in preferred_cols:
        if c in adata.obs.columns:
            celltype_col = c
            break
    df = pd.DataFrame({
        'cell_id': adata.obs_names.astype(str).values,
        'celltype': (adata.obs[celltype_col].astype(str).values if celltype_col else 'Unknown')
    })
    # Optional extras
    if leiden_key in adata.obs.columns:
        df['leiden'] = adata.obs[leiden_key].astype(str).values
    named_key = f"{leiden_key}_named"
    if named_key in adata.obs.columns:
        df['leiden_named'] = adata.obs[named_key].astype(str).values
    # Carry QC columns if present
    for extra in ['sens_n_proteins','dynr_value','dynr_low','dynr_high','dynr_scale']:
        if extra in adata.obs.columns:
            df[extra] = adata.obs[extra].values
    folder = os.path.dirname(csv_file_path)
    base   = os.path.splitext(os.path.basename(csv_file_path))[0]
    out    = os.path.join(folder, f"{base}_CellMetadata.csv")
    df.to_csv(out, index=False)
    print(f"✅ Cell‑metadata CSV exported → {out}")
    return (celltype_col, out)

# Export at end of script
try:
    # Determine a celltype column to include
    ct_col = 'predicted_cell_type' if 'predicted_cell_type' in adata.obs.columns else f"{leiden_key}_named"
    # Export metadata (ignore its return value if any)
    try:
        export_cell_metadata_dct()
    except Exception:
        pass
    # UMAP CSV export is called at the very end (after the function is defined)
except Exception as _e:
    print(f"⚠️ Failed to export metadata early: {_e}")


# =========================
# Export processed AnnData (.h5ad)
# =========================


def export_h5ad_dct(save_lite=False):
    """
    Writes the current AnnData object to disk.
    - Full: keeps .raw, neighbors/graphs, embeddings, obs/var/uns.
    - Lite (optional): drops .raw and forces X to float32 to reduce file size.
    """

    folder = os.path.dirname(csv_file_path)
    base   = os.path.splitext(os.path.basename(csv_file_path))[0]

    # Full object
    out_full = os.path.join(folder, f"{base}_processed.h5ad")
    try:
        adata.write(out_full, compression="gzip")
    except TypeError:
        # older anndata versions
        adata.write(out_full)
    print(f"✅ Full .h5ad saved → {out_full}")

    if save_lite:
        adata_lite = adata.copy()
        adata_lite.raw = None  # drop raw to shrink
        # ensure float32 for X
        if _sp.issparse(adata_lite.X):
            adata_lite.X = adata_lite.X.astype(np.float32)
        else:
            adata_lite.X = np.asarray(adata_lite.X, dtype=np.float32)
        out_lite = os.path.join(folder, f"{base}_processed_LITE.h5ad")
        try:
            adata_lite.write(out_lite, compression="gzip")
        except TypeError:
            adata_lite.write(out_lite)
        print(f"✅ Lite .h5ad saved → {out_lite}")

# Save the .h5ad now (set save_lite=True if you also want the smaller file)
try:
    export_h5ad_dct(save_lite=False)
except Exception as e:
    print(f"⚠️ Failed to export .h5ad: {e}")


"""
DCT in-silico perturbation analysis (FINAL)
-------------------------------------------
Features:
    pass
- Capacity score ("Aldo_machinery"): MR/HSD11B2/SGK1/SLC12A3/WNK–SPAK/KLHL3/CUL3/NaK/ENaC
- Activity score ("ALDO_signed"): ALDO_UP (no DOWN). For independence, we also compute
  "ALDO_activity_noMach" = activity excluding machinery proteins (with a lean fallback panel).
- Robust figure saving with return_fig=True
- Pseudotime trends + trend_bins_median.csv
- Correlations:
    * Aldo_machinery vs ALDO_signed
    * Aldo_machinery vs ALDO_activity_noMach
    * ALDO_activity_noMach vs pseudotime
- Pseudobulk (donor/sample × cell type) median-level tests:
    * DCT2 vs DCT1 for ALDO_signed, Aldo_machinery, ALDO_activity_noMach
- Output folder: <outdir_base or env DCT_PERT_OUTDIR or data_folder or CWD>/InSilico_Perturbation/<timestamp>/
"""


CELLLABEL_CANDIDATES = ["predicted_cell_type","cell_type","CellType","celltype","leiden","clusters"]
PSEUDOTIME_CANDIDATES = ["pseudotime","dpt_pseudotime","paga_pseudotime"]
PSEUDOBULK_GROUP_CANDIDATES = ["donor","sample","sample_id","batch","plate"]

# ---------------------------
# Signatures (activity/readout)
# ---------------------------
# ---------- Signatures (DOWN optional; no overlap with same-axis machinery) ----------
SIGNATURES = {
    "ALDO": {"UP": ["SGK1","NEDD4L","SLC12A3","ATP1A1","ATP1B1"], "DOWN": []},

    "ANGII": {"UP": ["JUN","FOS","EGR1","DUSP1","DUSP4","MAPK1","MAPK3","SLC12A3"], "DOWN": []},

    # High-K+ simple effector readout
    "KPLUS": {"UP": ["KCNN4","ATP1A1","ATP1B1","NEDD4L"], "DOWN": []},

    "THIAZIDE": {"UP": ["NEDD4L","UBC","DUSP1","DUSP4"], "DOWN": ["SLC12A3","STK39","WNK1","WNK4"]},

    # >>> FIXED: remove INSR/EGFR/PIK3CA/AKT1 from the signature (put them in machinery)
    "INSULIN_EGF": {"UP": ["DUSP1","DUSP4","EGR1","FOS","JUN","SLC12A3"], "DOWN": []},

    "VASOPRESSIN": {"UP": ["AVPR2","ADCY9","PRKACA","AQP2","AQP3"], "DOWN": []},
}

# --- PATCH: ALDO signature (human-curated UP/DOWN) ---
SIGNATURES["ALDO"]["UP"] = [
    # core UP
    "KLK1","PPARGC1A","KLF15","SGK1","USP2","SCNN1A","NDRG2","PER3","TSC22D3",
    "ZBTB16","MLPH","ATP1A1","FXYD2","FXYD4","COX17","CITED4","ARNT2","EMX2",
    "ACSS3","WNK1","FKBP5","NR3C2","SLC6A15","SLC9A4","SLC25A35","PITRM1",
    "TMPRSS2","TMPRSS4",
    # extras (included so you do not lose anything from the curation)
    "CWH43","DEF8","IQGAP2","IQSEC2","MYH14","NXPE3","OPTN","OSBP2","RAB27A",
    "SEC23B","SYBU","TANGO2","TBC1D24","TRAK1","VOPP1","YIPF1",
    "KLK1B1","KLK1B11","KLK1B24","KLK1B27","KLK1B3","KLK1B4","KLK1B5","KLK1B8","KLK1B9"
]

SIGNATURES["ALDO"]["DOWN"] = [
    "NEDD4L","MYC","EGR4","NFIL3","ATP12A","SLC1A3","SLC7A1","VMN1R32","GFRA1",
    "GDF15","WNT4","DUSP5","DUSP8","RNF39","PIM1","PDK4"
]

# ---------- Machinery (hardware/transducers; no overlap with same-axis signature) ----------
MACHINERY = {
    "Aldo_machinery":   ["NR3C2","HSD11B2","WNK4","STK39"],
    "AngII_machinery":  ["AGTR1","GNAQ","PLCB1","PRKCA"],   # GPCR→Gq→PLC→PKC
    "InsEGF_machinery": ["INSR","EGFR","PIK3CA","AKT1","GRB2","SOS1","MAPK1","MAPK3"],  # RTK→PI3K/AKT & RAS/ERK
}

# Extended aldosterone machinery fallback (detectable co-factors; no effectors)
ALDO_MACHINERY_EXT = [
    "NR3C2","HSD11B2","WNK4","STK39",             # core (may be missing at sc load)
    "HSP90AA1","HSP90AB1","FKBP5","TPR",          # MR chaperone complex
    "NCOA1","NCOA2","NCOA3",                      # co-activators
    "PPP2CA","PPP2R1A","YWHAE","YWHAB","YWHAZ"    # modulators (PP2A, 14-3-3)
]

# ---------------------------
# Aliases (case-insensitive normalization happens later)
# ---------------------------
ALIASES = {
    "MR": "NR3C2", "MCR": "NR3C2",
    "11BHSD2": "HSD11B2", "11B-HSD2": "HSD11B2",
    "SGK-1": "SGK1",
    "SPAK": "STK39",
    "NCC": "SLC12A3",
    "ENAC": "SCNN1A", "ENACA": "SCNN1A", "ENACB": "SCNN1B", "ENACG": "SCNN1G",
    "AT1R": "AGTR1", "AT2R": "AGTR2",
    "ERK1": "MAPK3", "ERK2": "MAPK1",
    "MEK1": "MAP2K1", "MEK2": "MAP2K2",
    "PI3K": "PIK3CA",
    "PKA": "PRKACA", "PKC": "PRKCA",
    "ADCY": "ADCY9",   # generic to a concrete isoform we track
    "GR": "NR3C1",
}

# --- PATCH: ALDO aliases (human-curated) ---
ALIASES.update({
    "GILZ": "TSC22D3",
    "NEDD4-2": "NEDD4L",
    "NEDD4L2": "NEDD4L",
    "PGC1A": "PPARGC1A",
    "ENACALPHA": "SCNN1A",
    "ENACA": "SCNN1A",
    "NHE4": "SLC9A4",
    "PLZF": "ZBTB16",
    "COX-17": "COX17",
    "WNK-1": "WNK1"
})

# ---------------------------
# Sanity check: warn if any signature overlaps its own machinery
# ---------------------------


def _warn_overlaps(SIGNATURES, MACHINERY):
    to_upper = lambda xs: {x.upper() for x in xs}
    for mach_name, mach_genes in MACHINERY.items():
        axis = mach_name.split("_")[0].upper()  # e.g., "InsEGF"
        if axis in SIGNATURES:
            M = to_upper(mach_genes)
            U = to_upper(SIGNATURES[axis]["UP"])
            D = to_upper(SIGNATURES[axis]["DOWN"])
            ov_up = sorted(M & U); ov_dn = sorted(M & D)
            if ov_up or ov_dn:
                print(f"⚠️ Overlap in {axis}: machinery∩UP={ov_up}, machinery∩DOWN={ov_dn}")

# === Pseudobulk capacity helpers (added) ===


def ensure_bin3_from_pseudotime(adata, pseudo_key="dpt_pseudotime", bin_key="bin3"):
    """Create 'early'/'mid'/'late' bins from pseudotime if missing."""
    import pandas as pd
    if pseudo_key in adata.obs.columns and bin_key not in adata.obs.columns:
        ps = adata.obs[pseudo_key].astype(float)
        adata.obs[bin_key] = pd.qcut(ps, q=3, labels=["early","mid","late"], duplicates="drop")


def _zscore_frame(X, np):
    import numpy as np
    X = np.asarray(X)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = np.nan
    return (X - mu) / sd


def pseudobulk_capacity(adata, outdir, machinery_genes,
                        group_keys, ylabel="Aldo_machinery (pseudobulk z)"):
    """
    Compute machinery capacity on aggregated cells (groups in group_keys).
    Saves CSV + bar plot (mean ± half-IQR across machinery genes) per group.
    Case-insensitive and alias-aware gene matching.
    """
    import numpy as np, pandas as pd, scipy.sparse as sp, matplotlib.pyplot as plt, os as _osloc

    # Case-insensitive & alias-aware map
    var_map = pd.Series(adata.var_names, index=adata.var_names.str.upper())
    def _canon(name):
        u = str(name).upper()
        if "ALIASES" in globals() and u in ALIASES:
            u = ALIASES[u].upper()
        return u
    def _resolve(names):
        out = []
        for n in names:
            key = _canon(n)
            if key in var_map:
                out.append(var_map[key])
        return out

    g = _resolve(machinery_genes)
    if not g:
        print("⚠️ No machinery genes detected after case-insensitive/alias matching; skipping pseudobulk capacity.")
        return {}

    # z-score per protein across cells
    X = adata[:, g].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    Z = _zscore_frame(X, np)
    zdf = pd.DataFrame(Z, index=adata.obs_names, columns=g)

    print(f"✓ Pseudobulk capacity: using {len(g)}/{len(machinery_genes)} machinery genes: {g}")
    print(f"✓ Pseudobulk output dir: {outdir}")
    avail = []
    for key,_ in group_keys:
        if key in adata.obs.columns:
            try:
                cats = sorted(list(map(str, pd.unique(adata.obs[key]))))
            except Exception:
                cats = []
            avail.append((key, cats))
    print(f"✓ Available groups: {avail}")

    _osloc.makedirs(outdir, exist_ok=True)
    out = {}
    for key, order in group_keys:
        if key not in adata.obs.columns:
            continue
        means = zdf.groupby(adata.obs[key]).mean()          # rows=groups, cols=genes

        # --- NEW: persist per-gene pseudobulk means so the panel can plot one dot per gene ---
        means_wide = means.copy()
        means_wide.index.name = key
        _out_wide = _osloc.path.join(outdir, f"Pseudobulk_Capacity_{key}_perGene_wide.csv")
        means_wide.to_csv(_out_wide)

        means_long = means.reset_index().melt(id_vars=key, var_name="gene", value_name="z_mean")
        _out_long = _osloc.path.join(outdir, f"Pseudobulk_Capacity_{key}_perGene_long.csv")
        means_long.to_csv(_out_long, index=False)

        if means.empty:
            continue
        cap_mean = means.mean(axis=1)                       # capacity per group
        cap_q1   = means.quantile(0.25, axis=1)
        cap_q3   = means.quantile(0.75, axis=1)
        n_cells  = adata.obs.groupby(key).size()

        tbl = (pd.DataFrame({
            "n_cells": n_cells,
            "n_genes": len(g),
            "capacity_mean": cap_mean,
            "capacity_q1": cap_q1,
            "capacity_q3": cap_q3,
        }).sort_index())

        csv_path = _osloc.path.join(outdir, f"Pseudobulk_Capacity_{key}.csv")
        tbl.to_csv(csv_path)

        # plotting order—only groups that actually exist
        existing = list(tbl.index)
        ord_idx = [x for x in (order or existing) if x in existing]
        vals = tbl.loc[ord_idx, "capacity_mean"].values
        err  = (tbl.loc[ord_idx, "capacity_q3"] - tbl.loc[ord_idx, "capacity_q1"]).values / 2.0

        fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=160)
        ax.bar(np.arange(len(ord_idx)), vals, yerr=err, capsize=4)
        ax.set_xticks(np.arange(len(ord_idx))); ax.set_xticklabels(ord_idx)
        ax.set_ylabel(ylabel); ax.set_title(f"Capacity per {key}")
        fig.tight_layout()
        png_path = _osloc.path.join(outdir, f"Pseudobulk_Capacity_{key}.png")
        fig.savefig(png_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"✅ Pseudobulk capacity → {png_path}")
        out[key] = {"csv": csv_path, "png": png_path}
    return out


def _pick_first_key(d, candidates):
    for k in candidates:
        if k in d:
            return k
    return None


def _ensure_outdir(outdir_base=None):
    base = outdir_base or os.environ.get("DCT_PERT_OUTDIR") or globals().get("data_folder", None) or os.getcwd()
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(base, "InSilico_Perturbation", ts)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _to_dense(X):
    return X.A if sp.issparse(X) else np.asarray(X)


def _mean_z(adata, genes: List[str]):
    if not genes:
        return np.zeros(adata.n_obs)
    X = _to_dense(adata[:, genes].X)
    mu = X.mean(0); sd = X.std(0, ddof=0); sd[sd==0] = 1.0
    Z = (X - mu) / sd
    return np.asarray(Z.mean(1)).ravel()


def _varname_map(adata):
    var_map = {str(v).upper(): str(v) for v in adata.var_names}
    for sym_col in ["gene_symbol","Gene symbol","Gene Symbol","symbol","SYMBOL","Gene","gene"]:
        if sym_col in adata.var.columns:
            for row_name, sym in zip(adata.var_names, adata.var[sym_col].astype(str).values):
                if sym and sym != "nan":
                    var_map[str(sym).upper()] = str(row_name)
    for alias, canon in ALIASES.items():
        if canon.upper() in var_map and alias.upper() not in var_map:
            var_map[alias.upper()] = var_map[canon.upper()]
    return var_map


def _present(var_map, genes: List[str]):
    pres, miss = [], []
    for g in genes:
        gg = g.upper()
        if gg in var_map:
            pres.append(var_map[gg])
        else:
            miss.append(g)
    return pres, miss


def run(adata, outdir_base=None):
    outdir = _ensure_outdir(outdir_base)

    cell_key   = _pick_first_key(adata.obs, CELLLABEL_CANDIDATES)
    pseudo_key = _pick_first_key(adata.obs, PSEUDOTIME_CANDIDATES)
    group_key  = _pick_first_key(adata.obs, PSEUDOBULK_GROUP_CANDIDATES)

    # Ensure UMAP exists
    try:
        if "X_umap" not in adata.obsm:
            if "neighbors" not in adata.uns:
                if "X_pca" not in adata.obsm:
                    sc.pp.pca(adata, n_comps=30, svd_solver="arpack")
                sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca" if "X_pca" in adata.obsm else None)
            sc.tl.umap(adata)
    except Exception as e:
        print(f"⚠️ Could not compute UMAP automatically: {e}")

    var_map = _varname_map(adata)

    # Machinery
    mach_meta = []
    for name, genes in MACHINERY.items():
        pres, miss = _present(var_map, genes)
        adata.obs[name] = _mean_z(adata, pres)
        mach_meta.append({"set": name, "n_present": len(pres), "n_missing": len(miss), "missing": ";".join(miss)})
    pd.DataFrame(mach_meta).to_csv(os.path.join(outdir, "machinery_gene_coverage.csv"), index=False)

    # Signed signatures
    sig_meta = []
    for sig, dd in SIGNATURES.items():
        up_pres, up_miss = _present(var_map, dd.get("UP", []))
        dn_pres, dn_miss = _present(var_map, dd.get("DOWN", []))
        adata.obs[f"{sig}_signed"] = _mean_z(adata, up_pres) - ( _mean_z(adata, dn_pres) if dn_pres else 0.0 )
        sig_meta.append({
            "signature": sig,
            "UP_present": len(up_pres), "UP_missing": len(up_miss),
            "DOWN_present": len(dn_pres), "DOWN_missing": len(dn_miss),
            "UP_missing_list": ";".join(up_miss),
            "DOWN_missing_list": ";".join(dn_miss),
        })
    pd.DataFrame(sig_meta).to_csv(os.path.join(outdir, "signature_gene_coverage.csv"), index=False)

    # Per-cell table
    score_cols = list(MACHINERY.keys()) + [f"{s}_signed" for s in SIGNATURES]
    keep_cols = [c for c in score_cols if c in adata.obs.columns]
    if cell_key:
        keep_cols = [cell_key] + keep_cols
    adata.obs[keep_cols].to_csv(os.path.join(outdir, "PerCell_scores.csv"))

    # Plots helpers
    def _umap(name):
        try:
            fig = sc.pl.umap(adata, color=name, color_map="viridis", show=False, legend_loc=None, return_fig=True)
            fig.savefig(os.path.join(outdir, f"UMAP_{name}.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
        except Exception as e:
            print(f"⚠️ UMAP failed for {name}: {e}")

    for k in ["Aldo_machinery","ALDO_signed","ANGII_signed","THIAZIDE_signed"]:
        if k in adata.obs.columns: _umap(k)

    if cell_key:
        try:
            sc.pl.violin(adata, keys=["Aldo_machinery","ALDO_signed"],
                         groupby=cell_key, stripplot=False, show=False)
            fig = plt.gcf()
            fig.savefig(os.path.join(outdir, "Violin_AldoMachinery_ALDOsigned_byCellType.png"),
                        dpi=160, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Violin failed: {e}")
        (adata.obs[[cell_key,"Aldo_machinery","ALDO_signed"]]
          .groupby(cell_key, observed=True)
          .agg(['median','mean','std','count'])
        ).to_csv(os.path.join(outdir, "Summary_byCellType_Aldo.csv"))

    # DCT1 vs DCT2 single-cell effect size (ALDO_signed)
    def _cliffs_delta(x, y):
        comp = 0
        for xi in x: comp += (y > xi).sum() - (y < xi).sum()
        denom = (x.size * y.size)
        return comp / denom if denom else np.nan

    delta = pval = np.nan
    if cell_key and set(["DCT1","DCT2"]).issubset(set(adata.obs[cell_key].astype(str).unique())) and "ALDO_signed" in adata.obs:
        x = adata.obs.loc[adata.obs[cell_key]=="DCT1","ALDO_signed"].values
        y = adata.obs.loc[adata.obs[cell_key]=="DCT2","ALDO_signed"].values
        try: _, pval = st.mannwhitneyu(x, y, alternative="two-sided")
        except Exception: pval = np.nan
        delta = _cliffs_delta(np.asarray(x), np.asarray(y))
    with open(os.path.join(outdir, "EffectSize_DCT2_vs_DCT1.json"), "w") as fh:
        json.dump({"CliffsDelta": None if (isinstance(delta, float) and math.isnan(delta)) else float(delta),
                   "MannWhitneyP": None if (isinstance(pval, float) and math.isnan(pval)) else float(pval)}, fh, indent=2)

    # Pseudotime trends + trend CSV (ALDO_signed, Aldo_machinery)
    if pseudo_key and pseudo_key in adata.obs:
        def _trend(key, bins=20):
            df = adata.obs[[pseudo_key, key]].dropna().copy()
            if df.empty: return
            df["bin"] = pd.qcut(df[pseudo_key], q=bins, duplicates="drop")
            mid = df.groupby("bin", observed=True)[pseudo_key].median()
            med = df.groupby("bin", observed=True)[key].median()
            lo  = df.groupby("bin", observed=True)[key].quantile(0.25)
            hi  = df.groupby("bin", observed=True)[key].quantile(0.75)
            fig = plt.figure(figsize=(5,3.2), dpi=160)
            plt.plot(mid.values, med.values, marker="o")
            plt.fill_between(mid.values, lo.values, hi.values, alpha=0.25)
            plt.xlabel(pseudo_key); plt.ylabel(key); plt.title(f"{key} vs {pseudo_key}")
            plt.tight_layout()
            fig.savefig(os.path.join(outdir, f"Trend_{key}_vs_{pseudo_key}.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
        for k in ["Aldo_machinery","ALDO_signed"]:
            if k in adata.obs.columns: _trend(k)

        # Export 10-bin medians
        try:
            _obs = adata.obs[[pseudo_key]].copy()
            for col in [c for c in ["Aldo_machinery","ALDO_signed"] if c in adata.obs.columns]:
                _obs[col] = adata.obs[col]
            _obs = _obs.dropna()
            _obs["pt_bin"] = pd.qcut(_obs[pseudo_key], q=10, duplicates="drop")
            med = _obs.groupby("pt_bin", observed=True).median(numeric_only=True)
            med.to_csv(os.path.join(outdir, "trend_bins_median.csv"))
        except Exception as e:
            print(f"⚠️ Trend-bin CSV export failed: {e}")

    # Correlation: machinery vs ALDO_signed
    try:
        x = adata.obs.get("Aldo_machinery", None)
        y = adata.obs.get("ALDO_signed", None)
        if x is not None and y is not None:
            m = np.isfinite(x) & np.isfinite(y)
            r, p = st.spearmanr(np.asarray(x)[m], np.asarray(y)[m])
            with open(os.path.join(outdir, "Aldo_machinery_vs_ALDO_signed_spearman.json"), "w") as fh:
                json.dump({"spearman_r": float(r), "p_value": float(p), "n": int(m.sum())}, fh, indent=2)
    except Exception as e:
        print(f"⚠️ Correlation failed: {e}")

    # Pseudobulk robustness for ALDO_signed & Aldo_machinery
    if group_key and cell_key:
        cols = [c for c in ["Aldo_machinery","ALDO_signed","ANGII_signed"] if c in adata.obs]
        if cols:
            df = adata.obs[[group_key, cell_key] + cols].copy()
            agg = df.groupby([group_key, cell_key], observed=True).median().reset_index()
            agg.to_csv(os.path.join(outdir, f"Pseudobulk_{group_key}_median_scores.csv"), index=False)
            rows = []
            if set(["DCT1","DCT2"]).issubset(set(agg[cell_key].unique())):
                if "ALDO_signed" in agg.columns:
                    d1 = agg.loc[agg[cell_key]=="DCT1","ALDO_signed"].values
                    d2 = agg.loc[agg[cell_key]=="DCT2","ALDO_signed"].values
                    try: t,p = st.ttest_ind(d1, d2, equal_var=False)
                    except Exception: t,p = np.nan, np.nan
                    rows.append({"score":"ALDO_signed","t":t,"p":p,"mean_DCT1":np.mean(d1),"mean_DCT2":np.mean(d2),"diff":np.mean(d2)-np.mean(d1)})
                if "Aldo_machinery" in agg.columns:
                    d1m = agg.loc[agg[cell_key]=="DCT1","Aldo_machinery"].values
                    d2m = agg.loc[agg[cell_key]=="DCT2","Aldo_machinery"].values
                    try: t2,p2 = st.ttest_ind(d1m, d2m, equal_var=False)
                    except Exception: t2,p2 = np.nan, np.nan
                    rows.append({"score":"Aldo_machinery","t":t2,"p":p2,"mean_DCT1":np.mean(d1m),"mean_DCT2":np.mean(d2m),"diff":np.mean(d2m)-np.mean(d1m)})
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(outdir, "Pseudobulk_DCT2_vs_DCT1_ttest.csv"), index=False)

    # === Pseudobulk capacity block (added) ===
    try:
        # make early/mid/late if missing
        ensure_bin3_from_pseudotime(adata, pseudo_key="dpt_pseudotime", bin_key="bin3")

        aldo_mach = MACHINERY.get("Aldo_machinery", ["NR3C2","HSD11B2","WNK4","STK39"])

        # Resolve machinery genes case-insensitively via var-name map
        _vm = _varname_map(adata)
        _aldo_resolved, _miss = _present(_vm, aldo_mach)

        if _aldo_resolved:
            aldo_mach = _aldo_resolved
        else:
            # Fallback: try extended machinery rich in detectable co-factors
            _ext_list = ALDO_MACHINERY_EXT
            _aldo_resolved2, _miss2 = _present(_vm, _ext_list)
            if _aldo_resolved2:
                print(f"ℹ️  Falling back to extended machinery ({len(_aldo_resolved2)} genes): {_aldo_resolved2}")
                aldo_mach = _aldo_resolved2
            else:
                print("⚠️ No Aldo machinery genes resolved even with extended list; pseudobulk will be skipped.")

        # groups to aggregate: pseudotime bins + (optionally) cell type
        group_keys = [("bin3", ["early","mid","late"])]
        if "predicted_cell_type" in adata.obs.columns:
            group_keys.append(("predicted_cell_type", None))
        else:
            # auto-pick a *_named column if present
            for col in adata.obs.columns:
                if col.lower().endswith("_named"):
                    group_keys.append((col, None))
                    break

        _pb = pseudobulk_capacity(
            adata=adata,
            outdir=outdir,   # ← use the local run() outdir
            machinery_genes=aldo_mach,
            group_keys=group_keys,
            ylabel="Aldo_machinery (pseudobulk z)"
        )

    except Exception as e:
        print(f"⚠️ Pseudobulk capacity block failed: {e}")
    # === end block ===

    # === In-silico responsiveness: stats + one-slide panel =======================
    try:
        # ensure bin3 exists (early/mid/late)
        if "bin3" not in adata.obs.columns:
            ensure_bin3_from_pseudotime(adata, pseudo_key="dpt_pseudotime", bin_key="bin3")
        order = [x for x in ["early","mid","late"] if x in list(pd.unique(adata.obs["bin3"]))]

        # ---- segmented activity (per cell) -------------------------------------
        def _seg_stats(series, grp_key, order_vals):
            out = []
            for g in order_vals:
                m = (adata.obs[grp_key] == g).values
                v = pd.to_numeric(series[m], errors="coerce").values
                v = v[np.isfinite(v)]
                if v.size:
                    q1, q3 = np.nanpercentile(v, [25, 75])
                    med = float(np.nanmedian(v))
                    hiqr = float((q3 - q1) / 2.0)
                    n = int(v.size)
                else:
                    med = hiqr = np.nan; n = 0
                out.append((g, med, hiqr, n))
            return pd.DataFrame(out, columns=["bin", "median", "half_IQR", "n"])

        s_act = pd.Series(adata.obs.get("ALDO_activity_noMach", np.nan), index=adata.obs_names)
        if (not np.isfinite(s_act).any()) or (np.nanstd(s_act.values) < 1e-8):
            # fall back to the signed score if noMach is missing/constant
            s_act = pd.Series(adata.obs.get("ALDO_signed", np.nan), index=adata.obs_names)

        act_tbl = _seg_stats(s_act, "bin3", order)
        act_tbl.to_csv(os.path.join(outdir, "Activity_bin3_segmented.csv"), index=False)

        # ---- capacity (pseudobulk, gene-wise then averaged) --------------------
        # Resolve machinery gene names present in var_names (case-insensitive, alias-aware)
        _vm = _varname_map(adata)
        _seed_mach = MACHINERY.get("Aldo_machinery", ["NR3C2","HSD11B2","WNK4","STK39"])
        aldo_resolved, _miss = _present(_vm, _seed_mach)
        if not aldo_resolved and "ALDO_MACHINERY_EXT" in globals():
            aldo_resolved, _ = _present(_vm, ALDO_MACHINERY_EXT)

        if len(aldo_resolved) == 0:
            print("⚠️ Capacity stats skipped (no machinery genes resolved).")
            raise RuntimeError("no_capacity_genes")

        X = adata[:, aldo_resolved].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        Z = _zscore_frame(X, np)                                     # z across cells (per gene)
        zdf = pd.DataFrame(Z, index=adata.obs_names, columns=aldo_resolved)
        means = zdf.groupby(adata.obs["bin3"], observed=True).mean() # mean z per gene per bin

        cap_mean = means.mean(axis=1)
        cap_q1   = means.quantile(0.25, axis=1)
        cap_q3   = means.quantile(0.75, axis=1)
        cap_tbl = (pd.DataFrame({
            "bin": means.index,
            "capacity_mean": cap_mean.values,
            "capacity_half_IQR": ((cap_q3 - cap_q1)/2.0).values,
            "n_cells": adata.obs.groupby("bin3", observed=True).size().reindex(means.index).values,
            "n_genes": len(aldo_resolved)
        }).set_index("bin").loc[[b for b in order if b in means.index]].reset_index())
        cap_tbl.to_csv(os.path.join(outdir, "Capacity_bin3_segmented.csv"), index=False)

        # ---- pairwise stats helpers --------------------------------------------
        def _paired_wilcoxon_and_boot(a, b, n_boot=2000, seed=42):
            a = np.asarray(a); b = np.asarray(b)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 3:
                return dict(delta=np.nan, p=np.nan, ci95_low=np.nan, ci95_high=np.nan, n=int(m.sum()))
            delta = float((b[m] - a[m]).mean())
            try:
                p = float(st.wilcoxon(b[m], a[m], zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
            rng = np.random.default_rng(seed)
            idx = np.arange(m.sum())
            boots = [(b[m][s] - a[m][s]).mean() for s in (rng.choice(idx, size=idx.size, replace=True) for _ in range(n_boot))]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            return dict(delta=delta, p=p, ci95_low=float(lo), ci95_high=float(hi), n=int(m.sum()))

        stats_rows = []

        # capacity: paired across genes (same genes in each bin)
        for g1, g2 in [("early","mid"),("mid","late"),("early","late")]:
            if g1 in means.index and g2 in means.index:
                res = _paired_wilcoxon_and_boot(means.loc[g1].values, means.loc[g2].values)
                res.update({"contrast": f"{g2}-{g1}", "type": "capacity_genewise"})
                stats_rows.append(res)

        # activity: across cells (two-sample)
        vals = {b: s_act[adata.obs["bin3"]==b].values for b in order}
        for g1, g2 in [("early","mid"),("mid","late"),("early","late")]:
            if g1 in vals and g2 in vals:
                a = vals[g1]; b = vals[g2]
                a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                if a.size > 2 and b.size > 2:
                    delta = float(np.nanmedian(b) - np.nanmedian(a))
                    p = float(st.mannwhitneyu(b, a, alternative="two-sided").pvalue)
                else:
                    delta = np.nan; p = np.nan
                stats_rows.append({"contrast": f"{g2}-{g1}", "type": "activity_cells",
                                   "delta": delta, "p": p, "ci95_low": np.nan, "ci95_high": np.nan,
                                   "n": int(a.size + b.size)})

        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(outdir, "Responsiveness_pairwise_stats.csv")
        stats_df.to_csv(stats_path, index=False)

        # ===================== One-slide responsiveness panel (CSV-backed dots) =====================
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=200)

        # Give both panels a bit of extra vertical headroom so brackets/labels
        # don't collide with the top frame (fixes mid→late label on the left plot).
        for _ax in axes:
            y0, y1 = _ax.get_ylim()
            _ax.set_ylim(y0, y1 + 0.08 * (y1 - y0))  # 8% headroom

        # --- Backward-compatibility shims (harmless if unused) ---
        def _add_sig_bar(ax, x1, x2, y, text, h=0.03, lw=1.0, fs=8):
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], c="black", lw=lw, clip_on=False, zorder=4)
            ax.text((x1+x2)/2.0, y+h, text, ha="center", va="bottom", fontsize=fs, zorder=4)

        # old code may call draw_bracket(...) – keep an alias
        draw_bracket = _add_sig_bar

        def draw_bracket_above(ax, x1, x2, y_ax, text, lw=1.2, fontsize=10):
            """
            Draw a bracket and label above the axes frame.
            x1, x2 in DATA coordinates (e.g., 0,1,2 for categories).
            y_ax in AXES coordinates (>1 means above the frame).
            """
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            # short vertical ticks plus top horizontal line
            ax.plot([x1, x1, x2, x2],
                    [y_ax - 0.01, y_ax, y_ax, y_ax - 0.01],
                    transform=trans, color='k', lw=lw, clip_on=False)
            ax.text((x1 + x2) / 2.0, y_ax + 0.005, text,
                    transform=trans, ha='center', va='bottom',
                    fontsize=fontsize, color='k', clip_on=False)

        # old code may expect a rows_to_plot list of pairwise comparisons – define a default
        if 'rows_to_plot' not in locals():
            rows_to_plot = [("early", "mid"), ("mid", "late"), ("early", "late")]

        SHOW_TITLES = False  # set True if you want titles back
        tick_fs, label_fs, title_fs, annot_fs = 9, 10, 13, 8
        capsize, dot_alpha, dot_size, jitter_sd = 3.5, 0.28, 11, 0.06

        # -------------------- Activity (per-cell) --------------------
        axes[0].bar(np.arange(len(order)), act_tbl["median"].values,
                    yerr=act_tbl["half_IQR"].values, capsize=capsize, zorder=1)
        axes[0].set_xticks(np.arange(len(order))); axes[0].set_xticklabels(order, fontsize=tick_fs)
        axes[0].set_ylabel("ALDO activity (median ± half-IQR)", fontsize=label_fs)
        if SHOW_TITLES:
            axes[0].set_title("Per-cell activity (noMach)", fontsize=title_fs, pad=12)

        # overlay per-cell dots (jittered)
        for i, g in enumerate(order):
            m = (adata.obs["bin3"] == g).values
            vals = pd.to_numeric(pd.Series(s_act)[m], errors="coerce").values
            vals = vals[np.isfinite(vals)]
            if vals.size:
                xs = np.random.normal(loc=i, scale=jitter_sd, size=vals.size)
                axes[0].scatter(xs, vals, s=dot_size, alpha=dot_alpha, color="k",
                                linewidths=0, zorder=3, rasterized=True)

        # headroom for annotations
        y0 = np.asarray(act_tbl["median"].values, float)
        e0 = np.nan_to_num(np.asarray(act_tbl["half_IQR"].values, float))
        r0 = float(np.nanmax(np.abs(y0) + e0)); r0 = 0.25 if not np.isfinite(r0) or r0 < 0.25 else r0
        axes[0].set_ylim(-1.25*r0, 1.45*r0)

        # -------------------- Capacity (pseudobulk) --------------------
        axes[1].bar(np.arange(len(order)), cap_tbl["capacity_mean"].values,
                    yerr=cap_tbl["capacity_half_IQR"].values, capsize=capsize, zorder=1)
        axes[1].set_xticks(np.arange(len(order))); axes[1].set_xticklabels(order, fontsize=tick_fs)
        axes[1].set_ylabel("ALDO machinery (pseudobulk z)", fontsize=label_fs)
        if SHOW_TITLES:
            axes[1].set_title("Pseudobulk capacity", fontsize=title_fs, pad=12)

        # --- NEW: draw colored per-gene dots on top of the capacity bars ---
        dot_done = False
        csv_pb = os.path.join(outdir, "Pseudobulk_Capacity_bin3_perGene_wide.csv")
        if os.path.exists(csv_pb):
            try:
                dfpb = pd.read_csv(csv_pb, index_col=0)  # index: early/mid/late, columns: genes
                # build a stable palette by gene name
                genes = list(dfpb.columns)
                cmap  = plt.get_cmap("tab10")
                gene_color = {g: cmap(i % 10) for i, g in enumerate(sorted(genes))}

                # jitter + dots
                for i, b in enumerate(order):  # order already defined as ["early","mid","late"]
                    if b not in dfpb.index:
                        continue
                    row = pd.to_numeric(dfpb.loc[b], errors="coerce")
                    for j, g in enumerate(genes):
                        val = row.get(g)
                        if pd.notna(val):
                            xj = np.random.normal(i, jitter_sd/1.8)  # light jitter
                            axes[1].scatter(
                                xj, val, s=dot_size+8, color=gene_color[g],
                                alpha=0.75, edgecolor="white", linewidths=0.25,
                                zorder=3, rasterized=True, label=g
                            )
                            dot_done = True

                # de-duplicate legend entries and park legend outside the right panel
                if dot_done:
                    handles, labels = axes[1].get_legend_handles_labels()
                    by_label = {}
                    for h, lab in zip(handles, labels):
                        if lab not in by_label and lab in genes:  # keep only gene labels
                            by_label[lab] = h
                    leg = axes[1].legend(
                        by_label.values(), by_label.keys(),
                        title="Machinery genes", fontsize=8, title_fontsize=9,
                        loc="center left", bbox_to_anchor=(1.02, 0.5),
                        frameon=False, borderaxespad=0.0, handletextpad=0.4
                    )
                    # make room for the legend outside the axes
                    fig.subplots_adjust(right=0.86)
            except Exception:
                pass

        # ensure y-limits include both bars and dots (prevents cropping)
        ybar = np.asarray(cap_tbl["capacity_mean"].values, float)
        ebar = np.nan_to_num(np.asarray(cap_tbl["capacity_half_IQR"].values, float))
        low  = np.nanmin(ybar - ebar)
        high = np.nanmax(ybar + ebar)
        if dot_done:
            # include dot extrema
            dot_vals = pd.read_csv(csv_pb, index_col=0).apply(pd.to_numeric, errors="coerce")
            low  = np.nanmin([low,  dot_vals.min().min()])
            high = np.nanmax([high, dot_vals.max().max()])
        if not (np.isfinite(low) and np.isfinite(high)) or low == high:
            low, high = -0.25, 0.25
        margin = 0.12 * (high - low)
        axes[1].set_ylim(low - margin, high + margin)

        # Fallback: try in-memory resolver if CSV missing/unreadable
        if not dot_done:
            ALIASES = {"MR":"NR3C2","SPAK":"STK39","NCC":"SLC12A3","AT1R":"AGTR1",
                       "PKA":"PRKACA","ENACB":"SCNN1B","ENACG":"SCNN1G"}
            def _resolve_varnames(adata, genes, aliases=None):
                var_map = {str(v).upper(): v for v in adata.var_names}
                out = []
                for g in genes or []:
                    if not isinstance(g, str): continue
                    g1 = (aliases or {}).get(g, g)
                    for cand in (g1, g1.upper(), g1.lower(), g):
                        key = str(cand).upper()
                        if key in var_map:
                            v = var_map[key]
                            if v not in out: out.append(v)
                            break
                return out

            mach_lists = []
            for name in ("machinery_used","mach_genes_used","mach_used","mach_final","machinery_final"):
                if name in locals() and isinstance(locals()[name], (list, tuple)) and locals()[name]:
                    mach_lists.append(list(locals()[name]))
            if not mach_lists:
                cov_csv = os.path.join(outdir, "machinery_gene_coverage.csv")
                if os.path.exists(cov_csv):
                    try:
                        df_cov = pd.read_csv(cov_csv)
                        genes = df_cov.iloc[:,0].astype(str).tolist()
                        mach_lists.append(genes)
                    except Exception:
                        pass

            mach_vars = _resolve_varnames(adata, mach_lists[0] if mach_lists else [], aliases=ALIASES)
            if mach_vars:
                X = adata[:, mach_vars].X
                if sp.issparse(X): X = X.toarray()
                mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-8
                Z  = (X - mu) / sd
                zdf = pd.DataFrame(Z, index=adata.obs_names, columns=mach_vars)
                for i, g in enumerate(order):
                    m = (adata.obs["bin3"] == g).values
                    if m.sum():
                        vals = zdf.loc[m].mean(axis=0).values  # per-gene mean z within the bin
                        xs = np.random.normal(loc=i, scale=jitter_sd/1.5, size=vals.size)
                        axes[1].scatter(xs, vals, s=dot_size+6, alpha=0.35, color="k",
                                        linewidths=0, zorder=3, rasterized=True)

        # headroom for capacity brackets
        y1 = np.asarray(cap_tbl["capacity_mean"].values, float)
        e1 = np.nan_to_num(np.asarray(cap_tbl["capacity_half_IQR"].values, float))
        r1 = float(np.nanmax(np.abs(y1) + e1)); r1 = 0.25 if not np.isfinite(r1) or r1 < 0.25 else r1
        axes[1].set_ylim(-1.25*r1, 1.45*r1)

        # -------------------- Bracket annotations (draw ABOVE the axes frame) --------------------
        if 'stats_df' in locals() and isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
            idx = {b: i for i, b in enumerate(order)}
            pairs = [("early", "mid"), ("mid", "late"), ("early", "late")]
            y_levels = [1.02, 1.12, 1.22]  # axes coordinates ( >1 sits above the top frame )

            # Left panel: activity
            sdfA = stats_df[stats_df["type"] == "activity_cells"].set_index("contrast")
            for (a, b), y_ax in zip(pairs, y_levels):
                if a in idx and b in idx:
                    key = f"{b}-{a}"
                    txt = "n/a"
                    if key in sdfA.index:
                        d = sdfA.loc[key].to_dict()
                        txt = f"Δ={d.get('delta', float('nan')):+.2f}, p={d.get('p', float('nan')):.2g}"
                    draw_bracket_above(axes[0], idx[a], idx[b], y_ax, txt, lw=1.2, fontsize=10)

            # Right panel: capacity
            sdfC = stats_df[stats_df["type"] == "capacity_genewise"].set_index("contrast")
            for (a, b), y_ax in zip(pairs, y_levels):
                if a in idx and b in idx:
                    key = f"{b}-{a}"
                    txt = "n/a"
                    if key in sdfC.index:
                        d = sdfC.loc[key].to_dict()
                        txt = f"Δ={d.get('delta', float('nan')):+.2f}, p={d.get('p', float('nan')):.2g}"
                    draw_bracket_above(axes[1], idx[a], idx[b], y_ax, txt, lw=1.2, fontsize=10)

        for ax in axes:
            ax.tick_params(labelsize=tick_fs)

        fig.tight_layout()
        panel_path = os.path.join(outdir, "Panel_Responsiveness_Summary.png")
        fig.savefig(panel_path, dpi=220, bbox_inches="tight"); plt.close(fig)
        print(f"✅ Pairwise stats → {stats_path}")
        print(f"✅ One-slide panel → {panel_path}")

        # ---- capacity by cell type: quick paired test DCT2 vs DCT1 ------------
        if "predicted_cell_type" in adata.obs.columns:
            means_ct = zdf.groupby(adata.obs["predicted_cell_type"], observed=True).mean()
            if {"DCT1","DCT2"}.issubset(set(means_ct.index)):
                res = _paired_wilcoxon_and_boot(means_ct.loc["DCT1"].values, means_ct.loc["DCT2"].values)
                res.update({"contrast": "DCT2-DCT1", "type": "capacity_celltype_genewise"})
                pd.DataFrame([res]).to_csv(os.path.join(outdir, "Capacity_celltype_pairwise_stats.csv"), index=False)
                print("✅ Capacity cell-type stats →", os.path.join(outdir, "Capacity_celltype_pairwise_stats.csv"))

    except Exception as e:
        print("⚠️ Stats/panel block failed:", e)
    # ============================================================================

    # ----- De-overlapped ALDO activity (exclude machinery) -----
    aldo_up = SIGNATURES["ALDO"]["UP"]
    mach_set = set([g.upper() for g in MACHINERY["Aldo_machinery"]])
    activity_candidates = sorted(list(set([g.upper() for g in aldo_up]) - mach_set))
    if len(activity_candidates) < 3:
        activity_candidates = ["SGK1","NEDD4L","SLC12A3","ATP1A1","ATP1B1"]
    pres, miss = _present(var_map, activity_candidates)
    with open(os.path.join(outdir, "ALDO_activity_noMach_gene_coverage.json"), "w") as fh:
        json.dump({"candidates": activity_candidates, "present": pres, "missing": miss}, fh, indent=2)
    if pres:
        adata.obs["ALDO_activity_noMach"] = _mean_z(adata, pres)
        # UMAP
        try:
            fig = sc.pl.umap(adata, color="ALDO_activity_noMach", color_map="viridis", show=False, legend_loc=None, return_fig=True)
            fig.savefig(os.path.join(outdir, "UMAP_ALDO_activity_noMach.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
        except Exception as e:
            print(f"⚠️ UMAP for ALDO_activity_noMach failed: {e}")
        # Trend
        if pseudo_key and pseudo_key in adata.obs:
            try:
                df = adata.obs[[pseudo_key, "ALDO_activity_noMach"]].dropna().copy()
                if not df.empty:
                    df["bin"] = pd.qcut(df[pseudo_key], q=10, duplicates="drop")
                    mid = df.groupby("bin", observed=True)[pseudo_key].median()
                    med = df.groupby("bin", observed=True)["ALDO_activity_noMach"].median()
                    lo  = df.groupby("bin", observed=True)["ALDO_activity_noMach"].quantile(0.25)
                    hi  = df.groupby("bin", observed=True)["ALDO_activity_noMach"].quantile(0.75)
                    fig = plt.figure(figsize=(5,3.2), dpi=160)
                    plt.plot(mid.values, med.values, marker="o")
                    plt.fill_between(mid.values, lo.values, hi.values, alpha=0.25)
                    plt.xlabel(pseudo_key); plt.ylabel("ALDO_activity_noMach"); plt.title("ALDO_activity_noMach vs " + pseudo_key)
                    plt.tight_layout(); fig.savefig(os.path.join(outdir, f"Trend_ALDO_activity_noMach_vs_{pseudo_key}.png"), dpi=160, bbox_inches="tight"); plt.close(fig)
                # Spearman vs pseudotime
                m = np.isfinite(adata.obs[pseudo_key]) & np.isfinite(adata.obs["ALDO_activity_noMach"])
                r, p = st.spearmanr(np.asarray(adata.obs[pseudo_key])[m], np.asarray(adata.obs["ALDO_activity_noMach"])[m])
                with open(os.path.join(outdir, "ALDO_activity_noMach_vs_pseudotime_spearman.json"), "w") as fh:
                    json.dump({"spearman_r": float(r), "p_value": float(p), "n": int(m.sum())}, fh, indent=2)
            except Exception as e:
                print(f"⚠️ De-overlap trend/association failed: {e}")
        # Correlation with machinery
        try:
            x = adata.obs.get("Aldo_machinery", None)
            y = adata.obs.get("ALDO_activity_noMach", None)
            if x is not None and y is not None:
                m = np.isfinite(x) & np.isfinite(y)
                r, p = st.spearmanr(np.asarray(x)[m], np.asarray(y)[m])
                with open(os.path.join(outdir, "Aldo_machinery_vs_ALDO_activity_noMach_spearman.json"), "w") as fh:
                    json.dump({"spearman_r": float(r), "p_value": float(p), "n": int(m.sum())}, fh, indent=2)
        except Exception as e:
            print(f"⚠️ De-overlap correlation failed: {e}")
        # Pseudobulk test
        if group_key and cell_key:
            try:
                cols = [group_key, cell_key, "ALDO_activity_noMach"]
                _df = adata.obs[cols].dropna().copy()
                if not _df.empty:
                    agg = _df.groupby([group_key, cell_key], observed=True).median().reset_index()
                    if set(["DCT1","DCT2"]).issubset(set(agg[cell_key].astype(str).unique())):
                        d1 = agg.loc[agg[cell_key]=="DCT1","ALDO_activity_noMach"].values
                        d2 = agg.loc[agg[cell_key]=="DCT2","ALDO_activity_noMach"].values
                        try: t,p = st.ttest_ind(d1, d2, equal_var=False)
                        except Exception: t,p = np.nan, np.nan
                        pd.DataFrame([{"score":"ALDO_activity_noMach","t":t,"p":p,"mean_DCT1":np.mean(d1),"mean_DCT2":np.mean(d2),"diff":np.mean(d2)-np.mean(d1)}]) \
                          .to_csv(os.path.join(outdir, "Pseudobulk_DCT2_vs_DCT1_ttest_ALDO_activity_noMach.csv"), index=False)

    # Manifest
            except Exception as _e:
                print(f"⚠️ Block failed: {_e}")
    try:
        items = [fn for fn in os.listdir(outdir) if fn.lower().endswith((".png",".csv",".json"))]
        items.sort()
        print("📦 In-silico outputs:")
        for fn in items: print(" -", os.path.join(outdir, fn))
    except Exception as e:
        print(f"⚠️ Could not list outputs: {e}")

    print(f"✅ In-silico perturbation done → {outdir}")
    return outdir

_pert_outdir = run(
    adata,
    outdir_base=r"C:\Users\au520563\OneDrive - Aarhus universitet\DATA\20251021\DCT"
)
# =========================
# Export UMAP CSV for the browser (coords + optional celltype/pseudotime)
# =========================


def export_umap_csv_dct(celltype_col=None):
    cols = ['UMAP1','UMAP2']
    df = pd.DataFrame(adata.obsm['X_umap'], columns=cols)
    if celltype_col and celltype_col in adata.obs.columns:
        df['celltype'] = adata.obs[celltype_col].astype(str).values
    if 'dpt_pseudotime' in adata.obs.columns:
        df['dpt_pseudotime'] = adata.obs['dpt_pseudotime'].values
    df['cell_id'] = adata.obs_names.astype(str).values
    out = os.path.join(data_folder, 'UMAP_coords_with_meta.csv')
    df.to_csv(out, index=False)
    print(f"✅ UMAP CSV exported → {out}")

# =========================
# Segmented pseudotime utilities (early, mid, late) + one-sided tests
# =========================


def _cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    comp = 0
    for xi in x:
        comp += (y > xi).sum() - (y < xi).sum()
    denom = x.size * y.size
    return (comp / denom) if denom else np.nan


def _segment_by_quantiles(series, n_segments=3, labels=None):
    import pandas as _p
    if labels is None:
        labels = ["early","mid","late"][:n_segments]
    q = _p.qcut(series, q=n_segments, labels=labels, duplicates="drop")
    return q.astype(str)


def compute_segmented_stats(adata, pseudo_key, score_keys, n_segments=3, labels=None):
    if labels is None:
        labels = ["early","mid","late"][:n_segments]
    out_rows = []
    for key in score_keys:
        if key not in adata.obs.columns:
            continue
        df = adata.obs[[pseudo_key, key]].dropna().copy()
        if df.empty:
            continue
        df['segment'] = _segment_by_quantiles(df[pseudo_key], n_segments=n_segments, labels=labels)
        grp = df.groupby('segment', observed=True)[key]
        med = grp.median()
        q1  = grp.quantile(0.25)
        q3  = grp.quantile(0.75)
        n   = grp.size()
        order = [lab for lab in labels if lab in med.index]
        if len(order) >= 3:
            a = df.loc[df['segment']==order[0], key].values
            b = df.loc[df['segment']==order[1], key].values
            c = df.loc[df['segment']==order[2], key].values
            try:
                p_up = st.mannwhitneyu(a, b, alternative="less").pvalue   # early < mid (increase)
            except Exception:
                p_up = np.nan
            cd_up = _cliffs_delta(a, b)
            try:
                p_dn = st.mannwhitneyu(b, c, alternative="greater").pvalue # mid > late (decrease)
            except Exception:
                p_dn = np.nan
            cd_dn = _cliffs_delta(b, c)
            out_rows.append({
                "score": key,
                "n_segments": len(order),
                "segment_labels": "|".join(order),
                "median_early": float(med.loc[order[0]]),
                "median_mid":   float(med.loc[order[1]]),
                "median_late":  float(med.loc[order[2]]),
                "IQR_early":    float(q3.loc[order[0]] - q1.loc[order[0]]),
                "IQR_mid":      float(q3.loc[order[1]] - q1.loc[order[1]]),
                "IQR_late":     float(q3.loc[order[2]] - q1.loc[order[2]]),
                "n_early":      int(n.loc[order[0]]),
                "n_mid":        int(n.loc[order[1]]),
                "n_late":       int(n.loc[order[2]]),
                "delta_early_to_mid": float(med.loc[order[1]] - med.loc[order[0]]),
                "delta_mid_to_late":  float(med.loc[order[2]] - med.loc[order[1]]),
                "p_early_lt_mid":     None if (isinstance(p_up,float) and np.isnan(p_up)) else float(p_up),
                "cliffs_delta_early_vs_mid": None if (isinstance(cd_up,float) and np.isnan(cd_up)) else float(cd_up),
                "p_mid_gt_late":      None if (isinstance(p_dn,float) and np.isnan(p_dn)) else float(p_dn),
                "cliffs_delta_mid_vs_late":  None if (isinstance(cd_dn,float) and np.isnan(cd_dn)) else float(cd_dn),
            })
    return pd.DataFrame(out_rows)


def plot_segmented_bars(adata, outdir, pseudo_key, key, n_segments=3, labels=None):
    if labels is None:
        labels = ["early","mid","late"][:n_segments]
    if key not in adata.obs.columns or pseudo_key not in adata.obs.columns:
        return None
    df = adata.obs[[pseudo_key, key]].dropna().copy()
    if df.empty:
        return None
    df['segment'] = _segment_by_quantiles(df[pseudo_key], n_segments=n_segments, labels=labels)
    grp = df.groupby('segment', observed=True)[key]
    med = grp.median()
    q1  = grp.quantile(0.25)
    q3  = grp.quantile(0.75)
    order = [lab for lab in labels if lab in med.index]
    y   = [med.loc[lab] for lab in order]
    err = [(q3.loc[lab]-q1.loc[lab])/2.0 for lab in order]
    fig = plt.figure(figsize=(5.2,3.2), dpi=160)
    xs  = np.arange(len(order))
    plt.bar(xs, y, yerr=err, capsize=4)
    plt.xticks(xs, order); plt.ylabel(key); plt.title(f"{key}: segmented medians ± half-IQR")
    plt.tight_layout()
    fn = os.path.join(outdir, f"Segmented_{key}_bars.png")
    fig.savefig(fn, dpi=160, bbox_inches="tight"); plt.close(fig)
    return fn


def compute_and_plot_segmented(adata, outdir, pseudo_key, score_keys, n_segments=3, labels=None):
    os.makedirs(outdir, exist_ok=True)
    stats = compute_segmented_stats(adata, pseudo_key, score_keys, n_segments=n_segments, labels=labels)
    csv_path = os.path.join(outdir, "SegmentedTrend_stats.csv")
    stats.to_csv(csv_path, index=False)
    made = []
    for k in score_keys:
        fn = plot_segmented_bars(adata, outdir, pseudo_key, k, n_segments=n_segments, labels=labels)
        if fn: made.append(fn)
    return {"csv": csv_path, "figs": made}

# =========================
# ProLIF placement on pseudotime axis (density + ALDO vs pseudotime by cluster)
# =========================


def plot_pseudotime_density_by_cluster(
    adata,
    outdir,
    pseudo_key="dpt_pseudotime",
    cluster_key="predicted_cell_type",
    bw_adjust=1.0,
    focus_clusters=("DCT1", "DCT2", "ProLIF"),
    base_font=16,
):
    """Plot KDE density of pseudotime per cluster and mark medians for focus clusters (no text labels)."""
    if pseudo_key not in adata.obs.columns or cluster_key not in adata.obs.columns:
        print(f"ℹ️  Missing '{pseudo_key}' or '{cluster_key}' — skipping pseudotime density plot.")
        return None

    os.makedirs(outdir, exist_ok=True)
    df = adata.obs[[pseudo_key, cluster_key]].copy().dropna()
    if df.empty:
        print("ℹ️  No data after dropping NA — skipping pseudotime density plot.")
        return None

    # keep only clusters with enough cells
    counts = df[cluster_key].astype(str).value_counts()
    keep = counts[counts >= 10].index.tolist()
    df = df[df[cluster_key].astype(str).isin(keep)]
    if df.empty:
        print("ℹ️  No clusters with >=10 cells — skipping pseudotime density plot.")
        return None

    clusters = list(counts.loc[keep].index)
    focus_present = [c for c in focus_clusters if c in clusters]
    rest = [c for c in clusters if c not in focus_present]
    rest = sorted(rest, key=lambda c: int(counts.get(c, 0)), reverse=True)
    order = focus_present + rest

    with plt.rc_context({
        "font.size": base_font,
        "axes.labelsize": base_font + 2,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font - 2,
    }):
        fig = plt.figure(figsize=(7.2, 4.6), dpi=200)
        ax = plt.gca()

        for c in order:
            sub = df[df[cluster_key].astype(str) == c]
            if sub.shape[0] < 10:
                continue
            sns.kdeplot(
                sub[pseudo_key].values,
                bw_adjust=bw_adjust,
                ax=ax,
                label=str(c),
            )

        # Mark medians for the focus clusters (no text labels)
        for c in focus_present:
            sub = df[df[cluster_key].astype(str) == c]
            if sub.empty:
                continue
            med = float(np.nanmedian(sub[pseudo_key].values))
            ax.axvline(med, linestyle="--", linewidth=1)

        ax.set_xlabel(pseudo_key)
        ax.set_ylabel("Density")
        # no title
        ax.legend(frameon=False)
        plt.tight_layout()

        fn = os.path.join(outdir, f"Pseudotime_density_by_{cluster_key}.png")
        fig.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fn


def plot_score_vs_pseudotime_stratified(
    adata,
    outdir,
    pseudo_key="dpt_pseudotime",
    score_key="ALDO_activity_noMach",
    cluster_key="predicted_cell_type",
    focus_clusters=("DCT1", "DCT2", "ProLIF"),
    n_bins=30,
    min_cells_per_cluster=20,
    annotate_midlate_stats=True,
    midlate_window=(0.65, 0.85),
    prof_cluster="ProLIF",
    ref_clusters=("DCT1", "DCT2"),
    stats_location="upper_left",  # or "lower_left"
    base_font=16,
):
    """Plot score vs pseudotime with cluster-stratified binned-median smoothers and IQR shading."""
    if (
        pseudo_key not in adata.obs.columns
        or score_key not in adata.obs.columns
        or cluster_key not in adata.obs.columns
    ):
        print(f"ℹ️  Missing '{pseudo_key}', '{score_key}', or '{cluster_key}' — skipping stratified smoother plot.")
        return None

    os.makedirs(outdir, exist_ok=True)
    df = adata.obs[[pseudo_key, score_key, cluster_key]].copy().dropna()
    if df.empty:
        print("ℹ️  No data after dropping NA — skipping stratified smoother plot.")
        return None

    df[cluster_key] = df[cluster_key].astype(str)
    counts = df[cluster_key].value_counts()
    clusters = [c for c in counts.index.tolist() if counts[c] >= min_cells_per_cluster]
    if not clusters:
        print(f"ℹ️  No clusters with >= {min_cells_per_cluster} cells — skipping stratified smoother plot.")
        return None

    focus_present = [c for c in focus_clusters if c in clusters]
    show_clusters = focus_present if focus_present else clusters[:6]

    x = df[pseudo_key].values.astype(float)
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        print("ℹ️  Pseudotime range is degenerate — skipping stratified smoother plot.")
        return None

    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cats = pd.Categorical(pd.cut(centers, bins=edges, include_lowest=True)).categories

    with plt.rc_context({
        "font.size": base_font,
        "axes.labelsize": base_font + 2,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font - 2,
    }):
        fig = plt.figure(figsize=(7.2, 4.8), dpi=200)
        ax = plt.gca()

        for c in show_clusters:
            sub = df[df[cluster_key] == c].copy()
            if sub.shape[0] < min_cells_per_cluster:
                continue

            sub["bin"] = pd.cut(sub[pseudo_key].astype(float), bins=edges, include_lowest=True)
            grp = sub.groupby("bin", observed=True)[score_key]
            med = grp.median()
            q1 = grp.quantile(0.25)
            q3 = grp.quantile(0.75)

            y_med = np.array([med.get(cat, np.nan) for cat in cats], dtype=float)
            y_q1  = np.array([q1.get(cat, np.nan) for cat in cats], dtype=float)
            y_q3  = np.array([q3.get(cat, np.nan) for cat in cats], dtype=float)

            ok = np.isfinite(y_med)
            if ok.sum() < 5:
                continue

            ax.plot(centers[ok], y_med[ok], linewidth=2, label=str(c))
            ax.fill_between(centers[ok], y_q1[ok], y_q3[ok], alpha=0.18)

        ax.set_xlabel(pseudo_key)
        ylabel = "ALDO_activity" if score_key == "ALDO_activity_noMach" else score_key
        ax.set_ylabel(ylabel)
        # no title

        # Legend (no n=..., and not tiny)
        ax.legend(frameon=False, fontsize=base_font - 2, loc="upper right")

        # --- Optional: annotate mid/late Δ(ALDO) stats on the figure (POOLED ONLY) ---
        if annotate_midlate_stats and (prof_cluster in df[cluster_key].unique()):
            try:
                from scipy.stats import mannwhitneyu

                lo, hi = midlate_window
                dfw = df[(df[pseudo_key].astype(float) >= lo) & (df[pseudo_key].astype(float) <= hi)].copy()

                a = dfw[dfw[cluster_key] == prof_cluster][score_key].astype(float).values
                pooled = dfw[dfw[cluster_key].isin(list(ref_clusters))][score_key].astype(float).values

                if len(a) >= 5 and len(pooled) >= 5:
                    delta_pool = np.nanmedian(a) - np.nanmedian(pooled)
                    p_pool = mannwhitneyu(a, pooled, alternative="two-sided").pvalue
                    text = (
                        f"Mid/late [{lo:.2f},{hi:.2f}]\n"
                        f"Δ({prof_cluster}−({'+'.join(ref_clusters)}))={delta_pool:+.2f}, p={p_pool:.2g}"
                    )
                else:
                    text = (
                        f"Mid/late [{lo:.2f},{hi:.2f}]\n"
                        f"Δ({prof_cluster}−({'+'.join(ref_clusters)}))=n/a"
                    )

                if stats_location == "upper_left":
                    xy = (0.02, 0.98); va = "top"
                else:
                    xy = (0.02, 0.02); va = "bottom"

                ax.text(
                    xy[0], xy[1], text,
                    transform=ax.transAxes,
                    ha="left", va=va,
                    fontsize=base_font - 4,  # smaller than axes labels
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
                )
            except Exception as e:
                print(f"⚠️  Failed to annotate mid/late stats: {e}")

        plt.tight_layout()

        fn = os.path.join(outdir, f"{score_key}_vs_{pseudo_key}_smoothers_by_{cluster_key}.png")
        fig.savefig(fn, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fn

# =========================
# Final calls (after in-silico run) — export UMAP CSV, run segmented stats, and save .h5ad
# =========================
try:
    # 1) Segmented stats (only if pseudotime exists)
    if 'dpt_pseudotime' in adata.obs.columns:
        keys = [k for k in ["Aldo_machinery","ALDO_signed","ALDO_activity_noMach"] if k in adata.obs.columns]
        seg = compute_and_plot_segmented(adata, _pert_outdir, "dpt_pseudotime", keys, n_segments=3, labels=["early","mid","late"])
        print("✅ Segmented-trend CSV →", seg["csv"])
        for f in seg["figs"]:
            print("✅ Segmented figure →", f)
    else:
        print("ℹ️  No 'dpt_pseudotime' found; skipping segmented stats.")
    # 2) Export UMAP CSV (now that the function is defined)
    ct_col = 'predicted_cell_type' if 'predicted_cell_type' in adata.obs.columns else f"{leiden_key}_named"
        # 1b) ProLIF placement on pseudotime axis (density + ALDO vs pseudotime stratified by cluster)
    if 'dpt_pseudotime' in adata.obs.columns and ct_col in adata.obs.columns:
        # choose ALDO score (prefer de-overlapped activity if present)
        _aldo_key = 'ALDO_activity_noMach' if 'ALDO_activity_noMach' in adata.obs.columns else (
            'ALDO_signed' if 'ALDO_signed' in adata.obs.columns else None
        )

        try:
            fn = plot_pseudotime_density_by_cluster(
                adata,
                _pert_outdir,
                pseudo_key='dpt_pseudotime',
                cluster_key=ct_col,
                bw_adjust=1.0,
                focus_clusters=('DCT1', 'DCT2', 'ProLIF'),
            )
            if fn:
                print('✅ Pseudotime density plot →', fn)
        except Exception as e:
            print(f'⚠️ Pseudotime density plot failed: {e}')
            # Make BOTH ALDO_activity_noMach and ALDO_signed smoothers (if present)

    for _aldo_key in ["ALDO_activity_noMach", "ALDO_signed"]:
        if _aldo_key in adata.obs.columns:
            try:
                fn2 = plot_score_vs_pseudotime_stratified(
                    adata,
                    _pert_outdir,
                    pseudo_key="dpt_pseudotime",
                    score_key=_aldo_key,
                    cluster_key=ct_col,
                    focus_clusters=("DCT1", "DCT2", "ProLIF"),
                    n_bins=30,
                    min_cells_per_cluster=20,
                    annotate_midlate_stats=True,
                    midlate_window=(0.65, 0.85),
                    stats_location="upper_left",
                )
                if fn2:
                    print(f"✅ {_aldo_key} stratified smoother plot → {fn2}")
            except Exception as e:
                print(f"⚠️ {_aldo_key} stratified smoother plot failed: {e}")
        else:
            print(f"ℹ️  '{_aldo_key}' not found in adata.obs — skipping.")

    export_umap_csv_dct(celltype_col=ct_col)
    # 3) Save .h5ad (full only; set to True if you want the lite too)
    try:
        export_h5ad_dct(save_lite=False)
    except Exception as e:
        print(f"⚠️ Failed to export .h5ad: {e}")
except Exception as e:
    print(f"⚠️ Finalization failed: {e}")
