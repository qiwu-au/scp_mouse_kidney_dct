#!/usr/bin/env python
"""dct_reviewer step 2: reviewer_core A.2 Task 4 cell-cycle correction per DCT condition (combined search, dct_labels labels).

Calls the ctrl_reviewer_core step-2 functions
(``ctrl_reviewer_core_cell_cycle``: correlated_proteins, removed_branch, regressed_branch, confounding;
unchanged) per condition through ``dct_reviewer_lib.bound``; ``run_dpt`` is the copy in
``dct_reviewer_lib`` (it hard-codes ``ctrl_cell_label`` in ctrl_reviewer_core).

Inputs (read-only): results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_task3_scored.h5ad (step 1); dct_labels canonical
labels; results/tables/dct_labels/dct_labels_submitted_dpt_roots.csv (root check).
Outputs:
  results/tables/dct_reviewer/dct_reviewer_task4_*.csv (both conditions)
  results/figures/dct_reviewer/dct_reviewer_task4_*.pdf/.png/.tif (Ctrl | ZeroK)
  results/h5ad/dct_reviewer_DCT_{ctrl,zerok}_<branch>.h5ad for the six corrected branches (no intensity layers)

Frozen ctrl_reviewer_core settings, within each condition: removal and regression with the 18-protein panel, the
Tirosh S/G2M lists and the Mki67/Top2a-correlated set (Spearman rho >= 0.3, BH q < 0.05 on the scaled
.X); rescale, arpack PCA, neighbours (10, 20 PCs), Leiden; Leiden 0.3, 0.5, 0.8, 1.0, 3.0 on every graph
(uncorrected = stored dct_labels graph); ARI/NMI against both label columns; kNN purity (k = 10, 20 PCs)
against 1,000 permutations; ProLIF versus protein IDs and plate (columns with ProLIF only); DPT from
the first per-cell DCT1 cell and 20 random per-cell DCT1 roots.

Command (project root):  python scripts/dct_reviewer_cell_cycle.py
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

import numpy as np
import pandas as pd
import scanpy as sc
from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import reviewer_core_lib as t2
import ctrl_reviewer_core_cell_cycle as m19
import dct_reviewer_lib as lib

ensure_scanpy_compat()
lib.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

SCRIPT = "dct_reviewer_cell_cycle.py"
OBJECT_INFO = {"uncorrected": ("none", "stored dct_labels graph"), **t2.CC_BRANCHES}
OBJECTS = list(OBJECT_INFO)
STORED_GENERIC = {"stored_primary_cluster": "PRIMARY_CLUSTER", "stored_leiden3_cluster": "LEIDEN3_CLUSTER"}
STORED_TICK = {"stored_primary_cluster": "stored 1.0 (dct_labels)", "stored_leiden3_cluster": "stored 3.0 (dct_labels)"}
SCORE_OF = {"tirosh_removed": "G2M_score", "tirosh_regressed": "G2M_score", "corr_removed": "cc_corr_score",
            "corr_regressed": "cc_corr_score"}
BLUES = LinearSegmentedColormap.from_list("b", t2.BLUE_RAMP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_reviewer step 2: Task 4 cell-cycle correction per DCT condition.")
    parser.add_argument("--resolutions", type=float, nargs="+", default=list(t2.RESOLUTIONS))
    parser.add_argument("--corr-min-rho", type=float, default=0.30)
    parser.add_argument("--corr-max-fdr", type=float, default=0.05)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-random-roots", type=int, default=20)
    cc.add_common_args(parser)
    return parser.parse_args()


def one_condition(run: cc.Run, cond: str, args, roots27: pd.DataFrame) -> dict:
    s = lib.SETTINGS[cond]
    adata = lib.read_h5ad(run, f"DCT_{s.KEY}_task3_scored")
    _, labels = lib.load(run, cond)
    if list(adata.obs_names) != list(labels.index):
        raise ValueError(f"{cond}: cell order of the A.1 object differs from the canonical labels.")
    stored_dpt = adata.obs["dpt_pseudotime"].to_numpy(float).copy()
    n_detected = labels["n_detected_proteins"].to_numpy()
    out: dict = {"notes": []}

    with lib.bound(s, m19):
        present_orig = [g for g in t2.CELL_CYCLE_MARKERS if g in adata.var_names]
        missing_orig = [g for g in t2.CELL_CYCLE_MARKERS if g not in adata.var_names]
        objects: dict[str, object] = {"uncorrected": adata}
        r3_keys = {}
        objects["cc_removed"] = m19.removed_branch(adata, present_orig, "leiden_cc_removed_res_3.00")
        objects["cc_regressed"] = m19.regressed_branch(adata, ["cell_cycle_score"], "leiden_cc_regressed_res_3.00")
        r3_keys.update(cc_removed="leiden_cc_removed_res_3.00", cc_regressed="leiden_cc_regressed_res_3.00")

        s_rows, s_present = t2.match_genes(t2.TIROSH_S, adata.var_names, "S")
        g_rows, g_present = t2.match_genes(t2.TIROSH_G2M, adata.var_names, "G2M")
        o_rows, _ = t2.match_genes(t2.CELL_CYCLE_MARKERS, adata.var_names, "original_18_panel")
        out["gene_coverage"] = pd.DataFrame(s_rows + g_rows + o_rows)
        order = list(adata.var_names)
        expanded = sorted(set(present_orig) | set(s_present) | set(g_present), key=order.index)
        if not (s_present and g_present):
            raise RuntimeError(f"{cond}: Tirosh S or G2M set has no detected genes.")
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g_present, random_state=t2.SEED, use_raw=False)
        objects["tirosh_removed"] = m19.removed_branch(adata, expanded, "leiden_tirosh_removed_res_3.00")
        objects["tirosh_regressed"] = m19.regressed_branch(adata, ["S_score", "G2M_score"],
                                                           "leiden_tirosh_regressed_res_3.00")
        r3_keys.update(tirosh_removed="leiden_tirosh_removed_res_3.00", tirosh_regressed="leiden_tirosh_regressed_res_3.00")

        corr_table, corr_selected = m19.correlated_proteins(adata, n_detected, args.corr_min_rho, args.corr_max_fdr)
        out["corr"] = corr_table.sort_values("selected", ascending=False, kind="stable")
        out["notes"].append(f"{cond}: circular set {len(corr_selected)} proteins incl. anchors (rho >= "
                            f"{args.corr_min_rho}, BH q < {args.corr_max_fdr}, {adata.n_obs} cells).")
        t2.score(adata, corr_selected, "cc_corr_score")
        objects["corr_removed"] = m19.removed_branch(adata, corr_selected, "leiden_corr_removed_res_3.00")
        objects["corr_regressed"] = m19.regressed_branch(adata, ["cc_corr_score"], "leiden_corr_regressed_res_3.00")
        r3_keys.update(corr_removed="leiden_corr_removed_res_3.00", corr_regressed="leiden_corr_regressed_res_3.00")

    out["sets"] = pd.DataFrame([
        {"object": name, "gene_set": OBJECT_INFO[name][0], "correction": OBJECT_INFO[name][1], "n_proteins_after": obj.n_vars,
         "proteins_removed": ";".join(sorted(set(adata.var_names) - set(obj.var_names), key=order.index)),
         "regressed_keys": {"cc_regressed": "cell_cycle_score", "tirosh_regressed": "S_score;G2M_score",
                            "corr_regressed": "cc_corr_score"}.get(name, "")}
        for name, obj in objects.items()])
    score_cols = [c for c in ("cell_cycle_score", "S_score", "G2M_score", "cc_corr_score") if c in adata.obs]
    scores = adata.obs[score_cols].set_axis(labels.index, axis=0)

    resolutions = sorted(set(args.resolutions) | {t2.ORIGINAL_RESOLUTION})
    clusters = pd.DataFrame(index=labels.index)
    for name, obj in objects.items():
        if list(obj.obs_names) != list(adata.obs_names):
            raise ValueError(f"{cond}: cell order differs in {name}")
        for res in resolutions:
            key = f"leiden_r{res:.2f}"
            if name in r3_keys and np.isclose(res, t2.ORIGINAL_RESOLUTION):
                obj.obs[key] = obj.obs[r3_keys[name]].astype(str).astype("category")
            else:
                t2.leiden(obj, res, key)
            clusters[f"{name}__r{res:.2f}"] = obj.obs[key].astype(str).to_numpy()
    for key, attr in STORED_GENERIC.items():
        clusters[f"uncorrected__{key}"] = labels[getattr(s, attr)].astype(str).to_numpy()
    checks = []
    for col, res in s.STORED_CLUSTERS.values():
        same = int((clusters[f"uncorrected__r{res:.2f}"].to_numpy() == labels[col].astype(str).to_numpy()).sum())
        checks.append({"condition": cond, "check": f"Leiden {res} (seed 42) on the stored dct_labels graph equals {col}",
                       "value": f"{same}/{len(labels)}", "pass": same == len(labels)})
    out["clusters"] = clusters.reset_index()

    out["ari"] = t2.ari_nmi_rows(clusters, labels, s.LABEL_COLUMNS)
    conc = t2.concentration_rows(clusters, labels, s.LABEL_COLUMNS)
    conc["applicable"] = conc["n_prolif"] > 0
    out["conc"] = conc
    comp_rows = []
    for column in clusters.columns:
        name, res = column.split("__")
        for col in s.LABEL_COLUMNS:
            table = pd.crosstab(clusters[column], labels[col])
            for cluster, counts in table.iterrows():
                for label, n in counts.items():
                    comp_rows.append({"object": name, "resolution": res.removeprefix("r"), "cluster": cluster,
                                      "label_column": col, "label": label, "n_cells": int(n),
                                      "fraction_of_cluster": n / counts.sum()})
    out["composition"] = pd.DataFrame(comp_rows)

    task3_cols = [c for c in ("ambiguous_tal_cnt_interstitial", "DCT_core_score", "DCT1_associated_score",
                              "DCT2_associated_score", "TAL_score") if c in adata.obs]
    context = adata.obs[task3_cols].set_axis(labels.index, axis=0).join(scores)
    summary_rows = []
    for name in objects:
        column = f"{name}__r{t2.ORIGINAL_RESOLUTION:.2f}"
        for cluster, idx in labels.groupby(clusters[column]).groups.items():
            row = {"object": name, "resolution": f"{t2.ORIGINAL_RESOLUTION:.2f}", "cluster": cluster, "n_cells": len(idx)}
            for role, col in s.COLUMN_OF.items():
                row[f"n_prolif_{role}"] = int((labels.loc[idx, col] == "ProLIF").sum())
                row[f"fraction_prolif_{role}"] = row[f"n_prolif_{role}"] / len(idx)
            for c in context.columns:
                vals = context.loc[idx, c]
                row[f"fraction_{c}" if vals.dtype == bool else f"{c}_median"] = vals.mean() if vals.dtype == bool else vals.median()
            summary_rows.append(row)
    out["cluster_summary"] = pd.DataFrame(summary_rows)

    out["purity"] = t2.knn_purity({name: obj.obsm["X_pca"][:, :t2.N_PCS] for name, obj in objects.items()},
                                  labels, s.LABEL_COLUMNS, args.knn_k, args.n_permutations)

    # ProLIF versus protein IDs and plate: only columns with ProLIF cells (none under the primary columns).
    with_prolif = lib.columns_with(labels, s, "ProLIF")
    with lib.bound(s.subset(with_prolif), m19):
        conf, logit, corr = m19.confounding(labels, scores)
    missing = [c for c in s.LABEL_COLUMNS if c not in with_prolif]
    na = lib.na_rows(cond, missing, "ProLIF vs protein IDs and plate", "no ProLIF cell under this column")
    out["conf"] = pd.concat([conf.assign(status="computed"), na.drop(columns=["condition", "label_role"])],
                            ignore_index=True)
    out["logit"] = pd.concat([logit.assign(status="computed"), na.drop(columns=["condition", "label_role"])],
                             ignore_index=True)
    out["corr_ids"] = corr

    dpt_detail, dpt_summary, dpt_cells = lib.run_dpt(s, objects, labels, args.n_random_roots)
    dpt_diff = float(np.nanmax(np.abs(dpt_cells["uncorrected__dpt_original_root"].to_numpy() - stored_dpt)))
    checks.append({"condition": cond, "check": "uncorrected DPT from the original root reproduces the stored dct_labels dpt_pseudotime",
                   "value": f"max abs diff {dpt_diff:.3g}", "pass": dpt_diff < 1e-5})
    r27 = roots27[roots27["condition"] == cond]
    ours = dpt_detail.loc[(dpt_detail["object"] == "uncorrected") & (dpt_detail["label_column"] == s.PRIMARY),
                          "root_index"].astype(int).tolist()
    same_roots = ours == r27["root_index"].astype(int).tolist()
    checks.append({"condition": cond, "check": "original and 20 random roots equal the dct_labels roots "
                   "(dct_labels_submitted_dpt_roots.csv)", "value": f"{same_roots}", "pass": same_roots})
    out["checks"] = checks
    out["dpt_detail"] = lib.mark_dpt_applicability(dpt_detail)
    out["dpt_summary"] = lib.mark_dpt_applicability(dpt_summary)
    out["dpt_cells"] = dpt_cells.reset_index()
    out["scores_per_cell"] = lib.to_generic(labels[["cell_key", "plate", *s.LABEL_COLUMNS, "n_detected_proteins"]]
                                            .join(scores), cond).reset_index()
    out.update(objects=objects, labels=labels, scores=scores, score_cols=score_cols,
               params={"cell_cycle_original_present": present_orig, "cell_cycle_original_missing": missing_orig,
                       "tirosh_S_present": s_present, "tirosh_G2M_present": g_present,
                       "expanded_removed_set": expanded, "correlated_set": corr_selected,
                       "dpt_max_abs_diff_vs_stored": dpt_diff})
    return out


# ---------------------------------------------------------------------------
# Figures (Ctrl | ZeroK)
# ---------------------------------------------------------------------------
def legend_labels(fig, order, y=0.005) -> None:
    handles = [Line2D([], [], marker="o", linestyle="", color=t2.LABEL_PALETTE.get(l, "0.5"), markersize=4, label=l)
               for l in order]
    fig.legend(handles=handles, loc="lower center", ncol=len(order), frameon=False, bbox_to_anchor=(0.5, y),
               handletextpad=0.2, columnspacing=1.2)


def plot_umaps(res: dict, name: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 92)
    gs = fig.add_gridspec(2, 5, left=0.04, right=0.955, top=0.9, bottom=0.1, hspace=0.35, wspace=0.45)
    for r, cond in enumerate(lib.CONDITIONS):
        s, d = lib.SETTINGS[cond], res[cond]
        obj, labels = d["objects"][name], d["labels"]
        umap = obj.obsm["X_umap"]
        score_col = SCORE_OF.get(name, "cell_cycle_score")
        values = {"cell_cycle_score": d["scores"]["cell_cycle_score"], "G2M_score": d["scores"]["G2M_score"],
                  "cc_corr_score": d["scores"]["cc_corr_score"]}[score_col].to_numpy(float)
        panels = [("primary", "cat"), ("second", "cat"), ("leiden_r3.00", "cluster"), (score_col, "num"),
                  ("dpt_pseudotime", "num")]
        for c, (col, kind) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            if kind == "cat":
                lab = labels[s.COLUMN_OF[col]].astype(str).to_numpy()
                for cat in cc.ordered_labels(lab):
                    m = lab == cat
                    ax.scatter(umap[m, 0], umap[m, 1], s=1.2, color=t2.LABEL_PALETTE.get(cat, "0.5"), linewidths=0,
                               rasterized=True)
                title = f"{col} label"
            elif kind == "cluster":
                codes = pd.Categorical(obj.obs[col].astype(str)).codes
                ax.scatter(umap[:, 0], umap[:, 1], c=codes, cmap="tab20", s=1.2, linewidths=0, rasterized=True)
                title = f"Leiden 3.0 ({obj.obs[col].nunique()})"
            else:
                v = values if col != "dpt_pseudotime" else obj.obs[col].to_numpy(float)
                sca = ax.scatter(umap[:, 0], umap[:, 1], c=v, cmap="viridis", s=1.2, linewidths=0, rasterized=True)
                cb = fig.colorbar(sca, ax=ax, fraction=0.05, pad=0.02)
                cb.ax.yaxis.set_major_locator(MaxNLocator(3))
                cb.ax.tick_params(labelsize=7, length=2)
                title = "DPT (original root)" if col == "dpt_pseudotime" else col.replace("_score", " score")
            t2.clean_axes(ax)
            ax.set_title(title, fontsize=7)
            if c == 0:
                ax.set_ylabel(cond, fontsize=7)
    info = OBJECT_INFO[name]
    fig.text(0.005, 0.995, f"{name}: {info[0]}, {info[1]} (UMAP of each condition's own object)", ha="left",
             va="top", fontsize=7)
    legend_labels(fig, list(lib.LABEL_ORDER))
    figs.save(fig, f"task4_umap_{name}", f"UMAP of the {name} object per condition")


def plot_heatmaps(res: dict, metric: str, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 150)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.04], left=0.13, right=0.94, top=0.95, bottom=0.17,
                          hspace=0.75, wspace=0.08)
    vmax = max(0.5, max(float(res[c]["ari"][metric].max()) for c in lib.CONDITIONS))
    im = None
    for r, role in enumerate(lib.ROLES):
        for c, cond in enumerate(lib.CONDITIONS):
            s, ari = lib.SETTINGS[cond], res[cond]["ari"]
            sub = ari[ari["label_column"] == s.COLUMN_OF[role]].pivot(index="object", columns="resolution", values=metric)
            sub = sub.reindex(index=[o for o in OBJECTS if o in sub.index])
            cols = [x for x in sub.columns if not x.startswith("stored")] + [x for x in sub.columns if x.startswith("stored")]
            sub = sub[cols]
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(sub.to_numpy(float), cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    v = sub.iat[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if v > 0.6 * vmax else "#0b0b0b")
            ax.set_xticks(range(sub.shape[1]), [STORED_TICK.get(x, x) for x in sub.columns], rotation=90)
            ax.set_yticks(range(sub.shape[0]), sub.index if c == 0 else [""] * sub.shape[0])
            ax.set_title(f"{cond}: {metric.upper()} vs {role} label", fontsize=7)
            if r == 1:
                ax.set_xlabel("Leiden resolution")
    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7)
    figs.save(fig, f"task4_{metric}_clusters_vs_labels", f"{metric.upper()} of clusters versus dct_labels labels per condition")


def plot_concentration(res: dict, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 110)
    gs = fig.add_gridspec(2, 2, left=0.13, right=0.98, top=0.93, bottom=0.11, hspace=0.45, wspace=0.08)
    for r, role in enumerate(lib.ROLES):
        for c, cond in enumerate(lib.CONDITIONS):
            s, conc = lib.SETTINGS[cond], res[cond]["conc"]
            ax = fig.add_subplot(gs[r, c])
            col = s.COLUMN_OF[role]
            sub = conc[np.isclose(conc["resolution_numeric"], t2.ORIGINAL_RESOLUTION) & conc["object"].isin(OBJECTS)
                       & (conc["label_column"] == col)].set_index("object").reindex(OBJECTS)
            if int(sub["n_prolif"].iloc[0]) == 0:
                lib.na_axis(ax, f"{cond}, {role} label: {lib.NOT_APPLICABLE}\n(no ProLIF cell under {col})")
                continue
            y = np.arange(len(OBJECTS))
            ax.barh(y - 0.18, sub["top1_percent"], height=0.34, color=t2.SERIES[0], label="largest cluster")
            ax.barh(y + 0.18, sub["top3_percent"], height=0.34, color=t2.SERIES[1], label="top 3 clusters")
            ax.set_yticks(y, OBJECTS if c == 0 else [""] * len(OBJECTS))
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.set_title(f"{cond}, {role} label (ProLIF n = {int(sub['n_prolif'].iloc[0])})", fontsize=7)
            ax.set_xlabel("% of ProLIF cells in largest / top-3 Leiden 3.0 clusters")
            ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.5)
            ax.set_axisbelow(True)
            if c == 0:
                ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
    figs.save(fig, "task4_prolif_concentration_r3.00", "ProLIF concentration in Leiden 3.0 clusters per condition")


def plot_purity(table_by_cond: dict, name: str, objects: list[str], title: str, figs: lib.Figures,
                height_mm: float = 140, bottom: float = 0.2) -> None:
    fig = lib.new_figure(lib.FULL_MM, height_mm)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.99, top=0.93, bottom=bottom, hspace=0.75, wspace=0.06)
    first = None
    for r, role in enumerate(lib.ROLES):
        for c, cond in enumerate(lib.CONDITIONS):
            s, table = lib.SETTINGS[cond], table_by_cond[cond]
            ax = fig.add_subplot(gs[r, c], sharey=first)
            first = first or ax
            sub = table[(table["label_column"] == s.COLUMN_OF[role]) & table["label"].isin(t2.CORE_LABELS)]
            for offset, label in zip((-0.22, 0, 0.22), t2.CORE_LABELS):
                part = sub[sub["label"] == label].set_index("object").reindex(objects)
                if part["observed_same_label_fraction"].notna().sum() == 0:
                    continue
                x = np.arange(len(objects)) + offset
                ax.vlines(x, part["null_q025"], part["null_q975"], color=t2.INK["background_cells"], linewidth=3.0)
                ax.scatter(x, part["observed_same_label_fraction"], color=t2.LABEL_PALETTE[label], s=12, zorder=3,
                           edgecolors="white", linewidths=0.4)
            ax.set_xticks(range(len(objects)), objects if r == 1 else [""] * len(objects), rotation=90)
            ax.set_title(f"{cond}, {role} label", fontsize=7)
            if c == 0:
                ax.set_ylabel("same-label fraction (k = 10)")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylim(-0.02, 1.02)
            lib.grid_y(ax)
    fig.text(0.005, 0.995, title + " (dot = observed; grey bar = permutation 95%; absent labels not drawn)",
             ha="left", va="top", fontsize=7)
    legend_labels(fig, list(t2.CORE_LABELS))
    figs.save(fig, name, title)


def plot_dpt(res: dict, figs: lib.Figures) -> None:
    fig = lib.new_figure(lib.FULL_MM, 160)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.99, top=0.92, bottom=0.24, hspace=1.15, wspace=0.06)
    rng = np.random.default_rng(0)  # jitter only
    order = ["DCT1", "ProLIF", "DCT2"]
    first = None
    for r, role in enumerate(lib.ROLES):
        for c, cond in enumerate(lib.CONDITIONS):
            s, detail = lib.SETTINGS[cond], res[cond]["dpt_detail"]
            ax = fig.add_subplot(gs[r, c], sharey=first)
            first = first or ax
            sub = detail[detail["label_column"] == s.COLUMN_OF[role]]
            ticks = []
            for k, name in enumerate(OBJECTS):
                part = sub[sub["object"] == name]
                rnd, orig = part[part["root"] != "original"], part[part["root"] == "original"].iloc[0]
                for off, label in zip((-0.25, 0, 0.25), order):
                    vals = rnd[f"median_{label}"].to_numpy(float)
                    if not np.isfinite(vals).any():
                        continue
                    ax.scatter(k + off + rng.uniform(-0.06, 0.06, len(vals)), vals, s=3, color=t2.LABEL_PALETTE[label],
                               alpha=0.7, linewidths=0)
                    ax.scatter([k + off], [orig[f"median_{label}"]], marker="D", color="#0b0b0b", s=8, zorder=4)
                between = (f"{int(rnd['prolif_between_dct1_dct2'].sum())}/{len(rnd)}"
                           if bool(orig["prolif_between_applicable"]) else "n/a")
                ticks.append(f"{name} ({between})")
            ax.set_xticks(range(len(OBJECTS)), ticks, rotation=90)
            ax.set_title(f"{cond}, {role} label", fontsize=7)
            if c == 0:
                ax.set_ylabel("median DPT per label")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            lib.grid_y(ax)
    fig.text(0.005, 0.995, "DPT medians per label (diamond = original root; dots = 20 random DCT1 roots); in brackets: "
             "random roots with the ProLIF median between DCT1 and DCT2 (n/a = a label is absent)", ha="left", va="top",
             fontsize=7, wrap=True)
    legend_labels(fig, order)
    figs.save(fig, "task4_dpt_medians", "DPT medians per label and root, per condition")


def main() -> None:
    args = parse_args()
    run = cc.Run(lib.TASK, "cell_cycle", args)
    lib.ported_from(run, "ctrl_reviewer_core_cell_cycle.py")
    roots27 = pd.read_csv(run.use_input(lib.DCT_LABELS_DPT_ROOTS, "dct_labels DPT roots (check)"))
    res = {c: one_condition(run, c, args, roots27) for c in lib.CONDITIONS}
    for c in lib.CONDITIONS:
        for n in res[c]["notes"]:
            run.note(n)
    checks = lib.check_rows_frame([row for c in lib.CONDITIONS for row in res[c]["checks"]])
    lib.stop_if_failed(run, checks, "task4_checks")
    for key, name in (("gene_coverage", "task4_cell_cycle_gene_coverage"), ("corr", "task4_mki67_top2a_correlated_proteins"),
                      ("sets", "task4_correction_definitions"), ("clusters", "task4_clusters_per_cell"),
                      ("ari", "task4_ari_nmi_clusters_vs_labels"), ("conc", "task4_prolif_concentration"),
                      ("composition", "task4_cluster_composition_long"),
                      ("cluster_summary", f"task4_cluster_summary_r{t2.ORIGINAL_RESOLUTION:.2f}"),
                      ("purity", "task4_knn_label_purity"), ("conf", "task4_confounding_prolif_vs_ids_and_plate"),
                      ("logit", "task4_confounding_logistic_prolif"), ("corr_ids", "task4_confounding_scores_vs_ids"),
                      ("dpt_detail", "task4_dpt_medians_per_root"), ("dpt_summary", "task4_dpt_summary"),
                      ("dpt_cells", "task4_dpt_per_cell_original_root"),
                      ("scores_per_cell", "task4_cell_cycle_scores_per_cell")):
        run.table(lib.stack({c: res[c][key] for c in lib.CONDITIONS}), name)
    run.note("Stored clusterings in the cluster tables: stored_primary_cluster = cs_ctrl_deg_cluster / "
             "cs_zerok_pipeline_cluster (Leiden 1.0); stored_leiden3_cluster = cs_<cond>_leiden3_cluster (Leiden 3.0).")
    run.note("ProLIF concentration, confounding and 'ProLIF between' rows under a column without ProLIF are marked "
             "not applicable (applicable / status / prolif_between_applicable columns).")

    figs = lib.Figures(run, "cell_cycle")
    for name in OBJECTS:
        plot_umaps(res, name, figs)
    plot_heatmaps(res, "ari", figs)
    plot_heatmaps(res, "nmi", figs)
    plot_concentration(res, figs)
    plot_purity({c: res[c]["purity"] for c in lib.CONDITIONS}, "task4_knn_label_purity", OBJECTS,
                "kNN label purity (k = 10, 20 PCs) versus 1,000 label permutations", figs)
    plot_dpt(res, figs)
    figs.finish()

    for cond in lib.CONDITIONS:
        s, d = lib.SETTINGS[cond], res[cond]
        for name, obj in d["objects"].items():
            if name == "uncorrected":
                continue
            for col in ("plate", "cell_key", *s.LABEL_COLUMNS):
                obj.obs[col] = pd.Categorical(d["labels"][col].to_numpy())
            for col in d["score_cols"]:
                obj.obs[col] = d["scores"][col].to_numpy()
            lib.write_h5ad_light(run, obj, f"DCT_{s.KEY}_{name}")
    run.finish({c: res[c]["params"] for c in lib.CONDITIONS} | {"figures_pass": figs.all_pass()})
    if not figs.all_pass():
        raise SystemExit("A figure check failed; see dct_reviewer_figure_checks_cell_cycle.csv.")


if __name__ == "__main__":
    main()
