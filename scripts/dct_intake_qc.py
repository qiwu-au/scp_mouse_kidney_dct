#!/usr/bin/env python
"""dct_intake step 4: technical QC of the combined search by condition and plate (descriptive; no biology).



intake_core step 5 and zerok_core step 3 on the combined search: technical and composition-free
summaries only; no protein-level or label comparison between conditions, and no
protein is named in the search-effect output.

Inputs (read-only): results/h5ad/dct_intake_DCT_{ctrl,zerok,joint}.h5ad;
results/tables/dct_intake/dct_intake_dct_cells.csv and dct_intake_mitocarta_protein_flags.csv.
Outputs:
  results/tables/dct_intake/
    dct_intake_qc_per_cell.csv               retained cells: IDs, missing fraction, total and ln intensity
                                             (own retained proteins and joint proteins)
    dct_intake_qc_by_condition_plate.csv     min / quartiles / max of each per-cell metric by condition x plate
    dct_intake_qc_per_protein.csv            detection and missing fraction per protein (overall and per plate)
    dct_intake_qc_intensity_values.csv       detected-value distribution per condition (incl. values equal to 1)
    dct_intake_technical_per_cell.csv        joint proteins: IDs, intensity, MitoCarta fractions, stress score
    dct_intake_technical_by_condition_plate.csv  median and IQR of those metrics by condition x plate
    dct_intake_search_effect_bins.csv        per joint protein, median ln over detected Ctrl cells minus the same
                                             over detected ZeroK cells, binned by detection (no protein named)
    dct_intake_search_effect_detection.csv   median detection fraction per condition
    dct_intake_stress_panel_coverage.csv     dissociation-stress panel genes among the joint proteins
    dct_intake_embedding_depth_correlation.csv  |Spearman rho| of PC1/PC2 with log n_detected, per variant
    dct_intake_embedding_mixing.csv          kNN condition mixing and plate mixing within condition, per variant
    dct_intake_technical_checks.csv
  results/figures/dct_intake/ (PDF + 600 dpi PNG)
    dct_intake_proteins_per_cell, dct_intake_total_intensity, dct_intake_missingness,
    dct_intake_ln_intensity_per_cell_all, dct_intake_ln_intensity_per_cell_joint, dct_intake_detection_rank,
    dct_intake_plate_maps, dct_intake_qc_by_condition_plate, dct_intake_search_effect,
    dct_intake_joint_embeddings, dct_intake_mixing_and_depth

Functions reused unchanged: intake_core_qc (ln_detected_stats, per_cell, intensity_values),
zerok_core_technical (stress_proteins, per_cell_table, qc_summary, search_effect),
zerok_core_lib (normalize_joint, mixing), qc_mito_lib (MitoCarta metrics, figure style).
The figures are the intake_core and zerok_core figures with titles for one combined search.

Command (project root):  python scripts/dct_intake_qc.py
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
import time

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc
import qc_mito_lib as t3
import cluster_de_lib as t5
import intake_core_lib as t17
import intake_core_qc as t17qc
import zerok_core_lib as t21
import zerok_core_technical as t21tech
import dct_intake_lib as lib

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT = "dct_intake_qc.py"
NOTE = lib.ONE_SEARCH_NOTE
GROUP_ORDER = [f"{c} {p}" for c in lib.CONDITIONS for p in lib.PLATES]
GROUP_COLORS = {g: cc.PLATE_PALETTE[g.split()[1]] for g in GROUP_ORDER}
PLATE_LINESTYLE = t17qc.PLATE_LINESTYLE
METRICS = {
    "condition": {"n_detected": "protein IDs per cell", "missing_fraction": "fraction missing per cell",
                  "log10_total_intensity": "log10 total intensity",
                  "median_ln1p_detected": "median ln(1+intensity), detected"},
    "joint": {"n_detected_joint": "protein IDs per cell", "missing_fraction_joint": "fraction missing per cell",
              "log10_total_intensity_joint": "log10 total intensity",
              "median_ln1p_detected_joint": "median ln(1+intensity), detected"},
}


def joint_names(columns) -> dict[str, str]:
    """intake_core/zerok_core helpers name the joint-protein columns '_matched'; dct_intake calls them '_joint'."""
    return {c: c.replace("_matched", "_joint") for c in columns if "_matched" in c}


def by_condition_plate(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond in lib.CONDITIONS:
        for plate in ("all", *lib.PLATES):
            sub = cells.loc[(cells["condition"] == cond) & ((cells["plate"] == plate) if plate != "all" else True)]
            for protein_set, metrics in METRICS.items():
                for metric, display in metrics.items():
                    v = sub[metric].to_numpy(float)
                    q = np.percentile(v, [25, 50, 75])
                    rows.append({"condition": cond, "plate": plate,
                                 "protein_set": ("own retained proteins" if protein_set == "condition"
                                                 else "joint proteins (retained in both conditions)"),
                                 "metric": metric, "metric_display": display, "n_cells": len(v), "min": v.min(),
                                 "q25": q[0], "median": q[1], "q75": q[2], "max": v.max()})
    return pd.DataFrame(rows)


def per_protein(objects: dict) -> pd.DataFrame:
    frames = []
    for cond in lib.CONDITIONS:
        ad_ = objects[cond]
        det = ad_.layers["intensity"] > 0
        ln = np.where(det, ad_.layers["ln1p_intensity"], np.nan)
        frame = pd.DataFrame({
            "condition": cond, "protein": ad_.var_names, "protein_group": ad_.var["protein_group"].to_numpy(),
            "gene_name": ad_.var["gene_name"].to_numpy(), "in_joint": ad_.var["in_joint"].to_numpy(),
            "n_cells_detected": det.sum(axis=0), "detection_fraction": det.mean(axis=0),
        })
        frame["missing_fraction"] = 1 - frame["detection_fraction"]
        plates = ad_.obs["plate"].astype(str).to_numpy()
        for plate in lib.PLATES:
            frame[f"detection_fraction_{plate}"] = det[plates == plate].mean(axis=0)
        with np.errstate(all="ignore"):
            frame["median_ln1p_detected"] = np.nanmedian(ln, axis=0)
        frame["detection_rank"] = frame["detection_fraction"].rank(ascending=False, method="first").astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def mito_flags(run: cc.Run, var: pd.DataFrame) -> pd.DataFrame:
    """dct_intake MitoCarta flags (one combined search), reindexed to the joint proteins."""
    path = lib.table_path("mitocarta_protein_flags")
    frame = pd.read_csv(run.use_input(path, "dct_intake MitoCarta flags"), keep_default_na=False)
    frame = frame[frame["condition"] == "Ctrl"].set_index("protein")
    flags = frame.reindex(var.index.astype(str).to_numpy())
    for col in t3.SETS:
        flags[col] = flags[col].astype(str).str.lower().eq("true")
    flags.index = pd.Index(var.index.astype(str), name="protein")
    return flags


def depth_and_mixing(joint, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """zerok_core step 3 depth correlation and mixing, on the PCs stored in the joint object."""
    depth_rows, mix_rows = [], []
    log_ids = frame["log_n_detected"].to_numpy()
    for name in t21.VARIANTS:
        pca = np.asarray(joint.obsm[f"X_pca__{name}"], dtype=np.float64)
        for pc in (0, 1):
            rho, p = stats.spearmanr(pca[:, pc], log_ids)
            depth_rows.append({"variant": name, "component": f"PC{pc + 1}", "spearman_rho": float(rho),
                               "abs_spearman_rho": float(abs(rho)), "p_value": float(p),
                               "against": "log n_detected (joint proteins)"})
        mix = t21.mixing(pca[:, :t21.N_PCS], frame["condition"], list(lib.CONDITIONS))
        mix.insert(0, "variant", name)
        mix.insert(1, "grouping", "condition")
        mix.insert(2, "cells", "all")
        mix_rows.append(mix)
        for cond in lib.CONDITIONS:
            own = (frame["condition"] == cond).to_numpy()
            pm = t21.mixing(pca[own][:, :t21.N_PCS], frame.loc[own, "plate"], list(lib.PLATES))
            pm.insert(0, "variant", name)
            pm.insert(1, "grouping", "plate")
            pm.insert(2, "cells", cond)
            mix_rows.append(pm)
    return pd.DataFrame(depth_rows), pd.concat(mix_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Figures (intake_core step 5 and zerok_core step 3 layouts)
# ---------------------------------------------------------------------------
def suptitle(fig, text: str) -> None:
    fig.suptitle(f"{text}\n{NOTE}", x=0.01, ha="left", fontsize=7.5)


def fig_proteins_per_cell(run, cells_all: pd.DataFrame, cells: pd.DataFrame, n_joint: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharey="row")
    for j, cond in enumerate(lib.CONDITIONS):
        sub = cells_all.loc[cells_all["condition"] == cond]
        ax = axes[0, j]
        t17.box_strip(ax, sub, "plate", "n_detected_all_rows", list(lib.PLATES), cc.PLATE_PALETTE,
                      highlight=~sub["retained"].to_numpy(bool))
        ax.axhline(cc.MIN_CELL_PROTEINS, color=t17.INK["secondary"], linewidth=0.6, linestyle=(0, (3, 2)))
        ax.text(1.55, cc.MIN_CELL_PROTEINS, f"{cc.MIN_CELL_PROTEINS} IDs", va="bottom", ha="right", fontsize=5.6,
                color=t17.INK["secondary"])
        n_removed = int((~sub["retained"].astype(bool)).sum())
        ax.set_title(f"{cond}: all {len(sub)} cells, all protein rows\n(ringed: {n_removed} removed, < "
                     f"{cc.MIN_CELL_PROTEINS} IDs)")
        sub2 = cells.loc[cells["condition"] == cond]
        t17.box_strip(axes[1, j], sub2, "plate", "n_detected_joint", list(lib.PLATES), cc.PLATE_PALETTE)
        axes[1, j].set_title(f"{cond}: {len(sub2)} retained cells, {n_joint:,} joint proteins")
    axes[0, 0].set_ylabel("protein IDs per cell")
    axes[1, 0].set_ylabel("protein IDs per cell (joint)")
    for ax in axes.flat:
        ax.set_xlabel("sort plate (cells per plate)")
    t17qc.plate_legend(axes[0, 1])
    suptitle(fig, "Protein IDs per cell by condition and sort plate")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    t17.footer(fig, SCRIPT)
    run.figure(fig, "proteins_per_cell", "dct_intake protein IDs per cell by condition and plate")


def fig_total_intensity(run, cells: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharey="row")
    for j, cond in enumerate(lib.CONDITIONS):
        sub = cells.loc[cells["condition"] == cond]
        t17.box_strip(axes[0, j], sub, "plate", "log10_total_intensity", list(lib.PLATES), cc.PLATE_PALETTE)
        axes[0, j].set_title(f"{cond}: {len(sub)} retained cells, own retained proteins")
        t17.box_strip(axes[1, j], sub, "plate", "log10_total_intensity_joint", list(lib.PLATES), cc.PLATE_PALETTE)
        axes[1, j].set_title(f"{cond}: joint proteins only")
    axes[0, 0].set_ylabel("log10 total intensity")
    axes[1, 0].set_ylabel("log10 total intensity (joint)")
    for ax in axes.flat:
        ax.set_xlabel("sort plate (cells per plate)")
    t17qc.plate_legend(axes[0, 1])
    suptitle(fig, "Total intensity per retained cell (sum of PG.Quantity)")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    t17.footer(fig, SCRIPT)
    run.figure(fig, "total_intensity", "dct_intake total intensity per cell by condition and plate")


def fig_missingness(run, cells: pd.DataFrame, proteins: pd.DataFrame, joint) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    t17.box_strip(axes[0, 0], cells, "group", "missing_fraction", GROUP_ORDER, GROUP_COLORS)
    axes[0, 0].set_title("Per cell, own retained proteins")
    axes[0, 0].set_ylabel("fraction of proteins missing (NaN)")
    t17.box_strip(axes[0, 1], cells, "group", "missing_fraction_joint", GROUP_ORDER, GROUP_COLORS)
    axes[0, 1].set_title(f"Per cell, {joint.n_vars:,} joint proteins")
    axes[0, 0].set_xlabel("condition, sort plate (cells)")
    axes[0, 1].set_xlabel("condition, sort plate (cells)")
    t17qc.plate_legend(axes[0, 1])
    bins = np.linspace(0, 1, 41)
    for ax, use_joint in ((axes[1, 0], False), (axes[1, 1], True)):
        for cond in lib.CONDITIONS:
            if use_joint:
                det = (joint.layers["intensity"][(joint.obs["condition"] == cond).to_numpy()] > 0).mean(axis=0)
            else:
                det = proteins.loc[proteins["condition"] == cond, "detection_fraction"].to_numpy()
            ax.hist(det, bins=bins, histtype="step", linewidth=1.2, color=lib.CONDITION_PALETTE[cond],
                    label=f"{cond} ({len(det):,} proteins)")
        t17.ygrid(ax)
        ax.set_xlabel("detection fraction per protein (1 - missing fraction)")
        ax.set_ylabel("proteins")
        ax.legend(loc="upper right", title="condition", title_fontsize=6)
    axes[1, 0].set_title("Per protein, own retained proteins")
    axes[1, 1].set_title("Per protein, joint proteins (each condition's cells)")
    suptitle(fig, "Missingness by condition and sort plate")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    t17.footer(fig, SCRIPT)
    run.figure(fig, "missingness", "dct_intake missingness per cell and per protein")


def fig_ln_intensity(run, objects: dict, joint, cells: pd.DataFrame, use_joint: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), sharey="row", gridspec_kw={"height_ratios": [1.1, 1]})
    bins = np.linspace(0, 17, 69)
    centres = (bins[:-1] + bins[1:]) / 2
    suffix = "_joint" if use_joint else ""
    for j, cond in enumerate(lib.CONDITIONS):
        if use_joint:
            mask = (joint.obs["condition"] == cond).to_numpy()
            ln = joint.layers["ln1p_intensity"][mask]
            plates = joint.obs["plate"].astype(str).to_numpy()[mask]
        else:
            ln = objects[cond].layers["ln1p_intensity"]
            plates = objects[cond].obs["plate"].astype(str).to_numpy()
        ax = axes[0, j]
        for plate in lib.PLATES:
            for r in ln[plates == plate]:
                h, _ = np.histogram(r[r > 0], bins=bins, density=True)
                ax.plot(centres, h, color=cc.PLATE_PALETTE[plate], linewidth=0.3, alpha=0.08, rasterized=True)
        for plate in lib.PLATES:
            v = ln[plates == plate]
            h, _ = np.histogram(v[v > 0], bins=bins, density=True)
            ax.plot(centres, h, color=cc.PLATE_PALETTE[plate], linewidth=2.0, linestyle=PLATE_LINESTYLE[plate],
                    label=f"{plate}, all cells pooled")
        ax.set_xlabel("ln(1 + intensity), detected values")
        ax.set_title(f"{cond}: one thin line per cell ({ln.shape[0]} cells x {ln.shape[1]:,} proteins)")
        t17.ygrid(ax)
        ax.legend(loc="upper right")
        sub = cells.loc[cells["condition"] == cond].sort_values(["plate", "well"])
        x = np.arange(len(sub))
        ax2 = axes[1, j]
        for plate in lib.PLATES:
            m = (sub["plate"] == plate).to_numpy()
            ax2.vlines(x[m], sub[f"q25_ln1p_detected{suffix}"].to_numpy()[m],
                       sub[f"q75_ln1p_detected{suffix}"].to_numpy()[m], color=cc.PLATE_PALETTE[plate],
                       linewidth=0.35, alpha=0.6, rasterized=True)
            ax2.scatter(x[m], sub[f"median_ln1p_detected{suffix}"].to_numpy()[m], s=1.5, color=t17.INK["primary"],
                        linewidths=0, rasterized=True)
        boundary = int((sub["plate"] == "P1").sum())
        ax2.axvline(boundary - 0.5, color=t17.INK["muted"], linewidth=0.6)
        ax2.text(boundary / 2, 1.0, "P1", transform=ax2.get_xaxis_transform(), ha="center", va="bottom", fontsize=6)
        ax2.text(boundary + (len(sub) - boundary) / 2, 1.0, "P2", transform=ax2.get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=6)
        ax2.set_xlabel("retained cells ordered by plate and well")
        ax2.set_title(f"{cond}: median (dot) and IQR (line) per cell", pad=9)
        t17.ygrid(ax2)
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("ln(1 + intensity), detected")
    which = f"joint proteins only ({joint.n_vars:,})" if use_joint else "all retained proteins of each condition"
    suptitle(fig, f"Per-cell ln-intensity distributions, {which}")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    t17.footer(fig, SCRIPT)
    run.figure(fig, "ln_intensity_per_cell_joint" if use_joint else "ln_intensity_per_cell_all",
               f"dct_intake per-cell ln intensity distributions ({'joint' if use_joint else 'all'} proteins)")


def fig_detection_rank(run, objects: dict, joint) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, use_joint in ((axes[0], False), (axes[1], True)):
        for cond in lib.CONDITIONS:
            if use_joint:
                mask = (joint.obs["condition"] == cond).to_numpy()
                det = joint.layers["intensity"][mask] > 0
                plates = joint.obs["plate"].astype(str).to_numpy()[mask]
            else:
                det = objects[cond].layers["intensity"] > 0
                plates = objects[cond].obs["plate"].astype(str).to_numpy()
            frac = np.sort(det.mean(axis=0))[::-1]
            ax.plot(np.arange(1, frac.size + 1), frac, color=lib.CONDITION_PALETTE[cond], linewidth=2.0)
            for plate in lib.PLATES:
                fp = np.sort(det[plates == plate].mean(axis=0))[::-1]
                ax.plot(np.arange(1, fp.size + 1), fp, color=lib.CONDITION_PALETTE[cond], linewidth=0.9,
                        linestyle=PLATE_LINESTYLE[plate])
        t17.ygrid(ax)
        ax.set_xlabel("protein rank (by detection fraction, within each condition)")
        ax.set_xlim(0, None)
    n_own = {c: objects[c].n_vars for c in lib.CONDITIONS}
    axes[0].set_title(f"Own retained proteins (Ctrl {n_own['Ctrl']:,}; ZeroK {n_own['ZeroK']:,})")
    axes[1].set_title(f"Joint proteins ({joint.n_vars:,}), ranked within each condition")
    axes[0].set_ylabel("fraction of retained cells with the protein")
    handles = [Line2D([0], [0], color=lib.CONDITION_PALETTE[c], linewidth=2.0, label=f"{c}, all retained cells")
               for c in lib.CONDITIONS]
    handles += [Line2D([0], [0], color=t17.INK["secondary"], linewidth=0.9, linestyle=PLATE_LINESTYLE[p],
                       label=f"{p} cells only") for p in lib.PLATES]
    axes[1].legend(handles=handles, loc="upper right")
    suptitle(fig, "Detection-rank curves")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    t17.footer(fig, SCRIPT)
    run.figure(fig, "detection_rank", "dct_intake detection-rank curves per condition and plate")


def fig_plate_maps(run, cells_all: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04], height_ratios=[1, 1, 0.95], hspace=0.55, wspace=0.12)
    vmax = float(cells_all["n_detected_all_rows"].max())
    cmap = t17.sequential_cmap()
    image = None
    for i, cond in enumerate(lib.CONDITIONS):
        for j, plate in enumerate(lib.PLATES):
            ax = fig.add_subplot(gs[i, j])
            sub = cells_all.loc[(cells_all["condition"] == cond) & (cells_all["plate"] == plate)]
            grid = np.full((t17.PLATE_ROWS, t17.PLATE_COLS), np.nan)
            removed = np.zeros_like(grid, dtype=bool)
            for well, n, keep in zip(sub["well"], sub["n_detected_all_rows"], sub["retained"]):
                r, c = t17.well_position(str(well).zfill(4))
                grid[r, c] = n
                removed[r, c] = not bool(keep)
            image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
            rr, cc_ = np.where(np.isnan(grid))
            for r, c in zip(rr, cc_):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=t17.INK["grid"], edgecolor="none"))
                ax.text(c, r, "×", ha="center", va="center", fontsize=6, color=t17.INK["primary"])
            rr, cc_ = np.where(removed)
            ax.scatter(cc_, rr, s=5, marker="o", facecolors="none", edgecolors=t17.INK["primary"], linewidths=0.6)
            ax.set_xticks([0, 23], ["1", "24"])
            ax.set_yticks([0, 15], ["A", "P"])
            ax.tick_params(length=0, labelsize=5.5)
            n_missing = int(np.isnan(grid).sum())
            ax.set_title(f"{cond} {plate}: {len(sub)} wells; {int(removed.sum())} removed (ring)"
                         + (f"; {n_missing} absent (×)" if n_missing else ""), fontsize=6.5)
    cax = fig.add_subplot(gs[0:2, 2])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("protein IDs per cell (all protein rows)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    bottom = gs[2, 0:2].subgridspec(1, 2, wspace=0.08)
    first = None
    for j, cond in enumerate(lib.CONDITIONS):
        ax = fig.add_subplot(bottom[0, j], sharey=first)
        first = first or ax
        for plate in lib.PLATES:
            sub = cells_all.loc[(cells_all["condition"] == cond) & (cells_all["plate"] == plate)]
            ax.scatter(sub["well"].astype(int), sub["n_detected_all_rows"], s=3, color=cc.PLATE_PALETTE[plate],
                       linewidths=0, alpha=0.85, rasterized=True, label=plate)
        ax.axhline(cc.MIN_CELL_PROTEINS, color=t17.INK["secondary"], linewidth=0.6, linestyle=(0, (3, 2)))
        t17.ygrid(ax)
        ax.set_xlim(0, 385)
        ax.set_xlabel("well index (1-384)")
        ax.set_title(f"{cond}: IDs per cell vs well index", fontsize=6.2)
        if j == 0:
            ax.set_ylabel("protein IDs per cell")
        else:
            ax.tick_params(labelleft=False)
            ax.legend(title="sort plate", title_fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      markerscale=2.5)
    fig.suptitle("Plate maps of protein IDs per cell (all cells, all protein rows; combined search)\n"
                 "Grid: 384-well plate, rows A-P x columns 1-24; wells 0001-0384 run row by row (A1-A24, then B1).",
                 x=0.01, ha="left", fontsize=7.5)
    t17.footer(fig, SCRIPT)
    run.figure(fig, "plate_maps", "dct_intake plate maps of protein IDs per cell")


def fig_qc(run, frame: pd.DataFrame) -> None:
    metrics = [("n_detected_joint", "protein IDs per cell (joint)"),
               ("log10_total_intensity", "log10 total intensity (joint)"),
               ("missing_fraction_joint", "missing fraction (joint)"),
               ("median_ln_detected", "median ln(1 + intensity), detected"),
               ("mitocarta_intensity_fraction", "MitoCarta intensity fraction"),
               ("stress_n_detected", "stress-panel proteins detected")]
    colors = t21tech.group_colors()
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8))
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        data = [frame.loc[frame["condition_plate"] == g, metric].dropna().to_numpy() for g in t21tech.GROUPS]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        ax.set_xticks(range(1, len(t21tech.GROUPS) + 1), t21tech.GROUPS)
        for patch, g in zip(bp["boxes"], t21tech.GROUPS):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.45 if g.endswith("P2") else 0.8)
        for median in bp["medians"]:
            median.set_color(t3.INK["primary"])
        ax.set_ylabel(label, fontsize=6.5)
        ax.tick_params(axis="x", labelsize=6, rotation=20)
        t3.ygrid(ax)
    suptitle(fig, "QC per condition and sort plate, joint proteins (boxes: median and IQR; whiskers 1.5 IQR; "
                  "lighter box = P2)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    t3.footer(fig, SCRIPT, "joint proteins (retained in both conditions)")
    run.figure(fig, "qc_by_condition_plate", "QC per condition and sort plate on joint proteins")


def fig_search_effect(run, scatter: dict[str, np.ndarray], bins: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(scatter), figsize=(7.4, 2.9), sharey=True)
    for ax, (name, xy) in zip(np.atleast_1d(axes), scatter.items()):
        det, diff = xy
        ok = np.isfinite(diff)
        ax.scatter(det[ok], diff[ok], s=2, alpha=0.18, color=t3.INK["background_cells"], linewidths=0,
                   rasterized=True)
        sub = bins[(bins["variant"] == name) & (bins["detection_bin"] != "all proteins")]
        centres = [np.mean([float(v) for v in b.split("-")]) for b in sub["detection_bin"]]
        ax.plot(centres, sub["median_diff_ln"], color=t3.SERIES[0], marker="o", ms=3, lw=1.2)
        ax.fill_between(centres, sub["q25_diff_ln"], sub["q75_diff_ln"], color=t3.SERIES[0], alpha=0.18, lw=0)
        ax.axhline(0, color=t3.INK["primary"], lw=0.6, ls="--")
        overall = bins[(bins["variant"] == name) & (bins["detection_bin"] == "all proteins")].iloc[0]
        ax.set_title(f"{name}\nmedian {overall['median_diff_ln']:+.2f}", fontsize=7)
        ax.set_xlabel("detection fraction (all cells)")
        t3.ygrid(ax)
    np.atleast_1d(axes)[0].set_ylabel("median ln Ctrl - ZeroK")
    suptitle(fig, "Per joint protein: median ln-intensity over detected Ctrl cells minus the same over detected "
                  "ZeroK cells")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    t3.footer(fig, SCRIPT, "one unnamed point per joint protein; line and band = median and IQR per detection decile")
    run.figure(fig, "search_effect", "Per-protein median ln-intensity difference against detection")


def fig_embeddings(run, joint, frame: pd.DataFrame) -> None:
    variants = list(t21.VARIANTS)
    fig, axes = plt.subplots(len(variants), 3, figsize=(7.1, 2.4 * len(variants)))
    for r, name in enumerate(variants):
        xy = np.asarray(joint.obsm[f"X_umap__{name}"])
        t3.umap_categorical(axes[r, 0], xy, frame["condition"], list(lib.CONDITIONS), lib.CONDITION_PALETTE,
                            f"{name}: condition")
        t3.umap_categorical(axes[r, 1], xy, frame["plate"], list(lib.PLATES), dict(cc.PLATE_PALETTE),
                            f"{name}: sort plate")
        t3.umap_continuous(fig, axes[r, 2], xy, frame["n_detected_joint"], f"{name}: protein IDs")
    counts = frame["condition"].value_counts()
    plates = frame["plate"].value_counts()
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=lib.CONDITION_PALETTE[c],
                      label=f"{c} (n = {int(counts[c])})") for c in lib.CONDITIONS]
    handles += [Line2D([], [], marker="o", ls="", ms=4, color=cc.PLATE_PALETTE[p],
                       label=f"{p} (n = {int(plates[p])})") for p in lib.PLATES]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.012))
    t3.footer(fig, SCRIPT, f"UMAP of the joint embedding (joint proteins, 20 PCs, k = 10, seed 42); {NOTE}")
    run.figure(fig, "joint_embeddings", "Joint UMAP per normalization variant")


def fig_mixing(run, mix: pd.DataFrame, depth: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    cond = mix[(mix["grouping"] == "condition") & (mix["label"] == "all_cells")]
    x = np.arange(len(cond))
    axes[0].bar(x - 0.2, cond["observed_same_label_fraction"], 0.4, color=t3.SERIES[0], label="observed")
    axes[0].bar(x + 0.2, cond["null_mean"], 0.4, color=t3.INK["background_cells"], label="permutation null")
    axes[0].set_xticks(x, cond["variant"], fontsize=7)
    axes[0].set_ylabel("same-condition neighbours")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=6, frameon=False)
    axes[0].set_title("condition mixing (k = 10)", fontsize=7)
    plate = mix[(mix["grouping"] == "plate") & (mix["label"] == "all_cells")]
    for i, c in enumerate(lib.CONDITIONS):
        sub = plate[plate["cells"] == c]
        axes[1].bar(np.arange(len(sub)) + (i - 0.5) * 0.4, sub["observed_same_label_fraction"], 0.4,
                    color=lib.CONDITION_PALETTE[c], label=c)
    axes[1].axhline(0.5, color=t3.INK["primary"], lw=0.6, ls="--")
    axes[1].set_xticks(np.arange(len(t21.VARIANTS)), list(t21.VARIANTS), fontsize=7)
    axes[1].set_ylabel("same-plate neighbours")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=6, frameon=False)
    axes[1].set_title("plate mixing within condition", fontsize=7)
    for i, pc in enumerate(("PC1", "PC2")):
        sub = depth[depth["component"] == pc]
        axes[2].bar(np.arange(len(sub)) + (i - 0.5) * 0.4, sub["abs_spearman_rho"], 0.4, color=t3.SERIES[i],
                    label=pc)
    axes[2].set_xticks(np.arange(len(t21.VARIANTS)), list(t21.VARIANTS), fontsize=7)
    axes[2].set_ylabel("|Spearman rho| with log IDs")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=6, frameon=False)
    axes[2].set_title("depth association", fontsize=7)
    for ax in axes:
        t3.ygrid(ax)
        ax.tick_params(axis="x", rotation=20)
    t3.footer(fig, SCRIPT, f"null = 1,000 label permutations; 0.5 line = chance for two plates; {NOTE}")
    run.figure(fig, "mixing_and_depth", "Condition and plate mixing, and depth association, per variant")


def main() -> None:
    parser = argparse.ArgumentParser(description="dct_intake: technical QC of the combined search.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "qc", args)
    t_start = time.time()
    checks = t5.Checks("dct_combined_joint")

    objects = {cond: lib.read_object(run, cond) for cond in lib.CONDITIONS}
    joint = lib.read_object(run, "joint")
    cells_all = lib.read_cells(run)

    # intake_core step 5 tables.
    cells = t17qc.per_cell(objects, joint)
    cells = cells.rename(columns=joint_names(cells.columns))
    proteins = per_protein(objects)
    run.table(cells.drop(columns="group").reset_index(), "qc_per_cell")
    run.table(by_condition_plate(cells), "qc_by_condition_plate")
    run.table(proteins, "qc_per_protein")
    values = t17qc.intensity_values(objects)
    run.table(values, "qc_intensity_values")
    for row in values.itertuples(index=False):
        run.note(f"{row.condition}: smallest detected value {row.min:g}; {row.n_equal_1} detected values equal 1 "
                 f"({row.fraction_equal_1:.4%}).")

    # zerok_core step 3 tables on the joint object.
    var_names = joint.var_names.astype(str)
    condition = joint.obs["condition"].astype(str).to_numpy()
    is_zerok = condition == "ZeroK"
    n_cells = {c: int((condition == c).sum()) for c in lib.CONDITIONS}
    intensity = cc.dense(joint.layers["intensity"]).astype(np.float64)
    norms = {name: t21.normalize_joint(intensity, condition, name)[0] for name in t21.VARIANTS}
    for name in t21.VARIANTS:
        stored = joint.layers["ln1p_intensity" if name == t21.BASELINE_VARIANT else f"norm__{name}"]
        diff = float(np.abs(norms[name] - np.asarray(stored, dtype=np.float64)).max())
        checks.add(f"{name}: recomputed variant equals the stored layer (float32)", f"max abs diff {diff:.2e}",
                   diff < 1e-5, "< 1e-5")
    flags = mito_flags(run, joint.var)
    checks.add("MitoCarta flags found for every joint protein",
               f"{int(flags['mitocarta'].notna().sum())}/{joint.n_vars}; {int(flags['mitocarta'].sum())} MitoCarta",
               len(flags) == joint.n_vars and bool(flags["mitocarta"].notna().all()))
    stress_cov, stress_present = t21tech.stress_proteins(
        pd.DataFrame({"pg_genes_ctrl": joint.var["pg_genes"].to_numpy()}, index=joint.var_names))
    stress_idx = [list(var_names).index(p) for p in stress_present]
    checks.add("dissociation-stress panel genes present among the joint proteins",
               f"{int(stress_cov['present'].sum())}/{len(stress_cov)} genes in {len(stress_present)} groups", None)

    frame = t21tech.per_cell_table(joint, intensity, flags, stress_idx, norms)
    frame = frame.rename(columns=joint_names(frame.columns))
    metrics = ["n_detected_joint", "log10_total_intensity", "missing_fraction_joint", "median_ln_detected",
               "mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
               "stress_n_detected", *[f"stress_mean_z__{n}" for n in t21.VARIANTS]]
    summary = t21tech.qc_summary(frame, metrics)
    bins, scatter, detection = t21tech.search_effect(intensity, is_zerok, norms, n_cells)
    depth, mix = depth_and_mixing(joint, frame)
    checks.add("per-cell joint IDs equal the joint object obs", "equal",
               np.array_equal(frame["n_detected_joint"].to_numpy(), joint.obs["n_detected"].to_numpy()))
    for row in mix[(mix["grouping"] == "condition") & (mix["label"] == "all_cells")].itertuples(index=False):
        run.note(f"{row.variant}: same-condition neighbours {row.observed_same_label_fraction:.3f} "
                 f"(null mean {row.null_mean:.3f}, q97.5 {row.null_q975:.3f})")
    for row in depth.itertuples(index=False):
        run.note(f"{row.variant} {row.component}: |rho| with log IDs {row.abs_spearman_rho:.3f}")

    run.table(frame, "technical_per_cell")
    run.table(summary, "technical_by_condition_plate")
    run.table(bins, "search_effect_bins")
    run.table(detection, "search_effect_detection")
    run.table(stress_cov, "stress_panel_coverage")
    run.table(depth, "embedding_depth_correlation")
    run.table(mix, "embedding_mixing")
    run.table(checks.frame(), "technical_checks")

    # Figures.
    cells["group"] = cells["condition"] + " " + cells["plate"]
    fig_proteins_per_cell(run, cells_all, cells, joint.n_vars)
    fig_total_intensity(run, cells)
    fig_missingness(run, cells, proteins, joint)
    fig_ln_intensity(run, objects, joint, cells, use_joint=False)
    fig_ln_intensity(run, objects, joint, cells, use_joint=True)
    fig_detection_rank(run, objects, joint)
    fig_plate_maps(run, cells_all)
    fig_qc(run, frame)
    fig_search_effect(run, scatter, bins)
    fig_embeddings(run, joint, frame)
    fig_mixing(run, mix, depth)

    run.finish(parameters={"plate_map_layout": "wells 0001-0384 run row by row, A1-A24 then B1 (authors, 28 Oct 2026)",
                           "ln_histogram_bins": "0-17, 68 bins", "n_permutations": t21.N_PERMUTATIONS,
                           "n_pcs": t21.N_PCS, "k_neighbors": t21.K_NEIGHBORS,
                           "search_effect_bins": "detection deciles over all joint cells (zerok_core_technical)",
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_intake_technical_checks.csv.")


if __name__ == "__main__":
    main()
