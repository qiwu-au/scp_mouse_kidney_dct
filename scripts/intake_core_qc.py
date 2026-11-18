#!/usr/bin/env python
"""intake_core step 3: technical QC by condition and plate (descriptive; no biology).



Inputs (read-only): results/h5ad/intake_core_DCT_{ctrl,zerok,joint}.h5ad;
results/tables/intake_core/intake_core_dct_cells.csv.
Outputs:
  results/tables/intake_core/
    intake_core_qc_per_cell.csv             retained cells: IDs, missing fraction, total and ln intensity
                                           (own retained proteins and matched proteins)
    intake_core_qc_by_condition_plate.csv   min / quartiles / max of each per-cell metric by condition x plate
    intake_core_qc_per_protein.csv          detection and missing fraction per protein (overall and per plate)
    intake_core_qc_intensity_values.csv     detected-value distribution per condition (incl. values equal to 1)
  results/figures/intake_core/ (PDF + 600 dpi PNG)
    intake_core_proteins_per_cell, intake_core_total_intensity, intake_core_missingness,
    intake_core_ln_intensity_per_cell_all, intake_core_ln_intensity_per_cell_matched,
    intake_core_detection_rank, intake_core_plate_maps

Proteins per cell, intensities and detection are not directly comparable
between conditions: the two conditions were searched separately in Spectronaut.

Command (project root):  python scripts/intake_core_qc.py
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

import scp_common as cc
import intake_core_lib as lib

SCRIPT = "intake_core_qc.py"
GROUP_ORDER = [f"{c} {p}" for c in lib.CONDITIONS for p in lib.PLATES]
GROUP_COLORS = {g: cc.PLATE_PALETTE[g.split()[1]] for g in GROUP_ORDER}
PLATE_LINESTYLE = {"P1": (0, (4, 1.5)), "P2": (0, (1, 1.2))}


def ln_detected_stats(ln1p: np.ndarray) -> pd.DataFrame:
    """Median and quartiles of ln(1+intensity) over the detected values of each cell."""
    masked = np.where(ln1p > 0, ln1p, np.nan)
    q = np.nanpercentile(masked, [25, 50, 75], axis=1)
    return pd.DataFrame({"q25_ln1p_detected": q[0], "median_ln1p_detected": q[1], "q75_ln1p_detected": q[2]})


def per_cell(objects: dict, joint) -> pd.DataFrame:
    frames = []
    for cond in lib.CONDITIONS:
        ad_ = objects[cond]
        x = ad_.layers["intensity"]
        ln = ad_.layers["ln1p_intensity"]
        jm = (joint.obs["condition"] == cond).to_numpy()
        jx = joint.layers["intensity"][jm]
        jl = joint.layers["ln1p_intensity"][jm]
        if list(joint.obs_names[jm]) != list(ad_.obs_names):
            raise ValueError(f"{cond}: joint and condition objects hold different cells")
        frame = ad_.obs[["condition", "plate", "well"]].astype(str).copy()
        frame["n_detected"] = (x > 0).sum(axis=1)
        frame["missing_fraction"] = 1 - frame["n_detected"] / ad_.n_vars
        frame["total_intensity"] = x.sum(axis=1, dtype=np.float64)
        frame["log10_total_intensity"] = np.log10(frame["total_intensity"])
        frame = pd.concat([frame, ln_detected_stats(ln).set_index(frame.index)], axis=1)
        frame["n_detected_matched"] = (jx > 0).sum(axis=1)
        frame["missing_fraction_matched"] = 1 - frame["n_detected_matched"] / joint.n_vars
        frame["total_intensity_matched"] = jx.sum(axis=1, dtype=np.float64)
        frame["log10_total_intensity_matched"] = np.log10(frame["total_intensity_matched"])
        stats_m = ln_detected_stats(jl).set_index(frame.index).add_suffix("_matched")
        frames.append(pd.concat([frame, stats_m], axis=1))
    out = pd.concat(frames)
    out.index.name = "cell_id"
    out["group"] = out["condition"] + " " + out["plate"]
    return out


METRICS = {
    "condition": {"n_detected": "protein IDs per cell", "missing_fraction": "fraction missing per cell",
                  "log10_total_intensity": "log10 total intensity", "median_ln1p_detected": "median ln(1+intensity), detected"},
    "matched": {"n_detected_matched": "protein IDs per cell", "missing_fraction_matched": "fraction missing per cell",
                "log10_total_intensity_matched": "log10 total intensity",
                "median_ln1p_detected_matched": "median ln(1+intensity), detected"},
}


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
                                 "protein_set": "own retained proteins" if protein_set == "condition" else "matched (joint) proteins",
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
            "gene_name": ad_.var["gene_name"].to_numpy(), "match_status": ad_.var["match_status"].to_numpy(),
            "match_level": ad_.var["match_level"].to_numpy(), "in_joint": ad_.var["in_joint"].to_numpy(),
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


def intensity_values(objects: dict) -> pd.DataFrame:
    rows = []
    for cond in lib.CONDITIONS:
        x = objects[cond].layers["intensity"]
        v = x[x > 0].astype(np.float64)
        q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
        rows.append({"condition": cond, "n_detected_values": v.size, "min": v.min(), "max": v.max(),
                     "n_equal_1": int((v == 1).sum()), "fraction_equal_1": float((v == 1).mean()),
                     "n_below_10": int((v < 10).sum()),
                     **{f"p{p:02d}": val for p, val in zip((1, 5, 25, 50, 75, 95, 99), q)}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plate_legend(ax) -> None:
    """Sort-plate legend outside the axes, to the right (never over data)."""
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4, markerfacecolor=cc.PLATE_PALETTE[p],
                      markeredgewidth=0, label=p) for p in lib.PLATES]
    ax.legend(handles=handles, title="sort plate", title_fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def fig_proteins_per_cell(run, cells_all: pd.DataFrame, cells: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharey="row")
    for j, cond in enumerate(lib.CONDITIONS):
        sub = cells_all.loc[cells_all["condition"] == cond]
        ax = axes[0, j]
        lib.box_strip(ax, sub, "plate", "n_detected_all_rows", list(lib.PLATES), cc.PLATE_PALETTE,
                      highlight=~sub["retained"].to_numpy(bool))
        ax.axhline(cc.MIN_CELL_PROTEINS, color=lib.INK["secondary"], linewidth=0.6, linestyle=(0, (3, 2)))
        ax.text(1.55, cc.MIN_CELL_PROTEINS, f"{cc.MIN_CELL_PROTEINS} IDs", va="bottom", ha="right", fontsize=5.6,
                color=lib.INK["secondary"])
        n_removed = int((~sub["retained"].astype(bool)).sum())
        ax.set_title(f"{cond}: all {len(sub)} cells, all protein rows\n(ringed: {n_removed} removed, < "
                     f"{cc.MIN_CELL_PROTEINS} IDs)")
        sub2 = cells.loc[cells["condition"] == cond]
        lib.box_strip(axes[1, j], sub2, "plate", "n_detected_matched", list(lib.PLATES), cc.PLATE_PALETTE)
        axes[1, j].set_title(f"{cond}: {len(sub2)} retained cells, matched proteins only")
    axes[0, 0].set_ylabel("protein IDs per cell")
    axes[1, 0].set_ylabel("protein IDs per cell (matched)")
    for ax in axes.flat:
        ax.set_xlabel("sort plate (cells per plate)")
    plate_legend(axes[0, 1])
    fig.suptitle("Protein IDs per cell by condition and sort plate\n" + lib.CROSS_CONDITION_NOTE, x=0.01, ha="left",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    lib.footer(fig, SCRIPT)
    run.figure(fig, "proteins_per_cell", "intake_core protein IDs per cell by condition and plate")


def fig_total_intensity(run, cells: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharey="row")
    for j, cond in enumerate(lib.CONDITIONS):
        sub = cells.loc[cells["condition"] == cond]
        lib.box_strip(axes[0, j], sub, "plate", "log10_total_intensity", list(lib.PLATES), cc.PLATE_PALETTE)
        axes[0, j].set_title(f"{cond}: {len(sub)} retained cells, own retained proteins")
        lib.box_strip(axes[1, j], sub, "plate", "log10_total_intensity_matched", list(lib.PLATES), cc.PLATE_PALETTE)
        axes[1, j].set_title(f"{cond}: matched proteins only")
    axes[0, 0].set_ylabel("log10 total intensity")
    axes[1, 0].set_ylabel("log10 total intensity (matched)")
    for ax in axes.flat:
        ax.set_xlabel("sort plate (cells per plate)")
    plate_legend(axes[0, 1])
    fig.suptitle("Total intensity per retained cell (sum of PG.Quantity)\n" + lib.CROSS_CONDITION_NOTE, x=0.01,
                 ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    lib.footer(fig, SCRIPT)
    run.figure(fig, "total_intensity", "intake_core total intensity per cell by condition and plate")


def fig_missingness(run, cells: pd.DataFrame, proteins: pd.DataFrame, joint) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0))
    lib.box_strip(axes[0, 0], cells, "group", "missing_fraction", GROUP_ORDER, GROUP_COLORS)
    axes[0, 0].set_title("Per cell, own retained proteins")
    axes[0, 0].set_ylabel("fraction of proteins missing (NaN)")
    lib.box_strip(axes[0, 1], cells, "group", "missing_fraction_matched", GROUP_ORDER, GROUP_COLORS)
    axes[0, 1].set_title(f"Per cell, {joint.n_vars:,} matched proteins")
    axes[0, 0].set_xlabel("condition, sort plate (cells)")
    axes[0, 1].set_xlabel("condition, sort plate (cells)")
    plate_legend(axes[0, 1])
    bins = np.linspace(0, 1, 41)
    for ax, matched in ((axes[1, 0], False), (axes[1, 1], True)):
        for cond in lib.CONDITIONS:
            if matched:
                det = (joint.layers["intensity"][(joint.obs["condition"] == cond).to_numpy()] > 0).mean(axis=0)
            else:
                det = proteins.loc[proteins["condition"] == cond, "detection_fraction"].to_numpy()
            ax.hist(det, bins=bins, histtype="step", linewidth=1.2, color=lib.CONDITION_PALETTE[cond],
                    label=f"{cond} ({len(det):,} proteins)")
        lib.ygrid(ax)
        ax.set_xlabel("detection fraction per protein (1 - missing fraction)")
        ax.set_ylabel("proteins")
        ax.legend(loc="upper right", title="condition", title_fontsize=6)
    axes[1, 0].set_title("Per protein, own retained proteins")
    axes[1, 1].set_title("Per protein, matched proteins (each condition's cells)")
    fig.suptitle("Missingness by condition and sort plate\n" + lib.CROSS_CONDITION_NOTE, x=0.01, ha="left",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    lib.footer(fig, SCRIPT)
    run.figure(fig, "missingness", "intake_core missingness per cell and per protein")


def fig_ln_intensity(run, objects: dict, joint, cells: pd.DataFrame, matched: bool) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), sharey="row", gridspec_kw={"height_ratios": [1.1, 1]})
    bins = np.linspace(0, 17, 69)
    centres = (bins[:-1] + bins[1:]) / 2
    suffix = "_matched" if matched else ""
    for j, cond in enumerate(lib.CONDITIONS):
        if matched:
            mask = (joint.obs["condition"] == cond).to_numpy()
            ln = joint.layers["ln1p_intensity"][mask]
            plates = joint.obs["plate"].astype(str).to_numpy()[mask]
        else:
            ln = objects[cond].layers["ln1p_intensity"]
            plates = objects[cond].obs["plate"].astype(str).to_numpy()
        ax = axes[0, j]
        for plate in lib.PLATES:
            rows = ln[plates == plate]
            for r in rows:
                v = r[r > 0]
                h, _ = np.histogram(v, bins=bins, density=True)
                ax.plot(centres, h, color=cc.PLATE_PALETTE[plate], linewidth=0.3, alpha=0.08, rasterized=True)
        for plate in lib.PLATES:
            v = ln[plates == plate]
            h, _ = np.histogram(v[v > 0], bins=bins, density=True)
            ax.plot(centres, h, color=cc.PLATE_PALETTE[plate], linewidth=2.0, linestyle=PLATE_LINESTYLE[plate],
                    label=f"{plate}, all cells pooled")
        ax.set_xlabel("ln(1 + intensity), detected values")
        ax.set_title(f"{cond}: one thin line per cell ({ln.shape[0]} cells x {ln.shape[1]:,} proteins)")
        lib.ygrid(ax)
        ax.legend(loc="upper right")
        sub = cells.loc[cells["condition"] == cond].sort_values(["plate", "well"])
        x = np.arange(len(sub))
        ax2 = axes[1, j]
        for plate in lib.PLATES:
            m = (sub["plate"] == plate).to_numpy()
            ax2.vlines(x[m], sub[f"q25_ln1p_detected{suffix}"].to_numpy()[m],
                       sub[f"q75_ln1p_detected{suffix}"].to_numpy()[m], color=cc.PLATE_PALETTE[plate],
                       linewidth=0.35, alpha=0.6, rasterized=True)
            ax2.scatter(x[m], sub[f"median_ln1p_detected{suffix}"].to_numpy()[m], s=1.5, color=lib.INK["primary"],
                        linewidths=0, rasterized=True)
        boundary = int((sub["plate"] == "P1").sum())
        ax2.axvline(boundary - 0.5, color=lib.INK["muted"], linewidth=0.6)
        ax2.text(boundary / 2, 1.0, "P1", transform=ax2.get_xaxis_transform(), ha="center", va="bottom", fontsize=6)
        ax2.text(boundary + (len(sub) - boundary) / 2, 1.0, "P2", transform=ax2.get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=6)
        ax2.set_xlabel("retained cells ordered by plate and well")
        ax2.set_title(f"{cond}: median (dot) and IQR (line) per cell", pad=9)
        lib.ygrid(ax2)
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("ln(1 + intensity), detected")
    which = f"matched proteins only ({joint.n_vars:,})" if matched else "all retained proteins of each condition"
    fig.suptitle(f"Per-cell ln-intensity distributions, {which}\n" + lib.CROSS_CONDITION_NOTE, x=0.01, ha="left",
                 fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    lib.footer(fig, SCRIPT)
    run.figure(fig, "ln_intensity_per_cell_matched" if matched else "ln_intensity_per_cell_all",
               f"intake_core per-cell ln intensity distributions ({'matched' if matched else 'all'} proteins)")


def fig_detection_rank(run, objects: dict, joint) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, matched in ((axes[0], False), (axes[1], True)):
        for cond in lib.CONDITIONS:
            if matched:
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
        lib.ygrid(ax)
        ax.set_xlabel("protein rank (by detection fraction, within each condition)")
        ax.set_xlim(0, None)
    n_own = {c: objects[c].n_vars for c in lib.CONDITIONS}
    axes[0].set_title(f"Own retained proteins (Ctrl {n_own['Ctrl']:,}; ZeroK {n_own['ZeroK']:,})")
    axes[1].set_title(f"Matched proteins ({joint.n_vars:,}), ranked within each condition")
    axes[0].set_ylabel("fraction of retained cells with the protein")
    handles = [Line2D([0], [0], color=lib.CONDITION_PALETTE[c], linewidth=2.0, label=f"{c}, all retained cells")
               for c in lib.CONDITIONS]
    handles += [Line2D([0], [0], color=lib.INK["secondary"], linewidth=0.9, linestyle=PLATE_LINESTYLE[p],
                       label=f"{p} cells only") for p in lib.PLATES]
    axes[1].legend(handles=handles, loc="upper right")
    fig.suptitle("Detection-rank curves\n" + lib.CROSS_CONDITION_NOTE, x=0.01, ha="left", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    lib.footer(fig, SCRIPT)
    run.figure(fig, "detection_rank", "intake_core detection-rank curves per condition and plate")


def fig_plate_maps(run, cells_all: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.04], height_ratios=[1, 1, 0.95], hspace=0.55, wspace=0.12)
    vmax = float(cells_all["n_detected_all_rows"].max())
    cmap = lib.sequential_cmap()
    image = None
    for i, cond in enumerate(lib.CONDITIONS):
        for j, plate in enumerate(lib.PLATES):
            ax = fig.add_subplot(gs[i, j])
            sub = cells_all.loc[(cells_all["condition"] == cond) & (cells_all["plate"] == plate)]
            grid = np.full((lib.PLATE_ROWS, lib.PLATE_COLS), np.nan)
            removed = np.zeros_like(grid, dtype=bool)
            for well, n, keep in zip(sub["well"], sub["n_detected_all_rows"], sub["retained"]):
                r, c = lib.well_position(str(well).zfill(4))
                grid[r, c] = n
                removed[r, c] = not bool(keep)
            image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax, aspect="equal", interpolation="nearest")
            rr, cc_ = np.where(np.isnan(grid))
            for r, c in zip(rr, cc_):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=lib.INK["grid"], edgecolor="none"))
                ax.text(c, r, "×", ha="center", va="center", fontsize=6, color=lib.INK["primary"])
            rr, cc_ = np.where(removed)
            ax.scatter(cc_, rr, s=5, marker="o", facecolors="none", edgecolors=lib.INK["primary"], linewidths=0.6)
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
        ax.axhline(cc.MIN_CELL_PROTEINS, color=lib.INK["secondary"], linewidth=0.6, linestyle=(0, (3, 2)))
        lib.ygrid(ax)
        ax.set_xlim(0, 385)
        ax.set_xlabel("well index (1-384)")
        ax.set_title(f"{cond}: IDs per cell vs well index", fontsize=6.2)
        if j == 0:
            ax.set_ylabel("protein IDs per cell")
        else:
            ax.tick_params(labelleft=False)
            ax.legend(title="sort plate", title_fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      markerscale=2.5)
    fig.suptitle("Plate maps of protein IDs per cell (all cells, all protein rows)\n"
                 "Grid: 384-well plate, rows A-P x columns 1-24; wells 0001-0384 run row by row (A1-A24, then B1).",
                 x=0.01, ha="left", fontsize=7.5)
    lib.footer(fig, SCRIPT)
    run.figure(fig, "plate_maps", "intake_core plate maps of protein IDs per cell")


def main() -> None:
    parser = argparse.ArgumentParser(description="intake_core: technical QC by condition and plate.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "qc", args)
    lib.apply_style()

    objects = {cond: lib.read_t17_object(run, cond) for cond in lib.CONDITIONS}
    joint = lib.read_t17_object(run, "joint")
    cells_all = lib.read_t17_table(run, "dct_cells", dtype={"well": str})
    cells_all["retained"] = cells_all["retained"].astype(str).str.lower().eq("true")

    cells = per_cell(objects, joint)
    proteins = per_protein(objects)
    run.table(cells.drop(columns="group").reset_index(), "qc_per_cell")
    run.table(by_condition_plate(cells), "qc_by_condition_plate")
    run.table(proteins, "qc_per_protein")
    values = intensity_values(objects)
    run.table(values, "qc_intensity_values")
    for row in values.itertuples(index=False):
        run.note(f"{row.condition}: smallest detected value {row.min:g}; {row.n_equal_1} detected values equal 1 "
                 f"({row.fraction_equal_1:.4%}).")

    fig_proteins_per_cell(run, cells_all, cells)
    fig_total_intensity(run, cells)
    fig_missingness(run, cells, proteins, joint)
    fig_ln_intensity(run, objects, joint, cells, matched=False)
    fig_ln_intensity(run, objects, joint, cells, matched=True)
    fig_detection_rank(run, objects, joint)
    fig_plate_maps(run, cells_all)
    run.finish(parameters={"plate_map_layout": "wells 0001-0384 run row by row, A1-A24 then B1 (authors, 28 Oct 2026)",
                           "ln_histogram_bins": "0-17, 68 bins"})


if __name__ == "__main__":
    main()
