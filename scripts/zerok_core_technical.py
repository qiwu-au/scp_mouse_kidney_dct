#!/usr/bin/env python
"""zerok_core step 3: technical comparability of the two conditions (descriptive).



Technical and composition-free summaries only (zerok_core plan, 'Order matters'): no
protein-level or label-fraction comparison between conditions, and no protein is named
in the search-effect output. Runs on the 5,130 matched proteins of the intake_core joint object.

What it does:
  1. QC per condition and sort plate: proteins per cell, total intensity, missingness,
     ln-intensity per cell, MitoCarta fractions, dissociation-stress score (ctrl_reviewer_core panel).
  2. Search effect: per matched protein, the difference in median ln-intensity between
     conditions against detection, reported as binned summaries (no protein named).
  3. The three normalization variants zerok_response_core reports (zerok_core_lib.VARIANTS), each with
     its formula and diagnostics. No winner is chosen.
  4. Joint embedding per variant: per-protein z-score over all 1,420 cells, arpack PCA
     (20 PCs), kNN graph (k = 10) and UMAP (seed 42); |Spearman rho| of PC1/PC2 with
     log n_detected; kNN condition mixing and, within each condition, plate mixing,
     both against a 1,000-permutation null.

Inputs (read-only): results/h5ad/intake_core_DCT_joint.h5ad;
results/tables/intake_core/intake_core_mitocarta_protein_flags.csv.
Outputs: tables/ and results/figures/zerok_core/, plus
results/h5ad/zerok_core_DCT_joint_norm.h5ad (the normalized layers and embeddings that
step 4 reads).

Command (project root):  python scripts/zerok_core_technical.py
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
import zerok_core_lib as lib

t3.apply_style()
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT = "zerok_core_technical.py"
GROUPS = ["Ctrl P1", "Ctrl P2", "ZeroK P1", "ZeroK P2"]
DETECTION_BINS = np.arange(0, 1.01, 0.1)


def group_colors() -> dict[str, str]:
    """Condition colour (intake_core slots 3-4), the second plate shown lighter."""
    out = {}
    for cond in lib.CONDITIONS:
        base = t17.CONDITION_PALETTE[cond]
        out[f"{cond} P1"] = base
        out[f"{cond} P2"] = base
    return out


def load_joint(run: cc.Run):
    import anndata as ad

    path = cc.DCT_NEW_OBJECTS["joint"]
    adata = ad.read_h5ad(run.use_input(path, "intake_core joint object (matched proteins, both conditions)"))
    return adata


def mito_flags(run: cc.Run, var: pd.DataFrame) -> pd.DataFrame:
    """intake_core MitoCarta flags of the Ctrl protein groups, reindexed to the joint proteins."""
    path = cc.out_path(t17.TASK, "tables", "mitocarta_protein_flags", ".csv")
    frame = pd.read_csv(run.use_input(path, "intake_core MitoCarta flags (both searches)"), keep_default_na=False)
    frame = frame[frame["search"] == "Ctrl"].set_index("protein")
    flags = frame.reindex(var["var_name_ctrl"].astype(str).to_numpy())
    for col in t3.SETS:
        flags[col] = flags[col].astype(str).str.lower().eq("true")
    flags.index = pd.Index(var.index.astype(str), name="protein")
    return flags


def stress_proteins(var: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Stress-panel genes present among the joint proteins (ctrl_reviewer_core rule, on the Ctrl PG.Genes strings)."""
    symbols = [t3.split_symbols(v) for v in var["pg_genes_ctrl"].astype(str)]
    names = list(var.index.astype(str))
    rows, present = [], []
    for gene in t3.STRESS_PANEL:
        hits = [n for n, syms in zip(names, symbols) if gene.lower() in syms]
        rows.append({"gene": gene, "present": bool(hits), "matrix_protein": ";".join(hits)})
        for h in hits:
            if h not in present:
                present.append(h)
    return pd.DataFrame(rows), present


def per_cell_table(adata, intensity: np.ndarray, flags: pd.DataFrame, stress_idx: list[int],
                   norms: dict[str, np.ndarray]) -> pd.DataFrame:
    detected = intensity > 0
    frame = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str), "condition": adata.obs["condition"].astype(str).to_numpy(),
        "plate": adata.obs["plate"].astype(str).to_numpy(),
        "n_detected_matched": detected.sum(axis=1),
        "missing_fraction_matched": 1 - detected.mean(axis=1),
        "total_intensity_matched": intensity.sum(axis=1),
    })
    frame["condition_plate"] = frame["condition"] + " " + frame["plate"]
    frame["log10_total_intensity"] = np.log10(frame["total_intensity_matched"])
    frame["log_n_detected"] = np.log(frame["n_detected_matched"])
    ln1p = norms[lib.BASELINE_VARIANT]
    frame["median_ln_detected"] = [np.median(row[d]) if d.any() else np.nan
                                   for row, d in zip(ln1p, detected)]
    # Positional concat: mito_metrics indexes by cell_id, this frame by position.
    mito = t3.mito_metrics(intensity, flags, frame["cell_id"]).reset_index(drop=True).drop(
        columns=["n_detected_proteins", "total_intensity", "log10_total_intensity"], errors="ignore")
    frame = pd.concat([frame, mito], axis=1)
    frame["stress_n_detected"] = detected[:, stress_idx].sum(axis=1)
    for name, matrix in norms.items():
        z = cc.scale_like_scanpy(matrix, max_value=10.0)
        frame[f"stress_mean_z__{name}"] = z[:, stress_idx].mean(axis=1)
    return frame


def qc_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for cond in (*lib.CONDITIONS, "both"):
        own = frame if cond == "both" else frame[frame["condition"] == cond]
        for plate in (*t17.PLATES, "all"):
            sub = own if plate == "all" else own[own["plate"] == plate]
            row = {"condition": cond, "plate": plate, "n_cells": int(len(sub))}
            for m in metrics:
                row[f"{m}_median"] = float(sub[m].median())
                row[f"{m}_q25"] = float(sub[m].quantile(0.25))
                row[f"{m}_q75"] = float(sub[m].quantile(0.75))
            rows.append(row)
    return pd.DataFrame(rows)


def search_effect(intensity: np.ndarray, is_zerok: np.ndarray, norms: dict[str, np.ndarray],
                  n_cells: dict[str, int]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-protein median ln difference against detection, as binned summaries only (no protein named)."""
    detected = intensity > 0
    det_ctrl = detected[~is_zerok].mean(axis=0)
    det_zerok = detected[is_zerok].mean(axis=0)
    det_overall = detected.mean(axis=0)
    rows, scatter = [], {}
    for name, matrix in norms.items():
        with np.errstate(invalid="ignore"):
            m_ctrl = np.nanmedian(np.where(detected[~is_zerok], matrix[~is_zerok], np.nan), axis=0)
            m_zerok = np.nanmedian(np.where(detected[is_zerok], matrix[is_zerok], np.nan), axis=0)
        diff = m_ctrl - m_zerok
        scatter[name] = np.vstack([det_overall, diff])
        usable = np.isfinite(diff)
        rows.append({"variant": name, "detection_bin": "all proteins", "n_proteins": int(usable.sum()),
                     "median_diff_ln": float(np.median(diff[usable])),
                     "q25_diff_ln": float(np.quantile(diff[usable], 0.25)),
                     "q75_diff_ln": float(np.quantile(diff[usable], 0.75)),
                     "fraction_positive": float((diff[usable] > 0).mean()),
                     "mean_abs_diff_ln": float(np.abs(diff[usable]).mean())})
        for lo, hi in zip(DETECTION_BINS[:-1], DETECTION_BINS[1:]):
            mask = usable & (det_overall >= lo) & (det_overall < hi + (1e-9 if hi == 1.0 else 0))
            if not mask.any():
                continue
            rows.append({"variant": name, "detection_bin": f"{lo:.1f}-{hi:.1f}", "n_proteins": int(mask.sum()),
                         "median_diff_ln": float(np.median(diff[mask])),
                         "q25_diff_ln": float(np.quantile(diff[mask], 0.25)),
                         "q75_diff_ln": float(np.quantile(diff[mask], 0.75)),
                         "fraction_positive": float((diff[mask] > 0).mean()),
                         "mean_abs_diff_ln": float(np.abs(diff[mask]).mean())})
    detection = pd.DataFrame({"detection_bin": "all proteins", "n_proteins": [len(det_overall)],
                              "median_detection_ctrl": [float(np.median(det_ctrl))],
                              "median_detection_zerok": [float(np.median(det_zerok))],
                              "n_cells_ctrl": n_cells["Ctrl"], "n_cells_zerok": n_cells["ZeroK"]})
    return pd.DataFrame(rows), scatter, detection


def embeddings(run: cc.Run, norms: dict[str, np.ndarray], frame: pd.DataFrame, cell_ids, var_names,
               checks: t5.Checks):
    """PCA, UMAP, depth correlation and mixing per variant."""
    depth_rows, mix_rows, obsm = [], [], {}
    log_ids = frame["log_n_detected"].to_numpy()
    for name, matrix in norms.items():
        h, umap_xy = lib.joint_embedding(matrix, cell_ids, var_names)
        pca = np.asarray(h.obsm["X_pca"])
        obsm[f"X_pca__{name}"] = pca[:, :lib.N_PCS]
        obsm[f"X_umap__{name}"] = umap_xy
        checks.add(f"{name}: PCA and UMAP shapes", f"{pca[:, :lib.N_PCS].shape}; {umap_xy.shape}",
                   pca.shape[0] == len(frame) and umap_xy.shape == (len(frame), 2))
        for pc in (0, 1):
            rho, p = stats.spearmanr(pca[:, pc], log_ids)
            depth_rows.append({"variant": name, "component": f"PC{pc + 1}", "spearman_rho": float(rho),
                               "abs_spearman_rho": float(abs(rho)), "p_value": float(p),
                               "against": "log n_detected (matched proteins)"})
        mix = lib.mixing(pca[:, :lib.N_PCS], frame["condition"], list(lib.CONDITIONS))
        mix.insert(0, "variant", name)
        mix.insert(1, "grouping", "condition")
        mix.insert(2, "cells", "all")
        mix_rows.append(mix)
        for cond in lib.CONDITIONS:
            own = (frame["condition"] == cond).to_numpy()
            pm = lib.mixing(pca[own][:, :lib.N_PCS], frame.loc[own, "plate"], list(t17.PLATES))
            pm.insert(0, "variant", name)
            pm.insert(1, "grouping", "plate")
            pm.insert(2, "cells", cond)
            mix_rows.append(pm)
    return pd.DataFrame(depth_rows), pd.concat(mix_rows, ignore_index=True), obsm


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_qc(run, frame: pd.DataFrame) -> None:
    metrics = [("n_detected_matched", "protein IDs per cell (matched)"),
               ("log10_total_intensity", "log10 total intensity"),
               ("missing_fraction_matched", "missing fraction"),
               ("median_ln_detected", "median ln(1 + intensity), detected")]
    colors = group_colors()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        data = [frame.loc[frame["condition_plate"] == g, metric].dropna().to_numpy() for g in GROUPS]
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        ax.set_xticks(range(1, len(GROUPS) + 1), GROUPS)
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.45 if g.endswith("P2") else 0.8)
        for median in bp["medians"]:
            median.set_color(t3.INK["primary"])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=7)
        t3.ygrid(ax)
    t3.footer(fig, SCRIPT, "matched proteins only; the two searches are separate, so values are not directly "
                           "comparable between conditions")
    run.figure(fig, "qc_by_condition_plate", "QC per condition and sort plate on matched proteins")


def fig_search_effect(run, scatter: dict[str, np.ndarray], bins: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(scatter), figsize=(7.4, 2.7), sharey=True)
    for ax, (name, xy) in zip(np.atleast_1d(axes), scatter.items()):
        det, diff = xy
        ok = np.isfinite(diff)
        ax.scatter(det[ok], diff[ok], s=2, alpha=0.18, color=t3.INK["background_cells"], linewidths=0)
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
    t3.footer(fig, SCRIPT, "one unnamed point per matched protein; line and band = median and IQR per "
                           "detection decile")
    run.figure(fig, "search_effect", "Per-protein median ln-intensity difference against detection")


def fig_embeddings(run, obsm: dict[str, np.ndarray], frame: pd.DataFrame) -> None:
    variants = list(lib.VARIANTS)
    fig, axes = plt.subplots(len(variants), 3, figsize=(7.1, 2.4 * len(variants)))
    for r, name in enumerate(variants):
        xy = obsm[f"X_umap__{name}"]
        t3.umap_categorical(axes[r, 0], xy, frame["condition"], list(lib.CONDITIONS),
                            t17.CONDITION_PALETTE, f"{name}: condition")
        t3.umap_categorical(axes[r, 1], xy, frame["plate"], list(t17.PLATES), dict(cc.PLATE_PALETTE),
                            f"{name}: sort plate")
        t3.umap_continuous(fig, axes[r, 2], xy, frame["n_detected_matched"], f"{name}: protein IDs")
    # One figure-level legend: the colours are the same in every row.
    from matplotlib.lines import Line2D

    counts = frame["condition"].value_counts()
    plates = frame["plate"].value_counts()
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=t17.CONDITION_PALETTE[c],
                      label=f"{c} (n = {int(counts[c])})") for c in lib.CONDITIONS]
    handles += [Line2D([], [], marker="o", ls="", ms=4, color=cc.PLATE_PALETTE[p],
                       label=f"{p} (n = {int(plates[p])})") for p in t17.PLATES]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    t3.footer(fig, SCRIPT, "UMAP of the joint embedding (matched proteins, 20 PCs, k = 10, seed 42)")
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
                    color=t17.CONDITION_PALETTE[c], label=c)
    axes[1].axhline(0.5, color=t3.INK["primary"], lw=0.6, ls="--")
    axes[1].set_xticks(np.arange(len(lib.VARIANTS)), list(lib.VARIANTS), fontsize=7)
    axes[1].set_ylabel("same-plate neighbours")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=6, frameon=False)
    axes[1].set_title("plate mixing within condition", fontsize=7)

    for i, pc in enumerate(("PC1", "PC2")):
        sub = depth[depth["component"] == pc]
        axes[2].bar(np.arange(len(sub)) + (i - 0.5) * 0.4, sub["abs_spearman_rho"], 0.4,
                    color=t3.SERIES[i], label=pc)
    axes[2].set_xticks(np.arange(len(lib.VARIANTS)), list(lib.VARIANTS), fontsize=7)
    axes[2].set_ylabel("|Spearman rho| with log IDs")
    axes[2].set_ylim(0, 1)
    axes[2].legend(fontsize=6, frameon=False)
    axes[2].set_title("depth association", fontsize=7)
    for ax in axes:
        t3.ygrid(ax)
        ax.tick_params(axis="x", rotation=20)
    t3.footer(fig, SCRIPT, "null = 1,000 label permutations; 0.5 line = chance for two plates")
    run.figure(fig, "mixing_and_depth", "Condition and plate mixing, and depth association, per variant")


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_core step 3: technical comparability.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "technical", args)
    t_start = time.time()
    checks = t5.Checks("dct_joint")

    adata = load_joint(run)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    condition = adata.obs["condition"].astype(str).to_numpy()
    is_zerok = condition == "ZeroK"
    n_cells = {c: int((condition == c).sum()) for c in lib.CONDITIONS}
    checks.add("joint object cells per condition", "; ".join(f"{k} {v}" for k, v in n_cells.items()),
               n_cells == {"Ctrl": 727, "ZeroK": 693}, "Ctrl 727; ZeroK 693")
    checks.add("joint object proteins", f"{adata.n_vars}", adata.n_vars == 5130, "5130 matched proteins")

    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    norms, infos = {}, {}
    for name in lib.VARIANTS:
        norms[name], infos[name] = lib.normalize_joint(intensity, condition, name)
        detected = intensity > 0
        checks.add(f"{name}: every detected value > 0 and every missing value 0",
                   f"min detected {norms[name][detected].min():.4g}; "
                   f"max missing {np.abs(norms[name][~detected]).max():.4g}",
                   bool(norms[name][detected].min() > 0) and bool(np.abs(norms[name][~detected]).max() == 0))
    stored_ln1p = cc.dense(adata.layers["ln1p_intensity"]).astype(np.float64)
    diff_ln1p = float(np.abs(norms["none"] - stored_ln1p).max())
    checks.add("variant 'none' equals the stored ln1p layer", f"max abs diff {diff_ln1p:.2e}",
               diff_ln1p < 1e-6, "< 1e-6")
    ctrl_unchanged = float(np.abs(norms["condition_align"][~is_zerok] - norms["none"][~is_zerok]).max())
    checks.add("condition_align leaves the control cells unchanged", f"max abs diff {ctrl_unchanged:.2e}",
               ctrl_unchanged == 0.0, "0 (control is the reference)")
    run.note(f"condition_align delta = {infos['condition_align']['delta_ln']:+.4f} ln units from "
             f"{infos['condition_align']['n_proteins_used_for_delta']} proteins "
             f"(uniform guard shift {infos['condition_align']['uniform_shift_applied']:.3g})")

    flags = mito_flags(run, adata.var)
    checks.add("MitoCarta flags found for every joint protein",
               f"{int(flags['mitocarta'].notna().sum())}/{adata.n_vars}; "
               f"{int(flags['mitocarta'].sum())} MitoCarta",
               len(flags) == adata.n_vars and flags["mitocarta"].notna().all())
    stress_cov, stress_present = stress_proteins(adata.var)
    stress_idx = [list(var_names).index(p) for p in stress_present]
    checks.add("dissociation-stress panel genes present among the matched proteins",
               f"{int(stress_cov['present'].sum())}/{len(stress_cov)} genes in {len(stress_present)} groups",
               None)

    frame = per_cell_table(adata, intensity, flags, stress_idx, norms)
    metrics = ["n_detected_matched", "log10_total_intensity", "missing_fraction_matched", "median_ln_detected",
               "mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction",
               "stress_n_detected", *[f"stress_mean_z__{n}" for n in lib.VARIANTS]]
    summary = qc_summary(frame, metrics)
    bins, scatter, detection = search_effect(intensity, is_zerok, norms, n_cells)
    depth, mix, obsm = embeddings(run, norms, frame, cell_ids, var_names, checks)

    for row in mix[(mix["grouping"] == "condition") & (mix["label"] == "all_cells")].itertuples(index=False):
        run.note(f"{row.variant}: same-condition neighbours {row.observed_same_label_fraction:.3f} "
                 f"(null mean {row.null_mean:.3f}, q97.5 {row.null_q975:.3f})")
    for row in depth.itertuples(index=False):
        run.note(f"{row.variant} {row.component}: |rho| with log IDs {row.abs_spearman_rho:.3f}")

    run.table(frame, "technical_per_cell")
    run.table(summary, "technical_by_condition_plate")
    run.table(lib.variant_table(infos), "normalization_variants")
    run.table(bins, "search_effect_bins")
    run.table(detection, "search_effect_detection")
    run.table(stress_cov, "stress_panel_coverage")
    run.table(depth, "embedding_depth_correlation")
    run.table(mix, "embedding_mixing")
    run.table(checks.frame(), "technical_checks")

    fig_qc(run, frame)
    fig_search_effect(run, scatter, bins)
    fig_embeddings(run, obsm, frame)
    fig_mixing(run, mix, depth)

    out = adata.copy()
    for name in lib.VARIANTS:
        if name != lib.BASELINE_VARIANT:
            out.layers[f"norm__{name}"] = norms[name].astype(np.float32)
    for key, value in obsm.items():
        out.obsm[key] = np.asarray(value, dtype=np.float32)
    for col in ("n_detected_matched", "missing_fraction_matched", "log10_total_intensity", "log_n_detected",
                "median_ln_detected", "mitocarta_intensity_fraction", "stress_n_detected"):
        out.obs[col] = frame[col].to_numpy()
    out.uns["zerok_core"] = {"variants": dict(lib.VARIANTS), "baseline_variant": lib.BASELINE_VARIANT,
                             "n_pcs": lib.N_PCS, "k_neighbors": lib.K_NEIGHBORS, "seed": lib.SEED,
                             "note": "variant 'none' is layers['ln1p_intensity']; the others are "
                                     "layers['norm__<variant>']. Labels are added in step 4."}
    lib.write_h5ad(run, out, "DCT_joint_norm")

    run.finish(parameters={"variants": dict(lib.VARIANTS), "n_permutations": lib.N_PERMUTATIONS,
                           "n_pcs": lib.N_PCS, "k_neighbors": lib.K_NEIGHBORS,
                           "condition_align": infos["condition_align"],
                           "pca": "arpack, 50 comps, seed 42 (preprocessing_lib.build_embedding)",
                           "runtime_s": round(time.time() - t_start, 1), "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_core_technical_checks.csv.")


if __name__ == "__main__":
    main()
