#!/usr/bin/env python
"""kidney_annotation addendum: kidney sensitivity of the kidney_annotation labels to excluding QC-flagged cells (qc_mito design).

Requested by the authors on 14 Oct 2026, so that the point-7
response can cover the adopted kidney labels (`kidney_t15_label`, option a).

Exclusion sets (within-label outliers, 3 x MAD, under kidney_t15_label; qc_mito_mito_qc.add_outlier_flags on the
qc_mito per-cell QC table):
  mito_flagged  any mitochondrial metric (MitoCarta intensity / ID fraction, OXPHOS intensity fraction)
  any_flagged   any qc_mito metric (mitochondrial or depth)
Runs (the same pipeline every time, protein set fixed):
  baseline  all cells (must reproduce kidney_t15_label at seed 42)
  observed  without the flagged cells
  random    100 draws per set removing the same number of cells, stratified by kidney_t15_label
            (qc_mito_sensitivity.stratified_draw; numpy default_rng(42))
Pipeline = kidney_annotation option (a): ln(1 + intensity) -> per-protein z-score (clip 10) -> PCA (arpack, 20) -> neighbours
(10, 20 PCs) (qc_mito_sensitivity.rebuild_graph) -> Leiden 2.0 (igraph) -> cluster_de DE rule + rank-free rescue with
V3 + the endothelial row, minimum cluster size 5 (kidney_annotation_scan_lib.annotate_clustering,
kidney_annotation_rankfree_sensitivity.rank_free). DE, marker z and detection use the remaining cells only.
Leiden seeds: baseline and observed 0, 1, 2, 3, 42; random draws 42 (kidney_annotation labels are defined at seed 42).
Outcomes on the remaining cells: agreement with kidney_t15_label, number of types resolved, recall per kidney_annotation type,
Unknown fraction, ARI of the clusters vs the baseline clusters (same seed).

Inputs (read-only): kidney_object canonical kidney object and labels; KIDNEY_markers_V3.xlsx; qc_mito kidney per-cell QC table;
  kidney_annotation_cluster_labels_per_cell_a_endo.csv.
Outputs: results/tables/kidney_annotation/kidney_annotation_sensitivity_*.csv;
  results/figures/kidney_annotation/kidney_annotation_sensitivity_vs_random.pdf/.png

Command (project root):  python scripts/kidney_annotation_kidney_sensitivity.py
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
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _scanpy_compat import ensure_scanpy_compat

import scp_common as cc
import qc_mito_lib as t3
import qc_mito_mito_qc as mq
import qc_mito_sensitivity as sens
import cluster_de_lib as t5
import kidney_annotation_scan_lib as s
from kidney_annotation_extended_markers import marker_set
from kidney_annotation_rankfree_sensitivity import as_assignment, rank_free

SCANPY_COMPAT_PATCHED = ensure_scanpy_compat()
SCRIPT = "kidney_annotation_kidney_sensitivity.py"
COL = "kidney_t15_label"
OBS_SEEDS = (0, 1, 2, 3, 42)
NULL_SEED = 42
N_DRAWS = 100
RESOLUTION = 2.0
KIDNEY_ANNOTATION_LABELS = cc.OUT_ROOTS["tables"] / s.TASK / "kidney_annotation_cluster_labels_per_cell_a_endo.csv"
QC_MITO_QC = cc.OUT_ROOTS["tables"] / "qc_mito" / "qc_mito_kidney_qc_per_cell.csv"
TYPES = ("Endothelial", "JG", "PCT", "PST", "TAL", "DCT1", "PC", "IC-B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kidney_annotation addendum: kidney sensitivity to excluding QC-flagged cells.")
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    cc.add_common_args(parser)
    return parser.parse_args()


class Pipeline:
    def __init__(self, intensity: np.ndarray, cell_ids, var_names, ms: t5.MarkerSets):
        self.intensity, self.cell_ids, self.var_names, self.ms = intensity, np.asarray(cell_ids), var_names, ms
        self.specific = {t: ms.specific_present(t) for t in ms.lists if ms.specific_present(t)}

    def run(self, keep: np.ndarray, seeds) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Rebuild on the kept cells; per seed: (clusters, rank-free labels)."""
        inten = self.intensity[keep]
        ln1p = np.log1p(inten)
        ids = self.cell_ids[keep]
        g = sens.rebuild_graph(ln1p, ids, self.var_names)
        det = inten > 0
        cache = t5.rank_cache(ln1p, det)
        x_z = pd.DataFrame(cc.scale_like_scanpy(ln1p), index=ids, columns=self.var_names)
        det_f = pd.DataFrame(det, index=ids, columns=self.var_names)
        out = {}
        for seed in seeds:
            cl = sens.leiden(g, RESOLUTION, seed, f"sens_{seed}").to_numpy().astype(str)
            de, _, asg = s.annotate_clustering(cache, cl, self.var_names, f"sens_s{seed}", self.ms, x_z, det_f)
            rf = rank_free(asg, de, self.specific)
            out[seed] = (cl, s.per_cell_labels(cl, as_assignment(rf, "rankfree")).astype(str))
        return out


def outcomes(lab_new: np.ndarray, cl_new: np.ndarray, truth: np.ndarray, cl_base: np.ndarray) -> dict:
    row = {"agreement": float((lab_new == truth).mean()),
           "n_types": len(set(lab_new) - {s.UNKNOWN, s.BELOW_MIN}),
           "unknown_fraction": float((lab_new == s.UNKNOWN).mean()),
           "ari_vs_baseline": float(adjusted_rand_score(cl_base, cl_new)),
           "types": ";".join(cc.ordered_labels(set(lab_new) - {s.UNKNOWN, s.BELOW_MIN}, (s.ENDO_TYPE, *s.ORDER)))}
    for t in TYPES:
        m = truth == t
        row[f"recall_{t}"] = float((lab_new[m] == t).mean()) if m.any() else np.nan
        row[f"n_{t}"] = int(m.sum())
    row["all_8_types_present"] = all(t in set(lab_new) for t in TYPES)
    return row


def figure(run: cc.Run, runs: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    t3.apply_style()
    outs = ["agreement", "n_types", "unknown_fraction", "ari_vs_baseline", "recall_Endothelial", "recall_JG",
            "recall_IC-B", "recall_DCT1", "recall_PC"]
    disp = {"agreement": "agreement with\nT15 labels", "n_types": "types\nresolved",
            "unknown_fraction": "Unknown\nfraction", "ari_vs_baseline": "ARI vs\nbaseline",
            **{f"recall_{t}": f"recall\n{t}" for t in TYPES}}
    sets = ["mito_flagged", "any_flagged"]
    fig, axes = plt.subplots(len(sets), len(outs), figsize=(180 / 25.4, 3.3), squeeze=False)
    rng = np.random.default_rng(s.SEED)
    for i, es in enumerate(sets):
        rnd = runs[(runs["exclusion_set"] == es) & (runs["run"] == "random")]
        obs = runs[(runs["exclusion_set"] == es) & (runs["run"] == "observed") & (runs["seed"] == NULL_SEED)]
        base = runs[(runs["run"] == "baseline") & (runs["seed"] == NULL_SEED)]
        for j, o in enumerate(outs):
            ax = axes[i, j]
            v = rnd[o].to_numpy(float)
            ax.scatter(rng.uniform(-0.25, 0.25, len(v)), v, s=3, color=t3.INK["muted"], alpha=0.6, linewidths=0)
            lo, hi = np.nanquantile(v, [0.025, 0.975])
            ax.vlines(0.45, lo, hi, color=t3.INK["secondary"], linewidth=1.2)
            ax.scatter([0.45], obs[o], s=18, marker="D", color=t3.SERIES[1], zorder=3)
            if o in base:
                ax.axhline(float(base[o].iloc[0]), color=t3.SERIES[0], linewidth=0.7, linestyle=(0, (3, 2)))
            ax.set_xticks([])
            ax.set_xlim(-0.5, 0.7)
            ax.tick_params(labelsize=5)
            if i == 0:
                ax.set_title(disp[o], fontsize=5.4)
            if j == 0:
                ax.set_ylabel(es.replace("_", " "), fontsize=6)
    fig.suptitle("KIDNEY, kidney_annotation labels: excluding QC-flagged cells (orange diamond) vs 100 stratified random removals "
                 "(grey; bar = central 95%); blue dashed = all cells", x=0.01, ha="left", fontsize=6.6)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    t3.footer(fig, SCRIPT, "seed 42; kidney_annotation option (a) pipeline rebuilt on the remaining cells")
    run.figure(fig, "sensitivity_vs_random", "kidney_annotation kidney sensitivity to excluding QC-flagged cells")


def main() -> None:
    args = parse_args()
    sys.unraisablehook = sens._unraisable
    run = cc.Run(s.TASK, "kidney_sensitivity", args)
    t_start = time.time()
    checks = t5.Checks(s.DATASET)
    adata, labels = t5.load(run, s.DATASET)
    cell_ids = adata.obs_names.astype(str)
    var_names = adata.var_names.astype(str)
    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    t15 = pd.read_csv(run.use_input(KIDNEY_ANNOTATION_LABELS, "kidney_annotation adopted kidney labels (a)"), dtype={"cell_id": str}).set_index("cell_id")
    checks.add("kidney_annotation label table has the canonical cells in order", f"{list(t15.index) == list(cell_ids)}",
               list(t15.index) == list(cell_ids))
    truth_all = t15[COL].astype(str).to_numpy()
    v3 = t5.read_markers(run, s.DATASET, var_names)
    ms = marker_set({**v3.lists, s.ENDO_TYPE: list(s.ENDOTHELIAL_MARKERS)}, var_names)

    # Exclusion sets under kidney_t15_label (qc_mito functions)
    t3.DATASETS["kidney"] = {**t3.DATASETS["kidney"], "label_columns": (COL,), "plot_columns": (COL,)}
    t3.SHORT_COLUMN[COL] = "t15"
    qc = pd.read_csv(run.use_input(QC_MITO_QC, "qc_mito kidney per-cell QC"), dtype={"cell_id": str}).set_index("cell_id")
    qc = qc.reindex(cell_ids)
    frame = qc[[c for c in qc.columns if not c.startswith(("outlier__", "flag__", "evaluated__"))]].copy()
    frame[COL] = truth_all
    frame, _ = mq.add_outlier_flags(frame, "kidney")
    masks = {"mito_flagged": frame["flag__t15__mito_any"].astype(bool).to_numpy(),
             "any_flagged": frame["flag__t15__any"].astype(bool).to_numpy()}
    for k, m in masks.items():
        checks.add(f"exclusion set {k} (kidney_t15_label, 3 x MAD)", f"{int(m.sum())} cells", m.sum() > 0)
    comp = []
    for k, m in masks.items():
        for col in (COL, "kidney_pre_label"):
            src = frame[COL] if col == COL else labels[col]
            vc = src[m].astype(str).value_counts()
            comp.append({"exclusion_set": k, "label_column": col, "n_excluded": int(m.sum()),
                         "excluded_by_label": "; ".join(f"{a}: {b}" for a, b in vc.items())})

    pipe = Pipeline(intensity, cell_ids, var_names, ms)
    rows = []
    print("[kidney] baseline", flush=True)
    all_keep = np.ones(len(cell_ids), bool)
    base = pipe.run(all_keep, OBS_SEEDS)
    same = int((base[NULL_SEED][1] == truth_all).sum())
    checks.add("baseline (all cells, rebuilt, seed 42) reproduces kidney_t15_label", f"{same}/{len(cell_ids)}",
               same == len(cell_ids), "= all cells")
    if not checks.ok:
        run.table(checks.frame(), "sensitivity_checks")
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit("Baseline does not reproduce kidney_t15_label; see kidney_annotation_sensitivity_checks.csv.")
    for seed in OBS_SEEDS:
        cl, lab = base[seed]
        rows.append({"exclusion_set": "", "run": "baseline", "draw": -1, "seed": seed, "n_removed": 0,
                     **outcomes(lab, cl, truth_all, base[seed][0])})
    rng = np.random.default_rng(s.SEED)
    for es, flagged in masks.items():
        keep = ~flagged
        t0 = time.time()
        obs = pipe.run(keep, OBS_SEEDS)
        for seed in OBS_SEEDS:
            cl, lab = obs[seed]
            rows.append({"exclusion_set": es, "run": "observed", "draw": -1, "seed": seed, "n_removed": int(flagged.sum()),
                         **outcomes(lab, cl, truth_all[keep], base[seed][0][keep])})
        for d in range(args.n_draws):
            removed = sens.stratified_draw(rng, truth_all, flagged)
            keep_r = ~removed
            cl, lab = pipe.run(keep_r, (NULL_SEED,))[NULL_SEED]
            rows.append({"exclusion_set": es, "run": "random", "draw": d, "seed": NULL_SEED, "n_removed": int(removed.sum()),
                         **outcomes(lab, cl, truth_all[keep_r], base[NULL_SEED][0][keep_r])})
            if (d + 1) % 20 == 0:
                print(f"[kidney] {es}: {d + 1}/{args.n_draws} random draws ({time.time() - t0:.0f} s)", flush=True)
    runs = pd.DataFrame(rows)

    summ = []
    outs = ["agreement", "n_types", "unknown_fraction", "ari_vs_baseline", *[f"recall_{t}" for t in TYPES]]
    for es in masks:
        rnd = runs[(runs["exclusion_set"] == es) & (runs["run"] == "random")]
        obs = runs[(runs["exclusion_set"] == es) & (runs["run"] == "observed") & (runs["seed"] == NULL_SEED)].iloc[0]
        b = runs[(runs["run"] == "baseline") & (runs["seed"] == NULL_SEED)].iloc[0]
        for o in outs:
            v = rnd[o].to_numpy(float)
            v = v[np.isfinite(v)]
            x = float(obs[o])
            lo, hi = np.quantile(v, [0.025, 0.975]) if len(v) else (np.nan, np.nan)
            summ.append({"exclusion_set": es, "outcome": o, "n_removed": int(obs["n_removed"]), "baseline": float(b[o]),
                         "observed": x, "random_n": len(v), "random_median": float(np.median(v)) if len(v) else np.nan,
                         "random_q025": lo, "random_q975": hi,
                         "observed_percentile_in_random": float((v < x).mean() + 0.5 * (v == x).mean()) if len(v) else np.nan,
                         "inside_central_95": bool(lo <= x <= hi) if len(v) else False,
                         "position": "inside" if lo <= x <= hi else ("above" if x > hi else "below")})
    summ = pd.DataFrame(summ)
    obs_rows = runs[runs["run"] == "observed"]
    for es in masks:
        o = obs_rows[obs_rows["exclusion_set"] == es]
        checks.add(f"{es}: observed over seeds 0-3, 42: agreement; types", f"{o['agreement'].min():.3f}-"
                   f"{o['agreement'].max():.3f}; {o['n_types'].min()}-{o['n_types'].max()}", None)
        sub = summ[summ["exclusion_set"] == es]
        checks.add(f"{es}: outcomes inside the central 95% of random removals (seed 42)",
                   f"{int(sub['inside_central_95'].sum())}/{len(sub)}; outside: "
                   + (", ".join(f"{r.outcome} ({r.position})" for r in sub[~sub['inside_central_95']].itertuples()) or "none"),
                   None)
    checks.add("igraph int32 random-number warnings counted (not printed; as qc_mito)", f"{sens._IGNORED['igraph_int32_randint']}",
               None)

    run.table(checks.frame(), "sensitivity_checks")
    run.table(pd.DataFrame(comp), "sensitivity_exclusion_sets")
    run.table(runs, "sensitivity_runs")
    run.table(summ, "sensitivity_vs_random")
    figure(run, runs)
    run.finish({"label_column": COL, "exclusion_sets": {k: int(m.sum()) for k, m in masks.items()},
                "pipeline": "kidney_annotation option (a) rebuilt on the remaining cells", "resolution": RESOLUTION,
                "observed_seeds": list(OBS_SEEDS), "null_seed": NULL_SEED, "n_draws": args.n_draws,
                "stratified_by": COL, "rng": "numpy default_rng(42)",
                "scanpy_compat_patched": SCANPY_COMPAT_PATCHED, "runtime_s": round(time.time() - t_start, 1),
                "checks_pass": checks.ok})


if __name__ == "__main__":
    main()
