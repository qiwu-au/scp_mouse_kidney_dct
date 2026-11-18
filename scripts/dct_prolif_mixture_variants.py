#!/usr/bin/env python
"""dct_prolif information only: two variants of the route-A DCT1/DCT2 mixture, beside the plan's fit.

Analysis plan
(frozen), "Fixed settings": two-component Gaussian mixture on axis A (n_init 10, random_state 42), posterior >= 0.8
-> DCT1 (lower-mean component) or DCT2, else "DCT, subtype not resolved". The authors asked on 3 Nov 2026, after the
breakdown of the cells without a subtype call ("Can you try these two methods then?"), for:
  - equal_variance: the same mixture with one shared variance (sklearn covariance_type 'tied'), posterior >= 0.8;
  - posterior_0.5: the plan's fit (unequal variances) with the larger posterior deciding (>= 0.5; an exact tie
    stays "DCT, subtype not resolved").
'plan' (unequal variances, >= 0.8) is refitted here as the reference and must reproduce cs_t31_identity /
cs_t31nf_identity exactly. Both runs (frozen floor 1,000; no floor) and both conditions, each condition alone.
Information only: the cs_t31_* / cs_t31nf_* labels and the canonical table are unchanged; dct_labels labels stay primary.

Everything else is fixed as in route A: the axis values are those of dct_prolif_route_a_per_cell.csv (recomputed
from the dct_intake object and checked); cycling and ProLIF calls are unchanged; 'Unknown (markers not detected)' and
'Below depth floor' cells are unchanged. Noise test as the acceptance step: 20 repeats of N(0, (1e-6 x
intensity)^2) on detected values (seed 42, same draws), identity refitted per repeat.

Outputs (results/tables/dct_prolif/):
  dct_prolif_mixvar_per_cell.csv: run, condition, variant, cell, axis, posteriors, identity, label, plan identity and
    label, noise repeats kept;
  dct_prolif_mixvar_summary.csv: per run x condition x variant: fit (means, SDs, weights, BIC), call zones on the
    observed axis range, identity and label counts, cells changed vs plan, cells below the DCT1 component mean not
    called DCT1, called cells with posterior < 0.8, noise result;
  dct_prolif_mixvar_crosstabs.csv: plan identity x variant identity;
  dct_prolif_mixvar_identity_odds.csv: DCT2 vs DCT1 ~ z(ln n_detected) + plate, per identity and label column;
  dct_prolif_mixvar_checks.csv (checks plus one format row per figure); dct_prolif_mixvar_figure_captions.md.
Figures (results/figures/dct_prolif/; dotplot format, 85 mm): dct_prolif_mixvar_<run>_<cond>_<variant>_85mm:
  axis A histogram, the variant's two components and its call zones shaded.

Command (project root):  python scripts/dct_prolif_mixture_variants.py
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
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import scp_common as cc
import cluster_de_lib as t5
import dotplot_dotplots as t6d
import dct_reviewer_lib as t28
import dct_prolif_lib as lib

SCRIPT = "dct_prolif_mixture_variants.py"
REQUEST = {
    "date": "2026-11-03",
    "request": "authors, dct_prolif session, after the breakdown of the cells without a subtype call: 'Can you try these two methods then?' "
               "(equal-variance mixture; lower posterior threshold 0.5)",
    "status": "information only; cs_t31_* and cs_t31nf_* labels unchanged; dct_labels labels stay primary",
}


@dataclass(frozen=True)
class Variant:
    key: str
    covariance: str      # sklearn GaussianMixture covariance_type
    threshold: float     # posterior needed for a call; 0.5 = the larger posterior decides
    display: str
    title: str


VARIANTS = {
    "plan": Variant("plan", "full", lib.POSTERIOR_MIN, "plan: unequal variances, posterior >= 0.8",
                     "unequal SD, ≥ 0.8 (plan)"),
    "equal_variance": Variant("equal_variance", "tied", lib.POSTERIOR_MIN, "equal variances (tied), posterior >= 0.8",
                              "equal SD, ≥ 0.8"),
    "posterior_0.5": Variant("posterior_0.5", "full", 0.5, "plan fit (unequal variances), larger posterior decides",
                             "unequal SD, ≥ 0.5"),
}
RUN_TITLE = {"frozen": "floor 1,000", "nofloor": "no floor"}
ZONE_ALPHA = 0.16
W85 = t28.HALF_MM


# ---------------------------------------------------------------------------
# Mixture variants
# ---------------------------------------------------------------------------
def fit(axis_defined: np.ndarray, covariance: str):
    from sklearn.mixture import GaussianMixture

    g = GaussianMixture(n_components=2, covariance_type=covariance, n_init=lib.GMM_N_INIT,
                        random_state=lib.SEED).fit(axis_defined.reshape(-1, 1))
    return g, int(np.argmin(g.means_.ravel()))


def calls_from(p1: np.ndarray, p2: np.ndarray, threshold: float) -> np.ndarray:
    if threshold > 0.5:
        return np.where(p1 >= threshold, lib.DCT1, np.where(p2 >= threshold, lib.DCT2, lib.UNRESOLVED))
    return np.where(p1 > p2, lib.DCT1, np.where(p2 > p1, lib.DCT2, lib.UNRESOLVED))


def posteriors(g, lo: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    post = g.predict_proba(x.reshape(-1, 1))
    return post[:, lo], post[:, 1 - lo]


def params(g, lo: int, axis_defined: np.ndarray) -> dict:
    var = g.covariances_.ravel()
    var = np.repeat(var, 2) if len(var) == 1 else var
    means, sd, w = g.means_.ravel(), np.sqrt(var), g.weights_
    return {"n_fit": int(len(axis_defined)), "mean_dct1": float(means[lo]), "mean_dct2": float(means[1 - lo]),
            "sd_dct1": float(sd[lo]), "sd_dct2": float(sd[1 - lo]), "weight_dct1": float(w[lo]),
            "weight_dct2": float(w[1 - lo]), "bic_2": float(g.bic(axis_defined.reshape(-1, 1))),
            "converged": bool(g.converged_)}


def call_zones(g, lo: int, threshold: float, lo_x: float, hi_x: float) -> list[tuple[str, float, float]]:
    """Calls along the observed axis range (grid of 20,001 points); boundaries at grid midpoints."""
    grid = np.linspace(lo_x, hi_x, 20001)
    calls = calls_from(*posteriors(g, lo, grid), threshold)
    cut = np.flatnonzero(calls[1:] != calls[:-1])
    edges = [lo_x, *((grid[cut] + grid[cut + 1]) / 2), hi_x]
    starts = [0, *(cut + 1)]
    return [(str(calls[s]), float(edges[i]), float(edges[i + 1])) for i, s in enumerate(starts)]


def zones_text(zones: list[tuple[str, float, float]]) -> str:
    return "; ".join(f"{c} {a:.2f} to {b:.2f}" for c, a, b in zones)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def zone_figure(axis: np.ndarray, p: dict, zones, counts: dict, title: str, note_text: str):
    W, H = W85, 94.0
    fig = t28.new_figure(W, H)
    ax = fig.add_axes([13.0 / W, 38.0 / H, 68.0 / W, 47.0 / H])
    zone_color = {lib.DCT1: lib.LABEL_COLORS[lib.DCT1], lib.DCT2: lib.LABEL_COLORS[lib.DCT2],
                  lib.UNRESOLVED: lib.LABEL_COLORS[lib.UNRESOLVED]}
    for c, a, b in zones:
        ax.axvspan(a, b, color=zone_color[c], alpha=ZONE_ALPHA, lw=0, zorder=0)
    ax.hist(axis, bins=40, density=True, color="#b5b4ad", edgecolor="white", lw=0.3, zorder=1)
    grid = np.linspace(axis.min(), axis.max(), 400)

    def comp(k):
        m, s, w = p[f"mean_{k}"], p[f"sd_{k}"], p[f"weight_{k}"]
        return w * np.exp(-0.5 * ((grid - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    ax.plot(grid, comp("dct1"), color=lib.LABEL_COLORS[lib.DCT1], lw=0.9, zorder=2)
    ax.plot(grid, comp("dct2"), color=lib.LABEL_COLORS[lib.DCT2], lw=0.9, zorder=2)
    ax.set_xlim(axis.min(), axis.max())
    ax.set_xlabel("Axis A (DCT2 score minus DCT1 score)")
    ax.set_ylabel("Density")
    ax.set_title(title, loc="left", fontsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], color=lib.LABEL_COLORS[lib.DCT1], lw=0.9, label="DCT1 component"),
               Line2D([], [], color=lib.LABEL_COLORS[lib.DCT2], lw=0.9, label="DCT2 component")]
    handles += [Patch(facecolor=zone_color[c], alpha=ZONE_ALPHA * 2.5, lw=0, label=f"{c} call ({counts.get(c, 0)})")
                for c in (lib.DCT1, lib.UNRESOLVED, lib.DCT2)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(13.0 / W, 27.0 / H), ncol=1, frameon=False,
               handlelength=1.4, handletextpad=0.4, columnspacing=1.2, borderaxespad=0, fontsize=7, labelspacing=0.3)
    fig.text(1.5 / W, 1.5 / H, note_text, fontsize=7, color=t6d.INK["secondary"], ha="left", va="bottom")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="dct_prolif information only: mixture variants for the DCT1/DCT2 call.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "mixture_variants", args)
    checks = t5.Checks("dct_combined")
    lib.freeze_or_stop(run, checks, "mixvar_checks")

    per_a = lib.read_table(run, "route_a_per_cell", dtype={"cell_id": str, "cell_key": str})
    mixture = lib.read_table(run, "mixture")
    noise_ref = lib.read_table(run, "noise_identity_per_cell", dtype={"cell_id": str})
    for b in ("above_floor", "slc12a3_detected"):
        per_a[b] = per_a[b].astype(str).str.lower().eq("true")
    conds = {c: lib.load_condition(run, c) for c in lib.CONDITIONS}
    cands = {c: lib.identity_candidates(run, conds[c].var_names)[0] for c in lib.CONDITIONS}

    cells, summary, ct_rows, odds = [], [], [], []
    figs = t28.Figures(run, "mixture_variants")
    captions = ["# dct_prolif mixture variants: figure captions (draft; observed facts only)", "",
                f"{cc.GENERATED_TAG} ({SCRIPT}). Information only (authors request, 3 Nov 2026); the dct_prolif labels are "
                "unchanged.", ""]
    for run_key, spec in lib.RUNS.items():
        for cond in lib.CONDITIONS:
            t0 = time.time()
            tag = f"{run_key} / {cond}"
            c = conds[cond]
            f = per_a[(per_a["run"] == run_key) & (per_a["condition"] == cond)].reset_index(drop=True)
            ok = list(f["cell_id"]) == list(c.cell_ids)
            checks.add(f"{tag}: route-A table cells in dct_intake object order", f"{len(f)} / {len(c.cell_ids)}", ok)
            if not ok:
                lib.stop(run, checks, "mixvar_checks")
            a = f["above_floor"].to_numpy(bool)
            used = {t: [g for g in cands[cond][t] if c.detected[a, c.var_names.get_loc(g)].sum() >= lib.MIN_CELLS_MARKER]
                    for t in (lib.DCT1, lib.DCT2)}
            ident = lib.identity_scores(c.intensity[a], c.detected[a], c.ln_ids[a], c.var_names, used)
            axis_a = ident["axis"].to_numpy(float)
            stored = f.loc[a, "axis"].to_numpy(float)
            diff = float(np.nanmax(np.abs(axis_a - stored)))
            same_nan = np.array_equal(np.isnan(axis_a), np.isnan(stored))
            checks.add(f"{tag}: recomputed axis A equals the route-A table (CSV round trip)",
                       f"max abs diff {diff:.2g}", same_nan and diff <= lib.TIE_TOL, "<= 1e-12")
            defined = np.isfinite(stored)
            x = stored[defined]
            cyc = (f["cycling"] == lib.CYCLING).to_numpy()
            ncc = f["slc12a3_detected"].to_numpy(bool)
            brief_identity = f["identity"].astype(str).to_numpy()
            brief_label = f["label"].astype(str).to_numpy()

            # Fits on the stored axis (unequal: plan and posterior_0.5; tied: equal_variance) ----------------
            models = {cov: fit(x, cov) for cov in ("full", "tied")}
            mrow = mixture[(mixture["run"] == run_key) & (mixture["condition"] == cond)].iloc[0]
            pf = params(*models["full"], x)
            worst = max(abs(pf[k] - float(mrow[k])) for k in ("mean_dct1", "mean_dct2", "sd_dct1", "sd_dct2",
                                                                  "weight_dct1", "weight_dct2", "bic_2"))
            checks.add(f"{tag}: unequal-variance refit equals dct_prolif_mixture.csv", f"max abs diff {worst:.2g}",
                       worst <= 1e-9, "<= 1e-9")

            # Noise repeats, shared by the variants (same draws as the acceptance step) -----------------------
            rng = np.random.default_rng(lib.SEED)
            base, det = c.intensity[a], c.detected[a]
            noise_calls = {k: [] for k in VARIANTS}
            for rep in range(1, lib.NOISE_REPEATS + 1):
                noisy = base + np.where(det, rng.normal(0.0, 1.0, base.shape) * lib.NOISE_REL_SD * base, 0.0)
                ax_n = lib.identity_scores(noisy, det, c.ln_ids[a], c.var_names, used)["axis"].to_numpy(float)
                ok_n = np.isfinite(ax_n)
                fits_n = {cov: fit(ax_n[ok_n], cov) for cov in ("full", "tied")}
                for k, v in VARIANTS.items():
                    g, lo = fits_n[v.covariance]
                    call = np.full(len(ax_n), lib.UNKNOWN_MARKERS, dtype=object)
                    call[ok_n] = calls_from(*posteriors(g, lo, ax_n[ok_n]), v.threshold)
                    noise_calls[k].append(call)

            for k, v in VARIANTS.items():
                g, lo = models[v.covariance]
                p = params(g, lo, x)
                p1, p2 = posteriors(g, lo, x)
                ident_above = np.full(int(a.sum()), lib.UNKNOWN_MARKERS, dtype=object)
                ident_above[defined] = calls_from(p1, p2, v.threshold)
                identity = np.full(len(f), lib.BELOW, dtype=object)
                identity[a] = ident_above
                label = identity.copy()
                label[cyc & ncc] = lib.PROLIF
                label[cyc & ~ncc] = lib.CYCLING_NO_NCC
                kept = np.sum([nc == ident_above for nc in noise_calls[k]], axis=0)
                n_changed_rep = [int((nc != ident_above).sum()) for nc in noise_calls[k]]
                post1 = np.full(len(f), np.nan)
                post2 = np.full(len(f), np.nan)
                idx = np.flatnonzero(a)[defined]
                post1[idx], post2[idx] = p1, p2
                noise_kept = np.full(len(f), -1, int)
                noise_kept[a] = kept
                cells.append(pd.DataFrame({
                    "run": run_key, "condition": cond, "variant": k, "cell_id": f["cell_id"], "cell_key": f["cell_key"],
                    "plate": f["plate"], "n_detected": f["n_detected"], "above_floor": a, "axis": f["axis"],
                    "cycling": f["cycling"], "slc12a3_detected": ncc, "posterior_dct1": post1, "posterior_dct2": post2,
                    "identity": identity, "label": label, "identity_brief": brief_identity, "label_brief": brief_label,
                    "noise_n_repeats_same": noise_kept,
                    "noise_fraction_same": np.where(a, noise_kept / lib.NOISE_REPEATS, np.nan)}))

                # Checks and summary ---------------------------------------------------------------------------
                vt = f"{tag} / {k}"
                checks.add(f"{vt}: mixture converged; DCT1 component has the lower mean",
                           f"means {p['mean_dct1']:.3f}, {p['mean_dct2']:.3f}; SD {p['sd_dct1']:.3f}, {p['sd_dct2']:.3f}",
                           p["converged"] and p["mean_dct1"] < p["mean_dct2"])
                if k == "plan":
                    checks.add(f"{vt}: refit reproduces the stored identity calls", int((identity != brief_identity).sum()),
                               np.array_equal(identity, brief_identity), "0 cells differ")
                    pd2 = float(np.nanmax(np.abs(post2[idx] - f.loc[idx, "posterior_dct2"].to_numpy(float))))
                    checks.add(f"{vt}: refit reproduces the stored DCT2 posterior", f"max abs diff {pd2:.2g}",
                               pd2 <= 1e-9, "<= 1e-9")
                    nref = noise_ref[(noise_ref["run"] == run_key) & (noise_ref["condition"] == cond)]
                    same_noise = (list(nref["cell_id"]) == list(f.loc[a, "cell_id"])
                                  and np.array_equal(nref["n_repeats_same"].to_numpy(int), kept))
                    checks.add(f"{vt}: noise repeats reproduce dct_prolif_noise_identity_per_cell.csv", str(same_noise),
                               same_noise)
                fixed = np.isin(brief_label, (lib.PROLIF, lib.CYCLING_NO_NCC, lib.BELOW, lib.UNKNOWN_MARKERS))
                checks.add(f"{vt}: ProLIF, cycling, Unknown (markers) and below-floor labels unchanged vs plan",
                           int((label[fixed] != brief_label[fixed]).sum()), np.array_equal(label[fixed], brief_label[fixed]),
                           "0 cells differ")
                zones = call_zones(g, lo, v.threshold, float(x.min()), float(x.max()))
                below_mean = a.copy()
                below_mean[a] = np.where(defined, np.nan_to_num(stored, nan=np.inf) < p["mean_dct1"], False)
                called = np.isin(identity, (lib.DCT1, lib.DCT2))
                pmax = np.fmax(post1, post2)
                n_changed = int((identity != brief_identity).sum())
                srow = {"run": run_key, "condition": cond, "floor": spec.floor, "variant": k, "variant_display": v.display,
                        "covariance_type": v.covariance, "posterior_min": v.threshold, **p,
                        "bic_1": float(mrow["bic_1"]), "call_zones_observed_range": zones_text(zones),
                        "n_dct2_zones": sum(z[0] == lib.DCT2 for z in zones),
                        "identity_counts": lib.counts_text(identity, lib.IDENTITY_ORDER),
                        "label_counts": lib.counts_text(label),
                        "n_DCT1_identity": int((identity == lib.DCT1).sum()),
                        "n_DCT2_identity": int((identity == lib.DCT2).sum()),
                        "n_unresolved_identity": int((identity == lib.UNRESOLVED).sum()),
                        "n_identity_changed_vs_brief": n_changed,
                        "n_below_dct1_mean": int(below_mean.sum()),
                        "n_below_dct1_mean_unresolved": int((below_mean & (identity == lib.UNRESOLVED)).sum()),
                        "n_below_dct1_mean_dct2": int((below_mean & (identity == lib.DCT2)).sum()),
                        "n_called": int(called.sum()),
                        "n_called_posterior_below_0.8": int((called & (pmax < lib.POSTERIOR_MIN)).sum()),
                        "n_exact_ties": int((p1 == p2).sum()),
                        "noise_n_kept_all_repeats": int((kept == lib.NOISE_REPEATS).sum()),
                        "noise_n_cells": int(a.sum()), "noise_max_changed_per_repeat": int(max(n_changed_rep))}
                summary.append(srow)
                checks.add(f"{vt}: identity counts; changed vs plan", f"{srow['identity_counts']}; {n_changed}", None)
                checks.add(f"{vt}: call zones on the observed axis range", srow["call_zones_observed_range"], None)
                checks.add(f"{vt}: cells below the DCT1 component mean not called DCT1 (DCT, subtype not resolved; DCT2)",
                           f"{srow['n_below_dct1_mean_unresolved']}; {srow['n_below_dct1_mean_dct2']} of "
                           f"{srow['n_below_dct1_mean']}", None)
                checks.add(f"{vt}: identity kept in all 20 noise repeats",
                           f"{srow['noise_n_kept_all_repeats']}/{srow['noise_n_cells']}; max changed per repeat "
                           f"{srow['noise_max_changed_per_repeat']}", None)
                ct_rows += lib.crosstab_long(brief_identity[a], identity[a], "identity_brief", "identity_variant",
                                             run=run_key, condition=cond, variant=k)
                for column, values in (("identity", identity), ("label", label)):
                    m = a & np.isin(values, (lib.DCT1, lib.DCT2))
                    res = lib.logit_or(values[m] == lib.DCT2, np.log(f.loc[m, "n_detected"].to_numpy(float)),
                                       f.loc[m, "plate"].to_numpy())
                    odds.append({"run": run_key, "condition": cond, "variant": k, "column": column,
                                 "model": "DCT2 ~ z(ln n_detected) + plate", "n_DCT1": int((values[m] == lib.DCT1).sum()),
                                 "n_DCT2": int((values[m] == lib.DCT2).sum()), **res})

                # Figure ---------------------------------------------------------------------------------------
                t28.style()
                counts = {lab: int((ident_above == lab).sum()) for lab in (lib.DCT1, lib.DCT2, lib.UNRESOLVED)}
                stem = f"mixvar_{run_key}_{lib.KEY[cond]}_{k.replace('.', '')}"
                note_text = (f"{len(x)} cells with both marker types detected.\n"
                             f"{v.display.capitalize()}.")
                fig = zone_figure(x, p, zones, counts, f"{cond}, {RUN_TITLE[run_key]}: {v.title}", note_text)
                figs.save(fig, f"{stem}_85mm", f"Axis A with {v.display}, {cond}, {run_key}", W85)
                plt.close(fig)
                captions += [f"- `dct_prolif_{stem}_85mm`: {cond}, {lib.RUNS[run_key].display}. Axis A (DCT2 minus "
                             f"DCT1 residual-z score) of the {len(x)} cells with both marker types detected; mixture "
                             f"{v.display}; component means {p['mean_dct1']:.2f} and {p['mean_dct2']:.2f}, SDs "
                             f"{p['sd_dct1']:.2f} and {p['sd_dct2']:.2f}, weights {p['weight_dct1']:.2f} and "
                             f"{p['weight_dct2']:.2f}. Shading: call zones ({zones_text(zones)}). Calls: DCT1 "
                             f"{counts[lib.DCT1]}; DCT, subtype not resolved {counts[lib.UNRESOLVED]}; DCT2 {counts[lib.DCT2]}."]
            captions.append("")
            print(f"[{tag}] done in {time.time() - t0:.1f} s", flush=True)

    frame = pd.DataFrame(figs.rows)
    n_fail = int((~frame["pass"]).sum())
    checks.add("figures: Arial, >= 7 pt, no text overlap or text outside, exact width, TrueType in PDF",
               f"{len(frame) - n_fail}/{len(frame)} pass" + (f"; failed: {';'.join(frame.loc[~frame['pass'], 'figure'])}"
                                                             if n_fail else ""), n_fail == 0)
    run.table(pd.concat(cells, ignore_index=True), "mixvar_per_cell")
    run.table(pd.DataFrame(summary), "mixvar_summary")
    run.table(pd.DataFrame(ct_rows), "mixvar_crosstabs")
    run.table(pd.DataFrame(odds), "mixvar_identity_odds")
    run.table(pd.concat([checks.frame(), frame], ignore_index=True), "mixvar_checks")
    run.text("\n".join(captions), "mixvar_figure_captions")
    run.finish({"request": REQUEST, "variants": {k: vars(v) for k, v in VARIANTS.items()},
                "runs": {k: vars(v) for k, v in lib.RUNS.items()}, "gmm_n_init": lib.GMM_N_INIT, "seed": lib.SEED,
                "noise": {"repeats": lib.NOISE_REPEATS, "relative_sd": lib.NOISE_REL_SD, "seed": lib.SEED},
                "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see dct_prolif_mixvar_checks.csv.")


if __name__ == "__main__":
    main()
