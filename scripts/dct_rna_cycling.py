#!/usr/bin/env python
"""dct_rna step 6: cycling fractions in the RNA atlas (atlas_dct, reused) next to the SCP fractions (dct_reviewer), per condition.

Analysis plan,
step 6 (port of ctrl_rna_core_cycling.py with a condition argument; ``MEAS_STYLE`` of atlas_dct_cycling imported
unchanged).

RNA (reused from atlas_dct, not rescored; condition 'both')
-----------------------------------------------------
- atlas_dct_cycling_rna_fractions.csv and atlas_dct_cycling_rna_score_thresholds.csv: Mki67 counts > 0, Top2a
  counts > 0, either, and the Seurat-style S / G2M phase per atlas group, Source, Technology, Origin and UMI-depth bin.
  Check (as ctrl_rna_core): the Mki67 / Top2a counts per published group and in the atlas_dct-derived split are recomputed from the
  region subset and equal the atlas_dct table.

SCP (from dct_reviewer, not rescored; fractions and Wilson 95% intervals computed here), per condition
-------------------------------------------------------------------------------------------
- dct_reviewer_reannot_per_cell.csv: Mki67 detected, Top2a detected, either, cycling-score rule (cell_cycle_score >
  90th percentile within condition, so 10% of the condition's cells by construction).
- dct_reviewer_task4_cell_cycle_scores_per_cell.csv: protein Tirosh S_score / G2M_score -> the same phase rule (G1 if
  both < 0).
- Populations: all cells, plates, the condition's protein-ID quartiles (CSV only), each label under the primary and
  second columns (absent labels: 'does not apply' rows), ProLIF label share per column, and the depth-matched ProLIF
  vs DCT1 + DCT2 rows (dct_reviewer_depth_matched_comparisons.csv, nn_matched_within_plate, caliper 0.05 ln(IDs);
  second columns only, the primary rows are 'does not apply' as in dct_reviewer).
- No test between conditions (zerok_vs_control). ZeroK rows are descriptive (atlas mice on a normal diet, another study).

Inputs (read-only): atlas_dct region subset, split table and the two atlas_dct cycling tables; dct_labels labelled objects and
                    canonical labels; dct_intake filtering summary; the dct_reviewer tables above and
                    dct_reviewer_task4_cell_cycle_gene_coverage.csv.
Outputs: results/tables/dct_rna/dct_rna_cycling_scp_fractions.csv, dct_rna_cycling_rna_fractions.csv,
         dct_rna_cycling_rna_score_thresholds.csv (atlas_dct rows used), dct_rna_cycling_side_by_side.csv,
         dct_rna_cycling_checks.csv, dct_rna_cycling_figure_checks.csv;
         results/figures/dct_rna/dct_rna_cycling (.pdf/.png/.tif)

Command (project root):  python scripts/dct_rna_cycling.py
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
import kidney_rna_lib as L7
import atlas_dct_lib as T
import dct_rna_lib as L
from atlas_dct_cycling import MEAS_STYLE

N_RULE = {"Ctrl": 73, "ZeroK": 71}                                    # dct_reviewer section 4 (10% within condition)
N_DETECTED = {"Ctrl": (15, 16, 21), "ZeroK": (52, 35, 56)}            # dct_reviewer reannot_per_cell: Mki67 / Top2a / either
NN_METHOD = "nn_matched_within_plate"
DM_METRICS = ("Mki67_detected", "Top2a_detected", "Mki67_or_Top2a_detected")
SCP_MEAS = ("Mki67", "Top2a", "Mki67_or_Top2a", "S_or_G2M", "cycling_score_rule")
CONTROLS = "DCT1 + DCT2 controls, depth-matched"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dct_rna step 6: cycling in RNA (atlas_dct) vs SCP (dct_reviewer), per condition.")
    cc.add_common_args(parser)
    return parser.parse_args()


def rna_tables(run, checks: L.Checks) -> tuple[pd.DataFrame, pd.DataFrame]:
    """atlas_dct RNA cycling tables, with the ctrl_rna_core recount of Mki67 / Top2a from the region subset."""
    rna = L.read_table(run, "cycling_rna_fractions", L.ATLAS_DCT_TASK)
    thr = L.read_table(run, "cycling_rna_score_thresholds", L.ATLAS_DCT_TASK)
    region = T.load_region(run, with_split=True)
    look = L7.atlas_gene_lookup(region.var_names)
    genes = region.var_names.astype(str)
    det = {g: np.asarray((region.X[:, genes.get_loc(look[g.lower()])] > 0).todense()).ravel() for g in ("Mki67", "Top2a")}
    det["Mki67_or_Top2a"] = det["Mki67"] | det["Top2a"]
    n_ok, n_cmp = True, 0
    for gcol, gs in (("rna_group", "published (atlas Celltype)"), ("rna_group_derived", "atlas_dct-derived split of atlas DCT")):
        grp = region.obs[gcol].astype(str).to_numpy()
        base = np.ones(len(grp), bool) if gcol == "rna_group" else (region.obs["rna_group"] == "DCT").to_numpy()
        for g in np.unique(grp[base]):
            m = base & (grp == g)
            for meas, v in det.items():
                row = rna[(rna["group_set"] == gs) & (rna["group"] == g) & (rna["by"] == "all") & (rna["measure"] == meas)]
                n_cmp += 1
                n_ok &= len(row) == 1 and int(row["k"].iloc[0]) == int(v[m].sum()) and int(row["n_cells"].iloc[0]) == int(m.sum())
    checks.add("atlas_dct RNA cycling table: Mki67 / Top2a / either counts per group recomputed from the region subset",
               f"{n_cmp} rows; equal {n_ok}", bool(n_ok))
    dct = rna[(rna["group_set"] == "published (atlas Celltype)") & (rna["group"] == "DCT") & (rna["by"] == "all")]
    dct = dct.set_index("measure")
    checks.add("atlas DCT cells (atlas_dct): Mki67 > 0; Top2a > 0; either; S or G2M (pooled; within batch)",
               "; ".join(f"{int(dct.loc[m, 'k'])}/{int(dct.loc[m, 'n_cells'])} ({dct.loc[m, 'fraction']:.3f})"
                         for m in dct.index), None)
    return rna, thr


def scp_cells(run, cond: str, checks: L.Checks) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell SCP cycling calls of one condition from dct_reviewer (port of ctrl_rna_core_cycling.scp_tables)."""
    s = L.SETTINGS[cond]
    adata, labels, _, _ = L.load_scp(run, cond, checks)
    del adata
    edges = L.depth_quartiles(run, cond, labels, checks)
    rp = L.t28_per_cell(run, "reannot_per_cell", cond, labels, checks)
    sc4 = L.t28_per_cell(run, "task4_cell_cycle_scores_per_cell", cond, labels, checks)
    same_depth = sc4["n_detected_proteins"].astype(int).equals(labels["n_detected_proteins"].astype(int))
    checks.add("dct_reviewer cell-cycle scores: n_detected_proteins equal the dct_labels canonical table", f"{same_depth}", same_depth,
               "", cond)
    d = pd.DataFrame(index=labels.index)
    d["Mki67"] = rp["Mki67_detected"].astype(str).eq("True")
    d["Top2a"] = rp["Top2a_detected"].astype(str).eq("True")
    d["Mki67_or_Top2a"] = rp["mki67_or_top2a_detected"].astype(str).eq("True")
    d["cycling_score_rule"] = rp["reviewer_cycling_status"].astype(str).eq("Cycling")
    d["S_or_G2M"] = ~((sc4["S_score"].astype(float) < 0) & (sc4["G2M_score"].astype(float) < 0))
    d["plate"] = labels["plate"].astype(str)
    d["protein_id_quartile"] = L.quartile_bins(labels["n_detected_proteins"], edges)
    for col in s.LABEL_COLUMNS:
        d[col] = labels[col].astype(str)
    return d, labels


def frac_row(pop: str, kind: str, meas: str, text: str, k: int, n: int, source: str, **extra) -> dict:
    lo, hi = T.wilson(k, n)
    return {"population": pop, "population_kind": kind, "measure": meas, "measure_text": text, "k": k, "n_cells": n,
            "fraction": k / n, "ci_low": lo, "ci_high": hi, "status": "computed", "reason": "", "source": source, **extra}


def na_row(pop: str, kind: str, meas: str, text: str, reason: str, source: str) -> dict:
    return {"population": pop, "population_kind": kind, "measure": meas, "measure_text": text, "k": np.nan,
            "n_cells": np.nan, "fraction": np.nan, "ci_low": np.nan, "ci_high": np.nan, "status": L.NOT_APPLICABLE,
            "reason": reason, "source": source}


def one_condition(run, cond: str, checks: L.Checks) -> tuple[pd.DataFrame, dict]:
    s = L.SETTINGS[cond]
    scp, labels = scp_cells(run, cond, checks)
    n_all = len(scp)
    n_rule = int(scp["cycling_score_rule"].sum())
    checks.add(f"SCP cycling-score rule: cells (10% of {n_all} by construction; dct_reviewer section 4)", f"{n_rule}",
               n_rule == N_RULE[cond], f"{N_RULE[cond]}", cond)
    nd = tuple(int(scp[m].sum()) for m in ("Mki67", "Top2a", "Mki67_or_Top2a"))
    checks.add("SCP Mki67 / Top2a / either detected equal dct_reviewer", " / ".join(map(str, nd)), nd == N_DETECTED[cond],
               " / ".join(map(str, N_DETECTED[cond])), cond)
    cov = L.read_table(run, "task4_cell_cycle_gene_coverage", L.DCT_REVIEWER_TASK)
    cov = cov[cov["condition"] == cond]
    n_s, t_s = int(cov[cov["set"] == "S"]["present"].astype(bool).sum()), int((cov["set"] == "S").sum())
    n_g, t_g = int(cov[cov["set"] == "G2M"]["present"].astype(bool).sum()), int((cov["set"] == "G2M").sum())
    checks.add("protein Tirosh genes quantified (dct_reviewer): S; G2M", f"{n_s} of {t_s}; {n_g} of {t_g}", None, "", cond)
    meas_text = {"Mki67": "Mki67 detected", "Top2a": "Top2a detected", "Mki67_or_Top2a": "Mki67 or Top2a detected",
                 "S_or_G2M": f"S or G2M phase (protein Tirosh scores; S {n_s}/{t_s}, G2M {n_g}/{t_g})",
                 "cycling_score_rule": "cycling-score rule (> 90th percentile within condition)"}
    src = "dct_reviewer per-cell tables"
    rows = []
    pops = ([("all cells", np.ones(n_all, bool), "all", "")]
            + [(f"plate {p}", (scp["plate"] == p).to_numpy(), "plate", "") for p in ("P1", "P2")]
            + [(f"protein IDs {q}", (scp["protein_id_quartile"] == q).to_numpy(), "protein-ID quartile", "")
               for q in L.QUARTILE_LABELS])
    for pop, m, kind, _ in pops:
        for meas in SCP_MEAS:
            rows.append(frac_row(pop, kind, meas, meas_text[meas], int(scp.loc[m, meas].sum()), int(m.sum()), src,
                                 label_column=""))
    for col in s.LABEL_COLUMNS:
        r = L.role_of(cond, col)
        for g in L.SCP_GROUPS:
            m = (scp[col] == g).to_numpy()
            for meas in SCP_MEAS:
                if m.any():
                    rows.append(frac_row(f"{g} ({r})", col, meas, meas_text[meas], int(scp.loc[m, meas].sum()),
                                         int(m.sum()), src, label_column=col))
                else:
                    rows.append({**na_row(f"{g} ({r})", col, meas, meas_text[meas],
                                          f"no {g} cell under this column (dct_labels)", src), "label_column": col})
    for col in s.LABEL_COLUMNS:
        r = L.role_of(cond, col)
        k = int((scp[col] == "ProLIF").sum())
        rows.append(frac_row(f"all cells ({r})", col, "ProLIF_share", f"ProLIF label share, {col}", k, n_all,
                             "dct_labels canonical labels", label_column=col))
    dm = L.read_table(run, "depth_matched_comparisons", L.DCT_REVIEWER_TASK)
    dm = dm[dm["condition"] == cond]
    for col in s.LABEL_COLUMNS:
        r = L.role_of(cond, col)
        sub = dm[dm["label_column"] == col]
        dsrc = f"dct_reviewer depth_matched_comparisons ({NN_METHOD}, caliper 0.05)"
        if sub["status"].eq(L.NOT_APPLICABLE).all():
            reason = "; ".join(sub["reason"].astype(str).unique()) or "no ProLIF cell under this column"
            checks.add(f"dct_reviewer depth-matched rows for {col} ({r}): 'does not apply'", f"{len(sub)} row(s); {reason}",
                       len(sub) == 1, "1 row", cond)
            for who in ("ProLIF, depth-matched", CONTROLS):
                for metric in DM_METRICS:
                    meas = metric.replace("_detected", "")
                    rows.append({**na_row(f"{who} ({r})", col, meas, meas_text[meas], f"{reason} (dct_reviewer B.3)", dsrc),
                                 "label_column": col})
            continue
        sub = sub[sub["method"].eq(NN_METHOD) & sub["metric"].isin(DM_METRICS)]
        checks.add(f"dct_reviewer depth-matched rows ({NN_METHOD}; Mki67, Top2a, either) for {col} ({r})", f"{len(sub)}",
                   len(sub) == 3, "3", cond)
        for q in sub.itertuples():
            meas = q.metric.replace("_detected", "")
            for who, val, n in (("ProLIF, depth-matched", q.value_prolif, q.n_prolif),
                                (CONTROLS, q.value_comparator, q.n_comparator)):
                rows.append(frac_row(f"{who} ({r})", col, meas, meas_text[meas], int(round(val * n)), int(n), dsrc,
                                     label_column=col, mcnemar_p=q.p_value, p_fdr_bh_t28=q.p_fdr_bh))
    out = pd.DataFrame(rows)
    a = out[out["population"] == "all cells"].set_index("measure")
    checks.add(f"SCP all {n_all}: Mki67; Top2a; either; protein S or G2M; cycling-score rule",
               "; ".join(f"{int(a.loc[m, 'k'])} ({a.loc[m, 'fraction']:.3f})" for m in SCP_MEAS), None, "", cond)
    for col in s.LABEL_COLUMNS:
        r = L.role_of(cond, col)
        share = out[(out["population"] == f"all cells ({r})") & (out["measure"] == "ProLIF_share")].iloc[0]
        pr = out[out["population"] == f"ProLIF ({r})"].set_index("measure")
        if pr["status"].eq(L.NOT_APPLICABLE).all():
            txt = "ProLIF rows: does not apply"
        else:
            txt = "; ".join(f"{int(pr.loc[m, 'k'])}/{int(pr.loc[m, 'n_cells'])}"
                            for m in ("Mki67", "Top2a", "Mki67_or_Top2a", "cycling_score_rule"))
        checks.add(f"{col} ({r}): ProLIF share; ProLIF cells with Mki67; Top2a; either; cycling-score rule",
                   f"{int(share['k'])}/{int(share['n_cells'])} ({share['fraction']:.3f}); {txt}", None, "", cond)
    return out, {"n_s": n_s, "t_s": t_s, "n_g": n_g, "t_g": t_g, "n_cells": n_all}


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "cycling", args)
    L.ported_from(run, "ctrl_rna_core_cycling.py", "atlas_dct_cycling.py")
    checks = L.Checks(L.DATASET)
    rna, thr = rna_tables(run, checks)
    res = {c: one_condition(run, c, checks) for c in L.CONDITIONS}
    scp = L.stack({c: res[c][0] for c in L.CONDITIONS})
    cov = {c: res[c][1] for c in L.CONDITIONS}
    L.stop_if_failed(run, checks, "cycling_checks")

    # side-by-side summary (RNA DCT-region rows, condition 'both'; SCP populations per condition) ---------------------
    side = []
    rsel = rna[(rna["by"].isin(["all", "Source"])) & rna["measure"].isin(["Mki67", "Top2a", "Mki67_or_Top2a",
                                                                          "S_or_G2M"])]
    for r in rsel.itertuples():
        side.append({"condition": "both", "modality": "RNA (Mouse Kidney Atlas; atlas_dct)", "population": r.group
                     if r.by == "all" else f"{r.group}, {r.value}", "label_role": "", "measure": r.measure, "k": r.k,
                     "n_cells": r.n_cells, "fraction": r.fraction, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "status": "computed"})
    for r in scp.itertuples():
        side.append({"condition": r.condition, "modality": f"SCP-MS {r.condition} DCT (combined search; dct_reviewer)",
                     "population": r.population, "label_role": r.label_role, "measure": r.measure, "k": r.k,
                     "n_cells": r.n_cells, "fraction": r.fraction, "ci_low": r.ci_low, "ci_high": r.ci_high,
                     "status": r.status})
    side = pd.DataFrame(side)
    rna_out = rna.copy()
    rna_out.insert(0, "condition", "both")
    thr_out = thr.copy()
    thr_out.insert(0, "condition", "both")

    fig_rows = []
    figure(run, rna, scp, fig_rows, cov)
    figs = L.figure_rows_frame(fig_rows, "cycling")
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(scp, "cycling_scp_fractions")
    run.table(rna_out, "cycling_rna_fractions")
    run.table(thr_out, "cycling_rna_score_thresholds")
    run.table(side, "cycling_side_by_side")
    run.table(figs, "cycling_figure_checks")
    run.table(checks.frame(), "cycling_checks")
    run.finish({"rna": "atlas_dct tables reused unchanged (atlas_dct_cycling.py)", "scp": "dct_reviewer per-cell tables, per condition",
                "depth_matched": f"dct_reviewer {NN_METHOD} (second columns; primary 'does not apply')",
                "protein_id_quartiles": "per condition, dct_intake filtering summary edges", "n_rule": N_RULE,
                "n_detected_mki67_top2a_either": {c: list(v) for c, v in N_DETECTED.items()},
                "tirosh_coverage": cov, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see dct_rna_cycling_checks.csv.")


def scp_rows(cond: str) -> list[tuple[str, str]]:
    """(population, display label) of one condition's SCP block, in figure order."""
    rows = [("all cells", "all cells"), ("plate P1", "plate P1"), ("plate P2", "plate P2")]
    for r in L.ROLES:
        rows += [(f"{g} ({r})", f"{g} ({r})") for g in L.SCP_GROUPS]
    for r in L.ROLES:
        rows += [(f"ProLIF, depth-matched ({r})", f"ProLIF, depth-matched ({r})"),
                 (f"{CONTROLS} ({r})", f"controls, depth-matched ({r})")]
    rows += [(f"all cells ({r})", f"ProLIF share ({r})") for r in L.ROLES]
    return rows


def figure(run, rna: pd.DataFrame, scp: pd.DataFrame, fig_rows: list[dict], cov: dict) -> None:
    """ctrl_rna_core_cycling.figure with a Ctrl block and a ZeroK block on the SCP side ('does not apply' rows grey)."""
    T.setup_matplotlib()
    LH = T.text_mm("Hg")[1]
    pub, der = "published (atlas Celltype)", "atlas_dct-derived split of atlas DCT"
    rna_rows = [(pub, "DCT", "all", "all", "DCT, all"), (pub, "DCT", "Source", "Nuclei", "DCT, nuclei"),
                (pub, "DCT", "Source", "Cell", "DCT, cells"), (der, T.DCT1_ATLAS_DCT, "all", "all", T.DCT1_ATLAS_DCT),
                (der, T.DCT2_ATLAS_DCT, "all", "all", T.DCT2_ATLAS_DCT), (der, T.DCT_UNASSIGNED, "all", "all", T.DCT_UNASSIGNED),
                (pub, "DCT-CNT", "all", "all", "DCT-CNT"), (pub, "CNT", "all", "all", "CNT"),
                (pub, "CTAL", "all", "all", "CTAL"), (pub, "MTAL", "all", "all", "MTAL")]
    blocks = {c: scp_rows(c) for c in L.CONDITIONS}
    title = L.fit_title("Cycling fractions: RNA atlas (left) and SCP-MS DCT data per condition (right)")
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    rh = 4.0
    hh = LH + 2.5                                                   # block header height
    lw1 = max(T.text_mm(r[4])[0] for r in rna_rows) + 1.5
    lw2 = max(T.text_mm(d)[0] for rows in blocks.values() for _, d in rows) + 1.5
    pw = (T.W - 2 * T.M - lw1 - lw2 - 8.0) / 2
    nscp = sum(len(v) for v in blocks.values())
    gap = 9.0                                                       # between condition blocks (x-axis labels)
    scp_h = len(blocks) * hh + nscp * rh + (len(blocks) - 1) * gap
    leg_h = LH + 2.0
    cov_txt = "; ".join(f"{c} S {cov[c]['n_s']}/{cov[c]['t_s']}, G2M {cov[c]['n_g']}/{cov[c]['t_g']}"
                        for c in L.CONDITIONS)
    note = ("Fractions with Wilson 95% intervals; log scale (0 shown at 0.001, open symbol). RNA (atlas_dct tables, reused): "
            "atlas Celltype cells; S or G2M phase: scanpy score_genes_cell_cycle with the Tirosh lists, G1 if both scores "
            "< 0 (Seurat rule). " + L.ATLAS_DCT_SPLIT_CAVEAT + " SCP (dct_reviewer tables, combined search): primary = DE-rule labels "
            "(cs_ctrl_deg_label, cs_zerok_pipeline_label), second = submitted cluster labels (cs_ctrl_cluster_label, "
            "cs_zerok_cluster_label); grey rows: the label has no cell under that column (does not apply). The "
            "cycling-score rule marks 10% of each condition's cells by construction; protein S / G2M scores rest on the "
            f"quantified Tirosh proteins ({cov_txt}); depth-matched: nearest neighbour within plate, caliper 0.05 "
            "ln(protein IDs) (dct_reviewer B.3; second columns only). One pool of 3 mice per condition; plates are sort plates. "
            + L.UNPAIRED_NOTE + " " + L.ZEROK_NOTE)
    note_text, note_h = T.note_block(note)
    body_h = max(len(rna_rows) * rh, scp_h)
    H = T.M + title_h + LH + 1.5 + body_h + 9.0 + leg_h + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    y0 = T.M + title_h + LH + 1.5
    x_rna, x_scp = T.M + lw1, T.M + lw1 + pw + 8.0 + lw2
    sh.text(T.M, y0 - LH - 1.5, "RNA (Mouse Kidney Atlas, atlas_dct)", va="top")
    sh.text(x_scp - lw2, y0 - LH - 1.5, "SCP-MS (combined search, dct_reviewer)", va="top")

    def panel(x0, ytop, rows, kind, cond=None):
        ax = sh.ax(x0, ytop, pw, len(rows) * rh)
        ax.set_xscale("log")
        ax.set_xlim(0.0008, 1.2)
        ax.set_ylim(len(rows), 0)
        meas = ["Mki67", "Top2a", "Mki67_or_Top2a", "S_or_G2M"] + (["cycling_score_rule", "ProLIF_share"]
                                                                    if kind == "scp" else [])
        offs = dict(zip(meas, np.linspace(-0.3, 0.3, len(meas))))
        for i, r in enumerate(rows):
            if kind == "rna":
                gs, g, by, val, _ = r
                sub = rna[(rna["group_set"] == gs) & (rna["group"] == g) & (rna["by"] == by)
                          & (rna["value"] == val)].set_index("measure")
            else:
                sub = scp[(scp["condition"] == cond) & (scp["population"] == r[0])].set_index("measure")
                if len(sub) and sub["status"].eq(L.NOT_APPLICABLE).all():
                    ax.axhspan(i, i + 1, color=L.ABSENT_FACE, lw=0)
                    sh.text(x0 + pw / 2, ytop + (i + 0.5) * rh, "does not apply", ha="center",
                            va="center_baseline", color=T.INK["muted"])
                    ax.axhline(i + 1, color=T.INK["grid"], lw=0.3)
                    continue
            for m in meas:
                if m not in sub.index:
                    continue
                f, lo, hi = sub.loc[m, ["fraction", "ci_low", "ci_high"]].astype(float)
                col, mk, _ = MEAS_STYLE[m]
                yy = i + 0.5 + offs[m]
                ax.plot([max(lo, 0.001), max(hi, 0.001)], [yy, yy], color=col, lw=0.6)
                ax.scatter([max(f, 0.001)], [yy], s=7, marker=mk, color=col if f > 0 else "white",
                           edgecolors=col, linewidths=0.6, zorder=3)
            ax.axhline(i + 1, color=T.INK["grid"], lw=0.3)
        ax.set_yticks([])
        ax.set_xticks([0.001, 0.01, 0.1, 1])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
        ax.minorticks_off()
        ax.set_xlabel("Fraction of cells", labelpad=1)
        T.style_axes(ax)
        ax.spines["left"].set_visible(False)
        labs = [r[4] for r in rows] if kind == "rna" else [r[1] for r in rows]
        for i, t_ in enumerate(labs):
            sh.text(x0 - 1.0, ytop + (i + 0.5) * rh, t_, ha="right", va="center_baseline")

    panel(x_rna, y0, rna_rows, "rna")
    yb = y0
    for cond in L.CONDITIONS:
        n = int(scp[(scp["condition"] == cond) & (scp["population"] == "all cells")]["n_cells"].iloc[0])
        sh.text(x_scp - lw2, yb, f"{cond} ({n} cells)", va="top", fontweight="bold")
        panel(x_scp, yb + hh, blocks[cond], "scp", cond)
        yb += hh + len(blocks[cond]) * rh + gap
    ly = y0 + body_h + 9.0
    x = T.M
    for m, (col, mk, txt) in MEAS_STYLE.items():
        ax = sh.ax(x, ly, 2.4, LH)
        ax.scatter([0.5], [0.5], s=9, marker=mk, color=col)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        sh.text(x + 3.0, ly + LH / 2, txt, va="center_baseline")
        x += 3.0 + T.text_mm(txt)[0] + 4.0
    sh.note(note_text, note_h)
    L.save_checked(run, sh, "cycling", title.replace("\n", " "), fig_rows)


if __name__ == "__main__":
    main()
