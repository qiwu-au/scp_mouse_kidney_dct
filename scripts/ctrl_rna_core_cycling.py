#!/usr/bin/env python
"""ctrl_rna_core step 6: cycling fractions in the RNA atlas (atlas_dct, reused) next to the control SCP fractions (ctrl_reviewer_core).

Analysis plan,
step 6 (R1 ProLIF prevalence; port of atlas_dct_cycling.py).

RNA (reused from atlas_dct, not rescored)
-----------------------------------
- atlas_dct_cycling_rna_fractions.csv and atlas_dct_cycling_rna_score_thresholds.csv: Mki67 counts > 0, Top2a
  counts > 0, either, and the Seurat-style S / G2M phase (scanpy score_genes_cell_cycle, Tirosh lists; pooled and
  within Origin x Source) per atlas group, Source, Technology, Origin and UMI-depth bin. Check: the Mki67 / Top2a
  counts per published group and in the atlas_dct-derived split are recomputed from the region subset and equal the atlas_dct
  table.

SCP (from ctrl_reviewer_core, not rescored; fractions and Wilson 95% intervals computed here)
-----------------------------------------------------------------------------
- ctrl_reviewer_core_reannot_per_cell.csv: Mki67 detected, Top2a detected, either, cycling-score rule (cell_cycle_score >
  90th percentile, so 10% of the 727 cells by construction).
- ctrl_reviewer_core_task4_cell_cycle_scores_per_cell.csv: protein Tirosh S_score / G2M_score -> the same phase rule
  (G1 if both < 0).
- Populations: all cells, plates, DCT1 / DCT2 / ProLIF / Unknown under ctrl_deg_label and ctrl_cluster_label,
  protein-ID quartiles (CSV only); ProLIF label share per column; depth-matched ProLIF vs DCT1 + DCT2 controls
  (ctrl_reviewer_core_depth_matched_comparisons.csv, nn_matched_within_plate, caliper 0.05 ln(IDs)).
- ctrl_reviewer_core dropped the kidney background (reviewer_core B.6); no total-kidney row.

Inputs (read-only): atlas_dct region subset, split table and the two atlas_dct cycling tables; labels_core canonical labels; the ctrl_reviewer_core
                    tables above.
Outputs: results/tables/ctrl_rna_core/ctrl_rna_core_cycling_scp_fractions.csv, ctrl_rna_core_cycling_rna_fractions.csv
         (atlas_dct rows used), ctrl_rna_core_cycling_side_by_side.csv, ctrl_rna_core_cycling_checks.csv,
         ctrl_rna_core_cycling_figure_checks.csv; results/figures/ctrl_rna_core/ctrl_rna_core_cycling.pdf/.png

Command (project root):  python scripts/ctrl_rna_core_cycling.py
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
import cluster_de_lib as t5
import kidney_rna_lib as L7
import atlas_dct_lib as T
import ctrl_rna_core_lib as L
from atlas_dct_cycling import MEAS_STYLE

N_RULE = 73                      # ctrl_reviewer_core A.4: cells above the 90th percentile of cell_cycle_score (727 cells)
QUARTILE_EDGES = (108.0, 1372.5, 2096.0, 2786.0, 5058.0)          # labels_core section 4b
QUARTILE_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
NN_METHOD = "nn_matched_within_plate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ctrl_rna_core step 6: cycling in RNA (atlas_dct) vs control SCP (ctrl_reviewer_core).")
    cc.add_common_args(parser)
    return parser.parse_args()


def scp_tables(run, labels: pd.DataFrame, checks: t5.Checks) -> pd.DataFrame:
    """Per-cell SCP cycling calls from ctrl_reviewer_core (port of atlas_dct_cycling.scp_tables)."""
    rp = L.per_cell_table(run, L.CTRL_REVIEWER_CORE_TASK, "reannot_per_cell", labels, checks)
    sc4 = L.per_cell_table(run, L.CTRL_REVIEWER_CORE_TASK, "task4_cell_cycle_scores_per_cell", labels, checks)
    d = pd.DataFrame(index=labels.index)
    d["Mki67"] = rp["Mki67_detected"].astype(bool)
    d["Top2a"] = rp["Top2a_detected"].astype(bool)
    d["Mki67_or_Top2a"] = rp["mki67_or_top2a_detected"].astype(bool)
    d["cycling_score_rule"] = rp["reviewer_cycling_status"].astype(str).eq("Cycling")
    d["S_or_G2M"] = ~((sc4["S_score"] < 0) & (sc4["G2M_score"] < 0))
    d["plate"] = labels["plate"].astype(str)
    d["protein_id_quartile"] = pd.cut(labels["n_detected_proteins"].astype(float), list(QUARTILE_EDGES),
                                      labels=QUARTILE_LABELS, include_lowest=True).astype(str)
    for col in L.LABEL_COLUMNS:
        d[col] = labels[col].astype(str)
    return d


def main() -> None:
    args = parse_args()
    run = cc.Run(L.TASK, "cycling", args)
    L.ported_from(run, "atlas_dct_cycling.py")
    checks = t5.Checks(L.DATASET)

    # RNA: atlas_dct tables, with a recount of Mki67 / Top2a from the region subset ---------------------------------
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
               f"{n_cmp} rows; equal {n_ok}", n_ok)
    dct = rna[(rna["group_set"] == "published (atlas Celltype)") & (rna["group"] == "DCT") & (rna["by"] == "all")]
    dct = dct.set_index("measure")
    checks.add("atlas DCT cells (atlas_dct): Mki67 > 0; Top2a > 0; either; S or G2M (pooled; within batch)",
               "; ".join(f"{int(dct.loc[m, 'k'])}/{int(dct.loc[m, 'n_cells'])} ({dct.loc[m, 'fraction']:.3f})"
                         for m in dct.index), None)

    # SCP: ctrl_reviewer_core per-cell tables ----------------------------------------------------------------------------------
    adata, labels, _, _ = L.load_scp(run, checks)
    del adata
    scp = scp_tables(run, labels, checks)
    n_rule = int(scp["cycling_score_rule"].sum())
    checks.add("SCP cycling-score rule: cells (10% of 727 by construction; ctrl_reviewer_core A.4)", f"{n_rule}", n_rule == N_RULE)
    nd = {m: int(scp[m].sum()) for m in ("Mki67", "Top2a", "Mki67_or_Top2a")}
    checks.add("SCP Mki67 / Top2a / either detected equal labels_core (22 / 23 / 31 cells)",
               " / ".join(str(v) for v in nd.values()), list(nd.values()) == [22, 23, 31])
    cov = L.read_table(run, "task4_cell_cycle_gene_coverage", L.CTRL_REVIEWER_CORE_TASK)
    n_s = int(cov[(cov["set"] == "S")]["present"].astype(bool).sum())
    n_g = int(cov[(cov["set"] == "G2M")]["present"].astype(bool).sum())
    checks.add("protein Tirosh genes quantified (ctrl_reviewer_core): S; G2M", f"{n_s} of 43; {n_g} of 54", None)
    scp_meas = {"Mki67": "Mki67 detected", "Top2a": "Top2a detected", "Mki67_or_Top2a": "Mki67 or Top2a detected",
                "S_or_G2M": f"S or G2M phase (protein Tirosh scores; S {n_s}/43, G2M {n_g}/54)",
                "cycling_score_rule": "cycling-score rule (> 90th percentile)"}
    srows = []
    pops = ([("all cells", np.ones(len(scp), bool), "all")]
            + [(f"plate {p}", (scp["plate"] == p).to_numpy(), "plate") for p in ("P1", "P2")]
            + [(f"protein IDs {q}", (scp["protein_id_quartile"] == q).to_numpy(), "protein-ID quartile")
               for q in QUARTILE_LABELS])
    for col in L.LABEL_COLUMNS:
        pops += [(f"{g} ({L.LABEL_SHORT[col]})", (scp[col] == g).to_numpy(), col) for g in L.SCP_GROUPS]
    for pop, m, kind in pops:
        for meas, txt in scp_meas.items():
            k, n = int(scp.loc[m, meas].sum()), int(m.sum())
            lo, hi = T.wilson(k, n)
            srows.append({"population": pop, "population_kind": kind, "measure": meas, "measure_text": txt, "k": k,
                          "n_cells": n, "fraction": k / n, "ci_low": lo, "ci_high": hi, "source": "ctrl_reviewer_core per-cell tables"})
    for col in L.LABEL_COLUMNS:
        k, n = int((scp[col] == "ProLIF").sum()), len(scp)
        lo, hi = T.wilson(k, n)
        srows.append({"population": f"all cells ({L.LABEL_SHORT[col]})", "population_kind": col,
                      "measure": "ProLIF_share", "measure_text": f"ProLIF label share, {L.LABEL_DISPLAY[col]}", "k": k,
                      "n_cells": n, "fraction": k / n, "ci_low": lo, "ci_high": hi, "source": "labels_core canonical labels"})
    dm = L.read_table(run, "depth_matched_comparisons", L.CTRL_REVIEWER_CORE_TASK)
    dm = dm[dm["method"].eq(NN_METHOD) & dm["metric"].isin(["Mki67_detected", "Top2a_detected",
                                                             "Mki67_or_Top2a_detected"])]
    checks.add("ctrl_reviewer_core depth-matched rows (nn within plate; Mki67, Top2a, either; both columns)", f"{len(dm)}", len(dm) == 6)
    for r in dm.itertuples():
        meas = r.metric.replace("_detected", "")
        for who, val, n in (("ProLIF, depth-matched", r.value_prolif, r.n_prolif),
                            ("DCT1 + DCT2 controls, depth-matched", r.value_comparator, r.n_comparator)):
            k = int(round(val * n))
            lo, hi = T.wilson(k, int(n))
            srows.append({"population": f"{who} ({L.LABEL_SHORT[r.label_column]})", "population_kind": r.label_column,
                          "measure": meas, "measure_text": scp_meas[meas], "k": k, "n_cells": int(n), "fraction": val,
                          "ci_low": lo, "ci_high": hi, "mcnemar_p": r.p_value, "p_fdr_bh_t19": r.p_fdr_bh,
                          "source": "ctrl_reviewer_core depth_matched_comparisons (nn within plate, caliper 0.05)"})
    scp_df = pd.DataFrame(srows)
    a = scp_df[scp_df["population"] == "all cells"].set_index("measure")
    checks.add("SCP all 727: Mki67; Top2a; either; protein S or G2M; cycling-score rule",
               "; ".join(f"{a.loc[m, 'k']} ({a.loc[m, 'fraction']:.3f})" for m in scp_meas), None)
    for col in L.LABEL_COLUMNS:
        s = scp_df[scp_df["population_kind"] == col]
        pr = s[(s["population"] == f"ProLIF ({L.LABEL_SHORT[col]})")].set_index("measure")
        sh_ = s[s["measure"] == "ProLIF_share"].iloc[0]
        checks.add(f"{col}: ProLIF share; ProLIF cells with Mki67; Top2a; either; cycling-score rule",
                   f"{sh_['k']}/{sh_['n_cells']} ({sh_['fraction']:.3f}); "
                   + "; ".join(f"{pr.loc[m, 'k']}/{pr.loc[m, 'n_cells']}" for m in
                               ("Mki67", "Top2a", "Mki67_or_Top2a", "cycling_score_rule")), None)

    # side-by-side summary (RNA DCT rows and SCP populations; Mki67 / Top2a / either) ---------------------------
    side = []
    rsel = rna[(rna["by"].isin(["all", "Source"])) & rna["measure"].isin(["Mki67", "Top2a", "Mki67_or_Top2a",
                                                                          "S_or_G2M"])]
    for r in rsel.itertuples():
        side.append({"modality": "RNA (Mouse Kidney Atlas; atlas_dct)", "population": r.group if r.by == "all"
                     else f"{r.group}, {r.value}", "measure": r.measure, "k": r.k, "n_cells": r.n_cells,
                     "fraction": r.fraction, "ci_low": r.ci_low, "ci_high": r.ci_high})
    for r in scp_df.itertuples():
        side.append({"modality": "SCP-MS control DCT (ctrl_reviewer_core)", "population": r.population, "measure": r.measure,
                     "k": r.k, "n_cells": r.n_cells, "fraction": r.fraction, "ci_low": r.ci_low, "ci_high": r.ci_high})
    side = pd.DataFrame(side)

    fig_rows = []
    figure(run, rna, scp_df, fig_rows, n_s, n_g)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(scp_df, "cycling_scp_fractions")
    run.table(rna, "cycling_rna_fractions")
    run.table(thr, "cycling_rna_score_thresholds")
    run.table(side, "cycling_side_by_side")
    run.table(figs, "cycling_figure_checks")
    run.table(checks.frame(), "cycling_checks")
    run.finish({"rna": "atlas_dct tables reused unchanged (atlas_dct_cycling.py)", "scp": "ctrl_reviewer_core per-cell tables",
                "depth_matched": f"ctrl_reviewer_core {NN_METHOD}", "protein_id_quartiles": list(QUARTILE_EDGES),
                "checks_pass": checks.ok}, status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see ctrl_rna_core_cycling_checks.csv.")


def figure(run, rna: pd.DataFrame, scp: pd.DataFrame, fig_rows: list[dict], n_s: int, n_g: int) -> None:
    """atlas_dct_cycling.figure with the ctrl_reviewer_core control populations."""
    T.setup_matplotlib()
    LH = T.text_mm("Hg")[1]
    pub, der = "published (atlas Celltype)", "atlas_dct-derived split of atlas DCT"
    rna_rows = [(pub, "DCT", "all", "all", "DCT, all"), (pub, "DCT", "Source", "Nuclei", "DCT, nuclei"),
                (pub, "DCT", "Source", "Cell", "DCT, cells"), (der, T.DCT1_ATLAS_DCT, "all", "all", T.DCT1_ATLAS_DCT),
                (der, T.DCT2_ATLAS_DCT, "all", "all", T.DCT2_ATLAS_DCT), (der, T.DCT_UNASSIGNED, "all", "all", T.DCT_UNASSIGNED),
                (pub, "DCT-CNT", "all", "all", "DCT-CNT"), (pub, "CNT", "all", "all", "CNT"),
                (pub, "CTAL", "all", "all", "CTAL"), (pub, "MTAL", "all", "all", "MTAL")]
    scp_pops = ["all cells", "plate P1", "plate P2"]
    for col in L.LABEL_COLUMNS:
        scp_pops += [f"{g} ({L.LABEL_SHORT[col]})" for g in L.SCP_GROUPS]
    for col in L.LABEL_COLUMNS:
        s = L.LABEL_SHORT[col]
        scp_pops += [f"ProLIF, depth-matched ({s})", f"DCT1 + DCT2 controls, depth-matched ({s})"]
    scp_pops += [f"all cells ({L.LABEL_SHORT[c]})" for c in L.LABEL_COLUMNS]
    scp_lab = {p: p for p in scp_pops}
    for col in L.LABEL_COLUMNS:
        s = L.LABEL_SHORT[col]
        scp_lab[f"all cells ({s})"] = f"ProLIF share ({s})"
        scp_lab[f"DCT1 + DCT2 controls, depth-matched ({s})"] = f"controls, depth-matched ({s})"
    title = "Cycling fractions: RNA atlas (left) and SCP-MS control DCT data (right)"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    rh = 4.2
    lw1 = max(T.text_mm(r[4])[0] for r in rna_rows) + 1.5
    lw2 = max(T.text_mm(v)[0] for v in scp_lab.values()) + 1.5
    pw = (T.W - 2 * T.M - lw1 - lw2 - 8.0) / 2
    nrow = max(len(rna_rows), len(scp_pops))
    leg_h = LH + 2.0
    note = ("Fractions with Wilson 95% intervals; log scale (0 shown at 0.001, open symbol). RNA (atlas_dct tables, reused): "
            "atlas Celltype cells; S or G2M phase: scanpy score_genes_cell_cycle with the Tirosh lists, G1 if both scores "
            "< 0 (Seurat rule). " + L.ATLAS_DCT_SPLIT_CAVEAT + " SCP (ctrl_reviewer_core tables): deg = DE-rule labels (primary), cluster = "
            f"submitted cluster labels; the cycling-score rule marks 10% of the 727 cells by construction; protein S / G2M "
            f"scores rest on {n_s} / {n_g} quantified proteins; depth-matched: nearest neighbour within plate, caliper "
            "0.05 ln(protein IDs) (ctrl_reviewer_core B.3). One pool of 3 mice; plates are sort plates. RNA counts and protein detection "
            "are different measurements from different animals.")
    note_text, note_h = T.note_block(note)
    H = T.M + title_h + LH + 1.5 + nrow * rh + 9.0 + leg_h + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    y0 = T.M + title_h + LH + 1.5
    panels = [(T.M + lw1, "RNA (Mouse Kidney Atlas)", rna_rows, "rna"),
              (T.M + lw1 + pw + 8.0 + lw2, "SCP-MS (control DCT data, ctrl_reviewer_core)", scp_pops, "scp")]
    for x0, ptitle, rows, kind in panels:
        sh.text(x0 - (lw1 if kind == "rna" else lw2), y0 - LH - 1.5, ptitle, va="top")
        ax = sh.ax(x0, y0, pw, len(rows) * rh)
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
                sub = scp[scp["population"] == r].set_index("measure")
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
        labs = [r[4] for r in rows] if kind == "rna" else [scp_lab[r] for r in rows]
        for i, t_ in enumerate(labs):
            sh.text(x0 - 1.0, y0 + (i + 0.5) * rh, t_, ha="right", va="center_baseline")
    ly = y0 + nrow * rh + 9.0
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
    T.save_checked(run, sh, "cycling", title, fig_rows)


if __name__ == "__main__":
    main()
