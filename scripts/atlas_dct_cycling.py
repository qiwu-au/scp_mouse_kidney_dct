#!/usr/bin/env python
"""atlas_dct step 7: cycling fractions in the RNA atlas next to the SCP fractions from reviewer_core.

Analysis plan, step 7
(R1 ProLIF prevalence).

RNA (atlas DCT-region cells, atlas_dct_atlas_dct_region.h5ad)
---------------------------------------------------------------
- Per cell: Mki67 counts > 0; Top2a counts > 0; either; Seurat-style S / G2M phase from scanpy
  score_genes_cell_cycle (Tirosh et al. 2016 S and G2M lists as in reviewer_core, mouse symbols, aliases Mlf1ip / Fam64a /
  Hn1; random_state 42) on log-normalized expression (10,000 per cell, log1p; all 16,711 genes). Threshold (scanpy =
  Seurat CellCycleScoring): a cell is G1 if both S_score and G2M_score are < 0, otherwise S or G2M (the higher
  score); 'cycling' = S or G2M.
- Scoring pool: all 30,086 region cells (primary); sensitivity: scored within each Origin x Source batch. The
  fractions with max(S_score, G2M_score) > 0.1, 0.2, 0.3 are also given.
- Fractions (Wilson 95% CI) by RNA group (published classes incl. LOH; atlas_dct-derived split) x Source, Technology,
  Origin (= Age; one Age per study) and UMI-depth bin.

SCP (from reviewer_core, not recomputed except fractions and Wilson CIs)
-------------------------------------------------------------
- reviewer_core_reannot_per_cell.csv: Mki67 detected, Top2a detected, either, cycling-score rule (cell_cycle_score >
  90th percentile, so 10% of all cells by construction) - all cells, plates, both V1 label columns.
- reviewer_core_task4_cell_cycle_scores_per_cell.csv: protein Tirosh S_score / G2M_score -> the same phase rule
  (S 5 of 43, G2M 10 of 54 proteins quantified; reviewer_core A.2).
- ProLIF label share (both columns); depth-matched values (reviewer_core_depth_matched_comparisons.csv,
  nn_matched_within_plate); total-kidney background (reviewer_core_kidney_bg_by_label.csv).

Inputs (read-only): atlas_dct region subset and split table; kidney_object canonical DCT labels; the reviewer_core tables above.
Outputs: results/tables/atlas_dct/atlas_dct_cycling_*.csv; results/figures/atlas_dct/atlas_dct_cycling.pdf/.png

Command (project root):  python scripts/atlas_dct_cycling.py
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
import reviewer_core_lib as t2
import cluster_de_lib as t5
import kidney_rna_lib as L7
import atlas_dct_lib as T

DEPTH_EDGES = [0, 1000, 2000, 4000, 8000, np.inf]
DEPTH_LABELS = ["<1,000", "1,000-1,999", "2,000-3,999", "4,000-7,999", ">=8,000"]
SCORE_THRESHOLDS = (0.0, 0.1, 0.2, 0.3)
RNA_MEASURES = {"Mki67": "Mki67 counts > 0", "Top2a": "Top2a counts > 0", "Mki67_or_Top2a": "Mki67 or Top2a > 0",
                "S_or_G2M": "S or G2M phase (Tirosh, Seurat rule)",
                "S_or_G2M_within_batch": "S or G2M phase, scored within Origin x Source"}
MIN_BATCH = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="atlas_dct step 7: cycling in RNA vs SCP.")
    cc.add_common_args(parser)
    return parser.parse_args()


def match_cc(var_names) -> tuple[list[str], list[str], pd.DataFrame]:
    look = L7.atlas_gene_lookup(var_names)
    rows, out = [], {"S": [], "G2M": []}
    for phase, genes in (("S", t2.TIROSH_S), ("G2M", t2.TIROSH_G2M)):
        for g in genes:
            hit = next((look[c.lower()] for c in (g, *t2.TIROSH_ALIASES.get(g, [])) if c.lower() in look), None)
            rows.append({"set": phase, "gene": g, "aliases": ";".join(t2.TIROSH_ALIASES.get(g, [])),
                         "atlas_gene": hit or "", "present": hit is not None})
            if hit:
                out[phase].append(hit)
    return out["S"], out["G2M"], pd.DataFrame(rows)


def score(ln, var_names, s_genes, g_genes, mask=None) -> pd.DataFrame:
    import anndata as ad
    import scanpy as sc

    idx = np.flatnonzero(np.ones(ln.shape[0], bool) if mask is None else mask)
    a = ad.AnnData(X=ln[idx], var=pd.DataFrame(index=list(var_names)))
    sc.tl.score_genes_cell_cycle(a, s_genes=s_genes, g2m_genes=g_genes, random_state=T.SEED, use_raw=False)
    return pd.DataFrame({"S_score": a.obs["S_score"].to_numpy(), "G2M_score": a.obs["G2M_score"].to_numpy(),
                         "phase": a.obs["phase"].astype(str).to_numpy()}, index=idx)


def frac_rows(df: pd.DataFrame, group_col: str, group_set: str, measures: dict, by_cols) -> list[dict]:
    rows = []
    for g, sub in df.groupby(group_col):
        for by in by_cols:
            parts = [("all", sub)] if by == "all" else list(sub.groupby(by))
            for val, s in parts:
                for m in measures:
                    k, n = int(s[m].sum()), len(s)
                    lo, hi = T.wilson(k, n)
                    rows.append({"group_set": group_set, "group": g, "by": by, "value": str(val), "n_cells": n,
                                 "measure": m, "measure_text": measures[m], "k": k, "fraction": k / n if n else np.nan,
                                 "ci_low": lo, "ci_high": hi})
    return rows


def scp_tables(run, labels: pd.DataFrame) -> pd.DataFrame:
    rp = pd.read_csv(run.use_input(T.REVIEWER_CORE_DIR / "reviewer_core_reannot_per_cell.csv", "reviewer_core per-cell cycling calls"),
                     dtype={"cell_id": str}).set_index("cell_id").reindex(labels.index)
    sc4 = pd.read_csv(run.use_input(T.REVIEWER_CORE_DIR / "reviewer_core_task4_cell_cycle_scores_per_cell.csv",
                                    "reviewer_core per-cell protein cell-cycle scores"), dtype={"cell_id": str}).set_index("cell_id")
    sc4 = sc4.reindex(labels.index)
    d = pd.DataFrame(index=labels.index)
    d["Mki67"] = rp["Mki67_detected"].astype(bool)
    d["Top2a"] = rp["Top2a_detected"].astype(bool)
    d["Mki67_or_Top2a"] = rp["mki67_or_top2a_detected"].astype(bool)
    d["cycling_score_rule"] = rp["reviewer_cycling_status"].astype(str).eq("Cycling")
    d["S_or_G2M"] = ~((sc4["S_score"] < 0) & (sc4["G2M_score"] < 0))
    d["plate"] = labels["plate"].astype(str)
    for col in T.LABEL_COLUMNS:
        d[col] = labels[col].astype(str)
    return d, rp


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "cycling", args)
    checks = t5.Checks(T.DATASET)
    region = T.load_region(run, with_split=True)
    obs = region.obs
    genes = region.var_names.astype(str)
    s_genes, g_genes, cov = match_cc(genes)
    checks.add("Tirosh genes in the atlas: S; G2M", f"{len(s_genes)} of 43; {len(g_genes)} of 54", None)
    look = L7.atlas_gene_lookup(genes)
    counts = region.X
    tot = np.asarray(counts.sum(axis=1)).ravel()
    ln = T.lognorm(counts)
    df = pd.DataFrame({"atlas_cell": region.obs_names.astype(str), "rna_group": obs["rna_group"].to_numpy(),
                       "rna_group_derived": obs["rna_group_derived"].to_numpy(), "Source": obs["Source"].to_numpy(),
                       "Technology": obs["Technology"].to_numpy(), "Origin": obs["Origin"].to_numpy(),
                       "Age": obs["Age"].to_numpy(), "total_counts_raw": tot})
    df["Origin_Age"] = df["Origin"] + " (" + df["Age"] + ")"
    for g in ("Mki67", "Top2a"):
        j = genes.get_loc(look[g.lower()])
        df[f"{g}_counts"] = np.asarray(counts[:, j].todense()).ravel()
        df[g] = df[f"{g}_counts"] > 0
    df["Mki67_or_Top2a"] = df["Mki67"] | df["Top2a"]
    sc_all = score(ln, genes, s_genes, g_genes)
    df["S_score"], df["G2M_score"], df["phase"] = sc_all["S_score"].to_numpy(), sc_all["G2M_score"].to_numpy(), \
        sc_all["phase"].to_numpy()
    df["S_or_G2M"] = df["phase"] != "G1"
    # within-batch scoring
    df["phase_within_batch"] = ""
    batches = df.groupby(["Origin", "Source"]).groups
    small = []
    for key, idx in batches.items():
        m = np.zeros(len(df), bool)
        m[df.index.get_indexer(idx)] = True
        if m.sum() < MIN_BATCH:
            small.append(f"{key[0]}/{key[1]} {int(m.sum())}")
            continue
        s = score(ln, genes, s_genes, g_genes, m)
        df.loc[df.index[s.index], "phase_within_batch"] = s["phase"].to_numpy()
    checks.add(f"batches (Origin x Source) scored separately; batches with < {MIN_BATCH} cells (not scored)",
               f"{len(batches) - len(small)}; {', '.join(small) or 'none'}", None)
    checks.add("cells without a within-batch phase", f"{int((df['phase_within_batch'] == '').sum())}",
               bool((df["phase_within_batch"] != "").all()))
    df["S_or_G2M_within_batch"] = df["phase_within_batch"].isin(["S", "G2M"])
    df["depth_bin"] = pd.cut(df["total_counts_raw"], DEPTH_EDGES, right=False, labels=DEPTH_LABELS).astype(str)

    by_cols = ("all", "Source", "Technology", "Origin_Age", "depth_bin")
    rows = frac_rows(df, "rna_group", "published (atlas Celltype)", RNA_MEASURES, by_cols)
    rows += frac_rows(df[df["rna_group"] == "DCT"], "rna_group_derived", "atlas_dct-derived split of atlas DCT",
                      RNA_MEASURES, by_cols)
    rna = pd.DataFrame(rows)
    thr_rows = []
    mx = np.maximum(df["S_score"], df["G2M_score"])
    for col, gs in (("rna_group", "published"), ("rna_group_derived", "derived")):
        for g, sub in df.groupby(col):
            for t_ in SCORE_THRESHOLDS:
                k, n = int((mx[sub.index] > t_).sum()), len(sub)
                thr_rows.append({"group_set": gs, "group": g, "threshold_max_S_G2M": t_, "n_cells": n, "k": k,
                                 "fraction": k / n})
    dct = rna[(rna["group"] == "DCT") & (rna["by"] == "all")].set_index("measure")
    checks.add("atlas DCT cells (16,381): Mki67 > 0; Top2a > 0; either; S or G2M (pooled; within batch)",
               "; ".join(f"{dct.loc[m, 'k']} ({dct.loc[m, 'fraction']:.3f})" for m in RNA_MEASURES), None)
    for src in ("Nuclei", "Cell"):
        s = rna[(rna["group"] == "DCT") & (rna["by"] == "Source") & (rna["value"] == src)].set_index("measure")
        checks.add(f"atlas DCT {src}: Mki67 > 0; Top2a > 0; either; S or G2M (pooled; within batch)",
                   "; ".join(f"{s.loc[m, 'k']}/{s.loc[m, 'n_cells']} ({s.loc[m, 'fraction']:.3f})" for m in RNA_MEASURES),
                   None)

    # SCP ---------------------------------------------------------------------------------------------------------
    _, labels = t5.load(run, T.DATASET)
    scp, rp = scp_tables(run, labels)
    same = rp[list(T.LABEL_COLUMNS)].astype(str).equals(labels[list(T.LABEL_COLUMNS)].astype(str))
    checks.add("reviewer_core per-cell table: same cells and labels as kidney_object", f"{same}", same)
    n_rule = int(scp["cycling_score_rule"].sum())
    checks.add("SCP cycling-score rule: cells (10% of 750 by construction)", f"{n_rule}", n_rule == 75)
    scp_meas = {"Mki67": "Mki67 detected", "Top2a": "Top2a detected", "Mki67_or_Top2a": "Mki67 or Top2a detected",
                "S_or_G2M": "S or G2M phase (protein Tirosh scores; S 5/43, G2M 10/54)",
                "cycling_score_rule": "cycling-score rule (> 90th percentile)"}
    srows = []
    pops = [("all cells", np.ones(len(scp), bool))] + [(f"plate {p}", (scp["plate"] == p).to_numpy()) for p in ("P1", "P2")]
    for col in T.LABEL_COLUMNS:
        pops += [(f"{g} ({T.LABEL_SHORT[col]})", (scp[col] == g).to_numpy()) for g in T.SCP_GROUPS]
    for pop, m in pops:
        for meas, txt in scp_meas.items():
            k, n = int(scp.loc[m, meas].sum()), int(m.sum())
            lo, hi = T.wilson(k, n)
            srows.append({"population": pop, "measure": meas, "measure_text": txt, "k": k, "n_cells": n,
                          "fraction": k / n, "ci_low": lo, "ci_high": hi, "source": "reviewer_core per-cell tables"})
    for col in T.LABEL_COLUMNS:
        k, n = int((scp[col] == "ProLIF").sum()), len(scp)
        lo, hi = T.wilson(k, n)
        srows.append({"population": f"all cells ({T.LABEL_SHORT[col]})", "measure": "ProLIF_share",
                      "measure_text": f"ProLIF label share, {T.LABEL_DISPLAY[col]}", "k": k, "n_cells": n,
                      "fraction": k / n, "ci_low": lo, "ci_high": hi, "source": "kidney_object canonical labels"})
    dm = pd.read_csv(run.use_input(T.REVIEWER_CORE_DIR / "reviewer_core_depth_matched_comparisons.csv", "reviewer_core depth-matched"))
    dm = dm[dm["method"].eq("nn_matched_within_plate") & dm["metric"].isin(["Mki67_detected", "Top2a_detected",
                                                                           "Mki67_or_Top2a_detected"])]
    for r in dm.itertuples():
        meas = r.metric.replace("_detected", "")
        for who, val, n in (("ProLIF, depth-matched", r.value_prolif, r.n_prolif),
                            ("DCT1 + DCT2 controls, depth-matched", r.value_comparator, r.n_comparator)):
            k = int(round(val * n))
            lo, hi = T.wilson(k, int(n))
            srows.append({"population": f"{who} ({T.LABEL_SHORT[r.label_column]})", "measure": meas,
                          "measure_text": scp_meas[meas], "k": k, "n_cells": int(n), "fraction": val, "ci_low": lo,
                          "ci_high": hi, "source": "reviewer_core depth_matched_comparisons (nn within plate, caliper 0.05)"})
    kb = pd.read_csv(run.use_input(T.REVIEWER_CORE_DIR / "reviewer_core_kidney_bg_by_label.csv", "reviewer_core kidney background"))
    kr = kb[(kb["dataset"] == "KIDNEY") & (kb["label_column"] == "all_cells")].iloc[0]
    for meas, k in (("Mki67", kr["n_Mki67_detected"]), ("Top2a", kr["n_Top2a_detected"]),
                    ("Mki67_or_Top2a", kr["n_Mki67_or_Top2a_detected"] if "n_Mki67_or_Top2a_detected" in kr
                     else None)):
        if k is None or pd.isna(k):
            continue
        n = int(kr["n_cells"])
        lo, hi = T.wilson(int(k), n)
        srows.append({"population": "total kidney SCP (712 cells)", "measure": meas, "measure_text": scp_meas[meas],
                      "k": int(k), "n_cells": n, "fraction": int(k) / n, "ci_low": lo, "ci_high": hi,
                      "source": "reviewer_core kidney_bg_by_label"})
    scp_df = pd.DataFrame(srows)
    a = scp_df[scp_df["population"] == "all cells"].set_index("measure")
    checks.add("SCP all 750: Mki67; Top2a; either; protein S or G2M; cycling-score rule",
               "; ".join(f"{a.loc[m, 'k']} ({a.loc[m, 'fraction']:.3f})" for m in scp_meas), None)

    fig_rows = []
    figure(run, rna, scp_df, fig_rows)
    figs = pd.DataFrame(fig_rows)
    checks.add("figures passing the format checks", f"{int(figs['pass'].sum())} of {len(figs)}", bool(figs["pass"].all()))
    run.table(cov, "cycling_gene_coverage")
    run.table(df.drop(columns=["Origin_Age"]), "cycling_rna_per_cell")
    run.table(rna, "cycling_rna_fractions")
    run.table(pd.DataFrame(thr_rows), "cycling_rna_score_thresholds")
    run.table(scp_df, "cycling_scp_fractions")
    run.table(figs, "cycling_figure_checks")
    run.table(checks.frame(), "cycling_checks")
    run.finish({"phase_rule": "G1 if S_score < 0 and G2M_score < 0, else S or G2M (scanpy score_genes_cell_cycle)",
                "scoring_pool": "all region cells (primary); Origin x Source batches (sensitivity)",
                "min_batch_cells": MIN_BATCH, "depth_bins_total_counts": DEPTH_LABELS,
                "score_thresholds_reported": list(SCORE_THRESHOLDS), "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see atlas_dct_cycling_checks.csv.")


MEAS_STYLE = {"Mki67": (T.SERIES[0], "o", "Mki67"), "Top2a": (T.SERIES[1], "s", "Top2a"),
              "Mki67_or_Top2a": (T.SERIES[2], "D", "Mki67 or Top2a"), "S_or_G2M": (T.SERIES[6], "^", "S or G2M phase"),
              "cycling_score_rule": (T.INK["muted"], "v", "cycling-score rule"),
              "ProLIF_share": (T.INK["primary"], "P", "ProLIF label share")}


def figure(run, rna: pd.DataFrame, scp: pd.DataFrame, fig_rows: list[dict]) -> None:
    plt = T.setup_matplotlib()
    LH = T.text_mm("Hg")[1]
    rna_rows = [("DCT", "all", "all", "DCT, all"), ("DCT", "Source", "Nuclei", "DCT, nuclei"),
                ("DCT", "Source", "Cell", "DCT, cells"), (T.DCT1_ATLAS_DCT, "all", "all", T.DCT1_ATLAS_DCT),
                (T.DCT2_ATLAS_DCT, "all", "all", T.DCT2_ATLAS_DCT), (T.DCT_UNASSIGNED, "all", "all", T.DCT_UNASSIGNED),
                ("DCT-CNT", "all", "all", "DCT-CNT"), ("CNT", "all", "all", "CNT"), ("CTAL", "all", "all", "CTAL"),
                ("MTAL", "all", "all", "MTAL")]
    scp_pops = ["all cells", "plate P1", "plate P2", "DCT1 (cell)", "DCT2 (cell)", "ProLIF (cell)", "DCT1 (cluster)",
                "DCT2 (cluster)", "ProLIF (cluster)", "ProLIF, depth-matched (cell)",
                "DCT1 + DCT2 controls, depth-matched (cell)", "ProLIF, depth-matched (cluster)",
                "DCT1 + DCT2 controls, depth-matched (cluster)", "total kidney SCP (712 cells)",
                "all cells (cell)", "all cells (cluster)"]
    scp_lab = {p: p for p in scp_pops}
    scp_lab.update({"all cells (cell)": "ProLIF share, per-cell labels", "all cells (cluster)": "ProLIF share, cluster labels",
                    "DCT1 + DCT2 controls, depth-matched (cell)": "controls, depth-matched (cell)",
                    "DCT1 + DCT2 controls, depth-matched (cluster)": "controls, depth-matched (cluster)"})
    title = "Cycling fractions: RNA atlas (left) and SCP-MS DCT dataset (right)"
    title_h = T.text_mm(title, T.TITLE, "bold")[1] + 1.5
    rh = 4.4
    lw1 = max(T.text_mm(r[3])[0] for r in rna_rows) + 1.5
    lw2 = max(T.text_mm(v)[0] for v in scp_lab.values()) + 1.5
    pw = (T.W - 2 * T.M - lw1 - lw2 - 8.0) / 2
    nrow = max(len(rna_rows), len(scp_pops))
    leg_h = LH + 2.0
    note = ("Fractions with Wilson 95% intervals; log scale (0 shown at 0.001, open symbol). RNA: atlas Celltype cells "
            "(DCT1 / DCT2 / unassigned (atlas_dct): atlas_dct-derived split of the atlas DCT cells); S or G2M phase: scanpy "
            "score_genes_cell_cycle with the Tirosh lists, G1 if both scores < 0 (Seurat rule). SCP: reviewer_core tables; "
            "the cycling-score rule marks 10% of all 750 cells by construction; protein S / G2M scores rest on 5 / 10 "
            "quantified proteins; depth-matched: nearest neighbour within plate (reviewer_core B.3). RNA counts and protein "
            "detection are different measurements from different animals.")
    note_text, note_h = T.note_block(note)
    H = T.M + title_h + LH + 1.5 + nrow * rh + 9.0 + leg_h + note_h + T.M
    sh = T.Sheet(H)
    sh.text(T.M, T.M, title, fontsize=T.TITLE, fontweight="bold", va="top")
    y0 = T.M + title_h + LH + 1.5
    panels = [(T.M + lw1, "RNA (Mouse Kidney Atlas)", rna_rows, "rna"),
              (T.M + lw1 + pw + 8.0 + lw2, "SCP-MS (DCT dataset, reviewer_core)", scp_pops, "scp")]
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
                g, by, val, _ = r
                sub = rna[(rna["group"] == g) & (rna["by"] == by) & (rna["value"] == val)].set_index("measure")
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
        labs = [r[3] for r in rows] if kind == "rna" else [scp_lab[r] for r in rows]
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
