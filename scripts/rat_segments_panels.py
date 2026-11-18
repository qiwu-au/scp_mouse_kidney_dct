#!/usr/bin/env python
"""rat_segments step 3: Limbutara segment-enriched protein panels (rule fixed before any SCP score).

Task rat_segments (authors, 30 Oct 2026).

Rule (approved plan, with the authors' deviation of 30 Oct 2026; constants in rat_segments_lib)
-------------------------------------------------------------------------------------------
Rat side, per KTEA entry and contrast (S1, S2, S3 one-vs-rest within PT; S1/S2 vs S3; cTAL vs mTAL and the reverse),
on log2 copies per cell centred per sample (sample median over the proteins quantified in all 16 PT / TAL samples):
- tested = quantified (value > 0 in >= 2/3 of the samples) in every segment of the contrast; entries quantified in
  every group segment but not in a reference segment are listed separately (``group_only``) and kept out;
- log2 FC = min(log2 mean of the group segments) - max(log2 mean of the reference segments) (means over the
  quantified samples); >= 1 (fc2, primary) or >= 0.585 (fc1.5, sensitivity);
- abundance: each group segment's mean >= the 25th percentile of that segment's quantified means;
- replicate separation: every quantified group sample above every quantified reference sample (replaces the planned
  Welch / BH q < 0.05 step, which no protein passed; Welch p and BH q are kept as information).
SCP side (pt_tal rule): the matched kidney protein (rat_segments_intake.py) is detected in >= 10% of the segment's SCP
cells under each label set (PT = PCT + PST; TAL), separately for ``kidney_t15_label`` and ``kidney_pre_label``;
panels keep the top 30 by log2 FC (ties: higher group abundance first).
Flags: V3 workbook markers (+ the kidney_annotation endothelial row), ribosomal, MitoCarta3.0 (pt_tal rules), pt_tal panel membership.
Rule diagnostics: panel sizes (rat side, matched, SCP detection) under the planned and the alternative rules, on raw
and centred values (the record behind the deviation; no score involved).

Inputs (read-only): rat_segments intake tables; KTEA per-sample file; kidney_object kidney object and labels; kidney_annotation labels; V3 workbook;
                    qc_mito MitoCarta annotation; pt_tal panels.
Outputs: results/tables/rat_segments/rat_segments_ktea_sample_medians.csv, rat_segments_ktea_segment_stats_centred.csv,
         rat_segments_rat_contrasts.csv, rat_segments_panels.csv, rat_segments_panel_funnel.csv,
         rat_segments_group_only.csv, rat_segments_rule_diagnostics.csv, rat_segments_panels_checks.csv,
         rat_segments_step_panels_manifest.json

Command (project root):  python scripts/rat_segments_panels.py
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
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc
import cluster_de_lib as t5
import pt_tal_lib as T9
import rat_segments_lib as T

RULES = ("planned_welch_bh", "welch_p_unadjusted", "separated", "fc_abundance_only")
RULE_TEXT = {"planned_welch_bh": "Welch, BH q < 0.05 (planned)", "welch_p_unadjusted": "Welch p < 0.05 (unadjusted)",
             "separated": "all samples separated (adopted)", "fc_abundance_only": "log2 FC and abundance only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rat_segments step 3: Limbutara segment panels.")
    cc.add_common_args(parser)
    return parser.parse_args()


def welch_max_p(lg: pd.DataFrame, seg: pd.Series, group, ref) -> np.ndarray:
    """Per entry: largest Welch p over (group segment, reference segment) pairs, on the quantified log2 values."""
    ps = []
    for g in group:
        a = lg[seg.index[seg == g]].to_numpy()
        for r in ref:
            b = lg[seg.index[seg == r]].to_numpy()
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy="omit").pvalue
            ps.append(np.asarray(np.ma.filled(p, np.nan), float))
    return np.max(np.vstack(ps), axis=0)          # NaN if any pair has no p


def contrast_table(st: pd.DataFrame, lg: pd.DataFrame, seg: pd.Series) -> pd.DataFrame:
    """Per KTEA entry x contrast: tested, log2 FCs, abundance, separation, Welch p (max), BH q."""
    p25 = {s: float(st.loc[st[f"quant_{s}"].astype(bool), f"log2_{s}"].quantile(T.ABUND_PCTL)) for s in T.STUDY_SEGMENTS}
    rows = []
    for name, (sg, group, ref, kind) in T.CONTRASTS.items():
        q_all = st[[f"quant_{s}" for s in (*group, *ref)]].astype(bool).all(axis=1)
        q_grp = st[[f"quant_{s}" for s in group]].astype(bool).all(axis=1)
        gmin = st[[f"log2_{s}" for s in group]].min(axis=1)
        rmax = st[[f"log2_{s}" for s in ref]].max(axis=1)
        gmean = st[[f"log2_{s}" for s in group]].mean(axis=1)
        rmean = st[[f"log2_{s}" for s in ref]].mean(axis=1)
        abund = np.logical_and.reduce([st[f"log2_{s}"].to_numpy() >= p25[s] for s in group])
        g_samples = lg[seg.index[seg.isin(group)]]
        r_samples = lg[seg.index[seg.isin(ref)]]
        sep = (g_samples.min(axis=1) > r_samples.max(axis=1)).to_numpy()
        p = welch_max_p(lg, seg, group, ref)
        d = pd.DataFrame({"contrast": name, "segment": sg, "kind": kind, "group_segments": ";".join(group),
                          "reference_segments": ";".join(ref), "rat_id": st.index, "rat_gene": st["rat_gene"].to_numpy(),
                          "quantified_all": q_all.to_numpy(), "quantified_group": q_grp.to_numpy(),
                          "group_only": (q_grp & ~q_all).to_numpy(), "log2fc_min_max": (gmin - rmax).to_numpy(),
                          "log2fc_mean": (gmean - rmean).to_numpy(), "group_log2_mean": gmean.to_numpy(),
                          "abundance_ok": abund, "samples_separated": sep & q_all.to_numpy(),
                          "n_group_samples_quantified": g_samples.notna().sum(axis=1).to_numpy(),
                          "n_reference_samples_quantified": r_samples.notna().sum(axis=1).to_numpy(),
                          "p_welch_max": np.where(q_all, p, np.nan)})
        for s in (*group, *ref):
            d[f"log2_{s}"] = st[f"log2_{s}"].to_numpy()
        rows.append(d)
    rc = pd.concat(rows, ignore_index=True)
    rc = cc.add_bh(rc, ["contrast"], p_col="p_welch_max", out_col="q_bh")
    return rc, p25


def rule_mask(rc: pd.DataFrame, rule: str, thr: float) -> pd.Series:
    base = rc["quantified_all"] & rc["abundance_ok"] & (rc["log2fc_min_max"] >= thr - 1e-12)
    if rule == "planned_welch_bh":
        return base & (rc["q_bh"] < T.Q_MAX)
    if rule == "welch_p_unadjusted":
        return base & (rc["p_welch_max"] < 0.05)
    if rule == "separated":
        return base & rc["samples_separated"]
    return base


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "panels", args)
    checks = t5.Checks(T.DATASET)
    T.check_sources(run)
    ann, vals, seg = T.read_samples()
    lgc, med, n_common = T.centred_log2(vals, seg)
    st_raw = T.read_table(run, "ktea_segment_stats", dtype={"rat_id": str}).set_index("rat_id").reindex(ann.index)
    stc = T.segment_stats(vals, seg, lgc)
    stc.insert(0, "rat_gene", ann[T.GENE_COL].to_numpy())
    stc.index.name = "rat_id"
    same_q = all((stc[f"quant_{s}"].to_numpy() == st_raw[f"quant_{s}"].astype(bool).to_numpy()).all() for s in T.SEGMENTS)
    checks.add("centred segment stats: same quantified flags as the intake stats", int(same_q), same_q)
    checks.add("sample medians use the proteins quantified in all 16 PT / TAL samples", n_common, n_common > 500)
    meds = pd.DataFrame({"sample": med.index, "segment": seg.reindex(med.index).to_numpy(), "median_log2_copies": med.to_numpy(),
                         "n_quantified": vals.notna().sum().reindex(med.index).to_numpy(),
                         "n_proteins_for_median": n_common})
    for s in T.STUDY_SEGMENTS:
        m = meds[meds["segment"] == s]["median_log2_copies"]
        checks.add(f"{s}: sample medians (log2 copies per cell)", ", ".join(f"{x:.2f}" for x in m), None)

    matches = T.read_table(run, "rat_mouse_matches", dtype={"rat_id": str, "mouse_protein": str},
                           keep_default_na=False)
    kept = matches[matches["kept"].astype(str).eq("True")].set_index("rat_id")["mouse_protein"]
    adata, labels, xz, ln1p, det = T9.load_scp(run, checks)
    v3 = T9.v3_marker_genes(run)
    fl = T9.protein_flags(run, adata.var_names.astype(str)).set_index("protein")
    t09 = T9.read_table(run, "panels")
    seg_det = {(c, sg): det[T9.segment_cells(labels, c, sg).to_numpy()].mean(axis=0)
               for c in T9.LABELS for sg in T9.SEGMENTS}

    def add_scp(rc: pd.DataFrame) -> pd.DataFrame:
        rc["mouse_protein"] = rc["rat_id"].map(kept).fillna("")
        for c in T9.LABELS:
            s = T9.LABEL_SHORT[c]
            rc[f"scp_det_{s}"] = [seg_det[(c, sg)].get(p, np.nan) if p else np.nan
                                  for sg, p in zip(rc["segment"], rc["mouse_protein"])]
            rc[f"passes_scp_{s}"] = rc[f"scp_det_{s}"] >= T.SCP_DET_MIN
        return rc

    # rule diagnostics (raw and centred values; planned and alternative rules) ------------------------------------
    rc_raw, _ = contrast_table(st_raw, np.log2(vals), seg)
    rc_raw = add_scp(rc_raw)
    rc, p25 = contrast_table(stc, lgc, seg)
    rc = add_scp(rc)
    diag = []
    for scale, tab in (("copies per cell (planned)", rc_raw), ("centred per sample (adopted)", rc)):
        for rule in RULES:
            for thr, t in T.THRESHOLDS.items():
                for name in T.CONTRASTS:
                    d = tab[tab["contrast"] == name]
                    m = rule_mask(d, rule, t)
                    diag.append({"scale": scale, "rule": RULE_TEXT[rule], "threshold": thr, "contrast": name,
                                 "tested": int(d["quantified_all"].sum()), "rat_pass": int(m.sum()),
                                 "matched": int((m & d["mouse_protein"].ne("")).sum()),
                                 "scp_10pct_t15": int((m & d["passes_scp_t15"]).sum()),
                                 "scp_10pct_pre": int((m & d["passes_scp_pre"]).sum()),
                                 "median_log2fc_mean_tested": float(d.loc[d["quantified_all"], "log2fc_mean"].median())})
    diag = pd.DataFrame(diag)
    planned = diag[(diag["scale"] == "copies per cell (planned)") & (diag["rule"] == RULE_TEXT["planned_welch_bh"])]
    checks.add("planned rule (copies per cell, Welch BH q < 0.05): rat-side passes over all contrasts and thresholds "
               "(the reason for the deviation)", int(planned["rat_pass"].sum()), None)
    for s in T.STUDY_SEGMENTS:
        checks.add(f"{s}: 25th percentile of the quantified centred log2 means", f"{p25[s]:.2f}", None)
    for thr, t in T.THRESHOLDS.items():
        rc[f"passes_rat_{thr}"] = rule_mask(rc, "separated", t)
        rc[f"group_only_abundant_{thr}"] = rc["group_only"] & rc["abundance_ok"]
    checks.add("Welch p only for tested entries (quantified in every contrast segment)",
               int(rc["p_welch_max"].notna().sum()), bool((rc["p_welch_max"].notna() <= rc["quantified_all"]).all()))
    checks.add("separation only for tested entries", int(rc["samples_separated"].sum()),
               bool((rc["samples_separated"] <= rc["quantified_all"]).all()))

    # panels -------------------------------------------------------------------------------------------------------
    t09_sets = {(c, x): set(t09.loc[(t09["label_column"] == c) & (t09["panel"] == x), "protein"])
                for c in T9.LABELS for x in t09["panel"].unique()}
    panel_rows, funnel = [], []
    for c in T9.LABELS:
        s = T9.LABEL_SHORT[c]
        for thr in T.THRESHOLDS:
            for name, (sg, group, ref, kind) in T.CONTRASTS.items():
                d = rc[rc["contrast"] == name]
                ok = d[f"passes_rat_{thr}"]
                base = d["quantified_all"] & d["abundance_ok"]
                step = {"label_column": c, "threshold": thr, "panel": name, "segment": sg, "kind": kind,
                        "rat_entries": len(d), "tested": int(d["quantified_all"].sum()),
                        "plus_abundance": int(base.sum()),
                        "plus_fc": int((base & (d["log2fc_min_max"] >= T.THRESHOLDS[thr] - 1e-12)).sum()),
                        "plus_separated": int(ok.sum()), "plus_matched": int((ok & d["mouse_protein"].ne("")).sum())}
                sel = d[ok & d["mouse_protein"].ne("") & d[f"passes_scp_{s}"]].copy()
                step["plus_scp_10pct"] = len(sel)
                sel = sel.sort_values(["log2fc_min_max", "group_log2_mean"], ascending=[False, False])
                if sel["mouse_protein"].duplicated().any():
                    raise ValueError(f"duplicate mouse protein in panel {name}")
                sel = sel.head(T.PANEL_TOP)
                pr = []
                for rank, r in enumerate(sel.itertuples(), start=1):
                    p = r.mouse_protein
                    in_t09 = [x for x in T.PT_TAL_PANELS[name] if p in t09_sets.get((c, x), set())]
                    pr.append({"label_column": c, "threshold": thr, "panel": name, "segment": sg, "kind": kind,
                               "rank": rank, "protein": p, "rat_id": r.rat_id, "rat_gene": r.rat_gene,
                               "log2fc_min_max": r.log2fc_min_max, "log2fc_mean": r.log2fc_mean,
                               "group_log2_mean": r.group_log2_mean, "p_welch_max": r.p_welch_max, "q_bh": r.q_bh,
                               "scp_detection": getattr(r, f"scp_det_{s}"),
                               **{f"log2_{x}": getattr(r, f"log2_{x}") for x in (*group, *ref)},
                               "v3_marker_types": ";".join(v3.get(p, [])),
                               "ribosomal": bool(fl.loc[p, "ribosomal"]), "mitocarta": bool(fl.loc[p, "mitocarta"]),
                               "in_t09_panel": bool(in_t09), "t09_panels": ";".join(in_t09)})
                panel_rows += pr
                step.update(panel_size=len(pr), n_v3=sum(bool(x["v3_marker_types"]) for x in pr),
                            n_ribosomal=sum(x["ribosomal"] for x in pr), n_mitocarta=sum(x["mitocarta"] for x in pr),
                            n_in_t09_panel=sum(x["in_t09_panel"] for x in pr),
                            median_scp_detection=float(np.median([x["scp_detection"] for x in pr])) if pr else np.nan,
                            group_only_abundant=int(d[f"group_only_abundant_{thr}"].sum()),
                            group_only_abundant_matched=int((d[f"group_only_abundant_{thr}"]
                                                             & d["mouse_protein"].ne("")).sum()))
                funnel.append(step)
    panels = pd.DataFrame(panel_rows)
    funnel = pd.DataFrame(funnel)
    for r in funnel.itertuples():
        checks.add(f"{T9.LABEL_SHORT[r.label_column]} {r.threshold} {r.panel}: tested / +abund / +FC / +separated / "
                   f"matched / +SCP 10% / panel; V3", f"{r.tested} / {r.plus_abundance} / {r.plus_fc} / "
                   f"{r.plus_separated} / {r.plus_matched} / {r.plus_scp_10pct} / {r.panel_size}; {r.n_v3}", None)
    checks.add("panels: each protein once per (label set, threshold, panel); <= 30 proteins",
               f"{len(panels)} rows; max {int(funnel['panel_size'].max())}",
               bool(len(panels) and not panels.duplicated(["label_column", "threshold", "panel", "protein"]).any()
                    and funnel["panel_size"].max() <= T.PANEL_TOP))
    checks.add("panel proteins are kidney matrix proteins", int(panels["protein"].isin(adata.var_names).sum()),
               bool(panels["protein"].isin(adata.var_names).all()))
    prim = funnel[funnel["threshold"] == T.PRIMARY_THR]
    need = prim[prim["panel"].isin(["S1/S2", "S3"])]
    checks.add("primary (fc2) S1/S2 and S3 panels non-empty under each label set (needed for the S1/S2 - S3 score)",
               "; ".join(f"{T9.LABEL_SHORT[r.label_column]} {r.panel} {r.panel_size}" for r in need.itertuples()),
               bool((need["panel_size"] > 0).all()))
    go = rc[rc["group_only"] & rc["abundance_ok"]].copy()

    run.table(meds, "ktea_sample_medians")
    run.table(stc.reset_index(), "ktea_segment_stats_centred")
    run.table(rc, "rat_contrasts")
    run.table(panels, "panels")
    run.table(funnel, "panel_funnel")
    run.table(go, "group_only")
    run.table(diag, "rule_diagnostics")
    run.table(checks.frame(), "panels_checks")
    run.finish({"thresholds_log2": T.THRESHOLDS, "quant_frac": T.QUANT_FRAC, "abundance_percentile": T.ABUND_PCTL,
                "centring": T.CENTRING_TEXT, "replicate_rule": T.SEPARATION_TEXT,
                "planned_rule_not_used": "Welch t (largest p over segment pairs), BH q < 0.05 within contrast",
                "scp_detection_min": T.SCP_DET_MIN, "panel_top": T.PANEL_TOP,
                "contrasts": {k: {"segment": v[0], "group": v[1], "reference": v[2], "kind": v[3]}
                              for k, v in T.CONTRASTS.items()},
                "percentile_25_centred_log2": p25, "checks_pass": checks.ok},
               status="completed" if checks.ok else "completed with failed checks")
    if not checks.ok:
        raise SystemExit("A check failed; see rat_segments_panels_checks.csv.")


if __name__ == "__main__":
    main()
