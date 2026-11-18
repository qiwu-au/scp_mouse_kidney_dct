#!/usr/bin/env python
"""The response to the reviewers: Supplementary Figures S-08-a to S-08-c (RNA-protein integration), revised annotation only.

The authors decided on 30 Oct 2026 that the kidney_annotation revised
annotation (kidney_t15_label) is the only kidney annotation in the manuscript and letter. The rna_protein_integration figures show the
submitted Figure 2A labels next to it and use internal names, so the the response to the reviewers figures are redrawn here from the
stored rna_protein_integration outputs. No new statistics: every plotted value is read from the rna_protein_integration tables or the joint embedding.

Figures (results/figures/report/, PDF + 600-dpi PNG, 180 mm wide; no in-figure title or footnote, the legend
text is in the the response to the reviewers file):
- report_08_joint_umaps (S-08-a): UMAPs of the embeddings A (joint PCA), B (A + Harmony, modality as batch) and
  C (atlas scVI space; SCP-MS cells placed by their 15 nearest atlas cells), RNA draw 42; top by modality, bottom by
  cell type (revised annotation; atlas cells by the label their class is matched to).
- report_08_mixing (S-08-b): modality mixing within each cell type, A vs B (median and range over 5 draws).
- report_08_transfer (S-08-c): atlas Celltype transferred to the SCP-MS cells in B (draw 42, k = 15), with the
  matched (solid) and accepted mixed (dashed) atlas classes outlined and the matched fraction per label.
Display names (authors, 30 Oct 2026): DCT1 -> DCT, PC -> CNT/PC.

Inputs (read-only): kidney_object canonical kidney object and kidney_annotation labels (via kidney_rna_lib.load_kidney), kidney_rna mapping table,
                    results/h5ad/rna_protein_integration_joint_embedding.h5ad, rna_protein_integration tables mixing_per_type, mixing_global,
                    transfer_confusion, transfer_per_label.
Outputs: the figures above; results/tables/report/report_08_integration_figures_{values,checks,
         figure_checks}.csv; results/tables/report/report_step_08_integration_figures_manifest.json

Command (project root):  python scripts/report_integration_figures.py
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
import math

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as L7
import rna_protein_integration_lib as T
import rna_protein_integration_figures as F10          # rna_protein_integration figure helpers and colours (chip legend, axes, embedding colours)
import kidney_annotation_chosen_labels as t15c   # kidney_annotation palette
import atlas_dct_lib as T8

plt = T8.setup_matplotlib()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

TASK = "report"
STEP = "08_integration_figures"
COL = "kidney_t15_label"
DISPLAY = {"DCT1": "DCT", "PC": "CNT/PC"}      # authors, 30 Oct 2026
SPACE = {"pca": "A: joint PCA", "harmony": "B: joint PCA + Harmony", "scvi": "C: atlas scVI space"}
INK = T8.INK
W, M, FONT = T8.W, T8.M, T8.FONT
EXPECTED_SCP = {"PCT": 160, "PST": 158, "TAL": 138, "Unknown": 109, "DCT1": 58, "PC": 38, "JG": 24,
                "Endothelial": 19, "IC-B": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="the response to the reviewers: S-08-a to S-08-c, revised annotation only.")
    cc.add_common_args(parser)
    return parser.parse_args()


def disp(label: str) -> str:
    return DISPLAY.get(label, label)


def tw(s: str) -> float:
    return T8.text_mm(s, FONT)[0]


# ---------------------------------------------------------------------------
# S-08-a: joint UMAPs
# ---------------------------------------------------------------------------
def umap_figure(run, rows_out, joint, palette: dict[str, str], order: list[str]) -> list[dict]:
    obs = joint.obs
    is_scp = (obs["modality"] == "SCP").to_numpy()
    common = obs["common_t15"].astype(str).to_numpy()
    n_rna, n_scp = int((~is_scp).sum()), int(is_scp.sum())
    gap = 5.0
    pw = (W - 2 * M - 2 * gap) / 3
    y_head1 = M + 2.0
    y1 = y_head1 + 4.0
    y_head2 = y1 + pw + 3.0
    y2 = y_head2 + 4.0
    y_leg = y2 + pw + 3.0
    labs = [lab for lab in order if lab in set(obs.loc[is_scp, COL])]
    items, vals = [], []
    for lab in labs:
        n_s = int(((obs[COL] == lab).to_numpy() & is_scp).sum())
        n_r = int(((common == lab) & ~is_scp).sum())
        items.append((palette.get(lab, INK["muted"]), f"{disp(lab)}: SCP-MS {n_s}, atlas {n_r}"))
        vals.append({"figure": "joint_umaps", "label": lab, "display": disp(lab), "n_scp": n_s, "n_atlas": n_r})
    n_none = int(((~is_scp) & pd.Series(common).str.startswith("atlas: ").to_numpy()).sum())
    items.append((F10.NO_COUNTERPART, f"atlas cells without an SCP-MS counterpart: {n_none:,}"))
    vals.append({"figure": "joint_umaps", "label": "no counterpart", "display": "", "n_scp": 0, "n_atlas": n_none})
    leg_h = (len(items) + 2) // 3 * 3.4
    H = y_leg + leg_h + M
    sh = T8.Sheet(H)
    sh.text(M, y_head1, "Coloured by modality", va="center", color=INK["secondary"])
    lx = M + 36
    sh.fig.add_artist(Line2D([(lx + 1) / W], [1 - y_head1 / H], marker="o", linestyle="", markersize=2.6,
                             markerfacecolor=F10.GREY_RNA, markeredgewidth=0))
    sh.text(lx + 2.5, y_head1, f"atlas RNA cells, n = {n_rna:,}", va="center", color=INK["secondary"])
    sh.fig.add_artist(Line2D([(lx + 48) / W], [1 - y_head1 / H], marker="o", linestyle="", markersize=2.6,
                             markerfacecolor=F10.SLOT[0], markeredgewidth=0))
    sh.text(lx + 49.5, y_head1, f"SCP-MS cells (protein), n = {n_scp}", va="center", color=INK["secondary"])
    sh.text(M, y_head2, "Coloured by cell type (atlas cells by the matched label)", va="center",
            color=INK["secondary"])
    for i, emb in enumerate(("pca", "harmony", "scvi")):
        xy = np.asarray(joint.obsm[f"X_umap_{emb}"])
        x0 = M + i * (pw + gap)
        for row, y0 in ((1, y1), (2, y2)):
            ax = sh.ax(x0, y0, pw, pw)
            F10.clean_axes(ax)
            if row == 1:
                ax.scatter(xy[~is_scp, 0], xy[~is_scp, 1], s=0.5, c=F10.GREY_RNA, linewidths=0, rasterized=True)
                ax.scatter(xy[is_scp, 0], xy[is_scp, 1], s=1.6, c=F10.SLOT[0], linewidths=0, rasterized=True)
                ax.set_title(SPACE[emb], fontsize=FONT, loc="left", pad=2, color=INK["primary"])
            else:
                rc = [palette.get(c, F10.NO_COUNTERPART) if not c.startswith("atlas: ") else F10.NO_COUNTERPART
                      for c in common[~is_scp]]
                ax.scatter(xy[~is_scp, 0], xy[~is_scp, 1], s=0.5, c=rc, linewidths=0, alpha=0.45, rasterized=True)
                sc_ = obs.loc[is_scp, COL].astype(str).to_numpy()
                ax.scatter(xy[is_scp, 0], xy[is_scp, 1], s=2.4, c=[palette.get(lab, INK["muted"]) for lab in sc_],
                           edgecolors=INK["primary"], linewidths=0.15, rasterized=True)
    F10.chip_legend(sh, M, y_leg, items, col_w=(W - 2 * M) / 3, ncol=3)
    T8.save_checked(run, sh, "08_joint_umaps", "S-08-a: joint UMAPs A / B / C, revised annotation", rows_out)
    return vals


# ---------------------------------------------------------------------------
# S-08-b: mixing within each cell type
# ---------------------------------------------------------------------------
def mixing_figure(run, rows_out, mix_t: pd.DataFrame, mix_g: pd.DataFrame) -> list[dict]:
    prim = mix_t[(mix_t["config"] == "primary") & (mix_t["skipped"].fillna("") == "") & (mix_t["label_column"] == COL)]
    metrics = (("scaled_other_fraction_scp", "atlas neighbours\n/ random"), ("ilisi_scib_scp", "iLISI − 1"))
    comps = L7.row_order(prim["compartment"].unique())
    left = 24.0
    pw = W - 2 * M - left
    ph, gap_in, tick_h = 24.0, 2.5, 8.0
    y_leg1, y_leg2 = M + 1.5, M + 5.5
    y0 = M + 9.5
    H = y0 + 2 * ph + gap_in + tick_h + M
    sh = T8.Sheet(H)
    gg = mix_g[mix_g["config"] == "primary"].groupby("embedding")["scaled_rna_neighbour_fraction_scp"].median()
    lx = M
    for emb in ("pca", "harmony"):
        sh.rect(lx, y_leg1 - 1.2, 2.4, 2.4, facecolor=F10.EMB_COLOR[emb], edgecolor="none")
        sh.text(lx + 3.2, y_leg1, SPACE[emb], va="center", color=INK["secondary"])
        lx += 3.2 + tw(SPACE[emb]) + 6.0
    sh.fig.add_artist(Line2D([lx / W, (lx + 5) / W], [1 - y_leg1 / H] * 2, color=INK["muted"], linewidth=0.6,
                             linestyle=(0, (3, 2))))
    sh.text(lx + 6.5, y_leg1, "random mixing (upper)", va="center", color=INK["secondary"])
    lx += 6.5 + tw("random mixing (upper)") + 6.0
    sh.fig.add_artist(Line2D([lx / W, (lx + 4) / W], [1 - y_leg1 / H] * 2, color=INK["primary"], linewidth=0.9))
    sh.text(lx + 5.5, y_leg1, "random mixing at the cell type's ratio (lower)", va="center", color=INK["secondary"])
    sh.text(M, y_leg2, f"All cells, atlas neighbours of SCP-MS cells / random: A {gg['pca']:.2f}, "
            f"B {gg['harmony']:.2f}. Under the bars: cell type, SCP-MS cells / atlas cells.", va="center",
            color=INK["secondary"])
    n = prim[prim["draw"] == T.PRIMARY_DRAW].groupby("compartment")[["n_scp", "n_rna"]].first()
    xs = np.arange(len(comps))
    vals = []
    for r, (metric, ylab) in enumerate(metrics):
        yy = y0 + r * (ph + gap_in)
        ax = sh.ax(M + left, yy, pw, ph)
        g = prim.groupby(["embedding", "compartment"])[metric].agg(["median", "min", "max"])
        for k, emb in enumerate(("pca", "harmony")):
            med = [g.loc[(emb, cp), "median"] for cp in comps]
            lo = [g.loc[(emb, cp), "min"] for cp in comps]
            hi = [g.loc[(emb, cp), "max"] for cp in comps]
            xx = xs + (k - 0.5) * 0.36
            ax.bar(xx, med, width=0.34, color=F10.EMB_COLOR[emb], edgecolor="white", linewidth=0.5)
            ax.vlines(xx, lo, hi, color=INK["primary"], linewidth=0.5)
            vals += [{"figure": "mixing", "metric": metric, "embedding": emb, "label": cp, "display": disp(cp),
                      "median": m_, "min": l_, "max": h_} for cp, m_, l_, h_ in zip(comps, med, lo, hi)]
        if metric == "scaled_other_fraction_scp":
            ax.axhline(1.0, color=INK["muted"], linewidth=0.6, linestyle=(0, (3, 2)))
            top = max(1.15, float(np.nanmax(g["max"].to_numpy())) * 1.08)
        else:
            perf = prim[prim["draw"] == T.PRIMARY_DRAW].groupby("compartment")["ilisi_perfect_scib"].first()
            ax.hlines([perf.loc[cp] for cp in comps], xs - 0.42, xs + 0.42, color=INK["primary"], linewidth=0.9)
            vals += [{"figure": "mixing", "metric": "ilisi_random_mixing", "embedding": "", "label": cp,
                      "display": disp(cp), "median": float(perf.loc[cp])} for cp in comps]
            top = 1.0
        ax.set_ylim(0, top)
        ax.set_xlim(-0.6, len(comps) - 0.4)
        if r == len(metrics) - 1:
            ax.set_xticks(xs, [f"{disp(cp)}\n{int(n.loc[cp, 'n_scp'])} / {int(n.loc[cp, 'n_rna'])}" for cp in comps],
                          fontsize=FONT, linespacing=1.0)
        else:
            ax.set_xticks(xs, [""] * len(xs))
        T8.style_axes(ax)
        ax.yaxis.grid(True, color=INK["grid"], linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_yticks([t for t in (0, 0.5, 1.0) if t <= top])
        ax.set_ylabel(ylab, fontsize=FONT, linespacing=1.1)
    T8.save_checked(run, sh, "08_mixing", "S-08-b: modality mixing within each cell type (A vs B), revised annotation",
                    rows_out)
    return vals


# ---------------------------------------------------------------------------
# S-08-c: label transfer in B
# ---------------------------------------------------------------------------
def transfer_figure(run, rows_out, conf: pd.DataFrame, per_label: pd.DataFrame, mapping: pd.DataFrame,
                    order: list[str], checks: list[dict]) -> list[dict]:
    space, lv = "harmony", "celltype"
    level = T.LEVELS[lv]
    sub = conf[(conf["embedding"] == space) & (conf["level"] == level) & (conf["label_column"] == COL)]
    pl = per_label[(per_label["config"] == "primary") & (per_label["draw"] == T.PRIMARY_DRAW) &
                   (per_label["embedding"] == space) & (per_label["level"] == level) &
                   (per_label["k"] == T.K_TRANSFER) & (per_label["label_column"] == COL)]
    sets = T.expected_sets(mapping, COL, lv)
    expected_all = set().union(*(mix for _, mix in sets.values()))
    cols_all = L7.atlas_order(set(sub["atlas_class"].unique()) | expected_all)   # matched classes always shown
    head = "matched / incl. mixed"
    left = max(tw(f"{disp(lab)} (000)") for lab in order) + 2.0
    right = max(tw(head), tw("0.00 / 0.00")) + 3.0
    cw = min(4.2, (W - 2 * M - left - right) / len(cols_all))
    hw = cw * len(cols_all)
    rh = 4.0
    lab_h = max(tw(c) for c in cols_all) + 2.0
    top = M + 4.0
    cb_y = top + len(order) * rh + lab_h + 2.0
    H = cb_y + 2.2 + LH_TICK + M
    sh = T8.Sheet(H)
    cmap = T8.sequential_cmap()
    cnt = sub.pivot_table(index="scp_label", columns="atlas_class", values="n", aggfunc="sum", fill_value=0)
    cnt = cnt.reindex(index=order, columns=cols_all, fill_value=0)
    sizes = cnt.sum(axis=1).astype(int)
    obs = {k: int(v) for k, v in sizes.items()}
    checks.append({"check": "S-08-c: transferred cells per label = revised-annotation counts", "observed": str(obs),
                   "expected": str(EXPECTED_SCP), "pass": obs == EXPECTED_SCP})
    frac = cnt.div(sizes.replace(0, np.nan), axis=0).fillna(0.0).to_numpy(float)
    sh.text(M + left + hw + 2.0, M + 1.5, head, va="center", color=INK["secondary"])
    ax = sh.ax(M + left, top, hw, len(order) * rh)
    ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xlim(-0.5, len(cols_all) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    vals = []
    for i, lab in enumerate(order):
        exp, mix = sets.get(lab, (set(), set()))
        for jx, ac in enumerate(cols_all):
            if ac in exp:
                ax.add_patch(Rectangle((jx - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK["primary"], linewidth=0.7))
            elif ac in (mix - exp):
                ax.add_patch(Rectangle((jx - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK["primary"],
                                       linewidth=0.6, linestyle=(0, (2, 1.5))))
            if cnt.loc[lab, ac]:
                vals.append({"figure": "transfer", "label": lab, "display": disp(lab), "atlas_class": ac,
                             "n": int(cnt.loc[lab, ac]), "fraction": float(frac[i, jx]),
                             "matched": ac in exp, "mixed": ac in (mix - exp)})
        row = pl[pl["scp_label"] == lab]
        if len(row) and np.isfinite(row["accuracy_strict"].iloc[0]):
            a_s, a_m = float(row["accuracy_strict"].iloc[0]), float(row["accuracy_incl_mixed"].iloc[0])
            txt = f"{a_s:.2f} / {a_m:.2f}"
            vals.append({"figure": "transfer_accuracy", "label": lab, "display": disp(lab), "accuracy_strict": a_s,
                         "accuracy_incl_mixed": a_m})
        else:
            txt = "–"
        sh.text(M + left + hw + 2.0, top + (i + 0.5) * rh, txt, va="center", color=INK["secondary"])
        sh.text(M + left - 1.0, top + (i + 0.5) * rh, f"{disp(lab)} ({int(sizes.loc[lab])})", va="center",
                ha="right")
    ax.set_xticks(range(len(cols_all)), cols_all, rotation=90, fontsize=FONT)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cax = sh.ax(M + left, cb_y, 40.0, 2.2)
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"], fontsize=FONT)
    cax.tick_params(length=1.5, pad=1)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color(INK["muted"])
    sh.text(M + left + 43.0, cb_y + 1.1, "fraction of the label's cells", va="center", color=INK["secondary"])
    T8.save_checked(run, sh, "08_transfer", "S-08-c: atlas Celltype transferred to SCP-MS cells in B, revised "
                    "annotation", rows_out)
    return vals


LH_TICK = T8.text_mm("0", FONT)[1] + 1.5


def main() -> None:
    args = parse_args()
    run = cc.Run(TASK, STEP, args)
    t5checks = t5.Checks(T.DATASET)
    import anndata as ad

    adata, labels = L7.load_kidney(run, t5checks)
    joint = ad.read_h5ad(run.use_input(T.JOINT_H5AD, "rna_protein_integration joint embedding (rna_protein_integration_integrate.py)"))
    mapping = T.read_mapping(run, t5checks)

    def tab(name: str) -> pd.DataFrame:
        return pd.read_csv(run.use_input(cc.out_path(T.TASK, "tables", name, ".csv"), f"rna_protein_integration {name}"))

    mix_t, mix_g = tab("mixing_per_type"), tab("mixing_global")
    conf, per_label = tab("transfer_confusion"), tab("transfer_per_label")
    order = L7.row_order(labels[COL].unique())
    palette = t15c.palette(adata, order)
    plt.rcParams.update({"font.family": "Arial", "font.size": FONT, "pdf.fonttype": 42, "ps.fonttype": 42})

    checks: list[dict] = []
    is_scp = (joint.obs["modality"] == "SCP").to_numpy()
    obs = joint.obs.loc[is_scp, COL].astype(str).value_counts().to_dict()
    checks.append({"check": "joint embedding: SCP-MS cells per revised-annotation label", "observed": str(obs),
                   "expected": str(EXPECTED_SCP), "pass": obs == EXPECTED_SCP})
    checks.append({"check": "joint embedding: SCP-MS / atlas cells (draw 42)",
                   "observed": f"{int(is_scp.sum())} / {int((~is_scp).sum())}", "expected": "712 / 6112",
                   "pass": int(is_scp.sum()) == 712 and int((~is_scp).sum()) == 6112})
    checks.append({"check": "every label has a display name or keeps its own", "observed": str(
        {lab: disp(lab) for lab in order}), "expected": "DCT1 -> DCT; PC -> CNT/PC", "pass":
        disp("DCT1") == "DCT" and disp("PC") == "CNT/PC"})

    fig_rows: list[dict] = []
    vals = umap_figure(run, fig_rows, joint, palette, order)
    vals += mixing_figure(run, fig_rows, mix_t, mix_g)
    vals += transfer_figure(run, fig_rows, conf, per_label, mapping, order, checks)
    pcts = {v["label"]: round(v["accuracy_strict"], 4) for v in vals if v["figure"] == "transfer_accuracy"}
    checks.append({"check": "S-08-c: matched fraction read from the revised-annotation rows (draw 42, B)",
                   "observed": f"PCT {pcts.get('PCT')}, PST {pcts.get('PST')}, TAL {pcts.get('TAL')}",
                   "expected": "PCT 0.5938, PST 0.462, TAL 0.0797",
                   "pass": pcts.get("PCT") == 0.5938 and pcts.get("PST") == 0.462 and pcts.get("TAL") == 0.0797})
    ok_fig = bool(pd.DataFrame(fig_rows)["pass"].all())
    checks.append({"check": "figures pass the format checks (>= 7 pt, Arial, no text overlaps, text inside, 180 mm)",
                   "observed": f"{int(pd.DataFrame(fig_rows)['pass'].sum())} of {len(fig_rows)}", "expected": "3 of 3",
                   "pass": ok_fig})
    for c in checks:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['observed']}")
    run.table(pd.DataFrame(vals), f"{STEP}_values")
    run.table(pd.DataFrame(fig_rows), f"{STEP}_figure_checks")
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    all_ok = all(c["pass"] for c in checks) and t5checks.ok
    run.finish({"label_column": COL, "display_names": DISPLAY, "space_transfer": "harmony", "level": "Celltype",
                "draw": T.PRIMARY_DRAW, "k": T.K_TRANSFER, "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit("A check failed; see report_08_integration_figures_checks.csv.")


if __name__ == "__main__":
    main()
