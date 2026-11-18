#!/usr/bin/env python
"""rna_protein_integration step 5: figures for the RNA-protein integration diagnostics.

Analysis plan,
step 5. All values are read from the rna_protein_integration tables and the joint AnnData; nothing is recomputed except layout.
Format (as kidney_rna / atlas_dct): 180 mm wide, Arial, >= 7 pt, PDF (TrueType) + 600-dpi PNG; text-overlap, text-outside,
font and width checks (atlas_dct_lib.save_checked) recorded in rna_protein_integration_figure_checks.csv.

Figures (results/figures/rna_protein_integration/)
-----------------------------------------
- rna_protein_integration_umap_{t15,pre}: UMAPs of embeddings A, B and C (primary configuration, draw 42); top row coloured by
  modality, bottom row by cell type (SCP cells: their label; RNA cells: the SCP label their atlas Celltype maps to;
  grey: atlas classes without an SCP counterpart).
- rna_protein_integration_mixing_bars: modality mixing within each cell type (compartment), A vs B: RNA-neighbour fraction of
  the SCP cells divided by its value under random mixing; iLISI - 1 of the SCP cells with the random-mixing value;
  bar = median over the 5 draws, whisker = range.
- rna_protein_integration_transfer_confusion_{celltype,finest}_{pca,harmony,scvi}: SCP label x predicted atlas class (kNN
  k = 15, draw 42), fill = fraction of the label's cells; outlined = expected class, dashed = accepted mixed class;
  right column = accuracy strict / incl. mixed.
- rna_protein_integration_transfer_summary: accuracy (strict) and macro-F1 per embedding, label set and atlas level (median and
  range over the 5 draws; chance accuracy marked).
- rna_protein_integration_strengths_t15: per kidney_annotation compartment and marker, detection in RNA and protein at 10% (kidney_rna tables);
  strip = kidney_rna class across compartments; triangles = transporter / channel (prefix rule); bars = kidney_rna classes of
  the transporter / channel features vs the other shared features.

Inputs (read-only): rna_protein_integration_joint_embedding.h5ad; rna_protein_integration tables (mixing_per_type, mixing_global, metrics_summary,
  transfer_confusion, transfer_per_label, transfer_summary, strengths_markers, strengths_classes); kidney_rna approved
  mapping; dotplot label order (Fig 2A colours); kidney_object canonical kidney object (kidney_annotation colours via kidney_annotation_chosen_labels).
Outputs: the figures above and results/tables/rna_protein_integration/rna_protein_integration_figure_checks.csv
Command (project root):  python scripts/rna_protein_integration_figures.py
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
import rna_protein_integration_lib as T
import kidney_annotation_chosen_labels as t15c     # kidney_annotation palette (imports set styles; Arial is set again below)
import atlas_dct_lib as T8

plt = T8.setup_matplotlib()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

INK = T8.INK
W, M, FONT = T8.W, T8.M, T8.FONT
SLOT = T8.SERIES                     # reference categorical order (slots 1-3 used)
GREY_RNA = "#d6d5cf"                 # = INK background_cells
NO_COUNTERPART = "#e1e0d9"
EMB_COLOR = {"pca": SLOT[0], "harmony": SLOT[1]}
LABEL_SET_COLOR = {"kidney_t15_label": SLOT[0], "kidney_pre_label": SLOT[1]}
DET_COLORS = {"detected in both": SLOT[0], "RNA only": SLOT[1], "protein only": SLOT[2],
              "detected in neither": "#e1e0d9", "not in atlas": "#ffffff"}
KIDNEY_RNA_COLORS = {"Detected in both; concordant": "#184f95", "Detected in both; intermediate": "#3987e5",
              "Detected in both; discordant": "#86b6ef", "RNA-detected only": SLOT[1],
              "Protein-detected only": SLOT[2], "Detected in neither": "#e1e0d9", "not in atlas": "#ffffff"}
KIDNEY_RNA_SHORT = {"Detected in both; concordant": "concordant (rho >= 0.5)",
             "Detected in both; intermediate": "intermediate (0.2-0.5)",
             "Detected in both; discordant": "discordant (< 0.2)", "RNA-detected only": "RNA only",
             "Protein-detected only": "protein only", "Detected in neither": "neither", "not in atlas": "not in atlas"}
SPACE_TEXT = {"pca": "A: joint PCA", "harmony": "B: PCA + Harmony", "scvi": "C: atlas scVI (SCP projected)",
              "shared": "shared-feature kNN (cosine)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rna_protein_integration step 5: figures.")
    cc.add_common_args(parser)
    return parser.parse_args()


def clean_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color(INK["grid"])


def chip_legend(sh: T8.Sheet, x: float, y: float, items: list[tuple[str, str]], col_w: float, ncol: int,
                row_h: float = 3.4, edge: str | None = None, hatch_white: bool = True) -> float:
    """Colour chips with labels in a grid (mm); returns the height used."""
    for i, (color, text) in enumerate(items):
        cx, cy = x + (i % ncol) * col_w, y + (i // ncol) * row_h
        sh.rect(cx, cy + 0.4, 2.2, 2.2, facecolor=color, edgecolor=edge or (INK["muted"] if color == "#ffffff" else
                                                                             "none"), linewidth=0.4,
                hatch="////" if (color == "#ffffff" and hatch_white) else None)
        sh.text(cx + 3.0, cy + 1.5, text, va="center", color=INK["secondary"])
    return (len(items) + ncol - 1) // ncol * row_h


# ---------------------------------------------------------------------------
# 1. UMAPs
# ---------------------------------------------------------------------------
def umap_figure(run, rows_out, joint, col: str, palette: dict[str, str], order: list[str]) -> None:
    obs = joint.obs
    is_scp = (obs["modality"] == "SCP").to_numpy()
    common = obs[f"common_{T.LABEL_SHORT[col]}"].astype(str).to_numpy()
    n_rna = int((~is_scp).sum())
    gap, pw = 5.0, (W - 2 * M - 2 * 5.0) / 3
    y_title, y_head1 = M + 2.0, M + 8.0
    y1 = y_head1 + 4.0
    y_head2 = y1 + pw + 3.0
    y2 = y_head2 + 4.0
    y_leg = y2 + pw + 3.0
    labs = [l for l in order if l in set(obs.loc[is_scp, col])]
    items = []
    for l in labs:
        n_s = int(((obs[col] == l).to_numpy() & is_scp).sum())
        n_r = int(((common == l) & ~is_scp).sum())
        items.append((palette.get(l, INK["muted"]), f"{l}: SCP {n_s}, RNA {n_r}"))
    items.append((NO_COUNTERPART, f"RNA, no SCP counterpart: {int(((~is_scp) & pd.Series(common).str.startswith('atlas: ').to_numpy()).sum())}"))
    leg_h = (len(items) + 2) // 3 * 3.4
    note = (f"Primary configuration, RNA draw 42 (6,112 atlas cells, 200 per atlas Celltype; smaller classes whole) and "
            f"712 SCP cells; 3,569 shared features, each modality z-scored separately. A: 30 PCs from the average of the "
            f"two within-modality covariance matrices. B: Harmony (harmonypy 0.0.10) on A with modality as the batch. "
            f"C: RNA cells at the atlas's stored X_scVI; each SCP cell at the mean X_scVI of its 15 nearest RNA cells in "
            f"shared-feature space (its position follows from that kNN by construction). UMAP: k = 15, seed 42. "
            f"Bottom row: SCP cells outlined; RNA cells coloured by the SCP label their atlas Celltype maps to "
            f"(approved kidney_rna mapping). RNA and protein come from different animals and studies (unpaired data).")
    note_txt, note_h = T8.note_block(note)
    H = y_leg + leg_h + 3.0 + note_h + M
    sh = T8.Sheet(H)
    sh.text(M, y_title, f"Joint embeddings of SCP and atlas RNA cells: {T.LABEL_DISPLAY[col]}", fontsize=T8.TITLE,
            fontweight="bold", va="center")
    sh.text(M, y_head1, "Coloured by modality", va="center", color=INK["secondary"])
    # modality legend on the header line (right)
    lx = M + 36
    sh.fig.add_artist(Line2D([(lx + 1) / W], [1 - y_head1 / H], marker="o", linestyle="", markersize=2.6,
                             markerfacecolor=GREY_RNA, markeredgewidth=0))
    sh.text(lx + 2.5, y_head1, f"RNA (atlas cells), n = {n_rna:,}", va="center", color=INK["secondary"])
    sh.fig.add_artist(Line2D([(lx + 48) / W], [1 - y_head1 / H], marker="o", linestyle="", markersize=2.6,
                             markerfacecolor=SLOT[0], markeredgewidth=0))
    sh.text(lx + 49.5, y_head1, f"SCP (protein), n = {int(is_scp.sum())}", va="center", color=INK["secondary"])
    sh.text(M, y_head2, "Coloured by cell type", va="center", color=INK["secondary"])
    for i, emb in enumerate(("pca", "harmony", "scvi")):
        xy = np.asarray(joint.obsm[f"X_umap_{emb}"])
        x0 = M + i * (pw + gap)
        for row, y0 in ((1, y1), (2, y2)):
            ax = sh.ax(x0, y0, pw, pw)
            clean_axes(ax)
            if row == 1:
                ax.scatter(xy[~is_scp, 0], xy[~is_scp, 1], s=0.5, c=GREY_RNA, linewidths=0, rasterized=True)
                ax.scatter(xy[is_scp, 0], xy[is_scp, 1], s=1.6, c=SLOT[0], linewidths=0, rasterized=True)
                ax.set_title(SPACE_TEXT[emb], fontsize=FONT, loc="left", pad=2, color=INK["primary"])
            else:
                rc = np.array([palette.get(c, NO_COUNTERPART) if not c.startswith("atlas: ") else NO_COUNTERPART
                               for c in common[~is_scp]], dtype=object)
                ax.scatter(xy[~is_scp, 0], xy[~is_scp, 1], s=0.5, c=list(rc), linewidths=0, alpha=0.45,
                           rasterized=True)
                sc_ = obs.loc[is_scp, col].astype(str).to_numpy()
                ax.scatter(xy[is_scp, 0], xy[is_scp, 1], s=2.4, c=[palette.get(l, INK["muted"]) for l in sc_],
                           edgecolors=INK["primary"], linewidths=0.15, rasterized=True)
    chip_legend(sh, M, y_leg, items, col_w=(W - 2 * M) / 3, ncol=3)
    sh.note(note_txt, note_h)
    T8.save_checked(run, sh, f"umap_{T.LABEL_SHORT[col]}", f"rna_protein_integration joint UMAPs ({col})", rows_out)


# ---------------------------------------------------------------------------
# 2. Mixing bars
# ---------------------------------------------------------------------------
def mixing_figure(run, rows_out, mix_t: pd.DataFrame, mix_g: pd.DataFrame) -> None:
    prim = mix_t[(mix_t["config"] == "primary") & (mix_t["skipped"].fillna("") == "")]
    metrics = (("scaled_other_fraction_scp", "RNA neighbours\n/ random"),
               ("ilisi_scib_scp", "iLISI - 1"))
    cols = list(T.LABEL_COLUMNS)
    comps = {c: L7.row_order(prim.loc[prim["label_column"] == c, "compartment"].unique()) for c in cols}
    left = 24.0
    pw = W - 2 * M - left
    ph, gap_in, tick_h, gap_out = 24.0, 2.5, 8.0, 5.0
    y0 = M + 12.0
    block_h = 4.5 + 2 * ph + gap_in + tick_h + gap_out
    note = ("Compartment = SCP cells of one label + atlas RNA cells of its expected Celltypes (approved kidney_rna mapping). "
            "Within each compartment, kNN (k = 15) among its cells only. Upper panel of each label set: mean fraction "
            "of RNA cells among the 15 neighbours of an SCP cell, divided by its value under random mixing (RNA cells "
            "/ (cells - 1)); 0 = only SCP neighbours, 1 = random mixing. Lower panel: median iLISI (perplexity 5) of "
            "the SCP cells minus 1 (scib scale for two modalities: 0 = one modality, 1 = equal shares); black tick = "
            "value for random mixing at the compartment's modality shares. Bars: median over the 5 RNA draws (seeds "
            "42-46); whiskers: range. Under the bars: SCP cells / RNA cells (draw 42). Embedding C is not shown: its "
            "SCP positions are averages of RNA positions by construction.")
    note_txt, note_h = T8.note_block(note)
    H = y0 + len(cols) * block_h + note_h + M
    sh = T8.Sheet(H)
    gg = mix_g[(mix_g["config"] == "primary")].groupby("embedding")["scaled_rna_neighbour_fraction_scp"].median()
    sh.text(M, M + 2.0, "Modality mixing within each cell type (primary configuration)", fontsize=T8.TITLE,
            fontweight="bold", va="center")
    sh.text(W - M, M + 2.0, f"all cells: RNA-neighbour fraction of SCP cells / random, A {gg['pca']:.2f}, "
            f"B {gg['harmony']:.2f}", ha="right", va="center", color=INK["secondary"])
    # legend
    lx = M
    for emb in ("pca", "harmony"):
        sh.rect(lx, M + 6.0, 2.4, 2.4, facecolor=EMB_COLOR[emb], edgecolor="none")
        sh.text(lx + 3.2, M + 7.2, SPACE_TEXT[emb], va="center", color=INK["secondary"])
        lx += 40
    sh.fig.add_artist(Line2D([(lx) / W, (lx + 5) / W], [1 - (M + 7.2) / H] * 2, color=INK["muted"], linewidth=0.6,
                             linestyle=(0, (3, 2))))
    sh.text(lx + 6.5, M + 7.2, "random mixing (upper)", va="center", color=INK["secondary"])
    lx += 34
    sh.fig.add_artist(Line2D([(lx) / W, (lx + 4) / W], [1 - (M + 7.2) / H] * 2, color=INK["primary"], linewidth=0.9))
    sh.text(lx + 5.5, M + 7.2, "random mixing at the compartment's shares (lower)", va="center",
            color=INK["secondary"])
    for b, c in enumerate(cols):
        yb = y0 + b * block_h
        sh.text(M, yb + 1.5, "kidney_annotation labels" if c == "kidney_t15_label" else "Fig 2A labels", va="center",
                fontweight="bold")
        sub = prim[prim["label_column"] == c]
        n = sub[sub["draw"] == T.PRIMARY_DRAW].groupby("compartment")[["n_scp", "n_rna"]].first()
        xs = np.arange(len(comps[c]))
        for r, (metric, ylab) in enumerate(metrics):
            yy = yb + 4.5 + r * (ph + gap_in)
            ax = sh.ax(M + left, yy, pw, ph)
            g = sub.groupby(["embedding", "compartment"])[metric].agg(["median", "min", "max"])
            for k, emb in enumerate(("pca", "harmony")):
                med = [g.loc[(emb, cp), "median"] for cp in comps[c]]
                lo = [g.loc[(emb, cp), "min"] for cp in comps[c]]
                hi = [g.loc[(emb, cp), "max"] for cp in comps[c]]
                xx = xs + (k - 0.5) * 0.36
                ax.bar(xx, med, width=0.34, color=EMB_COLOR[emb], edgecolor="white", linewidth=0.5)
                ax.vlines(xx, lo, hi, color=INK["primary"], linewidth=0.5)
            if metric == "scaled_other_fraction_scp":
                ax.axhline(1.0, color=INK["muted"], linewidth=0.6, linestyle=(0, (3, 2)))
                top = max(1.15, float(np.nanmax(g["max"].to_numpy())) * 1.08)
            else:
                perf = sub[sub["draw"] == T.PRIMARY_DRAW].groupby("compartment")["ilisi_perfect_scib"].first()
                ax.hlines([perf.loc[cp] for cp in comps[c]], xs - 0.42, xs + 0.42, color=INK["primary"],
                          linewidth=0.9)
                top = 1.0
            ax.set_ylim(0, top)
            ax.set_xlim(-0.6, len(comps[c]) - 0.4)
            if r == len(metrics) - 1:
                ax.set_xticks(xs, [f"{cp}\n{int(n.loc[cp, 'n_scp'])} / {int(n.loc[cp, 'n_rna'])}" for cp in comps[c]],
                              fontsize=FONT, linespacing=1.0)
            else:
                ax.set_xticks(xs, [""] * len(xs))
            T8.style_axes(ax)
            ax.yaxis.grid(True, color=INK["grid"], linewidth=0.4)
            ax.set_axisbelow(True)
            ax.set_yticks([t for t in (0, 0.5, 1.0) if t <= top])
            ax.set_ylabel(ylab, fontsize=FONT, linespacing=1.1)
    sh.note(note_txt, note_h)
    T8.save_checked(run, sh, "mixing_bars", "rna_protein_integration modality mixing within each cell type (A vs B)", rows_out)


# ---------------------------------------------------------------------------
# 3. Confusion heatmaps
# ---------------------------------------------------------------------------
def confusion_figure(run, rows_out, conf: pd.DataFrame, per_label: pd.DataFrame, mapping: pd.DataFrame,
                     labels: pd.DataFrame, space: str, lv: str) -> None:
    level = T.LEVELS[lv]
    sub = conf[(conf["embedding"] == space) & (conf["level"] == level)]
    pl = per_label[(per_label["config"] == "primary") & (per_label["draw"] == T.PRIMARY_DRAW) &
                   (per_label["embedding"] == space) & (per_label["level"] == level) & (per_label["k"] == T.K_TRANSFER)]
    expected_all = set()
    for c in T.LABEL_COLUMNS:
        for exp, mix in T.expected_sets(mapping, c, lv).values():
            expected_all |= mix
    cols_all = L7.atlas_order(set(sub["atlas_class"].unique()) | expected_all)   # expected classes always shown
    rows = {c: L7.row_order(labels[c].unique()) for c in T.LABEL_COLUMNS}
    left, right = 24.0, 17.0
    cw = min(4.2, (W - 2 * M - left - right) / len(cols_all))
    rh = 4.0
    lab_h = max(T8.text_mm(c)[0] for c in cols_all) + 2.0
    y = M + 6.0
    blocks = []
    for c in T.LABEL_COLUMNS:
        blocks.append((c, y))
        y += 5.0 + len(rows[c]) * rh + lab_h + 4.0
    note = (f"kNN label transfer from the RNA draw 42 to the 712 SCP cells in {SPACE_TEXT[space]} (k = 15; majority of "
            f"the nearest RNA cells; reference = RNA cells{' with a known Celltype_finest' if lv == 'finest' else ''}). "
            "Fill: fraction of the SCP label's cells predicted as each atlas class. Solid outline: expected class of "
            "the label; dashed: accepted mixed class (approved kidney_rna mapping). Right: accuracy strict / incl. mixed "
            "(this draw); '-' = no atlas counterpart. Counts per cell: rna_protein_integration_transfer_confusion.csv.")
    note_txt, note_h = T8.note_block(note)
    cb_h = 9.0
    H = y + cb_h + note_h + M
    sh = T8.Sheet(H)
    sh.text(M, M + 2.0, f"SCP label x predicted atlas {level}: {SPACE_TEXT[space]}", fontsize=T8.TITLE,
            fontweight="bold", va="center")
    cmap = T8.sequential_cmap()
    hw = cw * len(cols_all)
    for c, y0 in blocks:
        sets = T.expected_sets(mapping, c, lv)
        rr = rows[c]
        s = sub[sub["label_column"] == c]
        cnt = s.pivot_table(index="scp_label", columns="atlas_class", values="n", aggfunc="sum", fill_value=0)
        cnt = cnt.reindex(index=rr, columns=cols_all, fill_value=0)
        frac = cnt.div(cnt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).to_numpy(float)
        sh.text(M, y0 + 1.5, f"{'kidney_annotation labels' if c == 'kidney_t15_label' else 'Fig 2A labels'}", va="center",
                fontweight="bold")
        sh.text(M + left + hw + 2.0, y0 + 1.5, "strict / mixed", va="center", color=INK["secondary"])
        top = y0 + 4.0
        ax = sh.ax(M + left, top, hw, len(rr) * rh)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xlim(-0.5, len(cols_all) - 0.5)
        ax.set_ylim(len(rr) - 0.5, -0.5)
        for i, lab in enumerate(rr):
            exp, mix = sets.get(lab, (set(), set()))
            for jx, ac in enumerate(cols_all):
                if ac in exp:
                    ax.add_patch(Rectangle((jx - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK["primary"],
                                           linewidth=0.7))
                elif ac in (mix - exp):
                    ax.add_patch(Rectangle((jx - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK["primary"],
                                           linewidth=0.6, linestyle=(0, (2, 1.5))))
            row = pl[(pl["label_column"] == c) & (pl["scp_label"] == lab)]
            if len(row) and np.isfinite(row["accuracy_strict"].iloc[0]):
                txt = f"{row['accuracy_strict'].iloc[0]:.2f} / {row['accuracy_incl_mixed'].iloc[0]:.2f}"
            else:
                txt = "-"
            sh.text(M + left + hw + 2.0, top + (i + 0.5) * rh, txt, va="center", color=INK["secondary"])
            sh.text(M + left - 1.0, top + (i + 0.5) * rh, f"{lab} ({int(cnt.loc[lab].sum())})", va="center",
                    ha="right")
        ax.set_xticks(range(len(cols_all)), cols_all, rotation=90, fontsize=FONT)
        ax.set_yticks([])
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    # colour bar
    cax = sh.ax(M + left, y + 1.0, 40.0, 2.2)
    grad = np.linspace(0, 1, 256)[None, :]
    cax.imshow(grad, cmap=cmap, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"], fontsize=FONT)
    cax.tick_params(length=1.5, pad=1)
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color(INK["muted"])
    sh.text(M + left + 43.0, y + 2.1, "fraction of the SCP label's cells", va="center", color=INK["secondary"])
    sh.note(note_txt, note_h)
    T8.save_checked(run, sh, f"transfer_confusion_{lv}_{space}", f"rna_protein_integration transfer confusion ({level}, {space})", rows_out)


# ---------------------------------------------------------------------------
# 4. Transfer summary
# ---------------------------------------------------------------------------
def transfer_summary_figure(run, rows_out, ms: pd.DataFrame) -> None:
    tr = ms[(ms["metric_group"] == "transfer") & (ms["config"] == "primary") & (ms["k"] == T.K_TRANSFER)]
    spaces = ["pca", "harmony", "scvi", "shared"]
    left, gapx, gapy = 42.0, 8.0, 10.0
    pw = (W - 2 * M - left - gapx) / 2
    ph = 26.0
    y0 = M + 12.0
    note = ("kNN label transfer (k = 15) from the RNA draws to the 712 SCP cells, primary configuration. Dot: median over "
            "the 5 RNA draws; line: range. Accuracy strict: prediction among the label's expected atlas classes "
            "(approved kidney_rna mapping), over SCP cells of labels with a counterpart (kidney_annotation: 579 cells, 7 labels; Fig 2A: "
            "666 cells, 8 labels). Vertical tick: chance accuracy from the reference class frequencies. Macro-F1: "
            "predictions mapped back to SCP labels (classes without a counterpart = 'other'), averaged over labels. "
            "Shared-feature kNN: transfer directly in the 3,569-feature space (cosine), the basis of C.")
    note_txt, note_h = T8.note_block(note)
    H = y0 + 2 * ph + gapy + 9.0 + note_h + M
    sh = T8.Sheet(H)
    sh.text(M, M + 2.0, "kNN label transfer from atlas RNA to SCP cells", fontsize=T8.TITLE, fontweight="bold",
            va="center")
    lx = M
    for col in T.LABEL_COLUMNS:
        sh.fig.add_artist(Line2D([(lx + 1.2) / W], [1 - (M + 7.2) / H], marker="o", linestyle="", markersize=3.2,
                                 markerfacecolor=LABEL_SET_COLOR[col], markeredgewidth=0))
        sh.text(lx + 3.0, M + 7.2, T.LABEL_DISPLAY[col], va="center", color=INK["secondary"])
        lx += 62
    sh.fig.add_artist(Line2D([(lx + 1.2) / W], [1 - (M + 7.2) / H], marker="|", linestyle="", markersize=4.5,
                             markeredgecolor=INK["muted"], markeredgewidth=0.8))
    sh.text(lx + 3.0, M + 7.2, "chance accuracy", va="center", color=INK["secondary"])
    for r, (metric, mlab) in enumerate((("accuracy_strict", "Accuracy (strict)"), ("macro_f1", "Macro-F1"))):
        for cix, level in enumerate(("Celltype", "Celltype_finest")):
            x = M + left + cix * (pw + gapx)
            y = y0 + r * (ph + gapy)
            ax = sh.ax(x, y, pw, ph)
            for si, sp in enumerate(spaces):
                for li, col in enumerate(T.LABEL_COLUMNS):
                    rr = tr[(tr["embedding"] == sp) & (tr["label_column"] == col) & (tr["level"] == level)]
                    if not len(rr):
                        continue
                    rr = rr.iloc[0]
                    yy = si + (li - 0.5) * 0.36
                    ax.hlines(yy, rr[f"{metric}_min"], rr[f"{metric}_max"], color=LABEL_SET_COLOR[col], linewidth=1.0)
                    ax.plot(rr[f"{metric}_median"], yy, "o", markersize=3.2, color=LABEL_SET_COLOR[col],
                            markeredgewidth=0)
                    if metric == "accuracy_strict":
                        ax.plot(rr["chance_accuracy_strict_median"], yy, "|", markersize=4.5, color=INK["muted"],
                                markeredgewidth=0.8)
            ax.set_ylim(len(spaces) - 0.5, -0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
            ax.set_yticks(range(len(spaces)), [SPACE_TEXT[s] for s in spaces] if cix == 0 else [""] * len(spaces))
            if cix != 0:
                ax.tick_params(axis="y", length=0)
            T8.style_axes(ax)
            ax.xaxis.grid(True, color=INK["grid"], linewidth=0.4)
            ax.set_axisbelow(True)
            ax.set_title(f"{mlab}, atlas {level}", fontsize=FONT, loc="left", pad=2)
    sh.note(note_txt, note_h)
    T8.save_checked(run, sh, "transfer_summary", "rna_protein_integration kNN transfer accuracy and macro-F1", rows_out)


# ---------------------------------------------------------------------------
# 5. Strengths (kidney_annotation)
# ---------------------------------------------------------------------------
def strengths_figure(run, rows_out, mk: pd.DataFrame, classes: pd.DataFrame, palette: dict[str, str]) -> None:
    p = mk[(mk["label_column"] == "kidney_t15_label") & (mk["rna_source"] == "all")]
    markers = p.drop_duplicates("marker").sort_values("marker_position")
    comps = L7.row_order(p["compartment"].unique())
    left = 22.0
    cw = (W - 2 * M - left) / len(markers)
    rh = 3.6
    name_h = max(T8.text_mm(m)[0] for m in markers["marker"]) + 2.0
    y_chip = M + 7.0
    y_strip = y_chip + 3.2
    y_tri = y_strip + 3.4
    y_mat = y_tri + 3.0
    y_names = y_mat + len(comps) * rh
    y_leg = y_names + name_h + 2.0
    det_items = [(DET_COLORS[k], k) for k in ("detected in both", "RNA only", "protein only", "detected in neither",
                                             "not in atlas")]
    t07_items = [(KIDNEY_RNA_COLORS[k], KIDNEY_RNA_SHORT[k]) for k in KIDNEY_RNA_COLORS]
    y_bars = y_leg + 21.0
    bar_h = 4.0
    note = ("Per kidney_annotation compartment (SCP cells of the label; RNA = atlas cells of its expected Celltypes, all cells and nuclei) "
            "and marker (kidney V3 + kidney_annotation endothelial row, quantified as protein; kidney_annotation dotplot order, grouped by the type "
            "it is placed under): detected = fraction of cells with counts > 0 (RNA) or intensity > 0 (protein) >= 10% "
            "in that compartment (kidney_rna step 5). Strip: kidney_rna class across the 7 compartments (Spearman rho of RNA vs "
            "protein means). Triangles: transporter / channel by a gene-symbol prefix rule (Slc, Atp, Abc, Aqp, Kcn, "
            "Scn, Clcn, Trp channels, ...; rna_protein_integration_transporter_channel_rule.csv; not a curated annotation). Bars: "
            "kidney_rna classes of the 3,569 shared features. Unpaired data (different animals and studies).")
    note_txt, note_h = T8.note_block(note)
    H = y_bars + 2 * (bar_h + 1.6) + 6.0 + note_h + M
    sh = T8.Sheet(H)
    sh.text(M, M + 2.0, "Marker detection in RNA (atlas) and protein (SCP) per kidney_annotation cell type", fontsize=T8.TITLE,
            fontweight="bold", va="center")
    x0 = M + left
    groups = markers["marker_group"].astype(str).tolist()
    for j, (m, g) in enumerate(zip(markers["marker"], groups)):
        x = x0 + j * cw
        sh.rect(x + 0.15, y_chip, cw - 0.3, 2.2, facecolor=palette.get(g, INK["muted"]), edgecolor="none")
        cls = str(p.loc[p["marker"] == m, "t07_class_across_compartments"].iloc[0])
        sh.rect(x + 0.15, y_strip, cw - 0.3, 2.6, facecolor=KIDNEY_RNA_COLORS.get(cls, "#ffffff"),
                edgecolor=INK["muted"] if cls == "not in atlas" else "none", linewidth=0.3,
                hatch="////" if cls == "not in atlas" else None)
        if bool(p.loc[p["marker"] == m, "transporter_channel"].iloc[0]):
            sh.fig.add_artist(Line2D([(x + cw / 2) / W], [1 - (y_tri + 1.2) / H], marker="^", linestyle="",
                                     markersize=2.6, markerfacecolor=INK["primary"], markeredgewidth=0))
        for i, cp in enumerate(comps):
            r = p[(p["marker"] == m) & (p["compartment"] == cp)]
            k = str(r["detection_class"].iloc[0]) if len(r) else "not in atlas"
            sh.rect(x + 0.15, y_mat + i * rh + 0.15, cw - 0.3, rh - 0.3, facecolor=DET_COLORS[k],
                    edgecolor=INK["muted"] if k == "not in atlas" else "none", linewidth=0.3,
                    hatch="////" if k == "not in atlas" else None)
        sh.text(x + cw / 2, y_names + 1.0, m, rotation=90, ha="center", va="top", color=INK["primary"])
    sh.text(x0 - 1.0, y_chip + 1.1, "placed under", ha="right", va="center", color=INK["secondary"])
    sh.text(x0 - 1.0, y_strip + 1.3, "kidney_rna class", ha="right", va="center", color=INK["secondary"])
    sh.text(x0 - 1.0, y_tri + 1.2, "transp. / channel", ha="right", va="center", color=INK["secondary"])
    for i, cp in enumerate(comps):
        sh.text(x0 - 1.0, y_mat + (i + 0.5) * rh, cp, ha="right", va="center")
        sh.rect(x0 - 1.0 - T8.text_mm(cp)[0] - 3.2, y_mat + i * rh + 0.7, 2.2, 2.2,
                facecolor=palette.get(cp, INK["muted"]), edgecolor="none")
    sh.text(M, y_leg + 1.5, "Cell: detection in the compartment (>= 10%)", va="center", fontweight="bold")
    chip_legend(sh, M, y_leg + 3.4, det_items, col_w=(W - 2 * M) / 5, ncol=5)
    sh.text(M, y_leg + 9.3, "Strip and bars: kidney_rna class across compartments", va="center", fontweight="bold")
    chip_legend(sh, M, y_leg + 11.2, t07_items[:6], col_w=(W - 2 * M) / 3, ncol=3)
    # bars
    c = classes[(classes["label_column"] == "kidney_t15_label") & (classes["rna_source"] == "all") &
                np.isclose(classes["threshold"], 0.10)].set_index("feature_group")
    bl = 42.0
    bw = W - 2 * M - bl
    for b, grp in enumerate(("transporter / channel", "other shared features")):
        yb = y_bars + b * (bar_h + 1.6)
        sh.text(M + bl - 1.0, yb + bar_h / 2, f"{grp} ({int(c.loc[grp, 'n_features']):,})", ha="right", va="center")
        xx = M + bl
        for k in list(KIDNEY_RNA_COLORS)[:6]:
            f = float(c.loc[grp, f"fraction_{k}"])
            wseg = f * bw
            sh.rect(xx, yb, max(wseg - 0.25, 0), bar_h, facecolor=KIDNEY_RNA_COLORS[k], edgecolor="none")
            if f >= 0.05:
                dark = k in ("Detected in both; concordant", "Detected in both; intermediate")
                sh.text(xx + wseg / 2, yb + bar_h / 2, f"{100 * f:.0f}%", ha="center", va="center",
                        color="white" if dark else INK["primary"])
            xx += wseg
    sh.text(M + bl, y_bars + 2 * (bar_h + 1.6) + 1.5, "share of features (kidney_annotation labels, all atlas cells, 10%)",
            va="center", color=INK["secondary"])
    sh.note(note_txt, note_h)
    T8.save_checked(run, sh, "strengths_t15", "rna_protein_integration RNA vs protein marker detection per kidney_annotation cell type", rows_out)


def main() -> None:
    args = parse_args()
    run = cc.Run(T.TASK, "figures", args)
    checks = t5.Checks(T.DATASET)
    import anndata as ad

    adata, labels = L7.load_kidney(run, checks)
    joint = ad.read_h5ad(run.use_input(T.JOINT_H5AD, "rna_protein_integration joint embedding (rna_protein_integration_integrate.py)"))
    mapping = T.read_mapping(run, checks)

    def tab(name: str) -> pd.DataFrame:
        return pd.read_csv(run.use_input(cc.out_path(T.TASK, "tables", name, ".csv"), f"rna_protein_integration {name}"))

    mix_t, mix_g, ms = tab("mixing_per_type"), tab("mixing_global"), tab("metrics_summary")
    conf, per_label = tab("transfer_confusion"), tab("transfer_per_label")
    mk, classes = tab("strengths_markers"), tab("strengths_classes")
    t06_lo = pd.read_csv(run.use_input(cc.OUT_ROOTS["tables"] / "dotplot" / "dotplot_label_order.csv",
                                       "dotplot label order (Fig 2A colours)"))
    t15_rows = L7.row_order(labels["kidney_t15_label"].unique())
    pal_t15 = t15c.palette(adata, t15_rows)
    lo = t06_lo[t06_lo["label_column"] == "kidney_pre_label"]
    pal_pre = dict(zip(lo["label"], lo["colour"]))
    pal_pre.setdefault("Unknown", cc.UNKNOWN_COLOR)
    plt.rcParams.update({"font.family": "Arial", "font.size": FONT, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig_rows: list[dict] = []
    umap_figure(run, fig_rows, joint, "kidney_t15_label", pal_t15, L7.row_order(labels["kidney_t15_label"].unique()))
    umap_figure(run, fig_rows, joint, "kidney_pre_label", pal_pre, L7.row_order(labels["kidney_pre_label"].unique()))
    mixing_figure(run, fig_rows, mix_t, mix_g)
    for lv in T.LEVELS:
        for space in ("harmony", "pca", "scvi"):
            confusion_figure(run, fig_rows, conf, per_label, mapping, labels, space, lv)
    transfer_summary_figure(run, fig_rows, ms)
    strengths_figure(run, fig_rows, mk, classes, pal_t15)
    figs = pd.DataFrame(fig_rows)
    run.table(figs, "figure_checks")
    ok = bool(figs["pass"].all())
    checks.add("every figure passes the format checks (>= 7 pt, Arial, no text overlaps, text inside, 180 mm)",
               f"{int(figs['pass'].sum())} of {len(figs)}", ok)
    run.finish({"figures": figs["figure"].tolist(), "format_checks_pass": ok, "checks_pass": checks.ok},
               status="completed" if ok else "completed_with_failed_format_checks")
    if not ok:
        raise SystemExit("A figure failed the format checks; see rna_protein_integration_figure_checks.csv.")


if __name__ == "__main__":
    main()
