"""dotplot shared definitions and functions. Runs no analysis by itself.

Imported by the
``dotplot_*.py`` step scripts.

Contents (rules agreed with the authors on 14 Oct 2026; layout modelled on
SISKA 1.0 Fig 1c, the reviewer's reference, as shown by the authors)
---------------------------------------------------------------------
- Row (y-axis) orders:
    kidney: DVR, JG, then the nephron (PCT ... IC-B); pre-annotation labels in
    the same order with Unknown last; DCT: DCT1, DCT2, ProLIF (the 4 Unknown
    cells of ``v1_cluster_label`` are not plotted).
- Marker placement ("sort the proteins based on the cell types they are
  present in"): each workbook marker present in the matrix is placed once,
  under the type with the highest group mean z among the row types that list
  it in the workbook, on the primary labels (kidney: ``kidney_reannot_label``;
  DCT workbook version: ``v1_cell_label``). Column groups follow the row order,
  so the plot reads as a diagonal; within a group, markers keep the assigned
  type's workbook order. Markers absent from the matrix are not plotted
  (coverage table; recorded under the first row type listing them). Types
  without cells (AVR, Podo) are not plotted. The pre-annotation plots use the
  same columns as the re-annotated plots.
- Every row type gets a column group (authors, 14 Oct 2026): a type left without
  a placed marker (all its quantified listed markers sit higher in another
  listing type) gets its listed quantified marker with the highest mean z in
  that type (ties: detection) repeated under its own bracket. Repeated
  columns are flagged (``repeat``) and marked with a dagger in the figures.
- Main kidney version: per type, among its placed markers (re-annotated labels):
    detection   = fraction of the type's cells with intensity > 0;
    specificity = type mean z (canonical ``.X``) minus the highest mean z
                  among the other 13 types;
    eligible    = detection >= 0.25;
    keep        = top 3 eligible by specificity (ties: detection);
    fallback    = no eligible marker -> the single marker with the highest
                  specificity, flagged.
- DCT: the Fig 3B replacement panel (plan step 5; fixed groups) and, for the
  supplement, the V1 workbook markers with the placement rule above.
- Dot encoding (as SISKA Fig 1c): area = fraction of cells with intensity > 0
  (``layers['intensity']``); colour = group mean of ``.X`` (per-protein
  z-scored ln(1 + intensity)), rescaled 0-1 per protein across the rows shown
  in that plot (min-max, as ``standard_scale='var'``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc

TASK = "dotplot"
SEED = cc.RANDOM_SEED

MAIN_MAX_PER_TYPE = 3
MAIN_MIN_DETECTION = 0.25
PLACEMENT_RULE = "highest group mean z among the row types listing the marker in the workbook"

KIDNEY_ROW_ORDER = ("DVR", "JG", "PCT", "PST", "DTL", "ATL", "TAL", "MD", "DCT1", "DCT2", "CNT", "PC",
                    "IC-A", "IC-B")
DCT_ROW_ORDER = ("DCT1", "DCT2", "ProLIF")
UNKNOWN = "Unknown"

LABEL_COLUMNS = {
    "kidney": ("kidney_reannot_label", "kidney_pre_label"),
    "dct": ("v1_cell_label", "v1_cluster_label"),
}
PRIMARY = {"kidney": "kidney_reannot_label", "dct": "v1_cell_label"}
LABEL_DISPLAY = {
    "kidney_reannot_label": "re-annotated labels (Fig 2B)",
    "kidney_pre_label": "pre-annotation labels (Fig 2A)",
    "v1_cell_label": "V1 per-cell labels",
    "v1_cluster_label": "V1 cluster labels",
}
LABEL_SHORT = {"kidney_reannot_label": "reannot", "kidney_pre_label": "pre",
               "v1_cell_label": "v1_cell_label", "v1_cluster_label": "v1_cluster_label"}

# Plan step 5: Fig 3B replacement panel (group label, markers).
DCT_PANEL = (
    ("Core", ("Slc12a3",)),
    ("DCT1", ("Pvalb", "Egf")),
    ("DCT2", ("Calb1", "Slc8a1", "S100g", "Klk1")),
    ("TAL", ("Umod", "Slc12a1")),
    ("Cycling", ("Mki67", "Top2a")),
)

# the project notes counts (checked before any analysis).
EXPECTED_COUNTS = {
    "kidney_reannot_label": {"PCT": 219, "PC": 187, "PST": 72, "DTL": 56, "CNT": 53, "TAL": 45, "DCT2": 27,
                             "DCT1": 18, "DVR": 16, "ATL": 7, "JG": 6, "IC-B": 3, "IC-A": 2, "MD": 1},
    "kidney_pre_label": {"PCT": 168, "TAL": 154, "PST": 132, "DVR": 88, "CNT": 73, "DCT2": 24, "JG": 22,
                         "MD": 19, "IC-B": 8, "Unknown": 24},
    "v1_cell_label": {"DCT1": 262, "DCT2": 235, "ProLIF": 253},
    "v1_cluster_label": {"DCT1": 367, "DCT2": 139, "ProLIF": 240, "Unknown": 4},
}


def row_order(label_col: str, present) -> list[str]:
    """Plotted rows for one label column, top to bottom."""
    present = set(map(str, present))
    if label_col.startswith("kidney"):
        rows = [t for t in KIDNEY_ROW_ORDER if t in present]
        extra = sorted(present.difference(KIDNEY_ROW_ORDER).difference({UNKNOWN}))
        if extra:
            raise ValueError(f"{label_col}: labels without a row position: {extra}")
        return rows + ([UNKNOWN] if UNKNOWN in present else [])
    rows = [t for t in DCT_ROW_ORDER if t in present]
    extra = sorted(present.difference(DCT_ROW_ORDER).difference({UNKNOWN}))
    if extra:
        raise ValueError(f"{label_col}: labels without a row position: {extra}")
    return rows  # DCT Unknown (v1_cluster_label, 4 cells) is not plotted


def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run dotplot_marker_tables.py first.")
    return pd.read_csv(path, **kwargs)


# ---------------------------------------------------------------------------
# Marker workbooks
# ---------------------------------------------------------------------------
def read_kidney_workbook(path: Path) -> tuple[list[tuple[str, str, list[str]]], list[dict]]:
    """V3 workbook rows as (short, full name, markers in workbook order) plus within-row duplicate notes."""
    raw = pd.read_excel(path)
    rows, dups = [], []
    for rec in raw.itertuples(index=False):
        full = str(rec[0]).strip()
        short = str(rec[2]).strip() if len(rec) > 2 and pd.notna(rec[2]) else cc.KIDNEY_ABBREVIATIONS[full]
        if cc.KIDNEY_ABBREVIATIONS.get(full, short) != short:
            raise ValueError(f"workbook short name {short!r} differs from scp_common for {full!r}")
        genes = [cc.clean_gene_symbol(g) for g in str(rec[1]).split(",")]
        genes = [g for g in genes if g and g != "Nan"]
        seen: list[str] = []
        for g in genes:
            if g in seen:
                dups.append({"type": short, "marker": g, "note": f"listed more than once in the {short} row"})
            else:
                seen.append(g)
        rows.append((short, full, seen))
    return rows, dups


def read_dct_workbook(path: Path) -> list[tuple[str, str, list[str]]]:
    rows = []
    for name, genes in cc.read_marker_workbook(path).items():
        rows.append((name, name, list(dict.fromkeys(genes))))
    return rows


def listed_markers(workbook: list[tuple[str, str, list[str]]], row_types, var_names) -> dict[str, list[str]]:
    """Unique workbook markers of the row types -> row types listing them (y order); first-seen order."""
    by_type = {short: genes for short, _, genes in workbook}
    listed_for: dict[str, list[str]] = {}
    for t in row_types:
        for g in by_type.get(t, []):
            listed_for.setdefault(g, []).append(t)
    return listed_for


def place_markers(workbook: list[tuple[str, str, list[str]]], row_types, var_names,
                  mean_z: pd.DataFrame) -> pd.DataFrame:
    """Marker order with data-driven placement among the listing types (rule in the module docstring).

    ``mean_z`` is a (row type x marker) table of group mean z on the primary labels, covering every
    present marker. Returns one row per unique workbook marker of the row types, present or not.
    """
    var = set(map(str, var_names))
    by_type = {short: genes for short, _, genes in workbook}
    full_name = {short: full for short, full, _ in workbook}
    listed_for = listed_markers(workbook, row_types, var_names)
    recs = []
    for g, types in listed_for.items():
        present = g in var
        if present:
            listed_means = mean_z.loc[types, g]
            group = str(listed_means.idxmax())
            top_any = str(mean_z[g].idxmax())
            placement = "highest mean z among listing types" if len(types) > 1 else "only listing type"
            means_txt = "; ".join(f"{t} {mean_z.at[t, g]:.3f}" for t in types)
        else:
            group, top_any, placement, means_txt = types[0], "", "absent from matrix: first listing type", ""
        others = [o for o in types if o != group]
        recs.append({"group": group, "group_full_name": full_name.get(group, group),
                     "group_position": list(row_types).index(group) + 1, "marker": g,
                     "workbook_position_in_group": by_type[group].index(g) + 1, "in_protein_matrix": present,
                     "plotted": present, "listed_for": ";".join(types), "also_listed_for": ";".join(others),
                     "shared": bool(others), "placement": placement, "mean_z_listing_types": means_txt,
                     "top_row_all_types": top_any,
                     "assigned_is_top_of_all_rows": (top_any == group) if present else np.nan})
    out = pd.DataFrame(recs)
    out["repeat"] = False
    return renumber(out)


def renumber(order: pd.DataFrame) -> pd.DataFrame:
    out = order.sort_values(["group_position", "workbook_position_in_group"]).reset_index(drop=True)
    pos = out["plotted"].cumsum()
    out["position"] = np.where(out["plotted"], pos, np.nan)
    return out


def fill_empty_types(order: pd.DataFrame, workbook: list[tuple[str, str, list[str]]], row_types,
                     mean_z: pd.DataFrame, frac: pd.DataFrame) -> pd.DataFrame:
    """Give every row type a column: repeat a listed marker under types left without a placed marker."""
    by_type = {short: genes for short, _, genes in workbook}
    full_name = {short: full for short, full, _ in workbook}
    placed = order[order["plotted"]]
    extra = []
    for gpos, t in enumerate(row_types, start=1):
        if (placed["group"] == t).any():
            continue
        cands = [g for g in by_type.get(t, []) if g in mean_z.columns]
        if not cands:
            continue
        best = sorted(cands, key=lambda g: (-mean_z.at[t, g], -frac.at[t, g]))[0]
        src = placed[placed["marker"] == best].iloc[0]
        extra.append({**src.to_dict(), "group": t, "group_full_name": full_name.get(t, t), "group_position": gpos,
                      "workbook_position_in_group": by_type[t].index(best) + 1, "repeat": True,
                      "assigned_is_top_of_all_rows": src["top_row_all_types"] == t,
                      "placement": f"repeat: {t} had no placed marker; also placed under {src['group']}",
                      "also_listed_for": ";".join(o for o in src["listed_for"].split(";") if o != t)})
    if not extra:
        return order
    return renumber(pd.concat([order, pd.DataFrame(extra)], ignore_index=True))


def scaled_01(values: pd.DataFrame, key: str = "mean_z") -> pd.Series:
    """Per (label column, marker) min-max rescaling of a group mean across the plotted rows."""
    grp = values.groupby(["label_column", "marker"])[key]
    lo, hi = grp.transform("min"), grp.transform("max")
    span = (hi - lo).where(hi > lo)
    return ((values[key] - lo) / span).fillna(0.0)


# ---------------------------------------------------------------------------
# Per-group values
# ---------------------------------------------------------------------------
def group_values(adata, labels: pd.Series, rows: list[str], markers: list[str]) -> pd.DataFrame:
    """Long table: per (group, marker) n cells, n / fraction detected (intensity > 0), mean z, mean ln1p."""
    names = pd.Index(adata.var_names.astype(str))
    idx = names.get_indexer(markers)
    if (idx < 0).any():
        raise KeyError(f"markers not in matrix: {[m for m, i in zip(markers, idx) if i < 0]}")
    z = cc.dense(adata.X[:, idx]).astype(np.float64)
    det = cc.dense(adata.layers["intensity"][:, idx]) > 0
    ln1p = cc.dense(adata.layers["ln1p_intensity"][:, idx]).astype(np.float64)
    lab = labels.astype(str).to_numpy()
    out = []
    for gpos, g in enumerate(rows, start=1):
        mask = lab == g
        n = int(mask.sum())
        for j, m in enumerate(markers):
            out.append({"group": g, "group_position": gpos, "n_cells": n, "marker": m,
                        "n_detected": int(det[mask, j].sum()), "frac_detected": float(det[mask, j].mean()),
                        "mean_z": float(z[mask, j].mean()), "mean_ln1p_intensity": float(ln1p[mask, j].mean())})
    return pd.DataFrame(out)


def select_main(values: pd.DataFrame, order_full: pd.DataFrame, rows: list[str]) -> pd.DataFrame:
    """Main-version selection on the primary kidney labels (rule in the module docstring)."""
    plotted = order_full[order_full["plotted"]]
    mean_z = values.pivot(index="group", columns="marker", values="mean_z")
    frac = values.pivot(index="group", columns="marker", values="frac_detected")
    out = []
    for t in rows:
        cands = plotted.loc[plotted["group"] == t, "marker"].tolist()
        recs = []
        for m in cands:
            others = mean_z[m].drop(index=t)
            recs.append({"group": t, "marker": m, "n_cells": int(values.loc[values["group"] == t, "n_cells"].iloc[0]),
                         "frac_detected_own": float(frac.at[t, m]), "mean_z_own": float(mean_z.at[t, m]),
                         "max_mean_z_other": float(others.max()), "max_other_group": str(others.idxmax()),
                         "specificity": float(mean_z.at[t, m] - others.max()),
                         "own_rank_by_mean_z": int((mean_z[m] > mean_z.at[t, m]).sum() + 1),
                         "eligible": bool(frac.at[t, m] >= MAIN_MIN_DETECTION)})
        if not recs:
            continue
        frame = pd.DataFrame(recs).sort_values(["eligible", "specificity", "frac_detected_own"],
                                               ascending=[False, False, False]).reset_index(drop=True)
        frame["rank_in_group"] = np.arange(1, len(frame) + 1)
        n_eligible = int(frame["eligible"].sum())
        if n_eligible:
            frame["selected"] = frame["eligible"] & (frame["rank_in_group"] <= MAIN_MAX_PER_TYPE)
            frame["selection"] = np.where(frame["selected"], "eligible, top 3 by specificity",
                                          np.where(frame["eligible"], "eligible, not in top 3",
                                                   "below detection threshold"))
        else:
            best = frame["specificity"].idxmax()
            frame["selected"] = frame.index == best
            frame["selection"] = np.where(frame["selected"], "fallback: no eligible marker; highest specificity",
                                          "below detection threshold")
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def main_order(order_full: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Main-version order: selected markers, in the full version's order."""
    chosen = set(zip(selection.loc[selection["selected"], "group"], selection.loc[selection["selected"], "marker"]))
    main = order_full[order_full["plotted"]].copy()
    main = main[[(g, m) in chosen for g, m in zip(main["group"], main["marker"])]].copy()
    main = main.sort_values("position")
    main["position"] = np.arange(1, len(main) + 1)
    sel = selection.set_index(["group", "marker"])
    main["selection"] = [sel.at[(g, m), "selection"] for g, m in zip(main["group"], main["marker"])]
    return main
