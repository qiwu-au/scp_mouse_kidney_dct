#!/usr/bin/env python
"""dotplot step 1: marker order, coverage and per-group dot values for the dotplots.

Rules (placement, main selection,
encoding): ``dotplot_lib.py``.

Inputs (read-only):
  results/h5ad/kidney_object_KIDNEY_submitted.h5ad, results/h5ad/kidney_object_DCT_V1_submitted.h5ad
  results/tables/kidney_object/kidney_object_{kidney,dct}_canonical_labels.csv
  data/KIDNEY/KIDNEY_markers_V3.xlsx, data/DCT/DCT_markers_V1.xlsx
Outputs (results/tables/dotplot/):
  dotplot_label_order.csv                 rows (y-axis) per label column, n cells, colour, plotted
  dotplot_marker_order_kidney.csv         full and main versions (placement, positions); reused by kidney_rna
  dotplot_marker_order_dct.csv            panel (Fig 3B replacement) and workbook (V1) versions; reused by atlas_dct
  dotplot_kidney_marker_coverage.csv      every V3 workbook entry: in matrix, assigned group, plotted
  dotplot_dct_marker_coverage.csv         every panel / V1 workbook entry: in matrix, plotted
  dotplot_kidney_shared_markers.csv       markers listed for >= 2 row types: assigned group, mean z per type
  dotplot_kidney_main_selection.csv       main-version rule metrics per candidate marker
  dotplot_kidney_dotplot_values.csv       per (label column, group, marker): n, fraction detected, mean z,
                                             mean z rescaled 0-1 (the plotted colour)
  dotplot_dct_dotplot_values.csv          same for DCT (panel and workbook markers)
  dotplot_tables_checks.csv               sanity checks
  dotplot_step_marker_tables_manifest.json

Command (project root):
  python scripts/dotplot_marker_tables.py
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
import dotplot_lib as t6

TASK = t6.TASK


def label_colors(adata, col: str) -> dict[str, str]:
    cats = list(adata.obs[col].astype("category").cat.categories.astype(str))
    colors = list(adata.uns.get(f"{col}_colors", []))
    return dict(zip(cats, colors)) if len(colors) == len(cats) else {}


def check(rows: list[dict], name: str, ok: bool, observed, expected="", kind: str = "check") -> None:
    rows.append({"check": name, "observed": observed, "expected": expected,
                 "pass": "info" if kind == "info" else bool(ok)})
    if kind != "info" and not ok:
        print(f"CHECK FAILED: {name}: observed {observed}, expected {expected}")


def values_for(adata, labels, rows_by_col, columns, markers) -> pd.DataFrame:
    out = []
    for col in columns:
        v = t6.group_values(adata, labels[col], rows_by_col[col], markers)
        v.insert(0, "label_column", col)
        out.append(v)
    return pd.concat(out, ignore_index=True)


def placement_checks(checks, name, order, mean_z, row_types, ref: pd.DataFrame | None = None) -> None:
    """``ref``: the full-version order the repeats refer to (default: ``order`` itself)."""
    plotted = order[order["plotted"]]
    placed = plotted[~plotted["repeat"]]
    rep = plotted[plotted["repeat"]]
    ref_placed = placed if ref is None else ref[ref["plotted"] & ~ref["repeat"]]
    listed_ok = all(g in lst.split(";") for g, lst in zip(plotted["group"], plotted["listed_for"]))
    check(checks, f"{name}: every plotted column sits under a type that lists its marker", listed_ok, listed_ok, True)
    top_ok = all(mean_z.at[g, m] >= mean_z.loc[lst.split(";"), m].max() - 1e-12
                 for g, m, lst in zip(placed["group"], placed["marker"], placed["listed_for"]))
    check(checks, f"{name}: placed (non-repeat) marker sits under the listing type with the highest mean z",
          top_ok, top_ok, True)
    markers = placed["marker"].tolist()
    check(checks, f"{name}: each quantified marker placed once (repeats aside)", len(markers) == len(set(markers)),
          len(markers), len(set(markers)))
    pairs = list(zip(plotted["group"], plotted["marker"]))
    check(checks, f"{name}: each (group, marker) column once", len(pairs) == len(set(pairs)), len(pairs),
          len(set(pairs)))
    rep_ok = all(not ref_placed["group"].eq(g).any() and ref_placed["marker"].eq(m).any()
                 and ref_placed.loc[ref_placed["marker"].eq(m), "group"].iloc[0] != g
                 for g, m in zip(rep["group"], rep["marker"]))
    check(checks, f"{name}: repeats only for types without a placed marker, of markers placed elsewhere", rep_ok,
          "; ".join(f"{m} under {g}" for g, m in zip(rep["group"], rep["marker"])) or "none", "")
    missing = [t for t in row_types if not plotted["group"].eq(t).any()]
    check(checks, f"{name}: every row type has at least one column", not missing,
          ";".join(missing) or "all", "all")
    off = placed[placed["assigned_is_top_of_all_rows"] == False]  # noqa: E712
    check(checks, f"{name}: placed markers whose highest row is not the assigned type", True,
          f"{len(off)}: " + "; ".join(f"{m} ({g}; top {t})" for m, g, t in
                                      zip(off["marker"], off["group"], off["top_row_all_types"])), "", kind="info")


def positions(order: pd.DataFrame) -> dict[str, str]:
    """marker -> plotted column position(s), ';'-joined (a repeated marker has several)."""
    p = order[order["plotted"]]
    return p.groupby("marker")["position"].apply(lambda s: ";".join(str(int(v)) for v in sorted(s))).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="dotplot step 1: marker order, coverage and dot values.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, "marker_tables", args)
    checks: list[dict] = []

    data = {}
    for ds in ("kidney", "dct"):
        run.use_input(cc.CANONICAL[ds]["h5ad"], f"kidney_object canonical {ds} object")
        run.use_input(cc.CANONICAL[ds]["labels"], f"kidney_object canonical {ds} labels")
        adata, labels = cc.load_canonical(ds)
        data[ds] = (adata, labels)
        for col in t6.LABEL_COLUMNS[ds]:
            counts = labels[col].astype(str).value_counts().to_dict()
            check(checks, f"{col} counts = the project notes", counts == t6.EXPECTED_COUNTS[col],
                  ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
                  ", ".join(f"{k} {v}" for k, v in sorted(t6.EXPECTED_COUNTS[col].items())))
        for layer in ("intensity", "ln1p_intensity"):
            check(checks, f"{ds}: layers['{layer}'] present", layer in adata.layers, layer in adata.layers, True)

    # ------------------------------------------------------------------ label order
    order_rows = []
    rows_by_col = {}
    for ds in ("kidney", "dct"):
        adata, labels = data[ds]
        for col in t6.LABEL_COLUMNS[ds]:
            present = labels[col].astype(str)
            rows = t6.row_order(col, present.unique())
            rows_by_col[col] = rows
            colors = label_colors(adata, col)
            counts = present.value_counts()
            for label in [*rows, *sorted(set(present) - set(rows))]:
                plotted = label in rows
                order_rows.append({"dataset": ds, "label_column": col,
                                   "position": rows.index(label) + 1 if plotted else np.nan,
                                   "label": label, "n_cells": int(counts.get(label, 0)),
                                   "colour": colors.get(label, ""), "plotted": plotted,
                                   "note": "" if plotted else "not plotted (plan: DCT1 / DCT2 / ProLIF rows only)"})
            check(checks, f"{col}: every label has a row or is the excluded DCT Unknown",
                  set(present) - set(rows) <= ({t6.UNKNOWN} if ds == "dct" else set()),
                  ";".join(sorted(set(present) - set(rows))) or "none", "none (kidney) / Unknown (DCT)")
    label_order = pd.DataFrame(order_rows)
    label_order["position"] = label_order["position"].astype("Int64")
    run.table(label_order, "label_order")
    check(checks, "kidney_pre_label rows are a subset of the re-annotated rows (plus Unknown)",
          set(rows_by_col["kidney_pre_label"]) - {t6.UNKNOWN} <= set(rows_by_col["kidney_reannot_label"]),
          ";".join(rows_by_col["kidney_pre_label"]), "subset")

    # ------------------------------------------------------------------ kidney markers
    kad, klab = data["kidney"]
    names = pd.Index(kad.var_names.astype(str))
    kpath = run.use_input(cc.INPUTS["kidney"]["markers"], "kidney V3 marker workbook (Supp Table 1)")
    kwb, kdups = t6.read_kidney_workbook(kpath)
    krows = rows_by_col["kidney_reannot_label"]
    klisted = t6.listed_markers(kwb, krows, names)
    kpresent = [g for g in klisted if g in names]
    kvals = values_for(kad, klab, rows_by_col, t6.LABEL_COLUMNS["kidney"], kpresent)
    kprim = kvals[kvals["label_column"] == t6.PRIMARY["kidney"]]
    kmz = kprim.pivot(index="group", columns="marker", values="mean_z")
    kfr = kprim.pivot(index="group", columns="marker", values="frac_detected")
    order_full = t6.fill_empty_types(t6.place_markers(kwb, krows, names, kmz), kwb, krows, kmz, kfr)
    placement_checks(checks, "kidney full version", order_full, kmz, krows)
    plotted_full = list(dict.fromkeys(order_full.loc[order_full["plotted"], "marker"]))
    check(checks, "kidney full version: every plotted marker is a var name (exact match)",
          all(m in names for m in plotted_full), sum(m in names for m in plotted_full), len(plotted_full))

    reannot_vals = kvals[kvals["label_column"] == t6.PRIMARY["kidney"]]
    selection = t6.select_main(reannot_vals, order_full, krows)
    rep_pairs = set(zip(order_full.loc[order_full["repeat"], "group"], order_full.loc[order_full["repeat"], "marker"]))
    selection["repeat"] = [(g, m) in rep_pairs for g, m in zip(selection["group"], selection["marker"])]
    order_main = t6.main_order(order_full, selection)
    placement_checks(checks, "kidney main version", order_main, kmz, krows, ref=order_full)

    of = order_full.copy()
    of.insert(0, "version", "full")
    of["selection"] = ""
    om = order_main.copy()
    om.insert(0, "version", "main")
    marker_order_k = pd.concat([of, om], ignore_index=True)
    marker_order_k["position"] = marker_order_k["position"].astype("Int64")
    marker_order_k["placement_labels"] = "kidney_reannot_label (same columns for kidney_pre_label)"
    run.table(marker_order_k, "marker_order_kidney")

    pos_full, pos_main = positions(order_full), positions(order_main)
    placed_k = order_full[~order_full["repeat"]]
    first_pos = {m: int(p.split(";")[0]) for m, p in pos_full.items()}
    kvals["marker_group"] = kvals["marker"].map(dict(zip(placed_k["marker"], placed_k["group"])))
    kvals["x_positions_full"] = kvals["marker"].map(pos_full)
    kvals["x_positions_main"] = kvals["marker"].map(pos_main)
    kvals["mean_z_scaled_0_1"] = t6.scaled_01(kvals)
    kvals = (kvals.assign(_p=kvals["marker"].map(first_pos))
             .sort_values(["label_column", "_p", "group_position"]).drop(columns="_p").reset_index(drop=True))
    run.table(kvals, "kidney_dotplot_values")
    run.table(selection, "kidney_main_selection")

    # coverage: every workbook entry, including types without cells
    cov = []
    assigned = dict(zip(placed_k["marker"], placed_k["group"]))
    full_pairs = set(zip(order_full.loc[order_full["plotted"], "group"], order_full.loc[order_full["plotted"], "marker"]))
    main_pairs = set(zip(order_main["group"], order_main["marker"]))
    for short, full, genes in kwb:
        for wpos, g in enumerate(genes, start=1):
            in_matrix = g in names
            in_rows = short in krows
            group = assigned.get(g, "")
            if not in_rows:
                note = "type has no cells under either kidney label column; not plotted"
            elif not in_matrix:
                note = "absent from the 4,054-protein matrix"
            elif (short, g) in rep_pairs:
                note = f"repeated under this type (no placed marker); placed under {group}"
            elif group != short:
                note = f"plotted under {group} (highest mean z among the types listing it)"
            else:
                note = "plotted under this type"
            cov.append({"type": short, "type_full_name": full, "type_has_cells": in_rows, "marker": g,
                        "workbook_position": wpos, "in_protein_matrix": in_matrix, "assigned_group": group,
                        "plotted_full_under_this_type": (short, g) in full_pairs,
                        "plotted_main_under_this_type": (short, g) in main_pairs,
                        "repeat_under_this_type": (short, g) in rep_pairs, "note": note})
    for d in kdups:
        cov.append({"type": d["type"], "type_full_name": "", "type_has_cells": d["type"] in krows,
                    "marker": d["marker"], "workbook_position": np.nan, "in_protein_matrix": d["marker"] in names,
                    "assigned_group": assigned.get(d["marker"], ""), "plotted_full_under_this_type": False,
                    "plotted_main_under_this_type": False, "repeat_under_this_type": False,
                    "note": f"duplicate workbook entry ({d['note']}); counted once"})
    kcov = pd.DataFrame(cov)
    run.table(kcov, "kidney_marker_coverage")
    shared = placed_k[placed_k["shared"]][["marker", "group", "listed_for", "also_listed_for",
                                           "mean_z_listing_types", "top_row_all_types", "in_protein_matrix",
                                           "plotted"]].rename(columns={"group": "assigned_group"})
    reps = order_full[order_full["repeat"]].groupby("marker")["group"].apply(";".join)
    shared["repeated_under"] = shared["marker"].map(reps).fillna("")
    shared["positions_full"] = shared["marker"].map(pos_full)
    shared["positions_main"] = shared["marker"].map(pos_main)
    run.table(shared, "kidney_shared_markers")
    for r in order_full[order_full["repeat"]].itertuples():
        det = kfr.at[r.group, r.marker]
        check(checks, f"kidney: {r.group} column is a repeat of {r.marker}", True,
              f"placed under {placed_k.set_index('marker').at[r.marker, 'group']}; detected in "
              f"{det:.0%} of {r.group} cells (re-annotated); mean z {kmz.at[r.group, r.marker]:.3f}", "", kind="info")

    # ------------------------------------------------------------------ DCT markers
    dad, dlab = data["dct"]
    dnames = pd.Index(dad.var_names.astype(str))
    dpath = run.use_input(cc.INPUTS["dct"]["markers"], "DCT V1 marker workbook (Supp Table 1)")
    dwb = t6.read_dct_workbook(dpath)
    drows = list(t6.DCT_ROW_ORDER)
    dlisted = t6.listed_markers(dwb, drows, dnames)
    panel_markers = [m for _, ms in t6.DCT_PANEL for m in ms]
    dpresent = list(dict.fromkeys([m for m in panel_markers if m in dnames] + [g for g in dlisted if g in dnames]))
    dvals = values_for(dad, dlab, rows_by_col, t6.LABEL_COLUMNS["dct"], dpresent)
    dmz = dvals[dvals["label_column"] == t6.PRIMARY["dct"]].pivot(index="group", columns="marker", values="mean_z")

    panel_rows, pos = [], 0
    for gpos, (grp, markers) in enumerate(t6.DCT_PANEL, start=1):
        for wpos, m in enumerate(markers, start=1):
            present = m in dnames
            pos += present
            panel_rows.append({"version": "panel", "group": grp, "group_full_name": grp, "group_position": gpos,
                               "marker": m, "workbook_position_in_group": wpos, "in_protein_matrix": present,
                               "plotted": present, "position": pos if present else np.nan,
                               "listed_for": ";".join(dlisted.get(m, [])), "also_listed_for": "",
                               "shared": False, "repeat": False, "placement": "fixed panel group (plan step 5)",
                               "top_row_all_types": str(dmz[m].idxmax()) if present else ""})
    panel = pd.DataFrame(panel_rows)
    dfr = dvals[dvals["label_column"] == t6.PRIMARY["dct"]].pivot(index="group", columns="marker",
                                                                  values="frac_detected")
    workbook = t6.fill_empty_types(t6.place_markers(dwb, drows, dnames, dmz), dwb, drows, dmz, dfr)
    placement_checks(checks, "DCT workbook version", workbook, dmz, drows)
    workbook.insert(0, "version", "workbook")
    marker_order_d = pd.concat([panel, workbook], ignore_index=True)
    marker_order_d["position"] = marker_order_d["position"].astype("Int64")
    marker_order_d["listed_for_note"] = np.where(
        marker_order_d["version"] == "panel", "listed_for = types listing the marker in DCT_markers_V1.xlsx",
        "listed_for = row types listing the marker; placement on v1_cell_label")
    run.table(marker_order_d, "marker_order_dct")
    check(checks, "DCT panel: all 11 markers in the matrix", bool(panel["in_protein_matrix"].all()),
          int(panel["in_protein_matrix"].sum()), len(panel))

    for version in ("panel", "workbook"):
        sub = marker_order_d[(marker_order_d["version"] == version) & marker_order_d["plotted"]]
        dvals[f"x_position_{version}"] = dvals["marker"].map(dict(zip(sub["marker"], sub["position"]))).astype("Int64")
        dvals[f"marker_group_{version}"] = dvals["marker"].map(dict(zip(sub["marker"], sub["group"])))
    dvals["mean_z_scaled_0_1"] = t6.scaled_01(dvals)
    run.table(dvals, "dct_dotplot_values")

    dcov = [{"version": r["version"], "group": r["group"], "marker": r["marker"],
             "in_protein_matrix": r["in_protein_matrix"], "plotted": r["plotted"],
             "note": "plotted" if r["plotted"] else "absent from the 2,561-protein matrix"}
            for _, r in marker_order_d.iterrows()]
    run.table(pd.DataFrame(dcov), "dct_marker_coverage")

    # ------------------------------------------------------------------ value checks
    for name, vals in (("kidney", kvals), ("dct", dvals)):
        ok = vals["frac_detected"].between(0, 1).all() and np.allclose(
            vals["n_detected"] / vals["n_cells"].where(vals["n_cells"] > 0), vals["frac_detected"])
        check(checks, f"{name}: fractions in [0, 1] and = n_detected / n_cells", ok, ok, True)
        s = vals.groupby(["label_column", "marker"])["mean_z_scaled_0_1"].agg(["min", "max"])
        ok = bool(((s["min"] == 0) & ((s["max"] == 1) | (s["max"] == 0))).all())
        check(checks, f"{name}: scaled colour spans 0-1 per (label column, marker)", ok, ok, True)
        for col in t6.LABEL_COLUMNS[name]:
            sub = vals[vals["label_column"] == col]
            n_rows = sub.groupby("group")["n_cells"].first().sum()
            check(checks, f"{col}: plotted cells", True, int(n_rows), "", kind="info")
            check(checks, f"{col}: group mean z range over plotted dots", True,
                  f"{sub['mean_z'].min():.3f} to {sub['mean_z'].max():.3f}", "", kind="info")
    run.table(pd.DataFrame(checks), "tables_checks")

    # ------------------------------------------------------------------ printout
    for name, o in (("full", order_full), ("main", order_main)):
        print(f"\n=== kidney {name} x order (dagger = repeat) ===")
        for t in krows:
            ms = o[(o["group"] == t) & o["plotted"]]
            print(f"  {t}: " + ", ".join(m + ("†" if r else "") for m, r in zip(ms["marker"], ms["repeat"])))
    print("\n=== DCT workbook order ===")
    for t in drows:
        print(f"  {t}: " + ", ".join(workbook.loc[(workbook["group"] == t) & workbook["plotted"], "marker"]))
    n_fail = sum(1 for c in checks if c["pass"] is False)
    run.note(f"{len(checks)} checks, {n_fail} failed.")
    run.finish({"main_max_per_type": t6.MAIN_MAX_PER_TYPE, "main_min_detection": t6.MAIN_MIN_DETECTION,
                "placement_rule": t6.PLACEMENT_RULE, "placement_labels": t6.PRIMARY,
                "kidney_row_order": list(t6.KIDNEY_ROW_ORDER), "dct_row_order": list(t6.DCT_ROW_ORDER),
                "dct_panel": {g: list(m) for g, m in t6.DCT_PANEL}},
               status="completed" if n_fail == 0 else "completed with failed checks")


if __name__ == "__main__":
    main()
