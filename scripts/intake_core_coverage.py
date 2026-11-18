#!/usr/bin/env python
"""intake_core step 4: marker and MitoCarta coverage per condition (coverage only; no abundance comparison).



Inputs (read-only): data/DCT/DCT_markers_V1.xlsx; results/tables/intake_core/intake_core_protein_mapping.csv
and intake_core_dct_cells.csv; results/tables/qc_mito/qc_mito_mitocarta_mouse_list.csv (qc_mito parse of
data/reference/Mouse.MitoCarta3.0.xls; no download).
Outputs (results/tables/intake_core/):
  intake_core_marker_coverage.csv          one row per list x group x marker; per condition: in export,
                                          retained, cells detected and fraction of retained cells; in joint
  intake_core_marker_coverage_summary.csv  per list x group: markers in export / retained / in joint
  intake_core_mitocarta_coverage.csv       MitoCarta3.0 protein groups per condition and protein set
  intake_core_mitocarta_protein_flags.csv  MitoCarta flags of every protein group of both exports

Marker rule: a marker is present when it is the cleaned first PG.Genes symbol of
a group (the name later scoring uses). Groups where it is only a secondary
symbol are listed separately. When several groups share the first symbol, the
one detected in most retained cells is reported and the count of groups is
given. Cell fractions use the retained cells of each condition.

Command (project root):  python scripts/intake_core_coverage.py
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
import intake_core_lib as lib

MARKER_WORKBOOK = cc.DCT_DIR / "DCT_markers_V1.xlsx"


def marker_lists(run: cc.Run) -> list[tuple[str, str, list[str]]]:
    workbook = cc.read_marker_workbook(run.use_input(MARKER_WORKBOOK, "DCT marker workbook (Supp Table 1 DCT tab)"))
    lists = [("DCT_markers_V1", cell_type, genes) for cell_type, genes in workbook.items()]
    lists += [("contamination", group, genes) for group, genes in lib.CONTAMINATION_PANEL.items()]
    lists += [("zeroK_candidate_panel", group, genes) for group, genes in lib.ZEROK_PANEL.items()]
    lists += [("cell_cycle", group, genes) for group, genes in lib.CELL_CYCLE_PANELS.items()]
    return lists


def coverage(lists, mapping: pd.DataFrame, n_cells: dict[str, int]) -> pd.DataFrame:
    symbols = mapping["pg_genes"].fillna("").map(lib.split_symbols)
    rows = []
    for list_name, group, genes in lists:
        for marker in genes:
            names = [marker, *lib.TIROSH_ALIASES.get(marker, [])] if list_name == "cell_cycle" else [marker]
            lower = {n.lower() for n in names}
            row = {"list": list_name, "group": group, "marker": marker,
                   "aliases_checked": ";".join(names[1:])}
            joint_names = []
            for cond in lib.CONDITIONS:
                k = lib.FILE_KEY[cond]
                own = mapping["search"] == cond
                first = mapping.loc[own & mapping["gene_name"].isin(names)]
                secondary = mapping.loc[own & ~mapping["gene_name"].isin(names)
                                        & symbols.map(lambda s: bool(lower.intersection(s)))]
                row[f"{k}_in_export"] = not first.empty
                row[f"{k}_n_groups"] = len(first)
                if first.empty:
                    row.update({f"{k}_protein_group": "", f"{k}_var_name": "", f"{k}_matched_symbol": "",
                                f"{k}_retained": False, f"{k}_n_cells": 0, f"{k}_fraction_cells": 0.0})
                else:
                    best = first.sort_values(f"n_cells_{k}", ascending=False).iloc[0]
                    n = int(best[f"n_cells_{k}"])
                    row.update({f"{k}_protein_group": best["protein_group"], f"{k}_var_name": best["var_name"],
                                f"{k}_matched_symbol": best["gene_name"],
                                f"{k}_retained": bool(best[f"retained_{k}"]), f"{k}_n_cells": n,
                                f"{k}_fraction_cells": n / n_cells[cond]})
                    joint_names.extend(first.loc[first["in_joint"], "joint_var_name"].tolist())
                row[f"{k}_secondary_symbol_only"] = first.empty and not secondary.empty
                row[f"{k}_secondary_groups"] = " | ".join(secondary["protein_group"])
            row["in_joint"] = bool(joint_names)
            row["joint_var_name"] = ";".join(dict.fromkeys(joint_names))
            ctrl_status = mapping.loc[(mapping["search"] == "Ctrl") & mapping["gene_name"].isin(names), "match_status"]
            zerok_status = mapping.loc[(mapping["search"] == "ZeroK") & mapping["gene_name"].isin(names), "match_status"]
            row["match_status_ctrl"] = ";".join(sorted(set(ctrl_status)))
            row["match_status_zerok"] = ";".join(sorted(set(zerok_status)))
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_summary(cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (list_name, group), sub in cov.groupby(["list", "group"], sort=False):
        row = {"list": list_name, "group": group, "n_markers": len(sub)}
        for k in ("ctrl", "zerok"):
            row[f"{k}_in_export"] = int(sub[f"{k}_in_export"].sum())
            row[f"{k}_retained"] = int(sub[f"{k}_retained"].sum())
            row[f"{k}_secondary_symbol_only"] = int(sub[f"{k}_secondary_symbol_only"].sum())
            row[f"{k}_absent"] = ";".join(sub.loc[~sub[f"{k}_in_export"], "marker"])
        row["in_joint"] = int(sub["in_joint"].sum())
        row["not_in_joint"] = ";".join(sub.loc[~sub["in_joint"], "marker"])
        rows.append(row)
    return pd.DataFrame(rows)


def mitocarta_tables(run: cc.Run, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mitocarta = lib.read_mitocarta(run)
    xls = cc.PROJECT_ROOT / "data" / "reference" / "Mouse.MitoCarta3.0.xls"
    if xls.exists():
        run.use_input(xls, "MitoCarta3.0 mouse inventory (qc_mito download; parsed list used)")
    n_list = int((mitocarta["mitocarta_list"] == "MitoCarta3.0").sum()) if "mitocarta_list" in mitocarta else len(mitocarta)
    flags = []
    for cond in lib.CONDITIONS:
        k = lib.FILE_KEY[cond]
        own = mapping.loc[mapping["search"] == cond].reset_index(drop=True)
        meta = pd.DataFrame({"pg_genes": own["pg_genes"].fillna("").to_numpy(),
                             "protein_group": own["protein_group"].to_numpy(),
                             "first_symbol": own["gene_name"].to_numpy(),
                             "protein_description": ""}, index=pd.Index(own["var_name"].to_numpy(), name="protein"))
        ann = lib.annotate_proteins(meta, mitocarta).reset_index()
        ann.insert(0, "search", cond)
        ann["retained"] = own[f"retained_{k}"].astype(bool).to_numpy()
        ann["in_joint"] = own["in_joint"].to_numpy()
        ann["match_status"] = own["match_status"].to_numpy()
        flags.append(ann)
    flags = pd.concat(flags, ignore_index=True)
    rows = []
    for cond in lib.CONDITIONS:
        f = flags.loc[flags["search"] == cond]
        for protein_set, mask in (("all groups in export", np.ones(len(f), bool)),
                                  ("retained groups", f["retained"].to_numpy()),
                                  ("matched (joint) groups", f["in_joint"].to_numpy())):
            s = f.loc[mask]
            genes_hit = {g for hits in s.loc[s["mitocarta"], "mitocarta_hits"] for g in str(hits).split(";") if g}
            rows.append({"condition": cond, "protein_set": protein_set, "n_groups": len(s),
                         "n_mitocarta": int(s["mitocarta"].sum()),
                         "fraction_mitocarta": float(s["mitocarta"].mean()) if len(s) else np.nan,
                         "n_mitocarta_symbol_only": int(s["mitocarta_symbol_only"].sum()),
                         "n_oxphos": int(s["oxphos"].sum()), "n_oxphos_subunit": int(s["oxphos_subunit"].sum()),
                         "n_mt_encoded": int(s["mt_encoded"].sum()),
                         "mt_encoded_groups": ";".join(s.loc[s["mt_encoded"], "protein"]),
                         "n_distinct_mitocarta_genes": len(genes_hit),
                         "fraction_of_mitocarta3_list": len(genes_hit) / n_list,
                         "n_groups_matched_to_gt1_mitocarta_gene": int((s["mitocarta_hits"].str.count(";") > 0).sum())})
    return pd.DataFrame(rows), flags


def main() -> None:
    parser = argparse.ArgumentParser(description="intake_core: marker and MitoCarta coverage.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "coverage", args)

    mapping = lib.read_t17_table(run, "protein_mapping", keep_default_na=False, na_values={"n_cells_ctrl": [""],
                                                                                          "n_cells_zerok": [""]})
    for col in ("retained_ctrl", "retained_zerok"):
        mapping[col] = mapping[col].astype(str).str.lower().eq("true")
    for col in ("in_joint",):
        mapping[col] = mapping[col].astype(str).str.lower().eq("true")
    cells = lib.read_t17_table(run, "dct_cells")
    retained = cells["retained"].astype(str).str.lower().eq("true")
    n_cells = {cond: int((retained & (cells["condition"] == cond)).sum()) for cond in lib.CONDITIONS}

    cov = coverage(marker_lists(run), mapping, n_cells)
    run.table(cov, "marker_coverage")
    summary = coverage_summary(cov)
    run.table(summary, "marker_coverage_summary")
    mito, flags = mitocarta_tables(run, mapping)
    run.table(mito, "mitocarta_coverage")
    run.table(flags, "mitocarta_protein_flags")

    for row in summary.itertuples(index=False):
        run.note(f"{row.list} / {row.group}: Ctrl {row.ctrl_in_export}/{row.n_markers}, ZeroK "
                 f"{row.zerok_in_export}/{row.n_markers} in export; {row.in_joint}/{row.n_markers} in joint.")
    run.finish(parameters={"marker_rule": "cleaned first PG.Genes symbol (Tirosh aliases checked for cell-cycle "
                                          "lists); secondary-symbol hits listed separately",
                           "retained_cells": n_cells})


if __name__ == "__main__":
    main()
