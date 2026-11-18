#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S-08-d (RNA-protein integration and concordance), one workbook.

The authors asked on 12 Nov 2026 to assemble Supplementary
Tables S-08-d and S-09-c ("Yes, assemble them"). The response to the reviewers, section 5, fixes the content of S-08-d: one workbook
with eight sheets, built from 27 stored tables (kidney_rna atlas, rna_protein_integration integration, and the two value tables of the the response to the reviewers
figures). Kidney data only, revised annotation only (kidney_t15_label; one-annotation rule, authors 30 Oct 2026).
No new statistics: every value is copied from the stored tables. The layout follows Supplementary Table S-07-d
(scripts/report_pt_tal_tables.py), whose workbook helpers are imported unchanged.

What the assembly changes (all recorded per table in the sources table):
- rows: the rows of the submitted annotations are dropped (label_column or reference = kidney_pre_label; the
  8-cell-type row of the chance table, which belongs to kidney_pre_label);
- columns: the column that only names the annotation is dropped after a check that it is constant; the working
  notes of the matching table are dropped (they name tasks, the submitted annotation and decisions); columns that
  are empty after the row filter are dropped; two columns that repeat another column are dropped after a check;
- names: display names of the cell types (DCT1 -> DCT, PC -> CNT/PC; Unknown stays) in the columns that hold an
  SCP-MS cell type; the marker-list categories (listed_for, type) keep the names of Supplementary Table 1;
  internal names are replaced in column names and cell values (t07_, compartment, feature, scib, mka.h5ad); the
  three p_rho columns of the chance table are named fraction_rho (they hold the fraction of the orderings);
- two long-format tables are split by block, each with the columns that apply to it: the values of the three
  supplementary figures (by figure) and the summary over the atlas draws (by group of metrics);
- sheet 2 is one table: rna_protein_integration_features.csv holds every column of kidney_rna_shared_features.csv (checked equal).

This module also holds the general assembly code (Spec, Rules, Layout, build) that
scripts/report_negative_markers_tables.py imports for Supplementary Table S-09-c. Added on 12 Nov 2026 (authors:
"The readers will not know the revision process, so we only show the final products"): a check that no cell and
no sheet name of the saved workbook contains "submitted", "submission" or "re-annotation".

Workbook: results/tables/report/report_08_S08d_tables.xlsx (sheet Contents, then sheets 1-8; Arial 10;
values only, no formulas). The shared output guard knows no .xlsx extension, so the workbook path is checked with
cc._check_writable and added to the run outputs by hand.

Inputs (read-only): the 27 tables named in SPECS.
Outputs: the workbook; results/tables/report/report_08_s08d_table_{sources,checks}.csv;
         results/tables/report/report_step_08_s08d_table_manifest.json

Command (project root):  python scripts/report_rna_protein_tables.py --overwrite
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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment

import scp_common as cc
from report_pt_tal_tables import (F, FB, FN, FT, HEAD_BORDER, HEAD_FILL, REL_TOL, Book, _pick, check, native,
                                      rel_diff)

TASK = "report"


# ---------------------------------------------------------------------------
# General assembly code (also used for Supplementary Table S-09-c)
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    key: str                                   # table id in the workbook, e.g. "4a" (sheet 4, first table)
    src: tuple[str, ...]                       # (task folder, table name without <stage>_[, sub-folder])
    title: str
    notes: tuple[str, ...]
    rows: tuple[tuple[str, str, tuple], ...] = ()          # (column, "in" | "not in", values)
    constant: tuple[str, ...] = ()             # dropped after a check that they hold one value
    determined: dict[str, tuple[str, ...]] = field(default_factory=dict)   # dropped: given by these columns
    drop: dict[str, str] = field(default_factory=dict)     # column -> reason
    drop_empty: bool = False                   # drop the columns that are empty after the row filter
    display: tuple[str, ...] = ()              # columns that hold a label and take its display name
    values: dict[str, dict[str, str]] = field(default_factory=dict)        # column -> {old value: new value}
    same_as: dict[str, str] = field(default_factory=dict)  # dropped after a check: column -> column it repeats
    rename: dict[str, str] = field(default_factory=dict)
    also: tuple[tuple[str, ...], ...] = ()     # further source tables whose columns are all in this table

    @property
    def sheet(self) -> int:
        return int(re.match(r"\d+", self.key).group())


@dataclass
class Rules:
    label_display: dict[str, str]              # display names for the Spec.display columns
    label_allowed: frozenset[str]              # every value of a display column must be one of these
    column_values: dict[str, dict[str, str]]   # source column -> {old: new}, wherever the column occurs
    strict: frozenset[str]                     # columns whose values must all be in their mapping
    rename_exact: dict[str, str]
    rename_re: tuple[tuple[re.Pattern, str], ...]
    forbidden: re.Pattern                      # internal names that must not reach any cell
    forbidden_header: re.Pattern               # internal names that must not reach a column name
    other_rows: Callable[[pd.DataFrame], pd.Series]        # rows of a source that belong to another annotation


@dataclass
class Layout:
    number: str                                # "S-08-d"
    step: str                                  # run step, e.g. "08_s08d_table"
    workbook: str
    title: str
    intro: tuple[str, ...]
    sheets: dict[int, tuple[str, str]]
    terms: list[tuple[str, str]]
    sheet_lines: dict[int, tuple[str, ...]] = field(default_factory=dict)
    description: str = ""


def source_path(src: tuple[str, ...]) -> Path:
    return cc.out_path(src[0], "tables", src[1], ".csv", src[2] if len(src) > 2 else "")


def new_name(name: str, spec: Spec, rules: Rules) -> str:
    if name in spec.rename:
        return spec.rename[name]
    if name in rules.rename_exact:
        return rules.rename_exact[name]
    for pattern, new in rules.rename_re:
        name = pattern.sub(new, name)
    return name


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def assemble_table(spec: Spec, src: pd.DataFrame, rules: Rules) -> tuple[pd.DataFrame, dict, dict]:
    """The rows and columns of one table with reader-facing names; the record says what was done.

    Returns the table, its record, and what a comparison with the source needs (the filtered source rows and the
    map from the new column names to the source columns).
    """
    df, problems, filters, dropped, edits = src.copy(), [], [], [], []
    for col, op, kept in spec.rows:
        if col not in df.columns:
            problems.append(f"filter column {col} not in the source")
            continue
        mask = df[col].isin(kept)
        df = df[mask if op == "in" else ~mask]
        filters.append(f"{col} {op} {' / '.join(str(v) for v in kept)}")
    df = df.copy()
    filtered = df.copy()
    if df.empty:
        problems.append("no rows left")
    for col in spec.constant:
        if col not in df.columns or df[col].nunique(dropna=False) != 1:
            problems.append(f"{col} is missing or not constant after the row filter")
            continue
        dropped.append(f"{col} (one value: {df[col].iloc[0]})")
        df = df.drop(columns=col)
    for col, by in spec.determined.items():
        if col not in df.columns or any(b not in df.columns for b in by) or \
                int(df.groupby(list(by), dropna=False)[col].nunique(dropna=False).max()) != 1:
            problems.append(f"{col} is missing or not given by {' + '.join(by)}")
            continue
        dropped.append(f"{col} (given by {' + '.join(by)})")
        df = df.drop(columns=col)
    if spec.drop_empty:
        empty = [c for c in df.columns if df[c].isna().all()]
        if empty:
            dropped.append("empty after the row filter: " + ", ".join(empty))
            df = df.drop(columns=empty)
    for col, reason in spec.drop.items():
        if col not in df.columns:
            problems.append(f"column to drop not in the source: {col}")
            continue
        dropped.append(f"{col} ({reason})")
        df = df.drop(columns=col)
    for col in spec.display:
        if col not in df.columns:
            problems.append(f"label column not in the table: {col}")
            continue
        new = df[col].map(lambda v: rules.label_display.get(v, v))
        unknown = sorted(set(new.dropna().astype(str)) - rules.label_allowed)
        if unknown:
            problems.append(f"{col}: values that are no label of the annotation: {unknown}")
        used = {k: v for k, v in rules.label_display.items() if k in set(df[col].dropna())}
        if used:
            edits.append(f"{col}: " + "; ".join(f"{k} -> {v}" for k, v in used.items()))
        df[col] = new
    maps = {c: m for c, m in rules.column_values.items() if c in df.columns}
    maps.update(spec.values)
    for col, mapping in maps.items():
        if col not in df.columns:
            problems.append(f"column with value names not in the table: {col}")
            continue
        present = set(df[col].dropna())
        if col in rules.strict:
            unknown = sorted(str(v) for v in present - set(mapping))
            if unknown:
                problems.append(f"{col}: values without a reader-facing name: {unknown}")
        used = {k: v for k, v in mapping.items() if k in present and k != v}
        if used:
            edits.append(f"{col}: " + "; ".join(f"{k} -> {v}" for k, v in used.items()))
        df[col] = df[col].map(lambda v: mapping.get(v, v))
    for col, other in spec.same_as.items():
        if col not in df.columns or other not in df.columns:
            problems.append(f"{col} or {other} is missing")
            continue
        a, b = df[col], df[other]
        have = a.notna()
        if _is_text(a) or _is_text(b):
            same, how = bool((a[have].astype(str) == b[have].astype(str)).all()), "equal"
        else:
            diff = float((a[have].astype(float) - b[have].astype(float)).abs().max()) if have.any() else 0.0
            same, how = diff <= 1e-12, f"max abs difference {diff:.1e}"
        if not same:
            problems.append(f"{col} does not repeat {other}")
            continue
        dropped.append(f"{col} (repeats {other} in its {int(have.sum())} filled cells; {how})")
        df = df.drop(columns=col)
    colmap = {new_name(c, spec, rules): c for c in df.columns}
    renamed = {old: new for new, old in colmap.items() if new != old}
    if len(colmap) != df.shape[1]:
        problems.append("duplicate column names after renaming")
    df = df.rename(columns={old: new for new, old in colmap.items()})
    bad = [c for c in df.columns if rules.forbidden_header.search(str(c))]
    if bad:
        problems.append(f"internal name left in a column name: {bad}")
    for col in df.columns:                      # whole-number columns that pandas read as float (empty cells)
        s = df[col]
        if pd.api.types.is_float_dtype(s) and s.notna().any():
            v = s.dropna()
            if bool((v == v.round()).all()) and (col.startswith("n_") or float(v.abs().max()) >= 2):
                df[col] = s.astype("Int64")
    path = source_path(spec.src)
    record = {"sheet": spec.sheet, "table": spec.key, "title": spec.title, "source": cc.rel(path),
              "role": "values copied from this table", "rows_source": len(src),
              "rows_other_annotation": int(rules.other_rows(src).sum()), "rows_kept": len(df),
              "columns_source": src.shape[1], "columns_kept": df.shape[1],
              "row_filter": "; ".join(filters) or "all rows", "columns_dropped": " | ".join(dropped),
              "columns_renamed": "; ".join(f"{a} -> {b}" for a, b in renamed.items()),
              "values_changed": " | ".join(edits), "problems": " | ".join(problems)}
    return df.reset_index(drop=True), record, {"source": filtered.reset_index(drop=True), "colmap": colmap}


def compare_second_source(spec: Spec, other: tuple[str, ...], first: pd.DataFrame, second: pd.DataFrame,
                          rules: Rules) -> dict:
    """A further source table whose rows and columns are all in the table (same row order)."""
    problems, worst = [], 0.0
    if len(first) != len(second):
        problems.append("row numbers differ")
    missing = [c for c in second.columns if c not in first.columns]
    if missing:
        problems.append(f"columns not in the table's source: {missing}")
    if not problems:
        for col in second.columns:
            a, b = first[col], second[col]
            if _is_text(a) or _is_text(b) or pd.api.types.is_bool_dtype(a):
                if not bool((a.isna() == b.isna()).all()) or \
                        not bool((a[a.notna()].astype(str) == b[a.notna()].astype(str)).all()):
                    problems.append(f"{col} differs")
            else:
                if not bool((a.isna() == b.isna()).all()):
                    problems.append(f"{col}: empty cells differ")
                    continue
                diff = float((a.astype(float) - b.astype(float)).abs().max())
                worst = max(worst, 0.0 if np.isnan(diff) else diff)
                if worst > 1e-12:
                    problems.append(f"{col} differs by {diff:.1e}")
    path = source_path(other)
    return {"sheet": spec.sheet, "table": spec.key, "title": spec.title, "source": cc.rel(path),
            "role": f"second source: its {second.shape[1]} columns and {len(second)} rows are all in "
                    f"{cc.rel(source_path(spec.src))} (same order; text equal, numbers within {worst:.1e})",
            "rows_source": len(second), "rows_other_annotation": int(rules.other_rows(second).sum()),
            "rows_kept": len(second), "columns_source": second.shape[1], "columns_kept": second.shape[1],
            "row_filter": "all rows", "columns_dropped": "", "columns_renamed": "", "values_changed": "",
            "problems": " | ".join(problems)}


def write_workbook(path, layout: Layout, specs: list[Spec], frames: dict[str, pd.DataFrame],
                   left_out: dict[str, str]) -> Book:
    book = Book()
    ws = book.wb.create_sheet("Contents")
    book.cell(ws, 1, 1, layout.title, FT, width=False)
    for i, line in enumerate(layout.intro, 2):
        book.cell(ws, i, 1, line, FN, width=False)
    row = len(layout.intro) + 3
    heads = ("sheet", "table", "content", "rows", "columns") + (("notes",) if left_out else ())
    for j, name in enumerate(heads, 1):
        c = book.cell(ws, row, j, name, FB, width=j != 3)
        c.fill, c.border = HEAD_FILL, HEAD_BORDER
    for spec in specs:
        row += 1
        if spec.key in frames:
            values = (layout.sheets[spec.sheet][0], spec.key, spec.title, len(frames[spec.key]),
                      frames[spec.key].shape[1])
        else:
            values = (layout.sheets[spec.sheet][0], spec.key, spec.title, None, None, left_out[spec.key])
        for j, value in enumerate(values, 1):
            if value is not None:
                book.cell(ws, row, j, value, F, width=j != 3)
    row += 3
    for j, name in ((1, "term or column"), (3, "meaning")):
        c = book.cell(ws, row, j, name, FB, width=False)
        c.fill, c.border = HEAD_FILL, HEAD_BORDER
    ws.cell(row=row, column=2).fill = HEAD_FILL
    ws.cell(row=row, column=2).border = HEAD_BORDER
    for term, meaning in layout.terms:           # the term overflows into the empty column B; the meaning wraps
        row += 1
        book.cell(ws, row, 1, term, F, width=False).alignment = Alignment(vertical="top")
        book.cell(ws, row, 3, meaning, F, width=False).alignment = Alignment(wrap_text=True, vertical="top")
    book.widths[("Contents", 1)] = max(len(name) for name, _ in layout.sheets.values())
    book.widths[("Contents", 3)] = 118
    for number, (name, title) in layout.sheets.items():
        ws = book.wb.create_sheet(name)
        book.cell(ws, 1, 1, f"Supplementary Table {layout.number}, sheet {number}. {title}", FT, width=False)
        lines = ("Terms and column names: see the sheet Contents.", *layout.sheet_lines.get(number, ()))
        for i, line in enumerate(lines, 2):
            book.cell(ws, i, 1, line, FN, width=False)
        row = len(lines) + 3
        for spec in specs:
            if spec.sheet == number and spec.key in frames:
                row = book.table(ws, row, spec, frames[spec.key])
    book.set_widths()
    book.wb.properties.title = f"Supplementary Table {layout.number}"
    book.wb.properties.creator = cc.GENERATED_BY
    book.wb.properties.description = f"{cc.GENERATED_TAG}; scripts/report_{layout.step}.py"
    book.wb.save(path)
    return book


# Readers see only the final products (authors, 12 Nov 2026): no cell or sheet name may refer to the revision process.
PROCESS_WORDS = re.compile(r"submitted|submission|re-?annotat", re.I)


def read_back(path, positions: dict[str, dict], forbidden: re.Pattern) -> tuple[dict[str, pd.DataFrame], dict]:
    """The tables as stored in the saved workbook, the rows found under each header, and counts over every cell."""
    wb = load_workbook(path)
    frames, rows_found = {}, {}
    for key, pos in positions.items():
        ws = wb[pos["sheet"]]
        cols = range(1, pos["n_cols"] + 1)
        header = [ws.cell(row=pos["header_row"], column=j).value for j in cols]
        rows = [[ws.cell(row=pos["header_row"] + i, column=j).value for j in cols]
                for i in range(1, pos["n_rows"] + 1)]
        frames[key] = pd.DataFrame(rows, columns=header, dtype=object)       # cells as stored: None stays None
        n, r = 0, pos["header_row"] + 1                                      # rows until the first empty row
        while r <= ws.max_row and any(ws.cell(row=r, column=j).value is not None for j in cols):
            n, r = n + 1, r + 1
        rows_found[key] = n
    stats = {"cells": 0, "not_arial": 0, "formulas": 0, "forbidden": [], "long_names": [], "process": []}
    for ws in wb.worksheets:
        if len(ws.title) > 31:
            stats["long_names"].append(ws.title)
        if PROCESS_WORDS.search(ws.title):
            stats["process"].append(f"sheet name {ws.title}")
        for row in ws.iter_rows():
            for c in row:
                if c.value is None:
                    continue
                stats["cells"] += 1
                stats["not_arial"] += c.font.name != "Arial"
                stats["formulas"] += c.data_type == "f"
                if isinstance(c.value, str) and forbidden.search(c.value):
                    stats["forbidden"].append(f"{ws.title}!{c.coordinate}: {c.value[:60]}")
                if isinstance(c.value, str) and PROCESS_WORDS.search(c.value):
                    stats["process"].append(f"{ws.title}!{c.coordinate}: {c.value[:60]}")
    stats["sheets"] = wb.sheetnames
    stats["rows_found"] = rows_found
    return frames, stats


def source_differences(stored: pd.DataFrame, source: pd.DataFrame, colmap: dict[str, str]) -> dict:
    """Numbers of the saved workbook table against the filtered source table (same rows, source column names)."""
    out = {"values": 0, "mismatch": 0, "max_abs": 0.0, "max_rel": 0.0}
    for new, old in colmap.items():
        s = source[old]
        if _is_text(s) or new not in stored.columns:
            continue                             # text: covered by the comparison with the assembled table
        for a, b in zip(s.tolist(), stored[new].tolist()):
            a = native(a)
            out["values"] += 1
            if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
                out["mismatch"] += not ((a is None and b is None) or (isinstance(a, bool) and isinstance(b, bool)
                                                                      and a == b))
                continue
            d = abs(float(a) - float(b))
            if d:
                out["max_abs"] = max(out["max_abs"], d)
                out["max_rel"] = max(out["max_rel"], d / max(abs(float(a)), abs(float(b))))
    return out


def rows_where(frame: pd.DataFrame, **sel) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for key, value in sel.items():
        mask &= frame[key].astype(str).eq(str(value))
    return frame[mask]


def build(layout: Layout, specs: list[Spec], rules: Rules, letter: list, description: str,
          extra_checks: Callable[[dict, dict, list], None] | None = None) -> None:
    """Read the sources, assemble the tables, write the workbook, read it back, check, record."""
    parser = argparse.ArgumentParser(description=description)
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, layout.step, args)
    checks: list[dict] = []

    frames, records, state, left_out, raw = {}, [], {}, {}, {}
    n_files = 0
    for spec in specs:
        for src in (spec.src, *spec.also):
            path = source_path(src)
            key = cc.rel(path)
            if key in raw or not path.exists():
                continue
            run.use_input(path, f"{src[0]} {src[1]}")
            raw[key] = pd.read_csv(path)
            n_files += 1
        key = cc.rel(source_path(spec.src))
        missing = [cc.rel(source_path(s)) for s in (spec.src, *spec.also) if cc.rel(source_path(s)) not in raw]
        if missing:
            left_out[spec.key] = "left out: source file not found (" + ", ".join(missing) + ")"
            continue
        frames[spec.key], record, state[spec.key] = assemble_table(spec, raw[key], rules)
        record["source_sha256"] = run.inputs[key]["sha256"]
        records.append(record)
        for other in spec.also:
            record = compare_second_source(spec, other, raw[key], raw[cc.rel(source_path(other))], rules)
            record["source_sha256"] = run.inputs[cc.rel(source_path(other))]["sha256"]
            records.append(record)
    sources = pd.DataFrame(records)

    listed = sorted({cc.rel(source_path(s)) for spec in specs for s in (spec.src, *spec.also)})
    check(checks, "source files of the sheet plan found and read", f"{n_files} of {len(listed)} files; tables left "
          f"out: {len(left_out)}", f"{len(listed)} of {len(listed)}; none left out",
          n_files == len(listed) and not left_out)
    problems = sources.loc[sources["problems"].ne(""), ["table", "problems"]]
    check(checks, "assembly: row filters, constant and repeated columns, value names, unique column names, no "
          "internal name in a column name, no empty table",
          "; ".join(f"{r.table}: {r.problems}" for r in problems.itertuples()) or f"no problem in {len(frames)} tables",
          "no problem", problems.empty)
    by_source = sources[sources["role"].eq("values copied from this table")].groupby("source", sort=False).agg(
        rows_source=("rows_source", "first"), other=("rows_other_annotation", "first"), kept=("rows_kept", "sum"),
        tables=("table", lambda s: ", ".join(s)))
    wrong = by_source[by_source["rows_source"] - by_source["other"] != by_source["kept"]]
    check(checks, "rows written per source table = rows of the source minus the rows of another annotation "
          "(summed over the tables that share a source)",
          "; ".join(f"{Path(i).name}: {r.rows_source} - {r.other} != {r.kept}" for i, r in wrong.iterrows()) or
          f"{len(by_source)} source tables: {int(by_source['rows_source'].sum())} rows, "
          f"{int(by_source['other'].sum())} of another annotation, {int(by_source['kept'].sum())} written",
          "equal for every source table", wrong.empty and len(by_source) > 0)
    second = sources[sources["role"].ne("values copied from this table")]
    if len(second):
        check(checks, "second source tables: every row and column is in the table (text equal, numbers within 1e-12)",
              "; ".join(f"{Path(r.source).name}: {r.problems or 'equal'}" for r in second.itertuples()),
              "equal", bool(second["problems"].eq("").all()))
    if extra_checks is not None:
        extra_checks(frames, state, checks)

    out = cc._check_writable(cc.OUT_ROOTS["tables"] / TASK / layout.workbook, run.overwrite)
    book = write_workbook(out, layout, specs, frames, left_out)
    run.outputs.append(cc.rel(out))
    print(f"Saved workbook: {cc.rel(out)}")

    stored, stats = read_back(out, book.positions, rules.forbidden)
    names = ["Contents", *[layout.sheets[s][0] for s in sorted(layout.sheets)]]
    check(checks, "workbook sheets: Contents, then the numbered sheets; names of at most 31 characters that start "
          "with the sheet number", ", ".join(stats["sheets"]), ", ".join(names),
          stats["sheets"] == names and not stats["long_names"]
          and all(n.startswith(f"{s} ") for s, (n, _) in layout.sheets.items()))
    short = {k: (len(f), stats["rows_found"][k]) for k, f in frames.items() if stats["rows_found"][k] != len(f)}
    check(checks, "rows found under each table header of the saved workbook = rows of the table",
          short or f"{len(frames)} tables, {int(sum(len(f) for f in frames.values()))} rows", "equal for every table",
          not short)
    differ, largest = [], 0.0
    for key, frame in frames.items():
        got = stored[key]
        if list(got.columns) != [str(c) for c in frame.columns] or got.shape != frame.shape:
            differ.append(f"{key}: shape or header")
            continue
        diffs = [rel_diff(native(a), b) for row_a, row_b in zip(frame.itertuples(index=False, name=None),
                                                                 got.itertuples(index=False, name=None))
                 for a, b in zip(row_a, row_b)]
        n = sum(d > REL_TOL for d in diffs)
        largest = max([largest, *[d for d in diffs if d <= REL_TOL]])
        if n:
            differ.append(f"{key}: {n} cells")
    check(checks, "round trip: every table read back from the saved workbook equals the assembled table (header, "
          "shape, text and whole numbers exactly, other numbers to the 16 significant digits a workbook stores)",
          "; ".join(differ) or f"{len(frames)} tables, {int(sum(f.size for f in frames.values()))} values equal; "
          f"largest relative difference {largest:.1e}", f"all tables equal; relative difference <= {REL_TOL:.0e}",
          not differ)
    total = {"values": 0, "mismatch": 0, "max_abs": 0.0, "max_rel": 0.0}
    bad = []
    for key in frames:
        d = source_differences(stored[key], state[key]["source"], state[key]["colmap"])
        total["values"] += d["values"]
        total["mismatch"] += d["mismatch"]
        total["max_abs"], total["max_rel"] = max(total["max_abs"], d["max_abs"]), max(total["max_rel"], d["max_rel"])
        if d["mismatch"] or d["max_rel"] > REL_TOL:
            bad.append(f"{key}: {d['mismatch']} cells, relative difference {d['max_rel']:.1e}")
    check(checks, "numbers read back from the saved workbook = numbers of the filtered source tables (numeric and "
          "true/false columns; empty cells stay empty)",
          "; ".join(bad) or f"{total['values']} values in {len(frames)} tables; largest absolute difference "
          f"{total['max_abs']:.1e}, largest relative difference {total['max_rel']:.1e}",
          f"no mismatch; relative difference <= {REL_TOL:.0e} (float text rounding)", not bad)
    check(checks, "workbook cells: Arial, no formula", f"{stats['cells']} cells with a value; not Arial "
          f"{stats['not_arial']}; formulas {stats['formulas']}", "not Arial 0; formulas 0",
          stats["not_arial"] == 0 and stats["formulas"] == 0)
    check(checks, "no internal name in any cell of the workbook (task codes, internal column names, other "
          "annotation, file names)", "; ".join(stats["forbidden"][:12]) or "none", "none", not stats["forbidden"])
    check(checks, "no cell and no sheet name of the saved workbook contains 'submitted', 'submission' or "
          "'re-annotation' (readers see only the final products)",
          f"{len(stats['process'])} cells: " + "; ".join(stats["process"][:12]) if stats["process"] else
          f"none in {stats['cells']} cells with a value and {len(stats['sheets'])} sheet names", "none",
          not stats["process"])

    found = []
    for text, key, getter, expected, decimals in letter:
        try:
            value = getter(stored[key])
            value = value.item() if isinstance(value, np.generic) else value
            ok = (value == expected if decimals is None
                  else abs(float(value) - float(expected)) <= 0.5 * 10 ** -decimals + 1e-9)
        except Exception as error:                # a missing row or column is a failed check, not a crash
            value, ok = f"error: {error}", False
        found.append((text, key, value, expected, bool(ok)))
    wrong = [f"{t} (table {k}): {v} vs {e}" for t, k, v, e, ok in found if not ok]
    check(checks, f"numbers of the thread's letter (section 2) found in the saved workbook at the written precision",
          f"{sum(ok for *_, ok in found)} of {len(found)}" + (f"; not matched: {wrong}" if wrong else ""),
          f"{len(found)} of {len(found)}", not wrong)

    for c in checks:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['observed']}")
    run.table(sources, f"{layout.step}_sources")
    run.table(pd.DataFrame(checks), f"{layout.step}_checks")
    all_ok = all(c["pass"] for c in checks)
    per_sheet = {layout.sheets[n][0]: {"tables": sum(s.sheet == n and s.key in frames for s in specs),
                                       "rows": int(sum(len(frames[s.key]) for s in specs
                                                       if s.sheet == n and s.key in frames))}
                 for n in sorted(layout.sheets)}
    for name, d in per_sheet.items():
        print(f"  sheet {name}: {d['tables']} tables, {d['rows']} rows")
    run.finish({"workbook": cc.rel(out), "tables": len(frames), "tables_left_out": left_out,
                "rows": int(sum(len(f) for f in frames.values())),
                "values": int(sum(f.size for f in frames.values())), "sheets": per_sheet,
                "letter_numbers_checked": len(found),
                "style": "Contents sheet and numbered data sheets; Arial 10; values only; 16 significant digits "
                         "with a display format", "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit(f"A check failed; see report_{layout.step}_checks.csv.")


# ---------------------------------------------------------------------------
# Supplementary Table S-08-d
# ---------------------------------------------------------------------------
COL = "kidney_t15_label"
OTHER_ANNOTATIONS = ("kidney_pre_label", "kidney_reannot_label")
KIDNEY_RNA, RNA_PROTEIN_INTEGRATION = "kidney_rna", "rna_protein_integration"
ONLY = ("label_column", "in", (COL,))
NOT_OTHER_REFERENCE = ("reference", "not in", OTHER_ANNOTATIONS)
SCP_LABELS = frozenset({"Endothelial", "JG", "PCT", "PST", "TAL", "DCT", "CNT/PC", "IC-B", "Unknown",
                        "no counterpart"})
ENDOTHELIAL_ROW = "Endothelial (row added to Supplementary Table 1)"
TYPE_NAMES = {"Endothelial (kidney_annotation extension; atlas Endo / AVR markers)": ENDOTHELIAL_ROW}
NOTES_DROPPED = "working note: names tasks, the submitted annotation or decisions"


def other_annotation_rows(src: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=src.index)
    for col in ("label_column", "reference"):
        if col in src.columns:
            mask |= src[col].isin(OTHER_ANNOTATIONS)
    if "n_permutations" in src.columns:          # chance table: 8 cell types = the kidney_pre_label comparison
        mask |= src["n_compartments"].eq(8)
    return mask


RULES = Rules(
    label_display={"DCT1": "DCT", "PC": "CNT/PC"},
    label_allowed=SCP_LABELS,
    column_values={
        "reference": {"all": "all atlas cell types", COL: "matched atlas cell types"},
        "config": {"features_scp10": "genes_scp10"},
        "feature_group": {"other shared features": "other shared genes"},
        "type_full_name": TYPE_NAMES,
    },
    strict=frozenset({"reference"}),
    rename_exact={
        "t07_centroid_expected_top": "centroid_mapping_expected_top",
        "t07_centroid_expected_top_incl_mixed": "centroid_mapping_expected_top_incl_mixed",
        "t07_class_across_compartments": "class_across_cell_types",
        "t07_spearman_rho": "spearman_rho_across_cell_types",
        "in_atlas_X_3000_genes": "in_atlas_3000_variable_genes", "in_atlas_raw": "in_atlas",
        "in_features_scp10": "in_config_genes_scp10", "use_in_rna_protein_integration": "use",
        "ilisi_perfect_mixing": "ilisi_random_mixing", "ilisi_perfect_scib": "ilisi_minus_1_random_mixing",
        "workbook_position": "position_in_marker_list", "type_pos": "cell_type_order", "wb_pos": "marker_order",
    },
    rename_re=((re.compile(r"ilisi_perfect_scib"), "ilisi_minus_1_random_mixing"),
               (re.compile(r"ilisi_scib"), "ilisi_minus_1"),
               (re.compile(r"compartment"), "cell_type"), (re.compile(r"feature"), "gene")),
    forbidden=re.compile(r"(?<![A-Za-z0-9])t\d\d(?![0-9A-Za-z])|kidney_(t15|pre|reannot)|label_column|fig ?2[ab]"
                         r"|(?<![A-Za-z])pre(?![A-Za-z])|reannot|\bbrief\b|\buser\b"
                         r"|(?<![A-Za-z0-9])v[123](?![A-Za-z0-9])|mka|\.h5ad|\.csv|compartment|feature", re.I),
    forbidden_header=re.compile(r"scib|t07|t10|t15|compartment|feature|label_column", re.I),
    other_rows=other_annotation_rows,
)

SHEETS = {
    1: ("1 Matched atlas cell types", "Atlas cell types matched to each SCP-MS cell type"),
    2: ("2 Shared genes", "Genes quantified in both data sets and their detection"),
    3: ("3 Atlas subsample and runs", "Atlas subsample, embedding runs and software"),
    4: ("4 Mixing and separation", "Modality mixing and cell-type separation, with the values plotted in "
                                   "Supplementary Figures S-08-a to S-08-c"),
    5: ("5 Label transfer", "Label transfer, and the summary of all integration metrics over the atlas draws"),
    6: ("6 Concordance classes", "Detection and concordance classes of the shared genes, per cell type, and the "
                                 "chance distribution"),
    7: ("7 Markers and transporters", "Annotation markers (with the values plotted in Figure F-08) and "
                                      "transporters and channels"),
    8: ("8 Per gene", "Per gene: detection, correlation across cell types and class"),
}
FIGURE_VALUES = ("report", "08_integration_figures_values")
MARKER_VALUES = ("report", "08_marker_concordance_values")
SUMMARY = (RNA_PROTEIN_INTEGRATION, "metrics_summary")
SUMMARY_NOTE = "Median, min and max over the five atlas draws (n_draws)."

SPECS = [
    Spec("1a", (KIDNEY_RNA, "celltype_mapping"), "Atlas cell types matched to each SCP-MS cell type",
         ("One row per SCP-MS cell type. atlas_celltype_expected: the matched atlas cell types (level Celltype); "
          "atlas_celltype_mixed_accepted: atlas classes that do not separate the cell type (see Contents). "
          "n_atlas_ columns: atlas cells of these classes (all; single cells; nuclei).",
          "atlas_finest_ columns: the same at the finer level Celltype_finest. marker_support: see Contents. JG and "
          "Unknown have no atlas counterpart."),
         rows=(ONLY,), constant=("label_column", "label_role", "status"), display=("scp_label",),
         drop={"scp_label_full_name": "names of the marker list, which differ from the display names; the "
                                      "abbreviations are in Contents",
               "used_in": NOTES_DROPPED, "source": NOTES_DROPPED, "rationale": NOTES_DROPPED,
               "t15_observation": NOTES_DROPPED, "decision_for_user": NOTES_DROPPED, "user_decision": NOTES_DROPPED,
               "alternative_mapping": NOTES_DROPPED}),
    Spec("2a", (RNA_PROTEIN_INTEGRATION, "features"), "The 3,569 genes quantified in both data sets",
         ("protein, atlas_gene: the matched pair. scp_n_cells_detected, scp_detection_rate: of the 712 SCP-MS "
          "cells. atlas_ columns: all atlas cells, single cells (Cell) and nuclei (Nuclei).",
          "in_config_genes_scp10: detected in >= 10% of the SCP-MS cells (configuration genes_scp10). "
          "marker_listed_for, transporter_channel: see Contents."),
         same_as={"scp_detection_rate_recomputed": "scp_detection_rate"}, also=((KIDNEY_RNA, "shared_features"),)),
    Spec("3a", (RNA_PROTEIN_INTEGRATION, "rna_subsample_summary"), "Atlas cells drawn per atlas cell type and draw",
         ("n_eligible: atlas cells of the cell type; n_drawn: cells drawn (200, or all if fewer: taken_whole). "
          "n_nuclei: nuclei among the drawn cells; n_finest_known: drawn cells with a label at the level "
          "Celltype_finest.",
          "design, seed: see Contents (design; draw)."), rename={"Celltype": "atlas_cell_type"}),
    Spec("3b", (RNA_PROTEIN_INTEGRATION, "runs"), "Embedding runs: configuration, Harmony settings and cells",
         ("One row per configuration and draw. n_scp, n_rna: SCP-MS and atlas cells; n_genes: genes used. Columns: "
          "see Contents (config; reference; Harmony columns).",), rows=(NOT_OTHER_REFERENCE,)),
    Spec("3c", (RNA_PROTEIN_INTEGRATION, "pc_modality_variance"), "Joint principal components: variance within each modality",
         ("Embedding A, primary configuration, first draw. var_scp, var_rna: variance of the scores of a "
          "component within the SCP-MS cells and within the atlas cells; share_scp = var_scp / (var_scp + var_rna).",)),
    Spec("3d", (RNA_PROTEIN_INTEGRATION, "packages"), "Software and stored atlas embeddings",
         ("Packages with their version and use; 'not installed' packages were not used.",),
         values={"package": {"mka.h5ad obsm": "Mouse Kidney Atlas file, stored embeddings"}}),
    Spec("4a", FIGURE_VALUES, "Values of Supplementary Figure S-08-a: cells per cell type",
         ("n_scp: SCP-MS cells of the cell type. n_atlas: atlas cells of its matched atlas cell types in the draw "
          "shown; no counterpart: atlas cells of the atlas cell types without an SCP-MS counterpart.",),
         rows=(("figure", "in", ("joint_umaps",)),), constant=("figure",), drop_empty=True, display=("label",),
         same_as={"display": "label"}),
    Spec("4b", FIGURE_VALUES, "Values of Supplementary Figure S-08-b: mixing within each cell type",
         ("Median, min and max over the five atlas draws. metric: scaled_other_fraction_scp, atlas-neighbour "
          "fraction of the SCP-MS cells / random; ilisi_minus_1_scp; ilisi_minus_1_random_mixing (see Contents, "
          "Mixing within cell types).",),
         rows=(("figure", "in", ("mixing",)),), constant=("figure",), drop_empty=True, display=("label",),
         same_as={"display": "label"},
         values={"metric": {"scaled_other_fraction_scp": "scaled_other_fraction_scp",
                            "ilisi_scib_scp": "ilisi_minus_1_scp",
                            "ilisi_random_mixing": "ilisi_minus_1_random_mixing"}}),
    Spec("4c", FIGURE_VALUES, "Values of Supplementary Figure S-08-c: SCP-MS cells per transferred atlas cell type",
         ("Embedding harmony, one atlas draw, k 15. n: cells of the SCP-MS cell type (label) that received the "
          "atlas cell type (atlas_class); fraction: of the cells of the label. matched, mixed: matched atlas cell "
          "type; accepted mixed class.",),
         rows=(("figure", "in", ("transfer",)),), constant=("figure",), drop_empty=True, display=("label",),
         same_as={"display": "label"}),
    Spec("4d", FIGURE_VALUES, "Values of Supplementary Figure S-08-c: fraction with a matched atlas cell type",
         ("The same draw. accuracy_strict: fraction of the cells that received a matched atlas cell type; "
          "accuracy_incl_mixed: also counting the accepted mixed classes.",),
         rows=(("figure", "in", ("transfer_accuracy",)),), constant=("figure",), drop_empty=True,
         display=("label",), same_as={"display": "label"}),
    Spec("4e", (RNA_PROTEIN_INTEGRATION, "mixing_global"), "Modality mixing over all cells, per configuration and draw",
         ("Embeddings pca and harmony. Columns: see Contents (Mixing, all cells).",), rows=(NOT_OTHER_REFERENCE,)),
    Spec("4f", (RNA_PROTEIN_INTEGRATION, "mixing_per_type"), "Modality mixing within each cell type, per configuration and draw",
         ("cell_type: SCP-MS cell type; atlas_classes: its matched atlas cell types. Columns: see Contents (Mixing "
          "within cell types).",),
         rows=(ONLY,), constant=("label_column",), drop_empty=True, display=("compartment",)),
    Spec("4g", (RNA_PROTEIN_INTEGRATION, "separation"), "Cell-type separation, per configuration and draw",
         ("Embeddings pca, harmony and scvi (no clisi for scvi). Columns: see Contents (Separation).",),
         rows=(ONLY,), constant=("label_column",)),
    Spec("5a", SUMMARY, "Label transfer: summary over the atlas draws",
         (SUMMARY_NOTE + " One row per configuration, embedding, atlas level and k; primary rows: config primary, "
          "level Celltype, k 15.", "Columns: see Contents (Label transfer)."),
         rows=(("metric_group", "in", ("transfer",)), ONLY), constant=("metric_group", "label_column"),
         drop_empty=True),
    Spec("5b", (RNA_PROTEIN_INTEGRATION, "transfer_summary"), "Label transfer per configuration, embedding, atlas level, k and draw",
         ("n_cells_with_counterpart: SCP-MS cells of the cell types with an atlas counterpart; n_labels_macro_f1: "
          "these cell types. Columns: see Contents (Label transfer).",),
         rows=(ONLY,), constant=("label_column",)),
    Spec("5c", (RNA_PROTEIN_INTEGRATION, "transfer_per_label"), "Label transfer per SCP-MS cell type",
         ("One row per configuration, draw, embedding, atlas level, k and cell type. expected_atlas: matched atlas "
          "cell types. Columns: see Contents (Label transfer).",
          "JG and Unknown have no atlas counterpart: top_predicted lists their most frequent atlas cell types "
          "(first draw)."),
         rows=(ONLY,), constant=("label_column",), display=("scp_label",)),
    Spec("5d", (RNA_PROTEIN_INTEGRATION, "transfer_confusion"), "Transferred atlas cell types per SCP-MS cell type",
         ("Primary configuration, first draw, k 15; all embeddings and both atlas levels. n: cells of the SCP-MS "
          "cell type that received the atlas cell type; fraction_of_label: of the cells of the cell type.",
          "is_expected: matched atlas cell type; is_mixed_accepted: accepted mixed class."),
         rows=(ONLY,), constant=("label_column",), display=("scp_label",)),
    Spec("5e", SUMMARY, "Modality mixing over all cells: summary over the atlas draws",
         (SUMMARY_NOTE + " Per-draw values: table 4e.",),
         rows=(("metric_group", "in", ("mixing_global",)), NOT_OTHER_REFERENCE), constant=("metric_group",),
         drop_empty=True),
    Spec("5f", SUMMARY, "Modality mixing within each cell type: summary over the atlas draws",
         (SUMMARY_NOTE + " Per-draw values: table 4f.",),
         rows=(("metric_group", "in", ("mixing_per_type",)), ONLY), constant=("metric_group", "label_column"),
         drop_empty=True, display=("compartment",)),
    Spec("5g", SUMMARY, "Cell-type separation: summary over the atlas draws",
         (SUMMARY_NOTE + " Per-draw values: table 4g.",),
         rows=(("metric_group", "in", ("separation",)), ONLY), constant=("metric_group", "label_column"),
         drop_empty=True),
    Spec("6a", (KIDNEY_RNA, "concordance_class_counts"), "Shared genes per detection and concordance class",
         ("The 3,569 shared genes at each detection threshold, for all atlas cells and for atlas nuclei only "
          "(rna_source). fraction: of the 3,569 genes. Classes: see Contents (Detection classes).",),
         rows=(ONLY,), constant=("label_column",)),
    Spec("6b", (KIDNEY_RNA, "concordance_compartments"), "Detection and RNA-protein correlation within each cell type",
         ("Per cell type, atlas source and threshold. Columns: see Contents (Per cell type).",),
         rows=(ONLY,), constant=("label_column",), display=("compartment",)),
    Spec("6c", (KIDNEY_RNA, "concordance_null"), "Chance distribution of the correlation across 7 cell types",
         ("Spearman correlation of two unrelated rankings of 7 cell types, over all 5,040 orderings: fraction of "
          "the orderings in each class, and the median correlation.",),
         rows=(("n_compartments", "in", (7,)),),
         rename={"p_rho_ge_0.5_concordant": "fraction_rho_ge_0.5_concordant",
                 "p_rho_0.2_to_0.5_intermediate": "fraction_rho_0.2_to_0.5_intermediate",
                 "p_rho_lt_0.2_discordant": "fraction_rho_lt_0.2_discordant",
                 "n_compartments": "n_cell_types"}),
    Spec("7a", MARKER_VALUES, "Values of Figure F-08A: shared genes per class and detection threshold",
         ("All atlas cells. labelled: the number is printed in the figure.",),
         rows=(("panel", "in", ("A",)),), constant=("panel", "label_column"), drop_empty=True),
    Spec("7b", MARKER_VALUES, "Values of Figure F-08B: the 98 annotation markers",
         ("One row per marker, under the first cell type that lists it (row_label). spearman_rho: across the 7 "
          "cell types; class at the 10% threshold. coverage: see Contents.",
          "plotted_x: position in the figure (markers without a correlation are drawn in the 'no rho' column). "
          "colour_or_shape: colour of the dot (hex), or symbol (v, triangle; s, square; x, cross)."),
         rows=(("panel", "in", ("B",)),), constant=("panel",), drop_empty=True),
    Spec("7c", (KIDNEY_RNA, "paired_marker_rho"), "Annotation markers: correlation across cell types, class and coverage",
         ("One row per marker (98), under the first cell type that lists it; all atlas cells, threshold 10%. "
          "detected_both: detected in both modalities; mean_detection: mean of max_rna_detection and "
          "max_protein_detection.",
          "chance_q025, chance_q975: central 95% of the correlation for unrelated rankings of 7 cell types. "
          "cell_type_order, marker_order: order in the marker list."),
         rows=(ONLY,), constant=("label_column",), display=("compartment_max_rna", "compartment_max_protein"),
         rename={"label": "row_label"}),
    Spec("7d", (KIDNEY_RNA, "concordance_markers"), "Annotation markers: class per atlas source and threshold",
         ("One row per marker, listing cell type (listed_for), atlas source and threshold. Columns: see Contents "
          "(Detection classes; max_rna_detection).",),
         rows=(ONLY,), constant=("label_column",), display=("compartment_max_rna", "compartment_max_protein")),
    Spec("7e", (KIDNEY_RNA, "marker_coverage"), "Annotation marker list: coverage in the SCP-MS data and in the atlas",
         ("One row per cell type of the marker list (type) and marker; listed_for: all cell types that list the "
          "marker. in_protein_matrix: quantified as protein; in_atlas: among the atlas genes.",
          "atlas_detection_rate_ columns: all atlas cells, single cells and nuclei."),),
    Spec("7f", (RNA_PROTEIN_INTEGRATION, "strengths_markers"), "Annotation markers per cell type: detection as RNA and as protein",
         ("The 45 quantified annotation markers that are listed for one of the SCP-MS cell types, in each of the 7 "
          "cell types with an atlas counterpart; threshold 10%. rna_detection, protein_detection: detected "
          "fractions in the cell type.",
          "detection_class and the class across cell types: see Contents (detection_class). marker_group: cell "
          "type under which the marker is shown."),
         rows=(ONLY,), constant=("label_column",), display=("compartment", "marker_group")),
    Spec("7g", (RNA_PROTEIN_INTEGRATION, "strengths_per_type"), "Transporters and channels against other genes, per cell type",
         ("Genes detected in both modalities, as RNA only, as protein only or in neither, in each cell type; "
          "threshold 10%. gene_group: 151 transporters and channels; the other 3,418 shared genes.",),
         rows=(ONLY,), constant=("label_column",), display=("compartment",)),
    Spec("7h", (RNA_PROTEIN_INTEGRATION, "strengths_classes"), "Transporters and channels against other genes: classes across cell types",
         ("Number and fraction of the genes of each group per class; median_rho_detected_both: median correlation "
          "of the genes detected in both modalities.",),
         rows=(ONLY,), constant=("label_column",)),
    Spec("7i", (RNA_PROTEIN_INTEGRATION, "transporter_channel_rule"), "Gene-symbol prefixes that define transporters and channels",
         ("A gene is counted as transporter or channel if its symbol starts with an included prefix and with no "
          "excluded prefix. n_shared_genes, shared_genes: shared genes with the prefix.",)),
    Spec("8a", (KIDNEY_RNA, "concordance_features"), "Per gene: detection, correlation across cell types and class",
         ("The 3,569 shared genes, for each atlas source (rna_source) and detection threshold: 6 rows per gene. "
          "Primary rows: rna_source all, threshold 0.1.",
          "Columns: see Contents (Detection classes; max_rna_detection)."),
         rows=(ONLY,), constant=("label_column",), display=("compartment_max_rna", "compartment_max_protein")),
]

TERMS = [
    ("SCP-MS cell types", "Cell types of the 712 kidney SCP-MS cells as in Figure 2A: PCT (proximal convoluted "
     "tubule) 160, PST (proximal straight tubule) 158, TAL (thick ascending limb) 138, DCT (distal convoluted "
     "tubule) 58, CNT/PC (connecting tubule / principal cells) 38, JG (juxtaglomerular cells) 24, Endothelial 19 "
     "and IC-B (type B intercalated cells) 8 cells. Unknown (109 cells): cells without an assigned cell type."),
    ("Atlas; rna", "Mouse Kidney Atlas (Novella-Rausell et al., iScience 2023): single-cell and single-nucleus "
     "RNA-seq of healthy mouse kidneys from eight studies (141,401 cells and nuclei; 16,711 genes). Columns with "
     "rna or atlas refer to atlas cells. Atlas cell types carry the atlas names at the annotation level Celltype "
     "(35 classes); Celltype_finest is the finer, study-specific level. RNA and protein come from different "
     "animals and studies."),
    ("Matched atlas cell types", "Each SCP-MS cell type is compared with fixed atlas cell types (table 1a). "
     "expected, matched: these atlas cell types. mixed accepted: atlas classes that do not separate the cell type "
     "(LOH, loop of Henle; DCT-CNT; CD-Trans, collecting-duct transitional cells; at the finer level also PT and "
     "IC); they count only in the incl_mixed values. JG and Unknown have no atlas counterpart."),
    ("marker_support", "Atlas cell types ranked by the mean RNA level of the markers that Supplementary Table 1 "
     "lists for the SCP-MS cell type (per gene, mean log-normalized expression rescaled 0 to 1 across the atlas "
     "cell types). top3: the three highest with their value; rank_of_expected: best rank of a matched atlas cell "
     "type; excl_mixed: without the mixed classes. n_markers_in_atlas: listed markers among the atlas genes."),
    ("Shared genes", "The 3,569 genes quantified as protein in the SCP-MS data (4,054 proteins) and present among "
     "the atlas genes, matched by gene symbol."),
    ("detection; detected", "Fraction (or number) of cells with intensity > 0 (protein) or at least one count "
     "(RNA)."),
    ("mean_lognorm; mean_ln1p", "RNA: counts scaled to 10,000 per cell, ln(1 + x), mean over the cells. Protein: "
     "mean ln(1 + intensity), missing values set to 0."),
    ("listed_for; marker_listed_for; type", "Cell types of the annotation marker list (Supplementary Table 1 with "
     "the added endothelial row; 98 markers) that list the gene, with the names of that list (for example DCT1, "
     "DCT2, CNT, PC)."),
    ("transporter_channel", "Transporters and channels, selected by gene-symbol prefix (table 7i); 151 of the "
     "shared genes. This is not a curated annotation."),
    ("draw; seed", "Five independent draws of atlas cells (seeds 42 to 46): 200 cells per atlas cell type, smaller "
     "classes taken whole; 6,112 atlas cells per draw (4,428 with nuclei only). Values are given per draw, or as "
     "median, min and max over the draws."),
    ("design; rna_source", "all: atlas cells and nuclei. nuclei: atlas nuclei only."),
    ("embedding", "pca: joint principal component analysis (30 components) without correction (embedding A). "
     "harmony: the same with Harmony correction, modality as batch (B). scvi: the atlas authors' scVI latent "
     "space, each SCP-MS cell placed at the mean position of its 15 most similar atlas cells (C). shared: no "
     "embedding; nearest atlas cells by cosine distance on the shared genes."),
    ("config", "primary: all atlas cell types, 3,569 genes, both modalities weighted equally in the principal "
     "component analysis (pca_mode equal), Harmony theta 1. Each other configuration changes one setting: "
     "genes_scp10, the 1,731 genes detected in >= 10% of the SCP-MS cells; plain_pca, principal component "
     "analysis of the concatenated cells (pca_mode plain); harmony_theta2, Harmony theta 2; counterpart_reference, "
     "atlas cells of the matched atlas cell types only; nuclei, atlas nuclei only."),
    ("reference", "Atlas cells used: all atlas cell types, or the matched atlas cell types only (configuration "
     "counterpart_reference)."),
    ("Harmony columns", "harmonypy 0.0.10; batch = modality. harmony_theta, harmony_sigma, harmony_lambda, "
     "harmony_clusters: settings. harmony_iterations, harmony_objective_last: iterations run and last value of the "
     "objective. harmony_converged: relative change of the objective below 1e-4 within 10 iterations. runtime_s: "
     "run time in seconds."),
    ("Mixing, all cells", "rna_neighbour_fraction_scp: fraction of atlas cells among the 15 nearest neighbours of "
     "the SCP-MS cells; rna_neighbour_fraction_expected: its value under random mixing; scaled_ = observed / "
     "expected (0, only SCP-MS neighbours; 1, random mixing). ilisi: integration local inverse Simpson's index "
     "(perplexity 30); median_ilisi_scp, _rna, _all: median over the SCP-MS cells, the atlas cells and all cells; "
     "ilisi_minus_1: that median minus 1 (0, one modality; 1, equal shares); ilisi_random_mixing and "
     "ilisi_minus_1_random_mixing: the value under random mixing."),
    ("Mixing within cell types", "The SCP-MS cells of a cell type together with the atlas cells of its matched "
     "atlas cell types (n_scp, n_rna; scp_share = n_scp / (n_scp + n_rna)); 15 nearest neighbours within this set; "
     "iLISI perplexity 5. other_modality_fraction_scp: fraction of atlas cells among the neighbours of the SCP-MS "
     "cells; other_modality_fraction_rna: fraction of SCP-MS cells among the neighbours of the atlas cells; "
     "_expected and scaled_ as above. In cell types with few SCP-MS cells (IC-B, 8; Endothelial, 19) an SCP-MS "
     "cell cannot have many SCP-MS neighbours, which raises the values."),
    ("Separation", "Cell types with an atlas counterpart, the atlas cells carrying the SCP-MS cell type that "
     "their atlas cell type is matched to (n_cells, n_labels: cells and labels compared). clisi: cell-type local "
     "inverse Simpson's index; scaled_clisi = (L - median) / (L - 1) for L labels (1, neighbourhoods of one "
     "label). silhouette_joint, _rna, _scp: silhouette width (Euclidean) over all cells, the atlas cells and the "
     "SCP-MS cells."),
    ("Label transfer", "Each SCP-MS cell takes the most frequent atlas cell type among its k nearest atlas cells "
     "(k 15; also 5 and 30 for the primary configuration). level: atlas annotation level. accuracy_strict: "
     "fraction of the cells that received a matched atlas cell type; accuracy_incl_mixed: also counting the "
     "accepted mixed classes. macro_f1: mean F1 over the cell types with a counterpart. chance_accuracy_strict: "
     "accuracy expected from the class frequencies of the reference. precision, recall, f1: per cell type. "
     "centroid_mapping_expected_top: fraction of the cells nearest to the centroid of a matched atlas cell type "
     "(a separate mapping to atlas centroids, not a nearest-neighbour transfer)."),
    ("Detection classes", "threshold: a gene is detected in a modality if at least this fraction of the cells of "
     "at least one cell type has intensity > 0 (protein) or at least one count (RNA); 0.1 is the primary "
     "threshold. The 7 cell types with an atlas counterpart are compared. Genes detected in both modalities are "
     "classed by the Spearman correlation (spearman_rho) of their mean RNA and mean protein levels across the "
     "cell types: concordant, rho >= 0.5; intermediate, 0.2 <= rho < 0.5; discordant, rho < 0.2. Other classes: "
     "RNA-detected only; Protein-detected only; Detected in neither. Markers not in both data sets: not "
     "quantified as protein; not in atlas."),
    ("max_rna_detection; max_protein_detection", "Highest detected fraction over the cell types, and the cell "
     "type where it occurs (cell_type_max_rna, cell_type_max_protein). n_cell_types_rna_detected, "
     "n_cell_types_protein_detected: cell types at or above the threshold. spearman_defined: the RNA and the "
     "protein means both vary across the cell types."),
    ("Per cell type (table 6b)", "n_rna_detected, n_protein_detected, n_detected_both: genes at or above the "
     "threshold in the cell type. spearman_rho_detected_both, spearman_rho_all_shared: Spearman correlation of "
     "mean RNA and mean protein levels across the genes detected in both, and across all 3,569 shared genes."),
    ("coverage", "protein and atlas: quantified as protein and among the atlas genes; atlas only (protein not "
     "quantified); protein only (gene not in atlas); neither."),
    ("detection_class (table 7f)", "Detection of a marker within one cell type at the 10% threshold: detected in "
     "both, RNA only, protein only, detected in neither, not in atlas. class_across_cell_types, "
     "spearman_rho_across_cell_types: class and correlation of the gene across the 7 cell types."),
]

PRIMARY = dict(config="primary", level="Celltype", k=15)
BOTH = "Detected in both; "


def _median(frame: pd.DataFrame, col: str, **sel) -> float:
    return float(np.median([float(v) for v in rows_where(frame, **sel)[col]]))


def _names(frame: pd.DataFrame, **sel) -> str:
    return ", ".join(sorted(rows_where(frame, **sel)["marker"].astype(str)))


def _concordant(frame: pd.DataFrame, group: str, col: str) -> float:
    return float(_pick(frame, col, rna_source="all", threshold=0.1, gene_group=group))


LETTER = [
    ("genes quantified in both data sets", "2a", lambda f: len(f), 3569, None),
    ("atlas cells per draw (first draw, cells and nuclei)", "3a",
     lambda f: int(sum(int(v) for v in rows_where(f, design="all", seed=42)["n_drawn"])), 6112, None),
    ("atlas cell types drawn", "3a", lambda f: len(rows_where(f, design="all", seed=42)), 35, None),
    ("mixing over all cells, without correction: atlas neighbours / random", "5e",
     lambda f: _pick(f, "scaled_rna_neighbour_fraction_scp_median", config="primary", embedding="pca"), 0.040, 3),
    ("mixing over all cells, Harmony: atlas neighbours / random", "5e",
     lambda f: _pick(f, "scaled_rna_neighbour_fraction_scp_median", config="primary", embedding="harmony"),
     0.473, 3),
    *[(f"mixing within {name}, Harmony: atlas neighbours / random", "5f",
       lambda f, name=name: _pick(f, "scaled_other_fraction_scp_median", config="primary", embedding="harmony",
                                  cell_type=name), value, 2)
      for name, value in (("TAL", 0.30), ("PST", 0.42), ("PCT", 0.66), ("CNT/PC", 0.76), ("DCT", 0.78))],
    *[(f"label transfer, embedding {name}: matched atlas cell type", "5a",
       lambda f, name=name: _pick(f, "accuracy_strict_median", embedding=name, **PRIMARY), value, 2)
      for name, value in (("pca", 0.46), ("harmony", 0.47), ("scvi", 0.45))],
    ("label transfer: chance", "5a",
     lambda f: _pick(f, "chance_accuracy_strict_median", embedding="harmony", **PRIMARY), 0.063, 3),
    ("label transfer, atlas nuclei only, Harmony", "5a",
     lambda f: _pick(f, "accuracy_strict_median", embedding="harmony", config="nuclei", level="Celltype", k=15),
     0.45, 2),
    *[(f"label transfer of {name} cells, Harmony (median of the five draws)", "5c",
       lambda f, name=name, col=col: _median(f, col, embedding="harmony", scp_label=name, **PRIMARY), value, 2)
      for name, col, value in (("Endothelial", "accuracy_strict", 0.95), ("IC-B", "accuracy_strict", 0.75),
                               ("DCT", "accuracy_strict", 0.72), ("PCT", "accuracy_strict", 0.60),
                               ("TAL", "accuracy_strict", 0.12), ("TAL", "accuracy_incl_mixed", 0.43))],
    *[(f"shared genes, 10% threshold, all atlas cells: {name}", "6a",
       lambda f, name=name: _pick(f, "n_genes", rna_source="all", threshold=0.1, **{"class": name}), value, None)
      for name, value in ((BOTH + "concordant", 1082), (BOTH + "intermediate", 438), (BOTH + "discordant", 550),
                          ("RNA-detected only", 240), ("Protein-detected only", 867), ("Detected in neither", 392))],
    ("chance of rho >= 0.5 with 7 cell types", "6c", lambda f: f["fraction_rho_ge_0.5_concordant"].iloc[0], 0.133,
     3),
    ("median rho of the genes detected in both (10%, all atlas cells)", "8a",
     lambda f: float(np.median([float(v) for v in (lambda d: d[d["class"].astype(str).str.startswith(BOTH)])(
         rows_where(f, rna_source="all", threshold=0.1))["spearman_rho"]])), 0.53, 2),
    ("annotation markers", "7c", lambda f: len(f), 98, None),
    ("annotation markers not quantified as protein", "7c",
     lambda f: int(sum(not bool(v) for v in f["in_protein_matrix"])), 45, None),
    ("annotation markers detected in both: concordant, intermediate, discordant", "7b",
     lambda f: ", ".join(str(len(rows_where(f, **{"class": BOTH + c})))
                         for c in ("concordant", "intermediate", "discordant")), "32, 5, 3", None),
    ("markers detected only as protein", "7c", lambda f: _names(f, **{"class": "Protein-detected only"}),
     "Cldn19, Slc4a1, Slc5a2", None),
    ("markers detected only as RNA", "7c", lambda f: _names(f, **{"class": "RNA-detected only"}),
     "Adgrf5, Cnnm2, Hnf4a, Slc1a1, Slc5a3", None),
    ("transporters and channels: genes", "7h",
     lambda f: _pick(f, "n_genes", rna_source="all", threshold=0.1, gene_group="transporter / channel"), 151, None),
    ("transporters and channels: fraction concordant", "7h",
     lambda f: _concordant(f, "transporter / channel", "fraction_Detected in both; concordant"), 0.44, 2),
    ("other genes: fraction concordant", "7h",
     lambda f: _concordant(f, "other shared genes", "fraction_Detected in both; concordant"), 0.30, 2),
    ("transporters and channels: median rho", "7h",
     lambda f: _concordant(f, "transporter / channel", "median_rho_detected_both"), 0.64, 2),
    ("other genes: median rho", "7h",
     lambda f: _concordant(f, "other shared genes", "median_rho_detected_both"), 0.50, 2),
    ("atlas TAL cells", "1a", lambda f: _pick(f, "n_atlas_expected", scp_label="TAL"), 7440, None),
    ("atlas TAL nuclei", "1a", lambda f: _pick(f, "n_atlas_expected_nuclei", scp_label="TAL"), 5553, None),
]

LAYOUT = Layout(
    number="S-08-d", step="08_s08d_table", workbook="report_08_S08d_tables.xlsx",
    title="Supplementary Table S-08-d. Integration of the SCP-MS cells with the Mouse Kidney Atlas, and detection and "
          "concordance of RNA and protein across kidney cell types",
    intro=("SCP-MS: single-cell proteomics of 712 mouse kidney cells. All values are observed values; the terms used "
           "in the column names are explained below the list of tables.",),
    sheets=SHEETS, terms=TERMS,
    sheet_lines={4: ("Summary over the five atlas draws: sheet 5, tables 5e to 5g.",)},
)


def extra_checks(frames: dict, state: dict, checks: list[dict]) -> None:
    left = sorted({f"{k}: {v}" for k, f in frames.items() for c in f.columns if _is_text(f[c])
                   for v in f[c].dropna().unique() if str(v) in OTHER_ANNOTATIONS or str(v) == COL})
    check(checks, "no row or value of another annotation, and no annotation-name column, in any table",
          left or "none", "none", not left and not any("label_column" in f.columns for f in frames.values()))
    shown = {}
    for spec in SPECS:
        for col in spec.display:
            name = new_name(col, spec, RULES)
            if spec.key in frames and name in frames[spec.key].columns:
                shown.setdefault(spec.key, set()).update(frames[spec.key][name].dropna().astype(str))
    bad = {k: sorted(v - SCP_LABELS) for k, v in shown.items() if v - SCP_LABELS}
    check(checks, "cell-type columns hold the revised annotation with the display names (DCT, CNT/PC; no DCT1 or "
          "PC; Unknown kept)", bad or f"{sum(len(s.display) for s in SPECS)} columns in {len(shown)} tables; "
          f"values: {', '.join(sorted(set().union(*shown.values())))}", "labels of the revised annotation only",
          not bad and len(shown) == sum(bool(s.display) for s in SPECS))
    f = state["8a"]["source"]
    check(checks, "chance table: the 7-cell-type row is the one of the revised annotation (per-gene table: "
          "cell types compared under kidney_t15_label)", sorted(f["n_compartments"].unique().tolist()), "[7]",
          sorted(f["n_compartments"].unique().tolist()) == [7])
    a = rows_where(frames["4b"], metric="ilisi_minus_1_random_mixing").set_index("label")["median"].astype(float)
    b = rows_where(frames["5f"], config="primary", embedding="harmony").set_index("cell_type")[
        "ilisi_minus_1_random_mixing_median"].astype(float)
    diff = float((a - b.reindex(a.index)).abs().max()) if len(a) else float("nan")
    check(checks, "figure values: the random-mixing metric of Supplementary Figure S-08-b is the value named "
          "ilisi_minus_1_random_mixing in the summary table", f"{len(a)} cell types, max abs difference {diff:.1e}",
          "7 cell types, <= 1e-12", len(a) == 7 and diff <= 1e-12)


if __name__ == "__main__":
    build(LAYOUT, SPECS, RULES, LETTER, "the response to the reviewers: Supplementary Table S-08-d, revised annotation only.",
          extra_checks)
