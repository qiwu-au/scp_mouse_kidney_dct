#!/usr/bin/env python
"""report: shared assembly of supplementary table workbooks from stored tables (S-01-a, S-02-d, S-03-b).

(authors, 12 Nov 2026: "Yes, assemble them"). Used by
report_filtering_table.py, report_mito_tables.py and report_preprocessing_tables.py. The layout and the style
are those of Supplementary Table S-07-d (report_pt_tal_tables.py, 11 Nov 2026), whose workbook helpers are
imported unchanged: a Contents sheet, numbered data sheets, a title row and note lines per table, Arial 10, values
only, numbers stored with 16 significant digits and shown with a display format.

No new statistics: every value is copied from a stored table. Per table, a Spec states the source file, the row
filter, the dropped columns, the display names of values and the renamed columns; all of it is written to the
sources table. Checks (written to the checks table), per table:
- rows written to the workbook = rows of the source left after the stated filter;
- every numeric value read back from the saved workbook equals the value of the filtered source (whole numbers
  exactly; other numbers to the 16 significant digits a workbook stores), and every text value equals the source
  text with the stated display names.

Inputs and outputs: see the three scripts. This module writes nothing by itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment

import scp_common as cc
from report_pt_tal_tables import (F, FB, FN, FT, HEAD_BORDER, HEAD_FILL, REL_TOL, Book, _pick, check, native,
                                      number_format, rel_diff)

TASK = "report"
UNKNOWN_NOTE = "Unknown = DCT, subtype not resolved (cells of clusters without marker support)"

# Display names (the project notes; thread files, section 2; S-07-d).
CONDITION = {"Ctrl": "Control", "ZeroK": "Zero-K"}
DATASET = {"kidney": "Total kidney", "Ctrl": "DCT Control", "ZeroK": "DCT Zero-K"}
KIDNEY_LABEL = {"DCT1": "DCT", "PC": "CNT/PC", "all_cells": "all cells"}
ALL_CELLS = {"all_cells": "all cells"}
LABEL_SET = {"kidney_t15_label": "cell types", "cs_ctrl_deg_label": "cluster annotation",
             "cs_zerok_pipeline_label": "cluster annotation",
             "cs_ctrl_cluster_label": "marker-score label (clusters)",
             "cs_zerok_cluster_label": "marker-score label (clusters)", "plate": "sort plate"}
# Shared vocabulary of the supplementary figures and tables (authors, 12 Nov 2026: "The readers will not know the
# revision process, so we only show the final products"): no name in a workbook refers to the earlier manuscript.
MARKER_SCORE = ("Marker-score label: each cell takes the cell type whose marker proteins give the highest score (per "
                "cell); a cluster at Leiden resolution 3 takes the most frequent label of its cells (clusters).")
# Kidney labels inside free text ("PCT: 6; DCT1: 1", "TAL;DCT1;PC;IC-B").
KIDNEY_TEXT = ((r"(?<![A-Za-z0-9])DCT1(?![A-Za-z0-9])", "DCT"), (r"(?<![A-Za-z0-9/-])PC(?![A-Za-z0-9-])", "CNT/PC"))
LABEL_TEXT = ((r"\bprimary label\b", "cluster annotation"), (r"\bsecond label\b", "marker-score label (clusters)"),
              (r"\bper-cell label\b", "marker-score label (per cell)"))

# Internal names that must not reach a workbook. Run names in the cell_id columns ("[4] Ctrl_GFP_...") are exempt.
FORBIDDEN = re.compile(
    r"(?<![A-Za-z0-9])T\d{2}(?![A-Za-z0-9])|(?<![A-Za-z0-9])t15|kidney_(t15|pre|reannot)|reannot|fig ?2a"
    r"|(?<![A-Za-z0-9])cs_|label_role|label_column|primary[ _]label|second[ _]label|per-cell label|__primary__"
    r"|__second__|(?<![A-Za-z0-9])Ctrl(?![A-Za-z0-9])|ZeroK|\.csv|variant_id|is_baseline|task ?1"
    r"|\bbrief\b|(?<![A-Za-z0-9])v3(?![A-Za-z0-9])|(?<![A-Za-z])deg_|this script", re.I)
PROCESS_WORDS = re.compile(r"submitted|submission|re-?annotation", re.I)     # words for the revision process
CELL_ID = re.compile(r"^\[\d+\] \S+\.htrms\.PG\.Quantity$")
KIDNEY_OLD = re.compile(r"(?<![A-Za-z0-9])DCT1(?![A-Za-z0-9])|(?<![A-Za-z0-9/-])PC(?![A-Za-z0-9-])")


@dataclass
class Join:
    source: str                                # project-relative path of the table that holds the column
    on: str                                    # key column in both tables
    column: str                                # column taken from that table
    after: str                                 # the column is placed after this column


@dataclass
class Spec:
    key: str                                   # table id in the workbook, e.g. "4a" (sheet 4, first table)
    source: str                                # project-relative path of the source table
    title: str
    notes: tuple[str, ...]
    rows: tuple[tuple[str, tuple[str, ...]], ...] = ()         # rows kept: every (column, values) holds
    drop_rows: tuple[tuple[str, str], ...] = ()                # rows dropped: (column, regex) matches
    drop_const: tuple[str, ...] = ()           # columns dropped after a check that they hold one value
    drop_dependent: tuple[tuple[str, str], ...] = ()           # (column, by): dropped after a check that `by` fixes it
    drop: tuple[str, ...] = ()                 # columns dropped by name
    drop_re: str = ""                          # columns dropped by pattern
    drop_reason: str = ""
    join: Join | None = None
    values: dict[str, dict[str, str]] = field(default_factory=dict)      # source column -> {value: display name}
    strict: tuple[str, ...] = ()               # columns of `values` in which every value needs a display name
    value_re: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)   # source column -> substitutions
    map_by: tuple[tuple[str, str, str], ...] = ()              # (column, code, text): the codes in `column` are
    #                                                            replaced by the text that the table gives for the code
    rename: dict[str, str] = field(default_factory=dict)
    rename_re: tuple[tuple[str, str], ...] = ()
    kidney: bool = False                       # kidney table: DCT1 and PC must not be left in it
    dct_unknown: bool = False                  # shows the Unknown of the DCT cluster annotation
    contents_note: str = ""                    # notes column of the Contents sheet (for the reader)
    assembly_note: str = ""                    # notes column of the sources table (for the authors)

    @property
    def sheet(self) -> int:
        return int(self.key[:-1])


@dataclass
class WorkbookDef:
    label: str                                 # "S-02-d"
    step: str                                  # "02_s02d_table"
    workbook: str                              # file name of the workbook
    title: str
    subtitle: str
    sheets: dict[int, tuple[str, str, tuple[str, ...]]]        # number -> (sheet name, title, note lines)
    specs: list[Spec]
    terms: list[tuple[str, str]]
    letter: list[tuple[str, str, Callable[[pd.DataFrame], object], object, int | None]] = field(default_factory=list)
    left_out: list[tuple[int, str, str]] = field(default_factory=list)   # (sheet, content, note): tables left out
    extra_checks: Callable[[dict, dict, list], None] | None = None       # (assembled frames, stored frames, checks)
    parameters: dict[str, object] = field(default_factory=dict)


def read_source(path) -> pd.DataFrame:
    """A stored table with its numbers parsed exactly as written (round-trip parser)."""
    return pd.read_csv(path, float_precision="round_trip")


def as_stored(value):
    """A value as the workbook holds it: native, and an infinite number as the text inf or -inf.

    A workbook cannot hold an infinite number (the stored tables have a few infinite odds ratios).
    """
    value = native(value)
    if isinstance(value, float) and np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


class TableBook(Book):
    """The S-07-d workbook writer, with infinite numbers written as text."""

    def table(self, ws, row: int, spec, frame: pd.DataFrame) -> int:
        self.cell(ws, row, 1, f"Table {spec.key}. {spec.title}", FB, width=False)
        row += 1
        for line in spec.notes:
            self.cell(ws, row, 1, line, FN, width=False)
            row += 1
        for j, name in enumerate(frame.columns, 1):
            c = self.cell(ws, row, j, str(name), FB)
            c.fill, c.border = HEAD_FILL, HEAD_BORDER
        self.positions[spec.key] = {"sheet": ws.title, "header_row": row, "n_rows": len(frame),
                                    "n_cols": frame.shape[1]}
        formats = [number_format(str(name), frame[name].replace([np.inf, -np.inf], np.nan)
                                 if pd.api.types.is_float_dtype(frame[name]) else frame[name])
                   for name in frame.columns]
        for i, values in enumerate(frame.itertuples(index=False, name=None), row + 1):
            for j, value in enumerate(values, 1):
                value = as_stored(value)
                self.cell(ws, i, j, value, F, None if isinstance(value, str) else formats[j - 1])
        return row + len(frame) + 3              # two empty rows before the next table


def is_text(series: pd.Series) -> bool:
    return bool(pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))


def row_mask(spec: Spec, src: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=src.index)
    for col, kept in spec.rows:
        mask &= src[col].astype(str).isin(kept)
    for col, pattern in spec.drop_rows:
        mask &= ~src[col].astype(str).str.contains(pattern, regex=True)
    return mask


def row_filter_text(spec: Spec) -> str:
    parts = [f"{col} in {' / '.join(kept)}" for col, kept in spec.rows]
    parts += [f"rows dropped where {col} matches '{pattern}'" for col, pattern in spec.drop_rows]
    return "; ".join(parts) or "all rows"


def display_text(spec: Spec, column: str, value):
    """The text of a source cell as the workbook shows it (display names, then substitutions)."""
    if not isinstance(value, str):
        return value
    value = spec.values.get(column, {}).get(value, value)
    for pattern, new in spec.value_re.get(column, ()):
        value = re.sub(pattern, new, value)
    return value


def new_name(spec: Spec, name: str) -> str:
    if name in spec.rename:
        return spec.rename[name]
    for pattern, new in spec.rename_re:
        name = re.sub(pattern, new, name)
    return name


def assemble_table(spec: Spec, src: pd.DataFrame, joined: pd.DataFrame | None) -> tuple[pd.DataFrame, dict, dict]:
    """Filtered rows and kept columns with reader-facing names; the record says what was done.

    Returns the table, the record for the sources table, and the map workbook column -> source column.
    """
    problems, edits = [], []
    for target, code, text in spec.map_by:       # display names taken from the table itself (code -> text)
        pairs = src[[code, text]].drop_duplicates()
        if pairs[code].duplicated().any():
            problems.append(f"{code} does not fix {text}")
        spec.values[target] = {k: display_text(spec, text, v) for k, v in zip(pairs[code], pairs[text])}
        spec.strict = (*spec.strict, target)
    df = src[row_mask(spec, src)].reset_index(drop=True)
    join_text = ""
    if spec.join is not None:
        j, on, col = spec.join, spec.join.on, spec.join.column
        if joined[on].duplicated().any() or df[on].duplicated().any():
            problems.append(f"join: {on} is not unique")
        else:
            merged = df.merge(joined[[on, col]], on=on, how="left", validate="one_to_one")
            if merged[col].isna().any() or not merged[on].equals(df[on]):
                problems.append(f"join: {int(merged[col].isna().sum())} rows without {col}, or row order changed")
            cols = list(df.columns)
            cols.insert(cols.index(j.after) + 1, col)
            df = merged[cols]
            join_text = f"{col} from {j.source}, matched on {on} ({len(df)} of {len(df)} rows matched)"
    const = []
    for col in spec.drop_const:
        if col not in df.columns or df[col].nunique(dropna=False) != 1:
            problems.append(f"{col} is missing or not constant after the row filter")
            continue
        const.append(f"{col} = {df[col].iloc[0]}")
        df = df.drop(columns=col)
    for col, mapping in spec.values.items():
        if col not in df.columns:
            problems.append(f"{col}: column with display names is missing")
            continue
        present = set(df[col].dropna().astype(str))
        if col in spec.strict and present - set(mapping):
            problems.append(f"{col}: values without a display name: {sorted(present - set(mapping))}")
        used = [f"{k} -> {v}" for k, v in mapping.items() if k in present]
        if used:
            edits.append(f"{col}: " + "; ".join(used))
    for col, subs in spec.value_re.items():
        if col not in df.columns:
            problems.append(f"{col}: column with substitutions is missing")
            continue
        pairs = sorted({(v, display_text(spec, col, v)) for v in df[col].dropna()
                        if isinstance(v, str) and display_text(spec, col, v) != spec.values.get(col, {}).get(v, v)})
        if not pairs:
            problems.append(f"{col}: no substitution applied")
        edits.append(f"{col}: " + " || ".join(f"{a} -> {b}" for a, b in pairs[:40])
                     + (f" || ... ({len(pairs)} distinct texts)" if len(pairs) > 40 else ""))
    for col in set(spec.values) | set(spec.value_re):
        if col in df.columns:
            df[col] = df[col].map(lambda v, c=col: display_text(spec, c, v))
    dependent = []
    for col, by in spec.drop_dependent:
        if col not in df.columns or by not in df.columns or int(df.groupby(by, dropna=False)[col].nunique().max()) != 1:
            problems.append(f"{col} is missing or not fixed by {by}")
            continue
        pairs = df[[by, col]].drop_duplicates()
        dependent.append(f"{col} (" + "; ".join(f"{b}: {c}" for b, c in zip(pairs[by], pairs[col])) + ")")
        df = df.drop(columns=col)
    named = [c for c in spec.drop if c in df.columns]
    if len(named) != len(spec.drop):
        problems.append(f"columns to drop not found: {sorted(set(spec.drop) - set(named))}")
    by_pattern = [c for c in df.columns if spec.drop_re and re.search(spec.drop_re, c) and c not in named]
    if spec.drop_re and not by_pattern:
        problems.append(f"no column matches the drop pattern {spec.drop_re}")
    df = df.drop(columns=named + by_pattern)
    empty = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty)
    renamed = {c: new_name(spec, c) for c in df.columns if new_name(spec, c) != c}
    missing = sorted(set(spec.rename) - set(renamed))
    if missing:
        problems.append(f"columns to rename not found: {missing}")
    colmap = {new_name(spec, c): c for c in df.columns}
    df = df.rename(columns=renamed)
    if df.columns.duplicated().any():
        problems.append(f"duplicate column names after renaming: {list(df.columns[df.columns.duplicated()])}")
    if df.empty:
        problems.append("no rows left")
    floats = df.select_dtypes(include="float")
    record = {"sheet": spec.sheet, "table": spec.key, "title": spec.title, "source": spec.source,
              "rows_source": len(src), "rows_after_filter": len(df), "columns_source": src.shape[1],
              "columns_kept": df.shape[1], "row_filter": row_filter_text(spec), "joined": join_text,
              "infinite_values_written_as_text": int(np.isinf(floats.to_numpy(dtype=float)).sum()) if floats.size else 0,
              "columns_dropped_constant": "; ".join(const), "columns_dropped_fixed_by_another": "; ".join(dependent),
              "columns_dropped": "; ".join(named + by_pattern), "columns_dropped_reason": spec.drop_reason,
              "columns_dropped_empty": "; ".join(empty),
              "columns_renamed": "; ".join(f"{a} -> {b}" for a, b in renamed.items()),
              "values_changed": " | ".join(edits), "notes": spec.assembly_note, "problems": " | ".join(problems)}
    return df, record, colmap


def verify_table(spec: Spec, src: pd.DataFrame, joined: pd.DataFrame | None, stored: pd.DataFrame,
                 colmap: dict[str, str]) -> dict:
    """The saved workbook table against the filtered source, read again from disk."""
    kept = src[row_mask(spec, src)].reset_index(drop=True)
    out = {"rows_written": len(stored), "numeric_values_compared": 0, "numeric_mismatches": 0,
           "max_abs_difference": 0.0, "max_rel_difference": 0.0, "text_values_compared": 0, "text_mismatches": 0}
    if len(kept) != len(stored):
        return out | {"rows_ok": False}
    for final, source in colmap.items():
        if spec.join is not None and source == spec.join.column:
            lookup = joined.set_index(spec.join.on)[source]
            values = [lookup.get(key) for key in kept[spec.join.on]]
            numeric = False
        else:
            values = list(kept[source])
            numeric = bool(pd.api.types.is_numeric_dtype(kept[source]) and not pd.api.types.is_bool_dtype(kept[source]))
        for a, b in zip(values, stored[final]):
            a = as_stored(a)
            if not numeric:
                out["text_values_compared"] += 1
                out["text_mismatches"] += display_text(spec, source, a) != b
                continue
            out["numeric_values_compared"] += 1
            if a is None or b is None or isinstance(a, str) or isinstance(b, (bool, str)):
                out["numeric_mismatches"] += a != b      # empty cells, and infinite numbers held as the text inf
                continue
            diff = abs(float(a) - float(b))
            rel = diff / max(abs(float(a)), abs(float(b))) if diff else 0.0
            out["max_abs_difference"] = max(out["max_abs_difference"], diff)
            out["max_rel_difference"] = max(out["max_rel_difference"], rel)
            out["numeric_mismatches"] += rel > (0.0 if isinstance(a, int) else REL_TOL)
    return out | {"rows_ok": True}


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------
def write_workbook(path, wb: WorkbookDef, frames: dict[str, pd.DataFrame], script: str) -> Book:
    book = TableBook()
    ws = book.wb.create_sheet("Contents")
    book.cell(ws, 1, 1, wb.title, FT, width=False)
    book.cell(ws, 2, 1, wb.subtitle, FN, width=False)
    row = 4
    for j, name in enumerate(("sheet", "table", "content", "rows", "columns", "notes"), 1):
        c = book.cell(ws, row, j, name, FB, width=j not in (3, 6))
        c.fill, c.border = HEAD_FILL, HEAD_BORDER
    listed = [(spec.sheet, wb.sheets[spec.sheet][0], spec.key, spec.title, len(frames[spec.key]),
               frames[spec.key].shape[1], spec.contents_note) for spec in wb.specs]
    listed += [(sheet, wb.sheets[sheet][0], "-", content, None, None, note) for sheet, content, note in wb.left_out]
    for _, *values in sorted(listed, key=lambda item: item[0]):          # stable: tables keep their order
        row += 1
        for j, value in enumerate(values, 1):
            c = book.cell(ws, row, j, value, F, width=j not in (3, 6))
            c.alignment = Alignment(wrap_text=j == 6, vertical="top")
    row += 3
    for j, name in ((1, "term or column"), (3, "meaning")):
        c = book.cell(ws, row, j, name, FB, width=False)
        c.fill, c.border = HEAD_FILL, HEAD_BORDER
    ws.cell(row=row, column=2).fill = HEAD_FILL
    ws.cell(row=row, column=2).border = HEAD_BORDER
    for term, meaning in wb.terms:               # the term overflows into the empty column B; the meaning wraps
        row += 1
        book.cell(ws, row, 1, term, F, width=False).alignment = Alignment(vertical="top")
        book.cell(ws, row, 3, meaning, F, width=False).alignment = Alignment(wrap_text=True, vertical="top")
    book.widths[("Contents", 1)] = max(len(name) for name, _, _ in wb.sheets.values())
    book.widths[("Contents", 3)] = 118
    book.widths[("Contents", 6)] = 46
    for number, (name, title, notes) in wb.sheets.items():
        ws = book.wb.create_sheet(name)
        book.cell(ws, 1, 1, f"Supplementary Table {wb.label}, sheet {number}. {title}", FT, width=False)
        book.cell(ws, 2, 1, "Terms and column names: see the sheet Contents.", FN, width=False)
        row = 3
        for line in notes:
            book.cell(ws, row, 1, line, FN, width=False)
            row += 1
        row += 1
        for spec in wb.specs:
            if spec.sheet == number:
                row = book.table(ws, row, spec, frames[spec.key])
    book.set_widths()
    book.wb.properties.title = f"Supplementary Table {wb.label}"
    book.wb.properties.creator = cc.GENERATED_BY
    book.wb.properties.description = f"{cc.GENERATED_TAG}; {script}"
    book.wb.save(path)
    return book


def read_back(path, positions: dict[str, dict]) -> tuple[dict[str, pd.DataFrame], dict]:
    """The tables as stored in the saved workbook, and counts over every cell with a value."""
    wb = load_workbook(path)
    frames = {}
    for key, pos in positions.items():
        ws = wb[pos["sheet"]]
        header = [ws.cell(row=pos["header_row"], column=j).value for j in range(1, pos["n_cols"] + 1)]
        rows = [[ws.cell(row=pos["header_row"] + i, column=j).value for j in range(1, pos["n_cols"] + 1)]
                for i in range(1, pos["n_rows"] + 1)]
        frames[key] = pd.DataFrame(rows, columns=header, dtype=object)       # cells as stored: None stays None
    stats = {"cells": 0, "not_arial": 0, "formulas": 0, "forbidden": [], "notes": {}, "process": []}
    for ws in wb.worksheets:
        stats["notes"][ws.title] = [str(ws.cell(row=r, column=1).value or "") for r in range(1, 12)]
        if PROCESS_WORDS.search(ws.title):
            stats["process"].append(f"sheet name: {ws.title}")
        for row in ws.iter_rows():
            for c in row:
                if c.value is None:
                    continue
                stats["cells"] += 1
                stats["not_arial"] += c.font.name != "Arial"
                stats["formulas"] += c.data_type == "f"
                if isinstance(c.value, str) and not CELL_ID.match(c.value) and FORBIDDEN.search(c.value):
                    stats["forbidden"].append(f"{ws.title}!{c.coordinate}: {c.value[:60]}")
                if isinstance(c.value, str) and PROCESS_WORDS.search(c.value):
                    stats["process"].append(f"{ws.title}!{c.coordinate}: {c.value[:60]}")
    stats["sheets"] = wb.sheetnames
    return frames, stats


# ---------------------------------------------------------------------------
# One workbook: assemble, write, read back, check, record
# ---------------------------------------------------------------------------
def build(wb: WorkbookDef, args) -> None:
    run = cc.Run(TASK, wb.step, args)
    checks: list[dict] = []
    root = cc.PROJECT_ROOT
    script = cc.rel(cc.SCRIPTS_DIR / f"report_{wb.step}.py")

    sources, joins, frames, colmaps, records, missing = {}, {}, {}, {}, [], []
    for spec in wb.specs:
        paths = [root / spec.source] + ([root / spec.join.source] if spec.join else [])
        absent = [cc.rel(p) for p in paths if not p.exists()]
        if absent:
            missing.append(f"{spec.key}: {absent}")
            continue
        sources[spec.key] = read_source(run.use_input(paths[0], f"table {spec.key}"))
        if spec.join:
            joins[spec.key] = read_source(run.use_input(paths[1], f"table {spec.key}, {spec.join.column}"))
        frames[spec.key], record, colmaps[spec.key] = assemble_table(spec, sources[spec.key], joins.get(spec.key))
        record["source_sha256"] = run.inputs[cc.rel(paths[0])]["sha256"]
        record["joined_sha256"] = run.inputs[cc.rel(paths[1])]["sha256"] if spec.join else ""
        records.append(record)
    check(checks, "source tables: every listed file exists", "; ".join(missing) or f"{len(wb.specs)} of {len(wb.specs)}",
          f"{len(wb.specs)} of {len(wb.specs)}", not missing)
    if missing:                                   # rule: do not guess; the caller lists such tables as left out
        raise SystemExit(f"Source tables missing: {missing}. List them in left_out and remove their Spec.")
    record_frame = pd.DataFrame(records)
    problems = record_frame.loc[record_frame["problems"].ne(""), ["table", "problems"]]
    check(checks, "assembly: row filters, joins, constant and fixed columns, display names, substitutions, renamed "
          "columns, unique column names, no empty table",
          "; ".join(f"{r.table}: {r.problems}" for r in problems.itertuples()) or f"no problem in {len(frames)} tables",
          "no problem", problems.empty)
    infinite = record_frame.loc[record_frame["infinite_values_written_as_text"] > 0]
    check(checks, "information: infinite numbers of the source tables are written as the text inf or -inf (a "
          "workbook cannot hold them)",
          "; ".join(f"{r.table}: {r.infinite_values_written_as_text}" for r in infinite.itertuples()) or "none",
          "information", True)
    names = [wb.sheets[n][0] for n in sorted(wb.sheets)]
    bad_names =[n for i, n in zip(sorted(wb.sheets), names) if len(n) > 31 or not n.startswith(f"{i} ")]
    check(checks, "sheet names: at most 31 characters, starting with the sheet number", bad_names or
          f"{len(names)} names, longest {max(map(len, names))} characters", "all", not bad_names)
    used = sorted({spec.sheet for spec in wb.specs} | {s for s, _, _ in wb.left_out})
    check(checks, "every sheet holds at least one table (or a table left out with a note)", used, sorted(wb.sheets),
          used == sorted(wb.sheets))

    out = cc._check_writable(cc.OUT_ROOTS["tables"] / TASK / wb.workbook, run.overwrite)
    book = write_workbook(out, wb, frames, script)
    run.outputs.append(cc.rel(out))
    print(f"Saved workbook: {cc.rel(out)}")

    stored, stats = read_back(out, book.positions)
    check(checks, "workbook sheets", ", ".join(stats["sheets"]), "Contents, then the numbered sheets",
          stats["sheets"] == ["Contents", *names])

    # Per table: rows after the stated filter, and the stored values against the filtered source (read again).
    verified = {}
    for spec in wb.specs:
        fresh = read_source(root / spec.source)
        fresh_join = read_source(root / spec.join.source) if spec.join else None
        expected_rows = int(row_mask(spec, fresh).sum())
        v = verified[spec.key] = verify_table(spec, fresh, fresh_join, stored[spec.key], colmaps[spec.key])
        check(checks, f"table {spec.key}: rows written = rows of the source after the filter ({row_filter_text(spec)})",
              f"{v['rows_written']} written; {expected_rows} after the filter; {len(fresh)} in the source",
              f"{expected_rows} written", v["rows_written"] == expected_rows and v["rows_ok"])
        ok = v["rows_ok"] and v["numeric_mismatches"] == 0 and v["text_mismatches"] == 0
        check(checks, f"table {spec.key}: values read back from the workbook = values of the filtered source",
              f"{v['numeric_values_compared']} numeric values, max abs difference {v['max_abs_difference']:.3g}, "
              f"max relative difference {v['max_rel_difference']:.1e}, not equal {v['numeric_mismatches']}; "
              f"{v['text_values_compared']} text or logical values, not equal {v['text_mismatches']}",
              f"whole numbers and text equal; other numbers within the 16 significant digits a workbook stores "
              f"(relative difference <= {REL_TOL:.0e})", ok)
    for key in ("rows_written", "numeric_values_compared", "numeric_mismatches", "max_abs_difference",
                "max_rel_difference", "text_values_compared", "text_mismatches"):
        record_frame[key] = [verified[t][key] for t in record_frame["table"]]

    differ, largest = [], 0.0
    for key, frame in frames.items():
        got = stored[key]
        if list(got.columns) != [str(c) for c in frame.columns] or got.shape != frame.shape:
            differ.append(f"{key}: shape or header")
            continue
        diffs = [rel_diff(as_stored(a), b) for row_a, row_b in zip(frame.itertuples(index=False, name=None),
                                                                 got.itertuples(index=False, name=None))
                 for a, b in zip(row_a, row_b)]
        n = sum(d > REL_TOL for d in diffs)
        largest = max([largest, *[d for d in diffs if d <= REL_TOL]])
        if n:
            differ.append(f"{key}: {n} cells")
    n_values = int(sum(f.size for f in frames.values()))
    check(checks, "round trip: every table read back from the saved workbook equals the assembled table (header, "
          "shape, text and whole numbers exactly, other numbers to the 16 significant digits a workbook stores)",
          "; ".join(differ) or f"{len(frames)} tables, {n_values} values equal; largest relative difference "
          f"{largest:.1e}", f"{len(frames)} tables equal; relative difference <= {REL_TOL:.0e}", not differ)
    check(checks, "workbook cells: Arial, no formula", f"{stats['cells']} cells with a value; not Arial "
          f"{stats['not_arial']}; formulas {stats['formulas']}", "not Arial 0; formulas 0",
          stats["not_arial"] == 0 and stats["formulas"] == 0)
    check(checks, "no internal name in any cell of the workbook (task codes, internal column and label names, other "
          "annotations, Ctrl / ZeroK, file names); run names in the cell_id columns are exempt",
          "; ".join(stats["forbidden"][:12]) or "none", "none", not stats["forbidden"])
    check(checks, "final products only: no cell, column name or sheet name of the saved workbook contains "
          "'submitted', 'submission' or 're-annotation'",
          f"{len(stats['process'])} cells" + (": " + "; ".join(stats["process"][:8]) if stats["process"] else ""),
          "0 cells", not stats["process"])
    old = sorted({f"{spec.key}: {v}" for spec in wb.specs if spec.kidney
                  for v in [*map(str, stored[spec.key].columns),
                            *[x for x in stored[spec.key].to_numpy().ravel() if isinstance(x, str)]]
                  if not CELL_ID.match(v) and KIDNEY_OLD.search(v)})
    check(checks, "kidney tables: display names DCT and CNT/PC (no DCT1 or PC left in a cell or a column name)",
          old[:8] or "none", "none", not old)
    need = sorted({spec.sheet for spec in wb.specs if spec.dct_unknown})
    shown = sorted({spec.sheet for spec in wb.specs if not spec.kidney and any(
        isinstance(x, str) and re.search(r"(?<![A-Za-z])Unknown(?![A-Za-z])", x)
        for x in stored[spec.key].to_numpy().ravel())})
    lack = [n for n in sorted(set(need) | set(shown)) if not any(UNKNOWN_NOTE in line
                                                                 for line in stats["notes"][wb.sheets[n][0]])]
    check(checks, f"every sheet with a DCT table that shows Unknown carries the note '{UNKNOWN_NOTE}'",
          f"sheets with such a table: {sorted(set(need) | set(shown)) or 'none'}; without the note: {lack or 'none'}",
          "without the note: none", not lack)

    if wb.extra_checks is not None:
        wb.extra_checks(frames, stored, checks)
    for text, key, getter, expected, decimals in wb.letter:
        try:
            value = getter(stored[key])
            value = value.item() if isinstance(value, np.generic) else value
            ok = (value == expected if decimals is None
                  else abs(float(value) - float(expected)) <= 0.5 * 10 ** -decimals + 1e-9)
        except Exception as error:                # a missing row or column is a failed check, not a crash
            value, ok = f"error: {error}", False
        check(checks, f"number of the thread, section 2, in the saved workbook (table {key}): {text}", value,
              expected if decimals is None else f"{expected} (to {decimals} decimals)", ok)

    for c in checks:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['observed']}")
    run.table(record_frame, f"{wb.step}_sources")
    run.table(pd.DataFrame(checks), f"{wb.step}_checks")
    all_ok = all(c["pass"] for c in checks)
    n_fail = sum(not c["pass"] for c in checks)
    print(f"Checks: {len(checks) - n_fail} passed, {n_fail} failed")
    run.finish({"workbook": cc.rel(out), "tables": len(frames), "rows": int(sum(len(f) for f in frames.values())),
                "values": n_values, "tables_left_out": [f"sheet {s}: {c} ({n})" for s, c, n in wb.left_out],
                "style": "Contents sheet and numbered data sheets; Arial 10; values only; 16 significant digits "
                         "with a display format (as Supplementary Table S-07-d)",
                "checks": len(checks), "checks_failed": n_fail, "checks_pass": all_ok, **wb.parameters},
               status="completed" if all_ok else "completed_with_failed_checks")
    if not all_ok:
        raise SystemExit(f"A check failed; see report_{wb.step}_checks.csv.")


__all__ = ["ALL_CELLS", "CONDITION", "DATASET", "KIDNEY_LABEL", "KIDNEY_TEXT", "LABEL_SET", "LABEL_TEXT",
           "MARKER_SCORE", "Join", "Spec", "UNKNOWN_NOTE", "WorkbookDef", "build", "_pick"]
