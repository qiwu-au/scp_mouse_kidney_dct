#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S10 (annotation scans and evidence), one workbook.

The authors asked on 12 Nov 2026 to assemble the table
("Yes, assemble them"). The response to the reviewers, section 5, listed ten sheets from 17 stored tables. Later on 12 Nov 2026 the authors
ruled: "The readers will not know the revision process, so we only show the final products." The workbook therefore
has seven sheets, built from 13 stored tables (kidney sheets 1-4: kidney_annotation, cluster_de; DCT sheets 5-7: dct_labels, combined control +
zero-K search). No new statistics: every value is copied from the stored tables. The layout follows Supplementary Table S7 (report_pt_tal_tables.py, whose workbook helpers are imported unchanged).

Final products only (authors, 12 Nov 2026):
- the three sheets that showed the labels of the submitted version are not in the workbook (submitted Figure 2A / 2B
  labels against the revised annotation; what drives the submitted re-annotation; endothelial proteins in the
  submitted Figure 2A MD cells); the DCT sheets 8-10 are now sheets 5-7; the four removed tables stay in the sources
  table as rows with the reason, and the letter numbers that were read from them are information rows of the checks
  table;
- shared vocabulary: "marker list without the endothelial row" / "with the endothelial row"; "marker-score label
  (per cell)" and "(clusters)" for cs_<cond>_cell_label and cs_<cond>_cluster_label; "graph of the main analysis";
  "marker-score method"; no cell and no sheet name contains "submitted", "submission" or "re-annotation" (check).

What the assembly changes (all recorded per table in the sources table):
- rows: table 6a loses the row of the zero-K label transferred from the control cells (it is not made on the
  condition's own graph); every other table keeps all rows;
- columns: left out are the columns that compare the scan with the submitted kidney labels, the three columns of
  table 4b that refer to the 24 clusters stored in the submitted object (the final annotation has 23 clusters),
  duplicate, constant or empty columns, and columns that hold only internal names;
- names: internal names are replaced in column names and cell values (task codes, cs_ column names, rule names,
  confidence classes, graph names, Ctrl / ZeroK); kidney labels assigned by the rule are shown with the display names
  of the revised annotation (DCT1 -> DCT, PC -> CNT/PC), while marker-list categories keep their names; the DCT label
  value Unknown stays, with the note of decision D9 (12 Nov 2026) on sheets 5-7.

Checks (checks table): rows written = rows left after the stated filter; every table read back from the saved
workbook equals the assembled table; every numeric value read back equals the filtered source table; fonts, no
formulas, no internal name and no word of the revision process in any cell, sheet names; numbers of the the response to the reviewers
letter found in the workbook.

The functions and classes below Spec are generic and are imported by report_dotplot_tables.py.

Inputs (read-only): the 13 kidney_annotation / cluster_de / dct_labels tables named in SPECS.
Outputs: results/tables/report/report_05_S05b_tables.xlsx;
         results/tables/report/report_05_s05b_table_{sources,checks}.csv;
         results/tables/report/report_step_05_s05b_table_manifest.json

Command (project root):  python scripts/report_annotation_tables.py --overwrite
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
from typing import Callable

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment

import scp_common as cc
import report_item_names as pn      # names of the figures and tables in the paper
import report_pt_tal_tables as s7     # Book, native, number_format, rel_diff, check, fonts: reused unchanged

TASK = "report"
CLUSTER_DE, DOTPLOT, KIDNEY_ANNOTATION, DCT_LABELS = "cluster_de", "dotplot", "kidney_annotation", "dct_labels"
NOTE_D9 = "Unknown = DCT, subtype not resolved (cells of clusters without marker support)"
# User, 12 Nov 2026: "The readers will not know the revision process, so we only show the final products."
# "in the revision" (it also catches "added in the revision") was added to the list the same day.
PROCESS_WORDS = re.compile(r"submitted|submission|re-?annotation|in the revision", re.I)
NOT_IN_WORKBOOK = "not in the workbook (removed at the authors' request, 12 Nov 2026)"


# ---------------------------------------------------------------------------
# Generic part (also used by the S5 script)
# ---------------------------------------------------------------------------
@dataclass
class Spec:
    key: str                                   # table id in the workbook, e.g. "4a" (sheet 4, first table)
    task: str
    name: str                                  # source table, without the stage prefix
    title: str
    notes: tuple[str, ...]
    rows: tuple[str, tuple] | None = None      # (column, kept values); None keeps all rows
    rows_not: tuple[str, tuple] | None = None  # (column, values left out)
    pre: Callable | None = None                # special step on the filtered table: f(frame, problems, edits) -> frame
    drop: dict[str, str] = field(default_factory=dict)          # column -> reason
    drop_re: tuple[str, str] | None = None                      # (pattern, reason)
    constant: tuple[str, ...] = ()             # left out after a check that they hold one value
    empty: tuple[str, ...] = ()                # left out after a check that they are empty
    same_as: dict[str, str] = field(default_factory=dict)       # column -> column it must equal; then left out
    values: dict[str, object] = field(default_factory=dict)     # source column -> {old: new} or function(text)
    rename: dict[str, str] = field(default_factory=dict)
    rename_re: tuple[tuple[str, str], ...] = ()
    was: str = ""                              # table id before a renumbering (sources record only)

    @property
    def sheet(self) -> int:
        return int(self.key[:-1])

    @property
    def source(self) -> str:
        return f"results/tables/{self.task}/{self.task}_{self.name}.csv"


@dataclass
class Config:
    step: str                                  # e.g. "05_s05b_table"
    workbook: str
    table_id: str                              # e.g. "S10"
    title: str
    intro: str
    sheets: dict[int, tuple[str, str]]         # number -> (sheet name, title)
    specs: list[Spec]
    terms: list[tuple[str, str]]
    forbidden: re.Pattern
    script: str
    sheet_notes: dict[int, tuple[str, ...]] = field(default_factory=dict)
    numbers: list[tuple] = field(default_factory=list)          # (text, table, getter, expected, decimals or None)
    extra_checks: Callable | None = None       # f(frames, stored, cells, checks)
    removed: list[dict] = field(default_factory=list)           # tables taken out of the workbook (sources record)
    not_in_workbook: list[tuple[str, object]] = field(default_factory=list)   # letter numbers: (text, letter value)
    wording: list[dict] = field(default_factory=list)           # wording changes on request (sources record)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    cc.add_common_args(parser)
    return parser.parse_args()


def _same(a: pd.Series, b: pd.Series) -> bool:
    return bool((a.astype(object).where(a.notna(), None) == b.astype(object).where(b.notna(), None)).all()
                or a.astype(str).tolist() == b.astype(str).tolist())


def assemble(spec: Spec, src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """The table as written, the filtered source, the column map (source -> workbook) and the record."""
    problems, edits, row_filter = [], [], "all rows"
    df = src.copy()
    if spec.rows is not None:
        col, kept = spec.rows
        df = df[df[col].isin(kept)]
        row_filter = f"{col} in {' / '.join(map(str, kept))}"
    if spec.rows_not is not None:
        col, out = spec.rows_not
        missing = [v for v in out if v not in set(df[col])]
        if missing:
            problems.append(f"rows to leave out not found: {missing}")
        df = df[~df[col].isin(out)]
        row_filter = f"{col} not in {' / '.join(map(str, out))}"
    df = df.reset_index(drop=True)
    filt = df.copy()
    if spec.pre is not None:
        df = spec.pre(df, problems, edits)
    dropped = {}
    for col in spec.constant:
        if col not in df.columns or df[col].nunique(dropna=False) != 1:
            problems.append(f"{col} is missing or not constant")
            continue
        dropped[col] = f"one value ({df[col].iloc[0]})"
    for col in spec.empty:
        if col not in df.columns or df[col].notna().any():
            problems.append(f"{col} is missing or not empty")
            continue
        dropped[col] = "empty"
    for col, ref in spec.same_as.items():
        if col not in df.columns or ref not in df.columns or not _same(df[col], df[ref]):
            problems.append(f"{col} is missing or differs from {ref}")
            continue
        dropped[col] = f"equal to {ref}"
    for col, reason in spec.drop.items():
        if col not in df.columns:
            problems.append(f"column to leave out not found: {col}")
            continue
        dropped[col] = reason
    if spec.drop_re is not None:
        pattern, reason = spec.drop_re
        hits = [c for c in df.columns if re.search(pattern, c)]
        if not hits:
            problems.append(f"no column matches {pattern}")
        dropped.update({c: reason for c in hits})
    df = df.drop(columns=list(dropped))
    for col, how in spec.values.items():
        if col not in df.columns:
            problems.append(f"column with values to change not found: {col}")
            continue
        old = df[col]
        if isinstance(how, dict):
            unknown = sorted(set(old.dropna().astype(str)) - set(map(str, how)))
            if unknown:
                problems.append(f"{col}: values without a display name: {unknown}")
            new = old.map(lambda v: how.get(v, v))
        else:
            try:
                new = old.map(lambda v: how(v) if isinstance(v, str) else v)
            except ValueError as error:
                problems.append(f"{col}: {error}")
                continue
        changed = old.notna() & (new.astype(object) != old.astype(object))
        pairs = sorted(set(zip(old[changed].astype(str), new[changed].astype(str))))
        text = "; ".join(f"{a} -> {b}" for a, b in pairs[:6])
        edits.append(f"{col}: " + (text + (f"; ... ({len(pairs)} distinct values changed)" if len(pairs) > 6 else "")
                                   if pairs else "no value changed"))
        df[col] = new
    missing = [c for c in spec.rename if c not in df.columns]
    if missing:
        problems.append(f"columns to rename not found: {missing}")
    renamed = {}
    for col in df.columns:
        new = spec.rename.get(col, col)
        if new == col:
            for pattern, repl in spec.rename_re:
                new = re.sub(pattern, repl, new)
        if new != col:
            renamed[col] = new
    colmap = {c: renamed.get(c, c) for c in df.columns}
    df = df.rename(columns=renamed)
    if df.columns.duplicated().any():
        problems.append(f"duplicate column names after renaming: {list(df.columns[df.columns.duplicated()])}")
    if df.empty:
        problems.append("no rows left")
    for col in df.columns:                      # whole-number columns that pandas read as float (empty cells)
        s = df[col]
        if pd.api.types.is_float_dtype(s) and s.notna().any():
            v = s.dropna()
            if bool((v == v.round()).all()) and (str(col).startswith("n_") or float(v.abs().max()) >= 2):
                df[col] = s.astype("Int64")
    record = {"sheet": spec.sheet, "table": spec.key, "previous_table": spec.was, "status": "in the workbook",
              "title": spec.title, "source": spec.source,
              "rows_source": len(src), "rows_kept": len(df), "columns_source": src.shape[1],
              "columns_kept": df.shape[1], "row_filter": row_filter,
              "columns_left_out": "; ".join(f"{c} ({why})" for c, why in dropped.items()),
              "columns_renamed": "; ".join(f"{a} -> {b}" for a, b in renamed.items()),
              "values_changed": " | ".join(edits), "problems": " | ".join(problems)}
    return df, filt, colmap, record


def write_workbook(path, cfg: Config, frames: dict[str, pd.DataFrame], left_out: dict[str, str]) -> s7.Book:
    """Contents sheet, then one sheet per part; layout and style of Supplementary Table S7."""
    book = s7.Book()
    ws = book.wb.create_sheet("Contents")
    book.cell(ws, 1, 1, cfg.title, s7.FT, width=False)
    book.cell(ws, 2, 1, cfg.intro, s7.FN, width=False)
    head = ["sheet", "table", "content", "rows", "columns"] + (["notes"] if left_out else [])
    row = 4
    for j, name in enumerate(head, 1):
        c = book.cell(ws, row, j, name, s7.FB, width=j not in (3, 6))
        c.fill, c.border = s7.HEAD_FILL, s7.HEAD_BORDER
    for spec in cfg.specs:
        row += 1
        frame = frames.get(spec.key)
        values = [cfg.sheets[spec.sheet][0], spec.key, spec.title, "-" if frame is None else len(frame),
                  "-" if frame is None else frame.shape[1]]
        if left_out:
            values.append(left_out.get(spec.key, ""))
        for j, value in enumerate(values, 1):
            book.cell(ws, row, j, value, s7.F, width=j not in (3, 6))
    row += 3
    for j, name in ((1, "term or column"), (3, "meaning")):
        c = book.cell(ws, row, j, name, s7.FB, width=False)
        c.fill, c.border = s7.HEAD_FILL, s7.HEAD_BORDER
    ws.cell(row=row, column=2).fill = s7.HEAD_FILL
    ws.cell(row=row, column=2).border = s7.HEAD_BORDER
    for term, meaning in cfg.terms:              # the term overflows into the empty column B; the meaning wraps
        row += 1
        book.cell(ws, row, 1, term, s7.F, width=False).alignment = Alignment(vertical="top")
        book.cell(ws, row, 3, meaning, s7.F, width=False).alignment = Alignment(wrap_text=True, vertical="top")
    book.widths[("Contents", 1)] = max(len(name) for name, _ in cfg.sheets.values())
    book.widths[("Contents", 3)] = 118
    for number, (name, title) in cfg.sheets.items():
        ws = book.wb.create_sheet(pn.paper_names(name))       # names of the paper; cells: s7.Book.cell
        book.cell(ws, 1, 1, f"Supplementary Table {cfg.table_id}, sheet {number}. {title}", s7.FT, width=False)
        book.cell(ws, 2, 1, "Terms and column names: see the sheet Contents.", s7.FN, width=False)
        row = 3
        for line in cfg.sheet_notes.get(number, ()):
            book.cell(ws, row, 1, line, s7.FN, width=False)
            row += 1
        row += 1
        for spec in cfg.specs:
            if spec.sheet != number:
                continue
            if spec.key in frames:
                row = book.table(ws, row, spec, frames[spec.key])
            else:
                book.cell(ws, row, 1, f"Table {spec.key}. {spec.title}", s7.FB, width=False)
                book.cell(ws, row + 1, 1, f"Not included: {left_out[spec.key]}", s7.FN, width=False)
                row += 4
    book.set_widths()
    s7.set_properties(book.wb, f"Supplementary Table {cfg.table_id}")
    book.wb.save(path)
    return book


def read_back(path, positions: dict[str, dict], forbidden: re.Pattern) -> tuple[dict, dict]:
    """The tables as stored in the saved workbook, and counts over every cell with a value."""
    wb = load_workbook(path)
    frames = {}
    for key, pos in positions.items():
        ws = wb[pos["sheet"]]
        cols = range(1, pos["n_cols"] + 1)
        header = [ws.cell(row=pos["header_row"], column=j).value for j in cols]
        rows = [[ws.cell(row=pos["header_row"] + i, column=j).value for j in cols]
                for i in range(1, pos["n_rows"] + 1)]
        frames[key] = pd.DataFrame(rows, columns=header, dtype=object)       # cells as stored: None stays None
        found = 0                                                            # rows below the header up to an empty row
        while any(ws.cell(row=pos["header_row"] + found + 1, column=j).value is not None
                  for j in range(1, pos["n_cols"] + 2)):
            found += 1
        pos["rows_found"] = found
    stats = {"cells": 0, "not_arial": 0, "formulas": 0, "forbidden": [], "sheets": wb.sheetnames, "text": {}}
    for ws in wb.worksheets:
        texts = []
        for row in ws.iter_rows():
            for c in row:
                if c.value is None:
                    continue
                stats["cells"] += 1
                stats["not_arial"] += c.font.name != "Arial"
                stats["formulas"] += c.data_type == "f"
                if isinstance(c.value, str):
                    texts.append(c.value)
                    if forbidden.search(c.value):
                        stats["forbidden"].append(f"{ws.title}!{c.coordinate}: {c.value[:60]}")
        stats["text"][ws.title] = texts
    every = {wb.properties.title or "", *wb.sheetnames, *(t for texts in stats["text"].values() for t in texts)}
    stats["labels_left"], stats["texts_read"] = pn.labels_left(every), stats["cells"]
    return frames, stats


def source_difference(filt: pd.DataFrame, stored: pd.DataFrame, colmap: dict) -> tuple[int, int, float, float]:
    """Numeric source columns against the workbook: values compared, cells that differ, largest abs and rel diff."""
    n, bad, worst_abs, worst_rel = 0, 0, 0.0, 0.0
    for src_col, out_col in colmap.items():
        s = filt[src_col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        for a, b in zip(s.tolist(), stored[pn.paper_names(out_col)].tolist()):
            a = s7.native(a)
            n += 1
            d = s7.rel_diff(a, b)
            if d > s7.REL_TOL:
                bad += 1
            elif isinstance(a, (int, float)) and not isinstance(a, bool) and b is not None:
                worst_abs, worst_rel = max(worst_abs, abs(float(a) - float(b))), max(worst_rel, d)
    return n, bad, worst_abs, worst_rel


def pick(frame: pd.DataFrame, col: str, **sel):
    """The one cell of a re-read table in column col where every selector matches (numbers by value)."""
    mask = pd.Series(True, index=frame.index)
    for key, value in sel.items():
        key = pn.paper_names(key)                # the workbook holds the names of the paper
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mask &= pd.to_numeric(frame[key], errors="coerce").eq(value)
        else:
            mask &= frame[key].astype(str).eq(pn.paper_names(str(value)))
    hit = frame.loc[mask, pn.paper_names(col)]
    if len(hit) != 1:
        raise LookupError(f"{len(hit)} rows for {sel}")
    return hit.iloc[0]


def build(cfg: Config, args: argparse.Namespace) -> None:
    run = cc.Run(TASK, cfg.step, args)
    checks: list[dict] = []
    check = s7.check

    frames, filtered, colmaps, records, left_out = {}, {}, {}, [], {}
    for spec in cfg.specs:
        path = cc.out_path(spec.task, "tables", spec.name, ".csv")
        if not path.exists():
            left_out[spec.key] = "the source table is missing"
            records.append({"sheet": spec.sheet, "table": spec.key, "previous_table": spec.was,
                            "status": "left out: source table missing", "title": spec.title, "source": spec.source,
                            "problems": "source table missing: table left out"})
            continue
        run.use_input(path, f"{spec.task} {spec.name}")
        frames[spec.key], filtered[spec.key], colmaps[spec.key], record = assemble(spec, pd.read_csv(path))
        record["source_sha256"] = run.inputs[cc.rel(path)]["sha256"]
        records.append(record)
    sources = pd.DataFrame(records).fillna("")

    n_sheets = sources.loc[sources["table"].isin(frames), "sheet"].nunique()
    check(checks, "source tables read; none missing, each used once",
          f"{len(frames)} of {len(cfg.specs)} tables; sheets {n_sheets}; left out: {sorted(left_out) or 'none'}",
          f"{len(cfg.specs)} tables; sheets {len(cfg.sheets)}; left out: none",
          len(frames) == len(cfg.specs) and n_sheets == len(cfg.sheets) and sources["source"].is_unique)
    problems = sources.loc[sources["problems"].ne(""), ["table", "problems"]]
    check(checks, "assembly: row filters, columns left out (found, constant, empty or equal as stated), display "
          "names for every value, unique column names, no empty table",
          "; ".join(f"{r.table}: {r.problems}" for r in problems.itertuples()) or f"no problem in {len(frames)} tables",
          "no problem", problems.empty)

    out = cc._check_writable(cc.OUT_ROOTS["tables"] / TASK / cfg.workbook, run.overwrite)
    book = write_workbook(out, cfg, frames, left_out)
    run.outputs.append(cc.rel(out))
    print(f"Saved workbook: {cc.rel(out)}")

    stored, stats = read_back(out, book.positions, cfg.forbidden)
    names = [pn.paper_names(name) for name, _ in cfg.sheets.values()]       # as written: names of the paper
    check(checks, "workbook sheets: Contents, then the numbered sheets; names start with the sheet number and have "
          "at most 31 characters", ", ".join(stats["sheets"]), "Contents, " + ", ".join(names),
          stats["sheets"] == ["Contents", *names]
          and all(len(n) <= 31 and n.startswith(f"{k} ") for k, n in zip(cfg.sheets, names)))
    s7.check_paper_names(checks, stats)
    rows_bad = [f"{k}: filtered source {len(filtered[k])}, written {len(frames[k])}, in the sheet "
                f"{book.positions[k]['rows_found']}" for k in frames
                if not len(filtered[k]) == len(frames[k]) == book.positions[k]["rows_found"]]
    check(checks, "rows: for every table the rows in the saved sheet equal the rows of the source table left after "
          "the stated filter", "; ".join(rows_bad) or f"{len(frames)} tables, "
          f"{int(sum(len(f) for f in frames.values()))} rows", "all tables equal", not rows_bad)
    differ, largest = [], 0.0
    for key, frame in frames.items():
        got = stored[key]
        if list(got.columns) != [pn.paper_names(str(c)) for c in frame.columns] or got.shape != frame.shape:
            differ.append(f"{key}: shape or header")
            continue
        diffs = [s7.rel_diff(pn.named(s7.native(a)), b) for row_a, row_b in zip(frame.itertuples(index=False, name=None),
                                                                       got.itertuples(index=False, name=None))
                 for a, b in zip(row_a, row_b)]
        bad = sum(d > s7.REL_TOL for d in diffs)
        largest = max([largest, *[d for d in diffs if d <= s7.REL_TOL]])
        if bad:
            differ.append(f"{key}: {bad} cells")
    check(checks, "round trip: every table read back from the saved workbook equals the assembled table (header, "
          "shape, text and whole numbers exactly, other numbers to the 16 significant digits a workbook stores)",
          "; ".join(differ) or f"{len(frames)} tables, {int(sum(f.size for f in frames.values()))} values equal; "
          f"largest relative difference {largest:.1e}", f"all tables equal; relative difference <= {s7.REL_TOL:.0e}",
          not differ)
    n_all, bad_all, abs_all, rel_all, bad_tables, per_table = 0, 0, 0.0, 0.0, [], {}
    for key in frames:
        if key in [d.split(":")[0] for d in differ]:
            continue
        n, bad, worst_abs, worst_rel = per_table[key] = source_difference(filtered[key], stored[key], colmaps[key])
        n_all, bad_all, abs_all, rel_all = n_all + n, bad_all + bad, max(abs_all, worst_abs), max(rel_all, worst_rel)
        if bad:
            bad_tables.append(f"{key}: {bad}")
    for j, name in enumerate(("numeric_values_compared", "numeric_values_different", "max_abs_diff_vs_source",
                              "max_rel_diff_vs_source")):
        sources[name] = [per_table[k][j] if k in per_table else "" for k in sources["table"]]
    check(checks, "source values: every numeric value read back from the saved workbook equals the value of the "
          "filtered source table (numeric source columns of every table)",
          f"{n_all} values compared; different {bad_all}{' (' + '; '.join(bad_tables) + ')' if bad_tables else ''}; "
          f"max abs difference {abs_all:.1e}; max relative difference {rel_all:.1e}",
          f"different 0; relative difference <= {s7.REL_TOL:.0e} (float text rounding)",
          bad_all == 0 and not differ and n_all > 0)
    check(checks, "workbook cells: Arial, no formula", f"{stats['cells']} cells with a value; not Arial "
          f"{stats['not_arial']}; formulas {stats['formulas']}", "not Arial 0; formulas 0",
          stats["not_arial"] == 0 and stats["formulas"] == 0)
    check(checks, "no internal name in any cell of the workbook (task codes, internal column names, rule and "
          "confidence codes, file names)", "; ".join(stats["forbidden"][:12]) or "none", "none",
          not stats["forbidden"])
    process = [f"{title}: {text[:60]}" for title, texts in stats["text"].items() for text in [title, *texts]
               if PROCESS_WORDS.search(text)]
    check(checks, "final products only (authors, 12 Nov 2026): no cell and no sheet name of the saved workbook contains "
          "'submitted', 'submission', 're-annotation', 'added in the revision' or 'in the revision'",
          "; ".join(process[:12]) or
          f"none in {stats['cells']} cells and {len(stats['sheets'])} sheet names", "none", not process)
    if cfg.extra_checks is not None:
        cfg.extra_checks(frames, stored, stats["text"], checks)

    wrong, n_ok = [], 0
    for text, key, getter, expected, decimals in cfg.numbers:
        try:
            value = getter(stored[key])
            value = value.item() if isinstance(value, np.generic) else value
            ok = (value == expected if decimals is None
                  else abs(float(value) - float(expected)) <= 0.5 * 10 ** -decimals + 1e-9)
        except Exception as error:                # a missing row or column is a failed check, not a crash
            value, ok = f"error: {error}", False
        n_ok += bool(ok)
        if not ok:
            wrong.append(f"{text} (table {key}): {value} instead of {expected}")
    if cfg.numbers:
        check(checks, "numbers of the thread's letter (section 2) found in the saved workbook at the written "
              "precision", f"{n_ok} of {len(cfg.numbers)}" + (f"; not matched: {wrong}" if wrong else ""),
              f"{len(cfg.numbers)} of {len(cfg.numbers)}", not wrong)

    for c in checks:
        print(f"  [{'ok' if c['pass'] else 'FAIL'}] {c['check']}: {c['observed']}")
    info = [{"kind": "information", "check": f"number of the thread's letter: {text}", "observed": NOT_IN_WORKBOOK,
             "expected": str(value), "pass": ""} for text, value in cfg.not_in_workbook]
    for row in info:
        print(f"  [information] {row['check']}: {row['observed']}")
    if cfg.removed or cfg.wording:
        sources = pd.concat([sources, pd.DataFrame([*cfg.removed, *cfg.wording])], ignore_index=True).fillna("")
    run.table(sources, f"{cfg.step}_sources")
    run.table(pd.DataFrame([{"kind": "check", **c} for c in checks] + info), f"{cfg.step}_checks")
    all_ok = all(c["pass"] for c in checks)
    run.finish({"workbook": cc.rel(out), "table": cfg.table_id, "tables": len(frames),
                "tables_removed": [r["source"] for r in cfg.removed],
                "wording_changes_on_request": [r["title"] for r in cfg.wording],
                "rows": int(sum(len(f) for f in frames.values())),
                "values": int(sum(f.size for f in frames.values())), "tables_left_out": left_out,
                "style": "Contents sheet and numbered data sheets; Arial 10; values only; 16 significant digits with "
                         "a display format", "checks_pass": all_ok},
               status="completed" if all_ok else "completed_with_failed_checks")
    print(f"Sheets of {cfg.workbook}:")
    for number, (name, _) in cfg.sheets.items():
        keys = [s.key for s in cfg.specs if s.sheet == number and s.key in frames]
        print(f"  {name}: {len(keys)} tables ({', '.join(keys)}), {sum(len(frames[k]) for k in keys)} rows")
    if not all_ok:
        raise SystemExit(f"A check failed; see report_{cfg.step}_checks.csv.")


# ---------------------------------------------------------------------------
# Supplementary Table S10
# ---------------------------------------------------------------------------
STEP = "05_s05b_table"
TABLE_TITLE = ("Supplementary Table S-05-b. Cluster annotation from differentially expressed proteins: resolution "
               "scans and evidence for the kidney and the DCT data")
INTRO = ("Kidney: 712 cells (sheets 1 to 4). DCT: control and zero-potassium (Zero-K) cells, searched together and "
         "annotated per condition (sheets 5 to 7). All values are observed values; the terms used in the column "
         "names are explained below the list of tables.")

KIDNEY_DISPLAY = {"DCT1": "DCT", "PC": "CNT/PC"}        # display names of the revised annotation (authors, 30 Oct 2026)
CONF = {"high": "high", "medium": "medium", "specific_any_rank": "type-specific marker, any rank",
        "low_shared_only": "shared markers only", "unknown_no_marker_in_top_de": "no marker support",
        "below_min_size": "cluster below 5 cells"}
CONF_DCT = {**CONF, "unknown_no_marker_in_top_de": "no marker support (subtype not resolved)"}    # The response to the reviewers, section 4
CONF_TOP25 = {**CONF, "unknown_no_marker_in_top_de": "no marker among the top 25"}     # before the added step
# "graph of the main analysis": authors, 12 Nov 2026 (was "submitted graph"; same name as in Supplementary Figure S13)
GRAPH = {"stored": "graph of the main analysis", "regressed": "depth-regressed graph",
         "t04_total_unscaled_hvg500": "unscaled, 500 most variable proteins"}
RULE = {"top25": "top-25 rule only", "rankfree": "with the added step"}
# Shared vocabulary of the final products (authors, 12 Nov 2026): no reference to the submission.
MARKER_SET = {"v3": "marker list without the endothelial row", "v3_endo": "with the endothelial row"}
CONDITION = {"Ctrl": "Control", "ZeroK": "Zero-K"}
PER_CELL, CLUSTERS = "marker-score label (per cell)", "marker-score label (clusters)"
DCT_LABEL_NAMES = ((r"cs_(ctrl|zerok)_cell_label", PER_CELL), (r"cs_(ctrl|zerok)_cluster_label", CLUSTERS),
                   (r"cs_ctrl_deg_label|cs_zerok_pipeline_label", "cluster annotation"))
MARKER_SCORE_LABEL = ("Marker-score label: each cell takes the cell type whose marker proteins give the highest score "
                      "(per cell); a cluster at Leiden resolution 3 takes the most frequent label of its cells "
                      "(clusters).")
NOISE = {"A: noisy .X scored": "A: marker scores recomputed",
         "B: noisy .X, whole pipeline": "B: graph, clusters, scores and labels recomputed",
         "B: noisy graph, rank-free DE rule at Leiden 1.0": "B: graph, clusters and labels recomputed"}
OTHER_ANNOTATION = "comparison with the submitted kidney labels (the workbook shows the final annotation only)"
OTHER_CLUSTERS = ("refers to the 24 clusters stored in the submitted object, not to the 23 clusters of the final "
                  "annotation; left out because the workbook shows the final products only (authors, 12 Nov 2026)")
INTERNAL_GRAPH = "graph is shown with its display name"


def kidney_labels(text: str) -> str:
    """DCT1 -> DCT and PC -> CNT/PC in a list of labels assigned by the rule (other names untouched)."""
    text = re.sub(r"(?<![A-Za-z0-9/-])DCT1(?![A-Za-z0-9])", "DCT", text)
    return re.sub(r"(?<![A-Za-z0-9/-])PC(?![A-Za-z0-9])", "CNT/PC", text)


def types_text(conf: dict, display: bool) -> Callable[[str], str]:
    """'PCT(medium);DCT1(low_shared_only)' -> 'PCT (medium); DCT (shared markers only)'."""
    def convert(text: str) -> str:
        items = []
        for item in text.split(";"):
            m = re.fullmatch(r"([A-Za-z0-9-]+)\(([a-z0-9_]+)\)", item)
            if m is None or m[2] not in conf:
                raise ValueError(f"unexpected cell type entry {item!r}")
            items.append(f"{KIDNEY_DISPLAY.get(m[1], m[1]) if display else m[1]} ({conf[m[2]]})")
        return "; ".join(items)
    return convert


def clusters_text(text: str) -> str:
    """'2(58; medium);6(61; low_shared_only)' -> '2 (58; medium); 6 (61; shared markers only)'; '2(134);4(40)' too."""
    items = re.findall(r"(\d+)\((\d+)(?:; ([a-z0-9_]+))?\)", text)
    if ";".join(f"{c}({n}; {k})" if k else f"{c}({n})" for c, n, k in items) != text or any(
            k and k not in CONF for _, _, k in items):
        raise ValueError(f"unexpected cluster list {text!r}")
    return "; ".join(f"{c} ({n}; {CONF[k]})" if k else f"{c} ({n})" for c, n, k in items)


def dct_label_names(text: str) -> str:
    for pattern, name in DCT_LABEL_NAMES:
        text = re.sub(pattern, name, text)
    return text


SHEETS = {
    1: ("1 Kidney resolution scan", "Kidney: Leiden resolution scan on three graphs, top-25 rule only and with the "
                                    "added step"),
    2: ("2 Kidney types by resolution", "Kidney: cell types assigned at each resolution, and in how many Leiden seeds"),
    3: ("3 Endothelial row and seeds", "Kidney: resolution 2 without and with the endothelial row of the marker list, "
                                       "and seed stability"),
    4: ("4 Marker coverage unresolved", "Kidney: marker coverage, and cell types that were not resolved"),
    5: ("5 DCT resolution scan", "DCT: Leiden resolution scan per condition, top-25 rule only and with the added step"),
    6: ("6 DCT labels and clusters", "DCT: labels per condition, and evidence for each cluster at resolution 1"),
    7: ("7 DCT marker coverage", "DCT: quantification and detection of the listed markers per condition"),
}
UNKNOWN_NOTE = (f"This is the Unknown of the cluster annotation. The Unknown of the {CLUSTERS} is a different label, "
                "set by the Unknown rule of the marker-score method.")
SHEET_NOTES = {5: (NOTE_D9,), 6: (NOTE_D9, UNKNOWN_NOTE), 7: (NOTE_D9,)}

VS_MARKER_SCORE = {
    "ari_vs_cell_label": "ari_vs_marker_score_label_per_cell",
    "fraction_equal_cell_label": "fraction_equal_marker_score_label_per_cell",
    "fraction_equal_cell_label_core_labelled_cells": "fraction_equal_marker_score_label_per_cell_labelled_cells",
    "ari_clusters_vs_cell_label": "ari_clusters_vs_marker_score_label_per_cell",
    "ari_vs_cluster_label": "ari_vs_marker_score_label_clusters",
    "fraction_equal_cluster_label": "fraction_equal_marker_score_label_clusters",
    "fraction_equal_cluster_label_core_labelled_cells": "fraction_equal_marker_score_label_clusters_labelled_cells",
    "ari_clusters_vs_cluster_label": "ari_clusters_vs_marker_score_label_clusters",
}

SPECS = [
    Spec("1a", KIDNEY_ANNOTATION, "scan_summary", "Kidney scan with the top-25 rule only: clusters, cell types, Unknown fraction, "
         "proteins per cell and seed stability",
         ("One row per graph and resolution (seed 42); marker list without the endothelial row. types_resolved: cell "
          "types assigned to at least one cluster, with the best confidence among their clusters.",
          "type_sets_seen_all_seeds: the sets of cell types found in the 11 seeds (sets separated by |). Columns: see "
          "Contents (Scan columns; protein IDs)."),
         drop={"display": INTERNAL_GRAPH}, drop_re=(r"_vs_(reannot|pre)$", OTHER_ANNOTATION),
         values={"graph": GRAPH, "types_resolved": types_text(CONF, True), "type_sets_seen_all_seeds": kidney_labels}),
    Spec("1b", KIDNEY_ANNOTATION, "rankfree_summary", "Kidney scan: top-25 rule only and with the added step, side by side",
         ("One row per graph and resolution (seed 42); marker list without the endothelial row. top25_ columns: "
          "top-25 rule only; added_step_ columns: with the added step. n_clusters_added_step, n_cells_added_step: "
          "clusters and cells labelled by the added step.",
          "_n_types: cell types assigned; _n_types_specific: of these, with a type-specific marker (confidence high, "
          "medium or type-specific marker, any rank). _fraction_unknown: fraction of cells Unknown."),
         drop={"display": INTERNAL_GRAPH}, drop_re=(r"_ari_vs_(reannot|pre)$", OTHER_ANNOTATION),
         values={"graph": GRAPH, "top25_types": types_text(CONF, True), "rankfree_types": types_text(CONF, True)},
         rename={"n_clusters_rescued": "n_clusters_added_step", "n_cells_rescued": "n_cells_added_step"},
         rename_re=((r"^rankfree_", "added_step_"),)),
    Spec("2a", KIDNEY_ANNOTATION, "rankfree_types_by_resolution", "Kidney: clusters and cells per cell type at each graph, "
         "resolution and rule",
         ("One row per rule, graph, resolution and cell type of the marker list without the endothelial row (16 cell "
          "types). n_clusters_seed42, n_cells_seed42: clusters and cells with this label at seed 42; "
          "clusters_seed42: cluster (cells; confidence).",
          "n_seeds_resolved: seeds, of n_seeds, in which the cell type is assigned to at least one cluster; "
          "_ge_medium: with confidence high or medium."),
         values={"rule": RULE, "graph": GRAPH, "cell_type": {**{t: t for t in (
             "PCT", "PST", "DTL", "ATL", "TAL", "MD", "DCT2", "CNT", "IC-A", "IC-B", "DVR", "AVR", "JG", "Podo")},
             **KIDNEY_DISPLAY}, "best_confidence_seed42": CONF, "clusters_seed42": clusters_text}),
    Spec("3a", KIDNEY_ANNOTATION, "v3endo_summary", "Kidney at resolution 2: marker list without and with the endothelial row, "
         "both rules",
         ("One row per graph, marker list and rule. types: cell types at seed 42 with their best confidence. "
          "n_types_all_seeds_min, _max: over the 11 seeds.",
          "n_cells_endothelial: cells labelled Endothelial at seed 42; seeds_with_endothelial_label: seeds, of 11, "
          "with an Endothelial cluster. same_label_as_seed42_median: median over seeds 0 to 9 of the fraction of "
          "cells with the seed-42 label. Adopted: graph of the main analysis, with the endothelial row, with the added "
          "step."),
         drop={"display": INTERNAL_GRAPH},
         values={"graph": GRAPH, "marker_set": MARKER_SET, "rule": RULE, "types": types_text(CONF, True)}),
    Spec("3b", KIDNEY_ANNOTATION, "v3endo_seed_stability", "Kidney at resolution 2: every Leiden seed",
         ("One row per graph and seed. Column groups: list_without_endothelial_row__ (marker list without the "
          "endothelial row) and endothelial_row__ (with the endothelial row), each with top25__ (top-25 rule only) "
          "and added_step__ (with the added step).",
          "n_types, types: cell types assigned; fraction_unknown: fraction of cells Unknown; same_label_as_seed42: "
          "fraction of cells with the same label as at seed 42; n_cells_endothelial: cells labelled Endothelial."),
         values={"graph": GRAPH, **{f"{m}__{r}__types": types_text(CONF, True)
                                    for m in ("v3", "v3_endo") for r in ("top25", "rankfree")}},
         rename_re=((r"^v3__", "list_without_endothelial_row__"), (r"^v3_endo__", "endothelial_row__"),
                    (r"__rankfree__", "__added_step__"))),
    Spec("4a", CLUSTER_DE, "kidney_marker_coverage", "Kidney marker list: quantification and detection of every listed "
         "marker",
         ("One row per cell type and marker of the marker list without the endothelial row (16 cell types, named as "
          "in the list: DCT1 and PC are the list categories behind the labels DCT and CNT/PC).",
          "quantified: the protein is in the data. types_listing: cell types that list the marker; shared: listed "
          "for two or more. detection_rate_all_cells: fraction of the 712 cells with the protein detected. type_n_ "
          "columns: markers of the cell type listed, quantified, and quantified and type-specific."),
         rename={"cell_type_workbook": "marker_list_category", "in_matrix": "quantified",
                 "specific_in_matrix": "type_specific_quantified", "type_n_present": "type_n_quantified",
                 "type_n_specific_present": "type_n_type_specific_quantified"}),
    Spec("4b", KIDNEY_ANNOTATION, "rare_candidates_summary", "Kidney: cells with a type-specific marker detected, per cell type",
         ("Cell types of the marker list with at least one type-specific quantified marker (list names). "
          "n_cells_ge1_marker_detected: cells with at least one of these markers detected; ge2: at least two (empty "
          "for cell types with one such marker). These are counts of cells, not labels.",),
         drop={**{c: OTHER_ANNOTATION for c in (
             "n_cells_labelled_reannot", "n_labelled_reannot_that_are_candidates", "candidates_by_reannot_label",
             "n_cells_labelled_pre", "n_labelled_pre_that_are_candidates", "candidates_by_pre_label")},
             **{c: OTHER_CLUSTERS for c in ("stored_leiden2_clusters_spanned", "stored_leiden2_top_cluster",
                                            "stored_leiden2_top_cluster_n_candidates")}},
         rename={"n_specific_markers": "n_type_specific_markers", "specific_markers": "type_specific_markers",
                 "n_candidates_ge1": "n_cells_ge1_marker_detected",
                 "n_candidates_ge2": "n_cells_ge2_markers_detected"}),
    Spec("4c", KIDNEY_ANNOTATION, "rare_coverage_limits", "Kidney: cell types of the marker list without a type-specific "
         "quantified marker",
         ("markers_listed, markers_quantified: markers of the cell type in the list and in the data. "
          "quantified_markers_shared_with: for each quantified marker, the other cell types that list it (list "
          "names).",)),
    Spec("5a", DCT_LABELS, "scan_summary", "DCT scan per condition: clusters, labels, confidence, proteins per cell and "
         "seed stability",
         ("One row per condition, resolution and rule (seed 42). adopted_setting: resolution 1 with the added step, "
          "fixed before these data were annotated. label_counts, n_DCT1 to n_Unknown: cells per label; n_cells_ "
          "columns: cells per confidence of their cluster.",
          "n_cells_top25_dct1_from_umod_slc12a3_only: cells of clusters that the top-25 rule labels DCT1 from Umod "
          "or Slc12a3 only. _labelled_cells: among the cells labelled DCT1, DCT2 or ProLIF. Columns: see Contents "
          "(Scan columns; DCT labels; plate1)."),
         was="8a", drop={"rule_display": "rule is shown with its display name"},
         values={"condition": CONDITION, "rule": RULE, "types_resolved": types_text(CONF_DCT, False)},
         rename={"frozen_primary": "adopted_setting", "n_below_min_size": "n_in_clusters_below_5_cells",
                 "n_cells_conf_high": "n_cells_confidence_high", "n_cells_conf_medium": "n_cells_confidence_medium",
                 "n_cells_conf_specific_any_rank": "n_cells_type_specific_marker_any_rank",
                 "n_cells_conf_low_shared_only": "n_cells_shared_markers_only",
                 "n_cells_conf_unknown_no_marker_in_top_de": "n_cells_no_marker_support",
                 "n_cells_conf_below_min_size": "n_cells_cluster_below_5_cells",
                 "n_cells_top25_dct1_umod_slc12a3_only": "n_cells_top25_dct1_from_umod_slc12a3_only",
                 "cluster_P1_fraction_min": "cluster_plate1_fraction_min",
                 "cluster_P1_fraction_max": "cluster_plate1_fraction_max", **VS_MARKER_SCORE}),
    Spec("5b", DCT_LABELS, "scan_types_by_resolution", "DCT: clusters and cells per cell type at each resolution and rule",
         ("One row per condition, resolution, rule and cell type of the DCT marker list. n_clusters_seed42, "
          "n_cells_seed42: clusters and cells with this label at seed 42; clusters_seed42: cluster (cells).",
          "n_seeds_resolved: seeds, of n_seeds, in which the cell type is assigned to at least one cluster; "
          "_ge_medium: with confidence high or medium."),
         was="8b", values={"condition": CONDITION, "rule": RULE, "best_confidence_seed42": CONF_DCT,
                           "clusters_seed42": clusters_text}),
    Spec("6a", DCT_LABELS, "label_column_summary", "DCT labels per condition: cells, proteins per cell, sort plate and "
         "stability",
         ("One row per condition and label. label_counts: cells per label. eta2_protein_ids_on_label: share of the "
          "variance of the proteins per cell explained by the label; median_protein_ids_by_label; "
          "fraction_plate1_by_label.",
          "seed_, noise_, loo_ and cells_changed_without_ columns: see Contents (DCT stability columns). "
          "fraction_equal_other_labels: fraction of cells with the same label under each of the other two labels."),
         was="9a", rows_not=("label_column", ("cs_zerok_transfer_label",)),
         drop={"role": "internal role names; the label name says which label it is",
               "source": "internal description of the method; the Contents sheet describes the labels"},
         values={"condition": CONDITION, "label_column": dct_label_names, "noise_test": NOISE,
                 "agreement_with_other_columns": dct_label_names},
         rename={"label_column": "label", "fraction_P1_by_label": "fraction_plate1_by_label",
                 "noSlc12a3_cells_changed": "cells_changed_without_Slc12a3",
                 "noSlc12a3_noUmod_cells_changed": "cells_changed_without_Slc12a3_and_Umod",
                 "agreement_with_other_columns": "fraction_equal_other_labels"}),
    Spec("6b", DCT_LABELS, "deg_cluster_annotation", "DCT cluster annotation: evidence for each cluster at resolution 1",
         ("One row per condition and Leiden cluster (seed 42). median_protein_ids, _rest: median proteins per cell "
          "in the cluster and in the other cells (rank-biserial correlation, Mann-Whitney p, BH q over the "
          "clusters). n_tested, n_sig_up: proteins tested and significantly increased; top10_increased_proteins.",
          "marker_score_ columns: marker-score labels of the cluster's cells (most frequent label, its fraction, "
          "cells per label). detected_ columns: fraction of the cluster's cells with a marker of another cell "
          "population detected. top25_ columns: result of the top-25 rule only. added_step_ columns: candidate cell "
          "types (AUC of their best type-specific marker) and markers (rank; AUC; q; detected in the cluster vs the "
          "other cells)."),
         was="9b", same_as={"rankfree_label": "deg_label", "rankfree_confidence": "deg_confidence"},
         empty=("runner_up", "tied_types"),
         values={"condition": CONDITION, "deg_confidence": CONF_DCT, "top25_confidence": CONF_TOP25},
         rename={"deg_label": "cluster_annotation", "deg_confidence": "confidence",
                 "plate_P1_fraction": "plate1_fraction", "top10_de": "top10_increased_proteins",
                 "majority_cell_label": "majority_marker_score_label_per_cell",
                 "majority_cell_label_fraction": "majority_marker_score_label_per_cell_fraction",
                 "cell_label_counts": "marker_score_label_per_cell_counts",
                 "majority_cluster_label": "majority_marker_score_label_clusters",
                 "majority_cluster_label_fraction": "majority_marker_score_label_clusters_fraction",
                 "cluster_label_counts": "marker_score_label_clusters_counts",
                 "overlap": "top25_markers", "overlap_specific": "top25_type_specific_markers",
                 "p_hypergeom": "top25_p_hypergeom", "q_hypergeom": "top25_q_hypergeom",
                 "mean_marker_z": "top25_mean_marker_z", "mean_z_best_type": "type_with_highest_mean_marker_z",
                 "dct1_umod_slc12a3_only": "top25_dct1_from_umod_slc12a3_only",
                 "rescued": "labelled_by_added_step", "rescue_candidates": "added_step_candidates",
                 "rescue_markers": "added_step_markers", "rescue_best_rank": "added_step_best_rank"}),
    Spec("7a", DCT_LABELS, "submitted_marker_coverage", "DCT marker list: quantification and detection per condition",
         ("One row per condition, cell type and marker of the DCT marker list (Supplementary Table 1). quantified: "
          "the protein is in the data of the condition; scored: used for the marker-score label. n_cells_detected, "
          "fraction_cells_detected: cells of the condition with the protein detected.",
          "types_listing_in_marker_list; shared_in_marker_list: listed for two or more cell types; "
          "shared_for_annotation_rule: treated as shared by the cluster annotation (Slc12a3, and Umod as "
          "TAL-associated)."),
         was="10a",
         values={"condition": CONDITION, "note": {"TAL-associated (the project notes framing; labels_core plan)": "TAL-associated"}},
         rename={"in_matrix": "quantified", "types_listing_in_workbook": "types_listing_in_marker_list",
                 "shared_in_workbook": "shared_in_marker_list", "shared_for_de_rule": "shared_for_annotation_rule"}),
]

# Tables taken out of the workbook on 12 Nov 2026 (former sheets 5 to 7); kept in the sources record only.
REMOVED_REASON = "removed at the authors' request, 12 Nov 2026: shows the submitted labels"
REMOVED = [
    {"previous_table": key, "status": REMOVED_REASON, "title": f"former sheet {sheet}: {title}",
     "source": f"results/tables/{task}/{task}_{name}.csv"}
    for key, sheet, task, name, title in (
        ("5a", "5 (Submitted vs revised labels)", KIDNEY_ANNOTATION, "t15_vs_pre_crosstab_a_endo",
         "cells per submitted Figure 2A label and cell type of the revised annotation"),
        ("5b", "5 (Submitted vs revised labels)", KIDNEY_ANNOTATION, "t15_vs_reannot_crosstab_a_endo",
         "cells per submitted Figure 2B label and cell type of the revised annotation"),
        ("6a", "6 (Submitted re-annotation)", DOTPLOT, "reannot_drivers_by_label",
         "what the labels of the submitted re-annotation rest on"),
        ("7a", "7 (Endothelial proteins)", KIDNEY_ANNOTATION, "extmarkers_endothelial_detection",
         "endothelial and vasa recta proteins in the cells of three submitted Figure 2A labels"))
]

TERMS = [
    ("Marker list; shared; type-specific", "Supplementary Table 1 (kidney list: sheets 1 to 4; DCT list: sheets 5 to "
     "7). A marker listed for two or more cell types is shared, otherwise type-specific. Kidney: the scans were run "
     "on the marker list without and with its endothelial row (Pecam1, Emcn, Eng, Esam, Icam2, Fabp4, Ehd4). DCT: "
     "Slc12a3 and Umod are treated as shared."),
    ("Top-25 rule", "Each Leiden cluster is compared with all other cells (Wilcoxon rank-sum test on ln(1 + "
     "intensity); proteins detected in >= 20% of the cluster's cells; Benjamini-Hochberg (BH) correction). "
     "Increased: q < 0.05, AUC > 0.5 and a positive mean difference. The cluster takes the cell type whose markers "
     "are enriched among its 25 most increased proteins (hypergeometric test, BH over clusters x cell types). "
     "Without a marker among them it is Unknown."),
    ("Added step; rule", "A cluster that the top-25 rule leaves Unknown, or labels from shared markers only, takes "
     "the cell type whose type-specific marker is significantly increased at any rank. rule: 'top-25 rule only' or "
     "'with the added step' (the adopted rule)."),
    ("confidence", "high: a type-specific marker among the 25 most increased proteins and enrichment q < 0.05. "
     "medium: the same with q >= 0.05. type-specific marker, any rank: labelled by the added step. shared markers "
     "only: only shared markers among the 25 most increased proteins. no marker among the top 25: result of the "
     "top-25 rule only for a cluster without a marker among them. no marker support: no marker among them and no "
     "type-specific marker increased at any rank (kidney: Unknown; DCT: subtype not resolved). cluster below 5 "
     "cells: not annotated. ge_medium in a column name: high or medium."),
    ("Kidney cell-type names", "Sheets 1 to 3: labels assigned by the rule. They are the cell types of the marker "
     "list; DCT and CNT/PC are the names used in the annotation (Figure 2A) for the list categories DCT1 and PC. "
     "CNT and DCT2 are separate list categories without a type-specific quantified marker (table 4c). Sheet 4 uses "
     "the list names. Unknown: cells of clusters without marker support."),
    ("graph", "Kidney neighbour graphs (10 neighbours, 20 principal components). graph of the main analysis: the "
     "graph on which the annotation of Figure 2A was made. depth-regressed graph: after regressing out ln(protein "
     "IDs). unscaled, 500 most variable proteins: total-intensity normalization without scaling, 500 most variable "
     "proteins."),
    ("Scan columns", "Leiden algorithm; seed 42 unless stated; clusters below 5 cells are not annotated (below_min "
     "columns). cluster_size_: cells per cluster. n_types_resolved: cell types assigned to at least one cluster; "
     "n_types_high: with confidence high. fraction_cells_unknown: fraction of cells in clusters without a label. "
     "_all_seeds_min, _max: smallest and largest value over 11 seeds (42 and 0 to 9). _vs_seed42 and "
     "same_label_as_seed42: seeds 0 to 9 against seed 42, as adjusted Rand index (ARI) of the clusters or the labels "
     "and as fraction of cells with the same label (median, min, max). per_cell_stability_mean: mean over cells of "
     "the fraction of seeds 0 to 9 with the seed-42 label; fraction_cells_stable_all_10_seeds: cells with the "
     "seed-42 label in all of them."),
    ("protein IDs", "Number of proteins detected in a cell. eta2_protein_ids: share of its variance explained by "
     "the clusters (eta2_ln_protein_ids: of its logarithm). kruskal_p_protein_ids: Kruskal-Wallis test across the "
     "clusters. cluster_median_ids_min, _max: smallest and largest cluster median. "
     "rho_cluster_median_ids_vs_n_sig_up: Spearman correlation, over the clusters, between the median protein IDs "
     "and the number of significantly increased proteins."),
    ("DCT labels", "cluster annotation: the rule above on the Leiden clusters of each condition at resolution 1, "
     f"with the added step. {MARKER_SCORE_LABEL} Control and Zero-K are shown side by side; each condition has its "
     "own neighbour graph."),
    ("DCT Unknown", f"{NOTE_D9}. {UNKNOWN_NOTE}"),
    ("DCT stability columns (table 6a)", "seed_: Leiden seeds 0 to 9 against seed 42 (ARI of the labels; fraction "
     "of cells with the same label in all 10 seeds). noise_: Gaussian noise (SD 1e-6) added to the scaled values, "
     "20 repeats; cells whose label changes (median and maximum over the repeats, and cells changed in any repeat). "
     "loo_: one marker left out at a time; loo_mean_stability: mean over cells of the fraction of these runs with "
     "the same label; loo_n_cells_stability_below_0.5: cells that keep their label in fewer than half of them. "
     "cells_changed_without_: cells whose label changes when the marker is removed from every list."),
    ("plate1", "Fraction of the cells that come from sort plate 1. The two sort plates of a condition hold cells of "
     "the same pooled sample."),
    ("detected; mean_marker_z", "detected: fraction (or number) of cells with an intensity above zero. "
     "mean_marker_z: cluster mean of the scaled ln(1 + intensity) of the cell type's quantified markers."),
]

# Internal names that must not reach the workbook.
FORBIDDEN = re.compile(
    r"(?<![A-Za-z0-9])T\d{2}(?![0-9A-Za-z])|cs_|deg_|rank-?free|rescue|(?<![A-Za-z0-9])v3(?![A-Za-z0-9])|v3_endo"
    r"|kidney_|reannot(?!ation)|_vs_pre|\bFig\.? ?2|workbook|\bbrief\b|\bCtrl\b|ZeroK|score_genes|\.X\b"
    r"|\bstored\b|hvg|unknown_no_marker|low_shared_only|specific_any_rank|below_min_size|frozen|noSlc12a3|argmax"
    r"|dct_label|transfer|in_matrix|\bP1\b", re.I)


def _range(frame: pd.DataFrame, col: str, fmt: str, **sel) -> str:
    mask = pd.Series(True, index=frame.index)
    for key, value in sel.items():
        mask &= frame[key].astype(str).eq(str(value))
    values = pd.to_numeric(frame.loc[mask, col])
    return f"{values.min():{fmt}}, {values.max():{fmt}}"


SG, TOP, ADD = "graph of the main analysis", "top-25 rule only", "with the added step"
ENDO_ADD = "endothelial_row__added_step__"
NUMBERS = [
    ("kidney, top-25 rule only, graph of the main analysis: smallest and largest Unknown fraction over resolutions "
     "0.5 to 6", "1a", lambda f: _range(f, "fraction_cells_unknown", ".2f", graph=SG), "0.38, 0.54", None),
    ("kidney, graph of the main analysis: smallest and largest rho of cluster median protein IDs with increased "
     "proteins", "1a", lambda f: _range(f, "rho_cluster_median_ids_vs_n_sig_up", ".2f", graph=SG), "0.96, 0.99", None),
    ("kidney, with the added step, list without the endothelial row: cell types at 0.5", "1b",
     lambda f: pick(f, "added_step_n_types", graph=SG, resolution=0.5), 4, None),
    ("kidney, with the added step, list without the endothelial row: cell types at 1.0", "1b",
     lambda f: pick(f, "added_step_n_types", graph=SG, resolution=1.0), 6, None),
    ("kidney, with the added step, list without the endothelial row: cell types at 2.0", "1b",
     lambda f: pick(f, "added_step_n_types", graph=SG, resolution=2.0), 7, None),
    ("kidney, resolution 2, with the added step: clusters and cells labelled by the added step", "1b",
     lambda f: f"{pick(f, 'n_clusters_added_step', graph=SG, resolution=2.0)}, "
               f"{pick(f, 'n_cells_added_step', graph=SG, resolution=2.0)}", "10, 310", None),
    ("kidney, resolution 2 with the endothelial row and the added step: cell types (seed 42; min and max over seeds)",
     "3a", lambda f: ", ".join(str(pick(f, c, graph=SG, marker_set="with the endothelial row", rule=ADD))
                               for c in ("n_types", "n_types_all_seeds_min", "n_types_all_seeds_max")),
     "8, 8, 8", None),
    ("kidney, same setting: Unknown fraction", "3a",
     lambda f: pick(f, "fraction_unknown", graph=SG, marker_set="with the endothelial row", rule=ADD), 0.15, 2),
    ("kidney, same setting: cells labelled Endothelial", "3a",
     lambda f: pick(f, "n_cells_endothelial", graph=SG, marker_set="with the endothelial row", rule=ADD), 19, None),
    ("kidney, same setting: cells with the seed-42 label over seeds 0 to 9 (smallest, largest)", "3b",
     lambda f: (lambda v: f"{v.min():.2f}, {v.max():.2f}")(pd.to_numeric(
         f.loc[f["graph"].eq(SG) & f["seed"].ne(42), ENDO_ADD + "same_label_as_seed42"])), "0.90, 1.00", None),
    ("kidney, same setting: seeds with eight cell types", "3b",
     lambda f: int(pd.to_numeric(f.loc[f["graph"].eq(SG), ENDO_ADD + "n_types"]).eq(8).sum()), 11, None),
    ("DCT control, top-25 rule only: smallest and largest number of cells Unknown over the resolutions", "5a",
     lambda f: _range(f, "n_Unknown", ".0f", condition="Control", rule=TOP), "457, 624", None),
    ("DCT control, with the added step: ProLIF cells at resolutions 0.3, 0.5, 0.8, 1 and 3", "5a",
     lambda f: ", ".join(str(pick(f, "n_ProLIF", condition="Control", rule=ADD, resolution=r))
                         for r in (0.3, 0.5, 0.8, 1.0, 3.0)), "0, 0, 0, 0, 10", None),
    ("DCT control, adopted setting: cells per label", "5a",
     lambda f: pick(f, "label_counts", condition="Control", adopted_setting=True),
     "DCT1 382; DCT2 85; Unknown 262", None),
    ("DCT control: cluster annotation, cells per label", "6a",
     lambda f: pick(f, "label_counts", condition="Control", label="cluster annotation"),
     "DCT1 382; DCT2 85; Unknown 262", None),
    ("DCT control: marker-score label (clusters), cells per label", "6a",
     lambda f: pick(f, "label_counts", condition="Control", label=CLUSTERS),
     "DCT1 460; DCT2 126; ProLIF 119; Unknown 24", None),
    ("DCT control, cluster with the most proteins per cell: cells, median proteins, median of the other cells, "
     "annotation, most frequent marker-score label (per cell)", "6b",
     lambda f: (lambda r: f"{r['n_cells']}, {r['median_protein_ids']}, {r['median_protein_ids_rest']}, "
                          f"{r['cluster_annotation']}, {r['majority_marker_score_label_per_cell']}")(
         f[f["condition"].eq("Control")].sort_values("median_protein_ids", ascending=False).iloc[0]),
     "64, 3910, 1656, Unknown, ProLIF", None),
]
# Numbers of the letter that were read from the removed sheets: information rows of the checks table.
LETTER_NOT_IN_WORKBOOK = [
    ("revised annotation: cells per cell type (Endothelial, JG, PCT, PST, TAL, DCT, CNT/PC, IC-B, Unknown)",
     "19, 24, 160, 158, 138, 58, 38, 8, 109"),
    ("Figure 2A MD of the submitted version: cells, and cells now Endothelial", "19, 19"),
    ("Figure 2A DVR of the submitted version: cells; now PST, PCT, Unknown", "88, 52, 3, 33"),
    ("Figure 2A PCT and PST cells of the submitted version that keep their label, of all such cells", "103 of 300"),
    ("Figure 2A of the submitted version, DCT2 -> DCT; CNT -> DCT, CNT/PC; CNT cells", "24; 34, 38; 73"),
    ("Figure 2B labels of the submitted version: cells whose own markers are not above control, all cells",
     "283 (40%) of 712"),
    ("Figure 2B labels of the submitted version: the same for the PC label", "171 (91%) of 187"),
    ("Figure 2A MD cells of the submitted version: Fabp4, Ehd4, Pecam1 detected (cells of 19)", "16, 16, 12"),
]


def extra_checks(frames: dict, stored: dict, cells: dict, checks: list[dict]) -> None:
    """Display names of the kidney labels, the vocabulary of 12 Nov 2026, the D9 note, no condition comparison."""
    label_text = []
    for key, cols in (("1a", ("types_resolved", "type_sets_seen_all_seeds")),
                      ("1b", ("top25_types", "added_step_types")), ("2a", ("cell_type",)), ("3a", ("types",)),
                      ("3b", [c for c in stored["3b"].columns if c.endswith("__types")])):
        label_text += [str(v) for c in cols for v in stored[key][c].dropna()]
    internal = sorted({m for text in label_text
                       for m in re.findall(r"(?<![A-Za-z0-9/-])(?:DCT1|PC)(?![A-Za-z0-9])", text)})
    shown = {name: any(re.search(rf"(?<![A-Za-z0-9/-]){re.escape(name)}(?![A-Za-z0-9])", t) for t in label_text)
             for name in ("DCT", "CNT/PC")}
    s7.check(checks, "kidney labels assigned by the rule (sheets 1 to 3): display names DCT and CNT/PC, no DCT1 or "
             "PC", f"internal names: {internal or 'none'}; display names found: {shown}",
             "internal names: none; DCT and CNT/PC found", not internal and all(shown.values()))
    marker_sets = sorted(set(stored["3a"]["marker_set"]))
    prefixes = sorted({c.split("__")[0] for c in stored["3b"].columns if "__" in c})
    labels = sorted(set(stored["6a"]["label"]))
    definition = [t for texts in cells.values() for t in texts if MARKER_SCORE_LABEL in t]
    s7.check(checks, "shared vocabulary (authors, 12 Nov 2026): marker lists, column prefixes, DCT label names, and the "
             "definition of the marker-score label on the Contents sheet",
             f"marker lists {marker_sets}; prefixes {prefixes}; labels in 6a {labels}; definition found "
             f"{len(definition)} time(s)",
             "marker list without the endothelial row / with the endothelial row; list_without_endothelial_row / "
             "endothelial_row; cluster annotation and the two marker-score labels; definition found",
             marker_sets == sorted(MARKER_SET.values())
             and prefixes == ["endothelial_row", "list_without_endothelial_row"]
             and labels == sorted(["cluster annotation", CLUSTERS, PER_CELL]) and len(definition) >= 1)
    tables = sorted(stored, key=lambda k: (int(k[:-1]), k[-1]))
    s7.check(checks, "the three sheets that showed the labels of the submitted version are not in the workbook; the "
             "DCT sheets are sheets 5 to 7", f"sheets {sorted(cells)}; tables {tables}",
             "Contents and 7 sheets; 13 tables, 1a to 7a",
             len(cells) == 8 and tables == ["1a", "1b", "2a", "3a", "3b", "4a", "4b", "4c", "5a", "5b", "6a", "6b",
                                            "7a"])
    names = {n: SHEETS[n][0] for n in (5, 6, 7)}
    with_note = [n for n, name in names.items() if NOTE_D9 in cells.get(name, [])]
    unknown = [n for n, name in names.items() if any(re.search(r"\bUnknown\b", t) for t in cells.get(name, [])
                                                      if t != NOTE_D9 and not t.startswith("This is the Unknown"))]
    s7.check(checks, f"DCT sheets: the label value Unknown is kept and the sheets carry the note '{NOTE_D9}'",
             f"note on sheets {with_note}; Unknown in the cells of sheets {unknown}",
             "note on sheets [5, 6, 7]; Unknown in the cells of sheets 5 and 6",
             with_note == [5, 6, 7] and {5, 6} <= set(unknown))
    conditions = {key: sorted(set(stored[key]["condition"])) for key in ("5a", "5b", "6a", "6b", "7a")}
    s7.check(checks, "DCT tables: conditions shown as Control and Zero-K in separate rows; table 6a without the "
             "label transferred from the control cells",
             f"conditions {sorted({c for v in conditions.values() for c in v})}; rows in 6a: {len(stored['6a'])}",
             "Control, Zero-K; 6 rows",
             all(v == ["Control", "Zero-K"] for v in conditions.values()) and len(stored["6a"]) == 6)


CONFIG = Config(step=STEP, workbook="report_05_S05b_tables.xlsx", table_id="S-05-b", title=TABLE_TITLE,
                intro=INTRO, sheets=SHEETS, specs=SPECS, terms=TERMS, forbidden=FORBIDDEN,
                script="report_annotation_tables.py", sheet_notes=SHEET_NOTES, numbers=NUMBERS,
                extra_checks=extra_checks, removed=REMOVED, not_in_workbook=LETTER_NOT_IN_WORKBOOK)


if __name__ == "__main__":
    build(CONFIG, parse_args("the response to the reviewers: Supplementary Table S-05-b, annotation scans and evidence."))
