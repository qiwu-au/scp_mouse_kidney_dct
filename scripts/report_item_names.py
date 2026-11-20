#!/usr/bin/env python
"""The names that the figures and tables have in the paper. Shared module; runs nothing.

The scripts name a figure or a table by the label it had while the analyses were made: a supplementary item as
S-NN-x (Supplementary Table S-02-d, Supplementary Figure S-05-e) or by its number in the first version of the
manuscript (Supplemental Table 1), a main figure as Figure F-08, Figure F-19 or with a capital panel letter
(Figure 3A). These labels do not change when the paper is renumbered, so they are the keys. The names of the paper
are looked up here, in two places only:

  - the numbering map (MAP): one row per item, label -> name in the paper, main figures included;
  - the panel table below (PANEL_WORKING, PANELS_REMOVED): the panels of a main figure whose letter changed.

``paper_names(text)`` returns a text with every label replaced by the name of the paper. The scripts that write the
supplementary table workbooks pass every text through it when they write a cell, a sheet name or the title of the
workbook, so a renumbering needs a new map and no change in any script. A name that is already the name of the paper
(Supplementary Table S3, Figure 4f) is left as it is, so the function can be applied twice.

Rules (those of the numbering check of the supplementary tables, which imports them from here):
  - 'Supplementary' or 'Supplemental', 'Figure' or 'Table', and a label: the name of the map, written
    'Supplementary Figure S5' / 'Supplementary Table S3'; lists and ranges (S-08-a to S-08-c) item by item;
  - a short form (Table S3, Fig. S-05-e) is written out; a label on its own (S-12-d) becomes the bare name (S15);
  - a main figure with a capital panel letter is a label: the figure number of the map and the panel in lower case
    (Figure 2A -> Figure 2a), or the panel of the panel table (Figure 3A -> Figure 4e);
  - a main figure number without a panel cannot be told apart by its form and is left as it is, and so is a panel
    of a figure that is not in the paper any more;
  - a column name that carries a panel of a main figure (figure_3a_group) follows the panel table.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import scp_common as cc

# The numbering map: the one place that holds the numbers of the paper.
MAP = cc.DATA_DIR / "paper_item_names.csv"

# Main figures. Panel of the label -> panel in the paper, where the letter itself changed; the other panels of that
# figure were taken out of the paper, so they have no name there.
PANEL_WORKING = {("3", "A"): "e", ("3", "B"): "f"}
PANELS_REMOVED = {"3"}
# Columns of the workbooks that are named after a panel: by its label (their name in the paper follows from the map
# and the panel table), and those that the scripts write in their final form.
COLUMN_LABELS = ("figure_3a_group",)
COLUMN_FINAL_LITERAL = ("shown_in_figure_2b", "position_in_figure_2b")
# Panels that each main figure of the paper has (filled by the numbering check from the legends; empty here).
PANELS: dict[str, set[str]] = {}

SUP: dict[tuple[str, str], str] = {}           # (Figure | Table, label) -> name in the paper (S5)
BY_ID: dict[str, tuple[str, str]] = {}         # S-NN-x -> (Figure | Table, name in the paper)
MAP_TABLES: dict[str, dict] = {}               # name of a table in the paper -> its row of the map
MAIN_WORKING: dict[str, str] = {}              # main figure label (F-08, 3) -> number in the paper
COLUMN_WORKING: dict[str, str] = {}            # column name with a panel label -> column name of the paper
COLUMN_FINAL: set[str] = set()
REVIEWED_NUMBER_ONLY: dict[tuple[str, str], str] = {}   # (table, text): a figure number without a panel, read by eye
N_TABLES = 0
N_FIGURES = 0

SUP_ID = r"(?:S-\d\d-[a-z]|S\d+|\d+)"
SUP_RE = re.compile(rf"\b(Supplementary|Supplemental)\s+(Figure|Table)(s?)\s+({SUP_ID}(?:(?:, and |, | and | to ){SUP_ID})*)"
                    r"(?![A-Za-z0-9])")
SUP_ID_RE = re.compile(SUP_ID)
BARE_RE = re.compile(r"(?<!Supplementary )(?<!Supplemental )(?<!Response )\b(Table|Figure|Fig\.)s?\s+(S-\d\d-[a-z]|S\d+)"
                     r"(?![A-Za-z0-9])|\bSuppl?\.\s*(?:Table|Fig)")
FIG_ID = r"(?:F-08|F-19|\d)(?:[A-Za-z](?![A-Za-z0-9]))?(?:[–-][A-Za-z](?![A-Za-z0-9]))?"
MAIN_RE = re.compile(rf"(?<!Supplementary )(?<!Supplemental )(?<!Response )\b(Figures?)\s+({FIG_ID}"
                     rf"(?:(?:, and |, | and ){FIG_ID})*)(?![A-Za-z0-9])")
FIG_ID_RE = re.compile(r"(F-08|F-19|\d)([A-Za-z])?(?:([–-])([A-Za-z]))?")
COLUMN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9_]*figure_\d[a-z](?:_[A-Za-z0-9_]+)?(?![A-Za-z0-9_])")
WORKING_ID_RE = re.compile(r"(?<![A-Za-z0-9])(?:S-\d\d-[a-z]|F-\d\d)(?![a-z0-9])")
# A text without any of these cannot hold a name (most cells of a workbook: numbers as text, gene names, labels).
QUICK_RE = re.compile(r"Fig|Sup|Table|S-\d\d-|F-\d\d|figure_\d")


def load_map(path: Path | None = None) -> None:
    """Read the numbering map; every table of this module is filled from it (in place)."""
    global MAP, N_TABLES, N_FIGURES
    if path is not None:
        MAP = Path(path)
    sup, by_id, tables, main = {}, {}, {}, {}
    with MAP.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            figure = re.fullmatch(r"Figure (F-\d\d|\d)", row["working_name"])
            if figure:
                new = re.fullmatch(r"Figure (\d)", row["new_name"])
                if new:
                    main[figure.group(1)] = new.group(1)
                continue
            old = re.fullmatch(r"(Supplementary|Supplemental) (Figure|Table) (S-\d\d-[a-z]|\d+)", row["working_name"])
            new = re.fullmatch(r"Supplementary (Figure|Table) (S\d+)", row["new_name"])
            if not (old and new):
                continue
            sup[(old.group(2), old.group(3))] = new.group(2)
            if old.group(3).startswith("S-"):
                by_id[old.group(3)] = (old.group(2), new.group(2))
            if new.group(1) == "Table":
                tables[new.group(2)] = row
    for store, new in ((SUP, sup), (BY_ID, by_id), (MAP_TABLES, tables), (MAIN_WORKING, main)):
        store.clear()
        store.update(new)
    N_TABLES = max(int(v[1:]) for (k, _), v in SUP.items() if k == "Table")
    N_FIGURES = max(int(v[1:]) for (k, _), v in SUP.items() if k == "Figure")
    # What follows from the map and the panel table: column names, and the one figure number that was read by eye.
    COLUMN_WORKING.clear()
    for name in COLUMN_LABELS:
        fig, panel, rest = re.fullmatch(r"figure_(\d)([a-z])(_\w+)", name).groups()
        if fig in MAIN_WORKING:
            new = PANEL_WORKING.get((fig, panel.upper()), panel)
            COLUMN_WORKING[name] = f"figure_{MAIN_WORKING[fig]}{new}{rest}"
    COLUMN_FINAL.clear()
    COLUMN_FINAL.update(COLUMN_WORKING.values())
    COLUMN_FINAL.update(COLUMN_FINAL_LITERAL)
    REVIEWED_NUMBER_ONLY.clear()
    if ("Table", "S-08-d") in SUP and "F-08" in MAIN_WORKING:
        REVIEWED_NUMBER_ONLY[(SUP[("Table", "S-08-d")], f"Figure {MAIN_WORKING['F-08']}")] = (
            "the RNA and protein concordance figure (working name Figure F-08)")
    paper_names.cache_clear()


@dataclass
class Hit:
    found: str
    expected: str
    kind: str
    status: str            # mismatch | ok | check by eye | flag | info
    note: str = ""
    start: int = -1
    end: int = -1

    @property
    def replace(self) -> bool:
        return self.status == "mismatch" and self.start >= 0 and self.expected not in ("", self.found)


def sup_final(kind: str, wid: str) -> str | None:
    """Name in the paper (S5) of a supplementary item given by its label or by its name in the paper."""
    if re.fullmatch(r"S\d+", wid):
        limit = N_TABLES if kind == "Table" else N_FIGURES
        return wid if 1 <= int(wid[1:]) <= limit else None
    return SUP.get((kind, wid))


def main_final(fid: str, own: str, whole: str) -> tuple[str, str, str]:
    """(final text, status, note) of one figure id such as 3B, F-08A, 4f, 4b–d or 3."""
    fig, p1, dash, p2 = FIG_ID_RE.fullmatch(fid).groups()
    tail = (p1 or "") + ((dash + p2) if p2 else "")
    if fig in ("F-08", "F-19"):
        return MAIN_WORKING[fig] + tail.lower(), "mismatch", "working figure name"
    if p1 is None:
        if (own, whole) in REVIEWED_NUMBER_ONLY:
            return fid, "ok", "number only; checked by eye: " + REVIEWED_NUMBER_ONLY[(own, whole)]
        return fid, "check by eye", "figure number without a panel: working and final names look the same"
    if p1.isupper():                                  # working name (capital panel letter)
        final = MAIN_WORKING.get(fig)
        if final is None:
            return fid, "check by eye", "no such working figure"
        letters = []
        for p in (p1, p2):
            if p is None:
                continue
            if (fig, p) in PANEL_WORKING:
                letters.append(PANEL_WORKING[(fig, p)])
            elif fig in PANELS_REMOVED:
                return fid, "check by eye", f"panel {p} of the former Figure {fig} is not in the paper any more"
            else:
                letters.append(p.lower())
        text = final + letters[0] + ((dash + letters[1]) if p2 else "")
        if text.lower() == fid.lower():
            return text, "mismatch", "capital panel letter (figure and panel unchanged)"
        return text, "mismatch", "working figure name (figure number or panel changed)"
    known = PANELS.get(fig)
    if known is None:
        return fid, "check by eye", f"no legend of Figure {fig} in the manuscript copy"
    missing = [p for p in (p1, p2) if p and p not in known]
    if missing:
        return fid, "check by eye", f"Figure {fig} has no panel {', '.join(missing)} in the manuscript copy"
    return fid, "ok", ""


def scan_names(text: str, own: str) -> list[Hit]:
    """Mentions of supplementary items, main figures and figure column names in one text; own = final table id."""
    hits: list[Hit] = []
    taken: list[tuple[int, int]] = []
    for m in SUP_RE.finditer(text):
        kind, plural, ids = m.group(2), m.group(3), m.group(4)
        bad, own_hit = [], False

        def one(x, kind=kind):
            nonlocal own_hit
            final = sup_final(kind, x.group(0))
            if final is None:
                bad.append(x.group(0))
                return x.group(0)
            own_hit = own_hit or (kind == "Table" and final == own)
            return final
        new_ids = SUP_ID_RE.sub(one, ids)
        expected = f"Supplementary {kind}{plural} {new_ids}"
        k = "own number" if own_hit else "cross-reference"
        if bad:
            hits.append(Hit(m.group(0), "", k, "check by eye", f"no final name for {', '.join(bad)}", m.start(), m.end()))
        else:
            found = m.group(0)
            hits.append(Hit(found, expected, k, "ok" if found == expected else "mismatch",
                            "" if found == expected else "working name", m.start(), m.end()))
        taken.append((m.start(), m.end()))
    for m in BARE_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        wid = m.group(2)
        kind = {"Fig.": "Figure"}.get(m.group(1), m.group(1)) if m.group(1) else None
        final = sup_final(kind, wid) if kind and wid else None
        hits.append(Hit(m.group(0), f"Supplementary {kind} {final}" if final else "", "cross-reference",
                        "mismatch" if final else "check by eye", "short form of the name", m.start(), m.end()))
        taken.append((m.start(), m.end()))
    for m in MAIN_RE.finditer(text):
        if any(a < m.end() and m.start() < b for a, b in taken):
            continue
        statuses, notes = [], []

        def fig(x):
            new, status, note = main_final(x.group(0), own, m.group(0))
            statuses.append(status)
            if note:
                notes.append(note)
            return new
        new_ids = re.sub(FIG_ID, fig, m.group(2))
        expected = f"{m.group(1)} {new_ids}"
        status = ("check by eye" if "check by eye" in statuses else "mismatch" if "mismatch" in statuses else "ok")
        hits.append(Hit(m.group(0), expected if status != "check by eye" else "", "main figure", status,
                        "; ".join(dict.fromkeys(notes)), m.start(), m.end()))
        taken.append((m.start(), m.end()))
    for m in COLUMN_RE.finditer(text):
        name = m.group(0)
        if name in COLUMN_WORKING:
            hits.append(Hit(name, COLUMN_WORKING[name], "column name", "mismatch",
                            "column name with a working figure panel", m.start(), m.end()))
        elif name in COLUMN_FINAL:
            hits.append(Hit(name, name, "column name", "ok", "", m.start(), m.end()))
        else:
            hits.append(Hit(name, "", "column name", "check by eye", "column name with a figure panel, not in the list",
                            m.start(), m.end()))
        taken.append((m.start(), m.end()))
    for m in WORKING_ID_RE.finditer(text):
        if any(a <= m.start() < b for a, b in taken):
            continue
        wid = m.group(0)
        if wid in BY_ID:
            kind, final = BY_ID[wid]
            hits.append(Hit(wid, final, "own number" if (kind == "Table" and final == own) else "cross-reference",
                            "mismatch", f"working id of Supplementary {kind} {final}", m.start(), m.end()))
        elif wid in MAIN_WORKING:
            hits.append(Hit(wid, MAIN_WORKING[wid], "main figure", "check by eye", "working figure id without 'Figure'",
                            m.start(), m.end()))
        else:
            hits.append(Hit(wid, "", "cross-reference", "check by eye", "working id without a final name",
                            m.start(), m.end()))
    return hits


def final_text(text: str, own: str) -> str:
    """The text with every mismatch of scan_names replaced by its final name."""
    for h in sorted((h for h in scan_names(text, own) if h.replace), key=lambda h: -h.start):
        text = text[:h.start] + h.expected + text[h.end:]
    return text


@lru_cache(maxsize=None)
def paper_names(text: str) -> str:
    """The text with the names of the paper (what a workbook holds). Which table the text stands in plays no role."""
    if not QUICK_RE.search(text):
        return text
    return final_text(text, "")


def named(value):
    """A cell value as a workbook holds it: text with the names of the paper, anything else unchanged."""
    return paper_names(value) if isinstance(value, str) else value


def labels_left(texts) -> list[str]:
    """Labels that paper_names would still replace in these texts (empty for a workbook with the names of the paper)."""
    found = []
    for text in texts:
        if isinstance(text, str) and paper_names(text) != text:
            found.extend(f"{h.found} -> {h.expected}" for h in scan_names(text, "") if h.replace)
    return sorted(set(found))


load_map()
