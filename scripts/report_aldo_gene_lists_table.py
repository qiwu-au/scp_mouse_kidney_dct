#!/usr/bin/env python
"""The response to the reviewers: Supplementary Table S17 with the ALDO lists as used for the scores of the revision.

(authors, 12 Nov 2026, in the the response to the reviewers session: "Edit
Supplementary Table S17 and put a copy in revision folder"). The row changes are those of the response to the reviewers, section 3e
(results/tables/report/report_13_aldo_slc12a3.md), which the authors approved with the the response to the reviewers plan.

The submitted workbook (first_submission/Supplementary Table S17.xlsx) is read only and is not changed. The revised
table is a new workbook with the same sheet name, title, fonts and frozen header:
- one list per role: aldosterone signaling (15 proteins), induced by aldosterone (44 proteins and Slc12a3),
  repressed by aldosterone (15 proteins), with the score that uses each list;
- the fallback panels, the duplicated induced list, the nine kallikrein paralogs, Vmn1r32 and Tpr are not listed,
  because they were not part of the lists used for the scores (reviewer_core_lib.py, lines 105-128, used by dct_reviewer);
- Slc12a3 is listed in the induced list with a note, and the footnote of the response to the reviewers section 3e (read from the
  response file, not retyped) stands under the table;
- aliases are copied from the submitted workbook.
No protein list is typed here: the three lists come from reviewer_core_lib and are checked against the stored dct_reviewer
coverage table; every listed protein except Slc12a3 must be in the submitted table under the matching score.

Names (since 20 Nov 2026): the title, the two notes and the title property are written through
report_item_names.paper_names, so the saved workbook names itself and the table it cites as the paper does
(numbering map); the creator and description properties stay empty.

Inputs (read-only): first_submission/Supplementary Table S17.xlsx; scripts/reviewer_core_lib.py;
  results/tables/dct_reviewer/dct_reviewer_aldo_marker_coverage.csv;
  results/tables/report/report_13_aldo_slc12a3.md (the footnote)
Outputs: results/tables/report/report_aldo_gene_lists_table.xlsx (the revised table);
  report_13_supplemental_table_2_changes.csv (every row of the submitted table and what became of it);
  report_13_supplemental_table_2_checks.csv; report_step_13_supplemental_table_2_manifest.json

Command (project root):  python scripts/report_aldo_gene_lists_table.py --overwrite
The copy for the authors is revision/Supplementary Table S17.xlsx (copied by hand after the checks pass).
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
from copy import copy

import pandas as pd
from openpyxl import Workbook, load_workbook

import scp_common as cc
import report_item_names as pn
import reviewer_core_lib as t2

TASK = "report"
STEP = "13_supplemental_table_2"
SUBMITTED = cc.PROJECT_ROOT / "first_submission" / "Supplemental Table 2.xlsx"
COVERAGE = cc.OUT_ROOTS["tables"] / "dct_reviewer" / "dct_reviewer_aldo_marker_coverage.csv"
THREAD = cc.OUT_ROOTS["tables"] / TASK / "report_13_aldo_slc12a3.md"
WORKBOOK = "report_aldo_gene_lists_table.xlsx"

SIGNALING, INDUCED, REPRESSED = "Aldosterone signaling", "Induced by aldosterone", "Repressed by aldosterone"
SCORE = {SIGNALING: "ALDO machinery", INDUCED: "ALDO activity; ALDO signed", REPRESSED: "ALDO signed"}
HEADER = ["List", "Score", "Protein", "Aliases", "Note"]
WIDTHS = {"A": 26, "B": 28, "C": 14, "D": 24, "E": 46}
ADDED = "Slc12a3"
ADDED_NOTE = "Added to test its influence on the scores (see the note below the table)"
SCORE_NOTE = ("ALDO activity: mean z-score of the induced list. ALDO signed: mean z-score of the induced list minus "
              "that of the repressed list. ALDO machinery: mean z-score of the signaling list.")
PROCESS_WORDS = re.compile(r"submitted|submission|re-?annotation|fallback", re.I)


def check(checks: list[dict], name: str, observed, expected, ok: bool) -> None:
    checks.append({"check": name, "observed": str(observed), "expected": str(expected), "status": "pass" if ok else "FAIL"})
    print(f"{'pass' if ok else 'FAIL'}  {name}: {observed}")


def by_name(names) -> list[str]:
    return sorted(names, key=str.lower)


def read_submitted(path) -> tuple[pd.DataFrame, dict]:
    ws = load_workbook(path)["ALDO"]
    header = [ws.cell(row=3, column=j).value for j in range(1, 6)]
    rows = [[ws.cell(row=r, column=j).value for j in range(1, 6)] for r in range(4, ws.max_row + 1)]
    frame = pd.DataFrame(rows, columns=["score", "component", "direction", "protein", "aliases"])
    frame.insert(0, "submitted_row", range(4, ws.max_row + 1))
    frame = frame[frame["protein"].notna()].reset_index(drop=True)
    style = {"title": ws.cell(row=1, column=1).value, "header": header, "sheet": ws.title, "freeze": ws.freeze_panes,
             "title_font": copy(ws.cell(row=1, column=1).font), "head_font": copy(ws.cell(row=3, column=1).font),
             "head_alignment": copy(ws.cell(row=3, column=1).alignment), "body_font": copy(ws.cell(row=4, column=1).font),
             "title_height": ws.row_dimensions[1].height}
    return frame, style


def footnote_from_thread(path) -> str:
    lines = [s for s in path.read_text(encoding="utf-8").splitlines() if s.startswith("7. Footnote: ")]
    if len(lines) != 1:
        raise SystemExit(f"expected one line '7. Footnote: ...' in {cc.rel(path)}, found {len(lines)}")
    text = lines[0][len("7. Footnote: "):].strip()
    if not (text.startswith('"') and text.endswith('"')):
        raise SystemExit("the footnote line is not one quoted sentence block")
    return text[1:-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="the response to the reviewers: revised Supplemental Table 2 (ALDO lists as used).")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, STEP, args)
    checks: list[dict] = []

    sub, style = read_submitted(run.use_input(SUBMITTED, "submitted Supplemental Table 2"))
    coverage = pd.read_csv(run.use_input(COVERAGE, "dct_reviewer ALDO list coverage"))
    footnote = footnote_from_thread(run.use_input(THREAD, "the response to the reviewers, section 3e (footnote)"))
    run.use_input(cc.SCRIPTS_DIR / "reviewer_core_lib.py", "ALDO lists as used (reviewer_core / ctrl_reviewer_core / dct_reviewer)")

    lists = {SIGNALING: list(t2.FIG3_ALDO_MACHINERY), INDUCED: list(t2.ALDO_UP_WITH_SLC12A3),
             REPRESSED: list(t2.ALDO_SIGNATURE_DOWN)}
    check(checks, "lists as used: signaling 15, induced 45 (44 and Slc12a3), repressed 15; no protein twice in a list",
          {k: len(v) for k, v in lists.items()}, "15, 45, 15",
          [len(lists[k]) for k in (SIGNALING, INDUCED, REPRESSED)] == [15, 45, 15]
          and all(len(set(v)) == len(v) for v in lists.values())
          and lists[INDUCED] == [*t2.ALDO_SIGNATURE_UP, ADDED] and ADDED not in t2.ALDO_SIGNATURE_UP)

    # the same lists in the stored dct_reviewer coverage table (base variant; configured = present + missing), both conditions
    wanted = {SIGNALING: ("ALDO_machinery", "positive"), INDUCED: ("ALDO_activity", "positive"),
              REPRESSED: ("ALDO_signed", "negative")}
    bad = []
    for cond in sorted(coverage["condition"].unique()):
        for name, (family, direction) in wanted.items():
            row = coverage[(coverage["condition"] == cond) & (coverage["score_family"] == family)
                           & (coverage["variant"] == "base") & (coverage["direction"] == direction)]
            listed = [g for col in ("present", "missing") for g in str(row.iloc[0][col]).split(";") if g and g != "nan"]
            if len(row) != 1 or sorted(listed) != sorted(lists[name]) or int(row.iloc[0]["n_configured"]) != len(lists[name]):
                bad.append(f"{cond} {family}")
    signed_up = coverage[(coverage["score_family"] == "ALDO_signed") & (coverage["variant"] == "base")
                         & (coverage["direction"] == "positive")]
    same_up = all(sorted(g for col in ("present", "missing") for g in str(r[col]).split(";") if g and g != "nan")
                  == sorted(lists[INDUCED]) for _, r in signed_up.iterrows())
    check(checks, "lists equal the lists of the stored dct_reviewer coverage table (Control and Zero-K; ALDO activity and the "
          "induced part of ALDO signed use the same list)", bad or "equal in both conditions", "equal",
          not bad and same_up and len(signed_up) == 2)

    # the submitted table: scores, components, aliases
    sub["protein"] = sub["protein"].astype(str).str.strip()
    alias_sets = sub.groupby("protein")["aliases"].agg(lambda s: sorted({str(a).strip() for a in s if pd.notna(a)}))
    alias_bad = alias_sets[alias_sets.map(len) > 1]
    aliases = {p: (a[0] if a else None) for p, a in alias_sets.items()}
    check(checks, "submitted table: 146 rows; one alias entry per protein", f"{len(sub)} rows; proteins with two "
          f"alias entries: {list(alias_bad.index) or 'none'}", "146 rows; none", len(sub) == 146 and alias_bad.empty)

    in_sub = {SIGNALING: set(sub.loc[sub["score"] == "ALDO_machinery", "protein"]),
              INDUCED: set(sub.loc[(sub["score"] == "ALDO_signed") & (sub["direction"] == "UP"), "protein"]),
              REPRESSED: set(sub.loc[(sub["score"] == "ALDO_signed") & (sub["direction"] == "DOWN"), "protein"])}
    not_in_sub = {k: sorted(set(v) - in_sub[k]) for k, v in lists.items()}
    check(checks, "every listed protein is in the submitted table under the matching score, except the added Slc12a3",
          {k: v for k, v in not_in_sub.items() if v}, f"only {ADDED} (induced)",
          not_in_sub == {SIGNALING: [], INDUCED: [ADDED], REPRESSED: []})

    fallback = sub["component"].astype(str).eq("extended_fallback (no machinery)")
    slc_rows = sub[sub["protein"] == ADDED]
    check(checks, "submitted table: Slc12a3 is listed once, in the fallback panel of ALDO activity",
          slc_rows[["submitted_row", "score", "component"]].to_dict("records"), "one row, ALDO_activity fallback",
          len(slc_rows) == 1 and bool(fallback[slc_rows.index[0]]) and slc_rows.iloc[0]["score"] == "ALDO_activity")

    # the revised table
    rows = []
    twice = set(lists[SIGNALING]) & set(lists[INDUCED])
    for name in (SIGNALING, INDUCED, REPRESSED):
        for protein in by_name(lists[name]):
            note = ADDED_NOTE if (name == INDUCED and protein == ADDED) else None
            if protein in twice and note is None:
                note = f"Also in the {'induced' if name == SIGNALING else 'signaling'} list"
            alias = "NCC" if protein == ADDED else aliases.get(protein)
            rows.append({"List": name, "Score": SCORE[name], "Protein": protein, "Aliases": alias, "Note": note})
    table = pd.DataFrame(rows, columns=HEADER)
    table["revised_row"] = range(4, 4 + len(table))
    check(checks, "revised table: 75 rows (15 signaling, 45 induced, 15 repressed), no list and protein twice",
          table["List"].value_counts().to_dict(), "75 rows", len(table) == 75
          and not table.duplicated(["List", "Protein"]).any())
    check(checks, "Slc12a3: one row, induced list, alias NCC as in the submitted table, with the note",
          table.loc[table["Protein"] == ADDED, ["List", "Aliases", "Note"]].to_dict("records"), "one row",
          (table["Protein"] == ADDED).sum() == 1 and aliases.get(ADDED) == "NCC"
          and table.loc[table["Protein"] == ADDED, "List"].iloc[0] == INDUCED)
    check(checks, "proteins in two lists (noted in the table)", sorted(twice), "Fkbp5, Nr3c2",
          sorted(twice) == ["Fkbp5", "Nr3c2"])

    # every submitted row and what became of it
    row_of = {(r["List"], r["Protein"]): r["revised_row"] for _, r in table.iterrows()}
    outcome = []
    for _, r in sub.iterrows():
        score, comp, direction, protein = r["score"], str(r["component"]), r["direction"], r["protein"]
        if comp == "extended_fallback (no machinery)":
            where, what = None, "not listed: fallback panel of ALDO activity, which was not used"
        elif score == "ALDO_machinery" and protein in lists[SIGNALING]:
            where, what = SIGNALING, "listed: signaling list" + (" (core and extended list merged)" if comp else "")
        elif score == "ALDO_activity" and protein in lists[INDUCED]:
            where, what = INDUCED, "listed: induced list (one list for ALDO activity and ALDO signed)"
        elif score == "ALDO_signed" and direction == "UP" and protein in lists[INDUCED]:
            where, what = INDUCED, "listed: induced list"
        elif score == "ALDO_signed" and direction == "DOWN" and protein in lists[REPRESSED]:
            where, what = REPRESSED, "listed: repressed list"
        else:
            where, what = None, "not listed: not part of the lists used for the scores"
        outcome.append({**r.to_dict(), "outcome": what, "revised_list": where,
                        "revised_row": row_of.get((where, protein)) if where else None})
    changes = pd.DataFrame(outcome)
    changes.loc[len(changes)] = {"submitted_row": None, "score": None, "component": None, "direction": None,
                                 "protein": ADDED, "aliases": "NCC", "outcome": "added: induced list, with a note",
                                 "revised_list": INDUCED, "revised_row": row_of[(INDUCED, ADDED)]}
    gone = changes[changes["revised_list"].isna()]
    gone_unused = sorted(set(gone.loc[gone["outcome"].str.startswith("not listed: not part"), "protein"]))
    expected_unused = sorted([f"Klk1b{n}" for n in (1, 3, 4, 5, 8, 9, 11, 24, 27)] + ["Tpr", "Vmn1r32"])
    left_table = sorted(set(sub["protein"]) - set(table["Protein"]))
    check(checks, "rows not carried over: 5 fallback rows; 18 kallikrein-paralog rows, Vmn1r32 and Tpr (not part of "
          "the lists used)", f"{int(gone['outcome'].str.contains('fallback').sum())} fallback rows; "
          f"{len(gone) - int(gone['outcome'].str.contains('fallback').sum())} other rows: {gone_unused}",
          f"5; 20: {expected_unused}", int(gone["outcome"].str.contains("fallback").sum()) == 5
          and gone_unused == expected_unused and len(gone) == 25)
    check(checks, "proteins of the submitted table that are not in the revised table",
          left_table, expected_unused + ["Atp1b1"], left_table == sorted(expected_unused + ["Atp1b1"]))
    listed_rows = changes[changes["revised_list"].notna() & changes["submitted_row"].notna()]
    check(checks, "every submitted row is accounted for (146 = 121 carried over into 74 rows + 25 not listed), and "
          "every revised row except Slc12a3 has a submitted row",
          f"{len(listed_rows)} carried over into {listed_rows['revised_row'].nunique()} rows; {len(gone)} not listed",
          "121 into 74; 25", len(listed_rows) == 121 and listed_rows["revised_row"].nunique() == 74 and len(gone) == 25)

    # the workbook: layout of the submitted file (sheet name, title, fonts, frozen header)
    out = cc._check_writable(cc.OUT_ROOTS["tables"] / TASK / WORKBOOK, run.overwrite)
    wb = Workbook()
    ws = wb.active
    ws.title = style["sheet"]
    ws.cell(row=1, column=1, value=pn.paper_names(style["title"])).font = style["title_font"]   # name of the paper
    if style["title_height"]:
        ws.row_dimensions[1].height = style["title_height"]
    for j, name in enumerate(HEADER, 1):
        c = ws.cell(row=3, column=j, value=pn.named(name))
        c.font, c.alignment = copy(style["head_font"]), copy(style["head_alignment"])
    for i, r in table.iterrows():
        for j, name in enumerate(HEADER, 1):
            value = r[name]
            c = ws.cell(row=4 + i, column=j, value=None if value is None or pd.isna(value) else pn.named(value))
            c.font = copy(style["body_font"])
    note_row = 4 + len(table) + 1
    for k, text in enumerate((SCORE_NOTE, footnote)):
        ws.cell(row=note_row + k, column=1, value=pn.paper_names(text)).font = copy(style["body_font"])
    for col, width in WIDTHS.items():
        ws.column_dimensions[col].width = width
    ws.freeze_panes = style["freeze"]
    # title: the name the table has in the paper; creator and description stay empty (the script is in the manifest)
    wb.properties.title = pn.paper_names("Supplemental Table 2")
    wb.properties.creator = ""                   # an empty element; None would be written as the name of the library
    wb.properties.description = None
    wb.save(out)
    run.outputs.append(cc.rel(out))
    print(f"Saved workbook: {cc.rel(out)}")

    # read back
    back = load_workbook(out)
    bs = back[style["sheet"]]
    stored = [[bs.cell(row=4 + i, column=j).value for j in range(1, 6)] for i in range(len(table))]
    want = [[None if (v is None or pd.isna(v)) else v for v in r] for r in table[HEADER].itertuples(index=False)]
    texts = [str(c.value) for row in bs.iter_rows() for c in row if c.value is not None]
    process = [t[:60] for t in texts if PROCESS_WORDS.search(t)]
    check(checks, "workbook read back: one sheet named as in the submitted file, same title, header in row 3, the 75 "
          "rows equal the table, two note lines under the table, panes frozen at A4",
          f"sheets {back.sheetnames}; title equal {bs.cell(row=1, column=1).value == pn.paper_names(style['title'])}; "
          f"rows equal {stored == want}; notes in rows {note_row} and {note_row + 1}", "all equal",
          back.sheetnames == [style["sheet"]] and bs.cell(row=1, column=1).value == pn.paper_names(style["title"])
          and [bs.cell(row=3, column=j).value for j in range(1, 6)] == HEADER and stored == want
          and bs.cell(row=note_row, column=1).value == pn.paper_names(SCORE_NOTE)
          and bs.cell(row=note_row + 1, column=1).value == pn.paper_names(footnote)
          and bs.freeze_panes == "A4" and bs.max_row == note_row + 1)
    check(checks, "footnote equals the footnote of the response to the reviewers, section 3e", footnote[:70] + " ...", "equal",
          bs.cell(row=note_row + 1, column=1).value == pn.paper_names(footnote) and footnote.startswith("Slc12a3 was added"))
    left = pn.labels_left({back.properties.title or "", *back.sheetnames, *texts})
    check(checks, "names of the paper: no cell, sheet name or title of the saved workbook names a figure or a table by "
          "its working label (numbering map and panel table of report_item_names)",
          "; ".join(left[:12]) or f"none in {len(texts)} cells", "none", not left)
    check(checks, "no cell names the revision process or a fallback panel (submitted, submission, re-annotation, "
          "fallback, tool names)", process or f"0 of {len(texts)} cells", "0", not process)
    empty_words = [t for t in texts if t.strip().lower() in ("", "nan", "none", "null")]
    n_alias = int(table["Aliases"].notna().sum())
    sub_alias = {p for p, a in aliases.items() if a}
    check(checks, "aliases: a cell is filled only where the submitted table has an alias; no cell holds an empty "
          "word (nan, none)", f"{n_alias} alias cells; proteins with an alias in the submitted table and listed here: "
          f"{len(sub_alias & set(table['Protein']))}; empty words: {len(empty_words)}", "equal numbers; 0",
          not empty_words and set(table.loc[table["Aliases"].notna(), "Protein"]) == sub_alias & set(table["Protein"]))
    fonts = {(c.font.name, c.font.sz) for row in bs.iter_rows() for c in row if c.value is not None}
    check(checks, "fonts as in the submitted file (title 16 pt bold, other cells 11 pt, same typeface)", sorted(fonts),
          "the typeface of the submitted file, 16 and 11 pt",
          fonts == {(style["title_font"].name, style["title_font"].sz), (style["body_font"].name, style["body_font"].sz)})

    run.table(changes, f"{STEP}_changes")
    failed = [c for c in checks if c["status"] != "pass"]
    run.table(pd.DataFrame(checks), f"{STEP}_checks")
    run.note(f"{len(checks) - len(failed)} of {len(checks)} checks pass")
    run.finish({"rows": len(table), "lists": {k: len(v) for k, v in lists.items()}, "checks_failed": len(failed)},
               status="completed" if not failed else "failed checks")
    if failed:
        raise SystemExit(f"{len(failed)} check(s) failed; see the checks table.")


if __name__ == "__main__":
    main()
