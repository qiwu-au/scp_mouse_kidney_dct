#!/usr/bin/env python
"""zerok_vs_control step 1: design record and panel for the zero-K comparison on the combined search.

Analysis plan,
step 1. Checks the plan, the overview and the zerok_core panel hash; copies the zerok_core design facts into
zerok_vs_control_design.md and adds "searched jointly" with the date and the search settings recorded in dct_intake. The 12 facts
zerok_core lists as 'not supplied', plasma K and pNCC were asked for again in the zerok_vs_control plan (2 Nov 2026); the authors approved
recording them as not supplied (pending with the authors). The panel and its expected directions are not changed.

Inputs (read-only): results/tables/dct_intake_zerok_vs_control_frozen_briefs_sha256.csv; the zerok_vs_control plan and overview; zerok_core panel,
panel hash, design record and design questions; dct_intake search settings and per-cell table; the combined export (SHA256
only, from the dct_intake freeze record).
Outputs (results/tables/zerok_vs_control/): zerok_vs_control_design.md, zerok_vs_control_design_questions.csv,
zerok_vs_control_design_cells.csv, zerok_vs_control_design_checks.csv.

Command (project root):  python scripts/zerok_vs_control_design.py
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

import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import zerok_core_lib as t21
import dct_intake_lib as t26
import zerok_vs_control_lib as lib

SCRIPT = "zerok_vs_control_design.py"
ASKED = "2 Nov 2026 (zerok_vs_control plan); authors approved recording as not supplied"
EXTRA_REQUESTED = [
    ("plasma K at harvest (same mice)", "not supplied (pending with the authors)"),
    ("pNCC (phospho-NCC) in the same mice", "not supplied (pending with the authors)"),
]


def composition(run: cc.Run) -> pd.DataFrame:
    """Cells per condition and plate, as dct_intake filtered them (combined search)."""
    cells = t26.read_cells(run)
    rows = []
    for cond in lib.CONDITIONS:
        own = cells[cells["condition"] == cond]
        for plate in (*lib.PLATES, "all"):
            sub = own if plate == "all" else own[own["plate"] == plate]
            rows.append({"condition": cond, "plate": plate, "n_cells_sorted": int(len(sub)),
                         "n_cells_retained": int(sub["retained"].sum()),
                         "n_cells_removed": int((~sub["retained"]).sum())})
    return pd.DataFrame(rows)


def design_markdown(questions: pd.DataFrame, settings: pd.DataFrame, comp: pd.DataFrame, export_sha: str,
                    one_condition: dict[str, int]) -> str:
    known = list(t21.DESIGN_KNOWN)
    lines = [
        "# zerok_vs_control step 1: zero-K experiment design record (combined search)",
        "",
        f"(`scripts/{SCRIPT}`, {cc.now_utc()}).",
        "Written before any zerok_vs_control comparison between conditions was run. It copies the zerok_core design record "
        "(`zerok_core_design.md`, 28 Oct 2026) and adds the combined search. Objective record only: supplied facts, "
        "facts not supplied, and what the design allows.",
        "",
        "## 1. Supplied by the authors",
        "",
        "From zerok_core (28 Oct 2026):",
        "",
    ]
    lines += [f"- {fact}" for fact in known[:3]]
    lines += [f"- {known[3]} *(zerok_core fact for the separate searches intake_core-zerok_response_core, which stay on record; the analysis "
              "here uses the combined search below.)*", ""]
    lines += [
        "Searched jointly (authors, 29 Oct and 2 Nov 2026; dct_intake):",
        "",
        "- Both conditions were searched together in one Spectronaut search. The export "
        f"`data/DCT/DCT_Ctrl_ZeroK_combine_1535cells.csv` (SHA256 `{export_sha}`) was received on 2 Nov 2026, 00:34. "
        "Protein inference, FDR filtering and cross-run normalization ran over all 1,535 runs, so a row of the "
        "export is the same protein group in both conditions.",
        "- The Ctrl intensities therefore carry Spectronaut's cross-run normalization over all 1,535 runs; they are "
        "not the Ctrl-only search.",
        "- The combined search replaces the separate searches for every DCT analysis (dct_intake-zerok_vs_control; the dct_intake-zerok_vs_control plans "
        "were frozen and their SHA256 committed before any intensity value of the export was read).",
        "",
        "Search settings (authors, 2 Nov 2026; `dct_intake_search_settings.csv`; the authors stated the separate "
        "searches used the same settings):",
        "",
        "| setting | combined search |",
        "| --- | --- |",
    ]
    lines += [f"| {r.setting} | {r.combined_search} |" for r in settings.itertuples()]
    lines += [
        "",
        "## 2. Asked for in the plan",
        "",
        "The 12 facts zerok_core asked for, asked again in the zerok_vs_control plan (2 Nov 2026), plus plasma K and pNCC (pending with "
        "the authors). The authors approved recording them as `not supplied`. This table is the authority for the "
        "Methods text; anything still `not supplied` must come from the authors before submission.",
        "",
        "| fact | answer |",
        "| --- | --- |",
    ]
    lines += [f"| {r.fact} | {r.answer} |" for r in questions.itertuples()]
    lines += [
        "",
        "## 3. Observed from the files (not inferred beyond what is written)",
        "",
        "- The cell IDs contain 'GFP' (e.g. `[1] Ctrl_GFP_1cell_WISHDIA_P1_0001.htrms.PG.Quantity`). The reporter "
        "line and the sort gating are not supplied and are not inferred from the file names.",
        "- Diet duration is 4 days (authors, 28 Oct 2026).",
        "- Before this record, dct_intake-dct_rna reported per-condition results of the combined search (detection fraction of "
        "every panel protein per condition, label counts, per-condition summaries side by side). No test between "
        "conditions was run on the combined search before zerok_vs_control.",
        "",
        "## 4. Cells per condition and sort plate (dct_intake filtering, combined search)",
        "",
        cc.markdown_table(comp),
        "",
        "## 5. Consequences for the analysis",
        "",
        "- One pool of 3 mice per condition: condition and pool cannot be separated (n = 1 pool per condition). Any "
        "difference between conditions carries the pool difference with it.",
        "- Cells are not animal-level replicates. Cell-level p-values in zerok_vs_control are descriptive only and are reported "
        "with that statement beside them.",
        "- Plates P1/P2 are sort plates of the pooled cells, so P1 vs P2 agreement is a technical check, never a "
        "biological replication.",
        "- Both conditions were sorted and acquired in the same sessions and searched in one Spectronaut search, so "
        "sorting, acquisition, protein grouping, FDR filtering and cross-run normalization are not "
        "condition-specific. The search confound of intake_core-zerok_response_core does not apply here; the pool confound does.",
        f"- Protein groups retained in one condition only (Ctrl {one_condition.get('Ctrl only', 0)}, ZeroK "
        f"{one_condition.get('ZeroK only', 0)}; dct_intake) are listed and not tested; their presence or absence is not "
        "read as a condition effect.",
        "- Labels (dct_labels, frozen rule): no ProLIF in either condition and no DCT2 in ZeroK under `dct_label`; no "
        "ProLIF under `dct_label_alternative`. Comparisons that need an absent label are written as 'does not "
        "apply' rows, with the other label column's result beside them.",
        "",
        "## 6. Pre-specified panel",
        "",
        f"- `zerok_core_prespecified_panel.csv`, SHA256 `{lib.PANEL_SHA256}` (checked). The panel and its expected "
        "directions are not changed.",
        "",
        "## 7. What this record does not establish",
        "",
        "- No biological claim. The panel states expectations from the literature, not findings.",
        "- The intervention does not establish a new DCT subtype and does not validate the trajectory model (project "
        "framing; editors' letter).",
        "- Whether the diet worked physiologically cannot be checked from the proteome alone; plasma K and pNCC in "
        "the same mice are not supplied.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="zerok_vs_control step 1: design record and panel.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(lib.TASK, "design", args)
    checks = t5.Checks("dct_joint")
    lib.check_inputs(run, checks)

    panel = lib.t22.read_panel(run)
    checks.add("panel rows (zerok_core)", f"{len(panel)}", len(panel) == 33, "33")
    t21_q = pd.read_csv(run.use_input(t21.table_path("design_questions"), "zerok_core design questions"))
    run.use_input(lib.ZEROK_CORE_DESIGN_MD, "zerok_core design record")
    same = [w for w, _ in t21.DESIGN_REQUESTED] == t21_q["fact"].tolist()
    checks.add("zerok_core design questions equal zerok_core_lib.DESIGN_REQUESTED", f"{same}; {len(t21_q)} facts",
               same and len(t21_q) == 12, "12 facts")
    questions = pd.DataFrame(
        [{"fact": f, "answer": a, "supplied": False, "source": "zerok_core (28 Oct 2026)", "asked_again": ASKED}
         for f, a in zip(t21_q["fact"], t21_q["answer"])]
        + [{"fact": f, "answer": a, "supplied": False, "source": "zerok_vs_control plan step 1", "asked_again": ASKED}
           for f, a in EXTRA_REQUESTED])
    checks.add("facts still not supplied", f"{int((~questions['supplied']).sum())} of {len(questions)}", None)

    settings = t26.read_table(run, "search_settings", keep_default_na=False)
    checks.add("dct_intake search settings recorded", f"{len(settings)} rows", len(settings) >= 8, ">= 8")
    freeze = pd.read_csv(run.use_input(lib.FREEZE_TABLE, "freeze record"))
    export_sha = str(freeze.loc[freeze["path"] == "data/DCT/DCT_Ctrl_ZeroK_combine_1535cells.csv", "sha256"].iloc[0])
    comp = composition(run)
    tot = comp.query("plate == 'all'").set_index("condition")["n_cells_retained"].to_dict()
    checks.add("retained cells per condition equal dct_labels (Ctrl 729, ZeroK 702)", f"{tot}", tot == lib.N_CELLS)
    retention = t26.read_table(run, "retention_summary").set_index("retention")["n_protein_groups"].to_dict()
    checks.add("protein groups retained in both / one condition (dct_intake)", f"{retention}",
               retention.get("both conditions") == lib.N_PROTEINS, "both 5897")

    run.text(design_markdown(questions, settings, comp, export_sha, retention), "design")
    run.table(questions, "design_questions")
    run.table(comp, "design_cells")
    run.table(checks.frame(), "design_checks")
    run.finish(parameters={"design_facts_not_supplied": int((~questions["supplied"]).sum()),
                           "user_answer": "Approve as planned (2 Nov 2026): facts recorded as not supplied",
                           "checks_pass": checks.ok})
    if not checks.ok:
        raise SystemExit("A check failed; see zerok_vs_control_design_checks.csv.")


if __name__ == "__main__":
    main()
