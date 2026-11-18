#!/usr/bin/env python
"""zerok_core shared settings: design record, pre-specified panel, paths.



Holds what steps 1-2 freeze before any zero-K vs control result is inspected:

  DESIGN_KNOWN      design facts supplied by the authors (28 Oct 2026)
  DESIGN_REQUESTED  facts asked for in the plan; "not supplied" until the authors answer
  DESIGN_CONSEQUENCES  what the design allows and does not allow
  PANEL             the pre-specified response: one entry per protein, score or limit,
                    with the expected direction under zero-K and its source
  ALDO_LISTS / CELL_CYCLE_LIST  the score lists (imported from reviewer_core_lib, unchanged)

Expected directions were drafted from a literature search (revision, 28 Oct 2026) and
are the authors' to confirm or edit; ``needs_user_confirmation`` marks the entries the
search did not settle. Every source measures whole-kidney or cortex homogenates (or
cell counts), never single DCT cells; this is stated in the panel file and the README.

Imported by zerok_core_design.py and zerok_core_panel.py; no analysis libraries needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scp_common as cc
import reviewer_core_lib as t2
import intake_core_lib as t17

TASK = "zerok_core"
SEED = cc.RANDOM_SEED
CONDITIONS = t17.CONDITIONS                 # ("Ctrl", "ZeroK")
FILE_KEY = t17.FILE_KEY                     # {"Ctrl": "ctrl", "ZeroK": "zerok"}
LOW_COVERAGE_FRACTION = 0.10                # plan step 2: under 10% of cells in either condition

# ---------------------------------------------------------------------------
# Step 1: design record
# ---------------------------------------------------------------------------
DESIGN_KNOWN = list(t17.DESIGN_FACTS)       # authors, 28 Oct 2026 (recorded in intake_core; reused unchanged)

NOT_SUPPLIED = "not supplied"
DESIGN_REQUESTED = [
    ("diet: control diet product and supplier", NOT_SUPPLIED),
    ("diet: zero-K diet product and supplier", NOT_SUPPLIED),
    ("diet: K content of the control diet", NOT_SUPPLIED),
    ("diet: K content of the zero-K diet", NOT_SUPPLIED),
    ("diet: other differences between the two diets (Na, Cl, Mg, Ca)", NOT_SUPPLIED),
    ("mouse line / strain", NOT_SUPPLIED),
    ("sex", NOT_SUPPLIED),
    ("age or weight at the start of the diet", NOT_SUPPLIED),
    ("sort gating (GFP+? other gates, viability stain)", NOT_SUPPLIED),
    ("run order of the four plates on the instrument (interleaved?)", NOT_SUPPLIED),
    ("plasma or urine K measured at harvest", NOT_SUPPLIED),
    ("anything else to record", NOT_SUPPLIED),
]

DESIGN_OBSERVED = [
    "The cell IDs contain 'GFP' (e.g. `[1] Ctrl_GFP_1cell_WISHDIA_P1_0001.htrms.PG.Quantity`). The reporter "
    "line and the sort gating are not supplied and are not inferred from the file names.",
    "Diet duration is 4 days (authors, 28 Oct 2026). The published low-K studies closest to the panel use 3 days "
    "to 8 weeks; the 4-day design matches Castaneda-Bueno 2014 and Wade 2015.",
    "intake_core already reported, per panel protein, the fraction of cells it is detected in under each condition "
    "(intake_core_marker_coverage.csv, `zeroK_candidate_panel` rows). Per-condition detection was therefore "
    "visible before this freeze. No intensity, label or difference between conditions has been inspected.",
]

DESIGN_CONSEQUENCES = [
    "One pool of 3 mice per condition: condition and pool cannot be separated (n = 1 pool per condition). "
    "Any difference between conditions carries the pool difference with it.",
    "Cells are not animal-level replicates. Cell-level p-values in zerok_response_core are descriptive only and are reported "
    "with that statement beside them.",
    "Plates P1/P2 are sort plates of the pooled cells, so P1 vs P2 agreement is a technical check, never a "
    "biological replication.",
    "Both conditions were sorted and acquired in the same sessions, so sorting and acquisition are not "
    "confounded with condition; the Spectronaut searches were run separately, so protein grouping, cross-run "
    "normalization and intensity scales are condition-specific (intake_core).",
    "Proteins found in one search only cannot be read as condition effects (intake_core: 1,073 Ctrl and 401 ZeroK "
    "groups have no partner). zerok_response_core keeps them in a separate list.",
]

# ---------------------------------------------------------------------------
# Step 2: pre-specified response
# ---------------------------------------------------------------------------
# Literature sources (full citations in zerok_core_panel_sources.md).
SOURCES = {
    "castaneda2014": (
        "Castaneda-Bueno M, Cervantes-Perez LG, Rojas-Vega L, et al. Modulation of NCC activity by low and "
        "high K+ intake: insights into the signaling pathways involved. Am J Physiol Renal Physiol. "
        "2014;306(12):F1507-F1519. PMC4059971",
        "mouse, 4 days, 0% K+ diet vs 1.2% K+; renal cortex immunoblot"),
    "wade2015": (
        "Wade JB, Liu J, Coleman R, et al. SPAK-mediated NCC regulation in response to low-K+ diet. "
        "Am J Physiol Renal Physiol. 2015;308(8):F923-F931",
        "mouse, 4 days, low-K+ diet; cortical homogenate immunoblot"),
    "ishizawa2016": (
        "Ishizawa K, Xu N, Loffing J, et al. Potassium depletion stimulates Na-Cl cotransporter via "
        "phosphorylation and inactivation of the ubiquitin ligase Kelch-like 3. Biochem Biophys Res Commun. "
        "2016;480(4):745-751. PMC5144580",
        "mouse, low-K+ diet; kidney immunoblot and DCT immunostaining"),
    "wang2018": (
        "Wang MX, Cuevas CA, Su XT, et al. Potassium intake modulates the thiazide-sensitive sodium-chloride "
        "cotransporter (NCC) activity via the Kir4.1 potassium channel. Kidney Int. 2018;93(4):893-902. "
        "PMC6683724",
        "mouse, 7 days, K+-deficient diet; whole-cell recording and immunoblot (tNCC ~180% of control)"),
    "wu2019": (
        "Wu P, Gao ZX, Zhang DD, et al. Deletion of Kir5.1 impairs renal ability to excrete potassium during "
        "increased dietary potassium intake. J Am Soc Nephrol. 2019;30(8):1425-1438",
        "mouse, low- and high-K+ diets; pNCC/tNCC immunoblot in wild-type and Kcnj16-/- mice"),
    "murillo2022": (
        "Murillo-de-Ozores AR, Chavez-Canales M, de los Heros P, et al. Multiple molecular mechanisms are "
        "involved in the activation of the kidney sodium-chloride cotransporter by hypokalemia. "
        "Kidney Int. 2022;102(5):1030-1041",
        "mouse, low-K+ diet; WNK4, KS-WNK1 and phospho-WNK4-RRxS increased in wild-type and WNK4-L319F mice"),
    "ostrosky2021": (
        "Ostrosky-Frid M, Chavez-Canales M, Zhang J, et al. Role of KLHL3 and dietary K+ in regulating "
        "KS-WNK1 expression. Am J Physiol Renal Physiol. 2021;320(5):F734-F747",
        "mouse; KS-WNK1 detectable only after a low-K+ diet"),
    "saritas2018": (
        "Saritas T, Puelles VG, Su XT, McCormick JA, Welling PA, Ellison DH. Optical clearing in the kidney "
        "reveals potassium-mediated tubule remodeling. Cell Rep. 2018;25(10):2668-2675. PMC6339512",
        "mouse, 3 days, 0.002% K+ diet (Envigo TD.88239) vs 1.2% K+; BrdU and Ki-67 cell counts, DCT1 by "
        "parvalbumin; proliferation is a cell count, not abundance per cell"),
    "murali2022cells": (
        "Murali SK, Little R, Poulsen SB, Ferdaus MZ, Ellison DH, McCormick JA, Fenton RA. Potassium effects "
        "on NCC are attenuated during inhibition of Cullin E3-ubiquitin ligases. Cells. 2022;11(1):95. "
        "PMC8750104",
        "mouse, 0% / 1% / 5% K+ diets for 2-3 weeks (0% K+ for 2 weeks), whole-kidney homogenate immunoblot; "
        "total Cul1-Cul5 unchanged, neddylated/total CUL3 ratio lower on 0% K+. This project's own group"),
    "murali2026": (
        "Murali SK, D'Acierno M, Zheng X, Rosenbaek LK, Odgaard LN, Grimm PR, Ramesova A, Little R, "
        "Radloff J, Welling PA, Wu Q, Erben RG, Fenton RA. Dietary potassium restriction causes "
        "hypercalciuria, hypocalcemia, and bone loss in male mice. JCI Insight. 2026;11(5):e196339",
        "male C57BL/6J; study 1: 0% K+ for 2 weeks, kidney homogenate proteomics; study 2: 0.1% vs 1% K+ for "
        "4 or 8 weeks, immunoblot. This project's own group (source accepted by the authors, 28 Oct 2026)"),
}

UP, DOWN, NONE = "up", "down", "no expectation"
# Fourth value, added at the authors' request (28 Oct 2026): a cited prediction of NO difference, which zerok_response_core can
# score as match or mismatch. "no expectation" means the literature does not say; "no change" means it does.
NO_CHANGE = "no change"
DIRECTION_VALUES = (UP, DOWN, NO_CHANGE, NONE)
LIT = "literature (literature search, 28 Oct 2026)"
USER_CONFIRMED = "literature (literature search) + authors decision, 28 Oct 2026"

# Why no ALDO score carries a direction (authors decision, 28 Oct 2026, after the list coverage below).
# Plasma aldosterone falls on a low-K+ diet, but the members detected here are not the aldosterone-induced
# programme, so a score built from them is not an aldosterone readout in this matrix.
ALDO_WHY = (
    "Plasma aldosterone falls on a low-K+ diet (about 90 vs 232 pg/ml, p < 0.001; Castaneda-Bueno 2014) and "
    "the NCC response is aldosterone-independent, which would argue for 'down'. That argument does not hold "
    "for this matrix: the aldosterone-INDUCED members are mostly not detected (Sgk1, Per3, Zbtb16 and Scnn1a "
    "absent from the joint object; Tsc22d3 in 12% of Ctrl and 8% of ZeroK cells), so the score is dominated by "
    "its best-detected members, which are transport, metabolic and scaffold proteins with no expectation, plus "
    "Slc12a3, which is expected up. Direction therefore 'no expectation' (authors, 28 Oct 2026). zerok_response_core still reports "
    "every score, with and without Slc12a3 (the project notes).")

# Each entry: item, item_type, panel_group, expected_direction, direction_set_by, source keys, note.
# ``needs_user_confirmation`` is True where the search did not settle the direction (none after the authors'
# decisions of 28 Oct 2026; the frozen panel must have no such row).
PANEL: list[dict] = [
    dict(item="Slc12a3", item_type="protein", panel_group="Slc12a3 (NCC)", expected_direction=UP,
         direction_set_by=LIT, sources=["castaneda2014", "wang2018", "wu2019", "murali2026"],
         note="Total NCC abundance increased on a low-K+ diet: 4 days 0% K+ (p < 0.001); ~180% of control "
              "after 7 days; also increased in the 0.1% K+ study. The 4-day design matches Castaneda-Bueno 2014.",
         needs_user_confirmation=False),
    dict(item="Wnk1", item_type="protein", panel_group="WNK-SPAK/OSR1-KLHL3-CUL3", expected_direction=UP,
         direction_set_by=LIT, sources=["murillo2022", "ostrosky2021"],
         note="Kidney-specific WNK1 (KS-WNK1) increased on a low-K+ diet; reported as detectable only after "
              "low K+. The matrix cannot separate KS-WNK1 from L-WNK1, so this applies to total Wnk1.",
         needs_user_confirmation=False),
    dict(item="Wnk4", item_type="protein", panel_group="WNK-SPAK/OSR1-KLHL3-CUL3", expected_direction=UP,
         direction_set_by=LIT, sources=["ishizawa2016", "murillo2022"],
         note="WNK4 accumulates in the DCT under K+ depletion (reduced KLHL3-mediated degradation).",
         needs_user_confirmation=False),
    dict(item="Stk39", item_type="protein", panel_group="WNK-SPAK/OSR1-KLHL3-CUL3", expected_direction=NONE,
         direction_set_by=LIT, sources=["castaneda2014", "wade2015"],
         note="Conflicting at the abundance level: total SPAK unchanged in one 4-day study, increased in "
              "another. The consistent low-K+ effect is phosphorylation (pS383), which is not measurable here.",
         needs_user_confirmation=False),
    dict(item="Oxsr1", item_type="protein", panel_group="WNK-SPAK/OSR1-KLHL3-CUL3", expected_direction=NONE,
         direction_set_by=LIT, sources=["castaneda2014"],
         note="Total OSR1 unchanged after 4 days of 0% K+; phospho-OSR1 was not detectable in kidney.",
         needs_user_confirmation=False),
    dict(item="Klhl3", item_type="protein", panel_group="WNK-SPAK/OSR1-KLHL3-CUL3", expected_direction=DOWN,
         direction_set_by=LIT, sources=["ishizawa2016"],
         note="Total KLHL3 falls and pS433 rises under K+ depletion. intake_core: this protein is a level-2 match "
              "(different isoform sets in the two searches).",
         needs_user_confirmation=False),
    dict(item="Cul3", item_type="protein", panel_group="WNK-SPAK/OSR1-KLHL3-CUL3", expected_direction=NO_CHANGE,
         direction_set_by=USER_CONFIRMED, sources=["murali2022cells"],
         note="Total CUL3 explicitly unchanged: 'No significant differences in the abundances of Cul1, 2, 3, 4, "
              "and 5 were observed in whole kidney homogenates' across 0% / 1% / 5% K+ diets. What changes is "
              "the neddylated/total CUL3 ratio (lower on 0% K+), which this workflow cannot measure. Recorded "
              "as 'no change' (a cited prediction of no difference) at the authors' request, 28 Oct 2026.",
         needs_user_confirmation=False),
    dict(item="Kcnj10", item_type="protein", panel_group="Kir4.1 / Kir5.1", expected_direction=NONE,
         direction_set_by=LIT, sources=["wang2018", "wu2019"],
         note="Low K+ activates the basolateral Kir4.1/5.1 channel and hyperpolarizes the DCT; the papers "
              "report channel activity and membrane voltage, not channel protein abundance.",
         needs_user_confirmation=False),
    dict(item="Kcnj16", item_type="protein", panel_group="Kir4.1 / Kir5.1", expected_direction=NONE,
         direction_set_by=LIT, sources=["wang2018", "wu2019"],
         note="As Kcnj10: activity and voltage, no abundance data under low K+.",
         needs_user_confirmation=False),
    dict(item="Atp1a1", item_type="protein", panel_group="Na,K-ATPase", expected_direction=NONE,
         direction_set_by=LIT, sources=[],
         note="Older work measures Na,K-ATPase activity per nephron segment (no significant DCT change under "
              "K+ restriction), not alpha1 abundance. No protein-level low-K+ data found.",
         needs_user_confirmation=False),
    dict(item="Atp1b1", item_type="protein", panel_group="Na,K-ATPase", expected_direction=NONE,
         direction_set_by=LIT, sources=[], note="As Atp1a1: no low-K+ abundance data found.",
         needs_user_confirmation=False),
    dict(item="Fxyd2", item_type="protein", panel_group="Na,K-ATPase", expected_direction=NONE,
         direction_set_by=LIT, sources=[],
         note="No low-K+ abundance data found. Fxyd2 is also an ALDO_activity list member.",
         needs_user_confirmation=False),
    dict(item="Pvalb", item_type="protein", panel_group="DCT1 / Mg and Ca handling", expected_direction=DOWN,
         direction_set_by=LIT, sources=["murali2026"],
         note="Parvalbumin reduced at the protein level after 0% K+ for 2 weeks (kidney proteomics).",
         needs_user_confirmation=False),
    dict(item="Calb1", item_type="protein", panel_group="DCT1 / Mg and Ca handling", expected_direction=DOWN,
         direction_set_by=LIT, sources=["murali2026"],
         note="Calbindin-D28K reduced after 0% K+ for 2 weeks; mRNA also lower at 8 weeks on 0.1% K+ (protein "
              "not different at 8 weeks).",
         needs_user_confirmation=False),
    dict(item="Trpm6", item_type="protein", panel_group="DCT1 / Mg and Ca handling", expected_direction=NONE,
         direction_set_by=USER_CONFIRMED, sources=[],
         note="No study of potassium restriction on TRPM6 was found (TRPM6 up-regulation is reported for "
              "dietary Mg2+ restriction). Indirect only: TRPM6 is DCT1-restricted, EGF-dependent, and reduced "
              "when NCC is lost (NCC-knockout and Gitelman models). Detected in 13.2% of Ctrl and 1.4% of "
              "ZeroK cells (intake_core): at 1.4% only detection is testable, not intensity among detected cells.",
         needs_user_confirmation=False),
    dict(item="Egf", item_type="protein", panel_group="DCT1 / Mg and Ca handling", expected_direction=NONE,
         direction_set_by=USER_CONFIRMED, sources=[],
         note="No study of potassium restriction on renal EGF was found. Context only: EGF is produced in the "
              "DCT and stimulates TRPM6; Egf is reduced in a Bartter (Clcnkb-deficient) model, a different "
              "perturbation.",
         needs_user_confirmation=False),
    dict(item="Slc8a1", item_type="protein", panel_group="DCT2 / Ca extrusion", expected_direction=DOWN,
         direction_set_by=LIT, sources=["murali2026"],
         note="NCX1 protein reduced after 0% K+ for 2 weeks; mRNA lower on 0.1% K+.",
         needs_user_confirmation=False),
    dict(item="Atp2b1", item_type="protein", panel_group="DCT2 / Ca extrusion", expected_direction=NONE,
         direction_set_by=LIT, sources=["murali2026"],
         note="PMCA1 not reported under low K+ (PMCA4 mRNA lower at 8 weeks, protein unchanged).",
         needs_user_confirmation=False),
    dict(item="Mki67", item_type="protein", panel_group="Proliferation", expected_direction=NONE,
         direction_set_by=LIT, sources=["saritas2018"],
         note="Low K+ (3 days) increased the NUMBER of proliferating DCT1 cells (BrdU, Ki-67 counts). That is a "
              "cell-count result; it does not predict Mki67 abundance per cell. Detected in 3.0% of Ctrl and "
              "7.9% of ZeroK cells (intake_core).",
         needs_user_confirmation=False),
    dict(item="Top2a", item_type="protein", panel_group="Proliferation", expected_direction=NONE,
         direction_set_by=LIT, sources=["saritas2018"],
         note="As Mki67; not measured in the source. Detected in 3.2% of Ctrl and 5.6% of ZeroK cells (intake_core).",
         needs_user_confirmation=False),
    dict(item="Pcna", item_type="protein", panel_group="Proliferation", expected_direction=NONE,
         direction_set_by=LIT, sources=["saritas2018"], note="As Mki67; not measured in the source.",
         needs_user_confirmation=False),
    dict(item="Mcm2", item_type="protein", panel_group="Proliferation", expected_direction=NONE,
         direction_set_by=LIT, sources=["saritas2018"], note="As Mki67; not measured in the source.",
         needs_user_confirmation=False),
    # Scores (lists from reviewer_core_lib, unchanged; zerok_response_core step 2 scores them per cell).
    dict(item="ALDO_activity (base)", item_type="score", panel_group="ALDO score",
         expected_direction=NONE, direction_set_by=USER_CONFIRMED, sources=["castaneda2014", "murillo2022"],
         note="23 of 45 members in the joint object; best detected are Atp1a1 (98% Ctrl / 99% ZeroK), Fxyd2 "
              "(98/99), Acss3 (92/92), Slc25a35 (88/87) and Slc12a3 (88/87). " + ALDO_WHY,
         needs_user_confirmation=False),
    dict(item="ALDO_activity (no_Slc12a3)", item_type="score", panel_group="ALDO score",
         expected_direction=NONE, direction_set_by=USER_CONFIRMED, sources=["castaneda2014", "murillo2022"],
         note="As ALDO_activity (base) with Slc12a3 removed (the project notes: ALDO scores are tested with and without "
              "Slc12a3); 22 of 44 members in the joint object. Removing Slc12a3 takes out the one member "
              "expected up. " + ALDO_WHY,
         needs_user_confirmation=False),
    dict(item="ALDO_signed (base)", item_type="score", panel_group="ALDO score",
         expected_direction=NONE, direction_set_by=USER_CONFIRMED, sources=["castaneda2014", "murillo2022"],
         note="Up-list minus down-list, but only 2 of the 15 down-list members are in the joint object (Pdk4 "
              "28/39, Nedd4l 12/6), so the score is close to ALDO_activity. " + ALDO_WHY,
         needs_user_confirmation=False),
    dict(item="ALDO_signed (no_Slc12a3)", item_type="score", panel_group="ALDO score",
         expected_direction=NONE, direction_set_by=USER_CONFIRMED, sources=["castaneda2014", "murillo2022"],
         note="As ALDO_signed (base) without Slc12a3. " + ALDO_WHY,
         needs_user_confirmation=False),
    dict(item="ALDO_machinery", item_type="score", panel_group="ALDO score",
         expected_direction=NONE, direction_set_by=USER_CONFIRMED, sources=["ishizawa2016", "murillo2022"],
         note="Receptor and chaperone machinery; 11 of 15 members in the joint object, dominated by scaffolds "
              "and chaperones (Ywhaz 99/100, Ywhae 97/98, Hsp90ab1 96/98, Ywhab 96/96, Hsp90aa1 90/92), with "
              "Wnk4 at only 4/23 and Hsd11b2 at 9/10. Nr3c2 is not in the joint object. The list contains no "
              "Slc12a3, so base and no_Slc12a3 are identical. " + ALDO_WHY,
         needs_user_confirmation=False),
    dict(item="cell_cycle_score (18-protein panel)", item_type="score", panel_group="Proliferation",
         expected_direction=NONE, direction_set_by=LIT, sources=["saritas2018"],
         note="Included because zerok_response_core step 2 scores it. As for the individual cycling proteins, the published "
              "low-K+ effect is the number of proliferating cells, not the score per cell.",
         needs_user_confirmation=False),
    # Limits (plan step 2: stated, not tested).
    dict(item="NCC phosphorylation (pThr44/48/53)", item_type="limit", panel_group="Not measurable",
         expected_direction="", direction_set_by="", sources=["castaneda2014", "ishizawa2016"],
         note="The best-established low-K+ response in the DCT. Cannot be measured: no phospho-enrichment in "
              "this workflow, so phosphosites are not quantified.",
         needs_user_confirmation=False),
    dict(item="SPAK/OSR1 phosphorylation (pSer383 / pSer325)", item_type="limit", panel_group="Not measurable",
         expected_direction="", direction_set_by="", sources=["castaneda2014", "wade2015"],
         note="Increased by low K+ in the literature; not measurable here (no phospho-enrichment).",
         needs_user_confirmation=False),
    dict(item="KLHL3 phosphorylation (pSer433)", item_type="limit", panel_group="Not measurable",
         expected_direction="", direction_set_by="", sources=["ishizawa2016"],
         note="Increased by K+ depletion; not measurable here (no phospho-enrichment). Total Klhl3 is tested.",
         needs_user_confirmation=False),
    dict(item="Kcnj1 (ROMK)", item_type="limit", panel_group="Absent from both matrices",
         expected_direction="", direction_set_by="", sources=[],
         note="Absent from both Spectronaut exports (checked in this step against the intake_core protein mapping).",
         needs_user_confirmation=False),
    dict(item="Scnn1a / Scnn1b / Scnn1g (ENaC)", item_type="limit", panel_group="Absent from both matrices",
         expected_direction="", direction_set_by="", sources=[],
         note="All three ENaC subunits are absent from both exports (intake_core marker coverage; re-checked here).",
         needs_user_confirmation=False),
]

# Genes behind each entry: protein rows use the item name; limit rows list what was checked.
PANEL_GENES = {entry["item"]: ([entry["item"]] if entry["item_type"] == "protein" else [])
               for entry in PANEL}
PANEL_GENES["Kcnj1 (ROMK)"] = ["Kcnj1"]
PANEL_GENES["Scnn1a / Scnn1b / Scnn1g (ENaC)"] = ["Scnn1a", "Scnn1b", "Scnn1g"]

# Score lists (reviewer_core_lib, unchanged).
ALDO_LISTS = {
    "ALDO_activity (base)": (t2.ALDO_UP_WITH_SLC12A3, []),
    "ALDO_activity (no_Slc12a3)": ([g for g in t2.ALDO_UP_WITH_SLC12A3 if g != "Slc12a3"], []),
    "ALDO_signed (base)": (t2.ALDO_UP_WITH_SLC12A3, list(t2.ALDO_SIGNATURE_DOWN)),
    "ALDO_signed (no_Slc12a3)": ([g for g in t2.ALDO_UP_WITH_SLC12A3 if g != "Slc12a3"],
                                 list(t2.ALDO_SIGNATURE_DOWN)),
    "ALDO_machinery": (t2.FIG3_ALDO_MACHINERY, []),
}
CELL_CYCLE_LIST = {"cell_cycle_score (18-protein panel)": (list(t2.CELL_CYCLE_MARKERS), [])}
SCORE_LISTS = {**ALDO_LISTS, **CELL_CYCLE_LIST}

# The panel as accepted by the authors on 28 Oct 2026 (plan step 2), for a completeness check.
ACCEPTED_PANEL_PROTEINS = ["Slc12a3", "Wnk1", "Wnk4", "Stk39", "Oxsr1", "Klhl3", "Cul3", "Kcnj10", "Kcnj16",
                           "Atp1a1", "Atp1b1", "Fxyd2", "Pvalb", "Calb1", "Trpm6", "Egf", "Slc8a1", "Atp2b1",
                           "Mki67", "Top2a", "Pcna", "Mcm2"]

COVERAGE_CAVEAT = ("Every source measures whole-kidney or cortex homogenates, or counts of proliferating cells; "
                   "none measures single DCT cells. Directions are therefore expectations for the DCT "
                   "population, not predictions for per-cell SCP-MS values.")


# ---------------------------------------------------------------------------
# Coverage from the intake_core protein mapping
# ---------------------------------------------------------------------------
def read_t17_mapping(run: cc.Run) -> pd.DataFrame:
    """intake_core protein mapping (one row per protein group of each search) with booleans restored."""
    path = cc.out_path(t17.TASK, "tables", "protein_mapping", ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run intake_core_matching.py first.")
    frame = pd.read_csv(run.use_input(path, "intake_core protein mapping (both searches)"), keep_default_na=False,
                        na_values={"n_cells_ctrl": [""], "n_cells_zerok": [""]})
    for col in ("retained_ctrl", "retained_zerok", "in_joint", "gene_multi_group"):
        if col in frame:
            frame[col] = frame[col].astype(str).str.lower().eq("true")
    return frame


def retained_cells(run: cc.Run) -> dict[str, int]:
    """Retained cells per condition from the intake_core per-cell table (Ctrl 727, ZeroK 693)."""
    path = cc.out_path(t17.TASK, "tables", "dct_cells", ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run intake_core_matching.py first.")
    cells = pd.read_csv(run.use_input(path, "intake_core per-cell table (both conditions)"))
    keep = cells["retained"].astype(str).str.lower().eq("true")
    return {cond: int((keep & (cells["condition"] == cond)).sum()) for cond in CONDITIONS}


def gene_coverage(mapping: pd.DataFrame, gene: str, n_cells: dict[str, int]) -> dict:
    """Coverage of one gene symbol (cleaned first PG.Genes symbol, the intake_core rule) in both searches.

    When several groups share the first symbol, the one detected in most retained cells is reported.
    """
    out: dict[str, object] = {}
    joint_names, levels = [], []
    for cond in CONDITIONS:
        k = FILE_KEY[cond]
        hits = mapping.loc[(mapping["search"] == cond) & (mapping["gene_name"] == gene)]
        out[f"{k}_in_export"] = not hits.empty
        out[f"{k}_n_groups"] = int(len(hits))
        if hits.empty:
            out.update({f"{k}_protein_group": "", f"{k}_var_name": "", f"{k}_retained": False,
                        f"{k}_n_cells_detected": 0, f"{k}_fraction_cells": 0.0})
            continue
        best = hits.sort_values(f"n_cells_{k}", ascending=False).iloc[0]
        n = int(best[f"n_cells_{k}"])
        out.update({f"{k}_protein_group": str(best["protein_group"]), f"{k}_var_name": str(best["var_name"]),
                    f"{k}_retained": bool(best[f"retained_{k}"]), f"{k}_n_cells_detected": n,
                    f"{k}_fraction_cells": n / n_cells[cond]})
        joint_names.extend(hits.loc[hits["in_joint"], "joint_var_name"].astype(str).tolist())
        levels.extend(hits.loc[hits["in_joint"], "match_level"].astype(str).tolist())
        out[f"{k}_match_status"] = ";".join(sorted(set(hits["match_status"].astype(str))))
    out["in_joint"] = bool(joint_names)
    out["joint_var_name"] = ";".join(dict.fromkeys(joint_names))
    out["match_level"] = ";".join(dict.fromkeys(levels))
    return out


def list_coverage(mapping: pd.DataFrame, genes: list[str], n_cells: dict[str, int]) -> dict:
    """Coverage of a score list: how many of its genes are present, retained and matched."""
    per_gene = {gene: gene_coverage(mapping, gene, n_cells) for gene in dict.fromkeys(genes)}
    out = {"n_proteins_in_list": len(per_gene)}
    for cond in CONDITIONS:
        k = FILE_KEY[cond]
        out[f"{k}_n_present"] = sum(1 for c in per_gene.values() if c[f"{k}_in_export"])
        out[f"{k}_n_retained"] = sum(1 for c in per_gene.values() if c[f"{k}_retained"])
    out["n_in_joint"] = sum(1 for c in per_gene.values() if c["in_joint"])
    out["absent_both"] = ";".join(g for g, c in per_gene.items()
                                  if not c["ctrl_in_export"] and not c["zerok_in_export"])
    out["present_in_joint"] = ";".join(g for g, c in per_gene.items() if c["in_joint"])
    return out


# ---------------------------------------------------------------------------
# Step 3: normalization variants (zerok_response_core reports all three; no winner is chosen)
# ---------------------------------------------------------------------------
# (a) is the baseline. (b) reuses preprocessing_lib.normalize('median') unchanged. (c) is the authors' default C1
# (28 Oct 2026): control is the reference and only ZeroK detected values move.
VARIANTS = {
    "none": "ln(1 + intensity); missing = 0 (as the submitted pipelines and preprocessing_lib 'none')",
    "cell_median": ("per-cell median-centring of ln(intensity) over detected values: ln(x) - cell median + "
                    "median of the cell medians, then one constant added to every detected value so that the "
                    "smallest is 0.01 (preprocessing_lib.normalize('median'), the rule the authors approved on "
                    "13 Oct 2026); missing stays 0. Applied to both conditions together"),
    "condition_align": ("ln(1 + intensity), then every DETECTED ZeroK value shifted by delta; control is the "
                        "reference and is unchanged. delta = median over matched proteins of "
                        "[median ln(1 + intensity) over detected Ctrl cells - the same over detected ZeroK "
                        "cells], using proteins with at least 5 detected cells in each condition "
                        "(authors default C1, 28 Oct 2026); missing stays 0"),
}
BASELINE_VARIANT = "none"
ALIGN_MIN_CELLS = 5          # a protein needs this many detected cells per condition to contribute to delta
MIN_DETECTED_AFTER = 0.01    # as preprocessing_lib.MEDIAN_MIN_DETECTED, if a shift would push values to <= 0

N_PCS = 20                   # the submitted graph's setting (labels_core)
K_NEIGHBORS = 10
N_PERMUTATIONS = 1000


def normalize_joint(intensity, condition, method: str):
    """Normalized log matrix (cells x proteins, float64) and diagnostics, for one variant.

    ``intensity`` is the raw joint matrix (missing = 0) and ``condition`` the per-cell condition.
    """
    import numpy as np

    import preprocessing_lib as t4

    x = np.asarray(intensity, dtype=np.float64)
    if method in ("none", "cell_median"):
        out, info = t4.normalize(x, "none" if method == "none" else "median")
        info["method"] = method
        info["definition"] = VARIANTS[method]
        return out, info

    if method != "condition_align":
        raise ValueError(f"unknown variant {method!r}")

    condition = np.asarray(condition).astype(str)
    is_zerok = condition == "ZeroK"
    detected = x > 0
    ln = np.log1p(x)
    med = {}
    for cond, mask in (("Ctrl", ~is_zerok), ("ZeroK", is_zerok)):
        vals = np.where(detected[mask], ln[mask], np.nan)
        with np.errstate(invalid="ignore"):
            med[cond] = np.nanmedian(vals, axis=0)
        med[f"n_{cond}"] = detected[mask].sum(axis=0)
    usable = (med["n_Ctrl"] >= ALIGN_MIN_CELLS) & (med["n_ZeroK"] >= ALIGN_MIN_CELLS)
    per_protein = med["Ctrl"][usable] - med["ZeroK"][usable]
    delta = float(np.median(per_protein))
    out = ln.copy()
    shifted = detected & is_zerok[:, None]
    out[shifted] = ln[shifted] + delta
    info = {"method": method, "definition": VARIANTS[method], "delta_ln": delta,
            "n_proteins_used_for_delta": int(usable.sum()), "n_proteins": int(x.shape[1]),
            "align_min_cells_per_condition": ALIGN_MIN_CELLS,
            "per_protein_diff_q25": float(np.quantile(per_protein, 0.25)),
            "per_protein_diff_q75": float(np.quantile(per_protein, 0.75)),
            "global_median_diff_ln": float(np.nanmedian(np.where(detected[~is_zerok], ln[~is_zerok], np.nan))
                                           - np.nanmedian(np.where(detected[is_zerok], ln[is_zerok], np.nan))),
            "min_detected_before": float(ln[detected].min()),
            "min_detected_after_shift": float(out[detected].min())}
    # Guard (as the preprocessing median rule): a detected value must not fall to <= 0, where it would merge with missing.
    extra = max(0.0, MIN_DETECTED_AFTER - info["min_detected_after_shift"])
    if extra > 0:
        out[detected] = out[detected] + extra
    out[~detected] = 0.0
    info.update(uniform_shift_applied=extra, min_detected_after=float(out[detected].min()),
                n_detected_nonpositive_after=int((out[detected] <= 0).sum()))
    return out, info


def variant_table(infos: dict[str, dict]) -> "pd.DataFrame":
    rows = []
    for name, info in infos.items():
        row = {"variant": name, "is_baseline": name == BASELINE_VARIANT, "definition": VARIANTS[name]}
        row.update({k: v for k, v in info.items() if k not in ("method", "definition")})
        rows.append(row)
    return pd.DataFrame(rows)


def joint_embedding(norm_log, cell_ids, var_names, n_pcs: int = N_PCS, k: int = K_NEIGHBORS):
    """Per-protein z-score (clipped at +/-10) over ALL cells, arpack PCA, kNN graph and UMAP (seed 42).

    Explicit arpack PCA (preprocessing_lib), not the Scanpy fallback the submitted pipeline uses; step 4's
    pipeline route keeps the submitted settings so that it reproduces labels_core on the control cells.
    """
    import preprocessing_lib as t4

    g = t4.build_embedding(norm_log, np.ones(len(var_names), bool), "z10", cell_ids, var_names)
    h = t4.build_graph(g, n_pcs=n_pcs, k=k)
    return h, t4.umap(h)


def mixing(embedding, groups, order: list[str], k: int = K_NEIGHBORS, n_perm: int = N_PERMUTATIONS):
    """Same-group fraction among the k nearest neighbours against a label-permutation null."""
    import preprocessing_lib as t4

    idx = t4.neighbor_index(np.asarray(embedding), k)
    codes = pd.Categorical(np.asarray(groups).astype(str), categories=order).codes.astype(int)
    return t4.knn_purity(idx, codes, order, n_perm=n_perm)


def write_h5ad(run: cc.Run, adata, name: str) -> Path:
    """As ``cc.Run.h5ad`` but gzip-compressed (several float layers over 1,420 x 5,130)."""
    path = run.out("h5ad", name, ".h5ad")
    adata.write_h5ad(path, compression="gzip")
    print(f"Saved AnnData: {cc.rel(path)}")
    return path


def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the zerok_core step that writes it first.")
    return pd.read_csv(run.use_input(path, "zerok_core output"), **kwargs)
