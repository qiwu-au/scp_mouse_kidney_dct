"""rat_segments shared definitions and functions. Runs no analysis by itself.

Imported by the ``rat_segments_*.py`` step scripts.
Task: Limbutara et al. 2020 (JASN; rat microdissected tubule segments; manuscript ref 10) as a protein-level
comparator for PT (S1 / S2 / S3) and TAL (cortical / medullary) heterogeneity, next to the pt_tal RNA-atlas analysis.

Decisions by the authors (30 Oct 2026, plan approval)
--------------------------------------------------
- Task number rat_segments. Source: the authors' Kidney Tubule Expression Atlas (KTEA) download files, per-sample copies per
  cell (primary) and geometric means (cross-check); ``data/reference/KTEA/SOURCE.md``.
- Rat -> mouse: identical symbols only (case-insensitive); no ortholog table. Unmatched proteins are listed.
- Selection rule fixed before any SCP result (constants below): quantified = value > 0 in >= 2/3 of the segment's
  samples; every segment of a contrast quantified; min(group segments) - max(reference segments) of the log2 means
  >= 1 (sensitivity 0.585, i.e. 1.5-fold); group mean >= 25th percentile of the segment's quantified means; Welch t
  on log2 sample values against each reference segment (the larger p), BH q < 0.05 within contrast; then the pt_tal
  SCP rule (protein detected in >= 10% of the segment's SCP cells under each label set), top 30 by log2 FC.
- Every score with and without V3 markers (and without ribosomal / MitoCarta proteins), as pt_tal.

Deviation decided by the authors (30 Oct 2026, after the planned rule gave empty panels; rat-side counts and SCP
detection only, no score computed): (1) log2 copies per cell are centred per sample on the sample median over the
proteins quantified in all 16 PT and TAL samples (S1 samples were about 1.4 log2 lower than S2 / S3 overall);
(2) the BH q < 0.05 step (Welch, 3 rats per segment: no protein passed in any contrast) is replaced by replicate
separation: every quantified group sample above every quantified reference sample. Welch p and BH q stay in the
tables as information. ``rat_segments_rule_diagnostics.csv`` records the panel sizes under each rule.

Contents
--------
- Paths, segments, contrasts, thresholds; pt_tal counterparts of each panel.
- ``read_samples`` / ``read_means``: the two KTEA files (0 -> not quantified).
- ``segment_stats``: per protein and segment, samples quantified, log2 mean of the quantified samples.
- ``rat_symbols`` / ``match_to_mouse``: identical-symbol matching to the kidney protein names.
- ``score_col``: per-cell score column names.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import scp_common as cc
import pt_tal_lib as T9

TASK = "rat_segments"
SEED = cc.RANDOM_SEED
DATASET = "kidney"

# External reference ---------------------------------------------------------------------------------------------
REF_DIR = cc.PROJECT_ROOT / "data" / "reference" / "KTEA"
SAMPLES_FILE = REF_DIR / "KTEA_proteome_individual_sample.txt"
MEANS_FILE = REF_DIR / "KTEA_proteome.txt"
SOURCE_FILE = REF_DIR / "SOURCE.md"
EXPECTED_SHA256 = {SAMPLES_FILE.name: "1fb4ef9d6d64c506fa9a4aa5e9bd8032e4786c09183bf81e0307c7a10c471922",
                   MEANS_FILE.name: "86359d2028e5fdd2572b490e7a3a000baff6247cb6b8885389ec845a034ca0c7"}
SOURCE_TEXT = ("Limbutara K, Chou CL, Knepper MA. J Am Soc Nephrol 2020;31(6):1255-1266 (PMID 32358040); Kidney Tubule "
               "Expression Atlas, https://esbl.nhlbi.nih.gov/KTEA/ (downloaded 2026-10-30)")
ID_COL, NAME_COL, GENE_COL = "Majority protein IDs", "Protein names", "gene"
SEGMENTS = ("S1", "S2", "S3", "DTL1", "DTL2", "DTL3", "ATL", "mTAL", "cTAL", "DCT", "CNT", "CCD", "OMCD", "IMCD")
EXPECTED_SAMPLES = {s: (4 if s in ("mTAL", "CCD") else 3) for s in SEGMENTS}
EXPECTED_N_PROTEINS = 7429
PT_SEGMENTS = ("S1", "S2", "S3")
TAL_SEGMENTS = ("cTAL", "mTAL")
STUDY_SEGMENTS = (*PT_SEGMENTS, *TAL_SEGMENTS)
SEGMENT_OF = {**{s: "PT" for s in PT_SEGMENTS}, **{s: "TAL" for s in TAL_SEGMENTS}}

# Selection rule (approved plan, 30 Oct 2026; fixed before any SCP result) -----------------------------------------
QUANT_FRAC = 2 / 3                      # value > 0 in >= 2/3 of the segment's samples (3 -> 2; 4 -> 3)
THRESHOLDS = {"fc2": 1.0, "fc1.5": float(np.log2(1.5))}   # min(group) - max(reference), log2
PRIMARY_THR = "fc2"
THR_TEXT = {"fc2": "log2 FC >= 1 (2-fold; primary)", "fc1.5": "log2 FC >= 0.585 (1.5-fold; sensitivity)"}
ABUND_PCTL = 0.25                       # group mean >= this quantile of the segment's quantified means
Q_MAX = 0.05                            # planned BH rule; information only after the 30 Oct deviation
CENTRING_TEXT = ("log2 copies per cell centred per sample on the sample median over the proteins quantified in all "
                 "16 PT and TAL samples (authors, 30 Oct 2026)")
SEPARATION_TEXT = "every quantified group sample above every quantified reference sample (authors, 30 Oct 2026)"
SCP_DET_MIN = T9.SCP_DET_MIN            # 0.10, protein detected in >= 10% of the segment's SCP cells
PANEL_TOP = T9.PANEL_TOP                # 30

# name -> (SCP segment, group segments, reference segments, kind)
CONTRASTS = {
    "S1": ("PT", ("S1",), ("S2", "S3"), "one_vs_rest"),
    "S2": ("PT", ("S2",), ("S1", "S3"), "one_vs_rest"),
    "S3": ("PT", ("S3",), ("S1", "S2"), "one_vs_rest"),
    "S1/S2": ("PT", ("S1", "S2"), ("S3",), "grouped"),
    "cTAL": ("TAL", ("cTAL",), ("mTAL",), "one_vs_rest"),
    "mTAL": ("TAL", ("mTAL",), ("cTAL",), "one_vs_rest"),
}
PT_PANELS = ("S1", "S2", "S3", "S1/S2")
TAL_PANELS = ("cTAL", "mTAL")
# "S3" (S3 vs S1 and S2) is both the one-vs-rest S3 panel and the grouped S3 side (Limbutara has no S3T2).
DIFFS = {"S1/S2 - S3": ("S1/S2", "S3"), "S1 - S3": ("S1", "S3"), "cTAL - mTAL": ("cTAL", "mTAL")}
PT_SCORES = (*PT_PANELS, "S1/S2 - S3", "S1 - S3")
TAL_SCORES = (*TAL_PANELS, "cTAL - mTAL")
MAP_CLASSES = {"PT": PT_SEGMENTS, "TAL": TAL_SEGMENTS}
# expected nearest segment per SCP PT label (as pt_tal: PCT -> S1 or S2; PST -> S3)
EXPECTED_MAP = {"PCT": ("S1", "S2"), "PST": ("S3",)}

# pt_tal counterparts: Limbutara panel -> pt_tal panels (overlap) and the pt_tal atlas contrast (direction)
PT_TAL_PANELS = {"S1": ("PTS1",), "S2": ("PTS2",), "S3": ("PTS3", "S3"), "S1/S2": ("S1/S2",),
              "cTAL": ("CTAL",), "mTAL": ("MTAL",)}
PT_TAL_CONTRAST = {"S1": "PTS1", "S2": "PTS2", "S3": "S3", "S1/S2": "S1/S2", "cTAL": "CTAL", "mTAL": "MTAL"}
PT_TAL_SCORE_REF = "S1/S2 - S3"            # pt_tal score shown next to the Limbutara scores

# Scoring (as pt_tal step 3) -------------------------------------------------------------------------------------------
METHODS = ("mean_z", "score_genes", "detected_fraction")
GENE_SETS = ("all", "no_v3", "no_ribo_mito")
GENE_SET_TEXT = {"all": "panel", "no_v3": "panel without V3 markers",
                 "no_ribo_mito": "panel without ribosomal / MitoCarta"}
N_BOOT = 1000
GROUPING_PANELS = {"t15": "kidney_t15_label", "pre": "kidney_pre_label", T9.CONSENSUS: "kidney_t15_label"}

# Pseudobulk -------------------------------------------------------------------------------------------------------
N_PERM = 1000
DEPTH_BINS = 10                         # labels permuted within ln(protein IDs) deciles
PT_RATIOS = {"S1 - S3": (("S1",), ("S3",)), "S2 - S3": (("S2",), ("S3",)), "S1 - S2": (("S1",), ("S2",)),
             "S1/S2 - S3": (("S1", "S2"), ("S3",))}
VARIABLE_MIN = 1.0                      # Limbutara-variable proteins: max pairwise |log2 difference| among S1-S3

INK = T9.INK
SEG_PALETTE = {"S1": "#2a78d6", "S2": "#6da7ec", "S3": "#1baf7a", "cTAL": "#eda100", "mTAL": "#b07800"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def read_table(run: cc.Run, name: str, task: str = TASK, **kwargs) -> pd.DataFrame:
    path = cc.out_path(task, "tables", name, ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the step that writes it first.")
    return pd.read_csv(run.use_input(path, f"{task} output"), **kwargs)


def safe(name: str) -> str:
    """File- and column-safe panel / score name (S1/S2 -> S1S2; 'a - b' -> a_minus_b)."""
    return name.replace("/", "").replace(" - ", "_minus_")


def score_col(s: str, thr: str, score: str, gs: str, method: str) -> str:
    return f"score__{s}__{thr.replace('.', 'p')}__{safe(score)}__{gs}__{method}"


# ---------------------------------------------------------------------------
# KTEA files
# ---------------------------------------------------------------------------
def check_sources(run: cc.Run) -> dict[str, str]:
    """Record both files and SOURCE.md as inputs; return observed SHA256 per file."""
    run.use_input(SOURCE_FILE, "rat_segments external reference provenance")
    out = {}
    for path in (SAMPLES_FILE, MEANS_FILE):
        run.use_input(path, "Limbutara et al. 2020 KTEA download (external, read-only)")
        out[path.name] = run.inputs[cc.rel(path)]["sha256"]
    return out


def read_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Per-sample copies per cell. Returns (annotation, values with 0 -> NaN, sample -> segment)."""
    raw = pd.read_csv(SAMPLES_FILE, sep="\t", dtype={ID_COL: str, NAME_COL: str, GENE_COL: str},
                      keep_default_na=False, na_values=[])
    ann = raw[[ID_COL, NAME_COL, GENE_COL]].copy()
    ann.index = ann[ID_COL].astype(str)
    vals = raw.drop(columns=[ID_COL, NAME_COL, GENE_COL]).apply(pd.to_numeric, errors="raise")
    vals.index = ann.index
    if not np.isfinite(vals.to_numpy()).all() or (vals < 0).any().any():
        raise ValueError(f"{SAMPLES_FILE.name}: non-finite or negative copies per cell")
    seg = pd.Series({c: c.rsplit("_", 1)[0] for c in vals.columns}, name="segment")
    return ann, vals.where(vals > 0), seg


def read_means() -> pd.DataFrame:
    raw = pd.read_csv(MEANS_FILE, sep="\t", dtype={ID_COL: str, NAME_COL: str, GENE_COL: str},
                      keep_default_na=False, na_values=[])
    raw.index = raw[ID_COL].astype(str)
    return raw


def centred_log2(vals: pd.DataFrame, seg: pd.Series) -> tuple[pd.DataFrame, pd.Series, int]:
    """log2 copies per cell minus each sample's median over the proteins quantified in all PT and TAL samples.
    Returns (centred log2, NaN = not quantified; sample medians; number of proteins the medians use)."""
    lg = np.log2(vals)
    study = seg.index[seg.isin(STUDY_SEGMENTS)]
    common = lg[study].notna().all(axis=1)
    med = lg.loc[common].median(axis=0, skipna=True)
    return lg - med, med, int(common.sum())


def segment_stats(vals: pd.DataFrame, seg: pd.Series, lg: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per protein and segment: n samples, n quantified, quantified flag (>= 2/3), log2 mean and SD of the
    quantified samples (columns ``<stat>_<segment>``); ``lg`` = log2 values to use (default log2 copies per cell)."""
    lg = np.log2(vals) if lg is None else lg
    cols = {}
    for s in SEGMENTS:
        sc_ = seg.index[seg == s]
        n = len(sc_)
        nq = vals[sc_].notna().sum(axis=1)
        cols[f"n_samples_{s}"] = pd.Series(n, index=vals.index)
        cols[f"n_quant_{s}"] = nq
        cols[f"quant_{s}"] = nq >= int(np.ceil(QUANT_FRAC * n - 1e-9))
        cols[f"log2_{s}"] = lg[sc_].mean(axis=1, skipna=True)
        cols[f"sd_log2_{s}"] = lg[sc_].std(axis=1, ddof=1, skipna=True)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Rat -> mouse (identical symbols, case-insensitive)
# ---------------------------------------------------------------------------
UNIPROT_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$")


def rat_symbols(gene: str) -> list[str]:
    """Gene symbols of one KTEA entry (split on ';', stripped); UniProt accessions in the gene column are dropped."""
    out = []
    for g in str(gene).split(";"):
        g = g.strip()
        if g and g.upper() != "NA" and not UNIPROT_RE.match(g) and g not in out:
            out.append(g)
    return out


def match_to_mouse(ann: pd.DataFrame, protein_names, pg_genes) -> pd.DataFrame:
    """One row per KTEA entry: rat symbols, mouse protein matched and how.

    Rule (authors, 30 Oct 2026): (1) a rat symbol equals a kidney protein name, case-insensitive (protein names with
    '_' = second protein groups of a symbol are not used, as kidney_rna); (2) otherwise a rat symbol equals a gene of a
    kidney protein group (``pg_genes``) whose protein was not matched in (1). An entry matching several mouse
    proteins keeps all (flag ``one_to_many``); several entries matching one mouse protein are resolved later by
    abundance (``resolve_many_to_one``).
    """
    names = [str(p) for p in protein_names]
    by_name = {p.lower(): p for p in names if "_" not in p and p != "Nan"}
    by_pg: dict[str, list[str]] = {}
    for p, genes in zip(names, map(str, pg_genes)):
        if "_" in p or p == "Nan":
            continue
        for g in genes.split(";"):
            g = g.strip().lower()
            if g and g != "nan":
                by_pg.setdefault(g, [])
                if p not in by_pg[g]:
                    by_pg[g].append(p)
    rows = []
    for rid, gene in zip(ann.index, ann[GENE_COL]):
        syms = rat_symbols(gene)
        hits = [by_name[s.lower()] for s in syms if s.lower() in by_name]
        how = "protein_name" if hits else ""
        if not hits:
            hits = [p for s in syms for p in by_pg.get(s.lower(), [])]
            how = "protein_group_gene" if hits else ""
        hits = list(dict.fromkeys(hits))
        reason = "" if hits else ("no gene symbol in KTEA (UniProt ID only)" if not syms
                                  else "no identical mouse symbol among the kidney proteins")
        if not hits:
            rows.append({"rat_id": rid, "rat_gene": gene, "rat_symbols": ";".join(syms), "mouse_protein": "",
                         "match": "unmatched", "n_mouse_hits": 0, "reason": reason})
        for p in hits:
            rows.append({"rat_id": rid, "rat_gene": gene, "rat_symbols": ";".join(syms), "mouse_protein": p,
                         "match": how, "n_mouse_hits": len(hits), "reason": ""})
    out = pd.DataFrame(rows)
    # protein-name matches take precedence: drop group-gene hits on proteins already matched by name
    named = set(out.loc[out["match"] == "protein_name", "mouse_protein"])
    drop = (out["match"] == "protein_group_gene") & out["mouse_protein"].isin(named)
    out.loc[drop, ["match", "reason"]] = ["dropped", "mouse protein already matched by protein name"]
    out.loc[drop, "mouse_protein"] = ""
    return out


def resolve_many_to_one(matches: pd.DataFrame, abundance: pd.Series) -> pd.DataFrame:
    """Keep, per mouse protein, the KTEA entry with the highest mean log2 abundance over the study segments
    (ties: first entry); the others are flagged ``dropped`` with the reason."""
    m = matches.copy()
    m["kept"] = m["mouse_protein"].ne("")
    m["many_to_one"] = False
    ok = m["kept"]
    m.loc[ok, "abund"] = m.loc[ok, "rat_id"].map(abundance).to_numpy()
    for p, idx in m[ok].groupby("mouse_protein").groups.items():
        if len(idx) < 2:
            continue
        m.loc[idx, "many_to_one"] = True
        best = m.loc[idx, "abund"].fillna(-np.inf).idxmax()
        for i in idx:
            if i != best:
                m.loc[i, ["kept"]] = False
                m.loc[i, "reason"] = f"another KTEA entry with higher abundance kept for {p}"
    return m.drop(columns="abund")
