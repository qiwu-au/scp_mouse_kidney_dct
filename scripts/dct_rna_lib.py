"""dct_rna shared definitions and functions. Runs no analysis by itself.

Imported by the ``dct_rna_*.py`` step scripts.
Plan: ``the analysis plan`` (frozen; SHA256 in
``results/tables/dct_intake_zerok_vs_control_frozen_briefs_sha256.csv``), with the rules in
``the analysis plan``. Reviewer points: Editors 1 (scRNA comparator for DCT);
R1 ProLIF prevalence (RNA cycling fraction).

dct_rna repeats ctrl_rna_core (the RNA comparator for the DCT-enriched SCP-MS data against the Mouse Kidney Atlas) on the
combined Spectronaut search, for Ctrl and for ZeroK separately, set side by side (Ctrl first). The atlas side of
atlas_dct is reused unchanged: the DCT-region subset, the atlas_dct-derived DCT1 / DCT2 split and the atlas_dct atlas cycling tables.
ZeroK rows are descriptive: the atlas mice were not on a zero-K diet and come from another study, so ZeroK is
unpaired with the atlas in study, protocol and diet. No test between conditions is made (zerok_vs_control).

Label columns (plan; dct_labels canonical table)
------------------------------------------
- primary: ``cs_ctrl_deg_label`` (Ctrl: DCT1 382, DCT2 85, Unknown 262) and ``cs_zerok_pipeline_label``
  (ZeroK: DCT1 449, Unknown 253);
- second: ``cs_ctrl_cluster_label`` and ``cs_zerok_cluster_label`` (DCT1, DCT2, ProLIF, Unknown).
The primary columns have no ProLIF and the ZeroK primary has no DCT2 (dct_labels). Rows that need an absent label say
"does not apply"; the second column's result is reported beside them. Resolution, rule and markers are not changed.

How the ctrl_rna_core / atlas_dct code is reused
--------------------------------
- Imported unchanged: ``atlas_dct_lib`` and the helper functions of the atlas_dct step scripts (as ctrl_rna_core did);
  ``ctrl_rna_core_cell_mapping.class_fractions`` and ``ctrl_rna_core_dotplot.plotted``; ``labels_core_figures.panel_for``;
  ``kidney_rna_dotplots.build_paired`` / ``GreyPanel``; ``dct_reviewer_lib.load`` (dct_labels objects and canonical labels).
- ``class_fractions`` reads ``L.SCP_GROUPS`` from the module name ``L`` inside ``ctrl_rna_core_cell_mapping``;
  ``bound`` points that name at one condition's ``Settings`` while it runs (the dct_reviewer pattern) and restores it.
- Ported with a condition argument (as ctrl_rna_core ported atlas_dct): the ctrl_rna_core ``main()`` and figure functions, which hard-code
  the intake_core / labels_core / ctrl_reviewer_core paths, the 727-cell / 5,644-gene checks and a one-condition, two-column layout. The frozen
  parameters are those of the ctrl_rna_core step manifests.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np
import pandas as pd

import scp_common as cc
import cluster_de_lib as t5
import atlas_dct_lib as T
import dct_reviewer_lib as L28

TASK = "dct_rna"
SEED = cc.RANDOM_SEED
DATASET = "dct_combined"
CONDITIONS = L28.CONDITIONS                          # ("Ctrl", "ZeroK"): Ctrl first
KEY = L28.KEY
ROLES = ("primary", "second")
NOT_APPLICABLE = L28.NOT_APPLICABLE                  # "does not apply"
SCP_GROUPS = cc.DCT_LABEL_ORDER                      # DCT1, DCT2, ProLIF, Unknown (Unknown its own row, as ctrl_rna_core)
CORE_GROUPS = ("DCT1", "DCT2", "ProLIF")
SCP_PALETTE = T.SCP_PALETTE                          # Fig 3 colours
N_CELLS = L28.N_CELLS                                # Ctrl 729, ZeroK 702
N_PROTEINS = {"Ctrl": 6078, "ZeroK": 5953}           # dct_intake retained proteins
EXPECTED_COUNTS = L28.EXPECTED_COUNTS
ROLE_TEXT = {"primary": "primary", "second": "second"}

CTRL_RNA_CORE_TASK = "ctrl_rna_core"
DCT_INTAKE_TASK = "dct_intake"
DCT_LABELS_TASK = L28.DCT_LABELS_TASK
DCT_REVIEWER_TASK = L28.TASK
ATLAS_DCT_TASK = T.TASK
KIDNEY_RNA_PROVENANCE = cc.OUT_ROOTS["tables"] / "kidney_rna" / "kidney_rna_atlas_provenance.md"
ATLAS_DCT_SPLIT_EXPECTED = {T.DCT1_ATLAS_DCT: 8864, T.DCT2_ATLAS_DCT: 2752, T.DCT_UNASSIGNED: 4765}   # atlas_dct README

DESIGN_NOTE = L28.DESIGN_NOTE
ONE_SEARCH_NOTE = L28.ONE_SEARCH_NOTE
ATLAS_DCT_SPLIT_CAVEAT = ("DCT1 (atlas_dct) / DCT2 (atlas_dct): atlas_dct-derived split of the atlas DCT cells, not atlas labels; the DCT1 (atlas_dct) "
                    "call is not Pvalb-driven and DCT2 (atlas_dct) holds 39 nuclei.")
UNPAIRED_NOTE = "RNA and protein come from different animals and studies (unpaired comparison)."
ZEROK_NOTE = ("ZeroK rows are descriptive: the atlas mice were on a normal diet in another study, so ZeroK is unpaired "
              "with the atlas in study, protocol and diet; no ZeroK row is read as a diet effect and no test between "
              "conditions is made (zerok_vs_control).")
FORMATS = (".pdf", ".png", ".tif")                    # dct_reviewer / dct_labels format (ctrl_rna_core wrote TIFF for the dotplots only)


# ---------------------------------------------------------------------------
# Per-condition settings: the dct_reviewer settings plus the names the ctrl_rna_core helpers read from ``L``
# ---------------------------------------------------------------------------
def _settings(cond: str) -> L28.Settings:
    s = copy.copy(L28.SETTINGS[cond])
    s.SCP_GROUPS = SCP_GROUPS
    s.CORE_GROUPS = CORE_GROUPS
    s.LABEL_SHORT = {s.PRIMARY: "primary", s.SECOND: "second"}
    s.SCP_PALETTE = SCP_PALETTE
    return s


SETTINGS = {c: _settings(c) for c in CONDITIONS}


@contextmanager
def bound(settings, *modules, name: str = "L"):
    """Point the module name ``name`` (ctrl_rna_core step modules: ``L``) at ``settings`` while their functions run."""
    old = [getattr(m, name) for m in modules]
    try:
        for m in modules:
            setattr(m, name, settings)
        yield settings
    finally:
        for m, o in zip(modules, old):
            setattr(m, name, o)


def role_of(cond: str, column: str) -> str:
    return SETTINGS[cond].ROLE_OF[column]


def column_of(cond: str, role: str) -> str:
    return SETTINGS[cond].COLUMN_OF[role]


def present(labels: pd.DataFrame, column: str, order=SCP_GROUPS) -> list[str]:
    """Labels with at least one cell under ``column``, in label order."""
    have = set(labels[column].astype(str))
    return [g for g in order if g in have]


def absent(labels: pd.DataFrame, column: str, order=SCP_GROUPS) -> list[str]:
    have = set(labels[column].astype(str))
    return [g for g in order if g not in have]


# ---------------------------------------------------------------------------
# Checks with a condition column
# ---------------------------------------------------------------------------
class Checks(t5.Checks):
    """``cluster_de_lib.Checks`` with a ``condition`` column ("both" for checks that do not depend on it)."""

    def add(self, check: str, observed, passed: bool | None, threshold: str = "", condition: str = "both") -> None:
        print(f"  ({condition})", end="")
        super().add(check, observed, passed, threshold)
        self.rows[-1] = {"condition": condition, **self.rows[-1]}


def stop_if_failed(run: cc.Run, checks: Checks, name: str) -> None:
    if not checks.ok:
        run.table(checks.frame(), name)
        run.finish({"checks_pass": False}, status="stopped_failed_checks")
        raise SystemExit(f"A check failed; see dct_rna_{name}.csv.")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_scp(run: cc.Run, cond: str, checks: Checks | None = None):
    """dct_labels labelled object and canonical labels of one condition (``dct_reviewer_lib.load``: same cells and order in
    the CSV and h5ad, equal label / cluster / plate / depth columns, the project notes counts); ln(1 + intensity) with
    missing = 0 and detection (intensity > 0). Replaces ``ctrl_rna_core_lib.load_scp``.
    """
    adata, labels = L28.load(run, cond)
    s = SETTINGS[cond]
    if checks is not None:
        checks.add("dct_labels object: cells x proteins", f"{adata.n_obs} x {adata.n_vars}",
                   adata.shape == (N_CELLS[cond], N_PROTEINS[cond]), f"{N_CELLS[cond]} x {N_PROTEINS[cond]}", cond)
        for col in s.LABEL_COLUMNS:
            observed = labels[col].astype(str).value_counts().to_dict()
            checks.add(f"{col} ({role_of(cond, col)}) counts equal the project notes",
                       "; ".join(f"{k} {observed[k]}" for k in SCP_GROUPS if k in observed),
                       observed == EXPECTED_COUNTS[cond][col], "", cond)
    names = pd.Index(adata.var_names.astype(str))
    inten = cc.dense(adata.layers["intensity"]).astype(np.float64)
    ln1p = pd.DataFrame(np.log1p(inten), index=labels.index, columns=names)
    det = pd.DataFrame(inten > 0, index=labels.index, columns=names)
    return adata, labels, ln1p, det


def read_table(run: cc.Run, name: str, task: str = TASK, **kwargs) -> pd.DataFrame:
    """A table written by a dct_rna step (or, with ``task``, by another revision task)."""
    path = cc.out_path(task, "tables", name, ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the step that writes it first.")
    return pd.read_csv(run.use_input(path, f"{task} output"), **kwargs)


def t28_per_cell(run: cc.Run, name: str, cond: str, labels: pd.DataFrame, checks: Checks) -> pd.DataFrame:
    """One condition's rows of a dct_reviewer per-cell table, reindexed to the label order; checks cells, labels and plate."""
    s = SETTINGS[cond]
    tab = read_table(run, name, DCT_REVIEWER_TASK, dtype={"cell_id": str, "cell_key": str}, keep_default_na=False,
                     na_values=[""])
    tab = tab[tab["condition"] == cond].set_index("cell_id")
    same_cells = set(tab.index) == set(labels.index) and len(tab) == len(labels)
    tab = tab.reindex(labels.index)
    pairs = [("primary_label", s.PRIMARY), ("second_label", s.SECOND), ("plate", "plate")]
    same = same_cells and all(tab[a].astype(str).equals(labels[b].astype(str)) for a, b in pairs if a in tab.columns)
    checks.add(f"dct_reviewer {name}: same {len(labels)} cells, primary / second labels and plate as the dct_labels canonical table",
               f"{same}", same, "", cond)
    return tab


def protein_groups(run: cc.Run, cond: str, var_names) -> pd.DataFrame:
    """PG.Genes of one condition's proteins from the dct_intake protein table (the dct_labels objects keep gene names only).

    Port of ``ctrl_rna_core_shared_features.protein_groups`` (which reads the intake_core control object).
    """
    import dct_intake_lib as t26

    prot = t26.read_proteins(run)
    prot = prot[prot[f"retained_{KEY[cond]}"]].sort_values("file_row")
    if list(prot["var_name"].astype(str)) != list(map(str, var_names)):
        raise ValueError(f"{cond}: dct_intake retained proteins differ from the dct_labels object proteins or their order.")
    pg = prot["pg_genes"].astype(str).str.strip().replace("nan", "")
    genes = pg.str.split(";").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    return pd.DataFrame({"protein_group": prot["protein_group"].astype(str).to_numpy(), "pg_genes": pg.to_numpy(),
                         "n_genes_in_group": genes.apply(lambda xs: len(dict.fromkeys(xs))).to_numpy()},
                        index=pd.Index(prot["var_name"].astype(str).to_numpy(), name="protein"))


def depth_quartiles(run: cc.Run, cond: str, labels: pd.DataFrame, checks: Checks) -> np.ndarray:
    """Protein-ID quartile edges of one condition, checked against the dct_intake filtering summary (retained cells and
    proteins). Replaces the labels_core edges of ctrl_rna_core."""
    q = np.quantile(labels["n_detected_proteins"].astype(float), [0, 0.25, 0.5, 0.75, 1.0])
    f = read_table(run, "filtering_summary", DCT_INTAKE_TASK)
    r = f[(f["condition"] == cond) & (f["plate"] == "all") & (f["stage"] == "retained_cells_retained_proteins")]
    ref = r[["protein_ids_min", "protein_ids_q25", "protein_ids_median", "protein_ids_q75", "protein_ids_max"]]
    ref = ref.to_numpy(float).ravel()
    checks.add("protein-ID quartile edges equal the dct_intake filtering summary (retained cells, retained proteins)",
               "; ".join(f"{v:g}" for v in q), len(r) == 1 and bool(np.allclose(q, ref)),
               "; ".join(f"{v:g}" for v in ref), cond)
    return q


QUARTILE_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]


def quartile_bins(values, edges) -> pd.Series:
    return pd.Series(pd.cut(np.asarray(values, float), list(edges), labels=QUARTILE_LABELS,
                            include_lowest=True).astype(str), index=getattr(values, "index", None))


def ported_from(run: cc.Run, *scripts: str) -> None:
    """Record the ctrl_rna_core / atlas_dct scripts a step ports or imports functions from (reference code; not run)."""
    for name in scripts:
        run.use_input(cc.SCRIPTS_DIR / name, "reference code: ctrl_rna_core / atlas_dct script (functions imported / ported; not run)")


def tag(frame: pd.DataFrame, cond: str) -> pd.DataFrame:
    """``dct_reviewer_lib.tag`` for the dct_rna settings: ``condition`` first, ``label_role`` beside ``label_column``."""
    out = frame.copy()
    if "condition" in out.columns:
        out = out.drop(columns="condition")
    out.insert(0, "condition", cond)
    if "label_column" in out.columns and "label_role" not in out.columns:
        roles = out["label_column"].map(lambda c: SETTINGS[cond].ROLE_OF.get(c, ""))
        out.insert(int(out.columns.get_loc("label_column")), "label_role", roles)
    return out


def stack(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tag and concatenate per-condition frames (Ctrl first)."""
    return pd.concat([tag(frames[c], c) for c in CONDITIONS if c in frames], ignore_index=True)


def counts_text(values, order=SCP_GROUPS) -> str:
    counts = pd.Series(np.asarray(values).astype(str)).value_counts()
    return "; ".join(f"{k} {int(counts[k])}" for k in order if k in counts.index)


# ---------------------------------------------------------------------------
# Figures (atlas_dct sheet and checks; PDF + 600 dpi PNG + LZW TIFF)
# ---------------------------------------------------------------------------
ABSENT_FACE = "#f0efec"


def save_checked(run: cc.Run, sheet, name: str, desc: str, rows: list[dict], **extra) -> bool:
    """``atlas_dct_lib.save_checked`` with TIFF added; the TIFF must be RGB with LZW compression."""
    T.save_checked(run, sheet, name, desc, rows, formats=FORMATS)
    row = rows[-1]
    tif_ok = row.get("tif_mode") == "RGB" and "lzw" in str(row.get("tif_compression", "")).lower()
    row["pass"] = bool(row["pass"] and tif_ok)
    row.update(extra)
    return row["pass"]


def fit_title(title: str, width: float | None = None) -> str:
    """Greedy word wrap of a figure title (atlas_dct title size, bold) to the sheet width."""
    width = (T.W - 2 * T.M) if width is None else width
    lines, cur = [], ""
    for word in title.split():
        trial = f"{cur} {word}".strip()
        if cur and T.text_mm(trial, T.TITLE, "bold")[0] > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    lines.append(cur)
    return "\n".join(lines)


def absent_cell(sh, x: float, y: float, w: float, h: float, text: str = "–") -> None:
    """Grey cell with a dash: label absent under this column ('does not apply')."""
    sh.rect(x, y, w, h, facecolor=ABSENT_FACE, edgecolor="white", linewidth=0.6)
    if text:
        sh.text(x + w / 2, y + h / 2, text, ha="center", va="center_baseline", color=T.INK["muted"])


def figure_rows_frame(rows: list[dict], step: str) -> pd.DataFrame:
    """Figure-check rows with ``condition`` first ("both" for figures that show both conditions) and ``step``."""
    out = pd.DataFrame(rows)
    cond = out.pop("condition") if "condition" in out.columns else "both"
    out.insert(0, "step", step)
    out.insert(0, "condition", cond)
    return out
