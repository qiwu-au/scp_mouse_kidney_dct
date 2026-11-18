"""dct_reviewer shared definitions and functions. Runs no analysis by itself.

Imported by the
``dct_reviewer_*.py`` step scripts. Plan: ``the analysis plan``
(frozen; SHA256 in ``results/tables/dct_intake_zerok_vs_control_frozen_briefs_sha256.csv``), with the rules
in ``the analysis plan``.

dct_reviewer repeats every ctrl_reviewer_core analysis within each DCT condition of the combined Spectronaut search,
under the dct_labels labels, and sets the results side by side. No test between conditions (zerok_vs_control).
ZeroK is a second pool on another diet, not a replicate; plates P1/P2 are sort plates of one
pooled sample per condition (technical only).

Label columns (plan):
- primary: ``cs_ctrl_deg_label`` (Ctrl) and ``cs_zerok_pipeline_label`` (ZeroK);
- second: ``cs_ctrl_cluster_label`` and ``cs_zerok_cluster_label``;
- per-cell ``cs_<cond>_cell_label`` only for the DPT root rule and inside section D.

How the ctrl_reviewer_core code is reused
--------------------------
The ctrl_reviewer_core step functions read their label columns from the module name ``lib``
(``ctrl_reviewer_core_lib``: ``LABEL_COLUMNS``, ``PRIMARY``, ``LABEL_DISPLAY``, ...). ``bound`` points that
name, inside the ctrl_reviewer_core step modules, at one condition's ``Settings`` while the functions run, and
restores it afterwards. The ctrl_reviewer_core files are not edited. ``Settings`` has no fallback to the ctrl_reviewer_core
values: a name it does not define raises ``AttributeError``.

Copied here (they hard-code a ``ctrl_`` column or a intake_core/labels_core path, or fail on an absent label);
the logic is unchanged unless stated:
- ``ctrl_reviewer_core_cell_cycle.run_dpt``: ``ROOT_COLUMN = "ctrl_cell_label"`` -> ``settings.CELL_COLUMN``.
- ``ctrl_reviewer_core_reviewer_panel.ambiguity``: divides by the label size, so an absent core label
  (ZeroK primary: DCT2, ProLIF) raised ZeroDivisionError; absent labels now give 0 cells and NaN.
- ``ctrl_reviewer_core_lib.load_ctrl`` -> ``load`` (dct_labels object and canonical table).

Part 2 (steps 9-13) adds the qc_mito/preprocessing names of ``ctrl_reviewer_core_lib`` that the ctrl_reviewer_core Part 2 step functions
read from ``lib``: ``part2_settings`` (one condition's ``Settings`` plus ``qc_mito_DATASETS``,
``preprocessing_DATASETS``, ``SHORT_COLUMN``, ...), ``SHORT_ROLE``, ``preprocessing_MIN_LABEL_N``, ``t03_label_colors`` and
``macro_core``; ``protein_meta`` replaces ``ctrl_reviewer_core_lib.ctrl_protein_meta`` (dct_intake protein table)
and ``rebuild_raw`` replaces the intake_core rebuild of ``ctrl_reviewer_core_lib`` / ``ctrl_reviewer_core_mito_tests``
(combined export with the intake_core filters); ``check_rows_frame`` and ``stop_if_failed`` write the
check tables and stop a step on a failed check.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scp_common as cc
from reviewer_core_lib import CORE_LABELS, LABEL_PALETTE, SEED as REVIEWER_CORE_SEED

TASK = "dct_reviewer"
SEED = cc.RANDOM_SEED
CONDITIONS = ("Ctrl", "ZeroK")
KEY = {"Ctrl": "ctrl", "ZeroK": "zerok"}
ROLES = ("primary", "second")
ROLE_DISPLAY = {"primary": "primary label", "second": "second label (submitted cluster label)"}
LABEL_ORDER = cc.DCT_LABEL_ORDER
DESIGN_NOTE = ("One pool of 3 mice per condition; ZeroK is a second pool on another diet, not a replicate; "
               "plates P1/P2 are sort plates of the pooled cells (technical label, never a replicate).")
ONE_SEARCH_NOTE = ("Combined Spectronaut search of both conditions; each condition filtered and scaled on its own "
                   "(dct_intake); intensities carry the cross-run normalization over all 1,535 runs.")
NOT_APPLICABLE = "does not apply"

DCT_LABELS_TASK = "dct_labels"
DCT_LABELS_LABELS = cc.out_path(DCT_LABELS_TASK, "tables", "dct_canonical_labels", ".csv")
DCT_LABELS_H5AD = {c: cc.out_path(DCT_LABELS_TASK, "h5ad", f"DCT_{KEY[c]}_labelled", ".h5ad") for c in CONDITIONS}
DCT_LABELS_DPT_ROOTS = cc.out_path(DCT_LABELS_TASK, "tables", "submitted_dpt_roots", ".csv")
DCT_LABELS_MARKER_VARIANTS = cc.out_path(DCT_LABELS_TASK, "tables", "stability_marker_variants", ".csv")
CTRL_REVIEWER_CORE_TASK = "ctrl_reviewer_core"

# the project notes "DCT labels, combined search" (dct_labels; applied 2 Nov 2026).
EXPECTED_COUNTS = {
    "Ctrl": {"cs_ctrl_deg_label": {"DCT1": 382, "DCT2": 85, "Unknown": 262},
             "cs_ctrl_cluster_label": {"DCT1": 460, "DCT2": 126, "ProLIF": 119, "Unknown": 24},
             "cs_ctrl_cell_label": {"DCT1": 350, "DCT2": 207, "ProLIF": 172}},
    "ZeroK": {"cs_zerok_pipeline_label": {"DCT1": 449, "Unknown": 253},
              "cs_zerok_cluster_label": {"DCT1": 469, "DCT2": 114, "ProLIF": 98, "Unknown": 21},
              "cs_zerok_cell_label": {"DCT1": 321, "DCT2": 212, "ProLIF": 169}},
}
N_CELLS = {"Ctrl": 729, "ZeroK": 702}


# ---------------------------------------------------------------------------
# Per-condition settings (the names the ctrl_reviewer_core step functions read from ``lib``)
# ---------------------------------------------------------------------------
class Settings(SimpleNamespace):
    """One condition's label columns and display names; no fallback to ctrl_reviewer_core values."""

    def __getattr__(self, name):   # only called when the attribute is missing
        raise AttributeError(f"dct_reviewer Settings ({self.__dict__.get('CONDITION')}) has no {name!r}; "
                             "the ctrl_reviewer_core function needs a name dct_reviewer does not define.")

    def subset(self, columns) -> "Settings":
        """Same settings with LABEL_COLUMNS restricted to ``columns`` (kept in role order)."""
        out = copy.copy(self)
        out.LABEL_COLUMNS = tuple(c for c in self.LABEL_COLUMNS if c in set(columns))
        return out

    def role(self, column: str) -> str:
        return self.ROLE_OF.get(column, "per_cell" if column == self.CELL_COLUMN else column)


def _settings(cond: str) -> Settings:
    p = f"cs_{KEY[cond]}"
    stem = "deg" if cond == "Ctrl" else "pipeline"
    primary, second, cell = f"{p}_{stem}_label", f"{p}_cluster_label", f"{p}_cell_label"
    display = {primary: "DE-rule label (primary)", second: "submitted cluster label", cell: "submitted per-cell label"}
    sources = {
        primary: "dct_labels: cluster_de DE rule + kidney_annotation rank-free rescue on Leiden 1.0 clusters of the submitted graph",
        second: "dct_labels: submitted pipeline, Leiden 3.0 cluster mode with the original Unknown rule",
        cell: "dct_labels: submitted pipeline, per-cell argmax of the DCT V1 workbook scores",
    }
    return Settings(
        TASK=TASK, CONDITION=cond, KEY=KEY[cond], PREFIX=p,
        PRIMARY=primary, SECOND=second, CELL_COLUMN=cell, LABEL_COLUMNS=(primary, second),
        ROLE_OF={primary: "primary", second: "second"}, COLUMN_OF={"primary": primary, "second": second},
        PRIMARY_CLUSTER=f"{p}_{stem}_cluster", LEIDEN3_CLUSTER=f"{p}_leiden3_cluster",
        CONFIDENCE=f"{p}_{stem}_confidence",
        STORED_CLUSTERS={primary: (f"{p}_{stem}_cluster", 1.0), second: (f"{p}_leiden3_cluster", 3.0)},
        LABEL_DISPLAY=display, LABEL_SOURCES=sources, EXPECTED_COUNTS=EXPECTED_COUNTS[cond],
        LABEL_ORDER=LABEL_ORDER, DESIGN_NOTE=DESIGN_NOTE,
    )


SETTINGS = {c: _settings(c) for c in CONDITIONS}


@contextmanager
def bound(settings: Settings, *modules):
    """Point the ``lib`` name inside ctrl_reviewer_core step modules at ``settings`` while their functions run."""
    old = [m.lib for m in modules]
    try:
        for m in modules:
            m.lib = settings
        yield settings
    finally:
        for m, o in zip(modules, old):
            m.lib = o


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def read_canonical(run: cc.Run) -> pd.DataFrame:
    dtype = {"cell_id": str, "cell_key": str}
    for cond in CONDITIONS:
        s = SETTINGS[cond]
        dtype.update({s.PRIMARY_CLUSTER: str, s.LEIDEN3_CLUSTER: str})
    return pd.read_csv(run.use_input(DCT_LABELS_LABELS, "dct_labels canonical DCT labels (combined search)"), dtype=dtype,
                       keep_default_na=False, na_values=[""])


def load(run: cc.Run, cond: str):
    """dct_labels labelled object and canonical labels of one condition, matched on ``cell_id`` (replaces ``load_ctrl``).

    Checks: same cells in the same order in the CSV and the h5ad; equal plate, cell_key, depth,
    label, cluster and confidence columns; label counts equal the project notes. ``labels`` is indexed by
    ``cell_id`` in object order and gains ``n_detected_proteins`` (= ``n_detected``, the reviewer_core name).
    """
    import anndata as ad

    s = SETTINGS[cond]
    canon = read_canonical(run)
    labels = canon[canon["condition"] == cond].set_index("cell_id")
    adata = ad.read_h5ad(run.use_input(DCT_LABELS_H5AD[cond], f"dct_labels labelled object ({cond})"))
    if list(adata.obs_names.astype(str)) != list(labels.index):
        raise ValueError(f"{cond}: dct_labels canonical CSV and h5ad cell order differ.")
    compared = ("plate", "cell_key", "n_detected", s.CELL_COLUMN, s.SECOND, s.LEIDEN3_CLUSTER, s.PRIMARY,
                s.PRIMARY_CLUSTER, s.CONFIDENCE)
    for col in compared:
        a = adata.obs[col].astype(str).to_numpy()
        b = labels[col].astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"{cond}: column {col} differs between the dct_labels canonical CSV and the h5ad.")
    for col, expected in EXPECTED_COUNTS[cond].items():
        counts = labels[col].astype(str).value_counts().to_dict()
        if counts != expected:
            raise ValueError(f"{cond}: {col} counts {counts} differ from the project notes {expected}.")
    if len(labels) != N_CELLS[cond]:
        raise ValueError(f"{cond}: {len(labels)} cells, expected {N_CELLS[cond]}.")
    labels = labels.copy()
    labels.index.name = "cell_id"
    labels["n_detected_proteins"] = labels["n_detected"].astype(int)
    return adata, labels


# ---------------------------------------------------------------------------
# Paths and outputs
# ---------------------------------------------------------------------------
def h5ad_path(name: str) -> Path:
    return cc.out_path(TASK, "h5ad", name, ".h5ad")


def table_path(name: str, subdir: str = "") -> Path:
    return cc.out_path(TASK, "tables", name, ".csv", subdir)


def read_table(run: cc.Run, name: str, subdir: str = "", **kwargs) -> pd.DataFrame:
    path = table_path(name, subdir)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the dct_reviewer step that writes it first.")
    return pd.read_csv(run.use_input(path, "dct_reviewer output"), **kwargs)


def read_h5ad(run: cc.Run, name: str):
    import anndata as ad

    path = h5ad_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the dct_reviewer step that writes it first.")
    return ad.read_h5ad(run.use_input(path, "dct_reviewer AnnData"))


def ported_from(run: cc.Run, *scripts: str) -> None:
    """Record the ctrl_reviewer_core scripts a step ports or calls (reference code)."""
    for name in scripts:
        run.use_input(cc.SCRIPTS_DIR / name, "ctrl_reviewer_core code: functions called per condition or ported")


def tag(frame: pd.DataFrame, cond: str) -> pd.DataFrame:
    """Insert ``condition`` first and, when the frame has ``label_column``, ``label_role`` beside it."""
    out = frame.copy()
    if "condition" in out.columns:
        out = out.drop(columns="condition")
    out.insert(0, "condition", cond)
    if "label_column" in out.columns and "label_role" not in out.columns:
        s = SETTINGS[cond]
        out.insert(int(out.columns.get_loc("label_column")), "label_role", out["label_column"].map(s.role))
    return out


def stack(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tag and concatenate per-condition frames (Ctrl first)."""
    return pd.concat([tag(frames[c], c) for c in CONDITIONS if c in frames], ignore_index=True)


def present(labels: pd.DataFrame, column: str, label: str) -> bool:
    return bool((labels[column].astype(str) == label).any())


def columns_with(labels: pd.DataFrame, settings: Settings, *needed: str) -> list[str]:
    """Label columns of a condition in which every ``needed`` label has at least one cell."""
    return [c for c in settings.LABEL_COLUMNS if all(present(labels, c, n) for n in needed)]


def na_rows(cond: str, columns, analysis: str, reason: str, **extra) -> pd.DataFrame:
    """'does not apply' rows for label columns in which an analysis needs an absent label."""
    s = SETTINGS[cond]
    return pd.DataFrame([{"condition": cond, "label_role": s.role(c), "label_column": c, "analysis": analysis,
                          "status": NOT_APPLICABLE, "reason": reason, **extra} for c in columns])


# Per-cell tables hold both conditions, so the condition-specific column names are written
# under generic names (the README lists the source column per condition).
GENERIC = {"PRIMARY": "primary_label", "SECOND": "second_label", "CELL_COLUMN": "cell_label",
           "PRIMARY_CLUSTER": "primary_cluster", "LEIDEN3_CLUSTER": "leiden3_cluster",
           "CONFIDENCE": "primary_confidence"}


def generic_map(cond: str) -> dict[str, str]:
    s = SETTINGS[cond]
    return {getattr(s, k): v for k, v in GENERIC.items()}


def to_generic(frame: pd.DataFrame, cond: str) -> pd.DataFrame:
    return frame.rename(columns=generic_map(cond))


def from_generic(frame: pd.DataFrame, cond: str) -> pd.DataFrame:
    return frame.rename(columns={v: k for k, v in generic_map(cond).items()})


def counts_text(values, order=LABEL_ORDER) -> str:
    counts = pd.Series(np.asarray(values).astype(str)).value_counts()
    return "; ".join(f"{k} {int(counts[k])}" for k in cc.ordered_labels(counts.index, order))


# ---------------------------------------------------------------------------
# Figures (dotplot format through the dct_labels helpers: Arial >= 7 pt, PDF + 600 dpi PNG + LZW TIFF,
# exact width, text overlap and file checks)
# ---------------------------------------------------------------------------
MM = 1 / 25.4
FULL_MM, HALF_MM = 180.0, 85.0
COND_COLORS = {"Ctrl": "#4a4a4a", "ZeroK": "#b5651d"}


def style() -> None:
    """dct_labels figure style (labels_core style, Arial 7 pt). Call after importing the ctrl_reviewer_core step modules (they reset it)."""
    import dct_labels_figures as t27f

    t27f.arial()


def new_figure(width_mm: float, height_mm: float, **kwargs):
    import matplotlib.pyplot as plt

    return plt.figure(figsize=(width_mm * MM, height_mm * MM), **kwargs)


class Figures:
    """Saves figures with the dct_labels/dotplot checks and collects one check row per figure."""

    def __init__(self, run: cc.Run, step: str):
        self.run, self.step, self.rows = run, step, []

    def save(self, fig, name: str, description: str, width_mm: float = FULL_MM, subdir: str = "",
             label_role: str = "") -> dict:
        import dotplot_dotplots as t6d
        import dct_labels_figures as t27f

        shim = SimpleNamespace(fig=fig, texts=t27f.drawn_texts(fig))
        tc = t6d.text_checks(shim)
        w, h = fig.get_size_inches() / MM
        paths = self.run.figure(fig, name, description, subdir=subdir, formats=t27f.FORMATS, tight=False)
        fc = t6d.file_checks(paths, w, h)
        ok = (tc["min_font_pt"] >= 7 and tc["n_text_overlaps"] == 0 and tc["n_text_outside_figure"] == 0
              and "arial" in tc["fonts"] and fc["width_ok"] and fc.get("pdf_truetype_embedded", False)
              and abs(w - width_mm) < 0.05)
        row = {"step": self.step, "figure": name, "subdir": subdir, "label_role": label_role,
               "width_mm": round(w, 2), "height_mm": round(h, 2), **tc, **fc, "pass": ok}
        self.rows.append(row)
        if not ok:
            print(f"FIGURE CHECK FAILED: {name}: {tc['overlaps']} | outside: {tc['outside']} | "
                  f"min font {tc['min_font_pt']} | fonts {tc['fonts']} | width_ok {fc['width_ok']}")
        return row

    def finish(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows)
        self.run.table(frame, f"figure_checks_{self.step}")
        return frame

    def all_pass(self) -> bool:
        return bool(self.rows) and all(r["pass"] for r in self.rows)


def na_axis(ax, text: str) -> None:
    """Blank axis with a 'does not apply' note (absent label under this column)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, fontsize=7, color="#52514e",
            wrap=True)


def label_colors(labels) -> dict[str, str]:
    return {label: LABEL_PALETTE.get(label, "#898781") for label in labels}


def union_order(*value_sets, order=LABEL_ORDER) -> list[str]:
    """Labels present in any of the value sets, in label order (shared x axis for Ctrl and ZeroK)."""
    seen = set()
    for values in value_sets:
        seen |= set(pd.Series(values).astype(str))
    return cc.ordered_labels(seen, order)


def violins(ax, groups: dict[str, np.ndarray], order: list[str], width: float = 0.75) -> None:
    """Violin (n >= 2) plus median dot and IQR bar per label; absent labels leave an empty slot."""
    colors = label_colors(order)
    for i, lab in enumerate(order):
        v = np.asarray(groups.get(lab, []), dtype=float)
        v = v[np.isfinite(v)]
        if v.size >= 2 and np.nanstd(v) > 0:
            parts = ax.violinplot([v], positions=[i], widths=width, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(colors[lab])
                body.set_edgecolor("white")
                body.set_alpha(0.85)
        if v.size:
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.vlines(i, q1, q3, color="#0b0b0b", linewidth=1.0)
            ax.scatter([i], [med], color="white", edgecolor="#0b0b0b", s=8, zorder=3, linewidths=0.6)
    ax.set_xlim(-0.6, len(order) - 0.4)


def count_ticks(ax, order: list[str], counts: dict[str, int], rotation: int = 90) -> None:
    ax.set_xticks(range(len(order)), [f"{lab} ({counts.get(lab, 0)})" for lab in order], rotation=rotation)


def grid_y(ax) -> None:
    ax.yaxis.grid(True, color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Copied from ctrl_reviewer_core_cell_cycle.run_dpt (ROOT_COLUMN -> settings.CELL_COLUMN; otherwise unchanged)
# ---------------------------------------------------------------------------
def run_dpt(settings: Settings, objects: dict, labels: pd.DataFrame, n_random: int):
    import scanpy as sc

    dct1 = np.flatnonzero(labels[settings.CELL_COLUMN].to_numpy() == "DCT1")
    original = int(dct1[0])
    rng = np.random.default_rng(REVIEWER_CORE_SEED)
    random_roots = rng.choice(dct1[1:], size=min(n_random, len(dct1) - 1), replace=False)
    roots = [("original", original)] + [(f"random_{i + 1:02d}", int(r)) for i, r in enumerate(random_roots)]
    rows, per_cell = [], pd.DataFrame(index=labels.index)
    for name, obj in objects.items():
        sc.tl.diffmap(obj)
        for root_name, root in roots:
            obj.uns["iroot"] = root
            sc.tl.dpt(obj)
            pt = obj.obs["dpt_pseudotime"].to_numpy(float).copy()
            if root_name == "original":
                per_cell[f"{name}__dpt_original_root"] = pt
            finite = np.isfinite(pt)
            for col in settings.LABEL_COLUMNS:
                lab = labels[col].to_numpy()
                med = {l: float(np.median(pt[finite & (lab == l)])) if (finite & (lab == l)).any() else np.nan
                       for l in CORE_LABELS}
                lo, hi = sorted((med["DCT1"], med["DCT2"]))
                rows.append({"object": name, "root": root_name, "root_index": root, "root_cell_id": labels.index[root],
                             "label_column": col, "median_DCT1": med["DCT1"], "median_DCT2": med["DCT2"],
                             "median_ProLIF": med["ProLIF"], "prolif_between_dct1_dct2": bool(lo < med["ProLIF"] < hi),
                             "n_infinite": int((~finite).sum())})
        obj.uns["iroot"] = original
        sc.tl.dpt(obj)
    detail = pd.DataFrame(rows)
    summary = []
    for (name, col), sub in detail.groupby(["object", "label_column"], sort=False):
        orig = sub[sub["root"] == "original"].iloc[0]
        rnd = sub[sub["root"] != "original"]
        summary.append({"object": name, "label_column": col, "original_root_cell_id": orig["root_cell_id"],
                        "original_median_DCT1": orig["median_DCT1"], "original_median_DCT2": orig["median_DCT2"],
                        "original_median_ProLIF": orig["median_ProLIF"],
                        "original_prolif_between": orig["prolif_between_dct1_dct2"],
                        "n_random_roots": len(rnd), "n_random_prolif_between": int(rnd["prolif_between_dct1_dct2"].sum()),
                        "fraction_random_prolif_between": rnd["prolif_between_dct1_dct2"].mean()})
    return detail, pd.DataFrame(summary), per_cell


def mark_dpt_applicability(frame: pd.DataFrame, med_cols=("DCT1", "DCT2", "ProLIF")) -> pd.DataFrame:
    """Add ``prolif_between_applicable``: False where DCT1, DCT2 or ProLIF has no cell under the column.

    The copied ``run_dpt`` returns False for 'ProLIF between' when a median is NaN; dct_reviewer reports
    those rows as 'does not apply' rather than as a negative result.
    """
    out = frame.copy()
    prefix = "original_median_" if "original_median_DCT1" in out.columns else "median_"
    ok = np.ones(len(out), dtype=bool)
    for lab in med_cols:
        ok &= np.isfinite(out[f"{prefix}{lab}"].to_numpy(float))
    out["prolif_between_applicable"] = ok
    return out


# ---------------------------------------------------------------------------
# Copied from ctrl_reviewer_core_reviewer_panel.ambiguity (absent core labels give 0 cells instead of an error)
# ---------------------------------------------------------------------------
def ambiguity(task3: pd.DataFrame, col: str, order=CORE_LABELS) -> pd.DataFrame:
    rows = []
    for label in order:
        sub = task3[task3[col] == label]
        flagged, total = int(sub["ambiguous_tal_cnt_interstitial"].astype(bool).sum()), len(sub)
        pct = (lambda n: 100 * n / total) if total else (lambda n: np.nan)
        rows += [{"label": label, "status": "Flagged", "n_cells": flagged, "percent_cells": pct(flagged)},
                 {"label": label, "status": "Not flagged", "n_cells": total - flagged,
                  "percent_cells": pct(total - flagged)}]
    return pd.DataFrame(rows)


def write_h5ad_light(run: cc.Run, adata, name: str) -> Path:
    """``reviewer_core_lib.write_h5ad_light`` (intensity layers dropped; .X kept).

    In anndata 0.13 ``layers[None]`` is an alias of ``.X``: deleting it deletes ``.X``, so it is skipped
    (as in reviewer_core). Raises if ``.X`` is missing after the copy.
    """
    out = adata.copy()
    for key in [k for k in list(out.layers.keys()) if k is not None]:
        del out.layers[key]
    if out.X is None or out.X.shape != adata.shape:
        raise ValueError(f"{name}: .X missing before writing.")
    out.raw = None
    return run.h5ad(out, name)


# ---------------------------------------------------------------------------
# Part 2: the ctrl_reviewer_core_lib names (qc_mito / preprocessing parts) that the ctrl_reviewer_core Part 2 step functions read from ``lib``
# ---------------------------------------------------------------------------
SHORT_ROLE = {"primary": "primary", "second": "second"}
PREPROCESSING_MIN_LABEL_N = 10


def t03_label_colors(column: str) -> dict[str, str]:
    """``ctrl_reviewer_core_lib.label_colors``: plate palette or the reviewer_core label palette."""
    return dict(cc.PLATE_PALETTE) if column == "plate" else dict(LABEL_PALETTE)


def macro_core(frame: pd.DataFrame, value: str, min_n: int = PREPROCESSING_MIN_LABEL_N) -> float:
    """``ctrl_reviewer_core_lib.macro_core``: mean over DCT1, DCT2 and ProLIF with >= min_n cells (Unknown left out)."""
    sub = frame[frame["label"].isin(CORE_LABELS) & (frame["n_cells"] >= min_n)]
    return float(sub[value].mean()) if len(sub) else np.nan


def part2_settings(cond: str, columns=None) -> Settings:
    """One condition's settings plus the qc_mito/preprocessing names of ``ctrl_reviewer_core_lib`` (optionally restricted to ``columns``).

    Short column names are generic (``primary``, ``second``, ``cell``, ``plate``) so per-cell flag columns of the
    two conditions share names.
    """
    base = SETTINGS[cond]
    s = base.subset(columns) if columns is not None else copy.copy(base)
    s.SHORT_COLUMN = {base.PRIMARY: "primary", base.SECOND: "second", base.CELL_COLUMN: "cell", "plate": "plate"}
    s.QC_MITO_RUN = ("dct",)
    s.QC_MITO_DATASETS = {"dct": {"display": f"DCT {cond}", "label_columns": s.LABEL_COLUMNS,
                              "plot_columns": (*s.LABEL_COLUMNS, "plate"), "order": LABEL_ORDER,
                              "reference_label": "DCT1"}}
    s.QC_MITO_LABEL_DISPLAY = {**base.LABEL_DISPLAY, "plate": "plate (pooled sort plate)"}
    s.label_colors = t03_label_colors
    s.CORE_LABELS = CORE_LABELS
    s.PREPROCESSING_DATASETS = {"dct": {"display": f"DCT {cond}", "label_columns": s.LABEL_COLUMNS, "primary": (base.PRIMARY,),
                              "order": LABEL_ORDER, "stored_leiden": base.LEIDEN3_CLUSTER, "submitted_resolution": 3.0}}
    s.macro_core = macro_core
    s.ANNOTATION_NAME = "protein_annotation"
    return s


def protein_meta(run: cc.Run, cond: str, var_names) -> pd.DataFrame:
    """Protein-group metadata of one condition's matrix from the dct_intake protein table (the dct_labels objects keep gene names).

    Replaces ``ctrl_reviewer_core_lib.ctrl_protein_meta`` (intake_core object). Returns the ``ctrl_reviewer_core_lib.mito_meta`` layout
    (pg_genes, protein_group, first_symbol, protein_description) indexed by protein name in object order.
    """
    import ctrl_reviewer_core_lib as l19
    import dct_intake_lib as t26

    prot = t26.read_proteins(run)
    prot = prot[prot[f"retained_{KEY[cond]}"]].sort_values("file_row")
    if list(prot["var_name"].astype(str)) != list(map(str, var_names)):
        raise ValueError(f"{cond}: dct_intake retained proteins differ from the dct_labels object proteins or their order.")
    return l19.mito_meta(prot.set_index("var_name"), prot["var_name"])


def rebuild_raw(run: cc.Run, conditions=CONDITIONS) -> dict:
    """Each condition of the combined export with the intake_core filters (``dct_intake_lib.read_combined`` + intake_core rules).

    Replaces ``ctrl_reviewer_core_lib.rebuild_ctrl_intensity`` and the intake_core reader of ``ctrl_reviewer_core_mito_tests.rebuild_ctrl``.
    Per condition: raw (all cells x all protein rows), filt, names (all rows), retained intensity frame.
    """
    import intake_core_lib as t17
    import dct_intake_lib as t26

    comb = t26.read_combined(run)
    out = {}
    for cond in conditions:
        raw = comb.raw(cond)
        filt = t17.apply_filters(raw)
        names = raw.meta["var_name"].astype(str).to_numpy()
        cells = raw.cells.index.astype(str)
        retained = pd.DataFrame(t17.intensity_matrix(raw, filt), index=pd.Index(cells[filt.keep_cells], name="cell_id"),
                                columns=names[filt.keep_proteins])
        out[cond] = SimpleNamespace(raw=raw, filt=filt, names=names, retained=retained)
    return out


def check_rows_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["condition", "check", "value", "pass"])


def stop_if_failed(run: cc.Run, checks: pd.DataFrame, name: str) -> None:
    run.table(checks, name)
    if not checks["pass"].astype(bool).all():
        run.finish(status="stopped_failed_checks")
        raise SystemExit(f"A check failed; see dct_reviewer_{name}.csv.")


__all__ = ["LABEL_PALETTE", "CORE_LABELS"]
