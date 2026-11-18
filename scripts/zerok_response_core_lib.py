#!/usr/bin/env python
"""zerok_response_core shared settings and statistics: zero-K vs control on the DCT proteome.

Imported by the ``zerok_response_core_*.py``
step scripts; runs no analysis by itself. Plan: ``the analysis plan``.

Settings agreed with the authors (28 Oct 2026, plan defaults D1-D8), fixed before any
zero-K vs control comparison was run:

D1  Panel scoring. Primary test per protein: two-part model with log n_detected_matched as
    covariate: Firth-penalized logistic regression on detection plus OLS on ln-intensity among
    detected cells; each part tested by a likelihood-ratio test, the two LR statistics summed
    (2 df; 1 df when the intensity part is not estimable: < 5 detected cells in either
    condition). Scores: OLS score ~ condition + log n_detected (t test). BH within the panel
    per label column x label group x variant. Observed direction and match status: see
    ``observed_direction`` and ``match_status``.
D2  "all_DCT" = DCT1 + DCT2 + ProLIF (Unknown excluded); label-free "all_cells" in step 5;
    proteome-wide only for labels with >= 20 cells in each condition.
D3  Scores on the joint matrix per variant after sc.pp.scale(max_value=10) over all 1,420 cells:
    ALDO (reviewer_core_lib.aldo_scores), cell cycle (reviewer_core_lib.score, seed 42), DCT type
    scores (labels_core_lib.score_types) plus a deterministic mean-z check; cycling-score rule
    threshold = 90th percentile of the control cells.
D4  Depth matching: greedy 1:1 nearest neighbour on ln(n_detected_matched) within label group
    and sort plate, caliper 0.05 ln units (as ctrl_reviewer_core_depth_detection.nn_match), random
    order per repeat, 100 repeats.
D5  Pseudotime: mean labels_core DPT of the 10 nearest control cells in the joint 20-PC embedding
    (control cells leave themselves out).
D6  Pseudobulk: mean back-transformed intensity per cell (missing = 0) per condition x plate
    and pooled; log2 ratio ZeroK / Ctrl.
D7  No enrichment (would need a gene-set download).
D8  Robustness repeats: per-repeat p-values and BH for the panel only; proteome-wide effect
    sizes only.

Every p-value is descriptive: one pool of 3 mice per condition, so condition and pool cannot
be separated and cells are not animal-level replicates (``P_NOTE`` beside every test).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import intake_core_lib as t17
import labels_core_lib as t18
import zerok_core_lib as t21

TASK = "zerok_response_core"
SEED = cc.RANDOM_SEED
CONDITIONS = t17.CONDITIONS                   # ("Ctrl", "ZeroK"); ZeroK - Ctrl everywhere
PLATES = t17.PLATES
CONDITION_PALETTE = t17.CONDITION_PALETTE

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
PANEL_SHA256 = "c2fd78fc106d8e1334211b6a2b883aedd6bb606d34e3d10b2ee7533a5505e88a"   # zerok_core README / panel_hash
PANEL_CSV = t21.table_path("prespecified_panel")
PANEL_HASH_CSV = t21.table_path("panel_hash")
LABELS_CSV = t21.table_path("dct_joint_labels")
DESIGN_MD = cc.out_path(t21.TASK, "tables", "design", ".md")
JOINT_H5AD = cc.out_path(t21.TASK, "h5ad", "DCT_joint_labelled", ".h5ad")
LABELS_CORE_LABELS = cc.out_path(t18.TASK, "tables", "dct_ctrl_canonical_labels", ".csv")
LABELS_CORE_MARKER_ORDER = cc.out_path(t18.TASK, "tables", "marker_order_dct", ".csv")
INTAKE_CORE_MAPPING = cc.out_path(t17.TASK, "tables", "protein_mapping", ".csv")

# ---------------------------------------------------------------------------
# Variants, labels, groups
# ---------------------------------------------------------------------------
VARIANTS = tuple(t21.VARIANTS)                # none, cell_median, condition_align
BASELINE = t21.BASELINE_VARIANT               # "none"
VARIANT_LAYER = {"none": "ln1p_intensity", "cell_median": "norm__cell_median",
                 "condition_align": "norm__condition_align"}
LABEL_COLUMNS = ("dct_label", "dct_label_alternative")
LABEL_DISPLAY = {"dct_label": "dct_label (primary: Ctrl ctrl_deg_label + ZeroK zerok_pipeline_label)",
                 "dct_label_alternative": "alternative (Ctrl ctrl_deg_label + ZeroK zerok_transfer_label)",
                 "label_free": "label-free"}
LABEL_SHORT = {"dct_label": "primary", "dct_label_alternative": "alternative", "label_free": "label-free"}
CORE = ("DCT1", "DCT2", "ProLIF")
LABELS = (*CORE, "Unknown")
ALL_DCT = "all_DCT"
ALL_CELLS = "all_cells"
SLC12A3_DETECTED = "Slc12a3_detected"
GROUP_ORDER = (*LABELS, ALL_DCT, ALL_CELLS, SLC12A3_DETECTED)

DEPTH_COL = "n_detected_matched"
MIN_GROUP_PROTEOME = 20            # D2: proteome-wide for labels with >= 20 cells in each condition
MIN_DETECTED_FRACTION = 0.10       # plan step 3: detected in >= 10% of cells in at least one condition
MIN_DET_PER_CONDITION = 5          # D1: intensity part needs >= 5 detected cells in each condition
MIN_DET_TOTAL = 5                  # D1: < 5 detected cells in the group -> not testable
ALPHA = 0.05
CALIPER = 0.05                     # D4 (ctrl_reviewer_core)
N_REPEATS = 100
DOWNSAMPLE_FRACTIONS = (0.5, 0.25)
K_PSEUDOTIME = 10                  # D5
N_PCS = t21.N_PCS                  # 20
CYCLING_QUANTILE = 0.90            # D3: cycling-score rule (ctrl_reviewer_core A.4), threshold from control cells

P_NOTE = ("descriptive: one pool of 3 mice per condition (condition, pool and search cannot be separated); "
          "cells are not animal-level replicates")
DESIGN_NOTE = ("One pool of 3 mice per condition: condition and pool cannot be separated; cells are not "
               "animal-level replicates; P1/P2 are sort plates of the pooled cells (technical only); the two "
               "conditions were searched separately in Spectronaut.")

# Panel score items -> per-cell score columns (D3).
SCORE_ITEMS = {
    "ALDO_activity (base)": "ALDO_activity__base",
    "ALDO_activity (no_Slc12a3)": "ALDO_activity__no_Slc12a3",
    "ALDO_signed (base)": "ALDO_signed__base",
    "ALDO_signed (no_Slc12a3)": "ALDO_signed__no_Slc12a3",
    "ALDO_machinery": "ALDO_machinery__base",
    "cell_cycle_score (18-protein panel)": "cell_cycle_score",
}

# D1 values
UP, DOWN, NO_CHANGE, NO_EXPECTATION = t21.UP, t21.DOWN, t21.NO_CHANGE, t21.NONE
OBS_NS = "no BH-significant difference"
OBS_MIXED = "mixed"
NOT_TESTABLE = "not testable"
NOT_SCORED = "not scored (no expectation)"


# ---------------------------------------------------------------------------
# Paths and reading
# ---------------------------------------------------------------------------
def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the zerok_response_core step that writes it first.")
    return pd.read_csv(run.use_input(path, "zerok_response_core output"), **kwargs)


def check_panel_hash(run: cc.Run, checks: t5.Checks) -> None:
    """Plan: check the frozen panel's SHA256 first; stop if it differs."""
    observed = cc.sha256(run.use_input(PANEL_CSV, "zerok_core frozen pre-specified panel"))
    recorded = pd.read_csv(run.use_input(PANEL_HASH_CSV, "zerok_core panel hash"))["sha256"].iloc[0]
    ok = observed == PANEL_SHA256 == recorded
    checks.add("frozen panel SHA256 equals the zerok_core README and zerok_core_panel_hash.csv", observed, ok,
               PANEL_SHA256)
    if not ok:
        raise SystemExit(f"Panel SHA256 {observed} differs from {PANEL_SHA256}; stopping (zerok_response_core plan).")


def read_panel(run: cc.Run) -> pd.DataFrame:
    return pd.read_csv(run.use_input(PANEL_CSV, "zerok_core frozen pre-specified panel"), keep_default_na=False)


@dataclass
class Joint:
    """The zerok_core labelled joint object as plain arrays (cells x matched proteins)."""

    obs: pd.DataFrame                 # index cell_id
    var: pd.DataFrame                 # index joint var name
    detected: np.ndarray              # bool, intensity > 0
    mats: dict[str, np.ndarray]       # variant -> float64 normalized ln matrix (missing = 0)
    pca: dict[str, np.ndarray]        # variant -> first N_PCS joint PCs
    var_names: np.ndarray

    @property
    def is_zerok(self) -> np.ndarray:
        return self.obs["condition"].to_numpy() == "ZeroK"

    @property
    def log_depth(self) -> np.ndarray:
        return np.log(self.obs[DEPTH_COL].to_numpy(float))

    def var_index(self, name: str) -> int:
        return int(np.flatnonzero(self.var_names == name)[0])


def load_joint(run: cc.Run, checks: t5.Checks) -> Joint:
    """Labelled joint object + label CSV, with the input checks every step repeats."""
    import anndata as ad

    adata = ad.read_h5ad(run.use_input(JOINT_H5AD, "zerok_core labelled joint object (5,130 matched proteins)"))
    labels = pd.read_csv(run.use_input(LABELS_CSV, "zerok_core joint labels (dct_label)"), dtype={"cell_id": str})
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    for col in ("condition", "plate", "dct_label", "dct_label_alternative"):
        obs[col] = obs[col].astype(str)
    n_by = obs["condition"].value_counts().to_dict()
    checks.add("joint object cells (Ctrl / ZeroK)", f"{len(obs)} ({n_by.get('Ctrl', 0)} / {n_by.get('ZeroK', 0)})",
               len(obs) == 1420 and n_by.get("Ctrl") == 727 and n_by.get("ZeroK") == 693, "1420 (727 / 693)")
    levels = sorted(set(adata.var["match_level"].astype(str)))
    checks.add("joint proteins: matched levels 1 and 2 only", f"{adata.n_vars} proteins; levels {levels}",
               adata.n_vars == 5130 and set(levels) <= {"1", "2"}, "5130; levels 1, 2")
    same_order = labels["cell_id"].tolist() == obs.index.tolist()
    checks.add("label CSV in object cell order", f"{same_order}", same_order)
    for col in ("dct_label", "dct_label_alternative", "condition", "plate"):
        eq = int((labels[col].astype(str).to_numpy() == obs[col].to_numpy()).sum())
        checks.add(f"object obs {col} equals the zerok_core label CSV", f"{eq}/{len(obs)}", eq == len(obs))
    ctrl = obs["condition"] == "Ctrl"
    same_ctrl = int((obs.loc[ctrl, "dct_label"] == obs.loc[ctrl, "dct_label_alternative"]).sum())
    checks.add("control cells: alternative column equals dct_label (ctrl_deg_label)", f"{same_ctrl}/{int(ctrl.sum())}",
               same_ctrl == int(ctrl.sum()))

    intensity = cc.dense(adata.layers["intensity"]).astype(np.float64)
    detected = intensity > 0
    mats = {}
    for variant in VARIANTS:
        m = cc.dense(adata.layers[VARIANT_LAYER[variant]]).astype(np.float64)
        ok = bool((m[detected] > 0).all() and (m[~detected] == 0).all())
        checks.add(f"variant {variant}: detected > 0 and missing = 0", f"{ok}", ok)
        mats[variant] = m
    d_none = float(np.abs(mats["none"] - np.log1p(intensity)).max())
    checks.add("variant none equals ln(1 + intensity)", f"max abs diff {d_none:.2g}", d_none < 1e-4, "< 1e-4")
    n_det = detected.sum(axis=1)
    d_depth = int(np.abs(n_det - obs[DEPTH_COL].to_numpy(int)).max())
    checks.add(f"obs {DEPTH_COL} equals detected matched proteins per cell", f"max abs diff {d_depth}", d_depth == 0)
    pca = {v: np.asarray(adata.obsm[f"X_pca__{v}"], dtype=np.float64)[:, :N_PCS] for v in VARIANTS}
    var = adata.var.copy()
    var.index = var.index.astype(str)
    return Joint(obs=obs, var=var, detected=detected, mats=mats, pca=pca,
                 var_names=np.asarray(var.index.astype(str)))


def resolve_protein(joint: Joint, gene: str) -> str:
    """Joint var name for a gene symbol: exact var name, else the most detected var with that gene_name."""
    if gene in set(joint.var_names):
        return gene
    hits = joint.var.loc[joint.var["gene_name"].astype(str) == gene]
    if hits.empty:
        return ""
    return str(hits.sort_values("n_cells_detected", ascending=False).index[0])


def group_masks(obs: pd.DataFrame, label_column: str) -> dict[str, np.ndarray]:
    """Label groups of one label column (D2): each label, plus all_DCT (core labels, Unknown excluded)."""
    lab = obs[label_column].to_numpy()
    out = {g: lab == g for g in LABELS}
    out[ALL_DCT] = np.isin(lab, CORE)
    return out


def label_free_masks(joint: Joint) -> dict[str, np.ndarray]:
    slc = joint.var_index(resolve_protein(joint, "Slc12a3"))
    return {ALL_CELLS: np.ones(len(joint.obs), bool), SLC12A3_DETECTED: joint.detected[:, slc].copy()}


# ---------------------------------------------------------------------------
# Proportions
# ---------------------------------------------------------------------------
def wilson(k: int, n: int) -> tuple[float, float]:
    return t2.wilson_ci(int(k), int(n))


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p2 - p1 with the Newcombe hybrid score 95% interval (method 10; Wilson limits)."""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    return d, d - np.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + np.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def proportion_compare(k_c: int, n_c: int, k_z: int, n_z: int, prefix: str = "") -> dict:
    """Ctrl vs ZeroK proportion: fractions, Wilson CIs, difference (ZeroK - Ctrl) with CI, Fisher p."""
    d, lo, hi = newcombe_diff(k_c, n_c, k_z, n_z)
    lc, uc = wilson(k_c, n_c)
    lz, uz = wilson(k_z, n_z)
    p = stats.fisher_exact([[k_z, n_z - k_z], [k_c, n_c - k_c]])[1] if n_c and n_z else np.nan
    return {f"{prefix}n_ctrl": n_c, f"{prefix}k_ctrl": k_c, f"{prefix}fraction_ctrl": k_c / n_c if n_c else np.nan,
            f"{prefix}ci_low_ctrl": lc, f"{prefix}ci_high_ctrl": uc,
            f"{prefix}n_zerok": n_z, f"{prefix}k_zerok": k_z, f"{prefix}fraction_zerok": k_z / n_z if n_z else np.nan,
            f"{prefix}ci_low_zerok": lz, f"{prefix}ci_high_zerok": uz,
            f"{prefix}difference_zerok_minus_ctrl": d, f"{prefix}difference_ci_low": lo,
            f"{prefix}difference_ci_high": hi, f"{prefix}fisher_p": float(p)}


# ---------------------------------------------------------------------------
# Two-part model (D1)
# ---------------------------------------------------------------------------
def firth_logit(X: np.ndarray, y: np.ndarray, fixed_zero: int | None = None, max_iter: int = 200,
                tol: float = 1e-8):
    """Firth-penalized logistic regression (Heinze & Schemper 2002); finite estimates under separation.

    ``fixed_zero`` holds one coefficient at 0 while the penalty keeps the full design (the restricted fit of
    the penalized likelihood-ratio test, as logistf). Returns coefficients, Wald SEs from the penalized
    information, penalized log-likelihood, converged.
    """
    n, p = X.shape
    beta = np.zeros(p)
    free = np.ones(p, bool)
    if fixed_zero is not None:
        free[fixed_zero] = False

    def evaluate(b):
        eta = X @ b
        pi = expit(eta)
        w = pi * (1 - pi)
        info = (X * w[:, None]).T @ X
        sign, logdet = np.linalg.slogdet(info)
        ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
        return ll + 0.5 * logdet if sign > 0 else -np.inf, pi, w, info

    cur, pi, w, info = evaluate(beta)
    converged = False
    for _ in range(max_iter):
        inv = np.linalg.inv(info)
        h = w * np.einsum("ij,jk,ik->i", X, inv, X)
        score = X.T @ (y - pi + h * (0.5 - pi))
        step = np.zeros(p)
        step[free] = np.linalg.solve(info[np.ix_(free, free)], score[free])
        big = np.max(np.abs(step))
        if big > 5:
            step *= 5 / big
        new = beta + step
        val, pi_n, w_n, info_n = evaluate(new)
        halves = 0
        while val < cur - 1e-10 and halves < 40:
            step /= 2
            new = beta + step
            val, pi_n, w_n, info_n = evaluate(new)
            halves += 1
        beta, cur, pi, w, info = new, val, pi_n, w_n, info_n
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(info)), 0, None))
    return beta, se, cur, converged


def design(z: np.ndarray, log_depth: np.ndarray) -> np.ndarray:
    c = log_depth - log_depth.mean()
    return np.column_stack([np.ones(len(z)), z.astype(float), c])


def detection_part(det: np.ndarray, z: np.ndarray, log_depth: np.ndarray) -> dict:
    """Firth logistic detection ~ condition + log depth; LR test of condition (penalized likelihoods)."""
    n_c, n_z = int((~z).sum()), int(z.sum())
    k_c, k_z = int(det[~z].sum()), int(det[z].sum())
    out = {"det_ctrl": k_c, "det_zerok": k_z, "n_ctrl": n_c, "n_zerok": n_z,
           "detection_separation": bool(k_c in (0, n_c) or k_z in (0, n_z)),
           "det_logodds": np.nan, "det_logodds_se": np.nan, "det_lr": np.nan, "det_p": np.nan,
           "det_converged": False}
    if k_c + k_z < MIN_DET_TOTAL or n_c == 0 or n_z == 0:
        return out
    X = design(z, log_depth)
    y = det.astype(float)
    b, se, pll, conv = firth_logit(X, y)
    _, _, pll_red, conv_red = firth_logit(X, y, fixed_zero=1)
    lr = max(0.0, 2 * (pll - pll_red))
    out.update(det_logodds=float(b[1]), det_logodds_se=float(se[1]), det_lr=lr,
               det_p=float(stats.chi2.sf(lr, 1)), det_converged=bool(conv and conv_red))
    return out


def intensity_part(values: np.ndarray, z: np.ndarray, log_depth: np.ndarray) -> dict:
    """OLS ln-intensity ~ condition + log depth among detected cells; Gaussian LR test of condition."""
    out = {"int_coef_ln": np.nan, "int_coef_se": np.nan, "int_log2fc_adjusted": np.nan, "int_lr": np.nan,
           "int_p": np.nan, "intensity_part_estimable": False}
    if int((~z).sum()) < MIN_DET_PER_CONDITION or int(z.sum()) < MIN_DET_PER_CONDITION:
        return out
    X = design(z, log_depth)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    rss = float(np.sum((values - X @ beta) ** 2))
    br, *_ = np.linalg.lstsq(X[:, [0, 2]], values, rcond=None)
    rss_r = float(np.sum((values - X[:, [0, 2]] @ br) ** 2))
    if rss <= 0 or rss_r <= 0:
        return out
    lr = max(0.0, n * np.log(rss_r / rss))
    sigma2 = rss / (n - p)
    se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
    out.update(int_coef_ln=float(beta[1]), int_coef_se=se, int_log2fc_adjusted=float(beta[1] / np.log(2)),
               int_lr=float(lr), int_p=float(stats.chi2.sf(lr, 1)), intensity_part_estimable=True)
    return out


def combine_parts(det: dict, inten: dict) -> dict:
    if not np.isfinite(det["det_lr"]):
        return {"two_part_chi2": np.nan, "two_part_df": 0, "p_two_part": np.nan}
    chi2, df = det["det_lr"], 1
    if inten["intensity_part_estimable"]:
        chi2, df = chi2 + inten["int_lr"], 2
    return {"two_part_chi2": float(chi2), "two_part_df": df, "p_two_part": float(stats.chi2.sf(chi2, df))}


def detected_summary(values: np.ndarray, z: np.ndarray) -> dict:
    """Among detected cells: medians, mean-based log2 fold change and Mann-Whitney p (ZeroK vs Ctrl)."""
    vc, vz = values[~z], values[z]
    out = {"median_ln_ctrl": float(np.median(vc)) if len(vc) else np.nan,
           "median_ln_zerok": float(np.median(vz)) if len(vz) else np.nan,
           "log2fc_detected": np.nan, "median_difference_ln": np.nan, "wilcoxon_p": np.nan}
    if len(vc) >= MIN_DET_PER_CONDITION and len(vz) >= MIN_DET_PER_CONDITION:
        out.update(log2fc_detected=float((vz.mean() - vc.mean()) / np.log(2)),
                   median_difference_ln=float(np.median(vz) - np.median(vc)),
                   wilcoxon_p=float(stats.mannwhitneyu(vz, vc, alternative="two-sided").pvalue))
    return out


def protein_test(values: np.ndarray, det: np.ndarray, z: np.ndarray, log_depth: np.ndarray,
                 det_part: dict | None = None) -> dict:
    """All per-protein statistics for one group and one variant (``det_part`` reused across variants)."""
    if det_part is None:
        det_part = detection_part(det, z, log_depth)
    inten = intensity_part(values[det], z[det], log_depth[det])
    row = {**det_part, **proportion_compare(det_part["det_ctrl"], det_part["n_ctrl"], det_part["det_zerok"],
                                             det_part["n_zerok"], prefix="detection_")}
    for k in ("detection_n_ctrl", "detection_k_ctrl", "detection_n_zerok", "detection_k_zerok"):
        row.pop(k)
    row.update(detected_summary(values[det], z[det]))
    row.update(inten)
    row.update(combine_parts(det_part, inten))
    row["testable"] = bool(np.isfinite(row["p_two_part"]))
    return row


def score_test(values: np.ndarray, z: np.ndarray, log_depth: np.ndarray) -> dict:
    """Score ~ condition + log depth (OLS, t test of condition) and Mann-Whitney (scores; D1)."""
    vc, vz = values[~z], values[z]
    out = {"n_ctrl": len(vc), "n_zerok": len(vz), "median_ctrl": np.nan, "median_zerok": np.nan,
           "median_difference": np.nan, "wilcoxon_p": np.nan, "score_coef": np.nan, "score_coef_se": np.nan,
           "p_score_lm": np.nan, "testable": False}
    if len(vc) < MIN_DET_PER_CONDITION or len(vz) < MIN_DET_PER_CONDITION:
        return out
    X = design(z, log_depth)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, values, rcond=None)
    rss = float(np.sum((values - X @ beta) ** 2))
    se = float(np.sqrt(rss / (n - p) * np.linalg.inv(X.T @ X)[1, 1]))
    t = beta[1] / se if se > 0 else np.nan
    out.update(median_ctrl=float(np.median(vc)), median_zerok=float(np.median(vz)),
               median_difference=float(np.median(vz) - np.median(vc)),
               wilcoxon_p=float(stats.mannwhitneyu(vz, vc, alternative="two-sided").pvalue),
               score_coef=float(beta[1]), score_coef_se=se,
               p_score_lm=float(2 * stats.t.sf(abs(t), n - p)) if np.isfinite(t) else np.nan, testable=True)
    return out


# ---------------------------------------------------------------------------
# D1: observed direction and match status
# ---------------------------------------------------------------------------
def observed_direction(q: float, parts: list[tuple[float, float, float]]) -> str:
    """Observed direction from the BH q of the primary test and its parts [(coef, p, chi2), ...].

    q >= 0.05 -> no BH-significant difference. Otherwise the sign of the part(s) with nominal p < 0.05
    (the part with the larger chi2 when neither part reaches it); 'mixed' when the signs disagree.
    """
    if not np.isfinite(q):
        return NOT_TESTABLE
    if q >= ALPHA:
        return OBS_NS
    usable = [(c, p, x) for c, p, x in parts if np.isfinite(c) and np.isfinite(p)]
    if not usable:
        return NOT_TESTABLE
    sig = [u for u in usable if u[1] < ALPHA] or [max(usable, key=lambda u: u[2] if np.isfinite(u[2]) else -1)]
    signs = {int(np.sign(c)) for c, _, _ in sig if c != 0}
    if len(signs) > 1:
        return OBS_MIXED
    return UP if signs == {1} else DOWN if signs == {-1} else OBS_NS


def dominant_sign(parts: list[tuple[float, float, float]]) -> float:
    """Sign of the part with the larger chi2 (robustness 'keeps its sign')."""
    usable = [(c, p, x) for c, p, x in parts if np.isfinite(c) and np.isfinite(x)]
    if not usable:
        return np.nan
    return float(np.sign(max(usable, key=lambda u: u[2])[0]))


def match_status(expected: str, observed: str) -> tuple[str, str]:
    """Match column (D1): match / mismatch / not testable, plus not scored for 'no expectation'."""
    if observed == NOT_TESTABLE:
        return NOT_TESTABLE, "fewer than 5 detected cells in the group, or the model could not be fitted"
    if expected == NO_EXPECTATION:
        return NOT_SCORED, ""
    if expected == NO_CHANGE:
        if observed == OBS_NS:
            return "match", "no BH-significant difference (not a test of equivalence)"
        return "mismatch", f"BH-significant difference ({observed})"
    if observed == expected:
        return "match", f"BH-significant, {observed} as expected"
    if observed == OBS_NS:
        return "mismatch", "no BH-significant difference"
    if observed == OBS_MIXED:
        return "mismatch", "mixed: detection and intensity parts point in opposite directions"
    return "mismatch", f"BH-significant in the opposite direction ({observed})"


def protein_parts(row) -> list[tuple[float, float, float]]:
    return [(row["det_logodds"], row["det_p"], row["det_lr"]), (row["int_coef_ln"], row["int_p"], row["int_lr"])]


# ---------------------------------------------------------------------------
# Pseudobulk (D6) and vectorized effect sizes (D8)
# ---------------------------------------------------------------------------
def back_transform(joint: Joint, variant: str) -> np.ndarray:
    """Normalized ln values back to the intensity scale; missing = 0 (D6)."""
    m = joint.mats[variant]
    out = np.zeros_like(m)
    if variant == "cell_median":           # ln(x)-based (preprocessing_lib median rule)
        out[joint.detected] = np.exp(m[joint.detected])
    else:                                   # ln(1 + x)-based
        out[joint.detected] = np.expm1(m[joint.detected])
    return out


def pseudobulk_log2fc(bt: np.ndarray, idx_c: np.ndarray, idx_z: np.ndarray) -> np.ndarray:
    mc, mz = bt[idx_c].mean(axis=0), bt[idx_z].mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log2(mz / mc)
    out[~np.isfinite(out)] = np.nan
    return out


def effect_sizes(joint: Joint, variant: str, idx_c: np.ndarray, idx_z: np.ndarray, cols: np.ndarray,
                 bt: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per protein: detection difference, mean-based log2FC among detected (>= 5 per condition), pseudobulk."""
    m = joint.mats[variant][:, cols]
    d = joint.detected[:, cols]
    out = {}
    nd_c, nd_z = d[idx_c].sum(axis=0), d[idx_z].sum(axis=0)
    out["detection_difference"] = nd_z / len(idx_z) - nd_c / len(idx_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_c = (m[idx_c] * d[idx_c]).sum(axis=0) / nd_c
        mean_z = (m[idx_z] * d[idx_z]).sum(axis=0) / nd_z
    lfc = (mean_z - mean_c) / np.log(2)
    lfc[(nd_c < MIN_DET_PER_CONDITION) | (nd_z < MIN_DET_PER_CONDITION)] = np.nan
    out["log2fc_detected"] = lfc
    if bt is not None:
        out["pseudobulk_log2fc"] = pseudobulk_log2fc(bt[:, cols], idx_c, idx_z)
    return out


# ---------------------------------------------------------------------------
# D4: depth matching
# ---------------------------------------------------------------------------
def depth_match(log_depth: np.ndarray, plate: np.ndarray, is_zerok: np.ndarray, members: np.ndarray,
                rng: np.random.Generator, caliper: float = CALIPER) -> tuple[np.ndarray, np.ndarray]:
    """Greedy 1:1 nearest neighbour on ln(depth) within sort plate (ctrl_reviewer_core_depth_detection.nn_match).

    Within each plate the condition with fewer cells is processed in random order; each cell takes the
    nearest unmatched cell of the other condition if within the caliper. Returns (ctrl idx, zerok idx).
    """
    ctrl_out, zerok_out = [], []
    for p in sorted(set(plate[members])):
        zk = np.flatnonzero(members & (plate == p) & is_zerok)
        ct = np.flatnonzero(members & (plate == p) & ~is_zerok)
        first, pool = (zk, list(ct)) if len(zk) <= len(ct) else (ct, list(zk))
        pool_arr = np.array(pool, dtype=int)
        used = np.zeros(len(pool_arr), bool)
        for t in rng.permutation(first):
            if used.all():
                break
            d = np.abs(log_depth[pool_arr] - log_depth[t])
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= caliper:
                used[j] = True
                partner = int(pool_arr[j])
                zk_i, ct_i = (int(t), partner) if is_zerok[t] else (partner, int(t))
                ctrl_out.append(ct_i)
                zerok_out.append(zk_i)
    return np.array(ctrl_out, dtype=int), np.array(zerok_out, dtype=int)


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (as ctrl_reviewer_core_depth_detection.smd)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def repeat_rng(*keys: int) -> np.random.Generator:
    """Deterministic generator per procedure and repeat (seed 42 first)."""
    return np.random.default_rng([SEED, *keys])


LABEL_COLUMN_INDEX = {"dct_label": 0, "dct_label_alternative": 1, "label_free": 2}


def matched_sets(joint: Joint, label_column: str, group: str, mask: np.ndarray,
                 n_repeats: int = N_REPEATS) -> list[tuple[np.ndarray, np.ndarray]]:
    """D4 depth-matched (ctrl idx, zerok idx) per repeat; seeds depend only on label column, group and repeat,
    so steps 4 and 5 use the same matched cells."""
    plate = joint.obs["plate"].to_numpy()
    out = []
    for r in range(n_repeats):
        rng = repeat_rng(4, LABEL_COLUMN_INDEX[label_column], GROUP_ORDER.index(group), r)
        out.append(depth_match(joint.log_depth, plate, joint.is_zerok, mask, rng))
    return out


def summarize_repeats(values: np.ndarray, full: float, pvals: np.ndarray | None = None) -> dict:
    """Median and 2.5-97.5% range of a repeated effect, sign agreement with the full data, share p < 0.05."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    out = {"n_repeats": int(ok.sum()), "effect_full": full,
           "effect_median": float(np.median(v[ok])) if ok.any() else np.nan,
           "effect_q025": float(np.quantile(v[ok], 0.025)) if ok.any() else np.nan,
           "effect_q975": float(np.quantile(v[ok], 0.975)) if ok.any() else np.nan,
           "fraction_same_sign_as_full": (float(np.mean(np.sign(v[ok]) == np.sign(full)))
                                          if ok.any() and np.isfinite(full) and full != 0 else np.nan)}
    if pvals is not None:
        p = np.asarray(pvals, float)
        out["fraction_p_below_0.05"] = float(np.mean(p[np.isfinite(p)] < ALPHA)) if np.isfinite(p).any() else np.nan
    return out


# ---------------------------------------------------------------------------
# Scores (D3)
# ---------------------------------------------------------------------------
def scaled_object(joint: Joint, variant: str):
    """AnnData with the variant matrix (float32, as the submitted pipeline) after sc.pp.scale(max_value=10)."""
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=joint.mats[variant].astype(np.float32),
                   obs=pd.DataFrame(index=pd.Index(joint.obs.index.astype(str), name="cell_id")),
                   var=pd.DataFrame(index=pd.Index(joint.var_names)))
    sc.pp.scale(a, max_value=10)
    return a


def mean_z(adata, genes: list[str]) -> np.ndarray:
    present = [g for g in genes if g in set(adata.var_names)]
    if not present:
        return np.full(adata.n_obs, np.nan)
    idx = [int(np.flatnonzero(adata.var_names == g)[0]) for g in present]
    return np.asarray(cc.dense(adata.X)[:, idx], dtype=np.float64).mean(axis=1)


def score_cells(joint: Joint, variant: str, type_lists: dict[str, list[str]]) -> pd.DataFrame:
    """Per-cell scores for one variant (D3); columns without the variant prefix."""
    a = scaled_object(joint, variant)
    out = {}
    aldo = t2.aldo_scores(a, source=variant)
    for col in aldo.columns:
        out[col.split("__", 1)[1]] = aldo[col].to_numpy()
    t2.score(a, list(t2.CELL_CYCLE_MARKERS), "cell_cycle_score")
    out["cell_cycle_score"] = a.obs["cell_cycle_score"].to_numpy(float)
    no_slc = t18.drop_from_lists(type_lists, ["Slc12a3"])
    for suffix, lists in (("", type_lists), ("__no_Slc12a3", no_slc)):
        frame = t18.score_types(a, lists)
        for t in frame.columns:
            out[f"{t}_score{suffix}"] = frame[t].to_numpy(float)
            out[f"{t}_meanz{suffix}"] = mean_z(a, lists[t])
        if {"DCT1", "DCT2"} <= set(frame.columns):
            out[f"DCT1_minus_DCT2_score{suffix}"] = out[f"DCT1_score{suffix}"] - out[f"DCT2_score{suffix}"]
            out[f"DCT1_minus_DCT2_meanz{suffix}"] = out[f"DCT1_meanz{suffix}"] - out[f"DCT2_meanz{suffix}"]
    return pd.DataFrame(out, index=joint.obs.index)


def score_members_in_joint(joint: Joint) -> dict[str, int]:
    var = set(joint.var_names)
    return {item: len([g for g in dict.fromkeys(pos + neg) if g in var])
            for item, (pos, neg) in t21.SCORE_LISTS.items()}


# ---------------------------------------------------------------------------
# Panel statistics (step 2; reused by step 5)
# ---------------------------------------------------------------------------
@dataclass
class PanelSpec:
    proteins: list[tuple[str, str, int]]       # (item, joint var name, column index)
    scores: list[tuple[str, str]]              # (item, per-cell score column)
    expected: dict[str, str]                   # item -> expected direction
    item_order: list[str]


def panel_spec_items(panel: pd.DataFrame) -> list[str]:
    """Tested panel items (proteins, then scores) in frozen panel order."""
    return panel.loc[panel["item_type"].isin(["protein", "score"]), "item"].tolist()


def panel_spec(joint: Joint, panel: pd.DataFrame) -> PanelSpec:
    proteins = []
    for item in panel.loc[panel["item_type"] == "protein", "item"]:
        name = resolve_protein(joint, item)
        proteins.append((item, name, joint.var_index(name) if name else -1))
    scores = [(item, SCORE_ITEMS[item]) for item in panel.loc[panel["item_type"] == "score", "item"]]
    tested = panel.loc[panel["item_type"].isin(["protein", "score"])]
    return PanelSpec(proteins, scores, dict(zip(tested["item"], tested["expected_direction"])),
                     tested["item"].tolist())


def panel_rows(joint: Joint, spec: PanelSpec, idx: np.ndarray, scores: dict[str, pd.DataFrame],
               variants=VARIANTS, keys: dict | None = None) -> list[dict]:
    """Panel statistics for the cells ``idx`` (both conditions), every variant."""
    keys = dict(keys or {})
    z = joint.is_zerok[idx]
    ld = joint.log_depth[idx]
    rows = []
    for item, name, col in spec.proteins:
        if col < 0:
            for v in variants:
                rows.append({**keys, "variant": v, "item": item, "item_type": "protein", "testable": False})
            continue
        det = joint.detected[idx, col]
        dpart = detection_part(det, z, ld)
        for v in variants:
            row = protein_test(joint.mats[v][idx, col], det, z, ld, det_part=dpart)
            rows.append({**keys, "variant": v, "item": item, "item_type": "protein", "joint_var_name": name,
                         "p_primary": row["p_two_part"], **row})
    for item, col in spec.scores:
        for v in variants:
            row = score_test(scores[v][col].to_numpy(float)[idx], z, ld)
            rows.append({**keys, "variant": v, "item": item, "item_type": "score", "score_column": col,
                         "p_primary": row["p_score_lm"], **row})
    return rows


def finalize_panel(frame: pd.DataFrame, spec: PanelSpec, block: list[str]) -> pd.DataFrame:
    """BH within each panel block, observed direction and match status (D1)."""
    frame = cc.add_bh(frame, block, p_col="p_primary", out_col="q_primary")
    if "detection_fisher_p" in frame:
        frame = cc.add_bh(frame, block, p_col="detection_fisher_p", out_col="q_fisher")
    frame = cc.add_bh(frame, block, p_col="wilcoxon_p", out_col="q_wilcoxon")
    obs_dir, sign, status, reason = [], [], [], []
    for _, r in frame.iterrows():
        if r["item_type"] == "protein":
            parts = protein_parts(r) if "det_logodds" in r and np.isfinite(r.get("det_lr", np.nan)) else []
        else:   # one part; its chi2 placeholder only ranks parts
            parts = [(r.get("score_coef", np.nan), r.get("p_score_lm", np.nan), 1.0)]
        o = observed_direction(r["q_primary"], parts) if parts else NOT_TESTABLE
        s, why = match_status(spec.expected[r["item"]], o)
        obs_dir.append(o)
        sign.append(dominant_sign(parts) if parts else np.nan)
        status.append(s)
        reason.append(why)
    frame["expected_direction"] = frame["item"].map(spec.expected)
    frame["observed_direction"] = obs_dir
    frame["dominant_sign"] = sign
    frame["match_status"] = status
    frame["match_reason"] = reason
    frame["p_value_note"] = P_NOTE
    return frame


# ---------------------------------------------------------------------------
# Proteome-wide statistics (step 3; label-free version in step 5)
# ---------------------------------------------------------------------------
PROTEOME_FIRST = ["label_column", "group", "variant", "protein", "gene_name", "match_level", "in_panel",
                  "observed_direction", "p_two_part", "q_two_part", "two_part_chi2", "two_part_df",
                  "n_ctrl", "n_zerok", "det_ctrl", "det_zerok", "detection_fraction_ctrl", "detection_fraction_zerok",
                  "detection_difference_zerok_minus_ctrl", "detection_difference_ci_low",
                  "detection_difference_ci_high", "detection_fisher_p", "q_fisher", "det_logodds", "det_logodds_se",
                  "det_p", "detection_separation", "median_ln_ctrl", "median_ln_zerok", "log2fc_detected",
                  "wilcoxon_p", "q_wilcoxon", "int_log2fc_adjusted", "int_coef_se", "int_p",
                  "intensity_part_estimable", "pseudobulk_log2fc_pooled", "pseudobulk_log2fc_P1",
                  "pseudobulk_log2fc_P2"]


def proteome_group(joint: Joint, idx: np.ndarray, bts: dict[str, np.ndarray], keys: dict,
                   panel_names: set[str], variants=VARIANTS, z: np.ndarray | None = None) -> pd.DataFrame:
    """Two-part statistics for every matched protein detected in >= 10% of the group's cells in at least one
    condition; BH within the group per variant; observed direction (D1 rule).

    ``z`` (bool over ``idx``) replaces ZeroK as the comparison side (step 5b plate contrasts); the *_ctrl
    columns then hold the reference side and the *_zerok columns the comparison side.
    """
    z = joint.is_zerok[idx] if z is None else np.asarray(z, bool)
    ld = joint.log_depth[idx]
    det = joint.detected[idx]
    plate = joint.obs["plate"].to_numpy()[idx]
    frac_c, frac_z = det[~z].mean(axis=0), det[z].mean(axis=0)
    keep = np.flatnonzero((frac_c >= MIN_DETECTED_FRACTION) | (frac_z >= MIN_DETECTED_FRACTION))
    pb = {}
    for v in variants:
        b = bts[v][idx][:, keep]
        pb[v] = {"pooled": pseudobulk_log2fc(b, np.flatnonzero(~z), np.flatnonzero(z))}
        for p in PLATES:
            pb[v][p] = pseudobulk_log2fc(b, np.flatnonzero(~z & (plate == p)), np.flatnonzero(z & (plate == p)))
    var = joint.var
    rows = []
    for j, col in enumerate(keep):
        name = joint.var_names[col]
        dpart = detection_part(det[:, col], z, ld)
        for v in variants:
            row = protein_test(joint.mats[v][idx, col], det[:, col], z, ld, det_part=dpart)
            rows.append({**keys, "variant": v, "protein": name, "gene_name": var.at[name, "gene_name"],
                         "match_level": var.at[name, "match_level"], "in_panel": name in panel_names, **row,
                         "pseudobulk_log2fc_pooled": pb[v]["pooled"][j], "pseudobulk_log2fc_P1": pb[v]["P1"][j],
                         "pseudobulk_log2fc_P2": pb[v]["P2"][j]})
    frame = pd.DataFrame(rows)
    block = [*keys, "variant"]
    frame = cc.add_bh(frame, block, p_col="p_two_part", out_col="q_two_part")
    frame = cc.add_bh(frame, block, p_col="detection_fisher_p", out_col="q_fisher")
    frame = cc.add_bh(frame, block, p_col="wilcoxon_p", out_col="q_wilcoxon")
    frame["observed_direction"] = [observed_direction(q, protein_parts(r)) if np.isfinite(r["det_lr"]) else
                                   NOT_TESTABLE for q, (_, r) in zip(frame["q_two_part"], frame.iterrows())]
    frame["p_value_note"] = P_NOTE
    return ordered(frame, PROTEOME_FIRST)


def ordered(frame: pd.DataFrame, first: list[str]) -> pd.DataFrame:
    return frame[[c for c in first if c in frame] + [c for c in frame if c not in first]]
