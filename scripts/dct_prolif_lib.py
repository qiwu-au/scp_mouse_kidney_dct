"""dct_prolif shared definitions and functions. Runs no analysis by itself.

Imported by the ``dct_prolif_*.py`` step
scripts. Plan: ``the analysis plan`` (frozen 3 Nov 2026;
SHA256 in ``results/tables/dct_prolif_frozen_brief_sha256.csv``, committed in 9f6adb9).

dct_prolif builds DCT1, DCT2 and ProLIF labels on the combined-search dct_intake objects, each condition alone,
in which an undetected protein counts as not measured (never 0) and ProLIF needs positive evidence
of proliferation. It does not replace the dct_labels labels.

Runs
----
- ``frozen``: the plan as written; depth floor 1,000 protein IDs; columns ``cs_t31_*``
  (route A) and ``cs_t31b_*`` (route B). Kept for comparison since the switch below.
- ``nofloor``: change approved by the authors on 3 Nov 2026 ("Add a no-floor run"). Reason (authors):
  the 1,000-ID floor has no derivation in the plan and leaves 21-25% of cells without a type,
  whereas the kidney data exclude only cells with < 100 proteins. Floor = the >= 100-ID QC filter
  of the dct_intake objects (basic QC only), so every retained cell is labelled. Everything else as the
  frozen run; columns ``cs_t31nf_*`` and ``cs_t31nfb_*``. **dct_prolif primary run** since 3 Nov 2026
  (authors: 'Primary everywhere', 'Run it'; ``PRIMARY_RUN``); the canonical table copies its route-A
  label to ``t31_label``. The dct_labels labels stay the DCT primary until the authors adopt dct_prolif.
- Floors 500 and 1,500: route A only, information only (plan).

Fixed settings (plan, "Fixed settings")
---------------------------------------
- Depth = ``obs['n_detected']`` (retained proteins of the condition). Above the floor = n_detected >= floor.
- Depth bins: 20 equal-count bins of ln(n_detected) among the cells above the floor (ranks, ties
  broken by object order, so the bins are of equal size).
- Proliferation set P: Tirosh S and G2M proteins (``reviewer_core_lib``, aliases) retained in both
  conditions (dct_intake ``in_joint``), minus Mki67 and Top2a (held out). Expected 36.
- Detection part D_c = mean over p in P of (d_pc - pi_p,b); pi_p,b = detection rate of p in the
  cell's depth bin.
- Null: 1,000 random protein sets of size |P| from the condition's retained proteins, excluding the
  Tirosh lists (aliases included) and the original 18-protein panel, matched to P on
  detection-frequency deciles (detection over the cells above the floor; deciles by rank over all
  retained proteins of the condition). Sampling without replacement within a decile; seed 42.
  p = (1 + #null D >= observed D) / 1,001 (ties within 1e-12 count as >=); BH over the cells above
  the floor; cycling = q < 0.05.
- Intensity part (reported only): mean over the detected p in P of the residual z of ln(intensity)
  on ln(n_detected) (OLS over the detected cells above the floor; z = residual / SD of residuals,
  ddof 1; proteins detected in < 20 cells above the floor left out).
- ProLIF = cycling and Slc12a3 detected; cycling without Slc12a3 = "Cycling, NCC not detected".
- Identity: type-specific DCT V1 workbook DCT1 / DCT2 markers (Slc12a3 and Umod shared, as dct_labels)
  detected in >= 20 cells above the floor; residual z as above; type score = mean of the available
  z; axis A = DCT2 score - DCT1 score (needs >= 1 detected marker of each type); two-component
  Gaussian mixture (n_init 10, random_state 42); posterior >= 0.8 -> DCT1 (lower-mean component)
  or DCT2, else "DCT, subtype not resolved" (UNRESOLVED; renamed at the authors' request, 3 Nov 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

import scp_common as cc
import reviewer_core_lib as t2
import cluster_de_lib as t5
import labels_core_lib as t18
import dct_intake_lib as t26

TASK = "dct_prolif"
SEED = cc.RANDOM_SEED
CONDITIONS = ("Ctrl", "ZeroK")
KEY = {"Ctrl": "ctrl", "ZeroK": "zerok"}
DCT_LABELS_PREFIX = {"Ctrl": "cs_ctrl", "ZeroK": "cs_zerok"}

FREEZE_COMMIT = ""   # no planning document is part of the public code
OBJECTS = {c: t26.OBJECTS[c] for c in CONDITIONS}

DCT_LABELS_TASK = "dct_labels"
DCT_LABELS_LABELS = cc.out_path(DCT_LABELS_TASK, "tables", "dct_canonical_labels", ".csv")
DCT_LABELS_H5AD = {c: cc.out_path(DCT_LABELS_TASK, "h5ad", f"DCT_{KEY[c]}_labelled", ".h5ad") for c in CONDITIONS}
DCT_REVIEWER_TASK = "dct_reviewer"
DCT_REVIEWER_TASK3_H5AD = {c: cc.out_path(DCT_REVIEWER_TASK, "h5ad", f"DCT_{KEY[c]}_task3_scored", ".h5ad") for c in CONDITIONS}
DCT_REVIEWER_CYCLING_QUANTILE = 0.90


@dataclass(frozen=True)
class RunSpec:
    key: str
    floor: int
    prefix_a: str
    prefix_b: str
    display: str
    status: str


RUNS = {
    "frozen": RunSpec("frozen", 1000, "cs_t31", "cs_t31b",
                      "frozen plan, kept for comparison: depth floor 1,000 protein IDs",
                      "frozen plan; kept for comparison (authors, 3 Nov 2026)"),
    "nofloor": RunSpec("nofloor", 100, "cs_t31nf", "cs_t31nfb",
                       "dct_prolif primary (authors, 3 Nov 2026): no floor beyond the >= 100-ID QC filter",
                       "dct_prolif primary (approved change, 3 Nov 2026; primary since 3 Nov 2026)"),
}
RUN_KEYS = tuple(RUNS)
# dct_prolif primary run (authors, 3 Nov 2026: 'Primary everywhere', 'Run it'): its route-A label is copied to t31_label.
PRIMARY_RUN = "nofloor"
PRIMARY_SWITCH = {
    "date": "2026-11-03",
    "change": "dct_prolif primary run switched from the frozen 1,000-ID run to the no-floor run (basic QC only: the >= 100-ID "
              "filter of the dct_intake objects); the frozen run is kept for comparison",
    "approved_by": "authors, in the dct_prolif session ('Try the no floor version, use only basic QC'; 'Primary everywhere'; "
                   "'Run it')",
    "scope": "within dct_prolif only; the dct_labels labels stay the DCT primary until the authors adopt dct_prolif",
}
INFO_FLOORS = (500, 1500)
APPROVED_CHANGE = {
    "date": "2026-11-03",
    "change": "added a full dct_prolif run (route A, acceptance tests 1-5, route B, bridge, figures) with no depth floor "
              "beyond the >= 100-ID QC filter of the dct_intake objects; the frozen 1,000-ID run is unchanged",
    "reason": "authors, 3 Nov 2026: the 1,000-ID floor has no derivation in the plan and leaves 21-25% of cells without "
              "a type, whereas the kidney data exclude only cells with < 100 proteins",
    "approved_by": "authors, in the dct_prolif session ('Add a no-floor run')",
}
# Plan: expected cells removed by each floor (n_detected < floor).
EXPECTED_BELOW = {"Ctrl": {1000: 155, 500: 54, 1500: 288}, "ZeroK": {1000: 177, 500: 68, 1500: 286}}
EXPECTED_P = 36
EXPECTED_MARKERS = {"DCT1": ["Pvalb", "Egf", "Trpm7", "Ptgfr"], "DCT2": ["Calb1", "S100g", "Slc8a1", "Klk1"]}

N_BINS = 20
N_NULL = 1000
Q_CYCLING = 0.05
TIE_TOL = 1e-12
MIN_CELLS_INTENSITY = 20
MIN_CELLS_MARKER = 20
POSTERIOR_MIN = 0.8
GMM_N_INIT = 10
N_RANDOM_SETS = 100              # acceptance test 4
TEST4_MAX_GE = 5                 # pass if fewer than 5 of 100 sets give >= cycling cells
NULL_SEEDS = tuple(range(10))    # acceptance test 5
JACCARD_MIN = 0.8
NOISE_REPEATS = 20
NOISE_REL_SD = 1e-6
HELD_OUT = ("Mki67", "Top2a")
NCC = "Slc12a3"
SHARED_MARKERS = ("Slc12a3", "Umod")

# Route B (plan)
B_MIN_DETECTION = 0.5
B_N_PCS = 20
B_K = 10
B_RESOLUTION = 1.0
B_INFO_RESOLUTIONS = (0.5, 3.0)
B_INFO_SEEDS = tuple(range(10))
B_MAX_VALUE = 10.0

# Labels
DCT1, DCT2, PROLIF = "DCT1", "DCT2", "ProLIF"
UNRESOLVED = "DCT, subtype not resolved"
UNKNOWN_MARKERS = "Unknown (markers not detected)"
BELOW = "Below depth floor"
CYCLING_NO_NCC = "Cycling, NCC not detected"
CYCLING, NON_CYCLING = "Cycling", "Non-cycling"
LABEL_ORDER = (DCT1, DCT2, PROLIF, CYCLING_NO_NCC, UNRESOLVED, UNKNOWN_MARKERS, BELOW)
IDENTITY_ORDER = (DCT1, DCT2, UNRESOLVED, UNKNOWN_MARKERS, BELOW)
CYCLING_ORDER = (CYCLING, NON_CYCLING, BELOW)
TEST_SET = (DCT1, DCT2, UNRESOLVED, PROLIF)       # acceptance tests 1-3
LABEL_COLORS = {DCT1: t2.LABEL_PALETTE["DCT1"], DCT2: t2.LABEL_PALETTE["DCT2"], PROLIF: t2.LABEL_PALETTE["ProLIF"],
                CYCLING_NO_NCC: "#e87ba4", UNRESOLVED: "#8e7cc3", UNKNOWN_MARKERS: "#B3B3B3", BELOW: "#5f5e5a",
                "Unknown": "#B3B3B3", t18.BELOW_MIN: "#d6d5cf"}
DESIGN_NOTE = ("One pool of 3 mice per condition; ZeroK is a second pool on another diet, not a replicate; plates "
               "P1/P2 are sort plates of the pooled cells (technical label, never a replicate).")


def col(run_key: str, name: str, route: str = "a") -> str:
    """Canonical column name, e.g. col('frozen', 'label') -> cs_t31_label; route 'b' -> cs_t31b_label."""
    spec = RUNS[run_key]
    return f"{spec.prefix_a if route == 'a' else spec.prefix_b}_{name}"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class Condition:
    """One condition's dct_intake object as arrays (cells in object order, all retained proteins)."""

    cond: str
    cell_ids: pd.Index
    obs: pd.DataFrame
    var_names: pd.Index
    var: pd.DataFrame
    intensity: np.ndarray        # float64, missing = 0
    detected: np.ndarray         # bool
    ids: np.ndarray              # n_detected, float
    plate: np.ndarray            # 'P1' / 'P2'

    @property
    def ln_ids(self) -> np.ndarray:
        return np.log(self.ids)

    def idx(self, genes) -> np.ndarray:
        out = self.var_names.get_indexer(list(genes))
        if (out < 0).any():
            raise KeyError(f"{self.cond}: not in matrix: {[g for g, i in zip(genes, out) if i < 0]}")
        return out


def load_condition(run: cc.Run, cond: str) -> Condition:
    adata = t26.read_object(run, cond)
    intensity = np.asarray(cc.dense(adata.layers["intensity"]), dtype=np.float64)
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    return Condition(cond, adata.obs_names.astype(str), obs, pd.Index(adata.var_names.astype(str)), adata.var.copy(),
                     intensity, intensity > 0, obs["n_detected"].to_numpy(float), obs["plate"].astype(str).to_numpy())


def brief_hash_row() -> dict:
    return {"path": "", "role": "analysis plan (not part of the public code)", "bytes": 0,
            "sha256_observed": "", "sha256_frozen": "", "pass": True}


def tirosh_matches(var_names) -> pd.DataFrame:
    rows = []
    for phase, genes in (("S", t2.TIROSH_S), ("G2M", t2.TIROSH_G2M)):
        r, _ = t2.match_genes(genes, var_names, phase)
        rows.extend(r)
    return pd.DataFrame(rows)


def prolif_set(ctrl: Condition, zerok: Condition) -> tuple[list[str], pd.DataFrame]:
    """P = Tirosh S + G2M proteins retained in both conditions, minus Mki67 and Top2a (plan)."""
    m_c, m_z = tirosh_matches(ctrl.var_names), tirosh_matches(zerok.var_names)
    m = m_c.rename(columns={"matched_name": "matched_name_ctrl", "present": "retained_ctrl"})
    m["matched_name_zerok"] = m_z["matched_name"].to_numpy()
    m["retained_zerok"] = m_z["present"].to_numpy()
    m["retained_both"] = m["retained_ctrl"] & m["retained_zerok"]
    same = m.loc[m["retained_both"], "matched_name_ctrl"].eq(m.loc[m["retained_both"], "matched_name_zerok"])
    if not same.all():
        raise ValueError("Tirosh protein matched to different names in the two conditions")
    m["held_out"] = m["matched_name_ctrl"].isin(HELD_OUT)
    m["in_P"] = m["retained_both"] & ~m["held_out"]
    P = m.loc[m["in_P"], "matched_name_ctrl"].tolist()
    return P, m


def excluded_from_null(var_names) -> set[str]:
    """Tirosh lists (aliases included) and the original 18-protein panel."""
    return set(t2.TIROSH_ALL) | set(t2.CELL_CYCLE_MARKERS)


def tirosh_present(var_names) -> list[str]:
    """Every Tirosh protein in the matrix (aliases matched), Mki67 and Top2a included (route B features)."""
    m = tirosh_matches(var_names)
    return m.loc[m["present"], "matched_name"].tolist()


def identity_candidates(run: cc.Run, var_names) -> tuple[dict[str, list[str]], t5.MarkerSets]:
    """Type-specific DCT1 / DCT2 workbook markers in the matrix (Slc12a3 and Umod shared, as dct_labels)."""
    markers = t18.dct_markers(run, var_names)
    for g in SHARED_MARKERS:
        if g in markers.types_by_gene and not markers.is_shared(g):
            raise ValueError(f"{g} is not shared under the dct_labels marker sets")
    return {t: markers.specific_present(t) for t in (DCT1, DCT2)}, markers


def route_b_markers(markers: t5.MarkerSets, P: list[str], var_names) -> t5.MarkerSets:
    """Workbook marker sets with the ProLIF row replaced by Mki67, Top2a and P (plan route B, plan default 5).

    Slc12a3 leaves the ProLIF row and stays shared through DCT1 and DCT2; Umod stays shared (labels_core rule).
    """
    var = set(map(str, var_names))
    prolif = list(dict.fromkeys([*HELD_OUT, *P]))
    lists = {t: list(g) for t, g in markers.lists.items()}
    lists[PROLIF] = prolif
    present = {t: [g for g in genes if g in var] for t, genes in lists.items()}
    types_by_gene: dict[str, set[str]] = {}
    for t, genes in lists.items():
        for g in genes:
            types_by_gene.setdefault(g, set()).add(t)
    for gene, reason in t18.TREATED_AS_SHARED.items():
        if gene in types_by_gene:
            types_by_gene[gene] = set(types_by_gene[gene]) | {reason}
    full = dict(markers.full_names)
    return t5.MarkerSets(lists, present, full, types_by_gene)


# ---------------------------------------------------------------------------
# Route A: depth bins, detection part, null
# ---------------------------------------------------------------------------
def equal_count_bins(values: np.ndarray, n: int) -> np.ndarray:
    """Equal-count bins (0 .. n-1) of ``values``; ties broken by position (rank 'first')."""
    ranks = pd.Series(values).rank(method="first").to_numpy()
    return pd.qcut(ranks, n, labels=False).astype(int)


def detection_deciles(detected_above: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detection frequency over the cells above the floor and its decile (0-9, ranks over all retained proteins)."""
    freq = detected_above.mean(axis=0)
    return freq, equal_count_bins(freq, 10)


@dataclass
class Prolif:
    """Route A proliferation pieces for one condition and floor."""

    above: np.ndarray                 # bool over all cells
    bins: np.ndarray                  # depth bin per above cell
    rates: np.ndarray                 # bins x proteins
    R: np.ndarray                     # above cells x proteins: d - pi
    freq: np.ndarray                  # detection frequency per protein (above cells)
    decile: np.ndarray                # per protein
    P_idx: np.ndarray
    pool_idx: np.ndarray              # null pool (protein indices)
    need: dict[int, int] = field(default_factory=dict)   # decile -> proteins of P in it


def prolif_setup(c: Condition, floor: int, P: list[str]) -> Prolif:
    above = c.ids >= floor
    det = c.detected[above]
    bins = equal_count_bins(c.ln_ids[above], N_BINS)
    rates = np.vstack([det[bins == b].mean(axis=0) for b in range(N_BINS)])
    R = det.astype(np.float64) - rates[bins]
    freq, dec = detection_deciles(det)
    P_idx = c.idx(P)
    excl = excluded_from_null(c.var_names)
    pool = np.array([j for j, g in enumerate(c.var_names) if g not in excl])
    need = pd.Series(dec[P_idx]).value_counts().sort_index().to_dict()
    return Prolif(above, bins, rates, R, freq, dec, P_idx, pool, {int(k): int(v) for k, v in need.items()})


def draw_sets(rng: np.random.Generator, pool: np.ndarray, decile: np.ndarray, need: dict[int, int],
              n_sets: int) -> np.ndarray:
    """``n_sets`` protein sets matched on deciles: per decile, the ``need`` proteins with the smallest uniform keys
    among the pool proteins of that decile (sampling without replacement). Returns n_sets x |set| indices."""
    parts = []
    for d in sorted(need):
        cand = pool[decile[pool] == d]
        k = need[d]
        if len(cand) < k:
            raise ValueError(f"decile {d}: pool has {len(cand)} proteins, {k} needed")
        keys = rng.random((n_sets, len(cand)))
        pick = np.argpartition(keys, k - 1, axis=1)[:, :k] if k < len(cand) else np.tile(np.arange(len(cand)), (n_sets, 1))
        parts.append(cand[pick])
    return np.hstack(parts)


def set_means(R: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """Mean of R over each protein set: cells x sets."""
    S, k = sets.shape
    M = sparse.csr_matrix((np.full(S * k, 1.0 / k), (np.repeat(np.arange(S), k), sets.ravel())),
                          shape=(S, R.shape[1]))
    return np.asarray(M @ R.T).T


def empirical_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    return (1 + (null >= obs[:, None] - TIE_TOL).sum(axis=1)) / (null.shape[1] + 1)


def bh(p: np.ndarray) -> np.ndarray:
    return stats.false_discovery_control(p, method="bh")


def cycling_call(pr: Prolif, set_idx: np.ndarray, rng: np.random.Generator, pool: np.ndarray | None = None) -> dict:
    """D for one protein set and its 1,000-set matched null; p, BH q and the cycling call (above cells)."""
    D = pr.R[:, set_idx].mean(axis=1)
    need = pd.Series(pr.decile[set_idx]).value_counts().sort_index()
    null_sets = draw_sets(rng, pr.pool_idx if pool is None else pool, pr.decile,
                          {int(k): int(v) for k, v in need.items()}, N_NULL)
    null = set_means(pr.R, null_sets)
    p = empirical_p(D, null)
    q = bh(p)
    return {"D": D, "p": p, "q": q, "cycling": q < Q_CYCLING, "null_mean": null.mean(axis=1),
            "null_sd": null.std(axis=1, ddof=1), "null_sets": null_sets}


# ---------------------------------------------------------------------------
# Residual z (intensity part and identity markers)
# ---------------------------------------------------------------------------
def residual_z(intensity: np.ndarray, detected: np.ndarray, ln_ids: np.ndarray, min_cells: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Per protein: OLS ln(intensity) ~ ln(n_detected) over detected cells; z = residual / SD(residuals, ddof 1).

    Missing stays NaN (not measured). Proteins detected in fewer than ``min_cells`` cells are all NaN.
    Returns (cells x proteins z, per-protein fit table)."""
    n, m = intensity.shape
    Z = np.full((n, m), np.nan)
    rows = []
    for j in range(m):
        d = detected[:, j]
        nd = int(d.sum())
        if nd < min_cells:
            rows.append({"j": j, "n_detected": nd, "used": False, "slope": np.nan, "intercept": np.nan, "resid_sd": np.nan})
            continue
        x, y = ln_ids[d], np.log(intensity[d, j])
        slope, intercept = np.polyfit(x, y, 1)
        r = y - (intercept + slope * x)
        sd = r.std(ddof=1)
        Z[d, j] = r / sd if sd > 0 else np.nan
        rows.append({"j": j, "n_detected": nd, "used": True, "slope": slope, "intercept": intercept, "resid_sd": sd})
    return Z, pd.DataFrame(rows)


def intensity_part(c: Condition, pr: Prolif) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean residual z over the detected P proteins (proteins with >= 20 detected above cells)."""
    a = pr.above
    Z, fits = residual_z(c.intensity[a][:, pr.P_idx], c.detected[a][:, pr.P_idx], c.ln_ids[a], MIN_CELLS_INTENSITY)
    fits.insert(0, "protein", c.var_names[pr.P_idx][fits["j"].to_numpy()])
    n_used = np.isfinite(Z).sum(axis=1)
    part = np.where(n_used > 0, np.nansum(np.nan_to_num(Z, nan=0.0), axis=1) / np.maximum(n_used, 1), np.nan)
    return part, fits.drop(columns="j")


# ---------------------------------------------------------------------------
# Route A identity
# ---------------------------------------------------------------------------
def identity_markers(c: Condition, above: np.ndarray, candidates: dict[str, list[str]]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    rows, used = [], {}
    for t in (DCT1, DCT2):
        used[t] = []
        for g in candidates[t]:
            n = int(c.detected[above, c.var_names.get_loc(g)].sum())
            ok = n >= MIN_CELLS_MARKER
            rows.append({"cell_type": t, "marker": g, "n_cells_detected_above_floor": n, "used": ok})
            if ok:
                used[t].append(g)
    return used, pd.DataFrame(rows)


def identity_scores(intensity: np.ndarray, detected: np.ndarray, ln_ids: np.ndarray, var_names: pd.Index,
                    used: dict[str, list[str]]) -> pd.DataFrame:
    """Type scores (mean residual z of the detected markers), counts and axis A for the given cells."""
    out = {}
    for t in (DCT1, DCT2):
        j = var_names.get_indexer(used[t])
        Z, _ = residual_z(intensity[:, j], detected[:, j], ln_ids, MIN_CELLS_MARKER)
        n = np.isfinite(Z).sum(axis=1)
        with np.errstate(invalid="ignore"):
            s = np.where(n > 0, np.nansum(np.nan_to_num(Z, nan=0.0), axis=1) / np.maximum(n, 1), np.nan)
        out[f"{t}_score"], out[f"n_{t}_markers_detected"] = s, n
    frame = pd.DataFrame(out)
    frame["axis"] = frame["DCT2_score"] - frame["DCT1_score"]
    return frame


def fit_mixture(axis: np.ndarray) -> dict:
    """Two-component GMM on the defined axis values; BIC for one and two components; calls."""
    from sklearn.mixture import GaussianMixture

    ok = np.isfinite(axis)
    a = axis[ok].reshape(-1, 1)
    g2 = GaussianMixture(n_components=2, n_init=GMM_N_INIT, random_state=SEED).fit(a)
    g1 = GaussianMixture(n_components=1, n_init=GMM_N_INIT, random_state=SEED).fit(a)
    lo = int(np.argmin(g2.means_.ravel()))
    post = g2.predict_proba(a)
    p_dct1, p_dct2 = post[:, lo], post[:, 1 - lo]
    call = np.where(p_dct1 >= POSTERIOR_MIN, DCT1, np.where(p_dct2 >= POSTERIOR_MIN, DCT2, UNRESOLVED))
    full_call = np.full(len(axis), UNKNOWN_MARKERS, dtype=object)
    full_call[ok] = call
    post_max = np.full(len(axis), np.nan)
    post_max[ok] = np.maximum(p_dct1, p_dct2)
    post2 = np.full(len(axis), np.nan)
    post2[ok] = p_dct2
    sd = np.sqrt(g2.covariances_.ravel())
    return {"call": full_call, "posterior": post_max, "posterior_dct2": post2,
            "bic_1": float(g1.bic(a)), "bic_2": float(g2.bic(a)), "n_fit": int(ok.sum()),
            "mean_dct1": float(g2.means_.ravel()[lo]), "mean_dct2": float(g2.means_.ravel()[1 - lo]),
            "sd_dct1": float(sd[lo]), "sd_dct2": float(sd[1 - lo]),
            "weight_dct1": float(g2.weights_[lo]), "weight_dct2": float(g2.weights_[1 - lo]),
            "converged": bool(g2.converged_), "one_component_mean": float(g1.means_.ravel()[0]),
            "one_component_sd": float(np.sqrt(g1.covariances_.ravel()[0]))}


# ---------------------------------------------------------------------------
# One full route-A pass
# ---------------------------------------------------------------------------
def route_a(c: Condition, floor: int, P: list[str], candidates: dict[str, list[str]], seed: int = SEED,
            intensity: np.ndarray | None = None) -> dict:
    """Route A for one condition and floor: proliferation (seed = null seed) and identity. Per-cell frame over all cells."""
    pr = prolif_setup(c, floor, P)
    a = pr.above
    res = cycling_call(pr, pr.P_idx, np.random.default_rng(seed))
    ipart, fits = intensity_part(c, pr)
    used, mtab = identity_markers(c, a, candidates)
    inten = c.intensity if intensity is None else intensity
    ident = identity_scores(inten[a], c.detected[a], c.ln_ids[a], c.var_names, used)
    mix = fit_mixture(ident["axis"].to_numpy(float))

    n = len(c.cell_ids)
    frame = pd.DataFrame({"cell_id": c.cell_ids, "cell_key": c.obs["cell_key"].astype(str).to_numpy(), "plate": c.plate,
                          "n_detected": c.ids.astype(int), "above_floor": a})
    def put(values, fill=np.nan, dtype=float):
        out = np.full(n, fill, dtype=dtype)
        out[a] = values
        return out
    frame["depth_bin"] = put(pr.bins, -1, int)
    frame["n_P_detected"] = put(c.detected[a][:, pr.P_idx].sum(axis=1), -1, int)
    frame["prolif_D"] = put(res["D"])
    frame["prolif_null_mean"] = put(res["null_mean"])
    frame["prolif_null_sd"] = put(res["null_sd"])
    frame["prolif_p"] = put(res["p"])
    frame["prolif_q"] = put(res["q"])
    frame["prolif_intensity"] = put(ipart)
    ncc = c.detected[:, c.var_names.get_loc(NCC)]
    held = c.detected[:, c.idx(HELD_OUT)].any(axis=1)
    frame["slc12a3_detected"] = ncc
    frame["mki67_or_top2a_detected"] = held
    cyc = put(res["cycling"], False, bool)
    frame["cycling"] = np.where(~a, BELOW, np.where(cyc, CYCLING, NON_CYCLING))
    for k in ("DCT1_score", "DCT2_score", "axis"):
        frame[k] = put(ident[k].to_numpy(float))
    for t in (DCT1, DCT2):
        frame[f"n_{t}_markers_detected"] = put(ident[f"n_{t}_markers_detected"].to_numpy(), -1, int)
    frame["posterior"] = put(mix["posterior"])
    frame["posterior_dct2"] = put(mix["posterior_dct2"])
    frame["identity"] = np.where(~a, BELOW, put(mix["call"], "", object))
    label = frame["identity"].to_numpy(object).copy()
    label[cyc & ncc] = PROLIF
    label[cyc & ~ncc] = CYCLING_NO_NCC
    frame["label"] = label
    return {"frame": frame, "prolif": pr, "cycling": res, "fits": fits, "markers_used": used, "marker_table": mtab,
            "mixture": mix}


def labels_from(frame: pd.DataFrame, cycling: np.ndarray) -> np.ndarray:
    """cs_t31_label from an identity column and a cycling call over the above cells (same rule as route_a)."""
    a = frame["above_floor"].to_numpy(bool)
    ncc = frame["slc12a3_detected"].to_numpy(bool)
    cyc = np.zeros(len(frame), bool)
    cyc[a] = cycling
    label = frame["identity"].to_numpy(object).copy()
    label[cyc & ncc] = PROLIF
    label[cyc & ~ncc] = CYCLING_NO_NCC
    return label


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def unique_order(*orders) -> tuple[str, ...]:
    """Concatenated label orders without repeats (cc.ordered_labels lists a label once per occurrence in the order)."""
    return tuple(dict.fromkeys(label for order in orders for label in order))


def counts_text(values, order=LABEL_ORDER) -> str:
    s = pd.Series(np.asarray(values).astype(str)).value_counts()
    return "; ".join(f"{k} {int(s[k])}" for k in cc.ordered_labels(s.index, unique_order(order)))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def agreement(a, b) -> dict:
    from sklearn.metrics import adjusted_rand_score

    a, b = np.asarray(a).astype(str), np.asarray(b).astype(str)
    return {"n_cells": len(a), "n_equal": int((a == b).sum()), "fraction_equal": float((a == b).mean()) if len(a) else np.nan,
            "ari": float(adjusted_rand_score(a, b)) if len(a) else np.nan}


def crosstab_long(rows, cols, row_name: str, col_name: str, **keys) -> list[dict]:
    ct = pd.crosstab(pd.Series(np.asarray(rows).astype(str), name=row_name),
                     pd.Series(np.asarray(cols).astype(str), name=col_name))
    order = unique_order(LABEL_ORDER, cc.DCT_LABEL_ORDER, (t18.BELOW_MIN, CYCLING, NON_CYCLING))
    return [{**keys, row_name: r, col_name: c, "n_cells": int(ct.loc[r, c])}
            for r in cc.ordered_labels(ct.index, order) for c in cc.ordered_labels(ct.columns, order)]


def logit_or(y: np.ndarray, ln_ids: np.ndarray, plate: np.ndarray) -> dict:
    """Logistic y ~ z(ln n_detected) + plate (P2 vs P1): odds ratios with 95% CI (statsmodels)."""
    import warnings

    import statsmodels.api as sm

    y = np.asarray(y, float)
    out = {"n_cells": int(len(y)), "n_positive": int(y.sum()), "computable": False, "reason": ""}
    if y.sum() == 0 or y.sum() == len(y):
        out["reason"] = "no positive cells" if y.sum() == 0 else "all cells positive"
        return out
    z = (ln_ids - ln_ids.mean()) / ln_ids.std(ddof=1)
    X = sm.add_constant(np.column_stack([z, (np.asarray(plate) == "P2").astype(float)]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as exc:          # perfect separation, non-convergence
        out["reason"] = f"fit failed: {type(exc).__name__}: {exc}"[:300]
        return out
    ci = np.exp(fit.conf_int(0.05))
    out.update(computable=True, depth_or=float(np.exp(fit.params[1])), depth_ci_low=float(ci[1, 0]),
               depth_ci_high=float(ci[1, 1]), depth_p=float(fit.pvalues[1]), plate_or=float(np.exp(fit.params[2])),
               plate_ci_low=float(ci[2, 0]), plate_ci_high=float(ci[2, 1]), plate_p=float(fit.pvalues[2]),
               pseudo_r2=float(fit.prsquared), converged=bool(fit.mle_retvals.get("converged", True)))
    return out


def table_path(name: str) -> Path:
    return cc.out_path(TASK, "tables", name, ".csv")


def read_table(run: cc.Run, name: str, **kwargs) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run the dct_prolif step that writes it first.")
    return pd.read_csv(run.use_input(path, "dct_prolif output"), **kwargs)


def stop(run: cc.Run, checks: t5.Checks, name: str) -> None:
    run.table(checks.frame(), name)
    run.finish({"checks_pass": False}, status="stopped_failed_checks")
    raise SystemExit(f"A check failed; see dct_prolif_{name}.csv.")


def freeze_or_stop(run: cc.Run, checks: t5.Checks, name: str) -> None:
    """No planning document is part of the public code; nothing is checked here."""
    return
