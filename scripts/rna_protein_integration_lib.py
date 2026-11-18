"""rna_protein_integration shared definitions and functions. Runs no analysis by itself.

Imported by the ``rna_protein_integration_*.py`` step scripts.
Plan: ``the analysis plan`` (R1 integration bullet).

Design approved by the authors (15 Oct 2026)
-----------------------------------------
- Shared features: the 3,569 atlas genes matching a quantified kidney protein (kidney_rna rule, exact-name protein entry
  for duplicated symbols). Sensitivity: features detected in >= 10% of SCP cells.
- RNA: atlas ``.raw.X`` counts, 10,000 per cell over all 16,711 genes, natural log1p (as kidney_rna); shared genes;
  z-scored per gene within the RNA reference cells of the run (ddof 1, clip +/- 10, as ``sc.pp.scale``).
- SCP: canonical ``layers['ln1p_intensity']`` (missing = 0) on the shared proteins, z-scored per protein across
  the 712 cells (ddof 1, clip +/- 10); equals the canonical ``.X`` columns (checked).
- Balanced RNA subsample: 200 cells per atlas ``Celltype`` (35 classes, Unknown excluded; smaller classes whole),
  5 draws (seeds 42-46). Sensitivity design: atlas nuclei only (same rule).
- Embeddings: (A) joint PCA, 30 PCs, loadings from the average of the two within-modality covariance matrices
  (equal weight); (B) A + Harmony (harmonypy 0.0.10, batch = modality); (C) the atlas's stored ``X_scVI`` with
  each SCP cell placed at the mean position of its 15 nearest RNA cells in shared-gene space (cosine on the
  z-scored features). Sensitivities: plain concatenated PCA; Harmony theta 2; reference limited to the SCP
  counterpart classes; features >= 10% SCP detection; nuclei only; k = 5 / 30 for transfer.
- Labels: ``kidney_t15_label`` (primary; kidney_annotation file) and ``kidney_pre_label`` (Fig 2A). Atlas level ``Celltype``
  primary, ``Celltype_finest`` secondary (transfer only). Expected atlas classes per label: the approved kidney_rna
  mapping table.

Contents
--------
- Paths and settings; ``read_mapping`` / ``expected_sets`` / ``inverse_map``; ``common_labels``.
- Matrices: ``zscore`` (as ``sc.pp.scale``), ``covariance_pca``, ``run_harmony``, ``project_by_knn``.
- Metrics: ``lisi`` (harmonypy ``compute_lisi``), ``compartment_mixing``, ``knn_vote``, ``transfer_metrics``,
  ``separation``.
- Figure helpers are imported from ``atlas_dct_lib`` (``Sheet``, ``save_checked``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg, sparse

import scp_common as cc
import cluster_de_lib as t5
import kidney_rna_lib as L7

TASK = "rna_protein_integration"
SEED = cc.RANDOM_SEED
DATASET = "kidney"

# Atlas ----------------------------------------------------------------------------------------------
ATLAS = L7.ATLAS
ATLAS_EXPECTED_SHAPE = L7.ATLAS_EXPECTED_SHAPE
ATLAS_EXPECTED_RAW_GENES = L7.ATLAS_EXPECTED_RAW_GENES
TARGET_SUM = L7.TARGET_SUM
LEVEL = "Celltype"
FINEST = "Celltype_finest"
LEVELS = {"celltype": "Celltype", "finest": "Celltype_finest"}
ATLAS_UNKNOWN = L7.ATLAS_UNKNOWN
OBS_KEEP = ("Origin", "Source", "Technology", "Tissue_res", "Age", "batch", "Celltype", "Celltype_finest_lowres",
            "Celltype_finest", "total_counts", "n_genes_by_counts")
KIDNEY_RNA_LEVEL_COUNTS = cc.out_path(L7.TASK, "tables", "atlas_level_counts", ".csv")
KIDNEY_RNA_SHARED = cc.out_path(L7.TASK, "tables", "shared_features", ".csv")
KIDNEY_RNA_MAPPING = cc.out_path(L7.TASK, "tables", "celltype_mapping", ".csv")
KIDNEY_RNA_MAPPING_SUMMARY = cc.out_path(L7.TASK, "tables", "mapping_summary", ".csv")
SUBSAMPLE_H5AD = cc.out_path(TASK, "h5ad", "rna_subsamples", ".h5ad")
JOINT_H5AD = cc.out_path(TASK, "h5ad", "joint_embedding", ".h5ad")

# Design ----------------------------------------------------------------------------------------------
N_PER_TYPE = 200
SUBSAMPLE_SEEDS = (42, 43, 44, 45, 46)
PRIMARY_DRAW = 42
DESIGNS = {"all": None, "nuclei": "Nuclei"}
DESIGN_TEXT = {"all": "all atlas cells and nuclei", "nuclei": "atlas nuclei only"}
CLIP = 10.0
N_PCS = 30
K_GRAPH = 15
K_TRANSFER = 15
K_TRANSFER_SENS = (5, 30)
K_PROJECT = 15
LISI_PERPLEXITY = 30          # global LISI (90 neighbours)
MIX_PERPLEXITY = 5            # within-compartment LISI (15 neighbours)
K_MIX = 15
MIN_COMPARTMENT_RNA = 10
FEATURE_SENS_DETECTION = 0.10
HARMONY_THETA_DEFAULT = 1.0   # harmonypy 0.0.10 default when theta is None
UMAP_MIN_DIST = 0.5           # scanpy default

MODALITY = {"SCP": "SCP (protein)", "RNA": "RNA (atlas)"}
EMBEDDINGS = {"pca": "A: joint PCA", "harmony": "B: PCA + Harmony", "scvi": "C: atlas scVI (SCP projected)"}
EMB_SHORT = {"pca": "A", "harmony": "B", "scvi": "C"}
SHARED_KNN = "shared"         # transfer directly by cosine kNN in shared-gene space (basis of C)

CONFIGS = {
    "primary": {"design": "all", "features": "all", "pca": "equal", "reference": "all", "theta": None},
    "features_scp10": {"design": "all", "features": "scp10", "pca": "equal", "reference": "all", "theta": None},
    "plain_pca": {"design": "all", "features": "all", "pca": "plain", "reference": "all", "theta": None},
    "harmony_theta2": {"design": "all", "features": "all", "pca": "equal", "reference": "all", "theta": 2.0},
    "counterpart_reference": {"design": "all", "features": "all", "pca": "equal", "reference": "counterpart",
                              "theta": None},
    "nuclei": {"design": "nuclei", "features": "all", "pca": "equal", "reference": "all", "theta": None},
}
CONFIG_TEXT = {
    "primary": "primary: all atlas cells, 3,569 features, equal-weight PCA, Harmony theta 1, all 35 classes",
    "features_scp10": "features detected in >= 10% of SCP cells",
    "plain_pca": "plain concatenated PCA (RNA cells set the axes in proportion to their number)",
    "harmony_theta2": "Harmony theta 2 (R Harmony default)",
    "counterpart_reference": "RNA reference limited to the expected classes of the label set",
    "nuclei": "atlas nuclei only",
}

# Labels ----------------------------------------------------------------------------------------------
LABEL_COLUMNS = ("kidney_t15_label", "kidney_pre_label")
LABEL_DISPLAY = {"kidney_t15_label": "kidney_annotation labels (kidney_t15_label)", "kidney_pre_label": "Fig 2A labels (kidney_pre_label)"}
LABEL_SHORT = {"kidney_t15_label": "t15", "kidney_pre_label": "pre"}
EXPECTED_COUNTS = {k: L7.EXPECTED_COUNTS[k] for k in LABEL_COLUMNS}

# Transporter / channel rule (gene-symbol prefixes; not a curated annotation) ----------------------------------
TRANSPORTER_PREFIXES = ("Slc", "Atp", "Abc", "Aqp", "Kcn", "Scn", "Cacn", "Clcn", "Clca", "Trpa", "Trpc", "Trpm", "Trpv",
                        "Hcn", "Cftr", "Pkd2", "Piezo", "Ano", "Best", "Ttyh", "Lrrc8", "Asic", "Gja", "Gjb", "Gjc",
                        "Gjd", "Ryr", "Itpr", "Orai", "Mcoln", "Tpcn", "Nalcn", "Vdac")
# Symbols with a listed prefix that are not transporters or channels (ATP synthase and assembly factors, ABC
# family members without transport function, channel-interacting or splicing proteins, anosmin).
TRANSPORTER_EXCLUDE_PREFIXES = ("Atp5", "Atpaf", "Atpif", "Atp6ap", "Atp23", "Atps", "Abce", "Abcf", "Anos", "Scnm",
                                "Kcnip", "Itpri")


def transporter_channel_rule(symbol: str) -> str:
    """Matching prefix ('' if none or excluded)."""
    s = str(symbol)
    if any(s.startswith(p) for p in TRANSPORTER_EXCLUDE_PREFIXES):
        return ""
    hits = [p for p in TRANSPORTER_PREFIXES if s.startswith(p)]
    return max(hits, key=len) if hits else ""


def is_transporter_channel(symbol: str) -> bool:
    return transporter_channel_rule(symbol) != ""


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
def _split(v) -> set[str]:
    return set() if pd.isna(v) or str(v).strip() == "" else set(str(v).split(";"))


def read_mapping(run: cc.Run, checks: t5.Checks | None = None) -> pd.DataFrame:
    mapping = pd.read_csv(run.use_input(KIDNEY_RNA_MAPPING, "kidney_rna approved SCP label -> atlas mapping (step 2)"))
    if checks is not None:
        approved = mapping["status"].astype(str).str.contains("approved by the authors").all()
        checks.add("kidney_rna mapping table approved by the authors (all rows)", f"{approved}", approved)
    return mapping


def expected_sets(mapping: pd.DataFrame, label_col: str, level: str) -> dict[str, tuple[set[str], set[str]]]:
    """SCP label -> (expected atlas classes, expected plus accepted mixed classes) at 'celltype' or 'finest'."""
    exp_col, mix_col = (("atlas_celltype_expected", "atlas_celltype_mixed_accepted") if level == "celltype" else
                        ("atlas_finest_expected", "atlas_finest_mixed_accepted"))
    sub = mapping[mapping["label_column"] == label_col]
    out = {}
    for r in sub.itertuples():
        exp = _split(getattr(r, exp_col))
        out[str(r.scp_label)] = (exp, exp | _split(getattr(r, mix_col)))
    return out


def inverse_map(sets: dict[str, tuple[set[str], set[str]]]) -> dict[str, str]:
    """Atlas class -> SCP label (expected classes only); raises if one atlas class is expected for two labels."""
    inv: dict[str, str] = {}
    for lab, (exp, _) in sets.items():
        for c in exp:
            if c in inv and inv[c] != lab:
                raise ValueError(f"atlas class {c} is expected for both {inv[c]} and {lab}")
            inv[c] = lab
    return inv


def counterpart_labels(sets) -> list[str]:
    return [lab for lab, (exp, _) in sets.items() if exp]


def common_labels(scp_labels: pd.Series, rna_celltype: pd.Series, inv: dict[str, str]) -> np.ndarray:
    """One label space for both modalities: SCP label for SCP cells; for RNA cells the SCP label whose expected
    classes include the cell's atlas Celltype, else 'atlas: <Celltype>'."""
    rna = np.array([inv.get(c, f"atlas: {c}") for c in rna_celltype.astype(str)], dtype=object)
    return np.concatenate([scp_labels.astype(str).to_numpy(dtype=object), rna])


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------
def lognorm_rows(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    """counts / row total x 10,000 (totals over all columns given), natural log1p; float32 CSR (as kidney_rna/atlas_dct)."""
    m = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    tot = np.asarray(m.sum(axis=1), dtype=np.float64).ravel()
    m.data *= np.repeat((TARGET_SUM / np.where(tot > 0, tot, 1)).astype(np.float32), np.diff(m.indptr))
    np.log1p(m.data, out=m.data)
    return m


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-column z-score (ddof 1; zero SD -> 1) clipped at +/- 10, as sc.pp.scale (= scp_common)."""
    return cc.scale_like_scanpy(x, CLIP)


def covariance_pca(blocks: list[np.ndarray], mode: str, n_pcs: int = N_PCS) -> tuple[np.ndarray, np.ndarray]:
    """Loadings (features x n_pcs) and eigenvalues from the modality covariance (about zero; each block is
    z-scored within its modality). mode 'equal': average of the per-block covariance matrices; 'plain': covariance
    of the concatenated rows (blocks weighted by their number of cells). Sign: largest |loading| positive."""
    g = blocks[0].shape[1]
    if mode == "equal":
        cov = sum(b.T @ b / (b.shape[0] - 1) for b in blocks) / len(blocks)
    elif mode == "plain":
        n = sum(b.shape[0] for b in blocks)
        cov = sum(b.T @ b for b in blocks) / (n - 1)
    else:
        raise ValueError(mode)
    vals, vecs = linalg.eigh(cov, subset_by_index=[g - n_pcs, g - 1])
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    flip = np.sign(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(vecs.shape[1])])
    return vecs * flip, vals


def run_harmony(z: np.ndarray, modality: np.ndarray, seed: int = SEED, theta: float | None = None) -> tuple[np.ndarray, dict]:
    """harmonypy 0.0.10 run_harmony with batch = modality; returns corrected coordinates (cells x PCs) and info."""
    import harmonypy

    logging.getLogger("harmonypy").setLevel(logging.WARNING)
    meta = pd.DataFrame({"modality": pd.Categorical(np.asarray(modality, dtype=str))})
    ho = harmonypy.run_harmony(np.ascontiguousarray(z, dtype=np.float64), meta, ["modality"], theta=theta,
                               random_state=seed, verbose=False)
    obj = list(ho.objective_harmony)
    n_iter = len(obj) - 1
    converged = n_iter < ho.max_iter_harmony or (len(obj) >= 2 and (obj[-2] - obj[-1]) / abs(obj[-2]) < ho.epsilon_harmony)
    info = {"harmony_iterations": n_iter, "harmony_converged": bool(converged), "harmony_clusters": int(ho.K),
            "harmony_theta": float(np.ravel(ho.theta)[0]), "harmony_sigma": float(np.ravel(ho.sigma)[0]),
            "harmony_lambda": float(np.diag(ho.lamb)[1]), "harmony_objective_last": float(obj[-1])}
    return np.asarray(ho.Z_corr, dtype=np.float64).T, info


def cosine_knn(query: np.ndarray, ref: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine distances of the k nearest reference rows for every query row (ties by index)."""
    q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    r = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-12)
    sims = q @ r.T
    idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return idx, 1 - np.take_along_axis(sims, idx, axis=1)


def project_by_knn(scp_z: np.ndarray, rna_z: np.ndarray, rna_latent: np.ndarray, k: int = K_PROJECT):
    """Each SCP cell placed at the mean latent position of its k nearest RNA cells in shared-gene space."""
    idx, dist = cosine_knn(scp_z, rna_z, k)
    return rna_latent[idx].mean(axis=1), idx, dist


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def lisi(emb: np.ndarray, meta: pd.DataFrame, cols: list[str], perplexity: int) -> pd.DataFrame:
    import harmonypy

    vals = harmonypy.compute_lisi(np.ascontiguousarray(emb, dtype=np.float64), meta.reset_index(drop=True), cols,
                                  perplexity=int(perplexity))
    return pd.DataFrame(vals, columns=cols, index=meta.index)


def max_ilisi(p: float) -> float:
    """Inverse Simpson index of two groups with shares p and 1 - p (perfect mixing)."""
    return 1.0 / (p ** 2 + (1 - p) ** 2)


def compartment_mixing(emb: np.ndarray, is_scp: np.ndarray, mask: np.ndarray, k: int = K_MIX,
                       perplexity: int = MIX_PERPLEXITY) -> dict | None:
    """Modality mixing among the cells of one compartment (SCP label + RNA cells of its expected classes):
    within-compartment kNN; per cell iLISI (perplexity 5) and the fraction of other-modality neighbours."""
    from sklearn.neighbors import NearestNeighbors

    idx = np.flatnonzero(mask)
    s = is_scp[idx]
    n_s, n_r = int(s.sum()), int((~s).sum())
    if n_s < 2 or n_r < MIN_COMPARTMENT_RNA or len(idx) <= 3 * perplexity + 1:
        return None
    e = emb[idx]
    meta = pd.DataFrame({"modality": np.where(s, "SCP", "RNA")})
    il = lisi(e, meta, ["modality"], perplexity)["modality"].to_numpy()
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(e)
    _, nb = nn.kneighbors(e)
    nb = nb[:, 1:]
    other = (s[nb] != s[:, None]).mean(axis=1)
    p = n_s / (n_s + n_r)
    imax = max_ilisi(p)
    exp_other_scp = n_r / (n_s + n_r - 1)
    exp_other_rna = n_s / (n_s + n_r - 1)
    # iLISI is reported scib-style, (median - 1) / (2 batches - 1), next to the perfect-mixing value for the
    # compartment's modality shares (a ratio to that value is unstable when the SCP share is small).
    return {"n_scp": n_s, "n_rna": n_r, "scp_share": p, "ilisi_perfect_mixing": imax,
            "median_ilisi_scp": float(np.median(il[s])), "median_ilisi_rna": float(np.median(il[~s])),
            "median_ilisi_all": float(np.median(il)),
            "ilisi_scib_scp": float(np.median(il[s]) - 1), "ilisi_scib_all": float(np.median(il) - 1),
            "ilisi_perfect_scib": float(imax - 1),
            "other_modality_fraction_scp": float(other[s].mean()),
            "other_modality_fraction_scp_expected": exp_other_scp,
            "scaled_other_fraction_scp": float(other[s].mean() / exp_other_scp),
            "other_modality_fraction_rna": float(other[~s].mean()),
            "scaled_other_fraction_rna": float(other[~s].mean() / exp_other_rna),
            "_per_cell": pd.DataFrame({"pos": idx, "ilisi": il, "other_fraction": other})}


def knn_vote(ref: np.ndarray, ref_labels: np.ndarray, query: np.ndarray, k: int, metric: str = "euclidean"):
    """Majority vote of the k nearest reference cells; ties go to the tied class with the smallest mean distance."""
    from sklearn.neighbors import NearestNeighbors

    if metric == "cosine":
        nb, dist = cosine_knn(query, ref, k)
    else:
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(ref)
        dist, nb = nn.kneighbors(query)
    lab = np.asarray(ref_labels, dtype=object)[nb]
    pred, frac = [], []
    for row_lab, row_d in zip(lab, dist):
        vals, counts = np.unique(row_lab, return_counts=True)
        best = counts.max()
        tied = vals[counts == best]
        if len(tied) > 1:
            md = [row_d[row_lab == t].mean() for t in tied]
            choice = tied[int(np.argmin(md))]
        else:
            choice = tied[0]
        pred.append(choice)
        frac.append(best / k)
    return np.asarray(pred, dtype=object), np.asarray(frac)


def transfer_metrics(pred: pd.Series, labels: pd.Series, sets, ref_classes: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Accuracy (strict / incl. mixed), macro-F1 over labels with a counterpart, chance accuracy; per-label rows.

    pred: predicted atlas class per SCP cell; labels: SCP label per cell; sets: expected_sets(); ref_classes:
    atlas class of every reference cell (for the chance level)."""
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    inv = inverse_map(sets)
    cps = counterpart_labels(sets)
    m = labels.isin(cps).to_numpy()
    lab = labels[m].astype(str).to_numpy()
    pr = pred[m].astype(str).to_numpy()
    strict = np.array([p in sets[l][0] for l, p in zip(lab, pr)])
    mixed = np.array([p in sets[l][1] for l, p in zip(lab, pr)])
    mapped = np.array([inv.get(p, "other") for p in pr], dtype=object)
    freq = ref_classes.astype(str).value_counts(normalize=True)
    chance_l = {l: float(freq.reindex(list(sets[l][0])).fillna(0).sum()) for l in cps}
    chance = float(np.mean([chance_l[l] for l in lab]))
    macro = float(f1_score(lab, mapped, labels=cps, average="macro", zero_division=0))
    p_, r_, f_, s_ = precision_recall_fscore_support(lab, mapped, labels=cps, zero_division=0)
    rows = []
    for i, l in enumerate(cps):
        mm = lab == l
        rows.append({"scp_label": l, "n_cells": int(mm.sum()), "expected_atlas": ";".join(sorted(sets[l][0])),
                     "accuracy_strict": float(strict[mm].mean()) if mm.any() else np.nan,
                     "accuracy_incl_mixed": float(mixed[mm].mean()) if mm.any() else np.nan,
                     "precision": float(p_[i]), "recall": float(r_[i]), "f1": float(f_[i]),
                     "chance_accuracy_strict": chance_l[l]})
    for l in sorted(set(labels.astype(str)) - set(cps)):
        mm = (labels.astype(str) == l).to_numpy()
        vc = pred[mm].astype(str).value_counts()
        rows.append({"scp_label": l, "n_cells": int(mm.sum()), "expected_atlas": "",
                     "top_predicted": "; ".join(f"{k} {v}" for k, v in vc.head(3).items())})
    summary = {"n_cells_with_counterpart": int(m.sum()), "accuracy_strict": float(strict.mean()),
               "accuracy_incl_mixed": float(mixed.mean()), "macro_f1": macro, "chance_accuracy_strict": chance,
               "n_labels_macro_f1": len(cps)}
    return summary, pd.DataFrame(rows)


def separation(emb: np.ndarray, common: np.ndarray, is_scp: np.ndarray, keep: np.ndarray) -> dict:
    """Silhouette (euclidean) of the common labels: joint, RNA cells only, SCP cells only (cells in ``keep``)."""
    from sklearn.metrics import silhouette_score

    out = {}
    for name, m in (("joint", keep), ("rna", keep & ~is_scp), ("scp", keep & is_scp)):
        labs = common[m]
        out[f"silhouette_{name}"] = (float(silhouette_score(emb[m], labs, metric="euclidean"))
                                     if len(set(labs)) > 1 else np.nan)
    return out


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class ScpData:
    labels: pd.DataFrame                 # 712 rows (cell_id index): label columns, n_detected_proteins
    proteins: list[str]                  # shared proteins (atlas gene order of the kidney_rna table)
    genes: list[str]                     # matching atlas genes
    z: np.ndarray                        # 712 x 3,569 z-scored ln(1 + intensity)
    detection: np.ndarray                # fraction of SCP cells with intensity > 0 per shared protein
    var_names: list[str]                 # all 4,054 canonical protein names


def load_scp(run: cc.Run, checks: t5.Checks, shared: pd.DataFrame) -> ScpData:
    """Canonical kidney object (kidney_object) + kidney_annotation labels; SCP matrix on the shared proteins, checked against .X."""
    adata, labels = L7.load_kidney(run, checks)
    names = pd.Index(adata.var_names.astype(str))
    pidx = names.get_indexer(shared["protein"])
    if (pidx < 0).any():
        raise KeyError("shared proteins missing from the canonical object")
    ln1p = cc.dense(adata.layers["ln1p_intensity"]).astype(np.float64)
    z_all = zscore(ln1p)
    z = z_all[:, pidx]
    stored = cc.dense(adata.X).astype(np.float64)[:, pidx]
    diff = float(np.max(np.abs(z - stored)))
    checks.add("SCP matrix (ln(1 + intensity) z-scored per protein, ddof 1, clip 10) equals the canonical .X on the "
               "shared proteins (max abs difference)", f"{diff:.2e}", diff < 1e-4, "< 1e-4")
    det = (cc.dense(adata.layers["intensity"])[:, pidx] > 0).mean(axis=0)
    return ScpData(labels, shared["protein"].tolist(), shared["atlas_gene"].tolist(), z, det, names.tolist())


def read_shared(run: cc.Run, checks: t5.Checks | None = None) -> pd.DataFrame:
    sh = pd.read_csv(run.use_input(KIDNEY_RNA_SHARED, "kidney_rna shared features (protein <-> atlas gene)"))
    if checks is not None:
        checks.add("shared features (kidney_rna table)", f"{len(sh)}", len(sh) == L7.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES,
                   f"= {L7.KIDNEY_ANNOTATION_ATLAS_SHARED_GENES}")
    return sh


@dataclass
class RnaPool:
    obs: pd.DataFrame                    # union of the drawn cells (atlas barcode index)
    ln: sparse.csr_matrix                # log-normalized expression on the shared genes (all-gene totals)
    scvi: np.ndarray                     # stored atlas X_scVI
    genes: list[str]
    draws: pd.DataFrame                  # design, seed, atlas_cell


def load_pool(run: cc.Run) -> RnaPool:
    import anndata as ad

    if not SUBSAMPLE_H5AD.exists():
        raise FileNotFoundError(f"{cc.rel(SUBSAMPLE_H5AD)} not found; run rna_protein_integration_prepare.py first.")
    a = ad.read_h5ad(run.use_input(SUBSAMPLE_H5AD, "rna_protein_integration RNA subsample pool (rna_protein_integration_prepare.py)"))
    draws = pd.read_csv(run.use_input(cc.out_path(TASK, "tables", "rna_subsample", ".csv"),
                                      "rna_protein_integration RNA subsample draws (rna_protein_integration_prepare.py)"), dtype={"atlas_cell": str})
    return RnaPool(a.obs.copy(), sparse.csr_matrix(a.X), np.asarray(a.obsm["X_scVI"], dtype=np.float64),
                   a.var_names.astype(str).tolist(), draws)


# ---------------------------------------------------------------------------
# One run: reference cells, embeddings
# ---------------------------------------------------------------------------
@dataclass
class Joint:
    config: str
    draw: int
    reference: str                       # 'all' or a label column (counterpart reference)
    obs: pd.DataFrame                    # SCP rows first, then RNA rows
    is_scp: np.ndarray
    features: list[str]                  # atlas genes used
    z: np.ndarray                        # cells x features (z-scored within modality)
    loadings: np.ndarray
    eigenvalues: np.ndarray
    emb: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    project_idx: np.ndarray | None = None


def reference_cells(pool: RnaPool, design: str, draw: int, restrict: set[str] | None = None) -> np.ndarray:
    """Positions in the pool of the cells of one draw (optionally restricted to some atlas Celltypes)."""
    d = pool.draws[(pool.draws["design"] == design) & (pool.draws["seed"] == draw)]
    pos = pool.obs.index.get_indexer(d["atlas_cell"].astype(str))
    if (pos < 0).any():
        raise KeyError("drawn cells missing from the pool")
    if restrict is not None:
        pos = pos[np.isin(pool.obs[LEVEL].astype(str).to_numpy()[pos], sorted(restrict))]
    return np.sort(pos)


def build_joint(scp: ScpData, pool: RnaPool, config: str, draw: int, mapping: pd.DataFrame,
                reference: str = "all", feature_mask: np.ndarray | None = None, scvi: bool = True) -> Joint:
    cfg = CONFIGS[config]
    restrict = None
    if reference != "all":
        restrict = set().union(*[e for e, _ in expected_sets(mapping, reference, "celltype").values()])
    pos = reference_cells(pool, cfg["design"], draw, restrict)
    fm = np.ones(len(scp.genes), dtype=bool) if feature_mask is None else feature_mask
    gidx = pd.Index(pool.genes).get_indexer(np.asarray(scp.genes)[fm])
    if (gidx < 0).any():
        raise KeyError("shared genes missing from the pool")
    rna_ln = pool.ln[pos][:, gidx].toarray().astype(np.float64)
    rna_z = zscore(rna_ln)
    # SCP z-scores are recomputed on the subset of features only through column selection (per-feature scaling).
    scp_z = scp.z[:, fm]
    load, vals = covariance_pca([scp_z, rna_z], cfg["pca"])
    z = np.vstack([scp_z, rna_z])
    pcs = z @ load
    obs_s = scp.labels[["kidney_t15_label", "kidney_pre_label", "n_detected_proteins"]].copy()
    obs_s["modality"] = "SCP"
    obs_s["cell"] = obs_s.index.astype(str)
    obs_r = pool.obs.iloc[pos][[c for c in OBS_KEEP if c in pool.obs.columns]].copy()
    obs_r["modality"] = "RNA"
    obs_r["cell"] = obs_r.index.astype(str)
    obs = pd.concat([obs_s, obs_r], axis=0, ignore_index=True)
    for c in obs.columns:
        if obs[c].dtype == object or str(obs[c].dtype).startswith("string") or str(obs[c].dtype) == "category":
            obs[c] = obs[c].astype(str).replace({"nan": "", "<NA>": "", "None": ""})
    is_scp = (obs["modality"] == "SCP").to_numpy()
    j = Joint(config, draw, reference, obs, is_scp, list(np.asarray(scp.genes)[fm]), z, load, vals)
    j.emb["pca"] = pcs
    j.emb["harmony"], hinfo = run_harmony(pcs, obs["modality"].to_numpy(), SEED, cfg["theta"])
    j.info.update(hinfo)
    if scvi:
        lat_r = pool.scvi[pos]
        lat_s, nb, _ = project_by_knn(scp_z, rna_z, lat_r, K_PROJECT)
        j.emb["scvi"] = np.vstack([lat_s, lat_r])
        j.project_idx = nb
    j.info.update({"n_scp": int(is_scp.sum()), "n_rna": int((~is_scp).sum()), "n_features": int(fm.sum()),
                   "pca_mode": cfg["pca"], "design": cfg["design"], "reference": reference})
    return j


def pc_modality_variance(j: Joint) -> pd.DataFrame:
    """Per PC: variance of the scores within each modality and its share of the summed within-modality variance."""
    pcs = j.emb["pca"]
    vs = pcs[j.is_scp].var(axis=0, ddof=1)
    vr = pcs[~j.is_scp].var(axis=0, ddof=1)
    return pd.DataFrame({"pc": np.arange(1, pcs.shape[1] + 1), "eigenvalue": j.eigenvalues, "var_scp": vs,
                         "var_rna": vr, "share_scp": vs / (vs + vr)})
