"""qc_mito shared definitions and functions. Runs no analysis by itself.

Imported by the
``qc_mito_*.py`` step scripts.

Contents
--------
- Dataset and label-column settings (labels only from the kidney_object canonical objects).
- MitoCarta3.0 matching of protein groups (gene symbols; UniProt as a cross-check).
- Per-cell mitochondrial metrics from ``layers['intensity']`` (raw PG.Quantity, missing = 0).
- Within-type MAD outlier flags, label contrasts (Mann-Whitney + OLS with HC3
  standard errors), type-level tests and Spearman intervals.
- Figure style and small plotting helpers (style copied from reviewer_core_lib.py).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

import scp_common as cc

TASK = "qc_mito"
SEED = cc.RANDOM_SEED
MIN_GROUP = 10          # minimum cells per label for MAD flags and within-type correlations
N_MAD = 3.0             # outlier rule: |x - median| > N_MAD * MAD (MAD scaled to the normal SD)

DATASETS = {
    "kidney": {"display": "KIDNEY", "label_columns": ("kidney_reannot_label", "kidney_pre_label"),
               "plot_columns": ("kidney_reannot_label", "kidney_pre_label"), "order": cc.KIDNEY_LABEL_ORDER,
               "reference_label": "PCT"},
    "dct": {"display": "DCT", "label_columns": ("v1_cell_label", "v1_cluster_label"),
            "plot_columns": ("v1_cell_label", "v1_cluster_label", "plate"), "order": cc.DCT_LABEL_ORDER,
            "reference_label": "DCT1"},
}
LABEL_DISPLAY = {
    "kidney_reannot_label": "re-annotated label (Fig 2B)",
    "kidney_pre_label": "pre-annotation label (Fig 2A)",
    "v1_cell_label": "V1 per-cell label",
    "v1_cluster_label": "V1 cluster label",
    "plate": "plate (pooled sort plate)",
}
SHORT_COLUMN = {"kidney_reannot_label": "reannot", "kidney_pre_label": "pre", "v1_cell_label": "cell",
                "v1_cluster_label": "cluster", "plate": "plate"}

# ---------------------------------------------------------------------------
# Protein sets
# ---------------------------------------------------------------------------
MITOCARTA_CSV_NAME = "mitocarta_mouse_list"
SETS = {
    "mitocarta": "MitoCarta3.0, gene-symbol or UniProt-accession match (primary)",
    "mitocarta_symbol_only": "MitoCarta3.0, gene-symbol match only (sensitivity)",
    "oxphos": "MitoCarta3.0 MitoPathway 'OXPHOS' (subunits + assembly factors)",
    "oxphos_subunit": "MitoCarta3.0 MitoPathway 'OXPHOS > OXPHOS subunits'",
    "mt_encoded": "MitoCarta3.0 mt-encoded genes (13 'mt-' symbols)",
    "mt_prefix_task1": "Task 1 rule (the earlier analysis): first gene symbol starts with 'Mt-'/'mt-'",
}
SET_DISPLAY = {"mitocarta": "MitoCarta", "mitocarta_symbol_only": "MitoCarta (symbol only)", "oxphos": "OXPHOS",
               "oxphos_subunit": "OXPHOS subunits", "mt_encoded": "mt-encoded", "mt_prefix_task1": "Mt-/mt- prefix (Task 1)"}
MITO_METRICS = ("mitocarta_intensity_fraction", "mitocarta_id_fraction", "oxphos_intensity_fraction")
DEPTH_METRICS = ("log10_total_intensity", "log10_n_detected")
OUTLIER_METRICS = (*MITO_METRICS, *DEPTH_METRICS)
METRIC_DISPLAY = {
    "mitocarta_intensity_fraction": "MitoCarta fraction of summed intensity",
    "mitocarta_id_fraction": "MitoCarta fraction of protein IDs",
    "oxphos_intensity_fraction": "OXPHOS fraction of summed intensity",
    "oxphos_subunit_intensity_fraction": "OXPHOS-subunit fraction of summed intensity",
    "mt_encoded_intensity_fraction": "mt-encoded fraction of summed intensity",
    "mt_prefix_task1_intensity_fraction": "Mt-/mt- prefix fraction of summed intensity (Task 1)",
    "mitocarta_symbol_only_intensity_fraction": "MitoCarta (symbol match only) fraction of summed intensity",
    "log10_total_intensity": "log10 total intensity",
    "log10_n_detected": "log10 protein IDs",
    "n_detected_proteins": "protein IDs per cell",
    "n_detected_mitocarta": "detected MitoCarta proteins",
}

# Dissociation-stress panel (qc_mito plan, step 5), mouse capitalisation.
STRESS_PANEL = ["Hspa1a", "Hspa1b", "Hspa8", "Hsp90aa1", "Hsph1", "Dnajb1", "Hspb1", "Jun", "Junb", "Fos", "Fosb",
                "Atf3", "Egr1", "Ier2", "Ier3", "Zfp36", "Dusp1", "Socs3"]
STRESS_SCORES = ("stress_mean_z", "stress_score_genes", "stress_n_detected")
STRESS_DISPLAY = {"stress_mean_z": "Stress score: mean z (ln intensity)",
                  "stress_score_genes": "Stress score: score_genes",
                  "stress_n_detected": "Stress proteins detected (n)"}


def split_symbols(value: object) -> list[str]:
    """Gene symbols of a PG.Genes entry, lower-cased, unique, in order."""
    out: list[str] = []
    for part in str(value).split(";"):
        part = part.strip().replace(" ", "").lower()
        if part and part != "nan" and part not in out:
            out.append(part)
    return out


def split_accessions(value: object) -> list[str]:
    """UniProt accessions of a PG.ProteinGroups entry without isoform suffixes."""
    out: list[str] = []
    for part in str(value).split(";"):
        acc = re.sub(r"-\d+$", "", part.strip())
        if acc and acc != "nan" and acc not in out:
            out.append(acc)
    return out


def read_mitocarta(run: cc.Run) -> pd.DataFrame:
    path = cc.out_path(TASK, "tables", MITOCARTA_CSV_NAME, ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run qc_mito_mitocarta.py first.")
    frame = pd.read_csv(run.use_input(path, "qc_mito parsed MitoCarta3.0 mouse list"), dtype=str, keep_default_na=False)
    for col in ("is_oxphos", "is_oxphos_subunit", "is_mt_encoded"):
        frame[col] = frame[col].str.lower().eq("true")
    return frame


def annotate_proteins(meta: pd.DataFrame, mitocarta: pd.DataFrame) -> pd.DataFrame:
    """MitoCarta flags for protein groups.

    ``meta`` needs columns ``pg_genes`` (all PG.Genes symbols), ``protein_group``
    (PG.ProteinGroups), ``first_symbol`` (the matrix gene name used by Task 1)
    and ``protein_description``. A protein group is MitoCarta when any of its
    gene symbols equals a MitoCarta symbol (case-insensitive) or any of its
    UniProt accessions (isoform suffix removed) equals a MitoCarta UniProt
    accession (authors decision, 13 Oct 2026: symbols alone miss renamed genes such
    as Atp5f1a = Atp5a1 and the mt-encoded proteins named Mtco1, Mtnd1, Mt-Cyb).
    OXPHOS / OXPHOS-subunit / mt-encoded follow the matched MitoCarta genes.
    Synonym-only matches are recorded but never used.
    """
    symbol_of = {s.lower(): s for s in mitocarta["symbol"]}
    uniprot_of = {u: s for u, s in zip(mitocarta["uniprot"], mitocarta["symbol"]) if u}
    synonym_of: dict[str, list[str]] = {}
    for symbol, synonyms in zip(mitocarta["symbol"], mitocarta["synonyms"]):
        for syn in str(synonyms).split("|"):
            syn = syn.strip().lower()
            if syn and syn not in symbol_of:
                synonym_of.setdefault(syn, []).append(symbol)
    info = mitocarta.set_index("symbol")
    rows = []
    for name, row in meta.iterrows():
        symbols = split_symbols(row["pg_genes"])
        accessions = split_accessions(row["protein_group"])
        sym_hits = list(dict.fromkeys(symbol_of[s] for s in symbols if s in symbol_of))
        uni_hits = list(dict.fromkeys(uniprot_of[a] for a in accessions if a in uniprot_of))
        syn_hits = list(dict.fromkeys(m for s in symbols if s not in symbol_of for m in synonym_of.get(s, [])))
        hits = list(dict.fromkeys([*sym_hits, *uni_hits]))
        n_matched_symbols = sum(s in symbol_of for s in symbols)
        first = str(row["first_symbol"])
        route = ("symbol_and_uniprot" if sym_hits and uni_hits else "symbol" if sym_hits
                 else "uniprot" if uni_hits else "")
        rows.append({
            "protein": name,
            "pg_genes": row["pg_genes"],
            "protein_group": row["protein_group"],
            "n_gene_symbols": len(symbols),
            "mitocarta_hits": ";".join(hits),
            "mitocarta_symbol_hits": ";".join(sym_hits),
            "mitocarta_uniprot_hits": ";".join(uni_hits),
            "mitocarta_synonym_only_hits": ";".join(syn_hits),
            "match_route": route,
            "mitocarta": bool(hits),
            "mitocarta_symbol_only": bool(sym_hits),
            "mixed_group": bool(len(symbols) > 1 and 0 < n_matched_symbols < len(symbols)),
            "oxphos": any(info.at[s, "is_oxphos"] for s in hits),
            "oxphos_subunit": any(info.at[s, "is_oxphos_subunit"] for s in hits),
            "mt_encoded": any(info.at[s, "is_mt_encoded"] for s in hits),
            "mt_prefix_task1": first.startswith(("Mt-", "mt-")),
            "description_mentions_mitochondri": "mitochondri" in str(row["protein_description"]).lower(),
            "mito_pathways": " | ".join(info.at[s, "mito_pathways"] for s in hits if info.at[s, "mito_pathways"]),
            "sub_mito_localization": ";".join(dict.fromkeys(info.at[s, "sub_mito_localization"] for s in hits)),
        })
    return pd.DataFrame(rows).set_index("protein")


def canonical_meta(adata) -> pd.DataFrame:
    var = adata.var
    return pd.DataFrame({"pg_genes": var["pg_genes"].astype(str).to_numpy(),
                         "protein_group": var["protein_group"].astype(str).to_numpy(),
                         "first_symbol": var["gene_name"].astype(str).to_numpy(),
                         "protein_description": var["protein_description"].astype(str).to_numpy()},
                        index=pd.Index(adata.var_names.astype(str), name="protein"))


def read_annotation(run: cc.Run, dataset: str, var_names) -> pd.DataFrame:
    path = cc.out_path(TASK, "tables", f"{dataset}_protein_annotation", ".csv")
    if not path.exists():
        raise FileNotFoundError(f"{cc.rel(path)} not found; run qc_mito_mitocarta.py first.")
    frame = pd.read_csv(run.use_input(path, f"qc_mito protein annotation ({dataset})"), keep_default_na=False)
    frame = frame.set_index("protein")
    if list(frame.index.astype(str)) != list(map(str, var_names)):
        raise ValueError(f"{dataset}: protein annotation order differs from the canonical object.")
    for col in SETS:
        frame[col] = frame[col].astype(str).str.lower().eq("true")
    return frame


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------
def mito_metrics(intensity: np.ndarray, flags: pd.DataFrame, index) -> pd.DataFrame:
    """Detection and intensity fractions per protein set (intensity = raw PG.Quantity, 0 = not detected)."""
    intensity = np.asarray(intensity, dtype=np.float64)
    detected = intensity > 0
    n_det = detected.sum(axis=1).astype(float)
    total = intensity.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {"n_detected_proteins": n_det.astype(int), "total_intensity": total,
               "log10_n_detected": np.where(n_det > 0, np.log10(n_det), np.nan),
               "log10_total_intensity": np.where(total > 0, np.log10(total), np.nan)}
        for name in SETS:
            mask = flags[name].to_numpy(bool)
            n = detected[:, mask].sum(axis=1)
            signal = intensity[:, mask].sum(axis=1)
            out[f"n_detected_{name}"] = n.astype(int)
            out[f"{name}_id_fraction"] = np.where(n_det > 0, n / n_det, np.nan)
            out[f"{name}_intensity_fraction"] = np.where(total > 0, signal / total, np.nan)
    return pd.DataFrame(out, index=pd.Index(list(map(str, index)), name="cell_id"))


def load(run: cc.Run, dataset: str):
    """kidney_object canonical object and labels (inputs recorded)."""
    for path, label in ((cc.CANONICAL[dataset]["labels"], f"kidney_object canonical {dataset} labels"),
                        (cc.CANONICAL[dataset]["h5ad"], f"kidney_object canonical {dataset} object")):
        run.use_input(path, label)
    return cc.load_canonical(dataset)


def label_order(values, dataset: str) -> list[str]:
    if dataset == "dct" and set(map(str, values)) <= {"P1", "P2", "unknown"}:
        return [p for p in ("P1", "P2", "unknown") if p in set(map(str, values))]
    return cc.ordered_labels(values, DATASETS[dataset]["order"])


def label_colors(adata, column: str) -> dict[str, str]:
    if column == "plate":
        return dict(cc.PLATE_PALETTE)
    obs = adata.obs[column]
    return dict(zip(map(str, obs.cat.categories), adata.uns[f"{column}_colors"]))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def mad_flags(frame: pd.DataFrame, label_col: str, metrics=OUTLIER_METRICS, min_n: int = MIN_GROUP,
              n_mad: float = N_MAD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within-label outliers: 'high' / 'low' / 'none', or 'not_evaluated' (label with < min_n cells or MAD = 0).

    MAD is scaled to the normal SD (scipy ``scale='normal'``, factor 1.4826), as in scater's isOutlier.
    """
    flags = pd.DataFrame("not_evaluated", index=frame.index, columns=list(metrics), dtype=object)
    rows = []
    for label, sub in frame.groupby(frame[label_col].astype(str), observed=True):
        for metric in metrics:
            x = pd.to_numeric(sub[metric], errors="coerce")
            ok = x.notna()
            n = int(ok.sum())
            med = float(x[ok].median()) if n else np.nan
            mad = float(stats.median_abs_deviation(x[ok], scale="normal")) if n else np.nan
            evaluated = n >= min_n and np.isfinite(mad) and mad > 0
            lower, upper = (med - n_mad * mad, med + n_mad * mad) if evaluated else (np.nan, np.nan)
            if evaluated:
                state = np.where(x > upper, "high", np.where(x < lower, "low", "none"))
                state = np.where(ok, state, "not_evaluated")
                flags.loc[sub.index, metric] = state
            rows.append({"label_column": label_col, "label": label, "metric": metric, "n_cells": len(sub),
                         "n_finite": n, "median": med, "mad_scaled": mad, "lower": lower, "upper": upper,
                         "evaluated": evaluated,
                         "reason_not_evaluated": "" if evaluated else (f"n < {min_n}" if n < min_n else "MAD = 0"),
                         "n_high": int((flags.loc[sub.index, metric] == "high").sum()),
                         "n_low": int((flags.loc[sub.index, metric] == "low").sum())})
    return flags, pd.DataFrame(rows)


def ols(y: np.ndarray, columns: dict[str, np.ndarray]):
    """OLS with intercept and HC3 standard errors; returns the statsmodels result."""
    import statsmodels.api as sm

    design = pd.DataFrame(columns)
    design.insert(0, "const", 1.0)
    return sm.OLS(np.asarray(y, dtype=float), design.astype(float)).fit(cov_type="HC3")


PROLIF_CONTRASTS = (("ProLIF", ("DCT1",)), ("ProLIF", ("DCT2",)), ("ProLIF", ("DCT1", "DCT2")))
CONTRAST_MODELS = {
    "label_only": (),
    "label_plus_log10_ids": ("log10_n_detected",),
    "label_plus_log10_ids_plus_plate": ("log10_n_detected", "plate_P2"),
}


def prolif_contrasts(frame: pd.DataFrame, label_col: str, metrics, contrasts=PROLIF_CONTRASTS) -> pd.DataFrame:
    """ProLIF versus DCT1 / DCT2 / DCT1+DCT2: Mann-Whitney and OLS (label indicator coefficient, HC3).

    Models: y ~ label; y ~ label + log10(IDs); y ~ label + log10(IDs) + plate.
    The label coefficient is the mean difference of the group relative to the reference.
    """
    rows = []
    for group, refs in contrasts:
        mask = frame[label_col].astype(str).isin([group, *refs]).to_numpy()
        sub = frame.loc[mask]
        is_group = (sub[label_col].astype(str) == group).to_numpy(float)
        covars = {"log10_n_detected": sub["log10_n_detected"].to_numpy(float),
                  "plate_P2": (sub["plate"].astype(str) == "P2").to_numpy(float)}
        for metric in metrics:
            y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)
            ok = np.isfinite(y)
            row = {"label_column": label_col, "comparison": f"{group}_vs_{'+'.join(refs)}", "metric": metric}
            mw = cc.compare_groups(y[ok & (is_group == 1)], y[ok & (is_group == 0)])
            row.update({k: mw[k] for k in ("n_group", "n_reference", "median_group", "median_reference",
                                           "median_difference", "rank_biserial", "cohens_d")})
            row["mannwhitney_p"] = mw["p_value"]
            for model, extra in CONTRAST_MODELS.items():
                fit = ols(y[ok], {"label": is_group[ok], **{c: covars[c][ok] for c in extra}})
                ci = fit.conf_int().loc["label"]
                row.update({f"{model}_coef": fit.params["label"], f"{model}_ci_low": ci[0],
                            f"{model}_ci_high": ci[1], f"{model}_p": fit.pvalues["label"], f"{model}_r2": fit.rsquared})
            rows.append(row)
    out = pd.DataFrame(rows)
    for p_col in ["mannwhitney_p", *[f"{m}_p" for m in CONTRAST_MODELS]]:
        out = cc.add_bh(out, ["label_column"], p_col=p_col, out_col=p_col.removesuffix("_p") + "_p_fdr_bh")
    return out


def type_tests(frame: pd.DataFrame, label_col: str, metric: str, reference: str,
               min_n: int = MIN_GROUP) -> tuple[dict, pd.DataFrame]:
    """Across-label tests on labels with >= min_n cells.

    Kruskal-Wallis; nested-model F tests (classical OLS) for the label term
    without and with log10(IDs); partial R2 of the label after depth; HC3
    coefficients of each label against ``reference`` in both models.
    """
    import statsmodels.api as sm

    labels = frame[label_col].astype(str)
    counts = labels.value_counts()
    kept = [lab for lab in counts.index if counts[lab] >= min_n]
    if reference not in kept:
        reference = counts.index[0]
    sub = frame.loc[labels.isin(kept)].copy()
    sub_labels = sub[label_col].astype(str)
    y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(float)
    ok = np.isfinite(y)
    y, sub, sub_labels = y[ok], sub.loc[ok], sub_labels[ok]
    others = [lab for lab in kept if lab != reference]
    dummies = pd.DataFrame({f"label_{lab}": (sub_labels == lab).to_numpy(float) for lab in others},
                           index=sub.index)
    depth = sub["log10_n_detected"].to_numpy(float)
    groups = [y[(sub_labels == lab).to_numpy()] for lab in kept]
    kw = stats.kruskal(*groups) if len(groups) > 1 else None

    def fit(cols: pd.DataFrame, robust: bool):
        design = cols.copy()
        design.insert(0, "const", 1.0)
        model = sm.OLS(y, design.astype(float))
        return model.fit(cov_type="HC3") if robust else model.fit()

    null = fit(pd.DataFrame(index=sub.index), False)
    lab_only = fit(dummies, False)
    depth_only = fit(pd.DataFrame({"log10_n_detected": depth}, index=sub.index), False)
    full = fit(dummies.assign(log10_n_detected=depth), False)

    def f_test(small, big):
        df_num = big.df_model - small.df_model
        f = ((small.ssr - big.ssr) / df_num) / (big.ssr / big.df_resid)
        return float(f), float(stats.f.sf(f, df_num, big.df_resid)), int(df_num), int(big.df_resid)

    f0, p0, d0, r0 = f_test(null, lab_only)
    f1, p1, d1, r1 = f_test(depth_only, full)
    summary = {"label_column": label_col, "metric": metric, "n_labels_tested": len(kept),
               "labels_tested": ";".join(kept), "labels_excluded_n_lt_min": ";".join(l for l in counts.index if l not in kept),
               "n_cells": int(len(y)), "kruskal_h": kw.statistic if kw else np.nan,
               "kruskal_p": kw.pvalue if kw else np.nan,
               "F_label_unadjusted": f0, "p_label_unadjusted": p0, "df_label": d0, "df_resid_unadjusted": r0,
               "F_label_after_log10_ids": f1, "p_label_after_log10_ids": p1, "df_resid_adjusted": r1,
               "r2_label_only": lab_only.rsquared, "r2_log10_ids_only": depth_only.rsquared,
               "r2_label_plus_log10_ids": full.rsquared,
               "partial_r2_label_after_log10_ids": (depth_only.ssr - full.ssr) / depth_only.ssr,
               "reference_label": reference}
    rows = []
    for model_name, cols in (("label_only", dummies), ("label_plus_log10_ids", dummies.assign(log10_n_detected=depth))):
        res = fit(cols, True)
        ci = res.conf_int()
        for lab in others:
            key = f"label_{lab}"
            rows.append({"label_column": label_col, "metric": metric, "model": model_name, "label": lab,
                         "reference_label": reference, "n_label": int((sub_labels == lab).sum()),
                         "n_reference": int((sub_labels == reference).sum()), "coef": res.params[key],
                         "ci_low": ci.loc[key, 0], "ci_high": ci.loc[key, 1], "p_value": res.pvalues[key]})
    return summary, pd.DataFrame(rows)


def spearman_ci(x, y) -> dict[str, float]:
    """Spearman rho with a Fisher-z 95% interval (SE = 1.06 / sqrt(n - 3), Fieller et al. 1957)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 4 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return {"n": n, "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    rho, p = stats.spearmanr(x[ok], y[ok])
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    half = 1.959964 * 1.06 / np.sqrt(n - 3)
    return {"n": n, "rho": float(rho), "ci_low": float(np.tanh(z - half)), "ci_high": float(np.tanh(z + half)),
            "p_value": float(p)}


def summarise(frame: pd.DataFrame, group_col: str, metrics, dataset: str) -> pd.DataFrame:
    rows = []
    groups = [("all_cells", np.ones(len(frame), bool))]
    groups += [(lab, (frame[group_col].astype(str) == lab).to_numpy()) for lab in label_order(frame[group_col], dataset)]
    for label, mask in groups:
        for metric in metrics:
            x = pd.to_numeric(frame.loc[mask, metric], errors="coerce").dropna()
            rows.append({"dataset": dataset, "label_column": group_col, "label": label, "metric": metric,
                         "n_cells": int(mask.sum()), "n_finite": len(x), "mean": x.mean(), "median": x.median(),
                         "q25": x.quantile(0.25), "q75": x.quantile(0.75), "min": x.min(), "max": x.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "surface": "#fcfcfb", "background_cells": "#d6d5cf"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "axes.edgecolor": INK["muted"], "axes.labelcolor": INK["secondary"], "text.color": INK["primary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
        "axes.titlelocation": "left",
    })


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("t03_blue", ["#eef4fc", *BLUE_RAMP])


def footer(fig, script: str, note: str = "") -> None:
    text = f"{cc.GENERATED_TAG} · scripts/{script}" + (f" · {note}" if note else "")
    fig.text(0.005, 0.002, text, ha="left", va="bottom", fontsize=4.8, color=INK["muted"])


def ygrid(ax) -> None:
    ax.yaxis.grid(True, color=INK["grid"], linewidth=0.5)
    ax.set_axisbelow(True)


def box_strip(ax, frame: pd.DataFrame, group_col: str, metric: str, order: list[str], colors: dict[str, str],
              highlight: np.ndarray | None = None, seed: int = SEED) -> None:
    """Box (no fliers) plus jittered points per label; highlighted cells get a dark ring."""
    rng = np.random.default_rng(seed)
    groups = frame[group_col].astype(str).to_numpy()
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
    for i, label in enumerate(order):
        mask = (groups == label) & np.isfinite(values)
        vals = values[mask]
        if vals.size:
            bp = ax.boxplot(vals, positions=[i], widths=0.6, showfliers=False, patch_artist=True)
            bp["boxes"][0].set_facecolor("none")
            bp["boxes"][0].set_edgecolor(INK["secondary"])
            bp["boxes"][0].set_linewidth(0.6)
            for part in ("whiskers", "caps"):
                for line in bp[part]:
                    line.set_color(INK["secondary"])
                    line.set_linewidth(0.6)
            bp["medians"][0].set_color(INK["primary"])
            bp["medians"][0].set_linewidth(1.0)
            x = i + rng.uniform(-0.22, 0.22, vals.size)
            ax.scatter(x, vals, s=3.5, color=colors.get(label, INK["muted"]), alpha=0.75, linewidths=0,
                       rasterized=True, zorder=3)
            if highlight is not None:
                hl = highlight[mask]
                if hl.any():
                    ax.scatter(x[hl], vals[hl], s=9, facecolors="none", edgecolors=INK["primary"], linewidths=0.5,
                               zorder=4)
    counts = pd.Series(groups).value_counts()
    ax.set_xticks(range(len(order)), [f"{lab}\n{counts.get(lab, 0)}" for lab in order], fontsize=5.6)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ygrid(ax)


def scatter_by_label(ax, x, y, groups, order: list[str], colors: dict[str, str], size: float = 4.0) -> None:
    groups = np.asarray(groups).astype(str)
    for label in order:
        mask = groups == label
        ax.scatter(np.asarray(x)[mask], np.asarray(y)[mask], s=size, color=colors.get(label, INK["muted"]),
                   alpha=0.8, linewidths=0, rasterized=True, label=label)
    ygrid(ax)


def legend_with_counts(ax, groups, order: list[str], colors: dict[str, str], title: str = "", **kwargs) -> None:
    from matplotlib.lines import Line2D

    counts = pd.Series(np.asarray(groups).astype(str)).value_counts()
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4, markerfacecolor=colors.get(lab, INK["muted"]),
                      markeredgewidth=0, label=f"{lab} ({counts.get(lab, 0)})") for lab in order]
    kwargs.setdefault("loc", "upper left")
    kwargs.setdefault("bbox_to_anchor", (1.01, 1.0))
    ax.legend(handles=handles, title=title or None, title_fontsize=6, fontsize=5.8, **kwargs)


def umap_continuous(fig, ax, xy: np.ndarray, values, title: str, cmap=None, robust: bool = True) -> None:
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.where(np.isfinite(values), values, -np.inf))
    finite = values[np.isfinite(values)]
    vmin, vmax = (np.quantile(finite, [0.01, 0.99]) if robust and finite.size else (np.nanmin(values), np.nanmax(values)))
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values)) + 1e-12
    sca = ax.scatter(xy[order, 0], xy[order, 1], c=values[order], cmap=cmap or sequential_cmap(), vmin=vmin, vmax=vmax,
                     s=4, linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=6.5)
    cbar = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.outline.set_visible(False)


def umap_categorical(ax, xy: np.ndarray, groups, order: list[str], colors: dict[str, str], title: str) -> None:
    groups = np.asarray(groups).astype(str)
    for label in order:
        mask = groups == label
        ax.scatter(xy[mask, 0], xy[mask, 1], s=4, color=colors.get(label, INK["muted"]), linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=6.5)
