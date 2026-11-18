#!/usr/bin/env python
"""qc_mito step 1: MitoCarta3.0 mouse list and its coverage in both protein matrices.



Inputs (read-only):
  data/reference/Mouse.MitoCarta3.0.xls (downloaded 13 Oct 2026 with the authors' approval; see DOWNLOAD)
  results/h5ad/kidney_object_{KIDNEY_submitted,DCT_V1_submitted}.h5ad and the kidney_object canonical label tables
Outputs:
  results/tables/qc_mito/qc_mito_mitocarta_mouse_list.csv        parsed sheet 'A Mouse MitoCarta3.0'
  results/tables/qc_mito/qc_mito_mitocarta_checks.csv            parsing sanity checks
  results/tables/qc_mito/qc_mito_{kidney,dct}_protein_annotation.csv   flags for every matrix protein
  results/tables/qc_mito/qc_mito_mito_coverage.csv               coverage per protein set
  results/tables/qc_mito/qc_mito_mitopathway_coverage.csv        coverage per top-level MitoPathway

Matching: a protein group is MitoCarta when any of its PG.Genes symbols equals a
MitoCarta3.0 symbol (case-insensitive) or any of its UniProt accessions (isoform
suffix removed) equals a MitoCarta UniProt accession (authors decision, 13 Oct 2026,
after the symbol-only rule missed renamed genes). Symbol-only matching is kept as
a sensitivity set; synonym-only matches are listed but never used.
OXPHOS = MitoPathway 'OXPHOS' (any sub-path); mt-encoded = the 13 'mt-' genes.
The Task 1 rule (the earlier analysis, scripts/02_qc_mito_batch.py) is reproduced as
'first matrix gene symbol starts with Mt-/mt-'.

This script reads a downloaded file; run it in isolated mode (-I), quoted so
PowerShell does not take it as its own parameter:
  python '-I' scripts/qc_mito_mitocarta.py
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
    # Isolated mode (-I) does not put the script folder on sys.path.
    _runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parent))
    from _scanpy_runtime import ensure_scanpy_environment as _select_project_python
    _select_project_python(__file__)


import argparse
import sys
from importlib import metadata

import numpy as np
import pandas as pd

import scp_common as cc
import qc_mito_lib as t3

REFERENCE = cc.PROJECT_ROOT / "data" / "reference" / "Mouse.MitoCarta3.0.xls"
DOWNLOAD = {
    "file": "data/reference/Mouse.MitoCarta3.0.xls",
    "url": "https://sites.broadinstitute.org/files/mitocarta/files/documents/mouse.mitocarta3.0.xls",
    "landing_page": "https://www.broadinstitute.org/mitocarta/mitocarta30-inventory-mammalian-mitochondrial-proteins-and-pathways",
    "retrieved_utc": "2026-10-13T11:16:41Z",
    "bytes": 11000832,
    "sha256": "07a366b941d0a1e79edd96635bf8377ef83fb09e49a666bf0ced6e4f080fbe03",
    "http_last_modified": "Thu, 08 Oct 2026 13:10:53 GMT",
    "workbook_metadata": "Excel 97-2003 workbook; author Sarah Calvo; last saved 2022-01-11",
    "approval": "authors approved the download in chat on 2026-10-13",
    "note": ("The older link personal.broadinstitute.org/scalvo/MitoCarta3.0/Mouse.MitoCarta3.0.xls could not be "
             "reached from this machine; the file was taken from the 'Mouse.MitoCarta3.0.xls' link on the Broad "
             "MitoCarta3.0 page (read in the built-in browser)."),
}
SHEET_LIST = "A Mouse MitoCarta3.0"
SHEET_PATHWAYS = "C MitoPathways"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qc_mito step 1: MitoCarta3.0 list and matrix coverage.")
    cc.add_common_args(parser)
    return parser.parse_args()


def clean(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "" if text in ("0", "nan") else text


def parse_mitocarta(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=SHEET_LIST, engine="xlrd", dtype=object)
    pathways = pd.read_excel(path, sheet_name=SHEET_PATHWAYS, engine="xlrd", dtype=object)
    frame = pd.DataFrame({
        "mouse_gene_id": raw["MouseGeneID"].map(lambda v: str(int(v)) if pd.notna(v) else ""),
        "symbol": raw["Symbol"].map(clean),
        "synonyms": raw["Synonyms"].map(clean),
        "description": raw["Description"].map(clean),
        "uniprot": raw["UniProt"].map(clean),
        "ensembl_gene_id": raw["EnsemblGeneID"].map(clean),
        "mitocarta_list": raw["MitoCarta3.0_List"].map(clean),
        "sub_mito_localization": raw["MitoCarta3.0_SubMitoLocalization"].map(clean),
        "mito_pathways": raw["MitoCarta3.0_MitoPathways"].map(clean),
    })
    paths = frame["mito_pathways"].map(lambda s: [p.strip() for p in s.split("|") if p.strip()])
    frame["top_level_pathways"] = paths.map(lambda ps: ";".join(dict.fromkeys(p.split(" > ")[0].strip() for p in ps)))
    frame["is_oxphos"] = paths.map(lambda ps: any(p.split(" > ")[0].strip() == "OXPHOS" for p in ps))
    frame["is_oxphos_subunit"] = paths.map(lambda ps: any(p.startswith("OXPHOS > OXPHOS subunits") for p in ps))
    frame["is_mt_encoded"] = frame["symbol"].str.lower().str.startswith("mt-")
    return frame, pathways


def pathway_genes(pathways: pd.DataFrame, hierarchy: str) -> set[str]:
    row = pathways.loc[pathways["MitoPathway Hierarchy"].astype(str).str.strip() == hierarchy]
    if row.empty:
        return set()
    return {g.strip() for g in str(row["Genes"].iloc[0]).split(",") if g.strip()}


def main() -> None:
    args = parse_args()
    run = cc.Run(t3.TASK, "mitocarta", args)
    run.use_input(REFERENCE, "MitoCarta3.0 mouse inventory (downloaded, Broad Institute)")
    checks = []

    def check(name: str, observed, expected, passed: bool, info: bool = False) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected,
                       "pass": "info" if info else bool(passed)})

    sha = run.inputs[cc.rel(REFERENCE)]["sha256"]
    check("downloaded file SHA256 equals the download record", sha, DOWNLOAD["sha256"], sha == DOWNLOAD["sha256"])
    mitocarta, pathways = parse_mitocarta(REFERENCE)
    check("genes in sheet A", len(mitocarta), 1140, len(mitocarta) == 1140)
    check("unique symbols", mitocarta["symbol"].nunique(), len(mitocarta), mitocarta["symbol"].is_unique)
    check("rows with MitoCarta3.0_List = 'MitoCarta3.0'", int((mitocarta["mitocarta_list"] == "MitoCarta3.0").sum()),
          len(mitocarta), (mitocarta["mitocarta_list"] == "MitoCarta3.0").all())
    mt = sorted(mitocarta.loc[mitocarta["is_mt_encoded"], "symbol"])
    check("mt-encoded genes", f"{len(mt)}: {';'.join(mt)}", 13, len(mt) == 13)
    for flag, hierarchy in (("is_oxphos", "OXPHOS"), ("is_oxphos_subunit", "OXPHOS > OXPHOS subunits")):
        ours = set(mitocarta.loc[mitocarta[flag], "symbol"])
        sheet = pathway_genes(pathways, hierarchy)
        check(f"{flag}: genes equal sheet C '{hierarchy}'", f"{len(ours)} (sheet C {len(sheet)}; "
              f"only ours {len(ours - sheet)}, only sheet C {len(sheet - ours)})", len(sheet), ours == sheet)
    check("mt-encoded genes are OXPHOS subunits", int(mitocarta.loc[mitocarta["is_mt_encoded"], "is_oxphos_subunit"].sum()),
          13, True, info=True)
    check("genes without any MitoPathway", int((mitocarta["mito_pathways"] == "").sum()), "", True, info=True)
    check("genes without UniProt accession", int((mitocarta["uniprot"] == "").sum()), "", True, info=True)
    run.table(mitocarta, t3.MITOCARTA_CSV_NAME)
    run.note("Primary MitoCarta set = gene-symbol OR UniProt-accession match (authors decision 13 Oct 2026, after the "
             "first run's check 'Task 1 prefix proteins are MitoCarta mt-encoded' failed: Mt-Cyb (MitoCarta mt-Cytb) "
             "matched only by UniProt). Symbol-only kept as the 'mitocarta_symbol_only' sensitivity set.")

    coverage_rows, pathway_rows = [], []
    top_levels = sorted({p for s in mitocarta["top_level_pathways"] for p in s.split(";") if p})
    for dataset in ("kidney", "dct"):
        adata, _ = t3.load(run, dataset)
        meta = t3.canonical_meta(adata)
        annotation = t3.annotate_proteins(meta, mitocarta)
        n_cells = adata.n_obs
        annotation["n_cells_detected"] = adata.var["n_cells_detected"].to_numpy(int)
        annotation["fraction_cells_detected"] = annotation["n_cells_detected"] / n_cells
        task1 = sorted(annotation.index[annotation["mt_prefix_task1"]])
        run.note(f"{dataset}: Task 1 Mt-/mt- prefix proteins in the matrix: {task1 or 'none'}.")
        run.table(annotation.reset_index(), f"{dataset}_protein_annotation")

        set_genes = {"mitocarta": set(mitocarta["symbol"]), "mitocarta_symbol_only": set(mitocarta["symbol"]),
                     "oxphos": set(mitocarta.loc[mitocarta["is_oxphos"], "symbol"]),
                     "oxphos_subunit": set(mitocarta.loc[mitocarta["is_oxphos_subunit"], "symbol"]),
                     "mt_encoded": set(mitocarta.loc[mitocarta["is_mt_encoded"], "symbol"])}

        def represented(name: str, mask) -> float:
            if name not in set_genes:
                return np.nan
            column = "mitocarta_symbol_hits" if name == "mitocarta_symbol_only" else "mitocarta_hits"
            found = {s for hits in annotation.loc[mask, column] for s in hits.split(";") if s}
            return len(found & set_genes[name])

        extra = {
            "matched_by_uniprot_only (in primary, not by symbol)": annotation["mitocarta"] & ~annotation["mitocarta_symbol_only"],
            "synonym_only_candidates (not used)": (annotation["mitocarta_synonym_only_hits"] != "") & ~annotation["mitocarta"],
            "mixed_groups (MitoCarta and non-MitoCarta symbols)": annotation["mixed_group"],
            "description_mentions_mitochondri": annotation["description_mentions_mitochondri"],
            "description_mitochondri_and_mitocarta": annotation["description_mentions_mitochondri"] & annotation["mitocarta"],
            "description_mitochondri_not_mitocarta": annotation["description_mentions_mitochondri"] & ~annotation["mitocarta"],
            "mitocarta_without_description_mitochondri": annotation["mitocarta"] & ~annotation["description_mentions_mitochondri"],
        }
        for name, mask in [*((s, annotation[s]) for s in t3.SETS), *extra.items()]:
            mask = mask.to_numpy(bool)
            frac = annotation.loc[mask, "fraction_cells_detected"]
            coverage_rows.append({
                "dataset": dataset, "protein_set": name, "definition": t3.SETS.get(name, ""),
                "n_genes_in_list": len(set_genes[name]) if name in set_genes else np.nan,
                "n_list_genes_represented_in_matrix": represented(name, mask),
                "n_protein_groups_in_matrix": int(mask.sum()), "n_matrix_proteins": int(len(annotation)),
                "percent_of_matrix_proteins": 100 * mask.sum() / len(annotation),
                "n_groups_detected_ge10pct_cells": int((frac >= 0.10).sum()),
                "n_groups_detected_ge50pct_cells": int((frac >= 0.50).sum()),
                "median_fraction_cells_detected": float(frac.median()) if len(frac) else np.nan,
                "proteins": ";".join(annotation.index[mask]) if mask.sum() <= 40 else "",
            })
        for level in top_levels:
            genes = set(mitocarta.loc[mitocarta["top_level_pathways"].str.split(";").map(lambda ps: level in ps), "symbol"])
            in_matrix = {s for hits in annotation["mitocarta_hits"] for s in hits.split(";") if s in genes}
            groups = annotation["mitocarta_hits"].map(lambda h: any(s in genes for s in h.split(";") if s))
            pathway_rows.append({"dataset": dataset, "top_level_mitopathway": level, "n_genes_in_mitocarta": len(genes),
                                 "n_genes_represented_in_matrix": len(in_matrix),
                                 "percent_genes_represented": 100 * len(in_matrix) / len(genes) if genes else np.nan,
                                 "n_protein_groups_in_matrix": int(groups.sum()),
                                 "n_groups_detected_ge10pct_cells": int((groups & (annotation["fraction_cells_detected"] >= 0.1)).sum())})
        no_pathway = set(mitocarta.loc[mitocarta["mito_pathways"] == "", "symbol"])
        in_matrix = {s for hits in annotation["mitocarta_hits"] for s in hits.split(";") if s in no_pathway}
        pathway_rows.append({"dataset": dataset, "top_level_mitopathway": "(no MitoPathway)",
                             "n_genes_in_mitocarta": len(no_pathway), "n_genes_represented_in_matrix": len(in_matrix),
                             "percent_genes_represented": 100 * len(in_matrix) / len(no_pathway),
                             "n_protein_groups_in_matrix": np.nan, "n_groups_detected_ge10pct_cells": np.nan})
        n_mc = int(annotation["mitocarta"].sum())
        check(f"{dataset}: MitoCarta protein groups in matrix (symbol or UniProt; symbol only)",
              f"{n_mc}; {int(annotation['mitocarta_symbol_only'].sum())}", "", True, info=True)
        mt_found = annotation.loc[annotation["mt_encoded"]]
        check(f"{dataset}: mt-encoded proteins in matrix (matrix name -> MitoCarta gene)",
              "; ".join(f"{p} -> {h}" for p, h in zip(mt_found.index, mt_found["mitocarta_hits"])) or "none",
              "", True, info=True)
        multi = annotation.loc[annotation["mitocarta_hits"].str.contains(";")]
        check(f"{dataset}: protein groups matched to >1 MitoCarta gene", int(len(multi)), "", True, info=True)
        check(f"{dataset}: Task 1 prefix proteins are MitoCarta mt-encoded",
              ";".join(task1), "subset", set(task1) <= set(annotation.index[annotation["mt_encoded"]]))

    run.table(pd.DataFrame(checks), "mitocarta_checks")
    run.table(pd.DataFrame(coverage_rows), "mito_coverage")
    run.table(pd.DataFrame(pathway_rows), "mitopathway_coverage")
    failed = [c["check"] for c in checks if c["pass"] is False]
    if failed:
        run.note(f"FAILED checks: {failed}")
    run.finish({"download": DOWNLOAD, "xlrd_version": metadata.version("xlrd"), "isolated_mode": bool(sys.flags.isolated),
                "sheets": [SHEET_LIST, SHEET_PATHWAYS], "n_failed_checks": len(failed)},
               status="completed" if not failed else "completed_with_failed_checks")
    if failed:
        raise SystemExit(f"Sanity checks failed: {failed}")


if __name__ == "__main__":
    main()
