#!/usr/bin/env python
"""rna_protein_integration step 1: integration packages available in the Python environment.

Analysis plan,
step 1. Records which integration and metric packages are installed (read-only; installs nothing).

Decision by the authors (15 Oct 2026): install harmonypy; use the atlas's stored X_scVI for the scVI comparison
instead of installing scvi-tools. harmonypy 0.0.10 was installed (pip, PyPI): 2.1.0 (released 5 Oct 2026) and
2.0.x are C++ builds without a Windows wheel (a C++17 compiler is needed and none is installed); 0.2.0 requires
torch; 0.0.10 is the pure-Python implementation and needs only numpy, pandas, scipy and scikit-learn (installed).

Inputs: none (package metadata; mka.h5ad obsm keys via h5py, read-only).
Outputs: results/tables/rna_protein_integration/rna_protein_integration_packages.csv
Command (project root):  python scripts/rna_protein_integration_package_check.py
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
from importlib import metadata

import pandas as pd

import scp_common as cc
import rna_protein_integration_lib as T

PACKAGES = {
    "harmonypy": ("integration", "Harmony (embedding B) and compute_lisi (iLISI / cLISI)"),
    "scanorama": ("integration", "alternative to Harmony (not used)"),
    "scvi-tools": ("integration", "scVI model training (not used; the atlas's stored X_scVI is used instead)"),
    "torch": ("integration", "backend of scvi-tools (not used)"),
    "scib": ("metrics", "integration benchmark metrics (not used; LISI from harmonypy, silhouette from scikit-learn)"),
    "scib-metrics": ("metrics", "integration benchmark metrics (not used)"),
    "bbknn": ("integration", "batch-balanced kNN (not used)"),
    "scanpy": ("core", "neighbours graph and UMAP"),
    "anndata": ("core", "h5ad input / output"),
    "numpy": ("core", ""),
    "scipy": ("core", "eigendecomposition (PCA)"),
    "pandas": ("core", ""),
    "scikit-learn": ("core", "kNN, silhouette, F1"),
    "umap-learn": ("core", "UMAP"),
    "pynndescent": ("core", "approximate kNN for the UMAP graph"),
    "h5py": ("core", "reads the atlas obsm keys"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="rna_protein_integration step 1: integration packages in the Python environment.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(T.TASK, "package_check", args)
    rows = []
    for name, (role, use) in PACKAGES.items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"package": name, "role": role, "version": version, "use_in_rna_protein_integration": use})
    import h5py

    with h5py.File(run.use_input(T.ATLAS, "RNA atlas mka.h5ad (read-only; obsm keys only)"), "r") as f:
        obsm = {k: (tuple(f["obsm"][k].shape) if hasattr(f["obsm"][k], "shape") else "dataframe")
                for k in f["obsm"].keys()}
        uns = list(f["uns"].keys())
    rows.append({"package": "mka.h5ad obsm", "role": "stored latent", "version": "; ".join(f"{k} {v}" for k, v in obsm.items()),
                 "use_in_rna_protein_integration": "X_scVI (26 dims) for embedding C; the scVI model weights are not in the file "
                               f"(uns keys: {', '.join(uns)})"})
    frame = pd.DataFrame(rows)
    for r in rows:
        run.note(f"{r['package']}: {r['version']}")
    run.table(frame, "packages")
    run.finish({"harmonypy_version_choice": "0.0.10 (pure Python); 2.x = C++ build without Windows wheel; 0.2.0 needs torch",
                "scvi_tools": "not installed (authors: use the stored atlas X_scVI)"})


if __name__ == "__main__":
    main()
