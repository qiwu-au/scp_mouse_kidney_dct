#!/usr/bin/env python
"""kidney_object step 4: filtering numbers for both datasets (reviewer 2, minor 3).



Inputs (read-only): data/DCT/DCT_758cells.csv, data/KIDNEY/KIDNEY_768cells.csv, and the
kidney_object canonical objects (to confirm the filtered shapes).
Outputs:
  results/tables/kidney_object/kidney_object_filtering_summary.csv
  results/tables/kidney_object/kidney_object_filtering_removed_cells.csv

Filters (original scripts): cells with >= 100 proteins > 0 are kept, then
proteins > 0 in >= 2 retained cells are kept. A protein ID is a protein group
with intensity > 0 (missing values count as not detected).

Command (project root):  python scripts/kidney_object_filtering_summary.py
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
import sys

import numpy as np
import pandas as pd

import scp_common as cc

TASK = "kidney_object"
STAGES = {
    "raw_all_cells": "all cells in the raw CSV; IDs counted over all protein rows",
    "removed_cells": "cells with < 100 protein IDs (removed); IDs counted over all protein rows",
    "retained_cells_all_proteins": "cells with >= 100 protein IDs; IDs counted over all protein rows",
    "retained_cells_retained_proteins": "after both filters (final matrix); IDs counted over the retained proteins",
}


def id_stats(ids: np.ndarray) -> dict[str, float]:
    if ids.size == 0:
        return {"protein_ids_min": np.nan, "protein_ids_median": np.nan, "protein_ids_max": np.nan,
                "protein_ids_mean": np.nan}
    return {"protein_ids_min": int(ids.min()), "protein_ids_median": float(np.median(ids)),
            "protein_ids_max": int(ids.max()), "protein_ids_mean": float(ids.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="kidney_object: filtering numbers.")
    cc.add_common_args(parser)
    args = parser.parse_args()
    run = cc.Run(TASK, "filtering_summary", args)

    rows, removed_rows, failures = [], [], []
    for dataset in ("dct", "kidney"):
        rb = cc.rebuild_ln_intensity(dataset, run)
        raw = rb.raw_intensity.to_numpy()
        det_raw = raw > 0
        ids_raw = det_raw.sum(axis=1)
        keep = rb.keep_cells
        ids_final = (rb.intensity.to_numpy() > 0).sum(axis=1)
        n_rows, n_detected_any = raw.shape[1], int(det_raw.any(axis=0).sum())
        n_detected_kept_any = int(det_raw[keep].any(axis=0).sum())
        stage_data = {
            "raw_all_cells": (ids_raw, n_rows, n_detected_any),
            "removed_cells": (ids_raw[~keep], n_rows, int(det_raw[~keep].any(axis=0).sum())),
            "retained_cells_all_proteins": (ids_raw[keep], n_rows, n_detected_kept_any),
            "retained_cells_retained_proteins": (ids_final, int(rb.keep_proteins.sum()), int(rb.keep_proteins.sum())),
        }
        for stage, (ids, n_prot, n_any) in stage_data.items():
            rows.append({"dataset": dataset.upper(), "stage": stage, "n_cells": int(ids.size),
                         "n_protein_groups": n_prot, "n_protein_groups_detected_in_any_cell": n_any,
                         **id_stats(ids), "definition": STAGES[stage]})
        n_prot_dropped = n_rows - int(rb.keep_proteins.sum())
        run.note(f"{dataset.upper()}: {raw.shape[0]} -> {int(keep.sum())} cells; {n_rows} -> "
                 f"{int(rb.keep_proteins.sum())} protein groups ({n_prot_dropped} removed; "
                 f"{n_rows - n_detected_kept_any} never detected in retained cells, "
                 f"{n_detected_kept_any - int(rb.keep_proteins.sum())} detected in exactly one retained cell).")
        for cell, n in zip(rb.raw_intensity.index[~keep], ids_raw[~keep]):
            removed_rows.append({"dataset": dataset.upper(), "cell_id": cell, "protein_ids": int(n)})

        # Shapes must match the canonical objects.
        adata, _ = cc.load_canonical(dataset)
        if adata.shape != rb.intensity.shape or list(adata.obs_names) != list(rb.intensity.index):
            failures.append(f"{dataset}: rebuilt shape {rb.intensity.shape} != canonical {adata.shape}")
        run.use_input(cc.CANONICAL[dataset]["h5ad"], f"kidney_object canonical {dataset} object (shape check)")

    if failures:
        run.finish({"failures": failures}, status="failed")
        print("FILTERING CHECK FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    run.table(pd.DataFrame(rows), "filtering_summary")
    run.table(pd.DataFrame(removed_rows).sort_values(["dataset", "protein_ids"]), "filtering_removed_cells")
    run.finish()


if __name__ == "__main__":
    main()
