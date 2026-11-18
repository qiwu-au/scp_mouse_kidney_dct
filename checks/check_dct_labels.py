#!/usr/bin/env python
"""Compare a rerun of the DCT annotation with the deposited processed data.

Run the DCT steps first (README.md, "Run order": intake, labels, proliferation call). This script then compares

  results/tables/dct_labels/dct_labels_dct_canonical_labels.csv   (cluster annotation, transferred label)
  results/tables/dct_prolif/dct_prolif_dct_labels.csv             (per-cell proliferation call)

with the cell tables of the Zenodo record, which must be placed in ``data/zenodo/``:

  SCP_DCT_control_729cells_cell_metadata.csv
  SCP_DCT_zeroK_702cells_cell_metadata.csv

The deposited tables use public names. The intermediate tables keep the column names of the analysis:
``cs_ctrl_deg_*`` (control) and ``cs_zerok_pipeline_*`` (zero-K) are the cluster annotation, ``cs_zerok_transfer_label``
the transferred label and ``cs_t31nf_*`` the proliferation call. ``Unknown`` is written ``DCT, subtype not resolved``
in the deposit.

Output: one line per check on screen and ``results/tables/checks/check_dct_labels.csv``. The exit code is 1 when a
label check fails. UMAP coordinates are reported for information only; they can differ between platforms.

Command (from the folder that holds ``scripts/``):  python checks/check_dct_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "results" / "tables" / "dct_labels" / "dct_labels_dct_canonical_labels.csv"
PROLIF = ROOT / "results" / "tables" / "dct_prolif" / "dct_prolif_dct_labels.csv"
DEPOSIT = {
    "Ctrl": ROOT / "data" / "zenodo" / "SCP_DCT_control_729cells_cell_metadata.csv",
    "ZeroK": ROOT / "data" / "zenodo" / "SCP_DCT_zeroK_702cells_cell_metadata.csv",
}
COLUMNS = {"Ctrl": ("cs_ctrl_deg_label", "cs_ctrl_deg_cluster"), "ZeroK": ("cs_zerok_pipeline_label", "cs_zerok_pipeline_cluster")}
NOT_RESOLVED = "DCT, subtype not resolved"
OUT = ROOT / "results" / "tables" / "checks" / "check_dct_labels.csv"


def public_label(values: pd.Series) -> pd.Series:
    return values.astype(str).replace({"Unknown": NOT_RESOLVED})


def main() -> int:
    missing = [p for p in (LABELS, PROLIF, *DEPOSIT.values()) if not p.exists()]
    if missing:
        print("Missing file(s):")
        for path in missing:
            print("  ", path.relative_to(ROOT).as_posix())
        return 2
    labels = pd.read_csv(LABELS, dtype={"cell_id": str}).set_index("cell_id")
    prolif = pd.read_csv(PROLIF, dtype={"cell_id": str}).set_index("cell_id")
    rows = []

    def add(diet: str, check: str, observed: str, passed: bool | None) -> None:
        rows.append({"diet": diet, "check": check, "observed": observed,
                     "kind": "information" if passed is None else "check", "pass": True if passed is None else bool(passed)})
        flag = "info" if passed is None else ("ok" if passed else "FAILED")
        print(f"  [{flag}] {diet}: {check}: {observed}")

    for diet, path in DEPOSIT.items():
        dep = pd.read_csv(path, dtype={"cell_id": str}).set_index("cell_id")
        mine = labels[labels["condition"] == diet]
        same_cells = set(mine.index) == set(dep.index)
        add(diet, "same cells as the deposit", f"{len(mine)} rerun, {len(dep)} deposited", same_cells)
        if not same_cells:
            continue
        mine, pro = mine.loc[dep.index], prolif.loc[dep.index]
        label_col, cluster_col = COLUMNS[diet]
        n = len(dep)
        eq = (public_label(mine[label_col]) == dep["cell_type"].astype(str)).to_numpy()
        add(diet, "cell_type", f"{int(eq.sum())} of {n} cells equal", bool(eq.all()))
        eq = (mine[cluster_col].astype(float).astype(int).astype(str) == dep["cluster"].astype(str)).to_numpy()
        add(diet, "cluster", f"{int(eq.sum())} of {n} cells equal", bool(eq.all()))
        if diet == "ZeroK":
            eq = (public_label(mine["cs_zerok_transfer_label"]) == dep["cell_type_transferred"].astype(str)).to_numpy()
            add(diet, "cell_type_transferred", f"{int(eq.sum())} of {n} cells equal", bool(eq.all()))
        eq = ((pro["cs_t31nf_label"].astype(str) == "ProLIF").to_numpy() == dep["prolif"].astype(str).eq("True").to_numpy())
        add(diet, "prolif", f"{int(eq.sum())} of {n} cells equal; {int((pro['cs_t31nf_label'].astype(str) == 'ProLIF').sum())} "
                            f"ProLIF cells in the rerun", bool(eq.all()))
        eq = ((pro["cs_t31nf_cycling"].astype(str) == "Cycling").to_numpy() == dep["cycling"].astype(str).eq("True").to_numpy())
        add(diet, "cycling", f"{int(eq.sum())} of {n} cells equal", bool(eq.all()))
        for mine_col, dep_col in (("cs_t31nf_prolif_D", "proliferation_score"), ("cs_t31nf_prolif_q", "proliferation_q")):
            diff = np.nanmax(np.abs(pro[mine_col].to_numpy(float) - dep[dep_col].to_numpy(float)))
            add(diet, dep_col, f"largest absolute difference {diff:.3g}", bool(diff < 1e-9))
        if {"UMAP1", "UMAP2"} <= set(dep.columns) and {"umap1_own", "umap2_own"} <= set(mine.columns):
            diff = float(np.nanmax(np.abs(mine[["umap1_own", "umap2_own"]].to_numpy(float) - dep[["UMAP1", "UMAP2"]].to_numpy(float))))
            add(diet, "UMAP coordinates", f"largest absolute difference {diff:.3g}", None)

    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    failed = int((~frame["pass"]).sum())
    print(f"{len(frame)} rows, {failed} failed; written to {OUT.relative_to(ROOT).as_posix()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
