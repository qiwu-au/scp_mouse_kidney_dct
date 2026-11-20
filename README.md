# Single-cell proteomics: a powerful new tool to study kidney cell heterogeneity

Qi Wu, Lei Cheng, Adrienne M. Assmus, Xiang Zheng, Cristina Esteva Font, Charlotte C. Petersen, Xiangning Ding, Søren R. Paludan and Robert A. Fenton

Python code of the analyses of this manuscript, on single-cell proteomics of mouse kidney and distal convoluted
tubule (DCT) cells. Licence: MIT.

Claude Code (Anthropic) was used to aid the scripting of the analyses of the revised manuscript (`scripts/`).

## What is here

| path | content |
| --- | --- |
| `scripts/` | 247 scripts: 156 run steps and 91 libraries |
| `SCRIPTS.csv` | every script with its stage, its role, its place in the run order, the item of the paper it makes, and whether its inputs are public |
| `_scanpy_compat.py` | fix that makes the random seed of Leiden clustering take effect on Windows (see "Reproducing the clusters") |
| `_scanpy_runtime.py` | cache folders for Matplotlib and Numba |
| `checks/check_dct_labels.py` | compares a rerun of the DCT annotation with the deposited processed data |
| `data/` | the marker lists, the checksum of the DCT report, and `data/README.md`, which says where every other input goes |
| `first_submission/` | the two scripts of the first submission |
| `requirements.txt`, `environment.yml` | Python packages with their versions |

The scripts are the code as it was run. For publication, file names, output paths and comments were rewritten by
fixed rules; no computation was changed. Three comparisons with tables that are not public are skipped when those
tables are absent (see "Changes made for publication").

Of the libraries, 63 are earlier versions of steps (role "library (earlier version of a step; imported, not
run)"). The run steps import functions from them. Their own entry points were used on data that the revised paper
does not analyse, and they are not part of the run order.

## Install

Python 3.12.10 on Windows 11 was used.

```
pip install -r requirements.txt
```

## Inputs

`data/README.md` lists every input, its source and its place. In short:

- the two Spectronaut protein-group reports (ProteomeXchange/PRIDE, PXD087695);
- the processed data (Zenodo, https://doi.org/10.5281/zenodo.24365599);
- the marker lists (included): `data/marker_lists_Supplementary_Table_S2.xlsx` is Supplementary Table S2 of the
  manuscript, the kidney list with 17 cell types and the DCT list. It is the only marker file here. The scripts read
  the lists from two small workbooks, `data/KIDNEY/KIDNEY_markers_V3.xlsx` and `data/DCT/DCT_markers_V1.xlsx`, which
  `scripts/scp_common.py` writes from the table the first time a script runs. The kidney workbook holds the first 16
  cell types of the table; the steps that used the 17th row, Endothelial cells, add it in the code
  (`ENDOTHELIAL_MARKERS` in `scripts/kidney_annotation_scan_lib.py`);
- third-party data that the user downloads: the Mouse Kidney Atlas (Zenodo record 17395591), MitoCarta3.0, the
  Kidney Tubule Expression Atlas ([URL]) and a bulk proteome of mouse kidney on a low-potassium diet ([REF]).

## Running a step

Start every script from the folder that holds `scripts/`:

```
python scripts/dct_intake_intake.py
```

- Outputs go to `results/tables/<stage>/`, `results/figures/<stage>/` and `results/h5ad/`. Every output file name
  starts with the name of its stage.
- A step does not replace its earlier outputs unless it is started with `--overwrite`.
- Every step writes a manifest (`results/tables/<stage>/<stage>_step_<step>_manifest.json`) with its inputs and their
  SHA-256, the package versions, the seed and the parameters.
- Random seed: 42.

## Run order

Run the steps in the order of the column `run_order` of `SCRIPTS.csv`. The order follows the files each step read
when the analyses were run. The stages, in that order:

| stage | content | run steps |
| --- | --- | --- |
| `kidney_object` | kidney: object from the Spectronaut export and the stored embedding | 2 |
| `qc_mito` | QC: MitoCarta fraction, stress panel, exclusion tests | 4 |
| `preprocessing` | preprocessing sensitivity (normalization, scaling, protein selection) and the Spectronaut quantity | 3 |
| `cluster_de` | annotation of clusters from their differential proteins: the rule, and the kidney marker coverage | 1 |
| `dotplot` | marker dotplots: marker order and values | 1 |
| `kidney_rna` | kidney: RNA atlas comparison | 6 |
| `kidney_annotation` | kidney: cluster annotation | 11 |
| `kidney_figure2` | kidney: Figure 2 panels | 3 |
| `atlas_dct` | RNA atlas: DCT subset and DCT1/DCT2 split; library of the DCT RNA comparison | 2 |
| `pt_tal` | kidney: PT and TAL subtypes | 7 |
| `rat_segments` | kidney: rat segment proteome comparison | 5 |
| `rna_protein_integration` | kidney: joint RNA-protein embedding | 6 |
| `dct_intake` | DCT: intake, filtering, objects | 4 |
| `dct_labels` | DCT: cluster annotation | 7 |
| `dct_reviewer` | DCT: reviewer analyses per diet | 16 |
| `dct_rna` | DCT: RNA atlas comparison per diet | 6 |
| `zerok_vs_control` | DCT: zero-K vs control | 9 |
| `dct_prolif` | DCT: per-cell proliferation call | 9 |
| `deposit` | processed-data deposit: build and validation | 2 |
| `report` | figures and tables of the manuscript and the supplement | 52 |

Libraries that several stages share: `scp_common.py` (paths, output guard), `cluster_de_lib.py` (annotation of
clusters from their differential proteins), `dotplot_lib.py`, `preprocessing_lib.py`.

## What can be rerun from the public inputs

`SCRIPTS.csv`, column `inputs`, states this for every run step, from the inputs each step recorded when it was run:
46 of the 156 run steps read only public inputs and outputs of such steps.

- **DCT cells, from the report to the cell annotation and the proliferation call.** These 15 steps were rerun in a
  copy of this folder from the report alone, in this order: `dct_intake_intake`, `dct_intake_objects`,
  `dct_labels_submitted_pipeline`, `dct_labels_primary_labels`, `dct_labels_transfer`, `dct_labels_stability`,
  `dct_labels_depth_plate`, `dct_labels_figures`, `dct_labels_canonical`, `dct_reviewer_markers`, `dct_prolif_intake`,
  `dct_prolif_route_a`, `dct_prolif_acceptance`, `dct_prolif_route_b`, `dct_prolif_canonical` (about 8 minutes).
  In that rerun the cell types, the clusters, the transferred labels and the proliferation calls of all 1,431 cells equalled the deposited cell tables (15 checks of `checks/check_dct_labels.py`, none failed).
- **Zero-potassium against control diet** (`zerok_vs_control`, and `dct_intake_coverage.py`). These steps read the
  panel of proteins that was fixed before the diets were compared. The panel table is not part of this repository;
  its content is in Supplementary Table S18 of the manuscript.
- **Total kidney.** `kidney_object_kidney_canonical.py` builds the kidney object from the report and from two inputs
  that are in no public archive:
  - the processed object of an earlier preprocessing run of the same report (filter, ln(1 + x), scaling, principal
    components, graph of the 10 nearest neighbours on 20 components, UMAP, Leiden clustering at resolution 2). The
    kidney object of the Zenodo record holds the same matrix, the first 20 principal components, the graph and the
    UMAP;
  - the per-cell tables that `first_submission/scp_kidney_upload.py` wrote.

  It also reads two annotated objects of that earlier run, for a comparison of labels only. All later kidney steps
  read the object this step writes.
- **Steps that were run for two data sets at once.** Some early steps (`qc_mito`, `preprocessing`, `dotplot`,
  `kidney_object_filtering_summary.py`) were run for the total-kidney data together with the DCT data of the first
  submission, which the revised paper does not analyse. They are given as run.

## Reproducing the clusters

Clusters come from `scanpy.tl.leiden(..., flavor="igraph", random_state=42)` on the stored neighbour graph. On Windows,
scanpy 1.11.3 needs 64-bit random integers in its igraph random-number adapter for the seed to take effect;
`_scanpy_compat.py` sets this, and the scripts call it. A different platform or package version can return a
different partition. The cluster column of the processed data (Zenodo) is the reference.

## Names in the code

- **Columns of the intermediate tables** keep the names they had during the analysis. `kidney_t15_label` and
  `t15_cluster` are the kidney annotation; `cs_ctrl_deg_label` (control) and `cs_zerok_pipeline_label` (zero-K) are
  the DCT annotation; `cs_zerok_transfer_label` is the transferred label; `cs_t31nf_*` is the per-cell proliferation
  call. Short module aliases such as `t5` or `t27` in the code are of the same kind. The processed data use public
  names (`cell_type`, `cluster`, `prolif`); the data dictionary of the Zenodo record defines them.
- **Figures and tables.** `SCRIPTS.csv`, column `item_of_the_paper`, gives the figure or table of the manuscript that
  each script makes, and the comments and docstrings of the scripts use the names of the manuscript. The texts that a
  script writes name a figure or a table by the label it had while the analyses were made (`S-02-d`, `Figure F-08`,
  `Figure 3A`), and so do the names of result files, check tables and intermediate columns. The scripts that write
  the supplementary table workbooks replace every label by the name of the manuscript when they write a cell, a sheet
  name or the title: `scripts/report_item_names.py` holds the rule and reads the names from
  `data/paper_item_names.csv`.
- **Unknown** in the DCT tables is written "DCT, subtype not resolved" in the processed data and in the manuscript.

## Changes made for publication

- File names, module names, output folders and comments were rewritten by fixed rules.
- `_scanpy_runtime.py` only sets cache folders.
- `scp_common.py` writes the two marker workbooks that the scripts read from the marker table of the manuscript. The
  cells are those of the workbooks used in the analyses.
- The steps no longer check the checksums of the authors' analysis plans. `dct_intake_intake.py` still checks the
  checksum of the DCT report (`data/input_checksums.csv`).
- Three comparisons with tables of an earlier search of the same DCT runs are skipped when those tables are absent:
  the cell set in `dct_intake_intake.py`, the scored marker set in `dct_labels_submitted_pipeline.py` and the marker
  order in `dct_labels_figures.py`. None of them enters a result.

## First submission

`first_submission/` holds the two scripts of the first submission and a note on them. The DCT data of the first
submission are not analysed in the revised paper.

## Licence and citation

The code is available under the MIT licence (`LICENSE`). Please cite the manuscript (title and authors at the top of
this page).
