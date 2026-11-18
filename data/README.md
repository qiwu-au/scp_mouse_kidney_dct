# Inputs

Only two small files are part of this repository. Everything else is downloaded by the user and placed as shown.

## Included

| file | content |
| --- | --- |
| `marker_lists_Supplementary_Table_S2.xlsx` | the marker lists of the manuscript (Supplementary Table S2): tab "Markers of kidney" with 17 cell types, tab "Markers of DCT" with 3 |
| `input_checksums.csv` | size and SHA-256 of the combined DCT report; `scripts/dct_intake_intake.py` stops if the file differs |

## Written by the code

The scripts read the marker lists from two small workbooks. `scripts/scp_common.py` writes them from the table above
the first time a script runs; they are not part of the repository.

| file | content |
| --- | --- |
| `KIDNEY/KIDNEY_markers_V3.xlsx` | the first 16 cell types of the tab "Markers of kidney", the same rows in the same order. The 17th row of the table, Endothelial cells (Pecam1, Emcn, Eng, Esam, Icam2, Fabp4, Ehd4), is added in the code (`ENDOTHELIAL_MARKERS` in `scripts/kidney_annotation_scan_lib.py`) by the steps that used it, because some steps compare the annotation without and with that row |
| `DCT/DCT_markers_V1.xlsx` | the three rows of the tab "Markers of DCT" |

## To be added by the user

| file | source | place |
| --- | --- | --- |
| `DCT_Ctrl_ZeroK_combine_1535cells.csv`: Spectronaut protein-group report of the DCT cells, control and zero-potassium diet, one search of 1,535 runs | ProteomeXchange/PRIDE, [PRIDE accession of the new DCT raw data] | `DCT/` |
| `KIDNEY_768cells.csv`: Spectronaut protein-group report of the total-kidney cells, 768 runs | ProteomeXchange/PRIDE, PXD073729 | `KIDNEY/` |
| the processed data: four AnnData objects and their cell tables | Zenodo, [Zenodo DOI] | `zenodo/` |
| `mka.h5ad`: Mouse Kidney Atlas | Zenodo record 17395591 | `reference/` |
| `Mouse.MitoCarta3.0.xls`: MitoCarta3.0, mouse | Broad Institute, MitoCarta3.0 | `reference/` |
| `KTEA_proteome.txt`, `KTEA_proteome_individual_sample.txt`: Kidney Tubule Expression Atlas, rat tubule segments | [URL] | `reference/KTEA/` |
| `jci.insight.196339.sdd1.xlsx`: bulk proteome of mouse kidney on a low-potassium diet | supplementary data of [REF] | `reference/` |

The third-party files keep the licence and terms of their sources. They are not redistributed here.

## Not public

Some run steps read files that are in no public archive. `SCRIPTS.csv` (column `inputs`) names them per step, and
`README.md` in the top folder explains them ("What can be rerun from the public inputs").
