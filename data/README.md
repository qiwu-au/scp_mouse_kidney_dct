# Inputs

Only three small files are part of this repository. Everything else is downloaded by the user and placed as shown.

## Included

| file | content |
| --- | --- |
| `DCT/DCT_markers_V1.xlsx` | marker list of the DCT data (Supplementary Table S2 of the manuscript) |
| `KIDNEY/KIDNEY_markers_V3.xlsx` | marker list of the total-kidney data (Supplementary Table S2); the endothelial row is added in the code (`scripts/kidney_annotation_v3_endothelial.py`) |
| `input_checksums.csv` | size and SHA-256 of the combined DCT report; `scripts/dct_intake_intake.py` stops if the file differs |

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
