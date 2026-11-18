# First submission

The two scripts of the first submission of the manuscript, dated 11 February 2026, as in the first version of this
repository:

- `scp_kidney_upload.py`: total-kidney data (unchanged);
- `scp_dct_upload.py`: DCT data of the first submission. One comment was corrected: line 93 read
  "# Log-normalize the data" and now reads "# Log-transform the data: ln(1 + x); no per-cell normalization". The code
  of that line and the next (`sc.pp.log1p`) is unchanged; it never normalized per cell.

The DCT data of the first submission (`DCT_758cells.csv`, 758 runs) are not analysed in the revised paper. The DCT
results of the revised paper come from a new experiment (control and zero-potassium diet, 1,535 runs); see the top
folder.

The analyses of the revised paper read these two scripts as reference files (settings and order of the steps). They
do not run them.

Notes:

- The two input paths at the top of each script are placeholders (`PATH\TO\...`).
- `scp_dct_upload.py` still holds three output paths of the original run (lines 167, 190 and 2151).
- Comments and output tables of the revised analyses quote line numbers of these scripts from working copies that
  had 12 more lines at the top (15 for some quotes of `scp_dct_upload.py`). Subtract 12 (or 15) to find the line here.
