# Skin 250um min5 exploratory arm

This arm tests whether the 250um pseudo-FOV classification can run when FOV
filtering retains FOVs with at least 5 total cells. It reuses the completed
250um NMF output and writes to a separate `fullsweep_min5` output directory.

The neighborhood-enrichment implementation still requires at least 2 cells
per FOV with finite spatial coordinates and finite area-derived diameters.
This is an exploratory classification check and is not a replacement for the
primary 500um and 750um analyses.

Submit from HiPerGator:

```bash
bash scripts/submit_skin_pseudofov_250um_min5.sh
```

The chain is `post_nmf -> mlp_tune_once -> mlp_eval_fixed/report`. It does not
rerun cell2location or NMF.

## Failure diagnosis and fix

The first 250um min5 post-NMF jobs failed before classification. The retiled
H5AD, NMF H5AD, and metadata CSV each contained 13,417 rows with matching
unique_cell_id values and the expected spatial columns, so the source data
were not the problem.

The post-NMF metadata helper used a pandas merge on the cell key. That
operation reset the left observation index to a RangeIndex; the subsequent
reindex by the H5AD cell IDs therefore replaced patient, FOV, and coordinate
metadata with NaN. This appeared as one unknown_patient_nan FOV with zero
valid coordinates and caused neighborhood-enrichment construction to return
no FOV-level rows.

The fix preserves the observation index by joining metadata on _join_key.
The scientific protocol is unchanged: retain FOVs with at least 5 total
cells, and require at least 2 cells with finite coordinates and
area-derived diameters for neighborhood enrichment. The rerun should verify
that post_nmf_obs.csv contains multiple patient/FOV keys and non-null
coordinates before classification begins.

## Reused NMF artifact wiring

The post-NMF stage uses the completed 250um NMF artifact from the original
full-sweep output directory. The min5 MLP presets now set
mlp_cosmx_with_nmf_path explicitly, and run_pipeline.py passes that path to
the FOV input builder. This keeps the filtered min5 feature tables separate
without copying the large NMF H5AD.