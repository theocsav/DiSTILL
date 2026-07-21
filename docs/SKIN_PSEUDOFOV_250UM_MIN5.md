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

## Compact tuning profile

The first min5 tuning attempt generated 2,201 FOV rows but timed out after
48 hours because the default grid contains 2,160 parameter combinations.
With 14 patient groups, that requires approximately 30,240 grouped MLP fits,
each allowed up to 1,000 sklearn iterations.

The exploratory rerun uses a compact, predeclared grid with 64 combinations
and a 300-iteration cap. This changes the hyperparameter search budget, not
the scientific evaluation protocol: patient-grouped leave-one-group-out
evaluation, macro-F1 selection, minority oversampling, and the FOV unit are
unchanged. Results from this 250um arm should therefore be labeled
exploratory and compared primarily as a sensitivity analysis.

## 250um min5 classification result

Jobs 37656357 and 37656358 completed successfully on July 20, 2026. The
input builder produced 2,201 FOV rows across 14 patient groups. The compact
tuning selected:

- hidden layers: 32, 16, 8
- activation: relu
- alpha: 0.01
- learning rate: 0.001
- batch size: 16
- grouped full-data selection score: 0.405

Aggregate held-out results:

- accuracy: 0.38
- balanced accuracy: approximately 0.36
- macro-F1: 0.34
- healthy F1: 0.16
- systemic-sclerosis F1: 0.51

Confusion matrix, rows true and columns predicted:

- healthy: 132 correct, 843 called systemic sclerosis
- systemic sclerosis: 715 correct, 511 called healthy

The compact arm does not support the hypothesis that smaller FOVs improve
classification. It is substantially weaker than the completed 500um and
750um arms and should be reported as an exploratory negative sensitivity
result. The legacy fold-summary lines in older mlp_results.txt files label
balanced accuracy as Accuracy and weighted F1 as F1; the aggregate
classification report is the primary metric summary for this result.