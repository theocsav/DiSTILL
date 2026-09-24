# Exact patient-level McNemar report (pre-result protocol)

This report is a small, reproducible paired analysis of the **completed,
immutable** patient-level pooling output. It does not retrain, re-pool, read FOV
features, or change the source output.

## Frozen input and contract

The fixed input is:

```text
/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_patient_level_pooling_20260924T161344Z_3258476/historical164_patient_level_pooling
```

Only `patient_predictions_long.csv` and `pooling_manifest.json` are consumed.
Before any calculation, the script verifies the manifest's complete output hash
inventory and the SHA256 of every declared file, including the patient table.
It verifies the source hashes again immediately before publication and records
that the source was unchanged.

The patient table must contain exactly 336 rows:

- four configurations: `full`, `composition_only`, `composition_enrichment`,
  `composition_niche`;
- two arms: `nmf`, `novae`;
- three methods: `primary_mean`, `sensitivity_median`,
  `sensitivity_majority_vote`;
- 14 patients, one row per patient in every configuration/arm/method cell;
- `healthy` and `systemic_sclerosis` labels (4 and 10 patients), aligned across
  cells, with one patient per outer fold 1--14.

Stored `correct` and boolean `tie` fields are parsed and validated. Correctness
is recomputed as `predicted_label == true_label`; a disagreement aborts the
report. Duplicate cell/patient rows, alignment differences, unexpected labels,
invalid scores, and missing cells abort the report.

## Primary comparison (predeclared)

The primary comparison is `configuration=full`, `method=primary_mean`: paired
NMF versus NOVAE correctness over the 14 held-out patients. The four paired
cells are emitted with unambiguous direction names:

- `both_correct`
- `both_wrong`
- `nmf_only_correct`
- `novae_only_correct`

The report also emits `n=14` and `discordant = nmf_only_correct +
novae_only_correct`. The exact two-sided McNemar p-value is computed as:

```python
scipy.stats.binomtest(
    nmf_only_correct,
    n=nmf_only_correct + novae_only_correct,
    p=0.5,
    alternative="two-sided",
).pvalue
```

When there are no discordances, the p-value is explicitly `1.0`. No asymptotic
chi-square approximation, mid-p value, one-sided headline, equivalence claim,
or significance claim is used.

## Secondary sensitivity analyses

The same exact paired table and test are computed for all 4 × 3 configuration
and pooling-method cells. These are descriptive, exploratory,
multiplicity-unadjusted sensitivities and are not selected from their results.
They are not used for a one-sided headline.

## Outputs and reproducibility

`scripts/run_novae_patient_level_mcnemar.py` refuses an existing output root and
publishes atomically. It writes:

- `exact_mcnemar_all.csv` (the 12 primary/secondary rows),
- `primary_exact_mcnemar.json` (the predeclared primary result and protocol),
- `paired_primary_patients.csv` (the 14 paired primary patient rows), and
- `mcnemar_manifest.json` (input/code/output hashes and warnings).

The one-CPU SLURM launcher is:

```bash
scripts/submit_novae_patient_level_mcnemar.sh --render-only
scripts/submit_novae_patient_level_mcnemar.sh
```

It targets the `ibd_cosmx_k4` environment, requests one CPU and 16 GB RAM,
disables GPU visibility, applies safe path/environment checks, and supports
render-only operation without local access to HPG data. Direct CLI invocation
also bootstraps the repository import path:

```bash
python scripts/run_novae_patient_level_mcnemar.py \
  --input-root /blue/.../historical164_patient_level_pooling \
  --output-root /blue/.../runs/novae_patient_level_mcnemar_<timestamp>/historical164_patient_level_mcnemar
```

Synthetic tests cover orientation, one-versus-zero and balanced discordances,
zero discordances, input hash/alignment/duplicate failures, atomic no-overwrite
behavior, CLI bootstrap, and launcher rendering. No real data is required
locally to run those tests.
