# Historical-164 NOVAE classifier ablation

This is a separately named, predeclared exploratory ablation of the validated
historical-164 NMF-versus-NOVAE classifier comparison. It does **not** modify
the frozen primary runner (`scripts/run_novae_nmf_comparison.py`) or copy the
completed full result from job **43018668**.

## Frozen design

All four rows use the same 164 pseudo-FOV rows, target values, patient groups,
patient-LOGO outer folds (14 folds), compact sklearn nested CV, weighted-F1
selection, no resampling, decision threshold 0.5, 1,000 maximum epochs, seed
42, and SHAP disabled. The four predeclared feature-family configurations are:

| ablation | top enrichment | top niche-gene |
|---|---:|---:|
| `composition_only` | 0 | 0 |
| `composition_enrichment` | 5 | 0 |
| `composition_niche` | 0 | 20 |
| `full` (existing reference) | 5 | 20 |

Composition is always selected. The allowed candidate union is exactly
composition + enrichment + niche-gene features; no other feature family is
permitted. NOVAE remains `reference=all` and exploratory. This design does not
make a FOV-independence claim and reports no p-values.

## Run and resume

Render or submit the CPU-only SLURM job with:

```bash
scripts/submit_novae_classifier_ablation.sh --render-only
scripts/submit_novae_classifier_ablation.sh
```

Set `NOVAE_CLASSIFIER_ABLATION_FULL_PRIMARY_DIR` to the immutable
`historical164_nmf_vs_novae` output from job 43018668. The launcher fixes one
node, two CPUs, 96 GB, one thread per evaluator, GPU disabled, and a safe
12--24 hour walltime. Scientific `NICHERUNNER_*` overrides, unsafe path
characters, input/output overlap, and collisions fail closed.

The Python runner is `scripts/run_novae_classifier_ablation.py`. It validates
the full reference before starting new work: the full manifest protocol and
all three primary code hashes (evaluator, primary orchestrator, and primary
launcher), exact input paths and hashes, complete output hash inventory, and
all 14-fold artifacts and prediction alignment are required. The exact
atomic `run_manifest.json` plus complete output hash inventory serves as the
full-postflight completion gate. The full directory is read-only.

Each new configuration is written to an immutable `output_root/<ablation>`
directory only after both concurrent one-thread evaluator arms pass all
checks. A failed later configuration removes only its temporary directory, so
completed configurations can be resumed by invoking the Python runner again.
The cross-configuration `aggregate` directory is created last with an atomic
rename. It contains:

- `pooled_metrics_long.csv` and `ablation_metric_matrix.csv` (representation ×
  ablation pooled metrics and NOVAE-minus-NMF deltas),
- `paired_predictions_long.csv` and `per_patient_metrics_long.csv`,
- `feature_stability_long.csv`, and
- an aggregate manifest with compact consumed input/code/output hashes and an
  explicit unchanged full-reference path.

No output is copied into the full reference directory.

## Tests

Synthetic/mocked tests cover the four configuration declarations, strict
full-reference code/hash and exact atomic manifest/output-inventory
validation, dynamic candidate maxima and protocol mismatch rejection, resumable
per-ablation publication, atomic aggregate behavior, long-form matrix
construction, and launcher rendering and
safety. They do not open real H5AD/feature data. On Windows use the repository's
Conda Python/pytest and Ruff paths; the launcher can be syntax-checked and
rendered with Bash:

```bash
conda run -n <env> pytest -q tests/test_novae_classifier_ablation.py
bash scripts/submit_novae_classifier_ablation.sh --render-only
ruff check scripts/run_novae_classifier_ablation.py tests/test_novae_classifier_ablation.py
```
