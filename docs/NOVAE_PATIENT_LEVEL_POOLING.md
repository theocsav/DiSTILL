# Historical-164 predeclared patient-level pooling

This is a **prediction-pooling** report, not retraining on 14 patient feature
vectors. The script reads only completed immutable `fold_predictions.csv` files
from the validated full comparison and the completed classifier ablation, plus
their run/config/aggregate manifests:

- full primary: `/blue/.../runs/novae_nmf_comparison_20260922T214308Z_1310236/historical164_nmf_vs_novae`
- ablation root: `/blue/.../runs/novae_classifier_ablation_20260923T010701Z_3910105/historical164_classifier_ablation`
- configurations: `full`, `composition_only`, `composition_enrichment`, and `composition_niche`
- arms: NMF and NOVAE

## Frozen protocol (declared before results)

Every source must contain the same 164 item IDs in the same order, labels,
patient groups, outer folds, positive class, and fixed 0.5 threshold. Sources
must have healthy/systemic-sclerosis FOV counts 61/103, 14 patients (4 healthy,
10 systemic sclerosis), one class per patient, each patient in exactly one of 14
outer folds, and finite probabilities in `[0, 1]` whose stored labels agree
with the threshold. Hash inventories in the immutable manifests are checked
before any pooling and all consumed inputs are checked again afterward. The
full-primary, aggregate, and per-configuration manifests must also agree on
protocols, configuration order, input provenance, code hashes, environments,
and the immutable full-primary path/hash.

For each patient, the three predeclared methods are applied to held-out FOV
predictions only:

1. **Primary:** arithmetic mean of systemic-sclerosis probabilities; positive
   at `>= 0.5`.
2. **Sensitivity 1:** median probability; the same threshold.
3. **Sensitivity 2:** majority vote of stored FOV labels. Exact ties use the
   mean probability `>= 0.5`, and are explicitly recorded.

No threshold is tuned and no method or configuration is selected from results.
The underlying classifier was trained/evaluated at FOV level with patient-held-
out outer folds; pooling gives each patient equal endpoint weight but does not
turn training into patient-level training.

## Historical context and interpretation

The prior approximately `0.608` result was an older **compact LOGO balanced
accuracy** result, not merely an overall-accuracy number. This historical-164
NMF-versus-NOVAE comparison is a stricter new comparison and is not a direct
reproduction of that result. NOVAE remains exploratory (`reference=all`).

The output explicitly warns that `n=14` (4/10) is coarse exploratory evidence;
no p-values or significance claims are reported. Results should not be treated
as independent FOV-level evidence.

## Outputs

`scripts/run_novae_patient_level_pooling.py` atomically creates a new output
root and refuses overwrite. It writes:

- `patient_predictions_long.csv` (14 patients × 2 arms × 4 configurations × 3 methods),
- recomputed `patient_metrics_long.csv`, primary mean CSV/JSON summaries,
- long and per-method confusion matrices,
- paired NMF/NOVAE correctness and per-configuration descriptive tables, and
- `pooling_manifest.json` with input, output, and code SHA256 inventories.

No source directory is modified.

## HPG launcher and checks

Render or submit the one-node CPU job:

```bash
scripts/submit_novae_patient_level_pooling.sh --render-only
scripts/submit_novae_patient_level_pooling.sh
```

The launcher uses the fixed HPG paths above, `ibd_cosmx_k4`, one CPU, 16 GB,
GPU disabled, safe path/environment checks, and a new timestamped output root.
It renders a job without submitting when `--render-only` is used. No local real
data is required for the synthetic tests.

On Windows, run the repository Conda pytest/Ruff/compile checks. On a POSIX
host, syntax-check and render the launcher with Bash. The pooling script can
also be invoked directly with explicit paths:

```bash
python scripts/run_novae_patient_level_pooling.py \
  --full-primary-dir /blue/.../historical164_nmf_vs_novae \
  --ablation-root /blue/.../historical164_classifier_ablation \
  --output-root /blue/.../runs/novae_patient_level_pooling_<timestamp>/historical164_patient_level_pooling
```
