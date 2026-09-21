# Final/headline skin and kidney run audit

**Status: read-only evidence audit.** The audit tooling does not rerun models,
rewrite H5ADs, or alter existing run outputs. Real H5AD processing must be
submitted to HiPerGator with one CPU and 64 GB RAM:

```bash
scripts/submit_final_headline_runs_audit.sh --render-only
scripts/submit_final_headline_runs_audit.sh
```

The launcher writes a generated SLURM job and the Python audit publishes
`final_headline_runs_audit.json`, `source_summary.csv`, `run_summary.csv`,
`protocol_summary.csv`, and `fold_prediction_summary.csv` transactionally. Paths are CLI-configurable and default
to the HPG skin 1 mm source, kidney spatial/reference H5ADs, the 1 mm split,
the 1000/750/500 full-sweep run directories, and the kidney poisson75 run.
No GPU is requested.

## What is checked

In backed, read-only mode the audit records source shapes and checks required
patient, disease, and FOV metadata; every patient has one disease; FOV counts
are reported; and the declared cohort sizes are enforced (skin **14**,
kidney **6**). Run directories are checked for valid manifest/run-summary
JSON, post-NMF artifacts, MLP metadata (`MLP mode`, units, and group counts),
and `fold_predictions.csv` when present. Modern leakage-safe prediction
headers require report metadata for `MLP mode`, `Evaluation unit`, and `Outer
CV mode`. Unit and item/group counts are derived from `fold_predictions`; a
reported group count is cross-checked only when it is explicitly declared.
Legacy or discontinued hyperparameter-search artifacts are classified
separately rather than granted the modern contract. Fold predictions are
checked for unique item IDs, one fold per test group, group-label consistency,
exactly two labels including the declared positive class, finite probabilities
in `[0,1]`, and row-level threshold/prediction agreement. The pooled confusion matrix,
accuracy, balanced accuracy, macro F1, and weighted F1 are recomputed rather
than trusted from text reports. Nested-CV metadata parses unique
`Processing Outer Fold i/N` identities, rejects duplicates/inconsistent totals
or out-of-range IDs, and requires the final performance marker for completion.

The JSON and CSV outputs intentionally distinguish **technical artifact
consistency** from **scientific citability**. A technically coherent file is
not thereby an independent or publishable estimate.

## Current evidence register

These values are the current documented headline evidence and are recorded in
the audit JSON as a documented evidence register. When a corresponding
ProtocolComparison JSON exists, the audit independently verifies its observed
metrics and fold count against these values (within tolerance) and fails on a
mismatch:

- **Skin 1 mm honest compact LOGO:** pooled accuracy **0.634146**; macro F1 and
  balanced accuracy **0.608467**.
- **Skin SGKF3:** near chance, accuracy **0.530488** and balanced accuracy
  **0.499205**.
- **Kidney honest nested:** accuracy **1.0**, but **non-citable**: only six
  patients and chance separability make this a sample-size artifact, not a
  generalization claim.
- **Full sweep:** completed evidence is only selection-biased `evaluate_fixed`
  output: pooled macro F1 is **0.44** (1000), **0.61** (750), and **0.59**
  (500). The original nested runs are incomplete at **8/14**, **5/14**, and
  **9/14** folds (1000/750/500), respectively.

`evaluate_fixed` and `tune_once` results are classified as exploratory and
selection-biased. Files marked `DISCONTINUED` are non-citable. RCausalMGM
disease edges are exploratory because FOVs pseudoreplicate patients.
ProtocolComparison_skin, ProtocolComparison_skin_sgkf3, and
ProtocolComparison_kidney JSON files are summarized when present, with honest
and leaky metrics kept separate; leaky/search-on-the-same-fold metrics are not
honest estimates.

The audit is an evidence inventory and consistency gate. It does not upgrade
any result's scientific status, select a threshold, repair incomplete outputs,
or choose a model.
