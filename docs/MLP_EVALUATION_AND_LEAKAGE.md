# MLP Evaluation and Data Leakage

Status: active
Last reviewed: 2026-07-28

This document records why one family of MLP scripts was discontinued, how large the
resulting bias was, what the leakage-safe evaluation actually reports, and what the
remaining caveats are. It is the reference for any classification number that leaves
this repository.

Nothing was deleted. Discontinued code and outputs are retained for provenance and
marked in place.

---

## 1. Summary

Two MLP implementations exist and they disagree substantially on identical data.

**Measured on skin, 2026-07-28**, with the same features, targets and patient
groups in both arms so that the evaluation protocol is the only variable
(`scripts/compare_evaluation_protocols.py`, 164 FOVs / 14 patients / 85 features,
24-candidate grid):

| Outer CV | Leaky reported | Honest pooled | Inflation |
|---|---|---|---|
| leave-one-patient-out | 0.840 | 0.634 | **+0.206** |
| `sgkf3` (the published setting) | 0.686 | 0.531 | **+0.155** |

In both, the leaky arm's `Best F1-Score` and `Mean F1-Score` agree to three
decimals, reproducing the signature visible in the manuscript output (0.6978 and
0.698). They agree because they are the same computation.

The published IBD three-class figure is 0.698 weighted F1 under `sgkf3`; skin's
leaky `sgkf3` figure is 0.686. That parallel is suggestive, not evidence: different
organ, different cohort.

**Honest current state.** Under leave-one-patient-out on skin, the honest nested
estimate reaches 0.608 balanced accuracy against a 0.5 chance line - weak but real
discrimination (healthy recall 0.51, SSc recall 0.71). Under `sgkf3` it falls to
0.499, i.e. chance, because each fold trains on roughly 9 patients instead of 13.
Accuracy alone is misleading here: 0.634 against a 0.628 majority baseline looks
like nothing, but on imbalanced classes balanced accuracy is the right comparison.

**Correction, 2026-07-28.** An earlier version of this document reported kidney as
1.000 accuracy under the discontinued script versus 0.33 under the leakage-safe
one, and attributed that gap to leakage. That was wrong. Those two runs differed in
feature construction as well as protocol, so the comparison was confounded. Running
both protocols over the *same* kidney feature table returns **1.000 from both**.

The kidney result is a sample-size artifact, not a leakage artifact, and it is
quantifiable. With 6 samples split 3/3, a continuous feature perfectly separates the
classes for exactly 2 of the C(6,3) = 20 possible label assignments, so P = 0.1 per
feature. With 106 features, about **10 features separate the classes perfectly by
chance alone**. Any classifier finds one. At that shape 1.000 is the expected result
under the null. No evaluation protocol fixes it; only more subjects would.

Kidney should therefore carry no classification number at all, in either direction.

---

## 2. The defect in the discontinued script

`pipeline_assets/IBD_MLP_44Features.py`, lines ~117-147:

```python
random_search = RandomizedSearchCV(
    estimator=pipe, param_distributions=param_distributions,
    n_iter=30000, cv=cv_splitter, scoring='f1_weighted', ...
)
random_search.fit(X, y, groups=groups)        # searches the FULL dataset

# ...

for i, (train_idx, test_idx) in enumerate(split_iterator()):   # SAME cv_splitter
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
```

`cv_splitter` is a single `StratifiedGroupKFold(n_splits=3, shuffle=True,
random_state=42)` (or `LeaveOneGroupOut`) instance, so `split_iterator()` yields the
identical folds the search already optimised over. The "final cross-validation" is a
re-run of the winning configuration on the folds that selected it.

With `n_iter=30000`, the reported score is the maximum over 30,000 configurations
evaluated on those same folds. At the cohort sizes here that is largely fitting fold
noise.

### Direct evidence

`skin_visium_manuscript_package/tables/mlp_results.txt`:

```
Best F1-Score: 0.6978          <- hyperparameter search
Mean F1-Score: 0.698           <- "independent" evaluation
```

The same quantity computed twice. `MLP_44Features_full/mlp_results.txt` shows the
pathology more starkly: 90,000 fits, `Best F1-Score: 0.7778`, per-fold accuracies
`[1.0, 0.5, 1.0]`.

### Scope

Same pattern, all discontinued:

- `pipeline_assets/IBD_MLP_44Features.py`
- `pipeline_assets/scripts/IBD_MLP_44Features.py`
- `pipeline_assets/scripts/IBD_MLP_51Features.py`
- `pipeline_assets/scripts/IBD_MLP_FewerParams.py`

---

## 3. What the leakage-safe script gets right

`pipeline_assets/IBD_MLP_LeakageSafe.py` was audited specifically for leakage. These
are correct and should not be "simplified" away:

- **Patient grouping.** `LeaveOneGroupOut` on patient for outer CV, and grouped
  splits for inner tuning. No patient's FOVs span train and test.
- **In-fold feature selection.** `_select_features` is called per outer fold on
  `y_train.index` only (lines ~721-728). Mutual information never sees test labels.
- **In-fold scaling.** `StandardScaler` is fit inside `model.fit()` (torch backend)
  or inside the sklearn `Pipeline`. Never fit on the full matrix.
- **Training-only resampling.** `_maybe_balance_training_data` is applied to the
  training split only. Early stopping uses *training* loss with no internal
  validation split, so oversampled duplicates cannot contaminate a validation set.
- **No upstream leakage.** The post-NMF notebook computes FDR-filtered
  `selected_enrichment_features`, but the tables handed to the MLP
  (`enrichment_features_fov.parquet`, `niche_gene_features_fov.parquet`) are written
  from the *unfiltered* column lists. The only thing taken from
  `combined_features_filtered.parquet` is the `nmf_prop_*` columns, which are
  unsupervised. The MLP therefore selects from the full candidate pool in-fold.
- **Honest labelling.** The full-data SHAP model prints
  `"not part of unbiased evaluation"`.

Statistics elsewhere in the pipeline are also correct: the post-NMF notebook applies
Benjamini-Hochberg via `multipletests(..., method="fdr_bh")` and filters at
`p_adj < 0.05`.

---

## 4. Remaining caveats in the leakage-safe path

### 4.1 `tune_once` -> `evaluate_fixed` carries selection bias

The workflow documented in `docs/SKIN_KIDNEY_MLP_FINDINGS_2026-07-03.md` is:

1. `tune_once` - one grouped hyperparameter search **across all patients**
2. `evaluate_fixed` - LOGO evaluation reusing those parameters
3. optional `explain`

Step 1 sees every patient, including each one later held out in step 2. With the
`expanded` grid (13 x 2 x 9 x 5 x 3 = **3,510 configurations**) that bias is not
negligible.

This exists for a real reason: full `nested_cv` is O(P^2 x |grid|) model fits, which
was too slow for iteration. The tradeoff is legitimate for exploration and not
legitimate for reporting.

**Important:** this bias inflates results. The skin deliverable still lands at macro
F1 0.51 *with* the bias present. The negative conclusion is therefore robust -
removing the bias can only lower the numbers.

Use `mlp_mode=nested_cv` for anything reported. `run_pipeline.py` now emits a
validation warning when `mlp_mode=evaluate_fixed` is configured.

### 4.2 Per-fold mean +/- std is misleading at patient level

With `mlp_unit=patient` and `LeaveOneGroupOut`, each fold contains exactly one
sample. `balanced_accuracy_score` on one sample is 0 or 1, so:

- "Mean Balanced Accuracy" degenerates to plain accuracy, **not** balanced accuracy,
  which is optimistic under class imbalance
- the reported `+/-` is Bernoulli spread over single predictions, not fold-to-fold
  variability of a metric

The pooled `classification_report` over `all_y_true` / `all_y_pred` is the correct
summary and is already produced. Prefer it. Figures in the style of
`0.774 +/- 0.161` (see `docs/paper.tex` lines ~309-312) should be replaced with
pooled metrics and an interval computed appropriately for the design.

### 4.3 Cohort size

Kidney at n = 6 patients cannot support a classifier claim regardless of methodology.
Skin at 14 patients supports FOV-level evaluation only weakly, and patient-level
evaluation not at all.

---

## 5. What was changed

Code and configuration:

- `run_pipeline.py`: `DEFAULT_MLP_SCRIPT` now points at `IBD_MLP_LeakageSafe.py`.
  No preset relied on the previous default, so no existing run changes behaviour;
  this prevents new presets from silently inheriting the discontinued script.
- `run_pipeline.py`: validation warns on a discontinued MLP script, on a preset
  marked `status: discontinued`, and on `mlp_mode=evaluate_fixed`.
- The four discontinued scripts carry a `DISCONTINUED` header. The top-level one
  also emits a `DeprecationWarning` and prints a banner at runtime. They remain
  executable so historical runs stay reproducible.
- Eight presets referencing the discontinued script are marked with
  `"status": "discontinued"`, `"status_reason"`, and `"superseded_by_script"`.
  They were **not** repointed, because silently swapping the script would change
  what those presets mean.

Outputs:

- Every directory containing a leaked `mlp_results.txt` has a `DISCONTINUED.md`
  marker. Files are retained, not deleted.

Affected output directories:

```
kidney_dataset/loocv_report/MLP_44Features/
kidney_dataset/new_report/MLP_44Features/
kidney_dataset/patched_report/MLP_44Features/
kidney_dataset/real_shap_report/MLP_44Features/
kidney_dataset/report/MLP_44Features/
kidney_dataset/shap_report/MLP_44Features/
MLP_44Features_full/
skin_visium_manuscript_package/tables/
```

A leaked result file is identifiable by its header:

```
--- Starting Hyperparameter Search with RandomizedSearchCV ---
```

A leakage-safe result file begins:

```
--- Starting Leakage-Safe Nested Grouped Evaluation ---
```

---

## 5a. Executable guard

`tests/test_evaluation_leakage.py` encodes the invariant that this whole document
rests on:

> On data where the label is independent of every feature, an honest evaluation
> must land at chance. Anything above chance is leakage.

It contains:

- `test_leakage_safe_evaluation_is_chance_on_null_data` - runs the real
  `IBD_MLP_LeakageSafe.py` on a synthetic null cohort and asserts pooled accuracy
  stays near chance. It measured 0.48 on 12 patients x 8 FOVs of pure noise.
- `test_leakage_safe_evaluation_detects_real_signal` - positive control, so the
  null test cannot pass merely because the evaluation is broken.
- `test_nested_cv_is_chance_on_null_data` - same invariant through the full nested
  path including inner tuning. Marked `slow`; run with `pytest tests -m slow`.
- `test_search_then_report_on_same_folds_inflates_score` - reproduces the
  discontinued pattern and asserts that reporting the search maximum exceeds the
  median candidate on label-independent data, across five draws. This keeps the bug
  class visible rather than only described.

Run with `pytest tests`. Wired into CI as `.github/workflows/pipeline-tests.yml`.

Note on why this matters: the discontinued script's defect would have been caught by
the first test years earlier. A permuted-label run is the cheapest possible check on
an evaluation pipeline, and it is now automatic.

## 6. Regenerating a citable result

1. Choose a preset backed by `pipeline_assets/IBD_MLP_LeakageSafe.py`.
2. Set `"mlp_mode": "nested_cv"`. Do not use `evaluate_fixed` for reported numbers.
3. Consider `"mlp_grid_profile": "compact"` to keep the nested search tractable
   (4 x 2 x 2 x 2 x 2 = 64 configurations rather than 3,510).
4. Prefer `"mlp_unit": "fov"` over `patient` at current cohort sizes.
5. Report the pooled classification report, not per-fold mean +/- std.
6. Compare against the majority-class baseline explicitly. For skin FOV that
   baseline is 0.628 accuracy.

---

## 6a. The published IBD result cannot be re-evaluated here - BLOCKED

The IBD numbers in `original_paper.tex` and reproduced in the DiSTILL manuscript
(`0.774 +/- 0.161` three-class, `0.916 +/- 0.118` two-class) were produced by
`IBD_MLP_44Features.py` with `cv_mode: sgkf3`. Re-running them honestly is not
possible from the UF HiPerGator filesystem, because the input artifacts are absent.

What was checked, 2026-07-28:

- `runs/ibd_cosmx_k4/output` holds **9 rows x 14 features**, and the features are
  cell morphology and QC summaries (`diameter_um_mean`, `Area_std`,
  `CenterX_global_px_mean`, ...). The paper reports **171 FOVs x 44 niche
  features** with support 55/62/54. Different unit, different count, different
  features - this is not the published analysis.
- `post_nmf_artifacts.json` lists only `reduced_features_final_15.parquet`,
  `targets_y.parquet`, `groups.parquet`, `combined_features_filtered.parquet`.
  No `enrichment_features_fov`, no `niche_gene_features_fov`.
- The hardcoded fallback in `IBD_MLP_LeakageSafe.py`,
  `/blue/kejun.huang/tan.m/IBDCosMx_scRNAseq/CosMx/Post-NMF_Analysis`, now contains
  only an `RCausalMGM` subdirectory.
- A filesystem-wide search for `combined_features_filtered.parquet` and
  `enrichment_features_fov.*` returns skin and kidney runs only.

`presets/ibd_cosmx_mlp_leakagesafe_{threeclass,twoclass}.json` are therefore marked
`"status": "blocked"`. They are correct in structure and ready to run if the
original artifacts are recovered from the first author; only `output_dir` needs
repointing.

### This is a reproducibility finding in its own right

The published IBD analysis is not regenerable from any preset in this repository.
For a manuscript whose central claim is reproducible orchestration, and which uses
this IBD workflow as its case study, that gap is worth addressing directly - and it
is independent of whether the numbers themselves are correct.

## 6b. The substitute experiment: measure the protocol delta where data exists

`scripts/compare_evaluation_protocols.py` runs both protocols over the *same*
feature matrix, targets, and patient groups, so the only variable is how
hyperparameters are selected. Skin and kidney runs have complete artifacts, so the
magnitude of the bias can be measured there.

```bash
RUN=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs
python scripts/compare_evaluation_protocols.py     --features $RUN/MLP_FOVFeatures_inputs/combined_features_filtered.parquet     --targets  $RUN/MLP_FOVFeatures_inputs/targets_y.parquet     --groups   $RUN/MLP_FOVFeatures_inputs/groups.parquet     --outdir   $RUN/ProtocolComparison     --outer-cv logo --grid compact --jobs 8
```

It reports, side by side:

- the honest nested estimate, pooled
- the leaky `Best F1-Score` and the leaky `Mean F1-Score`, which agree because they
  are the same computation - this reproduces the signature visible in a
  discontinued `mlp_results.txt`
- the median candidate score, showing what is being maximised over
- the majority-class baseline

On a synthetic null cohort (48 rows, 8 patients, label independent of every
feature) with only 24 candidates, it measures `Best F1-Score 0.581` and
`Mean F1-Score 0.581` against an honest pooled `0.357`. The published run used
30,000 candidates.

## 7. Open items

- Replace the per-fold mean +/- std figures in `docs/paper.tex` with pooled metrics.
- Decide the fate of the numbers currently in `skin_visium_manuscript_package/`.
  They are not citable as they stand.
- The remaining stages (`cell2loc`, `nmf`, `post_nmf`, `rcausal_mgm`) and
  `apps/api/app/validation.py` have not had an equivalent review.
