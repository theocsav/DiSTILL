# Skin/Kidney MLP Findings - 2026-07-03

## Objective
Track the current leakage-safe Distill MLP findings for skin and kidney, including exact run outputs, best current baselines, and next steps.

## Canonical workflow now in Distill

> **Update 2026-07-28.** The `tune_once` -> `evaluate_fixed` workflow below tunes
> hyperparameters across every patient, including those later held out, so metrics
> produced by it carry selection bias and are not citable. It remains fine for
> iteration. Use `mlp_mode=nested_cv` for any reported result.
> See [MLP_EVALUATION_AND_LEAKAGE.md](MLP_EVALUATION_AND_LEAKAGE.md).

The old exhaustive nested MLP path was too slow for iterative work. The current supported workflow is:

1. `tune_once`
2. `evaluate_fixed`
3. optional `explain`

This preserves patient-grouped leakage-safe evaluation while making iteration practical.

## Important code / presets

### Core code
- `run_pipeline.py`
- `pipeline_assets/IBD_MLP_LeakageSafe.py`
- `scripts/evaluate_mlp_thresholds.py`
- `pipeline_assets/rCausalMGM_Rscript_NicheComposition.R`
- `pipeline_assets/rCausalMGM_Rscript_NeighborhoodInteractions.R`

### Main preset families used
- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_tune_once_cpu.json`
- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_eval_fixed_cpu.json`
- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_tune_once_cpu_balanced.json`
- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_eval_fixed_cpu_balanced.json`
- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_tune_once_cpu_balanced_resampled.json`
- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_eval_fixed_cpu_balanced_resampled.json`
- `presets/kidney_cosmx_ssc_poisson75_hpg_mlp_tune_once_cpu.json`
- `presets/kidney_cosmx_ssc_poisson75_hpg_mlp_eval_fixed_cpu.json`

## Dataset scale used in the successful skin FOV workflow
- 164 FOVs total
- 14 patients total
- outer CV is leave-one-patient-out grouped CV
- each outer fold trains on 13 patients and tests on 1 held-out patient


## Causal figure beautification

The "better looking" causal graph rendering is built into the RCausalMGM R scripts, not a separate post-processing utility.

Relevant functions:
- `pipeline_assets/rCausalMGM_Rscript_NicheComposition.R`
- `pipeline_assets/rCausalMGM_Rscript_NeighborhoodInteractions.R`
- helper: `write_clean_dot_from_sif(...)`

What those scripts do:
- convert `.sif` causal edge exports into a cleaner Graphviz `.dot`
- render polished PNG graphs with `dot`
- set left-to-right layout, larger fonts, cleaner arrows, and disease-state-specific labels

Main outputs produced by that path:
- `rcausal_graphviz_niche_disease_state.png`
- `rcausal_graphviz_neighborhood_disease_state.png`

Best skin canonical sources for presentation:
- `skin_visium_manuscript_package/figures/rcausal_mgm__NeighborhoodInteractions__FOV_Neighborhood_Enrichment_withDisease_correlation_heatmap.png`
- `skin_visium_manuscript_package/figures/rcausal_mgm__NeighborhoodInteractions__rcausal_graphviz_neighborhood_disease_state.png`
- `skin_visium_manuscript_package/figures/rcausal_mgm__NicheCompositions__rcausal_graphviz_niche_disease_state.png`

Note:
- `cross_organ_comparison/CANONICAL_SOURCES.md` marks `skin_visium_manuscript_package/` as the canonical analytical source set for skin.
- `skin_visium_ibd_paper_order_package_v2/` is presentation-oriented, but the curated manuscript package already contains the polished causal figures we want.

## Kidney findings

### Kidney tune-once
Path:
- `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/kidney_cosmx_ssc_poisson75/outputs/MLP_44Features_tune_once/`

Tuned params:
- hidden layers: `(8,)`
- activation: `relu`
- alpha: `1e-5`
- learning rate init: `1e-4`
- batch size: `8`

Tuning score:
- grouped full-data weighted F1: `1.000`

Interpretation:
- this was overly optimistic because kidney only has 6 patients

### Kidney fixed evaluation
Path:
- `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/kidney_cosmx_ssc_poisson75/outputs/MLP_44Features_eval_fixed/`

Results:
- mean accuracy: `0.333 (+/- 0.471)`
- mean precision: `0.333 (+/- 0.471)`
- mean recall: `0.333 (+/- 0.471)`
- mean F1: `0.333 (+/- 0.471)`
- overall confusion matrix:
  - healthy: `1 correct / 2 called systemic_sclerosis`
  - systemic_sclerosis: `1 correct / 2 called healthy`

Interpretation:
- kidney currently does not generalize under leakage-safe held-out evaluation
- runtime is no longer the blocker; model/data behavior is the blocker

## Skin findings

### Skin first fixed evaluation baseline
Path:
- `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed/`

Tuned params came from:
- `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_tune_once/`

Tuned params:
- hidden layers: `(40, 20, 10, 5)`
- activation: `tanh`
- alpha: `0.1`
- learning rate init: `0.01`
- batch size: `8`

Fixed-eval results:
- mean accuracy: `0.650 (+/- 0.346)`
- mean F1: `0.720 (+/- 0.328)`
- overall accuracy: `0.51`
- healthy precision/recall/F1: `0.29 / 0.23 / 0.26`
- systemic_sclerosis precision/recall/F1: `0.59 / 0.67 / 0.63`
- confusion matrix:
  - healthy: `14 correct / 47 called systemic_sclerosis`
  - systemic_sclerosis: `69 correct / 34 called healthy`

Interpretation:
- baseline skin model was still too biased toward predicting systemic_sclerosis

### Skin balanced tuning and fixed evaluation
Paths:
- tune: `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_tune_once_balanced/`
- eval: `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced/`

Changes:
- broader grid
- selection metric changed from `weighted_f1` to `macro_f1`

Tuned params:
- hidden layers: `(128,)`
- activation: `tanh`
- alpha: `0.3`
- learning rate init: `0.01`
- batch size: `16`

Tune score:
- macro F1 selection score: `0.432`

Fixed-eval results:
- mean accuracy: `0.693 (+/- 0.352)`
- mean F1: `0.751 (+/- 0.329)`
- overall accuracy: `0.58`
- healthy precision/recall/F1: `0.40 / 0.26 / 0.32`
- systemic_sclerosis precision/recall/F1: `0.64 / 0.77 / 0.70`
- confusion matrix:
  - healthy: `16 correct / 45 called systemic_sclerosis`
  - systemic_sclerosis: `79 correct / 24 called healthy`

Interpretation:
- better overall than the first skin baseline
- healthy class improved, but still weak

### Skin balanced + minority-oversampled tuning and fixed evaluation
Paths:
- tune: `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_tune_once_balanced_resampled/`
- eval: `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced_resampled/`

Changes:
- broader grid
- selection metric: `macro_f1`
- training-only resampling: `oversample_minority`

Tuned params:
- hidden layers: `(32, 16, 8)`
- activation: `tanh`
- alpha: `0.001`
- learning rate init: `0.01`
- batch size: `8`

Tune score:
- macro F1 selection score: `0.441`

Fixed-eval results:
- mean accuracy: `0.671 (+/- 0.311)`
- mean F1: `0.755 (+/- 0.265)`
- overall accuracy: `0.55`
- healthy precision/recall/F1: `0.40 / 0.41 / 0.40`
- systemic_sclerosis precision/recall/F1: `0.64 / 0.63 / 0.64`
- confusion matrix:
  - healthy: `25 correct / 36 called systemic_sclerosis`
  - systemic_sclerosis: `65 correct / 38 called healthy`

Interpretation:
- strongest healthy-class result so far
- more balanced than prior models
- tradeoff: systemic_sclerosis performance dropped relative to the previous balanced model

## Current best baselines to remember

### Best skin overall / SSc-oriented baseline
Use:
- `MLP_FOVFeatures_eval_fixed_balanced`

Why:
- best overall held-out performance so far
- mean F1 `0.751`
- better SSc performance than the resampled version

### Best skin healthy-recall baseline
Use:
- `MLP_FOVFeatures_eval_fixed_balanced_resampled`

Why:
- healthy recall improved to `0.41`
- healthy F1 improved to `0.40`
- confusion matrix improved from `16/45` healthy to `25/36` healthy

### Kidney baseline
Use:
- `MLP_44Features_eval_fixed`

Why:
- currently the honest leakage-safe kidney result
- mean F1 `0.333`

## Threshold analysis support
New output now available from evaluation runs:
- `fold_predictions.csv`

This stores per-item fold predictions and positive-class probabilities so threshold tuning can be done without retraining.

Threshold scan helper:
- `scripts/evaluate_mlp_thresholds.py`

Example usage on HPG for the latest skin resampled run:

```bash
module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4

cd /blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool

python scripts/evaluate_mlp_thresholds.py   --predictions /blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced_resampled/fold_predictions.csv   --positive-class systemic_sclerosis   --threshold-start 0.05   --threshold-stop 0.95   --threshold-step 0.05   --output /blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced_resampled/threshold_scan.csv
```

## Suggested next step
The most informative next step is threshold tuning / calibration on the skin evaluation probabilities, because the current differences between the best skin models look like a decision-boundary tradeoff rather than a total model failure.


## Threshold scan result for latest skin balanced-resampled model
Using `scripts/evaluate_mlp_thresholds.py` on:
- `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced_resampled/fold_predictions.csv`

Main finding:
- threshold tuning changes the healthy/SSc tradeoff substantially
- it is not a free improvement; higher thresholds improve healthy performance but reduce systemic_sclerosis recall

Selected thresholds from the scan:
- threshold `0.50` (default-ish operating point)
  - balanced accuracy: `0.520`
  - macro F1: `0.520`
  - healthy precision/recall/F1: `0.397 / 0.410 / 0.403`
  - SSc precision/recall/F1: `0.644 / 0.631 / 0.637`
  - confusion matrix:
    - healthy: `25 correct / 36 called systemic_sclerosis`
    - systemic_sclerosis: `65 correct / 38 called healthy`
- threshold `0.75`
  - balanced accuracy: `0.593`
  - macro F1: `0.579`
  - healthy precision/recall/F1: `0.458 / 0.623 / 0.528`
  - SSc precision/recall/F1: `0.716 / 0.563 / 0.630`
  - confusion matrix:
    - healthy: `38 correct / 23 called systemic_sclerosis`
    - systemic_sclerosis: `58 correct / 45 called healthy`
- threshold `0.95`
  - balanced accuracy: `0.610`
  - macro F1: `0.584`
  - healthy precision/recall/F1: `0.462 / 0.705 / 0.558`
  - SSc precision/recall/F1: `0.746 / 0.515 / 0.609`
  - confusion matrix:
    - healthy: `43 correct / 18 called systemic_sclerosis`
    - systemic_sclerosis: `53 correct / 50 called healthy`

Interpretation:
- raising the threshold improves the healthy class substantially
- but it causes many more systemic_sclerosis FOVs to be called healthy
- this confirms that the current skin problem is partly a decision-threshold tradeoff, not just a feature/model collapse

## Practical recommendation from current evidence
If the scientific priority is stronger overall SSc-oriented performance:
- use `MLP_FOVFeatures_eval_fixed_balanced`

If the scientific priority is healthier balance / better healthy recall:
- use `MLP_FOVFeatures_eval_fixed_balanced_resampled`
- optionally explore thresholds around `0.65` to `0.75` as compromise operating points

Do not use threshold `0.95` as the final operating point without explicit justification, because it improves healthy recall at too large a cost to systemic_sclerosis recall.

## Thesis chapter outline draft

### Chapter question
Can niche-derived spatial and molecular features support leakage-safe prediction of disease state in systemic sclerosis spatial transcriptomics data across organs?

### Data chapter sections
1. Motivation
2. Datasets
3. Feature engineering pipeline
4. Evaluation protocol
5. Kidney results
6. Skin results
7. Class-imbalance mitigation experiments
8. Threshold-tradeoff analysis
9. Interpretation and limitations
10. Conclusion

### Key points to write
- The evaluation protocol was changed to patient-grouped leakage-safe held-out evaluation.
- This removed optimistic results that appeared under less rigorous setups.
- Kidney did not generalize under the leakage-safe protocol.
- Skin retained moderate predictive signal and was the stronger organ.
- Macro-F1 tuning and minority oversampling improved the healthy class, but introduced a tradeoff with SSc recall.
- The final conclusion is not that the project failed, but that organ-specific predictability differed and that evaluation rigor materially changed the apparent performance.

## Suggested chapter-level conclusion
A defensible chapter conclusion would be:
- skin contains moderate predictive signal under leakage-safe grouped evaluation
- kidney does not currently support robust prediction with the present feature/model setup
- class imbalance and operating-threshold choices substantially affect the biological interpretation of classifier performance
- therefore, negative or weak predictive performance in one organ is itself an informative scientific finding rather than merely a failed experiment


## 5:5:5 PPT outline
Use 5 slides, no more than 5 bullets per slide, and keep each bullet to about 5 words when possible.

### Slide 1 - Question and setup
- Can niches classify disease?
- Skin and kidney tested
- Leakage-safe grouped evaluation
- Leave-one-patient-out cross-validation
- MLP on niche features

Speaker note:
- The core question is whether Distill-derived niche features can separate healthy vs systemic sclerosis under strict patient-held-out testing.

### Slide 2 - Why the first 100% changed
- Initial result looked perfect
- Tuning was too optimistic
- Few patients in kidney
- Full-data tuning inflated score
- Held-out evaluation corrected this

Speaker note:
- The early 100% was not the final scientific answer. The stricter workflow separates parameter tuning from held-out evaluation and keeps all samples from a patient in the same fold.

### Slide 3 - Kidney final result
- Kidney did not generalize
- Mean F1 was 0.333
- Accuracy also 0.333
- Confusion: 1/3 each class
- Kidney is weak-result arm

Speaker note:
- Kidney currently does not support a publishable classifier claim. It is still valid as a thesis result because it shows the limits of this feature/model setup.

### Slide 4 - Skin best results
- Skin showed real signal
- Best mean F1 0.755
- Best balanced accuracy 0.671
- Healthy F1 reached 0.40
- Tradeoff remains between classes

Speaker note:
- The best balanced-resampled skin model improved healthy detection, while the macro-F1 balanced model kept stronger systemic-sclerosis performance.

### Slide 5 - Confusion and next steps
- Best balanced confusion: 16/45
- Resampled confusion: 25/36
- Threshold shifts class tradeoff
- Explainability is next focus
- Poster feasible, paper uncertain

Speaker note:
- For the balanced run, healthy was 16 correct and 45 miscalled as SSc; SSc was 79 correct and 24 miscalled healthy. For the resampled run, healthy improved to 25 correct and 36 miscalled, but SSc dropped to 65 correct and 38 miscalled.

## Suggested figures for the PPT
- one pipeline / evaluation schematic
- one kidney confusion matrix
- one skin confusion matrix for `MLP_FOVFeatures_eval_fixed_balanced`
- one skin confusion matrix for `MLP_FOVFeatures_eval_fixed_balanced_resampled`
- one threshold-tradeoff plot from `threshold_scan.csv`

## Exact paths for the current slide-ready outputs

### Kidney
- tune-once params:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/kidney_cosmx_ssc_poisson75/outputs/MLP_44Features_tune_once/fixed_params.json`
- fixed evaluation report:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/kidney_cosmx_ssc_poisson75/outputs/MLP_44Features_eval_fixed/mlp_results.txt`

### Skin balanced
- tune-once params:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_tune_once_balanced/fixed_params.json`
- fixed evaluation report:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced/mlp_results.txt`

### Skin balanced + resampled
- tune-once params:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_tune_once_balanced_resampled/fixed_params.json`
- fixed evaluation report:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced_resampled/mlp_results.txt`
- fold predictions for threshold tuning:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_eval_fixed_balanced_resampled/fold_predictions.csv`

## Short summary to tell Myles
Skin has moderate leakage-safe signal and kidney does not. The best overall skin run is the macro-F1 balanced model with mean F1 `0.751`, while the best healthy-class skin run is the balanced-plus-resampled model with healthy F1 `0.40` and confusion matrix `25 healthy correct / 36 healthy called SSc / 65 SSc correct / 38 SSc called healthy`. Kidney remains at mean F1 `0.333`, so it should be presented as a weak-result arm rather than a strong classifier result.


## Explainability runs to generate next
Use these presets to generate SHAP outputs for the slide deck.

### HPG setup
```bash
module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4
cd /blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool
```

### Kidney explain-only
```bash
python run_pipeline.py --config presets/kidney_cosmx_ssc_poisson75_hpg_mlp_explain_cpu.json --validate
python run_pipeline.py --config presets/kidney_cosmx_ssc_poisson75_hpg_mlp_explain_cpu.json
sbatch runs/kidney_cosmx_ssc_poisson75_mlp_explain_cpu/submit.sh
```

### Skin explain-only, balanced
```bash
python run_pipeline.py --config presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_explain_cpu_balanced.json --validate
python run_pipeline.py --config presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_explain_cpu_balanced.json
sbatch runs/skin_visium_ssc_1mmfov_poisson75_mlp_explain_cpu_balanced/submit.sh
```

### Skin explain-only, balanced plus resampled
```bash
python run_pipeline.py --config presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_explain_cpu_balanced_resampled.json --validate
python run_pipeline.py --config presets/skin_visium_ssc_1mmfov_poisson75_hpg_mlp_explain_cpu_balanced_resampled.json
sbatch runs/skin_visium_ssc_1mmfov_poisson75_mlp_explain_cpu_balanced_resampled/submit.sh
```

Expected explainability outputs in each MLP output folder:
- `shap_importance.csv`
- `shap_importance_top20.png`
- `shap_values_positive_class.csv`


## Explainability results added 2026-07-04
All three explain-only jobs completed successfully on HPG.

Job IDs:
- kidney explain: `36364963`
- skin balanced explain: `36364964`
- skin balanced-resampled explain: `36364965`

### Explainability artifact folders
- kidney:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/kidney_cosmx_ssc_poisson75/outputs/MLP_44Features_explain/`
- skin balanced:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_explain_balanced/`
- skin balanced-resampled:
  - `/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_explain_balanced_resampled/`

Each folder contains:
- `mlp_results.txt`
- `shap_importance.csv`
- `shap_importance_top20.png`
- `shap_values_positive_class.csv`

### Kidney SHAP summary
Top kidney features:
- `niche_0_gene_NPR2`
- `niche_1_gene_SystemControl64`
- `enrichment_2-11`
- `niche_4_gene_LGALS1`
- `nmf_prop_9`
- `niche_10_gene_NCAM1`
- `niche_1_gene_CRIP1`

Interpretation:
- kidney SHAP is exploratory only
- because kidney fixed held-out performance remained weak: mean F1 `0.333`
- do not present this as a strong biological classifier signal

### Skin balanced SHAP summary
Top skin balanced features:
- `niche_3_gene_DUSP1`
- `niche_7_gene_FOS`
- `niche_3_gene_JUN`
- `nmf_prop_4`
- `nmf_prop_3`
- `niche_8_gene_GJA8`
- `niche_7_gene_IGFBP3`
- `niche_2_gene_F10`

Interpretation:
- this is the cleaner explainability result for the stronger SSc-oriented skin model
- the model relies on both niche-gene features and niche proportion features

### Skin balanced-resampled SHAP summary
Top skin balanced-resampled features:
- `nmf_prop_3`
- `niche_7_gene_FOS`
- `niche_3_gene_DUSP1`
- `nmf_prop_4`
- `nmf_prop_2`
- `niche_7_gene_IGFBP3`
- `nmf_prop_0`
- `nmf_prop_1`

Interpretation:
- after minority oversampling, the model leans more heavily on NMF proportion features
- this matches the class-tradeoff change seen in evaluation
- healthy detection improved, but SSc performance weakened somewhat

### Explainability conclusion for the deck
Use this framing:
- kidney explainability is included for completeness, but kidney is still a weak-result arm
- skin balanced is the main explainability result for stronger disease-oriented prediction
- skin balanced-resampled is the main explainability result for improved healthy detection
- together they show that model-selection and class-balancing change both performance and feature reliance


## Mapping to the paper structure
Myles asked for outputs analogous to the IBD paper figures/tables, specifically Figure 4, Table 2, Figure 6, Table 6, Table 7, and Figure 8 from `paper.pdf`.

### What those items represent in the paper
- `Figure 4`: niche neighborhood enrichment heatmaps across condition groups
- `Table 2`: biologically selected niche-gene features
- `Figure 6`: causal graphs for composition, enrichment, and niche-gene features
- `Table 6`: statistical tests for niche composition differences across groups
- `Table 7`: statistical tests for niche interaction / enrichment differences across groups
- `Figure 8`: top feature-importance / explainability summary

### Distill equivalents for the current skin/kidney thesis chapter
- `Figure 4 equivalent`
  - use the niche-enrichment outputs already produced upstream from the Poisson75 runs
  - for skin and kidney, these belong to the post-NMF / RCAusal-statistics side rather than the MLP reruns
- `Table 2 equivalent`
  - use the selected niche-gene features plus the SHAP top features from the final skin models
  - this should be a compact biology-facing table listing the highest-importance genes/features per model
- `Figure 6 equivalent`
  - use the RCAusal / causal discovery outputs already generated from the niche features
  - if available, show separate causal views for composition, enrichment, and niche-gene relationships
- `Table 6 equivalent`
  - use pairwise statistical tests for niche composition differences between healthy and systemic sclerosis
- `Table 7 equivalent`
  - use pairwise statistical tests for enrichment / neighborhood interaction differences between healthy and systemic sclerosis
- `Figure 8 equivalent`
  - use `shap_importance_top20.png` from the explain-only runs
  - main figure: skin balanced and skin balanced-resampled
  - kidney can be included as supplementary / comparison only

### Current best candidates for presentation
- Main classifier figure set:
  - skin balanced fixed evaluation
  - skin balanced-resampled fixed evaluation
  - kidney fixed evaluation as a weak-result comparison
- Main biology / explainability figure set:
  - `runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_explain_balanced/shap_importance_top20.png`
  - `runs/skin_visium_ssc_1mmfov_poisson75_split/outputs/MLP_FOVFeatures_explain_balanced_resampled/shap_importance_top20.png`
- Kidney explainability figure:
  - `runs/kidney_cosmx_ssc_poisson75/outputs/MLP_44Features_explain/shap_importance_top20.png`
  - use cautiously because kidney performance was weak

### Important narrative point
For this project, the classification report shows whether the pipeline works under leakage-safe evaluation, but the biological story should be built mostly from:
- SHAP / feature-importance outputs
- niche-gene features
- causal outputs
- statistical tests on niche composition and enrichment

### Organ order note
The first HPG submission sequence shown in the logs launched both kidney and skin on the same day, but kidney was listed first in the command block. For presentation purposes, it is safer to say that both organs were part of the initial analysis phase, rather than claiming a strong chronological distinction.
