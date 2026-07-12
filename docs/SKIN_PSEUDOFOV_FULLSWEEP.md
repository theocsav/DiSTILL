# Skin Pseudo-FOV Full Sweep

This experiment is the current best-practice follow-up for the skin Visium HC/SSc arm.

## Goal

Test whether increasing the number of pseudo-FOVs improves leakage-safe patient-grouped classification performance while keeping the evaluation protocol fixed.

## Sweep design

- Tile sizes: `1000um`, `750um`, `500um`
- Organ/platform: skin Visium HC/SSc
- NMF selection: Poisson/KL 75% cumulative improvement
- Evaluation: grouped leakage-safe FOV classification
- Grouping rule: all FOVs from the same patient stay in the same fold
- MLP selection metric: `macro_f1`
- MLP resampling: `oversample_minority`
- MLP grid profile: `default`

The smaller MLP grid is intentional. It reduces runtime while preserving the class-balancing changes that improved the skin results.

## Stage split

Each tile size is now submitted as a six-job dependency chain:

1. GPU `cell2loc`
2. GPU `nmf`
3. CPU `post_nmf`
4. CPU `rcausal_mgm`
5. CPU `mlp_tune_once`
6. CPU `mlp_evaluate_fixed + report`

`rcausal_mgm` and `mlp_tune_once` both depend on `post_nmf`.
The final evaluation job depends on both `rcausal_mgm` and `mlp_tune_once`.

GPU jobs use regular QoS. CPU follow-up jobs use burst QoS.

## Presets

For each tile size, the sweep uses:

- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_cell2loc_gpu.json`
- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_nmf_gpu.json`
- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_post_nmf_cpu.json`
- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_rcausal_cpu.json`
- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_mlp_tune_once_cpu.json`
- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_mlp_eval_fixed_cpu.json`

where `<tile>` is `1000`, `750`, or `500`.

## One-shot submission

Run:

```bash
bash scripts/submit_skin_pseudofov_fullsweep.sh
```

That script:

- analyzes pseudo-FOV sizes
- retiles the existing processed skin spatial `.h5ad` for each tile size
- validates the presets
- materializes the run directories
- queues all dependency chains

Note:

- the original draft assumed raw Visium ZIP inputs under `/blue/.../data/skin_dataset`
- on HPG, the working setup uses the existing processed source file:
  `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_1mmfov_spatial.h5ad`
- retiled datasets are generated with `scripts/retile_skin_visium_spatial_h5ad.py`

## Output layout

Each tile size writes into its own run family:

- `runs/skin_visium_ssc_1000umfov_poisson75_fullsweep/`
- `runs/skin_visium_ssc_750umfov_poisson75_fullsweep/`
- `runs/skin_visium_ssc_500umfov_poisson75_fullsweep/`

The MLP output subdirectory is:

- `MLP_FOVFeatures_macrof1_resampled_defaultgrid`

## HPG launch record

Launch date:

- July 7, 2026

Queued job chains:

- `1000um`: `cell2loc=36538113`, `nmf=36538114`, `downstream=36538115`
- `750um`: `cell2loc=36538272`, `nmf=36538273`, `downstream=36538274`
- `500um`: `cell2loc=36538290`, `nmf=36538291`, `downstream=36538292`

Retiled artifacts written to:

- `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_1000umfov_spatial.h5ad`
- `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_750umfov_spatial.h5ad`
- `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_500umfov_spatial.h5ad`

Pseudo-FOV analysis tables written to:

- `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/pseudo_fov_sweep_analysis/skin_visium_pseudo_fov_tile_counts.csv`
- `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/pseudo_fov_sweep_analysis/skin_visium_pseudo_fov_summary_by_sample.csv`
- `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/pseudo_fov_sweep_analysis/skin_visium_pseudo_fov_summary_by_cohort.csv`

## Initial sizing findings

Healthy cohort summary:

- `1000um`: `61` total pseudo-FOVs, `45` valid pseudo-FOVs, `fraction_ge_min=0.737705`
- `750um`: `92` total pseudo-FOVs, `70` valid pseudo-FOVs, `fraction_ge_min=0.760870`
- `500um`: `170` total pseudo-FOVs, `124` valid pseudo-FOVs, `fraction_ge_min=0.729412`

Systemic sclerosis cohort summary:

- `1000um`: `103` total pseudo-FOVs, `68` valid pseudo-FOVs, `fraction_ge_min=0.660194`
- `750um`: `153` total pseudo-FOVs, `105` valid pseudo-FOVs, `fraction_ge_min=0.686275`
- `500um`: `290` total pseudo-FOVs, `180` valid pseudo-FOVs, `fraction_ge_min=0.620690`

Interpretation:

- `750um` is the strongest compromise in the initial sizing pass
- it increases pseudo-FOV count substantially relative to `1000um`
- it preserves a better valid-pseudo-FOV fraction in systemic sclerosis than `500um`
- `500um` gives the largest sample count but pushes several SSc samples into sparse pseudo-FOV territory
- `1000um` is the most conservative option and remains useful as a denser comparison arm

Per-sample caution:

- `SSc5380` is the main sparse outlier at all three tile sizes
- valid pseudo-FOV fraction for `SSc5380` was `0.400000` at `1000um`, `0.352941` at `750um`, and `0.380000` at `500um`
- this sample may disproportionately affect downstream generalization and should be watched in the evaluation results

## Why this is the current research choice

This setup matches the current research direction:

- keep leakage-safe grouped CV fixed
- do not expand the hyperparameter search further
- test the "more FOVs" hypothesis directly
- preserve interpretability and report generation in the same run family
## Timeout outcome

All three downstream full-sweep jobs eventually hit the SLURM walltime limit and did not finish end-to-end.

- downstream jobs used a uniform 96:00:00 time limit
- each downstream preset ran the full chain: post_nmf + rcausal_mgm + mlp + report
- the MLP portion was configured as leakage-safe grouped nested_cv, not a faster tune_once plus evaluate_fixed workflow

Observed intermediate MLP input sizes:

- 1000um: 225 FOV rows, 85 columns
- 750um: 352 FOV rows, 85 columns
- 500um: 664 FOV rows, 85 columns

Interpretation:

- the sweep succeeded at generating more pseudo-FOVs
- but the current end-to-end downstream design is too expensive to complete within 96h for all three tile sizes
- runtime pressure comes from the combination of grouped nested CV, resampling, and the full downstream bundle in one job

Research takeaway:

- the more-FOVs hypothesis is still testable
- but the execution strategy should change before rerunning
- the next iteration should separate feature generation from evaluation and use a cheaper evaluation protocol for the sweep itself

Recommended rerun strategy:

1. keep the pseudo-FOV generation and upstream cell2loc and nmf steps
2. materialize downstream features once per tile size
3. replace full nested_cv sweep runs with:
   - one tune_once run per tile size
   - one evaluate_fixed run per tile size
4. reserve SHAP and heavier explainability only for the best-performing tile size
