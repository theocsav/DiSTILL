# Distill Pipeline Audit

This audit defines the canonical execution path for Distill/NicheRunner and separates it from legacy one-off scripts that still exist in the repository for historical or exploratory work.

## Canonical path

The supported application path is:

1. `apps/web` or `apps/api`
2. `apps/api/app/runner.py`
3. `run_pipeline.py`
4. generated run directory under `runs/<run_name>/`
5. stage scripts and notebooks referenced by the selected preset

This is already how the app works today.

## What is already standardized

- `apps/api/app/runner.py` invokes `run_pipeline.py` to prepare runs.
- `apps/api/app/registry.py` loads presets directly from `presets/*.json`.
- `apps/web` talks to the API and does not bypass the runner.
- Current HPG execution is preset-driven, not hardcoded in the app.

In practice, this means the Distill app already uses the preset plus runner contract as its main control plane.

## Current recommended execution model

For large HPG runs, the recommended flow is:

1. Prepare a preset in `presets/`.
2. Generate the run via `run_pipeline.py`.
3. Submit the generated `submit.sh`.
4. For large Poisson / cell2location runs, prefer split execution:
   - GPU stage: `cell2loc`
   - CPU follow-up: `nmf`, `post_nmf`, `rcausal_mgm`, `mlp`, `report`

This split is now supported by the runner and dedicated split presets.

## New canonical stage model

The canonical stage vocabulary is now:

- `cell2loc_nmf`
- `cell2loc`
- `nmf`
- `post_nmf`
- `rcausal_mgm`
- `mlp`
- `report`

Use `cell2loc_nmf` for the original monolithic flow.
Use `cell2loc` plus `nmf` follow-up when you want GPU `cell2location` and CPU downstream stages.

## Presets that follow the split model

Skin:

- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_cell2loc_gpu.json`
- `presets/skin_visium_ssc_1mmfov_poisson75_hpg_followup_cpu.json`

Kidney:

- `presets/kidney_cosmx_ssc_poisson75_hpg_cell2loc_gpu.json`
- `presets/kidney_cosmx_ssc_poisson75_hpg_followup_cpu.json`

These are the current reference presets for HPG split execution.

## Discontinued: MLP scripts with evaluation leakage

The following report a hyperparameter-selection maximum as a held-out estimate and
must not be used for reported results. They are retained, marked in place, and still
executable for provenance:

- `pipeline_assets/IBD_MLP_44Features.py`
- `pipeline_assets/scripts/IBD_MLP_44Features.py`
- `pipeline_assets/scripts/IBD_MLP_51Features.py`
- `pipeline_assets/scripts/IBD_MLP_FewerParams.py`

Use `pipeline_assets/IBD_MLP_LeakageSafe.py` with `mlp_mode=nested_cv`.
Rationale, measured bias, and the regeneration procedure:
[MLP_EVALUATION_AND_LEAKAGE.md](MLP_EVALUATION_AND_LEAKAGE.md).

## Legacy entrypoints that should not be treated as the app contract

The following files are still useful for history, debugging, or reference, but they should not be treated as the supported Distill entrypoint:

- `pipeline_assets/IBD_3000epochs_500samples_NMF-k4.py`
- `pipeline_assets/IBD_MLP_44Features.py`
- `pipeline_assets/IBD_RCausalMGM_Preparation.py`
- `pipeline_assets/scripts/IBD_3000epochs_500samples_NMF-k*.py`
- `pipeline_assets/scripts/IBD_3000epochs_500samples_NMF-k*.sh`
- `pipeline_assets/scripts/IBD_MLP*.py`
- `pipeline_assets/scripts/IBD_MLP*.sh`
- `pipeline_assets/scripts/IBD_3000epochs_systematicNMFapproach_250samp.py`
- `pipeline_assets/scripts/IBD_3000epochs_systematicNMFapproach_250samp.sh`

These scripts predate the preset-driven runner and may still be scientifically useful, but they are not the canonical way the app should submit or reproduce runs.

## Utility scripts that remain valid

The following remain valid as supporting utilities rather than pipeline entrypoints:

- dataset builders in `scripts/build_*`
- manifest generation in `scripts/generate_dataset_manifest.py`
- export helpers such as `scripts/export_top_niche_genes.py`
- figure/table helpers such as `scripts/plot_enrichment_lower_triangle.py`
- cross-organ comparison helpers such as `scripts/compare_kidney_skin_organs.py`

These scripts are compatible with the app model because they operate on artifacts, not on the orchestration contract.

## Recommended cleanup policy

Short term:

- Keep legacy scripts for reference.
- Route all new HPG runs through `run_pipeline.py`.
- Add new run variants only as presets, not as ad hoc shell scripts.

Medium term:

- Mark legacy launch scripts as deprecated in comments or docs.
- Move historical one-off launchers into an `archive/` or `legacy/` location if they are no longer actively used.
- Keep notebooks for interpretation and validation, but avoid using them as the primary orchestration surface.

## Practical rule

If someone asks, "How should Distill run this analysis?", the answer should be:

- define or update a preset
- prepare with `run_pipeline.py`
- submit the generated run
- inspect outputs from the run directory

That is the canonical path this audit formalizes.
