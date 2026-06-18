# Pipeline Assets

This directory contains a mix of:

- canonical stage scripts used by `run_pipeline.py`
- notebooks used by canonical downstream stages
- historical standalone launchers kept for reference

## Canonical app contract

The Distill/NicheRunner app should launch analyses through:

1. `presets/*.json`
2. `run_pipeline.py`
3. generated run directory under `runs/<run_name>/`

That is the supported orchestration path for reproducible runs.

## Canonical scripts in this directory

These top-level files are still part of the supported pipeline contract:

- `IBD_MLP_44Features.py`
- `IBD_MLP_LeakageSafe.py`
- `IBD_RCausalMGM_Preparation.py`
- `IBD_Post_NMF_Analysis.ipynb`

These are stage implementations or stage assets. They are not intended to be the
primary user-facing entrypoint for job submission; presets and `run_pipeline.py`
should call them.

## Canonical templates under `pipeline_assets/scripts`

The canonical script templates there are:

- `IBD_3000epochs_systematicNMFapproach.py`
- `IBD_Run_NMF_From_Cell2Loc.py`

Those are patched and launched by `run_pipeline.py`.

## Legacy launchers

Many other files under `pipeline_assets/scripts/` are historical fixed-K or
one-off SLURM launchers. They remain in the repository for reference, but they
should not define the app contract going forward.
