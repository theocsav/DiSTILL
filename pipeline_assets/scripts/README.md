# Script Status

This folder mixes current pipeline templates with historical launchers.

## Canonical templates

These are the current templates that `run_pipeline.py` should own:

- `IBD_3000epochs_systematicNMFapproach.py`
- `IBD_Run_NMF_From_Cell2Loc.py`

## Deprecated standalone launchers

The following families are retained for reference, not as the recommended
Distill entrypoint:

- `IBD_3000epochs_500samples_NMF-k*.py`
- `IBD_3000epochs_500samples_NMF-k*.sh`
- `IBD_3000epochs_systematicNMFapproach.sh`
- `IBD_3000epochs_systematicNMFapproach_250samp.py`
- `IBD_3000epochs_systematicNMFapproach_250samp.sh`
- `IBD_MLP*.py`
- `IBD_MLP*.sh`

For new reproducible runs, prefer presets plus `run_pipeline.py`.
