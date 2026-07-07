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

Each tile size is submitted as a three-job dependency chain:

1. GPU `cell2loc`
2. GPU `nmf`
3. CPU `post_nmf + rcausal_mgm + mlp + report`

GPU jobs use regular QoS. CPU follow-up jobs use burst QoS.

## Presets

For each tile size, the sweep uses:

- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_cell2loc_gpu.json`
- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_nmf_gpu.json`
- `presets/skin_visium_ssc_<tile>umfov_poisson75_hpg_downstream_cpu.json`

where `<tile>` is `1000`, `750`, or `500`.

## One-shot submission

Run:

```bash
bash scripts/submit_skin_pseudofov_fullsweep.sh
```

That script:

- analyzes pseudo-FOV sizes
- rebuilds the skin spatial `.h5ad` for each tile size
- validates the presets
- materializes the run directories
- queues all dependency chains

## Output layout

Each tile size writes into its own run family:

- `runs/skin_visium_ssc_1000umfov_poisson75_fullsweep/`
- `runs/skin_visium_ssc_750umfov_poisson75_fullsweep/`
- `runs/skin_visium_ssc_500umfov_poisson75_fullsweep/`

The MLP output subdirectory is:

- `MLP_FOVFeatures_macrof1_resampled_defaultgrid`

## Why this is the current research choice

This setup matches the current research direction:

- keep leakage-safe grouped CV fixed
- do not expand the hyperparameter search further
- test the "more FOVs" hypothesis directly
- preserve interpretability and report generation in the same run family
