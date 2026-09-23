# NOVAE spatial/biological validation

This is a read-only, exploratory comparison of frozen historical NMF domains and
calibrated NOVAE domains on skin Visium. It is not a disease classifier and does
not use disease labels for selection or scoring.

## Frozen FAIR contract

Both arms use **K=9**, the exact `unique_cell_id` join, calibrated NOVAE H5AD
under `paired_cpu_diagnostic/skin_visium_ssc_paired_cpu_calibrated`, and NMF
labels/factors from historical `post_nmf_obs.csv`. The NOVAE contract requires
`reference=all`, the calibrated input/checkpoint hashes, `domain_key=
novae_domains_res1.0`, and valid labels `L0` through `L8`. The analysis uses
only NOVAE-valid rows as one shared complete case (13,372 expected valid and 45
invalid); invalid rows are disclosed and never imputed. Expression always
comes from the preserved integer raw-count layer `adata.layers['counts']`, never
post-inference/transformed `adata.X`; shape and finite/nonnegative/integer
contracts are enforced. Coordinates, raw counts, and the induced undirected
`spatial_connectivities` graph are identical between
arms. Non-finite values, duplicate/missing IDs, asymmetric or cross-slide edges,
wrong K, metadata disagreement, and provenance failures stop the run. The
canonical input file hashes are H5AD
`40b60eba32c0716637eae98a28c852c11c53dfe4168570ad3230cf0d43219ffd` and
`post_nmf_obs.csv`
`a79a4e5949752110593f45eccd1ff34786b7d39cea3ce144b635444638bb354b`.

Run only through SLURM (2 CPU, 96 GB, GPU disabled). The launcher defaults to
`/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/ibd_cosmx_k4`, the
production-validated CPU environment with parquet support:

```bash
scripts/submit_novae_spatial_biological_validation.sh --render-only
scripts/submit_novae_spatial_biological_validation.sh
```

Environment overrides are `NOVAE_SPATIAL_H5AD`,
`NOVAE_SPATIAL_POST_NMF_OBS`, `NOVAE_SPATIAL_OUTPUT_DIR`, and the corresponding
`NOVAE_SPATIAL_*` run/job/log/environment settings. The launcher refuses path
collisions and inherited unsafe scheduler/path values.

## Measurements

For each slide and arm, the report computes within-domain edge fraction,
boundary rate, categorical assortativity, PAS (a spot disagrees with the unique
neighbor majority; ties count as disagreement; zero-degree spots are excluded
and reported), and per-domain induced-component fragmentation. One thousand
seed-42 within-slide permutations preserve domain counts and produce null mean,
sd, z, and normalized excess homophily. Slide summaries are both unweighted and
edge-weighted; there are no naive FOV p-values.

Expression uses the same raw count matrix, deterministic library normalization
and `log1p`, label-free top-variance genes (up to 2,000), and fixed-seed PCA.
Per-slide silhouettes are reported where defined. Patient-LOGO domain signatures
compare held-out domain-vs-patient-background **mean log-normalized expression
 differences** with training signatures using Spearman correlation (`heldout
>=5`, `training >=20`), including best-other-domain correlation and margin.
HVG selection and the PCA embedding are computed once and scored with both label
sets. These cohort-derived labels make all such results descriptive/exploratory.

Coverage/prevalence, patient-domain NMI and Cramer's V are reported as
confounding diagnostics (lower is not automatically biologically better).
Cell-type coherence is explicitly blocked because NMF derives from
cell2location/dominant cell type; pathway/histology is blocked because those
annotations are unavailable; symmetric seed stability is blocked because no
comparable NMF reruns exist. Existing NOVAE calibration ARI is prior sensitivity
only and cannot select an arm.

## Outputs

The atomic output directory contains the shared observation contract,
`spatial_metrics_per_slide.parquet`, `fragmentation_per_slide_domain.parquet`,
`permutation_nulls.parquet`, `expression_silhouette.parquet`,
`patient_logo_signatures.parquet`, `unweighted_per_slide_summary.parquet`,
prevalence/coverage tables, `top_marker_signature_genes.csv` (top 20 positive
per-arm/domain label-only signatures), `selected_hvgs.csv`,
`method_summary.json`, `blocked_status.json`, and a hash manifest covering code,
inputs, and outputs. Staging is removed on any failure and an existing output is
never overwritten.
