# NOVAE res1.0 historical-164 FOV feature contract

## Frozen decision

The downstream contract audit was validated by job **43008275**, commit
**b8218ba**, `COMPLETED 0:0 1:47`, with output
`/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_downstream_contract_audit/validated_rerun/audit`.
Both eligible contracts were retained for reference: historical canonical164
(61 healthy/103 SSc, 14 patients, 14 singleton structural-zero enrichment
FOVs) and fullsweep225 (89 healthy/136 SSc, 14 patients, 18 singleton
structural-zero enrichment FOVs), with canonical intersection 134. The
predeclared policy selects **historical_164**, never performance.

The standalone CPU adapter is `scripts/build_novae_fov_features.py`; its
launcher is `scripts/submit_novae_fov_features.sh`. It uses the authoritative
post-NMF `cosmx_with_nmf.h5ad` as the expression/metadata context and the
calibrated NOVAE H5AD only for `novae_domains_res1.0` and
`neighborhood_valid`. Inputs are read-only and are never rewritten. The
embedded `novae_pilot_provenance` must identify the exploratory, `reference=all`,
zero-shot, non-confirmatory calibrated CPU run with dataset
`skin_visium_ssc_paired_cpu_calibrated`, `visium_explicit_scale`, primary
resolution 1.0, and deterministic seed 42.

## Predeclared rules

- The exact canonical row/index order comes from frozen
  `MLP_FOVFeatures_inputs/combined_features_filtered.parquet`; targets and
  patient groups must match its index, order, and values. No retile, outcome
  filter, or resolution choice is allowed.
- Cells are joined by unique `unique_cell_id`, not by positional order. The
  calibrated NOVAE original-section H5AD need not and does not supply a
  pseudo-FOV key: FOV mapping comes solely from the authoritative base H5AD.
  Only shared patient, disease, and sample metadata are compared. Existing
  `NMF_factor` and `dominant_nmf_factor` columns are immutable.
- The observed calibrated res1.0 domain vocabulary is exactly `L0` through
  `L8`, naturally ordered. Features are named `novae_prop_<domain>` and never
  use NMF aliases. `combined_features_filtered.parquet` contains only these
  nine NOVAE composition columns, with the frozen canonical index; it contains
  no inherited NMF or globally selected columns.
- Each authoritative/post-NMF FOV must map to exactly one patient and disease
  label; mixed FOV metadata fails before aggregation. Invalid neighborhoods are
  excluded from all domain aggregates. Validity values must be recognized
  boolean/status values, not silently coerced. Valid rows must be labeled and
  invalid rows unlabeled. Every canonical FOV is retained;
  zero assigned rows receive all-zero composition/features and a QC flag.
  Coverage and total/valid/assigned/unassigned/invalid counts are reported.
  Proportions sum to one when assigned > 0 and zero otherwise.
- Enrichment reproduces the legacy notebook: per-FOV BallTree, coordinates
  `CenterX_global_px`/`CenterY_global_px`, `Area`, radius
  `2 * (2 * sqrt(Area/pi))`, self excluded, symmetric interactions, and
  `log2((interaction+1)/(expected+1))`. No cross-patient/FOV neighbors or graph
  edges are used. No-neighbor and singleton values are formula-derived zero.
- Niche-gene means use notebook `_choose_expression_matrix` semantics (`raw.X`
  first, otherwise `X`) and sparse-safe per-FOV/domain means. The full candidate
  table is emitted; absent domain/FOV combinations are zero and disclosed in
  the manifest. No global disease-label MI/FDR filtering is performed. The
  niche table is emitted as Parquet only (about 26.7 million values); a
  redundant CSV mirror is intentionally not produced. Small composition,
  enrichment, and QC tables retain CSV mirrors.

Outputs are staged and atomically renamed into a new, non-overwritable output
folder. The manifest records input and output SHA256 hashes, formulas,
dimensions, vocabulary,
coverage, structural zeros, and the **exploratory/reference=all limitation**.

## Evaluation protocol

The primary evaluation is historical164. fullsweep225 is a predeclared
sensitivity analysis only; it cannot be selected using outcome or classifier
performance. The next evaluation uses the same nested patient LOGO protocol,
sets `NICHERUNNER_MLP_UNIT=fov` and
`NICHERUNNER_COMPOSITION_PREFIX=novae_prop_`, passes the complete enrichment and
niche candidate tables, and performs mutual-information selection inside each
outer training fold. No global ranked feature file is used.
