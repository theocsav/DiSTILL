# Exploratory NOVAE/NMF skin-domain interpretation

`scripts/run_novae_domain_interpretation.py` is a separate, read-only stage for
interpreting the frozen calibrated NOVAE H5AD beside the paired NMF factors.
It consumes the completed spatial-validation report and uses exactly its
NOVAE-valid shared complete-case rows (K=9); invalid rows are disclosed and
never imputed. The expression source is exclusively `layers['counts']`, and
input SHA-256 values are recorded before and after execution. Existing outputs
are never overwritten; publication is atomic.

The stage is exploratory and does **not** assign cell types or treat domains as
cell-type truth. Disease is used only after marker/program/pathway calculation,
for patient-level descriptive prevalence. Marker signatures are patient-aware
pseudobulk CPM contrasts (domain versus same-patient background), with the
predeclared >=5 in-domain and >=20 background thresholds, label-free >=1%
detection, >=3-patient aggregation, and deterministic top-100 positive gate.
Fixed programs are in `presets/novae_skin_marker_programs.csv`; top program candidates are predeclared as positive-score programs with >=50% gene coverage (top five per arm/domain); negative scores remain only in the full score table. Hallmark Human
2025.1 is the pinned public GMT and checksum in `presets/`. The resource
manifest records the source URL and attribution: `https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/h.all.v2025.1.Hs.symbols.gmt`.

Outputs include full and aggregated patient-aware markers, top markers, fixed
program scores and top coverage-gated candidates, exact scipy hypergeometric Hallmark ORA with within-domain BH,
NMF↔NOVAE Spearman similarity, deterministic Hungarian matching, and non-exclusive full-matrix best matches, contingency
and patient/global ARI/NMI/Jaccard agreement, complete zero-preserving prevalence summaries, exact 4-vs-10 patient prevalence enumeration, seeded 10,000-replicate stratified patient bootstrap intervals,
prior LOGO stability summaries, `best_signature_matches.csv`, maps with explicit invalid-row/legend disclosures, manifests, and machine-readable blocked
status/limitations. Matching is descriptive and is used only for map colors;
it never selects a domain or resolution. No FOV-level p-values or confirmatory
claims are produced.

## SLURM

Render safely without submitting (the real H5AD is not processed locally):

```bash
NOVAE_INTERPRET_H5AD=/blue/.../calibrated.h5ad \
NOVAE_INTERPRET_POST_NMF_OBS=/blue/.../post_nmf_obs.csv \
NOVAE_INTERPRET_VALIDATION_DIR=/blue/.../runs/novae_spatial_biological_validation_20260923T005445Z_3191045/validation \
scripts/submit_novae_domain_interpretation.sh --render-only
```

The generated job requests one node, 2 CPUs, 96 GB RAM, disables CUDA, and
runs with fixed seed 42. Paths, scheduler values, output conflicts, and
pre-existing reports are fail-closed.

## Resources and citations

The small fixed marker-program resource records source IDs and is curated from
Ganier et al. ([PMC10786309](https://pmc.ncbi.nlm.nih.gov/articles/PMC10786309/)), Chen et al. ([PMC11090198](https://pmc.ncbi.nlm.nih.gov/articles/PMC11090198/)), and Apostolidis et al.
([PMC6174292](https://pmc.ncbi.nlm.nih.gov/articles/PMC6174292/)). These references motivate gene programs; they do not make an
observed domain a biological label. Hallmark Human 2025.1 is redistributed from Broad Institute MSigDB under CC
BY 4.0. License terms are at
`https://gsea-msigdb.org/gsea/msigdb_license_terms.jsp`; copyright attribution
is Broad Institute, Massachusetts Institute of Technology, and The Regents of
the University of California, 2004-2025. The source URL and SHA-256 are
recorded alongside the GMT.
