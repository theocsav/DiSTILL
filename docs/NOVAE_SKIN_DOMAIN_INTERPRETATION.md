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

## Final verified results

The authoritative exploratory run is job **43493489**, with output
`/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_domain_interpretation_20260927T050535Z_3097537/interpretation`;
independent verification was job **43518049**. Runs **43480400** and
**43492438** are superseded reporting iterations; their scientific tables are
unchanged and are not the authoritative presentation artifacts.

Global shared-spot agreement was **ARI = 0.0900106** and **NMI = 0.1658423**.
The one-to-one Hungarian assignment is used only for descriptive color/map
alignment. The separate `best_signature_matches.csv` is non-exclusive and
uses the full similarity matrix: NOVAE L4 best matched NMF5 (rho 0.93085);
L2, L5, and L6 best matched NMF7 (rho 0.64679, 0.63296, and 0.52291);
L1 best matched NMF3 (rho 0.63573); L0 and L3 best matched NMF3 (approximately
0.633); and L7 and L8 best matched NMF0 (rho 0.62796 and 0.71137). These are
alignment summaries, not labels, resolution selection, or evidence that one
arm is biologically superior.

### Descriptive domain hypotheses

| NOVAE domain | exploratory hypothesis (not a cell-type label) |
|---|---|
| L0 | mural-contractile |
| L1 | immune/adnexal mixed |
| L2 | fibroblast/ECM |
| L3 | fibrovascular/mural |
| L4 | differentiated epidermal; strongest |
| L5 | activated fibrotic ECM; internally coherent but weak stability |
| L6 | heterogeneous fibroblast/ECM |
| L7 | weak vascular/immune; unstable |
| L8 | contractile vascular/mural |

The marker/program evidence was concordant with these cautious hypotheses:
epidermal keratin differentiation programs (including KRT1/KRT10/FLG/LOR/IVL)
were strongest for L4; collagen/ECM, reticular/fascia/dermal fibroblast, and
fibroblast-activation programs supported L2/L5/L6. L5 additionally showed
CTHRC1, FN1, COL10A1, and COL11A1 markers and Hallmark EMT enrichment. Mural/
vascular programs (ACTA2/TAGLN/RGS5/PDGFRB/PECAM1/VWF/KDR) contributed to
L0/L3/L8. L1 showed a T/NK plus sweat/adnexal/mast/endothelial mixture,
including TRBC2, IL7R, CCL19, GABRP, and MMP7. Hallmark results are
over-representation of domain-marker genes against the eligible detected-gene
background, with BH correction within each arm/domain; they are not disease
tests or independent cell-type validation.

Patient-LOGO stability further separates these observations: L4 had same-domain
Spearman **0.928**, margin **+0.647**, and positive margin in **100%** of
eligible held-out patients; L7 had **0.123**, margin **−0.183**, and positive
margin in **1/14**; L5 had 11 eligible patients and margin **−0.0196**.

Disease prevalence remains descriptive and patient-level. NOVAE L4 had effect
**−0.06167**, p **=.02597**, q **=.16184**; L6 **−.05213**, p **=.03596**,
q **=.16184**; and L3 **+.04864**, p **=.09790**, q **=.29371**. For NMF,
domain 2 had **+.17405**, p **=.06294**, q **=.18656**, and domain 4 had
**−.20923**, p **=.07892**, q **=.18656**. No prevalence result survived
FDR < .05. Bootstrap intervals are descriptive; these analyses did not perform
disease discovery or use disease for marker/pathway selection.

Together with the prior spatial validation, NOVAE was more spatially coherent
and less fragmented, but showed lower molecular identity and no classifier
superiority. The defensible professor/grant conclusion is that NOVAE reorganizes
recognizable biology into smoother context domains: the strongest evidence is
along epidermal and mural axes, with exploratory fibroblast subdivisions. It is
not evidence for replacement superiority over NMF, automatic biological labels,
or confirmatory disease mechanisms.
