# NOVAE skin Visium pilot run report (2026-09-06)

## Disposition

This report records the completed NOVAE HPG pilot and the research needed to
interpret it. It is an **exploratory Phase 0/1 result**, not confirmatory
held-out evidence. The run used `reference="all"`, so its prototypes use the
complete cohort. There was no DiSTILL integration. Do not use these domains to
claim fold-safe classification or generalisation.

The work was performed on branch `feature/novae-representations` with these
implementation commits:

- `79c9298` — initial hardened pilot;
- `8476f9b` — executable HPG launcher;
- `415eb09` — explicit invalid-neighborhood coverage handling; and
- `69bbb25` — H5AD-safe provenance.

The focused tests ultimately reported **32 passed** in both project
environments and in the exact compatibility environment with `anndata==0.11.4`
and `h5py==3.12.1`.

## Inputs and resolved environment

- Input H5AD: `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_spatial.h5ad`
- Sibling manifest: `/blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_sample_manifest.csv`
- Conda environment: `/blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312`
- NOVAE: `1.1.1`
- Model: `/blue/kejun.huang/vasco.hinostroza/models/novae-human-0`
- Requested model revision: `b8c0a5d7612bac6bc719ab57ed3cd16ad814728c`

The model is a local checkpoint. The requested revision is retained as
provenance, but **local checkpoint file hashes are authoritative** for this
run; a requested local-directory revision is not independent verification.
The resolved manifest/provenance contains the file hashes.

## Run history

The audit-only output at
`runs/novae_skin_pilot/audit-20260905_023330/` succeeded. It validated the
input IDs, slide boundaries, coordinates, and calibration without loading the
NOVAE model and did not constitute a scheduled HPG run.

| Job | Result | Finding |
|---|---|---|
| 41152531 | Failed, ExitCode 2 | Inference completed, but the adapter incorrectly rejected NOVAE's expected NA leaf/domain values for `neighborhood_valid=False`. No transactional output was published. |
| 41197867 | Failed, ExitCode 1 | Representations, domain assignment, and metric validation succeeded; AnnData serialization then failed on the list-of-dict `radius_pruning.per_slide` provenance. No output was published. |
| 41213616 | **COMPLETED, ExitCode 0, 44 sec** | H5AD-safe provenance fix passed and the transactional output was published. Run tag: UTC `2026-09-06 02:46:02`. |

The run tag is UTC. An `ls` listing may display September 5 because the login
shell's local timezone differs; that is not a second run date.

NOVAE 1.1.1 intentionally treats an unavailable `n_hops_view` neighborhood as
invalid: its latent row is zero-filled and its leaf/domain values are NA. A
valid neighborhood must receive a domain. The adapter was corrected to preserve
that contract rather than treating expected invalid-row NAs as an error. The
primary sources are the official v1.1.1 files:

- [`novae/model.py`](https://raw.githubusercontent.com/prism-oncology/novae/v1.1.1/novae/model.py)
- [`novae/data/dataset.py`](https://raw.githubusercontent.com/prism-oncology/novae/v1.1.1/novae/data/dataset.py)
- [`novae/utils/_utils.py`](https://raw.githubusercontent.com/prism-oncology/novae/v1.1.1/novae/utils/_utils.py)

## Successful output

The successful transactional run directory is:

`/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool/runs/novae_skin_pilot/h5ad-provenance-fix-20260906_024602/`

It contains the annotated H5AD, the saved zero-shot model, external
provenance/resolved manifest, coordinate/neighbor/graph QC, latent summaries,
domain proportions and adjacency, resolution summary, and science metrics.
The annotated H5AD is **842,785,755 bytes** and the model safetensors artifact
is **128,274,284 bytes**. The external `{run, artifacts}` provenance envelope
records final artifact SHA-256 values without a circular self-hash; the same
core run provenance is embedded in `adata.uns['novae_pilot_provenance']`.

### Assignment coverage

Assignments were identical at every requested resolution: **13,372 / 13,417**
assigned, **45** invalid/unassigned, and **99.66460460609674%** coverage. The
predeclared minimum was 0.70 overall and per slide, and every slide passed.
Per-slide unassigned counts are:

| Slide | Unassigned / total | Coverage |
|---|---:|---:|
| HC01 | 1 / 2172 | 99.95395948434622% |
| HC02 | 0 / 668 | 100% |
| HC03 | 3 / 1812 | 99.83443708609272% |
| HC05 | 0 / 827 | 100% |
| SSc-HL01 | 1 / 460 | 99.78260869565217% |
| SSc-HL05 | 0 / 508 | 100% |
| SSc-HL06 | 7 / 626 | 98.88178913738019% |
| SSc-HL11 | 0 / 701 | 100% |
| SSc-HL13 | 0 / 605 | 100% |
| SSc-HL25 | 0 / 751 | 100% |
| SSc-HL33 | 0 / 738 | 100% |
| SSc-HL35 | 0 / 878 | 100% |
| SSc4994 | 0 / 1752 | 100% |
| SSc5380 | 33 / 919 | 96.40914036996736% |

The 45 invalid rows are retained as invalid; no labels were imputed and no
literal `"nan"` domain was fabricated. SSc5380 was the worst slide, still
above the predeclared 0.70 threshold.

### Resolution sweep and metrics

| Resolution | Domains | Domain sizes (min / median / max) | FIDE | JSD |
|---|---:|---:|---:|---:|
| 0.5 | 5 | 1899 / 2497 / 4211 | 0.7096254585734768 | 0.06096970813074565 |
| **1.0 (primary)** | **9** | **484 / 1481 / 2453** | **0.578818397995689** | **0.1493878445089467** |
| 2.0 | 17 | 110 / 736 / 1733 | 0.45887614268185184 | 0.28268963161415606 |

FIDE and JSD are comparative diagnostics, not absolute pass/fail criteria.
The finer partitions predictably reduce continuity (FIDE) and increase
between-slide divergence (JSD); these values do not justify selecting a
resolution post hoc. Resolution 1.0 remains primary because it was declared
before the run.

### Graph and radius diagnostics

The graph had **38,107 undirected edges** both before and after the explicit
100 µm radius check: zero edges were removed. There were **32 zero-degree
observations** both before and after, and there were no cross-slide edges. The
zero-degree counts were HC01 1, HC03 3, SSc-HL06 5, and SSc5380 23; all other
slides had zero. NOVAE's 45 invalid rows exceed the 32 direct isolates, which
is consistent with multi-hop/component validity and disproves radius pruning
as their cause.

### Expression QC

Six rows (0.0447%) had zero expression counts. All were in SSc5380 and were
retained for QC:

- `SSc5380_GTCGAACCGTTCACTC-1`
- `SSc5380_ACATCTCAGTATTGCA-1`
- `SSc5380_ATGCTGCCGGCATACT-1`
- `SSc5380_GAATGTCAGAATTCTG-1`
- `SSc5380_AGTTGGAAGAGTATTG-1`
- `SSc5380_AATTACCAACTCCGTA-1`

The builder keeps filtered-matrix barcodes marked `in_tissue=1` and applies no
count threshold. SSc5380 is a known sparse outlier. This report does **not**
claim that the six zero-count rows are invalid-neighborhood rows: that would
require a cross-tabulation not performed here.

## Coordinate calibration research

Every slide's median positive graph-edge distance after the adapter's
estimated scale conversion was between **78.582 and 78.975 µm** (approximately
78.75 µm). This passes only the predeclared broad 50% guardrail **[50, 150]
µm** around the 100 µm expected Visium center spacing; it is not independent
validation of physical-micron calibration because the same estimated
`spot_diameter_fullres` heuristic generated the scale. The result is
**21.25% below the nominal** center spacing.

The relevant platform references report a standard Visium spot diameter of
55 µm and 100 µm center pitch: [10x Genomics Space Ranger
Glossary](https://www.10xgenomics.com/support/software/space-ranger/latest/getting-started/space-ranger-glossary).
Space Ranger's [`pxl_*_in_fullres` and
`spot_diameter_fullres` output semantics`](https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs)
identify the former as full-resolution spot centers and the latter as an
estimated value primarily intended for visualization; 10x recommends using
known microscope pixel dimensions when available.

The builder reads raw full-resolution positions without crop or resize. The
adapter converts each slide with `55 / spot_diameter_fullres`, then performs
NOVAE processing in materialized micrometres. The observed/nominal ratios are
78.75/55 = **1.4318** and 100/55 = **1.8182**, giving a diagnostic correction
ratio of **1.2698**. This is a calibration lead, not a post-hoc correction:
do not rescale or filter this completed run. Any alternate preprocessing must
be a prespecified sensitivity run after auditing raw array coordinates and
microscope calibration in a separate SLURM job.

NOVAE's warning for distances above 60 µm is generic and can also appear for a
correctly calibrated Visium input. See the official v1.1.1 validation code:
[`novae/utils/_validate.py`](https://raw.githubusercontent.com/prism-oncology/novae/v1.1.1/novae/utils/_validate.py).

## Scientific interpretation and limits

The zero removals establish that this run's stored Visium graph topology was
unchanged by the explicit 100 µm pruning step. They do not establish
domain/representation invariance under corrected coordinate preprocessing;
domain assignments are calibration-sensitive and provisional pending a
separately named, prespecified sensitivity run. Physical micrometre claims and
pseudo-FOV size claims likewise remain provisional until raw
array-coordinate/microscope calibration is audited. The completed output is
exploratory because `reference=all` uses complete-cohort prototypes; it is not
fold-safe confirmatory evidence. There was no DiSTILL integration. Preserve the
result as the declared run rather than changing it after seeing the QC; any
rescaled or differently filtered result must be a separately named,
prespecified sensitivity run.

## Locate and inspect from a fresh login shell

```bash
export REPO=/blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool
export RUN_DIR="$REPO/runs/novae_skin_pilot/h5ad-provenance-fix-20260906_024602"
cd "$REPO"
test -d "$RUN_DIR"
find "$RUN_DIR" -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort
stat -c '%n\t%s bytes' "$RUN_DIR"/novae_skin_visium_ssc_zero_shot.h5ad "$RUN_DIR"/novae_zero_shot_model/model.safetensors
python3 - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
manifest = next(run.glob("novae_resolved_manifest_*.json"))
envelope = json.loads(manifest.read_text())
print("scope:", envelope["run"]["analysis_scope"], "reference:", envelope["run"]["reference"])
for name, artifact in envelope["artifacts"].items():
    print(f"{name}\t{run / artifact['path']}\t{artifact['sha256']}")
PY
printf '%s\n' 'CSV headers:'
for csv in "$RUN_DIR"/*.csv; do printf '%s: ' "$(basename "$csv")"; head -n 1 "$csv"; done
```

These are metadata/JSON/CSV inspections only: they do not import AnnData, open
the H5AD, or recompute hashes. Keep the complete run directory, scheduler
logs, resolved provenance and manifest, QC/metric tables, annotated H5AD, and
saved model on Blue/shared storage or an institutional archive. H5AD content
validation and any full artifact rehash after archival belong in a scheduled
SLURM job; compare there against the recorded SHA-256 values. Do **not** add
these large H5AD/model artifacts to Git (the H5AD alone is approximately 804
MiB); Git should retain code and this report, not large run binaries.
