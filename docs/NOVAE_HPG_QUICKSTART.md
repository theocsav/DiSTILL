# NOVAE skin Visium pilot on UF HiPerGator

This is a standalone **exploratory Phase 0/1** branch. It is not a
`run_pipeline.py` stage and its `reference=all` domains must not be used for
confirmatory held-out classification. The scientific claim is exploratory
domain/representation biology, not a fully inductive held-out domain
classifier. Continuous latents may later be evaluated with patient-held-out
CPU modeling; whole-cohort domain labels remain exploratory. Use the dedicated
NOVAE environment, not the existing Python 3.10 pipeline environment (NOVAE
1.1.1 requires Python >=3.11).

## NOVAE neighborhood validity and coverage

NOVAE 1.1.1 writes the official boolean `obs['neighborhood_valid']` mask. A
false value means the requested `n_hops_view` neighborhood is unavailable;
NOVAE intentionally emits a zero-filled latent vector and missing (NaN)
leaf/domain values for that observation. Missing values are expected only for
false-mask rows: a valid neighborhood without a nonblank domain is a hard
failure, as is a domain assigned to an invalid row. Missing values remain
missing and are never converted to a literal `"nan"` category or imputed.

Every requested resolution is audited overall and per slide, with total,
valid-neighborhood, assigned, unassigned, coverage, and a bounded sample of
unassigned observation IDs in the resolution summary and provenance. Coverage
is assigned/total and must be at least **0.70** overall and for every slide by
default (NOVAE's
documented validity threshold). Use `--min-domain-assignment-coverage` or
`NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE` to override it with a value in `[0,1]`.
Validity is shared across resolutions, but each resolution is still audited.

Domain proportions exclude unassigned observations from domain counts while
reporting total/assigned/unassigned/coverage. Domain adjacency excludes edges
touching unassigned observations and reports total/used/excluded edge counts and
edge coverage. Latent summaries exclude invalid-neighborhood rows (whose zero
vectors are not biological latent values) and report total/valid/excluded rows
per unit. Official FIDE/JSD calls still run at every resolution with categorical
missing values intact as needed. If coverage is below threshold, inspect the
slide-level audit, coordinate/graph construction, tissue extent, and input QC
before resubmitting; never fill labels or silently filter rows.

Expression audits report zero-count rows, which are retained as QC requiring
scientific interpretation rather than auto-filtered. Graph diagnostics report
zero-degree observations, and radius-pruning provenance records pre/post
zero-degree counts.

## Create the environment

Environment creation and model caching are one-time setup (not scheduled
HPG work). On a network-capable login context, create a Python 3.12 environment and install
the pinned direct dependency set:

```bash
module load conda
conda create -y -p /blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312 python=3.12
conda activate /blue/kejun.huang/vasco.hinostroza/nicherunner/conda/envs/novae_pilot_py312
cd /blue/kejun.huang/vasco.hinostroza/nicherunner/src/sptx-tool
python -m pip install -r requirements-novae.txt
python - <<'PY'
import novae, torch
print("novae", novae.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
# Retain the exact environment used by the run (the requirements file does not
# pin CUDA runtime libraries or system GPU drivers).
mkdir -p runs/novae_skin_pilot
python -m pip freeze > runs/novae_skin_pilot/hpg-pip-freeze.txt
conda env export --no-builds > runs/novae_skin_pilot/hpg-conda-export.yml
```

Compute nodes should not need internet access. Cache and pin the model before
submitting. A local snapshot is preferred:

```bash
export MODEL_DIR=/blue/kejun.huang/vasco.hinostroza/models/novae-human-0
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "prism-oncology/novae-human-0",
    revision="<PINNED_COMMIT_SHA>",
    local_dir="/blue/kejun.huang/vasco.hinostroza/models/novae-human-0",
)
PY
# Keep this value recorded with the run and verify files are present.
find "$MODEL_DIR" -type f -maxdepth 2 -print
```

The currently observed revision is
`b8c0a5d7612bac6bc719ab57ed3cd16ad814728c`; pin it (or set
`NOVAE_MODEL_REVISION` to another reviewed revision) and retain it in the run
notes. Replace `<PINNED_COMMIT_SHA>` with the commit returned by
`HfApi().model_info` when refreshing the cache. Alternatively set
`NOVAE_MODEL_REVISION` and
let the launcher resolve the pinned Hugging Face snapshot into the normal
Hugging Face cache (only where network access is available). Model resolution
occurs before output staging; do not rely on compute-node internet. For a
local `--model-source` directory, `--model-revision` is requested provenance
only: `revision_verified` remains false, and the content/file hashes are
authoritative unless a separately verified sidecar or cache identity exists.

## Optional audit (not an HPG run)

The audit is optional and validates IDs, slide boundaries, coordinates, and
calibration without loading NOVAE or a model. It is audit-only, not a scheduled
HPG job:

```bash
python scripts/run_novae_pilot.py \
  --input-h5ad /blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_spatial.h5ad \
  --output-dir runs/novae_skin_pilot/audit \
  --dataset-id skin_visium_ssc \
  --slide-key sample_id --group-key patient --technology visium \
  --expression-mode raw_counts --model-source "$MODEL_DIR" --resolutions 0.5 1.0 2.0 \
  --primary-resolution 1.0 --expected-neighbor-distance-um 100 \
  --coordinate-strategy visium_manifest \
  --sample-manifest /blue/kejun.huang/vasco.hinostroza/data/skin_dataset/processed/skin_visium_ssc_sample_manifest.csv \
  --physical-spot-diameter-um 55.0 --audit-only
```

`--expression-mode` is required: use `raw_counts` only when `X` is finite,
nonnegative, and integer-like; use `preprocessed` for already transformed data.
The resolved provenance records expression dtype/shape/sparsity, layer/raw state,
and the explicit `novae.settings.auto_preprocessing` value (true for raw counts,
false for preprocessed). For raw counts, post-NOVAE normalized/log X is accepted
only when `layers['counts']` preserves the original counts (or X is unchanged).
No expression mode is inferred.

The adapter preserves input pixel coordinates as `obsm['spatial_original_px']`
and materializes `obsm['spatial']` in microns. After graph construction, the
Visium distance QC uses calibrated `obsp['spatial_distances']` and compares
per-slide median positive edge distance with the nominal 100 µm center spacing.
When `--graph-radius-um` is supplied, the adapter explicitly prunes the
already-built `spatial_connectivities` (and matching `spatial_distances`) by
Euclidean coordinate distance in physical microns before
`compute_representations`. This post-construction step is required because
NOVAE 1.1.1 ignores its `radius` argument for `technology='visium'`; it keeps
the Visium lattice topology rather than replacing it with NOVAE's generic-graph
radius construction. The two radius concepts are therefore distinct: the
launcher defaults to and records a 100 µm physical pruning threshold, while a
future generic-graph radius would control graph construction itself. Per-slide
pre/post/removed edge counts and cross-slide checks are recorded; a nontrivial
slide may not lose all edges.
The default relative tolerance is 0.5 (50%); diagonal lattice edges and tissue
gaps are expected, so this is not an all-edges-equal check. A slide outside the
range fails the run and is recorded with its pass/fail decision. This check is
run when coordinates are materialized in microns (the launcher uses the
manifest strategy); a shared scalar remains explicitly recorded rather than
being mislabeled as materialized distances.

For each observed
`sample_id`, the manifest's `spot_diameter_fullres` is used to calculate
`microns_per_pixel = 55.0 / spot_diameter_fullres`. The resolved coordinate
mapping, source key, per-slide factors, and ranges are written to the audit
JSON/CSV; no pixel-to-micron factor is guessed.

## Submit

Run from the login node. The launcher creates run/log parents before writing
and submitting an sbatch script; the final output directory is created only by
the successful transactional pilot:

```bash
export NOVAE_MODEL_PATH="$MODEL_DIR"
# One scheduled job computes all resolutions and QC from one encoder pass.
export NOVAE_RESOLUTIONS="0.5 1.0 2.0" # label-free; 1.0 is primary
export NOVAE_PRIMARY_RESOLUTION=1.0
export NOVAE_EXPECTED_NEIGHBOR_DISTANCE_UM=100
export NOVAE_GRAPH_RADIUS_UM=100 # validated and passed to the pilot
export NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE=0.70
export NOVAE_WORKERS=8
scripts/submit_novae_skin_pilot.sh
```

Normal submission is exactly one `sbatch` job: do not fine-tune or submit
per-sample jobs. All resolution assignments, domain/QC tables, and latent
summaries come from this one encoder/representation computation. Useful
overrides include `NOVAE_INPUT_H5AD`, `NOVAE_SAMPLE_MANIFEST`,
`NOVAE_CONDA_ENV`, `NOVAE_OUTPUT_DIR`, `NOVAE_QOS`, `NOVAE_PARTITION`,
`NOVAE_TIME`, `NOVAE_MODEL_REVISION`, `NOVAE_MIN_DOMAIN_ASSIGNMENT_COVERAGE`,
and `NOVAE_DATASET_ID`. The default input
is the original-sections `skin_visium_ssc_spatial.h5ad`, not a pseudo-FOV
artifact. The launcher requests one GPU, 8 CPUs, about 96 GB RAM, and the
current `kejun.huang` GPU QOS; no partition is assumed unless
`NOVAE_PARTITION` is set.

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,JobName%28,State,ExitCode,Elapsed,MaxRSS
tail -f runs/novae_skin_pilot/logs/novae_skin_pilot_<JOB_ID>.out
```

A run publishes all artifacts transactionally: the requested output directory
must not already exist, and a failed run leaves no final directory. Embedded
`adata.uns['novae_pilot_provenance']` contains the immutable core run provenance;
the external provenance/resolved-manifest JSON has `{run, artifacts}` where
`artifacts` contains final SHA-256 values, avoiding a circular self-hash.

Use `scripts/submit_novae_skin_pilot.sh --render-only` to generate and inspect
the sbatch file without submitting. Dataset IDs are validated safe slugs and
scheduler directive values reject unsafe characters.

Outputs include a separate annotated `novae_*_zero_shot.h5ad`, one
resolution-specific adjacency/proportion set for each requested resolution,
a combined domain-resolution summary, official NOVAE FIDE/JSD metrics (which
are comparative, not absolute pass/fail: high FIDE indicates continuity and
JSD must not be blindly minimized because biology can differ), neighbor-distance
QC against 100 µm by default, coordinate and
graph diagnostics, graph-edge domain adjacency, per-slide (and optional
patient) domain/latent summaries, and a resolved provenance JSON (also emitted
as a resolved-manifest JSON) with package versions, model/checkpoint file hashes, revisions, device and
Torch/CUDA details when available, seed, settings, and input/output hashes. Existing NMF columns
are checked and never aliased or overwritten. The run also saves the v1.1.1
zero-shot model state when `save_pretrained` is available and hashes it.
Because zero-shot representation computation updates prototypes from the whole
cohort, this saved state contains cohort-derived prototype weights. It can
support cheap additional `assign_domains` calls on the already annotated H5AD
without re-encoding, but is exploratory/contaminated and must never initialize
held-out confirmatory analysis.
