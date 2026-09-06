# NOVAE integration plan

**Status:** standalone Phase 0/1 implementation exists and is under validation. The pilot now includes explicit expression-mode auditing, transactional branch artifacts, and a pinned provenance envelope. It is intentionally parallel to the production pipeline; runner/API/UI and confirmatory fold-safe integration remain out of scope. See the [completed skin pilot run report](NOVAE_SKIN_PILOT_RUN_2026-09-06.md) for HPG outcomes and calibration limits.

## Decision and boundary

NOVAE should be evaluated as a **parallel representation branch** from a spatial
H5AD and merged with DiSTILL only at the feature-table boundary. It is not an
NMF replacement. In particular, `NMF_factor` and `dominant_nmf_factor` retain
their current meaning and must never be aliased to a NOVAE domain, latent
coordinate, or other NOVAE output.

The agreed sequence is:

1. coordinate and input audit;
2. standalone NOVAE pilot;
3. biological and predictive benchmark;
4. runner/API/UI preset integration only if the go criteria below pass.

Until step 4 is approved, NOVAE artifacts are produced and consumed outside
the canonical production stages.

## Confirmed NOVAE behavior and dependencies

The following is the working external contract, based on the Nature Methods
2025 paper and the current project and documentation pages:

- Paper: [Nature Methods (2025)](https://www.nature.com/articles/s41592-025-02899-6).
- Current repository: <https://github.com/prism-oncology/novae>.
- Official docs: <https://prism-oncology.github.io/novae/>.
- Model API: <https://prism-oncology.github.io/novae/api/Novae/>.
- Advice on use: <https://prism-oncology.github.io/novae/advice/>.
- The currently checked package is v1.1.1 and requires Python >=3.11. These
  values, the repository revision, and the pretrained checkpoint digest must be
  pinned for any reproducible run; current code and docs may evolve. The
  observed HF model revision is
  `b8c0a5d7612bac6bc719ab57ed3cd16ad814728c`; retain the environment override
  for deliberate changes and prefer a local cached snapshot. For a local model
  directory, `--model-revision` is requested provenance only, not independent
  verification; content/file hashes are authoritative unless a separately
  verified sidecar or cache identity exists.
- Input is AnnData. NOVAE expects `obsm["spatial"]`, accepts a `slide_key`, and
  supports Visium and CosMx. It builds/records the spatial graph in
  `obsp["spatial_connectivities"]`.
- The published/current default latent representation is 64-dimensional and
  is written to `obsm["novae_latent"]`. In v1.1.1, `obs["neighborhood_valid"]`
  is the official validity mask: NOVAE fills invalid-neighborhood latent rows
  with zeros and leaf/domain rows with missing values. The adapter retains
  those values, audits assignment coverage (default minimum 0.70) overall and
  per slide at every resolution, and fails if a valid neighborhood is missing
  a domain or an invalid row receives one. Domain and latent summaries exclude
  unassigned/invalid rows and report their denominators; no literal `nan`
  domain is fabricated. Zero-count expression rows and zero-degree graph rows
  are retained as QC, and radius-pruning pre/post zero-degree counts are
  recorded. If coverage is below threshold, inspect graph/coordinate and tissue
  QC and resubmit rather than imputing labels.
- Domain assignments are written to
  `obs["novae_domains_*"]` (the exact suffix is a model/config detail and must
  be recorded). Zero-shot exploratory inference uses `reference="all"` and
  computes one representation and updates cohort-derived zero-shot prototypes,
  after which multiple resolutions can be assigned quickly without rerunning
  the encoder, graph, or prototypes.
- The official v1.1.1 QC API is `novae.monitor.mean_fide_score(adatas, obs_key,
  slide_key=None, n_classes=None)` and
  `novae.monitor.jensen_shannon_divergence(adatas, obs_key, slide_key=None)`.
  FIDE is a continuity metric (high means continuity); both are comparative,
  not absolute pass/fail metrics, and JSD across biologically different slides
  must not be blindly minimized.
- The Visium model configuration is expected to use `n_hops_local=2` and
  `n_hops_view=2` (the allowed published values are 1 or 2); the pilot records
  and validates these checkpoint hyperparameters and does not override them.
- The pretrained zero-shot model is `prism-oncology/novae-human-0`.
  Zero-shot inference is the default first experiment. NOVAE zero-shot with
  `reference="all"` recomputes prototypes from all referenced slides, so a
  whole-cohort zero-shot result is exploratory only.
- For every patient-held-out confirmatory fold, use a fresh instance of the
  pinned base model and set its prototype reference to training slides only
  (for example, the training slide IDs as `reference`). Compute and freeze the
  prototypes and domain definition before assigning held-out slides. Held-out
  patients must not influence prototype computation, domain definition,
  resolution choice, preprocessing, scaling, or fine-tuning. Fine-tuning and
  prototype adaptation, if used, are likewise training-fold-only.
- Continuous latent embeddings from a fixed pretrained encoder may be evaluated
  as a separate arm, but the run must document and verify whether any
  cohort-derived state (including normalization, scaling, or other fitted
  state) was used.

These are **confirmed behavior** statements, not a promise that future NOVAE
releases preserve the same keys or defaults. The pilot must record the resolved
package version, source revision, model identifier, checkpoint hash, effective
latent dimension, domain key, and all graph/calibration settings. Confirmatory
runs must additionally record the training-only prototype reference IDs and
assert that no held-out slide ID appears in them.

## Where this fits in this repository

The current DiSTILL execution contract is the preset-driven path
`run_pipeline.py` -> generated run directory -> stage scripts/notebooks. The
relevant existing materials are:

- `run_pipeline.py` (canonical stage vocabulary: `cell2loc_nmf`, `cell2loc`,
  `nmf`, `post_nmf`, `rcausal_mgm`, `mlp`, `report`);
- `pipeline_assets/IBD_Post_NMF_Analysis.ipynb` (post-NMF enrichment and
  neighborhood features);
- `pipeline_assets/IBD_Build_FOV_MLP_Inputs.py` (FOV feature-table assembly,
  including `nmf_prop_*` columns);
- `pipeline_assets/IBD_MLP_LeakageSafe.py` (patient-grouped nested evaluation);
- `docs/DATASET_CONTRACT.md` (spatial H5AD identifiers and CosMx pixel
  coordinate requirements);
- `docs/MASLD_THESIS_PLAN.md` (MASLD label and patient-mapping stop
  condition).

The existing post-NMF neighborhood implementation uses an area/radius
`BallTree` over `CenterX_global_px`/`CenterY_global_px`. That is not the NOVAE
feature definition. NOVAE domain adjacency must be derived from graph edges in
`obsp["spatial_connectivities"]`, especially for Visium, where an area/radius
heuristic does not represent the array lattice reliably.

The current standalone adapter (`scripts/run_novae_pilot.py`) reads the spatial H5AD, performs explicit coordinate and expression audits, runs one exploratory NOVAE zero-shot representation computation, assigns the prespecified 0.5/1.0/2.0 resolution sweep, and transactionally publishes a separate annotated H5AD plus branch-specific tables and a `{run, artifacts}` provenance envelope. `scripts/submit_novae_skin_pilot.sh` is the UF HiPerGator launcher and submits exactly one GPU job; `requirements-novae.txt` defines the dedicated pinned environment, and the focused adapter tests live in `tests/test_novae_pilot.py`. It must not silently rewrite the NMF artifact or change the meaning of existing columns. Runner/API/UI invocation and confirmatory fold-safe integration remain future work. The launcher defaults to a recorded 100 µm physical graph-radius prune; the adapter applies it after Visium graph construction because NOVAE 1.1.1 ignores its generic `radius` argument for Visium.

## Dataset-specific input and boundary contract

All spatial coordinates must have auditable physical-distance semantics
(micrometres) for distance diagnostics and model distance use. For Visium, the
standalone pilot defaults to an expected 100 µm center spacing with a 0.5
relative tolerance and checks each slide's median positive calibrated graph
edge distance; this accommodates diagonal lattice edges and tissue gaps rather
than assuming every edge is exactly 100 µm. Current
repository coordinate columns are in pixels, while NOVAE guidance expects
physical coordinates/microns. If all slides share one scale, either
materialize coordinates in micrometres or retain the original coordinates and
pass the single scalar float `scale_to_microns` to NOVAE. NOVAE does not accept
a per-slide scale map. If scales differ by slide, the standalone adapter uses
its own per-slide conversion and materializes harmonized micrometre coordinates
before combined processing, or processes slides separately; a per-slide map may
be retained only as adapter provenance/configuration. A pixel-to-micron factor
must come from platform metadata or a validated dataset manifest; it must not
be guessed from the observed coordinate range. Record the source, factor, axis
convention, and any per-slide calibration in the audit artifact.

Keep these concepts separate: `scale_to_microns` affects distance use during
NOVAE training/inference, but does not change an already determined Delaunay
topology; uniform coordinate scaling likewise does not change Delaunay
connectivity. Calibration is still mandatory for distance semantics and for
any radius pruning or distance threshold. For Visium, explicitly prune the
constructed `spatial_connectivities` and corresponding `spatial_distances` by
Euclidean coordinate distance in µm before representation computation: NOVAE
1.1.1 ignores its `radius` argument for `technology="visium"`, and replacing
the lattice with a generic radius graph would change the intended topology.
Record pre/post/removed edge counts per slide and retain the zero-cross-slide
assertion. This explicit physical prune is distinct from NOVAE's generic-graph
radius construction. Graph partition/topology, model distance scaling, and
radius-based graph construction must each be recorded and audited independently.

`slide_key` is a hard graph boundary: no edge may join observations with
different values, even if their coordinates overlap after concatenation. The
following are proposed boundary units (the stable IDs must come from the
validated dataset manifest, not inferred from disease labels):

| Dataset | Current readiness | Boundary used for NOVAE | Required coordinate/metadata audit |
|---|---|---|---|
| Skin Visium | Ready; roughly 14 patients | Each Visium capture area/library/section (`slide_key`), with patient retained as a separate grouping key | Map each section/library to its patient; identify `obsm["spatial"]` coordinate semantics; use materialized µm coordinates or a recorded `scale_to_microns`; verify tissue/library extents and zero cross-library edges |
| Kidney CosMx | Ready; 259,978 cells x 1,207 genes, 6 samples (3 HD, 3 SSc) | Each CosMx sample (`slide_key`); FOV is nested within the sample and is not a substitute for the sample boundary | Preserve sample, patient/subject if available, FOV, cell ID, and global pixel coordinates; record px -> µm calibration or `scale_to_microns`; verify every FOV belongs to exactly one sample |
| MASLD Visium | 17,573 spots x 18,085 genes across 10 sections; no validated patient/fibrosis labels | Each Visium section/library (`slide_key`), one boundary per section | Validate barcode/section identity and coordinate scale; use materialized µm coordinates or recorded `scale_to_microns`; retain labels as unavailable; unsupervised only until the barcode-to-patient/fibrosis mapping in `docs/MASLD_THESIS_PLAN.md` is obtained |

The audit must fail closed for missing or ambiguous boundary IDs, non-finite or
duplicate coordinates, missing `obsm["spatial"]`, or an unverified unit
conversion. For concatenated inputs, preserve globally unique observation IDs
and test that all graph edges are within a boundary.

## Proposed configuration contract

This is a **proposed future contract**, not a change to the current runner
schema. Names can be adapted to existing preset conventions after the pilot.
A NOVAE run should resolve and persist at least:

```json
{
  "novae": {
    "enabled": true,
    "input_h5ad": "<spatial source artifact>",
    "output_dir": "<run output>/novae",
    "slide_key": "<validated section/sample column>",
    "group_key": "patient",
    "coordinate_key": "spatial",
    "coordinate_source_columns": ["<x>", "<y>"],
    "coordinate_representation": "source_plus_scale",
    "coordinate_input_unit": "px",
    "coordinate_target_unit": "um",
    "coordinate_strategy": "shared_scalar",
    "scale_to_microns": "<single shared scalar float or null>",
    "microns_per_pixel_by_slide": "<optional adapter-owned map or null>",
    "graph_topology": "per-slide; no cross-slide edges",
    "radius_pruning": "<none or recorded physical threshold>",
    "cross_slide_edges": false,
    "model_id": "prism-oncology/novae-human-0",
    "model_revision": "<pinned revision>",
    "checkpoint_sha256": "<pinned digest>",
    "package_version": "1.1.1",
    "inference_mode": "zero_shot",
    "reference": "all",
    "scope": "exploratory (confirmatory uses training slide IDs)",
    "confirmatory_reference_scope": "training_slides_only",
    "fresh_base_model_per_fold": true,
    "freeze_prototypes_before_holdout_assignment": true,
    "latent_key": "novae_latent",
    "domain_key_prefix": "novae_domains_",
    "seed": 42
  }
}
```

`coordinate_source_columns` and `scale_to_microns` may instead be recorded in
an input-manifest sidecar when the builder already materializes
`obsm["spatial"]`; `coordinate_representation` must distinguish materialized
micrometre coordinates from original coordinates plus scale. `scale_to_microns`
is a single scalar float when passed to NOVAE. The optional
`microns_per_pixel_by_slide` map is adapter-owned and must not be passed to
NOVAE. These are mutually appropriate strategies: use `shared_scalar` with
that scalar, or use `adapter_harmonized_um` with the map and materialized
harmonized coordinates, or use `per_slide_separate` with separate NOVAE runs.
The resolved manifest must state whether coordinates were copied, reordered,
transformed, or scaled. Graph topology/partition, radius pruning, model resolution,
preprocessing/scaling state, fine-tuning hyperparameters, software
lockfile/environment, and device should also be captured. For each
confirmatory fold it must include the training slide IDs used as `reference`,
the frozen prototype artifact, and the held-out slide IDs assigned afterward.
If the API exposes additional required parameters, the adapter must record them
rather than relying on undocumented defaults.

## Artifacts and feature comparison

Proposed names below are intentionally branch-specific and should be scoped by
run ID and dataset ID:

- `novae_coordinate_audit_<dataset>.json` and
  `novae_coordinate_audit_<dataset>.csv`: source columns, units, scale,
  boundaries, counts, ranges, and graph edge checks;
- `novae_<dataset>_zero_shot.h5ad`: standalone NOVAE output, including the
  source expression, `obsm["spatial"]`, `obsp["spatial_connectivities"]`,
  `obsm["novae_latent"]`, and `obs["novae_domains_*"]`; a `reference="all"`
  artifact must be marked exploratory and must not be used as a confirmatory
  domain assignment;
- `novae_<dataset>_fold-<id>.h5ad`: confirmatory fold output containing the
  training-only reference slide IDs, frozen prototypes/domain definition,
  preprocessing/scaling state, held-out slide IDs, and assignments made after
  freezing; the artifact must assert that held-out IDs are absent from the
  prototype references;
- `novae_domain_adjacency_<dataset>_res-<token>.csv` and matching slide/patient
  proportion tables: graph-edge counts/proportions by resolution (excluding
  edges touching unassigned observations and reporting total/used/excluded
  edge coverage), with a safe normalized resolution token;
- `novae_domain_resolution_summary_<dataset>.csv`: resolution, domain key,
  domain count/size summaries, overall and per-slide total/valid/assigned/
  unassigned coverage audits, and domains present per slide;
- `novae_science_metrics_<dataset>.csv`: official FIDE/JSD per resolution and
  interpretation metadata; `novae_neighbor_distance_qc_<dataset>.csv` records
  calibrated physical edge-distance medians and tolerance decisions;
- `novae_latent_summary_<dataset>.csv`: one-time 64-D latent summaries (for
  example mean and standard deviation) at the analysis unit, with slide keys;
- corresponding fold feature tables only for confirmatory fold-specific
  zero-shot, fine-tuned, or prototype-adapted models;
- `novae_feature_matrix_<dataset>_<branch>.parquet`, plus a schema/provenance
  JSON describing every column;
- `novae_benchmark_metrics_<dataset>.csv`, per-unit predictions, fold
  assignments, and a comparison report.

The benchmark must keep these feature families separately identifiable:

| Feature table/arm | Contents | Role |
|---|---|---|
| Expression baseline | Pre-specified QC/normalised expression summaries | Baseline |
| DiSTILL NMF | Existing NMF proportions and downstream NMF-derived features | Existing branch; retain `NMF_factor` semantics |
| NOVAE domains/adjacency | Domain proportions and domain-pair adjacency computed from NOVAE graph edges | Spatial NOVAE branch |
| NOVAE latent summaries | Aggregated summaries of `obsm["novae_latent"]` | Continuous NOVAE branch |
| Combined | Explicitly joined baseline + NMF + NOVAE families | Incremental-value test |

Feature generation and joins must preserve the analysis unit and stable keys.
For confirmatory domain features, use the frozen training-only prototypes and
assign held-out slides only afterward. Do not use the current area/radius
`BallTree` output as a proxy for NOVAE adjacency. Do not choose one global NOVAE
resolution using disease labels; this pilot predeclares 1.0 as primary and
emits 0.5, 1.0, and 2.0 in one run. Resolution/model settings must be
pre-specified or selected using label-free stability/biological criteria inside
each required training scope. A fixed pretrained encoder's continuous latent
arm is separate and must disclose any cohort-derived preprocessing state. The
pilot saves `Novae.save_pretrained` state when v1.1.1 supports it. Because
zero-shot inference updates prototypes from the whole cohort, the saved state
contains cohort-derived prototypes and can support cheap further
`assign_domains` calls on this annotated cohort without re-encoding. It remains
exploratory/contaminated and must never initialize held-out confirmatory
analysis; the emitted sweep is therefore the prespecified cheap option.

## Evaluation design and phases

### Phase 0 — coordinate/input audit

For each dataset, validate AnnData shape, genes, observation IDs, expression
layer, `obsm["spatial"]`, slide boundaries, units, and sample/patient metadata.
Construct or validate the graph without cross-slide edges and emit edge-count,
degree, component, and distance diagnostics. Record whether topology was
already determined (for example, Delaunay), whether NOVAE uses
`scale_to_microns`, and any radius pruning/thresholds; do not claim that
micrometre scaling changes Delaunay connectivity. Confirm that a round trip
preserves row order and IDs. This phase is a prerequisite for every subsequent
comparison.

**Exit:** zero unresolved coordinate/boundary errors; physical-distance
calibration or `scale_to_microns` provenance is complete; graph topology and
partition are boundary-safe; radius settings are auditable; the input and
environment manifest is reviewable.

### Phase 1 — standalone pilot

Submit exactly one GPU job for the current initial exploratory question: the
full skin Visium cohort. It computes the latent representation once, then
assigns the predeclared 0.5, 1.0 (primary), and 2.0 resolutions and all
resolution-specific QC/tables. No fine-tuning or per-sample jobs are warranted
for this pilot. Kidney and MASLD are deferred until the skin go/no-go; MASLD
will remain unlabeled/unsupervised when reconsidered. A full-cohort run with
`reference="all"` is exploratory only because its prototypes use all referenced
slides. Keep NMF untouched. Check output keys, latent shape (expected 64
columns unless explicitly configured otherwise), domain coverage, graph
sparsity, model distance scaling, and memory/runtime. FIDE depends on domain
granularity/domain count, so compare its values across resolutions cautiously;
do not choose the highest FIDE as an automatic selection rule. A repeat run is
only justified if the first run exposes a failure or later reproducibility
confirmation is needed; it is not a prerequisite for interpreting the initial
exploratory output.

**Exit:** all expected artifacts are produced; domain and latent outputs join
back to source observations without row drift; resource requirements and failure
modes are known. Reproducibility confirmation, if needed, is a subsequent
operational/scientific check rather than a Phase 1 prerequisite.

### Phase 2 — biological and predictive benchmark

First compare representation stability and biological structure without using
disease labels to select global resolution: domain balance/coverage, spatial
autocorrelation or neighborhood consistency, replicate/section stability, and
known cell-state or tissue-structure correspondence where annotations exist.

Then construct the five feature arms above. Use patient-grouped,
leakage-safe evaluation for any supervised endpoint. The outer split must hold
out complete patients (and must not split spots/cells/FOVs from one patient
across train and test). For every confirmatory fold, start from a fresh pinned
base model, reference training slide IDs only, compute and freeze prototypes
and the domain definition on training slides, then assign held-out slides.
Held-out patients must not influence zero-shot prototype computation, domain
definition, resolution choice, preprocessing/scaling, fine-tuning, or prototype
adaptation. Fixed-encoder continuous latents can be a separate arm only after
verifying whether any cohort-derived state is used. Model tuning, feature
selection, scaling, and resampling also belong inside the training fold.
Report pooled out-of-fold metrics, per-patient predictions, fold assignments,
class counts, and uncertainty—not a maximum selected during the same folds.
Whole-cohort exploratory tables/figures (including `reference="all"` zero-shot
outputs) must be labelled **exploratory**; fold-safe results are
**confirmatory**.

The eventual adapter must include a regression test that constructs a held-out
fold and asserts every held-out slide ID is absent from the NOVAE prototype
reference list before assignment.

Skin is the primary predictive candidate (roughly 14 patients). Kidney has only
six samples (3 HD/3 SSc): use it for representation/biological checks and label
any predictive analysis as exploratory; do not present a disease classification
number as a generalisable result. MASLD remains unsupervised until validated
patient/fibrosis labels are available.

**Exit:** the NOVAE branch has a prespecified comparison against expression and
NMF, no label leakage, and a written interpretation of both positive and null
results.

### Phase 3 — conditional product integration

Only after Phase 2 go criteria pass, add a runner/API/UI preset that invokes the
standalone adapter and publishes branch-specific artifacts. Integration must
be opt-in, preserve existing stages and outputs, expose pinned versions and
calibration in run provenance, and keep the feature-table merge explicit.
No UI/API work is authorized by this document's current scope.

## Go/no-go criteria

Proceed to product integration only if all are true:

- Input audit passes for each selected dataset, including either materialized
  µm coordinates or a verified `scale_to_microns`, complete slide boundaries,
  zero cross-slide graph edges, and separately recorded topology, model
  distance scaling, and radius settings.
- The pinned zero-shot run is reproducible within a documented tolerance and
  has complete provenance (package, Python, repository/model revisions,
  checkpoint hash, seed, device, graph/calibration settings).
- Every confirmatory zero-shot fold uses a fresh pinned base model, training
  slide IDs only as prototype references, frozen training-derived prototypes,
  and held-out assignment afterward. The artifact and regression test assert
  that held-out slide IDs are absent from prototype references; no held-out
  state enters domain definition, resolution, preprocessing/scaling, or
  adaptation. Fixed-encoder latent results separately document cohort-derived
  state, if any.
- Outputs meet the schema contract: expected latent dimension/key, domain key,
  complete IDs, and deterministic feature-table joins; `NMF_factor` is unchanged.
- Biological checks show non-empty, interpretable, and reasonably stable
  domains/adjacency across sections or replicates, without disease-guided
  global resolution selection.
- Predictive comparisons use patient-held-out nested evaluation. A NOVAE arm
  is promoted only when its predeclared primary metric shows a consistent,
  uncertainty-quantified improvement over the relevant baseline or provides a
  clearly supported complementary signal in the combined arm; a null result is
  a valid no-go.
- Runtime, memory, artifact size, and failure behavior are acceptable for the
  intended execution environment and are documented before enabling a preset.

## Risks and mitigations

- **Pixel/physical-scale mismatch:** wrong distance semantics or radius
  thresholds can invalidate results. Require manifest-backed calibration or a
  recorded `scale_to_microns`, unit audits, and explicit per-slide scales.
  Scaling alone does not alter an already determined Delaunay topology, so
  audit topology separately from model distance scaling and radius pruning.
- **Cross-slide edges after concatenation:** overlapping coordinate frames can
  create false neighborhoods. Make `slide_key` a hard graph partition and
  assert zero violations.
- **Model/docs drift:** pin package, source/model revisions, checkpoint digest,
  and environment; retain the resolved config and output manifest. Keep the
  actual HPG `pip freeze` and conda export with the run; package pins do not
  claim to pin CUDA runtime libraries or system GPU drivers.
- **Zero-shot domain mismatch or prototype contamination:** human pretraining
  may not represent these organs/platforms, and `reference="all"` can let
  held-out slides influence prototypes. Whole-cohort zero-shot is exploratory;
  confirmatory folds require a fresh pinned base model, training-slide-only
  references, frozen prototypes, and held-out assignment afterward. Test that
  held-out IDs are absent from every prototype reference list. Fine-tuning and
  prototype adaptation are training-fold-only.
- **Small or unavailable labels:** kidney has six samples and MASLD currently
  lacks validated labels. Keep kidney exploratory and MASLD unsupervised; do
  not substitute section names or guessed labels.
- **Pseudo-replication:** many spots/cells/FOVs are not independent patients.
  Aggregate/report at the declared unit and hold out patients throughout.
- **NMF semantic contamination:** merging outputs in-place can cause accidental
  aliases. Use NOVAE-prefixed keys and separate artifacts; assert NMF columns
  are preserved.
- **Resource cost and nondeterminism:** measure pilot memory/runtime, set seeds,
  record device/dtype, and retain logs and manifests.

## Reproducibility checklist

Before sharing a NOVAE result, retain:

- dataset/archive and source H5AD checksums, builder version, and exact input
  observation/feature counts;
- source-to-`obsm["spatial"]` mapping, axis order, either materialized µm
  coordinates or the `scale_to_microns` value and source, slide-boundary key,
  graph topology/partition, radius pruning/thresholds, boundary counts, and
  cross-slide-edge assertion;
- NOVAE package version, Python/environment lock, repository/model revisions,
  checkpoint hash, effective defaults, graph/model parameters, seed, device,
  and dtype;
- exact command/config, run ID, code revision, logs, warnings, and artifact
  checksums;
- feature schema and aggregation unit; patient/fold assignments; fresh base
  model per confirmatory fold; training slide IDs used as prototype references;
  frozen prototype/domain artifacts; held-out slide IDs; an assertion that
  held-out IDs are absent from prototype references; training-only selection,
  preprocessing/scaling, fine-tuning, and adaptation records;
- pooled out-of-fold predictions and metrics, per-patient results, and a clear
  label of exploratory whole-cohort versus confirmatory fold-safe outputs;
- unchanged NMF artifact/columns and the explicit feature-family join used for
  every benchmark arm.

## Not in current scope

- Wiring NOVAE into `run_pipeline.py`, existing stages, presets, configs, API,
  or web UI; the current implementation is standalone and exploratory.
- Confirmatory patient-held-out/fold-safe prototype computation and assignment;
  this pilot uses `reference="all"` and cannot feed confirmatory classification.
- Modifying existing stages, notebooks, or README.
- Replacing, renaming, recalibrating, or reinterpreting DiSTILL NMF or
  `NMF_factor`.
- Choosing disease-guided NOVAE resolution, tuning, or checkpoint selection on
  the whole cohort.
- Supervised MASLD fibrosis/disease claims before validated barcode-level
  patient and fibrosis metadata is obtained.
- Generalisable kidney disease classification from six samples.
- Treating area/radius `BallTree` enrichment as NOVAE graph adjacency.
- Mixing NOVAE domain IDs across independently fitted models as if IDs were
  intrinsically equivalent.
- Calling exploratory whole-cohort representations (including
  `reference="all"` zero-shot domains) confirmatory evidence.
- Treating zero-shot prototype computation, domain definition, resolution,
  preprocessing/scaling, or adaptation as valid when any held-out slide was
  referenced; the future fold regression test must prevent this.
