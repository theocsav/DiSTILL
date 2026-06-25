# DiSTILL

DiSTILL (Disease Diagnosis from Spatial Transcriptomics via Interpretable Latent Learning) is a hybrid cloud-HPC workflow system for reproducible spatial transcriptomics analysis. The repository contains the web application, API control plane, pipeline runner, presets, dataset-registry structure, and deployment documentation used to turn preset-backed workflow definitions into reproducible, `SLURM`-ready runs.

DiSTILL is designed as an application-layer wrapper around existing spatial transcriptomics analysis code. Its main contribution is operational: preflight validation, per-run materialization, queue-aware scheduler orchestration, artifact handling across split deployment planes, and support for schema-constrained dataset intake.

## Features

- Preset-driven configs with organ/platform compatibility filters.
- Dataset registry with cached schema manifests for fast preflight checks.
- Run orchestration (prepare, submit, queue) with artifact browsing and logs.
- Multiple execution backends, including direct local submission, SSH-mediated submission, and pull-based HPC poller execution.
- Optional `SLURM` fallback for preflight join-key validation.
- Schema-constrained dataset intake for both raw-entry and downstream-entry workflows.

## Repository layout

- `apps/api` FastAPI backend (auth, preflight, runs, registry endpoints).
- `apps/web` Next.js UI (wizard for presets, datasets, runs).
- `presets/` analysis presets used by the API/UI.
- `registries/` dataset registry (for example, `registries/datasets.json`).
- `pipeline_assets/` notebooks, scripts, and templates used by the pipeline.
- `scripts/` helper utilities such as dataset-manifest generation.
- `examples/` sanitized example configs and smoke-test artifacts.
- `runs/` runtime output (ignored in git; do not commit).
- `deploy/` deployment notes.
- `docs/` architecture, contract, deployment, and manuscript-support documentation.

## Quickstart (local)

### API

```bash
cd apps/api
python -m venv .venv

# On Linux/Mac:
. .venv/bin/activate

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

### Web UI

```bash
cd apps/web
npm install
export NEXT_PUBLIC_API_BASE=http://localhost:8000
npm run dev
```

PowerShell:

```powershell
cd apps/web
npm install
$env:NEXT_PUBLIC_API_BASE = "http://localhost:8000"
npm run dev
```

Login uses a secure cookie. Set `BASIC_AUTH_USER` and `BASIC_AUTH_PASS` (or `AUTH_PASSWORD_HASH`) together with `SESSION_SECRET` before deployment.

## Running a pipeline

Runs can be created via the UI or the API. The canonical execution path is:

1. `apps/web` or `apps/api`
2. `apps/api/app/runner.py`
3. `run_pipeline.py`
4. generated run directory under `runs/<run_name>/`
5. stage scripts and notebooks referenced by the selected preset

Legacy one-off scripts under `pipeline_assets/` remain in the repo for reference, but new reproducible runs should be expressed as presets and launched through `run_pipeline.py`.

Useful docs:

- CLI-style flow and SSH submitter guidance: `docs/SSH_SUBMITTER.md`
- HPG environment setup: `docs/HPG_SETUP.md`
- Cross-site env and upload tuning: `docs/ENV_TEMPLATE.md`
- Current RunSpec/RunState contract: `docs/RUNSPEC_RUNSTATE.md`
- Supported upload formats and dataset requirements: `docs/DATASET_CONTRACT.md`
- Canonical versus legacy entrypoints: `docs/DISTILL_PIPELINE_AUDIT.md`

Typical run config fields include:

- `run_name`, `mode`, `stages`
- `cosmx_h5ad_path`, `reference_h5ad_path`, `cell_metadata_path`
- optional `dataset_id` to resolve paths from the registry
- `slurm` settings for scheduling

## Dataset intake model

DiSTILL supports schema-constrained dataset intake rather than arbitrary file ingestion.

Raw-entry workflows typically begin from:

- a spatial `.h5ad`
- matching metadata
- a reference `.h5ad`, supplied either by the dataset record itself or by a compatible preset or registry-backed default

Downstream-entry workflows can begin from an existing NMF-annotated artifact such as `cosmx_with_nmf.h5ad`.

For exact accepted contracts and stage requirements, see `docs/DATASET_CONTRACT.md`.

## Split execution for large runs

Large HPG runs can be split into:

- GPU `cell2loc`
- CPU `nmf`, `post_nmf`, `rcausal_mgm`, `mlp`, `report`

This split is useful when `cell2location` benefits from CUDA but downstream stages do not.

The leakage-safe MLP stage also supports explicit execution modes via presets:

- `nested_cv`: full nested grouped tuning plus outer evaluation.
- `tune_once`: one grouped hyperparameter search on the full training universe, then save fixed params.
- `evaluate_fixed`: grouped outer CV using fixed hyperparameters from a saved params file.
- `explain`: fit a final explanatory model and emit SHAP artifacts without rerunning evaluation.

A typical manuscript-oriented flow is:

1. `cell2loc` and `nmf`
2. `tune_once`
3. `evaluate_fixed`
4. optional `explain`

This keeps fast iteration separate from the most expensive evaluation runs.

## Dataset registry

The dataset registry lives in `registries/datasets.json`. Use `scripts/generate_dataset_manifest.py` to compute cached schema manifests such as observed keys, metadata columns, and raw-count flags and store them in the registry.

See `apps/api/README.md` for fuller API and registry details.

## Preflight validation

Preflight checks include:

- required config keys
- paths inside allowlisted roots
- path existence and readability
- dataset-registry compatibility checks
- join-key compatibility using cached manifests or live validation

If `anndata` and `pandas` are not available on the API host, enable `PREFLIGHT_SLURM_FALLBACK=true` to run a short `SLURM` validation job.

## Deployment

Deployment notes live under `deploy/` and `docs/`, especially:

- `deploy/README.md`
- `docs/HPG_SETUP.md`
- `docs/SSH_SUBMITTER.md`
- `docs/HPG_POLLER.md`

## Testing

```bash
pytest apps/api/tests
```

## Notes for publishing

- Keep runtime output in `runs/` (gitignored).
- Keep examples sanitized in `examples/`.
- Set strong secrets for `SESSION_SECRET` and auth credentials.
- Verify figure assets and bibliography inputs before manuscript submission.

## Citation

If you use DiSTILL, please cite:

Engineering Spatial and Molecular Features from Cellular Niches to Inform Predictions of Inflammatory Bowel Disease.  
Myles Joshua Toledo Tan, Maria Kapetanaki, Panayiotis V. Benos.  
arXiv:2509.09923 (2025). https://doi.org/10.48550/arXiv.2509.09923

BibTeX:

```bibtex
@article{tan2025engineering,
  title = {Engineering Spatial and Molecular Features from Cellular Niches to Inform Predictions of Inflammatory Bowel Disease},
  author = {Tan, Myles Joshua Toledo and Kapetanaki, Maria and Benos, Panayiotis V.},
  year = {2025},
  eprint = {2509.09923},
  archivePrefix = {arXiv},
  primaryClass = {q-bio},
  doi = {10.48550/arXiv.2509.09923},
  url = {https://arxiv.org/abs/2509.09923}
}
```
