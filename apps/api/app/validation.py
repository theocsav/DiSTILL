import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import get_dataset
from .settings import ARTIFACT_ROOTS, QUEUE_POLLER_TOKEN, REPO_ROOT
from .ssh_exec import is_ssh_backend, remote_path_exists, remote_path_readable
from .storage import enforce_allowed_path


RAW_STAGE_REQUIRED_KEYS = ("cosmx_h5ad_path", "reference_h5ad_path", "cell_metadata_path")
PATH_KEYS = RAW_STAGE_REQUIRED_KEYS + ("cosmx_with_nmf_path", "ref_model_dir")
ALLOWED_STAGES = ("cell2loc_nmf", "post_nmf", "rcausal_mgm", "mlp", "report")
DEFAULT_POST_NMF_NOTEBOOK = "pipeline_assets/IBD_Post_NMF_Analysis.ipynb"
DEFAULT_RCAUSAL_NOTEBOOK = "pipeline_assets/IBD_RCausalMGM_Preparation.ipynb"
DEFAULT_MLP_SCRIPT = "pipeline_assets/IBD_MLP_44Features.py"
DEFAULT_REQUIRED_OBS_KEYS = ("fov", "cell_ID", "patient", "disease_status")
DEFAULT_REQUIRED_METADATA_COLUMNS = ("CenterX_global_px", "CenterY_global_px", "fov", "cell_ID")
NMF_LABEL_COLUMNS = ("NMF_factor", "dominant_nmf_factor")
COORD_COLUMNS = ("CenterX_global_px", "CenterY_global_px")
MORPHOLOGY_COLUMNS = ("Area", "Width", "Height")


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


class DependencyMissingError(RuntimeError):
    pass


def _read_h5ad_obs(path: Path):
    try:
        import anndata as ad
        import numpy as np
    except ImportError as exc:
        raise DependencyMissingError("anndata is required for join-key validation.") from exc
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs.copy()
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    return obs


def _read_metadata_header(path: Path) -> list[str]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyMissingError("pandas is required for join-key validation.") from exc
    return list(pd.read_csv(path, nrows=0).columns)


def _read_metadata_columns(path: Path, columns: list[str]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyMissingError("pandas is required for join-key validation.") from exc
    return pd.read_csv(path, usecols=columns)


def _build_join_keys(frame, strategy: str, delimiter: str) -> tuple[list[str], str]:
    if strategy == "unique_cell_id":
        if "unique_cell_id" not in frame.columns:
            raise RuntimeError("unique_cell_id not found.")
        return frame["unique_cell_id"].astype(str).tolist(), "unique_cell_id"
    if strategy == "fov_cell_id":
        if "fov" not in frame.columns or "cell_ID" not in frame.columns:
            raise RuntimeError("fov and cell_ID are required for join-key validation.")
        return (frame["fov"].astype(str) + delimiter + frame["cell_ID"].astype(str)).tolist(), "fov+cell_ID"
    raise RuntimeError("Unknown join key strategy.")


def _resolve_join_strategy(obs_columns: list[str], meta_columns: list[str]) -> str:
    if "unique_cell_id" in obs_columns and "unique_cell_id" in meta_columns:
        return "unique_cell_id"
    if "fov" in obs_columns and "cell_ID" in obs_columns and "fov" in meta_columns and "cell_ID" in meta_columns:
        return "fov_cell_id"
    raise RuntimeError("No compatible join keys found (unique_cell_id or fov+cell_ID).")


def _has_all(columns: list[str], required: tuple[str, ...]) -> bool:
    return all(column in columns for column in required)


def _has_any(columns: list[str], required: tuple[str, ...]) -> bool:
    return any(column in columns for column in required)


def _has_morphology(columns: list[str]) -> bool:
    return "Area" in columns or ("Width" in columns and "Height" in columns)


def _validate_stage_data_contract(
    stages: list[str],
    config: Dict[str, Any],
    check_paths: bool,
    errors: List[str],
    warnings: List[str],
    checks: Dict[str, Any],
) -> None:
    if "post_nmf" in stages and "cell2loc_nmf" not in stages and not config.get("cosmx_with_nmf_path"):
        errors.append(
            "post_nmf without cell2loc_nmf requires cosmx_with_nmf_path or an existing <output_dir>/cosmx_with_nmf.h5ad artifact."
        )

    if (
        "rcausal_mgm" in stages
        and "cell2loc_nmf" not in stages
        and not config.get("rcausal_h5ad_path")
        and not config.get("rcausal_niche_h5ad_path")
    ):
        errors.append(
            "rcausal_mgm without cell2loc_nmf requires rcausal_h5ad_path, rcausal_niche_h5ad_path, or an existing cosmx_with_nmf.h5ad artifact."
        )

    dataset = get_dataset(str(config.get("dataset_id") or "")) if config.get("dataset_id") else None
    manifest = (dataset or {}).get("schema_manifest", {}) if dataset else {}
    metadata_manifest = (dataset or {}).get("metadata_manifest", {}) if dataset else {}
    obs_columns = list(manifest.get("obs_keys", [])) if manifest.get("obs_keys") else []
    metadata_columns = list((dataset or {}).get("metadata_columns", [])) if dataset and dataset.get("metadata_columns") else []

    if not obs_columns or not metadata_columns:
        if not check_paths:
            return
        try:
            obs = _read_h5ad_obs(Path(config["cosmx_h5ad_path"]))
            obs_columns = list(obs.columns)
            metadata_columns = _read_metadata_header(Path(config["cell_metadata_path"]))
        except DependencyMissingError as exc:
            warnings.append(f"Stage data-contract validation skipped due to missing dependencies: {exc}")
            return
        except Exception as exc:
            warnings.append(f"Stage data-contract validation skipped: {exc}")
            return

    checks["stage_data_contract"] = {
        "cosmx_obs_columns": obs_columns,
        "metadata_columns": metadata_columns,
    }

    has_coords = bool(manifest.get("has_spatial_coordinates")) or bool(metadata_manifest.get("has_spatial_coordinates")) or _has_all(obs_columns, COORD_COLUMNS) or _has_all(metadata_columns, COORD_COLUMNS)
    has_morphology = bool(manifest.get("has_morphology")) or bool(metadata_manifest.get("has_morphology")) or _has_morphology(obs_columns) or _has_morphology(metadata_columns)
    has_nmf_labels = bool(manifest.get("has_nmf_labels")) or _has_any(obs_columns, NMF_LABEL_COLUMNS)

    if "cell2loc_nmf" in stages:
        if not has_coords:
            errors.append(
                "cell2loc_nmf requires spatial coordinates in h5ad obs or metadata CSV: CenterX_global_px and CenterY_global_px."
            )

    if "post_nmf" in stages:
        if not has_coords:
            errors.append(
                "post_nmf requires spatial coordinates in h5ad obs or metadata CSV: CenterX_global_px and CenterY_global_px."
            )
        if not has_morphology:
            errors.append(
                "post_nmf requires morphology in h5ad obs or metadata CSV: provide Area or both Width and Height."
            )
        if "cell2loc_nmf" not in stages:
            standalone_nmf_path = config.get("cosmx_with_nmf_path")
            if standalone_nmf_path:
                try:
                    standalone_obs = _read_h5ad_obs(Path(standalone_nmf_path))
                    standalone_columns = list(standalone_obs.columns)
                    checks["stage_data_contract"]["cosmx_with_nmf_obs_columns"] = standalone_columns
                    if not _has_any(standalone_columns, NMF_LABEL_COLUMNS):
                        errors.append(
                            "post_nmf without cell2loc_nmf requires an NMF-annotated h5ad with NMF_factor or dominant_nmf_factor."
                        )
                except Exception as exc:
                    errors.append(f"post_nmf standalone input unreadable: {standalone_nmf_path} ({exc})")
        elif not has_nmf_labels:
            warnings.append(
                "post_nmf will rely on cell2loc_nmf to generate NMF_factor before downstream analysis."
            )

    if "rcausal_mgm" in stages:
        if not has_coords:
            errors.append(
                "rcausal_mgm requires spatial coordinates in h5ad obs or metadata CSV: CenterX_global_px and CenterY_global_px."
            )
        if not has_morphology:
            warnings.append(
                "rcausal_mgm is most reliable when Area or Width/Height are available in h5ad obs or metadata CSV."
            )
        if "cell2loc_nmf" not in stages and not has_nmf_labels:
            warnings.append(
                "rcausal_mgm standalone runs should point to an NMF-annotated h5ad; the raw cosmx_h5ad_path does not expose NMF_factor."
            )


def _apply_join_key_thresholds(
    result: Dict[str, Any],
    max_missing_fraction: float,
    max_missing_count: int,
    errors: List[str],
    warnings: List[str],
) -> None:
    missing = int(result.get("missing", 0))
    missing_fraction = float(result.get("missing_fraction", 0.0))
    extra = int(result.get("extra", 0))
    if missing > max_missing_count or missing_fraction > max_missing_fraction:
        errors.append("Join-key validation failed: missing rows exceed threshold.")
    elif missing > 0:
        warnings.append("Join-key validation: some metadata rows are missing.")
    if extra > 0:
        warnings.append("Join-key validation: extra metadata rows not found in h5ad obs.")


def validate_config(
    config: Dict[str, Any],
    check_paths: bool = True,
    allow_join_fallback: bool = False,
    join_key_result: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, Any] = {"exists": {}, "roots": {}, "permissions": {}, "compute_access": {}}
    if is_ssh_backend():
        checks["compute_access"]["mode"] = "ssh_backend"
        checks["compute_access"]["summary"] = "API path checks run against the compute environment over SSH."
    elif QUEUE_POLLER_TOKEN:
        checks["compute_access"]["mode"] = "external_poller"
        checks["compute_access"]["summary"] = (
            "API path checks reflect API-host visibility only; they do not confirm HPG compute readability."
        )
    else:
        checks["compute_access"]["mode"] = "local"
        checks["compute_access"]["summary"] = "API path checks reflect the local execution host."

    def record_path_checks(key: str, value: Optional[str]) -> None:
        if not value:
            return
        path = Path(value)
        try:
            enforce_allowed_path(path, ARTIFACT_ROOTS)
            checks["roots"][key] = True
        except ValueError:
            checks["roots"][key] = False
            errors.append(f"{key} is outside ARTIFACT_ROOTS.")

        if not check_paths:
            checks["exists"][key] = "skipped"
            checks["permissions"][key] = "skipped"
            checks["compute_access"][key] = "skipped"
            return

        exists_result = remote_path_exists(value)
        exists = bool(exists_result)
        checks["exists"][key] = exists
        if not exists:
            errors.append(f"Path does not exist: {key} -> {value}")
            checks["permissions"][key] = "skipped"
            checks["compute_access"][key] = "missing"
            return

        if is_ssh_backend():
            readable_result = remote_path_readable(value)
            readable = bool(readable_result)
            compute_value: Any = readable
        else:
            readable = os.access(path, os.R_OK)
            compute_value = "unknown_external_poller" if QUEUE_POLLER_TOKEN else readable
        checks["permissions"][key] = readable
        checks["compute_access"][key] = compute_value
        if not readable:
            errors.append(f"Path not readable: {key} -> {value}")

    if QUEUE_POLLER_TOKEN and not is_ssh_backend():
        warnings.append(
            "API-host path checks do not confirm HPG compute readability in external poller mode; verify dataset paths on HPG or run an HPG smoke test."
        )

    stages = config.get("stages") or ["cell2loc_nmf"]
    if isinstance(stages, str):
        stages = [item.strip() for item in stages.split(",") if item.strip()]
    if not isinstance(stages, list):
        errors.append("stages must be a list of stage names.")
        stages = []

    required_keys = list(RAW_STAGE_REQUIRED_KEYS if "cell2loc_nmf" in stages else ("cosmx_h5ad_path",))
    for key in required_keys:
        if not config.get(key):
            errors.append(f"Missing required config key: {key}")

    mode = config.get("mode", "fixed_k")
    if mode == "fixed_k":
        if config.get("n_components") is None and config.get("k") is None:
            errors.append("Fixed-k mode requires n_components or k.")
    elif mode == "elbow_k":
        k_min = int(config.get("k_min", 2))
        k_max = int(config.get("k_max", 20))
        if k_max < k_min:
            errors.append("elbow_k requires k_max >= k_min.")
    else:
        errors.append("mode must be fixed_k or elbow_k.")

    output_dir = config.get("output_dir")
    if output_dir:
        try:
            enforce_allowed_path(Path(output_dir), ARTIFACT_ROOTS)
            checks["roots"]["output_dir"] = True
        except ValueError:
            checks["roots"]["output_dir"] = False
            errors.append("output_dir is outside ARTIFACT_ROOTS.")

    for key in PATH_KEYS:
        value = config.get(key)
        record_path_checks(key, value)

    dataset_id = config.get("dataset_id")
    if dataset_id:
        dataset = get_dataset(dataset_id)
        if not dataset:
            errors.append(f"Dataset not found: {dataset_id}")
        else:
            staged_path = dataset.get("staged_path") or dataset.get("cosmx_h5ad_path")
            if not staged_path:
                errors.append(f"Dataset {dataset_id} missing staged_path.")

            metadata_path = dataset.get("cell_metadata_path")
            if not metadata_path:
                errors.append(f"Dataset {dataset_id} missing cell_metadata_path.")

            manifest = dataset.get("schema_manifest", {})
            obs_keys = manifest.get("obs_keys", [])
            required_obs_keys = config.get("required_obs_keys", DEFAULT_REQUIRED_OBS_KEYS)
            missing_obs = [key for key in required_obs_keys if key not in obs_keys]
            if missing_obs:
                errors.append(f"Dataset missing obs keys: {', '.join(missing_obs)}")
            require_raw = config.get("require_raw_counts", True)
            if require_raw and manifest and not manifest.get("has_raw_counts", False):
                errors.append("Dataset schema manifest reports no raw counts.")

            required_meta = config.get("required_metadata_columns", DEFAULT_REQUIRED_METADATA_COLUMNS)
            metadata_columns = dataset.get("metadata_columns", [])
            missing_meta = [key for key in required_meta if key not in metadata_columns]
            if missing_meta:
                errors.append(f"Dataset metadata missing columns: {', '.join(missing_meta)}")

    slurm = config.get("slurm", {})
    if slurm.get("enabled") and not slurm.get("conda_env"):
        warnings.append("slurm.enabled is true but slurm.conda_env is missing.")

    invalid = [stage for stage in stages if stage not in ALLOWED_STAGES]
    if invalid:
        errors.append(f"Invalid stages: {', '.join(invalid)}")
    if not stages:
        errors.append("stages must include at least one stage.")

    post_nmf_mode = config.get("post_nmf_mode", "papermill")
    if post_nmf_mode not in ("papermill", "python"):
        errors.append("post_nmf_mode must be 'papermill' or 'python'.")

    if "post_nmf" in stages:
        if post_nmf_mode == "python":
            script_path = config.get("post_nmf_script_path")
            if not script_path:
                errors.append("post_nmf_script_path is required when post_nmf_mode=python.")
            elif check_paths and not resolve_repo_path(script_path).exists():
                errors.append(f"Post-NMF script not found: {script_path}")
        else:
            notebook_path = config.get("post_nmf_notebook_path", DEFAULT_POST_NMF_NOTEBOOK)
            if check_paths and not resolve_repo_path(notebook_path).exists():
                errors.append(f"Post-NMF notebook not found: {notebook_path}")

    rcausal_mode = config.get("rcausal_mode", "papermill")
    if rcausal_mode not in ("papermill", "python"):
        errors.append("rcausal_mode must be 'papermill' or 'python'.")

    if "rcausal_mgm" in stages:
        if rcausal_mode == "python":
            script_path = config.get("rcausal_script_path")
            if not script_path:
                errors.append("rcausal_script_path is required when rcausal_mode=python.")
            elif check_paths and not resolve_repo_path(script_path).exists():
                errors.append(f"RCausalMGM script not found: {script_path}")
        else:
            notebook_path = config.get("rcausal_notebook_path", DEFAULT_RCAUSAL_NOTEBOOK)
            if check_paths and not resolve_repo_path(notebook_path).exists():
                errors.append(f"RCausalMGM notebook not found: {notebook_path}")

    if "mlp" in stages:
        script_path = config.get("mlp_script_path", DEFAULT_MLP_SCRIPT)
        if check_paths and not resolve_repo_path(script_path).exists():
            errors.append(f"MLP script not found: {script_path}")

    _validate_stage_data_contract(stages, config, check_paths, errors, warnings, checks)

    if check_paths and config.get("check_join_keys", True):
        join_delimiter = config.get("join_key_delimiter", "__")
        max_missing_fraction = float(config.get("max_missing_fraction", 0.0))
        max_missing_count = int(config.get("max_missing_count", 0))
        join_strategy = config.get("join_key_strategy", "auto")
        try:
            if join_key_result:
                checks["join_keys"] = join_key_result
                if join_key_result.get("status") == "missing_deps":
                    if allow_join_fallback:
                        warnings.append("Join-key validation skipped due to missing dependencies.")
                    else:
                        errors.append("Join-key validation skipped due to missing dependencies.")
                else:
                    _apply_join_key_thresholds(
                        join_key_result, max_missing_fraction, max_missing_count, errors, warnings
                    )
                return errors, warnings, checks
            h5ad_value = config.get("cosmx_h5ad_path")
            metadata_value = config.get("cell_metadata_path")
            if not h5ad_value or not metadata_value:
                raise RuntimeError("cosmx_h5ad_path and cell_metadata_path are required for join-key validation.")
            h5ad_path = Path(h5ad_value)
            metadata_path = Path(metadata_value)
            obs = _read_h5ad_obs(h5ad_path)
            obs_columns = list(obs.columns)
            metadata_columns = _read_metadata_header(metadata_path)
            if join_strategy == "auto":
                join_strategy = _resolve_join_strategy(obs_columns, metadata_columns)
            if join_strategy == "unique_cell_id":
                meta_frame = _read_metadata_columns(metadata_path, ["unique_cell_id"])
            else:
                meta_frame = _read_metadata_columns(metadata_path, ["fov", "cell_ID"])
            obs_keys, strategy_used = _build_join_keys(obs, join_strategy, join_delimiter)
            meta_keys, _ = _build_join_keys(meta_frame, join_strategy, join_delimiter)
            obs_set = set(obs_keys)
            meta_set = set(meta_keys)
            matched = len(obs_set & meta_set)
            missing = len(obs_set - meta_set)
            extra = len(meta_set - obs_set)
            obs_total = len(obs_set)
            meta_total = len(meta_set)
            missing_fraction = (missing / obs_total) if obs_total else 0.0
            extra_fraction = (extra / meta_total) if meta_total else 0.0
            checks["join_keys"] = {
                "strategy": strategy_used,
                "obs_total": obs_total,
                "metadata_total": meta_total,
                "matched": matched,
                "missing": missing,
                "extra": extra,
                "missing_fraction": round(missing_fraction, 6),
                "extra_fraction": round(extra_fraction, 6),
            }
            _apply_join_key_thresholds(
                checks["join_keys"], max_missing_fraction, max_missing_count, errors, warnings
            )
        except DependencyMissingError as exc:
            checks["join_keys"] = {"status": "missing_deps", "reason": str(exc)}
            if allow_join_fallback:
                warnings.append("Join-key validation skipped due to missing dependencies.")
            else:
                errors.append(f"Join-key validation error: {exc}")
        except Exception as exc:
            errors.append(f"Join-key validation error: {exc}")

    return errors, warnings, checks
