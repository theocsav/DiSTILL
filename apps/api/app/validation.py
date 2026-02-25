import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import get_dataset
from .settings import ARTIFACT_ROOTS, REPO_ROOT
from .ssh_exec import is_ssh_backend, remote_path_exists, remote_path_readable
from .storage import enforce_allowed_path


REQUIRED_KEYS = ("cosmx_h5ad_path", "reference_h5ad_path", "cell_metadata_path")
PATH_KEYS = REQUIRED_KEYS + ("ref_model_dir",)
ALLOWED_STAGES = ("cell2loc_nmf", "post_nmf", "rcausal_mgm", "mlp", "report")
DEFAULT_POST_NMF_NOTEBOOK = "pipeline_assets/IBD_Post_NMF_Analysis.ipynb"
DEFAULT_RCAUSAL_NOTEBOOK = "pipeline_assets/IBD_RCausalMGM_Preparation.ipynb"
DEFAULT_MLP_SCRIPT = "pipeline_assets/IBD_MLP_44Features.py"
DEFAULT_REQUIRED_OBS_KEYS = ("fov", "cell_ID", "patient", "disease_status")
DEFAULT_REQUIRED_METADATA_COLUMNS = ("CenterX_global_px", "CenterY_global_px", "fov", "cell_ID")


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
    except ImportError as exc:
        raise DependencyMissingError("anndata is required for join-key validation.") from exc
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
    checks: Dict[str, Any] = {"exists": {}, "roots": {}, "permissions": {}}

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
            return

        exists_result = remote_path_exists(value)
        exists = bool(exists_result)
        checks["exists"][key] = exists
        if not exists:
            errors.append(f"Path does not exist: {key} -> {value}")
            checks["permissions"][key] = "skipped"
            return

        if is_ssh_backend():
            readable_result = remote_path_readable(value)
            readable = bool(readable_result)
        else:
            readable = os.access(path, os.R_OK)
        checks["permissions"][key] = readable
        if not readable:
            errors.append(f"Path not readable: {key} -> {value}")

    for key in REQUIRED_KEYS:
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

    stages = config.get("stages") or ["cell2loc_nmf"]
    if isinstance(stages, str):
        stages = [item.strip() for item in stages.split(",") if item.strip()]
    if not isinstance(stages, list):
        errors.append("stages must be a list of stage names.")
        stages = []
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
