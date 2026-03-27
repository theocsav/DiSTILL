from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SlurmConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: Optional[bool] = None
    job_name: Optional[str] = None
    time: Optional[str] = None
    mem: Optional[str] = None
    cpus_per_task: Optional[int] = None
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    mail_user: Optional[str] = None
    mail_type: Optional[str] = None
    conda_env: Optional[str] = None
    use_module_conda: Optional[bool] = None
    output: Optional[str] = None
    error: Optional[str] = None


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_name: Optional[str] = None
    preset_id: Optional[str] = None
    mode: Optional[str] = None
    stages: Optional[List[str]] = None

    dataset_id: Optional[str] = None
    cosmx_h5ad_path: Optional[str] = None
    reference_h5ad_path: Optional[str] = None
    cell_metadata_path: Optional[str] = None
    cosmx_with_nmf_path: Optional[str] = None

    run_dir: Optional[str] = None
    output_dir: Optional[str] = None
    template_path: Optional[str] = None
    ref_model_dir: Optional[str] = None
    ref_model_name: Optional[str] = None
    inf_aver_name: Optional[str] = None

    n_components: Optional[int] = None
    k: Optional[int] = None
    k_min: Optional[int] = None
    k_max: Optional[int] = None

    post_nmf_mode: Optional[str] = None
    post_nmf_notebook_path: Optional[str] = None
    post_nmf_script_path: Optional[str] = None
    post_nmf_parameters: Optional[Dict[str, Any]] = None
    post_nmf_args: Optional[List[str]] = None

    rcausal_mode: Optional[str] = None
    rcausal_notebook_path: Optional[str] = None
    rcausal_script_path: Optional[str] = None
    rcausal_parameters: Optional[Dict[str, Any]] = None
    rcausal_args: Optional[List[str]] = None
    rcausal_output_dir: Optional[str] = None
    rcausal_h5ad_path: Optional[str] = None
    rcausal_niche_h5ad_path: Optional[str] = None
    rcausal_neighborhood_h5ad_path: Optional[str] = None

    mlp_script_path: Optional[str] = None
    mlp_args: Optional[List[str]] = None

    report_title: Optional[str] = None
    report_notes: Optional[str] = None
    report_script: Optional[str] = None

    slurm: Optional[SlurmConfig] = None
    preflight_slurm: Optional[SlurmConfig] = None

    check_join_keys: Optional[bool] = None
    join_key_strategy: Optional[str] = None
    join_key_delimiter: Optional[str] = None
    max_missing_fraction: Optional[float] = None
    max_missing_count: Optional[int] = None
    required_obs_keys: Optional[List[str]] = None
    required_metadata_columns: Optional[List[str]] = None
    require_raw_counts: Optional[bool] = None


class RunCreate(BaseModel):
    run_name: str = Field(..., min_length=1)
    preset_path: Optional[str] = None
    config: Optional[RunSpec] = None
    submit: bool = False
    queue: bool = False


class RunRerun(BaseModel):
    run_name: str = Field(..., min_length=1)
    submit: bool = False
    queue: bool = False


class RunState(BaseModel):
    id: int
    run_name: str
    status: str
    stage: Optional[str] = None
    run_dir: Optional[str] = None
    output_dir: Optional[str] = None
    config_path: Optional[str] = None
    job_id: Optional[str] = None
    slurm_state: Optional[str] = None
    slurm_reason: Optional[str] = None
    slurm_exit_code: Optional[int] = None
    slurm_exit_signal: Optional[int] = None
    slurm_elapsed: Optional[str] = None
    submitted_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    updated_at: str


class RunResponse(RunState):
    pass


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str
    csrf_token: Optional[str] = None


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=8, max_length=256)


class UserCreateResponse(BaseModel):
    username: str
    created_by: str
    created_at: str


class ShareRunLinkRequest(BaseModel):
    expires_hours: Optional[int] = Field(default=None, ge=1, le=24 * 30)


class ShareRunLinkResponse(BaseModel):
    run_id: int
    token: str
    url: str
    expires_at: str


class PublicRunProgressResponse(BaseModel):
    id: int
    run_name: str
    status: str
    stage: Optional[str] = None
    job_id: Optional[str] = None
    slurm_state: Optional[str] = None
    slurm_reason: Optional[str] = None
    slurm_exit_code: Optional[int] = None
    slurm_exit_signal: Optional[int] = None
    slurm_elapsed: Optional[str] = None
    submitted_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    message: Optional[str] = None
    created_at: str
    updated_at: str


class UploadInitRequest(BaseModel):
    dataset_id: str = Field(..., min_length=3, max_length=64)
    file_role: Literal["staged", "metadata", "reference"]
    file_name: str = Field(..., min_length=1, max_length=255)
    total_size: int = Field(..., ge=0)
    content_type: Optional[str] = None
    expected_sha256: Optional[str] = Field(default=None, min_length=64, max_length=64)


class UploadStatusResponse(BaseModel):
    upload_id: str
    dataset_id: str
    file_role: str
    file_name: str
    total_size: int
    received_bytes: int
    completed: bool
    final_path: Optional[str] = None
    expected_sha256: Optional[str] = None
    sha256: Optional[str] = None
    checksum_valid: Optional[bool] = None
    created_at: str
    updated_at: str


class UploadFinalizeDatasetRequest(BaseModel):
    dataset_id: str = Field(..., min_length=3, max_length=64)
    label: str = Field(..., min_length=1, max_length=128)
    organ: str = Field(..., min_length=1, max_length=64)
    platform: str = Field(..., min_length=1, max_length=64)
    notes: Optional[str] = None
    recommended_preset: Optional[str] = None
    staged_upload_id: str = Field(..., min_length=8)
    cell_metadata_upload_id: str = Field(..., min_length=8)
    reference_upload_id: Optional[str] = None
    public: bool = True


class DatasetUpdateRequest(BaseModel):
    label: Optional[str] = Field(default=None, min_length=1, max_length=128)
    notes: Optional[str] = Field(default=None, max_length=2048)
    public: Optional[bool] = None


class PreflightRequest(BaseModel):
    preset_path: Optional[str] = None
    config: Optional[RunSpec] = None
    check_paths: bool = True


class PreflightResponse(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    checks: Dict[str, Any] = Field(default_factory=dict)


class DryRunRequest(BaseModel):
    run_name: str = Field(..., min_length=1)
    preset_path: Optional[str] = None
    config: Optional[RunSpec] = None
    check_paths: bool = True
    emit_sbatch: bool = True


class DryRunResponse(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    checks: Dict[str, Any] = Field(default_factory=dict)
    run_dir: Optional[str] = None
    output_dir: Optional[str] = None
    config_path: Optional[str] = None
    resolved_config_path: Optional[str] = None
    resolved_config: Optional[Dict[str, Any]] = None
    pipeline_stdout: Optional[str] = None
    pipeline_stderr: Optional[str] = None
