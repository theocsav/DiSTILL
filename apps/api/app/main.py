import json
import logging
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .auth import (
    authenticate,
    canonical_identifier,
    clear_session_cookie,
    create_user,
    create_session,
    create_progress_token,
    ensure_csrf_cookie,
    require_csrf,
    require_session,
    set_session_cookie,
    verify_progress_token,
)
from .db import (
    claim_next_run,
    complete_claim,
    create_run,
    enqueue_run,
    append_run_message,
    fetch_queue_item,
    fetch_run,
    init_db,
    list_runs,
    release_claim,
    update_run,
)
from .preflight_cache import build_cache_key, get_cached_join_result, set_cached_join_result
from .registry import (
    dataset_manifest_hash,
    delete_dataset,
    get_dataset,
    list_datasets,
    list_presets,
    update_dataset,
    upsert_dataset,
)
from .runner import (
    PIPELINE_RUNNER,
    apply_dataset_registry,
    load_preset,
    prepare_run,
    prepare_run_bundle,
    resolve_run_paths,
)
from .schemas import (
    DryRunRequest,
    DatasetUpdateRequest,
    DryRunResponse,
    LoginRequest,
    LoginResponse,
    UserCreateRequest,
    UserCreateResponse,
    PublicRunProgressResponse,
    PreflightRequest,
    PreflightResponse,
    RunCreate,
    RunRerun,
    RunResponse,
    ShareRunLinkRequest,
    ShareRunLinkResponse,
    UploadFinalizeDatasetRequest,
    UploadInitRequest,
    UploadStatusResponse,
)
from .preflight_runner import run_slurm_preflight
from .upload_store import (
    append_chunk,
    cleanup_stale_uploads,
    complete_upload,
    get_status as get_upload_status,
    init_upload,
)
from .settings import (
    ALLOWED_ORIGINS,
    ARTIFACT_ROOTS,
    PREFLIGHT_CHECK_PATHS,
    QUEUE_ENABLED,
    RUNS_DIR,
    DISK_WARN_FREE_GB,
    DISK_WARN_PERCENT,
    DATA_UPLOADS_DIR,
    RUN_RETENTION_DAYS,
    PUBLIC_PROGRESS_TTL_HOURS,
    UPLOAD_CLEANUP_ENABLED,
    UPLOAD_CLEANUP_INTERVAL_SECONDS,
    UPLOAD_SESSION_TTL_HOURS,
    PREFLIGHT_SLURM_FALLBACK,
    PREFLIGHT_CACHE_TTL_SECONDS,
    WORKER_ENABLED,
    QUEUE_CLAIM_LEASE_SECONDS,
    QUEUE_POLLER_TOKEN,
    QUEUE_REMOTE_RUNS_DIR,
    SYNCED_ARTIFACT_RETENTION_DAYS,
    SYNCED_ARTIFACT_MAX_TOTAL_GB,
    validate_settings,
)
from .slurm import cancel_job
from .logging import configure_logging
from .worker import loop as worker_loop
from .storage import enforce_allowed_path, list_artifacts, safe_join
from .synced_artifacts import (
    cleanup_synced_artifacts,
    find_synced_log,
    has_synced_artifacts,
    read_sync_manifest,
    replace_synced_artifacts,
    synced_root,
)
from .validation import validate_config
from .ssh_exec import run_command, remote_path_exists

logger = logging.getLogger(__name__)
VALID_RUN_STATUSES = {"created", "queued", "prepared", "submitted", "running", "succeeded", "failed", "canceled", "error", "unknown"}
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "canceled", "error"}


def require_queue_poller(x_queue_token: Optional[str] = Header(default=None, alias="X-Queue-Token")) -> None:
    if not QUEUE_POLLER_TOKEN:
        raise HTTPException(status_code=503, detail="Queue poller token is not configured.")
    if not x_queue_token or not secrets.compare_digest(x_queue_token, QUEUE_POLLER_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid queue poller token")


def maintenance_loop() -> None:
    while True:
        try:
            result = cleanup_stale_uploads(UPLOAD_SESSION_TTL_HOURS)
            if result.get("removed_sessions", 0) > 0:
                logger.info("Upload cleanup removed %s stale sessions", result["removed_sessions"])
            sync_result = cleanup_synced_artifacts(SYNCED_ARTIFACT_RETENTION_DAYS, SYNCED_ARTIFACT_MAX_TOTAL_GB)
            if sync_result.get("removed_dirs", 0) > 0:
                logger.info(
                    "Synced artifact cleanup removed %s dirs (%s bytes)",
                    sync_result["removed_dirs"],
                    sync_result["removed_bytes"],
                )
        except Exception as exc:
            logger.exception("Maintenance loop failed: %s", exc)
        time.sleep(UPLOAD_CLEANUP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    validate_settings()
    init_db()
    if WORKER_ENABLED:
        thread = threading.Thread(target=worker_loop, daemon=True)
        thread.start()
    if UPLOAD_CLEANUP_ENABLED:
        cleanup_thread = threading.Thread(target=maintenance_loop, daemon=True)
        cleanup_thread.start()
    yield


app = FastAPI(title="NicheRunner API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "disk": _disk_usage_report()}


@app.get("/health/hpg", dependencies=[Depends(require_session)])
def hpg_health() -> dict:
    checks = {
        "sbatch": _check_command("sbatch"),
        "squeue": _check_command("squeue"),
        "sacct": _check_command("sacct"),
        "runs_dir": _check_path(RUNS_DIR),
        "artifact_roots": _check_artifact_roots(),
    }
    ok = all(value is True for value in checks.values())
    return {"ok": ok, "checks": checks}


@app.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest, request: Request, response: Response) -> dict:
    identifier = canonical_identifier(payload.username)
    if not authenticate(identifier, payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_session(identifier)
    set_session_cookie(response, token)
    csrf_token = ensure_csrf_cookie(request, response, rotate=True)
    return {"username": identifier, "csrf_token": csrf_token}


@app.post("/auth/logout", dependencies=[Depends(require_session), Depends(require_csrf)])
def logout(response: Response) -> dict:
    clear_session_cookie(response)
    return {"status": "ok"}


@app.get("/auth/me")
def whoami(request: Request, response: Response, username: str = Depends(require_session)) -> dict:
    csrf_token = ensure_csrf_cookie(request, response)
    return {"username": username, "csrf_token": csrf_token}


@app.get("/auth/csrf", dependencies=[Depends(require_session)])
def csrf_token(request: Request, response: Response) -> dict:
    csrf_token_value = ensure_csrf_cookie(request, response)
    return {"csrf_token": csrf_token_value}


@app.post(
    "/auth/users",
    response_model=UserCreateResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def create_auth_user(payload: UserCreateRequest, username: str = Depends(require_session)) -> dict:
    try:
        return create_user(payload.username, payload.password, username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/datasets", dependencies=[Depends(require_session)])
def get_datasets(
    organ: Optional[str] = Query(default=None),
    platform: Optional[str] = Query(default=None),
    preset_id: Optional[str] = Query(default=None),
) -> list[dict]:
    datasets = list_datasets()
    if preset_id:
        presets = list_presets()
        preset = next((item for item in presets if item.get("id") == preset_id), None)
        if preset:
            organ = organ or preset.get("organ")
            platform = platform or preset.get("platform")
    if organ:
        datasets = [item for item in datasets if item.get("organ") == organ]
    if platform:
        datasets = [item for item in datasets if item.get("platform") == platform]
    if preset_id:
        datasets = [
            item
            for item in datasets
            if item.get("recommended_preset") == preset_id
            or (not item.get("recommended_preset"))
        ]
    return datasets


@app.get("/datasets/public")
def get_public_datasets() -> list[dict]:
    datasets = list_datasets()
    return [item for item in datasets if bool(item.get("public"))]


@app.patch(
    "/datasets/{dataset_id}",
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def patch_dataset(
    dataset_id: str,
    payload: DatasetUpdateRequest,
    username: str = Depends(require_session),
) -> dict:
    existing = get_dataset(dataset_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Dataset not found")
    updates = payload.model_dump(exclude_none=True)
    updates["updated_by"] = username
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    updated = update_dataset(dataset_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"ok": True, "dataset": updated}


@app.delete(
    "/datasets/{dataset_id}",
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def remove_dataset(dataset_id: str, username: str = Depends(require_session)) -> dict:
    existing = get_dataset(dataset_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Dataset not found")
    deleted = delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"ok": True, "deleted_id": dataset_id, "deleted_by": username}


@app.post(
    "/uploads/init",
    response_model=UploadStatusResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def init_upload_endpoint(payload: UploadInitRequest, username: str = Depends(require_session)) -> dict:
    try:
        return init_upload(
            username=username,
            dataset_id=payload.dataset_id,
            file_role=payload.file_role,
            file_name=payload.file_name,
            total_size=payload.total_size,
            content_type=payload.content_type,
            expected_sha256=payload.expected_sha256,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/uploads/{upload_id}/status", response_model=UploadStatusResponse, dependencies=[Depends(require_session)])
def upload_status_endpoint(upload_id: str, username: str = Depends(require_session)) -> dict:
    try:
        return get_upload_status(upload_id, username)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put(
    "/uploads/{upload_id}/chunk",
    response_model=UploadStatusResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
async def upload_chunk_endpoint(
    upload_id: str,
    request: Request,
    offset: int = Query(..., ge=0),
    username: str = Depends(require_session),
) -> dict:
    data = await request.body()
    try:
        return append_chunk(upload_id, username, offset, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post(
    "/uploads/{upload_id}/complete",
    response_model=UploadStatusResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def complete_upload_endpoint(upload_id: str, username: str = Depends(require_session)) -> dict:
    try:
        return complete_upload(upload_id, username)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/datasets/upload", dependencies=[Depends(require_csrf)])
async def upload_dataset(
    username: str = Depends(require_session),
    dataset_id: str = Form(...),
    label: str = Form(...),
    organ: str = Form(...),
    platform: str = Form(...),
    notes: str = Form(default=""),
    recommended_preset: str = Form(default=""),
    staged_file: UploadFile = File(...),
    cell_metadata_file: UploadFile = File(...),
    reference_file: UploadFile | None = File(default=None),
) -> dict:
    safe_id = dataset_id.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", safe_id):
        raise HTTPException(
            status_code=400,
            detail="dataset_id must match [a-z0-9][a-z0-9._-]{2,63}",
        )
    safe_user = re.sub(r"[^a-z0-9._-]+", "_", username.strip().lower()).strip("._-")
    if not safe_user:
        raise HTTPException(status_code=400, detail="Invalid authenticated username for upload path.")

    base_dir = (DATA_UPLOADS_DIR / safe_user / safe_id).resolve()
    enforce_allowed_path(base_dir, ARTIFACT_ROOTS)
    base_dir.mkdir(parents=True, exist_ok=True)

    async def _save(upload: UploadFile, target_name: str | None = None) -> str:
        file_name = Path(target_name or upload.filename or "upload.bin").name
        destination = (base_dir / file_name).resolve()
        enforce_allowed_path(destination, ARTIFACT_ROOTS)
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        await upload.close()
        return str(destination)

    staged_path = await _save(staged_file)
    metadata_path = await _save(cell_metadata_file)
    reference_path = await _save(reference_file) if reference_file else ""

    dataset = {
        "id": safe_id,
        "label": label.strip() or safe_id,
        "organ": organ.strip(),
        "platform": platform.strip(),
        "staged_path": staged_path,
        "cell_metadata_path": metadata_path,
        "reference_h5ad_path": reference_path or None,
        "recommended_preset": recommended_preset.strip() or None,
        "notes": notes.strip() or None,
        "public": True,
        "uploaded_by": username,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": username,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "web_upload",
        "upload_root": str(base_dir),
        "checksums": {},
    }
    upsert_dataset(dataset)
    return {"ok": True, "dataset": dataset}


@app.post(
    "/datasets/upload/finalize",
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def finalize_uploaded_dataset(
    payload: UploadFinalizeDatasetRequest,
    username: str = Depends(require_session),
) -> dict:
    try:
        staged = get_upload_status(payload.staged_upload_id, username)
        metadata = get_upload_status(payload.cell_metadata_upload_id, username)
        reference = (
            get_upload_status(payload.reference_upload_id, username) if payload.reference_upload_id else None
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uploads = [staged, metadata] + ([reference] if reference else [])
    for item in uploads:
        if not item:
            continue
        if not item.get("completed"):
            raise HTTPException(status_code=400, detail=f"Upload not complete: {item.get('upload_id')}")
        if item.get("dataset_id") != payload.dataset_id.strip().lower():
            raise HTTPException(status_code=400, detail=f"Dataset mismatch in upload: {item.get('upload_id')}")

    if staged.get("file_role") != "staged":
        raise HTTPException(status_code=400, detail="staged_upload_id must reference role 'staged'.")
    if metadata.get("file_role") != "metadata":
        raise HTTPException(status_code=400, detail="cell_metadata_upload_id must reference role 'metadata'.")
    if reference and reference.get("file_role") != "reference":
        raise HTTPException(status_code=400, detail="reference_upload_id must reference role 'reference'.")

    upload_root = str(Path(staged["final_path"]).resolve().parent)
    dataset_checksums = {
        "staged": staged.get("sha256"),
        "metadata": metadata.get("sha256"),
    }
    if reference:
        dataset_checksums["reference"] = reference.get("sha256")
    dataset = {
        "id": payload.dataset_id.strip().lower(),
        "label": payload.label.strip(),
        "organ": payload.organ.strip(),
        "platform": payload.platform.strip(),
        "staged_path": staged["final_path"],
        "cell_metadata_path": metadata["final_path"],
        "reference_h5ad_path": reference["final_path"] if reference else None,
        "recommended_preset": payload.recommended_preset.strip() if payload.recommended_preset else None,
        "notes": payload.notes.strip() if payload.notes else None,
        "public": payload.public,
        "uploaded_by": username,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": username,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "chunked_upload",
        "upload_root": upload_root,
        "checksums": dataset_checksums,
    }
    upsert_dataset(dataset)
    return {"ok": True, "dataset": dataset}


@app.get("/presets", dependencies=[Depends(require_session)])
def get_presets(
    organ: Optional[str] = Query(default=None),
    platform: Optional[str] = Query(default=None),
) -> list[dict]:
    presets = list_presets()
    if organ:
        presets = [item for item in presets if item.get("organ") == organ]
    if platform:
        presets = [item for item in presets if item.get("platform") == platform]
    return presets


@app.get("/runs", response_model=list[RunResponse], dependencies=[Depends(require_session)])
def get_runs() -> list[dict]:
    return list_runs()


@app.post("/queue/claim", dependencies=[Depends(require_queue_poller)])
def queue_claim(request: Request) -> dict:
    claimed = claim_next_run(QUEUE_CLAIM_LEASE_SECONDS)
    if not claimed:
        return {"ok": True, "job": None}

    run = fetch_run(int(claimed["run_id"]))
    if not run:
        release_claim(int(claimed["run_id"]), str(claimed["claim_id"]), state="error")
        return {"ok": False, "error": "Run not found for claimed queue item."}

    config_path = run.get("config_path")
    if not config_path:
        release_claim(int(claimed["run_id"]), str(claimed["claim_id"]), state="error")
        return {"ok": False, "error": "Run config_path is missing."}
    try:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        remote_run_dir = f"{QUEUE_REMOTE_RUNS_DIR.rstrip('/')}/{run['run_name']}" if QUEUE_REMOTE_RUNS_DIR else None
        prepared = prepare_run_bundle(run["run_name"], config, remote_run_dir=remote_run_dir)
        update_run(
            int(run["id"]),
            status="prepared",
            run_dir=remote_run_dir or prepared["run_dir"],
            output_dir=prepared["output_dir"],
            config_path=prepared["config_path"],
        )
    except Exception as exc:
        message = f"Failed to prepare run bundle: {type(exc).__name__}: {exc}"
        release_claim(int(claimed["run_id"]), str(claimed["claim_id"]), state="error")
        update_run(int(run["id"]), status="error", message=message)
        raise HTTPException(status_code=500, detail=message) from exc

    origin = str(request.base_url).rstrip("/")
    run_id = int(run["id"])
    claim_id = str(claimed["claim_id"])
    bundle_url = f"{origin}/queue/bundles/{run_id}?claim_id={claim_id}"
    return {
        "ok": True,
        "job": {
            "run_id": run_id,
            "run_name": run["run_name"],
            "claim_id": claim_id,
            "lease_expires_at": claimed["lease_expires_at"],
            "bundle_url": bundle_url,
            "submit_script": prepared["submit_script"],
        },
    }


@app.get("/queue/bundles/{run_id}", dependencies=[Depends(require_queue_poller)])
def queue_bundle(run_id: int, claim_id: str = Query(...)) -> FileResponse:
    queue_item = fetch_queue_item(run_id)
    if not queue_item:
        raise HTTPException(status_code=404, detail="Queue item not found.")
    if queue_item.get("state") != "claimed":
        raise HTTPException(status_code=409, detail="Queue item is not currently claimed.")
    if str(queue_item.get("claim_id") or "") != claim_id:
        raise HTTPException(status_code=403, detail="Claim ID mismatch.")

    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    config_path = run.get("config_path")
    if not config_path:
        raise HTTPException(status_code=404, detail="Run config path not found.")
    bundle_path = Path(str(config_path)).parent / "run_bundle.tar.gz"
    if not bundle_path.exists():
        raise HTTPException(status_code=404, detail="Run bundle not found.")
    return FileResponse(path=str(bundle_path), filename=f"run_{run_id}_bundle.tar.gz")


@app.post("/queue/report-submission", dependencies=[Depends(require_queue_poller)])
def queue_report_submission(
    run_id: int = Form(...),
    claim_id: str = Form(...),
    slurm_job_id: str = Form(...),
    message: str = Form(default="Submitted by HPG poller"),
) -> dict:
    ok = complete_claim(run_id, claim_id)
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    if not ok:
        if str(run.get("job_id") or "") == slurm_job_id and run.get("status") in {"submitted", "running", "succeeded", "failed", "canceled", "error", "unknown"}:
            if message:
                append_run_message(run_id, f"Duplicate submission callback acknowledged: {message}")
            return {"ok": True, "idempotent": True}
        raise HTTPException(status_code=409, detail="Claim is invalid or expired.")
    update_run(run_id, status="submitted", job_id=slurm_job_id, message=message)
    if run and run.get("config_path"):
        bundle_path = Path(str(run["config_path"])).parent / "run_bundle.tar.gz"
        bundle_path.unlink(missing_ok=True)
    return {"ok": True}


@app.post("/queue/report-status", dependencies=[Depends(require_queue_poller)])
def queue_report_status(
    run_id: int = Form(...),
    status: str = Form(...),
    slurm_state: str = Form(default=""),
    slurm_reason: str = Form(default=""),
    slurm_elapsed: str = Form(default=""),
    started_at: str = Form(default=""),
    finished_at: str = Form(default=""),
    message: str = Form(default=""),
) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    if status not in VALID_RUN_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    update_fields: dict = {"status": status}
    if slurm_state:
        update_fields["slurm_state"] = slurm_state
    if slurm_reason:
        update_fields["slurm_reason"] = slurm_reason
    if slurm_elapsed:
        update_fields["slurm_elapsed"] = slurm_elapsed
    if started_at:
        update_fields["started_at"] = started_at
    if finished_at:
        update_fields["finished_at"] = finished_at
    update_run(run_id, **update_fields)
    if message:
        append_run_message(run_id, message)
    return {"ok": True}


@app.post("/queue/report-artifacts", dependencies=[Depends(require_queue_poller)])
async def queue_report_artifacts(
    run_id: int = Form(...),
    manifest_json: str = Form(default=""),
    paths: list[str] = Form(default=[]),
    files: list[UploadFile] = File(default=[]),
) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    if not files:
        raise HTTPException(status_code=400, detail="At least one artifact file is required.")
    try:
        items = await replace_synced_artifacts(run_id, files, paths, manifest_json=manifest_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sync_info = read_sync_manifest(run_id) or {}
    if run.get("status") in TERMINAL_RUN_STATUSES:
        append_run_message(run_id, f"Synced {len(items)} artifacts to API storage.")
    return {
        "ok": True,
        "root": str(synced_root(run_id)),
        "count": len(items),
        "items": items,
        "sync_manifest": sync_info,
    }


@app.get("/queue/active", dependencies=[Depends(require_queue_poller)])
def queue_active_runs() -> dict:
    active = []
    for run in list_runs():
        if run.get("job_id") and run.get("status") in {"submitted", "running", "queued", "unknown"}:
            active.append(
                {
                    "run_id": run["id"],
                    "run_name": run["run_name"],
                    "job_id": run.get("job_id"),
                    "status": run.get("status"),
                    "output_dir": run.get("output_dir"),
                }
            )
    return {"ok": True, "items": active}


@app.get("/runs/{run_id}", response_model=RunResponse, dependencies=[Depends(require_session)])
def get_run(run_id: int) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post(
    "/runs/{run_id}/share",
    response_model=ShareRunLinkResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def create_share_link(
    run_id: int,
    payload: ShareRunLinkRequest,
    request: Request,
) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    ttl_hours = payload.expires_hours or PUBLIC_PROGRESS_TTL_HOURS
    token = create_progress_token(run_id, ttl_hours)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)
    origin = str(request.base_url).rstrip("/")
    return {
        "run_id": run_id,
        "token": token,
        "url": f"{origin}/progress/{token}",
        "expires_at": expires_at.isoformat(),
    }


@app.get("/public/runs/progress", response_model=PublicRunProgressResponse)
def public_run_progress(token: str = Query(..., min_length=16)) -> dict:
    payload = verify_progress_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    run = fetch_run(int(payload["run_id"]))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    fields = (
        "id",
        "run_name",
        "status",
        "stage",
        "job_id",
        "slurm_state",
        "slurm_reason",
        "slurm_exit_code",
        "slurm_exit_signal",
        "slurm_elapsed",
        "submitted_at",
        "started_at",
        "finished_at",
        "message",
        "created_at",
        "updated_at",
    )
    return {key: run.get(key) for key in fields}


def _validate_config_payload(
    config: dict,
    check_paths: bool,
    preset_path: Optional[str] = None,
) -> tuple[list[str], list[str], dict]:
    dataset_id = config.get("dataset_id")
    preset_id = config.get("preset_id") or config.get("id")
    if preset_path:
        preset_id = Path(preset_path).stem

    cached_join = None
    cache_key = None
    if dataset_id:
        dataset = get_dataset(dataset_id)
        if dataset:
            manifest_hash = dataset_manifest_hash(dataset)
            cache_key = build_cache_key(dataset_id, manifest_hash, preset_id)
            cached_join = get_cached_join_result(cache_key)

    errors, warnings, checks = validate_config(
        config,
        check_paths=check_paths,
        allow_join_fallback=PREFLIGHT_SLURM_FALLBACK,
        join_key_result=cached_join,
    )
    join_status = checks.get("join_keys", {}).get("status")
    if join_status == "missing_deps" and PREFLIGHT_SLURM_FALLBACK and check_paths:
        slurm_result = run_slurm_preflight(config)
        if slurm_result.get("ok"):
            errors, warnings, checks = validate_config(
                config,
                check_paths=check_paths,
                allow_join_fallback=False,
                join_key_result=slurm_result["result"],
            )
        else:
            errors.append(slurm_result.get("error", "Preflight SLURM fallback failed."))

    join_result = checks.get("join_keys", {})
    if cache_key and join_result and "matched" in join_result:
        set_cached_join_result(cache_key, join_result, PREFLIGHT_CACHE_TTL_SECONDS)

    return errors, warnings, checks


@app.post(
    "/runs/preflight",
    response_model=PreflightResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def preflight(payload: PreflightRequest) -> dict:
    if not payload.config and not payload.preset_path:
        raise HTTPException(status_code=400, detail="Provide preset_path or config")

    config = {}
    if payload.preset_path:
        config = load_preset(payload.preset_path)
    if payload.config:
        config.update(payload.config.model_dump(exclude_none=True))

    try:
        config = apply_dataset_registry(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    errors, warnings, checks = _validate_config_payload(config, payload.check_paths, payload.preset_path)

    return {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}


@app.post(
    "/runs/dry-run",
    response_model=DryRunResponse,
    dependencies=[Depends(require_session), Depends(require_csrf)],
)
def dry_run(payload: DryRunRequest) -> dict:
    if not payload.config and not payload.preset_path:
        raise HTTPException(status_code=400, detail="Provide preset_path or config")

    config = {}
    if payload.preset_path:
        config = load_preset(payload.preset_path)
    if payload.config:
        config.update(payload.config.model_dump(exclude_none=True))

    try:
        config = apply_dataset_registry(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_name = payload.run_name or config.get("run_name")
    if not run_name:
        raise HTTPException(status_code=400, detail="run_name is required")
    config["run_name"] = run_name

    errors, warnings, checks = _validate_config_payload(config, payload.check_paths, payload.preset_path)
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "checks": checks}

    try:
        run_dir, output_dir, resolved_config = resolve_run_paths(run_name, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")

    cmd = [sys.executable, str(PIPELINE_RUNNER), "--config", str(config_path)]
    if payload.emit_sbatch:
        cmd.append("--emit-sbatch")
    result = subprocess.run(cmd, capture_output=True, text=True)

    resolved_config_path = run_dir / "config.resolved.json"
    resolved_config_data = None
    if resolved_config_path.exists():
        resolved_config_data = json.loads(resolved_config_path.read_text(encoding="utf-8"))

    response = {
        "ok": result.returncode == 0,
        "errors": [],
        "warnings": warnings,
        "checks": checks,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "resolved_config_path": str(resolved_config_path) if resolved_config_path.exists() else None,
        "resolved_config": resolved_config_data,
        "pipeline_stdout": result.stdout or "",
        "pipeline_stderr": result.stderr or "",
    }
    if result.returncode != 0:
        message = result.stderr or result.stdout or "Pipeline dry-run failed."
        response["errors"] = [message.strip()]
    return response


def _create_run_from_config(run_name: str, config: dict, submit: bool, queue: bool) -> dict:
    run_id = create_run(run_name, status="queued" if (queue or (submit and QUEUE_ENABLED)) else "created")
    try:
        run_dir, output_dir, resolved_config = resolve_run_paths(run_name, config)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(resolved_config, indent=2), encoding="utf-8")
        update_run(
            run_id,
            run_dir=str(run_dir),
            output_dir=str(output_dir),
            config_path=str(config_path),
        )

        errors, warnings, _checks = validate_config(
            resolved_config,
            check_paths=PREFLIGHT_CHECK_PATHS,
            allow_join_fallback=False,
        )
        if errors:
            raise HTTPException(status_code=400, detail={"errors": errors, "warnings": warnings})

        if queue or (submit and QUEUE_ENABLED):
            enqueue_run(run_id, submit=submit)
            update_run(run_id, status="queued")
        else:
            if submit:
                run_dir_str, output_dir_str, config_path_str, job_id = prepare_run(run_name, config, submit=True)
                update_run(
                    run_id,
                    status="submitted",
                    run_dir=run_dir_str,
                    output_dir=output_dir_str,
                    config_path=config_path_str,
                    job_id=job_id,
                )
            else:
                prepared = prepare_run_bundle(run_name, config)
                update_run(
                    run_id,
                    status="prepared",
                    run_dir=prepared["run_dir"],
                    output_dir=prepared["output_dir"],
                    config_path=prepared["config_path"],
                )
    except HTTPException as exc:
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        update_run(run_id, status="error", message=message)
        raise
    except Exception as exc:
        update_run(run_id, status="error", message=f"{type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Run creation failed") from exc

    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=500, detail="Run creation failed")
    return run


@app.post("/runs", response_model=RunResponse, dependencies=[Depends(require_session), Depends(require_csrf)])
def create_run_endpoint(payload: RunCreate) -> dict:
    if not payload.config and not payload.preset_path:
        raise HTTPException(status_code=400, detail="Provide preset_path or config")

    config = {}
    if payload.preset_path:
        config = load_preset(payload.preset_path)
    if payload.config:
        config.update(payload.config.model_dump(exclude_none=True))

    try:
        config = apply_dataset_registry(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_name = payload.run_name or config.get("run_name")
    if not run_name:
        raise HTTPException(status_code=400, detail="run_name is required")

    return _create_run_from_config(run_name, config, payload.submit, payload.queue)


def _strip_runtime_keys(config: dict) -> dict:
    remove_keys = {
        "run_dir",
        "config_path",
        "patched_script",
        "run_script",
        "report_script",
        "job_id",
    }
    return {key: value for key, value in config.items() if key not in remove_keys}


@app.post("/runs/{run_id}/rerun", response_model=RunResponse, dependencies=[Depends(require_session), Depends(require_csrf)])
def rerun_run(run_id: int, payload: RunRerun) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    config_path = run.get("config_path")
    if not config_path:
        raise HTTPException(status_code=400, detail="Run has no config_path")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = _strip_runtime_keys(config)
    run_dir = run.get("run_dir")
    output_dir = run.get("output_dir")
    if run_dir and output_dir:
        try:
            run_dir_path = Path(run_dir).resolve()
            output_dir_path = Path(output_dir).resolve()
            if run_dir_path == output_dir_path or run_dir_path in output_dir_path.parents:
                config.pop("output_dir", None)
        except OSError:
            pass
    config["run_name"] = payload.run_name
    report_title = config.get("report_title")
    if isinstance(report_title, str) and report_title.startswith("NicheRunner "):
        config["report_title"] = f"NicheRunner {payload.run_name}"
    try:
        config = apply_dataset_registry(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _create_run_from_config(payload.run_name, config, payload.submit, payload.queue)


def read_tail(path: Path, max_bytes: int = 65536) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        seek = max(size - max_bytes, 0)
        handle.seek(seek)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _resolve_output_root(run: dict) -> Path:
    output_dir = run.get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=400, detail="Run has no output_dir")
    output_path = Path(output_dir)
    enforce_allowed_path(output_path, ARTIFACT_ROOTS)
    return output_path


def _artifact_root_for_run(run: dict) -> Path:
    run_id = int(run["id"])
    if has_synced_artifacts(run_id):
        return synced_root(run_id)
    return _resolve_output_root(run)


@app.get("/runs/{run_id}/logs", dependencies=[Depends(require_session)])
def get_logs(run_id: int, path: Optional[str] = Query(default=None)) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run_id_int = int(run["id"])
    log_path = find_synced_log(run_id_int, path)
    if log_path is None:
        output_path = _resolve_output_root(run)
        if path:
            try:
                log_path = safe_join(output_path, path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid log path") from exc
        else:
            direct_candidates = list(output_path.glob("*.out")) + list(output_path.glob("*.err"))
            nested_logs = []
            logs_dir = output_path / "logs"
            if logs_dir.exists() and logs_dir.is_dir():
                nested_logs = list(logs_dir.glob("*.out")) + list(logs_dir.glob("*.err"))
            candidates = direct_candidates + nested_logs
            if not candidates:
                raise HTTPException(status_code=404, detail="No log files found")
            log_path = max(candidates, key=lambda p: p.stat().st_mtime)

    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    return {"path": str(log_path), "content": read_tail(log_path)}


@app.get("/runs/{run_id}/artifacts", dependencies=[Depends(require_session)])
def get_artifacts(run_id: int, path: Optional[str] = Query(default="")) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    base = _artifact_root_for_run(run)
    return {"items": list_artifacts(base, path)}


@app.get("/runs/{run_id}/artifact", dependencies=[Depends(require_session)])
def get_artifact(run_id: int, path: str = Query(...)) -> FileResponse:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    base = _artifact_root_for_run(run)
    try:
        file_path = safe_join(base, path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path") from exc
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(path=str(file_path), filename=file_path.name)


@app.get("/runs/{run_id}/summary", dependencies=[Depends(require_session)])
def get_run_summary(run_id: int) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    base = _artifact_root_for_run(run)
    try:
        summary_path = safe_join(base, "artifacts/run_summary.json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid summary path") from exc
    if not summary_path.exists() or not summary_path.is_file():
        raise HTTPException(status_code=404, detail="Run summary not found")
    return json.loads(summary_path.read_text(encoding="utf-8"))


@app.post("/runs/{run_id}/cancel", dependencies=[Depends(require_session), Depends(require_csrf)])
def cancel_run(run_id: int) -> dict:
    run = fetch_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    job_id = run.get("job_id")
    if job_id and cancel_job(job_id):
        update_run(run_id, status="canceled", message="Canceled via scancel")
        return {"status": "canceled"}
    update_run(run_id, status="canceled", message="Canceled without job_id")
    return {"status": "canceled"}


def _check_command(name: str) -> bool:
    try:
        result = run_command([name, "--version"])
        return result.returncode == 0
    except (FileNotFoundError, RuntimeError):
        return False


def _check_path(path: Path) -> bool:
    result = remote_path_exists(str(path))
    return bool(result)


def _check_artifact_roots() -> bool:
    return all(bool(remote_path_exists(str(root))) for root in ARTIFACT_ROOTS)


def _disk_usage_report() -> dict:
    roots = [RUNS_DIR] + ARTIFACT_ROOTS
    seen = set()
    items = []
    warnings = []
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        if not root.exists():
            items.append({"path": str(root), "exists": False})
            continue
        usage = shutil.disk_usage(root)
        total_gb = round(usage.total / (1024**3), 1)
        free_gb = round(usage.free / (1024**3), 1)
        percent_free = round((usage.free / usage.total) * 100, 1) if usage.total else 0.0
        warning = free_gb < DISK_WARN_FREE_GB or percent_free < DISK_WARN_PERCENT
        if warning:
            warnings.append(str(root))
        items.append(
            {
                "path": str(root),
                "exists": True,
                "total_gb": total_gb,
                "free_gb": free_gb,
                "percent_free": percent_free,
                "warning": warning,
            }
        )
    return {
        "roots": items,
        "warnings": warnings,
        "thresholds": {"free_gb": DISK_WARN_FREE_GB, "percent_free": DISK_WARN_PERCENT},
        "retention_days": RUN_RETENTION_DAYS,
    }
