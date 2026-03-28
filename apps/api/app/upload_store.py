import json
import re
import uuid
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Dict

from .settings import (
    ARTIFACT_ROOTS,
    DATA_UPLOADS_DIR,
    UPLOAD_ALLOWED_EXT_METADATA,
    UPLOAD_ALLOWED_EXT_REFERENCE,
    UPLOAD_ALLOWED_EXT_STAGED,
    UPLOAD_MAX_CONCURRENT_PER_USER,
    UPLOAD_MAX_SIZE_METADATA_GB,
    UPLOAD_MAX_SIZE_REFERENCE_GB,
    UPLOAD_MAX_SIZE_STAGED_GB,
)
from .storage import enforce_allowed_path

ALLOWED_FILE_ROLES = {"staged", "metadata", "reference", "nmf_artifact"}
ROLE_MAX_SIZE_BYTES = {
    "staged": int(UPLOAD_MAX_SIZE_STAGED_GB * 1024 * 1024 * 1024),
    "metadata": int(UPLOAD_MAX_SIZE_METADATA_GB * 1024 * 1024 * 1024),
    "reference": int(UPLOAD_MAX_SIZE_REFERENCE_GB * 1024 * 1024 * 1024),
    "nmf_artifact": int(UPLOAD_MAX_SIZE_STAGED_GB * 1024 * 1024 * 1024),
}
ROLE_ALLOWED_EXTS = {
    "staged": {item.strip().lower() for item in UPLOAD_ALLOWED_EXT_STAGED.split(",") if item.strip()},
    "metadata": {item.strip().lower() for item in UPLOAD_ALLOWED_EXT_METADATA.split(",") if item.strip()},
    "reference": {item.strip().lower() for item in UPLOAD_ALLOWED_EXT_REFERENCE.split(",") if item.strip()},
    "nmf_artifact": {item.strip().lower() for item in UPLOAD_ALLOWED_EXT_STAGED.split(",") if item.strip()},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sessions_dir() -> Path:
    path = (DATA_UPLOADS_DIR / ".sessions").resolve()
    enforce_allowed_path(path, ARTIFACT_ROOTS)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_path(upload_id: str) -> Path:
    safe_id = re.sub(r"[^a-f0-9-]+", "", upload_id.lower())
    if not safe_id:
        raise ValueError("Invalid upload id.")
    return (_sessions_dir() / f"{safe_id}.json").resolve()


def _sanitize_user(username: str) -> str:
    safe = re.sub(r"[^a-z0-9._-]+", "_", username.strip().lower()).strip("._-")
    if not safe:
        raise ValueError("Invalid authenticated username for upload path.")
    return safe


def _sanitize_dataset_id(dataset_id: str) -> str:
    safe = dataset_id.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,63}", safe):
        raise ValueError("dataset_id must match [a-z0-9][a-z0-9._-]{2,63}")
    return safe


def _sanitize_file_name(file_name: str) -> str:
    base = Path(file_name).name.strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    if not safe:
        raise ValueError("Invalid file name.")
    return safe


def _file_extensions(file_name: str) -> list[str]:
    lower = file_name.lower()
    suffixes = Path(lower).suffixes
    if lower.endswith(".csv.gz"):
        return [".csv.gz", ".gz", ".csv"]
    if lower.endswith(".tsv.gz"):
        return [".tsv.gz", ".gz", ".tsv"]
    return suffixes[-2:] + suffixes[-1:]


def _active_sessions_for_user(username: str) -> int:
    count = 0
    for session_file in _sessions_dir().glob("*.json"):
        try:
            manifest = json.loads(session_file.read_text(encoding="utf-8"))
            if manifest.get("uploaded_by") == username and not manifest.get("completed", False):
                count += 1
        except Exception:
            continue
    return count


def _load(upload_id: str) -> Dict[str, Any]:
    path = _session_path(upload_id)
    if not path.exists():
        raise ValueError("Upload session not found.")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(manifest: Dict[str, Any]) -> Dict[str, Any]:
    path = _session_path(str(manifest["upload_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _owned(manifest: Dict[str, Any], username: str) -> Dict[str, Any]:
    if manifest.get("uploaded_by") != username:
        raise ValueError("Upload session does not belong to this user.")
    return manifest


def _to_response(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "upload_id": manifest["upload_id"],
        "dataset_id": manifest["dataset_id"],
        "file_role": manifest["file_role"],
        "file_name": manifest["file_name"],
        "total_size": manifest["total_size"],
        "received_bytes": manifest["received_bytes"],
        "completed": manifest["completed"],
        "final_path": manifest.get("final_path"),
        "expected_sha256": manifest.get("expected_sha256"),
        "sha256": manifest.get("sha256"),
        "checksum_valid": manifest.get("checksum_valid"),
        "created_at": manifest["created_at"],
        "updated_at": manifest["updated_at"],
    }


def init_upload(
    *,
    username: str,
    dataset_id: str,
    file_role: str,
    file_name: str,
    total_size: int,
    content_type: str | None = None,
    expected_sha256: str | None = None,
) -> Dict[str, Any]:
    if file_role not in ALLOWED_FILE_ROLES:
        raise ValueError("Invalid file_role.")
    if total_size < 0:
        raise ValueError("total_size must be >= 0.")
    max_size = ROLE_MAX_SIZE_BYTES[file_role]
    if total_size > max_size:
        raise ValueError(f"File exceeds max size for role '{file_role}' ({max_size} bytes).")

    expected = (expected_sha256 or "").strip().lower()
    if expected and not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("expected_sha256 must be a 64-char lowercase hex string.")

    safe_user = _sanitize_user(username)
    safe_dataset = _sanitize_dataset_id(dataset_id)
    safe_name = _sanitize_file_name(file_name)
    allowed_exts = ROLE_ALLOWED_EXTS[file_role]
    if allowed_exts:
        exts = {ext.lower() for ext in _file_extensions(safe_name)}
        if not (exts & allowed_exts):
            raise ValueError(
                f"File extension not allowed for role '{file_role}'. Allowed: {', '.join(sorted(allowed_exts))}"
            )
    active_sessions = _active_sessions_for_user(username)
    if active_sessions >= UPLOAD_MAX_CONCURRENT_PER_USER:
        raise ValueError(
            f"Too many active uploads for user. Limit: {UPLOAD_MAX_CONCURRENT_PER_USER} concurrent sessions."
        )
    upload_id = uuid.uuid4().hex

    dataset_dir = (DATA_UPLOADS_DIR / safe_user / safe_dataset).resolve()
    enforce_allowed_path(dataset_dir, ARTIFACT_ROOTS)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    temp_path = (dataset_dir / f".{file_role}_{upload_id}.part").resolve()
    final_path = (dataset_dir / f"{file_role}__{safe_name}").resolve()
    enforce_allowed_path(temp_path, ARTIFACT_ROOTS)
    enforce_allowed_path(final_path, ARTIFACT_ROOTS)
    received_bytes = temp_path.stat().st_size if temp_path.exists() else 0
    now = _utc_now()
    manifest = {
        "upload_id": upload_id,
        "uploaded_by": username,
        "dataset_id": safe_dataset,
        "file_role": file_role,
        "file_name": safe_name,
        "total_size": total_size,
        "received_bytes": received_bytes,
        "content_type": content_type or "",
        "expected_sha256": expected or None,
        "sha256": None,
        "checksum_valid": None,
        "temp_path": str(temp_path),
        "final_path": str(final_path),
        "completed": False,
        "created_at": now,
        "updated_at": now,
    }
    return _to_response(_save(manifest))


def get_status(upload_id: str, username: str) -> Dict[str, Any]:
    manifest = _owned(_load(upload_id), username)
    temp_path = Path(manifest["temp_path"])
    if not manifest.get("completed", False) and temp_path.exists():
        actual = temp_path.stat().st_size
        if actual != int(manifest.get("received_bytes", 0)):
            manifest["received_bytes"] = actual
            manifest["updated_at"] = _utc_now()
            _save(manifest)
    return _to_response(manifest)


def append_chunk(upload_id: str, username: str, offset: int, data: bytes) -> Dict[str, Any]:
    manifest = _owned(_load(upload_id), username)
    if manifest.get("completed", False):
        raise ValueError("Upload session already completed.")
    temp_path = Path(manifest["temp_path"]).resolve()
    enforce_allowed_path(temp_path, ARTIFACT_ROOTS)
    current = temp_path.stat().st_size if temp_path.exists() else 0
    if offset != current:
        raise ValueError(f"Offset mismatch. expected={current} provided={offset}")
    if data:
        with temp_path.open("ab") as handle:
            handle.write(data)
    manifest["received_bytes"] = current + len(data)
    manifest["updated_at"] = _utc_now()
    if int(manifest["received_bytes"]) > int(manifest["total_size"]):
        raise ValueError("Received bytes exceed declared total size.")
    return _to_response(_save(manifest))


def complete_upload(upload_id: str, username: str) -> Dict[str, Any]:
    manifest = _owned(_load(upload_id), username)
    total = int(manifest["total_size"])
    received = int(manifest["received_bytes"])
    if received != total:
        raise ValueError(f"Cannot complete upload before all bytes are received ({received}/{total}).")
    temp_path = Path(manifest["temp_path"]).resolve()
    final_path = Path(manifest["final_path"]).resolve()
    enforce_allowed_path(temp_path, ARTIFACT_ROOTS)
    enforce_allowed_path(final_path, ARTIFACT_ROOTS)
    if not temp_path.exists():
        raise ValueError("Upload temp file not found.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists():
        final_path.unlink()
    hasher = hashlib.sha256()
    with temp_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    computed_sha256 = hasher.hexdigest()
    temp_path.rename(final_path)
    expected_sha256 = (manifest.get("expected_sha256") or "").strip().lower()
    checksum_valid = True
    if expected_sha256:
        checksum_valid = expected_sha256 == computed_sha256
    manifest["sha256"] = computed_sha256
    manifest["checksum_valid"] = checksum_valid
    if not checksum_valid:
        if final_path.exists():
            final_path.unlink()
        raise ValueError("Checksum validation failed for uploaded file.")
    manifest["completed"] = True
    manifest["updated_at"] = _utc_now()
    return _to_response(_save(manifest))


def cleanup_stale_uploads(ttl_hours: int) -> Dict[str, int]:
    sessions_dir = _sessions_dir()
    now = datetime.now(timezone.utc)
    removed_sessions = 0
    removed_temp_files = 0
    total_sessions = 0
    ttl_seconds = max(ttl_hours, 1) * 3600

    for session_file in sessions_dir.glob("*.json"):
        total_sessions += 1
        try:
            manifest = json.loads(session_file.read_text(encoding="utf-8"))
            updated_at = manifest.get("updated_at") or manifest.get("created_at")
            if not updated_at:
                raise ValueError("missing timestamps")
            updated_dt = datetime.fromisoformat(str(updated_at))
            age_seconds = (now - updated_dt).total_seconds()
            if age_seconds < ttl_seconds:
                continue
            temp_path = Path(str(manifest.get("temp_path", ""))).resolve()
            final_path = Path(str(manifest.get("final_path", ""))).resolve()
            for candidate in (temp_path,):
                if candidate and candidate.exists():
                    try:
                        enforce_allowed_path(candidate, ARTIFACT_ROOTS)
                        candidate.unlink()
                        removed_temp_files += 1
                    except Exception:
                        pass
            # If upload never completed and a final path somehow exists from partial runs, remove it.
            if not manifest.get("completed", False) and final_path and final_path.exists():
                try:
                    enforce_allowed_path(final_path, ARTIFACT_ROOTS)
                    final_path.unlink()
                except Exception:
                    pass
            session_file.unlink(missing_ok=True)
            removed_sessions += 1
        except Exception:
            # Leave unreadable/broken manifests in place for manual inspection.
            continue

    return {
        "total_sessions": total_sessions,
        "removed_sessions": removed_sessions,
        "removed_temp_files": removed_temp_files,
    }
