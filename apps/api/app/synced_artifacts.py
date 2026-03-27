import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import UploadFile

from .settings import SYNCED_ARTIFACTS_DIR
from .storage import list_artifacts, safe_join


def synced_root(run_id: int) -> Path:
    return (SYNCED_ARTIFACTS_DIR / str(run_id)).resolve()


def has_synced_artifacts(run_id: int) -> bool:
    root = synced_root(run_id)
    return root.exists() and root.is_dir()


def resolve_synced_path(run_id: int, relative_path: str) -> Path:
    return safe_join(synced_root(run_id), relative_path)


def read_sync_manifest(run_id: int) -> Optional[dict]:
    manifest_path = synced_root(run_id) / ".sync_manifest.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        return None


def synced_root_size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


async def replace_synced_artifacts(
    run_id: int,
    files: List[UploadFile],
    paths: List[str],
    manifest_json: str = "",
) -> List[Dict[str, str]]:
    if len(files) != len(paths):
        raise ValueError("paths and files must have the same length")

    root = synced_root(run_id)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    stored_items: List[Dict[str, str]] = []
    for upload, relative_path in zip(files, paths):
        rel = relative_path.strip().lstrip("/")
        if not rel:
            raise ValueError("Artifact path must not be empty")
        destination = safe_join(root, rel)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        await upload.close()
        stored_items.append(
            {
                "path": rel.replace("\\", "/"),
                "size": str(destination.stat().st_size),
            }
        )

    sync_manifest = {
        "run_id": run_id,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "items": sorted(stored_items, key=lambda item: item["path"]),
    }
    if manifest_json.strip():
        try:
            sync_manifest["poller_manifest"] = json.loads(manifest_json)
        except json.JSONDecodeError:
            sync_manifest["poller_manifest_raw"] = manifest_json

    (root / ".sync_manifest.json").write_text(json.dumps(sync_manifest, indent=2), encoding="utf-8")
    return sorted(stored_items, key=lambda item: item["path"])


def list_synced_artifacts(run_id: int, subpath: str = "") -> List[Dict[str, str]]:
    return list_artifacts(synced_root(run_id), subpath)


def find_synced_log(run_id: int, relative_path: Optional[str] = None) -> Optional[Path]:
    root = synced_root(run_id)
    if not root.exists():
        return None
    logs_dir = root / "logs"
    if relative_path:
        rel = relative_path.strip().lstrip("/")
        candidates = [rel]
        if not rel.startswith("logs/"):
            candidates.append(f"logs/{rel}")
        for candidate in candidates:
            path = safe_join(root, candidate)
            if path.exists() and path.is_file():
                return path
        return None
    if not logs_dir.exists() or not logs_dir.is_dir():
        return None
    candidates = sorted(list(logs_dir.glob("*.out")) + list(logs_dir.glob("*.err")), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def cleanup_synced_artifacts(retention_days: int, max_total_gb: float) -> Dict[str, int]:
    removed_dirs = 0
    removed_bytes = 0
    roots = [path for path in SYNCED_ARTIFACTS_DIR.iterdir()] if SYNCED_ARTIFACTS_DIR.exists() else []
    entries = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days) if retention_days > 0 else None

    for root in roots:
        if not root.is_dir():
            continue
        manifest = None
        manifest_path = root / ".sync_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                manifest = None
        synced_at_raw = (manifest or {}).get("synced_at") or datetime.fromtimestamp(root.stat().st_mtime, tz=timezone.utc).isoformat()
        try:
            synced_at = datetime.fromisoformat(str(synced_at_raw))
        except ValueError:
            synced_at = datetime.fromtimestamp(root.stat().st_mtime, tz=timezone.utc)
        size_bytes = synced_root_size_bytes(root)
        entries.append({"root": root, "synced_at": synced_at, "size_bytes": size_bytes})

    for entry in entries:
        if cutoff and entry["synced_at"] < cutoff:
            shutil.rmtree(entry["root"], ignore_errors=True)
            removed_dirs += 1
            removed_bytes += int(entry["size_bytes"])

    active_entries = []
    for entry in entries:
        if entry["root"].exists():
            active_entries.append(
                {
                    "root": entry["root"],
                    "synced_at": entry["synced_at"],
                    "size_bytes": synced_root_size_bytes(entry["root"]),
                }
            )

    max_total_bytes = int(max_total_gb * 1024 * 1024 * 1024)
    total_bytes = sum(int(entry["size_bytes"]) for entry in active_entries)
    if total_bytes > max_total_bytes:
        for entry in sorted(active_entries, key=lambda item: item["synced_at"]):
            if total_bytes <= max_total_bytes:
                break
            shutil.rmtree(entry["root"], ignore_errors=True)
            removed_dirs += 1
            removed_bytes += int(entry["size_bytes"])
            total_bytes -= int(entry["size_bytes"])

    return {
        "removed_dirs": removed_dirs,
        "removed_bytes": removed_bytes,
        "remaining_bytes": max(0, total_bytes),
    }
