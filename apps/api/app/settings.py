import os
from pathlib import Path

from dotenv import load_dotenv

from .storage import enforce_allowed_path

# Load environment variables from .env file
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = Path(os.environ.get("RUNS_DIR", str(REPO_ROOT / "runs"))).resolve()
PRESETS_DIR = Path(os.environ.get("PRESETS_DIR", str(REPO_ROOT / "presets"))).resolve()
DATASETS_REGISTRY_PATH = Path(
    os.environ.get("DATASETS_REGISTRY_PATH", str(REPO_ROOT / "registries" / "datasets.json"))
).resolve()
USERS_REGISTRY_PATH = Path(
    os.environ.get("USERS_REGISTRY_PATH", str(REPO_ROOT / "registries" / "users.json"))
).resolve()
DB_PATH = os.environ.get("DB_PATH", str(REPO_ROOT / "runs.db"))
BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER", "admin")
BASIC_AUTH_PASS = os.environ.get("BASIC_AUTH_PASS", "admin")
AUTH_PASSWORD_HASH = os.environ.get("AUTH_PASSWORD_HASH")
AUTH_IDENTIFIER_DOMAIN = os.environ.get("AUTH_IDENTIFIER_DOMAIN", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "change-me")
SESSION_TTL_MINUTES = int(os.environ.get("SESSION_TTL_MINUTES", "480"))
PUBLIC_PROGRESS_TTL_HOURS = int(os.environ.get("PUBLIC_PROGRESS_TTL_HOURS", "168"))
COOKIE_NAME = os.environ.get("COOKIE_NAME", "sptx_session")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "false").lower() == "true"
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "lax")
CSRF_COOKIE_NAME = os.environ.get("CSRF_COOKIE_NAME", "sptx_csrf")
CSRF_HEADER_NAME = os.environ.get("CSRF_HEADER_NAME", "X-CSRF-Token")
ARTIFACT_ROOTS = [
    Path(value.strip()).resolve()
    for value in os.environ.get("ARTIFACT_ROOTS", str(RUNS_DIR)).split(",")
    if value.strip()
]
SYNCED_ARTIFACTS_DIR = Path(
    os.environ.get("SYNCED_ARTIFACTS_DIR", str(RUNS_DIR / "_synced_artifacts"))
).resolve()
WORKER_ENABLED = os.environ.get("WORKER_ENABLED", "false").lower() == "true"
WORKER_POLL_SECONDS = int(os.environ.get("WORKER_POLL_SECONDS", "10"))
QUEUE_ENABLED = os.environ.get("QUEUE_ENABLED", "true").lower() == "true"
PREFLIGHT_CHECK_PATHS = os.environ.get("PREFLIGHT_CHECK_PATHS", "true").lower() == "true"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
SLURM_BACKEND = os.environ.get("SLURM_BACKEND", "local").strip().lower()
SSH_HOST = os.environ.get("SSH_HOST", "").strip()
SSH_PORT = int(os.environ.get("SSH_PORT", "22"))
SSH_USER = os.environ.get("SSH_USER", "").strip()
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "").strip()
SSH_KNOWN_HOSTS = os.environ.get("SSH_KNOWN_HOSTS", "").strip()
SSH_STRICT_HOST_KEY_CHECKING = os.environ.get("SSH_STRICT_HOST_KEY_CHECKING", "yes").strip()
SSH_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("SSH_CONNECT_TIMEOUT_SECONDS", "10"))
SSH_REMOTE_RUNS_DIR = os.environ.get("SSH_REMOTE_RUNS_DIR", "").strip()
QUEUE_REMOTE_RUNS_DIR = os.environ.get("QUEUE_REMOTE_RUNS_DIR", SSH_REMOTE_RUNS_DIR).strip()
QUEUE_POLLER_TOKEN = os.environ.get("QUEUE_POLLER_TOKEN", "").strip()
QUEUE_CLAIM_LEASE_SECONDS = int(os.environ.get("QUEUE_CLAIM_LEASE_SECONDS", "600"))
RUN_RETENTION_DAYS = int(os.environ.get("RUN_RETENTION_DAYS", "30"))
CLEANUP_INTERVAL_SECONDS = int(os.environ.get("CLEANUP_INTERVAL_SECONDS", "3600"))
DISK_WARN_FREE_GB = int(os.environ.get("DISK_WARN_FREE_GB", "50"))
DISK_WARN_PERCENT = int(os.environ.get("DISK_WARN_PERCENT", "10"))
PREFLIGHT_SLURM_FALLBACK = os.environ.get("PREFLIGHT_SLURM_FALLBACK", "false").lower() == "true"
PREFLIGHT_SLURM_TIMEOUT_SECONDS = int(os.environ.get("PREFLIGHT_SLURM_TIMEOUT_SECONDS", "600"))
PREFLIGHT_SLURM_POLL_SECONDS = int(os.environ.get("PREFLIGHT_SLURM_POLL_SECONDS", "5"))
PREFLIGHT_CACHE_TTL_SECONDS = int(os.environ.get("PREFLIGHT_CACHE_TTL_SECONDS", "300"))
DATA_UPLOADS_DIR = Path(
    os.environ.get("DATA_UPLOADS_DIR", str(RUNS_DIR / "public_uploads" / "datasets"))
).resolve()
UPLOAD_SESSION_TTL_HOURS = int(os.environ.get("UPLOAD_SESSION_TTL_HOURS", "72"))
UPLOAD_CLEANUP_INTERVAL_SECONDS = int(os.environ.get("UPLOAD_CLEANUP_INTERVAL_SECONDS", "900"))
UPLOAD_CLEANUP_ENABLED = os.environ.get("UPLOAD_CLEANUP_ENABLED", "true").lower() == "true"
UPLOAD_MAX_CONCURRENT_PER_USER = int(os.environ.get("UPLOAD_MAX_CONCURRENT_PER_USER", "6"))
UPLOAD_MAX_SIZE_STAGED_GB = float(os.environ.get("UPLOAD_MAX_SIZE_STAGED_GB", "100"))
UPLOAD_MAX_SIZE_METADATA_GB = float(os.environ.get("UPLOAD_MAX_SIZE_METADATA_GB", "5"))
UPLOAD_MAX_SIZE_REFERENCE_GB = float(os.environ.get("UPLOAD_MAX_SIZE_REFERENCE_GB", "50"))
UPLOAD_ALLOWED_EXT_STAGED = os.environ.get("UPLOAD_ALLOWED_EXT_STAGED", ".h5ad")
UPLOAD_ALLOWED_EXT_METADATA = os.environ.get("UPLOAD_ALLOWED_EXT_METADATA", ".csv,.tsv,.gz")
UPLOAD_ALLOWED_EXT_REFERENCE = os.environ.get("UPLOAD_ALLOWED_EXT_REFERENCE", ".h5ad")
SYNCED_ARTIFACT_RETENTION_DAYS = int(os.environ.get("SYNCED_ARTIFACT_RETENTION_DAYS", "14"))
SYNCED_ARTIFACT_MAX_TOTAL_GB = float(os.environ.get("SYNCED_ARTIFACT_MAX_TOTAL_GB", "20"))


def validate_settings() -> None:
    if not SESSION_SECRET or SESSION_SECRET.strip() == "" or SESSION_SECRET == "change-me":
        raise RuntimeError("SESSION_SECRET must be set to a non-default value.")
    if not AUTH_PASSWORD_HASH:
        if not BASIC_AUTH_USER or not BASIC_AUTH_PASS:
            raise RuntimeError(
                "BASIC_AUTH_USER and BASIC_AUTH_PASS must be set when AUTH_PASSWORD_HASH is not provided."
            )
        if BASIC_AUTH_USER == "admin" and BASIC_AUTH_PASS == "admin":
            raise RuntimeError("BASIC_AUTH_* must be changed from the default or set AUTH_PASSWORD_HASH.")
    if not ARTIFACT_ROOTS:
        raise RuntimeError("ARTIFACT_ROOTS must include at least one path.")
    if not RUNS_DIR.is_absolute():
        raise RuntimeError("RUNS_DIR must be an absolute path.")
    if not DATA_UPLOADS_DIR.is_absolute():
        raise RuntimeError("DATA_UPLOADS_DIR must be an absolute path.")
    if not SYNCED_ARTIFACTS_DIR.is_absolute():
        raise RuntimeError("SYNCED_ARTIFACTS_DIR must be an absolute path.")
    if UPLOAD_SESSION_TTL_HOURS <= 0:
        raise RuntimeError("UPLOAD_SESSION_TTL_HOURS must be > 0.")
    if UPLOAD_CLEANUP_INTERVAL_SECONDS <= 0:
        raise RuntimeError("UPLOAD_CLEANUP_INTERVAL_SECONDS must be > 0.")
    if UPLOAD_MAX_CONCURRENT_PER_USER <= 0:
        raise RuntimeError("UPLOAD_MAX_CONCURRENT_PER_USER must be > 0.")
    if min(UPLOAD_MAX_SIZE_STAGED_GB, UPLOAD_MAX_SIZE_METADATA_GB, UPLOAD_MAX_SIZE_REFERENCE_GB) <= 0:
        raise RuntimeError("Upload max size limits must be > 0.")
    if SYNCED_ARTIFACT_RETENTION_DAYS < 0:
        raise RuntimeError("SYNCED_ARTIFACT_RETENTION_DAYS must be >= 0.")
    if SYNCED_ARTIFACT_MAX_TOTAL_GB <= 0:
        raise RuntimeError("SYNCED_ARTIFACT_MAX_TOTAL_GB must be > 0.")
    if SLURM_BACKEND not in {"local", "ssh"}:
        raise RuntimeError("SLURM_BACKEND must be 'local' or 'ssh'.")
    if SLURM_BACKEND == "ssh":
        if not SSH_HOST:
            raise RuntimeError("SSH_HOST is required when SLURM_BACKEND=ssh.")
        if not SSH_USER:
            raise RuntimeError("SSH_USER is required when SLURM_BACKEND=ssh.")
        if not SSH_REMOTE_RUNS_DIR:
            raise RuntimeError("SSH_REMOTE_RUNS_DIR is required when SLURM_BACKEND=ssh.")
    if WORKER_ENABLED and QUEUE_POLLER_TOKEN and SLURM_BACKEND != "ssh":
        raise RuntimeError("Disable WORKER_ENABLED when using the external HPG queue poller.")
    if QUEUE_CLAIM_LEASE_SECONDS < 30:
        raise RuntimeError("QUEUE_CLAIM_LEASE_SECONDS must be >= 30.")
    for root in ARTIFACT_ROOTS:
        if not root.is_absolute():
            raise RuntimeError("ARTIFACT_ROOTS must contain absolute paths.")
    enforce_allowed_path(RUNS_DIR, ARTIFACT_ROOTS)
    enforce_allowed_path(DATA_UPLOADS_DIR, ARTIFACT_ROOTS)
    enforce_allowed_path(SYNCED_ARTIFACTS_DIR, ARTIFACT_ROOTS)
