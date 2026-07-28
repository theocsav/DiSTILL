import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, Response, status

from .settings import (
    ADMIN_USERS,
    AUTH_PASSWORD_HASH,
    AUTH_IDENTIFIER_DOMAIN,
    BASIC_AUTH_PASS,
    BASIC_AUTH_USER,
    COOKIE_NAME,
    COOKIE_SAMESITE,
    COOKIE_SECURE,
    CSRF_COOKIE_NAME,
    CSRF_HEADER_NAME,
    USERS_REGISTRY_PATH,
    SESSION_SECRET,
    SESSION_TTL_MINUTES,
)


# Serializes read-modify-write cycles against the users registry within this process.
_USERS_LOCK = threading.RLock()


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * ((4 - len(data) % 4) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload: bytes) -> str:
    digest = hmac.new(SESSION_SECRET.encode("utf-8"), payload, hashlib.sha256).digest()
    return _b64url_encode(digest)


def _hash_password(password: str, salt: str, iterations: int) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return _b64url_encode(dk)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_users() -> list[dict]:
    path = Path(USERS_REGISTRY_PATH)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        users = data.get("users", [])
        return users if isinstance(users, list) else []
    if isinstance(data, list):
        return data
    return []


def _save_users(users: list[dict]) -> None:
    payload = json.dumps({"users": users}, indent=2) + "\n"
    USERS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(USERS_REGISTRY_PATH.parent),
        prefix=f".{USERS_REGISTRY_PATH.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, USERS_REGISTRY_PATH)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _verify_password_hash(password: str, encoded_hash: str) -> bool:
    # Format: pbkdf2_sha256$iterations$salt$hash
    try:
        algo, iterations_str, salt, stored = encoded_hash.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(iterations_str)
    except ValueError:
        return False
    computed = _hash_password(password, salt, iterations)
    return secrets.compare_digest(computed, stored)


def _verify_password(password: str) -> bool:
    if AUTH_PASSWORD_HASH:
        return _verify_password_hash(password, AUTH_PASSWORD_HASH)
    return secrets.compare_digest(password, BASIC_AUTH_PASS)


def _find_registry_user(identifier: str) -> Optional[dict]:
    normalized = normalize_identifier(identifier)
    target_candidates = set(identifier_candidates(normalized))
    if not target_candidates:
        return None
    for user in _load_users():
        username = user.get("username")
        password_hash = user.get("password_hash")
        if not isinstance(username, str) or not isinstance(password_hash, str):
            continue
        user_candidates = set(identifier_candidates(username))
        if target_candidates & user_candidates:
            return user
    return None


def authenticate(username: str, password: str) -> bool:
    registry_user = _find_registry_user(username)
    if registry_user:
        return _verify_password_hash(password, registry_user["password_hash"])

    username_ok = False
    normalized = normalize_identifier(username)
    configured = normalize_identifier(BASIC_AUTH_USER)
    for candidate in identifier_candidates(normalized):
        if secrets.compare_digest(candidate, configured):
            username_ok = True
            break
    password_ok = _verify_password(password)
    return username_ok and password_ok


def normalize_identifier(value: str) -> str:
    return value.strip().lower()


def identifier_candidates(identifier: str) -> list[str]:
    value = normalize_identifier(identifier)
    if not value:
        return []
    candidates = {value}
    if "@" in value:
        candidates.add(value.split("@", 1)[0])
    elif AUTH_IDENTIFIER_DOMAIN:
        candidates.add(f"{value}@{AUTH_IDENTIFIER_DOMAIN.lower()}")
    return list(candidates)


def canonical_identifier(identifier: str) -> str:
    registry_user = _find_registry_user(identifier)
    if registry_user and isinstance(registry_user.get("username"), str):
        return registry_user["username"]

    normalized = normalize_identifier(identifier)
    configured = normalize_identifier(BASIC_AUTH_USER)
    for candidate in identifier_candidates(normalized):
        if secrets.compare_digest(candidate, configured):
            return BASIC_AUTH_USER
    return identifier.strip()


def is_admin(identifier: str) -> bool:
    """Admins are the bootstrap BASIC_AUTH_USER, anyone in ADMIN_USERS, or a
    registry user carrying role=admin."""
    normalized = normalize_identifier(identifier)
    candidates = set(identifier_candidates(normalized))
    if not candidates:
        return False
    if candidates & {normalize_identifier(BASIC_AUTH_USER)}:
        return True
    for admin in ADMIN_USERS:
        if candidates & set(identifier_candidates(admin)):
            return True
    registry_user = _find_registry_user(identifier)
    if registry_user:
        if registry_user.get("role") == "admin" or registry_user.get("is_admin") is True:
            return True
    return False


def create_user(username: str, password: str, created_by: str, role: str = "user") -> dict:
    normalized = normalize_identifier(username)
    if not re.fullmatch(r"[a-z0-9._-]{3,64}(@[a-z0-9.-]{3,255})?", normalized):
        raise ValueError("Username must be 3-64 chars and use [a-z0-9._-], optional @domain.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if role not in {"user", "admin"}:
        raise ValueError("Role must be 'user' or 'admin'.")
    salt = secrets.token_urlsafe(16)
    iterations = 210000
    password_hash = f"pbkdf2_sha256${iterations}${salt}${_hash_password(password, salt, iterations)}"
    now = _utc_now()
    record = {
        "username": normalized,
        "password_hash": password_hash,
        "role": role,
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
    }
    with _USERS_LOCK:
        if _find_registry_user(normalized):
            raise ValueError("User already exists.")
        users = _load_users()
        users.append(record)
        _save_users(users)
    return {
        "username": normalized,
        "role": role,
        "created_by": created_by,
        "created_at": now,
    }


def create_session(username: str) -> str:
    exp = int(time.time()) + SESSION_TTL_MINUTES * 60
    payload = {"sub": username, "exp": exp}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = _sign(payload_bytes)
    return f"{_b64url_encode(payload_bytes)}.{signature}"


def verify_session(token: str) -> Optional[str]:
    try:
        payload_b64, signature = token.split(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        expected = _sign(payload_bytes)
        if not secrets.compare_digest(signature, expected):
            return None
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None
    if payload.get("exp", 0) < int(time.time()):
        return None
    return payload.get("sub")


def create_progress_token(run_id: int, ttl_hours: int) -> tuple[str, str, int]:
    """Return (token, jti, exp). The jti lets the token be revoked server-side."""
    exp = int(time.time()) + max(ttl_hours, 1) * 3600
    jti = secrets.token_urlsafe(16)
    payload = {"typ": "run_progress", "run_id": run_id, "exp": exp, "jti": jti}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = _sign(payload_bytes)
    return f"{_b64url_encode(payload_bytes)}.{signature}", jti, exp


def verify_progress_token(token: str) -> Optional[dict]:
    try:
        payload_b64, signature = token.split(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        expected = _sign(payload_bytes)
        if not secrets.compare_digest(signature, expected):
            return None
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None
    if payload.get("typ") != "run_progress":
        return None
    if payload.get("exp", 0) < int(time.time()):
        return None
    run_id = payload.get("run_id")
    if not isinstance(run_id, int):
        return None
    if not isinstance(payload.get("jti"), str) or not payload["jti"]:
        return None
    return payload


def set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE.lower(),
        max_age=SESSION_TTL_MINUTES * 60,
    )


def _new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def set_csrf_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        httponly=False,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE.lower(),
        max_age=SESSION_TTL_MINUTES * 60,
    )


def ensure_csrf_cookie(request: Request, response: Response, rotate: bool = False) -> str:
    existing = request.cookies.get(CSRF_COOKIE_NAME)
    if existing and not rotate:
        return existing
    token = _new_csrf_token()
    set_csrf_cookie(response, token)
    return token


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME)
    response.delete_cookie(CSRF_COOKIE_NAME)


def _extract_token(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.cookies.get(COOKIE_NAME)


def _uses_bearer_auth(request: Request) -> bool:
    auth_header = request.headers.get("Authorization", "")
    return auth_header.lower().startswith("bearer ")


def require_csrf(request: Request) -> None:
    if _uses_bearer_auth(request):
        return
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_HEADER_NAME)
    if not cookie_token or not header_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing CSRF token")
    if not secrets.compare_digest(cookie_token, header_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")


def require_session(request: Request) -> str:
    token = _extract_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    username = verify_session(token)
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return username
