import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, Response, status

from .settings import (
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
    payload = {"users": users}
    USERS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    USERS_REGISTRY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def create_user(username: str, password: str, created_by: str) -> dict:
    normalized = normalize_identifier(username)
    if not re.fullmatch(r"[a-z0-9._-]{3,64}(@[a-z0-9.-]{3,255})?", normalized):
        raise ValueError("Username must be 3-64 chars and use [a-z0-9._-], optional @domain.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if _find_registry_user(normalized):
        raise ValueError("User already exists.")

    users = _load_users()
    salt = secrets.token_urlsafe(16)
    iterations = 210000
    password_hash = f"pbkdf2_sha256${iterations}${salt}${_hash_password(password, salt, iterations)}"
    now = _utc_now()
    record = {
        "username": normalized,
        "password_hash": password_hash,
        "created_by": created_by,
        "created_at": now,
        "updated_at": now,
    }
    users.append(record)
    _save_users(users)
    return {
        "username": normalized,
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


def create_progress_token(run_id: int, ttl_hours: int) -> str:
    exp = int(time.time()) + max(ttl_hours, 1) * 3600
    payload = {"typ": "run_progress", "run_id": run_id, "exp": exp}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = _sign(payload_bytes)
    return f"{_b64url_encode(payload_bytes)}.{signature}"


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
