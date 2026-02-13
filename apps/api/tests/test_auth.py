import importlib
import os
from pathlib import Path

from fastapi.testclient import TestClient


def create_client(tmp_path: Path) -> TestClient:
    os.environ["DB_PATH"] = str(tmp_path / "runs.db")
    os.environ["RUNS_DIR"] = str(tmp_path / "runs")
    os.environ["PRESETS_DIR"] = str(tmp_path / "presets")
    os.environ["ARTIFACT_ROOTS"] = str(tmp_path)
    os.environ["WORKER_ENABLED"] = "false"
    os.environ["QUEUE_ENABLED"] = "false"
    os.environ["UPLOAD_CLEANUP_ENABLED"] = "false"
    os.environ["SESSION_SECRET"] = "test-secret"
    os.environ["PREFLIGHT_CHECK_PATHS"] = "false"
    os.environ["BASIC_AUTH_USER"] = "test-user"
    os.environ["BASIC_AUTH_PASS"] = "test-pass"
    os.environ["USERS_REGISTRY_PATH"] = str(tmp_path / "registries" / "users.json")

    from app import auth, db, main, registry, runner, schemas, settings, upload_store, validation, worker

    importlib.reload(settings)
    importlib.reload(auth)
    importlib.reload(db)
    importlib.reload(runner)
    importlib.reload(worker)
    importlib.reload(registry)
    importlib.reload(validation)
    importlib.reload(upload_store)
    importlib.reload(schemas)
    importlib.reload(main)

    return TestClient(main.app)


def test_login_and_me(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    resp = client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "test-user"
    assert resp.json().get("csrf_token")
    assert "set-cookie" in resp.headers

    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["username"] == "test-user"


def test_create_user_and_login(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        login = client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
        assert login.status_code == 200
        csrf = login.json().get("csrf_token")
        assert csrf

        create = client.post(
            "/auth/users",
            json={"username": "analyst", "password": "new-password-123"},
            headers={"X-CSRF-Token": csrf},
        )
        assert create.status_code == 200
        assert create.json()["username"] == "analyst"
        assert create.json()["created_by"] == "test-user"

        client.post("/auth/logout", headers={"X-CSRF-Token": csrf})

        analyst_login = client.post(
            "/auth/login",
            json={"username": "analyst", "password": "new-password-123"},
        )
        assert analyst_login.status_code == 200
        assert analyst_login.json()["username"] == "analyst"
