"""Regression tests for the correctness and hardening fixes."""

import importlib
import os
from pathlib import Path

from fastapi.testclient import TestClient


def create_client(tmp_root: Path, extra_env: dict[str, str] | None = None) -> TestClient:
    os.environ["DB_PATH"] = str(tmp_root / "runs.db")
    os.environ["RUNS_DIR"] = str(tmp_root / "runs")
    os.environ["PRESETS_DIR"] = str(tmp_root / "presets")
    os.environ["ARTIFACT_ROOTS"] = str(tmp_root)
    os.environ["DATA_UPLOADS_DIR"] = str(tmp_root / "uploads")
    os.environ["DATASETS_REGISTRY_PATH"] = str(tmp_root / "registries" / "datasets.json")
    os.environ["USERS_REGISTRY_PATH"] = str(tmp_root / "registries" / "users.json")
    os.environ["WORKER_ENABLED"] = "false"
    os.environ["QUEUE_ENABLED"] = "true"
    os.environ["UPLOAD_CLEANUP_ENABLED"] = "false"
    os.environ["UPLOAD_MAX_CONCURRENT_PER_USER"] = "6"
    os.environ["UPLOAD_MAX_CHUNK_BYTES"] = str(64 * 1024 * 1024)
    os.environ["UPLOAD_MAX_SIZE_STAGED_GB"] = "100"
    os.environ["UPLOAD_MAX_SIZE_METADATA_GB"] = "5"
    os.environ["UPLOAD_MAX_SIZE_REFERENCE_GB"] = "50"
    os.environ["UPLOAD_ALLOWED_EXT_STAGED"] = ".h5ad"
    os.environ["UPLOAD_ALLOWED_EXT_METADATA"] = ".csv,.tsv,.gz"
    os.environ["UPLOAD_ALLOWED_EXT_REFERENCE"] = ".h5ad"
    os.environ["ADMIN_USERS"] = ""
    os.environ["SESSION_SECRET"] = "test-secret"
    os.environ["PREFLIGHT_CHECK_PATHS"] = "false"
    os.environ["BASIC_AUTH_USER"] = "test-user"
    os.environ["BASIC_AUTH_PASS"] = "test-pass"
    if extra_env:
        os.environ.update(extra_env)

    from app import auth, db, main, registry, runner, schemas, settings, synced_artifacts, upload_store, validation, worker

    importlib.reload(settings)
    importlib.reload(auth)
    importlib.reload(db)
    importlib.reload(runner)
    importlib.reload(worker)
    importlib.reload(registry)
    importlib.reload(validation)
    importlib.reload(upload_store)
    importlib.reload(synced_artifacts)
    importlib.reload(schemas)
    importlib.reload(main)

    return TestClient(main.app)


def login(client: TestClient, username: str = "test-user", password: str = "test-pass") -> str:
    resp = client.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["csrf_token"]


def make_run(client: TestClient, csrf: str, tmp_path: Path, run_name: str) -> int:
    resp = client.post(
        "/runs",
        json={
            "run_name": run_name,
            "queue": True,
            "config": {
                "output_dir": str(tmp_path / "outputs"),
                "cosmx_h5ad_path": str(tmp_path / "data.h5ad"),
                "reference_h5ad_path": str(tmp_path / "ref.h5ad"),
                "cell_metadata_path": str(tmp_path / "meta.csv"),
                "n_components": 4,
            },
        },
        headers={"X-CSRF-Token": csrf},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


# --- correctness -----------------------------------------------------------


def test_sync_manifest_round_trips(tmp_path: Path) -> None:
    """read_sync_manifest used to always return None: its body had been spliced
    into synced_root_size_bytes as unreachable code."""
    create_client(tmp_path)
    from app import synced_artifacts

    root = synced_artifacts.synced_root(7)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".sync_manifest.json").write_text('{"run_id": 7, "items": []}', encoding="utf-8")

    manifest = synced_artifacts.read_sync_manifest(7)
    assert manifest is not None
    assert manifest["run_id"] == 7


def test_sync_manifest_missing_and_corrupt(tmp_path: Path) -> None:
    create_client(tmp_path)
    from app import synced_artifacts

    assert synced_artifacts.read_sync_manifest(999) is None

    root = synced_artifacts.synced_root(8)
    root.mkdir(parents=True, exist_ok=True)
    (root / ".sync_manifest.json").write_text("{not json", encoding="utf-8")
    assert synced_artifacts.read_sync_manifest(8) is None


def test_synced_root_size_bytes_counts_files(tmp_path: Path) -> None:
    create_client(tmp_path)
    from app import synced_artifacts

    root = synced_artifacts.synced_root(9)
    (root / "nested").mkdir(parents=True, exist_ok=True)
    (root / "a.txt").write_bytes(b"12345")
    (root / "nested" / "b.txt").write_bytes(b"678")
    assert synced_artifacts.synced_root_size_bytes(root) == 8


def test_failed_checksum_leaves_upload_retryable(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login(client)
        payload = b"abcdef"
        init = client.post(
            "/uploads/init",
            json={
                "dataset_id": "retry_ds",
                "file_role": "staged",
                "file_name": "x.h5ad",
                "total_size": len(payload),
                "expected_sha256": "0" * 64,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert init.status_code == 200
        upload_id = init.json()["upload_id"]
        assert client.put(
            f"/uploads/{upload_id}/chunk?offset=0",
            content=payload,
            headers={"X-CSRF-Token": csrf},
        ).status_code == 200

        first = client.post(f"/uploads/{upload_id}/complete", headers={"X-CSRF-Token": csrf})
        assert first.status_code == 400
        assert "Checksum validation failed" in first.text

        # The temp file must survive so the client can retry rather than dead-end
        # on "Upload temp file not found".
        second = client.post(f"/uploads/{upload_id}/complete", headers={"X-CSRF-Token": csrf})
        assert second.status_code == 400
        assert "Checksum validation failed" in second.text


# --- authorization ---------------------------------------------------------


def test_only_admins_can_create_users(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login(client)
        created = client.post(
            "/auth/users",
            json={"username": "analyst", "password": "analyst-password"},
            headers={"X-CSRF-Token": csrf},
        )
        assert created.status_code == 200
        assert created.json()["role"] == "user"

        client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
        analyst_csrf = login(client, "analyst", "analyst-password")

        me = client.get("/auth/me")
        assert me.status_code == 200
        assert me.json()["is_admin"] is False

        denied = client.post(
            "/auth/users",
            json={"username": "intruder", "password": "intruder-password"},
            headers={"X-CSRF-Token": analyst_csrf},
        )
        assert denied.status_code == 403


def test_admin_role_and_admin_users_env_grant_access(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login(client)
        assert client.get("/auth/me").json()["is_admin"] is True
        promoted = client.post(
            "/auth/users",
            json={"username": "lead", "password": "lead-password", "role": "admin"},
            headers={"X-CSRF-Token": csrf},
        )
        assert promoted.status_code == 200
        assert promoted.json()["role"] == "admin"

        client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
        lead_csrf = login(client, "lead", "lead-password")
        assert client.get("/auth/me").json()["is_admin"] is True
        allowed = client.post(
            "/auth/users",
            json={"username": "hire", "password": "hire-password"},
            headers={"X-CSRF-Token": lead_csrf},
        )
        assert allowed.status_code == 200


def test_health_does_not_leak_disk_layout(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        anon = client.get("/health")
        assert anon.status_code == 200
        assert anon.json() == {"status": "ok"}

        assert client.get("/health/disk").status_code == 401

        login(client)
        authed = client.get("/health/disk")
        assert authed.status_code == 200
        assert "roots" in authed.json()["disk"]


# --- upload limits ---------------------------------------------------------


def test_chunk_larger_than_cap_is_rejected(tmp_path: Path) -> None:
    client = create_client(tmp_path, extra_env={"UPLOAD_MAX_CHUNK_BYTES": "8"})
    with client:
        csrf = login(client)
        init = client.post(
            "/uploads/init",
            json={
                "dataset_id": "cap_ds",
                "file_role": "staged",
                "file_name": "big.h5ad",
                "total_size": 1024,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert init.status_code == 200
        upload_id = init.json()["upload_id"]

        too_big = client.put(
            f"/uploads/{upload_id}/chunk?offset=0",
            content=b"0123456789",
            headers={"X-CSRF-Token": csrf},
        )
        assert too_big.status_code == 413

        ok = client.put(
            f"/uploads/{upload_id}/chunk?offset=0",
            content=b"01234567",
            headers={"X-CSRF-Token": csrf},
        )
        assert ok.status_code == 200


def test_chunk_overflowing_declared_size_is_not_written(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login(client)
        init = client.post(
            "/uploads/init",
            json={
                "dataset_id": "overflow_ds",
                "file_role": "staged",
                "file_name": "small.h5ad",
                "total_size": 4,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert init.status_code == 200
        upload_id = init.json()["upload_id"]

        overflow = client.put(
            f"/uploads/{upload_id}/chunk?offset=0",
            content=b"way-too-many-bytes",
            headers={"X-CSRF-Token": csrf},
        )
        assert overflow.status_code == 400
        assert "exceed declared total size" in overflow.text

        # Nothing should have landed on disk, so the offset is still 0.
        status = client.get(f"/uploads/{upload_id}/status")
        assert status.status_code == 200
        assert status.json()["received_bytes"] == 0


def test_legacy_multipart_upload_enforces_extension_and_size(tmp_path: Path) -> None:
    client = create_client(tmp_path, extra_env={"UPLOAD_MAX_SIZE_METADATA_GB": "0.000000001"})
    with client:
        csrf = login(client)
        form = {
            "dataset_id": "legacy_ds",
            "label": "Legacy",
            "organ": "colon",
            "platform": "cosmx",
        }

        bad_ext = client.post(
            "/datasets/upload",
            data=form,
            files={
                "staged_file": ("payload.txt", b"abc", "text/plain"),
                "cell_metadata_file": ("cells.csv", b"id\n1\n", "text/csv"),
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert bad_ext.status_code == 400
        assert "File extension not allowed" in bad_ext.text

        too_big = client.post(
            "/datasets/upload",
            data=form,
            files={
                "staged_file": ("payload.h5ad", b"abc", "application/octet-stream"),
                "cell_metadata_file": ("cells.csv", b"x" * 4096, "text/csv"),
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert too_big.status_code == 413


# --- share links -----------------------------------------------------------


def test_share_link_can_be_revoked(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login(client)
        run_id = make_run(client, csrf, tmp_path, "revoke-run")

        share = client.post(f"/runs/{run_id}/share", json={}, headers={"X-CSRF-Token": csrf})
        assert share.status_code == 200
        token = share.json()["token"]
        jti = share.json()["jti"]
        assert jti

        assert client.get(f"/public/runs/progress?token={token}").status_code == 200

        listed = client.get(f"/runs/{run_id}/share")
        assert listed.status_code == 200
        assert [item["jti"] for item in listed.json()] == [jti]

        revoked = client.delete(f"/runs/{run_id}/share/{jti}", headers={"X-CSRF-Token": csrf})
        assert revoked.status_code == 200

        assert client.get(f"/public/runs/progress?token={token}").status_code == 401
        # Revoking twice is a no-op, not a success.
        assert client.delete(f"/runs/{run_id}/share/{jti}", headers={"X-CSRF-Token": csrf}).status_code == 404


def test_revoke_all_share_links_for_run(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login(client)
        run_id = make_run(client, csrf, tmp_path, "revoke-all-run")
        tokens = [
            client.post(f"/runs/{run_id}/share", json={}, headers={"X-CSRF-Token": csrf}).json()["token"]
            for _ in range(3)
        ]
        for token in tokens:
            assert client.get(f"/public/runs/progress?token={token}").status_code == 200

        bulk = client.delete(f"/runs/{run_id}/share", headers={"X-CSRF-Token": csrf})
        assert bulk.status_code == 200
        assert bulk.json()["revoked"] == 3

        for token in tokens:
            assert client.get(f"/public/runs/progress?token={token}").status_code == 401


def test_share_token_for_other_run_is_rejected(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login(client)
        run_id = make_run(client, csrf, tmp_path, "mismatch-run")
        from app import auth, db

        # Validly signed, unrevoked, unexpired — but its jti is registered against a
        # different run, so the payload's run_id must not be trusted.
        forged, forged_jti, _exp = auth.create_progress_token(run_id, 24)
        db.record_share_token(forged_jti, run_id + 1, "test-user", "2999-01-01T00:00:00+00:00")
        assert db.fetch_share_token(forged_jti) is not None
        assert client.get(f"/public/runs/progress?token={forged}").status_code == 401

        # A signed token with no server-side record is rejected too.
        orphan, _orphan_jti, _ = auth.create_progress_token(run_id, 24)
        assert client.get(f"/public/runs/progress?token={orphan}").status_code == 401
