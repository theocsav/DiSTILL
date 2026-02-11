import hashlib
import importlib
import json
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
    os.environ["WORKER_ENABLED"] = "false"
    os.environ["QUEUE_ENABLED"] = "true"
    os.environ["UPLOAD_CLEANUP_ENABLED"] = "false"
    os.environ["UPLOAD_MAX_CONCURRENT_PER_USER"] = "6"
    os.environ["UPLOAD_MAX_SIZE_STAGED_GB"] = "100"
    os.environ["UPLOAD_MAX_SIZE_METADATA_GB"] = "5"
    os.environ["UPLOAD_MAX_SIZE_REFERENCE_GB"] = "50"
    os.environ["UPLOAD_ALLOWED_EXT_STAGED"] = ".h5ad"
    os.environ["UPLOAD_ALLOWED_EXT_METADATA"] = ".csv,.tsv,.gz"
    os.environ["UPLOAD_ALLOWED_EXT_REFERENCE"] = ".h5ad"
    os.environ["SESSION_SECRET"] = "test-secret"
    os.environ["PREFLIGHT_CHECK_PATHS"] = "false"
    os.environ["BASIC_AUTH_USER"] = "test-user"
    os.environ["BASIC_AUTH_PASS"] = "test-pass"
    if extra_env:
        for key, value in extra_env.items():
            os.environ[key] = value

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


def login_and_csrf(client: TestClient) -> str:
    resp = client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
    assert resp.status_code == 200
    token = resp.json().get("csrf_token")
    assert token
    return token


def test_chunked_upload_finalize_and_public_listing(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login_and_csrf(client)

        staged_bytes = b"staged-data-123"
        metadata_bytes = b"cell_id,fov\n1,A\n"
        staged_sha = hashlib.sha256(staged_bytes).hexdigest()

        init_staged = client.post(
            "/uploads/init",
            json={
                "dataset_id": "ibd_chunk_test",
                "file_role": "staged",
                "file_name": "cosmx.h5ad",
                "total_size": len(staged_bytes),
                "expected_sha256": staged_sha,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert init_staged.status_code == 200
        staged_id = init_staged.json()["upload_id"]

        init_meta = client.post(
            "/uploads/init",
            json={
                "dataset_id": "ibd_chunk_test",
                "file_role": "metadata",
                "file_name": "cells.csv",
                "total_size": len(metadata_bytes),
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert init_meta.status_code == 200
        meta_id = init_meta.json()["upload_id"]

        part_a = staged_bytes[:5]
        part_b = staged_bytes[5:]
        up_a = client.put(
            f"/uploads/{staged_id}/chunk?offset=0",
            content=part_a,
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"},
        )
        assert up_a.status_code == 200
        assert up_a.json()["received_bytes"] == len(part_a)
        up_b = client.put(
            f"/uploads/{staged_id}/chunk?offset={len(part_a)}",
            content=part_b,
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"},
        )
        assert up_b.status_code == 200
        assert up_b.json()["received_bytes"] == len(staged_bytes)

        up_meta = client.put(
            f"/uploads/{meta_id}/chunk?offset=0",
            content=metadata_bytes,
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"},
        )
        assert up_meta.status_code == 200
        assert up_meta.json()["received_bytes"] == len(metadata_bytes)

        done_staged = client.post(f"/uploads/{staged_id}/complete", headers={"X-CSRF-Token": csrf})
        assert done_staged.status_code == 200
        assert done_staged.json()["completed"] is True
        assert done_staged.json()["checksum_valid"] is True
        assert done_staged.json()["sha256"] == staged_sha

        done_meta = client.post(f"/uploads/{meta_id}/complete", headers={"X-CSRF-Token": csrf})
        assert done_meta.status_code == 200
        assert done_meta.json()["completed"] is True

        finalize = client.post(
            "/datasets/upload/finalize",
            json={
                "dataset_id": "ibd_chunk_test",
                "label": "IBD Chunk Test",
                "organ": "colon",
                "platform": "cosmx",
                "staged_upload_id": staged_id,
                "cell_metadata_upload_id": meta_id,
                "public": True,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert finalize.status_code == 200
        dataset = finalize.json()["dataset"]
        assert dataset["id"] == "ibd_chunk_test"
        assert dataset["source"] == "chunked_upload"
        assert dataset["checksums"]["staged"] == staged_sha
        assert Path(dataset["staged_path"]).exists()
        assert Path(dataset["cell_metadata_path"]).exists()

        listed = client.get("/datasets/public")
        assert listed.status_code == 200
        ids = {item["id"] for item in listed.json()}
        assert "ibd_chunk_test" in ids


def test_chunked_upload_checksum_mismatch_fails_complete(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login_and_csrf(client)
        payload = b"abcdef"
        wrong_sha = "0" * 64
        init_resp = client.post(
            "/uploads/init",
            json={
                "dataset_id": "ibd_bad_sha",
                "file_role": "staged",
                "file_name": "bad.h5ad",
                "total_size": len(payload),
                "expected_sha256": wrong_sha,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert init_resp.status_code == 200
        upload_id = init_resp.json()["upload_id"]
        put_resp = client.put(
            f"/uploads/{upload_id}/chunk?offset=0",
            content=payload,
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"},
        )
        assert put_resp.status_code == 200
        done = client.post(f"/uploads/{upload_id}/complete", headers={"X-CSRF-Token": csrf})
        assert done.status_code == 400
        assert "Checksum validation failed" in done.text


def test_share_link_and_public_progress(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login_and_csrf(client)
        payload = {
            "run_name": "share-run",
            "queue": True,
            "config": {
                "output_dir": str(tmp_path / "outputs"),
                "cosmx_h5ad_path": str(tmp_path / "data.h5ad"),
                "reference_h5ad_path": str(tmp_path / "ref.h5ad"),
                "cell_metadata_path": str(tmp_path / "meta.csv"),
                "n_components": 4,
            },
        }
        create = client.post("/runs", json=payload, headers={"X-CSRF-Token": csrf})
        assert create.status_code == 200
        run_id = create.json()["id"]

        share = client.post(f"/runs/{run_id}/share", json={}, headers={"X-CSRF-Token": csrf})
        assert share.status_code == 200
        token = share.json()["token"]
        assert token
        assert f"/progress/{token}" in share.json()["url"]

        public = client.get(f"/public/runs/progress?token={token}")
        assert public.status_code == 200
        body = public.json()
        assert body["id"] == run_id
        assert body["run_name"] == "share-run"
        assert body["status"] in {"queued", "created", "prepared", "submitted"}

        tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
        invalid = client.get(f"/public/runs/progress?token={tampered}")
        assert invalid.status_code == 401


def test_cleanup_stale_upload_sessions(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login_and_csrf(client)
        data = b"stale"
        init_resp = client.post(
            "/uploads/init",
            json={
                "dataset_id": "stale_dataset",
                "file_role": "staged",
                "file_name": "stale.h5ad",
                "total_size": len(data),
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert init_resp.status_code == 200
        upload_id = init_resp.json()["upload_id"]
        up = client.put(
            f"/uploads/{upload_id}/chunk?offset=0",
            content=data,
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"},
        )
        assert up.status_code == 200

    from app import upload_store

    session_file = Path(os.environ["DATA_UPLOADS_DIR"]) / ".sessions" / f"{upload_id}.json"
    manifest = json.loads(session_file.read_text(encoding="utf-8"))
    manifest["updated_at"] = "2000-01-01T00:00:00+00:00"
    session_file.write_text(json.dumps(manifest), encoding="utf-8")
    temp_path = Path(manifest["temp_path"])
    assert temp_path.exists()

    result = upload_store.cleanup_stale_uploads(ttl_hours=1)
    assert result["removed_sessions"] >= 1
    assert not session_file.exists()
    assert not temp_path.exists()


def test_upload_limits_extensions_and_concurrency(tmp_path: Path) -> None:
    client = create_client(
        tmp_path,
        extra_env={
            "UPLOAD_MAX_CONCURRENT_PER_USER": "1",
            "UPLOAD_MAX_SIZE_METADATA_GB": "0.000000001",
        },
    )
    with client:
        csrf = login_and_csrf(client)

        invalid_ext = client.post(
            "/uploads/init",
            json={
                "dataset_id": "limits_ds",
                "file_role": "staged",
                "file_name": "file.txt",
                "total_size": 10,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert invalid_ext.status_code == 400
        assert "File extension not allowed" in invalid_ext.text

        too_big = client.post(
            "/uploads/init",
            json={
                "dataset_id": "limits_ds",
                "file_role": "metadata",
                "file_name": "meta.csv",
                "total_size": 1000,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert too_big.status_code == 400
        assert "exceeds max size" in too_big.text

        first = client.post(
            "/uploads/init",
            json={
                "dataset_id": "limits_ds",
                "file_role": "staged",
                "file_name": "a.h5ad",
                "total_size": 10,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert first.status_code == 200

        second = client.post(
            "/uploads/init",
            json={
                "dataset_id": "limits_ds",
                "file_role": "reference",
                "file_name": "ref.h5ad",
                "total_size": 10,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert second.status_code == 400
        assert "Too many active uploads" in second.text


def test_dataset_patch_and_delete(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        csrf = login_and_csrf(client)
        staged_bytes = b"abc"
        metadata_bytes = b"id,fov\n1,A\n"

        s = client.post(
            "/uploads/init",
            json={
                "dataset_id": "moderate_ds",
                "file_role": "staged",
                "file_name": "s.h5ad",
                "total_size": len(staged_bytes),
            },
            headers={"X-CSRF-Token": csrf},
        )
        m = client.post(
            "/uploads/init",
            json={
                "dataset_id": "moderate_ds",
                "file_role": "metadata",
                "file_name": "m.csv",
                "total_size": len(metadata_bytes),
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert s.status_code == 200 and m.status_code == 200
        sid = s.json()["upload_id"]
        mid = m.json()["upload_id"]
        assert client.put(
            f"/uploads/{sid}/chunk?offset=0",
            content=staged_bytes,
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"},
        ).status_code == 200
        assert client.put(
            f"/uploads/{mid}/chunk?offset=0",
            content=metadata_bytes,
            headers={"X-CSRF-Token": csrf, "Content-Type": "application/octet-stream"},
        ).status_code == 200
        assert client.post(f"/uploads/{sid}/complete", headers={"X-CSRF-Token": csrf}).status_code == 200
        assert client.post(f"/uploads/{mid}/complete", headers={"X-CSRF-Token": csrf}).status_code == 200
        fin = client.post(
            "/datasets/upload/finalize",
            json={
                "dataset_id": "moderate_ds",
                "label": "Moderation Test",
                "organ": "colon",
                "platform": "cosmx",
                "staged_upload_id": sid,
                "cell_metadata_upload_id": mid,
                "public": True,
            },
            headers={"X-CSRF-Token": csrf},
        )
        assert fin.status_code == 200

        patch_resp = client.patch(
            "/datasets/moderate_ds",
            json={"label": "Updated Label", "notes": "reviewed", "public": False},
            headers={"X-CSRF-Token": csrf},
        )
        assert patch_resp.status_code == 200
        body = patch_resp.json()["dataset"]
        assert body["label"] == "Updated Label"
        assert body["notes"] == "reviewed"
        assert body["public"] is False
        assert body["updated_by"] == "test-user"

        delete_resp = client.delete("/datasets/moderate_ds", headers={"X-CSRF-Token": csrf})
        assert delete_resp.status_code == 200
        assert delete_resp.json()["deleted_id"] == "moderate_ds"
        listed = client.get("/datasets/public")
        assert listed.status_code == 200
        assert "moderate_ds" not in {item["id"] for item in listed.json()}
