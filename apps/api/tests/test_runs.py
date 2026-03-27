import os
import importlib
import json
import sys
import types
from pathlib import Path
from fastapi import HTTPException

from fastapi.testclient import TestClient


def create_client(tmp_path: Path, extra_env: dict[str, str] | None = None) -> TestClient:
    os.environ["DB_PATH"] = str(tmp_path / "runs.db")
    os.environ["RUNS_DIR"] = str(tmp_path / "runs")
    os.environ["PRESETS_DIR"] = str(tmp_path / "presets")
    os.environ["ARTIFACT_ROOTS"] = str(tmp_path)
    os.environ["WORKER_ENABLED"] = "false"
    os.environ["QUEUE_ENABLED"] = "true"
    os.environ["SESSION_SECRET"] = "test-secret"
    os.environ["PREFLIGHT_CHECK_PATHS"] = "false"
    os.environ["BASIC_AUTH_USER"] = "test-user"
    os.environ["BASIC_AUTH_PASS"] = "test-pass"
    if extra_env:
        for key, value in extra_env.items():
            os.environ[key] = value
    multipart = types.ModuleType("multipart")
    multipart.__version__ = "0.0"
    multipart_submodule = types.ModuleType("multipart.multipart")
    multipart_submodule.parse_options_header = lambda value: ("", {})
    sys.modules["multipart"] = multipart
    sys.modules["multipart.multipart"] = multipart_submodule

    from app import settings, db, main, runner, worker, registry, validation, auth
    importlib.reload(settings)
    importlib.reload(auth)
    importlib.reload(db)
    importlib.reload(runner)
    importlib.reload(worker)
    importlib.reload(registry)
    importlib.reload(validation)
    importlib.reload(main)

    return TestClient(main.app)


def test_create_run_queued(tmp_path: Path) -> None:
    client = create_client(tmp_path)
    with client:
        login = client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
        assert login.status_code == 200
        csrf_token = login.json().get("csrf_token")
        assert csrf_token

        output_dir = tmp_path / "outputs"
        data_path = tmp_path / "data.h5ad"
        ref_path = tmp_path / "ref.h5ad"
        meta_path = tmp_path / "meta.csv"

        payload = {
            "run_name": "test-run",
            "queue": True,
            "config": {
                "output_dir": str(output_dir),
                "cosmx_h5ad_path": str(data_path),
                "reference_h5ad_path": str(ref_path),
                "cell_metadata_path": str(meta_path),
                "n_components": 10,
            },
        }
        resp = client.post("/runs", json=payload, headers={"X-CSRF-Token": csrf_token or ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"


def test_create_run_prepare_only_stays_local_when_ssh_backend_configured(tmp_path: Path, monkeypatch) -> None:
    client = create_client(tmp_path)
    run_name = "prepare-local"
    output_dir = tmp_path / "outputs"
    data_path = tmp_path / "data.h5ad"
    ref_path = tmp_path / "ref.h5ad"
    meta_path = tmp_path / "meta.csv"

    os.environ["SLURM_BACKEND"] = "ssh"
    os.environ["SSH_HOST"] = "example.org"
    os.environ["SSH_USER"] = "tester"
    os.environ["SSH_REMOTE_RUNS_DIR"] = "/remote/runs"

    from app import settings, db, main, runner, worker, registry, validation, auth
    importlib.reload(settings)
    importlib.reload(auth)
    importlib.reload(db)
    importlib.reload(runner)
    importlib.reload(worker)
    importlib.reload(registry)
    importlib.reload(validation)
    importlib.reload(main)

    calls = []

    def fake_run(cmd, capture_output, text):
        calls.append(cmd)
        run_dir = Path(cmd[cmd.index("--config") + 1]).parent
        (run_dir / "run.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (run_dir / "submit.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (run_dir / "config.resolved.json").write_text("{}", encoding="utf-8")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with TestClient(main.app) as reloaded_client:
        login = reloaded_client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
        csrf_token = login.json().get("csrf_token")
        payload = {
            "run_name": run_name,
            "submit": False,
            "queue": False,
            "config": {
                "output_dir": str(output_dir),
                "cosmx_h5ad_path": str(data_path),
                "reference_h5ad_path": str(ref_path),
                "cell_metadata_path": str(meta_path),
                "n_components": 10,
            },
        }
        resp = reloaded_client.post("/runs", json=payload, headers={"X-CSRF-Token": csrf_token or ""})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "prepared"
    assert calls
    config_path = Path(body["config_path"])
    assert config_path.exists()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["run_name"] == run_name


def test_build_sbatch_temporarily_disables_nounset_for_conda_activation() -> None:
    from run_pipeline import build_sbatch

    script = build_sbatch(
        run_dir=Path("/tmp/run"),
        run_command="bash run.sh",
        slurm={"conda_env": "/tmp/env"},
        output_dir="/tmp/run/output",
        run_name="demo",
    )

    assert "set +u" in script
    assert 'export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"' in script
    assert "conda activate /tmp/env" in script
    assert "set -u" in script


def test_prepare_run_bundle_rewrites_remote_output_dir(tmp_path: Path, monkeypatch) -> None:
    os.environ["RUNS_DIR"] = str(tmp_path / "runs")
    os.environ["ARTIFACT_ROOTS"] = str(tmp_path)

    from app import settings, runner
    importlib.reload(settings)
    importlib.reload(runner)

    run_name = "bundle-remote"
    run_dir = tmp_path / "runs" / run_name
    output_dir = run_dir / "outputs"

    def fake_run(cmd, capture_output, text):
        generated_run_dir = Path(cmd[cmd.index("--config") + 1]).parent
        (generated_run_dir / "run.sh").write_text("#!/bin/bash\n", encoding="utf-8")
        (generated_run_dir / "submit.sh").write_text("#!/bin/bash\ncd /tmp\n", encoding="utf-8")

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    prepared = runner.prepare_run_bundle(
        run_name,
        {
            "run_name": run_name,
            "cosmx_h5ad_path": str(tmp_path / "cosmx.h5ad"),
            "reference_h5ad_path": str(tmp_path / "reference.h5ad"),
            "cell_metadata_path": str(tmp_path / "meta.csv"),
            "output_dir": str(output_dir),
            "n_components": 4,
        },
        remote_run_dir="/blue/group/user/runs/bundle-remote",
    )

    assert prepared["output_dir"] == "/blue/group/user/runs/bundle-remote/outputs"


def test_external_poller_mode_disables_vm_worker(tmp_path: Path) -> None:
    os.environ["RUNS_DIR"] = str(tmp_path / "runs")
    os.environ["ARTIFACT_ROOTS"] = str(tmp_path)
    os.environ["SESSION_SECRET"] = "test-secret"
    os.environ["BASIC_AUTH_USER"] = "test-user"
    os.environ["BASIC_AUTH_PASS"] = "test-pass"
    os.environ["WORKER_ENABLED"] = "true"
    os.environ["QUEUE_POLLER_TOKEN"] = "poller-token"
    os.environ["SLURM_BACKEND"] = "local"

    from app import settings
    importlib.reload(settings)

    try:
        settings.validate_settings()
    except RuntimeError as exc:
        assert "Disable WORKER_ENABLED when using the external HPG queue poller." in str(exc)
    else:
        raise AssertionError("Expected validate_settings to reject WORKER_ENABLED with external poller mode")


def test_queue_claim_bundle_submission_and_status_append(tmp_path: Path, monkeypatch) -> None:
    client = create_client(
        tmp_path,
        extra_env={
            "QUEUE_POLLER_TOKEN": "poller-token",
            "QUEUE_REMOTE_RUNS_DIR": "/blue/group/user/runs",
        },
    )
    with client:
        login = client.post("/auth/login", json={"username": "test-user", "password": "test-pass"})
        csrf_token = login.json().get("csrf_token")

        payload = {
            "run_name": "queue-e2e",
            "queue": True,
            "submit": True,
            "config": {
                "output_dir": str(tmp_path / "outputs"),
                "cosmx_h5ad_path": str(tmp_path / "data.h5ad"),
                "reference_h5ad_path": str(tmp_path / "ref.h5ad"),
                "cell_metadata_path": str(tmp_path / "meta.csv"),
                "n_components": 4,
            },
        }
        create = client.post("/runs", json=payload, headers={"X-CSRF-Token": csrf_token or ""})
        assert create.status_code == 200
        run_id = create.json()["id"]

        from app import main

        def fake_prepare_run_bundle(run_name, config, remote_run_dir=None):
            run_dir = Path(config.get("output_dir", tmp_path / "outputs")).parent
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            bundle_path = run_dir / "run_bundle.tar.gz"
            bundle_path.write_bytes(b"bundle")
            submit_path = run_dir / "submit.sh"
            submit_path.write_text("#!/bin/bash\n", encoding="utf-8")
            return {
                "run_dir": str(run_dir),
                "output_dir": f"{remote_run_dir}/outputs" if remote_run_dir else str(run_dir / "outputs"),
                "config_path": str(config_path),
                "bundle_path": str(bundle_path),
                "submit_script": submit_path.read_text(encoding="utf-8"),
            }

        monkeypatch.setattr(main, "prepare_run_bundle", fake_prepare_run_bundle)

        claim = client.post("/queue/claim", headers={"X-Queue-Token": "poller-token"})
        assert claim.status_code == 200
        body = claim.json()
        assert body["ok"] is True
        assert body["job"]["run_id"] == run_id

        bundle = client.get(body["job"]["bundle_url"], headers={"X-Queue-Token": "poller-token"})
        assert bundle.status_code == 200

        report_submission = main.queue_report_submission(
            run_id=run_id,
            claim_id=body["job"]["claim_id"],
            slurm_job_id="12345",
            message="Submitted by HPG poller",
        )
        assert report_submission["ok"] is True

        active = client.get("/queue/active", headers={"X-Queue-Token": "poller-token"})
        assert active.status_code == 200
        items = active.json()["items"]
        assert len(items) == 1
        assert items[0]["output_dir"] == "/blue/group/user/runs/queue-e2e/outputs"

        first_status = main.queue_report_status(
            run_id=run_id,
            status="running",
            slurm_state="",
            slurm_reason="",
            slurm_elapsed="",
            started_at="",
            finished_at="",
            message="== cell2loc_nmf.err ==\nfirst",
        )
        assert first_status["ok"] is True
        second_status = main.queue_report_status(
            run_id=run_id,
            status="failed",
            slurm_state="",
            slurm_reason="",
            slurm_elapsed="",
            started_at="",
            finished_at="",
            message="== cell2loc_nmf.err ==\nsecond",
        )
        assert second_status["ok"] is True
        try:
            main.queue_report_status(
                run_id=run_id,
                status="oops",
                slurm_state="",
                slurm_reason="",
                slurm_elapsed="",
                started_at="",
                finished_at="",
                message="",
            )
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("Expected invalid poller status to raise HTTPException")

        run = client.get(f"/runs/{run_id}")
        assert run.status_code == 200
        run_body = run.json()
        assert run_body["status"] == "failed"
        assert "first" in (run_body["message"] or "")
        assert "second" in (run_body["message"] or "")
        bundle_path = Path(tmp_path / "run_bundle.tar.gz")
        assert not bundle_path.exists()
