import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .registry import get_dataset
from .settings import ARTIFACT_ROOTS, PRESETS_DIR, REPO_ROOT, RUNS_DIR, SLURM_BACKEND, SSH_REMOTE_RUNS_DIR
from .ssh_exec import run_ssh_command, scp_upload, shell_quote
from .storage import enforce_allowed_path

PIPELINE_RUNNER = REPO_ROOT / "run_pipeline.py"


def load_preset(preset_path: str) -> Dict[str, Any]:
    preset_file = Path(preset_path)
    if not preset_file.is_absolute():
        preset_file = PRESETS_DIR / preset_file
    preset_file = preset_file.resolve()
    presets_root = PRESETS_DIR.resolve()
    if presets_root not in preset_file.parents and preset_file != presets_root:
        raise ValueError("Preset path must be inside the presets directory.")
    if not preset_file.exists():
        raise FileNotFoundError(f"Preset not found: {preset_file}")
    return json.loads(preset_file.read_text(encoding="utf-8"))


def apply_dataset_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_id = config.get("dataset_id")
    if not dataset_id:
        return config
    dataset = get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset not found: {dataset_id}")

    mapping = {
        "staged_path": "cosmx_h5ad_path",
        "cosmx_h5ad_path": "cosmx_h5ad_path",
        "cosmx_with_nmf_path": "cosmx_with_nmf_path",
        "cell_metadata_path": "cell_metadata_path",
        "reference_h5ad_path": "reference_h5ad_path",
        "ref_model_dir": "ref_model_dir",
    }
    merged = dict(config)
    for source, dest in mapping.items():
        if dest not in merged and dataset.get(source):
            merged[dest] = dataset[source]
    if "dataset_label" not in merged and dataset.get("label"):
        merged["dataset_label"] = dataset["label"]
    return merged


def resolve_run_paths(run_name: str, config: Dict[str, Any]) -> tuple[Path, Path, Dict[str, Any]]:
    config = dict(config)
    config["run_name"] = run_name
    requested_run_dir = config.get("run_dir")
    run_dir = Path(requested_run_dir) if requested_run_dir else RUNS_DIR / run_name
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if RUNS_DIR.resolve() not in run_dir.resolve().parents and run_dir.resolve() != RUNS_DIR.resolve():
        raise ValueError("run_dir must be inside RUNS_DIR")

    output_dir = config.get("output_dir")
    if not output_dir:
        output_dir = str(run_dir / "outputs")
        config["output_dir"] = output_dir
    enforce_allowed_path(Path(output_dir), ARTIFACT_ROOTS)
    return run_dir, Path(output_dir), config


def prepare_run(run_name: str, config: Dict[str, Any], submit: bool) -> Tuple[str, str, str, Optional[str]]:
    # Prepare-only runs must stay local so queue=false/submit=false never depends on ssh.
    if submit and SLURM_BACKEND == "ssh":
        return _prepare_run_ssh(run_name, config, submit)

    return _prepare_run_local(run_name, config, submit)


def _prepare_run_local(run_name: str, config: Dict[str, Any], submit: bool) -> Tuple[str, str, str, Optional[str]]:
    run_dir, output_dir, config = resolve_run_paths(run_name, config)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    emit_sbatch = submit or bool(config.get("slurm", {}).get("enabled"))
    cmd = [sys.executable, str(PIPELINE_RUNNER), "--config", str(config_path)]
    if emit_sbatch:
        cmd.append("--emit-sbatch")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Pipeline preparation failed")

    job_id = None
    if submit:
        submit_path = run_dir / "submit.sh"
        if not submit_path.exists():
            raise FileNotFoundError("submit.sh not found; use --emit-sbatch or enable slurm in config")
        submit_result = subprocess.run(["sbatch", str(submit_path)], capture_output=True, text=True)
        if submit_result.returncode != 0:
            raise RuntimeError(submit_result.stderr or submit_result.stdout or "sbatch failed")
        match = re.search(r"Submitted batch job (\d+)", submit_result.stdout)
        if match:
            job_id = match.group(1)

    return str(run_dir), str(output_dir), str(config_path), job_id


def _prepare_run_ssh(run_name: str, config: Dict[str, Any], submit: bool) -> Tuple[str, str, str, Optional[str]]:
    run_dir, output_dir, config = resolve_run_paths(run_name, config)
    remote_run_dir = _remote_run_dir(run_name)

    with tempfile.TemporaryDirectory(prefix=f"nicherunner-{run_name}-") as tmp_dir:
        staging_root = Path(tmp_dir)
        local_run_dir = staging_root / run_name
        local_run_dir.mkdir(parents=True, exist_ok=True)

        local_config = dict(config)
        local_config["run_dir"] = str(local_run_dir)
        local_config_path = local_run_dir / "config.json"
        local_config_path.write_text(json.dumps(local_config, indent=2), encoding="utf-8")

        emit_sbatch = submit or bool(config.get("slurm", {}).get("enabled"))
        cmd = [sys.executable, str(PIPELINE_RUNNER), "--config", str(local_config_path)]
        if emit_sbatch:
            cmd.append("--emit-sbatch")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Pipeline preparation failed")

        _rewrite_staged_paths(local_run_dir, str(local_run_dir).replace("\\", "/"), remote_run_dir)
        _upload_run_dir_ssh(local_run_dir, remote_run_dir)

    job_id = None
    if submit:
        submit_path = f"{remote_run_dir}/submit.sh"
        submit_result = run_ssh_command(f"sbatch {shell_quote(submit_path)}")
        if submit_result.returncode != 0:
            raise RuntimeError(submit_result.stderr or submit_result.stdout or "sbatch failed")
        match = re.search(r"Submitted batch job (\d+)", submit_result.stdout)
        if match:
            job_id = match.group(1)

    output_dir_str = str(output_dir).replace("\\", "/")
    remote_output_dir = output_dir_str.replace(str(run_dir).replace("\\", "/"), remote_run_dir, 1)
    config_path = f"{remote_run_dir}/config.json"
    return remote_run_dir, remote_output_dir, config_path, job_id


def _remote_run_dir(run_name: str) -> str:
    base = SSH_REMOTE_RUNS_DIR.rstrip("/")
    return f"{base}/{run_name}"


def _rewrite_staged_paths(run_dir: Path, local_prefix: str, remote_prefix: str) -> None:
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".sh", ".json", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        updated = text.replace(local_prefix, remote_prefix)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _upload_run_dir_ssh(local_run_dir: Path, remote_run_dir: str) -> None:
    # Copy prepared run artifacts to the remote HPG path before submission.
    remote_parent = str(Path(remote_run_dir).parent).replace("\\", "/")
    remote_target = remote_run_dir.replace("\\", "/")
    mkdir = run_ssh_command(f"mkdir -p {shell_quote(remote_parent)}")
    if mkdir.returncode != 0:
        raise RuntimeError(mkdir.stderr or mkdir.stdout or f"Failed to create remote directory: {remote_parent}")

    archive_base = local_run_dir.parent / f"{local_run_dir.name}.tar"
    archive_path = shutil.make_archive(str(archive_base.with_suffix("")), "tar", root_dir=local_run_dir.parent, base_dir=local_run_dir.name)
    try:
        copied = scp_upload(archive_path, remote_parent)
        if copied.returncode != 0:
            raise RuntimeError(copied.stderr or copied.stdout or "scp upload failed")
        archive_name = Path(archive_path).name
        remote_archive = f"{remote_parent}/{archive_name}"
        extract = run_ssh_command(
            f"tar -xf {shell_quote(remote_archive)} -C {shell_quote(remote_parent)} && rm -f {shell_quote(remote_archive)}"
        )
        if extract.returncode != 0:
            raise RuntimeError(extract.stderr or extract.stdout or "Remote extract failed")
        run_ssh_command(f"chmod +x {shell_quote(remote_target)}/run.sh || true")
        run_ssh_command(f"chmod +x {shell_quote(remote_target)}/submit.sh || true")
    finally:
        Path(archive_path).unlink(missing_ok=True)


def prepare_run_bundle(run_name: str, config: Dict[str, Any], remote_run_dir: Optional[str] = None) -> Dict[str, str]:
    run_dir, output_dir, config = resolve_run_paths(run_name, config)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cmd = [sys.executable, str(PIPELINE_RUNNER), "--config", str(config_path), "--emit-sbatch"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Pipeline preparation failed")

    submit_path = run_dir / "submit.sh"
    if not submit_path.exists():
        raise FileNotFoundError("submit.sh not found after preparation")
    submit_script = submit_path.read_text(encoding="utf-8")

    remote_output_dir = str(output_dir)
    if remote_run_dir:
        _rewrite_staged_paths(run_dir, str(run_dir).replace("\\", "/"), remote_run_dir.replace("\\", "/"))
        submit_script = submit_path.read_text(encoding="utf-8")
        output_dir_str = str(output_dir).replace("\\", "/")
        remote_output_dir = output_dir_str.replace(str(run_dir).replace("\\", "/"), remote_run_dir.replace("\\", "/"), 1)

    bundle_path = run_dir / "run_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        for path in run_dir.rglob("*"):
            if not path.is_file() or path == bundle_path:
                continue
            arcname = path.relative_to(run_dir)
            tar.add(path, arcname=str(arcname))

    return {
        "run_dir": str(run_dir),
        "output_dir": remote_output_dir,
        "config_path": str(config_path),
        "bundle_path": str(bundle_path),
        "submit_script": submit_script,
    }
