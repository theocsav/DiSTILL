import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .settings import (
    PREFLIGHT_SLURM_POLL_SECONDS,
    PREFLIGHT_SLURM_TIMEOUT_SECONDS,
    RUNS_DIR,
)
from .slurm import get_job_state
from .ssh_exec import is_ssh_backend


TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAIL = {"CANCELLED", "CANCELLED+", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


def run_slurm_preflight(config: Dict[str, Any]) -> Dict[str, Any]:
    if is_ssh_backend():
        return {
            "ok": False,
            "error": "PREFLIGHT_SLURM_FALLBACK is not supported with SLURM_BACKEND=ssh. Install join-key dependencies on the API host.",
        }

    base_slurm = config.get("slurm", {})
    preflight_override = config.get("preflight_slurm", {})
    slurm_cfg = dict(base_slurm)
    slurm_cfg.update(preflight_override)
    conda_env = slurm_cfg.get("conda_env")
    if not conda_env:
        return {"ok": False, "error": "preflight_slurm requires slurm.conda_env"}

    version_check = subprocess.run(["sbatch", "--version"], capture_output=True, text=True)
    if version_check.returncode != 0:
        return {"ok": False, "error": "sbatch is not available on the API host."}

    preflight_dir = RUNS_DIR / "_preflight" / uuid.uuid4().hex
    preflight_dir.mkdir(parents=True, exist_ok=True)
    config_path = preflight_dir / "config.json"
    result_path = preflight_dir / "result.json"
    script_path = preflight_dir / "preflight.py"
    submit_path = preflight_dir / "submit.sh"

    config_payload = {
        "cosmx_h5ad_path": config.get("cosmx_h5ad_path"),
        "cell_metadata_path": config.get("cell_metadata_path"),
        "join_key_strategy": config.get("join_key_strategy", "auto"),
        "join_key_delimiter": config.get("join_key_delimiter", "__"),
    }
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    script_path.write_text(
        _build_preflight_script(str(config_path), str(result_path)),
        encoding="utf-8",
    )

    submit_path.write_text(
        _build_submit_script(str(script_path), conda_env, slurm_cfg, str(preflight_dir)),
        encoding="utf-8",
    )

    submit = subprocess.run(["sbatch", str(submit_path)], capture_output=True, text=True)
    if submit.returncode != 0:
        return {"ok": False, "error": submit.stderr or submit.stdout or "sbatch failed"}
    job_id = _parse_job_id(submit.stdout)
    if not job_id:
        return {"ok": False, "error": f"Could not parse job id from sbatch output: {submit.stdout}"}

    start = time.time()
    while True:
        state = get_job_state(job_id)
        if state:
            if state in TERMINAL_SUCCESS:
                break
            if state in TERMINAL_FAIL:
                return {"ok": False, "error": f"Preflight job failed with state {state}."}
        if time.time() - start > PREFLIGHT_SLURM_TIMEOUT_SECONDS:
            return {"ok": False, "error": "Preflight job timed out."}
        time.sleep(PREFLIGHT_SLURM_POLL_SECONDS)

    if not result_path.exists():
        return {"ok": False, "error": "Preflight job completed but result.json is missing."}
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return {"ok": True, "result": result}


def _parse_job_id(output: str) -> Optional[str]:
    for token in output.split():
        if token.isdigit():
            return token
    return None


def _build_submit_script(script_path: str, conda_env: str, slurm_cfg: Dict[str, Any], run_dir: str) -> str:
    time_limit = slurm_cfg.get("time", "00:10:00")
    mem = slurm_cfg.get("mem", "8gb")
    cpus = slurm_cfg.get("cpus_per_task", 1)
    account = slurm_cfg.get("account")
    partition = slurm_cfg.get("partition")
    qos = slurm_cfg.get("qos")
    output_path = slurm_cfg.get("output") or f"{run_dir}/preflight.out"
    error_path = slurm_cfg.get("error") or f"{run_dir}/preflight.err"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=preflight",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={error_path}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        "#SBATCH --nodes=1",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={mem}",
    ]
    if account:
        lines.append(f"#SBATCH --account={account}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    lines += [
        "",
        "set -euo pipefail",
        "module load conda",
        f"conda activate {conda_env}",
        f"python {script_path}",
        "",
    ]
    return "\n".join(lines)


def _build_preflight_script(config_path: str, result_path: str) -> str:
    return f"""#!/usr/bin/env python3
import json
from pathlib import Path

import anndata as ad
import pandas as pd


def resolve_strategy(obs_cols, meta_cols):
    if "unique_cell_id" in obs_cols and "unique_cell_id" in meta_cols:
        return "unique_cell_id"
    if "fov" in obs_cols and "cell_ID" in obs_cols and "fov" in meta_cols and "cell_ID" in meta_cols:
        return "fov_cell_id"
    raise RuntimeError("No compatible join keys found (unique_cell_id or fov+cell_ID).")


def build_keys(frame, strategy, delimiter):
    if strategy == "unique_cell_id":
        return frame["unique_cell_id"].astype(str)
    return frame["fov"].astype(str) + delimiter + frame["cell_ID"].astype(str)


def main():
    config = json.loads(Path({json.dumps(config_path)}).read_text())
    h5ad_path = config.get("cosmx_h5ad_path")
    metadata_path = config.get("cell_metadata_path")
    if not h5ad_path or not metadata_path:
        raise RuntimeError("cosmx_h5ad_path and cell_metadata_path are required.")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    if getattr(adata, "file", None) is not None:
        adata.file.close()
    meta_header = pd.read_csv(metadata_path, nrows=0).columns.tolist()

    strategy = config.get("join_key_strategy", "auto")
    if strategy == "auto":
        strategy = resolve_strategy(list(obs.columns), meta_header)

    if strategy == "unique_cell_id":
        meta = pd.read_csv(metadata_path, usecols=["unique_cell_id"])
    else:
        meta = pd.read_csv(metadata_path, usecols=["fov", "cell_ID"])

    delimiter = config.get("join_key_delimiter", "__")
    obs_keys = set(build_keys(obs, strategy, delimiter).tolist())
    meta_keys = set(build_keys(meta, strategy, delimiter).tolist())

    matched = len(obs_keys & meta_keys)
    missing = len(obs_keys - meta_keys)
    extra = len(meta_keys - obs_keys)
    obs_total = len(obs_keys)
    meta_total = len(meta_keys)
    result = {{
        "strategy": "unique_cell_id" if strategy == "unique_cell_id" else "fov+cell_ID",
        "obs_total": obs_total,
        "metadata_total": meta_total,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "missing_fraction": round((missing / obs_total) if obs_total else 0.0, 6),
        "extra_fraction": round((extra / meta_total) if meta_total else 0.0, 6),
    }}
    Path({json.dumps(result_path)}).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
"""
