import shlex
import subprocess
import os
from pathlib import Path
from typing import Optional

from .settings import (
    SSH_CONNECT_TIMEOUT_SECONDS,
    SSH_HOST,
    SSH_KEY_PATH,
    SSH_KNOWN_HOSTS,
    SSH_PORT,
    SSH_STRICT_HOST_KEY_CHECKING,
    SSH_USER,
    SLURM_BACKEND,
)


def is_ssh_backend() -> bool:
    return SLURM_BACKEND == "ssh"


def _require_ssh_settings() -> None:
    if not SSH_HOST or not SSH_USER:
        raise RuntimeError("SSH backend requires SSH_HOST and SSH_USER.")


def _ssh_base_cmd() -> list[str]:
    _require_ssh_settings()
    cmd = [
        "ssh",
        "-p",
        str(SSH_PORT),
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        "-o",
        f"StrictHostKeyChecking={SSH_STRICT_HOST_KEY_CHECKING}",
    ]
    if SSH_KNOWN_HOSTS:
        cmd += ["-o", f"UserKnownHostsFile={SSH_KNOWN_HOSTS}"]
    if SSH_KEY_PATH:
        cmd += ["-i", SSH_KEY_PATH]
    cmd.append(f"{SSH_USER}@{SSH_HOST}")
    return cmd


def _scp_base_cmd() -> list[str]:
    _require_ssh_settings()
    cmd = [
        "scp",
        "-P",
        str(SSH_PORT),
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        "-o",
        f"StrictHostKeyChecking={SSH_STRICT_HOST_KEY_CHECKING}",
    ]
    if SSH_KNOWN_HOSTS:
        cmd += ["-o", f"UserKnownHostsFile={SSH_KNOWN_HOSTS}"]
    if SSH_KEY_PATH:
        cmd += ["-i", SSH_KEY_PATH]
    return cmd


def run_ssh_command(command: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = _ssh_base_cmd() + [command]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    if is_ssh_backend():
        remote_cmd = " ".join(shlex.quote(item) for item in args)
        return run_ssh_command(remote_cmd)
    return subprocess.run(args, capture_output=True, text=True)


def remote_path_exists(path: str) -> Optional[bool]:
    if not is_ssh_backend():
        return Path(path).exists()
    quoted = shlex.quote(path)
    result = run_ssh_command(f"test -e {quoted}")
    if result.returncode in (0, 1):
        return result.returncode == 0
    return None


def remote_path_readable(path: str) -> Optional[bool]:
    if not is_ssh_backend():
        resolved = Path(path)
        return resolved.exists() and os.access(resolved, os.R_OK)
    quoted = shlex.quote(path)
    result = run_ssh_command(f"test -r {quoted}")
    if result.returncode in (0, 1):
        return result.returncode == 0
    return None


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def scp_upload(local_path: str, remote_dir: str) -> subprocess.CompletedProcess[str]:
    _require_ssh_settings()
    remote_target = f"{SSH_USER}@{SSH_HOST}:{remote_dir}"
    cmd = _scp_base_cmd() + [local_path, remote_target]
    return subprocess.run(cmd, capture_output=True, text=True)
