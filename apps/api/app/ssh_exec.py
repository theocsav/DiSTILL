import shlex
import subprocess
import os
from pathlib import Path
from typing import Optional

from .settings import (
    SCP_TIMEOUT_SECONDS,
    SLURM_COMMAND_TIMEOUT_SECONDS,
    SSH_COMMAND_TIMEOUT_SECONDS,
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


def run_ssh_command(
    command: str,
    check: bool = False,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    cmd = _ssh_base_cmd() + [command]
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            timeout=timeout if timeout is not None else SSH_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ssh client binary not found in API runtime.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ssh command timed out after {exc.timeout}s: {command}") from exc


def run_command(args: list[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess[str]:
    effective_timeout = timeout if timeout is not None else SLURM_COMMAND_TIMEOUT_SECONDS
    if is_ssh_backend():
        remote_cmd = " ".join(shlex.quote(item) for item in args)
        return run_ssh_command(remote_cmd, timeout=effective_timeout)
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=effective_timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {exc.timeout}s: {args[0]}") from exc


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
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=SCP_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise RuntimeError("scp client binary not found in API runtime.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"scp upload timed out after {exc.timeout}s.") from exc
