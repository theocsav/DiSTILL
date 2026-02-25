from typing import Dict, Optional, Tuple, Union

from .ssh_exec import run_command


def _run_command(cmd: list[str]) -> Optional[str]:
    try:
        result = run_command(cmd)
    except (FileNotFoundError, RuntimeError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _parse_exit_code(value: str) -> Tuple[Optional[int], Optional[int]]:
    if not value:
        return None, None
    parts = value.split(":")
    if len(parts) < 2:
        return None, None
    try:
        exit_code = int(parts[0])
    except ValueError:
        exit_code = None
    try:
        signal = int(parts[1])
    except ValueError:
        signal = None
    return exit_code, signal


def _parse_sacct(job_id: str) -> Optional[Dict[str, Optional[Union[int, str]]]]:
    output = _run_command(
        [
            "sacct",
            "-j",
            job_id,
            "--format=JobID,State,ExitCode,Elapsed,Reason,Submit,Start,End",
            "-n",
            "-P",
        ]
    )
    if not output:
        return None
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    record = None
    for line in lines:
        parts = line.split("|")
        if len(parts) < 8:
            continue
        if parts[0].strip() == job_id:
            record = parts
            break
    if record is None:
        record = lines[0].split("|")
    if len(record) < 8:
        return None
    state = record[1].strip() if record[1] else None
    exit_code_raw = record[2].strip() if record[2] else ""
    exit_code, exit_signal = _parse_exit_code(exit_code_raw)
    elapsed = record[3].strip() if record[3] else None
    reason = record[4].strip() if record[4] else None
    submit = record[5].strip() if record[5] else None
    start = record[6].strip() if record[6] else None
    end = record[7].strip() if record[7] else None
    return {
        "state": state,
        "reason": reason or None,
        "exit_code": exit_code,
        "exit_signal": exit_signal,
        "elapsed": elapsed or None,
        "submitted_at": submit or None,
        "started_at": start or None,
        "finished_at": end or None,
    }


def _parse_squeue(job_id: str) -> Optional[Dict[str, Optional[Union[int, str]]]]:
    output = _run_command(["squeue", "-j", job_id, "-h", "-o", "%T|%r"])
    if not output:
        return None
    line = output.splitlines()[0]
    parts = line.split("|", 1)
    state = parts[0].strip() if parts else None
    reason = parts[1].strip() if len(parts) > 1 else None
    return {
        "state": state,
        "reason": reason or None,
        "exit_code": None,
        "exit_signal": None,
        "elapsed": None,
        "submitted_at": None,
        "started_at": None,
        "finished_at": None,
    }


def get_job_info(job_id: str) -> Optional[Dict[str, Optional[Union[int, str]]]]:
    info = _parse_sacct(job_id)
    if info:
        return info
    return _parse_squeue(job_id)


def get_job_state(job_id: str) -> Optional[str]:
    info = get_job_info(job_id)
    if not info:
        return None
    return info.get("state")


def cancel_job(job_id: str) -> bool:
    try:
        result = run_command(["scancel", job_id])
    except (FileNotFoundError, RuntimeError):
        return False
    return result.returncode == 0
