#!/usr/bin/env bash
set -euo pipefail

# Required environment:
#   API_BASE=https://api.example.com
#   QUEUE_POLLER_TOKEN=<shared token>
# Optional:
#   POLLER_RUNS_DIR=/blue/<group>/<user>/nicherunner/runs

API_BASE="${API_BASE:-}"
QUEUE_POLLER_TOKEN="${QUEUE_POLLER_TOKEN:-}"
POLLER_RUNS_DIR="${POLLER_RUNS_DIR:-/blue/kejun.huang/vasco.hinostroza/nicherunner/runs}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-200}"
POLLER_API_RETRIES="${POLLER_API_RETRIES:-4}"
POLLER_RETRY_DELAY_SECONDS="${POLLER_RETRY_DELAY_SECONDS:-3}"

if [[ -z "$API_BASE" || -z "$QUEUE_POLLER_TOKEN" ]]; then
  echo "API_BASE and QUEUE_POLLER_TOKEN are required." >&2
  exit 2
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

curl_retry() {
  curl --retry "$POLLER_API_RETRIES" \
    --retry-delay "$POLLER_RETRY_DELAY_SECONDS" \
    --retry-connrefused \
    --connect-timeout 15 \
    --max-time 120 \
    "$@"
}

claim_json="$tmpdir/claim.json"
curl_retry -fsS -X POST \
  -H "X-Queue-Token: $QUEUE_POLLER_TOKEN" \
  "$API_BASE/queue/claim" > "$claim_json"

job_json="$(python3 - <<'PY' "$claim_json"
import json, sys
data=json.load(open(sys.argv[1], encoding="utf-8"))
if not data.get("ok", True):
    raise SystemExit(data.get("error") or "queue claim failed")
job=data.get("job")
print(json.dumps(job) if job else "null")
PY
)"

if [[ "$job_json" == "null" ]]; then
  # No queued job right now.
  exit 0
fi

run_id="$(python3 - <<'PY' "$job_json"
import json, sys
print(json.loads(sys.argv[1])["run_id"])
PY
)"
run_name="$(python3 - <<'PY' "$job_json"
import json, sys
print(json.loads(sys.argv[1])["run_name"])
PY
)"
claim_id="$(python3 - <<'PY' "$job_json"
import json, sys
print(json.loads(sys.argv[1])["claim_id"])
PY
)"
bundle_url="$(python3 - <<'PY' "$job_json"
import json, sys
print(json.loads(sys.argv[1])["bundle_url"])
PY
)"

bundle_url="${bundle_url/http:\/\//https:\/\/}"

target_dir="$POLLER_RUNS_DIR/$run_name"
mkdir -p "$target_dir"

bundle_path="$tmpdir/run_bundle.tar.gz"
curl_retry -fsS \
  -H "X-Queue-Token: $QUEUE_POLLER_TOKEN" \
  "$bundle_url" -o "$bundle_path"
tar -xzf "$bundle_path" -C "$target_dir"
chmod +x "$target_dir/run.sh" "$target_dir/submit.sh" || true

submit_out="$tmpdir/sbatch.out"
if ! sbatch "$target_dir/submit.sh" > "$submit_out" 2>&1; then
  err_msg="$(tr '\n' ' ' < "$submit_out" | sed 's/"/\\"/g')"
  curl_retry -fsS -X POST \
    -H "X-Queue-Token: $QUEUE_POLLER_TOKEN" \
    -F "run_id=$run_id" \
    -F "status=error" \
    -F "message=Poller submission failed: $err_msg" \
    "$API_BASE/queue/report-status" >/dev/null
  exit 1
fi

job_id="$(python3 - <<'PY' "$submit_out"
import re, sys
text=open(sys.argv[1], encoding="utf-8").read()
m=re.search(r"Submitted batch job (\d+)", text)
print(m.group(1) if m else "")
PY
)"

if [[ -z "$job_id" ]]; then
  out_msg="$(tr '\n' ' ' < "$submit_out" | sed 's/"/\\"/g')"
  curl_retry -fsS -X POST \
    -H "X-Queue-Token: $QUEUE_POLLER_TOKEN" \
    -F "run_id=$run_id" \
    -F "status=error" \
    -F "message=Poller could not parse sbatch job id: $out_msg" \
    "$API_BASE/queue/report-status" >/dev/null
  exit 1
fi

curl_retry -fsS -X POST \
  -H "X-Queue-Token: $QUEUE_POLLER_TOKEN" \
  -F "run_id=$run_id" \
  -F "claim_id=$claim_id" \
  -F "slurm_job_id=$job_id" \
  -F "message=Submitted by HPG poller" \
  "$API_BASE/queue/report-submission" >/dev/null

# Lightweight status sync for active jobs.
active_json="$tmpdir/active.json"
curl_retry -fsS -H "X-Queue-Token: $QUEUE_POLLER_TOKEN" "$API_BASE/queue/active" > "$active_json"
python3 - <<'PY' "$active_json" "$API_BASE" "$QUEUE_POLLER_TOKEN" "$POLLER_RUNS_DIR" "$LOG_TAIL_LINES"
from collections import deque
from pathlib import Path
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request

active = json.load(open(sys.argv[1], encoding="utf-8")).get("items", [])
api_base = sys.argv[2].rstrip("/")
token = sys.argv[3]
runs_dir = Path(sys.argv[4])
log_tail_lines = max(1, int(sys.argv[5]))


def map_state(state: str) -> str:
    s = (state or "").upper()
    if s in {"PENDING", "CONFIGURING"}:
        return "queued"
    if s in {"RUNNING", "COMPLETING"}:
        return "running"
    if s in {"COMPLETED"}:
        return "succeeded"
    if s in {"CANCELLED", "CANCELLED+"}:
        return "canceled"
    if s in {"FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}:
        return "failed"
    return "unknown"


def get_info(job_id: str):
    sacct = subprocess.run(
        [
            "sacct",
            "-j",
            job_id,
            "--format=JobID,State,ExitCode,Elapsed,Reason,Submit,Start,End",
            "-n",
            "-P",
        ],
        capture_output=True,
        text=True,
    )
    if sacct.returncode == 0 and sacct.stdout.strip():
        line = None
        for row in sacct.stdout.splitlines():
            parts = row.split("|")
            if len(parts) >= 8 and parts[0].strip() == job_id:
                line = parts
                break
        if line is None:
            row = sacct.stdout.splitlines()[0]
            line = row.split("|")
        if len(line) >= 8:
            return {
                "state": line[1].strip(),
                "reason": line[4].strip(),
                "elapsed": line[3].strip(),
                "started_at": line[6].strip(),
                "finished_at": line[7].strip(),
            }
    squeue = subprocess.run(["squeue", "-j", job_id, "-h", "-o", "%T|%r"], capture_output=True, text=True)
    if squeue.returncode == 0 and squeue.stdout.strip():
        first = squeue.stdout.splitlines()[0]
        parts = first.split("|", 1)
        return {
            "state": parts[0].strip(),
            "reason": parts[1].strip() if len(parts) > 1 else "",
            "elapsed": "",
            "started_at": "",
            "finished_at": "",
        }
    return None


def tail_lines(path: Path, max_lines: int) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, max_lines)).strip()


def collect_log_tail(run_name: str, output_dir: str) -> str:
    candidates_dirs = []
    if output_dir:
        candidates_dirs.append(Path(output_dir) / "logs")
    candidates_dirs.append(runs_dir / str(run_name) / "outputs" / "logs")
    candidates_dirs.append(runs_dir / str(run_name) / "output" / "logs")

    logs_dir = None
    for candidate in candidates_dirs:
        if candidate.exists() and candidate.is_dir():
            logs_dir = candidate
            break
    if logs_dir is None:
        return ""

    candidates = []
    preferred = ["cell2loc_nmf.err", "cell2loc_nmf.out"]
    seen = set()
    for name in preferred:
        path = logs_dir / name
        if path.exists() and path.is_file():
            candidates.append(path)
            seen.add(path.name)

    others = sorted(logs_dir.glob("*.err")) + sorted(logs_dir.glob("*.out"))
    for path in others:
        if path.name in seen:
            continue
        candidates.append(path)
        seen.add(path.name)

    if not candidates:
        return ""

    sections = []
    for path in candidates[:2]:
        try:
            content = tail_lines(path, log_tail_lines)
        except Exception:
            continue
        if content:
            sections.append(f"== {path.name} ==\n{content}")
    if not sections:
        return ""
    message = "\n\n".join(sections)
    return message[:12000]


for item in active:
    run_id = item.get("run_id")
    run_name = str(item.get("run_name") or "").strip()
    job_id = str(item.get("job_id") or "").strip()
    if not run_id or not job_id:
        continue
    info = get_info(job_id)
    if not info:
        continue
    output_dir = str(item.get("output_dir") or "").strip()
    payload = {
        "run_id": str(run_id),
        "status": map_state(info.get("state", "")),
        "slurm_state": info.get("state", ""),
        "slurm_reason": info.get("reason", ""),
        "slurm_elapsed": info.get("elapsed", ""),
        "started_at": info.get("started_at", ""),
        "finished_at": info.get("finished_at", ""),
        "message": collect_log_tail(run_name, output_dir) if run_name else "",
    }
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{api_base}/queue/report-status",
        data=data,
        headers={"X-Queue-Token": token},
        method="POST",
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=20):
                break
        except Exception:
            if attempt == 3:
                break
            time.sleep(2 ** attempt)
PY
