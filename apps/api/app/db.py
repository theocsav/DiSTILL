import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .settings import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT NOT NULL,
                status TEXT NOT NULL,
                stage TEXT,
                run_dir TEXT,
                output_dir TEXT,
                config_path TEXT,
                job_id TEXT,
                slurm_state TEXT,
                slurm_reason TEXT,
                slurm_exit_code INTEGER,
                slurm_exit_signal INTEGER,
                slurm_elapsed TEXT,
                submitted_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_queue (
                run_id INTEGER PRIMARY KEY,
                state TEXT NOT NULL,
                submit INTEGER NOT NULL DEFAULT 1,
                claim_id TEXT,
                lease_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        missing = {
            "slurm_state": "TEXT",
            "slurm_reason": "TEXT",
            "slurm_exit_code": "INTEGER",
            "slurm_exit_signal": "INTEGER",
            "slurm_elapsed": "TEXT",
            "submitted_at": "TEXT",
            "started_at": "TEXT",
            "finished_at": "TEXT",
        }
        for name, column_type in missing.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {name} {column_type}")

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(run_queue)").fetchall()}
        if "submit" not in columns:
            conn.execute("ALTER TABLE run_queue ADD COLUMN submit INTEGER NOT NULL DEFAULT 1")
        if "claim_id" not in columns:
            conn.execute("ALTER TABLE run_queue ADD COLUMN claim_id TEXT")
        if "lease_expires_at" not in columns:
            conn.execute("ALTER TABLE run_queue ADD COLUMN lease_expires_at TEXT")
        conn.commit()


def create_run(run_name: str, status: str = "created") -> int:
    created = utc_now()
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO runs (run_name, status, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_name, status, created, created),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_run(run_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = utc_now()
    columns = ", ".join([f"{key} = ?" for key in fields.keys()])
    values = list(fields.values()) + [run_id]
    with get_conn() as conn:
        conn.execute(f"UPDATE runs SET {columns} WHERE id = ?", values)
        conn.commit()


def fetch_run(run_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def list_runs() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM runs ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [dict(row) for row in rows]


def enqueue_run(run_id: int, submit: bool = True) -> None:
    created = utc_now()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO run_queue (run_id, state, submit, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, "queued", 1 if submit else 0, created, created),
        )
        conn.commit()


def next_queued_run() -> Optional[Dict[str, int]]:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT run_id, submit FROM run_queue WHERE state = ? ORDER BY created_at ASC LIMIT 1",
            ("queued",),
        )
        row = cur.fetchone()
    return {"run_id": int(row["run_id"]), "submit": int(row["submit"])} if row else None


def mark_run_running(run_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE run_queue SET state = ?, updated_at = ? WHERE run_id = ?",
            ("running", utc_now(), run_id),
        )
        conn.commit()


def remove_from_queue(run_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM run_queue WHERE run_id = ?", (run_id,))
        conn.commit()


def claim_next_run(lease_seconds: int) -> Optional[Dict[str, Any]]:
    now = utc_now()
    lease_until = (datetime.now(timezone.utc) + timedelta(seconds=max(lease_seconds, 30))).isoformat()
    claim_id = uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """
            SELECT run_id, submit
            FROM run_queue
            WHERE state = 'queued'
               OR (state = 'claimed' AND lease_expires_at IS NOT NULL AND lease_expires_at < ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (now,),
        )
        row = cur.fetchone()
        if not row:
            conn.commit()
            return None
        run_id = int(row["run_id"])
        submit = int(row["submit"])
        conn.execute(
            """
            UPDATE run_queue
            SET state = 'claimed', claim_id = ?, lease_expires_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (claim_id, lease_until, now, run_id),
        )
        conn.commit()
        return {
            "run_id": run_id,
            "submit": submit,
            "claim_id": claim_id,
            "lease_expires_at": lease_until,
        }


def complete_claim(run_id: int, claim_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT claim_id FROM run_queue WHERE run_id = ? AND state = 'claimed'",
            (run_id,),
        )
        row = cur.fetchone()
        if not row or str(row["claim_id"] or "") != claim_id:
            return False
        conn.execute("DELETE FROM run_queue WHERE run_id = ?", (run_id,))
        conn.commit()
        return True


def release_claim(run_id: int, claim_id: str, state: str = "queued") -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "SELECT claim_id FROM run_queue WHERE run_id = ? AND state = 'claimed'",
            (run_id,),
        )
        row = cur.fetchone()
        if not row or str(row["claim_id"] or "") != claim_id:
            return False
        conn.execute(
            """
            UPDATE run_queue
            SET state = ?, claim_id = NULL, lease_expires_at = NULL, updated_at = ?
            WHERE run_id = ?
            """,
            (state, utc_now(), run_id),
        )
        conn.commit()
        return True


def fetch_queue_item(run_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM run_queue WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
    return dict(row) if row else None
