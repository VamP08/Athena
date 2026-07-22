"""
api/registry.py
Thread registry + approval audit trail, backed by SQLite.

The registry answers "what jobs exist and where are they?" (the graph
checkpointer only answers "what is the state of thread X if you already
know X"). The audit table records every human decision — who approved or
revised what, when — which is what turns the review gate into a
compliance feature rather than a UI affordance.
"""

import sqlite3
import threading
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    topic       TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id   TEXT NOT NULL,
    action      TEXT NOT NULL,
    detail      TEXT,
    actor       TEXT,
    created_at  TEXT NOT NULL
);
"""

# Thread lifecycle: running → awaiting_review → (running → …) → completed | failed
VALID_STATUSES = {"running", "awaiting_review", "completed", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Registry:
    """SQLite-backed job registry. Safe for multi-threaded use."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def create(self, thread_id: str, topic: str) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO threads (thread_id, topic, status, created_at, updated_at) "
                "VALUES (?, ?, 'running', ?, ?)",
                (thread_id, topic, now, now),
            )
            self._conn.commit()

    def set_status(self, thread_id: str, status: str, error: str | None = None) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"invalid status: {status}")
        with self._lock:
            self._conn.execute(
                "UPDATE threads SET status = ?, error = ?, updated_at = ? WHERE thread_id = ?",
                (status, error, _now(), thread_id),
            )
            self._conn.commit()

    def get(self, thread_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM threads ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def log_action(self, thread_id: str, action: str, detail: str = "", actor: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (thread_id, action, detail, actor, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (thread_id, action, detail, actor, _now()),
            )
            self._conn.commit()

    def audit_trail(self, thread_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT action, detail, actor, created_at FROM audit_log "
                "WHERE thread_id = ? ORDER BY id",
                (thread_id,),
            ).fetchall()
        return [dict(r) for r in rows]
