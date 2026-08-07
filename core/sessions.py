"""
core/sessions.py
Ephemeral, per-chat document storage — the second of Athena's two document tiers.

  KNOWLEDGE BASE (core/index.py)   a connected folder, embedded once, persistent
                                   across restarts, shared by every chat.
  SESSION DOCUMENTS (this module)  files a user attaches inside one chat. They
                                   exist only for that chat and must be gone when
                                   it ends.

WHY A PRIVATE :memory: DATABASE, AND NOT A `scope` COLUMN IN THE SHARED INDEX

The simplest implementation — one index file with a `session_id` column and a
WHERE clause — was rejected, and not on style grounds. Two measured facts killed it:

  1. SQLite's `secure_delete` defaults to OFF. Writing rows to a file-backed
     database, deleting them, committing and checkpointing left the plaintext
     recoverable in the raw file bytes. Under the shared-file design that file is
     the customer's durable archive — the one they back up and hand to an
     auditor. "We deleted your document" would be provably false with `strings`.
  2. Isolation would rest on never forgetting a WHERE clause. One missed
     predicate leaks one chat's upload into another's answers, permanently and
     silently. For a product whose entire thesis is data protection, that is the
     wrong place to put the guarantee.

A bare `sqlite3.connect(":memory:")` handle is reachable only through the Python
object that owns it. There is no path, no name, and no shared cache, so another
chat cannot address it even in principle — isolation becomes a property of the
object graph rather than of query discipline. Two separate `:memory:` connections
are genuinely separate databases (verified: the second cannot see the first's
tables). The pages die with the process, so a `kill -9` leaves nothing on disk.
`PRAGMA journal_mode` is silently refused for in-memory databases, which means
the `-wal`/`-shm` sidecar files this project has already been bitten by cannot be
produced here by construction rather than by care.

WHY IT REUSES core/index.py's SCHEMA AND RANKING VERBATIM

Session and archive retrieval must rank identically, otherwise the grounding-eval
numbers measured against the archive say nothing about the session tier and every
future ranking change has to be made twice. Only the connection differs.

WHAT THIS MODULE DOES NOT DO

It does not scrub the LangGraph checkpointer. Retrieved passages also flow into
graph state and message history, so destroying a session store is necessary but
NOT sufficient for "the document is gone" — see purge_thread_state() and the
honest limits recorded in docs/DECISIONS.md.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid

from . import index as idx
from .documents import ParsedElement, parse_file_bytes

# A chat with no activity for this long is reaped. Streamlit has no reliable
# "tab closed" event — a browser tab can vanish without any signal — so a TTL is
# the only mechanism that actually guarantees eventual disposal.
SESSION_TTL_SECONDS = int(os.getenv("ATHENA_SESSION_TTL", "3600"))

# Upload ceilings. A session store is RAM, so it is bounded on purpose; the user
# is told when a limit bites rather than being silently given partial retrieval.
MAX_SESSION_FILES = int(os.getenv("ATHENA_SESSION_MAX_FILES", "20"))
MAX_SESSION_BYTES = int(os.getenv("ATHENA_SESSION_MAX_BYTES", str(50 * 1024 * 1024)))
MAX_SESSION_CHUNKS = int(os.getenv("ATHENA_SESSION_MAX_CHUNKS", "4000"))


class SessionClosed(RuntimeError):
    """Raised when a destroyed session is used again."""


class SessionStore:
    """
    One chat's private document store.

    Thread-safe: the FastAPI layer runs graph work on a ThreadPoolExecutor, so
    several nodes may query one chat's store concurrently. The lock is per-store,
    never the global index lock, so one chat's ingest cannot block another's.
    """

    def __init__(self, session_id: str, embed_model: str | None = None):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_seen = self.created_at
        self.embed_model = embed_model or idx.DEFAULT_EMBED_MODEL
        self._lock = threading.Lock()
        self._closed = False
        self._files: dict[str, dict] = {}
        self._bytes = 0
        # Threads this chat's documents have been used by. Needed at teardown:
        # the passages also live in the LangGraph checkpointer, keyed by
        # thread_id, and app.py mints a NEW thread_id per question — so one chat
        # spans many threads and all of them must be scrubbed.
        self.thread_ids: set[str] = set()

        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        import sqlite_vec

        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        idx._init_schema(self._conn)

        # The archive records the model it was embedded with; the session store
        # must record the same one. If they drifted, one answer would mix
        # passages ranked in two different vector spaces.
        idx._meta_set(self._conn, "embed_model", self.embed_model)
        self._conn.commit()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _check(self) -> None:
        if self._closed:
            raise SessionClosed(f"session {self.session_id} has been destroyed")

    def destroy(self) -> None:
        """
        Drop the database. Idempotent.

        Closing an in-memory connection frees its pages; there is no file to
        unlink and nothing to overwrite.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            except Exception:
                pass
            self._files.clear()
            self._bytes = 0

    @property
    def closed(self) -> bool:
        return self._closed

    # ── ingest ───────────────────────────────────────────────────────────────

    def add_document(self, filename: str, data: bytes, progress=None) -> dict:
        """
        Parse, embed and index one uploaded file into this chat only.

        Returns a summary dict. Refusals (too large, too many files, unsupported
        type) are returned as a result with `ok=False`, never raised, so the UI
        can show the reason next to the file.
        """
        self._check()

        if len(self._files) >= MAX_SESSION_FILES:
            return {"ok": False, "filename": filename,
                    "error": f"Limit of {MAX_SESSION_FILES} attached files reached."}
        if self._bytes + len(data) > MAX_SESSION_BYTES:
            return {"ok": False, "filename": filename,
                    "error": f"Attachments exceed {MAX_SESSION_BYTES // (1024 * 1024)} MB for this chat."}

        doc_id, elements = parse_file_bytes(filename, data)

        notices = [e for e in elements if e.kind == "notice"]
        indexable = [e for e in elements if e.kind != "notice" and e.text.strip()]

        with self._lock:
            self._check()
            n_existing = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        room = max(0, MAX_SESSION_CHUNKS - n_existing)
        truncated = len(indexable) > room
        if truncated:
            indexable = indexable[:room]

        vectors: list[list[float] | None] = [None] * len(indexable)
        if indexable:
            emb = idx.get_embeddings(self.embed_model)
            batch = 32
            for i in range(0, len(indexable), batch):
                part = indexable[i : i + batch]
                try:
                    vs = emb.embed_documents([p.text for p in part])
                except Exception as e:  # noqa: BLE001
                    if progress:
                        progress(f"embedding failed: {e}")
                    vs = [None] * len(part)
                for j, v in enumerate(vs):
                    vectors[i + j] = v
                if progress:
                    progress(f"embedded {min(i + batch, len(indexable))}/{len(indexable)}")

        doc_type = idx._infer_doc_type(filename, indexable[0].text if indexable else "")
        year = idx._infer_year(filename, indexable[0].text if indexable else "")

        to_store: list[ParsedElement] = indexable + notices
        store_vectors = vectors + [None] * len(notices)

        with self._lock:
            self._check()
            self._conn.execute(
                "INSERT OR REPLACE INTO documents"
                "(doc_id,source,rel_path,doc_type,year,n_elements,parse_status,parse_note,ingested_at)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (doc_id, filename, "", doc_type, year, len(to_store),
                 "partial" if notices else "ok", None, time.time()),
            )
            for e, vec in zip(to_store, store_vectors):
                self._conn.execute(
                    "INSERT INTO chunks"
                    "(doc_id,element_id,source,kind,text,locator,page,sheet,row,table_id,"
                    " lang_hint,doc_type,year,embedding)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (doc_id, e.element_id, filename, e.kind, e.text, e.locator,
                     e.page, e.sheet, e.row, e.table_id, e.lang_hint,
                     doc_type, year, idx._serialize(vec) if vec else None),
                )
            self._conn.commit()
            self._files[doc_id] = {"filename": filename, "chunks": len(to_store)}
            self._bytes += len(data)
            self.last_seen = time.time()

        return {
            "ok": True,
            "filename": filename,
            "doc_id": doc_id,
            "chunks": len(to_store),
            "notices": [n.text for n in notices],
            "truncated": truncated,
            "doc_type": doc_type,
            "year": year,
        }

    def remove_document(self, doc_id: str) -> bool:
        self._check()
        with self._lock:
            cur = self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self._conn.commit()
            self._files.pop(doc_id, None)
            return cur.rowcount > 0

    # ── query ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 4, doc_type: str = "", year: str = "") -> list[dict]:
        """Same ranking as the archive, over this chat's documents only."""
        self._check()
        self.last_seen = time.time()
        hits = idx.search_conn(self._conn, self._lock, query, k=k, doc_type=doc_type, year=year)
        for h in hits:
            h["scope"] = "session"
        return hits

    def list_documents(self) -> list[dict]:
        self._check()
        self.last_seen = time.time()
        return idx.list_documents_conn(self._conn, self._lock)

    def stats(self) -> dict:
        if self._closed:
            return {"documents": 0, "chunks": 0, "closed": True}
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return {
            "session_id": self.session_id,
            "documents": docs,
            "chunks": chunks,
            "bytes": self._bytes,
            "embed_model": self.embed_model,
            "closed": False,
        }


class SessionRegistry:
    """
    Process-global map of session_id -> SessionStore.

    Lock ordering, which must not be violated: registry lock, then store lock.
    The registry lock is only ever held for a dict operation — never across an
    embedding call or a query — so ingesting into one chat cannot block another
    chat from being looked up.
    """

    def __init__(self):
        self._stores: dict[str, SessionStore] = {}
        self._lock = threading.RLock()

    def create(self, session_id: str | None = None, embed_model: str | None = None) -> SessionStore:
        sid = session_id or f"s-{uuid.uuid4().hex[:16]}"
        with self._lock:
            existing = self._stores.get(sid)
            if existing is not None and not existing.closed:
                return existing
            store = SessionStore(sid, embed_model=embed_model)
            self._stores[sid] = store
            return store

    def get(self, session_id: str | None) -> SessionStore | None:
        if not session_id:
            return None
        with self._lock:
            store = self._stores.get(session_id)
        if store is None or store.closed:
            return None
        store.last_seen = time.time()
        return store

    def destroy(self, session_id: str) -> bool:
        with self._lock:
            store = self._stores.pop(session_id, None)
        if store is None:
            return False
        store.destroy()
        return True

    def destroy_all(self) -> int:
        with self._lock:
            stores = list(self._stores.values())
            self._stores.clear()
        for s in stores:
            s.destroy()
        return len(stores)

    def reap_expired(self, ttl: int | None = None, now: float | None = None) -> int:
        """
        Dispose of chats that stopped being used.

        Called opportunistically on create/get rather than from a background
        thread: a daemon thread would keep running under Streamlit's rerun model
        and in the API's worker pool with no clean shutdown hook, and an
        abandoned chat costs only RAM until the process exits anyway.
        """
        ttl = SESSION_TTL_SECONDS if ttl is None else ttl
        now = time.time() if now is None else now
        with self._lock:
            dead = [sid for sid, s in self._stores.items() if now - s.last_seen > ttl]
            stores = [self._stores.pop(sid) for sid in dead]
        for s in stores:
            s.destroy()
        return len(stores)

    def active(self) -> list[dict]:
        with self._lock:
            return [s.stats() for s in self._stores.values() if not s.closed]


REGISTRY = SessionRegistry()


# ── checkpoint hygiene ───────────────────────────────────────────────────────
#
# Destroying the session store is NECESSARY BUT NOT SUFFICIENT.
#
# Retrieved passages do not stay inside the store: document_search's output
# becomes a ToolMessage in `messages`, and selected passages become
# `retrieved_chunks` in graph state. LangGraph checkpoints all of it at every
# superstep. With ATHENA_CHECKPOINTER=sqlite that means verbatim uploaded text is
# written into athena.db — and, because SqliteSaver puts that file in WAL mode,
# into athena.db-wal as well. Deleting the store while leaving the checkpoint
# would make "the document disappears when the chat closes" false at the byte
# level, which is exactly the kind of claim this project cannot afford to get
# wrong.
#
# Two ordering details that were measured, not assumed:
#   * DELETE alone leaves the plaintext recoverable, because SQLite's
#     secure_delete defaults to OFF. It must be switched on for the connection
#     before the delete.
#   * VACUUM must be followed by a TRAILING wal_checkpoint(TRUNCATE). Doing the
#     checkpoint first and vacuuming after leaves copies in the file.


def bind_thread(session_id: str, thread_id: str) -> None:
    """Record that `thread_id` ran with this chat's documents attached."""
    store = REGISTRY.get(session_id)
    if store is not None:
        store.thread_ids.add(thread_id)


def purge_thread_state(thread_ids, checkpointer=None) -> dict:
    """
    Remove checkpointed state for the given threads, best effort.

    Returns a report rather than raising: teardown must not fail loudly and leave
    a half-deleted chat, but it must also never claim success it cannot back up —
    so the report says exactly what was and was not scrubbed.
    """
    thread_ids = [t for t in (thread_ids or []) if t]
    report = {"threads": len(thread_ids), "sqlite_rows": 0, "in_memory": 0, "errors": []}
    if not thread_ids:
        return report

    kind = os.getenv("ATHENA_CHECKPOINTER", "memory").lower()

    if kind == "sqlite":
        path = os.getenv("ATHENA_DB_PATH", "athena.db")
        try:
            conn = sqlite3.connect(path)
            conn.execute("PRAGMA secure_delete=ON")
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            for table in tables:
                cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
                if "thread_id" not in cols:
                    continue
                for tid in thread_ids:
                    cur = conn.execute(f"DELETE FROM {table} WHERE thread_id=?", (tid,))
                    report["sqlite_rows"] += cur.rowcount if cur.rowcount > 0 else 0
            conn.commit()
            conn.execute("VACUUM")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"sqlite checkpoint scrub failed: {type(e).__name__}: {e}")
    else:
        # InMemorySaver is the DEFAULT. Without this branch, "it disappears when
        # the chat closes" would be false in the configuration most runs use:
        # the passages would stay alive in process memory, keyed by old
        # thread_ids, until the process exits.
        if checkpointer is not None:
            for tid in thread_ids:
                try:
                    if hasattr(checkpointer, "delete_thread"):
                        checkpointer.delete_thread(tid)
                        report["in_memory"] += 1
                except Exception as e:  # noqa: BLE001
                    report["errors"].append(f"in-memory purge failed for {tid}: {e}")
    return report


def end_session(session_id: str, checkpointer=None) -> dict:
    """
    Full teardown for one chat: drop its documents AND its checkpointed state.

    This is the only call that makes the user-facing promise true. Destroying the
    store alone leaves verbatim passages in the checkpointer.
    """
    store = REGISTRY.get(session_id)
    threads = sorted(store.thread_ids) if store is not None else []
    report = purge_thread_state(threads, checkpointer=checkpointer)
    report["store_destroyed"] = REGISTRY.destroy(session_id)
    return report


def get_or_create(session_id: str | None, embed_model: str | None = None) -> SessionStore:
    REGISTRY.reap_expired()
    store = REGISTRY.get(session_id)
    if store is not None:
        return store
    return REGISTRY.create(session_id, embed_model=embed_model)
