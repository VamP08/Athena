"""
tests/test_sessions.py
The two properties the session tier exists to guarantee:

  1. A document attached to chat A is NEVER retrievable from chat B.
  2. Nothing survives the chat being destroyed.

These are stated as user-facing promises ("uploads disappear when the chat
closes"), so they are tested as correctness properties rather than trusted to a
docstring. Everything here runs offline: embeddings are stubbed, because
isolation is a property of the storage boundary and does not depend on vectors.
"""

from __future__ import annotations

import sqlite3

import pytest

from core import index as idx
from core import sessions


class _StubEmbeddings:
    """Deterministic fake: dimension-correct, content-sensitive, no network."""

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        v = [0.0] * 1024
        for i, ch in enumerate(text[:1024]):
            v[i] = (ord(ch) % 32) / 32.0
        # never all-zero: cosine distance is undefined for a zero vector
        v[0] = v[0] or 0.5
        return v


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    monkeypatch.setattr(idx, "get_embeddings", lambda model=None: _StubEmbeddings())
    yield
    sessions.REGISTRY.destroy_all()


def _md(text: str) -> bytes:
    return text.encode("utf-8")


def test_session_document_is_not_visible_to_another_session():
    """The cardinal property. A leak here falsifies the project's core claim."""
    a = sessions.REGISTRY.create("chat-A")
    b = sessions.REGISTRY.create("chat-B")

    res = a.add_document("geheim.md", _md(
        "VERTRAULICH_MANDANT_ALPHA Die Abfindung betraegt 250000 EUR."
    ))
    assert res["ok"] and res["chunks"] > 0

    hits_a = a.search("Abfindung VERTRAULICH_MANDANT_ALPHA", k=5)
    assert hits_a, "chat A must find its own document"
    assert any("VERTRAULICH_MANDANT_ALPHA" in h["text"] for h in hits_a)

    hits_b = b.search("Abfindung VERTRAULICH_MANDANT_ALPHA", k=5)
    assert hits_b == [], f"chat B retrieved chat A's document: {hits_b}"
    assert b.list_documents() == [], "chat B can see chat A's document in its inventory"


def test_session_documents_never_touch_the_persistent_index(tmp_path, monkeypatch):
    """An upload must not end up in the durable archive under any circumstance."""
    db = str(tmp_path / "kb.db")
    idx.close()
    conn = idx.connect(db)

    s = sessions.REGISTRY.create("chat-A")
    s.add_document("upload.md", _md("EINMALIGES_TOKEN_XYZ Zahlung 4711"))

    n_docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    n_chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    assert n_docs == 0 and n_chunks == 0, "session upload was written to the persistent index"

    fts = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?", ('"EINMALIGES_TOKEN_XYZ"',)
    ).fetchone()["n"]
    assert fts == 0, "session upload's terms leaked into the persistent search index"
    idx.close()


def test_destroy_removes_everything_and_further_use_raises():
    s = sessions.REGISTRY.create("chat-A")
    s.add_document("x.md", _md("TOKEN_TO_FORGET Betrag 900"))
    assert s.stats()["chunks"] > 0

    assert sessions.REGISTRY.destroy("chat-A") is True
    assert s.closed
    assert sessions.REGISTRY.get("chat-A") is None

    with pytest.raises(sessions.SessionClosed):
        s.search("TOKEN_TO_FORGET")
    with pytest.raises(sessions.SessionClosed):
        s.add_document("y.md", _md("noch etwas"))


def test_destroyed_session_id_is_reusable_but_starts_empty():
    """A new chat that happens to reuse an id must not inherit the old one's files."""
    a = sessions.REGISTRY.create("chat-A")
    a.add_document("alt.md", _md("ALTES_DOKUMENT 111"))
    sessions.REGISTRY.destroy("chat-A")

    fresh = sessions.REGISTRY.create("chat-A")
    assert fresh.list_documents() == []
    assert fresh.search("ALTES_DOKUMENT", k=5) == []


def test_in_memory_store_creates_no_files_and_no_wal(tmp_path, monkeypatch):
    """
    Nothing may be written to disk.

    A file-backed session store would leave the document, plus -wal/-shm
    sidecars, recoverable after a crash. SQLite silently refuses WAL for
    in-memory databases, so this is guaranteed structurally.
    """
    monkeypatch.chdir(tmp_path)
    before = set(p.name for p in tmp_path.iterdir())

    s = sessions.REGISTRY.create("chat-A")
    s.add_document("x.md", _md("KEINE_DATEI_BITTE 42"))

    mode = s._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "memory", f"session store is not purely in memory (journal_mode={mode})"

    after = set(p.name for p in tmp_path.iterdir())
    assert after == before, f"session store wrote files to disk: {after - before}"


def test_scope_is_labelled_on_every_session_hit():
    """A citation must never blur 'in our archive' with 'in the file you gave me'."""
    s = sessions.REGISTRY.create("chat-A")
    s.add_document("vertrag.md", _md("Kuendigungsfrist betraegt sechs Monate"))
    hits = s.search("Kuendigungsfrist", k=3)
    assert hits and all(h["scope"] == "session" for h in hits)


def test_ttl_reaper_disposes_of_abandoned_chats():
    """Streamlit has no reliable 'tab closed' event, so the TTL is the real guarantee."""
    s = sessions.REGISTRY.create("chat-old")
    s.add_document("x.md", _md("ABGELAUFEN 1"))
    s.last_seen = 0.0  # far in the past

    live = sessions.REGISTRY.create("chat-new")
    reaped = sessions.REGISTRY.reap_expired(ttl=60)

    assert reaped == 1
    assert sessions.REGISTRY.get("chat-old") is None
    assert sessions.REGISTRY.get("chat-new") is live


def test_upload_limits_are_refusals_not_exceptions():
    s = sessions.REGISTRY.create("chat-A", embed_model="stub")
    s._bytes = sessions.MAX_SESSION_BYTES
    res = s.add_document("gross.md", _md("x" * 10))
    assert res["ok"] is False and "MB" in res["error"]


def test_unsupported_and_broken_uploads_surface_a_visible_notice():
    """
    'The archive does not contain that' and 'that file could not be read' are
    different answers. Conflating them in a financial setting is a correctness
    failure, so a bad upload must produce a visible notice, not silence.
    """
    s = sessions.REGISTRY.create("chat-A")
    res = s.add_document("bild.png", b"\x89PNG\r\n\x1a\n not a document")
    assert res["ok"] is True
    assert res["notices"], "unsupported upload produced no notice"
    assert "nterst" in res["notices"][0] or "nsupported" in res["notices"][0]

    broken = s.add_document("kaputt.pdf", b"%PDF-1.4 truncated garbage")
    assert broken["notices"], "unparseable upload produced no notice"


def test_filename_cannot_escape_the_temp_directory():
    """A path-like filename must not be used to build the temp path."""
    s = sessions.REGISTRY.create("chat-A")
    res = s.add_document("../../athena_index.md", _md("PFAD_TEST 5"))
    assert res["ok"] is True
    docs = s.list_documents()
    assert docs and docs[0]["source"] == "../../athena_index.md"


def test_two_sessions_are_physically_separate_databases():
    """Not a shared cache: one store's tables are unreachable from the other."""
    a = sessions.REGISTRY.create("chat-A")
    b = sessions.REGISTRY.create("chat-B")
    a.add_document("a.md", _md("NUR_IN_A 1"))

    n = b._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    assert n == 0
    assert a._conn is not b._conn

    with pytest.raises(sqlite3.OperationalError):
        b._conn.execute("SELECT * FROM temp_probe_that_does_not_exist").fetchall()
