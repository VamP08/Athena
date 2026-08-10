"""
tests/test_api_documents.py
The document endpoints, including the isolation property across HTTP.

The API is where isolation is most easily lost: one process, a worker pool, and
several callers. A chat's attachments must be reachable through its own
session_id and no other.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core import index as idx
from core import sessions


class _StubEmbeddings:
    def embed_documents(self, texts):
        return [self._v(t) for t in texts]

    def embed_query(self, text):
        return self._v(text)

    @staticmethod
    def _v(text):
        v = [0.0] * 1024
        for i, ch in enumerate(text[:1024]):
            v[i] = (ord(ch) % 32) / 32.0
        v[0] = v[0] or 0.5
        return v


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(idx, "get_embeddings", lambda model=None: _StubEmbeddings())
    monkeypatch.delenv("ATHENA_API_TOKEN", raising=False)
    from api.main import app

    with TestClient(app) as c:
        yield c
    sessions.REGISTRY.destroy_all()


def _upload(client, sid, name, body):
    return client.post(
        f"/sessions/{sid}/documents",
        files={"file": (name, body, "text/markdown")},
    )


def test_attach_list_and_end_session(client):
    r = _upload(client, "chat-A", "notes.md", b"Kuendigungsfrist betraegt sechs Monate")
    assert r.status_code == 201, r.text
    assert r.json()["chunks"] > 0

    r = client.get("/sessions/chat-A/documents")
    assert r.status_code == 200
    assert [d["source"] for d in r.json()["documents"]] == ["notes.md"]

    r = client.delete("/sessions/chat-A")
    assert r.status_code == 200
    assert r.json()["store_destroyed"] is True

    # Gone, not merely emptied.
    assert client.get("/sessions/chat-A/documents").status_code == 404


def test_one_chats_attachment_is_not_listed_by_another(client):
    _upload(client, "chat-A", "secret.md", b"MANDANT_ALPHA Abfindung 250000")
    _upload(client, "chat-B", "other.md", b"MANDANT_BETA Rechnung 400")

    a = client.get("/sessions/chat-A/documents").json()
    b = client.get("/sessions/chat-B/documents").json()

    assert [d["source"] for d in a["documents"]] == ["secret.md"]
    assert [d["source"] for d in b["documents"]] == ["other.md"]


def test_unsupported_type_is_rejected(client):
    r = client.post(
        "/sessions/chat-A/documents",
        files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert r.status_code == 422
    assert "Unsupported type" in r.json()["detail"]


def test_empty_file_is_rejected(client):
    r = _upload(client, "chat-A", "empty.md", b"")
    assert r.status_code == 422


def test_oversized_upload_is_rejected(client, monkeypatch):
    import api.main as m

    monkeypatch.setattr(m, "_MAX_UPLOAD_BYTES", 32)
    r = _upload(client, "chat-A", "big.md", b"x" * 100)
    assert r.status_code == 413


def test_unknown_session_returns_404_not_an_empty_list(client):
    """
    An expired chat must be distinguishable from a chat with no files. Returning
    an empty list for a session that no longer exists would let a caller believe
    its attachments were searched when they were silently gone.
    """
    assert client.get("/sessions/never-existed/documents").status_code == 404
    assert client.delete("/sessions/never-existed").status_code == 404


def test_archive_endpoint_reports_the_persistent_index(client):
    r = client.get("/documents/archive")
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body and "chunks" in body and "embed_model" in body


def test_attachments_never_enter_the_archive(client):
    before = client.get("/documents/archive").json()["documents"]
    _upload(client, "chat-A", "attached.md", b"NUR_IN_DIESER_SITZUNG 4711")
    after = client.get("/documents/archive").json()["documents"]
    assert after == before, "an attachment was added to the permanent archive"


def test_research_accepts_a_session_id_and_binds_the_thread(client, monkeypatch):
    """The thread must be recorded, or teardown cannot scrub its checkpoints."""
    import api.main as m

    monkeypatch.setattr(m, "_submit", lambda fn, *a: None)  # don't run the graph
    _upload(client, "chat-A", "x.md", b"Etwas Text")

    r = client.post("/research", json={"topic": "Wie lang ist die Frist?",
                                       "session_id": "chat-A"})
    assert r.status_code == 202
    tid = r.json()["thread_id"]

    store = sessions.REGISTRY.get("chat-A")
    assert store is not None and tid in store.thread_ids
