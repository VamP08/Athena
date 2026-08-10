"""
tests/test_scope_binding.py
The researcher may only ever reach the documents attached to ITS OWN chat.

The session id travels in graph STATE rather than a module global, because one
process serves many browser sessions and many API worker threads at once. A
global would let one chat's uploads surface in another's answers — the single
failure this tier exists to prevent, and the one that would falsify the
project's central claim.

These tests exercise the real binding path: state -> researcher_node ->
_get_search_tools -> get_document_tools -> a closure over one SessionStore.
"""

from __future__ import annotations

from core import doc_tools
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


def _setup(monkeypatch, tmp_path):
    monkeypatch.setattr(idx, "get_embeddings", lambda model=None: _StubEmbeddings())
    monkeypatch.setenv("ATHENA_MODE", "documents")
    monkeypatch.setenv("ATHENA_INDEX_PATH", str(tmp_path / "kb.db"))
    idx.close()
    sessions.REGISTRY.destroy_all()


def test_tools_bound_to_one_chat_cannot_see_another(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)

    a = sessions.REGISTRY.create("chat-A")
    b = sessions.REGISTRY.create("chat-B")
    a.add_document("secret.md", b"MANDANT_ALPHA Abfindung 250000 EUR")
    b.add_document("other.md", b"MANDANT_BETA Rechnung 400 EUR")

    tools_a = {t.name: t for t in doc_tools.get_document_tools("chat-A")}
    tools_b = {t.name: t for t in doc_tools.get_document_tools("chat-B")}

    out_a = tools_a["search_documents"].invoke(
        {"query": "MANDANT_ALPHA Abfindung", "doc_type": "", "year": ""}
    )
    text_a = out_a[0] if isinstance(out_a, tuple) else str(out_a)
    assert "MANDANT_ALPHA" in text_a, "chat A cannot find its own attachment"

    out_b = tools_b["search_documents"].invoke(
        {"query": "MANDANT_ALPHA Abfindung", "doc_type": "", "year": ""}
    )
    text_b = out_b[0] if isinstance(out_b, tuple) else str(out_b)
    assert "MANDANT_ALPHA" not in text_b, "chat B retrieved chat A's attachment"
    assert "250000" not in text_b

    idx.close()
    sessions.REGISTRY.destroy_all()


def test_no_session_id_yields_the_archive_only_toolset(monkeypatch, tmp_path):
    """A run with no chat bound must behave exactly as before this feature."""
    _setup(monkeypatch, tmp_path)
    names = sorted(t.name for t in doc_tools.get_document_tools(""))
    assert names == ["document_search", "list_documents"]
    idx.close()


def test_unknown_session_id_falls_back_to_archive_not_an_error(monkeypatch, tmp_path):
    """
    An expired or reaped chat must degrade to archive-only rather than raising.
    A hard failure here would break a conversation mid-flight after a TTL sweep.
    """
    _setup(monkeypatch, tmp_path)
    names = sorted(t.name for t in doc_tools.get_document_tools("does-not-exist"))
    assert names == ["document_search", "list_documents"]
    idx.close()


def test_researcher_node_passes_state_session_id_through(monkeypatch, tmp_path):
    """The binding must come from STATE, not from any ambient global."""
    _setup(monkeypatch, tmp_path)
    from core import nodes

    seen = {}

    def fake_tools(session_id=""):
        seen["session_id"] = session_id
        return []

    monkeypatch.setattr(nodes, "_get_search_tools", fake_tools)
    monkeypatch.setattr(nodes, "_execute_research", lambda topic, tools: ("synth", []))

    nodes.researcher_node({
        "topic": "x", "messages": [], "search_results": [], "retrieved_chunks": [],
        "draft_report": "", "next_agent": "", "human_feedback": "", "iterations": 0,
        "session_id": "chat-XYZ",
    })
    assert seen["session_id"] == "chat-XYZ"
    idx.close()


def test_hits_are_labelled_with_their_scope(monkeypatch, tmp_path):
    """
    Attached and archived evidence must stay distinguishable all the way to the
    citation: "in our records" and "in the file you gave me" are different claims.
    """
    _setup(monkeypatch, tmp_path)
    s = sessions.REGISTRY.create("chat-A")
    s.add_document("v.md", b"Kuendigungsfrist betraegt sechs Monate")

    tools = {t.name: t for t in doc_tools.get_document_tools("chat-A")}
    # A content_and_artifact tool only returns its artifact when invoked with a
    # full tool-call payload; passing bare args yields the content string alone.
    # This is also how core/nodes.py drives it, so the test exercises the real path.
    msg = tools["search_documents"].invoke({
        "name": "search_documents",
        "args": {"query": "Kuendigungsfrist", "doc_type": "", "year": ""},
        "id": "call-1",
        "type": "tool_call",
    })
    assert "ATTACHED TO THIS CHAT" in msg.content

    # Both tiers are searched by design, so the assertion is not "everything is
    # session" — it is that each hit is labelled with the tier it came from and
    # the attached file leads. (Note DEFAULT_INDEX_PATH is captured at import, so
    # the archive here is the real one; that is fine, it makes the mixed case the
    # thing under test.)
    scopes = {a["scope"] for a in msg.artifact}
    assert scopes <= {"session", "knowledge_base"}, f"unlabelled scope: {scopes}"

    session_hits = [a for a in msg.artifact if a["scope"] == "session"]
    assert session_hits, "the attached document was not returned"
    assert session_hits[0]["source"] == "v.md"
    assert msg.artifact[0]["scope"] == "session", "attached evidence must be listed first"

    for a in msg.artifact:
        if a["source"] == "v.md":
            assert a["scope"] == "session"
        else:
            assert a["scope"] == "knowledge_base", (
                f"{a['source']} came from the archive but was labelled {a['scope']}"
            )
    idx.close()
    sessions.REGISTRY.destroy_all()
