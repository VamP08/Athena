"""
tests/test_api.py
API-layer tests: job lifecycle, review gate over HTTP, auth, audit trail.

Runs the REAL graph with a mocked LLM (same seams as test_graph.py), inline
execution (ATHENA_API_SYNC=1), in-memory checkpointer, and a temp registry DB -
deterministic, no network, CI-safe.
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """Fresh API app per test: clean registry DB, sync execution, no auth."""
    monkeypatch.setenv("ATHENA_API_SYNC", "1")
    monkeypatch.setenv("ATHENA_CHECKPOINTER", "memory")
    monkeypatch.setenv("ATHENA_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("ATHENA_API_TOKEN", raising=False)

    import api.main
    importlib.reload(api.main)
    return TestClient(api.main.app)


def _mock_llm():
    """LLM whose researcher pass exits immediately and writer returns a draft."""
    mock = MagicMock()
    mock.bind_tools.return_value = mock
    mock.invoke.side_effect = lambda *a, **k: AIMessage(
        content="## Report\n\nDeterministic test content.", tool_calls=[]
    )
    return mock


@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_full_lifecycle_approve(mock_llm, mock_tools, client):
    mock_llm.return_value = _mock_llm()

    # Start
    r = client.post("/research", json={"topic": "test topic"})
    assert r.status_code == 202
    thread_id = r.json()["thread_id"]

    # Paused at review, draft available
    r = client.get(f"/research/{thread_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "awaiting_review"
    assert body["draft_report"].startswith("## Report")

    # Approve -> completed
    r = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
    assert r.status_code == 202
    assert client.get(f"/research/{thread_id}").json()["status"] == "completed"


@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_revision_loop_over_http(mock_llm, mock_tools, client):
    mock_llm.return_value = _mock_llm()

    thread_id = client.post("/research", json={"topic": "test topic"}).json()["thread_id"]

    r = client.post(
        f"/research/{thread_id}/resume",
        json={"action": "revise", "feedback": "Add more figures."},
    )
    assert r.status_code == 202
    # Writer re-ran -> paused at review again
    assert client.get(f"/research/{thread_id}").json()["status"] == "awaiting_review"


@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_audit_trail_records_decisions(mock_llm, mock_tools, client):
    mock_llm.return_value = _mock_llm()

    thread_id = client.post("/research", json={"topic": "test topic"}).json()["thread_id"]
    client.post(
        f"/research/{thread_id}/resume",
        json={"action": "revise", "feedback": "shorter"},
    )
    client.post(f"/research/{thread_id}/resume", json={"action": "approve"})

    audit = client.get(f"/research/{thread_id}/audit").json()["audit"]
    actions = [a["action"] for a in audit]
    assert actions == ["created", "revise", "approve"]
    assert audit[1]["detail"] == "shorter"


def test_resume_requires_awaiting_review(client):
    with patch("core.nodes._get_search_tools", return_value=[]), \
         patch("core.nodes.get_llm") as mock_llm:
        mock_llm.return_value = _mock_llm()
        thread_id = client.post("/research", json={"topic": "test topic"}).json()["thread_id"]
        client.post(f"/research/{thread_id}/resume", json={"action": "approve"})

        # Already completed -> conflict
        r = client.post(f"/research/{thread_id}/resume", json={"action": "approve"})
        assert r.status_code == 409


def test_revise_without_feedback_rejected(client):
    with patch("core.nodes._get_search_tools", return_value=[]), \
         patch("core.nodes.get_llm") as mock_llm:
        mock_llm.return_value = _mock_llm()
        thread_id = client.post("/research", json={"topic": "test topic"}).json()["thread_id"]

        r = client.post(f"/research/{thread_id}/resume", json={"action": "revise"})
        assert r.status_code == 422


def test_unknown_thread_404(client):
    assert client.get("/research/nope").status_code == 404


def test_failed_run_surfaces_error(client):
    with patch("core.nodes._get_search_tools", return_value=[]), \
         patch("core.nodes.get_llm") as mock_llm:
        broken = MagicMock()
        broken.bind_tools.return_value = broken
        broken.invoke.side_effect = RuntimeError("LLM down")
        mock_llm.return_value = broken

        thread_id = client.post("/research", json={"topic": "test topic"}).json()["thread_id"]
        body = client.get(f"/research/{thread_id}").json()
        assert body["status"] == "failed"
        assert "LLM down" in body["error"]


def test_auth_enforced_when_token_set(monkeypatch, tmp_path):
    monkeypatch.setenv("ATHENA_API_SYNC", "1")
    monkeypatch.setenv("ATHENA_CHECKPOINTER", "memory")
    monkeypatch.setenv("ATHENA_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("ATHENA_API_TOKEN", "secret-token")

    import api.main
    importlib.reload(api.main)
    client = TestClient(api.main.app)

    assert client.post("/research", json={"topic": "test topic"}).status_code == 401
    assert client.get("/health").status_code == 200  # health stays open

    with patch("core.nodes._get_search_tools", return_value=[]), \
         patch("core.nodes.get_llm") as mock_llm:
        mock_llm.return_value = _mock_llm()
        r = client.post(
            "/research",
            json={"topic": "test topic"},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert r.status_code == 202
