"""
tests/test_supervisor.py
Unit tests for supervisor_node routing logic.

The supervisor uses deterministic rules for the common cases (no results →
researcher, no draft → writer, iters >= 3 → end). Only edge cases invoke
the LLM, so most tests here require no mocking at all.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.nodes import supervisor_node


def make_state(
    topic="test topic",
    n_results=0,
    has_draft=False,
    iters=0,
    human_feedback="",
):
    return {
        "topic": topic,
        "messages": [],
        "search_results": ["result"] * n_results,
        "draft_report": "some draft" if has_draft else "",
        "next_agent": "",
        "human_feedback": human_feedback,
        "iterations": iters,
    }


# ── Deterministic routing (no LLM mock needed) ────────────────────────────────

def test_routes_to_researcher_when_no_results():
    result = supervisor_node(make_state(n_results=0))
    assert result["next_agent"] == "researcher"


def test_routes_to_writer_when_results_exist_no_draft():
    result = supervisor_node(make_state(n_results=3, has_draft=False))
    assert result["next_agent"] == "writer"


def test_routes_to_end_at_max_iterations():
    result = supervisor_node(make_state(n_results=2, has_draft=True, iters=3))
    assert result["next_agent"] == "end"


def test_iterations_increments_on_each_call():
    result = supervisor_node(make_state(iters=0))
    assert result["iterations"] == 1

    result2 = supervisor_node(make_state(iters=5))
    assert result2["iterations"] == 6


def test_returns_ai_message_in_messages():
    from langchain_core.messages import AIMessage
    result = supervisor_node(make_state(n_results=0))
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "[Supervisor]" in result["messages"][0].content


# ── LLM-assisted routing (edge case: has draft, iters < 3) ────────────────────

@patch("core.nodes.get_llm")
def test_llm_routing_called_for_edge_case(mock_get_llm):
    """When draft exists and iters < 3, the LLM is consulted."""
    from pydantic import BaseModel
    from typing import Literal

    class FakeRoute(BaseModel):
        next: Literal["researcher", "writer", "end"] = "writer"
        reason: str = "needs revision"

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.return_value = FakeRoute()
    mock_get_llm.return_value = mock_llm

    result = supervisor_node(make_state(n_results=2, has_draft=True, iters=1))

    assert result["next_agent"] == "writer"
    mock_llm.with_structured_output.assert_called_once()


@patch("core.nodes.get_llm")
def test_llm_routing_failure_defaults_to_writer(mock_get_llm):
    """If the LLM call throws, supervisor falls back gracefully."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM down")
    mock_get_llm.return_value = mock_llm

    result = supervisor_node(make_state(n_results=2, has_draft=True, iters=1))
    assert result["next_agent"] == "writer"
