"""
tests/test_writer_guard.py
The writer must never hand back an empty report.

Observed on the local backend: a question whose retrieval was perfect — 11
passages, the correct figure ranked first — produced a completely empty draft.
The graph carried "" to the review gate, so a user would have seen a blank
report with no indication of failure, and the grounding eval could only see it
as a missing answer, which reads like a retrieval bug and sends you debugging
the wrong component.

Silence is the failure mode being tested here, not wording.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from core import nodes


class _ScriptedLLM:
    """Returns queued responses in order, recording the prompts it received."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self._replies.pop(0) if self._replies else "")


def _state(n_chunks: int = 11) -> dict:
    return {
        "topic": "What was total revenue in the 2024 financial year?",
        "messages": [],
        "search_results": ["Die Umsatzerloese betrugen 18.452.000,00 EUR."],
        "draft_report": "",
        "next_agent": "",
        "human_feedback": "",
        "iterations": 1,
        "retrieved_chunks": [
            {
                "source": f"Doc_{i}.pdf",
                "locator": f"S. {i}",
                "text": f"Position: Umsatzerloese | 2024 (EUR): 18.452.000,00 (Beleg {i})",
                "scope": "knowledge_base",
            }
            for i in range(1, n_chunks + 1)
        ],
    }


def test_empty_draft_triggers_a_retry_with_fewer_passages(monkeypatch):
    llm = _ScriptedLLM(["", "## Executive Summary\nUmsatz 18.452.000,00 EUR [1]."])
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setenv("ATHENA_MODE", "documents")

    out = nodes.writer_node(_state(11))

    assert len(llm.calls) == 2, "an empty first draft must trigger exactly one retry"
    assert "18.452.000,00" in out["draft_report"]

    first_prompt = llm.calls[0][1].content
    second_prompt = llm.calls[1][1].content
    assert len(second_prompt) < len(first_prompt), (
        "the retry must shrink the evidence block — asking again identically "
        "would just reproduce a context overflow"
    )
    assert "retried" in out["messages"][0].content.lower()


def test_persistent_empty_draft_becomes_a_loud_visible_failure(monkeypatch):
    """Two empty replies must produce a visible explanation, never ''."""
    llm = _ScriptedLLM(["", ""])
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setenv("ATHENA_MODE", "documents")

    out = nodes.writer_node(_state(11))
    draft = out["draft_report"]

    assert draft.strip(), "writer returned an empty report"
    assert "failed" in draft.lower()
    # It must distinguish "generation failed" from "the archive has nothing",
    # which are completely different answers to the user.
    assert "11" in draft, "the failure must state that retrieval succeeded"
    assert "not an empty archive" in draft.lower()


def test_successful_draft_is_untouched(monkeypatch):
    """The guard must not alter the normal path."""
    good = "## Executive Summary\nUmsatz 18.452.000,00 EUR [1]."
    llm = _ScriptedLLM([good])
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setenv("ATHENA_MODE", "documents")

    out = nodes.writer_node(_state(6))

    assert out["draft_report"] == good
    assert len(llm.calls) == 1, "no retry should happen when the first draft is fine"
    assert "retried" not in out["messages"][0].content.lower()


def test_whitespace_only_draft_counts_as_empty(monkeypatch):
    llm = _ScriptedLLM(["   \n\n  ", "## Executive Summary\nOK [1]."])
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.setenv("ATHENA_MODE", "documents")

    out = nodes.writer_node(_state(8))
    assert len(llm.calls) == 2
    assert out["draft_report"].startswith("## Executive Summary")


def test_web_mode_without_chunks_still_guards(monkeypatch):
    """
    With no retrieved passages there is nothing to halve, so the retry cannot
    help — but the failure must still be visible rather than an empty string.
    """
    llm = _ScriptedLLM(["", ""])
    monkeypatch.setattr(nodes, "get_llm", lambda *a, **k: llm)
    monkeypatch.delenv("ATHENA_MODE", raising=False)

    st = _state(0)
    st["retrieved_chunks"] = []
    out = nodes.writer_node(st)

    assert out["draft_report"].strip()
    assert "failed" in out["draft_report"].lower()
