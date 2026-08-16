"""
tests/test_graph.py
Integration tests for the full LangGraph pipeline.

Mocks both the LLM and web search so tests run in <10 seconds
with zero network calls — safe for CI environments.

Test coverage:
  - Graph interrupts at the review node after researcher + writer run
  - Approval ends the graph cleanly (no next nodes)
  - Feedback triggers the revision loop (writer re-runs, graph interrupts again)
  - Revised draft differs from the first draft
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langgraph.types import Command

from core.graph import build_graph, make_initial_state

# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_mock_llm(drafts=None):
    """
    Returns a mock LLM that:
      - bind_tools() returns itself (researcher tool-calling path)
      - invoke() returns AIMessage with no tool_calls (exits tool loop immediately)
      - with_structured_output().invoke() returns routing decisions in sequence
    """
    if drafts is None:
        drafts = ["## Report\n\nTest report content for integration testing."]

    draft_iter = iter(drafts)
    call_log = {"n": 0}

    def fake_invoke(messages, *args, **kwargs):
        call_log["n"] += 1
        # First call from researcher → no tool calls, return synthesis
        # Subsequent calls from writer → return draft
        try:
            content = next(draft_iter)
        except StopIteration:
            content = "## Report\n\nRevised report content."
        return AIMessage(content=content, tool_calls=[])

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm   # bind_tools returns self
    mock_llm.invoke.side_effect = fake_invoke

    # Supervisor structured output — routes researcher first, then writer
    routing_sequence = [
        MagicMock(next="researcher", reason="no results"),
        MagicMock(next="writer", reason="research done"),
    ]
    mock_llm.with_structured_output.return_value.invoke.side_effect = routing_sequence

    return mock_llm


# ── Test 1: Graph interrupts at review ───────────────────────────────────────

@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_graph_interrupts_at_review(mock_get_llm, mock_tools):
    mock_get_llm.return_value = _make_mock_llm()

    graph = build_graph()
    config = {"configurable": {"thread_id": "integ-001"}}

    graph.invoke(make_initial_state("test topic"), config=config)
    state = graph.get_state(config)

    assert state.next == ("review",), (
        f"Expected graph paused at ('review',), got: {state.next}"
    )
    assert state.values.get("draft_report"), "draft_report should be non-empty"


# ── Test 2: Approval ends the graph ──────────────────────────────────────────

@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_approval_ends_graph(mock_get_llm, mock_tools):
    mock_get_llm.return_value = _make_mock_llm()

    graph = build_graph()
    config = {"configurable": {"thread_id": "integ-002"}}

    graph.invoke(make_initial_state("test topic"), config=config)
    graph.invoke(Command(resume="approve"), config=config)

    final = graph.get_state(config)
    assert not final.next, (
        f"Expected graph finished (empty next), got: {final.next}"
    )
    assert final.values.get("human_feedback") == "", (
        "human_feedback should be cleared after approval"
    )


# ── Test 3: Revision loop ─────────────────────────────────────────────────────

@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_feedback_triggers_revision_loop(mock_get_llm, mock_tools):
    first_draft = "## Report\n\nFirst draft content."
    revised_draft = "## Report\n\nRevised draft with more economic statistics."

    mock_get_llm.return_value = _make_mock_llm(
        drafts=[
            "Research synthesis",     # researcher invoke
            first_draft,              # writer first pass
            "Research synthesis 2",   # researcher not called again, but writer is
            revised_draft,            # writer revision
        ]
    )

    graph = build_graph()
    config = {"configurable": {"thread_id": "integ-003"}}

    # First run → interrupt at review
    graph.invoke(make_initial_state("test topic"), config=config)
    first = graph.get_state(config)
    assert first.next == ("review",)

    # Send feedback → writer re-runs → interrupt again
    graph.invoke(
        Command(resume="Add more economic statistics to the Analysis section."),
        config=config,
    )
    second = graph.get_state(config)

    assert second.next == ("review",), (
        f"Expected graph to interrupt again for second review, got: {second.next}"
    )
    assert second.values.get("human_feedback") == "", (
        "human_feedback should be cleared by writer after revision"
    )


# ── Test 4: Revised draft is different ───────────────────────────────────────

@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_revised_draft_differs_from_original(mock_get_llm, mock_tools):
    first_draft = "## Report\n\nOriginal content here."
    revised_draft = "## Report\n\nRevised content with extra statistics."

    # Each call to get_llm returns the same mock; invoke cycles through drafts
    draft_iter = iter(["Research synthesis", first_draft, revised_draft])

    def fake_invoke(messages, *args, **kwargs):
        return AIMessage(content=next(draft_iter, revised_draft), tool_calls=[])

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.side_effect = fake_invoke
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        MagicMock(next="researcher", reason=""),
        MagicMock(next="writer", reason=""),
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_graph()
    config = {"configurable": {"thread_id": "integ-004"}}

    graph.invoke(make_initial_state("test topic"), config=config)
    original = graph.get_state(config).values["draft_report"]

    graph.invoke(Command(resume="Please revise."), config=config)
    revised = graph.get_state(config).values["draft_report"]

    assert original != revised, "Revised draft should be different from the original"


# ── Test 5: human_feedback cleared by writer ─────────────────────────────────

@patch("core.nodes._get_search_tools", return_value=[])
@patch("core.nodes.get_llm")
def test_human_feedback_cleared_after_revision(mock_get_llm, mock_tools):
    mock_get_llm.return_value = _make_mock_llm(
        drafts=["Research", "First draft", "Revised draft"]
    )

    graph = build_graph()
    config = {"configurable": {"thread_id": "integ-005"}}

    graph.invoke(make_initial_state("test topic"), config=config)
    graph.invoke(Command(resume="Add more detail please."), config=config)

    state = graph.get_state(config)
    # Writer should have consumed and cleared the feedback
    assert state.values.get("human_feedback") == "", (
        "human_feedback must be empty string after writer processes it"
    )
