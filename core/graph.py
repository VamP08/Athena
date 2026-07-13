"""
core/graph.py
Assembles the four nodes into a LangGraph StateGraph.

Graph topology:
                     ┌─────────────────────────┐
              ┌──────┤        supervisor        ├──────┐
              │      └─────────────────────────┘      │
              │          ↑ always loops back           │
              ▼                                        ▼
        ┌──────────┐                           ┌──────────┐       ┌─────┐
        │researcher│                           │  writer  │──────►│review│
        └──────────┘                           └──────────┘       └──┬──┘
                                                     ▲               │
                                                     │ if feedback    │ if "approve"
                                                     └───────────────│──► END
                                                                     │
                                                                  interrupt()
                                                               (pauses for human)
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import human_review_node, researcher_node, supervisor_node, writer_node
from .state import ResearchState


def _route_supervisor(state: ResearchState) -> str:
    """Reads supervisor's routing decision from state."""
    return state["next_agent"]


def _route_after_review(state: ResearchState) -> str:
    """
    After human review:
      - human_feedback is non-empty → writer needs to revise
      - human_feedback is empty (approved) → END
    """
    return "writer" if state.get("human_feedback") else END


def build_graph():
    """
    Builds and compiles the Athena research graph.

    Returns:
        A compiled LangGraph CompiledGraph with in-memory checkpointing.
        The checkpointer enables interrupt() to persist state across HTTP
        request boundaries (critical for Streamlit's HITL panel).
        Dev/demo only — swap for SqliteSaver/PostgresSaver in production.
    """
    g = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────────────
    g.add_node("supervisor", supervisor_node)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("review", human_review_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    g.add_edge(START, "supervisor")

    # ── Edges ─────────────────────────────────────────────────────────────────
    # Supervisor routes to researcher, writer, or end
    g.add_conditional_edges(
        "supervisor",
        _route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "end": END,
        },
    )

    # Researcher always returns to supervisor for re-evaluation
    g.add_edge("researcher", "supervisor")

    # Writer always goes to human review
    g.add_edge("writer", "review")

    # Review either loops back to writer (revision) or ends (approved)
    g.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "writer": "writer",
            END: END,
        },
    )

    return g.compile(checkpointer=InMemorySaver())


def make_initial_state(topic: str) -> dict:
    """
    Returns a clean initial state dict for a new research session.

    Convention, not a framework requirement: LangGraph tolerates missing keys,
    but nodes index some keys directly (e.g. state["topic"]), so we always
    provide every ResearchState key for predictable behaviour.
    """
    return {
        "topic": topic,
        "messages": [],
        "search_results": [],
        "draft_report": "",
        "next_agent": "",
        "human_feedback": "",
        "iterations": 0,
    }
