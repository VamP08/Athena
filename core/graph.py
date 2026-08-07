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

import os

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import human_review_node, researcher_node, supervisor_node, writer_node
from .state import ResearchState


def _make_checkpointer():
    """
    Checkpointer selected by ATHENA_CHECKPOINTER:

      memory  (default) — fast, no files; paused runs die with the process.
                          Right for tests and CI.
      sqlite            — durable; paused reviews survive a restart. Right for
                          the app and the API. DB path: ATHENA_DB_PATH.

    Postgres (multi-instance) is the planned production step — same interface,
    swap here only.
    """
    kind = os.getenv("ATHENA_CHECKPOINTER", "memory").lower()
    if kind == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        path = os.getenv("ATHENA_DB_PATH", "athena.db")
        # One long-lived connection shared across threads; SqliteSaver
        # serialises access with its own internal lock.
        conn = sqlite3.connect(path, check_same_thread=False)
        return SqliteSaver(conn)
    return InMemorySaver()


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


def build_graph(checkpointer=None):
    """
    Builds and compiles the Athena research graph.

    Args:
        checkpointer: Optional explicit checkpointer; defaults to the one
            selected by ATHENA_CHECKPOINTER (see _make_checkpointer).

    The checkpointer enables interrupt() to persist state across request
    boundaries — the foundation of the human review gate.
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

    return g.compile(checkpointer=checkpointer or _make_checkpointer())


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
        "retrieved_chunks": [],
        "draft_report": "",
        "next_agent": "",
        "human_feedback": "",
        "iterations": 0,
    }
