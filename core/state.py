"""
core/state.py
Defines the single shared state that flows through every node in the graph.
The Annotated[List, add] reducer means nodes *append* to these lists rather
than overwriting — multiple nodes can contribute without knowing about each other.
"""

from operator import add
from typing import Annotated

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # The research topic submitted by the user
    topic: str

    # Full message history — each node appends; never overwritten
    messages: Annotated[list[BaseMessage], add]

    # Accumulated research summaries from the researcher agent
    search_results: Annotated[list[str], add]

    # Verbatim evidence behind those summaries, in document mode: one dict per
    # retrieved passage ({source, locator, kind, text, score, ...}).
    #
    # This exists because `search_results` holds the researcher's *prose
    # synthesis*, not its sources. The writer therefore never saw the underlying
    # passages, so it could not cite them, and Ragas was being handed one fused
    # blob as `retrieved_contexts` — which is why context_precision scored a
    # meaningless 1.00. Carrying the real chunks makes citation possible and
    # makes the retrieval metrics measure retrieval.
    retrieved_chunks: Annotated[list[dict], add]

    # The current draft report (overwritten on each revision)
    draft_report: str

    # Supervisor's routing decision — consumed by conditional edges
    next_agent: str

    # Human feedback from the review node
    # Non-empty → writer revises; cleared by writer after use
    human_feedback: str

    # Tracks how many supervisor cycles have run (prevents infinite loops)
    iterations: int

    # Which chat's attached documents this run may see. Carried in STATE rather
    # than a module global on purpose: one process serves many browser sessions
    # and many API threads at once, so a global would let one chat's uploads
    # reach another's answers — the one failure this tier exists to prevent.
    # Empty string means "archive only".
    session_id: str
