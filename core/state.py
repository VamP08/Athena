"""
core/state.py
Defines the single shared state that flows through every node in the graph.
The Annotated[List, add] reducer means nodes *append* to these lists rather
than overwriting — multiple nodes can contribute without knowing about each other.
"""

from operator import add
from typing import Annotated, List

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    # The research topic submitted by the user
    topic: str

    # Full message history — each node appends; never overwritten
    messages: Annotated[List[BaseMessage], add]

    # Accumulated research summaries from the researcher agent
    search_results: Annotated[List[str], add]

    # The current draft report (overwritten on each revision)
    draft_report: str

    # Supervisor's routing decision — consumed by conditional edges
    next_agent: str

    # Human feedback from the review node
    # Non-empty → writer revises; cleared by writer after use
    human_feedback: str

    # Tracks how many supervisor cycles have run (prevents infinite loops)
    iterations: int
