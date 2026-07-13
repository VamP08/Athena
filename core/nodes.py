"""
core/nodes.py
The four nodes that make up the Athena research graph.

Node responsibilities:
  supervisor     → deterministic routing + LLM fallback for edge cases
  researcher     → manual tool-calling loop (no deprecated create_react_agent)
  writer         → generates / revises the structured report
  human_review   → HITL checkpoint using LangGraph interrupt()
"""

import os
from typing import Literal, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt
from pydantic import BaseModel

from .llm import get_llm
from .state import ResearchState

# ── Prompts ──────────────────────────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a research assistant with access to web search tools.

Your job: gather comprehensive, factual information on the given topic.

Instructions:
1. Run at least 3 different searches covering different angles:
   - Broad overview of the topic
   - Recent developments or news (add year to query)
   - Key facts, statistics, or notable examples
2. After gathering information, write a detailed synthesis.
3. Include specific facts: names, dates, numbers, organisations.
4. Do NOT stop after one search — depth matters.
"""

WRITER_SYSTEM = """You are an expert research analyst producing professional reports.

Requirements:
- Length: 400–600 words
- Tone: factual, specific, suitable for an intelligent non-specialist audience
- CRITICAL: Do not include ANY claim not directly supported by the research provided.
  If the research doesn't cover something, omit it.

Exact structure (use these markdown headers):

## Executive Summary
2–3 sentences covering the single most important takeaway.

## Key Findings
- Finding 1 with supporting evidence or data point
- Finding 2 with supporting evidence or data point
- Finding 3 with supporting evidence or data point
- Finding 4 with supporting evidence or data point

## Analysis
2–3 paragraphs synthesising the findings, discussing implications, and providing context.

## Conclusion
1–2 sentences: what should the reader walk away knowing or doing?
"""

# ── Tool selection ────────────────────────────────────────────────────────────

def _get_search_tools() -> List[BaseTool]:
    """
    Returns search tools for the researcher agent.

    MCP_MODE=true  → connects to the FastMCP web server (run web_server.py first)
    MCP_MODE=false → uses the in-process ddgs web_search tool (default, cloud-safe)
    """
    if os.getenv("MCP_MODE", "false").lower() == "true":
        try:
            from mcp_servers.connect import get_mcp_tools_sync
            tools = get_mcp_tools_sync()
            if tools:
                return tools
            raise RuntimeError("MCP server returned no tools")
        except Exception as e:
            print(f"[Athena] MCP connection failed — falling back to direct tools. Error: {e}")

    from .tools import web_search
    return [web_search]


# ── Research execution (separated for testability) ────────────────────────────

def _execute_research(topic: str, tools: List[BaseTool]) -> str:
    """
    Runs the tool-calling loop and returns the final synthesis.

    Separated from researcher_node so tests can mock this cleanly
    without needing to replicate the full LLM + tool interaction.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=f"Research this topic thoroughly: {topic}"),
    ]

    for iteration in range(6):  # hard cap prevents runaway loops
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # Model decided it has enough information — exit loop
            break

        # Execute each tool call and feed results back
        for tc in response.tool_calls:
            tool = tool_map.get(tc["name"])
            if tool is None:
                result = f"Unknown tool: {tc['name']}"
            else:
                try:
                    result = tool.invoke(tc["args"])
                except Exception as e:
                    result = f"Tool error: {e}"

            messages.append(
                ToolMessage(
                    content=str(result)[:3000],  # truncate to avoid context overflow
                    tool_call_id=tc["id"],
                )
            )

    # Normally the loop exits on a tool-call-free AIMessage — the synthesis.
    # If the iteration cap exhausted while the model was still calling tools,
    # the last message is a raw ToolMessage — force a final synthesis, without
    # tools bound, so the writer never receives a raw search dump as "research".
    last = messages[-1]
    if isinstance(last, AIMessage) and not last.tool_calls:
        return last.content

    messages.append(HumanMessage(
        content="Stop searching. Write your comprehensive synthesis of all findings now."
    ))
    return llm.invoke(messages).content


# ── Supervisor node ───────────────────────────────────────────────────────────

class Route(BaseModel):
    next: Literal["researcher", "writer", "end"]
    reason: str


def supervisor_node(state: ResearchState) -> dict:
    """
    Routes to the next agent. Deterministic for clear cases; LLM for edge cases.

    Routing rules:
      No search results          → researcher  (gather information)
      Results but no draft       → writer      (produce first draft)
      Draft exists, iters >= 3   → end         (prevent infinite loops)
      Otherwise                  → LLM decides (e.g., need more research?)
    """
    n_results = len(state.get("search_results", []))
    has_draft = bool(state.get("draft_report", ""))
    iters = state.get("iterations", 0)

    # ── Deterministic routing (no LLM needed for the common cases) ──
    if n_results == 0:
        next_agent, reason = "researcher", "No research gathered yet"

    elif not has_draft:
        next_agent, reason = "writer", "Research complete — drafting first report"

    elif iters >= 3:
        next_agent, reason = "end", "Maximum iteration limit reached"

    else:
        # ── LLM routing for ambiguous cases ──
        # (e.g., human asked for major changes that need more research)
        llm = get_llm()
        router = llm.with_structured_output(Route)
        try:
            decision = router.invoke([
                SystemMessage(content=f"""You are a research supervisor.

Topic: {state['topic']}
Search results gathered: {n_results}
Draft report exists: {has_draft}
Iterations completed: {iters}

Decide the next step:
- 'researcher': Need more / different research before revising
- 'writer':     Research is sufficient; write or revise the report now
- 'end':        The workflow is complete
""")
            ])
            next_agent, reason = decision.next, decision.reason
        except Exception as e:
            next_agent, reason = "writer", f"LLM routing failed ({e}); defaulting to writer"

    return {
        "next_agent": next_agent,
        "iterations": iters + 1,
        "messages": [AIMessage(content=f"[Supervisor] → {next_agent}: {reason}")],
    }


# ── Researcher node ───────────────────────────────────────────────────────────

def researcher_node(state: ResearchState) -> dict:
    """
    Gathers web research using a manual tool-calling loop.

    Uses _execute_research() which is mockable in unit tests.
    Connects to either the FastMCP web server (MCP_MODE=true) or
    DuckDuckGo directly (MCP_MODE=false, default).
    """
    tools = _get_search_tools()
    synthesis = _execute_research(state["topic"], tools)

    return {
        "search_results": [synthesis],
        "messages": [AIMessage(content=f"[Researcher] Research complete on '{state['topic']}'")],
    }


# ── Writer node ───────────────────────────────────────────────────────────────

def writer_node(state: ResearchState) -> dict:
    """
    Produces or revises the structured markdown report.

    If human_feedback is set, incorporates it surgically.
    Clears human_feedback after use so the revision flag resets.
    """
    llm = get_llm()

    research_context = "\n\n---\n\n".join(state.get("search_results", []))
    feedback = state.get("human_feedback", "").strip()

    revision_block = ""
    if feedback:
        # The previous draft must be in the prompt — otherwise the model
        # regenerates from research and "targeted changes" is impossible.
        revision_block = (
            f"\n\nPREVIOUS DRAFT (revise this — do not start from scratch):\n"
            f"{state.get('draft_report', '')}\n\n"
            f"**REVISION REQUEST** — address this feedback precisely:\n"
            f"{feedback}\n\n"
            f"Make targeted changes to the specific sections mentioned. "
            f"Keep sections that were not mentioned identical to the previous draft."
        )

    response = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=(
            f"Write a research report on: **{state['topic']}**\n\n"
            f"Research gathered:\n{research_context}"
            f"{revision_block}"
        )),
    ])

    return {
        "draft_report": response.content,
        "human_feedback": "",   # reset — writer has consumed the feedback
        "messages": [AIMessage(content="[Writer] Draft report generated.")],
    }


# ── Human review node ─────────────────────────────────────────────────────────

def human_review_node(state: ResearchState) -> dict:
    """
    HITL checkpoint. Pauses graph execution until the user acts.

    How it works:
      1. interrupt() saves full state to the checkpointer and halts execution.
      2. The Streamlit UI reads the draft and shows Approve / Revise buttons.
      3. User action resumes via graph.invoke(Command(resume=value), config=config).
      4. interrupt() returns the resume value.
         - "approve" → human_feedback="" → review→END edge fires
         - Any other text → human_feedback=text → review→writer edge fires
    """
    feedback = interrupt({
        "draft_report": state["draft_report"],
        "message": "Review the draft. 'approve' to finalise, or provide specific feedback.",
    })

    return {
        "human_feedback": "" if str(feedback).strip().lower() == "approve" else str(feedback),
    }
