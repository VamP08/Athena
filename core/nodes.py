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

RESEARCHER_DOCUMENTS_SYSTEM = """You are a financial analyst with access to this
organisation's internal document archive. The archive is the ONLY source of truth —
you have no web access and no outside knowledge of this organisation.

Instructions:
1. If the request is broad (e.g. "write a report on our finances"), call
   list_documents FIRST so you know what the archive actually contains.
2. Use document_search for every fact. Run several searches covering different
   angles and different wordings — figures, account names, periods, parties.
3. Documents are in German and English. If a German search returns little, try the
   English wording of the same idea, and vice versa.
4. Copy figures EXACTLY as they appear, including the original number format
   (e.g. 4.821.000,00 EUR). Never round, convert, or recalculate a figure.
5. Record which document and locator each fact came from — the report must cite them.
6. If the archive does not contain something, say so plainly. Never fill a gap
   from general knowledge; an invented figure is far worse than a stated gap.
7. If a document is reported as only partly indexed (a scan, or formula-only
   totals), mention that limitation rather than treating it as absent.

After searching, write a detailed synthesis of what you found, with the source
document named next to every figure.
"""

WRITER_DOCUMENTS_SYSTEM = """You are a financial analyst producing an internal report
from this organisation's own documents.

Requirements:
- Length: 400-600 words
- Tone: precise, factual, suitable for a finance or audit reader
- CRITICAL: every figure and claim must come from the retrieved passages provided.
  If the passages do not support something, omit it or state that the archive does
  not cover it. Do NOT use outside knowledge about any company.
- Reproduce figures exactly as written in the sources. Do not round or recompute.
- Cite sources inline as [1], [2] matching the numbered passages you were given,
  and end with a "## Quellen / Sources" section listing them.
- Answer in the language of the user's question.

Exact structure (use these markdown headers):

## Executive Summary
2-3 sentences with the single most important takeaway.

## Key Findings
- Finding with the exact figure and a [n] citation
- (4 findings, each cited)

## Analysis
2-3 paragraphs of context and implications, cited.

## Conclusion
1-2 sentences: what the reader should know or do.

## Quellen / Sources
[1] filename — locator
"""

# Per-tool output budgets. A web snippet dump is fine at 3000 chars; a document
# search returns several verbatim passages that must reach the model intact,
# because the writer cites them by number.
_DEFAULT_TOOL_BUDGET = 3000
_TOOL_CHAR_BUDGET = {
    "document_search": int(os.getenv("ATHENA_SEARCH_CHAR_BUDGET", "6000")) + 800,
    "list_documents": 4000,
}


def _document_mode() -> bool:
    return os.getenv("ATHENA_MODE", "web").lower() == "documents"


# ── Tool selection ────────────────────────────────────────────────────────────

def _get_search_tools(session_id: str = "") -> List[BaseTool]:
    """
    Returns search tools for the researcher agent.

    ATHENA_MODE=documents → the document archive (document_search + list_documents)
    MCP_MODE=true         → connects to the FastMCP web server (run web_server.py first)
    otherwise             → the in-process ddgs web_search tool (default, cloud-safe)

    Document mode deliberately does NOT also expose web_search. Two reasons: it
    is what makes "no data leaves this machine" literally true rather than
    aspirational, and a 9B local model routes far more reliably between two
    tools than four.
    """
    if os.getenv("ATHENA_MODE", "web").lower() == "documents":
        from .doc_tools import get_document_tools

        return get_document_tools(session_id)

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

def _execute_research(topic: str, tools: List[BaseTool]) -> tuple[str, List[dict]]:
    """
    Runs the tool-calling loop.

    Returns (synthesis, retrieved_chunks). The second element is empty in web
    mode and holds the verbatim passages behind the synthesis in document mode,
    so the writer can cite them and the eval harness can score real retrieval.

    Separated from researcher_node so tests can mock this cleanly
    without needing to replicate the full LLM + tool interaction.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(
            content=RESEARCHER_DOCUMENTS_SYSTEM if _document_mode() else RESEARCHER_SYSTEM
        ),
        HumanMessage(content=f"Research this topic thoroughly: {topic}"),
    ]
    chunks: List[dict] = []
    seen_keys: set = set()

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
                messages.append(ToolMessage(
                    content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]
                ))
                continue

            # Invoked with the whole ToolCall (not just .args) so LangChain
            # returns a ToolMessage carrying .artifact for content_and_artifact
            # tools. That artifact is how verbatim chunks escape the tool loop.
            try:
                msg = tool.invoke(tc)
            except Exception as e:
                messages.append(ToolMessage(content=f"Tool error: {e}", tool_call_id=tc["id"]))
                continue

            if not isinstance(msg, ToolMessage):
                msg = ToolMessage(content=str(msg), tool_call_id=tc["id"])

            for rec in (getattr(msg, "artifact", None) or []):
                if not isinstance(rec, dict):
                    continue
                key = (rec.get("source"), rec.get("locator"), (rec.get("text") or "")[:80])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                chunks.append(rec)

            # Budget is per tool, not global: 3000 chars is right for a web
            # snippet dump but silently decapitates a multi-passage document
            # result, which would make the writer cite a source it never saw.
            budget = _TOOL_CHAR_BUDGET.get(tc["name"], _DEFAULT_TOOL_BUDGET)
            if isinstance(msg.content, str) and len(msg.content) > budget:
                msg.content = msg.content[:budget].rstrip() + "\n[... truncated]"
            messages.append(msg)

    # Normally the loop exits on a tool-call-free AIMessage — the synthesis.
    # If the iteration cap exhausted while the model was still calling tools,
    # the last message is a raw ToolMessage — force a final synthesis, without
    # tools bound, so the writer never receives a raw search dump as "research".
    last = messages[-1]
    if isinstance(last, AIMessage) and not last.tool_calls:
        return last.content, chunks

    messages.append(HumanMessage(
        content="Stop searching. Write your comprehensive synthesis of all findings now."
    ))
    return llm.invoke(messages).content, chunks


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
    # session_id comes from STATE, so each run can only ever reach the documents
    # attached to its own chat. Two chats running concurrently in the same
    # process get different tool objects bound to different stores.
    tools = _get_search_tools(state.get("session_id", ""))
    synthesis, chunks = _execute_research(state["topic"], tools)

    note = f" — {len(chunks)} passages retrieved" if chunks else ""
    return {
        "search_results": [synthesis],
        "retrieved_chunks": chunks,
        "messages": [AIMessage(
            content=f"[Researcher] Research complete on '{state['topic']}'{note}"
        )],
    }


# ── Writer node ───────────────────────────────────────────────────────────────

# Last-resort character ceiling for the writer's prompt.
#
# This is a SAFETY NET, not the primary mechanism. The primary fix is the context
# window itself: num_ctx now defaults to 16384 (see core/llm.py), at which a
# 12-passage, 11,791-char prompt produces a correct 553-word report. Under the
# old 8192 the same prompt returned nothing.
#
# Dropping evidence to fit a window is a quality compromise and is only
# acceptable when the alternative is a silent failure, so the ceiling sits ABOVE
# every prompt the system normally builds — a full 12-passage revision measures
# ~13,000 chars. It exists because the prompt can still grow three ways inside
# ONE run: the researcher's loop resends its history each step, `search_results`
# accumulates across researcher passes, and a REVISION adds the previous draft
# (+49% measured). The revise path is the most likely to overflow, and it is the
# project's headline HITL feature.
#
# Raise OLLAMA_NUM_CTX before raising this: more window costs latency, but
# trimming costs evidence.
_WRITER_PROMPT_BUDGET = int(os.getenv("ATHENA_WRITER_PROMPT_BUDGET", "14000"))


def _fit(research: str, sources: str, revision: str, system_len: int) -> tuple[str, str, str]:
    """
    Trim the writer's prompt to the budget, sacrificing the least useful part first.

    Priority order, most protected first:
      1. the revision block — without the previous draft, a "targeted revision"
         silently becomes a rewrite, which is a correctness failure, not a
         cosmetic one;
      2. the retrieved passages — the only thing citations can point at;
      3. the researcher's prose synthesis — the most redundant, since the
         passages it summarises are already present verbatim.
    """
    room = _WRITER_PROMPT_BUDGET - system_len - 200  # 200 for the fixed scaffolding
    if room <= 0:
        return research[:500], sources[:1000], revision

    if len(research) + len(sources) + len(revision) <= room:
        return research, sources, revision

    # 1. Squeeze the synthesis first.
    keep_research = max(600, room - len(sources) - len(revision))
    if len(research) > keep_research:
        research = research[:keep_research].rstrip() + "\n[... synthesis truncated]"
    if len(research) + len(sources) + len(revision) <= room:
        return research, sources, revision

    # 2. Then drop whole passages from the end — never mid-passage, which would
    #    leave a citation pointing at half a sentence.
    if sources:
        blocks = sources.split("\n\n")
        header, items = blocks[0], blocks[1:]
        while items and len(research) + len("\n\n".join([header, *items])) + len(revision) > room:
            items.pop()
        sources = "\n\n".join([header, *items]) if items else ""
    if len(research) + len(sources) + len(revision) <= room:
        return research, sources, revision

    # 3. Last resort: shorten the previous draft, keeping its head and tail so
    #    the model still sees the structure it is meant to preserve.
    if len(revision) > 1200:
        revision = revision[:800] + "\n[... draft truncated ...]\n" + revision[-400:]
    return research, sources, revision


def writer_node(state: ResearchState) -> dict:
    """
    Produces or revises the structured markdown report.

    If human_feedback is set, incorporates it surgically.
    Clears human_feedback after use so the revision flag resets.
    """
    llm = get_llm()

    research_context = "\n\n---\n\n".join(state.get("search_results", []))
    feedback = state.get("human_feedback", "").strip()

    # In document mode the writer is given the verbatim passages, numbered, in
    # addition to the researcher's synthesis. Citing a synthesis is not citing a
    # source: without the passages the [n] references would be fabricated, and
    # the faithfulness metric would be scoring prose against prose.
    chunks = state.get("retrieved_chunks", []) or []
    sources_block = ""
    if chunks:
        listed = []
        for i, c in enumerate(chunks[:12], start=1):
            loc = f" — {c.get('locator')}" if c.get("locator") else ""
            text = (c.get("text") or "").strip()
            if len(text) > 700:
                text = text[:700].rstrip() + "..."
            listed.append(f"[{i}] {c.get('source', '?')}{loc}\n{text}")
        sources_block = (
            "\n\nRETRIEVED PASSAGES — cite these by number, and use no other source:\n"
            + "\n\n".join(listed)
        )

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

    system = WRITER_DOCUMENTS_SYSTEM if _document_mode() else WRITER_SYSTEM

    # Fit before calling, rather than discovering the overflow as an empty reply.
    research_context, sources_block, revision_block = _fit(
        research_context, sources_block, revision_block, len(system)
    )

    def _write(src_block: str) -> str:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=(
                f"Write a research report on: **{state['topic']}**\n\n"
                f"Research gathered:\n{research_context}"
                f"{src_block}"
                f"{revision_block}"
            )),
        ])
        return (resp.content or "").strip()

    draft = _write(sources_block)

    # An EMPTY draft must never be returned silently.
    #
    # Observed on the local backend: a question that retrieved 11 passages (the
    # correct figure ranked first) produced a completely empty report. Retrieval
    # was perfect; the writer returned "". The graph happily carried the empty
    # string to the review gate, so the user would have been shown a blank report
    # with no indication anything had gone wrong — and the eval could only detect
    # it as a missing answer, which reads like a retrieval problem and sends you
    # debugging the wrong component.
    #
    # The likely cause is prompt size: passages plus synthesis can exceed the
    # local model's context window, and Ollama truncates from the front, cutting
    # the instructions off. So the retry halves the evidence block rather than
    # simply asking again, and a persistent failure becomes a LOUD, visible
    # message instead of silence.
    retried = False
    if not draft and sources_block:
        retried = True
        half = max(1, len(chunks) // 2)
        listed = []
        for i, c in enumerate(chunks[:half], start=1):
            loc = f" — {c.get('locator')}" if c.get("locator") else ""
            text = (c.get("text") or "").strip()
            if len(text) > 400:
                text = text[:400].rstrip() + "..."
            listed.append(f"[{i}] {c.get('source', '?')}{loc}\n{text}")
        draft = _write(
            "\n\nRETRIEVED PASSAGES — cite these by number, and use no other source:\n"
            + "\n\n".join(listed)
        )

    if not draft:
        draft = (
            "## Report generation failed\n\n"
            "The model returned an empty report for this question, even after "
            "retrying with a reduced set of passages. This is a generation "
            "failure, not an empty archive: "
            f"{len(chunks)} relevant passage(s) were retrieved successfully.\n\n"
            "The most likely cause is that the prompt exceeded the local model's "
            "context window. Try a more specific question, raise OLLAMA_NUM_CTX, "
            "or switch the backend to Groq.\n\n"
            "No content is shown rather than a partial or invented one."
        )

    note = "Draft report generated."
    if retried:
        note = "Draft report generated (first attempt returned empty; retried with fewer passages)."

    return {
        "draft_report": draft,
        "human_feedback": "",   # reset — writer has consumed the feedback
        "messages": [AIMessage(content=f"[Writer] {note}")],
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
