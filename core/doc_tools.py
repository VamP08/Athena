"""
core/doc_tools.py
The tools that give the EXISTING researcher agent access to the document corpus.

This is the whole point of document mode's design: the graph does not change.
`_get_search_tools()` hands the researcher a different set of tools, and the
supervisor / researcher / writer / review topology is untouched. New capability,
zero graph surgery.

Two tools, deliberately — not one, and not five:

  document_search   semantic + lexical retrieval over the corpus
  list_documents    the corpus inventory

`list_documents` exists because semantic search structurally cannot answer
"how many invoices are there" or "which years do we hold". Aggregate questions
over a corpus are a well-documented RAG failure mode: the retriever returns the
k most similar chunks, which says nothing about totals. Reading the manifest
directly is the exact, cheap fix, and it also lets the agent orient before it
searches — which matters for "write me a report on our finances", where the
model has to know what exists before it can plan.

Docstrings here are STATIC. A dynamically built f-string placed as the first
statement of a function is not a docstring at all — Python discards it, __doc__
is None, and @tool then raises. The doc_type vocabulary is therefore written out
literally, which is also what a 9B local model needs: a closed, visible set of
values it can copy, rather than an open-ended field it will hallucinate into.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool

from . import index as idx

# Per-call character budget for what a tool returns to the model. Sized so a
# full k=6 result set still fits comfortably inside the researcher's tool-output
# allowance; see _TOOL_CHAR_BUDGET in core/nodes.py.
_SEARCH_BUDGET = int(os.getenv("ATHENA_SEARCH_CHAR_BUDGET", "6000"))
_DEFAULT_K = int(os.getenv("ATHENA_SEARCH_K", "6"))

VALID_DOC_TYPES = (
    "annual_report", "balance_sheet", "invoice", "ledger", "statement",
    "audit", "contract", "tax", "budget", "payroll", "other",
)


def _fmt_hit(i: int, h: dict, budget: int) -> str:
    """One retrieved chunk, rendered so the model can cite it verbatim."""
    where = h.get("locator") or ""
    head = f"[{i}] {h['source']}" + (f" — {where}" if where else "")
    body = (h.get("text") or "").strip()
    if len(body) > budget:
        body = body[: budget - 3].rstrip() + "..."
    return f"{head}\n{body}"


@tool(response_format="content_and_artifact")
def document_search(query: str, doc_type: str = "", year: str = "") -> tuple:
    """Search the institution's financial document archive and return matching passages.

    Use this for every factual question about the organisation's finances. It searches
    both by meaning and by exact wording, so it finds figures, account names, invoice
    numbers, dates and clauses. Documents are in German and English; a question in one
    language will still find documents in the other.

    Args:
        query: What to look for. A natural question or keywords both work.
               Prefer specific terms, e.g. "Umsatzerlöse Q3 2025" or "equity 2024".
        doc_type: Optional filter. One of: annual_report, balance_sheet, invoice,
               ledger, statement, audit, contract, tax, budget, payroll, other.
               Leave empty to search everything.
        year: Optional four-digit year filter, e.g. "2024". Leave empty for all years.
    """
    used_fallback = ""
    hits = idx.search(query, k=_DEFAULT_K, doc_type=doc_type.strip(), year=year.strip())

    # A filter the model invented (or one that is simply too narrow) must never
    # dead-end the research. Retry unfiltered and SAY SO, so the agent knows the
    # results are broader than it asked for rather than silently trusting them.
    if not hits and (doc_type or year):
        hits = idx.search(query, k=_DEFAULT_K)
        if hits:
            used_fallback = (
                f"(No document matched doc_type='{doc_type}' year='{year}'. "
                f"Showing unfiltered results instead.)\n\n"
            )

    if not hits:
        return (
            f"No passage in the archive matched '{query}'. "
            f"Try different wording, or call list_documents to see what the archive contains.",
            [],
        )

    per = max(300, _SEARCH_BUDGET // max(1, len(hits)))
    body = "\n\n".join(_fmt_hit(i, h, per) for i, h in enumerate(hits, start=1))

    artifact = [
        {
            "source": h["source"],
            "locator": h.get("locator") or "",
            "kind": h["kind"],
            "doc_type": h.get("doc_type") or "",
            "year": h.get("year") or "",
            "text": h["text"],
            "score": h.get("score"),
        }
        for h in hits
    ]
    return used_fallback + body, artifact


@tool
def list_documents(doc_type: str = "", year: str = "") -> str:
    """List the documents held in the archive, with their type, year and size.

    Call this FIRST when asked to write a report about the organisation, or whenever
    you need to know what the archive contains, how many documents of a kind exist,
    or which years are covered. Semantic search cannot answer counting questions —
    this can.

    Args:
        doc_type: Optional filter. One of: annual_report, balance_sheet, invoice,
               ledger, statement, audit, contract, tax, budget, payroll, other.
        year: Optional four-digit year filter, e.g. "2024".
    """
    docs = idx.list_documents(doc_type=doc_type.strip(), year=year.strip())
    if not docs:
        return (
            f"No documents matched doc_type='{doc_type}' year='{year}'. "
            f"Call list_documents with no arguments to see everything."
        )

    stats = idx.index_stats()
    lines = [
        f"The archive holds {stats['documents']} documents "
        f"({stats['chunks']} indexed passages).",
        "",
    ]
    for d in docs:
        flag = "" if d["parse_status"] == "ok" else f"  [!{d['parse_status']}]"
        lines.append(
            f"- {d['source']} | type={d['doc_type'] or 'other'} | "
            f"year={d['year'] or 'n/a'} | {d['n_chunks']} passages{flag}"
        )

    # A document that only partly parsed is reported here on purpose. "The
    # archive does not contain that" and "that file could not be read" are
    # different answers, and conflating them in a financial context is a
    # correctness failure, not a cosmetic one.
    partial = [d for d in docs if d["parse_status"] != "ok"]
    if partial:
        lines.append("")
        lines.append(
            "Note: documents marked [!partial] or [!error] were not fully indexed "
            "(for example a scanned page with no text layer, or spreadsheet totals "
            "stored only as formulas). Search them with care and say so if you rely on them."
        )
    return "\n".join(lines)


def get_document_tools() -> list:
    """The tool set for document mode."""
    return [document_search, list_documents]
