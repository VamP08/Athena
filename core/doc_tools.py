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

`aggregate_documents` is the same argument one level down. Knowing the archive
holds one invoice register does not tell you how many invoices are IN it, or
what they come to. Asking the retriever returns six rows out of sixty and the
model adds up the six — an answer that is not vague but specifically wrong, and
that gets wronger as the archive grows. So counting is answered by SQL over
every matching row instead (core/facts.py), and this tool is the agent's door
to it.

Three tools, then, and the third earns its place by being the only one that can
be exactly right about a quantity.

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

# The corpus can hold a thousand quarterly registers. Listing every one of their
# column sets would bury the inventory the model actually needs.
_MAX_TABLES_LISTED = int(os.getenv("ATHENA_MAX_TABLES_LISTED", "25"))

# Provenance inside a single answer is tighter still: the point is to show the
# shape of the scope, not to reprint it.
_MAX_TABLES_IN_ANSWER = 8

VALID_DOC_TYPES = (
    "annual_report", "balance_sheet", "invoice", "ledger", "statement",
    "audit", "contract", "tax", "budget", "payroll", "other",
)


def _as_artifact(h: dict, scope: str) -> dict:
    """
    One retrieved passage as structured evidence for the writer and for Ragas.

    `scope` rides along so a citation can state whether a source is archived or
    attached — a reader must never confuse "this is in our records" with "this is
    in the file you just handed me".
    """
    return {
        "source": h["source"],
        "locator": h.get("locator") or "",
        "kind": h["kind"],
        "doc_type": h.get("doc_type") or "",
        "year": h.get("year") or "",
        "text": h["text"],
        "score": h.get("score"),
        "scope": scope,
    }


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

    # Told to the MODEL, not only to the operator, and this is the highest-value
    # place to say it. The measured failure past the tested size is that the top
    # result is a document about the right client and the right quarter but the
    # wrong TYPE, and the model then quotes its figure without hesitating —
    # published work finds a highly-ranked distractor does more damage than a
    # random one precisely because rank reads as evidence. Warning it here is
    # what turns a confident wrong figure into a checked one.
    if not stats.get("within_tested_envelope", True):
        lines.append(
            f"WARNING: this archive is LARGER than the {stats['tested_doc_limit']} "
            f"documents Athena has been measured on. Above that size, search "
            f"sometimes ranks first a document about the right client and period "
            f"but of the WRONG TYPE (a statement instead of a budget). Before you "
            f"quote any figure, check that the passage's file name matches the "
            f"document type the question asked about, and say which file each "
            f"figure came from."
        )
        lines.append("")
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

    # The column vocabulary for aggregate_documents. Without it the model has to
    # guess column names, and a 9B model guesses `amount` for `Nettobetrag (EUR)`
    # every time — so the tool that must be exact would be driven by invention.
    wanted = {d["source"] for d in docs}
    tables = [t for t in idx.tables_overview() if t["source"] in wanted]
    if tables:
        lines.append("")
        lines.append(
            f"TABLES you can count or total exactly with aggregate_documents "
            f"({len(tables)} of them):"
        )
        for t in tables[:_MAX_TABLES_LISTED]:
            where = f" ({t['where']})" if t["where"] else ""
            cols = ", ".join(f"{c['name']} [{c['role']}]" for c in t["columns"])
            lines.append(f"- {t['source']}{where} — {t['rows']} rows | {cols}")
        if len(tables) > _MAX_TABLES_LISTED:
            lines.append(
                f"... and {len(tables) - _MAX_TABLES_LISTED} more. Narrow with "
                f"doc_type or year to see them."
            )
        lines.append(
            "Column roles say what may be done with a column: [amount] can be "
            "summed; [balance] is a running total, so ask aggregate_documents for "
            "operation=closing instead of summing it — do not estimate it from "
            "search results; [identifier] and [period] label rows rather than "
            "measuring them."
        )
    return "\n".join(lines)


def _fmt_num(v) -> str:
    """German number formatting: the reports are German and 4581668.99 is not."""
    if v is None:
        return "—"
    if isinstance(v, int) or float(v).is_integer():
        return f"{int(v):,}".replace(",", ".")
    return f"{v:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _fmt_table_line(p: dict) -> str:
    where = f" ({p['where']})" if p.get("where") else ""
    val = p.get("value")
    shown = _fmt_num(val) if isinstance(val, (int, float)) else "—"
    excl = f", {p['totals_excluded']} total row(s) excluded" if p["totals_excluded"] else ""
    return (
        f"- {p['source']}{where}: {shown}  "
        f"[{p['rows_matched']} of {p['rows_total']} rows{excl}]"
    )


def _render(r: dict) -> str:
    """
    One aggregate result as text the model can quote.

    The number comes first and the scope comes with it, always. A total whose
    scope is invisible cannot be checked by the person reading the report, and a
    figure nobody can check is exactly what this project exists not to produce.
    """
    if not r.get("ok") and r.get("refused"):
        lines = [
            f"REFUSED: \"{r['column']}\" in {r['source']} cannot be used with "
            f"{r['operation'].upper()}, because {r['reason']}."
        ]
        c = r.get("closing") or {}
        if c:
            lines.append(
                f"The figure you almost certainly want is the CLOSING BALANCE: "
                f"{_fmt_num(c['value'])}"
                + (f" on {c['date']}" if c.get("date") else "")
                + f" ({c['source']}, {c['locator']})."
            )
        if r.get("alternatives"):
            lines.append(
                "To total the movements instead, use column "
                + " or ".join(f'"{a}"' for a in r["alternatives"])
                + "."
            )
        return "\n".join(lines)

    if not r.get("ok"):
        lines = [r.get("error", "The aggregation could not be run.")]
        if r.get("operations"):
            lines.append("Valid operations: " + ", ".join(r["operations"]) + ".")
        for w in r.get("warnings", []):
            lines.append(f"- {w}")
        for t in (r.get("available") or [])[:12]:
            where = f" ({t['where']})" if t.get("where") else ""
            cols = ", ".join(
                f"{c['name']} [{c['role']}]" for c in t.get("columns", [])
            )
            lines.append(f"- {t['source']}{where}: {cols}")
        return "\n".join(lines)

    op = r["operation"]

    if op == "closing":
        lines = []
        for c in r["closing"]:
            where = f" ({c['where']})" if c.get("where") else ""
            lines.append(
                f"CLOSING VALUE of \"{c['column']}\" = {_fmt_num(c['value'])}"
                + (f" as at {c['date']}" if c.get("date") else "")
                + f" ({c['source']}{where}, {c['locator']})."
            )
        lines.append(
            "This is the LAST row in date order, not the largest — an account that "
            "peaked mid-period and then paid out has a maximum that is not its balance."
        )
        if r.get("filters"):
            lines.append("Filter: " + "; ".join(r["filters"]))
        for w in r.get("warnings", []):
            lines.append(f"CHECK: {w}")
        return "\n".join(lines)

    if op == "breakdown":
        lines = [f"BREAKDOWN of \"{r['column']}\" over {r['matched_rows']} rows:"]
        for g in r["groups"]:
            amt = (
                f" · {_fmt_num(g['sum'])} {g['amount_column']}"
                if g["sum"] is not None else ""
            )
            lines.append(f"- {g['key']}: {g['count']} row(s){amt}")
        if r["groups_truncated"]:
            lines.append(f"... and {r['groups_truncated']} more distinct value(s).")
    else:
        head = {
            "count": "COUNT", "sum": "SUM", "average": "AVERAGE",
            "min": "MINIMUM", "max": "MAXIMUM", "distinct": "DISTINCT VALUES",
        }[op]
        col = f' of "{r["column"]}"' if r.get("column") else ""
        lines = [
            f"{head}{col} = {_fmt_num(r['value'])}"
            + (" rows" if op in ("count", "distinct") else "")
        ]
        lines.append(
            f"Computed over EVERY matching row: {r['matched_rows']} of "
            f"{r['scanned_rows']} rows in {r['tables']} table(s), "
            f"{r['documents']} document(s). This is exact, not a top-k sample."
        )

    if r.get("filters"):
        lines.append("Filter: " + "; ".join(r["filters"]))
    if r.get("totals_excluded"):
        lines.append(
            f"{r['totals_excluded']} subtotal/total row(s) were excluded so they "
            f"are not counted twice."
        )
    # The per-table breakdown is the provenance, but a 266-table scope would
    # bury the answer and overrun the researcher's tool-output budget — which
    # truncates from the END, so an unbounded list keeps the number and drops
    # the CHECK warnings that make it safe to quote. Show the largest
    # contributors and account for the rest.
    per = r.get("per_table") or []
    if len(per) > 1:
        shown = sorted(per, key=lambda p: -(p["rows_matched"] or 0))[:_MAX_TABLES_IN_ANSWER]
        lines.append(
            f"Per table ({len(per)} in scope"
            + (f", {len(shown)} largest shown" if len(per) > len(shown) else "")
            + "):"
        )
        lines.extend(_fmt_table_line(p) for p in shown)
        if len(per) > len(shown):
            rest = sum(p["rows_matched"] or 0 for p in per) - sum(
                p["rows_matched"] or 0 for p in shown
            )
            lines.append(
                f"- ... {len(per) - len(shown)} further table(s) contributing "
                f"{rest} row(s)."
            )

    ov = r.get("overlap") or {}
    if ov.get("documents_without_rows"):
        lines.append(
            "NOT included (these documents match the filter but hold no table rows, "
            "so they are documents rather than rows): "
            + ", ".join(ov["documents_without_rows"])
        )
    if ov.get("identifiers_in_multiple_tables"):
        lines.append(
            "CHECK: these identifiers appear in more than one table, so the same "
            "record may be represented twice: "
            + ", ".join(ov["identifiers_in_multiple_tables"])
        )
    for w in r.get("warnings", []):
        lines.append(f"CHECK: {w}")
    return "\n".join(lines)


@tool
def aggregate_documents(
    operation: str,
    column: str = "",
    source: str = "",
    doc_type: str = "",
    year: str = "",
    where_column: str = "",
    where_value: str = "",
    period: str = "",
) -> str:
    """Count or total the rows of the archive's tables EXACTLY. Use this for any number.

    Search returns only the most similar few rows, so it can never answer "how many"
    or "what is the total" — it would add up six invoices out of sixty. This reads
    every matching row instead. Always use this tool for counts, totals and averages.

    Examples:
        operation="count", doc_type="invoice", period="Q3 2025"
        operation="count", source="Rechnungsausgang", where_column="Status", where_value="offen"
        operation="sum", column="Nettobetrag", source="Rechnungsausgang"
        operation="breakdown", column="Status", source="Rechnungsausgang"
        operation="closing", column="Saldo", source="Kontoauszug"

    Call list_documents first if you do not know the column names.

    Args:
        operation: One of: count, sum, average, min, max, distinct, breakdown, closing.
               "count" counts rows. "breakdown" groups rows by `column` and counts each.
               "closing" gives a running balance's final value, in date order — use it
               for any [balance] column, which must never be summed.
        column: The column to total or group by, e.g. "Nettobetrag" or "Status".
               Not needed for count. Partial names work.
        source: Optional file name, or part of one, e.g. "Rechnungsausgang".
        doc_type: Optional filter. One of: annual_report, balance_sheet, invoice,
               ledger, statement, audit, contract, tax, budget, payroll, other.
        year: Optional four-digit year of the DOCUMENT, e.g. "2024".
        where_column: Optional column to filter rows on, e.g. "Status".
        where_value: The value that column must have, e.g. "offen".
        period: Optional date range for the rows, e.g. "Q3 2025", "2025-07" or "2025".
    """
    return _render(idx.aggregate(
        operation=operation.strip(),
        column=column.strip(),
        source=source.strip(),
        doc_type=doc_type.strip(),
        year=year.strip(),
        where_column=where_column.strip(),
        where_value=where_value.strip(),
        period=period.strip(),
    ))


def get_document_tools(session_id: str = "") -> list:
    """
    The tool set for document mode.

    With no session_id this is the archive-only set. With one, `document_search`
    is replaced by a closure that ALSO searches that chat's attached files. The
    binding is a closure over one SessionStore object, not a filter argument — so
    another chat has no reference to it and therefore no way to reach it, even if
    a query or a filter were malformed.

    `aggregate_documents` is archive-only in both cases, deliberately. Session
    uploads live in a private in-memory store with no fact tables, and quietly
    folding them into an archive total would produce a figure that cannot be
    reproduced from the archive it claims to describe.
    """
    if not session_id:
        return [document_search, list_documents, aggregate_documents]

    from . import sessions

    store = sessions.REGISTRY.get(session_id)
    if store is None:
        return [document_search, list_documents, aggregate_documents]

    @tool(response_format="content_and_artifact")
    def search_documents(query: str, doc_type: str = "", year: str = "") -> tuple:
        """Search the financial archive AND the documents attached to this chat.

        Use this for every factual question. It searches by meaning and by exact
        wording, across German and English documents. Results are shown in two
        clearly separated groups: files ATTACHED TO THIS CHAT, and the permanent
        ARCHIVE. When they disagree, say so and cite both — do not silently
        prefer one.

        Args:
            query: What to look for. A natural question or keywords both work.
            doc_type: Optional filter, e.g. invoice, ledger, statement, audit,
                   contract, tax, budget, annual_report. Leave empty for all.
            year: Optional four-digit year, e.g. "2024". Leave empty for all.
        """
        dt, yr = doc_type.strip(), year.strip()
        # Deliberately NOT fused into one ranked list. RRF ranks are
        # corpus-relative: rank 1 among 40 attached passages is not the same
        # evidence strength as rank 1 among 900 archived ones, so merging them
        # would silently promote whichever corpus is smaller.
        session_hits = store.search(query, k=3, doc_type=dt, year=yr)
        archive_hits = idx.search(query, k=_DEFAULT_K, doc_type=dt, year=yr)

        if not session_hits and not archive_hits:
            return (
                f"Nothing in the attached documents or the archive matched '{query}'.",
                [],
            )

        blocks, artifact, n = [], [], 0
        per = max(300, _SEARCH_BUDGET // max(1, len(session_hits) + len(archive_hits)))

        if session_hits:
            blocks.append("ATTACHED TO THIS CHAT (not part of the archive):")
            for h in session_hits:
                n += 1
                blocks.append(_fmt_hit(n, h, per))
                artifact.append(_as_artifact(h, "session"))
        if archive_hits:
            blocks.append("\nARCHIVE (permanent records):")
            for h in archive_hits:
                n += 1
                blocks.append(_fmt_hit(n, h, per))
                artifact.append(_as_artifact(h, "knowledge_base"))

        return "\n\n".join(blocks), artifact

    return [search_documents, list_documents, aggregate_documents]
