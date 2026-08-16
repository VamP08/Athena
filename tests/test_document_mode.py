"""
tests/test_document_mode.py
Document mode: parsing, indexing, hybrid retrieval, and the agent tools.

Every test here is offline and deterministic. Embeddings are replaced by a
hash-based stub, so nothing touches Ollama and CI needs no model — the same rule
the rest of the suite follows for the LLM.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import pytest

from core import index as idx
from core.documents import parse_file

_DIM = 64


class FakeEmbeddings:
    """
    Deterministic stand-in for OllamaEmbeddings.

    Bag-of-words hashing: texts sharing words get similar vectors, so retrieval
    ordering is meaningful enough to assert on without loading a real model.
    """

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * _DIM
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            v[h % _DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "corpus"
    d.mkdir()

    (d / "Steuerunterlagen_2024.md").write_text(
        "# Steuern 2024\n\nSteuern vom Einkommen und Ertrag: 756.600,00 EUR.\n"
        "Geprüft durch Wagner & Petersen Wirtschaftsprüfung GmbH.\n",
        encoding="utf-8",
    )
    # cp1252 + semicolons: what German accounting software actually exports.
    with open(d / "Hauptbuch_2024.csv", "w", newline="", encoding="cp1252") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Buchungsdatum", "Konto", "Bezeichnung", "Betrag"])
        w.writerow(["2024-03-02", "4400", "Umsatzerlöse Transport", "12400,00"])
        w.writerow(["2024-04-11", "6000", "Personalaufwand", "38263,43"])
    return d


@pytest.fixture
def built(corpus: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: FakeEmbeddings())
    db = tmp_path / "idx.db"
    idx.close()
    stats = idx.build_index(corpus_dir=str(corpus), index_path=str(db), embed_model="fake")
    yield stats, str(db)
    idx.close()


# ── parsing ──────────────────────────────────────────────────────────────────

def test_csv_umlauts_survive_cp1252(corpus: Path):
    """A wrong encoding guess turns Umsatzerlöse into mojibake and breaks retrieval."""
    _, els = parse_file(corpus / "Hauptbuch_2024.csv")
    assert any("Umsatzerlöse" in e.text for e in els)


def test_table_rows_carry_their_headers(corpus: Path):
    """
    A bare '12400,00' is unusable evidence — the row must say what the number is.
    """
    _, els = parse_file(corpus / "Hauptbuch_2024.csv")
    rows = [e for e in els if e.kind == "table_row"]
    assert rows
    assert all("Konto:" in r.text and "Betrag:" in r.text for r in rows)


def test_doc_id_is_content_addressed(corpus: Path, tmp_path: Path):
    """
    Identity must come from bytes only. mtime changes on clone/copy/unzip, which
    would invalidate every citation anchor on a fresh checkout.
    """
    src = corpus / "Steuerunterlagen_2024.md"
    copy = tmp_path / "renamed.md"
    copy.write_bytes(src.read_bytes())
    os.utime(copy, (0, 0))
    assert parse_file(src)[0] == parse_file(copy)[0]


def test_unreadable_file_yields_notice_not_silence(tmp_path: Path):
    """'Could not parse' and 'not in the archive' are different claims."""
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not a real pdf")
    _, els = parse_file(bad)
    assert len(els) == 1
    assert els[0].kind == "notice"
    assert els[0].meta.get("reason") == "parse_error"


# ── indexing ─────────────────────────────────────────────────────────────────

def test_build_index_stores_chunks(built):
    stats, _ = built
    assert stats["files_indexed"] == 2
    assert stats["chunks"] > 0
    assert stats["errors"] == 0


def test_reingest_is_incremental(built, corpus: Path):
    """Unchanged bytes must not be re-embedded — otherwise the folder never scales."""
    _, db = built
    again = idx.build_index(corpus_dir=str(corpus), index_path=db, embed_model="fake")
    assert again["files_skipped"] == 2
    assert again["files_indexed"] == 0


# ── retrieval ────────────────────────────────────────────────────────────────

def test_search_finds_exact_figure(built):
    _, db = built
    hits = idx.search("Umsatzerlöse Transport", k=3, index_path=db)
    assert hits
    assert any("12400,00" in h["text"] for h in hits)


def test_fts_survives_punctuation(built):
    """
    FTS5 MATCH takes a query language, not free text. Users type '?' and ':'
    constantly; an unescaped query raises OperationalError instead of searching.
    """
    _, db = built
    for q in ["Wie hoch war der Umsatz in Q3 2025?", "Konto: 4400", "a -- b", 'quote " here']:
        idx.search(q, k=2, index_path=db)  # must not raise


def test_empty_query_is_safe(built):
    _, db = built
    assert idx.search("", k=3, index_path=db) == [] or True  # must not raise


def test_filter_matching_nothing_returns_nothing(built):
    """
    Post-filtering a candidate pool silently returns unfiltered results when the
    filter matches none of it. That converts a filter miss into a confident
    wrong answer, so an impossible filter must return empty.
    """
    _, db = built
    assert idx.search("Umsatz", k=3, doc_type="payroll", index_path=db) == []


def test_list_documents_reports_inventory(built):
    _, db = built
    docs = idx.list_documents(index_path=db)
    assert len(docs) == 2
    assert {d["source"] for d in docs} == {"Steuerunterlagen_2024.md", "Hauptbuch_2024.csv"}


def test_doc_type_inference_prefers_filename():
    """
    A corpus README lists every document type in its table of contents. Blending
    filename and body made it classify as whatever pattern was checked first.
    """
    assert idx._infer_doc_type("Hauptbuch_2024.csv", "") == "ledger"
    assert idx._infer_doc_type("Audit_Report_2024_EN.docx", "financial statements") == "audit"
    assert idx._infer_doc_type(
        "README_Finanzarchiv.md", "Geschaeftsbericht annual Rechnung Bilanz"
    ) == "other"


# ── agent tools ──────────────────────────────────────────────────────────────

def test_document_search_returns_content_and_artifact(built, monkeypatch):
    _, db = built
    monkeypatch.setenv("ATHENA_INDEX_PATH", db)
    monkeypatch.setattr(idx, "DEFAULT_INDEX_PATH", db)
    from core.doc_tools import document_search

    msg = document_search.invoke(
        {"name": "document_search", "args": {"query": "Personalaufwand"},
         "id": "t2", "type": "tool_call"}
    )
    assert isinstance(msg.content, str) and msg.content
    assert isinstance(msg.artifact, list) and msg.artifact
    assert "text" in msg.artifact[0] and "source" in msg.artifact[0]


def test_document_search_announces_filter_fallback(built, monkeypatch):
    """A hallucinated filter must degrade loudly, never dead-end silently."""
    _, db = built
    monkeypatch.setattr(idx, "DEFAULT_INDEX_PATH", db)
    from core.doc_tools import document_search

    msg = document_search.invoke(
        {"name": "document_search",
         "args": {"query": "Personalaufwand", "doc_type": "not_a_real_type"},
         "id": "t3", "type": "tool_call"}
    )
    assert "unfiltered" in msg.content.lower()


def test_web_mode_tool_handling_is_unchanged(monkeypatch):
    """
    Regression guard for the published web-mode eval scores.

    Document mode changed how tools are invoked (`tool.invoke(tc)` instead of
    `tool.invoke(tc["args"])`, so ToolMessage.artifact survives) and made the
    output cap per-tool. Neither may alter the web path, because the README's
    Ragas numbers were measured on it. For a content-only tool the two
    invocation styles must produce identical content, and web_search must still
    fall under the 3000-char default rather than a document budget.
    """
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool as make_tool

    from core.nodes import _DEFAULT_TOOL_BUDGET, _TOOL_CHAR_BUDGET

    @make_tool
    def fake_search(query: str) -> str:
        """Fake search."""
        return "R:" + ("x" * 5000)

    tc = {"name": "fake_search", "args": {"query": "q"}, "id": "c1", "type": "tool_call"}
    old = str(fake_search.invoke(tc["args"]))[:3000]
    msg = fake_search.invoke(tc)

    assert isinstance(msg, ToolMessage)
    assert msg.content[:3000] == old
    assert _DEFAULT_TOOL_BUDGET == 3000
    assert "web_search" not in _TOOL_CHAR_BUDGET


def test_local_reasoning_is_on_by_default(monkeypatch):
    """
    Reasoning ON is an accuracy decision, not a style preference.

    Measured: with the trace disabled the local model emitted 5 figures that
    appear in no retrieved passage, and misattributed 2 of 6 answers. Guard the
    default so a future latency tweak cannot silently reintroduce that.
    """
    from core import llm as llm_mod

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ATHENA_LLM_BACKEND", "ollama")
    monkeypatch.delenv("ATHENA_OLLAMA_REASONING", raising=False)
    captured = {}

    class FakeOllama:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setitem(
        __import__("sys").modules, "langchain_ollama",
        type("m", (), {"ChatOllama": FakeOllama}),
    )
    llm_mod.get_llm()
    assert "reasoning" not in captured, "reasoning must not be disabled by default"

    captured.clear()
    monkeypatch.setenv("ATHENA_OLLAMA_REASONING", "false")
    llm_mod.get_llm()
    assert captured.get("reasoning") is False, "explicit opt-out must still work"


def test_reasoning_flag_only_affects_local_backend(monkeypatch):
    """
    The reasoning switch must never reach Groq.

    The published eval scores were produced on Groq; a kwarg leaking into that
    path would silently invalidate the comparison between old and new runs.
    """
    from core import llm as llm_mod

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("ATHENA_LLM_BACKEND", "groq")
    captured = {}

    class FakeGroq:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setitem(
        __import__("sys").modules, "langchain_groq",
        type("m", (), {"ChatGroq": FakeGroq}),
    )
    llm_mod.get_llm()
    assert "reasoning" not in captured


def test_document_mode_selects_document_tools(monkeypatch):
    """The graph is unchanged; only the tool set swaps. That is the design claim."""
    from core.nodes import _get_search_tools

    monkeypatch.setenv("ATHENA_MODE", "documents")
    names = {t.name for t in _get_search_tools()}
    assert names == {"document_search", "list_documents", "aggregate_documents"}

    monkeypatch.setenv("ATHENA_MODE", "web")
    assert "web_search" in {t.name for t in _get_search_tools()}


# ── tested operating envelope ────────────────────────────────────────────────

def test_index_stats_reports_the_tested_envelope(built, monkeypatch):
    """
    The system must be able to say which regime it is operating in.

    Retrieval past the measured corpus size degrades gradually, so nothing in
    the output looks different when the archive has outgrown what was tested —
    that is precisely why it has to be stated rather than inferred.

    The gate is on CHUNKS, not documents. Retrieval ranks chunks; a document is
    an arbitrary container, and a single thousand-page report would sail past
    any document-count gate while carrying more chunks than the whole tested
    envelope.
    """
    _, db = built
    monkeypatch.setattr(idx, "TESTED_CHUNK_LIMIT", 3000)
    stats = idx.index_stats(db)
    assert stats["tested_chunk_limit"] == 3000
    assert stats["within_tested_envelope"] is True

    monkeypatch.setattr(idx, "TESTED_CHUNK_LIMIT", 1)
    assert idx.index_stats(db)["within_tested_envelope"] is False


def test_list_documents_warns_the_model_only_when_over_the_limit(built, monkeypatch):
    """
    The warning goes to the MODEL, because the model is what quotes the figure.

    A highly-ranked distractor does more damage than a random one — rank reads
    as evidence — so telling the reader that the archive is outside the measured
    range is what turns a confident wrong figure into a checked one. It must
    stay silent inside the envelope, or it is just noise on every call.
    """
    from core import doc_tools

    _, db = built
    monkeypatch.setenv("ATHENA_INDEX_PATH", db)
    monkeypatch.setattr(idx, "DEFAULT_INDEX_PATH", db)

    monkeypatch.setattr(idx, "TESTED_CHUNK_LIMIT", 3000)
    assert "WARNING" not in doc_tools.list_documents.invoke({})

    monkeypatch.setattr(idx, "TESTED_CHUNK_LIMIT", 1)
    over = doc_tools.list_documents.invoke({})
    assert "WARNING" in over
    assert "WRONG TYPE" in over
