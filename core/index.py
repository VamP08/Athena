"""
core/index.py
The document index: one SQLite file holding chunks, their embeddings, and a
full-text index, with hybrid (dense + lexical) retrieval fused by RRF.

Why one SQLite file rather than Chroma (see docs/DECISIONS.md):
  sqlite-vec is a 0.3 MB extension with zero Python dependencies, and SQLite
  already ships FTS5 — so the lexical half costs nothing at all. The project
  already depends on SQLite for the LangGraph checkpointer and the audit trail.
  For an on-premise product the operational story matters as much as the
  benchmark: "your entire corpus index is one file you can back up, encrypt,
  move, or delete" is a real DSGVO answer. Chroma would add a large transitive
  dependency tree for capability this corpus does not need.

Why hybrid and not dense-only:
  Financial questions are dominated by exact tokens — invoice numbers, "Q3 2025",
  account codes, IBANs. Dense embeddings represent these poorly; published work
  repeatedly finds BM25 beating strong embedding models on financial retrieval.
  Conversely, lexical alone cannot bridge "revenue" -> "Umsatzerlöse" across a
  bilingual corpus. Each half covers the other's blind spot, so both ship.

Why exact (brute-force) vector search rather than an ANN index:
  At institution-demo scale (thousands of chunks) an exact scan with
  vec_distance_cosine costs single-digit milliseconds, is always correct, and —
  critically — supports arbitrary PRE-filtering. ANN indexes force filtering to
  happen after the top-k is chosen, so a narrow filter like doc_type='invoice'
  can match nothing in the candidate pool and silently degrade to unfiltered
  results. Exactness here buys correct filtering, and vec0/ANN remains the
  documented upgrade path when the corpus outgrows a linear scan.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

from . import facts as fx
from .documents import ParsedElement, content_hash, iter_corpus, parse_file

DEFAULT_INDEX_PATH = os.getenv("ATHENA_INDEX_PATH", "athena_index.db")
DEFAULT_CORPUS_DIR = os.getenv("ATHENA_CORPUS_DIR", "corpus")
DEFAULT_EMBED_MODEL = os.getenv("ATHENA_EMBED_MODEL", "bge-m3")

# The archive size this system has been MEASURED at, not a hard cap.
#
# Retrieval degrades gradually rather than breaking, so nothing announces itself
# when a corpus outgrows what was tested — the published failure mode is that
# top-k quietly fills with passages that are topically right and factually
# wrong, and the reader model treats their rank as evidence. A system that
# cannot say which regime it is operating in is asking to be trusted outside
# the range anyone checked.
#
# The unit is CHUNKS, not documents, because chunks are what retrieval actually
# ranks — a document is an arbitrary container. The envelope sweep happened to
# use a uniform corpus (~15 chunks/document), so "200 documents" and "3,000
# chunks" named the same boundary there; on a real archive they do not. One
# thousand-page annual report is thousands of chunks on its own, and a
# document-count gate would wave it through as "1 document, fine".
#
# The number comes from eval/run_scale_envelope.py (where hit@1 last held 1.000)
# and eval/run_scale_composition.py (which separates chunk count from corpus
# composition). Raise it by re-running those, not by editing this line.
TESTED_CHUNK_LIMIT = int(os.getenv("ATHENA_TESTED_CHUNK_LIMIT", "3000"))

# One connection is shared process-wide, and api/main.py runs graph work on a
# ThreadPoolExecutor. sqlite3 objects are not safe to use concurrently from
# several threads even with check_same_thread=False, so every statement goes
# through this lock. api/registry.py already established this pattern.
_LOCK = threading.Lock()
_CONN: sqlite3.Connection | None = None
_CONN_PATH: str | None = None


# ── connection ───────────────────────────────────────────────────────────────

def connect(path: str | None = None) -> sqlite3.Connection:
    """Open (once) the index database with the sqlite-vec extension loaded."""
    global _CONN, _CONN_PATH
    target = str(path or DEFAULT_INDEX_PATH)

    with _LOCK:
        if _CONN is not None and _CONN_PATH == target:
            return _CONN
        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:
                pass

        conn = sqlite3.connect(target, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute("PRAGMA journal_mode=WAL")
        _init_schema(conn)
        _CONN, _CONN_PATH = conn, target
        return conn


def close() -> None:
    """Close the shared connection (used by tests to release file handles)."""
    global _CONN, _CONN_PATH
    with _LOCK:
        if _CONN is not None:
            try:
                _CONN.close()
            except Exception:
                pass
        _CONN, _CONN_PATH = None, None


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS index_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS documents (
            doc_id       TEXT PRIMARY KEY,
            source       TEXT NOT NULL,
            rel_path     TEXT,
            doc_type     TEXT,
            year         TEXT,
            n_elements   INTEGER DEFAULT 0,
            parse_status TEXT DEFAULT 'ok',
            parse_note   TEXT,
            ingested_at  REAL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id         INTEGER PRIMARY KEY,
            doc_id     TEXT NOT NULL,
            element_id TEXT NOT NULL,
            source     TEXT NOT NULL,
            kind       TEXT NOT NULL,
            text       TEXT NOT NULL,
            locator    TEXT,
            page       INTEGER,
            sheet      TEXT,
            row        INTEGER,
            table_id   TEXT,
            lang_hint  TEXT,
            doc_type   TEXT,
            year       TEXT,
            embedding  BLOB
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_doc  ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(doc_type);
        CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );

        -- The FTS5 index MUST be maintained by triggers, not by hand.
        --
        -- chunks_fts is an external-content table: it stores its own posting
        -- lists and only borrows the text from `chunks`. Deleting a row from
        -- `chunks` therefore does NOT remove its terms from the index, and
        -- SQLite reuses the freed rowid for the next insert. Reproduced on this
        -- machine: after deleting a document and inserting another, searching
        -- for the DELETED document's unique token still matched — and resolved
        -- to the NEW document's text. That is simultaneously a wrong answer and
        -- a data-protection failure, since supposedly deleted content stays
        -- searchable. Doing the delete by hand at every call site is exactly the
        -- kind of invariant that gets forgotten, so it is enforced in the schema.
        CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_fts_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_fts_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text)
            VALUES ('delete', old.id, old.text);
            INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
        END;
        """
    )
    # The relational half of the index: every table row, uncapped, for exact
    # aggregation. Retrieval and counting are different questions and this is
    # where they get different machinery (see core/facts.py).
    conn.executescript(fx.SCHEMA)
    _ensure_column(conn, "documents", "facts_at", "REAL")
    _ensure_column(conn, "fact_tables", "layout", "TEXT")
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """
    Add a column to an existing database if it is missing.

    Aggregation shipped after the index did, so an index built by the previous
    version is on disk and must not have to be rebuilt from scratch to gain the
    feature — a rebuild costs a full re-embed of the whole corpus.
    """
    have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if col not in have:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _meta_get(conn, key, default=None):
    r = conn.execute("SELECT value FROM index_meta WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def _meta_set(conn, key, value):
    conn.execute(
        "INSERT INTO index_meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


# ── embeddings ───────────────────────────────────────────────────────────────

def get_embeddings(model: str | None = None):
    """
    Embedding client, mirroring core/llm.py's env handling.

    OLLAMA_BASE_URL is honoured because in docker-compose Ollama is a sibling
    service, not localhost — omitting it silently breaks document mode inside
    the project's own container stack.
    """
    from langchain_ollama import OllamaEmbeddings

    kwargs = {}
    if os.getenv("OLLAMA_BASE_URL"):
        kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL")
    return OllamaEmbeddings(model=model or DEFAULT_EMBED_MODEL, **kwargs)


def _serialize(vec: list[float]) -> bytes:
    import sqlite_vec

    return sqlite_vec.serialize_float32(vec)


# ── ingestion ────────────────────────────────────────────────────────────────

def _infer_doc_type(name: str, text_sample: str = "") -> str:
    """
    Coarse document class from the filename, with a text fallback.

    Deliberately a small fixed vocabulary: these values are shown to the agent so
    it can filter, and a small closed set is far easier for a 9B local model to
    use correctly than an open-ended one.
    """
    # Ordered most-specific first: an auditor's report is full of the word
    # "statements", and a balance sheet mentions "Bilanz" inside an annual
    # report. Whichever pattern is checked first wins, so the order is the rule.
    table = [
        ("audit", ("pruefbericht", "prüfbericht", "audit", "testat", "auditor")),
        ("invoice", ("rechnung", "invoice", "faktura")),
        ("ledger", ("hauptbuch", "ledger", "buchung", "journal")),
        ("payroll", ("lohn", "gehalt", "payroll")),
        ("tax", ("steuer", "tax", "umsatzsteuer", "vat")),
        ("budget", ("budget", "forecast", "plan")),
        ("contract", ("vertrag", "contract", "vereinbarung")),
        ("balance_sheet", ("bilanz", "balance", "guv")),
        ("statement", ("kontoauszug", "statement", "auszug")),
        ("annual_report", ("geschaeftsbericht", "geschäftsbericht", "annual", "jahresabschluss")),
    ]

    # The FILENAME decides when it can. Only an uninformative name falls through
    # to the content, because a table of contents (or any document that merely
    # *lists* other documents, e.g. a corpus README) otherwise matches every
    # keyword at once and gets classified as whatever is checked first.
    n = name.lower()
    for label, keys in table:
        if any(k in n for k in keys):
            return label

    # An index/readme file lists what the archive contains, so its BODY names
    # every document type at once. Falling through to content would classify it
    # as whichever pattern is checked first. It is metadata, not a record.
    if any(k in n for k in ("readme", "index", "inhalt", "uebersicht", "übersicht")):
        return "other"

    body = text_sample[:400].lower()
    for label, keys in table:
        if any(k in body for k in keys):
            return label
    return "other"


def _infer_year(name: str, text_sample: str = "") -> str:
    for hay in (name, text_sample[:600]):
        m = re.findall(r"\b(19|20)\d{2}\b", hay)
        if m:
            found = re.findall(r"\b((?:19|20)\d{2})\b", hay)
            if found:
                return found[0]
    return ""


def _extract_tables(elements, doc_id, source, doc_type, year, progress=None) -> list[dict]:
    """Every table payload a parser attached, turned into storable fact records."""
    out = []
    for el in elements:
        payload = (el.meta or {}).get("table")
        if not payload:
            continue
        try:
            rec = fx.extract_table(
                payload, doc_id=doc_id, source=source,
                doc_type=doc_type, year=year, table_id=el.table_id,
            )
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"  table extraction failed ({el.table_id}): {e}")
            continue
        if rec:
            out.append(rec)
    return out


def _backfill_facts(conn, path, doc_id, progress=None) -> tuple[int, int]:
    """
    Extract facts for a document that is already indexed, without re-embedding.

    `facts_at` is stamped even when the document yields no tables at all. Without
    that marker a prose-only PDF would be re-parsed on every single ingest,
    forever, looking for tables it does not have.
    """
    with _LOCK:
        row = conn.execute(
            "SELECT doc_type, year FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
    doc_type = row["doc_type"] if row else ""
    year = row["year"] if row else ""

    try:
        _, elements = parse_file(path)
    except Exception as e:  # noqa: BLE001
        if progress:
            progress(f"  backfill parse failed for {path.name}: {e}")
        return (0, 0)

    tables = _extract_tables(elements, doc_id, path.name, doc_type, year, progress)
    with _LOCK:
        fx.clear_doc(conn, doc_id)
        for rec in tables:
            fx.store_table(conn, rec)
        conn.execute("UPDATE documents SET facts_at=? WHERE doc_id=?", (time.time(), doc_id))
        conn.commit()
    return (len(tables), sum(r["n_rows"] for r in tables))


def build_index(
    corpus_dir: str | None = None,
    index_path: str | None = None,
    embed_model: str | None = None,
    batch_size: int = 32,
    progress=None,
    rebuild: bool = False,
) -> dict:
    """
    Ingest every supported file under `corpus_dir` into the index.

    Incremental by content hash: a document whose bytes are unchanged is skipped
    entirely, so re-running after adding one file costs one file's work.

    Embeddings are computed in BATCHES. One HTTP round-trip per chunk measured
    ~273 ms on this machine, which makes an institution-scale corpus infeasible;
    batching is a correctness-of-scale requirement, not an optimisation.
    """
    conn = connect(index_path)
    model = embed_model or DEFAULT_EMBED_MODEL
    emb = get_embeddings(model)

    stats = {
        "files_seen": 0, "files_indexed": 0, "files_skipped": 0,
        "chunks": 0, "notices": 0, "errors": 0, "embed_seconds": 0.0,
        "fact_tables": 0, "fact_rows": 0,
        "model": model,
    }

    # Dimension is fixed by the first vector written. Mixing models silently
    # produces garbage similarities, so the model name is recorded and enforced.
    stored_model = _meta_get(conn, "embed_model")
    if rebuild or (stored_model and stored_model != model):
        with _LOCK:
            conn.executescript(
                "DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM documents;"
                "DELETE FROM fact_values; DELETE FROM facts; DELETE FROM fact_tables;"
            )
            conn.commit()
    with _LOCK:
        _meta_set(conn, "embed_model", model)
        conn.commit()

    files = iter_corpus(corpus_dir or DEFAULT_CORPUS_DIR)
    stats["files_seen"] = len(files)
    present: set[str] = set()

    for path in files:
        doc_id = content_hash(path)
        present.add(doc_id)
        with _LOCK:
            seen = conn.execute(
                "SELECT doc_id, facts_at FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
        if seen and not rebuild:
            stats["files_skipped"] += 1
            # An index built before aggregation existed holds chunks but no
            # facts. Backfilling costs one parse and NO embedding, which is the
            # whole reason it is worth doing here rather than telling the
            # operator to rebuild: re-embedding a real corpus takes hours.
            if seen["facts_at"] is None:
                n = _backfill_facts(conn, path, doc_id, progress)
                stats["fact_tables"] += n[0]
                stats["fact_rows"] += n[1]
                if progress and n[0]:
                    progress(f"backfilled {n[0]} table(s) from {path.name}")
            elif progress:
                progress(f"skip (unchanged): {path.name}")
            continue

        if progress:
            progress(f"parsing: {path.name}")
        _, elements = parse_file(path)

        sample = next((e.text for e in elements if e.kind in ("text", "table_summary")), "")
        doc_type = _infer_doc_type(path.name, sample)
        year = _infer_year(path.name, sample)

        notices = [e for e in elements if e.kind == "notice"]
        indexable = [e for e in elements if e.kind != "notice" and e.text.strip()]
        stats["notices"] += len(notices)
        if any(n.meta.get("reason") == "parse_error" for n in notices):
            stats["errors"] += 1

        # Notices are stored too: "that file failed to parse" must be a
        # retrievable, visible fact, never silence.
        to_store: list[ParsedElement] = indexable + notices

        texts = [e.text for e in to_store]
        vectors: list[list[float] | None] = [None] * len(texts)
        if indexable:
            t0 = time.time()
            done = 0
            for i in range(0, len(indexable), batch_size):
                batch = indexable[i : i + batch_size]
                try:
                    vs = emb.embed_documents([b.text for b in batch])
                except Exception as e:  # noqa: BLE001
                    if progress:
                        progress(f"  embedding failed on batch {i}: {e}")
                    vs = [None] * len(batch)
                for j, v in enumerate(vs):
                    vectors[i + j] = v
                done += len(batch)
                if progress:
                    progress(f"  embedded {done}/{len(indexable)} chunks of {path.name}")
            stats["embed_seconds"] += time.time() - t0

        # Every table this document contains, in full. Extraction happens before
        # the write so a malformed table cannot leave a half-written document
        # behind; it is deliberately not allowed to fail the ingest, because a
        # table that resists classification should cost aggregation, not search.
        tables = _extract_tables(elements, doc_id, path.name, doc_type, year, progress)

        with _LOCK:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            # Facts are keyed by doc_id and MUST go before the insert. doc_id is
            # a content hash, so this only ever fires on an explicit rebuild of
            # the same bytes — but without it that rebuild silently doubles every
            # SUM in the archive, and a doubled total looks exactly like a real one.
            fx.clear_doc(conn, doc_id)
            conn.execute(
                "INSERT OR REPLACE INTO documents"
                "(doc_id,source,rel_path,doc_type,year,n_elements,parse_status,parse_note,"
                "ingested_at,facts_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id, path.name, str(path), doc_type, year, len(to_store),
                    "error" if stats and any(
                        n.meta.get("reason") == "parse_error" for n in notices
                    ) else ("partial" if notices else "ok"),
                    json.dumps([n.meta.get("reason") for n in notices]) if notices else None,
                    time.time(),
                    time.time(),
                ),
            )
            # No explicit chunks_fts insert here: the chunks_fts_ai trigger does
            # it. Doing both would index every passage twice and corrupt bm25().
            for e, vec in zip(to_store, vectors, strict=True):
                conn.execute(
                    "INSERT INTO chunks"
                    "(doc_id,element_id,source,kind,text,locator,page,sheet,row,table_id,"
                    " lang_hint,doc_type,year,embedding)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        e.doc_id, e.element_id, e.source, e.kind, e.text, e.locator,
                        e.page, e.sheet, e.row, e.table_id, e.lang_hint,
                        doc_type, year,
                        _serialize(vec) if vec else None,
                    ),
                )
            for rec in tables:
                fx.store_table(conn, rec)
            conn.commit()

        stats["files_indexed"] += 1
        stats["chunks"] += len(to_store)
        stats["fact_tables"] += len(tables)
        stats["fact_rows"] += sum(r["n_rows"] for r in tables)

    # Reconcile the index against the folder.
    #
    # doc_id is a content hash, which makes re-ingest cheap but means EDITING a
    # file produces a new doc_id while the previous version's rows stay behind —
    # both versions then remain retrievable and citable, and an analyst can be
    # served last quarter's figures from a document that no longer exists.
    # Deleting a file from the folder likewise removed nothing at all. Anything
    # whose content hash is no longer present in the corpus is therefore dropped.
    # The chunks_fts_ad trigger removes the matching search terms.
    with _LOCK:
        known = {r["doc_id"] for r in conn.execute("SELECT doc_id FROM documents").fetchall()}
        stale = known - present
        for doc_id in stale:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            fx.clear_doc(conn, doc_id)
        if stale:
            conn.commit()
    stats["files_removed"] = len(stale)
    if stale and progress:
        progress(f"pruned {len(stale)} document(s) no longer in the corpus folder")

    return stats


# ── search ───────────────────────────────────────────────────────────────────

_FTS_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

# How much a lexical hit counts when the query contains nothing literal to match.
# 0.0 would disable BM25 entirely on prose queries; a small non-zero weight keeps
# it as a tie-breaker and as insurance against an embedding failure, without
# letting it outvote the dense ranker. Tune only with the retrieval eval.
_LEXICAL_SOFT_WEIGHT = float(os.getenv("ATHENA_LEXICAL_SOFT_WEIGHT", "0.25"))

# A "literal anchor" is something BM25 can match exactly and an embedding tends
# to blur: a document/reference code, an account number, an invoice number, an
# IBAN, or a figure. Their presence is what makes the lexical channel reliable.
# A bare four-digit YEAR must not qualify. Every query in a financial archive
# carries one, and hundreds of documents share it, so it identifies nothing —
# treating it as an anchor gave the lexical channel full weight on every query
# and made this whole mechanism a no-op (measured: results byte-identical to
# equal-weight RRF). Anchors are things that pick out a handful of documents.
_ANCHOR = re.compile(
    r"\d{1,3}(?:[.,]\d{3})+"        # grouped figure: 128.400  1.234.567
    r"|\d{5,}"                      # long bare run: 4821000 (a year is only 4)
    r"|\b[A-Z]{2,}[-_/]?\d+\b"      # BEL-2024-3015, DE44, INV/778
    r"|\b\d{4}[-_/]\w+[-_/]\w+\b"   # 2025-Q3-00123 (year PLUS more structure)
    r"|\b\w*\d+[-_/]\d{3,}\b",      # 3015-4711 style pairs
    re.UNICODE,
)


def _has_literal_anchor(query: str) -> bool:
    """True when the query carries an exact token BM25 can be trusted on."""
    return bool(_ANCHOR.search(query or ""))


def _fts_query(raw: str) -> str:
    """
    Turn free user text into a valid FTS5 MATCH expression.

    FTS5 MATCH takes a query LANGUAGE, not free text: bare '?', '-', ':' or an
    unbalanced quote raises OperationalError, and users type those constantly
    ("Wie hoch war der Umsatz in Q3 2025?"). Every alphanumeric token is
    therefore extracted and quoted as a literal, then OR-ed. Quoting also stops
    a token like "AND" or "NEAR" being read as an operator.
    """
    tokens = [t for t in _FTS_TOKEN.findall(raw or "") if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens[:32])


def _filters_sql(doc_type: str, year: str) -> tuple[str, list]:
    clauses, params = [], []
    if doc_type:
        clauses.append("doc_type = ?")
        params.append(doc_type)
    if year:
        clauses.append("year = ?")
        params.append(year)
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def search(
    query: str,
    k: int = 6,
    doc_type: str = "",
    year: str = "",
    index_path: str | None = None,
    rrf_k: int = 60,
    depth: int = 50,
) -> list[dict]:
    """Hybrid retrieval over the persistent knowledge base. See search_conn."""
    return search_conn(
        connect(index_path), _LOCK, query,
        k=k, doc_type=doc_type, year=year, rrf_k=rrf_k, depth=depth,
    )


def search_conn(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    query: str,
    k: int = 6,
    doc_type: str = "",
    year: str = "",
    rrf_k: int = 60,
    depth: int = 50,
    channel: str = "hybrid",
) -> list[dict]:
    """
    Hybrid retrieval: dense cosine + FTS5 bm25, fused with Reciprocal Rank Fusion.

    `channel` selects which rankers run: "hybrid" (both, the product behaviour),
    "dense" or "lexical". The single-channel modes exist for evaluation only —
    hybrid is a documented design decision and should be re-justified with
    measured recall per query type rather than inherited on faith. They are also
    the fastest way to diagnose a recall regression: if dense collapses and
    lexical holds, the embedding model or its index is at fault, not the fusion.

    RRF score = sum over rankers of 1 / (rrf_k + rank). It needs no score
    normalisation between two incomparable scales (cosine distance and bm25),
    which is exactly why it is the standard choice for this fusion.

    Filters are applied INSIDE both rankers, not after fusion — post-filtering a
    fixed candidate pool means a narrow filter can match none of it and quietly
    return nothing (or, worse, unfiltered results).

    The connection is a parameter so the persistent index and a per-chat
    in-memory session store (core/sessions.py) share ONE ranking implementation.
    If the two scopes ranked differently, accuracy measured on the archive would
    say nothing about uploaded documents, and every future ranking change would
    have to be made twice.
    """
    where, fparams = _filters_sql(doc_type, year)

    dense: list[int] = []
    try:
        if channel == "lexical":
            raise RuntimeError("dense channel disabled")
        qvec = _serialize(get_embeddings(_meta_get(conn, "embed_model")).embed_query(query))
        with lock:
            rows = conn.execute(
                f"SELECT id FROM chunks "
                f"WHERE embedding IS NOT NULL{where} "
                f"ORDER BY vec_distance_cosine(embedding, ?) ASC LIMIT ?",
                (*fparams, qvec, depth),
            ).fetchall()
        dense = [r["id"] for r in rows]
    except Exception:
        dense = []

    lexical: list[int] = []
    match = "" if channel == "dense" else _fts_query(query)
    if match:
        try:
            with lock:
                rows = conn.execute(
                    f"SELECT c.id AS id FROM chunks_fts f "
                    f"JOIN chunks c ON c.id = f.rowid "
                    f"WHERE chunks_fts MATCH ?{where} "
                    f"ORDER BY bm25(chunks_fts) ASC LIMIT ?",
                    (match, *fparams, depth),
                ).fetchall()
            lexical = [r["id"] for r in rows]
        except Exception:
            lexical = []

    # Weighted RRF, and the weights are not arbitrary — they come from measurement.
    #
    # Equal-weight RRF made hybrid WORSE than dense alone (MRR 0.923 vs 0.937,
    # hit@1 0.864 vs 0.883 over 360 queries on an 18,591-chunk archive). The
    # cause: BM25 scores hit@1 = 0.138 on paraphrased and synonym queries, barely
    # above noise, because an OR of common tokens matches half the archive in a
    # repetitive financial corpus. Giving a near-random ranker an equal vote
    # drags a good one down.
    #
    # But lexical is not useless — it is 1.000 on exact identifiers and amount
    # lookups, where dense dips to 0.975. The channels are complementary only
    # when the query carries something literal to match. So its weight is full
    # when the query contains a literal anchor (a reference code, a figure) and
    # discounted otherwise. Dense keeps full weight throughout.
    lex_weight = 1.0 if _has_literal_anchor(query) else _LEXICAL_SOFT_WEIGHT

    scores: dict[int, float] = {}
    for ranked, weight in ((dense, 1.0), (lexical, lex_weight)):
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + rank)

    if not scores:
        return []

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    ids = [cid for cid, _ in top]
    placeholders = ",".join("?" * len(ids))
    with lock:
        rows = conn.execute(
            f"SELECT id,doc_id,source,kind,text,locator,page,sheet,row,doc_type,year,lang_hint "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

    by_id = {r["id"]: dict(r) for r in rows}
    out = []
    for cid, score in top:
        rec = by_id.get(cid)
        if not rec:
            continue
        rec["score"] = round(score, 6)
        rec["in_dense"] = cid in dense
        rec["in_lexical"] = cid in lexical
        # Every hit is labelled with the tier it came from. A reader must never
        # have to guess whether source [3] is the institution's archive or a file
        # attached to this one chat — those are different claims about provenance.
        rec.setdefault("scope", "knowledge_base")
        out.append(rec)
    return out


def search_channel(
    query: str,
    k: int = 6,
    index_path: str | None = None,
    channel: str = "hybrid",
    depth: int = 50,
) -> list[dict]:
    """Single-channel search, for evaluation. See search_conn's `channel` note."""
    return search_conn(
        connect(index_path), _LOCK, query, k=k, depth=depth, channel=channel,
    )


def list_documents(doc_type: str = "", year: str = "", index_path: str | None = None) -> list[dict]:
    """Inventory of the persistent knowledge base. See list_documents_conn."""
    return list_documents_conn(connect(index_path), _LOCK, doc_type=doc_type, year=year)


def list_documents_conn(
    conn: sqlite3.Connection,
    lock: threading.Lock,
    doc_type: str = "",
    year: str = "",
) -> list[dict]:
    """
    Corpus inventory.

    Semantic search cannot answer "how many invoices are there" or "which years
    do we hold" — aggregate questions over a corpus are a documented RAG failure
    mode. Exposing the manifest directly is the cheap, exact fix, and it also
    lets the agent orient before searching.
    """
    where, params = _filters_sql(doc_type, year)
    sql = (
        "SELECT d.doc_id,d.source,d.doc_type,d.year,d.n_elements,d.parse_status,"
        "(SELECT COUNT(*) FROM chunks c WHERE c.doc_id=d.doc_id) AS n_chunks "
        "FROM documents d WHERE 1=1" + where.replace("doc_type", "d.doc_type").replace("year", "d.year")
    )
    with lock:
        rows = conn.execute(sql + " ORDER BY d.doc_type, d.source", params).fetchall()
    return [dict(r) for r in rows]


# ── exact aggregation ────────────────────────────────────────────────────────
#
# Counting does not go through search. See core/facts.py for why top-k is the
# wrong instrument for "how many" and what it takes to be exactly right instead.

def aggregate(index_path: str | None = None, **kw) -> dict:
    """Exact aggregate over every matching table row. See facts.aggregate."""
    return fx.aggregate(connect(index_path), _LOCK, **kw)


def tables_overview(index_path: str | None = None) -> list[dict]:
    """Every aggregatable table with its columns and their roles."""
    return fx.available_columns(connect(index_path), _LOCK)


def index_stats(index_path: str | None = None) -> dict:
    """Summary used by the UI, /health, and the tool docstrings."""
    conn = connect(index_path)
    with _LOCK:
        docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        embedded = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()["n"]
        types = [
            dict(r) for r in conn.execute(
                "SELECT doc_type, COUNT(*) AS n FROM documents GROUP BY doc_type ORDER BY n DESC"
            ).fetchall()
        ]
        years = [
            r["year"] for r in conn.execute(
                "SELECT DISTINCT year FROM documents WHERE year<>'' ORDER BY year"
            ).fetchall()
        ]
        model = _meta_get(conn, "embed_model", "")
    counts = fx.fact_stats(conn, _LOCK)
    return {
        "documents": docs,
        "chunks": chunks,
        "embedded_chunks": embedded,
        "fact_tables": counts["tables"],
        "fact_rows": counts["rows"],
        "tested_chunk_limit": TESTED_CHUNK_LIMIT,
        "within_tested_envelope": chunks <= TESTED_CHUNK_LIMIT,
        "doc_types": types,
        "years": years,
        "embed_model": model,
        "index_path": _CONN_PATH,
    }


def index_exists(index_path: str | None = None) -> bool:
    p = Path(str(index_path or DEFAULT_INDEX_PATH))
    if not p.exists():
        return False
    try:
        return index_stats(index_path)["chunks"] > 0
    except Exception:
        return False
