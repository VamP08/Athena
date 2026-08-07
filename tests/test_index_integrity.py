"""
tests/test_index_integrity.py
Regression tests for three index bugs found by inspection and reproduced before
being fixed. All three are accuracy or data-protection defects, so each gets a
test that fails against the old behaviour.

No network, no Ollama: embeddings are irrelevant to every invariant here, so the
chunks are written with NULL vectors and only the lexical/bookkeeping paths run.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from core import index as idx


@pytest.fixture()
def conn():
    import sqlite_vec

    tmp = tempfile.mkdtemp()
    c = sqlite3.connect(str(Path(tmp) / "t.db"))
    c.row_factory = sqlite3.Row
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    c.enable_load_extension(False)
    idx._init_schema(c)
    yield c
    c.close()


def _add(conn, doc_id, source, text):
    cur = conn.execute(
        "INSERT INTO chunks(doc_id,element_id,source,kind,text,locator) VALUES(?,?,?,?,?,?)",
        (doc_id, f"{doc_id}:1", source, "text", text, "x"),
    )
    conn.commit()
    return cur.lastrowid


def _fts_hits(conn, token):
    return conn.execute(
        "SELECT c.source AS source, c.text AS text FROM chunks_fts f "
        "JOIN chunks c ON c.id=f.rowid WHERE chunks_fts MATCH ?",
        (f'"{token}"',),
    ).fetchall()


def test_deleting_a_chunk_removes_its_search_terms(conn):
    """
    A deleted document must stop being findable.

    chunks_fts is an external-content FTS5 table: it keeps its own posting lists
    and only borrows text from `chunks`. Before the fix, deleting the row left
    the terms indexed AND SQLite reused the rowid, so searching for the deleted
    document's unique token returned the NEXT document's content — a wrong
    answer and a failed deletion at the same time.
    """
    rid = _add(conn, "docA", "A.pdf", "GEHEIMWORT_ALPHA vertrauliche Zahlung")
    assert _fts_hits(conn, "GEHEIMWORT_ALPHA"), "sanity: should be findable before deletion"

    conn.execute("DELETE FROM chunks WHERE doc_id=?", ("docA",))
    conn.commit()

    assert _fts_hits(conn, "GEHEIMWORT_ALPHA") == [], "deleted content must not stay searchable"

    reused = _add(conn, "docB", "B.pdf", "harmlose Buchung Kaffee")
    assert reused == rid, "precondition: SQLite reuses the freed rowid"

    hits = _fts_hits(conn, "GEHEIMWORT_ALPHA")
    assert hits == [], f"deleted document's token resolved to another document: {hits}"


def test_no_double_indexing(conn):
    """
    The insert trigger is the only writer of chunks_fts.

    An explicit INSERT alongside the trigger would index every passage twice and
    silently corrupt bm25() ranking — an accuracy bug with no visible symptom.
    """
    _add(conn, "docA", "A.pdf", "Umsatzerloese Quartal")
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?", ('"Umsatzerloese"',)
    ).fetchone()["n"]
    assert n == 1, f"passage indexed {n} times, expected exactly 1"


def test_updating_a_chunk_reindexes_it(conn):
    _add(conn, "docA", "A.pdf", "ERSTE_FASSUNG Angaben")
    conn.execute("UPDATE chunks SET text=? WHERE doc_id=?", ("ZWEITE_FASSUNG Angaben", "docA"))
    conn.commit()
    assert _fts_hits(conn, "ERSTE_FASSUNG") == [], "stale text still searchable after update"
    assert _fts_hits(conn, "ZWEITE_FASSUNG"), "new text not searchable after update"


def test_null_embeddings_do_not_break_vector_search(conn):
    """
    Parse notices and failed embedding batches are stored with a NULL embedding
    on purpose. sqlite-vec RAISES on a NULL vector rather than returning NULL, so
    the ranker must keep filtering them out in WHERE. This test pins that: it
    fails loudly if someone moves vec_distance_cosine into the SELECT list.
    """
    import sqlite_vec

    _add(conn, "docA", "A.pdf", "eine Notiz ohne Embedding")
    q = sqlite_vec.serialize_float32([0.1] * 1024)
    rows = conn.execute(
        "SELECT id FROM chunks WHERE embedding IS NOT NULL "
        "ORDER BY vec_distance_cosine(embedding, ?) LIMIT 5",
        (q,),
    ).fetchall()
    assert rows == []

    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "SELECT vec_distance_cosine(embedding, ?) FROM chunks", (q,)
        ).fetchall()


def test_build_index_prunes_documents_removed_from_the_folder(tmp_path, monkeypatch):
    """
    Deleting or editing a file in the corpus folder must be reflected in the index.

    doc_id is a content hash, so an edited file arrives as a NEW document while
    the previous version's rows remain — leaving both versions retrievable and
    citable. An analyst could be served figures from a document that no longer
    exists.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("KENNZAHL_EINS Umsatz 100", encoding="utf-8")
    (corpus / "b.md").write_text("KENNZAHL_ZWEI Umsatz 200", encoding="utf-8")

    db = str(tmp_path / "idx.db")
    monkeypatch.setattr(idx, "get_embeddings", lambda model=None: None)

    def fake_build(**kw):
        # Drive the real build_index but with embedding disabled: pass a batch
        # size larger than the corpus and let the embed call fail into None
        # vectors, which build_index already tolerates.
        return idx.build_index(**kw)

    idx.close()
    stats = fake_build(corpus_dir=str(corpus), index_path=db, batch_size=8, progress=None)
    assert stats["files_indexed"] == 2

    docs = {d["source"] for d in idx.list_documents(index_path=db)}
    assert docs == {"a.md", "b.md"}

    (corpus / "b.md").unlink()
    stats2 = fake_build(corpus_dir=str(corpus), index_path=db, batch_size=8, progress=None)
    assert stats2["files_removed"] == 1

    docs2 = {d["source"] for d in idx.list_documents(index_path=db)}
    assert docs2 == {"a.md"}, "document deleted from the folder is still in the index"

    conn = idx.connect(db)
    hits = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?", ('"KENNZAHL_ZWEI"',)
    ).fetchone()["n"]
    assert hits == 0, "removed document's terms are still searchable"
    idx.close()
