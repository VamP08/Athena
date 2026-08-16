"""
tests/test_aggregation.py
Exact aggregation over the archive's tables.

Every test here pins a way of being SILENTLY wrong rather than visibly broken.
That is the whole risk profile of this feature: a count is a plain number with
no error bars and no citation to check it against, so a total that is quietly
double-counted, quietly truncated, or quietly meaningless reads exactly like a
correct one. Each test below names the wrong answer it prevents.

Fully offline. Aggregation never touches the embedder, so these tests build real
tables through the real parsers and never call Ollama.
"""

from __future__ import annotations

import pytest

from core import documents as docs
from core import facts as fx
from core import index as idx

# ── fixtures ─────────────────────────────────────────────────────────────────

class _NoEmbeddings:
    """build_index requires an embedder; aggregation does not use the vectors."""

    def embed_documents(self, texts):
        return [None] * len(texts)


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """A temp corpus folder plus a freshly built index over it."""
    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: _NoEmbeddings())

    folder = tmp_path / "corpus"
    folder.mkdir()

    # An invoice register: the motivating case. 6 invoices, 2 of them open.
    (folder / "Rechnungsausgang_Q3_2025.csv").write_text(
        "Rechnungsnummer;Kunde;Datum;Nettobetrag (EUR);Status\n"
        "2025-1001;Elbe GmbH;2025-07-04;1.000,00;bezahlt\n"
        "2025-1002;Elbe GmbH;2025-07-19;2.500,50;bezahlt\n"
        "2025-1003;Baltic OY;2025-08-02;3.000,00;offen\n"
        "2025-1004;Baltic OY;2025-09-30;4.000,00;bezahlt\n"
        "2025-1005;Kranich AG;2025-10-01;5.000,00;offen\n"
        "2025-1006;Kranich AG;2025-06-30;6.000,00;bezahlt\n",
        encoding="utf-8",
    )

    # A bank statement: a movement column that may be summed, and a running
    # balance column that may not.
    (folder / "Kontoauszug_2025.csv").write_text(
        "Datum;Verwendungszweck;Betrag (EUR);Saldo (EUR)\n"
        "2025-07-01;Zahlungseingang;100,00;1.100,00\n"
        "2025-07-02;Miete;-400,00;700,00\n"
        "2025-07-03;Zahlungseingang;250,00;950,00\n"
        "2025-07-04;Umsatzsteuer-Vorauszahlung;-50,00;900,00\n",
        encoding="utf-8",
    )

    # A balance sheet: line items followed by their own total, in one column.
    (folder / "Bilanz_2024.csv").write_text(
        "Position;2024 (EUR)\n"
        "Anlagevermögen;1.000\n"
        "Umlaufvermögen;2.000\n"
        "Summe Aktiva;3.000\n",
        encoding="utf-8",
    )

    idx.close()
    path = str(tmp_path / "idx.db")
    idx.build_index(corpus_dir=str(folder), index_path=path)
    yield folder, path
    idx.close()


def agg(path, **kw):
    return idx.aggregate(index_path=path, **kw)


# ── the core claim: exactness ────────────────────────────────────────────────

def test_count_covers_every_row_not_a_top_k_sample(corpus):
    """
    The failure this feature exists to remove.

    Retrieval returns k rows and says nothing about how many exist. A count
    derived from it is not vague, it is specifically wrong — and the error grows
    with the corpus, so it looks MORE trustworthy exactly as it gets worse.
    """
    _, path = corpus
    r = agg(path, operation="count", source="Rechnungsausgang")
    assert r["ok"] and r["value"] == 6
    assert r["matched_rows"] == r["scanned_rows"] == 6


def test_sum_matches_the_arithmetic_exactly(corpus):
    _, path = corpus
    r = agg(path, operation="sum", column="Nettobetrag", source="Rechnungsausgang")
    assert r["value"] == pytest.approx(21500.50)


def test_where_and_period_filters_compose(corpus):
    """
    A regression on parameter binding, which failed silently.

    SQLite binds `?` by position in the finished statement, where every join
    placeholder precedes every where placeholder — but a filter is built as
    (join, condition) pairs. Interleaving them bound a column name where a date
    belonged, which does not raise: it matched nothing and returned 0, which is
    indistinguishable from a legitimate empty result.
    """
    _, path = corpus
    r = agg(path, operation="count", source="Rechnungsausgang",
            where_column="Status", where_value="bezahlt", period="Q3 2025")
    # 1001, 1002, 1004 are paid and inside Q3; 1006 is paid but in Q2.
    assert r["value"] == 3, r

    r2 = agg(path, operation="count", source="Rechnungsausgang", period="Q3 2025")
    assert r2["value"] == 4      # 1005 is October, 1006 is June
    assert "2025-07-01..2025-09-30" in " ".join(r2["filters"])


def test_an_inapplicable_filter_is_an_error_not_a_count_of_zero(corpus):
    """
    "0 rows in Q3" and "this table has no date column" are different answers.

    Returning the first for the second is the failure this whole module exists
    to prevent: an empty result reads as a fact about the archive, and nothing
    downstream can tell it apart from a real zero.
    """
    _, path = corpus
    r = agg(path, operation="count", source="Bilanz", period="Q3 2025")
    assert r["ok"] is False
    assert "NOT a count of zero" in r["error"]
    assert any("no date column" in w for w in r["warnings"])


def test_an_unreadable_period_is_reported_not_ignored(corpus):
    """Silently dropping a filter answers a different question than the one asked."""
    _, path = corpus
    r = agg(path, operation="count", source="Rechnungsausgang", period="last quarter")
    assert r["ok"] is False


def test_breakdown_groups_and_totals_agree_with_the_whole(corpus):
    _, path = corpus
    r = agg(path, operation="breakdown", column="Status", source="Rechnungsausgang")
    by = {g["key"]: g for g in r["groups"]}
    assert by["bezahlt"]["count"] == 4 and by["offen"]["count"] == 2
    assert by["bezahlt"]["sum"] + by["offen"]["sum"] == pytest.approx(21500.50)


# ── columns that must never be added up ──────────────────────────────────────

def test_running_balance_is_refused_and_the_closing_figure_offered(corpus):
    """
    Summing `Saldo` produces a number with no meaning that still looks like money.

    Each row of a running balance already contains every row before it, so the
    sum is roughly N times the truth. Nothing downstream can catch it: it has the
    right units, the right magnitude and a real column name behind it.
    """
    _, path = corpus
    r = agg(path, operation="sum", column="Saldo", source="Kontoauszug")
    assert r["ok"] is False and r["refused"] is True
    assert r["role"] == "balance"
    # The refusal must answer the question the user meant, not just decline.
    assert r["closing"]["value"] == pytest.approx(900.00)
    assert r["closing"]["date"] == "2025-07-04"
    assert "Betrag (EUR)" in r["alternatives"]


def test_closing_is_the_last_value_not_the_largest(corpus):
    """
    `max` is the wrong answer and an unusually convincing one.

    An account that peaks mid-period and then pays out has a maximum drawn from
    a real row with a real date and a real citation — and it is not the balance.
    The model needs a way to ASK for the closing figure, or it reads the refusal,
    agrees that summing is wrong, and estimates from search results instead.
    """
    _, path = corpus
    r = agg(path, operation="closing", column="Saldo", source="Kontoauszug")
    assert r["ok"] and r["value"] == pytest.approx(900.00)
    assert r["closing"][0]["date"] == "2025-07-04"

    peak = agg(path, operation="max", column="Saldo", source="Kontoauszug")
    assert peak["value"] == pytest.approx(1100.00)
    assert peak["value"] != r["value"], "the test corpus must distinguish the two"


def test_closing_respects_a_period_filter(corpus):
    """"What did the account stand at, at the end of the quarter" is the question."""
    _, path = corpus
    r = agg(path, operation="closing", column="Saldo",
            source="Kontoauszug", period="2025-07")
    assert r["value"] == pytest.approx(900.00)
    assert "2025-07-01..2025-07-31" in " ".join(r["filters"])


def test_the_movement_column_beside_it_still_sums(corpus):
    _, path = corpus
    r = agg(path, operation="sum", column="Betrag", source="Kontoauszug")
    assert r["value"] == pytest.approx(-100.00)


def test_identifier_columns_are_refused(corpus):
    """Account codes and invoice numbers are numerals, not quantities."""
    _, path = corpus
    r = agg(path, operation="sum", column="Rechnungsnummer", source="Rechnungsausgang")
    assert r["ok"] is False and r["role"] == "identifier"


def test_umsatz_is_an_amount_not_a_rate():
    """
    German compounding put the single most important column at risk.

    `Steuersatz` and `Zinssatz` are rates. `Umsatz` is revenue — and ends in the
    same four letters. An unanchored `satz` pattern classified every Umsatz
    column in the archive as a percentage and then refused to add it up.
    """
    assert fx.classify_column("Umsatz (EUR)", ["4310000", "4655000"])["role"] == "amount"
    assert fx.classify_column("Umsatzerlöse", ["18452000"])["role"] == "amount"
    assert fx.classify_column("Steuersatz", ["19", "7"])["role"] == "rate"
    assert fx.classify_column("Zinssatz", ["1,5", "2,0"])["role"] == "rate"


def test_year_columns_are_not_summable():
    assert fx.classify_column("Jahr", ["2025", "2025", "2024"])["role"] == "period"


# ── rows that must not be counted twice ──────────────────────────────────────

def test_a_tables_own_total_row_is_excluded(corpus):
    """`Summe Aktiva` sits in the same column as the two figures it totals."""
    _, path = corpus
    r = agg(path, operation="sum", column="2024", source="Bilanz")
    assert r["value"] == pytest.approx(3000.0)   # not 6000
    assert r["totals_excluded"] == 1


def test_invoice_summary_block_is_excluded_when_there_is_no_date_column():
    """Line items plus a net/VAT/gross block would count every euro twice."""
    cols = [
        fx.classify_column("Zusammenfassung", ["Nettobetrag", "Umsatzsteuer 19%"]),
        fx.classify_column("Betrag (EUR)", ["128.400,00", "24.396,00"]),
    ]
    assert fx.classify_row_kind(["Nettobetrag", "128.400,00"], cols, has_date=False) == "total"
    assert fx.classify_row_kind(["Umsatzsteuer 19%", "24.396,00"], cols, has_date=False) == "total"


def test_a_ledgers_vat_booking_is_not_mistaken_for_a_total(corpus):
    """
    The false positive the date-column restriction exists to prevent.

    `Umsatzsteuer-Vorauszahlung` in a bank statement is a real payment that must
    be summed. Treating it as a total would delete real money from the answer —
    the same class of silent error, pointing the other way.
    """
    _, path = corpus
    r = agg(path, operation="count", source="Kontoauszug")
    assert r["value"] == 4 and r["totals_excluded"] == 0


def test_unlike_tables_are_flagged_rather_than_merged(corpus):
    """A row is not a record: line items and invoices are not addable."""
    _, path = corpus
    r = agg(path, operation="count")
    assert len({p["signature"] for p in r["per_table"]}) > 1
    assert any("DIFFERENT columns" in w for w in r["warnings"])


def test_summing_across_several_tables_says_so(corpus):
    _, path = corpus
    r = agg(path, operation="sum", column="Betrag")
    if len(r["per_table"]) > 1:
        assert any("COMBINES" in w for w in r["warnings"])


# ── the three documented blockers ────────────────────────────────────────────

def test_reingest_does_not_double_the_totals(corpus, monkeypatch):
    """
    doc_id is a content hash, so a rebuild rewrites the same rows.

    Without a delete path the second ingest appends a second copy and every SUM
    in the archive doubles. A doubled total is not detectable by looking at it.
    """
    folder, path = corpus
    before = agg(path, operation="sum", column="Nettobetrag", source="Rechnungsausgang")

    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: _NoEmbeddings())
    idx.build_index(corpus_dir=str(folder), index_path=path, rebuild=True)

    after = agg(path, operation="sum", column="Nettobetrag", source="Rechnungsausgang")
    assert after["value"] == before["value"]
    assert after["matched_rows"] == before["matched_rows"] == 6


def test_aggregation_sees_rows_the_retrieval_cap_hides(tmp_path, monkeypatch):
    """
    The retrieval cap must not become the aggregation cap.

    Chunks stop at MAX_ROWS_EMBEDDED so a large ledger cannot swamp the index —
    harmless for top-k, fatal for counting. A count derived from chunks reports
    exactly the cap, and "2000" for a 34,000-row register looks like a real
    figure rather than a truncation.
    """
    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: _NoEmbeddings())
    monkeypatch.setattr(docs, "MAX_ROWS_EMBEDDED", 5)

    folder = tmp_path / "big"
    folder.mkdir()
    rows = "\n".join(f"2025-07-01;Posten {i};{i}00,00" for i in range(1, 51))
    (folder / "Hauptbuch_2025.csv").write_text(
        "Datum;Bezeichnung;Betrag (EUR)\n" + rows + "\n", encoding="utf-8"
    )

    idx.close()
    path = str(tmp_path / "big.db")
    idx.build_index(corpus_dir=str(folder), index_path=path)
    try:
        conn = idx.connect(path)
        indexed = conn.execute(
            "SELECT COUNT(*) n FROM chunks WHERE kind='table_row'"
        ).fetchone()["n"]
        assert indexed == 5, "sanity: retrieval really is capped"

        r = agg(path, operation="count", source="Hauptbuch")
        assert r["value"] == 50, "aggregation must see every row, cap or no cap"

        total = agg(path, operation="sum", column="Betrag", source="Hauptbuch")
        assert total["value"] == pytest.approx(sum(i * 100 for i in range(1, 51)))
    finally:
        idx.close()


def test_deleting_a_document_removes_its_facts(corpus, monkeypatch):
    """A file removed from the corpus must stop contributing to totals."""
    folder, path = corpus
    (folder / "Bilanz_2024.csv").unlink()

    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: _NoEmbeddings())
    idx.build_index(corpus_dir=str(folder), index_path=path)

    r = agg(path, operation="sum", column="2024", source="Bilanz")
    assert r["ok"] is False, "the removed document must no longer be aggregatable"

    conn = idx.connect(path)
    left = conn.execute(
        "SELECT COUNT(*) n FROM fact_values WHERE col='2024eur'"
    ).fetchone()["n"]
    assert left == 0, "fact_values must not outlive the facts they belong to"


# ── value parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.245.300,50", 1245300.50),    # German grouping
        ("1,245,300.50", 1245300.50),    # English grouping
        ("-55022,59", -55022.59),
        ("(1.234)", -1234.0),            # accounting negative
        ("128.400,00 EUR", 128400.00),
        ("4400", 4400.0),
        ("12,5 %", 12.5),
        ("2025-1042", None),             # invoice number, NOT arithmetic
        ("2024-11-19", None),            # date, NOT arithmetic
        ("", None),
        ("Elbe Spedition GmbH", None),
    ],
)
def test_number_parsing(raw, expected):
    got = fx.parse_number(raw)
    if expected is None:
        assert got is None, f"{raw!r} must not parse as a quantity"
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw,expected",
    [("2025-07-26", "2025-07-26"), ("26.07.2025", "2025-07-26"),
     ("2025-07-26 00:00:00", "2025-07-26"), ("not a date", None)],
)
def test_date_parsing(raw, expected):
    assert fx.parse_date(raw) == expected


@pytest.mark.parametrize(
    "period,expected",
    [
        ("Q3 2025", ("2025-07-01", "2025-09-30")),
        ("2025-Q1", ("2025-01-01", "2025-03-31")),
        ("2024", ("2024-01-01", "2024-12-31")),
        ("2025-07", ("2025-07-01", "2025-07-31")),
        ("Q2 2024", ("2024-04-01", "2024-06-30")),
        ("nonsense", None),
    ],
)
def test_period_parsing(period, expected):
    assert fx._period_range(period) == expected


def test_running_balance_detected_without_a_header_hint():
    """
    A column called `Stand` carries no keyword and behaves exactly like `Saldo`.

    Header matching is the primary signal, but it only covers vocabulary someone
    thought of. The arithmetic is the same either way: each value equals the
    previous one plus that row's movement.
    """
    columns = [
        fx.classify_column("Betrag", ["100", "-400", "250", "-50"]),
        fx.classify_column("Stand", ["1100", "700", "950", "900"]),
    ]
    rows = [["100", "1100"], ["-400", "700"], ["250", "950"], ["-50", "900"]]
    assert "stand" in fx.detect_running_balance(columns, rows)


def test_a_property_sheet_pivots_to_one_record(tmp_path, monkeypatch):
    """
    `Feld;Wert` files hold ONE record written down the page.

    Counting their rows counts fields — across 266 such invoices that reports
    3,598, arithmetically perfect and thirteen times the truth. Pivoted, each
    sheet is one fact whose columns are its keys, so counting invoices counts
    invoices and their Nettobetrag becomes summable across the archive, which
    row-wise storage could never offer at all (the amounts share a value column
    with dates, names and currency codes).
    """
    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: _NoEmbeddings())
    folder = tmp_path / "kv"
    folder.mkdir()
    for i in (1, 2, 3):
        (folder / f"invoice_kunde_{i}.csv").write_text(
            "Feld;Wert\n"
            "Dokumenttyp;invoice\n"
            f"Kunde;Kunde {i}\n"
            "Geschaeftsjahr;2024\n"
            "Belegnummer;2024-Q1-0011\n"
            f"Nettobetrag (EUR);{i}00.000,00\n"
            "Waehrung;EUR\n",
            encoding="utf-8",
        )

    idx.close()
    path = str(tmp_path / "kv.db")
    idx.build_index(corpus_dir=str(folder), index_path=path)
    try:
        r = agg(path, operation="count", doc_type="invoice")
        assert r["value"] == 3, "one record per sheet, not one per field"
        assert not any("NOT RECORDS" in w for w in r["warnings"]), r["warnings"]

        s = agg(path, operation="sum", column="Nettobetrag", doc_type="invoice")
        assert s["value"] == pytest.approx(600000.0)
        assert s["matched_rows"] == 3
    finally:
        idx.close()


def test_a_stacked_csv_splits_into_its_real_tables(tmp_path, monkeypatch):
    """
    One CSV, two tables: a property block, a blank line, a line-item register.

    Parsed as one table, the line items are read against the property block's
    Feld/Wert headers — the register's own columns are unrecoverable and every
    aggregate over the file is wrong. Split, the property block pivots to one
    record and the register keeps its own summable columns.
    """
    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: _NoEmbeddings())
    folder = tmp_path / "stacked"
    folder.mkdir()
    (folder / "invoice_kranich_2024_Q1.csv").write_text(
        "Feld;Wert\n"
        "Dokumenttyp;invoice\n"
        "Kunde;Kranich AG\n"
        "Belegnummer;2024-Q1-0042\n"
        "Nettobetrag (EUR);300,00\n"
        "\n"
        "Position;Beschreibung;Betrag (EUR)\n"
        "1;Lagerflaeche;100,00\n"
        "2;Transport;200,00\n",
        encoding="utf-8",
    )

    idx.close()
    path = str(tmp_path / "stacked.db")
    idx.build_index(corpus_dir=str(folder), index_path=path)
    try:
        conn = idx.connect(path)
        layouts = sorted(
            r["layout"] for r in conn.execute("SELECT layout FROM fact_tables")
        )
        assert layouts == ["keyvalue_record", "records"], layouts

        head = agg(path, operation="sum", column="Nettobetrag")
        assert head["value"] == pytest.approx(300.0)

        items = agg(path, operation="sum", column="Betrag")
        assert items["value"] == pytest.approx(300.0)
        assert items["matched_rows"] == 2

        # Line-item locators must cite the file's REAL line numbers — the
        # register's first data row is line 8 of the file, not "Zeile 2".
        rows = conn.execute(
            "SELECT locator FROM facts WHERE row_kind='data' ORDER BY row"
        ).fetchall()
        assert any(r["locator"] == "Zeile 8" for r in rows), [r["locator"] for r in rows]
    finally:
        idx.close()


def test_legacy_unpivoted_keyvalue_still_warns(corpus):
    """
    An index whose facts predate the pivot must keep the rows-are-fields warning.

    The pivot only lands when facts are (re-)extracted; a pre-existing database
    still holds field-per-row tables marked layout='keyvalue', and silently
    treating those as record counts is the exact failure the warning existed for.
    """
    _, path = corpus
    conn = idx.connect(path)
    fx.store_table(conn, {
        "table_uid": "legacy:kv", "doc_id": "legacydoc", "source": "legacy_invoice.csv",
        "table_id": "legacy:kv", "sheet": None, "page": None,
        "doc_type": "payroll", "year": "2024",
        "n_rows": 6, "n_rows_indexed": 6, "n_rows_total": 6, "n_rows_seen": 6,
        "n_total_rows": 0, "layout": "keyvalue",
        "columns": [
            {"name": "feld", "raw": "Feld", "role": "label", "kind": "text"},
            {"name": "wert", "raw": "Wert", "role": "label", "kind": "text"},
        ],
        "facts": [{"row": i, "locator": f"Zeile {i}", "row_kind": "data", "label": "x"}
                  for i in range(2, 8)],
        "values": [],
    })
    conn.commit()

    r = agg(path, operation="count", doc_type="payroll")
    assert any("NOT RECORDS" in w for w in r["warnings"]), r["warnings"]


def test_a_two_column_register_is_not_mistaken_for_a_property_sheet(corpus):
    """`Position` / `2024 (EUR)` is a register: its value column is all numbers."""
    _, path = corpus
    r = agg(path, operation="sum", column="2024", source="Bilanz")
    assert all(p["layout"] == "records" for p in r["per_table"])
    assert not any("NOT RECORDS" in w for w in r["warnings"])


def test_the_per_table_listing_is_bounded(tmp_path, monkeypatch):
    """
    A 266-table scope must not bury the answer.

    The researcher's per-tool budget truncates from the END, so an unbounded
    provenance list pushes the CHECK warnings off the bottom — the answer
    survives while the caveats that make it safe to quote do not, which is worse
    than losing both.
    """
    from core.doc_tools import _MAX_TABLES_IN_ANSWER, _render

    monkeypatch.setattr(idx, "get_embeddings", lambda *a, **k: _NoEmbeddings())
    folder = tmp_path / "many"
    folder.mkdir()
    # The contents must differ: doc_id is a content hash, so byte-identical
    # files are one document by design, not forty.
    for i in range(40):
        (folder / f"ledger_{i:03d}.csv").write_text(
            f"Datum;Bezeichnung;Betrag (EUR)\n2025-01-01;Posten {i};100,00\n",
            encoding="utf-8",
        )

    idx.close()
    path = str(tmp_path / "many.db")
    idx.build_index(corpus_dir=str(folder), index_path=path)
    try:
        r = agg(path, operation="count", doc_type="ledger")
        assert r["value"] == 40
        text = _render(r)
        listed = [ln for ln in text.splitlines() if ln.startswith("- ledger_")]
        assert len(listed) == _MAX_TABLES_IN_ANSWER
        assert "further table(s) contributing 32 row(s)" in text
        assert len(text) < 4000, "the answer must stay inside the tool budget"
    finally:
        idx.close()


def test_two_columns_normalising_alike_do_not_shadow_each_other():
    """`Betrag` and `Betrag ` must remain distinguishable, not silently merge."""
    payload = {
        "headers": ["Betrag", "Betrag "],
        "rows": [{"row": 2, "locator": "Zeile 2", "cells": ["10", "20"]}],
        "n_rows_total": 1,
        "n_rows_tabulated": 1,
        "n_rows_indexed": 1,
    }
    rec = fx.extract_table(payload, doc_id="d" * 32, source="x.csv",
                           doc_type="", year="", table_id="t1")
    names = [c["name"] for c in rec["columns"]]
    assert len(set(names)) == 2, names
