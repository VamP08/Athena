"""
core/facts.py
Exact aggregation over the tabular parts of the archive.

The problem this exists to solve
───────────────────────────────
Retrieval answers "what does the archive say about X" by returning the k most
similar passages. That is structurally the wrong instrument for "how many
invoices did we issue in Q3" or "what is the total net value of the open ones".
Top-k returns 6 rows out of 340 and the model then adds up the 6 it can see, so
the answer is not approximately right — it is confidently, specifically wrong,
and it gets wronger as the corpus grows, which is the opposite of how a system
is supposed to behave. Semantic similarity has no notion of completeness, and no
amount of prompt engineering gives it one.

So counting does not go through the retriever at all. At ingest time every table
is also written to a small relational store, and aggregate questions are
answered by SQL over EVERY matching row.

Four things make that harder than `SELECT SUM(...)`, and each one is a way to be
silently wrong rather than visibly broken:

 1. Not every number may be added up. `Saldo (EUR)` in a bank statement is a
    RUNNING balance — each row already contains every row before it, so summing
    it produces a number with no meaning that still looks like money. `Konto`
    4401 is an account code. A VAT rate is a rate. Summing any of these is
    nonsense that no downstream check would catch, so columns carry a ROLE and
    the store refuses the operations that role does not support — and answers
    the question the user actually meant instead of just declining.

 2. Tables contain their own totals. `Summe Aktiva`, `Gesamt`, `Betriebsergebnis
    (EBIT)` sit in the same column as the operands they total. Summing the column
    double-counts every one of them. Such rows are classified and excluded, and
    the exclusion is always reported by name, because a silent exclusion is just
    a different silent error.

 3. The retrieval row cap is not the aggregation row cap. Chunks stop at
    MAX_ROWS_EMBEDDED (2000); facts do not. Deriving a count from indexed chunks
    would report exactly 2000 rows for a 34,000-row ledger.

 4. Re-ingesting must not double the totals. doc_id is a content hash, so an
    edited file arrives as a NEW document while the old rows sit there. Facts are
    therefore deleted per doc_id on every write path, and the schema enforces it
    with a trigger rather than trusting call sites to remember.

Why an EAV table rather than one table per source
─────────────────────────────────────────────────
The archive's tables have nothing in common: five columns here, three there,
German headers in one file and English in the next. Creating a real table per
source would mean DDL at ingest time, a migration story for every re-ingest, and
SQL built from user-supplied identifiers. One narrow (fact_id, column, value)
table keeps all of that in data, where filters compose and a malformed column
name is a lookup miss instead of a syntax error. The corpus is thousands of rows,
not millions, so the width costs nothing measurable.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata

# Operations the tool exposes. Kept small and closed on purpose: a 9B local
# model copies from a visible vocabulary reliably and invents plausible-looking
# values for open-ended fields.
OPERATIONS = (
    "count", "sum", "average", "min", "max", "distinct", "breakdown", "closing",
)

# How many rows a value/label listing shows before it is truncated.
_MAX_LISTED = 25


# ── schema ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS fact_tables (
    table_uid       TEXT PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    source          TEXT NOT NULL,
    table_id        TEXT,
    sheet           TEXT,
    page            INTEGER,
    doc_type        TEXT,
    year            TEXT,
    n_rows          INTEGER,   -- data rows stored here, whatever the retrieval cap did
    n_rows_indexed  INTEGER,   -- how many of them are also retrievable chunks
    n_rows_total    INTEGER,   -- rows the file actually has
    n_rows_seen     INTEGER,   -- rows that survived MAX_ROWS_TABULATED
    n_total_rows    INTEGER,   -- rows classified as totals and excluded
    layout          TEXT,      -- 'records' | 'keyvalue'
    columns         TEXT       -- JSON [{name, raw, role, kind, n_num, n_date}]
);

CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY,
    table_uid  TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    source     TEXT NOT NULL,
    row        INTEGER,
    locator    TEXT,
    doc_type   TEXT,
    year       TEXT,
    row_kind   TEXT NOT NULL,   -- 'data' | 'total'
    label      TEXT             -- the row's first text cell, for reporting
);

CREATE TABLE IF NOT EXISTS fact_values (
    fact_id    INTEGER NOT NULL,
    table_uid  TEXT NOT NULL,
    col        TEXT NOT NULL,   -- normalised column key
    text       TEXT,
    num        REAL,
    date       TEXT             -- ISO yyyy-mm-dd when the cell parsed as a date
);

CREATE INDEX IF NOT EXISTS idx_facts_doc    ON facts(doc_id);
CREATE INDEX IF NOT EXISTS idx_facts_table  ON facts(table_uid);
CREATE INDEX IF NOT EXISTS idx_fv_fact      ON fact_values(fact_id);
CREATE INDEX IF NOT EXISTS idx_fv_col       ON fact_values(table_uid, col);
CREATE INDEX IF NOT EXISTS idx_ft_doc       ON fact_tables(doc_id);

-- Re-ingest correctness, enforced in the schema rather than at call sites.
--
-- doc_id is a content hash: editing one number in a ledger produces a brand new
-- doc_id, and without this the previous version's rows stay in `facts` forever.
-- Every subsequent SUM would then quietly include both versions. build_index
-- deletes facts explicitly too — this trigger is the backstop for the prune
-- path and for any future writer that forgets, because "the total is now double"
-- is not a failure anyone notices by looking at it.
CREATE TRIGGER IF NOT EXISTS facts_cascade_ad AFTER DELETE ON facts BEGIN
    DELETE FROM fact_values WHERE fact_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS fact_tables_cascade_ad AFTER DELETE ON fact_tables BEGIN
    DELETE FROM facts WHERE table_uid = old.table_uid;
END;
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# ── value parsing ────────────────────────────────────────────────────────────

_CURRENCY = re.compile(r"(?i)\b(eur|usd|gbp|chf|tsd|mio|mrd)\b|[€$£¥]")
_NUMERIC_BODY = re.compile(r"^[+-]?[\d.,\s]+$")
_DE_GROUPED = re.compile(r"^[+-]?\d{1,3}(\.\d{3})+(,\d+)?$")
_EN_GROUPED = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")


def parse_number(raw: str) -> float | None:
    """
    A cell to a float, or None if it is not a quantity.

    German and English financial exports disagree about which of `.` and `,` is
    the decimal point, and the same archive contains both — `1.245.300,50` and
    `1,245,300.50` are the same amount written by two systems. Guessing wrong is
    a factor-of-1000 error in a financial answer, so the grouping patterns are
    matched explicitly and only the genuinely ambiguous single-separator case
    falls back to a rule (a separator trailing 1-2 digits is a decimal point;
    exactly 3 digits with a leading group is a thousands separator).

    Deliberately strict about what is NOT a number: `2025-1042` is an invoice
    number and `2024-11-19` is a date. Both would otherwise parse as arithmetic
    and both would be catastrophic to sum.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    s = _CURRENCY.sub("", s).replace("%", "").strip()

    negative = False
    if s.startswith("(") and s.endswith(")"):   # (1.234) accounting negative
        negative, s = True, s[1:-1].strip()

    s = s.replace(" ", "").replace(" ", "")
    if not s or not _NUMERIC_BODY.match(s):
        return None

    if "," in s and "." in s:
        # Whichever separator comes last is the decimal point.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if _EN_GROUPED.match(s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s:
        if _DE_GROUPED.match(s) or re.match(r"^[+-]?\d{1,3}(\.\d{3})+$", s):
            s = s.replace(".", "")
        # else: already a plain decimal

    if s.count(".") > 1:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if negative else v


_DATE_PATTERNS = (
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})"), (1, 2, 3)),
    (re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})"), (3, 2, 1)),
    (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})"), (3, 2, 1)),
    (re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})"), (1, 2, 3)),
)


def parse_date(raw: str) -> str | None:
    """A cell to an ISO date string, or None. openpyxl datetimes arrive as text."""
    if raw is None:
        return None
    s = str(raw).strip()
    if len(s) < 8:
        return None
    for pat, (yi, mi, di) in _DATE_PATTERNS:
        m = pat.match(s)
        if not m:
            continue
        y, mo, d = int(m.group(yi)), int(m.group(mi)), int(m.group(di))
        if not (1900 <= y <= 2200 and 1 <= mo <= 12 and 1 <= d <= 31):
            return None
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


# ── column semantics ─────────────────────────────────────────────────────────

def norm_col(name: str) -> str:
    """Column key: lowercase, letters and digits only, diacritics folded."""
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ß", "ss")
    return re.sub(r"[^0-9a-z]+", "", s.lower())


# Order matters. `Kontostand` must reach the balance test before the identifier
# test sees `konto` inside it.
_ROLE_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("balance", re.compile(
        r"saldo|kontostand|endbestand|anfangsbestand|bestand|balance|"
        r"runningtotal|laufend|kumuliert|cumulative|closing|opening"
    )),
    # `satz` is anchored, and excluded after `um`, because German compounds it
    # both ways: Steuersatz and Zinssatz are rates, but Umsatz is revenue and
    # every Umsatz column in the corpus ends in those same four letters. An
    # unanchored `satz` classifies the single most important amount column in a
    # German financial archive as a percentage and then refuses to add it up.
    ("rate", re.compile(
        r"prozent|percent|proz|(?<!um)satz$|quote|^rate|kurs$|marge|anteilin"
    )),
    ("period", re.compile(
        r"^jahr|^year|geschaeftsjahr|^monat|^month|^quartal|^quarter|^periode|"
        r"^period|^kw$|^woche|^week"
    )),
    ("identifier", re.compile(
        r"nummer|^nr|nr$|number|^id$|beleg|konto|iban|bic|referenz|"
        r"reference|kennzeichen|ustid|steuernummer|code$|kostenstelle"
    )),
)


def classify_column(header: str, values: list[str]) -> dict:
    """
    One column's role, from its header and its values.

    The header is the primary signal and the values are the check. A column of
    numbers whose header says nothing (`Spalte 3`) is still classified from what
    it holds; a column whose header says `Saldo` is a balance even if the values
    happen to look like ordinary amounts, because the header is the only place
    the semantics of "running" is ever written down.
    """
    raw = str(header or "").strip()
    key = norm_col(raw)

    nums = [parse_number(v) for v in values]
    dates = [parse_date(v) for v in values]
    non_empty = [v for v in values if str(v).strip()]
    n_num = sum(1 for n in nums if n is not None)
    n_date = sum(1 for d in dates if d is not None)
    n = max(1, len(non_empty))

    numeric = n_num / n >= 0.6
    datey = n_date / n >= 0.6

    hinted = ""
    for candidate, pat in _ROLE_PATTERNS:
        if key and pat.search(key):
            hinted = candidate
            break

    # Values overrule the header where they contradict it. `Belegdatum` matches
    # the identifier pattern on `beleg` but holds dates, and a date column is far
    # more useful classified as one. Conversely `Saldo` holding text is a label,
    # not a balance — there is nothing to refuse to add up.
    if datey:
        role = "date"
    elif hinted == "balance":
        role = "balance" if numeric else "label"
    elif hinted in ("rate", "period", "identifier"):
        role = hinted
    elif numeric:
        # Anything numeric that no rule claimed is treated as an amount. That is
        # the permissive direction on purpose: the alternative is refusing to sum
        # a genuine `Spalte 3` of euros, and an unwanted sum is visible in the
        # per-table breakdown while a refusal just looks like the tool is broken.
        role = "amount"
    else:
        role = "label"

    return {
        "name": key or f"spalte{abs(hash(raw)) % 1000}",
        "raw": raw,
        "role": role,
        "kind": "number" if numeric else ("date" if datey else "text"),
        "n_num": n_num,
        "n_date": n_date,
        "n_values": len(non_empty),
    }


def detect_running_balance(columns: list[dict], rows: list[list[str]]) -> list[str]:
    """
    Find balance columns that the header did NOT announce.

    A running balance is recognisable from its arithmetic: sorted into document
    order, every value equals the previous one plus some other column of the same
    row. A column called `Stand` or `Cumulative` carries no keyword but behaves
    exactly like `Saldo`, and summing it is the same error. Returns the column
    names to promote.

    Requires at least four rows — below that the relation holds by coincidence
    often enough to be worthless.
    """
    idx = {c["name"]: i for i, c in enumerate(columns)}
    numeric = [c for c in columns if c["kind"] == "number" and c["role"] == "amount"]
    if len(numeric) < 2 or len(rows) < 4:
        return []

    promoted: list[str] = []
    for cand in numeric:
        ci = idx[cand["name"]]
        series = [parse_number(r[ci]) if ci < len(r) else None for r in rows]
        if sum(1 for v in series if v is not None) < 4:
            continue
        for other in numeric:
            if other["name"] == cand["name"]:
                continue
            oi = idx[other["name"]]
            deltas = [parse_number(r[oi]) if oi < len(r) else None for r in rows]
            hits = tested = 0
            for i in range(1, len(series)):
                if series[i] is None or series[i - 1] is None or deltas[i] is None:
                    continue
                tested += 1
                if abs((series[i - 1] + deltas[i]) - series[i]) <= 0.01:
                    hits += 1
            if tested >= 3 and hits / tested >= 0.8:
                promoted.append(cand["name"])
                break
    return promoted


_TOTAL_ROW = re.compile(
    r"^\s*(summe|zwischensumme|gesamtsumme|gesamt|insgesamt|total|subtotal|sum|"
    r"bilanzsumme|ergebnis|betriebsergebnis|ebit|ebitda|jahresueberschuss|"
    r"jahresergebnis|grand total|net total|saldo)\b",
    re.I,
)

# The closing block of an invoice: net, VAT, gross. Summing the `Betrag` column
# of an invoice that has both line items AND this block counts every euro twice
# and the tax a third time.
#
# These words are only treated as totals in a table with NO DATE COLUMN, and that
# restriction is the whole reason the rule is safe. A summary block has no dates.
# A ledger does — and a ledger legitimately contains a booking line described as
# `Umsatzsteuer-Vorauszahlung`, which is a real movement that must be summed, not
# a total to be dropped. Without the date test this rule would silently delete
# real money from the ledger's totals.
_SUMMARY_ROW = re.compile(
    r"^\s*(nettobetrag|nettosumme|bruttobetrag|rechnungsbetrag|gesamtbetrag|"
    r"endbetrag|zu zahlen|zahlbetrag|umsatzsteuer|mehrwertsteuer|mwst|"
    r"net amount|gross amount|amount due|vat|tax total)\b",
    re.I,
)


def classify_row_kind(cells: list[str], columns: list[dict], has_date: bool = True) -> str:
    """
    'total' for a row that restates other rows, otherwise 'data'.

    `Summe Aktiva`, `Gesamt` and `Betriebsergebnis (EBIT)` sit in the same column
    as their own operands. Adding the column up counts them twice, and the result
    is off by exactly the plausible amount that makes it hard to spot.

    Only label-ish columns are examined, so a ledger booking whose *description*
    happens to be `Gesamtabrechnung` is safe.
    """
    for i, c in enumerate(columns):
        if c["role"] not in ("label", "identifier"):
            continue
        if i >= len(cells) or not cells[i]:
            continue
        cell = str(cells[i])
        if _TOTAL_ROW.match(cell):
            return "total"
        if not has_date and _SUMMARY_ROW.match(cell):
            return "total"
    return "data"


def detect_layout(columns: list[dict], grids: list[list[str]]) -> str:
    """
    'keyvalue' for a property sheet, 'records' for a register.

    A file shaped `Feld;Wert` / `Kunde;Alster GmbH` / `Nettobetrag;891.358,17`
    is ONE record written vertically. Its rows are fields, so counting them
    counts nothing anybody asked for — 266 such invoices report 3,598 "rows" and
    the figure looks entirely authoritative. The count is arithmetically perfect
    and answers the wrong question, which is the exact failure this module
    exists to prevent, so the shape has to be recognised rather than counted.

    Two columns, a near-unique first column, and a second column that mixes
    numbers with text. A real two-column register (`Position` / `2024 (EUR)`)
    fails the last test: its value column is numbers all the way down.
    """
    if len(columns) != 2:
        return "records"
    keys = [str(g[0]).strip() for g in grids if g and str(g[0]).strip()]
    vals = [str(g[1]).strip() for g in grids if len(g) > 1 and str(g[1]).strip()]
    if len(keys) < 3 or not vals:
        return "records"
    n_num = sum(1 for v in vals if parse_number(v) is not None)
    mixed = 0 < n_num < len(vals)
    return "keyvalue" if (len(set(keys)) / len(keys) > 0.9 and mixed) else "records"


def column_signature(columns: list[dict]) -> str:
    """
    What kind of record a table's rows are.

    Two tables with different columns hold different things, and counting their
    rows together answers a question nobody asked: an invoice's three line items
    plus an invoice register's sixty invoices is 63 of nothing.
    """
    return "|".join(sorted(c["name"] for c in columns))


# ── extraction ───────────────────────────────────────────────────────────────

def _pivot_keyvalue(payload: dict, rows: list[dict], grids: list[list[str]],
                    *, doc_id, source, doc_type, year, table_id) -> dict:
    """
    A `Feld;Wert` property sheet, pivoted into ONE record.

    The sheet's rows are the FIELDS of a single record written down the page, so
    stored row-wise it makes every aggregate wrong: 266 such invoices count as
    3,598 "rows", and their `Nettobetrag` cannot be summed at all because the
    amounts sit in a value column mixed with dates, names and currency codes.
    Pivoted, each sheet is one fact whose columns are its keys — so counting
    invoices counts invoices, and summing Nettobetrag across the archive answers
    the question that was actually asked.

    Column roles come from the key text plus its single value, through the same
    classifier as ordinary headers: `Belegnummer` is an identifier, `Nettobetrag
    (EUR)` an amount, `Geschaeftsjahr` a period. No total-row classification —
    one record has no subtotals — and no running-balance detection, which needs
    a series.
    """
    pairs = [
        (str(g[0]).strip(), str(g[1]).strip() if len(g) > 1 else "")
        for g in grids if g and str(g[0]).strip()
    ]

    columns, values, seen = [], [], set()
    for key, val in pairs:
        col = classify_column(key, [val])
        name = col["name"]
        if name in seen:
            k = 2
            while f"{name}{k}" in seen:
                k += 1
            name = col["name"] = f"{name}{k}"
        seen.add(name)
        columns.append(col)
        if val:
            values.append((0, name, val, parse_number(val), parse_date(val)))

    label = next(
        (v for (k, v), c in zip(pairs, columns, strict=True)
         if c["role"] in ("identifier", "label") and v),
        source,
    )
    first = rows[0].get("locator") or ""
    last = rows[-1].get("locator") or ""
    locator = f"{first}–{last}" if first and last and first != last else first

    return {
        "table_uid": str(table_id or f"{doc_id[:16]}:t"),
        "doc_id": doc_id,
        "source": source,
        "table_id": table_id,
        "sheet": payload.get("sheet"),
        "page": payload.get("page"),
        "doc_type": doc_type,
        "year": year,
        "n_rows": 1,
        "n_rows_indexed": 1 if payload.get("n_rows_indexed") else 0,
        "n_rows_total": 1,
        "n_rows_seen": 1,
        "n_total_rows": 0,
        # 'keyvalue_record' (pivoted, rows are records) is distinct from the
        # legacy 'keyvalue' (unpivoted, rows are fields) so that an index built
        # before this change still triggers the rows-are-fields warning until
        # its facts are re-extracted.
        "layout": "keyvalue_record",
        "columns": columns,
        "facts": [{"row": rows[0].get("row"), "locator": locator,
                   "row_kind": "data", "label": str(label)[:200]}],
        "values": values,
    }


def extract_table(payload: dict, *, doc_id, source, doc_type, year, table_id) -> dict | None:
    """
    One parsed table's full-row payload into records ready for storage.

    Returns None when the table carries nothing worth aggregating — no headers
    or no rows. A two-column key/value sheet is NOT rejected: it is detected and
    pivoted into a single record (see _pivot_keyvalue).
    """
    headers = [str(h or "").strip() for h in payload.get("headers") or []]
    rows = payload.get("rows") or []
    if len(headers) < 2 or not rows:
        return None

    grids = [r.get("cells") or [] for r in rows]
    columns: list[dict] = []
    seen: set[str] = set()
    for i, h in enumerate(headers):
        col = classify_column(h, [g[i] if i < len(g) else "" for g in grids])
        name = col["name"]
        # Two columns can normalise to the same key ("Betrag" and "Betrag ").
        # Left alone, one would silently shadow the other in every query.
        if name in seen:
            k = 2
            while f"{name}{k}" in seen:
                k += 1
            name = col["name"] = f"{name}{k}"
        seen.add(name)
        columns.append(col)

    if detect_layout(columns, grids) == "keyvalue":
        return _pivot_keyvalue(
            payload, rows, grids, doc_id=doc_id, source=source,
            doc_type=doc_type, year=year, table_id=table_id,
        )

    for name in detect_running_balance(columns, grids):
        for c in columns:
            if c["name"] == name:
                c["role"] = "balance"
                c["detected"] = "structure"

    # table_id is already prefixed with the document's content hash by the
    # parsers, so it is unique across the corpus and needs no further qualifying.
    table_uid = str(table_id or f"{doc_id[:16]}:t")
    facts, values, n_total = [], [], 0

    has_date = any(c["role"] == "date" for c in columns)
    for r, grid in zip(rows, grids, strict=True):
        if not any(str(c).strip() for c in grid):
            continue
        kind = classify_row_kind(grid, columns, has_date=has_date)
        if kind == "total":
            n_total += 1
        label = next(
            (str(grid[i]) for i, c in enumerate(columns)
             if c["role"] in ("label", "identifier") and i < len(grid) and str(grid[i]).strip()),
            "",
        )
        fact_ix = len(facts)
        facts.append({
            "row": r.get("row"),
            "locator": r.get("locator") or "",
            "row_kind": kind,
            "label": label[:200],
        })
        for i, c in enumerate(columns):
            cell = str(grid[i]).strip() if i < len(grid) else ""
            if not cell:
                continue
            values.append((fact_ix, c["name"], cell, parse_number(cell), parse_date(cell)))

    return {
        "table_uid": table_uid,
        "doc_id": doc_id,
        "source": source,
        "table_id": table_id,
        "sheet": payload.get("sheet"),
        "page": payload.get("page"),
        "doc_type": doc_type,
        "year": year,
        "n_rows": len(facts) - n_total,
        "n_rows_indexed": payload.get("n_rows_indexed", 0),
        "n_rows_total": payload.get("n_rows_total", len(facts)),
        "n_rows_seen": payload.get("n_rows_tabulated", len(rows)),
        "n_total_rows": n_total,
        "layout": "records",  # keyvalue returned early via _pivot_keyvalue
        "columns": columns,
        "facts": facts,
        "values": values,
    }


def clear_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    """Drop every fact belonging to a document. Caller holds the lock."""
    conn.execute("DELETE FROM facts WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM fact_tables WHERE doc_id=?", (doc_id,))


def store_table(conn: sqlite3.Connection, rec: dict) -> None:
    """Write one extracted table. Caller holds the lock and commits."""
    conn.execute("DELETE FROM fact_tables WHERE table_uid=?", (rec["table_uid"],))
    conn.execute(
        "INSERT INTO fact_tables(table_uid,doc_id,source,table_id,sheet,page,doc_type,"
        "year,n_rows,n_rows_indexed,n_rows_total,n_rows_seen,n_total_rows,layout,columns) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            rec["table_uid"], rec["doc_id"], rec["source"], rec["table_id"],
            rec["sheet"], rec["page"], rec["doc_type"], rec["year"],
            rec["n_rows"], rec["n_rows_indexed"], rec["n_rows_total"],
            rec["n_rows_seen"], rec["n_total_rows"], rec["layout"],
            json.dumps(rec["columns"], ensure_ascii=False),
        ),
    )
    ids: list[int] = []
    for f in rec["facts"]:
        cur = conn.execute(
            "INSERT INTO facts(table_uid,doc_id,source,row,locator,doc_type,year,row_kind,label)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (
                rec["table_uid"], rec["doc_id"], rec["source"], f["row"], f["locator"],
                rec["doc_type"], rec["year"], f["row_kind"], f["label"],
            ),
        )
        ids.append(int(cur.lastrowid))
    conn.executemany(
        "INSERT INTO fact_values(fact_id,table_uid,col,text,num,date) VALUES(?,?,?,?,?,?)",
        [(ids[ix], rec["table_uid"], col, txt, num, dt)
         for ix, col, txt, num, dt in rec["values"]],
    )


# ── querying ─────────────────────────────────────────────────────────────────

def _period_range(period: str) -> tuple[str, str] | None:
    """
    'Q3 2025' / '2025-Q3' / '2025-07' / '2025' to an inclusive ISO date range.

    Quarters are the unit the question is actually asked in — "how many invoices
    in Q3" — and making the agent compute 2025-07-01..2025-09-30 itself is a
    reliable source of off-by-one-month errors.
    """
    p = str(period or "").strip().upper().replace("_", "-")
    if not p:
        return None

    m = re.match(r"^(\d{4})$", p)
    if m:
        y = m.group(1)
        return f"{y}-01-01", f"{y}-12-31"

    m = re.match(r"^Q([1-4])[ \-/]*(\d{4})$", p) or re.match(r"^(\d{4})[ \-/]*Q([1-4])$", p)
    if m:
        a, b = m.group(1), m.group(2)
        q, y = (a, b) if len(a) == 1 else (b, a)
        q = int(q)
        start_m = 3 * (q - 1) + 1
        end_m = start_m + 2
        last = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[end_m]
        if end_m == 2 and int(y) % 4 == 0 and (int(y) % 100 != 0 or int(y) % 400 == 0):
            last = 29
        return f"{y}-{start_m:02d}-01", f"{y}-{end_m:02d}-{last:02d}"

    m = re.match(r"^(\d{4})-(\d{1,2})$", p)
    if m:
        y, mo = m.group(1), int(m.group(2))
        last = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}[mo]
        return f"{y}-{mo:02d}-01", f"{y}-{mo:02d}-{last:02d}"
    return None


def _match_tiers(requested: str, columns: list[dict]) -> tuple[int, list[dict]]:
    """
    Resolve a column name the model typed against a table's real headers.

    The model will ask for `Nettobetrag` when the header reads
    `Nettobetrag (EUR)`, and for `amount` when it reads `Betrag (EUR)`. Exact
    matching would fail on nearly every real call; matching too loosely would
    silently aggregate the wrong column, which is worse than failing. So the best
    tier that matches wins (0 exact, 1 prefix, 2 contains, 3 shared word), and
    ties are reported to the caller rather than resolved by guessing.

    The tier number is returned alongside the matches because it must be
    comparable ACROSS tables: `Betrag` prefix-matches a register's `Betrag
    (EUR)` and merely contains-matches an invoice header's `Nettobetrag (EUR)` —
    and the invoice header IS the total of those line items, so letting both
    tables join one sum double-counts every euro. The caller keeps only the
    tables that matched at the globally best tier.
    """
    want = norm_col(requested)
    if not want:
        return (99, [])
    tiers: list[list[dict]] = [[], [], [], []]
    for c in columns:
        name = c["name"]
        if name == want:
            tiers[0].append(c)
        elif name.startswith(want) or want.startswith(name):
            tiers[1].append(c)
        elif want in name or name in want:
            tiers[2].append(c)
        elif set(re.findall(r"[a-z]{3,}", want)) & set(re.findall(r"[a-z]{3,}", name)):
            tiers[3].append(c)
    for i, t in enumerate(tiers):
        if t:
            return (i, t)
    return (99, [])


def _match_column(requested: str, columns: list[dict]) -> list[dict]:
    """One table's best-tier matches. See _match_tiers for the tier semantics."""
    return _match_tiers(requested, columns)[1]


def select_tables(conn, lock, *, source="", doc_type="", year="") -> list[dict]:
    sql = "SELECT * FROM fact_tables WHERE 1=1"
    params: list = []
    if source:
        sql += " AND source LIKE ?"
        params.append(f"%{source}%")
    if doc_type:
        sql += " AND doc_type=?"
        params.append(doc_type)
    if year:
        sql += " AND year=?"
        params.append(year)
    with lock:
        rows = conn.execute(sql + " ORDER BY source, sheet", params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["columns"] = json.loads(d["columns"] or "[]")
        out.append(d)
    return out


class _Filter:
    """
    The per-table row filter, with join and where parameters kept apart.

    They have to be. SQLite binds `?` strictly by position in the finished
    statement, and every join placeholder precedes every where placeholder there
    — while a filter is naturally built as (join, its condition) pairs. Carrying
    both in one list binds a column name where a date belongs, which does not
    raise: it silently matches nothing, and an aggregate that answers 0 looks
    like a legitimate empty result rather than a bug.
    """

    __slots__ = ("joins", "join_params", "where", "where_params", "notes", "skip")

    def __init__(self):
        self.joins: list[str] = []
        self.join_params: list = []
        self.where: list[str] = []
        self.where_params: list = []
        self.notes: list[str] = []
        self.skip: str = ""


def _row_filter(table, *, where_column="", where_value="", period="") -> _Filter:
    f = _Filter()

    if where_column and where_value:
        cands = _match_column(where_column, table["columns"])
        if not cands:
            f.skip = f"no column like '{where_column}'"
            return f
        col = cands[0]
        f.joins.append("JOIN fact_values w ON w.fact_id=f.id AND w.col=?")
        f.join_params.append(col["name"])
        num = parse_number(where_value)
        if num is not None and col["kind"] == "number":
            f.where.append("w.num = ?")
            f.where_params.append(num)
        else:
            # Equality first, contains as the fallback: `Status = offen` must not
            # also match `offen (Mahnung 2)` unless nothing matched exactly.
            f.where.append("(w.text = ? COLLATE NOCASE OR w.text LIKE ?)")
            f.where_params.extend([where_value, f"%{where_value}%"])
        f.notes.append(f"{col['raw']} ~ '{where_value}'")

    rng = _period_range(period)
    if period and not rng:
        f.skip = f"'{period}' is not a period I can read (try '2025', 'Q3 2025' or '2025-07')"
        return f
    if rng:
        dates = [c for c in table["columns"] if c["role"] == "date"]
        if not dates:
            f.skip = "no date column"
            return f
        f.joins.append("JOIN fact_values d ON d.fact_id=f.id AND d.col=?")
        f.join_params.append(dates[0]["name"])
        f.where.append("d.date IS NOT NULL AND d.date >= ? AND d.date <= ?")
        f.where_params.extend(list(rng))
        f.notes.append(f"{dates[0]['raw']} in {period} ({rng[0]}..{rng[1]})")

    return f


_UNSUMMABLE = {
    "balance": (
        "it is a running balance — every row already contains the rows before it, "
        "so a sum of the column is a number with no meaning"
    ),
    "identifier": (
        "it holds identifiers (account codes, invoice or document numbers), not "
        "quantities"
    ),
    "rate": (
        "it holds rates or percentages, which cannot be added across rows without "
        "weighting them"
    ),
    "period": (
        "it holds a period (a year, month or quarter), which labels rows rather "
        "than measuring them"
    ),
    "date": "it holds dates",
    "label": "it holds text, not numbers",
}


def aggregate(
    conn,
    lock,
    *,
    operation: str,
    column: str = "",
    source: str = "",
    doc_type: str = "",
    year: str = "",
    where_column: str = "",
    where_value: str = "",
    period: str = "",
) -> dict:
    """
    Run one exact aggregate over every matching row, and report how it was scoped.

    The return value carries the answer AND its provenance — which tables were
    read, how many rows each contributed, what was excluded and why. That is not
    decoration: a total whose scope is invisible cannot be checked by the person
    reading the report, and this system's whole claim is that its figures can be.
    """
    op = (operation or "").strip().lower()
    if op in ("avg", "mean"):
        op = "average"
    if op not in OPERATIONS:
        return {
            "ok": False,
            "error": f"Unknown operation '{operation}'.",
            "operations": list(OPERATIONS),
        }

    tables = select_tables(conn, lock, source=source, doc_type=doc_type, year=year)
    if not tables:
        return {
            "ok": False,
            "error": "No table in the archive matched that scope.",
            "scope": {"source": source, "doc_type": doc_type, "year": year},
            "available": available_columns(conn, lock),
        }

    if op != "count" and not column:
        return {
            "ok": False,
            "error": f"'{op}' needs a column. Say which one.",
            "available": available_columns(conn, lock),
        }

    resolved: list[tuple[dict, dict | None]] = []
    ambiguous: list[dict] = []
    excluded_looser: list[str] = []

    if column:
        scored = []
        for t in tables:
            tier, cands = _match_tiers(column, t["columns"])
            if cands:
                scored.append((tier, t, cands))
        # Only the globally best tier participates. A table that merely
        # contains-matches must not join tables that matched exactly — the
        # loose match is usually a DIFFERENT quantity that happens to share a
        # word, and in an invoice it is the total of the very rows the precise
        # match is summing.
        best = min((s[0] for s in scored), default=99)
        for tier, t, cands in scored:
            if tier != best:
                excluded_looser.append(f"{cands[0]['raw']} ({t['source']})")
                continue
            if len(cands) > 1 and op != "count":
                ambiguous.append({"source": t["source"], "sheet": t["sheet"],
                                  "columns": [c["raw"] for c in cands]})
            resolved.append((t, cands[0]))
    else:
        resolved = [(t, None) for t in tables]

    if not resolved:
        return {
            "ok": False,
            "error": f"No column like '{column}' exists in the tables that matched that scope.",
            "scope": {"source": source, "doc_type": doc_type, "year": year},
            "available": [
                {"source": t["source"], "sheet": t["sheet"],
                 "columns": [{"name": c["raw"], "role": c["role"]} for c in t["columns"]]}
                for t in tables
            ],
        }

    # Refuse the operations a column's role cannot support — and say what to ask
    # instead. A bare refusal makes the agent retry the same call with different
    # words; naming the right column ends the loop.
    if op in ("sum", "average") and resolved:
        blocked = [(t, c) for t, c in resolved if c and c["role"] in _UNSUMMABLE]
        if blocked and len(blocked) == len([1 for _, c in resolved if c]):
            t, c = blocked[0]
            alt = [x["raw"] for x in t["columns"] if x["role"] == "amount"]
            res = {
                "ok": False,
                "refused": True,
                "operation": op,
                "column": c["raw"],
                "role": c["role"],
                "reason": _UNSUMMABLE[c["role"]],
                "source": t["source"],
                "alternatives": alt,
            }
            if c["role"] == "balance":
                flt = _row_filter(
                    t, where_column=where_column, where_value=where_value,
                    period=period,
                )
                res["closing"] = _closing_balance(
                    conn, lock, t, c, None if flt.skip else flt
                )
            return res

    if op == "closing":
        return _finish_closing(
            conn, lock, resolved, where_column, where_value, period,
        )

    per_table, warnings, notes = [], [], []
    if excluded_looser:
        warnings.append(
            f"also matched '{column}' but only loosely, so NOT included: "
            + ", ".join(excluded_looser[:8])
            + ". Name that column precisely if you meant it."
        )
    groups: dict[str, dict] = {}
    total_rows = matched_rows = excluded = 0
    values: list[float] = []
    distinct_values: set[str] = set()
    covered_docs: set[str] = set()

    for t, col in resolved:
        flt = _row_filter(
            t, where_column=where_column, where_value=where_value, period=period,
        )
        if flt.skip:
            where = f" ({t['sheet']})" if t["sheet"] else ""
            warnings.append(f"{t['source']}{where}: skipped — {flt.skip}")
            continue
        notes.extend(n for n in flt.notes if n not in notes)

        joins = list(flt.joins)
        join_params = list(flt.join_params)
        sql = "SELECT f.id, f.label, f.locator" + (", v.num AS num, v.text AS text" if col else "")
        sql += " FROM facts f "
        if col:
            joins.append("LEFT JOIN fact_values v ON v.fact_id=f.id AND v.col=?")
            join_params.append(col["name"])
        sql += " ".join(joins)
        sql += " WHERE f.table_uid=? AND f.row_kind='data'"
        params = join_params + [t["table_uid"]] + flt.where_params
        if flt.where:
            sql += " AND " + " AND ".join(flt.where)

        with lock:
            rows = conn.execute(sql, params).fetchall()

        nums = [r["num"] for r in rows if col and r["num"] is not None] if col else []
        texts = [r["text"] for r in rows if col and r["text"]] if col else []

        total_rows += t["n_rows"]
        matched_rows += len(rows)
        excluded += t["n_total_rows"]
        covered_docs.add(t["doc_id"])
        values.extend(nums)
        distinct_values.update(texts)

        if op == "breakdown" and col:
            for r in rows:
                key = (r["text"] or "(leer)").strip()
                g = groups.setdefault(key, {"count": 0, "sum": 0.0, "has_sum": False})
                g["count"] += 1

        per_table.append({
            "source": t["source"],
            "sheet": t["sheet"],
            "where": t["sheet"] or (f"S. {t['page']}" if t["page"] else ""),
            "layout": t.get("layout") or "records",
            "signature": column_signature(t["columns"]),
            "column": col["raw"] if col else "",
            "role": col["role"] if col else "",
            "rows_total": t["n_rows"],
            "rows_matched": len(rows),
            "rows_indexed": t["n_rows_indexed"],
            "totals_excluded": t["n_total_rows"],
            "value": _op_value(op, len(rows), nums, texts),
        })

    # A filter that could not be applied anywhere is NOT a count of zero.
    # "0 invoices in Q3" and "none of these tables has a date column" are
    # different answers, and returning the first for the second is the exact
    # failure this module exists to prevent — an empty result reads as a fact.
    if not per_table:
        return {
            "ok": False,
            "error": (
                "The filter could not be applied to any table in scope, so there "
                "is no figure to report — this is NOT a count of zero."
            ),
            "warnings": warnings,
            "scope": {"source": source, "doc_type": doc_type, "year": year},
        }

    if op == "breakdown":
        return _finish_breakdown(
            conn, lock, resolved, groups, column, per_table, warnings, notes,
            where_column, where_value, period, matched_rows, excluded,
        )

    value = _op_value(op, matched_rows, values, list(distinct_values))

    # Every kind of incompleteness the answer could be hiding, stated.
    capped = [t for t, _ in resolved if (t["n_rows_total"] or 0) > (t["n_rows_seen"] or 0)]
    if capped:
        warnings.append(
            f"{len(capped)} table(s) exceeded the aggregation row cap "
            f"({', '.join(t['source'] for t in capped)}); this figure is a LOWER BOUND."
        )
    # A sum that silently spans two tables is the quiet failure mode here: a
    # balance sheet and a P&L both have a `2024 (EUR)` column, and adding them
    # together produces a real-looking number that means nothing. Combining is
    # still allowed — across 300 quarterly invoice registers it is exactly what
    # was asked for — but never without saying that it happened.
    if op in ("sum", "average") and len(per_table) > 1:
        srcs = sorted({p["source"] for p in per_table})
        shown = ", ".join(srcs[:6]) + (
            f", … and {len(srcs) - 6} more" if len(srcs) > 6 else ""
        )
        warnings.append(
            f"this figure COMBINES {len(per_table)} separate tables ({shown}); "
            f"the per-table figures are listed below — check they measure the "
            f"same thing before quoting the combined one."
        )
    thin = [p for p in per_table if p["rows_indexed"] < p["rows_total"] + p["totals_excluded"]]
    if thin:
        notes.append(
            f"{len(thin)} table(s) have more rows than are searchable "
            f"({sum(p['rows_indexed'] for p in thin)} of {sum(p['rows_total'] for p in thin)} "
            f"indexed for retrieval) — this figure still covers all of them."
        )
    if op in ("sum", "average") and any(
        c and c["role"] not in ("amount",) for _, c in resolved if c
    ):
        mixed = {c["role"] for _, c in resolved if c and c["role"] != "amount"}
        warnings.append(
            f"some matched columns are not amounts ({', '.join(sorted(mixed))}); "
            f"they were included in the per-table figures — check them."
        )
    # A row is not a record. Counting rows across tables that hold different
    # kinds of row produces a number that is arithmetically correct and answers
    # nothing — an invoice's line items added to an invoice register's invoices.
    # The tool cannot know which the question meant, so it refuses to let the
    # single figure stand alone.
    # A property sheet's rows are the fields of ONE record. Counting them across
    # 266 invoice files answers "how many fields", presents it as a row count,
    # and is off by a factor of thirteen while looking completely authoritative.
    kv = [t for t, _ in resolved if (t.get("layout") or "records") == "keyvalue"]
    if kv and op == "count":
        warnings.append(
            f"{len(kv)} of these table(s) are property sheets — one record written "
            f"down the page, so their ROWS ARE FIELDS, NOT RECORDS. If you meant "
            f"how many records there are, the answer is {len(covered_docs)} "
            f"(the document count), not {matched_rows}."
        )
    shapes = {p["signature"] for p in per_table}
    if len(shapes) > 1:
        warnings.append(
            f"these {len(per_table)} tables have DIFFERENT columns, so their rows are "
            f"not the same kind of record — the combined figure counts unlike things. "
            f"Use the per-table figures, or narrow the scope with `source`."
        )
    if ambiguous:
        warnings.append(
            "the column name matched more than one header in "
            + ", ".join(a["source"] for a in ambiguous)
            + "; the first was used: "
            + "; ".join(f"{a['source']}: {' / '.join(a['columns'])}" for a in ambiguous)
        )

    return {
        "ok": True,
        "operation": op,
        "column": next((c["raw"] for _, c in resolved if c), ""),
        "value": value,
        "matched_rows": matched_rows,
        "scanned_rows": total_rows,
        "totals_excluded": excluded,
        "documents": len(covered_docs),
        "tables": len(per_table),
        "per_table": per_table,
        "filters": notes,
        "warnings": warnings,
        "overlap": _overlap_report(conn, lock, resolved, doc_type, covered_docs),
    }


def _op_value(op, n_rows, nums, texts):
    if op == "count":
        return n_rows
    if op == "distinct":
        return len(set(texts))
    if not nums:
        return None
    if op == "sum":
        return round(sum(nums), 2)
    if op == "average":
        return round(sum(nums) / len(nums), 2)
    if op == "min":
        return min(nums)
    if op == "max":
        return max(nums)
    return None


def _finish_closing(conn, lock, resolved, where_column, where_value, period) -> dict:
    """
    The closing value of a column, per table, in date order.

    A first-class operation rather than only a consolation prize inside the
    refusal, because the model needs a way to ASK for it. Left without one it
    reads the refusal, agrees that summing is wrong, and then estimates the
    closing balance from whatever rows search happened to return — trading a
    meaningless number for a plausible one, which is not an improvement.
    """
    results, warnings, notes = [], [], []
    for t, col in resolved:
        if not col:
            continue
        flt = _row_filter(
            t, where_column=where_column, where_value=where_value, period=period,
        )
        if flt.skip:
            where = f" ({t['sheet']})" if t["sheet"] else ""
            warnings.append(f"{t['source']}{where}: skipped — {flt.skip}")
            continue
        notes.extend(n for n in flt.notes if n not in notes)
        got = _closing_balance(conn, lock, t, col, flt)
        if got:
            got["where"] = t["sheet"] or (f"S. {t['page']}" if t["page"] else "")
            got["role"] = col["role"]
            results.append(got)
        if col["role"] not in ("balance", "amount"):
            warnings.append(
                f"\"{col['raw']}\" is a {col['role']} column, so its last value is "
                f"not a closing balance — check this is what you meant."
            )

    if not results:
        return {"ok": False, "error": "No closing value could be read for that column."}
    if len(results) > 1:
        warnings.append(
            f"{len(results)} tables have this column; each closing value is listed "
            f"separately. Closing balances of different accounts must not be added."
        )
    return {
        "ok": True,
        "operation": "closing",
        "column": results[0].get("column", ""),
        "value": results[0]["value"],
        "closing": results,
        "filters": notes,
        "warnings": warnings,
        "matched_rows": len(results),
        "per_table": [],
    }


def _finish_breakdown(conn, lock, resolved, groups, column, per_table,
                      warnings, notes, where_column, where_value, period,
                      matched_rows, excluded):
    """Counts and amounts per distinct value of a column."""
    for t, col in resolved:
        if not col:
            continue
        amounts = [c for c in t["columns"] if c["role"] == "amount"]
        if not amounts:
            continue
        acol = amounts[0]
        flt = _row_filter(
            t, where_column=where_column, where_value=where_value, period=period,
        )
        if flt.skip:
            continue
        sql = (
            "SELECT g.text AS grp, COUNT(*) AS n, SUM(a.num) AS total FROM facts f "
            + " ".join(flt.joins)
            + " JOIN fact_values g ON g.fact_id=f.id AND g.col=?"
            " LEFT JOIN fact_values a ON a.fact_id=f.id AND a.col=?"
            " WHERE f.table_uid=? AND f.row_kind='data'"
        )
        params = (
            flt.join_params + [col["name"], acol["name"], t["table_uid"]] + flt.where_params
        )
        if flt.where:
            sql += " AND " + " AND ".join(flt.where)
        sql += " GROUP BY g.text ORDER BY n DESC"
        with lock:
            for r in conn.execute(sql, params).fetchall():
                key = (r["grp"] or "(leer)").strip()
                g = groups.setdefault(key, {"count": 0, "sum": 0.0, "has_sum": False})
                if r["total"] is not None:
                    g["sum"] += r["total"]
                    g["has_sum"] = True
                    g["amount_column"] = acol["raw"]

    ordered = sorted(groups.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    return {
        "ok": True,
        "operation": "breakdown",
        "column": next((c["raw"] for _, c in resolved if c), column),
        "value": len(ordered),
        "groups": [
            {"key": k, "count": v["count"],
             "sum": round(v["sum"], 2) if v["has_sum"] else None,
             "amount_column": v.get("amount_column", "")}
            for k, v in ordered[:_MAX_LISTED]
        ],
        "groups_truncated": max(0, len(ordered) - _MAX_LISTED),
        "matched_rows": matched_rows,
        "totals_excluded": excluded,
        "per_table": per_table,
        "filters": notes,
        "warnings": warnings,
        "overlap": {},
    }


def _closing_balance(conn, lock, table, col, flt: _Filter | None = None) -> dict:
    """
    The last value of a balance column, in date order.

    This is the answer that a request to sum a balance actually meant: someone
    asking for the total of `Saldo` wants to know where the account stands.

    It must be the LAST value, never the largest. On an account that peaked
    mid-quarter and then paid out, `max` returns the peak — a real figure from a
    real row, wrong by whatever was spent since, and impossible to spot as wrong
    because it is correctly formatted and correctly cited.
    """
    dates = [c for c in table["columns"] if c["role"] == "date"]
    # A distinct alias: a period filter has already joined `d` for the same
    # column, and reusing the name makes every reference ambiguous — SQLite
    # raises rather than guessing, which is the good outcome, but only once.
    order = "oc.date DESC, f.row DESC" if dates else "f.row DESC"
    joins = list(flt.joins) if flt else []
    join_params = list(flt.join_params) if flt else []

    sql = "SELECT v.num AS num, v.text AS text, f.locator AS locator"
    sql += (", oc.date AS on_date" if dates else ", NULL AS on_date")
    sql += " FROM facts f JOIN fact_values v ON v.fact_id=f.id AND v.col=?"
    params: list = [col["name"]]
    if dates:
        joins.append("LEFT JOIN fact_values oc ON oc.fact_id=f.id AND oc.col=?")
        join_params.append(dates[0]["name"])
    sql += " " + " ".join(joins)
    sql += " WHERE f.table_uid=? AND f.row_kind='data' AND v.num IS NOT NULL"
    params = params + join_params + [table["table_uid"]]
    if flt and flt.where:
        sql += " AND " + " AND ".join(flt.where)
        params += flt.where_params
    sql += f" ORDER BY {order} LIMIT 1"
    with lock:
        r = conn.execute(sql, params).fetchone()
    if not r:
        return {}
    return {"value": r["num"], "date": r["on_date"], "locator": r["locator"],
            "source": table["source"], "column": col["raw"]}


def _overlap_report(conn, lock, resolved, doc_type, covered_docs) -> dict:
    """
    What this count did NOT see, and what it may have seen twice.

    Two ways a corpus total goes wrong that no amount of SQL correctness fixes:
    a standalone invoice PDF is a document with no rows, so counting register
    rows misses it; and the same invoice existing in both places would count it
    twice if it were naively added. Both are reported rather than resolved,
    because only the reader knows which of the two the question meant.
    """
    out: dict = {}
    if not resolved:
        return out

    # Documents that match the scope but contributed no table rows at all.
    if doc_type:
        with lock:
            rows = conn.execute(
                "SELECT source, doc_id FROM documents WHERE doc_type=?", (doc_type,)
            ).fetchall()
        missing = [r["source"] for r in rows if r["doc_id"] not in covered_docs]
        if missing:
            out["documents_without_rows"] = missing[:_MAX_LISTED]

    # Identifier values that appear in more than one table.
    uids = [t["table_uid"] for t, _ in resolved]
    if len(uids) > 1:
        id_cols = {
            t["table_uid"]: [c["name"] for c in t["columns"] if c["role"] == "identifier"]
            for t, _ in resolved
        }
        seen: dict[str, set[str]] = {}
        for uid, cols in id_cols.items():
            for cname in cols:
                with lock:
                    vals = conn.execute(
                        "SELECT DISTINCT text FROM fact_values WHERE table_uid=? AND col=?",
                        (uid, cname),
                    ).fetchall()
                for v in vals:
                    if v["text"]:
                        seen.setdefault(v["text"], set()).add(uid)
        dupes = [k for k, v in seen.items() if len(v) > 1]
        if dupes:
            out["identifiers_in_multiple_tables"] = sorted(dupes)[:_MAX_LISTED]
    return out


def available_columns(conn, lock) -> list[dict]:
    """Every table with its columns and their roles — the tool's vocabulary."""
    out = []
    for t in select_tables(conn, lock):
        out.append({
            "source": t["source"],
            "sheet": t["sheet"],
            "where": t["sheet"] or (f"S. {t['page']}" if t["page"] else ""),
            "rows": t["n_rows"],
            "rows_indexed": t["n_rows_indexed"],
            "totals_excluded": t["n_total_rows"],
            "doc_type": t["doc_type"],
            "year": t["year"],
            "columns": [
                {"name": c["raw"], "role": c["role"], "kind": c["kind"]}
                for c in t["columns"]
            ],
        })
    return out


def fact_stats(conn, lock) -> dict:
    with lock:
        t = conn.execute("SELECT COUNT(*) n FROM fact_tables").fetchone()["n"]
        r = conn.execute("SELECT COUNT(*) n FROM facts WHERE row_kind='data'").fetchone()["n"]
    return {"tables": t, "rows": r}
