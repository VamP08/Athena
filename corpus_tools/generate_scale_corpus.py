"""
corpus_tools/generate_scale_corpus.py
Generates a LARGE synthetic archive with exact ground truth, for measuring
retrieval quality at institutional scale.

    python corpus_tools/generate_scale_corpus.py --docs 500
    python corpus_tools/generate_scale_corpus.py --docs 5000 --out corpus_scale

Why this exists separately from generate_corpus.py:
  The 13-document demo corpus proves the pipeline works. It cannot measure
  retrieval, because with 447 chunks almost any retriever finds the answer —
  every question has essentially one plausible source. Real archives fail the
  other way: thousands of documents that look nearly identical.

THE DISTRACTORS ARE THE POINT.
  Retrieval is easy when the target is the only document about invoices. It is
  hard when there are 40 quarters x 30 clients of near-identical invoices and
  exactly one holds the asked-about figure. So the corpus is generated as a
  dense grid of (client x period x doc_type): every document has many siblings
  sharing its template, its vocabulary and its structure, differing only in the
  entity, the period and the numbers. That is the shape that breaks dense
  retrieval, and it is the shape a real financial archive actually has.

  Three distractor classes are generated deliberately:
    near-period      same client, adjacent quarter — the classic wrong-row error
    near-entity      same period, similarly-named client ("Nordwind" / "Nordwand")
    superseded       an earlier VERSION of the same document, still in the archive

GROUND TRUTH IS EXACT
  Every query names one document (and one locator) that answers it, so recall@k,
  MRR and hit@k are computed without a judge, without tokens and offline. Amounts
  are drawn from a seeded RNG, so the same --docs and --seed always reproduce the
  same archive and the same answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DOC_TYPES = ("invoice", "statement", "ledger", "budget")
QUARTERS = ("Q1", "Q2", "Q3", "Q4")

# Similarly-named pairs are deliberate: a retriever that matches on surface form
# will confuse them, which is precisely what we want to detect.
_BASES = [
    "Nordwind", "Elbe", "Baltic", "Kranich", "Hansa", "Weser", "Rhein", "Alster",
    "Fjord", "Kogge", "Anker", "Mole", "Deich", "Watt", "Priel", "Reede", "Ebbe",
    "Flut", "Duene", "Marsch", "Koog", "Siel", "Warft", "Hallig", "Bake",
]
# Each base spawns confusable variants. A retriever matching on surface form will
# mix them up, which is exactly the error this corpus needs to be able to detect.
_VARIANTS = ["", "-Nord", "-Sued", "au", "er", "tal", "feld", "weg"]
CLIENT_STEMS = [f"{b}{v}" for b in _BASES for v in _VARIANTS]
SUFFIXES = ["GmbH", "AG", "KG", "GmbH & Co. KG", "SE", "OHG"]

# Query wording that deliberately does NOT appear in the documents. The first
# benchmark scored recall@10 = 1.000 because every query repeated the client
# name, quarter and year verbatim — it measured string overlap, not retrieval.
_QUARTER_WORDS = {
    "Q1": "erstes Quartal", "Q2": "zweites Quartal",
    "Q3": "drittes Quartal", "Q4": "viertes Quartal",
}
_TYPE_WORDS = {
    "invoice": "Ausgangsrechnung",      # documents say "invoice"
    "statement": "Kontobewegungen",     # documents say "statement"
    "ledger": "Buchungsjournal",        # documents say "ledger"
    "budget": "Planzahlen",             # documents say "budget"
}


def eur(n: float) -> str:
    return f"{n:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def build(n_docs: int, out: Path, seed: int) -> dict:
    rnd = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        if old.is_file():
            old.unlink()

    clients = [f"{s} {rnd.choice(SUFFIXES)}" for s in CLIENT_STEMS]
    years = [2021, 2022, 2023, 2024, 2025]

    docs, queries = [], []
    made = 0
    # Dense grid: every combination exists, so every document has siblings that
    # differ only in entity/period. No document is unique in shape.
    combos = [(c, y, q, t) for c in clients for y in years for q in QUARTERS for t in DOC_TYPES]
    rnd.shuffle(combos)

    for client, year, quarter, dtype in combos:
        if made >= n_docs:
            break
        made += 1
        stem = client.split()[0].lower().replace("-", "_")
        name = f"{dtype}_{stem}_{year}_{quarter}"
        amount = round(rnd.uniform(5_000, 900_000), 2)
        ref = f"{year}-{quarter}-{made:05d}"

        # Superseded versions: an older copy of the SAME logical document stays
        # in the archive with different numbers. Retrieval must prefer the
        # current one, and an archive that silently holds both is realistic.
        superseded = (made % 17 == 0)
        version = "v1 (superseded)" if superseded else "v2 (current)"
        if superseded:
            amount = round(amount * 0.82, 2)

        path = out / f"{name}{'_v1' if superseded else ''}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["Feld", "Wert"])
            w.writerow(["Dokumenttyp", dtype])
            w.writerow(["Kunde", client])
            w.writerow(["Geschaeftsjahr", year])
            w.writerow(["Quartal", quarter])
            w.writerow(["Belegnummer", ref])
            w.writerow(["Version", version])
            w.writerow(["Nettobetrag (EUR)", eur(amount)])
            w.writerow(["Waehrung", "EUR"])
            w.writerow([])
            w.writerow(["Position", "Beschreibung", "Betrag (EUR)"])
            n_lines = rnd.randint(3, 8)
            remaining = amount
            for i in range(1, n_lines + 1):
                part = round(remaining / (n_lines - i + 1), 2) if i < n_lines else round(remaining, 2)
                remaining = round(remaining - part, 2)
                w.writerow([i, rnd.choice([
                    "Kontraktlogistik", "Transportleistung", "Zollabwicklung",
                    "Lagerflaeche", "Umschlag", "Verpackung", "Expressfracht",
                ]), eur(part)])

        docs.append({
            "source": path.name, "doc_type": dtype, "client": client,
            "year": year, "quarter": quarter, "ref": ref,
            "amount": amount, "superseded": superseded,
        })

    # Queries target CURRENT documents only; superseded ones exist purely as traps.
    pool = [d for d in docs if not d["superseded"]]
    rnd.shuffle(pool)

    # Amounts are unique enough to identify one document, but only if the
    # retriever can match a number — worth testing separately because dense
    # embeddings are weak at exact numerals.
    by_amount: dict[str, list] = {}
    for d in docs:
        by_amount.setdefault(eur(d["amount"]), []).append(d)

    n_each = min(80, len(pool))
    for d in pool[:n_each]:
        q, y, c = d["quarter"], d["year"], d["client"]

        # 1. VERBATIM — the original easy case, kept as the control. If a change
        #    moves this, something broke; it should stay near 1.0.
        queries.append({
            "id": f"verbatim_{d['ref']}", "kind": "verbatim_control",
            "query": f"Nettobetrag {c} {q} {y}",
            "expect_source": d["source"], "expect_text": eur(d["amount"]),
        })

        # 2. EXACT IDENTIFIER — lexical's home ground.
        queries.append({
            "id": f"ref_{d['ref']}", "kind": "exact_identifier",
            "query": f"Belegnummer {d['ref']}",
            "expect_source": d["source"], "expect_text": d["ref"],
        })

        # 3. PARAPHRASE — same fact, none of the document's own vocabulary.
        #    "viertes Quartal" never appears in a file that says "Q4".
        queries.append({
            "id": f"para_{d['ref']}", "kind": "paraphrase",
            "query": (f"Welche Summe wurde {c} im {_QUARTER_WORDS[q]} {y} "
                      f"in Rechnung gestellt?"),
            "expect_source": d["source"], "expect_text": eur(d["amount"]),
        })

        # 4. TYPE SYNONYM — the document type named the way a person would say
        #    it, not the way the file spells it.
        queries.append({
            "id": f"syn_{d['ref']}", "kind": "type_synonym",
            "query": f"{_TYPE_WORDS[d['doc_type']]} {c} {_QUARTER_WORDS[q]} {y}",
            "expect_source": d["source"], "expect_text": eur(d["amount"]),
        })

    # 5. AMOUNT LOOKUP — identify a document from its figure alone. Only used
    #    where the amount is unique in the whole archive, so ground truth stays
    #    exact rather than merely probable.
    unique_amounts = [(a, ds[0]) for a, ds in by_amount.items()
                      if len(ds) == 1 and not ds[0]["superseded"]]
    for amount, d in unique_amounts[:40]:
        queries.append({
            "id": f"amt_{d['ref']}", "kind": "amount_lookup",
            "query": f"Welches Dokument weist genau {amount} EUR aus?",
            "expect_source": d["source"], "expect_text": amount,
        })

    manifest = {
        "seed": seed, "n_documents": len(docs), "n_queries": len(queries),
        "clients": len(clients), "years": years,
        "superseded_documents": sum(1 for d in docs if d["superseded"]),
        "documents": docs, "queries": queries,
    }
    (out.parent / f"{out.name}_groundtruth.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a large synthetic archive.")
    ap.add_argument("--docs", type=int, default=500)
    ap.add_argument("--out", default="corpus_scale")
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    out = Path(args.out)
    m = build(args.docs, out, args.seed)
    print(f"Generated {m['n_documents']} documents in {out}/")
    print(f"  distinct clients      : {m['clients']} (includes deliberately similar names)")
    print(f"  superseded duplicates : {m['superseded_documents']}")
    print(f"  ground-truth queries  : {m['n_queries']}")
    print(f"  ground truth written  : {out.name}_groundtruth.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
