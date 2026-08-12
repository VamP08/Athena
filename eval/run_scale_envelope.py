"""
eval/run_scale_envelope.py
How large an archive can Athena actually be trusted with?

    python eval/run_scale_envelope.py --answers

Why this exists
───────────────
"It works on 13 documents" and "it works on 1200" are different claims, and only
the first had ever been tested. `run_retrieval_eval.py` measures whether the
right passage is FOUND at scale (it is: recall@10 1.000 at 1200 documents).
Nothing measured whether the 9B model then USES the right one — and that is the
half the literature says breaks.

The published failure mode is vector search dilution: as a corpus grows, dense
similarity loses discriminative power and top-k fills with passages that are
topically plausible and factually wrong. Reported effects are severe — one
deployed corpus fell from 75% to below 40% accuracy going from 54 to 1,128
documents, and a highly-ranked distractor does more damage than a random one,
because the reader model treats rank as evidence. Athena should resist this
better than a naive dense pipeline: it is hybrid, so an exact token cannot be
diluted away, and its filters apply BEFORE ranking rather than after. "Should"
is not a measurement.

The experiment
──────────────
The question set is held CONSTANT and only the number of distractor documents
varies. That is the whole design: if the questions changed with the corpus, a
drop would be a different exam rather than a harder one.

  targets      N documents that the questions are actually about
  distractors  padding drawn from the rest of the corpus, seeded
  sizes        20 → 1200 documents, targets present in every one

Subset indexes are made by COPYING the full index and deleting the documents
that are not in the subset — never by re-embedding. A chunk's vector does not
depend on how many other documents exist, so the copy is exact, and it turns a
multi-hour re-embed into a file copy. FTS5 statistics DO depend on the corpus,
and they recompute correctly because deletion goes through the schema triggers.

Metrics, per corpus size:
  hit@1 / recall@k   does retrieval still rank the right document first
  answer_correct     is the ground-truth figure in the answer
  misattributed      did it answer with a figure belonging to a DIFFERENT
                     document that retrieval put in front of it — the
                     distractor failure, and the one that matters, because the
                     answer looks completely normal
  refused            did it decline instead of guessing (not a failure)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from core import index as idx  # noqa: E402
from eval.run_grounding_eval import (  # noqa: E402
    contains_value,
    looks_like_refusal,
    numbers_in,
)

DEFAULT_SIZES = (20, 50, 100, 200, 400, 800, 1200)
DEFAULT_ANSWER_SIZES = (20, 100, 400, 1200)


ANSWER_SYSTEM = """You are a financial analyst answering from an internal document archive.

Rules:
- Answer ONLY from the numbered passages provided. You have no other knowledge.
- Several passages may look similar and belong to DIFFERENT clients, quarters or
  years. Check that the passage you quote is the one the question actually asks
  about before you use its figure.
- Reproduce figures exactly as written, including the original number format.
- Name the source file you took the figure from.
- If the passages do not contain the answer, say so plainly. A stated gap is a
  correct answer; a figure from the wrong document is not.

Answer in two lines:
Antwort: <the figure>
Quelle: <source file name>
"""


# ── subset construction ──────────────────────────────────────────────────────

def make_subset(src: str, dst: str, keep: set[str]) -> dict:
    """
    A smaller index, by deletion rather than re-ingestion.

    Deleting goes through `chunks`, which fires the FTS5 delete trigger, so the
    lexical index shrinks with it. Doing this by hand instead would leave the
    removed documents' terms searchable and resolvable to another document's
    text — the exact defect `test_index_integrity.py` exists to prevent.
    """
    idx.close()
    # Fold the WAL back into the main file first, or the copy is a stale snapshot.
    conn = sqlite3.connect(src)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    Path(dst).unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(dst + suffix).unlink(missing_ok=True)
    shutil.copyfile(src, dst)

    conn = idx.connect(dst)
    with idx._LOCK:
        rows = conn.execute("SELECT doc_id, source FROM documents").fetchall()
        drop = [r["doc_id"] for r in rows if r["source"] not in keep]
        for doc_id in drop:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM facts WHERE doc_id=?", (doc_id,))
            conn.execute("DELETE FROM fact_tables WHERE doc_id=?", (doc_id,))
        conn.commit()
    stats = idx.index_stats(dst)
    idx.close()
    return stats


def choose_targets(gt: dict, n_targets: int, seed: int) -> tuple[list[str], list[dict]]:
    """Target documents and every query that asks about one of them."""
    by_source: dict[str, list[dict]] = {}
    for q in gt["queries"]:
        by_source.setdefault(q["expect_source"], []).append(q)

    # Prefer documents carrying the most queries, so a given LLM budget buys the
    # most questions; break ties by name so the selection is reproducible.
    ranked = sorted(by_source, key=lambda s: (-len(by_source[s]), s))
    targets = sorted(ranked[:n_targets])
    queries = [q for s in targets for q in by_source[s]]
    queries.sort(key=lambda q: q["id"])
    rng = random.Random(seed)
    rng.shuffle(queries)
    return targets, queries


def subset_sources(gt: dict, targets: list[str], size: int, seed: int) -> set[str]:
    """`size` documents that always include every target."""
    all_sources = [d["source"] if isinstance(d, dict) else d for d in gt["documents"]]
    pool = sorted(set(all_sources) - set(targets))
    rng = random.Random(seed)
    rng.shuffle(pool)
    need = max(0, size - len(targets))
    return set(targets) | set(pool[:need])


# ── scoring ──────────────────────────────────────────────────────────────────

def score_answer(text: str, q: dict, hits: list[dict]) -> dict:
    """
    One answer against ground truth, and against the distractors it was shown.

    `misattributed` is the metric this harness was built for. It is not "got it
    wrong" — it is "answered with a real figure from the wrong document", which
    is the failure that survives every surface check: correctly formatted, drawn
    from a genuinely retrieved passage, and cited.

    `amount_lookup` is scored on the CITATION, not the figure. Its question is
    "which document shows exactly 660.211,29 EUR?", so the expected answer is a
    filename and `expect_text` is the search key rather than the thing to say
    back. Scoring it like the others marked a completely correct answer wrong,
    and since that kind is 11% of the query set, it would have understated the
    supported corpus size across the whole curve. Caught by reading three
    answers before trusting an aggregate.
    """
    want = q["expect_text"]
    cited = q["expect_source"].lower() in (text or "").lower()
    refused = looks_like_refusal(text)
    kind = q.get("kind", "")

    # Figures that belong to some OTHER document in the retrieved window.
    want_d = numbers_in(want, minimum=Decimal(0))
    neighbour = set()
    named_others = set()
    for h in hits:
        if h["source"] == q["expect_source"]:
            continue
        neighbour |= numbers_in(h.get("text") or "", minimum=Decimal(1000))
        if h["source"] and h["source"].lower() in (text or "").lower():
            named_others.add(h["source"])
    said = numbers_in(text or "", minimum=Decimal(1000))

    if kind == "amount_lookup":
        correct = cited
        wrong_doc = bool(not cited and not refused and named_others)
    else:
        correct = bool(contains_value(text, [want]))
        wrong_doc = bool(not correct and not refused and (said & (neighbour - want_d)))

    return {
        "answer_correct": bool(correct),
        "cited_correct": bool(cited),
        "refused": bool(refused),
        "misattributed": wrong_doc,
    }


def retrieval_pass(queries: list[dict], index_path: str, k: int) -> dict:
    hit1 = hit = 0
    rr = 0.0
    for q in queries:
        hits = idx.search(q["query"], k=k, index_path=index_path)
        sources = [h["source"] for h in hits]
        rank = sources.index(q["expect_source"]) + 1 if q["expect_source"] in sources else 0
        hit += 1 if rank else 0
        hit1 += 1 if rank == 1 else 0
        rr += (1.0 / rank) if rank else 0.0
    n = len(queries) or 1
    return {"n": len(queries), "hit_at_1": hit1 / n, "recall_at_k": hit / n, "mrr": rr / n}


def answer_pass(queries: list[dict], index_path: str, k: int, llm, progress=True) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage

    rows = []
    for i, q in enumerate(queries, start=1):
        hits = idx.search(q["query"], k=k, index_path=index_path)
        block = "\n\n".join(
            f"[{j}] {h['source']}" + (f" — {h['locator']}" if h.get("locator") else "")
            + f"\n{(h.get('text') or '').strip()[:900]}"
            for j, h in enumerate(hits, start=1)
        )
        msg = llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Passagen:\n\n{block}\n\nFrage: {q['query']}"),
        ])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        rec = {"id": q["id"], "kind": q.get("kind", "?")}
        rec.update(score_answer(text, q, hits))
        rec["answer"] = text[:400]
        rows.append(rec)
        if progress:
            mark = "OK " if rec["answer_correct"] else ("MIS" if rec["misattributed"] else "-- ")
            print(f"    {mark} [{i}/{len(queries)}] {q['id']}", flush=True)

    n = len(rows) or 1
    return {
        "n": len(rows),
        "answer_correct": sum(r["answer_correct"] for r in rows) / n,
        "cited_correct": sum(r["cited_correct"] for r in rows) / n,
        "misattributed": sum(r["misattributed"] for r in rows) / n,
        "refused": sum(r["refused"] for r in rows) / n,
        "rows": rows,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="corpus_scale_groundtruth.json")
    ap.add_argument("--index", default="athena_scale.db")
    ap.add_argument("--work", default="scale_subset.db")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--targets", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    ap.add_argument("--answer-sizes", default=",".join(str(s) for s in DEFAULT_ANSWER_SIZES))
    ap.add_argument("--answers", action="store_true", help="also run the LLM answer pass")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of questions")
    ap.add_argument("--out", default="eval/scale_envelope.json")
    args = ap.parse_args()

    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    targets, queries = choose_targets(gt, args.targets, args.seed)
    if args.limit:
        queries = queries[: args.limit]

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    answer_sizes = {int(s) for s in args.answer_sizes.split(",") if s.strip()}

    print(f"Targets: {len(targets)} documents · Questions: {len(queries)} (held constant)")
    print(f"Sizes: {sizes} · k={args.k} · seed={args.seed}")
    if args.answers:
        print(f"Answer pass at: {sorted(answer_sizes)}")
    print()

    llm = None
    if args.answers:
        from core.llm import get_llm
        llm = get_llm()

    results = []
    for size in sizes:
        keep = subset_sources(gt, targets, size, args.seed)
        stats = make_subset(args.index, args.work, keep)
        t0 = time.time()
        row = {
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "distractors": stats["documents"] - len(targets),
            "retrieval": retrieval_pass(queries, args.work, args.k),
        }
        print(f"  {row['documents']:>5} docs / {row['chunks']:>6} chunks  "
              f"hit@1 {row['retrieval']['hit_at_1']:.3f}  "
              f"recall@{args.k} {row['retrieval']['recall_at_k']:.3f}  "
              f"MRR {row['retrieval']['mrr']:.3f}   ({time.time()-t0:.0f}s)", flush=True)

        if args.answers and size in answer_sizes:
            a = answer_pass(queries, args.work, args.k, llm)
            row["answers"] = a
            print(f"        answer {a['answer_correct']:.3f}  "
                  f"misattributed {a['misattributed']:.3f}  "
                  f"refused {a['refused']:.3f}  cited {a['cited_correct']:.3f}", flush=True)
        results.append(row)
        idx.close()

    Path(args.work).unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(args.work + suffix).unlink(missing_ok=True)

    print("\n── Envelope ──")
    print(f"{'docs':>6} {'chunks':>7} {'hit@1':>7} {'recall':>7} {'MRR':>7} "
          f"{'answer':>7} {'misattr':>8}")
    for r in results:
        a = r.get("answers") or {}
        print(f"{r['documents']:>6} {r['chunks']:>7} "
              f"{r['retrieval']['hit_at_1']:>7.3f} {r['retrieval']['recall_at_k']:>7.3f} "
              f"{r['retrieval']['mrr']:>7.3f} "
              f"{a.get('answer_correct', float('nan')):>7.3f} "
              f"{a.get('misattributed', float('nan')):>8.3f}")

    Path(args.out).write_text(
        json.dumps(
            {
                "targets": targets,
                "n_questions": len(queries),
                "k": args.k,
                "seed": args.seed,
                "results": results,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
