"""
eval/run_scale_composition.py
Is the scale limit really about SIZE — or about what the corpus is made of?

    python eval/run_scale_composition.py

Why this exists
───────────────
The envelope sweep (run_scale_envelope.py) measured degradation against document
count, and by construction its chunk count moved in lockstep (the synthetic
corpus is uniform, ~15 chunks/document). That leaves two confounded hypotheses:

  H1  chunk count: more chunks = more candidates near any query, so top-k
      dilutes no matter what the chunks contain.
  H2  confusable density: what dilutes top-k is documents that RESEMBLE the
      target — same client, same period, different type. The one misattribution
      the envelope run produced was exactly that shape (statement_deich quoted
      when budget_deich held the truth).

They prescribe different units for the operating envelope. A 1000-page annual
report is thousands of chunks but few confusables; a folder of quarterly files
for one client is few chunks and all confusables. Under H1 the first is the
danger; under H2 the second is. Publishing a limit in the wrong unit would
be precise and misleading at the same time.

The probe
─────────
Same 12 target documents, same 51 questions, same seed as the envelope run —
only the PADDING changes. The ground truth records each document's client, so
"confusable" is exact: a padding document is confusable iff its client appears
among the target documents' clients.

  targets-only      floor: 12 documents, no padding at all
  confusable-only   every other document belonging to a target client — small
                    corpus, maximum confusability
  disjoint@3k       non-target-client padding to ~3,064 chunks (the 200-doc
                    scale where the envelope was perfect)
  disjoint@6k       ... to ~6,197 chunks (the 400-doc scale where hit@1 fell
                    to 0.961)
  disjoint@max      every non-target-client document (~1,100+ docs) — the kill
                    shot: envelope-run chunk counts with near-zero confusables
  both@6k           confusables PLUS disjoint padding to ~6,197 chunks — the
                    realistic composition, for direct comparison with the
                    envelope run's random 400-doc point

Retrieval-only: no LLM, minutes not hours, and the envelope run already showed
answer failures are strictly downstream of hit@1 failures — so hit@1 is the
signal worth isolating.

Reading the result:
  disjoint@max ≈ 1.000 and confusable-only < 1.000  → H2. Composition drives
      the limit; chunk count is only the conservative proxy for it.
  disjoint@6k falls like the envelope's 400-doc point → H1. Raw scale is the
      driver and chunks are the honest unit outright.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from core import index as idx  # noqa: E402
from eval.run_scale_envelope import (  # noqa: E402
    choose_targets,
    make_subset,
    retrieval_pass,
)


def chunk_counts(index_path: str) -> dict[str, int]:
    """Chunks per source file, read from the FULL index (no re-embedding ever)."""
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT d.source AS source, COUNT(c.id) AS n "
        "FROM documents d JOIN chunks c ON c.doc_id = d.doc_id GROUP BY d.source"
    ).fetchall()
    conn.close()
    return {r["source"]: r["n"] for r in rows}


def pad_to_chunks(pool: list[str], counts: dict[str, int], base: int,
                  budget: int, seed: int) -> list[str]:
    """Padding documents until the subset's chunk count reaches `budget`."""
    rng = random.Random(seed)
    pool = list(pool)
    rng.shuffle(pool)
    out, total = [], base
    for src in pool:
        if total >= budget:
            break
        out.append(src)
        total += counts.get(src, 0)
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="corpus_scale_groundtruth.json")
    ap.add_argument("--index", default="athena_scale.db")
    ap.add_argument("--work", default="scale_subset_comp.db")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--targets", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260812)  # MUST match the envelope run
    ap.add_argument("--out", default="eval/scale_composition.json")
    args = ap.parse_args()

    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    targets, queries = choose_targets(gt, args.targets, args.seed)
    counts = chunk_counts(args.index)

    by_source = {d["source"]: d for d in gt["documents"]}
    target_clients = {by_source[t]["client"] for t in targets}
    confusable = sorted(
        d["source"] for d in gt["documents"]
        if d["client"] in target_clients and d["source"] not in targets
    )
    disjoint = sorted(
        d["source"] for d in gt["documents"] if d["client"] not in target_clients
    )
    base_chunks = sum(counts.get(t, 0) for t in targets)

    print(f"Targets: {len(targets)} docs / {base_chunks} chunks · "
          f"{len(queries)} questions (same set as the envelope run)")
    print(f"Target clients: {len(target_clients)} · confusable docs: {len(confusable)} "
          f"({sum(counts.get(s, 0) for s in confusable)} chunks) · "
          f"disjoint pool: {len(disjoint)} docs")
    print()

    conditions = [
        ("targets-only", set(targets)),
        ("confusable-only", set(targets) | set(confusable)),
        ("disjoint@3k", set(targets) | set(
            pad_to_chunks(disjoint, counts, base_chunks, 3064, args.seed))),
        ("disjoint@6k", set(targets) | set(
            pad_to_chunks(disjoint, counts, base_chunks, 6197, args.seed))),
        ("disjoint@max", set(targets) | set(disjoint)),
        ("both@6k", set(targets) | set(confusable) | set(
            pad_to_chunks(
                disjoint, counts,
                base_chunks + sum(counts.get(s, 0) for s in confusable),
                6197, args.seed))),
    ]

    results = []
    for name, keep in conditions:
        stats = make_subset(args.index, args.work, keep)
        r = retrieval_pass(queries, args.work, args.k)
        idx.close()
        row = {
            "condition": name,
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "confusable_docs": len(keep & set(confusable)),
            **r,
        }
        results.append(row)
        print(f"  {name:<16} {row['documents']:>5} docs {row['chunks']:>6} chunks "
              f"({row['confusable_docs']:>3} confusable)   "
              f"hit@1 {r['hit_at_1']:.3f}  recall@{args.k} {r['recall_at_k']:.3f}  "
              f"MRR {r['mrr']:.3f}", flush=True)

    Path(args.work).unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(args.work + suffix).unlink(missing_ok=True)

    Path(args.out).write_text(
        json.dumps({"targets": targets, "n_questions": len(queries),
                    "k": args.k, "seed": args.seed, "results": results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
