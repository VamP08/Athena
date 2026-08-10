"""
eval/run_retrieval_eval.py
Retrieval quality, measured without a model.

    python eval/run_retrieval_eval.py --ground-truth corpus_scale_groundtruth.json

Why this is separate from run_grounding_eval.py:
  The grounding harness runs the whole graph and therefore measures retrieval AND
  generation together. When a case fails you cannot tell which broke — and that
  ambiguity has already cost real debugging time here: an empty report scored as
  a missing answer, which reads like a retrieval bug but was a writer bug.

  This harness calls the retriever DIRECTLY. No LLM, no tokens, no rate limits,
  runs offline in seconds, and its numbers are deterministic. That makes it the
  only eval in the project that can gate CI, and it is the ROADMAP M5
  "retrieval-only metrics" item.

Metrics (standard IR, computed over document-level ground truth):
  recall@k   was the correct document retrieved in the top k at all
  hit@1      was it ranked first
  MRR@k      1/rank of the first correct document, averaged — rewards ranking it
             high rather than merely including it
  Reported per query KIND as well as overall, because the aggregate hides the
  finding that matters: exact-identifier lookups and semantic lookups fail for
  opposite reasons, and a single average lets one mask the other.

Also reported: how often a SUPERSEDED document outranks the current one. An
archive that keeps old versions is realistic, and serving last year's figure as
if it were current is a wrong answer that no grounding metric would catch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from core import index as idx  # noqa: E402


def evaluate(queries: list[dict], k: int, index_path: str | None,
             mode: str = "hybrid") -> dict:
    """
    Score every query. `mode` selects which ranker(s) run, so the contribution of
    the dense and lexical halves can be measured separately — the hybrid design
    is a documented decision and should be re-justified with numbers, not
    inherited on faith.
    """
    per_kind: dict[str, list] = defaultdict(list)
    rows = []
    superseded_wins = 0
    t0 = time.time()

    for q in queries:
        if mode == "hybrid":
            hits = idx.search(q["query"], k=k, index_path=index_path)
        else:
            hits = idx.search_channel(q["query"], k=k, index_path=index_path, channel=mode)

        sources = [h["source"] for h in hits]
        want = q["expect_source"]
        rank = sources.index(want) + 1 if want in sources else 0

        # Did an older version of some document outrank the target?
        if rank != 1 and hits and "_v1" in (sources[0] or ""):
            superseded_wins += 1

        rec = {
            "id": q["id"], "kind": q.get("kind", "?"), "rank": rank,
            "hit": rank > 0, "hit1": rank == 1,
            "rr": (1.0 / rank) if rank else 0.0,
            "top": sources[0] if sources else None,
        }
        rows.append(rec)
        per_kind[rec["kind"]].append(rec)

    def agg(items):
        n = len(items) or 1
        return {
            "n": len(items),
            "recall_at_k": sum(1 for r in items if r["hit"]) / n,
            "hit_at_1": sum(1 for r in items if r["hit1"]) / n,
            "mrr": sum(r["rr"] for r in items) / n,
        }

    return {
        "mode": mode, "k": k, "seconds": round(time.time() - t0, 1),
        "overall": agg(rows),
        "by_kind": {kind: agg(items) for kind, items in sorted(per_kind.items())},
        "superseded_outranked_current": superseded_wins,
        "rows": rows,
    }


def _print(res: dict) -> None:
    o = res["overall"]
    print(f"\n── {res['mode']} (k={res['k']}, {res['seconds']}s, {o['n']} queries) ──")
    print(f"  recall@{res['k']:<3d} {o['recall_at_k']:.3f}")
    print(f"  hit@1      {o['hit_at_1']:.3f}")
    print(f"  MRR        {o['mrr']:.3f}")
    for kind, a in res["by_kind"].items():
        print(f"    {kind:<24s} n={a['n']:<4d} recall={a['recall_at_k']:.3f} "
              f"hit@1={a['hit_at_1']:.3f} mrr={a['mrr']:.3f}")
    if res["superseded_outranked_current"]:
        print(f"  superseded version ranked first: {res['superseded_outranked_current']} "
              f"<- serving an outdated figure as current")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="corpus_scale_groundtruth.json")
    ap.add_argument("--index", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N queries")
    ap.add_argument("--channels", action="store_true",
                    help="also score dense-only and lexical-only, to show what each contributes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    queries = gt["queries"][: args.limit] if args.limit else gt["queries"]

    stats = idx.index_stats(args.index)
    print(f"Index: {stats['index_path']} · {stats['documents']} documents · "
          f"{stats['chunks']} chunks · model {stats['embed_model']}")
    print(f"Ground truth: {args.ground_truth} · {len(queries)} queries")

    results = [evaluate(queries, args.k, args.index, "hybrid")]
    for r in results:
        _print(r)

    if args.channels:
        for mode in ("dense", "lexical"):
            r = evaluate(queries, args.k, args.index, mode)
            results.append(r)
            _print(r)

    if args.out:
        Path(args.out).write_text(
            json.dumps({"index": stats, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
