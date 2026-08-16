"""
ingest.py
Build (or refresh) the document index for Athena's document mode.

    python ingest.py                      # index ./corpus into ./athena_index.db
    python ingest.py --corpus data/finanz # a different folder
    python ingest.py --rebuild            # discard and re-embed everything
    python ingest.py --stats              # show what is currently indexed

Ingestion is incremental by file content hash: unchanged files are skipped, so
adding one document to the folder costs one document's work, not a full rebuild.
Nothing here touches the network except the local Ollama server.
"""

from __future__ import annotations

import argparse
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from core.index import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    DEFAULT_EMBED_MODEL,
    build_index,
    index_stats,
)


def _print_stats(index_path: str | None = None) -> None:
    # The path must be threaded through: without it this reported on the DEFAULT
    # index while --index built a different one, so a scale-corpus ingest printed
    # the demo corpus's document types and looked like it had indexed the wrong
    # files. A summary that describes a different database than the one you just
    # wrote is worse than no summary.
    s = index_stats(index_path)
    print(f"Index:        {s['index_path']}")
    print(f"Model:        {s['embed_model']}")
    print(f"Documents:    {s['documents']}")
    print(f"Chunks:       {s['chunks']} ({s['embedded_chunks']} embedded)")
    if s["years"]:
        print(f"Years:        {', '.join(y for y in s['years'] if y)}")
    if s["doc_types"]:
        print("Types:")
        for t in s["doc_types"]:
            print(f"  {t['doc_type'] or '(none)':<16} {t['n']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Athena's document index.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS_DIR, help="folder of documents")
    ap.add_argument("--index", default=None, help="index database path")
    ap.add_argument("--model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")
    ap.add_argument("--batch", type=int, default=32, help="embedding batch size")
    ap.add_argument("--rebuild", action="store_true", help="re-embed everything")
    ap.add_argument("--stats", action="store_true", help="show index contents and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.stats:
        _print_stats(args.index)
        return 0

    t0 = time.time()
    stats = build_index(
        corpus_dir=args.corpus,
        index_path=args.index,
        embed_model=args.model,
        batch_size=args.batch,
        rebuild=args.rebuild,
        progress=None if args.quiet else (lambda m: print(f"  {m}", flush=True)),
    )
    wall = time.time() - t0

    print("\n-- Ingest complete --")
    print(f"Files seen:     {stats['files_seen']}")
    print(f"Files indexed:  {stats['files_indexed']}")
    print(f"Files skipped:  {stats['files_skipped']} (unchanged)")
    print(f"Chunks stored:  {stats['chunks']}")
    print(f"Notices:        {stats['notices']}  (scans, row caps, formula misses)")
    print(f"Parse errors:   {stats['errors']}")
    print(f"Embedding time: {stats['embed_seconds']:.1f}s of {wall:.1f}s total")
    print(f"Model:          {stats['model']}")
    print()
    _print_stats(args.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
