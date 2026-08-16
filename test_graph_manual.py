"""
test_graph_manual.py
Validates the graph architecture end-to-end before building the UI.

Run this after completing Phase 1 (stub nodes) to confirm:
  1. Graph routes correctly (supervisor → researcher → writer → review)
  2. Graph interrupts at the review node
  3. Approval ends the graph cleanly
  4. Feedback triggers the revision loop (writer re-runs, interrupt again)

Usage:
    python test_graph_manual.py

Set GROQ_API_KEY in .env for better quality, or leave blank for Ollama.
"""

import sys

# Windows consoles default to cp1252, which can't print ─ or ✅
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

from core.graph import build_graph, make_initial_state


def separator(label: str):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")


def run_all_tests():
    graph = build_graph()

    # ── Test 1: Routing + interrupt ───────────────────────────────────────────
    separator("Test 1: Routing + Interrupt")

    config_1 = {"configurable": {"thread_id": "test-001"}}
    state = make_initial_state("The history of the Roman Empire")

    print("Running graph until interrupt...")
    result = graph.invoke(state, config=config_1)

    snapshot = graph.get_state(config_1)
    assert snapshot.next == ("review",), (
        f"Expected graph to pause at ('review',), got: {snapshot.next}"
    )
    assert result.get("draft_report"), "Expected a non-empty draft report"
    assert result.get("search_results"), "Expected non-empty search results"

    word_count = len(result["draft_report"].split())
    print(f"✅ Graph interrupted at: {snapshot.next}")
    print(f"✅ Draft produced: {word_count} words")
    print(f"✅ Search results: {len(result['search_results'])} item(s)")
    print(f"\nDraft preview (first 200 chars):\n{result['draft_report'][:200]}...")

    # ── Test 2: Approval ends the graph ──────────────────────────────────────
    separator("Test 2: Approval → Graph Ends")

    print("Sending 'approve'...")
    graph.invoke(Command(resume="approve"), config=config_1)

    final_snapshot = graph.get_state(config_1)
    assert not final_snapshot.next, (
        f"Expected graph to be finished, got next={final_snapshot.next}"
    )
    print("✅ Graph finished cleanly after approval")
    print(f"✅ Final iterations: {final_snapshot.values.get('iterations')}")

    # ── Test 3: Revision loop ─────────────────────────────────────────────────
    separator("Test 3: Feedback → Revision Loop")

    config_2 = {"configurable": {"thread_id": "test-002"}}
    state_2 = make_initial_state("The economic impact of artificial intelligence")

    print("Running graph until first interrupt...")
    first_result = graph.invoke(state_2, config=config_2)
    first_draft = first_result.get("draft_report", "")

    print(f"✅ First draft: {len(first_draft.split())} words")
    print("Sending feedback...")

    graph.invoke(
        Command(resume="Add more specific economic statistics and cite the Analysis section more clearly."),
        config=config_2,
    )

    revision_snapshot = graph.get_state(config_2)
    assert revision_snapshot.next == ("review",), (
        f"Expected graph to interrupt again at review, got: {revision_snapshot.next}"
    )

    revised_draft = revision_snapshot.values.get("draft_report", "")
    assert revised_draft != first_draft, "Revised draft should differ from original"
    print(f"✅ Revised draft ready: {len(revised_draft.split())} words")
    print(f"✅ Graph interrupted again at: {revision_snapshot.next}")
    print("✅ Revision loop confirmed working")

    # ── Approve the revision ──────────────────────────────────────────────────
    graph.invoke(Command(resume="approve"), config=config_2)
    end_snapshot = graph.get_state(config_2)
    assert not end_snapshot.next
    print("✅ Revision approved — graph ended cleanly")

    separator("All 3 tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
