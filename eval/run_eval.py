"""
eval/run_eval.py
Runs the full Athena graph on each golden test case, auto-approves the draft,
and scores the outputs with Ragas (0.4 class-based metrics API).

Metrics:
  answer_relevancy   — is the report on-topic?             target > 0.85
  faithfulness       — claims grounded in research?        target > 0.80
  context_precision  — research actually used in report?   target > 0.75

Zero-OpenAI-cost by construction:
  Judge LLM    — Groq (openai/gpt-oss-120b) if GROQ_API_KEY is set,
                 else local Ollama. Both via their OpenAI-compatible endpoints.
  Embeddings   — always local Ollama (nomic-embed-text); Groq has no embeddings.
                 Ragas silently defaults to paid OpenAI if you omit either — we
                 always pass both explicitly.

Usage:
    python eval/run_eval.py

Output:
    Scores to stdout (incl. README-ready markdown table) + eval/results.csv.
"""

import asyncio
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

import eval._ragas_compat  # noqa: F401  — must precede any ragas import

from langgraph.types import Command
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

from core.graph import build_graph, make_initial_state
from eval.test_dataset import TEST_CASES

TARGETS = {
    "answer_relevancy": 0.85,
    "faithfulness": 0.80,
    "context_precision": 0.75,
}


def _build_judge_and_embeddings():
    """Judge LLM: Groq if key present, else local Ollama. Embeddings: always Ollama."""
    ollama_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    embeddings = OpenAIEmbeddings(
        client=ollama_client,
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        judge_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        judge_client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        print(f"Judge: Groq {judge_model} · Embeddings: local Ollama")
        # Groq free tier caps gpt-oss at 8K tokens/minute PER REQUEST budget
        # (prompt + max_tokens counted together). Low reasoning effort + a small
        # completion budget keeps every judge call under the ceiling.
        # 3500 completion + ~2.5-4K prompt stays under the 8K TPM per-request
        # budget; 2500 proved too tight for faithfulness verdicts on ~500-word
        # drafts. Mind the 200K tokens-per-DAY cap when scheduling full runs.
        judge = llm_factory(
            judge_model, provider="openai", client=judge_client,
            max_tokens=3500, reasoning_effort="low",
        )
    else:
        judge_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        print(f"Judge: local Ollama {judge_model} (set GROQ_API_KEY for a stronger judge)")
        # Local reasoning models think before the JSON verdict — needs headroom.
        judge = llm_factory(judge_model, provider="openai", client=ollama_client, max_tokens=6000)

    return judge, embeddings


def run_case(graph, case: dict, case_num: int, total: int) -> dict:
    """Run one golden case through the full graph; return Ragas inputs."""
    topic = case["topic"]
    safe_id = topic[:30].replace(" ", "-").lower()
    config = {"configurable": {"thread_id": f"eval-{safe_id}"}}

    print(f"  [{case_num}/{total}] {topic}")

    graph.invoke(make_initial_state(topic), config=config)
    mid_state = graph.get_state(config)
    if mid_state.next != ("review",):
        print(f"    ⚠️  Graph did not interrupt at review — next={mid_state.next}")

    # Auto-approve — we score what the system produces unsupervised
    graph.invoke(Command(resume="approve"), config=config)
    final_state = graph.get_state(config)

    draft = final_state.values.get("draft_report", "")
    contexts = final_state.values.get("search_results", []) or ["No research gathered"]
    print(f"    Draft: {len(draft.split())} words | Research chunks: {len(contexts)}")

    return {
        "question": topic,
        "answer": draft,
        "contexts": contexts,
        "reference": ". ".join(case["expected_facts"]),
    }


_PACE_SECONDS = 20 if os.getenv("GROQ_API_KEY") else 0  # respect the free-tier TPM window


async def _score(metric_call):
    """One metric call with pacing + a single retry on rate limiting."""
    try:
        result = await metric_call()
    except Exception as e:
        if "rate_limit" in str(e) or "429" in str(e):
            print("    Rate limited — waiting 65s and retrying once...")
            time.sleep(65)
            result = await metric_call()
        else:
            raise
    if _PACE_SECONDS:
        time.sleep(_PACE_SECONDS)
    return result.value


async def score_rows(rows: list[dict]) -> list[dict]:
    judge, embeddings = _build_judge_and_embeddings()

    answer_relevancy = AnswerRelevancy(llm=judge, embeddings=embeddings)
    faithfulness = Faithfulness(llm=judge)
    context_precision = ContextPrecision(llm=judge)

    scored = []
    for i, row in enumerate(rows, 1):
        print(f"  Scoring [{i}/{len(rows)}] {row['question'][:50]}...")
        result = dict(row)
        try:
            result["answer_relevancy"] = await _score(
                lambda: answer_relevancy.ascore(user_input=row["question"], response=row["answer"])
            )
            result["faithfulness"] = await _score(
                lambda: faithfulness.ascore(
                    user_input=row["question"],
                    response=row["answer"],
                    retrieved_contexts=row["contexts"],
                )
            )
            result["context_precision"] = await _score(
                lambda: context_precision.ascore(
                    user_input=row["question"],
                    reference=row["reference"],
                    retrieved_contexts=row["contexts"],
                )
            )
        except Exception as e:
            print(f"    ❌ Scoring failed: {e}")
            continue
        scored.append(result)
    return scored


def main():
    print("=" * 60)
    print("Athena — Ragas Evaluation Suite")
    print("=" * 60)

    print("\nBuilding graph...")
    graph = build_graph()

    print(f"\nRunning {len(TEST_CASES)} golden cases through the full graph...\n")
    rows = []
    for i, case in enumerate(TEST_CASES, 1):
        try:
            rows.append(run_case(graph, case, i, len(TEST_CASES)))
        except Exception as e:
            print(f"    ❌ Case failed: {e}")

    if not rows:
        print("\n❌ All cases failed — cannot run evaluation.")
        sys.exit(1)

    print(f"\nScoring {len(rows)} reports (LLM-as-judge)...\n")
    scored = asyncio.run(score_rows(rows))

    if not scored:
        print("\n❌ Scoring produced no results.")
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ragas Evaluation Results")
    print("=" * 60)

    means = {m: sum(r[m] for r in scored) / len(scored) for m in TARGETS}
    for metric, target in TARGETS.items():
        status = "✅" if means[metric] >= target else "❌"
        print(f"  {status} {metric:<22} {means[metric]:.3f}  (target > {target})")

    print("\nPer-case breakdown:")
    for r in scored:
        print(
            f"  {r['question'][:45]:<47}"
            f" AR {r['answer_relevancy']:.2f} | F {r['faithfulness']:.2f}"
            f" | CP {r['context_precision']:.2f}"
        )

    # ── CSV artifact ──────────────────────────────────────────────────────────
    # A partial run must never clobber the last complete results file.
    filename = "results.csv" if len(scored) == len(TEST_CASES) else "results_partial.csv"
    if filename != "results.csv":
        print(f"\n⚠️  Only {len(scored)}/{len(TEST_CASES)} cases scored — "
              f"writing {filename}; results.csv left untouched.")
    output_path = Path(__file__).parent / filename
    fields = ["question", *TARGETS.keys()]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    print(f"\n✅ Full results saved to {output_path}")

    print("\nREADME-ready table:")
    print("| Metric | Score | Target |")
    print("|---|---|---|")
    for metric, target in TARGETS.items():
        print(f"| {metric} | {means[metric]:.2f} | > {target} |")


if __name__ == "__main__":
    main()
