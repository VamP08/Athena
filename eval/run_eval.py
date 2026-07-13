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
    else:
        judge_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
        judge_client = ollama_client
        print(f"Judge: local Ollama {judge_model} (set GROQ_API_KEY for a stronger judge)")

    # Generous max_tokens: reasoning models (qwen3.5, gpt-oss) spend tokens on
    # thinking before the JSON verdict — the instructor default truncates them.
    judge = llm_factory(judge_model, provider="openai", client=judge_client, max_tokens=6000)
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
            result["answer_relevancy"] = (
                await answer_relevancy.ascore(user_input=row["question"], response=row["answer"])
            ).value
            result["faithfulness"] = (
                await faithfulness.ascore(
                    user_input=row["question"],
                    response=row["answer"],
                    retrieved_contexts=row["contexts"],
                )
            ).value
            result["context_precision"] = (
                await context_precision.ascore(
                    user_input=row["question"],
                    reference=row["reference"],
                    retrieved_contexts=row["contexts"],
                )
            ).value
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
    output_path = Path(__file__).parent / "results.csv"
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
