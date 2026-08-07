"""
core/llm.py
Single function that returns the right LLM based on available credentials.

Strategy:
  - GROQ_API_KEY set  → Groq openai/gpt-oss-120b   (production, prompt tuning)
  - No key            → Ollama qwen3.5:9b           (local plumbing, free, offline)

OpenAI intentionally excluded — this project is zero-cost by design.

Groq model note (checked 2026-07): llama-3.3-70b-versatile is deprecated on free
tiers with shutdown 2026-08-16; Groq's recommended replacement is
openai/gpt-oss-120b (production, tool-use capable, 1K req/day free).

Local model note: num_ctx defaults to 16384. Ollama's own default is 2048, which
truncates research context badly; 8192 was set in July as the fix for that and
was never revisited against what the model can actually do — qwen3.5:9b supports
262,144 tokens, so the project had been running at 3% of its context capability.

Why 16384 and not more (measured 2026-08-03, RTX 4060 Laptop 8 GB):

  num_ctx  12 passages (11,791 chars)   VRAM        on GPU   time
  8192     EMPTY REPORT  <-- failure    7230/8188   100%      75s
  16384    553 words, correct           6375/8188    84%     181s
  32768    439 words, correct           6393/8188    76%     326s

8192 is the ONLY setting that stays entirely on the GPU, and it is also the only
setting that fails — so "keep it fully on GPU" would have selected the broken
configuration. 16384 is the smallest window that removes the failure; 32768 adds
no capability in this range while costing ~67% more latency.

Why this matters beyond an empty report: when a prompt exceeds num_ctx, Ollama
truncates from the FRONT. The system prompt sits at the front, so the first thing
discarded is the instruction block — including "Do not include ANY claim not
directly supported by the research provided". The model keeps partial evidence
and loses the rule forbidding invention around it, which is a hallucination
mechanism, not merely a formatting problem. Raising the window is therefore an
accuracy fix, and the ~1.8x latency is the price.

Local reasoning note (measured 2026-08-03, this machine): qwen3.5:9b emits a
thinking trace before answering. Disabling it is worth ~71x on latency (a
two-sentence German summary: 126.7s -> 1.8s), which is tempting — but latency is
the wrong thing to optimise here, and measuring accuracy reversed the decision.

On the 7-case document-mode grounding set (eval/run_grounding_eval.py):

                        reasoning ON      reasoning OFF
  fabricated figures    0                 5   (across 2 cases)
  misattributed         0/6               2/6
  answer accuracy       5/6               6/6
  seconds per case      208s              48s

With reasoning off the model produced figures — 10.355.000, 631.000, 760.000,
912.000, 8.489,51 — that appear in NO retrieved passage. In a financial report
an invented number is the worst possible failure: it is authoritative-looking
and silently wrong. Reasoning off did answer one extra case, but its single
failure mode under reasoning ON was an OMISSION, which is visible and safe.

Reasoning is therefore ON by default. ATHENA_OLLAMA_REASONING=false trades
grounding accuracy for ~4x speed and should be used only for plumbing work and
UI development, never for a demo or an evaluation run.

Caveat on the evidence: n=6 answerable cases. The one-case answer-accuracy
difference is well within noise; the 5-fabrications-to-0 gap is the finding that
actually drove the decision.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def resolve_backend() -> tuple[str, str]:
    """
    Resolve the active (backend, model) pair.

    ATHENA_LLM_BACKEND: auto (default) | groq | ollama
      auto   — Groq if GROQ_API_KEY is set, else Ollama
      groq   — force Groq (falls back to Ollama if no key)
      ollama — force local Ollama even when a Groq key exists

    Read at every LLM construction, so a runtime switch (e.g. from the UI)
    takes effect on the next node execution. Process-wide by design — this is
    an operator control, not a per-session setting.
    """
    forced = os.getenv("ATHENA_LLM_BACKEND", "auto").lower()
    has_groq = bool(os.getenv("GROQ_API_KEY"))

    if forced == "ollama" or not has_groq:
        return "ollama", os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    return "groq", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def get_llm(temperature: float = 0):
    """
    Returns the appropriate ChatModel.

    Args:
        temperature: Sampling temperature. 0 for deterministic routing/writing.

    Returns:
        A LangChain BaseChatModel instance.
    """
    backend, model = resolve_backend()

    if backend == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            temperature=temperature,
        )

    from langchain_ollama import ChatOllama

    kwargs = {}
    # In containers Ollama is a sibling service, not localhost
    if os.getenv("OLLAMA_BASE_URL"):
        kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL")

    # See "Local reasoning note" above. ON by default: disabling it produced
    # fabricated figures in the grounding eval. Only an explicit opt-out is
    # honoured, and it costs accuracy.
    if os.getenv("ATHENA_OLLAMA_REASONING", "true").lower() == "false":
        kwargs["reasoning"] = False

    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
        **kwargs,
    )
