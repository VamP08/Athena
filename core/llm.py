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

Local model note: num_ctx=8192 is critical; Ollama's default 2048 truncates
research context.
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

    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
        **kwargs,
    )
