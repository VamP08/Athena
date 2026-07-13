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


def get_llm(temperature: float = 0):
    """
    Returns the appropriate ChatModel.

    Args:
        temperature: Sampling temperature. 0 for deterministic routing/writing.

    Returns:
        A LangChain BaseChatModel instance.
    """
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=temperature,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:9b"),
        temperature=temperature,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    )
