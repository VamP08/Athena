"""
tests/test_llm.py
Backend resolution: ATHENA_LLM_BACKEND override + GROQ_API_KEY auto-detection.
No LLM is constructed — resolve_backend is pure env logic.
"""

from core.llm import resolve_backend


def test_auto_prefers_groq_when_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.delenv("ATHENA_LLM_BACKEND", raising=False)

    assert resolve_backend() == ("groq", "openai/gpt-oss-120b")


def test_auto_falls_back_to_ollama_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")
    monkeypatch.delenv("ATHENA_LLM_BACKEND", raising=False)

    assert resolve_backend() == ("ollama", "qwen3.5:9b")


def test_forced_ollama_wins_even_with_groq_key(monkeypatch):
    """The UI's local-mode switch: privacy demo must beat the cloud default."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("ATHENA_LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")

    assert resolve_backend() == ("ollama", "qwen3.5:9b")


def test_forced_groq_without_key_degrades_to_ollama(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("ATHENA_LLM_BACKEND", "groq")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:9b")

    assert resolve_backend() == ("ollama", "qwen3.5:9b")
