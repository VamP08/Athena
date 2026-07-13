"""
core/tools.py
Web search built directly on the `ddgs` metasearch library.

Why not langchain-community's DuckDuckGoSearchRun?
  langchain-community was sunset and archived (June 2026). LangChain's official
  guidance for integrations without a partner package is to implement the tool
  directly in application code — which is what this module does. It also gives us
  structured results (title/href/body) instead of a flattened string, so the
  researcher can cite sources.

The same `search_text()` function backs both the direct LangChain tool below and
the FastMCP server in mcp_servers/web_server.py — one implementation, two protocols.
"""

import time

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException
from langchain_core.tools import tool

_MAX_RETRIES = 3
_BACKOFF_SECONDS = 2.0


def search_text(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a web search and return structured results.

    Uses ddgs' "auto" backend, which rotates across several search engines —
    more resilient to rate limiting than any single engine.

    Returns:
        List of dicts with keys 'title', 'href', 'body'.

    Raises:
        DDGSException: If all retries are exhausted.
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return DDGS().text(query, max_results=max_results)
        except RatelimitException as e:
            last_error = e
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        except DDGSException as e:
            last_error = e
            time.sleep(_BACKOFF_SECONDS)
    raise DDGSException(f"Search failed after {_MAX_RETRIES} attempts: {last_error}")


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results with titles, URLs, and snippets.

    Args:
        query: The search query. Be specific for better results.
        max_results: Number of results to return (1-10).
    """
    try:
        results = search_text(query, max_results=max_results)
    except DDGSException as e:
        return f"Search error: {e}"

    if not results:
        return f"No results found for: {query}"

    return "\n\n".join(
        f"[{i}] {r.get('title', 'Untitled')}\nURL: {r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, 1)
    )
