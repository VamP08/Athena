"""
mcp_servers/web_server.py
A standalone FastMCP 2.x server exposing web search and page fetch
as standardised MCP tools over streamable HTTP transport.

Run separately before launching the main app in MCP mode:
    python -m mcp_servers.web_server

Then in .env set:
    MCP_MODE=true
    MCP_WEB_URL=http://localhost:8001/mcp

Any MCP-compatible client can connect (Claude Desktop, Cursor, LangGraph).
This is the core resume differentiator — you are BUILDING an MCP server,
not just consuming one.

Transport note (2026-07): the MCP spec deprecated HTTP+SSE in protocol revision
2025-03-26; streamable HTTP is the standard. FastMCP 2.x serves it at /mcp.
FastMCP pinned to 2.14.x — 3.x is a separate, planned migration.
"""

import re
import sys
from pathlib import Path

import httpx
from fastmcp import FastMCP

# Allow running as `python -m mcp_servers.web_server` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tools import search_text

mcp = FastMCP("athena-web-search")


@mcp.tool()
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web and return structured results.

    Uses a multi-engine metasearch backend (ddgs) with retry/backoff —
    the same implementation the app uses in direct mode.

    Args:
        query: The search query string. Be specific for better results.
        max_results: Number of results to return. Between 1 and 10.

    Returns:
        List of result dicts, each with keys: 'title', 'body', 'href'.
    """
    return search_text(query, max_results=max_results)


@mcp.tool()
async def fetch_page(url: str) -> str:
    """
    Fetch the readable text content of a web page.

    Strips HTML tags and returns up to 4000 characters of clean text.
    Useful for reading full articles after finding them via search_web.

    Args:
        url: Full URL including https://. Must be a publicly accessible page.

    Returns:
        Plain text content of the page (first 4000 characters).
    """
    headers = {
        "User-Agent": "Athena-Research-Bot/1.0 (educational project)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", resp.text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


if __name__ == "__main__":
    print("Starting Athena Web Search MCP server on port 8001...")
    print("Connect at: http://localhost:8001/mcp  (streamable HTTP)")
    print("Stop with Ctrl+C\n")
    mcp.run(transport="http", host="0.0.0.0", port=8001)
