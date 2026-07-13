"""
mcp_servers/connect.py
Helper for connecting to the Athena MCP web server and retrieving
LangChain-compatible tool objects.

Requires the MCP server to be running:
    python -m mcp_servers.web_server

The returned tools are drop-in replacements for the direct web_search tool —
researcher_node uses them identically regardless of source.

API note (langchain-mcp-adapters >= 0.3): MultiServerMCPClient is instantiated
directly (it is NOT an async context manager — that API was removed in 0.1.0),
and get_tools() is a coroutine. Each tool invocation opens a fresh MCP session.
"""

import asyncio
import os
from typing import List

from langchain_core.tools import BaseTool

MCP_CONFIG = {
    "web": {
        "url": os.getenv("MCP_WEB_URL", "http://localhost:8001/mcp"),
        "transport": "streamable_http",
    }
}

_cached_tools: List[BaseTool] | None = None


def get_mcp_tools_sync() -> List[BaseTool]:
    """
    Synchronously fetch LangChain tools from the running MCP server.

    Caches the result — first call handshakes with the server;
    subsequent calls return the cached list immediately.

    Raises:
        ConnectionError: If the MCP server is not running or unreachable.
    """
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools

    async def _fetch():
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(MCP_CONFIG)
        return await client.get_tools()

    try:
        # Handle the case where an event loop is already running
        # (common in Streamlit and Jupyter environments)
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            tools = loop.run_until_complete(_fetch())
        except RuntimeError:
            # No running loop — safe to use asyncio.run()
            tools = asyncio.run(_fetch())

        _cached_tools = tools
        print(f"[MCP] Connected — {len(tools)} tool(s): {[t.name for t in tools]}")
        return tools

    except Exception as e:
        raise ConnectionError(
            f"Could not connect to MCP server at {MCP_CONFIG['web']['url']}.\n"
            f"Make sure web_server.py is running first:\n"
            f"  python -m mcp_servers.web_server\n"
            f"Original error: {e}"
        )
