"""
tests/test_tools.py
Unit tests for the ddgs-backed web_search tool.

All DDGS calls are mocked — no network.
"""

from unittest.mock import patch

import pytest
from ddgs.exceptions import DDGSException, RatelimitException

from core.tools import search_text, web_search

FAKE_RESULTS = [
    {"title": "Result One", "href": "https://example.com/1", "body": "First snippet."},
    {"title": "Result Two", "href": "https://example.com/2", "body": "Second snippet."},
]


@patch("core.tools.DDGS")
def test_search_text_returns_structured_results(mock_ddgs):
    mock_ddgs.return_value.text.return_value = FAKE_RESULTS

    results = search_text("test query", max_results=2)

    assert results == FAKE_RESULTS
    mock_ddgs.return_value.text.assert_called_once_with("test query", max_results=2)


@patch("core.tools.time.sleep")  # don't actually wait in tests
@patch("core.tools.DDGS")
def test_search_text_retries_on_ratelimit(mock_ddgs, mock_sleep):
    mock_ddgs.return_value.text.side_effect = [
        RatelimitException("slow down"),
        FAKE_RESULTS,
    ]

    results = search_text("test query")

    assert results == FAKE_RESULTS
    assert mock_ddgs.return_value.text.call_count == 2


@patch("core.tools.time.sleep")
@patch("core.tools.DDGS")
def test_search_text_raises_after_exhausted_retries(mock_ddgs, mock_sleep):
    mock_ddgs.return_value.text.side_effect = RatelimitException("always limited")

    with pytest.raises(DDGSException):
        search_text("test query")

    assert mock_ddgs.return_value.text.call_count == 3


@patch("core.tools.DDGS")
def test_web_search_tool_formats_results_with_urls(mock_ddgs):
    mock_ddgs.return_value.text.return_value = FAKE_RESULTS

    output = web_search.invoke({"query": "test query", "max_results": 2})

    assert "[1] Result One" in output
    assert "https://example.com/1" in output
    assert "[2] Result Two" in output
    assert "Second snippet." in output


@patch("core.tools.time.sleep")
@patch("core.tools.DDGS")
def test_web_search_tool_returns_error_string_not_exception(mock_ddgs, mock_sleep):
    """The agent loop feeds tool output back to the LLM — errors must be strings."""
    mock_ddgs.return_value.text.side_effect = DDGSException("engine down")

    output = web_search.invoke({"query": "test query"})

    assert output.startswith("Search error:")


@patch("core.tools.DDGS")
def test_web_search_tool_handles_empty_results(mock_ddgs):
    mock_ddgs.return_value.text.return_value = []

    output = web_search.invoke({"query": "obscure query"})

    assert "No results found" in output
