"""
Tavily web search tool wrapper.

Provides async and sync helpers that query the Tavily API and return results
as typed ``SourceItem`` objects — never raw dicts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from tavily import TavilyClient

from schemas.report_schema import SourceItem

logger = logging.getLogger(__name__)


def _get_client() -> TavilyClient:
    """Instantiate a TavilyClient using the TAVILY_API_KEY environment variable.

    Raises:
        ValueError: If TAVILY_API_KEY is not set.
    """
    api_key: Optional[str] = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")
    return TavilyClient(api_key=api_key)


async def search(query: str, max_results: int = 5) -> list[SourceItem]:
    """Run a Tavily web search and return structured SourceItem results.

    The search is executed in a thread-pool executor so it does not block the
    event loop (the underlying Tavily SDK is synchronous).

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of ``SourceItem`` objects.  Returns an empty list on any error.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _search_sync_inner, query, max_results)


def search_sync(query: str, max_results: int = 5) -> list[SourceItem]:
    """Synchronous wrapper around the Tavily search for non-async contexts.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A list of ``SourceItem`` objects.  Returns an empty list on any error.
    """
    return _search_sync_inner(query, max_results)


def _search_sync_inner(query: str, max_results: int) -> list[SourceItem]:
    """Internal synchronous implementation shared by both public entry points.

    Args:
        query:       The search query string.
        max_results: Maximum number of results to return.

    Returns:
        A list of ``SourceItem`` objects.
    """
    try:
        client = _get_client()
        response = client.search(query=query, max_results=max_results)
        results: list[SourceItem] = []
        for item in response.get("results", []):
            results.append(
                SourceItem(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                )
            )
        logger.info("Tavily search returned %d results for query: %r", len(results), query)
        return results
    except ValueError as exc:
        logger.error("Tavily configuration error: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.error("Tavily search failed for query %r: %s", query, exc)
        return []
