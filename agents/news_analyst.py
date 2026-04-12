"""
News Analyst agent.

Searches the web for recent developments related to the user query, then uses
an LLM to synthesise the raw source content into typed ``DevelopmentItem``
objects.  Retrieved documents are stored in ChromaDB for downstream RAG use.

Agent pattern (shared with all sub-agents):
  1. Generate targeted search queries with the LLM
  2. Run searches in parallel (asyncio.gather)
  3. Deduplicate results by URL
  4. LLM structures raw snippets into typed output
  5. Store raw sources in ChromaDB
  6. Return updated AgentState with findings appended
  7. All errors are caught and appended to state["errors"] — never raised
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from observability.langfuse_tracer import trace_agent
from schemas.report_schema import AgentState, DevelopmentItem, NewsFindings, SourceItem
from tools.vector_retriever import VectorRetriever
from tools.web_search import search as tavily_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal LLM output schemas (never leak outside this module)
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "news_analyst"
_DEFAULT_MODEL = "gemini-2.5-flash"


class _SearchQueriesOutput(BaseModel):
    """Structured output for the query-generation step."""

    queries: list[str] = Field(
        description="Exactly 3 targeted web search queries",
        min_length=1,
    )


class _RawDevItem(BaseModel):
    """Single development item as produced by the LLM before source resolution."""

    title: str = Field(description="Headline or short title")
    summary: str = Field(description="2-4 sentence plain-language summary")
    date: str = Field(description="Publication or event date; use 'Unknown' if not found")
    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs from the provided source list that back this item",
    )


class _NewsLLMOutput(BaseModel):
    """Full structured output for the summarisation step."""

    items: list[_RawDevItem] = Field(
        default_factory=list,
        description="List of recent development items found in the sources",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_llm() -> ChatGoogleGenerativeAI:
    """Construct the primary LLM from environment configuration.

    Returns:
        A ``ChatGoogleGenerativeAI`` instance configured for Gemini 2.5 Flash.

    Raises:
        ValueError: If ``GOOGLE_API_KEY`` is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    model = os.getenv("GEMINI_MODEL", _DEFAULT_MODEL)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,
    )


def _deduplicate(sources: list[SourceItem]) -> list[SourceItem]:
    """Return a URL-deduplicated copy of a SourceItem list (order-preserving)."""
    seen: set[str] = set()
    out: list[SourceItem] = []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            out.append(s)
    return out


def _resolve_sources(
    source_urls: list[str],
    url_map: dict[str, SourceItem],
    fallback: list[SourceItem],
) -> list[SourceItem]:
    """Map LLM-provided URLs back to SourceItem objects.

    Falls back to the first 3 retrieved sources if no URLs match.
    """
    resolved = [url_map[u] for u in source_urls if u in url_map]
    return resolved if resolved else fallback[:3]


# ---------------------------------------------------------------------------
# Public agent entry point
# ---------------------------------------------------------------------------


async def run(state: AgentState) -> AgentState:
    """Execute the News Analyst agent and return an updated AgentState.

    Reads ``state["query"]``, performs parallel web searches, stores results in
    ChromaDB, and appends a ``NewsFindings`` object to ``state["news_findings"]``.

    Args:
        state: Current LangGraph shared state.  Must contain ``"query"``.

    Returns:
        Updated state with ``news_findings`` and ``all_sources`` populated.
        Any exception is caught and appended to ``state["errors"]``.
    """
    query: str = state.get("query", "")
    errors: list[str] = list(state.get("errors", []))

    with trace_agent("news_analyst", query) as trace:
        try:
            llm = _build_llm()

            # ------------------------------------------------------------------
            # Step 1 — generate 3 targeted search queries
            # ------------------------------------------------------------------
            query_span = trace.span(
                name="generate_search_queries",
                input={"query": query},
            )
            system_prompt = (
                "You are an expert news analyst specialising in competitive intelligence. "
                "Given a topic or company name, generate exactly 3 highly targeted web "
                "search queries that will surface the most recent and relevant news "
                "articles, press releases, and industry developments."
            )
            structured_llm: Any = llm.with_structured_output(_SearchQueriesOutput)
            queries_result: _SearchQueriesOutput = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"Topic: {query}\n\n"
                            "Generate 3 targeted search queries to find the latest news."
                        )
                    ),
                ]
            )
            search_queries = queries_result.queries[:3]
            query_span.end(output={"queries": search_queries})
            logger.info("Generated search queries: %s", search_queries)

            # ------------------------------------------------------------------
            # Step 2 — run all searches in parallel
            # ------------------------------------------------------------------
            search_span = trace.span(
                name="parallel_web_search",
                input={"queries": search_queries},
            )
            search_results: list[list[SourceItem]] = await asyncio.gather(
                *[tavily_search(q, max_results=5) for q in search_queries]
            )
            raw_sources = _deduplicate(
                [item for sublist in search_results for item in sublist]
            )
            search_span.end(output={"total_raw_sources": len(raw_sources)})
            logger.info("Collected %d unique sources", len(raw_sources))

            # ------------------------------------------------------------------
            # Step 3 — LLM structures snippets into DevelopmentItems
            # ------------------------------------------------------------------
            if raw_sources:
                summarise_span = trace.span(
                    name="summarise_findings",
                    input={"source_count": len(raw_sources)},
                )
                sources_text = "\n\n".join(
                    f"[{i + 1}] URL: {s.url}\nTitle: {s.title}\nSnippet: {s.snippet}"
                    for i, s in enumerate(raw_sources)
                )
                summarise_prompt = (
                    f"You are an expert news analyst. Analyse the following retrieved "
                    f"sources about '{query}' and extract the most significant recent "
                    f"developments. For each development, cite the source URLs from the "
                    f"list below that support it.\n\n"
                    f"SOURCES:\n{sources_text}\n\n"
                    f"Return up to 5 of the most relevant recent developments."
                )
                news_structured_llm: Any = llm.with_structured_output(_NewsLLMOutput)
                llm_output: _NewsLLMOutput = await news_structured_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "You are an expert news analyst. Extract structured "
                                "development items from the provided sources."
                            )
                        ),
                        HumanMessage(content=summarise_prompt),
                    ]
                )
                url_map = {s.url: s for s in raw_sources}
                dev_items: list[DevelopmentItem] = [
                    DevelopmentItem(
                        title=raw.title,
                        summary=raw.summary,
                        date=raw.date,
                        sources=_resolve_sources(raw.source_urls, url_map, raw_sources),
                    )
                    for raw in llm_output.items
                ]
                summarise_span.end(output={"item_count": len(dev_items)})
            else:
                dev_items = []
                logger.warning("No sources retrieved for query: %r", query)

            # ------------------------------------------------------------------
            # Step 4 — store raw sources in ChromaDB
            # ------------------------------------------------------------------
            if raw_sources:
                try:
                    retriever = VectorRetriever(collection_name=_COLLECTION_NAME)
                    await retriever.add_documents(raw_sources)
                    logger.info("Stored %d documents in ChromaDB", len(raw_sources))
                except Exception as ve:  # noqa: BLE001
                    logger.warning("ChromaDB storage failed: %s", ve)

            # ------------------------------------------------------------------
            # Step 5 — build output and return updated state
            # ------------------------------------------------------------------
            findings = NewsFindings(items=dev_items, raw_sources=raw_sources)
            return {
                **state,
                "news_findings": [findings],
                "all_sources": raw_sources,
                "errors": errors,
            }

        except Exception as exc:  # noqa: BLE001
            msg = f"news_analyst error: {exc}"
            logger.error(msg)
            errors.append(msg)
            return {
                **state,
                "news_findings": [NewsFindings()],
                "errors": errors,
            }
