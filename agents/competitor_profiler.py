"""
Competitor Profiler agent.

Maps the competitive landscape around the user query by searching for main
competitors, product comparisons, market positioning, and pricing/feature data.
Follows the identical pattern established in Session 2 for all sub-agents.

Agent pattern:
  1. Generate 4 search queries (competitors / comparisons / positioning / pricing)
  2. Run all 4 searches in parallel (asyncio.gather)
  3. Deduplicate results by URL
  4. LLM structures raw snippets into typed CompetitorItem output
  5. Store raw sources in ChromaDB collection "competitor_profiler"
  6. Return updated AgentState with competitor_findings appended
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
from schemas.report_schema import (
    AgentState,
    CompetitorFindings,
    CompetitorItem,
    SourceItem,
)
from tools.vector_retriever import VectorRetriever
from tools.web_search import search as tavily_search

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "competitor_profiler"
_DEFAULT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Internal LLM output schemas (never leak outside this module)
# ---------------------------------------------------------------------------


class _SearchQueriesOutput(BaseModel):
    """Structured output for the query-generation step."""

    queries: list[str] = Field(
        description="Exactly 4 targeted competitive-intelligence search queries",
        min_length=1,
    )


class _RawCompetitorItem(BaseModel):
    """Single competitor profile as produced by the LLM before source resolution."""

    name: str = Field(description="Company or organisation name")
    description: str = Field(description="Brief overview of what they do and their market")
    strengths: list[str] = Field(
        default_factory=list,
        description="Key competitive advantages and differentiators",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Known limitations, vulnerabilities, or gaps",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs from the provided source list that back this profile",
    )


class _CompetitorLLMOutput(BaseModel):
    """Full structured output for the competitor-profiling step."""

    items: list[_RawCompetitorItem] = Field(
        default_factory=list,
        description="List of competitor profiles extracted from the sources",
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
    """Execute the Competitor Profiler agent and return an updated AgentState.

    Reads ``state["query"]``, performs 4 parallel web searches targeting the
    competitive landscape, stores results in ChromaDB, and appends a
    ``CompetitorFindings`` object to ``state["competitor_findings"]``.

    If no clear competitors are found the agent returns an empty
    ``CompetitorFindings`` without raising an error.

    Args:
        state: Current LangGraph shared state.  Must contain ``"query"``.

    Returns:
        Updated state with ``competitor_findings`` and ``all_sources`` populated.
        Any exception is caught and appended to ``state["errors"]``.
    """
    query: str = state.get("query", "")
    errors: list[str] = list(state.get("errors", []))

    with trace_agent("competitor_profiler", query) as trace:
        try:
            llm = _build_llm()

            # ------------------------------------------------------------------
            # Step 1 — generate 4 targeted competitive-intelligence search queries
            # ------------------------------------------------------------------
            query_span = trace.span(
                name="generate_search_queries",
                input={"query": query},
            )
            system_prompt = (
                "You are an expert competitive intelligence analyst specialising in "
                "mapping market landscapes. Given a topic or company name, generate "
                "exactly 4 targeted web search queries designed to surface: "
                "(1) the main direct competitors, "
                "(2) product feature and capability comparisons, "
                "(3) market positioning and share data, "
                "(4) pricing models and go-to-market strategies."
            )
            structured_llm: Any = llm.with_structured_output(_SearchQueriesOutput)
            queries_result: _SearchQueriesOutput = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"Topic: {query}\n\n"
                            "Generate 4 targeted search queries to map the competitive "
                            "landscape: competitors, product comparisons, market "
                            "positioning, and pricing/features."
                        )
                    ),
                ]
            )
            search_queries = queries_result.queries[:4]
            query_span.end(output={"queries": search_queries})
            logger.info("Generated competitor search queries: %s", search_queries)

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
            logger.info("Collected %d unique competitor sources", len(raw_sources))

            # ------------------------------------------------------------------
            # Step 3 — LLM structures snippets into CompetitorItems
            # ------------------------------------------------------------------
            if raw_sources:
                profile_span = trace.span(
                    name="profile_competitors",
                    input={"source_count": len(raw_sources)},
                )
                sources_text = "\n\n".join(
                    f"[{i + 1}] URL: {s.url}\nTitle: {s.title}\nSnippet: {s.snippet}"
                    for i, s in enumerate(raw_sources)
                )
                profile_prompt = (
                    f"You are an expert competitive intelligence analyst. Analyse the "
                    f"following retrieved sources about '{query}' and identify the key "
                    f"competitors and comparable organisations. For each competitor, "
                    f"extract their name, a brief description, their main strengths, "
                    f"known weaknesses, and cite the source URLs that support your "
                    f"assessment. If no clear competitors are identifiable from the "
                    f"sources, return an empty list.\n\n"
                    f"SOURCES:\n{sources_text}\n\n"
                    f"Return up to 6 competitor profiles."
                )
                competitor_structured_llm: Any = llm.with_structured_output(
                    _CompetitorLLMOutput
                )
                llm_output: _CompetitorLLMOutput = await competitor_structured_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "You are an expert competitive intelligence analyst. "
                                "Extract structured competitor profiles from the "
                                "provided sources."
                            )
                        ),
                        HumanMessage(content=profile_prompt),
                    ]
                )
                url_map = {s.url: s for s in raw_sources}
                competitor_items: list[CompetitorItem] = [
                    CompetitorItem(
                        name=raw.name,
                        description=raw.description,
                        strengths=raw.strengths,
                        weaknesses=raw.weaknesses,
                        sources=_resolve_sources(raw.source_urls, url_map, raw_sources),
                    )
                    for raw in llm_output.items
                ]
                profile_span.end(output={"competitor_count": len(competitor_items)})

                if not competitor_items:
                    logger.info(
                        "No clear competitors identified for query: %r", query
                    )
            else:
                competitor_items = []
                logger.warning("No competitor sources retrieved for query: %r", query)

            # ------------------------------------------------------------------
            # Step 4 — store raw sources in ChromaDB
            # ------------------------------------------------------------------
            if raw_sources:
                try:
                    retriever = VectorRetriever(collection_name=_COLLECTION_NAME)
                    await retriever.add_documents(raw_sources)
                    logger.info(
                        "Stored %d competitor documents in ChromaDB", len(raw_sources)
                    )
                except Exception as ve:  # noqa: BLE001
                    logger.warning("ChromaDB storage failed: %s", ve)

            # ------------------------------------------------------------------
            # Step 5 — build output and return updated state
            # ------------------------------------------------------------------
            findings = CompetitorFindings(
                items=competitor_items, raw_sources=raw_sources
            )
            return {
                **state,
                "competitor_findings": [findings],
                "all_sources": raw_sources,
                "errors": errors,
            }

        except Exception as exc:  # noqa: BLE001
            msg = f"competitor_profiler error: {exc}"
            logger.error(msg)
            errors.append(msg)
            return {
                **state,
                "competitor_findings": [CompetitorFindings()],
                "errors": errors,
            }
