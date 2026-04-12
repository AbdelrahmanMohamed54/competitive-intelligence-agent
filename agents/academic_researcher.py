"""
Academic Researcher agent.

Searches for academic papers, technical reports, and patents related to the user
query, then uses an LLM to synthesise the raw source content into typed
``AcademicItem`` objects.  Follows the identical pattern established by the
News Analyst agent.

Agent pattern:
  1. Generate 3 search queries focused on papers / patents / technical reports
  2. Run all searches in parallel (asyncio.gather)
  3. Deduplicate results by URL
  4. LLM structures raw snippets into typed AcademicItem output
  5. Store raw sources in ChromaDB
  6. Return updated AgentState with academic_findings appended
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
    AcademicFindings,
    AcademicItem,
    AgentState,
    SourceItem,
)
from tools.vector_retriever import VectorRetriever
from tools.web_search import search as tavily_search

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "academic_researcher"
_DEFAULT_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Internal LLM output schemas
# ---------------------------------------------------------------------------


class _SearchQueriesOutput(BaseModel):
    """Structured output for the query-generation step."""

    queries: list[str] = Field(
        description="Exactly 3 targeted academic/technical search queries",
        min_length=1,
    )


class _RawAcademicItem(BaseModel):
    """Single academic item as produced by the LLM before source resolution."""

    title: str = Field(description="Full title of the paper or publication")
    summary: str = Field(
        description="2-4 sentence plain-language summary of the contribution"
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author names extracted from the source",
    )
    url: str = Field(
        description="Primary URL or DOI for the paper; use the most authoritative link"
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs from the provided source list that back this item",
    )


class _AcademicLLMOutput(BaseModel):
    """Full structured output for the summarisation step."""

    items: list[_RawAcademicItem] = Field(
        default_factory=list,
        description="List of academic items found in the sources",
    )


# ---------------------------------------------------------------------------
# Helpers (shared logic mirrors news_analyst.py)
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
    """Map LLM-provided URLs back to SourceItem objects."""
    resolved = [url_map[u] for u in source_urls if u in url_map]
    return resolved if resolved else fallback[:3]


# ---------------------------------------------------------------------------
# Public agent entry point
# ---------------------------------------------------------------------------


async def run(state: AgentState) -> AgentState:
    """Execute the Academic Researcher agent and return an updated AgentState.

    Reads ``state["query"]``, performs parallel web searches focused on academic
    and technical sources, stores results in ChromaDB, and appends an
    ``AcademicFindings`` object to ``state["academic_findings"]``.

    Args:
        state: Current LangGraph shared state.  Must contain ``"query"``.

    Returns:
        Updated state with ``academic_findings`` and ``all_sources`` populated.
        Any exception is caught and appended to ``state["errors"]``.
    """
    query: str = state.get("query", "")
    errors: list[str] = list(state.get("errors", []))

    with trace_agent("academic_researcher", query) as trace:
        try:
            llm = _build_llm()

            # ------------------------------------------------------------------
            # Step 1 — generate 3 academic-focused search queries
            # ------------------------------------------------------------------
            query_span = trace.span(
                name="generate_search_queries",
                input={"query": query},
            )
            system_prompt = (
                "You are an expert academic researcher specialising in competitive "
                "intelligence. Given a topic or company name, generate exactly 3 "
                "targeted search queries designed to surface academic papers, "
                "technical reports, patents, and peer-reviewed publications. "
                "Include terms like 'research paper', 'arxiv', 'IEEE', 'patent', "
                "or 'technical report' where appropriate."
            )
            structured_llm: Any = llm.with_structured_output(_SearchQueriesOutput)
            queries_result: _SearchQueriesOutput = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            f"Topic: {query}\n\n"
                            "Generate 3 targeted search queries to find academic "
                            "papers, patents, and technical publications."
                        )
                    ),
                ]
            )
            search_queries = queries_result.queries[:3]
            query_span.end(output={"queries": search_queries})
            logger.info("Generated academic search queries: %s", search_queries)

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
            logger.info("Collected %d unique academic sources", len(raw_sources))

            # ------------------------------------------------------------------
            # Step 3 — LLM structures snippets into AcademicItems
            # ------------------------------------------------------------------
            if raw_sources:
                summarise_span = trace.span(
                    name="summarise_academic_findings",
                    input={"source_count": len(raw_sources)},
                )
                sources_text = "\n\n".join(
                    f"[{i + 1}] URL: {s.url}\nTitle: {s.title}\nSnippet: {s.snippet}"
                    for i, s in enumerate(raw_sources)
                )
                summarise_prompt = (
                    f"You are an expert academic researcher. Analyse the following "
                    f"retrieved sources about '{query}' and identify the most "
                    f"significant academic papers, technical publications, and patents. "
                    f"For each item, extract: title, authors, a plain-language summary, "
                    f"and the primary URL. Cite the source URLs from the list below.\n\n"
                    f"SOURCES:\n{sources_text}\n\n"
                    f"Return up to 5 of the most relevant academic items."
                )
                academic_structured_llm: Any = llm.with_structured_output(
                    _AcademicLLMOutput
                )
                llm_output: _AcademicLLMOutput = await academic_structured_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "You are an expert academic researcher. Extract "
                                "structured academic items from the provided sources."
                            )
                        ),
                        HumanMessage(content=summarise_prompt),
                    ]
                )
                url_map = {s.url: s for s in raw_sources}
                academic_items: list[AcademicItem] = [
                    AcademicItem(
                        title=raw.title,
                        summary=raw.summary,
                        authors=raw.authors,
                        url=raw.url,
                        sources=_resolve_sources(raw.source_urls, url_map, raw_sources),
                    )
                    for raw in llm_output.items
                ]
                summarise_span.end(output={"item_count": len(academic_items)})
            else:
                academic_items = []
                logger.warning("No academic sources retrieved for query: %r", query)

            # ------------------------------------------------------------------
            # Step 4 — store raw sources in ChromaDB
            # ------------------------------------------------------------------
            if raw_sources:
                try:
                    retriever = VectorRetriever(collection_name=_COLLECTION_NAME)
                    await retriever.add_documents(raw_sources)
                    logger.info(
                        "Stored %d academic documents in ChromaDB", len(raw_sources)
                    )
                except Exception as ve:  # noqa: BLE001
                    logger.warning("ChromaDB storage failed: %s", ve)

            # ------------------------------------------------------------------
            # Step 5 — build output and return updated state
            # ------------------------------------------------------------------
            findings = AcademicFindings(
                items=academic_items, raw_sources=raw_sources
            )
            return {
                **state,
                "academic_findings": [findings],
                "all_sources": raw_sources,
                "errors": errors,
            }

        except Exception as exc:  # noqa: BLE001
            msg = f"academic_researcher error: {exc}"
            logger.error(msg)
            errors.append(msg)
            return {
                **state,
                "academic_findings": [AcademicFindings()],
                "errors": errors,
            }
