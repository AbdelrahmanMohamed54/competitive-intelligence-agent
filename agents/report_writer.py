"""
Report Writer agent.

Synthesises the outputs of all three sub-agents (news analyst, academic researcher,
competitor profiler) into a single structured ``ReportSchema``.  This is the final
agent in the pipeline and is the most important — its output is what the user sees.

Execution steps:
  1. Retrieve additional RAG context from all three ChromaDB collections
  2. Build a unified context string from all agent findings
  3. Generate executive_summary via a single LLM call
  4. Pass-through the already-structured sections (developments, academic, competitors)
  5. Generate strategic_recommendations via a single LLM call
  6. Deduplicate all sources collected across every agent
  7. Assemble and validate a ReportSchema using model_validate
  8. Record generation_time_seconds from state["start_time"]
  9. Log an initial quality marker via score_output()

All failures are caught per-section; placeholders are written so the report is
always returned in a valid state.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from observability.langfuse_tracer import score_output, trace_agent
from schemas.report_schema import (
    AcademicFindings,
    AcademicItem,
    AgentState,
    CompetitorFindings,
    CompetitorItem,
    DevelopmentItem,
    NewsFindings,
    ReportSchema,
    SourceItem,
)
from tools.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"
_RAG_COLLECTIONS = ["news_analyst", "academic_researcher", "competitor_profiler"]


# ---------------------------------------------------------------------------
# Internal LLM output schemas
# ---------------------------------------------------------------------------


class _ExecutiveSummaryOutput(BaseModel):
    """Structured output for the executive-summary generation step."""

    executive_summary: str = Field(
        description=(
            "A comprehensive 3-5 paragraph executive summary that synthesises "
            "all research findings into a coherent strategic narrative"
        )
    )


class _RecommendationsOutput(BaseModel):
    """Structured output for the strategic-recommendations generation step."""

    recommendations: list[str] = Field(
        description=(
            "5-7 specific, actionable strategic recommendations derived from "
            "the news, academic, and competitive intelligence findings"
        )
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


def _deduplicate_sources(sources: list[SourceItem]) -> list[SourceItem]:
    """Return a URL-deduplicated copy of a SourceItem list (order-preserving)."""
    seen: set[str] = set()
    out: list[SourceItem] = []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            out.append(s)
    return out


def _flatten_news(findings_list: list[NewsFindings]) -> list[DevelopmentItem]:
    """Flatten a list of NewsFindings into a single list of DevelopmentItems."""
    items: list[DevelopmentItem] = []
    for f in findings_list:
        items.extend(f.items)
    return items


def _flatten_academic(findings_list: list[AcademicFindings]) -> list[AcademicItem]:
    """Flatten a list of AcademicFindings into a single list of AcademicItems."""
    items: list[AcademicItem] = []
    for f in findings_list:
        items.extend(f.items)
    return items


def _flatten_competitors(
    findings_list: list[CompetitorFindings],
) -> list[CompetitorItem]:
    """Flatten a list of CompetitorFindings into a single list of CompetitorItems."""
    items: list[CompetitorItem] = []
    for f in findings_list:
        items.extend(f.items)
    return items


def _collect_all_sources(state: AgentState) -> list[SourceItem]:
    """Gather every SourceItem from every agent findings in state."""
    sources: list[SourceItem] = list(state.get("all_sources", []))
    for nf in state.get("news_findings", []):
        sources.extend(nf.raw_sources)
    for af in state.get("academic_findings", []):
        sources.extend(af.raw_sources)
    for cf in state.get("competitor_findings", []):
        sources.extend(cf.raw_sources)
    return _deduplicate_sources(sources)


def _build_context_string(
    query: str,
    dev_items: list[DevelopmentItem],
    academic_items: list[AcademicItem],
    competitor_items: list[CompetitorItem],
    rag_context: list[SourceItem],
) -> str:
    """Build a comprehensive context string for LLM synthesis calls."""
    parts: list[str] = [f"RESEARCH QUERY: {query}\n"]

    if dev_items:
        parts.append("=== RECENT DEVELOPMENTS ===")
        for item in dev_items:
            parts.append(f"- {item.title} ({item.date}): {item.summary}")

    if academic_items:
        parts.append("\n=== ACADEMIC & TECHNICAL FINDINGS ===")
        for item in academic_items:
            authors = ", ".join(item.authors) if item.authors else "Unknown"
            parts.append(f"- {item.title} by {authors}: {item.summary}")

    if competitor_items:
        parts.append("\n=== COMPETITIVE LANDSCAPE ===")
        for item in competitor_items:
            strengths = "; ".join(item.strengths[:3]) if item.strengths else "N/A"
            weaknesses = "; ".join(item.weaknesses[:3]) if item.weaknesses else "N/A"
            parts.append(
                f"- {item.name}: {item.description} | "
                f"Strengths: {strengths} | Weaknesses: {weaknesses}"
            )

    if rag_context:
        parts.append("\n=== ADDITIONAL RETRIEVED CONTEXT ===")
        for src in rag_context[:10]:
            parts.append(f"- {src.title}: {src.snippet}")

    return "\n".join(parts)


async def _retrieve_rag_context(query: str) -> list[SourceItem]:
    """Query all three ChromaDB collections and return combined results.

    Failures in individual collections are silently skipped.
    """
    all_results: list[SourceItem] = []
    for collection in _RAG_COLLECTIONS:
        try:
            retriever = VectorRetriever(collection_name=collection)
            results = await retriever.search(query, top_k=3)
            all_results.extend(results)
        except Exception as exc:  # noqa: BLE001
            logger.debug("RAG retrieval from %r failed: %s", collection, exc)
    return _deduplicate_sources(all_results)


# ---------------------------------------------------------------------------
# Public agent entry point
# ---------------------------------------------------------------------------


async def run(state: AgentState) -> AgentState:
    """Execute the Report Writer agent and return an updated AgentState.

    Reads all agent findings from state, synthesises them into a validated
    ``ReportSchema``, and writes it to ``state["final_report"]``.

    Args:
        state: Complete LangGraph shared state after all sub-agents have run.

    Returns:
        Updated state with ``final_report`` set to a validated ``ReportSchema``.
        Per-section failures write a placeholder; the report is always returned.
    """
    query: str = state.get("query", "")
    errors: list[str] = list(state.get("errors", []))
    start_time: float = state.get("start_time", time.time())

    with trace_agent("report_writer", query) as trace:
        try:
            llm = _build_llm()

            # ------------------------------------------------------------------
            # Step 1 — flatten all sub-agent findings
            # ------------------------------------------------------------------
            dev_items = _flatten_news(state.get("news_findings", []))
            academic_items = _flatten_academic(state.get("academic_findings", []))
            competitor_items = _flatten_competitors(state.get("competitor_findings", []))

            logger.info(
                "Synthesising: %d developments, %d academic items, %d competitors",
                len(dev_items),
                len(academic_items),
                len(competitor_items),
            )

            # ------------------------------------------------------------------
            # Step 2 — retrieve additional RAG context from all ChromaDB collections
            # ------------------------------------------------------------------
            rag_span = trace.span(name="rag_retrieval", input={"query": query})
            rag_context = await _retrieve_rag_context(query)
            rag_span.end(output={"rag_sources": len(rag_context)})

            # ------------------------------------------------------------------
            # Step 3 — build unified context string
            # ------------------------------------------------------------------
            context = _build_context_string(
                query, dev_items, academic_items, competitor_items, rag_context
            )

            # ------------------------------------------------------------------
            # Step 4 — generate executive summary
            # ------------------------------------------------------------------
            exec_span = trace.span(name="generate_executive_summary")
            executive_summary = ""
            try:
                exec_structured_llm: Any = llm.with_structured_output(
                    _ExecutiveSummaryOutput
                )
                exec_result: _ExecutiveSummaryOutput = await exec_structured_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "You are an expert strategic analyst. Write a comprehensive "
                                "executive summary that synthesises all the research findings "
                                "provided into a coherent narrative for a senior decision-maker."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"Write a 3-5 paragraph executive summary for the following "
                                f"competitive intelligence research on '{query}'.\n\n"
                                f"{context}"
                            )
                        ),
                    ]
                )
                executive_summary = exec_result.executive_summary
            except Exception as exc:  # noqa: BLE001
                executive_summary = (
                    f"Executive summary generation failed: {exc}. "
                    "Please review the individual sections below."
                )
                errors.append(f"report_writer executive_summary error: {exc}")
                logger.error("Executive summary generation failed: %s", exc)
            exec_span.end(output={"length": len(executive_summary)})

            # ------------------------------------------------------------------
            # Step 5 — generate strategic recommendations
            # ------------------------------------------------------------------
            rec_span = trace.span(name="generate_recommendations")
            recommendations: list[str] = []
            try:
                rec_structured_llm: Any = llm.with_structured_output(
                    _RecommendationsOutput
                )
                rec_result: _RecommendationsOutput = await rec_structured_llm.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "You are an expert strategic consultant. Based on competitive "
                                "intelligence research, generate specific, actionable "
                                "strategic recommendations."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"Based on the following competitive intelligence research "
                                f"on '{query}', provide 5-7 specific, actionable strategic "
                                f"recommendations.\n\n{context}"
                            )
                        ),
                    ]
                )
                recommendations = rec_result.recommendations
            except Exception as exc:  # noqa: BLE001
                recommendations = [
                    "Strategic recommendations could not be generated. "
                    "Please review the research findings manually."
                ]
                errors.append(f"report_writer recommendations error: {exc}")
                logger.error("Recommendations generation failed: %s", exc)
            rec_span.end(output={"count": len(recommendations)})

            # ------------------------------------------------------------------
            # Step 6 — collect and deduplicate all sources
            # ------------------------------------------------------------------
            all_sources = _collect_all_sources(state)
            logger.info("Total deduplicated sources: %d", len(all_sources))

            # ------------------------------------------------------------------
            # Step 7 — assemble and validate ReportSchema
            # ------------------------------------------------------------------
            generation_time = round(time.time() - start_time, 2)

            report = ReportSchema.model_validate(
                {
                    "executive_summary": executive_summary,
                    "recent_developments": [i.model_dump() for i in dev_items],
                    "academic_landscape": [i.model_dump() for i in academic_items],
                    "competitive_analysis": [i.model_dump() for i in competitor_items],
                    "strategic_recommendations": recommendations,
                    "total_sources_used": len(all_sources),
                    "generation_time_seconds": generation_time,
                }
            )
            logger.info(
                "Report assembled in %.2fs with %d sources",
                generation_time,
                len(all_sources),
            )

            # ------------------------------------------------------------------
            # Step 8 — log initial quality marker
            # ------------------------------------------------------------------
            trace_id = getattr(trace, "id", "null-trace-id")
            completeness = min(
                1.0,
                (
                    (1 if report.executive_summary else 0)
                    + (1 if report.recent_developments else 0)
                    + (1 if report.academic_landscape else 0)
                    + (1 if report.competitive_analysis else 0)
                    + (1 if report.strategic_recommendations else 0)
                )
                / 5.0,
            )
            score_output(
                trace_id,
                completeness,
                f"Section completeness score: {completeness:.2f}",
            )

            return {
                **state,
                "final_report": report,
                "all_sources": all_sources,
                "errors": errors,
            }

        except Exception as exc:  # noqa: BLE001
            msg = f"report_writer error: {exc}"
            logger.error(msg)
            errors.append(msg)
            # Return a minimal valid report so the API layer never crashes
            fallback_report = ReportSchema(
                executive_summary=(
                    f"Report generation failed for query '{query}'. "
                    f"Error: {exc}"
                ),
                strategic_recommendations=["Report generation failed. Please retry."],
                generation_time_seconds=round(time.time() - start_time, 2),
            )
            return {
                **state,
                "final_report": fallback_report,
                "errors": errors,
            }
