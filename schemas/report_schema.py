"""
Pydantic v2 models for all agent inputs/outputs and the final report structure.

This module defines every typed schema used throughout the competitive intelligence
pipeline. All inter-agent communication must use these models — no raw dicts or strings.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Base source model
# ---------------------------------------------------------------------------


class SourceItem(BaseModel):
    """A single retrieved source document used to ground agent outputs.

    Every claim made by any agent must be traceable to a SourceItem.
    """

    url: str = Field(description="Full URL of the source document")
    title: str = Field(description="Title of the page or document")
    snippet: str = Field(description="Relevant excerpt or summary from the source")


# ---------------------------------------------------------------------------
# Report section item models
# ---------------------------------------------------------------------------


class DevelopmentItem(BaseModel):
    """A single recent development or news item grounded in retrieved sources."""

    title: str = Field(description="Headline or short title of the development")
    summary: str = Field(description="2-4 sentence summary of the development")
    date: str = Field(description="Publication or event date (ISO 8601 or human-readable)")
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Source documents used to ground this development item",
    )


class AcademicItem(BaseModel):
    """A single academic paper or technical publication with source citations."""

    title: str = Field(description="Full title of the paper or publication")
    summary: str = Field(
        description="2-4 sentence plain-language summary of the paper's contribution"
    )
    authors: list[str] = Field(
        default_factory=list,
        description="List of author names",
    )
    url: str = Field(description="Canonical URL or DOI link for the paper")
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Source documents used to ground this academic item",
    )


class CompetitorItem(BaseModel):
    """A profile of a single competitor or comparable organisation."""

    name: str = Field(description="Company or organisation name")
    description: str = Field(description="Brief overview of what they do")
    strengths: list[str] = Field(
        default_factory=list,
        description="Key competitive advantages",
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Known limitations or vulnerabilities",
    )
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Source documents used to ground this competitor profile",
    )


# ---------------------------------------------------------------------------
# Final report model
# ---------------------------------------------------------------------------


class ReportSchema(BaseModel):
    """Fully structured competitive intelligence report produced by the report writer.

    Contains five sections:
    1. executive_summary
    2. recent_developments
    3. academic_landscape
    4. competitive_analysis
    5. strategic_recommendations

    Also tracks provenance metadata: total sources used and generation time.
    """

    executive_summary: str = Field(
        description="High-level 3-5 paragraph synthesis of all findings"
    )
    recent_developments: list[DevelopmentItem] = Field(
        default_factory=list,
        description="Curated list of the most relevant recent news and developments",
    )
    academic_landscape: list[AcademicItem] = Field(
        default_factory=list,
        description="Key academic papers and technical publications",
    )
    competitive_analysis: list[CompetitorItem] = Field(
        default_factory=list,
        description="Profiles of competitors and comparable organisations",
    )
    strategic_recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations derived from the research",
    )
    total_sources_used: int = Field(
        default=0,
        description="Count of unique source documents cited across the entire report",
    )
    generation_time_seconds: float = Field(
        default=0.0,
        description="Wall-clock seconds taken to generate this report",
    )


# ---------------------------------------------------------------------------
# Individual agent output models
# ---------------------------------------------------------------------------


class NewsFindings(BaseModel):
    """Output model emitted by the News Analyst agent.

    Wraps discovered development items and the raw sources retrieved before
    any filtering or deduplication.
    """

    items: list[DevelopmentItem] = Field(
        default_factory=list,
        description="List of development items discovered by the news analyst",
    )
    raw_sources: list[SourceItem] = Field(
        default_factory=list,
        description="All source documents retrieved during the search, before filtering",
    )


class AcademicFindings(BaseModel):
    """Output model emitted by the Academic Researcher agent.

    Wraps discovered academic items and the raw sources retrieved before
    any filtering or deduplication.
    """

    items: list[AcademicItem] = Field(
        default_factory=list,
        description="List of academic items discovered by the researcher",
    )
    raw_sources: list[SourceItem] = Field(
        default_factory=list,
        description="All source documents retrieved during the search, before filtering",
    )


class CompetitorFindings(BaseModel):
    """Output model emitted by the Competitor Profiler agent.

    Wraps discovered competitor profiles and the raw sources retrieved before
    any filtering or deduplication.
    """

    items: list[CompetitorItem] = Field(
        default_factory=list,
        description="List of competitor profiles discovered by the profiler",
    )
    raw_sources: list[SourceItem] = Field(
        default_factory=list,
        description="All source documents retrieved during the search, before filtering",
    )


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Shared state object passed between every node in the LangGraph state machine.

    All fields are optional (total=False) so each agent updates only the keys it
    owns without touching the rest of the state.  List fields that are written by
    parallel nodes use ``operator.add`` as their LangGraph reducer so concurrent
    writes are merged rather than overwritten.
    """

    query: str
    """The original user query that kicked off the pipeline."""

    news_findings: Annotated[list[NewsFindings], operator.add]
    """Populated by the news_analyst node.  Merged across parallel runs."""

    academic_findings: Annotated[list[AcademicFindings], operator.add]
    """Populated by the academic_researcher node.  Merged across parallel runs."""

    competitor_findings: Annotated[list[CompetitorFindings], operator.add]
    """Populated by the competitor_profiler node.  Merged across parallel runs."""

    final_report: Optional[ReportSchema]
    """Populated by the report_writer node once all sub-agents have finished."""

    all_sources: Annotated[list[SourceItem], operator.add]
    """Accumulated list of every SourceItem retrieved across all agents."""

    errors: Annotated[list[str], operator.add]
    """Any agent-level error messages for graceful handling and observability."""

    start_time: float
    """Unix timestamp (from time.time()) recorded when the pipeline was invoked."""
