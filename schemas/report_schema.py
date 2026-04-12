"""
Pydantic v2 models for all agent inputs/outputs and the final report structure.

This module defines every typed schema used throughout the competitive intelligence
pipeline. All inter-agent communication must use these models — no raw dicts or strings.
"""

from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Report section item models
# ---------------------------------------------------------------------------


class DevelopmentItem(BaseModel):
    """A single recent development or news item with source citations."""

    title: str = Field(description="Headline or short title of the development")
    summary: str = Field(description="2-4 sentence summary of the development")
    date: str = Field(description="Publication or event date (ISO 8601 or human-readable)")
    relevance: str = Field(description="Why this development matters to the query topic")
    sources: list[str] = Field(
        default_factory=list,
        description="URLs or document identifiers used to ground this item",
    )


class AcademicItem(BaseModel):
    """A single academic paper or technical publication with source citations."""

    title: str = Field(description="Full title of the paper or publication")
    authors: list[str] = Field(
        default_factory=list,
        description="List of author names",
    )
    year: str = Field(description="Publication year")
    venue: str = Field(description="Journal, conference, or pre-print server")
    abstract_summary: str = Field(
        description="2-4 sentence plain-language summary of the paper's contribution"
    )
    relevance: str = Field(description="Why this paper matters to the query topic")
    sources: list[str] = Field(
        default_factory=list,
        description="URLs or DOIs used to ground this item",
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
    market_position: str = Field(description="Where they sit in the competitive landscape")
    recent_moves: list[str] = Field(
        default_factory=list,
        description="Notable recent actions (launches, acquisitions, partnerships)",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="URLs or document identifiers used to ground this profile",
    )


# ---------------------------------------------------------------------------
# Final report model
# ---------------------------------------------------------------------------


class ReportSchema(BaseModel):
    """
    Fully structured competitive intelligence report produced by the report writer.

    Contains five sections as mandated by the project specification:
    1. executive_summary
    2. recent_developments
    3. academic_landscape
    4. competitive_analysis
    5. strategic_recommendations
    """

    query: str = Field(description="The original user query that generated this report")
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
    all_sources: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of every source URL cited in the report",
    )


# ---------------------------------------------------------------------------
# Individual agent output models
# ---------------------------------------------------------------------------


class NewsFindings(BaseModel):
    """Output model emitted by the News Analyst agent."""

    query: str = Field(description="The search query used to retrieve these findings")
    items: list[DevelopmentItem] = Field(
        default_factory=list,
        description="List of development items discovered",
    )
    raw_sources: list[str] = Field(
        default_factory=list,
        description="All source URLs retrieved during the search, before filtering",
    )


class AcademicFindings(BaseModel):
    """Output model emitted by the Academic Researcher agent."""

    query: str = Field(description="The search query used to retrieve these findings")
    items: list[AcademicItem] = Field(
        default_factory=list,
        description="List of academic items discovered",
    )
    raw_sources: list[str] = Field(
        default_factory=list,
        description="All source URLs retrieved during the search, before filtering",
    )


class CompetitorFindings(BaseModel):
    """Output model emitted by the Competitor Profiler agent."""

    query: str = Field(description="The search query used to retrieve these findings")
    items: list[CompetitorItem] = Field(
        default_factory=list,
        description="List of competitor profiles discovered",
    )
    raw_sources: list[str] = Field(
        default_factory=list,
        description="All source URLs retrieved during the search, before filtering",
    )


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """
    Shared state object passed between every node in the LangGraph state machine.

    All fields are optional (total=False) so individual agents can update only
    the keys they own without touching the rest of the state.
    """

    query: str
    """The original user query that kicked off the pipeline."""

    news_findings: Annotated[list[NewsFindings], "output from the News Analyst agent"]
    """Populated by the news_analyst node."""

    academic_findings: Annotated[
        list[AcademicFindings], "output from the Academic Researcher agent"
    ]
    """Populated by the academic_researcher node."""

    competitor_findings: Annotated[
        list[CompetitorFindings], "output from the Competitor Profiler agent"
    ]
    """Populated by the competitor_profiler node."""

    final_report: Annotated[ReportSchema | None, "structured final output"]
    """Populated by the report_writer node."""

    sources: list[str]
    """Accumulated list of all retrieved URLs and document identifiers."""

    errors: list[str]
    """Any agent-level error messages for graceful handling and observability."""
