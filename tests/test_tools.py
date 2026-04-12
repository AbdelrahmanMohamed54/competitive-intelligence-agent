"""
Unit tests for Pydantic schemas (Task 1) and tool wrappers (Tasks 2 & 3).

These tests run without any API keys or network access.  Tool tests use
monkeypatching / mocking to isolate the units under test.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

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


# ---------------------------------------------------------------------------
# Schema smoke tests — valid instantiation
# ---------------------------------------------------------------------------


class TestSourceItem:
    """SourceItem must accept valid data and reject missing required fields."""

    def test_valid_instantiation(self) -> None:
        item = SourceItem(
            url="https://example.com/article",
            title="Example Article",
            snippet="This is a short excerpt from the article.",
        )
        assert item.url == "https://example.com/article"
        assert item.title == "Example Article"
        assert item.snippet == "This is a short excerpt from the article."

    def test_missing_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            SourceItem(title="No URL", snippet="snippet")  # type: ignore[call-arg]

    def test_missing_title_raises(self) -> None:
        with pytest.raises(ValidationError):
            SourceItem(url="https://x.com", snippet="snippet")  # type: ignore[call-arg]

    def test_missing_snippet_raises(self) -> None:
        with pytest.raises(ValidationError):
            SourceItem(url="https://x.com", title="title")  # type: ignore[call-arg]


class TestDevelopmentItem:
    """DevelopmentItem requires title, summary, and date."""

    def test_valid_with_sources(self) -> None:
        src = SourceItem(url="https://news.com", title="News", snippet="Breaking news.")
        item = DevelopmentItem(
            title="AI Breakthrough",
            summary="Researchers achieved a major milestone.",
            date="2026-04-01",
            sources=[src],
        )
        assert len(item.sources) == 1
        assert item.sources[0].url == "https://news.com"

    def test_sources_defaults_to_empty(self) -> None:
        item = DevelopmentItem(
            title="T", summary="S", date="2026-01-01"
        )
        assert item.sources == []

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            DevelopmentItem(summary="S", date="2026-01-01")  # type: ignore[call-arg]


class TestAcademicItem:
    """AcademicItem requires title, summary, and url."""

    def test_valid_instantiation(self) -> None:
        item = AcademicItem(
            title="Attention Is All You Need",
            summary="Introduces the Transformer architecture.",
            authors=["Vaswani et al."],
            url="https://arxiv.org/abs/1706.03762",
        )
        assert item.url == "https://arxiv.org/abs/1706.03762"
        assert item.authors == ["Vaswani et al."]

    def test_missing_url_raises(self) -> None:
        with pytest.raises(ValidationError):
            AcademicItem(title="T", summary="S")  # type: ignore[call-arg]


class TestCompetitorItem:
    """CompetitorItem requires name and description."""

    def test_valid_instantiation(self) -> None:
        item = CompetitorItem(
            name="Acme Corp",
            description="Leading provider of widgets.",
            strengths=["brand recognition"],
            weaknesses=["legacy tech stack"],
        )
        assert item.name == "Acme Corp"
        assert "brand recognition" in item.strengths

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            CompetitorItem(description="D")  # type: ignore[call-arg]


class TestReportSchema:
    """ReportSchema must contain all five required sections."""

    def test_minimal_valid_report(self) -> None:
        report = ReportSchema(
            executive_summary="This is the executive summary.",
            strategic_recommendations=["Invest in R&D"],
        )
        assert report.executive_summary == "This is the executive summary."
        assert report.total_sources_used == 0
        assert report.generation_time_seconds == 0.0

    def test_full_report(self) -> None:
        src = SourceItem(url="https://x.com", title="X", snippet="snippet")
        dev = DevelopmentItem(title="T", summary="S", date="2026-01-01", sources=[src])
        acad = AcademicItem(title="P", summary="A", url="https://arxiv.org/1", sources=[src])
        comp = CompetitorItem(name="Rival", description="D", sources=[src])

        report = ReportSchema(
            executive_summary="Exec summary.",
            recent_developments=[dev],
            academic_landscape=[acad],
            competitive_analysis=[comp],
            strategic_recommendations=["Do X", "Do Y"],
            total_sources_used=3,
            generation_time_seconds=42.5,
        )
        assert len(report.recent_developments) == 1
        assert len(report.academic_landscape) == 1
        assert len(report.competitive_analysis) == 1
        assert report.total_sources_used == 3
        assert report.generation_time_seconds == 42.5

    def test_missing_executive_summary_raises(self) -> None:
        with pytest.raises(ValidationError):
            ReportSchema()  # type: ignore[call-arg]


class TestAgentOutputModels:
    """NewsFindings, AcademicFindings, and CompetitorFindings."""

    def test_news_findings_defaults(self) -> None:
        nf = NewsFindings()
        assert nf.items == []
        assert nf.raw_sources == []

    def test_academic_findings_with_data(self) -> None:
        src = SourceItem(url="https://x.com", title="T", snippet="S")
        item = AcademicItem(title="Paper", summary="Sum", url="https://x.com")
        af = AcademicFindings(items=[item], raw_sources=[src])
        assert len(af.items) == 1
        assert len(af.raw_sources) == 1

    def test_competitor_findings_with_data(self) -> None:
        src = SourceItem(url="https://x.com", title="T", snippet="S")
        item = CompetitorItem(name="Rival", description="D")
        cf = CompetitorFindings(items=[item], raw_sources=[src])
        assert len(cf.items) == 1


class TestAgentState:
    """AgentState TypedDict must accept all required keys."""

    def test_valid_state(self) -> None:
        import time

        state: AgentState = {
            "query": "OpenAI competitive landscape",
            "news_findings": [],
            "academic_findings": [],
            "competitor_findings": [],
            "final_report": None,
            "all_sources": [],
            "errors": [],
            "start_time": time.time(),
        }
        assert state["query"] == "OpenAI competitive landscape"
        assert state["final_report"] is None

    def test_partial_state_allowed(self) -> None:
        """total=False means a state dict with only some keys is valid."""
        state: AgentState = {"query": "test"}
        assert state["query"] == "test"


# ---------------------------------------------------------------------------
# Schema rejection tests — invalid data
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Cross-model tests verifying Pydantic rejects bad data."""

    def test_source_item_rejects_non_string_url(self) -> None:
        with pytest.raises(ValidationError):
            SourceItem(url=123, title="T", snippet="S")  # type: ignore[arg-type]

    def test_development_item_rejects_wrong_type_for_sources(self) -> None:
        with pytest.raises(ValidationError):
            DevelopmentItem(
                title="T",
                summary="S",
                date="2026-01-01",
                sources=["not-a-source-item"],  # type: ignore[list-item]
            )

    def test_report_schema_rejects_wrong_type_for_total_sources(self) -> None:
        with pytest.raises(ValidationError):
            ReportSchema(
                executive_summary="E",
                total_sources_used="many",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Web search tool tests (mocked)
# ---------------------------------------------------------------------------


class TestWebSearch:
    """tools.web_search — unit tests with mocked Tavily client."""

    def test_search_sync_returns_source_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools import web_search

        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "url": "https://news.com/1",
                    "title": "Big News",
                    "content": "Something happened.",
                }
            ]
        }
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setattr(web_search, "_get_client", lambda: mock_client)

        results = web_search.search_sync("AI news", max_results=1)

        assert len(results) == 1
        assert isinstance(results[0], SourceItem)
        assert results[0].url == "https://news.com/1"
        assert results[0].title == "Big News"
        assert results[0].snippet == "Something happened."

    def test_search_sync_returns_empty_list_on_api_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import web_search

        def _boom() -> None:
            raise RuntimeError("Network failure")

        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setattr(web_search, "_get_client", _boom)

        results = web_search.search_sync("query")
        assert results == []

    def test_search_sync_returns_empty_list_when_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tools import web_search

        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        results = web_search.search_sync("query")
        assert results == []

    def test_async_search_delegates_to_sync(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tools import web_search

        expected = [SourceItem(url="https://x.com", title="T", snippet="S")]
        monkeypatch.setattr(web_search, "_search_sync_inner", lambda q, n: expected)

        result = asyncio.run(web_search.search("test", max_results=3))
        assert result == expected
