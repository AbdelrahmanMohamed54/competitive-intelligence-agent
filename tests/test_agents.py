"""
Unit tests for all four sub-agents: News Analyst, Academic Researcher,
Competitor Profiler, and Report Writer.

All LLM calls, web searches, and vector-store writes are mocked so these tests
run without any API keys or network access.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

_FAKE_SOURCES = [
    SourceItem(url="https://news.com/1", title="Story 1", snippet="Snippet 1"),
    SourceItem(url="https://news.com/2", title="Story 2", snippet="Snippet 2"),
]


def _make_base_state(query: str = "OpenAI") -> AgentState:
    return {
        "query": query,
        "news_findings": [],
        "academic_findings": [],
        "competitor_findings": [],
        "final_report": None,
        "all_sources": [],
        "errors": [],
        "start_time": 0.0,
    }


# ---------------------------------------------------------------------------
# News Analyst tests
# ---------------------------------------------------------------------------


class TestNewsAnalyst:
    """Tests for agents.news_analyst.run()"""

    def _mock_llm_for_news(self) -> MagicMock:
        """Return a mock LLM whose structured-output variant returns canned data."""
        from agents.news_analyst import _NewsLLMOutput, _RawDevItem, _SearchQueriesOutput

        queries_output = _SearchQueriesOutput(
            queries=["OpenAI news", "OpenAI GPT-5 release", "OpenAI funding 2026"]
        )
        news_output = _NewsLLMOutput(
            items=[
                _RawDevItem(
                    title="OpenAI releases GPT-5",
                    summary="OpenAI announced GPT-5 with major improvements.",
                    date="2026-04-01",
                    source_urls=["https://news.com/1"],
                )
            ]
        )

        # The code calls structured_llm.ainvoke(...) — NOT structured_llm(...).
        # side_effect must be on .ainvoke, not on the structured_llm itself.
        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(side_effect=[queries_output, news_output])
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock
        return llm_mock

    def test_run_populates_news_findings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """run() must return state with news_findings containing DevelopmentItems."""
        import agents.news_analyst as na

        monkeypatch.setattr(na, "_build_llm", self._mock_llm_for_news)
        monkeypatch.setattr(
            "agents.news_analyst.tavily_search",
            AsyncMock(return_value=_FAKE_SOURCES),
        )

        # Mock VectorRetriever so no ChromaDB I/O happens
        mock_retriever = MagicMock()
        mock_retriever.add_documents = AsyncMock(return_value=None)
        monkeypatch.setattr(na, "VectorRetriever", MagicMock(return_value=mock_retriever))

        # Mock trace_agent to a no-op context manager
        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _null_trace(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        monkeypatch.setattr(na, "trace_agent", _null_trace)

        state = _make_base_state("OpenAI")
        result: AgentState = asyncio.run(na.run(state))

        assert "news_findings" in result
        assert len(result["news_findings"]) == 1
        findings: NewsFindings = result["news_findings"][0]
        assert isinstance(findings, NewsFindings)
        assert len(findings.items) >= 1
        assert isinstance(findings.items[0], DevelopmentItem)
        assert findings.items[0].title == "OpenAI releases GPT-5"
        assert len(findings.raw_sources) == 2  # deduplicated across 3 searches

    def test_run_catches_llm_error_and_appends_to_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the LLM raises, run() must not propagate the exception.

        The error message must appear in state["errors"] and news_findings must
        contain an empty NewsFindings sentinel so downstream agents can still run.
        """
        import agents.news_analyst as na

        def _boom_llm() -> None:
            raise RuntimeError("LLM unreachable")

        monkeypatch.setattr(na, "_build_llm", _boom_llm)

        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _null_trace(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        monkeypatch.setattr(na, "trace_agent", _null_trace)

        state = _make_base_state("OpenAI")
        result: AgentState = asyncio.run(na.run(state))

        assert len(result.get("errors", [])) > 0
        assert "news_analyst error" in result["errors"][0]
        # Pipeline must not crash — findings must still be present (empty)
        assert len(result["news_findings"]) == 1
        assert result["news_findings"][0].items == []

    def test_run_preserves_existing_state_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() must not clobber unrelated state keys."""
        import agents.news_analyst as na

        monkeypatch.setattr(na, "_build_llm", self._mock_llm_for_news)
        monkeypatch.setattr(
            "agents.news_analyst.tavily_search",
            AsyncMock(return_value=_FAKE_SOURCES),
        )
        mock_retriever = MagicMock()
        mock_retriever.add_documents = AsyncMock(return_value=None)
        monkeypatch.setattr(na, "VectorRetriever", MagicMock(return_value=mock_retriever))

        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _null_trace(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        monkeypatch.setattr(na, "trace_agent", _null_trace)

        state = _make_base_state("OpenAI")
        state["start_time"] = 1234567890.0  # type: ignore[typeddict-unknown-key]
        result: AgentState = asyncio.run(na.run(state))

        assert result["query"] == "OpenAI"
        assert result["start_time"] == 1234567890.0  # type: ignore[typeddict-item]


# ---------------------------------------------------------------------------
# Academic Researcher tests
# ---------------------------------------------------------------------------


class TestAcademicResearcher:
    """Tests for agents.academic_researcher.run()"""

    def _mock_llm_for_academic(self) -> MagicMock:
        """Return a mock LLM for the academic researcher."""
        from agents.academic_researcher import (
            _AcademicLLMOutput,
            _RawAcademicItem,
            _SearchQueriesOutput,
        )

        queries_output = _SearchQueriesOutput(
            queries=[
                "OpenAI research papers arxiv",
                "OpenAI patents 2025",
                "OpenAI technical reports IEEE",
            ]
        )
        academic_output = _AcademicLLMOutput(
            items=[
                _RawAcademicItem(
                    title="Scaling Laws for Neural Language Models",
                    summary="Establishes empirical scaling laws for LLMs.",
                    authors=["Kaplan et al."],
                    url="https://arxiv.org/abs/2001.08361",
                    source_urls=["https://news.com/1"],
                )
            ]
        )

        # Must set side_effect on .ainvoke, not on the structured_mock itself.
        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(side_effect=[queries_output, academic_output])
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock
        return llm_mock

    def test_run_populates_academic_findings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() must return state with academic_findings containing AcademicItems."""
        import agents.academic_researcher as ar

        monkeypatch.setattr(ar, "_build_llm", self._mock_llm_for_academic)
        monkeypatch.setattr(
            "agents.academic_researcher.tavily_search",
            AsyncMock(return_value=_FAKE_SOURCES),
        )
        mock_retriever = MagicMock()
        mock_retriever.add_documents = AsyncMock(return_value=None)
        monkeypatch.setattr(ar, "VectorRetriever", MagicMock(return_value=mock_retriever))

        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _null_trace(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        monkeypatch.setattr(ar, "trace_agent", _null_trace)

        state = _make_base_state("OpenAI")
        result: AgentState = asyncio.run(ar.run(state))

        assert "academic_findings" in result
        assert len(result["academic_findings"]) == 1
        findings: AcademicFindings = result["academic_findings"][0]
        assert isinstance(findings, AcademicFindings)
        assert len(findings.items) >= 1
        assert isinstance(findings.items[0], AcademicItem)
        assert findings.items[0].title == "Scaling Laws for Neural Language Models"
        assert findings.items[0].url == "https://arxiv.org/abs/2001.08361"

    def test_run_catches_llm_error_and_appends_to_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the LLM raises, run() must not propagate the exception."""
        import agents.academic_researcher as ar

        def _boom_llm() -> None:
            raise RuntimeError("API key expired")

        monkeypatch.setattr(ar, "_build_llm", _boom_llm)

        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _null_trace(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        monkeypatch.setattr(ar, "trace_agent", _null_trace)

        state = _make_base_state("OpenAI")
        result: AgentState = asyncio.run(ar.run(state))

        assert len(result.get("errors", [])) > 0
        assert "academic_researcher error" in result["errors"][0]
        assert len(result["academic_findings"]) == 1
        assert result["academic_findings"][0].items == []

    def test_run_handles_empty_search_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() must gracefully handle Tavily returning no results."""
        import agents.academic_researcher as ar

        from agents.academic_researcher import _SearchQueriesOutput

        queries_output = _SearchQueriesOutput(queries=["q1", "q2", "q3"])
        structured_mock = AsyncMock(return_value=queries_output)
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock

        monkeypatch.setattr(ar, "_build_llm", lambda: llm_mock)
        # Tavily returns nothing
        monkeypatch.setattr(
            "agents.academic_researcher.tavily_search",
            AsyncMock(return_value=[]),
        )

        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _null_trace(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        monkeypatch.setattr(ar, "trace_agent", _null_trace)

        state = _make_base_state("obscure-topic-xyz")
        result: AgentState = asyncio.run(ar.run(state))

        # Should succeed with empty findings, not crash
        assert result.get("errors", []) == []
        assert len(result["academic_findings"]) == 1
        assert result["academic_findings"][0].items == []


# ---------------------------------------------------------------------------
# Langfuse tracer tests
# ---------------------------------------------------------------------------


class TestLangfuseTracer:
    """Smoke tests for the observability tracer."""

    def test_get_tracer_returns_null_when_keys_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from observability import langfuse_tracer
        from observability.langfuse_tracer import _NullTracer

        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        tracer = langfuse_tracer.get_tracer()
        assert isinstance(tracer, _NullTracer)

    def test_trace_agent_context_manager_yields_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from observability import langfuse_tracer
        from observability.langfuse_tracer import _NullTrace, _NullTracer

        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        with langfuse_tracer.trace_agent("test_agent", "test query") as trace:
            assert isinstance(trace, _NullTrace)

    def test_score_output_is_noop_for_null_trace_id(self) -> None:
        from observability import langfuse_tracer

        # Must not raise
        langfuse_tracer.score_output("null-trace-id", 0.9, "good output")


# ---------------------------------------------------------------------------
# Competitor Profiler tests
# ---------------------------------------------------------------------------


class TestCompetitorProfiler:
    """Tests for agents.competitor_profiler.run()"""

    def _mock_llm_for_competitors(self) -> MagicMock:
        """Return a mock LLM for the competitor profiler."""
        from agents.competitor_profiler import (
            _CompetitorLLMOutput,
            _RawCompetitorItem,
            _SearchQueriesOutput,
        )

        queries_output = _SearchQueriesOutput(
            queries=[
                "OpenAI competitors AI",
                "OpenAI vs Anthropic vs Google comparison",
                "AI market positioning 2026",
                "OpenAI pricing enterprise",
            ]
        )
        competitor_output = _CompetitorLLMOutput(
            items=[
                _RawCompetitorItem(
                    name="Anthropic",
                    description="AI safety company behind Claude.",
                    strengths=["safety focus", "strong research"],
                    weaknesses=["smaller market share"],
                    source_urls=["https://news.com/1"],
                ),
                _RawCompetitorItem(
                    name="Google DeepMind",
                    description="Alphabet's AI division behind Gemini.",
                    strengths=["scale", "data access"],
                    weaknesses=["slow commercialisation"],
                    source_urls=["https://news.com/2"],
                ),
            ]
        )

        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(
            side_effect=[queries_output, competitor_output]
        )
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock
        return llm_mock

    def _null_trace_ctx(self) -> Any:
        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _ctx(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        return _ctx

    def test_run_populates_competitor_findings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() must return state with competitor_findings containing CompetitorItems."""
        import agents.competitor_profiler as cp

        monkeypatch.setattr(cp, "_build_llm", self._mock_llm_for_competitors)
        monkeypatch.setattr(
            "agents.competitor_profiler.tavily_search",
            AsyncMock(return_value=_FAKE_SOURCES),
        )
        mock_retriever = MagicMock()
        mock_retriever.add_documents = AsyncMock(return_value=None)
        monkeypatch.setattr(cp, "VectorRetriever", MagicMock(return_value=mock_retriever))
        monkeypatch.setattr(cp, "trace_agent", self._null_trace_ctx())

        state = _make_base_state("OpenAI")
        result: AgentState = asyncio.run(cp.run(state))

        assert "competitor_findings" in result
        assert len(result["competitor_findings"]) == 1
        findings: CompetitorFindings = result["competitor_findings"][0]
        assert isinstance(findings, CompetitorFindings)
        assert len(findings.items) == 2
        assert isinstance(findings.items[0], CompetitorItem)
        assert findings.items[0].name == "Anthropic"
        assert "safety focus" in findings.items[0].strengths

    def test_run_catches_llm_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the LLM raises, competitor_findings must be empty and error logged."""
        import agents.competitor_profiler as cp

        monkeypatch.setattr(cp, "_build_llm", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(cp, "trace_agent", self._null_trace_ctx())

        state = _make_base_state("OpenAI")
        result: AgentState = asyncio.run(cp.run(state))

        assert any("competitor_profiler error" in e for e in result.get("errors", []))
        assert result["competitor_findings"][0].items == []

    def test_run_handles_no_competitors_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LLM returning an empty items list must not cause errors."""
        import agents.competitor_profiler as cp
        from agents.competitor_profiler import _CompetitorLLMOutput, _SearchQueriesOutput

        queries_output = _SearchQueriesOutput(queries=["q1", "q2", "q3", "q4"])
        empty_output = _CompetitorLLMOutput(items=[])

        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(
            side_effect=[queries_output, empty_output]
        )
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock

        monkeypatch.setattr(cp, "_build_llm", lambda: llm_mock)
        monkeypatch.setattr(
            "agents.competitor_profiler.tavily_search",
            AsyncMock(return_value=_FAKE_SOURCES),
        )
        mock_retriever = MagicMock()
        mock_retriever.add_documents = AsyncMock(return_value=None)
        monkeypatch.setattr(cp, "VectorRetriever", MagicMock(return_value=mock_retriever))
        monkeypatch.setattr(cp, "trace_agent", self._null_trace_ctx())

        state = _make_base_state("obscure-niche-product")
        result: AgentState = asyncio.run(cp.run(state))

        assert result.get("errors", []) == []
        assert result["competitor_findings"][0].items == []


# ---------------------------------------------------------------------------
# Report Writer tests
# ---------------------------------------------------------------------------

_FAKE_DEV_ITEM = DevelopmentItem(
    title="OpenAI launches GPT-5",
    summary="Major capability leap.",
    date="2026-04-01",
    sources=[SourceItem(url="https://openai.com/gpt5", title="GPT-5", snippet="...")],
)
_FAKE_ACADEMIC_ITEM = AcademicItem(
    title="Scaling Laws",
    summary="Empirical scaling laws for LLMs.",
    authors=["Kaplan et al."],
    url="https://arxiv.org/abs/2001.08361",
    sources=[SourceItem(url="https://arxiv.org/abs/2001.08361", title="Paper", snippet="...")],
)
_FAKE_COMPETITOR_ITEM = CompetitorItem(
    name="Anthropic",
    description="AI safety company.",
    strengths=["safety"],
    weaknesses=["scale"],
    sources=[SourceItem(url="https://anthropic.com", title="Anthropic", snippet="...")],
)


def _make_full_state() -> AgentState:
    """Create a fully-populated state as it would appear after all sub-agents run."""
    return {
        "query": "OpenAI",
        "news_findings": [NewsFindings(items=[_FAKE_DEV_ITEM], raw_sources=_FAKE_SOURCES)],
        "academic_findings": [
            AcademicFindings(
                items=[_FAKE_ACADEMIC_ITEM],
                raw_sources=[SourceItem(url="https://arxiv.org/1", title="S", snippet="x")],
            )
        ],
        "competitor_findings": [
            CompetitorFindings(
                items=[_FAKE_COMPETITOR_ITEM],
                raw_sources=[SourceItem(url="https://anthropic.com", title="A", snippet="y")],
            )
        ],
        "final_report": None,
        "all_sources": _FAKE_SOURCES,
        "errors": [],
        "start_time": 0.0,
    }


class TestReportWriter:
    """Tests for agents.report_writer.run()"""

    def _mock_llm_for_report(self) -> MagicMock:
        """Mock LLM that returns canned executive summary + recommendations."""
        from agents.report_writer import _ExecutiveSummaryOutput, _RecommendationsOutput

        exec_output = _ExecutiveSummaryOutput(
            executive_summary=(
                "OpenAI remains the dominant player in the generative AI space, "
                "having recently launched GPT-5 with significant capability improvements. "
                "Academic research confirms continued scaling benefits. Competitors such "
                "as Anthropic are differentiating on safety but lack scale."
            )
        )
        rec_output = _RecommendationsOutput(
            recommendations=[
                "Invest in safety research to match Anthropic's positioning.",
                "Accelerate enterprise sales to capitalise on GPT-5 launch.",
                "Monitor academic publications for emerging capability threats.",
            ]
        )

        structured_mock = MagicMock()
        structured_mock.ainvoke = AsyncMock(side_effect=[exec_output, rec_output])
        llm_mock = MagicMock()
        llm_mock.with_structured_output.return_value = structured_mock
        return llm_mock

    def _null_trace_ctx(self) -> Any:
        from contextlib import contextmanager
        from observability.langfuse_tracer import _NullTrace

        @contextmanager
        def _ctx(agent_name: str, query: str):  # type: ignore[override]
            yield _NullTrace()

        return _ctx

    def _mock_retriever(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch VectorRetriever.search to return empty list (no I/O)."""
        import agents.report_writer as rw

        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search = AsyncMock(return_value=[])
        monkeypatch.setattr(
            rw, "VectorRetriever", MagicMock(return_value=mock_retriever_instance)
        )

    def test_run_produces_valid_report_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run() must return state with final_report set to a valid ReportSchema."""
        import agents.report_writer as rw

        monkeypatch.setattr(rw, "_build_llm", self._mock_llm_for_report)
        monkeypatch.setattr(rw, "trace_agent", self._null_trace_ctx())
        monkeypatch.setattr(rw, "score_output", MagicMock())
        self._mock_retriever(monkeypatch)

        state = _make_full_state()
        result: AgentState = asyncio.run(rw.run(state))

        assert result["final_report"] is not None
        assert isinstance(result["final_report"], ReportSchema)

    def test_all_five_sections_present_and_non_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All 5 ReportSchema sections must be populated after a successful run."""
        import agents.report_writer as rw

        monkeypatch.setattr(rw, "_build_llm", self._mock_llm_for_report)
        monkeypatch.setattr(rw, "trace_agent", self._null_trace_ctx())
        monkeypatch.setattr(rw, "score_output", MagicMock())
        self._mock_retriever(monkeypatch)

        state = _make_full_state()
        result: AgentState = asyncio.run(rw.run(state))

        report: ReportSchema = result["final_report"]
        assert report.executive_summary, "executive_summary must not be empty"
        assert len(report.recent_developments) >= 1, "recent_developments must be populated"
        assert len(report.academic_landscape) >= 1, "academic_landscape must be populated"
        assert len(report.competitive_analysis) >= 1, "competitive_analysis must be populated"
        assert len(report.strategic_recommendations) >= 1, "recommendations must be populated"
        assert report.generation_time_seconds >= 0.0

    def test_sources_collected_from_all_agent_findings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """total_sources_used must count sources from news, academic, and competitor agents."""
        import agents.report_writer as rw

        monkeypatch.setattr(rw, "_build_llm", self._mock_llm_for_report)
        monkeypatch.setattr(rw, "trace_agent", self._null_trace_ctx())
        monkeypatch.setattr(rw, "score_output", MagicMock())
        self._mock_retriever(monkeypatch)

        state = _make_full_state()
        result: AgentState = asyncio.run(rw.run(state))

        report: ReportSchema = result["final_report"]
        # State has sources from news (2), academic (1), competitor (1), all_sources (2)
        # After deduplication there should be at least 3 distinct URLs
        assert report.total_sources_used >= 3
        # all_sources in returned state must also be populated
        assert len(result["all_sources"]) >= 3

    def test_run_returns_fallback_report_on_llm_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the LLM fails entirely, a fallback ReportSchema must still be returned."""
        import agents.report_writer as rw

        monkeypatch.setattr(
            rw, "_build_llm", lambda: (_ for _ in ()).throw(RuntimeError("no key"))
        )
        monkeypatch.setattr(rw, "trace_agent", self._null_trace_ctx())

        state = _make_full_state()
        result: AgentState = asyncio.run(rw.run(state))

        # Must not be None — fallback report must be present
        assert result["final_report"] is not None
        assert isinstance(result["final_report"], ReportSchema)
        assert any("report_writer error" in e for e in result.get("errors", []))
