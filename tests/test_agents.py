"""
Unit tests for the News Analyst and Academic Researcher agents.

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
    DevelopmentItem,
    NewsFindings,
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
