"""
Integration tests for the LangGraph orchestrator.

All four agent run() functions are mocked so these tests run without any API
keys, network access, or ChromaDB I/O.  The tests verify:

  1. The full graph executes all four agents and returns a ReportSchema.
  2. The three search agents (news / academic / competitor) run concurrently
     (parallel fan-out), not sequentially.
  3. The report_writer node receives a state that contains findings from all
     three search agents.
  4. A single timed-out agent does not block the rest of the pipeline.
  5. The early-exit route fires when all agents fail with empty findings.
"""

from __future__ import annotations

import asyncio
import time
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
# Shared canned data
# ---------------------------------------------------------------------------

_SOURCE = SourceItem(url="https://example.com", title="Title", snippet="Snippet")

_NEWS_FINDINGS = NewsFindings(
    items=[DevelopmentItem(title="AI News", summary="Summary", date="2026-04-01",
                           sources=[_SOURCE])],
    raw_sources=[_SOURCE],
)
_ACADEMIC_FINDINGS = AcademicFindings(
    items=[AcademicItem(title="Paper", summary="Sum", url="https://arxiv.org/1",
                        sources=[_SOURCE])],
    raw_sources=[_SOURCE],
)
_COMPETITOR_FINDINGS = CompetitorFindings(
    items=[CompetitorItem(name="Rival", description="Desc", sources=[_SOURCE])],
    raw_sources=[_SOURCE],
)
_REPORT = ReportSchema(
    executive_summary="Excellent summary.",
    recent_developments=[
        DevelopmentItem(title="AI News", summary="Summary", date="2026-04-01")
    ],
    academic_landscape=[
        AcademicItem(title="Paper", summary="Sum", url="https://arxiv.org/1")
    ],
    competitive_analysis=[CompetitorItem(name="Rival", description="Desc")],
    strategic_recommendations=["Invest more in R&D"],
    total_sources_used=3,
    generation_time_seconds=12.5,
)


def _make_news_return(state: AgentState) -> dict:
    # Return ONLY the keys this agent owns.  Returning {**state, ...} would write
    # non-annotated fields (query, start_time) from multiple parallel nodes in the
    # same step, which LangGraph rejects with InvalidUpdateError.
    return {"news_findings": [_NEWS_FINDINGS], "all_sources": [_SOURCE], "errors": []}


def _make_academic_return(state: AgentState) -> dict:
    return {"academic_findings": [_ACADEMIC_FINDINGS], "all_sources": [_SOURCE], "errors": []}


def _make_competitor_return(state: AgentState) -> dict:
    return {"competitor_findings": [_COMPETITOR_FINDINGS], "all_sources": [_SOURCE], "errors": []}


def _make_report_return(state: AgentState) -> dict:
    return {"final_report": _REPORT, "all_sources": [], "errors": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_all_agents(
    monkeypatch: pytest.MonkeyPatch,
    news_fn: Any = None,
    academic_fn: Any = None,
    competitor_fn: Any = None,
    report_fn: Any = None,
) -> None:
    """Patch all four agent run() functions with AsyncMocks."""
    monkeypatch.setattr(
        "agents.news_analyst.run",
        AsyncMock(side_effect=news_fn or _make_news_return),
    )
    monkeypatch.setattr(
        "agents.academic_researcher.run",
        AsyncMock(side_effect=academic_fn or _make_academic_return),
    )
    monkeypatch.setattr(
        "agents.competitor_profiler.run",
        AsyncMock(side_effect=_make_competitor_return if competitor_fn is None else competitor_fn),
    )
    monkeypatch.setattr(
        "agents.report_writer.run",
        AsyncMock(side_effect=report_fn or _make_report_return),
    )


# ---------------------------------------------------------------------------
# Test 1 — full pipeline returns a valid ReportSchema
# ---------------------------------------------------------------------------


class TestOrchestratorHappyPath:
    """End-to-end happy-path tests."""

    def test_run_research_returns_valid_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_research() must return a non-None ReportSchema with all 5 sections."""
        _patch_all_agents(monkeypatch)

        from agents.orchestrator import run_research

        result = asyncio.run(run_research("OpenAI"))

        assert isinstance(result, ReportSchema)
        assert result.executive_summary
        assert len(result.recent_developments) >= 1
        assert len(result.academic_landscape) >= 1
        assert len(result.competitive_analysis) >= 1
        assert len(result.strategic_recommendations) >= 1

    def test_all_four_agent_run_functions_are_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every agent's run() function must be invoked exactly once."""
        news_mock = AsyncMock(side_effect=_make_news_return)
        academic_mock = AsyncMock(side_effect=_make_academic_return)
        competitor_mock = AsyncMock(side_effect=_make_competitor_return)
        report_mock = AsyncMock(side_effect=_make_report_return)

        monkeypatch.setattr("agents.news_analyst.run", news_mock)
        monkeypatch.setattr("agents.academic_researcher.run", academic_mock)
        monkeypatch.setattr("agents.competitor_profiler.run", competitor_mock)
        monkeypatch.setattr("agents.report_writer.run", report_mock)

        from agents.orchestrator import run_research

        asyncio.run(run_research("OpenAI"))

        news_mock.assert_called_once()
        academic_mock.assert_called_once()
        competitor_mock.assert_called_once()
        report_mock.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2 — parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Verify the three search agents execute concurrently, not sequentially."""

    def test_search_agents_run_concurrently(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The three search agents must start within 0.5s of each other.

        Each mock agent sleeps for 0.15s. If they ran sequentially the total
        would be ≥ 0.45s. If parallel, all three should complete in ≈ 0.15s.
        """
        start_times: dict[str, float] = {}
        end_times: dict[str, float] = {}

        async def slow_news(state: AgentState) -> dict:
            start_times["news"] = time.perf_counter()
            await asyncio.sleep(0.15)
            end_times["news"] = time.perf_counter()
            return _make_news_return(state)

        async def slow_academic(state: AgentState) -> dict:
            start_times["academic"] = time.perf_counter()
            await asyncio.sleep(0.15)
            end_times["academic"] = time.perf_counter()
            return _make_academic_return(state)

        async def slow_competitor(state: AgentState) -> dict:
            start_times["competitor"] = time.perf_counter()
            await asyncio.sleep(0.15)
            end_times["competitor"] = time.perf_counter()
            return _make_competitor_return(state)

        monkeypatch.setattr("agents.news_analyst.run", slow_news)
        monkeypatch.setattr("agents.academic_researcher.run", slow_academic)
        monkeypatch.setattr("agents.competitor_profiler.run", slow_competitor)
        monkeypatch.setattr(
            "agents.report_writer.run",
            AsyncMock(side_effect=_make_report_return),
        )

        from agents.orchestrator import run_research

        t0 = time.perf_counter()
        asyncio.run(run_research("OpenAI"))
        total_elapsed = time.perf_counter() - t0

        # All three agents must have started
        assert "news" in start_times
        assert "academic" in start_times
        assert "competitor" in start_times

        # Total elapsed should be well under 3 × 0.15s = 0.45s (sequential).
        # Allow generous headroom for CI overhead.
        assert total_elapsed < 0.45, (
            f"Total elapsed {total_elapsed:.3f}s suggests sequential execution "
            f"(expected < 0.45s for parallel agents sleeping 0.15s each)"
        )

        # All three must have started before the slowest one finished
        latest_start = max(start_times.values())
        earliest_end = min(end_times.values())
        # Latest start should be before the first agent finished
        # (i.e., they were all running at the same time)
        assert latest_start < earliest_end, (
            f"latest start {latest_start:.3f} >= earliest end {earliest_end:.3f} "
            "— agents appear to have run sequentially"
        )


# ---------------------------------------------------------------------------
# Test 3 — report_writer receives findings from all three agents
# ---------------------------------------------------------------------------


class TestReportWriterReceivesAllFindings:
    """Verify that the report_writer node sees the merged state."""

    def test_report_writer_state_contains_all_three_findings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The state passed to report_writer.run must include findings from
        all three search agents merged via operator.add reducers."""

        captured_state: dict[str, Any] = {}

        async def spy_report_writer(state: AgentState) -> dict:
            # Capture a snapshot of what report_writer actually received
            captured_state.update(dict(state))
            return _make_report_return(state)

        monkeypatch.setattr("agents.news_analyst.run",
                            AsyncMock(side_effect=_make_news_return))
        monkeypatch.setattr("agents.academic_researcher.run",
                            AsyncMock(side_effect=_make_academic_return))
        monkeypatch.setattr("agents.competitor_profiler.run",
                            AsyncMock(side_effect=_make_competitor_return))
        monkeypatch.setattr("agents.report_writer.run", spy_report_writer)

        from agents.orchestrator import run_research

        asyncio.run(run_research("OpenAI"))

        assert captured_state, "spy was never called — report_writer did not run"

        # All three findings must be present and non-empty
        assert len(captured_state.get("news_findings", [])) >= 1, \
            "report_writer did not receive news_findings"
        assert len(captured_state.get("academic_findings", [])) >= 1, \
            "report_writer did not receive academic_findings"
        assert len(captured_state.get("competitor_findings", [])) >= 1, \
            "report_writer did not receive competitor_findings"

        # Verify they contain actual items, not empty placeholders
        assert captured_state["news_findings"][0].items, \
            "news_findings items are empty in report_writer state"
        assert captured_state["academic_findings"][0].items, \
            "academic_findings items are empty in report_writer state"
        assert captured_state["competitor_findings"][0].items, \
            "competitor_findings items are empty in report_writer state"

        # Original query must be preserved
        assert captured_state.get("query") == "OpenAI"


# ---------------------------------------------------------------------------
# Test 4 — one timeout does not block the pipeline
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """A timed-out agent must not block report_writer from running."""

    def test_timeout_adds_error_and_pipeline_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If news_analyst times out, the pipeline must still produce a report."""
        import agents.orchestrator as orch

        # Override timeout to be very short for this test
        original_timeout = orch._AGENT_TIMEOUT
        monkeypatch.setattr(orch, "_AGENT_TIMEOUT", 0.05)

        async def slow_news(state: AgentState) -> dict:
            await asyncio.sleep(10)  # will time out
            return _make_news_return(state)

        monkeypatch.setattr("agents.news_analyst.run", slow_news)
        monkeypatch.setattr("agents.academic_researcher.run",
                            AsyncMock(side_effect=_make_academic_return))
        monkeypatch.setattr("agents.competitor_profiler.run",
                            AsyncMock(side_effect=_make_competitor_return))
        monkeypatch.setattr("agents.report_writer.run",
                            AsyncMock(side_effect=_make_report_return))

        from agents.orchestrator import run_research

        result = asyncio.run(run_research("OpenAI"))

        # Report must still be produced despite the timeout
        assert isinstance(result, ReportSchema)

        # Restore
        monkeypatch.setattr(orch, "_AGENT_TIMEOUT", original_timeout)


# ---------------------------------------------------------------------------
# Test 5 — early-exit route fires when all agents fail
# ---------------------------------------------------------------------------


class TestEarlyExit:
    """Conditional edge to END must fire when all sub-agents return empty findings."""

    def test_early_exit_skips_report_writer_and_returns_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all three agents produce empty findings with errors, report_writer
        must be skipped and run_research must return a fallback ReportSchema."""

        async def failing_news(state: AgentState) -> dict:
            return {
                "news_findings": [NewsFindings()],
                "errors": ["news_analyst error: API key missing"],
            }

        async def failing_academic(state: AgentState) -> dict:
            return {
                "academic_findings": [AcademicFindings()],
                "errors": ["academic_researcher error: API key missing"],
            }

        async def failing_competitor(state: AgentState) -> dict:
            return {
                "competitor_findings": [CompetitorFindings()],
                "errors": ["competitor_profiler error: API key missing"],
            }

        report_spy = AsyncMock(side_effect=_make_report_return)

        monkeypatch.setattr("agents.news_analyst.run", failing_news)
        monkeypatch.setattr("agents.academic_researcher.run", failing_academic)
        monkeypatch.setattr("agents.competitor_profiler.run", failing_competitor)
        monkeypatch.setattr("agents.report_writer.run", report_spy)

        from agents.orchestrator import run_research

        result = asyncio.run(run_research("OpenAI"))

        # report_writer must NOT have been called
        report_spy.assert_not_called()

        # run_research must still return a fallback ReportSchema (not raise)
        assert isinstance(result, ReportSchema)
