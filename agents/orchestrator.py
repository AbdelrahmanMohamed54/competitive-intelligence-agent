"""
LangGraph orchestrator — wires all four sub-agents into a compiled StateGraph.

Graph structure
---------------
                    START
                   / | \\
                  /  |  \\
     news_analyst  academic  competitor_profiler   (parallel fan-out)
                  \\  |  /
                   \\ | /
                   check                           (fan-in + routing)
                     |
              [conditional edge]
             /                 \\
      report_writer            END (early exit — all agents failed)
             |
            END

Key design choices
------------------
- Fan-out: three edges from START to the three search agents; LangGraph runs
  them as concurrent async tasks.
- Fan-in: three edges into a single no-op "check" node; LangGraph waits for
  all three before proceeding.
- Reducers: list fields in AgentState use ``operator.add`` so that all three
  parallel nodes can safely write to the same field without overwriting each
  other (LangGraph merges them automatically).
- Timeout: every node wraps its agent call in ``asyncio.wait_for``
  (_AGENT_TIMEOUT seconds).  On timeout, empty findings are returned and the
  error is appended to state["errors"]; the pipeline continues.
- Early exit: if every sub-agent produced empty findings AND errors were
  recorded, the conditional edge skips report_writer and terminates with a
  fallback minimal report.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents import academic_researcher, competitor_profiler, news_analyst, report_writer
from schemas.report_schema import (
    AcademicFindings,
    AgentState,
    CompetitorFindings,
    NewsFindings,
    ReportSchema,
)

logger = logging.getLogger(__name__)

_AGENT_TIMEOUT: float = 25.0  # seconds; each node must complete within this window


# ---------------------------------------------------------------------------
# Timeout-wrapped node functions
# ---------------------------------------------------------------------------


def _extract(result: dict, *keys: str) -> dict:
    """Return a sub-dict containing only the listed keys from *result*.

    This is the critical adapter between agent run() functions (which return
    the full ``{**state, ...}`` dict for standalone use) and LangGraph node
    return values (which must only contain keys the node is responsible for).

    LangGraph raises ``InvalidUpdateError`` when multiple parallel nodes write
    to the same non-annotated field (e.g. ``query``, ``start_time``).  By
    extracting only owned keys we avoid that conflict entirely.
    """
    return {k: result[k] for k in keys if k in result}


async def _news_analyst_node(state: AgentState) -> dict:
    """LangGraph node: run the News Analyst agent with a timeout guard.

    Returns only the keys this node owns so that parallel execution does not
    conflict on non-annotated state fields such as ``query`` or ``start_time``.
    """
    try:
        result = await asyncio.wait_for(
            news_analyst.run(state), timeout=_AGENT_TIMEOUT
        )
        return _extract(result, "news_findings", "all_sources", "errors")
    except asyncio.TimeoutError:
        msg = f"news_analyst timed out after {_AGENT_TIMEOUT}s"
        logger.error(msg)
        return {"news_findings": [NewsFindings()], "errors": [msg]}


async def _academic_researcher_node(state: AgentState) -> dict:
    """LangGraph node: run the Academic Researcher agent with a timeout guard."""
    try:
        result = await asyncio.wait_for(
            academic_researcher.run(state), timeout=_AGENT_TIMEOUT
        )
        return _extract(result, "academic_findings", "all_sources", "errors")
    except asyncio.TimeoutError:
        msg = f"academic_researcher timed out after {_AGENT_TIMEOUT}s"
        logger.error(msg)
        return {"academic_findings": [AcademicFindings()], "errors": [msg]}


async def _competitor_profiler_node(state: AgentState) -> dict:
    """LangGraph node: run the Competitor Profiler agent with a timeout guard."""
    try:
        result = await asyncio.wait_for(
            competitor_profiler.run(state), timeout=_AGENT_TIMEOUT
        )
        return _extract(result, "competitor_findings", "all_sources", "errors")
    except asyncio.TimeoutError:
        msg = f"competitor_profiler timed out after {_AGENT_TIMEOUT}s"
        logger.error(msg)
        return {"competitor_findings": [CompetitorFindings()], "errors": [msg]}


async def _report_writer_node(state: AgentState) -> dict:
    """LangGraph node: run the Report Writer agent with a timeout guard."""
    try:
        result = await asyncio.wait_for(
            report_writer.run(state), timeout=_AGENT_TIMEOUT
        )
        return _extract(result, "final_report", "all_sources", "errors")
    except asyncio.TimeoutError:
        msg = f"report_writer timed out after {_AGENT_TIMEOUT}s"
        logger.error(msg)
        fallback = ReportSchema(
            executive_summary=(
                f"Report generation timed out for query: "
                f"'{state.get('query', '')}'. Please retry."
            ),
            strategic_recommendations=["Report timed out. Please retry."],
            generation_time_seconds=_AGENT_TIMEOUT,
        )
        return {"final_report": fallback, "errors": [msg]}


# ---------------------------------------------------------------------------
# Fan-in pass-through and routing
# ---------------------------------------------------------------------------


async def _check_node(state: AgentState) -> dict:
    """No-op fan-in node.

    LangGraph waits for all three parallel nodes before executing this node.
    Returns an empty dict so state is left unchanged; routing is handled by
    the conditional edge attached to this node.
    """
    return {}


def _route_after_parallel(state: AgentState) -> str:
    """Decide whether to run report_writer or exit early.

    Returns:
        ``"report_writer"`` in the normal case.
        ``END`` if every sub-agent produced empty findings and at least one
        error was recorded (i.e., the pipeline has nothing to synthesise).
    """
    has_news = any(f.items for f in state.get("news_findings", []))
    has_academic = any(f.items for f in state.get("academic_findings", []))
    has_competitors = any(f.items for f in state.get("competitor_findings", []))
    errors = state.get("errors", [])

    if not has_news and not has_academic and not has_competitors and errors:
        logger.warning(
            "All sub-agents returned empty findings — routing to END early. "
            "Recorded errors: %s",
            errors,
        )
        return END

    return "report_writer"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    """Build and compile the LangGraph StateGraph.

    Called once per ``run_research`` invocation so that agent functions are
    looked up at call time — making the graph trivially mockable in tests.

    Returns:
        A compiled ``CompiledStateGraph`` ready to be invoked with ``ainvoke``.
    """
    g: StateGraph = StateGraph(AgentState)

    # Register all nodes
    g.add_node("news_analyst", _news_analyst_node)
    g.add_node("academic_researcher", _academic_researcher_node)
    g.add_node("competitor_profiler", _competitor_profiler_node)
    g.add_node("check", _check_node)
    g.add_node("report_writer", _report_writer_node)

    # Fan-out: START → three parallel agents
    g.add_edge(START, "news_analyst")
    g.add_edge(START, "academic_researcher")
    g.add_edge(START, "competitor_profiler")

    # Fan-in: all three agents → check node
    g.add_edge("news_analyst", "check")
    g.add_edge("academic_researcher", "check")
    g.add_edge("competitor_profiler", "check")

    # Conditional routing after fan-in
    g.add_conditional_edges("check", _route_after_parallel)

    # Linear tail: report_writer → END
    g.add_edge("report_writer", END)

    return g.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_research(query: str) -> ReportSchema:
    """Run the full competitive intelligence pipeline for a given query.

    Initialises the shared AgentState, invokes the compiled graph, and returns
    the validated ReportSchema produced by the report_writer agent.

    The three search agents (news, academic, competitor) run concurrently.
    Total wall-clock time is typically under 90 seconds when API latencies are
    within normal bounds.

    Args:
        query: The company name or research topic to investigate.

    Returns:
        A validated ``ReportSchema`` containing all five report sections.
        A fallback minimal report is returned if the pipeline fails entirely.
    """
    t0 = time.perf_counter()
    logger.info("Starting research pipeline for query: %r", query)

    initial_state: AgentState = {
        "query": query,
        "news_findings": [],
        "academic_findings": [],
        "competitor_findings": [],
        "final_report": None,
        "all_sources": [],
        "errors": [],
        "start_time": time.time(),
    }

    graph = _build_graph()
    result: dict = await graph.ainvoke(initial_state)

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline completed in %.2fs for query: %r", elapsed, query)

    errors = result.get("errors", [])
    if errors:
        logger.warning(
            "Pipeline finished with %d error(s): %s", len(errors), errors
        )

    report: ReportSchema | None = result.get("final_report")
    if report is None:
        logger.error("final_report missing from state — returning fallback report")
        report = ReportSchema(
            executive_summary=(
                f"The research pipeline completed but produced no report "
                f"for query: '{query}'. Please retry."
            ),
            strategic_recommendations=["Pipeline produced no output. Please retry."],
            generation_time_seconds=round(elapsed, 2),
        )

    return report
