"""
LLM-as-judge evaluation pipeline for the Competitive Intelligence Agent.

Uses Gemini to score generated reports on four criteria (1–5 each):
  1. Relevance      — does the report actually answer the query?
  2. Factual grounding — are claims linked to real sources?
  3. Completeness   — are all 5 sections substantive?
  4. Actionability  — are recommendations specific and useful?

Run a batch evaluation::

    import asyncio, json
    from evaluation.llm_judge import LLMJudge

    judge = LLMJudge()
    results = asyncio.run(judge.run_batch_evaluation(
        json.load(open("evaluation/test_queries.json"))
    ))

Or from the command line::

    python evaluation/llm_judge.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()

# Make sure project root is on sys.path when run directly
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.orchestrator import run_research  # noqa: E402
from schemas.report_schema import ReportSchema  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gemini-2.5-flash"
_RESULTS_PATH = Path(__file__).parent / "results.json"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Evaluation scores for a single research report."""

    query: str = Field(description="The research query that generated this report")
    relevance: float = Field(
        ge=1.0, le=5.0,
        description="How well the report answers the specific query (1–5)",
    )
    factual_grounding: float = Field(
        ge=1.0, le=5.0,
        description="How well claims are linked to cited sources (1–5)",
    )
    completeness: float = Field(
        ge=1.0, le=5.0,
        description="Whether all 5 sections are substantive and non-empty (1–5)",
    )
    actionability: float = Field(
        ge=1.0, le=5.0,
        description="How specific and useful the strategic recommendations are (1–5)",
    )
    overall_score: float = Field(
        ge=1.0, le=5.0,
        description="Average of the four individual criterion scores",
    )
    reasoning: str = Field(
        description="Brief reasoning for the scores from the judge model",
    )
    report_generation_time: float = Field(
        default=0.0,
        description="Wall-clock seconds the research pipeline took",
    )


class _JudgeOutput(BaseModel):
    """Structured output from the judge LLM call."""

    relevance: float = Field(ge=1.0, le=5.0)
    factual_grounding: float = Field(ge=1.0, le=5.0)
    completeness: float = Field(ge=1.0, le=5.0)
    actionability: float = Field(ge=1.0, le=5.0)
    reasoning: str


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Evaluate research reports using Gemini as the judge model.

    Args:
        model: Gemini model name override (defaults to ``GEMINI_MODEL`` env var
               or ``gemini-2.5-flash``).
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        model_name = model or os.getenv("GEMINI_MODEL", _DEFAULT_MODEL)
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
        self._structured_llm = self._llm.with_structured_output(_JudgeOutput)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_report_text(report: ReportSchema) -> str:
        """Render a ReportSchema into a compact text block for the judge prompt."""
        lines: list[str] = []

        lines.append("=== EXECUTIVE SUMMARY ===")
        lines.append(report.executive_summary or "(empty)")

        lines.append("\n=== RECENT DEVELOPMENTS ===")
        for item in report.recent_developments:
            src_count = len(item.sources)
            lines.append(f"- [{item.date}] {item.title} ({src_count} sources)")

        lines.append("\n=== ACADEMIC LANDSCAPE ===")
        for item in report.academic_landscape:
            src_count = len(item.sources)
            lines.append(f"- {item.title} ({src_count} sources)")

        lines.append("\n=== COMPETITIVE ANALYSIS ===")
        for item in report.competitive_analysis:
            src_count = len(item.sources)
            strengths = len(item.strengths)
            weaknesses = len(item.weaknesses)
            lines.append(
                f"- {item.name}: {strengths} strengths, {weaknesses} weaknesses "
                f"({src_count} sources)"
            )

        lines.append("\n=== STRATEGIC RECOMMENDATIONS ===")
        for i, rec in enumerate(report.strategic_recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append(f"\nTotal sources cited: {report.total_sources_used}")
        return "\n".join(lines)

    async def score_report(
        self, report: ReportSchema, query: str
    ) -> EvaluationResult:
        """Score a single research report using the LLM judge.

        Args:
            report: The ``ReportSchema`` to evaluate.
            query:  The original research query.

        Returns:
            An ``EvaluationResult`` with individual criterion scores (1–5),
            overall average, and the judge's reasoning.
        """
        report_text = self._build_report_text(report)

        prompt = f"""You are an expert evaluator assessing the quality of competitive intelligence reports.

Research Query: "{query}"

Report to Evaluate:
{report_text}

Score this report on the following four criteria, each on a scale of 1–5:

1. RELEVANCE (1–5): Does the report directly and thoroughly address the research query?
   - 5: All sections are highly relevant to the specific query
   - 3: Mostly relevant with some tangential content
   - 1: Largely off-topic or generic

2. FACTUAL GROUNDING (1–5): Are claims backed by cited sources?
   - 5: Nearly every claim has an associated source citation
   - 3: Most major claims are cited but some are unsupported
   - 1: Very few citations; mostly unsupported assertions

3. COMPLETENESS (1–5): Are all 5 report sections substantive?
   - 5: All sections have meaningful, detailed content (≥3 items each)
   - 3: Most sections have content but some are thin
   - 1: Multiple sections are empty or have only 1 item

4. ACTIONABILITY (1–5): Are the strategic recommendations specific and useful?
   - 5: Recommendations are specific, evidence-backed, and immediately actionable
   - 3: Recommendations are sensible but somewhat generic
   - 1: Recommendations are vague, obvious, or not grounded in the report findings

Provide a brief reasoning explaining your scores, then return all four scores."""

        try:
            result: _JudgeOutput = await self._structured_llm.ainvoke(prompt)
            overall = round(
                (result.relevance + result.factual_grounding +
                 result.completeness + result.actionability) / 4.0,
                2,
            )
            return EvaluationResult(
                query=query,
                relevance=result.relevance,
                factual_grounding=result.factual_grounding,
                completeness=result.completeness,
                actionability=result.actionability,
                overall_score=overall,
                reasoning=result.reasoning,
                report_generation_time=report.generation_time_seconds,
            )
        except Exception as exc:
            logger.error("Judge LLM call failed for query %r: %s", query, exc)
            return EvaluationResult(
                query=query,
                relevance=1.0,
                factual_grounding=1.0,
                completeness=1.0,
                actionability=1.0,
                overall_score=1.0,
                reasoning=f"Evaluation failed: {exc}",
                report_generation_time=report.generation_time_seconds,
            )

    async def run_batch_evaluation(
        self, queries: list[str]
    ) -> list[EvaluationResult]:
        """Run the full pipeline (research + judge) for every query.

        Generates a report for each query, scores it, saves all results to
        ``evaluation/results.json``, and prints a summary table.

        Args:
            queries: List of company names or research topics to evaluate.

        Returns:
            A list of ``EvaluationResult`` objects, one per query.
        """
        results: list[EvaluationResult] = []

        for query in queries:
            logger.info("Evaluating query: %r", query)
            print(f"\nResearching: {query} …", flush=True)

            t0 = time.perf_counter()
            try:
                report = await run_research(query)
            except Exception as exc:
                logger.error("Pipeline failed for %r: %s", query, exc)
                print(f"  Pipeline error: {exc}")
                results.append(
                    EvaluationResult(
                        query=query,
                        relevance=1.0,
                        factual_grounding=1.0,
                        completeness=1.0,
                        actionability=1.0,
                        overall_score=1.0,
                        reasoning=f"Pipeline error: {exc}",
                        report_generation_time=round(time.perf_counter() - t0, 2),
                    )
                )
                continue

            gen_time = round(time.perf_counter() - t0, 2)
            print(f"  Research done in {gen_time:.1f}s — scoring …", flush=True)

            evaluation = await self.score_report(report, query)
            # Ensure generation time is recorded even if the pipeline didn't set it
            if evaluation.report_generation_time == 0.0:
                evaluation = evaluation.model_copy(
                    update={"report_generation_time": gen_time}
                )
            results.append(evaluation)
            print(
                f"  Overall score: {evaluation.overall_score:.2f}/5.0  "
                f"(R={evaluation.relevance} G={evaluation.factual_grounding} "
                f"C={evaluation.completeness} A={evaluation.actionability})",
                flush=True,
            )

        # Save results
        _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _RESULTS_PATH.open("w", encoding="utf-8") as fh:
            json.dump(
                [r.model_dump() for r in results],
                fh,
                indent=2,
                default=str,
            )
        logger.info("Results saved to %s", _RESULTS_PATH)

        # Print summary table
        _print_summary(results)
        return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(results: list[EvaluationResult]) -> None:
    """Print a formatted summary table to stdout."""
    col_w = 30
    print("\n" + "=" * 90)
    print(
        f"{'Query':<{col_w}} {'Relevance':>9} {'Grounding':>9} "
        f"{'Complete':>9} {'Action':>7} {'Avg':>6} {'Time':>8}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r.query:<{col_w}} {r.relevance:>9.1f} {r.factual_grounding:>9.1f} "
            f"{r.completeness:>9.1f} {r.actionability:>7.1f} "
            f"{r.overall_score:>6.2f} {r.report_generation_time:>7.1f}s"
        )
    print("=" * 90)
    if results:
        avg_overall = sum(r.overall_score for r in results) / len(results)
        print(f"{'Mean overall score':<{col_w}} {'':>9} {'':>9} {'':>9} {'':>7} {avg_overall:>6.2f}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    queries_path = Path(__file__).parent / "test_queries.json"
    if not queries_path.exists():
        print(f"test_queries.json not found at {queries_path}")
        sys.exit(1)

    with queries_path.open() as fh:
        queries: list[str] = json.load(fh)

    judge = LLMJudge()
    asyncio.run(judge.run_batch_evaluation(queries))
