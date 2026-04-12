"""
Langfuse observability tracing integration.

Provides a thin wrapper around the Langfuse SDK so that every agent execution,
LLM call, and tool call is recorded as a structured trace.  If the Langfuse
keys are absent or the SDK is unavailable the module silently falls back to a
no-op tracer so the rest of the pipeline is never blocked.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# No-op fallback objects
# ---------------------------------------------------------------------------


class _NullSpan:
    """Span-like object that silently discards all calls."""

    def end(self, **kwargs: Any) -> None:  # noqa: ANN401
        pass

    def update(self, **kwargs: Any) -> None:  # noqa: ANN401
        pass


class _NullTrace:
    """Trace-like object that silently discards all calls."""

    id: str = "null-trace-id"

    def update(self, **kwargs: Any) -> None:  # noqa: ANN401
        pass

    def span(self, **kwargs: Any) -> _NullSpan:  # noqa: ANN401
        return _NullSpan()

    def generation(self, **kwargs: Any) -> _NullSpan:  # noqa: ANN401
        return _NullSpan()

    def event(self, **kwargs: Any) -> None:  # noqa: ANN401
        pass


class _NullTracer:
    """Langfuse-like client that silently discards all calls."""

    def trace(self, **kwargs: Any) -> _NullTrace:  # noqa: ANN401
        return _NullTrace()

    def score(self, **kwargs: Any) -> None:  # noqa: ANN401
        pass

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_tracer() -> Any:  # noqa: ANN401
    """Return an initialised Langfuse client, or a no-op tracer if keys are missing.

    Reads ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and optionally
    ``LANGFUSE_HOST`` from the environment.

    Returns:
        A ``Langfuse`` instance when keys are present, otherwise a ``_NullTracer``.
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set — "
            "tracing disabled, using no-op tracer."
        )
        return _NullTracer()

    try:
        from langfuse import Langfuse  # type: ignore[import]

        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        logger.info("Langfuse tracer initialised (host=%s)", host)
        return client
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialise Langfuse client: %s — using no-op tracer.", exc)
        return _NullTracer()


@contextmanager
def trace_agent(
    agent_name: str,
    query: str,
) -> Generator[Any, None, None]:
    """Context manager that wraps an agent execution in a Langfuse trace.

    Records the agent name, input query, start/end time, and whether the
    execution succeeded or raised an exception.  The trace object is yielded
    so callers can attach child spans and generations to it.

    Example::

        with trace_agent("news_analyst", state["query"]) as trace:
            span = trace.span(name="tavily_search", input={"query": q})
            results = await search(q)
            span.end(output={"count": len(results)})

    Args:
        agent_name: Human-readable name of the agent (used as trace name).
        query:      The original user query driving this execution.

    Yields:
        The active Langfuse trace (or ``_NullTrace`` when tracing is disabled).
    """
    tracer = get_tracer()
    trace = tracer.trace(
        name=agent_name,
        input={"query": query},
        metadata={"agent": agent_name},
    )
    start = time.perf_counter()
    try:
        yield trace
        elapsed = time.perf_counter() - start
        trace.update(
            output={
                "status": "success",
                "duration_seconds": round(elapsed, 3),
            }
        )
        logger.info("%s completed in %.2fs", agent_name, elapsed)
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start
        trace.update(
            output={
                "status": "error",
                "error": str(exc),
                "duration_seconds": round(elapsed, 3),
            }
        )
        logger.error("%s failed after %.2fs: %s", agent_name, elapsed, exc)
        raise
    finally:
        try:
            tracer.flush()
        except Exception:  # noqa: BLE001
            pass


def score_output(trace_id: str, score: float, comment: str = "") -> None:
    """Submit an LLM-as-judge quality score for a previously recorded trace.

    This is called by the evaluation pipeline after the report has been
    generated and assessed.  Fails silently if the tracer is unavailable.

    Args:
        trace_id: The Langfuse trace ID returned by a ``trace_agent`` call.
        score:    A normalised quality score between 0.0 and 1.0.
        comment:  Optional explanation from the judge model.
    """
    if trace_id == "null-trace-id":
        return
    try:
        tracer = get_tracer()
        tracer.score(
            trace_id=trace_id,
            name="llm_judge_quality",
            value=score,
            comment=comment,
        )
        logger.info("Scored trace %s: %.3f", trace_id, score)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record score for trace %s: %s", trace_id, exc)
