"""
FastAPI backend for the Competitive Intelligence Agent.

Endpoints
---------
POST /research          — Run the full agent pipeline and return a ReportSchema.
GET  /health            — Liveness probe.
GET  /report/{id}       — Retrieve a previously generated report from the in-memory cache.

Run with:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agents.orchestrator import run_research
from schemas.report_schema import ReportSchema

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Competitive Intelligence Agent",
    version="1.0.0",
    description="LangGraph multi-agent system that produces structured competitive intelligence reports.",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Streamlit and any local frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    """Log method, path, status code, and response time for every request."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    logger.info(
        "%s %s → %d  (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a structured JSON error body instead of crashing on unhandled exceptions."""
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )


# ---------------------------------------------------------------------------
# In-memory report cache  (last _MAX_CACHE_SIZE reports, LRU eviction)
# ---------------------------------------------------------------------------

_MAX_CACHE_SIZE = 10
_report_cache: OrderedDict[str, tuple[str, ReportSchema]] = OrderedDict()


def _cache_report(report_id: str, query: str, report: ReportSchema) -> None:
    """Insert a report into the cache, evicting the oldest entry when full."""
    _report_cache[report_id] = (query, report)
    while len(_report_cache) > _MAX_CACHE_SIZE:
        _report_cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    """Request body for POST /research."""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="The company name or topic to research",
        examples=["Anthropic", "Tesla autonomous driving"],
    )
    depth: Literal["quick", "full"] = Field(
        default="full",
        description="Research depth — 'full' runs all agents; 'quick' is reserved for future use",
    )


class ResearchResponse(BaseModel):
    """Response body for POST /research and GET /report/{report_id}."""

    report_id: str = Field(description="UUID that can be used to retrieve this report later")
    query: str = Field(description="The original research query")
    report: ReportSchema = Field(description="The full structured report")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 when the server is running."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/research", response_model=ResearchResponse, tags=["Research"])
async def create_research(request: ResearchRequest) -> ResearchResponse:
    """Run the full competitive intelligence pipeline for the given query.

    Executes news analyst, academic researcher, and competitor profiler agents
    in parallel, then synthesises findings into a structured ReportSchema.

    The generated report is cached by its UUID for retrieval via
    GET /report/{report_id}.
    """
    logger.info("Research request: query=%r depth=%s", request.query, request.depth)
    t0 = time.perf_counter()

    report: ReportSchema = await run_research(request.query)

    elapsed = round(time.perf_counter() - t0, 2)
    # Backfill generation time if the pipeline did not record it
    if report.generation_time_seconds == 0.0:
        report = report.model_copy(update={"generation_time_seconds": elapsed})

    report_id = str(uuid.uuid4())
    _cache_report(report_id, request.query, report)

    logger.info("Research complete: report_id=%s  time=%.2fs", report_id, elapsed)
    return ResearchResponse(report_id=report_id, query=request.query, report=report)


@app.get("/report/{report_id}", response_model=ResearchResponse, tags=["Research"])
async def get_report(report_id: str) -> ResearchResponse:
    """Retrieve a previously generated report by its UUID.

    Reports are cached in memory (last 10).  Older reports are evicted on a
    least-recently-inserted basis.

    Raises:
        HTTPException 404: If the report_id is not in the cache.
    """
    if report_id not in _report_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found. "
                   "It may have been evicted from the cache (only last 10 are kept).",
        )
    query, report = _report_cache[report_id]
    return ResearchResponse(report_id=report_id, query=query, report=report)
