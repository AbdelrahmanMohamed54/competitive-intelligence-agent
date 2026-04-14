# Project: Autonomous Research & Competitive Intelligence Agent

## What this project does
A LangGraph multi-agent system that takes a company or topic as input and
autonomously produces a structured 5-section competitive intelligence report
in under 90 seconds. Each section is grounded in retrieved source documents
(RAG via ChromaDB) to eliminate hallucination. Full observability via Langfuse.

## Architecture
User Input (company/topic)
    → Orchestrator Agent (LangGraph state machine)
        → News Analyst Agent (web search + RAG)
        → Academic Researcher Agent (web search + RAG)
        → Competitor Profiler Agent (web search + RAG)
        → Report Writer Agent (synthesizes all outputs)
    → Pydantic-validated structured report
    → Langfuse observability trace
    → Streamlit frontend display

## Tech stack
- Orchestration: LangGraph (LangChain)
- LLM primary: Google Gemini 2.5 Flash via langchain-google-genai
- LLM fallback: Groq Llama 3.3 70B via langchain-groq
- Web search tool: Tavily API
- Vector store: ChromaDB with sentence-transformers embeddings
- Output validation: Pydantic v2
- Observability: Langfuse
- Frontend: Streamlit
- API: FastAPI

## File structure
- agents/orchestrator.py — LangGraph state machine, routes between sub-agents
- agents/news_analyst.py — searches and summarizes recent news
- agents/academic_researcher.py — finds papers and technical publications
- agents/competitor_profiler.py — maps competitive landscape
- agents/report_writer.py — synthesizes all agent outputs into final report
- tools/web_search.py — Tavily search tool wrapper
- tools/vector_retriever.py — ChromaDB retrieval tool
- schemas/report_schema.py — Pydantic models for all agent inputs/outputs
- observability/langfuse_tracer.py — Langfuse tracing integration
- app/streamlit_app.py — Streamlit frontend
- app/api.py — FastAPI backend
- evaluation/llm_judge.py — LLM-as-judge scoring pipeline

## LangGraph state design
The shared state object passed between all agents must contain:
- query: str — the original user query
- news_findings: list — output from news analyst
- academic_findings: list — output from academic researcher
- competitor_findings: list — output from competitor profiler
- final_report: ReportSchema — structured final output
- sources: list — all retrieved URLs and documents
- errors: list — any agent errors for graceful handling

## Pydantic output schemas
Every agent must emit a typed Pydantic model — no raw dicts passed between agents.
The final ReportSchema has 5 sections:
1. executive_summary: str
2. recent_developments: list[DevelopmentItem]
3. academic_landscape: list[AcademicItem]
4. competitive_analysis: list[CompetitorItem]
5. strategic_recommendations: list[str]
Each item includes a sources: list[str] field for citation tracking.

## Key design decisions
- RAG-grounded outputs: every claim must link to a retrieved source document
- Parallel sub-agent execution where possible to stay under 90 seconds
- All inter-agent communication via typed Pydantic schemas, never raw strings
- Langfuse traces every LLM call, tool call, latency, tokens, and cost
- Provider-agnostic LLM: swap Gemini for Groq by changing one config value

## Environment variables
GOOGLE_API_KEY — primary LLM (Gemini 2.5 Flash)
GROQ_API_KEY — fallback LLM (Llama 3.3 70B)
TAVILY_API_KEY — web search
LANGFUSE_PUBLIC_KEY — observability
LANGFUSE_SECRET_KEY — observability

## Commands
- Start Streamlit app: streamlit run app/streamlit_app.py
- Start FastAPI: uvicorn app.api:app --reload
- Run evaluation: python evaluation/llm_judge.py
- Run tests: pytest tests/

## Coding conventions
- All functions must have type hints
- All modules must have docstrings
- Use async/await for all LLM and API calls
- Pydantic v2 models for all structured data
- Never hardcode API keys — always use os.getenv()
- Log all agent steps with Langfuse tracer

## Git workflow — mandatory after every change
After every meaningful change run:
    git add .
    git commit -m "type: description"
    git push origin main

Commit types: feat / fix / refactor / test / docs / chore

What counts as a meaningful change:
- Any new file created
- Any function or class completed
- Any bug fixed
- Any test written

## Build order — follow this exactly
1. ✅ schemas/report_schema.py — define ALL Pydantic models first
2. ✅ tools/web_search.py — Tavily search wrapper
3. ✅ tools/vector_retriever.py — ChromaDB retrieval tool
4. ✅ observability/langfuse_tracer.py — tracing setup
5. ✅ agents/news_analyst.py — first sub-agent
6. ✅ agents/academic_researcher.py — second sub-agent
7. ✅ agents/competitor_profiler.py — third sub-agent
8. ✅ agents/report_writer.py — synthesis agent
9. ✅ agents/orchestrator.py — LangGraph state machine wiring all agents
10. ✅ app/api.py — FastAPI endpoint
11. ✅ app/streamlit_app.py — Streamlit frontend
12. ✅ evaluation/llm_judge.py — quality scoring

## Session 1 — complete (2026-04-12)
### Tasks finished
- schemas/report_schema.py: SourceItem, DevelopmentItem, AcademicItem, CompetitorItem,
  ReportSchema (5 sections + total_sources_used + generation_time_seconds),
  NewsFindings, AcademicFindings, CompetitorFindings, AgentState TypedDict
- tools/web_search.py: async search() + sync search_sync() → list[SourceItem]
- tools/vector_retriever.py: VectorRetriever class with add_documents/search/clear_collection
- tests/test_tools.py: 26 unit tests — all passing

### Design decisions
- SourceItem is the atomic citation type; every agent output embeds list[SourceItem]
  instead of bare strings so URL + title + snippet travel together
- AgentState uses operator.add as reducer on all list fields so parallel nodes merge
  rather than overwrite each other's output
- NewsFindings/AcademicFindings/CompetitorFindings carry no query field — the query
  lives only in AgentState to keep agent outputs self-contained
- ReportSchema has no query field; query tracking stays in AgentState
- VectorRetriever runs embeddings in a thread-pool executor (never blocks event loop)
- Tavily SDK is synchronous; search() wraps it in run_in_executor for async callers

## Session 2 — complete (2026-04-12)
### Tasks finished
- observability/langfuse_tracer.py: get_tracer(), trace_agent() context manager,
  score_output(); _NullTracer/_NullTrace/_NullSpan fallback hierarchy
- agents/news_analyst.py: 5-step pattern (generate queries → parallel search →
  deduplicate → LLM structure → ChromaDB store); Langfuse-traced
- agents/academic_researcher.py: same pattern, targets papers/patents/reports
- tests/test_agents.py: 9 tests — all passing (35 total across both test files)

### Agent pattern established for Session 3 (competitor_profiler)
Every sub-agent follows this exact structure — copy this for competitor_profiler.py:
1. _build_llm() — ChatGoogleGenerativeAI from GOOGLE_API_KEY / GEMINI_MODEL env vars
2. _SearchQueriesOutput Pydantic model — for LLM query generation step
3. _Raw<Item> and _<Agent>LLMOutput — internal schemas for structured LLM summarisation
4. _deduplicate(sources) — URL-based dedup, order-preserving
5. _resolve_sources(urls, url_map, fallback) — maps LLM-cited URLs to SourceItems
6. async run(state) — wraps all steps in trace_agent(); catches all exceptions into
   state["errors"]; returns updated state with findings + all_sources appended

### Design decisions
- Internal LLM output schemas (_Raw*, _*LLMOutput) never leak outside each agent module;
  the public API always returns typed Pydantic schemas from schemas/report_schema.py
- Mock pattern for tests: set side_effect on structured_mock.ainvoke (not on
  structured_mock itself) because agents call .ainvoke(), not the mock directly
- Each agent uses its own named ChromaDB collection so search spaces don't mix
- trace_agent context manager is synchronous (contextmanager, not asynccontextmanager)
  because it just wraps async code, not awaits inside the context body itself

## Session 3 — complete (2026-04-12)
### Tasks finished
- agents/competitor_profiler.py: 4-query search pattern (competitors, comparisons,
  positioning, pricing); handles no-competitor case gracefully; 3 unit tests
- agents/report_writer.py: synthesis agent — RAG retrieval from all 3 ChromaDB
  collections, 2 LLM calls (executive_summary + strategic_recommendations),
  pass-through of already-structured sections, model_validate() for safety,
  score_output() quality marker, per-section fallback placeholders; 4 unit tests
- tests/test_agents.py: expanded to 16 tests covering all 4 sub-agents (42 total)

### All 4 sub-agents are complete and unit-tested
1. news_analyst — DevelopmentItem list, 3 parallel searches
2. academic_researcher — AcademicItem list, 3 parallel searches
3. competitor_profiler — CompetitorItem list, 4 parallel searches
4. report_writer — synthesises all findings into validated ReportSchema

### Design decisions (Session 3)
- Report writer makes exactly 2 LLM calls (executive_summary + recommendations);
  the 3 structured sections are pass-throughs from sub-agent output to avoid
  hallucination and stay under the 90-second budget
- RAG context retrieval in report_writer is best-effort: individual collection
  failures are silently skipped (debug-logged) so one missing collection doesn't
  block synthesis
- ReportSchema.model_validate({...}) used instead of direct constructor so that
  nested model dicts (from .model_dump()) are correctly coerced
- score_output() receives a section-completeness float (0.0–1.0) as initial
  quality signal; the LLM-as-judge evaluation pipeline in Session 6 will overwrite
  this with a semantic quality score

## Session 4 — complete (2026-04-12)
### Tasks finished
- agents/orchestrator.py: LangGraph StateGraph with parallel fan-out/fan-in,
  conditional early-exit routing, 25s per-node timeout, run_research() entry point
- tests/test_orchestrator.py: 6 integration tests — all passing (48 total)

### Graph structure
START → [news_analyst, academic_researcher, competitor_profiler] (parallel) →
check_node → (conditional) → report_writer → END
                           ↘ END (early if all agents fail)

### LangGraph-specific design decisions
- Node wrappers use _extract() to return ONLY owned keys — not {**state, ...}.
  Returning full state from parallel nodes causes InvalidUpdateError because
  non-annotated fields (query, start_time) can only be written once per step.
  AgentState list fields use operator.add reducer so parallel writes merge safely.
- Fan-in is implicit: LangGraph waits for all edges into 'check' before running it.
  No explicit join/barrier node is needed.
- _build_graph() is called fresh each run_research() invocation so node function
  references are resolved at call time — making agent functions trivially patchable
  in tests without needing to rebuild the graph.
- Conditional edge returns END sentinel directly (no path_map dict needed in
  this LangGraph version) when all sub-agents produced empty findings + errors.
- Timeout test uses monkeypatch on _AGENT_TIMEOUT module attribute (0.05s) with
  a 10s sleeping mock to trigger TimeoutError without real waiting.
- _AGENT_TIMEOUT raised from 25s to 60s after live testing revealed cold-start
  latency from sentence-transformers model loading exceeds 25s on first run.

## Session 5 — complete (2026-04-15)
### Tasks finished
- app/api.py: FastAPI backend with POST /research, GET /health, GET /report/{id};
  CORSMiddleware, request-logging middleware, global exception handler,
  OrderedDict LRU report cache (max 10); ResearchRequest/ResearchResponse Pydantic models
- app/streamlit_app.py: Streamlit frontend with text input + Generate Report button,
  5 expandable report sections (Executive Summary expanded by default, all others collapsed),
  per-item source badges, all-sources rollup, JSON download button; st.session_state management
- observability/langfuse_tracer.py: fixed for Langfuse v4.2.0 API
  (tracer.trace() → tracer.start_as_current_observation(as_type='agent'),
   tracer.score() → tracer.create_score(); added _ObservationCompat/_ChildSpanCompat wrappers
   to preserve trace.span() interface used by all 4 sub-agents)
- Live end-to-end tests: Anthropic (67s, 49 sources), Tesla autonomous driving (87s, 47 sources)
  — all 5 sections populated, no errors

### Langfuse v4.x design decisions
- v4 removed Langfuse.trace() — replacement is start_as_current_observation(as_type='agent')
  which returns an OTel-backed context manager; observation objects have .update()/.end()/.id
  but not .span() — added _ObservationCompat wrapper that translates .span(name, input) to
  observation.start_observation(name, as_type='span') so all existing agent code works unchanged
- _NullTracer path unchanged — if keys missing, trace_agent() yields _NullTrace immediately
  without touching the Langfuse SDK at all; all 48 tests still pass with null tracer

### How to run
- Start API:      uvicorn app.api:app --reload --port 8000
- Start frontend: streamlit run app/streamlit_app.py
- Both must be running simultaneously for the UI to work

## Session 6 — complete (2026-04-15)
### Tasks finished
- evaluation/llm_judge.py: LLMJudge class with score_report() and run_batch_evaluation();
  EvaluationResult Pydantic model (relevance / factual_grounding / completeness /
  actionability / overall_score / reasoning / report_generation_time);
  _JudgeOutput structured LLM output schema; _build_report_text() renderer;
  _print_summary() table; CLI entry point; results saved to evaluation/results.json
- evaluation/test_queries.json: 5 benchmark queries — OpenAI, Tesla, Anthropic, DeepMind, Mistral AI
- evaluation/results.json: batch evaluation output (OpenAI: 5.0/5.0; rest hit Gemini free-tier
  daily quota of 20 req/day — pipeline ran, judge quota exhausted)
- Dockerfile: python:3.11-slim base, requirements install, source copy, ChromaDB volume,
  EXPOSE 8000/8501, CMD uvicorn
- docker-compose.yml: api + frontend services, shared .env, chroma_data named volume,
  healthcheck on GET /health, frontend depends_on api healthy
- .env.example: safe-to-commit template with all 5 env vars documented
- README.md: Demo section with real OpenAI report output, Results table with evaluation
  scores, corrected Getting Started commands (venv, both services, Docker Compose option)
- Final test run: 48/48 passing

### Evaluation results (OpenAI, 2026-04-15)
- Relevance:          5.0 / 5.0
- Factual grounding:  5.0 / 5.0
- Completeness:       5.0 / 5.0
- Actionability:      5.0 / 5.0
- Overall:            5.0 / 5.0
- Report time:        64.9s
- Sources cited:      52
- Judge reasoning: "The report is highly relevant, focusing entirely on OpenAI's current
  status. Factual grounding is excellent with explicit source citations for nearly every
  claim. All five sections contain substantive and detailed content. Strategic
  recommendations are specific, actionable, and directly derived from the findings."

### Known limitations
- Gemini free tier: 20 requests/day — running the full 5-query evaluation suite requires
  a paid API key (or spread across multiple days)
- sentence-transformers cold-start: ~15-20s on first run; model is cached after first load
- ChromaDB persistence: in Docker the .chroma volume persists between runs; locally the
  .chroma/ folder in the project root is created automatically
- Tesla query occasionally exceeds 90s on slow connections due to 4 parallel search rounds

### Future improvements
- Cache sentence-transformers model in Docker image to eliminate cold-start
- Add streaming SSE endpoint to FastAPI so Streamlit can show per-agent progress
- Implement Groq fallback in report_writer when Gemini quota is exhausted
- Add evaluation/results.json comparison across runs (track score trends over time)