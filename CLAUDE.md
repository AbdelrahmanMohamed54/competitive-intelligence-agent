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
7. agents/competitor_profiler.py — third sub-agent
8. agents/report_writer.py — synthesis agent
9. agents/orchestrator.py — LangGraph state machine wiring all agents
10. app/api.py — FastAPI endpoint
11. app/streamlit_app.py — Streamlit frontend
12. evaluation/llm_judge.py — quality scoring

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