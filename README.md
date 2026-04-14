# Autonomous Research & Competitive Intelligence Agent

> A production-grade multi-agent system built with LangGraph that autonomously researches any company or topic and delivers a structured, source-grounded competitive intelligence report in under 90 seconds.

---

## Demo

**Query:** `"OpenAI"`  
**Time:** 64.9s &nbsp;|&nbsp; **Sources:** 52 &nbsp;|&nbsp; **LLM-as-judge score:** 5.0 / 5.0

```
=== EXECUTIVE SUMMARY ===
OpenAI is strategically positioning itself at the forefront of artificial general
intelligence development, marked by the highly anticipated release of GPT-5 and a
significant enterprise push. The company is navigating a complex landscape of rapid
technological advancement, intensifying competition, and increasing regulatory scrutiny.
Financially, OpenAI has secured massive funding and is aggressively expanding its
commercial offerings, including a potential integration with Apple devices and a
restructuring to a for-profit model to sustain its capital-intensive research.

=== RECENT DEVELOPMENTS (5 items) ===
- [2025-05] GPT-5 Release Imminent, Promising Significant Leap (3 sources)
- [2025-03] OpenAI Raises $40B at $300B Valuation (2 sources)
- [2025-04] OpenAI Restructures to For-Profit Benefit Corporation (2 sources)
- [2025-02] OpenAI Partners with Apple for Siri Integration (2 sources)
- [2025-04] FTC Investigation into OpenAI's Data Practices (2 sources)

=== ACADEMIC LANDSCAPE (5 papers) ===
- Advances in Reinforcement Learning from Human Feedback (RLHF) (3 sources)
- Scaling Laws for Neural Language Models (2 sources)
- Constitutional AI: Harmlessness from AI Feedback (3 sources)
- Sparks of Artificial General Intelligence: Early experiments with GPT-4 (2 sources)
- GPT-4 Technical Report (2 sources)

=== COMPETITIVE ANALYSIS (6 competitors) ===
- Anthropic: 3 strengths, 2 weaknesses (3 sources)
- Google DeepMind: 4 strengths, 2 weaknesses (3 sources)
- Meta AI: 3 strengths, 2 weaknesses (2 sources)
- Mistral AI: 3 strengths, 2 weaknesses (2 sources)
- xAI (Grok): 3 strengths, 2 weaknesses (2 sources)
- Cohere: 3 strengths, 2 weaknesses (2 sources)

=== STRATEGIC RECOMMENDATIONS (6 items) ===
1. Accelerate GPT-5 rollout with a phased enterprise-first strategy ...
2. Strengthen safety and alignment research to pre-empt regulatory risk ...
3. Diversify revenue beyond API access via vertical SaaS products ...
...
```

---

## Overview

Manual competitive research is slow, inconsistent, and expensive. This system replaces it with a **LangGraph-orchestrated multi-agent pipeline** where specialized sub-agents handle discrete research tasks in parallel — news analysis, academic research, competitor profiling, and report writing — each grounded in retrieved source documents to eliminate hallucination.

Every claim in the final report is traceable to a source. Full observability (latency, token cost, quality scores) is tracked via Langfuse.

---

## Architecture

```
User Input: Company / Topic
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                     │
│         (LangGraph — plans, routes, supervises)           │
└──────┬──────────┬──────────┬──────────┬─────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐
  │  News   │ │Academic │ │Competi- │ │   Report    │
  │Analyst  │ │Research │ │tor Pro- │ │   Writer    │
  │ Agent   │ │  Agent  │ │ filer   │ │   Agent     │
  └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘
       │           │           │              │
       ▼           ▼           ▼              ▼
  Web Search   Web Search   Web Search   Synthesizes all
  + RAG store  + RAG store  + RAG store  agent outputs
  (Tavily)     (Tavily)     (Tavily)     into structured
  ChromaDB     ChromaDB     ChromaDB     Pydantic report
       │           │           │              │
       └───────────┴───────────┴──────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Pydantic Report     │
              │   (5 structured       │
              │    sections +         │
              │    source citations)  │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Langfuse Observability│
              │  latency · cost ·      │
              │  LLM-as-judge score   │
              └───────────┬───────────┘
                          │
                          ▼
              Streamlit Web Application
```

---

## Agent Roles

| Agent | Responsibility | Tools |
|---|---|---|
| **Orchestrator** | Decomposes query, routes to sub-agents, synthesizes final output | LangGraph state machine |
| **News Analyst** | Finds and summarizes recent news, press releases, developments | Tavily web search, ChromaDB |
| **Academic Researcher** | Searches papers, patents, technical publications | Tavily search, ChromaDB |
| **Competitor Profiler** | Maps competitive landscape, products, pricing, positioning | Tavily search, ChromaDB |
| **Report Writer** | Synthesizes all agent outputs into a structured 5-section report | Gemini 2.5 Flash, Pydantic schemas |

---

## Key Features

- **LangGraph state machine orchestration** — deterministic agent routing with typed state, conditional edges, and error recovery
- **RAG-grounded outputs** — all report claims are backed by retrieved source documents stored in ChromaDB; no unsupported assertions
- **Structured Pydantic schemas** — each agent emits typed output that downstream agents consume without parsing failures
- **Full observability** — Langfuse traces every LLM call, tool invocation, latency, token count, and cost; LLM-as-judge quality scores on each report section
- **Sub-90-second end-to-end** — parallel sub-agent execution reduces total wall time dramatically vs sequential research
- **Source citations in output** — final report includes numbered citations mapped to retrieved URLs

---

## Report Output Structure

Each generated report contains 5 sections:

1. **Executive Summary** — 3-paragraph overview of key findings
2. **Recent Developments** — timestamped news and announcements (last 90 days)
3. **Academic & Technical Landscape** — relevant research, patents, technical publications
4. **Competitive Analysis** — market positioning, key competitors, product comparison
5. **Strategic Recommendations** — actionable insights grounded in retrieved evidence

---

## Results — Evaluation Suite

Evaluated with LLM-as-judge (Gemini 2.5 Flash, 1–5 scale):

| Query | Relevance | Grounding | Completeness | Actionability | Avg | Time |
|---|---|---|---|---|---|---|
| OpenAI | 5.0 | 5.0 | 5.0 | 5.0 | **5.00** | 64.9s |
| Anthropic | — | — | — | — | — | 67.5s |
| Tesla | — | — | — | — | — | 87.1s |
| DeepMind | — | — | — | — | — | 35.6s |
| Mistral AI | — | — | — | — | — | 35.6s |

> Note: Gemini free tier limited to 20 requests/day. Reports were generated successfully for all 5 queries; only the OpenAI judge evaluation completed before the daily quota was reached. The pipeline and judge code are fully functional — run with a paid API key for full batch results.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph (LangChain) |
| LLM Backbone | Google Gemini 2.5 Flash |
| Fallback LLM | Groq Llama 3.3 70B |
| Web Search Tool | Tavily Search API |
| Vector Store | ChromaDB + sentence-transformers |
| Output Validation | Pydantic v2 |
| Observability | Langfuse v4 |
| Frontend | Streamlit |
| API | FastAPI |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
competitive-intelligence-agent/
├── agents/
│   ├── orchestrator.py        # LangGraph orchestrator + state machine
│   ├── news_analyst.py        # News research sub-agent
│   ├── academic_researcher.py # Academic research sub-agent
│   ├── competitor_profiler.py # Competitive landscape sub-agent
│   └── report_writer.py       # Report synthesis agent
├── schemas/
│   └── report_schema.py       # Pydantic output models for all agents
├── tools/
│   ├── web_search.py          # Tavily search tool wrapper
│   └── vector_retriever.py    # ChromaDB retrieval tool
├── observability/
│   └── langfuse_tracer.py     # Langfuse v4 integration
├── app/
│   ├── streamlit_app.py       # Streamlit frontend
│   └── api.py                 # FastAPI backend
├── evaluation/
│   ├── llm_judge.py           # LLM-as-judge evaluation pipeline
│   ├── test_queries.json      # 5 benchmark queries
│   └── results.json           # Evaluation results
├── tests/
│   ├── test_tools.py          # 26 unit tests
│   ├── test_agents.py         # 16 unit tests
│   └── test_orchestrator.py   # 6 integration tests
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Google AI Studio API key (free at [aistudio.google.com](https://aistudio.google.com/app/apikey))
- Tavily API key (free tier: 1,000 searches/month at [app.tavily.com](https://app.tavily.com))
- Langfuse account (optional — free tier at [cloud.langfuse.com](https://cloud.langfuse.com))

### Installation

```bash
git clone https://github.com/AbdelrahmanMohamed54/competitive-intelligence-agent
cd competitive-intelligence-agent
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and fill in: GOOGLE_API_KEY, TAVILY_API_KEY
```

### Run the full stack

**Option 1 — Local (two terminals):**

```bash
# Terminal 1 — FastAPI backend
uvicorn app.api:app --reload --port 8000

# Terminal 2 — Streamlit frontend
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Option 2 — Docker Compose:**

```bash
docker compose up --build
```

- API: [http://localhost:8000](http://localhost:8000)
- UI:  [http://localhost:8501](http://localhost:8501)

### Run tests

```bash
pytest tests/ -v
# 48 tests, all passing
```

### Run evaluation suite

```bash
python evaluation/llm_judge.py
# Evaluates 5 benchmark queries; results saved to evaluation/results.json
```

### Use the API directly

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "OpenAI", "depth": "full"}'
```

---

## Observability

Every research run is traced in Langfuse with:

- Per-agent token usage and cost
- Per-step latency breakdown
- LLM-as-judge quality score (1–5) for each report section
- Full prompt/response logs for debugging

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to enable.  
Dashboard: [https://cloud.langfuse.com](https://cloud.langfuse.com)

---

## Author

**Abdelrahman Mohamed** — AI Engineer  
[LinkedIn](https://linkedin.com/in/abdelrahman-25-mohamed) · [GitHub](https://github.com/AbdelrahmanMohamed54)  
abdelrahman.mohamed2505@gmail.com
