# Autonomous Research & Competitive Intelligence Agent

> A production-grade multi-agent system built with LangGraph that autonomously researches any company or topic and delivers a structured, source-grounded competitive intelligence report in under 90 seconds.

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
| **Report Writer** | Synthesizes all agent outputs into a structured 5-section report | Claude API, Pydantic schemas |

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

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph (LangChain) |
| LLM Backbone | Claude API (Anthropic) |
| Web Search Tool | Tavily Search API |
| Vector Store | ChromaDB |
| Output Validation | Pydantic v2 |
| Observability | Langfuse |
| Frontend | Streamlit |
| API | FastAPI |

---

## Project Structure

```
competitive-intelligence-agent/
├── agents/
│   ├── orchestrator.py       # LangGraph orchestrator + state machine
│   ├── news_analyst.py       # News research sub-agent
│   ├── academic_researcher.py
│   ├── competitor_profiler.py
│   └── report_writer.py      # Report synthesis agent
├── schemas/
│   └── report_schema.py      # Pydantic output models for all agents
├── tools/
│   ├── web_search.py         # Tavily search tool wrapper
│   └── vector_retriever.py   # ChromaDB retrieval tool
├── observability/
│   └── langfuse_tracer.py    # Langfuse integration
├── app/
│   ├── streamlit_app.py      # Streamlit frontend
│   └── api.py                # FastAPI backend
├── evaluation/
│   └── llm_judge.py          # LLM-as-judge evaluation pipeline
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Anthropic API key
- Tavily API key (free tier: 1,000 searches/month)
- Langfuse account (free tier available)

### Installation

```bash
git clone https://github.com/AbdelrahmanMohamed54/competitive-intelligence-agent
cd competitive-intelligence-agent
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, TAVILY_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
pip install -r requirements.txt
```

### Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### Run via API

```bash
uvicorn app.api:app --reload
# POST /research {"query": "OpenAI", "depth": "full"}
```

### Run evaluation suite

```bash
python evaluation/llm_judge.py --queries evaluation/test_queries.json
```

---

## Observability

Every research run is traced in Langfuse with:

- Per-agent token usage and cost
- Per-step latency breakdown
- LLM-as-judge quality score (1–5) for each report section
- Full prompt/response logs for debugging

Access your Langfuse dashboard at `https://cloud.langfuse.com` after setting API keys.

---

## Example Output

```
Query: "Anthropic"
Total time: 73 seconds
Sections generated: 5
Sources cited: 18
LLM-as-judge score: 4.6 / 5.0
Token cost: ~$0.18
```

---

## Author

**Abdelrahman Mohamed** — AI Engineer  
[LinkedIn](https://linkedin.com/in/abdelrahman-mohamed) · [GitHub](https://github.com/AbdelrahmanMohamed54)  
abdelrahman.mohamed2505@gmail.com
