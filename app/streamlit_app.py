"""
Streamlit frontend for the Competitive Intelligence Agent.

Calls the FastAPI backend at http://localhost:8000/research and displays
the structured report in an interactive, expandable layout.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_BASE = "http://localhost:8000"
_REQUEST_TIMEOUT = 120  # seconds — matches pipeline budget

st.set_page_config(
    page_title="Competitive Intelligence Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "report_query" not in st.session_state:
    st.session_state.report_query = ""
if "report_id" not in st.session_state:
    st.session_state.report_id = ""
if "generation_error" not in st.session_state:
    st.session_state.generation_error = ""

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Competitive Intelligence Agent")
st.markdown("**Powered by LangGraph + Gemini** — Autonomous multi-agent research pipeline")
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

col_input, col_btn = st.columns([5, 1])

with col_input:
    query = st.text_input(
        "Enter a company or topic to research",
        placeholder="e.g. Anthropic, Tesla autonomous driving, OpenAI GPT-5",
        label_visibility="collapsed",
    )

with col_btn:
    generate_clicked = st.button("Generate Report", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

if generate_clicked and query.strip():
    st.session_state.generation_error = ""
    st.session_state.report_data = None

    with st.spinner("Researching… this typically takes 60–90 seconds"):
        try:
            t0 = time.perf_counter()
            response = requests.post(
                f"{_API_BASE}/research",
                json={"query": query.strip(), "depth": "full"},
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            st.session_state.report_data = payload
            st.session_state.report_query = query.strip()
            st.session_state.report_id = payload.get("report_id", "")
        except requests.Timeout:
            st.session_state.generation_error = (
                "The request timed out after 120 seconds. "
                "The pipeline may still be running — try again shortly."
            )
        except requests.ConnectionError:
            st.session_state.generation_error = (
                "Could not connect to the backend. "
                "Make sure the API server is running:  "
                "`uvicorn app.api:app --reload --port 8000`"
            )
        except requests.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail") or exc.response.json().get("error")
            except Exception:
                detail = str(exc)
            st.session_state.generation_error = f"API error: {detail}"
        except Exception as exc:
            st.session_state.generation_error = f"Unexpected error: {exc}"

elif generate_clicked and not query.strip():
    st.warning("Please enter a company or topic before generating a report.")

# ---------------------------------------------------------------------------
# Error display
# ---------------------------------------------------------------------------

if st.session_state.generation_error:
    st.error(st.session_state.generation_error)

# ---------------------------------------------------------------------------
# Report display
# ---------------------------------------------------------------------------


def _render_source_badge(sources: list[dict[str, Any]]) -> None:
    """Render a compact list of source links inside an expander."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for src in sources:
            url = src.get("url", "#")
            title = src.get("title") or url
            snippet = src.get("snippet", "")
            st.markdown(f"- [{title}]({url})")
            if snippet:
                st.caption(snippet[:160] + ("…" if len(snippet) > 160 else ""))


def _render_report(payload: dict[str, Any]) -> None:
    """Render a complete research report from the API response payload."""
    report: dict[str, Any] = payload.get("report", {})
    query_text: str = payload.get("query", "")
    report_id: str = payload.get("report_id", "")
    gen_time: float = report.get("generation_time_seconds", 0.0)
    total_sources: int = report.get("total_sources_used", 0)

    # ---- Meta bar ----
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    with meta_col1:
        st.metric("Report generated in", f"{gen_time:.1f}s")
    with meta_col2:
        st.metric("Sources cited", total_sources)
    with meta_col3:
        if report_id:
            st.caption(f"Report ID: `{report_id}`")

    st.divider()

    # ---- 1. Executive Summary ----
    with st.expander("Executive Summary", expanded=True):
        summary = report.get("executive_summary", "")
        if summary:
            st.markdown(summary)
        else:
            st.info("No executive summary available.")

    # ---- 2. Recent Developments ----
    developments: list[dict] = report.get("recent_developments", [])
    with st.expander(f"Recent Developments  ({len(developments)} items)", expanded=False):
        if developments:
            for item in developments:
                with st.container(border=True):
                    st.markdown(f"**{item.get('title', 'Untitled')}**  ·  {item.get('date', '')}")
                    st.write(item.get("summary", ""))
                    _render_source_badge(item.get("sources", []))
        else:
            st.info("No recent developments found.")

    # ---- 3. Academic & Technical Landscape ----
    academic: list[dict] = report.get("academic_landscape", [])
    with st.expander(f"Academic & Technical Landscape  ({len(academic)} papers)", expanded=False):
        if academic:
            for item in academic:
                with st.container(border=True):
                    authors = ", ".join(item.get("authors", [])) or "Unknown authors"
                    url = item.get("url", "")
                    title = item.get("title", "Untitled")
                    title_md = f"**[{title}]({url})**" if url else f"**{title}**"
                    st.markdown(f"{title_md}  ·  {authors}")
                    st.write(item.get("summary", ""))
                    _render_source_badge(item.get("sources", []))
        else:
            st.info("No academic or technical publications found.")

    # ---- 4. Competitive Analysis ----
    competitors: list[dict] = report.get("competitive_analysis", [])
    with st.expander(f"Competitive Analysis  ({len(competitors)} companies)", expanded=False):
        if competitors:
            for item in competitors:
                with st.container(border=True):
                    st.markdown(f"### {item.get('name', 'Unknown')}")
                    st.write(item.get("description", ""))

                    col_str, col_weak = st.columns(2)
                    with col_str:
                        st.markdown("**Strengths**")
                        strengths = item.get("strengths", [])
                        if strengths:
                            for s in strengths:
                                st.markdown(f"- {s}")
                        else:
                            st.caption("None identified")
                    with col_weak:
                        st.markdown("**Weaknesses**")
                        weaknesses = item.get("weaknesses", [])
                        if weaknesses:
                            for w in weaknesses:
                                st.markdown(f"- {w}")
                        else:
                            st.caption("None identified")

                    _render_source_badge(item.get("sources", []))
        else:
            st.info("No competitors identified.")

    # ---- 5. Strategic Recommendations ----
    recommendations: list[str] = report.get("strategic_recommendations", [])
    with st.expander(f"Strategic Recommendations  ({len(recommendations)})", expanded=False):
        if recommendations:
            for i, rec in enumerate(recommendations, start=1):
                st.markdown(f"**{i}.** {rec}")
        else:
            st.info("No strategic recommendations generated.")

    st.divider()

    # ---- All cited sources ----
    all_source_urls: set[str] = set()
    all_source_items: list[dict] = []
    for section_key in ("recent_developments", "academic_landscape", "competitive_analysis"):
        for item in report.get(section_key, []):
            for src in item.get("sources", []):
                if src.get("url") and src["url"] not in all_source_urls:
                    all_source_urls.add(src["url"])
                    all_source_items.append(src)

    if all_source_items:
        with st.expander(f"All Cited Sources  ({len(all_source_items)})", expanded=False):
            for src in all_source_items:
                url = src.get("url", "#")
                title = src.get("title") or url
                st.markdown(f"- [{title}]({url})")

    # ---- Download button ----
    st.download_button(
        label="Download Report as JSON",
        data=json.dumps(payload, indent=2, default=str),
        file_name=f"report_{query_text.replace(' ', '_')[:40]}.json",
        mime="application/json",
        use_container_width=False,
    )


if st.session_state.report_data:
    _render_report(st.session_state.report_data)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Competitive Intelligence Agent · LangGraph + Gemini 2.5 Flash · "
    "Sources grounded via Tavily + ChromaDB RAG"
)
