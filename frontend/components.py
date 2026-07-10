"""Reusable Streamlit presentation components."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly is optional.
    go = None  # type: ignore[assignment]

import streamlit as st

from utils.display import (
    confidence_bucket,
    confidence_summary,
    format_percent,
    normalize_text,
    relevant_citations,
    timestamp_label,
)


@dataclass
class WorkflowStep:
    """Display model for one workflow step."""

    name: str
    icon: str
    status: str = "pending"
    note: str = ""


def _section_title(title: str) -> None:
    """Render a compact, consistent section heading."""
    st.markdown(f"##### {title}")


def _card_label(title: str, icon: str | None = None) -> str:
    return f"{icon} {title}" if icon else title


def _accent_style(index: int) -> str:
    accents = ["#7a8f3a", "#5f7330", "#8da24a", "#6a7f37"]
    return accents[index % len(accents)]


def _step_short_label(name: str) -> str:
    aliases = {
        "Planner Agent": "Planner",
        "Gemini Vision Agent": "Vision",
        "Agronomy RAG Agent": "RAG",
        "Economic Agent": "Economic",
        "Market Agent": "Market",
        "Critic Agent": "Critic",
        "Memory Manager": "Memory",
        "Response Synthesizer": "Synth",
    }
    return aliases.get(name, name.replace(" Agent", ""))


def _status_chip(status: str) -> str:
    return {
        "running": "Running",
        "completed": "Completed",
        "skipped": "Skipped",
        "failed": "Failed",
        "done": "Completed",
        "active": "Running",
        "pending": "Pending",
    }.get(status, status.title())


def _status_dot(status: str) -> str:
    return {
        "running": "🟡",
        "completed": "🟢",
        "skipped": "⚪",
        "failed": "🔴",
        "done": "🟢",
        "active": "🟡",
        "pending": "⚪",
    }.get(status, "⚪")


def render_detail_grid(items: list[tuple[str, str, str | None]], columns: int = 2) -> None:
    """Render a consistent grid of compact detail cards."""
    rows = [items[idx : idx + columns] for idx in range(0, len(items), columns)]
    for row in rows:
        cols = st.columns(columns, gap="small")
        for col_idx, col in enumerate(cols):
            with col:
                if col_idx >= len(row):
                    continue
                label, value, icon = row[col_idx]
                with st.container(border=True):
                    st.caption(_card_label(label, icon))
                    st.caption(normalize_text(value) or "Unavailable")


def render_premium_metric_cards(cards: list[tuple[str, str, str, str | None]], columns: int = 4) -> None:
    """Render polished metric cards with a consistent product feel."""
    rows = [cards[idx : idx + columns] for idx in range(0, len(cards), columns)]
    for row in rows:
        cols = st.columns(columns, gap="small")
        for idx, col in enumerate(cols):
            with col:
                if idx >= len(row):
                    continue
                label, value, meta, icon = row[idx]
                with st.container(border=True):
                    st.markdown(f"**{_card_label(label, icon)}**")
                    st.markdown(f"#### {normalize_text(value) or '0'}")
                    st.caption(meta)


def _render_confidence_gauge(confidence: float) -> None:
    """Render a visual confidence gauge with a graceful fallback."""
    confidence = max(0.0, min(1.0, confidence))
    if go is not None:
        fig = go.Figure(
            go.Pie(
                values=[confidence * 100, max(0.0, 100 - confidence * 100)],
                hole=0.72,
                sort=False,
                direction="clockwise",
                marker={"colors": ["#7a8f3a", "#243326"], "line": {"color": "#0f1b16", "width": 1}},
                textinfo="none",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.update_layout(
            height=180,
            margin={"l": 8, "r": 8, "t": 8, "b": 8},
            paper_bgcolor="#0f1b16",
            font={"color": "#f5f2ea"},
            annotations=[
                {
                    "text": f"{format_percent(confidence)}<br><span style='font-size:12px'>confidence</span>",
                    "showarrow": False,
                    "font": {"size": 22, "color": "#f5f2ea"},
                }
            ],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        return

    filled = max(0, min(10, round(confidence * 10)))
    ring = "🟢" * filled + "⚪" * (10 - filled)
    st.markdown(f"**Confidence Ring**  \n`{ring}`")
    st.caption(f"{format_percent(confidence)} confidence")


def render_workflow_stepper(steps: list[WorkflowStep]) -> None:
    """Render the workflow progress as a compact horizontal stepper."""
    if not steps:
        st.caption("No workflow progress available.")
        return

    layout: list[float] = []
    for index in range(len(steps)):
        layout.append(1.0)
        if index < len(steps) - 1:
            layout.append(0.35)

    cols = st.columns(layout, gap="small")
    for index, step in enumerate(steps):
        step_col = cols[index * 2]
        with step_col:
            with st.container(border=True):
                badge = _status_chip(step.status)
                dot = {
                    "completed": "🟢",
                    "running": "🟢",
                    "skipped": "🟩",
                    "failed": "🟢",
                    "pending": "⚪",
                    "active": "🟢",
                    "done": "🟢",
                }.get(step.status, "🟢")
                st.markdown(dot)
                st.caption(_step_short_label(step.name))
                st.markdown(f":green[{badge}]")

        if index < len(steps) - 1:
            line_col = cols[index * 2 + 1]
            with line_col:
                connector = "━━" if step.status in {"completed", "running", "active", "done"} else "──"
                st.markdown(f":green[{connector}]")


def render_agent_list(steps: list[WorkflowStep]) -> None:
    """Render the workflow as a simple vertical list of cards."""
    if not steps:
        st.caption("No workflow progress available.")
        return

    for index, step in enumerate(steps):
        with st.container(border=True):
            cols = st.columns([0.12, 0.68, 0.20], gap="small")
            with cols[0]:
                st.markdown(_status_dot(step.status))
            with cols[1]:
                st.markdown(f"**{step.name}**")
                st.caption(step.note or "Waiting")
            with cols[2]:
                st.markdown(f":green[{_status_chip(step.status)}]")


def render_agent_timeline(steps: list[WorkflowStep], execution_meta: dict[str, str] | None = None) -> None:
    """Render the live workflow panel using native Streamlit blocks."""
    _section_title("AI Agent Status")
    step_cols = st.columns(2, gap="small")
    for index, step in enumerate(steps):
        col = step_cols[index % 2]
        with col:
            with st.container(border=True):
                cols = st.columns([0.12, 0.68, 0.20], gap="small")
                with cols[0]:
                    st.markdown(_status_dot(step.status))
                with cols[1]:
                    st.markdown(f"**{step.name}**")
                    st.caption(step.note or "Waiting")
                with cols[2]:
                    st.caption(_status_chip(step.status))

    if execution_meta:
        _section_title("Workflow Summary")
        summary_cards = [(label, value, "", None) for label, value in execution_meta.items()]
        render_premium_metric_cards(summary_cards, columns=2)


def render_workflow_summary(execution_meta: dict[str, str]) -> None:
    """Render the compact workflow summary card."""
    _section_title("Workflow Summary")
    items = list(execution_meta.items())
    if not items:
        with st.container(border=True):
            st.caption("No workflow data available.")
        return

    render_detail_grid([(label, value, None) for label, value in items], columns=2)


def render_confidence_card(confidence: float) -> None:
    """Render a simple confidence card with native widgets."""
    band = confidence_bucket(confidence)
    percent = format_percent(confidence)
    threshold = "Threshold Passed" if confidence >= 0.8 else "Below Threshold"

    _section_title("Confidence")
    gauge_cols = st.columns([0.55, 0.45], gap="small")
    with gauge_cols[0]:
        _render_confidence_gauge(confidence)
    with gauge_cols[1]:
        with st.container(border=True):
            st.caption("Confidence")
            st.markdown(f"**{percent}**")
            st.caption("Band")
            st.markdown(f"**{band}**")
            st.caption(f"{band} · {threshold}")


def render_evidence_cards(
    citations: list[dict[str, str]],
    focus_disease: str | None = None,
    focus_crop: str | None = None,
) -> None:
    """Render retrieved evidence in clean citation cards."""
    if not citations:
        with st.container(border=True):
            st.caption("Evidence")
            st.write("No retrieved documents were available for this case.")
        return

    matched_citations = relevant_citations(citations, focus_disease, focus_crop)
    if focus_disease and not matched_citations:
        with st.container(border=True):
            st.caption("Evidence")
            st.write(
                f"No relevant documents met the retrieval threshold for {normalize_text(focus_disease).replace('_', ' ').title()}, "
                "so the related documents were hidden to avoid misleading support."
            )
        return

    display_citations = matched_citations or relevant_citations(citations)
    for citation in display_citations:
        title = citation.get("title", citation.get("source", "Document"))
        source = citation.get("source", "Unknown")
        page = citation.get("page", "Unknown")
        score = normalize_text(citation.get("score", citation.get("retrieval_score", "n/a")))
        snippet = normalize_text(citation.get("snippet", ""))
        disease = normalize_text(citation.get("disease", ""))
        if len(snippet) > 260:
            snippet = snippet[:257].rstrip() + "..."

        with st.container(border=True):
            meta_cols = st.columns([0.12, 0.58, 0.30], gap="small")
            with meta_cols[0]:
                st.markdown("📘")
            with meta_cols[1]:
                st.markdown(f"**{title}**")
                st.caption(f"{source} · Page {page}")
            with meta_cols[2]:
                st.caption(f"Retrieval score: {score}")
                if disease:
                    st.caption(f"Matched disease: {disease}")
            st.write(snippet or "No summary available.")


def render_kv_grid(items: list[tuple[str, str]], columns: int = 2) -> None:
    """Render a consistent key/value grid without metric clutter."""
    rows = [items[idx : idx + columns] for idx in range(0, len(items), columns)]
    for row in rows:
        cols = st.columns(columns, gap="small")
        for col_idx, col in enumerate(cols):
            with col:
                if col_idx < len(row):
                    label, value = row[col_idx]
                    display_value = value or "Unavailable"
                    st.metric(label, display_value)
