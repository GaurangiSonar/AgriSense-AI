"""Streamlit entry point for AgriSense AI."""

from __future__ import annotations

import hashlib
import base64
from datetime import datetime
from pathlib import Path

try:
    import streamlit as st
except Exception:  # pragma: no cover - Streamlit may be absent in non-UI test runs.
    st = None  # type: ignore[assignment]

from agents.planner import PlannerAgent
from agents.state import AgriSenseState
from frontend.components import (
    WorkflowStep,
    render_confidence_card,
    render_evidence_cards,
    render_detail_grid,
    render_workflow_summary,
)
from utils.display import confidence_bucket, format_crop_label, format_disease_label, format_percent, relevant_citations
from utils.economics import build_economic_analysis_items
from utils.report_pdf import generate_pdf_report


WORKFLOW_STEPS = [
    ("Planner Agent", "🧠"),
    ("Gemini Vision Agent", "👁"),
    ("Agronomy RAG Agent", "📚"),
    ("Economic Agent", "💰"),
    ("Market Agent", "📈"),
    ("Critic Agent", "🛡"),
    ("Memory Manager", "🧠"),
    ("Response Synthesizer", "📄"),
]

CROP_OPTIONS = [
    "Tomato",
    "Potato",
    "Pepper",
    "Chili",
    "Brinjal",
    "Cucumber",
    "Wheat",
    "Rice",
    "Maize",
    "Cotton",
    "Other",
]


def _workflow_nodes() -> list[WorkflowStep]:
    return [WorkflowStep(name=name, icon=icon) for name, icon in WORKFLOW_STEPS]


def _agent_skip_reason(agent_name: str, state: AgriSenseState) -> str:
    """Return a human-readable reason when an agent did not run."""
    if state.needs_clarification:
        return f"{agent_name} skipped — diagnosis requires clarification"
    if state.diagnosis_validation_error or state.vision_error:
        return f"{agent_name} skipped — workflow stopped after an earlier failure"

    disease = (state.disease or "").lower()
    healthy_crop = "healthy" in disease or state.treatment == "No treatment required"
    if healthy_crop and agent_name in {
        "Agronomy RAG Agent",
        "Market Agent",
        "Economic Agent",
        "Memory Manager",
    }:
        return f"{agent_name} skipped — healthy crop detected"

    if agent_name == "Memory Manager":
        return "Memory Manager skipped — personalization unavailable"

    if agent_name == "Market Agent" and state.market_error:
        return "Market Agent skipped — market API unavailable"

    return f"{agent_name} skipped"


def _workflow_meta(state: AgriSenseState) -> dict[str, str]:
    meta = {
        "Execution Time": f"{state.execution_time_seconds:.1f}s" if state.execution_time_seconds else "0.0s",
        "Agents Executed": str(state.agents_executed),
        "Documents Retrieved": str(state.documents_retrieved),
        "Workflow Status": state.workflow_status.replace("_", " ").title(),
    }
    if state.skipped_agents:
        meta["Agents Skipped"] = str(len(state.skipped_agents))
        meta["Skip Reason"] = "; ".join(_agent_skip_reason(agent, state) for agent in state.skipped_agents)
    return meta


def _workflow_nodes_from_state(state: AgriSenseState) -> list[WorkflowStep]:
    executed = {name for name in state.executed_agents}
    skipped = {name for name in state.skipped_agents}
    nodes = _workflow_nodes()
    for node in nodes:
        if node.name in skipped:
            node.status = "skipped"
            node.note = "Skipped"
        elif node.name in executed:
            node.status = "completed"
            node.note = _node_note(node.name, "completed", state)
        else:
            node.status = "pending"
            node.note = "Waiting"
    return nodes


def _build_input_key(uploaded_image, crop_name: str, symptoms: str, farmer_id: str) -> str:
    digest = hashlib.sha256()
    digest.update(uploaded_image.getvalue())
    digest.update(crop_name.encode("utf-8"))
    digest.update(symptoms.encode("utf-8"))
    digest.update(farmer_id.encode("utf-8"))
    return digest.hexdigest()


def _save_uploaded_image(uploaded_image) -> Path:
    upload_dir = Path("work")
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / uploaded_image.name
    image_path.write_bytes(uploaded_image.getbuffer())
    return image_path


def _render_hero() -> None:
    st.title("🌾AgriSense AI")
    st.subheader("Agentic AI Powered Crop Disease Diagnosis & Decision Support Platform")


def _section_title(title: str) -> None:
    """Render a compact, consistent section heading."""
    st.markdown(f"##### {title}")

@st.cache_resource
def _load_base64_asset(path: str) -> str:
    return base64.b64encode(
        Path(path).read_bytes()
    ).decode("utf-8")

def _inject_custom_css() -> None:

    css_path = Path(__file__).parent / "frontend" / "styles.css"

    if css_path.exists():
        css_text = css_path.read_text(encoding="utf-8")
        st.markdown(
            f"<style>{css_text}</style>",
            unsafe_allow_html=True,
        )

    bg_path = (
        Path(__file__).parent
        / "frontend"
        / "assets"
        / "farm-bg.png"
    )

    if not bg_path.exists():
        return

    b64_image = _load_base64_asset(str(bg_path))

    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            background-image:
                linear-gradient(
                    180deg,
                    rgba(14, 26, 18, 0.88) 0%,
                    rgba(14, 26, 18, 0.85) 45%,
                    rgba(14, 26, 18, 0.9) 100%
                ),
                url("data:image/jpeg;base64,{b64_image}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_upload_card() -> tuple[object | None, str, str, str, bool]:
    with st.container(border=True):
        _section_title("Upload Crop Image")
        st.caption("Choose a crop image and provide a little context so the analysis stays grounded.")
        with st.form("agrisense-analysis-form", clear_on_submit=False):
            uploaded_image = st.file_uploader(
                "Choose crop image",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible",
            )
            crop_choice = st.selectbox("Crop", CROP_OPTIONS, index=0)
            other_crop = ""
            if crop_choice == "Other":
                other_crop = st.text_input("Custom crop name", placeholder="Enter crop name")
            symptoms = st.text_area("Symptoms", placeholder="Describe visible symptoms...", height=128)
            with st.expander("Additional context", expanded=False):
                st.text_input("Crop area / farm notes", placeholder="Optional context for the recommendation")
            submitted = st.form_submit_button("Analyze Crop", use_container_width=True)

    crop_name = other_crop if crop_choice == "Other" else crop_choice
    farmer_id = ""
    return uploaded_image, crop_name, symptoms, farmer_id, submitted


def _render_progress_indicator(placeholder, steps: list[WorkflowStep]) -> None:
    """Keep workflow progress hidden while the analysis runs."""
    _ = placeholder, steps


def _render_result_sections(result: AgriSenseState, pdf_bytes: bytes) -> None:
    _section_title("Analysis Dashboard")

    row1 = st.columns(3, gap="small")
    with row1[0]:
        with st.container(border=True):
            _section_title("Diagnosis")
            render_detail_grid(
                [
                    ("Crop", format_crop_label(result.crop), "🌱"),
                    ("Disease", format_disease_label(result.disease), "🧬"),
                ],
                columns=2,
            )

    with row1[1]:
        with st.container(border=True):
            render_confidence_card(result.confidence)

    with row1[2]:
        with st.container(border=True):
            render_workflow_summary(_workflow_meta(result))

    row2 = st.columns(2, gap="small")
    with row2[0]:
        with st.container(border=True):
            _section_title("Treatment")
            render_detail_grid(
                [
                    ("Primary Treatment", result.treatment or "Unavailable", "🧴"),
                    ("Dosage", result.dosage or "Unavailable", "💧"),
                    ("Frequency", result.frequency or "Unavailable", "⏱"),
                    ("Recovery", result.expected_recovery or "Unavailable", "🌿"),
                ],
                columns=2,
            )
            with st.container(border=True):
                st.caption("Prevention")
                st.write(result.prevention or "Unavailable")

    with row2[1]:
        with st.container(border=True):
            _section_title("Economic Analysis")
            if result.economics_estimated:
                st.caption("Market data is unavailable. Market value, net profit, and ROI use benchmark estimates.")
            render_detail_grid(
                build_economic_analysis_items(
                    treatment_cost=result.treatment_cost,
                    estimated_yield_recovered=result.estimated_yield_recovered,
                    market_value_recovered=result.market_value_recovered,
                    net_profit=result.net_profit,
                    roi=result.roi,
                    economics_estimated=result.economics_estimated,
                ),
                columns=2,
            )

    row4 = st.columns(2, gap="small")
    with row4[0]:
        with st.container(border=True):
            _section_title("Evidence Used")
            render_evidence_cards(result.citations, result.disease, result.crop)

    with row4[1]:
        with st.container(border=True):
            _section_title("Why This Recommendation")
            matched_evidence_count = len(relevant_citations(result.citations, result.disease, result.crop))
            sections = [
                (
                    "Confidence",
                    f"{format_percent(result.confidence)} confidence ({confidence_bucket(result.confidence)}) guided the recommendation.",
                ),
                (
                    "Treatment",
                    f"{result.treatment or 'Unavailable'} was selected as the primary treatment path.",
                ),
                (
                    "Evidence",
                    (
                        f"{matched_evidence_count} relevant document(s) met the retrieval threshold."
                        if matched_evidence_count
                        else "No relevant documents met the retrieval threshold, so related documents were hidden."
                    ),
                ),
            ]
            for label, body in sections:
                with st.expander(label, expanded=False):
                    st.write(body)

    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"agrisense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=False,
    )


def main() -> None:
    """Render the polished Streamlit app."""
    if st is None:
        print("Streamlit is not installed. Install requirements to run the UI.")
        return

    st.set_page_config(
        page_title="AgriSense AI",
        page_icon="🌾",
        layout="wide",
    )
    _inject_custom_css()

    if "agrisense_result" not in st.session_state:
        st.session_state.agrisense_result = None
    if "agrisense_error" not in st.session_state:
        st.session_state.agrisense_error = None
    if "agrisense_input_key" not in st.session_state:
        st.session_state.agrisense_input_key = None

    _render_hero()

    uploaded_image, crop_name, symptoms, farmer_id, submitted = _render_upload_card()
    workflow_nodes = _workflow_nodes()
    status_placeholder = st.empty()

    current_key = (
        _build_input_key(uploaded_image, crop_name, symptoms, farmer_id)
        if uploaded_image
        else None
    )

    if not uploaded_image:
        if st.session_state.agrisense_error:
            st.error(st.session_state.agrisense_error)
        st.caption("Upload a crop image to run the analysis workflow and generate the dashboard below.")
        _inject_custom_css()
        return

    cached_result = (
        st.session_state.agrisense_result
        if st.session_state.agrisense_input_key == current_key and isinstance(st.session_state.agrisense_result, AgriSenseState)
        else None
    )

    if submitted:
        image_path = _save_uploaded_image(uploaded_image)
        state = AgriSenseState(
            image_path=str(image_path),
            crop_name_optional=crop_name or None,
            symptom_description_optional=symptoms or None,
            farmer_id=farmer_id or None,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        progress_placeholder = st.empty()
        status_state = _workflow_nodes()
        status_index = {step.name: idx for idx, step in enumerate(status_state)}

        def _render_live_status(current_state: AgriSenseState) -> None:
            _ = current_state
            status_placeholder.empty()

        def progress_callback(agent_name: str, status: str, current_state: AgriSenseState) -> None:
            if agent_name in status_index:
                node = status_state[status_index[agent_name]]
                node.status = status
                node.note = _node_note(agent_name, status, current_state)
            _render_progress_indicator(progress_placeholder, status_state)
            _render_live_status(current_state)

        planner = PlannerAgent()
        with st.spinner("Analyzing crop..."):
            result = planner.execute(state, progress_callback=progress_callback)

        _render_progress_indicator(progress_placeholder, status_state)
        _render_live_status(result)

        if result.diagnosis_validation_error:
            st.session_state.agrisense_error = result.diagnosis_validation_error
            st.session_state.agrisense_result = None
            st.session_state.agrisense_input_key = current_key
            st.error(result.diagnosis_validation_error)
            return

        if result.vision_error:
            st.session_state.agrisense_error = result.vision_error
            st.session_state.agrisense_result = None
            st.session_state.agrisense_input_key = current_key
            st.error(result.vision_error)
            return

        if result.needs_clarification:
            st.session_state.agrisense_error = result.clarification_prompt
            st.session_state.agrisense_result = None
            st.session_state.agrisense_input_key = current_key
            st.warning(result.clarification_prompt)
            return

        st.session_state.agrisense_result = result
        st.session_state.agrisense_error = None
        st.session_state.agrisense_input_key = current_key
        st.success("Analysis complete")
        pdf_bytes = generate_pdf_report(result)
        _render_result_sections(result, pdf_bytes)
        _inject_custom_css()
        return

    if cached_result is not None:
        _render_result_sections(cached_result, generate_pdf_report(cached_result))
        _inject_custom_css()
        return

    if st.session_state.agrisense_error:
        st.error(st.session_state.agrisense_error)

    st.caption("The dashboard will appear here after the first successful analysis.")
    _inject_custom_css()


def _node_note(agent_name: str, status: str, state: AgriSenseState) -> str:
    """Return a short display note for the workflow node."""
    if status == "running":
        return "Running"
    if status == "skipped":
        return "Skipped"
    if status == "failed":
        return "Failed"
    if agent_name == "Agronomy RAG Agent":
        return f"{len(state.citations)} document(s) retrieved"
    if agent_name == "Memory Manager":
        return "Completed"
    if agent_name == "Market Agent" and state.market_error:
        return "API unavailable"
    if agent_name == "Response Synthesizer":
        return "Report ready"
    return "Completed"


if __name__ == "__main__":
    main()