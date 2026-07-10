"""Final response synthesizer."""

from __future__ import annotations

from datetime import datetime

from agents.base_agent import BaseAgent
from agents.state import AgriSenseState
from utils.display import (
    build_recommendation_summary,
    relevant_citations,
    confidence_bucket,
    confidence_bar,
    format_crop_label,
    format_disease_label,
    format_percent,
)


class SynthesizerAgent(BaseAgent[AgriSenseState]):
    """Compose the final markdown report."""

    def run(self, state: AgriSenseState) -> AgriSenseState:
        generated_at = state.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
        symptoms = ", ".join(state.symptoms) if state.symptoms else "Not specified"
        sources_list = relevant_citations(state.citations, state.disease, state.crop)
        sources = "\n".join(
            f"- {citation.get('title', citation['source'])} | {citation['source']} (page {citation['page']}, score {citation.get('score', 'n/a')})"
            for citation in sources_list
        ) or "- No directly matched citations available"
        recommendation_summary = build_recommendation_summary(
            confidence=state.confidence,
            treatment=state.treatment,
            roi=state.roi,
            market_text=state.market_rationale
            if state.market_error is None
            else "Economic recommendation generated without market analysis.",
            evidence_count=len(sources_list),
        )

        state.final_report = f"""# AgriSense Diagnosis Report
**Generated:** {generated_at}

## Diagnosis
- **Crop:** {format_crop_label(state.crop)}
- **Disease:** {format_disease_label(state.disease)}
- **Confidence:** {format_percent(state.confidence)} ({confidence_bucket(state.confidence)})
- **Confidence Bar:** {confidence_bar(state.confidence)}
- **Visible Symptoms:** {symptoms}

## Treatment
- **Primary Treatment:** {state.treatment or "Unavailable"}
- **Dosage:** {state.dosage or "Unavailable"}
- **Application Frequency:** {state.frequency or "Unavailable"}
- **Expected Recovery:** {state.expected_recovery or "Unavailable"}
- **Prevention:** {state.prevention or "Unavailable"}

## Economic Analysis
- **Treatment Cost:** {state.treatment_cost or "Unavailable"}
- **Expected Recovery:** {state.estimated_yield_recovered or "Unavailable"}
- **Estimated Market Value:** {state.market_value_recovered or "Unavailable"}
- **Net Profit:** {state.net_profit or "Unavailable"}
- **ROI:** {state.roi or "Unavailable"}

## Next Steps
1. Follow the recommended treatment carefully.
2. Recheck symptoms after the expected recovery window.

## Why this recommendation?
{recommendation_summary}

## Evidence Used
{sources}

## Important
- This is not a substitute for expert agronomist consultation.
- If the disease does not improve, contact local agricultural extension support.
"""
        state.workflow_status = "synthesizing"
        return state
