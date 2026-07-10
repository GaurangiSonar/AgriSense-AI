"""Memory and personalization agent."""

from __future__ import annotations

from uuid import uuid4

from agents.base_agent import BaseAgent
from agents.state import AgriSenseState
from database.models import get_farmer_history


class MemoryManagerAgent(BaseAgent[AgriSenseState]):
    """Retrieve farmer history and adapt the recommendation."""

    def run(self, state: AgriSenseState) -> AgriSenseState:
        if not state.farmer_id or not state.disease:
            state.memory_insight = "No personalized recommendation available."
            state.workflow_status = "memory_done"
            return state

        history = get_farmer_history(state.farmer_id, state.disease)
        if not history:
            state.memory_insight = "This is the first recorded interaction for this farmer."
            state.workflow_status = "memory_done"
            return state

        last = history[0]
        outcome = str(last.get("outcome") or "").lower()
        treatment = str(last.get("treatment") or "")
        if outcome == "successful" and treatment.lower() == state.treatment.lower():
            state.confidence = round(min(1.0, state.confidence * 1.15), 2)
            state.memory_insight = f"You've successfully treated this before with {treatment}."
        elif outcome == "failed" and treatment.lower() == state.treatment.lower():
            state.confidence = round(max(0.0, state.confidence * 0.85), 2)
            state.memory_insight = "Your previous treatment didn't work. Try this alternative."
        elif outcome == "successful" and treatment.lower() != state.treatment.lower():
            state.memory_insight = f"This treatment worked for you before: {treatment}."
        else:
            state.memory_insight = "Relevant history found and factored into the recommendation."

        state.workflow_status = "memory_done"
        return state
