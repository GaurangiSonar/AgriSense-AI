"""LangGraph planner orchestrating all agents."""

from __future__ import annotations

from dataclasses import dataclass
import time

from agents.critic import CriticAgent
from agents.economic import EconomicAgent
from agents.market import MarketAgent
from agents.memory import MemoryManagerAgent
from agents.rag import RAGAgent
from agents.state import AgriSenseState
from agents.synthesizer import SynthesizerAgent
from agents.vision import VisionAgent
from config import MAX_RETRIES
from database.models import ensure_farmer, initialize_database, save_interaction
from utils.display import diagnosis_conflicts, format_crop_label, format_disease_label
from utils.logger import get_logger


@dataclass
class PlannerAgent:
    """Coordinate the agent pipeline."""

    logger_name: str = "PlannerAgent"

    def __post_init__(self) -> None:
        self.logger = get_logger(self.logger_name)
        self.vision = VisionAgent()
        self.critic = CriticAgent()
        self.rag = RAGAgent()
        self.economic = EconomicAgent()
        self.market = MarketAgent()
        self.memory = MemoryManagerAgent()
        self.synthesizer = SynthesizerAgent()

    def execute(self, state: AgriSenseState, progress_callback=None) -> AgriSenseState:
        """Run the full workflow and keep going on recoverable failures."""
        started_at = time.monotonic()
        initialize_database()
        state.workflow_status = "initialized"
        state.max_retries = MAX_RETRIES
        state.diagnosis_validation_error = None
        state.execution_time_seconds = 0.0
        state.agents_executed = 0
        state.documents_retrieved = 0
        state.executed_agents = []
        state.skipped_agents = []

        self._notify(progress_callback, "Planner Agent", "running", state)
        self._mark_executed(state, "Planner Agent")

        state = self._with_retry(self.vision, state)
        self._mark_executed(state, "Gemini Vision Agent")
        self._notify(progress_callback, "Gemini Vision Agent", "completed", state)
        if state.vision_error:
            self._notify(progress_callback, "Gemini Vision Agent", "failed", state)
            return self._finalize(state, started_at)

        state = self.critic.execute(state)
        self._mark_executed(state, "Critic Agent")
        self._notify(progress_callback, "Critic Agent", "completed", state)
        
        # Handle Vision Agent reconsideration if validation fails
        if state.needs_vision_reconsideration:
            state = self._reconsider_vision_prediction(state, progress_callback)
            if state.needs_vision_reconsideration:
                # If reconsideration still fails, proceed with current diagnosis
                state.needs_vision_reconsideration = False
        
        state = self._validate_diagnosis(state, progress_callback)
        if state.diagnosis_validation_error:
            self._notify(progress_callback, "Critic Agent", "failed", state)
            return self._finalize(state, started_at)

        if self._is_healthy_crop(state):
            self._mark_skipped(state, ["Agronomy RAG Agent", "Market Agent", "Economic Agent", "Memory Manager"])
            self._notify(progress_callback, "Agronomy RAG Agent", "skipped", state)
            self._notify(progress_callback, "Market Agent", "skipped", state)
            self._notify(progress_callback, "Economic Agent", "skipped", state)
            self._notify(progress_callback, "Memory Manager", "skipped", state)
            state.treatment = "No treatment required"
            state.dosage = "N/A"
            state.frequency = "Monitor weekly"
            state.expected_recovery = "Ongoing monitoring"
            state.prevention = self._healthy_prevention(state)
            state.memory_insight = "Healthy crop detected. Personalized recommendation not required."
        else:
            state = self._with_retry(self.rag, state)
            state.documents_retrieved = len(state.citations)
            self._mark_executed(state, "Agronomy RAG Agent")
            self._notify(progress_callback, "Agronomy RAG Agent", "completed", state)
            if state.rag_error:
                self._notify(progress_callback, "Agronomy RAG Agent", "failed", state)
                return self._finalize(state, started_at)

            state = self._with_retry(self.market, state)
            self._mark_executed(state, "Market Agent")
            self._notify(progress_callback, "Market Agent", "completed", state)

            state = self._with_retry(self.economic, state)
            self._mark_executed(state, "Economic Agent")
            self._notify(progress_callback, "Economic Agent", "completed", state)

            state = self._with_retry(self.memory, state)
            self._mark_executed(state, "Memory Manager")
            self._notify(progress_callback, "Memory Manager", "completed", state)

        state = self._with_retry(self.synthesizer, state)
        self._mark_executed(state, "Response Synthesizer")
        self._notify(progress_callback, "Response Synthesizer", "completed", state)

        state = self.critic.execute(state)
        self._mark_executed(state, "Critic Agent")
        self._notify(progress_callback, "Critic Agent", "completed", state)
        self._persist(state)
        self._notify(progress_callback, "Planner Agent", "completed", state)
        return self._finalize(state, started_at)

    def build_graph(self):  # pragma: no cover - optional dependency integration
        """Build a LangGraph workflow when the package is installed."""
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(AgriSenseState)
        graph.add_node("vision", self.vision.execute)
        graph.add_node("critic", self.critic.execute)
        graph.add_node("rag", self.rag.execute)
        graph.add_node("market", self.market.execute)
        graph.add_node("economic", self.economic.execute)
        graph.add_node("memory", self.memory.execute)
        graph.add_node("synthesizer", self.synthesizer.execute)

        graph.set_entry_point("vision")
        graph.add_edge("vision", "critic")

        def route_after_critic(state: AgriSenseState) -> str:
            if state.diagnosis_validation_error:
                return "end"
            if self._is_healthy_crop(state):
                return "synthesizer"
            return "rag"

        def route_after_rag(state: AgriSenseState) -> str:
            return "market"

        def route_after_market(state: AgriSenseState) -> str:
            return "economic"

        def route_after_economic(state: AgriSenseState) -> str:
            return "memory" if state.farmer_id else "synthesizer"

        graph.add_conditional_edges("critic", route_after_critic, {"synthesizer": "synthesizer", "rag": "rag", "end": END})
        graph.add_conditional_edges("rag", route_after_rag, {"market": "market"})
        graph.add_conditional_edges("market", route_after_market, {"economic": "economic"})
        graph.add_conditional_edges("economic", route_after_economic, {"memory": "memory", "synthesizer": "synthesizer"})
        graph.add_edge("memory", "synthesizer")
        graph.add_edge("synthesizer", END)
        return graph.compile()

    def _notify(self, progress_callback, agent_name: str, status: str, state: AgriSenseState) -> None:
        """Emit lightweight progress updates for the UI."""
        if progress_callback:
            try:
                progress_callback(agent_name, status, state)
            except Exception:  # pragma: no cover - UI callback safety net
                self.logger.debug("Progress callback failed", exc_info=True)

    def _mark_executed(self, state: AgriSenseState, agent_name: str) -> None:
        if agent_name not in state.executed_agents:
            state.executed_agents.append(agent_name)

    def _mark_skipped(self, state: AgriSenseState, agent_names: list[str]) -> None:
        for agent_name in agent_names:
            if agent_name and agent_name not in state.skipped_agents:
                state.skipped_agents.append(agent_name)

    def _is_healthy_crop(self, state: AgriSenseState) -> bool:
        disease = (state.disease or "").lower()
        crop = (state.crop or "").lower()
        return "healthy" in disease or "healthy" in crop

    def _healthy_prevention(self, state: AgriSenseState) -> str:
        crop = format_crop_label(state.crop)
        return f"Continue routine monitoring for {crop.lower()}, maintain irrigation, and remove any infected debris promptly."

    def _reconsider_vision_prediction(self, state: AgriSenseState, progress_callback=None) -> AgriSenseState:
        """Reconsider Vision Agent prediction by choosing the next highest-scoring disease."""
        if not state.disease_class_scores or len(state.disease_class_scores) < 2:
            # No alternative candidates available
            state.needs_vision_reconsideration = False
            return state
        
        # Add current disease to rejected list
        state.rejected_diseases.append(state.disease)
        
        # Find the next highest-scoring disease that hasn't been rejected
        for disease, score in state.disease_class_scores:
            if disease not in state.rejected_diseases:
                # Select this disease
                from utils.display import format_crop_label, format_disease_label
                state.disease = format_disease_label(disease)
                state.confidence = score
                state.critic_validation_errors = []
                state.needs_vision_reconsideration = False
                
                self._notify(progress_callback, "Critic Agent", "Reconsidered diagnosis", state)
                return state
        
        # No valid alternatives found
        state.needs_vision_reconsideration = False
        return state

    def _validate_diagnosis(self, state: AgriSenseState, progress_callback=None) -> AgriSenseState:
        """Detect conflicting diagnosis outputs and re-run vision once if needed."""
        if not diagnosis_conflicts(state.crop, state.disease):
            state.crop = format_crop_label(state.crop)
            state.disease = format_disease_label(state.disease)
            return state

        self._notify(progress_callback, "Gemini Vision Agent", "Re-checking", state)
        original_crop = state.crop
        original_disease = state.disease

        state = self.vision.execute(state)
        self._notify(progress_callback, "Gemini Vision Agent", "Completed", state)
        if diagnosis_conflicts(state.crop, state.disease) or (
            state.crop == original_crop and state.disease == original_disease
        ):
            state.diagnosis_validation_error = "Diagnosis could not be validated. Please upload another image."
            state.add_error("critic", state.diagnosis_validation_error)
            state.workflow_status = "needs_attention"
            return state

        state.crop = format_crop_label(state.crop)
        state.disease = format_disease_label(state.disease)
        return state

    def _finalize(self, state: AgriSenseState, started_at: float) -> AgriSenseState:
        """Attach workflow summary data before returning."""
        state.execution_time_seconds = round(time.monotonic() - started_at, 2)
        state.agents_executed = len(state.executed_agents)
        state.documents_retrieved = len(state.citations)
        if state.diagnosis_validation_error:
            state.workflow_status = "needs_attention"
        elif state.needs_clarification:
            state.workflow_status = "clarifying"
        else:
            state.workflow_status = "complete" if not state.error_log else "complete_with_warnings"
        return state

    def _persist(self, state: AgriSenseState) -> None:
        """Persist the interaction for memory and auditability."""
        if not state.crop or not state.disease:
            return

        farmer_id = state.farmer_id or "anonymous-farmer"
        try:
            ensure_farmer(farmer_id, primary_crop=state.crop)
            save_interaction(
                farmer_id=farmer_id,
                crop=state.crop,
                disease=state.disease,
                confidence=state.confidence,
                treatment=state.treatment,
                dosage=state.dosage,
                roi=self._parse_percent(state.roi),
                market_recommendation=state.market_recommendation,
                outcome="pending",
                notes=state.memory_insight or "",
            )
        except Exception as exc:  # pragma: no cover - persistence safety net
            self.logger.warning("Failed to persist workflow result: %s", exc)
            state.add_error("database", str(exc))

    def _with_retry(self, agent, state: AgriSenseState) -> AgriSenseState:
        """Retry failed stages with bounded attempts."""
        last_error_count = len(state.error_log)
        last_result = state
        for attempt in range(state.max_retries + 1):
            result = agent.execute(last_result)
            if len(result.error_log) == last_error_count or not self._agent_failed(agent, result):
                return result
            last_error_count = len(result.error_log)
            self.logger.warning("Retrying %s (attempt %s)", agent.__class__.__name__, attempt + 1)
            last_result = result
        return last_result

    def _agent_failed(self, agent, state: AgriSenseState) -> bool:
        """Determine whether a given agent stage failed."""
        name = agent.__class__.__name__.lower()
        if name == "visionagent":
            return bool(state.vision_error)
        if name == "ragagent":
            return bool(state.rag_error)
        if name == "economicagent":
            return bool(state.economic_error)
        if name == "marketagent":
            return bool(state.market_error)
        if name == "memorymanageragent":
            return False
        if name == "synthesizeragent":
            return not state.final_report
        return False

    def _parse_percent(self, value: str) -> float:
        """Convert a percentage string like '5,900%' into a float."""
        cleaned = value.replace("%", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
