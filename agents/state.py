"""Shared LangGraph state for AgriSense AI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgriSenseState:
    """Shared state passed between all agents."""

    image_path: str = ""
    crop_name_optional: str | None = None
    symptom_description_optional: str | None = None

    crop: str = ""
    disease: str = ""
    plant_part: str = ""
    confidence: float = 0.0
    confidence_adjusted: bool = False
    confidence_adjustment_reason: str = ""
    symptoms: list[str] = field(default_factory=list)
    disease_class_scores: list[tuple[str, float]] = field(default_factory=list)
    vision_reasoning: str = ""
    vision_error: str | None = None

    needs_clarification: bool = False
    clarification_prompt: str = ""
    diagnosis_validation_error: str | None = None
    critic_validation_errors: list[str] = field(default_factory=list)
    needs_vision_reconsideration: bool = False
    rejected_diseases: list[str] = field(default_factory=list)

    treatment: str = ""
    dosage: str = ""
    frequency: str = ""
    prevention: str = ""
    expected_recovery: str = ""
    retrieved_chunks: list[str] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    rag_error: str | None = None

    treatment_cost: str = ""
    estimated_recovery_days: int = 0
    estimated_yield_recovered: str = ""
    market_value_recovered: str = ""
    net_profit: str = ""
    roi: str = ""
    economics_estimated: bool = False
    economic_error: str | None = None

    current_price: str = ""
    trend_7_day: str = ""
    trend_30_day: str = ""
    volatility: str = ""
    market_recommendation: str = ""
    market_rationale: str = ""
    market_data_source: str = ""
    market_error: str | None = None

    memory_insight: str | None = None

    final_report: str = ""

    timestamp: str = ""
    retry_count: int = 0
    max_retries: int = 2
    workflow_status: str = "initialized"
    error_log: list[str] = field(default_factory=list)

    farmer_id: str | None = None
    crop_area_acres: float = 1.0
    market_price_per_unit: float = 0.0

    # Workflow summary
    execution_time_seconds: float = 0.0
    agents_executed: int = 0
    documents_retrieved: int = 0
    workflow_note: str = ""
    executed_agents: list[str] = field(default_factory=list)
    skipped_agents: list[str] = field(default_factory=list)

    def add_error(self, source: str, error_msg: str) -> None:
        """Log an error without crashing the workflow."""
        self.error_log.append(f"[{source}] {error_msg}")

    def is_failed(self) -> bool:
        """Return whether a critical error occurred."""
        return any(entry.startswith("[vision]") for entry in self.error_log) and not self.crop
