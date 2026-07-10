"""Base class for all AgriSense agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from agents.state import AgriSenseState
from utils.logger import get_logger


T = TypeVar("T", bound=AgriSenseState)


class BaseAgent(ABC, Generic[T]):
    """Stateless agent interface with graceful error handling."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, state: T) -> T:
        """Execute agent-specific logic."""

    def execute(self, state: T) -> T:
        """Run the agent and capture any unexpected failure."""
        try:
            return self.run(state)
        except Exception as exc:  # pragma: no cover - safety net
            self.logger.exception("Agent execution failed")
            state.add_error(self.__class__.__name__.lower(), str(exc))
            return state

