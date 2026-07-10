from __future__ import annotations

import unittest

from agents.critic import CriticAgent
from agents.state import AgriSenseState


class CriticAgentTests(unittest.TestCase):
    def test_requires_clarification_for_low_confidence(self):
        state = AgriSenseState(confidence=0.4, treatment="Copper Fungicide")
        result = CriticAgent().execute(state)
        self.assertTrue(result.needs_clarification)
        self.assertTrue(result.clarification_prompt)

    def test_accepts_high_confidence(self):
        state = AgriSenseState(confidence=0.9, treatment="Copper Fungicide", citations=[{"source": "doc", "page": "1"}])
        result = CriticAgent().execute(state)
        self.assertFalse(result.needs_clarification)


if __name__ == "__main__":
    unittest.main()

