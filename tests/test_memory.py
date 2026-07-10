from __future__ import annotations

import unittest
from pathlib import Path

from agents.memory import MemoryManagerAgent
from agents.state import AgriSenseState
from database.models import initialize_database


class MemoryAgentTests(unittest.TestCase):
    def setUp(self):
        initialize_database()

    def test_handles_empty_history(self):
        state = AgriSenseState(farmer_id="farmer-1", disease="early_blight", treatment="Copper Fungicide")
        result = MemoryManagerAgent().execute(state)
        self.assertTrue(result.memory_insight)


if __name__ == "__main__":
    unittest.main()

