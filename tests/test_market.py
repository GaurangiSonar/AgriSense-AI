from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.market import MarketAgent
from agents.state import AgriSenseState


class MarketAgentTests(unittest.TestCase):
    def test_uses_local_cache_or_fallback(self):
        state = AgriSenseState(crop="tomato")
        result = MarketAgent().execute(state)
        self.assertTrue(result.current_price.startswith("₹"))
        self.assertIn(result.market_recommendation, {"SELL", "HOLD", "MONITOR"})


if __name__ == "__main__":
    unittest.main()

