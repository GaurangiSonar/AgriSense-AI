from __future__ import annotations

import unittest

from agents.economic import EconomicAgent
from agents.state import AgriSenseState
from utils.economics import compute_economics, format_economic_snapshot, snapshot_matches_formatted


class EconomicAgentTests(unittest.TestCase):
    def test_calculates_roi(self):
        state = AgriSenseState(treatment="Copper Fungicide", crop="tomato", market_price_per_unit=2500)
        result = EconomicAgent().execute(state)
        self.assertTrue(result.treatment_cost.startswith("₹"))
        self.assertTrue(result.roi.endswith("%"))
        self.assertFalse(result.economics_estimated)

    def test_marks_estimated_values_when_market_unavailable(self):
        snapshot = compute_economics(
            treatment="Copper Fungicide",
            disease="leaf_spot",
            confidence=0.87,
            symptoms=["spots", "yellowing"],
            market_live=False,
        )
        assert snapshot is not None
        formatted = format_economic_snapshot(snapshot, estimated=True)
        self.assertIn("(Estimated)", formatted["market_value_recovered"])
        self.assertAlmostEqual(snapshot.net_profit, snapshot.market_value - snapshot.treatment_cost_total, places=2)

        state = AgriSenseState(
            treatment="Copper Fungicide",
            crop="tomato",
            disease="leaf_spot",
            confidence=0.87,
            market_error="Market API temporarily unavailable.",
        )
        result = EconomicAgent().execute(state)
        self.assertTrue(result.economics_estimated)
        self.assertIn("(Estimated)", result.net_profit)

    def test_formatted_values_match_snapshot(self):
        snapshot = compute_economics(
            treatment="Copper Fungicide",
            disease="leaf_spot",
            confidence=0.87,
            symptoms=["spots"],
            market_price_per_unit=2500,
            market_live=True,
        )
        assert snapshot is not None
        formatted = format_economic_snapshot(snapshot, estimated=False)
        self.assertTrue(snapshot_matches_formatted(snapshot, formatted))


if __name__ == "__main__":
    unittest.main()

