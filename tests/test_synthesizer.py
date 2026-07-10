from __future__ import annotations

import unittest

from agents.state import AgriSenseState
from agents.synthesizer import SynthesizerAgent


class SynthesizerAgentTests(unittest.TestCase):
    def test_builds_report(self):
        state = AgriSenseState(
            crop="tomato",
            disease="early_blight",
            confidence=0.92,
            treatment="Copper Fungicide",
            dosage="500ml per 15L water",
            frequency="Every 10 days",
            prevention="Remove infected leaves",
            expected_recovery="18-21 days",
            treatment_cost="₹500/acre",
            estimated_yield_recovered="60% of normal",
            market_value_recovered="₹30,000",
            net_profit="₹29,500",
            roi="5,900%",
            current_price="₹2,500/quintal",
            trend_7_day="up 5%",
            trend_30_day="up 8%",
            market_recommendation="HOLD",
            citations=[{"source": "ICAR", "page": "12"}],
        )
        result = SynthesizerAgent().execute(state)
        self.assertIn("# AgriSense Diagnosis Report", result.final_report)
        self.assertIn("Copper Fungicide", result.final_report)


if __name__ == "__main__":
    unittest.main()

