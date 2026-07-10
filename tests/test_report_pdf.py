from __future__ import annotations

import unittest

from agents.state import AgriSenseState
from utils.report_pdf import generate_pdf_report


class ReportPdfTests(unittest.TestCase):
    def test_generates_pdf_bytes(self):
        state = AgriSenseState(
            crop="tomato",
            disease="early_blight",
            confidence=0.91,
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
            market_rationale="Prices improving.",
            final_report="preview",
        )
        pdf_bytes = generate_pdf_report(state)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 1000)


if __name__ == "__main__":
    unittest.main()

