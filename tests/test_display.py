from __future__ import annotations

import unittest

from utils.display import (
    relevant_citations,
    build_recommendation_summary,
    confidence_bar,
    confidence_bucket,
    diagnosis_conflicts,
    format_crop_label,
    format_disease_label,
    timestamp_label,
)


class DisplayHelperTests(unittest.TestCase):
    def test_formats_disease_labels(self):
        self.assertEqual(format_disease_label("early_blight"), "Early Blight")
        self.assertEqual(format_disease_label("late-blight"), "Late Blight")
        self.assertEqual(format_crop_label("Tomato - Late Blight"), "Tomato")

    def test_confidence_labels(self):
        self.assertEqual(confidence_bucket(0.93), "Very High")
        self.assertEqual(confidence_bucket(0.82), "High")
        self.assertEqual(confidence_bucket(0.74), "Medium")
        self.assertEqual(confidence_bucket(0.55), "Low")
        self.assertEqual(len(confidence_bar(0.8)), 10)

    def test_detects_conflicts(self):
        self.assertTrue(diagnosis_conflicts("Tomato - Late Blight", "early_blight"))
        self.assertFalse(diagnosis_conflicts("Tomato", "early_blight"))

    def test_recommendation_summary_and_timestamp(self):
        summary = build_recommendation_summary(
            confidence=0.86,
            treatment="Copper Fungicide",
            roi="ROI positive",
            market_text="Market trend stable",
            memory_text="Relevant history found",
            evidence_count=3,
        )
        self.assertIn("High", summary)
        self.assertIn("Copper Fungicide", summary)
        self.assertIn("3 retrieved document(s)", summary)
        self.assertEqual(timestamp_label("2026-07-06T09:30:00"), "06 Jul 2026, 09:30 AM")

    def test_filters_unrelated_citations(self):
        citations = [
            {
                "title": "ICAR Research Bulletin #2024-TB-45",
                "source": "ICAR Research Bulletin #2024-TB-45",
                "page": "12",
                "snippet": "Early blight in tomato responds to copper fungicide.",
                "score": "0.91",
            },
            {
                "title": "Leaf spot management note",
                "source": "Leaf spot management note",
                "page": "7",
                "snippet": "Leaf spot treatment with sanitation.",
                "score": "0.88",
            },
        ]

        filtered = relevant_citations(citations, "early blight")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["title"], "ICAR Research Bulletin #2024-TB-45")
        leaf_filtered = relevant_citations(citations, "leaf spot")
        self.assertEqual(len(leaf_filtered), 1)
        self.assertEqual(leaf_filtered[0]["title"], "Leaf spot management note")


if __name__ == "__main__":
    unittest.main()
