from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from utils.economics import compute_economics
from utils.image_analysis import (
    image_fingerprint,
    infer_confidence_from_context,
)


class ImageAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path("work/image-analysis-tests")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _write_image(self, name: str, color: tuple[int, int, int]) -> str:
        path = self.work_dir / name
        Image.new("RGB", (96, 96), color=color).save(path)
        return str(path)

    def test_different_images_produce_different_fingerprints(self):
        red = image_fingerprint(self._write_image("red.jpg", (210, 40, 40)))
        green = image_fingerprint(self._write_image("green.jpg", (40, 180, 60)))
        self.assertNotEqual(red["seed"], green["seed"])

    def test_confidence_avoids_flat_eighty(self):
        path = self._write_image("yellow.jpg", (220, 210, 40))
        fingerprint = image_fingerprint(path)
        confidence = infer_confidence_from_context(fingerprint, "tomato", ["spots"], None)
        self.assertNotAlmostEqual(confidence, 0.80, places=2)
        self.assertGreaterEqual(confidence, 0.72)
        self.assertLessEqual(confidence, 0.96)


class EconomicsConsistencyTests(unittest.TestCase):
    def test_market_value_net_profit_and_roi_align(self):
        snapshot = compute_economics(
            treatment="Copper Fungicide",
            disease="leaf_spot",
            confidence=0.87,
            symptoms=["spots", "yellowing"],
            market_price_per_unit=2500,
            crop_area_acres=1.0,
        )
        assert snapshot is not None
        self.assertAlmostEqual(snapshot.net_profit, snapshot.market_value - snapshot.treatment_cost_total, places=2)
        expected_roi = (snapshot.net_profit / snapshot.treatment_cost_total) * 100
        self.assertAlmostEqual(snapshot.roi_pct, expected_roi, places=2)


if __name__ == "__main__":
    unittest.main()
