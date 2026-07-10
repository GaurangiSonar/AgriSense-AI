from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from agents.state import AgriSenseState
from agents.vision import VisionAgent
from models.disease_predictor import DiseasePredictionResult


class VisionAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = VisionAgent()
        self.work_dir = Path("work/vision-tests")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = self.work_dir / "test_tomato.jpg"
        Image.new("RGB", (64, 64), color="red").save(self.image_path)

    def test_vision_identifies_top_disease(self):
        prediction = DiseasePredictionResult(
            disease="early_blight",
            confidence=0.92,
            class_scores=(("early_blight", 0.92), ("late_blight", 0.08)),
            symptoms=("brown_rings", "yellowing"),
        )
        with patch("agents.vision.predict_disease", return_value=prediction):
            state = self.agent.execute(
                AgriSenseState(
                    image_path=str(self.image_path),
                    crop_name_optional="tomato",
                )
            )
        self.assertEqual(state.crop, "tomato")
        self.assertEqual(state.disease, "early_blight")
        self.assertAlmostEqual(state.confidence, 0.92)
        self.assertEqual(state.disease_class_scores[0][0], "early_blight")

    def test_vision_handles_missing_file(self):
        state = self.agent.execute(AgriSenseState(image_path="work/does_not_exist.jpg"))
        self.assertIn("Cannot read image", state.vision_error or "")

    def test_vision_requires_crop(self):
        image_path = self.work_dir / "leaf_sample.jpg"
        Image.new("RGB", (64, 64), color="green").save(image_path)
        state = self.agent.execute(AgriSenseState(image_path=str(image_path)))
        self.assertIn("Crop could not be determined", state.vision_error or "")


if __name__ == "__main__":
    unittest.main()
