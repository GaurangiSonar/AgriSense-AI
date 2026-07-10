from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from models.disease_predictor import predict_disease
from rag.crop_disease_kb import disease_candidates_for_crop


class DiseasePredictorTests(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path("work/disease-predictor-tests")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _write_image(self, name: str, color: tuple[int, int, int]) -> str:
        path = self.work_dir / name
        Image.new("RGB", (96, 96), color=color).save(path)
        return str(path)

    def test_candidates_are_crop_specific(self):
        tomato = {item["disease"] for item in disease_candidates_for_crop("tomato")}
        potato = {item["disease"] for item in disease_candidates_for_crop("potato")}
        self.assertIn("early_blight", tomato)
        self.assertIn("early_blight", potato)
        self.assertIn("rice_blast", {item["disease"] for item in disease_candidates_for_crop("rice")})
        self.assertNotIn("rice_blast", tomato)

    def test_predict_returns_single_top_class(self):
        path = self._write_image("tomato.jpg", (210, 40, 40))
        result = predict_disease(path, "tomato")
        self.assertTrue(result.disease)
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(result.class_scores[0][0], result.disease)

    def test_different_images_can_rank_different_top_diseases(self):
        diseases = set()
        colors = ((210, 40, 40), (40, 180, 60), (220, 210, 40), (80, 80, 180))
        for index, color in enumerate(colors):
            path = self._write_image(f"sample_{index}.jpg", color)
            diseases.add(predict_disease(path, "tomato").disease)
        self.assertGreater(len(diseases), 1)

    def test_plant_part_detection(self):
        from models.disease_predictor import _detect_plant_part
        # Test context-based detection
        self.assertEqual(_detect_plant_part("test.jpg", "leaf symptoms"), "leaf")
        self.assertEqual(_detect_plant_part("test.jpg", "fruit rot"), "fruit")
        self.assertEqual(_detect_plant_part("test.jpg", "stem wilt"), "stem")
        self.assertEqual(_detect_plant_part("test.jpg", "root scurf"), "root")
        self.assertEqual(_detect_plant_part("test.jpg", "panicle grain"), "panicle/grain")

    def test_plant_part_filtering(self):
        from models.disease_predictor import _filter_candidates_by_plant_part
        candidates = disease_candidates_for_crop("wheat")
        # Filter for leaf diseases
        leaf_filtered = _filter_candidates_by_plant_part(candidates, "wheat", "leaf")
        self.assertGreater(len(leaf_filtered), 0)
        # Filter for panicle/grain diseases
        grain_filtered = _filter_candidates_by_plant_part(candidates, "wheat", "panicle/grain")
        self.assertGreater(len(grain_filtered), 0)


if __name__ == "__main__":
    unittest.main()
