from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from agents.planner import PlannerAgent
from agents.state import AgriSenseState


class PlannerConflictTests(unittest.TestCase):
    def setUp(self):
        self.image_path = Path("work/conflict_test.jpg")
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color="green").save(self.image_path)

    def test_conflicting_diagnosis_returns_validation_error(self):
        planner = PlannerAgent()

        first_result = AgriSenseState(
            image_path=str(self.image_path),
            crop="Tomato - Late Blight",
            disease="early_blight",
            confidence=0.91,
            symptoms=["spots"],
        )
        second_result = AgriSenseState(
            image_path=str(self.image_path),
            crop="Tomato - Late Blight",
            disease="early_blight",
            confidence=0.91,
            symptoms=["spots"],
        )

        with patch.object(planner.vision, "execute", side_effect=[first_result, second_result]):
            with patch.object(planner.rag, "execute", return_value=second_result):
                with patch.object(planner.market, "execute", return_value=second_result):
                    with patch.object(planner.economic, "execute", return_value=second_result):
                        with patch.object(planner.memory, "execute", return_value=second_result):
                            with patch.object(planner.synthesizer, "execute", return_value=second_result):
                                result = planner.execute(AgriSenseState(image_path=str(self.image_path)))

        self.assertEqual(result.diagnosis_validation_error, "Diagnosis could not be validated. Please upload another image.")


if __name__ == "__main__":
    unittest.main()

