from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from agents.planner import PlannerAgent
from agents.state import AgriSenseState


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.image_path = Path("work/planner_test_tomato.jpg")
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), color="red").save(self.image_path)

    def test_complete_workflow(self):
        state = AgriSenseState(image_path=str(self.image_path), crop_name_optional="tomato", farmer_id="farmer-db")
        result = PlannerAgent().execute(state)
        self.assertIn(result.workflow_status, {"complete", "complete_with_warnings"})
        self.assertTrue(result.final_report)
        self.assertTrue(result.treatment)

    def test_runs_all_eight_agents_without_farmer_id(self):
        state = AgriSenseState(image_path=str(self.image_path), crop_name_optional="tomato")
        result = PlannerAgent().execute(state)
        self.assertIn(result.workflow_status, {"complete", "complete_with_warnings"})
        self.assertEqual(result.agents_executed, 8)
        self.assertIn("Memory Manager", result.executed_agents)
        self.assertNotIn("Memory Manager", result.skipped_agents)


if __name__ == "__main__":
    unittest.main()

