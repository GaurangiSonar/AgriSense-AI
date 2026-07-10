"""Quick end-to-end workflow smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image

from agents.planner import PlannerAgent
from agents.state import AgriSenseState


def main() -> None:
    path = Path("work/smoke_test.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color="red").save(path)
    result = PlannerAgent().execute(AgriSenseState(image_path=str(path), crop_name_optional="tomato"))
    print(result.workflow_status)
    print(result.final_report[:500])


if __name__ == "__main__":
    main()
