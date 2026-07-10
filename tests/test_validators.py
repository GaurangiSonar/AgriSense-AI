from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from utils.validators import clamp_confidence, normalize_text, validate_image_file


class ValidatorTests(unittest.TestCase):
    def test_clamp_confidence(self):
        self.assertEqual(clamp_confidence(1.5), 1.0)
        self.assertEqual(clamp_confidence(-2), 0.0)

    def test_normalize_text(self):
        self.assertEqual(normalize_text("  tomato "), "tomato")
        self.assertEqual(normalize_text(None), "")

    def test_validate_image_file(self):
        path = Path("work/validator_test.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), color="blue").save(path)
        result = validate_image_file(str(path))
        self.assertTrue(result.valid)


if __name__ == "__main__":
    unittest.main()

