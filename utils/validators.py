"""Input validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from utils.constants import MAX_IMAGE_SIZE_BYTES, SUPPORTED_IMAGE_TYPES


@dataclass
class ValidationResult:
    """Simple validation response."""

    valid: bool
    message: str = ""


def validate_image_file(image_path: str) -> ValidationResult:
    """Validate image type, size, and readability."""
    path = Path(image_path)
    if not path.exists():
        return ValidationResult(False, "Cannot read image. Try another.")

    if path.suffix.lower() not in SUPPORTED_IMAGE_TYPES:
        return ValidationResult(False, "Please upload JPG or PNG only")

    if path.stat().st_size > MAX_IMAGE_SIZE_BYTES:
        return ValidationResult(False, "Image must be under 10MB")

    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return ValidationResult(False, "Cannot read image. Try another.")

    return ValidationResult(True, "")


def clamp_confidence(value: float) -> float:
    """Clamp a confidence value to the 0.0-1.0 range."""
    return max(0.0, min(1.0, float(value)))


def normalize_text(value: str | None) -> str:
    """Normalize text input for downstream use."""
    return (value or "").strip()

