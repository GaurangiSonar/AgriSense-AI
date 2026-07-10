"""Mock Gemini-style responses."""

from __future__ import annotations


def mock_gemini_response(
    crop: str = "tomato",
    disease: str = "early_blight",
    confidence: float = 0.92,
    symptoms: list[str] | None = None,
):
    """Return a response object compatible with the vision parser."""

    class Response:
        text = (
            "{"
            f"\"crop\": \"{crop}\", "
            f"\"disease\": \"{disease}\", "
            f"\"confidence\": {confidence}, "
            f"\"symptoms\": {symptoms or ['brown_rings', 'yellowing']}"
            "}"
        )

    return Response()


def mock_workflow_responses():
    """Yield a series of mock responses for sequential Gemini calls."""
    return [
        mock_gemini_response(),
    ]

