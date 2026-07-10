"""Reusable Streamlit UI helpers."""

from __future__ import annotations

from pathlib import Path


def load_stylesheet() -> str:
    """Return the bundled CSS if available."""
    css_path = Path(__file__).resolve().parent / "styles.css"
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def render_error_message(message: str) -> str:
    """Format a user-facing error message."""
    return f"Error: {message}"
