"""Presentation helpers for polished product rendering."""

from __future__ import annotations

from datetime import datetime


KNOWN_DISEASES = (
    "late blight",
    "early blight",
    "healthy leaf",
    "powdery mildew",
    "fusarium wilt",
    "leaf spot",
    "fruit rot",
    "bacterial wilt",
)


def title_case_label(value: str | None) -> str:
    """Convert snake_case or kebab-case text to a polished label."""
    if not value:
        return "Unknown"
    text = value.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def format_crop_label(value: str | None) -> str:
    """Format crop labels without disease fragments."""
    if not value:
        return "Unknown"
    text = value.strip()
    for separator in (" - ", " — ", ":", "|", ","):
        if separator in text:
            text = text.split(separator, 1)[0].strip()
    return title_case_label(text)


def format_disease_label(value: str | None) -> str:
    """Format disease names for consistent display."""
    return title_case_label(value)


def confidence_bucket(confidence: float) -> str:
    """Return a qualitative confidence label."""
    if confidence >= 0.9:
        return "Very High"
    if confidence >= 0.8:
        return "High"
    if confidence >= 0.7:
        return "Medium"
    return "Low"


def confidence_bar(confidence: float, width: int = 10) -> str:
    """Render a simple block bar for confidence display."""
    filled = max(0, min(width, round(confidence * width)))
    return "█" * filled + "░" * (width - filled)


def format_percent(value: float) -> str:
    """Format a percentage as a whole-number string."""
    return f"{round(value * 100)}%"


def format_currency_text(value: str | float | int | None) -> str:
    """Return a stable currency-like display string."""
    if value in (None, ""):
        return "Unavailable"
    if isinstance(value, str):
        return value
    return f"₹{value:,.0f}"


def normalize_text(value: object) -> str:
    """Collapse whitespace and safely coerce a value for display or matching."""
    return " ".join(str(value or "").split()).strip()


def confidence_summary(confidence: float) -> tuple[str, str]:
    """Return a label and bar string for confidence."""
    return confidence_bucket(confidence), confidence_bar(confidence)


def diagnosis_conflicts(crop: str, disease: str) -> bool:
    """Detect obvious conflicts between crop and disease outputs."""
    crop_text = (crop or "").lower()
    disease_text = (disease or "").lower()
    if not crop_text or not disease_text:
        return False
    if crop_text == disease_text.replace("_", " "):
        return True
    for separator in (" - ", " — ", ":", "|", ","):
        if separator in crop_text:
            left, right = crop_text.split(separator, 1)
            right = right.strip()
            if any(fragment in right for fragment in KNOWN_DISEASES) and right != disease_text.replace("_", " "):
                return True
            if any(fragment in left for fragment in KNOWN_DISEASES):
                return True
            return False
    return False


def build_recommendation_summary(
    *,
    confidence: float,
    treatment: str,
    roi: str,
    market_text: str,
    evidence_count: int,
) -> str:
    """Summarize the final justification without exposing internal reasoning."""
    lines = [
        f"- Disease confidence: {format_percent(confidence)} ({confidence_bucket(confidence)})",
        f"- Treatment chosen: {title_case_label(treatment)}",
        f"- ROI summary: {roi or 'Unavailable'}",
        f"- Market reasoning: {market_text or 'Economic recommendation generated without market analysis.'}",
        f"- Evidence used: {evidence_count} retrieved document(s)",
    ]
    
    # Add low confidence note if confidence is below threshold
    if confidence < 0.65:
        lines.append("- ⚠️ Low confidence – verify with additional inspection")
    
    return "\n".join(lines)


def citation_matches_focus(
    citation: dict[str, str],
    focus_disease: str | None,
    focus_crop: str | None = None,
) -> bool:
    """Return whether a citation directly matches the diagnosed crop and disease."""
    focus_text = normalize_text(focus_disease).replace("_", " ").lower()
    if not focus_text:
        return True

    if focus_crop:
        citation_crop = normalize_text(citation.get("crop", "")).replace("_", " ").lower()
        focus_crop_text = normalize_text(focus_crop).replace("_", " ").lower()
        if citation_crop and citation_crop != focus_crop_text:
            return False

    citation_disease = normalize_text(citation.get("disease", "")).replace("_", " ").lower()
    if citation_disease and citation_disease == focus_text:
        return True

    haystack = " ".join(
        normalize_text(citation.get(field, "")).lower()
        for field in ("crop", "disease", "title", "source", "snippet")
    )
    return focus_text in haystack


def relevant_citations(
    citations: list[dict[str, str]],
    focus_disease: str | None = None,
    focus_crop: str | None = None,
) -> list[dict[str, str]]:
    """Filter citations to those that directly support the diagnosed crop and disease."""
    if not citations:
        return []
    if not focus_disease:
        return [
            {
                **citation,
                "title": normalize_text(citation.get("title", citation.get("source", "Document"))),
                "source": normalize_text(citation.get("source", "Unknown")),
                "page": normalize_text(citation.get("page", "Unknown")),
                "snippet": normalize_text(citation.get("snippet", "")),
            }
            for citation in citations
        ]

    filtered: list[dict[str, str]] = []
    for citation in citations:
        if not citation_matches_focus(citation, focus_disease, focus_crop):
            continue
        filtered.append(
            {
                **citation,
                "title": normalize_text(citation.get("title", citation.get("source", "Document"))),
                "source": normalize_text(citation.get("source", "Unknown")),
                "page": normalize_text(citation.get("page", "Unknown")),
                "snippet": normalize_text(citation.get("snippet", "")),
            }
        )
    return filtered


def timestamp_label(value: str | None) -> str:
    """Format an ISO or display timestamp for UI cards."""
    if not value:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return value
