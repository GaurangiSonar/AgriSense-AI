"""Shared economic calculations for consistent dashboard and report values."""

from __future__ import annotations

from dataclasses import dataclass

from config import (
    DEFAULT_CROP_AREA_ACRES,
    DEFAULT_MARKET_PRICE,
    DEFAULT_YIELD_RECOVERY_PCT,
    TREATMENT_COST,
    TREATMENT_DURATION,
)


@dataclass(frozen=True)
class EconomicSnapshot:
    treatment_cost_total: float
    market_value: float
    net_profit: float
    roi_pct: float
    recovery_fraction: float
    recovery_days: int


def parse_market_price(value: str) -> float:
    cleaned = (value or "").replace("₹", "").replace("/quintal", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def infer_treatment_type(treatment: str) -> str:
    text = treatment.lower()
    if any(token in text for token in ("fungicide", "copper", "mancozeb", "sulfur")):
        return "fungicide"
    if any(token in text for token in ("insecticide", "insect", "neem")):
        return "insecticide"
    if any(token in text for token in ("bacteria", "bacterial", "copper bacter")):
        return "bacterial"
    if any(token in text for token in ("nutrient", "fertil", "micronutrient", "rotation", "sanitation")):
        return "nutrient"
    return ""


def compute_economics(
    *,
    treatment: str,
    disease: str,
    confidence: float,
    symptoms: list[str],
    current_price: str = "",
    market_price_per_unit: float = 0.0,
    crop_area_acres: float = 0.0,
    market_live: bool = True,
) -> EconomicSnapshot | None:
    """Compute treatment economics with internally consistent totals."""
    treatment_type = infer_treatment_type(treatment)
    if not treatment_type:
        return None

    crop_area = crop_area_acres or DEFAULT_CROP_AREA_ACRES
    parsed_price = parse_market_price(current_price)
    if market_live and (market_price_per_unit or parsed_price):
        market_price = market_price_per_unit or parsed_price
    else:
        market_price = DEFAULT_MARKET_PRICE

    confidence_factor = 0.85 + (confidence * 0.25)
    disease_pressure = 1.0 + min(0.25, max(0.0, len(symptoms) * 0.04))
    treatment_cost_total = float(TREATMENT_COST[treatment_type]) * crop_area
    if treatment_type == "fungicide" and "blight" in disease.lower():
        treatment_cost_total *= 0.95

    recovery_days = max(3, int(round(TREATMENT_DURATION[treatment_type] * disease_pressure)))
    recovery_fraction = min(0.9, DEFAULT_YIELD_RECOVERY_PCT * confidence_factor)

    market_value = market_price * recovery_fraction * crop_area
    net_profit = market_value - treatment_cost_total
    roi_pct = (net_profit / treatment_cost_total * 100) if treatment_cost_total else 0.0

    return EconomicSnapshot(
        treatment_cost_total=round(treatment_cost_total, 2),
        market_value=round(market_value, 2),
        net_profit=round(net_profit, 2),
        roi_pct=round(roi_pct, 2),
        recovery_fraction=recovery_fraction,
        recovery_days=recovery_days,
    )


def format_economic_snapshot(snapshot: EconomicSnapshot, *, estimated: bool = False) -> dict[str, str]:
    """Format a snapshot for state/display fields."""
    estimated_suffix = " (Estimated)" if estimated else ""
    return {
        "treatment_cost": f"₹{snapshot.treatment_cost_total:,.0f}",
        "estimated_yield_recovered": f"{snapshot.recovery_fraction * 100:.0f}% of normal",
        "market_value_recovered": f"₹{snapshot.market_value:,.0f}{estimated_suffix}",
        "net_profit": f"₹{snapshot.net_profit:,.0f}{estimated_suffix}",
        "roi": f"{snapshot.roi_pct:,.0f}%{estimated_suffix}",
    }


def parse_currency_text(value: str) -> float:
    """Parse a formatted rupee string back to a float."""
    cleaned = (value or "").replace("₹", "").replace("(Estimated)", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_percent_text(value: str) -> float:
    """Parse a formatted percentage string back to a float."""
    cleaned = (value or "").replace("%", "").replace("(Estimated)", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def snapshot_matches_formatted(snapshot: EconomicSnapshot, formatted: dict[str, str]) -> bool:
    """Verify formatted values match the underlying snapshot."""
    treatment_cost = parse_currency_text(formatted["treatment_cost"])
    market_value = parse_currency_text(formatted["market_value_recovered"])
    net_profit = parse_currency_text(formatted["net_profit"])
    roi = parse_percent_text(formatted["roi"])

    if abs(treatment_cost - snapshot.treatment_cost_total) > 0.5:
        return False
    if abs(market_value - snapshot.market_value) > 0.5:
        return False
    if abs(net_profit - snapshot.net_profit) > 0.5:
        return False
    if abs(roi - snapshot.roi_pct) > 0.5:
        return False
    if abs(net_profit - (market_value - treatment_cost)) > 0.5:
        return False
    if treatment_cost and abs(roi - (net_profit / treatment_cost * 100)) > 0.5:
        return False
    return True


def build_economic_analysis_items(
    *,
    treatment_cost: str,
    estimated_yield_recovered: str,
    market_value_recovered: str,
    net_profit: str,
    roi: str,
    economics_estimated: bool,
) -> list[tuple[str, str, str | None]]:
    """Build dashboard rows for the economic analysis card."""
    items = [
        ("Treatment Cost", treatment_cost, "💵"),
        ("Expected Recovery", estimated_yield_recovered, "📈"),
        ("Estimated Market Value", market_value_recovered, "🏷"),
    ]
    if economics_estimated:
        items.extend(
            [
                ("Estimated Net Profit", net_profit, "💰"),
                ("Estimated ROI", roi, "🔁"),
            ]
        )
    else:
        items.extend(
            [
                ("Net Profit", net_profit, "💰"),
                ("ROI", roi, "🔁"),
            ]
        )
    return items
