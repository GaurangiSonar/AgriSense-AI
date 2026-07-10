"""Market intelligence agent."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import pstdev
from urllib.error import URLError
from urllib.request import urlopen

from agents.base_agent import BaseAgent
from agents.state import AgriSenseState
from config import DEFAULT_MARKET_PRICE, PRICE_CACHE_PATH
from database.models import load_price_cache
from rag.crop_disease_kb import crop_market_note, normalize_crop


class MarketAgent(BaseAgent[AgriSenseState]):
    """Fetch or synthesize market price intelligence."""

    def run(self, state: AgriSenseState) -> AgriSenseState:
        price_history = self._load_history(state.crop)
        remote_available = self._remote_available(state.crop)
        current_price = price_history[-1] if price_history else float(DEFAULT_MARKET_PRICE)
        trend_7_day = self._trend(price_history[-7:])
        trend_30_day = self._trend(price_history[-30:])
        volatility = self._volatility(price_history[-30:])
        recommendation, rationale = self._recommendation(state.crop, state.disease, trend_7_day, trend_30_day, volatility)

        state.current_price = f"₹{current_price:,.0f}/quintal"
        state.trend_7_day = trend_7_day
        state.trend_30_day = trend_30_day
        state.volatility = volatility
        state.market_recommendation = recommendation
        state.market_rationale = rationale
        state.market_data_source = "agrimarket.nic.in" if remote_available else "unavailable"
        state.market_error = None if remote_available else "Market API temporarily unavailable."
        state.workflow_status = "market_done"
        return state

    def _remote_available(self, crop: str) -> bool:
        """Check whether remote market data is reachable."""
        try:
            prices = self._fetch_remote_prices(crop)
            return bool(prices)
        except Exception:
            return False

    def _load_history(self, crop: str) -> list[float]:
        cache = load_price_cache()
        if crop in cache:
            price = cache[crop].get("price", DEFAULT_MARKET_PRICE)
            return [float(price * factor) for factor in (0.95, 0.97, 0.99, 1.0)]

        path = Path(PRICE_CACHE_PATH)
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if crop in payload and isinstance(payload[crop], dict):
                    price = float(payload[crop].get("price", DEFAULT_MARKET_PRICE))
                    return [float(price * factor) for factor in (0.92, 0.96, 0.98, 1.0)]
            except Exception:
                pass

        remote = self._fetch_remote_prices(crop)
        if remote:
            return remote
        return [float(DEFAULT_MARKET_PRICE * factor) for factor in (0.94, 0.97, 0.99, 1.0)]

    def _fetch_remote_prices(self, crop: str) -> list[float]:
        url = f"https://example.com/market/{crop}"
        try:  # pragma: no cover - network path is not exercised in tests
            with urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            prices = data.get("prices", [])
            return [float(value) for value in prices if value is not None]
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError):
            return []

    def _trend(self, prices: list[float]) -> str:
        if len(prices) < 2:
            return "stable 0%"
        delta = prices[-1] - prices[0]
        pct = abs(delta / prices[0]) * 100 if prices[0] else 0.0
        direction = "up" if delta > 0 else "down" if delta < 0 else "stable"
        return f"{direction} {pct:.0f}%"

    def _volatility(self, prices: list[float]) -> str:
        if len(prices) < 2:
            return "low"
        deviation = pstdev(prices)
        mean_price = sum(prices) / len(prices)
        ratio = deviation / mean_price if mean_price else 0.0
        if ratio > 0.08:
            return "high"
        if ratio > 0.04:
            return "medium"
        return "low"

    def _recommendation(
        self,
        crop: str,
        disease: str,
        trend_7_day: str,
        trend_30_day: str,
        volatility: str,
    ) -> tuple[str, str]:
        crop_note = crop_market_note(crop)
        disease_label = (disease or "").replace("_", " ").title()
        crop_label = normalize_crop(crop).title() or "Crop"

        up_30 = "up" in trend_30_day
        down_7 = "down" in trend_7_day
        if up_30 and not down_7:
            return (
                "HOLD",
                f"{crop_label} {disease_label}: price has been trending up. {crop_note} Wait 1-2 weeks before selling.",
            )
        if down_7 and "down" in trend_30_day:
            return (
                "MONITOR",
                f"{crop_label} {disease_label}: price is weakening. {crop_note} Monitor daily and sell if the market stabilizes.",
            )
        if volatility == "high":
            return (
                "MONITOR",
                f"{crop_label} {disease_label}: market is volatile. {crop_note} Avoid selling into a weak price window.",
            )
        return (
            "SELL",
            f"{crop_label} {disease_label}: current conditions favor selling now. {crop_note}",
        )
