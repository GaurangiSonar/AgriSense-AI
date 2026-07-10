"""Application configuration for AgriSense AI."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")
FAISS_INDEX_PATH = _env("FAISS_INDEX_PATH", str(DATA_DIR / "faiss_index"))
DB_PATH = _env("DB_PATH", str(DATA_DIR / "agrisense.db"))
METADATA_PATH = _env("METADATA_PATH", str(DATA_DIR / "metadata.json"))
PRICE_CACHE_PATH = _env("PRICE_CACHE_PATH", str(DATA_DIR / "price_cache.json"))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VISION_MODELS = [
    "qwen/qwen3-vl-32b-instruct",
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen2.5-vl-72b-instruct"
]
TEXT_MODEL = "openai/gpt-4o-mini"
DISEASE_CLASSIFIER_PATH = _env(
    "DISEASE_CLASSIFIER_PATH",
    str(DATA_DIR / "models" / "disease_classifier.joblib"),
)
DISEASE_CLIP_MODEL = _env("DISEASE_CLIP_MODEL", "clip-ViT-B-32")

RAG_TOP_K = 5
CONFIDENCE_THRESHOLD = 0.80
CRITIC_MIN_CONFIDENCE = 0.60

TREATMENT_COST = {
    "fungicide": 500,
    "insecticide": 300,
    "bacterial": 400,
    "nutrient": 200,
}

TREATMENT_DURATION = {
    "fungicide": 21,
    "insecticide": 14,
    "bacterial": 28,
    "nutrient": 7,
}

DEFAULT_CROP_AREA_ACRES = 1.0
DEFAULT_YIELD_LOSS_PCT = 0.40
DEFAULT_YIELD_RECOVERY_PCT = 0.60
DEFAULT_MARKET_PRICE = 2500
MAX_RETRIES = 2

LOG_LEVEL = _env("LOG_LEVEL", "INFO")
LOG_FILE = str(LOG_DIR / "agrisense.log")

