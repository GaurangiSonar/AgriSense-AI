"""SQLite models and queries."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from config import DB_PATH, PRICE_CACHE_PATH


def _ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection(db_path: str | None = None):
    """Yield a SQLite connection with foreign keys enabled."""
    path = db_path or DB_PATH
    _ensure_parent(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_database(schema_path: str | None = None) -> None:
    """Create tables from the migration SQL file."""
    target_schema = schema_path or str(Path(__file__).resolve().parent / "migrations" / "init_db.sql")
    with open(target_schema, "r", encoding="utf-8") as handle:
        sql = handle.read()
    with get_connection() as conn:
        conn.executescript(sql)


def save_interaction(
    farmer_id: str,
    crop: str,
    disease: str,
    confidence: float,
    treatment: str,
    dosage: str,
    roi: float,
    market_recommendation: str,
    outcome: str = "pending",
    notes: str = "",
    db_path: str | None = None,
) -> str:
    """Persist a workflow interaction and return its id."""
    interaction_id = str(uuid4())
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO interactions (
                id, farmer_id, crop, disease, confidence, treatment, dosage,
                roi, market_recommendation, outcome, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                interaction_id,
                farmer_id,
                crop,
                disease,
                confidence,
                treatment,
                dosage,
                roi,
                market_recommendation,
                outcome,
                notes,
            ),
        )
    return interaction_id


def ensure_farmer(
    farmer_id: str,
    name: str = "",
    location: str = "",
    primary_crop: str = "",
    db_path: str | None = None,
) -> None:
    """Create a farmer row if it does not already exist."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO farmers (id, name, location, primary_crop)
            VALUES (?, ?, ?, ?)
            """,
            (farmer_id, name, location, primary_crop),
        )


def get_farmer_history(farmer_id: str, disease: str, db_path: str | None = None) -> list[dict[str, object]]:
    """Fetch recent treatment history for a farmer and disease."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT disease, treatment, dosage, roi, market_recommendation, outcome, timestamp
            FROM interactions
            WHERE farmer_id = ? AND disease = ?
            ORDER BY timestamp DESC
            LIMIT 5
            """,
            (farmer_id, disease),
        ).fetchall()
    return [dict(row) for row in rows]


def record_price_cache(crop: str, price: float, source: str, db_path: str | None = None) -> None:
    """Persist a cached market price in SQLite and JSON cache."""
    with get_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO price_cache (crop, price, source) VALUES (?, ?, ?)",
            (crop, price, source),
        )
    cache_path = Path(PRICE_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = load_price_cache()
    payload[crop] = {"price": price, "source": source}
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_price_cache() -> dict[str, dict[str, object]]:
    """Load cached market prices from disk."""
    cache_path = Path(PRICE_CACHE_PATH)
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
