from __future__ import annotations

import unittest
from pathlib import Path

from database.models import get_connection, get_farmer_history, initialize_database, save_interaction


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        initialize_database()
        with get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO farmers (id, name, location, primary_crop) VALUES (?, ?, ?, ?)",
                         ("farmer-db", "Test Farmer", "India", "tomato"))

    def test_save_and_fetch_interaction(self):
        save_interaction(
            farmer_id="farmer-db",
            crop="tomato",
            disease="early_blight",
            confidence=0.9,
            treatment="Copper Fungicide",
            dosage="500ml",
            roi=5900,
            market_recommendation="HOLD",
        )
        history = get_farmer_history("farmer-db", "early_blight")
        self.assertTrue(history)


if __name__ == "__main__":
    unittest.main()

