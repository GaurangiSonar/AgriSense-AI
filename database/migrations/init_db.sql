CREATE TABLE IF NOT EXISTS farmers (
    id TEXT PRIMARY KEY,
    name TEXT,
    location TEXT,
    primary_crop TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    farmer_id TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    crop TEXT NOT NULL,
    disease TEXT NOT NULL,
    confidence REAL,
    treatment TEXT,
    dosage TEXT,
    roi REAL,
    market_recommendation TEXT,
    outcome TEXT,
    notes TEXT,
    FOREIGN KEY (farmer_id) REFERENCES farmers(id)
);

CREATE TABLE IF NOT EXISTS price_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    crop TEXT NOT NULL,
    price REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT
);

CREATE INDEX IF NOT EXISTS idx_farmer_disease ON interactions(farmer_id, disease);
CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp);

