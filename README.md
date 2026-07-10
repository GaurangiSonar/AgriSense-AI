# AgriSense AI

AgriSense AI is a software-only multi-agent crop disease decision platform built with Streamlit and a LangGraph-style planner.

## Local Setup

1. Create a virtual environment with Python 3.11+.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and configure `GOOGLE_API_KEY` if you want Gemini access.
4. Initialize the database with `python scripts/init_db.py`.
5. Run the app with `streamlit run app.py`.

## Structure

- `agents/` contains the modular agents.
- `database/` contains SQLite helpers and schema.
- `rag/` contains the local retrieval pipeline.
- `frontend/` contains Streamlit helpers and styles.
- `scripts/` contains maintenance scripts.
- `tests/` contains unit and integration tests.

