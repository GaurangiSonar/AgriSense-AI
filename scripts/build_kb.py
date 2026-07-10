"""Build the local knowledge base index."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.faiss_builder import build_index_from_documents


if __name__ == "__main__":
    base = Path("knowledge_base/sources")
    docs = [str(path) for path in base.rglob("*") if path.is_file()]
    build_index_from_documents(docs, "data/faiss_index", "data/metadata.json")
