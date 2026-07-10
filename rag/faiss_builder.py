"""Fallback knowledge base index builder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.chunker import chunk_text
from rag.embedder import simple_embedding


@dataclass
class IndexedChunk:
    """Stored chunk representation."""

    chunk_id: str
    source: str
    page: str
    text: str
    embedding: dict[str, float]


def build_index_from_documents(document_paths: list[str], output_path: str, metadata_path: str) -> list[IndexedChunk]:
    """Build a lightweight searchable index from documents."""
    indexed: list[IndexedChunk] = []
    for path in document_paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        text = file_path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(text), start=1):
            indexed.append(
                IndexedChunk(
                    chunk_id=f"{file_path.stem}-{idx}",
                    source=file_path.name,
                    page=str(idx),
                    text=chunk,
                    embedding=simple_embedding(chunk),
                )
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([chunk.__dict__ for chunk in indexed], indent=2),
        encoding="utf-8",
    )
    Path(metadata_path).write_text(
        json.dumps(
            [
                {"source": chunk.source, "page": chunk.page, "chunk_id": chunk.chunk_id}
                for chunk in indexed
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return indexed

