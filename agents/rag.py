"""RAG agent for crop-specific treatment retrieval and citations."""

from __future__ import annotations

import json
from pathlib import Path

from agents.base_agent import BaseAgent
from agents.state import AgriSenseState
from config import METADATA_PATH, RAG_TOP_K
from rag.crop_disease_kb import all_documents, match_documents, normalize_crop, normalize_disease
from rag.embedder import cosine_similarity, simple_embedding
from utils.image_analysis import image_fingerprint


class RAGAgent(BaseAgent[AgriSenseState]):
    """Retrieve evidence-backed, crop-specific treatment recommendations."""

    def run(self, state: AgriSenseState) -> AgriSenseState:
        query = self._build_query(state)
        fingerprint = image_fingerprint(state.image_path) if state.image_path else {"seed": 0}
        seed = int(fingerprint.get("seed", 0))

        scored = match_documents(
            state.crop,
            state.disease,
            query,
            top_k=RAG_TOP_K,
            seed=seed,
        )
        if not scored:
            scored = self._search_external_index(query, state.crop, state.disease, seed)
        
        # Strict validation: ensure all returned documents are for the correct crop
        normalized_crop = normalize_crop(state.crop)
        scored = [
            (doc, score) for doc, score in scored
            if normalize_crop(doc.get("crop")) == normalized_crop
        ]

        if not scored:
            crop_label = normalize_crop(state.crop) or "this crop"
            disease_label = normalize_disease(state.disease).replace("_", " ")
            state.rag_error = (
                f"No crop-specific treatment info available for {crop_label} {disease_label}. Consult agronomist."
            )
            state.add_error("rag", state.rag_error)
            # Lower confidence when RAG evidence disagrees with vision diagnosis
            state.confidence = max(0.5, state.confidence - 0.15)
            state.confidence_adjusted = True
            state.confidence_adjustment_reason = "No RAG evidence found to support vision diagnosis"
            return state

        best = scored[:RAG_TOP_K]
        best_doc = best[0][0]
        best_score = best[0][1]
        
        # Lower confidence if RAG evidence has low relevance (indicates weak support for vision diagnosis)
        if best_score < 1.3:
            state.confidence = max(0.5, state.confidence - 0.1)
            state.confidence_adjusted = True
            state.confidence_adjustment_reason = f"Low RAG relevance score ({best_score:.2f}) indicates weak support for vision diagnosis"
        
        state.treatment = best_doc.get("treatment", "")
        state.dosage = best_doc.get("dosage", "")
        state.frequency = best_doc.get("frequency", "")
        state.prevention = best_doc.get("prevention", "")
        state.expected_recovery = best_doc.get("expected_recovery", "")
        state.retrieved_chunks = [doc.get("text", "") for doc, _score in best]
        state.citations = [
            {
                "title": doc.get("title", doc.get("source", "Document")),
                "source": doc.get("source", "unknown"),
                "page": doc.get("page", "1"),
                "snippet": doc.get("text", "")[:240],
                "score": f"{score:.3f}",
                "crop": doc.get("crop", ""),
                "disease": doc.get("disease", ""),
            }
            for doc, score in best
        ]
        state.rag_error = None
        state.workflow_status = "rag_done"
        return state

    def _build_query(self, state: AgriSenseState) -> str:
        symptom_text = ", ".join(state.symptoms) if state.symptoms else ""
        return f"Treatment for {state.disease} in {state.crop} with symptoms {symptom_text}"

    def _search_external_index(
        self,
        query: str,
        crop: str,
        disease: str,
        seed: int,
    ) -> list[tuple[dict[str, str], float]]:
        indexed_docs = self._load_index()
        if not indexed_docs:
            return []

        query_embedding = simple_embedding(query)
        image_bias = (seed % 1000) / 10000.0
        normalized_crop = normalize_crop(crop)
        normalized_disease = normalize_disease(disease).replace("_", " ")

        scored: list[tuple[dict[str, str], float]] = []
        for doc in indexed_docs:
            doc_crop = normalize_crop(doc.get("crop"))
            doc_disease = doc.get("disease", "").replace("_", " ").lower()
            # Strict crop validation - skip if crop doesn't match exactly
            if doc_crop and doc_crop != normalized_crop:
                continue
            if normalized_disease and normalized_disease not in doc_disease:
                continue
            score = cosine_similarity(query_embedding, simple_embedding(doc.get("text", "")))
            if score > 0:
                doc_bias = (hash(doc.get("source", "")) % 100) / 10000.0
                scored.append((doc, score + image_bias + doc_bias))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _load_index(self) -> list[dict[str, str]]:
        path = Path(METADATA_PATH)
        if not path.exists():
            return all_documents()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return all_documents()

        if not data:
            return all_documents()

        docs: list[dict[str, str]] = []
        for row in data:
            if isinstance(row, dict) and "text" in row:
                docs.append(row)
        return docs or all_documents()
