from __future__ import annotations

import unittest

from agents.rag import RAGAgent
from agents.state import AgriSenseState


class RAGAgentTests(unittest.TestCase):
    def test_retrieves_tomato_early_blight(self):
        state = AgriSenseState(crop="tomato", disease="early_blight", symptoms=["brown", "yellowing"])
        result = RAGAgent().execute(state)
        self.assertTrue(result.treatment)
        self.assertTrue(result.citations)
        self.assertIn("tomato", result.citations[0]["snippet"].lower())
        self.assertIn("copper", result.treatment.lower())

    def test_retrieves_potato_early_blight_with_cip_reference(self):
        state = AgriSenseState(crop="potato", disease="early_blight", symptoms=["blight", "brown"])
        result = RAGAgent().execute(state)
        self.assertTrue(result.treatment)
        self.assertNotEqual(result.treatment.lower(), "copper fungicide")
        combined = " ".join(
            citation.get("source", "") + " " + citation.get("snippet", "")
            for citation in result.citations
        ).lower()
        self.assertIn("potato", combined)
        self.assertTrue("cip" in combined or "chlorothalonil" in result.treatment.lower())

    def test_potato_and_tomato_early_blight_differ(self):
        tomato = RAGAgent().execute(
            AgriSenseState(crop="tomato", disease="early_blight", symptoms=["blight"])
        )
        potato = RAGAgent().execute(
            AgriSenseState(crop="potato", disease="early_blight", symptoms=["blight"])
        )
        self.assertNotEqual(tomato.treatment, potato.treatment)
        self.assertNotEqual(tomato.dosage, potato.dosage)

    def test_retrieves_rice_blast(self):
        state = AgriSenseState(crop="rice", disease="rice_blast", symptoms=["spots"])
        result = RAGAgent().execute(state)
        self.assertIn("tricyclazole", result.treatment.lower())
        self.assertIn("rice", result.citations[0]["snippet"].lower())

    def test_retrieves_leaf_spot_matching_evidence(self):
        state = AgriSenseState(crop="tomato", disease="leaf_spot", symptoms=["spots", "yellowing"])
        result = RAGAgent().execute(state)
        self.assertFalse(result.rag_error)
        self.assertTrue(result.citations)
        self.assertTrue(all("leaf spot" in (citation.get("disease", "") + " " + citation.get("snippet", "")).lower() for citation in result.citations))


if __name__ == "__main__":
    unittest.main()
