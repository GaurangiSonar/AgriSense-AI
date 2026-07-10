"""Critic agent for confidence gating and response validation."""

from __future__ import annotations

from agents.base_agent import BaseAgent
from agents.state import AgriSenseState
from config import CONFIDENCE_THRESHOLD, CRITIC_MIN_CONFIDENCE
from models.disease_predictor import DISEASE_PLANT_PARTS
from models.visual_validation_rules import validate_visual_prediction
from rag.crop_disease_kb import normalize_crop, normalize_disease


class CriticAgent(BaseAgent[AgriSenseState]):
    """Validate confidence, crop/disease consistency, and final response structure."""

    def run(self, state: AgriSenseState) -> AgriSenseState:
        # Validate Vision Agent prediction before proceeding
        validation_errors = self._validate_vision_prediction(state)
        
        if validation_errors:
            # If validation fails, trigger Vision Agent reconsideration
            state.critic_validation_errors = validation_errors
            state.needs_vision_reconsideration = True
            return state
        
        # If validation passes, verify crop and disease consistency across diagnosis, treatment, and documents
        self._verify_crop_disease_consistency(state)

        if state.final_report and not state.treatment:
            state.add_error("critic", "Treatment missing from final report")

        if state.final_report and not state.citations:
            state.add_error("critic", "Citations missing from final report")

        if state.final_report and state.confidence < CRITIC_MIN_CONFIDENCE:
            state.add_error("critic", "Final confidence is below the minimum validation threshold")

        state.needs_clarification = False
        return state

    def _validate_vision_prediction(self, state: AgriSenseState) -> list[str]:
        """Validate the Vision Agent prediction before proceeding."""
        errors = []
        
        if not state.crop or not state.disease:
            errors.append("Missing crop or disease in prediction")
            return errors
        
        normalized_crop = normalize_crop(state.crop)
        normalized_disease = normalize_disease(state.disease)
        
        # 1. Validate disease belongs to the selected crop
        if normalized_crop in DISEASE_PLANT_PARTS:
            crop_diseases = DISEASE_PLANT_PARTS[normalized_crop]
            if normalized_disease not in crop_diseases:
                errors.append(f"Disease '{state.disease}' is not a known disease for crop '{state.crop}'")
        
        # 2. Validate disease affects the detected plant part
        if state.plant_part and normalized_crop in DISEASE_PLANT_PARTS:
            crop_diseases = DISEASE_PLANT_PARTS[normalized_crop]
            if normalized_disease in crop_diseases:
                expected_part = crop_diseases[normalized_disease]
                if expected_part != state.plant_part:
                    errors.append(
                        f"Disease '{state.disease}' affects '{expected_part}' but detected plant part is '{state.plant_part}'"
                    )
        
        # 3. Validate confidence is reasonable
        if state.confidence < 0.5:
            errors.append(f"Confidence {state.confidence:.2f} is too low for reliable diagnosis")
        
        # 4. Validate RAG documents match the disease (if RAG has run)
        if state.citations:
            for citation in state.citations:
                citation_disease = normalize_disease(citation.get("disease", ""))
                if citation_disease and citation_disease != normalized_disease:
                    if not self._are_related_diseases(citation_disease, normalized_disease):
                        errors.append(
                            f"RAG document refers to '{citation.get('disease')}' but diagnosis is '{state.disease}'"
                        )
        
        # 5. Validate treatment and prevention correspond to the disease (if available)
        if state.treatment:
            treatment_lower = state.treatment.lower()
            disease_lower = normalized_disease.replace("_", " ")
            if disease_lower not in treatment_lower and len(treatment_lower) < 20:
                errors.append(
                    f"Treatment '{state.treatment}' may not be specific to disease '{state.disease}'"
                )
        
        return errors

    def _extract_visual_features_from_symptoms(self, symptoms: list[str]) -> list[str]:
        """Extract visual features from symptoms list for validation."""
        features = []
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            
            # Lesion shapes
            if "circular" in symptom_lower or "round" in symptom_lower:
                features.append("circular_lesions")
            if "angular" in symptom_lower:
                features.append("angular_lesions")
            if "irregular" in symptom_lower:
                features.append("irregular_lesions")
            if "oval" in symptom_lower:
                features.append("oval_circular_lesions")
            if "diamond" in symptom_lower or "spindle" in symptom_lower:
                features.append("diamond_lesions")
            if "spindle" in symptom_lower:
                features.append("spindle_lesions")
            if "rectangular" in symptom_lower or "elongated" in symptom_lower:
                features.append("rectangular_lesions")
            if "cigar" in symptom_lower:
                features.append("cigar_shaped")
            if "long" in symptom_lower and ("lesion" in symptom_lower or "spot" in symptom_lower):
                features.append("long_lesions")
            
            # Lesion sizes
            if "large" in symptom_lower:
                features.append("large_lesions")
            if "small" in symptom_lower or "tiny" in symptom_lower:
                features.append("tiny_lesions")
            if "numerous" in symptom_lower or "many" in symptom_lower:
                features.append("numerous_spots")
            
            # Lesion colors
            if "brown" in symptom_lower:
                features.append("brown_to_black")
            if "black" in symptom_lower:
                features.append("brown_to_black")
            if "gray" in symptom_lower or "grey" in symptom_lower:
                features.append("gray_centers")
            if "tan" in symptom_lower:
                features.append("tan_centers")
            if "yellow" in symptom_lower:
                features.append("yellow_halo")
            if "white" in symptom_lower:
                features.append("white_to_gray")
            if "orange" in symptom_lower:
                features.append("orange_brown")
            if "red" in symptom_lower:
                features.append("red_brown")
            
            # Lesion margins/borders
            if "concentric" in symptom_lower or "ring" in symptom_lower or "target" in symptom_lower:
                features.append("concentric_rings")
            if "target" in symptom_lower:
                features.append("target_pattern")
            if "halo" in symptom_lower:
                features.append("yellow_halo")
            if "water" in symptom_lower or "soaked" in symptom_lower:
                features.append("water_soaked_lesions")
            if "sunken" in symptom_lower:
                features.append("sunken_lesions")
            if "greasy" in symptom_lower:
                features.append("greasy_dark_lesions")
            
            # Distribution
            if "scattered" in symptom_lower:
                features.append("scattered")
            if "spreading" in symptom_lower:
                features.append("spreading_rapidly")
            if "vein" in symptom_lower:
                features.append("along_veins")
            if "vein" in symptom_lower and ("limited" in symptom_lower or "restricted" in symptom_lower):
                features.append("vein_limited")
            if "parallel" in symptom_lower:
                features.append("parallel_veins")
            if "linear" in symptom_lower:
                features.append("linear_rows")
            
            # Special features - fungal growth
            if "mildew" in symptom_lower or "powdery" in symptom_lower:
                features.append("powdery_coating")
            if "white" in symptom_lower and "powder" in symptom_lower:
                features.append("white_powder")
            if "mold" in symptom_lower:
                features.append("white_fungal_growth_underside")
            if "gray" in symptom_lower and "mold" in symptom_lower:
                features.append("gray_purple_mold_underside")
            
            # Special features - rust
            if "rust" in symptom_lower or "pustule" in symptom_lower:
                features.append("rust_pustules")
            if "pustule" in symptom_lower:
                features.append("raised_spots")
            if "raised" in symptom_lower:
                features.append("raised")
            
            # Special features - virus symptoms
            if "curl" in symptom_lower:
                features.append("leaf_curling")
            if "upward" in symptom_lower and "curl" in symptom_lower:
                features.append("upward_curling")
            if "twist" in symptom_lower:
                features.append("twisted_leaves")
            if "pucker" in symptom_lower:
                features.append("puckering")
            if "thick" in symptom_lower and "vein" in symptom_lower:
                features.append("thick_veins")
            if "vein" in symptom_lower and "swell" in symptom_lower:
                features.append("vein_swelling")
            if "enation" in symptom_lower:
                features.append("enations")
            if "stunt" in symptom_lower:
                features.append("stunted_growth")
            
            # Special features - general
            if "yellowing" in symptom_lower:
                features.append("yellowing")
            if "necrosis" in symptom_lower or "dead" in symptom_lower:
                features.append("necrosis")
            if "vein" in symptom_lower and ("thick" in symptom_lower or "distort" in symptom_lower):
                features.append("vein_thickening")
            if "spot" in symptom_lower or "lesion" in symptom_lower:
                features.append("lesions")
            if "collapse" in symptom_lower:
                features.append("rapid_collapse")
            if "blight" in symptom_lower:
                features.append("rapid_blighting")
            if "scald" in symptom_lower:
                features.append("scalded_appearance")
            
            # Plant part detection
            if "fruit" in symptom_lower:
                features.append("fruit_visible")
            if "leaf" in symptom_lower and "only" in symptom_lower:
                features.append("leaf_only")
            if "tuber" in symptom_lower:
                features.append("tuber_visible")
            if "panicle" in symptom_lower or "grain" in symptom_lower:
                features.append("panicle_visible")
        
        return features

    def _verify_crop_disease_consistency(self, state: AgriSenseState) -> None:
        """Verify that diagnosis, treatment, and retrieved documents all match the selected crop and disease."""
        if not state.crop or not state.disease:
            return
        
        normalized_crop = normalize_crop(state.crop)
        normalized_disease = normalize_disease(state.disease)
        
        # Check citations for crop and disease consistency
        if state.citations:
            for citation in state.citations:
                citation_crop = normalize_crop(citation.get("crop", ""))
                citation_disease = normalize_disease(citation.get("disease", ""))
                
                if citation_crop and citation_crop != normalized_crop:
                    state.add_error(
                        "critic",
                        f"Citation crop mismatch: citation refers to '{citation.get('crop')}' but diagnosis is for '{state.crop}'"
                    )
                
                if citation_disease and citation_disease != normalized_disease:
                    # Allow some flexibility for related diseases (e.g., different rust types)
                    if not self._are_related_diseases(citation_disease, normalized_disease):
                        state.add_error(
                            "critic",
                            f"Citation disease mismatch: citation refers to '{citation.get('disease')}' but diagnosis is '{state.disease}'"
                        )
        
        # Check treatment text for crop and disease consistency
        if state.treatment:
            treatment_lower = state.treatment.lower()
            crop_lower = normalized_crop.replace("_", " ")
            disease_lower = normalized_disease.replace("_", " ")
            
            # Warn if treatment doesn't mention the crop or disease
            if crop_lower not in treatment_lower and disease_lower not in treatment_lower:
                state.add_error(
                    "critic",
                    f"Treatment may not be specific to {state.crop} {state.disease}"
                )
        
        # Check retrieved chunks for relevance
        if state.retrieved_chunks:
            relevant_chunks = 0
            for chunk in state.retrieved_chunks:
                chunk_lower = chunk.lower()
                if crop_lower in chunk_lower or disease_lower in chunk_lower:
                    relevant_chunks += 1
            
            # If less than half of chunks are relevant, flag as warning
            if state.retrieved_chunks and relevant_chunks < len(state.retrieved_chunks) / 2:
                state.add_error(
                    "critic",
                    f"Only {relevant_chunks}/{len(state.retrieved_chunks)} retrieved chunks appear relevant to {state.crop} {state.disease}"
                )

    def _are_related_diseases(self, disease1: str, disease2: str) -> bool:
        """Check if two diseases are related (e.g., different rust types)."""
        # Normalize for comparison
        d1 = disease1.replace("_", " ").lower()
        d2 = disease2.replace("_", " ").lower()
        
        # Check if one is a substring of the other
        if d1 in d2 or d2 in d1:
            return True
        
        # Check for related disease groups
        rust_variants = {"rust", "leaf rust", "stem rust", "stripe rust", "yellow rust", "brown rust"}
        if d1 in rust_variants and d2 in rust_variants:
            return True
        
        blight_variants = {"blight", "late blight", "early blight", "leaf blight"}
        if d1 in blight_variants and d2 in blight_variants:
            return True
        
        spot_variants = {"spot", "leaf spot", "brown spot", "septoria leaf spot", "bacterial spot"}
        if d1 in spot_variants and d2 in spot_variants:
            return True
        
        wilt_variants = {"wilt", "fusarium wilt", "verticillium wilt", "bacterial wilt"}
        if d1 in wilt_variants and d2 in wilt_variants:
            return True
        
        return False
