"""Vision agent for crop disease analysis using Gemini Vision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from agents.state import AgriSenseState
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, VISION_MODELS
from rag.crop_disease_kb import DISEASE_DATABASE, disease_candidates_for_crop, normalize_crop, normalize_disease
from utils.validators import clamp_confidence, normalize_text, validate_image_file


class VisionAgent(BaseAgent[AgriSenseState]):
    """Analyze crop imagery using OpenRouter Vision and infer the single most likely disease."""

    def run(self, state: AgriSenseState) -> AgriSenseState:
        validation = validate_image_file(state.image_path)
        if not validation.valid:
            state.vision_error = validation.message
            state.add_error("vision", validation.message)
            return state

        crop = normalize_crop(state.crop_name_optional) or self._infer_crop_from_filename(state.image_path)
        if not crop or crop == "unknown":
            message = "Crop could not be determined. Select a crop before uploading the image."
            state.vision_error = message
            state.add_error("vision", message)
            return state

        context = normalize_text(state.symptom_description_optional)
        
        # Configurable confidence threshold (60%)
        CONFIDENCE_THRESHOLD = 0.60
        
        try:
            # Use single Gemini Vision reasoning pipeline
            analysis_result = self._analyze_with_gemini_reasoning(
                state.image_path, crop, context
            )
            
            # Extract results from JSON response
            detected_crop = normalize_crop(analysis_result.get("crop", crop))
            detected_part = analysis_result.get("plant_part", "leaf")
            visible_symptoms = analysis_result.get("visible_symptoms", [])
            candidate_scores = analysis_result.get("candidate_scores", {})
            best_disease = analysis_result.get("best_disease", "")
            confidence_raw = analysis_result.get("confidence", 0)
            reason = analysis_result.get("reason", "")
            
            print(f"[Vision Agent] Extracted results:")
            print(f"  detected_crop: {detected_crop}")
            print(f"  detected_part: {detected_part}")
            print(f"  best_disease: {best_disease}")
            print(f"  confidence_raw: {confidence_raw}")
            print(f"  candidate_scores count: {len(candidate_scores)}")
            
            # Check if Gemini failed to classify
            if not best_disease or not candidate_scores or confidence_raw == 0:
                message = f"Gemini Vision classification failed: best_disease='{best_disease}', confidence={confidence_raw}, candidate_scores={len(candidate_scores)}"
                state.vision_error = message
                state.add_error("vision", message)
                state.crop = detected_crop
                state.disease = ""
                state.plant_part = detected_part
                state.confidence = 0.0
                state.symptoms = [normalize_text(s) for s in visible_symptoms if normalize_text(s)]
                state.disease_class_scores = []
                state.vision_reasoning = reason
                return state
            
            # Convert confidence from 0-100 to 0-1 (use Gemini's value directly)
            confidence = confidence_raw / 100.0
            
            # Normalize disease name
            normalized_disease = normalize_disease(best_disease)
            
            # Store all candidate scores for reconsideration (use Gemini's scores directly)
            sorted_scores = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            normalized_scores = [(normalize_disease(d), s/100.0) for d, s in sorted_scores]
            
            print(f"[Vision Agent] Selected disease:")
            print(f"  disease: {best_disease}")
            print(f"  confidence: {confidence:.2f}")
            print(f"  reason: {reason}")
            
            # Validate confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                print(f"[Vision Agent] Confidence {confidence:.2f} below threshold {CONFIDENCE_THRESHOLD}")
                
                # Check if highest-scoring disease has supporting visual evidence
                has_visual_evidence = self._has_supporting_visual_evidence(
                    visible_symptoms, detected_part, candidate_scores, best_disease
                )
                
                if has_visual_evidence:
                    print(f"[Vision Agent] Accepting low-confidence prediction due to visual evidence")
                    # Use Gemini's confidence directly, don't boost it artificially
                else:
                    # Return classification failure instead of random disease
                    message = f"Classification failed: No disease scored above {CONFIDENCE_THRESHOLD*100}% confidence threshold. Highest score: {confidence_raw}% for {best_disease}."
                    state.vision_error = message
                    state.add_error("vision", message)
                    state.crop = detected_crop
                    state.disease = ""
                    state.plant_part = detected_part
                    state.confidence = 0.0
                    state.symptoms = [normalize_text(s) for s in visible_symptoms if normalize_text(s)]
                    state.disease_class_scores = normalized_scores
                    state.vision_reasoning = reason
                    return state
            
            state.crop = detected_crop
            state.disease = normalize_text(best_disease)
            state.plant_part = detected_part
            state.confidence = clamp_confidence(confidence)
            state.symptoms = [normalize_text(s) for s in visible_symptoms if normalize_text(s)]
            state.disease_class_scores = normalized_scores
            state.vision_reasoning = reason
            state.vision_error = None
            state.workflow_status = "vision_done"
            
        except Exception as exc:
            message = f"Disease prediction failed: {exc}"
            state.vision_error = message
            state.add_error("vision", message)
            return state

        return state

    def _analyze_with_gemini_reasoning(
        self, image_path: str, crop: str, context: str
    ) -> dict[str, Any]:
        """Use OpenRouter Vision for structured reasoning and disease scoring."""
        try:
            from openai import OpenAI
        except ImportError:
            return self._fallback_analysis(image_path, crop, context, "OpenAI library not installed")
        
        if not OPENROUTER_API_KEY:
            return self._fallback_analysis(image_path, crop, context, "No OPENROUTER_API_KEY found")
        
        # Log API key status (masked for security)
        masked_key = OPENROUTER_API_KEY[:8] + "..." + OPENROUTER_API_KEY[-4:] if len(OPENROUTER_API_KEY) > 12 else "***"
        print(f"[Vision Agent] Using OpenRouter API key: {masked_key}")
        
        # Read image and convert to base64
        import base64
        from PIL import Image as PILImage
        image = PILImage.open(image_path)
        
        # Convert image to base64
        import io
        buffered = io.BytesIO()
        # Convert RGBA to RGB if necessary (JPEG doesn't support alpha channel)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Get all disease candidates for the crop
        candidates = disease_candidates_for_crop(crop)
        if not candidates:
            return self._fallback_analysis(image_path, crop, context, "No disease candidates found")
        
        # Build disease descriptions with visual characteristics from DISEASE_DATABASE
        disease_descriptions = []
        for d in candidates:
            disease_name = d["disease"]
            disease_display = disease_name.replace("_", " ").title()
            
            # Get detailed disease information from DISEASE_DATABASE
            if crop in DISEASE_DATABASE and disease_name in DISEASE_DATABASE[crop]:
                disease_info = DISEASE_DATABASE[crop][disease_name]
                visual_symptoms = disease_info.get("visual_symptoms", "")
                plant_part = disease_info.get("plant_part", "unknown")
                disease_descriptions.append(
                    f"- {disease_display}: {visual_symptoms} (affects {plant_part})"
                )
            else:
                # Fallback to label_text if not in database
                disease_descriptions.append(f"- {disease_display}: {d.get('label_text', '')}")
        
        disease_list = "\n".join(disease_descriptions)
        
        prompt = f"""You are an expert plant pathologist. Analyze this crop disease image for the crop: {crop}

Follow this structured reasoning process:

STEP 1: Verify the crop
- Confirm the image shows {crop} plants
- If the crop doesn't match, note this in your reasoning

STEP 2: Detect the affected plant part
Identify which plant part is affected:
- leaf: flat green structures with veins
- fruit: fleshy or seed-bearing structures
- stem: main stalk or branches
- root: underground structures (tubers for potato)
- grain/panicle: grain heads (rice, wheat)
- cob: corn ear structure
- boll: cotton fruit structure

STEP 3: Extract visible symptoms
Analyze and extract these specific visual features:
- lesion shape (circular, oval, angular, irregular, diamond, elliptical, cigar-shaped, rectangular)
- lesion size (small <2mm, medium 2-10mm, large >10mm, very large spreading)
- lesion color (brown, black, tan, gray, white, yellow, orange, red, purple, reddish-brown, pink)
- lesion borders/margins (distinct, diffuse, yellow halo, water-soaked, concentric rings, wavy edges, raised)
- lesion distribution (scattered, clustered, along veins, along margins, spreading, random, uniform)
- yellow halo presence (present/absent, color, width)
- water-soaked appearance (translucent, wet, oily look)
- powdery growth (white/gray powdery coating, fluffy, sparse)
- rust pustules (orange/brown raised spots containing spores, numerous, scattered)
- mildew (white/gray fungal growth on surface)
- chlorosis (yellowing of tissue, interveinal yellowing, general yellowing)
- necrosis (dead tissue, black/brown areas, dry, mushy)
- curling (leaf curling upward/downward, distortion, rolling)
- vein thickening (vein thickening, vein clearing, vein discoloration)
- vein distortion (vein distortion, vein swelling)
- fungal growth (white/gray/purple fungal growth, mold)

STEP 4: Compare against ALL supported diseases for {crop}
These are the diseases to compare against:
{disease_list}

For each disease, evaluate how well the image matches the expected visual characteristics. Score each disease on a scale of 0-100 based on:
- Lesion shape match
- Lesion size match
- Lesion color match
- Border characteristics match
- Distribution pattern match
- Special features match (halo, mildew, rust, etc.)
- Plant part match (critical - exclude diseases that don't affect the detected plant part)

CROP-SPECIFIC EXCLUSION RULES:
{self._get_crop_specific_rules(crop)}

STEP 5: Return the highest-scoring disease
Select the disease with the highest score as your final diagnosis.

Additional context: {context if context else "None provided"}

Return ONLY valid JSON in this exact format:
{{
  "crop": "the verified crop name",
  "plant_part": "the detected plant part (leaf, fruit, stem, root, grain/panicle, cob, or boll)",
  "visible_symptoms": ["list of extracted visual symptoms"],
  "candidate_scores": {{
    "Disease1": 95,
    "Disease2": 74,
    "Disease3": 52
  }},
  "best_disease": "the highest-scoring disease name",
  "confidence": 95,
  "reason": "brief explanation of why this disease was selected"
}}

CRITICAL RULES:
- Compare against ALL diseases listed above for {crop}
- Do not guess diseases not in the list
- Exclude diseases that don't match the detected plant part
- Apply crop-specific exclusion rules strictly
- Score based on visual feature matching, not random guessing
- Return ONLY the JSON, no additional text
- Use disease names exactly as shown in the list above
- Ensure all candidate scores are provided in the JSON response"""

        # Try with retry logic using OpenRouter
        import json
        from openai import OpenAI
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Try each vision model in order
                for model_name in VISION_MODELS:
                    try:
                        client = OpenAI(
                            base_url=OPENROUTER_BASE_URL,
                            api_key=OPENROUTER_API_KEY
                        )
                        
                        print(f"[Vision Agent] Using model: {model_name}")
                        
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{img_str}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            temperature=0.3,
                        )
                        
                        response_text = response.choices[0].message.content.strip()
                        
                        # Log raw response for debugging
                        print(f"[Vision Agent] Raw OpenRouter response (attempt {attempt + 1}/{max_retries}):")
                        print(f"  {response_text[:500]}..." if len(response_text) > 500 else f"  {response_text}")
                        
                        # Parse JSON response
                        if "```json" in response_text:
                            response_text = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            response_text = response_text.split("```")[1].split("```")[0].strip()
                        
                        result = json.loads(response_text)
                        
                        # Validate JSON schema
                        if not self._validate_json_schema(result):
                            print(f"[Vision Agent] JSON schema validation failed")
                            if attempt == max_retries - 1:
                                return self._fallback_analysis(image_path, crop, context, "JSON schema validation failed")
                            # Retry with stricter prompt
                            prompt = self._get_strict_json_prompt(crop, disease_list)
                            continue
                        
                        # Log parsed JSON for debugging
                        print(f"[Vision Agent] Parsed JSON:")
                        print(f"  crop: {result.get('crop')}")
                        print(f"  plant_part: {result.get('plant_part')}")
                        print(f"  best_disease: {result.get('best_disease')}")
                        print(f"  confidence: {result.get('confidence')}")
                        
                        # Log candidate scores for debugging
                        if "candidate_scores" in result:
                            print(f"[Vision Agent] Candidate scores for {crop}:")
                            for disease, score in sorted(result["candidate_scores"].items(), key=lambda x: x[1], reverse=True):
                                print(f"  {disease}: {score}%")
                        else:
                            print(f"[Vision Agent] WARNING: No candidate_scores in response")
                        
                        return result
                        
                    except Exception as model_error:
                        print(f"[Vision Agent] Model {model_name} failed: {model_error}")
                        continue  # Try next model
                
                # All models failed
                if attempt == max_retries - 1:
                    return self._fallback_analysis(image_path, crop, context, "All vision models failed")
                
            except json.JSONDecodeError as e:
                print(f"[Vision Agent] JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"[Vision Agent] Response text: {response_text[:200]}")
                if attempt == max_retries - 1:
                    return self._fallback_analysis(image_path, crop, context, str(e))
            except Exception as e:
                error_msg = str(e)
                print(f"[Vision Agent] OpenRouter error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if attempt == max_retries - 1:
                    return self._fallback_analysis(image_path, crop, context, error_msg)
        
        return self._fallback_analysis(image_path, crop, context, "Max retries exceeded")

    def _validate_json_schema(self, result: dict[str, Any]) -> bool:
        """Validate that the JSON response has the required schema."""
        required_fields = ["crop", "plant_part", "visible_symptoms", "candidate_scores", "best_disease", "confidence", "reason"]
        
        for field in required_fields:
            if field not in result:
                print(f"[Vision Agent] Missing required field: {field}")
                return False
        
        # Validate candidate_scores is a dict with numeric values
        if not isinstance(result["candidate_scores"], dict):
            print(f"[Vision Agent] candidate_scores is not a dict")
            return False
        
        # Validate confidence is numeric
        if not isinstance(result["confidence"], (int, float)):
            print(f"[Vision Agent] confidence is not numeric: {type(result['confidence'])}")
            return False
        
        # Validate visible_symptoms is a list
        if not isinstance(result["visible_symptoms"], list):
            print(f"[Vision Agent] visible_symptoms is not a list")
            return False
        
        # Validate candidate_scores values are in reasonable range (0-100)
        for disease, score in result["candidate_scores"].items():
            if not isinstance(score, (int, float)):
                print(f"[Vision Agent] Score for {disease} is not numeric: {type(score)}")
                return False
            if score < 0 or score > 100:
                print(f"[Vision Agent] Score for {disease} is out of range: {score}")
                return False
        
        return True

    def _get_strict_json_prompt(self, crop: str, disease_list: str) -> str:
        """Get a stricter prompt that requests JSON only."""
        return f"""You are an expert plant pathologist. Analyze this crop disease image for the crop: {crop}

Compare against these diseases:
{disease_list}

Return ONLY valid JSON in this exact format (no additional text, no markdown):
{{
  "crop": "the verified crop name",
  "plant_part": "the detected plant part",
  "visible_symptoms": ["list of extracted visual symptoms"],
  "candidate_scores": {{
    "Disease1": 95,
    "Disease2": 74,
    "Disease3": 52
  }},
  "best_disease": "the highest-scoring disease name",
  "confidence": 95,
  "reason": "brief explanation"
}}

CRITICAL: Return ONLY the JSON. No markdown, no explanation, no extra text."""

    def _fallback_analysis(self, image_path: str, crop: str, context: str, error_msg: str = "") -> dict[str, Any]:
        """Fallback analysis when OpenRouter Vision is unavailable."""
        print(f"[Vision Agent] Fallback analysis triggered - OpenRouter Vision failed: {error_msg}")
        
        # Provide specific error message based on the error
        if "quota" in error_msg.lower() or "429" in error_msg:
            reason = "OpenRouter API quota exceeded. Please check your plan or wait for quota reset."
        elif "api key" in error_msg.lower():
            reason = "Invalid or missing OpenRouter API key. Check your OPENROUTER_API_KEY environment variable."
        elif "timeout" in error_msg.lower():
            reason = "OpenRouter API request timed out. Please try again."
        elif "json" in error_msg.lower():
            reason = f"OpenRouter returned invalid JSON: {error_msg}"
        else:
            reason = f"OpenRouter Vision API failure: {error_msg}"
        
        return {
            "crop": crop,
            "plant_part": "leaf",
            "visible_symptoms": [],
            "candidate_scores": {},
            "best_disease": "",
            "confidence": 0,
            "reason": reason
        }

    def _infer_crop_from_filename(self, image_path: str) -> str:
        name = Path(image_path).stem.lower()
        for crop in (
            "tomato",
            "potato",
            "rice",
            "wheat",
            "chili",
            "chilli",
            "cotton",
            "pepper",
            "brinjal",
            "cucumber",
            "maize",
        ):
            if crop in name:
                return "chili" if crop == "chilli" else crop
        return "unknown"

    def _get_crop_specific_rules(self, crop: str) -> str:
        """Get crop-specific exclusion rules for disease scoring."""
        rules = {
            "pepper": """
For Pepper:
- NEVER predict Anthracnose unless infected fruit with sunken dark lesions is clearly visible
- If only leaves are visible and there is white or gray powdery fungal growth on the leaf surface, prioritize Powdery Mildew
- Compare Powdery Mildew, Bacterial Spot, Cercospora Leaf Spot, and other leaf diseases before considering fruit diseases
- Score all candidate diseases and return only the highest-scoring diagnosis
""",
            "tomato": """
For Tomato:
- NEVER predict Fruit Rot unless infected fruit with sunken brown lesions is clearly visible
- If only leaves are visible, prioritize leaf diseases over fruit diseases
- Compare Early Blight, Late Blight, Septoria Leaf Spot, Bacterial Spot before considering fruit diseases
""",
            "potato": """
For Potato:
- NEVER predict Black Scurf unless tubers are clearly visible with black crust
- If only leaves are visible, prioritize leaf diseases over tuber diseases
- Compare Early Blight, Late Blight before considering tuber diseases
""",
            "chili": """
For Chili:
- NEVER predict Anthracnose unless infected fruit with sunken lesions is clearly visible
- If only leaves are visible with upward curling, thick veins, and puckering, prioritize Leaf Curl Virus
- Compare leaf diseases before considering fruit diseases
""",
            "brinjal": """
For Brinjal:
- Compare Cercospora Leaf Spot, Phomopsis Blight, Alternaria Leaf Spot before considering fruit diseases
- NEVER predict fruit diseases unless fruit is clearly visible
""",
            "cucumber": """
For Cucumber:
- If only leaves are visible with angular yellow lesions and gray-purple mold underside, prioritize Downy Mildew
- Compare Downy Mildew, Powdery Mildew, Angular Leaf Spot before considering fruit diseases
- NEVER predict Anthracnose unless fruit with sunken lesions is clearly visible
""",
            "rice": """
For Rice:
- NEVER predict False Smut unless panicles/grains are clearly visible with orange/green fungal balls
- If only leaves are visible, compare Brown Spot, Blast, Leaf Scald before considering grain diseases
- NEVER predict grain diseases from leaf images
""",
            "wheat": """
For Wheat:
- If only leaves are visible with orange/brown pustules in linear rows, prioritize Rust
- Compare Rust, Leaf Blight, Powdery Mildew before considering other diseases
""",
            "maize": """
For Maize:
- If only leaves are visible with long rectangular gray lesions parallel to veins, prioritize Gray Leaf Spot
- Compare Gray Leaf Spot, Northern Leaf Blight, Common Rust before considering ear diseases
- NEVER predict ear diseases unless cobs are clearly visible
""",
            "cotton": """
For Cotton:
- If only leaves are visible with curling, thickened veins, and enations, prioritize Cotton Leaf Curl Virus
- Compare Cotton Leaf Curl Virus, Cercospora Leaf Spot, Bacterial Blight before considering boll diseases
- NEVER predict boll diseases unless bolls are clearly visible
""",
        }
        return rules.get(crop, "")

    def _has_supporting_visual_evidence(
        self, visible_symptoms: list[str], plant_part: str, candidate_scores: dict[str, int], best_disease: str
    ) -> bool:
        """Check if the highest-scoring disease has supporting visual evidence."""
        # If no visible symptoms, no evidence
        if not visible_symptoms:
            return False
        
        # If symptoms are generic, no strong evidence
        if len(visible_symptoms) == 1 and visible_symptoms[0] in ["visible_damage", "damage"]:
            return False
        
        # Check if the best disease score is significantly higher than others
        if len(candidate_scores) > 1:
            sorted_scores = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            best_score = sorted_scores[0][1]
            second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
            
            # If best score is at least 20% higher than second best, it's a clear winner
            if best_score - second_best_score >= 20:
                return True
        
        # Check for disease-specific visual evidence
        symptom_text = " ".join(visible_symptoms).lower()
        
        # Disease-specific evidence patterns
        disease_evidence = {
            "powdery_mildew": ["powdery", "white", "gray", "coating", "mildew"],
            "rust": ["rust", "pustule", "orange", "brown", "raised"],
            "leaf_curl_virus": ["curl", "twist", "thick vein", "pucker", "distort"],
            "late_blight": ["water", "soaked", "mold", "white", "underside", "irregular"],
            "early_blight": ["concentric", "ring", "target", "halo"],
            "anthracnose": ["sunken", "fruit", "lesion"],
            "downy_mildew": ["angular", "yellow", "gray", "purple", "underside"],
            "bacterial_spot": ["greasy", "water", "soaked", "halo"],
            "cercospora_leaf_spot": ["circular", "gray", "center", "border"],
            "gray_leaf_spot": ["rectangular", "gray", "parallel"],
            "blast": ["diamond", "spindle", "gray", "center"],
            "brown_spot": ["circular", "brown", "gray", "halo"],
        }
        
        disease_lower = best_disease.lower().replace("_", " ")
        evidence_keywords = []
        
        for disease_name, keywords in disease_evidence.items():
            if disease_name in disease_lower or disease_lower in disease_name:
                evidence_keywords = keywords
                break
        
        # Check if any evidence keywords are present in symptoms
        if evidence_keywords:
            for keyword in evidence_keywords:
                if keyword in symptom_text:
                    return True
        
        # Default: if we have specific symptoms (not generic), consider it evidence
        if len(visible_symptoms) >= 2:
            return True
        
        return False
