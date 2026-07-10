"""Crop-specific visual validation rules for disease classification."""

from __future__ import annotations

from typing import Any

# Visual validation rules based on disease characteristics
# These rules help validate Vision Agent predictions by checking key visual features
VISUAL_VALIDATION_RULES: dict[str, dict[str, dict[str, Any]]] = {
    "tomato": {
        "late_blight": {
            "required_features": [
                "water_soaked_lesions",
                "white_mold_underside",
                "irregular_lesions",
                "large_lesions",
            ],
            "excluded_features": [
                "concentric_rings",
                "circular_lesions",
                "target_pattern",
            ],
            "lesion_shape": "irregular_large",
            "lesion_color": "dark_brown_water_soaked",
            "lesion_margins": "water_soaked",
            "distribution": "spreading_rapidly",
            "special_features": ["white_fungal_growth_underside", "rapid_blighting"],
            "plant_part": "leaf",
            "min_confidence": 0.75,
        },
        "early_blight": {
            "required_features": [
                "concentric_rings",
                "circular_lesions",
                "yellow_halo",
                "target_pattern",
            ],
            "excluded_features": [
                "water_soaked_lesions",
                "white_mold_underside",
            ],
            "lesion_shape": "circular_concentric",
            "lesion_color": "dark_brown",
            "lesion_margins": "concentric_rings",
            "distribution": "scattered",
            "special_features": ["target_pattern", "yellow_halo", "older_leaves"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "septoria_leaf_spot": {
            "required_features": [
                "circular_lesions",
                "gray_centers",
                "dark_borders",
                "numerous_tiny_spots",
            ],
            "excluded_features": [
                "concentric_rings",
                "water_soaked_lesions",
            ],
            "lesion_shape": "circular_tiny",
            "lesion_color": "gray_center_dark_border",
            "lesion_margins": "distinct",
            "distribution": "numerous_scattered",
            "special_features": ["black_pycnidia", "tiny_spots"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "bacterial_spot": {
            "required_features": [
                "tiny_lesions",
                "greasy_dark_lesions",
                "yellow_halo",
            ],
            "excluded_features": [
                "concentric_rings",
                "gray_centers",
            ],
            "lesion_shape": "tiny_greasy",
            "lesion_color": "dark_brown",
            "lesion_margins": "yellow_halo",
            "distribution": "scattered",
            "special_features": ["greasy_appearance", "tiny_spots"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "fruit_rot": {
            "required_features": [
                "fruit_visible",
                "sunken_lesions",
                "brown_lesions",
            ],
            "excluded_features": [
                "leaf_only",
            ],
            "lesion_shape": "sunken",
            "lesion_color": "brown",
            "lesion_margins": "distinct",
            "distribution": "fruit_surface",
            "special_features": ["fruit_infection", "sunken_lesions"],
            "plant_part": "fruit",
            "min_confidence": 0.75,
        },
    },
    "potato": {
        "late_blight": {
            "required_features": [
                "water_soaked_lesions",
                "white_mold_underside",
                "large_lesions",
                "rapid_collapse",
            ],
            "excluded_features": [
                "concentric_rings",
                "target_pattern",
            ],
            "lesion_shape": "irregular_large",
            "lesion_color": "brown_water_soaked",
            "lesion_margins": "water_soaked",
            "distribution": "spreading_rapidly",
            "special_features": ["white_fungal_growth_underside", "rapid_collapse"],
            "plant_part": "leaf",
            "min_confidence": 0.75,
        },
        "early_blight": {
            "required_features": [
                "concentric_rings",
                "circular_lesions",
                "yellow_halo",
                "target_pattern",
            ],
            "excluded_features": [
                "water_soaked_lesions",
                "white_mold_underside",
            ],
            "lesion_shape": "circular_concentric",
            "lesion_color": "dark_brown",
            "lesion_margins": "concentric_rings",
            "distribution": "scattered",
            "special_features": ["target_pattern", "yellow_halo"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "black_scurf": {
            "required_features": [
                "tuber_visible",
                "black_crust",
            ],
            "excluded_features": [
                "leaf_only",
                "lesions",
            ],
            "lesion_shape": "crust",
            "lesion_color": "black",
            "lesion_margins": "irregular",
            "distribution": "tuber_surface",
            "special_features": ["tuber_infection", "black_crust"],
            "plant_part": "root",
            "min_confidence": 0.75,
        },
    },
    "pepper": {
        "powdery_mildew": {
            "required_features": [
                "powdery_coating",
                "white_powder",
                "curling_leaves",
            ],
            "excluded_features": [
                "water_soaked_lesions",
                "rust_pustules",
                "lesions",
            ],
            "lesion_shape": "diffuse_powdery",
            "lesion_color": "white_powder",
            "lesion_margins": "diffuse",
            "distribution": "surface_coating",
            "special_features": ["white_powder_coating", "leaf_curling"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "bacterial_spot": {
            "required_features": [
                "small_brown_lesions",
                "yellow_halo",
            ],
            "excluded_features": [
                "concentric_rings",
                "gray_centers",
            ],
            "lesion_shape": "small_circular",
            "lesion_color": "brown",
            "lesion_margins": "yellow_halo",
            "distribution": "scattered",
            "special_features": ["small_lesions", "yellow_halo"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "cercospora_leaf_spot": {
            "required_features": [
                "circular_lesions",
                "gray_centers",
                "dark_borders",
            ],
            "excluded_features": [
                "concentric_rings",
                "water_soaked_lesions",
            ],
            "lesion_shape": "circular",
            "lesion_color": "gray_center_dark_border",
            "lesion_margins": "distinct",
            "distribution": "scattered",
            "special_features": ["gray_center", "dark_border"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "anthracnose": {
            "required_features": [
                "fruit_visible",
                "fruit_lesions",
            ],
            "excluded_features": [
                "leaf_only",
            ],
            "lesion_shape": "sunken",
            "lesion_color": "dark_brown",
            "lesion_margins": "distinct",
            "distribution": "fruit_surface",
            "special_features": ["fruit_infection"],
            "plant_part": "fruit",
            "min_confidence": 0.75,
        },
    },
    "chili": {
        "leaf_curl_virus": {
            "required_features": [
                "leaf_curling",
                "upward_curling",
                "twisted_leaves",
                "puckering",
                "thick_veins",
            ],
            "excluded_features": [
                "lesions",
                "spots",
                "rust_pustules",
            ],
            "lesion_shape": "none",
            "lesion_color": "yellowing",
            "lesion_margins": "none",
            "distribution": "general_distortion",
            "special_features": [
                "upward_curling",
                "twisted_leaves",
                "puckering",
                "vein_thickening",
                "small_distorted_leaves",
                "stunted_growth",
            ],
            "plant_part": "leaf",
            "min_confidence": 0.75,
        },
        "bacterial_leaf_spot": {
            "required_features": [
                "brown_circular_lesions",
                "yellow_halo",
            ],
            "excluded_features": [
                "concentric_rings",
                "gray_centers",
            ],
            "lesion_shape": "circular",
            "lesion_color": "brown",
            "lesion_margins": "yellow_halo",
            "distribution": "scattered",
            "special_features": ["circular_lesions", "yellow_halo"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "powdery_mildew": {
            "required_features": [
                "powdery_coating",
                "white_powder",
            ],
            "excluded_features": [
                "water_soaked_lesions",
                "rust_pustules",
            ],
            "lesion_shape": "diffuse_powdery",
            "lesion_color": "white_powder",
            "lesion_margins": "diffuse",
            "distribution": "surface_coating",
            "special_features": ["white_powder_coating"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "anthracnose": {
            "required_features": [
                "fruit_visible",
                "fruit_lesions",
            ],
            "excluded_features": [
                "leaf_only",
            ],
            "lesion_shape": "sunken",
            "lesion_color": "dark_brown",
            "lesion_margins": "distinct",
            "distribution": "fruit_surface",
            "special_features": ["fruit_infection"],
            "plant_part": "fruit",
            "min_confidence": 0.75,
        },
    },
    "brinjal": {
        "cercospora_leaf_spot": {
            "required_features": [
                "circular_brown_spots",
                "gray_centers",
                "dark_borders",
                "yellow_halo",
                "numerous_spots",
            ],
            "excluded_features": [
                "concentric_rings",
                "water_soaked_lesions",
            ],
            "lesion_shape": "circular",
            "lesion_color": "brown_center_gray_border",
            "lesion_margins": "yellow_halo",
            "distribution": "numerous_scattered",
            "special_features": ["gray_center", "dark_border", "yellow_halo", "numerous_spots"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "phomopsis_blight": {
            "required_features": [
                "large_irregular_lesions",
                "stem_cankers",
            ],
            "excluded_features": [
                "concentric_rings",
                "tiny_spots",
            ],
            "lesion_shape": "large_irregular",
            "lesion_color": "brown",
            "lesion_margins": "irregular",
            "distribution": "spreading",
            "special_features": ["stem_cankers", "fruit_infection", "large_lesions"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "alternaria_leaf_spot": {
            "required_features": [
                "concentric_rings",
                "circular_lesions",
                "target_pattern",
            ],
            "excluded_features": [
                "water_soaked_lesions",
                "gray_centers",
            ],
            "lesion_shape": "circular_concentric",
            "lesion_color": "dark_brown",
            "lesion_margins": "concentric_rings",
            "distribution": "scattered",
            "special_features": ["concentric_rings", "target_pattern"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
    },
    "cucumber": {
        "downy_mildew": {
            "required_features": [
                "angular_lesions",
                "yellow_lesions",
                "gray_purple_mold_underside",
                "vein_limited_lesions",
            ],
            "excluded_features": [
                "powdery_coating",
                "rust_pustules",
                "circular_lesions",
            ],
            "lesion_shape": "angular",
            "lesion_color": "yellow",
            "lesion_margins": "angular_vein_limited",
            "distribution": "scattered",
            "special_features": ["gray_purple_mold_underside", "vein_limited", "angular_lesions"],
            "plant_part": "leaf",
            "min_confidence": 0.75,
        },
        "powdery_mildew": {
            "required_features": [
                "powdery_coating",
                "white_powder",
                "circular_patches",
            ],
            "excluded_features": [
                "water_soaked_lesions",
                "rust_pustules",
                "angular_lesions",
            ],
            "lesion_shape": "circular_powdery",
            "lesion_color": "white_powder",
            "lesion_margins": "diffuse",
            "distribution": "circular_patches",
            "special_features": ["white_powder_coating", "circular_patches"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "angular_leaf_spot": {
            "required_features": [
                "water_soaked_lesions",
                "angular_lesions",
            ],
            "excluded_features": [
                "powdery_coating",
                "rust_pustules",
            ],
            "lesion_shape": "angular",
            "lesion_color": "water_soaked",
            "lesion_margins": "angular",
            "distribution": "scattered",
            "special_features": ["water_soaked_angular_lesions"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "anthracnose": {
            "required_features": [
                "fruit_visible",
                "sunken_lesions",
            ],
            "excluded_features": [
                "leaf_only",
            ],
            "lesion_shape": "sunken",
            "lesion_color": "dark_brown",
            "lesion_margins": "distinct",
            "distribution": "fruit_surface",
            "special_features": ["fruit_infection", "sunken_lesions"],
            "plant_part": "fruit",
            "min_confidence": 0.75,
        },
    },
    "rice": {
        "brown_spot": {
            "required_features": [
                "small_circular_lesions",
                "dark_brown_lesions",
                "gray_centers",
                "yellow_halo",
                "numerous_spots",
            ],
            "excluded_features": [
                "diamond_lesions",
                "water_soaked_lesions",
                "spindle_lesions",
            ],
            "lesion_shape": "circular_small",
            "lesion_color": "dark_brown_gray_center",
            "lesion_margins": "yellow_halo",
            "distribution": "numerous_scattered",
            "special_features": ["gray_center", "yellow_halo", "small_circular", "numerous_spots"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "blast": {
            "required_features": [
                "spindle_diamond_lesions",
                "gray_centers",
                "brown_margins",
            ],
            "excluded_features": [
                "circular_lesions",
                "water_soaked_lesions",
            ],
            "lesion_shape": "spindle_diamond",
            "lesion_color": "gray_center_brown_margin",
            "lesion_margins": "brown",
            "distribution": "scattered",
            "special_features": ["spindle_shape", "diamond_shape", "gray_center"],
            "plant_part": "leaf",
            "min_confidence": 0.75,
        },
        "leaf_scald": {
            "required_features": [
                "long_irregular_lesions",
                "leaf_tip_start",
                "leaf_edge_start",
            ],
            "excluded_features": [
                "circular_lesions",
                "concentric_rings",
            ],
            "lesion_shape": "long_irregular",
            "lesion_color": "tan_white_gray",
            "lesion_margins": "irregular",
            "distribution": "from_tip_edge",
            "special_features": ["long_lesions", "tip_edge_start", "scalded_appearance"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "false_smut": {
            "required_features": [
                "panicle_visible",
                "orange_fungal_balls",
                "green_fungal_balls",
            ],
            "excluded_features": [
                "leaf_only",
                "lesions",
            ],
            "lesion_shape": "fungal_balls",
            "lesion_color": "orange_green",
            "lesion_margins": "round",
            "distribution": "panicle_only",
            "special_features": ["panicle_infection", "fungal_balls", "orange_green"],
            "plant_part": "panicle/grain",
            "min_confidence": 0.75,
        },
    },
    "wheat": {
        "rust": {
            "required_features": [
                "rust_pustules",
                "orange_brown_pustules",
                "powdery_pustules",
                "linear_rows",
            ],
            "excluded_features": [
                "lesions",
                "water_soaked_lesions",
                "tan_lesions",
            ],
            "lesion_shape": "pustules",
            "lesion_color": "orange_brown_powdery",
            "lesion_margins": "raised",
            "distribution": "linear_rows",
            "special_features": ["orange_brown_pustules", "powdery", "linear_rows"],
            "plant_part": "leaf",
            "min_confidence": 0.75,
        },
        "leaf_blight": {
            "required_features": [
                "long_tan_lesions",
                "yellow_borders",
            ],
            "excluded_features": [
                "rust_pustules",
                "powdery_coating",
            ],
            "lesion_shape": "long_elliptical",
            "lesion_color": "tan_yellow_border",
            "lesion_margins": "yellow",
            "distribution": "scattered",
            "special_features": ["long_lesions", "yellow_borders"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "powdery_mildew": {
            "required_features": [
                "powdery_coating",
                "white_powder",
            ],
            "excluded_features": [
                "rust_pustules",
                "water_soaked_lesions",
            ],
            "lesion_shape": "diffuse_powdery",
            "lesion_color": "white_powder",
            "lesion_margins": "diffuse",
            "distribution": "surface_coating",
            "special_features": ["white_powder_coating"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
    },
    "maize": {
        "gray_leaf_spot": {
            "required_features": [
                "long_rectangular_lesions",
                "gray_lesions",
                "vein_limited",
                "parallel_veins",
            ],
            "excluded_features": [
                "circular_lesions",
                "rust_pustules",
                "cigar_shaped",
            ],
            "lesion_shape": "long_rectangular",
            "lesion_color": "gray_tan",
            "lesion_margins": "parallel_veins",
            "distribution": "parallel_veins",
            "special_features": ["rectangular_lesions", "vein_limited", "parallel_veins"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "northern_leaf_blight": {
            "required_features": [
                "large_cigar_shaped_lesions",
                "tan_lesions",
            ],
            "excluded_features": [
                "rectangular_lesions",
                "rust_pustules",
            ],
            "lesion_shape": "cigar_shaped",
            "lesion_color": "tan",
            "lesion_margins": "distinct",
            "distribution": "scattered",
            "special_features": ["cigar_shaped", "large_lesions"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "common_rust": {
            "required_features": [
                "orange_raised_pustules",
                "rust_pustules",
            ],
            "excluded_features": [
                "lesions",
                "tan_lesions",
                "rectangular_lesions",
            ],
            "lesion_shape": "pustules",
            "lesion_color": "orange",
            "lesion_margins": "raised",
            "distribution": "scattered",
            "special_features": ["orange_pustules", "raised"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
    },
    "cotton": {
        "cotton_leaf_curl_virus": {
            "required_features": [
                "leaf_curling",
                "thickened_veins",
                "vein_swelling",
                "enations",
            ],
            "excluded_features": [
                "lesions",
                "spots",
                "rust_pustules",
            ],
            "lesion_shape": "none",
            "lesion_color": "yellowing",
            "lesion_margins": "none",
            "distribution": "general_distortion",
            "special_features": [
                "leaf_curling",
                "vein_thickening",
                "vein_swelling",
                "enations",
            ],
            "plant_part": "leaf",
            "min_confidence": 0.75,
        },
        "cercospora_leaf_spot": {
            "required_features": [
                "small_circular_brown_spots",
                "gray_centers",
                "dark_borders",
            ],
            "excluded_features": [
                "concentric_rings",
                "water_soaked_lesions",
            ],
            "lesion_shape": "circular_small",
            "lesion_color": "brown_center_gray_border",
            "lesion_margins": "dark_border",
            "distribution": "scattered",
            "special_features": ["gray_center", "dark_border", "small_spots"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "bacterial_blight": {
            "required_features": [
                "angular_water_soaked_lesions",
                "black_veins",
            ],
            "excluded_features": [
                "powdery_coating",
                "rust_pustules",
                "circular_lesions",
            ],
            "lesion_shape": "angular",
            "lesion_color": "water_soaked",
            "lesion_margins": "angular",
            "distribution": "along_veins",
            "special_features": ["water_soaked_angular", "black_veins"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
        "alternaria_leaf_spot": {
            "required_features": [
                "concentric_rings",
                "circular_lesions",
                "target_pattern",
            ],
            "excluded_features": [
                "water_soaked_lesions",
                "gray_centers",
            ],
            "lesion_shape": "circular_concentric",
            "lesion_color": "dark_brown",
            "lesion_margins": "concentric_rings",
            "distribution": "scattered",
            "special_features": ["concentric_rings", "target_pattern"],
            "plant_part": "leaf",
            "min_confidence": 0.70,
        },
    },
}


def validate_visual_prediction(
    crop: str, disease: str, detected_features: list[str], confidence: float, detected_plant_part: str = ""
) -> tuple[bool, list[str]]:
    """
    Validate a Vision Agent prediction against visual rules.

    Args:
        crop: Normalized crop name
        disease: Normalized disease name
        detected_features: List of visual features detected in the image
        confidence: Confidence score from Vision Agent
        detected_plant_part: Detected plant part from image analysis

    Returns:
        (is_valid, list of validation errors)
    """
    errors = []

    if crop not in VISUAL_VALIDATION_RULES:
        # No rules for this crop, accept prediction
        return True, []

    if disease not in VISUAL_VALIDATION_RULES[crop]:
        # No rules for this disease, accept prediction
        return True, []

    rules = VISUAL_VALIDATION_RULES[crop][disease]

    # Check plant part match if provided
    if detected_plant_part and "plant_part" in rules:
        expected_plant_part = rules["plant_part"]
        if detected_plant_part != expected_plant_part:
            errors.append(
                f"Disease {disease} affects '{expected_plant_part}' but detected plant part is '{detected_plant_part}'"
            )

    # Check minimum confidence
    if confidence < rules.get("min_confidence", 0.5):
        errors.append(
            f"Confidence {confidence:.2f} below minimum threshold {rules['min_confidence']}"
        )

    # Check required features
    required_features = rules.get("required_features", [])
    for feature in required_features:
        if feature not in detected_features:
            errors.append(f"Missing required feature: {feature}")

    # Check excluded features
    excluded_features = rules.get("excluded_features", [])
    for feature in excluded_features:
        if feature in detected_features:
            errors.append(f"Detected excluded feature: {feature}")

    # Validate lesion characteristics if lesions are present
    if "lesions" in detected_features or "spots" in detected_features:
        # If lesions are present, check if disease should have lesions
        if rules.get("lesion_shape") == "none":
            errors.append(f"Disease {disease} should not have visible lesions")

    return len(errors) == 0, errors


def get_disease_visual_rules(crop: str, disease: str) -> dict[str, Any] | None:
    """Get visual validation rules for a specific crop-disease pair."""
    if crop in VISUAL_VALIDATION_RULES and disease in VISUAL_VALIDATION_RULES[crop]:
        return VISUAL_VALIDATION_RULES[crop][disease]
    return None
