"""Crop-specific disease knowledge base and recommendation templates."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

CropDoc = dict[str, str]

# Comprehensive disease database with visual symptoms, plant parts, aliases, and RAG keywords
DISEASE_DATABASE: dict[str, dict[str, dict[str, Any]]] = {
    "tomato": {
        "early_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular lesions with concentric rings (target pattern), dark brown centers with yellow halos, medium to large size, affects lower leaves first",
            "aliases": ["alternaria solani", "target spot"],
            "rag_keywords": "early blight alternaria concentric rings target pattern brown lesions yellow halo tomato leaf",
        },
        "late_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Irregular water-soaked lesions, brown to black with white mold on underside, no concentric rings, spreads rapidly in humid conditions",
            "aliases": ["phytophthora infestans", "late blight"],
            "rag_keywords": "late blight phytophthora water-soaked white mold brown lesions tomato leaf humid",
        },
        "septoria_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Small circular lesions with brown centers and black pycnidia (tiny black dots), yellow halos, affects lower leaves",
            "aliases": ["septoria lycopersici", "leaf spot"],
            "rag_keywords": "septoria leaf spot brown lesions black pycnidia yellow halo tomato leaf",
        },
        "bacterial_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Angular water-soaked lesions with brown/black borders, yellow halos, small to medium size, scabby appearance on fruit",
            "aliases": ["xanthomonas campestris", "bacterial spot"],
            "rag_keywords": "bacterial spot angular water-soaked brown lesions yellow halo tomato leaf fruit",
        },
        "leaf_mold": {
            "plant_part": "leaf",
            "visual_symptoms": "Yellow patches on upper leaf surface with olive-green to gray mold on underside, leaf curling",
            "aliases": ["cladosporium fulvum", "leaf mold"],
            "rag_keywords": "leaf mold yellow patches gray mold olive green tomato leaf underside",
        },
        "target_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular lesions with concentric rings similar to early blight but smaller, tan to brown centers",
            "aliases": ["coronaria leaf spot", "target spot"],
            "rag_keywords": "target spot circular lesions concentric rings tan brown tomato leaf",
        },
        "mosaic_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Mottled light and dark green areas, leaf distortion, stunting, yellowing, reduced fruit set",
            "aliases": ["tomato mosaic virus", "tmv"],
            "rag_keywords": "mosaic virus mottled green yellow distortion stunting tomato leaf",
        },
        "yellow_leaf_curl_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Upward curling of leaves, yellowing, stunting, vein thickening, reduced fruit size",
            "aliases": ["tylcv", "leaf curl virus"],
            "rag_keywords": "yellow leaf curl virus upward curling yellowing stunting vein thickening tomato",
        },
        "fusarium_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Yellowing and wilting of lower leaves, vascular browning, stunting, one-sided wilting",
            "aliases": ["fusarium oxysporum", "vascular wilt"],
            "rag_keywords": "fusarium wilt yellowing wilting vascular browning stunting tomato stem",
        },
        "verticillium_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "V-shaped yellow lesions on lower leaves, wilting, vascular discoloration, stunting",
            "aliases": ["verticillium dahliae", "vascular wilt"],
            "rag_keywords": "verticillium wilt v-shaped lesions yellowing wilting vascular tomato stem",
        },
        "powdery_mildew": {
            "plant_part": "leaf",
            "visual_symptoms": "White to gray powdery coating on leaf surfaces, yellowing, leaf curling",
            "aliases": ["leveillula taurica", "powdery mildew"],
            "rag_keywords": "powdery mildew white gray powdery coating tomato leaf",
        },
        "anthracnose": {
            "plant_part": "fruit",
            "visual_symptoms": "Sunken circular lesions with dark centers on fruit, concentric rings, water-soaked margins",
            "aliases": ["colletotrichum spp", "ripe rot"],
            "rag_keywords": "anthracnose sunken circular lesions dark centers tomato fruit concentric rings",
        },
        "fruit_rot": {
            "plant_part": "fruit",
            "visual_symptoms": "Soft watery rot on fruit, brown to black discoloration, fungal growth",
            "aliases": ["blossom end rot", "fruit rot"],
            "rag_keywords": "fruit rot soft watery brown black tomato fungal growth",
        },
        "bacterial_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Sudden wilting of entire plant, green stems that remain erect, bacterial ooze from cut stems",
            "aliases": ["ralstonia solanacearum", "southern bacterial wilt"],
            "rag_keywords": "bacterial wilt sudden wilting green stem bacterial ooze tomato",
        },
        "leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Small circular lesions with tan/gray centers and brown borders, no concentric rings",
            "aliases": ["cercospora leaf spot", "leaf spot"],
            "rag_keywords": "leaf spot circular lesions tan gray brown borders tomato",
        },
    },
    "potato": {
        "early_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular lesions with concentric rings, dark brown with yellow halos, affects lower leaves first",
            "aliases": ["alternaria solani", "early blight"],
            "rag_keywords": "early blight alternaria concentric rings brown lesions yellow halo potato leaf",
        },
        "late_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Water-soaked lesions, white mold on underside, spreads rapidly in humid conditions, black lesions on tubers",
            "aliases": ["phytophthora infestans", "late blight"],
            "rag_keywords": "late blight phytophthora water-soaked white mold potato leaf tuber",
        },
        "black_scurf": {
            "plant_part": "root",
            "visual_symptoms": "Dark brown to black crusty lesions on tuber surface, resembles soil particles",
            "aliases": ["rhizoctonia solani", "black scurf"],
            "rag_keywords": "black scurf dark brown crusty lesions potato tuber surface",
        },
        "common_scab": {
            "plant_part": "root",
            "visual_symptoms": "Corky raised lesions on tuber surface, brown to tan color, superficial to deep pits",
            "aliases": ["streptomyces scabies", "common scab"],
            "rag_keywords": "common scab corky raised lesions brown tan potato tuber",
        },
        "powdery_scab": {
            "plant_part": "root",
            "visual_symptoms": "Small raised wart-like lesions on tubers, white to tan color, superficial",
            "aliases": ["spongospora subterranea", "powdery scab"],
            "rag_keywords": "powdery scab raised wart-like lesions white tan potato tuber",
        },
        "fusarium_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Yellowing and wilting of lower leaves, vascular browning, stunting",
            "aliases": ["fusarium oxysporum", "vascular wilt"],
            "rag_keywords": "fusarium wilt yellowing wilting vascular browning potato stem",
        },
        "bacterial_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Sudden wilting, bacterial slime in vascular tissue, brown discoloration",
            "aliases": ["ralstonia solanacearum", "brown rot"],
            "rag_keywords": "bacterial wilt sudden wilting bacterial slime brown potato stem",
        },
        "ring_rot": {
            "plant_part": "root",
            "visual_symptoms": "Ring-shaped discoloration in tuber vascular ring, brown to black, mushy texture",
            "aliases": ["clavibacter michiganensis", "ring rot"],
            "rag_keywords": "ring rot discoloration vascular ring brown black potato tuber",
        },
        "powdery_mildew": {
            "plant_part": "leaf",
            "visual_symptoms": "White to gray powdery coating on leaf surfaces, yellowing",
            "aliases": ["leveillula taurica", "powdery mildew"],
            "rag_keywords": "powdery mildew white gray powdery coating potato leaf",
        },
    },
    "pepper": {
        "bacterial_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Angular water-soaked lesions with yellow borders, small to medium size, scabby lesions on fruit",
            "aliases": ["xanthomonas campestris", "bacterial spot"],
            "rag_keywords": "bacterial spot angular water-soaked yellow borders pepper leaf fruit",
        },
        "anthracnose": {
            "plant_part": "fruit",
            "visual_symptoms": "Sunken circular lesions with dark centers on fruit, pink spore masses in wet conditions",
            "aliases": ["colletotrichum spp", "ripe rot"],
            "rag_keywords": "anthracnose sunken circular lesions dark centers pepper fruit pink spores",
        },
        "phytophthora_blight": {
            "plant_part": "stem",
            "visual_symptoms": "Water-soaked lesions on stems, white fungal growth, sudden wilting, fruit rot",
            "aliases": ["phytophthora capsici", "phytophthora blight"],
            "rag_keywords": "phytophthora blight water-soaked lesions white fungal growth pepper stem",
        },
        "cercospora_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular to angular lesions with gray centers and brown borders, small to medium size",
            "aliases": ["cercospora capsici", "leaf spot"],
            "rag_keywords": "cercospora leaf spot circular angular lesions gray brown pepper leaf",
        },
        "powdery_mildew": {
            "plant_part": "leaf",
            "visual_symptoms": "White to gray powdery coating on leaf surfaces, yellowing",
            "aliases": ["leveillula taurica", "powdery mildew"],
            "rag_keywords": "powdery mildew white gray powdery coating pepper leaf",
        },
        "mosaic_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Mottled light and dark green areas, leaf distortion, stunting, yellowing",
            "aliases": ["cucumber mosaic virus", "mosaic virus"],
            "rag_keywords": "mosaic virus mottled green yellow distortion stunting pepper leaf",
        },
        "leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Small circular lesions with tan centers and brown borders",
            "aliases": ["alternaria solani", "leaf spot"],
            "rag_keywords": "leaf spot circular lesions tan brown pepper",
        },
    },
    "chili": {
        "leaf_curl_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Upward curling of leaves, yellowing, stunting, vein thickening, reduced fruit size",
            "aliases": ["chili leaf curl virus", "clcv", "leaf curl"],
            "rag_keywords": "leaf curl virus upward curling黄色ing stunting vein thickening chili leaf",
        },
        "bacterial_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Angular water-soaked lesions with yellow borders, small to medium size",
            "aliases": ["xanthomonas campestris", "bacterial spot"],
            "rag_keywords": "bacterial spot angular water-soaked yellow borders chili leaf",
        },
        "cercospora_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular to angular lesions with gray centers and brown borders",
            "aliases": ["cercospora capsici", "leaf spot"],
            "rag_keywords": "cercospora leaf spot circular angular lesions gray brown chili leaf",
        },
        "anthracnose": {
            "plant_part": "fruit",
            "visual_symptoms": "Sunken circular lesions with dark centers on fruit, pink spore masses",
            "aliases": ["colletotrichum spp", "ripe rot"],
            "rag_keywords": "anthracnose sunken circular lesions dark centers chili fruit pink spores",
        },
        "powdery_mildew": {
            "plant_part": "leaf",
            "visual_symptoms": "White to gray powdery coating on leaf surfaces, yellowing",
            "aliases": ["leveillula taurica", "powdery mildew"],
            "rag_keywords": "powdery mildew white gray powdery coating chili leaf",
        },
        "damping_off": {
            "plant_part": "stem",
            "visual_symptoms": "Water-soaked lesions at stem bases, seedling collapse, shriveling",
            "aliases": ["pythium spp", "rhizoctonia solani", "damping off"],
            "rag_keywords": "damping off water-soaked stem base seedling collapse chili",
        },
        "fruit_rot": {
            "plant_part": "fruit",
            "visual_symptoms": "Soft watery rot on fruit, brown to black discoloration, fungal growth",
            "aliases": ["anthracnose", "ripe rot", "fruit rot"],
            "rag_keywords": "fruit rot soft watery brown black chili fungal growth",
        },
        "leaf_curl": {
            "plant_part": "leaf",
            "visual_symptoms": "Curling and distortion of leaves, yellowing, stunting",
            "aliases": ["leaf curl virus", "leaf curl"],
            "rag_keywords": "leaf curl curling distortion yellowing stunting chili leaf",
        },
    },
    "brinjal": {
        "phomopsis_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular lesions with tan centers and brown borders, fruiting bodies in lesions",
            "aliases": ["phomopsis vexans", "fruit rot"],
            "rag_keywords": "phomopsis blight circular lesions tan brown fruiting bodies brinjal leaf",
        },
        "little_leaf": {
            "plant_part": "leaf",
            "visual_symptoms": "Very small leaves, chlorosis, stunting, excessive branching",
            "aliases": ["phytoplasma", "little leaf"],
            "rag_keywords": "little leaf small leaves chlorosis stunting brinjal",
        },
        "cercospora_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular to angular lesions with gray centers and brown borders",
            "aliases": ["cercospora melongenae", "leaf spot"],
            "rag_keywords": "cercospora leaf spot circular angular lesions gray brown brinjal leaf",
        },
        "alternaria_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular lesions with concentric rings, dark brown with yellow halos",
            "aliases": ["alternaria solani", "early blight"],
            "rag_keywords": "alternaria leaf spot circular lesions concentric rings brown brinjal leaf",
        },
        "bacterial_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Sudden wilting, bacterial slime in vascular tissue, brown discoloration",
            "aliases": ["ralstonia solanacearum", "bacterial wilt"],
            "rag_keywords": "bacterial wilt sudden wilting bacterial slime brown brinjal stem",
        },
        "fusarium_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Yellowing and wilting of lower leaves, vascular browning, stunting",
            "aliases": ["fusarium oxysporum", "vascular wilt"],
            "rag_keywords": "fusarium wilt yellowing wilting vascular browning brinjal stem",
        },
        "fruit_rot": {
            "plant_part": "fruit",
            "visual_symptoms": "Soft watery rot on fruit, brown to black discoloration, fungal growth",
            "aliases": ["phomopsis vexans", "fruit rot"],
            "rag_keywords": "fruit rot soft watery brown black brinjal fungal growth",
        },
        "leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Small circular lesions with tan centers and brown borders",
            "aliases": ["cercospora leaf spot", "leaf spot"],
            "rag_keywords": "leaf spot circular lesions tan brown brinjal",
        },
    },
    "cucumber": {
        "downy_mildew": {
            "plant_part": "leaf",
            "visual_symptoms": "Angular yellow lesions on upper leaf surface, gray-purple mold on underside, leaf death",
            "aliases": ["pseudoperonospora cubensis", "downy mildew"],
            "rag_keywords": "downy mildew angular yellow lesions gray mold cucumber leaf underside",
        },
        "powdery_mildew": {
            "plant_part": "leaf",
            "visual_symptoms": "White to gray powdery coating on leaf surfaces, yellowing",
            "aliases": ["sphaerotheca fuliginea", "powdery mildew"],
            "rag_keywords": "powdery mildew white gray powdery coating cucumber leaf",
        },
        "angular_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Angular water-soaked lesions with brown centers, small to medium size",
            "aliases": ["pseudomonas syringae", "angular leaf spot"],
            "rag_keywords": "angular leaf spot water-soaked brown lesions cucumber leaf",
        },
        "anthracnose": {
            "plant_part": "fruit",
            "visual_symptoms": "Sunken circular lesions with dark centers on fruit, pink spore masses",
            "aliases": ["colletotrichum orbiculare", "anthracnose"],
            "rag_keywords": "anthracnose sunken circular lesions dark centers cucumber fruit pink spores",
        },
        "gummy_stem_blight": {
            "plant_part": "stem",
            "visual_symptoms": "Water-soaked lesions on stems, gummy exudate, stem cankers, wilting",
            "aliases": ["didymella bryoniae", "gummy stem blight"],
            "rag_keywords": "gummy stem blight water-soaked lesions gummy exudate cucumber stem",
        },
        "mosaic_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Mottled light and dark green areas, leaf distortion, stunting, yellowing",
            "aliases": ["cucumber mosaic virus", "cmv"],
            "rag_keywords": "mosaic virus mottled green yellow distortion stunting cucumber leaf",
        },
        "fruit_rot": {
            "plant_part": "fruit",
            "visual_symptoms": "Soft watery rot on fruit, brown to black discoloration",
            "aliases": ["pythium spp", "fruit rot"],
            "rag_keywords": "fruit rot soft watery brown black cucumber",
        },
    },
    "rice": {
        "rice_blast": {
            "plant_part": "leaf",
            "visual_symptoms": "Diamond or elliptical lesions with gray/white centers and brown borders, pointed ends, can cause neck rot",
            "aliases": ["magnaporthe oryzae", "blast"],
            "rag_keywords": "rice blast diamond elliptical lesions gray white brown pointed neck rot rice leaf",
        },
        "brown_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Oval/circular lesions with brown centers and yellow halos, small to medium size (2-10mm), scattered, reddish-brown",
            "aliases": ["helminthosporium oryzae", "brown spot"],
            "rag_keywords": "brown spot oval circular lesions brown yellow halo scattered reddish rice leaf",
        },
        "bacterial_leaf_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Water-soaked lesions with wavy edges, yellow to gray/white color, spreads along veins, starts from leaf tips",
            "aliases": ["xanthomonas oryzae", "bacterial blight"],
            "rag_keywords": "bacterial leaf blight water-soaked wavy edges yellow gray white rice leaf veins",
        },
        "sheath_blight": {
            "plant_part": "stem",
            "visual_symptoms": "Irregular oval lesions on leaf sheaths with white/gray centers and brown margins, spreads upward from water line",
            "aliases": ["rhizoctonia solani", "sheath blight"],
            "rag_keywords": "sheath blight irregular oval lesions white gray brown margins rice sheath",
        },
        "false_smut": {
            "plant_part": "panicle/grain",
            "visual_symptoms": "Orange to black spore masses replacing grains, only on panicles/grains",
            "aliases": ["ustilaginoidea virens", "false smut"],
            "rag_keywords": "false smut orange black spore masses grains panicle rice",
        },
        "sheath_rot": {
            "plant_part": "stem",
            "visual_symptoms": "Irregular lesions on sheaths with brown/gray/pink color, fungal growth, wilting, lodging",
            "aliases": ["sarocladium oryzae", "sheath rot"],
            "rag_keywords": "sheath rot irregular lesions brown gray pink fungal growth rice sheath",
        },
        "tungro_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Irregular orange/yellow lesions, stunting, yellowing, orange coloration",
            "aliases": ["rice tungro bacilliform virus", "tungro"],
            "rag_keywords": "tungro virus irregular orange yellow lesions stunting rice leaf",
        },
        "narrow_brown_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Elongated narrow lesions with brown/reddish-brown color, along veins",
            "aliases": ["cercospora janseana", "narrow brown spot"],
            "rag_keywords": "narrow brown leaf spot elongated narrow lesions brown reddish rice leaf veins",
        },
        "leaf_scald": {
            "plant_part": "leaf",
            "visual_symptoms": "Irregular scalded lesions with tan/white/gray color, water-soaked appearance, large spreading areas, often along leaf margins",
            "aliases": ["monographella albescens", "leaf scald"],
            "rag_keywords": "leaf scald irregular scalded tan white gray water-soaked rice leaf margins",
        },
    },
    "wheat": {
        "stem_rust": {
            "plant_part": "stem",
            "visual_symptoms": "Red/brown pustules raised on stem, elongated, can girdle stem",
            "aliases": ["puccinia graminis", "stem rust"],
            "rag_keywords": "stem rust red brown pustules raised elongated wheat stem",
        },
        "leaf_rust": {
            "plant_part": "leaf",
            "visual_symptoms": "Orange/brown pustules raised on leaf surface, small, numerous, scattered",
            "aliases": ["puccinia triticina", "leaf rust"],
            "rag_keywords": "leaf rust orange brown pustules raised small numerous wheat leaf",
        },
        "stripe_rust": {
            "plant_part": "leaf",
            "visual_symptoms": "Yellow-orange pustules in stripes along veins, numerous",
            "aliases": ["puccinia striiformis", "yellow rust", "stripe rust"],
            "rag_keywords": "stripe rust yellow orange pustules stripes veins wheat leaf",
        },
        "septoria_leaf_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Irregular oval lesions with tan centers and black pycnidia, yellow halos",
            "aliases": ["septoria tritici", "septoria blotch"],
            "rag_keywords": "septoria leaf blight irregular oval lesions tan black pycnidia wheat leaf",
        },
        "powdery_mildew": {
            "plant_part": "leaf",
            "visual_symptoms": "White to gray powdery coating on leaf surfaces, yellowing",
            "aliases": ["blumeria graminis", "powdery mildew"],
            "rag_keywords": "powdery mildew white gray powdery coating wheat leaf",
        },
        "loose_smut": {
            "plant_part": "panicle/grain",
            "visual_symptoms": "Black spore masses replacing grains, powdery, only on panicles",
            "aliases": ["ustilago tritici", "loose smut"],
            "rag_keywords": "loose smut black spore masses grains panicle wheat",
        },
        "karnal_bunt": {
            "plant_part": "panicle/grain",
            "visual_symptoms": "Black sori replacing grains, partial grain replacement, fishy odor",
            "aliases": ["tilletia indica", "karnal bunt"],
            "rag_keywords": "karnal bunt black sori grains partial fishy odor wheat panicle",
        },
        "fusarium_head_blight": {
            "plant_part": "panicle/grain",
            "visual_symptoms": "Bleached spikelets with pink/salmon spore masses, shriveled grains",
            "aliases": ["fusarium graminearum", "head blight", "scab"],
            "rag_keywords": "fusarium head blight bleached spikelets pink salmon spores wheat grain",
        },
        "rust": {
            "plant_part": "leaf",
            "visual_symptoms": "Orange/brown pustules raised on leaf surface, numerous small spots",
            "aliases": ["puccinia spp", "rust"],
            "rag_keywords": "rust orange brown pustules raised small wheat leaf",
        },
    },
    "maize": {
        "gray_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Rectangular/elongated lesions with gray/tan centers, parallel to veins, small to medium",
            "aliases": ["cercospora zeae-maydis", "gray leaf spot"],
            "rag_keywords": "gray leaf spot rectangular elongated lesions gray tan parallel veins maize leaf",
        },
        "northern_leaf_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Cigar-shaped/elliptical lesions with tan/gray centers, large, spreading",
            "aliases": ["exserohilum turcicum", "northern corn leaf blight"],
            "rag_keywords": "northern leaf blight cigar-shaped elliptical lesions tan gray large maize leaf",
        },
        "southern_leaf_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Elliptical lesions with tan/brown centers, medium to large",
            "aliases": ["bipolaris maydis", "southern corn leaf blight"],
            "rag_keywords": "southern leaf blight elliptical lesions tan brown medium large maize leaf",
        },
        "common_rust": {
            "plant_part": "leaf",
            "visual_symptoms": "Orange/brown circular pustules raised on leaf surface, numerous small spots",
            "aliases": ["puccinia sorghi", "common rust"],
            "rag_keywords": "common rust orange brown circular pustules raised maize leaf",
        },
        "southern_rust": {
            "plant_part": "leaf",
            "visual_symptoms": "Orange small pustules raised on leaf surface, numerous",
            "aliases": ["puccinia polysora", "southern rust"],
            "rag_keywords": "southern rust orange small pustules raised maize leaf",
        },
        "maize_streak_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Streaks of chlorotic tissue, yellow/white bands, stunting, reduced yield",
            "aliases": ["maize streak virus", "msv"],
            "rag_keywords": "maize streak virus streaks chlorotic yellow white bands stunting maize leaf",
        },
        "ear_rot": {
            "plant_part": "cob",
            "visual_symptoms": "Moldy growth on corn cobs, pink/white/gray fungal growth, shriveled kernels",
            "aliases": ["fusarium spp", "ear rot"],
            "rag_keywords": "ear rot moldy growth pink white gray corn cob kernels",
        },
        "stalk_rot": {
            "plant_part": "stem",
            "visual_symptoms": "Brown/pink/gray discoloration in stalk, soft rot, lodging",
            "aliases": ["fusarium spp", "stalk rot"],
            "rag_keywords": "stalk rot brown pink gray discoloration soft rot maize stem lodging",
        },
    },
    "cotton": {
        "bacterial_blight": {
            "plant_part": "leaf",
            "visual_symptoms": "Angular water-soaked lesions with brown/black color, leaf spots, defoliation",
            "aliases": ["xanthomonas axonopodis", "bacterial blight"],
            "rag_keywords": "bacterial blight angular water-soaked brown black cotton leaf defoliation",
        },
        "alternaria_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular lesions with concentric rings, dark brown with yellow halos, defoliation",
            "aliases": ["alternaria spp", "leaf spot"],
            "rag_keywords": "alternaria leaf spot circular lesions concentric rings brown yellow cotton leaf",
        },
        "cercospora_leaf_spot": {
            "plant_part": "leaf",
            "visual_symptoms": "Circular to angular lesions with gray centers and brown borders",
            "aliases": ["cercospora gossypina", "leaf spot"],
            "rag_keywords": "cercospora leaf spot circular angular lesions gray brown cotton leaf",
        },
        "cotton_leaf_curl_virus": {
            "plant_part": "leaf",
            "visual_symptoms": "Upward curling of leaves, yellowing, stunting, vein thickening, enations",
            "aliases": ["cotton leaf curl virus", "clcv"],
            "rag_keywords": "cotton leaf curl virus upward curling yellowing stunting vein thickening cotton leaf",
        },
        "fusarium_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Yellowing and wilting of lower leaves, vascular browning, stunting, one-sided wilting",
            "aliases": ["fusarium oxysporum", "vascular wilt"],
            "rag_keywords": "fusarium wilt yellowing wilting vascular browning cotton stem",
        },
        "verticillium_wilt": {
            "plant_part": "stem",
            "visual_symptoms": "V-shaped yellow lesions on lower leaves, wilting, vascular discoloration, stunting",
            "aliases": ["verticillium dahliae", "vascular wilt"],
            "rag_keywords": "verticillium wilt v-shaped lesions yellowing wilting vascular cotton stem",
        },
        "boll_rot": {
            "plant_part": "boll",
            "visual_symptoms": "Moldy growth on cotton bolls, brown/black/pink fungal growth, boll opening issues",
            "aliases": ["fusarium spp", "boll rot"],
            "rag_keywords": "boll rot moldy growth brown black pink cotton boll fungal",
        },
        "root_rot": {
            "plant_part": "root",
            "visual_symptoms": "Brown/black rotted roots, stunting, wilting, poor nutrient uptake",
            "aliases": ["rhizoctonia spp", "root rot"],
            "rag_keywords": "root rot brown black rotted stunting wilting cotton root",
        },
        "wilt": {
            "plant_part": "stem",
            "visual_symptoms": "Sudden wilting, vascular discoloration, stunting",
            "aliases": ["fusarium wilt", "verticillium wilt", "wilt"],
            "rag_keywords": "wilt sudden wilting vascular discoloration stunting cotton stem",
        },
    },
}

CROP_ALIASES = {
    "chilli": "chili",
    "chillies": "chili",
    "eggplant": "brinjal",
    "aubergine": "brinjal",
    "corn": "maize",
    "paddy": "rice",
}

SUPPORTED_CROPS = (
    "tomato",
    "potato",
    "pepper",
    "chili",
    "brinjal",
    "cucumber",
    "rice",
    "wheat",
    "maize",
    "cotton",
)

CROP_DISEASES: dict[str, tuple[str, ...]] = {
    "tomato": ("early_blight", "late_blight", "septoria_leaf_spot", "bacterial_spot", "leaf_mold", "target_spot", "mosaic_virus", "yellow_leaf_curl_virus", "fusarium_wilt", "verticillium_wilt", "powdery_mildew", "anthracnose", "fruit_rot", "bacterial_wilt", "leaf_spot"),
    "potato": ("early_blight", "late_blight", "black_scurf", "common_scab", "powdery_scab", "fusarium_wilt", "bacterial_wilt", "ring_rot", "powdery_mildew"),
    "pepper": ("bacterial_spot", "anthracnose", "phytophthora_blight", "cercospora_leaf_spot", "powdery_mildew", "mosaic_virus", "leaf_spot"),
    "chili": ("leaf_curl_virus", "bacterial_spot", "cercospora_leaf_spot", "anthracnose", "powdery_mildew", "damping_off", "fruit_rot", "leaf_curl"),
    "brinjal": ("phomopsis_blight", "little_leaf", "cercospora_leaf_spot", "alternaria_leaf_spot", "bacterial_wilt", "fusarium_wilt", "fruit_rot", "leaf_spot"),
    "cucumber": ("downy_mildew", "powdery_mildew", "angular_leaf_spot", "anthracnose", "gummy_stem_blight", "mosaic_virus", "fruit_rot"),
    "rice": ("rice_blast", "brown_spot", "bacterial_leaf_blight", "sheath_blight", "false_smut", "sheath_rot", "tungro_virus", "narrow_brown_leaf_spot", "leaf_scald"),
    "wheat": ("stem_rust", "leaf_rust", "stripe_rust", "septoria_leaf_blight", "powdery_mildew", "loose_smut", "karnal_bunt", "fusarium_head_blight", "rust"),
    "maize": ("gray_leaf_spot", "northern_leaf_blight", "southern_leaf_blight", "common_rust", "southern_rust", "maize_streak_virus", "ear_rot", "stalk_rot"),
    "cotton": ("bacterial_blight", "alternaria_leaf_spot", "cercospora_leaf_spot", "cotton_leaf_curl_virus", "fusarium_wilt", "verticillium_wilt", "boll_rot", "root_rot", "wilt"),
}

CROP_MARKET_NOTES: dict[str, str] = {
    "tomato": "Tomato mandi prices are sensitive to off-season supply gaps and perishable logistics.",
    "potato": "Potato prices track cold-storage release and rabi harvest arrivals from Punjab and UP.",
    "pepper": "Bell pepper prices respond to greenhouse supply and urban retail demand.",
    "chili": "Chili prices are volatile during peak harvest in Guntur and Byadgi markets.",
    "brinjal": "Brinjal prices depend on local mandi arrivals and monsoon transport conditions.",
    "cucumber": "Cucumber prices weaken during peak summer harvest in polyhouse clusters.",
    "rice": "Rice prices follow MSP procurement and kharif arrival patterns in major mandis.",
    "wheat": "Wheat prices align with rabi procurement and buffer-stock release policy.",
    "maize": "Maize prices track poultry feed demand and ethanol-industry offtake.",
    "cotton": "Cotton prices follow ginning arrivals and export demand from textile mills.",
}


def _entry(
    crop: str,
    disease: str,
    *,
    treatment: str,
    dosage: str,
    frequency: str,
    prevention: str,
    expected_recovery: str,
    source: str,
    page: str,
    text: str,
    title: str | None = None,
) -> CropDoc:
    return {
        "crop": crop,
        "disease": disease,
        "title": title or source,
        "source": source,
        "page": page,
        "text": text,
        "treatment": treatment,
        "dosage": dosage,
        "frequency": frequency,
        "prevention": prevention,
        "expected_recovery": expected_recovery,
    }


def _build_documents() -> list[CropDoc]:
    docs: list[CropDoc] = []

    # ---- Tomato ----
    docs.extend(
        [
            _entry(
                "tomato",
                "early_blight",
                treatment="Copper Fungicide",
                dosage="500ml per 15L water",
                frequency="Every 7-10 days",
                prevention="Remove infected leaves, avoid overhead irrigation, and maintain row spacing.",
                expected_recovery="14-18 days",
                source="ICAR Research Bulletin #2024-TB-45",
                page="12",
                text="Early blight in tomato (Alternaria solani) responds to copper fungicide sprays, leaf sanitation, and improved canopy airflow.",
            ),
            _entry(
                "tomato",
                "late_blight",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 5-7 days",
                prevention="Monitor humidity, remove infected plants, and improve drainage.",
                expected_recovery="21-28 days",
                source="ICAR Late Blight Protocol #2024-LB-07",
                page="14",
                text="Tomato late blight (Phytophthora infestans) requires mancozeb or copper sprays with rapid canopy management during humid weather.",
            ),
            _entry(
                "tomato",
                "septoria_leaf_spot",
                treatment="Chlorothalonil Fungicide",
                dosage="2g per liter water",
                frequency="Every 7-10 days",
                prevention="Remove lower leaves, ensure good air circulation, and avoid overhead irrigation.",
                expected_recovery="14-21 days",
                source="ICAR Tomato Disease Manual #2024-TB-SLS-03",
                page="15",
                text="Tomato septoria leaf spot (Septoria lycopersici) is controlled with chlorothalonil sprays and strict field sanitation.",
            ),
            _entry(
                "tomato",
                "bacterial_spot",
                treatment="Copper Oxychloride",
                dosage="3g per liter water",
                frequency="Every 7 days",
                prevention="Use disease-free seeds, avoid working wet plants, and rotate crops.",
                expected_recovery="14-18 days",
                source="ICAR Bacterial Spot Advisory #2024-TB-BS-02",
                page="8",
                text="Tomato bacterial spot (Xanthomonas spp.) requires copper sprays and strict nursery sanitation.",
            ),
            _entry(
                "tomato",
                "leaf_mold",
                treatment="Chlorothalonil Fungicide",
                dosage="2g per liter water",
                frequency="Every 7 days",
                prevention="Improve greenhouse ventilation and reduce humidity.",
                expected_recovery="10-14 days",
                source="ICAR Tomato Pathology #2024-TB-LM-05",
                page="11",
                text="Tomato leaf mold (Passalora fulva) is managed with chlorothalonil and improved greenhouse airflow.",
            ),
            _entry(
                "tomato",
                "target_spot",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 7-10 days",
                prevention="Remove infected leaves and maintain plant spacing.",
                expected_recovery="14-18 days",
                source="ICAR Target Spot Protocol #2024-TB-TS-01",
                page="6",
                text="Tomato target spot (Corynespora cassiicola) responds to mancozeb sprays and leaf removal.",
            ),
            _entry(
                "tomato",
                "mosaic_virus",
                treatment="Remove Infected Plants + Insect Control",
                dosage="N/A",
                frequency="Immediate removal",
                prevention="Control aphids and use virus-resistant varieties.",
                expected_recovery="Preventive",
                source="ICAR Virus Management #2024-TB-MV-04",
                page="3",
                text="Tomato mosaic virus requires vector control and rogueing of infected plants.",
            ),
            _entry(
                "tomato",
                "yellow_leaf_curl_virus",
                treatment="Imidacloprid + Roguing",
                dosage="0.5ml per liter water",
                frequency="Every 14 days",
                prevention="Control whitefly vectors and use virus-free seedlings.",
                expected_recovery="Varies",
                source="ICAR TYLCV Advisory #2024-TB-YLCV-06",
                page="7",
                text="Tomato yellow leaf curl virus is managed through whitefly control and resistant varieties.",
            ),
            _entry(
                "tomato",
                "verticillium_wilt",
                treatment="Crop Rotation + Resistant Varieties",
                dosage="N/A",
                frequency="Long-term",
                prevention="Use wilt-resistant varieties and soil solarization.",
                expected_recovery="Varies",
                source="ICAR Wilt Management #2024-TB-VW-02",
                page="5",
                text="Tomato verticillium wilt requires resistant varieties and crop rotation away from solanaceous crops.",
            ),
            _entry(
                "tomato",
                "anthracnose",
                treatment="Chlorothalonil Fungicide",
                dosage="2g per liter water",
                frequency="Every 7 days",
                prevention="Avoid fruit injury and maintain field hygiene.",
                expected_recovery="14-18 days",
                source="ICAR Anthracnose Guide #2024-TB-AN-01",
                page="9",
                text="Tomato anthracnose (Colletotrichum spp.) is controlled with chlorothalonil and careful harvest handling.",
            ),
            _entry(
                "tomato",
                "leaf_spot",
                treatment="Copper Fungicide",
                dosage="400ml per 15L water",
                frequency="Every 7 days",
                prevention="Maintain plant spacing and avoid leaf wetness.",
                expected_recovery="12-16 days",
                source="ICAR Leaf Spot Advisory #2024-LS-18",
                page="9",
                text="Tomato leaf spot is managed with copper fungicide, infected-leaf removal, and balanced nutrition.",
            ),
        ]
    )

    # ---- Potato ----
    docs.extend(
        [
            _entry(
                "potato",
                "early_blight",
                treatment="Chlorothalonil + Mancozeb Rotation",
                dosage="2.5g Mancozeb or 1ml Chlorothalonil per liter water",
                frequency="Every 7-10 days",
                prevention="Use certified seed tubers, rotate with cereals, and hill plants to protect stolons.",
                expected_recovery="16-22 days",
                source="CIP Potato Disease Guide #PB-12",
                page="18",
                text="Potato early blight (Alternaria solani) is managed with chlorothalonil or mancozeb sprays, defoliation of lower leaves, and CIP-recommended crop rotation.",
            ),
            _entry(
                "potato",
                "early_blight",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 7 days",
                prevention="Avoid overhead irrigation on potato canopy and maintain field sanitation.",
                expected_recovery="14-20 days",
                source="ICAR-CIP Joint Advisory #2024-PT-EB-03",
                page="7",
                text="ICAR and CIP advise mancozeb-based protectant sprays for potato early blight during warm humid periods.",
            ),
            _entry(
                "potato",
                "late_blight",
                treatment="Metalaxyl + Mancozeb",
                dosage="2g Metalaxyl-M per liter water",
                frequency="Every 5-7 days",
                prevention="Destroy volunteer potato plants and avoid irrigation during late afternoon.",
                expected_recovery="21-30 days",
                source="CIP Late Blight Management #PB-28",
                page="11",
                text="Potato late blight requires systemic metalaxyl combinations and rapid removal of infected haulms per CIP protocols.",
            ),
            _entry(
                "potato",
                "common_scab",
                treatment="Seed Treatment + Soil pH Management",
                dosage="N/A",
                frequency="Before planting",
                prevention="Maintain soil pH below 5.5 and use scab-resistant varieties.",
                expected_recovery="Preventive",
                source="ICAR Potato Scab Advisory #2024-PT-CS-04",
                page="9",
                text="Potato common scab (Streptomyces spp.) is managed through soil acidification and resistant varieties.",
            ),
            _entry(
                "potato",
                "powdery_scab",
                treatment="Seed Treatment with Flutriafol",
                dosage="Seed treatment",
                frequency="Before planting",
                prevention="Use certified seed and avoid wet, poorly drained soils.",
                expected_recovery="Preventive",
                source="ICAR Potato Scab Guide #2024-PT-PS-02",
                page="6",
                text="Potato powdery scab (Spongospora subterranea) requires seed treatment and soil drainage management.",
            ),
            _entry(
                "potato",
                "ring_rot",
                treatment="Use Certified Disease-Free Seed",
                dosage="N/A",
                frequency="Preventive",
                prevention="Plant only certified seed tubers and disinfect equipment.",
                expected_recovery="Preventive",
                source="ICAR Ring Rot Protocol #2024-PT-RR-01",
                page="4",
                text="Potato ring rot (Clavibacter michiganensis) is prevented through certified seed and strict sanitation.",
            ),
        ]
    )

    # ---- Pepper ----
    docs.extend(
        [
            _entry(
                "pepper",
                "anthracnose",
                treatment="Mancozeb + Carbendazim",
                dosage="2g Mancozeb per liter water",
                frequency="Every 10 days",
                prevention="Avoid fruit-wound injury and maintain field hygiene.",
                expected_recovery="18-24 days",
                source="ICAR Vegetable Crop Protection #2024-PP-09",
                page="16",
                text="Bell pepper anthracnose is controlled with mancozeb sprays and removal of infected fruit.",
            ),
            _entry(
                "pepper",
                "bacterial_spot",
                treatment="Copper Oxychloride",
                dosage="3g per liter water",
                frequency="Every 7 days",
                prevention="Use disease-free transplants and avoid working wet fields.",
                expected_recovery="14-18 days",
                source="ICAR Capsicum Advisory #2024-CP-04",
                page="8",
                text="Pepper bacterial spot requires copper sprays and strict sanitation in nursery and field.",
            ),
            _entry(
                "pepper",
                "cercospora_leaf_spot",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 7-10 days",
                prevention="Remove infected leaves and improve air circulation.",
                expected_recovery="14-18 days",
                source="ICAR Pepper Disease Manual #2024-CP-CLS-03",
                page="12",
                text="Pepper cercospora leaf spot (Cercospora capsici) responds to mancozeb sprays and leaf sanitation.",
            ),
            _entry(
                "pepper",
                "mosaic_virus",
                treatment="Remove Infected Plants + Aphid Control",
                dosage="N/A",
                frequency="Immediate removal",
                prevention="Control aphids and use virus-resistant varieties.",
                expected_recovery="Preventive",
                source="ICAR Pepper Virus Advisory #2024-CP-MV-02",
                page="5",
                text="Pepper mosaic virus requires vector control and rogueing of infected plants.",
            ),
            _entry(
                "pepper",
                "leaf_spot",
                treatment="Copper Fungicide",
                dosage="400ml per 15L water",
                frequency="Every 7 days",
                prevention="Maintain plant spacing and avoid leaf wetness.",
                expected_recovery="12-16 days",
                source="ICAR Pepper Leaf Spot #2024-CP-LS-01",
                page="7",
                text="Pepper leaf spot is managed with copper fungicide and infected-leaf removal.",
            ),
        ]
    )

    # ---- Chili ----
    docs.extend(
        [
            _entry(
                "chili",
                "anthracnose",
                treatment="Azoxystrobin Fungicide",
                dosage="1ml per liter water",
                frequency="Every 10 days",
                prevention="Harvest dry fruit and remove infected plant debris.",
                expected_recovery="16-20 days",
                source="ICAR Chili Production Manual #2024-CH-11",
                page="21",
                text="Chili anthracnose on fruit is managed with azoxystrobin sprays and post-rain field sanitation.",
            ),
            _entry(
                "chili",
                "leaf_curl_virus",
                treatment="Imidacloprid + Neem Oil",
                dosage="0.5ml Imidacloprid + 5ml neem oil per liter water",
                frequency="Every 14 days",
                prevention="Control whitefly vectors and remove infected plants early.",
                expected_recovery="21-28 days",
                source="ICAR Chili Virus Advisory #2024-CH-V-02",
                page="5",
                text="Chili leaf curl virus is managed through whitefly control and rogueing of infected chili plants.",
            ),
            _entry(
                "chili",
                "cercospora_leaf_spot",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 7-10 days",
                prevention="Remove infected leaves and improve air circulation.",
                expected_recovery="14-18 days",
                source="ICAR Chili Disease Manual #2024-CH-CLS-04",
                page="14",
                text="Chili cercospora leaf spot responds to mancozeb sprays and leaf sanitation.",
            ),
            _entry(
                "chili",
                "damping_off",
                treatment="Seed Treatment with Thiram",
                dosage="3g per kg seed",
                frequency="Before sowing",
                prevention="Use well-drained soil and avoid overwatering seedlings.",
                expected_recovery="Preventive",
                source="ICAR Chili Nursery Guide #2024-CH-DO-01",
                page="8",
                text="Chili damping off is prevented through seed treatment and proper nursery management.",
            ),
            _entry(
                "chili",
                "leaf_curl",
                treatment="Imidacloprid + Neem Oil",
                dosage="0.5ml Imidacloprid + 5ml neem oil per liter water",
                frequency="Every 14 days",
                prevention="Control whitefly vectors and remove infected plants early.",
                expected_recovery="21-28 days",
                source="ICAR Chili Virus Advisory #2024-CH-V-02",
                page="5",
                text="Chili leaf curl virus is managed through whitefly control and rogueing of infected chili plants.",
            ),
        ]
    )

    # ---- Brinjal ----
    docs.extend(
        [
            _entry(
                "brinjal",
                "phomopsis_blight",
                treatment="Carbendazim Fungicide",
                dosage="1g per liter water",
                frequency="Every 10 days",
                prevention="Rotate brinjal with legumes and avoid continuous cropping.",
                expected_recovery="18-22 days",
                source="ICAR Brinjal Crop Guide #2024-BR-06",
                page="13",
                text="Brinjal phomopsis blight responds to carbendazim sprays and crop rotation away from solanaceous hosts.",
            ),
            _entry(
                "brinjal",
                "bacterial_wilt",
                treatment="Streptocycline + Copper Oxychloride",
                dosage="0.5g Streptocycline + 3g copper per liter water",
                frequency="Every 7 days",
                prevention="Use grafted brinjal on wilt-resistant rootstock where available.",
                expected_recovery="Varies",
                source="ICAR Brinjal Wilt Protocol #2024-BR-W-01",
                page="4",
                text="Brinjal bacterial wilt requires copper-bactericide sprays and resistant rootstocks for long-term control.",
            ),
            _entry(
                "brinjal",
                "cercospora_leaf_spot",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 7-10 days",
                prevention="Remove infected leaves and improve air circulation.",
                expected_recovery="14-18 days",
                source="ICAR Brinjal Disease Manual #2024-BR-CLS-05",
                page="11",
                text="Brinjal cercospora leaf spot responds to mancozeb sprays and leaf sanitation.",
            ),
            _entry(
                "brinjal",
                "alternaria_leaf_spot",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 7-10 days",
                prevention="Remove infected leaves and avoid overhead irrigation.",
                expected_recovery="14-18 days",
                source="ICAR Brinjal Pathology #2024-BR-ALS-03",
                page="9",
                text="Brinjal alternaria leaf spot is managed with mancozeb sprays and field sanitation.",
            ),
            _entry(
                "brinjal",
                "leaf_spot",
                treatment="Copper Fungicide",
                dosage="400ml per 15L water",
                frequency="Every 7 days",
                prevention="Maintain plant spacing and avoid leaf wetness.",
                expected_recovery="12-16 days",
                source="ICAR Brinjal Leaf Spot #2024-BR-LS-02",
                page="7",
                text="Brinjal leaf spot is managed with copper fungicide and infected-leaf removal.",
            ),
        ]
    )

    # ---- Cucumber ----
    docs.extend(
        [
            _entry(
                "cucumber",
                "powdery_mildew",
                treatment="Sulfur Spray",
                dosage="2g sulfur per liter water",
                frequency="Every 7 days",
                prevention="Prune dense vines and improve greenhouse ventilation.",
                expected_recovery="10-14 days",
                source="ICAR Cucurbit Advisory #2024-CU-07",
                page="10",
                text="Cucumber powdery mildew is managed with sulfur sprays and canopy pruning in protected cultivation.",
            ),
            _entry(
                "cucumber",
                "downy_mildew",
                treatment="Metalaxyl + Mancozeb",
                dosage="2g per liter water",
                frequency="Every 5-7 days",
                prevention="Avoid evening irrigation and maintain row orientation for airflow.",
                expected_recovery="14-18 days",
                source="ICAR Downy Mildew Bulletin #2024-CU-DM-02",
                page="6",
                text="Cucumber downy mildew requires metalaxyl combinations during humid weather in cucurbit fields.",
            ),
            _entry(
                "cucumber",
                "anthracnose",
                treatment="Chlorothalonil Fungicide",
                dosage="2g per liter water",
                frequency="Every 7 days",
                prevention="Avoid overhead irrigation and remove infected plant debris.",
                expected_recovery="14-18 days",
                source="ICAR Cucurbit Anthracnose #2024-CU-AN-03",
                page="8",
                text="Cucumber anthracnose is controlled with chlorothalonil sprays and field sanitation.",
            ),
            _entry(
                "cucumber",
                "gummy_stem_blight",
                treatment="Benomyl Fungicide",
                dosage="1g per liter water",
                frequency="Every 10 days",
                prevention="Use disease-free seed and improve field drainage.",
                expected_recovery="18-24 days",
                source="ICAR Cucurbit Blight Guide #2024-CU-GSB-01",
                page="12",
                text="Cucumber gummy stem blight (Didymella bryoniae) responds to benomyl sprays and seed treatment.",
            ),
            _entry(
                "cucumber",
                "mosaic_virus",
                treatment="Remove Infected Plants + Aphid Control",
                dosage="N/A",
                frequency="Immediate removal",
                prevention="Control aphids and use virus-resistant varieties.",
                expected_recovery="Preventive",
                source="ICAR Cucurbit Virus Advisory #2024-CU-MV-04",
                page="5",
                text="Cucumber mosaic virus requires vector control and rogueing of infected plants.",
            ),
        ]
    )

    # ---- Rice ----
    docs.extend(
        [
            _entry(
                "rice",
                "rice_blast",
                treatment="Tricyclazole Fungicide",
                dosage="0.6g per liter water",
                frequency="At first symptom and 10 days later",
                prevention="Use blast-resistant varieties and avoid excessive nitrogen.",
                expected_recovery="21-28 days",
                source="ICAR Rice Pathology Bulletin #2024-RC-BL-05",
                page="19",
                text="Rice blast (Magnaporthe oryzae) is controlled with tricyclazole at booting stage and balanced nitrogen management.",
            ),
            _entry(
                "rice",
                "brown_spot",
                treatment="Propiconazole",
                dosage="1ml per liter water",
                frequency="Every 15 days",
                prevention="Treat seeds with fungicide and maintain potash nutrition.",
                expected_recovery="18-24 days",
                source="ICAR Rice Disease Manual #2024-RC-BS-03",
                page="12",
                text="Rice brown spot responds to propiconazole sprays and seed treatment with recommended fungicides.",
            ),
            _entry(
                "rice",
                "false_smut",
                treatment="Seed Treatment with Carbendazim",
                dosage="2g per kg seed",
                frequency="Before sowing",
                prevention="Use resistant varieties and avoid late planting.",
                expected_recovery="Preventive",
                source="ICAR Rice Smut Advisory #2024-RC-FS-04",
                page="8",
                text="Rice false smut (Ustilaginoidea virens) is managed through seed treatment and resistant varieties.",
            ),
            _entry(
                "rice",
                "sheath_rot",
                treatment="Carbendazim Fungicide",
                dosage="1g per liter water",
                frequency="At booting stage",
                prevention="Avoid excessive nitrogen and improve field drainage.",
                expected_recovery="14-18 days",
                source="ICAR Rice Sheath Rot #2024-RC-SR-02",
                page="10",
                text="Rice sheath rot (Sarocladium oryzae) responds to carbendazim sprays at booting stage.",
            ),
            _entry(
                "rice",
                "tungro_virus",
                treatment="Remove Infected Plants + Green Leafhopper Control",
                dosage="N/A",
                frequency="Immediate removal",
                prevention="Control green leafhopper vectors and use resistant varieties.",
                expected_recovery="Preventive",
                source="ICAR Rice Virus Advisory #2024-RC-TV-06",
                page="5",
                text="Rice tungro virus requires vector control and rogueing of infected plants.",
            ),
            _entry(
                "rice",
                "narrow_brown_leaf_spot",
                treatment="Propiconazole Fungicide",
                dosage="1ml per liter water",
                frequency="Every 15 days",
                prevention="Use resistant varieties and maintain balanced fertilization.",
                expected_recovery="18-24 days",
                source="ICAR Rice Leaf Spot #2024-RC-NBLS-01",
                page="7",
                text="Rice narrow brown leaf spot (Cercospora janseana) is controlled with propiconazole and resistant varieties.",
            ),
            _entry(
                "rice",
                "leaf_scald",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 10 days",
                prevention="Avoid excessive nitrogen and improve field drainage.",
                expected_recovery="14-18 days",
                source="ICAR Rice Scald Guide #2024-RC-LS-03",
                page="9",
                text="Rice leaf scald (Monographella albescens) responds to mancozeb sprays and balanced nutrition.",
            ),
        ]
    )

    # ---- Wheat ----
    docs.extend(
        [
            _entry(
                "wheat",
                "rust",
                treatment="Tebuconazole",
                dosage="1ml per liter water",
                frequency="At flag-leaf stage if rust appears",
                prevention="Grow rust-resistant wheat varieties and monitor early pustules.",
                expected_recovery="14-21 days",
                source="ICAR Wheat Rust Advisory #2024-WH-RU-04",
                page="8",
                text="Wheat rust (yellow/brown rust) is managed with tebuconazole at flag-leaf emergence and resistant cultivars.",
            ),
            _entry(
                "wheat",
                "stem_rust",
                treatment="Tebuconazole Fungicide",
                dosage="1ml per liter water",
                frequency="At flag-leaf stage",
                prevention="Use rust-resistant varieties and monitor early infection.",
                expected_recovery="14-21 days",
                source="ICAR Wheat Stem Rust #2024-WH-SR-05",
                page="10",
                text="Wheat stem rust (Puccinia graminis) is controlled with tebuconazole at flag-leaf stage.",
            ),
            _entry(
                "wheat",
                "leaf_rust",
                treatment="Propiconazole Fungicide",
                dosage="1ml per liter water",
                frequency="Every 14 days",
                prevention="Use resistant varieties and monitor early pustules.",
                expected_recovery="14-18 days",
                source="ICAR Wheat Leaf Rust #2024-WH-LR-03",
                page="7",
                text="Wheat leaf rust (Puccinia triticina) responds to propiconazole sprays and resistant varieties.",
            ),
            _entry(
                "wheat",
                "stripe_rust",
                treatment="Tebuconazole Fungicide",
                dosage="1ml per liter water",
                frequency="Every 14 days",
                prevention="Use resistant varieties and avoid late sowing.",
                expected_recovery="14-18 days",
                source="ICAR Wheat Stripe Rust #2024-WH-STR-06",
                page="9",
                text="Wheat stripe rust (yellow rust) is managed with tebuconazole and resistant cultivars.",
            ),
            _entry(
                "wheat",
                "septoria_leaf_blight",
                treatment="Propiconazole + Mancozeb",
                dosage="1ml Propiconazole + 2g Mancozeb per liter water",
                frequency="Every 12-15 days",
                prevention="Avoid dense sowing and remove volunteer wheat plants.",
                expected_recovery="16-22 days",
                source="ICAR Wheat Pathology Guide #2024-WH-SP-02",
                page="15",
                text="Wheat septoria leaf blight requires propiconazole sprays during tillering to heading stages.",
            ),
            _entry(
                "wheat",
                "loose_smut",
                treatment="Seed Treatment with Carbendazim",
                dosage="2g per kg seed",
                frequency="Before sowing",
                prevention="Use certified disease-free seed and treat seeds before sowing.",
                expected_recovery="Preventive",
                source="ICAR Wheat Smut Advisory #2024-WH-LS-04",
                page="6",
                text="Wheat loose smut (Ustilago tritici) is prevented through seed treatment with carbendazim.",
            ),
            _entry(
                "wheat",
                "fusarium_head_blight",
                treatment="Tebuconazole Fungicide",
                dosage="1ml per liter water",
                frequency="At flowering stage",
                prevention="Use resistant varieties and avoid crop residue.",
                expected_recovery="Varies",
                source="ICAR Wheat FHB Guide #2024-WH-FHB-07",
                page="11",
                text="Wheat fusarium head blight (scab) is managed with tebuconazole at flowering and resistant varieties.",
            ),
            _entry(
                "wheat",
                "karnal_bunt",
                treatment="Seed Treatment with Carbendazim",
                dosage="2g per kg seed",
                frequency="Before sowing",
                prevention="Use certified disease-free seed and avoid late sowing.",
                expected_recovery="Preventive",
                source="ICAR Wheat Disease Manual #2024-WH-KB-01",
                page="22",
                text="Wheat karnal bunt is managed through seed treatment with carbendazim and use of certified pathogen-free seed.",
            ),
        ]
    )

    # ---- Maize ----
    docs.extend(
        [
            _entry(
                "maize",
                "gray_leaf_spot",
                treatment="Azoxystrobin Fungicide",
                dosage="1ml per liter water",
                frequency="Every 14 days",
                prevention="Rotate maize with non-host crops and use tolerant hybrids.",
                expected_recovery="18-24 days",
                source="ICAR Maize Disease Manual #2024-MZ-GLS-01",
                page="9",
                text="Maize gray leaf spot is managed with azoxystrobin and hybrid rotation in kharif maize.",
            ),
            _entry(
                "maize",
                "northern_leaf_blight",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 10 days",
                prevention="Plant resistant maize hybrids and maintain field residue management.",
                expected_recovery="16-20 days",
                source="ICAR Maize Pathology Bulletin #2024-MZ-NLB-03",
                page="11",
                text="Maize northern leaf blight responds to mancozeb protectant sprays and resistant hybrid selection.",
            ),
            _entry(
                "maize",
                "southern_leaf_blight",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 10 days",
                prevention="Use resistant hybrids and avoid dense planting.",
                expected_recovery="14-18 days",
                source="ICAR Maize Southern Blight #2024-MZ-SLB-04",
                page="8",
                text="Maize southern leaf blight is controlled with mancozeb sprays and resistant hybrids.",
            ),
            _entry(
                "maize",
                "southern_rust",
                treatment="Propiconazole Fungicide",
                dosage="1ml per liter water",
                frequency="Every 14 days",
                prevention="Use resistant varieties and monitor early infection.",
                expected_recovery="14-18 days",
                source="ICAR Maize Rust Guide #2024-MZ-SR-05",
                page="7",
                text="Maize southern rust (Puccinia polysora) responds to propiconazole and resistant varieties.",
            ),
            _entry(
                "maize",
                "maize_streak_virus",
                treatment="Remove Infected Plants + Leafhopper Control",
                dosage="N/A",
                frequency="Immediate removal",
                prevention="Control leafhopper vectors and use virus-resistant varieties.",
                expected_recovery="Preventive",
                source="ICAR Maize Virus Advisory #2024-MZ-MSV-06",
                page="5",
                text="Maize streak virus requires vector control and rogueing of infected plants.",
            ),
            _entry(
                "maize",
                "stalk_rot",
                treatment="Carbendazim Fungicide",
                dosage="1g per liter water",
                frequency="At whorl stage",
                prevention="Use resistant hybrids and avoid excessive nitrogen.",
                expected_recovery="Varies",
                source="ICAR Maize Stalk Rot #2024-MZ-STK-02",
                page="10",
                text="Maize stalk rot is managed with carbendazim sprays and balanced fertilization.",
            ),
        ]
    )

    # ---- Cotton ----
    docs.extend(
        [
            _entry(
                "cotton",
                "cotton_leaf_curl_virus",
                treatment="Imidacloprid + Roguing",
                dosage="0.5ml Imidacloprid per liter water",
                frequency="Every 14 days",
                prevention="Use virus-free planting material and control whitefly populations.",
                expected_recovery="Varies",
                source="ICAR Cotton Virus Advisory #2024-CT-CLCV-02",
                page="6",
                text="Cotton leaf curl virus requires whitefly control, rogueing of infected plants, and resistant varieties.",
            ),
            _entry(
                "cotton",
                "alternaria_leaf_spot",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 10 days",
                prevention="Avoid waterlogging and maintain balanced fertilization.",
                expected_recovery="14-18 days",
                source="ICAR Cotton Crop Protection #2024-CT-ALS-04",
                page="14",
                text="Cotton alternaria leaf spot is managed with mancozeb sprays and improved field drainage.",
            ),
            _entry(
                "cotton",
                "bacterial_blight",
                treatment="Copper Oxychloride",
                dosage="3g per liter water",
                frequency="Every 7 days",
                prevention="Use disease-free seed and avoid working wet fields.",
                expected_recovery="14-18 days",
                source="ICAR Cotton Bacterial Blight #2024-CT-BB-01",
                page="8",
                text="Cotton bacterial blight (Xanthomonas axonopodis) requires copper sprays and seed treatment.",
            ),
            _entry(
                "cotton",
                "cercospora_leaf_spot",
                treatment="Mancozeb Fungicide",
                dosage="2g per liter water",
                frequency="Every 10 days",
                prevention="Remove infected leaves and improve air circulation.",
                expected_recovery="14-18 days",
                source="ICAR Cotton Leaf Spot #2024-CT-CLS-03",
                page="10",
                text="Cotton cercospora leaf spot responds to mancozeb sprays and leaf sanitation.",
            ),
            _entry(
                "cotton",
                "verticillium_wilt",
                treatment="Crop Rotation + Resistant Varieties",
                dosage="N/A",
                frequency="Long-term",
                prevention="Use wilt-resistant varieties and soil solarization.",
                expected_recovery="Varies",
                source="ICAR Cotton Wilt Management #2024-CT-VW-05",
                page="7",
                text="Cotton verticillium wilt requires resistant varieties and crop rotation away from solanaceous crops.",
            ),
            _entry(
                "cotton",
                "root_rot",
                treatment="Seed Treatment with Trichoderma",
                dosage="Seed treatment",
                frequency="Before sowing",
                prevention="Use well-drained soil and avoid waterlogging.",
                expected_recovery="Preventive",
                source="ICAR Cotton Root Rot #2024-CT-RR-06",
                page="9",
                text="Cotton root rot is prevented through seed treatment and proper soil drainage.",
            ),
            _entry(
                "cotton",
                "cotton_leaf_curl",
                treatment="Imidacloprid + Roguing",
                dosage="0.5ml Imidacloprid per liter water",
                frequency="Every 14 days",
                prevention="Use virus-free planting material and control whitefly populations.",
                expected_recovery="Varies",
                source="ICAR Cotton Virus Advisory #2024-CT-CLCV-02",
                page="6",
                text="Cotton leaf curl virus requires whitefly control, rogueing of infected plants, and resistant varieties.",
            ),
            _entry(
                "cotton",
                "wilt",
                treatment="Crop Rotation + Resistant Varieties",
                dosage="N/A",
                frequency="Long-term",
                prevention="Use wilt-resistant varieties and soil solarization.",
                expected_recovery="Varies",
                source="ICAR Cotton Wilt Management #2024-CT-WL-07",
                page="5",
                text="Cotton wilt requires resistant varieties and crop rotation away from susceptible hosts.",
            ),
        ]
    )

    # Shared disease templates per remaining crop-disease pairs
    shared_templates = {
        "powdery_mildew": (
            "Sulfur Spray",
            "2g sulfur per liter water",
            "Every 7-10 days",
            "Improve ventilation and avoid dense canopy.",
            "12-16 days",
        ),
        "fusarium_wilt": (
            "Crop Rotation + Resistant Varieties",
            "N/A",
            "Immediate and ongoing",
            "Use wilt-resistant varieties and soil sanitation.",
            "Varies",
        ),
        "fruit_rot": (
            "Copper Fungicide",
            "500ml per 15L water",
            "Every 10 days",
            "Remove infected fruit and improve airflow.",
            "16-20 days",
        ),
        "leaf_spot": (
            "Copper Fungicide",
            "400ml per 15L water",
            "Every 7 days",
            "Remove infected leaves and avoid overhead irrigation.",
            "12-16 days",
        ),
        "bacterial_wilt": (
            "Streptocycline + Copper Oxychloride",
            "0.5g Streptocycline + 3g copper per liter water",
            "Every 7 days",
            "Use disease-free seed and avoid working wet fields.",
            "Varies",
        ),
        "anthracnose": (
            "Mancozeb Fungicide",
            "2g per liter water",
            "Every 10 days",
            "Avoid fruit injury and maintain field hygiene.",
            "16-20 days",
        ),
        "cercospora_leaf_spot": (
            "Mancozeb Fungicide",
            "2g per liter water",
            "Every 7-10 days",
            "Remove infected leaves and improve air circulation.",
            "14-18 days",
        ),
        "alternaria_leaf_spot": (
            "Mancozeb Fungicide",
            "2g per liter water",
            "Every 7-10 days",
            "Remove infected leaves and avoid overhead irrigation.",
            "14-18 days",
        ),
        "verticillium_wilt": (
            "Crop Rotation + Resistant Varieties",
            "N/A",
            "Long-term",
            "Use wilt-resistant varieties and soil solarization.",
            "Varies",
        ),
        "bacterial_spot": (
            "Copper Oxychloride",
            "3g per liter water",
            "Every 7 days",
            "Use disease-free seeds and avoid working wet plants.",
            "14-18 days",
        ),
        "mosaic_virus": (
            "Remove Infected Plants + Insect Control",
            "N/A",
            "Immediate removal",
            "Control insect vectors and use virus-resistant varieties.",
            "Preventive",
        ),
        "leaf_curl": (
            "Imidacloprid + Neem Oil",
            "0.5ml Imidacloprid + 5ml neem oil per liter water",
            "Every 14 days",
            "Control whitefly vectors and remove infected plants early.",
            "21-28 days",
        ),
        "leaf_curl_virus": (
            "Imidacloprid + Neem Oil",
            "0.5ml Imidacloprid + 5ml neem oil per liter water",
            "Every 14 days",
            "Control whitefly vectors and remove infected plants early.",
            "21-28 days",
        ),
        "rust": (
            "Tebuconazole Fungicide",
            "1ml per liter water",
            "At flag-leaf stage",
            "Use rust-resistant varieties and monitor early pustules.",
            "14-21 days",
        ),
        "common_rust": (
            "Tebuconazole Fungicide",
            "1ml per liter water",
            "At flag-leaf stage",
            "Use rust-resistant varieties and monitor early pustules.",
            "14-21 days",
        ),
    }

    for crop, diseases in CROP_DISEASES.items():
        existing = {(doc["crop"], doc["disease"]) for doc in docs}
        for disease in diseases:
            if (crop, disease) in existing:
                continue
            if disease not in shared_templates:
                continue
            treatment, dosage, frequency, prevention, recovery = shared_templates[disease]
            crop_label = crop.replace("_", " ").title()
            disease_label = disease.replace("_", " ").title()
            docs.append(
                _entry(
                    crop,
                    disease,
                    treatment=treatment,
                    dosage=dosage,
                    frequency=frequency,
                    prevention=prevention,
                    expected_recovery=recovery,
                    source=f"ICAR {crop_label} Protection Manual",
                    page="1",
                    text=f"{disease_label} in {crop_label.lower()} is managed with {treatment.lower()} following ICAR field recommendations.",
                )
            )

    return docs


@lru_cache(maxsize=1)
def all_documents() -> list[CropDoc]:
    return _build_documents()


def normalize_crop(crop: str | None) -> str:
    text = (crop or "").strip().lower().replace("-", " ")
    if text in CROP_ALIASES:
        return CROP_ALIASES[text]
    return text


def normalize_disease(disease: str | None) -> str:
    return (disease or "").strip().lower().replace("-", " ").replace(" ", "_")


def diseases_for_crop(crop: str | None) -> tuple[str, ...]:
    normalized = normalize_crop(crop)
    return CROP_DISEASES.get(normalized, ("leaf_spot", "early_blight", "powdery_mildew"))


def disease_candidates_for_crop(crop: str | None) -> list[dict[str, str]]:
    """Return unique disease classes for a crop with text labels for model scoring."""
    normalized_crop = normalize_crop(crop)
    seen: set[str] = set()
    candidates: list[dict[str, str]] = []

    for doc in all_documents():
        if normalize_crop(doc.get("crop")) != normalized_crop:
            continue
        disease = normalize_disease(doc.get("disease"))
        if disease in seen:
            continue
        seen.add(disease)
        candidates.append(
            {
                "disease": disease,
                "label_text": (
                    f"{normalized_crop} {disease.replace('_', ' ')}: "
                    f"{doc.get('text', '').strip()}"
                ),
            }
        )

    if not candidates:
        for disease in diseases_for_crop(normalized_crop):
            slug = normalize_disease(disease)
            candidates.append(
                {
                    "disease": slug,
                    "label_text": f"{normalized_crop} {slug.replace('_', ' ')}",
                }
            )
    return candidates


def infer_disease_for_crop(
    crop: str | None,
    symptoms: list[str],
    description: str | None,
    seed: int,
) -> str:
    """Pick a plausible disease for the crop from symptoms or fingerprint."""
    normalized_crop = normalize_crop(crop)
    options = diseases_for_crop(normalized_crop)
    text = " ".join([normalized_crop, " ".join(symptoms), description or ""]).lower()

    keyword_map = {
        "late_blight": ("late blight", "water soaked", "water_soaked"),
        "early_blight": ("early blight", "brown rings", "brown"),
        "leaf_spot": ("leaf spot", "spots"),
        "powdery_mildew": ("powdery", "mildew", "mold"),
        "fusarium_wilt": ("wilt", "wilting"),
        "fruit_rot": ("rot", "fruit rot"),
        "bacterial_wilt": ("bacterial", "wilt"),
        "rice_blast": ("blast",),
        "rust": ("rust",),
        "anthracnose": ("anthracnose",),
        "leaf_curl": ("curl", "leaf curl"),
        "downy_mildew": ("downy",),
    }

    for disease in options:
        keys = keyword_map.get(disease, (disease.replace("_", " "),))
        if any(key in text for key in keys):
            return disease

    return options[seed % len(options)]


def match_documents(
    crop: str | None,
    disease: str | None,
    query: str = "",
    *,
    top_k: int = 5,
    seed: int = 0,
) -> list[tuple[CropDoc, float]]:
    """Return crop+disease matched documents ranked by relevance."""
    from rag.embedder import cosine_similarity, simple_embedding

    normalized_crop = normalize_crop(crop)
    normalized_disease = normalize_disease(disease).replace("_", " ")
    query_embedding = simple_embedding(query) if query else {}

    matches: list[tuple[CropDoc, float]] = []
    for doc in all_documents():
        doc_crop = normalize_crop(doc.get("crop"))
        doc_disease = doc.get("disease", "").replace("_", " ").lower()
        if doc_crop != normalized_crop:
            continue
        if normalized_disease and normalized_disease not in doc_disease:
            continue

        base_score = 1.0 + (seed % 100) / 1000.0
        if query_embedding:
            base_score += cosine_similarity(query_embedding, simple_embedding(doc.get("text", "")))
        doc_bias = (hash(doc.get("source", "")) % 100) / 1000.0
        matches.append((doc, base_score + doc_bias))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[:top_k]


def crop_market_note(crop: str | None) -> str:
    return CROP_MARKET_NOTES.get(normalize_crop(crop), "Monitor local mandi trends before selling.")
