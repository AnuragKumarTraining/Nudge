"""
Central configuration. Nothing here does work — it's the single place
every module reads thresholds/class-lists from, so tuning the pipeline
never means hunting through function bodies.
"""
import os

# ---- YOLO ----

# ============================================================
# YOLO — STANDARD OBJECT DETECTION
# ============================================================

YOLO_WEIGHTS_PATH = os.getenv(
    "YOLO_WEIGHTS_PATH",
    "yolov8n.pt"
)

YOLO_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "YOLO_CONF_THRESHOLD",
        "0.35"
    )
)


# ============================================================
# YOLO-WORLD — OPEN VOCABULARY OBJECT DETECTION
# ============================================================
#
# YOLOv8n is trained on COCO and cannot detect things such as:
# floor, ceiling, window, wall, spoon, napkin, etc.
#
# YOLO-World is used for these additional/domain-specific
# objects.
# ============================================================

YOLO_WORLD_WEIGHTS_PATH = os.getenv(
    "YOLO_WORLD_WEIGHTS_PATH",
    "yolov8s-worldv2.pt"
)

YOLO_WORLD_CONFIDENCE_THRESHOLD = float(
    os.getenv(
        "YOLO_WORLD_CONF_THRESHOLD",
        "0.15"
    )
)

'''
# COCO (what yolov8n.pt was trained on) has no classes for these — this is
# exactly the "not floors, windows, other things" gap. Two ways to close it:
#   1) swap YOLO_WEIGHTS_PATH for a fine-tuned model that includes these classes
#   2) wire up detect_open_vocabulary_objects() in object_detection/detector.py,
#      which uses YOLO-World with this exact list as the prompt classes
NON_COCO_SURFACE_CLASSES = [
    "floor", "window", "wall", "napkin", "spoon", "fork",
    "flower vase", "menu card", "condiment holder",
]

# Which detected classes actually need surface-condition checking (stains,
# texture). A "person" or "cell phone" detection should never enter that path.
SURFACE_RELEVANT_CLASSES = {
    "dining table", "chair", "cup", "bowl", "wine glass", "vase", "bottle",
}

# Cheap heuristic fallback so material_hint isn't blank before the VLM runs.
# The VLM call in enrich_with_vlm() overwrites this when a client is configured.
MATERIAL_HINT_BY_CLASS = {
    "dining table": "wood",
    "chair": "fabric",
    "cup": "ceramic",
    "bowl": "ceramic",
    "wine glass": "glass",
    "bottle": "glass",
    "vase": "ceramic",
}
'''

# Objects that anchor spatial relationships (everything else gets located
# "relative to" the nearest one of these).
SURFACE_ANCHOR_CLASSES = {"dining table", "table"}



# Objects that YOLO-World should search for.
#
# Keep this list focused on things that can actually occur in
# your inspection environment.

YOLO_WORLD_CLASSES = [

    # --------------------------------------------------------
    # STRUCTURE / ROOM
    # --------------------------------------------------------

    "floor",
    "ceiling",
    "wall",

    "window",
    "door",

    # --------------------------------------------------------
    # FURNITURE
    # --------------------------------------------------------

    "table",
    "dining table",
    "chair",
    "sofa",
    "bench",

    # --------------------------------------------------------
    # DINING OBJECTS
    # --------------------------------------------------------

    "plate",
    "bowl",
    "cup",
    "coffee cup",
    "glass",
    "wine glass",

    "spoon",
    "fork",
    "knife",

    "tray",
    "napkin",

    # --------------------------------------------------------
    # TABLE ACCESSORIES
    # --------------------------------------------------------

    "menu",
    "menu card",
    "condiment holder",
    "salt shaker",
    "pepper shaker",

    "flower vase",
    "vase",
    "table decoration",

    # --------------------------------------------------------
    # CAFE EQUIPMENT / ENVIRONMENT
    # --------------------------------------------------------

    "fan",
    "ceiling fan",

    "light",
    "light bulb",
    "lamp",

    "air conditioner",

    "dustbin",
    "trash bin",

    "fire extinguisher",

    # --------------------------------------------------------
    # OTHER VISIBLE ELEMENTS
    # --------------------------------------------------------

    "plant",
    "mirror",
    "clock",
    "television",
    "shelf",
    "sign",
    "menu board",
]


# ============================================================
# SURFACE-CONDITION OBJECTS
# ============================================================
#
# These objects can subsequently be inspected for:
# - dirt
# - stains
# - scratches
# - damage
# - cleanliness
#
# This is NOT the object-detection list.
# It controls which detected objects enter the
# surface/condition pipeline.
# ============================================================

SURFACE_RELEVANT_CLASSES = {

    # Furniture / surfaces
    "table",
    "dining table",
    "chair",
    "sofa",
    "bench",

    # Structural surfaces
    "floor",
    "wall",
    "ceiling",

    # Glass / openings
    "window",
    "door",

    # Dining objects
    "plate",
    "bowl",
    "cup",
    "coffee cup",
    "glass",
    "wine glass",

    # Accessories
    "tray",
    "napkin",
    "vase",
    "flower vase",

    # Equipment
    "fan",
    "ceiling fan",
    "light",
    "light bulb",
    "lamp",

    # Other
    "mirror",
    "shelf",
}


# ============================================================
# MATERIAL HINTS
# ============================================================
#
# These are only cheap defaults.
# VLM can overwrite/refine them later.
# ============================================================

MATERIAL_HINT_BY_CLASS = {

    "table": "wood",
    "dining table": "wood",

    "chair": "mixed",

    "sofa": "fabric",
    "bench": "wood",

    "floor": "unknown",
    "wall": "unknown",
    "ceiling": "unknown",

    "window": "glass",
    "door": "wood",

    "plate": "ceramic",
    "bowl": "ceramic",

    "cup": "ceramic",
    "coffee cup": "ceramic",

    "glass": "glass",
    "wine glass": "glass",

    "spoon": "metal",
    "fork": "metal",
    "knife": "metal",

    "tray": "metal",
    "napkin": "fabric",

    "vase": "ceramic",
    "flower vase": "ceramic",

    "mirror": "glass",

    "light": "mixed",
    "light bulb": "glass",
    "lamp": "mixed",

    "fan": "metal/plastic",
    "ceiling fan": "metal/plastic",
}


# ---- Quality gate ----
BLUR_VARIANCE_THRESHOLD = float(os.getenv("BLUR_VARIANCE_THRESHOLD", "100.0"))

# ---- Lighting ----
AMBIENT_LUX_CLAHE_THRESHOLD = float(os.getenv("AMBIENT_LUX_CLAHE_THRESHOLD", "250.0"))

# ---- Alignment ----
ORB_MAX_FEATURES = 5000
ORB_MATCH_RATIO = 0.75
ORB_RANSAC_THRESHOLD = 0.5
ORB_MIN_GOOD_MATCHES = 20
YAW_DELTA_WARNING_DEG = 45.0  # beyond this, alignment is likely unreliable

# ---- Surface / stain detection ----
STAIN_MIN_AREA_PX = 30
STAIN_MAX_AREA_PX = 5000
STAIN_ADAPTIVE_BLOCK_SIZE = 25
STAIN_ADAPTIVE_C = 5

# ---- VLM: Qwen-VL via Alibaba Cloud DashScope (OpenAI-compatible endpoint) ----
DASHSCOPE_API_KEY_ENV = "DASHSCOPE_API_KEY"
# Singapore/international endpoint by default. Use
# https://dashscope.aliyuncs.com/compatible-mode/v1 for the Beijing region, or
# https://dashscope-us.aliyuncs.com/compatible-mode/v1 for Virginia — set
# DASHSCOPE_BASE_URL to override rather than editing this file per-environment.
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")


# config/settings.py

VLM_API_KEY_ENV = os.getenv("VLM_API_KEY_ENV", "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")

VLM_BASE_URL = os.getenv(
    "VLM_BASE_URL",
    "https://integrate.api.nvidia.com/v1"
)

VLM_MODEL = os.getenv(
    "VLM_MODEL",
    "nvidia/nemotron-nano-vl-v2"
)