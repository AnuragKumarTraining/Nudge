"""
Central configuration. Nothing here does work — it's the single place
every module reads thresholds/class-lists from, so tuning the pipeline
never means hunting through function bodies.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from the project root

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


CLASS_SYNONYMS = {
    "dining table": "table",
    "coffee cup": "cup",
    "wine glass": "glass",
    "flower vase": "vase",
    "ceiling fan": "fan",
    "light bulb": "light",
    "trash bin": "dustbin",
     "lamp": "light",
    "potted plant": "plant",      # COCO name  vs  YOLO-World name
    "couch": "sofa",              # COCO
    "tv": "television",           # COCO
    "menu card": "menu",
    "salt shaker": "condiment holder",
    "pepper shaker": "condiment holder",
}
# Big "stuff" regions. Their bboxes cover a large part of the frame, so they are never
# treated as objects sitting ON something, and never count as "on top of" a surface.
STRUCTURAL_CLASSES = {"floor", "wall", "ceiling", "window", "door"}
# Surfaces whose "exposed" area (bbox minus the objects sitting on it) is what gets
# scanned for stains. SURFACE_ANCHOR_CLASSES (above) stays table-only because it is
# also what relationships.py uses to group objects.
EXPOSED_SURFACE_CLASSES = {"table", "dining table", "floor", "wall", "ceiling"}
# Classes that are worth an OpenCV stain scan (windows / lights / fans / mirrors are
# covered by the VLM condition+cleanliness check instead - adaptive thresholding on
# glass and light fixtures only produces reflections).
STAIN_SCAN_CLASSES = {
    "table", "dining table", "chair", "sofa", "bench", "floor", "wall",
    "plate", "bowl", "cup", "coffee cup", "glass", "wine glass",
    "tray", "napkin", "vase", "flower vase",
}
# Never send these to the condition VLM (people are not inspection targets, and it
# avoids uploading staff faces to a third-party API for no reason).
CONDITION_SKIP_CLASSES = {"person"}
# Output folder for the per-stage JSON artifacts (one file per capture_id, so
# concurrent captures from different cafes never overwrite each other).
OUTPUT_DIR = os.getenv("NUDGE_OUTPUT_DIR", "outputs")

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
STAIN_MAX_CANDIDATES_PER_OBJECT = 5   # each candidate = 1 VLM call, so cap it
STAIN_PERIODIC_PEAK_RATIO = 60.0      # DC-removed, windowed FFT peak/mean (see stain_detection.py)
STAIN_MAX_CANDIDATES_TOTAL = 30       # hard ceiling of stain crops sent to the VLM per image
STAIN_MASK_ERODE_PX = 4               # shrink the "exposed surface" mask so object edges don't leak in

VLM_PROVIDER = os.getenv("VLM_PROVIDER", "nvidia").strip().lower()

_VLM_PROVIDER_DEFAULTS = {
    "nvidia": {
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "image_url_format": "nested",
    },
    "ollama": {
        "api_key_env": "OLLAMA_API_KEY",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5vl:7b ",
        "extra_body": {},
        "image_url_format": "flat",
    },
    "huggingface": {
        "api_key_env": "HF_TOKEN",
        "base_url": "https://router.huggingface.co/v1",
        "model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp",
        "extra_body": {"chat_template_kwargs": {
            "enable_thinking": False
        }},
        "image_url_format": "nested",
    },
}

if VLM_PROVIDER not in _VLM_PROVIDER_DEFAULTS:
    raise ValueError(
        f"Unknown VLM_PROVIDER={VLM_PROVIDER!r} - expected one of {list(_VLM_PROVIDER_DEFAULTS)}"
    )

_vlm_defaults = _VLM_PROVIDER_DEFAULTS[VLM_PROVIDER]

VLM_API_KEY_ENV = os.getenv("VLM_API_KEY_ENV", _vlm_defaults["api_key_env"])
VLM_BASE_URL = os.getenv("VLM_BASE_URL", _vlm_defaults["base_url"])
VLM_MODEL = os.getenv("VLM_MODEL", _vlm_defaults["model"])
VLM_EXTRA_BODY = _vlm_defaults["extra_body"]
VLM_IMAGE_URL_FORMAT = _vlm_defaults["image_url_format"]

# ---- VLM call tuning (provider-agnostic) ----
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.1"))
VLM_INVENTORY_MAX_TOKENS = int(os.getenv("VLM_INVENTORY_MAX_TOKENS", "5000"))
VLM_CONDITION_MAX_TOKENS = int(os.getenv("VLM_CONDITION_MAX_TOKENS", "300"))
VLM_DEBUG = os.getenv("VLM_DEBUG", "0") == "1"
VLM_MAX_WORKERS = int(os.getenv("VLM_MAX_WORKERS", "4"))