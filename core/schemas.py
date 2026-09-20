"""
These TypedDicts document the JSON contract passed between pipeline stages.
They aren't enforced at runtime (plain dicts still flow through the code) —
they exist so every module and every new contributor can see the agreed
shape in one place instead of reverse-engineering it from usage.
"""
from typing import TypedDict, Optional


class BBox(TypedDict):
    x1: int
    y1: int
    x2: int
    y2: int


class BBoxNormalized(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


class Centroid(TypedDict):
    x: int
    y: int


class MaskPolygon(TypedDict):
    points_pixel: list[list[int]]
    points_normalized: list[list[float]]


class DetectedObject(TypedDict, total=False):
    object_id: str
    class_: str  # "class" is a reserved word, code uses the literal key "class"
    confidence: float
    bbox: BBox
    bbox_normalized: BBoxNormalized
    centroid: Centroid
    orientation_deg: float
    mask_polygon: MaskPolygon
    surface_relevant: bool
    material_hint: str
    condition: str          # good | worn | damaged | broken | missing_parts | unclear | not_assessed | pending_vlm
    cleanliness: str        # clean | slightly_dirty | dirty | unclear | not_assessed | pending_vlm
    issues: list[str]       # short VLM phrases, e.g. "chipped rim"
    relationships: dict     # relation (on|near), belongs_to, relative_position, objects_on_surface
    source: str             # best-geometry source: yolo | yolo_world | vlm_inventory
    sources: list[str]      # every branch that saw this object (after dedup)
    reported_count: int     # VLM-only: how many instances the single loose box stands for


class StainCandidate(TypedDict, total=False):
    bbox: BBox
    area_px: int
    circularity: float
    is_periodic_pattern: bool
    raw_confidence: float
    adjusted_confidence: float
    vlm_verdict: Optional[str]
    is_confirmed_stain: bool  # True if VLM confirmed the stain, False otherwise


class SurfaceAnalysisEntry(TypedDict, total=False):
    object_id: str
    class_: str
    stain_candidates: list[StainCandidate]


class Stage3Output(TypedDict, total=False):
    capture_id: str
    status: str                 # "processed" | "rejected"
    preprocessing: dict         # quality_gate + lighting (no alignment - there is no master image)
    extractor_used: str
    sources: dict               # raw hits per discovery branch before dedup
    total_objects: int
    object_counts: dict         # over the MERGED inventory
    objects: list[DetectedObject]
    surface_analysis: list[SurfaceAnalysisEntry]
    layout_description: str
    artifacts: dict            # paths to all artifacts written by this stage
