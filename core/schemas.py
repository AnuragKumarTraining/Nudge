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
    condition: str
    relationships: dict


class StainCandidate(TypedDict, total=False):
    bbox: BBox
    area_px: int
    circularity: float
    is_periodic_pattern: bool
    raw_confidence: float
    adjusted_confidence: float
    vlm_verdict: Optional[str]


class SurfaceAnalysisEntry(TypedDict, total=False):
    object_id: str
    class_: str
    stain_candidates: list[StainCandidate]


class Stage3Output(TypedDict, total=False):
    capture_id: str
    master_reference_id: Optional[str]
    quality_gate: dict
    alignment_result: dict
    objects: list[DetectedObject]
    object_counts: dict
    surface_analysis: list[SurfaceAnalysisEntry]
    layout_description: str
    extractor_used: str
    detection_confidence: float
