"""
Third discovery branch: VLM scene/object inventory.

YOLOv8 knows COCO classes, YOLO-World knows whatever is in YOLO_WORLD_CLASSES —
both can miss things. This branch asks the VLM to list anything else visible.
Its boxes are LOW precision: entries are gap-fillers (presence evidence), never
used for geometry, stain crops, or polygon extraction.
"""
import numpy as np

from config.settings import SURFACE_RELEVANT_CLASSES, MATERIAL_HINT_BY_CLASS
from models.vlm_client import VLMClient


def detect_vlm_inventory(image: np.ndarray, objects: list[dict], vlm_client: VLMClient) -> list[dict]:
    if not vlm_client.is_configured:
        return []

    known = ", ".join(sorted({o["class"] for o in objects}))
    h, w = image.shape[:2]

    items = vlm_client.get_scene_inventory(image, known)
    if not items:
        return []

    inventory = []
    for i, item in enumerate(items):
        x1, y1, x2, y2 = item["bbox_normalized"]
        cls = item["class"]
        inventory.append({
            "object_id": f"obj_vlm_{i:03d}_{cls.replace(' ', '_')}",
            "class": cls,
            "confidence": 0.5,  # VLM boxes are low-precision; mark them
            "bbox": {"x1": int(x1 * w), "y1": int(y1 * h), "x2": int(x2 * w), "y2": int(y2 * h)},
            "bbox_normalized": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "centroid": {"x": int((x1 + x2) * w / 2), "y": int((y1 + y2) * h / 2)},
            "reported_count": item.get("count", 1),
            "surface_relevant": cls in SURFACE_RELEVANT_CLASSES,
            "material_hint": MATERIAL_HINT_BY_CLASS.get(cls, "pending_vlm"),
            "condition": "pending_vlm",
            "relationships": {},
            "source": "vlm_inventory",  # downstream: don't trust this bbox for geometry
        })
    return inventory