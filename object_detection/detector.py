"""
Discovery branches 1 and 2 (both are cheap, precise-geometry detectors):
  * detect_objects()                 - YOLOv8 (COCO vocabulary)
  * detect_open_vocabulary_objects() - YOLO-World (any text class list: floor, window,
                                       spoon, napkin, fire extinguisher, ...)
Both return objects in the SAME dict shape, tagged with "source", so
object_detection/merge.py can combine them with the VLM inventory branch.
"""
import numpy as np

from config.settings import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_WORLD_CONFIDENCE_THRESHOLD,
    SURFACE_RELEVANT_CLASSES,
    MATERIAL_HINT_BY_CLASS,
)
from models.yolo_loader import load_yolo_world

def _make_object(object_id: str, class_name: str, conf: float, xyxy, img_w: int, img_h: int,
                 source: str) -> dict:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    return {
        "object_id": object_id,
        "class": class_name,
        "confidence": round(conf, 3),
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "bbox_normalized": {
            "x1": round(x1 / img_w, 4), "y1": round(y1 / img_h, 4),
            "x2": round(x2 / img_w, 4), "y2": round(y2 / img_h, 4),
        },
        "centroid": {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
        "surface_relevant": class_name in SURFACE_RELEVANT_CLASSES,
        "material_hint": MATERIAL_HINT_BY_CLASS.get(class_name, "pending_vlm"),
        "condition": "pending_vlm",
        "relationships": {},
        "source": source,
    }


def detect_objects(image: np.ndarray, yolo_model) -> dict:
    img_h, img_w = image.shape[:2]
    results = yolo_model(image, verbose=False)

    """Branch 1: YOLOv8 / COCO."""

    objects: list[dict] = []
    object_counts: dict[str, int] = {}
    total_conf = 0.0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            n = len(objects) + 1
            total_conf += conf
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            objects.append(_make_object(
                f"obj_{n:03d}_{class_name.replace(' ', '_')}", class_name, conf,
                box.xyxy[0].tolist(), img_w, img_h, source="yolo",
            ))

    return {
        "objects": objects,
        "object_counts": object_counts,
        "detection_confidence": round(total_conf / len(objects), 3) if objects else 0.0,
    }


def detect_open_vocabulary_objects(image: np.ndarray, class_list: list[str]) -> list[dict]:
    """Branch 2: YOLO-World with a free-text class list."""
    model = load_yolo_world(class_list)          # set_classes() only re-runs if the list changed
    results = model(image, verbose=False, conf=YOLO_WORLD_CONFIDENCE_THRESHOLD)

    img_h, img_w = image.shape[:2]

    objects:list[dict] = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])

            if conf< YOLO_WORLD_CONFIDENCE_THRESHOLD or class_id >= len(class_list):
                continue

            class_name = class_list[class_id]
            n = len(objects) + 1
            objects.append(_make_object(
                f"obj_ov_{n:03d}_{class_name.replace(' ', '_')}", class_name, conf,
                box.xyxy[0].tolist(), img_w, img_h, source="yolo_world",
            ))
    return objects