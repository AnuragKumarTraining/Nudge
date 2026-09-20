"""
Object identity + count + bbox. This is PASS 1 of the original
feature_extraction.py, pulled out on its own so polygon extraction and
relationship computation (separate concerns) live in their own modules.
"""
import numpy as np
from ultralytics import YOLOWorld
from config.settings import (
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_WORLD_CONFIDENCE_THRESHOLD,
    YOLO_WORLD_WEIGHTS_PATH,
    SURFACE_RELEVANT_CLASSES,
    MATERIAL_HINT_BY_CLASS,
)
from models.yolo_loader import load_yolo_world


def detect_objects(image: np.ndarray, yolo_model) -> dict:
    img_h, img_w = image.shape[:2]
    results = yolo_model(image, verbose=False)

    objects = []
    object_counts: dict[str, int] = {}
    total_conf, n = 0.0, 0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            n += 1
            total_conf += conf
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            object_id = f"obj_{n:03d}_{class_name.replace(' ', '_')}"

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            objects.append({
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
            })

    return {
        "objects": objects,
        "object_counts": object_counts,
        "detection_confidence": round(total_conf / n, 3) if n else 0.0,
    }

'''
def detect_open_vocabulary_objects(image: np.ndarray, class_list: list[str]) -> list[dict]:
    """
    Extensibility hook for exactly the gap you flagged: plain COCO-YOLO
    (yolov8n.pt) has no classes for floor, window, spoon, napkin, etc.
    This uses YOLO-World, which takes a free-text class list instead of a
    fixed vocabulary.

    Not wired into the main pipeline yet on purpose — validate it on real
    cafe photos first (accuracy on "floor"/"window" from an open-vocab model
    is noticeably weaker than a real fine-tuned detector), then merge its
    output into detect_objects()'s object list using the same dict shape
    before switching this on by default. See NON_COCO_SURFACE_CLASSES in
    config/settings.py for the class list to start with.
    """


    model = load_yolo_world()
    model.set_classes(class_list)
    results = model(image, verbose=False)

    img_h, img_w = image.shape[:2]
    objects = []
    for r in results:
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_name = class_list[int(box.cls[0])]
            objects.append({
                "object_id": f"obj_ov_{i:03d}_{class_name.replace(' ', '_')}",
                "class": class_name,
                "confidence": round(float(box.conf[0]), 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "bbox_normalized": {
                    "x1": round(x1 / img_w, 4), "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4), "y2": round(y2 / img_h, 4),
                },
                "centroid": {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2},
                "surface_relevant": False,
                "material_hint": "pending_vlm",
                "condition": "pending_vlm",
                "relationships": {},
                "source": "open_vocabulary",
            })
    return objects

'''

def detect_open_vocabulary_objects(
    image: np.ndarray,
    class_list: list[str]
) -> list[dict]:

    model = load_yolo_world()

    model.set_classes(class_list)

    results = model(
        image,
        verbose=False,
        conf=YOLO_WORLD_CONFIDENCE_THRESHOLD
    )

    img_h, img_w = image.shape[:2]

    objects = []
    counter = 0

    for r in results:

        for box in r.boxes:

            conf = float(box.conf[0])

            if conf < YOLO_WORLD_CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])

            if class_id >= len(class_list):
                continue

            class_name = class_list[class_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            counter += 1

            objects.append({
                "object_id": (
                    f"obj_ov_{counter:03d}_"
                    f"{class_name.replace(' ', '_')}"
                ),

                "class": class_name,

                "confidence": round(
                    conf,
                    3
                ),

                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },

                "bbox_normalized": {
                    "x1": round(x1 / img_w, 4),
                    "y1": round(y1 / img_h, 4),
                    "x2": round(x2 / img_w, 4),
                    "y2": round(y2 / img_h, 4),
                },

                "centroid": {
                    "x": (x1 + x2) // 2,
                    "y": (y1 + y2) // 2,
                },

                "surface_relevant": (
                    class_name in SURFACE_RELEVANT_CLASSES
                    or class_name in {
                        "floor",
                        "window",
                        "wall",
                        "ceiling",
                        "countertop",
                        "table",
                    }
                ),

                "material_hint": (
                    MATERIAL_HINT_BY_CLASS.get(
                        class_name,
                        "pending_vlm"
                    )
                ),

                "condition": "pending_vlm",
                "relationships": {},
                "source": "yolo_world",
            })

    return objects