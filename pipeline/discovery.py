"""
STAGE A - OBJECT DISCOVERY  (hybrid: YOLOv8 + YOLO-World + VLM inventory)

Input : ONE pre-processed image (+ capture metadata)      <- no master / reference image
Output: {out_dir}/{capture_id}_object_discovery.json      <- the artifact Stage B consumes

Answers only "what is in the photo, where, and how is it arranged".
It never judges condition, cleanliness or stains - that is Stage B.

    image ─┬─► YOLOv8      (COCO objects)      ─┐
           ├─► YOLO-World  (open-vocab objects) ├─► 3-way merge + dedup ─► inventory
           └─► VLM         (gap-filling list)  ─┘
                                                  │
                                                  ├─► bboxes
                                                  ├─► polygons (+ orientation)
                                                  └─► relationships
"""
import numpy as np

from config.settings import YOLO_WORLD_CLASSES, OUTPUT_DIR
from core.artifacts import save_artifact
from models.yolo_loader import load_yolo
from models.vlm_client import VLMClient
from object_detection.detector import detect_objects, detect_open_vocabulary_objects
from object_detection.vlm_inventory import detect_vlm_inventory
from object_detection.merge import merge_detections, count_objects
from object_detection.polygon import extract_polygon
from object_detection.relationships import compute_relationships


def run_object_discovery(image: np.ndarray, capture_metadata: dict,
                         vlm_client: VLMClient | None = None,
                         image_path: str | None = None,
                         preprocessing: dict | None = None,
                         out_dir: str = OUTPUT_DIR) -> dict:
    img_h, img_w = image.shape[:2]
    vlm_client = vlm_client or VLMClient()

    # ---- Branch 1: YOLOv8 (COCO) ----
    detection = detect_objects(image, load_yolo())
    standard_objects = detection["objects"]

    # ---- Branch 2: YOLO-World (open vocabulary) ----
    open_vocab_objects = detect_open_vocabulary_objects(image, YOLO_WORLD_CLASSES)

    # ---- Branch 3: VLM inventory (told what is already found, so it fills GAPS only) ----
    vlm_inventory = detect_vlm_inventory(image, standard_objects + open_vocab_objects, vlm_client)

    # ---- 3-way merge + dedup -> COMPLETE OBJECT INVENTORY ----
    objects = merge_detections(standard_objects, open_vocab_objects, vlm_inventory)

    # ---- per-object polygons + orientation ----
    for obj in objects:
        poly = extract_polygon(image, obj)
        obj["orientation_deg"] = poly["orientation_deg"]
        obj["mask_polygon"] = {
            "points_pixel": poly["points_pixel"],
            "points_normalized": poly["points_normalized"],
        }

    # ---- relationships (on / near / belongs_to / objects_on_surface) ----
    objects = compute_relationships(objects)

    # ---- DISCOVERY OUTPUT ARTIFACT ----
    discovery = {
        "capture_id": capture_metadata.get("capture_id"),
        "master_reference_id": capture_metadata.get("master_reference_id"),
        "status": "object_discovery_complete",
        "image": {"path": image_path, "width": img_w, "height": img_h},
        "preprocessing": preprocessing or {},
        "vlm_configured": vlm_client.is_configured,
        "sources": {
            "yolo": len(standard_objects),
            "yolo_world": len(open_vocab_objects),
            "vlm_inventory": len(vlm_inventory),
        },
        "total_objects": len(objects),
        "object_counts": count_objects(objects),
        "objects": objects,
    }
    discovery["artifact_path"] = save_artifact(out_dir, discovery["capture_id"], "object_discovery", discovery)

    return discovery