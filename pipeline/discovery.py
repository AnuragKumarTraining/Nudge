"""
STAGE A — OBJECT DISCOVERY (hybrid: YOLOv8 + YOLO-World + VLM inventory).

Input : raw capture image (+ optional master for alignment)
Output: metadata/object_discovery.json  +  a result dict the next stage consumes.

Nothing in this file judges condition, cleanliness, or stains — it only
answers "what is in the photo, where, and how is it arranged."
"""
import json
import cv2
import numpy as np

from config.settings import YOLO_WORLD_CLASSES
from models.yolo_loader import load_yolo
from models.vlm_client import VLMClient
from preprocessing.quality_gate import check_quality
from preprocessing.lighting import normalize_lighting
from preprocessing.alignment import align_to_master
from object_detection.detector import detect_objects, detect_open_vocabulary_objects
from object_detection.vlm_inventory import detect_vlm_inventory
from object_detection.merge import merge_detections
from object_detection.polygon import extract_polygon
from object_detection.relationships import compute_relationships


def run_object_discovery(image: np.ndarray, capture_metadata: dict,
                         master_image_path: str | None = None,
                         vlm_client: VLMClient | None = None) -> dict:
    # ---- Stage 0: quality gate ----
    quality = check_quality(image)
    if not quality["is_passed"]:
        return {
            "capture_id": capture_metadata.get("capture_id"),
            "status": "rejected",
            "reason": "blurry",
            "quality_gate": quality,
        }

    # ---- Stage 1: alignment (only if a master image is provided) ----
    spatial_alignment = capture_metadata.get("spatial_alignment", {})
    if master_image_path:
        master_img = cv2.imread(master_image_path)
        image, alignment_result = align_to_master(master_img, image, spatial_alignment)
    else:
        alignment_result = {"method": "none", "success": None, "warnings": []}

    # ---- Stage 2: lighting normalization ----
    ambient_lux = capture_metadata.get("environment", {}).get("ambient_lux")
    image = normalize_lighting(image, ambient_lux)

    # ---- Stage 3a: three discovery branches ----
    detection = detect_objects(image, load_yolo())
    standard_objects = detection["objects"]

    open_vocab_objects = detect_open_vocabulary_objects(image, YOLO_WORLD_CLASSES)

    vlm_client = vlm_client or VLMClient()
    vlm_inventory = detect_vlm_inventory(image, standard_objects + open_vocab_objects, vlm_client)

    # ---- Stage 3b: 3-way merge + dedup ----
    objects = merge_detections(standard_objects, open_vocab_objects, vlm_inventory)

    # ---- Stage 3c: polygons + orientation, per object ----
    for obj in objects:
        poly = extract_polygon(image, obj)
        obj["orientation_deg"] = poly["orientation_deg"]
        obj["mask_polygon"] = {
            "points_pixel": poly["points_pixel"],
            "points_normalized": poly["points_normalized"],
        }

    # ---- Stage 3d: spatial relationships ----
    objects = compute_relationships(objects)

    # ---- DISCOVERY OUTPUT ARTIFACT ----
    discovery = {
        "capture_id": capture_metadata.get("capture_id"),
        "master_reference_id": capture_metadata.get("master_reference_id"),
        "status": "object_discovery_complete",
        "quality_gate": quality,
        "alignment_result": alignment_result,
        "objects": objects,
        "object_counts": detection["object_counts"],
        "sources": {
            "yolo": len(standard_objects),
            "yolo_world": len(open_vocab_objects),
            "vlm_inventory": len(vlm_inventory),
        },
    }
    with open("metadata/object_discovery.json", "w") as f:
        json.dump(discovery, f, indent=2, default=str)

    return discovery