"""
STAGE A - OBJECT DISCOVERY  (hybrid: YOLOv8 + YOLO-World + VLM + Grounding)

Input : ONE pre-processed image (+ capture metadata)        <- no master / reference image
Output: {out_dir}/{capture_id}_object_discovery.json      <- the artifact Stage B consumes
"""
import numpy as np

from config.settings import YOLO_WORLD_CLASSES, OUTPUT_DIR, USE_SEGMENTATION
from core.artifacts import save_artifact
from models.yolo_loader import load_yolo
from models.vlm_client import VLMClient
from models.grounding_service import GroundingService  
from object_detection.detector import detect_objects, detect_open_vocabulary_objects
from object_detection.vlm_inventory import detect_vlm_inventory
from object_detection.merge import merge_detections, count_objects
from object_detection.polygon import extract_polygon
from object_detection.relationships import compute_relationships

from object_detection.segmentation import segment_objects


def run_object_discovery(image: np.ndarray, capture_metadata: dict,
                         vlm_client: VLMClient | None = None,
                         grounding_service: GroundingService | None = None,  # <--- 2. NEW PARAMETER
                         image_path: str | None = None,
                         preprocessing: dict | None = None,
                         out_dir: str = OUTPUT_DIR) -> dict:
    img_h, img_w = image.shape[:2]
    vlm_client = vlm_client or VLMClient()
    
    # Lazy-load grounding service if not passed in
    if grounding_service is None and vlm_client.is_configured:
        try:
            grounding_service = GroundingService()
        except Exception as exc:
            print(f"[DISCOVERY] ⚠️ Could not initialize GroundingService: {exc}")

    # ---- Branch 1: YOLOv8 (COCO) ----
    detection = detect_objects(image, load_yolo())
    standard_objects = detection["objects"]

    # ---- Branch 2: YOLO-World (open vocabulary) ----
    open_vocab_objects = detect_open_vocabulary_objects(image, YOLO_WORLD_CLASSES)

    # ---- Branch 3: VLM inventory + Grounding DINO ----
    # <--- 3. PASS grounding_service HERE
    vlm_inventory = detect_vlm_inventory(
        image, 
        standard_objects + open_vocab_objects, 
        vlm_client,
        grounding_service
    )

    # ---- 3-way merge + dedup -> COMPLETE OBJECT INVENTORY ----
    objects = merge_detections(standard_objects, open_vocab_objects, vlm_inventory)

    '''
    # ---- per-object polygons + orientation ----
    for obj in objects:
        poly = extract_polygon(image, obj)
        obj["orientation_deg"] = poly["orientation_deg"]
        obj["mask_polygon"] = {
            "points_pixel": poly["points_pixel"],
            "points_normalized": poly["points_normalized"],
        }

    '''

    # ---- per-object polygons + orientation ----
    if USE_SEGMENTATION:
        objects = segment_objects(image, objects)          # SAM: true outline per object
    else:
        for obj in objects:
            poly = extract_polygon(image, obj)             # old Otsu contour inside the box
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