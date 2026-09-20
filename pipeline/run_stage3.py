"""
Single entry point for processing one capture end to end:
quality gate -> alignment -> lighting -> object detection -> polygon ->
relationships -> VLM enrichment -> surface/stain analysis -> layout
description -> Stage 3 JSON.

This is the piece that was missing from the original repo (main.py called
extract_features() directly, skipping quality gate, alignment, and every
VLM feature entirely).
"""
import cv2
import numpy as np
from config.settings import YOLO_WORLD_CLASSES

from models.yolo_loader import load_yolo, load_yolo_world
from models.vlm_client import VLMClient
from preprocessing.quality_gate import check_quality
from preprocessing.lighting import normalize_lighting
from preprocessing.alignment import align_to_master
from object_detection.detector import detect_objects, detect_open_vocabulary_objects
from object_detection.merge import merge_detections
from object_detection.polygon import extract_polygon
from object_detection.relationships import compute_relationships
from object_detection.merge import merge_detections
from surface_condition.masking import (
    get_objects_on_surface, get_exposed_surface_crop, get_object_crop, is_surface_anchor,
)
from surface_condition.stain_detection import detect_stain_candidates
from semantic.enrich import enrich_objects_with_vlm, confirm_stains_with_vlm, generate_layout_description


def run_stage3_pipeline(image_path: str, capture_metadata: dict, master_image_path: str | None = None) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

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

    # ---- Stage 3: object detection (identity, count, bbox) ----
    yolo_model = load_yolo()

    # Standard YOLO
    detection = detect_objects(
        image,
        yolo_model
    )

    standard_objects = detection["objects"]


    # YOLO-World
    open_vocab_objects = detect_open_vocabulary_objects(
        image,
        YOLO_WORLD_CLASSES
    )


    # Merge both detectors
    objects = merge_detections(
        standard_objects,
        open_vocab_objects
    )

    # ---- polygon + orientation, per object ----
    for obj in objects:
        poly = extract_polygon(image, obj)
        obj["orientation_deg"] = poly["orientation_deg"]
        obj["mask_polygon"] = {
            "points_pixel": poly["points_pixel"],
            "points_normalized": poly["points_normalized"],
        }

    # ---- spatial relationships ----
    objects = compute_relationships(objects)

    # ---- Stage 4: VLM enrichment (material_hint, condition) ----
    vlm_client = VLMClient()
    objects = enrich_objects_with_vlm(image, objects, vlm_client)

    # ---- Stage 4: surface condition / stain candidates ----
    surface_results = []
    for obj in objects:
        if not obj.get("surface_relevant"):
            continue

        if is_surface_anchor(obj):
            objects_on_top = get_objects_on_surface(obj, objects)
            crop, mask, (ox, oy) = get_exposed_surface_crop(image, obj, objects_on_top)
        else:
            crop, (ox, oy) = get_object_crop(image, obj)
            mask = None

        if crop is None or crop.size == 0:
            continue

        stains = detect_stain_candidates(crop, mask=mask)
        for stain in stains:
            local = stain.pop("bbox_local")
            stain["bbox"] = {
                "x1": local["x1"] + ox, "y1": local["y1"] + oy,
                "x2": local["x2"] + ox, "y2": local["y2"] + oy,
            }

        surface_results.append({
            "object_id": obj["object_id"],
            "class": obj["class"],
            "stain_candidates": stains,
        })

    surface_results = confirm_stains_with_vlm(image, surface_results, vlm_client)

    # ---- layout description ----
    layout_description = generate_layout_description(image, objects, vlm_client)

    return {
        "capture_id": capture_metadata.get("capture_id"),
        "master_reference_id": capture_metadata.get("master_reference_id"),
        "status": "processed",
        "quality_gate": quality,
        "alignment_result": alignment_result,
        "objects": objects,
        "object_counts": detection["object_counts"],
        # "surface_analysis": surface_results,
        # "layout_description": layout_description,
        # "extractor_used": "hybrid_yolo_opencv_vlm" if vlm_client.is_configured else "yolo_opencv_only",
        # "detection_confidence": detection["detection_confidence"],
    }
