"""
Orchestrator: discovery -> condition analysis -> final Stage-3 JSON.

Image loading + quality rejection live here so both stages share one
in-memory image and one VLM client instance.
"""
import cv2
import numpy as np
import json

from models.vlm_client import VLMClient
from pipeline.discovery import run_object_discovery
from pipeline.condition_analysis import run_condition_analysis


def run_stage3_pipeline(image_path: str, capture_metadata: dict,
                        master_image_path: str | None = None) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    vlm_client = VLMClient()  # one instance shared by both stages

    # ---- STAGE A: object discovery (writes metadata/object_discovery.json) ----
    discovery = run_object_discovery(image, capture_metadata, master_image_path, vlm_client)
    if discovery["status"] == "rejected":
        return discovery

    # ---- STAGE B: condition / cleanliness / stains ----
    # condition = run_condition_analysis(image, discovery["objects"], vlm_client)

    # ---- FINAL JSON ----
    return {
        "capture_id": capture_metadata.get("capture_id"),
        "master_reference_id": capture_metadata.get("master_reference_id"),
        "status": "processed",
        "quality_gate": discovery["quality_gate"],
        "alignment_result": discovery["alignment_result"],
        # "objects": condition["objects"],
        "object_counts": discovery["object_counts"],
        "sources": discovery["sources"],
        # "surface_analysis": condition["surface_analysis"],
        # "layout_description": condition["layout_description"],
        "extractor_used": "hybrid_yolo_opencv_vlm" if vlm_client.is_configured else "yolo_opencv_only",
    }