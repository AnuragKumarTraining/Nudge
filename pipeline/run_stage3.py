"""
Orchestrator:  image + input JSON  ->  preprocess -> Stage A (discovery) -> Stage B (condition) -> final JSON

ONE image in. No master / reference image is loaded, aligned to, or compared against.

  run_stage3_pipeline()          - the whole thing
  run_condition_from_artifact()  - resume from a saved object_discovery.json (Stage B only)
"""
import cv2
import numpy as np
import json

from config.settings import OUTPUT_DIR
from core.artifacts import load_artifact, save_artifact
from models.vlm_client import VLMClient
from pipeline.discovery import run_object_discovery
from pipeline.condition_analysis import run_condition_analysis



def _read_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    return image


def _final_json(capture_metadata: dict, discovery: dict, condition: dict, vlm_client: VLMClient) -> dict:
    return {
        "capture_id": capture_metadata.get("capture_id"),
        "master_reference_id": capture_metadata.get("master_reference_id"),
        "status": "processed",
        "preprocessing": discovery["preprocessing"],
        "extractor_used": ("hybrid_yolo_yoloworld_vlm_opencv" if vlm_client.is_configured
                           else "yolo_yoloworld_opencv_only"),
        "sources": discovery["sources"],
        "total_objects": discovery["total_objects"],
        "object_counts": discovery["object_counts"],
        "objects": condition["objects"],
        "surface_analysis": condition["surface_analysis"],
        "layout_description": condition["layout_description"],
        "artifacts": {
            "object_discovery": discovery["artifact_path"],
            "condition_analysis": condition["artifact_path"],
        },
    }


def run_stage3_pipeline(image_path: str, capture_metadata: dict,
                        out_dir: str = OUTPUT_DIR) -> dict:
    vlm_client = VLMClient()       # one instance shared by both stages
    image = _read_image(image_path)


    # ---- STAGE A: object discovery  (writes <capture_id>_object_discovery.json) ----
    discovery = run_object_discovery(image, capture_metadata, vlm_client,
                                     image_path=image_path, out_dir=out_dir)

    # ---- STAGE B: condition / cleanliness / stains  (writes <capture_id>_condition_analysis.json) ----
    # condition = run_condition_analysis(image, discovery["objects"], capture_metadata, vlm_client, out_dir)
    condition = {}

    final = _final_json(capture_metadata, discovery, condition, vlm_client)
    save_artifact(out_dir, final["capture_id"], "final", final)
    return final


def run_condition_from_artifact(discovery_path: str, capture_metadata: dict | None = None,
                                image_path: str | None = None, out_dir: str = OUTPUT_DIR,
                                vlm_client: VLMClient | None = None) -> dict:
    """Stage B only, driven by a saved discovery artifact (no YOLO / YOLO-World / VLM-inventory re-run)."""
    vlm_client = vlm_client or VLMClient()
    discovery = load_artifact(discovery_path)
    capture_metadata = capture_metadata or {"capture_id": discovery["capture_id"]}
    image_path = image_path or discovery["image"]["path"]
    image, prep = prepare_image(_read_image(image_path), capture_metadata)   # same deterministic preprocessing
    if image is None:
        return prep

    condition = run_condition_analysis(image, discovery["objects"], capture_metadata, vlm_client, out_dir)
    discovery["artifact_path"] = discovery_path
    final = _final_json(capture_metadata, discovery, condition, vlm_client)
    save_artifact(out_dir, final["capture_id"], "final", final)
    return final