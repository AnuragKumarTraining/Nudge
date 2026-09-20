"""
Everything in this file needs judgment, not measurement — this is the "VLM
owns it" column from the feature/source table. Every function degrades
gracefully when vlm_client.is_configured is False, so the pipeline still
produces complete (if less precise) output without an API key configured.
"""
import numpy as np

from models.vlm_client import VLMClient


def crop_object(image: np.ndarray, obj: dict) -> np.ndarray:
    x1, y1, x2, y2 = obj["bbox"]["x1"], obj["bbox"]["y1"], obj["bbox"]["x2"], obj["bbox"]["y2"]
    return image[y1:y2, x1:x2]


def enrich_objects_with_vlm(image: np.ndarray, objects: list[dict], vlm_client: VLMClient) -> list[dict]:
    """Fills material_hint (only where still pending — the class-based
    heuristic from detector.py may already be good enough) and condition."""
    if not vlm_client.is_configured:
        return objects

    for obj in objects:
        crop = crop_object(image, obj)
        if crop.size == 0:
            continue
        if obj.get("material_hint") == "pending_vlm":
            obj["material_hint"] = vlm_client.get_material_hint(crop, obj["class"])
        obj["condition"] = vlm_client.get_object_condition(crop, obj["class"])

    return objects


def confirm_stains_with_vlm(image: np.ndarray, surface_results: list[dict], vlm_client: VLMClient) -> list[dict]:
    """
    This is the highest-leverage hybrid call in the whole pipeline: OpenCV's
    adaptive threshold over-flags patterned surfaces badly (a print on a
    tablecloth looks like dozens of "stains"). Sending just the small
    candidate crop — not the whole image — to the VLM for a stain/pattern/
    shadow verdict fixes most of that at low cost.
    """
    for surface in surface_results:
        for stain in surface["stain_candidates"]:
            bbox = stain["bbox"]
            crop = image[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]

            if not vlm_client.is_configured or crop.size == 0:
                stain["vlm_verdict"] = None
                stain["adjusted_confidence"] = stain["raw_confidence"]
                continue

            verdict = vlm_client.confirm_stain(crop)
            stain["vlm_verdict"] = verdict
            if verdict == "stain":
                stain["adjusted_confidence"] = round(min(1.0, stain["raw_confidence"] * 1.3), 2)
            elif verdict in ("pattern", "shadow"):
                stain["adjusted_confidence"] = round(stain["raw_confidence"] * 0.15, 2)
            else:  # unclear
                stain["adjusted_confidence"] = round(stain["raw_confidence"] * 0.6, 2)

    return surface_results


def generate_layout_description(image: np.ndarray, objects: list[dict], vlm_client: VLMClient) -> str:
    if not vlm_client.is_configured:
        return "pending_vlm"
    summary = ", ".join(
        f"{o['class']} at ({o['centroid']['x']},{o['centroid']['y']})" for o in objects
    )
    return vlm_client.describe_layout(image, summary)
