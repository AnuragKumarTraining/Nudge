"""
STAGE B — CONDITION / CLEANLINESS / STAIN ANALYSIS.

Input : aligned+normalized image + merged inventory from Stage A
Output: metadata/condition_analysis.json  +  result dict for the final JSON
"""
import json

from models.vlm_client import VLMClient
from surface_condition.masking import (
    get_objects_on_surface, get_exposed_surface_crop, get_object_crop, is_surface_anchor,
)
from surface_condition.stain_detection import detect_stain_candidates
from semantic.enrich import (
    enrich_objects_with_vlm, confirm_stains_with_vlm, generate_layout_description,
)


def run_condition_analysis(image, objects: list[dict], capture_metadata: dict,
                           vlm_client: VLMClient | None = None) -> dict:
    vlm_client = vlm_client or VLMClient()

    # ---- CONDITION VLM: material_hint, condition (+ cleanliness) ----
    objects = enrich_objects_with_vlm(image, objects, vlm_client)

    # ---- SURFACE / STAIN ----
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
    layout_description = generate_layout_description(image, objects, vlm_client)

    # ---- STAGE B OUTPUT ARTIFACT ----
    condition = {
        "capture_id": capture_metadata.get("capture_id"),
        "status": "condition_analysis_complete",
        "objects": objects,
        "surface_analysis": surface_results,
        "layout_description": layout_description,
    }
    with open("metadata/condition_analysis.json", "w") as f:
        json.dump(condition, f, indent=2, default=str)

    return condition