"""
STAGE B - CONDITION / CLEANLINESS / STAIN ANALYSIS

Input : the pre-processed image + the merged inventory from Stage A
        (either in memory, or loaded from {capture_id}_object_discovery.json)
Output: {out_dir}/{capture_id}_condition_analysis.json

    inventory ─► CONDITION VLM ─► condition + cleanliness (+ material, issues)
              ─► SURFACE / STAIN: OpenCV candidates ─► VLM verdict (stain / pattern / shadow)
              ─► layout description (VLM)
"""
import numpy as np

from config.settings import OUTPUT_DIR, STAIN_SCAN_CLASSES, STAIN_MAX_CANDIDATES_TOTAL
from core.artifacts import save_artifact
from models.vlm_client import VLMClient
from semantic.enrich import (
    enrich_objects_with_vlm,
    confirm_stains_with_vlm,
    generate_layout_description,
)
from surface_condition.masking import (
    get_exposed_surface_crop,
    get_object_crop,
    get_objects_on_surface,
    is_surface_anchor,
)
from surface_condition.stain_detection import detect_stain_candidates


def _scan_surfaces(image: np.ndarray, objects: list[dict]) -> list[dict]:
    """OpenCV half: candidate marks per surface/object, in full-image coordinates."""
    surface_results = []
    for obj in objects:
        if not obj.get("surface_relevant") or obj["class"] not in STAIN_SCAN_CLASSES:
            continue

        if is_surface_anchor(obj):
            crop, mask, (ox, oy) = get_exposed_surface_crop(
                image, obj, get_objects_on_surface(obj, objects)
            )
        else:
            crop, (ox, oy) = get_object_crop(image, obj)
            mask = None

        if crop is None or crop.size == 0:
            continue

        stains = detect_stain_candidates(crop, mask=mask)
        for stain in stains:
            b = stain.pop("bbox_local")
            stain["bbox"] = {
                "x1": b["x1"] + ox,
                "y1": b["y1"] + oy,
                "x2": b["x2"] + ox,
                "y2": b["y2"] + oy,
            }

        surface_results.append({
            "object_id": obj["object_id"],
            "class": obj["class"],
            "stain_candidates": stains,
        })

    # global ceiling on VLM crops per image: keep the most promising candidates overall
    flat = sorted(
        ((s, r) for r in surface_results for s in r["stain_candidates"]),
        key=lambda t: (t[0]["raw_confidence"], t[0]["area_px"]),
        reverse=True,
    )
    keep = {id(s) for s, _ in flat[:STAIN_MAX_CANDIDATES_TOTAL]}
    for r in surface_results:
        r["stain_candidates"] = [s for s in r["stain_candidates"] if id(s) in keep]

    return surface_results


def run_condition_analysis(
    image: np.ndarray,
    objects: list[dict],
    capture_metadata: dict,
    vlm_client: VLMClient | None = None,
    out_dir: str = OUTPUT_DIR,
) -> dict:
    vlm_client = vlm_client or VLMClient()

    # ---- CONDITION VLM: condition + cleanliness (+ material) per object ----
    objects = enrich_objects_with_vlm(image, objects, vlm_client)

    # ---- SURFACE / STAIN: OpenCV candidates -> VLM verdict ----
    surface_results = _scan_surfaces(image, objects)
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
    condition["artifact_path"] = save_artifact(
        out_dir, condition["capture_id"], "condition_analysis", condition
    )

    return condition