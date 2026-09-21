
"""
3-way merge + dedup of the discovery branches (YOLOv8, YOLO-World, VLM + Grounding),
plus the merged object counts.
"""
from config.settings import CLASS_SYNONYMS

# <--- 1. UPDATED PRECISION HIERARCHY
# YOLO (fine-tuned) > Grounding DINO / YOLO-World > Raw VLM box fallback
_PRECISION = {
    "yolo": 4,
    "yolo_world": 3,
    "vlm_grounded": 3,    # Grounding DINO accuracy is on par with YOLO-World
    "vlm_inventory": 1,   # Raw VLM generated boxes (low precision)
}

def calculate_iou(box_a, box_b):
    ix1, iy1 = max(box_a["x1"], box_b["x1"]), max(box_a["y1"], box_b["y1"])
    ix2, iy2 = min(box_a["x2"], box_b["x2"]), min(box_a["y2"], box_b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter
    return inter / union if union else 0.0

def _canonical(name: str) -> str:
    name = name.lower().strip()
    return CLASS_SYNONYMS.get(name, name)

def _absorb(existing: dict, cand: dict) -> None:
    """Fold a duplicate detection into the object we already have."""
    src = cand.get("source", "unknown")
    if src not in existing["sources"]:
        existing["sources"].append(src)
    # take the candidate's geometry only if its source is MORE precise than ours
    if _PRECISION.get(src, 0) > _PRECISION.get(existing.get("source"), 0):
        for key in ("bbox", "bbox_normalized", "centroid"):
            existing[key] = cand[key]
        existing["source"] = src
    # agreement between branches is evidence -> keep the best confidence seen
    existing["confidence"] = max(existing["confidence"], cand["confidence"])


def merge_detections(standard_objects, open_vocab_objects, vlm_inventory=None, iou_threshold=0.5):
    merged = [dict(o) for o in standard_objects]
    for o in merged:
        o["sources"] = [o.get("source", "yolo")]
    candidates = list(open_vocab_objects) + list(vlm_inventory or [])
    for cand in candidates:
        cand = dict(cand)
        cand_class = _canonical(cand["class"])
        best, best_iou = None, 0.0

        for existing in merged:
            if cand_class != _canonical(existing["class"]):
                continue

            iou = calculate_iou(cand["bbox"], existing["bbox"])
            if iou > best_iou:
                best, best_iou = existing, iou
        if best is not None and best_iou >= iou_threshold:
            _absorb(best, cand)
        else:
            cand["sources"] = [cand.get("source", "unknown")]
            merged.append(cand)

    return merged

def count_objects(objects: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in objects:
        name = _canonical(o["class"])
        # <--- 2. UPDATED SOURCE CHECK FOR vlm_grounded
        src = o.get("source")
        is_vlm_source = src in ("vlm_inventory", "vlm_grounded")
        n = max(1, int(o.get("reported_count", 1))) if is_vlm_source else 1
        counts[name] = counts.get(name, 0) + n
    return dict(sorted(counts.items()))