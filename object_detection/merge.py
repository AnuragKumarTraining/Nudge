"""
3-way merge + dedup of the discovery branches (YOLOv8, YOLO-World, VLM inventory),
plus the merged object counts.
Rules (in priority order):
  1. Two detections are the same object if their CANONICAL class names match
     (CLASS_SYNONYMS, so "potted plant" == "plant") and their boxes overlap (IoU).
  2. Geometry comes from the most precise source: YOLO > YOLO-World > VLM. A VLM box
     never overwrites a YOLO/YOLO-World box.
  3. Nothing is silently lost: every merged object keeps `sources` (which branches
     saw it), so "seen by 2-3 branches" can be used later as a confidence signal.
"""
from config.settings import CLASS_SYNONYMS
_PRECISION = {"yolo": 1, "yolo_world": 2, "vlm_inventory": 3}

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
            # synonym-aware: "dining table" (YOLO) == "table" (YOLO-World)
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
    """
    Counts over the MERGED inventory (the old code reported YOLO-only counts, so
    everything YOLO-World / the VLM found was missing from `object_counts`).
    A VLM entry may stand for several instances ("reported_count") when the VLM
    could only give one loose box for a group, e.g. "6 spoons".
    """
    counts: dict[str, int] = {}
    for o in objects:
        name = _canonical(o["class"])            # "dining table" and "table" count together
        n = max(1, int(o.get("reported_count", 1))) if o.get("source") == "vlm_inventory" else 1
        counts[name] = counts.get(name, 0) + n
    return dict(sorted(counts.items()))