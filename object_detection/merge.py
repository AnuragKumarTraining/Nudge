def calculate_iou(box_a, box_b):
    ax1, ay1 = box_a["x1"], box_a["y1"]
    ax2, ay2 = box_a["x2"], box_a["y2"]
    bx1, by1 = box_b["x1"], box_b["y1"]
    bx2, by2 = box_b["x2"], box_b["y2"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0

from config.settings import CLASS_SYNONYMS

def _canonical(name: str) -> str:
    name = name.lower().strip()
    return CLASS_SYNONYMS.get(name, name)


def merge_detections(
    standard_objects,
    open_vocab_objects,
    vlm_inventory=None,       # NEW: third branch (defaults to None so old calls don't break)
    iou_threshold=0.5,        # lowered: VLM boxes are loose, 0.6 would miss near-dupes
):
    merged = list(standard_objects)
    candidates = list(open_vocab_objects) + list(vlm_inventory or [])

    for cand in candidates:
        duplicate = False

        for existing in merged:
            # synonym-aware: "dining table" (YOLO) == "table" (YOLO-World)
            if _canonical(cand["class"]) != _canonical(existing["class"]):
                continue

            iou = calculate_iou(cand["bbox"], existing["bbox"])
            if iou >= iou_threshold:
                # Keep higher-confidence detection, but NEVER:
                # - overwrite a YOLO bbox with a VLM bbox (low precision)
                # - lose the original object_id
                if (cand["confidence"] > existing["confidence"]
                        and existing.get("source") != "vlm_inventory"
                        and cand.get("source") != "vlm_inventory"):
                    keep_id = existing["object_id"]
                    existing.update(cand)
                    existing["object_id"] = keep_id
                duplicate = True
                break

        if not duplicate:
            merged.append(cand)

    return merged