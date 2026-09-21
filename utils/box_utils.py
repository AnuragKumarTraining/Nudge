# utils/box_utils.py
import numpy as np

def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two normalized boxes [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0

    return interArea / unionArea


def validate_and_normalize_bbox(
    bbox: list[float], 
    img_w: int, 
    img_h: int, 
    min_pixel_size: int = 10,
    min_area_ratio: float = 0.0001
) -> list[float] | None:
    """
    CHECK 7, 8, 9: Validates, clamps, and normalizes coordinates into [x1, y1, x2, y2] within [0, 1].
    """
    try:
        v = [float(x) for x in bbox]
    except (TypeError, ValueError):
        return None

    if len(v) != 4:
        return None

    m = max(v)
    if m <= 1.0:
        x1, y1, x2, y2 = v
    elif m <= 1000.0:
        x1, y1, x2, y2 = (c / 1000.0 for c in v)
    else:
        x1, y1, x2, y2 = v[0] / img_w, v[1] / img_h, v[2] / img_w, v[3] / img_h

    # Clamp coordinates to [0.0, 1.0]
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    x2 = min(1.0, max(0.0, x2))
    y2 = min(1.0, max(0.0, y2))

    # Check non-zero dimensions
    if x2 <= x1 or y2 <= y1:
        return None

    # Check minimum pixel dimensions
    pw = (x2 - x1) * img_w
    ph = (y2 - y1) * img_h
    if pw < min_pixel_size or ph < min_pixel_size:
        return None

    # Check minimum area ratio
    area = (x2 - x1) * (y2 - y1)
    if area < min_area_ratio:
        return None

    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def apply_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """
    CHECK 11: Non-Maximum Suppression to filter duplicate overlapping boxes per class.
    """
    if not detections:
        return []

    # Group by class
    by_class = {}
    for det in detections:
        cls = det["class"]
        by_class.setdefault(cls, []).append(det)

    keep_detections = []
    for cls, items in by_class.items():
        # Sort by confidence descending
        items.sort(key=lambda x: x["confidence"], reverse=True)
        
        while items:
            best = items.pop(0)
            keep_detections.append(best)
            
            items = [
                item for item in items 
                if compute_iou(best["bbox_normalized"], item["bbox_normalized"]) < iou_threshold
            ]

    return keep_detections