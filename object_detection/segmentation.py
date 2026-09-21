"""
Pixel-accurate object masks.

Why: a bounding box is always a rectangle, so it includes floor, wall and neighbouring objects.
Here every detected box is used as a PROMPT for Segment Anything (SAM), which returns only the
pixels that belong to that object. It works for all three detection branches because they all
end in a box.

    detect (boxes)  ->  segment_objects()  ->  obj["mask_polygon"] (true outline)

The output keys are the same ones extract_polygon() produced (`orientation_deg`,
`mask_polygon.points_pixel`, `mask_polygon.points_normalized`), plus:
    mask_source   "sam" | "otsu_fallback"
    mask_area_px  number of pixels inside the outline
so nothing downstream breaks.
"""
import cv2
import numpy as np

from config.settings import POLYGON_EPSILON_RATIO
from object_detection.polygon import extract_polygon


def _polygon_fields(points_xy: np.ndarray, img_w: int, img_h: int):
    """SAM outline (N,2 float pixels) -> simplified polygon, orientation and area."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        return None
    area = float(cv2.contourArea(pts))
    if area < 4:
        return None
    eps = POLYGON_EPSILON_RATIO * cv2.arcLength(pts, True)
    approx = cv2.approxPolyDP(pts, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = pts.reshape(-1, 2)
    points_pixel = [[int(round(x)), int(round(y))] for x, y in approx]
    points_norm = [[round(x / img_w, 4), round(y / img_h, 4)] for x, y in points_pixel]
    angle = round(float(cv2.minAreaRect(pts)[2]), 2)
    return {
        "orientation_deg": angle,
        "points_pixel": points_pixel,
        "points_normalized": points_norm,
        "area_px": int(area),
    }


def _apply(obj: dict, fields: dict, source: str) -> None:
    obj["orientation_deg"] = fields["orientation_deg"]
    obj["mask_polygon"] = {
        "points_pixel": fields["points_pixel"],
        "points_normalized": fields["points_normalized"],
    }
    obj["mask_source"] = source
    if "area_px" in fields:
        obj["mask_area_px"] = fields["area_px"]


def _fallback_all(image: np.ndarray, objects: list[dict]) -> list[dict]:
    for obj in objects:
        poly = extract_polygon(image, obj)
        _apply(obj, poly, "otsu_fallback")
    return objects


def segment_objects(image: np.ndarray, objects: list[dict], sam_model=None) -> list[dict]:
    """
    Adds a true-outline polygon to every object using box-prompted SAM.
    Falls back to the old Otsu contour (extract_polygon) if SAM is unavailable or misbehaves,
    so discovery never crashes because of segmentation.
    """
    if not objects:
        return objects
    img_h, img_w = image.shape[:2]

    try:
        if sam_model is None:
            from models.yolo_loader import load_sam
            sam_model = load_sam()

        boxes = []
        for o in objects:
            b = o["bbox"]
            boxes.append([max(0, b["x1"]), max(0, b["y1"]), min(img_w, b["x2"]), min(img_h, b["y2"])])

        results = sam_model(image, bboxes=boxes, verbose=False)
        masks = results[0].masks
        polys = masks.xy if masks is not None else []
    except Exception as exc:
        print(f"[SEGMENTATION] SAM unavailable, using Otsu fallback: {type(exc).__name__}: {exc}")
        return _fallback_all(image, objects)

    if len(polys) != len(objects):
        # order can't be trusted if SAM dropped a box
        print(f"[SEGMENTATION] SAM returned {len(polys)} masks for {len(objects)} boxes, using Otsu fallback")
        return _fallback_all(image, objects)

    for obj, xy in zip(objects, polys):
        fields = _polygon_fields(xy, img_w, img_h)
        if fields is None:
            _apply(obj, extract_polygon(image, obj), "otsu_fallback")
        else:
            _apply(obj, fields, "sam")
    return objects


def polygon_to_mask(obj: dict, shape: tuple[int, int]) -> np.ndarray | None:
    """Rasterise obj['mask_polygon'] to a uint8 mask (255 = object) of the given (h, w)."""
    pts = (obj.get("mask_polygon") or {}).get("points_pixel") or []
    if len(pts) < 3:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask
