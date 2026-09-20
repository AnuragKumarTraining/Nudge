import numpy as np

from config.settings import SURFACE_ANCHOR_CLASSES


def get_objects_on_surface(anchor_obj: dict, all_objects: list[dict]) -> list[dict]:
    """Objects whose centroid falls inside the anchor's (e.g. table's) bbox.
    Simplification: centroid-containment, not polygon-containment — fine for
    v1, revisit once real photos show edge cases (an object mostly off the
    table edge but with its centroid just inside it, etc.)."""
    tx1, ty1, tx2, ty2 = anchor_obj["bbox"]["x1"], anchor_obj["bbox"]["y1"], anchor_obj["bbox"]["x2"], anchor_obj["bbox"]["y2"]
    on_surface = []
    for obj in all_objects:
        if obj["object_id"] == anchor_obj["object_id"]:
            continue
        cx, cy = obj["centroid"]["x"], obj["centroid"]["y"]
        if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
            on_surface.append(obj)
    return on_surface


def get_exposed_surface_crop(image: np.ndarray, anchor_obj: dict, objects_on_surface: list[dict]):
    """Crop the anchor's bbox, mask out (zero) the regions of objects sitting on it."""
    x1, y1, x2, y2 = anchor_obj["bbox"]["x1"], anchor_obj["bbox"]["y1"], anchor_obj["bbox"]["x2"], anchor_obj["bbox"]["y2"]
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return crop, None, (x1, y1)

    mask = np.ones(crop.shape[:2], dtype=np.uint8) * 255
    for obj in objects_on_surface:
        ox1 = max(0, obj["bbox"]["x1"] - x1)
        oy1 = max(0, obj["bbox"]["y1"] - y1)
        ox2 = min(crop.shape[1], obj["bbox"]["x2"] - x1)
        oy2 = min(crop.shape[0], obj["bbox"]["y2"] - y1)
        mask[oy1:oy2, ox1:ox2] = 0

    return crop, mask, (x1, y1)


def get_object_crop(image: np.ndarray, obj: dict):
    x1, y1, x2, y2 = obj["bbox"]["x1"], obj["bbox"]["y1"], obj["bbox"]["x2"], obj["bbox"]["y2"]
    return image[y1:y2, x1:x2], (x1, y1)


def is_surface_anchor(obj: dict) -> bool:
    return obj["class"] in SURFACE_ANCHOR_CLASSES
