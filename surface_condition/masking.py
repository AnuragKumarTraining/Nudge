import numpy as np

from config.settings import EXPOSED_SURFACE_CLASSES, STRUCTURAL_CLASSES


def _clamp_box(image: np.ndarray, box: dict) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    return (max(0, box["x1"]), max(0, box["y1"]), min(w, box["x2"]), min(h, box["y2"]))


def get_objects_on_surface(anchor_obj: dict, all_objects: list[dict]) -> list[dict]:
    """Objects whose centroid falls inside the surface's bbox (table, floor, wall...).
    Structural regions (floor/wall/window/...) are never "on top of" anything."""
    b = anchor_obj["bbox"]
    on_surface = []
    for obj in all_objects:
        if obj["object_id"] == anchor_obj["object_id"] or obj["class"] in STRUCTURAL_CLASSES:
            continue
        cx, cy = obj["centroid"]["x"], obj["centroid"]["y"]
        if b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"]:
            on_surface.append(obj)
    return on_surface


def get_exposed_surface_crop(image: np.ndarray, anchor_obj: dict, objects_on_surface: list[dict]):
    """Crop the surface's bbox; mask value 0 = covered by another object, 255 = exposed."""
    x1, y1, x2, y2 = _clamp_box(image, anchor_obj["bbox"])
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return crop, None, (x1, y1)

    mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
    for obj in objects_on_surface:
        ox1 = max(0, obj["bbox"]["x1"] - x1)
        oy1 = max(0, obj["bbox"]["y1"] - y1)
        ox2 = min(crop.shape[1], obj["bbox"]["x2"] - x1)
        oy2 = min(crop.shape[0], obj["bbox"]["y2"] - y1)
        mask[oy1:oy2, ox1:ox2] = 0

    return crop, mask, (x1, y1)


def get_object_crop(image: np.ndarray, obj: dict):
    x1, y1, x2, y2 = _clamp_box(image, obj["bbox"])
    return image[y1:y2, x1:x2], (x1, y1)


def is_surface_anchor(obj: dict) -> bool:
    """True for surfaces that other objects sit on/in front of (table, floor, wall, ceiling)."""
    return obj["class"] in EXPOSED_SURFACE_CLASSES