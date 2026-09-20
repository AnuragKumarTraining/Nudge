"""
Groups objects relative to the nearest table/anchor — refactored from PASS 2
of the original feature_extraction.py. This is also exactly the building
block that layout analysis (place-setting completeness) needs later:
"which spoon/plate belongs to which chair" is just this same nearest-anchor
logic run against chairs instead of tables.
"""
import math

from config.settings import SURFACE_ANCHOR_CLASSES


def compute_relationships(objects: list[dict]) -> list[dict]:
    anchors = [o for o in objects if o["class"] in SURFACE_ANCHOR_CLASSES]

    for obj in objects:
        obj.setdefault("relationships", {})
        if obj["class"] in SURFACE_ANCHOR_CLASSES:
            continue

        closest, min_dist = None, float("inf")
        for anchor in anchors:
            dist = math.hypot(
                anchor["centroid"]["x"] - obj["centroid"]["x"],
                anchor["centroid"]["y"] - obj["centroid"]["y"],
            )
            if dist < min_dist:
                min_dist, closest = dist, anchor

        if closest:
            vertical = "top" if obj["centroid"]["y"] < closest["centroid"]["y"] else "bottom"
            horizontal = "left" if obj["centroid"]["x"] < closest["centroid"]["x"] else "right"
            obj["relationships"]["relative_position"] = f"{vertical}-{horizontal} of {closest['object_id']}"
            obj["relationships"]["belongs_to"] = closest["object_id"]

    return objects
