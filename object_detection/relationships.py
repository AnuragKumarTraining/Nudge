"""
Spatial relationships between objects in the merged inventory.
For every non-structural object:
  * "on"   - its centroid is inside a table's box  -> belongs to that table
  * "near" - otherwise the nearest table by centroid distance
Tables get the reverse list (`objects_on_surface`). Floor/wall/ceiling/window/door
are skipped: "belongs to the nearest table" is meaningless for them.
This is also the building block for place-setting completeness ("which spoon/plate
belongs to which chair") later.
"""
import math

from config.settings import SURFACE_ANCHOR_CLASSES, STRUCTURAL_CLASSES


def _inside(pt: dict, box: dict) -> bool:
    return box["x1"] <= pt["x"] <= box["x2"] and box["y1"] <= pt["y"] <= box["y2"]


def _area(box: dict) -> int:
    return max(0, box["x2"] - box["x1"]) * max(0, box["y2"] - box["y1"])


def compute_relationships(objects: list[dict]) -> list[dict]:
    anchors = [o for o in objects if o["class"] in SURFACE_ANCHOR_CLASSES]

    for a in anchors:
        a.setdefault("relationships", {})["objects_on_surface"] = []

    for obj in objects:
        rel = obj.setdefault("relationships", {})
        if obj["class"] in SURFACE_ANCHOR_CLASSES or obj["class"] in STRUCTURAL_CLASSES:
            continue

        inside = [a for a in anchors if _inside(obj["centroid"], a["bbox"])]
        if inside:
            closest, relation = min(inside, key=lambda a: _area(a["bbox"])), "on"
        else:
            closest, relation = None, "near"
            best = float("inf")
            for a in anchors:
                d = math.hypot(a["centroid"]["x"] - obj["centroid"]["x"],
                               a["centroid"]["y"] - obj["centroid"]["y"])
                if d < best:
                    best, closest = d, a

        if closest is None:
            continue

        vertical = "top" if obj["centroid"]["y"] < closest["centroid"]["y"] else "bottom"
        horizontal = "left" if obj["centroid"]["x"] < closest["centroid"]["x"] else "right"
        
        rel["relation"] = relation
        rel["belongs_to"] = closest["object_id"]
        rel["relative_position"] = f"{vertical}-{horizontal} of {closest['object_id']}"
        
        if relation == "on":
            closest["relationships"]["objects_on_surface"].append(obj["object_id"])

    return objects