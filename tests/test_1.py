"""
Draw detected objects on the photo using their MASK POLYGONS only - no bounding boxes.

Each object is marked by the pixels it actually covers: a semi-transparent fill, a thin outline
and a small label. A rectangle is never drawn unless you switch DRAW_BBOX on for debugging.

Run from the project root (paths are relative to it, nothing is hardcoded):

    python tests/test.py
    python tests/test.py --json outputs/cap_690205_object_discovery.json \
                         --image images/daily_2.jpg --out tests/marked_masks.jpg
    python tests/test.py --cutouts tests/cutouts      # also save one transparent PNG per object

Masks come from object_detection/segmentation.py (SAM). If discovery ran with USE_SEGMENTATION=0,
the polygons are the old Otsu contours and will look rough - that is expected.
"""
import argparse
import hashlib
import json
import os

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ============================================================
# WHAT TO DRAW
# ============================================================
DRAW_MASK_FILL = True        # semi-transparent fill of the object's pixels
DRAW_OUTLINE = True          # thin contour around the mask
DRAW_LABEL = True
DRAW_CONFIDENCE = True
DRAW_CENTROID = False        # centre of the MASK (not of the box)
DRAW_BBOX = False            # debugging only: this is the rectangle you want to get rid of
DRAW_WITHOUT_MASK = False    # objects with no polygon: False = skip them, True = fall back to a thin box

SKIP_CLASSES = set()         # e.g. {"floor", "wall", "ceiling"} to hide the huge structural regions

MASK_ALPHA = 0.40
OUTLINE_THICKNESS = 2
FONT_SCALE = 0.50
FONT_THICKNESS = 1

COLORS = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 50), (255, 80, 220),
    (80, 220, 255), (180, 80, 255), (80, 255, 200), (255, 150, 80), (150, 255, 80),
]


def class_color(name: str):
    """Same class -> same colour, so all chairs look alike."""
    idx = int(hashlib.md5(name.encode()).hexdigest(), 16) % len(COLORS)
    return COLORS[idx]


def get_polygon(obj: dict, width: int, height: int):
    """
    Mask polygon in pixels of the ACTUAL image. Prefers the normalised points so the drawing still
    lines up if the image was resized after discovery ran.
    """
    poly = obj.get("mask_polygon") or {}
    norm = poly.get("points_normalized") or []
    pix = poly.get("points_pixel") or []
    if len(norm) >= 3:
        pts = np.array(norm, dtype=np.float64) * [width, height]
    elif len(pix) >= 3:
        pts = np.array(pix, dtype=np.float64)
    else:
        return None
    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts.reshape((-1, 1, 2))


def mask_centroid(poly: np.ndarray):
    m = cv2.moments(poly)
    if m["m00"] > 0:
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    p = poly.reshape(-1, 2)
    return int(p[:, 0].mean()), int(p[:, 1].mean())


def draw_label(img, text, anchor_xy, color, width):
    """Label sits just above the topmost point of the mask (below it if there is no room)."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    x, y = anchor_xy
    y = y - 6
    if y - th - baseline < 48:            # keep clear of the title bar
        y = anchor_xy[1] + th + 10
    x = max(0, min(x, width - tw - 6))
    cv2.rectangle(img, (x, y - th - baseline - 3), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)


def save_cutout(image, poly, path):
    """Transparent PNG containing ONLY the object's pixels (background is fully transparent)."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    x, y, bw, bh = cv2.boundingRect(poly)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    cv2.imwrite(path, bgra[y:y + bh, x:x + bw])


def resolve_image_path(cli_image, data):
    if cli_image:
        return cli_image
    from_json = (data.get("image", {}).get("path") or "").replace("\\", os.sep)
    for cand in (from_json, os.path.join(ROOT, from_json), os.path.join(ROOT, "images", "daily_2.jpg")):
        if cand and os.path.exists(cand):
            return cand
    return os.path.join(ROOT, "images", "daily_2.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "outputs", "cap_690205_object_discovery.json"))
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "marked_masks.jpg"))
    ap.add_argument("--cutouts", default=None, help="folder for one transparent PNG per object")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = resolve_image_path(args.image, data)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    print("=" * 70)
    print(f"Image : {image_path}  ({width} x {height})")
    jw, jh = data.get("image", {}).get("width"), data.get("image", {}).get("height")
    if (jw, jh) != (width, height):
        print(f"WARNING: JSON says {jw} x {jh}. Normalised points are used, so masks are rescaled to fit.")
    objects = data.get("objects", [])
    print(f"Objects in JSON: {len(objects)}")
    print("=" * 70)

    # ---- collect drawable objects; largest first so small objects end up on top of big regions ----
    items, without_mask = [], []
    for index, obj in enumerate(objects):
        if obj.get("class", "unknown") in SKIP_CLASSES:
            continue
        poly = get_polygon(obj, width, height)
        if poly is None:
            without_mask.append((index, obj))
            continue
        items.append((cv2.contourArea(poly), index, obj, poly))
    items.sort(key=lambda t: t[0], reverse=True)

    # ---- fill: one overlay, blended once (a plain alpha overlay, no brightening of the photo) ----
    output = image.copy()
    if DRAW_MASK_FILL:
        overlay = image.copy()
        for _, index, obj, poly in items:
            cv2.fillPoly(overlay, [poly], class_color(obj.get("class", "unknown")))
        output = cv2.addWeighted(overlay, MASK_ALPHA, image, 1.0 - MASK_ALPHA, 0)

    # ---- outlines, labels, optional extras ----
    if args.cutouts:
        os.makedirs(args.cutouts, exist_ok=True)

    for _, index, obj, poly in items:
        name = obj.get("class", "unknown")
        color = class_color(name)

        if DRAW_OUTLINE:
            cv2.polylines(output, [poly], True, color, OUTLINE_THICKNESS, cv2.LINE_AA)

        if DRAW_BBOX and obj.get("bbox"):
            b = obj["bbox"]
            cv2.rectangle(output, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 1)

        if DRAW_CENTROID:
            cv2.drawMarker(output, mask_centroid(poly), color, cv2.MARKER_CROSS, 12, 2)

        if DRAW_LABEL:
            label = f"{index + 1}. {name}"
            conf = obj.get("confidence")
            if DRAW_CONFIDENCE and conf is not None:
                label += f" {conf:.2f}"
            pts = poly.reshape(-1, 2)
            top = pts[pts[:, 1].argmin()]
            draw_label(output, label, (int(top[0]), int(top[1])), color, width)

        if args.cutouts:
            save_cutout(image, poly, os.path.join(args.cutouts, f"{index + 1:02d}_{name.replace(' ', '_')}.png"))

        print(f"{index + 1:02d} {name:22s} mask_points={len(poly):3d} "
              f"area={int(cv2.contourArea(poly)):7d}px source={obj.get('mask_source', 'unknown')}")

    for index, obj in without_mask:
        print(f"{index + 1:02d} {obj.get('class', 'unknown'):22s} NO MASK POLYGON")
        if DRAW_WITHOUT_MASK and obj.get("bbox"):
            b = obj["bbox"]
            cv2.rectangle(output, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (150, 150, 150), 1)

    # ---- title bar ----
    title = f"Objects marked by mask: {len(items)}"
    if without_mask:
        title += f"   (no mask: {len(without_mask)})"
    cv2.rectangle(output, (0, 0), (width, 45), (30, 30, 30), -1)
    cv2.putText(output, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not cv2.imwrite(args.out, output):
        raise RuntimeError(f"Could not save: {args.out}")

    print("=" * 70)
    print(f"Marked : {len(items)} objects   Without mask: {len(without_mask)}")
    print(f"Saved  : {args.out}")
    print("=" * 70)


if __name__ == "__main__":
    main()