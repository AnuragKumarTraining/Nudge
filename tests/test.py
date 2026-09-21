
import json
import cv2
import os
import numpy as np


# ============================================================
# CONFIG
# ============================================================

JSON_PATH = (
    r"C:\Users\anjishnu.kumbhakar"
    r"\OneDrive - JK Technosoft Ltd"
    r"\Desktop\nudge_pipeline"
    r"\outputs\cap_690205_object_discovery.json"
)

IMAGE_PATH = (
    r"C:\Users\anjishnu.kumbhakar"
    r"\OneDrive - JK Technosoft Ltd"
    r"\Desktop\nudge_pipeline"
    r"\images\daily_2.jpg"
)

OUTPUT_PATH = "marked_objects_corrected-2.jpg"

# IMPORTANT:
# Start with polygons OFF.
# First verify that bounding boxes are correct.
DRAW_BBOX = True
DRAW_MASK = False
DRAW_LABEL = True
DRAW_CENTROID = True
DRAW_CONFIDENCE = True

MASK_ALPHA = 0.20
BOX_THICKNESS = 3
POLYGON_THICKNESS = 2

FONT_SCALE = 0.50
FONT_THICKNESS = 1


# ============================================================
# COLORS
# ============================================================

COLORS = [
    (255, 80, 80),
    (80, 255, 80),
    (80, 80, 255),
    (255, 200, 50),
    (255, 80, 220),
    (80, 220, 255),
    (180, 80, 255),
    (80, 255, 200),
    (255, 150, 80),
    (150, 255, 80),
]


def get_color(index):
    return COLORS[index % len(COLORS)]


# ============================================================
# LOAD JSON
# ============================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image:\n{IMAGE_PATH}"
    )

height, width = image.shape[:2]

print("=" * 70)
print("IMAGE")
print("=" * 70)
print(f"Actual image size : {width} x {height}")

json_width = data["image"]["width"]
json_height = data["image"]["height"]

print(f"JSON image size   : {json_width} x {json_height}")

if width != json_width or height != json_height:
    print("\nWARNING:")
    print("The actual image size does NOT match the JSON image size.")
    print("Coordinates may therefore be shifted/scaled.")

print()


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def normalized_to_pixel(bbox_normalized):
    """
    Convert normalized [0,1] coordinates into actual image pixels.
    """

    x1 = round(bbox_normalized["x1"] * width)
    y1 = round(bbox_normalized["y1"] * height)

    x2 = round(bbox_normalized["x2"] * width)
    y2 = round(bbox_normalized["y2"] * height)

    return x1, y1, x2, y2


def clamp_bbox(x1, y1, x2, y2):

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))

    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))

    # Ensure proper ordering
    if x1 > x2:
        x1, x2 = x2, x1

    if y1 > y2:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


# ============================================================
# OBJECTS
# ============================================================

objects = data.get("objects", [])

print("=" * 70)
print(f"OBJECTS IN JSON: {len(objects)}")
print("=" * 70)


# ============================================================
# CREATE SEPARATE LAYERS
# ============================================================

bbox_layer = image.copy()
mask_layer = image.copy()

valid_objects = 0


# ============================================================
# DRAW OBJECTS
# ============================================================

for index, obj in enumerate(objects):

    object_id = obj.get(
        "object_id",
        f"object_{index + 1}"
    )

    class_name = obj.get(
        "class",
        "unknown"
    )

    confidence = obj.get(
        "confidence",
        None
    )

    color = get_color(index)

    # --------------------------------------------------------
    # GET BOUNDING BOX
    # --------------------------------------------------------

    bbox = obj.get("bbox")
    bbox_normalized = obj.get("bbox_normalized")

    x1 = y1 = x2 = y2 = None

    # Prefer normalized coordinates because they can be
    # recalculated against the ACTUAL image dimensions.
    if bbox_normalized:

        x1, y1, x2, y2 = normalized_to_pixel(
            bbox_normalized
        )

        coordinate_source = "normalized"

    elif bbox:

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        coordinate_source = "pixel"

    else:
        coordinate_source = "none"

    if x1 is not None:

        x1, y1, x2, y2 = clamp_bbox(
            x1, y1, x2, y2
        )

        # ----------------------------------------------------
        # DRAW MASK FIRST
        # ----------------------------------------------------

        if DRAW_MASK:

            polygon_data = obj.get(
                "mask_polygon",
                {}
            )

            points = polygon_data.get(
                "points_pixel",
                []
            )

            if len(points) >= 3:

                polygon = np.array(
                    points,
                    dtype=np.int32
                ).reshape((-1, 1, 2))

                # Clamp polygon points
                polygon[:, :, 0] = np.clip(
                    polygon[:, :, 0],
                    0,
                    width - 1
                )

                polygon[:, :, 1] = np.clip(
                    polygon[:, :, 1],
                    0,
                    height - 1
                )

                cv2.fillPoly(
                    mask_layer,
                    [polygon],
                    color
                )

                cv2.polylines(
                    bbox_layer,
                    [polygon],
                    True,
                    color,
                    POLYGON_THICKNESS,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # DRAW BBOX
        # ----------------------------------------------------

        if DRAW_BBOX:

            cv2.rectangle(
                bbox_layer,
                (x1, y1),
                (x2, y2),
                color,
                BOX_THICKNESS
            )

        # ----------------------------------------------------
        # CENTROID
        # ----------------------------------------------------

        if DRAW_CENTROID:

            centroid = obj.get("centroid")

            if centroid:

                cx = int(centroid["x"])
                cy = int(centroid["y"])

                # Clamp
                cx = max(0, min(cx, width - 1))
                cy = max(0, min(cy, height - 1))

                cv2.drawMarker(
                    bbox_layer,
                    (cx, cy),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2
                )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        if DRAW_LABEL:

            label = f"{index + 1}. {class_name}"

            if DRAW_CONFIDENCE and confidence is not None:
                label += f" {confidence:.2f}"

            label += f" [{coordinate_source}]"

            (tw, th), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                FONT_THICKNESS
            )

            # Prefer placing label above bbox
            label_x = x1
            label_y = y1 - 5

            # If there isn't enough space above,
            # put it inside the bbox.
            if label_y - th - baseline < 0:
                label_y = y1 + th + 5

            label_x = max(
                0,
                min(
                    label_x,
                    width - tw - 6
                )
            )

            # Background
            cv2.rectangle(
                bbox_layer,
                (
                    label_x,
                    label_y - th - baseline - 4
                ),
                (
                    label_x + tw + 6,
                    label_y + 2
                ),
                color,
                -1
            )

            # Text
            cv2.putText(
                bbox_layer,
                label,
                (
                    label_x + 3,
                    label_y - 2
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                FONT_THICKNESS,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # PRINT COORDINATES
        # ----------------------------------------------------

        print(
            f"{index + 1:02d} "
            f"{class_name:15s} "
            f"bbox=({x1},{y1})-({x2},{y2}) "
            f"source={coordinate_source}"
        )

        valid_objects += 1


# ============================================================
# APPLY MASK
# ============================================================

if DRAW_MASK:

    # Apply mask BEFORE final bbox/label layer
    output = cv2.addWeighted(
        mask_layer,
        MASK_ALPHA,
        bbox_layer,
        1.0,
        0
    )

else:

    output = bbox_layer


# ============================================================
# TITLE
# ============================================================

title = f"Detected Objects: {valid_objects}"

cv2.rectangle(
    output,
    (0, 0),
    (width, 45),
    (30, 30, 30),
    -1
)

cv2.putText(
    output,
    title,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ============================================================
# SAVE
# ============================================================

success = cv2.imwrite(
    OUTPUT_PATH,
    output
)

if not success:
    raise RuntimeError(
        f"Could not save:\n{OUTPUT_PATH}"
    )


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"Objects marked : {valid_objects}")
print(f"Input image    : {IMAGE_PATH}")
print(f"Output image   : {OUTPUT_PATH}")
print("=" * 70)
