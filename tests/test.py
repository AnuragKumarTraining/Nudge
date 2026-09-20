import json
import cv2
import os
import numpy as np


# ============================================================
# CONFIG
# ============================================================

JSON_PATH = r"C:\Users\anjishnu.kumbhakar\OneDrive - JK Technosoft Ltd\Desktop\nudge_pipeline\outputs\cap_690205_object_discovery.json"

# If your image is actually at another location, change this.
# The JSON currently says: images\daily_2.jpg
IMAGE_OVERRIDE = r"C:\Users\anjishnu.kumbhakar\OneDrive - JK Technosoft Ltd\Desktop\nudge_pipeline\images\daily_2.jpg"

OUTPUT_PATH = "marked_objects.jpg"

# Visualization settings
DRAW_BBOX = True
DRAW_MASK = True
DRAW_LABEL = True
DRAW_CONFIDENCE = True

MASK_ALPHA = 0.25
BOX_THICKNESS = 2
POLYGON_THICKNESS = 2
FONT_SCALE = 0.45
FONT_THICKNESS = 1


# ============================================================
# COLOR GENERATION
# ============================================================

def get_color(index):
    """
    Generate a deterministic color for each object.
    """
    colors = [
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

    return colors[index % len(colors)]


# ============================================================
# LOAD JSON
# ============================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# RESOLVE IMAGE PATH
# ============================================================

if IMAGE_OVERRIDE:
    image_path = IMAGE_OVERRIDE
else:
    json_image_path = data["image"]["path"]

    # JSON contains:
    # images\daily_2.jpg
    #
    # Try relative to JSON file first.
    json_dir = os.path.dirname(os.path.abspath(JSON_PATH))

    image_path = os.path.join(
        json_dir,
        json_image_path
    )


# If the above does not exist, try the JSON path directly
if not os.path.exists(image_path):
    image_path = data["image"]["path"]


if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"\nImage not found.\n"
        f"Tried:\n"
        f"  {image_path}\n\n"
        f"Set IMAGE_OVERRIDE to the actual image path."
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(image_path)

if image is None:
    raise RuntimeError(f"Could not read image: {image_path}")


original = image.copy()

height, width = image.shape[:2]

print(f"Image: {width} x {height}")
print(f"Objects in JSON: {len(data.get('objects', []))}")


# ============================================================
# DRAW OBJECTS
# ============================================================

objects = data.get("objects", [])

mask_layer = image.copy()

valid_objects = 0


for index, obj in enumerate(objects):

    object_id = obj.get("object_id", f"object_{index + 1}")
    class_name = obj.get("class", "unknown")
    confidence = obj.get("confidence", None)

    color = get_color(index)

    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    bbox = obj.get("bbox")

    if bbox:

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        # Clamp coordinates to image
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if DRAW_BBOX:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                BOX_THICKNESS
            )

    # --------------------------------------------------------
    # SEGMENTATION POLYGON
    # --------------------------------------------------------

    polygon_data = obj.get("mask_polygon", {})
    points = polygon_data.get("points_pixel", [])

    if DRAW_MASK and len(points) >= 3:

        polygon = np.array(
            points,
            dtype=np.int32
        ).reshape((-1, 1, 2))

        # Fill mask
        cv2.fillPoly(
            mask_layer,
            [polygon],
            color
        )

        # Polygon boundary
        cv2.polylines(
            image,
            [polygon],
            isClosed=True,
            color=color,
            thickness=POLYGON_THICKNESS
        )

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    if DRAW_LABEL:

        centroid = obj.get("centroid")

        if centroid:
            label_x = int(centroid["x"])
            label_y = int(centroid["y"])
        elif bbox:
            label_x = (x1 + x2) // 2
            label_y = (y1 + y2) // 2
        else:
            continue

        if DRAW_CONFIDENCE and confidence is not None:
            label = (
                f"{index + 1}. {class_name} "
                f"{confidence:.2f}"
            )
        else:
            label = f"{index + 1}. {class_name}"

        # Get label size
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            FONT_THICKNESS
        )

        # Put label above object if possible
        label_x = max(0, min(label_x, width - tw - 4))
        label_y = max(th + 4, label_y)

        # Background rectangle
        cv2.rectangle(
            image,
            (label_x, label_y - th - baseline - 4),
            (label_x + tw + 4, label_y + 2),
            color,
            -1
        )

        # Text
        cv2.putText(
            image,
            label,
            (label_x + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    valid_objects += 1


# ============================================================
# APPLY TRANSPARENT MASK
# ============================================================

if DRAW_MASK:
    image = cv2.addWeighted(
        mask_layer,
        MASK_ALPHA,
        image,
        1 - MASK_ALPHA,
        0
    )


# ============================================================
# TITLE
# ============================================================

title = (
    f"Detected Objects: {valid_objects}"
)

cv2.rectangle(
    image,
    (0, 0),
    (width, 40),
    (30, 30, 30),
    -1
)

cv2.putText(
    image,
    title,
    (10, 27),
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
    image
)

if not success:
    raise RuntimeError(
        f"Failed to save output: {OUTPUT_PATH}"
    )


print("\n====================================")
print("OBJECT VISUALIZATION COMPLETE")
print("====================================")
print(f"Objects marked : {valid_objects}")
print(f"Input image    : {image_path}")
print(f"Output image   : {OUTPUT_PATH}")
print("====================================")


# ============================================================
# PRINT OBJECT SUMMARY
# ============================================================

print("\nDetected objects:")

for i, obj in enumerate(objects, start=1):

    cls = obj.get("class", "unknown")
    conf = obj.get("confidence")

    if conf is not None:
        print(
            f"{i:02d}. "
            f"{obj.get('object_id', ''):30s} "
            f"{cls:25s} "
            f"confidence={conf:.3f}"
        )
    else:
        print(
            f"{i:02d}. "
            f"{obj.get('object_id', ''):30s} "
            f"{cls}"
        )