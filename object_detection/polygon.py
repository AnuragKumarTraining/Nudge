"""
Contour-based polygon extraction, pulled out of feature_extraction.py so it
can be called/tested independently of YOLO.

Known limitation, same as the original: this approximates a polygon from
Otsu-thresholded contours within the bbox crop, which works reasonably for
objects that contrast against their background but is not a true
segmentation mask. The clean upgrade path is switching detector.py's YOLO
weights to a "-seg" variant (e.g. yolov8n-seg.pt) which returns proper
instance masks/polygons directly from the model — swap this module out
for reading result.masks.xy at that point rather than patching this one
further.
"""
import cv2
import numpy as np


def extract_polygon(image: np.ndarray, obj: dict) -> dict:
    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = obj["bbox"]["x1"], obj["bbox"]["y1"], obj["bbox"]["x2"], obj["bbox"]["y2"]
    roi = image[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]

    empty = {"orientation_deg": 0.0, "points_pixel": [], "points_normalized": []}
    if roi.size == 0:
        return empty

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return empty

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = round(rect[2], 2)

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    points_pixel, points_normalized = [], []
    for point in approx:
        px, py = int(point[0][0] + x1), int(point[0][1] + y1)
        points_pixel.append([px, py])
        points_normalized.append([round(px / img_w, 4), round(py / img_h, 4)])

    return {
        "orientation_deg": angle,
        "points_pixel": points_pixel,
        "points_normalized": points_normalized,
    }
