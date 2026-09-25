import cv2
import sys
import math
import numpy as np

from .model import *

def preprocess_illumination(image, ambient_lux):
    """
    Applies OpenCV CLAHE contrast normalization if the environment is too dark.
    """
    if ambient_lux < 250:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return image

def extract_orientation_and_polygon(roi_crop, offset_x, offset_y, img_w, img_h):
    """
    Computes the rotation angle and returns both pixel and normalized polygon boundaries.
    """
    gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0, [], []
        
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = round(rect[2], 2)

    # Approximate the contour to create a clean, simplified polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    polygon_pixels = []
    polygon_normalized = []

    for point in approx_contour:
        px = int(point[0][0] + offset_x)
        py = int(point[0][1] + offset_y)
        
        polygon_pixels.append([px, py])
        polygon_normalized.append([round(px / img_w, 4), round(py / img_h, 4)])

    return angle, polygon_pixels, polygon_normalized

def extract_features(image_path, model, stage_metadata):
    """
    Extracts Stage 3 JSON metadata with bounding boxes, spatial relationships,
    and polygon masks.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot find or read image at: {image_path}. Check file path.")

    img_h, img_w = img.shape[:2]
    capture_id = stage_metadata.get("capture_id", "unknown_cap")
    environment = stage_metadata.get("environment", {})
    ambient_lux = environment.get("ambient_lux", 350)

    # 1. Preprocess Lighting
    img = preprocess_illumination(img, ambient_lux)

    # 2. YOLO Object Detection
    results = model(img, verbose=False)
    
    objects = []
    object_counts = {}
    total_confidence = 0.0
    detected_count = 0

    # PASS 1: Extract Base Architecture + Mask Polygons
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = round(float(box.conf[0]), 2)
            
            total_confidence += conf
            detected_count += 1
            
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            instance_id = f"obj_{detected_count:03d}_{class_name}"
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            
            bbox_norm = {
                "x1": round(x1 / img_w, 3), 
                "y1": round(y1 / img_h, 3), 
                "x2": round(x2 / img_w, 3), 
                "y2": round(y2 / img_h, 3)
            }
            
            # Crop ROI
            roi = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            
            # Extract rotation angle and polygon masks
            angle, poly_pixels, poly_norm = extract_orientation_and_polygon(
                roi_crop=roi, 
                offset_x=max(0, x1), 
                offset_y=max(0, y1), 
                img_w=img_w, 
                img_h=img_h
            )
            
            objects.append({
                "object_id": instance_id,
                "class": class_name,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "bbox_normalized": bbox_norm,
                "centroid": {"x": centroid_x, "y": centroid_y},
                "orientation_deg": angle,
                "mask_polygon": {
                    "points_pixel": poly_pixels,
                    "points_normalized": poly_norm
                },
                "surface_relevant": True,
                "material_hint": "pending_vlm",
                "state": {
                    "surface_condition": "pending_vlm",
                    "semantic_alignment": "pending_vlm"
                },
                "relationships": {}
            })

    # PASS 2: Object Relationships
    tables = [obj for obj in objects if obj["class"] in ["table", "dining_table"]]
    
    for obj in objects:
        if obj["class"] in ["table", "dining_table", "floor", "wall"]:
            continue
            
        closest_table = None
        min_dist = float('inf')
        
        for table in tables:
            tx, ty = table["centroid"]["x"], table["centroid"]["y"]
            ox, oy = obj["centroid"]["x"], obj["centroid"]["y"]
            dist = math.hypot(tx - ox, ty - oy)
            
            if dist < min_dist:
                min_dist = dist
                closest_table = table
                
        if closest_table:
            tx, ty = closest_table["centroid"]["x"], closest_table["centroid"]["y"]
            ox, oy = obj["centroid"]["x"], obj["centroid"]["y"]
            
            vertical = "top" if oy < ty else "bottom"
            horizontal = "left" if ox < tx else "right"
            
            obj["relationships"]["relative_position"] = f"{vertical}-{horizontal} of {closest_table['object_id']}"
            obj["relationships"]["belongs_to"] = closest_table['object_id']

    avg_confidence = round(total_confidence / detected_count, 2) if detected_count > 0 else 0.0

    stage_3_output = {
        "capture_id": capture_id,
        "objects": objects,
        "object_counts": object_counts,
        "extractor_used": "hybrid_yolo_opencv",
        "detection_confidence": avg_confidence
    }
    
    return stage_3_output, img