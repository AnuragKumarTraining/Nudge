import cv2
import numpy as np


def detect_stains(masked_crop, crop_mask):
    # Convert to CIELAB space to separate Lightness from Color
    lab = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Adaptive threshold on Lightness (detects dark/light spots)
    thresh = cv2.adaptiveThreshold(
        l_channel, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25, 5
    )
    
    # CRITICAL: Restrict thresholds strictly inside the polygon mask
    thresh = cv2.bitwise_and(thresh, thresh, mask=crop_mask)
    
    # Clean noise using morphological opening
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours (candidate stains)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stain_candidates = []
    total_stain_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 4000: # Filter out tiny noise and huge surface selections
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            x, y, w, h = cv2.boundingRect(cnt)
            stain_candidates.append({
                "local_bbox": [x, y, x + w, y + h],
                "area_px": area,
                "circularity": round(circularity, 3) # Low circularity = organic stain shape
            })
            total_stain_area += area
            
    return stain_candidates, total_stain_area