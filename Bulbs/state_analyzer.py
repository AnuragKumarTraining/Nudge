import cv2
import numpy as np

def get_luminosity(image):
    """Safely converts 1-channel, 3-channel (BGR), or 4-channel (BGRA) images to grayscale."""
    if image is None:
        raise ValueError("Image passed to get_luminosity is None.")
    
    # Already single-channel grayscale
    if len(image.shape) == 2:
        return image
    
    # 4-channel (PNG with alpha channel)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    # Standard 3-channel BGR
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    raise ValueError(f"Unexpected image shape: {image.shape}")

def reduce_haziness(gray_image, gamma=2.2):
    """Reduces haziness/glare using Gamma Correction via a Look-Up Table (LUT)."""
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(gray_image, table)

def isolate_active_bulbs(smoothed_image, threshold_value=210):
    """Applies binary thresholding to isolate bright glowing regions."""
    _, thresh = cv2.threshold(smoothed_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def is_round_or_oval(contour, min_circularity=0.10, min_axis_ratio=0.15, min_solidity=0.60):
    """Checks if a light emission blob resembles a bulb source."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0 or area == 0:
        return False

    circularity = (4 * np.pi * area) / (perimeter ** 2)

    if len(contour) >= 5:
        (_, _), (d1, d2), _ = cv2.fitEllipse(contour)
        axis_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 0.0
    else:
        _, _, w, h = cv2.boundingRect(contour)
        axis_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return (circularity >= min_circularity) or (axis_ratio >= min_axis_ratio) or (solidity >= min_solidity)

def detect_bulb_candidates(image, threshold_value=210, min_area=15, pad_pixels=12):
    """
    Finds potential active light sources using luminance and shape filtering.
    Returns: list of dicts with bounding boxes padded for VLM context.
    """
    h, w, _ = image.shape
    gray = get_luminosity(image)
    smooth = reduce_haziness(gray, gamma=2.2)
    thresh = isolate_active_bulbs(smooth, threshold_value=threshold_value)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        if cv2.contourArea(c) > min_area and is_round_or_oval(c):
            bx, by, bw, bh = cv2.boundingRect(c)
            
            # Add spatial padding around the glowing core so the VLM sees the fixture / context
            x1 = max(0, bx - pad_pixels)
            y1 = max(0, by - pad_pixels)
            x2 = min(w, bx + bw + pad_pixels)
            y2 = min(h, by + bh + pad_pixels)

            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "area": float(cv2.contourArea(c))
            })

    return candidates