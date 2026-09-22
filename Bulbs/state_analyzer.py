import cv2
import numpy as np

def get_luminosity(image):
    """Safely converts 1-channel, 3-channel (BGR), or 4-channel (BGRA) images to grayscale."""
    if image is None:
        raise ValueError("Image passed to get_luminosity is None.")

    # Ensure memory buffer is contiguous and standard uint8
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

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

def isolate_active_bulbs(smoothed_image, threshold_value=225):
    """Applies binary thresholding to isolate bright glowing regions."""
    _, thresh = cv2.threshold(smoothed_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def is_round_or_oval(contour, min_circularity=0.05, min_axis_ratio=0.15, min_solidity=0.55):
    """Checks if a light emission blob resembles a bulb source or lens bloom."""
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

def is_linear_light_source(contour, min_aspect_ratio=2.5, max_aspect_ratio=25.0, min_solidity=0.70):
    """
    Identifies straight, elongated linear light fixtures (tube lights, batten LEDs).
    Rejects scattered specular glares on floors/tiles by checking rectangular solidity.
    """
    area = cv2.contourArea(contour)
    if area < 60:
        return False

    _, _, bw, bh = cv2.boundingRect(contour)
    major = max(bw, bh)
    minor = min(bw, bh) + 1e-5
    aspect_ratio = major / minor

    # Must be noticeably elongated, but not a thin scratch or tile seam
    if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
        return False

    # A genuine tube fixture maintains a cohesive rectangular footprint
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return solidity >= min_solidity

def detect_bulb_candidates(image, threshold_value=None, min_area=None, pad_pixels=25):
    """
    Finds potential active light sources using adaptive resolution scaling
    and luminance statistics to handle both small string bulbs and large fixtures.
    """
    h, w = image.shape[:2]
    gray = get_luminosity(image)
    smooth = reduce_haziness(gray, gamma=2.2)

    # 1. Adapt min_area to image size (e.g. 4-8 px for tiny bulbs, scales up with resolution)
    if min_area is None:
        min_area = max(4, int(h * w * 0.000006))

    # 2. Adapt threshold based on the top 0.3% brightest pixels in the frame
    if threshold_value is None:
        peak_luminance = float(np.percentile(smooth, 99.7))
        threshold_value = max(190, int(peak_luminance * 0.90))

    thresh = isolate_active_bulbs(smooth, threshold_value=threshold_value)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        if not (is_round_or_oval(c) or is_linear_light_source(c)):
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        actual_pad = pad_pixels + 15 if max(bw, bh) < 25 else pad_pixels

        x1 = max(0, bx - actual_pad)
        y1 = max(0, by - actual_pad)
        x2 = min(w, bx + bw + actual_pad)
        y2 = min(h, by + bh + actual_pad)

        candidates.append({
            "bbox": [x1, y1, x2, y2],
            "area": float(area)
        })

    return candidates