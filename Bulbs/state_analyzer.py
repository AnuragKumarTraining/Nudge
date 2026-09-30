import cv2
import numpy as np


def get_luminosity(image):
    """Safely converts 1-channel, 3-channel (BGR), or 4-channel (BGRA) images to grayscale."""
    if image is None:
        raise ValueError("Image passed to get_luminosity is None.")

    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        return image

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unexpected image shape: {image.shape}")


def scene_brightness(gray):
    """
    Continuous scene-brightness factor.
    0.0 = night scene, 1.0 = bright indoor / daylight scene.
    Uses the median (not the mean) so flares and big lit walls don't fool it.
    """
    return float(np.clip((float(np.median(gray)) - 50) / 70.0, 0.0, 1.0))


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
    """Identifies straight, elongated linear light fixtures (tube lights, batten LEDs)."""
    area = cv2.contourArea(contour)
    if area < 60:
        return False

    _, _, bw, bh = cv2.boundingRect(contour)
    major = max(bw, bh)
    minor = min(bw, bh) + 1e-5
    aspect_ratio = major / minor

    if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    return solidity >= min_solidity


def blob_light_stats(image, gray, contour, white_thr=235):
    """
    Returns (core_frac, contrast)
      core_frac : fraction of blob pixels that are near-white in ALL channels
                  (bulb cores clip to white; green leaf glare doesn't).
                  white_thr is lowered for bright scenes, where warm-white LEDs
                  rarely reach 235 in the blue channel.
      contrast  : mean brightness inside blob minus mean of surrounding ring
    """
    H, W = gray.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)

    # Ring width scales with the blob's SHORT side and is capped, so long
    # linear strips don't measure contrast against half the room.
    r = min(max(6, int(0.75 * min(bw, bh)) + 6), 30)

    x1, y1 = max(0, x - r - 2), max(0, y - r - 2)
    x2, y2 = min(W, x + bw + r + 2), min(H, y + bh + r + 2)

    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-x1, -y1))

    inner = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    outer = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    inside = mask > 0
    ring = (outer > 0) & (inner == 0)

    roi_gray = gray[y1:y2, x1:x2]
    if image.ndim == 3:
        whiteness = image[y1:y2, x1:x2, :3].min(axis=2)   # min(B,G,R)
    else:
        whiteness = roi_gray

    core_frac = float((whiteness[inside] >= white_thr).mean()) if inside.any() else 0.0
    contrast = float(roi_gray[inside].mean() - roi_gray[ring].mean()) if ring.any() else 0.0
    return core_frac, contrast


def detect_bulb_candidates(image, threshold_value=None, min_area=None, pad_pixels=15, debug=False):
    """
    Finds potential active light sources, applying NMS to prevent duplicates
    and separating the VLM crop box from the tight visual drawing box.

    All thresholds adapt continuously to scene brightness `t`
    (0 = night, 1 = bright indoor/day). At t = 0 the behaviour is identical
    to the original night-tuned pipeline.
    """
    h, w = image.shape[:2]
    gray = get_luminosity(image)
    smooth = reduce_haziness(gray, gamma=2.0)

    # Scene-adaptive parameters
    t = scene_brightness(gray)
    white_thr = int(round(235 - 30 * t))     # 235 night -> 205 day
    min_contrast = 50 - 20 * t               # 50 night  -> 30 day

    # 1. Lower the min_area multiplier slightly to catch distant garden pathway lights
    if min_area is None:
        min_area = max(4, int(h * w * 0.000004))

    # 2. Adaptive threshold. Night: floor 175 / cap 215 (unchanged).
    #    Day: floor 200 / cap 230 so ceiling glow isn't picked up,
    #    while brighter LEDs still survive.
    if threshold_value is None:
        peak_luminance = float(np.percentile(smooth, 99.7))
        lo = 175 + int(25 * t)
        hi = 215 + int(15 * t)
        threshold_value = min(hi, max(lo, int(peak_luminance * 0.85)))

    thresh = isolate_active_bulbs(smooth, threshold_value=threshold_value)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_candidates = []
    safe_pad = 15 if pad_pixels is None else int(pad_pixels)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        if not (is_round_or_oval(c) or is_linear_light_source(c)):
            continue

        core_frac, contrast = blob_light_stats(image, gray, c, white_thr)
        if debug:
            print(f"t={t:.2f} thr={threshold_value} area={area:.0f} "
                  f"core={core_frac:.2f} contrast={contrast:.0f}")
        if core_frac < 0.10 or contrast < min_contrast:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        # --- Tight Box for Drawing (Very small padding) ---
        draw_pad = 4
        dx1 = max(0, bx - draw_pad)
        dy1 = max(0, by - draw_pad)
        dx2 = min(w, bx + bw + draw_pad)
        dy2 = min(h, by + bh + draw_pad)

        # --- Padded Box for VLM Classification (Context) ---
        actual_pad = safe_pad + 10 if max(bw, bh) < 25 else safe_pad
        x1 = max(0, bx - actual_pad)
        y1 = max(0, by - actual_pad)
        x2 = min(w, bx + bw + actual_pad)
        y2 = min(h, by + bh + actual_pad)

        raw_candidates.append({
            "bbox": [x1, y1, x2, y2],
            "draw_bbox": [dx1, dy1, dx2, dy2],
            "area": float(area),
            "core_frac": core_frac,
            "contrast": contrast,
        })

    if not raw_candidates:
        return []

    # 3. Non-Maximum Suppression (NMS)
    boxes_xywh = [
        [c["draw_bbox"][0], c["draw_bbox"][1],
         c["draw_bbox"][2] - c["draw_bbox"][0],
         c["draw_bbox"][3] - c["draw_bbox"][1]]
        for c in raw_candidates
    ]
    scores = [c["area"] for c in raw_candidates]

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=0.0,
        nms_threshold=0.2
    )

    if len(indices) > 0:
        return [raw_candidates[i] for i in np.array(indices).flatten()]
    return []