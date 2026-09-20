import cv2
import numpy as np

from config.settings import (
    STAIN_MIN_AREA_PX, STAIN_MAX_AREA_PX,
    STAIN_ADAPTIVE_BLOCK_SIZE, STAIN_ADAPTIVE_C,
)


def _is_periodic_pattern(region: np.ndarray, threshold: float = 0.6) -> bool:
    """FFT-based check: fabric/print patterns are periodic, real stains aren't."""
    if region.shape[0] < 4 or region.shape[1] < 4:
        return False
    f_shift = np.fft.fftshift(np.fft.fft2(region))
    magnitude = np.abs(f_shift)
    peak_ratio = np.max(magnitude) / (np.mean(magnitude) + 1e-6)
    return peak_ratio > threshold


def detect_stain_candidates(
    crop: np.ndarray,
    mask: np.ndarray | None = None,
    min_area: int = STAIN_MIN_AREA_PX,
    max_area: int = STAIN_MAX_AREA_PX,
) -> list[dict]:
    """
    Returns candidates with bbox_local (crop-relative — caller must translate
    to full-image coords) plus raw_confidence. This is the OpenCV half of the
    hybrid stain pipeline; semantic.enrich.confirm_stains_with_vlm() runs
    next and produces adjusted_confidence + vlm_verdict.
    """
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)
    if mask is not None:
        l_channel = cv2.bitwise_and(l_channel, l_channel, mask=mask)

    thresh = cv2.adaptiveThreshold(
        l_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        STAIN_ADAPTIVE_BLOCK_SIZE, STAIN_ADAPTIVE_C,
    )
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

        region = l_channel[y:y + h, x:x + w]
        periodic = _is_periodic_pattern(region) if region.size > 0 else False

        confidence = 0.8
        if periodic:
            confidence *= 0.2  # very likely a fabric/print pattern, not a real mark
        if circularity > 0.85:
            confidence *= 0.6  # unnaturally regular shape

        candidates.append({
            "bbox_local": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
            "area_px": int(area),
            "circularity": round(float(circularity), 2),
            "is_periodic_pattern": bool(periodic),
            "raw_confidence": round(confidence, 2),
        })

    return candidates
