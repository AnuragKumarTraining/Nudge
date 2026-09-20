"""
OpenCV half of the hybrid stain pipeline: OVER-generate candidate marks on a surface.
semantic/enrich.py::confirm_stains_with_vlm() then decides stain / pattern / shadow.

Two bugs fixed vs the previous version (both verified with synthetic tests):
  1. Mask handling. The old code zeroed the object regions in the L channel and THEN ran
     adaptive thresholding, so every masked-out object (cup, plate, chair...) produced a
     phantom "stain" exactly at its own bounding box. Now: threshold the untouched L
     channel, then AND the result with the (eroded) exposed-surface mask.
  2. Periodic-pattern check. The old FFT test compared the spectrum peak - which is
     always the DC term - to 0.6 x the mean, so EVERY region was flagged "periodic" and
     every candidate was down-weighted to 0.16. Now the DC term is removed, a Hann window
     is applied, and low frequencies are excluded before measuring the peak.
"""
import cv2
import numpy as np

from config.settings import (
    STAIN_MAX_CANDIDATES_PER_OBJECT,
    STAIN_PERIODIC_PEAK_RATIO,
    STAIN_MASK_ERODE_PX,
)


def _is_periodic_pattern(region: np.ndarray, threshold: float = STAIN_PERIODIC_PEAK_RATIO) -> bool:
    """Fabric weaves / prints / tiles have a sharp spectral peak; real stains do not."""
    if min(region.shape[:2]) < 16:
        # too few pixels for a meaningful spectrum
        return False

    r = region.astype(np.float32)
    r -= r.mean()  # remove DC
    if r.std() < 1.0:
        # flat patch: nothing periodic about it
        return False

    h, w = r.shape
    r *= np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(r)))

    yy, xx = np.ogrid[:h, :w]
    mag[(yy - h // 2) ** 2 + (xx - w // 2) ** 2 <= 3 ** 2] = 0  # drop very low frequencies

    return bool(mag.max() / (mag.mean() + 1e-6) > threshold)


def detect_stain_candidates(
    crop: np.ndarray,
    mask: np.ndarray | None = None,
    max_candidates: int = STAIN_MAX_CANDIDATES_PER_OBJECT,
) -> list[dict]:
    """
    Returns at most `max_candidates` candidates with bbox_local (crop-relative - the caller
    translates to full-image coords) and raw_confidence.
    """
    if crop is None or crop.size == 0:
        return []

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    if mask is not None:
        # exposed surface only, with a safety margin so object edges/contact shadows don't leak in
        k = 2 * STAIN_MASK_ERODE_PX + 1
        safe = cv2.erode(mask, np.ones((k, k), np.uint8))
        cleaned = cv2.bitwise_and(cleaned, safe)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        confidence = 0.8

        periodic = _is_periodic_pattern(l_channel[y:y + h, x:x + w])
        if periodic:
            confidence *= 0.2  # very likely a fabric/print pattern, not a real mark

        solidity = area / float(w * h + 1e-6)
        if solidity > 0.85 or solidity < 0.15:
            confidence *= 0.6  # unnaturally regular shape

        candidates.append({
            "bbox_local": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},
            "raw_confidence": round(confidence, 2),
            "area_px": int(area),
        })

    # each surviving candidate costs one VLM call -> keep only the most promising ones
    candidates.sort(key=lambda c: (c["raw_confidence"], c["area_px"]), reverse=True)
    return candidates[:max_candidates]