import cv2
import numpy as np

from config.settings import AMBIENT_LUX_CLAHE_THRESHOLD


def normalize_lighting(
    image: np.ndarray,
    ambient_lux: float | None,
    threshold: float = AMBIENT_LUX_CLAHE_THRESHOLD,
) -> np.ndarray:
    """
    CLAHE on the L channel only (LAB space) — corrects brightness/contrast
    without distorting the color channels that stain detection depends on.
    Same idea as the original preprocess_illumination(), pulled out of
    feature_extraction.py so it can be unit tested and reused independently
    of YOLO.
    """
    if ambient_lux is None or ambient_lux < threshold:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return image
