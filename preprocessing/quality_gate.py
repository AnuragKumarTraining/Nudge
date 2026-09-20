import cv2
import numpy as np

from config.settings import BLUR_VARIANCE_THRESHOLD


def check_quality(image: np.ndarray, threshold: float = BLUR_VARIANCE_THRESHOLD) -> dict:
    """
    Laplacian-variance blur check. Everything downstream should stop if
    is_passed is False — running detection on an unusable photo wastes
    compute and produces garbage flags.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return {
        "blur_variance": round(float(variance), 2),
        "blur_threshold": threshold,
        "is_passed": bool(variance >= threshold),
    }
