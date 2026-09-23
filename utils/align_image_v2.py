
import cv2
from dataclasses import dataclass
import numpy as np
from typing import Optional
from skimage.metrics import structural_similarity as ssim
from config.align_config import*
@dataclass
class AlignResult:
    ok: bool
    aligned: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None   # True where warped pixels are real
    n_good: int = 0
    inlier_ratio: float = 0.0
    reason: Optional[str] = None
 
 
def align(master: np.ndarray, current: np.ndarray, max_features: int = 5000,
          match_ratio: float = 0.75) -> AlignResult:
    """Replacement for utils.align_images.align_images that reports quality."""
    mg = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(max_features)
    kp_c, des_c = orb.detectAndCompute(cg, None)
    kp_m, des_m = orb.detectAndCompute(mg, None)
    if des_c is None or des_m is None:
        return AlignResult(False, reason="no_descriptors")
 
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_c, des_m, k=2)
    good = [p[0] for p in pairs
            if len(p) == 2 and p[0].distance < match_ratio * p[1].distance]
    if len(good) < MIN_GOOD_MATCHES:
        return AlignResult(False, n_good=len(good), reason="too_few_matches")
 
    src = np.float32([kp_c[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    if H is None:
        return AlignResult(False, n_good=len(good), reason="homography_failed")
 
    inliers = int(mask.sum())
    ratio = inliers / len(good)
    if inliers < MIN_GOOD_MATCHES or ratio < MIN_INLIER_RATIO:
        return AlignResult(False, n_good=len(good), inlier_ratio=ratio, reason="low_inliers")
    if not (0.25 <= abs(np.linalg.det(H[:2, :2])) <= 4.0):
        return AlignResult(False, n_good=len(good), inlier_ratio=ratio,
                           reason="implausible_transform")
 
    h, w = master.shape[:2]
    aligned = cv2.warpPerspective(current, H, (w, h))
    valid = cv2.warpPerspective(np.full(current.shape[:2], 255, np.uint8), H, (w, h)) > 0
    return AlignResult(True, aligned, valid, len(good), ratio)
 