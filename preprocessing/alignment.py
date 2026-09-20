"""
ORB + homography alignment, refactored from the original align_images.py.
Two real changes from the original:
  1. It no longer raises exceptions on failure — it returns a result dict
     with success=False and a reason, so the pipeline can decide what to do
     (reject the capture, still run detection at lower confidence, etc.)
     rather than crashing.
  2. It uses the capture's own spatial_alignment gyro deltas to pre-rotate
     before running ORB, which meaningfully helps ORB's matching when the
     phone was held at an angle.
"""
import cv2
import numpy as np

from config.settings import (
    ORB_MAX_FEATURES, ORB_MATCH_RATIO, ORB_RANSAC_THRESHOLD,
    ORB_MIN_GOOD_MATCHES, YAW_DELTA_WARNING_DEG,
)


def pre_rotate_by_roll(image: np.ndarray, roll_delta_deg: float) -> np.ndarray:
    """
    roll_delta_deg is the in-plane camera tilt (phone rotated sideways) —
    this is the component a simple 2D rotation can actually correct.
    pitch/yaw are out-of-plane (perspective) changes; those are what the
    homography step below has to absorb, a 2D rotation can't fix them.
    """
    if abs(roll_delta_deg) < 2:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), -roll_delta_deg, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))


def align_to_master(
    master_img: np.ndarray,
    new_img: np.ndarray,
    spatial_alignment: dict | None = None,
) -> tuple[np.ndarray, dict]:
    warnings = []
    spatial_alignment = spatial_alignment or {}
    roll = spatial_alignment.get("roll_delta_deg", 0.0)
    yaw = spatial_alignment.get("yaw_delta_deg", 0.0)

    if abs(yaw) > YAW_DELTA_WARNING_DEG:
        warnings.append(
            f"yaw_delta_deg={yaw} exceeds {YAW_DELTA_WARNING_DEG} — capture angle is very "
            "different from the master reference; homography below may not fully recover "
            "this and a retake is likely the better outcome."
        )

    working_img = pre_rotate_by_roll(new_img, roll)

    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(ORB_MAX_FEATURES)
    kp_new, desc_new = orb.detectAndCompute(new_gray, None)
    kp_master, desc_master = orb.detectAndCompute(master_gray, None)

    if desc_new is None or desc_master is None:
        return new_img, {
            "method": "orb_homography", "success": False,
            "reason": "no_descriptors_found", "warnings": warnings,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc_new, desc_master, k=2)
    good = [m for m, n in matches if m.distance < ORB_MATCH_RATIO * n.distance]

    if len(good) < ORB_MIN_GOOD_MATCHES:
        return new_img, {
            "method": "orb_homography", "success": False,
            "reason": f"only_{len(good)}_good_matches_need_{ORB_MIN_GOOD_MATCHES}",
            "warnings": warnings,
        }

    pts_new = np.float32([kp_new[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_master = np.float32([kp_master[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(pts_new, pts_master, cv2.RANSAC, ORB_RANSAC_THRESHOLD)

    if matrix is None:
        return new_img, {
            "method": "orb_homography", "success": False,
            "reason": "homography_computation_failed", "warnings": warnings,
        }

    h, w = master_img.shape[:2]
    aligned = cv2.warpPerspective(working_img, matrix, (w, h))
    inliers = int(mask.sum()) if mask is not None else len(good)
    confidence = round(min(1.0, inliers / max(len(good), 1)), 2)

    return aligned, {
        "method": "orb_homography", "success": True,
        "matches_used": len(good), "inliers": inliers,
        "confidence": confidence, "warnings": warnings,
    }
