import math
import numpy as np

def wrap_angle_deg(angle: float) -> float:
    """Wraps angles into the [-180, 180] degree domain."""
    return (angle + 180.0) % 360.0 - 180.0

def extract_head_pose_deg(transformation_matrix: list | np.ndarray) -> tuple[float, float]:
    """Computes yaw and pitch in degrees from facial transformation matrix."""
    rot = np.array(transformation_matrix, dtype=float)[:3, :3]
    rot = rot / np.linalg.norm(rot, axis=0)
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2])))
    return yaw, pitch

def compute_hysteresis(value: float, prev_state: int, enter_th: float, exit_th: float) -> int:
    """Three-state (-1, 0, 1) hysteresis classifier."""
    if prev_state == 1:
        if value > exit_th:
            return 1
        return -1 if value < -enter_th else 0
    if prev_state == -1:
        if value < -exit_th:
            return -1
        return 1 if value > enter_th else 0
    if value > enter_th:
        return 1
    if value < -enter_th:
        return -1
    return 0