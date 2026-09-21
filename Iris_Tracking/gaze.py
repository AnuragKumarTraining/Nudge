import numpy as np
from config import AppConfig
from geometry import compute_hysteresis, extract_head_pose_deg, wrap_angle_deg

def extract_face_signals(detection_result, config: AppConfig) -> tuple[float, float, float, float, float]:
    """Extracts horizontal (h), vertical (v), blink, yaw, and pitch from detection data."""
    blendshapes = {c.category_name: c.score for c in detection_result.face_blendshapes[0]}

    look_left = (blendshapes.get("eyeLookOutLeft", 0) + blendshapes.get("eyeLookInRight", 0)) / 2.0
    look_right = (blendshapes.get("eyeLookInLeft", 0) + blendshapes.get("eyeLookOutRight", 0)) / 2.0
    look_up = (blendshapes.get("eyeLookUpLeft", 0) + blendshapes.get("eyeLookUpRight", 0)) / 2.0
    look_down = (blendshapes.get("eyeLookDownLeft", 0) + blendshapes.get("eyeLookDownRight", 0)) / 2.0
    blink = (blendshapes.get("eyeBlinkLeft", 0) + blendshapes.get("eyeBlinkRight", 0)) / 2.0

    h = look_right - look_left
    if config.swap_left_right:
        h = -h
    v = look_down - look_up

    yaw, pitch = extract_head_pose_deg(detection_result.facial_transformation_matrixes[0])
    return h, v, blink, yaw, pitch

class Calibrator:
    """Calibrates neutral screen-center baseline coordinates."""
    def __init__(self, target_frames: int):
        self.target_frames = target_frames
        self.active = False
        self.samples: list[tuple[float, float, float, float]] = []
        self.baseline = {"h": 0.0, "v": 0.0, "yaw": 0.0, "pitch": 0.0}

    def start(self) -> None:
        self.active = True
        self.samples.clear()

    def update(self, h: float, v: float, yaw: float, pitch: float) -> bool:
        self.samples.append((h, v, yaw, pitch))
        if len(self.samples) >= self.target_frames:
            med = np.median(np.array(self.samples), axis=0)
            self.baseline = {
                "h": float(med[0]),
                "v": float(med[1]),
                "yaw": float(med[2]),
                "pitch": float(med[3]),
            }
            self.active = False
            return True
        return False

class GazeClassifier:
    """Filters signals via EMA and hysteresis to assign gaze direction."""
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self.h_ema: float | None = None
        self.v_ema: float | None = None
        self.h_state = 0
        self.v_state = 0

    def classify(self, h: float, v: float, yaw: float, pitch: float, baseline: dict[str, float]) -> str:
        h -= baseline["h"]
        v -= baseline["v"]
        yaw = wrap_angle_deg(yaw - baseline["yaw"])
        pitch = wrap_angle_deg(pitch - baseline["pitch"])

        if abs(yaw) > self.cfg.head_yaw_limit or abs(pitch) > self.cfg.head_pitch_limit:
            return self.cfg.head_turned

        self.h_ema = h if self.h_ema is None else self.cfg.ema_alpha * h + (1.0 - self.cfg.ema_alpha) * self.h_ema
        self.v_ema = v if self.v_ema is None else self.cfg.ema_alpha * v + (1.0 - self.cfg.ema_alpha) * self.v_ema

        self.h_state = compute_hysteresis(self.h_ema, self.h_state, self.cfg.h_enter, self.cfg.h_exit)
        self.v_state = compute_hysteresis(self.v_ema, self.v_state, self.cfg.v_enter, self.cfg.v_exit)

        parts = []
        if self.h_state == 1:
            parts.append("Right")
        elif self.h_state == -1:
            parts.append("Left")
        if self.v_state == 1:
            parts.append("Down")
        elif self.v_state == -1:
            parts.append("Up")

        return "Center" if not parts else "Looking " + " ".join(parts)