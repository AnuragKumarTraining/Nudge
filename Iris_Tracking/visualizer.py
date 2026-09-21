import cv2
import numpy as np
from config import AppConfig
from gaze import Calibrator
from tracker import StateTracker

class Visualizer:
    """Renders eye anchors and HUD analytics onto OpenCV frames."""
    def __init__(self, config: AppConfig):
        self.cfg = config

    def draw_landmarks(self, frame: np.ndarray, landmarks, w: int, h: int) -> None:
        for idx, color, radius in (
            (self.cfg.r_iris_center, (0, 0, 255), 3),
            (self.cfg.l_iris_center, (0, 0, 255), 3),
            (self.cfg.nose_tip, (255, 0, 0), 2),
        ):
            if idx < len(landmarks):
                cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)

    def draw_hud(
        self,
        frame: np.ndarray,
        tracker: StateTracker,
        calibrator: Calibrator,
        dbg_signals: dict[str, float],
        frame_height: int,
    ) -> None:
        shown = tracker.committed or "..."
        cv2.putText(frame, f"Gaze: {shown}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if calibrator.active:
            status_text = f"Calibrating: ({len(calibrator.samples)}/{self.cfg.calibration_frames})"
            cv2.putText(frame, status_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

        y = 100
        for direction in sorted(tracker.visits):
            text = f"{direction}: {tracker.duration[direction]:.1f}s ({tracker.visits[direction]} times)"
            cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            y += 25

        telemetry_str = (
            f"h={dbg_signals['h']:+.2f} v={dbg_signals['v']:+.2f} "
            f"yaw={dbg_signals['yaw']:+.0f} pitch={dbg_signals['pitch']:+.0f}"
        )
        cv2.putText(frame, telemetry_str, (30, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "[c] recalibrate   [q] quit", (30, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)