import os
from dataclasses import dataclass, field

@dataclass(frozen=True)
class AppConfig:
    # Source: 0 for webcam, or video path string
    video_source: int | str = 0
    # video_source: int | str = r"C:\Users\soham.dalui\OneDrive - JK Technosoft Ltd\Desktop\Cafe\Nudge\Iris_Tracking\WIN_20260921_11_11_39_Pro.mp4"

    script_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    model_path: str = field(init=False)
    json_output_path: str = field(init=False)

    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    # MediaPipe Mesh indices
    r_iris_center: int = 468
    l_iris_center: int = 473
    nose_tip: int = 1

    # Blendshape classification thresholds
    h_enter: float = 0.28
    h_exit: float = 0.16
    v_enter: float = 0.24
    v_exit: float = 0.14
    blink_threshold: float = 0.55

    # Head pose bounds (degrees)
    head_yaw_limit: float = 25.0
    head_pitch_limit: float = 20.0

    # Temporal parameters
    ema_alpha: float = 0.50
    min_dwell_sec: float = 0.15
    blink_hold_sec: float = 0.40

    # Calibration parameters
    calibration_frames: int = 45
    swap_left_right: bool = False

    # Sentinel states
    no_face: str = "No Face"
    blink: str = "Blink/Transition"
    head_turned: str = "Head Turned"
    calibrating: str = "Calibrating"

    def __post_init__(self):
        object.__setattr__(self, "model_path", os.path.join(self.script_dir, "face_landmarker.task"))
        object.__setattr__(self, "json_output_path", os.path.join(self.script_dir, "gaze_report.json"))

    @property
    def is_live(self) -> bool:
        return isinstance(self.video_source, int)

    @property
    def ignored_states(self) -> set[str]:
        return {self.no_face, self.blink, self.calibrating}