"""
This is the file that was completely empty in the original repo (utils/model.py)
despite main.py and master.py both importing load_yolo() from it — that import
would fail immediately. This module is the fix.
"""
from ultralytics import YOLO, YOLOWorld
from config.settings import YOLO_WEIGHTS_PATH, YOLO_WORLD_WEIGHTS_PATH

_model_cache: YOLO | None = None


def load_yolo() -> YOLO:
    """
    Loads the YOLO model once and caches it in-process. Re-loading a YOLO
    model per-request is a common source of needless latency — don't call
    YOLO(...) anywhere else in the codebase, always go through this.
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(YOLO_WEIGHTS_PATH)
    return _model_cache

_yo_world_model: YOLOWorld | None = None

def load_yolo_world():
    global _yo_world_model

    if _yo_world_model is None:

        _yo_world_model = YOLOWorld(
            YOLO_WORLD_WEIGHTS_PATH
        )

    return _yo_world_model