"""
Model loaders. Each model is loaded ONCE per process and cached - re-loading weights
per request is the single biggest avoidable latency cost in a SaaS deployment.
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

_yolo_world_model: YOLOWorld | None = None
_yolo_world_classes: tuple | None = None
def load_yolo_world(class_list: list[str] | None = None) -> YOLOWorld:
    """
    YOLO-World turns the class list into CLIP text embeddings inside set_classes().
    That is slow, so it is done only when the list actually changes - NOT once per
    image like before.
    """
    global _yolo_world_model, _yolo_world_classes
    if _yolo_world_model is None:
        _yolo_world_model = YOLOWorld(YOLO_WORLD_WEIGHTS_PATH)
    if class_list is not None and tuple(class_list) != _yolo_world_classes:
        _yolo_world_model.set_classes(list(class_list))
        _yolo_world_classes = tuple(class_list)
    return _yolo_world_model

_sam_model = None


def load_sam():
    """Segment Anything (via Ultralytics), loaded once and cached like the YOLO models."""
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        from config.settings import SAM_WEIGHTS_PATH
        _sam_model = SAM(SAM_WEIGHTS_PATH)
    return _sam_model