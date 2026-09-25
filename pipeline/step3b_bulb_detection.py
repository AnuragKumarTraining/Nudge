from __future__ import annotations
from pipeline_helper.context import PipelineContext
from Bulbs.bulb_detector import BulbDetector
import os
import cv2

# Cache the detector instance so weights load only once
_BULB_DETECTOR: BulbDetector | None = None

def get_bulb_detector() -> BulbDetector:
    global _BULB_DETECTOR
    if _BULB_DETECTOR is None:
        _BULB_DETECTOR = BulbDetector()
    return _BULB_DETECTOR

def run(ctx: PipelineContext) -> bool:
    """STEP 3b: Bulb Detection & Lighting Discrepancy Verification."""
    if ctx.aligned_current_img is None:
        return False

    detector = getattr(ctx, "bulb_engine", None) or get_bulb_detector()

    # Run detection on rectified current frame
    bulb_result = detector.detect_bulbs(ctx.aligned_current_img)
    
    current_count = bulb_result["count"]
    baseline_count = ctx.master_data.get("bulb_count", 0)

    # Store findings in context for downstream steps / checklists
    ctx.current_bulb_count = current_count
    ctx.baseline_bulb_count = baseline_count
    ctx.bulb_detections = bulb_result["detections"]

    print(f"[INFO] Active Bulbs Detected: {current_count} (Baseline: {baseline_count})")

    # =========================================================================
    # SAVE IMAGE BEFORE VLM INSPECTION (Only if at least one bulb is detected)
    # =========================================================================
    annotated_frame = bulb_result.get("annotated_frame")
    if current_count > 0 and annotated_frame is not None:
        out_dir = "output_images"
        os.makedirs(out_dir, exist_ok=True)

        room_tag = getattr(ctx, "room_name", "room")
        save_path = os.path.join(out_dir, f"{room_tag}_bulbs_annotated.jpg")

        cv2.imwrite(save_path, annotated_frame)
        print(f"[INFO] Bulb detection visual saved to: {save_path} ({current_count} bulb(s) marked)")
    # =========================================================================

    return True