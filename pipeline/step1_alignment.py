from __future__ import annotations
from utils.align_images import align_images
from pipeline_helper.context import PipelineContext
import cv2

def run(ctx: PipelineContext) -> bool:
    """STEP 1: Alignment & Homography Warp (with Graceful Fallback)."""
    try:
        ctx.aligned_current_img = align_images(ctx.master_img, ctx.raw_current_img)
        print("[PASS] Homography Alignment Successful (Image perspective rectified).")
        return True
    except ValueError as e:
        print(f"[WARN] Alignment failed ({e}).")
        print("[INFO] Falling back to direct scale alignment so bulb & VLM inspection can proceed.")
        ctx.errors.append(str(e))
        
        # Fallback: Resize current image to reference dimensions
        h, w = ctx.master_img.shape[:2]
        ctx.aligned_current_img = cv2.resize(ctx.raw_current_img, (w, h))
        
        # Keep pipeline running
        ctx.halt = False
        return True
