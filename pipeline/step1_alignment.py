from __future__ import annotations
from utils.align_images import align_images
from pipeline_helper.context import PipelineContext
import cv2

def run(ctx: PipelineContext) -> bool:
    """STEP 1: Alignment & Homography Warp."""
    try:
        ctx.aligned_current_img = align_images(ctx.master_img, ctx.raw_current_img)
        print("[PASS] Homography Alignment Successful (Image perspective rectified).")
        return True
    except ValueError as e:
        print(f"[FAIL] Alignment failed ({e}). Halting pipeline.")
        ctx.errors.append(str(e))
        ctx.halt = True
        return False
