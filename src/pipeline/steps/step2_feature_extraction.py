from __future__ import annotations
import os
import cv2
from src.utils.features import extract_features
from src.pipeline.pipeline_helper.context import PipelineContext

def run(ctx: PipelineContext) -> bool:
    """STEP 2: Feature Extraction on Aligned Image."""
    if ctx.aligned_current_img is None:
        ctx.halt = True
        return False

    temp_aligned_path = "temp_aligned.jpg"
    cv2.imwrite(temp_aligned_path, ctx.aligned_current_img)

    try:
        ctx.current_data, _ = extract_features(temp_aligned_path, ctx.model)
    finally:
        if os.path.exists(temp_aligned_path):
            os.remove(temp_aligned_path)

    return True
