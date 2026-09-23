from __future__ import annotations
import os
import cv2
from utils.vlm import analyze_room_with_vlm, load_prompt
from pipeline_helper.rules import get_room_rule
from pipeline_helper.context import PipelineContext
from config.inference_config import BASELINES_DIR

def run(ctx: PipelineContext) -> bool:
    """STEP 5: VLM Multimodal Visual Inspection."""
    print("\n--- VLM Visual Inspection ---")
    prompt_path = os.path.join("prompts", "master_prompt.txt")
    if os.path.exists(prompt_path):
        base_prompt = load_prompt(prompt_path)
    else:
        base_prompt = "Compare the CURRENT IMAGE against the baseline state established by the MASTER IMAGE and MASTER JSON."

    ctx.room_rule = get_room_rule(ctx.room_name)
    if ctx.room_rule:
        print(f"[INFO] Applied Room Rule for '{ctx.room_name}': {ctx.room_rule}")
        combined_prompt = f"{base_prompt}\n\n### SPECIFIC ROOM RULES FOR {ctx.room_name.upper()}\n{ctx.room_rule}"
    else:
        combined_prompt = base_prompt

    if ctx.aligned_current_img is None:
        print("[VLM ERROR] No aligned image available for VLM inspection.")
        return False

    ref_img_path = os.path.join(BASELINES_DIR, f"{ctx.room_name}_ref.jpg")

    success, encoded_img = cv2.imencode(".jpg", ctx.aligned_current_img)
    if success:
        try:
            ctx.vlm_result = analyze_room_with_vlm(
                processed_image=encoded_img.tobytes(),
                master_image=ref_img_path,
                master_json=ctx.master_data,
                master_prompt=combined_prompt,
            )
            print(f"[VLM STATUS] {ctx.vlm_result.status}")
            print(f"[VLM SUMMARY] {ctx.vlm_result.summary}")
            if ctx.vlm_result.issues:
                print("[VLM DETECTED DISCREPANCIES]")
                for issue in ctx.vlm_result.issues:
                    print(f" - [{issue.type}] ({issue.severity}) {issue.object}: {issue.description} (Confidence: {issue.confidence:.2f})")
            else:
                print("[VLM] No visual discrepancies detected.")
        except Exception as e:
            print(f"[VLM ERROR] Inspection failed: {e}")
    else:
        print("[VLM ERROR] Failed to encode aligned image for VLM processing.")

    return True
