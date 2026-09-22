from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import cv2

from utils.model import load_yolo
from utils.matcher import identify_room
from config.inference_config import CURRENT_DIR, BASELINES_DIR, VALID_EXTENSIONS
from pipeline_helper.context import PipelineContext
from pipeline_helper.selector import select_image_dialog
from pipeline.runner import PipelineRunner

def resolve_room_name(file_path: str, root_dir: str) -> str | None:
    rel_path = os.path.relpath(file_path, root_dir)
    parts = [p for p in rel_path.split(os.sep) if p]

    # 1) Prefer explicit folder names first
    if len(parts) > 1:
        room_name = parts[0].strip()
        if room_name:
            return room_name

    # 2) If it is a direct file like Bathroom.jpg, use the filename before feature matching
    file_stem = os.path.splitext(os.path.basename(file_path))[0]
    normalized_file = "".join(ch.lower() for ch in file_stem if ch.isalnum())

    if os.path.isdir(BASELINES_DIR):
        for name in os.listdir(BASELINES_DIR):
            if not name.lower().endswith("_baseline.json"):
                continue

            room_name = name[:-len("_baseline.json")]
            normalized_room = "".join(ch.lower() for ch in room_name if ch.isalnum())

            if normalized_file == normalized_room:
                return room_name

    # 3) Fallback to the old visual matcher only if no filename match is found
    return identify_room(file_path)

def process_single_image(img_path: str, model=None, runner=None) -> PipelineContext | None:
    if not os.path.exists(img_path):
        print(f"[ERROR] Specified image file '{img_path}' does not exist.")
        return None

    if model is None:
        model = load_yolo()

    if runner is None:
        runner = PipelineRunner()

    filename = os.path.basename(img_path)
    room_name = resolve_room_name(img_path, CURRENT_DIR)

    if not room_name:
        print(f"[SKIP] Could not identify room space for '{img_path}'.")
        return None

    json_path = os.path.join(BASELINES_DIR, f"{room_name}_baseline.json")
    ref_img_path = os.path.join(BASELINES_DIR, f"{room_name}_ref.jpg")

    if not os.path.exists(json_path) or not os.path.exists(ref_img_path):
        print(f"[SKIP] Missing baseline files for room '{room_name}'.")
        return None

    with open(json_path, "r", encoding="utf-8") as f_json:
        master_data = json.load(f_json)

    master_img = cv2.imread(ref_img_path)
    raw_current_img = cv2.imread(img_path)

    if raw_current_img is None:
        print(f"[ERROR] Could not read image file '{img_path}'.")
        return None

    print(f"\n ROOM DELTA REPORT: {room_name.upper()} ({filename})")

    ctx = PipelineContext(
        img_path=img_path,
        filename=filename,
        room_name=room_name,
        master_img=master_img,
        raw_current_img=raw_current_img,
        master_data=master_data,
        model=model,
    )

    return runner.run(ctx)

def process_uploaded_images(target_path: str | None = None):
    """Entry point to process a target image, CLI specified image, GUI selected image, or all images."""
    runner = PipelineRunner()

    if target_path:
        process_single_image(target_path, runner=runner)
        return

    # Check CLI arguments
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        process_single_image(args[0], runner=runner)
        return

    if "--all" in sys.argv:
        model = load_yolo()
        for root, _, files in os.walk(CURRENT_DIR):
            for f in files:
                if f.lower().endswith(VALID_EXTENSIONS):
                    process_single_image(os.path.join(root, f), model=model, runner=runner)
        return

    # Default: Open GUI Pop-up Dialog to select image
    print("[INFO] Opening pop-up window to select image...")
    selected_file = select_image_dialog()
    if selected_file:
        process_single_image(selected_file, runner=runner)
    else:
        print("[INFO] No image selected. Exiting.")

if __name__ == "__main__":
    process_uploaded_images()