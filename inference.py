from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import os
import json
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from utils.model import load_yolo
from utils.features import extract_features
from utils.align_images import align_images
from utils.matcher import identify_room
from utils.vlm import analyze_room_with_vlm, load_prompt
from config.inference_config import *

RULES_DIR = "image_rules"

def get_room_rule(room_name: str) -> str | None:
    if not room_name or not os.path.exists(RULES_DIR):
        return None
    candidates = [
        f"{room_name}.txt",
        f"{room_name.lower()}.txt",
    ]
    for candidate in candidates:
        rule_path = os.path.join(RULES_DIR, candidate)
        if os.path.exists(rule_path):
            with open(rule_path, "r", encoding="utf-8") as f:
                return f.read().strip()
    for fname in os.listdir(RULES_DIR):
        if fname.lower() == f"{room_name.lower()}.txt":
            rule_path = os.path.join(RULES_DIR, fname)
            with open(rule_path, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None

def resolve_room_name(file_path, root_dir):
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    return parts[0] if len(parts) > 1 else identify_room(file_path)

import sys

def select_image_dialog() -> str | None:
    """Opens a native GUI file selection pop-up window."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        init_dir = os.path.abspath(CURRENT_DIR) if os.path.exists(CURRENT_DIR) else os.getcwd()
        file_path = filedialog.askopenfilename(
            title="Select Room Image for Visual Inspection",
            initialdir=init_dir,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp"), ("All Files", "*.*")],
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[WARN] Could not open GUI file selector: {e}")
        return None

def process_single_image(img_path: str, model=None):
    """Processes a single image through the complete alignment, CV delta, and VLM inspection pipeline."""
    if not os.path.exists(img_path):
        print(f"[ERROR] Specified image file '{img_path}' does not exist.")
        return

    if model is None:
        model = load_yolo()

    filename = os.path.basename(img_path)
    room_name = resolve_room_name(img_path, CURRENT_DIR)

    if not room_name:
        print(f"[SKIP] Could not identify room space for '{img_path}'.")
        return

    json_path = os.path.join(BASELINES_DIR, f"{room_name}_baseline.json")
    ref_img_path = os.path.join(BASELINES_DIR, f"{room_name}_ref.jpg")

    if not os.path.exists(json_path) or not os.path.exists(ref_img_path):
        print(f"[SKIP] Missing baseline files for room '{room_name}'.")
        return

    with open(json_path, "r") as f_json:
        master_data = json.load(f_json)

    master_img = cv2.imread(ref_img_path)
    raw_current_img = cv2.imread(img_path)

    if raw_current_img is None:
        print(f"[ERROR] Could not read image file '{img_path}'.")
        return

    print(f"\n ROOM DELTA REPORT: {room_name.upper()} ({filename})")

    # STEP 1: ALIGNMENT & HOMOGRAPHY WARP
    try:
        aligned_current_img = align_images(master_img, raw_current_img)
        print("[PASS] Homography Alignment Successful (Image perspective rectified).")
    except ValueError as e:
        print(f"[REJECT] {e}")
        return

    # Save temporary aligned image for YOLO feature extraction
    temp_aligned_path = "temp_aligned.jpg"
    cv2.imwrite(temp_aligned_path, aligned_current_img)

    # STEP 2: FEATURE EXTRACTION ON ALIGNED IMAGE
    current_data, _ = extract_features(temp_aligned_path, model)
    if os.path.exists(temp_aligned_path):
        os.remove(temp_aligned_path)

    # STEP 3: SSIM & LIGHTING DELTA
    master_gray = cv2.cvtColor(cv2.resize(master_img, (640, 480)), cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(cv2.resize(aligned_current_img, (640, 480)), cv2.COLOR_BGR2GRAY)
    
    ssim_score = ssim(master_gray, aligned_gray)
    brightness_diff = current_data["brightness"] - master_data["brightness"]

    print(f"[INFO] Alignment SSIM Score: {ssim_score:.2f}")
    print(f"[INFO] Lighting Delta: {brightness_diff:+.2f} intensity units")

    # STEP 4: DELTA CALCULATIONS (INVENTORY, DRIFT, CLUTTER)
    master_objs = master_data["objects"]
    current_objs = current_data["objects"].copy()
    matched_master = []
    drift_alerts = []

    for m_obj in master_objs:
        best_match_idx = None
        min_dist = float("inf")

        for idx, c_obj in enumerate(current_objs):
            if m_obj["label"] == c_obj["label"]:
                dist = math.hypot(
                    m_obj["centroid"][0] - c_obj["centroid"][0],
                    m_obj["centroid"][1] - c_obj["centroid"][1]
                )
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx

        if best_match_idx is not None:
            c_obj = current_objs.pop(best_match_idx)
            matched_master.append(m_obj)
            if min_dist > MAX_DRIFT_PIXELS:
                drift_alerts.append(f"{m_obj['label'].capitalize()} shifted {int(min_dist)}px")

    missing = [obj["label"] for obj in master_objs if obj not in matched_master]
    foreign = [obj["label"] for obj in current_objs]

    print("\n--- Reset Checklist ---")
    if not missing:
        print("[OK] All required room items present.")
    else:
        for item in set(missing):
            print(f"[MISSING] {missing.count(item)} x {item}")

    if not drift_alerts:
        print("[OK] Furniture positions verified.")
    else:
        for alert in drift_alerts:
            print(f"[DRIFT] {alert}")

    if not foreign:
        print("[OK] Room is clean (No clutter detected).")
    else:
        for item in set(foreign):
            print(f"[CLUTTER] Remove {foreign.count(item)} x {item}")

    # STEP 5: VLM MULTIMODAL VISUAL INSPECTION
    print("\n--- VLM Visual Inspection ---")
    prompt_path = os.path.join("prompts", "master_prompt.txt")
    if os.path.exists(prompt_path):
        base_prompt = load_prompt(prompt_path)
    else:
        base_prompt = "Compare the CURRENT IMAGE against the baseline state established by the MASTER IMAGE and MASTER JSON."

    room_rule = get_room_rule(room_name)
    if room_rule:
        print(f"[INFO] Applied Room Rule for '{room_name}': {room_rule}")
        combined_prompt = f"{base_prompt}\n\n### SPECIFIC ROOM RULES FOR {room_name.upper()}\n{room_rule}"
    else:
        combined_prompt = base_prompt

    success, encoded_img = cv2.imencode(".jpg", aligned_current_img)
    if success:
        try:
            vlm_result = analyze_room_with_vlm(
                processed_image=encoded_img.tobytes(),
                master_image=ref_img_path,
                master_json=master_data,
                master_prompt=combined_prompt,
            )
            print(f"[VLM STATUS] {vlm_result.status}")
            print(f"[VLM SUMMARY] {vlm_result.summary}")
            if vlm_result.issues:
                print("[VLM DETECTED DISCREPANCIES]")
                for issue in vlm_result.issues:
                    print(f" - [{issue.type}] ({issue.severity}) {issue.object}: {issue.description} (Confidence: {issue.confidence:.2f})")
            else:
                print("[VLM] No visual discrepancies detected.")
        except Exception as e:
            print(f"[VLM ERROR] Inspection failed: {e}")
    else:
        print("[VLM ERROR] Failed to encode aligned image for VLM processing.")

def process_uploaded_images(target_path: str | None = None):
    """Processes a single selected image or all images in CURRENT_DIR if requested."""
    if target_path:
        process_single_image(target_path)
        return

    # Check CLI arguments
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        process_single_image(args[0])
        return

    if "--all" in sys.argv:
        model = load_yolo()
        for root, _, files in os.walk(CURRENT_DIR):
            for f in files:
                if f.lower().endswith(VALID_EXTENSIONS):
                    process_single_image(os.path.join(root, f), model=model)
        return

    # Default: Open GUI Pop-up Dialog to select image
    print("[INFO] Opening pop-up window to select image...")
    selected_file = select_image_dialog()
    if selected_file:
        process_single_image(selected_file)
    else:
        print("[INFO] No image selected. Exiting.")

if __name__ == "__main__":
    process_uploaded_images()