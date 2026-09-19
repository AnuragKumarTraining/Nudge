import os
import json
import math
import cv2
from skimage.metrics import structural_similarity as ssim
from utils.model import load_yolo
from utils.features import extract_features
from utils.align_images import align_images
from utils.matcher import identify_room
from config.inference_config import *

def resolve_room_name(file_path, root_dir):
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    return parts[0] if len(parts) > 1 else identify_room(file_path)

def process_uploaded_images():
    model = load_yolo()
    
    for root, _, files in os.walk(CURRENT_DIR):
        for f in files:
            if not f.lower().endswith(VALID_EXTENSIONS):
                continue
                
            img_path = os.path.join(root, f)
            room_name = 'Media'

            if not room_name:
                print(f"[SKIP] Could not identify room space for '{img_path}'.")
                continue

            json_path = os.path.join(BASELINES_DIR, f"{room_name}_baseline.json")
            ref_img_path = os.path.join(BASELINES_DIR, f"{room_name}_ref.jpg")

            if not os.path.exists(json_path) or not os.path.exists(ref_img_path):
                print(f"[SKIP] Missing baseline files for room '{room_name}'.")
                continue

            with open(json_path, "r") as f_json:
                master_data = json.load(f_json)

            master_img = cv2.imread(ref_img_path)
            raw_current_img = cv2.imread(img_path)

            print(f"\n ROOM DELTA REPORT: {room_name.upper()} ({f})")

            # STEP 1: ALIGNMENT & HOMOGRAPHY WARP

            try:
                aligned_current_img = align_images(master_img, raw_current_img)
                print("[PASS] Homography Alignment Successful (Image perspective rectified).")
            except ValueError as e:
                print(f"[REJECT] {e}")
                continue  # Stopped processing if alignment is beyond match limit

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
process_uploaded_images()